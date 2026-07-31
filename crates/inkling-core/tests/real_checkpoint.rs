//! Assertions against a real Inkling-Small checkpoint, which is far too large
//! to commit. Set `INKLINGRS_CHECKPOINT` to a checkpoint directory to run them;
//! unset, each test reports a skip and passes.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::time::Instant;

use inkling_core::attention::{Attention, AttentionConfig, AttentionWeights};
use inkling_core::embed::Embed;
use inkling_core::fixture::{
    self, ACTIVATIONS, CAPTURED_LAYERS, TokenizerFixture, deviation, indices,
};
use inkling_core::generate::{Generator, greedy};
use inkling_core::head::LmHead;
use inkling_core::layer::{
    DecoderCache, DecoderLayer, DecoderWeights, Experts, LayerMlp, NoExperts,
};
use inkling_core::model::{ModelCache, ModelWeights};
use inkling_core::moe::{GateWeights, MoeConfig, SparseMoe};
use inkling_core::ops::{DenseMlp, top_k};
use inkling_core::quant::{Scratch, dequantize};
use inkling_core::tokenizer::{Tokenizer, TokenizerError};
use inkling_core::weights::{
    CheckpointWeights, Packed, PackedExperts, expert_scratch_floats, layer_scratch_floats,
};
use inkling_core::{Checkpoint, Dtype, TensorView, TextConfig};

const CHECKPOINT_VAR: &str = "INKLINGRS_CHECKPOINT";

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os(CHECKPOINT_VAR).map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: {CHECKPOINT_VAR} is unset");
    }
    dir
}

/// A named tensor, still packed or still bfloat16 as the checkpoint stores it.
fn checkpoint_tensor<'a>(ckpt: &'a Checkpoint, name: &str) -> TensorView<'a> {
    ckpt.tensor(name)
        .unwrap_or_else(|err| panic!("checkpoint holds {name}: {err}"))
}

#[test]
fn mxfp4_checkpoint_spans_thirty_shards() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");

    assert_eq!(ckpt.num_shards(), 30);
    assert_eq!(ckpt.tensor_names().count(), 1508);
}

#[test]
fn routed_expert_weights_are_packed_eight_nibbles_per_u32() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");

    // 256 experts x 2048 intermediate x 4096 hidden, packed 8 per u32.
    let moe = ckpt
        .tensor("language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight")
        .expect("layer 2 is MoE");
    assert_eq!(moe.dtype(), Dtype::U32);
    assert_eq!(moe.shape(), [256, 2048, 512]);
    assert_eq!(moe.data().len(), 256 * 2048 * 512 * 4);

    // Layers 0 and 1 are dense, at the 16384-wide `dense_intermediate_size`.
    let dense = ckpt
        .tensor("language_model.model.layers.0.mlp.gate_proj.weight")
        .expect("layer 0 is dense");
    assert_eq!(dense.dtype(), Dtype::U32);
    assert_eq!(dense.shape(), [16384, 512]);
}

#[test]
fn opening_does_not_fault_in_the_weights() {
    let Some(dir) = checkpoint_dir() else { return };

    let before = fixture::resident_bytes();
    let started = Instant::now();
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let elapsed = started.elapsed();
    let after = fixture::resident_bytes();
    assert_eq!(ckpt.num_shards(), 30);

    let grew = after.saturating_sub(before);
    eprintln!("open took {elapsed:?}, RSS {before} -> {after} (+{grew} bytes)");

    // The mapped bytes are ~131 GiB. Reading 30 headers should cost single-digit
    // MiB; a gibibyte is a loose bound that still catches an eager read.
    assert!(
        grew < (1 << 30),
        "open grew RSS by {grew} bytes ({before} -> {after})"
    );
}

/// What the synthetic attention fixture cannot settle: that the tensor names,
/// the checkpoint's layouts and the config fields a layer reads are the ones
/// this port assumes. Its weights are `[4096, 4096]` MXFP4 — 67 MB each once
/// decoded — so they are far too large to commit, and only a real checkpoint
/// carries them.
///
/// The reference multiplies its 4-bit weights without decoding them, in
/// bfloat16, through `mx.quantized_matmul`; this decodes them and multiplies in
/// float32. Every intermediate of the recorded pass was rounded to bfloat16 and
/// none of this port's are, over a chain of five projections, two convolutions,
/// two norms and a softmax — so the gap is a dtype's rather than an arithmetic
/// one, and 6e-3 is the same bound, for the same reason, as the recorded
/// attention step and the trained masks. Worst observed when this landed:
/// 2.6e-3 on layer 0, against a mask flattened to no learned bias at 1.4e-1.
/// Layer 5, the global one, sits at 2.4e-3.
///
/// What this settles is the wiring; the synthetic cases settle the arithmetic.
/// One of the three captured layers is global, so which of the two sets of head
/// fields a layer reads is settled here too. What eight tokens cannot settle is
/// what a global layer does differently: nothing is far enough back to be capped
/// by a window or to fall outside the 1024-token band, so a port that handed a
/// global layer a window would agree here. Only a sequence past the band settles
/// that.
const ATTENTION_TOLERANCE: f32 = 6e-3;

fn text_config(dir: &Path) -> TextConfig {
    fixture::config(dir).text_config
}

/// One layer's attention tensors, decoded out of the checkpoint.
///
/// The projections are MXFP4 — packed codes plus a scale byte per group of 32 —
/// and everything else is bfloat16, so the two are read different ways.
struct Weights {
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    r_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
    rel_proj: Vec<f32>,
}

/// Where one decoder layer's tensors live.
fn layer_module(layer: usize) -> String {
    format!("language_model.model.layers.{layer}")
}

impl Weights {
    fn load(ckpt: &Checkpoint, layer: usize) -> Self {
        let of = |name: &str| {
            checkpoint_tensor(ckpt, &format!("{}.self_attn.{name}", layer_module(layer)))
        };
        let quantized = |name: &str| {
            dequantize(
                &of(&format!("{name}.weight")),
                &of(&format!("{name}.scales")),
            )
            .unwrap_or_else(|err| panic!("layer {layer} {name} decodes: {err}"))
            .values
        };
        let widened = |name: &str| of(name).to_f32().expect("a bfloat16 tensor");

        Self {
            q_proj: quantized("q_proj"),
            k_proj: quantized("k_proj"),
            v_proj: quantized("v_proj"),
            r_proj: quantized("r_proj"),
            o_proj: quantized("o_proj"),
            q_norm: widened("q_norm.weight"),
            k_norm: widened("k_norm.weight"),
            k_sconv: widened("k_sconv.conv.weight"),
            v_sconv: widened("v_sconv.conv.weight"),
            rel_proj: widened("rel_proj"),
        }
    }

    fn view(&self) -> AttentionWeights<'_> {
        AttentionWeights {
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            r_proj: &self.r_proj,
            o_proj: &self.o_proj,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            k_sconv: &self.k_sconv,
            v_sconv: &self.v_sconv,
            rel_proj: &self.rel_proj,
        }
    }
}

#[test]
fn attention_reproduces_the_reference_layers_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);

    let mut worst = 0.0f32;
    for layer in CAPTURED_LAYERS {
        let recorded =
            |name: &str| fixture::f32s(&fixture::layer_tensor(&activations, layer, name));
        let (x, want) = (recorded("input_layernorm_out"), recorded("o_proj_out"));
        let weights = Weights::load(&ckpt, layer);
        let layer_config = AttentionConfig::for_layer(&config, layer);

        let attention = Attention::new(layer_config, weights.view());
        let against_reference = deviation(&attention.forward(&mut attention.cache(), &x), &want);
        assert!(
            against_reference <= ATTENTION_TOLERANCE,
            "layer {layer}: deviation {against_reference:e}"
        );
        worst = worst.max(against_reference);

        // The learned relative-position bias is most of what makes this an
        // Inkling layer rather than an ordinary causal one, and at eight tokens
        // it is the only part of the mask that is not plain causality. Without
        // it the layer still attends, over a flat band.
        let flat = vec![0.0; weights.rel_proj.len()];
        let mut unbiased = weights.view();
        unbiased.rel_proj = &flat;
        let unbiased = Attention::new(layer_config, unbiased);
        let flattened = deviation(&unbiased.forward(&mut unbiased.cache(), &x), &want);
        assert!(
            flattened > ATTENTION_TOLERANCE,
            "layer {layer}: a flat band deviates by only {flattened:e}"
        );
    }
    assert!(
        worst > 0.0,
        "a run that matched exactly would mean the reference's bfloat16 vanished"
    );
}

/// What the synthetic MoE fixture cannot settle. The committed gate makes the
/// whole routing computation hermetic, but the experts it routes to are
/// `[256, 2048, 4096]` and `[2, 2048, 4096]` of MXFP4 — 25 GB once decoded — so
/// only a real checkpoint carries them, and with them the tensor names, the
/// `[experts, out, in]` layout and the per-expert group boundaries.
///
/// The reference multiplies its 4-bit weights without decoding them, in
/// bfloat16, through `mx.gather_qmm`; this decodes them and multiplies in
/// float32, then sums a token's six experts in expert order rather than in
/// selection order. The gap is a dtype's, so 6e-3 — three bfloat16 quanta at
/// 2^-9 — is the same bound, for the same reason, as the recorded attention
/// step and the trained masks.
///
/// It holds less of that in reserve than the attention layer does, and knowably
/// so: the reference rounds *twice* here, once on each expert's output and
/// again on the routing weight, which `InklingSparseMoE` casts to the input's
/// dtype before multiplying. Worst observed when this landed: 4.2e-3 on the
/// routed half, two quanta, against a shared pair exchanged at 6.6e-2 — an
/// order of magnitude above the bound.
const MOE_TOLERANCE: f32 = 6e-3;

/// One layer's `SwitchGLU`, left packed.
fn packed_experts<'a>(ckpt: &'a Checkpoint, layer: usize, module: &str) -> PackedExperts<'a> {
    PackedExperts::open(ckpt, &format!("{}.mlp.{module}", layer_module(layer)))
        .unwrap_or_else(|err| panic!("layer {layer} holds {module}: {err}"))
}

/// `[rows, dim]` through one expert of a packed bank, decoded into a buffer of
/// its own. The stack decodes into one it reuses; here each call is on its own,
/// which a test can afford.
fn through_expert(bank: &PackedExperts<'_>, expert: usize, dim: usize, rows: &[f32]) -> Vec<f32> {
    let mut buffer = vec![0.0; bank.expert_floats()];
    bank.forward_into(expert, dim, rows, &mut Scratch::new(&mut buffer))
}

#[test]
fn the_moe_layer_reproduces_the_reference_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);

    let layer = CAPTURED_LAYERS
        .into_iter()
        .find(|layer| !config.layer_is_dense(*layer))
        .expect("the capture covers a MoE layer");
    let recorded = |name: &str| fixture::f32s(&fixture::layer_tensor(&activations, layer, name));

    let gate = |name: &str| {
        checkpoint_tensor(&ckpt, &format!("{}.mlp.{name}", layer_module(layer)))
            .to_f32()
            .expect("the gate is not packed")
    };
    let (gate_weight, correction_bias) = (gate("gate_weight"), gate("e_score_correction_bias"));
    let global_scale = gate("global_scale")[0];

    let moe = SparseMoe::new(
        MoeConfig::for_layer(&config, layer).expect("a MoE layer has a router"),
        GateWeights {
            gate_weight: &gate_weight,
            correction_bias: &correction_bias,
            global_scale,
        },
    );
    let (x, hidden) = (recorded("post_attention_ln_out"), moe.hidden());

    // Which experts run is the one thing here that has to be exact rather than
    // close: a single different expert changes the answer by far more than the
    // bound below, and would fail as an arithmetic disagreement rather than as
    // the wiring mistake it is. The order within a token is more than the
    // reference promises — see `SparseMoe::route` — and a capture regenerated on
    // a backend that partitions differently would fail here for that reason
    // rather than for a routing one.
    let routing = moe.route(&moe.gate(&x));
    assert_eq!(
        routing.experts(),
        indices(&fixture::layer_tensor(&activations, layer, "topk_idx")),
        "layer {layer}: selection"
    );

    let routed = packed_experts(&ckpt, layer, "switch_mlp");
    let shared = packed_experts(&ckpt, layer, "shared_experts");
    let got = moe.forward(
        &x,
        |expert, rows| through_expert(&routed, expert, hidden, rows),
        |expert, rows| through_expert(&shared, expert, hidden, rows),
    );

    let mut worst = 0.0f32;
    for (what, got, want) in [
        ("routed_out", &got.routed, recorded("routed_out")),
        ("shared_out", &got.shared, recorded("shared_out")),
        ("mlp_out", &got.total(), recorded("mlp_out")),
    ] {
        let deviation = deviation(got, &want);
        assert!(
            deviation <= MOE_TOLERANCE,
            "layer {layer}: {what} deviation {deviation:e}"
        );
        worst = worst.max(deviation);
    }
    assert!(
        worst > 0.0,
        "a run that matched exactly would mean the reference's bfloat16 vanished"
    );

    // The two shared experts are told apart by their index alone, and a port
    // that paired `shared_gammas[0]` with the second of them would still add
    // two always-on experts to every token. Cheap to state here because the
    // shared bank is two experts rather than the routed bank's 256.
    let exchanged = moe.forward(
        &x,
        |_, rows| vec![0.0; rows.len()],
        |expert, rows| through_expert(&shared, shared.experts() - 1 - expert, hidden, rows),
    );
    let deviation = deviation(&exchanged.shared, &recorded("shared_out"));
    assert!(
        deviation > MOE_TOLERANCE,
        "layer {layer}: exchanging the shared experts deviates by only {deviation:e}"
    );

    // The correction bias is a weight like any other, and a port that never
    // loaded it routes every token somewhere else. Stated on the routing rather
    // than on the output, which would mean decoding a second set of experts.
    let unbiased = vec![0.0; correction_bias.len()];
    let flat = SparseMoe::new(
        moe.config(),
        GateWeights {
            gate_weight: &gate_weight,
            correction_bias: &unbiased,
            global_scale,
        },
    );
    assert_ne!(
        flat.route(&flat.gate(&x)).experts(),
        routing.experts(),
        "layer {layer}: dropping the correction bias selects the same experts"
    );
}

/// What no committed fixture can settle about the assembled layer: that the
/// tensor names outside attention and the MLP are the ones this port assumes,
/// and that a whole trained layer — quantised projections, banded mask, trained
/// convolutions, router and 256-expert bank — reproduces the reference from its
/// input to its output.
///
/// Error accumulates across the whole layer, so this is legitimately looser than
/// the 6e-3 each of its halves is held to, and the reason is `mlp_sconv` rather
/// than the length of the chain. The trained convolutions amplify: on layer 0 the
/// MLP's output peaks at 10.75 and what the convolution makes of it peaks at 192,
/// so the bfloat16 quantum the reference rounded `mlp_out` to arrives at the
/// residual multiplied by about eighteen. That is why the short convolution's own
/// trained pairs are held to 2e-2 and not to 6e-3, and the layer inherits that
/// regime: its output is `h + mlp_sconv(...)`, of which the convolution is about
/// three quarters, so 2e-2 scaled by that share is 1.5e-2.
///
/// Written out, the observed error is exactly that sum. On layer 0, `h` deviates
/// by 4.7e-3 of its own peak and `mlp_sconv_out` by 8.3e-3 of its own, which at
/// their shares of the output's peak — 65 and 192 against 254 — come to 1.2e-3
/// and 6.2e-3, against 7.0e-3 measured end to end.
///
/// Worst observed when this landed: 7.0e-3 on layer 0 and 6.6e-3 on layer 2, a
/// factor of two in hand. Against the weakest mutation this bound has to catch,
/// layer 0's two residual-path convolutions exchanged, at 4.9e-1 — a factor of
/// thirty above it.
///
/// Layer 5 comes in an order of magnitude lower, at 5.5e-4, and that is the same
/// account read the other way rather than a better port. Activations grow with
/// depth: its residual peaks at 13248 where layer 0's peaks at 65, while its
/// `mlp_sconv_out` peaks at 1664 — 11% of the output's 14912 against layer 0's
/// 76%. The convolution's amplified error is the same error; it is a tenth of
/// the tensor it is measured against.
const LAYER_TOLERANCE: f32 = 1.5e-2;

/// One decoder layer's tensors outside its attention and its MLP: a weight for
/// each of the two RMSNorms and a kernel for each of the two short convolutions
/// on the residual path, all bfloat16.
struct LayerWeights {
    attention: Weights,
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
}

impl LayerWeights {
    fn load(ckpt: &Checkpoint, layer: usize) -> Self {
        let widened = |name: &str| {
            checkpoint_tensor(ckpt, &format!("{}.{name}", layer_module(layer)))
                .to_f32()
                .expect("a bfloat16 tensor")
        };
        Self {
            attention: Weights::load(ckpt, layer),
            input_layernorm: widened("input_layernorm.weight"),
            post_attention_layernorm: widened("post_attention_layernorm.weight"),
            attn_sconv: widened("attn_sconv.conv.weight"),
            mlp_sconv: widened("mlp_sconv.conv.weight"),
        }
    }

    fn view(&self) -> DecoderWeights<'_> {
        DecoderWeights {
            attention: self.attention.view(),
            input_layernorm: &self.input_layernorm,
            post_attention_layernorm: &self.post_attention_layernorm,
            attn_sconv: &self.attn_sconv,
            mlp_sconv: &self.mlp_sconv,
        }
    }
}

/// Whichever MLP a layer index called for, out of the checkpoint.
///
/// A dense layer's three `[16384, 4096]` projections are decoded once and held —
/// 800 MB, which a test can afford. A MoE layer's 256 experts are 25 GB and are
/// decoded per expert per call instead, which is what [`Experts`] exists for.
enum Mlp<'a> {
    Dense(Dense),
    Sparse(Box<Sparse<'a>>),
}

/// `InklingDenseMLP`'s three decoded projections and its learned output scale.
struct Dense {
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    global_scale: f32,
}

/// `InklingSparseMoE`'s gate, and the two banks it routes to left packed.
struct Sparse<'a> {
    config: MoeConfig,
    hidden: usize,
    gate_weight: Vec<f32>,
    correction_bias: Vec<f32>,
    global_scale: f32,
    routed: PackedExperts<'a>,
    shared: PackedExperts<'a>,
}

impl<'a> Mlp<'a> {
    fn load(ckpt: &'a Checkpoint, config: &TextConfig, layer: usize) -> Self {
        let of =
            |name: &str| checkpoint_tensor(ckpt, &format!("{}.mlp.{name}", layer_module(layer)));
        let widened = |name: &str| of(name).to_f32().expect("an unpacked tensor");
        let global_scale = widened("global_scale")[0];

        let Some(moe) = MoeConfig::for_layer(config, layer) else {
            let quantized = |name: &str| {
                dequantize(
                    &of(&format!("{name}.weight")),
                    &of(&format!("{name}.scales")),
                )
                .unwrap_or_else(|err| panic!("layer {layer} {name} decodes: {err}"))
                .values
            };
            return Self::Dense(Dense {
                gate_proj: quantized("gate_proj"),
                up_proj: quantized("up_proj"),
                down_proj: quantized("down_proj"),
                global_scale,
            });
        };

        Self::Sparse(Box::new(Sparse {
            hidden: config.hidden_size,
            config: moe,
            gate_weight: widened("gate_weight"),
            correction_bias: widened("e_score_correction_bias"),
            global_scale,
            routed: packed_experts(ckpt, layer, "switch_mlp"),
            shared: packed_experts(ckpt, layer, "shared_experts"),
        }))
    }

    fn view(&self, hidden: usize) -> LayerMlp<'_> {
        match self {
            Self::Dense(dense) => LayerMlp::Dense(DenseMlp::new(
                hidden,
                &dense.gate_proj,
                &dense.up_proj,
                &dense.down_proj,
                dense.global_scale,
            )),
            Self::Sparse(sparse) => LayerMlp::Sparse(SparseMoe::new(
                sparse.config,
                GateWeights {
                    gate_weight: &sparse.gate_weight,
                    correction_bias: &sparse.correction_bias,
                    global_scale: sparse.global_scale,
                },
            )),
        }
    }
}

impl Experts for Mlp<'_> {
    fn routed(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense(_) => NoExperts.routed(expert, rows),
            Self::Sparse(moe) => through_expert(&moe.routed, expert, moe.hidden, rows),
        }
    }

    fn shared(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense(_) => NoExperts.shared(expert, rows),
            Self::Sparse(moe) => through_expert(&moe.shared, expert, moe.hidden, rows),
        }
    }
}

#[test]
fn the_decoder_layer_reproduces_the_reference_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);

    let mut worst = 0.0f32;
    for layer in CAPTURED_LAYERS {
        let recorded =
            |name: &str| fixture::f32s(&fixture::layer_tensor(&activations, layer, name));
        let (x, want) = (recorded("input"), recorded("out"));

        let weights = LayerWeights::load(&ckpt, layer);
        let mlp = Mlp::load(&ckpt, &config, layer);
        let attention = AttentionConfig::for_layer(&config, layer);
        let decoder = DecoderLayer::new(attention, weights.view(), mlp.view(config.hidden_size));

        let against_reference = deviation(&decoder.forward(&mut decoder.cache(), &x, &mlp), &want);
        assert!(
            against_reference <= LAYER_TOLERANCE,
            "layer {layer}: deviation {against_reference:e}"
        );
        worst = worst.max(against_reference);

        // The two convolutions on the residual path are the same width, so a
        // port that exchanged their kernels runs. Stated on the dense layer
        // alone: its projections are already decoded and held, where a second
        // MoE pass would decode a fresh 25 GB bank of experts to say the same
        // thing about the same two tensors.
        if !config.layer_is_dense(layer) {
            continue;
        }
        let mut exchanged = weights.view();
        std::mem::swap(&mut exchanged.attn_sconv, &mut exchanged.mlp_sconv);
        let exchanged = DecoderLayer::new(attention, exchanged, mlp.view(config.hidden_size));
        let swapped = deviation(&exchanged.forward(&mut exchanged.cache(), &x, &mlp), &want);
        assert!(
            swapped > LAYER_TOLERANCE,
            "layer {layer}: exchanging the two convolutions deviates by only {swapped:e}"
        );
    }
    assert!(
        worst > 0.0,
        "a run that matched exactly would mean the reference's bfloat16 vanished"
    );
}

/// `embed_norm` over a real lookup, which is the bound `embed::tests` accounts
/// for: MLX rounds this norm's intermediate to bfloat16 before applying the
/// weight and rounds again after, and this port rounds nowhere, so one quantum
/// of the tensor's peak — 2^-9 — is the floor. Two of them is the bound.
/// Worst observed when this landed: 1.9e-3, the same as the hermetic case,
/// because the lookup under it is exact.
const EMBED_NORM_TOLERANCE: f32 = 4e-3;

/// The embedding table, left packed. Decoded whole it is `[201024, 4096]` of
/// float32 — 3.3 GB — so a row is decoded when a token asks for it, which is
/// the same slice of a leading axis an expert of a bank is.
fn embedding_table(ckpt: &Checkpoint) -> Packed<'_> {
    Packed::open(ckpt, "language_model.model.embed_tokens").expect("the table is packed")
}

/// One row of the embedding table.
fn embedding_row(table: &Packed<'_>, id: usize) -> Vec<f32> {
    table
        .decode_slice(id)
        .unwrap_or_else(|err| panic!("row {id} decodes: {err}"))
}

fn embed_norm_weight(ckpt: &Checkpoint) -> Vec<f32> {
    checkpoint_tensor(ckpt, "language_model.model.embed_norm.weight")
        .to_f32()
        .expect("a bfloat16 tensor")
}

fn root_mean_square(values: &[f32]) -> f64 {
    let sum: f64 = values.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    (sum / values.len() as f64).sqrt()
}

/// What no committed fixture can settle about the lookup: that `embed_tokens`
/// is named what this port thinks, that it is MXFP4 like every projection
/// rather than the float array a lookup usually slices, and that decoding one
/// of its rows is what the reference's embedding returns. The table is 3.3 GB
/// decoded, so only a real checkpoint carries it.
///
/// Exact rather than bounded, and that is the assertion rather than a
/// convenience: every MXFP4 element is a magnitude of at most three significant
/// bits times a power of two, so a decoded row is representable in bfloat16 and
/// the reference's dequantisation into it loses nothing. A lookup that
/// disagreed at all would be reading the wrong row or the wrong bytes, which is
/// not a difference a tolerance should absorb.
#[test]
fn the_embedding_lookup_reproduces_the_reference_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let activations = fixture::open(ACTIVATIONS);

    let ids = indices(&fixture::tensor(&activations, "input_ids"));
    let want = fixture::f32s(&fixture::tensor(&activations, "embed_out"));

    let table = embedding_table(&ckpt);
    let got = Embed::new(None, 0.0).forward(&ids, |id| embedding_row(&table, id));

    assert_eq!(got.len(), want.len(), "length");
    for (i, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "value {i}: {got:e} is not {want:e}"
        );
    }
}

/// The whole of `InklingModel.embed`: ids in, the tensor layer 0 consumes out.
/// The hermetic case drives the norm from a recorded `embed_out`, so only this
/// one runs the two steps against each other.
#[test]
fn the_embedding_reproduces_the_reference_from_ids_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    assert!(
        config.use_embed_norm,
        "the recorded pass normalised; a checkpoint that does not is a different capture"
    );

    let ids = indices(&fixture::tensor(&activations, "input_ids"));
    let want = fixture::f32s(&fixture::tensor(&activations, "embed_norm_out"));

    let table = embedding_table(&ckpt);
    let weight = embed_norm_weight(&ckpt);
    let embed = Embed::new(Some(&weight), config.rms_norm_eps);

    let deviation = deviation(&embed.forward(&ids, |id| embedding_row(&table, id)), &want);
    assert!(deviation <= EMBED_NORM_TOLERANCE, "deviation {deviation:e}");
}

/// What the table's padding rows hold, which is the question that decides
/// whether the lookup needs a guard at all.
///
/// `lm_head`'s padding is all-zero codes under all-zero scales — the MXFP4
/// fixture's `vocab_padding` slice pins that — and the natural assumption is
/// that the embedding table at the other end of the model matches. It does not.
/// Every one of its 966 padding rows carries small nonzero values, so an id
/// past `unpadded_vocab_size` returns noise rather than nothing.
///
/// How much noise is the second half of the answer, and it is not the
/// thousandfold attenuation the table suggests. `embed_norm` divides by the
/// row's own RMS, which would erase the difference outright; what stops it is
/// that a padding row's mean square lands 46 times below `rms_norm_eps`, so the
/// epsilon dominates the divide. The gap survives, compressed by two orders of
/// magnitude: measured when this landed, 1228x in the table and 7.8x after the
/// norm. Both halves of that are asserted, because either one alone would leave
/// the wrong impression.
///
/// Nothing is guarded on the strength of this: the reference does not guard,
/// and a tokenizer cannot emit such an id. It is stated so that the assumption
/// is on the record rather than inherited from `lm_head`.
#[test]
fn the_embedding_tables_padding_rows_are_small_but_neither_zero_nor_normalised_away() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);

    let unpadded = config
        .unpadded_vocab_size
        .expect("the checkpoint states an unpadded vocabulary");
    let table = embedding_table(&ckpt);
    assert_eq!(table.slices(), config.vocab_size, "table rows");
    assert!(unpadded < config.vocab_size, "the table is padded at all");

    let weight = embed_norm_weight(&ckpt);
    let padding: Vec<usize> = (unpadded..config.vocab_size).collect();
    let vocabulary = indices(&fixture::tensor(&activations, "input_ids"));
    let looked_up =
        |ids: &[usize]| Embed::new(None, 0.0).forward(ids, |id| embedding_row(&table, id));
    let normalised = |ids: &[usize]| {
        Embed::new(Some(&weight), config.rms_norm_eps).forward(ids, |id| embedding_row(&table, id))
    };

    let rows = looked_up(&padding);
    for (row, values) in rows.chunks_exact(config.hidden_size).enumerate() {
        assert!(
            values.iter().any(|x| *x != 0.0),
            "row {} is all zeros",
            unpadded + row
        );
    }

    let in_the_table = root_mean_square(&looked_up(&vocabulary)) / root_mean_square(&rows);
    assert!(
        in_the_table > 100.0,
        "padding is only {in_the_table:.1}x below a real row in the table"
    );

    let after_the_norm =
        root_mean_square(&normalised(&vocabulary)) / root_mean_square(&normalised(&padding));
    assert!(
        after_the_norm > 2.0,
        "the norm erased the gap entirely, at {after_the_norm:.1}x"
    );
    assert!(
        in_the_table / after_the_norm > 100.0,
        "the norm compressed the gap by only {:.0}x, from {in_the_table:.1} to {after_the_norm:.1}",
        in_the_table / after_the_norm
    );
}

/// What only the whole model can settle: that forty-two layers run in order,
/// each built with the attention config and the MLP its index calls for, each
/// against its own cache, and that the tensor names outside a layer are the ones
/// this port assumes.
///
/// This is legitimately looser than any single layer's 1.5e-2, and by more than
/// the length of the chain would suggest, because what it measures is different.
/// Every intermediate of the recorded pass was rounded to bfloat16 and none of
/// this port's are, so each layer contributes about one quantum — 2^-9 = 2.0e-3
/// — of its own output. Forty-two of those added in quadrature is 1.3e-2, and
/// the measurement is twice that: worst observed when this landed, 2.8e-2.
/// Accumulation, not compounding; a port with an arithmetic mistake in it would
/// not land within a factor of two of the rounding it inherited.
///
/// A factor of two in hand. Against the weakest mutation this bound has to catch
/// — the normed state returned where the pre-norm one belongs, which is the one
/// wiring mistake `LanguageModel.__call__`'s two last lines invite — 1.0, a
/// decade above.
///
/// What this bound is *not* is a description of the tensor, and forty-two layers
/// deep that stops being a quibble. The worst element disagrees by 2.8e-2 of the
/// peak and the root mean square of the disagreements by 6.5e-4 — a factor of
/// forty-four — over a tensor whose own peak is sixty-six times its RMS. So the
/// number is set by a handful of elements at the top of a very long tail, and
/// widening it to admit them admits far more everywhere else.
///
/// B3 should not define "matches the oracle" for logits this way. A tensor-wide
/// epsilon over 201024 logits would be pinned by the same few outliers while
/// saying nothing about the 200000 that decide nothing; what survives 42 layers
/// of accumulated bfloat16 is the *ordering*, so the assertion with teeth is
/// argmax agreement and top-k identity, with an epsilon kept only as a coarse
/// guard beside it.
const STACK_TOLERANCE: f32 = 6e-2;

/// The same pass through the final norm, which is looser again for a reason
/// worth writing down: the norm divides each row by its own RMS and so
/// *compresses* the tensor it is measured against. `layers_out` peaks at 794624
/// against an RMS of 11983 — one element sixty-six times the typical one — and
/// `norm_out` peaks at 27.6 against 3.1. The same disagreement measured against
/// a peak that dropped by a factor of nearly thirty relative to the bulk is a
/// larger fraction of it.
///
/// On top of that this norm rounds where MLX rounds twice, which is the 2e-3
/// `EMBED_NORM_TOLERANCE` accounts for and is not what dominates here. Worst
/// observed when this landed: 4.9e-2, again a factor of two in hand.
const FINAL_NORM_TOLERANCE: f32 = 1e-1;

/// What a forward pass may hold resident, mapped pages included.
///
/// Observed when this landed: 16.7 GiB, of which 1.01 GiB is the scratch every
/// weight is decoded into and the rest is packed bytes the pass touched and the
/// kernel therefore kept — every layer's five projections, the two dense FFNs,
/// and the experts eight tokens routed to. If no two of those tokens had ever
/// agreed on an expert, all 48 selections a layer could make would be distinct
/// and the same pass would touch about 28 GiB, so this sits above the arithmetic
/// ceiling for eight tokens rather than above one measurement of one prompt.
///
/// What it has to catch is decades away and not a matter of margin: one layer's
/// routed bank decoded eagerly is 25 GB on top of this, and the whole model
/// decoded eagerly is 1.1 TB — against a 512 GiB host, and against the 130.6 GiB
/// the checkpoint occupies packed.
const RESIDENT_BOUND: u64 = 32 << 30;

/// A [`ModelWeights`] that watches a pass go by.
///
/// It reports two things about the pass it wraps and changes nothing about it:
/// the resident set, sampled per layer, and how far the layers the fixture
/// captured have drifted from what the reference recorded for them.
///
/// The resident set is sampled per layer rather than once at the end because the
/// scratch is faulted in as it is first written and the packed pages a layer
/// reads join the resident set as it reads them, so a single reading afterwards
/// would find whatever the last layer left rather than the worst any layer
/// reached.
///
/// The drift is what a layer-at-a-time comparison cannot see. Each captured
/// layer is measured here against the reference's output for it while being fed
/// *this port's* input, where `the_decoder_layer_reproduces_the_reference` feeds
/// it the reference's own — so what these numbers carry, and those do not, is
/// everything the layers below contributed.
struct Watched<'a> {
    inner: &'a CheckpointWeights<'a>,
    recorded: Vec<(usize, Vec<f32>)>,
    peak: Cell<u64>,
    drift: RefCell<Vec<(usize, f32)>>,
}

impl<'a> Watched<'a> {
    fn new(inner: &'a CheckpointWeights<'a>, activations: &Checkpoint) -> Self {
        Self {
            inner,
            recorded: CAPTURED_LAYERS
                .into_iter()
                .map(|layer| {
                    (
                        layer,
                        fixture::f32s(&fixture::layer_tensor(activations, layer, "out")),
                    )
                })
                .collect(),
            peak: Cell::new(fixture::resident_bytes()),
            drift: RefCell::new(Vec::new()),
        }
    }

    /// How far each captured layer had drifted by the time the pass reached it,
    /// shallowest first.
    fn drift(&self) -> Vec<(usize, f32)> {
        self.drift.borrow().clone()
    }
}

impl ModelWeights for Watched<'_> {
    fn embedding_row(&self, id: usize) -> Vec<f32> {
        self.inner.embedding_row(id)
    }

    fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: &[f32]) -> Vec<f32> {
        let out = self.inner.run_layer(index, cache, x);
        self.peak
            .set(self.peak.get().max(fixture::resident_bytes()));
        if let Some((_, want)) = self.recorded.iter().find(|(layer, _)| *layer == index) {
            self.drift.borrow_mut().push((index, deviation(&out, want)));
        }
        out
    }
}

/// The typical disagreement beside [`deviation`]'s worst one: the root mean
/// square of the differences, over the same tensor peak.
///
/// Reported and never asserted. What it answers is whether a bound of 2.8e-2 is
/// describing the tensor or one element of it — and at 42 layers deep the answer
/// decides how B3 can compare logits at all.
fn typical_deviation(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |worst, w| worst.max(w.abs()));
    let squares: f64 = got
        .iter()
        .zip(want)
        .map(|(got, want)| f64::from(got - want).powi(2))
        .sum();
    (squares / got.len() as f64).sqrt() as f32 / scale
}

/// The recorded ids, and the two answers the reference ended its forward pass
/// with.
fn recorded_stack(activations: &Checkpoint) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let of = |name: &str| fixture::f32s(&fixture::tensor(activations, name));
    (
        indices(&fixture::tensor(activations, "input_ids")),
        of("layers_out"),
        of("norm_out"),
    )
}

#[test]
fn the_whole_stack_reproduces_the_reference_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (ids, layers_out, norm_out) = recorded_stack(&activations);

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let model = weights.model();
    assert_eq!(model.layers(), 42, "Inkling-Small is forty-two layers");

    let watched = Watched::new(&weights, &activations);
    let started = Instant::now();
    let got = model.forward(&mut ModelCache::new(&config), &ids, &watched);
    let normed = model.final_norm(&got);
    eprintln!(
        "42 layers over {} tokens in {:?}",
        ids.len(),
        started.elapsed()
    );

    // How the error grew on the way down, which is the question a fixture of
    // three layers cannot answer on its own. Reported rather than bounded: the
    // captured layers are 0, 2 and 5, and three points inside the first six of
    // forty-two do not make a curve worth asserting a shape for.
    for (layer, drift) in watched.drift() {
        eprintln!("layer {layer}: drift {drift:e}");
    }

    let mut worst = 0.0f32;
    for (what, got, want, bound) in [
        ("layers_out", &got, &layers_out, STACK_TOLERANCE),
        ("norm_out", &normed, &norm_out, FINAL_NORM_TOLERANCE),
    ] {
        let deviation = deviation(got, want);
        eprintln!(
            "{what}: worst {deviation:e}, typical {:e}",
            typical_deviation(got, want)
        );
        assert!(deviation <= bound, "{what}: deviation {deviation:e}");
        worst = worst.max(deviation);
    }
    assert!(
        worst > 0.0,
        "a run that matched exactly would mean the reference's bfloat16 vanished"
    );

    // The two ends of `LanguageModel.__call__`'s last two lines are four decades
    // apart in magnitude, so returning either where the other belongs is a
    // mistake the numbers catch outright. Free, because both are already here.
    let swapped = deviation(&normed, &layers_out);
    assert!(
        swapped > STACK_TOLERANCE,
        "the normed state stands in for the pre-norm one at only {swapped:e}"
    );
}

#[test]
fn the_whole_stack_holds_its_resident_set_under_a_bound() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (ids, _, _) = recorded_stack(&activations);

    let before = fixture::resident_bytes();
    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let watched = Watched::new(&weights, &activations);
    weights
        .model()
        .forward(&mut ModelCache::new(&config), &ids, &watched);

    // The structural half of the bound, and the one with teeth. What the pass
    // decodes into is one buffer, sized from the config before it starts, and
    // `Scratch` panics rather than growing — so the pass having completed at all
    // says nothing was decoded that this did not allow for. What remains to say
    // is how much that is, and 2 GiB is chosen against the thing it must refuse:
    // a MoE layer's routed bank decoded whole, which at 256 experts of 100.66 MB
    // is 25 GB and would land a decade above.
    let scratch = weights.scratch_floats() * size_of::<f32>();
    assert!(
        scratch < (2 << 30),
        "the pass decodes into {scratch} bytes at once"
    );
    assert!(
        expert_scratch_floats(&config) * config.n_routed_experts
            > 8 * layer_scratch_floats(&config),
        "a bank decoded whole would fit in the layer buffer, so this bounds nothing"
    );

    let (peak, gib) = (watched.peak.get(), (1u64 << 30) as f64);
    eprintln!(
        "scratch {:.2} GiB, RSS {:.2} -> peak {:.2} GiB",
        scratch as f64 / gib,
        before as f64 / gib,
        peak as f64 / gib,
    );
    assert!(
        peak < RESIDENT_BOUND,
        "peak RSS {peak} bytes is over the bound of {RESIDENT_BOUND}"
    );
    // A pass that never faulted a weight in would mean the checkpoint was
    // already resident, and the reading would be measuring the machine rather
    // than this.
    assert!(peak > before, "the pass grew the resident set by nothing");
}

/// The coarse guard beside the ordering claims, for a head driven from the
/// reference's own normed state.
///
/// A logit is one row of `nn.Linear` over a hidden state this port has in
/// common with the reference, so what separates the two answers is the dtype
/// and nothing else: `mx.quantized_matmul` multiplies the packed 4-bit weights
/// in bfloat16 and rounds the result to bfloat16, and this decodes them and
/// multiplies in float32. One quantum at 2^-9 is 2.0e-3 of the tensor's peak,
/// and 6e-3 is the same three-quanta bound, for the same reason, as the recorded
/// attention step and the MoE layer.
///
/// Worst observed when this landed: 3.5e-3 over the ranked logits and 2.7e-3
/// over all 200058 of the last position, a factor of 1.7 in hand. Against the
/// weakest mutation it has to catch — the muP divide dropped, which multiplies
/// every logit by sixteen — fifteen, four decades above.
const HEAD_TOLERANCE: f32 = 6e-3;

/// The same guard for logits made from a hidden state this port produced itself,
/// forty-two layers deep, where the drift the stack accumulated arrives on top
/// of the head's own rounding.
///
/// An order of magnitude looser than [`HEAD_TOLERANCE`], and it is the stack's
/// number rather than the head's: `layers_out` deviates by 2.8e-2 of its peak
/// and `norm_out` by 4.9e-2, so 6e-2 is `STACK_TOLERANCE` inherited unchanged.
/// The head adds nothing measurable to it — 3.5e-3 against 3.1e-2 — because one
/// projection over a drifted input carries the drift and little else.
///
/// Worst observed when this landed: 3.2e-2 over the ranked logits and 3.1e-2
/// over all 200058 of the last position, a factor of 1.9 in hand.
const MODEL_LOGIT_TOLERANCE: f32 = 6e-2;

/// The ranking the reference ended its forward pass with: the top ids of every
/// position and the logits they carried.
///
/// The depth comes from the fixture rather than from a constant here, so a dump
/// regenerated at a different `--top-k` needs nothing changed on this side.
struct Ranking {
    k: usize,
    ids: Vec<usize>,
    values: Vec<f32>,
}

impl Ranking {
    fn load(activations: &Checkpoint) -> Self {
        let recorded = fixture::tensor(activations, "logits_topk_ids");
        Self {
            k: *recorded.shape().last().expect("a ranking has a depth"),
            ids: indices(&recorded),
            values: fixture::f32s(&fixture::tensor(activations, "logits_topk_values")),
        }
    }

    fn positions(&self) -> usize {
        self.ids.len() / self.k
    }

    fn at(&self, position: usize) -> (&[usize], &[f32]) {
        let row = position * self.k;
        (&self.ids[row..][..self.k], &self.values[row..][..self.k])
    }
}

/// One position's computed logits against the reference's recorded ranking of
/// them.
struct Agreement {
    position: usize,
    /// The id this port ranked first.
    argmax: usize,
    /// How many ids the two orders agree on, from the top down.
    depth: usize,
    /// Whether the rank they first differ at is one the reference's own values
    /// leave undetermined. `true` when the two orders agree all the way down.
    tied_at_the_break: bool,
    /// The best logit the reference recorded here.
    top: f32,
    /// Its gap to the runner-up.
    margin: f32,
    /// The worst disagreement over the logits the ranking names.
    ///
    /// Over those rather than over the tensor, and that is the claim rather
    /// than a convenience: what a ranking is about is the logits at the top,
    /// and a bound taken over 200058 of them would be set by the outliers among
    /// the 200000 that decide nothing.
    deviation: f32,
}

impl Agreement {
    /// The deviation as a fraction of the logit it was measured beside, which is
    /// what the tolerances are stated in.
    fn relative(&self) -> f32 {
        self.deviation / self.top.abs()
    }
}

fn agreement(position: usize, logits: &[f32], ids: &[usize], values: &[f32]) -> Agreement {
    let mine = top_k(logits, ids.len());
    let depth = mine.iter().zip(ids).take_while(|(a, b)| a == b).count();
    Agreement {
        position,
        argmax: mine[0],
        depth,
        tied_at_the_break: depth >= values.len() || undetermined_at(values, depth),
        top: values[0],
        margin: values[0] - values[1],
        deviation: ids.iter().zip(values).fold(0.0f32, |worst, (id, want)| {
            worst.max((logits[*id] - want).abs())
        }),
    }
}

/// Whether the reference's own logits leave `rank` undetermined: sorted
/// descending, a value tied with either neighbour is one bfloat16 cannot tell
/// apart from it, and which id comes first is then a tie-break rather than an
/// order.
fn undetermined_at(values: &[f32], rank: usize) -> bool {
    let neighbours = [rank.checked_sub(1), Some(rank + 1)];
    neighbours
        .into_iter()
        .flatten()
        .filter_map(|r| values.get(r))
        .any(|value| *value == values[rank])
}

/// Every position's logits against what the reference recorded: reported in
/// full, then asserted on the two claims both callers make, with the ordering
/// claim beneath the argmax left to each of them.
///
/// **The argmax, on every position.** The one assertion no amount of accumulated
/// bfloat16 is allowed to move.
///
/// **The values, over the logits the ranking names.** Ordering cannot see the
/// muP divide, and this is where it is caught.
///
/// The margin is reported beside the deviation because their ratio is what says
/// whether an argmax that agreed did so with room to spare or by luck. Nothing
/// asserts it, and the reason is on the record rather than assumed: end to end
/// the ratio falls to 0.5 at the first position, so the argmax there survives a
/// deviation twice its own margin — it agrees, and it does not agree robustly. A
/// bound on the ratio would be asserting the prompt rather than the port.
fn check_logits(
    what: &str,
    logits: &[f32],
    vocab: usize,
    ranking: &Ranking,
    bound: f32,
) -> Vec<Agreement> {
    assert_eq!(
        logits.len(),
        ranking.positions() * vocab,
        "{what}: {} logits over {} positions of {vocab}",
        logits.len(),
        ranking.positions()
    );

    let agreements: Vec<Agreement> = (0..ranking.positions())
        .map(|position| {
            let (ids, values) = ranking.at(position);
            agreement(position, &logits[position * vocab..][..vocab], ids, values)
        })
        .collect();

    for agreement in &agreements {
        let (ids, _) = ranking.at(agreement.position);
        eprintln!(
            "{what} {}: top-1 {}, depth {} of {}{}, margin {:.4} against deviation {:.4} ({:.1}x)",
            agreement.position,
            ids[0],
            agreement.depth,
            ranking.k,
            if agreement.tied_at_the_break {
                " (breaks on a tie)"
            } else {
                " (breaks on a reordering)"
            },
            agreement.margin,
            agreement.deviation,
            agreement.margin / agreement.deviation,
        );

        assert_eq!(
            agreement.argmax, ids[0],
            "{what} {}: argmax",
            agreement.position
        );
        assert!(
            agreement.relative() <= bound,
            "{what} {}: deviation {:e} over the ranked logits",
            agreement.position,
            agreement.relative()
        );
    }
    agreements
}

/// The recorded logits of the position the dump kept every logit of, which is
/// the last one — the one that decides the next token — and which is recorded
/// *before* the truncation so that the padding is in the fixture.
fn recorded_logits(activations: &Checkpoint) -> Vec<f32> {
    fixture::f32s(&fixture::tensor(activations, "logits_untruncated"))
}

/// One row of the head, decoded. The engine never asks for one — it multiplies
/// through [`CheckpointWeights::head_projection`] — but which rows the head
/// holds is a claim about the checkpoint that only reading them settles.
fn head_row(weights: &CheckpointWeights<'_>, id: usize) -> Vec<f32> {
    weights
        .head_packed()
        .decode_slice(id)
        .unwrap_or_else(|err| panic!("head row {id} decodes: {err}"))
}

fn head_logits(weights: &CheckpointWeights<'_>, normed: &[f32]) -> Vec<f32> {
    let head = weights.head();
    let started = Instant::now();
    let logits = head.forward(normed, weights.head_projection());
    eprintln!(
        "{} rows of the head in {:?}",
        head.vocab(),
        started.elapsed()
    );
    logits
}

/// What no committed fixture can settle about the head: that
/// `language_model.lm_head` is named what this port assumes, that it is MXFP4
/// like every projection, and that the muP divide, the projection and the
/// truncation reproduce `_logits_from_norm` from the reference's own normed
/// state. Decoded it is `[201024, 4096]` of float32 — 3.3 GB — so only a real
/// checkpoint carries it.
///
/// **The assertion is the ordering, and that is a decision.** A tensor-wide
/// epsilon has stopped describing these tensors: at the stack's output the worst
/// element disagrees by 2.8e-2 of the peak while the root mean square of the
/// disagreements is 6.5e-4, a factor of forty-four, so the bound is pinned by a
/// handful of outliers in a long tail. Over 200058 logits that gets worse — the
/// same few outliers would set a bound that says nothing about the 200000 logits
/// which decide nothing. What is worth asserting is the *ordering*, so
/// [`check_logits`] asserts the argmax, with the tensor-wide comparison kept
/// beside it as a coarse guard and nothing more.
///
/// **Below the argmax, the claim is stronger than top-k identity.** "Identical
/// to depth k" is the claim that has to be tuned until it passes — at k = 5 it
/// would not pass here, position 7 breaks at rank 4 — and tuning it would report
/// the wrong thing. What is true instead is that wherever the reference's logits
/// determine an order, this reproduces it: at every position the rank the two
/// orders first differ at is one where the reference's own recorded logits are
/// *equal*. Three significant digits of bfloat16 over 200058 values leaves ties
/// everywhere — position 7's break is 15.75 against 15.75 — and a tie broken by
/// index is not an order to reproduce. Measured depths: 8, 11, 14, 18, 17, 9, 19
/// and 4 of a recorded 32, every one of them ending on a tie. A genuine
/// reordering at any depth fails this, where a depth bound set to the shallowest
/// observed would not.
///
/// The coarse guard is the last position's 200058 logits, which is the one
/// position the fixture carries them all for. It answers a different question
/// from the ranking — whether the arithmetic is right anywhere below the top —
/// and 6.4 MB for the other seven positions would answer it seven more times.
#[test]
fn the_head_reproduces_the_reference_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (_, _, norm_out) = recorded_stack(&activations);

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let head = weights.head();
    let logits = head_logits(&weights, &norm_out);
    let ranking = Ranking::load(&activations);
    for agreement in check_logits("head", &logits, head.vocab(), &ranking, HEAD_TOLERANCE) {
        assert!(
            agreement.tied_at_the_break,
            "head {}: the two orders differ at rank {}, where the reference's own logits are \
             distinct — so this is a reordering rather than a tie broken the other way",
            agreement.position, agreement.depth,
        );
    }

    let recorded = recorded_logits(&activations);
    let last = (ranking.positions() - 1) * head.vocab();
    let deviation = deviation(&logits[last..], &recorded[..head.vocab()]);
    eprintln!("the last position's {} logits: {deviation:e}", head.vocab());
    assert!(deviation <= HEAD_TOLERANCE, "deviation {deviation:e}");
    assert!(
        deviation > 0.0,
        "a run that matched exactly would mean the reference's bfloat16 vanished"
    );
}

/// The muP divide, which no ordering test can reach: it scales every logit by
/// sixteen and moves no argmax at all. Asserted on the values the ranking names,
/// against the same recorded values the agreement above is measured on.
#[test]
fn dropping_the_mup_divide_moves_the_logits_and_not_the_ranking() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (_, _, norm_out) = recorded_stack(&activations);
    assert!(
        config.logits_mup_width_multiplier > 1.0,
        "a checkpoint that did not scale its logits could not settle this"
    );

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let head = weights.head();
    let undivided = LmHead::new(config.hidden_size, head.vocab(), 1.0)
        .forward(&norm_out, weights.head_projection());

    let multiplier = config.logits_mup_width_multiplier;
    let ranking = Ranking::load(&activations);
    for position in 0..ranking.positions() {
        let (ids, values) = ranking.at(position);
        let mine = &undivided[position * head.vocab()..][..head.vocab()];
        assert_eq!(
            top_k(mine, 1)[0],
            ids[0],
            "position {position}: the multiplier moved the argmax, so an ordering \
             test would have caught it and this one says nothing"
        );

        let scaled = mine[ids[0]] / values[0];
        assert!(
            (scaled / multiplier - 1.0).abs() <= HEAD_TOLERANCE,
            "position {position}: the undivided top logit is {scaled:.4} times the recorded \
             one, not {multiplier}",
        );
    }
}

/// What the head's padding rows hold, which is what decides whether the
/// truncation is load-bearing.
///
/// They are all-zero MXFP4 codes under all-zero scales, so they decode to
/// exactly 0.0 and produce a logit of exactly 0.0 — and a zero is not nothing.
/// It outranks every logit that came out negative, which at the recorded
/// position is 78.5% of the vocabulary, so an untruncated head inserts 966 ids
/// that cannot be generated into the middle of every distribution a sampler
/// would draw from.
///
/// What it does not do at this prompt is take the argmax, and that is worth
/// stating precisely rather than leaving as a near miss: every recorded
/// position's best real logit is positive — 11.8 at the weakest — so a zero
/// lands about forty thousand ranks below it. The argmax-flipping case is the
/// hermetic one in `head::tests`, where the head is built so that nothing real
/// beats zero. Here the claim is the ranking, and it is a live one: nucleus and
/// top-k sampling both read the order this would corrupt.
///
/// The other end of the model does not agree about any of this — `embed_tokens`'
/// padding rows are small but nonzero — so neither end can be read off the
/// other.
#[test]
fn the_heads_padding_rows_are_zero_and_would_outrank_most_of_the_vocabulary() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);

    let unpadded = config
        .unpadded_vocab_size
        .expect("the checkpoint states an unpadded vocabulary");
    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    assert_eq!(
        weights.head().vocab(),
        unpadded,
        "the head stops at the cut"
    );
    assert!(unpadded < config.vocab_size, "the head is padded at all");

    for id in unpadded..config.vocab_size {
        assert!(
            head_row(&weights, id).iter().all(|w| *w == 0.0),
            "head row {id} is not all zeros"
        );
    }

    // Restated on the reference's own logits, because a row of zeros decoding to
    // zero is this port's arithmetic and a logit of zero is the reference's.
    let recorded = recorded_logits(&activations);
    assert_eq!(recorded.len(), config.vocab_size, "recorded untruncated");
    let (vocabulary, padding) = recorded.split_at(unpadded);
    assert!(padding.iter().all(|logit| *logit == 0.0), "padding logits");

    let above = vocabulary.iter().filter(|logit| **logit > 0.0).count();
    let below = vocabulary.len() - above;
    eprintln!(
        "a padding zero would rank {} of {}, ahead of {below} real ids ({:.1}% of the vocabulary)",
        above + 1,
        config.vocab_size,
        100.0 * below as f32 / vocabulary.len() as f32,
    );
    assert!(
        below > vocabulary.len() / 2,
        "only {below} of {} real logits fall below zero, so the padding would sit in the tail \
         and truncation would be a formality",
        vocabulary.len()
    );
    assert!(
        above > 0,
        "no real logit beats zero, so the padding takes the argmax and the hermetic case is \
         reachable here too"
    );
}

/// The whole engine over this checkpoint: the stack, its final norm, and the
/// head, which is what `LanguageModel` is.
fn generator<'a>(weights: &'a CheckpointWeights<'a>) -> Generator<'a> {
    Generator::new(weights.model(), weights.head(), weights.head_projection())
}

/// One call of the engine, timed: `ids` through the model and the head, and the
/// logits of the last of them.
///
/// The two regimes are one call apart — a prompt against fresh caches, or one
/// token against caches carrying everything before it — and they are not the
/// same price. Measured when this landed: 7 tokens prefilled in 54.7 s, the 8th
/// decoded in 9.2 s.
///
/// The routed experts are why, and not the matmuls. A layer's five projections
/// and the head are decoded whichever regime it is — 9.0 GB and 3.3 GB, fixed —
/// but a MoE layer decodes the experts its tokens *chose*, and seven tokens
/// choose up to seven times as many as one. That is what makes a decode step a
/// sixth of a seven-token prefill rather than a seventh of it: the fixed part is
/// what stops it from being cheaper still, and it is exactly the part a
/// quantised Metal kernel deletes by never decoding at all.
fn timed_logits(
    what: &str,
    weights: &CheckpointWeights<'_>,
    cache: &mut ModelCache,
    ids: &[usize],
) -> Vec<f32> {
    let started = Instant::now();
    let logits = generator(weights).logits(cache, ids, weights);
    eprintln!("{what}: {} token(s) in {:?}", ids.len(), started.elapsed());
    logits
}

/// The invariant the whole engine rests on, against trained weights: prefilling
/// N tokens and then decoding the (N+1)th gives the same logits as one prefill
/// over all N+1.
///
/// **Exactly the same, and that is the assertion.** Every tolerance in this file
/// exists because the reference computes in bfloat16 and this port in float32.
/// Nothing of the sort is involved here — both sides of this comparison are this
/// port, running the same scalar float32 arithmetic over the same weights, and
/// the only thing a split changes is where a key comes from. A tolerance would
/// absorb precisely what this exists to catch: a query offset off by one moves a
/// mask entry by one distance along a band the checkpoint trained to be smooth,
/// so the answer would only drift.
///
/// This is the same claim `generate::tests` makes on the synthetic stack, and
/// what it adds is everything a five-layer model of width 32 cannot reach: the
/// checkpoint's own 42 layers, its sliding and global attentions, the packed
/// experts a router chooses per token, and a head over 200058 rows. The
/// synthetic stack cannot say that a decode step routes to the same experts a
/// prefill did, because its expert bank is sixteen held tensors rather than 256
/// decoded on demand into a buffer the pass reuses.
#[test]
fn prefilling_then_decoding_matches_one_prefill_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (ids, _, _) = recorded_stack(&activations);

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let (prompt, last) = ids.split_at(ids.len() - 1);
    assert_eq!(last.len(), 1, "one token is decoded");

    let cache = &mut ModelCache::new(&config);
    timed_logits("prefill", &weights, cache, prompt);
    let decoded = timed_logits("decode", &weights, cache, last);
    let whole = timed_logits("one prefill", &weights, &mut ModelCache::new(&config), &ids);

    let deviation = deviation(&decoded, &whole);
    eprintln!(
        "prefill {} + decode 1 against one prefill of {}: deviation {deviation:e}, top-1 {} against {}",
        prompt.len(),
        ids.len(),
        greedy(&decoded),
        greedy(&whole),
    );
    assert_eq!(decoded, whole, "the two paths are the same arithmetic");
}

/// How many tokens the generation case decodes, which is the whole of what the
/// fixture recorded.
///
/// A decoded token costs the CPU path 9.2 s, so eight of them and the prefill
/// under them is about two minutes — spent deliberately, and the reason the
/// fixture records exactly this many rather than a length a test would have to
/// be trimmed to.
const GENERATED: usize = 8;

/// The milestone: this engine and mlx-vlm producing the same tokens.
///
/// Everything else in this file compares a tensor against a tensor. This
/// compares a *sequence of decisions* — each token sampled from logits this port
/// produced, fed back through this port's caches, and asked to agree with what
/// the reference did from the same prompt. It is the first assertion that would
/// fail for a cache mistake the equivalence test above cannot see, because that
/// one feeds the recorded ids back and this one feeds back whatever it decided.
///
/// **Why token agreement is the right claim and tensor agreement is not.** Forty
/// two layers of accumulated bfloat16 put this port's logits 0.13 to 0.56 away
/// from the reference's over the ranked ids, where adjacent logits in the top 32
/// are routinely 0.0625 apart. So the tail reorders — at six of eight recorded
/// positions, on values that are distinct — and the argmax does not. Greedy
/// decoding is what that leaves reproducible, which is why greedy is the only
/// sampler the engine has.
///
/// **And it is not guaranteed to hold forever.** Every generated token is one
/// argmax over a drifted distribution, so a step whose top two logits are closer
/// together than the drift is a coin toss, and once one token differs the
/// sequences are incomparable from there. All eight agreed when this landed —
/// `[656, 13, 623, 180069, 86333, 60500, 220, 23]`, the whole of what the
/// fixture recorded — but that is a measurement and not a proof, and the drift
/// it survived is the same drift that reorders the tail at six of eight recorded
/// positions. How many agree is therefore reported as well as asserted, and the
/// assertion is on all of `GENERATED`: a test shrunk until it passed would
/// report the opposite of what is true.
#[test]
fn the_generated_tokens_match_the_oracle_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (ids, _, _) = recorded_stack(&activations);

    let oracle = indices(&fixture::tensor(&activations, "greedy_continuation"));
    assert!(
        oracle.len() >= GENERATED,
        "the fixture records {} of {GENERATED} tokens",
        oracle.len()
    );
    let want = &oracle[..GENERATED];

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let started = Instant::now();
    let got =
        generator(&weights).generate(&mut ModelCache::new(&config), &ids, GENERATED, &weights);
    let elapsed = started.elapsed();

    // No mean over these: one prefill of the prompt and `GENERATED - 1` decode
    // steps produced them, and those are the two regimes at six times each
    // other's price. `timed_logits` reports each on its own.
    //
    // A token that stops agreeing is a tie before it is a bug. Both sides break
    // one towards the lower id — `mx.argmax` there and `top_k` here — so a step
    // whose top two logits are equal in the reference's bfloat16 but ordered in
    // this port's float32 disagrees without either being wrong. Check the
    // recorded `logits_topk_values` at the position first.
    let agreed = got.iter().zip(want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{GENERATED} tokens in {elapsed:?} — one prefill of {} and {} decode steps — \
         {agreed} agreeing with the oracle\n  got  {got:?}\n  want {want:?}",
        ids.len(),
        GENERATED - 1,
    );
    assert_eq!(got, want, "{agreed} of {GENERATED} tokens agree");
}

/// Whether this checkpoint ties its embeddings, which decides which tensor the
/// head's rows come out of.
///
/// It does not: `tie_word_embeddings` is absent from the config — so false, the
/// reference's default — and the checkpoint carries a `language_model.lm_head`
/// beside `embed_tokens`. Asserted on the rows rather than on the flag alone,
/// because a port that read the wrong table would still produce a full set of
/// plausible logits.
#[test]
fn the_checkpoint_does_not_tie_its_embeddings() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    assert!(!config.tie_word_embeddings);

    let head = Packed::open(&ckpt, "language_model.lm_head").expect("an untied checkpoint");
    assert_eq!(head.slices(), config.vocab_size);
    assert_eq!(head.slice_len(), config.hidden_size);

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let table = embedding_table(&ckpt);
    for id in [0, config.vocab_size / 2, config.vocab_size - 1] {
        assert_eq!(
            head_row(&weights, id),
            head.decode_slice(id).expect("decodes")
        );
        assert_ne!(
            head_row(&weights, id),
            embedding_row(&table, id),
            "row {id}: the two ends of the model hold the same weights, so this settles nothing"
        );
    }
}

/// The whole model, ids to logits: what only a pass that ran every layer itself
/// can settle, which is how much of the ordering survives the drift the layers
/// accumulate.
///
/// `the_head_reproduces_the_reference_against_real_weights` feeds the head the
/// reference's own normed state, so what it measures is the head. This feeds it
/// this port's, and the difference between the two sets of numbers is everything
/// the stack contributed.
///
/// **The argmax survives and the ordering beneath it does not.** Every one of
/// the eight positions agrees on its top-1. Below that, the depth the two orders
/// agree to — 8, 7, 5, 3, 4, 1, 23 and 8 of a recorded 32 — is reordering rather
/// than ties broken differently: the head alone breaks only where the
/// reference's own bfloat16 logits are equal, and here six of the eight breaks
/// land on values that are distinct. At position 5 the *runner-up* already
/// reorders. So top-5 identity holds at three positions of eight, and reporting
/// that is the point; shrinking k until it passed would have reported the
/// opposite of what is true.
///
/// That is the same drift measured elsewhere and not a new one. The deviation
/// over the ranked logits is 0.13 to 0.56 against the head's own 0.03 to 0.06 —
/// two to nine bfloat16 quanta, where adjacent logits in the top 32 are
/// routinely 0.0625 apart — so a tail that reorders is arithmetic, not a
/// mistake. What it says about the engine is that greedy decoding is
/// reproducible and a sampler's tail is not, position by position.
///
/// **And the argmax that survives is not always robust.** At positions 0 and 2
/// the deviation is twice the reference's own top-1/top-2 margin (0.5x and 0.4x
/// of it), so those two agree by luck rather than by margin. The other six carry
/// 1.6x to 19x. Reported and not asserted: a bound on that ratio would be a
/// bound on the prompt.
#[test]
fn the_argmax_survives_the_whole_model_against_real_weights() {
    let Some(dir) = checkpoint_dir() else { return };
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let config = text_config(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let (ids, _, _) = recorded_stack(&activations);

    let weights = CheckpointWeights::open(&ckpt, &config).expect("the checkpoint's weights map");
    let model = weights.model();
    let started = Instant::now();
    let hidden = model.forward(&mut ModelCache::new(&config), &ids, &weights);
    let logits = head_logits(&weights, &model.final_norm(&hidden));
    eprintln!("{} tokens to logits in {:?}", ids.len(), started.elapsed());

    let vocab = weights.head().vocab();
    let agreements = check_logits(
        "model",
        &logits,
        vocab,
        &Ranking::load(&activations),
        MODEL_LOGIT_TOLERANCE,
    );

    // Nothing asserts a depth here, and that is the finding rather than a gap:
    // the shallowest is 1, so any bound that passed would say no more than the
    // argmax assertion already does. The head's tie-only claim is where the
    // ordering below the top is pinned; what the stack does to it is reported.
    let reordered = agreements
        .iter()
        .filter(|agreement| !agreement.tied_at_the_break)
        .count();
    eprintln!(
        "{reordered} of {} positions reorder below the argmax",
        agreements.len()
    );

    let recorded = recorded_logits(&activations);
    let last = (ids.len() - 1) * vocab;
    let deviation = deviation(&logits[last..], &recorded[..vocab]);
    eprintln!("the last position's {vocab} logits: {deviation:e}");
    assert!(
        deviation <= MODEL_LOGIT_TOLERANCE,
        "deviation {deviation:e}"
    );
}

/// The checkpoint's own tokenizer, and the eos the config names for it.
fn tokenizer(dir: &Path) -> Tokenizer {
    Tokenizer::open(dir, &fixture::config(dir)).expect("the checkpoint's tokenizer opens")
}

/// The same `tokenizer.json` through the loader's own decode, which is what
/// this crate's assembled one is checked against. Held rather than reopened:
/// the file is 27 MB.
fn reference_tokenizer(dir: &Path) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json loads")
}

fn reference_decode(reference: &tokenizers::Tokenizer, ids: &[u32]) -> String {
    reference.decode(ids, false).expect("the loader decodes")
}

/// The text/id pair every other fixture was captured from. `dump_activations.py`
/// keeps the first eight of these ids, so this is also what says the committed
/// activations belong to this sentence.
#[test]
fn the_recorded_prompt_encodes_to_the_ids_the_fixtures_hold() {
    let Some(dir) = checkpoint_dir() else { return };
    let fixture = TokenizerFixture::load();
    let case = fixture.case("prompt");

    let ids = tokenizer(&dir).encode(&case.text).expect("encodes");
    assert_eq!(ids, case.ids);

    let captured = indices(&fixture::tensor(&fixture::open(ACTIVATIONS), "input_ids"));
    let prefix: Vec<usize> = ids[..captured.len()]
        .iter()
        .map(|&id| id as usize)
        .collect();
    assert_eq!(
        prefix, captured,
        "the activation dump tokenised something else"
    );
}

#[test]
fn every_recorded_case_decodes_to_the_text_the_reference_decoded() {
    let Some(dir) = checkpoint_dir() else { return };
    let tokenizer = tokenizer(&dir);

    for (name, case) in &TokenizerFixture::load().cases {
        assert_eq!(
            tokenizer.decode(&case.ids).expect("decodes"),
            case.text,
            "{name}"
        );
        if case.round_trips {
            assert_eq!(
                tokenizer.encode(&case.text).expect("encodes"),
                case.ids,
                "{name}"
            );
        }
    }
}

/// Assembling the text out of each token's bytes has to give what the loader's
/// own decode gives, for every piece in the vocabulary rather than for the few
/// a fixture can hold — a spelling this crate mapped wrongly would otherwise
/// surface only on whatever text happened to use it.
#[test]
fn decoding_matches_the_loaders_own_across_the_whole_vocabulary() {
    let Some(dir) = checkpoint_dir() else { return };
    let tokenizer = tokenizer(&dir);
    let reference = reference_tokenizer(&dir);
    let config = text_config(&dir);
    let filled = config.unpadded_vocab_size.expect("an unpadded vocab size") as u32;

    // In runs rather than one id at a time: a run is where a character spelled
    // across several pieces has to survive being reassembled.
    for start in (0..filled).step_by(64) {
        let ids: Vec<u32> = (start..(start + 64).min(filled)).collect();
        assert_eq!(
            tokenizer.decode(&ids).expect("decodes"),
            reference_decode(&reference, &ids),
            "ids {start}..{}",
            start + ids.len() as u32
        );
    }
}

/// The one thing the tokenizer's own files do not say. They name no eos at all —
/// every special token is listed under `additional_special_tokens` — so a port
/// that asked them either finds nothing or settles for `<|endoftext|>`, and
/// generation then runs until it hits a length cap.
#[test]
fn the_eos_id_comes_from_the_config_and_no_tokenizer_file_names_one() {
    let Some(dir) = checkpoint_dir() else { return };
    let tokenizer = tokenizer(&dir);
    let recorded = TokenizerFixture::load();

    assert_eq!(tokenizer.eos(), recorded.eos_token_id);
    assert_eq!(
        tokenizer.piece(tokenizer.eos()).as_deref(),
        Some(recorded.eos_token.as_str())
    );

    for file in ["tokenizer_config.json", "special_tokens_map.json"] {
        let text = std::fs::read_to_string(dir.join(file)).expect("the checkpoint carries it");
        let declared: serde_json::Value = serde_json::from_str(&text).expect("parses");
        assert!(
            declared.get("eos_token").is_none(),
            "{file} names an eos after all, so the config is no longer the only source"
        );
    }
    assert_eq!(tokenizer.id_of("<|endoftext|>"), Some(199999));
    assert_ne!(tokenizer.eos(), 199999, "the guess a missing eos invites");
}

#[test]
fn every_piece_in_the_filled_vocabulary_is_a_byte_level_spelling() {
    let Some(dir) = checkpoint_dir() else { return };
    let tokenizer = tokenizer(&dir);
    let config = text_config(&dir);
    let filled = config.unpadded_vocab_size.expect("an unpadded vocab size") as u32;

    for id in 0..filled {
        tokenizer.token_bytes(id).expect("a byte-level spelling");
    }
    // The rest of the 201024 the embedding is padded to hold no token at all.
    assert!(matches!(
        tokenizer.token_bytes(filled),
        Err(TokenizerError::UnknownToken(_))
    ));
}

/// The oracle's own continuation, decoded a token at a time as a generator
/// would surface it, against decoding the whole of it at once.
#[test]
fn streaming_the_oracles_continuation_matches_decoding_it_whole() {
    let Some(dir) = checkpoint_dir() else { return };
    let tokenizer = tokenizer(&dir);
    let activations = fixture::open(ACTIVATIONS);
    let ids: Vec<u32> = indices(&fixture::tensor(&activations, "greedy_continuation"))
        .iter()
        .map(|&id| id as u32)
        .collect();

    let mut stream = tokenizer.stream();
    let mut streamed = String::new();
    for &id in &ids {
        streamed.push_str(&stream.push(id).expect("decodes"));
    }
    streamed.push_str(&stream.finish());

    eprintln!("the oracle continued with {streamed:?}");
    assert_eq!(streamed, reference_decode(&reference_tokenizer(&dir), &ids));
}
