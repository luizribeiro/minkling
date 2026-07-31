//! Assertions against a real Inkling-Small checkpoint, which is far too large
//! to commit. Set `INKLINGRS_CHECKPOINT` to a checkpoint directory to run them;
//! unset, each test reports a skip and passes.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use inkling_core::attention::{Attention, AttentionConfig, AttentionWeights};
use inkling_core::embed::Embed;
use inkling_core::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, deviation, indices};
use inkling_core::layer::{
    DecoderCache, DecoderLayer, DecoderWeights, Experts, LayerMlp, NoExperts,
};
use inkling_core::model::{ModelCache, ModelWeights};
use inkling_core::moe::{GateWeights, MoeConfig, SparseMoe};
use inkling_core::ops::DenseMlp;
use inkling_core::quant::{Scratch, dequantize};
use inkling_core::weights::{
    CheckpointWeights, Packed, PackedExperts, expert_scratch_floats, layer_scratch_floats,
};
use inkling_core::{Checkpoint, Config, Dtype, TensorView, TextConfig};

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

fn resident_bytes() -> u64 {
    let pid = std::process::id().to_string();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("ps runs");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .expect("ps reports rss in KiB")
        * 1024
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

    let before = resident_bytes();
    let started = Instant::now();
    let ckpt = Checkpoint::open(&dir).expect("checkpoint opens");
    let elapsed = started.elapsed();
    let after = resident_bytes();
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
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).expect("checkpoint carries a config.json");
    serde_json::from_str::<Config>(&text)
        .expect("config.json parses")
        .text_config
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
            peak: Cell::new(resident_bytes()),
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
        self.peak.set(self.peak.get().max(resident_bytes()));
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

    let before = resident_bytes();
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
