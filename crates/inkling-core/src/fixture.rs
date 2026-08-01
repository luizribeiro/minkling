//! Test-only access to the committed reference fixtures, behind the
//! `test-support` feature: the paths here are relative to this crate's source
//! tree, so nothing outside a test can use them.
//!
//! Each fixture is a safetensors bundle under `reference/fixtures`, written by
//! a `just dump-*` recipe and read back through [`Checkpoint`]'s single-file
//! layout.
//!
//! Two things here are not fixtures at all — [`config`] and [`resident_bytes`] —
//! and they are here for the same reason the fixtures are: three test binaries
//! in three crates need them, and a test binary cannot see another's helpers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::attention::{
    AttentionConfig, AttentionProjections, AttentionWeights, DecodedProjections,
};
use crate::checkpoint::{Checkpoint, Dtype, TensorView};
use crate::config::{Config, TextConfig};
use crate::layer::{DecoderCache, DecoderLayer, DecoderWeights, Experts, LayerMlp, NoExperts};
use crate::model::{Model, ModelWeights};
use crate::moe::{ExpertBank, Gate, GateWeights, Gathered, MoeConfig, SparseMoe};
use crate::ops::{DenseMlp, DenseProjection};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/fixtures");

/// The forward pass `just dump-activations` recorded: eight tokens through the
/// whole model, keeping every intermediate of the layers it captured.
pub const ACTIVATIONS: &str = "layer_activations.safetensors";

/// The same capture over a sequence long enough to reach past the relative-
/// position band, from `just dump-long-activations`.
///
/// Not committed. Its masks alone are `[1, 32, L, L]` — 210 MB a layer at the
/// 1280 tokens the recipe captures, against a bundle of a gibibyte — so
/// reproducing it matters more than holding it, and every test that reads it
/// skips when it is absent the way the `INKLINGRS_CHECKPOINT` tests do.
///
/// 1280 is the band plus a quarter of it. That puts 32896 query-key pairs per
/// head outside the band on a global layer, spread over 256 rows and 256
/// distances rather than heaped in one corner, and 295296 pairs past the window
/// on a sliding one. Everything about the capture is quadratic in that length,
/// so a longer one costs a fixture nobody can hold and buys margin no test
/// spends.
pub const LONG_ACTIVATIONS: &str = "long_activations.safetensors";

/// The MXFP4 slices `just dump-quant-fixture` cut out of the checkpoint, each
/// held both packed and as `mx.dequantize` decoded it.
///
/// The only committed bytes in the tree that are a real quantised weight, which
/// is why three crates' tests read them: the dequantiser is pinned against them,
/// and so are both of the projections that multiply against a packed tensor
/// without decoding it whole.
pub const MXFP4: &str = "mxfp4_dequant.safetensors";

/// mlx-vlm's answers to synthetic inputs for the two ops a dense layer is built
/// from, from `just dump-op-fixture`.
///
/// Read by both backends' RMSNorm, which is what puts its cases here rather than
/// in either of them: the kernel is checked against MLX itself, and a second
/// list of case names would be a second answer to which shapes the two have to
/// agree about.
pub const OPS: &str = "ops.safetensors";

/// The RMSNorm cases [`OPS`] carries: a wide row, one whose width is not a
/// multiple of eight, a batch of rows, one that is all zeros, and one large
/// enough to matter to an f32 accumulator.
pub const NORM_CASES: [&str; 5] = [
    "norm_wide",
    "norm_odd",
    "norm_batched",
    "norm_zero_row",
    "norm_large",
];

/// The `rms_norm_eps` the reference ran [`NORM_CASES`] under.
pub fn norm_eps(ckpt: &Checkpoint) -> f32 {
    f32s(&tensor(ckpt, "rms_norm_eps"))[0]
}

/// One case's input, weight and mlx-vlm's output for it.
pub fn norm_case(ckpt: &Checkpoint, case: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let of = |field| f32s(&tensor(ckpt, &format!("{case}.{field}")));
    (of("input"), of("weight"), of("output"))
}

/// The slice of that bundle which straddles the head's cut:
/// `lm_head[200026:200090]` against an `unpadded_vocab_size` of 200058, so the
/// first [`VOCAB_PADDING_ROWS`] rows are vocabulary and the rest are the
/// all-zero padding.
pub const VOCAB_PADDING: &str = "vocab_padding";

/// How many of [`VOCAB_PADDING`]'s 64 rows a head cut to the vocabulary keeps.
pub const VOCAB_PADDING_ROWS: usize = 32;

/// The decoder layers that pass kept, and so the layers every trained case is
/// cut from. Which three comes from the checkpoint — the dump script refuses a
/// set that does not cover both a dense and a MoE MLP, and both a sliding and a
/// global attention.
pub const CAPTURED_LAYERS: [usize; 3] = [0, 2, 5];

/// The `config.json` of a checkpoint directory, which is what every test gated
/// on `INKLINGRS_CHECKPOINT` starts from.
///
/// The checkpoint itself is 130.6 GiB and named by an environment variable, so
/// it is nothing like a fixture — but reading its config is the same three lines
/// in `inkling-core`, `inkling-cli` and `inkling-metal`, and the third copy
/// would mean a `serde_json` dependency for a crate that reads no json.
pub fn config(dir: &Path) -> Config {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{} reads: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{} parses: {err}", path.display()))
}

/// What this process holds resident, in bytes.
///
/// A bound on what a forward pass may hold is made of this reading, and the
/// reading has to come from outside the process: the packed weights are mapped,
/// so what a pass costs is how many of those pages it touched, and nothing this
/// side of `ps` knows that.
pub fn resident_bytes() -> u64 {
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

pub fn open(file: &str) -> Checkpoint {
    let path = PathBuf::from(DIR).join(file);
    Checkpoint::open(&path).unwrap_or_else(|err| panic!("{file} opens: {err}"))
}

/// A dump's text manifest, beside the bundle of the same name. The synthetic
/// stack's is the config it was built from, spelled the way a checkpoint spells
/// it, so that one file stands up both the reference and this port.
pub fn read(file: &str) -> String {
    let path = PathBuf::from(DIR).join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{file} reads: {err}"))
}

/// [`open`], for a bundle too large to commit: absent, it reports a skip and
/// the caller returns rather than failing.
pub fn try_open(file: &str) -> Option<Checkpoint> {
    if !PathBuf::from(DIR).join(file).exists() {
        eprintln!("skipping: {file} has not been generated");
        return None;
    }
    Some(open(file))
}

/// Whether a bundle recorded a given decoder layer, which the long capture
/// answers differently from the committed one.
pub fn holds_layer(ckpt: &Checkpoint, layer: usize) -> bool {
    let prefix = format!("layer{layer}.");
    ckpt.tensor_names().any(|name| name.starts_with(&prefix))
}

pub fn tensor<'a>(ckpt: &'a Checkpoint, name: &str) -> TensorView<'a> {
    ckpt.tensor(name)
        .unwrap_or_else(|err| panic!("fixture holds {name}: {err}"))
}

/// A tensor a dump recorded per decoder layer, which every bundle names
/// `layer{layer}.{name}`.
pub fn layer_tensor<'a>(ckpt: &'a Checkpoint, layer: usize, name: &str) -> TensorView<'a> {
    tensor(ckpt, &format!("layer{layer}.{name}"))
}

/// A fixture tensor's values. Every dump casts to float32 before saving, so a
/// comparison never has to reason about the reference's dtype choices, and
/// anything else in a fixture is a dump that stopped doing that.
pub fn f32s(view: &TensorView<'_>) -> Vec<f32> {
    assert_eq!(view.dtype(), Dtype::F32);
    view.to_f32().expect("float32 widens")
}

/// An index tensor's values. Every dump casts integers to int32 before saving,
/// for the same reason it casts floats to float32.
pub fn indices(view: &TensorView<'_>) -> Vec<usize> {
    assert_eq!(view.dtype(), Dtype::I32);
    view.data()
        .chunks_exact(size_of::<i32>())
        .map(|b| i32::from_le_bytes(b.try_into().expect("chunked into ints")) as usize)
        .collect()
}

/// One `SwitchGLU`'s three projections, owned so the borrowed [`ExpertBank`]
/// can be handed out repeatedly.
///
/// Every bundle names them `{prefix}.{gate,up,down}_proj.weight`. A synthetic
/// bank is held whole, which the checkpoint's 25 GB of routed experts cannot be
/// — those are decoded per expert per call instead.
pub struct Bank {
    experts: usize,
    dim: usize,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl Bank {
    pub fn load(ckpt: &Checkpoint, prefix: &str, experts: usize, dim: usize) -> Self {
        let of = |name: &str| f32s(&tensor(ckpt, &format!("{prefix}.{name}.weight")));
        Self {
            experts,
            dim,
            gate_proj: of("gate_proj"),
            up_proj: of("up_proj"),
            down_proj: of("down_proj"),
        }
    }

    /// `[rows, dim]` through one of its experts.
    pub fn expert(&self, index: usize, rows: &[f32]) -> Vec<f32> {
        ExpertBank::new(
            self.experts,
            self.dim,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        )
        .expert(index)
        .forward(rows)
    }

    /// Every row a routing sent here, each through the expert it named.
    pub fn gathered(&self, gathered: Gathered<'_>) -> Vec<f32> {
        gathered
            .batches()
            .flat_map(|(expert, rows)| self.expert(expert, rows))
            .collect()
    }
}

/// One synthetic decoder layer's tensors, owned so the borrowed
/// [`DecoderWeights`] and [`LayerMlp`] can be handed out repeatedly.
///
/// Every dump names a layer's tensors `{prefix}.{module}` after the module that
/// holds them, so one reader serves a bundle of standalone layers and a bundle
/// of a whole stack. The expert banks are held whole, which sixteen experts of
/// width sixteen allow and Inkling's 25 GB do not.
pub struct LayerTensors {
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
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
    mlp: Mlp,
}

/// Whichever MLP a layer index called for, owned the same way.
enum Mlp {
    Dense {
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        global_scale: f32,
    },
    Sparse {
        config: MoeConfig,
        gate_weight: Vec<f32>,
        correction_bias: Vec<f32>,
        global_scale: f32,
        routed: Bank,
        shared: Bank,
    },
}

impl LayerTensors {
    pub fn load(ckpt: &Checkpoint, prefix: &str) -> Self {
        let of = |name: &str| f32s(&tensor(ckpt, &format!("{prefix}.{name}")));
        let input_layernorm = of("input_layernorm.weight");
        Self {
            q_proj: of("self_attn.q_proj.weight"),
            k_proj: of("self_attn.k_proj.weight"),
            v_proj: of("self_attn.v_proj.weight"),
            r_proj: of("self_attn.r_proj.weight"),
            o_proj: of("self_attn.o_proj.weight"),
            q_norm: of("self_attn.q_norm.weight"),
            k_norm: of("self_attn.k_norm.weight"),
            k_sconv: of("self_attn.k_sconv.conv.weight"),
            v_sconv: of("self_attn.v_sconv.conv.weight"),
            rel_proj: of("self_attn.rel_proj"),
            post_attention_layernorm: of("post_attention_layernorm.weight"),
            attn_sconv: of("attn_sconv.conv.weight"),
            mlp_sconv: of("mlp_sconv.conv.weight"),
            mlp: Mlp::load(ckpt, prefix, input_layernorm.len()),
            input_layernorm,
        }
    }

    pub fn view(&self) -> DecoderWeights<'_> {
        DecoderWeights {
            attention: AttentionWeights {
                projections: AttentionProjections::decoded(
                    self.hidden(),
                    DecodedProjections {
                        q_proj: &self.q_proj,
                        k_proj: &self.k_proj,
                        v_proj: &self.v_proj,
                        r_proj: &self.r_proj,
                        o_proj: &self.o_proj,
                    },
                ),
                q_norm: &self.q_norm,
                k_norm: &self.k_norm,
                k_sconv: &self.k_sconv,
                v_sconv: &self.v_sconv,
                rel_proj: &self.rel_proj,
            },
            input_layernorm: &self.input_layernorm,
            post_attention_layernorm: &self.post_attention_layernorm,
            attn_sconv: &self.attn_sconv,
            mlp_sconv: &self.mlp_sconv,
        }
    }

    pub fn hidden(&self) -> usize {
        self.input_layernorm.len()
    }

    /// `sconv_kernel_size`, which every one of the layer's four convolutions
    /// shares and which the residual-path pair's kernels state directly.
    pub fn kernel_size(&self) -> usize {
        self.attn_sconv.len() / self.hidden()
    }

    pub fn is_dense(&self) -> bool {
        matches!(self.mlp, Mlp::Dense { .. })
    }

    pub fn mlp(&self) -> LayerMlp<'_> {
        let hidden = self.hidden();
        match &self.mlp {
            Mlp::Dense {
                gate_proj,
                up_proj,
                down_proj,
                global_scale,
            } => LayerMlp::Dense(DenseMlp::new(
                hidden,
                gate_proj,
                up_proj,
                down_proj,
                *global_scale,
            )),
            Mlp::Sparse {
                config,
                gate_weight,
                correction_bias,
                global_scale,
                ..
            } => LayerMlp::Sparse(SparseMoe::new(
                *config,
                GateWeights {
                    gate: Gate::Widened(gate_weight),
                    correction_bias,
                    global_scale: *global_scale,
                },
            )),
        }
    }
}

impl Mlp {
    /// A layer with a router records its `[n_routed, n_shared, top_k,
    /// route_scale]` and a dense layer records none, which is the dump saying
    /// what `MoeConfig::for_layer` says by returning `None`.
    fn load(ckpt: &Checkpoint, prefix: &str, hidden: usize) -> Self {
        let of = |name: &str| f32s(&tensor(ckpt, &format!("{prefix}.mlp.{name}")));
        let global_scale = of("global_scale")[0];

        let moe_config = format!("{prefix}.moe_config");
        if !ckpt.tensor_names().any(|name| name == moe_config) {
            return Self::Dense {
                gate_proj: of("gate_proj.weight"),
                up_proj: of("up_proj.weight"),
                down_proj: of("down_proj.weight"),
                global_scale,
            };
        }

        let recorded = f32s(&tensor(ckpt, &moe_config));
        let &[n_routed, n_shared, top_k, route_scale] = recorded.as_slice() else {
            panic!("{prefix}: moe_config carries four scalars, got {recorded:?}")
        };
        let config = MoeConfig {
            n_routed: n_routed as usize,
            n_shared: n_shared as usize,
            top_k: top_k as usize,
            route_scale,
        };
        let bank = |module: &str, experts| {
            Bank::load(ckpt, &format!("{prefix}.mlp.{module}"), experts, hidden)
        };
        Self::Sparse {
            gate_weight: of("gate_weight"),
            correction_bias: of("e_score_correction_bias"),
            global_scale,
            routed: bank("switch_mlp", config.n_routed),
            shared: bank("shared_experts", config.n_shared),
            config,
        }
    }
}

/// The banks a layer carries, which is what makes one reader serve both MLPs: a
/// dense layer asks for nothing, and asking anyway is the panic [`NoExperts`]
/// raises.
impl Experts for LayerTensors {
    fn routed(&self, gathered: Gathered<'_>) -> Vec<f32> {
        match &self.mlp {
            Mlp::Dense { .. } => NoExperts.routed(gathered),
            Mlp::Sparse { routed, .. } => routed.gathered(gathered),
        }
    }

    fn shared(&self, gathered: Gathered<'_>) -> Vec<f32> {
        match &self.mlp {
            Mlp::Dense { .. } => NoExperts.shared(gathered),
            Mlp::Sparse { shared, .. } => shared.gathered(gathered),
        }
    }
}

/// A five-layer synthetic model and the two calls mlx-vlm drove it with, from
/// `just dump-stack-fixture`.
pub const STACK: &str = "stack.safetensors";

/// The config that model was built from, in the checkpoint's own spelling so
/// that the same JSON stands the reference and this port up.
pub const STACK_CONFIG: &str = "stack.json";

/// The synthetic stack's weights, held whole, as the [`ModelWeights`] a model
/// runs against — five layers of width 32, which is what makes a stack testable
/// without the 131 GB checkpoint.
///
/// The layer this builds per call is built from the *config*, not from anything
/// the fixture recorded per layer: which attention config and which MLP each
/// index gets is exactly what a stack has to get right.
///
/// Shared rather than kept beside the stack's own tests because everything
/// downstream of a stack — generation, and the caches it carries between steps —
/// needs a whole model to have anything to say, and the only whole model that
/// fits in a test is this one.
pub struct Stack {
    pub config: TextConfig,
    /// The prompt the reference prefilled, and the continuation it fed through
    /// the caches that prefill left behind.
    pub ids: Vec<usize>,
    pub continue_ids: Vec<usize>,
    /// The two norms outside the layers, which [`Stack::model`] borrows and
    /// which a test that refuses a malformed model hands in itself.
    pub embed_norm: Vec<f32>,
    pub norm: Vec<f32>,
    table: Vec<f32>,
    layers: Vec<LayerTensors>,
    order: Vec<usize>,
}

impl Stack {
    pub fn load() -> Self {
        let ckpt = open(STACK);
        let config = serde_json::from_str::<Config>(&read(STACK_CONFIG))
            .expect("the recorded config parses")
            .text_config;
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|layer| LayerTensors::load(&ckpt, &format!("layers.{layer}")))
                .collect(),
            order: (0..config.num_hidden_layers).collect(),
            embed_norm: f32s(&tensor(&ckpt, "embed_norm.weight")),
            norm: f32s(&tensor(&ckpt, "norm.weight")),
            table: f32s(&tensor(&ckpt, "embed_tokens.weight")),
            ids: indices(&tensor(&ckpt, "input_ids")),
            continue_ids: indices(&tensor(&ckpt, "continue_ids")),
            config,
        }
    }

    pub fn model(&self) -> Model<'_> {
        Model::new(&self.config, Some(&self.embed_norm), &self.norm)
    }

    /// The head's weights, which for a stack that carries no `lm_head` are its
    /// embedding table read as a linear — the tie `tie_word_embeddings`
    /// describes, over the same `[vocab, hidden]` rows the lookup returns.
    ///
    /// Held decoded, which is what makes a fixture a fixture: the checkpoint's
    /// own head is 3.3 GB and is reached through
    /// [`PackedRows`](crate::weights::PackedRows) instead.
    pub fn head(&self) -> DenseProjection<'_> {
        DenseProjection::new(self.config.hidden_size, &self.table)
    }

    pub fn layers(&self) -> &[LayerTensors] {
        &self.layers
    }

    /// Everything the reference drove this stack with, in order: the prompt and
    /// then the continuation.
    pub fn sequence(&self) -> Vec<usize> {
        [self.ids.clone(), self.continue_ids.clone()].concat()
    }

    /// The same weights with two layers' tensors exchanged, which is the
    /// mutation a stack that ran its layers out of order would make.
    pub fn exchanging(mut self, a: usize, b: usize) -> Self {
        self.order.swap(a, b);
        self
    }
}

impl ModelWeights for Stack {
    fn embedding_row(&self, id: usize) -> Vec<f32> {
        let hidden = self.config.hidden_size;
        self.table[id * hidden..][..hidden].to_vec()
    }

    fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: &[f32]) -> Vec<f32> {
        let tensors = &self.layers[self.order[index]];
        let config = AttentionConfig::for_layer(&self.config, index);
        DecoderLayer::new(config, tensors.view(), tensors.mlp()).forward(cache, x, tensors)
    }
}

/// The text/id pairs `just dump-tokenizer-fixture` recorded, from
/// [`TokenizerFixture::load`].
pub const TOKENIZER_CASES: &str = "tokenizer_cases.json";

/// What the checkpoint's tokenizer did with a handful of cases, so that the
/// 27 MB `tokenizer.json` is needed only by the checkpoint-gated tests.
#[derive(Debug, Deserialize)]
pub struct TokenizerFixture {
    /// The eos the *config* named, which is the only file that names one.
    pub eos_token_id: u32,
    pub eos_token: String,
    pub cases: BTreeMap<String, TokenizerCase>,
}

/// One sequence of ids, and everything the reference made of it.
#[derive(Debug, Deserialize)]
pub struct TokenizerCase {
    pub ids: Vec<u32>,
    pub text: String,
    /// The vocabulary pieces those ids name, which is what lets a test
    /// reconstruct each token's bytes without the vocabulary itself.
    pub pieces: Vec<String>,
    /// What the reference's streaming detokenizer surfaced as each token
    /// arrived, and then what its flush left over — one longer than `ids`.
    pub segments: Vec<String>,
    /// Whether `text` encodes back to `ids`. A case assembled from pieces
    /// splits characters no encoder would split, and says so.
    pub round_trips: bool,
}

impl TokenizerFixture {
    pub fn load() -> Self {
        serde_json::from_str(&read(TOKENIZER_CASES)).expect("the recorded cases parse")
    }

    pub fn case(&self, name: &str) -> &TokenizerCase {
        self.cases
            .get(name)
            .unwrap_or_else(|| panic!("{TOKENIZER_CASES} holds a {name} case"))
    }
}

/// The worst absolute disagreement with a reference tensor, as a fraction of
/// that tensor's largest value.
///
/// Scaled by the tensor rather than element by element: an output that lands
/// near zero by cancellation has no meaningful relative error of its own, and
/// every op here reduces over an axis where cancellation is ordinary.
pub fn deviation(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(0.0f32, |worst, w| worst.max(w.abs()));
    assert!(scale > 0.0, "reference tensor is all zeros");
    got.iter()
        .zip(want)
        .fold(0.0f32, |worst, (got, want)| worst.max((got - want).abs()))
        / scale
}
