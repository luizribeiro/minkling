//! Test-only access to the committed reference fixtures, behind the
//! `test-support` feature: the paths here are relative to this crate's source
//! tree, so nothing outside a test can use them.
//!
//! Each fixture is a safetensors bundle under `reference/fixtures`, written by
//! a `just dump-*` recipe and read back through [`Checkpoint`]'s single-file
//! layout.

use std::path::PathBuf;

use crate::attention::AttentionWeights;
use crate::checkpoint::{Checkpoint, Dtype, TensorView};
use crate::layer::{DecoderWeights, Experts, LayerMlp, NoExperts};
use crate::moe::{ExpertBank, GateWeights, MoeConfig, SparseMoe};
use crate::ops::DenseMlp;

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

/// The decoder layers that pass kept, and so the layers every trained case is
/// cut from. Which three comes from the checkpoint — the dump script refuses a
/// set that does not cover both a dense and a MoE MLP, and both a sliding and a
/// global attention.
pub const CAPTURED_LAYERS: [usize; 3] = [0, 2, 5];

pub fn open(file: &str) -> Checkpoint {
    let path = PathBuf::from(DIR).join(file);
    Checkpoint::open(&path).unwrap_or_else(|err| panic!("{file} opens: {err}"))
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
                    gate_weight,
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
    fn routed(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
        match &self.mlp {
            Mlp::Dense { .. } => NoExperts.routed(expert, rows),
            Mlp::Sparse { routed, .. } => routed.expert(expert, rows),
        }
    }

    fn shared(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
        match &self.mlp {
            Mlp::Dense { .. } => NoExperts.shared(expert, rows),
            Mlp::Sparse { shared, .. } => shared.expert(expert, rows),
        }
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
