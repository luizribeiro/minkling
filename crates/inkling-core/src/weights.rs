//! The checkpoint as a [`ModelWeights`]: forty-two layers stood up one at a
//! time, out of bytes that stay packed.
//!
//! This is where [`crate::model`]'s residency decision is actually made. Nothing
//! here holds a decoded weight between calls. A layer's projections are decoded
//! into a [`Scratch`] the whole pass shares, the layer is built around those
//! runs, run, and dropped; the next layer overwrites them. The routed experts go
//! further and are not decoded at all unless a token chose them.
//!
//! What that bounds is stated up front rather than observed: the scratch is
//! sized by [`CheckpointWeights::scratch_floats`] before the pass starts, from
//! the config alone, and [`Scratch`] panics rather than growing. At
//! Inkling-Small's shapes it comes to 981 MB for a layer and 100 MB for an
//! expert, and the dense layers set the first of those — a dense FFN is
//! `3 x [16384, 4096]`, four and a half times the five attention projections
//! together.
//!
//! Resident memory is *not* only that. The packed bytes are mapped, and a page
//! joins the resident set when it is first touched, so a forward pass drags in
//! whatever it reads: every layer's projections, and one expert's bytes per
//! expert a token routed to. What it never drags in is the rest of the 130.6 GiB
//! — 250 of every layer's 256 experts, on an eight-token pass — which is the
//! difference between a bounded run and an unbounded one.
//!
//! A malformed checkpoint is a panic here rather than an error. Every name and
//! shape this reads is fixed by the architecture, so a tensor that is missing at
//! layer 17 is not a condition a caller can do anything about, and threading a
//! `Result` through forty-two layers of a forward pass would say otherwise.
//! What [`CheckpointWeights::open`] can check before the pass starts, it does.

use std::cell::RefCell;

use crate::attention::{AttentionConfig, AttentionWeights};
use crate::checkpoint::{Checkpoint, CheckpointError, Dtype, TensorView};
use crate::config::TextConfig;
use crate::layer::{DecoderCache, DecoderLayer, DecoderWeights, Experts, LayerMlp, NoExperts};
use crate::model::{Model, ModelWeights};
use crate::moe::{ExpertBank, GateWeights, MoeConfig, SparseMoe};
use crate::ops::DenseMlp;
use crate::quant::{BITS, QuantError, Scratch, dequantize_blocks_into};

/// Where the language model's tensors live in a multimodal checkpoint.
const MODEL: &str = "language_model.model";

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    #[error("{name} is {dtype:?}, which is not a float dtype")]
    NotFloat { name: String, dtype: Dtype },

    #[error("{name} holds {got} values, not {expected}")]
    WrongLength {
        name: String,
        expected: usize,
        got: usize,
    },
}

/// A tensor the checkpoint stores packed: `{name}.weight`, a `U32` of MXFP4
/// codes, beside `{name}.scales`, a `U8` of one block scale per 32 of them.
///
/// Opening decodes nothing. A caller asks either for the whole tensor — a
/// projection, which every token touches all of — or for one slice of its
/// leading axis: an expert of a bank, a row of the embedding table.
#[derive(Debug, Clone, Copy)]
pub struct Packed<'a> {
    weight: TensorView<'a>,
    scales: TensorView<'a>,
}

impl<'a> Packed<'a> {
    pub fn open(ckpt: &'a Checkpoint, name: &str) -> Result<Self, CheckpointError> {
        Ok(Self {
            weight: ckpt.tensor(&format!("{name}.weight"))?,
            scales: ckpt.tensor(&format!("{name}.scales"))?,
        })
    }

    /// How many values the whole tensor decodes to.
    pub fn len(&self) -> usize {
        self.weight.data().len() * (u8::BITS as usize / BITS)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The length of the leading axis, which is what a slice indexes: a bank's
    /// experts, or the embedding table's rows.
    pub fn slices(&self) -> usize {
        *self
            .weight
            .shape()
            .first()
            .expect("a packed tensor has a shape")
    }

    /// How many values one slice decodes to.
    pub fn slice_len(&self) -> usize {
        self.len() / self.slices()
    }

    /// Every value, decoded into `out`.
    pub fn decode_into(&self, out: &mut [f32]) -> Result<(), QuantError> {
        dequantize_blocks_into(self.weight.data(), self.scales.data(), out)
    }

    /// Slice `index` of the leading axis, decoded into `out`.
    pub fn decode_slice_into(&self, index: usize, out: &mut [f32]) -> Result<(), QuantError> {
        let stride = |view: &TensorView<'_>| view.data().len() / self.slices();
        let (codes, scales) = (stride(&self.weight), stride(&self.scales));
        dequantize_blocks_into(
            &self.weight.data()[index * codes..][..codes],
            &self.scales.data()[index * scales..][..scales],
            out,
        )
    }

    /// Slice `index`, into a fresh vector, for a caller that wants one row and
    /// keeps it — which the embedding lookup does and no matmul does.
    pub fn decode_slice(&self, index: usize) -> Result<Vec<f32>, QuantError> {
        let mut values = vec![0.0; self.slice_len()];
        self.decode_slice_into(index, &mut values)?;
        Ok(values)
    }
}

/// One `SwitchGLU`'s three packed banks, which are a MoE layer's routed 256 or
/// its shared 2.
#[derive(Debug, Clone, Copy)]
pub struct PackedExperts<'a> {
    gate_proj: Packed<'a>,
    up_proj: Packed<'a>,
    down_proj: Packed<'a>,
}

impl<'a> PackedExperts<'a> {
    pub fn open(ckpt: &'a Checkpoint, module: &str) -> Result<Self, CheckpointError> {
        let of = |projection: &str| Packed::open(ckpt, &format!("{module}.{projection}"));
        Ok(Self {
            gate_proj: of("gate_proj")?,
            up_proj: of("up_proj")?,
            down_proj: of("down_proj")?,
        })
    }

    pub fn experts(&self) -> usize {
        self.gate_proj.slices()
    }

    /// How many values one expert's three projections decode to, which is what
    /// running a single expert costs.
    pub fn expert_floats(&self) -> usize {
        self.gate_proj.slice_len() + self.up_proj.slice_len() + self.down_proj.slice_len()
    }

    /// `[rows, dim]` through one expert, decoded into `scratch` and dropped with
    /// it. A bank of one is the same bank, which is what lets [`ExpertBank`]
    /// serve a bank decoded an expert at a time.
    pub fn forward_into<'s>(
        &self,
        expert: usize,
        dim: usize,
        rows: &[f32],
        scratch: &mut Scratch<'s>,
    ) -> Vec<f32> {
        let gate = decode_expert(&self.gate_proj, expert, "gate_proj", scratch);
        let up = decode_expert(&self.up_proj, expert, "up_proj", scratch);
        let down = decode_expert(&self.down_proj, expert, "down_proj", scratch);
        ExpertBank::new(1, dim, gate, up, down)
            .expert(0)
            .forward(rows)
    }
}

fn decode_expert<'s>(
    packed: &Packed<'_>,
    expert: usize,
    what: &str,
    scratch: &mut Scratch<'s>,
) -> &'s [f32] {
    let run = scratch.take(packed.slice_len());
    packed
        .decode_slice_into(expert, run)
        .unwrap_or_else(|err| panic!("expert {expert}'s {what} decodes: {err}"));
    run
}

/// The whole model's weights, out of a checkpoint, decoded only where a call
/// reaches them.
pub struct CheckpointWeights<'a> {
    ckpt: &'a Checkpoint,
    config: &'a TextConfig,
    embed_tokens: Packed<'a>,
    embed_norm: Option<Vec<f32>>,
    norm: Vec<f32>,
    layer_scratch: RefCell<Vec<f32>>,
    expert_scratch: RefCell<Vec<f32>>,
}

impl<'a> CheckpointWeights<'a> {
    /// Map the checkpoint's weights and allocate the pass's scratch.
    ///
    /// The two stack-level norms are widened and held — 16 KB each — because
    /// [`Model`] borrows them for as long as it exists. Everything else is left
    /// where it is.
    pub fn open(ckpt: &'a Checkpoint, config: &'a TextConfig) -> Result<Self, WeightsError> {
        let norm = widened(ckpt, &format!("{MODEL}.norm.weight"))?;
        expect_len(&norm, config.hidden_size, "the final norm")?;

        let embed_norm = config
            .use_embed_norm
            .then(|| widened(ckpt, &format!("{MODEL}.embed_norm.weight")))
            .transpose()?;
        if let Some(weight) = &embed_norm {
            expect_len(weight, config.hidden_size, "embed_norm")?;
        }

        Ok(Self {
            embed_tokens: Packed::open(ckpt, &format!("{MODEL}.embed_tokens"))?,
            layer_scratch: RefCell::new(vec![0.0; layer_scratch_floats(config)]),
            expert_scratch: RefCell::new(vec![0.0; expert_scratch_floats(config)]),
            embed_norm,
            norm,
            ckpt,
            config,
        })
    }

    /// The model around the layers, borrowing the two norms this holds.
    pub fn model(&self) -> Model<'_> {
        Model::new(self.config, self.embed_norm.as_deref(), &self.norm)
    }

    /// How many float32 values the pass's scratch holds, which is the whole of
    /// what this decodes into and so the bound on what it decodes at once.
    ///
    /// The layer buffer is sized by the widest layer and the expert buffer by
    /// one expert; both are allocated at [`CheckpointWeights::open`] and neither
    /// grows.
    pub fn scratch_floats(&self) -> usize {
        self.layer_scratch.borrow().len() + self.expert_scratch.borrow().len()
    }

    fn widened(&self, name: &str) -> Vec<f32> {
        widened(self.ckpt, name).unwrap_or_else(|err| panic!("{err}"))
    }

    /// One packed tensor of this checkpoint, decoded whole into `scratch` and
    /// valid for as long as the run it was given.
    fn decoded<'s>(&self, name: &str, scratch: &mut Scratch<'s>) -> &'s [f32] {
        let packed = Packed::open(self.ckpt, name).unwrap_or_else(|err| panic!("{err}"));
        let run = scratch.take(packed.len());
        packed
            .decode_into(run)
            .unwrap_or_else(|err| panic!("{name} decodes: {err}"));
        run
    }

    fn attention<'s>(&self, module: &str, scratch: &mut Scratch<'s>) -> Attention<'s> {
        let widened = |name: &str| self.widened(&format!("{module}.self_attn.{name}"));
        Attention {
            q_norm: widened("q_norm.weight"),
            k_norm: widened("k_norm.weight"),
            k_sconv: widened("k_sconv.conv.weight"),
            v_sconv: widened("v_sconv.conv.weight"),
            rel_proj: widened("rel_proj"),
            projections: ["q_proj", "k_proj", "v_proj", "r_proj", "o_proj"]
                .map(|name| self.decoded(&format!("{module}.self_attn.{name}"), scratch)),
        }
    }

    /// Whichever MLP the layer index called for, and the experts it can reach.
    ///
    /// A dense layer's three projections are decoded into the scratch beside its
    /// attention's; a MoE layer decodes only its gate, which is bfloat16 and
    /// unpacked, and leaves both banks alone until a token routes into them.
    fn mlp<'s>(&'s self, layer: usize, module: &str, scratch: &mut Scratch<'s>) -> Mlp<'s> {
        let widened = |name: &str| self.widened(&format!("{module}.mlp.{name}"));
        let global_scale = widened("global_scale")[0];

        let Some(config) = MoeConfig::for_layer(self.config, layer) else {
            return Mlp::Dense {
                projections: ["gate_proj", "up_proj", "down_proj"]
                    .map(|name| self.decoded(&format!("{module}.mlp.{name}"), scratch)),
                global_scale,
            };
        };

        let bank = |name: &str| {
            PackedExperts::open(self.ckpt, &format!("{module}.mlp.{name}"))
                .unwrap_or_else(|err| panic!("layer {layer}: {err}"))
        };
        Mlp::Sparse(Box::new(Sparse {
            config,
            hidden: self.config.hidden_size,
            gate_weight: widened("gate_weight"),
            correction_bias: widened("e_score_correction_bias"),
            global_scale,
            routed: bank("switch_mlp"),
            shared: bank("shared_experts"),
            scratch: &self.expert_scratch,
        }))
    }
}

impl ModelWeights for CheckpointWeights<'_> {
    fn embedding_row(&self, id: usize) -> Vec<f32> {
        self.embed_tokens
            .decode_slice(id)
            .unwrap_or_else(|err| panic!("embedding row {id} decodes: {err}"))
    }

    fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: &[f32]) -> Vec<f32> {
        let mut buffer = self.layer_scratch.borrow_mut();
        let mut scratch = Scratch::new(&mut buffer);
        let module = format!("{MODEL}.layers.{index}");

        let attention = self.attention(&module, &mut scratch);
        let mlp = self.mlp(index, &module, &mut scratch);
        let [
            input_layernorm,
            post_attention_layernorm,
            attn_sconv,
            mlp_sconv,
        ] = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "attn_sconv.conv.weight",
            "mlp_sconv.conv.weight",
        ]
        .map(|name| self.widened(&format!("{module}.{name}")));

        let weights = DecoderWeights {
            attention: attention.view(),
            input_layernorm: &input_layernorm,
            post_attention_layernorm: &post_attention_layernorm,
            attn_sconv: &attn_sconv,
            mlp_sconv: &mlp_sconv,
        };
        let config = AttentionConfig::for_layer(self.config, index);
        DecoderLayer::new(config, weights, mlp.view(self.config.hidden_size))
            .forward(cache, x, &mlp)
    }
}

/// One layer's attention tensors: the five projections decoded into the pass's
/// scratch, and the small bfloat16 ones widened into vectors of their own.
struct Attention<'a> {
    projections: [&'a [f32]; 5],
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
    rel_proj: Vec<f32>,
}

impl Attention<'_> {
    fn view(&self) -> AttentionWeights<'_> {
        let [q_proj, k_proj, v_proj, r_proj, o_proj] = self.projections;
        AttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            r_proj,
            o_proj,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            k_sconv: &self.k_sconv,
            v_sconv: &self.v_sconv,
            rel_proj: &self.rel_proj,
        }
    }
}

/// Whichever MLP a layer index called for. A dense layer is three runs of the
/// scratch and a scale; a MoE layer is its gate, and the two banks its router
/// can reach but has not.
enum Mlp<'a> {
    Dense {
        projections: [&'a [f32]; 3],
        global_scale: f32,
    },
    Sparse(Box<Sparse<'a>>),
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
    scratch: &'a RefCell<Vec<f32>>,
}

impl Mlp<'_> {
    fn view(&self, hidden: usize) -> LayerMlp<'_> {
        match self {
            Self::Dense {
                projections: [gate_proj, up_proj, down_proj],
                global_scale,
            } => LayerMlp::Dense(DenseMlp::new(
                hidden,
                gate_proj,
                up_proj,
                down_proj,
                *global_scale,
            )),
            Self::Sparse(moe) => LayerMlp::Sparse(SparseMoe::new(
                moe.config,
                GateWeights {
                    gate_weight: &moe.gate_weight,
                    correction_bias: &moe.correction_bias,
                    global_scale: moe.global_scale,
                },
            )),
        }
    }
}

/// The bargain the whole module is built on: an expert is decoded when a token
/// routes to it and dropped again, into a buffer that outlives neither. A dense
/// layer never asks, and asking anyway is the panic [`NoExperts`] raises.
impl Experts for Mlp<'_> {
    fn routed(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense { .. } => NoExperts.routed(expert, rows),
            Self::Sparse(moe) => through(&moe.routed, moe.hidden, expert, rows, moe.scratch),
        }
    }

    fn shared(&self, expert: usize, rows: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense { .. } => NoExperts.shared(expert, rows),
            Self::Sparse(moe) => through(&moe.shared, moe.hidden, expert, rows, moe.scratch),
        }
    }
}

/// One expert of one bank, over the rows that chose it. The expert's three
/// projections are decoded into the pass's expert buffer, which the next expert
/// overwrites — so a layer whose eight experts ran costs one expert's float32,
/// not eight.
fn through(
    bank: &PackedExperts<'_>,
    hidden: usize,
    expert: usize,
    rows: &[f32],
    scratch: &RefCell<Vec<f32>>,
) -> Vec<f32> {
    let mut buffer = scratch.borrow_mut();
    bank.forward_into(expert, hidden, rows, &mut Scratch::new(&mut buffer))
}

/// A `BF16` or `F32` tensor's values.
fn widened(ckpt: &Checkpoint, name: &str) -> Result<Vec<f32>, WeightsError> {
    let view = ckpt.tensor(name)?;
    view.to_f32().ok_or_else(|| WeightsError::NotFloat {
        name: name.to_string(),
        dtype: view.dtype(),
    })
}

fn expect_len(values: &[f32], expected: usize, name: &str) -> Result<(), WeightsError> {
    if values.len() == expected {
        return Ok(());
    }
    Err(WeightsError::WrongLength {
        name: name.to_string(),
        expected,
        got: values.len(),
    })
}

/// How many float32 values the widest layer decodes into at once: its five
/// attention projections, and a dense layer's three FFN projections beside them.
///
/// A MoE layer's banks are not here, and that is the decision: 256 experts of
/// float32 would be 25 GB a layer, and what a call decodes instead is the six a
/// token chose, one at a time, into [`expert_scratch_floats`].
pub fn layer_scratch_floats(config: &TextConfig) -> usize {
    (0..config.num_hidden_layers)
        .map(|layer| {
            let attention = AttentionConfig::for_layer(config, layer);
            let projections = 2 * attention.heads * attention.head_dim
                + 2 * attention.kv_channels()
                + attention.heads * attention.d_rel;
            let dense = if config.layer_is_dense(layer) {
                3 * config.dense_intermediate_size
            } else {
                0
            };
            (projections + dense) * config.hidden_size
        })
        .max()
        .unwrap_or(0)
}

/// How many float32 values one expert's three projections decode into.
pub fn expert_scratch_floats(config: &TextConfig) -> usize {
    3 * config.moe_intermediate_size * config.hidden_size
}
