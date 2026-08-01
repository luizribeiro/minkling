//! The checkpoint as a [`ModelWeights`]: forty-two layers stood up one at a
//! time, out of bytes that stay packed.
//!
//! This is where [`crate::model`]'s residency decision is actually made. Nothing
//! here holds a decoded weight between calls. A layer's projections are decoded
//! into a [`Scratch`] the whole pass shares, the layer is built around those
//! runs, run, and dropped; the next layer overwrites them — or they are not
//! decoded at all, which is what handing them to a backend does and what
//! [`CheckpointWeights::with_projections`] is. The routed experts go further
//! still and are not decoded even here unless a token chose them.
//!
//! **A bfloat16 tensor is the exception, and it is held.** The bargain above is
//! about weights that are packed, and a norm has no packed form to be left in —
//! so the only question it poses is whether to widen it again on every step,
//! which [`Widened`] answers once at construction. That is 179 MB for the whole
//! stack, against the 1.1 TB the same bargain declines to hold.
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
//! The two tables at the ends of the model are the same bargain read twice.
//! `embed_tokens` and `lm_head` are both `[201024, 4096]` — 3.3 GB of float32
//! each — and both are reached a row at a time, the embedding for the rows its
//! tokens asked for and the head for the rows the vocabulary runs to. What
//! separates them is how many: a pass over eight tokens decodes eight rows of
//! the one and 200058 of the other, so the head is the only place where
//! per-row decoding costs the whole tensor. It still never holds it — and being
//! the one weight every token touches every row of is what makes it the first to
//! hand to a backend that does not decode at all, which
//! [`CheckpointWeights::with_head`] is.
//!
//! A malformed checkpoint is a panic here rather than an error. Every name and
//! shape this reads is fixed by the architecture, so a tensor that is missing at
//! layer 17 is not a condition a caller can do anything about, and threading a
//! `Result` through forty-two layers of a forward pass would say otherwise.
//! What [`CheckpointWeights::open`] can check before the pass starts, it does.

use std::cell::{OnceCell, RefCell};

use crate::attention::{
    AttentionConfig, AttentionProjections, AttentionWeights, DecodedProjections, Projections,
};
use crate::checkpoint::{Checkpoint, CheckpointError, Dtype, TensorView};
use crate::config::TextConfig;
use crate::generate::Generator;
use crate::head::LmHead;
use crate::layer::{DecoderCache, DecoderLayer, DecoderWeights, Experts, LayerMlp, NoExperts};
use crate::model::{Model, ModelWeights};
use crate::moe::{ExpertBank, Gate, GateWeights, Gathered, MoeConfig, SparseMoe};
use crate::ops::{DenseMlp, MlpProjections, Projection, linear};
use crate::profile::{self, Op};
use crate::quant::{BITS, QuantError, Scratch, dequantize_blocks_into};

/// Where the language model's tensors live in a multimodal checkpoint.
const MODEL: &str = "language_model.model";

/// The final projection, which sits beside `MODEL` rather than inside it —
/// `LanguageModel` holds `lm_head` and `InklingModel` holds everything else.
const LM_HEAD: &str = "language_model.lm_head";

/// The embedding table, which a tied checkpoint reaches twice.
const EMBED_TOKENS: &str = "language_model.model.embed_tokens";

/// Which tensor the head's rows come out of, which is the whole of what
/// `tie_word_embeddings` decides.
///
/// A tied checkpoint carries no `lm_head` at all and reads the embedding table
/// instead: `nn.Embedding.as_linear` is `h @ Wᵀ` over the same `[vocab, hidden]`
/// rows the lookup returns, so the two cases differ in the name and in nothing
/// else. Inkling-Small is untied.
fn head_module(config: &TextConfig) -> &'static str {
    match config.tie_word_embeddings {
        true => EMBED_TOKENS,
        false => LM_HEAD,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    #[error("{name} is {dtype:?}, which is not a float dtype")]
    NotFloat { name: String, dtype: Dtype },

    #[error("{name} is {dtype:?}, not the bfloat16 its bytes would be read as")]
    NotBf16 { name: String, dtype: Dtype },

    #[error("{name} is {shape:?}, which is not a matrix")]
    NotAMatrix { name: String, shape: Vec<usize> },

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

    /// The first `slices` slices as the checkpoint stores them: the packed
    /// bytes, and the scale bytes that go with them.
    ///
    /// For a backend that takes the weight rather than the values. What it gets
    /// is a range of the mapping and not a transcoding of 411 MB: the codes are
    /// the bytes of the `U32` tensor, which a kernel reading MXFP4 two codes to
    /// a byte needs nothing more of.
    ///
    /// A leading run rather than an arbitrary slice because that is the shape
    /// the truncation has: the vocabulary is the first `unpadded_vocab_size`
    /// rows of the head, so a backend cut to it is handed bytes that stop where
    /// the padding starts.
    pub fn prefix(&self, slices: usize) -> (&'a [u8], &'a [u8]) {
        assert!(
            slices <= self.slices(),
            "{slices} slices of a tensor that holds {}",
            self.slices()
        );
        let stride = |view: &TensorView<'a>| view.data().len() / self.slices();
        (
            &self.weight.data()[..slices * stride(&self.weight)],
            &self.scales.data()[..slices * stride(&self.scales)],
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

/// An `[out_dim, in_dim]` tensor the checkpoint stores as bfloat16, left where
/// it is mapped.
///
/// [`Packed`] above is every weight a matmul reads *but one*. The routers'
/// gates are `[258, 4096]` bfloat16 — the quantiser left them alone, at 2.1 MB
/// a layer — so a backend that wants to multiply against one is handed this
/// instead: the bytes as the checkpoint holds them, for a kernel that widens
/// each value where it multiplies rather than widening the tensor into memory.
///
/// The other bfloat16 tensors a layer holds are norms and convolution kernels,
/// which no matmul reads and which [`CheckpointWeights`] widens once.
#[derive(Debug, Clone, Copy)]
pub struct Bf16<'a> {
    view: TensorView<'a>,
    out_dim: usize,
    in_dim: usize,
}

impl<'a> Bf16<'a> {
    /// One bfloat16 matrix of this checkpoint.
    ///
    /// The dtype is checked here because it is what the bytes *mean*: a backend
    /// handed a float32 tensor under this type would read each half of a float
    /// as a value, which is a weight of the right length and of no relation to
    /// the one asked for.
    pub fn open(ckpt: &'a Checkpoint, name: &str) -> Result<Self, WeightsError> {
        let view = ckpt.tensor(name)?;
        if view.dtype() != Dtype::BF16 {
            return Err(WeightsError::NotBf16 {
                name: name.to_string(),
                dtype: view.dtype(),
            });
        }
        let &[out_dim, in_dim] = view.shape() else {
            return Err(WeightsError::NotAMatrix {
                name: name.to_string(),
                shape: view.shape().to_vec(),
            });
        };
        Ok(Self {
            view,
            out_dim,
            in_dim,
        })
    }

    /// The width a row of the input has to be.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// How many rows the weight holds, which is the width it maps to.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// The bytes themselves, row-major, as the checkpoint holds them.
    pub fn bytes(&self) -> &'a [u8] {
        self.view.data()
    }

    /// The same tensor widened, for a caller that multiplies against float32
    /// because it has no kernel that reads these bytes.
    pub fn to_f32(&self) -> Vec<f32> {
        let _timed = profile::scope(Op::Decode);
        self.view.to_f32().expect("a bfloat16 tensor widens")
    }
}

/// The CPU's answer to a projection whose weights stay packed: an
/// `[out_dim, in_dim]` MXFP4 tensor, one row of it decoded at a time into a
/// buffer the next row overwrites.
///
/// This is what [`crate::model`]'s bargain costs where the weight is one every
/// token touches all of. A decode step through the head decodes 200058 rows of
/// 4096 — 3.3 GB — to produce one token's logits, and holds 16 KB of them at
/// once. It is also the oracle: every kernel that multiplies the same bytes
/// without decoding them is checked against this, so it stays selectable rather
/// than becoming a fallback.
///
/// `out_dim` is how many of the leading axis' slices are rows of the weight,
/// which for the head is where the vocabulary ends. It bounds the loop rather
/// than cutting its answer, so honouring the truncation costs nothing: the rows
/// past it are not decoded at all.
#[derive(Debug)]
pub struct PackedRows<'a> {
    packed: Packed<'a>,
    out_dim: usize,
    decoded: RefCell<Vec<f32>>,
}

impl<'a> PackedRows<'a> {
    pub fn new(packed: Packed<'a>, out_dim: usize) -> Self {
        assert!(
            out_dim <= packed.slices(),
            "{out_dim} rows of a tensor that holds {}",
            packed.slices()
        );
        Self {
            decoded: RefCell::new(vec![0.0; packed.slice_len()]),
            packed,
            out_dim,
        }
    }
}

impl Projection for PackedRows<'_> {
    fn in_dim(&self) -> usize {
        self.packed.slice_len()
    }

    fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// `[rows, in_dim]` in, `[rows, out_dim]` out, a weight row at a time:
    /// decoded, multiplied against every row of `x`, and overwritten by the
    /// next one.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let in_dim = self.in_dim();
        assert_eq!(
            x.len() % in_dim,
            0,
            "{} values are not whole rows of {in_dim}",
            x.len()
        );

        let rows = x.len() / in_dim;
        let mut weight = self.decoded.borrow_mut();
        let mut out = vec![0.0; rows * self.out_dim];
        for col in 0..self.out_dim {
            profile::timed(Op::Decode, || {
                self.packed
                    .decode_slice_into(col, &mut weight)
                    .unwrap_or_else(|err| panic!("row {col} of the projection decodes: {err}"))
            });
            for (row, value) in linear(x, &weight, in_dim).into_iter().enumerate() {
                out[row * self.out_dim + col] = value;
            }
        }
        out
    }
}

/// One attention layer's five projections, still packed, for a backend that
/// takes the weight rather than the values.
#[derive(Debug, Clone, Copy)]
pub struct PackedAttention<'a> {
    pub q_proj: Packed<'a>,
    pub k_proj: Packed<'a>,
    pub v_proj: Packed<'a>,
    pub r_proj: Packed<'a>,
    pub o_proj: Packed<'a>,
}

/// One dense layer's feed-forward network, still packed.
///
/// The three tensors a [`PackedExperts`] holds, under a module that has no
/// experts: what a bank's leading axis indexes is which expert, and what these
/// index is which row of one projection.
#[derive(Debug, Clone, Copy)]
pub struct PackedMlp<'a> {
    pub gate_proj: Packed<'a>,
    pub up_proj: Packed<'a>,
    pub down_proj: Packed<'a>,
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

    /// The three banks still packed, for a backend that takes the weight rather
    /// than the values.
    pub fn gate_proj(&self) -> Packed<'a> {
        self.gate_proj
    }

    pub fn up_proj(&self) -> Packed<'a> {
        self.up_proj
    }

    pub fn down_proj(&self) -> Packed<'a> {
        self.down_proj
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
    let _timed = profile::scope(Op::Decode);
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
    lm_head: Packed<'a>,
    head: Box<dyn Projection + 'a>,
    experts: Option<Box<dyn ExpertBackend + 'a>>,
    projections: Option<Box<dyn ProjectionBackend + 'a>>,
    embed_norm: Option<Vec<f32>>,
    norm: Vec<f32>,
    layers: Vec<Widened<'a>>,
    layer_scratch: RefCell<Vec<f32>>,
    expert_scratch: RefCell<Vec<f32>>,
}

/// The tensors a layer holds that are not packed: bfloat16 in the checkpoint,
/// float32 in the arithmetic, and widened here once.
///
/// **These are the weights the residency argument does not apply to.** A
/// projection is MXFP4 and stays that way — decoding one is the thing
/// [`crate::model`] exists to avoid — but a norm weight has no packed form to
/// leave in place, so the choice is only whether to widen it again on every
/// layer of every step. A decode step widened 500 of these, which is every one
/// of them, and none had changed since the last time.
///
/// What that costs to hold is the whole of what it trades: 9.8 MB of norms,
/// convolution kernels and relative-position projections across forty-two
/// layers, beside 169 MB of routers' gates — one tensor is 95% of it, and it is
/// the one [`GateWeights`] describes.
#[derive(Debug)]
struct Widened<'a> {
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
    rel_proj: Vec<f32>,
    /// The trailing scale `InklingDenseMLP` and `InklingSparseMoE` both carry,
    /// which every layer has one of.
    global_scale: f32,
    /// The router's own two, and `None` on a dense layer — which has no router
    /// to hold them, the way it has no `switch_mlp` for
    /// [`CheckpointWeights::expert_banks`] to name.
    router: Option<WidenedRouter<'a>>,
}

/// The two tensors a router reads.
///
/// **The gate is not widened with the rest, and is the reason there is a cell
/// here at all.** It is 4.2 MB of float32 a layer against 9.8 MB for every
/// other bfloat16 tensor in the whole stack, and a backend that multiplies
/// against the checkpoint's own bytes never asks for it — see
/// [`Gate::Backend`]. The CPU path asks on its first step and never again.
#[derive(Debug)]
struct WidenedRouter<'a> {
    gate: Bf16<'a>,
    widened: OnceCell<Vec<f32>>,
    correction_bias: Vec<f32>,
}

impl<'a> WidenedRouter<'a> {
    fn widened(&self) -> &[f32] {
        self.widened.get_or_init(|| self.gate.to_f32())
    }
}

impl<'a> Widened<'a> {
    fn open(ckpt: &'a Checkpoint, config: &TextConfig, layer: usize) -> Result<Self, WeightsError> {
        let module = layer_module(layer);
        let of = |name: &str| widened(ckpt, &format!("{module}.{name}"));
        Ok(Self {
            input_layernorm: of("input_layernorm.weight")?,
            post_attention_layernorm: of("post_attention_layernorm.weight")?,
            attn_sconv: of("attn_sconv.conv.weight")?,
            mlp_sconv: of("mlp_sconv.conv.weight")?,
            q_norm: of("self_attn.q_norm.weight")?,
            k_norm: of("self_attn.k_norm.weight")?,
            k_sconv: of("self_attn.k_sconv.conv.weight")?,
            v_sconv: of("self_attn.v_sconv.conv.weight")?,
            rel_proj: of("self_attn.rel_proj")?,
            global_scale: of("mlp.global_scale")?[0],
            router: match config.layer_is_dense(layer) {
                true => None,
                false => Some(WidenedRouter {
                    gate: Bf16::open(ckpt, &format!("{module}.mlp.gate_weight"))?,
                    widened: OnceCell::new(),
                    correction_bias: of("mlp.e_score_correction_bias")?,
                }),
            },
        })
    }

    /// The router's tensors, of a layer that has one.
    fn router(&self, layer: usize) -> &WidenedRouter<'a> {
        self.router
            .as_ref()
            .unwrap_or_else(|| panic!("layer {layer} routes to experts and has no gate"))
    }
}

/// Where the MoE layers' experts run, when it is not here.
///
/// Per layer because the answer is per layer twice over: the first two are dense
/// and have no bank to run anywhere, and a backend that could not stand one
/// layer's banks up should be able to say so about that layer rather than about
/// the model.
pub trait ExpertBackend {
    /// Layer `layer`'s experts, or `None` for a layer this does not answer for —
    /// which leaves the CPU path to decode them.
    fn layer(&self, layer: usize) -> Option<&dyn Experts>;
}

/// One MoE layer's two banks and its router's gate, still as the checkpoint
/// stores them, from [`CheckpointWeights::expert_banks`].
#[derive(Debug, Clone, Copy)]
pub struct LayerBanks<'a> {
    pub layer: usize,
    /// The 256 a token reads six of.
    pub routed: PackedExperts<'a>,
    /// The two every token reads.
    pub shared: PackedExperts<'a>,
    /// The `[n_routed + n_shared, hidden]` gate that chooses between them,
    /// bfloat16 rather than packed.
    ///
    /// Handed over with the banks because of what it can be dispatched beside:
    /// what it produces weights what the shared bank produces, and does not
    /// decide what the shared bank runs, so a backend holding both can multiply
    /// them in one command buffer. See
    /// [`Experts::gated_shared`](crate::layer::Experts::gated_shared).
    pub gate_weight: Bf16<'a>,
}

/// One layer's own projections, still packed, from
/// [`CheckpointWeights::layer_projections`].
#[derive(Debug, Clone)]
pub struct LayerPacked<'a> {
    pub layer: usize,
    pub attention: PackedAttention<'a>,
    /// The feed-forward network of a dense layer, and `None` where the layer
    /// routes to experts instead — which is [`LayerBanks`]'s question.
    pub dense_mlp: Option<PackedMlp<'a>>,
    /// The weight of the norm those five projections consume the output of,
    /// widened rather than packed — it is bfloat16 in the checkpoint and 16 KB
    /// of it, so there is nothing here to leave in place.
    ///
    /// Handed over with the projections because it is theirs: nothing else in
    /// the model reads what `input_layernorm` produces, so a backend holding the
    /// five holds the norm in front of them or the value crosses back for
    /// nobody. See [`Projections::normed_qkvr`].
    pub input_layernorm: Vec<f32>,
}

/// Where a layer's own projections multiply, when it is not here.
///
/// Per layer for the reason [`ExpertBackend`] is asked per layer: a backend
/// that could not stand one layer's projections up should be able to say so
/// about that layer rather than about the model, and the two answers are
/// separate because only two layers of forty-two have a dense feed-forward
/// network at all.
///
/// Unlike the experts, every layer's attention projections are read by every
/// token — there is nothing here a decode step might skip — so what a backend
/// saves is the decoding rather than the reading.
pub trait ProjectionBackend {
    /// Layer `layer`'s five attention projections, or `None` for a layer this
    /// does not answer for — which leaves them to be decoded here.
    fn attention(&self, layer: usize) -> Option<&dyn Projections>;

    /// Layer `layer`'s dense feed-forward network, or `None` for a layer this
    /// does not answer for — which leaves it to be decoded here.
    ///
    /// Never asked of a layer that routes to experts instead: those have no
    /// feed-forward network at all, and what runs there is [`ExpertBackend`]'s
    /// question.
    fn dense_mlp(&self, layer: usize) -> Option<&dyn MlpProjections>;
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

        let table = |name: &str, what: &str| {
            let packed = Packed::open(ckpt, name)?;
            expect_packed(&packed, config.vocab_size, config.hidden_size, what)?;
            Ok::<_, WeightsError>(packed)
        };
        let embed_tokens = table(EMBED_TOKENS, "the embedding table")?;
        let lm_head = table(head_module(config), "the head")?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer| Widened::open(ckpt, config, layer))
            .collect::<Result<Vec<Widened<'a>>, WeightsError>>()?;

        Ok(Self {
            layers,
            layer_scratch: RefCell::new(vec![0.0; layer_scratch_floats(config)]),
            expert_scratch: RefCell::new(vec![0.0; expert_scratch_floats(config)]),
            head: Box::new(PackedRows::new(lm_head, LmHead::for_config(config).vocab())),
            experts: None,
            projections: None,
            embed_tokens,
            lm_head,
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

    /// The whole of `LanguageModel`: the stack, its final norm and the head,
    /// against whichever backend this was opened with.
    ///
    /// Assembled here rather than by each caller because the three pieces are
    /// one answer. A generator built from one checkpoint's model and another's
    /// head would run, and would be wrong in the way this whole module exists to
    /// make unrepresentable.
    pub fn generator(&self) -> Generator<'_> {
        Generator::new(self.model(), self.head(), self.head_projection())
    }

    /// The final projection this config asks for.
    pub fn head(&self) -> LmHead {
        LmHead::for_config(self.config)
    }

    /// Every MoE layer's two banks, still packed, for a backend that takes the
    /// weight rather than the values.
    ///
    /// The dense layers are absent rather than empty: they have no `switch_mlp`
    /// at all, so there is nothing for a backend to be handed and nothing for it
    /// to answer for.
    pub fn expert_banks(&self) -> Vec<LayerBanks<'a>> {
        (0..self.config.num_hidden_layers)
            .filter(|layer| !self.config.layer_is_dense(*layer))
            .map(|layer| {
                let (routed, shared) = self.banks(layer);
                LayerBanks {
                    layer,
                    routed,
                    shared,
                    gate_weight: self.layers[layer].router(layer).gate,
                }
            })
            .collect()
    }

    /// Every layer's own projections, still packed, for a backend that takes the
    /// weight rather than the values.
    ///
    /// Every layer has five attention projections and the first two have a dense
    /// feed-forward network besides, which is `dense_mlp` being `Some` exactly
    /// where [`CheckpointWeights::expert_banks`] has no entry.
    pub fn layer_projections(&self) -> Vec<LayerPacked<'a>> {
        (0..self.config.num_hidden_layers)
            .map(|layer| {
                let module = layer_module(layer);
                let packed = |name: &str| {
                    Packed::open(self.ckpt, &format!("{module}.{name}"))
                        .unwrap_or_else(|err| panic!("layer {layer}: {err}"))
                };
                LayerPacked {
                    layer,
                    input_layernorm: self.layers[layer].input_layernorm.clone(),
                    attention: PackedAttention {
                        q_proj: packed("self_attn.q_proj"),
                        k_proj: packed("self_attn.k_proj"),
                        v_proj: packed("self_attn.v_proj"),
                        r_proj: packed("self_attn.r_proj"),
                        o_proj: packed("self_attn.o_proj"),
                    },
                    dense_mlp: self.config.layer_is_dense(layer).then(|| PackedMlp {
                        gate_proj: packed("mlp.gate_proj"),
                        up_proj: packed("mlp.up_proj"),
                        down_proj: packed("mlp.down_proj"),
                    }),
                }
            })
            .collect()
    }

    /// The same weights with every layer's projections multiplied somewhere
    /// else, in place of decoding them into the pass's scratch a layer at a
    /// time.
    ///
    /// The largest of the three handovers by what it stops decoding: the
    /// attention projections are 7.40 GB of float32 a decode step and the two
    /// dense layers 1.61 GB, against the experts' 32 GB and the head's 3.3 —
    /// and unlike the experts, all of it is read by every token.
    pub fn with_projections(mut self, projections: Box<dyn ProjectionBackend + 'a>) -> Self {
        self.projections = Some(projections);
        self
    }

    /// The same weights with the experts run somewhere else — gathered Metal
    /// dispatches against banks that are never decoded, in place of the CPU's
    /// expert-at-a-time decode.
    ///
    /// The other half of [`CheckpointWeights::with_head`], and the larger one: a
    /// decode step decodes 32 GB of experts against the head's 3.3.
    pub fn with_experts(mut self, experts: Box<dyn ExpertBackend + 'a>) -> Self {
        self.experts = Some(experts);
        self
    }

    /// The tensor its rows come out of, still packed. Which tensor that is is
    /// settled once, by [`head_module`], at [`CheckpointWeights::open`] — so a
    /// backend that wants the bytes rather than the values asks here rather than
    /// answering `tie_word_embeddings` a second time.
    pub fn head_packed(&self) -> Packed<'a> {
        self.lm_head
    }

    /// What the head multiplies against, which is the backend this was opened
    /// with or the one [`CheckpointWeights::with_head`] put in its place.
    pub fn head_projection(&self) -> &dyn Projection {
        self.head.as_ref()
    }

    /// The same weights with the head projected somewhere else — a Metal
    /// dispatch against codes that are never decoded, in place of the CPU's
    /// row-at-a-time decode.
    ///
    /// **This is where the backend is chosen, and it is the only place.** A
    /// caller holding a [`CheckpointWeights`] cannot tell which one answered:
    /// the head is a [`Projection`] either way, the stack above it is untouched,
    /// and the CPU path is what this returns when nobody says otherwise.
    ///
    /// A projection that is not the head's shape is refused here rather than
    /// discovered by [`LmHead::forward`], which is one prefill later — and it is
    /// [`LmHead::expects`] that decides, so that the two answers cannot differ.
    pub fn with_head(mut self, head: Box<dyn Projection + 'a>) -> Self {
        self.head().expects(head.as_ref());
        self.head = head;
        self
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

    /// One packed tensor of this checkpoint, decoded whole into `scratch` and
    /// valid for as long as the run it was given.
    fn decoded<'s>(&self, name: &str, scratch: &mut Scratch<'s>) -> &'s [f32] {
        let _timed = profile::scope(Op::Decode);
        let packed = Packed::open(self.ckpt, name).unwrap_or_else(|err| panic!("{err}"));
        let run = scratch.take(packed.len());
        packed
            .decode_into(run)
            .unwrap_or_else(|err| panic!("{name} decodes: {err}"));
        run
    }

    /// One layer's attention tensors, its five projections decoded into the
    /// pass's scratch unless a backend answers for them.
    ///
    /// Nothing is decoded on the backend's arm, which is the whole of what this
    /// buys: 176 MB of float32 a layer that a decode step reads all of.
    fn attention<'s>(&'s self, layer: usize, scratch: &mut Scratch<'s>) -> AttentionWeights<'s> {
        let module = layer_module(layer);
        let widened = &self.layers[layer];
        AttentionWeights {
            q_norm: &widened.q_norm,
            k_norm: &widened.k_norm,
            k_sconv: &widened.k_sconv,
            v_sconv: &widened.v_sconv,
            rel_proj: &widened.rel_proj,
            projections: match self
                .projections
                .as_deref()
                .and_then(|backend| backend.attention(layer))
            {
                Some(handed) => AttentionProjections::backend(handed),
                None => {
                    let [q_proj, k_proj, v_proj, r_proj, o_proj] =
                        ["q_proj", "k_proj", "v_proj", "r_proj", "o_proj"].map(|name| {
                            self.decoded(&format!("{module}.self_attn.{name}"), scratch)
                        });
                    AttentionProjections::decoded(
                        self.config.hidden_size,
                        DecodedProjections {
                            q_proj,
                            k_proj,
                            v_proj,
                            r_proj,
                            o_proj,
                        },
                    )
                }
            },
        }
    }

    /// Whichever MLP the layer index called for, and the experts it can reach.
    ///
    /// A dense layer's three projections are decoded into the scratch beside its
    /// attention's, or left to whoever answers for them; a MoE layer decodes
    /// nothing at all and leaves both banks alone until a token routes into
    /// them.
    fn mlp<'s>(&'s self, layer: usize, scratch: &mut Scratch<'s>, gated: bool) -> Mlp<'s> {
        let module = layer_module(layer);
        let widened = &self.layers[layer];
        let global_scale = widened.global_scale;

        let Some(config) = MoeConfig::for_layer(self.config, layer) else {
            let (dim, hidden_dim) = (self.config.hidden_size, self.config.dense_intermediate_size);
            return Mlp::Dense(
                match self
                    .projections
                    .as_deref()
                    .and_then(|backend| backend.dense_mlp(layer))
                {
                    Some(handed) => DenseMlp::backend(dim, hidden_dim, handed, global_scale),
                    None => {
                        let [gate_proj, up_proj, down_proj] = ["gate_proj", "up_proj", "down_proj"]
                            .map(|name| self.decoded(&format!("{module}.mlp.{name}"), scratch));
                        DenseMlp::new(dim, gate_proj, up_proj, down_proj, global_scale)
                    }
                },
            );
        };

        let (routed, shared) = self.banks(layer);
        let router = widened.router(layer);
        Mlp::Sparse(Box::new(Sparse {
            config,
            gate: match gated {
                true => Gate::Backend {
                    hidden: self.config.hidden_size,
                },
                false => Gate::Widened(router.widened()),
            },
            correction_bias: &router.correction_bias,
            global_scale,
            routed,
            shared,
            scratch: &self.expert_scratch,
        }))
    }

    /// One MoE layer's routed and shared banks, still packed.
    fn banks(&self, layer: usize) -> (PackedExperts<'a>, PackedExperts<'a>) {
        let bank = |name: &str| {
            PackedExperts::open(self.ckpt, &format!("{}.mlp.{name}", layer_module(layer)))
                .unwrap_or_else(|err| panic!("layer {layer}: {err}"))
        };
        (bank("switch_mlp"), bank("shared_experts"))
    }
}

/// Where one decoder layer's tensors live.
fn layer_module(layer: usize) -> String {
    format!("{MODEL}.layers.{layer}")
}

impl ModelWeights for CheckpointWeights<'_> {
    fn embedding_row(&self, id: usize) -> Vec<f32> {
        let _timed = profile::scope(Op::Embedding);
        self.embed_tokens
            .decode_slice(id)
            .unwrap_or_else(|err| panic!("embedding row {id} decodes: {err}"))
    }

    fn run_layer(&self, index: usize, cache: &mut DecoderCache, x: &[f32]) -> Vec<f32> {
        let mut buffer = self.layer_scratch.borrow_mut();
        let mut scratch = Scratch::new(&mut buffer);
        let widened = &self.layers[index];

        let experts = self
            .experts
            .as_ref()
            .and_then(|backend| backend.layer(index));
        let attention = self.attention(index, &mut scratch);
        let mlp = self.mlp(index, &mut scratch, experts.is_some_and(Experts::gates));
        let weights = DecoderWeights {
            attention,
            input_layernorm: &widened.input_layernorm,
            post_attention_layernorm: &widened.post_attention_layernorm,
            attn_sconv: &widened.attn_sconv,
            mlp_sconv: &widened.mlp_sconv,
        };
        let config = AttentionConfig::for_layer(self.config, index);
        let layer = DecoderLayer::new(config, weights, mlp.view());
        match experts {
            Some(experts) => layer.forward(cache, x, experts),
            None => layer.forward(cache, x, &mlp),
        }
    }
}

/// Whichever MLP a layer index called for. A dense layer is a feed-forward
/// network over three projections; a MoE layer is its gate, and the two banks
/// its router can reach but has not.
enum Mlp<'a> {
    Dense(DenseMlp<'a>),
    Sparse(Box<Sparse<'a>>),
}

/// `InklingSparseMoE`'s gate, and the two banks it routes to left packed.
struct Sparse<'a> {
    config: MoeConfig,
    gate: Gate<'a>,
    correction_bias: &'a [f32],
    global_scale: f32,
    routed: PackedExperts<'a>,
    shared: PackedExperts<'a>,
    scratch: &'a RefCell<Vec<f32>>,
}

impl Mlp<'_> {
    fn view(&self) -> LayerMlp<'_> {
        match self {
            Self::Dense(mlp) => LayerMlp::Dense(*mlp),
            Self::Sparse(moe) => LayerMlp::Sparse(SparseMoe::new(
                moe.config,
                GateWeights {
                    gate: moe.gate,
                    correction_bias: moe.correction_bias,
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
    fn routed(&self, gathered: Gathered<'_>) -> Vec<f32> {
        match self {
            Self::Dense(_) => NoExperts.routed(gathered),
            Self::Sparse(moe) => through(&moe.routed, gathered, moe.scratch),
        }
    }

    fn shared(&self, gathered: Gathered<'_>) -> Vec<f32> {
        match self {
            Self::Dense(_) => NoExperts.shared(gathered),
            Self::Sparse(moe) => through(&moe.shared, gathered, moe.scratch),
        }
    }
}

/// One bank, an expert at a time. Each expert's three projections are decoded
/// into the pass's expert buffer, which the next expert overwrites — so a layer
/// whose eight experts ran costs one expert's float32, not eight, and the runs
/// [`Gathered::batches`] hands out are what says no expert is decoded twice.
fn through(
    bank: &PackedExperts<'_>,
    gathered: Gathered<'_>,
    scratch: &RefCell<Vec<f32>>,
) -> Vec<f32> {
    let mut buffer = scratch.borrow_mut();
    let mut out = Vec::with_capacity(gathered.rows().len());
    for (expert, rows) in gathered.batches() {
        out.extend(bank.forward_into(expert, gathered.dim(), rows, &mut Scratch::new(&mut buffer)));
    }
    out
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
    expect(values.len(), expected, name)
}

/// A packed tensor's two axes, which for a table of rows is how many rows it
/// holds and how wide each is.
fn expect_packed(
    packed: &Packed<'_>,
    slices: usize,
    slice_len: usize,
    name: &str,
) -> Result<(), WeightsError> {
    expect(packed.slices(), slices, &format!("{name}'s rows"))?;
    expect(packed.slice_len(), slice_len, &format!("a row of {name}"))
}

fn expect(got: usize, expected: usize, name: &str) -> Result<(), WeightsError> {
    if got == expected {
        return Ok(());
    }
    Err(WeightsError::WrongLength {
        name: name.to_string(),
        expected,
        got,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::fixture;
    use crate::quant::GROUP_SIZE;

    /// The MoE fixture, which carries the one thing no other committed fixture
    /// does: the trained `[258, 4096]` gate of the captured layer, as bfloat16
    /// rather than widened. It is the shape and the dtype [`Bf16`] exists for.
    const GATE_FIXTURE: &str = "moe.safetensors";
    const GATE: &str = "trained.gate_weight";

    /// The synthetic stack's config, read for its shape rather than its
    /// contents: any real `TextConfig` settles the question below.
    const FIXTURE_CONFIG: &str = "stack.json";

    fn config() -> TextConfig {
        serde_json::from_str::<Config>(&fixture::read(FIXTURE_CONFIG))
            .expect("the recorded config parses")
            .text_config
    }

    /// The tie is a choice of tensor name and nothing else, so this is the whole
    /// of it — and it is worth a test of its own because no Inkling checkpoint
    /// ties, which leaves the branch unreachable everywhere else. Exchanged, the
    /// two arms would read an untied checkpoint's embedding table as its head and
    /// still produce a full set of plausible logits.
    #[test]
    fn tying_the_word_embeddings_moves_the_head_to_the_embedding_table() {
        let mut config = config();
        assert!(!config.tie_word_embeddings, "the fixture does not tie");
        assert_eq!(head_module(&config), LM_HEAD);

        config.tie_word_embeddings = true;
        assert_eq!(head_module(&config), EMBED_TOKENS);
        assert_ne!(LM_HEAD, EMBED_TOKENS);
    }

    /// A bfloat16 weight is its two axes and the checkpoint's own bytes, and it
    /// is those bytes rather than a copy of them — which is the whole of what a
    /// backend that wraps a mapping needs to be true.
    ///
    /// The lengths are the claim about the axes. `[258, 4096]` of bfloat16 is
    /// 2.11 MB, and a reader that took the shape the other way round would
    /// still describe a tensor of exactly that many bytes.
    #[test]
    fn a_bfloat16_weight_is_the_checkpoints_own_bytes_under_its_two_axes() {
        let ckpt = fixture::open(GATE_FIXTURE);
        let gate = Bf16::open(&ckpt, GATE).expect("the fixture holds the trained gate");

        assert_eq!(
            gate.out_dim(),
            258,
            "the routed experts and the shared pair"
        );
        assert_eq!(gate.in_dim(), 4096, "the hidden width");
        assert_eq!(
            gate.bytes().len(),
            gate.out_dim() * gate.in_dim() * 2,
            "two bytes a value"
        );
        assert_eq!(
            gate.bytes().as_ptr(),
            fixture::tensor(&ckpt, GATE).data().as_ptr(),
            "the bytes were copied rather than addressed"
        );
    }

    /// The dtype is what the bytes *mean*, so it is checked rather than
    /// assumed: the fixture's `x` is float32 of exactly the shape a gate could
    /// have, and read as bfloat16 it would be a weight of twice the rows with
    /// each half of a float standing in for a value.
    #[test]
    fn a_weight_that_is_not_bfloat16_is_refused() {
        let ckpt = fixture::open(GATE_FIXTURE);
        let err = Bf16::open(&ckpt, "x").expect_err("x is float32");

        assert!(
            matches!(
                err,
                WeightsError::NotBf16 {
                    dtype: Dtype::F32,
                    ..
                }
            ),
            "{err}"
        );
    }

    /// 64 rows of `layers.0.mlp.gate_proj`, which is a projection the engine
    /// runs in anger.
    const PROJECTION: &str = "dense_ffn";

    fn packed<'a>(ckpt: &'a Checkpoint, slice: &str) -> Packed<'a> {
        Packed::open(ckpt, slice).expect("the fixture holds the slice packed")
    }

    /// The same slice as MLX decoded it, which is the oracle the packed rows are
    /// multiplied against.
    fn decoded(ckpt: &Checkpoint, slice: &str) -> Vec<f32> {
        fixture::f32s(&fixture::tensor(ckpt, &format!("{slice}.dequantized")))
    }

    /// Two rows of input, spread over both signs so that a reduction cancels
    /// the way a trained one does.
    fn rows(in_dim: usize) -> Vec<f32> {
        (0..2 * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
            .collect()
    }

    /// What the whole seam claims: a projection over packed rows is the same
    /// multiply as one over the decoded weight, and *exactly* the same — both
    /// sides decode identically and then run the same [`linear`], so the only
    /// thing that could separate them is which bytes were read.
    #[test]
    fn a_packed_projection_multiplies_what_the_decoded_weight_does() {
        let ckpt = fixture::open(fixture::MXFP4);
        let packed = packed(&ckpt, PROJECTION);
        let weight = decoded(&ckpt, PROJECTION);

        let projection = PackedRows::new(packed, packed.slices());
        assert_eq!(projection.in_dim(), packed.slice_len());
        assert_eq!(projection.out_dim(), packed.slices());

        let x = rows(projection.in_dim());
        assert_eq!(
            projection.forward(&x),
            linear(&x, &weight, projection.in_dim())
        );
    }

    /// What a backend that uploads a weight is handed: the bytes themselves, as
    /// many of them as the rows asked for, and the ones the mapping already
    /// holds rather than a copy.
    ///
    /// The lengths are the claim. A cut that produced the whole tensor would
    /// still decode correctly wherever it was indexed, so nothing about the
    /// values would say the padding had been moved onto a device — only the
    /// count of bytes does.
    #[test]
    fn a_prefix_is_the_leading_slices_bytes_and_no_more() {
        let ckpt = fixture::open(fixture::MXFP4);
        let packed = packed(&ckpt, fixture::VOCAB_PADDING);
        let rows = fixture::VOCAB_PADDING_ROWS;
        assert!(rows < packed.slices(), "a cut that cuts nothing");

        let (codes, scales) = packed.prefix(packed.slices());
        assert_eq!(codes.len(), packed.len() * BITS / u8::BITS as usize);
        assert_eq!(scales.len(), packed.len() / GROUP_SIZE);

        let (cut_codes, cut_scales) = packed.prefix(rows);
        assert_eq!(cut_codes.len(), rows * codes.len() / packed.slices());
        assert_eq!(cut_scales.len(), rows * scales.len() / packed.slices());
        assert_eq!(cut_codes, &codes[..cut_codes.len()]);
        assert_eq!(cut_scales, &scales[..cut_scales.len()]);
    }

    #[test]
    #[should_panic(expected = "65 slices of a tensor that holds 64")]
    fn a_prefix_longer_than_the_tensor_is_refused() {
        let ckpt = fixture::open(fixture::MXFP4);
        let packed = packed(&ckpt, PROJECTION);
        packed.prefix(packed.slices() + 1);
    }

    /// The truncation, on the checkpoint's own bytes: a projection cut at the
    /// vocabulary decodes the rows below the cut and no others.
    ///
    /// Stated on the values rather than on a spy, because a padding row is
    /// all-zero codes under all-zero scales and so multiplies to exactly 0.0 —
    /// which is what [`crate::head`] says the truncation exists to keep out of
    /// the ranking. The untruncated projection here holds 32 of them and the
    /// truncated one holds none.
    #[test]
    fn a_truncated_projection_stops_at_the_row_it_was_cut_to() {
        let ckpt = fixture::open(fixture::MXFP4);
        let packed = packed(&ckpt, fixture::VOCAB_PADDING);
        let whole = PackedRows::new(packed, packed.slices());
        let cut = PackedRows::new(packed, fixture::VOCAB_PADDING_ROWS);

        let x = rows(whole.in_dim());
        let untruncated = whole.forward(&x);
        let truncated = cut.forward(&x);
        assert_eq!(truncated.len(), 2 * fixture::VOCAB_PADDING_ROWS);

        for row in 0..2 {
            let (whole, cut) = (
                &untruncated[row * whole.out_dim()..][..whole.out_dim()],
                &truncated[row * fixture::VOCAB_PADDING_ROWS..][..fixture::VOCAB_PADDING_ROWS],
            );
            assert_eq!(
                &whole[..fixture::VOCAB_PADDING_ROWS],
                cut,
                "row {row} up to the cut"
            );
            assert!(
                whole[fixture::VOCAB_PADDING_ROWS..]
                    .iter()
                    .all(|logit| *logit == 0.0),
                "row {row}: the padding this fixture holds is not all-zero, so the cut is not \
                 what it keeps out"
            );
            assert!(
                cut.iter().any(|logit| *logit != 0.0),
                "row {row}: the real rows multiply to zero too"
            );
        }
    }

    #[test]
    #[should_panic(expected = "65 rows of a tensor that holds 64")]
    fn a_projection_over_more_rows_than_the_tensor_holds_is_refused() {
        let ckpt = fixture::open(fixture::MXFP4);
        let packed = packed(&ckpt, PROJECTION);
        PackedRows::new(packed, packed.slices() + 1);
    }

    #[test]
    #[should_panic(expected = "are not whole rows of 4096")]
    fn a_packed_projection_over_a_ragged_input_is_refused() {
        let ckpt = fixture::open(fixture::MXFP4);
        let packed = packed(&ckpt, PROJECTION);
        PackedRows::new(packed, packed.slices()).forward(&rows(packed.slice_len())[1..]);
    }
}
