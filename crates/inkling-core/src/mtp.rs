//! The multi-token prediction heads: eight of them, each guessing one token
//! further ahead than the last.
//!
//! Nothing here is a new op. A head is a decoder layer with three tensors in
//! front of it:
//!
//! ```text
//! hidden ─→ hidden_norm ─┐
//!                        ├─→ [hidden; embed] ─→ input_proj ─→ block ─→ hidden'
//! embed  ─→ embed_norm  ─┘        [2*h]          [h, 2*h]
//! ```
//!
//! Head `d` at position `i` consumes the chained hidden state and the embedding
//! of the token at `i + d + 1`, and its output predicts the token at `i + d + 2`
//! — through the model's own final norm and `lm_head`, which is what makes a
//! head's guess comparable to the model's own answer at all. Head 0 is chained
//! from the main stack's *post*-final-norm hidden state; every head after it is
//! chained from the one before it, raw. Both halves of that are `mtp_config`'s:
//! `chain_hidden_post_norm: false` governs the links between heads, which are
//! the raw ones, and says nothing about the main stack's, which is normed.
//!
//! # What the tensors fix and what they do not
//!
//! `input_proj` is `[hidden, 2 * hidden]`, which fixes that it consumes two
//! normed vectors and not which half is which; and the checkpoint's embedding
//! is already normed by the main stack's `embed_norm`, so whether a head's own
//! `embed_norm` stacks on that or replaces it is undetermined. Neither is a
//! choice this makes: `reference/results/mtp_acceptance.md` scored all eight
//! combinations over 2171 positions and the answer is the doubly normed
//! embedding beside the post-norm hidden state, `[hidden; embed]` — reversed,
//! the projection reads the hidden state through the half of the weight trained
//! for embeddings and the head agrees with the model on nothing.
//!
//! # The heads are a stack of their own
//!
//! `mtp_config` carries its own `local_layer_ids`, so head 1 and head 3 are
//! global attention where the other six are sliding, and every head is dense —
//! there is no `switch_mlp` under a head at all. [`head_config`] is the whole of
//! that: the main stack's `TextConfig` with the heads' layer plan in place of
//! its own, which is what every shape below is then derived from the same way a
//! layer's are.
//!
//! # A head's guess is never load-bearing
//!
//! Everything here feeds [`crate::generate`]'s speculative loop, which verifies
//! what a head guessed against the model's own answer and keeps only the prefix
//! that matched. So a head standing on a cache that has drifted costs
//! acceptance and cannot cost correctness — which is what makes it safe for the
//! chain to guess from its own guesses, and what the proposer in
//! [`crate::generate`] states.
//!
//! Pinned to mlx-vlm by `reference/fixtures/mtp.safetensors`: two synthetic
//! heads, one sliding and one global, driven through the reference's own
//! `InklingMTPLayer` twice against one cache.

use std::cell::RefCell;

use crate::attention::{
    AttentionConfig, AttentionProjections, AttentionWeights, DecodedProjections, Projections,
};
use crate::checkpoint::Checkpoint;
use crate::config::{MtpConfig, TextConfig};
use crate::generate::{Generator, Proposer, Round};
use crate::layer::{
    DecoderCache, DecoderLayer, DecoderStep, DecoderWeights, Hidden, LayerMlp, NoExperts, Passed,
};
use crate::model::{ModelCache, ModelWeights};
use crate::ops::{DenseMlp, DenseProjection, MlpProjections, Projection, rms_norm};
use crate::quant::Scratch;
use crate::weights::{Bf16, WeightsError, widened};

/// Where the heads' tensors live, which is a namespace of their own: the
/// quantisers rewrote the main stack's names to mlx-vlm's and left these under
/// the ones the original ships.
const MTP: &str = "model.mtp.layers";

/// The layer plan the heads run under: their own local/global split, all of
/// them dense, and as many layers as there are heads.
///
/// The main stack's config in every other respect, because every other shape a
/// head has is the model's — the hidden width, the head counts, the band, the
/// convolution kernel and the dense FFN width are all read from the same
/// fields. This is `mtp_text_config` in the acceptance study, which is what the
/// study's numbers were measured through.
pub fn head_config(text: &TextConfig, mtp: &MtpConfig) -> TextConfig {
    TextConfig {
        num_hidden_layers: mtp.num_nextn_predict_layers,
        local_layer_ids: mtp.local_layer_ids.clone(),
        dense_mlp_idx: mtp.num_nextn_predict_layers,
        ..text.clone()
    }
}

/// One head, from the pair of vectors it consumes to the hidden state it
/// answers with.
#[derive(Clone, Copy)]
pub struct MtpHead<'a> {
    hidden_norm: &'a [f32],
    embed_norm: &'a [f32],
    input_proj: &'a dyn Projection,
    block: DecoderLayer<'a>,
    eps: f32,
}

impl<'a> MtpHead<'a> {
    pub fn new(
        norms: HeadNorms<'a>,
        input_proj: &'a dyn Projection,
        block: DecoderLayer<'a>,
        eps: f32,
    ) -> Self {
        let hidden = block.hidden();
        assert_eq!(norms.hidden_norm.len(), hidden, "hidden_norm");
        assert_eq!(norms.embed_norm.len(), hidden, "embed_norm");
        assert_eq!(
            (input_proj.in_dim(), input_proj.out_dim()),
            (2 * hidden, hidden),
            "input_proj against a hidden {hidden}"
        );

        Self {
            hidden_norm: norms.hidden_norm,
            embed_norm: norms.embed_norm,
            input_proj,
            block,
            eps,
        }
    }

    pub fn hidden(&self) -> usize {
        self.block.hidden()
    }

    /// The state a sequence starts a head from, for this head's own shape.
    pub fn cache(&self) -> DecoderCache {
        self.block.cache()
    }

    /// `[rows, hidden]` of chained hidden state and `[rows, hidden]` of
    /// embeddings in, `[rows, hidden]` out, continuing from `cache` and leaving
    /// this call's keys and convolution windows behind in it.
    ///
    /// `head` is the head's index, which is a decoder layer's index in the
    /// heads' own stack — see [`head_config`].
    ///
    /// **`device` is handed the projection and the block together**, and that
    /// is the whole of why [`HeadDevice`] exists beside
    /// [`DecoderDevice`](crate::layer::DecoderDevice): what `input_proj`
    /// produces is what the block's first dispatch reads and nothing else looks
    /// at it, so a backend given both runs a head in one command buffer where a
    /// backend given the layer alone has to close one in front of it. The two
    /// norms and the concatenation between them stay here — they are what makes
    /// the pair a row of `2 * hidden`, and the projection is the first thing
    /// that reads it.
    pub fn forward(
        &self,
        head: usize,
        cache: &mut DecoderCache,
        hidden: &[f32],
        embed: &[f32],
        device: Option<&dyn HeadDevice>,
    ) -> Vec<f32> {
        assert_eq!(hidden.len(), embed.len(), "a hidden state per embedding");
        let rows = Hidden::Rows(hidden).tokens(self.hidden());
        let input = self.concatenated(hidden, embed);
        let passed = device.and_then(|device| {
            let seen = cache.attention().seen();
            self.block
                .described(seen, Hidden::Carried(rows), rows, |block| {
                    device.run(
                        head,
                        cache,
                        HeadStep {
                            input: &input,
                            block,
                        },
                    )
                })
        });
        let passed = match passed {
            Some(passed) => passed,
            None => {
                let x = self.input_proj.forward(&input);
                self.block
                    .forward(head, cache, Hidden::Rows(&x), &NoExperts, None)
            }
        };
        match passed {
            Passed::Rows(rows) => rows,
            passed => panic!("a head's block answered with {passed:?}"),
        }
    }

    /// Each row's two normed halves laid end to end, which is what
    /// `input_proj` was trained to read.
    fn concatenated(&self, hidden: &[f32], embed: &[f32]) -> Vec<f32> {
        let width = self.hidden();
        let normed = |rows: &[f32], weight: &[f32]| rms_norm(rows, weight, self.eps);
        normed(hidden, self.hidden_norm)
            .chunks_exact(width)
            .zip(normed(embed, self.embed_norm).chunks_exact(width))
            .flat_map(|(hidden, embed)| [hidden, embed].concat())
            .collect()
    }
}

/// The two RMSNorms in front of a head, which are the head's own and not the
/// model's — the embedding they normalise has already been through
/// `embed_norm` once.
#[derive(Debug, Clone, Copy)]
pub struct HeadNorms<'a> {
    pub hidden_norm: &'a [f32],
    pub embed_norm: &'a [f32],
}

/// Where a whole head runs, when it is not here.
///
/// The mirror of [`DecoderDevice`](crate::layer::DecoderDevice) one step out: a
/// head is a decoder layer with a projection in front of it, and what that
/// projection produces is read by the block's first dispatch and by nothing
/// else — so the two belong in one command buffer, and a backend that could
/// only be asked for the layer would have to close one between them.
///
/// **What does not come here is the pair of norms**, which is where a head's
/// seam stops being a layer's: they are `[rows, hidden]` each and what
/// `input_proj` reads is the `[rows, 2 * hidden]` they are laid into, so the
/// concatenation is on this side and the projection is the first thing past it.
pub trait HeadDevice {
    /// Head `head`, or `None` where this backend does not hold all of it.
    fn run(&self, head: usize, cache: &mut DecoderCache, step: HeadStep<'_>) -> Option<Passed>;
}

/// Everything one head runs past the two norms in front of it, described rather
/// than run.
#[derive(Debug, Clone, Copy)]
pub struct HeadStep<'a> {
    /// `[rows, 2 * hidden]`: each row's two normed halves laid end to end,
    /// which is what `input_proj` was trained to read — see
    /// [`MtpHead::concatenated`].
    pub input: &'a [f32],
    /// The decoder layer behind that projection, whose own input is what the
    /// projection produced and is therefore a value nobody on this side sees —
    /// which is what [`Hidden::Carried`] says.
    pub block: DecoderStep<'a>,
}

/// Where a head's multiplies run, when they are not here.
///
/// The mirror of [`LayerBackend`](crate::weights::LayerBackend), and shorter for
/// the reason a head is: every head is dense, so there is no bank to answer for,
/// and what a head has that a layer does not is the projection in front of it.
pub trait HeadBackend {
    /// Head `head`'s `[hidden, 2 * hidden]` input projection, or `None` for a
    /// head this does not answer for.
    fn input_proj(&self, head: usize) -> Option<&dyn Projection>;

    /// Head `head`'s five attention projections, or `None`.
    fn attention(&self, head: usize) -> Option<&dyn Projections>;

    /// Head `head`'s feed-forward network, or `None`.
    fn mlp(&self, head: usize) -> Option<&dyn MlpProjections>;

    /// The whole of head `head` in one command buffer, or `None` where this
    /// backend holds only some of it — see [`HeadDevice`].
    ///
    /// Defaulted where the three above are not, for
    /// [`LayerBackend::decoder`](crate::weights::LayerBackend::decoder)'s
    /// reason: a backend can answer for every weight a head has and still have
    /// nothing to gain from being asked the head whole.
    fn device(&self, head: usize) -> Option<&dyn HeadDevice> {
        let _ = head;
        None
    }

    /// Take back the last `rows` timesteps of whatever state this backend holds
    /// for head `head` — the keys it appended, and the convolution windows it
    /// advanced.
    ///
    /// Defaulted to nothing, for the reason
    /// [`LayerBackend::rewind`](crate::weights::LayerBackend::rewind) is: a
    /// backend that holds only weights holds nothing a sequence can be rewound
    /// out of. A backend that runs a head's block keeps that head's span and its
    /// four windows, and a [`DecoderCache`] rewound on this side alone would
    /// leave them where the frontier row put them.
    ///
    /// Per head rather than over all of them, because a round runs the heads its
    /// depth asked for and no others — see [`MtpProposer::propose`].
    fn rewind(&self, head: usize, rows: usize) {
        let (_, _) = (head, rows);
    }
}

/// One head's tensors, as the checkpoint stores them.
///
/// The eight big ones stay bfloat16 where they are mapped — 532 MiB a head,
/// which is 4.2 GiB over the eight heads and the whole reason
/// [`CheckpointHeads`] holds a scratch rather than a decoded weight. The small
/// ones are widened once, for the reason [`Widened`](crate::weights) states: a
/// norm has no packed form to be left in, and these come to 260 KB a head.
#[derive(Debug)]
struct HeadTensors<'a> {
    hidden_norm: Vec<f32>,
    embed_norm: Vec<f32>,
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    attn_sconv: Vec<f32>,
    mlp_sconv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_sconv: Vec<f32>,
    v_sconv: Vec<f32>,
    rel_proj: Vec<f32>,
    global_scale: f32,
    input_proj: Bf16<'a>,
    attention: HeadAttention<'a>,
    /// The fused gate and up of the head's SwiGLU, `[2 * dense, hidden]` with
    /// the two **interleaved** row by row — see [`Mlp::split`].
    w13: Bf16<'a>,
    w2: Bf16<'a>,
}

/// A head's five attention projections, still bfloat16.
#[derive(Debug, Clone, Copy)]
pub struct HeadAttention<'a> {
    pub q_proj: Bf16<'a>,
    pub k_proj: Bf16<'a>,
    pub v_proj: Bf16<'a>,
    pub r_proj: Bf16<'a>,
    pub o_proj: Bf16<'a>,
}

/// One head's tensors, still bfloat16, for a backend that takes the weight
/// rather than the values.
#[derive(Debug, Clone)]
pub struct HeadPacked<'a> {
    pub head: usize,
    pub config: AttentionConfig,
    pub input_proj: Bf16<'a>,
    pub attention: HeadAttention<'a>,
    /// The fused `[2 * dense, hidden]` gate and up, interleaved row by row.
    pub w13: Bf16<'a>,
    pub w2: Bf16<'a>,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub attn_sconv: Vec<f32>,
    pub mlp_sconv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub k_sconv: Vec<f32>,
    pub v_sconv: Vec<f32>,
    pub rel_proj: Vec<f32>,
    pub global_scale: f32,
}

impl<'a> HeadTensors<'a> {
    fn open(ckpt: &'a Checkpoint, head: usize) -> Result<Self, WeightsError> {
        let module = format!("{MTP}.{head}");
        let widened = |name: &str| widened(ckpt, &format!("{module}.{name}"));
        let block = |name: &str| format!("transformer_block.{name}");
        let matrix = |name: &str| Bf16::open(ckpt, &format!("{module}.{name}"));
        let attn = |name: &str| matrix(&block(&format!("attn.{name}.weight")));
        Ok(Self {
            hidden_norm: widened("hidden_norm.weight")?,
            embed_norm: widened("embed_norm.weight")?,
            input_layernorm: widened(&block("attn_norm.weight"))?,
            post_attention_layernorm: widened(&block("mlp_norm.weight"))?,
            attn_sconv: widened(&block("attn_sconv.weight"))?,
            mlp_sconv: widened(&block("mlp_sconv.weight"))?,
            q_norm: widened(&block("attn.q_norm.weight"))?,
            k_norm: widened(&block("attn.k_norm.weight"))?,
            k_sconv: widened(&block("attn.k_sconv.weight"))?,
            v_sconv: widened(&block("attn.v_sconv.weight"))?,
            rel_proj: widened(&block("attn.rel_logits_proj.proj"))?,
            global_scale: widened(&block("mlp.global_scale"))?[0],
            input_proj: matrix("input_proj.weight")?,
            attention: HeadAttention {
                q_proj: attn("wq_du")?,
                k_proj: attn("wk_dv")?,
                v_proj: attn("wv_dv")?,
                r_proj: attn("wr_du")?,
                o_proj: attn("wo_ud")?,
            },
            w13: matrix(&block("mlp.w13_dn.weight"))?,
            w2: matrix(&block("mlp.w2_md.weight"))?,
        })
    }

    fn packed(&self, head: usize, config: AttentionConfig) -> HeadPacked<'a> {
        HeadPacked {
            head,
            config,
            input_proj: self.input_proj,
            attention: self.attention,
            w13: self.w13,
            w2: self.w2,
            input_layernorm: self.input_layernorm.clone(),
            post_attention_layernorm: self.post_attention_layernorm.clone(),
            attn_sconv: self.attn_sconv.clone(),
            mlp_sconv: self.mlp_sconv.clone(),
            q_norm: self.q_norm.clone(),
            k_norm: self.k_norm.clone(),
            k_sconv: self.k_sconv.clone(),
            v_sconv: self.v_sconv.clone(),
            rel_proj: self.rel_proj.clone(),
            global_scale: self.global_scale,
        }
    }
}

/// The eight heads out of a checkpoint, widened only where a call reaches them.
///
/// The same bargain [`CheckpointWeights`](crate::weights::CheckpointWeights)
/// makes, one format further back: a head's weights are bfloat16 rather than
/// packed, so what this declines to do is widen them into memory. A backend
/// that multiplies against the checkpoint's own bytes never widens one at all;
/// the CPU path widens a head into a scratch the next head overwrites.
///
/// **The scratch is not allocated until a head is run here.** It is 1.1 GB —
/// larger than everything else this process holds — and a run that never
/// speculates never touches it, which is what keeps the resident set of a
/// generation the same whether or not the heads are loaded.
pub struct CheckpointHeads<'a> {
    config: TextConfig,
    heads: Vec<HeadTensors<'a>>,
    backend: Option<Box<dyn HeadBackend + 'a>>,
    scratch: RefCell<Vec<f32>>,
}

impl<'a> CheckpointHeads<'a> {
    /// Map the heads' tensors, or say which one the checkpoint does not hold.
    ///
    /// An error rather than a panic, unlike the stack's: a checkpoint without
    /// MTP tensors is an ordinary checkpoint, and the caller's answer to that
    /// is to decode one token at a time rather than to abort.
    pub fn open(
        ckpt: &'a Checkpoint,
        text: &TextConfig,
        mtp: &MtpConfig,
    ) -> Result<Self, WeightsError> {
        if mtp.chain_hidden_post_norm {
            return Err(WeightsError::Unsupported {
                what: "chain_hidden_post_norm, which norms the link between two heads".to_string(),
            });
        }
        let config = head_config(text, mtp);
        let heads = (0..config.num_hidden_layers)
            .map(|head| HeadTensors::open(ckpt, head))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            config,
            heads,
            backend: None,
            scratch: RefCell::new(Vec::new()),
        })
    }

    /// The same heads with their multiplies run somewhere else.
    pub fn with_backend(mut self, backend: Box<dyn HeadBackend + 'a>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// The heads' own layer plan.
    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    pub fn heads(&self) -> usize {
        self.heads.len()
    }

    /// Every head's tensors, still bfloat16, for a backend that takes the
    /// weight rather than the values.
    pub fn head_projections(&self) -> Vec<HeadPacked<'a>> {
        self.heads
            .iter()
            .enumerate()
            .map(|(head, tensors)| {
                tensors.packed(head, AttentionConfig::for_layer(&self.config, head))
            })
            .collect()
    }

    /// How many float32 values widening one head takes, which is what the CPU
    /// path decodes into and the whole of what it holds at once.
    pub fn scratch_floats(&self) -> usize {
        self.heads
            .iter()
            .map(|head| {
                let attention = &head.attention;
                head.input_proj.values()
                    + attention.q_proj.values()
                    + attention.k_proj.values()
                    + attention.v_proj.values()
                    + attention.r_proj.values()
                    + attention.o_proj.values()
                    + head.w13.values()
                    + head.w2.values()
            })
            .max()
            .unwrap_or(0)
    }

    /// `[rows, hidden]` of chained hidden state and the same of embeddings
    /// through head `head`.
    ///
    /// The head is stood up around whatever answers for its weights and dropped
    /// again, which is what lets the widened ones live in a buffer the next
    /// head overwrites.
    pub fn forward(
        &self,
        head: usize,
        cache: &mut DecoderCache,
        hidden: &[f32],
        embed: &[f32],
    ) -> Vec<f32> {
        let mut buffer = self.scratch.borrow_mut();
        if buffer.is_empty() && self.backend.is_none() {
            buffer.resize(self.scratch_floats(), 0.0);
        }
        let mut scratch = Scratch::new(&mut buffer);

        let tensors = &self.heads[head];
        let backend = self.backend.as_deref();
        let config = AttentionConfig::for_layer(&self.config, head);

        let mlp = Mlp::of(self, head, &mut scratch);
        let attention = self.attention(head, &mut scratch);
        let input_proj = match backend.and_then(|backend| backend.input_proj(head)) {
            Some(handed) => Handed::Backend(handed),
            None => Handed::Widened(DenseProjection::new(
                2 * self.config.hidden_size,
                widen(&tensors.input_proj, &mut scratch),
            )),
        };
        let weights = DecoderWeights {
            attention,
            input_layernorm: &tensors.input_layernorm,
            post_attention_layernorm: &tensors.post_attention_layernorm,
            attn_sconv: &tensors.attn_sconv,
            mlp_sconv: &tensors.mlp_sconv,
        };
        MtpHead::new(
            HeadNorms {
                hidden_norm: &tensors.hidden_norm,
                embed_norm: &tensors.embed_norm,
            },
            input_proj.projection(),
            DecoderLayer::new(config, weights, mlp.view()),
            self.config.rms_norm_eps,
        )
        .forward(
            head,
            cache,
            hidden,
            embed,
            backend.and_then(|backend| backend.device(head)),
        )
    }

    /// Take back the last `rows` timesteps of head `head`'s own state, wherever
    /// it is.
    ///
    /// **The cache is the caller's and this is the rest of it**, which is the
    /// same division [`ModelWeights::rewind`](crate::model::ModelWeights::rewind)
    /// makes for the stack: a backend running the head holds the keys it
    /// appended and the windows it advanced, and a head rewound on one side only
    /// is one whose position is one thing here and another there.
    pub fn rewind(&self, head: usize, rows: usize) {
        if let Some(backend) = self.backend.as_deref() {
            backend.rewind(head, rows);
        }
    }

    /// One head's attention tensors, its five projections widened into the
    /// call's scratch unless a backend answers for them.
    fn attention<'s>(&'s self, head: usize, scratch: &mut Scratch<'s>) -> AttentionWeights<'s> {
        let tensors = &self.heads[head];
        AttentionWeights {
            q_norm: &tensors.q_norm,
            k_norm: &tensors.k_norm,
            k_sconv: &tensors.k_sconv,
            v_sconv: &tensors.v_sconv,
            rel_proj: &tensors.rel_proj,
            projections: AttentionProjections::held_or(
                self.config.hidden_size,
                self.backend
                    .as_deref()
                    .and_then(|backend| backend.attention(head)),
                || {
                    let attention = &tensors.attention;
                    let [q_proj, k_proj, v_proj, r_proj, o_proj] = [
                        attention.q_proj,
                        attention.k_proj,
                        attention.v_proj,
                        attention.r_proj,
                        attention.o_proj,
                    ]
                    .map(|weight| widen(&weight, scratch));
                    DecodedProjections {
                        q_proj,
                        k_proj,
                        v_proj,
                        r_proj,
                        o_proj,
                    }
                },
            ),
        }
    }
}

/// The heads as the thing a speculative round asks for its guesses: a chain of
/// them over the rows the round committed.
///
/// # A round runs every head over every row it committed
///
/// Head `d` at row `j` consumes what head `d - 1` produced at row `j` and the
/// embedding of the token `d + 1` positions further on — so a round hands head
/// `d` the sequence `next ++ guesses` from offset `d`, which is committed
/// tokens for as long as they last and the chain's own guesses after that. The
/// guess a round *uses* is the last row's, which is the only one past the end
/// of the sequence.
///
/// **The earlier rows are run for the cache rather than for the guess**, and
/// they cost almost nothing to run: a head is bfloat16 weights read once
/// whatever the row count, so a round that accepted three tokens runs its heads
/// over four rows for about what one row costs. What that buys is that a head's
/// attention sees every position the sequence has, rather than only the ones
/// rounds happened to land on — which is the trajectory the acceptance study's
/// figures were measured against.
///
/// # The frontier row is run and taken back
///
/// At the last row, head `d` embeds a token no head has been proved right about
/// — the chain's own guess. So every head is rewound by one row at the end of a
/// round, and the next round runs that position again with the token the model
/// went on to produce. It costs one row of a head per round and keeps the heads
/// standing on what the model actually did.
///
/// Neither of those can cost a token: [`Proposer`] is asked for guesses and the
/// loop above verifies them.
pub struct MtpProposer<'a, W> {
    heads: &'a CheckpointHeads<'a>,
    generator: Generator<'a>,
    weights: &'a W,
    /// One decoder cache per head, able to give the frontier row back.
    caches: ModelCache,
    depth: usize,
    /// The row the round before this one ran and took back: the hidden state
    /// the stack produced there, and the token that follows it.
    carried: Option<Carried>,
    guesses: Vec<usize>,
    /// Rows accepted and rows guessed, which is acceptance as this run measures
    /// it — see [`MtpProposer::accepted`].
    accepted: Vec<usize>,
    proposed: Vec<usize>,
    /// Rounds this has been asked for guesses by, which is every round of the
    /// generation but the one it ended in.
    rounds: usize,
}

/// The frontier row of the round before this one, which is run again because
/// what it embedded was a guess.
struct Carried {
    hidden: Vec<f32>,
    next: usize,
}

impl<'a, W: ModelWeights> MtpProposer<'a, W> {
    /// The heads as a proposer of at most `depth` tokens a round.
    ///
    /// `generator` is what turns a head's hidden state into a token — the
    /// model's own final norm and `lm_head`, which is what makes a head's guess
    /// comparable to the model's answer — and `weights` is what the embedding
    /// rows come from.
    pub fn new(
        heads: &'a CheckpointHeads<'a>,
        generator: Generator<'a>,
        weights: &'a W,
        depth: usize,
    ) -> Self {
        assert!(
            depth <= heads.heads(),
            "{depth} tokens a round from {} heads",
            heads.heads()
        );
        Self {
            caches: ModelCache::speculating(heads.config(), FRONTIER),
            heads,
            generator,
            weights,
            depth,
            carried: None,
            guesses: Vec::new(),
            accepted: Vec::new(),
            proposed: Vec::new(),
            rounds: 0,
        }
    }

    /// How many rounds the generation took, which is what its tokens divide by
    /// to say what a round banked.
    ///
    /// One more than the rounds this was asked to guess in: every round ends by
    /// asking, except the one the generation ended in.
    pub fn rounds(&self) -> usize {
        self.rounds + 1
    }

    /// How many of the tokens this guessed at each depth the model went on to
    /// agree with, and how many it guessed at all.
    ///
    /// Kept because it is the number that decides the depth worth running, and
    /// it is a property of the workload rather than of the engine — the study
    /// measured 99.7% at depth 1 on enumeration and 44.9% on prose. A caller
    /// reports it; nothing here reads it.
    pub fn accepted(&self) -> (&[usize], &[usize]) {
        (&self.accepted, &self.proposed)
    }

    /// The same as a rate per depth, which is what a report prints and what a
    /// depth is chosen from.
    ///
    /// **Joint rather than marginal**, and it cannot be otherwise in an engine:
    /// a round whose first guess was rejected never learns what its second was
    /// worth, because the position that guess was about is not the position the
    /// model went to. The acceptance study's teacher-forced replay could
    /// measure both; what is here is what was banked, which is also what the
    /// speedup is made of.
    pub fn rates(&self) -> Vec<f64> {
        self.proposed
            .iter()
            .zip(&self.accepted)
            .map(|(asked, got)| match asked {
                0 => 0.0,
                asked => *got as f64 / *asked as f64,
            })
            .collect()
    }

    /// What the round's rows are, once the row taken back at the end of the
    /// last round is put in front of them.
    fn rows(&self, round: &Round<'_>) -> (Vec<f32>, Vec<usize>) {
        let mut hidden = Vec::with_capacity(round.hidden.len() + self.heads.config().hidden_size);
        let mut next = Vec::with_capacity(round.next.len() + 1);
        if let Some(carried) = &self.carried {
            hidden.extend_from_slice(&carried.hidden);
            next.push(carried.next);
        }
        hidden.extend_from_slice(round.hidden);
        next.extend_from_slice(round.next);
        (hidden, next)
    }

    /// What the model went on to do with the guesses of the round before this
    /// one, read off the tokens it committed.
    fn score(&mut self, next: &[usize]) {
        let banked = next.len().saturating_sub(1);
        for (depth, guess) in self.guesses.iter().enumerate() {
            if self.proposed.len() <= depth {
                self.proposed.push(0);
                self.accepted.push(0);
            }
            self.proposed[depth] += 1;
            self.accepted[depth] += usize::from(depth < banked && next[depth + 1] == *guess);
        }
    }
}

/// How many rows a round takes back from every head, which is the one whose
/// embedding was the chain's own guess.
///
/// Public because it is what a backend holding a head's state has to be wrapped
/// with: a window a rewind shifts has to have been built with room for it, and
/// that is decided where the heads are wrapped rather than here. See
/// [`HeadBackend::rewind`].
pub const FRONTIER: usize = 1;

impl<W: ModelWeights> Proposer for MtpProposer<'_, W> {
    fn depth(&self) -> usize {
        self.depth
    }

    /// **A round that asks for fewer than the last one leaves the heads past
    /// it where they are**, which is right for the one case that happens — a
    /// generation whose budget is running out, and which will not ask again —
    /// and would be wrong for a depth that went back up: those heads' caches
    /// would be missing the rows the shallower rounds ran. Nothing chooses a
    /// depth adaptively yet; whatever does has to run the heads it skipped or
    /// start their caches over.
    fn propose(&mut self, round: Round<'_>) -> &[usize] {
        self.rounds += 1;
        let (mut chained, mut tokens) = self.rows(&round);
        self.score(&tokens);
        let hidden = self.heads.config().hidden_size;
        let rows = tokens.len();
        assert_eq!(chained.len(), rows * hidden, "a hidden state per row");

        self.guesses.clear();
        for head in 0..round.depth {
            let embed = self
                .generator
                .model()
                .embeddings(&tokens[head..head + rows], self.weights);
            chained = self
                .heads
                .forward(head, self.caches.layer(head), &chained, &embed);
            let guess = self
                .generator
                .id_from_hidden(&chained[(rows - 1) * hidden..]);
            self.guesses.push(guess);
            tokens.push(guess);
        }

        // The frontier row again next time, against the token the model
        // produced rather than the one this chain guessed. Both sides of the
        // head's state, for the reason `CheckpointHeads::rewind` gives.
        for head in 0..round.depth {
            self.caches.layer(head).rewind(FRONTIER);
            self.heads.rewind(head, FRONTIER);
        }
        self.carried = Some(Carried {
            hidden: chained_row(round.hidden, hidden),
            next: *round.next.last().expect("a round commits a row"),
        });
        &self.guesses
    }
}

/// The last row of a round's hidden state, which is the row the next round runs
/// again.
fn chained_row(hidden: &[f32], width: usize) -> Vec<f32> {
    hidden[hidden.len() - width..].to_vec()
}

/// A head's projection, wherever it is multiplied.
enum Handed<'a> {
    Backend(&'a dyn Projection),
    Widened(DenseProjection<'a>),
}

impl Handed<'_> {
    fn projection(&self) -> &dyn Projection {
        match self {
            Self::Backend(handed) => *handed,
            Self::Widened(widened) => widened,
        }
    }
}

/// A head's feed-forward network, wherever its three multiplies run.
enum Mlp<'a> {
    Backend {
        dim: usize,
        hidden_dim: usize,
        handed: &'a dyn MlpProjections,
        global_scale: f32,
    },
    Widened {
        dim: usize,
        gate: &'a [f32],
        up: &'a [f32],
        down: &'a [f32],
        global_scale: f32,
    },
}

impl<'a> Mlp<'a> {
    fn of<'s>(heads: &'s CheckpointHeads<'a>, head: usize, scratch: &mut Scratch<'s>) -> Mlp<'s> {
        let tensors = &heads.heads[head];
        let (dim, hidden_dim) = (
            heads.config.hidden_size,
            heads.config.dense_intermediate_size,
        );
        let global_scale = tensors.global_scale;
        match heads
            .backend
            .as_deref()
            .and_then(|backend| backend.mlp(head))
        {
            Some(handed) => Mlp::Backend {
                dim,
                hidden_dim,
                handed,
                global_scale,
            },
            None => {
                let (gate, up) = Self::split(&tensors.w13, dim, scratch);
                Mlp::Widened {
                    dim,
                    gate,
                    up,
                    down: widen(&tensors.w2, scratch),
                    global_scale,
                }
            }
        }
    }

    /// The fused `w13_dn` widened into the two projections a SwiGLU multiplies
    /// through.
    ///
    /// **The two are interleaved, not stacked.** `_split_gate_up` reads the
    /// `[2 * dense, hidden]` tensor as `[dense, 2, hidden]` and takes the two
    /// along the middle axis, so row `2i` is the gate's row `i` and row `2i + 1`
    /// is the up's. Read as two halves instead, both projections are rows of the
    /// right shape drawn from the wrong places, and the layer still runs.
    fn split<'s>(w13: &Bf16<'_>, dim: usize, scratch: &mut Scratch<'s>) -> (&'s [f32], &'s [f32]) {
        let rows = w13.out_dim() / 2;
        let (gate, up) = (scratch.take(rows * dim), scratch.take(rows * dim));
        w13.widen_rows_into(0, 2, gate);
        w13.widen_rows_into(1, 2, up);
        (gate, up)
    }

    fn view(&self) -> LayerMlp<'_> {
        LayerMlp::Dense(match self {
            Self::Backend {
                dim,
                hidden_dim,
                handed,
                global_scale,
            } => DenseMlp::backend(*dim, *hidden_dim, *handed, *global_scale),
            Self::Widened {
                dim,
                gate,
                up,
                down,
                global_scale,
            } => DenseMlp::new(*dim, gate, up, down, *global_scale),
        })
    }
}

/// One bfloat16 weight widened into the call's scratch, which the next head
/// overwrites.
fn widen<'s>(weight: &Bf16<'_>, scratch: &mut Scratch<'s>) -> &'s [f32] {
    let run = scratch.take(weight.values());
    weight.widen_into(run);
    run
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::attention::LogScaling;
    use crate::config::Config;
    use crate::fixture::{self, LayerTensors, Stack, deviation};

    /// Two synthetic heads and the two calls mlx-vlm drove each of them with,
    /// from `just dump-mtp-fixture`.
    const FIXTURE: &str = "mtp.safetensors";

    /// One sliding head and one global one, which is the split
    /// `mtp_config.local_layer_ids` makes and the main stack's does not.
    const CASES: [&str; 2] = ["local", "global"];

    /// The synthetic heads are float32 end to end, so only summation order
    /// separates this from MLX — the same bound, for the same reason, as the
    /// layer this runs one of. A head runs a longer chain than a layer does,
    /// two norms and a projection further, and holds about as much of the bound
    /// in reserve as that layer: worst observed when this landed, 3.5e-7, a
    /// factor of nearly three. The weakest mutation these tests rely on
    /// catching — the two norms in front of the head exchanged — moves the
    /// answer by 6.9e-1, six decades above.
    const TOLERANCE: f32 = 1e-6;

    /// One synthetic head: the block it was built around, the three tensors in
    /// front of it, and what it produced over the two calls.
    struct Head {
        name: String,
        config: AttentionConfig,
        block: LayerTensors,
        hidden_norm: Vec<f32>,
        embed_norm: Vec<f32>,
        input_proj: Vec<f32>,
        prefill_out: Vec<f32>,
        continue_out: Vec<f32>,
    }

    /// The two sequences the fixture drove both heads with, hidden state and
    /// embedding apart.
    struct Calls {
        hidden: Vec<f32>,
        embed: Vec<f32>,
        continue_hidden: Vec<f32>,
        continue_embed: Vec<f32>,
    }

    impl Calls {
        fn load() -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, name));
            Self {
                hidden: of("hidden"),
                embed: of("embed"),
                continue_hidden: of("continue_hidden"),
                continue_embed: of("continue_embed"),
            }
        }
    }

    impl Head {
        fn load(case: &str) -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, &format!("{case}.{name}")));
            let recorded = of("config");
            let &[
                heads,
                kv_heads,
                head_dim,
                d_rel,
                sliding,
                _,
                eps,
                floor,
                alpha,
            ] = recorded.as_slice()
            else {
                panic!("{case}: config carries nine scalars, got {recorded:?}")
            };
            let block = LayerTensors::load(&ckpt, &format!("{case}.transformer_block"));

            Self {
                name: case.to_string(),
                config: AttentionConfig {
                    hidden: block.hidden(),
                    heads: heads as usize,
                    kv_heads: kv_heads as usize,
                    head_dim: head_dim as usize,
                    d_rel: d_rel as usize,
                    sliding: sliding as usize,
                    rms_norm_eps: eps,
                    log_scaling: (floor > 0.0).then(|| LogScaling::new(floor, alpha)),
                },
                hidden_norm: of("hidden_norm.weight"),
                embed_norm: of("embed_norm.weight"),
                input_proj: of("input_proj.weight"),
                prefill_out: of("prefill_out"),
                continue_out: of("continue_out"),
                block,
            }
        }

        fn all() -> Vec<Self> {
            CASES.iter().map(|case| Self::load(case)).collect()
        }

        fn hidden(&self) -> usize {
            self.block.hidden()
        }

        fn head<'a>(
            &'a self,
            norms: HeadNorms<'a>,
            input_proj: &'a DenseProjection<'a>,
        ) -> MtpHead<'a> {
            MtpHead::new(
                norms,
                input_proj,
                DecoderLayer::new(self.config, self.block.view(), self.block.mlp()),
                self.config.rms_norm_eps,
            )
        }

        fn norms(&self) -> HeadNorms<'_> {
            HeadNorms {
                hidden_norm: &self.hidden_norm,
                embed_norm: &self.embed_norm,
            }
        }

        fn projection(&self) -> DenseProjection<'_> {
            DenseProjection::new(2 * self.hidden(), &self.input_proj)
        }

        /// The prefill and the continuation against one cache, as the dump
        /// script drove the reference.
        fn forward(&self, norms: HeadNorms<'_>, calls: &Calls) -> (Vec<f32>, Vec<f32>) {
            let projection = self.projection();
            let head = self.head(norms, &projection);
            let cache = &mut head.cache();
            (
                head.forward(0, cache, &calls.hidden, &calls.embed, None),
                head.forward(
                    0,
                    cache,
                    &calls.continue_hidden,
                    &calls.continue_embed,
                    None,
                ),
            )
        }

        fn deviation(&self, (prefill, rest): &(Vec<f32>, Vec<f32>)) -> f32 {
            deviation(prefill, &self.prefill_out).max(deviation(rest, &self.continue_out))
        }
    }

    #[test]
    fn the_synthetic_heads_reproduce_mlx() {
        let calls = Calls::load();
        let mut worst = 0.0f32;
        for head in Head::all() {
            let deviation = head.deviation(&head.forward(head.norms(), &calls));
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                head.name
            );
            worst = worst.max(deviation);
        }
        assert!(worst > 0.0, "float32 summation order cannot agree exactly");
    }

    /// The two halves `input_proj` reads are not interchangeable, and nothing
    /// in its shape says which is which — a head fed them the other way round
    /// runs, and is what the acceptance study measured at 0.8% against 77.7%.
    ///
    /// Stated by exchanging the two norms rather than the two vectors, which is
    /// the same exchange on this side of them: a head that concatenated in the
    /// other order would normalise the hidden state with `embed_norm` too.
    #[test]
    fn concatenating_the_two_halves_the_other_way_round_changes_the_answer() {
        let calls = Calls::load();
        for head in Head::all() {
            let exchanged = HeadNorms {
                hidden_norm: &head.embed_norm,
                embed_norm: &head.hidden_norm,
            };
            let deviation = head.deviation(&head.forward(exchanged, &calls));
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                head.name
            );
        }
    }

    /// A backend that answers for a whole head, and what it was handed.
    ///
    /// Nothing here multiplies. What a device makes of a head is that device's
    /// own tests' question — `crates/inkling-metal` holds them — and what this
    /// stands in for is the seam: which rows cross it, what the block behind
    /// them is described as, and that the head answers with what came back.
    #[derive(Default)]
    struct Recorded {
        input: RefCell<Vec<f32>>,
        described: Cell<(usize, usize)>,
        answer: Vec<f32>,
    }

    impl HeadDevice for Recorded {
        fn run(&self, _: usize, _: &mut DecoderCache, step: HeadStep<'_>) -> Option<Passed> {
            *self.input.borrow_mut() = step.input.to_vec();
            self.described
                .set((step.block.queries, step.block.attention.x.len()));
            Some(Passed::Rows(self.answer.clone()))
        }
    }

    /// A backend that holds a head's weights and not the whole of it, which is
    /// the partial handover [`HeadBackend::device`] answers `None` for.
    struct Declined;

    impl HeadDevice for Declined {
        fn run(&self, _: usize, _: &mut DecoderCache, _: HeadStep<'_>) -> Option<Passed> {
            None
        }
    }

    /// **What crosses the seam a whole head runs behind**: the pair of normed
    /// rows `input_proj` reads, and a block described over rows this side does
    /// not hold.
    ///
    /// The rows are the two halves laid end to end and normed apart — the same
    /// value [`MtpHead::concatenated`] forms, which is asserted against a
    /// separate spelling of it rather than against itself. What the block is
    /// handed is a query count and *no rows at all*, because the rows it reads
    /// are what the projection in front of it produced and nothing on this side
    /// ever sees them.
    #[test]
    fn a_head_hands_a_device_the_normed_pair_and_a_block_over_rows_it_does_not_hold() {
        let calls = Calls::load();
        for head in Head::all() {
            let width = head.hidden();
            let rows = calls.hidden.len() / width;
            let projection = head.projection();
            let built = head.head(head.norms(), &projection);

            let device = Recorded {
                answer: (0..rows * width).map(|i| i as f32 / 8.0 - 1.0).collect(),
                ..Recorded::default()
            };
            let got = built.forward(
                0,
                &mut built.cache(),
                &calls.hidden,
                &calls.embed,
                Some(&device),
            );
            assert_eq!(
                got, device.answer,
                "{}: the head answered its own",
                head.name
            );
            assert_eq!(
                device.described.get(),
                (rows, 0),
                "{}: the block was described over {rows} rows and no values",
                head.name
            );

            let eps = head.config.rms_norm_eps;
            let normed = |rows: &[f32], weight: &[f32]| rms_norm(rows, weight, eps);
            let pair: Vec<f32> = normed(&calls.hidden, &head.hidden_norm)
                .chunks_exact(width)
                .zip(normed(&calls.embed, &head.embed_norm).chunks_exact(width))
                .flat_map(|(hidden, embed)| [hidden, embed].concat())
                .collect();
            assert_eq!(*device.input.borrow(), pair, "{}", head.name);
        }
    }

    /// A device that declines the head leaves it where it always ran, and the
    /// two answers are the same values rather than near ones: nothing about the
    /// arithmetic moved, only who was asked.
    #[test]
    fn a_head_a_device_declines_is_the_head_this_side_runs() {
        let calls = Calls::load();
        for head in Head::all() {
            let projection = head.projection();
            let built = head.head(head.norms(), &projection);
            let run = |device: Option<&dyn HeadDevice>| {
                built.forward(0, &mut built.cache(), &calls.hidden, &calls.embed, device)
            };
            assert_eq!(run(Some(&Declined)), run(None), "{}", head.name);
        }
    }

    /// A head carries a cache like any decoder layer, and the continuation
    /// reads it. A head handed a fresh one every round would still guess.
    #[test]
    fn the_continuation_reads_what_the_prefill_cached() {
        let calls = Calls::load();
        for head in Head::all() {
            let projection = head.projection();
            let built = head.head(head.norms(), &projection);
            let fresh = built.forward(
                0,
                &mut built.cache(),
                &calls.continue_hidden,
                &calls.continue_embed,
                None,
            );
            let deviation = deviation(&fresh, &head.continue_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                head.name
            );
        }
    }

    /// The heads' own local/global split, which is `mtp_config`'s and not the
    /// main stack's. Read from the checkpoint's own config, so a config whose
    /// two plans agreed could not settle it.
    #[test]
    fn a_head_reads_the_layer_plan_the_mtp_config_carries() {
        let config: Config = serde_json::from_str(crate::config::INKLING_SMALL).expect("parses");
        let mtp = config.mtp_config.expect("mtp_config");
        let heads = head_config(&config.text_config, &mtp);

        assert_eq!(heads.num_hidden_layers, 8);
        assert_eq!(heads.global_layers(), vec![1, 3]);
        assert_ne!(
            heads.global_layers(),
            config.text_config.global_layers(),
            "a config whose two plans agreed could not settle this"
        );
        assert!(
            (0..heads.num_hidden_layers).all(|head| heads.layer_is_dense(head)),
            "every head is dense"
        );
    }

    /// The config the fixture's heads were drawn against, in the shape a
    /// checkpoint states one — every width here is one of the fixture's own,
    /// and the fields a head does not read are the checkpoint's verbatim.
    const FIXTURE_CONFIG: &str = r#"{
      "text_config": {
        "model_max_length": 1048576, "hidden_size": 32, "num_hidden_layers": 2,
        "vocab_size": 64, "unpadded_vocab_size": null,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 8,
        "sliding_window_size": 4, "local_layer_ids": [0, 1],
        "d_rel": 3, "rel_extent": 6,
        "log_scaling_n_floor": null, "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06, "use_embed_norm": true,
        "logits_mup_width_multiplier": 1.0,
        "use_sconv": true, "sconv_kernel_size": 4,
        "dense_mlp_idx": 0, "dense_intermediate_size": 48, "intermediate_size": 16,
        "n_routed_experts": 16, "num_experts_per_tok": 3, "n_shared_experts": 2,
        "route_scale": 8.0, "use_gate_bias": true, "norm_after_topk": true,
        "shared_expert_sink": true
      },
      "mtp_config": { "num_nextn_predict_layers": 2, "chain_hidden_post_norm": false,
                      "local_layer_ids": [0] }
    }"#;

    fn fixture_config() -> Config {
        serde_json::from_str(FIXTURE_CONFIG).expect("the fixture's config parses")
    }

    /// The heads' layer plan, derived here, against the one the reference
    /// derived for the same head — `mtp_text_config` in the dump script, read
    /// back off the config tensor it recorded beside each head's output.
    ///
    /// The main stack's plan makes both of these heads sliding, so a port that
    /// read `local_layer_ids` from the wrong config would give head 1 a window
    /// it does not have and a band of the wrong width.
    #[test]
    fn the_derived_layer_plan_is_the_one_the_reference_derived() {
        let config = fixture_config();
        let plan = head_config(
            &config.text_config,
            config.mtp_config.as_ref().expect("an mtp_config"),
        );
        for (index, head) in Head::all().iter().enumerate() {
            let derived = AttentionConfig::for_layer(&plan, index);
            let recorded = head.config;
            assert_eq!(
                [
                    derived.hidden,
                    derived.heads,
                    derived.kv_heads,
                    derived.head_dim,
                    derived.d_rel,
                    derived.sliding
                ],
                [
                    recorded.hidden,
                    recorded.heads,
                    recorded.kv_heads,
                    recorded.head_dim,
                    recorded.d_rel,
                    recorded.sliding
                ],
                "{}",
                head.name
            );
            assert!(
                config.text_config.layer_is_sliding(index),
                "the main stack's plan makes head {index} sliding, so reading it \
                 instead has to show up in the case that is global here"
            );
        }
    }

    /// A head out of a checkpoint, against the same head handed its weights
    /// directly.
    ///
    /// What this pins is everything between the two: the twenty tensor names
    /// the heads ship under, which are the original's rather than mlx-vlm's;
    /// that they are read as bfloat16; and the de-interleave of `w13_dn`, whose
    /// even rows are the gate and whose odd rows are the up. Every one of those
    /// is a mapping a wrong version of would still stand a head up — `wq_du`
    /// and `wo_ud` are the same shape, and the two halves of `w13_dn` are the
    /// same shape as each other.
    ///
    /// Not exact, because the checkpoint written here holds the fixture's
    /// float32 weights rounded to bfloat16 once — half a quantum of each
    /// weight's own magnitude, which is 2^-9 relative, carried through a whole
    /// head. Worst observed when this landed: 1.4e-2. The weakest mapping
    /// mistake it has to catch — the gate and the up read as two halves rather
    /// than as the two interleaves — moves the answer by 2.8e-1, so this bound
    /// sits about twice the one and a tenth of the other.
    const CHECKPOINT_TOLERANCE: f32 = 3e-2;

    #[test]
    fn the_heads_a_checkpoint_holds_answer_what_their_tensors_answer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mtp.safetensors");
        write_heads(&path);

        let ckpt = crate::Checkpoint::open(&path).expect("the shard opens");
        let config = fixture_config();
        let heads = CheckpointHeads::open(
            &ckpt,
            &config.text_config,
            config.mtp_config.as_ref().expect("an mtp_config"),
        )
        .expect("the heads open");
        assert_eq!(heads.heads(), 2);

        let calls = Calls::load();
        for (index, head) in Head::all().iter().enumerate() {
            let cache = &mut DecoderCache::new(head.config, head.hidden(), 4);
            let got = heads.forward(index, cache, &calls.hidden, &calls.embed);
            let deviation = deviation(&got, &head.prefill_out);
            assert!(
                deviation <= CHECKPOINT_TOLERANCE,
                "{}: deviation {deviation:e}",
                head.name
            );
        }
    }

    /// What a head is called here, against what the checkpoint calls it.
    ///
    /// The heads kept the original's names where the quantiser rewrote the main
    /// stack's to mlx-vlm's, so this table is `_map_llm_layer` read backwards —
    /// and it is the thing under test. Spelled out on both sides rather than
    /// derived, because a rule that produced both names could produce both of
    /// them wrong.
    const NAMES: [(&str, &str); 17] = [
        ("hidden_norm.weight", "hidden_norm.weight"),
        ("embed_norm.weight", "embed_norm.weight"),
        ("input_proj.weight", "input_proj.weight"),
        (
            "transformer_block.input_layernorm.weight",
            "transformer_block.attn_norm.weight",
        ),
        (
            "transformer_block.post_attention_layernorm.weight",
            "transformer_block.mlp_norm.weight",
        ),
        (
            "transformer_block.attn_sconv.conv.weight",
            "transformer_block.attn_sconv.weight",
        ),
        (
            "transformer_block.mlp_sconv.conv.weight",
            "transformer_block.mlp_sconv.weight",
        ),
        (
            "transformer_block.self_attn.q_norm.weight",
            "transformer_block.attn.q_norm.weight",
        ),
        (
            "transformer_block.self_attn.k_norm.weight",
            "transformer_block.attn.k_norm.weight",
        ),
        (
            "transformer_block.self_attn.k_sconv.conv.weight",
            "transformer_block.attn.k_sconv.weight",
        ),
        (
            "transformer_block.self_attn.v_sconv.conv.weight",
            "transformer_block.attn.v_sconv.weight",
        ),
        (
            "transformer_block.self_attn.rel_proj",
            "transformer_block.attn.rel_logits_proj.proj",
        ),
        (
            "transformer_block.self_attn.q_proj.weight",
            "transformer_block.attn.wq_du.weight",
        ),
        (
            "transformer_block.self_attn.k_proj.weight",
            "transformer_block.attn.wk_dv.weight",
        ),
        (
            "transformer_block.self_attn.v_proj.weight",
            "transformer_block.attn.wv_dv.weight",
        ),
        (
            "transformer_block.self_attn.r_proj.weight",
            "transformer_block.attn.wr_du.weight",
        ),
        (
            "transformer_block.self_attn.o_proj.weight",
            "transformer_block.attn.wo_ud.weight",
        ),
    ];

    /// The fixture's heads written out as a checkpoint writes them: bfloat16,
    /// under the names the original ships, with the gate and the up fused back
    /// into the one interleaved tensor the checkpoint holds.
    fn write_heads(path: &std::path::Path) {
        let fixture = fixture::open(FIXTURE);
        let mut tensors: Vec<(String, Blob)> = Vec::new();
        for (index, case) in CASES.iter().enumerate() {
            let of = |name: &str| {
                let view = fixture::tensor(&fixture, &format!("{case}.{name}"));
                (fixture::f32s(&view), view.shape().to_vec())
            };
            let mut put = |name: &str, (values, shape): (Vec<f32>, Vec<usize>)| {
                tensors.push((format!("{MTP}.{index}.{name}"), Blob::bf16(&values, shape)))
            };

            for (mine, theirs) in NAMES {
                put(theirs, of(mine));
            }
            put(
                "transformer_block.mlp.global_scale",
                of("transformer_block.mlp.global_scale"),
            );
            put(
                "transformer_block.mlp.w2_md.weight",
                of("transformer_block.mlp.down_proj.weight"),
            );

            let (gate, shape) = of("transformer_block.mlp.gate_proj.weight");
            let (up, _) = of("transformer_block.mlp.up_proj.weight");
            let width = shape[1];
            put(
                "transformer_block.mlp.w13_dn.weight",
                (interleave(&gate, &up, width), vec![2 * shape[0], width]),
            );
        }

        safetensors::serialize_to_file(tensors.iter().map(|(name, blob)| (name, blob)), None, path)
            .expect("the shard is written");
    }

    /// Two projections' rows alternated, which is how `w13_dn` holds them.
    fn interleave(gate: &[f32], up: &[f32], width: usize) -> Vec<f32> {
        gate.chunks_exact(width)
            .zip(up.chunks_exact(width))
            .flat_map(|(gate, up)| [gate, up].concat())
            .collect()
    }

    /// A tensor as a checkpoint holds one: bfloat16, which is a float32 with
    /// its low sixteen mantissa bits dropped.
    struct Blob {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl Blob {
        fn bf16(values: &[f32], shape: Vec<usize>) -> Self {
            assert_eq!(
                values.len(),
                shape.iter().product::<usize>(),
                "{shape:?} against {} values",
                values.len()
            );
            Self {
                shape,
                data: values
                    .iter()
                    .flat_map(|value| {
                        ((value.to_bits() >> crate::checkpoint::BF16_SHIFT) as u16).to_le_bytes()
                    })
                    .collect(),
            }
        }
    }

    impl safetensors::View for &Blob {
        fn dtype(&self) -> crate::Dtype {
            crate::Dtype::BF16
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    /// **The heads driven through the loop they exist for**: a generation that
    /// chains them produces the tokens a generation that does not chain them
    /// produces.
    ///
    /// `speculation_changes_no_token` in [`crate::generate`] says this of the
    /// loop against proposers written to be right and wrong on cue; what this
    /// adds is the proposer nobody wrote by hand — the chain, its carried row,
    /// the embedding each head is handed, and the rewind at the end of every
    /// round, none of which the loop knows anything about.
    ///
    /// The heads here are the fixture's and the model is the synthetic stack's,
    /// which are two different models: what a head guesses is nonsense, and
    /// that is the point of running it. A proposer whose guesses are wrong is
    /// the case an engine has to be right about — 44.9% of them are, on prose —
    /// and the tokens are the model's either way.
    #[test]
    fn a_generation_that_chains_the_heads_produces_the_tokens_it_produces_without_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mtp.safetensors");
        write_heads(&path);
        let ckpt = crate::Checkpoint::open(&path).expect("the shard opens");
        let config = fixture_config();
        let heads = CheckpointHeads::open(
            &ckpt,
            &config.text_config,
            config.mtp_config.as_ref().expect("an mtp_config"),
        )
        .expect("the heads open");

        let stack = Stack::load();
        let head = stack.head();
        let generator = Generator::new(
            stack.model(),
            crate::LmHead::for_config(&stack.config),
            &head,
        );
        let ending = crate::generate::Ending {
            budget: TOKENS,
            eos: None,
        };
        let decoded = |depth: usize| {
            let cache = &mut ModelCache::speculating(&stack.config, depth);
            let mut proposer = MtpProposer::new(&heads, generator, &stack, depth);
            let mut tokens = Vec::new();
            generator.speculate(cache, &stack.ids, ending, &stack, &mut proposer, |id| {
                tokens.push(id);
                std::ops::ControlFlow::Continue(())
            });
            (tokens, proposer.rounds(), proposer.rates())
        };

        let (alone, rounds, rates) = decoded(0);
        assert_eq!(alone.len(), TOKENS);
        assert_eq!(rounds, TOKENS, "a round a token, which is decoding");
        assert!(rates.is_empty(), "a proposer of no depth guessed something");

        for depth in 1..=heads.heads() {
            let (tokens, rounds, rates) = decoded(depth);
            assert_eq!(tokens, alone, "chaining {depth} heads changed the tokens");
            assert_eq!(rates.len(), depth, "a rate per head asked");
            assert!(rounds <= TOKENS, "{rounds} rounds for {TOKENS} tokens");
        }
    }

    /// How many tokens the chained cases decode. Enough that a round reads what
    /// more than one round before it left — a head's convolution window is
    /// three inputs deep — and that the rewind at the end of every round has to
    /// have been right several times over.
    const TOKENS: usize = 6;

    /// The two sequences a head consumes are not the same tensor, and a head
    /// that read one of them twice would still run — the shapes are equal.
    #[test]
    fn a_head_reads_both_of_the_sequences_it_is_handed() {
        let calls = Calls::load();
        for head in Head::all() {
            let projection = head.projection();
            let built = head.head(head.norms(), &projection);
            for (what, hidden, embed) in [
                ("the hidden state twice", &calls.hidden, &calls.hidden),
                ("the embedding twice", &calls.embed, &calls.embed),
            ] {
                let got = built.forward(0, &mut built.cache(), hidden, embed, None);
                let deviation = deviation(&got, &head.prefill_out);
                assert!(
                    deviation > TOLERANCE,
                    "{}: reading {what} deviates by only {deviation:e}",
                    head.name
                );
            }
        }
    }
}
