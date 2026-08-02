//! An attention layer, and the scaled dot product at the middle of it.
//!
//! Two details of that middle step are the kind that produce fluent text and
//! wrong numbers. The scale is `1 / head_dim` and not the conventional
//! `1 / sqrt(head_dim)`, and the eight KV heads are shared across the
//! thirty-two query heads in contiguous blocks of four rather than by striding.
//! Either mistake runs, and neither is visible in anything but the tensors.
//!
//! The step is pinned to mlx-vlm by the `q_norm_out`, `k_norm_out`,
//! `v_sconv_out`, `mask` and `sdpa_out` tensors of
//! `reference/fixtures/layer_activations.safetensors`, which between them carry
//! everything it consumes and produces and involve no weights at all, and by
//! the same tensors of `long_activations.safetensors` for the widths a
//! committed capture cannot hold. The layer around it is pinned by
//! `reference/fixtures/attention.safetensors`, whose synthetic cases are
//! float32 throughout and reach the one branch a recorded forward pass cannot:
//! log scaling, which fires past position 128000.
//!
//! Everything here takes one sequence at a time: batching is the scheduler's,
//! and a batch of sequences is a loop over these.

use std::fmt::Debug;

use crate::config::TextConfig;
use crate::mask::{BandedMask, is_masked};
use crate::ops::{DenseProjection, Projection, rms_norm, softmax};
use crate::profile::{self, Op};
use crate::sconv::{ConvState, ShortConv};

/// The softmax attention step, over `[heads, queries, head_dim]` queries and
/// `[kv_heads, keys, head_dim]` keys and values.
#[derive(Debug, Clone, Copy)]
pub struct Sdpa {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Sdpa {
    /// `kv_heads` divides `heads`: grouped-query attention, in which each KV
    /// head is read by `heads / kv_heads` query heads.
    pub fn new(heads: usize, kv_heads: usize, head_dim: usize) -> Self {
        assert!(kv_heads > 0, "attention needs at least one KV head");
        assert_eq!(
            heads % kv_heads,
            0,
            "{heads} query heads do not divide into {kv_heads} groups"
        );
        assert!(head_dim > 0, "a head needs at least one channel");

        Self {
            heads,
            kv_heads,
            head_dim,
            scale: 1.0 / head_dim as f32,
        }
    }

    /// The logit scale, which `InklingAttention` sets to `1 / head_dim`.
    ///
    /// Not `1 / sqrt(head_dim)`. Attention under the conventional scale is still
    /// a distribution over the same keys, just a flatter one, so the model keeps
    /// generating and only the numbers say otherwise.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The three widths a step is over, which whoever answers for one has to be
    /// able to read off it — a backend holding the step holds these and not the
    /// [`AttentionConfig`] they were derived from.
    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The KV head that query head `head` reads.
    ///
    /// `mx.fast.scaled_dot_product_attention` repeats each KV head over a
    /// contiguous block of query heads, so heads 0..4 all read KV head 0. The
    /// other reading — query head `h` takes KV head `h % kv_heads` — pairs every
    /// query head with a key of the right shape and produces a plausible
    /// distribution over the wrong keys.
    pub fn kv_head(&self, head: usize) -> usize {
        head / (self.heads / self.kv_heads)
    }

    /// `q` is `[heads, queries, head_dim]`, `k` and `v` are `[kv_heads, keys,
    /// head_dim]` and `mask` is the additive `[heads, queries, keys]` the banded
    /// mask produced. Out comes `[heads, queries, head_dim]`.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], mask: &[f32]) -> Vec<f32> {
        let _timed = profile::scope(Op::Sdpa);
        let query_stride = self.heads * self.head_dim;
        assert_eq!(
            q.len() % query_stride,
            0,
            "{} values are not whole queries of {query_stride}",
            q.len()
        );
        let queries = q.len() / query_stride;

        let key_stride = self.kv_heads * self.head_dim;
        assert_eq!(
            k.len() % key_stride,
            0,
            "{} values are not whole keys of {key_stride}",
            k.len()
        );
        let keys = k.len() / key_stride;
        assert_eq!(v.len(), k.len(), "values against keys");
        assert_eq!(mask.len(), self.heads * queries * keys, "mask");

        let mut out = vec![0.0; q.len()];
        let mut weights = vec![0.0; keys];
        for head in 0..self.heads {
            let at = self.kv_head(head) * keys * self.head_dim;
            let (k, v) = (&k[at..], &v[at..]);
            for i in 0..queries {
                let at = (head * queries + i) * self.head_dim;
                let q = &q[at..at + self.head_dim];
                let mask = &mask[(head * queries + i) * keys..][..keys];

                for (j, (weight, mask)) in weights.iter_mut().zip(mask).enumerate() {
                    let key = &k[j * self.head_dim..][..self.head_dim];
                    *weight = dot(q, key) * self.scale + mask;
                }
                softmax(&mut weights);

                let out = &mut out[at..at + self.head_dim];
                for (weight, value) in weights.iter().zip(v.chunks_exact(self.head_dim)) {
                    for (out, value) in out.iter_mut().zip(value) {
                        *out += weight * value;
                    }
                }
            }
        }
        out
    }
}

/// The positional scaling a global layer applies to long contexts.
///
/// Below the floor `tau` is exactly 1 and the whole branch is inert, which is
/// why no recorded forward pass can reach it: the checkpoint's floor is 128000
/// tokens.
#[derive(Debug, Clone, Copy)]
pub struct LogScaling {
    floor: f32,
    alpha: f32,
}

impl LogScaling {
    /// The checkpoint's `log_scaling_n_floor` and `log_scaling_alpha`.
    ///
    /// `InklingAttention` leaves this off on a sliding layer whatever the
    /// config says, so a sliding layer has none rather than one that never
    /// fires.
    pub fn new(floor: f32, alpha: f32) -> Self {
        assert!(floor > 0.0, "a floor of {floor} would divide by zero");
        Self { floor, alpha }
    }

    /// `1 + alpha * ln(max(position / floor, 1))`.
    ///
    /// `position` counts from zero — query `i` of a call at cache offset `o`
    /// sits at `i + o` — and the reference counts the first token as 1, so the
    /// ratio is taken against `position + 1`.
    pub fn tau(&self, position: usize) -> f32 {
        1.0 + self.alpha * (((position + 1) as f32 / self.floor).max(1.0)).ln()
    }
}

/// The shapes and scalars `InklingAttention.__init__` derives for one layer.
/// Sliding and global layers read different fields of the config for all of
/// them, so which set a layer got is not recoverable from the tensors.
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    /// The width the layer maps from and back to, which is the model's hidden
    /// size.
    ///
    /// Here rather than read off `q_proj`, because it is what the five
    /// projections are *checked* against: a backend answering for them reports
    /// its own widths, and a width taken from one of the weights being checked
    /// would agree with itself.
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    /// The layer's window, or zero on a global layer.
    pub sliding: usize,
    pub rms_norm_eps: f32,
    pub log_scaling: Option<LogScaling>,
}

impl AttentionConfig {
    /// What `InklingAttention.__init__` reads for one layer.
    ///
    /// A sliding layer takes its head counts, its head width and its window
    /// from the `swa_` fields and a global layer from the plain ones. The two
    /// sets hold the same numbers in Inkling-Small, so nothing about that
    /// checkpoint can tell a port that read the wrong one — but they are
    /// separate fields for a reason, and mlx-vlm's own defaults give a sliding
    /// layer twice a global layer's KV heads.
    ///
    /// Log scaling is a global layer's alone, whatever the checkpoint sets
    /// `log_scaling_n_floor` to.
    pub fn for_layer(config: &TextConfig, layer: usize) -> Self {
        let sliding = config.layer_is_sliding(layer);
        let either = |swa, global| if sliding { swa } else { global };

        Self {
            hidden: config.hidden_size,
            heads: either(config.swa_num_attention_heads, config.num_attention_heads),
            kv_heads: either(config.swa_num_key_value_heads, config.num_key_value_heads),
            head_dim: either(config.swa_head_dim, config.head_dim),
            d_rel: config.d_rel,
            sliding: either(config.sliding_window_size, 0),
            rms_norm_eps: config.rms_norm_eps,
            log_scaling: config
                .log_scaling_n_floor
                .filter(|_| !sliding)
                .map(|floor| LogScaling::new(floor, config.log_scaling_alpha)),
        }
    }

    /// The width the key and value projections produce, which is the width of
    /// their convolutions and of every key the cache holds.
    pub fn kv_channels(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// The five projections one attention layer multiplies through, wherever their
/// weights live.
///
/// The seam [`crate::ops::Projection`] is, said once for the five together
/// rather than five times: they are the projections of one layer, they are
/// handed over as a set, and a backend that holds one holds all of them.
///
/// `Debug` because [`AttentionWeights`] derives it.
pub trait Projections: Debug {
    fn q_proj(&self) -> &dyn Projection;
    fn k_proj(&self) -> &dyn Projection;
    fn v_proj(&self) -> &dyn Projection;
    fn r_proj(&self) -> &dyn Projection;
    fn o_proj(&self) -> &dyn Projection;

    /// The four that consume the same normed hidden state, in one call.
    ///
    /// Asked for together because they *are* together: `q`, `k`, `v` and `r`
    /// read one input and nothing of each other, so a backend can answer all
    /// four before it is asked about any of them. Where a multiply is a loop
    /// that buys nothing and this default is the whole story; where a multiply
    /// is a dispatch it is four round trips against one.
    ///
    /// `o_proj` is not here and cannot be: it multiplies what the attention
    /// step produced out of these.
    fn qkvr(&self, x: &[f32]) -> Qkvr {
        Qkvr {
            q: self.q_proj().forward(x),
            k: self.k_proj().forward(x),
            v: self.v_proj().forward(x),
            r: self.r_proj().forward(x),
        }
    }

    /// The same four, over the layer's input layernorm applied to `x`.
    ///
    /// One step further out than [`Projections::qkvr`], and the same bargain:
    /// the normed hidden state is consumed by these four and by nothing else, so
    /// a backend that can produce it where they will read it never has to hand
    /// it back. Where a norm is a loop over a slice this default is the whole
    /// story; where it is a dispatch it is a value that stays on the device
    /// between two of them.
    ///
    /// It is the *first* of a layer's two norms and not both, because the second
    /// feeds an MLP whose router still multiplies its gate here — the seam moves
    /// as far as the next thing that has to come back, and no further.
    fn normed_qkvr(&self, x: &[f32], weight: &[f32], eps: f32) -> Qkvr {
        self.qkvr(&rms_norm(x, weight, eps))
    }

    /// The attention step and `o_proj`, in one call, `[queries, hidden]` out.
    ///
    /// The two are together for the reason [`Projections::normed_qkvr`]'s norm
    /// is together with the four that read it: `o_proj` multiplies what the step
    /// produced and nothing else does, so what the step produced is not
    /// something the CPU wants — it is something the next dispatch wants. The
    /// seam moves as far as the next thing that has to come back, which here is
    /// the layer's own residual add.
    ///
    /// It is also the first method here that hands over an operation rather than
    /// a weight. Every projection above multiplies against something the
    /// checkpoint stores; this multiplies activations against activations, so
    /// what a backend has to be able to do differently with it is not decline to
    /// decode a weight but decline to build a tensor: `inkling_metal::attention`
    /// derives each entry of the `[heads, queries, keys]` mask below where it
    /// scores the key it belongs to, and the mask is quadratic in the sequence
    /// where nothing else in a layer is.
    ///
    /// The default builds it, and is the oracle the other side is checked
    /// against.
    fn attend(&self, step: AttentionStep<'_>) -> Vec<f32> {
        self.o_proj().forward(&step.on_the_cpu())
    }

    /// The whole of one attention layer in one call — the input layernorm, the
    /// four projections, the two short convolutions, the two head norms, the
    /// attention step and `o_proj` — or `None` where this backend does not hold
    /// the layer's state.
    ///
    /// **This is where the seam stops moving outwards, and the reason it had to
    /// get this far is the cache.** [`Projections::normed_qkvr`] and
    /// [`Projections::attend`] are the two ends of a layer's attention and each
    /// is a value that never crosses back; what is between them —
    /// [`ShortConv`] on the key and the value, and an RMSNorm over each head of
    /// the query and the key — is four small operations whose inputs and outputs
    /// are read by nobody else either. A backend answering both ends and not the
    /// middle has to close and wait for the first before the second's inputs
    /// exist, which is a round trip a layer takes for four operations that are
    /// 2% of a step between them.
    ///
    /// And it cannot be closed by handing over those four alone, because two of
    /// them write to state that outlives the call: the convolutions' windows and
    /// the keys and values already attended over. Whoever runs them has to be
    /// whoever holds that state, which is why this takes the [`AttentionCache`]
    /// rather than the values in it — see [`AttentionCache::keys`], which is
    /// what a sequence still carries when a backend holds the rest.
    ///
    /// `None` rather than a default that does the work, because the work is
    /// [`Attention::attend`]'s and a default here would be a second spelling of
    /// it. The CPU path answers `None` and stays the oracle.
    fn layer(&self, cache: &mut AttentionCache, step: LayerStep<'_>) -> Option<Vec<f32>> {
        let _ = (cache, step);
        None
    }
}

/// Everything one attention layer runs, from the hidden state it is handed to
/// the `[queries, hidden]` `o_proj` returns.
///
/// The shapes are carried by the [`Sdpa`] and the band by the [`BandedMask`], as
/// [`AttentionStep`] carries them; what this adds is the four operations between
/// them and the two weights each needs. The state they read and write is not
/// here — it is the [`AttentionCache`] handed beside this.
#[derive(Debug, Clone, Copy)]
pub struct LayerStep<'a> {
    pub sdpa: Sdpa,
    pub mask: BandedMask<'a>,
    /// `[queries, hidden]`, before the layer's input layernorm — and empty
    /// where a backend already holds those rows, which is what
    /// [`Hidden::Carried`](crate::layer::Hidden::Carried) is. How many rows a
    /// call is is [`DecoderStep::queries`](crate::layer::DecoderStep::queries)
    /// either way.
    pub x: &'a [f32],
    /// The layer's input layernorm weight, or `None` where `x` arrives
    /// normalised already — which is what every recorded attention case is.
    pub input_layernorm: Option<&'a [f32]>,
    /// The `rms_norm_eps` all three of the norms here share.
    pub eps: f32,
    /// `[head_dim]`, over each head of the query.
    pub q_norm: &'a [f32],
    /// `[head_dim]`, over each head of the key — *after* its convolution.
    pub k_norm: &'a [f32],
    pub k_sconv: ShortConv<'a>,
    pub v_sconv: ShortConv<'a>,
    /// One `tau` per query, multiplying the query itself, or `None` on a layer
    /// with no log scaling.
    ///
    /// Separate from [`LayerStep::bias_taus`] because the reference applies the
    /// same `tau` to both and they are separable — see [`Attention::attend`],
    /// which drives them apart to show that neither alone reproduces mlx-vlm.
    pub q_taus: Option<&'a [f32]>,
    /// One `tau` per query, multiplying that query's biases, or `None`.
    pub bias_taus: Option<&'a [f32]>,
    /// Where this call's queries sit: query `i` is at absolute position
    /// `i + q_offset`.
    pub q_offset: usize,
}

/// What the four operations between a layer's projections and its attention
/// step produce.
#[derive(Debug, Clone)]
pub struct Convolved {
    /// `[heads, queries, head_dim]`, normed over each head's channels and
    /// scaled by the query's own `tau`.
    pub q: Vec<f32>,
    /// `[queries, kv_heads * head_dim]`, convolved and *then* normed over each
    /// head's channels, which is the order the reference caches it in.
    pub k: Vec<f32>,
    /// `[queries, kv_heads * head_dim]`, convolved and never normed.
    pub v: Vec<f32>,
}

impl LayerStep<'_> {
    /// The two short convolutions and the two head norms, on the CPU.
    ///
    /// Here rather than spelled out by each of the two callers, because they are
    /// the same four operations: [`Attention::attend`] runs them on the path a
    /// backend declined, and a backend that has taken [`Projections::layer`]
    /// runs them until it has a kernel for each. The windows they advance are
    /// `cache`'s; the keys and values they produce are the caller's to put
    /// wherever it holds them.
    pub fn convolved(&self, cache: &mut AttentionCache, projected: &Qkvr) -> Convolved {
        let (k_state, v_state) = cache.convolutions();
        let k = self.k_sconv.forward(k_state, &projected.k, None);
        let v = self.v_sconv.forward(v_state, &projected.v, None);
        let (q, k) = self.head_norms(&projected.q, &k);
        Convolved { q, k, v }
    }

    /// The two head norms alone, on the CPU: the query as the attention step
    /// reads it and the key as the cache holds it.
    ///
    /// `k` is the key *after* its convolution, which is where the reference
    /// norms it — before it would be a layer that still runs.
    ///
    /// Apart from [`LayerStep::convolved`] because the four operations move to a
    /// backend one at a time: a backend that has a kernel for the convolutions
    /// and not for the norms runs the convolutions there and asks for these.
    pub fn head_norms(&self, q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (heads, head_dim) = (self.sdpa.heads(), self.sdpa.head_dim());
        let norm = |x: &[f32], weight| rms_norm(x, weight, self.eps);
        let mut q = split_heads(&norm(q, self.q_norm), heads, head_dim);

        // Both tensors log scaling touches are `[heads, queries, stride]` with
        // the query minor, so cycling over the queries walks their rows in
        // order.
        if let Some(taus) = self.q_taus {
            for (q, tau) in q.chunks_exact_mut(head_dim).zip(taus.iter().cycle()) {
                for q in q {
                    *q *= tau;
                }
            }
        }
        (q, norm(k, self.k_norm))
    }
}

/// Everything the attention step reads, once the projections, the two short
/// convolutions and the two head norms have run.
///
/// The shape is carried by the [`Sdpa`] and the band by the [`BandedMask`],
/// which is what lets the default [`Projections::attend`] be the whole of the
/// step rather than a signature a backend has to be handed the config beside.
#[derive(Debug, Clone, Copy)]
pub struct AttentionStep<'a> {
    pub sdpa: Sdpa,
    pub mask: BandedMask<'a>,
    /// `[heads, queries, head_dim]`, with log scaling's `tau` already
    /// multiplied through it — see [`Attention::attend`], which scales the
    /// queries where they are formed and leaves the biases to whoever forms
    /// them.
    pub q: &'a [f32],
    /// `[kv_heads, keys, head_dim]`, the whole cached span.
    pub k: &'a [f32],
    pub v: &'a [f32],
    /// `[queries, heads, d_rel]` — query-major and head-minor, which is what
    /// `r_proj` produces and is the opposite of everything else here.
    pub rel: &'a [f32],
    /// One `tau` per query, multiplying the biases of that query, or `None` on
    /// a layer with no log scaling — which is every sliding layer and, below
    /// the floor, every global one. `None` rather than a row of ones so that
    /// the branch costs a pass over the mask only where it does something.
    pub taus: Option<&'a [f32]>,
    /// Where this call's queries sit: query `i` is at absolute position
    /// `i + q_offset`.
    pub q_offset: usize,
}

impl AttentionStep<'_> {
    /// How many keys the cached span holds, which is what the keys divide into
    /// rather than something the caller says again.
    pub fn keys(&self) -> usize {
        self.k.len() / (self.sdpa.kv_heads * self.sdpa.head_dim)
    }

    /// How many queries this call is over.
    pub fn queries(&self) -> usize {
        self.q.len() / (self.sdpa.heads * self.sdpa.head_dim)
    }

    /// The step here, `[queries, heads * head_dim]` out — the layout `o_proj`
    /// reads, and the whole of what [`Projections::attend`] hands over bar the
    /// multiply at the end of it.
    ///
    /// Named apart from that default so that a backend answering the step can
    /// be measured against the step rather than against the step and a
    /// projection: what would otherwise separate the two answers is a matmul
    /// each of them ran somewhere different, which is a question its own tests
    /// already settle.
    pub fn on_the_cpu(&self) -> Vec<f32> {
        let (heads, head_dim) = (self.sdpa.heads, self.sdpa.head_dim);
        let mut mask = self
            .mask
            .forward(self.rel, 1, heads, self.q_offset, self.keys());

        // An entry the mask ruled out keeps the constant it carries: scaling
        // -1e30 would overflow, and it rules the key out at any magnitude.
        if let Some(taus) = self.taus {
            for (row, tau) in mask.chunks_exact_mut(self.keys()).zip(taus.iter().cycle()) {
                for bias in row.iter_mut().filter(|entry| !is_masked(**entry)) {
                    *bias *= tau;
                }
            }
        }

        let out = self.sdpa.forward(self.q, self.k, self.v, &mask);
        merge_heads(&out, heads, head_dim)
    }
}

/// What the four projections of one call produced, each `[rows, out_dim]`.
///
/// Named rather than a tuple or an array, because `k` and `v` are the same
/// shape and are the pair an attention layer can silently exchange.
#[derive(Debug, Clone)]
pub struct Qkvr {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub r: Vec<f32>,
}

/// One attention layer's five projections as the checkpoint's weights decoded
/// to float32, `[out, in]` row-major and without bias — the layout `nn.Linear`
/// stores.
#[derive(Debug, Clone, Copy)]
pub struct DecodedProjections<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub r_proj: &'a [f32],
    pub o_proj: &'a [f32],
}

/// Where one attention layer's five projections multiply.
///
/// Held by value rather than borrowed, so that a caller holding the decoded
/// weights can hand over a view of them without owning anything else.
#[derive(Debug, Clone, Copy)]
pub struct AttentionProjections<'a>(Held<'a>);

#[derive(Debug, Clone, Copy)]
enum Held<'a> {
    /// Weights decoded to float32 and multiplied here, which is the path every
    /// other one is checked against.
    Decoded(Decoded<'a>),
    /// A backend holding the weights itself, which may never decode them.
    Backend(&'a dyn Projections),
}

/// [`DecodedProjections`] with each weight's widths settled, which is what a
/// [`Projections`] has to be able to answer.
#[derive(Debug, Clone, Copy)]
struct Decoded<'a> {
    q_proj: DenseProjection<'a>,
    k_proj: DenseProjection<'a>,
    v_proj: DenseProjection<'a>,
    r_proj: DenseProjection<'a>,
    o_proj: DenseProjection<'a>,
}

impl<'a> AttentionProjections<'a> {
    /// The five decoded, against a layer of `hidden` width.
    ///
    /// Four of them map from `hidden` and `o_proj` maps back to it, so its own
    /// width — the heads' channels merged — is what its length divides into
    /// rather than something the caller has to say again.
    pub fn decoded(hidden: usize, weights: DecodedProjections<'a>) -> Self {
        assert_eq!(
            weights.o_proj.len() % hidden,
            0,
            "{} o_proj weights are not whole rows into {hidden}",
            weights.o_proj.len()
        );
        Self(Held::Decoded(Decoded {
            q_proj: DenseProjection::new(hidden, weights.q_proj),
            k_proj: DenseProjection::new(hidden, weights.k_proj),
            v_proj: DenseProjection::new(hidden, weights.v_proj),
            r_proj: DenseProjection::new(hidden, weights.r_proj),
            o_proj: DenseProjection::new(weights.o_proj.len() / hidden, weights.o_proj),
        }))
    }

    /// The five wherever a backend holds them.
    pub fn backend(projections: &'a dyn Projections) -> Self {
        Self(Held::Backend(projections))
    }

    /// Whichever of the two the backend's answer called for, with `values`
    /// reached only where it did not answer.
    ///
    /// A closure rather than five slices, because producing them is the cost
    /// this exists to skip: on the arm a backend answers, nothing of the layer
    /// is decoded or widened at all. Both stacks in this engine ask the same
    /// question here and differ only in what that closure does — the main
    /// stack's decodes MXFP4 into the pass's scratch, a head's widens bfloat16
    /// into its own.
    pub fn held_or(
        hidden: usize,
        backend: Option<&'a dyn Projections>,
        values: impl FnOnce() -> DecodedProjections<'a>,
    ) -> Self {
        match backend {
            Some(handed) => Self::backend(handed),
            None => Self::decoded(hidden, values()),
        }
    }

    fn held(&self) -> &dyn Projections {
        match &self.0 {
            Held::Decoded(decoded) => decoded,
            Held::Backend(projections) => *projections,
        }
    }
}

impl Projections for Decoded<'_> {
    fn q_proj(&self) -> &dyn Projection {
        &self.q_proj
    }

    fn k_proj(&self) -> &dyn Projection {
        &self.k_proj
    }

    fn v_proj(&self) -> &dyn Projection {
        &self.v_proj
    }

    fn r_proj(&self) -> &dyn Projection {
        &self.r_proj
    }

    fn o_proj(&self) -> &dyn Projection {
        &self.o_proj
    }
}

impl Projections for AttentionProjections<'_> {
    fn q_proj(&self) -> &dyn Projection {
        self.held().q_proj()
    }

    fn k_proj(&self) -> &dyn Projection {
        self.held().k_proj()
    }

    fn v_proj(&self) -> &dyn Projection {
        self.held().v_proj()
    }

    fn r_proj(&self) -> &dyn Projection {
        self.held().r_proj()
    }

    fn o_proj(&self) -> &dyn Projection {
        self.held().o_proj()
    }

    /// Delegated rather than left to the default, which is the one method here
    /// that has to be: the default would ask this for its four projections and
    /// multiply through them one at a time, which is exactly what a backend
    /// that overrode `qkvr` said not to do.
    fn qkvr(&self, x: &[f32]) -> Qkvr {
        self.held().qkvr(x)
    }

    /// Delegated for the same reason, one step further out: the default would
    /// normalise here and hand the result down, which is the crossing a backend
    /// that overrode this said it did not need.
    fn normed_qkvr(&self, x: &[f32], weight: &[f32], eps: f32) -> Qkvr {
        self.held().normed_qkvr(x, weight, eps)
    }

    /// And delegated for the same reason again: the default would build the
    /// mask here and multiply `o_proj` through this, which is the tensor and the
    /// crossing a backend that overrode this said it needed neither of.
    fn attend(&self, step: AttentionStep<'_>) -> Vec<f32> {
        self.held().attend(step)
    }

    /// Delegated for the last time, and the one whose default is a refusal
    /// rather than a computation: a backend that holds the layer's state says so
    /// here, and one that does not leaves [`Attention::attend`] to run it.
    fn layer(&self, cache: &mut AttentionCache, step: LayerStep<'_>) -> Option<Vec<f32>> {
        self.held().layer(cache, step)
    }
}

/// One attention layer's tensors: the five projections wherever they multiply,
/// a per-head-channel RMSNorm weight for the queries and one for the keys, a
/// kernel per short convolution, and the mask's `[d_rel, rel_extent]`
/// projection.
#[derive(Debug, Clone, Copy)]
pub struct AttentionWeights<'a> {
    pub projections: AttentionProjections<'a>,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub k_sconv: &'a [f32],
    pub v_sconv: &'a [f32],
    pub rel_proj: &'a [f32],
}

/// Everything one sequence carries between calls: the two short convolutions'
/// windows, and the keys and values already attended over.
///
/// The keys and values are held in the layout their projections produced,
/// `[keys, kv_heads * head_dim]`, so a call appends to them and splits into
/// heads once at the end. What is cached is the *normed* key and the
/// *convolved* value, which is where the reference caches them too.
///
/// **Where they are held is not the same question as how many there are**, and
/// this is the only place the two come apart. A backend answering
/// [`Projections::layer`] keeps the span in device memory across steps rather
/// than handing it back to be copied over again, so on that path the two vectors
/// below stay empty and [`AttentionCache::seen`] is the whole of what a sequence
/// carries here. That count is what says whether a resident span belongs to
/// *this* sequence: a cache that has seen nothing is a sequence starting, and
/// one that disagrees with what a backend is holding is two sequences
/// interleaved through one layer.
#[derive(Debug, Clone)]
pub struct AttentionCache {
    k_sconv: ConvState,
    v_sconv: ConvState,
    keys: Vec<f32>,
    values: Vec<f32>,
    /// How many keys the sequence has seen, wherever they are.
    cached: usize,
}

impl AttentionCache {
    /// The state a sequence starts from: empty convolution windows and no keys.
    ///
    /// Built from the config rather than from an [`Attention`], because a stack
    /// of forty-two layers allocates its caches before it has decoded a single
    /// weight — and at Inkling-Small's size, standing a layer up to ask it for
    /// one would mean decoding 176 MB of projections to learn two integers.
    pub fn new(config: AttentionConfig, kernel_size: usize) -> Self {
        let channels = config.kv_channels();
        Self {
            k_sconv: ConvState::new(channels, kernel_size),
            v_sconv: ConvState::new(channels, kernel_size),
            keys: Vec::new(),
            values: Vec::new(),
            cached: 0,
        }
    }

    /// How many keys the sequence has seen, which is also where this call's
    /// queries sit.
    ///
    /// Named apart from the vector above because it answers a different question
    /// from the one that vector holds the answer to: this counts the sequence's
    /// keys wherever they are, and stays right on the path where that vector is
    /// empty.
    pub fn seen(&self) -> usize {
        self.cached
    }

    /// The two convolution windows inside attention, for a backend answering
    /// [`Projections::layer`] that runs the convolutions itself.
    ///
    /// Handed out mutably because a convolution's window *is* written by the
    /// call that reads it — see [`ShortConv::forward`] — and there is no reading
    /// of it that leaves it alone.
    pub fn convolutions(&mut self) -> (&mut ConvState, &mut ConvState) {
        (&mut self.k_sconv, &mut self.v_sconv)
    }

    /// Record `rows` keys and values a backend appended to a span it holds
    /// itself.
    ///
    /// The count and nothing else, because the values are not here. What this
    /// keeps true is that [`AttentionCache::keys`] answers the same question on
    /// both paths.
    pub fn appended(&mut self, rows: usize) {
        self.cached += rows;
    }
}

/// One attention layer, from the hidden state its input layernorm produced to
/// the one `o_proj` returns.
#[derive(Debug, Clone, Copy)]
pub struct Attention<'a> {
    config: AttentionConfig,
    sdpa: Sdpa,
    mask: BandedMask<'a>,
    k_sconv: ShortConv<'a>,
    v_sconv: ShortConv<'a>,
    hidden: usize,
    weights: AttentionWeights<'a>,
}

impl<'a> Attention<'a> {
    pub fn new(config: AttentionConfig, weights: AttentionWeights<'a>) -> Self {
        let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);
        let sdpa = Sdpa::new(heads, kv_heads, head_dim);

        let projections = &weights.projections;
        let hidden = config.hidden;
        for (name, projection, from, to) in [
            ("q_proj", projections.q_proj(), hidden, heads * head_dim),
            ("k_proj", projections.k_proj(), hidden, kv_heads * head_dim),
            ("v_proj", projections.v_proj(), hidden, kv_heads * head_dim),
            ("r_proj", projections.r_proj(), hidden, heads * config.d_rel),
            ("o_proj", projections.o_proj(), heads * head_dim, hidden),
        ] {
            assert_eq!(projection.in_dim(), from, "the width {name} maps from");
            assert_eq!(projection.out_dim(), to, "the width {name} maps to");
        }
        assert_eq!(weights.q_norm.len(), head_dim, "q_norm");
        assert_eq!(weights.k_norm.len(), head_dim, "k_norm");

        Self {
            config,
            sdpa,
            mask: BandedMask::new(config.d_rel, weights.rel_proj, config.sliding),
            k_sconv: ShortConv::new(kv_heads * head_dim, weights.k_sconv),
            v_sconv: ShortConv::new(kv_heads * head_dim, weights.v_sconv),
            hidden,
            weights,
        }
    }

    /// What `InklingAttention.__init__` derived for this layer.
    pub fn config(&self) -> AttentionConfig {
        self.config
    }

    /// The state a sequence starts from, for this layer's own shape.
    pub fn cache(&self) -> AttentionCache {
        AttentionCache::new(self.config, self.k_sconv.kernel_size())
    }

    /// `[queries, hidden]` in and out, continuing from `cache` and leaving this
    /// call's keys, values and convolution windows behind in it.
    ///
    /// `norm` is the layer's input layernorm weight, applied to `x` on the way
    /// into the four projections — and `None` says `x` arrives normalised
    /// already, which is what every recorded attention case is: the reference's
    /// capture starts at `input_layernorm_out`, and the synthetic cases were
    /// driven through `InklingAttention` directly.
    ///
    /// Here rather than in the layer above for the reason
    /// [`Projections::normed_qkvr`] gives: the normed state is these four
    /// projections' input and nothing else's, so whoever multiplies them should
    /// be the one who decides where it is formed.
    pub fn forward(&self, cache: &mut AttentionCache, x: &[f32], norm: Option<&[f32]>) -> Vec<f32> {
        let log_scaling = self.config.log_scaling;
        self.attend(
            cache,
            x,
            norm,
            QueryOffset::Cached,
            log_scaling,
            log_scaling,
        )
    }

    /// The scaling this layer applies to a call of `queries` queries whose first
    /// sits at `offset`.
    ///
    /// Handed out because [`LayerStep`] borrows it: a caller building the step
    /// itself — see [`Attention::step`] — has to hold this for as long as the
    /// step lives.
    pub fn taus(&self, offset: usize, queries: usize) -> Taus {
        let log_scaling = self.config.log_scaling;
        self.scaled(offset, queries, log_scaling, log_scaling)
    }

    /// [`Attention::taus`], with log scaling's two multiplications named apart —
    /// see [`Attention::attend`].
    fn scaled(
        &self,
        offset: usize,
        queries: usize,
        on_queries: Option<LogScaling>,
        on_biases: Option<LogScaling>,
    ) -> Taus {
        // The queries are scaled where they are formed and the biases are not,
        // because the biases are not formed here: whoever answers the step forms
        // them, and a `tau` per query is what says by how much.
        let taus = |log: Option<LogScaling>| -> Option<Vec<f32>> {
            log.map(|log| (0..queries).map(|i| log.tau(offset + i)).collect())
        };
        Taus {
            q: taus(on_queries),
            bias: taus(on_biases),
        }
    }

    /// The whole of this layer's attention as a value, without running any of
    /// it.
    ///
    /// What [`Attention::forward`] hands [`Projections::layer`], handed out so
    /// that a caller running the whole *decoder* layer can encode this beside
    /// what follows it — which it cannot do through [`Attention::forward`],
    /// because the value that would come back is the one the layer's residual
    /// add wants and nothing else does.
    pub fn step<'s>(
        &'s self,
        x: &'s [f32],
        input_layernorm: Option<&'s [f32]>,
        taus: &'s Taus,
        q_offset: usize,
    ) -> LayerStep<'s> {
        LayerStep {
            sdpa: self.sdpa,
            mask: self.mask,
            x,
            input_layernorm,
            eps: self.config.rms_norm_eps,
            q_norm: self.weights.q_norm,
            k_norm: self.weights.k_norm,
            k_sconv: self.k_sconv,
            v_sconv: self.v_sconv,
            q_taus: taus.q.as_deref(),
            bias_taus: taus.bias.as_deref(),
            q_offset,
        }
    }

    /// [`Attention::forward`], with the query offset and log scaling's two
    /// multiplications named apart.
    ///
    /// The reference applies the same `tau` to the queries and to the mask's
    /// unmasked entries: both or neither. They are separable, and either one
    /// alone is a plausible misreading that leaves a model which still
    /// generates, so the tests drive this to show that neither alone reproduces
    /// mlx-vlm. [`QueryOffset`] is there for the same reason.
    fn attend(
        &self,
        cache: &mut AttentionCache,
        x: &[f32],
        input_layernorm: Option<&[f32]>,
        query_offset: QueryOffset,
        on_queries: Option<LogScaling>,
        on_biases: Option<LogScaling>,
    ) -> Vec<f32> {
        assert_eq!(
            x.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.hidden
        );
        let head_dim = self.sdpa.head_dim;
        let queries = x.len() / self.hidden;
        let offset = query_offset.of(cache.seen());

        let taus = self.scaled(offset, queries, on_queries, on_biases);
        let projections = &self.weights.projections;
        let step = self.step(x, input_layernorm, &taus, offset);
        if let Some(out) = projections.layer(cache, step) {
            return out;
        }

        let projected = match input_layernorm {
            Some(weight) => projections.normed_qkvr(x, weight, self.config.rms_norm_eps),
            None => projections.qkvr(x),
        };
        let Convolved { q, k, v } = step.convolved(cache, &projected);

        cache.keys.extend(k);
        cache.values.extend(v);
        cache.appended(queries);

        projections.attend(AttentionStep {
            sdpa: self.sdpa,
            mask: self.mask,
            q: &q,
            k: &split_heads(&cache.keys, self.sdpa.kv_heads, head_dim),
            v: &split_heads(&cache.values, self.sdpa.kv_heads, head_dim),
            rel: &projected.r,
            taus: step.bias_taus,
            q_offset: offset,
        })
    }
}

/// One `tau` per query, which log scaling multiplies the query and its biases
/// by — or nothing at all on a layer without it.
///
/// A value of its own because [`LayerStep`] borrows the two vectors and a caller
/// that builds the step has to hold them. They are separate because the
/// reference applies the same `tau` to both and they are separable: either one
/// alone is a plausible misreading that leaves a model which still generates.
#[derive(Debug, Clone, Default)]
pub struct Taus {
    q: Option<Vec<f32>>,
    bias: Option<Vec<f32>>,
}

/// Where a call's queries sit in the sequence, which is `InklingAttention`'s
/// `q_offset` and is read off the cache rather than passed in.
///
/// Two things index by it: the relative-position bias, whose entries are
/// `(i + offset) - j`, and log scaling's `tau`. They are one decision — the
/// reference derives both from one `cache.offset` — so a port that never
/// threaded it gets both wrong, and this names that port rather than half of it.
///
/// A prefill starts from an empty cache, so the two arms agree on every call
/// that has one. They part from the first token decoded after it.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum QueryOffset {
    /// How many keys the cache already holds, which is how many tokens the
    /// sequence has seen.
    Cached,
    /// Zero, whatever the cache holds.
    Ignored,
}

impl QueryOffset {
    fn of(&self, cached: usize) -> usize {
        match self {
            Self::Cached => cached,
            Self::Ignored => 0,
        }
    }
}

/// `[rows, heads * head_dim]` — the layout a projection produces — as `[heads,
/// rows, head_dim]`, the layout attention reads.
///
/// `InklingAttention` writes this as a reshape into `[B, L, H, D]` followed by a
/// transpose to `[B, H, L, D]`, inline at each of its three call sites.
pub fn split_heads(x: &[f32], heads: usize, head_dim: usize) -> Vec<f32> {
    let stride = heads * head_dim;
    assert_eq!(
        x.len() % stride,
        0,
        "{} values are not whole rows of {stride}",
        x.len()
    );
    let mut out = Vec::with_capacity(x.len());
    for head in 0..heads {
        for row in x.chunks_exact(stride) {
            out.extend_from_slice(&row[head * head_dim..][..head_dim]);
        }
    }
    out
}

/// The inverse of [`split_heads`]: `[heads, rows, head_dim]` back to `[rows,
/// heads * head_dim]`, which is what `o_proj` reads.
pub fn merge_heads(x: &[f32], heads: usize, head_dim: usize) -> Vec<f32> {
    let stride = heads * head_dim;
    assert_eq!(
        x.len() % stride,
        0,
        "{} values are not whole rows of {stride}",
        x.len()
    );
    let rows = x.len() / stride;

    let mut out = Vec::with_capacity(x.len());
    for row in 0..rows {
        for head in 0..heads {
            out.extend_from_slice(&x[(head * rows + row) * head_dim..][..head_dim]);
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, LONG_ACTIVATIONS, deviation};
    use crate::mask::MASKED;

    /// Synthetic layers and the sequences mlx-vlm ran through them, from
    /// `just dump-attention-fixture`.
    const FIXTURE: &str = "attention.safetensors";

    /// Each case, and the weight set it was built on. `global_inert` and
    /// `global_scaled` share theirs and differ only in their floor, which is
    /// what makes their disagreement log scaling and nothing else.
    const SYNTHETIC: [(&str, &str); 3] = [
        ("sliding", "sliding"),
        ("global_inert", "global"),
        ("global_scaled", "global"),
    ];

    /// The synthetic cases are float32 end to end, so only summation order
    /// separates this from MLX — the same bound, for the same reason, as the
    /// RMSNorm, MLP, sconv and mask cases. They run a longer chain than any of
    /// those, though: five projections, two convolutions, two norms, a mask and
    /// a softmax, each feeding the next, so the same bound holds less of it in
    /// reserve. Worst observed when this landed: 4.1e-7, a factor of two in
    /// hand, against a weakest mutation of 2.9e-2 five decades above.
    const TOLERANCE: f32 = 1e-6;

    /// The recorded step ran in bfloat16 on trained numbers, so its output is
    /// this computation rounded once on the way out, and the tolerance is that
    /// quantum rather than an arithmetic one: 2^-9 = 2.0e-3 relative, measured
    /// against the tensor's largest value rather than each entry's own
    /// magnitude, which puts it slightly above the ceiling. The same bound, for
    /// the same reason, as the trained masks.
    ///
    /// Worst observed when this landed: 2.8e-3 on layer 5 of the committed
    /// capture, and 2.9e-3 on layer 0 of the long one against 2.0e-3 on its
    /// layer 5, whose rows reduce over a thousand keys rather than eight — the
    /// width buys the bound no error worth naming. The weakest mutation
    /// these tests rely on catching, the conventional `1/sqrt(head_dim)` scale,
    /// moves the answer by 1.0 — over two decades above this bound.
    const TRAINED_TOLERANCE: f32 = 6e-3;

    /// One captured layer's attention step: everything it consumed, and what
    /// mlx-vlm produced from it.
    struct Case {
        name: String,
        sdpa: Sdpa,
        queries: usize,
        keys: usize,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        mask: Vec<f32>,
        want: Vec<f32>,
    }

    impl Case {
        /// `v` is the one input the fixture holds in the projection's own
        /// layout: `k_norm_out` and `q_norm_out` were taken as attention passed
        /// them to the kernel, already split into heads, and `v_sconv_out` was
        /// taken one step earlier.
        fn load(layer: usize) -> Self {
            Self::from(&fixture::open(ACTIVATIONS), layer)
        }

        fn from(activations: &Checkpoint, layer: usize) -> Self {
            let of = |name: &str| fixture::layer_tensor(activations, layer, name);

            let q = of("q_norm_out");
            let k = of("k_norm_out");
            let &[_, heads, queries, head_dim] = q.shape() else {
                panic!("q_norm_out is [batch, heads, queries, head_dim]")
            };
            let &[_, kv_heads, keys, _] = k.shape() else {
                panic!("k_norm_out is [batch, kv_heads, keys, head_dim]")
            };

            Self {
                name: format!("layer{layer}"),
                sdpa: Sdpa::new(heads, kv_heads, head_dim),
                queries,
                keys,
                q: fixture::f32s(&q),
                k: fixture::f32s(&k),
                v: split_heads(&fixture::f32s(&of("v_sconv_out")), kv_heads, head_dim),
                mask: fixture::f32s(&of("mask")),
                want: fixture::f32s(&of("sdpa_out")),
            }
        }

        fn all() -> Vec<Self> {
            CAPTURED_LAYERS.iter().copied().map(Self::load).collect()
        }

        /// The long capture's steps, one per layer it recorded, each cut down to
        /// its last `queries` rows. `None` when it has not been generated.
        ///
        /// Cut because the step is quadratic and this one is not: 1280 queries
        /// over 1280 keys is 13 GMAC of scalar float32, where the rows kept are
        /// the ones whose oldest keys are the far side of the band. The keys and
        /// values stay whole — what is under test is a query attending over all
        /// of them.
        fn long(queries: usize) -> Option<Vec<Self>> {
            let activations = fixture::try_open(LONG_ACTIVATIONS)?;
            Some(
                CAPTURED_LAYERS
                    .iter()
                    .filter(|&&layer| fixture::holds_layer(&activations, layer))
                    .map(|&layer| Self::from(&activations, layer).tail(queries))
                    .collect(),
            )
        }

        /// The last `queries` rows of the step. `q`, the mask and the recorded
        /// output are all query-minor within a head, so each is cut per head.
        fn tail(&self, queries: usize) -> Self {
            assert!(queries <= self.queries, "{}: {queries} rows", self.name);
            let cut = |values: &[f32], stride: usize| -> Vec<f32> {
                values
                    .chunks_exact(self.queries * stride)
                    .flat_map(|head| head[(self.queries - queries) * stride..].to_vec())
                    .collect()
            };
            Self {
                name: format!("{}.tail{queries}", self.name),
                sdpa: self.sdpa,
                queries,
                keys: self.keys,
                q: cut(&self.q, self.sdpa.head_dim),
                k: self.k.clone(),
                v: self.v.clone(),
                mask: cut(&self.mask, self.keys),
                want: cut(&self.want, self.sdpa.head_dim),
            }
        }

        fn forward(&self) -> Vec<f32> {
            self.sdpa.forward(&self.q, &self.k, &self.v, &self.mask)
        }

        fn deviation(&self, got: &[f32]) -> f32 {
            deviation(got, &self.want)
        }

        /// The same keys and values with one KV head per query head, so a
        /// grouping rule becomes an ordinary gather and can be written out.
        fn ungrouped(&self, kv_head: impl Fn(usize) -> usize) -> (Sdpa, Vec<f32>, Vec<f32>) {
            let span = self.keys * self.sdpa.head_dim;
            let gather = |kv: &[f32]| {
                (0..self.sdpa.heads)
                    .flat_map(|head| kv[kv_head(head) * span..][..span].to_vec())
                    .collect()
            };
            (
                Sdpa::new(self.sdpa.heads, self.sdpa.heads, self.sdpa.head_dim),
                gather(&self.k),
                gather(&self.v),
            )
        }
    }

    #[test]
    fn the_captured_layers_reproduce_the_reference_attention() {
        let mut worst = 0.0f32;
        for case in Case::all() {
            let deviation = case.deviation(&case.forward());
            assert!(
                deviation <= TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
            worst = worst.max(deviation);
        }
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean bfloat16 rounding vanished"
        );
    }

    /// How many of the long capture's queries the step below runs. Enough that
    /// the softmax widths span a range rather than sit at a point, and few
    /// enough that a quadratic step stays a test rather than a benchmark.
    const LONG_TAIL: usize = 64;

    /// The widest band any captured layer's mask is built over, which is a
    /// global layer's — a sliding layer's ends at its 512-token window. A number
    /// here because the step never sees either: attention is handed a finished
    /// mask, and all this has to know is that the long capture's key span
    /// outruns both.
    const REL_EXTENT: usize = 1024;

    /// The attention step over a span past the band, where the mask it is handed
    /// carries window caps on a sliding layer and exact zeros on a global one,
    /// and every softmax runs over more than a thousand keys rather than eight.
    /// Skips when the long capture has not been generated.
    ///
    /// What this adds to the eight-token cases is width. Which entries the mask
    /// holds is `mask.rs`'s to settle; what is left here is whether a step still
    /// reproduces the reference when the row it reduces over is two orders of
    /// magnitude longer — a softmax that lost its shift, or a masked key that
    /// stopped being negligible against a thousand others.
    #[test]
    fn the_long_capture_reproduces_the_reference_attention_past_the_band() {
        let Some(cases) = Case::long(LONG_TAIL) else {
            return;
        };
        assert!(
            !cases.is_empty(),
            "the long capture holds none of the captured layers"
        );

        let mut worst = 0.0f32;
        for case in &cases {
            assert!(
                case.keys > REL_EXTENT,
                "{}: {} keys do not outrun a band of {REL_EXTENT}",
                case.name,
                case.keys
            );
            let deviation = case.deviation(&case.forward());
            assert!(
                deviation <= TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
            worst = worst.max(deviation);
        }
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean bfloat16 rounding vanished"
        );
    }

    /// `InklingAttention` sets `scale = 1.0 / self.head_dim`. Writing the
    /// conventional form is the single most likely way to port this wrong.
    #[test]
    fn the_logit_scale_is_one_over_head_dim_not_its_square_root() {
        for case in Case::all() {
            let head_dim = case.sdpa.head_dim as f32;
            assert_eq!(case.sdpa.scale(), 1.0 / head_dim);

            let conventional = Sdpa {
                scale: head_dim.sqrt().recip(),
                ..case.sdpa
            };
            let got = conventional.forward(&case.q, &case.k, &case.v, &case.mask);
            let deviation = case.deviation(&got);
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: deviation {deviation:e}",
                case.name
            );
        }
    }

    /// Each KV head serves a contiguous block of query heads: with 32 query
    /// heads over 8 KV heads, query heads 0..4 all read KV head 0.
    ///
    /// Stated against an attention with one KV head per query head, which has no
    /// grouping to get wrong: gathering the KV heads under the rule and handing
    /// them over ungrouped has to reproduce the grouped answer exactly, and the
    /// striding rule has to miss.
    #[test]
    fn each_kv_head_serves_a_contiguous_block_of_query_heads() {
        for case in Case::all() {
            let group = case.sdpa.heads / case.sdpa.kv_heads;
            assert_eq!((case.sdpa.heads, case.sdpa.kv_heads, group), (32, 8, 4));

            let (ungrouped, k, v) = case.ungrouped(|head| head / group);
            assert_eq!(
                ungrouped.forward(&case.q, &k, &v, &case.mask),
                case.forward(),
                "{}: blocks of {group}",
                case.name
            );

            let (ungrouped, k, v) = case.ungrouped(|head| head % case.sdpa.kv_heads);
            let deviation = case.deviation(&ungrouped.forward(&case.q, &k, &v, &case.mask));
            assert!(
                deviation > TRAINED_TOLERANCE,
                "{}: striding deviates by {deviation:e}",
                case.name
            );
        }
    }

    /// A key the mask rules out contributes nothing to its query, whatever the
    /// value at that key holds.
    ///
    /// This is the test that fails if the mask is added to the softmax's output
    /// rather than to its input: post-softmax, a masked key carries a weight of
    /// about `-1e30` instead of one of about zero, and the value behind it
    /// dominates the answer rather than vanishing from it.
    #[test]
    fn a_masked_key_cannot_reach_its_query() {
        for case in Case::all() {
            let head_dim = case.sdpa.head_dim;
            let last = case.keys - 1;
            let masked: Vec<usize> = (0..case.queries)
                .filter(|i| is_masked(case.mask[i * case.keys + last]))
                .collect();
            assert!(!masked.is_empty(), "{}: nothing is masked", case.name);

            let mut v = case.v.clone();
            for kv in 0..case.sdpa.kv_heads {
                for value in &mut v[(kv * case.keys + last) * head_dim..][..head_dim] {
                    *value = 1e6;
                }
            }

            let want = case.forward();
            let got = case.sdpa.forward(&case.q, &case.k, &v, &case.mask);
            for head in 0..case.sdpa.heads {
                for &i in &masked {
                    let at = (head * case.queries + i) * head_dim;
                    assert_eq!(
                        got[at..at + head_dim],
                        want[at..at + head_dim],
                        "{}: head {head} query {i}",
                        case.name
                    );
                }
            }
            assert_ne!(got, want, "{}: the value went unread", case.name);
        }
    }

    /// A row with no key it may attend to still has to leave softmax with finite
    /// numbers, which is why the mask carries a magnitude rather than an
    /// infinity and why the softmax shifts by the row's largest entry.
    #[test]
    fn a_row_with_every_key_masked_stays_finite() {
        let (heads, head_dim, keys) = (1, 4, 3);
        let sdpa = Sdpa::new(heads, heads, head_dim);
        let got = sdpa.forward(
            &vec![1.0; heads * head_dim],
            &vec![1.0; heads * keys * head_dim],
            &vec![2.0; heads * keys * head_dim],
            &vec![MASKED; heads * keys],
        );
        assert_eq!(got, vec![2.0; heads * head_dim]);
    }

    /// The two reshapes attention brackets itself with. Written out for one
    /// small tensor, because a transpose that is wrong in both directions is a
    /// round trip that still holds.
    #[test]
    fn splitting_and_merging_heads_are_the_transposes_they_claim() {
        let (rows, heads, head_dim) = (2, 3, 2);
        let x: Vec<f32> = (0..(rows * heads * head_dim) as u16)
            .map(f32::from)
            .collect();

        // [rows, heads, head_dim] read down the rows of each head in turn.
        assert_eq!(
            split_heads(&x, heads, head_dim),
            [0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]
        );
        assert_eq!(
            merge_heads(&split_heads(&x, heads, head_dim), heads, head_dim),
            x
        );
    }

    /// One synthetic layer's weights, owned so the borrowed [`AttentionWeights`]
    /// can be handed out repeatedly.
    struct Tensors {
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

    impl Tensors {
        fn load(ckpt: &Checkpoint, kind: &str) -> Self {
            let of = |name: &str| fixture::f32s(&fixture::tensor(ckpt, &format!("{kind}.{name}")));
            Self {
                q_proj: of("q_proj.weight"),
                k_proj: of("k_proj.weight"),
                v_proj: of("v_proj.weight"),
                r_proj: of("r_proj.weight"),
                o_proj: of("o_proj.weight"),
                q_norm: of("q_norm.weight"),
                k_norm: of("k_norm.weight"),
                k_sconv: of("k_sconv.conv.weight"),
                v_sconv: of("v_sconv.conv.weight"),
                rel_proj: of("rel_proj"),
            }
        }

        fn decoded(&self) -> DecodedProjections<'_> {
            DecodedProjections {
                q_proj: &self.q_proj,
                k_proj: &self.k_proj,
                v_proj: &self.v_proj,
                r_proj: &self.r_proj,
                o_proj: &self.o_proj,
            }
        }

        fn view(&self, hidden: usize) -> AttentionWeights<'_> {
            AttentionWeights {
                projections: AttentionProjections::decoded(hidden, self.decoded()),
                q_norm: &self.q_norm,
                k_norm: &self.k_norm,
                k_sconv: &self.k_sconv,
                v_sconv: &self.v_sconv,
                rel_proj: &self.rel_proj,
            }
        }
    }

    /// One synthetic case: the layer `InklingAttention` was built as, the two
    /// sequences it was driven with, and what it produced for each.
    struct Layer {
        name: String,
        config: AttentionConfig,
        weights: Tensors,
        x: Vec<f32>,
        continue_x: Vec<f32>,
        prefill_out: Vec<f32>,
        continue_out: Vec<f32>,
    }

    impl Layer {
        /// `config` is the `[heads, kv_heads, head_dim, d_rel, sliding,
        /// rel_extent, rms_norm_eps, log_floor, log_alpha]` the dump script
        /// recorded — everything `InklingAttention.__init__` derived from the
        /// config and the layer index, which the shapes do not carry. A
        /// `log_floor` of zero is the `None` a sliding layer gets.
        fn load(case: &str, kind: &str) -> Self {
            let ckpt = fixture::open(FIXTURE);
            let of = |name: &str| fixture::f32s(&fixture::tensor(&ckpt, name));
            let recorded = of(&format!("{case}.config"));
            let &[
                heads,
                kv_heads,
                head_dim,
                d_rel,
                sliding,
                rel_extent,
                eps,
                floor,
                alpha,
            ] = recorded.as_slice()
            else {
                panic!("{case}: config carries nine scalars, got {recorded:?}")
            };

            let weights = Tensors::load(&ckpt, kind);
            assert_eq!(
                weights.rel_proj.len(),
                (d_rel * rel_extent) as usize,
                "{case}: rel_proj against its recorded extent"
            );

            Self {
                name: case.to_string(),
                config: AttentionConfig {
                    // The dump script records what `__init__` derived, and the
                    // hidden size is not among it: the synthetic case's weights
                    // are what say how wide the layer is.
                    hidden: weights.q_proj.len() / (heads * head_dim) as usize,
                    heads: heads as usize,
                    kv_heads: kv_heads as usize,
                    head_dim: head_dim as usize,
                    d_rel: d_rel as usize,
                    sliding: sliding as usize,
                    rms_norm_eps: eps,
                    log_scaling: (floor > 0.0).then(|| LogScaling::new(floor, alpha)),
                },
                weights,
                x: of("x"),
                continue_x: of("continue_x"),
                prefill_out: of(&format!("{case}.prefill_out")),
                continue_out: of(&format!("{case}.continue_out")),
            }
        }

        fn all() -> Vec<Self> {
            SYNTHETIC
                .iter()
                .map(|(case, kind)| Self::load(case, kind))
                .collect()
        }

        fn attention(&self) -> Attention<'_> {
            Attention::new(self.config, self.weights.view(self.hidden()))
        }

        /// The prefill alone, under weights this layer's own may have been
        /// mutated into.
        fn run(&self, weights: AttentionWeights<'_>) -> Vec<f32> {
            let attention = Attention::new(self.config, weights);
            attention.forward(&mut attention.cache(), &self.x, None)
        }

        /// The prefill and the continuation, against one cache, as the dump
        /// script drove the reference.
        fn forward(&self) -> (Vec<f32>, Vec<f32>) {
            self.with(self.config.log_scaling, self.config.log_scaling)
        }

        fn with(
            &self,
            on_queries: Option<LogScaling>,
            on_biases: Option<LogScaling>,
        ) -> (Vec<f32>, Vec<f32>) {
            self.at(QueryOffset::Cached, on_queries, on_biases)
        }

        fn at(
            &self,
            query_offset: QueryOffset,
            on_queries: Option<LogScaling>,
            on_biases: Option<LogScaling>,
        ) -> (Vec<f32>, Vec<f32>) {
            let attention = self.attention();
            let cache = &mut attention.cache();
            (
                attention.attend(cache, &self.x, None, query_offset, on_queries, on_biases),
                attention.attend(
                    cache,
                    &self.continue_x,
                    None,
                    query_offset,
                    on_queries,
                    on_biases,
                ),
            )
        }

        /// The worst of the two calls' deviations. The continuation is where the
        /// cache's offset enters, so a prefill that matched alone would say
        /// nothing about decoding.
        fn deviation(&self, (prefill, rest): &(Vec<f32>, Vec<f32>)) -> f32 {
            deviation(prefill, &self.prefill_out).max(deviation(rest, &self.continue_out))
        }

        /// Which absolute positions the two calls put queries at.
        fn positions(&self) -> std::ops::Range<usize> {
            0..(self.x.len() + self.continue_x.len()) / self.hidden()
        }

        fn hidden(&self) -> usize {
            self.config.hidden
        }
    }

    fn synthetic(case: &str) -> Layer {
        Layer::all()
            .into_iter()
            .find(|layer| layer.name == case)
            .unwrap_or_else(|| panic!("no {case} case"))
    }

    /// A [`Projections`] that is not this module's — the five answered by
    /// something the layer cannot see inside, which is the whole of what a
    /// backend is from here.
    #[derive(Debug)]
    struct Handed<'a> {
        five: AttentionProjections<'a>,
        /// How many times the layer asked for the four that share an input
        /// together, which is what says it did not ask for them one at a time.
        qkvr: Cell<usize>,
        /// And how many times it asked for the step and `o_proj` together,
        /// which is what says it did not multiply `o_proj` itself.
        attended: Cell<usize>,
        /// The shape of the last step it was handed, which is what says the
        /// tensors arrived in the layouts the seam names.
        step: Cell<Option<(usize, usize, usize)>>,
    }

    impl<'a> Handed<'a> {
        fn new(five: AttentionProjections<'a>) -> Self {
            Self {
                five,
                qkvr: Cell::new(0),
                attended: Cell::new(0),
                step: Cell::new(None),
            }
        }
    }

    impl Projections for Handed<'_> {
        fn qkvr(&self, x: &[f32]) -> Qkvr {
            self.qkvr.set(self.qkvr.get() + 1);
            self.five.qkvr(x)
        }

        fn attend(&self, step: AttentionStep<'_>) -> Vec<f32> {
            self.attended.set(self.attended.get() + 1);
            self.step
                .set(Some((step.queries(), step.keys(), step.q_offset)));
            self.five.attend(step)
        }

        fn q_proj(&self) -> &dyn Projection {
            self.five.q_proj()
        }

        fn k_proj(&self) -> &dyn Projection {
            self.five.k_proj()
        }

        fn v_proj(&self) -> &dyn Projection {
            self.five.v_proj()
        }

        fn r_proj(&self) -> &dyn Projection {
            self.five.r_proj()
        }

        fn o_proj(&self) -> &dyn Projection {
            self.five.o_proj()
        }
    }

    /// The five in the order [`DecodedProjections`] names them.
    const PROJECTIONS: [&str; 5] = ["q_proj", "k_proj", "v_proj", "r_proj", "o_proj"];

    /// `five` with the projection at `index` replaced by as much of `zeroed` as
    /// it was long, so that a layer run through it cannot be reading the
    /// original.
    fn without<'a>(
        mut five: DecodedProjections<'a>,
        index: usize,
        zeroed: &'a [f32],
    ) -> DecodedProjections<'a> {
        let slot = match index {
            0 => &mut five.q_proj,
            1 => &mut five.k_proj,
            2 => &mut five.v_proj,
            3 => &mut five.r_proj,
            _ => &mut five.o_proj,
        };
        *slot = &zeroed[..slot.len()];
        five
    }

    /// The seam, stated where nothing else states it: a layer whose five
    /// projections are answered by a backend is the same layer.
    ///
    /// Exact rather than bounded, because the backend here multiplies the same
    /// weights through the same [`linear`](crate::ops::linear) — what changes is
    /// only who was asked. A backend whose arithmetic differs is what
    /// `inkling-metal`'s own tests bound.
    #[test]
    fn a_layer_whose_projections_come_from_a_backend_is_the_same_layer() {
        for layer in Layer::all() {
            let hidden = layer.hidden();
            let handed = Handed::new(layer.weights.view(hidden).projections);
            let mut weights = layer.weights.view(hidden);
            weights.projections = AttentionProjections::backend(&handed);

            assert_eq!(
                layer.run(weights),
                layer.run(layer.weights.view(hidden)),
                "{}",
                layer.name
            );
        }
    }

    /// And each of the five it answers with is one the layer multiplies through,
    /// rather than one the layer went back to the weights for. Zeroed, each has
    /// to move the answer — which is also what says none of the five is
    /// unreachable.
    #[test]
    fn every_projection_a_backend_answers_with_is_one_the_layer_multiplies() {
        for layer in Layer::all() {
            let hidden = layer.hidden();
            let whole = layer.run(layer.weights.view(hidden));
            let zeroed = vec![0.0; layer.weights.o_proj.len().max(layer.weights.q_proj.len())];

            for (index, name) in PROJECTIONS.iter().enumerate() {
                let handed = Handed::new(AttentionProjections::decoded(
                    hidden,
                    without(layer.weights.decoded(), index, &zeroed),
                ));
                let mut weights = layer.weights.view(hidden);
                weights.projections = AttentionProjections::backend(&handed);

                assert_ne!(layer.run(weights), whole, "{}: {name}", layer.name);
            }
        }
    }

    /// The four that share an input are asked for in one call, not four.
    ///
    /// This is the whole of what [`Projections::qkvr`] is: on a backend where a
    /// multiply is a dispatch, four calls are four round trips to a device that
    /// could have answered all of them at once. A layer that went back to
    /// `q_proj()` and the rest one at a time would produce exactly the same
    /// answer, which is why the count is what is asserted.
    #[test]
    fn a_layer_asks_a_backend_for_the_four_that_share_an_input_together() {
        for layer in Layer::all() {
            let hidden = layer.hidden();
            let handed = Handed::new(layer.weights.view(hidden).projections);
            let mut weights = layer.weights.view(hidden);
            weights.projections = AttentionProjections::backend(&handed);

            layer.run(weights);
            assert_eq!(handed.qkvr.get(), 1, "{}: one call a forward", layer.name);
        }
    }

    /// And the attention step is asked for once, with `o_proj` inside it.
    ///
    /// The same bargain [`Projections::qkvr`] strikes, at the other end of the
    /// layer: what the step produces is read by `o_proj` and by nothing else, so
    /// a layer that took the step's answer back and then asked for `o_proj`
    /// would be crossing a seam nothing needed crossed. It would also produce
    /// exactly the same tensor, which is why the count is what is asserted.
    ///
    /// The shape is asserted with it, because a step handed the wrong span is
    /// the mistake the count cannot see: a continuation attends over the whole
    /// cache and not over its own three tokens, and its queries sit at the
    /// offset the prefill left.
    #[test]
    fn a_layer_asks_a_backend_for_the_attention_step_with_o_proj_inside_it() {
        for layer in Layer::all() {
            let hidden = layer.hidden();
            let handed = Handed::new(layer.weights.view(hidden).projections);
            let mut weights = layer.weights.view(hidden);
            weights.projections = AttentionProjections::backend(&handed);

            let attention = Attention::new(layer.config, weights);
            let cache = &mut attention.cache();
            let prefill = attention.forward(cache, &layer.x, None);
            assert_eq!(handed.attended.get(), 1, "{}", layer.name);

            let prefilled = layer.x.len() / hidden;
            assert_eq!(
                handed.step.get(),
                Some((prefilled, prefilled, 0)),
                "{}: the prefill",
                layer.name
            );

            let rest = attention.forward(cache, &layer.continue_x, None);
            let decoded = layer.continue_x.len() / hidden;
            assert_eq!(handed.attended.get(), 2, "{}", layer.name);
            assert_eq!(
                handed.step.get(),
                Some((decoded, prefilled + decoded, prefilled)),
                "{}: the continuation",
                layer.name
            );

            assert_eq!((prefill, rest), layer.forward(), "{}", layer.name);
        }
    }

    /// A projection of the wrong width is refused where it is handed over, which
    /// is one prefill before it would be discovered. Both of its widths are
    /// checked against the config rather than against each other, because a
    /// backend answering with another layer's `k_proj` reports widths that agree
    /// with themselves.
    #[test]
    #[should_panic(expected = "the width k_proj maps to")]
    fn a_projection_that_is_not_the_layers_shape_is_refused() {
        let layer = synthetic("sliding");
        let hidden = layer.hidden();
        let narrow = vec![0.0; layer.weights.k_proj.len() - hidden];
        let mut five = layer.weights.decoded();
        five.k_proj = &narrow;

        let handed = Handed::new(AttentionProjections::decoded(hidden, five));
        let mut weights = layer.weights.view(hidden);
        weights.projections = AttentionProjections::backend(&handed);
        Attention::new(layer.config, weights);
    }

    #[test]
    fn the_synthetic_layers_reproduce_mlx() {
        for layer in Layer::all() {
            let deviation = layer.deviation(&layer.forward());
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// A sliding layer has no log scaling at all, however the checkpoint sets
    /// `log_scaling_n_floor`: `InklingAttention` reads that field only on a
    /// global layer. The dump script builds the sliding case from the same
    /// floor the `global_scaled` case keeps.
    #[test]
    fn a_sliding_layer_has_no_log_scaling() {
        assert!(synthetic("sliding").config.log_scaling.is_none());
        assert!(synthetic("global_scaled").config.log_scaling.is_some());
    }

    /// Below the floor `tau` is exactly 1, so the whole branch is inert and the
    /// layer is the same layer without it — bit for bit, not within a tolerance.
    ///
    /// This is why the committed activations cannot pin log scaling. The
    /// checkpoint's floor is 128000 tokens; every recorded forward pass runs
    /// here, where a port with no log scaling at all agrees exactly.
    #[test]
    fn log_scaling_is_inert_below_its_floor() {
        let layer = synthetic("global_inert");
        let log = layer.config.log_scaling.expect("a floor");
        for position in layer.positions() {
            assert_eq!(log.tau(position), 1.0, "position {position}");
        }
        assert_eq!(layer.with(None, None), layer.forward());
    }

    /// Past the floor `tau` multiplies the queries **and** the mask's unmasked
    /// entries. Either alone leaves a layer that still runs and still attends,
    /// and neither reproduces the reference.
    #[test]
    fn log_scaling_multiplies_the_queries_and_the_biases_together() {
        let layer = synthetic("global_scaled");
        let log = layer.config.log_scaling.expect("a floor");
        assert!(
            layer.positions().any(|position| log.tau(position) > 1.0),
            "no position clears the floor"
        );

        for (what, halved) in [
            ("neither", layer.with(None, None)),
            ("the queries alone", layer.with(Some(log), None)),
            ("the biases alone", layer.with(None, Some(log))),
        ] {
            let deviation = layer.deviation(&halved);
            assert!(deviation > TOLERANCE, "{what}: deviation {deviation:e}");
        }
    }

    /// `global_inert` and `global_scaled` are one weight set and one input under
    /// two floors, so what separates their recorded outputs is log scaling and
    /// nothing else — which is what makes the test above a test of log scaling
    /// rather than of the layer around it.
    #[test]
    fn the_two_global_cases_differ_only_by_their_floor() {
        let (inert, scaled) = (synthetic("global_inert"), synthetic("global_scaled"));
        assert_eq!(inert.weights.q_proj, scaled.weights.q_proj);
        assert_eq!(inert.weights.rel_proj, scaled.weights.rel_proj);
        assert!(deviation(&inert.prefill_out, &scaled.prefill_out) > TOLERANCE);
    }

    /// `q_offset` is where a call's queries sit, and it is the one thing about a
    /// cache that a prefill cannot expose: a prefill starts from an empty cache,
    /// so a layer that never advanced the offset computes the same tensor as one
    /// that does — to the bit, not within a tolerance. The first token attended
    /// after it is where the two part, and this asserts both halves, because
    /// either alone would leave the wrong impression of what the mistake costs.
    ///
    /// What it costs is not a small drift. A query at position `i + offset`
    /// indexes the band at backward distances `i + offset - j`; pinned to zero,
    /// every key but the very first sits at a negative distance, which the mask
    /// reads as a position that has not happened yet and rules out. So the
    /// continuation attends over almost nothing.
    #[test]
    fn ignoring_the_query_offset_shows_only_once_a_prefill_is_behind_it() {
        for layer in Layer::all() {
            let log = layer.config.log_scaling;
            let (prefill, rest) = layer.at(QueryOffset::Ignored, log, log);

            let agreed = deviation(&prefill, &layer.prefill_out);
            assert!(
                agreed <= TOLERANCE,
                "{}: the prefill deviates by {agreed:e}",
                layer.name
            );
            assert_eq!(prefill, layer.forward().0, "{}", layer.name);

            let deviation = deviation(&rest, &layer.continue_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// The continuation reads the cache: the keys and values of the prefill, and
    /// the last `kernel_size - 1` inputs of each short convolution. A layer that
    /// dropped it would still answer, over its own three tokens.
    #[test]
    fn the_continuation_attends_over_the_cached_prefill() {
        for layer in Layer::all() {
            let attention = layer.attention();
            let fresh = attention.forward(&mut attention.cache(), &layer.continue_x, None);
            let deviation = deviation(&fresh, &layer.continue_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// The two norms are interchangeable, and no fixture can tell them apart.
    ///
    /// RMSNorm divides a row by its own RMS and then multiplies by its weight,
    /// so a logit contracts the two weights elementwise — `sum_d q_d k_d gq_d
    /// gk_d` — and depends on the pair rather than on which is which. Nothing
    /// downstream sees either one again: the values carry no norm at all.
    ///
    /// Stated as a test because it bounds what this fixture settles. A port
    /// that exchanged them is not wrong, and one that dropped either is caught
    /// below. Held to the tolerance rather than to equality only because the
    /// products are formed in the other order.
    #[test]
    fn exchanging_the_query_and_key_norms_is_a_no_op() {
        for layer in Layer::all() {
            let mut weights = layer.weights.view(layer.hidden());
            std::mem::swap(&mut weights.q_norm, &mut weights.k_norm);

            let deviation = deviation(&layer.run(weights), &layer.prefill_out);
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// Their product is not interchangeable with one. A trained RMSNorm weight
    /// sits near 1, so a port that never loaded one is off by a few percent per
    /// channel and still generates.
    #[test]
    fn dropping_either_norm_changes_the_answer() {
        for layer in Layer::all() {
            let ones = vec![1.0; layer.config.head_dim];
            for (what, dropped) in [("q_norm", true), ("k_norm", false)] {
                let mut weights = layer.weights.view(layer.hidden());
                *(if dropped {
                    &mut weights.q_norm
                } else {
                    &mut weights.k_norm
                }) = &ones;

                let deviation = deviation(&layer.run(weights), &layer.prefill_out);
                assert!(
                    deviation > TOLERANCE,
                    "{}: {what} deviation {deviation:e}",
                    layer.name
                );
            }
        }
    }

    /// The key's convolution and the value's are separate kernels over separate
    /// cache slots. They have the same shape, so exchanging them runs.
    #[test]
    fn exchanging_the_key_and_value_convolutions_changes_the_answer() {
        for layer in Layer::all() {
            let mut weights = layer.weights.view(layer.hidden());
            std::mem::swap(&mut weights.k_sconv, &mut weights.v_sconv);

            let deviation = deviation(&layer.run(weights), &layer.prefill_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }
}
