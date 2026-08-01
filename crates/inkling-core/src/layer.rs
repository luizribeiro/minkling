//! A decoder layer, which is the whole of Inkling's language model repeated
//! forty-two times.
//!
//! Nothing here is a new op. The layer is the order its pieces run in and the
//! two residual adds around them:
//!
//! ```text
//! r   = self_attn(input_layernorm(x))
//! h   = x + attn_sconv(r)
//! r   = mlp(post_attention_layernorm(h))
//! out = h + mlp_sconv(r)
//! ```
//!
//! Two things about that shape decide the answer and neither is visible in a
//! single forward pass:
//!
//! - **Each residual is added to the value before its norm**, `x` and `h`, and
//!   not to what the norm produced. Adding the normalised value instead is the
//!   ordinary pre-norm/post-norm slip, and it leaves a layer that still runs.
//! - **The layer's two short convolutions have a cache slot each.** mlx-vlm
//!   gives every layer one `ArraysCache(4)` and threads that one object through
//!   all four of its convolutions, which pick a slot by `conv_idx`: 0 and 1 are
//!   the key's and the value's inside attention, 2 is `attn_sconv` and 3 is
//!   `mlp_sconv`. The first pair are `kv_heads * head_dim` wide and the second
//!   `hidden`, so only the second pair can be exchanged — and exchanging them
//!   is invisible until the second call, when each reads what the other left.
//!
//! Pinned to mlx-vlm by `reference/fixtures/layer.safetensors`, whose synthetic
//! dense and MoE layers are float32 throughout and are each driven twice against
//! one cache, and by the recorded residuals in
//! `reference/fixtures/layer_activations.safetensors`. The trained layers are
//! checkpoint-sized and are left to `tests/real_checkpoint.rs`.

use std::borrow::Cow;

use crate::attention::{Attention, AttentionCache, AttentionConfig, AttentionWeights, LayerStep};
use crate::moe::{BankRows, Gathered, Routed, Rows, SparseMoe};
use crate::ops::{DenseMlp, rms_norm};
use crate::profile::{self, Op};
use crate::sconv::{ConvState, ShortConv};

/// Where a whole decoder layer runs, when it is not here.
///
/// [`Projections::layer`](crate::attention::Projections::layer) is the same
/// bargain one step in, and its documentation is the argument for this one: a
/// backend holding the attention runs everything between the input layernorm
/// and `o_proj` because none of it is a value anybody else reads. What is
/// between `o_proj` and the MLP's own dispatches — the convolution on the
/// residual path, the add behind it, and the second norm — is three more of
/// those, and the only thing that kept them out was that whoever held the
/// attention did not also hold the MLP.
///
/// **What comes back is the first value that has to.** The routing's weights are
/// a softmax over eight numbers taken from logits a dispatch in this same
/// command buffer produced — see [`SparseMoe::weighted`] — so the rows both
/// banks answered have to cross back to be weighted, and `h` with them because
/// the layer's second residual is added to it here. Everything before that is
/// one command buffer.
pub trait DecoderDevice {
    /// The layer as far as it goes on the device, or `None` where this backend
    /// does not hold all of it.
    fn run(&self, cache: &mut DecoderCache, step: DecoderStep<'_>) -> Option<DecoderHalves>;
}

/// Everything one decoder layer runs before the first value that has to come
/// back.
///
/// [`LayerStep`] carries the attention layer's half, as it does for
/// [`Projections::layer`](crate::attention::Projections::layer); what this adds
/// is the residual path behind it and the MLP past it.
#[derive(Debug, Clone, Copy)]
pub struct DecoderStep<'a> {
    pub attention: LayerStep<'a>,
    /// The convolution on the residual path around attention, whose rows are
    /// added to the layer's input.
    pub attn_sconv: ShortConv<'a>,
    /// `[hidden]`, the weight of the norm the MLP consumes the output of.
    pub post_attention_layernorm: &'a [f32],
    /// The `rms_norm_eps` this layer's norms share, which is also
    /// [`LayerStep::eps`].
    pub eps: f32,
    pub mlp: LayerMlp<'a>,
}

/// What a decoder layer's device work left for this side: the value its second
/// residual is added to, and what its MLP made of the normed form of that value.
///
/// Named halves rather than a pair because they are two `[tokens, hidden]`
/// vectors of the same length, and exchanging them is a layer that still runs.
#[derive(Debug, Clone)]
pub struct DecoderHalves {
    /// `x + attn_sconv(attention(x))`, which the reference calls `h`.
    pub h: Vec<f32>,
    /// What the MLP returned over `post_attention_layernorm(h)`, weighted and
    /// scattered where the layer routes to experts.
    pub projected: Vec<f32>,
}

/// How a layer reaches the experts its router chose.
///
/// Asked for rather than held, for the reason [`SparseMoe::forward`] asks:
/// Inkling's routed bank is 25 GB per layer in float32, so an expert is decoded
/// when a token routes to it and dropped again. A dense layer never asks.
///
/// A whole bank's work at a time rather than an expert's, because what the two
/// backends want is not the same shape. The CPU wants an expert to decode and
/// [`Gathered::batches`] hands it one; a Metal dispatch wants every row it will
/// index and could not be handed that by a call per expert.
pub trait Experts {
    /// The routed bank, over every row that chose one of its experts.
    fn routed(&self, gathered: Gathered<'_>) -> Vec<f32>;

    /// The always-on shared bank, over every token, once per shared expert.
    fn shared(&self, gathered: Gathered<'_>) -> Vec<f32>;

    /// Both banks and, where this backend holds the router's gate, what its own
    /// router got to — the whole layer in one call. See [`Routed`].
    ///
    /// **One call because what is adjacent inside a layer is what a backend can
    /// merge, and only a backend handed the layer can see what is adjacent.**
    /// [`SparseMoe::forward`] states the ordering that decides it: the shared
    /// bank does not wait for the gate, so the gate's multiply rides in the
    /// command buffer that bank already opens. What a backend that also picks
    /// the experts adds is that nothing between the gate and the routed bank's
    /// last dispatch waits for this side at all. On this machine a marginal
    /// submission is about 156 microseconds around work that is already done.
    ///
    /// The default holds no gate and runs the two banks in the order the layer
    /// names them, which is what a backend that decodes an expert at a time
    /// wants and what leaves the layer's own weight to be multiplied where it
    /// always was.
    fn banks(
        &self,
        x: &[f32],
        shared: Gathered<'_>,
        route: &mut dyn FnMut(Option<Routed<'_>>) -> Option<Rows>,
    ) -> BankRows {
        let _ = x;
        let shared = self.shared(shared);
        let routed = route(None).expect("a layer that gathers nothing was not asked to");
        BankRows {
            routed: self.routed(routed.gathered()),
            shared,
        }
    }

    /// Whether this holds the layer's gate, which is what [`Experts::banks`]
    /// will answer with.
    ///
    /// Asked *before* the layer is stood up rather than inferred from that
    /// answer, because the weight has to be widened to be multiplied here and
    /// widening it is the cost being avoided — 4.2 MB a layer, 169 MB over the
    /// stack. A layer whose backend says yes never widens its gate at all.
    fn gates(&self) -> bool {
        false
    }
}

/// The [`Experts`] a dense layer needs, which is none.
///
/// Unreachable rather than empty: a dense layer that routed anything would be a
/// layer built with the wrong MLP.
#[derive(Debug, Clone, Copy)]
pub struct NoExperts;

impl Experts for NoExperts {
    fn routed(&self, gathered: Gathered<'_>) -> Vec<f32> {
        panic!("a dense layer routed to {:?}", gathered.experts())
    }

    fn shared(&self, gathered: Gathered<'_>) -> Vec<f32> {
        panic!("a dense layer routed to shared {:?}", gathered.experts())
    }
}

/// A layer's MLP slot, which is the only thing that varies between decoder
/// layers: `InklingDenseMLP` below `dense_mlp_idx` and `InklingSparseMoE` above
/// it. Everything around it is the same layer either way.
#[derive(Debug, Clone, Copy)]
pub enum LayerMlp<'a> {
    Dense(DenseMlp<'a>),
    Sparse(SparseMoe<'a>),
}

impl LayerMlp<'_> {
    /// The width this MLP maps between, which is the layer's hidden size.
    pub fn hidden(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.dim(),
            Self::Sparse(moe) => moe.hidden(),
        }
    }

    fn forward(&self, x: &[f32], experts: &(impl Experts + ?Sized)) -> Vec<f32> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe
                .forward(x, |x, shared, route| experts.banks(x, shared, route))
                .total(),
        }
    }
}

/// One decoder layer's tensors, as the checkpoint stores them: the attention
/// layer's own, a weight for each of the two RMSNorms, and a kernel for each of
/// the two short convolutions on the residual path.
#[derive(Debug, Clone, Copy)]
pub struct DecoderWeights<'a> {
    pub attention: AttentionWeights<'a>,
    pub input_layernorm: &'a [f32],
    pub post_attention_layernorm: &'a [f32],
    pub attn_sconv: &'a [f32],
    pub mlp_sconv: &'a [f32],
}

/// Everything one sequence carries between calls to one layer: the attention
/// layer's keys, values and two convolution windows, and the windows of the two
/// convolutions outside it.
///
/// The reference holds all four windows in one `ArraysCache(4)` and tells them
/// apart by index. They are separate fields here, which is the same thing said
/// so that a slot cannot be miscounted; what can still go wrong is which
/// convolution reads which, and that is what these fields' tests drive.
#[derive(Debug, Clone)]
pub struct DecoderCache {
    attention: AttentionCache,
    attn_sconv: ConvState,
    mlp_sconv: ConvState,
}

impl DecoderCache {
    /// The attention layer's own state, for a backend answering
    /// [`DecoderDevice::run`] — which runs the attention itself and so needs
    /// what [`Projections::layer`](crate::attention::Projections::layer) is
    /// handed directly.
    pub fn attention(&mut self) -> &mut AttentionCache {
        &mut self.attention
    }

    /// The state a sequence starts from: no keys, and four empty convolution
    /// windows.
    ///
    /// Built from the config and the two widths rather than from a
    /// [`DecoderLayer`], because the stack allocates all forty-two of these
    /// before it decodes any layer's weights.
    pub fn new(config: AttentionConfig, hidden: usize, kernel_size: usize) -> Self {
        Self {
            attention: AttentionCache::new(config, kernel_size),
            attn_sconv: ConvState::new(hidden, kernel_size),
            mlp_sconv: ConvState::new(hidden, kernel_size),
        }
    }
}

/// One decoder layer, from the hidden state it is handed to the one it passes
/// on.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayer<'a> {
    attention: Attention<'a>,
    mlp: LayerMlp<'a>,
    attn_sconv: ShortConv<'a>,
    mlp_sconv: ShortConv<'a>,
    input_layernorm: &'a [f32],
    post_attention_layernorm: &'a [f32],
    rms_norm_eps: f32,
    hidden: usize,
}

impl<'a> DecoderLayer<'a> {
    /// `config` is the attention layer's, and carries the `rms_norm_eps` both of
    /// this layer's norms share; `mlp` is whichever of the two MLPs the layer
    /// index called for.
    pub fn new(config: AttentionConfig, weights: DecoderWeights<'a>, mlp: LayerMlp<'a>) -> Self {
        let hidden = weights.input_layernorm.len();
        assert_eq!(
            weights.post_attention_layernorm.len(),
            hidden,
            "the two layernorms are over the same width"
        );
        assert_eq!(mlp.hidden(), hidden, "the MLP against a hidden {hidden}");

        Self {
            attention: Attention::new(config, weights.attention),
            mlp,
            attn_sconv: ShortConv::new(hidden, weights.attn_sconv),
            mlp_sconv: ShortConv::new(hidden, weights.mlp_sconv),
            input_layernorm: weights.input_layernorm,
            post_attention_layernorm: weights.post_attention_layernorm,
            rms_norm_eps: config.rms_norm_eps,
            hidden,
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The state a sequence starts from, for this layer's own shape.
    pub fn cache(&self) -> DecoderCache {
        DecoderCache::new(
            self.attention.config(),
            self.hidden,
            self.attn_sconv.kernel_size(),
        )
    }

    /// `[tokens, hidden]` in and out, continuing from `cache` and leaving this
    /// call's keys, values and convolution windows behind in it.
    ///
    /// `device` runs the layer as far as it goes without a value crossing back —
    /// see [`DecoderDevice`] — and `None` leaves every operation here.
    pub fn forward(
        &self,
        cache: &mut DecoderCache,
        x: &[f32],
        experts: &(impl Experts + ?Sized),
        device: Option<&dyn DecoderDevice>,
    ) -> Vec<f32> {
        self.run(cache, x, experts, device, Residual::PreNorm)
    }

    /// [`DecoderLayer::forward`], with the two ways of wiring a residual named
    /// apart.
    ///
    /// Both leave a layer that runs, normalises and generates, so the tests
    /// drive them from here to show that only one reproduces mlx-vlm. Only the
    /// reference's wiring reaches `device`: the other exists to be measured
    /// against this side's arithmetic, and a backend that spelled it too would
    /// be a second spelling of a mistake.
    fn run(
        &self,
        cache: &mut DecoderCache,
        x: &[f32],
        experts: &(impl Experts + ?Sized),
        device: Option<&dyn DecoderDevice>,
        residual: Residual,
    ) -> Vec<f32> {
        assert_eq!(
            x.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.hidden
        );

        // Guarded before the call and not after it: a device that ran the layer
        // has advanced the cache, so asking one and then discarding the answer
        // would leave the arithmetic below reading a sequence that had already
        // moved on.
        let device = match residual {
            Residual::PreNorm => device,
            Residual::Normed => None,
        };
        if let Some(halves) = self.on_device(cache, x, device) {
            return self.residual_path(cache, &halves.h, &halves.projected);
        }

        let attended = self
            .attention
            .forward(&mut cache.attention, x, Some(self.input_layernorm));
        let h = add(
            &residual.around_attention(x, self.input_layernorm, self.rms_norm_eps),
            &self
                .attn_sconv
                .forward(&mut cache.attn_sconv, &attended, None),
        );

        let normed = rms_norm(&h, self.post_attention_layernorm, self.rms_norm_eps);
        let projected = self.mlp.forward(&normed, experts);
        self.residual_path(cache, residual.of(&h, &normed), &projected)
    }

    /// The layer's second residual path, which both routes take: what the MLP
    /// produced, convolved, added to the value before the norm that fed it.
    ///
    /// **This is where a layer's device work stops.** The convolution's window
    /// is the one piece of a layer's state that a backend running everything
    /// else still does not hold, because the rows it reads are the routing's
    /// weights applied to what the banks answered and those weights are
    /// this side's — see [`SparseMoe::weighted`]. So the seam that reached
    /// `o_proj` and then the second norm stops here, between one layer and the
    /// next rather than inside either.
    fn residual_path(
        &self,
        cache: &mut DecoderCache,
        residual: &[f32],
        projected: &[f32],
    ) -> Vec<f32> {
        add(
            residual,
            &self
                .mlp_sconv
                .forward(&mut cache.mlp_sconv, projected, None),
        )
    }

    /// The layer as far as `device` takes it, with the step it needs built here
    /// — because [`Attention::forward`] would run the attention rather than
    /// describe it, and what it would hand back is the one value the residual
    /// add wants.
    fn on_device(
        &self,
        cache: &mut DecoderCache,
        x: &[f32],
        device: Option<&dyn DecoderDevice>,
    ) -> Option<DecoderHalves> {
        let device = device?;
        let queries = x.len() / self.hidden;
        let offset = cache.attention.seen();
        let taus = self.attention.taus(offset, queries);
        let step = DecoderStep {
            attention: self
                .attention
                .step(x, Some(self.input_layernorm), &taus, offset),
            attn_sconv: self.attn_sconv,
            post_attention_layernorm: self.post_attention_layernorm,
            eps: self.rms_norm_eps,
            mlp: self.mlp,
        };
        device.run(cache, step)
    }
}

/// What a residual is added to.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum Residual {
    /// The value before its norm, which is what `InklingDecoderLayer` adds.
    PreNorm,
    /// What the norm produced, which is the classic pre-norm/post-norm slip.
    Normed,
}

impl Residual {
    fn of<'a>(&self, pre_norm: &'a [f32], normed: &'a [f32]) -> &'a [f32] {
        match self {
            Self::PreNorm => pre_norm,
            Self::Normed => normed,
        }
    }

    /// What the *first* residual is added to, which cannot be [`Residual::of`]:
    /// the value the attention layer normalised is no longer formed here — see
    /// [`Attention::forward`] — so the slip this names has to normalise the
    /// input again to be able to take it.
    ///
    /// Which is the shape of the finding rather than a workaround. A value the
    /// engine does not compute is one a mistake has to go out of its way to
    /// reach, and the norm below runs on no path but this one.
    fn around_attention<'a>(&self, x: &'a [f32], weight: &[f32], eps: f32) -> Cow<'a, [f32]> {
        match self {
            Self::PreNorm => Cow::Borrowed(x),
            Self::Normed => Cow::Owned(rms_norm(x, weight, eps)),
        }
    }
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let _timed = profile::scope(Op::Residual);
    assert_eq!(a.len(), b.len(), "a residual against what it is added to");
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::LogScaling;
    use crate::fixture::{self, ACTIVATIONS, CAPTURED_LAYERS, LayerTensors, deviation};

    /// Synthetic dense and MoE layers, and the two calls mlx-vlm drove each of
    /// them with, from `just dump-layer-fixture`.
    const FIXTURE: &str = "layer.safetensors";

    const CASES: [&str; 2] = ["dense", "moe"];

    /// The four cache slots `InklingDecoderLayer` threads one `ArraysCache(4)`
    /// through, in `conv_idx` order.
    const CONV_SLOTS: [&str; 4] = ["k_sconv", "v_sconv", "attn_sconv", "mlp_sconv"];

    /// The synthetic cases are float32 end to end, so only summation order
    /// separates this from MLX — the same bound, for the same reason, as the
    /// RMSNorm, MLP, sconv, mask, attention and MoE cases. This runs the longest
    /// chain of any of them, a whole layer rather than a piece of one, so it
    /// holds the least of the bound in reserve. Worst observed when this landed:
    /// 5.3e-7, on the window the dense case's `mlp_sconv` left behind, a factor
    /// of about two in hand. The weakest mutation these tests rely on catching —
    /// the dense case's two layernorms exchanged — moves the answer by 5.5e-1,
    /// five decades above.
    const TOLERANCE: f32 = 1e-6;

    /// The recorded residuals were formed and stored in bfloat16, so `h` is
    /// `input + attn_sconv_out` rounded once: half a quantum of `h`'s own
    /// magnitude, which is at worst 2^-9 = 2.0e-3 measured against the tensor's
    /// largest value. Worst observed: 2.3e-3 on layer 2's `h`, a little over
    /// that, which is what a peak sitting just above a power of two costs. The
    /// weakest mutation this bound has to catch — layer 0's `h` taken from the
    /// normalised value — moves the answer by 1.2e-1, a factor of forty above.
    const RECORDED_TOLERANCE: f32 = 3e-3;

    /// One synthetic case: the layer `InklingDecoderLayer` was built as, the two
    /// sequences it was driven with, and what it produced and cached.
    struct Layer {
        name: String,
        config: AttentionConfig,
        weights: LayerTensors,
        x: Vec<f32>,
        continue_x: Vec<f32>,
        prefill_out: Vec<f32>,
        continue_out: Vec<f32>,
        conv_state: [Vec<f32>; 4],
    }

    impl Layer {
        /// `config` is the `[heads, kv_heads, head_dim, d_rel, sliding,
        /// rel_extent, rms_norm_eps, log_floor, log_alpha]` the dump script
        /// recorded, in the layout the attention fixture uses. A `log_floor` of
        /// zero is the `None` a sliding layer gets.
        fn load(case: &str) -> Self {
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

            let weights = LayerTensors::load(&ckpt, case);
            assert_eq!(
                weights.view().attention.rel_proj.len(),
                (d_rel * rel_extent) as usize,
                "{case}: rel_proj against its recorded extent"
            );

            Self {
                name: case.to_string(),
                config: AttentionConfig {
                    hidden: weights.hidden(),
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
                conv_state: CONV_SLOTS.map(|slot| of(&format!("{case}.conv_state.{slot}"))),
            }
        }

        fn all() -> Vec<Self> {
            CASES.iter().map(|case| Self::load(case)).collect()
        }

        fn hidden(&self) -> usize {
            self.weights.hidden()
        }

        fn layer(&self) -> DecoderLayer<'_> {
            self.with(self.weights.view())
        }

        /// This layer under weights its own may have been mutated into.
        fn with<'w>(&'w self, weights: DecoderWeights<'w>) -> DecoderLayer<'w> {
            DecoderLayer::new(self.config, weights, self.weights.mlp())
        }

        /// The prefill alone, from a fresh cache.
        fn prefill(&self, layer: &DecoderLayer<'_>) -> Vec<f32> {
            layer.forward(&mut layer.cache(), &self.x, &self.weights, None)
        }

        /// The prefill and the continuation, against one cache, as the dump
        /// script drove the reference.
        fn forward(&self) -> (Vec<f32>, Vec<f32>) {
            self.wired(Residual::PreNorm)
        }

        fn wired(&self, residual: Residual) -> (Vec<f32>, Vec<f32>) {
            let layer = self.layer();
            let cache = &mut layer.cache();
            (
                layer.run(cache, &self.x, &self.weights, None, residual),
                layer.run(cache, &self.continue_x, &self.weights, None, residual),
            )
        }

        /// The worst of the two calls' deviations. The continuation is where the
        /// cache enters, so a prefill that matched alone would say nothing about
        /// decoding.
        fn deviation(&self, (prefill, rest): &(Vec<f32>, Vec<f32>)) -> f32 {
            deviation(prefill, &self.prefill_out).max(deviation(rest, &self.continue_out))
        }
    }

    #[test]
    fn the_synthetic_layers_reproduce_mlx() {
        let mut worst = 0.0f32;
        for layer in Layer::all() {
            let deviation = layer.deviation(&layer.forward());
            assert!(
                deviation <= TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
            worst = worst.max(deviation);
        }
        assert!(worst > 0.0, "float32 summation order cannot agree exactly");
    }

    /// The one thing the two cases do not share. A dense layer has an
    /// `InklingDenseMLP` and no router at all; every layer past `dense_mlp_idx`
    /// has a router and no dense MLP.
    #[test]
    fn the_two_cases_cover_both_mlps() {
        let dense: Vec<bool> = Layer::all().iter().map(|l| l.weights.is_dense()).collect();
        assert_eq!(dense, [true, false], "the fixture covers one of each");
    }

    /// The residual is added to `x` and to `h`, the values before their norms.
    /// Adding what the norm produced instead leaves a layer that still
    /// normalises, still attends and still generates.
    #[test]
    fn the_residual_is_taken_from_before_the_norm() {
        for layer in Layer::all() {
            let deviation = layer.deviation(&layer.wired(Residual::Normed));
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// A sequence split anywhere and carried across the split by the cache is
    /// the same sequence, bit for bit.
    ///
    /// Exact equality rather than a tolerance, for the reason the short
    /// convolution's own split test demands it: both paths multiply the same
    /// numbers in the same order, and the only thing a split changes is where
    /// they come from. It is also the test a swapped pair of convolution slots
    /// fails, because within one call a slot is written and never read.
    #[test]
    fn splitting_a_call_in_two_matches_feeding_it_whole() {
        for layer in Layer::all() {
            let decoder = layer.layer();

            let mut whole = layer.x.clone();
            whole.extend_from_slice(&layer.continue_x);
            let at_once = decoder.forward(&mut decoder.cache(), &whole, &layer.weights, None);

            let (prefill, rest) = layer.forward();
            let mut split = prefill;
            split.extend(rest);
            assert_eq!(split, at_once, "{}", layer.name);
        }
    }

    /// A stack allocates a cache for every layer before it decodes any layer's
    /// weights, so a cache has to come from the shapes alone. It has to be the
    /// same cache: a window of the wrong width or keys of the wrong stride
    /// would show only on the second call, which is why both are driven.
    #[test]
    fn a_cache_built_from_the_shapes_alone_drives_the_same_two_calls() {
        for layer in Layer::all() {
            let decoder = layer.layer();
            let kernel_size = layer.weights.kernel_size();
            let mut cache = DecoderCache::new(layer.config, layer.hidden(), kernel_size);
            let prefill = decoder.forward(&mut cache, &layer.x, &layer.weights, None);
            let rest = decoder.forward(&mut cache, &layer.continue_x, &layer.weights, None);
            assert_eq!((prefill, rest), layer.forward(), "{}", layer.name);
        }
    }

    /// `attn_sconv` is `conv_idx` 2 and `mlp_sconv` is 3. They are the same
    /// width, so a port that exchanged their slots runs — and agrees for one
    /// call, because a slot is written at the end of a call and read at the
    /// start of the next.
    #[test]
    fn exchanging_the_two_convolution_slots_changes_the_next_call() {
        for layer in Layer::all() {
            let decoder = layer.layer();
            let mut cache = decoder.cache();

            let prefill = decoder.forward(&mut cache, &layer.x, &layer.weights, None);
            let agreed = deviation(&prefill, &layer.prefill_out);
            assert!(
                agreed <= TOLERANCE,
                "{}: the prefill itself deviates by {agreed:e}",
                layer.name
            );

            std::mem::swap(&mut cache.attn_sconv, &mut cache.mlp_sconv);
            let rest = decoder.forward(&mut cache, &layer.continue_x, &layer.weights, None);
            let deviation = deviation(&rest, &layer.continue_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// What the prefill left in the layer's two slots, against what mlx-vlm's
    /// own `ArraysCache(4)` held after the same call.
    ///
    /// The four slots are the two the attention layer owns and the two this one
    /// does, and the recorded widths say which pair is which: attention's are
    /// `kv_heads * head_dim` and the layer's are `hidden`. The pair that could
    /// be confused for each other is therefore the layer's own, which is the
    /// pair the test above drives.
    #[test]
    fn the_prefill_leaves_the_windows_the_reference_kept() {
        for layer in Layer::all() {
            let decoder = layer.layer();
            let mut cache = decoder.cache();
            decoder.forward(&mut cache, &layer.x, &layer.weights, None);

            let attention = layer.config.kv_heads * layer.config.head_dim;
            let hidden = layer.hidden();
            assert_ne!(attention, hidden, "the two pairs have to differ in width");

            let kept = decoder.attn_sconv.kernel_size() - 1;
            for ((slot, want), width) in CONV_SLOTS
                .iter()
                .zip(&layer.conv_state)
                .zip([attention, attention, hidden, hidden])
            {
                assert_eq!(
                    want.len(),
                    kept * width,
                    "{}: {slot} is {width} wide",
                    layer.name
                );
            }

            for (name, got, want) in [
                ("attn_sconv", &cache.attn_sconv, &layer.conv_state[2]),
                ("mlp_sconv", &cache.mlp_sconv, &layer.conv_state[3]),
            ] {
                let deviation = deviation(got.history(), want);
                assert!(
                    deviation <= TOLERANCE,
                    "{}: {name} deviation {deviation:e}",
                    layer.name
                );
            }
        }
    }

    /// The continuation reads the cache, so a layer that dropped it would still
    /// answer — over its own three tokens, from four zeroed convolution windows
    /// and no keys.
    #[test]
    fn the_continuation_reads_what_the_prefill_cached() {
        for layer in Layer::all() {
            let decoder = layer.layer();
            let fresh = decoder.forward(
                &mut decoder.cache(),
                &layer.continue_x,
                &layer.weights,
                None,
            );
            let deviation = deviation(&fresh, &layer.continue_out);
            assert!(
                deviation > TOLERANCE,
                "{}: deviation {deviation:e}",
                layer.name
            );
        }
    }

    /// The two convolutions on the residual path are separate kernels over the
    /// same width, and so are the two layernorms. Exchanging either pair runs.
    #[test]
    fn exchanging_either_pair_of_same_width_weights_changes_the_answer() {
        for layer in Layer::all() {
            for (what, weights) in [
                ("the convolutions", {
                    let mut weights = layer.weights.view();
                    std::mem::swap(&mut weights.attn_sconv, &mut weights.mlp_sconv);
                    weights
                }),
                ("the layernorms", {
                    let mut weights = layer.weights.view();
                    std::mem::swap(
                        &mut weights.input_layernorm,
                        &mut weights.post_attention_layernorm,
                    );
                    weights
                }),
            ] {
                let decoder = layer.with(weights);
                let deviation = deviation(&layer.prefill(&decoder), &layer.prefill_out);
                assert!(
                    deviation > TOLERANCE,
                    "{}: {what} deviate by {deviation:e}",
                    layer.name
                );
            }
        }
    }

    /// One captured layer's residual adds, as the recorded tensors show them.
    struct Recorded {
        name: String,
        input: Vec<f32>,
        input_layernorm_out: Vec<f32>,
        attn_sconv_out: Vec<f32>,
        h: Vec<f32>,
        post_attention_ln_out: Vec<f32>,
        mlp_sconv_out: Vec<f32>,
        out: Vec<f32>,
    }

    impl Recorded {
        fn load(layer: usize) -> Self {
            let activations = fixture::open(ACTIVATIONS);
            let of = |name: &str| fixture::f32s(&fixture::layer_tensor(&activations, layer, name));
            Self {
                name: format!("layer{layer}"),
                input: of("input"),
                input_layernorm_out: of("input_layernorm_out"),
                attn_sconv_out: of("attn_sconv_out"),
                h: of("h"),
                post_attention_ln_out: of("post_attention_ln_out"),
                mlp_sconv_out: of("mlp_sconv_out"),
                out: of("out"),
            }
        }

        fn all() -> Vec<Self> {
            CAPTURED_LAYERS.iter().copied().map(Self::load).collect()
        }

        fn residuals(&self) -> [ResidualAdd<'_>; 2] {
            [
                ResidualAdd {
                    what: "h",
                    sum: &self.h,
                    pre_norm: &self.input,
                    normed: &self.input_layernorm_out,
                    added: &self.attn_sconv_out,
                },
                ResidualAdd {
                    what: "out",
                    sum: &self.out,
                    pre_norm: &self.h,
                    normed: &self.post_attention_ln_out,
                    added: &self.mlp_sconv_out,
                },
            ]
        }
    }

    /// One of a captured layer's two residual adds: the sum the reference
    /// recorded, the pre-norm value it was taken from, the normalised value a
    /// slip would take instead, and what was added to either.
    struct ResidualAdd<'a> {
        what: &'a str,
        sum: &'a [f32],
        pre_norm: &'a [f32],
        normed: &'a [f32],
        added: &'a [f32],
    }

    /// The wiring, stated against the reference's own intermediates and no
    /// weights at all: `h` is the layer's input plus what `attn_sconv` returned,
    /// and the output is `h` plus what `mlp_sconv` returned.
    ///
    /// Not exact, because the reference formed and stored both sums in bfloat16
    /// — see [`RECORDED_TOLERANCE`].
    #[test]
    fn the_recorded_layers_add_each_residual_to_the_value_before_its_norm() {
        let mut worst = 0.0f32;
        for recorded in Recorded::all() {
            for residual in recorded.residuals() {
                let deviation = deviation(&add(residual.pre_norm, residual.added), residual.sum);
                assert!(
                    deviation <= RECORDED_TOLERANCE,
                    "{}: {} deviation {deviation:e}",
                    recorded.name,
                    residual.what
                );
                worst = worst.max(deviation);
            }
        }
        assert!(
            worst > 0.0,
            "a run that matched exactly would mean the reference's bfloat16 vanished"
        );
    }

    /// The same two adds taken from the normalised value instead. Both norms
    /// divide their row by its own RMS, so the sum keeps its shape and loses its
    /// magnitude — a layer that trains, runs and generates.
    #[test]
    fn adding_the_normalised_value_instead_changes_the_recorded_answer() {
        for recorded in Recorded::all() {
            for residual in recorded.residuals() {
                let deviation = deviation(&add(residual.normed, residual.added), residual.sum);
                assert!(
                    deviation > RECORDED_TOLERANCE,
                    "{}: {} deviation {deviation:e}",
                    recorded.name,
                    residual.what
                );
            }
        }
    }
}
