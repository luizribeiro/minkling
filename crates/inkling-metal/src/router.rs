//! Which six of 256 experts a token reads, and with what weight — chosen and
//! weighted where its gate logits were computed.
//!
//! **The router is two halves and they are two kernels.** What
//! `InklingSparseMoE` does with its gate is rank the experts under
//! `sigmoid(logit) + e_score_correction_bias` and take the best `top_k`, and
//! then weight what it took. The selection is what the routed bank's rows are
//! named by, so it is what a bank cannot be dispatched without; the weighting is
//! what the rows both banks answered are scaled by, so it is what the layer's
//! output cannot be formed without. They are separate here because they are
//! separate mistakes: [`router_top_k`](BODY) reads the *biased* scores and
//! [`router_weights`](WEIGHTS) reads the *raw* logits, and a kernel that did
//! both at once would be one place for two readings of one row to be confused.
//!
//! **Three of the four ways of misreading this gate are in the weighting**, and
//! moving it does not move where they are settled. [`inkling_core::moe`] is
//! still the authority — the weights come from the raw logits of whatever was
//! picked, one softmax spans the picked routed experts and both shared ones, and
//! `route_scale` and the learned `global_scale` both multiply what comes out —
//! and `SparseMoe::weigh` is still what every fixture holds to mlx-vlm. What
//! this kernel is measured against is that function, over the shape and the
//! trained scales the checkpoint carries.
//!
//! **What that buys is the CPU leaving a MoE layer entirely.** The gate's
//! multiply, the top-k over what it produced, both banks and the activation
//! inside each, the softmax over eight numbers and the weighted sum of both
//! banks' rows are one command buffer, because every value between them is a
//! buffer the next dispatch reads and no dispatch waits for an answer from this
//! side. What crosses back at the end is `[tokens, hidden]` and nothing else.
//!
//! **The set is the reference's and the order is not.** `mx.argpartition`
//! promises only that the k-th element lands where a sort would put it, and
//! MLX's own CPU and Metal streams return the k before it in different orders
//! for one input. What is reproducible is the *set*, ties going to the lower
//! index — see [`SparseMoe::route`](inkling_core::SparseMoe::route) — so that is
//! what this kernel reproduces and what its cases assert. Nothing downstream
//! reads the order: every slot of a token scatters into the same row.

use std::cell::RefCell;

use inkling_core::moe::MoeConfig;
use inkling_core::profile::{self, Op};

use crate::buffer::Buffer;
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "router_top_k";
const WEIGHTS_ENTRY: &str = "router_weights";

/// Threads one threadgroup of a dispatch holds, which is a threadgroup to a
/// token and a thread to each of its 256 experts.
///
/// **A decode step ranks one row, so how wide that row is ranked is the whole
/// of what this kernel costs.** A thread to a token gives the GPU one thread for
/// the entire dispatch, walking 256 experts six times over with nothing to hide
/// a memory latency behind; measured that way it was 380 microseconds of device
/// time a layer, which is twice the round trip it was put here to save. A
/// threadgroup to a token reads the row coalesced and reduces in a tree.
const THREADS_PER_GROUP: usize = 256;

/// Entries the kernel's per-simdgroup arrays hold, which has to be a constant
/// where the number of simdgroups is not: 1024 threads is the widest threadgroup
/// any Apple GPU allows and 32 the narrowest simdgroup any reports, so 32
/// partials is the most a threadgroup can produce.
const MOST_SIMDGROUPS: usize = 32;

/// Slots the kernel's threadgroup array of picked experts holds, which bounds
/// the `top_k` a router here can have. Inkling reads six of 256; a model that
/// read more than this would need the array widened rather than a fallback,
/// which is why it is refused where a router is stood up.
const MOST_SLOTS: usize = 64;

/// Always-on experts the weighting kernel's row holds beside the picked ones,
/// which bounds the `n_shared` a router here can have for the same reason
/// [`MOST_SLOTS`] bounds its `top_k`. Inkling has two.
const MOST_SHARED: usize = 8;

/// Threads one threadgroup of the weighting holds.
///
/// A thread to a token, where the selection gives a threadgroup to one: this
/// reads `top_k + n_shared` of a row rather than all `n_routed` of it, so what
/// a threadgroup would have to reduce over is eight numbers.
const THREADS_PER_WEIGHING: usize = 64;

/// The compiled kernel, which every router on a device shares.
///
/// Per source string rather than per layer, like [`crate::PackedMatmul`] and
/// [`crate::RmsNorm`]: the source names no shape, so one of these serves all
/// forty routers in the model.
#[derive(Debug)]
pub struct Router {
    kernel: Kernel,
}

impl Router {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`Router::new`] out of a source string of the caller's own, which is how
    /// a test puts a deliberately wrong kernel through the same plumbing as the
    /// right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }
}

/// The compiled weighting, which every router on a device shares.
///
/// Beside [`Router`] rather than inside it, because they are two entry points:
/// a library holds one function each here — see
/// [`Device::compile`](crate::Device::compile) — and a test that puts a
/// deliberately wrong selection through the plumbing has no business
/// recompiling the weighting to do it.
#[derive(Debug)]
pub struct RouterWeights {
    kernel: Kernel,
}

impl RouterWeights {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &weights_source())
    }

    /// [`RouterWeights::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, WEIGHTS_ENTRY)?,
        })
    }
}

/// The `[tokens, top_k]` weights a selection carries and the `[tokens,
/// n_shared]` the always-on experts do, as the dispatch that computed them left
/// them.
///
/// Named halves rather than a pair for the reason
/// [`BankRows`](inkling_core::moe::BankRows) is: they are two buffers of the
/// same element and different lengths, and exchanging them is a layer that
/// still runs.
#[derive(Debug)]
pub struct RoutingWeights {
    pub routed: Buffer<f32>,
    pub shared: Buffer<f32>,
}

/// One layer's router: the shape it chooses under, the correction bias it ranks
/// with, and the two scales the weights it hands out carry.
///
/// The bias is `[n_routed]` and is copied rather than wrapped, for the reason a
/// norm weight is: it is bfloat16 in the checkpoint and the kernel wants
/// float32, so there are no packed bytes here to hand over in place. 1 KB a
/// layer, copied once.
#[derive(Debug)]
pub struct LayerRouter<'a> {
    device: &'a Device,
    router: &'a Router,
    weights: &'a RouterWeights,
    config: MoeConfig,
    /// The layer's learned output scale, which multiplies every weight this
    /// hands out alongside [`MoeConfig::route_scale`]. Layer 2's is 0.00704, so
    /// a router that applied `route_scale` alone would weight every expert 142
    /// times too heavily — see [`LayerRouter::scale`].
    global_scale: f32,
    /// Held behind a cell for the reason [`crate::PackedBank`]'s resident
    /// tensors are: binding a buffer to a dispatch borrows it exclusively, and
    /// the bias belongs to the layer rather than to a call.
    bias: RefCell<Buffer<f32>>,
}

impl<'a> LayerRouter<'a> {
    /// A router over `config`, ranking with `correction_bias` and weighting
    /// with `global_scale` beside the config's own `route_scale`.
    pub fn new(
        device: &'a Device,
        router: &'a Router,
        weights: &'a RouterWeights,
        config: MoeConfig,
        correction_bias: &[f32],
        global_scale: f32,
    ) -> Result<Self, MetalError> {
        assert_eq!(
            correction_bias.len(),
            config.n_routed,
            "one correction bias per routed expert"
        );
        assert!(
            (1..=config.n_routed).contains(&config.top_k),
            "{} experts per token do not come out of {}",
            config.top_k,
            config.n_routed
        );
        assert!(
            config.top_k <= MOST_SLOTS,
            "{} experts per token are more than the {MOST_SLOTS} a threadgroup holds",
            config.top_k
        );
        assert!(
            (1..=MOST_SHARED).contains(&config.n_shared),
            "{} shared experts are not between one and the {MOST_SHARED} a row holds",
            config.n_shared
        );
        Ok(Self {
            bias: RefCell::new(device.buffer(correction_bias)?),
            device,
            router,
            weights,
            config,
            global_scale,
        })
    }

    /// What every weight this hands out is multiplied by: `route_scale` from
    /// the config and the learned `global_scale` from the checkpoint.
    ///
    /// **Both, and the second is the one worth naming.** The eight weights of a
    /// token sum to this product — 8 times 0.00704, or 0.0563, on the layer the
    /// activation capture covers — so a router carrying `route_scale` alone
    /// runs every expert 142 times too hot, which is the largest single error
    /// any of the four ways of misreading this gate can produce.
    pub fn scale(&self) -> f32 {
        self.config.route_scale * self.global_scale
    }

    /// The width of a row of gate logits, which is every expert the layer has.
    pub fn width(&self) -> usize {
        self.config.n_routed + self.config.n_shared
    }

    pub fn config(&self) -> MoeConfig {
        self.config
    }

    /// How many rows the routed bank runs for `tokens` tokens, which is the one
    /// thing about the selection that is known before it is made.
    pub fn assignments(&self, tokens: usize) -> usize {
        tokens * self.config.top_k
    }

    /// `[tokens, n_routed + n_shared]` logits in, `[tokens, top_k]` experts out,
    /// encoded into `batch` over a buffer a dispatch already left there.
    ///
    /// The output is a buffer and not values, and that is the whole point: it is
    /// what the routed bank's two dispatches index their expert by, so the
    /// selection never crosses back to be handed over again.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        logits: &mut Buffer<f32>,
    ) -> Result<Buffer<u32>, MetalError> {
        let _timed = profile::scope(Op::Encode);
        let width = self.width();
        assert_eq!(
            logits.len() % width,
            0,
            "{} logits are not whole rows of {width} experts",
            logits.len()
        );
        let tokens = logits.len() / width;

        let fields = [
            extent(tokens, "the tokens of a call"),
            extent(self.config.n_routed, "the routed experts of a layer"),
            extent(width, "the width of a row of logits"),
            extent(self.config.top_k, "the experts a token reads"),
        ];
        let mut shape = self.device.inline(&fields)?;
        let mut bias = self.bias.borrow_mut();
        let mut chosen = self.device.zeroed::<u32>(self.assignments(tokens))?;

        batch.add(
            &self.router.kernel,
            &[shape.arg(), logits.arg(), bias.arg(), chosen.arg()],
            Grid::new(tokens * THREADS_PER_GROUP, THREADS_PER_GROUP),
        )?;
        Ok(chosen)
    }

    /// The weights a selection carries, encoded into `batch` over two buffers
    /// dispatches already left there: `[tokens, width]` logits and the
    /// `[tokens, top_k]` experts the top-k took out of them.
    ///
    /// **The weights come from the raw logits and the selection is read for its
    /// indices alone**, which is the first of the three ways of misreading this
    /// gate that live here: `e_score_correction_bias` ranked the experts in the
    /// dispatch before this one and takes no further part. It is not bound to
    /// this dispatch at all, which is that stated in the arguments.
    ///
    /// The other two are in the kernel — one softmax over the picked routed
    /// experts and both shared ones together, and both scales on what comes out
    /// — and [`inkling_core::moe`] is the authority on all three.
    pub fn encode_weights(
        &self,
        batch: &mut Batch<'_>,
        logits: &mut Buffer<f32>,
        picked: &mut Buffer<u32>,
    ) -> Result<RoutingWeights, MetalError> {
        let _timed = profile::scope(Op::Encode);
        let MoeConfig {
            n_routed,
            n_shared,
            top_k,
            ..
        } = self.config;
        let width = self.width();
        assert_eq!(
            logits.len() % width,
            0,
            "{} logits are not whole rows of {width} experts",
            logits.len()
        );
        let tokens = logits.len() / width;
        assert_eq!(
            picked.len(),
            self.assignments(tokens),
            "{} selected experts are not {top_k} for each of {tokens} tokens",
            picked.len()
        );

        let fields = [
            extent(tokens, "the tokens of a call"),
            extent(n_routed, "the routed experts of a layer"),
            extent(width, "the width of a row of logits"),
            extent(top_k, "the experts a token reads"),
            extent(n_shared, "the experts every token reads"),
        ];
        let mut shape = self.device.inline(&fields)?;
        let scaled_by = [self.scale()];
        let mut scaling = self.device.inline(&scaled_by)?;
        let mut weights = RoutingWeights {
            routed: self.device.zeroed::<f32>(picked.len())?,
            shared: self.device.zeroed::<f32>(tokens * n_shared)?,
        };

        batch.add(
            &self.weights.kernel,
            &[
                shape.arg(),
                scaling.arg(),
                logits.arg(),
                picked.arg(),
                weights.routed.arg(),
                weights.shared.arg(),
            ],
            Grid::new(tokens, THREADS_PER_WEIGHING),
        )?;
        Ok(weights)
    }

    /// The same selection submitted on its own, for a caller with nothing to
    /// batch it against — which is the cases here and nothing in the engine.
    pub fn select(&self, logits: &[f32]) -> Result<Vec<u32>, MetalError> {
        let mut input = self.device.buffer(logits)?;
        let mut batch = self.device.batch()?;
        let chosen = self.encode(&mut batch, &mut input)?;
        batch.wait()?;
        Ok(chosen.to_vec())
    }

    /// The same weighting submitted on its own, over a selection this side
    /// holds — which is the cases here and nothing in the engine.
    pub fn weigh(&self, logits: &[f32], picked: &[u32]) -> Result<RoutingWeights, MetalError> {
        let mut logits = self.device.buffer(logits)?;
        let mut picked = self.device.buffer(picked)?;
        let mut batch = self.device.batch()?;
        let weights = self.encode_weights(&mut batch, &mut logits, &mut picked)?;
        batch.wait()?;
        Ok(weights)
    }
}

/// The kernel, with the two bounds its threadgroup arrays are sized by written
/// into its prelude rather than spelled twice.
pub(crate) fn source() -> String {
    format!(
        "constant uint MOST_SIMDGROUPS = {MOST_SIMDGROUPS};\n\
         constant uint MOST_SLOTS = {MOST_SLOTS};\n{BODY}"
    )
}

/// The weighting, with the bound its per-thread row is sized by written into
/// its prelude for the same reason.
pub(crate) fn weights_source() -> String {
    format!(
        "constant uint MOST_WEIGHTS = {};\n{WEIGHTS}",
        MOST_SLOTS + MOST_SHARED
    )
}

/// Everything of the kernel that those two bounds do not decide.
///
/// `sigmoid` is written as `1 / (1 + exp(-x))`, which is
/// [`inkling_core::moe`]'s own form. The two `exp`s agree to a few ulps rather
/// than exactly, so a selection can only part company with the CPU's where two
/// scores straddling the last slot are that close — and
/// `the_trained_selection_clears_the_gates_float32_drift` measures the trained
/// margin at four times the drift a float32 gate already introduces, which is
/// decades above this.
pub(crate) const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint tokens;
    uint n_routed;
    uint width;
    uint top_k;
};

/// The `top_k` routed experts of each token, ranked by `sigmoid(logit) + bias`.
///
/// **A threadgroup to a token and `top_k` passes over its row**, rather than an
/// ordering of the 256. Six passes over a row each thread holds one element of
/// is six coalesced reads and six tree reductions; any sort of 256 is more work
/// and needs somewhere to put it. A pass skips what the passes before it took by
/// scanning the handful already written, which for six of 256 is cheaper than a
/// mask over the row.
///
/// **Ties go to the lower index**, which is what `mx.argpartition` was measured
/// to do and what a stable descending sort does. That survives the reduction
/// because the reduction is two: the largest score, and then the smallest index
/// holding it.
///
/// `n_routed` stands in for "this thread found nothing", which is what makes the
/// index written always an expert some thread actually looked at — a thread
/// whose whole stripe was already taken has no candidate to offer, and a
/// comparison against a NaN score is false either way round.
///
/// The shared experts are the *last* `width - n_routed` of a row and are never
/// ranked: they are read by every token whatever the gate says.
kernel void router_top_k(
    constant Shape &shape [[buffer(0)]],
    device const float *logits [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device uint *chosen [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]
) {
    threadgroup float bests[MOST_SIMDGROUPS];
    threadgroup uint ats[MOST_SIMDGROUPS];
    threadgroup uint picked[MOST_SLOTS];

    // A whole threadgroup turns away or none of it does, which is what makes the
    // barriers below uniform. The grid gives one threadgroup to each token, so
    // this is unreachable; a bounds check on `local` instead would leave some
    // threads at a barrier and others past it, which is undefined rather than
    // slow.
    if (token >= shape.tokens) {
        return;
    }
    device const float *row = logits + (ulong)token * shape.width;

    for (uint slot = 0; slot < shape.top_k; ++slot) {
        float best = -INFINITY;
        uint at = shape.n_routed;
        for (uint expert = local; expert < shape.n_routed; expert += threads) {
            bool taken = false;
            for (uint seen = 0; seen < slot; ++seen) {
                taken = taken || picked[seen] == expert;
            }
            if (taken) {
                continue;
            }
            const float score = 1.0f / (1.0f + exp(-row[expert])) + bias[expert];
            if (at == shape.n_routed || score > best) {
                best = score;
                at = expert;
            }
        }

        // The largest score this simdgroup held, and then the lowest index
        // holding it — which is the tie rule, stated as a second reduction
        // because a maximum over scores alone cannot express it.
        const float top = simd_max(at == shape.n_routed ? -INFINITY : best);
        const bool holds = at != shape.n_routed && best == top;
        const uint first = simd_min(holds ? at : shape.n_routed);

        if (lane == 0) {
            bests[simd] = top;
            ats[simd] = first;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The same pass over the simdgroups' partials in every thread, which is
        // cheaper than reducing them once and broadcasting at 32 entries and
        // needs one barrier rather than two.
        float overall = -INFINITY;
        uint chose = shape.n_routed;
        for (uint s = 0; s < simds; ++s) {
            if (ats[s] == shape.n_routed) {
                continue;
            }
            if (chose == shape.n_routed || bests[s] > overall
                || (bests[s] == overall && ats[s] < chose)) {
                overall = bests[s];
                chose = ats[s];
            }
        }
        if (local == 0) {
            picked[slot] = chose;
            chosen[(ulong)token * shape.top_k + slot] = chose;
        }
        // Two barriers a pass and not three. This one both publishes `picked`
        // to the next pass's skip and says every thread has finished reading the
        // partials — so the next pass may overwrite them without one of its own.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// The weighting. What it computes is `SparseMoe::weigh`'s two lines, and every
/// clause of them is one of the ways of misreading this gate that
/// [`inkling_core::moe`] pins:
///
/// - the weight of a chosen expert is `log(sigmoid(raw logit))`, the correction
///   bias having ranked the experts in the dispatch before this one and taken no
///   further part;
/// - one softmax spans the `top_k` picked routed experts and all `n_shared`
///   shared ones together, so a token's eight weights sum to one scale rather
///   than to two;
/// - and that scale is `route_scale * global_scale`, arriving as the one number
///   both are already multiplied into.
///
/// The shared experts are the *last* `width - n_routed` of a row, which is the
/// fourth way and is the same reading [`BODY`] takes of the same row.
pub(crate) const WEIGHTS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Weighing {
    uint tokens;
    uint n_routed;
    uint width;
    uint top_k;
    uint n_shared;
};

/// `log(sigmoid(x))`, which mlx-vlm writes as `-logaddexp(0, -x)` and
/// `inkling_core::moe` as this.
///
/// Shifted for the reason both of those are: `exp(-x)` overflows below about
/// -88 in float32, where the answer is simply `x`, and the naive form would
/// return an infinity through a softmax that has no way back from one.
///
/// The CPU writes the second term as `ln_1p` and this writes it out, because the
/// Metal standard library has no `log1p`. They part company only where `exp(-|x|)`
/// is small enough to vanish into the 1 it is added to, which is half an ulp of
/// a value already below 2^-24 — under the bound the two `exp`s themselves need.
inline float log_sigmoid(float x) {
    return fmin(x, 0.0f) - log(1.0f + exp(-fabs(x)));
}

/// The weight each of a token's chosen experts carries, and each of the shared
/// ones — one thread to a token.
///
/// **The bias is not here and that is the first claim.** `chosen` is read for
/// which logits to weight and for nothing else: the weights come from the raw
/// logits, where the selection that produced `chosen` ranked
/// `sigmoid(logit) + bias`. A kernel handed the bias could add it a second time
/// and would pick the same experts while weighting them wrong.
///
/// **One softmax over `top_k + n_shared` and that is the second.** The shifted
/// form is `inkling_core::ops::softmax`'s, which is what `exp(x - logsumexp(x))`
/// is written out. Normalising the routed and the shared apart leaves each group
/// summing to the full scale and the shared experts three times too heavy.
///
/// **Both scales multiply what comes out and that is the third.** `scale` is
/// `route_scale * global_scale`, so the eight weights of a token sum to it.
kernel void router_weights(
    constant Weighing &shape [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    device const float *logits [[buffer(2)]],
    device const uint *chosen [[buffer(3)]],
    device float *weights [[buffer(4)]],
    device float *shared [[buffer(5)]],
    uint token [[thread_position_in_grid]]
) {
    if (token >= shape.tokens) {
        return;
    }
    device const float *row = logits + (ulong)token * shape.width;
    device const uint *picked = chosen + (ulong)token * shape.top_k;

    float weighed[MOST_WEIGHTS];
    for (uint slot = 0; slot < shape.top_k; ++slot) {
        weighed[slot] = log_sigmoid(row[picked[slot]]);
    }
    for (uint expert = 0; expert < shape.n_shared; ++expert) {
        weighed[shape.top_k + expert] = log_sigmoid(row[shape.n_routed + expert]);
    }

    const uint spanned = shape.top_k + shape.n_shared;
    float peak = -INFINITY;
    for (uint i = 0; i < spanned; ++i) {
        peak = fmax(peak, weighed[i]);
    }
    float total = 0.0f;
    for (uint i = 0; i < spanned; ++i) {
        weighed[i] = exp(weighed[i] - peak);
        total += weighed[i];
    }

    for (uint slot = 0; slot < shape.top_k; ++slot) {
        weights[(ulong)token * shape.top_k + slot] = weighed[slot] / total * scale;
    }
    for (uint expert = 0; expert < shape.n_shared; ++expert) {
        shared[(ulong)token * shape.n_shared + expert] =
            weighed[shape.top_k + expert] / total * scale;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::fixture::deviation;
    use inkling_core::moe::{Gate, GateWeights, SparseMoe};

    use crate::testing::{GLOBAL_SCALE, ROUTING as CONFIG, correction_bias, device, gate_logits};

    /// Tokens enough that the dispatch spans several threadgroups, and not a
    /// multiple of the threadgroup size, so the tail group runs threads the
    /// bounds check has to turn away.
    const TOKENS: usize = THREADS_PER_GROUP * 2 + 13;

    /// How far a weight a dispatch computed may land from the CPU's.
    ///
    /// Both sides take the same eight logits through the same `log(sigmoid(x))`
    /// and the same shifted softmax, so what separates them is `exp` and `log`
    /// — the GPU's and libm's, which agree to a few ulps rather than exactly —
    /// over a chain of three of them. Worst observed when this landed: 3.0e-7,
    /// a factor of three in hand. The weakest mutation these tests rely on
    /// catching, the shared logits read off the front of the row, moves the
    /// routed weights by 2.3e-1 — six decades above.
    const TOLERANCE: f32 = 1e-6;

    /// A router with `bias` and no gate of its own — the layer's weight is not
    /// what any of this is about.
    fn on_the_cpu(bias: &[f32]) -> SparseMoe<'_> {
        SparseMoe::new(
            CONFIG,
            GateWeights {
                gate: Gate::Backend { hidden: 4096 },
                correction_bias: bias,
                global_scale: GLOBAL_SCALE,
            },
        )
    }

    /// The router's two kernels, compiled together — which is what a layer is
    /// stood up from, and what lets a case put a mutant of one of them beside
    /// the real other.
    struct Kernels {
        select: Router,
        weigh: RouterWeights,
    }

    impl Kernels {
        fn compile(device: &Device) -> Self {
            Self {
                select: Router::new(device).expect("the router compiles"),
                weigh: RouterWeights::new(device).expect("the weighting compiles"),
            }
        }

        fn layer<'a>(&'a self, device: &'a Device, bias: &[f32]) -> LayerRouter<'a> {
            self.scaled(device, bias, GLOBAL_SCALE)
        }

        fn scaled<'a>(
            &'a self,
            device: &'a Device,
            bias: &[f32],
            global_scale: f32,
        ) -> LayerRouter<'a> {
            LayerRouter::new(
                device,
                &self.select,
                &self.weigh,
                CONFIG,
                bias,
                global_scale,
            )
            .expect("the router stands up")
        }
    }

    /// The weights the CPU gives a selection, which is what the kernel is
    /// measured against: `SparseMoe::weigh`, the function every fixture in
    /// `inkling_core::moe` holds to mlx-vlm.
    fn weighed(bias: &[f32], logits: &[f32], picked: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let routing = on_the_cpu(bias).weigh(logits, &widened(picked));
        (
            routing.weights().to_vec(),
            routing.shared_weights().to_vec(),
        )
    }

    /// The set a token's slots name, which is the whole of what the reference
    /// promises and so the whole of what can be compared.
    fn as_sets(picked: &[usize], top_k: usize) -> Vec<Vec<usize>> {
        picked
            .chunks_exact(top_k)
            .map(|row| {
                let mut row = row.to_vec();
                row.sort_unstable();
                row
            })
            .collect()
    }

    fn widened(chosen: &[u32]) -> Vec<usize> {
        chosen.iter().map(|expert| *expert as usize).collect()
    }

    /// One case's logits and the bias they are ranked under.
    fn case(seed: usize) -> (Vec<f32>, Vec<f32>) {
        (gate_logits(TOKENS, seed), correction_bias())
    }

    /// The whole claim: the kernel selects the set `SparseMoe::route` selects,
    /// which is the set `mx.argpartition` was measured to select.
    #[test]
    fn a_dispatch_selects_the_experts_the_cpu_selects() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(0);
        let layer = kernels.layer(&device, &bias);

        let got = layer.select(&logits).expect("the dispatch completes");
        assert_eq!(got.len(), layer.assignments(TOKENS));

        let want = on_the_cpu(&bias).route(&logits);
        assert_eq!(
            as_sets(&widened(&got), CONFIG.top_k),
            as_sets(want.experts(), CONFIG.top_k),
            "the selected sets"
        );

        // A case where every token selected the same six would say nothing about
        // whether the row index reached the kernel.
        let sets = as_sets(&widened(&got), CONFIG.top_k);
        assert!(sets.iter().any(|row| *row != sets[0]), "{:?}", sets[0]);
    }

    /// The correction bias ranks, which is the one input to this kernel that a
    /// router still selecting six of 256 can silently drop.
    #[test]
    fn dropping_the_correction_bias_selects_different_experts() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(0);
        let flat = vec![0.0; CONFIG.n_routed];

        let biased = kernels
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");
        let unbiased = kernels
            .layer(&device, &flat)
            .select(&logits)
            .expect("the dispatch completes");

        assert_ne!(
            as_sets(&widened(&biased), CONFIG.top_k),
            as_sets(&widened(&unbiased), CONFIG.top_k)
        );
    }

    /// A tie at the last slot goes to the lower index, which is what the
    /// reference was measured to do over sixteen hundred tied rows and what a
    /// kernel comparing the other way round would get backwards on every one of
    /// them.
    ///
    /// Every expert of the row carries the same logit and the same bias, so all
    /// 256 scores agree bit for bit and the selection is entirely the tie rule.
    #[test]
    fn a_row_of_ties_is_selected_in_index_order() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let width = CONFIG.n_routed + CONFIG.n_shared;
        let logits = vec![0.25; width];
        let bias = vec![0.5; CONFIG.n_routed];

        let got = kernels
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");

        assert_eq!(widened(&got), (0..CONFIG.top_k).collect::<Vec<usize>>());
        assert_eq!(
            widened(&got),
            on_the_cpu(&bias).route(&logits).experts(),
            "the CPU breaks the same tie the same way"
        );
    }

    /// A kernel that took the highest index of a tied run rather than the
    /// lowest, which is the mutation the case above exists to catch — the same
    /// six-of-256 selection, off by one expert wherever two scores agree.
    ///
    /// The tie rule is the second of the two reductions, so that is where the
    /// mutation goes: the largest score is still the largest score, and what
    /// changes is which of the indices holding it is kept.
    #[test]
    fn taking_the_last_of_a_tied_run_selects_different_experts() {
        let Some(device) = device() else { return };
        let source = source();
        let reversed = source.replace(
            "simd_min(holds ? at : shape.n_routed)",
            "simd_max(holds ? at : 0)",
        );
        assert_ne!(reversed, source, "the mutation changed nothing");
        let kernels = Kernels {
            select: Router::from_source(&device, &reversed).expect("the mutant compiles"),
            weigh: RouterWeights::new(&device).expect("the weighting compiles"),
        };

        let width = CONFIG.n_routed + CONFIG.n_shared;
        let logits = vec![0.25; width];
        let bias = vec![0.5; CONFIG.n_routed];
        let got = kernels
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");

        let want: Vec<usize> = (0..CONFIG.top_k).collect();
        assert_ne!(widened(&got), want, "the tie went the same way");
        let mut distinct = widened(&got);
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            CONFIG.top_k,
            "a mutation that stopped selecting six of 256 would prove nothing: {got:?}"
        );
    }

    /// The shared experts are the last rows of the gate and are never ranked. A
    /// kernel that scanned the whole row would select one of them wherever its
    /// logit was high enough, which is an index the routed bank does not hold.
    #[test]
    fn the_shared_logits_are_never_selected() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let width = CONFIG.n_routed + CONFIG.n_shared;
        let (mut logits, bias) = case(7);
        for token in 0..TOKENS {
            for shared in 0..CONFIG.n_shared {
                logits[token * width + CONFIG.n_routed + shared] = 1e3;
            }
        }

        let got = kernels
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");

        let past = got
            .iter()
            .find(|expert| **expert as usize >= CONFIG.n_routed);
        assert_eq!(past, None, "a shared expert was routed to");
    }

    /// One row's `top_k` slots are distinct, which the skip inside the kernel is
    /// the whole of: a pass that did not skip what the passes before it took
    /// would name the same best expert six times and run one expert's rows six
    /// times over.
    #[test]
    fn a_tokens_slots_name_six_different_experts() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(3);

        let got = kernels
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");

        for (token, row) in got.chunks_exact(CONFIG.top_k).enumerate() {
            let mut distinct = row.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), CONFIG.top_k, "token {token}: {row:?}");
        }
    }

    /// A bias of the wrong length is a router paired with another layer's gate,
    /// which the kernel would read past the end of rather than fault on.
    #[test]
    #[should_panic(expected = "one correction bias per routed expert")]
    fn a_bias_that_is_not_one_per_routed_expert_is_refused() {
        let Some(device) = device() else {
            panic!("one correction bias per routed expert: no device to ask")
        };
        let kernels = Kernels::compile(&device);
        let _ = kernels.layer(&device, &[0.0; 8]);
    }

    /// The whole claim of the second kernel: the weights a dispatch gives a
    /// selection are the weights `SparseMoe::weigh` gives it, which is the
    /// function `inkling_core::moe`'s fixtures hold to mlx-vlm.
    ///
    /// Both halves, because they are two buffers and only one of them is
    /// `top_k` long: the routed weights and the shared ones come out of one
    /// softmax and a kernel could reproduce either alone.
    #[test]
    fn a_dispatch_weights_a_selection_the_way_the_cpu_weights_it() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(0);
        let layer = kernels.layer(&device, &bias);

        let picked = layer.select(&logits).expect("the dispatch completes");
        let got = layer
            .weigh(&logits, &picked)
            .expect("the dispatch completes");
        assert_eq!(got.routed.len(), layer.assignments(TOKENS));
        assert_eq!(got.shared.len(), TOKENS * CONFIG.n_shared);

        let (routed, shared) = weighed(&bias, &logits, &picked);
        for (what, got, want) in [
            ("routed", got.routed.to_vec(), routed),
            ("shared", got.shared.to_vec(), shared),
        ] {
            let deviation = deviation(&got, &want);
            eprintln!("{TOKENS} tokens weighted: {what} deviation {deviation:e}");
            assert!(deviation <= TOLERANCE, "{what}: deviation {deviation:e}");
        }
    }

    /// **The correction bias selects and does not weight.** It ranked the
    /// experts in the dispatch before this one; a weighting that let it through
    /// would pick the same six and scale them wrong.
    ///
    /// Stated as two routers over one selection rather than as a mutant kernel,
    /// because the bias is not bound to this dispatch at all — so the mistake
    /// this has to rule out is not in the source it could mutate. Two biases
    /// that rank differently, one selection, and the weights have to agree bit
    /// for bit.
    #[test]
    fn the_correction_bias_does_not_reach_the_weights() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(0);
        // The same values against the other end of the row, which ranks the
        // experts differently — where a bias shifted by a constant would rank
        // them identically and prove nothing.
        let elsewhere: Vec<f32> = bias.iter().rev().copied().collect();

        let picked = kernels
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");
        let ranked_elsewhere = kernels
            .layer(&device, &elsewhere)
            .select(&logits)
            .expect("the dispatch completes");
        assert_ne!(
            picked, ranked_elsewhere,
            "two biases that ranked alike would prove nothing"
        );

        let weighed = |bias: &[f32]| {
            let weights = kernels
                .layer(&device, bias)
                .weigh(&logits, &picked)
                .expect("the dispatch completes");
            (weights.routed.to_vec(), weights.shared.to_vec())
        };
        assert_eq!(weighed(&bias), weighed(&elsewhere));
    }

    /// **One softmax spans the picked routed experts and both shared ones**, so
    /// a token's eight weights sum to one scale. Normalising the two groups
    /// apart would leave each of them summing to it, which is a router that
    /// still runs and reads the shared experts three times too heavily.
    ///
    /// **And that scale is both of them.** `route_scale` is 8 and the trained
    /// `global_scale` is 0.00704, so the sum is 0.0563 — and a router carrying
    /// `route_scale` alone lands 142 times above it, which is the largest single
    /// error any of these traps produces.
    #[test]
    fn a_tokens_weights_sum_to_both_scales_together() {
        let Some(device) = device() else { return };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(0);
        let layer = kernels.layer(&device, &bias);
        let picked = layer.select(&logits).expect("the dispatch completes");

        let scale = CONFIG.route_scale * GLOBAL_SCALE;
        assert_eq!(layer.scale(), scale, "both scales, multiplied");
        let got = layer
            .weigh(&logits, &picked)
            .expect("the dispatch completes");
        for token in 0..TOKENS {
            let total: f32 = got.routed.as_slice()[token * CONFIG.top_k..][..CONFIG.top_k]
                .iter()
                .chain(&got.shared.as_slice()[token * CONFIG.n_shared..][..CONFIG.n_shared])
                .sum();
            assert!(
                (total - scale).abs() <= TOLERANCE * scale,
                "token {token} sums to {total}, not {scale}"
            );
        }

        // The same selection weighted without the learned scale, which is what a
        // port reading `route_scale` alone computes.
        let unscaled = kernels
            .scaled(&device, &bias, 1.0)
            .weigh(&logits, &picked)
            .expect("the dispatch completes");
        let hot: f32 = unscaled.routed.as_slice()[..CONFIG.top_k].iter().sum();
        let cool: f32 = got.routed.as_slice()[..CONFIG.top_k].iter().sum();
        eprintln!(
            "without global_scale a token's routed weights are {:.0}x",
            hot / cool
        );
        assert!(
            ((hot / cool) - GLOBAL_SCALE.recip()).abs() < 1.0,
            "{hot} against {cool}"
        );
    }

    /// **The shared experts are the last rows of the gate**, in the weighting as
    /// in the selection. Read off the front they are two routed experts, the
    /// softmax still spans eight logits, and every weight is wrong.
    #[test]
    fn weighting_the_first_rows_as_the_shared_experts_changes_the_answer() {
        let Some(device) = device() else { return };
        let real = Kernels::compile(&device);
        let source = weights_source();
        let flipped = source.replace("row[shape.n_routed + expert]", "row[expert]");
        assert_ne!(flipped, source, "the mutation changed nothing");
        let mutant = Kernels {
            select: Router::new(&device).expect("the router compiles"),
            weigh: RouterWeights::from_source(&device, &flipped).expect("the mutant compiles"),
        };

        let (logits, bias) = case(0);
        let picked = real
            .layer(&device, &bias)
            .select(&logits)
            .expect("the dispatch completes");
        let weighed = |kernels: &Kernels| {
            let weights = kernels
                .layer(&device, &bias)
                .weigh(&logits, &picked)
                .expect("the dispatch completes");
            (weights.routed.to_vec(), weights.shared.to_vec())
        };

        let (routed, shared) = weighed(&real);
        let (off_the_front, its_shared) = weighed(&mutant);
        for (what, got, want) in [
            ("routed", off_the_front, routed),
            ("shared", its_shared, shared),
        ] {
            let deviation = deviation(&got, &want);
            eprintln!("the shared logits read off the front: {what} {deviation:e}");
            assert!(deviation > TOLERANCE, "{what}: deviation {deviation:e}");
        }
    }

    /// A selection that is not `top_k` for every token, which is a router paired
    /// with another layer's shape — and would otherwise weight whatever the rows
    /// of the two buffers happened to line up with.
    #[test]
    #[should_panic(expected = "selected experts are not")]
    fn weighting_a_selection_that_is_not_one_per_slot_per_token_is_refused() {
        let Some(device) = device() else {
            panic!("selected experts are not: no device to ask")
        };
        let kernels = Kernels::compile(&device);
        let (logits, bias) = case(0);
        let layer = kernels.layer(&device, &bias);
        let picked = layer.select(&logits).expect("the dispatch completes");
        let _ = layer.weigh(&logits, &picked[..CONFIG.top_k]);
    }
}
