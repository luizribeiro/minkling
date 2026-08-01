//! Which six of 256 experts a token reads, chosen where its gate logits were
//! computed.
//!
//! **The selection is the half of the router that decides a dispatch, and the
//! only half that has to be here.** What `InklingSparseMoE` does with its gate
//! is two things: rank the experts under `sigmoid(logit) +
//! e_score_correction_bias` and take the best `top_k`, and then weight what it
//! took. The weighting reads the *raw* logits, spans the picked routed experts
//! and both shared ones in one softmax, and carries two scales — three of the
//! four ways of misreading the gate that [`inkling_core::moe`] pins, and none of
//! them is worth moving: it is a softmax over eight numbers a layer, and every
//! fixture that holds it to mlx-vlm is on that side. The selection is what the
//! routed bank's rows are named by, so it is what a bank cannot be dispatched
//! without.
//!
//! **What that buys is the CPU leaving the middle of a MoE layer.** The gate's
//! multiply, the top-k over what it produced, both banks and the activation
//! inside each are one command buffer, because every value between them is a
//! buffer the next dispatch reads and no dispatch waits for an answer from this
//! side. What crosses back at the end is the logits, the six indices a token
//! chose, and the rows the two banks answered.
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

/// One layer's router: the shape it chooses under, and the correction bias it
/// ranks with.
///
/// The bias is `[n_routed]` and is copied rather than wrapped, for the reason a
/// norm weight is: it is bfloat16 in the checkpoint and the kernel wants
/// float32, so there are no packed bytes here to hand over in place. 1 KB a
/// layer, copied once.
#[derive(Debug)]
pub struct LayerRouter<'a> {
    device: &'a Device,
    router: &'a Router,
    config: MoeConfig,
    /// Held behind a cell for the reason [`crate::PackedBank`]'s resident
    /// tensors are: binding a buffer to a dispatch borrows it exclusively, and
    /// the bias belongs to the layer rather than to a call.
    bias: RefCell<Buffer<f32>>,
}

impl<'a> LayerRouter<'a> {
    /// A router over `config`, ranking with `correction_bias`.
    pub fn new(
        device: &'a Device,
        router: &'a Router,
        config: MoeConfig,
        correction_bias: &[f32],
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
        Ok(Self {
            bias: RefCell::new(device.buffer(correction_bias)?),
            device,
            router,
            config,
        })
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

    /// The same selection submitted on its own, for a caller with nothing to
    /// batch it against — which is the cases here and nothing in the engine.
    pub fn select(&self, logits: &[f32]) -> Result<Vec<u32>, MetalError> {
        let mut input = self.device.buffer(logits)?;
        let mut batch = self.device.batch()?;
        let chosen = self.encode(&mut batch, &mut input)?;
        batch.wait()?;
        Ok(chosen.to_vec())
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

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::moe::{Gate, GateWeights, SparseMoe};

    use crate::testing::device;

    /// The checkpoint's own shape, so that the kernel is exercised over the row
    /// it will actually rank: 256 routed experts, two shared, six per token.
    const CONFIG: MoeConfig = MoeConfig {
        n_routed: 256,
        n_shared: 2,
        top_k: 6,
        route_scale: 8.0,
    };

    /// Tokens enough that the dispatch spans several threadgroups, and not a
    /// multiple of the threadgroup size, so the tail group runs threads the
    /// bounds check has to turn away.
    const TOKENS: usize = THREADS_PER_GROUP * 2 + 13;

    /// A router with `bias` and no gate of its own — the layer's weight is not
    /// what any of this is about.
    fn on_the_cpu(bias: &[f32]) -> SparseMoe<'_> {
        SparseMoe::new(
            CONFIG,
            GateWeights {
                gate: Gate::Backend { hidden: 4096 },
                correction_bias: bias,
                global_scale: 0.007_042_432_7,
            },
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

    /// Logits and a bias spread over both signs and over the range where
    /// `sigmoid` is not saturated, so that the bias decides some of the ranking
    /// and the logits decide the rest.
    fn case(seed: usize) -> (Vec<f32>, Vec<f32>) {
        let width = CONFIG.n_routed + CONFIG.n_shared;
        (
            (0..TOKENS * width)
                .map(|i| ((i * 37 + seed) % 401) as f32 / 40.0 - 5.0)
                .collect(),
            (0..CONFIG.n_routed)
                .map(|i| ((i * 53) % 97) as f32 / 400.0 - 0.12)
                .collect(),
        )
    }

    /// The whole claim: the kernel selects the set `SparseMoe::route` selects,
    /// which is the set `mx.argpartition` was measured to select.
    #[test]
    fn a_dispatch_selects_the_experts_the_cpu_selects() {
        let Some(device) = device() else { return };
        let router = Router::new(&device).expect("the router compiles");
        let (logits, bias) = case(0);
        let layer =
            LayerRouter::new(&device, &router, CONFIG, &bias).expect("the router stands up");

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
        let router = Router::new(&device).expect("the router compiles");
        let (logits, bias) = case(0);
        let flat = vec![0.0; CONFIG.n_routed];

        let biased = LayerRouter::new(&device, &router, CONFIG, &bias)
            .expect("the router stands up")
            .select(&logits)
            .expect("the dispatch completes");
        let unbiased = LayerRouter::new(&device, &router, CONFIG, &flat)
            .expect("the router stands up")
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
        let router = Router::new(&device).expect("the router compiles");
        let width = CONFIG.n_routed + CONFIG.n_shared;
        let logits = vec![0.25; width];
        let bias = vec![0.5; CONFIG.n_routed];

        let got = LayerRouter::new(&device, &router, CONFIG, &bias)
            .expect("the router stands up")
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
        let mutant = Router::from_source(&device, &reversed).expect("the mutant compiles");

        let width = CONFIG.n_routed + CONFIG.n_shared;
        let logits = vec![0.25; width];
        let bias = vec![0.5; CONFIG.n_routed];
        let got = LayerRouter::new(&device, &mutant, CONFIG, &bias)
            .expect("the router stands up")
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
        let router = Router::new(&device).expect("the router compiles");
        let width = CONFIG.n_routed + CONFIG.n_shared;
        let (mut logits, bias) = case(7);
        for token in 0..TOKENS {
            for shared in 0..CONFIG.n_shared {
                logits[token * width + CONFIG.n_routed + shared] = 1e3;
            }
        }

        let got = LayerRouter::new(&device, &router, CONFIG, &bias)
            .expect("the router stands up")
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
        let router = Router::new(&device).expect("the router compiles");
        let (logits, bias) = case(3);

        let got = LayerRouter::new(&device, &router, CONFIG, &bias)
            .expect("the router stands up")
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
        let router = Router::new(&device).expect("the router compiles");
        let _ = LayerRouter::new(&device, &router, CONFIG, &[0.0; 8]);
    }
}
