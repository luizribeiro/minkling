//! Both banks' rows weighted and summed back into `[tokens, hidden]`, which is
//! what a MoE layer returns.
//!
//! **This is the scatter, and there is nothing to scatter.** On the CPU the
//! routed bank's rows arrive grouped by expert, so putting them back means
//! walking a list that says which token each row came from — see
//! `SparseMoe::scatter`. A bank dispatched from a selection the device made
//! never regrouped them: its rows are token-major, one per slot, so the row a
//! token's slot produced is at `token * top_k + slot` and the list is arithmetic
//! rather than a value. The shared bank's are the other reading of the same
//! thing, every token once per shared expert, so its rows are at
//! `expert * tokens + token`.
//!
//! **The two halves are summed here and not read apart.** `InklingSparseMoE`
//! returns them separately and [`MoeOutput`](inkling_core::moe::MoeOutput) keeps
//! them that way, because the reference records them separately and the gated
//! fixtures pin both. Nothing downstream of *this* reads either alone: what the
//! layer's second residual path is handed is their sum, so a dispatch that
//! produced two tensors for the CPU to add would be an allocation and a pass to
//! hand back a value nobody wants.
//!
//! **A thread to an element of the output**, which is `tokens * hidden` threads
//! reading `top_k + n_shared` weighted rows each. There is no reduction across
//! threads and so no barrier: every thread owns one column of one token and no
//! thread reads what another writes.

use inkling_core::profile::{self, Op};

use crate::buffer::Buffer;
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};
use crate::router::RoutingWeights;

const ENTRY: &str = "moe_combine";

/// Threads one threadgroup of a dispatch holds. One thread to one element of
/// the output, like [`crate::SwiGlu`] and for the same reason: this reduces over
/// eight rows in a thread and over nothing across threads.
const THREADS_PER_GROUP: usize = 256;

/// The compiled kernel, which every MoE layer on a device shares.
///
/// Per source string rather than per layer, like [`crate::PackedMatmul`] and
/// [`crate::SwiGlu`]: the source names no shape, so one of these serves all
/// forty layers that route.
#[derive(Debug)]
pub struct MoeCombine {
    kernel: Kernel,
}

impl MoeCombine {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, BODY)
    }

    /// [`MoeCombine::new`] out of a source string of the caller's own, which is
    /// how a test puts a deliberately wrong kernel through the same plumbing as
    /// the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }

    /// The layer's `[tokens, hidden]` output, encoded into `batch` over four
    /// buffers dispatches already left there.
    ///
    /// **Every shape is derived and cross-checked rather than passed**, because
    /// the four buffers are four halves of two pairs and the mistake worth
    /// catching is one of them belonging to another call. `top_k` and `n_shared`
    /// come off the two weight buffers, the hidden width off the routed bank's
    /// rows against the weights that name them, and the shared bank's rows have
    /// to be the width and the count both of those imply. A pair that disagreed
    /// would be read off the end of whichever ran out first, which a GPU answers
    /// with whatever is there.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        tokens: usize,
        weights: &mut RoutingWeights,
        routed: &mut Buffer<f32>,
        shared: &mut Buffer<f32>,
    ) -> Result<Buffer<f32>, MetalError> {
        let _timed = profile::scope(Op::Encode);
        assert!(tokens > 0, "a call has tokens");
        let per_token = |what: &str, weights: &Buffer<f32>| {
            assert_eq!(
                weights.len() % tokens,
                0,
                "{} {what} weights are not whole rows of {tokens} tokens",
                weights.len()
            );
            weights.len() / tokens
        };
        let top_k = per_token("routed", &weights.routed);
        let n_shared = per_token("shared", &weights.shared);
        assert_eq!(
            routed.len() % weights.routed.len(),
            0,
            "{} routed values are not whole rows of the {} weights that scale them",
            routed.len(),
            weights.routed.len()
        );
        let hidden = routed.len() / weights.routed.len();
        assert_eq!(
            shared.len(),
            n_shared * tokens * hidden,
            "{} shared values are not {n_shared} passes of {tokens} rows of {hidden}",
            shared.len()
        );

        let fields = [
            extent(tokens, "the tokens of a call"),
            extent(hidden, "the width a bank maps back to"),
            extent(top_k, "the experts a token reads"),
            extent(n_shared, "the experts every token reads"),
        ];
        let mut shape = batch.device().inline(&fields)?;
        let mut out = batch.device().zeroed::<f32>(tokens * hidden)?;
        let moves = size_of::<f32>()
            * (weights.routed.len()
                + weights.shared.len()
                + routed.len()
                + shared.len()
                + out.len());
        batch.add(
            &self.kernel,
            &[
                shape.arg(),
                weights.routed.arg(),
                weights.shared.arg(),
                routed.arg(),
                shared.arg(),
                out.arg(),
            ],
            Grid::new(tokens * hidden, THREADS_PER_GROUP),
            moves,
        )?;
        Ok(out)
    }
}

/// The kernel. No constant of this crate's decides anything here — the tokens,
/// the width and both expert counts are a call's — so the source is the whole of
/// it.
pub(crate) const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint tokens;
    uint hidden;
    uint top_k;
    uint n_shared;
};

/// One element of one token's output: its `top_k` routed rows and its
/// `n_shared` shared ones, each scaled by the weight the routing gave it.
///
/// **The two banks are accumulated apart and added once**, which is the order
/// `SparseMoe::scattered` sums them in — a routed half and a shared half, added
/// on the way out. Float addition is not associative, so a single accumulator
/// walking all eight would be a different answer from the CPU that is the oracle
/// for this kernel by a few ulps, for nothing.
///
/// **The rows of the two banks are indexed differently and that is not a
/// symmetry this lost.** A token's routed rows are consecutive, one per slot it
/// selected, because the bank was dispatched from the selection itself; the
/// shared bank's are `tokens` apart, because every token goes through every
/// shared expert and the bank ran the hidden state once per expert.
kernel void moe_combine(
    constant Shape &shape [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *shared_weights [[buffer(2)]],
    device const float *routed [[buffer(3)]],
    device const float *shared [[buffer(4)]],
    device float *out [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= shape.tokens * shape.hidden) {
        return;
    }
    const uint token = id / shape.hidden;
    const uint channel = id % shape.hidden;

    float chosen = 0.0f;
    for (uint slot = 0; slot < shape.top_k; ++slot) {
        const ulong row = (ulong)token * shape.top_k + slot;
        chosen += weights[row] * routed[row * shape.hidden + channel];
    }

    float always = 0.0f;
    for (uint expert = 0; expert < shape.n_shared; ++expert) {
        const ulong row = (ulong)expert * shape.tokens + token;
        always += shared_weights[(ulong)token * shape.n_shared + expert]
            * shared[row * shape.hidden + channel];
    }

    out[id] = chosen + always;
}
"#;

#[cfg(test)]
mod tests {
    use inkling_core::fixture::deviation;
    use inkling_core::moe::{BankRows, Gate, GateWeights, SparseMoe};

    use super::*;
    use crate::router::{LayerRouter, Router, RouterWeights};
    use crate::testing::{GLOBAL_SCALE, ROUTING as CONFIG, correction_bias, device, gate_logits};

    /// Tokens enough that the dispatch spans several threadgroups over a narrow
    /// width, and not a multiple of anything here.
    const TOKENS: usize = 5;

    /// The width a bank maps back to, cut down from 4096: wide enough that the
    /// dispatch runs more than one threadgroup and narrow enough to hold the
    /// rows in a test.
    const HIDDEN: usize = 129;

    /// Both sides multiply the same weights by the same rows and add them in the
    /// same order, so what separates them is that Metal may contract `a += w * y`
    /// to an FMA where the CPU rounds twice, and that the weights themselves came
    /// out of two `exp`s that are not libm's. Worst observed when this landed:
    /// 2.1e-7, a factor of five in hand. The weakest mutation these tests rely on
    /// catching, the shared bank's rows read token-major, moves the answer by
    /// 6.1e-1 — six decades above.
    const TOLERANCE: f32 = 1e-6;

    /// The router the weights come from, which is the layer's own.
    fn on_the_cpu(bias: &[f32]) -> SparseMoe<'_> {
        SparseMoe::new(
            CONFIG,
            GateWeights {
                gate: Gate::Backend { hidden: HIDDEN },
                correction_bias: bias,
                global_scale: GLOBAL_SCALE,
            },
        )
    }

    /// The rows a bank answered, one per assignment, spread over both signs.
    fn rows(count: usize, seed: usize) -> Vec<f32> {
        (0..count * HIDDEN)
            .map(|i| ((i * 29 + seed) % 71) as f32 / 35.0 - 1.0)
            .collect()
    }

    /// The whole claim: the two banks' rows weighted by the device and summed by
    /// the device are what `SparseMoe` weights and sums them to.
    ///
    /// Driven through the router's own weighting rather than from weights this
    /// side made up, because the two dispatches are what the layer runs and the
    /// order the second reads the first's output in is half of what is being
    /// asserted.
    #[test]
    fn a_dispatch_weights_and_sums_what_the_cpu_weights_and_sums() {
        let Some(device) = device() else { return };
        let kernel = MoeCombine::new(&device).expect("the combine compiles");
        let router = Router::new(&device).expect("the router compiles");
        let weighing = RouterWeights::new(&device).expect("the weighting compiles");
        let bias = correction_bias();
        let layer = LayerRouter::new(&device, &router, &weighing, CONFIG, &bias, GLOBAL_SCALE)
            .expect("the router stands up");

        let logits = gate_logits(TOKENS, 0);
        let routing = on_the_cpu(&bias).route(&logits);
        let picked: Vec<u32> = routing.experts().iter().map(|e| *e as u32).collect();
        let answered = BankRows {
            routed: rows(TOKENS * CONFIG.top_k, 1),
            shared: rows(TOKENS * CONFIG.n_shared, 7),
        };

        let mut logits_on_the_device = device.buffer(&logits).expect("the logits upload");
        let mut picked_on_the_device = device.buffer(&picked).expect("the selection uploads");
        let mut routed = device.buffer(&answered.routed).expect("the rows upload");
        let mut shared = device.buffer(&answered.shared).expect("the rows upload");
        let mut batch = device.batch().expect("a command buffer opens");
        let mut weights = layer
            .encode_weights(
                &mut batch,
                &mut logits_on_the_device,
                &mut picked_on_the_device,
            )
            .expect("the weighting encodes");
        let out = kernel
            .encode(&mut batch, TOKENS, &mut weights, &mut routed, &mut shared)
            .expect("the combine encodes");
        batch.wait().expect("the batch completes");
        let out = out.to_vec();

        let want = on_the_cpu(&bias)
            .weighted(TOKENS * HIDDEN, &logits, routing.experts(), &answered)
            .total();
        assert_eq!(out.len(), want.len());
        let deviation = deviation(&out, &want);
        eprintln!("{TOKENS} tokens combined: deviation {deviation:e}");
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
        assert!(
            out.iter().any(|y| *y != 0.0),
            "an output of zeros would prove nothing"
        );
    }

    /// The shared bank's rows are `tokens` apart and the routed bank's are
    /// consecutive, which is the one thing a kernel over four buffers of
    /// plausible lengths can get backwards while filling the output.
    ///
    /// Reading the shared rows the routed way is what a kernel written once for
    /// both banks would do, and against a single token it would agree — so this
    /// runs the five the case above does.
    #[test]
    fn a_dispatch_declares_both_banks_rows_and_the_weights_that_scale_them() {
        let Some(device) = device() else { return };
        let kernel = MoeCombine::new(&device).expect("the combine compiles");
        let mut weights = RoutingWeights {
            routed: device
                .buffer(&[0.5f32; TOKENS * CONFIG.top_k])
                .expect("the weights upload"),
            shared: device
                .buffer(&[0.5f32; TOKENS * CONFIG.n_shared])
                .expect("the weights upload"),
        };
        let mut routed = device
            .buffer(&rows(TOKENS * CONFIG.top_k, 1))
            .expect("the rows upload");
        let mut shared = device
            .buffer(&rows(TOKENS * CONFIG.n_shared, 7))
            .expect("the rows upload");

        let moved = crate::testing::moved(&device, |batch| {
            kernel
                .encode(batch, TOKENS, &mut weights, &mut routed, &mut shared)
                .expect("the combine encodes");
        });

        let assignments = TOKENS * (CONFIG.top_k + CONFIG.n_shared);
        assert_eq!(
            moved as usize,
            size_of::<f32>() * (assignments + assignments * HIDDEN + TOKENS * HIDDEN)
        );
        // **Eight rows in and one out.** This is where a token's experts stop
        // being eight rows, so a figure that charged the output per assignment
        // rather than per token would be describing a sum nobody takes.
        assert!(
            (moved as usize) < size_of::<f32>() * 2 * assignments * (HIDDEN + 1),
            "the summed row was charged once an assignment"
        );
    }

    /// What the bandwidth column divides by, against what the kernel reads:
    /// both banks' rows, both banks' weights, and one row a token out.
    #[test]
    fn reading_the_shared_rows_token_major_changes_the_answer() {
        let Some(device) = device() else { return };
        let router = Router::new(&device).expect("the router compiles");
        let weighing = RouterWeights::new(&device).expect("the weighting compiles");
        let bias = correction_bias();
        let layer = LayerRouter::new(&device, &router, &weighing, CONFIG, &bias, GLOBAL_SCALE)
            .expect("the router stands up");

        let logits = gate_logits(TOKENS, 0);
        let picked: Vec<u32> = on_the_cpu(&bias)
            .route(&logits)
            .experts()
            .iter()
            .map(|e| *e as u32)
            .collect();
        let answered = BankRows {
            routed: rows(TOKENS * CONFIG.top_k, 1),
            shared: rows(TOKENS * CONFIG.n_shared, 7),
        };

        let through = |kernel: &MoeCombine| {
            let mut logits_on_the_device = device.buffer(&logits).expect("the logits upload");
            let mut picked_on_the_device = device.buffer(&picked).expect("the selection uploads");
            let mut routed = device.buffer(&answered.routed).expect("the rows upload");
            let mut shared = device.buffer(&answered.shared).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let mut weights = layer
                .encode_weights(
                    &mut batch,
                    &mut logits_on_the_device,
                    &mut picked_on_the_device,
                )
                .expect("the weighting encodes");
            let out = kernel
                .encode(&mut batch, TOKENS, &mut weights, &mut routed, &mut shared)
                .expect("the combine encodes");
            batch.wait().expect("the batch completes");
            out.to_vec()
        };

        let want = through(&MoeCombine::new(&device).expect("the combine compiles"));
        let flipped = BODY.replace(
            "const ulong row = (ulong)expert * shape.tokens + token;",
            "const ulong row = (ulong)token * shape.n_shared + expert;",
        );
        assert_ne!(flipped, BODY, "the mutation changed nothing");
        let mutant = MoeCombine::from_source(&device, &flipped).expect("the mutant compiles");

        let deviation = deviation(&through(&mutant), &want);
        eprintln!("the shared rows read token-major: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// Rows and weights that do not pair are the mistake the checks on the way
    /// in exist to catch: the kernel indexes all four under one shape, so
    /// whichever ran out first would be read past the end of its allocation —
    /// which a GPU answers with whatever is there rather than with a fault.
    ///
    /// Both banks, because the two are checked differently: the routed rows are
    /// what the hidden width is *derived* from, so what refuses them is that
    /// they are not whole rows of the weights that scale them, where the shared
    /// rows are then measured against the width and the count both of those
    /// imply.
    #[test]
    fn rows_that_do_not_pair_with_their_weights_are_refused() {
        let Some(device) = device() else { return };
        let kernel = MoeCombine::new(&device).expect("the combine compiles");
        let of = |len: usize| device.zeroed::<f32>(len).expect("the buffer allocates");
        let refused = |routed: usize, shared: usize| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut weights = RoutingWeights {
                    routed: of(TOKENS * CONFIG.top_k),
                    shared: of(TOKENS * CONFIG.n_shared),
                };
                let (mut routed, mut shared) = (of(routed), of(shared));
                let mut batch = device.batch().expect("a command buffer opens");
                kernel
                    .encode(&mut batch, TOKENS, &mut weights, &mut routed, &mut shared)
                    .expect("the combine encodes");
            }))
            .err()
            .and_then(|err| err.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| panic!("a call of {routed} and {shared} was accepted"))
        };

        let whole = (
            TOKENS * CONFIG.top_k * HIDDEN,
            TOKENS * CONFIG.n_shared * HIDDEN,
        );
        assert!(
            refused(whole.0 + 1, whole.1).contains("routed values are not"),
            "the routed rows"
        );
        assert!(
            refused(whole.0, whole.1 - HIDDEN).contains("shared values are not"),
            "the shared rows"
        );
    }
}
