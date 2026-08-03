//! What the tests here need before they can assert anything.

use inkling_core::moe::MoeConfig;
use inkling_core::profile;

use crate::device::{Device, MetalError};
use crate::kernel::Batch;

/// The checkpoint's own routing shape, so that a kernel is exercised over the
/// row it will actually read: 256 routed experts, two shared, six per token.
///
/// Here rather than in one of the modules that route because two of them do —
/// the selection and the weighting are separate kernels over one row of logits,
/// and a case that disagreed with another about the shape would be measuring a
/// layer neither of them runs.
pub const ROUTING: MoeConfig = MoeConfig {
    n_routed: 256,
    n_shared: 2,
    top_k: 6,
    route_scale: 8.0,
};

/// The trained `global_scale` of the layer the activation capture covers, which
/// is not 1 — so a router that dropped it answers 142 times hot rather than
/// identically, which is the largest single error any of the four ways of
/// misreading this gate produces.
pub const GLOBAL_SCALE: f32 = 0.007_042_432_7;

/// `[tokens, n_routed + n_shared]` gate logits, spread over both signs and over
/// the range where `sigmoid` is not saturated — so that the correction bias
/// decides part of the ranking, the logits decide the rest, and every one of the
/// eight weights is a number the softmax has something to do with.
pub fn gate_logits(tokens: usize, seed: usize) -> Vec<f32> {
    let width = ROUTING.n_routed + ROUTING.n_shared;
    (0..tokens * width)
        .map(|i| ((i * 37 + seed) % 401) as f32 / 40.0 - 5.0)
        .collect()
}

/// A correction bias that is not all one value, so that a router which dropped
/// it would rank differently.
pub fn correction_bias() -> Vec<f32> {
    (0..ROUTING.n_routed)
        .map(|i| ((i * 53) % 97) as f32 / 400.0 - 0.12)
        .collect()
}

/// `source` with `what` written as `with`, refusing a pattern the source does
/// not hold.
///
/// **A replacement that matched nothing is the failure mode every mutation
/// measurement shares**: it compiles, it runs, and it reports the shipped kernel
/// under another name. Comparing the whole mutated string against the whole
/// source catches that only where a mutation makes one replacement, and the
/// limiter tables in `attention` and `matmul` both make several.
///
/// Here rather than beside either of them because both do it, and a second
/// spelling is one that could drift into an unchecked `str::replace`.
pub fn instead_of(source: &str, what: &str, with: &str) -> String {
    assert!(source.contains(what), "the source holds `{what}`");
    source.replace(what, with)
}

/// The default device, or `None` with a reported skip.
///
/// These tests need no checkpoint and no fixture, so they are ordinary tests
/// rather than gated ones. The one thing they do need is hardware, and a
/// machine without a Metal device should report and pass — the way the
/// `INKLINGRS_CHECKPOINT` tests do — rather than fail on something the code
/// under test has no say in. Any other error is a real failure and panics.
pub fn device() -> Option<Device> {
    match Device::open() {
        Ok(device) => Some(device),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: this machine has no Metal device");
            None
        }
        Err(err) => panic!("the default device opens: {err}"),
    }
}

/// How far an answer lands from the exact one, as a fraction of the exact
/// tensor's peak — [`inkling_core::fixture::deviation`]'s measure, against an
/// f64 accumulation of the same products rather than against the other f32
/// answer.
///
/// **This is what turns a disagreement into either float noise or a bug.** Both
/// matmuls here decode exactly and neither rounds anywhere else, so a dispatch
/// and the CPU differ only in the order they sum — and which of the two is
/// drifting is a question only a third accumulation can answer.
pub fn drift(got: &[f32], exact: &[f64]) -> f64 {
    assert_eq!(got.len(), exact.len(), "length");
    let scale = exact.iter().fold(0.0f64, |peak, w| peak.max(w.abs()));
    got.iter().zip(exact).fold(0.0f64, |worst, (got, exact)| {
        worst.max((f64::from(*got) - exact).abs())
    }) / scale
}

/// `out = alpha * x + y`, the smallest kernel that still has to get every part
/// of the plumbing right: a scalar and a count read through their own bindings,
/// two arrays in, one array out, and a bounds check for the threads the last
/// threadgroup runs past the end.
pub const SAXPY: &str = r#"
#include <metal_stdlib>

kernel void saxpy(
    constant float &alpha [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const float *y [[buffer(3)]],
    device float *out [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) {
        return;
    }
    out[i] = alpha * x[i] + y[i];
}
"#;

pub const SAXPY_ENTRY: &str = "saxpy";

/// What the dispatches `encode` encoded declared they move, in bytes.
///
/// **A declared figure is the one number in the profile nothing else checks.**
/// The device's clock is the device's, the call counts are the model's shape,
/// and both are wrong loudly; a byte count that dropped a factor would move a
/// whole column of the bandwidth table and read as a finding. So each kernel
/// has a case asserting its own against what its source reads, and this is what
/// those cases go through.
///
/// Sampling has to be on for the figure to reach the profile at all — it is
/// charged per timed pass — so this switches it on and off around the batch.
pub fn moved(device: &Device, encode: impl FnOnce(&mut Batch<'_>)) -> u64 {
    device
        .time_each_dispatch(true)
        .expect("the device times a dispatch");
    profile::take();

    let mut batch = device.batch().expect("a command buffer opens");
    encode(&mut batch);
    batch.wait().expect("the batch completes");

    device.time_each_dispatch(false).expect("sampling stops");
    profile::take()
        .kernels()
        .iter()
        .map(|(_, dispatches)| dispatches.bytes)
        .sum()
}

/// What the device's own clock makes of one dispatch of `encode`, over a
/// command buffer holding `calls` of them.
///
/// **A submission costs 225 microseconds and the dispatches a sweep asks about
/// are tens**, so a figure taken one submission at a time is a figure about the
/// round trip. Repeating the dispatch inside one command buffer is what leaves
/// the device's clock measuring the dispatch, and it is what both sweeps here
/// — the packed matmul's width and the dense matmul's reduction run — do to get
/// a number their tables can be read across.
///
/// The profile is emptied after the encoding rather than before it, so that
/// what the clock is divided by is these dispatches and nothing a caller did to
/// set them up.
pub fn device_time(
    device: &Device,
    calls: usize,
    mut encode: impl FnMut(&mut Batch<'_>),
) -> std::time::Duration {
    let mut batch = device.batch().expect("a command buffer opens");
    for _ in 0..calls {
        encode(&mut batch);
    }
    profile::take();
    batch.wait().expect("the batch completes");
    profile::take().gpu() / calls as u32
}

/// Work the device until it has been busy this long, so that the pass which
/// counts opens on a GPU already at its clock rather than climbing to it.
///
/// **A clock that ramps over a sweep manufactures a monotone row**, and this is
/// not hypothetical: external work reproducing the threadgroup-memory
/// experiment in `matmul.rs` reported a 1.7× win that vanished entirely once
/// thirty warm-up dispatches were added — larger than the effect either sweep in
/// this crate reports.
///
/// A budget rather than a count of dispatches, because the two sweeps' arms
/// differ by two decades in what one costs and what a clock answers to is
/// elapsed load rather than dispatches. Two seconds is past where this part's
/// boost window closes, so what a sweep reports after it is the sustained clock
/// — which is the one a prefill of any length runs at.
pub fn warmed(mut work: impl FnMut()) {
    const BUSY: std::time::Duration = std::time::Duration::from_secs(2);
    let opened = std::time::Instant::now();
    while opened.elapsed() < BUSY {
        work();
    }
}

/// Every arm of a sweep, measured up the list and then down it.
///
/// **One order cannot separate a turn the kernel has from one the clock drew.**
/// A ramp flatters whichever arm ran last, and the arms of an occupancy sweep
/// are listed by what they declare — so a ramp and a real turn read the same way
/// in a single pass. Run both ways they do not: a turn sits at the same arm
/// whichever end the sweep opened from, and a ramp follows the order instead.
///
/// The second vector is returned in the arms' own order rather than the order it
/// was taken in, so that the two read down the page against each other.
pub fn both_ways<T: Copy, R>(arms: &[T], mut measure: impl FnMut(T) -> R) -> (Vec<R>, Vec<R>) {
    let up: Vec<R> = arms.iter().map(|&arm| measure(arm)).collect();
    let mut down: Vec<R> = arms.iter().rev().map(|&arm| measure(arm)).collect();
    down.reverse();
    (up, down)
}

/// A dispatch of the caller's grid with nothing in it, which is the floor under
/// every figure a sweep here reports.
///
/// **A kernel measured against nothing cannot say how much of what it costs is
/// its own.** The saxpy kernel over a count of zero returns on its first
/// instruction in every thread, so a dispatch of it over the grid a real kernel
/// runs is that kernel's launch and none of its work — which is what separates
/// a row that is slow from a row that is small.
///
/// The grid is the caller's and has to be the one the real dispatch uses, down
/// to the threadgroup width: what a launch costs grows with the threads in it,
/// so a floor taken over a different grid is a floor under a different kernel.
pub struct EmptyDispatch {
    kernel: crate::kernel::Kernel,
    alpha: crate::Buffer<f32>,
    count: crate::Buffer<u32>,
    x: crate::Buffer<f32>,
    y: crate::Buffer<f32>,
    out: crate::Buffer<f32>,
}

impl EmptyDispatch {
    pub fn new(device: &Device) -> Self {
        Self {
            kernel: device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles"),
            alpha: device.buffer(&[0.0f32]).expect("the buffer allocates"),
            count: device.buffer(&[0u32]).expect("the buffer allocates"),
            x: device.zeroed(1).expect("the buffer allocates"),
            y: device.zeroed(1).expect("the buffer allocates"),
            out: device.zeroed(1).expect("the buffer allocates"),
        }
    }

    /// What the device's own clock makes of one such dispatch, over a command
    /// buffer holding `calls` of them — the same arrangement, and for the same
    /// reason, as [`device_time`].
    pub fn cost(
        &mut self,
        device: &Device,
        calls: usize,
        grid: crate::kernel::Grid,
    ) -> std::time::Duration {
        let Self {
            kernel,
            alpha,
            count,
            x,
            y,
            out,
        } = self;
        device_time(device, calls, |batch| {
            batch
                .add(
                    kernel,
                    &[alpha.arg(), count.arg(), x.arg(), y.arg(), out.arg()],
                    grid,
                    0,
                )
                .expect("the empty dispatch encodes");
        })
    }
}

/// What one saxpy over `len` elements moves: `x` and `y` read, `out` written.
///
/// Here rather than at each case because every dispatch has to declare what it
/// moves — see [`Batch::add`](crate::kernel::Batch::add) — and a test kernel's
/// answer is the same wherever it is asked.
pub fn saxpy_moves(len: usize) -> usize {
    3 * size_of::<f32>() * len
}

#[cfg(test)]
mod tests {
    /// **The second pass runs backwards and reports forwards**, which is the
    /// whole of what [`super::both_ways`] promises and the one thing a stray
    /// edit could undo silently: the sweeps that read it are `#[ignore]`d, so a
    /// dropped `reverse` would surface as a table nobody could line up rather
    /// than as a failure.
    #[test]
    fn a_sweep_taken_both_ways_reports_the_second_pass_in_the_arms_own_order() {
        let mut order = Vec::new();
        let (up, down) = super::both_ways(&[10, 20, 30], |arm| {
            order.push(arm);
            arm
        });

        assert_eq!(up, [10, 20, 30]);
        assert_eq!(down, [10, 20, 30], "the second pass is reported forwards");
        assert_eq!(
            order,
            [10, 20, 30, 30, 20, 10],
            "the second pass is taken backwards"
        );
    }

    /// An arm list of one is the degenerate case both passes have to agree on,
    /// and an empty one is the case a `reverse` of nothing must not panic over.
    #[test]
    fn a_sweep_of_one_arm_and_a_sweep_of_none_are_both_taken_twice_and_never() {
        let mut calls = 0;
        let (up, down) = super::both_ways(&[7], |arm| {
            calls += 1;
            arm
        });
        assert_eq!((up, down, calls), (vec![7], vec![7], 2));

        let mut never = 0;
        let (up, down) = super::both_ways(&[] as &[usize], |arm| {
            never += 1;
            arm
        });
        assert_eq!((up, down, never), (Vec::new(), Vec::new(), 0));
    }
}
