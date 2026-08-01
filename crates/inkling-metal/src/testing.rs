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

/// What one saxpy over `len` elements moves: `x` and `y` read, `out` written.
///
/// Here rather than at each case because every dispatch has to declare what it
/// moves — see [`Batch::add`](crate::kernel::Batch::add) — and a test kernel's
/// answer is the same wherever it is asked.
pub fn saxpy_moves(len: usize) -> usize {
    3 * size_of::<f32>() * len
}
