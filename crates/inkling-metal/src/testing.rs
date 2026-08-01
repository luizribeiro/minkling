//! What the tests here need before they can assert anything.

use crate::device::{Device, MetalError};

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
