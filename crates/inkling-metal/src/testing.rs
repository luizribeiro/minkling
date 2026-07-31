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
