//! The GPU this process runs against, and everything that can go wrong there.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("this machine has no Metal device")]
    NoDevice,

    #[error("{len} elements of {size} bytes is not a size that can be addressed")]
    Overflow { len: usize, size: usize },

    #[error("the Metal device would not allocate a buffer of {bytes} bytes")]
    Allocation { bytes: usize },

    #[error("kernel source does not compile:\n{0}")]
    Compile(String),

    #[error("the compiled source has no kernel named {0}")]
    NoSuchKernel(String),

    #[error("{entry} does not make a compute pipeline: {diagnostic}")]
    Pipeline { entry: String, diagnostic: String },
}

/// The default Metal device.
#[derive(Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Device {
    /// The system default device, which on an Apple silicon machine is the only
    /// one there is.
    pub fn open() -> Result<Self, MetalError> {
        let raw = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        Ok(Self { raw })
    }

    /// The largest single allocation the device will make. A hard ceiling and
    /// not a share of anything: a weight above it has to be split however much
    /// memory is free.
    pub fn max_buffer_bytes(&self) -> usize {
        self.raw.maxBufferLength()
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::device;

    /// What one allocation can hold is a design input for M2 and not trivia:
    /// `lm_head` decoded is 3.3 GB and the routed experts are 25 GB. A device
    /// that opens but reports nothing it can hold is not one this engine can
    /// run on, and every other test here would fail obscurely rather than say
    /// so.
    #[test]
    fn the_device_that_opens_can_hold_a_weight() {
        let Some(device) = device() else { return };

        let bytes = device.max_buffer_bytes();
        assert!(bytes >= 1 << 30, "{bytes} bytes");
    }
}
