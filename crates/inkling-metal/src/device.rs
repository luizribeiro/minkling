//! The GPU this process runs against, and everything that can go wrong there.

use std::cell::Cell;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("this machine has no Metal device")]
    NoDevice,

    #[error("the Metal device would not open a command queue")]
    NoCommandQueue,

    #[error("{len} elements of {size} bytes is not a size that can be addressed")]
    Overflow { len: usize, size: usize },

    #[error("the Metal device would not allocate a buffer of {bytes} bytes")]
    Allocation { bytes: usize },

    #[error("bytes {offset} into their page are not aligned for an element of {size}")]
    Misaligned { offset: usize, size: usize },

    #[error("kernel source does not compile:\n{0}")]
    Compile(String),

    #[error("the compiled source has no kernel named {0}")]
    NoSuchKernel(String),

    #[error("{entry} does not make a compute pipeline: {diagnostic}")]
    Pipeline { entry: String, diagnostic: String },

    #[error("{entry} asks for {asked} threads a group, more than the {most} this pipeline allows")]
    ThreadgroupTooLarge {
        entry: String,
        asked: usize,
        most: usize,
    },

    #[error("{entry} is given {asked} buffers, more than the {most} a function can bind")]
    TooManyArguments {
        entry: String,
        asked: usize,
        most: usize,
    },

    #[error("the Metal device would not open a command buffer")]
    NoCommandBuffer,

    #[error("the command buffer would not open a compute encoder")]
    NoCommandEncoder,

    #[error("{entry} did not complete: {diagnostic}")]
    Execution { entry: String, diagnostic: String },
}

/// The default Metal device and one command queue onto it.
///
/// The queue is opened once and held, rather than per dispatch: a queue is the
/// serial ordering the GPU executes command buffers in, so a fresh one per call
/// would be both an allocation and a claim that the calls are unordered.
#[derive(Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    dispatches: Cell<u64>,
    submissions: Cell<u64>,
}

impl Device {
    /// The system default device, which on an Apple silicon machine is the only
    /// one there is.
    pub fn open() -> Result<Self, MetalError> {
        let raw = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = raw.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;
        Ok(Self {
            raw,
            queue,
            dispatches: Cell::new(0),
            submissions: Cell::new(0),
        })
    }

    /// How many kernels this device has been asked to run.
    ///
    /// Counted because the granularity question cannot be settled without it: a
    /// decode step's dispatches are a number the engine's shape decides — five
    /// projections a layer, six an MoE layer, one head — and what they cost is
    /// how many command buffers they were submitted in, which is a different
    /// number. Both are measurements rather than arithmetic on paper, and the
    /// gated tests print the pair.
    pub fn dispatches(&self) -> u64 {
        self.dispatches.get()
    }

    /// How many command buffers those dispatches were submitted and waited for
    /// in, which is what the round trips cost.
    pub fn submissions(&self) -> u64 {
        self.submissions.get()
    }

    pub(crate) fn counted(&self, dispatches: usize) {
        self.dispatches
            .set(self.dispatches.get() + dispatches as u64);
        self.submissions.set(self.submissions.get() + 1);
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

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
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
