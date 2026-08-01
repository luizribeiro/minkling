//! The GPU this process runs against, and everything that can go wrong there.

use std::cell::{Cell, RefCell};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandQueue, MTLCounterSamplingPoint, MTLCreateSystemDefaultDevice, MTLDevice,
};

use crate::sampling::Timestamps;

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

    #[error("this Metal device does not sample timestamps at the boundaries of a compute pass")]
    NoDispatchTiming,

    #[error("the Metal device would not open a counter sample buffer: {0}")]
    NoCounterSampleBuffer(String),

    #[error("one command buffer cannot have more than {most} of its dispatches timed")]
    TooManySampledDispatches { most: usize },
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
    allocations: Cell<u64>,
    /// The GPU's clock, while somebody is asking each dispatch what it cost on
    /// it. `None` — nobody is — is the default, because sampling is not free:
    /// see [`crate::sampling`].
    timestamps: RefCell<Option<Timestamps>>,
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
            allocations: Cell::new(0),
            timestamps: RefCell::new(None),
        })
    }

    /// Whether this device timestamps the two ends of a compute pass, which is
    /// what [`Device::time_each_dispatch`] rests on.
    pub fn times_a_pass(&self) -> bool {
        self.raw
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
    }

    /// Whether this device timestamps between two dispatches of *one* compute
    /// pass, which no Apple silicon GPU does and which would make timing a
    /// dispatch cost nothing but the sample.
    ///
    /// Asked rather than assumed, and asked separately from
    /// [`Device::times_a_pass`], because the difference between the two is the
    /// whole overhead of what this backend does instead — see
    /// [`crate::sampling`].
    pub fn times_a_dispatch_inside_a_pass(&self) -> bool {
        self.raw
            .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary)
    }

    /// Ask the device what each dispatch costs it, or stop asking.
    ///
    /// Off by default and switched on for a measurement rather than left on:
    /// a timed dispatch is a compute pass of its own, and what that costs is
    /// reported beside what it measures rather than absorbed into it.
    pub fn time_each_dispatch(&self, sampling: bool) -> Result<(), MetalError> {
        let timestamps = match sampling {
            false => None,
            true => Some(Timestamps::of(self)?),
        };
        *self.timestamps.borrow_mut() = timestamps;
        Ok(())
    }

    /// Whether [`Device::time_each_dispatch`] is on.
    pub fn timing_each_dispatch(&self) -> bool {
        self.timestamps.borrow().is_some()
    }

    pub(crate) fn timestamps(&self) -> std::cell::Ref<'_, Option<Timestamps>> {
        self.timestamps.borrow()
    }

    /// How many buffers this device has been asked to allocate.
    ///
    /// The third number the granularity question needs, and the one a dispatch
    /// count hides. A step's dispatches are the model's shape and its
    /// submissions are how they were scheduled; what it *allocates* is neither
    /// — it is how much of what a dispatch reads was already on the device, and
    /// it moves under changes that leave both other numbers alone. A layer that
    /// hands the same rows over twice and one that hands them over once
    /// dispatch identically.
    ///
    /// Counted here because this is the one place it can be: [`Device::zeroed`]
    /// is what every buffer in the crate is made through, wrapped pages
    /// excepted, and those allocate nothing by construction.
    pub fn allocations(&self) -> u64 {
        self.allocations.get()
    }

    pub(crate) fn allocated(&self) {
        self.allocations.set(self.allocations.get() + 1);
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

    /// Every buffer is counted once, however it was asked for — and an
    /// allocation the device refuses is not one of them, which is what keeps the
    /// count a count of memory rather than of calls.
    #[test]
    fn a_device_counts_the_buffers_it_allocated() {
        let Some(device) = device() else { return };
        let before = device.allocations();

        let zeroed = device.zeroed::<f32>(16).expect("the buffer allocates");
        let filled = device.buffer(&[1.0f32, 2.0]).expect("the buffer allocates");
        assert_eq!(device.allocations() - before, 2, "one each");

        assert!(device.zeroed::<f32>(0).is_err(), "a buffer of nothing");
        assert_eq!(
            device.allocations() - before,
            2,
            "an allocation the device refused was counted as one it made"
        );

        assert_eq!(zeroed.len(), 16);
        assert_eq!(filled.to_vec(), [1.0, 2.0]);
    }

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
