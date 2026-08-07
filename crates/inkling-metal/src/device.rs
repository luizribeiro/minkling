//! The GPU this process runs against, and everything that can go wrong there.

use std::cell::{Cell, RefCell};
use std::time::Duration;

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

    #[error("{entry} runs {width} threads in lockstep where its layout is cut for {wanted}")]
    UnexpectedSimdWidth {
        entry: String,
        width: usize,
        wanted: usize,
    },

    #[error("the Metal device would not open a command buffer")]
    NoCommandBuffer,

    #[error("the command buffer would not open a compute encoder")]
    NoCommandEncoder,

    #[error("the Metal device would not open an indirect command buffer")]
    NoIndirectCommandBuffer,

    #[error("{entry} did not complete: {diagnostic}")]
    Execution { entry: String, diagnostic: String },

    #[error("this Metal device does not sample timestamps at the boundaries of a compute pass")]
    NoDispatchTiming,

    #[error("the Metal device would not open a counter sample buffer: {0}")]
    NoCounterSampleBuffer(String),

    #[error("one command buffer cannot have more than {most} of its dispatches timed")]
    TooManySampledDispatches { most: usize },
}

/// What one command buffer's round trip was made of.
///
/// **`submit and wait` is one row and four things happen inside it.** The wall
/// time is this process's, and every other figure here is the driver's own
/// clock: a committed buffer is scheduled, sits in the queue, executes, and
/// then somebody has to notice it finished. Only the third is work, and a
/// backend deciding how many command buffers a step should be needs the other
/// three separated from it rather than summed into a round trip nobody can
/// divide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundTrip {
    /// How many dispatches were in it.
    pub dispatches: usize,
    /// What this process blocked for.
    ///
    /// **Not the buffer's whole life**, and the difference is the point: a
    /// buffer committed and waited for in the same breath is blocked on for all
    /// of it, and one committed while there was still encoding to do is blocked
    /// on for whatever was left when the caller ran out of other work. So a row
    /// whose `executed` is larger than this is a submission that ran while this
    /// process was busy, which is a thing a table should be able to show.
    pub waited: Duration,
    /// What the driver spent turning a committed buffer into work the GPU could
    /// start — `kernelEndTime - kernelStartTime`, which grows with the
    /// dispatches in it.
    pub scheduled: Duration,
    /// How long it then sat before the GPU picked it up.
    pub queued: Duration,
    /// What the GPU was executing it for, which is the only part of a round trip
    /// that is the model's arithmetic.
    pub executed: Duration,
    /// How long the GPU ran nothing of this queue's between the buffer before
    /// this one finishing and this one starting.
    ///
    /// **The one column here that is about the gaps rather than about the
    /// buffers**, and the only one that says whether a wall a duty cycle cannot
    /// see is the device standing still. `scheduled` and `queued` are each
    /// buffer's own life and they *overlap* between buffers in flight — a
    /// caller that commits three and waits for the third has three queue
    /// intervals covering one stretch of clock, so their sum is not a share of
    /// anything. These do not overlap: the queue executes one buffer at a time,
    /// so the gaps between them and the executions between them tile the round.
    ///
    /// Nothing for the first buffer after recording was switched on, which has
    /// no buffer before it to be idle since, and nothing after one whose own
    /// clock the device never reported — see [`Device::idle_before`], where the
    /// cases are.
    pub idle: Duration,
}

impl RoundTrip {
    /// The part of the wait that was none of the three: the commit reaching the
    /// driver, and this thread being woken once the buffer completed.
    ///
    /// Nothing for a submission that overlapped its caller's own work, since
    /// the three then account for more than the wait rather than less. Nothing
    /// too where the two clocks disagree in their last microsecond, which they
    /// may: the wait is `Instant`'s and the rest is the driver's, and neither a
    /// buffer that ran early nor a rounding is a reason to build a negative
    /// interval.
    pub fn unattributed(&self) -> Duration {
        self.waited
            .saturating_sub(self.scheduled + self.queued + self.executed)
    }
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
    barriers: Cell<u64>,
    submissions: Cell<u64>,
    allocations: Cell<u64>,
    allocated_bytes: Cell<u64>,
    /// The GPU's clock, while somebody is asking each dispatch what it cost on
    /// it. `None` — nobody is — is the default, because sampling is not free:
    /// see [`crate::sampling`].
    timestamps: RefCell<Option<Timestamps>>,
    /// Each command buffer's round trip, while somebody is asking what the waits
    /// were made of. `None` — nobody is — is the default, because a decode loop
    /// that nobody is measuring would otherwise grow a record per submission for
    /// as long as it runs.
    round_trips: RefCell<Option<Vec<RoundTrip>>>,
    /// When the last recorded command buffer stopped executing, on the GPU's
    /// own clock, which is what [`RoundTrip::idle`] is measured from.
    ///
    /// **An `Option` rather than a zero, because zero is a reading this device
    /// gives.** A command buffer the watchdog killed completes with no
    /// timestamps at all — see [`crate::Submitted::wait`], which is where that
    /// error arrives — so a sentinel spelled `0.0` would be indistinguishable
    /// from the buffer most likely to have a long gap after it.
    finished: Cell<Option<f64>>,
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
            barriers: Cell::new(0),
            submissions: Cell::new(0),
            allocations: Cell::new(0),
            allocated_bytes: Cell::new(0),
            timestamps: RefCell::new(None),
            round_trips: RefCell::new(None),
            finished: Cell::new(None),
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

    /// Keep a [`RoundTrip`] for each command buffer from here, or stop and
    /// discard what was kept.
    ///
    /// **The figures are read off a buffer that has already completed, so what
    /// this switches on changes nothing it measures.** What it is opt-in for is
    /// the accumulation: a decode loop nobody is measuring would otherwise grow
    /// a record per submission for as long as it runs, and a device that opens
    /// once and serves a whole process runs for a long time.
    pub fn record_round_trips(&self, recording: bool) {
        *self.round_trips.borrow_mut() = recording.then(Vec::new);
        self.finished.set(None);
    }

    /// Every round trip since [`Device::record_round_trips`] was switched on,
    /// in the order they were waited for, and the record cleared.
    ///
    /// Cleared rather than read for [`inkling_core::profile::take`]'s reason: a
    /// caller measuring one step of a loop wants that step rather than the run
    /// so far.
    pub fn round_trips(&self) -> Vec<RoundTrip> {
        // Cleared with the records, so that the first buffer of the next stretch
        // is not called idle since one the caller has already been handed.
        self.finished.set(None);
        match self.round_trips.borrow_mut().as_mut() {
            None => Vec::new(),
            Some(taken) => std::mem::take(taken),
        }
    }

    /// How long the GPU ran nothing of this queue's before a buffer that started
    /// at `started`, and that buffer's end remembered for the next one.
    ///
    /// **Nothing, in each of the three cases where the gap is not a gap this
    /// can see** — and each of them is a gap this must not invent rather than
    /// one it may round to zero. No buffer recorded since the record was
    /// cleared, so the clock it would measure from belongs to the caller's
    /// previous stretch. A buffer whose own clock the device never reported,
    /// which leaves the next one nothing to measure from. And two readings the
    /// wrong way round, which the driver's clock may give in its last
    /// microsecond.
    pub(crate) fn idle_before(&self, started: f64, ended: f64) -> Duration {
        match self.finished.replace((ended > 0.0).then_some(ended)) {
            Some(since) if started > since => Duration::from_secs_f64(started - since),
            _ => Duration::ZERO,
        }
    }

    /// The record taken and kept, where somebody asked for one.
    ///
    /// A closure rather than a value so that a caller who would have to read the
    /// driver's timestamps to build one does not read them when nobody is
    /// recording, which is every run but a measurement.
    pub(crate) fn round_tripped(&self, trip: impl FnOnce() -> RoundTrip) {
        if let Some(taken) = self.round_trips.borrow_mut().as_mut() {
            taken.push(trip());
        }
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

    /// How many bytes those allocations came to.
    ///
    /// A count of buffers says how a call was scheduled; what it does not say is
    /// how much memory the call is holding, and the two move apart by orders of
    /// magnitude under nothing but a row count — a layer allocates the same
    /// buffers for one token as for seven hundred. This is the number a caller
    /// that has to *bound* what it holds needs, and the difference between two
    /// readings of it is what everything allocated in between came to.
    ///
    /// **A command buffer retains what is bound into it**, so nothing allocated
    /// while one is being encoded can be freed before it completes — which is
    /// what makes a difference of two readings a measurement of what is still
    /// held rather than of what passed through. See
    /// [`ModelLayers::carries`](crate::ModelLayers), which is the caller this
    /// exists for.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.get()
    }

    pub(crate) fn allocated(&self, bytes: usize) {
        self.allocations.set(self.allocations.get() + 1);
        self.allocated_bytes
            .set(self.allocated_bytes.get() + bytes as u64);
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

    /// How many barriers those dispatches needed, which is the whole of what a
    /// concurrent pass buys back the ordering with.
    ///
    /// Counted for the reason the dispatches are, and for one more: it is what
    /// says the division [`Groups`](crate::ordering::Groups) reports and the one
    /// the engine encoded are the same division. Both come out of the same
    /// [`Open`](crate::ordering), so they cannot disagree — and a count nothing
    /// checked is a claim rather than a fact, which is not what this milestone
    /// can afford.
    pub fn barriers(&self) -> u64 {
        self.barriers.get()
    }

    pub(crate) fn counted(&self, dispatches: usize, barriers: usize) {
        self.dispatches
            .set(self.dispatches.get() + dispatches as u64);
        self.barriers.set(self.barriers.get() + barriers as u64);
        self.submissions.set(self.submissions.get() + 1);
    }

    /// The largest single allocation the device will make. A hard ceiling and
    /// not a share of anything: a weight above it has to be split however much
    /// memory is free.
    pub fn max_buffer_bytes(&self) -> usize {
        self.raw.maxBufferLength()
    }

    /// The threadgroup memory one threadgroup may declare, which is what a
    /// kernel's own [`Kernel::threadgroup_memory`](crate::Kernel::threadgroup_memory)
    /// has to be read against: the two divide into how many threadgroups a core
    /// can hold at once, and that is the one occupancy figure this side can ask
    /// for rather than infer from a clock.
    pub fn most_threadgroup_bytes(&self) -> usize {
        self.raw.maxThreadgroupMemoryLength()
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
    use std::time::Duration;

    use super::RoundTrip;
    use crate::testing::device;

    /// What a round trip's parts leave over is the wait minus all three, and a
    /// record whose parts came to more than the wait leaves nothing rather than
    /// panicking on a negative interval: the wall time is `Instant`'s clock and
    /// the other three are the driver's, so the two disagree in the last
    /// microsecond by construction.
    ///
    /// **The gap before the buffer is not one of the three**, which is what the
    /// 5 ms here says: it is time before this wait began and charging it to a
    /// wait it is not inside would make the remainder negative in exactly the
    /// case the remainder exists to describe.
    #[test]
    fn a_round_trips_unattributed_time_is_what_its_three_parts_leave_over() {
        let trip = |waited: u64| RoundTrip {
            dispatches: 1,
            waited: Duration::from_micros(waited),
            scheduled: Duration::from_micros(100),
            queued: Duration::from_micros(60),
            executed: Duration::from_micros(600),
            idle: Duration::from_micros(5000),
        };
        assert_eq!(trip(1000).unattributed(), Duration::from_micros(240));
        assert_eq!(trip(700).unattributed(), Duration::ZERO);
    }

    /// The record is off by default, kept only while somebody asks for it, and
    /// cleared by the reading — so a caller measuring one step of a loop is
    /// handed that step rather than every submission since the device opened.
    #[test]
    fn a_device_records_round_trips_only_while_it_is_asked_to() {
        let Some(device) = device() else { return };
        let mut empty = crate::testing::EmptyDispatch::new(&device);
        let mut run = || empty.cost(&device, 1, crate::kernel::Grid::new(1, 1));

        run();
        assert!(device.round_trips().is_empty(), "nobody was recording");

        device.record_round_trips(true);
        run();
        run();
        let recorded = device.round_trips();
        assert_eq!(recorded.len(), 2, "one a submission");
        assert!(device.round_trips().is_empty(), "the reading clears");
        assert!(
            recorded
                .iter()
                .all(|trip| trip.dispatches == 1 && trip.waited > trip.executed),
            "{recorded:?}"
        );

        device.record_round_trips(false);
        run();
        assert!(device.round_trips().is_empty(), "recording stopped");
    }

    /// **A gap is between two buffers, so the first of a stretch has none** —
    /// and the clock it would otherwise be measured from is a buffer the caller
    /// has already been handed, which would charge the time between two
    /// measurements to the second of them.
    ///
    /// The second dispatch's gap is asserted only to exist. **It is not bounded
    /// by that buffer's own wait and must not be asserted to be**: the gap opens
    /// when the buffer before it finished, which is before this one was
    /// committed — so it covers the caller's own encoding as well as anything
    /// the driver added, and a two-dispatch test spends more clock there than
    /// inside either wait.
    #[test]
    fn the_first_buffer_of_a_recorded_stretch_is_idle_since_nothing() {
        let Some(device) = device() else { return };
        let mut empty = crate::testing::EmptyDispatch::new(&device);
        let mut run = || empty.cost(&device, 1, crate::kernel::Grid::new(1, 1));

        device.record_round_trips(true);
        run();
        run();
        let recorded = device.round_trips();
        assert_eq!(recorded[0].idle, Duration::ZERO, "{recorded:?}");
        assert!(recorded[1].idle > Duration::ZERO, "{recorded:?}");

        // And the reading clears it, so the next stretch opens the same way.
        run();
        run();
        let next = device.round_trips();
        assert_eq!(next[0].idle, Duration::ZERO, "{next:?}");
        device.record_round_trips(false);
    }

    /// **A buffer that reported no clock leaves the next one nothing to measure
    /// from**, which is the case a sentinel spelled `0.0` gets wrong in the one
    /// direction that matters: the watchdog kills a command buffer that ran too
    /// long, it completes with zero timestamps, and the buffer after it is the
    /// one most likely to have a long gap in front of it. Reported as no gap it
    /// would be an invented zero; measured from that zero it would be the whole
    /// age of the machine.
    ///
    /// Driven against the clock directly rather than through a dispatch,
    /// because a timed-out buffer is not a thing a test may ask this device for.
    #[test]
    fn a_buffer_that_reported_no_clock_is_not_a_gap_the_next_one_is_measured_from() {
        let Some(device) = device() else { return };
        device.record_round_trips(true);

        assert_eq!(device.idle_before(4.0, 5.0), Duration::ZERO, "no baseline");
        assert_eq!(
            device.idle_before(6.0, 7.0),
            Duration::from_secs_f64(1.0),
            "6.0 is a second after the 5.0 before it"
        );
        // The watchdog's buffer: it neither reports a gap nor becomes one.
        assert_eq!(device.idle_before(0.0, 0.0), Duration::ZERO, "no clock");
        assert_eq!(
            device.idle_before(400.0, 401.0),
            Duration::ZERO,
            "measured from a clock that was never read"
        );
        // And two readings the wrong way round are no gap rather than a
        // negative one, which `Duration::from_secs_f64` would panic on.
        assert_eq!(device.idle_before(400.5, 402.0), Duration::ZERO);

        device.record_round_trips(false);
    }

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

    /// The bytes rather than the buffers, which is the number a caller bounding
    /// what it holds reads — and the whole of what it adds is that the two
    /// buffers below are one count apart and eight thousand bytes apart.
    ///
    /// Both are charged what the element type makes of the length, because a
    /// `Buffer<T>`'s length is in elements and an allocation is in bytes; a
    /// counter that charged the length would put a buffer of a thousand floats
    /// at a quarter of what it costs.
    #[test]
    fn a_device_counts_the_bytes_those_buffers_came_to() {
        let Some(device) = device() else { return };
        let before = device.allocated_bytes();

        let small = device.zeroed::<f32>(16).expect("the buffer allocates");
        let large = device.zeroed::<f32>(2048).expect("the buffer allocates");
        assert_eq!(
            device.allocated_bytes() - before,
            (16 + 2048) * size_of::<f32>() as u64,
            "the bytes of both"
        );

        assert!(device.zeroed::<u8>(0).is_err(), "a buffer of nothing");
        assert_eq!(
            device.allocated_bytes() - before,
            (16 + 2048) * size_of::<f32>() as u64,
            "an allocation the device refused was charged"
        );

        assert_eq!(small.len() + large.len(), 16 + 2048);
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
