//! What the device's own clock says about one dispatch rather than one
//! submission.
//!
//! A decode step is 1077 dispatches in two command buffers, and
//! `MTLCommandBuffer`'s `GPUStartTime`/`GPUEndTime` describe the pair. Which of
//! the nine kernels owns which of the 26 milliseconds is a different question,
//! and every plan this project has built without an answer to a question like
//! it has been overturned by the first measurement taken afterwards.
//!
//! # What this hardware samples, and what it does not
//!
//! Metal takes timestamps into an `MTLCounterSampleBuffer` at points the device
//! declares support for, and the two that matter here are different APIs:
//!
//! - **`AtDispatchBoundary`** is `sampleCountersInBuffer:atSampleIndex:` on an
//!   open compute encoder — a timestamp between two dispatches of one pass,
//!   which is exactly the grain wanted.
//! - **`AtStageBoundary`** is `MTLComputePassDescriptor`'s sample buffer
//!   attachment — one timestamp as a pass begins and one as it ends.
//!
//! **This machine supports only the second.** An M3 Ultra answers
//! `supportsCounterSampling:` with true for `AtStageBoundary` and false for
//! `AtDispatchBoundary`, `AtDrawBoundary`, `AtTileDispatchBoundary` and
//! `AtBlitBoundary`, and offers one counter set, `timestamp`.
//! [`Device::times_a_pass`] and [`Device::times_a_dispatch_inside_a_pass`] are
//! those questions asked rather than assumed, and
//! `the_sampling_points_this_device_offers_are_reported` is what would notice
//! hardware answering differently.
//!
//! So a dispatch is timed by being **a compute pass of its own**, and the
//! honest thing to say about that is that it is not free and not quite what the
//! engine does unsampled. What it does *not* do is put each dispatch in a
//! command buffer of its own: the passes still go into the same two submissions
//! a step already had, so the round trip M14 removed stays removed and what is
//! being timed is still a step of the shape this engine runs. What it costs is
//! a pass boundary per dispatch instead of per submission — a number the gated
//! `what_timing_each_dispatch_costs` measures rather than argues about, and the
//! reason this is off unless somebody asks for it.
//!
//! # Ticks
//!
//! A resolved sample is a GPU tick count in an epoch of the device's own.
//! `sampleTimestamps:gpuTimestamp:` reads both clocks at once, and two readings
//! a known interval apart give what a tick is worth — 1.000 ns on this machine,
//! where the two clocks turn out to be the same one, but a ratio that is
//! measured costs one sleep and stops being an assumption.

use std::ptr::NonNull;
use std::time::Duration;

use inkling_core::profile;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLComputePassDescriptor, MTLComputePassSampleBufferAttachmentDescriptor,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSet, MTLDevice, MTLStorageMode,
};

use crate::device::{Device, MetalError};

/// The counter set holding the GPU's clock, which is the only one this asks
/// for.
const TIMESTAMP: &str = "timestamp";

/// What a resolved sample holds where the device could not take one —
/// `MTLCounterErrorValue`, which the bindings do not carry.
const NO_SAMPLE: u64 = u64::MAX;

/// How many dispatches one command buffer may have timed.
///
/// **The device's own ceiling rather than a number chosen here**: a counter
/// sample buffer on this machine takes 8 to 32768 bytes, which at one 8-byte
/// timestamp a sample and two samples a pass is 2048 passes. Asking for more is
/// refused at construction with exactly that range in the message, so a machine
/// with a different ceiling says so rather than being guessed at.
///
/// A decode step's largest command buffer is the 1076 dispatches of its
/// forty-two layers, which leaves most of the room spare. A command buffer past
/// the ceiling is refused rather than sampled to here and truncated: a table
/// that quietly described the first 2048 dispatches of a longer submission
/// would be wrong in the direction nobody checks.
const SAMPLED_DISPATCHES: usize = 2048;

/// Long enough that the ratio between two clocks reading in nanoseconds is
/// exact to six figures, and short enough to pay once when sampling is switched
/// on.
const CORRELATION: Duration = Duration::from_millis(2);

/// The GPU's clock, and what one of its ticks is worth here.
///
/// Held by a [`Device`] for as long as it is sampling, because both halves are
/// answers to questions about the device rather than about a command buffer:
/// which counter set carries timestamps, and how its ticks relate to this
/// process's nanoseconds.
#[derive(Debug)]
pub(crate) struct Timestamps {
    /// One sample buffer for every command buffer this device will submit.
    ///
    /// **Made once rather than per submission**, and not only to save the
    /// allocation: a device holds a fixed pool of these and a decode loop
    /// opening one a submission exhausts it — "Cannot allocate sample buffer"
    /// after a few hundred, whatever this side has dropped. Reuse is sound
    /// because nothing here overlaps two command buffers: a batch is submitted,
    /// waited for and resolved before the next one is opened.
    buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    nanos_per_tick: f64,
}

impl Timestamps {
    /// The timestamp counter set and a sample buffer over it, if this device
    /// samples at the boundaries of a compute pass and offers one.
    pub(crate) fn of(device: &Device) -> Result<Self, MetalError> {
        if !device.times_a_pass() {
            return Err(MetalError::NoDispatchTiming);
        }
        let counters = device
            .raw()
            .counterSets()
            .into_iter()
            .flatten()
            .find(|set| set.name().to_string() == TIMESTAMP)
            .ok_or(MetalError::NoDispatchTiming)?;

        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&counters));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a sample count is a plain property of the descriptor; the
        // device is what refuses one it cannot hold, below.
        unsafe { descriptor.setSampleCount(2 * SAMPLED_DISPATCHES) };
        let buffer = device
            .raw()
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|err| MetalError::NoCounterSampleBuffer(crate::kernel::diagnostic(&err)))?;

        Ok(Self {
            nanos_per_tick: nanos_per_tick(device),
            buffer,
        })
    }
}

/// GPU ticks against this process's nanoseconds, from two readings of both
/// clocks a known interval apart.
fn nanos_per_tick(device: &Device) -> f64 {
    let read = || {
        let (mut cpu, mut gpu) = (0u64, 0u64);
        // SAFETY: both arguments are out parameters the call writes one
        // timestamp into, and both point at live locals.
        unsafe {
            device
                .raw()
                .sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu))
        };
        (cpu, gpu)
    };

    let (cpu_before, gpu_before) = read();
    std::thread::sleep(CORRELATION);
    let (cpu_after, gpu_after) = read();

    match gpu_after.saturating_sub(gpu_before) {
        0 => 1.0,
        ticks => cpu_after.saturating_sub(cpu_before) as f64 / ticks as f64,
    }
}

/// One command buffer's timestamps, and which kernel each pair of them belongs
/// to.
///
/// Opened with the command buffer and read once it has completed — a resolved
/// sample before then is whatever the counter held when the pass had not run.
#[derive(Debug)]
pub(crate) struct Sampled {
    buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    nanos_per_tick: f64,
    /// One descriptor, its indices moved between encoders rather than a fresh
    /// descriptor per dispatch: it is read where the encoder is opened and is
    /// nobody's afterwards.
    pass: Retained<MTLComputePassDescriptor>,
    attachment: Retained<MTLComputePassSampleBufferAttachmentDescriptor>,
    /// The distinct kernels this command buffer encoded, first seen first —
    /// a handful, against the thousand passes that name them.
    kernels: Vec<String>,
    /// Which of those each pass ran, in the order the passes were encoded.
    passes: Vec<usize>,
    /// What each of those said it moves, in the same order.
    moved: Vec<usize>,
}

impl Sampled {
    /// One command buffer's share of the device's sample buffer.
    pub(crate) fn open(timestamps: &Timestamps) -> Self {
        let pass = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: attachment 0 is the one every compute pass has.
        let attachment = unsafe { pass.sampleBufferAttachments().objectAtIndexedSubscript(0) };
        attachment.setSampleBuffer(Some(&timestamps.buffer));

        Self {
            buffer: timestamps.buffer.clone(),
            nanos_per_tick: timestamps.nanos_per_tick,
            pass,
            attachment,
            kernels: Vec::new(),
            passes: Vec::new(),
            moved: Vec::new(),
        }
    }

    /// The descriptor for the next pass, timestamped at the two samples that
    /// pass will own.
    ///
    /// Nothing is recorded here, because a descriptor is not a pass: an encoder
    /// the command buffer refuses to open would leave a sample index claimed by
    /// a pass that never ran, and every one after it reading a neighbour's
    /// timestamps. [`Sampled::ran`] is what says the pass exists.
    pub(crate) fn pass(&self) -> Result<&MTLComputePassDescriptor, MetalError> {
        if self.passes.len() == SAMPLED_DISPATCHES {
            return Err(MetalError::TooManySampledDispatches {
                most: SAMPLED_DISPATCHES,
            });
        }
        let sample = 2 * self.passes.len();
        // SAFETY: both indices are inside the sample count the buffer was made
        // with, by the check above.
        unsafe {
            self.attachment.setStartOfEncoderSampleIndex(sample);
            self.attachment.setEndOfEncoderSampleIndex(sample + 1);
        }
        Ok(&self.pass)
    }

    /// The pass [`Sampled::pass`] described opened, and runs `entry`.
    pub(crate) fn ran(&mut self, entry: &str) {
        let kernel = match self.kernels.iter().position(|name| name == entry) {
            Some(kernel) => kernel,
            None => {
                self.kernels.push(entry.to_owned());
                self.kernels.len() - 1
            }
        };
        self.passes.push(kernel);
    }

    /// What the pass that just encoded moves between memory and the GPU, from
    /// the caller that is the only one able to say — see
    /// [`Batch::add`](crate::kernel::Batch::add).
    ///
    /// Separate from [`Sampled::ran`] because it happens after the dispatch is
    /// encoded rather than before the pass is opened, and one pass has exactly
    /// one of each.
    pub(crate) fn moved(&mut self, bytes: usize) {
        self.moved.push(bytes);
    }

    /// Charge what the device reported to the kernels that ran, once the
    /// command buffer has completed.
    ///
    /// A pass the device could not sample is charged nothing rather than
    /// guessed at, which leaves it in the gap between this and the command
    /// buffer's own clock — a gap the table prints, so a device dropping
    /// samples shows up as accounting that stops adding up rather than as rows
    /// that quietly shrink.
    pub(crate) fn charge(&self) {
        if self.passes.is_empty() {
            return;
        }
        // SAFETY: the range is the passes encoded, each of which took two
        // samples inside the buffer's own count.
        let resolved = unsafe {
            self.buffer
                .resolveCounterRange(NSRange::new(0, 2 * self.passes.len()))
        };
        let Some(resolved) = resolved else { return };
        let bytes = resolved.to_vec();
        let stamps: &[MTLCounterResultTimestamp] = as_timestamps(&bytes);

        let mut totals = vec![Charge::default(); self.kernels.len()];
        for (pass, kernel) in self.passes.iter().enumerate() {
            let Some([started, ended]) = stamps.get(2 * pass..2 * pass + 2) else {
                continue;
            };
            let (started, ended) = (started.timestamp, ended.timestamp);
            if started == NO_SAMPLE || ended == NO_SAMPLE {
                continue;
            }
            let ticks = ended.saturating_sub(started) as f64;
            let total = &mut totals[*kernel];
            total.calls += 1;
            total.elapsed += Duration::from_nanos((ticks * self.nanos_per_tick) as u64);
            total.bytes += self.moved.get(pass).copied().unwrap_or_default() as u64;
        }

        for (kernel, total) in self.kernels.iter().zip(totals) {
            profile::dispatched(kernel, total.calls, total.elapsed, total.bytes);
        }
    }
}

/// One kernel's share of one command buffer, on the way to the profile.
#[derive(Debug, Default, Clone, Copy)]
struct Charge {
    calls: u64,
    elapsed: Duration,
    bytes: u64,
}

/// The resolved bytes as the timestamps they are.
///
/// `resolveCounterRange:` answers an `NSData` of `MTLCounterResultTimestamp`,
/// which is one `uint64_t`. Reading it as a slice needs the length to be whole
/// samples and the pointer to be aligned for them, and a short or misaligned
/// answer is a device that did not do what it said rather than something to
/// reinterpret.
fn as_timestamps(bytes: &[u8]) -> &[MTLCounterResultTimestamp] {
    let size = size_of::<MTLCounterResultTimestamp>();
    assert_eq!(bytes.len() % size, 0, "{} bytes of samples", bytes.len());
    assert!(
        bytes
            .as_ptr()
            .cast::<MTLCounterResultTimestamp>()
            .is_aligned()
    );
    // SAFETY: the length is whole samples and the pointer is aligned for them
    // by the checks above, `MTLCounterResultTimestamp` is a `#[repr(C)]` struct
    // of one `u64` with no invalid bit pattern, and the borrow is the slice's.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;
    use crate::testing::device;
    use objc2_metal::MTLCounterSamplingPoint;

    /// What this hardware will sample, asked rather than assumed.
    ///
    /// **The distinction this exists for is `AtDispatchBoundary` against
    /// `AtStageBoundary`**, and on Apple silicon only the second is offered —
    /// which is why a timed dispatch is a compute pass of its own here. A
    /// machine that answered otherwise would make a cheaper arrangement
    /// possible, and this is what would say so.
    #[test]
    fn the_sampling_points_this_device_offers_are_reported() {
        let Some(device) = device() else { return };

        let points = [
            ("AtStageBoundary", MTLCounterSamplingPoint::AtStageBoundary),
            ("AtDrawBoundary", MTLCounterSamplingPoint::AtDrawBoundary),
            (
                "AtDispatchBoundary",
                MTLCounterSamplingPoint::AtDispatchBoundary,
            ),
            (
                "AtTileDispatchBoundary",
                MTLCounterSamplingPoint::AtTileDispatchBoundary,
            ),
            ("AtBlitBoundary", MTLCounterSamplingPoint::AtBlitBoundary),
        ];
        for (name, point) in points {
            eprintln!("{name}: {}", device.raw().supportsCounterSampling(point));
        }
        eprintln!(
            "counter sets: {:?}",
            device
                .raw()
                .counterSets()
                .into_iter()
                .flatten()
                .map(|set| set.name().to_string())
                .collect::<Vec<String>>()
        );

        assert_eq!(
            device.times_a_pass(),
            Timestamps::of(&device).is_ok(),
            "a device that samples at a stage boundary is one this can time passes on"
        );
        assert!(
            !device.times_a_dispatch_inside_a_pass(),
            "this device times a dispatch without a pass of its own, which is cheaper than what \
             `Sampled` does — see the module docs"
        );
    }

    /// The two clocks against each other. Nothing asserts the ratio — a machine
    /// whose GPU ticks were not nanoseconds would be as correct — and what is
    /// asserted is that it is a number a duration can be built from.
    #[test]
    fn a_gpu_tick_is_worth_a_measured_number_of_nanoseconds() {
        let Some(device) = device() else { return };
        let Ok(timestamps) = Timestamps::of(&device) else {
            eprintln!("skipping: this device does not sample at a stage boundary");
            return;
        };

        let nanos = timestamps.nanos_per_tick;
        eprintln!("a gpu tick is {nanos} ns");
        assert!(nanos > 0.0 && nanos.is_finite(), "{nanos}");
    }

    /// The buffers one saxpy dispatch binds, for the cases below that are about
    /// the passes around a dispatch rather than about its arithmetic.
    ///
    /// A grid of nothing, so what a case here times is the pass and not the
    /// work — see `kernel::tests` for the same kernel asked to answer.
    struct Empty {
        alpha: crate::Buffer<f32>,
        count: crate::Buffer<u32>,
        x: crate::Buffer<f32>,
        y: crate::Buffer<f32>,
        out: crate::Buffer<f32>,
    }

    impl Empty {
        fn new(device: &Device) -> Self {
            Self {
                alpha: device.buffer(&[0.0f32]).expect("the buffer allocates"),
                count: device.buffer(&[0u32]).expect("the buffer allocates"),
                x: device.zeroed(1).expect("the buffer allocates"),
                y: device.zeroed(1).expect("the buffer allocates"),
                out: device.zeroed(1).expect("the buffer allocates"),
            }
        }

        fn args(&mut self) -> [crate::Arg<'_>; 5] {
            [
                self.alpha.arg(),
                self.count.arg(),
                self.x.arg(),
                self.y.arg(),
                self.out.arg(),
            ]
        }
    }

    /// **The one cap in here, refused rather than truncated.** A sample buffer
    /// holds 2048 passes and a command buffer past that has to be told so: rows
    /// that silently described the first 2048 dispatches of a longer submission
    /// would read as a description of the whole of it.
    #[test]
    fn a_command_buffer_past_the_sample_buffer_is_refused_rather_than_truncated() {
        let Some(device) = device() else { return };
        if device.time_each_dispatch(true).is_err() {
            eprintln!("skipping: this device does not sample at a stage boundary");
            return;
        }
        let kernel = device
            .compile(crate::testing::SAXPY, crate::testing::SAXPY_ENTRY)
            .expect("saxpy compiles");
        let mut empty = Empty::new(&device);
        let mut batch = device.batch().expect("a command buffer opens");

        let mut encoded = 0;
        let err = loop {
            match batch.add(&kernel, &empty.args(), Grid::new(0, 64), 0) {
                Ok(()) => encoded += 1,
                Err(err) => break err,
            }
            assert!(encoded <= SAMPLED_DISPATCHES, "the cap did not hold");
        };

        assert_eq!(encoded, SAMPLED_DISPATCHES, "{encoded} passes were sampled");
        assert!(
            matches!(err, MetalError::TooManySampledDispatches { most } if most == SAMPLED_DISPATCHES),
            "{err}"
        );
        // The refused dispatch left no pass behind it, so the batch is still a
        // batch and still submits — a refusal to *time* a dispatch is not a
        // command buffer nobody can close.
        batch.wait().expect("the batch completes");
        device.time_each_dispatch(false).expect("sampling stops");
    }

    /// Sampling is a switch and not a mode a device is built in, so the rows a
    /// run produces are the rows of the batches that were open while it was on.
    #[test]
    fn a_batch_opened_after_sampling_stopped_is_not_timed() {
        let Some(device) = device() else { return };
        if device.time_each_dispatch(true).is_err() {
            eprintln!("skipping: this device does not sample at a stage boundary");
            return;
        }
        let kernel = device
            .compile(crate::testing::SAXPY, crate::testing::SAXPY_ENTRY)
            .expect("saxpy compiles");
        let mut empty = Empty::new(&device);
        let mut run = || {
            device
                .run(&kernel, &empty.args(), Grid::new(0, 64), 0)
                .expect("the dispatch completes");
        };

        profile::take();
        run();
        device.time_each_dispatch(false).expect("sampling stops");
        run();

        let profile = profile::take();
        let rows = profile.kernels();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].1.calls, 1, "the unsampled batch was timed too");
    }
}
