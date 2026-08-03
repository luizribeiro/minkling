//! Source string to compute pipeline, and pipeline to result.

use std::ptr::NonNull;
use std::time::Duration;

use inkling_core::profile::{self, Op};
use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use crate::buffer::Arg;
use crate::device::{Device, MetalError, RoundTrip};
use crate::sampling::Sampled;

/// Entries in one compute function's buffer argument table. Every Apple GPU
/// family states 31, and binding past it raises an Objective-C exception —
/// which unwinds through no Rust destructor and takes the process with it, so
/// it has to be caught on this side.
const ARGUMENT_SLOTS: usize = 31;

/// One compiled entry point, ready to dispatch.
///
/// Compiling produces a whole library and then a pipeline for one function in
/// it. Only the pipeline is kept: the library exists to be searched, and a
/// second entry point out of the same source is a second [`Device::compile`].
#[derive(Debug)]
pub struct Kernel {
    entry: String,
    /// What the profile calls this, which is the entry point's own name unless
    /// [`Kernel::under`] gave it another — see there for why one pipeline is
    /// worth two rows.
    label: String,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Device {
    /// Compile `source` and take `entry` out of it.
    ///
    /// The three ways this fails are three different mistakes — source that
    /// does not compile, a name that is not in it, and a function that compiles
    /// but cannot be a compute pipeline — and they are three errors, because
    /// the first one carries the compiler's own diagnostic and the others have
    /// nothing to do with it.
    pub fn compile(&self, source: &str, entry: &str) -> Result<Kernel, MetalError> {
        let library = self
            .raw()
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(|err| MetalError::Compile(diagnostic(&err)))?;

        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| MetalError::NoSuchKernel(entry.to_owned()))?;

        let pipeline = self
            .raw()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| MetalError::Pipeline {
                entry: entry.to_owned(),
                diagnostic: diagnostic(&err),
            })?;

        Ok(Kernel {
            label: entry.to_owned(),
            entry: entry.to_owned(),
            pipeline,
        })
    }

    /// Run `kernel` over `grid`, with `args` bound to buffer slots `0..`, and
    /// wait for it.
    ///
    /// A [`Batch`] of one, which is what a caller with a single dispatch wants
    /// and what everything here did before there was a second one to put beside
    /// it.
    pub fn run(
        &self,
        kernel: &Kernel,
        args: &[Arg<'_>],
        grid: Grid,
        moves: usize,
    ) -> Result<(), MetalError> {
        let mut batch = self.batch()?;
        batch.add(kernel, args, grid, moves)?;
        batch.wait()
    }

    /// An open command buffer, to encode dispatches into and wait for once.
    pub fn batch(&self) -> Result<Batch<'_>, MetalError> {
        let commands = self
            .queue()
            .commandBuffer()
            .ok_or(MetalError::NoCommandBuffer)?;
        let samples = self.timestamps().as_ref().map(Sampled::open);
        Ok(Batch {
            device: self,
            commands,
            encoder: None,
            samples,
            entry: None,
            dispatches: 0,
        })
    }
}

/// Dispatches encoded into one command buffer, submitted and waited for
/// together.
///
/// **A submission costs 225 microseconds and the arithmetic inside one costs
/// less.** Measured on this machine with a kernel that writes a single float:
/// opening a command buffer, committing it and waiting for it is 225 µs
/// whatever is in it, where a decode-shaped `[1, 4096] @ [4096, 4096]ᵀ`
/// projection against packed weights adds 105 µs of its own. So a decode step's
/// dispatches, each submitted alone, were 94 ms of round trip — most of the 163
/// ms the step took then — and what shares an input should share a command
/// buffer. A step now encodes 869 dispatches into 87 of these, and what it
/// spends waiting for them is 81% of it, of which the device is executing for a
/// little over half.
///
/// **225 µs is what one costs alone, and it is not what one costs at the
/// margin.** Four alternating measurements have taken 40 or 42 command buffers
/// out of a step and read the wait row either side: 157 µs each out of 249, 172
/// out of 209, 156 out of 167 and 165 out of 127. The larger number is what a
/// round trip is worth in isolation; the four are what removing one from a
/// stream of them is worth, and they are what an estimate should be built on.
/// None of them is the same number twice, so a plan that turns on the
/// difference between 156 and 172 is a plan that has over-fitted.
///
/// **The dispatches are ordered.** Metal's default dispatch type is serial, so
/// each one here runs after the one before it and reads what it wrote. That is
/// not what makes batching worth doing — the four projections that consume a
/// layer's normed hidden state are independent — but it is what a layer's
/// attention now rests on end to end: eleven dispatches where each reads what
/// the one before it wrote, including into a span that outlives the call.
///
/// Waiting is still what makes [`Buffer::as_slice`](crate::Buffer::as_slice)
/// safe to read afterwards. What a batch removes is the *number* of waits, not
/// the wait: the alternative to that is a second thing for this process to do
/// while the GPU works, and there is not one yet.
pub struct Batch<'a> {
    device: &'a Device,
    commands: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// The open compute pass, if one is open. One pass holds every dispatch in
    /// the batch — unless the device is timing each of them, which it can only
    /// do at the ends of a pass, and then a dispatch is a pass.
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// The timestamps those passes take, when there are passes to take them at.
    samples: Option<Sampled>,
    /// The first kernel encoded, for the error a failure comes back as. A
    /// command buffer fails as a whole and Metal does not say which dispatch
    /// inside it did, so naming the first is as precise as this can be.
    entry: Option<String>,
    dispatches: usize,
}

impl<'a> Batch<'a> {
    /// The device this was opened on, for a kernel that holds no device of its
    /// own — an activation between two dispatches belongs to neither weight, so
    /// [`SwiGlu`](crate::SwiGlu) is one pipeline for the whole model and takes
    /// the device it allocates against from the batch it encodes into.
    pub(crate) fn device(&self) -> &'a Device {
        self.device
    }

    /// Encode `kernel` over `grid`, with `args` bound to buffer slots `0..`.
    ///
    /// The buffers have to outlive the [`Batch::wait`] and not only this call.
    /// A command buffer retains what is bound into it, so nothing is freed
    /// under a running dispatch — but a caller that wants to *read* an output
    /// still holds the buffer it reads.
    ///
    /// An [`Arg::Inline`] is the exception and is why the two are separate
    /// variants: `setBytes:` copies, so those bytes are the command buffer's
    /// own by the time this returns and the caller's may go.
    ///
    /// `moves` is what this dispatch reads and writes, in bytes, and it is the
    /// caller's to state because nothing here can work it out. **What is bound
    /// is an upper bound and a loose one**: a bank binds 256 experts and reads
    /// six, a layer's attention binds a span with room for a thousand keys and
    /// reads the eight there are, the weighting binds a 258-wide row of logits
    /// and reads eight of it, and a packed weight is bound as the bytes it is
    /// packed into rather than the values it holds. Only the call knows which.
    ///
    /// **Everything that scales with the call, wherever it travels.** A list of
    /// experts or a scale a row is small enough to go in the command buffer
    /// rather than in an allocation — see [`Device::inline`] — and that is a
    /// question about where the bytes are, not about whether they are read. The
    /// one thing left out is the fixed shape struct a dispatch carries, which
    /// is a few dozen bytes whatever the call is.
    ///
    /// It is charged beside the device's own clock, and the two together are
    /// what say how far a kernel is from the memory it waits on — which is the
    /// ranking [`crate::sampling`] exists to produce.
    pub fn add(
        &mut self,
        kernel: &Kernel,
        args: &[Arg<'_>],
        grid: Grid,
        moves: usize,
    ) -> Result<(), MetalError> {
        if grid.threads_per_group > kernel.max_threads_per_group() {
            return Err(MetalError::ThreadgroupTooLarge {
                entry: kernel.entry.clone(),
                asked: grid.threads_per_group,
                most: kernel.max_threads_per_group(),
            });
        }
        if args.len() > ARGUMENT_SLOTS {
            return Err(MetalError::TooManyArguments {
                entry: kernel.entry.clone(),
                asked: args.len(),
                most: ARGUMENT_SLOTS,
            });
        }

        let encoder = self.encoder(kernel)?;
        encoder.setComputePipelineState(&kernel.pipeline);
        for (slot, arg) in args.iter().enumerate() {
            // SAFETY: the memory outlives the encoding through `Arg`'s borrow,
            // offset 0 is within every allocation, `Inline`'s length is the
            // slice's own and is inside the 4 KiB `setBytes:` takes, and `slot`
            // is inside the argument table by the check above.
            //
            // What is *not* checked is that a slot's element type is the one
            // the source declared for it, or that the kernel indexes inside the
            // length it was given. Neither is knowable from here — the source
            // string is the only thing that says — and both stay the kernel
            // author's to get right, the way the body of any `unsafe fn` is.
            unsafe {
                match arg {
                    Arg::Bound(buffer) => encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot),
                    Arg::Inline(bytes) => encoder.setBytes_length_atIndex(
                        NonNull::from(*bytes).cast(),
                        bytes.len(),
                        slot,
                    ),
                }
            };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            one_dimensional(grid.groups()),
            one_dimensional(grid.threads_per_group),
        );

        self.entry.get_or_insert_with(|| kernel.entry.clone());
        self.dispatches += 1;
        if let Some(samples) = &mut self.samples {
            samples.moved(moves);
        }
        Ok(())
    }

    /// The pass this dispatch goes in: the one the batch already has, or a new
    /// one of its own where the device is timing each dispatch.
    ///
    /// **A pass of its own is not a command buffer of its own.** The passes
    /// still go into the one submission the batch is, so a sampled step makes
    /// the same round trips an unsampled one does; what it adds is a boundary
    /// between dispatches that Metal's serial dispatch type already puts a
    /// barrier at.
    fn encoder(
        &mut self,
        kernel: &Kernel,
    ) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, MetalError> {
        let sampled = match &self.samples {
            None => None,
            Some(samples) => Some(samples.pass()?.retain()),
        };
        match sampled {
            Some(pass) => {
                self.end();
                self.encoder = Some(
                    self.commands
                        .computeCommandEncoderWithDescriptor(&pass)
                        .ok_or(MetalError::NoCommandEncoder)?,
                );
                self.samples.as_mut().expect("sampling").ran(&kernel.label);
            }
            None if self.encoder.is_none() => {
                self.encoder = Some(
                    self.commands
                        .computeCommandEncoder()
                        .ok_or(MetalError::NoCommandEncoder)?,
                );
            }
            None => {}
        }
        Ok(self.encoder.as_ref().expect("a pass is open"))
    }

    /// How many dispatches are in this so far, which is what a caller deciding
    /// whether it is worth committing yet has to go on.
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }

    /// Submit everything encoded and wait for all of it.
    pub fn wait(self) -> Result<(), MetalError> {
        self.submit().wait()
    }

    /// Submit everything encoded and do not wait for it.
    ///
    /// **What a caller does with the gap is encode the next command buffer**,
    /// which is the one thing this process has to do that the GPU is not waiting
    /// on. A dispatch costs 4.5 µs to encode and 16 µs to run at a decode step's
    /// shapes, so a caller that commits part way through stays ahead of the
    /// device for the rest of the call — see
    /// [`ModelLayers`](crate::ModelLayers), which is the caller this exists for.
    ///
    /// Nothing about ordering changes. One queue executes its command buffers in
    /// the order they were committed, so a dispatch here still reads what the
    /// dispatch before it wrote whichever buffer either of them went in. What
    /// does change is that **the buffers this retains are held until the
    /// [`Submitted::wait`]**, not until the next one — which is why a caller that
    /// leaves several in flight is the one that has to bound what they hold.
    pub fn submit(mut self) -> Submitted<'a> {
        let _timed = profile::scope(Op::Submit);
        self.end();
        self.commands.commit();
        self.device.counted(self.dispatches);
        Submitted {
            device: self.device,
            commands: self.commands.clone(),
            samples: self.samples.take(),
            entry: self.entry.take(),
            dispatches: self.dispatches,
        }
    }

    /// The open pass closed, which has to happen before the buffer is committed
    /// and before another pass opens, and exactly once for each.
    fn end(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
    }
}

/// A command buffer committed and not yet waited for.
///
/// Waiting is still what makes [`Buffer::as_slice`](crate::Buffer::as_slice)
/// safe to read, and it is still the whole of what a caller gets out of one.
/// What separating the two adds is that the wait can happen later than the
/// commit, and everything encoded between them runs on this process while the
/// device runs what was committed.
///
/// What a reader of one in flight wants is how much is in it, which is what a
/// caller holding several decides against; nothing else here is printable.
#[derive(Debug)]
pub struct Submitted<'a> {
    device: &'a Device,
    commands: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: Option<Sampled>,
    entry: Option<String>,
    dispatches: usize,
}

impl Submitted<'_> {
    /// Wait for it, and charge what it cost.
    pub fn wait(self) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Submit);
        let blocked = std::time::Instant::now();
        // The GPU watchdog kills a command buffer that runs too long, and this
        // project has already met it once: `mlx_lm` mapping tensors off NFS at
        // ~80 MB/s took a kernel past the limit and the driver returned
        // `kIOGPUCommandBufferCallbackErrorTimeout`. It arrives here, as an
        // error on the completed buffer, not as a hang.
        //
        // Batching is what makes the limit worth watching again: it is a limit
        // on the buffer and not on the dispatch, so a caller that put a whole
        // layer in one is asking for the sum to finish in time. A decode step's
        // largest is four projections at 105 µs each, four decades below it.
        self.commands.waitUntilCompleted();
        let waited = blocked.elapsed();
        // One reading of the device's clock, charged to the profile and handed
        // to the record, so that the two figures are the same figure and a case
        // that reads them against each other is reading one measurement.
        let executed = self.executed();
        self.device
            .round_tripped(|| self.round_trip(waited, executed));
        profile::ran_on_the_gpu(executed);
        if let Some(samples) = &self.samples {
            samples.charge();
        }

        match self.commands.error() {
            None => Ok(()),
            Some(err) => Err(MetalError::Execution {
                entry: self.entry.clone().unwrap_or_default(),
                diagnostic: diagnostic(&err),
            }),
        }
    }

    /// How long the GPU was executing this command buffer, which it timestamps
    /// itself and which no clock on this side can see.
    ///
    /// The whole of what makes a submission worth reasoning about separately: a
    /// round trip is 225 microseconds and the arithmetic inside one is less, so
    /// what a profile has to be able to say is how much of the wait was work.
    /// Both timestamps are only meaningful once the buffer has completed, which
    /// is where this is read.
    fn executed(&self) -> Duration {
        since(self.commands.GPUStartTime(), self.commands.GPUEndTime())
    }

    /// The same wait divided into what the driver, the queue and the GPU each
    /// held it for, which is what says whether a round trip is work or asking.
    ///
    /// `executed` is passed in rather than read again for the reason its caller
    /// states. The other two are read here, and only where a caller asked for
    /// the record — see [`Device::record_round_trips`].
    fn round_trip(&self, waited: Duration, executed: Duration) -> RoundTrip {
        RoundTrip {
            dispatches: self.dispatches,
            waited,
            scheduled: since(
                self.commands.kernelStartTime(),
                self.commands.kernelEndTime(),
            ),
            queued: since(self.commands.kernelEndTime(), self.commands.GPUStartTime()),
            executed,
        }
    }
}

/// A batch nobody waited for still has an encoder open, and Metal raises on a
/// command buffer that is released holding one — an Objective-C exception,
/// which unwinds through no Rust destructor. So dropping one closes it, which
/// is what happens when [`Batch::add`] refuses halfway through.
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        self.end();
    }
}

impl Kernel {
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The same compiled pipeline, charged to a row of its own.
    ///
    /// **One kernel is not always one question.** The attention entry runs on
    /// both kinds of layer this checkpoint has — 35 whose queries may reach 512
    /// keys back and 7 whose queries may reach every key there is — and summed
    /// into one row those two are a number about neither: at a prefill of 16384
    /// tokens the second is quadratic in the prompt and the first is linear, and
    /// a table that added them could not say so. Nothing about the dispatch
    /// changes, only the name the profile files it under.
    ///
    /// The pipeline is shared rather than compiled again, so a second row costs
    /// a retain and a string. The entry point stays what it is, because that is
    /// what a failed dispatch has to name for anyone to find it in the source.
    pub fn under(&self, label: &str) -> Self {
        Self {
            entry: self.entry.clone(),
            label: label.to_owned(),
            pipeline: self.pipeline.clone(),
        }
    }

    /// The widest threadgroup this kernel can be dispatched in, which is a
    /// property of the compiled kernel and not of the device: register pressure
    /// lowers it below the device's 1024.
    pub fn max_threads_per_group(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    /// How many threads of this kernel execute in lockstep, which is what a
    /// `simd_`-prefixed reduction inside it reduces over.
    ///
    /// Asked rather than assumed, because a kernel that gives one simdgroup to
    /// each output has to divide its grid by the same number the hardware is
    /// using. Every Apple GPU states 32 and Metal makes no promise that it
    /// always will.
    pub fn simd_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }
}

/// A one-dimensional dispatch: how many threads the work needs, and how many of
/// them share a threadgroup.
///
/// Threadgroups are dispatched whole, so the last one runs
/// `groups * threads_per_group - threads` threads past the end of the work and
/// the kernel has to say so itself — the usual `if (i >= count) return;`. Metal
/// will also dispatch a non-uniform grid and trim that tail in hardware, but
/// only from a threadgroup count it derives, and stating the count is what
/// makes a kernel that tiles by threadgroup index able to reason about it.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    threads: usize,
    threads_per_group: usize,
}

impl Grid {
    pub fn new(threads: usize, threads_per_group: usize) -> Self {
        assert!(threads_per_group > 0, "a threadgroup holds some threads");
        Self {
            threads,
            threads_per_group,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn threads_per_group(&self) -> usize {
        self.threads_per_group
    }

    /// Threadgroups enough to cover the threads, rounding up.
    pub fn groups(&self) -> usize {
        self.threads.div_ceil(self.threads_per_group)
    }
}

/// A shape a kernel reads as a `uint`.
///
/// Here rather than beside the one kernel that first needed it, because every
/// kernel that takes a shape needs the same check for the same reason:
/// unreachable through any real call — four billion of anything is decades past
/// what one allocation can hold — and a truncation would not fail. It would
/// dispatch a grid for the wrong shape over buffers of the right one.
pub(crate) fn extent(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{value} is wider than a kernel's uint: {what}"))
}

/// The interval between two of a command buffer's timestamps, which the driver
/// reports as seconds on the host's clock.
///
/// Clamped at nothing because the four are only meaningful on a buffer that has
/// completed, and a device that reported them out of order — or reported none
/// at all, which is a zero — is not a reason to build a negative interval.
fn since(from: f64, to: f64) -> Duration {
    Duration::from_secs_f64((to - from).max(0.0))
}

fn one_dimensional(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

/// What an `NSError` has to say, which for a compile failure is the compiler's
/// own output — file, line, column, caret and all.
pub(crate) fn diagnostic(err: &NSError) -> String {
    err.localizedDescription().to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::buffer::Buffer;
    use crate::testing::{SAXPY, SAXPY_ENTRY, device, saxpy_moves};

    /// Long enough that a dispatch over it spans several threadgroups, and not
    /// a multiple of the threadgroup size, so the tail group runs threads the
    /// kernel's bounds check has to turn away.
    const LEN: usize = 4099;
    const THREADS_PER_GROUP: usize = 64;
    const ALPHA: f32 = 2.5;

    /// Everything one saxpy dispatch needs, held together so the buffers
    /// outlive the [`Arg`]s taken off them.
    struct Saxpy {
        alpha: Buffer<f32>,
        count: Buffer<u32>,
        x: Buffer<f32>,
        y: Buffer<f32>,
        out: Buffer<f32>,
    }

    impl Saxpy {
        /// `LEN` inputs, but only `count` of them claimed, so a caller can ask
        /// the kernel to stop short of the buffers it was given.
        fn new(device: &Device, count: usize) -> Self {
            let x: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.125 - 7.0).collect();
            let y: Vec<f32> = (0..LEN).map(|i| 3.0 - i as f32 * 0.0625).collect();
            Self {
                alpha: device.buffer(&[ALPHA]).unwrap(),
                count: device.buffer(&[count as u32]).unwrap(),
                x: device.buffer(&x).unwrap(),
                y: device.buffer(&y).unwrap(),
                out: device.zeroed(LEN).unwrap(),
            }
        }

        fn args(&mut self) -> [Arg<'_>; 5] {
            [
                self.alpha.arg(),
                self.count.arg(),
                self.x.arg(),
                self.y.arg(),
                self.out.arg(),
            ]
        }

        /// The same arithmetic the kernel is asked for, over the same inputs.
        fn on_the_cpu(&self) -> Vec<f32> {
            self.x
                .as_slice()
                .iter()
                .zip(self.y.as_slice())
                .map(|(x, y)| ALPHA * x + y)
                .collect()
        }
    }

    #[test]
    fn a_kernel_reports_the_threadgroup_it_can_be_dispatched_in() {
        let Some(device) = device() else { return };

        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");

        assert_eq!(kernel.entry(), SAXPY_ENTRY);
        assert!(kernel.max_threads_per_group() >= THREADS_PER_GROUP);
    }

    /// A dispatch that gives one simdgroup to each unit of work sizes its grid
    /// from this and takes its output index from `thread_position_in_grid`
    /// divided by it, so what it needs is a width that divides the threadgroup
    /// evenly — not the 32 every Apple GPU happens to report.
    #[test]
    fn a_kernel_reports_the_simdgroup_it_executes_in() {
        let Some(device) = device() else { return };

        let width = device
            .compile(SAXPY, SAXPY_ENTRY)
            .expect("saxpy compiles")
            .simd_width();

        assert!(width > 0 && width.is_power_of_two(), "{width}");
        assert_eq!(THREADS_PER_GROUP % width, 0, "{width}");
    }

    /// The whole point of the crate: source in, dispatch, and the same
    /// arithmetic out that the CPU does. Exact rather than approximate — one
    /// multiply and one add of the same f32s round the same way on both — so a
    /// tolerance here would only be hiding a plumbing bug.
    #[test]
    fn saxpy_matches_the_cpu_elementwise() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);

        let grid = Grid::new(LEN, THREADS_PER_GROUP);
        assert!(grid.groups() > 1, "one threadgroup proves nothing");
        device
            .run(&kernel, &saxpy.args(), grid, saxpy_moves(LEN))
            .expect("the dispatch completes");

        assert_eq!(saxpy.out.to_vec(), saxpy.on_the_cpu());
    }

    /// The tail threads of the last group index past the end of the work, and
    /// the arithmetic above only proves the kernel turned them away — not that
    /// nothing was written where they would have landed. This asks the second
    /// question, by claiming one element fewer than the buffers hold.
    #[test]
    fn the_tail_threadgroup_writes_nothing_past_the_count() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let short = LEN - 1;
        let mut saxpy = Saxpy::new(&device, short);

        device
            .run(
                &kernel,
                &saxpy.args(),
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect("the dispatch completes");

        assert_eq!(saxpy.out.as_slice()[short], 0.0);
        assert_eq!(saxpy.out.as_slice()[..short], saxpy.on_the_cpu()[..short]);
    }

    /// A grid of nothing is a dispatch of no threadgroups, which Metal takes
    /// and which has to stay a no-op rather than a refusal: a batch that
    /// happens to be empty is the caller's business, not an error.
    #[test]
    fn an_empty_grid_runs_and_writes_nothing() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);

        device
            .run(
                &kernel,
                &saxpy.args(),
                Grid::new(0, THREADS_PER_GROUP),
                saxpy_moves(0),
            )
            .expect("an empty dispatch completes");

        assert_eq!(saxpy.out.to_vec(), vec![0.0; LEN]);
    }

    /// A Metal compile error with the message swallowed is the worst thing this
    /// crate could do to whoever writes the next kernel, so the compiler's own
    /// text has to survive the trip through `NSError`.
    #[test]
    fn a_malformed_kernel_reports_the_compilers_own_diagnostic() {
        let Some(device) = device() else { return };
        let source = "#include <metal_stdlib>\nkernel void broken() { not_a_function(); }\n";

        let err = device
            .compile(source, "broken")
            .expect_err("the source does not compile");

        let message = err.to_string();
        assert!(message.contains("not_a_function"), "{message}");
        assert!(message.contains("program_source:2"), "{message}");
    }

    #[test]
    fn a_missing_entry_point_is_not_a_compile_error() {
        let Some(device) = device() else { return };

        let err = device
            .compile(SAXPY, "daxpy")
            .expect_err("the source has no daxpy");

        assert!(matches!(err, MetalError::NoSuchKernel(name) if name == "daxpy"));
    }

    /// Source can compile, hold the name, and still not be dispatchable: a
    /// vertex function is found by `newFunctionWithName` like any other and
    /// fails only at the pipeline. Three failures, and the caller is told which.
    #[test]
    fn a_function_that_is_not_a_kernel_fails_at_the_pipeline() {
        let Some(device) = device() else { return };
        let source = "#include <metal_stdlib>\nvertex float4 shade() { return float4(0); }\n";

        let err = device
            .compile(source, "shade")
            .expect_err("a vertex function makes no compute pipeline");

        assert!(
            matches!(err, MetalError::Pipeline { ref entry, .. } if entry == "shade"),
            "{err}"
        );
    }

    /// Metal raises an Objective-C exception for an oversized threadgroup,
    /// which unwinds through no Rust destructor and aborts the test binary. The
    /// check has to happen before the encoder sees it.
    #[test]
    fn an_oversized_threadgroup_is_refused_before_dispatch() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let too_many = kernel.max_threads_per_group() + 1;

        let err = device
            .run(&kernel, &[], Grid::new(LEN, too_many), saxpy_moves(LEN))
            .expect_err("the threadgroup is too large");

        assert!(matches!(err, MetalError::ThreadgroupTooLarge { asked, .. } if asked == too_many));
    }

    /// The same aborting exception, from the other end: one binding past the
    /// argument table.
    #[test]
    fn too_many_bound_buffers_are_refused_before_dispatch() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut buffers: Vec<Buffer<f32>> = (0..ARGUMENT_SLOTS + 1)
            .map(|_| device.zeroed(1).unwrap())
            .collect();
        let args: Vec<Arg<'_>> = buffers.iter_mut().map(Buffer::arg).collect();

        let err = device
            .run(
                &kernel,
                &args,
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect_err("one buffer too many");

        assert!(matches!(err, MetalError::TooManyArguments { most, .. } if most == ARGUMENT_SLOTS));
    }

    /// Several dispatches in one command buffer are the same arithmetic as
    /// several command buffers, and one wait rather than several.
    #[test]
    fn a_batch_runs_every_dispatch_it_was_given() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut first = Saxpy::new(&device, LEN);
        let mut second = Saxpy::new(&device, LEN);

        let (dispatches, submissions) = (device.dispatches(), device.submissions());
        let mut batch = device.batch().expect("a command buffer opens");
        for saxpy in [&mut first, &mut second] {
            batch
                .add(
                    &kernel,
                    &saxpy.args(),
                    Grid::new(LEN, THREADS_PER_GROUP),
                    saxpy_moves(LEN),
                )
                .expect("the dispatch encodes");
        }
        batch.wait().expect("the batch completes");

        assert_eq!(first.out.to_vec(), first.on_the_cpu());
        assert_eq!(second.out.to_vec(), second.on_the_cpu());
        assert_eq!(device.dispatches() - dispatches, 2);
        assert_eq!(device.submissions() - submissions, 1, "one command buffer");
    }

    /// A batch is a sequence and not a race: Metal's default dispatch type is
    /// serial, so a dispatch reads what the one before it wrote.
    ///
    /// Nothing batched today depends on that — the projections that share a
    /// submission are independent of each other — but a dependent pair in one
    /// command buffer is the next thing anyone will reach for, and whether that
    /// is allowed is a property of Metal rather than of this crate.
    #[test]
    fn a_dispatch_reads_what_the_one_before_it_in_the_batch_wrote() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);

        // The second reads the first's output as its own `x`, so its answer is
        // `alpha * (alpha * x + y) + y` and only an ordered batch produces it.
        let mut chained = Saxpy::new(&device, LEN);
        let mut batch = device.batch().expect("a command buffer opens");
        batch
            .add(
                &kernel,
                &saxpy.args(),
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect("the dispatch encodes");
        let args = [
            chained.alpha.arg(),
            chained.count.arg(),
            saxpy.out.arg(),
            chained.y.arg(),
            chained.out.arg(),
        ];
        batch
            .add(
                &kernel,
                &args,
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect("the dispatch encodes");
        batch.wait().expect("the batch completes");

        let want: Vec<f32> = saxpy
            .on_the_cpu()
            .iter()
            .zip(chained.y.as_slice())
            .map(|(x, y)| ALPHA * x + y)
            .collect();
        assert_eq!(chained.out.to_vec(), want);
    }

    /// **The same sequence across two command buffers, the first committed and
    /// not waited for**, which is the whole of what lets a caller keep encoding
    /// while the device runs what it has already been given.
    ///
    /// One queue is a serial ordering — see [`Device`], where the queue is
    /// opened once for that reason — so the second buffer's dispatch runs after
    /// the first's and reads what it wrote. What is not obvious is that this
    /// holds while the first is still *running*: the second is encoded, bound
    /// and committed against a buffer nothing on this side has waited for.
    ///
    /// The answer is the case above's, exactly, which is what says the split
    /// changed the scheduling and not the arithmetic.
    #[test]
    fn a_dispatch_reads_what_a_command_buffer_nobody_waited_for_wrote() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);
        let mut chained = Saxpy::new(&device, LEN);
        let submissions = device.submissions();

        let mut first = device.batch().expect("a command buffer opens");
        first
            .add(
                &kernel,
                &saxpy.args(),
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect("the dispatch encodes");
        let running = first.submit();

        let mut second = device.batch().expect("a command buffer opens");
        let args = [
            chained.alpha.arg(),
            chained.count.arg(),
            saxpy.out.arg(),
            chained.y.arg(),
            chained.out.arg(),
        ];
        second
            .add(
                &kernel,
                &args,
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect("the dispatch encodes");
        running.wait().expect("the first completes");
        second.wait().expect("the second completes");

        let want: Vec<f32> = saxpy
            .on_the_cpu()
            .iter()
            .zip(chained.y.as_slice())
            .map(|(x, y)| ALPHA * x + y)
            .collect();
        assert_eq!(chained.out.to_vec(), want);
        assert_eq!(
            device.submissions() - submissions,
            2,
            "the commit is what counts a submission, not the wait"
        );
    }

    /// **A timed dispatch is a pass of its own and not a command buffer of its
    /// own**, which is the whole design of [`crate::sampling`]: a step that
    /// sampled by submitting each dispatch alone would be measuring an engine
    /// nobody runs. Same arithmetic, same one submission, and a row a kernel
    /// with the device's own clock in it.
    #[test]
    fn a_sampled_batch_times_each_dispatch_without_submitting_each_one() {
        let Some(device) = device() else { return };
        if device.time_each_dispatch(true).is_err() {
            eprintln!("skipping: this device does not sample at a stage boundary");
            return;
        }
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut first = Saxpy::new(&device, LEN);
        let mut second = Saxpy::new(&device, LEN);

        let submissions = device.submissions();
        profile::take();
        let mut batch = device.batch().expect("a command buffer opens");
        for saxpy in [&mut first, &mut second] {
            batch
                .add(
                    &kernel,
                    &saxpy.args(),
                    Grid::new(LEN, THREADS_PER_GROUP),
                    saxpy_moves(LEN),
                )
                .expect("the dispatch encodes");
        }
        batch.wait().expect("the batch completes");
        let profile = profile::take();

        assert_eq!(
            first.out.to_vec(),
            first.on_the_cpu(),
            "the same arithmetic"
        );
        assert_eq!(second.out.to_vec(), second.on_the_cpu());
        assert_eq!(device.submissions() - submissions, 1, "one command buffer");

        let rows = profile.kernels();
        assert_eq!(rows.len(), 1, "one kernel ran: {rows:?}");
        assert_eq!(rows[0].0, SAXPY_ENTRY);
        assert_eq!(rows[0].1.calls, 2, "both dispatches were timed");
        assert!(rows[0].1.elapsed > Duration::ZERO, "{rows:?}");
        eprintln!(
            "{:.2?} over two sampled passes, inside a command buffer the device clocked at {:.2?}",
            profile.dispatched(),
            profile.gpu()
        );
    }

    /// A batch nobody encoded anything into opens no pass at all now that a
    /// pass is opened where a dispatch asks for one, and it still submits: a
    /// caller whose work turned out to be empty gets an empty command buffer
    /// rather than a refusal.
    #[test]
    fn a_batch_of_no_dispatches_submits() {
        let Some(device) = device() else { return };

        let submissions = device.submissions();
        device
            .batch()
            .expect("a command buffer opens")
            .wait()
            .expect("the batch completes");

        assert_eq!(device.submissions() - submissions, 1);
    }

    /// Sampling switched off leaves the batch as it was — one pass for every
    /// dispatch in it, and no rows.
    #[test]
    fn an_unsampled_batch_has_no_kernel_rows() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);

        profile::take();
        device
            .run(
                &kernel,
                &saxpy.args(),
                Grid::new(LEN, THREADS_PER_GROUP),
                saxpy_moves(LEN),
            )
            .expect("the dispatch completes");

        assert!(!device.timing_each_dispatch());
        assert!(profile::take().kernels().is_empty());
        assert_eq!(saxpy.out.to_vec(), saxpy.on_the_cpu());
    }

    /// What the granularity question rests on: a submission costs the same
    /// whatever is in it, so N dispatches in one command buffer cost what one
    /// does and N command buffers cost N times that.
    ///
    /// The kernel here is deliberately trivial — a saxpy over 4099 elements —
    /// so that what is being timed is the round trip and not the arithmetic.
    /// Nothing asserts a ratio; what is asserted is the direction, and the
    /// numbers go to stderr for the commit message to quote.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn a_batch_of_dispatches_costs_less_than_the_same_dispatches_apart() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);
        let grid = Grid::new(LEN, THREADS_PER_GROUP);
        const DISPATCHES: usize = 8;

        // Warm: the first dispatch of a fresh pipeline pays for the driver's
        // first look at these buffers, which a decode loop pays once.
        for _ in 0..2 {
            device
                .run(&kernel, &saxpy.args(), grid, saxpy_moves(LEN))
                .expect("it runs");
        }

        let started = Instant::now();
        for _ in 0..DISPATCHES {
            device
                .run(&kernel, &saxpy.args(), grid, saxpy_moves(LEN))
                .expect("it runs");
        }
        let apart = started.elapsed();

        let started = Instant::now();
        let mut batch = device.batch().expect("a command buffer opens");
        for _ in 0..DISPATCHES {
            batch
                .add(&kernel, &saxpy.args(), grid, saxpy_moves(LEN))
                .expect("it encodes");
        }
        batch.wait().expect("the batch completes");
        let together = started.elapsed();

        eprintln!(
            "{DISPATCHES} dispatches: {apart:.2?} apart ({:.2?} each), {together:.2?} in one \
             command buffer ({:.2?} each)",
            apart / DISPATCHES as u32,
            together / DISPATCHES as u32,
        );
        assert!(together < apart, "{together:?} against {apart:?}");
    }

    #[test]
    fn a_grid_covers_every_thread() {
        let grid = Grid::new(4099, 64);

        assert_eq!(grid.threads(), 4099);
        assert_eq!(grid.threads_per_group(), 64);
        assert_eq!(grid.groups(), 65);
        assert!(grid.groups() * grid.threads_per_group() >= grid.threads());
    }
}
