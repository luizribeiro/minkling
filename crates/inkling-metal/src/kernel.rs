//! Source string to compute pipeline, and pipeline to result.

use std::time::Duration;

use inkling_core::profile::{self, Op};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use crate::buffer::Arg;
use crate::device::{Device, MetalError};

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
    pub fn run(&self, kernel: &Kernel, args: &[Arg<'_>], grid: Grid) -> Result<(), MetalError> {
        let mut batch = self.batch()?;
        batch.add(kernel, args, grid)?;
        batch.wait()
    }

    /// An open command buffer, to encode dispatches into and wait for once.
    pub fn batch(&self) -> Result<Batch<'_>, MetalError> {
        let commands = self
            .queue()
            .commandBuffer()
            .ok_or(MetalError::NoCommandBuffer)?;
        let encoder = commands
            .computeCommandEncoder()
            .ok_or(MetalError::NoCommandEncoder)?;
        Ok(Batch {
            device: self,
            commands,
            encoder,
            entry: None,
            dispatches: 0,
            ended: false,
        })
    }
}

/// Dispatches encoded into one command buffer, submitted and waited for
/// together.
///
/// **A submission costs 206 microseconds and the arithmetic inside one costs
/// less.** Measured on this machine with a kernel that writes a single float:
/// opening a command buffer, committing it and waiting for it is 206 µs
/// whatever is in it, where a decode-shaped `[1, 4096] @ [4096, 4096]ᵀ`
/// projection against packed weights adds 105 µs of its own. So a decode step's
/// dispatches, each submitted alone, were 94 ms of round trip — most of the 163
/// ms the step took then — and what shares an input should share a command
/// buffer. A step now encodes 539 dispatches into 209 of these, and what it
/// spends waiting for them is 79% of it, of which the device is executing for a
/// third.
///
/// **206 µs is what one costs alone, and it is not what one costs at the
/// margin.** Taking 40 command buffers out of a step of 249 — the same
/// dispatches, merged — took about 6.3 ms off the wait over seven alternating
/// pairs, which is about 157 µs each. The larger number is what a round trip is
/// worth in isolation; the smaller is what removing one from a stream of them
/// is worth, and it is the one an estimate should be built on.
///
/// **The dispatches are ordered.** Metal's default dispatch type is serial, so
/// each one here runs after the one before it and reads what it wrote. That is
/// not what makes batching worth doing — the four projections that consume a
/// layer's normed hidden state are independent — but it is what makes it safe
/// to put a dependent pair in one buffer, which `gate` and `down` will want.
///
/// Waiting is still what makes [`Buffer::as_slice`](crate::Buffer::as_slice)
/// safe to read afterwards. What a batch removes is the *number* of waits, not
/// the wait: the alternative to that is a second thing for this process to do
/// while the GPU works, and there is not one yet.
pub struct Batch<'a> {
    device: &'a Device,
    commands: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// The first kernel encoded, for the error a failure comes back as. A
    /// command buffer fails as a whole and Metal does not say which dispatch
    /// inside it did, so naming the first is as precise as this can be.
    entry: Option<String>,
    dispatches: usize,
    ended: bool,
}

impl Batch<'_> {
    /// Encode `kernel` over `grid`, with `args` bound to buffer slots `0..`.
    ///
    /// The buffers have to outlive the [`Batch::wait`] and not only this call.
    /// A command buffer retains what is bound into it, so nothing is freed
    /// under a running dispatch — but a caller that wants to *read* an output
    /// still holds the buffer it reads.
    pub fn add(&mut self, kernel: &Kernel, args: &[Arg<'_>], grid: Grid) -> Result<(), MetalError> {
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

        self.encoder.setComputePipelineState(&kernel.pipeline);
        for (slot, arg) in args.iter().enumerate() {
            // SAFETY: the buffer outlives the encoding through `Arg`'s borrow,
            // offset 0 is within every allocation, and `slot` is inside the
            // argument table by the check above.
            //
            // What is *not* checked is that a slot's element type is the one
            // the source declared for it, or that the kernel indexes inside the
            // length it was given. Neither is knowable from here — the source
            // string is the only thing that says — and both stay the kernel
            // author's to get right, the way the body of any `unsafe fn` is.
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(arg.raw()), 0, slot)
            };
        }
        self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
            one_dimensional(grid.groups()),
            one_dimensional(grid.threads_per_group),
        );

        self.entry.get_or_insert_with(|| kernel.entry.clone());
        self.dispatches += 1;
        Ok(())
    }

    /// Submit everything encoded and wait for all of it.
    pub fn wait(mut self) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Submit);
        self.end();
        self.commands.commit();

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
        self.device.counted(self.dispatches);
        profile::ran_on_the_gpu(self.executed());

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
    /// round trip is 206 microseconds and the arithmetic inside one is less, so
    /// what a profile has to be able to say is how much of the wait was work.
    /// Both timestamps are only meaningful once the buffer has completed, which
    /// is where this is read.
    fn executed(&self) -> Duration {
        Duration::from_secs_f64(
            (self.commands.GPUEndTime() - self.commands.GPUStartTime()).max(0.0),
        )
    }

    /// The encoder closed, which has to happen before the buffer is committed
    /// and exactly once.
    fn end(&mut self) {
        if !self.ended {
            self.encoder.endEncoding();
            self.ended = true;
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

fn one_dimensional(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

/// What an `NSError` has to say, which for a compile failure is the compiler's
/// own output — file, line, column, caret and all.
fn diagnostic(err: &NSError) -> String {
    err.localizedDescription().to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::buffer::Buffer;
    use crate::testing::{SAXPY, SAXPY_ENTRY, device};

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
            .run(&kernel, &saxpy.args(), grid)
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
            .run(&kernel, &saxpy.args(), Grid::new(LEN, THREADS_PER_GROUP))
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
            .run(&kernel, &saxpy.args(), Grid::new(0, THREADS_PER_GROUP))
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
            .run(&kernel, &[], Grid::new(LEN, too_many))
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
            .run(&kernel, &args, Grid::new(LEN, THREADS_PER_GROUP))
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
                .add(&kernel, &saxpy.args(), Grid::new(LEN, THREADS_PER_GROUP))
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
            .add(&kernel, &saxpy.args(), Grid::new(LEN, THREADS_PER_GROUP))
            .expect("the dispatch encodes");
        let args = [
            chained.alpha.arg(),
            chained.count.arg(),
            saxpy.out.arg(),
            chained.y.arg(),
            chained.out.arg(),
        ];
        batch
            .add(&kernel, &args, Grid::new(LEN, THREADS_PER_GROUP))
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

    /// What the granularity question rests on: a submission costs the same
    /// whatever is in it, so N dispatches in one command buffer cost what one
    /// does and N command buffers cost N times that.
    ///
    /// The kernel here is deliberately trivial — a saxpy over 4099 elements —
    /// so that what is being timed is the round trip and not the arithmetic.
    /// Nothing asserts a ratio; what is asserted is the direction, and the
    /// numbers go to stderr for the commit message to quote.
    #[test]
    fn a_batch_of_dispatches_costs_less_than_the_same_dispatches_apart() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let mut saxpy = Saxpy::new(&device, LEN);
        let grid = Grid::new(LEN, THREADS_PER_GROUP);
        const DISPATCHES: usize = 8;

        // Warm: the first dispatch of a fresh pipeline pays for the driver's
        // first look at these buffers, which a decode loop pays once.
        for _ in 0..2 {
            device.run(&kernel, &saxpy.args(), grid).expect("it runs");
        }

        let started = Instant::now();
        for _ in 0..DISPATCHES {
            device.run(&kernel, &saxpy.args(), grid).expect("it runs");
        }
        let apart = started.elapsed();

        let started = Instant::now();
        let mut batch = device.batch().expect("a command buffer opens");
        for _ in 0..DISPATCHES {
            batch.add(&kernel, &saxpy.args(), grid).expect("it encodes");
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
