//! A command sequence encoded once, and what it costs to change between runs.
//!
//! **A decode step encodes the same thousand dispatches every step** — same
//! entries, same order, same threadgroups, same argument slots — and about a
//! third of the step is this process doing that encoding while the GPU runs the
//! step before it. `MTLIndirectCommandBuffer` is the only mechanism Metal offers
//! for encoding a sequence once and running it many times, so the question this
//! module exists to answer is whether the second run is cheaper than the first.
//!
//! **It is a probe rather than a path the engine takes, and it stays one.** The
//! answer is that patching a command is an order cheaper than encoding one and
//! that it does not matter: a decode step's device is executing for 92% of it,
//! so the host has 8% to give — and an indirect command costs the device 2.02×
//! what the dispatch it replaces costs, which is more than the whole of what the
//! host had to offer. The README section "The encode a decode step could stop
//! doing, and what it would be worth" is the arithmetic.
//!
//! What is here is the smallest arrangement that can price a patch against an
//! encode, the constraints found while building it, and one case asserting that
//! a command answers what the dispatch it stands for answers. Nothing here is
//! dispatched by the model.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResource, MTLResourceOptions, MTLResourceUsage,
};

use crate::device::{Device, MetalError};
use crate::kernel::{Grid, Kernel, one_dimensional};

/// A sequence of compute commands the GPU reads, held so that this side can
/// change one of them without re-stating the rest.
///
/// **What it holds beside the Metal object is a command per index.**
/// `indirectComputeCommandAtIndex:` builds an Objective-C object each time it is
/// asked, so a caller that reached for one per patch would be paying an
/// allocation to avoid an encode. They are taken once, at construction, and the
/// probe below prices both ways.
pub struct Indirect {
    raw: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    commands: Vec<Retained<ProtocolObject<dyn MTLIndirectComputeCommand>>>,
}

impl Device {
    /// An indirect command buffer of `commands` compute commands, each able to
    /// bind `slots` buffers.
    ///
    /// The bind count is declared rather than discovered: it is the stride the
    /// driver lays the commands out at, and a command asked for a slot past it
    /// raises rather than grows.
    pub fn indirect(&self, commands: usize, slots: usize) -> Result<Indirect, MetalError> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        descriptor.setMaxKernelBufferBindCount(slots);

        // SAFETY: the count is the caller's and is what the commands below are
        // indexed against; `Shared` is the storage a buffer this process writes
        // and the GPU reads has to be in.
        let raw = unsafe {
            self.raw()
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    commands,
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or(MetalError::NoIndirectCommandBuffer)?;

        // SAFETY: every index is inside the count the buffer was made with.
        let commands = (0..commands)
            .map(|at| unsafe { raw.indirectComputeCommandAtIndex(at) })
            .collect();
        Ok(Indirect { raw, commands })
    }
}

impl Indirect {
    /// How many commands it holds.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// One command, from the vector taken when the buffer was made.
    pub fn at(&self, index: usize) -> &ProtocolObject<dyn MTLIndirectComputeCommand> {
        &self.commands[index]
    }

    /// The same command, asked of the driver rather than held — which is what a
    /// caller that kept no vector would do, and what the probe prices it
    /// against.
    pub fn asked_for(
        &self,
        index: usize,
    ) -> Retained<ProtocolObject<dyn MTLIndirectComputeCommand>> {
        // SAFETY: the index is inside the count the buffer was made with, which
        // is what `commands` was filled from.
        assert!(index < self.commands.len(), "a command this buffer holds");
        unsafe { self.raw.indirectComputeCommandAtIndex(index) }
    }

    /// Write command `index` whole: the entry it runs, the buffers it binds, and
    /// the grid it covers.
    ///
    /// **`barrier` is the difference between this and the encoder it replaces.**
    /// A command buffer's dispatches are serial by default and an indirect
    /// command's are concurrent by default, so a sequence that means what the
    /// encoded one meant asks for a barrier on every command that reads what the
    /// one before it wrote.
    ///
    /// # Safety
    ///
    /// **A bound buffer has to outlive every execution of this command, and the
    /// borrow here says nothing about that.** This is the one place an indirect
    /// command differs in kind from an encoded dispatch rather than in cost: a
    /// command buffer retains what is bound into it and releases it when it
    /// completes, so an encoded binding cannot outlive its allocation. An
    /// indirect command retains nothing. It is a binding written into a
    /// device-side structure that this side will hand over again and again —
    /// which is the whole mechanism — so the buffer has to be held for as long
    /// as the sequence may be executed, which is a promise no signature here can
    /// make. A buffer dropped after this returns leaves a command pointing at
    /// memory the allocator has given away.
    ///
    /// `bound` must also be no longer than the bind count this buffer was opened
    /// with; a slot past it raises an Objective-C exception, which unwinds
    /// through no Rust destructor and takes the process with it.
    pub unsafe fn write(
        &self,
        index: usize,
        kernel: &Kernel,
        bound: &[&ProtocolObject<dyn MTLBuffer>],
        grid: Grid,
        barrier: bool,
    ) {
        let command = self.at(index);
        command.setComputePipelineState(kernel.pipeline());
        for (slot, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffers are kept alive and the slots are inside the
            // bind count by this function's own contract, and offset 0 is inside
            // every allocation.
            unsafe { command.setKernelBuffer_offset_atIndex(buffer, 0, slot) };
        }
        command.concurrentDispatchThreadgroups_threadsPerThreadgroup(
            one_dimensional(grid.groups()),
            one_dimensional(grid.threads_per_group()),
        );
        match barrier {
            true => command.setBarrier(),
            false => command.clearBarrier(),
        }
    }

    /// Bind one buffer into one slot of one command, which is the whole of what
    /// a step that reuses this sequence would do between two runs of it.
    ///
    /// # Safety
    ///
    /// [`Indirect::write`]'s, for the same buffer and the same slot.
    pub unsafe fn rebind(&self, index: usize, slot: usize, buffer: &ProtocolObject<dyn MTLBuffer>) {
        // SAFETY: this function's own contract, which is `write`'s.
        unsafe {
            self.at(index)
                .setKernelBuffer_offset_atIndex(buffer, 0, slot)
        };
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLIndirectCommandBuffer> {
        &self.raw
    }
}

/// `commands` of `indirect` executed by one compute pass, with `resources`
/// declared to it.
///
/// **The declaration is what an indirect sequence costs that an encoded one does
/// not.** A dispatch encoded into a pass tells the driver what it binds; a
/// command inside an indirect buffer does not, so every allocation any of them
/// reads has to be named on the encoder before the pass can be committed. What
/// that comes to is a per-run cost proportional to the *distinct buffers* a step
/// touches rather than to its dispatches, which is why it is priced separately
/// below.
pub fn execute(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    indirect: &Indirect,
    commands: usize,
    resources: &[&ProtocolObject<dyn MTLResource>],
) {
    if !resources.is_empty() {
        let mut pointers: Vec<NonNull<ProtocolObject<dyn MTLResource>>> =
            resources.iter().map(|r| NonNull::from(*r)).collect();
        // SAFETY: the pointers are the borrowed resources' own and the count is
        // the vector's own length.
        unsafe {
            encoder.useResources_count_usage(
                NonNull::new(pointers.as_mut_ptr()).expect("a non-empty vector"),
                pointers.len(),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            )
        };
    }
    // SAFETY: the range is inside the commands the buffer was made with, and
    // the buffer outlives the pass through its borrow and the command buffer's
    // own retain.
    unsafe { encoder.executeCommandsInBuffer_withRange(indirect.raw(), NSRange::new(0, commands)) };
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    use super::*;
    use crate::buffer::Buffer;
    use crate::testing::{SAXPY, SAXPY_ENTRY, device, warmed};

    /// The dispatches a decode step makes, so that a per-command figure is read
    /// beside the count it would be multiplied by.
    const COMMANDS: usize = 1000;

    /// The saxpy kernel's own bindings — a scalar, a count, two inputs and an
    /// output — which is the low end of what this engine's kernels bind and so
    /// the arrangement least flattering to the patch.
    const SLOTS: usize = 5;

    /// Rounds each arm is taken over, the median and the mean of them both
    /// reported: a host-side loop of a thousand Objective-C messages is where
    /// this machine's scheduler shows up, and one reading cannot say whether it
    /// did.
    const ROUNDS: usize = 9;

    /// Everything a saxpy command binds, held so the buffers outlive every
    /// command that names them — **an indirect command retains nothing**, which
    /// is the first thing that separates one from an encoded dispatch.
    struct Bindings {
        alpha: Buffer<f32>,
        count: Buffer<u32>,
        x: Buffer<f32>,
        y: Buffer<f32>,
        out: Buffer<f32>,
    }

    impl Bindings {
        /// A count of zero, so that every thread returns on its first
        /// instruction and what a dispatch costs is its launch rather than its
        /// arithmetic.
        fn new(device: &Device) -> Self {
            Self {
                alpha: device.buffer(&[0.0f32]).expect("the buffer allocates"),
                count: device.buffer(&[0u32]).expect("the buffer allocates"),
                x: device.zeroed(1).expect("the buffer allocates"),
                y: device.zeroed(1).expect("the buffer allocates"),
                out: device.zeroed(1).expect("the buffer allocates"),
            }
        }

        fn bound(&self) -> [&ProtocolObject<dyn MTLBuffer>; SLOTS] {
            [
                self.alpha.raw(),
                self.count.raw(),
                self.x.raw(),
                self.y.raw(),
                self.out.raw(),
            ]
        }

        /// The same five as an encoded dispatch's argument list, so that the
        /// path this crate actually takes is one of the arms.
        fn args(&mut self) -> [crate::Arg<'_>; SLOTS] {
            [
                self.alpha.arg(),
                self.count.arg(),
                self.x.arg(),
                self.y.arg(),
                self.out.arg(),
            ]
        }

        /// The same five as things a pass has to be told an indirect command
        /// may reach.
        fn resources(&self) -> Vec<&ProtocolObject<dyn MTLResource>> {
            self.bound()
                .iter()
                .map(|buffer| ProtocolObject::from_ref(*buffer))
                .collect()
        }
    }

    /// The middle reading and the average of them, which this file reports
    /// together for the reason D2's decode table gives: they can disagree, and
    /// the disagreement is the finding.
    fn middle_and_mean(mut taken: Vec<Duration>) -> (Duration, Duration) {
        taken.sort_unstable();
        let sum: Duration = taken.iter().sum();
        (taken[taken.len() / 2], sum / taken.len() as u32)
    }

    /// `work` timed over [`ROUNDS`] rounds, warm, with whatever it leaves
    /// running waited for outside the clock.
    ///
    /// **The second closure is what makes the first one's figure a host
    /// figure.** An arm that commits a thousand empty dispatches produces 3 ms
    /// of device work in about 1.5 ms of encoding, so an arm that never waited
    /// would fill the queue within two rounds and every reading after that would
    /// be the GPU's backpressure arriving through `commit`. Waiting between
    /// rounds is what leaves each round starting against an idle device.
    fn rounds<T>(mut work: impl FnMut() -> T, mut settle: impl FnMut(T)) -> (Duration, Duration) {
        warmed(|| settle(work()));
        let mut taken = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let opened = Instant::now();
            let running = work();
            taken.push(opened.elapsed());
            settle(running);
        }
        middle_and_mean(taken)
    }

    /// An arm that leaves nothing running, which is every arm that only writes
    /// into an indirect buffer.
    fn nothing<T>(_: T) {}

    /// What the *device* made of the command buffer `encode` filled and closed,
    /// over [`ROUNDS`] rounds, warm.
    ///
    /// The buffer is committed and waited for here rather than by the caller,
    /// because the two timestamps this reads are only meaningful once it has
    /// completed — and the wait is outside no clock, since the clock is the
    /// driver's rather than this process's.
    fn on_the_device(
        mut encode: impl FnMut() -> Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> (Duration, Duration) {
        let mut ran = || {
            let commands = encode();
            commands.commit();
            commands.waitUntilCompleted();
            assert!(commands.error().is_none(), "the pass completes");
            Duration::from_secs_f64((commands.GPUEndTime() - commands.GPUStartTime()).max(0.0))
        };
        warmed(|| {
            ran();
        });
        middle_and_mean((0..ROUNDS).map(|_| ran()).collect())
    }

    /// **What an indirect command costs to patch, against what a dispatch costs
    /// to encode fresh.** The whole of whether there is a milestone in encoding
    /// a decode step's thousand dispatches once instead of every step.
    ///
    /// Every arm is a thousand of something, timed on this process's clock, and
    /// **what any arm leaves running is waited for outside the clock** — see
    /// [`rounds`], where the reason is that a thousand empty dispatches are 3 ms
    /// of device work encoded in 1.5 and an arm that never waited would be
    /// reading the queue rather than the encoding.
    ///
    /// **The last two rows are what a run of a reused sequence costs that an
    /// encoded one does not**, and they are a pair on purpose: an indirect
    /// buffer is executed by a command the driver expands, so a table that
    /// priced the patch and not the expansion would be quoting half of it. One
    /// command and a thousand say whether that expansion is a fixed cost or a
    /// per-command one, which is the whole question.
    ///
    /// The saxpy kernel binds five buffers where this engine's kernels bind six
    /// to a dozen, so the fresh-encode arm here is *cheaper* than the dispatches
    /// it stands for and every ratio below is the conservative one.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_patching_an_indirect_command_costs_against_encoding_it_fresh() {
        let Some(device) = device() else { return };
        let encoded = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let indirect_entry = device
            .compile_indirect(SAXPY, SAXPY_ENTRY)
            .expect("saxpy compiles for an indirect command");
        assert!(
            indirect_entry.supports_indirect() && !encoded.supports_indirect(),
            "the flag is a property of the pipeline and is not inherited"
        );

        let mut bindings = Bindings::new(&device);
        let grid = Grid::new(1, 1);
        let indirect = device
            .indirect(COMMANDS, SLOTS)
            .expect("the indirect command buffer opens");
        for at in 0..COMMANDS {
            // SAFETY: `bindings` owns every buffer named here and outlives both
            // this sequence and every execution of it, and five slots is the
            // bind count the buffer was opened with.
            unsafe { indirect.write(at, &indirect_entry, &bindings.bound(), grid, true) };
        }

        let raw = rounds(
            || {
                let commands = device.queue().commandBuffer().expect("a command buffer");
                let encoder = commands.computeCommandEncoder().expect("a compute pass");
                for _ in 0..COMMANDS {
                    encoder.setComputePipelineState(encoded.pipeline());
                    for (slot, buffer) in bindings.bound().iter().enumerate() {
                        // SAFETY: as `Batch::add`, which this is the raw form of.
                        unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot) };
                    }
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        one_dimensional(grid.groups()),
                        one_dimensional(grid.threads_per_group()),
                    );
                }
                encoder.endEncoding();
                commands.commit();
                commands
            },
            |commands| commands.waitUntilCompleted(),
        );
        let through_batch = rounds(
            || {
                let mut batch = device.batch().expect("a command buffer opens");
                for _ in 0..COMMANDS {
                    batch
                        .add(&encoded, &bindings.args(), grid, 0)
                        .expect("the dispatch encodes");
                }
                batch.submit()
            },
            |running| running.wait().expect("the batch completes"),
        );

        let whole = rounds(
            || {
                for at in 0..COMMANDS {
                    // SAFETY: as above.
                    unsafe { indirect.write(at, &indirect_entry, &bindings.bound(), grid, true) };
                }
            },
            nothing,
        );
        let every_slot = rounds(
            || {
                for at in 0..COMMANDS {
                    for (slot, buffer) in bindings.bound().iter().enumerate() {
                        // SAFETY: as above.
                        unsafe { indirect.rebind(at, slot, buffer) };
                    }
                }
            },
            nothing,
        );
        let one_slot = rounds(
            || {
                for at in 0..COMMANDS {
                    // SAFETY: as above.
                    unsafe { indirect.rebind(at, 2, bindings.bound()[2]) };
                }
            },
            nothing,
        );
        let reached = rounds(
            || {
                for at in 0..COMMANDS {
                    std::hint::black_box(indirect.asked_for(at));
                }
            },
            nothing,
        );

        let resources = bindings.resources();
        let executing = |commands: usize| {
            rounds(
                || {
                    let buffer = device.queue().commandBuffer().expect("a command buffer");
                    let encoder = buffer.computeCommandEncoder().expect("a compute pass");
                    execute(&encoder, &indirect, commands, &resources);
                    encoder.endEncoding();
                    buffer.commit();
                    buffer
                },
                |buffer| buffer.waitUntilCompleted(),
            )
        };
        let one_executed = executing(1);
        let all_executed = executing(COMMANDS);

        eprintln!(
            "  {:>46}{:>11}{:>14}{:>14}",
            "", "each", "a thousand", "mean"
        );
        let row = |what: &str, (middle, mean): (Duration, Duration)| {
            eprintln!(
                "  {what:>46}{:>11}{:>14}{:>14}",
                format!("{:.3}µs", middle.as_secs_f64() * 1e6 / COMMANDS as f64),
                format!("{:.3}ms", middle.as_secs_f64() * 1e3),
                format!("{:.3}ms", mean.as_secs_f64() * 1e3),
            );
            middle
        };
        let encoding = row("a dispatch encoded into a pass, and committed", raw);
        row(
            "the same dispatch through this crate's own path",
            through_batch,
        );
        row("an indirect command written whole", whole);
        row("every one of its five slots rebound", every_slot);
        let patch = row("one slot of it rebound", one_slot);
        row("a command reached for rather than held", reached);
        row("a pass executing one of them, and committed", one_executed);
        let expansion = row("a pass executing the thousand, and committed", all_executed);

        assert!(
            patch * 4 < encoding,
            "patching a command has to be a different order from encoding one: {patch:?} against \
             {encoding:?}"
        );
        assert!(
            expansion < encoding,
            "a sequence handed over whole cannot cost more than encoding it: {expansion:?} against \
             {encoding:?}"
        );
    }

    /// **What the device makes of a thousand indirect commands, against the
    /// thousand dispatches they stand for.**
    ///
    /// The other half of the question, and the half a host-side table cannot
    /// answer: a sequence that is cheaper to hand over is worth nothing if the
    /// GPU is slower to run it. Three arms, one grid, one entry, the kernel
    /// returning on its first instruction so that what is timed is the launch —
    /// dispatches through a pass, indirect commands with a barrier apiece, and
    /// indirect commands with none.
    ///
    /// **The barrier is what makes the second arm the first arm's equal.** A
    /// pass's dispatches are serial and an indirect buffer's are concurrent, so
    /// the third arm answers a different question about ordering. What the two
    /// indirect arms are read against is *each other*, which is the only pair
    /// here that differs in one thing: the pass arm reaches the GPU by another
    /// route and any gap to it is that route rather than the barrier.
    ///
    /// **The clock is the driver's own and not this process's.** Every arm ends
    /// in a wait, and a wall time around one would carry the commit, the wake
    /// and the queue — which for a thousand dispatches that compute nothing is
    /// the same order as what is being measured. `GPUEndTime - GPUStartTime` is
    /// what the device says it was executing for, and it is the figure the rest
    /// of this file's device columns are read off.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_the_device_makes_of_a_thousand_indirect_commands() {
        let Some(device) = device() else { return };
        let encoded = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let entry = device
            .compile_indirect(SAXPY, SAXPY_ENTRY)
            .expect("saxpy compiles for an indirect command");
        let bindings = Bindings::new(&device);
        let grid = Grid::new(1, 1);
        let indirect = device
            .indirect(COMMANDS, SLOTS)
            .expect("the indirect command buffer opens");

        let through_a_pass = on_the_device(|| {
            let commands = device.queue().commandBuffer().expect("a command buffer");
            let encoder = commands.computeCommandEncoder().expect("a compute pass");
            for _ in 0..COMMANDS {
                encoder.setComputePipelineState(encoded.pipeline());
                for (slot, buffer) in bindings.bound().iter().enumerate() {
                    // SAFETY: as `Batch::add`, which this is the raw form of.
                    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot) };
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    one_dimensional(grid.groups()),
                    one_dimensional(grid.threads_per_group()),
                );
            }
            encoder.endEncoding();
            commands
        });

        let resources = bindings.resources();
        let executed = |barrier: bool| {
            for at in 0..COMMANDS {
                // SAFETY: as `what_patching_an_indirect_command_costs_against_
                // encoding_it_fresh` — `bindings` outlives every execution.
                unsafe { indirect.write(at, &entry, &bindings.bound(), grid, barrier) };
            }
            on_the_device(|| {
                let commands = device.queue().commandBuffer().expect("a command buffer");
                let encoder = commands.computeCommandEncoder().expect("a compute pass");
                execute(&encoder, &indirect, COMMANDS, &resources);
                encoder.endEncoding();
                commands
            })
        };
        let barriered = executed(true);
        let concurrent = executed(false);

        eprintln!(
            "  {:>46}{:>11}{:>14}{:>14}",
            "", "each", "a thousand", "mean"
        );
        for (what, (middle, mean)) in [
            ("a thousand dispatches through one pass", through_a_pass),
            ("a thousand indirect commands, a barrier each", barriered),
            ("a thousand indirect commands, no barriers", concurrent),
        ] {
            eprintln!(
                "  {what:>46}{:>11}{:>14}{:>14}",
                format!("{:.3}µs", middle.as_secs_f64() * 1e6 / COMMANDS as f64),
                format!("{:.3}ms", middle.as_secs_f64() * 1e3),
                format!("{:.3}ms", mean.as_secs_f64() * 1e3),
            );
        }
    }

    /// **What declaring a step's allocations costs, as their number grows** —
    /// the one per-run cost an indirect sequence has that an encoded one does
    /// not.
    ///
    /// A dispatch encoded into a pass tells the driver which buffers it binds;
    /// a command inside an indirect buffer does not, so a pass that executes one
    /// has to be told every allocation any command in it may reach. **That is a
    /// cost in the step's distinct buffers rather than in its dispatches**, and
    /// this engine's are hundreds — a wrapped weight per tensor, and the
    /// activations between them — so whether it is priced per buffer or per call
    /// decides whether the declaration is noise or is the whole win back.
    ///
    /// The buffers are one float each, because what is being timed is the
    /// declaration and not the memory.
    ///
    /// **A residency set would remove this row**, and is not measured here: it
    /// is attached to a queue once rather than to a pass each time, which is a
    /// different arrangement and one this probe does not need to price to say
    /// whether the sequence is worth building.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_declaring_a_steps_allocations_to_an_indirect_pass_costs() {
        let Some(device) = device() else { return };
        let entry = device
            .compile_indirect(SAXPY, SAXPY_ENTRY)
            .expect("saxpy compiles for an indirect command");
        let bindings = Bindings::new(&device);
        let indirect = device
            .indirect(1, SLOTS)
            .expect("the indirect command buffer opens");
        // SAFETY: `bindings` outlives the pass below and every buffer in it.
        unsafe { indirect.write(0, &entry, &bindings.bound(), Grid::new(1, 1), true) };

        /// Distinct allocations a step might touch, spanning what this engine
        /// has: a layer's own handful, a stack's weights, and past it.
        const DECLARED: [usize; 5] = [8, 64, 256, 1024, 4096];
        let held: Vec<Buffer<f32>> = (0..*DECLARED.last().expect("a longest arm"))
            .map(|_| device.zeroed(1).expect("the buffer allocates"))
            .collect();

        eprintln!(
            "  {:>28}{:>12}{:>14}{:>14}",
            "declared", "each", "a pass", "mean"
        );
        for declared in DECLARED {
            let resources: Vec<&ProtocolObject<dyn MTLResource>> = held[..declared]
                .iter()
                .map(|buffer| ProtocolObject::from_ref(buffer.raw()))
                .collect();
            let (middle, mean) = rounds(
                || {
                    let commands = device.queue().commandBuffer().expect("a command buffer");
                    let encoder = commands.computeCommandEncoder().expect("a compute pass");
                    execute(&encoder, &indirect, 1, &resources);
                    encoder.endEncoding();
                    commands.commit();
                    commands
                },
                |commands| commands.waitUntilCompleted(),
            );
            eprintln!(
                "  {declared:>28}{:>12}{:>14}{:>14}",
                format!("{:.3}µs", middle.as_secs_f64() * 1e6 / declared as f64),
                format!("{:.3}ms", middle.as_secs_f64() * 1e3),
                format!("{:.3}ms", mean.as_secs_f64() * 1e3),
            );
        }
    }

    /// **What a pipeline built for an indirect command costs the kernel it
    /// compiles**, which is a tax on every dispatch whether or not the indirect
    /// path is the one taken.
    ///
    /// `supportIndirectCommandBuffers` is set on a descriptor before the
    /// pipeline is built and cannot be told to one afterwards, so an engine that
    /// moved to a reused sequence would run every kernel out of a pipeline
    /// carrying it. If that changes what the kernel costs, the change is paid on
    /// the arithmetic rather than on the encoding — which no amount of cheaper
    /// encoding buys back.
    ///
    /// The saxpy is given real work here rather than a count of zero: a launch
    /// measures the dispatch and this row is about the code inside it.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_pipeline_built_for_an_indirect_command_costs() {
        let Some(device) = device() else { return };
        /// Long enough that the dispatch is memory rather than launch, and not a
        /// multiple of the threadgroup so the tail group is exercised.
        const LEN: usize = 1 << 22;
        const WIDE: usize = 256;
        const CALLS: usize = 16;

        let plain = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let flagged = device
            .compile_indirect(SAXPY, SAXPY_ENTRY)
            .expect("saxpy compiles for an indirect command");
        let mut alpha = device.buffer(&[2.5f32]).expect("the buffer allocates");
        let mut count = device.buffer(&[LEN as u32]).expect("the buffer allocates");
        let mut x: Buffer<f32> = device.zeroed(LEN).expect("the buffer allocates");
        let mut y: Buffer<f32> = device.zeroed(LEN).expect("the buffer allocates");
        let mut out: Buffer<f32> = device.zeroed(LEN).expect("the buffer allocates");
        let grid = Grid::new(LEN, WIDE);

        eprintln!(
            "  {:>28}{:>12}{:>14}{:>10}{:>12}",
            "pipeline", "threadgroup", "a dispatch", "moved", "achieved"
        );
        // Both ways round, so that a device climbing to its clock across the
        // sitting cannot be read as a difference between the two pipelines.
        let mut cost = |kernel: &Kernel| {
            crate::testing::device_time(&device, CALLS, |batch| {
                batch
                    .add(
                        kernel,
                        &[alpha.arg(), count.arg(), x.arg(), y.arg(), out.arg()],
                        grid,
                        crate::testing::saxpy_moves(LEN),
                    )
                    .expect("the dispatch encodes");
            })
        };
        crate::testing::warmed(|| {
            cost(&plain);
        });
        let mut measure = |kernel: &Kernel, what: &str| {
            let taken = cost(kernel);
            let moved = crate::testing::saxpy_moves(LEN) as f64;
            eprintln!(
                "  {what:>28}{:>12}{:>14}{:>10}{:>12}",
                kernel.max_threads_per_group(),
                format!("{taken:.2?}"),
                format!("{:.0} MB", moved / 1e6),
                format!("{:.0} GB/s", moved / taken.as_secs_f64() / 1e9),
            );
            taken
        };
        let up = [
            measure(&plain, "plain"),
            measure(&flagged, "for an indirect command"),
        ];
        let down = [
            measure(&flagged, "for an indirect command"),
            measure(&plain, "plain"),
        ];

        assert_eq!(
            plain.max_threads_per_group(),
            flagged.max_threads_per_group(),
            "the flag changed the widest threadgroup this kernel can be dispatched in"
        );
        let ratio = |a: Duration, b: Duration| a.as_secs_f64() / b.as_secs_f64();
        eprintln!(
            "  the flagged pipeline against the plain one: {:.3}× up the list and {:.3}× down it",
            ratio(up[1], up[0]),
            ratio(down[0], down[1]),
        );
    }

    /// **A command written into an indirect buffer answers what the dispatch it
    /// stands for answers**, which is what any of the timings above are worth
    /// anything only if.
    ///
    /// The same saxpy, over the same inputs, once through a compute encoder and
    /// once through a command in an indirect buffer executed by one — and the
    /// two answers compared exactly, because one multiply and one add of the
    /// same floats round the same way whichever encoded them.
    #[test]
    fn an_indirect_command_answers_what_the_dispatch_it_stands_for_answers() {
        let Some(device) = device() else { return };
        const LEN: usize = 4099;
        const WIDE: usize = 64;

        let entry = device
            .compile_indirect(SAXPY, SAXPY_ENTRY)
            .expect("saxpy compiles for an indirect command");
        let x: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.125 - 7.0).collect();
        let y: Vec<f32> = (0..LEN).map(|i| 3.0 - i as f32 * 0.0625).collect();
        let alpha = device.buffer(&[2.5f32]).expect("the buffer allocates");
        let count = device.buffer(&[LEN as u32]).expect("the buffer allocates");
        let xs = device.buffer(&x).expect("the buffer allocates");
        let ys = device.buffer(&y).expect("the buffer allocates");
        let out: Buffer<f32> = device.zeroed(LEN).expect("the buffer allocates");

        let grid = Grid::new(LEN, WIDE);
        let indirect = device
            .indirect(1, SLOTS)
            .expect("the indirect command buffer opens");
        let bound = [alpha.raw(), count.raw(), xs.raw(), ys.raw(), out.raw()];
        // SAFETY: every buffer named is held by this case past the wait below.
        unsafe { indirect.write(0, &entry, &bound, grid, true) };

        let commands = device.queue().commandBuffer().expect("a command buffer");
        let encoder = commands.computeCommandEncoder().expect("a compute pass");
        let resources: Vec<&ProtocolObject<dyn MTLResource>> = bound
            .iter()
            .map(|buffer| ProtocolObject::from_ref(*buffer))
            .collect();
        execute(&encoder, &indirect, 1, &resources);
        encoder.endEncoding();
        commands.commit();
        commands.waitUntilCompleted();
        assert!(commands.error().is_none(), "the indirect pass completes");

        let want: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 2.5 * x + y).collect();
        assert_eq!(out.to_vec(), want);
    }
}
