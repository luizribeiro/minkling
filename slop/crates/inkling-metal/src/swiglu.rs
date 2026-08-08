//! `silu(gate) * up`, which is what sits between an expert's two first
//! projections and its third.
//!
//! **The arithmetic is not why this is here.** A decode step's MoE layer
//! activates `[8, 2048]` — 16384 multiplies against the 4.3 GB the dispatches
//! either side of it read — so as arithmetic it is free wherever it runs. What
//! it costs where it ran is the command buffer it closed: `down` reads what
//! `gate` and `up` produced, and a CPU activation between them means those two
//! outputs are copied off the device, multiplied, and copied back as `down`'s
//! input. That is a submission, two readbacks and an allocation a bank, and
//! eighty banks a step.
//!
//! Encoded between them it is none of those. This reads what the pair wrote and
//! writes what `down` reads, so the batch puts a barrier either side of it — see
//! [`crate::ordering`] — which is what makes an activation a third dispatch in
//! the command buffer the pair already opened rather than a reason to close it.
//! The ordering is derived rather than free: the gate is bound to a slot this
//! kernel declares `device float *`, so nothing has to be asserted for it to
//! hold.
//!
//! **In place, into the gate's own buffer.** Each thread reads one element of
//! each and writes the one it read, so nothing here needs a fourth allocation
//! to put the answer in — and what `down` is then handed is the buffer `gate`
//! wrote, which is the same handover [`crate::LayerNorm::encode`] makes to the
//! projections that read it.

use inkling_core::profile::{self, Op};

use crate::buffer::Buffer;
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "swiglu";

/// The entry that runs two of those calls as one dispatch — see
/// [`SwiGlu::encode_pair`].
const PAIRED_ENTRY: &str = "swiglu_pair";

/// Threads one threadgroup of a dispatch holds. One thread to one element,
/// where the matmuls either side give a whole simdgroup to each — this reduces
/// over nothing.
const THREADS_PER_GROUP: usize = 256;

/// The compiled kernel, which every bank on a device shares.
///
/// Per source string rather than per bank, like [`crate::PackedMatmul`] and
/// [`crate::RmsNorm`]: the source names no shape, so one of these serves every
/// expert in the model.
#[derive(Debug)]
pub struct SwiGlu {
    kernel: Kernel,
    /// The same source's paired entry, compiled beside it because a model
    /// wanting one wants the other: every MoE layer has two banks to activate.
    kernel_pair: Kernel,
}

impl SwiGlu {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, BODY)
    }

    /// [`SwiGlu::new`] out of a source string of the caller's own, which is how
    /// a test puts a deliberately wrong kernel through the same plumbing as the
    /// right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
            kernel_pair: device.compile(source, PAIRED_ENTRY)?,
        })
    }

    /// `gate = silu(gate) * up`, encoded into `batch` over two buffers a
    /// dispatch already left on the device.
    ///
    /// Both are borrowed exclusively for the encoding and not for the batch,
    /// which is what [`crate::PackedBank::encode_over`] relies on too: Metal
    /// retains what is bound into a command buffer, so the binding outlives the
    /// borrow and the caller can hand `gate` on to the next dispatch.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        gate: &mut Buffer<f32>,
        up: &mut Buffer<f32>,
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        let elements = paired(gate, up);
        let fields = [extent(elements, "the elements of an activation")];
        let mut count = batch.device().inline(&fields)?;
        batch.add(
            &self.kernel,
            &[count.arg(), gate.arg(), up.arg()],
            Grid::new(elements, THREADS_PER_GROUP),
            moves(elements),
        )
    }

    /// **Two activations as one dispatch**, which is what a MoE layer's two
    /// banks are to each other.
    ///
    /// A layer's routed and shared SwiGLU read different rows of different
    /// widths and write into their own gates, and neither reads what the other
    /// writes — so what separated them was that the layer finished one bank
    /// before starting the other. See
    /// [`LayerExperts`](crate::LayerExperts), where both banks' pairs are now
    /// dispatched before either activation.
    ///
    /// **Nothing has to agree between the halves.** This kernel gives one thread
    /// to one element, declares no threadgroup memory and takes no barrier, so a
    /// thread does the work it did alone and the answer is the same bits.
    pub fn encode_pair(
        &self,
        batch: &mut Batch<'_>,
        first: (&mut Buffer<f32>, &mut Buffer<f32>),
        second: (&mut Buffer<f32>, &mut Buffer<f32>),
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        let (first_gate, first_up) = first;
        let (second_gate, second_up) = second;
        let (firsts, seconds) = (paired(first_gate, first_up), paired(second_gate, second_up));

        let fields = [
            extent(firsts, "the elements of an activation"),
            extent(seconds, "the elements of an activation"),
        ];
        let mut counts = batch.device().inline(&fields)?;
        batch.add(
            &self.kernel_pair,
            &[
                counts.arg(),
                first_gate.arg(),
                first_up.arg(),
                second_gate.arg(),
                second_up.arg(),
            ],
            Grid::new(firsts + seconds, THREADS_PER_GROUP),
            moves(firsts) + moves(seconds),
        )
    }
}

/// The elements one activation covers, refusing a pair whose two halves are not
/// one call's.
///
/// The lengths are checked here because the kernel cannot. `silu(gate) * up` is
/// elementwise, so an `up` shorter than `gate` would be read off the end of its
/// allocation — a GPU read of whatever is there rather than a fault — and one
/// longer would be silently truncated. That the two are one bank's pair is
/// [`ExpertBanks::new`](crate::ExpertBanks::new)'s to say; that the two calls
/// are one call's is this.
fn paired(gate: &Buffer<f32>, up: &Buffer<f32>) -> usize {
    assert_eq!(
        gate.len(),
        up.len(),
        "the gate against what gates it: {} values against {}",
        gate.len(),
        up.len()
    );
    gate.len()
}

/// The gate is read and written back where it was, so it crosses twice and an
/// activation is three passes over its elements rather than two.
fn moves(elements: usize) -> usize {
    3 * size_of::<f32>() * elements
}

/// The kernel.
///
/// `silu` is written as the one division [`inkling_core::ops::silu`] is, rather
/// than as `x * sigmoid(x)`: the two are the same value in exact arithmetic and
/// not in float32, and the CPU path is the oracle every kernel here is
/// validated against.
pub(crate) const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// `gate[i] = silu(gate[i]) * up[i]`, one element.
///
/// In place, which is safe for the reason an elementwise kernel always is: a
/// thread reads and writes one index and no thread reads another's.
///
/// Here rather than inside an entry point because two entry points run it, and
/// what a paired dispatch changes is which of two calls a thread is in and
/// nothing else.
static void activate(device float *gate, device const float *up, uint i) {
    const float x = gate[i];
    gate[i] = x / (1.0f + exp(-x)) * up[i];
}

/// One call of it, one thread to an element.
kernel void swiglu(
    constant uint &count [[buffer(0)]],
    device float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) {
        return;
    }
    activate(gate, up, i);
}

/// Two calls of it as one dispatch: a grid of both activations' threads, the
/// first call's at the front of it.
kernel void swiglu_pair(
    constant uint2 &counts [[buffer(0)]],
    device float *first_gate [[buffer(1)]],
    device const float *first_up [[buffer(2)]],
    device float *second_gate [[buffer(3)]],
    device const float *second_up [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < counts.x) {
        activate(first_gate, first_up, i);
    } else if (i - counts.x < counts.y) {
        activate(second_gate, second_up, i - counts.x);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::fixture::deviation;
    use inkling_core::ops::swiglu;

    use crate::testing::device;

    /// Wide enough that the dispatch spans several threadgroups, and not a
    /// multiple of the threadgroup size, so the tail group runs threads the
    /// bounds check has to turn away.
    const LEN: usize = 2048 * 3 + 17;

    /// Both sides compute one division and one multiply over the same f32s, so
    /// what separates them is `exp` — the GPU's and libm's, which agree to a few
    /// ulps rather than exactly. Worst observed when this landed: 6.0e-8.
    const TOLERANCE: f32 = 1e-6;

    /// Values spread over both signs and past the range where `silu` is
    /// straight, which is where the two `exp`s have anything to disagree about.
    fn values(seed: usize) -> Vec<f32> {
        (0..LEN)
            .map(|i| ((i * 37 + seed) % 211) as f32 / 8.0 - 13.0)
            .collect()
    }

    /// The whole claim: a dispatch is `inkling_core::ops::swiglu`, which is what
    /// the CPU path is pinned to mlx-vlm by.
    #[test]
    fn a_dispatch_activates_what_the_cpu_activates() {
        let Some(device) = device() else { return };
        let kernel = SwiGlu::new(&device).expect("the swiglu compiles");
        let (gate, up) = (values(0), values(97));

        let mut on_the_device = device.buffer(&gate).expect("the gate uploads");
        let mut up_buffer = device.buffer(&up).expect("the up uploads");
        let mut batch = device.batch().expect("a command buffer opens");
        kernel
            .encode(&mut batch, &mut on_the_device, &mut up_buffer)
            .expect("the dispatch encodes");
        batch.wait().expect("the batch completes");

        let mut want = gate;
        swiglu(&mut want, &up);
        let got = on_the_device.to_vec();
        let deviation = deviation(&got, &want);
        eprintln!("{LEN} elements activated: deviation {deviation:e}");
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
        assert!(
            got.iter().any(|y| *y < 0.0) && got.iter().any(|y| *y > 0.0),
            "an activation of one sign would not exercise silu's knee"
        );
    }

    /// What the bandwidth column divides by, against what the kernel reads:
    /// `gate` and `up` in, `gate` out over the top of itself.
    #[test]
    fn a_dispatch_declares_the_three_passes_it_makes() {
        let Some(device) = device() else { return };
        let kernel = SwiGlu::new(&device).expect("the swiglu compiles");
        let mut gate = device.buffer(&values(0)).expect("the gate uploads");
        let mut up = device.buffer(&values(97)).expect("the up uploads");

        let moved = crate::testing::moved(&device, |batch| {
            kernel
                .encode(batch, &mut gate, &mut up)
                .expect("the dispatch encodes")
        });

        assert_eq!(moved as usize, 3 * size_of::<f32>() * LEN);
        // **Three passes and not two.** The gate is read and then written back
        // over itself, which is the crossing an elementwise kernel writing into
        // one of its inputs is easy to charge once.
        assert!(
            moved as usize > 2 * size_of_val(gate.as_slice()),
            "the gate was charged for reading or for writing but not both"
        );
    }

    /// **A paired dispatch answers what the two dispatches it replaces answer,
    /// exactly** — the whole of what a merge is allowed to be, and a case a
    /// tolerance would be the wrong instrument for: both arms run the same
    /// kernel over the same floats, so anything but equality is a plumbing bug.
    ///
    /// The two halves are of different lengths, neither a multiple of the
    /// threadgroup, so the second half starts part way through a threadgroup the
    /// first half's tail is in — which is the one thing a merge over a shared
    /// grid can get wrong while still filling both buffers.
    #[test]
    fn a_paired_activation_answers_what_the_two_it_replaces_answer() {
        let Some(device) = device() else { return };
        let kernel = SwiGlu::new(&device).expect("the swiglu compiles");
        assert!(
            kernel.kernel_pair.max_threads_per_group() >= THREADS_PER_GROUP,
            "the paired entry cannot be dispatched in the threadgroup this kernel uses"
        );
        let (gate, up) = (values(0), values(97));
        let short = LEN / 3 + 5;

        let activated = |paired: bool| {
            let mut first_gate = device.buffer(&gate).expect("the gate uploads");
            let mut first_up = device.buffer(&up).expect("the up uploads");
            let mut second_gate = device.buffer(&gate[..short]).expect("the gate uploads");
            let mut second_up = device.buffer(&up[..short]).expect("the up uploads");
            let mut batch = device.batch().expect("a command buffer opens");
            match paired {
                true => kernel
                    .encode_pair(
                        &mut batch,
                        (&mut first_gate, &mut first_up),
                        (&mut second_gate, &mut second_up),
                    )
                    .expect("the pair encodes"),
                false => {
                    kernel
                        .encode(&mut batch, &mut first_gate, &mut first_up)
                        .expect("the first encodes");
                    kernel
                        .encode(&mut batch, &mut second_gate, &mut second_up)
                        .expect("the second encodes");
                }
            }
            batch.wait().expect("the batch completes");
            (first_gate.to_vec(), second_gate.to_vec())
        };

        let together = activated(true);
        assert_eq!(together, activated(false));
        assert!(
            together.0.iter().any(|y| *y < 0.0) && together.0.iter().any(|y| *y > 0.0),
            "an activation of one sign would not exercise silu's knee"
        );
    }

    /// A pair declares what its two halves declared apart, which is what the
    /// bandwidth column over a merged dispatch has to keep meaning.
    #[test]
    fn a_pair_declares_what_its_two_halves_declare() {
        let Some(device) = device() else { return };
        let kernel = SwiGlu::new(&device).expect("the swiglu compiles");
        let (gate, up) = (values(0), values(97));
        let short = LEN / 3;
        let mut first_gate = device.buffer(&gate).expect("the gate uploads");
        let mut first_up = device.buffer(&up).expect("the up uploads");
        let mut second_gate = device.buffer(&gate[..short]).expect("the gate uploads");
        let mut second_up = device.buffer(&up[..short]).expect("the up uploads");

        let moved = crate::testing::moved(&device, |batch| {
            kernel
                .encode_pair(
                    batch,
                    (&mut first_gate, &mut first_up),
                    (&mut second_gate, &mut second_up),
                )
                .expect("the pair encodes")
        });

        assert_eq!(moved as usize, 3 * size_of::<f32>() * (LEN + short));
    }

    /// `silu` goes on the gate and not on the up, which is the one thing a
    /// kernel over two buffers of the same width can get backwards while
    /// producing an answer of exactly the right shape.
    ///
    /// Stated as a kernel rather than as exchanged inputs, because that is where
    /// the mistake would live: the same dispatch, the same buffers, and the
    /// activation on the other one.
    #[test]
    fn activating_the_up_projection_instead_is_a_different_answer() {
        let Some(device) = device() else { return };
        let reversed = BODY
            .replace("const float x = gate[i];", "const float x = up[i];")
            .replace("* up[i];", "* gate[i];");
        assert_ne!(reversed, BODY, "the mutation changed nothing");
        let mutant = SwiGlu::from_source(&device, &reversed).expect("the mutant compiles");

        let (gate, up) = (values(0), values(97));
        let mut on_the_device = device.buffer(&gate).expect("the gate uploads");
        let mut up_buffer = device.buffer(&up).expect("the up uploads");
        let mut batch = device.batch().expect("a command buffer opens");
        mutant
            .encode(&mut batch, &mut on_the_device, &mut up_buffer)
            .expect("the dispatch encodes");
        batch.wait().expect("the batch completes");

        let mut want = gate;
        swiglu(&mut want, &up);
        let deviation = deviation(&on_the_device.to_vec(), &want);
        eprintln!("the activation on the other projection: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// Two activations of different lengths in one command buffer, which is
    /// what says the count belongs to the call rather than to the kernel — the
    /// case a prefill followed by decodes makes of every bank in the model.
    #[test]
    fn two_activations_in_one_batch_keep_their_own_counts() {
        let Some(device) = device() else { return };
        let kernel = SwiGlu::new(&device).expect("the swiglu compiles");
        let (gate, up) = (values(0), values(97));
        let short = LEN / 3;

        let mut tall = device.buffer(&gate).expect("the gate uploads");
        let mut tall_up = device.buffer(&up).expect("the up uploads");
        let mut low = device.buffer(&gate[..short]).expect("the gate uploads");
        let mut low_up = device.buffer(&up[..short]).expect("the up uploads");

        let mut batch = device.batch().expect("a command buffer opens");
        kernel
            .encode(&mut batch, &mut tall, &mut tall_up)
            .expect("the dispatch encodes");
        kernel
            .encode(&mut batch, &mut low, &mut low_up)
            .expect("the dispatch encodes");
        batch.wait().expect("the batch completes");

        let mut want = gate;
        swiglu(&mut want, &up);
        assert!(deviation(&tall.to_vec(), &want) <= TOLERANCE);
        assert!(
            deviation(&low.to_vec(), &want[..short]) <= TOLERANCE,
            "the taller call's count reached the shorter one"
        );
    }

    /// Two projections of widths that disagree are the mistake the check on the
    /// way in exists to catch: the kernel indexes both under one count, so the
    /// shorter would be read past the end of its allocation.
    #[test]
    #[should_panic(expected = "the gate against what gates it")]
    fn an_activation_over_two_widths_is_refused() {
        let Some(device) = device() else {
            panic!("the gate against what gates it: no device to ask")
        };
        let kernel = SwiGlu::new(&device).expect("the swiglu compiles");
        let mut gate = device.zeroed::<f32>(64).expect("the gate allocates");
        let mut up = device.zeroed::<f32>(32).expect("the up allocates");

        let mut batch = device.batch().expect("a command buffer opens");
        let _ = kernel.encode(&mut batch, &mut gate, &mut up);
    }
}
