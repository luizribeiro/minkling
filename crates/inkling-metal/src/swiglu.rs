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
//! Encoded between them it is none of those. Metal's default dispatch type is
//! serial, so a dispatch reads what the one before it wrote — which is what
//! makes an activation a third dispatch in the command buffer the pair already
//! opened rather than a reason to close it.
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
        })
    }

    /// `gate = silu(gate) * up`, encoded into `batch` over two buffers a
    /// dispatch already left on the device.
    ///
    /// Both are borrowed exclusively for the encoding and not for the batch,
    /// which is what [`crate::PackedBank::encode_over`] relies on too: Metal
    /// retains what is bound into a command buffer, so the binding outlives the
    /// borrow and the caller can hand `gate` on to the next dispatch.
    ///
    /// The lengths are checked here because the kernel cannot. `silu(gate) * up`
    /// is elementwise, so an `up` shorter than `gate` would be read off the end
    /// of its allocation — a GPU read of whatever is there rather than a fault —
    /// and one longer would be silently truncated. That the two are one bank's
    /// pair is [`ExpertBanks::new`](crate::ExpertBanks::new)'s to say; that the
    /// two calls are one call's is this.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        gate: &mut Buffer<f32>,
        up: &mut Buffer<f32>,
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        assert_eq!(
            gate.len(),
            up.len(),
            "the gate against what gates it: {} values against {}",
            gate.len(),
            up.len()
        );

        let elements = gate.len();
        let fields = [extent(elements, "the elements of an activation")];
        let mut count = batch.device().inline(&fields)?;
        // The gate is read and written back where it was, so it crosses twice
        // and the pair is three passes over `elements` rather than two.
        batch.add(
            &self.kernel,
            &[count.arg(), gate.arg(), up.arg()],
            Grid::new(elements, THREADS_PER_GROUP),
            3 * size_of::<f32>() * elements,
        )
    }
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

/// `gate[i] = silu(gate[i]) * up[i]`, one thread to an element.
///
/// In place, which is safe for the reason an elementwise kernel always is: a
/// thread reads and writes one index and no thread reads another's.
kernel void swiglu(
    constant uint &count [[buffer(0)]],
    device float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= count) {
        return;
    }
    const float x = gate[i];
    gate[i] = x / (1.0f + exp(-x)) * up[i];
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
