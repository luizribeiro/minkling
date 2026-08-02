//! `out = x @ wᵀ` against a weight the quantiser left in bfloat16.
//!
//! [`crate::matmul`] is every weight in the model but one. The quant packs each
//! projection, each expert and the head into MXFP4 and leaves the routers'
//! `[258, 4096]` gates alone — 2.1 MB a layer, which is 0.06% of the checkpoint
//! and the last matmul in the model with no kernel of its own.
//!
//! **Dense means every value is stored, not that the layer is dense.** What
//! separates this kernel from the packed one is the format it reads and nothing
//! else: no codes, no block scales, no gather. A row of the weight is a run of
//! bfloat16 values and the multiply is a dot product over it.
//!
//! **Widening happens in registers, which is the same bargain.** A bfloat16 is
//! an f32 with the low sixteen mantissa bits dropped, so a thread shifts two
//! bytes into place and multiplies — the tensor is never widened into memory
//! anywhere, on the device or off it. That matters here for a reason the packed
//! matmul's is a scaled-up version of: the CPU path widens each gate on every
//! layer of every step, and holding all forty widened instead is 169 MB of
//! float32 against 84 MB of bfloat16 nobody has to hold at all.
//!
//! **A byte at a time, and here that is forced rather than convenient.** The
//! quant's shard headers are not padded, and every bfloat16 tensor in the shard
//! this reads begins at an *odd* byte — so a `device const ushort *` cannot be
//! pointed at one at all, and [`Device::wrap`](crate::Device::wrap) refuses to
//! promise two-byte elements over those pages. Two `uchar` loads shifted
//! together read the same value wherever it starts, which is what lets the GPU
//! be handed the gate where the checkpoint mapped it.
//!
//! **A run of simdgroups per output element**, where [`crate::matmul`] gives
//! each one a simdgroup: lane `l` walks its weight row from value
//! `l * VALUES_PER_LANE` in strides of that many times the run's width and the
//! run sums what its lanes held, so 32 lanes of one reduction read 256
//! consecutive bytes and a run of eight reads 2048. How long the run is comes
//! from how many elements the dispatch has to spread over the machine — see
//! [`SIMDGROUPS_IN_FLIGHT`] — and how much of a row one lane holds at a time
//! from [`VALUES_PER_LANE`].

use std::cell::RefCell;

use inkling_core::checkpoint::{BF16_BYTES, BF16_SHIFT};
use inkling_core::ops::Projection;
use inkling_core::profile::{self, Op};
use inkling_core::weights::Bf16;

use crate::buffer::{Buffer, Bytes};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};
use crate::matmul::{MatmulError, Pending};

const ENTRY: &str = "dense_matmul";

/// Threads one threadgroup of a dispatch holds, and a multiple of every
/// simdgroup width Metal reports — which is what lets the kernel take its
/// output element from where its simdgroup sits inside the threadgroup.
const THREADS_PER_GROUP: usize = 256;

/// Entries the kernel's threadgroup array holds, for [`crate::RmsNorm`]'s
/// reason: 1024 threads is the widest threadgroup any Apple GPU allows and 32
/// the narrowest simdgroup any reports, so 32 partials is the most a threadgroup
/// can produce.
const MOST_SIMDGROUPS: usize = 32;

/// Simdgroups a dispatch wants in flight before it stops widening the reduction
/// over one output element.
///
/// A count of simdgroups and not of rows, so it is the same number whatever the
/// weight's shape — though the only weight this kernel has is the `[258, 4096]`
/// gate, which is what the sweep it was chosen from measures.
///
/// **A decode-shaped gate is 258 output elements, and one simdgroup each is a
/// lane walking 128 strides of two bytes.** That is `rms_norm`'s finding at a
/// different width: nothing here is waiting on bandwidth, it is waiting on the
/// requests one thread has outstanding, and the fix is to give the reduction
/// more threads rather than the machine more bytes. Eight simdgroups to an
/// element takes a decode-shaped gate from 39 microseconds to 10.
///
/// **And it has to be the dispatch that decides.** The same widening at prefill
/// makes the kernel twice as slow — 385 rows are 99330 elements, which fill the
/// machine at one simdgroup apiece, and eight of them is eight times the
/// reduction overhead for work that was never idle. So the count comes from how
/// many elements there are: this many simdgroups between them, and each element
/// takes what is left over. Measured across rows 1 to 385 and one to eight
/// simdgroups, it picks the best run at a decode step's one row and at a
/// prefill's hundreds, and lands within a sixth of the best at every row count
/// between — `what_a_gate_costs_the_device_at_each_reduction_width` is the
/// sweep, and the row counts in between are not shapes this engine dispatches.
const SIMDGROUPS_IN_FLIGHT: usize = 16512;

/// Values of a weight row one lane reads before the next lane's, which is how
/// much of that row one thread has in flight at a time.
///
/// **A value read one at a time is two byte loads and an input float for one
/// multiply-add**, and this kernel was reaching 171 GB/s against the packed
/// matmul's 414 on the same chain of heads — which for a multiply that streams a
/// weight once is a kernel waiting on its own requests rather than on memory.
/// Four values a lane issue those same loads as twelve independent ones, because
/// the trip count is a compile-time constant and the four inputs are consecutive
/// floats. That is [`SIMDGROUPS_IN_FLIGHT`]'s finding on the other axis: more
/// memory in flight per thread, no extra dispatch, and nothing about the grid.
///
/// **What it does move is summation order.** A lane holds a run of consecutive
/// values where it held every `width`th one, so the products enter its sum in a
/// different order — exactly as [`crate::matmul`]'s own width did. Every product
/// is the same float32 either way; what a caller has to check is the tokens, and
/// that is checked rather than argued.
///
/// A lane's last chunk may run past a row that is not whole chunks, which the
/// kernel finishes one value at a time rather than the caller refusing the
/// width. Every weight this engine dispatches is a multiple of 4096 wide, so
/// nothing here reaches that tail; what it buys is that the sweep below can ask
/// the kernel for any width against any shape.
const VALUES_PER_LANE: usize = 4;

/// The compiled kernel, which every bfloat16 weight on a device shares.
///
/// Per source string rather than per weight, like [`crate::PackedMatmul`] and
/// [`crate::RmsNorm`]: the source names no shape, so one of these serves every
/// router in the model.
#[derive(Debug)]
pub struct DenseMatmul {
    kernel: Kernel,
}

impl DenseMatmul {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`DenseMatmul::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }
}

/// One `[out_dim, in_dim]` bfloat16 weight on the device, and the multiply
/// against it.
#[derive(Debug)]
pub struct DenseWeight<'a> {
    device: &'a Device,
    matmul: &'a DenseMatmul,
    in_dim: usize,
    out_dim: usize,
    /// Rows of the tensor one row of this weight steps over, and where its
    /// first row starts in bytes — 1 and 0 for a weight that is a tensor, and
    /// what makes two weights out of one for a tensor that holds two.
    stride: usize,
    first: usize,
    /// Held behind a cell for the reason [`crate::PackedBank`]'s resident
    /// tensors are: binding a buffer to a dispatch borrows it exclusively, and
    /// the weight belongs to this rather than to a call.
    resident: RefCell<Bytes<'a>>,
}

impl<'a> DenseWeight<'a> {
    /// Copy an `[out_dim, in_dim]` bfloat16 weight onto the device, for bytes
    /// that are not a mapping's — a test's synthetic weight, and anything a
    /// future path builds rather than reads.
    pub fn upload(
        device: &'a Device,
        matmul: &'a DenseMatmul,
        in_dim: usize,
        out_dim: usize,
        bytes: &[u8],
    ) -> Result<Self, MatmulError> {
        pairs(in_dim, out_dim, bytes.len())?;
        Self::over(
            device,
            matmul,
            in_dim,
            out_dim,
            Bytes::Copied(device.buffer(bytes)?),
        )
    }

    /// A checkpoint's own bfloat16 weight, read where it is mapped.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a DenseMatmul,
        weight: &Bf16<'a>,
    ) -> Result<Self, MatmulError> {
        Self::wrap_rows(device, matmul, weight, 0, 1)
    }

    /// Every `stride`th row of a checkpoint's weight from `first`, read where
    /// it is mapped.
    ///
    /// **One tensor is two weights where the checkpoint fused them.** An MTP
    /// head's `w13_dn` is its SwiGLU's gate and up interleaved row by row, so
    /// the two projections a multiply wants are its even rows and its odd ones
    /// — and a stride is what lets both be read where they are rather than
    /// de-interleaved into 268 MB of copies a head.
    pub fn wrap_rows(
        device: &'a Device,
        matmul: &'a DenseMatmul,
        weight: &Bf16<'a>,
        first: usize,
        stride: usize,
    ) -> Result<Self, MatmulError> {
        assert!(stride > 0, "a weight's rows step over its own");
        assert!(first < stride, "row {first} of every {stride}");
        let (in_dim, rows) = (weight.in_dim(), weight.out_dim());
        pairs(in_dim, rows, weight.bytes().len())?;
        // SAFETY: the bytes are a `Checkpoint`'s mapping, which outlives this by
        // the lifetime they carry and which nothing writes — the assumption that
        // module already maps under.
        let mapped = unsafe { device.wrap(weight.bytes())? };
        Self::over(
            device,
            matmul,
            in_dim,
            (rows - first).div_ceil(stride),
            Bytes::Mapped(mapped),
        )
        .map(|held| Self {
            first,
            stride,
            ..held
        })
    }

    fn over(
        device: &'a Device,
        matmul: &'a DenseMatmul,
        in_dim: usize,
        out_dim: usize,
        resident: Bytes<'a>,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            resident: RefCell::new(resident),
            device,
            matmul,
            in_dim,
            out_dim,
            stride: 1,
            first: 0,
        })
    }

    /// The width a row of the input has to be.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// The width a row of the output is.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// The device this multiplies on, for a caller batching several of these
    /// into one command buffer.
    pub fn device(&self) -> &'a Device {
        self.device
    }

    /// `[rows, in_dim]` in, `[rows, out_dim]` out, submitted on its own.
    ///
    /// What a caller with nothing to batch it against wants, and what the cases
    /// here drive. The router has something to batch it against — see
    /// [`crate::LayerExperts`] — because a submission costs 225 microseconds
    /// and a `[1, 4096] @ [258, 4096]ᵀ` multiply does not.
    pub fn multiply(&self, x: &[f32]) -> Result<Vec<f32>, MatmulError> {
        let mut batch = self.device.batch()?;
        let pending = self.encode(&mut batch, x)?;
        batch.wait()?;
        Ok(pending.take())
    }

    /// The same multiply, encoded into `batch` rather than submitted on its own.
    ///
    /// A call over no rows is the device's refusal of a zero-length allocation
    /// rather than an empty answer, which is where this parts company with
    /// [`PackedBank::encode`](crate::PackedBank::encode) and sides with
    /// [`LayerNorm::encode`](crate::LayerNorm::encode) — for the same reason
    /// that one gives. A gather of no rows is an ordinary step of the router's,
    /// because a bank nobody chose is a thing that happens; the *gate* reads
    /// every token there is, so a gate over no rows is a forward pass over no
    /// tokens.
    pub fn encode(&self, batch: &mut Batch<'_>, x: &[f32]) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        let mut input = self.device.buffer(x)?;
        self.encoding(batch, &mut input)
    }

    /// The same multiply over rows a dispatch already left on the device — see
    /// [`PackedBank::encode_over`](crate::PackedBank::encode_over).
    ///
    /// The gate is the first thing a MoE layer's command buffer holds and the
    /// hidden state it reads is what both banks read, so uploading it once and
    /// binding the same buffer three times is what a layer wants of this.
    pub fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        self.encoding(batch, x)
    }

    /// One dispatch encoded, without the scope its two callers each open — so
    /// that the profile counts a dispatch once however it was reached.
    fn encoding(&self, batch: &mut Batch<'_>, x: &mut Buffer<f32>) -> Result<Pending, MatmulError> {
        assert_eq!(
            x.len() % self.in_dim,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.in_dim
        );

        let rows = x.len() / self.in_dim;
        let kernel = &self.matmul.kernel;
        let elements = rows * self.out_dim;
        let simdgroups = simdgroups_an_element(elements, THREADS_PER_GROUP / kernel.simd_width());
        let fields = self.shape(rows, simdgroups);
        let mut shape = self.device.inline(&fields)?;
        let mut resident = self.resident.borrow_mut();
        let mut out = self.device.zeroed::<f32>(rows * self.out_dim)?;

        let grid = Grid::new(
            elements * simdgroups * kernel.simd_width(),
            THREADS_PER_GROUP,
        );
        let moves = self.in_dim * self.out_dim * BF16_BYTES
            + size_of::<f32>() * (x.len() + rows * self.out_dim);

        batch.add(
            kernel,
            &[shape.arg(), x.arg(), resident.arg(), out.arg()],
            grid,
            moves,
        )?;
        Ok(Pending::holding(out))
    }

    /// The scalars the kernel's `Shape` struct declares, in its order — of this
    /// call's own, which is what two multiplies of different heights sharing a
    /// command buffer needs them to be.
    fn shape(&self, rows: usize, simdgroups: usize) -> [u32; SHAPE_FIELDS] {
        let first = self.first * self.in_dim * BF16_BYTES;
        [
            extent(rows, "the rows of a call"),
            extent(self.in_dim, "the width a weight maps from"),
            extent(self.out_dim, "the width a weight maps to"),
            extent(
                self.resident.borrow().offset() + first,
                "where a weight starts",
            ),
            extent(self.stride, "the tensor's rows a row of this steps over"),
            extent(simdgroups, "the simdgroups over one output element"),
        ]
    }
}

/// The same multiply as [`crate::PackedProjection`]'s, over the format the
/// quantiser left alone — so that a caller holding a projection cannot tell
/// which of the two answered.
impl Projection for DenseWeight<'_> {
    fn in_dim(&self) -> usize {
        self.in_dim
    }

    fn out_dim(&self) -> usize {
        self.out_dim
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.multiply(x)
            .unwrap_or_else(|err| panic!("the dense matmul did not run: {err}"))
    }
}

/// How many of a threadgroup's simdgroups reduce one output element, given how
/// many elements the whole dispatch has to spread over the machine.
///
/// **A run has to divide the threadgroup's simdgroups**, because the kernel
/// takes an element from where its run sits inside the threadgroup and a
/// threadgroup that did not cut into whole runs would put two elements' lanes in
/// one run. So the ceiling is not `most` but the largest power of two dividing
/// it — the same number for the 8 every Apple GPU's simdgroup width and this
/// threadgroup produce, and a number rather than an assumption for a device that
/// reported a width making them anything else.
///
/// See [`SIMDGROUPS_IN_FLIGHT`] for where the count it is weighed against comes
/// from.
fn simdgroups_an_element(elements: usize, most: usize) -> usize {
    let whole = 1 << most.trailing_zeros();
    let wanted = (SIMDGROUPS_IN_FLIGHT / elements.max(1)).clamp(1, whole);
    match wanted.is_power_of_two() {
        true => wanted,
        false => wanted.next_power_of_two() / 2,
    }
}

/// Whether `bytes` are an `[out_dim, in_dim]` bfloat16 weight, which has to be
/// settled on the way in: the kernel takes its bounds from the shape it was
/// told and would read off the end of a tensor that was short.
fn pairs(in_dim: usize, out_dim: usize, bytes: usize) -> Result<(), MatmulError> {
    let expected = in_dim * out_dim * BF16_BYTES;
    match bytes == expected {
        true => Ok(()),
        false => Err(MatmulError::WrongWeightLen {
            in_dim,
            out_dim,
            expected,
            got: bytes,
        }),
    }
}

/// How many `uint`s the kernel's `Shape` struct declares.
const SHAPE_FIELDS: usize = 6;

/// The kernel, with the format's two facts written into its prelude.
///
/// Generated rather than spelled out because [`inkling_core::checkpoint`] is
/// the authority on both — how wide a bfloat16 is, and where its bits sit in
/// the float32 it widens to — and a second copy of them living in a source
/// string is a copy that can drift from the widening the CPU path is pinned by.
pub(crate) fn source() -> String {
    format!(
        "constant uint BF16_BYTES = {BF16_BYTES};\n\
         constant uint BF16_SHIFT = {BF16_SHIFT};\n\
         constant uint MOST_SIMDGROUPS = {MOST_SIMDGROUPS};\n\
         constant uint VALUES_PER_LANE = {VALUES_PER_LANE};\n{BODY}"
    )
}

/// Everything of the kernel that the format does not decide.
pub(crate) const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint rows;
    uint in_dim;
    uint out_dim;
    uint base;
    uint stride;
    uint simdgroups;
};

/// One value of a weight, from the two bytes that hold it.
///
/// A value is two bytes little-endian, which are the top sixteen bits of the
/// float32 it stands for — so widening is a shift, exact, and the low mantissa
/// bits it puts back are zeros.
inline float widened(device const uchar *pair) {
    const uint low = pair[0];
    const uint high = pair[1];
    return as_type<float>(((high << 8) | low) << BF16_SHIFT);
}

/// One output element: a thread walks its weight row from value
/// `lane * VALUES_PER_LANE` in strides of that many times `width`, and the
/// caller reduces what the threads held. The pair is where a thread sits in the
/// run over this element and how wide that run is, which is a simdgroup's own
/// width only when the run is one simdgroup.
///
/// The inner loop's trip count is a compile-time constant, so it is unrolled and
/// its loads have no dependency on each other — which is the whole of what a
/// chunk buys, and the same thing `crate::matmul`'s own width buys there.
///
/// **The second loop is the row that is not whole chunks**, which the first
/// stops one chunk short of. Only the lane whose chunk straddles the end reaches
/// it — the chunks are disjoint and a lane that has passed the row does nothing
/// — so every value of the row is read exactly once whatever the width.
inline float weight_dot(
    device const uchar *weight,
    device const float *values,
    uint in_dim,
    uint lane,
    uint width
) {
    float sum = 0.0f;
    uint i = lane * VALUES_PER_LANE;
    for (; i + VALUES_PER_LANE <= in_dim; i += width * VALUES_PER_LANE) {
        float dot = 0.0f;
        for (uint j = 0; j < VALUES_PER_LANE; ++j) {
            dot += widened(weight + (i + j) * BF16_BYTES) * values[i + j];
        }
        sum += dot;
    }
    for (; i < in_dim; ++i) {
        sum += widened(weight + i * BF16_BYTES) * values[i];
    }
    return sum;
}

/// `out[i] = x[i] @ w^T` over an `[out_dim, in_dim]` bfloat16 weight.
///
/// **`shape.simdgroups` of a threadgroup's simdgroups reduce one output
/// element**, and the caller chooses that from how many elements the dispatch
/// has — see `simdgroups_an_element`. At one it is a simdgroup an element and a
/// threadgroup holds several, which is what a prefill wants; at the threadgroup's
/// whole width it is one element reduced by every thread, which is what a decode
/// step wants of the same weight.
///
/// The two are one body because the run of simdgroups over an element is where
/// they differ and nothing else is: `slot` is which element of this threadgroup
/// a simdgroup works on, and a run of one is the case where that is its own
/// index.
kernel void dense_matmul(
    constant Shape &shape [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uchar *weight [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint width [[threads_per_simdgroup]]
) {
    threadgroup float partials[MOST_SIMDGROUPS];

    const uint slot = simd / shape.simdgroups;
    const uint element = group * (simds / shape.simdgroups) + slot;
    // Not a return, because the barrier below is reached by the whole
    // threadgroup and the tail element of a dispatch is one simdgroup's own
    // rather than the threadgroup's. An out-of-range simdgroup reads nothing,
    // contributes a zero and waits where the others wait.
    const bool live = element < shape.rows * shape.out_dim;

    const uint row = element / shape.out_dim;
    const uint col = element % shape.out_dim;
    // The threads over one element are the `shape.simdgroups` of them starting
    // at the run's own, so a lane walks the row from where it sits in the run.
    const uint reach = shape.simdgroups * width;
    const uint start = (simd % shape.simdgroups) * width + lane;

    // `stride` is how many of the tensor's rows one row of *this* weight steps
    // over, which is 1 for every weight but the two an MTP head's `w13_dn`
    // holds interleaved — see `DenseWeight::wrap_rows`. Where the weight starts
    // is `base`, which carries the first row too.
    float sum = live ? weight_dot(
        weight + shape.base + (ulong)col * shape.stride * shape.in_dim * BF16_BYTES,
        x + (ulong)row * shape.in_dim,
        shape.in_dim,
        start,
        reach
    ) : 0.0f;
    sum = simd_sum(sum);

    // A run of one is the whole reduction and has nothing to wait for. `shape`
    // is the dispatch's rather than a thread's, so every threadgroup takes this
    // branch or none does, which is what makes the barrier below reachable by a
    // whole threadgroup or by none of it.
    if (shape.simdgroups == 1) {
        if (live && lane == 0) {
            out[element] = sum;
        }
        return;
    }

    if (lane == 0) {
        partials[simd] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && local == slot * reach) {
        float total = 0.0f;
        for (uint s = 0; s < shape.simdgroups; ++s) {
            total += partials[slot * shape.simdgroups + s];
        }
        out[element] = total;
    }
}
"#;

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A weight held as the checkpoint holds one: bfloat16, little-endian, row
    /// after row.
    ///
    /// Written from f32s by dropping the low mantissa bits rather than by
    /// rounding, which is what makes [`widened`] its exact inverse — so a case
    /// built this way has an oracle that no rounding rule stands between.
    pub(crate) fn narrowed(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> BF16_SHIFT) as u16).to_le_bytes())
            .collect()
    }

    /// The same bytes as float32, which is what the CPU multiplies against.
    pub(crate) fn widened(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(BF16_BYTES)
            .map(|pair| {
                f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << BF16_SHIFT)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::testing::{narrowed, widened};
    use super::*;
    use inkling_core::fixture::deviation;
    use inkling_core::ops::linear;

    use crate::matmul::testing::Noise;
    use crate::testing::{device, drift};

    /// The reduction the router's gate is: `[258, 4096]`.
    const IN_DIM: usize = 4096;

    /// The routed experts plus the two shared ones, which is the gate's own
    /// row count and is not a multiple of the threadgroup — so the tail group
    /// runs lanes past the end of the work and the bounds check is what stops
    /// them writing.
    const OUT_DIM: usize = 258;

    /// How far a dispatch may land from the CPU's answer for the same weights.
    ///
    /// The same account as `matmul::tests::TOLERANCE`, and for the same reason:
    /// both sides read exactly the same float32 values — widening bfloat16 is a
    /// shift on either — so summation order is the whole of what separates
    /// them. The CPU adds 4096 products serially and the kernel sums 128 a lane
    /// and reduces 32 lanes in a tree, which is the better-conditioned order of
    /// the two.
    const TOLERANCE: f32 = 6e-6;

    /// One multiply: an `[out_dim, in_dim]` bfloat16 weight, and the rows of
    /// `x` to put through it.
    struct Case {
        in_dim: usize,
        out_dim: usize,
        weight: Vec<u8>,
        x: Vec<f32>,
    }

    impl Case {
        /// Values of mixed sign, which is what makes the reduction cancel the
        /// way a trained one does and so what makes two summation orders part
        /// company at all.
        fn noisy(in_dim: usize, out_dim: usize, rows: usize) -> Self {
            let mut noise = Noise(0x9e37_79b9);
            let values: Vec<f32> = (0..out_dim * in_dim).map(|_| noise.signed()).collect();
            Self {
                weight: narrowed(&values),
                x: (0..rows * in_dim).map(|_| noise.signed()).collect(),
                in_dim,
                out_dim,
            }
        }

        /// A weight of exact dyadics, two rows carrying the same values in
        /// opposite order, beside the values themselves.
        ///
        /// Two tests want different things of it. The arithmetic is exact —
        /// every value is a dyadic of three significant bits and the inputs are
        /// small integers — so the answer can be checked bit for bit. And every
        /// value's bfloat16 pair stays a finite float when it is read the other
        /// way round, which is what lets *that* be measured at all: a 4096-wide
        /// row of noise read backwards overflows to NaN, and a tensor with no
        /// numbers in it has no deviation to compare against a tolerance.
        fn dyadic() -> (Self, [Vec<f32>; 2]) {
            let forwards: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) / 4.0).collect();
            let backwards: Vec<f32> = forwards.iter().rev().copied().collect();
            let rows = [forwards, backwards];
            (
                Self {
                    in_dim: rows[0].len(),
                    out_dim: rows.len(),
                    weight: narrowed(&rows.concat()),
                    x: (0..rows[0].len()).map(|i| i as f32 + 1.0).collect(),
                },
                rows,
            )
        }

        fn upload<'a>(&self, device: &'a Device, matmul: &'a DenseMatmul) -> DenseWeight<'a> {
            DenseWeight::upload(device, matmul, self.in_dim, self.out_dim, &self.weight)
                .expect("the case's shapes pair")
        }

        /// What the bandwidth column divides by, against what the kernel reads:
        /// the one weight, still bfloat16 where it lies, the rows in and the
        /// rows out.
        fn moves(&self, rows: usize) -> usize {
            self.in_dim * self.out_dim * BF16_BYTES
                + size_of::<f32>() * rows * (self.in_dim + self.out_dim)
        }

        /// The same multiply against the widened weight, through the `linear`
        /// the CPU path runs — which is the oracle for everything below.
        fn on_the_cpu(&self) -> Vec<f32> {
            linear(&self.x, &widened(&self.weight), self.in_dim)
        }

        /// The same multiply summed in f64, which neither side does.
        ///
        /// Widening is exact on both sides, so the products are the same f32s
        /// either way and summation order is the only thing left to differ
        /// about. Accumulating them with 29 bits of headroom settles which of
        /// the two orders is drifting.
        fn exactly(&self) -> Vec<f64> {
            let weight = widened(&self.weight);
            let mut out = Vec::new();
            for x in self.x.chunks_exact(self.in_dim) {
                out.extend(weight.chunks_exact(self.in_dim).map(|row| {
                    x.iter()
                        .zip(row)
                        .map(|(x, w)| f64::from(*x) * f64::from(*w))
                        .sum::<f64>()
                }));
            }
            out
        }
    }

    fn matmul(device: &Device) -> DenseMatmul {
        DenseMatmul::new(device).expect("the dense matmul compiles")
    }

    /// The smallest claim there is: a bfloat16 value times an input is what
    /// lands in the output.
    ///
    /// Exact rather than bounded, and that is affordable rather than lucky —
    /// every value here is a dyadic of a few significant bits and every partial
    /// sum is representable, so no ordering can move a bit. A tolerance would
    /// only be hiding a plumbing mistake.
    ///
    /// The two rows carry the same values in opposite order, so a kernel that
    /// indexed a row from the wrong end produces the other row's answer rather
    /// than a near miss.
    #[test]
    fn a_dispatch_multiplies_what_the_bfloat16_values_stand_for() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let (case, rows) = Case::dyadic();

        let want: Vec<f32> = rows
            .iter()
            .map(|row| row.iter().zip(&case.x).map(|(w, x)| w * x).sum())
            .collect();
        assert_ne!(want[0], want[1], "two rows that agreed would prove nothing");

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(got, want);
    }

    /// The kernel against the CPU it replaces, at the gate's own shape.
    #[test]
    fn a_dispatch_declares_the_bfloat16_weight_it_reads_where_it_lies() {
        let Some(device) = device() else { return };
        let matmul = DenseMatmul::new(&device).expect("the dense matmul compiles");
        const ROWS: usize = 2;
        let case = Case::noisy(64, 8, ROWS);
        let weight = case.upload(&device, &matmul);

        let moved = crate::testing::moved(&device, |batch| {
            weight.encode(batch, &case.x).expect("the dispatch encodes");
        });

        assert_eq!(moved as usize, case.moves(ROWS));
        assert!(
            (moved as usize) < case.in_dim * case.out_dim * size_of::<f32>(),
            "a widened weight was charged for one the kernel reads two bytes at a time"
        );
    }

    /// What the bandwidth column divides by, against what the kernel reads. The
    /// gate is the one weight in the model the quantiser left in bfloat16, and
    /// it is read as the two bytes it is rather than widened first.
    #[test]
    fn the_kernel_reproduces_the_cpu_at_the_gates_shape() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 3);
        let weight = case.upload(&device, &matmul);
        assert_eq!(weight.in_dim(), IN_DIM);
        assert_eq!(weight.out_dim(), OUT_DIM);

        let got = weight.multiply(&case.x).expect("the dispatch completes");
        let on_the_cpu = case.on_the_cpu();
        let deviation = deviation(&got, &on_the_cpu);
        assert!(
            deviation > 0.0,
            "an exact match would mean the two are not summing independently"
        );

        // Which of the two is drifting, which is what says whether a
        // disagreement of this size is float noise or a bug — and is the
        // assertion with teeth, since a widened tolerance would let a kernel
        // sitting further out than a serial f32 loop through.
        let exact = case.exactly();
        let (mine, theirs) = (drift(&got, &exact), drift(&on_the_cpu, &exact));
        eprintln!(
            "[3, {IN_DIM}] @ [{OUT_DIM}, {IN_DIM}]^T: deviation {deviation:e}, drift from exact \
             {mine:e} against the CPU's {theirs:e}"
        );
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
        assert!(mine < theirs, "{mine:e} against the CPU's {theirs:e}");
    }

    /// **A dispatch chooses how many simdgroups reduce one output element**, and
    /// the ends of that choice index differently enough to be two kernels: a run
    /// of one is a simdgroup an element with several elements to a threadgroup
    /// and no barrier at all, and a run of the threadgroup's whole width is one
    /// element reduced by every thread of it. Every run between is a threadgroup
    /// holding several elements *and* reducing each of them across simdgroups,
    /// which is the case neither end reaches. All of them have to answer what
    /// the CPU answers.
    ///
    /// The row counts are chosen for the run they select rather than for their
    /// shape — asserted, so that a different [`SIMDGROUPS_IN_FLIGHT`] makes this
    /// fail rather than quietly measure one run twice. The shapes the engine
    /// dispatches are elsewhere: a decode step's single row is
    /// `the_kernel_reproduces_the_cpu_at_the_gates_shape`'s, and a prefill of any
    /// length reaches the shortest run this covers.
    #[test]
    fn every_reduction_width_answers_what_the_cpu_answers() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let most = THREADS_PER_GROUP / matmul.kernel.simd_width();
        // Narrow enough that the oracle over sixty-five rows is not the test.
        const NARROW: usize = 64;
        let rows_for = |run: usize| match run {
            // Past the boundary rather than on it, so the shortest run is also
            // the ragged case: a threadgroup holds several elements there, and
            // the element count has to not be a whole number of them.
            1 => SIMDGROUPS_IN_FLIGHT / OUT_DIM + 1,
            _ => SIMDGROUPS_IN_FLIGHT / (run * OUT_DIM),
        };

        let mut runs = Vec::new();
        for run in (0..).map(|i| 1usize << i).take_while(|run| *run <= most) {
            let rows = rows_for(run);
            let elements = rows * OUT_DIM;
            assert_eq!(
                simdgroups_an_element(elements, most),
                run,
                "{rows} rows did not reach a run of {run}"
            );
            let case = Case::noisy(NARROW, OUT_DIM, rows);
            let got = case
                .upload(&device, &matmul)
                .multiply(&case.x)
                .expect("the dispatch completes");

            let deviation = deviation(&got, &case.on_the_cpu());
            eprintln!("a run of {run} over {rows} rows: deviation {deviation:e}");
            assert!(
                deviation <= TOLERANCE,
                "a run of {run}: deviation {deviation:e}"
            );
            runs.push((run, elements));
        }
        assert!(runs.len() > 2, "the runs between the two ends went untried");

        // A threadgroup can only hold a partial element where it holds more than
        // one, so the shortest run is the only one whose bounds check does
        // anything — and the row count above has to be one that leaves a tail.
        let (shortest, elements) = runs[0];
        assert_eq!(shortest, 1);
        assert_ne!(elements % most, 0, "the tail threadgroup was full");
    }

    /// The rule the choice above is made by, at the shapes it is made for and at
    /// the two ends it has to be defensive about.
    ///
    /// A run has to divide the threadgroup's simdgroups, so it is a power of two
    /// and never wider than they are; and a dispatch of more elements than the
    /// machine wants in flight still has to reduce each of them with something.
    /// The counts include ones no power of two divides, which is what a device
    /// reporting an unusual simdgroup width would produce.
    #[test]
    fn a_run_of_simdgroups_divides_the_threadgroup_it_is_cut_from() {
        for most in [1usize, 4, 6, 8, 12, 32] {
            for elements in [0usize, 1, 258, 774, 3000, 16512, 99330, usize::MAX] {
                let run = simdgroups_an_element(elements, most);
                assert!(run >= 1 && run <= most, "{elements} of {most}: {run}");
                assert_eq!(
                    most % run,
                    0,
                    "{elements} of {most}: {run} is a partial run"
                );
            }
        }
        // A decode step's gate takes the whole threadgroup and a prefill's takes
        // a simdgroup, which is the finding the rule exists to act on.
        assert_eq!(simdgroups_an_element(OUT_DIM, 8), 8);
        assert_eq!(simdgroups_an_element(97 * OUT_DIM, 8), 1);
    }

    /// **What a gate costs the device at each width it is dispatched over**, and
    /// the sweep [`SIMDGROUPS_IN_FLIGHT`] was chosen from.
    ///
    /// The row counts run from a decode step's one to a prefill's, and each is
    /// dispatched at every run of simdgroups a threadgroup can be cut into — so
    /// what the table shows is both the best run at each shape and how far the
    /// rule's choice lands from it. Nothing asserts a duration; the numbers go
    /// to stderr for the commit message to quote.
    ///
    /// Read off the device's own clock over a command buffer of `CALLS`
    /// dispatches, for `norm`'s reason: a submission is 225 microseconds and
    /// most of these are under a hundred.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_gate_costs_the_device_at_each_reduction_width() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let simd = matmul.kernel.simd_width();
        let most = THREADS_PER_GROUP / simd;
        const CALLS: usize = 64;
        const ROUNDS: usize = 3;

        let case = Case::noisy(IN_DIM, OUT_DIM, 1);
        let weight = case.upload(&device, &matmul);
        let runs: Vec<usize> = (0..)
            .map(|i| 1 << i)
            .take_while(|run| *run <= most)
            .collect();
        let cost = |rows: usize, run: usize| -> Duration {
            let x: Vec<f32> = (0..rows * IN_DIM)
                .map(|i| ((i % 37) as f32 - 18.0) / 16.0)
                .collect();
            let mut input = device.buffer(&x).expect("the rows upload");
            let mut out = device
                .zeroed::<f32>(rows * OUT_DIM)
                .expect("the answer allocates");
            let elements = rows * OUT_DIM;
            // The dispatch's own fields, from the weight rather than restated:
            // the kernel's `Shape` is an ABI, and a second spelling of it here
            // is one that goes stale the next time a field is added to it.
            let fields = weight.shape(rows, run);
            let mut shape = device.inline(&fields).expect("the shape inlines");
            let mut resident = weight.resident.borrow_mut();
            let grid = Grid::new(elements * run * simd, THREADS_PER_GROUP);

            crate::testing::device_time(&device, CALLS, |batch| {
                batch
                    .add(
                        &matmul.kernel,
                        &[shape.arg(), input.arg(), resident.arg(), out.arg()],
                        grid,
                        0,
                    )
                    .expect("the dispatch encodes");
            })
        };

        let shapes = [1usize, 2, 4, 8, 16, 32, 97, 385];
        let mut taken = vec![vec![Vec::new(); runs.len()]; shapes.len()];
        for round in 0..=ROUNDS {
            for (s, rows) in shapes.iter().enumerate() {
                for (r, run) in runs.iter().enumerate() {
                    let each = cost(*rows, *run);
                    if round > 0 {
                        taken[s][r].push(each);
                    }
                }
            }
        }
        for (s, rows) in shapes.iter().enumerate() {
            let means: Vec<Duration> = taken[s]
                .iter()
                .map(|each| each.iter().sum::<Duration>() / each.len() as u32)
                .collect();
            let best = means
                .iter()
                .enumerate()
                .min_by_key(|(_, each)| **each)
                .expect("a run")
                .0;
            let chose = simdgroups_an_element(rows * OUT_DIM, most);
            eprintln!(
                "[{rows}, {IN_DIM}] @ [{OUT_DIM}, {IN_DIM}]^T over {:?}: {}, best {} at {:.2?}, \
                 the rule chose {chose} at {:.2?}",
                runs,
                means
                    .iter()
                    .map(|each| format!("{each:.2?}"))
                    .collect::<Vec<String>>()
                    .join(" "),
                runs[best],
                means[best],
                means[runs.iter().position(|run| *run == chose).expect("a run")],
            );
        }
    }

    /// Which of a value's two bytes is the high one, which is the one fact
    /// about the format a kernel can get backwards while still producing finite
    /// weights of a plausible shape.
    ///
    /// Stated as a kernel rather than as a mutated input, because that is where
    /// the mistake would live: the same source, the same dispatch, the same
    /// bytes, and the pair read the other way round.
    ///
    /// Over the dyadic case rather than the noisy one, and [`Case::dyadic`]
    /// says why: the swap moves a value's mantissa into its exponent field, so
    /// a wide row of noise read this way overflows to NaN, which `deviation`
    /// refuses to give a number for.
    #[test]
    fn reading_a_values_two_bytes_the_other_way_round_is_a_different_answer() {
        let Some(device) = device() else { return };
        let (case, _) = Case::dyadic();

        let reversed = source().replace("((high << 8) | low)", "((low << 8) | high)");
        assert_ne!(reversed, source(), "the mutation changed nothing");
        let mutant = DenseMatmul::from_source(&device, &reversed).expect("the mutant compiles");

        let got = case
            .upload(&device, &mutant)
            .multiply(&case.x)
            .expect("the dispatch completes");
        let deviation = deviation(&got, &case.on_the_cpu());
        eprintln!("the two bytes read the other way round: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// Rows of `x` are independent, and each gets its own row of the output at
    /// its own offset. A kernel that took the row index off the wrong axis
    /// would still fill the buffer.
    #[test]
    fn every_row_of_the_input_gets_its_own_row_of_the_output() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 2);
        let weight = case.upload(&device, &matmul);

        let both = weight.multiply(&case.x).expect("the dispatch completes");
        assert_eq!(both.len(), 2 * OUT_DIM);
        for (row, x) in case.x.chunks_exact(IN_DIM).enumerate() {
            let alone = weight.multiply(x).expect("the dispatch completes");
            assert_eq!(both[row * OUT_DIM..][..OUT_DIM], alone[..], "row {row}");
        }
    }

    /// Two multiplies of different heights in one command buffer, which is what
    /// says a dispatch's shape belongs to the dispatch rather than to the
    /// weight — the case the gate meets on its first step, where a prefill of
    /// eight rows is followed by decodes of one.
    ///
    /// The taller call is encoded first, which is what makes that visible: the
    /// kernel's bounds check would cull two thirds of its output against the
    /// shorter call's row count, where a shorter call reading a taller one's
    /// shape agrees by accident.
    #[test]
    fn two_multiplies_against_one_weight_in_one_batch_keep_their_own_shapes() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 3);
        let weight = case.upload(&device, &matmul);
        let (one, three) = (&case.x[..IN_DIM], &case.x[..]);

        let [batched_three, batched_one] = crate::matmul::together(&device, |batch| {
            Ok([weight.encode(batch, three)?, weight.encode(batch, one)?])
        })
        .expect("both dispatches complete");

        assert_eq!(batched_three.len(), 3 * OUT_DIM);
        assert_eq!(batched_one.len(), OUT_DIM, "one row in, one row out");
        assert_eq!(
            batched_three,
            weight.multiply(three).expect("the dispatch completes"),
            "the shorter call's shape reached the taller one"
        );
        assert_eq!(
            batched_one,
            weight.multiply(one).expect("the dispatch completes")
        );
    }

    /// A weight of the wrong length for the shape it was given is the mistake
    /// the check on the way in exists to catch: the kernel takes its bounds
    /// from the shape and would read off the end of the buffer.
    #[test]
    fn a_weight_that_does_not_fill_its_shape_is_refused() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(64, 4, 1);

        let err = DenseWeight::upload(&device, &matmul, 64, 4, &case.weight[2..])
            .expect_err("the shape does not pair");
        assert!(
            matches!(
                err,
                MatmulError::WrongWeightLen {
                    expected: 512,
                    got: 510,
                    ..
                }
            ),
            "{err}"
        );
    }

    /// The same kernel with a different [`VALUES_PER_LANE`] written into its
    /// prelude, which is how a sweep prices the widths and how the case below
    /// puts each of them against the CPU.
    fn a_lane_reading(values_per_lane: usize) -> String {
        let wanted = format!("constant uint VALUES_PER_LANE = {values_per_lane};");
        let source = source().replace(
            &format!("constant uint VALUES_PER_LANE = {VALUES_PER_LANE};"),
            &wanted,
        );
        assert!(source.contains(&wanted), "the prelude declares {wanted}");
        source
    }

    /// The widths the sweep prices and this case holds to the CPU.
    const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];

    /// **Every width answers what the CPU answers, including one that does not
    /// divide the row.**
    ///
    /// A lane holds a run of consecutive values, so a row that is not a whole
    /// number of chunks leaves one lane's chunk straddling its end — and the
    /// kernel finishes that lane one value at a time rather than the caller
    /// refusing the width. A width that read past the row would be reading the
    /// *next* output element's weight, which is a plausible number rather than a
    /// fault, and a width that stopped short would drop the tail's products.
    ///
    /// So the shapes are chosen against the widths rather than against the
    /// engine: 4096 is what every weight this dispatches is a multiple of, and
    /// 250 is a width no chunk above one divides — 62 chunks of four and two
    /// values over, at every reduction run the row counts here reach.
    #[test]
    fn every_width_a_lane_reads_answers_what_the_cpu_answers() {
        let Some(device) = device() else { return };
        assert!(WIDTHS.contains(&VALUES_PER_LANE), "{VALUES_PER_LANE}");

        for in_dim in [IN_DIM, 250] {
            // Two rows, so a lane's chunk offset is checked against the row it
            // is inside rather than against the start of the weight.
            let case = Case::noisy(in_dim, OUT_DIM, 2);
            let want = case.on_the_cpu();
            for width in WIDTHS {
                let matmul = DenseMatmul::from_source(&device, &a_lane_reading(width))
                    .unwrap_or_else(|err| panic!("{width} values a lane compiles: {err}"));
                let got = case
                    .upload(&device, &matmul)
                    .multiply(&case.x)
                    .expect("the dispatch completes");

                let deviation = deviation(&got, &want);
                assert!(
                    deviation <= TOLERANCE,
                    "{width} values a lane over a row of {in_dim}: deviation {deviation:e}"
                );
            }
        }
    }

    /// **What a dense multiply costs the device at each width a lane reads**,
    /// and the sweep [`VALUES_PER_LANE`] was chosen from.
    ///
    /// The shapes are the ones this kernel is dispatched at rather than round
    /// numbers, and they are two workloads: a chain of MTP heads, where 75% of
    /// the device time is this kernel and every weight is 4096 wide, and the
    /// routers' `[258, 4096]` gate, which is the only thing a decode step or a
    /// prefill sends here at all. A width that helped the heads and cost the
    /// gate would be paying for speculation out of the step it accelerates.
    ///
    /// Nothing asserts a rate. The numbers go to stderr for the commit message
    /// to quote, and what is asserted is that the shipped width was among the
    /// ones tried.
    ///
    /// **The rates rank the widths and do not state a bandwidth**, for the
    /// reason `matmul::tests::what_a_packed_multiply_costs_at_each_width_a_lane_reads`
    /// gives: one weight is dispatched against `CALLS` times in a row, so what
    /// the second call reads is what the first left in cache, where a head's own
    /// 532 MiB is cold every round.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_dense_multiply_costs_at_each_width_a_lane_reads() {
        let Some(device) = device() else { return };
        const CALLS: usize = 8;
        const ROUNDS: usize = 3;

        // `(what, in_dim, out_dim, rows)`.
        let shapes = [
            ("a head's input_proj", 8192, 4096, 2),
            ("a head's q_proj", 4096, 4096, 2),
            ("a head's k_proj", 4096, 1024, 2),
            ("a head's gate or up", 4096, 16384, 2),
            ("a head's down_proj", 16384, 4096, 2),
            ("a router's gate, decode", 4096, 258, 1),
            ("a router's gate, 97 tokens", 4096, 258, 97),
        ];

        assert!(WIDTHS.contains(&VALUES_PER_LANE), "{VALUES_PER_LANE}");
        eprintln!(
            "  {:<28}{}",
            "shape",
            WIDTHS
                .iter()
                .map(|width| format!("{:>10}", format!("{width}/lane")))
                .collect::<String>()
        );
        for (what, in_dim, out_dim, rows) in shapes {
            let case = Case::noisy(in_dim, out_dim, rows);
            let mut x = device.buffer(&case.x).expect("the rows upload");
            let mut line = format!("  {what:<28}");
            for width in WIDTHS {
                let matmul = DenseMatmul::from_source(&device, &a_lane_reading(width))
                    .unwrap_or_else(|err| panic!("{width} values a lane compiles: {err}"));
                let weight = case.upload(&device, &matmul);

                let mut best = Duration::MAX;
                for _ in 0..ROUNDS {
                    best = best.min(crate::testing::device_time(&device, CALLS, |batch| {
                        weight
                            .encode_over(batch, &mut x)
                            .expect("the dispatch encodes");
                    }));
                }
                line.push_str(&format!(
                    "{:>10}",
                    format!(
                        "{:.0} GB/s",
                        case.moves(rows) as f64 / best.as_secs_f64() / 1e9
                    )
                ));
            }
            eprintln!("{line}");
        }
    }
}
