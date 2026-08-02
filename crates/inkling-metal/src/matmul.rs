//! `out = x @ wᵀ` against a weight that stays MXFP4-packed.
//!
//! This is the primitive the engine rests on. `lm_head`, every projection and —
//! with a gather — every expert are the same operation over the same format, so
//! what this module gets right applies everywhere and what it gets wrong applies
//! everywhere too.
//!
//! **The codes are decoded in registers, inside the multiply loop.** A kernel
//! that dequantised into a buffer and multiplied that would be the CPU path with
//! a GPU under it, and would have given away the whole point before it started:
//! every token touches all of every projection, so a decode step moves 41 GB of
//! float32 where the packed bytes are 5 GB. Here a thread loads one packed byte,
//! turns its two nibbles into two floats it holds in registers, multiplies them
//! against two inputs, and never writes a decoded weight anywhere.
//!
//! **A byte at a time, because the checkpoint is not word-aligned.** MXFP4 packs
//! eight codes into a little-endian `u32`, which is two codes to a byte with the
//! low nibble first — so a byte is a whole number of codes and reading the
//! weight bytewise needs nothing of the format that reading it wordwise does.
//! What it buys is that the bytes can be the checkpoint's own: this quant's
//! shard headers are not padded, so every tensor in it begins one byte past a
//! word and a `device const uint *` cannot be pointed at one. Bound as bytes it
//! can, and [`Device::wrap`](crate::Device::wrap) then gives the GPU `lm_head`
//! where the checkpoint mapped it rather than 0.41 GiB of copy. It costs 2% of
//! the dispatch, measured on that same head.
//!
//! **One simdgroup per output element, and [`BYTES_PER_LANE`] bytes to a lane a
//! step.** Lane `l` walks its weight row from byte `l * BYTES_PER_LANE` in
//! strides of that many times the simdgroup width, and the group sums what the
//! lanes held — so the 32 lanes of one reduction read 128 consecutive bytes,
//! which is what makes a matmul this memory-bound run at the bandwidth rather
//! than at a thirty-second of it. The cost is that a call with several rows of
//! `x` reads the weight once per row; decode is one row, and a prefill shape
//! that wants the weight read once is a tiling commit of its own.
//!
//! **`0x00` scale bytes decode to zero here.** The scale is
//! `as_type<float>(byte << 23)`, which is exact for `0x01..=0xfe` and gives 0
//! where [`inkling_core::quant`] gives `2^-127`. That is the divergence that
//! module licenses: it surveys all 458 quantised tensors and has never found a
//! group where the two readings decode to different weights, because `0x00`
//! appears only against all-zero codes. Taking the shift is a branch removed
//! from the inner loop, and the CPU path stays pinned to MLX.

use std::cell::RefCell;

use inkling_core::ops::Projection;
use inkling_core::profile::{self, Op};
use inkling_core::quant::{BITS, ELEMENTS, GROUP_SIZE};
use inkling_core::weights::Packed;

use crate::buffer::{Arg, Buffer, Bytes};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "packed_matmul";

/// Threads one threadgroup of a dispatch holds.
///
/// A multiple of every simdgroup width Metal reports, which is what lets the
/// kernel take its output element from `thread_position_in_grid` divided by
/// `threads_per_simdgroup` and get the same answer as `thread_index_in_simdgroup`
/// gives for the lane.
const THREADS_PER_GROUP: usize = 256;

/// Codes packed into one byte, which is what makes a byte a whole number of
/// codes and so what lets the weight be read without regard to word alignment.
pub(crate) const CODES_PER_BYTE: usize = u8::BITS as usize / BITS;

/// Packed bytes one scale byte covers.
const BYTES_PER_GROUP: usize = GROUP_SIZE / CODES_PER_BYTE;

/// Packed bytes one lane reads before the next lane's, which is how much of a
/// weight row one thread has in flight at a time.
///
/// **What a chunk buys is the three loads beside the codes amortised over it**,
/// which is `rms_norm`'s and `dense_matmul`'s finding met a third time: this
/// kernel waits on the requests one thread has outstanding rather than on the
/// memory behind them. A code byte is read alongside its group's scale byte and
/// the two inputs its two codes multiply, and of those only the codes are the
/// dispatch's own bytes — so a lane holding four of them issues four memory
/// instructions where four lane-steps of one byte issue sixteen, because the
/// scale is one byte for the whole chunk and the eight inputs are consecutive
/// floats the compiler loads together.
///
/// **Four rather than eight or sixteen, and the sweep says why.** Wider chunks
/// are faster at the shapes with few output elements — a decode step's
/// `[1, 4096] @ [1024, 4096]ᵀ` key projection reaches 168 GB/s here, 191 at
/// eight and 193 at sixteen — and slower at every shape with elements enough to
/// fill the machine, which is where a step's bytes are: the routed banks are
/// 55% of what a step reads and give up 4% at eight, and a 97-token prefill 7%.
/// See `what_a_packed_multiply_costs_at_each_width_a_lane_reads`.
///
/// A chunk has to divide [`BYTES_PER_GROUP`], so that the bytes one lane holds
/// share one scale byte, and it has to divide a weight row, so that no lane
/// reads past one. Both hold for 4, 8 and 16 against any `in_dim` this module
/// accepts: [`pairs`] refuses a width that is not whole groups of
/// [`GROUP_SIZE`] codes, which is 16 packed bytes.
const BYTES_PER_LANE: usize = 4;

/// Where an f32's exponent field starts, above its 23 stored mantissa bits.
const EXPONENT_SHIFT: u32 = f32::MANTISSA_DIGITS - 1;

#[derive(Debug, thiserror::Error)]
pub enum MatmulError {
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[error("an input width of {0} is not whole groups of {GROUP_SIZE} codes")]
    PartialGroup(usize),

    #[error("{rows} rows of {in_dim} codes are {expected} packed bytes, got {got}")]
    WrongCodeLen {
        rows: usize,
        in_dim: usize,
        expected: usize,
        got: usize,
    },

    #[error("{rows} rows of {in_dim} codes need {expected} scale bytes, got {got}")]
    WrongScaleLen {
        rows: usize,
        in_dim: usize,
        expected: usize,
        got: usize,
    },

    #[error("expert {expert} of a bank that holds {experts}")]
    NoSuchExpert { expert: usize, experts: usize },

    #[error("{what} is {got}, not the {expected} gate_proj states")]
    MismatchedBanks {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("{out_dim} rows of {in_dim} bfloat16 values are {expected} bytes, got {got}")]
    WrongWeightLen {
        in_dim: usize,
        out_dim: usize,
        expected: usize,
        got: usize,
    },
}

/// The compiled kernel, which every packed projection on a device shares.
///
/// Compilation is per source string rather than per weight, and the source does
/// not mention a shape, so one of these serves the whole model.
#[derive(Debug)]
pub struct PackedMatmul {
    kernel: Kernel,
}

impl PackedMatmul {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`PackedMatmul::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }
}

/// An `[experts, out_dim, in_dim]` MXFP4 weight on the device, and the gathered
/// multiply against it.
///
/// **The gather is what makes 137 GB of routed experts affordable to have
/// resident.** A decode step's token picks 6 of 256, and every row of a call
/// names the expert it goes through, so one dispatch reads the six banks that
/// were chosen and never touches the other 250 — the win is precisely in not
/// reading them, and a kernel that took the expert as a shape rather than as a
/// per-row index could not express it in one dispatch.
///
/// Made resident once and multiplied against many times. Wrapped, "once" is
/// about 50 microseconds a gibibyte and no copy at all, so what that phrase used
/// to be defending against is gone.
#[derive(Debug)]
pub struct PackedBank<'a> {
    device: &'a Device,
    matmul: &'a PackedMatmul,
    experts: usize,
    in_dim: usize,
    out_dim: usize,
    resident: RefCell<Resident<'a>>,
}

/// What a bank holds on the device between calls: the two tensors the checkpoint
/// pairs, and nothing else.
///
/// The shape is not here. It carries the row count of a *call*, so a bank that
/// held one could not have two calls of different heights encoded into one
/// command buffer — the second would overwrite what the first is waiting to be
/// read with — and a batch that could not hold two multiplies against one bank
/// would be a rule nothing states. What that rules out is a *resident* shape,
/// not an allocated one: [`Device::inline`](crate::Device::inline) copies a
/// call's shape into the command buffer as the dispatch is encoded, which costs
/// no allocation and is still the call's own.
#[derive(Debug)]
struct Resident<'a> {
    codes: Bytes<'a>,
    scales: Bytes<'a>,
}

/// The output of a multiply encoded into a [`Batch`] that has not run yet.
///
/// Only the output is held. Everything else a dispatch reads — the shape, the
/// expert list, the input — is retained by the command buffer it was bound
/// into, so it outlives this side of it whatever happens here; the output is
/// held because somebody has to read it afterwards.
#[derive(Debug)]
pub struct Pending {
    /// `None` for a call with no rows, which is not a dispatch at all: the
    /// device refuses a zero-length buffer.
    out: Option<Buffer<f32>>,
}

impl Pending {
    /// What a dispatch of another kernel left to be read, for the modules
    /// beside this one that encode into the same [`Batch`] — see
    /// [`crate::dense`].
    pub(crate) fn holding(out: Buffer<f32>) -> Self {
        Self { out: Some(out) }
    }

    /// What a dispatch over no rows produced, which is nothing — a bank no token
    /// routed to is an ordinary step of the router's and not an error.
    pub(crate) fn empty() -> Self {
        Self { out: None }
    }

    /// The buffer itself, for a caller that has another dispatch to feed with it
    /// rather than a value to read.
    ///
    /// Panics on a call with no rows, which is a caller asking a *second*
    /// dispatch to consume nothing — and the device refuses a zero-length
    /// buffer, so there would be nothing to hand it. A forward pass over no
    /// tokens is not a chain of dispatches; it is not a forward pass.
    pub(crate) fn buffer(self) -> Buffer<f32> {
        self.out.expect("a dispatch over no rows has no output")
    }

    /// The same buffer where a call over no rows is a case the caller handles
    /// rather than a contradiction. A bank of 256 experts that no token routed
    /// to is exactly that, and it has nothing to feed the dispatch after it.
    pub(crate) fn into_buffer(self) -> Option<Buffer<f32>> {
        self.out
    }

    /// The values, once the batch this was encoded into has been waited for.
    pub fn take(self) -> Vec<f32> {
        let _timed = profile::scope(Op::Readback);
        self.out.map(|out| out.to_vec()).unwrap_or_default()
    }
}

/// Everything `encode` put in one command buffer, run and read back.
///
/// The shape a caller with several multiplies in hand wants, and the reason
/// [`PackedBank::encode`] is public: encode them all, wait once, and the array
/// that comes back is what each of them produced.
///
/// Fallible, like the multiplies it runs. A caller standing behind an
/// infallible seam panics on the way past — which is what
/// [`Projection::forward`] already does — and one that promised a `Result`
/// keeps that promise for every dispatch rather than for the last of them.
pub fn together<const N: usize>(
    device: &Device,
    encode: impl FnOnce(&mut Batch<'_>) -> Result<[Pending; N], MatmulError>,
) -> Result<[Vec<f32>; N], MatmulError> {
    let mut batch = device.batch()?;
    let pending = encode(&mut batch)?;
    batch.wait()?;
    Ok(pending.map(Pending::take))
}

/// Whether `codes` and `scales` are an `[experts, out_dim, in_dim]` MXFP4
/// weight, which has to be settled on the way in: the kernel takes its bounds
/// from the shape it was told and would read off the end of whichever tensor was
/// short.
fn pairs(
    experts: usize,
    in_dim: usize,
    out_dim: usize,
    codes: usize,
    scales: usize,
) -> Result<(), MatmulError> {
    if in_dim == 0 || in_dim % GROUP_SIZE != 0 {
        return Err(MatmulError::PartialGroup(in_dim));
    }

    let expected = experts * out_dim * in_dim / CODES_PER_BYTE;
    if codes != expected {
        return Err(MatmulError::WrongCodeLen {
            rows: experts * out_dim,
            in_dim,
            expected,
            got: codes,
        });
    }
    let expected = experts * out_dim * in_dim / GROUP_SIZE;
    if scales != expected {
        return Err(MatmulError::WrongScaleLen {
            rows: experts * out_dim,
            in_dim,
            expected,
            got: scales,
        });
    }
    Ok(())
}

impl<'a> PackedBank<'a> {
    /// Copy an `[experts, out_dim, in_dim]` MXFP4 weight onto the device, as the
    /// two tensors the checkpoint pairs: `codes` is the bytes of the `U32`
    /// weight and `scales` the bytes of the `U8` one.
    ///
    /// For weights that are not a mapping's — a test's synthetic bytes, and
    /// anything a future path builds rather than reads. [`PackedBank::wrap`] is
    /// what a checkpoint's own tensor takes.
    pub fn upload(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        experts: usize,
        in_dim: usize,
        out_dim: usize,
        codes: &[u8],
        scales: &[u8],
    ) -> Result<Self, MatmulError> {
        pairs(experts, in_dim, out_dim, codes.len(), scales.len())?;
        Self::over(
            device,
            matmul,
            experts,
            in_dim,
            out_dim,
            Bytes::Copied(device.buffer(codes)?),
            Bytes::Copied(device.buffer(scales)?),
        )
    }

    /// A checkpoint's own bank, read where it is mapped: every slice of its
    /// leading axis is an expert, and `in_dim` says how each expert's remaining
    /// values divide into rows.
    ///
    /// `in_dim` rather than `out_dim` because that is the axis the checkpoint's
    /// shape does not give. A routed bank is `[256, 2048, 512]` `U32`, whose last
    /// axis is packed words rather than either dimension of the multiply.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &Packed<'a>,
        in_dim: usize,
    ) -> Result<Self, MatmulError> {
        let experts = packed.slices();
        Self::wrapping(
            device,
            matmul,
            packed,
            experts,
            in_dim,
            packed.slice_len() / in_dim,
            experts,
        )
    }

    fn wrapping(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &Packed<'a>,
        slices: usize,
        in_dim: usize,
        out_dim: usize,
        experts: usize,
    ) -> Result<Self, MatmulError> {
        let (codes, scales) = packed.prefix(slices);
        pairs(experts, in_dim, out_dim, codes.len(), scales.len())?;
        // SAFETY: the bytes are a `Checkpoint`'s mapping, which outlives this by
        // the lifetime they carry and which nothing writes — the assumption that
        // module already maps under.
        let (codes, scales) = unsafe { (device.wrap(codes)?, device.wrap(scales)?) };
        Self::over(
            device,
            matmul,
            experts,
            in_dim,
            out_dim,
            Bytes::Mapped(codes),
            Bytes::Mapped(scales),
        )
    }

    fn over(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        experts: usize,
        in_dim: usize,
        out_dim: usize,
        codes: Bytes<'a>,
        scales: Bytes<'a>,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            resident: RefCell::new(Resident { codes, scales }),
            device,
            matmul,
            experts,
            in_dim,
            out_dim,
        })
    }

    pub(crate) fn device(&self) -> &'a Device {
        self.device
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// The scalars the kernel's `Shape` struct declares, in its order — the
    /// caller's to hold, because they are read where the dispatch is encoded and
    /// an array made here would be gone by then.
    fn shape(&self, rows: usize, per_source: usize, sources: usize) -> [u32; SHAPE_FIELDS] {
        let resident = self.resident.borrow();
        let stride = self.out_dim * self.in_dim;
        [
            extent(rows, "the rows of a call"),
            extent(self.in_dim, "the width a bank maps from"),
            extent(self.out_dim, "the width a bank maps to"),
            extent(
                per_source,
                "the rows of a call that read one row of the input",
            ),
            extent(sources, "the rows of the input"),
            extent(resident.codes.offset(), "where a bank's codes start"),
            extent(resident.scales.offset(), "where a bank's scales start"),
            extent(stride / CODES_PER_BYTE, "the packed bytes of an expert"),
            extent(stride / GROUP_SIZE, "the scale bytes of an expert"),
        ]
    }

    /// `[rows, in_dim]` in, `[rows, out_dim]` out, row `i` multiplied against
    /// expert `experts[i]`.
    ///
    /// Fallible where [`Projection::forward`] is not, because a dispatch can
    /// fail in ways no arithmetic can: the watchdog kills a command buffer that
    /// runs too long, and a caller that wants to say so rather than die needs
    /// the error.
    pub fn multiply(&self, experts: &[u32], x: &[f32]) -> Result<Vec<f32>, MatmulError> {
        let mut batch = self.device.batch()?;
        let pending = self.encode(&mut batch, experts, x)?;
        batch.wait()?;
        Ok(pending.take())
    }

    /// The same multiply, encoded into `batch` rather than submitted on its own.
    ///
    /// What this buys is the whole of the granularity question: a submission
    /// costs 225 microseconds whatever is in it, so a caller with several
    /// multiplies whose inputs are all in hand — the four projections of an
    /// attention layer, a feed-forward network's gate and up — encodes them
    /// together and waits once.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &[f32],
    ) -> Result<Pending, MatmulError> {
        assert_eq!(
            self.sources(x.len()),
            experts.len(),
            "{} values are not {} rows of {}",
            x.len(),
            experts.len(),
            self.in_dim
        );
        if experts.is_empty() {
            return Ok(Pending::empty());
        }
        let _timed = profile::scope(Op::Encode);
        let mut x = self.device.buffer(x)?;
        self.encoding(batch, experts, &mut x)
    }

    /// The same multiply over rows a dispatch already left on the device.
    ///
    /// **What the engine is heading towards.** [`PackedBank::encode`] above
    /// takes a `&[f32]`, and that signature is a claim the way
    /// [`linear`](inkling_core::ops::linear)'s is: it says the rows are in this
    /// process's memory, which for anything a kernel produced means they were
    /// copied off the device to be copied back on. An activation that is only
    /// ever read by another kernel never has to make that crossing, and the four
    /// projections that consume one normed hidden state are the first four
    /// callers that can say so.
    ///
    /// The input is borrowed exclusively for the encoding and not for the batch,
    /// which is what lets one buffer feed several dispatches of the same command
    /// buffer: Metal retains what is bound into it, so the binding outlives the
    /// borrow.
    pub fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        self.encoding(batch, experts, x)
    }

    /// The same multiply over an expert list a dispatch already left on the
    /// device, one row of the input feeding `per_source` consecutive rows of the
    /// call.
    ///
    /// **This is a gather nobody gathered.** A token that reads `per_source`
    /// experts is that many rows here and one row of `x`, so the rows the bank
    /// runs are the hidden state read at a stride rather than a tensor cut out
    /// of it — and the expert each goes through is whatever the dispatch that
    /// wrote `chosen` put there, which is what lets a router on this device sit
    /// in the same command buffer as the bank it routes.
    ///
    /// **The expert indices are not checked and cannot be**: they are in device
    /// memory and this side has not seen them. An index past the bank is an
    /// offset past the buffer, which a GPU read answers with whatever is there
    /// rather than with a fault, so whoever writes `chosen` owes this the range
    /// — see [`LayerExperts::wrap`](crate::LayerExperts::wrap), which is why
    /// this is not public.
    pub(crate) fn encode_picked(
        &self,
        batch: &mut Batch<'_>,
        chosen: &mut Buffer<u32>,
        x: &mut Buffer<f32>,
        per_source: usize,
    ) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        let rows = chosen.len();
        assert!(
            per_source > 0,
            "a row of a call reads some row of the input"
        );
        assert_eq!(
            self.sources(x.len()),
            rows.div_ceil(per_source),
            "{} values are not {} rows of {}",
            x.len(),
            rows.div_ceil(per_source),
            self.in_dim
        );
        self.dispatch(batch, rows, per_source, chosen.arg(), x)
    }

    /// The same multiply over rows a dispatch already left on the device, every
    /// one of them read once per expert `experts` names in turn.
    ///
    /// **This is the shared bank's shape, and it is the same gather nobody
    /// gathered.** Every token goes through every shared expert, so the rows the
    /// bank runs are the input laid end to end after itself once per expert —
    /// `[n_shared * tokens, in_dim]` of which every row is already `[tokens,
    /// in_dim]` somewhere. Read at a modulo it is not laid out at all, and the
    /// input can be the buffer the router's gate and the routed bank are reading
    /// in the same command buffer.
    ///
    pub(crate) fn encode_repeating(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        let sources = self.sources(x.len());
        assert!(
            experts.is_empty() || (sources > 0 && experts.len() % sources == 0),
            "{} rows are not whole passes over {sources} rows of input",
            experts.len()
        );
        self.listed(batch, experts, x)
    }

    /// One dispatch encoded, without the scope its callers each open — so that
    /// the profile counts a dispatch once however it was reached.
    fn encoding(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        assert_eq!(
            self.sources(x.len()),
            experts.len(),
            "{} values are not {} rows of {}",
            x.len(),
            experts.len(),
            self.in_dim
        );
        self.listed(batch, experts, x)
    }

    /// A dispatch over an expert list this side holds, which is every one but
    /// [`PackedBank::encode_picked`]'s.
    ///
    /// The range *is* checked here, and it has to be: the kernel cannot, because
    /// an index past the bank is an offset past the buffer and a GPU read
    /// answers that with whatever is there rather than with a fault. A list a
    /// dispatch wrote is past this side and is why that one method is not
    /// public.
    ///
    /// `per_source` is 1 for both callers. What separates them is only how many
    /// rows of input the same list is spread over, which is
    /// [`PackedBank::sources`] and is read off the buffer rather than passed.
    fn listed(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        if let Some(past) = experts
            .iter()
            .find(|expert| **expert as usize >= self.experts)
        {
            return Err(MatmulError::NoSuchExpert {
                expert: *past as usize,
                experts: self.experts,
            });
        }
        if experts.is_empty() {
            return Ok(Pending::empty());
        }

        let mut chosen = self.device.inline(experts)?;
        self.dispatch(batch, experts.len(), 1, chosen.arg(), x)
    }

    /// How many rows of this bank's input width `values` is, which is what the
    /// kernel takes its modulo over.
    fn sources(&self, values: usize) -> usize {
        assert_eq!(
            values % self.in_dim,
            0,
            "{values} values are not whole rows of {}",
            self.in_dim
        );
        values / self.in_dim
    }

    /// The dispatch itself, over an expert list wherever it lies.
    fn dispatch(
        &self,
        batch: &mut Batch<'_>,
        rows: usize,
        per_source: usize,
        chosen: Arg<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        if rows == 0 {
            return Ok(Pending::empty());
        }

        // The shape is read out of `resident` and so is built before it is
        // borrowed mutably for the binding below.
        let fields = self.shape(rows, per_source, self.sources(x.len()));
        let mut shape = self.device.inline(&fields)?;
        let mut resident = self.resident.borrow_mut();
        let resident = &mut *resident;
        let mut out = self.device.zeroed::<f32>(rows * self.out_dim)?;

        let elements = rows * self.out_dim;
        let kernel = &self.matmul.kernel;
        let grid = Grid::new(elements * kernel.simd_width(), THREADS_PER_GROUP);
        let moves = self.moves(rows, x.len());
        batch.add(
            kernel,
            &[
                shape.arg(),
                chosen,
                x.arg(),
                resident.codes.arg(),
                resident.scales.arg(),
                out.arg(),
            ],
            grid,
            moves,
        )?;

        Ok(Pending { out: Some(out) })
    }

    /// What one dispatch of `rows` rows over `values` of input moves.
    ///
    /// **A row is a whole weight.** Each output row goes through one expert and
    /// every element of that row reads a different `in_dim`-long slice of it, so
    /// `rows` rows read `rows` weights whichever experts they name. That is what
    /// makes this the kernel a decode step's bandwidth is mostly about: six
    /// routed rows of a `[2048, 4096]` bank are 27 MB of packed bytes for six
    /// rows of output.
    ///
    /// The weight is charged as the bytes it is packed into rather than the
    /// values it holds — a code is half a byte and a group of 32 codes shares
    /// one scale byte — because the whole of this kernel is that nothing decodes
    /// it on the way. Beside it, the input, the output, and the one expert index
    /// a row is read through, which a bank's own dispatch leaves on the device
    /// and a projection's travels in the command buffer.
    fn moves(&self, rows: usize, values: usize) -> usize {
        let elements = rows * self.out_dim * self.in_dim;
        let weight = elements * BITS / u8::BITS as usize + elements / GROUP_SIZE;
        weight + size_of::<f32>() * (values + rows * self.out_dim) + size_of::<u32>() * rows
    }
}

/// One `[out_dim, in_dim]` weight, as the bank of one expert it is.
///
/// A projection and a bank are the same operation over the same format — the
/// difference is only whether the leading axis has anything in it — so this is
/// a [`PackedBank`] that names the one expert every row goes through rather
/// than a second way of dispatching. `lm_head` is the caller.
#[derive(Debug)]
pub struct PackedProjection<'a> {
    bank: PackedBank<'a>,
    /// The expert each row goes through, which for a bank of one is row after
    /// row of zero. Grown to the tallest call and never shrunk — a call reads
    /// the prefix it needs — because a decode step asks for one row where the
    /// prefill before it asked for eight, and the list is the same zeros either
    /// way.
    chosen: RefCell<Vec<u32>>,
}

impl<'a> PackedProjection<'a> {
    /// Copy an `[out_dim, in_dim]` MXFP4 weight onto the device.
    pub fn upload(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        in_dim: usize,
        out_dim: usize,
        codes: &[u8],
        scales: &[u8],
    ) -> Result<Self, MatmulError> {
        Ok(Self::of(PackedBank::upload(
            device, matmul, 1, in_dim, out_dim, codes, scales,
        )?))
    }

    /// A checkpoint's own tensor, cut to its first `out_dim` slices and read
    /// where it is mapped.
    ///
    /// The cut is what the head's truncation is: `lm_head` is `[201024, 4096]`
    /// and 200058 of those rows are vocabulary, so a projection built to the
    /// vocabulary stops dispatching where the padding starts. Wrapped, the 966
    /// padding rows are not even a range that was declined — they are pages
    /// nothing indexes.
    pub fn wrap_packed(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &Packed<'a>,
        out_dim: usize,
    ) -> Result<Self, MatmulError> {
        Ok(Self::of(PackedBank::wrapping(
            device,
            matmul,
            packed,
            out_dim,
            packed.slice_len(),
            out_dim,
            1,
        )?))
    }

    fn of(bank: PackedBank<'a>) -> Self {
        Self {
            bank,
            chosen: RefCell::new(Vec::new()),
        }
    }

    /// `[rows, in_dim]` in, `[rows, out_dim]` out.
    pub fn multiply(&self, x: &[f32]) -> Result<Vec<f32>, MatmulError> {
        let mut batch = self.bank.device().batch()?;
        let pending = self.encode(&mut batch, x)?;
        batch.wait()?;
        Ok(pending.take())
    }

    /// The same multiply, encoded into `batch` rather than submitted on its own.
    pub fn encode(&self, batch: &mut Batch<'_>, x: &[f32]) -> Result<Pending, MatmulError> {
        let rows = self.rows(x.len());
        let chosen = self.chosen.borrow();
        self.bank.encode(batch, &chosen[..rows], x)
    }

    /// The same multiply over rows a dispatch already left on the device — see
    /// [`PackedBank::encode_over`].
    pub fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let rows = self.rows(x.len());
        let chosen = self.chosen.borrow();
        self.bank.encode_over(batch, &chosen[..rows], x)
    }

    /// How many rows of this projection's width `values` is, with the list of
    /// experts grown to name one for each of them.
    fn rows(&self, values: usize) -> usize {
        let in_dim = self.bank.in_dim();
        assert_eq!(
            values % in_dim,
            0,
            "{values} values are not whole rows of {in_dim}"
        );
        let rows = values / in_dim;
        let mut chosen = self.chosen.borrow_mut();
        if chosen.len() < rows {
            chosen.resize(rows, 0);
        }
        rows
    }

    pub(crate) fn device(&self) -> &Device {
        self.bank.device()
    }
}

/// The seam [`inkling_core::ops`] names, so that a caller holding a projection
/// does not know whether its weights were ever decoded.
///
/// Infallible where [`PackedProjection::multiply`] is not. The CPU side of the
/// seam cannot fail, and a `Result` on the trait would be one every caller of
/// every projection carries for a case only this one has; a dispatch that does
/// not complete is a panic here for the reason a missing tensor is one in
/// `inkling_core::weights`, that nothing above it can do anything about it.
impl Projection for PackedProjection<'_> {
    fn in_dim(&self) -> usize {
        self.bank.in_dim()
    }

    fn out_dim(&self) -> usize {
        self.bank.out_dim()
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.multiply(x)
            .unwrap_or_else(|err| panic!("the packed matmul did not run: {err}"))
    }
}

/// The kernel, with the format's own constants and element table written into
/// its prelude.
///
/// Generated rather than spelled out because [`inkling_core::quant`] is the
/// authority on every one of them — the nibble order, where the sign lives, the
/// eight magnitudes — and each is a fact read off MLX rather than off the OCP
/// specification. A second copy of that table living in a source string is a
/// copy that can drift from the checkpoint it decodes.
pub(crate) fn source() -> String {
    let elements: Vec<String> = ELEMENTS.iter().map(|value| format!("{value:?}f")).collect();
    format!(
        "\
#include <metal_stdlib>
using namespace metal;

constant uint BITS = {BITS};
constant uint CODE_MASK = {};
constant uint CODES_PER_BYTE = {CODES_PER_BYTE};
constant uint BYTES_PER_GROUP = {BYTES_PER_GROUP};
constant uint BYTES_PER_LANE = {BYTES_PER_LANE};
constant uint EXPONENT_SHIFT = {EXPONENT_SHIFT};
constant float ELEMENTS[] = {{ {} }};
{BODY}",
        (1u32 << BITS) - 1,
        elements.join(", "),
    )
}

/// How many `uint`s the kernel's `Shape` struct declares.
const SHAPE_FIELDS: usize = 9;

/// Everything of the kernel that the format does not decide.
///
/// `weight_dot` is the decode, kept a function of its own because it is the one
/// reading of the format on this side of the engine and a second copy of it
/// would be a second reading that could drift.
pub(crate) const BODY: &str = r#"
/// One output element: lane `l` walks the weight row from byte
/// `l * BYTES_PER_LANE` in strides of that many times the simdgroup width, and
/// the caller reduces what the lanes held.
///
/// A byte is two codes, low nibble first, and its group's scale is a power of
/// two — so every product here is exact and only the order they are summed in
/// separates this from any other way of adding them up.
///
/// The inner loop's trip count is a compile-time constant, so it is unrolled
/// and its loads have no dependency on each other — which is the whole of what
/// a chunk buys.
///
/// **The chunk is bounded by `bytes` and not by anything inside it**, so what
/// keeps a lane inside its own weight row is that a row is a whole number of
/// chunks. `pairs` on the Rust side refuses a width that is not whole groups of
/// 32 codes, which makes every row a whole number of 16 packed bytes, and 4
/// divides 16 — and the same fact is what puts a whole chunk under the one
/// scale byte read for it. A GPU read past a row is an address inside the next
/// one rather than a fault, so this is the invariant to check first if either
/// constant moves.
inline float weight_dot(
    device const uchar *packed,
    device const uchar *scale,
    device const float *values,
    uint bytes,
    uint lane,
    uint width
) {
    float sum = 0.0f;
    for (uint b = lane * BYTES_PER_LANE; b < bytes; b += width * BYTES_PER_LANE) {
        float dot = 0.0f;
        for (uint i = 0; i < BYTES_PER_LANE; ++i) {
            const uint code = packed[b + i];
            const uint low = code & CODE_MASK;
            const uint high = (code >> BITS) & CODE_MASK;
            device const float *v = values + (b + i) * CODES_PER_BYTE;

            dot += ELEMENTS[low] * v[0] + ELEMENTS[high] * v[1];
        }
        sum += dot * as_type<float>(uint(scale[b / BYTES_PER_GROUP]) << EXPONENT_SHIFT);
    }
    return sum;
}

struct Shape {
    uint rows;
    uint in_dim;
    uint out_dim;
    uint per_source;
    uint sources;
    uint code_base;
    uint scale_base;
    uint code_stride;
    uint scale_stride;
};

/// `out[i] = x[(i / per_source) % sources] @ w[experts[i]]^T` over an
/// `[experts, out_dim, in_dim]` bank.
///
/// The expert is per row and not per dispatch, which is the whole of what a
/// gather is: the six of 256 a token chose are six strides into the same bank,
/// and the 250 it did not choose are addresses no thread forms.
///
/// **The divide and the modulo are the other half of that gather**, and they are
/// why the rows a bank runs need not be a tensor anyone built. A token that
/// reads six experts is six rows of this call and one row of `x`, so consecutive
/// rows read the same input; a bank every token reads once per expert is `x`
/// laid end to end after itself, so a row `sources` further on reads the same
/// input again. The routed bank is the first shape and the shared bank the
/// second, and the copy that would have laid either out is two integer
/// operations here.
///
/// A dispatch whose rows are its own input's takes `per_source` of 1 and
/// `sources` of `rows`, which makes the divide and the modulo both the identity.
kernel void packed_matmul(
    constant Shape &shape [[buffer(0)]],
    device const uint *experts [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    device float *out [[buffer(5)]],
    uint position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]
) {
    const uint element = position / width;
    if (element >= shape.rows * shape.out_dim) {
        return;
    }

    const uint row = element / shape.out_dim;
    const uint col = element % shape.out_dim;
    const uint expert = experts[row];
    const uint source = (row / shape.per_source) % shape.sources;
    const uint bytes = shape.in_dim / CODES_PER_BYTE;

    float sum = weight_dot(
        codes + shape.code_base + (ulong)expert * shape.code_stride + (ulong)col * bytes,
        scales + shape.scale_base + (ulong)expert * shape.scale_stride
            + (ulong)col * (bytes / BYTES_PER_GROUP),
        x + (ulong)source * shape.in_dim,
        bytes,
        lane,
        width
    );

    sum = simd_sum(sum);
    if (lane == 0) {
        out[element] = sum;
    }
}
"#;

/// The synthetic weights this module's tests are built from, reachable by
/// [`crate::experts`]'s tests too: a bank is three of these, and a second copy
/// of the packing would be a second reading of the format.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// Codes to a little-endian word, which is the shape [`pack`] writes.
    const CODES_PER_WORD: usize = u32::BITS as usize / BITS;

    /// Codes packed the way the checkpoint packs them: eight to a little-endian
    /// word, code `i` of a word in bits `4i..4i+4`.
    ///
    /// Written as words where the kernel reads bytes, on purpose. That the two
    /// framings describe the same bytes is the claim the byte-addressed decode
    /// rests on, and a test that packed bytewise would assume it rather than
    /// check it.
    pub(crate) fn pack(codes: &[u8]) -> Vec<u8> {
        codes
            .chunks_exact(CODES_PER_WORD)
            .flat_map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |packed, (i, code)| {
                        packed | (u32::from(*code) << (BITS * i))
                    })
                    .to_le_bytes()
            })
            .collect()
    }

    /// A deterministic stand-in for trained weights. Any bit pattern does, so
    /// long as a rerun sees the same one.
    ///
    /// The low eight bits of a linear congruential state are the poorly mixed
    /// ones, so they are shifted off and what is left is 24 bits.
    pub(crate) struct Noise(pub(crate) u32);

    impl Noise {
        pub(crate) fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0 >> 8
        }

        /// A value spread over `-1.0..1.0`.
        ///
        /// The spread is load-bearing rather than cosmetic: against a *constant*
        /// input a row's dot product is its weights summed, which no permutation
        /// of codes within a word can move — so a kernel reading its nibbles
        /// backwards would agree to four digits and the mutation this file
        /// measures would measure nothing.
        pub(crate) fn signed(&mut self) -> f32 {
            self.next() as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    /// One multiply: an `[out_dim, in_dim]` weight held as one code per element
    /// beside one scale byte per group, and the rows of `x` to put through it.
    pub(crate) struct Case {
        pub(crate) in_dim: usize,
        pub(crate) out_dim: usize,
        pub(crate) codes: Vec<u8>,
        pub(crate) scales: Vec<u8>,
        pub(crate) x: Vec<f32>,
    }

    impl Case {
        /// Codes over the whole table and inputs of mixed sign, which is what
        /// makes the reduction cancel the way a trained one does and so what
        /// makes two summation orders part company at all.
        ///
        /// The scales are shaped after the checkpoint's rather than spread over
        /// the byte: `lm_head`'s span `0x74..=0x7e` across the tensor while the
        /// 128 groups *within* a row span a median of one byte and at most four.
        /// That structure is what sets how ill-conditioned the reduction is, and
        /// a synthetic weight whose groups spanned twenty-six powers of two
        /// would be measuring a serial f32 loop falling apart on a case no
        /// checkpoint contains. `0x00` is left out on purpose — it is the one
        /// reading the two sides deliberately disagree about.
        ///
        /// `seed` is what lets a bank be three weights that differ. Against
        /// three identical ones, exchanging two would change nothing.
        pub(crate) fn seeded(seed: u32, in_dim: usize, out_dim: usize, rows: usize) -> Self {
            let mut noise = Noise(seed);
            let groups = in_dim / GROUP_SIZE;
            let mut scales = Vec::with_capacity(out_dim * groups);
            for _ in 0..out_dim {
                let row = 0x74 + (noise.next() % 11) as u8;
                scales.extend((0..groups).map(|_| row + (noise.next() % 5) as u8));
            }

            Self {
                codes: (0..out_dim * in_dim)
                    .map(|_| (noise.next() % 16) as u8)
                    .collect(),
                x: (0..rows * in_dim).map(|_| noise.signed()).collect(),
                scales,
                in_dim,
                out_dim,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::testing::{Case, Noise, pack};
    use super::*;
    use inkling_core::fixture::{self, deviation};
    use inkling_core::ops::DenseProjection;
    use inkling_core::quant::dequantize_blocks;
    use inkling_core::weights::PackedRows;

    use crate::testing::{device, drift};

    /// The reduction the checkpoint's projections are: `lm_head`, every
    /// attention projection and every expert reduce over 4096.
    const IN_DIM: usize = 4096;

    /// Output elements enough to span many threadgroups without filling the last
    /// one, and prime so that no divisor of the dispatch happens to line up with
    /// it.
    const OUT_DIM: usize = 257;

    /// How far a dispatch may land from the CPU's answer for the same weights.
    ///
    /// Decoding is exact on both sides — a table lookup times a power of two —
    /// and neither rounds anywhere else, so summation order is the whole of what
    /// separates them. **This bound is therefore not a bound on the kernel; it
    /// is a bound on the oracle.** The CPU adds 4096 products serially, whose
    /// drift grows like the square root of the reduction: 64 ulps at this
    /// length, and an f32 ulp is 6e-8, so 3.8e-6 of the tensor's peak is what a
    /// serial f32 loop is expected to give up. The kernel sums 128 a lane and
    /// reduces 32 lanes in a tree, which is the better-conditioned order by a
    /// factor of the same shape.
    ///
    /// Measured against an f64 accumulation of the same products, that is
    /// exactly what happens: the kernel drifts 1.4e-7 — under three ulps — where
    /// the CPU drifts 2.8e-6, and the 2.8e-6 they disagree by is the CPU's own
    /// error arriving whole. 6e-6 admits that with a factor of two in hand.
    ///
    /// Which is why the assertion beside it is the one with teeth: the kernel
    /// has to be *closer to exact* than the CPU, not merely inside a bound. A
    /// dispatch that decoded something wrongly would fail that while a widened
    /// tolerance would still let it through — and the weakest mutation this has
    /// to catch, a kernel reading each word's nibbles from the top down, lands
    /// at 8.1e-1, five decades above.
    const TOLERANCE: f32 = 6e-6;

    /// The eight E2M1 magnitudes, written out here rather than read off the
    /// table the kernel is built from, so that a case computed by hand is
    /// computed independently of what it checks.
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    /// What a code under a scale byte stands for, from the format rather than
    /// from a decoder: bit 3 carries the sign, the three below it index the
    /// magnitudes, and the byte is an exponent biased by 127.
    fn element(code: u8, scale: u8) -> f32 {
        let magnitude = MAGNITUDES[usize::from(code & 7)];
        let signed = if code & 8 == 0 { magnitude } else { -magnitude };
        signed * f32::from_bits(u32::from(scale) << EXPONENT_SHIFT)
    }

    /// One multiply: an `[out_dim, in_dim]` weight held as one code per element
    /// beside one scale byte per group, and the rows of `x` to put through it.
    impl Case {
        /// Codes over the whole table and inputs of mixed sign, which is what
        /// makes the reduction cancel the way a trained one does and so what
        /// makes two summation orders part company at all.
        ///
        /// The scales are shaped after the checkpoint's rather than spread over
        /// the byte: `lm_head`'s span `0x74..=0x7e` across the tensor while the
        /// 128 groups *within* a row span a median of one byte and at most four.
        /// That structure is what sets how ill-conditioned the reduction is, and
        /// a synthetic weight whose groups spanned twenty-six powers of two
        /// would be measuring a serial f32 loop falling apart on a case no
        /// checkpoint contains. `0x00` is left out on purpose — it is the one
        /// reading the two sides deliberately disagree about, and
        /// [`a_zero_scale_byte_multiplies_to_zero_where_the_cpu_gives_two_to_the_minus_127`]
        /// is where that is stated.
        fn noisy(in_dim: usize, out_dim: usize, rows: usize) -> Self {
            Self::seeded(0x1234_5678, in_dim, out_dim, rows)
        }

        fn packed(&self) -> Vec<u8> {
            pack(&self.codes)
        }

        fn upload<'a>(&self, device: &'a Device, matmul: &'a PackedMatmul) -> PackedProjection<'a> {
            PackedProjection::upload(
                device,
                matmul,
                self.in_dim,
                self.out_dim,
                &self.packed(),
                &self.scales,
            )
            .expect("the case's shapes pair")
        }

        /// The same multiply through the decoder and the CPU projection this
        /// kernel exists to replace, which is the oracle for everything below.
        fn on_the_cpu(&self) -> Vec<f32> {
            self.on_the_cpu_with(&self.packed(), &self.scales)
        }

        /// [`Case::on_the_cpu`] over bytes of the caller's choosing, which is
        /// how a mutation is measured against the same machinery.
        fn on_the_cpu_with(&self, packed: &[u8], scales: &[u8]) -> Vec<f32> {
            let weight = dequantize_blocks(packed, scales).expect("the case decodes");
            DenseProjection::new(self.in_dim, &weight).forward(&self.x)
        }

        /// The same multiply summed in f64, which neither side does.
        ///
        /// Decoding is exact on both sides — a table lookup times a power of two
        /// — so the products are the same f32s either way and summation order is
        /// the only thing left to differ about. Accumulating those products with
        /// 29 bits of headroom settles which of the two orders is drifting,
        /// which is what turns a disagreement into either float noise or a bug.
        fn exactly(&self) -> Vec<f64> {
            let weight = dequantize_blocks(&self.packed(), &self.scales).expect("the case decodes");
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

    /// What the bandwidth column divides by, against what the kernel reads.
    ///
    /// **A row is a whole weight and the weight is never decoded**, which are
    /// the two things this figure has to get right: every element of an output
    /// row reads a different slice of the expert that row goes through, and what
    /// it reads is packed — half a byte a code, plus one scale byte for every
    /// 32 of them. A figure that charged the decoded float32 would be eight
    /// times high and would put this kernel past the machine's bandwidth
    /// rather than at a third of it.
    #[test]
    fn a_dispatch_declares_the_packed_bytes_it_reads_rather_than_the_values() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const IN_DIM: usize = 128;
        const OUT_DIM: usize = 8;
        const ROWS: usize = 3;
        let case = Case::seeded(1, IN_DIM, OUT_DIM, ROWS);
        let projection = case.upload(&device, &matmul);

        let moved = crate::testing::moved(&device, |batch| {
            projection
                .encode(batch, &case.x)
                .expect("the dispatch encodes");
        });

        let elements = ROWS * OUT_DIM * IN_DIM;
        assert_eq!(
            moved as usize,
            elements / 2
                + elements / GROUP_SIZE
                + size_of::<f32>() * (ROWS * IN_DIM + ROWS * OUT_DIM)
                + size_of::<u32>() * ROWS,
            "codes, scales, the rows in, the rows out, and an expert a row"
        );
        assert!(
            (moved as usize) < elements * size_of::<f32>(),
            "a decoded weight was charged for one nothing decodes"
        );
    }

    /// Everything a dispatch needs, so that no test opens a device twice.
    fn matmul(device: &Device) -> PackedMatmul {
        PackedMatmul::new(device).expect("the packed matmul compiles")
    }

    /// The smallest claim there is, and the one every other test here assumes:
    /// that a code times its group's scale times an input is what lands in the
    /// output.
    ///
    /// Exact rather than bounded, and that is affordable rather than lucky: every
    /// magnitude is a dyadic of three significant bits, the inputs are small
    /// integers, and the scales are powers of two either side of one, so every
    /// product and every partial sum is representable and no ordering can move a
    /// bit. A tolerance here would only be hiding a plumbing mistake.
    ///
    /// The two rows carry the same codes in opposite order under different
    /// scales, so a kernel that read one row's scale for both, or that indexed
    /// the codes from the wrong end, produces the other row's answer rather than
    /// a near miss.
    #[test]
    fn a_dispatch_multiplies_what_the_codes_and_their_scale_stand_for() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);

        let forwards: Vec<u8> = (0..GROUP_SIZE).map(|i| (i % 16) as u8).collect();
        let backwards: Vec<u8> = forwards.iter().rev().copied().collect();
        let case = Case {
            in_dim: GROUP_SIZE,
            out_dim: 2,
            codes: [forwards.clone(), backwards.clone()].concat(),
            scales: vec![0x7f, 0x80],
            x: (0..GROUP_SIZE).map(|i| i as f32 + 1.0).collect(),
        };

        let want: Vec<f32> = [(&forwards, 0x7f), (&backwards, 0x80)]
            .into_iter()
            .map(|(codes, scale)| {
                codes
                    .iter()
                    .zip(&case.x)
                    .map(|(code, x)| element(*code, scale) * x)
                    .sum()
            })
            .collect();
        assert_ne!(want[0], want[1], "two rows that agreed would prove nothing");

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(got, want);
    }

    /// The kernel against the CPU it replaces, over the reduction length and the
    /// spread of codes and scales the checkpoint actually holds.
    ///
    /// The dispatch is deliberately ragged: 257 outputs over three rows is 771
    /// simdgroups, which is neither a whole number of threadgroups nor a whole
    /// number of anything else, so the tail group runs lanes past the end of the
    /// work and the bounds check is what stops them writing.
    #[test]
    fn the_kernel_reproduces_the_cpu_over_synthetic_packed_weights() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 3);

        let projection = case.upload(&device, &matmul);
        let width = matmul.kernel.simd_width();
        let elements = 3 * OUT_DIM;
        assert!(
            elements * width % THREADS_PER_GROUP != 0,
            "a dispatch that filled its last threadgroup would not exercise the bounds check"
        );

        let got = projection
            .multiply(&case.x)
            .expect("the dispatch completes");
        let on_the_cpu = case.on_the_cpu();
        let deviation = deviation(&got, &on_the_cpu);
        assert!(
            deviation > 0.0,
            "an exact match would mean the two are not summing independently"
        );

        // Which of the two is drifting, which is what says whether a
        // disagreement of this size is float noise or a bug. The kernel sums 128
        // products a lane and then reduces 32 lanes in a tree; the CPU sums 4096
        // serially. The tree is the better-conditioned order, so the kernel has
        // to be the *closer* of the two to the exact answer — a kernel that was
        // merely within the bound while sitting further out than a serial f32
        // loop would be one hiding a mistake inside a tolerance.
        let exact = case.exactly();
        let (mine, theirs) = (drift(&got, &exact), drift(&on_the_cpu, &exact));
        eprintln!(
            "synthetic weights: deviation {deviation:e}, drift from exact {mine:e} against the \
             CPU's {theirs:e}"
        );
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
        assert!(mine < theirs, "{mine:e} against the CPU's {theirs:e}");
    }

    /// The nibble order, which is the one fact about the format a kernel can get
    /// backwards while still producing plausible weights of the right magnitude.
    ///
    /// Stated as a kernel rather than as a mutated input, because that is where
    /// the mistake would live: the same source, the same dispatch, the same
    /// bytes, and each byte's two codes taken high nibble first.
    #[test]
    fn reading_each_bytes_nibbles_the_other_way_round_is_a_different_answer() {
        let Some(device) = device() else { return };
        let case = Case::noisy(IN_DIM, OUT_DIM, 1);

        let reversed = source().replace(
            "ELEMENTS[low] * v[0] + ELEMENTS[high] * v[1]",
            "ELEMENTS[high] * v[0] + ELEMENTS[low] * v[1]",
        );
        assert_ne!(reversed, source(), "the mutation changed nothing");
        let mutant = PackedMatmul::from_source(&device, &reversed).expect("the mutant compiles");

        let got = case
            .upload(&device, &mutant)
            .multiply(&case.x)
            .expect("the dispatch completes");
        let deviation = deviation(&got, &case.on_the_cpu());
        eprintln!("nibbles read the other way round: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// A group's scale is its own, and a kernel that read one per weight row —
    /// or that took them an index out — would agree with everything above on a
    /// weight whose groups happened to share a scale. Exchanging two adjacent
    /// scale bytes has to move the answer.
    #[test]
    fn each_group_multiplies_under_its_own_scale() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 1);

        // Inside one weight row, not across two of them: the scales are laid out
        // a row at a time, and a swap that straddled the boundary would be
        // exchanging two rows' scales rather than two groups' — which is a
        // different mistake, and one the row indexing already answers for.
        let groups = IN_DIM / GROUP_SIZE;
        let mut swapped = case.scales.clone();
        let boundary = (0..swapped.len() - 1)
            .find(|i| i % groups != groups - 1 && swapped[*i] != swapped[i + 1])
            .expect("some two adjacent groups of one row differ in scale");
        swapped.swap(boundary, boundary + 1);

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        let deviation = deviation(&got, &case.on_the_cpu_with(&case.packed(), &swapped));
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// The licensed divergence, stated where it can be seen rather than left in
    /// a comment. `inkling_core::quant` reads `0x00` as `2^-127`, which is
    /// MLX's own reading and what the CPU path is pinned to; this kernel shifts
    /// the byte into the exponent field and so reads it as zero.
    ///
    /// Both are safe on this checkpoint because `0x00` appears only against
    /// all-zero codes, where the readings agree. A group with nonzero codes
    /// under it is what tells them apart, and it takes a synthetic weight to
    /// build one.
    #[test]
    fn a_zero_scale_byte_multiplies_to_zero_where_the_cpu_gives_two_to_the_minus_127() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);

        let case = Case {
            in_dim: GROUP_SIZE,
            out_dim: 1,
            codes: vec![7; GROUP_SIZE],
            scales: vec![0x00],
            x: vec![1.0; GROUP_SIZE],
        };

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(got, [0.0]);

        // The same group under the CPU's reading: nonzero, and thirty decades
        // below any weight a checkpoint carries. That gap is the whole of what
        // the shift throws away.
        let on_the_cpu = case.on_the_cpu()[0];
        assert!(on_the_cpu > 0.0 && on_the_cpu < 1e-30, "{on_the_cpu:e}");
    }

    /// Rows of `x` are independent, and each gets its own row of the output at
    /// its own offset. A kernel that took the row index off the wrong axis would
    /// still fill the buffer.
    #[test]
    fn every_row_of_the_input_gets_its_own_row_of_the_output() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 2);
        let projection = case.upload(&device, &matmul);

        let both = projection
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(both.len(), 2 * OUT_DIM);
        for (row, x) in case.x.chunks_exact(IN_DIM).enumerate() {
            let alone = projection.multiply(x).expect("the dispatch completes");
            assert_eq!(both[row * OUT_DIM..][..OUT_DIM], alone[..], "row {row}");
        }
    }

    /// A checkpoint's packed tensor read where it is mapped, cut where the
    /// vocabulary ends — and answering what the CPU makes of the same rows.
    ///
    /// [`PackedRows`] is the oracle rather than a decoded tensor here on
    /// purpose: it is the projection the engine runs today, so what this states
    /// is that exchanging one for the other is a change of backend and not a
    /// change of answer.
    ///
    /// It is also where the misalignment is met on real bytes. This tensor sits
    /// one byte past a word in its file, like every other tensor in the quant,
    /// so a kernel that wanted words could not be pointed at it at all — and the
    /// deviation below is what says reading it a byte at a time decodes the same
    /// weight.
    ///
    /// The rows past the cut are the assertion's other half. They decode to
    /// exactly 0.0 — [`inkling_core::head`] is where that matters — so a
    /// dispatch that quietly ran the whole tensor would still agree on the 32
    /// rows it was asked about, and only the length says otherwise.
    #[test]
    fn a_checkpoints_packed_tensor_runs_the_rows_it_was_cut_to() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let ckpt = fixture::open(fixture::MXFP4);
        let packed =
            Packed::open(&ckpt, fixture::VOCAB_PADDING).expect("the fixture holds the slice");
        assert_ne!(
            packed.prefix(1).0.as_ptr() as usize % size_of::<u32>(),
            0,
            "a word-aligned fixture would not exercise the reading this needs"
        );

        let mut noise = Noise(0x0f1e_2d3c);
        let x: Vec<f32> = (0..2 * packed.slice_len())
            .map(|_| noise.signed())
            .collect();
        let got =
            PackedProjection::wrap_packed(&device, &matmul, &packed, fixture::VOCAB_PADDING_ROWS)
                .expect("the cut tensor wraps")
                .multiply(&x)
                .expect("the dispatch completes");
        assert_eq!(
            got.len(),
            2 * fixture::VOCAB_PADDING_ROWS,
            "the padding rows were dispatched over"
        );

        let want = PackedRows::new(packed, fixture::VOCAB_PADDING_ROWS).forward(&x);
        let deviation = deviation(&got, &want);
        eprintln!(
            "{} rows of {}: deviation {deviation:e}",
            fixture::VOCAB_PADDING_ROWS,
            fixture::VOCAB_PADDING
        );
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    /// The gather, against the projections it is a gather over.
    ///
    /// A `[3, OUT_DIM, IN_DIM]` bank is three `[OUT_DIM, IN_DIM]` weights laid
    /// end to end, so what a gathered dispatch has to produce is exactly what
    /// three separate projections produce over the rows that named them — and
    /// that is the whole claim, because the arithmetic inside is the same kernel
    /// either way and only the address it forms differs.
    ///
    /// The list repeats an expert and skips one, which is what a decode step's
    /// routing does: three of the four rows here go through expert 2 and none
    /// goes through expert 1.
    ///
    /// Its two axes are then separated, which is the mistake worth catching. A
    /// kernel that read the expert off the dispatch, or the input row off the
    /// expert, would still fill the buffer with plausible numbers — so three of
    /// the rows carry the same input under different experts and the fourth
    /// carries a different input under a repeated one, and each pair says which
    /// index moved.
    #[test]
    fn a_gathered_dispatch_multiplies_each_row_against_the_expert_it_named() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const EXPERTS: usize = 3;

        let case = Case::noisy(IN_DIM, EXPERTS * OUT_DIM, 2);
        let bank = PackedBank::upload(
            &device,
            &matmul,
            EXPERTS,
            IN_DIM,
            OUT_DIM,
            &case.packed(),
            &case.scales,
        )
        .expect("the bank's shapes pair");

        let chosen = [2u32, 0, 2, 2];
        let inputs = [0usize, 0, 0, 1];
        let x: Vec<f32> = inputs
            .iter()
            .flat_map(|row| case.x[row * IN_DIM..][..IN_DIM].to_vec())
            .collect();
        let got = bank.multiply(&chosen, &x).expect("the dispatch completes");
        assert_eq!(got.len(), chosen.len() * OUT_DIM);

        let codes_per_expert = OUT_DIM * IN_DIM / CODES_PER_BYTE;
        let scales_per_expert = OUT_DIM * IN_DIM / GROUP_SIZE;
        let packed = case.packed();
        for (row, (expert, input)) in chosen.iter().zip(inputs).enumerate() {
            let at = *expert as usize;
            let alone = PackedProjection::upload(
                &device,
                &matmul,
                IN_DIM,
                OUT_DIM,
                &packed[at * codes_per_expert..][..codes_per_expert],
                &case.scales[at * scales_per_expert..][..scales_per_expert],
            )
            .expect("one expert's shapes pair")
            .multiply(&case.x[input * IN_DIM..][..IN_DIM])
            .expect("the dispatch completes");
            assert_eq!(got[row * OUT_DIM..][..OUT_DIM], alone[..], "row {row}");
        }

        let row = |i: usize| &got[i * OUT_DIM..][..OUT_DIM];
        assert_eq!(row(0), row(2), "one expert, one input, twice");
        assert_ne!(row(0), row(1), "the expert index is read off the row");
        assert_ne!(row(0), row(3), "the input row is read off the row");
    }

    /// The same gather with the input read over again rather than at a stride,
    /// which is the shared bank's shape: every row of `x` through every expert
    /// the list names, expert-major.
    ///
    /// Three passes over three rows rather than two over two, because two of
    /// each cannot tell a modulo from the division beside it — `row % 2` and
    /// `row / 2` disagree on only one of four rows, and both are the identity on
    /// a call that wraps once. Nine rows over three make the two readings
    /// disagree everywhere but the corners.
    ///
    /// Each row is checked against the projection its expert is, run alone over
    /// the input row the modulo says it reads. A kernel that divided instead
    /// would put row 3's answer — the second pass's first row — in the wrong
    /// place, with numbers of exactly the right magnitude.
    #[test]
    fn a_repeating_dispatch_reads_every_row_of_its_input_once_per_expert() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const EXPERTS: usize = 3;
        const SOURCES: usize = 3;

        let case = Case::noisy(IN_DIM, EXPERTS * OUT_DIM, SOURCES);
        let bank = PackedBank::upload(
            &device,
            &matmul,
            EXPERTS,
            IN_DIM,
            OUT_DIM,
            &case.packed(),
            &case.scales,
        )
        .expect("the bank's shapes pair");

        // What `SparseMoe::shared_rows` names: expert `row / sources` over input
        // row `row % sources`.
        let chosen: Vec<u32> = (0..EXPERTS * SOURCES)
            .map(|row| (row / SOURCES) as u32)
            .collect();
        let x = &case.x[..SOURCES * IN_DIM];

        let mut batch = device.batch().expect("a command buffer opens");
        let mut input = device.buffer(x).expect("the input uploads");
        let got = bank
            .encode_repeating(&mut batch, &chosen, &mut input)
            .expect("the dispatch encodes");
        batch.wait().expect("the dispatch completes");
        let got = got.take();
        assert_eq!(got.len(), chosen.len() * OUT_DIM);

        let codes_per_expert = OUT_DIM * IN_DIM / CODES_PER_BYTE;
        let scales_per_expert = OUT_DIM * IN_DIM / GROUP_SIZE;
        let packed = case.packed();
        for (row, expert) in chosen.iter().enumerate() {
            let at = *expert as usize;
            let alone = PackedProjection::upload(
                &device,
                &matmul,
                IN_DIM,
                OUT_DIM,
                &packed[at * codes_per_expert..][..codes_per_expert],
                &case.scales[at * scales_per_expert..][..scales_per_expert],
            )
            .expect("one expert's shapes pair")
            .multiply(&x[(row % SOURCES) * IN_DIM..][..IN_DIM])
            .expect("the dispatch completes");
            assert_eq!(got[row * OUT_DIM..][..OUT_DIM], alone[..], "row {row}");
        }

        let row = |i: usize| &got[i * OUT_DIM..][..OUT_DIM];
        assert_ne!(row(0), row(1), "the input row is read off the row");
        assert_ne!(row(0), row(3), "the expert is read off the row");
    }

    /// The one argument a real call sends past the inline threshold.
    ///
    /// A shape is a few dozen bytes and always travels in the command buffer;
    /// an expert list is one `uint` a row, so a decode step's six do and a
    /// prefill's do not — 4614 of them for a 769-token prompt. This drives 1025,
    /// one past the 1024 that fill the 4 KiB `setBytes:` takes, so the fallback
    /// is exercised through the caller that reaches it rather than through a
    /// kernel of the test's own.
    ///
    /// Every row's answer is checked against the expert it named run alone, and
    /// the two experts disagree — so a list read short, truncated at the
    /// threshold or bound at the wrong length lands the wrong expert's answer in
    /// the tail rather than a near miss.
    #[test]
    fn a_gathered_dispatch_takes_an_expert_list_too_wide_for_the_command_buffer() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const EXPERTS: usize = 2;
        const NARROW_OUT: usize = 3;
        const ROWS: usize = 1025;

        let case = Case::noisy(GROUP_SIZE, EXPERTS * NARROW_OUT, EXPERTS);
        let bank = PackedBank::upload(
            &device,
            &matmul,
            EXPERTS,
            GROUP_SIZE,
            NARROW_OUT,
            &case.packed(),
            &case.scales,
        )
        .expect("the bank's shapes pair");

        let chosen: Vec<u32> = (0..ROWS).map(|row| (row % EXPERTS) as u32).collect();
        let x: Vec<f32> = chosen
            .iter()
            .flat_map(|expert| case.x[*expert as usize * GROUP_SIZE..][..GROUP_SIZE].to_vec())
            .collect();

        let got = bank.multiply(&chosen, &x).expect("the dispatch completes");

        assert_eq!(got.len(), ROWS * NARROW_OUT);
        let codes_per_expert = NARROW_OUT * GROUP_SIZE / CODES_PER_BYTE;
        let packed = case.packed();
        let alone: Vec<Vec<f32>> = (0..EXPERTS)
            .map(|at| {
                PackedProjection::upload(
                    &device,
                    &matmul,
                    GROUP_SIZE,
                    NARROW_OUT,
                    &packed[at * codes_per_expert..][..codes_per_expert],
                    &case.scales[at * NARROW_OUT..][..NARROW_OUT],
                )
                .expect("one expert's shapes pair")
                .multiply(&case.x[at * GROUP_SIZE..][..GROUP_SIZE])
                .expect("the dispatch completes")
            })
            .collect();
        assert_ne!(
            alone[0], alone[1],
            "two experts that agreed would prove nothing"
        );

        for (row, expert) in chosen.iter().enumerate() {
            assert_eq!(
                got[row * NARROW_OUT..][..NARROW_OUT],
                alone[*expert as usize][..],
                "row {row}"
            );
        }
    }

    /// An index past the bank is an offset past the buffer, and a GPU read
    /// answers that with whatever is there rather than with a fault — so it has
    /// to be refused on this side, where the bank's length is known.
    #[test]
    fn a_row_that_names_an_expert_the_bank_does_not_hold_is_refused() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(GROUP_SIZE, 2 * 4, 1);
        let bank = PackedBank::upload(
            &device,
            &matmul,
            2,
            GROUP_SIZE,
            4,
            &case.packed(),
            &case.scales,
        )
        .expect("the bank's shapes pair");

        assert!(bank.multiply(&[1], &case.x).is_ok(), "the last expert");
        assert!(
            matches!(
                bank.multiply(&[2], &case.x),
                Err(MatmulError::NoSuchExpert {
                    expert: 2,
                    experts: 2
                })
            ),
            "one past the last"
        );
    }

    /// Two multiplies against *one* bank in one command buffer, of different
    /// heights — which is what says a dispatch's shape belongs to the dispatch.
    ///
    /// This is the case the shape buffer was moved out of [`Resident`] for. Held
    /// per bank, the second call's row count would overwrite the first's before
    /// either had run, and both dispatches would read the second.
    ///
    /// The taller call is encoded first, which is what makes that visible: the
    /// kernel's `element >= rows * out_dim` check would then cull two thirds of
    /// its output against the shorter call's row count and leave it zeroed,
    /// where a shorter call reading a taller one's shape agrees by accident —
    /// the elements it was dispatched for are inside the larger bound either
    /// way.
    #[test]
    fn two_multiplies_against_one_bank_in_one_batch_keep_their_own_shapes() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 3);
        let projection = case.upload(&device, &matmul);
        let (one, three) = (&case.x[..IN_DIM], &case.x[..]);

        let [batched_three, batched_one] = together(&device, |batch| {
            Ok([
                projection.encode(batch, three)?,
                projection.encode(batch, one)?,
            ])
        })
        .expect("both dispatches complete");

        assert_eq!(batched_three.len(), 3 * OUT_DIM);
        assert_eq!(batched_one.len(), OUT_DIM, "one row in, one row out");
        assert_eq!(
            batched_three,
            projection.multiply(three).expect("the dispatch completes"),
            "the shorter call's shape reached the taller one"
        );
        assert_eq!(
            batched_one,
            projection.multiply(one).expect("the dispatch completes")
        );
    }

    /// Rows a dispatch left on the device multiply to what the same rows handed
    /// over as a slice do — and one buffer feeds two dispatches of one command
    /// buffer.
    ///
    /// The seam `LayerProjections::normed_qkvr` rests on, stated on the matmul
    /// alone. Four projections read one normed hidden state there, so what has
    /// to hold is both halves of this: that a buffer is the same input a copy
    /// would have been, and that binding it to a second dispatch of the same
    /// batch does not disturb the first — which is what a caller has to believe
    /// when `Buffer::arg` says a binding is exclusive.
    #[test]
    fn a_multiply_over_a_device_buffer_answers_what_the_same_rows_as_a_slice_do() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 2);
        let projection = case.upload(&device, &matmul);
        let mut rows = device.buffer(&case.x).expect("the rows upload");

        let [first, second] = together(&device, |batch| {
            Ok([
                projection.encode_over(batch, &mut rows)?,
                projection.encode_over(batch, &mut rows)?,
            ])
        })
        .expect("both dispatches complete");

        let want = projection
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(first.len(), 2 * OUT_DIM);
        assert_eq!(first, want, "a buffer against the slice it was filled from");
        assert_eq!(second, want, "the second dispatch read a disturbed buffer");
    }

    /// A batch that happens to be empty is the caller's business rather than an
    /// error, and it cannot become a dispatch: the device refuses a zero-length
    /// buffer, so an output of nothing has to be answered without allocating one.
    #[test]
    fn no_rows_of_input_produce_no_output() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(GROUP_SIZE, 2, 0);

        assert_eq!(
            case.upload(&device, &matmul)
                .multiply(&[])
                .expect("an empty multiply completes"),
            Vec::<f32>::new()
        );
    }

    /// **What a packed multiply costs the device at each width a lane reads**,
    /// and the sweep [`BYTES_PER_LANE`] was chosen from.
    ///
    /// The shapes are the ones a step dispatches rather than round numbers: a
    /// decode step's six routed rows of a `[2048, 4096]` bank, the `[4096, 4096]`
    /// and `[1024, 4096]` projections beside them, and the same two at the row
    /// counts a 97-token prefill gives them. What the table shows is that the
    /// width that wins depends on how many output elements the dispatch has —
    /// which is the same shape of finding `dense_matmul`'s reduction width is,
    /// and the reason the constant is weighed against the shapes that carry a
    /// step's bytes rather than against the fastest row.
    ///
    /// Nothing asserts a rate. The numbers go to stderr for the commit message
    /// to quote, and what is asserted is that the shipped width was among the
    /// ones tried.
    ///
    /// Read off the device's own clock over a command buffer of `CALLS`
    /// dispatches, because a submission is 225 microseconds and most of these
    /// are under a hundred.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_width_a_lane_reads() {
        let Some(device) = device() else { return };
        const CALLS: usize = 16;
        const ROUNDS: usize = 3;
        const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];

        // `(what, in_dim, out_dim, rows)`.
        let shapes = [
            ("routed gate/up, decode", 4096, 2048, 6),
            ("routed down, decode", 2048, 4096, 6),
            ("q_proj, decode", 4096, 4096, 1),
            ("k_proj, decode", 4096, 1024, 1),
            ("shared gate/up, decode", 4096, 2048, 2),
            ("routed gate/up, 97 tokens", 4096, 2048, 6 * 97),
            ("q_proj, 97 tokens", 4096, 4096, 97),
        ];

        assert!(WIDTHS.contains(&BYTES_PER_LANE), "{BYTES_PER_LANE}");
        eprintln!(
            "  {:<28}{}",
            "shape",
            WIDTHS
                .iter()
                .map(|width| format!("{:>10}", format!("{width} B/lane")))
                .collect::<String>()
        );
        for (what, in_dim, out_dim, rows) in shapes {
            let case = Case::seeded(1, in_dim, out_dim, rows);
            let mut x = device.buffer(&case.x).expect("the rows upload");
            let mut line = format!("  {what:<28}");
            for width in WIDTHS {
                let matmul = PackedMatmul::from_source(&device, &a_lane_reading(width))
                    .unwrap_or_else(|err| panic!("{width} bytes a lane compiles: {err}"));
                let projection = case.upload(&device, &matmul);

                let mut best = Duration::MAX;
                for _ in 0..ROUNDS {
                    let mut batch = device.batch().expect("a command buffer opens");
                    for _ in 0..CALLS {
                        projection
                            .encode_over(&mut batch, &mut x)
                            .expect("the dispatch encodes");
                    }
                    profile::take();
                    batch.wait().expect("the batch completes");
                    best = best.min(profile::take().gpu() / CALLS as u32);
                }

                let codes = rows * out_dim * in_dim;
                let moved = (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
                line.push_str(&format!(
                    "{:>10}",
                    format!("{:.0} GB/s", moved / best.as_secs_f64() / 1e9)
                ));
            }
            eprintln!("{line}");
        }
    }

    /// The shipped kernel with a different [`BYTES_PER_LANE`] written into its
    /// prelude, which is the one thing the sweep above varies.
    ///
    /// The declaration is named in full on both sides of the rewrite, and the
    /// assertion asks for the whole of it: the prelude holds four other `uint`
    /// constants and three of them are 2, 4 and 16, so a check for the value
    /// alone would pass against `CODES_PER_BYTE` while the sweep measured one
    /// width five times.
    fn a_lane_reading(bytes_per_lane: usize) -> String {
        let wanted = format!("constant uint BYTES_PER_LANE = {bytes_per_lane};");
        let source = source().replace(
            &format!("constant uint BYTES_PER_LANE = {BYTES_PER_LANE};"),
            &wanted,
        );
        assert!(source.contains(&wanted), "the prelude declares {wanted}");
        source
    }

    /// A weight paired with another tensor's scales is the mistake the shapes
    /// exist to catch, and it has to be caught on the way in: the kernel takes
    /// its bounds from the shape it was told and would read off the end of
    /// whichever buffer was short.
    #[test]
    fn a_weight_and_scales_that_do_not_pair_are_refused() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(GROUP_SIZE, 4, 1);
        let upload = |in_dim, out_dim, codes: &[u8], scales: &[u8]| {
            PackedProjection::upload(&device, &matmul, in_dim, out_dim, codes, scales)
                .expect_err("the shapes do not pair")
        };

        let (packed, scales) = (case.packed(), case.scales.clone());
        assert!(
            matches!(
                upload(GROUP_SIZE, 4, &packed[..packed.len() - 4], &scales),
                MatmulError::WrongCodeLen { expected: 64, .. }
            ),
            "short codes"
        );
        assert!(
            matches!(
                upload(GROUP_SIZE, 4, &packed, &scales[..3]),
                MatmulError::WrongScaleLen {
                    expected: 4,
                    got: 3,
                    ..
                }
            ),
            "short scales"
        );
        assert!(
            matches!(
                upload(GROUP_SIZE / 2, 8, &packed, &scales),
                MatmulError::PartialGroup(16)
            ),
            "a width that is not whole groups"
        );
    }
}
