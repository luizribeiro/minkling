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
//! than at a thirty-second of it.
//!
//! **Or one simdgroup per [`ROWS_A_TILE`] rows of it by [`COLS_A_TILE`]
//! columns, where the rows share an expert.** The arrangement above reads the
//! weight once per row of `x`, which is free at a decode step's single row and
//! is the whole of a prefill's cost at hundreds: 385 tokens through this kernel
//! moved the same bytes a token that 385 decode steps would have.
//! `packed_matmul_rows` is the same walk with the sums of a tile carried through
//! it — the rows to share a weight read, the columns to share the input read
//! that sharing the weight then left it waiting on — and the two are separate
//! entries out of one source so that the shape with nothing to win keeps the
//! register budget it had.
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
use crate::grouping::Grouped;
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "packed_matmul";

/// The second entry point of the same source: one simdgroup per *tile* of rows
/// rather than per output element, so a weight row read once serves every row
/// of the tile that named the same expert.
const TILED_ENTRY: &str = "packed_matmul_rows";

/// The third: the same tile, over rows a dispatch laid out expert by expert.
///
/// **A routed bank's rows are the one layout the tile above cannot reach**, and
/// they are 59% of a prefill's bytes. A token reads six experts, so its six rows
/// name six different weights and no two consecutive rows of a call ever share
/// one — which is true however long the prompt. What this entry adds is an
/// indirection at each end: [`crate::grouping`] writes the order the rows would
/// be in if they were sorted by expert, and this reads its input through that
/// order or writes its output through it, so the tile in between sees a run of
/// rows that do name one weight.
const GROUPED_ENTRY: &str = "packed_matmul_grouped";

/// Rows one simdgroup of [`TILED_ENTRY`] multiplies against one weight row.
///
/// **What this is worth is the whole of the prefill gap and none of the decode
/// step.** A row of this kernel is a whole weight — see [`PackedBank::moves`] —
/// so a call of `n` rows reads its weight `n` times, and a prefill measured
/// 5496 MB a token at 385 tokens and at 769, against the 5495 MB a *decode*
/// step moves. It amortised nothing at all. A tile of this many rows reads the
/// weight once for all of them and multiplies it against each, which is the
/// same weight traffic a decode step pays for that many tokens' worth of work.
///
/// **It is the rows and not the whole call**, because only some of a prefill's
/// rows can share a read: the ones naming the same expert. A projection's rows
/// all name expert zero and a shared bank's name one of two, which is 40.8% of
/// what a prefill reads; the routed bank's six rows a token are six different
/// experts by construction, and getting at that 59.1% means moving rows rather
/// than tiling them.
///
/// **Four, and the sweep turns hard on either side of it.** Every shape a
/// prefill gives this kernel is fastest here and six is already slower than
/// two: a tile carries a running sum and an input offset a row, and past four
/// of each the occupancy that buys them costs more than the reads it saves.
/// That is the same shape of finding [`BYTES_PER_LANE`] and `dense_matmul`'s
/// reduction width both are, met a third time — see
/// `what_a_packed_multiply_costs_at_each_height_a_tile_reads`.
const ROWS_A_TILE: usize = 4;

/// Output columns one simdgroup of [`TILED_ENTRY`] computes beside each other,
/// against one read of the input.
///
/// **What this is worth is the ratio the row tile left behind.** A tile of
/// [`ROWS_A_TILE`] rows over one column reads a packed byte and multiplies its
/// two codes against that many rows of input — so per byte of weight it reads
/// `8 * ROWS_A_TILE` input floats, which at four rows is 32 bytes of input for
/// every byte the dispatch is charged. The row tile stopped waiting on the
/// weight and started waiting on that. A tile that is also this many columns
/// wide reads that many weight bytes against the same input floats, so the
/// ratio falls by this factor and the loads the input costs are divided between
/// the columns that wanted them.
///
/// **It shares no weight byte and is not meant to.** Every output column is its
/// own weight row, so what a column tile moves is exactly what the same columns
/// moved apart — [`PackedBank::moves`] does not mention it, and the figure a
/// prefill declares does not change. What changes is the time, which is the
/// whole of the difference between this and the row tile above.
///
/// **Four, and the sweep turns as hard here as it does on the other axis** —
/// see `what_a_packed_multiply_costs_at_each_width_a_tile_spans`. Every shape a
/// prefill gives this kernel gets faster at every width up to four and gives it
/// all back at eight, which is slower than one column. A tile carries a running
/// sum per row *per column*, so what it asks of the register file is the
/// product of the two: four beside [`ROWS_A_TILE`] is 32 accumulators a lane
/// where the row tile alone wanted eight, and eight columns is 64. **That the
/// turn is register pressure is a reading and not a measurement** — the widest
/// threadgroup the pipeline reports is the device's own 1024 at every width
/// tried, eight included, which is the one place this side could have seen it.
const COLS_A_TILE: usize = 4;

/// Rows an expert needs, on average, before sorting a call's rows by expert
/// pays for the sort.
///
/// **Two measurements disagree about where this goes and the engine's is the
/// one it is set from.** Over runs of a uniform length,
/// `what_grouping_a_banks_rows_by_expert_is_worth_at_each_run_length` turns at
/// four and is a loss below it — which is the tile height and no coincidence:
/// runs of two laid out end to end put a boundary inside *every* tile, so every
/// tile falls back to walking each row's own weight and the sort bought a
/// dispatch and nothing else. But a routing's runs are not uniform. At 97 tokens
/// a routed bank averages 2.3 rows an expert and the grouping takes 711 ms of
/// device time to 592, because the experts a prompt favours reach a tile even
/// where the mean does not — and `what_a_grouped_call_reads_against_what_it
/// _declares` says that saving is not in the weight reads, which barely move at
/// that length. So the line is drawn from the prefill rather than from the
/// sweep, one step under the shortest prompt measured.
///
/// **What it has to exclude is everything a decode step and a speculative round
/// dispatch**, and it is nowhere near them: one token is six rows over 256
/// experts and the widest block the eight heads can propose is nine tokens,
/// which is 54 — 0.02 and 0.2 rows an expert against this two.
const RUNS_A_GROUPING: usize = 2;

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
/// **The two cancel in a step, which is what makes this a constant and not a
/// rule.** A dispatch knows its own element count and could pick a width from
/// it the way [`crate::dense`] picks a reduction run, and eight bytes a lane was
/// measured that far: 18.14 ms of device time against four's 18.12 over five
/// alternating pairs. So what decides the number is the prefill, where four is
/// ahead on its own.
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
    tiled: Kernel,
    grouped: Kernel,
    /// The rows a tile of `tiled` holds, which is [`ROWS_A_TILE`] for the
    /// shipped source and whatever the sweep wrote into a mutant's prelude.
    rows_a_tile: usize,
    /// The columns it spans, the same way — see [`COLS_A_TILE`].
    cols_a_tile: usize,
}

impl PackedMatmul {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`PackedMatmul::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Self::tiling(device, source, ROWS_A_TILE, COLS_A_TILE)
    }

    /// The same, where the source declares a tile of another shape.
    ///
    /// **The three are given together because none is any use alone**, and
    /// the assertions are what make that safe rather than conventional. The grid
    /// this side dispatches covers the tiles the two heights cut a call into and
    /// the kernel takes its own shape from its prelude, so a pair that
    /// disagreed would leave elements no simdgroup computed or run tiles off the
    /// end of the call — and both are wrong answers rather than failures.
    pub(crate) fn tiling(
        device: &Device,
        source: &str,
        rows_a_tile: usize,
        cols_a_tile: usize,
    ) -> Result<Self, MetalError> {
        for declares in [
            format!("constant uint ROWS_A_TILE = {rows_a_tile};"),
            format!("constant uint COLS_A_TILE = {cols_a_tile};"),
        ] {
            assert!(
                source.contains(&declares),
                "a source dispatched at {rows_a_tile}x{cols_a_tile} a tile does not declare \
                 `{declares}`"
            );
        }
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
            tiled: device.compile(source, TILED_ENTRY)?,
            grouped: device.compile(source, GROUPED_ENTRY)?,
            rows_a_tile,
            cols_a_tile,
        })
    }

    /// The entry a call goes through and the simdgroups it takes to cover
    /// `rows` rows of `out_dim` columns.
    ///
    /// **An untiled call stays on the untiled kernel rather than on a tile of
    /// one.** The two compute the same thing at that height, and the ordinary
    /// one is what every decode step this project has measured was dispatching
    /// — so the shape with nothing to win stays on the code that won what is
    /// already there, rather than carrying a tile's worth of registers to use
    /// one row of it.
    fn entry(&self, layout: &Layout<'_>, rows: usize, out_dim: usize) -> (&Kernel, usize) {
        let tiles = rows.div_ceil(self.rows_a_tile) * out_dim.div_ceil(self.cols_a_tile);
        match layout {
            Layout::Each => (&self.kernel, rows * out_dim),
            Layout::Tiled => (&self.tiled, tiles),
            Layout::Grouped { .. } => (&self.grouped, tiles),
        }
    }

    /// Whether laying a call's rows out by the expert each named is worth the
    /// dispatch that lays them out.
    ///
    /// **The question is the length of the runs the sort would produce**, which
    /// is the rows over the experts they can name — where [`tiles`] asks the
    /// same question of a layout nobody moved. A call with fewer rows than the
    /// bank has experts sorts into runs of about one and has nothing for a tile
    /// to share; the rows a prefill gives a routed bank are 6 a token against
    /// 256 experts, so 97 tokens are runs of 2.3, 385 of nine and 769 of
    /// eighteen. [`RUNS_A_GROUPING`] is where the line is and what it was drawn
    /// from.
    pub(crate) fn groups(&self, rows: usize, experts: usize) -> bool {
        self.rows_a_tile >= 2 && rows >= experts.saturating_mul(RUNS_A_GROUPING)
    }
}

/// Where a grouped call's permutation applies: to the rows it reads, or to the
/// rows it writes.
///
/// **Both ends of a bank need it and only one end of each dispatch does.** The
/// gate and up projections are handed the hidden state in the order the tokens
/// are in and produce rows the activation and `down_proj` then read, so they
/// gather on the way in and leave everything after them grouped; `down_proj`
/// reads those grouped rows where they lie and scatters on the way out, so what
/// it answers with is back in the order the weighting and the combine expect. A
/// dispatch that did both would be undoing the grouping it was dispatched for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Through {
    /// Row `i` reads row `order[i]` of the input and writes row `i`.
    Gathered,
    /// Row `i` reads row `i` of the input and writes row `order[i]`.
    Scattered,
}

/// How a dispatch's rows are cut up between simdgroups, and the permutation the
/// third way of cutting them reads.
///
/// The [`Arg`] rather than the buffer, because a grouped call's order is a
/// binding like the expert list beside it — [`PackedBank::dispatch`] is handed
/// both the same way and never learns where either came from.
enum Layout<'a> {
    /// One simdgroup an output element.
    Each,
    /// One simdgroup a tile of consecutive rows.
    Tiled,
    /// One simdgroup a tile of consecutive rows of the grouping.
    Grouped { order: Arg<'a>, through: Through },
}

/// Whether a call's rows are laid out so that tiling them is worth a dispatch.
///
/// **The question is not how many rows there are but how long a run of them
/// names one expert**, since that is what a tile can share a weight read
/// across. A run of at least a tile leaves at most one straddling tile per
/// boundary, and a straddling tile is correct and simply saves nothing.
///
/// The three layouts this bank is dispatched in answer it differently, and by
/// construction rather than by luck. A projection's rows all name expert zero,
/// so the run is the whole call. A shared bank is its input laid end to end
/// once per expert, so the run is the tokens. A routed bank's rows are a
/// token's six experts one after another, so every run is one row and this is
/// false however long the prompt — which is where the other 59% of a prefill's
/// bytes are, and it wants the rows moved rather than tiled.
///
/// False for every shape a decode step dispatches, which is what says this
/// cannot cost one: a decode step's projections are a single row, its shared
/// bank is two rows naming two experts, and its routed bank is six naming six.
fn tiles(experts: &[u32], rows_a_tile: usize) -> bool {
    if rows_a_tile < 2 || experts.len() < rows_a_tile {
        return false;
    }
    let shortest = experts
        .chunk_by(|a, b| a == b)
        .map(<[u32]>::len)
        .min()
        .unwrap_or(0);
    shortest >= rows_a_tile
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

/// One weight a layer's dispatches multiply through, whichever format it is
/// stored in.
///
/// **Two formats and one layer.** Every projection of the model's own forty-two
/// layers is MXFP4 and every weight of the eight MTP heads is bfloat16 — the
/// quantiser dropped the heads rather than packing them — so what separates
/// [`PackedProjection`] from [`DenseWeight`](crate::DenseWeight) is the bytes a
/// dispatch reads and nothing about the layer around it. This is what says so:
/// [`LayerProjections`](crate::LayerProjections) holds five of these and
/// [`DenseFfn`](crate::DenseFfn) three, and neither knows which kernel answered.
///
/// [`Projection`] is the supertrait because that is the seam the CPU side
/// already names — a caller with one row to multiply and nothing to batch it
/// against wants that one, and both of these already answer it.
pub trait Multiply: Projection + std::fmt::Debug {
    /// The device this multiplies on, for a caller batching several of these
    /// into one command buffer.
    fn device(&self) -> &Device;

    /// `[rows, in_dim]` this side holds, encoded into `batch` rather than
    /// submitted on its own.
    fn encode(&self, batch: &mut Batch<'_>, x: &[f32]) -> Result<Pending, MatmulError>;

    /// The same multiply over rows a dispatch already left on the device.
    fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError>;
}

impl Multiply for PackedProjection<'_> {
    fn device(&self) -> &Device {
        PackedProjection::device(self)
    }

    fn encode(&self, batch: &mut Batch<'_>, x: &[f32]) -> Result<Pending, MatmulError> {
        PackedProjection::encode(self, batch, x)
    }

    fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        PackedProjection::encode_over(self, batch, x)
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

    /// Whether a call of `rows` rows against this bank is worth sorting by
    /// expert first — see [`PackedMatmul::groups`], which decides it, and which
    /// needs the bank's own expert count to.
    pub(crate) fn groups(&self, rows: usize) -> bool {
        self.matmul.groups(rows, self.experts)
    }

    /// The scalars the kernel's `Shape` struct declares, in its order — the
    /// caller's to hold, because they are read where the dispatch is encoded and
    /// an array made here would be gone by then.
    fn shape(
        &self,
        rows: usize,
        per_source: usize,
        sources: usize,
        layout: &Layout<'_>,
    ) -> [u32; SHAPE_FIELDS] {
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
            u32::from(matches!(
                layout,
                Layout::Grouped {
                    through: Through::Scattered,
                    ..
                }
            )),
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
        // Untiled, because what the list holds is a token's `per_source`
        // experts one after another — the one layout no tile can share a weight
        // across, and the one [`PackedBank::encode_grouped`] moves the rows of.
        self.dispatch(batch, rows, per_source, chosen.arg(), x, Layout::Each)
    }

    /// The same multiply over rows a dispatch already laid out expert by
    /// expert, `per_source` rows of the *ungrouped* call reading each row of
    /// the input.
    ///
    /// **This is the gather the routed bank never had.** Its rows are the same
    /// rows [`PackedBank::encode_picked`] runs and its answers are the same
    /// answers, row for row; what moved is the order the tile sees them in, so
    /// that the rows naming one expert are consecutive and one weight read
    /// serves up to [`ROWS_A_TILE`] of them. Which expert a row goes through is
    /// [`Grouped::experts`](crate::Grouped) and is `chosen[order[i]]` — the
    /// grouping cannot change it, and `a_grouping_moves_rows_and_never_the_
    /// expert_a_row_named` is what says so.
    ///
    /// `through` says which end of this dispatch the permutation applies at,
    /// and the two are not interchangeable — see [`Through`].
    pub(crate) fn encode_grouped(
        &self,
        batch: &mut Batch<'_>,
        grouped: &mut Grouped,
        x: &mut Buffer<f32>,
        per_source: usize,
        through: Through,
    ) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        let rows = grouped.order.len();
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
        let layout = Layout::Grouped {
            order: grouped.order.arg(),
            through,
        };
        self.dispatch(batch, rows, per_source, grouped.experts.arg(), x, layout)
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
        let layout = match tiles(experts, self.matmul.rows_a_tile) {
            false => Layout::Each,
            true => Layout::Tiled,
        };
        self.dispatch(batch, experts.len(), 1, chosen.arg(), x, layout)
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
    ///
    /// `layout` decides which entry runs it and nothing else about it: the
    /// three take the same shape and the same expert list and answer the same
    /// values — see [`tiles`] and [`PackedMatmul::groups`] for who can say yes
    /// to each, and `packed_matmul_rows` for why the answer does not depend on
    /// the caller being right about an expert list.
    ///
    /// **The grouped entry is the one that takes a seventh binding**, and the
    /// two arms below are that difference stated where it is: a kernel that
    /// reads no permutation is handed none, so nothing a decode step dispatches
    /// grew an argument.
    fn dispatch(
        &self,
        batch: &mut Batch<'_>,
        rows: usize,
        per_source: usize,
        chosen: Arg<'_>,
        x: &mut Buffer<f32>,
        layout: Layout<'_>,
    ) -> Result<Pending, MatmulError> {
        if rows == 0 {
            return Ok(Pending::empty());
        }

        // The shape is read out of `resident` and so is built before it is
        // borrowed mutably for the binding below.
        let fields = self.shape(rows, per_source, self.sources(x.len()), &layout);
        let mut shape = self.device.inline(&fields)?;
        let mut resident = self.resident.borrow_mut();
        let resident = &mut *resident;
        let mut out = self.device.zeroed::<f32>(rows * self.out_dim)?;

        let (kernel, elements) = self.matmul.entry(&layout, rows, self.out_dim);
        let grid = Grid::new(elements * kernel.simd_width(), THREADS_PER_GROUP);
        let moves = self.moves(rows, x.len(), &layout);
        let bound = [
            shape.arg(),
            chosen,
            x.arg(),
            resident.codes.arg(),
            resident.scales.arg(),
            out.arg(),
        ];
        match layout {
            Layout::Grouped { order, .. } => {
                let [shape, chosen, x, codes, scales, written] = bound;
                batch.add(
                    kernel,
                    &[shape, chosen, x, codes, scales, written, order],
                    grid,
                    moves,
                )?;
            }
            Layout::Each | Layout::Tiled => batch.add(kernel, &bound, grid, moves)?,
        }

        Ok(Pending { out: Some(out) })
    }

    /// What one dispatch of `rows` rows over `values` of input moves.
    ///
    /// **A weight is read once per tile of rows that shares it**, which
    /// untiled is once per row. Each output row goes through one expert and
    /// every element of that row reads a different `in_dim`-long slice of it,
    /// so an untiled call of `rows` rows reads `rows` weights whichever experts
    /// they name — six routed rows of a `[2048, 4096]` bank are 27 MB of packed
    /// bytes for six rows of output, which is what makes this the kernel a
    /// decode step's bandwidth is mostly about. Tiled, the same six rows under
    /// one expert are two tiles and so 9.0 MB.
    ///
    /// **The tiled figure is what a call of uniform tiles reads, and a
    /// straddling tile reads one weight more than is charged here.** A tile
    /// whose rows name two experts walks both — see `packed_matmul_rows` — and
    /// there is at most one such tile per run of equal experts, which [`tiles`]
    /// keeps at least a tile long. So this under-counts by under a run in
    /// however many tiles the call has: nothing at all for a projection, whose
    /// rows are one run, and one tile in ninety-six for a shared bank at 385
    /// tokens.
    ///
    /// **A grouped call is charged the other way round, and it has to be.** Its
    /// runs are as long as the routing made them, so a tile boundary falls
    /// inside a run far more often than at the end of one — at 97 tokens a
    /// routed bank's runs are 2.3 rows and nearly every tile straddles. What
    /// this side can say is a bound rather than a count: there is at most one
    /// straddling tile per run, a run per expert, and a straddling tile reads
    /// [`ROWS_A_TILE`] weights where a uniform one reads one. So a grouped call
    /// is charged the *worst* layout its shape allows, which is never below what
    /// it reads — where charging one weight a tile would be a figure that
    /// flattered the change this kernel exists for.
    ///
    /// It is loose, and how loose is measured rather than argued:
    /// `what_a_grouped_call_reads_against_what_it_declares` puts it 0.2% above
    /// the truth at 97 tokens, 24% at 385 and 12% at 769. Nothing on this side
    /// can narrow it — the expert each row named is in device memory and was
    /// never read back.
    ///
    /// The weight is charged as the bytes it is packed into rather than the
    /// values it holds — a code is half a byte and a group of 32 codes shares
    /// one scale byte — because the whole of this kernel is that nothing decodes
    /// it on the way. Beside it, the input, the output, and the one expert index
    /// a row is read through, which a bank's own dispatch leaves on the device
    /// and a projection's travels in the command buffer.
    fn moves(&self, rows: usize, values: usize, layout: &Layout<'_>) -> usize {
        let tiles = rows.div_ceil(self.matmul.rows_a_tile);
        let read = match layout {
            Layout::Each => rows,
            Layout::Tiled => tiles,
            Layout::Grouped { .. } => rows.min(
                tiles + (self.matmul.rows_a_tile - 1) * tiles.min(self.experts.saturating_sub(1)),
            ),
        };
        let elements = read * self.out_dim * self.in_dim;
        let weight = elements * BITS / u8::BITS as usize + elements / GROUP_SIZE;
        // A grouped call reads the order beside the expert list, one index a
        // row of each.
        let indices = match layout {
            Layout::Grouped { .. } => 2,
            Layout::Each | Layout::Tiled => 1,
        };
        weight
            + size_of::<f32>() * (values + rows * self.out_dim)
            + size_of::<u32>() * indices * rows
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
constant uint ROWS_A_TILE = {ROWS_A_TILE};
constant uint COLS_A_TILE = {COLS_A_TILE};
constant float ELEMENTS[] = {{ {} }};
{BODY}{}{}",
        (1u32 << BITS) - 1,
        elements.join(", "),
        tiled_entry(TILED_ENTRY, false),
        tiled_entry(GROUPED_ENTRY, true),
    )
}

/// How many `uint`s the kernel's `Shape` struct declares.
const SHAPE_FIELDS: usize = 10;

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
    /// Which end of a grouped call the permutation applies at: 0 names the row
    /// each of the call's rows reads, 1 names the row each of them writes. Read
    /// by `packed_matmul_grouped` and by nothing else.
    uint scatters;
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

/// Where a row of the input the tile's `r`th row multiplies begins.
inline uint tile_source(constant Shape &shape, uint row) {
    return ((row / shape.per_source) % shape.sources) * shape.in_dim;
}

"#;

/// The tiled kernel, which is emitted twice: once over the rows as the caller
/// laid them out and once over a permutation a dispatch wrote.
///
/// **Written once and compiled twice.** The two differ in three places — the
/// binding the order arrives in, the row of the input a tile's `r`th row reads,
/// and the row of the output it writes — and in nothing else at all: the same
/// tile check, the same fallback, the same walk, the same reduction. Sharing
/// them at *run* time is what would cost something, because everything a tile
/// carries has to stay in registers and a body reached through a function is a
/// body whose arrays the compiler may put somewhere else. Sharing them here
/// costs nothing and is the only reading of the walk.
///
/// **`packed_matmul_grouped` is the one that reaches a routed bank**, and what
/// it reaches it with is an indirection at one end of the call. `gate` and `up`
/// read the hidden state through `order` and leave their rows grouped, the
/// activation between them is elementwise and does not care, and `down` reads
/// those grouped rows where they lie and writes each of them back to the row the
/// router named — so the tensor the weighting behind it reads is in the order it
/// was always in, and nothing downstream of the bank knows the rows were ever
/// moved. `shape.scatters` is which of the two a call is.
///
/// **The expert list is the grouping's**, not the router's: `experts[i]` is
/// `chosen[order[i]]`, which `group_by_expert` writes in the same dispatch that
/// writes `order`. So the question the tile asks — do these rows name one weight
/// — is asked of the sorted list, which is where the runs are. And a tile that
/// straddles two runs is correct and saves nothing, exactly as it is for the
/// ungrouped entry, so neither of the two assumes anything the other does not.
fn tiled_entry(entry: &str, grouped: bool) -> String {
    let (order, reads, writes) = match grouped {
        false => ("", "row", "row"),
        true => (
            "\n    device const uint *order [[buffer(6)]],",
            "shape.scatters ? row : order[row]",
            "shape.scatters ? order[row] : row",
        ),
    };
    // Each substitution replaces every occurrence, so what keeps the four from
    // reaching into each other is that none of them writes a placeholder — which
    // is a property of these four values rather than of the mechanism, and so is
    // asserted rather than read off them.
    let written = [entry, order, reads, writes];
    assert!(
        !written.iter().any(|value| value.contains("__")),
        "a substitution that writes a placeholder would be substituted again"
    );
    TILE.replace("__ENTRY__", entry)
        .replace("__ORDER__", order)
        .replace("__READS__", reads)
        .replace("__WRITES__", writes)
}

/// The tiled kernel with the three expressions [`tiled_entry`] decides written
/// as placeholders, which is what makes one walk serve both entries.
const TILE: &str = r#"
/// One simdgroup per `ROWS_A_TILE` consecutive rows of `COLS_A_TILE`
/// consecutive columns rather than per output element.
///
/// **The whole of what the rows buy is that the weight row is read once.**
/// `packed_matmul` walks the weight each of a call's rows names from end to
/// end, so a call of `n` rows against one expert reads that expert `n` times —
/// which is a decode step's price paid once a token by a prefill that could
/// have paid it once. Here the walk is shared: one lane loads a packed byte,
/// decodes its two codes, and multiplies them against the `ROWS_A_TILE` rows of
/// `x` that want them.
///
/// **And what the columns buy is the other side of the same loop.** A row tile
/// reads one packed byte and `8 * ROWS_A_TILE` input floats around it, which is
/// 32 bytes of input for every byte of weight the dispatch is charged — so a
/// tile of rows alone stops waiting on the weight and starts waiting on the
/// input. `COLS_A_TILE` columns of the same rows read that many weight bytes
/// against one read of those same floats, and the ratio falls with it. The
/// columns share no weight byte: every column is its own weight row, and what a
/// column tile moves is exactly what the same columns moved apart.
///
/// **Only rows naming the same expert can share it, and the check is here
/// rather than in the caller's head.** A tile whose rows disagree falls back to
/// walking each row's own weight, which is exactly what `packed_matmul` does
/// and is the same arithmetic — so a caller that tiled a routed bank gets a
/// correct answer and no saving, and correctness never rests on a claim about
/// an expert list this side may not have seen.
///
/// **The answer is the untiled kernel's bit for bit and that is by
/// construction**, not within a tolerance. An output element is still one
/// simdgroup's `simd_sum` over lanes that still walk the row in `BYTES_PER_LANE`
/// chunks from the same byte in the same stride, and a chunk is still summed
/// into `dot` and then into `sum` under one scale. Nothing about the order any
/// product enters any sum has moved; what moved is how many sums one load
/// feeds. `a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers`
/// and `a_grouped_dispatch_answers_what_the_dispatch_it_reorders_answers` are
/// where that is held, one for each entry.
kernel void __ENTRY__(
    constant Shape &shape [[buffer(0)]],
    device const uint *experts [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    device float *out [[buffer(5)]],__ORDER__
    uint position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]
) {
    const uint tile = position / width;
    const uint down = (shape.rows + ROWS_A_TILE - 1) / ROWS_A_TILE;
    const uint across = (shape.out_dim + COLS_A_TILE - 1) / COLS_A_TILE;
    if (tile >= down * across) {
        return;
    }

    const uint first = (tile / across) * ROWS_A_TILE;
    const uint leftmost = (tile % across) * COLS_A_TILE;
    const uint bytes = shape.in_dim / CODES_PER_BYTE;
    const uint rows = min((uint)ROWS_A_TILE, shape.rows - first);
    const uint cols = min((uint)COLS_A_TILE, shape.out_dim - leftmost);
    const uint scale_bytes = bytes / BYTES_PER_GROUP;

    const uint expert = experts[first];
    bool one_weight = true;
    for (uint r = 1; r < rows; ++r) {
        one_weight = one_weight && (experts[first + r] == expert);
    }

    if (!one_weight) {
        for (uint r = 0; r < rows; ++r) {
            const uint row = first + r;
            const uint each = experts[row];
            for (uint c = 0; c < cols; ++c) {
                const uint col = leftmost + c;
                float sum = weight_dot(
                    codes + shape.code_base + (ulong)each * shape.code_stride + (ulong)col * bytes,
                    scales + shape.scale_base + (ulong)each * shape.scale_stride
                        + (ulong)col * scale_bytes,
                    x + tile_source(shape, __READS__),
                    bytes,
                    lane,
                    width
                );
                sum = simd_sum(sum);
                if (lane == 0) {
                    out[(__WRITES__) * shape.out_dim + col] = sum;
                }
            }
        }
        return;
    }

    // The rows and the columns past this tile's own read the last of each, so
    // that every load below is inside the buffer it indexes whatever the call's
    // shape leaves over. What they produce is never written.
    device const uchar *packed[COLS_A_TILE];
    device const uchar *scale[COLS_A_TILE];
    for (uint c = 0; c < COLS_A_TILE; ++c) {
        const uint col = leftmost + min(c, cols - 1);
        packed[c] = codes + shape.code_base + (ulong)expert * shape.code_stride
            + (ulong)col * bytes;
        scale[c] = scales + shape.scale_base + (ulong)expert * shape.scale_stride
            + (ulong)col * scale_bytes;
    }

    uint sources[ROWS_A_TILE];
    float sums[ROWS_A_TILE][COLS_A_TILE];
    for (uint r = 0; r < ROWS_A_TILE; ++r) {
        const uint row = first + min(r, rows - 1);
        sources[r] = tile_source(shape, __READS__);
        for (uint c = 0; c < COLS_A_TILE; ++c) {
            sums[r][c] = 0.0f;
        }
    }

    for (uint b = lane * BYTES_PER_LANE; b < bytes; b += width * BYTES_PER_LANE) {
        float dots[ROWS_A_TILE][COLS_A_TILE];
        for (uint r = 0; r < ROWS_A_TILE; ++r) {
            for (uint c = 0; c < COLS_A_TILE; ++c) {
                dots[r][c] = 0.0f;
            }
        }
        for (uint i = 0; i < BYTES_PER_LANE; ++i) {
            float low[COLS_A_TILE];
            float high[COLS_A_TILE];
            for (uint c = 0; c < COLS_A_TILE; ++c) {
                const uint code = packed[c][b + i];
                low[c] = ELEMENTS[code & CODE_MASK];
                high[c] = ELEMENTS[(code >> BITS) & CODE_MASK];
            }
            const uint at = (b + i) * CODES_PER_BYTE;

            for (uint r = 0; r < ROWS_A_TILE; ++r) {
                device const float *v = x + sources[r] + at;
                const float even = v[0];
                const float odd = v[1];
                for (uint c = 0; c < COLS_A_TILE; ++c) {
                    dots[r][c] += low[c] * even + high[c] * odd;
                }
            }
        }
        for (uint c = 0; c < COLS_A_TILE; ++c) {
            const float by =
                as_type<float>(uint(scale[c][b / BYTES_PER_GROUP]) << EXPONENT_SHIFT);
            for (uint r = 0; r < ROWS_A_TILE; ++r) {
                sums[r][c] += dots[r][c] * by;
            }
        }
    }

    // The order is read again here rather than carried through the loop above:
    // an index a tile holds is a register its rows compete for, and this one is
    // wanted once a row of the tile where the sources are wanted once a byte.
    for (uint r = 0; r < rows; ++r) {
        const uint row = first + r;
        for (uint c = 0; c < cols; ++c) {
            const float sum = simd_sum(sums[r][c]);
            if (lane == 0) {
                out[(__WRITES__) * shape.out_dim + leftmost + c] = sum;
            }
        }
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
    use std::time::{Duration, Instant};

    use super::testing::{Case, Noise, pack};
    use super::*;
    use inkling_core::fixture::{self, deviation};
    use inkling_core::ops::DenseProjection;
    use inkling_core::quant::dequantize_blocks;
    use inkling_core::weights::PackedRows;
    use inkling_core::workload::BEST;

    use crate::grouping::ExpertGrouping;
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
    /// **A weight is read once per tile and the weight is never decoded**,
    /// which are the two things this figure has to get right: every element of
    /// an output row reads a different slice of the expert that row goes
    /// through, and what it reads is packed — half a byte a code, plus one
    /// scale byte for every 32 of them. A figure that charged the decoded
    /// float32 would be eight times high and would put this kernel past the
    /// machine's bandwidth rather than at a third of it.
    ///
    /// **Both heights, because the whole of the prefill change is this number.**
    /// A kernel that tiled without the declaration following it would report a
    /// bandwidth it never reached and a saving nothing measured; one that
    /// declared the tiling without the kernel doing it would report the reverse.
    #[test]
    fn a_dispatch_declares_the_packed_bytes_it_reads_rather_than_the_values() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const IN_DIM: usize = 128;
        const OUT_DIM: usize = 8;

        let declared = |rows: usize| {
            let case = Case::seeded(1, IN_DIM, OUT_DIM, rows);
            let projection = case.upload(&device, &matmul);
            crate::testing::moved(&device, |batch| {
                projection
                    .encode(batch, &case.x)
                    .expect("the dispatch encodes");
            }) as usize
        };
        let besides = |rows: usize| {
            size_of::<f32>() * (rows * IN_DIM + rows * OUT_DIM) + size_of::<u32>() * rows
        };
        let weight = |read: usize| {
            let elements = read * OUT_DIM * IN_DIM;
            elements / 2 + elements / GROUP_SIZE
        };

        // Under a tile, so every row reads its own weight — which is what a
        // decode step's shapes all are.
        const UNTILED: usize = 3;
        assert!(!tiles(&[0; UNTILED], ROWS_A_TILE));
        assert_eq!(
            declared(UNTILED),
            weight(UNTILED) + besides(UNTILED),
            "codes, scales, the rows in, the rows out, and an expert a row"
        );
        assert!(
            declared(UNTILED) < UNTILED * OUT_DIM * IN_DIM * size_of::<f32>(),
            "a decoded weight was charged for one nothing decodes"
        );

        // Two whole tiles and a row over, all of them one expert, so three
        // weights are read for every row of the call.
        const TILED: usize = 2 * ROWS_A_TILE + 1;
        assert!(tiles(&[0; TILED], ROWS_A_TILE));
        assert_eq!(
            declared(TILED),
            weight(TILED.div_ceil(ROWS_A_TILE)) + besides(TILED),
            "a weight a tile, and the rows in and out of every row"
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

    /// Which layouts share a weight read across a tile, decided on the expert
    /// list alone.
    ///
    /// **The three the engine dispatches answer differently and none of them by
    /// luck**, so they are written out here as the lists they are rather than
    /// described. What matters most is the last pair: a decode step's every
    /// shape says no, which is what makes this incapable of moving the step
    /// this project spent four milestones on.
    #[test]
    fn a_calls_rows_share_a_weight_read_only_where_they_name_one_expert() {
        let projection = |rows: usize| vec![0u32; rows];
        let shared = |tokens: usize| -> Vec<u32> {
            (0..2 * tokens).map(|row| (row / tokens) as u32).collect()
        };
        let routed =
            |tokens: usize| -> Vec<u32> { (0..6 * tokens).map(|row| (row % 6) as u32).collect() };

        assert!(
            tiles(&projection(385), ROWS_A_TILE),
            "a prefill's projection"
        );
        assert!(tiles(&shared(385), ROWS_A_TILE), "a prefill's shared bank");
        assert!(!tiles(&routed(385), ROWS_A_TILE), "a prefill's routed bank");

        assert!(
            !tiles(&projection(1), ROWS_A_TILE),
            "a decode step's projection"
        );
        assert!(
            !tiles(&shared(1), ROWS_A_TILE),
            "a decode step's shared bank"
        );
        assert!(
            !tiles(&routed(1), ROWS_A_TILE),
            "a decode step's routed bank"
        );

        // A run exactly a tile long is the shortest that pays, and one row
        // under it is the longest that cannot.
        assert!(tiles(&projection(ROWS_A_TILE), ROWS_A_TILE));
        assert!(!tiles(&projection(ROWS_A_TILE - 1), ROWS_A_TILE));
        assert!(
            !tiles(&shared(ROWS_A_TILE - 1), ROWS_A_TILE),
            "runs under a tile"
        );

        // **And a speculative round's block, which is the shape a decode figure
        // is most easily lost through.** A round of depth `k` verifies `k + 1`
        // rows in one pass, so the deepest depth this repo quotes dispatches a
        // three-row projection — and three is one under the tile. Every block a
        // round can propose stays on the untiled kernel, which is what says the
        // 16.48 ms at `k = 2` cannot be reached from here.
        for verified in 1..=BEST + 1 {
            assert!(
                !tiles(&projection(verified), ROWS_A_TILE),
                "a block of {verified} rows"
            );
            assert!(!tiles(&routed(verified), ROWS_A_TILE), "its routed bank");
        }
        // **Which is a fact about the height and not about the block**, and it
        // is why the height may rise and may not fall: a tile of three would
        // take a `k = 2` verify's projections into the tiled path, and what a
        // cold routed bank says that is worth is 3% of one row of a prefill —
        // see `whether_the_tile_height_turns_at_four_on_a_bank_too_big_to_cache`.
        assert!(tiles(&projection(BEST + 1), BEST + 1));
    }

    /// **The tiled kernel's answer against the untiled one's, bit for bit.**
    ///
    /// This is the claim the whole change rests on. A tile reads one weight row
    /// once and multiplies it against every row of the tile that named that
    /// expert, where the untiled kernel reads the row again for each — and
    /// since nothing about the order any product enters any sum moved, the two
    /// have to agree exactly rather than within a tolerance. A bound here would
    /// be hiding the one mistake this kernel can make.
    ///
    /// The oracle is each row run *alone*, which is a call of one row and so
    /// goes through the untiled kernel by [`tiles`]'s own rule — the same
    /// arrangement `a_gathered_dispatch_multiplies_each_row_against_the_expert
    /// _it_named` uses, and it needs no second implementation of anything.
    ///
    /// **The shape is chosen so that all three kinds of tile occur**, and the
    /// three assertions below are what say it still does if [`ROWS_A_TILE`]
    /// moves: a run of eleven rows is not a whole number of tiles, so one tile
    /// straddles the two experts and walks each of its rows' own weight, and
    /// twenty-two rows are not either, so the call ends inside a tile. It is
    /// the repeating shape rather than
    /// the gathered one because the input row a tile's rows read is then a
    /// modulo rather than the row index, so a tiled kernel that hoisted the
    /// source out of the loop instead of computing it per row lands the wrong
    /// input on the second expert's rows.
    #[test]
    fn a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const EXPERTS: usize = 2;
        const SOURCES: usize = 11;

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

        let chosen: Vec<u32> = (0..EXPERTS * SOURCES)
            .map(|row| (row / SOURCES) as u32)
            .collect();
        assert!(
            tiles(&chosen, ROWS_A_TILE),
            "the call under test was not tiled"
        );
        assert_ne!(
            chosen.len() % ROWS_A_TILE,
            0,
            "a call that filled its last tile would not exercise the partial one"
        );
        assert_ne!(
            SOURCES % ROWS_A_TILE,
            0,
            "a run that filled whole tiles would not exercise the straddling one"
        );
        assert_ne!(
            OUT_DIM % COLS_A_TILE,
            0,
            "columns that filled their last tile would not exercise the partial one"
        );

        let mut batch = device.batch().expect("a command buffer opens");
        let mut input = device.buffer(&case.x).expect("the input uploads");
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
            .multiply(&case.x[(row % SOURCES) * IN_DIM..][..IN_DIM])
            .expect("the dispatch completes");
            assert_eq!(got[row * OUT_DIM..][..OUT_DIM], alone[..], "row {row}");
        }

        // The rows have to disagree for the check above to have said anything:
        // against a bank whose two experts answered alike, a tile that read the
        // wrong weight for the rows past a boundary would still match.
        let row = |i: usize| &got[i * OUT_DIM..][..OUT_DIM];
        assert_ne!(row(0), row(1), "the input row is read off the row");
        assert_ne!(row(0), row(SOURCES), "the expert is read off the row");
    }

    /// The routing a grouped case is dispatched over: `TOP_K` experts a token,
    /// no two of a token's the same, and enough tokens that every expert of the
    /// bank is named several times over.
    ///
    /// The shape a routed bank runs at, cut down: 6 of 256 becomes 2 of 3, and
    /// what carries over is the one property that matters — a token's rows name
    /// different experts, so no two consecutive rows of the ungrouped call ever
    /// share a weight.
    const TOP_K: usize = 2;
    const TOKENS: usize = 7;

    fn routing(experts: usize) -> Vec<u32> {
        (0..TOKENS)
            .flat_map(|token| (0..TOP_K).map(move |slot| ((token + slot * 2) % experts) as u32))
            .collect()
    }

    /// Which shapes are worth laying out by expert first, decided on the row
    /// and expert counts alone.
    ///
    /// **The shapes that must say no are the ones this milestone is under a
    /// constraint about**, and they are written out here as the numbers they
    /// are rather than described. A decode step's routed bank is six rows over
    /// 256 experts and the widest block the eight multi-token heads can propose
    /// is nine tokens, which is 54 — both sort into runs of one, where a tile
    /// shares nothing and the sort is a dispatch spent for it.
    ///
    /// The shapes that say yes are the prefill's, and the shortest of them is
    /// what [`RUNS_A_GROUPING`] was set from: 97 tokens are 582 rows and 2.3 an
    /// expert.
    #[test]
    fn only_a_prefills_routed_bank_is_worth_laying_out_by_expert() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const ROUTED: usize = 256;
        const ROUTED_TOP_K: usize = 6;
        let tokens = |tokens: usize| matmul.groups(tokens * ROUTED_TOP_K, ROUTED);

        assert!(!tokens(1), "a decode step's routed bank");
        assert!(!tokens(9), "the widest block a speculative round proposes");
        assert!(tokens(97), "the shortest prefill this file measures");
        assert!(tokens(385));
        assert!(tokens(769));

        // The line itself, from either side, and that no bank is grouped at a
        // height below its own expert count however many rows it has.
        assert!(tokens(86) && !tokens(85), "the line is at 86 tokens");
        assert!(
            !matmul.groups(ROUTED, ROUTED),
            "a call of one row an expert sorts into runs of one"
        );
    }

    /// **A grouped dispatch answers what the ungrouped one answers, row for
    /// row and bit for bit** — the rows having been moved through the tile and
    /// back.
    ///
    /// This is the claim the whole change rests on, and it is the same claim
    /// `a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers`
    /// makes one step earlier: nothing about the order any product enters any
    /// sum moved, so a bound here would be hiding the one mistake this can
    /// make. What moved is which rows a weight read is shared across, and — new
    /// here — which row of the input a row of the call reads and which row of
    /// the output it writes.
    ///
    /// **Both ends, because they are two different dispatches of the bank.**
    /// `gate` and `up` gather: the rows arrive in the router's order and leave
    /// in the grouping's, so what is checked is that grouped row `i` is the
    /// ungrouped row `order[i]`. `down` scatters: the rows arrive grouped and
    /// leave in the router's order, so what is checked is that the answer is
    /// the ungrouped one exactly, in place.
    ///
    /// The routing is the one a router writes — a token's slots naming
    /// different experts — so the ungrouped call is the untiled kernel by
    /// [`tiles`]'s own rule, which is what makes it the before of this change
    /// rather than a tiled kernel imitating it.
    #[test]
    fn a_grouped_dispatch_answers_what_the_dispatch_it_reorders_answers() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const EXPERTS: usize = 3;
        const ROWS: usize = TOKENS * TOP_K;

        let case = Case::noisy(IN_DIM, EXPERTS * OUT_DIM, ROWS);
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

        let chosen = routing(EXPERTS);
        assert!(
            !tiles(&chosen, ROWS_A_TILE),
            "a routing a tile could already reach would prove nothing"
        );
        let (order, grouped) = grouping
            .group(&device, &chosen, EXPERTS)
            .expect("the dispatch completes");
        assert_ne!(
            order,
            (0..ROWS as u32).collect::<Vec<u32>>(),
            "the identity"
        );
        assert!(
            grouped
                .chunk_by(|a, b| a == b)
                .any(|run| run.len() > ROWS_A_TILE),
            "no run outlasts a tile, so nothing here exercises a whole one: {grouped:?}"
        );
        assert_ne!(
            OUT_DIM % COLS_A_TILE,
            0,
            "columns that filled their last tile would not exercise the partial one"
        );

        // What `gate` and `up` are handed: one row of the hidden state per
        // token, read `TOP_K` times over.
        let tokens = &case.x[..TOKENS * IN_DIM];
        let ungrouped = |x: &[f32], per_source: usize| {
            let mut chosen = device.buffer(&chosen).expect("the selection uploads");
            let mut x = device.buffer(x).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let pending = bank
                .encode_picked(&mut batch, &mut chosen, &mut x, per_source)
                .expect("the dispatch encodes");
            batch.wait().expect("the dispatch completes");
            pending.take()
        };
        let regrouped = |x: &[f32], per_source: usize, through: Through| {
            let mut selection = device.buffer(&chosen).expect("the selection uploads");
            let mut x = device.buffer(x).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let mut sorted = grouping
                .encode(&mut batch, &mut selection, EXPERTS)
                .expect("the grouping encodes");
            let pending = bank
                .encode_grouped(&mut batch, &mut sorted, &mut x, per_source, through)
                .expect("the dispatch encodes");
            batch.wait().expect("the dispatch completes");
            pending.take()
        };
        let row = |rows: &[f32], i: usize| rows[i * OUT_DIM..][..OUT_DIM].to_vec();

        let want = ungrouped(tokens, TOP_K);
        let gathered = regrouped(tokens, TOP_K, Through::Gathered);
        assert_eq!(gathered.len(), want.len());
        for (at, from) in order.iter().enumerate() {
            assert_eq!(
                row(&gathered, at),
                row(&want, *from as usize),
                "grouped row {at} is the call's row {from}"
            );
        }

        // What `down` is handed: the rows the pair before it produced, which are
        // already in the grouping's order.
        let want = ungrouped(&case.x, 1);
        let sorted: Vec<f32> = order
            .iter()
            .flat_map(|from| case.x[*from as usize * IN_DIM..][..IN_DIM].to_vec())
            .collect();
        let scattered = regrouped(&sorted, 1, Through::Scattered);
        assert_eq!(
            scattered, want,
            "the rows did not come back where they went"
        );
        assert_ne!(
            row(&want, 0),
            row(&want, 1),
            "rows that agreed would prove nothing"
        );
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
    /// **The rates here rank the widths and do not state a bandwidth.** One
    /// weight is dispatched against `CALLS` times in a row, and the weights at
    /// these shapes are a few megabytes, so what the second call reads is what
    /// the first left in cache — where the banks a step routes through are six
    /// of 256 out of 137 GB and are cold every time. A cold read of the same
    /// kernel is the 299 GB/s
    /// `one_dispatch_does_an_lm_head_shaped_multiply_without_meeting_the_watchdog`
    /// measures over 0.41 GiB. Both are the same arithmetic, and a table that
    /// mixed them would be a table about the cache.
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
                    best = best.min(crate::testing::device_time(&device, CALLS, |batch| {
                        projection
                            .encode_over(batch, &mut x)
                            .expect("the dispatch encodes");
                    }));
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

    /// **What a packed multiply costs the device at each height a tile reads**,
    /// and the sweep [`ROWS_A_TILE`] was chosen from.
    ///
    /// The shapes are the ones a *prefill* dispatches, because that is where a
    /// tile has anything to share: the two lengths the profile is taken at, by
    /// the two layouts whose rows name one expert. A tile of one is the untiled
    /// kernel and is the column the rest are read against.
    ///
    /// **The rate is over the bytes a tiled call actually reads**, which is a
    /// weight a tile rather than a weight a row — so the column does not climb
    /// merely because the denominator fell, and a height that read the same
    /// bytes in the same time reports the same number. What it ranks is
    /// throughput at a fixed amount of *work*, and the wall time behind it is
    /// what falls.
    ///
    /// Nothing asserts a rate. What is asserted is that the shipped height was
    /// among the ones tried, the way the width sweep above does.
    ///
    /// The same caution applies as to that sweep, and harder: one weight is
    /// dispatched against `CALLS` times in a row, so the table ranks the
    /// heights and does not state a bandwidth.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_height_a_tile_reads() {
        let Some(device) = device() else { return };
        const CALLS: usize = 8;
        const ROUNDS: usize = 3;
        const HEIGHTS: [usize; 6] = [1, 2, 3, 4, 6, 8];

        // `(what, in_dim, out_dim, rows)`.
        let shapes = [
            ("q_proj, 385 tokens", 4096, 4096, 385),
            ("k_proj, 385 tokens", 4096, 1024, 385),
            ("shared gate/up, 385 tokens", 4096, 2048, 2 * 385),
            ("shared down, 385 tokens", 2048, 4096, 2 * 385),
            ("q_proj, 769 tokens", 4096, 4096, 769),
            ("shared gate/up, 769 tokens", 4096, 2048, 2 * 769),
        ];

        assert!(HEIGHTS.contains(&ROWS_A_TILE), "{ROWS_A_TILE}");
        eprintln!(
            "  {:<28}{}",
            "shape",
            HEIGHTS
                .iter()
                .map(|rows| format!("{:>16}", format!("{rows} rows a tile")))
                .collect::<String>()
        );
        for (what, in_dim, out_dim, rows) in shapes {
            let case = Case::seeded(1, in_dim, out_dim, rows);
            let mut x = device.buffer(&case.x).expect("the rows upload");
            let mut line = format!("  {what:<28}");
            for height in HEIGHTS {
                let matmul = PackedMatmul::tiling(
                    &device,
                    &a_tile_of(height, COLS_A_TILE),
                    height,
                    COLS_A_TILE,
                )
                .unwrap_or_else(|err| panic!("{height} rows a tile compiles: {err}"));
                let projection = case.upload(&device, &matmul);

                let mut best = Duration::MAX;
                for _ in 0..ROUNDS {
                    best = best.min(crate::testing::device_time(&device, CALLS, |batch| {
                        projection
                            .encode_over(batch, &mut x)
                            .expect("the dispatch encodes");
                    }));
                }

                let codes = rows.div_ceil(height) * out_dim * in_dim;
                let moved = (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
                line.push_str(&format!(
                    "{:>16}",
                    format!(
                        "{:.0}µs {:.0} GB/s",
                        1e6 * best.as_secs_f64() / CALLS as f64,
                        moved / best.as_secs_f64() / 1e9
                    )
                ));
            }
            eprintln!("{line}");
        }
    }

    /// **What a packed multiply costs the device at each width a tile spans**,
    /// and the sweep [`COLS_A_TILE`] was chosen from.
    ///
    /// The same shapes and the same reading as the height sweep above, over the
    /// other axis of the same tile and at the shipped height. A tile one column
    /// wide is the row tile exactly, so that column is the before of this change
    /// rather than a column-tiled kernel imitating it.
    ///
    /// **The rate here is over bytes that do not move with the width**, which is
    /// the difference between this sweep and the one above and is the whole
    /// point of the change. A column is its own weight row, so the columns of a
    /// tile share no weight byte and the denominator is the same at every width
    /// — what the column reports is throughput at a fixed amount of work *and*
    /// a fixed number of bytes, so a width that is faster is faster.
    ///
    /// Nothing asserts a rate. What is asserted is that the shipped width was
    /// among the ones tried, the way the two sweeps above do.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_width_a_tile_spans() {
        let Some(device) = device() else { return };
        const CALLS: usize = 8;
        const ROUNDS: usize = 3;
        const WIDTHS: [usize; 6] = [1, 2, 3, 4, 6, 8];

        // `(what, in_dim, out_dim, rows)`.
        let shapes = [
            ("q_proj, 385 tokens", 4096, 4096, 385),
            ("k_proj, 385 tokens", 4096, 1024, 385),
            ("shared gate/up, 385 tokens", 4096, 2048, 2 * 385),
            ("shared down, 385 tokens", 2048, 4096, 2 * 385),
            ("q_proj, 769 tokens", 4096, 4096, 769),
            ("shared gate/up, 769 tokens", 4096, 2048, 2 * 769),
        ];

        assert!(WIDTHS.contains(&COLS_A_TILE), "{COLS_A_TILE}");
        eprintln!(
            "  {:<28}{}",
            "shape",
            WIDTHS
                .iter()
                .map(|cols| format!("{:>16}", format!("{cols} cols a tile")))
                .collect::<String>()
        );
        for (what, in_dim, out_dim, rows) in shapes {
            let case = Case::seeded(1, in_dim, out_dim, rows);
            let mut x = device.buffer(&case.x).expect("the rows upload");
            let mut line = format!("  {what:<28}");
            for cols in WIDTHS {
                let matmul =
                    PackedMatmul::tiling(&device, &a_tile_of(ROWS_A_TILE, cols), ROWS_A_TILE, cols)
                        .unwrap_or_else(|err| panic!("{cols} columns a tile compiles: {err}"));
                let projection = case.upload(&device, &matmul);

                let mut best = Duration::MAX;
                for _ in 0..ROUNDS {
                    best = best.min(crate::testing::device_time(&device, CALLS, |batch| {
                        projection
                            .encode_over(batch, &mut x)
                            .expect("the dispatch encodes");
                    }));
                }

                let codes = rows.div_ceil(ROWS_A_TILE) * out_dim * in_dim;
                let moved = (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
                line.push_str(&format!(
                    "{:>16}",
                    format!(
                        "{:.0}µs {:.0} GB/s",
                        1e6 * best.as_secs_f64() / CALLS as f64,
                        moved / best.as_secs_f64() / 1e9
                    )
                ));
            }
            eprintln!("{line}");
        }
    }

    /// **What a tile shape costs to compile, and what it leaves a
    /// threadgroup**, which are the two prices a wider tile pays that the sweep
    /// above cannot see.
    ///
    /// **The first is here because P3 measured a compile this side had not
    /// thought to price.** Putting a third entry in this source cost a prefill 2
    /// to 3% of its device time before any call reached it, so a change that
    /// widens a tile owes the same figure — the more so at 97 tokens, where a
    /// fixed cost is a larger share of a shorter prefill. A column tile is not a
    /// fourth entry: it is the same three, of which two got wider bodies. This
    /// is what that is worth in wall time, taken over the whole source rather
    /// than one entry because [`PackedMatmul::new`] is what a model load runs.
    ///
    /// **The second is where the sweep's cliff was expected to be, and is
    /// not.** Every accumulator a tile carries is a register a lane holds for
    /// the whole walk, and a pipeline that wants more of them than the hardware
    /// will give a thread reports a narrower threadgroup — which
    /// [`Batch::add`](crate::kernel::Batch) refuses rather than clamps, so a
    /// tile too wide for [`THREADS_PER_GROUP`] would be a failure rather than a
    /// slow kernel. The column that reports it is flat at the device's own 1024
    /// from one column to eight, including the width that is half the speed of
    /// four — so whatever the sweep turns on, it is not a limit this side can
    /// read off the pipeline, and the guard below is a guard rather than an
    /// explanation.
    ///
    /// Nothing asserts a duration. What is asserted is that the shipped shape
    /// still fits the threadgroup this module dispatches in.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_tile_shape_costs_to_compile_and_what_it_leaves_a_threadgroup() {
        let Some(device) = device() else { return };
        const WIDTHS: [usize; 5] = [1, 2, 4, 6, 8];

        // The first compile a process runs carries the compiler's own start-up
        // and reads five times what the ones behind it do, which would land the
        // whole of it on whichever width this loop happened to try first.
        PackedMatmul::new(&device).expect("the shipped source compiles");
        eprintln!(
            "  {:<16}{:>12}{:>16}{:>18}",
            "cols a tile", "compiling", "the tiled entry", "threads a group"
        );
        for cols in WIDTHS {
            let source = a_tile_of(ROWS_A_TILE, cols);
            let taken = Instant::now();
            PackedMatmul::tiling(&device, &source, ROWS_A_TILE, cols)
                .unwrap_or_else(|err| panic!("{cols} columns a tile compiles: {err}"));
            let whole = taken.elapsed();

            let taken = Instant::now();
            let tiled = device
                .compile(&source, TILED_ENTRY)
                .expect("the tiled entry compiles");
            let alone = taken.elapsed();

            eprintln!(
                "  {:<16}{:>12}{:>16}{:>18}",
                cols,
                format!("{whole:.0?}"),
                format!("{alone:.0?}"),
                tiled.max_threads_per_group(),
            );
            if cols == COLS_A_TILE {
                assert!(
                    tiled.max_threads_per_group() >= THREADS_PER_GROUP,
                    "the shipped tile leaves a threadgroup {} threads where this module \
                     dispatches {THREADS_PER_GROUP}",
                    tiled.max_threads_per_group()
                );
            }
        }
    }

    /// How many weight rows a call whose experts are `experts` actually reads,
    /// counted the way `packed_matmul_rows` and `packed_matmul_grouped` read
    /// them: one for a tile whose rows agree about the expert, and one per row
    /// for a tile whose rows do not.
    ///
    /// The oracle for what [`PackedBank::moves`] declares, and the only thing
    /// that can be: the declared figure charges one weight a tile, and what a
    /// grouped call reads depends on where the runs the routing made happen to
    /// fall against the tile boundaries — which is a property of the selection
    /// and not of the shape.
    fn weights_read(experts: &[u32], rows_a_tile: usize) -> usize {
        experts
            .chunks(rows_a_tile)
            .map(|tile| match tile.iter().all(|expert| *expert == tile[0]) {
                true => 1,
                false => tile.len(),
            })
            .sum()
    }

    /// A selection of the shape a router writes at prefill: `TOP_K` experts a
    /// token out of `experts`, no two of a token's the same.
    fn prefill_routing(tokens: usize, experts: usize, top_k: usize) -> Vec<u32> {
        (0..tokens)
            .flat_map(|token| {
                let first = token * 37 % experts;
                (0..top_k).map(move |slot| ((first + slot * 41) % experts) as u32)
            })
            .collect()
    }

    /// **What a grouped call actually reads against what it declares**, which is
    /// the one figure in the bandwidth table this side cannot state exactly.
    ///
    /// [`PackedBank::moves`] charges a grouped call the worst layout its shape
    /// allows — one straddling tile per expert, each reading [`ROWS_A_TILE`]
    /// weights — because the expert each row named is in device memory and this
    /// side never sees it. This is how loose that is against the layout the
    /// device actually produces, at the three shapes a prefill gives a routed
    /// bank, beside the untiled call the grouping replaces.
    ///
    /// **The 97-token row is the finding and it is a negative one.** At 2.3 rows
    /// an expert the runs are shorter than a tile, nearly every tile straddles,
    /// and a grouped call reads 581 weights where the untiled one reads 582 —
    /// so whatever a 97-token prefill gains from being grouped, it is not bytes.
    /// The saving arrives at 385 and 769, where the runs are 9 and 18 rows.
    ///
    /// Nothing asserts a ratio. What is asserted is that the declared figure is
    /// a bound in the direction it claims: never below what the kernel reads,
    /// and never above what the untiled call reads — the first is what keeps the
    /// bandwidth column from flattering this change, and the second is what
    /// keeps it from reporting a loss the kernel cannot have.
    #[test]
    fn what_a_grouped_call_reads_against_what_it_declares() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const EXPERTS: usize = 256;
        const ROUTED_TOP_K: usize = 6;

        eprintln!(
            "  {:<14}{:>8}{:>12}{:>10}{:>10}{:>12}",
            "tokens", "rows", "a run", "declared", "read", "untiled"
        );
        for tokens in [97usize, 385, 769] {
            let chosen = prefill_routing(tokens, EXPERTS, ROUTED_TOP_K);
            let (_, grouped) = grouping
                .group(&device, &chosen, EXPERTS)
                .expect("the dispatch completes");

            let rows = chosen.len();
            let tiles = rows.div_ceil(ROWS_A_TILE);
            let declared = rows.min(tiles + (ROWS_A_TILE - 1) * tiles.min(EXPERTS - 1));
            let read = weights_read(&grouped, ROWS_A_TILE);
            let runs = grouped.chunk_by(|a, b| a == b).count();
            eprintln!(
                "  {:<14}{rows:>8}{:>12}{declared:>10}{read:>10}{rows:>12}",
                format!("{tokens}"),
                format!("{:.1}", rows as f64 / runs as f64),
            );

            assert!(
                read <= declared && declared <= rows,
                "{tokens} tokens: read {read}, declared {declared}, untiled {rows}"
            );
        }
    }

    /// **What grouping a bank's rows by expert is worth at each length of run
    /// it produces**, and the sweep [`RUNS_A_GROUPING`] was chosen from.
    ///
    /// The rows a routed bank runs are the tokens six times over and the runs
    /// the sort makes of them are those over the bank's experts — 2.3 rows an
    /// expert at 97 tokens, 9 at 385 and 18 at 769 — so the run length is what a
    /// grouping is worth at, and the token count is only how a caller reaches
    /// one. That is what this sweeps, over a bank narrow enough in experts to
    /// hold in a test and a shape wide enough to be the one a layer dispatches.
    ///
    /// **A run of one is the case that must not pay**, because it is a decode
    /// step's: six rows over 256 experts sort into runs of one and a tile of
    /// them shares nothing at all, so what the column says is the price of the
    /// sort and the tile's registers with nothing bought.
    ///
    /// The same caution as the two sweeps above, and harder: one weight is
    /// dispatched against `CALLS` times in a row, so the table ranks the run
    /// lengths and does not state a bandwidth.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_grouping_a_banks_rows_by_expert_is_worth_at_each_run_length() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const CALLS: usize = 8;
        const ROUNDS: usize = 3;
        const EXPERTS: usize = 16;
        const IN: usize = 4096;
        const OUT: usize = 2048;
        const RUNS: [usize; 7] = [1, 2, 3, 4, 6, 9, 18];

        eprintln!(
            "  {:<16}{:>8}{:>14}{:>14}{:>10}",
            "rows an expert", "rows", "ungrouped", "grouped", "reads"
        );
        for run in RUNS {
            let rows = EXPERTS * run;
            let case = Case::seeded(1, IN, EXPERTS * OUT, rows);
            let bank = PackedBank::upload(
                &device,
                &matmul,
                EXPERTS,
                IN,
                OUT,
                &case.packed(),
                &case.scales,
            )
            .expect("the bank's shapes pair");

            // Round-robin, which is a routed bank's own layout: consecutive rows
            // name different experts, so no tile can share a read until the sort
            // moves them.
            let chosen: Vec<u32> = (0..rows).map(|row| (row % EXPERTS) as u32).collect();
            let mut x = device.buffer(&case.x).expect("the rows upload");
            let mut best = [Duration::MAX; 2];
            for _ in 0..ROUNDS {
                for (at, grouped) in [false, true].into_iter().enumerate() {
                    let taken = crate::testing::device_time(&device, CALLS, |batch| {
                        let mut picked = device.buffer(&chosen).expect("the selection uploads");
                        match grouped {
                            false => bank
                                .encode_picked(batch, &mut picked, &mut x, 1)
                                .expect("the dispatch encodes"),
                            true => {
                                let mut sorted = grouping
                                    .encode(batch, &mut picked, EXPERTS)
                                    .expect("the grouping encodes");
                                bank.encode_grouped(
                                    batch,
                                    &mut sorted,
                                    &mut x,
                                    1,
                                    Through::Gathered,
                                )
                                .expect("the dispatch encodes")
                            }
                        };
                    });
                    best[at] = best[at].min(taken);
                }
            }

            let mut sorted = chosen.clone();
            sorted.sort_unstable();
            eprintln!(
                "  {:<16}{rows:>8}{:>14}{:>14}{:>10}",
                run,
                format!("{:.0}µs", 1e6 * best[0].as_secs_f64()),
                format!("{:.0}µs", 1e6 * best[1].as_secs_f64()),
                format!("{}/{rows}", weights_read(&sorted, ROWS_A_TILE)),
            );
        }
        assert!(RUNS.contains(&RUNS_A_GROUPING), "{RUNS_A_GROUPING}");
    }

    /// **What separates the two tiled entries**, which is the largest
    /// unexplained number this repo's prefill table has carried:
    /// `packed_matmul_rows` reports 281 GB/s where `packed_matmul_grouped`
    /// reports 556, at the same tile shape.
    ///
    /// **The two cannot differ in the walk.** They are one source string with
    /// three expressions substituted — the binding the order arrives in, the row
    /// of the input a tile reads and the row of the output it writes — and in
    /// nothing else at all. So what is left is the call each of them is given,
    /// and the two things that differ about those are crossed here rather than
    /// argued about:
    ///
    /// - **the weight a dispatch walks.** `packed_matmul_rows` runs the
    ///   projections and the shared banks, whose whole weight is 2 to 34 MB;
    ///   `packed_matmul_grouped` runs the routed banks, whose weight is 256
    ///   experts and 1.07 GB.
    /// - **the indirection.** A grouped call reads its input through a
    ///   permutation another dispatch wrote.
    ///
    /// One shape held fixed at a 769-token routed bank's, over banks of 1 to 256
    /// experts, through both entries with the untiled kernel beside them. **The
    /// rows are already sorted**, so the grouping is the identity and the tiles
    /// the two entries see are the same tiles — which is what makes the pair of
    /// columns the indirection on its own.
    ///
    /// **And the rate is the one the profile table prints**, which is
    /// [`PackedBank::moves`] over device time — a whole weight charged per
    /// *tile*. A bank of one expert declares 288 times the weight it has and a
    /// bank of 256 declares 7.5 times, so the `distinct` column beside it is what
    /// says whether a rate is a bandwidth or a cache.
    ///
    /// Nothing asserts a rate. What is asserted is that the three arms answer
    /// the same thing, because a column that got faster by computing less would
    /// explain nothing.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_separates_a_tiled_call_from_a_grouped_one() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const CALLS: usize = 4;
        const ROUNDS: usize = 3;
        const IN: usize = 4096;
        const OUT: usize = 2048;
        /// A 769-token prefill's routed bank: six rows a token.
        const ROWS: usize = 769 * 6;
        const BANKS: [usize; 5] = [1, 4, 16, 64, 256];

        eprintln!(
            "  {:<10}{:>10}{:>11}{:>11}{:>12}{:>12}{:>13}{:>13}",
            "experts", "a run", "distinct", "untiled", "tiled", "grouped", "tiled", "grouped"
        );
        for experts in BANKS {
            let case = Case::seeded(1, IN, experts * OUT, ROWS);
            let bank = PackedBank::upload(
                &device,
                &matmul,
                experts,
                IN,
                OUT,
                &case.packed(),
                &case.scales,
            )
            .expect("the bank's shapes pair");

            // Sorted, so that the tiles are uniform and the grouping is the
            // identity: what is being separated is the indirection from the
            // weight, and a layout only one of the two entries can tile would
            // confound them.
            let chosen: Vec<u32> = (0..ROWS).map(|row| (row * experts / ROWS) as u32).collect();
            let mut x = device.buffer(&case.x).expect("the rows upload");

            let mut best = [Duration::MAX; 3];
            for _ in 0..ROUNDS {
                for (at, arm) in [Arm::Untiled, Arm::Tiled, Arm::Grouped]
                    .into_iter()
                    .enumerate()
                {
                    let taken = crate::testing::device_time(&device, CALLS, |batch| {
                        let mut picked = device.buffer(&chosen).expect("the selection uploads");
                        match arm {
                            Arm::Untiled => bank
                                .encode_picked(batch, &mut picked, &mut x, 1)
                                .expect("the dispatch encodes"),
                            Arm::Tiled => bank
                                .encode_over(batch, &chosen, &mut x)
                                .expect("the dispatch encodes"),
                            Arm::Grouped => {
                                let mut sorted = grouping
                                    .encode(batch, &mut picked, experts)
                                    .expect("the grouping encodes");
                                bank.encode_grouped(
                                    batch,
                                    &mut sorted,
                                    &mut x,
                                    1,
                                    Through::Gathered,
                                )
                                .expect("the dispatch encodes")
                            }
                        };
                    });
                    best[at] = best[at].min(taken);
                }
            }

            let tiles = ROWS.div_ceil(ROWS_A_TILE);
            let rate = |read: usize, taken: Duration| {
                let codes = read * OUT * IN;
                let moved = (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
                format!(
                    "{:.0} GB/s",
                    moved / taken.as_secs_f64() * CALLS as f64 / 1e9
                )
            };
            let held = experts * OUT * IN;
            eprintln!(
                "  {experts:<10}{:>10}{:>11}{:>11}{:>12}{:>12}{:>13}{:>13}",
                format!("{}", ROWS / experts),
                format!(
                    "{:.0} MB",
                    (held / CODES_PER_BYTE + held / GROUP_SIZE) as f64 / 1e6
                ),
                format!("{:.0}µs", 1e6 * best[0].as_secs_f64() / CALLS as f64),
                format!("{:.0}µs", 1e6 * best[1].as_secs_f64() / CALLS as f64),
                format!("{:.0}µs", 1e6 * best[2].as_secs_f64() / CALLS as f64),
                rate(tiles, best[1]),
                rate(
                    ROWS.min(tiles + (ROWS_A_TILE - 1) * tiles.min(experts - 1)),
                    best[2]
                ),
            );

            // The three arms are three ways of cutting the same multiply up, so
            // a difference between their answers would be a difference in what
            // was measured rather than in how fast it ran.
            let [untiled, tiled, grouped] = together(&device, |batch| {
                let mut picked = device.buffer(&chosen)?;
                let untiled = bank.encode_picked(batch, &mut picked, &mut x, 1)?;
                let tiled = bank.encode_over(batch, &chosen, &mut x)?;
                let mut sorted = grouping.encode(batch, &mut picked, experts)?;
                let grouped =
                    bank.encode_grouped(batch, &mut sorted, &mut x, 1, Through::Gathered)?;
                Ok([untiled, tiled, grouped])
            })
            .expect("the three dispatches run");
            assert_eq!(tiled, untiled, "a bank of {experts} tiled");
            assert_eq!(grouped, untiled, "a bank of {experts} grouped");
        }
    }

    /// Which way one call of the sweep above cuts the multiply up.
    #[derive(Debug, Clone, Copy)]
    enum Arm {
        Untiled,
        Tiled,
        Grouped,
    }

    /// **What each shape a prefill gives this kernel actually costs**, which is
    /// what the two rows of the profile table are made of.
    ///
    /// `packed_matmul_rows` is 336 calls and `packed_matmul_grouped` is 120, and
    /// neither is one shape: the first is five projections a layer, both halves
    /// of a shared bank and the two dense feed-forward networks, and the second
    /// is a routed bank's three. The sweep above says the two *entries* are
    /// within 6% of each other at one shape, so whatever separates the two rows
    /// is in here — and a row is only diagnosable once the shapes under it are
    /// named.
    ///
    /// **The `a prefill` column is the check.** It is this shape's own device
    /// time times how many of it a 769-token prefill dispatches, so the column
    /// sums to the two rows of the profile table if the decomposition is right
    /// and does not if it is not.
    ///
    /// One expert for a projection, two for a shared bank and 256 for a routed
    /// one, because that is what each of them has — and the rows are sorted, so
    /// each shape reaches the entry the engine would put it through.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_each_shape_of_a_prefills_packed_calls_costs() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const CALLS: usize = 4;
        const ROUNDS: usize = 3;
        const TOKENS: usize = 769;

        // `(what, rows, in_dim, out_dim, experts, calls a prefill, grouped)`.
        // The counts are the engine's: 42 layers of five projections, 40 MoE
        // layers of a shared bank's three and a routed bank's three, and two
        // dense ones. **Which entry is spelled out rather than asked of
        // `PackedMatmul::groups`**, which answers about a routed bank's layout
        // and says yes to a projection's 769 rows over one expert — a call the
        // engine never puts through that entry at all.
        let shapes = [
            ("q_proj, o_proj", TOKENS, 4096, 4096, 1, 84, false),
            ("k_proj, v_proj", TOKENS, 4096, 1024, 1, 84, false),
            ("r_proj", TOKENS, 4096, 512, 1, 42, false),
            ("shared gate, up", 2 * TOKENS, 4096, 2048, 2, 80, false),
            ("shared down", 2 * TOKENS, 2048, 4096, 2, 40, false),
            ("dense gate, up", TOKENS, 4096, 16384, 1, 4, false),
            ("dense down", TOKENS, 16384, 4096, 1, 2, false),
            ("routed gate, up", 6 * TOKENS, 4096, 2048, 256, 80, true),
            ("routed down", 6 * TOKENS, 2048, 4096, 256, 40, true),
        ];

        eprintln!(
            "  {:<18}{:>8}{:>10}{:>10}{:>8}{:>10}{:>12}{:>12}",
            "shape", "rows", "distinct", "declared", "over", "a call", "achieved", "a prefill"
        );
        let mut totals = [Duration::ZERO; 2];
        for (what, rows, in_dim, out_dim, experts, calls, groups) in shapes {
            let case = Case::seeded(1, in_dim, experts * out_dim, rows);
            let bank = PackedBank::upload(
                &device,
                &matmul,
                experts,
                in_dim,
                out_dim,
                &case.packed(),
                &case.scales,
            )
            .expect("the bank's shapes pair");
            let chosen: Vec<u32> = (0..rows).map(|row| (row * experts / rows) as u32).collect();
            let mut x = device.buffer(&case.x).expect("the rows upload");

            let mut best = Duration::MAX;
            for _ in 0..ROUNDS {
                best = best.min(crate::testing::device_time(&device, CALLS, |batch| {
                    let mut picked = device.buffer(&chosen).expect("the selection uploads");
                    match groups {
                        false => bank
                            .encode_over(batch, &chosen, &mut x)
                            .expect("the dispatch encodes"),
                        true => {
                            let mut sorted = grouping
                                .encode(batch, &mut picked, experts)
                                .expect("the grouping encodes");
                            bank.encode_grouped(batch, &mut sorted, &mut x, 1, Through::Gathered)
                                .expect("the dispatch encodes")
                        }
                    };
                }));
            }

            let call = best / CALLS as u32;
            let tiles = rows.div_ceil(ROWS_A_TILE);
            let read = match groups {
                false => tiles,
                true => rows.min(tiles + (ROWS_A_TILE - 1) * tiles.min(experts - 1)),
            };
            let bytes = |codes: usize| (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
            let moved = bytes(read * out_dim * in_dim);
            // What the call could read at most, which is the bank itself. The
            // ratio between the two is how many times over a dispatch is charged
            // for a weight it may still have in cache, and it is the column that
            // says whether `achieved` is a bandwidth or an amplification.
            let held = bytes(experts * out_dim * in_dim);
            totals[usize::from(groups)] += call * calls as u32;
            eprintln!(
                "  {what:<18}{rows:>8}{:>10}{:>10}{:>8}{:>10}{:>12}{:>12}",
                format!("{:.0} MB", held / 1e6),
                format!("{:.0} MB", moved / 1e6),
                format!("×{:.0}", moved / held),
                format!("{:.0}µs", 1e6 * call.as_secs_f64()),
                format!("{:.0} GB/s", moved / call.as_secs_f64() / 1e9),
                format!("{:.2?}", call * calls as u32),
            );
        }
        eprintln!(
            "  the shapes on the tiled entry are {:.2?} of a prefill and the grouped ones {:.2?}",
            totals[0], totals[1]
        );
    }

    /// **Whether the tile height still turns at four once the bank is too big to
    /// cache**, which is the one thing [`ROWS_A_TILE`]'s own sweep could not ask.
    ///
    /// That sweep runs over `Case::seeded(1, ..)` — a single expert, 2 to 36 MB,
    /// which this machine holds in cache and re-reads for free. Every extra row a
    /// tile carries there buys nothing, because the weight read it saves was
    /// already a cache hit, and it costs registers — so the sweep turns at four
    /// and says so emphatically. **A routed bank is 1141 MB a dispatch and 129 GB
    /// across a prefill**, where a saved weight read is a saved trip to memory
    /// rather than a saved hit, and the trade could land somewhere else entirely.
    ///
    /// So: the same height sweep, over the bank a 769-token prefill's routed gate
    /// actually is. If the turn moves outward here, `ROWS_A_TILE` was fitted on
    /// the wrong shape.
    ///
    /// **A taller tile is the safe direction and a shorter one is not.** [`tiles`]
    /// refuses a run shorter than the height, so raising it can only take calls
    /// *out* of the tiled path — and the three rows a `k = 2` verify dispatches
    /// stay untiled at four and at anything above it. Nothing here lowers it.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn whether_the_tile_height_turns_at_four_on_a_bank_too_big_to_cache() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const CALLS: usize = 4;
        const ROUNDS: usize = 3;
        const IN: usize = 4096;
        const OUT: usize = 2048;
        const EXPERTS: usize = 256;
        /// A 769-token prefill's routed bank: six rows a token.
        const ROWS: usize = 769 * 6;
        const HEIGHTS: [usize; 5] = [2, 3, 4, 6, 8];

        assert!(HEIGHTS.contains(&ROWS_A_TILE), "{ROWS_A_TILE}");
        let case = Case::seeded(1, IN, EXPERTS * OUT, ROWS);
        let packed = case.packed();
        let chosen: Vec<u32> = (0..ROWS).map(|row| (row * EXPERTS / ROWS) as u32).collect();

        eprintln!(
            "  {:<16}{:>10}{:>10}{:>12}{:>12}",
            "rows a tile", "a call", "declared", "achieved", "a prefill"
        );
        for height in HEIGHTS {
            let matmul = PackedMatmul::tiling(
                &device,
                &a_tile_of(height, COLS_A_TILE),
                height,
                COLS_A_TILE,
            )
            .unwrap_or_else(|err| panic!("{height} rows a tile compiles: {err}"));
            let bank =
                PackedBank::upload(&device, &matmul, EXPERTS, IN, OUT, &packed, &case.scales)
                    .expect("the bank's shapes pair");
            let mut x = device.buffer(&case.x).expect("the rows upload");

            let mut best = Duration::MAX;
            for _ in 0..ROUNDS {
                best = best.min(crate::testing::device_time(&device, CALLS, |batch| {
                    let mut picked = device.buffer(&chosen).expect("the selection uploads");
                    let mut sorted = grouping
                        .encode(batch, &mut picked, EXPERTS)
                        .expect("the grouping encodes");
                    bank.encode_grouped(batch, &mut sorted, &mut x, 1, Through::Gathered)
                        .expect("the dispatch encodes");
                }));
            }

            let call = best / CALLS as u32;
            let tiles = ROWS.div_ceil(height);
            let read = ROWS.min(tiles + (height - 1) * tiles.min(EXPERTS - 1));
            let codes = read * OUT * IN;
            let moved = (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
            eprintln!(
                "  {height:<16}{:>10}{:>10}{:>12}{:>12}",
                format!("{:.0}µs", 1e6 * call.as_secs_f64()),
                format!("{:.0} MB", moved / 1e6),
                format!("{:.0} GB/s", moved / call.as_secs_f64() / 1e9),
                // 120 routed dispatches a prefill, which is what this shape is.
                format!("{:.2?}", call * 120),
            );
        }
    }

    /// **How many times a long prefill reads one expert's weight, against how
    /// many times it must** — a question about [`ROWS_A_TILE`] rather than about
    /// the grouping, and one the two sweeps above ask at a prompt where it does
    /// not arise.
    ///
    /// A routed bank runs six rows a token over 256 experts, so an expert is
    /// named by `6n/256` rows: 2.3 at a 97-token prompt, 18 at 769 and **384 at
    /// 16384**. A tile is four rows, so at the long end one expert's weight is
    /// walked 96 times where once would serve every row that named it. The
    /// arithmetic is not in doubt; what it is worth is, and only a shape with the
    /// runs that long can say.
    ///
    /// So: one bank of 256 experts, too big to cache at 1.07 GB, dispatched at
    /// each of four run lengths, and the column that answers it is **device time
    /// a row**. A tile reads `out_dim × in_dim / 4` codes per row whatever the
    /// run, so the *declared* bytes a row are flat by construction — and the
    /// bytes that have to come from memory are not: at four rows an expert the
    /// weight is fetched once per four rows, and at 384 the same fetch could
    /// serve 384 if the cache holds it between the 96 tiles that want it. A time
    /// a row that falls with the run is that cache; one that is flat is a kernel
    /// the weight reads were never the bound on.
    ///
    /// **The sort is measured and taken off rather than carried**, which matters
    /// here in a way it does not in the sweeps above: `group_by_expert` is a pass
    /// over the rows, so it grows with the arm while the question is about the
    /// weight, and left in it would be 96-fold re-reading and a linear term
    /// charged to the same column. The two are dispatched apart and the
    /// difference is what the table prints.
    ///
    /// The same caution as every sweep here: one weight is dispatched against
    /// repeatedly, so this ranks the run lengths and does not state a bandwidth.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn how_often_a_long_prefill_reads_one_experts_weight() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        // Four dispatches to a command buffer, so that what the device's clock
        // is divided by is the dispatch rather than the buffer around it — the
        // reason every sweep above takes the same number.
        const CALLS: usize = 4;
        const ROUNDS: usize = 3;
        const IN: usize = 4096;
        const OUT: usize = 2048;
        const EXPERTS: usize = 256;
        /// Rows an expert, at six of 256 a token: 4 is a 171-token prompt, 24 a
        /// 1024-token one, 96 a 4096-token one and 384 a 16384-token one.
        const RUNS: [usize; 4] = [4, 24, 96, 384];

        let most = EXPERTS * RUNS[RUNS.len() - 1];
        let case = Case::seeded(1, IN, EXPERTS * OUT, most);
        let bank = PackedBank::upload(
            &device,
            &matmul,
            EXPERTS,
            IN,
            OUT,
            &case.packed(),
            &case.scales,
        )
        .expect("the bank's shapes pair");
        let bytes = |codes: usize| (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64;
        let held = bytes(EXPERTS * OUT * IN);

        eprintln!(
            "  {:<16}{:>10}{:>10}{:>12}{:>12}{:>12}",
            "rows an expert", "rows", "reads", "a call", "a row", "declared"
        );
        for run in RUNS {
            let rows = EXPERTS * run;
            // Sorted already, so the tiles this measures are the tiles the sort
            // produces and no tile straddles two experts by accident of the
            // routing — which is what makes `reads` the run length's own figure.
            let chosen: Vec<u32> = (0..rows).map(|row| (row * EXPERTS / rows) as u32).collect();
            let mut x = device
                .buffer(&case.x[..rows * IN])
                .expect("the rows upload");

            let mut best = [Duration::MAX; 2];
            for _ in 0..ROUNDS {
                for (at, multiplies) in [false, true].into_iter().enumerate() {
                    let taken = crate::testing::device_time(&device, CALLS, |batch| {
                        let mut picked = device.buffer(&chosen).expect("the selection uploads");
                        let mut sorted = grouping
                            .encode(batch, &mut picked, EXPERTS)
                            .expect("the grouping encodes");
                        if multiplies {
                            bank.encode_grouped(batch, &mut sorted, &mut x, 1, Through::Gathered)
                                .expect("the dispatch encodes");
                        }
                    });
                    best[at] = best[at].min(taken);
                }
            }

            let call = best[1].saturating_sub(best[0]);
            let reads = weights_read(&chosen, ROWS_A_TILE) / EXPERTS;
            eprintln!(
                "  {run:<16}{rows:>10}{:>12}{:>12}{:>12}{:>12}",
                format!("{reads}×"),
                format!("{:.1}ms", 1e3 * call.as_secs_f64()),
                format!("{:.0}ns", 1e9 * call.as_secs_f64() / rows as f64),
                format!("{:.0} MB", held * reads as f64 / 1e6),
            );
        }
        // The shortest run is the arm with no redundancy in it at all — one tile
        // an expert, so the weight is read the once it must be — and without it
        // the table ranks four amounts of re-reading against no baseline.
        assert!(
            RUNS[0] <= ROWS_A_TILE,
            "no arm here reads an expert's weight only the once it must"
        );
    }

    /// The shipped kernel with a different tile shape written into its prelude,
    /// which is the one thing the two sweeps above vary.
    ///
    /// A tile one row high is not a tile at all — [`tiles`] refuses it, so the
    /// call goes through the untiled entry — which is what makes that column of
    /// the height sweep the before of the row tile rather than a tiled kernel
    /// imitating it. One column wide *is* a tile, and is the before of the
    /// column one: it is the kernel as the row tile left it.
    fn a_tile_of(rows_a_tile: usize, cols_a_tile: usize) -> String {
        let source = source();
        let mut written = source;
        for (declared, wanted) in [
            (
                format!("constant uint ROWS_A_TILE = {ROWS_A_TILE};"),
                format!("constant uint ROWS_A_TILE = {rows_a_tile};"),
            ),
            (
                format!("constant uint COLS_A_TILE = {COLS_A_TILE};"),
                format!("constant uint COLS_A_TILE = {cols_a_tile};"),
            ),
        ] {
            written = written.replace(&declared, &wanted);
            assert!(written.contains(&wanted), "the prelude declares {wanted}");
        }
        written
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
