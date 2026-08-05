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
use crate::numerics::Numerics;

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

/// The fourth: [`ENTRY`]'s walk over the two banks a SwiGLU reads one set of
/// rows through, in one dispatch of twice the output elements.
///
/// **The two are a dispatch each because they are two tensors, and for no other
/// reason.** A bank's `gate` and `up` read the same rows of the same input
/// through the same expert list and never read each other, so what separates
/// them is which weight a simdgroup walks — and that is an index rather than a
/// submission. Fusing them halves the dispatches the banks of a decode step
/// cost and doubles the output elements each of the survivors covers, which is
/// the axis `whether_a_decode_dispatch_is_short_of_the_machine` measured this
/// kernel's rate along: 12288 elements read 621 GB/s where 24576 read 660.
const PAIRED_ENTRY: &str = "packed_matmul_pair";

/// The two entries behind [`Numerics::Production`], which are the two above with
/// the reduction carried by `simdgroup_matrix` instead of by a lane-strided walk.
///
/// **They stand in front of the two tiled entries and nowhere else.** An untiled
/// call is a decode step's single row, which has nothing for a block of 32 to
/// carry; a tiled or grouped one is a prefill's, which is where the 59.69 s the
/// two matmul rows cost at 16384 tokens is. Compiled only where the flag asked
/// for them — see [`PackedMatmul::compiled`] — so a reference run has neither in
/// its pipeline cache.
const MMA_TILED_ENTRY: &str = "mma_matmul_rows";
const MMA_GROUPED_ENTRY: &str = "mma_matmul_grouped";

/// The two entries behind [`Numerics::Production`] that stand in front of the
/// *untiled* pair, which is the decode path.
///
/// **The same walk over the same bytes, cut between several simdgroups.**
/// [`ENTRY`] gives one simdgroup to an output element and has it reduce a whole
/// weight row from end to end; here several share the row and their partials are
/// added afterwards. That is the one thing about the walk that nothing this
/// module has shipped was allowed to change — every other lever on it kept the
/// bits, and the chunks a lane holds are read and added in the order the walk
/// visits them for exactly that reason — because two lane-strided partitions of
/// one row, summed and then added, do not associate the way one does.
///
/// So the flag is what lets it be written, and it is the first entry behind the
/// flag a decode step reaches: both blocked entries refuse a call under 64 rows,
/// and a decode step's widest is nine.
///
/// **Nothing dispatches either and they are compiled only where a sweep asks**,
/// which is what this milestone concluded — see [`source_cut`].
#[cfg(test)]
const SPLIT_ENTRY: &str = "split_matmul";
#[cfg(test)]
const SPLIT_PAIRED_ENTRY: &str = "split_matmul_pair";

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

/// The square the hardware matrix instruction multiplies.
///
/// **Not a tuning knob.** `simdgroup_multiply_accumulate` takes
/// `simdgroup_matrix<float, 8, 8>` and the eight is the instruction's; every
/// other constant behind [`Numerics::Production`] is a multiple of it because a
/// block is laid out in these.
const MMA_FRAGMENT: usize = 8;

/// Rows of the call one threadgroup of the production entries carries.
///
/// **This is the arithmetic intensity the flag was opened for.** The reference
/// tile reads a packed byte and multiplies its two codes against [`ROWS_A_TILE`]
/// rows of input, which is 16 multiply-adds a weight byte. A block of this many
/// rows decodes the same byte once into threadgroup memory and drives it through
/// every row of the block, which is eight times that — and the decode, whose
/// dependency chain "What each way of decoding a packed byte costs" measured at
/// 30% of the reference kernel, is paid once for eight times the work.
///
/// **Thirty-two rather than more, because of what a block of rows costs when its
/// rows disagree.** A weight block is one expert's, so a block whose rows name
/// several runs the whole walk once per expert its rows name — see the pass loop
/// in [`MMA`]. A routed bank's runs at 16384 tokens average 384 rows, so a block
/// this tall straddles a boundary at most once per expert and is uniform
/// otherwise; a taller one buys reuse the routing cannot feed it.
const MMA_ROWS_A_BLOCK: usize = 32;

/// Output columns one threadgroup of the production entries spans.
///
/// Twice the rows, which is what makes the two staged tiles the same shape as
/// the eight simdgroups that read them: two down and four across, each holding
/// `MMA_ROWS_A_BLOCK / (2 * MMA_FRAGMENT)` by
/// `MMA_COLS_A_BLOCK / (4 * MMA_FRAGMENT)` fragments, which is two by two.
const MMA_COLS_A_BLOCK: usize = 64;

/// Simdgroups of a threadgroup laid down the rows, and across the columns.
///
/// Their product is the threadgroup, which is [`THREADS_PER_GROUP`] — so the
/// production entries dispatch the same threadgroup the reference ones do and
/// nothing about the submission changes.
const MMA_SIMDS_DOWN: usize = 2;
const MMA_SIMDS_ACROSS: usize = 4;

/// Codes of the reduction one staging step brings in, at the shipped shape.
///
/// **[`GROUP_SIZE`], and every wider step is a whole number of groups.** A weight
/// row's codes share one scale byte per group of this many, so a step of one
/// group reads exactly one scale byte per column of the block; a step of two
/// spans two, which is why the scale a thread decodes under is indexed from where
/// that thread reads rather than from where the step starts. [`Block::step`] is
/// what a sweep of it turns, and
/// [`what_a_packed_multiply_costs_at_each_step_its_staging_brings_in`] is what it
/// was set from.
const MMA_CODES_A_STEP: usize = GROUP_SIZE;

/// How the shipped block gets a fragment out of the staging.
///
/// **The two elements each lane owns, and the sweep that chose it is
/// [`what_a_packed_multiply_costs_at_each_way_it_reads_a_fragment`]:** 0.7 to
/// 0.9% under `simdgroup_load` at every shape and every length, swept both ways,
/// and a prefill's device time 0.3 to 0.4% under it with the ranges apart at
/// seven pairs of seven.
///
/// **A twelfth of what it was priced at, and the arm that priced it is what says
/// why.** The hoisting arm leaves three of every four fragments unread and reads
/// 6%, so what it prices is three of every four *reads of threadgroup memory*;
/// this changes only the instruction that carries the fourth, and the bytes it
/// carries are the same bytes.
const MMA_FRAGMENTS: Fragments = Fragments::PerLane;

/// Floats of padding between two staged rows.
///
/// **Load-bearing rather than defensive.** Threadgroup memory is 32 banks of four
/// bytes, and an 8×8 fragment is read down eight rows at the staged stride:
/// unpadded at 32 floats every one of the eight lands on the same bank and the
/// read serialises eight ways. Four floats of padding puts them on banks 0, 4,
/// 8 … 28, which are eight distinct ones — and it does so at every step this
/// block admits, since a step is a multiple of 32 and four is what shifts it off
/// the bank count.
const MMA_STAGED_PADDING: usize = 4;

/// **Every division a block's layout makes has to be exact**, and a block whose
/// staging left a remainder would leave part of a tile holding whatever the last
/// step put there — a wrong answer rather than a failure, and one no shape check
/// downstream could catch. Checked here for the shipped shape and again at
/// compile time for a swept one, out of [`Block::holds`]'s one reading.
const _: () = Block::SHIPPED.holds();

/// The shape one threadgroup of the production entries covers, and the threads
/// that cover it.
///
/// **A value rather than the constants above, because the constants are on both
/// sides of the dispatch.** The grid a call is covered by is
/// `rows.div_ceil(rows) * out_dim.div_ceil(cols)` threadgroups of `threads`,
/// [`PackedMatmul::rows_a_read`] charges a weight per block of rows, and the
/// kernel's own prelude declares all three. An entry compiled at one shape and
/// dispatched over a grid sized for another leaves output no threadgroup
/// reached — **a wrong answer rather than a slow one**, which is why no sweep
/// could reach the width or the height while these were constants and why this
/// carries them together.
///
/// [`Mma`] holds one of these beside the two entries it compiled from it, so
/// what a dispatch is covered by is read off the entry that runs rather than off
/// this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Block {
    rows: usize,
    cols: usize,
    threads: usize,
    simds_down: usize,
    simds_across: usize,
    step: usize,
    fragments: Fragments,
}

/// How a fragment gets out of the staging and into the registers the matrix
/// instruction reads it from.
///
/// **The one structural difference between this inner loop and mlx's.** Its
/// `BaseMMAFrag::load` never calls `simdgroup_load`: it reads the two elements
/// each lane owns straight out of threadgroup memory through a static
/// lane-to-coordinate map, and takes the transpose by swapping the two strides it
/// indexes with. This kernel issues a `simdgroup_load` a fragment and a
/// *transposing* one for the weight, and
/// [`what_a_prefills_blocked_matmul_is_bound_by`]'s hoisting arm prices all of
/// them together at **6%**.
///
/// A value the block carries rather than a second kernel, for the reason every
/// other axis here is one: the two answer the same bits out of the same staging
/// and differ in the instruction that moves it, so a sweep is what says which is
/// faster and a fork would be two of everything the next edit touches.
// One of the two is what [`MMA_FRAGMENTS`] names and the other is the arm the
// sweep holds it against, and which is which is that constant's to say — so the
// unnamed one is reachable from the sweep alone, and gating it on the value
// would make a table of two arms a fact about a `cfg`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fragments {
    /// `simdgroup_load`, and a transposing one for the weight.
    Loaded,
    /// The two elements a lane owns, read where they lie.
    PerLane,
}

impl Block {
    /// The shape the engine ships, which is what every figure in the README was
    /// taken at.
    const SHIPPED: Self = Self {
        rows: MMA_ROWS_A_BLOCK,
        cols: MMA_COLS_A_BLOCK,
        threads: THREADS_PER_GROUP,
        simds_down: MMA_SIMDS_DOWN,
        simds_across: MMA_SIMDS_ACROSS,
        step: MMA_CODES_A_STEP,
        fragments: MMA_FRAGMENTS,
    };

    /// Rows an expert needs, on average, before a grouped call is dispatched
    /// through the production entries.
    ///
    /// **A block is correct however its rows are laid out and fast only where
    /// they agree**, so this is the line between the two. The kernel runs its
    /// whole walk once per distinct expert a block's rows name, which is one
    /// pass where a run covers the block and as many as the block is tall where
    /// the routing is spread thin — and a routed bank at 97 tokens averages 2.3
    /// rows an expert, which would be a dozen passes a block.
    ///
    /// A block's own height, so that the average run covers it: six rows a token
    /// against 256 experts puts the line at about 1366 tokens, under which a
    /// grouped call stays on the reference tile whatever the flag says.
    /// [`RUNS_A_GROUPING`] is the same question asked of the sort rather than of
    /// the block, and is a much lower bar because a sort that saves nothing
    /// costs one dispatch where a block that straddles costs a whole extra walk.
    const fn runs_an_expert(&self) -> usize {
        self.rows
    }

    /// Fragments one simdgroup holds down the rows and across the columns, which
    /// is what is left of the block once its simdgroups have taken their share.
    const fn fragments_down(&self) -> usize {
        self.rows / (self.simds_down * MMA_FRAGMENT)
    }

    const fn fragments_across(&self) -> usize {
        self.cols / (self.simds_across * MMA_FRAGMENT)
    }

    /// Floats of the input one thread stages a step, and the threads a staged
    /// row takes.
    const fn floats_a_thread(&self) -> usize {
        self.rows * self.step / self.threads
    }

    const fn threads_a_staged_row(&self) -> usize {
        self.step / self.floats_a_thread()
    }

    /// Packed bytes of the weight one thread decodes a step, and the threads a
    /// weight row takes.
    const fn bytes_a_thread(&self) -> usize {
        self.cols * (self.step / CODES_PER_BYTE) / self.threads
    }

    const fn threads_a_weight_row(&self) -> usize {
        (self.step / CODES_PER_BYTE) / self.bytes_a_thread()
    }

    /// Floats between two staged rows, which is the step and
    /// [`MMA_STAGED_PADDING`].
    const fn staged_stride(&self) -> usize {
        self.step + MMA_STAGED_PADDING
    }

    /// Bytes of threadgroup memory the two staged tiles declare, which is what
    /// decides how many threadgroups of this block a core holds — and so what a
    /// wider step is bought at.
    ///
    /// The sweep's own column, which is why it is behind `cfg(test)`: nothing a
    /// dispatch does reads it, because what a threadgroup declares is declared
    /// in the kernel's own source out of [`Block::declares`].
    #[cfg(test)]
    const fn staged_bytes(&self) -> usize {
        (self.rows + self.cols) * self.staged_stride() * size_of::<f32>()
    }

    /// Floats between two rows of the block's answer, padded against 32 banks
    /// for the reason [`MMA_STAGED_PADDING`] is.
    const fn answer_stride(&self) -> usize {
        self.cols + MMA_STAGED_PADDING
    }

    /// **Every division has to be exact**, or part of a tile holds whatever the
    /// last step put there.
    ///
    /// `const` so that [`Block::SHIPPED`] is checked where it is written and a
    /// swept shape is checked where it is compiled, out of one reading of the
    /// arithmetic rather than two.
    const fn holds(&self) {
        assert!(self.simds_down * self.simds_across * NARROWEST_SIMD == self.threads);
        assert!(self.fragments_down() * self.simds_down * MMA_FRAGMENT == self.rows);
        assert!(self.fragments_across() * self.simds_across * MMA_FRAGMENT == self.cols);
        assert!(self.floats_a_thread() * self.threads == self.rows * self.step);
        assert!(self.threads_a_staged_row() * self.floats_a_thread() == self.step);
        assert!(self.bytes_a_thread() * self.threads == self.cols * (self.step / CODES_PER_BYTE));
        assert!(self.threads_a_weight_row() * self.bytes_a_thread() == self.step / CODES_PER_BYTE);
        // A step is whole groups of the format, so that a step boundary never
        // falls inside the codes one scale byte covers, and whole fragments, so
        // that the `k` loop reaches every code the staging brought in.
        assert!(self.step % GROUP_SIZE == 0);
        assert!(self.step % MMA_FRAGMENT == 0);
        // **The bytes one thread decodes lie inside one group**, which is what
        // lets the scale be read once a step rather than once a byte. A step
        // wider than a group spans several, so what the read is indexed by is
        // where the thread reads and not where the step starts — and a thread
        // whose chunk straddled a boundary would decode half of it under the
        // wrong exponent.
        assert!(BYTES_PER_GROUP % self.bytes_a_thread() == 0);
        // The block's expert list is read by one thread a row, so a threadgroup
        // narrower than the block is tall would leave part of it unread.
        assert!(self.threads >= self.rows);
        // The answer is written over the two staged tiles, so a block whose
        // answer were the larger of the two would write past them.
        assert!(self.rows * self.answer_stride() <= (self.rows + self.cols) * self.staged_stride());
    }

    /// The prelude the two production entries are written from, which is every
    /// number above spelled once.
    fn declares(&self) -> String {
        format!(
            "
constant uint MMA_FRAGMENT = {MMA_FRAGMENT};
constant uint MMA_ROWS_A_BLOCK = {};
constant uint MMA_COLS_A_BLOCK = {};
constant uint MMA_SIMDS_ACROSS = {};
constant uint MMA_CODES_A_STEP = {};
constant uint MMA_STAGED_STRIDE = {};
constant uint MMA_ANSWER_STRIDE = {};
constant uint MMA_FRAGMENTS_DOWN = {};
constant uint MMA_FRAGMENTS_ACROSS = {};
constant uint MMA_FLOATS_A_THREAD = {};
constant uint MMA_THREADS_A_STAGED_ROW = {};
constant uint MMA_BYTES_A_THREAD = {};
constant uint MMA_THREADS_A_WEIGHT_ROW = {};
constant uint THREADS_PER_GROUP = {};
",
            self.rows,
            self.cols,
            self.simds_across,
            self.step,
            self.staged_stride(),
            self.answer_stride(),
            self.fragments_down(),
            self.fragments_across(),
            self.floats_a_thread(),
            self.threads_a_staged_row(),
            self.bytes_a_thread(),
            self.threads_a_weight_row(),
            self.threads,
        ) + self.reads_a_fragment()
    }

    /// The two ways [`MMA`] reads a fragment out of the staging, written the way
    /// this block asks for.
    ///
    /// **One call site apiece in the kernel whichever this is**, which is what
    /// makes the axis a value rather than a second copy of the inner loop.
    ///
    /// Templated on the matrix and on what the staging holds because the 16-bit
    /// arm reads narrow tiles through these same two calls — see
    /// [`through_sixteen_bit_operands`], which swaps the fragment declarations
    /// and nothing else.
    fn reads_a_fragment(&self) -> &'static str {
        match self.fragments {
            Fragments::Loaded => MMA_LOADED_FRAGMENTS,
            Fragments::PerLane => MMA_PER_LANE_FRAGMENTS,
        }
    }
}

/// A fragment through `simdgroup_load`, which is what this kernel has always
/// issued.
const MMA_LOADED_FRAGMENTS: &str = r#"
template <typename M, typename T>
inline void mma_read(thread M &into, threadgroup const T *from, uint lane) {
    simdgroup_load(into, from, MMA_STAGED_STRIDE);
}

template <typename M, typename T>
inline void mma_read_transposed(thread M &into, threadgroup const T *from, uint lane) {
    simdgroup_load(into, from, MMA_STAGED_STRIDE, ulong2(0, 0), true);
}
"#;

/// A fragment through the two elements each lane owns, read where they lie.
///
/// **The map is verified against this part rather than inherited.** These three
/// lines are mlx's `BaseMMAFrag<float, 8, 8>::get_coord`, and what a header says
/// about a lane layout is a claim about hardware nobody here has a document for —
/// so `the_lane_map_a_fragment_is_read_through_is_this_parts_own` puts the same
/// tile through both forms and holds them equal lane for lane and element for
/// element. A map that were wrong would not fail: it would read somebody else's
/// two elements and answer a plausible number.
///
/// The transpose is the same two elements with the strides swapped, which is
/// what a transposing `simdgroup_load` costs an instruction to express and this
/// costs an index.
const MMA_PER_LANE_FRAGMENTS: &str = r#"
inline ushort2 mma_lane_holds(uint lane) {
    const ushort quad = lane / 4;
    return ushort2((quad & 2) * 2 + (lane % 2) * 2, (quad & 4) + ((lane / 2) % 4));
}

template <typename M, typename T>
inline void mma_read(thread M &into, threadgroup const T *from, uint lane) {
    const ushort2 at = mma_lane_holds(lane);
    thread vec<T, 2> &held = (thread vec<T, 2> &)(into.thread_elements());
    held[0] = from[at.y * MMA_STAGED_STRIDE + at.x];
    held[1] = from[at.y * MMA_STAGED_STRIDE + at.x + 1];
}

template <typename M, typename T>
inline void mma_read_transposed(thread M &into, threadgroup const T *from, uint lane) {
    const ushort2 at = mma_lane_holds(lane);
    thread vec<T, 2> &held = (thread vec<T, 2> &)(into.thread_elements());
    held[0] = from[at.x * MMA_STAGED_STRIDE + at.y];
    held[1] = from[(at.x + 1) * MMA_STAGED_STRIDE + at.y];
}
"#;

/// The narrowest simdgroup Metal reports, which is what the layout above
/// divides the threadgroup into simdgroups by.
///
/// **A device that reported a wider one would put fewer simdgroups in a
/// threadgroup than the block was laid out for**, which is why
/// `a_block_is_the_simdgroups_this_device_gives_a_threadgroup` reads it back off
/// the compiled pipeline rather than trusting this.
const NARROWEST_SIMD: usize = 32;

/// Blocks a call has to bring rows enough for before it is dispatched through
/// one at all.
///
/// **Two, and the sweep either side of it is in [`PackedMatmul::blocks`].** A
/// block computes its full height whatever the call brought, and a call of one
/// or two blocks does not put threadgroups enough on this part to fill it — so
/// the block is behind the reference tile at 40 and 48 rows and ahead from 64,
/// which is where this is set.
const MMA_BLOCKS_A_CALL: usize = 2;

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

/// Threads one threadgroup of the untiled entry holds, which is the only thing
/// about that dispatch a caller decides.
///
/// **A threadgroup of [`ENTRY`] is that many independent simdgroups and nothing
/// else** — no threadgroup memory, no barrier, no shared load, no fragment
/// anything shares — so this is a scheduling knob rather than a kernel one, and
/// every arm of a sweep over it answers the same bits. What it decides is how
/// evenly a call's simdgroups land on 80 cores: at [`THREADS_PER_GROUP`] a
/// 1024-element dispatch is 128 threadgroups of eight, so 48 cores take two and
/// 32 take one and the whole dispatch waits for the slower half.
///
/// **Sixty-four, and the sweep that chose it is in the model.** Five alternating
/// pairs of a decode step against the shipped 256, every pair the same way and
/// the ranges apart at every arm:
///
/// ```text
/// a threadgroup      a decode step   the device
///  32 threads             -2.1%         -2.2%
///  64 threads             -2.3%         -2.3%
/// 128 threads             -1.8%         -1.9%
/// 256 threads, before         —             —
/// 512 threads             +3.5%         +3.8%
/// ```
///
/// **The middle of a plateau rather than its edge**, which is the discipline
/// [`RESIDENCY`] is sized under. Seven pairs each of 64 against 32 and 64 against
/// 128 read +0.1% and +0.2% with the ranges across and no claim, so the three are
/// one arm and 64 is the middle of it; 256 and 512 are behind all three.
///
/// [`what_a_packed_multiply_costs_at_each_threadgroup_an_untiled_call_takes`] is
/// the same question of one dispatch at a time. It reproduces the ranking and it
/// is what holds every arm to the shipped kernel's bits; what it is not is a
/// second reading of the size, because one arm of each direction comes back on a
/// clock the others were not measured on.
///
/// **Its own constant because [`THREADS_PER_GROUP`] is not free to move.** That
/// one is the tiled entries' as well as this one's, and a block's is derived
/// against it — `Block::holds` asserts the simdgroups a block is laid out in
/// cover it — so a sweep that turned it turned the prefill's shape too, and the
/// two want opposite widths.
const THREADS_AN_ELEMENTS_GROUP: usize = 64;

/// Simdgroups a split call may cut one element's walk between, which bounds the
/// threadgroup array their partials land in.
///
/// **A bound and not the number.** How many a call actually takes is
/// [`PackedMatmul::cut_at`]'s, read off the threadgroup the dispatch was given;
/// this is what the kernel has to declare an array for before it knows. Eight
/// floats is 32 bytes of a threadgroup's 32 KiB — which is what makes the count
/// free to be a runtime value, and, as it turned out, not what the array costs.
#[cfg(test)]
const MOST_SPLITS: usize = 8;

/// Floats of threadgroup memory a tile declares and never reads for anything,
/// which is what puts the two tiled entries on the fast side of their occupancy
/// turn.
///
/// **A tile of this kernel is a simdgroup and wants no threadgroup memory at
/// all**, so it ran at whatever residency this part gives a kernel that asks for
/// nothing — which turns out to be too many threadgroups a core rather than too
/// few. [`how_many_threadgroups_of_a_prefills_packed_matmul_a_core_holds`]
/// sweeps it: a 4096-token prompt's `q_proj` is 5.90 ms declaring nothing and
/// 5.19 declaring this, and its routed bank 19.55 ms against 16.37.
///
/// **Dead memory is the whole mechanism and there is nothing here it could
/// hold.** A tile's working set is registers — [`ROWS_A_TILE`] by
/// [`COLS_A_TILE`] running sums, four weight pointers and four scale pointers —
/// and none of it is shared across the simdgroups of a threadgroup, which is
/// what makes a threadgroup of this kernel eight independent tiles rather than
/// a unit. So there is no cooperative load to move here and no tile to stage;
/// what the declaration buys is bought by being declared.
///
/// **Sized to the middle of its plateau rather than to its edge.** The row
/// improves at every step down to three threadgroups a core and gives 17% back
/// at two: 20 KiB is 5.20 ms, 24 is 5.19, 26 is 5.19 and 28 is 6.07. Three of
/// them is every declaration in (20, 26.67] KiB and this is 24.
///
/// **It reaches no dispatch a decode step makes.** Only the tiled entries
/// declare it, and [`tiles`] is false for every shape a decode step has — a
/// single-row projection, a two-row shared bank naming two experts, a six-row
/// routed bank naming six.
///
/// **`fused_attention` declares memory the same way and needs no `volatile` for
/// it, and the difference is what the compiler can see.** There the fill is a
/// strided loop over one array and the read is a different loop over another,
/// with a runtime bound between them; here it is one store to a thread's own
/// slot and a load of the same slot on the next line, which is the shape a
/// forwarding pass is looking for. Neither kernel argues the point: each has a
/// case that reads the bytes its pipeline reports, and a compiler that started
/// or stopped folding either would fail it rather than quietly give the memory
/// back.
const RESIDENCY: usize = 6144;

/// **Every thread of a threadgroup fills and reads its own entry**, which is
/// what keeps the residency free of a barrier and free of a race.
const _: () = assert!(RESIDENCY >= THREADS_PER_GROUP);

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

/// Chunks of its weight row one lane of [`ENTRY`] reads before it adds any of
/// them, which is how much of that row it has outstanding at a time.
///
/// **This is the term a decode step is bound by that is not its own bytes.** A
/// walk's trip count is a runtime value — a row is `in_dim` wide and the kernel
/// is compiled once for every width the checkpoint has — so nothing unrolls it,
/// and a lane holding one chunk issues four bytes, waits a memory latency, adds
/// them and issues four more.
/// [`what_a_decode_dispatchs_fixed_cost_is_made_of`] prices that wait by sweeping
/// the reduction at a fixed width: a dispatch costs **4.4 µs plus 0.41 µs a
/// step** on top of its bytes at the bus, and 0.41 µs is a memory round trip.
///
/// **Nothing about the summation order moves, which is what makes this a constant
/// rather than a second numerics.** Every other lever this module has taken on
/// the innermost walk had to go behind [`Numerics`] for changing that order; the
/// chunks here are read in the order the walk visits them and added in that
/// order, so every arm of the sweep answers the shipped kernel's bits — see
/// `weight_dot`, and
/// [`a_walk_that_holds_several_chunks_answers_the_bits_one_holding_one_answers`].
///
/// **Four, and the sweep either side of it is measured in the model rather than
/// only in a fixture.** Seven alternating pairs of a decode step, every pair the
/// same way and the ranges apart: two held is 0.9% behind four and eight is 1.8%
/// behind it.
/// [`what_a_packed_multiply_costs_at_each_chunk_a_lane_holds`] is the same
/// question of one dispatch at a time, and reads the turn at the same place —
/// 1.43× at `k_proj`, 1.19× at a shared bank, 1.08× at a routed one and 1.02× at
/// a call wide enough to hide its own loads. **That last column is the finding
/// rather than the first**: what a held chunk buys is buying a narrow dispatch
/// the latency hiding a wide one already has, which is the shape of finding
/// [`BYTES_PER_LANE`] is from the other side.
const CHUNKS_IN_FLIGHT: usize = 4;

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
    /// `kernel`'s walk over both banks of a [`PackedPair`] at once, which is the
    /// entry every untiled call against a SwiGLU's two halves goes through.
    paired: Kernel,
    tiled: Kernel,
    grouped: Kernel,
    /// The two entries that stand in front of `tiled` and `grouped` where the
    /// flag asked for them, and **nothing at all under the reference** — which
    /// is what makes "nothing changes for a caller who does not ask" a fact
    /// about the pipeline cache rather than about a branch nobody takes.
    mma: Option<Mma>,
    /// The two that would have stood in front of `kernel` and `paired` on the
    /// same terms, which nothing dispatches — see [`SPLIT_ENTRY`].
    #[cfg(test)]
    split: Option<Split>,
    /// Which arithmetic the innermost accumulation is allowed to use, which is
    /// [`Numerics::Reference`] unless a command line asked otherwise.
    numerics: Numerics,
    /// The rows a tile of `tiled` holds, which is [`ROWS_A_TILE`] for the
    /// shipped source and whatever the sweep wrote into a mutant's prelude.
    rows_a_tile: usize,
    /// The columns it spans, the same way — see [`COLS_A_TILE`].
    cols_a_tile: usize,
    /// The threads a threadgroup of `kernel` holds, which is
    /// [`THREADS_AN_ELEMENTS_GROUP`] for every caller but the sweep that chose
    /// it.
    ///
    /// **Carried rather than read off the module**, for the reason [`Block`]
    /// gives about the block's own shape: a width that is a constant is a width
    /// no sweep can turn, and a value the entry carries is one arm rather than a
    /// milestone.
    threads_an_elements_group: usize,
    /// The cuts every untiled call takes whatever its element count, which only
    /// the sweep that separates the entry from the cut sets — see
    /// [`PackedMatmul::cut_at`].
    #[cfg(test)]
    cuts: Option<usize>,
}

/// The two untiled entries behind [`Numerics::Production`], which are
/// [`PackedMatmul::kernel`] and [`PackedMatmul::paired`] with an element's walk
/// cut between the simdgroups of a threadgroup.
///
/// **A pair rather than one**, because a decode step dispatches both: a
/// projection is one bank and a SwiGLU's gate and up are the pair D2 fused, and
/// the second is 55% of what a step reads. Cutting the walk on one of them would
/// leave the other where it was.
#[cfg(test)]
#[derive(Debug)]
struct Split {
    kernel: Kernel,
    paired: Kernel,
}

impl PackedMatmul {
    /// The kernel under the numerics every caller gets who does not ask for the
    /// other, which is the reference — see [`Numerics`].
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// The same, under numerics the caller chose.
    ///
    /// **One place holds the default**, which is [`PackedMatmul::tiling`]
    /// below, so that "reference unless asked" is a fact about one line rather
    /// than a convention twenty-eight call sites keep.
    pub fn under(device: &Device, numerics: Numerics) -> Result<Self, MetalError> {
        Self::blocked(device, numerics, Block::SHIPPED)
    }

    /// The same, where the production entries are cut to a block of another
    /// shape — the one way a sweep of the block's height, its width or its
    /// threadgroup can be compiled and dispatched consistently.
    ///
    /// **The source and the shape are taken together and never apart.** The grid
    /// a call is covered by is sized from the block and the kernel takes its own
    /// layout from its prelude, so an arm that wrote one and dispatched the
    /// other would leave output no threadgroup reached — a wrong answer rather
    /// than a slow one, and the reason the height and the width were unsweepable
    /// while these were module constants.
    pub(crate) fn blocked(
        device: &Device,
        numerics: Numerics,
        block: Block,
    ) -> Result<Self, MetalError> {
        Self::compiled(
            device,
            &source_blocked(numerics, block),
            ROWS_A_TILE,
            COLS_A_TILE,
            numerics,
            block,
        )
    }

    /// [`PackedMatmul::blocked`] out of a source string of the caller's own,
    /// which is how a limiter arm puts a deliberately wrong block through the
    /// same plumbing as the right one and measures the difference.
    ///
    /// **The block is the caller's too**, because an arm that rewrites the
    /// prelude and an arm that rewrites the body are the same kind of mutation
    /// and only one of them can be spotted by reading the source string here.
    #[cfg(test)]
    pub(crate) fn blocked_from_source(
        device: &Device,
        source: &str,
        block: Block,
    ) -> Result<Self, MetalError> {
        Self::compiled(
            device,
            source,
            ROWS_A_TILE,
            COLS_A_TILE,
            Numerics::Production,
            block,
        )
    }

    /// [`PackedMatmul::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    ///
    /// **Under the reference, which is what a mutant is for.** Every arm this
    /// module compiles is held against the shipped kernel's own bits, and there
    /// is nothing to hold a mutant against on the other side of the flag.
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
        Self::compiled(
            device,
            source,
            rows_a_tile,
            cols_a_tile,
            Numerics::default(),
            Block::SHIPPED,
        )
    }

    /// The whole of it, which is [`PackedMatmul::tiling`] and the two things
    /// that are not properties of the source string.
    fn compiled(
        device: &Device,
        source: &str,
        rows_a_tile: usize,
        cols_a_tile: usize,
        numerics: Numerics,
        block: Block,
    ) -> Result<Self, MetalError> {
        let mut declared = vec![
            format!("constant uint ROWS_A_TILE = {rows_a_tile};"),
            format!("constant uint COLS_A_TILE = {cols_a_tile};"),
        ];
        // The block reaches the grid as well as the kernel, so what says the two
        // agree is that the source declares the shape this side will dispatch
        // it at — the same check the tile gets, for the same reason.
        if numerics.is_production() {
            declared.extend([
                format!("constant uint MMA_ROWS_A_BLOCK = {};", block.rows),
                format!("constant uint MMA_COLS_A_BLOCK = {};", block.cols),
                format!("constant uint THREADS_PER_GROUP = {};", block.threads),
            ]);
        }
        for declares in declared {
            assert!(
                source.contains(&declares),
                "a source dispatched at {rows_a_tile}x{cols_a_tile} a tile and {:?} a block does \
                 not declare `{declares}`",
                block
            );
        }
        // Compiled only where the flag asked, which is what puts a reference run
        // through the same three pipelines it went through before this flag
        // existed — the sources are byte for byte the same string, and the two
        // entries below are not in it to compile.
        let mma = match numerics.is_production() {
            false => None,
            true => Some(Mma::of(
                device.compile(source, MMA_TILED_ENTRY)?,
                device.compile(source, MMA_GROUPED_ENTRY)?,
                block,
            )?),
        };
        // **The two tiled entries are compiled before the untiled one, and the
        // order is worth 2.4% of a prefill.** Nothing about the three kernels
        // says it should be: they are three libraries out of one source string,
        // dispatched at different shapes with nothing shared between them. But
        // with the untiled entry created first — which is how this read since it
        // was the only entry — a 2048-token prefill cost 2.4% of its device time
        // and 8192 cost 2.2%, five and three alternating pairs, every pair the
        // same way and the ranges apart, on the two tiled entries that are 87% of
        // it. Swapping the two lines gives all of it back and takes 0.5% more.
        //
        // **What it is not is the source, and that was measured before this was
        // written.** Compiling the untiled entry from a string of its own, so
        // that the tiled two never saw the walk it runs, left the 2.4% exactly
        // where it was. What moved it was which of the three was created first.
        //
        // **The paired entry is created after them too**, which is where the
        // untiled entry it is a second emission of is created. Nothing was
        // measured about the paired one's own position — what was measured is
        // that a prefill did not move when it was added, which is the row this
        // order exists to protect.
        let tiled = device.compile(source, TILED_ENTRY)?;
        let grouped = device.compile(source, GROUPED_ENTRY)?;
        // **After the tiled two and before the untiled two**, which is where the
        // entries they stand in front of are created. What the order above
        // protects is a prefill, and these run no prefill dispatch at all.
        // Only where the source carries them, which is the sweep's source and
        // no other: nothing dispatches a cut walk, so a run that is not pricing
        // one has neither entry in its pipeline cache.
        #[cfg(test)]
        let split = match source.contains(SPLIT_ENTRY) {
            false => None,
            true => Some(Split {
                kernel: device.compile(source, SPLIT_ENTRY)?,
                paired: device.compile(source, SPLIT_PAIRED_ENTRY)?,
            }),
        };
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
            paired: device.compile(source, PAIRED_ENTRY)?,
            tiled,
            grouped,
            mma,
            #[cfg(test)]
            split,
            numerics,
            rows_a_tile,
            cols_a_tile,
            threads_an_elements_group: THREADS_AN_ELEMENTS_GROUP,
            #[cfg(test)]
            cuts: None,
        })
    }

    /// The same, dispatching an untiled call at a threadgroup of the caller's
    /// own — which is what makes [`THREADS_AN_ELEMENTS_GROUP`] swept rather than
    /// asserted.
    ///
    /// **Nothing about the kernel moves, which is why this is a field and not a
    /// source string.** A threadgroup of [`ENTRY`] is that many independent
    /// simdgroups and shares nothing between them, so the width reaches the grid
    /// and stops there — see [`PackedMatmul::entry`].
    #[cfg(test)]
    pub(crate) fn dispatching_at(device: &Device, threads: usize) -> Result<Self, MetalError> {
        Ok(Self {
            threads_an_elements_group: threads,
            ..Self::new(device)?
        })
    }

    /// The split entries, cutting every untiled call exactly `splits` ways
    /// whatever its element count — **including one**.
    ///
    /// **What this separates is the entry from the cut**, which is the whole of
    /// what the sweep found. At one cut the split entry is the plain one's own
    /// walk — `the_cuts_of_a_split_walk_read_each_byte_of_the_row_exactly_once`
    /// holds the partition and the answer is the same bits — so what a cut of
    /// one costs over the plain entry is the threadgroup array, the barrier and
    /// the combine, with no partition at all under them. It is 21 to 42%, and
    /// the arm with the barrier taken out is the same.
    #[cfg(test)]
    pub(crate) fn cut_at(device: &Device, splits: usize) -> Result<Self, MetalError> {
        Self::cut_from_source(device, &source_cut(Block::SHIPPED), splits)
    }

    /// The same out of a source string of the caller's own, which is how an arm
    /// prices one term of the cut against the rest of it.
    #[cfg(test)]
    pub(crate) fn cut_from_source(
        device: &Device,
        source: &str,
        splits: usize,
    ) -> Result<Self, MetalError> {
        assert!(
            (1..=MOST_SPLITS).contains(&splits),
            "{splits} cuts is outside the array their partials land in"
        );
        Ok(Self {
            cuts: Some(splits),
            ..Self::blocked_from_source(device, source, Block::SHIPPED)?
        })
    }

    /// The shortest call the entries behind [`Numerics::Production`] are given,
    /// in rows.
    ///
    /// **Public because a differential run is worthless without it.** A call
    /// under this height runs the same kernel under both words, so a corpus of
    /// prompts shorter than this would report perfect agreement between a thing
    /// and itself — and would keep reporting it after the arithmetic behind the
    /// flag had changed. Whoever assembles a corpus has to be able to ask.
    pub const SHORTEST_BLOCKED_CALL: usize = MMA_BLOCKS_A_CALL * MMA_ROWS_A_BLOCK;

    /// Every entry this module puts behind [`Numerics::Production`].
    ///
    /// **A height is not a reach and this milestone is why.** While the flag
    /// selected the two blocked entries alone, a corpus that cleared
    /// [`PackedMatmul::SHORTEST_BLOCKED_CALL`] reached everything behind it and
    /// a length was the whole of the question. An entry a *decode step*
    /// dispatches is reached by a call of one row, so no prompt length gates it
    /// and no floor can say whether a differential run ever ran it — which is
    /// the same "a thing agreeing with itself" the floor exists to refuse,
    /// arriving on the axis the floor does not watch.
    ///
    /// So a differential run holds what it *dispatched* against this list rather
    /// than holding its prompts against a number. See
    /// [`Encoded::symbol`](crate::trace::Encoded::symbol), which is what lets a
    /// trace name the entry rather than the row it is charged to.
    pub const BEHIND_THE_FLAG: &'static [&'static str] = &[MMA_TILED_ENTRY, MMA_GROUPED_ENTRY];

    /// Which arithmetic this one accumulates with.
    ///
    /// Read back rather than only obeyed, because the first thing a report about
    /// a wrong token has to say is which of the two produced it.
    pub fn numerics(&self) -> Numerics {
        self.numerics
    }

    /// The entry a call goes through and the grid that covers it.
    ///
    /// **An untiled call stays on the untiled kernel rather than on a tile of
    /// one.** The two compute the same thing at that height, and the ordinary
    /// one is what every decode step this project has measured was dispatching
    /// — so the shape with nothing to win stays on the code that won what is
    /// already there, rather than carrying a tile's worth of registers to use
    /// one row of it.
    ///
    /// **The grid rather than a simdgroup count, because the two entries behind
    /// the flag count in blocks where the three in front of it count in
    /// simdgroups.** Neither changes anything else about the submission; what
    /// differs is only how many threads cover a call and how many of them share a
    /// threadgroup, which is the one thing a caller cannot work out from the
    /// kernel alone. A tile takes [`THREADS_PER_GROUP`], a block takes its own,
    /// and an untiled call takes [`THREADS_AN_ELEMENTS_GROUP`] — or, cut, one
    /// element's cuts.
    fn entry(
        &self,
        layout: &Layout<'_>,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
        experts: usize,
    ) -> (&Kernel, Grid) {
        // Off the entry that will run rather than off this module, which is the
        // whole of what makes the block's shape sweepable — see [`Block`].
        let blocks = |mma: &Mma| {
            let over = rows.div_ceil(mma.block.rows) * out_dim.div_ceil(mma.block.cols);
            Grid::new(over * mma.block.threads, mma.block.threads)
        };
        let tiles = rows.div_ceil(self.rows_a_tile) * out_dim.div_ceil(self.cols_a_tile);
        match (self.blocks(layout, rows, experts), layout) {
            (_, Layout::Each) => self.each(rows * out_dim, in_dim, false),
            (None, Layout::Tiled) => (
                &self.tiled,
                simdgroups(&self.tiled, tiles, THREADS_PER_GROUP),
            ),
            (None, Layout::Grouped { .. }) => (
                &self.grouped,
                simdgroups(&self.grouped, tiles, THREADS_PER_GROUP),
            ),
            (Some(mma), Layout::Tiled) => (&mma.tiled, blocks(mma)),
            (Some(mma), Layout::Grouped { .. }) => (&mma.grouped, blocks(mma)),
        }
    }

    /// The paired entry and the grid that covers both halves of one call, which
    /// is twice the elements of [`PackedMatmul::entry`]'s [`Layout::Each`] arm
    /// over the same threadgroup.
    ///
    /// **Only the untiled layout has one.** A tiled or grouped call is a
    /// prefill's, where a dispatch already has output elements enough to fill
    /// the machine and what a tile carries is registers rather than a launch —
    /// so the pair there is two dispatches and stays two. See
    /// [`PackedPair::listed`].
    fn paired(&self, rows: usize, out_dim: usize, in_dim: usize) -> (&Kernel, Grid) {
        self.each(2 * rows * out_dim, in_dim, true)
    }

    /// An untiled call's entry and grid.
    ///
    /// **One reading for the bank and for the pair**, because the question is the
    /// same question about the same call: a pair is the untiled entry over twice
    /// the elements, and how far a walk is worth cutting is decided by how many
    /// of them there are. `paired` is which of the two a call is, and it selects
    /// on both sides of the flag — a call that stays uncut has to stay on the
    /// entry taking the bindings it was given, and the two entries against one
    /// bank and against two do not take the same ones.
    fn each(&self, elements: usize, in_dim: usize, paired: bool) -> (&Kernel, Grid) {
        let whole = match paired {
            false => &self.kernel,
            true => &self.paired,
        };
        let uncut = |kernel: &_| simdgroups(kernel, elements, self.threads_an_elements_group);
        let _ = in_dim;
        #[cfg(test)]
        if let (Some(split), Some(splits)) = (self.split.as_ref(), self.cuts) {
            let cut = match paired {
                false => &split.kernel,
                true => &split.paired,
            };
            // Off the entry that will run rather than off this module, for the
            // reason the block's shape is: the grid is sized from the width, and
            // a width this side assumed would leave the kernel dividing by
            // another.
            let width = cut.simd_width();
            return (cut, Grid::new(elements * splits * width, splits * width));
        }
        (whole, uncut(whole))
    }

    /// The production entries, where this call's shape is one they should run.
    ///
    /// **The production entries are correct for every shape and fast for some**,
    /// so this is where the difference is drawn rather than in the kernel. A
    /// block's weight is one expert's, so a block whose rows name several runs
    /// the walk once per expert — which is nothing where the runs cover a block
    /// and a dozen walks where the routing is spread thin.
    ///
    /// A tiled call's runs are as long as the call: a projection's rows all name
    /// expert zero and a shared bank's are its input laid end to end once per
    /// expert. A grouped call's are the routing's, and [`Block::runs_an_expert`] is
    /// the line — under it the call stays on the reference tile whatever the
    /// flag says, which is a rate and never an answer.
    ///
    /// **[`Layout::Each`] never reaches it**, and that is the decode path stated
    /// where it is decided: a single-row projection, a two-row shared bank and a
    /// six-row routed bank are what a decode step dispatches, and a block of 32
    /// rows has nothing to carry for any of them.
    ///
    /// **And a call has to bring [`MMA_BLOCKS_A_CALL`] blocks' worth of rows
    /// before it is given a block, which a measurement rather than an argument
    /// put here.** A block computes [`MMA_ROWS_A_BLOCK`] rows whether the call
    /// has that many or not — the rows past its own stage zeros and are walked
    /// with the rest — and a call of a few blocks does not put threadgroups
    /// enough on the machine to fill it either way. A prefill at each length,
    /// one sitting apiece, on the device's own clock:
    ///
    /// ```text
    /// rows   reference   production
    ///   40    264.6ms      278.8ms
    ///   48    315.0ms      320.7ms
    ///   64    415.1ms      390.1ms
    ///   80    518.2ms      487.3ms
    ///   97    502.8ms      456.8ms
    /// ```
    ///
    /// So the block is behind at 40 and 48 rows and ahead from 64, and the line
    /// is drawn where the measurement turns. **Padding alone does not explain
    /// it** — 48 rows waste a third of two blocks and 97 waste a third of four,
    /// and only the second wins — so what the shorter calls are also short of is
    /// threadgroups: a 48-row call of `q_proj` is 128 of them against 240 slots
    /// on this part's 80 cores, where 97 rows are 256 and fill it.
    ///
    /// **What a floor here is worth is a speculative round, and that was
    /// measured before the line existed.** A verify block is the depth plus one
    /// rows through every projection, so at a depth of three or four it clears
    /// [`tiles`]'s four-row bar and lands on a block eight times too tall:
    /// `k = 3` read **37.33 ms a token against the reference's 17.08** and
    /// `k = 4` 36.30 against 19.98, where `k` of 0, 1 and 2 were untouched
    /// because their blocks are under four rows and never left [`Layout::Each`].
    /// **Speculation is a decode-time path and this flag was never meant to
    /// reach it.**
    ///
    /// The floor binds on [`Layout::Tiled`] alone in practice: a grouped call
    /// already needs a block's worth of runs an expert, which for the 256-expert
    /// routed bank is 8192 rows and past this many times over.
    fn blocks(&self, layout: &Layout<'_>, rows: usize, experts: usize) -> Option<&Mma> {
        // Off the block these entries were compiled to rather than off the
        // module, for the reason [`Block`] gives: both thresholds are a count of
        // its rows, and a shape swept to another height would otherwise be gated
        // at the shipped one's.
        let mma = self.mma.as_ref()?;
        let worth = rows >= MMA_BLOCKS_A_CALL * mma.block.rows
            && match layout {
                Layout::Each => false,
                Layout::Tiled => true,
                Layout::Grouped { .. } => {
                    rows >= experts.saturating_mul(mma.block.runs_an_expert())
                }
            };
        worth.then_some(mma)
    }

    /// The rows one dispatch of this shape reads a weight once for, which is the
    /// block where the production entries run it and the tile where they do not.
    ///
    /// **One predicate decides the entry and the accounting**, so that a
    /// bandwidth column cannot describe a dispatch other than the one that ran —
    /// see [`PackedBank::moves`], which charges a weight per tile of rows that
    /// shares it.
    fn rows_a_read(&self, layout: &Layout<'_>, rows: usize, experts: usize) -> usize {
        match self.blocks(layout, rows, experts) {
            Some(mma) => mma.block.rows,
            None => self.rows_a_tile,
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

/// The two entries behind [`Numerics::Production`], held together because a
/// dispatch reaches one or the other by the same predicate and neither is any
/// use without the other.
#[derive(Debug)]
struct Mma {
    tiled: Kernel,
    grouped: Kernel,
    /// The shape these two were cut to, which is what the grid covering a call
    /// and the weight the accounting charges are both taken from — see
    /// [`Block`].
    block: Block,
}

impl Mma {
    /// The pair, refused where this device's simdgroup is not the one the block
    /// was cut for.
    ///
    /// **This is the one assumption in the layout that a wrong answer rather
    /// than a failure would follow from.** Every fragment origin in the kernel
    /// is `simdgroup_index_in_threadgroup` divided and taken modulo against
    /// [`MMA_SIMDS_ACROSS`], and the eight simdgroups that share the work are
    /// [`THREADS_PER_GROUP`] over [`NARROWEST_SIMD`]. A device that ran wider
    /// simdgroups would put fewer of them in the threadgroup than the block was
    /// cut into, and the rows no simdgroup reached would come back as the zeros
    /// `out` was allocated with — plausible values, no error, nothing downstream
    /// that could tell.
    ///
    /// **Asked of the compiled pipeline rather than assumed**, for the reason
    /// [`Kernel::simd_width`](crate::Kernel::simd_width) exists to be asked:
    /// every Apple GPU states 32 and Metal makes no promise that it always will.
    /// Refused at compile time rather than at dispatch, so that a machine this
    /// layout cannot serve says so once, before a token is decoded, instead of
    /// answering wrongly on every prefill.
    ///
    /// **A threadgroup the device will not dispatch is refused the same way**,
    /// and it is the other half of the same hazard: `block.threads` is swept,
    /// and a width past what the pipeline reports would otherwise be a dispatch
    /// Metal clamps rather than one it runs.
    fn of(tiled: Kernel, grouped: Kernel, block: Block) -> Result<Self, MetalError> {
        for kernel in [&tiled, &grouped] {
            if kernel.simd_width() != NARROWEST_SIMD {
                return Err(MetalError::UnexpectedSimdWidth {
                    entry: kernel.entry().to_string(),
                    width: kernel.simd_width(),
                    wanted: NARROWEST_SIMD,
                });
            }
            assert!(
                kernel.max_threads_per_group() >= block.threads,
                "{} takes at most {} threads a threadgroup, where the block wants {}",
                kernel.entry(),
                kernel.max_threads_per_group(),
                block.threads
            );
        }
        Ok(Self {
            tiled,
            grouped,
            block,
        })
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
/// The grid that covers `elements` output elements at a simdgroup each, which is
/// how every entry in front of the numerics flag counts a call.
///
/// One reading for the untiled entry and for the paired one, because a pair is
/// the untiled entry over twice the elements and nothing else — a second copy of
/// this would be a second place for the simdgroup width to be left out.
fn simdgroups(kernel: &Kernel, elements: usize, threads: usize) -> Grid {
    Grid::new(elements * kernel.simd_width(), threads)
}

/// How a call over an expert list this side holds is cut up, which is the tile
/// where the runs are long enough for one and an element apiece where they are
/// not.
///
/// One reading for a bank and for a [`PackedPair`], because it is the same
/// question about the same list: what a pair fuses is the two dispatches an
/// untiled call is, and which calls those are cannot be decided twice.
fn laid_out<'a>(experts: &[u32], matmul: &PackedMatmul) -> Layout<'a> {
    match tiles(experts, matmul.rows_a_tile) {
        false => Layout::Each,
        true => Layout::Tiled,
    }
}

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

    /// Where this bank's bytes start, which is everything the second half of a
    /// [`PackedPair`] differs from the first in — the other eight fields of
    /// [`PackedBank::shape`] are the call's rather than the bank's, and
    /// [`PackedPair::new`] is what says the two banks agree about the three that
    /// are not.
    fn bases(&self) -> [u32; PAIR_FIELDS] {
        let resident = self.resident.borrow();
        [
            extent(resident.codes.offset(), "where a bank's codes start"),
            extent(resident.scales.offset(), "where a bank's scales start"),
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
        self.a_row_each(experts, x.len());
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
        self.rows_a_source(rows, per_source, x.len());
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
        self.rows_a_source(rows, per_source, x.len());
        let layout = Layout::Grouped {
            order: grouped.order.arg(),
            through,
        };
        self.dispatch(batch, rows, per_source, grouped.experts.arg(), x, layout)
    }

    /// That a call of `rows` rows reads whole rows of the input, `per_source` of
    /// them apiece — the check [`PackedBank::encode_picked`] and
    /// [`PackedBank::encode_grouped`] each make of a list a dispatch wrote, and
    /// [`PackedPair`] makes of the same list on behalf of both its banks.
    fn rows_a_source(&self, rows: usize, per_source: usize, values: usize) {
        assert!(
            per_source > 0,
            "a row of a call reads some row of the input"
        );
        assert_eq!(
            self.sources(values),
            rows.div_ceil(per_source),
            "{values} values are not {} rows of {}",
            rows.div_ceil(per_source),
            self.in_dim
        );
    }

    /// That a call's rows are its input's, one each.
    fn a_row_each(&self, experts: &[u32], values: usize) {
        assert_eq!(
            self.sources(values),
            experts.len(),
            "{values} values are not {} rows of {}",
            experts.len(),
            self.in_dim
        );
    }

    /// That a call's rows are the input read over whole times, which is the
    /// shared bank's shape.
    fn whole_passes(&self, experts: &[u32], values: usize) {
        let sources = self.sources(values);
        assert!(
            experts.is_empty() || (sources > 0 && experts.len() % sources == 0),
            "{} rows are not whole passes over {sources} rows of input",
            experts.len()
        );
    }

    /// That every expert `experts` names is one this bank holds.
    ///
    /// The range *is* checked on this side, and it has to be: the kernel cannot,
    /// because an index past the bank is an offset past the buffer and a GPU
    /// read answers that with whatever is there rather than with a fault. A list
    /// a dispatch wrote is past this side, which is why
    /// [`PackedBank::encode_picked`] is not public.
    fn holds(&self, experts: &[u32]) -> Result<(), MatmulError> {
        match experts
            .iter()
            .find(|expert| **expert as usize >= self.experts)
        {
            Some(past) => Err(MatmulError::NoSuchExpert {
                expert: *past as usize,
                experts: self.experts,
            }),
            None => Ok(()),
        }
    }

    /// The same multiply over rows a dispatch already left on the device, every
    /// one of them read once per expert `experts` names in turn.
    ///
    /// **The shared bank's shape against one of its two banks**, which is what
    /// the cases about the modulo hold the kernel to. Only they: the model
    /// repeats both banks at once — see [`PackedPair::encode_repeating`], which
    /// is the same gather over the pair the shared bank is — so a caller with
    /// half a pair in hand is a case and nothing else.
    #[cfg(test)]
    pub(crate) fn encode_repeating(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let _timed = profile::scope(Op::Encode);
        self.whole_passes(experts, x.len());
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
        self.a_row_each(experts, x.len());
        self.listed(batch, experts, x)
    }

    /// A dispatch over an expert list this side holds, which is every one but
    /// [`PackedBank::encode_picked`]'s.
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
        self.holds(experts)?;
        if experts.is_empty() {
            return Ok(Pending::empty());
        }

        let mut chosen = self.device.inline(experts)?;
        self.dispatch(
            batch,
            experts.len(),
            1,
            chosen.arg(),
            x,
            laid_out(experts, self.matmul),
        )
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

        let (kernel, grid) =
            self.matmul
                .entry(&layout, rows, self.out_dim, self.in_dim, self.experts);
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
        let rows_a_read = self.matmul.rows_a_read(layout, rows, self.experts);
        let tiles = rows.div_ceil(rows_a_read);
        let boundaries = tiles.min(self.experts.saturating_sub(1));
        // **A straddling tile is charged every weight it could name, and a
        // straddling block only the two it can.** The two arms differ because
        // their gates do. A tile of [`ROWS_A_TILE`] rows is dispatched at runs of
        // four and a grouped call is dispatched at runs of two, so a tile can
        // hold as many runs as it has rows and is charged that many weights. A
        // block is dispatched at [`Block::runs_an_expert`] runs an expert — a
        // block's worth — so a block of its own height spans at most two of
        // them, and charging it thirty-two would put a routed bank's declared
        // bytes thirteen times over what it reads and its bandwidth column at
        // 290% of this part's peak. Both are bounds and neither is below what the
        // call reads; this is the tighter one its own predicate licenses.
        let extra = match rows_a_read > self.matmul.rows_a_tile {
            true => boundaries,
            false => (rows_a_read - 1) * boundaries,
        };
        let straddling = rows.min(tiles + extra);
        let read = match layout {
            Layout::Each => rows,
            // **A tiled call is charged flat only while a tile is the height
            // [`tiles`] gated it at.** That predicate keeps the shortest run at
            // least [`ROWS_A_TILE`] long, which bounds a straddling tile to two
            // experts and the whole call to one extra read a run. A block is
            // eight times that height against the same gate, so it can span more
            // runs than the flat charge admits — and a bandwidth column that
            // flattered this change is exactly what this method exists not to
            // print.
            Layout::Tiled if rows_a_read > self.matmul.rows_a_tile => straddling,
            Layout::Tiled => tiles,
            Layout::Grouped { .. } => straddling,
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

/// How many `uint`s the kernel's `Pair` struct declares.
const PAIR_FIELDS: usize = 2;

/// The two banks a SwiGLU multiplies one set of rows against, as the one
/// dispatch they can be.
///
/// **`gate` and `up` were two dispatches because they are two tensors and for
/// no other reason.** They read the same rows of the same input through the
/// same expert list, they read nothing either of them writes, and the
/// activation between them is what finally puts them together. So the only
/// thing that separates them inside a dispatch is which weight a simdgroup
/// walks — an index — and a decode step was paying a launch and half a
/// dispatch's worth of output elements for it, 82 times a step.
///
/// **What it buys is on two axes and the second is the larger.** Eighty-two
/// launches at about 4 microseconds is 0.35 ms of a step; doubling the output
/// elements of what remains moves each of them along the rate curve
/// `whether_a_decode_dispatch_is_short_of_the_machine` measured — a routed
/// bank's 12288 elements to 24576, a shared bank's 4096 to 8192 — and every
/// shape a decode step has is on the steep part of it.
///
/// **The two banks agree about everything but their bytes, and this is where
/// that is settled.** [`PackedBank::shape`] describes the call rather than the
/// weight in eight of its ten fields, so one shape serves both halves and what
/// the second half needs beside it is where its codes and scales start — see
/// [`PackedBank::bases`] and the kernel's `Pair`. A pair whose banks disagreed
/// about the experts, the width in or the width out would be one shape
/// describing two different calls, which is why those three are refused here.
#[derive(Debug)]
pub struct PackedPair<'a> {
    gate: PackedBank<'a>,
    up: PackedBank<'a>,
}

impl<'a> PackedPair<'a> {
    /// The two halves of one SwiGLU, checked to be each other's.
    pub fn new(gate: PackedBank<'a>, up: PackedBank<'a>) -> Result<Self, MatmulError> {
        let pair = |what, got, expected| match got == expected {
            true => Ok(()),
            false => Err(MatmulError::MismatchedBanks {
                what,
                expected,
                got,
            }),
        };
        pair("experts of up_proj", up.experts(), gate.experts())?;
        pair("the width up_proj maps from", up.in_dim(), gate.in_dim())?;
        pair("the width up_proj maps to", up.out_dim(), gate.out_dim())?;
        // A fused dispatch runs the gate's pipeline over the up's bytes, so two
        // banks wrapped against different compilations would be one of them
        // walked by the other's kernel. Not a shape and so not a
        // `MismatchedBanks`: every bank of a model is wrapped against the one
        // `PackedMatmul` its device compiled, and a caller that reached two is
        // a caller that built the model wrong.
        assert!(
            std::ptr::eq(gate.matmul, up.matmul),
            "a pair's two banks were wrapped against different compilations"
        );
        Ok(Self { gate, up })
    }

    /// The first half, which is what a caller asking a pair its shape asks —
    /// [`PackedPair::new`] is what says the second answers the same.
    pub(crate) fn gate(&self) -> &PackedBank<'a> {
        &self.gate
    }

    /// Both halves of [`PackedBank::encode`], over rows this side holds.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &[f32],
    ) -> Result<[Pending; 2], MatmulError> {
        self.gate.a_row_each(experts, x.len());
        if experts.is_empty() {
            return Ok([Pending::empty(), Pending::empty()]);
        }
        let _timed = profile::scope(Op::Encode);
        let mut x = self.gate.device.buffer(x)?;
        self.listed(batch, experts, &mut x)
    }

    /// Both halves over rows a dispatch already left on the device, every one of
    /// them read once per expert `experts` names in turn.
    ///
    /// **This is the shared bank's shape, and it is the same gather nobody
    /// gathered.** Every token goes through every shared expert, so the rows the
    /// bank runs are the input laid end to end after itself once per expert —
    /// `[n_shared * tokens, in_dim]` of which every row is already `[tokens,
    /// in_dim]` somewhere. Read at a modulo it is not laid out at all, and the
    /// input can be the buffer the router's gate and the routed bank are reading
    /// in the same command buffer.
    pub(crate) fn encode_repeating(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<[Pending; 2], MatmulError> {
        let _timed = profile::scope(Op::Encode);
        self.gate.whole_passes(experts, x.len());
        self.listed(batch, experts, x)
    }

    /// Both halves of [`PackedBank::encode_picked`], which is the routed bank's
    /// shape at a decode step and the one call that is untiled whatever its
    /// rows.
    pub(crate) fn encode_picked(
        &self,
        batch: &mut Batch<'_>,
        chosen: &mut Buffer<u32>,
        x: &mut Buffer<f32>,
        per_source: usize,
    ) -> Result<[Pending; 2], MatmulError> {
        let _timed = profile::scope(Op::Encode);
        let rows = chosen.len();
        self.gate.rows_a_source(rows, per_source, x.len());
        self.fused(batch, rows, per_source, chosen.arg(), x)
    }

    /// Both halves of [`PackedBank::encode_grouped`], which is **two dispatches
    /// and stays two**.
    ///
    /// A grouped call is a prefill's routed bank: hundreds of rows sorted by
    /// expert, dispatched through a tile that carries a running sum and an input
    /// offset per row per column. There is no launch to save that is worth
    /// noticing beside the walk, and no width to win — a 769-token prefill's
    /// routed gate is 9.4 million output elements and past the top of the rate
    /// curve twice over. What a fused tile would cost is registers, on the entry
    /// where they are already the binding constraint.
    pub(crate) fn encode_grouped(
        &self,
        batch: &mut Batch<'_>,
        grouped: &mut Grouped,
        x: &mut Buffer<f32>,
        per_source: usize,
    ) -> Result<[Pending; 2], MatmulError> {
        Ok([
            self.gate
                .encode_grouped(batch, grouped, x, per_source, Through::Gathered)?,
            self.up
                .encode_grouped(batch, grouped, x, per_source, Through::Gathered)?,
        ])
    }

    /// Both halves over an expert list this side holds: one dispatch where the
    /// call is untiled, and the two the tile has always been where it is not.
    ///
    /// **The tiled arm is a prefill's shared bank** — its rows are the input
    /// laid end to end once per shared expert, so they run long — and the reason
    /// it is not fused is [`PackedPair::encode_grouped`]'s.
    fn listed(
        &self,
        batch: &mut Batch<'_>,
        experts: &[u32],
        x: &mut Buffer<f32>,
    ) -> Result<[Pending; 2], MatmulError> {
        // Asked of the gate alone because [`PackedPair::new`] is what says the
        // two banks hold the same experts.
        self.gate.holds(experts)?;
        if experts.is_empty() {
            return Ok([Pending::empty(), Pending::empty()]);
        }
        if let Layout::Tiled = laid_out(experts, self.gate.matmul) {
            return Ok([
                self.gate.listed(batch, experts, x)?,
                self.up.listed(batch, experts, x)?,
            ]);
        }

        let mut chosen = self.gate.device.inline(experts)?;
        self.fused(batch, experts.len(), 1, chosen.arg(), x)
    }

    /// The dispatch itself: one grid of twice the call's output elements, the
    /// first half of it against `gate` and the second against `up`.
    ///
    /// **What each half declares is what it declared apart.** Both banks read
    /// the whole input — every output element is a simdgroup that walks one
    /// weight row against all of it — so the bytes a fused call moves are the
    /// two calls' summed, and a bandwidth column over this dispatch reads what
    /// the same column read over the pair it replaces.
    fn fused(
        &self,
        batch: &mut Batch<'_>,
        rows: usize,
        per_source: usize,
        chosen: Arg<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<[Pending; 2], MatmulError> {
        if rows == 0 {
            return Ok([Pending::empty(), Pending::empty()]);
        }
        let device = self.gate.device;
        let out_dim = self.gate.out_dim;
        // Both are read out of `resident` and so are built before either bank is
        // borrowed mutably for the binding below.
        let fields = self
            .gate
            .shape(rows, per_source, self.gate.sources(x.len()), &Layout::Each);
        let mut shape = device.inline(&fields)?;
        let bases = self.up.bases();
        let mut beside = device.inline(&bases)?;
        let mut gate = self.gate.resident.borrow_mut();
        let mut up = self.up.resident.borrow_mut();
        let (gate, up) = (&mut *gate, &mut *up);
        let mut out = device.zeroed::<f32>(rows * out_dim)?;
        let mut beside_out = device.zeroed::<f32>(rows * out_dim)?;

        let (kernel, grid) = self.gate.matmul.paired(rows, out_dim, self.gate.in_dim);
        let moves = self.gate.moves(rows, x.len(), &Layout::Each)
            + self.up.moves(rows, x.len(), &Layout::Each);
        batch.add(
            kernel,
            &[
                shape.arg(),
                chosen,
                x.arg(),
                gate.codes.arg(),
                gate.scales.arg(),
                out.arg(),
                beside.arg(),
                up.codes.arg(),
                up.scales.arg(),
                beside_out.arg(),
            ],
            grid,
            moves,
        )?;

        Ok([
            Pending { out: Some(out) },
            Pending {
                out: Some(beside_out),
            },
        ])
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
constant uint CHUNKS_IN_FLIGHT = {CHUNKS_IN_FLIGHT};
constant uint EXPONENT_SHIFT = {EXPONENT_SHIFT};
constant uint ROWS_A_TILE = {ROWS_A_TILE};
constant uint COLS_A_TILE = {COLS_A_TILE};
constant uint RESIDENCY = {RESIDENCY};
constant float ELEMENTS[] = {{ {} }};
{BODY}{}{}{}{}",
        (1u32 << BITS) - 1,
        elements.join(", "),
        element_entry(ELEMENT, ENTRY, false),
        element_entry(ELEMENT, PAIRED_ENTRY, true),
        tiled_entry(TILED_ENTRY, false),
        tiled_entry(GROUPED_ENTRY, true),
    )
}

/// The kernel under the numerics asked for, which is [`source`] and — where the
/// flag says so — the two entries that carry the reduction the other way.
///
/// **Appended rather than substituted**, so that a reference run compiles the
/// same string it compiled before this flag existed: every byte of `source`
/// above is where it was, and what production adds is written after it.
/// **The shape and the source have to travel together**, which is why this is
/// one function and not a source plus a knob: the grid a dispatch covers a call
/// with is sized from the block, so a source written here and dispatched under
/// another shape would leave output no threadgroup reached.
/// [`PackedMatmul::blocked`] is the only way to compile one, and it takes both.
pub(crate) fn source_blocked(numerics: Numerics, block: Block) -> String {
    let mut written = source();
    if numerics.is_production() {
        block.holds();
        written.push_str(&block.declares());
        written.push_str(&mma_entry(MMA_TILED_ENTRY, false));
        written.push_str(&mma_entry(MMA_GROUPED_ENTRY, true));
    }
    written
}

/// [`source_blocked`] with the two split entries after it, which is the source
/// the sweep that priced them compiles and the only one that carries them.
///
/// **They are not in the source anything dispatches**, which is what this
/// milestone concluded rather than what it set out to build:
/// `what_a_packed_multiply_costs_at_each_count_its_walk_is_cut_into` reads a
/// cut of *one* — the plain entry's own walk, over the same bytes, answering the
/// same bits — at 121 to 142% of the entry it stands in front of, and the arm
/// with the barrier taken out reads the same. What the untiled entry has that
/// this cannot have is that a threadgroup of it shares nothing at all, and there
/// is nowhere to put a partial that does not end that.
#[cfg(test)]
pub(crate) fn source_cut(block: Block) -> String {
    let mut written = source_blocked(Numerics::Production, block);
    written.push_str(&format!("\nconstant uint MOST_SPLITS = {MOST_SPLITS};\n"));
    written.push_str(&element_entry(SPLIT, SPLIT_ENTRY, false));
    written.push_str(&element_entry(SPLIT, SPLIT_PAIRED_ENTRY, true));
    written
}

/// How many `uint`s the kernel's `Shape` struct declares.
const SHAPE_FIELDS: usize = 10;

/// Everything of the kernel that the format does not decide.
///
/// `weight_dot` is the decode, kept a function of its own because it is the one
/// reading of the format on this side of the engine and a second copy of it
/// would be a second reading that could drift.
pub(crate) const BODY: &str = r#"
/// The float a 4-bit code stands for.
///
/// **A gather into sixteen constant floats, which is the cheapest decode
/// measured** — see `what_each_way_of_decoding_a_packed_byte_costs`, where
/// assembling the same value out of the code's own bits costs 11.0 and 13.6% of
/// this kernel's two shapes and one gather into a table of whole bytes costs 7.5
/// and 2.1%. The table is 64 bytes and every lane of a simdgroup indexes it
/// separately; what that reads as, against those two, is a load nothing waits
/// on.
///
/// One function for both entries because it is the one reading of the format on
/// this side of the engine, and a second copy of it would be a second reading
/// that could drift.
inline float element(uint code) {
    return ELEMENTS[code];
}

/// One chunk of a weight row: [`BYTES_PER_LANE`] packed bytes from `b`, decoded
/// and multiplied against the input floats their codes stand against, under the
/// one scale byte the chunk lies inside.
///
/// A byte is two codes, low nibble first, and its group's scale is a power of
/// two — so every product here is exact and only the order they are summed in
/// separates this from any other way of adding them up.
///
/// The loop's trip count is a compile-time constant, so it is unrolled and its
/// loads have no dependency on each other — which is the whole of what a chunk
/// buys.
///
/// **The chunk is bounded by `bytes` at the caller and not by anything inside
/// it**, so what keeps a lane inside its own weight row is that a row is a whole
/// number of chunks. `pairs` on the Rust side refuses a width that is not whole
/// groups of 32 codes, which makes every row a whole number of 16 packed bytes,
/// and 4 divides 16 — and the same fact is what puts a whole chunk under the one
/// scale byte read for it. A GPU read past a row is an address inside the next
/// one rather than a fault, so this is the invariant to check first if either
/// constant moves.
inline float chunk_dot(
    device const uchar *packed,
    device const uchar *scale,
    device const float *values,
    uint b
) {
    float dot = 0.0f;
    for (uint i = 0; i < BYTES_PER_LANE; ++i) {
        const uint code = packed[b + i];
        const uint low = code & CODE_MASK;
        const uint high = (code >> BITS) & CODE_MASK;
        device const float *v = values + (b + i) * CODES_PER_BYTE;

        dot += element(low) * v[0] + element(high) * v[1];
    }
    return dot * as_type<float>(uint(scale[b / BYTES_PER_GROUP]) << EXPONENT_SHIFT);
}

/// One output element: lane `l` walks the weight row from byte
/// `l * BYTES_PER_LANE` in strides of that many times the simdgroup width, and
/// the caller reduces what the lanes held.
///
/// **`CHUNKS_IN_FLIGHT` chunks are read before any of them is added**, and that
/// is the whole of what this loop's shape is for. The walk's trip count is a
/// runtime value, so nothing unrolls it and a lane holds one chunk at a time: it
/// issues four bytes, waits a memory latency for them, adds them, and issues four
/// more. `what_a_decode_dispatchs_fixed_cost_is_made_of` prices that wait at 0.41
/// microseconds a step, which across the 457 calls a decode step makes is 2.75 ms
/// of an 18.6 ms step spent waiting rather than reading.
///
/// **The order every product enters every sum in is untouched, and that is by
/// construction rather than within a tolerance.** The chunks are read in the
/// order the walk visits them and added in that same order, so this reduces to
/// the loop it replaces at [`CHUNKS_IN_FLIGHT`] of one, and the remainder walk
/// below carries on from where the held one stopped, in the same stride, adding
/// in the same direction.
/// `a_walk_that_holds_several_chunks_answers_the_bits_one_holding_one_answers`
/// is where that is held, over every reduction the checkpoint has and two that do
/// not divide.
inline float weight_dot(
    device const uchar *packed,
    device const uchar *scale,
    device const float *values,
    uint bytes,
    uint lane,
    uint width
) {
    const uint step = width * BYTES_PER_LANE;
    float sum = 0.0f;
    uint b = lane * BYTES_PER_LANE;
    for (; b + (CHUNKS_IN_FLIGHT - 1) * step < bytes; b += CHUNKS_IN_FLIGHT * step) {
        float held[CHUNKS_IN_FLIGHT];
        for (uint c = 0; c < CHUNKS_IN_FLIGHT; ++c) {
            held[c] = chunk_dot(packed, scale, values, b + c * step);
        }
        for (uint c = 0; c < CHUNKS_IN_FLIGHT; ++c) {
            sum += held[c];
        }
    }
    for (; b < bytes; b += step) {
        sum += chunk_dot(packed, scale, values, b);
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

/// Where the second half of a pair's weight begins, which is everything the two
/// banks of one SwiGLU differ in — see `PackedPair`, which is what says the
/// shape above describes both of them.
struct Pair {
    uint code_base;
    uint scale_base;
};

/// Where a row of the input the tile's `r`th row multiplies begins.
inline uint tile_source(constant Shape &shape, uint row) {
    return ((row / shape.per_source) % shape.sources) * shape.in_dim;
}

"#;

/// An untiled kernel, emitted twice: once against one bank and once against the
/// two a SwiGLU reads the same rows through.
///
/// **Written once and compiled twice**, for the reason [`tiled_entry`] gives.
/// The two differ in what an element index is taken modulo, and in nothing else
/// at all — the same walk, the same reduction, the same one output element. So
/// the paired entry's answer is the plain one's bit for bit by construction
/// rather than within a tolerance, and
/// `a_paired_dispatch_answers_what_the_two_dispatches_it_replaces_answer` is
/// where that is held.
///
/// **One reading of the pair for both untiled walks**, which is what
/// [`over_the_rows`] is to the two tiled ones: [`ELEMENT`] and [`SPLIT`] differ
/// in how many simdgroups reduce a row and in nothing about which of a pair's
/// weights one of them walks, so a second copy of these two would be a second
/// place for a pair to be pointed at the wrong bank.
fn element_entry(walk: &str, entry: &str, paired: bool) -> String {
    let (beside, picks) = match paired {
        false => ("", ALONE),
        true => (BESIDE, BOTH),
    };
    substituted(
        walk,
        &[
            ("__ENTRY__", entry),
            ("__BESIDE__", beside),
            ("__PICKS__", picks),
        ],
    )
}

/// `walk` with each placeholder replaced by what it stands for.
///
/// Each substitution replaces every occurrence, so what keeps them from reaching
/// into each other is that none of them writes a placeholder — which is a
/// property of the values rather than of the mechanism, and so is asserted
/// rather than read off them.
fn substituted(walk: &str, written: &[(&str, &str)]) -> String {
    assert!(
        !written.iter().any(|(_, value)| value.contains("__")),
        "a substitution that writes a placeholder would be substituted again"
    );
    written
        .iter()
        .fold(walk.to_string(), |walk, (placeholder, value)| {
            walk.replace(placeholder, value)
        })
}

/// The four bindings the second half of a pair arrives in, which the entry that
/// runs one bank is handed none of.
const BESIDE: &str = "
    constant Pair &beside [[buffer(6)]],
    device const uchar *beside_codes [[buffer(7)]],
    device const uchar *beside_scales [[buffer(8)]],
    device float *beside_out [[buffer(9)]],";

/// Which weight a simdgroup of the plain entry walks, which is the only one it
/// was given.
const ALONE: &str = "    const uint index = element;
    device const uchar *packed = codes;
    device const uchar *scaled = scales;
    device float *written = out;
    const uint code_base = shape.code_base;
    const uint scale_base = shape.scale_base;";

/// Which of a pair's two weights a simdgroup walks, which is the first for the
/// call's own elements and the second for the elements after them.
///
/// **A simdgroup is one output element, so this is uniform across every lane of
/// one** and costs no divergence: the whole simdgroup takes one side of it,
/// and which side is a comparison against a value the dispatch's shape fixes.
const BOTH: &str = "    const bool second = element >= elements;
    const uint index = element - (second ? elements : 0);
    device const uchar *packed = second ? beside_codes : codes;
    device const uchar *scaled = second ? beside_scales : scales;
    device float *written = second ? beside_out : out;
    const uint code_base = second ? beside.code_base : shape.code_base;
    const uint scale_base = second ? beside.scale_base : shape.scale_base;";

/// The untiled kernel with the three expressions [`element_entry`] decides
/// written as placeholders.
const ELEMENT: &str = r#"
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
///
/// **`packed_matmul_pair` is the same call against two banks**, which is what a
/// SwiGLU's gate and up are to each other: they read the same rows of the same
/// input through the same expert list against different weights, so one grid of
/// twice the output elements computes both and the second half of it is the
/// first half's element index over again. Nothing about a lane's walk moves —
/// which weight it walks is decided once, above the loop, out of an index a
/// simdgroup shares.
kernel void __ENTRY__(
    constant Shape &shape [[buffer(0)]],
    device const uint *experts [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    device float *out [[buffer(5)]],__BESIDE__
    uint position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]
) {
    const uint element = position / width;
    const uint elements = shape.rows * shape.out_dim;
__PICKS__
    if (index >= elements) {
        return;
    }

    const uint row = index / shape.out_dim;
    const uint col = index % shape.out_dim;
    const uint expert = experts[row];
    const uint source = (row / shape.per_source) % shape.sources;
    const uint bytes = shape.in_dim / CODES_PER_BYTE;

    float sum = weight_dot(
        packed + code_base + (ulong)expert * shape.code_stride + (ulong)col * bytes,
        scaled + scale_base + (ulong)expert * shape.scale_stride
            + (ulong)col * (bytes / BYTES_PER_GROUP),
        x + (ulong)source * shape.in_dim,
        bytes,
        lane,
        width
    );

    sum = simd_sum(sum);
    if (lane == 0) {
        written[index] = sum;
    }
}
"#;

/// The untiled kernel with an element's walk cut between the simdgroups of its
/// threadgroup, with the two expressions [`element_entry`] decides written as
/// placeholders.
#[cfg(test)]
const SPLIT: &str = r#"
/// `packed_matmul` with one element's weight row walked by several simdgroups
/// rather than by one.
///
/// **A threadgroup is one output element here, where a threadgroup of
/// `packed_matmul` is several independent ones.** That is the whole of the
/// difference in shape: the simdgroups of this threadgroup all reduce the same
/// row, so they have something to say to each other and need somewhere to say
/// it, and `partial` and the barrier below are that.
///
/// **What it buys is simdgroups in flight.** An output element is one simdgroup,
/// so a call of 512 elements is 512 simdgroups over 80 cores whatever else is
/// true of it — and `whether_a_decode_dispatch_is_short_of_the_machine` puts
/// that call at 12% of the bus where 131072 elements reach 95%. Cutting the walk
/// `n` ways is `n` times the simdgroups over the same bytes, each holding `1/n`
/// of the row and each issuing its own loads: the term the walk waits on is the
/// loads it has outstanding, which `what_a_decode_steps_packed_matmul_is_bound
/// _by` measured at a fifth to two fifths a term, and not the bandwidth behind
/// them.
///
/// **The split is the lane stride widened and nothing else.** Simdgroup `cut`
/// takes the lanes `cut * width` to `(cut + 1) * width` of one walk `splits *
/// width` lanes wide, so the bytes the whole threadgroup reads are exactly the
/// bytes one simdgroup read, in the same chunks, under the same scales. It is
/// the same `weight_dot` — one reading of the walk, as there has only ever been
/// one.
///
/// **And it is the order that moves, which is why this is behind the flag.** The
/// unsplit walk is one `simd_sum` over 32 lanes each holding every 32nd chunk;
/// this is `splits` of those over a coarser partition, added afterwards in the
/// order the cuts lie. Both sum the same exact products — a code is one of
/// sixteen table values and a group scale is a power of two, so nothing either
/// forms is rounded — and they associate them differently. There is no array of
/// bits to hold this to and there cannot be one.
kernel void __ENTRY__(
    constant Shape &shape [[buffer(0)]],
    device const uint *experts [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    device float *out [[buffer(5)]],__BESIDE__
    uint element [[threadgroup_position_in_grid]],
    uint cut [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    /// **Read off the dispatch rather than taken as a constant**, which is what
    /// makes the count a value the host may choose per call: a threadgroup is
    /// one element's cuts, so the width it was dispatched at *is* the count, and
    /// the two cannot disagree the way a source constant and a grid can.
    const uint splits = threads / width;
    threadgroup float partial[MOST_SPLITS];

    const uint elements = shape.rows * shape.out_dim;
__PICKS__
    // **Uniform across the threadgroup, which is what the barrier below needs**:
    // a threadgroup is one element, so this returns all of the cuts or none of
    // them. The grid is an exact multiple of the threadgroup — `elements` of
    // them, one apiece — so nothing reaches it; it is here because the bound is
    // the kernel's to state and a grid this side sized differently would find
    // it.
    if (index >= elements) {
        return;
    }

    const uint row = index / shape.out_dim;
    const uint col = index % shape.out_dim;
    const uint expert = experts[row];
    const uint source = (row / shape.per_source) % shape.sources;
    const uint bytes = shape.in_dim / CODES_PER_BYTE;

    float sum = weight_dot(
        packed + code_base + (ulong)expert * shape.code_stride + (ulong)col * bytes,
        scaled + scale_base + (ulong)expert * shape.scale_stride
            + (ulong)col * (bytes / BYTES_PER_GROUP),
        x + (ulong)source * shape.in_dim,
        bytes,
        cut * width + lane,
        splits * width
    );

    sum = simd_sum(sum);
    if (lane == 0) {
        partial[cut] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (cut != 0 || lane != 0) {
        return;
    }

    // In the order the cuts lie, which is the order the row does. Nothing about
    // the answer needs it — the cuts are added, and a sum is a sum — but a
    // reduction whose order is a scheduling accident is one whose answer moves
    // between two runs of the same build, and a differential instrument cannot
    // tell that apart from the arithmetic it is there to measure.
    float whole = 0.0f;
    for (uint c = 0; c < splits; ++c) {
        whole += partial[c];
    }
    written[index] = whole;
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
    over_the_rows(TILE, entry, grouped)
}

/// The three expressions that separate a call over the rows as they lie from one
/// over a permutation a dispatch wrote, written into whichever walk asked for
/// them.
///
/// **One reading of the indirection for both walks**, because it is the same
/// indirection: [`TILE`] and [`MMA`] differ in how they accumulate and in
/// nothing about which row of the input a row reads or which row of the output
/// it writes. A second copy of these three would be a second place for
/// `shape.scatters` to be got backwards.
fn over_the_rows(walk: &str, entry: &str, grouped: bool) -> String {
    let (order, reads, writes) = match grouped {
        false => ("", "row", "row"),
        true => (
            "\n    device const uint *order [[buffer(6)]],",
            "shape.scatters ? row : order[row]",
            "shape.scatters ? order[row] : row",
        ),
    };
    substituted(
        walk,
        &[
            ("__ENTRY__", entry),
            ("__ORDER__", order),
            ("__READS__", reads),
            ("__WRITES__", writes),
        ],
    )
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
    uint local [[thread_position_in_threadgroup]],
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

    // Declared so that a core holds three threadgroups of this kernel rather
    // than as many as it has room for, and read only for the zero it was filled
    // with — see RESIDENCY. Every thread fills and reads its own entry, so
    // nothing here needs a barrier.
    //
    // **`volatile` is what keeps it**, and without it this array is not here at
    // all: a store of a zero to a thread's own slot followed by a load of the
    // same slot is exactly what a forwarding pass removes, and measured against
    // a pipeline that reported no threadgroup memory whatever this declared.
    threadgroup float residency[RESIDENCY];
    threadgroup volatile float *held = residency;
    held[local] = 0.0f;

    uint sources[ROWS_A_TILE];
    float sums[ROWS_A_TILE][COLS_A_TILE];
    for (uint r = 0; r < ROWS_A_TILE; ++r) {
        const uint row = first + min(r, rows - 1);
        sources[r] = tile_source(shape, __READS__);
        for (uint c = 0; c < COLS_A_TILE; ++c) {
            sums[r][c] = held[local];
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
                low[c] = element(code & CODE_MASK);
                high[c] = element((code >> BITS) & CODE_MASK);
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

/// The production entries, written once and compiled twice the way [`TILE`] is
/// and against the same three placeholders.
///
/// **What is behind [`Numerics::Production`] is this and nothing else.** It
/// takes the same six bindings from the same encoder, reads the same `Shape`,
/// answers the same `[rows, out_dim]`, and is chosen by the same predicates at
/// the same shapes. The tiling above it, the grouping in front of it,
/// `splits_for`, the occupancy declarations and every submission decision are
/// shared. What differs is the accumulate, and the accumulate cannot be
/// bit-compared — which is the whole reason the flag exists.
fn mma_entry(entry: &str, grouped: bool) -> String {
    over_the_rows(MMA, entry, grouped)
}

/// The block, the staging and the hardware multiply, with the three expressions
/// [`mma_entry`] decides written as placeholders.
///
/// # What mlx's own quantised matmul does that this does not
///
/// **Everything read off the other engine before this was its dense path** —
/// `steel_gemm`, `steel_attention`. mlx ships the direct analogue of this
/// kernel and the metallib in `reference/.venv` at mlx 0.32.0 carries it under
/// this checkpoint's own format:
/// `mxfp4_gather_qmm_rhs_nt_float_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2`, whose
/// source is `fp_quantized.h`'s `gather_qmm_rhs` over `steel/gemm/mma.h`.
///
/// **Its shape is smaller than this one's and its inner loop is the same.** 16
/// rows by 32 columns over `wm × wn` = 1 × 2 simdgroups, which is 64 threads
/// where this runs 256; `BK` is 32, which is this kernel's step; and its `k`
/// loop is barrier, fill both tiles, barrier, multiply `BK / 8` times — the
/// arrangement here, barrier for barrier. `TM = BM / (8 × WM)` and
/// `TN = BN / (8 × WN)` are both 2, so a simdgroup there holds **four
/// accumulators against four fragment loads, which is 1:1** — the ratio a
/// taller block was wanted for on this side, run by the kernel that was
/// supposed to be the reason to want it.
///
/// What it does differently is three things, and none of them is the shape:
///
/// - **It never calls `simdgroup_load`.** `BaseMMAFrag::load` reads the two
///   elements each lane owns straight out of threadgroup memory through a
///   static lane-to-coordinate map, and takes the transpose by swapping the two
///   strides it indexes with. This kernel issues a `simdgroup_load` a fragment
///   and a *transposing* one for the weight.
/// - **It has no answer tile.** `BlockMMA::store_result` writes the fragments to
///   device memory where they lie, and `store_result_slice` is how it refuses a
///   ragged edge.
/// - **It walks runs rather than passes.** A block whose rows name several
///   experts is cut at the boundaries of the sorted index list and each run gets
///   its own gemm and its own slice of the store, where this kernel runs the
///   whole reduction once per distinct expert and zeroes the rows that are not
///   that pass's.
///
/// # The three claims about mlx this file inherited
///
/// Two hold and the third is false, and all three are read off that metallib
/// rather than off anything written here. Its steel GEMM does run 64 to 128
/// threads, and its 64×64 dense tile does reuse fragments 2.7:1. **What is false
/// is that mlx ships no fp32 quantised matmul**: `mxfp4_qmm_t_float_gs_32_b_4`
/// is in the same binary, and every one of the four quantised families it
/// carries — affine, mxfp4, mxfp8, nvfp4 — is instantiated at `float` beside
/// `float16_t` and `bfloat16_t`. That claim is what the 16-bit arm was argued
/// from, and `what_sixteen_bit_operands_cost_and_what_they_give_up` is what
/// answers it on this part instead.
///
/// # What the generated code says, and what it will not say
///
/// The Metal toolchain is installed off the nix PATH —
/// `xcodebuild -showComponent MetalToolchain` gives the asset, and
/// `Metal.xctoolchain/usr/bin` holds `metal`, `metallib`, `metal-objdump` and
/// `applegpu-nt`, the AIR-to-native translator. Three readings come out of it
/// and the one everybody wants does not:
///
/// - **The pipeline this device builds reports 1024 threads a threadgroup for
///   both production entries**, which is this part's own maximum — so nothing
///   about the block's occupancy is decided by its register allocation.
/// - **The AIR the frontend emits keeps every fragment loop rolled**, and holds
///   `lhs`, `rhs` and `sums` in `alloca`s indexed by the loop variable rather
///   than in registers. That is not what runs: unrolling and register allocation
///   happen in the AGX backend, and `#pragma clang loop unroll(full)` on all
///   four loops changes the emitted metallib by one line. **So the metallib is
///   not evidence about the inner loop's schedule**, which is the reading a
///   source-level intuition would most like to take from it.
/// - **The native object can be built and cannot be read.** `applegpu-nt -arch
///   applegpu_g15d` translates the metallib into a Mach-O for this part —
///   `mma_matmul_rows` is 4736 bytes of `__compute` against `packed_matmul_rows`
///   at 7136 — but the toolchain ships no instruction printer for the `agx`
///   targets (`metal-objdump: no instruction printer for target agx3---macho`),
///   so the instruction stream is not readable with what Apple distributes.
///
/// **`what_the_matrix_instruction_issues_at` is what replaced that reading**,
/// and it answers the question the disassembly was wanted for: how many of this
/// kernel's cycles are the instruction. A ceiling measured in registers is a
/// stronger answer than a listing anyway, because it needs no model of what the
/// part does with what it is issued.
const MMA: &str = r#"
/// One threadgroup per MMA_ROWS_A_BLOCK rows of MMA_COLS_A_BLOCK columns, with
/// the reduction carried by `simdgroup_multiply_accumulate` rather than by a
/// lane-strided walk under a `simd_sum`.
///
/// **The weight is decoded once into threadgroup memory and multiplied against
/// every row of the block.** That is the whole of what this buys and it is two
/// things at once. A packed byte's two codes are gathered out of ELEMENTS once
/// for MMA_ROWS_A_BLOCK rows where the reference tile gathers them once for
/// ROWS_A_TILE, so the decode's dependency chain — 30% of that kernel by
/// ablation — is amortised eight times as far; and the multiply that follows is
/// an instruction carrying 512 multiply-adds where the reference carries one.
///
/// **The answer is not the reference kernel's bit for bit and cannot be.** A
/// hardware 8x8x8 multiply-accumulate sums its `k` dimension in an order the
/// instruction defines and this side does not choose. Every product either
/// kernel forms is exact — a code is one of sixteen table values and a group
/// scale is a power of two — so the two differ by summation order alone.
///
/// **And the order this one takes is the worse-conditioned of the two, which is
/// measured rather than assumed.** A fragment accumulator is one running sum
/// over the whole reduction: 4096 codes are 512 accumulate steps into the same
/// register, one after another. The reference splits the same reduction 32 ways
/// — a lane walks 128 products of it and a tree adds the 32 partials — so its
/// longest chain is a quarter as long and its drift does not grow with the
/// reduction at all. Measured against an f64 accumulation of the same products,
/// this one drifts 1.6e-7 at a reduction of 32 and 4.1e-6 at 4096 where the
/// reference holds 9.0e-8 to 1.4e-7 across the same range. See
/// MMA_TOLERANCE and `a_block_answers_the_reference_tile_where_neither_extent
/// _divides_it`, which is where those figures are taken.
///
/// **The scale is applied at the staging rather than at the accumulate**, which
/// the reference kernel cannot do and this one can afford: a scale is a power of
/// two, so a code times its scale is exact and no rounding is moved by folding
/// it in early. What it buys is that the MMA sees plain floats and owes nothing
/// per step.
///
/// **A block whose rows name several experts runs once per expert they name.** A
/// staged weight block is one expert's, so there is no one block that serves
/// rows disagreeing about which weight they read. Each pass zeroes the rows that
/// named some other expert on the way in, and a zeroed row contributes exactly
/// 0.0 to its own outputs — so what a row ends up holding is the pass that was
/// its own, and the passes that were not its own added nothing to it. Correct
/// for any expert list, exactly as the reference tile is, and fast only where
/// the runs are at least a block — which is what `runs_an_expert` is for and
/// what keeps a grouped call off this entry until the routing can feed it.
///
/// **Every barrier below is reached by every thread of the threadgroup**, and
/// what makes that true rather than hoped is that the pass loop and the step
/// loop are both bounded by threadgroup-uniform values: `rows`, `opens` and
/// `bytes` are the same for all 256 threads, so no thread takes a branch that
/// skips a barrier another thread waits at.
kernel void __ENTRY__(
    constant Shape &shape [[buffer(0)]],
    device const uint *experts [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    device float *out [[buffer(5)]],__ORDER__
    uint block [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint across = (shape.out_dim + MMA_COLS_A_BLOCK - 1) / MMA_COLS_A_BLOCK;
    const uint first = (block / across) * MMA_ROWS_A_BLOCK;
    const uint leftmost = (block % across) * MMA_COLS_A_BLOCK;
    // The grid covers exactly the blocks the two extents cut the call into, so
    // this is the same guard the reference entries carry for the same reason:
    // a dispatch whose shape and grid disagreed would be a wrong answer rather
    // than a failure.
    if (first >= shape.rows) {
        return;
    }

    const uint rows = min((uint)MMA_ROWS_A_BLOCK, shape.rows - first);
    const uint cols = min((uint)MMA_COLS_A_BLOCK, shape.out_dim - leftmost);
    const uint bytes = shape.in_dim / CODES_PER_BYTE;
    const uint scale_bytes = bytes / BYTES_PER_GROUP;
    const uint step_bytes = MMA_CODES_A_STEP / CODES_PER_BYTE;

    // The answer is written over the staging, because the two are never both
    // live: nothing reads a staged tile after the last multiply-accumulate and
    // nothing writes the answer before it. What the overlap buys is not the
    // memory but the occupancy: what a threadgroup declares decides how many of
    // them a core holds, and how much of a barrier one waits at the others
    // cover. RESIDENCY's sweep is where that turn was measured — three
    // threadgroups a core at a declaration of 24 KiB and four at 20.
    //
    // `Block::holds` is where the answer is held to fitting inside the staging.
    threadgroup float staged[(MMA_ROWS_A_BLOCK + MMA_COLS_A_BLOCK) * MMA_STAGED_STRIDE];
    threadgroup float *staged_x = staged;
    threadgroup float *staged_w = staged + MMA_ROWS_A_BLOCK * MMA_STAGED_STRIDE;
    threadgroup float *answered = staged;
    threadgroup uint held[MMA_ROWS_A_BLOCK];
    threadgroup bool opens[MMA_ROWS_A_BLOCK];

    // The block's expert list, read into threadgroup memory once. The pass loop
    // below asks after it O(rows^2) times and a device load an ask would cost
    // more than the uniform case it is there to serve.
    //
    // A slot past the block's own rows repeats the last of them, so that the
    // read is inside the list whatever the call's shape leaves over; `opens`
    // then refuses it, which is what keeps a repeat from being a pass.
    if (local < MMA_ROWS_A_BLOCK) {
        held[local] = experts[first + min(local, rows - 1)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local < MMA_ROWS_A_BLOCK) {
        bool first_time = local < rows;
        for (uint before = 0; before < local; ++before) {
            first_time = first_time && (held[before] != held[local]);
        }
        opens[local] = first_time;
    }

    // Where this thread stages. The two tiles are filled cooperatively and the
    // divisions are exact by construction — the prelude's static assertions are
    // where that is held — so every thread reads the same count and no lane is
    // idle in either fill.
    const uint x_row = local / MMA_THREADS_A_STAGED_ROW;
    const uint x_at = (local % MMA_THREADS_A_STAGED_ROW) * MMA_FLOATS_A_THREAD;
    const uint w_col = local / MMA_THREADS_A_WEIGHT_ROW;
    const uint w_at = (local % MMA_THREADS_A_WEIGHT_ROW) * MMA_BYTES_A_THREAD;

    // The row of the input this thread's staged row reads, and the column of the
    // weight its staged column is. Both clamp to the last live one so that every
    // load below is inside the buffer it indexes; what a clamped slot produces
    // is zeroed on the way in or never written out.
    const uint row = first + min(x_row, rows - 1);
    const uint source = tile_source(shape, __READS__);
    const uint column = leftmost + min(w_col, cols - 1);

    const uint down = simd / MMA_SIMDS_ACROSS;
    const uint alongside = simd % MMA_SIMDS_ACROSS;

    simdgroup_float8x8 sums[MMA_FRAGMENTS_DOWN][MMA_FRAGMENTS_ACROSS];
    for (uint i = 0; i < MMA_FRAGMENTS_DOWN; ++i) {
        for (uint j = 0; j < MMA_FRAGMENTS_ACROSS; ++j) {
            sums[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0; pass < rows; ++pass) {
        if (!opens[pass]) {
            continue;
        }
        const uint expert = held[pass];
        const bool live = (x_row < rows) && (held[x_row] == expert);
        device const uchar *packed = codes + shape.code_base
            + (ulong)expert * shape.code_stride + (ulong)column * bytes;
        device const uchar *scale = scales + shape.scale_base
            + (ulong)expert * shape.scale_stride + (ulong)column * scale_bytes;

        for (uint b = 0; b < bytes; b += step_bytes) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // The input, as it lies. A row this pass is not for stages zeros,
            // which is what makes a block of disagreeing rows several passes
            // rather than several kernels: zeros multiply to zeros and a sum
            // gains nothing from them.
            device const float *values = x + source + b * CODES_PER_BYTE + x_at;
            for (uint i = 0; i < MMA_FLOATS_A_THREAD; ++i) {
                staged_x[x_row * MMA_STAGED_STRIDE + x_at + i] = live ? values[i] : 0.0f;
            }

            // The weight, decoded. One scale byte covers every code this thread
            // reads, because `Block::holds` refuses a chunk that straddles a
            // group — so this is one load for all of them however many groups
            // the step spans, and the exponent is shifted into place exactly as
            // the reference entries shift it.
            const float by =
                as_type<float>(uint(scale[(b + w_at) / BYTES_PER_GROUP]) << EXPONENT_SHIFT);
            for (uint i = 0; i < MMA_BYTES_A_THREAD; ++i) {
                const uint code = packed[b + w_at + i];
                const uint at = w_col * MMA_STAGED_STRIDE + (w_at + i) * CODES_PER_BYTE;
                staged_w[at] = element(code & CODE_MASK) * by;
                staged_w[at + 1] = element((code >> BITS) & CODE_MASK) * by;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 0; k < MMA_CODES_A_STEP / MMA_FRAGMENT; ++k) {
                simdgroup_float8x8 lhs[MMA_FRAGMENTS_DOWN];
                simdgroup_float8x8 rhs[MMA_FRAGMENTS_ACROSS];
                for (uint i = 0; i < MMA_FRAGMENTS_DOWN; ++i) {
                    mma_read(
                        lhs[i],
                        staged_x
                            + (down * MMA_FRAGMENTS_DOWN + i) * MMA_FRAGMENT * MMA_STAGED_STRIDE
                            + k * MMA_FRAGMENT,
                        lane
                    );
                }
                // Transposed on the way in, because the staged weight is
                // [column, code] where the instruction wants [code, column] —
                // which is the same transpose `out = x @ w^T` names and is free
                // here rather than a second staging.
                for (uint j = 0; j < MMA_FRAGMENTS_ACROSS; ++j) {
                    mma_read_transposed(
                        rhs[j],
                        staged_w
                            + (alongside * MMA_FRAGMENTS_ACROSS + j) * MMA_FRAGMENT
                                * MMA_STAGED_STRIDE
                            + k * MMA_FRAGMENT,
                        lane
                    );
                }
                for (uint i = 0; i < MMA_FRAGMENTS_DOWN; ++i) {
                    for (uint j = 0; j < MMA_FRAGMENTS_ACROSS; ++j) {
                        simdgroup_multiply_accumulate(sums[i][j], lhs[i], rhs[j], sums[i][j]);
                    }
                }
            }
        }
    }

    // Reached before the first fragment is stored, because the store lands on
    // the staged tiles and another simdgroup may still be reading them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < MMA_FRAGMENTS_DOWN; ++i) {
        for (uint j = 0; j < MMA_FRAGMENTS_ACROSS; ++j) {
            simdgroup_store(
                sums[i][j],
                answered
                    + (down * MMA_FRAGMENTS_DOWN + i) * MMA_FRAGMENT * MMA_ANSWER_STRIDE
                    + (alongside * MMA_FRAGMENTS_ACROSS + j) * MMA_FRAGMENT,
                MMA_ANSWER_STRIDE
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Written out here rather than by `simdgroup_store` straight to `out`,
    // because a grouped call's rows are not consecutive in the output: the
    // permutation names a row apiece, where a store to device memory takes one
    // stride for the whole fragment. This is also where the block's ragged edges
    // are refused, which a fragment store could not express either.
    for (uint at = local; at < MMA_ROWS_A_BLOCK * MMA_COLS_A_BLOCK; at += THREADS_PER_GROUP) {
        const uint r = at / MMA_COLS_A_BLOCK;
        const uint c = at % MMA_COLS_A_BLOCK;
        if (r < rows && c < cols) {
            const uint row = first + r;
            out[(__WRITES__) * shape.out_dim + leftmost + c] =
                answered[r * MMA_ANSWER_STRIDE + c];
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
        /// Bits [`Noise::next`] returns, which is the state less the poorly
        /// mixed low eight it shifts off.
        const SPREAD: u32 = u32::BITS - 8;

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

        /// One of the sixteen codes, off the top of the state rather than the
        /// bottom of it.
        ///
        /// **Bit `i` of a power-of-two linear congruential state has period
        /// `2^(i+1)`, so which four bits a code is taken from decides how far a
        /// weight goes before it repeats itself.** Four off the bottom of
        /// [`Noise::next`] are the state's bits 8 to 11 and repeat every 4096 —
        /// which is the width of every reduction in this checkpoint, so a weight
        /// built from them held one row's codes repeated for every row of every
        /// expert, and a kernel that read some other column's codes or some
        /// other expert's would have answered bit for bit what the right one
        /// answers.
        ///
        /// **The top four are the state's own top four**, whose period is the
        /// whole 2^32 — so no two rows of a weight repeat at any size this file
        /// can build. Bits 24 to 27 would not have been enough: their period is
        /// 2^28 codes and the routed bank [`BOUND_SHAPES`] builds at the
        /// longest of [`BLOCKED_LENGTHS`] is 2^31, which would have left experts
        /// 32 apart holding identical codes.
        ///
        /// The scales never had the problem — `% 11` and `% 5` are not powers of
        /// two — which is why the mutation tables here have been measuring
        /// something rather than nothing all along, and measuring it through the
        /// scale alone.
        pub(crate) fn code(&mut self) -> u8 {
            (self.next() >> (Self::SPREAD - BITS as u32)) as u8
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
                codes: (0..out_dim * in_dim).map(|_| noise.code()).collect(),
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
    use std::collections::BTreeSet;
    use std::time::{Duration, Instant};

    use super::testing::{Case, Noise, pack};
    use super::*;
    use inkling_core::fixture::{self, deviation};
    use inkling_core::ops::DenseProjection;
    use inkling_core::quant::dequantize_blocks;
    use inkling_core::weights::PackedRows;
    use inkling_core::workload::{BEST, SWEPT};

    use crate::grouping::ExpertGrouping;
    use crate::kernel::MEMORY_BANDWIDTH;
    use crate::testing::{better, device, drift, entries_dispatched};

    /// What a count of packed codes weighs on the wire: half a byte a code and
    /// one scale byte a group of [`GROUP_SIZE`].
    ///
    /// **One reading of it, because every table here divides by it.** A rate is
    /// this over a duration, and a second spelling of the numerator is a second
    /// place for a factor to go missing — which is the shape of the bug commit
    /// 504e6fe left across six tables.
    fn packed_bytes(codes: usize) -> f64 {
        (codes / CODES_PER_BYTE + codes / GROUP_SIZE) as f64
    }

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

    /// How far the production path may land from the reference one, and from
    /// exact, over the same reduction.
    ///
    /// **Wider than [`TOLERANCE`] because the order is worse-conditioned, and
    /// this is the number that says by how much.** A fragment accumulator
    /// carries the whole reduction as one running sum where the reference splits
    /// it 32 ways across lanes, so its drift grows with the reduction where the
    /// reference's does not: 1.6e-7, 4.8e-7, 7.4e-7, 1.4e-6 and 4.1e-6 at
    /// reductions of 32, 128, 512, 2048 and 4096, against a reference that holds
    /// 9.0e-8 to 1.4e-7 across all five.
    ///
    /// Twice the widest of those, which is what leaves the bound a claim about
    /// this kernel rather than about this sitting — and the 4096 it is set from
    /// is the reduction every projection and every expert in the checkpoint has,
    /// so nothing in the model reaches past it.
    const MMA_TOLERANCE: f32 = 1e-5;

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

    /// **No two rows of a synthetic weight hold the same codes**, which is the
    /// property every mutation table in this file rests on and which it did not
    /// have.
    ///
    /// A [`Noise`] state is a power-of-two linear congruence, so its low bits
    /// have short periods — and four taken off the bottom repeat every 4096,
    /// which is the width of every reduction in this checkpoint. A weight built
    /// that way is one row of codes repeated for every row of every expert, so a
    /// kernel that read the wrong column's codes, or the wrong expert's, would
    /// have answered bit for bit what the right one answers and every arm that
    /// confines the weight would have been priced through its scale alone.
    ///
    /// **Asked at the reduction this checkpoint has and at the two widths either
    /// side of it**, because a period that divides the row length is the failure
    /// and a case fixed at one width could not see it move.
    #[test]
    fn no_two_rows_of_a_synthetic_weight_hold_the_same_codes() {
        for in_dim in [GROUP_SIZE, 2048, 4096, 8192] {
            const ROWS: usize = 8;
            let case = Case::seeded(1, in_dim, ROWS, 1);
            let rows: Vec<&[u8]> = case.codes.chunks_exact(in_dim).collect();
            assert_eq!(rows.len(), ROWS);
            for (at, row) in rows.iter().enumerate() {
                assert!(
                    !rows[..at].contains(row),
                    "at a reduction of {in_dim}, row {at} of a weight repeats an earlier one"
                );
            }
            // And every code is reachable, so that the table the kernel decodes
            // through is exercised across the whole of it rather than over
            // whichever sixteenth a narrower generator happened to reach.
            let mut seen = [false; 1 << BITS];
            for &code in &case.codes {
                seen[code as usize] = true;
            }
            assert!(
                seen.iter().all(|&code| code),
                "{in_dim}: a code never came up"
            );
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

    /// A kernel's own decode given an entry point of its own, so that the
    /// sixteen floats it produces can be read out one code at a time.
    ///
    /// Appended to a source rather than written beside it: what this compiles is
    /// that source's own `element` and not a second copy of it, through the same
    /// compiler and the same options as the dispatch that multiplies.
    const DECODED_ELEMENTS: &str = r#"
kernel void decoded_elements(
    device float *out [[buffer(0)]],
    uint code [[thread_position_in_grid]]
) {
    out[code] = element(code);
}
"#;

    /// What `written`'s decode makes of each of the sixteen codes, as the bits
    /// the device wrote.
    fn each_code_decoded(device: &Device, written: &str) -> Vec<u32> {
        let kernel = device
            .compile(&format!("{written}{DECODED_ELEMENTS}"), "decoded_elements")
            .expect("the probe compiles");
        let mut out = device
            .zeroed::<f32>(ELEMENTS.len())
            .expect("the buffer allocates");
        device
            .run(
                &kernel,
                &[out.arg()],
                Grid::new(ELEMENTS.len(), ELEMENTS.len()),
                size_of::<f32>() * ELEMENTS.len(),
            )
            .expect("the dispatch completes");
        bit_patterns(&out.to_vec())
    }

    fn bit_patterns(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// The kernel's decode and the one that would replace it, against
    /// [`ELEMENTS`], as bit patterns.
    ///
    /// **The two that decode a code have a row here and the one that decodes a
    /// byte does not.** A table of pairs is indexed by both codes at once, so
    /// there is no `element` in it to give an entry point to; what stands in for
    /// this on that arm is
    /// [`every_way_of_decoding_a_byte_multiplies_to_the_same_bits`], which holds
    /// its whole multiply against the shipped one's.
    ///
    /// **Bits rather than floats, for the two readings a comparison of values
    /// would let through.** Code 8 is `-0.0`, which `==` calls equal to code 0's
    /// `+0.0` — MLX carries that sign through the scale multiply, and a decode
    /// that lost it would agree with this table everywhere `assert_eq!` looks.
    ///
    /// **And this is where the arithmetic arm's one constraint was found.** Its
    /// first form laid a code's three bits straight into an f32's own and
    /// multiplied by `2^126`, which is exact for all sixteen and needs no
    /// comparison: the code E2M1 calls subnormal lands on the f32 subnormal
    /// `2^-127`, which is `0.5` under the same factor. **This device flushes f32
    /// subnormals to zero**, so that form answered `0.0` for code 1 and `-0.0`
    /// for code 9 and was bit-exact for the other fourteen — caught here rather
    /// than in the fourth decimal of a projection, and the reason
    /// [`assembled_from_the_bits`] counts up from `HALF_FIELD` instead.
    #[test]
    fn every_way_of_decoding_a_code_answers_the_tables_sixteen_floats_bit_for_bit() {
        let Some(device) = device() else { return };
        let want = bit_patterns(&ELEMENTS);
        assert_ne!(
            want[0],
            want[ELEMENTS.len() / 2],
            "a table whose two zeros were one bit pattern would prove nothing here"
        );

        for (what, written) in [
            ("the shipped gather", source()),
            (
                "the field its bits assemble",
                assembled_from_the_bits(&source()),
            ),
        ] {
            assert_eq!(each_code_decoded(&device, &written), want, "{what}");
        }
    }

    /// The shipped source with the gather replaced by the same sixteen floats
    /// assembled out of each code's own bits.
    ///
    /// **E2M1 is f32's own layout an exponent field apart**, which is why
    /// sixteen values could do without a table at all. A code's three low bits
    /// are an exponent over a mantissa in the order an f32 keeps them, so a
    /// magnitude with a nonzero exponent is `0.5`'s exponent field counted up by
    /// the magnitude itself; the one magnitude below that is subnormal in E2M1,
    /// where the format reads the same bit as half of the smallest step — the
    /// same `HALF_FIELD`, multiplied rather than added. The sign is or'ed into
    /// the bits rather than applied to the value, which is what gives code 8 its
    /// `-0.0` without a branch.
    ///
    /// The constants are written into the arm rather than into the shipped
    /// prelude, where nothing would read them, and every replacement is asserted
    /// by [`crate::testing::instead_of`] — an arm that matched nothing would be
    /// the shipped kernel measured against itself under another name.
    fn assembled_from_the_bits(shipped: &str) -> String {
        let sign_bit = 1u32 << (BITS - 1);
        let half_field = (EXPONENT_BIAS - 1) << 1;
        let carried = crate::testing::instead_of(
            shipped,
            "using namespace metal;",
            &format!(
                "using namespace metal;\n\
                 constant uint SIGN_BIT = {sign_bit};\n\
                 constant uint SIGN_SHIFT = {};\n\
                 constant uint MAGNITUDE_MASK = {};\n\
                 constant uint MAGNITUDE_SHIFT = {};\n\
                 constant uint HALF_FIELD = {half_field};\n\
                 constant uint NORMAL_FLOOR = {NORMAL_FLOOR};",
                u32::BITS - BITS as u32,
                sign_bit - 1,
                EXPONENT_SHIFT - 1,
            ),
        );
        crate::testing::instead_of(
            &carried,
            "    return ELEMENTS[code];",
            "    const uint magnitude = code & MAGNITUDE_MASK;\n    const uint field = magnitude \
             < NORMAL_FLOOR\n        ? magnitude * HALF_FIELD\n        : magnitude + \
             HALF_FIELD;\n    return as_type<float>(((code & SIGN_BIT) << SIGN_SHIFT) | (field << \
             MAGNITUDE_SHIFT));",
        )
    }

    /// What an f32's exponent field is biased by, which is what a code's own
    /// exponent counts up from.
    const EXPONENT_BIAS: u32 = f32::MAX_EXP as u32 - 1;

    /// The first magnitude whose exponent is not zero, which is where the
    /// counting turns from a multiply into an add.
    const NORMAL_FLOOR: u32 = 2;

    /// Each decode's whole multiply against each other decode's, over the widths
    /// and the spread of codes and scales the checkpoint holds.
    ///
    /// **This is what licenses shipping whichever of them is fastest.**
    /// Bit-safety here is by construction rather than by tolerance: each decode
    /// produces the same sixteen floats, every product and every partial sum
    /// after it takes the same operands in the same order, and nothing else in
    /// either entry moves. So the claim is equality of bit patterns and not a
    /// bound — a deviation of any size at all would mean the values were not the
    /// same values.
    ///
    /// Both entries, because the two decode in different places: the tiled one
    /// decodes four columns of a byte into registers before it multiplies, and
    /// `weight_dot` decodes one code at a time inside its own accumulate.
    #[test]
    fn every_way_of_decoding_a_byte_multiplies_to_the_same_bits() {
        let Some(device) = device() else { return };
        let case = Case::noisy(IN_DIM, OUT_DIM, 3);
        let through = |matmul: &PackedMatmul| {
            case.upload(&device, matmul)
                .multiply(&case.x)
                .expect("the dispatch completes")
        };

        let shipped = matmul(&device);
        let (want, tiled) = (through(&shipped), a_tiled_call_answers(&device, &shipped));
        for (what, written) in each_way_of_decoding_a_byte() {
            let arm = PackedMatmul::from_source(&device, &written).expect("the arm compiles");
            assert_eq!(
                bit_patterns(&through(&arm)),
                bit_patterns(&want),
                "{what}: a projection"
            );
            assert_eq!(
                bit_patterns(&a_tiled_call_answers(&device, &arm)),
                bit_patterns(&tiled),
                "{what}: a tile"
            );
        }
    }

    /// **Both numerics answer the same call, and the flag reaches the kernel
    /// that ran it.**
    ///
    /// The two entries this flag selects between are the tiled ones, so the case
    /// is taken through both of the shapes that reach them — an untiled
    /// projection, which no flag touches, and a tiled call over a bank, which is
    /// where a kernel behind the flag would run.
    ///
    /// **The two arms are bounded differently and that is the finding rather
    /// than an oversight.** The projection goes through [`Layout::Each`], which
    /// no flag reaches, so the two paths run the same kernel and the deviation
    /// there is a zero this asserts rather than tolerates. The tile goes through
    /// the entry the flag selects, where equality of bits is exactly what cannot
    /// be asserted: a production kernel accumulates in an order the instruction
    /// picks. Both are summing the same exact products — nothing either path
    /// forms is rounded — so the two may only differ by what summation order is
    /// worth, which is [`MMA_TOLERANCE`].
    #[test]
    fn a_call_under_either_numerics_answers_what_the_other_answers() {
        let Some(device) = device() else { return };
        let compiled =
            |numerics| PackedMatmul::under(&device, numerics).expect("the packed matmul compiles");
        let (reference, production) = (
            compiled(Numerics::Reference),
            compiled(Numerics::Production),
        );
        assert_eq!(reference.numerics(), Numerics::Reference);
        assert_eq!(production.numerics(), Numerics::Production);
        assert_eq!(
            matmul(&device).numerics(),
            Numerics::Reference,
            "the constructor nobody passes a word to is the reference one"
        );

        let case = Case::noisy(IN_DIM, OUT_DIM, 3);
        let untiled = |matmul: &PackedMatmul| {
            case.upload(&device, matmul)
                .multiply(&case.x)
                .expect("the dispatch completes")
        };
        assert_eq!(
            bit_patterns(&untiled(&production)),
            bit_patterns(&untiled(&reference)),
            "an untiled call is not the same kernel under either word"
        );

        let (was, is) = (
            a_tiled_call_answers(&device, &reference),
            a_tiled_call_answers(&device, &production),
        );
        let deviation = deviation(&is, &was);
        eprintln!("a tile: the two numerics deviate {deviation:e}");
        assert!(deviation > 0.0, "a tile is not the same kernel either way");
        assert!(deviation <= MMA_TOLERANCE, "deviation {deviation:e}");
    }

    /// **What cutting a walk costs the answer**, which is the question the
    /// deviation above cannot settle: two orders differing says nothing about
    /// which of them is further from the sum.
    ///
    /// Against an f64 accumulation of the same products, at the reduction the
    /// checkpoint has and at a short one.
    ///
    /// **This is the one entry behind the flag that holds the reference's own
    /// bound**, and that is the finding rather than the tolerance being generous.
    /// A cut is a *shorter* chain than the walk it cuts — `n` partials, each a
    /// lane-strided walk of `1/n` of the row, added at the end — where the block
    /// carries the whole reduction as one running sum and needs
    /// [`MMA_TOLERANCE`] for it. So what is asserted here is [`TOLERANCE`]: the
    /// cut walk is f32 noise at every reduction, not a chain that grows with one.
    ///
    /// **Which of the two is nearer is a wash at a short reduction and the cut
    /// at a long one**, which is what the reported pair is for rather than a
    /// second assertion. At 1024 codes each cut holds one chunk and the tail
    /// that adds the partials is as long as the walk that fed it; at 4096 the
    /// cut is 9.2e-8 against the uncut walk's 1.3e-7.
    #[test]
    fn a_cut_walk_is_f32_noise_at_every_reduction() {
        let Some(device) = device() else { return };
        let reference = matmul(&device);
        // Four cuts, which for the checkpoint's 4096-code row is the most that
        // leaves each of them the four chunks `CHUNKS_IN_FLIGHT` holds — and for
        // the 1024-code row is one chunk apiece, which is the other end of it.
        let cut = PackedMatmul::cut_at(&device, 4).expect("four cuts compile");

        for in_dim in [IN_DIM, MMA_CODES_A_STEP * 32] {
            let case = Case::noisy(in_dim, OUT_DIM, 1);
            let exact = case.exactly();
            let through = |matmul: &PackedMatmul| {
                drift(
                    &case
                        .upload(&device, matmul)
                        .multiply(&case.x)
                        .expect("the dispatch completes"),
                    &exact,
                )
            };
            let (cut, whole) = (through(&cut), through(&reference));
            eprintln!("a reduction of {in_dim}: cut {cut:e} against the uncut walk's {whole:e}");
            assert!(
                cut <= f64::from(TOLERANCE),
                "a cut walk drifts {cut:e} where the walk it cuts drifts {whole:e}"
            );
        }
    }

    /// **A reference run compiles the string it compiled before the flag
    /// existed**, which is the cheapest reading there is of "nothing changes for
    /// a caller who does not ask".
    ///
    /// Stated on the source rather than on the pipeline because that is where it
    /// is decided: an entry point a source does not contain is one
    /// `newLibraryWithSource:` never sees, so there is nothing to skip
    /// dispatching later.
    #[test]
    fn the_reference_source_does_not_carry_the_production_entries() {
        assert_eq!(
            source_blocked(Numerics::Reference, Block::SHIPPED),
            source()
        );
        // Held against the list the flag itself keeps rather than against a
        // second copy of it, so that an entry put behind the flag and left out
        // here is a case that fails rather than a check that shrank.
        for entry in PackedMatmul::BEHIND_THE_FLAG.iter().copied() {
            assert!(
                !source().contains(entry),
                "{entry} is in the reference source"
            );
            assert!(
                source_blocked(Numerics::Production, Block::SHIPPED).contains(entry),
                "{entry} is not in the production source"
            );
        }
        // And what production adds is added rather than substituted, so no byte
        // of the reference source moved under it.
        assert!(source_blocked(Numerics::Production, Block::SHIPPED).starts_with(&source()));
    }

    /// **The bytes a split walk reads are the bytes the walk it cuts reads**,
    /// which is the one thing about the cut that could be wrong without being
    /// visibly wrong: a partition that dropped a chunk would answer a little
    /// low, and one that read a chunk twice a little high, and both are inside
    /// the tolerance a differential run holds the two orders to.
    ///
    /// So it is asserted as a set rather than measured as a deviation. A cut is
    /// the lane stride widened — simdgroup `cut` takes lanes `cut * width` to
    /// `(cut + 1) * width` of one walk `splits * width` lanes wide — so the union
    /// over the cuts has to be the whole row, once each, at every count.
    #[test]
    fn the_cuts_of_a_split_walk_read_each_byte_of_the_row_exactly_once() {
        /// Which bytes of a row of `bytes` the lane at `lane` of a walk `width`
        /// lanes wide reads, which is `weight_dot`'s two loops as a set: both
        /// step by the same stride and the first is only the second four chunks
        /// at a time.
        fn visits(bytes: usize, lane: usize, width: usize) -> Vec<usize> {
            let step = width * BYTES_PER_LANE;
            (lane * BYTES_PER_LANE..bytes)
                .step_by(step)
                .flat_map(|at| at..at + BYTES_PER_LANE)
                .collect()
        }

        // The checkpoint's own reduction, one that is not a whole number of
        // held chunks, and one short enough that the clamp is what answers.
        for in_dim in [IN_DIM, 4096 + 2 * GROUP_SIZE, GROUP_SIZE] {
            let bytes = in_dim / CODES_PER_BYTE;
            let width = NARROWEST_SIMD;
            let whole: Vec<usize> = (0..width)
                .flat_map(|lane| visits(bytes, lane, width))
                .collect();
            assert_eq!(
                whole.iter().copied().collect::<BTreeSet<_>>().len(),
                bytes,
                "the unsplit walk does not read {bytes} bytes once each"
            );

            for splits in 1..=MOST_SPLITS {
                let cut: Vec<usize> = (0..splits * width)
                    .flat_map(|lane| visits(bytes, lane, splits * width))
                    .collect();
                let (mut sorted, mut against) = (cut.clone(), whole.clone());
                sorted.sort_unstable();
                against.sort_unstable();
                assert_eq!(
                    sorted, against,
                    "{splits} cuts of a {bytes}-byte row read other bytes"
                );
            }
        }
    }

    /// **The production entries against the reference ones over a shape that
    /// leaves both of a block's extents ragged**, which is the one thing about
    /// this kernel that a shape filling its blocks could not catch: a block that
    /// ran its clamped rows or wrote its clamped columns would answer wrongly
    /// where nothing else here would notice.
    ///
    /// **The bound is [`MMA_TOLERANCE`] and not [`TOLERANCE`], and the reason is
    /// a finding rather than a slackening.** The production path is the
    /// worse-conditioned of the two orders: a fragment accumulator carries the
    /// whole reduction as one running sum where the reference splits it 32 ways
    /// across lanes. Measured across reduction lengths against an f64
    /// accumulation of the same products, the reference holds 9.0e-8, 9.5e-8,
    /// 9.5e-8, 9.6e-8 and 1.4e-7 at reductions of 32, 128, 512, 2048 and 4096,
    /// where this path reads 1.6e-7, 4.8e-7, 7.4e-7, 1.4e-6 and 4.1e-6.
    ///
    /// **That the drift is f32 noise at a reduction of 32 is what says the
    /// arithmetic is right and only the chain is long.** Four accumulate steps
    /// land within an ulp or two of exact; the growth after that is the chain
    /// and nothing else, and a kernel that had the transpose or the staging
    /// wrong would be decades out at every length rather than exact at the
    /// short one.
    ///
    /// The assertion with teeth is therefore the *upper* bound on how much worse
    /// it is, which is what a tolerance alone would hide: a drift that grew past
    /// [`MMA_TOLERANCE`] would be a mistake and not a chain.
    #[test]
    fn a_block_answers_the_reference_tile_where_neither_extent_divides_it() {
        let Some(device) = device() else { return };
        // Two whole blocks of rows and thirteen left over, against four whole
        // spans of columns and one left over.
        const ROWS: usize = 77;
        assert_ne!(ROWS % MMA_ROWS_A_BLOCK, 0, "rows that filled their blocks");
        assert_ne!(
            OUT_DIM % MMA_COLS_A_BLOCK,
            0,
            "columns that filled their blocks"
        );

        let case = Case::noisy(IN_DIM, OUT_DIM, ROWS);
        let through = |matmul: &PackedMatmul| {
            case.upload(&device, matmul)
                .multiply(&case.x)
                .expect("the dispatch completes")
        };
        let was = through(&matmul(&device));
        let is = through(&PackedMatmul::under(&device, Numerics::Production).expect("compiles"));

        let deviation = deviation(&is, &was);
        let exact = case.exactly();
        let (mine, theirs) = (drift(&is, &exact), drift(&was, &exact));
        eprintln!(
            "a ragged block: deviation {deviation:e}, drift from exact {mine:e} against the \
             reference's {theirs:e}"
        );
        assert!(
            deviation > 0.0,
            "an exact match would mean the two are not summing independently"
        );
        assert!(deviation <= MMA_TOLERANCE, "deviation {deviation:e}");
        assert!(
            f64::from(MMA_TOLERANCE) >= mine,
            "drift {mine:e} against the reference's {theirs:e}"
        );

        // The short reduction, where four accumulate steps put this path back
        // inside the reference's own bound — which is what says the growth above
        // is the chain rather than the arithmetic.
        let short = Case::noisy(MMA_CODES_A_STEP, OUT_DIM, ROWS);
        let brief = |matmul: &PackedMatmul| {
            short
                .upload(&device, matmul)
                .multiply(&short.x)
                .expect("the dispatch completes")
        };
        let close = drift(
            &brief(&PackedMatmul::under(&device, Numerics::Production).expect("compiles")),
            &short.exactly(),
        );
        eprintln!("a reduction of {MMA_CODES_A_STEP}: drift from exact {close:e}");
        assert!(close <= f64::from(TOLERANCE), "drift {close:e}");
    }

    /// The block shapes this file sweeps, beside the one it ships.
    ///
    /// **The threadgroup is what the sweep is for and the taller block is what
    /// says the height moves too.** mlx-vlm runs its steel GEMM at 64 to 128
    /// threads where this ships 256, and nothing here had ever tried another
    /// width — because the width was a host-side constant as well as a source
    /// one, and an entry compiled at one and dispatched at another answers
    /// wrongly rather than slowly. [`Block`] is what closed that.
    ///
    /// Each is the shipped 32×64 block over a different threadgroup, except the
    /// last, which is twice as tall — the shape a fragment-reuse argument wants
    /// and whose floor `what_a_block_of_query_rows_is_worth_at_each_height`
    /// prices on the kernel beside this one.
    const SWEPT_BLOCKS: [Block; 5] = [
        Block {
            threads: 64,
            simds_down: 1,
            simds_across: 2,
            ..Block::SHIPPED
        },
        Block {
            threads: 128,
            simds_down: 2,
            simds_across: 2,
            ..Block::SHIPPED
        },
        Block::SHIPPED,
        Block {
            threads: 512,
            simds_down: 2,
            simds_across: 8,
            ..Block::SHIPPED
        },
        Block {
            rows: 2 * MMA_ROWS_A_BLOCK,
            simds_down: 4,
            simds_across: 2,
            ..Block::SHIPPED
        },
    ];

    /// The steps this file sweeps, beside the one it ships.
    ///
    /// **Two groups and not four, and the arithmetic is what stops at two.** The
    /// staging is `(rows + cols)` rows of the step and its padding, so a step of
    /// 32 declares 13.5 KiB, 64 declares 25.5 and 128 would declare 49.5 —
    /// past the 32 KiB a threadgroup gets. So the whole of what a wider step can
    /// be asked here is one doubling, and `a_block_of` is what would print the
    /// refusal if a third were added.
    const SWEPT_STEPS: [Block; 2] = [
        Block::SHIPPED,
        Block {
            step: 2 * MMA_CODES_A_STEP,
            ..Block::SHIPPED
        },
    ];

    /// The two ways of reading a fragment, at the shipped shape.
    ///
    /// **Both named rather than one of them left to [`Block::SHIPPED`]**, which
    /// is what keeps this a sweep whichever way [`MMA_FRAGMENTS`] is set — an arm
    /// written as "the shipped block, with the other read" collapses into the
    /// shipped block itself the moment the other read becomes the shipped one,
    /// and a table of one arm printed twice is what it collapses into.
    /// [`distinct`] is that hazard stated once for all three sweeps.
    const SWEPT_FRAGMENTS: [Block; 2] = [
        Block {
            fragments: Fragments::Loaded,
            ..Block::SHIPPED
        },
        Block {
            fragments: Fragments::PerLane,
            ..Block::SHIPPED
        },
    ];

    /// **Every arm of a sweep has to be a different block**, or the table reads
    /// as a comparison and is one shape measured against itself.
    ///
    /// The way that happens is not a typo: an arm written as the shipped block
    /// with one field overridden stops being an arm when that field's shipped
    /// value moves to what the arm overrides it to. Nothing about such a table
    /// looks wrong — the rows are the same shape and read within the noise of
    /// each other, which is exactly what an arm that made no difference would
    /// print.
    fn distinct(swept: &[Block]) {
        for (at, block) in swept.iter().enumerate() {
            assert!(
                !swept[..at].contains(block),
                "a sweep offers {block:?} twice, so one of its rows is the other measured again"
            );
        }
    }

    /// **A block cut to another shape answers what the shipped one answers**,
    /// which is the whole of what makes the sweep beside it a measurement rather
    /// than a comparison of two different computations.
    ///
    /// **Bit for bit, and that is the assertion rather than a tolerance.**
    /// Changing the threadgroup changes which simdgroup owns an output element
    /// and which thread staged the value it reads; it does not change the order
    /// the element is accumulated in, which is the instruction's over `k` inside
    /// a step and the step loop outside it. So a shape that came back merely
    /// *close* would mean something moved that this file does not think moves.
    ///
    /// **A wider step is the one axis here where that has to be argued rather
    /// than observed**, since it moves what a step is and so how many of them a
    /// reduction is cut into. It does not move the order: a fragment accumulator
    /// is one running sum over the whole reduction and the `k` loop visits the
    /// staged codes in the order they lie, so a step of 64 runs the same eight
    /// accumulates the two steps of 32 ran, in the same sequence, into the same
    /// register. What it changes is where the barriers fall between them.
    ///
    /// A shape whose threadgroup this device will not dispatch is reported and
    /// skipped rather than failed — the width is swept past what a register
    /// allocation may allow on purpose, and a pipeline that says so is an answer
    /// and not a fault.
    #[test]
    fn a_block_cut_to_another_shape_answers_what_the_shipped_one_answers() {
        let Some(device) = device() else { return };
        let want = a_tiled_call_answers(
            &device,
            &PackedMatmul::under(&device, Numerics::Production).expect("the block compiles"),
        );

        let swept: Vec<Block> = SWEPT_BLOCKS
            .into_iter()
            .chain(SWEPT_STEPS)
            .chain(SWEPT_FRAGMENTS)
            .collect();
        let mut ran = 0;
        for block in swept.iter().copied() {
            let Some(matmul) = a_block_of(&device, block) else {
                continue;
            };
            assert_eq!(
                a_tiled_call_answers(&device, &matmul),
                want,
                "a block of {block:?} answered other bits than the shipped one"
            );
            ran += usize::from(block != Block::SHIPPED);
        }
        // **A case that skipped every shape would pass by comparing the shipped
        // block to itself**, which is the failure mode `a_block_of` opens by
        // treating a refusal as a skip. This part runs every shape in
        // [`SWEPT_BLOCKS`] and [`SWEPT_STEPS`], so the one shape of slack is for
        // a part that is not this one rather than for a shape this one is known
        // to decline — and it is counted off the arms rather than written down,
        // so that an arm added to either list raises the bar with it.
        let offered = swept
            .iter()
            .filter(|&&block| block != Block::SHIPPED)
            .count();
        assert!(
            ran + 1 >= offered,
            "only {ran} shapes of the {offered} other than the shipped one ran, so this case is \
             close to comparing the shipped block against itself"
        );
    }

    /// One swept shape compiled, or `None` with a printed reason where this part
    /// will not run it.
    ///
    /// **The two refusals are not the same and only one of them is a skip.** A
    /// source that will not compile is a mistake in the prelude this file writes
    /// and it panics; a source that compiles into a pipeline this part refuses —
    /// too much threadgroup memory, too many threads — is the sweep finding its
    /// own edge, and printing where that edge is *is* the measurement. A
    /// threadgroup gets at most 32768 bytes of memory and at most the threads
    /// the compiled function's register allocation leaves room for, and a shape
    /// asking past either of those is the answer to what a sweep of it would
    /// have said.
    ///
    /// The width is asked of the pipeline rather than of the device because it
    /// is a property of the compiled function — a kernel's register allocation
    /// is what decides how many of its threads fit in a threadgroup.
    fn a_block_of(device: &Device, block: Block) -> Option<PackedMatmul> {
        let source = source_blocked(Numerics::Production, block);
        // **Both entries, because both are compiled and one asserts.** They are
        // the same block over the same body and differ only in how a row finds
        // its input, but their register allocations need not agree — and
        // `PackedMatmul::blocked` compiles the pair, so a shape that cleared the
        // tiled probe and failed the grouped one would panic where this
        // documents a skip.
        for entry in [MMA_TILED_ENTRY, MMA_GROUPED_ENTRY] {
            let compiled = match device.compile(&source, entry) {
                Ok(compiled) => compiled,
                Err(MetalError::Pipeline { diagnostic, .. }) => {
                    eprintln!("  {block:?} is refused at {entry}: {diagnostic}");
                    return None;
                }
                Err(err) => {
                    panic!("the source this file wrote for {block:?} does not compile: {err}")
                }
            };
            if compiled.max_threads_per_group() < block.threads {
                eprintln!(
                    "  {block:?} is refused at {entry}: this part dispatches at most {} threads a \
                     threadgroup",
                    compiled.max_threads_per_group()
                );
                return None;
            }
        }
        Some(PackedMatmul::blocked(device, Numerics::Production, block).expect("the arm compiles"))
    }

    /// **A block whose rows name two experts answers each row its own expert's
    /// weight**, which is the pass loop and is the one thing in this kernel with
    /// no counterpart in the reference tile.
    ///
    /// A staged weight block is one expert's, so a block straddling a run cannot
    /// serve both from one staging; it runs the whole walk once per expert its
    /// rows name and zeroes the rows the pass is not for. **What would go wrong
    /// without it is not a crash**: every row of the block would be multiplied
    /// against whichever expert the pass happened to stage, which is a plausible
    /// answer of the right magnitude — so the control below checks that the two
    /// experts answer differently before the comparison checks that each row got
    /// its own.
    #[test]
    fn a_block_whose_rows_name_two_experts_answers_each_row_the_weight_it_named() {
        let Some(device) = device() else { return };
        const EXPERTS: usize = 2;
        const SOURCES: usize = 50;
        assert_ne!(
            SOURCES % MMA_ROWS_A_BLOCK,
            0,
            "a boundary falling on a block's own edge would never straddle one"
        );

        let case = Case::noisy(IN_DIM, EXPERTS * OUT_DIM, SOURCES);
        let chosen: Vec<u32> = (0..EXPERTS * SOURCES)
            .map(|row| (row / SOURCES) as u32)
            .collect();
        let through = |matmul: &PackedMatmul| {
            let bank = PackedBank::upload(
                &device,
                matmul,
                EXPERTS,
                IN_DIM,
                OUT_DIM,
                &case.packed(),
                &case.scales,
            )
            .expect("the bank's shapes pair");
            let mut batch = device.batch().expect("a command buffer opens");
            let mut input = device.buffer(&case.x).expect("the input uploads");
            let got = bank
                .encode_repeating(&mut batch, &chosen, &mut input)
                .expect("the dispatch encodes");
            batch.wait().expect("the dispatch completes");
            got.take()
        };
        let was = through(&matmul(&device));
        let is = through(&PackedMatmul::under(&device, Numerics::Production).expect("compiles"));

        // The control: the same input row through the two experts is two
        // different answers, so a block that staged one weight for all of its
        // rows would be caught by the comparison below rather than agreeing.
        fn row(rows: &[f32], at: usize) -> &[f32] {
            &rows[at * OUT_DIM..][..OUT_DIM]
        }
        assert!(
            deviation(row(&was, 0), row(&was, SOURCES)) > TOLERANCE,
            "the two experts answer the same, so nothing here separates them"
        );

        for (at, expert) in chosen.iter().enumerate() {
            let deviation = deviation(row(&is, at), row(&was, at));
            assert!(
                deviation <= MMA_TOLERANCE,
                "row {at} of expert {expert}: deviation {deviation:e}"
            );
        }
    }

    /// **The grouped production entry against the grouped reference one**, over
    /// a routing whose runs are the routing's rather than a tile's.
    ///
    /// The rows arrive through one permutation and leave through another, and a
    /// block writes its answer a row at a time for exactly that reason — a
    /// `simdgroup_store` straight to device memory takes one stride for a whole
    /// fragment, and the rows of a scattered call are not at one. So this is the
    /// case that would catch a block writing its answer where the *grouping* put
    /// the row rather than where the router did.
    #[test]
    fn a_grouped_block_answers_the_grouped_tile_through_both_permutations() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const EXPERTS: usize = 3;
        const TOKENS: usize = 64;
        const SLOTS: usize = 2;
        const ROWS: usize = TOKENS * SLOTS;
        // A call under the line stays on the reference tile and would prove
        // nothing here, so the shape is held above it where it is written.
        const _: () = assert!(ROWS >= EXPERTS * Block::SHIPPED.runs_an_expert());

        let case = Case::noisy(IN_DIM, EXPERTS * OUT_DIM, ROWS);
        let chosen: Vec<u32> = (0..TOKENS)
            .flat_map(|token| (0..SLOTS).map(move |slot| ((token + slot * 2) % EXPERTS) as u32))
            .collect();
        assert!(
            !tiles(&chosen, ROWS_A_TILE),
            "a routing a tile could already reach would prove nothing"
        );

        let through = |matmul: &PackedMatmul, through: Through| {
            let bank = PackedBank::upload(
                &device,
                matmul,
                EXPERTS,
                IN_DIM,
                OUT_DIM,
                &case.packed(),
                &case.scales,
            )
            .expect("the bank's shapes pair");
            let mut selection = device.buffer(&chosen).expect("the selection uploads");
            let mut x = device
                .buffer(&case.x[..TOKENS * IN_DIM])
                .expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let mut sorted = grouping
                .encode(&mut batch, &mut selection, EXPERTS)
                .expect("the grouping encodes");
            let pending = bank
                .encode_grouped(&mut batch, &mut sorted, &mut x, SLOTS, through)
                .expect("the dispatch encodes");
            batch.wait().expect("the dispatch completes");
            pending.take()
        };

        let production = PackedMatmul::under(&device, Numerics::Production).expect("compiles");
        for end in [Through::Gathered, Through::Scattered] {
            let was = through(&matmul(&device), end);
            let is = through(&production, end);
            let deviation = deviation(&is, &was);
            eprintln!("a grouped block, {end:?}: deviation {deviation:e}");
            assert!(deviation > 0.0, "{end:?}: an exact match is not two orders");
            assert!(
                deviation <= MMA_TOLERANCE,
                "{end:?}: deviation {deviation:e}"
            );
        }
    }

    /// **A block is the simdgroups this device gives a threadgroup**, and the
    /// layout above is written from a width this side assumed.
    ///
    /// Every fragment origin in the kernel is `simdgroup_index_in_threadgroup`
    /// divided and taken modulo against [`MMA_SIMDS_ACROSS`], so a device whose
    /// simdgroup were wider would put fewer simdgroups in the threadgroup than
    /// the block was cut into and leave whole fragments computed by nobody —
    /// a wrong answer rather than a failure. Read off the compiled pipeline,
    /// which is the only place this side can see it.
    #[test]
    fn a_block_is_the_simdgroups_this_device_gives_a_threadgroup() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::under(&device, Numerics::Production).expect("compiles");
        let mma = matmul
            .mma
            .as_ref()
            .expect("the production entries compiled");
        for kernel in [&mma.tiled, &mma.grouped] {
            assert_eq!(kernel.simd_width(), NARROWEST_SIMD);
            assert!(kernel.max_threads_per_group() >= THREADS_PER_GROUP);
        }
    }

    /// One 8×8 tile of the staging read both ways, so that the two can be held
    /// against each other lane for lane.
    ///
    /// **Both reads out of one kernel and one tile**, because what is asked is
    /// whether the two forms name the same element and not whether two dispatches
    /// agree about a number.
    ///
    /// `__T__` is what the staging holds, and the entry is emitted once per width
    /// the block is ever compiled at — see [`lane_probe`]. The sink stays `float`
    /// at both, since the values a probe puts through it are whole numbers under
    /// 64 and exact in either.
    const LANE_PROBE: &str = r#"
kernel void mma_lane_probe___T__(
    device const float *from [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint local [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup __T__ tile[MMA_FRAGMENT * MMA_STAGED_STRIDE];
    for (uint at = local; at < MMA_FRAGMENT * MMA_STAGED_STRIDE; at += width) {
        tile[at] = (__T__)from[at];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup___T__8x8 plain, across;
    mma_read(plain, tile, lane);
    mma_read_transposed(across, tile, lane);
    thread vec<__T__, 2> &held = (thread vec<__T__, 2> &)(plain.thread_elements());
    thread vec<__T__, 2> &turned = (thread vec<__T__, 2> &)(across.thread_elements());
    out[lane * 4 + 0] = (float)held[0];
    out[lane * 4 + 1] = (float)held[1];
    out[lane * 4 + 2] = (float)turned[0];
    out[lane * 4 + 3] = (float)turned[1];
}
"#;

    /// [`LANE_PROBE`] at both widths a fragment of this kernel is ever read at.
    ///
    /// **The narrow one is here because [`through_sixteen_bit_operands`] puts the
    /// same two calls through `simdgroup_half8x8`**, and the map they compute
    /// takes no notice of what the staging holds. That independence is the claim,
    /// and a claim about a lane layout is what this whole case exists not to
    /// inherit.
    fn lane_probe() -> String {
        LANE_WIDTHS
            .iter()
            .map(|holds| crate::testing::instead_of(LANE_PROBE, "__T__", holds))
            .collect()
    }

    /// What the staging holds, at each width this kernel's fragments are ever
    /// read at.
    const LANE_WIDTHS: [&str; 2] = ["float", "half"];

    /// **The lane-to-coordinate map a per-lane read is written from is this
    /// part's own**, which is the one thing about [`Fragments::PerLane`] that
    /// could be wrong without anything failing.
    ///
    /// The three lines it computes are mlx's `BaseMMAFrag<float, 8, 8>` header,
    /// and a header is a claim about which two elements of an 8×8 fragment a lane
    /// holds — hardware nobody here has a document for. **A map that named
    /// somebody else's two elements would not crash and would not read zero**: it
    /// would read two staged floats and multiply them, and every call would come
    /// back a plausible number of the right magnitude.
    ///
    /// So the same tile goes through both forms and the two are held equal
    /// element for element, transposed as well as plain, **at both widths a
    /// fragment of this kernel is ever read at** — the map takes no notice of
    /// what the staging holds, and `through_sixteen_bit_operands` is where that
    /// independence is relied on.
    ///
    /// **And the tile's own coverage is checked beside the agreement, because
    /// the two forms share the tile if not the map.** They read one threadgroup
    /// array filled by one loop, so a probe that filled part of it would have
    /// both forms agreeing about the same wrong thing — which is what a fill
    /// striding by the wrong width did here, and what the 64 distinct values
    /// `1..=64` against a padding none of them takes is what caught.
    #[test]
    fn the_lane_map_a_fragment_is_read_through_is_this_parts_own() {
        let Some(device) = device() else { return };

        let stride = Block::SHIPPED.staged_stride();
        let mut staged: Vec<f32> = vec![-1.0; MMA_FRAGMENT * stride];
        for down in 0..MMA_FRAGMENT {
            for across in 0..MMA_FRAGMENT {
                staged[down * stride + across] = (down * MMA_FRAGMENT + across + 1) as f32;
            }
        }

        let read = |fragments: Fragments, holds: &str| {
            let block = Block {
                fragments,
                ..Block::SHIPPED
            };
            let source = source_blocked(Numerics::Production, block) + &lane_probe();
            let kernel = device
                .compile(&source, &format!("mma_lane_probe_{holds}"))
                .expect("the probe compiles");
            let mut from = device.buffer(&staged).expect("the tile uploads");
            let mut out = device
                .zeroed::<f32>(4 * NARROWEST_SIMD)
                .expect("the sink allocates");
            let mut batch = device.batch().expect("a command buffer opens");
            batch
                .add(
                    &kernel,
                    &[from.arg(), out.arg()],
                    Grid::new(NARROWEST_SIMD, NARROWEST_SIMD),
                    0,
                )
                .expect("the probe encodes");
            batch.wait().expect("the probe completes");
            out.to_vec()
        };

        let want: Vec<f32> = (1..=(MMA_FRAGMENT * MMA_FRAGMENT) as u32)
            .map(|value| value as f32)
            .collect();
        for holds in LANE_WIDTHS {
            let loaded = read(Fragments::Loaded, holds);
            let per_lane = read(Fragments::PerLane, holds);
            assert_eq!(
                per_lane, loaded,
                "over {holds}, the per-lane map named other elements than `simdgroup_load` reads \
                 on this part"
            );

            for (what, at) in [("plain", 0), ("transposed", 2)] {
                let mut covered: Vec<f32> = (0..NARROWEST_SIMD)
                    .flat_map(|lane| [per_lane[lane * 4 + at], per_lane[lane * 4 + at + 1]])
                    .collect();
                covered.sort_by(f32::total_cmp);
                assert_eq!(
                    covered, want,
                    "over {holds}, the {what} read does not cover the tile's 64 elements exactly \
                     once"
                );
            }
        }
    }

    /// **Neither a decode step nor a speculative round reaches the production
    /// entries**, which is the same predicate `tiles` states for the reference
    /// tile and is checked the same way: on the shapes rather than on a run.
    ///
    /// **The speculative rows are here because they were measured before they
    /// were refused.** A verify block is the depth plus one rows through every
    /// projection, which at a depth of three clears [`tiles`]'s four-row bar and
    /// would land on a block eight times too tall — and read 37.33 ms a token
    /// against the reference's 17.08. A block computes its full height whether
    /// the call has the rows or not, so the shapes below are the ones this flag
    /// was never meant to reach and the ones a future edit to [`blocks`] would
    /// most easily let back in.
    #[test]
    fn neither_a_decode_step_nor_a_speculative_round_reaches_a_block() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::under(&device, Numerics::Production).expect("compiles");
        let grouped = Layout::Grouped {
            order: Arg::Inline(&[]),
            through: Through::Gathered,
        };
        // A single-row projection, a two-row shared bank naming two experts, and
        // a six-row routed bank naming six of 256.
        assert!(matmul.blocks(&Layout::Each, 1, 1).is_none());
        assert!(matmul.blocks(&Layout::Each, 2, 2).is_none());
        assert!(matmul.blocks(&Layout::Each, 6, 256).is_none());
        assert!(matmul.blocks(&grouped, 6, 256).is_none());

        // A verify block at every depth the sweep runs, through the two shapes
        // that reached [`Layout::Tiled`] and so were the ones that regressed:
        // the projections at a row a token, and the shared bank at two.
        //
        // **A round's routed bank is not among them and never could be.** Its
        // rows sort into runs of one, so `ExpertBanks::groups` is false for
        // every shape a round has and the call goes through
        // [`PackedBank::encode_picked`], which is [`Layout::Each`] by
        // construction. The grouped rows below are here to hold [`blocks`]
        // itself rather than because a round ever dispatched one.
        for depth in 0..=SWEPT {
            let rows = depth + 1;
            assert!(
                matmul.blocks(&Layout::Tiled, rows, 1).is_none(),
                "a projection verifying {rows} rows"
            );
            assert!(
                matmul.blocks(&Layout::Tiled, rows * 2, 2).is_none(),
                "a shared bank verifying {rows} rows"
            );
            assert!(
                matmul.blocks(&grouped, rows * 6, 256).is_none(),
                "a routed bank of {rows} tokens, which no round dispatches grouped"
            );
        }
        // And the widest block the eight heads can propose, which is nine
        // tokens.
        assert!(matmul.blocks(&Layout::Tiled, 9, 1).is_none());
        assert!(matmul.blocks(&grouped, 9 * 6, 256).is_none());

        // The line a tiled call crosses, from either side, and the two prefill
        // lengths the sweep behind `blocks` reads either side of — so a line
        // moved without re-measuring fails here rather than quietly.
        let shortest = PackedMatmul::SHORTEST_BLOCKED_CALL;
        assert!(matmul.blocks(&Layout::Tiled, shortest - 1, 1).is_none());
        assert!(matmul.blocks(&Layout::Tiled, shortest, 1).is_some());
        assert!(matmul.blocks(&Layout::Tiled, 48, 1).is_none());
        assert!(matmul.blocks(&Layout::Tiled, 64, 1).is_some());

        // The line a routed bank crosses, from either side. Six rows a token
        // against 256 experts puts it at 1366 tokens.
        let tokens = |tokens: usize| matmul.blocks(&grouped, tokens * 6, 256).is_some();
        assert!(!tokens(1365) && tokens(1366), "the line is at 1366 tokens");
        assert!(tokens(2048) && tokens(16384));
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
            elements * width % THREADS_AN_ELEMENTS_GROUP != 0,
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
            "element(low) * v[0] + element(high) * v[1]",
            "element(high) * v[0] + element(low) * v[1]",
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

    /// **A fused pair answers what the two dispatches it replaces answered, bit
    /// for bit**, at each of the three layouts a pair is dispatched in.
    ///
    /// This is the claim the fusion rests on and it is the same shape of claim
    /// `a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers`
    /// makes: an output element is still one simdgroup's `simd_sum` over lanes
    /// that still walk one weight row in the same chunks from the same byte in
    /// the same stride. What moved is which weight the walk is pointed at, which
    /// is decided once above the loop. So a tolerance here would be hiding the
    /// one mistake this can make — pointing half the grid at the wrong bank —
    /// and the check is exact.
    ///
    /// **Two banks that differ, because against one weight read twice every
    /// arm agrees.** The two cases are seeded apart and the case asserts they
    /// disagree before it compares anything.
    ///
    /// **The picked layout is the one a decode step takes** — a token's `TOP_K`
    /// slots naming different experts, which no tile reaches — and the repeating
    /// one is the shared bank's. The third arm is the grouped layout, which is
    /// two dispatches by construction and is here so that the pair's answer is
    /// held at the layout it does *not* fuse as well as at the two it does.
    ///
    /// **And what the pair costs the pipeline is asserted beside what it
    /// answers**, because the two together are what say a fused dispatch is
    /// worth the launch it saves: a merge that halved the threadgroup would be
    /// giving width back on the axis it was taken for.
    #[test]
    fn a_paired_dispatch_answers_what_the_two_dispatches_it_replaces_answer() {
        let Some(device) = device() else { return };
        // **Both words, and the equality is exact under each.** The paired
        // entry is the untiled one emitted a second time on either side of the
        // flag, so what a pair can get wrong — pointing half the grid at the
        // wrong bank — is the same mistake whichever walk it is a second
        // emission of, and a tolerance would hide it under both.
        for numerics in [Numerics::Reference, Numerics::Production] {
            let matmul =
                PackedMatmul::under(&device, numerics).expect("the packed matmul compiles");
            let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
            const EXPERTS: usize = 3;
            const ROWS: usize = TOKENS * TOP_K;

            fn uploaded<'a>(
                device: &'a Device,
                matmul: &'a PackedMatmul,
                experts: usize,
                case: &Case,
            ) -> PackedBank<'a> {
                PackedBank::upload(
                    device,
                    matmul,
                    experts,
                    IN_DIM,
                    OUT_DIM,
                    &case.packed(),
                    &case.scales,
                )
                .expect("the bank's shapes pair")
            }
            let upload = |case: &Case| uploaded(&device, &matmul, EXPERTS, case);
            // **What the second bank costs the pipeline, which is nothing.** A
            // threadgroup of either entry is that many independent simdgroups with
            // no threadgroup memory between them, and what the paired one adds is
            // four bindings and two selections above the walk — so the widest
            // threadgroup the device gives it is the widest it gives the entry it is
            // a second emission of. A pair that had cost occupancy would say so
            // here, and would have to be weighed against the launch it saves rather
            // than simply taken.
            assert_eq!(
                (
                    matmul.paired.max_threads_per_group(),
                    matmul.paired.threadgroup_memory()
                ),
                (
                    matmul.kernel.max_threads_per_group(),
                    matmul.kernel.threadgroup_memory()
                ),
                "a paired dispatch is not the threadgroup an untiled one is"
            );

            let gate_case = Case::seeded(0x9a1e_0001, IN_DIM, EXPERTS * OUT_DIM, ROWS);
            let up_case = Case::seeded(0x9a1e_0002, IN_DIM, EXPERTS * OUT_DIM, ROWS);
            let pair = PackedPair::new(upload(&gate_case), upload(&up_case))
                .expect("the pair's shapes pair");
            let (gate, up) = (upload(&gate_case), upload(&up_case));

            let alone = |bank: &PackedBank<'_>, chosen: &[u32], x: &[f32], per_source: usize| {
                let mut selection = device.buffer(chosen).expect("the selection uploads");
                let mut x = device.buffer(x).expect("the rows upload");
                let mut batch = device.batch().expect("a command buffer opens");
                let pending = bank
                    .encode_picked(&mut batch, &mut selection, &mut x, per_source)
                    .expect("the dispatch encodes");
                batch.wait().expect("the dispatch completes");
                pending.take()
            };
            let together = |chosen: &[u32], x: &[f32], per_source: usize| {
                let mut selection = device.buffer(chosen).expect("the selection uploads");
                let mut x = device.buffer(x).expect("the rows upload");
                let mut batch = device.batch().expect("a command buffer opens");
                let pending = pair
                    .encode_picked(&mut batch, &mut selection, &mut x, per_source)
                    .expect("the dispatch encodes");
                batch.wait().expect("the dispatch completes");
                pending.map(Pending::take)
            };

            // The routed bank's layout: a token's row read once per slot, the slots
            // naming different experts.
            let chosen = routing(EXPERTS);
            assert!(
                !tiles(&chosen, ROWS_A_TILE),
                "a routing a tile could reach is not the layout a decode step has"
            );
            let tokens = &gate_case.x[..TOKENS * IN_DIM];
            let [fused_gate, fused_up] = together(&chosen, tokens, TOP_K);
            let want_gate = alone(&gate, &chosen, tokens, TOP_K);
            let want_up = alone(&up, &chosen, tokens, TOP_K);
            assert_ne!(
                want_gate, want_up,
                "two banks that agreed would prove nothing"
            );
            assert_eq!(fused_gate, want_gate, "the pair's first half");
            assert_eq!(fused_up, want_up, "the pair's second half");
            assert_eq!(fused_gate.len(), ROWS * OUT_DIM);

            // The shared bank's layout: every row of the input read once per expert
            // in turn, which is what the modulo is for. Short enough that no run
            // reaches a tile, which is what keeps this the arm that fuses — a
            // decode step's shared bank is one token through two experts, so its
            // runs are one row.
            const REPEATED_TOKENS: usize = ROWS_A_TILE - 1;
            let repeated: Vec<u32> = (0..EXPERTS * REPEATED_TOKENS)
                .map(|row| (row / REPEATED_TOKENS) as u32)
                .collect();
            assert!(
                !tiles(&repeated, ROWS_A_TILE),
                "a repeat a tile could reach is not the arm that fuses"
            );
            let repeated_x = &gate_case.x[..REPEATED_TOKENS * IN_DIM];
            let repeat_alone = |bank: &PackedBank<'_>| {
                let mut x = device.buffer(repeated_x).expect("the rows upload");
                let mut batch = device.batch().expect("a command buffer opens");
                let pending = bank
                    .encode_repeating(&mut batch, &repeated, &mut x)
                    .expect("the dispatch encodes");
                batch.wait().expect("the dispatch completes");
                pending.take()
            };
            let mut x = device.buffer(repeated_x).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let fused = pair
                .encode_repeating(&mut batch, &repeated, &mut x)
                .expect("the dispatch encodes");
            batch.wait().expect("the dispatch completes");
            let [fused_gate, fused_up] = fused.map(Pending::take);
            assert_eq!(fused_gate, repeat_alone(&gate), "the repeated gate");
            assert_eq!(fused_up, repeat_alone(&up), "the repeated up");
            assert_eq!(fused_gate.len(), EXPERTS * REPEATED_TOKENS * OUT_DIM);

            // And the layout the pair does not fuse, which still has to answer the
            // same two tensors.
            let mut selection = device.buffer(&chosen).expect("the selection uploads");
            let mut x = device.buffer(tokens).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let mut sorted = grouping
                .encode(&mut batch, &mut selection, EXPERTS)
                .expect("the grouping encodes");
            let grouped = pair
                .encode_grouped(&mut batch, &mut sorted, &mut x, TOP_K)
                .expect("the dispatches encode");
            batch.wait().expect("the dispatches complete");
            let [grouped_gate, grouped_up] = grouped.map(Pending::take);
            let order = grouping
                .group(&device, &chosen, EXPERTS)
                .expect("the dispatch completes")
                .0;
            let row = |rows: &[f32], i: usize| rows[i * OUT_DIM..][..OUT_DIM].to_vec();
            for (at, from) in order.iter().enumerate() {
                let from = *from as usize;
                assert_eq!(
                    row(&grouped_gate, at),
                    row(&want_gate, from),
                    "gate row {at}"
                );
                assert_eq!(row(&grouped_up, at), row(&want_up, from), "up row {at}");
            }
        }
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
                let moved = packed_bytes(codes);
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
                let moved = packed_bytes(codes);
                line.push_str(&format!(
                    "{:>16}",
                    format!(
                        "{:.0}µs {:.0} GB/s",
                        1e6 * best.as_secs_f64(),
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
                let moved = packed_bytes(codes);
                line.push_str(&format!(
                    "{:>16}",
                    format!(
                        "{:.0}µs {:.0} GB/s",
                        1e6 * best.as_secs_f64(),
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
                let moved = packed_bytes(codes);
                format!("{:.0} GB/s", moved / taken.as_secs_f64() / 1e9)
            };
            let held = experts * OUT * IN;
            eprintln!(
                "  {experts:<10}{:>10}{:>11}{:>11}{:>12}{:>12}{:>13}{:>13}",
                format!("{}", ROWS / experts),
                format!("{:.0} MB", packed_bytes(held) / 1e6),
                format!("{:.0}µs", 1e6 * best[0].as_secs_f64()),
                format!("{:.0}µs", 1e6 * best[1].as_secs_f64()),
                format!("{:.0}µs", 1e6 * best[2].as_secs_f64()),
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
            let call = a_packed_call_costs(
                &device, &matmul, &grouping, rows, in_dim, out_dim, experts, groups,
            );
            let tiles = rows.div_ceil(ROWS_A_TILE);
            let read = match groups {
                false => tiles,
                true => rows.min(tiles + (ROWS_A_TILE - 1) * tiles.min(experts - 1)),
            };
            let moved = packed_bytes(read * out_dim * in_dim);
            // What the call could read at most, which is the bank itself. The
            // ratio between the two is how many times over a dispatch is charged
            // for a weight it may still have in cache, and it is the column that
            // says whether `achieved` is a bandwidth or an amplification.
            let held = packed_bytes(experts * out_dim * in_dim);
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

    /// The shapes a decode step gives this kernel, and how many calls a step
    /// makes of each.
    ///
    /// **A decode step is one token of everything, and nothing here has ever
    /// been tuned for it.** Both entries behind [`Numerics::Production`] refuse a
    /// call this short — see [`PackedMatmul::blocks`] — [`tiles`] is false for
    /// every row of this list, and [`RESIDENCY`] is declared by entries none of
    /// them reach. So every shape below runs [`ENTRY`], one simdgroup an output
    /// element, which is the kernel the engine shipped before any of that work.
    ///
    /// `(what, rows, in_dim, out_dim, experts, calls a step)`. The counts are the
    /// engine's — 42 layers of five projections, 40 MoE layers of a shared bank's
    /// three and a routed bank's three, two dense feed-forward networks and the
    /// head — and they sum to the 457 `packed_matmul` calls
    /// `which_kernels_own_a_decode_step` counts in a step.
    const DECODE_SHAPES: [(&str, usize, usize, usize, usize, usize); 10] = [
        ("q_proj, o_proj", 1, 4096, 4096, 1, 84),
        ("k_proj, v_proj", 1, 4096, 1024, 1, 84),
        ("r_proj", 1, 4096, 512, 1, 42),
        ("shared gate, up", 2, 4096, 2048, 2, 80),
        ("shared down", 2, 2048, 4096, 2, 40),
        ("routed gate, up", 6, 4096, 2048, 256, 80),
        ("routed down", 6, 2048, 4096, 256, 40),
        ("dense gate, up", 1, 4096, 16384, 1, 4),
        ("dense down", 1, 16384, 4096, 1, 2),
        ("lm_head", 1, 4096, 200058, 1, 1),
    ];

    /// **What each shape a decode step gives this kernel actually costs**, which
    /// is what the one `packed_matmul` row of the decode profile is made of.
    ///
    /// That row is 457 calls and 70% of the device's time, and it is ten shapes
    /// rather than one — from a `[1, 4096] @ [512, 4096]ᵀ` router projection to
    /// the head's `[200058, 4096]`. A row is only diagnosable once the shapes
    /// under it are named, which is what
    /// [`what_each_shape_of_a_prefills_packed_calls_costs`] does from the other
    /// side.
    ///
    /// **The `a step` column is the check**: this shape's own device time times
    /// how many of it a step dispatches, so the column sums to the profile's row
    /// if the decomposition is right and does not if it is not.
    ///
    /// **`elements` is the column the diagnosis turns on.** An output element is
    /// one simdgroup here, so it is also how many simdgroups the dispatch puts on
    /// a part with 80 cores — and the `input` column beside it is what those
    /// simdgroups read between them, which is every element re-reading the whole
    /// input row. That ratio is `4 / (1/2 + 1/32)` — 7.5 bytes of input load
    /// issued for every byte of weight — at every shape here, because it is a
    /// property of the walk rather than of the call.
    ///
    /// **The rates rank the shapes and do not state a bandwidth**, the way
    /// [`what_a_packed_multiply_costs_at_each_width_a_lane_reads`]'s do and for
    /// the same reason: one weight is dispatched against four times in a row, so
    /// a shape whose weight fits in cache is read once from memory and three
    /// times from it. What the shapes carrying a step's bytes have is 27 MB a
    /// call and 435, and what the cold figure is
    /// `one_dispatch_does_an_lm_head_shaped_multiply_without_meeting_the_watchdog`
    /// says.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_each_shape_of_a_decode_steps_packed_calls_costs() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");

        // **The fixtures are built before anything is timed and swept twice.**
        // The shapes run from 512 output elements to 200058, which is the order
        // a clock ramp would draw this table in by itself — and a fixture built
        // between two measurements is a gigabyte of codes generated with the
        // device idle, which is the same hazard arriving a second way. See
        // [`crate::testing::warmed`] and [`crate::testing::both_ways`].
        let shapes: Vec<Blocked> = DECODE_SHAPES
            .iter()
            .map(|&(what, rows, in_dim, out_dim, experts, _)| {
                Blocked::of(what, rows, in_dim, out_dim, experts, false)
            })
            .collect();
        let banks: Vec<PackedBank<'_>> = shapes
            .iter()
            .map(|shape| shape.upload(&device, &matmul))
            .collect();
        let at: Vec<usize> = (0..DECODE_SHAPES.len()).collect();
        let cost = |shape: usize| shapes[shape].costs_on(&device, &banks[shape], &grouping);
        crate::testing::warmed(|| {
            cost(0);
        });
        let (up, down) = crate::testing::both_ways(&at, cost);
        let taken = better(&up, &down);

        eprintln!(
            "  {:<18}{:>10}{:>10}{:>10}{:>10}{:>11}{:>8}{:>10}{:>11}",
            "shape",
            "elements",
            "weight",
            "input",
            "a call",
            "achieved",
            "of bus",
            "a step",
            "disagreed"
        );
        let (mut step, mut weight) = (Duration::ZERO, 0.0);
        for (shape, (what, rows, in_dim, out_dim, _, calls)) in
            DECODE_SHAPES.into_iter().enumerate()
        {
            let call = taken[shape];
            let codes = rows * out_dim * in_dim;
            let moved = packed_bytes(codes);
            let achieved = moved / call.as_secs_f64();
            step += call * calls as u32;
            weight += moved * calls as f64;
            eprintln!(
                "  {what:<18}{:>10}{:>10}{:>10}{:>10}{:>11}{:>8}{:>10}{:>11}",
                rows * out_dim,
                format!("{:.1} MB", moved / 1e6),
                format!("{:.0} MB", (codes * size_of::<f32>()) as f64 / 1e6),
                format!("{:.0}µs", 1e6 * call.as_secs_f64()),
                format!("{:.0} GB/s", achieved / 1e9),
                format!("{:.0}%", 1e2 * achieved / MEMORY_BANDWIDTH),
                format!("{:.2?}", call * calls as u32),
                format!(
                    "{:.0}%",
                    1e2 * (up[shape].max(down[shape]).as_secs_f64() / call.as_secs_f64() - 1.0)
                ),
            );
        }
        eprintln!(
            "  the ten shapes are {step:.2?} of a decode step and {:.2} GB of weight, whose floor \
             at {:.0} GB/s is {:.2?}",
            weight / 1e9,
            MEMORY_BANDWIDTH / 1e9,
            Duration::from_secs_f64(weight / MEMORY_BANDWIDTH),
        );
    }

    /// **Whether a decode dispatch is short of the machine**, which is the
    /// hypothesis on record for this kernel and has never been measured.
    ///
    /// K2 left it in one line — "the laggards are dispatches with 1024–4096
    /// output elements, too few to fill 80 cores" — and every shape a decode step
    /// has is in or near that range. An output element is one simdgroup and a
    /// threadgroup is [`THREADS_AN_ELEMENTS_GROUP`] threads, so 1024 elements is
    /// 512 threadgroups on a part with 80 cores and 4096 is 2048.
    ///
    /// The same one-row call at every width from a quarter of that to fifty times
    /// it, so the curve says whether what a narrow dispatch is short of is
    /// threadgroups. **If the rate climbs with the width and keeps climbing, the
    /// kernel wants more of the machine; if it flattens well under the bus, what
    /// bounds it is inside a simdgroup and no amount of them helps.**
    ///
    /// Nothing asserts a rate. What is asserted is that the widths span the ones
    /// a decode step actually dispatches.
    ///
    /// **Warm and swept both ways**, because a row ordered by how much work it
    /// does is exactly the row a clock ramp draws by itself — see
    /// [`crate::testing::warmed`], which is on record for a 1.7× effect that was
    /// entirely the clock. The two passes are printed beside each other so that
    /// an arm which reads differently up the list and down it says so.
    ///
    /// **The narrow arms are flattered and the wide ones are not**, which is the
    /// direction that makes the reading safe: four dispatches share a command
    /// buffer, so a 512-column weight is 1 MB read four times out of cache where
    /// a 65536-column one is 143 MB and cannot be.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn whether_a_decode_dispatch_is_short_of_the_machine() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const WIDTHS: [usize; 9] = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

        for width in [1024, 2048, 4096] {
            assert!(
                WIDTHS.contains(&width),
                "{width} is a width a decode step dispatches and is not swept"
            );
        }
        let shapes: Vec<Blocked> = WIDTHS
            .iter()
            .map(|&out_dim| Blocked::of("", 1, IN_DIM, out_dim, 1, false))
            .collect();
        let arms: Vec<usize> = (0..WIDTHS.len()).collect();
        let cost = |at: usize| shapes[at].costs(&device, &matmul, &grouping);
        crate::testing::warmed(|| {
            cost(arms.len() / 2);
        });
        let (up, down) = crate::testing::both_ways(&arms, cost);
        let taken = better(&up, &down);

        eprintln!(
            "  {:>10}{:>14}{:>10}{:>10}{:>11}{:>8}{:>11}",
            "elements", "threadgroups", "weight", "a call", "achieved", "of bus", "disagreed"
        );
        for (at, out_dim) in WIDTHS.into_iter().enumerate() {
            let codes = out_dim * IN_DIM;
            let moved = packed_bytes(codes);
            let achieved = moved / taken[at].as_secs_f64();
            eprintln!(
                "  {out_dim:>10}{:>14}{:>10}{:>10}{:>11}{:>8}{:>11}",
                out_dim * NARROWEST_SIMD / THREADS_AN_ELEMENTS_GROUP,
                format!("{:.1} MB", moved / 1e6),
                format!("{:.0}µs", 1e6 * taken[at].as_secs_f64()),
                format!("{:.0} GB/s", achieved / 1e9),
                format!("{:.0}%", 1e2 * achieved / MEMORY_BANDWIDTH),
                format!(
                    "{:.0}%",
                    1e2 * (up[at].max(down[at]).as_secs_f64() / taken[at].as_secs_f64() - 1.0)
                ),
            );
        }
    }

    /// The threadgroups an untiled dispatch is swept over, one of which is the
    /// shipped one.
    const THREADGROUPS_SWEPT: [usize; 5] = [32, 64, 128, 256, 512];

    /// **What a packed multiply costs at each threadgroup an untiled call takes**,
    /// and the sweep [`THREADS_AN_ELEMENTS_GROUP`] was chosen from.
    ///
    /// **The one knob here that changes no arithmetic at all.** A threadgroup of
    /// [`ENTRY`] is that many independent simdgroups sharing nothing, so the
    /// width reaches the grid and stops; what it decides is how evenly a call's
    /// simdgroups land on 80 cores. Every arm answers the shipped kernel's bit
    /// patterns and the case asserts it, which is what separates this from a
    /// sweep that measured five different computations.
    ///
    /// The shapes are the narrow ones, because those are where the imbalance is:
    /// a 512-element call is eight threadgroups of 64 against 80 cores whichever
    /// way it is cut, and a call wide enough to fill the machine has nothing to
    /// gain from being cut differently.
    ///
    /// Warm and swept both ways, for the reason
    /// [`whether_a_decode_dispatch_is_short_of_the_machine`] gives.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_threadgroup_an_untiled_call_takes() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        assert!(
            THREADGROUPS_SWEPT.contains(&THREADS_AN_ELEMENTS_GROUP),
            "{THREADS_AN_ELEMENTS_GROUP}"
        );

        // `(what, rows, in_dim, out_dim, experts)`.
        let shapes: Vec<Blocked> = [
            ("r_proj", 1, 4096, 512, 1),
            ("k_proj, v_proj", 1, 4096, 1024, 1),
            ("q_proj, o_proj", 1, 4096, 4096, 1),
            ("routed gate, up", 6, 4096, 2048, 16),
        ]
        .into_iter()
        .map(|(what, rows, in_dim, out_dim, experts)| {
            Blocked::of(what, rows, in_dim, out_dim, experts, false)
        })
        .collect();
        let arms: Vec<PackedMatmul> = THREADGROUPS_SWEPT
            .iter()
            .map(|&threads| {
                PackedMatmul::dispatching_at(&device, threads)
                    .unwrap_or_else(|err| panic!("{threads} threads compile: {err}"))
            })
            .collect();

        let want = bit_patterns(&an_untiled_call_answers(&device, &arms[0]));
        for (threads, arm) in THREADGROUPS_SWEPT.into_iter().zip(&arms) {
            assert_eq!(
                bit_patterns(&an_untiled_call_answers(&device, arm)),
                want,
                "a threadgroup of {threads} answered other bits"
            );
        }

        let at: Vec<usize> = (0..THREADGROUPS_SWEPT.len()).collect();
        let cost = |arm: usize| -> Vec<Duration> {
            // The banks are uploaded before the warm-up rather than inside it,
            // for the reason the chunk sweep gives: an upload is a hundred
            // megabytes copied with the device idle, and a warm-up loop that
            // pays one an iteration warms the host.
            let banks: Vec<PackedBank<'_>> = shapes
                .iter()
                .map(|shape| shape.upload(&device, &arms[arm]))
                .collect();
            let widest = shapes.len() - 1;
            crate::testing::warmed(|| {
                shapes[widest].costs_on(&device, &banks[widest], &grouping);
            });
            shapes
                .iter()
                .zip(&banks)
                .map(|(shape, bank)| shape.costs_on(&device, bank, &grouping))
                .collect()
        };
        let (up, down) = crate::testing::both_ways(&at, cost);

        eprintln!(
            "  {:<20}{}",
            "shape",
            THREADGROUPS_SWEPT
                .iter()
                .map(|threads| format!("{:>18}", format!("{threads} threads")))
                .collect::<String>()
        );
        for (which, shape) in shapes.iter().enumerate() {
            let moved = packed_bytes(shape.rows * shape.out_dim * shape.in_dim);
            let cells: String = (0..THREADGROUPS_SWEPT.len())
                .map(|arm| {
                    format!(
                        "{:>10}{:>8}",
                        format!("{:.1}\u{b5}s", 1e6 * up[arm][which].as_secs_f64()),
                        format!("{:.0} GB/s", moved / up[arm][which].as_secs_f64() / 1e9),
                    )
                })
                .collect();
            eprintln!("  {:<20}{cells}", shape.what);
            let other: String = (0..THREADGROUPS_SWEPT.len())
                .map(|arm| {
                    format!(
                        "{:>18}",
                        format!("{:.1}\u{b5}s", 1e6 * down[arm][which].as_secs_f64())
                    )
                })
                .collect();
            eprintln!("  {:<20}{other}", "  the other way");
        }
    }

    /// The cut counts an untiled call is swept over.
    ///
    /// **One of them is a cut of one**, which no element count can ask for and
    /// which is what separates the entry from the partition: at one cut the
    /// split entry walks the row the plain entry walks, in the same chunks, and
    /// answers the same bits — so what it costs over the plain entry is the
    /// threadgroup array, the barrier and the combine with nothing cut.
    const CUTS_SWEPT: [usize; 4] = [1, 2, 3, 4];

    /// **What a packed multiply costs at each count its walk is cut into**, and
    /// the sweep that refused the cut.
    ///
    /// The zero arm is the entry the cut stands in front of — every call left
    /// uncut, out of the same build under the same word — so what the table
    /// prices is the cut alone and not the flag.
    ///
    /// **The shapes are a decode step's, which is the whole of where this can
    /// pay.** Every one of them is on the steep part of
    /// [`whether_a_decode_dispatch_is_short_of_the_machine`]'s curve, and the
    /// head is the one call a step makes that is not — so it is here as the row
    /// that should not move, and a sweep that improved it would be measuring
    /// something other than what it claims to.
    ///
    /// **No arm answers the shipped kernel's bits and none can.** A cut is a
    /// different partition of the row, so what is asserted instead is
    /// [`MMA_TOLERANCE`] against the uncut arm — the same bound the block is
    /// held to, for the same reason, and an arm outside it is a mistake rather
    /// than an order.
    ///
    /// Warm and swept both ways, for the reason
    /// [`whether_a_decode_dispatch_is_short_of_the_machine`] gives.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_count_its_walk_is_cut_into() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        assert_eq!(CUTS_SWEPT[0], 1, "the arm that cuts once is the baseline");

        // `(what, rows, in_dim, out_dim, experts)`, and the elements each is.
        let shapes: Vec<Blocked> = [
            ("r_proj", 1, 4096, 512, 1),
            ("k_proj, v_proj", 1, 4096, 1024, 1),
            ("q_proj, o_proj", 1, 4096, 4096, 1),
            ("routed gate, up", 6, 4096, 2048, 16),
            ("lm_head", 1, 4096, 200058, 1),
        ]
        .into_iter()
        .map(|(what, rows, in_dim, out_dim, experts)| {
            Blocked::of(what, rows, in_dim, out_dim, experts, false)
        })
        .collect();
        // The entry the cut stands in front of, which is the row every arm is
        // read against and the one no cut count can ask for.
        let plain =
            PackedMatmul::under(&device, Numerics::Production).expect("the production compiles");
        // **And the split entry at one cut with nothing shared in it**, which is
        // what separates the two things a cut of one adds to the plain walk: a
        // threadgroup array and the barrier over it. The arm races and answers
        // whatever the scheduler leaves — at one cut it happens not to, since
        // the only partial is written and read by the same thread — so what it
        // prices is the barrier and what the compiler may do across it.
        let unbarriered = PackedMatmul::cut_from_source(
            &device,
            &crate::testing::instead_of(
                &source_cut(Block::SHIPPED),
                "    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
                "",
            ),
            1,
        )
        .expect("the mutant compiles");
        let arms: Vec<PackedMatmul> = CUTS_SWEPT
            .iter()
            .map(|&splits| {
                PackedMatmul::cut_at(&device, splits)
                    .unwrap_or_else(|err| panic!("{splits} cuts compile: {err}"))
            })
            .collect();

        let want = an_untiled_call_answers(&device, &plain);
        assert_eq!(
            bit_patterns(&an_untiled_call_answers(&device, &arms[0])),
            bit_patterns(&want),
            "one cut is not the walk it cuts"
        );
        for (splits, arm) in CUTS_SWEPT.into_iter().zip(&arms) {
            let apart = deviation(&an_untiled_call_answers(&device, arm), &want);
            assert!(
                apart <= MMA_TOLERANCE,
                "{splits} cuts deviate {apart:e} from the uncut walk"
            );
        }

        // The uncut plain entry first, then a cut apiece, then the barrier arm.
        let at: Vec<usize> = (0..CUTS_SWEPT.len() + 2).collect();
        let cost = |arm: usize| -> Vec<Duration> {
            // The banks are uploaded before the warm-up rather than inside it,
            // for the reason the chunk sweep gives: an upload is a hundred
            // megabytes copied with the device idle, and a warm-up loop that
            // pays one an iteration warms the host.
            let against = match arm {
                0 => &plain,
                arm if arm <= CUTS_SWEPT.len() => &arms[arm - 1],
                _ => &unbarriered,
            };
            let banks: Vec<PackedBank<'_>> = shapes
                .iter()
                .map(|shape| shape.upload(&device, against))
                .collect();
            let widest = shapes.len() - 1;
            crate::testing::warmed(|| {
                shapes[widest].costs_on(&device, &banks[widest], &grouping);
            });
            shapes
                .iter()
                .zip(&banks)
                .map(|(shape, bank)| shape.costs_on(&device, bank, &grouping))
                .collect()
        };
        let (up, down) = crate::testing::both_ways(&at, cost);

        eprintln!(
            "  {:<20}{:>10}{}",
            "shape",
            "elements",
            std::iter::once("uncut, plain".to_string())
                .chain(CUTS_SWEPT.iter().map(|splits| format!("{splits} cuts")))
                .chain(std::iter::once("1 cut, no barrier".to_string()))
                .map(|what| format!("{what:>18}"))
                .collect::<String>()
        );
        for (which, shape) in shapes.iter().enumerate() {
            let moved = packed_bytes(shape.rows * shape.out_dim * shape.in_dim);
            let taken = |arm: usize| better(&up[arm], &down[arm])[which];
            let uncut = taken(0).as_secs_f64();
            let cells: String = (0..CUTS_SWEPT.len() + 2)
                .map(|arm| {
                    format!(
                        "{:>10}{:>8}",
                        format!("{:.1}\u{b5}s", 1e6 * taken(arm).as_secs_f64()),
                        format!("{:.0}%", 1e2 * taken(arm).as_secs_f64() / uncut),
                    )
                })
                .collect();
            eprintln!(
                "  {:<20}{:>10}{cells}",
                shape.what,
                shape.rows * shape.out_dim
            );
            let rate: String = (0..CUTS_SWEPT.len() + 2)
                .map(|arm| {
                    format!(
                        "{:>18}",
                        format!("{:.0} GB/s", moved / taken(arm).as_secs_f64() / 1e9)
                    )
                })
                .collect();
            eprintln!("  {:<30}{rate}", "  achieved");
        }
    }

    /// **What a packed multiply costs at each chunk a lane holds**, and the sweep
    /// [`CHUNKS_IN_FLIGHT`] was chosen from.
    ///
    /// One is the kernel this engine shipped for eleven milestones: the walk's
    /// trip count is a runtime value, nothing unrolls it, and a lane waits a
    /// memory latency for four bytes before it issues four more. Every arm above
    /// one issues that many chunks before it adds any of them, and every arm
    /// answers the same bits —
    /// [`a_walk_that_holds_several_chunks_answers_the_bits_one_holding_one_answers`]
    /// is where that is held, so what this table ranks is the same arithmetic
    /// waiting differently.
    ///
    /// **The shapes are the ones that carry a decode step's bytes**, and the
    /// widest of them is there to say where the lever stops paying: a call of
    /// 32768 elements hides its own loads behind other simdgroups, which is what
    /// a held chunk buys for the shapes that cannot.
    ///
    /// **A routed bank of sixteen experts rather than the engine's 256**, because
    /// an arm here re-uploads its banks and at 256 that is a gigabyte copied
    /// between two measurements. The rows still name six different experts and
    /// still walk the same reduction; what the smaller bank gives up is the claim
    /// that the weight was cold, which is a claim about the rate and not about
    /// which arm won.
    ///
    /// Warm and swept both ways, for the reason
    /// [`whether_a_decode_dispatch_is_short_of_the_machine`] gives. Nothing
    /// asserts a rate; what is asserted is that the shipped count was among the
    /// ones tried.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_chunk_a_lane_holds() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        assert!(
            CHUNKS_HELD.contains(&CHUNKS_IN_FLIGHT),
            "{CHUNKS_IN_FLIGHT}"
        );

        // `(what, rows, in_dim, out_dim, experts)`.
        let shapes: Vec<Blocked> = [
            ("k_proj, v_proj", 1, 4096, 1024, 1),
            ("q_proj, o_proj", 1, 4096, 4096, 1),
            ("shared gate, up", 2, 4096, 2048, 2),
            ("routed gate, up", 6, 4096, 2048, 16),
            ("a call wide enough", 1, 4096, 32768, 1),
        ]
        .into_iter()
        .map(|(what, rows, in_dim, out_dim, experts)| {
            Blocked::of(what, rows, in_dim, out_dim, experts, false)
        })
        .collect();
        let arms: Vec<PackedMatmul> = CHUNKS_HELD
            .iter()
            .map(|&chunks| {
                PackedMatmul::from_source(&device, &a_lane_holding(chunks))
                    .unwrap_or_else(|err| panic!("{chunks} chunks compile: {err}"))
            })
            .collect();

        let at: Vec<usize> = (0..CHUNKS_HELD.len()).collect();
        let cost = |arm: usize| -> Vec<Duration> {
            let banks: Vec<PackedBank<'_>> = shapes
                .iter()
                .map(|shape| shape.upload(&device, &arms[arm]))
                .collect();
            // **Warmed on the widest shape and not the narrowest.** A warm-up
            // iteration is a submission the host waits 225 µs for, so warming
            // on a 13 µs dispatch leaves the device idle four fifths of the
            // budget and its clock where it was — which read as the first arm
            // of each direction, and only the first, coming back twice as slow.
            let widest = shapes.len() - 1;
            crate::testing::warmed(|| {
                shapes[widest].costs_on(&device, &banks[widest], &grouping);
            });
            shapes
                .iter()
                .zip(&banks)
                .map(|(shape, bank)| shape.costs_on(&device, bank, &grouping))
                .collect()
        };
        let (up, down) = crate::testing::both_ways(&at, cost);

        eprintln!(
            "  {:<20}{}",
            "shape",
            CHUNKS_HELD
                .iter()
                .map(|chunks| format!("{:>18}", format!("{chunks} held")))
                .collect::<String>()
        );
        for (which, shape) in shapes.iter().enumerate() {
            let moved = packed_bytes(shape.rows * shape.out_dim * shape.in_dim);
            let cells: String = (0..CHUNKS_HELD.len())
                .map(|arm| {
                    format!(
                        "{:>10}{:>8}",
                        format!("{:.1}µs", 1e6 * up[arm][which].as_secs_f64()),
                        format!("{:.0} GB/s", moved / up[arm][which].as_secs_f64() / 1e9),
                    )
                })
                .collect();
            eprintln!("  {:<20}{cells}", shape.what);
            let other: String = (0..CHUNKS_HELD.len())
                .map(|arm| {
                    format!(
                        "{:>18}",
                        format!("{:.1}µs", 1e6 * down[arm][which].as_secs_f64())
                    )
                })
                .collect();
            eprintln!("  {:<20}{other}", "  the other way");
        }
    }

    /// **What a decode dispatch's fixed cost is made of**, which is the length of
    /// one output element's own walk.
    ///
    /// The width sweep fits to a constant plus the bytes at the bus, and a
    /// constant that is really one simdgroup's *latency* would grow with how far
    /// that simdgroup has to walk — a lane covers [`BYTES_PER_LANE`] bytes a step
    /// and a simdgroup covers 32 of those, so a reduction of `in_dim` is
    /// `in_dim / 256` steps whatever the call's width. Sweeping the reduction at a
    /// fixed width is what says whether the constant is a launch or a walk.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_decode_dispatchs_fixed_cost_is_made_of() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        const REDUCTIONS: [usize; 6] = [512, 1024, 2048, 4096, 8192, 16384];
        const OUT_DIM: usize = 1024;

        let shapes: Vec<Blocked> = REDUCTIONS
            .iter()
            .map(|&in_dim| Blocked::of("", 1, in_dim, OUT_DIM, 1, false))
            .collect();
        let arms: Vec<usize> = (0..REDUCTIONS.len()).collect();
        let cost = |at: usize| shapes[at].costs(&device, &matmul, &grouping);
        crate::testing::warmed(|| {
            cost(arms.len() / 2);
        });
        let (up, down) = crate::testing::both_ways(&arms, cost);

        eprintln!(
            "  {:>10}{:>8}{:>10}{:>10}{:>12}{:>10}{:>14}",
            "reduction", "steps", "weight", "a call", "at the bus", "the rest", "the other way"
        );
        for (at, in_dim) in REDUCTIONS.into_iter().enumerate() {
            let codes = OUT_DIM * in_dim;
            let moved = packed_bytes(codes);
            let compulsory = moved / MEMORY_BANDWIDTH;
            eprintln!(
                "  {in_dim:>10}{:>8}{:>10}{:>10}{:>12}{:>10}{:>14}",
                in_dim / (NARROWEST_SIMD * BYTES_PER_LANE * CODES_PER_BYTE),
                format!("{:.1} MB", moved / 1e6),
                format!("{:.1}µs", 1e6 * up[at].as_secs_f64()),
                format!("{:.1}µs", 1e6 * compulsory),
                format!("{:.1}µs", 1e6 * (up[at].as_secs_f64() - compulsory)),
                format!("{:.1}µs", 1e6 * down[at].as_secs_f64()),
            );
        }
    }

    /// **What a decode-shaped dispatch costs beside more of its own kind**,
    /// which is what says whether the fixed cost the sweep above finds is the
    /// dispatch's or the command buffer's.
    ///
    /// [`whether_a_decode_dispatch_is_short_of_the_machine`] fits to a fixed cost
    /// plus the bytes at the bus, and [`crate::testing::device_time`] divides one
    /// command buffer's device clock by the dispatches in it — so a cost that is
    /// really the *buffer's* would arrive divided by that many and read as a
    /// per-dispatch constant. Sweeping how many dispatches share a buffer
    /// separates them: a per-buffer cost falls like `1/calls` and a per-dispatch
    /// one does not move.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_decode_dispatch_costs_beside_more_of_its_own_kind() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        const CALLS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
        // `k_proj` and `v_proj` at a decode step, which are 168 of the 457 calls
        // a step makes. `r_proj` is narrower at 512 elements and carries a
        // twentieth of the bytes; this is the narrow shape a step is actually
        // made of.
        const OUT_DIM: usize = 1024;

        let case = Case::seeded(1, IN_DIM, OUT_DIM, 1);
        let projection = case.upload(&device, &matmul);
        let mut x = device.buffer(&case.x).expect("the rows upload");
        let codes = OUT_DIM * IN_DIM;
        let moved = packed_bytes(codes);

        let mut cost = |calls: usize| {
            crate::testing::device_time(&device, calls, |batch| {
                projection
                    .encode_over(batch, &mut x)
                    .expect("the dispatch encodes");
            })
        };
        crate::testing::warmed(|| {
            cost(16);
        });
        let (up, down) = crate::testing::both_ways(&CALLS, cost);

        eprintln!(
            "  {:>8}{:>12}{:>12}{:>12}",
            "a buffer", "a dispatch", "achieved", "the other way"
        );
        for (at, calls) in CALLS.into_iter().enumerate() {
            eprintln!(
                "  {calls:>8}{:>12}{:>12}{:>12}",
                format!("{:.1}µs", 1e6 * up[at].as_secs_f64()),
                format!("{:.0} GB/s", moved / up[at].as_secs_f64() / 1e9),
                format!("{:.1}µs", 1e6 * down[at].as_secs_f64()),
            );
        }
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

            let call = best;
            let tiles = ROWS.div_ceil(height);
            let read = ROWS.min(tiles + (height - 1) * tiles.min(EXPERTS - 1));
            let codes = read * OUT * IN;
            let moved = packed_bytes(codes);
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
        let held = packed_bytes(EXPERTS * OUT * IN);

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

    /// The two dispatches every limiter table here is read across: one shape a
    /// prefill gives each tiled entry, per token of the prompt.
    ///
    /// `q_proj` is a row a token through one expert, which is the tiled entry's
    /// largest shape; a routed bank is six rows a token over 256 experts, which
    /// at [`BOUND_TOKENS`] is the 96-rows-an-expert run length
    /// [`how_often_a_long_prefill_reads_one_experts_weight`]'s third arm
    /// measured and a 1.07 GB weight nothing caches.
    ///
    /// **Per token rather than at one length, because two tables read it at
    /// different ones.** The tile's is quoted at [`BOUND_TOKENS`] and the
    /// block's at each of [`BLOCKED_LENGTHS`], and two constants naming the same
    /// two calls are two that can drift into describing different work.
    ///
    /// `(what, rows a token, in_dim, out_dim, experts, grouped)`.
    const BOUND_SHAPES: [(&str, usize, usize, usize, usize, bool); 2] = [
        ("q_proj, tiled", 1, 4096, 4096, 1, false),
        ("a routed bank, grouped", 6, 4096, 2048, 256, true),
    ];

    /// The prompt the tile's own limiter tables are quoted at.
    const BOUND_TOKENS: usize = 4096;

    fn bound_header() -> String {
        header(BOUND_SHAPES.iter().map(|(what, ..)| *what), 26)
    }

    /// The column headings of a limiter table, which is the shapes it is read
    /// across at whatever width its cells are printed at.
    fn header<'a>(shapes: impl Iterator<Item = &'a str>, width: usize) -> String {
        shapes.map(|what| format!("{what:>width$}")).collect()
    }

    /// What one dispatch of a bank of this shape costs `matmul` on the device,
    /// through whichever entry the shape reaches.
    ///
    /// The fixture is made and thrown away per call, which is what a caller
    /// sweeping shapes rather than kernels wants — see [`Blocked`], which is the
    /// same measurement over a fixture kept across many arms.
    #[allow(clippy::too_many_arguments)]
    fn a_packed_call_costs(
        device: &Device,
        matmul: &PackedMatmul,
        grouping: &ExpertGrouping,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        experts: usize,
        grouped: bool,
    ) -> Duration {
        Blocked::of("", rows, in_dim, out_dim, experts, grouped).costs(device, matmul, grouping)
    }

    /// The same over both of [`BOUND_SHAPES`] at [`BOUND_TOKENS`], which is what
    /// a row of either of the tile's tables is.
    fn a_prefills_shapes_cost(device: &Device, matmul: &PackedMatmul) -> Vec<Duration> {
        let grouping = ExpertGrouping::new(device).expect("the grouping compiles");
        BOUND_SHAPES
            .iter()
            .map(|&shape| Blocked::at(BOUND_TOKENS, shape).costs(device, matmul, &grouping))
            .collect()
    }

    /// One shape's fixture and the dispatch that measures it — the one place
    /// this file encodes a packed call for a clock, whichever table is asking.
    ///
    /// **A struct rather than six arguments, because the [`Case`] is what
    /// costs.** A routed bank's is 1.07 GB of codes and a couple of seconds to
    /// generate, and the block's table puts six shapes through six arms — so a
    /// fixture made per measurement would be most of that case's runtime and
    /// none of its subject. A caller sweeping shapes rather than arms wants the
    /// opposite and gets it from [`a_packed_call_costs`], which makes one of
    /// these and drops it.
    struct Blocked {
        what: &'static str,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        experts: usize,
        grouped: bool,
        case: Case,
        /// The codes as the kernel reads them, packed once. `Case::packed` walks
        /// every code, and at a routed bank's 2^31 of them that is not something
        /// to redo per arm.
        packed: Vec<u8>,
        chosen: Vec<u32>,
    }

    impl Blocked {
        /// The shape a prompt of `tokens` gives one of [`BOUND_SHAPES`].
        fn at(tokens: usize, shape: (&'static str, usize, usize, usize, usize, bool)) -> Self {
            let (what, rows_a_token, in_dim, out_dim, experts, grouped) = shape;
            Self::of(
                what,
                tokens * rows_a_token,
                in_dim,
                out_dim,
                experts,
                grouped,
            )
        }

        fn of(
            what: &'static str,
            rows: usize,
            in_dim: usize,
            out_dim: usize,
            experts: usize,
            grouped: bool,
        ) -> Self {
            let case = Case::seeded(1, in_dim, experts * out_dim, rows);
            Self {
                what,
                rows,
                in_dim,
                out_dim,
                experts,
                grouped,
                packed: case.packed(),
                case,
                // Sorted already, so a block straddles two experts only where
                // the run length made it and not by accident of the routing.
                chosen: (0..rows).map(|row| (row * experts / rows) as u32).collect(),
            }
        }

        /// This call encoded against `bank`, which is the one place the two
        /// layouts differ and so the one place they are written.
        fn encode(
            &self,
            batch: &mut Batch<'_>,
            device: &Device,
            bank: &PackedBank<'_>,
            x: &mut Buffer<f32>,
            grouping: &ExpertGrouping,
        ) {
            match self.grouped {
                false => {
                    bank.encode_over(batch, &self.chosen, x)
                        .expect("the dispatch encodes");
                }
                true => {
                    let mut picked = device.buffer(&self.chosen).expect("the selection uploads");
                    let mut sorted = grouping
                        .encode(batch, &mut picked, self.experts)
                        .expect("the grouping encodes");
                    bank.encode_grouped(batch, &mut sorted, x, 1, Through::Gathered)
                        .expect("the dispatch encodes");
                }
            }
        }

        /// What one dispatch of this shape costs `matmul` on the device.
        ///
        /// Four dispatches to a command buffer, best of three rounds: a
        /// submission costs 225 µs and the dispatches a sweep asks about are
        /// tens, so what the device's clock is divided by has to be the dispatch
        /// rather than the submission around it.
        ///
        /// **The division is [`crate::testing::device_time`]'s and this side
        /// does not repeat it**, which is what
        /// [`a_blocks_table_reports_one_dispatch_of_the_shape_it_names`] holds.
        fn costs(
            &self,
            device: &Device,
            matmul: &PackedMatmul,
            grouping: &ExpertGrouping,
        ) -> Duration {
            self.costs_on(device, &self.upload(device, matmul), grouping)
        }

        /// The same against a bank the caller made, which is what a sweep over
        /// one kernel and many shapes wants.
        ///
        /// **An upload is a gigabyte of codes copied with the device idle**, and
        /// a sweep that pays one between every two measurements is a sweep whose
        /// arms each open on a colder machine than the one before — which reads
        /// as a monotone row for the same reason
        /// [`crate::testing::warmed`] exists. A caller that holds its banks
        /// across the sweep pays none of them.
        fn costs_on(
            &self,
            device: &Device,
            bank: &PackedBank<'_>,
            grouping: &ExpertGrouping,
        ) -> Duration {
            const CALLS: usize = 4;
            const ROUNDS: usize = 3;
            let mut x = device.buffer(&self.case.x).expect("the rows upload");

            let mut best = Duration::MAX;
            for _ in 0..ROUNDS {
                best = best.min(crate::testing::device_time(device, CALLS, |batch| {
                    self.encode(batch, device, bank, &mut x, grouping);
                }));
            }
            best
        }

        fn upload<'a>(&self, device: &'a Device, matmul: &'a PackedMatmul) -> PackedBank<'a> {
            PackedBank::upload(
                device,
                matmul,
                self.experts,
                self.in_dim,
                self.out_dim,
                &self.packed,
                &self.case.scales,
            )
            .expect("the bank's shapes pair")
        }

        /// What this call has to move across memory however it is written: every
        /// expert's weight once, the rows in, the rows out.
        ///
        /// **Not what [`PackedBank::moves`] declares, and the difference is the
        /// whole question.** That method charges a weight per block of rows that
        /// shares it, which is what the dispatch *issues* — 1791 weight reads at
        /// 8192 tokens where there are 256 experts. This is the floor underneath
        /// it: the bytes a kernel with an infinite cache would still have to
        /// fetch, and so the only denominator against which "bandwidth-bound"
        /// means bound by the machine rather than by the kernel's own re-reading.
        /// The multiply-adds this call performs, twice over, which is what a
        /// rate against the instruction's own ceiling divides.
        fn flops(&self) -> f64 {
            2.0 * self.rows as f64 * self.in_dim as f64 * self.out_dim as f64
        }

        fn compulsory(&self) -> usize {
            let elements = self.experts * self.out_dim * self.in_dim;
            elements * BITS / u8::BITS as usize
                + elements / GROUP_SIZE
                + size_of::<f32>() * self.rows * (self.in_dim + self.out_dim)
        }
    }

    /// One arm of the limiter table: a name, and the shipped source with one
    /// term of the tile's inner loop replaced by something that costs an
    /// instruction over the same operands and cannot be folded away.
    ///
    /// **The replacements are cheap rather than absent**, so that nothing an arm
    /// removes takes something else with it: a value deleted outright would let
    /// the compiler drop every load that fed it and the arm would price two
    /// things at once.
    fn without_each_term_of_a_tile() -> Vec<(&'static str, String)> {
        let shipped = source();
        vec![
            // The weight, which is the term A3's slot-zero mutation is on the
            // attention kernel: every column of every tile walks expert zero's
            // first row instead of its own, so the whole weight working set is
            // one 2 KB row and its scales. The tile walks the same bytes, decodes
            // the same codes and does the same arithmetic. `col * 0` rather than
            // a constant, so that nothing the tile computed goes unused and the
            // arm cannot be a shorter kernel by accident.
            (
                "the weight it reads",
                crate::testing::instead_of(
                    &shipped,
                    "(ulong)expert * shape.code_stride\n            + (ulong)col * bytes;\n        \
                     scale[c] = scales + shape.scale_base + (ulong)expert * shape.scale_stride\n    \
                     \x20       + (ulong)col * scale_bytes;",
                    "(ulong)(expert * 0u) * shape.code_stride\n            \
                     + (ulong)(col * 0u) * bytes;\n        \
                     scale[c] = scales + shape.scale_base + (ulong)(expert * 0u) * \
                     shape.scale_stride\n            + (ulong)(col * 0u) * scale_bytes;",
                ),
            ),
            // The input, which is what the column tile was taken for: a row tile
            // alone reads 32 input floats for every byte of weight and four
            // columns bring that to eight. **Confined rather than removed**, the
            // way the weight arm confines the weight: the same two loads are
            // issued from the same place in the loop and land inside eight
            // floats, so what comes off is the traffic and not the load.
            (
                "the input rows it walks",
                crate::testing::instead_of(
                    &shipped,
                    "device const float *v = x + sources[r] + at;",
                    "device const float *v = x + ((sources[r] + at) & 7u);",
                ),
            ),
            // The dequantisation, which is two gathers into a 16-entry constant
            // array for every packed byte — and the one term here whose cost is
            // a memory access nobody counts, since `PackedBank::moves` charges
            // the codes and not the table they index. What this arm removes is
            // the decode and not the table: see
            // `what_each_way_of_decoding_a_packed_byte_costs`, which holds the
            // gather against the two ways of answering the same bits for less.
            (
                "the table it decodes through",
                crate::testing::instead_of(
                    &shipped,
                    "                low[c] = element(code & CODE_MASK);\n                high[c] = element((code >> BITS) & CODE_MASK);",
                    "                low[c] = (float)(code & CODE_MASK);\n                high[c] = (float)((code >> BITS) & CODE_MASK);",
                ),
            ),
            // Three quarters of the multiply-adds, twice: once down the tile and
            // once across it. The columns arm leaves every weight byte and every
            // input float where it was and takes only the accumulate; the rows
            // arm takes the input reads with it, which is why the two are read
            // beside each other rather than either alone.
            (
                "three quarters of the columns",
                crate::testing::instead_of(
                    &shipped,
                    "                for (uint c = 0; c < COLS_A_TILE; ++c) {\n                    dots[r][c] +=",
                    "                for (uint c = 0; c < 1u; ++c) {\n                    dots[r][c] +=",
                ),
            ),
            (
                "three quarters of the rows",
                crate::testing::instead_of(
                    &shipped,
                    "            for (uint r = 0; r < ROWS_A_TILE; ++r) {\n                device const float *v",
                    "            for (uint r = 0; r < 1u; ++r) {\n                device const float *v",
                ),
            ),
            // The group scale, which is a byte read and a divide per chunk and
            // then a multiply-add per accumulator. Reading the row's first scale
            // instead leaves the load in the kernel and takes it out of the loop.
            (
                "the scale it walks to",
                crate::testing::instead_of(
                    &shipped,
                    "as_type<float>(uint(scale[c][b / BYTES_PER_GROUP]) << EXPONENT_SHIFT);",
                    "as_type<float>(uint(scale[c][0]) << EXPONENT_SHIFT);",
                ),
            ),
            // The cross-lane reduction, which is one per output element at the
            // end of a walk thousands of bytes long.
            (
                "the simd_sum",
                crate::testing::instead_of(
                    &shipped,
                    "const float sum = simd_sum(sums[r][c]);",
                    "const float sum = sums[r][c];",
                ),
            ),
        ]
    }

    /// One arm of the block's limiter table: a name, and the shipped production
    /// source with one term of [`MMA`]'s inner loop replaced by something that
    /// costs an instruction over the same operands and cannot be folded away.
    ///
    /// **The same discipline as [`without_each_term_of_a_tile`] and against a
    /// different kernel.** Every "this is not bandwidth-bound" finding in this
    /// repo was taken on the reference tile — A3's 10.4% re-read, A4's whole
    /// limiter table — and the block is 2.85× faster than that tile. A kernel
    /// that was issue-bound can become bandwidth-bound when it gets faster, and
    /// nothing here had asked the question of the kernel that actually runs a
    /// prefill under the production flag.
    ///
    /// **Both entries mutate, because both are the same string.**
    /// [`source_under`] appends [`MMA`] twice — once per entry — so a
    /// replacement made here lands in `mma_matmul_rows` and `mma_matmul_grouped`
    /// alike, which is what lets one table carry a column for each.
    fn without_each_term_of_a_block() -> Vec<(&'static str, String)> {
        let shipped = source_blocked(Numerics::Production, Block::SHIPPED);
        vec![
            // The weight, which is A3's slot-zero mutation moved onto the block:
            // every block stages expert zero's first column instead of its own,
            // so the whole weight working set is one 2 KB column and its scales.
            // The block stages the same bytes, decodes the same codes and drives
            // the same fragments. `* 0u` rather than a constant, so nothing the
            // block computed goes unused.
            //
            // **Both pointers, because confining one of them is not a working
            // set.** The codes and the scales are separate walks over separate
            // buffers, and an arm that pinned the codes alone would leave a
            // megabyte of scales being fetched per expert.
            (
                "the weight it reads",
                crate::testing::instead_of(
                    &shipped,
                    "+ (ulong)expert * shape.code_stride + (ulong)column * bytes;\n        device \
                     const uchar *scale = scales + shape.scale_base\n            + (ulong)expert * \
                     shape.scale_stride + (ulong)column * scale_bytes;",
                    "+ (ulong)(expert * 0u) * shape.code_stride + (ulong)(column * 0u) * \
                     bytes;\n        device const uchar *scale = scales + shape.scale_base\n       \
                     \x20    + (ulong)(expert * 0u) * shape.scale_stride + (ulong)(column * 0u) * \
                     scale_bytes;",
                ),
            ),
            // The input, confined rather than removed the way the tile's arm
            // confines it: the same load issued from the same place in the step
            // loop, landing inside a kilobyte. **This is the term the block
            // re-reads hardest** — a call of `out_dim` columns is
            // `out_dim / MMA_COLS_A_BLOCK` block-columns and every one of them
            // stages the same input rows again, which is 32 re-reads for a
            // routed bank and 64 for `q_proj`.
            (
                "the input rows it walks",
                crate::testing::instead_of(
                    &shipped,
                    "device const float *values = x + source + b * CODES_PER_BYTE + x_at;",
                    "device const float *values = x + ((source + b * CODES_PER_BYTE + x_at) & \
                     255u);",
                ),
            ),
            // The dequantisation, which the block already amortises eight times
            // as far as the tile does — the whole reason the flag was opened. A4
            // measured it at 30% of the tile; what is left of it here is what
            // says whether that amortisation finished the term or only moved it.
            (
                "the table it decodes through",
                crate::testing::instead_of(
                    &shipped,
                    "staged_w[at] = element(code & CODE_MASK) * by;\n                \
                     staged_w[at + 1] = element((code >> BITS) & CODE_MASK) * by;",
                    "staged_w[at] = (float)(code & CODE_MASK) * by;\n                \
                     staged_w[at + 1] = (float)((code >> BITS) & CODE_MASK) * by;",
                ),
            ),
            // **Three of every four fragment loads, which is the reuse question
            // asked without the refactor that would answer it properly.** A
            // simdgroup here holds MMA_FRAGMENTS_DOWN × MMA_FRAGMENTS_ACROSS =
            // four accumulators and issues two `lhs` and two `rhs` loads to feed
            // them, which is 1:1 — where mlx-vlm's 64×64 steel tile holds 32
            // accumulators against 12 loads and runs 2.7:1. A taller block is
            // what buys that ratio and it costs a doubled floor.
            //
            // This arm is the ratio's ceiling bought for nothing: the loads are
            // hoisted out of the `k` loop, so the four fragments of the first
            // step drive all sixteen multiply-accumulates and the ratio is 4:1.
            // **It answers wrongly — it is three quarters of the reduction — so
            // what it prices is the loads and not the arithmetic**, and it is
            // the upper bound on anything a taller block could return.
            (
                "three of every four fragment loads",
                fragments_loaded_once(&shipped),
            ),
            // Three quarters of the multiply-accumulates, with every fragment
            // load left where it was — the other side of the arm above, and the
            // one that says whether the instruction itself is the wall.
            (
                "three quarters of the multiply-accumulates",
                crate::testing::instead_of(&shipped, EVERY_ACCUMULATE, ONE_ACCUMULATE),
            ),
            // **The two barriers a step**, which is the term no ablation across
            // the bus can reach and the largest thing left inside the
            // threadgroup. A step is MMA_CODES_A_STEP codes wide, so a
            // reduction of 4096 is 128 of them and 256 barriers a pass — every
            // one of which stops all eight simdgroups until the slowest
            // arrives.
            //
            // **This arm races and that is what it prices.** Without them a
            // simdgroup reads a tile another is still filling, so what it
            // answers is nothing in particular; what it costs is the step loop
            // with the same loads, the same decode and the same
            // multiply-accumulates and none of the waiting. It is the upper
            // bound on what a wider step, a double buffer or a smaller
            // threadgroup could return between them.
            (
                "the two barriers a step",
                without_the_steps_barriers(&shipped),
            ),
        ]
    }

    /// The shipped production source with the step loop's two barriers removed
    /// and every other barrier left where it is.
    ///
    /// Anchored on what follows each of them, because [`MMA`] reaches
    /// `threadgroup_barrier` six times and only these two are per step.
    fn without_the_steps_barriers(shipped: &str) -> String {
        let opened = crate::testing::instead_of(
            shipped,
            "            threadgroup_barrier(mem_flags::mem_threadgroup);\n\n            // The \
             input, as it lies.",
            "            // The input, as it lies.",
        );
        crate::testing::instead_of(
            &opened,
            "            threadgroup_barrier(mem_flags::mem_threadgroup);\n\n            for \
             (uint k = 0;",
            "            for (uint k = 0;",
        )
    }

    /// The shipped production source with the two fragment loads hoisted out of
    /// the step's `k` loop, so that one set of fragments drives every
    /// multiply-accumulate of the step.
    ///
    /// Two replacements rather than one because the declarations move and the
    /// `if` has to be closed, and both anchors carry enough of their
    /// surroundings to be unique in a source that holds three loops over
    /// `MMA_FRAGMENTS_DOWN`.
    fn fragments_loaded_once(shipped: &str) -> String {
        let hoisted = crate::testing::instead_of(shipped, EVERY_STEP_LOADS, ONE_STEP_LOADS);
        crate::testing::instead_of(
            &hoisted,
            EVERY_ACCUMULATE,
            &format!("\n                }}{EVERY_ACCUMULATE}"),
        )
    }

    /// The block's multiply-accumulate loops, as the two arms above anchor on
    /// them: one prices the arithmetic by cutting three quarters of it and the
    /// other closes a hoisted `if` in front of it, and a single spelling is what
    /// keeps the two from drifting apart under an edit to the kernel.
    ///
    /// **Written with the newline and the indentation [`MMA`] has them at**, and
    /// long enough to be unique — that source opens three loops over
    /// `MMA_FRAGMENTS_DOWN` and only this one carries a
    /// `simdgroup_multiply_accumulate` under it.
    const EVERY_ACCUMULATE: &str = r#"
                for (uint i = 0; i < MMA_FRAGMENTS_DOWN; ++i) {
                    for (uint j = 0; j < MMA_FRAGMENTS_ACROSS; ++j) {
                        simdgroup_multiply_accumulate("#;

    /// The same loops cut to one accumulator of the four, with every fragment
    /// load left where it was.
    const ONE_ACCUMULATE: &str = r#"
                for (uint i = 0; i < 1u; ++i) {
                    for (uint j = 0; j < 1u; ++j) {
                        simdgroup_multiply_accumulate("#;

    /// The step's `k` loop as it opens, with the two fragment arrays declared
    /// inside it — so a set of fragments is loaded for every eighth of the
    /// reduction and feeds four multiply-accumulates.
    const EVERY_STEP_LOADS: &str = r#"
            for (uint k = 0; k < MMA_CODES_A_STEP / MMA_FRAGMENT; ++k) {
                simdgroup_float8x8 lhs[MMA_FRAGMENTS_DOWN];
                simdgroup_float8x8 rhs[MMA_FRAGMENTS_ACROSS];
"#;

    /// The same with the declarations hoisted above the loop and the loads put
    /// behind `k == 0`, so one set of fragments feeds sixteen.
    const ONE_STEP_LOADS: &str = r#"
            simdgroup_float8x8 lhs[MMA_FRAGMENTS_DOWN];
            simdgroup_float8x8 rhs[MMA_FRAGMENTS_ACROSS];
            for (uint k = 0; k < MMA_CODES_A_STEP / MMA_FRAGMENT; ++k) {
                if (k == 0) {
"#;

    /// **What a prefill's two tiled matmul rows are bound by, one term at a
    /// time** — 72.3 s of a 132.97 s prefill at 16384 tokens, and neither row
    /// near a plausible ceiling of anything.
    ///
    /// A3 priced the weight re-reading and found it worth 10.4%: on a fixed 1.07
    /// GB bank, ideal placement costs 2758 ns a row and 96-fold re-reading costs
    /// 3046. **So all but a tenth of those reads are served without reaching
    /// memory, and what the other nine tenths of the row are was never asked.**
    /// The same instrument the attention kernel got, on the kernel beside it.
    ///
    /// **Both entries, because they are the same source and two rows.** A tiled
    /// call is a projection over one expert and a grouped one a routed bank over
    /// 256, and the profile puts them 38.15 s and 34.16 s apart — so a term that
    /// reads differently on the two is a fact about the call rather than about
    /// the walk.
    ///
    /// The shapes are a 4096-token prompt's: `q_proj` is 4096 rows of one expert
    /// and the routed bank is 96 rows an expert over 256 of them, which is the
    /// run length A3's third arm measured. Every arm answers wrongly and the
    /// case asserts that it does.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_prefills_packed_matmul_is_bound_by() {
        let Some(device) = device() else { return };

        let shipped = matmul(&device);
        let want = a_tiled_call_answers(&device, &shipped);
        eprintln!("  {:<32}{}", "without", bound_header());

        let cost = |matmul: &PackedMatmul| a_prefills_shapes_cost(&device, matmul);
        let whole = cost(&shipped);
        let row = |what: &str, taken: &[Duration]| {
            let cells: String = taken
                .iter()
                .zip(&whole)
                .map(|(each, whole)| {
                    format!(
                        "{:>17}{:>9}",
                        format!("{:.2}ms", 1e3 * each.as_secs_f64()),
                        format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                    )
                })
                .collect();
            eprintln!("  {what:<32}{cells}");
        };
        row("nothing — the kernel", &whole);

        for (what, written) in without_each_term_of_a_tile() {
            let mutant = PackedMatmul::from_source(&device, &written).expect("the mutant compiles");
            assert_ne!(
                a_tiled_call_answers(&device, &mutant),
                want,
                "{what}: the mutation answered what the kernel answers"
            );
            row(what, &cost(&mutant));
        }
    }

    /// The shapes the decode limiter table is read across: the projection that
    /// carries most of a step's projection bytes, the narrowest dispatch a step
    /// makes, and the bank that carries most of everything.
    ///
    /// `(what, rows, in_dim, out_dim, experts)`, and the experts are each shape's
    /// own — one for a projection and 256 for a routed bank, whose six rows are
    /// spread across it the way a router spreads them.
    const DECODE_BOUND_SHAPES: [(&str, usize, usize, usize, usize); 3] = [
        ("q_proj", 1, 4096, 4096, 1),
        ("k_proj", 1, 4096, 1024, 1),
        ("a routed bank", 6, 4096, 2048, 256),
    ];

    /// One arm of the decode limiter table: a name, and the shipped source with
    /// one term of [`ENTRY`]'s walk replaced by something that costs an
    /// instruction over the same operands and cannot be folded away.
    ///
    /// **The same discipline as [`without_each_term_of_a_tile`], on the entry
    /// beside it.** Every "this is not bandwidth-bound" finding in this file was
    /// taken on a *tile* — A3's re-read, A4's limiter table, A9's on the block —
    /// and a tile is dispatched by a prefill and by nothing else. This is the
    /// walk a decode step runs, and no milestone has asked it anything.
    ///
    /// **Two arms a term where the tile's tables have one**, because the two
    /// answers are different levers. Confining a walk to a kilobyte leaves every
    /// load issued and takes the traffic off, so it prices the memory; taking the
    /// load out and forming the same value from the index it was at prices the
    /// load as well. A term whose traffic is free and whose loads are not is a
    /// term that wants fewer instructions rather than fewer bytes, and the tile's
    /// tables could not tell the two apart.
    fn without_each_term_of_the_untiled_walk() -> Vec<(&'static str, String)> {
        let shipped = source();
        vec![
            // The weight, confined the way A3's slot-zero mutation confines it:
            // every element of every call walks expert zero's first row, so the
            // whole weight working set is one 2 KB row and its scales. `* 0u`
            // rather than a constant, so nothing the kernel computed goes unused.
            //
            // **Both emissions of the walk are mutated and only one is
            // dispatched here.** The untiled entry and the paired one are one
            // template — see [`element_entry`] — so the text below occurs twice
            // and `instead_of` writes both; what this table times is the
            // untiled arm, and a paired dispatch it never makes would have been
            // mutated the same way.
            (
                "the weight's traffic",
                crate::testing::instead_of(
                    &shipped,
                    "        packed + code_base + (ulong)expert * shape.code_stride + \
                     (ulong)col * bytes,\n        scaled + scale_base + (ulong)expert * \
                     shape.scale_stride\n            + (ulong)col * (bytes / BYTES_PER_GROUP),",
                    "        packed + code_base + (ulong)(expert * 0u) * shape.code_stride + \
                     (ulong)(col * 0u) * bytes,\n        scaled + scale_base + (ulong)(expert \
                     * 0u) * shape.scale_stride\n            + (ulong)(col * 0u) * (bytes / \
                     BYTES_PER_GROUP),",
                ),
            ),
            // The same term with the load itself gone: the code is the index it
            // would have been read from, so every mask, every gather and every
            // multiply-add downstream of it stays where it was.
            (
                "the weight's loads",
                crate::testing::instead_of(
                    &shipped,
                    "        const uint code = packed[b + i];",
                    "        const uint code = b + i;",
                ),
            ),
            // **The input, which is the term this walk re-reads hardest.** Every
            // output element is a simdgroup of its own and every one of them
            // reads the whole input row, so a call of `out_dim` columns reads it
            // that many times — 7.5 bytes of input load issued per byte of
            // weight, at every shape a decode step has. Confined to eight floats,
            // so the two loads are issued from the same place and land in a cache
            // line.
            (
                "the input's traffic",
                crate::testing::instead_of(
                    &shipped,
                    "        device const float *v = values + (b + i) * CODES_PER_BYTE;",
                    "        device const float *v = values + (((b + i) * CODES_PER_BYTE) & 7u);",
                ),
            ),
            // The same term with both loads gone and the two multiply-adds left,
            // fed by the indices the loads were at.
            (
                "the input's loads",
                crate::testing::instead_of(
                    &shipped,
                    "        dot += element(low) * v[0] + element(high) * v[1];",
                    "        dot += element(low) * (float)b + element(high) * (float)i;",
                ),
            ),
            // The dequantisation, which is two gathers into a 16-entry constant
            // array per packed byte and the one term whose cost is a memory
            // access nobody counts.
            (
                "the table it decodes through",
                crate::testing::instead_of(
                    &shipped,
                    "        dot += element(low) * v[0] + element(high) * v[1];",
                    "        dot += (float)low * v[0] + (float)high * v[1];",
                ),
            ),
            // The group scale, which is a byte read and a shift per chunk.
            // Reading the row's first scale leaves the load in the kernel and
            // takes it out of the walk.
            (
                "the scale it walks to",
                crate::testing::instead_of(
                    &shipped,
                    "    return dot * as_type<float>(uint(scale[b / BYTES_PER_GROUP]) << \
                     EXPONENT_SHIFT);",
                    "    return dot * as_type<float>(uint(scale[0]) << EXPONENT_SHIFT);",
                ),
            ),
            // The cross-lane reduction, which is one per output element at the
            // end of a walk a thousand bytes long — and which a call with few
            // elements pays as often as a call with many.
            (
                "the cross-lane reduction",
                crate::testing::instead_of(
                    &shipped,
                    "    sum = simd_sum(sum);\n    if (lane == 0) {",
                    "    if (lane == 0) {",
                ),
            ),
        ]
    }

    /// What one untiled call answers, which is the oracle every arm of the decode
    /// limiter table has to disagree with.
    ///
    /// One row, so that [`tiles`] is false and the call reaches [`ENTRY`] — the
    /// same predicate a decode step's own projections go through.
    fn an_untiled_call_answers(device: &Device, matmul: &PackedMatmul) -> Vec<f32> {
        let case = Case::noisy(IN_DIM, OUT_DIM, 1);
        case.upload(device, matmul)
            .multiply(&case.x)
            .expect("the dispatch runs")
    }

    /// **What a decode step's packed matmul is bound by, one term at a time** —
    /// 70% of the device's time in a step, the largest untuned number in this
    /// engine, and a kernel this file has never pointed an instrument at.
    ///
    /// The prefill tables above are all about tiles, and a tile is what a decode
    /// step is refused: [`PackedMatmul::blocks`] excludes a call this short and
    /// [`tiles`] is false for every shape in [`DECODE_SHAPES`]. So what runs is
    /// [`ENTRY`], one simdgroup an output element, and the questions A4 asked of
    /// the tile have never been asked of it.
    ///
    /// **The two arms a term are what makes this readable.** A decode step's
    /// weight is compulsory — a token reads the model once — so an arm that takes
    /// the traffic off is not a lever, it is a measurement of how far the call is
    /// from being about its own bytes. The input is the opposite: it is 16 KB read
    /// once per output element, which is traffic this kernel invents and a tile
    /// would not.
    ///
    /// Every arm answers wrongly and the case asserts that it does.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_decode_steps_packed_matmul_is_bound_by() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let shipped = matmul(&device);
        let want = an_untiled_call_answers(&device, &shipped);
        let shapes: Vec<Blocked> = DECODE_BOUND_SHAPES
            .iter()
            .map(|&(what, rows, in_dim, out_dim, experts)| {
                Blocked::of(what, rows, in_dim, out_dim, experts, false)
            })
            .collect();

        eprintln!(
            "  {:<32}{}",
            "without",
            header(DECODE_BOUND_SHAPES.iter().map(|(what, ..)| *what), 22)
        );
        // **Warmed inside the arm rather than once before them all.** Compiling
        // a mutant and uploading its banks is a second or more with the device
        // idle, and an arm measured on the clock that leaves reads 70% slower
        // than the same arm measured warm — larger than any term this table
        // prices. See [`crate::testing::warmed`].
        let cost = |matmul: &PackedMatmul| -> Vec<Duration> {
            let banks: Vec<PackedBank<'_>> = shapes
                .iter()
                .map(|shape| shape.upload(&device, matmul))
                .collect();
            crate::testing::warmed(|| {
                shapes[0].costs_on(&device, &banks[0], &grouping);
            });
            shapes
                .iter()
                .zip(&banks)
                .map(|(shape, bank)| shape.costs_on(&device, bank, &grouping))
                .collect()
        };
        let whole = cost(&shipped);
        let row = |what: &str, taken: &[Duration]| {
            let cells: String = taken
                .iter()
                .zip(&whole)
                .map(|(each, whole)| {
                    format!(
                        "{:>14}{:>8}",
                        format!("{:.0}µs", 1e6 * each.as_secs_f64()),
                        format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                    )
                })
                .collect();
            eprintln!("  {what:<32}{cells}");
        };
        row("nothing — the kernel", &whole);

        for (what, written) in without_each_term_of_the_untiled_walk() {
            let mutant = PackedMatmul::from_source(&device, &written).expect("the mutant compiles");
            assert_ne!(
                an_untiled_call_answers(&device, &mutant),
                want,
                "{what}: the mutation answered what the kernel answers"
            );
            row(what, &cost(&mutant));
        }
    }

    /// The lengths a coding session opens at, which is where this milestone's
    /// figures are taken.
    ///
    /// **16384 is not here, and the reason is not the one it would have been
    /// before A8.** The argument used to be that attention is quadratic and the
    /// matmul is not, so a long prompt flatters the matmul's share into looking
    /// small — at 16384 the two attention rows were 64% of the passes and the
    /// two matmul rows 56%. `mma_attention` took the attention rows nineteen
    /// times down, and the in-model table now reads the two matmul rows at 88,
    /// 85, 83 and 80% of every pass a prefill runs at 2048, 4096, 8192 and
    /// 16384. **So no length flatters it any more**; the matmul is where a
    /// prefill's time is at all of them.
    ///
    /// What is left is the plainer reason: a coding session opens between these
    /// three, and a prefill here is seconds where 16384 is half a minute — so
    /// this is the range a claim about the workload has to be made in, and 16384
    /// is a check rather than a target.
    const BLOCKED_LENGTHS: [usize; 3] = [2048, 4096, 8192];

    /// The shipped production source with both staged tiles, and the fragments
    /// loaded out of them, carried as 16-bit floats.
    ///
    /// **The accumulator stays `simdgroup_float8x8` and that is the whole of
    /// what makes this arguable at all.** Metal's matrix instruction takes a
    /// mixed form on this family — half operands into a float accumulator — so
    /// what a 16-bit operand costs here is a rounding of the two *operands* and
    /// nothing about the summation order, which is already the instruction's.
    /// A half accumulator compiles too and is not what this is: 512 accumulate
    /// steps into a 10-bit significand is a different claim and a much worse
    /// one.
    ///
    /// **What is rounded is a product that used to be exact.** A code is one of
    /// sixteen table values and a group scale is a power of two, so `element ×
    /// by` is exact in f32 and every MXFP4 element is exactly representable in
    /// half — but their product carries eleven bits of significand where it had
    /// twenty-four, and that rounding is what the table below prices.
    ///
    /// `stride` is what a staged row is padded to, and it is a parameter here
    /// for one reason: [`MMA_STAGED_PADDING`]'s argument is derived
    /// against 32 banks of *four* bytes and a staged element that is four bytes
    /// wide. At two bytes the same 36 puts a fragment's eight rows on banks 0,
    /// 18, 4, 22, 8, 26, 12 and 30 — still eight distinct ones, so the argument
    /// survives — but "survives by arithmetic" is what this repo's own history
    /// says to distrust, and a second stride measured beside the first is what
    /// says the clock below is about the operand width rather than about a bank
    /// conflict.
    fn through_sixteen_bit_operands(shipped: &str, stride: usize) -> String {
        let staged = crate::testing::instead_of(
            shipped,
            "    threadgroup float staged[(MMA_ROWS_A_BLOCK + MMA_COLS_A_BLOCK) * \
             MMA_STAGED_STRIDE];\n    threadgroup float *staged_x = staged;\n    \
             threadgroup float *staged_w = staged + MMA_ROWS_A_BLOCK * MMA_STAGED_STRIDE;",
            // The scratch stays the width the answer wants and the two staged
            // tiles are read out of it narrow, so that what this arm changes is
            // the operand and not how many threadgroups a core holds.
            "    threadgroup float staged[(MMA_ROWS_A_BLOCK + MMA_COLS_A_BLOCK) * \
             MMA_STAGED_STRIDE];\n    threadgroup half *staged_x = (threadgroup half *)staged;\n  \
             \x20 threadgroup half *staged_w = staged_x + MMA_ROWS_A_BLOCK * MMA_STAGED_STRIDE;",
        );
        let staged = crate::testing::instead_of(
            &staged,
            &format!(
                "constant uint MMA_STAGED_STRIDE = {};",
                Block::SHIPPED.staged_stride()
            ),
            &format!("constant uint MMA_STAGED_STRIDE = {stride};"),
        );
        let filled = crate::testing::instead_of(
            &staged,
            "staged_x[x_row * MMA_STAGED_STRIDE + x_at + i] = live ? values[i] : 0.0f;",
            "staged_x[x_row * MMA_STAGED_STRIDE + x_at + i] = live ? (half)values[i] : 0.0h;",
        );
        let decoded = crate::testing::instead_of(
            &filled,
            "staged_w[at] = element(code & CODE_MASK) * by;\n                staged_w[at + 1] = \
             element((code >> BITS) & CODE_MASK) * by;",
            "staged_w[at] = (half)(element(code & CODE_MASK) * by);\n                staged_w[at + \
             1] = (half)(element((code >> BITS) & CODE_MASK) * by);",
        );
        crate::testing::instead_of(
            &decoded,
            "                simdgroup_float8x8 lhs[MMA_FRAGMENTS_DOWN];\n                \
             simdgroup_float8x8 rhs[MMA_FRAGMENTS_ACROSS];",
            "                simdgroup_half8x8 lhs[MMA_FRAGMENTS_DOWN];\n                \
             simdgroup_half8x8 rhs[MMA_FRAGMENTS_ACROSS];",
        )
    }

    /// Both of [`BOUND_SHAPES`] at a prompt of `tokens`, which is what a column
    /// of either of the block's tables is.
    fn shapes_at(tokens: usize) -> Vec<Blocked> {
        BOUND_SHAPES
            .iter()
            .map(|&shape| Blocked::at(tokens, shape))
            .collect()
    }

    /// The reductions the conditioning table below is taken across, which are
    /// `a_block_answers_the_reference_tile_where_neither_extent_divides_it`'s
    /// own — so a third column can be read straight down beside the two that
    /// case already records.
    const CONDITIONING_REDUCTIONS: [usize; 5] = [32, 128, 512, 2048, 4096];

    /// **What 16-bit operands cost and what they give up** — the lever Apple's
    /// guidance for this hardware is most emphatic about, and the one mlx's
    /// quantised matmul takes completely: the metallib in `reference/.venv` at
    /// mlx 0.32.0 carries `qmm` in `bfloat16` and `float16` and in no `float32`
    /// at all. **Its dense steel GEMM does ship fp32**, which is worth saying
    /// because this repo has asserted otherwise —
    /// `steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm1_wn2` is in the
    /// same binary.
    ///
    /// A8 declined it and named the reason to respect: "this flag's two sides
    /// sum the same exact products in different orders, and a 16-bit operand
    /// **rounds the product itself**. It would have to move into the
    /// conditioning table, not sit beside it." So it is here, in the
    /// conditioning table, with the clock beside it rather than instead of it.
    ///
    /// **What is given up is exactness of the operand, not of the sum.** The
    /// accumulator stays `simdgroup_float8x8` — the instruction takes that mixed
    /// form on this family — so the summation order is the one the block already
    /// has and nothing about the chain moves. What moves is that `element × by`
    /// stops being exact: it carries eleven bits of significand where it carried
    /// twenty-four.
    ///
    /// **Which is why the shape of the drift column is the finding and not its
    /// size.** The block's own drift grows with the reduction, because its
    /// fragment accumulator is one running sum and a longer chain is a worse
    /// one. A drift that is *flat* in the reduction is not a chain at all — it
    /// is the operand, arriving already rounded and staying that far off however
    /// few products are summed. That is the column this prints, and it is what
    /// separates "16-bit operands are worse here" from "this arm has a bug".
    ///
    /// **Both halves of the question in one case, because either alone would be
    /// misread.** A drift with no clock beside it cannot say whether the
    /// accuracy was worth anything, and a clock with no drift beside it is the
    /// claim A8 refused.
    ///
    /// The clock is warm and swept both ways, for the reason
    /// [`what_a_prefills_blocked_matmul_is_bound_by`] gives: the arms it
    /// separates are a few percent apart, and an arm measured always second is
    /// an arm measured on whatever clock the one before it left.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_sixteen_bit_operands_cost_and_what_they_give_up() {
        let Some(device) = device() else { return };

        let reference = matmul(&device);
        let shipped =
            PackedMatmul::under(&device, Numerics::Production).expect("the block compiles");
        // Three strides rather than one, and the reason turned out to matter.
        // The shipped padding is derived against a *four*-byte staged element;
        // at two bytes 36, 40 and 44 all land a fragment's eight rows on eight
        // distinct banks, so on the argument alone they should read alike. They
        // do not — which is what says a single-stride reading of this arm would
        // have been a measurement of the padding rather than of the operand
        // width, and why the refusal is stated against the best of them.
        let shipped_stride = Block::SHIPPED.staged_stride();
        let halved: Vec<(usize, PackedMatmul)> =
            [shipped_stride, shipped_stride + 4, shipped_stride + 8]
                .into_iter()
                .map(|stride| {
                    let arm = PackedMatmul::blocked_from_source(
                        &device,
                        &through_sixteen_bit_operands(
                            &source_blocked(Numerics::Production, Block::SHIPPED),
                            stride,
                        ),
                        Block::SHIPPED,
                    )
                    .expect("the 16-bit arm compiles");
                    (stride, arm)
                })
                .collect();

        // The conditioning table first, because it is what decides whether the
        // clock is worth reading at all. The padding reaches no arithmetic, so
        // one arm answers for both.
        eprintln!(
            "  {:<14}{:>16}{:>16}{:>18}",
            "a reduction", "the reference", "the block", "16-bit operands"
        );
        let mut worst = 0.0f64;
        for reduction in CONDITIONING_REDUCTIONS {
            let case = Case::noisy(reduction, OUT_DIM, 77);
            let exact = case.exactly();
            let through = |matmul: &PackedMatmul| {
                drift(
                    &case
                        .upload(&device, matmul)
                        .multiply(&case.x)
                        .expect("the dispatch completes"),
                    &exact,
                )
            };
            let (theirs, block, half) = (
                through(&reference),
                through(&shipped),
                through(&halved[0].1),
            );
            worst = worst.max(half);
            eprintln!(
                "  {reduction:<14}{:>16}{:>16}{:>18}",
                format!("{theirs:.1e}"),
                format!("{block:.1e}"),
                format!("{half:.1e}")
            );
        }

        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        eprintln!(
            "  {:<8}{:<24}{:>16}{:>14}{:>10}{:>10}",
            "at",
            "",
            "the block",
            format!("16-bit at {}", halved[0].0),
            format!("at {}", halved[1].0),
            format!("at {}", halved[2].0)
        );
        let mut slowest = f64::MAX;
        for tokens in BLOCKED_LENGTHS {
            let shapes = shapes_at(tokens);
            let arms: Vec<&PackedMatmul> = std::iter::once(&shipped)
                .chain(halved.iter().map(|(_, arm)| arm))
                .collect();
            crate::testing::warmed(|| {
                shapes[0].costs(&device, &shipped, &grouping);
            });
            let listed: Vec<usize> = (0..arms.len()).collect();
            let (up, down) = crate::testing::both_ways(&listed, |at| {
                shapes
                    .iter()
                    .map(|shape| shape.costs(&device, arms[at], &grouping))
                    .collect::<Vec<Duration>>()
            });

            for (shape, at) in shapes.iter().zip(0..) {
                // The slower of the two passes per arm, so that what is reported
                // is not whichever direction the clock happened to favour.
                let taken = |arm: usize| up[arm][at].max(down[arm][at]).as_secs_f64();
                let block = taken(0);
                let against = |arm: usize| taken(arm) / block;
                slowest = (1..arms.len()).fold(slowest, |least, arm| least.min(against(arm)));
                eprintln!(
                    "  {tokens:<8}{:<24}{:>16}{:>14}{:>10}{:>10}",
                    shape.what,
                    format!("{:.2}ms", 1e3 * block),
                    format!("{:+.0}%", 1e2 * (against(1) - 1.0)),
                    format!("{:+.0}%", 1e2 * (against(2) - 1.0)),
                    format!("{:+.0}%", 1e2 * (against(3) - 1.0))
                );
            }
        }

        // **Both halves of the refusal are asserted, because either could turn
        // on its own and the table would still print.** The drift bound is the
        // one with teeth; the clock bound is loose on purpose — the best padding
        // reads one to three percent the wrong way, and what this refuses is a
        // reading that has changed sign by more than any noise on this host.
        assert!(
            worst > f64::from(MMA_TOLERANCE),
            "16-bit operands drift {worst:e}, which is inside the block's own {MMA_TOLERANCE:e} — \
             the reason this arm is refused has changed and the table above wants re-reading"
        );
        assert!(
            slowest > 0.95,
            "16-bit operands came back {:.0}% of the block at some shape, so the clock half of \
             this refusal has changed and the table above wants re-reading",
            1e2 * slowest
        );
    }

    /// **What the block's threadgroup is worth** — the last of the four levers
    /// this milestone opened with, and the only one nothing in this repo had
    /// ever tried.
    ///
    /// mlx-vlm runs its steel GEMM at **64 to 128 threads** where this ships
    /// **256** — read off the metallib this repo's own oracle runs rather than
    /// inherited, which is how it had stood since A8: mlx 0.32.0 carries
    /// `steel_gemm_fused_*_bm64_bn64_bk16_wm1_wn2` and `_wm2_wn2`, and
    /// `steel/gemm/mma.h` puts `WM × WN` simdgroups of 32 over the tile, which
    /// is 64 threads and 128. The same header is where the 2.7:1 comes from:
    /// `TM = BM / (8 × WM)` and `TN = BN / (8 × WN)` are 8 and 4 at that shape,
    /// so 32 accumulators against `TM + TN` = 12 fragment loads.
    ///
    /// The reason it was never swept is not that it looked
    /// unpromising: the width is a host-side constant as well as a source one,
    /// so an entry compiled at one width and dispatched over a grid sized for
    /// another leaves output no threadgroup reached — a wrong answer rather
    /// than a slow one. [`Block`] is what made the question askable, and
    /// `a_block_cut_to_another_shape_answers_what_the_shipped_one_answers` is
    /// what says every arm here computes the same bits.
    ///
    /// **A narrower threadgroup is also more fragment reuse, which is why this
    /// is one sweep and not two.** At 256 threads a simdgroup holds two
    /// fragments down by two across — four accumulators against four loads,
    /// which is the 1:1 that mlx-vlm's 64×64 tile beats at 2.7:1. At 128 it is
    /// two by four, which is eight accumulators against six loads; at 64, four
    /// by four, sixteen against eight. So the width walks the reuse ratio from
    /// 1:1 to 2:1 without a taller block and without the doubled floor one
    /// would cost — **and the reuse column is printed beside the clock because
    /// the two run in opposite directions**, which is the finding.
    ///
    /// The reason they do is that a block's threadgroup memory is a property of
    /// the block and not of its threads. Two staged tiles and an answer tile
    /// come to about 22 KiB whatever covers them, so a core holding 32 KiB holds
    /// one threadgroup either way — and at 64 threads that is two simdgroups a
    /// core against eight, with the same memory and a quarter of the work to
    /// hide a load behind.
    ///
    /// Warm and swept both ways, for the reason
    /// [`what_a_prefills_blocked_matmul_is_bound_by`] gives.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_blocks_threadgroup_is_worth() {
        let Some(device) = device() else { return };

        distinct(&SWEPT_BLOCKS);
        let arms: Vec<(Block, PackedMatmul)> = SWEPT_BLOCKS
            .iter()
            .filter_map(|&block| a_block_of(&device, block).map(|matmul| (block, matmul)))
            .collect();
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");

        for tokens in BLOCKED_LENGTHS {
            let shapes = shapes_at(tokens);
            crate::testing::warmed(|| {
                shapes[0].costs(&device, &arms[0].1, &grouping);
            });
            let listed: Vec<usize> = (0..arms.len()).collect();
            let (up, down) = crate::testing::both_ways(&listed, |at| {
                shapes
                    .iter()
                    .map(|shape| shape.costs(&device, &arms[at].1, &grouping))
                    .collect::<Vec<Duration>>()
            });

            eprintln!("\n  a prefill of {tokens} tokens");
            eprintln!(
                "  {:<34}{:>10}{}",
                "a threadgroup",
                "reuse",
                shapes
                    .iter()
                    .map(|shape| format!("{:>26}", shape.what))
                    .collect::<String>()
            );
            let shipped = arms
                .iter()
                .position(|(block, _)| *block == Block::SHIPPED)
                .expect("the shipped shape is one of the arms");
            for (at, (block, _)) in arms.iter().enumerate() {
                // The slower of the two passes, so a row is not whichever
                // direction the clock happened to favour.
                let taken = |arm: usize, column: usize| up[arm][column].max(down[arm][column]);
                let cells: String = (0..shapes.len())
                    .map(|column| {
                        let (each, whole) = (taken(at, column), taken(shipped, column));
                        format!(
                            "{:>17}{:>9}",
                            format!("{:.2}ms", 1e3 * each.as_secs_f64()),
                            format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                        )
                    })
                    .collect();
                let loads = block.fragments_down() + block.fragments_across();
                eprintln!(
                    "  {:<34}{:>10}{cells}",
                    format!(
                        "{} threads, {}x{} simdgroups",
                        block.threads, block.simds_down, block.simds_across
                    ),
                    format!(
                        "{:.1}:1",
                        (block.fragments_down() * block.fragments_across()) as f64 / loads as f64
                    ),
                );
            }
        }
    }

    /// **What a wider step is worth, which is the barrier count bought at the
    /// occupancy** — the first of the two levers T2 priced off the instruction
    /// mix rather than off a build.
    ///
    /// A step is `MMA_CODES_A_STEP` codes wide and carries two barriers, so a
    /// reduction of 4096 is 128 steps and **256 barriers a pass**, every one of
    /// which stops all eight simdgroups until the slowest arrives.
    /// [`what_a_prefills_blocked_matmul_is_bound_by`]'s barrier arm — the one
    /// that removes them and races — prices the whole term at **10%**, and a step
    /// of two groups halves the count, so **5% is what this can return at its
    /// ceiling.**
    ///
    /// **And it is bought rather than free, which is why the declaration is
    /// printed beside the clock.** The staging is the step wide, so doubling it
    /// doubles the threadgroup memory: 13.5 KiB to 25.5, which by `RESIDENCY`'s
    /// sweep of what a core holds takes the block from five threadgroups a core
    /// to three. **The two effects run opposite ways**, which is what makes this
    /// a sweep and not a change — and what the table says is which of them is
    /// larger.
    ///
    /// **And the third row is what says which of the two the table is reading.**
    /// It is the shipped step — the same barriers, the same loop, the same bits —
    /// declaring the wider step's threadgroup memory and never touching the extra.
    /// That is [`RESIDENCY`]'s own trick pointed the other way: what a
    /// threadgroup declares decides how many of them a core holds, so an arm that
    /// declares 25.5 KiB and steps 32 codes carries the wider step's occupancy
    /// and none of its benefit. **A row that lands where the 64-code row lands
    /// says the whole difference is the declaration**, and one that lands at 100%
    /// says it is not.
    ///
    /// Warm and swept both ways, for the reason
    /// [`what_a_prefills_blocked_matmul_is_bound_by`] gives. Nothing asserts a
    /// rate; what is asserted is that the shipped step was among the arms, so a
    /// table that measured only the alternative could not read as a comparison.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_step_its_staging_brings_in() {
        let Some(device) = device() else { return };

        distinct(&SWEPT_STEPS);
        let mut arms: Vec<(String, usize, PackedMatmul)> = SWEPT_STEPS
            .iter()
            .filter_map(|&block| {
                a_block_of(&device, block).map(|matmul| {
                    (
                        format!("{} codes", block.step),
                        block.staged_bytes(),
                        matmul,
                    )
                })
            })
            .collect();
        let shipped = arms
            .iter()
            .position(|(what, ..)| *what == format!("{} codes", Block::SHIPPED.step))
            .expect("the shipped step is one of the arms");

        let widened = SWEPT_STEPS
            .iter()
            .map(Block::staged_bytes)
            .max()
            .expect("the sweep has an arm");
        let control = PackedMatmul::blocked_from_source(
            &device,
            &declaring(
                &source_blocked(Numerics::Production, Block::SHIPPED),
                widened,
            ),
            Block::SHIPPED,
        )
        .expect("the widened declaration compiles");
        // **A control that answered other bits would be a second change rather
        // than a declaration**, and the whole of what this row is for is that it
        // is the shipped kernel with one number in it.
        assert_eq!(
            a_tiled_call_answers(&device, &control),
            a_tiled_call_answers(&device, &arms[shipped].2),
            "the widened declaration answered other bits than the step it was cut from"
        );
        arms.push((
            format!("{} codes, declaring the wider", Block::SHIPPED.step),
            widened,
            control,
        ));

        // **And a racing arm at each step, which is what says whether a barrier
        // costs by the count or by the work between two of them.** These are
        // [`without_the_steps_barriers`] — the same arm
        // [`what_a_prefills_blocked_matmul_is_bound_by`] prices the whole term
        // at — asked once per step width. **They answer nothing in particular**,
        // because without the barriers a simdgroup reads a tile another is still
        // filling; what they cost is the step loop with the same loads, the same
        // decode and the same multiply-accumulates and none of the waiting. Two
        // rows that give back the same share are a term that is the staging's
        // imbalance rather than the barrier's count, and a step that halves the
        // count cannot touch it.
        for block in SWEPT_STEPS {
            let racing = PackedMatmul::blocked_from_source(
                &device,
                &without_the_steps_barriers(&source_blocked(Numerics::Production, block)),
                block,
            )
            .expect("the racing arm compiles");
            arms.push((
                format!("{} codes, racing", block.step),
                block.staged_bytes(),
                racing,
            ));
        }

        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");

        for tokens in BLOCKED_LENGTHS {
            let shapes = shapes_at(tokens);
            crate::testing::warmed(|| {
                shapes[0].costs(&device, &arms[shipped].2, &grouping);
            });
            let listed: Vec<usize> = (0..arms.len()).collect();
            let (up, down) = crate::testing::both_ways(&listed, |at| {
                shapes
                    .iter()
                    .map(|shape| shape.costs(&device, &arms[at].2, &grouping))
                    .collect::<Vec<Duration>>()
            });

            eprintln!("\n  a prefill of {tokens} tokens");
            eprintln!(
                "  {:<32}{:>12}{}",
                "a step of",
                "declares",
                shapes
                    .iter()
                    .map(|shape| format!("{:>26}", shape.what))
                    .collect::<String>()
            );
            for (at, (what, declared, _)) in arms.iter().enumerate() {
                // The slower of the two passes, so a row is not whichever
                // direction the clock happened to favour.
                let taken = |arm: usize, column: usize| up[arm][column].max(down[arm][column]);
                let cells: String = (0..shapes.len())
                    .map(|column| {
                        let (each, whole) = (taken(at, column), taken(shipped, column));
                        format!(
                            "{:>17}{:>9}",
                            format!("{:.2}ms", 1e3 * each.as_secs_f64()),
                            format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                        )
                    })
                    .collect();
                eprintln!(
                    "  {what:<32}{:>12}{cells}",
                    format!("{:.1} KiB", *declared as f64 / 1024.0),
                );
            }
        }
    }

    /// The shipped production source with the staged declaration widened to
    /// `bytes` and every index into it left where it was, so what the arm changes
    /// is how many threadgroups a core holds and nothing a thread reads.
    ///
    /// Written as a count of floats past the two tiles rather than as a whole new
    /// expression, so that the arm carries the shipped declaration in it and a
    /// change to the staging would not quietly leave this measuring a shape the
    /// kernel no longer has.
    fn declaring(shipped: &str, bytes: usize) -> String {
        let staged = "threadgroup float staged[(MMA_ROWS_A_BLOCK + MMA_COLS_A_BLOCK) * \
                      MMA_STAGED_STRIDE";
        let spare = bytes / size_of::<f32>() - Block::SHIPPED.staged_bytes() / size_of::<f32>();
        crate::testing::instead_of(shipped, staged, &format!("{staged} + {spare}"))
    }

    /// **What reading a fragment per lane is worth** — the second of the two
    /// levers T2 priced off the instruction mix, and the one structural
    /// difference between this inner loop and the quantised kernel mlx runs at
    /// these same shapes.
    ///
    /// `gather_qmm`'s `BaseMMAFrag::load` never calls `simdgroup_load`: it reads
    /// the two elements each lane owns straight out of threadgroup memory through
    /// a lane-to-coordinate map, and takes the transpose by swapping the two
    /// strides it indexes with. This kernel issues a `simdgroup_load` a fragment
    /// and a *transposing* one for the weight, and
    /// [`what_a_prefills_blocked_matmul_is_bound_by`]'s hoisting arm — which
    /// leaves three of every four loads unissued and answers wrongly for it —
    /// prices all of them together at **6%**. So 6% is the whole of what this can
    /// return, and only if the two instructions cost nothing at all.
    ///
    /// **The transposing load is the one that was expected to pay.** It is the
    /// one most likely to be doing more than the two element reads it owes, since
    /// the per-lane form expresses the same transpose as an index rather than as
    /// an instruction.
    ///
    /// Warm and swept both ways, and every arm answers the shipped kernel bit for
    /// bit — which it must, since what moves is where a fragment's two elements
    /// come from and not what they are.
    /// [`the_lane_map_a_fragment_is_read_through_is_this_parts_own`] is what says
    /// the map is this part's rather than a header's.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_packed_multiply_costs_at_each_way_it_reads_a_fragment() {
        let Some(device) = device() else { return };

        distinct(&SWEPT_FRAGMENTS);
        let arms: Vec<(Block, PackedMatmul)> = SWEPT_FRAGMENTS
            .iter()
            .filter_map(|&block| a_block_of(&device, block).map(|matmul| (block, matmul)))
            .collect();
        let shipped = arms
            .iter()
            .position(|(block, _)| *block == Block::SHIPPED)
            .expect("the shipped read is one of the arms");
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");

        for tokens in BLOCKED_LENGTHS {
            let shapes = shapes_at(tokens);
            crate::testing::warmed(|| {
                shapes[0].costs(&device, &arms[shipped].1, &grouping);
            });
            let listed: Vec<usize> = (0..arms.len()).collect();
            let (up, down) = crate::testing::both_ways(&listed, |at| {
                shapes
                    .iter()
                    .map(|shape| shape.costs(&device, &arms[at].1, &grouping))
                    .collect::<Vec<Duration>>()
            });

            eprintln!("\n  a prefill of {tokens} tokens");
            eprintln!(
                "  {:<38}{}",
                "a fragment read",
                shapes
                    .iter()
                    .map(|shape| format!("{:>26}", shape.what))
                    .collect::<String>()
            );
            for (at, (block, _)) in arms.iter().enumerate() {
                // The slower of the two passes, so a row is not whichever
                // direction the clock happened to favour.
                let taken = |arm: usize, column: usize| up[arm][column].max(down[arm][column]);
                let cells: String = (0..shapes.len())
                    .map(|column| {
                        let (each, whole) = (taken(at, column), taken(shipped, column));
                        format!(
                            "{:>17}{:>9}",
                            format!("{:.2}ms", 1e3 * each.as_secs_f64()),
                            format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                        )
                    })
                    .collect();
                eprintln!(
                    "  {:<38}{cells}",
                    match block.fragments {
                        Fragments::Loaded => "through simdgroup_load",
                        Fragments::PerLane => "the two elements a lane owns",
                    }
                );
            }
        }
    }

    /// **What each shape the block is swept at costs to compile**, which is the
    /// price a lever pays before any call reaches it.
    ///
    /// **D1 is why this is priced at all.** Merely reordering which of this
    /// source's entries are created first was worth 2.4% of a 2048-token
    /// prefill's device time — a fixed cost this side had not thought to have —
    /// so a shape that made the entries longer to build owes the same figure, and
    /// the more so at a short prompt where a fixed cost is a larger share.
    ///
    /// **Neither of the two production entries is added or removed by any arm
    /// here**, which is the first thing the count column says: every row builds
    /// the same three pipelines a model load builds, out of a source that differs
    /// in the block's own prelude. So what a row prices is a longer or shorter
    /// body rather than a fourth kernel, and a row that grew a pipeline would
    /// show in the count rather than only in the clock.
    ///
    /// **The clock is a warm compile and says so.** The first build in a process
    /// carries the compiler's own start-up and this part's shader cache is cold
    /// for it, so what every row here is timed behind is one shipped build — which
    /// is the arrangement a model load is in for its second entry and after, and
    /// not the one it is in for its first.
    ///
    /// Nothing asserts a duration. What is asserted is the count, which is the
    /// part of this that is a fact rather than a measurement.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_each_shape_the_block_is_swept_at_costs_to_compile() {
        let Some(device) = device() else { return };

        // The first compile a process runs carries the compiler's own start-up
        // and reads several times what the ones behind it do, which would land
        // the whole of it on whichever arm this loop happened to try first.
        PackedMatmul::under(&device, Numerics::Production).expect("the shipped source compiles");

        eprintln!(
            "\n  {:<44}{:>12}{:>12}{:>12}",
            "a block of", "entries", "compiling", "the source"
        );
        // Deduplicated, because the shipped shape is an arm of both sweeps and
        // a row printed three times reads as three measurements of nothing.
        let mut swept: Vec<Block> = Vec::new();
        for block in SWEPT_STEPS.into_iter().chain(SWEPT_FRAGMENTS) {
            if !swept.contains(&block) {
                swept.push(block);
            }
        }
        for block in swept {
            let source = source_blocked(Numerics::Production, block);
            let taken = Instant::now();
            let Ok(matmul) = PackedMatmul::blocked(&device, Numerics::Production, block) else {
                eprintln!("  {block:?} is refused by this part");
                continue;
            };
            let whole = taken.elapsed();
            let mma = matmul
                .mma
                .as_ref()
                .expect("the production entries compiled");
            let entries = 1 + usize::from(mma.tiled.entry() != mma.grouped.entry());
            assert_eq!(
                entries, MMA_ENTRIES,
                "a block of {block:?} compiled {entries} production entries where a model load \
                 builds {MMA_ENTRIES}"
            );
            eprintln!(
                "  {:<44}{:>12}{:>12}{:>12}",
                format!(
                    "{} codes a step, {}",
                    block.step,
                    match block.fragments {
                        Fragments::Loaded => "through simdgroup_load",
                        Fragments::PerLane => "the elements a lane owns",
                    }
                ),
                entries,
                format!("{:.0} us", 1e6 * whole.as_secs_f64()),
                format!("{:.1} KiB", source.len() as f64 / 1024.0),
            );
        }
    }

    /// Production entries a model load builds, which is the pair
    /// [`PackedMatmul::compiled`] creates and no arm of any sweep changes.
    const MMA_ENTRIES: usize = 2;

    /// **What a table here reports has to be one dispatch of the shape it
    /// names**, and the way it stops being one is silent.
    ///
    /// [`crate::testing::device_time`] takes a count of dispatches and returns
    /// what one of them cost, so a caller that divides by its own count again
    /// reports every arm of every table four times faster than it ran. **Nothing
    /// in such a table looks wrong**: each arm is low by the same four, so the
    /// ratios a sweep is read for are exactly right and only the absolute
    /// figures — the ones a roofline and a cross-engine column divide by — are
    /// not.
    ///
    /// Asked against a single dispatch measured on its own, and asked at a shape
    /// small enough to be a case rather than a sitting. The bound is loose
    /// because a dispatch shares a command buffer with three others in one arm
    /// and has one to itself in the other; what it refuses is a factor.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn a_blocks_table_reports_one_dispatch_of_the_shape_it_names() {
        let Some(device) = device() else { return };
        let shipped =
            PackedMatmul::under(&device, Numerics::Production).expect("the block compiles");
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");

        // The tiled shape at a narrow output, which is the smallest call that
        // still reaches the block: many rows, one expert, a fixture of a
        // megabyte rather than a gigabyte.
        let shape = Blocked::of("q_proj, tiled", 2048, IN_DIM, 256, 1, false);
        let bank = shape.upload(&device, &shipped);
        let mut x = device.buffer(&shape.case.x).expect("the rows upload");
        crate::testing::warmed(|| {
            shape.costs(&device, &shipped, &grouping);
        });

        let mut alone = Duration::MAX;
        for _ in 0..5 {
            alone = alone.min(crate::testing::device_time(&device, 1, |batch| {
                shape.encode(batch, &device, &bank, &mut x, &grouping);
            }));
        }
        let table = shape.costs(&device, &shipped, &grouping);

        let ratio = table.as_secs_f64() / alone.as_secs_f64();
        assert!(
            (0.75..1.35).contains(&ratio),
            "the table reads {table:.2?} where one dispatch on its own reads {alone:.2?}, which is \
             {ratio:.2}× — so what every block table in this file reports is not a dispatch of the \
             shape its row names"
        );
    }

    /// The matrix instruction and the scalar one, each over registers alone.
    ///
    /// **What a ceiling has to have none of is memory.** Every entry loads its
    /// operands once, before the loop, and none reads or writes anything inside
    /// it — so what the clock divides is the issue rate of the instruction and
    /// nothing around it. The stores are under `at == 0xffffffffu`, which no
    /// thread of any grid dispatched here satisfies and no compiler can prove
    /// it does not: `thread_position_in_grid` is settled at the dispatch and
    /// not at the compile.
    ///
    /// **Each accumulator lands somewhere of its own**, which is the difference
    /// between four chains and one. Four `simdgroup_store`s to one address are
    /// three dead stores, and a dead store is licence to drop the chain that
    /// fed it — so a ceiling written that way could quietly be the single-chain
    /// figure under the four-chain name.
    ///
    /// `mma_held` carries the shipped block's own inner shape — four
    /// accumulators against two `lhs` and two `rhs` fragments — so the ceiling is
    /// the one this kernel could reach rather than the one some other
    /// arrangement could. `mma_chained` is the same loop down to a single
    /// accumulator, which is what the kernel reads as if every multiply waits on
    /// the last: the gap between the two is what independent accumulators are
    /// worth and it is the only thing here a block shape can change.
    const CEILING: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mma_held(
    device const float *seed [[buffer(0)]],
    constant uint &rounds [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint at [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8, 8>(seed[lane]);
    simdgroup_float8x8 a1 = make_filled_simdgroup_matrix<float, 8, 8>(seed[lane] + 1.0f);
    simdgroup_float8x8 b0 = make_filled_simdgroup_matrix<float, 8, 8>(seed[lane] + 2.0f);
    simdgroup_float8x8 b1 = make_filled_simdgroup_matrix<float, 8, 8>(seed[lane] + 3.0f);
    simdgroup_float8x8 s00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 s01 = s00, s10 = s00, s11 = s00;

    for (uint r = 0; r < rounds; ++r) {
        simdgroup_multiply_accumulate(s00, a0, b0, s00);
        simdgroup_multiply_accumulate(s01, a0, b1, s01);
        simdgroup_multiply_accumulate(s10, a1, b0, s10);
        simdgroup_multiply_accumulate(s11, a1, b1, s11);
    }

    if (at == 0xffffffffu) {
        simdgroup_store(s00, out, 8);
        simdgroup_store(s01, out + 64, 8);
        simdgroup_store(s10, out + 128, 8);
        simdgroup_store(s11, out + 192, 8);
    }
}

kernel void mma_chained(
    device const float *seed [[buffer(0)]],
    constant uint &rounds [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint at [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8, 8>(seed[lane]);
    simdgroup_float8x8 b0 = make_filled_simdgroup_matrix<float, 8, 8>(seed[lane] + 2.0f);
    simdgroup_float8x8 s00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint r = 0; r < rounds; ++r) {
        simdgroup_multiply_accumulate(s00, a0, b0, s00);
    }

    if (at == 0xffffffffu) {
        simdgroup_store(s00, out, 8);
    }
}

kernel void mma_half_held(
    device const float *seed [[buffer(0)]],
    constant uint &rounds [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint at [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8, 8>((half)seed[lane]);
    simdgroup_half8x8 a1 = make_filled_simdgroup_matrix<half, 8, 8>((half)seed[lane] + 1.0h);
    simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8, 8>((half)seed[lane] + 2.0h);
    simdgroup_half8x8 b1 = make_filled_simdgroup_matrix<half, 8, 8>((half)seed[lane] + 3.0h);
    simdgroup_float8x8 s00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 s01 = s00, s10 = s00, s11 = s00;

    for (uint r = 0; r < rounds; ++r) {
        simdgroup_multiply_accumulate(s00, a0, b0, s00);
        simdgroup_multiply_accumulate(s01, a0, b1, s01);
        simdgroup_multiply_accumulate(s10, a1, b0, s10);
        simdgroup_multiply_accumulate(s11, a1, b1, s11);
    }

    if (at == 0xffffffffu) {
        simdgroup_store(s00, out, 8);
        simdgroup_store(s01, out + 64, 8);
        simdgroup_store(s10, out + 128, 8);
        simdgroup_store(s11, out + 192, 8);
    }
}

kernel void scalar_held(
    device const float *seed [[buffer(0)]],
    constant uint &rounds [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint at [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    float4 a = float4(seed[lane]);
    float4 b = float4(seed[lane] + 1.0f);
    float4 c0 = float4(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (uint r = 0; r < rounds; ++r) {
        c0 = fma(a, b, c0);
        c1 = fma(a, b, c1);
        c2 = fma(a, b, c2);
        c3 = fma(a, b, c3);
    }
    if (at == 0xffffffffu) {
        out[0] = dot(c0 + c1, c2 + c3);
    }
}
"#;

    /// The four ceiling entries, with the multiply-adds one round of each
    /// performs per simdgroup: an 8×8×8 fragment is 512 of them, and a `float4`
    /// fused multiply-add is four across each of 32 lanes.
    const CEILING_ENTRIES: [(&str, &str, usize); 4] = [
        ("the matrix instruction, four held", "mma_held", 4 * 512),
        ("the same chain held once", "mma_chained", 512),
        ("the same four on 16-bit operands", "mma_half_held", 4 * 512),
        ("scalar fused multiply-adds", "scalar_held", 4 * 4 * 32),
    ];

    /// Rounds of the ceiling loop one dispatch runs, chosen so that the dispatch
    /// is tens of milliseconds — the same decade as the calls it is a ceiling
    /// for, so neither is being read at a resolution the other is not.
    const CEILING_ROUNDS: u32 = 20_000;

    /// Simdgroups a ceiling dispatch runs, which is eight threadgroups of 256 on
    /// each of this part's 80 cores — enough that the part is covered several
    /// times over and no core is idle at the end.
    const CEILING_THREADS: usize = 640 * 256;

    /// **What the matrix instruction issues at on this part, and what the block
    /// reaches against it.**
    ///
    /// Every "of peak" figure this file has ever recorded about the matmul
    /// divides by a *bandwidth*: `what_a_streaming_read_achieves_on_this_machine`
    /// measures 725 GB/s and the roofline in
    /// [`what_a_prefills_blocked_matmul_is_bound_by`] reads the compulsory
    /// traffic against it. **That denominator says nothing about a kernel this
    /// far from memory.** The block moves 2.35 GB in 54 ms, which is 43 GB/s —
    /// six percent of what the bus does — so what bounds it is on the other side
    /// of the arithmetic, and the arithmetic has a ceiling of its own that
    /// nothing here had measured.
    ///
    /// **It is not a specification figure.** This part is quoted at about 28
    /// TFLOP/s of fp32 fused multiply-add, and what a `simdgroup_matrix` op does
    /// against that is a question the vendor does not answer: the instruction
    /// could be a wide datapath of its own, in which case a kernel carrying 512
    /// multiply-adds an instruction has a much higher ceiling than a scalar one,
    /// or it could be the same lanes under another name. The scalar row is what
    /// answers it.
    ///
    /// **The assertion is that the block is under its own ceiling**, which is
    /// weak as a claim about the kernel and strong as one about the clock: a
    /// dispatch reading above the rate its own instruction issues at is a
    /// measurement error, and it is exactly the measurement error
    /// `the_device_clock_counts_every_dispatch_it_is_given` was written for.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_the_matrix_instruction_issues_at() {
        let Some(device) = device() else { return };

        let seed: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let mut seed = device.buffer(&seed).expect("the seed uploads");
        let mut rounds = device.buffer(&[CEILING_ROUNDS]).expect("the count uploads");
        let mut out = device.zeroed::<f32>(4 * 64).expect("the sink allocates");
        let grid = Grid::new(CEILING_THREADS, THREADS_PER_GROUP);

        let kernels: Vec<Kernel> = CEILING_ENTRIES
            .iter()
            .map(|(_, entry, _)| {
                device
                    .compile(CEILING, entry)
                    .expect("the ceiling kernel compiles")
            })
            .collect();
        let mut taken_by = |kernel: &Kernel| {
            crate::testing::device_time(&device, 3, |batch| {
                batch
                    .add(kernel, &[seed.arg(), rounds.arg(), out.arg()], grid, 0)
                    .expect("the ceiling dispatch encodes");
            })
        };
        // A ceiling read cold is one taken inside this part's boost window
        // rather than at the clock a prefill runs at, and three of these four
        // arms sit within a tenth of each other — which is inside what a ramp
        // over a four-row sweep manufactures.
        crate::testing::warmed(|| {
            taken_by(&kernels[0]);
        });

        let mut rates = Vec::new();
        eprintln!("\n  {:<38}{:>10}{:>16}", "what issues", "device", "rate");
        for ((what, _, per_round), kernel) in CEILING_ENTRIES.iter().zip(&kernels) {
            let mut taken = Duration::MAX;
            for _ in 0..5 {
                taken = taken.min(taken_by(kernel));
            }
            let simds = (CEILING_THREADS / NARROWEST_SIMD) as f64;
            let flops = 2.0 * simds * f64::from(CEILING_ROUNDS) * *per_round as f64;
            let rate = flops / taken.as_secs_f64();
            rates.push(rate);
            eprintln!(
                "  {what:<38}{:>10}{:>16}",
                format!("{:.2}ms", 1e3 * taken.as_secs_f64()),
                format!("{:.1} TFLOP/s", 1e-12 * rate),
            );
        }
        let ceiling = rates[0];

        // **The four rows are one rate, and that is both the finding and the
        // check.** A part whose matrix instruction were a datapath of its own
        // would put the first row a multiple above the last, and a loop that
        // collapsed to a closed form — the scalar one is the reassociable
        // shape — would put its own row orders above the rest. Neither is what
        // an issue-limited part reading one rate four ways looks like.
        let (fastest, slowest) = rates.iter().fold((0.0f64, f64::MAX), |(hi, lo), rate| {
            (hi.max(*rate), lo.min(*rate))
        });
        assert!(
            fastest / slowest < 2.0,
            "the four ways of asking this part for arithmetic span {:.1}× ({:.1} to {:.1} \
             TFLOP/s), where they have read within a quarter of each other — either something \
             here is issuing on a datapath of its own or one of these loops is no longer being \
             run, and the ceiling below divides by the first of them",
            fastest / slowest,
            1e-12 * slowest,
            1e-12 * fastest,
        );

        let shipped =
            PackedMatmul::under(&device, Numerics::Production).expect("the block compiles");
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        for tokens in BLOCKED_LENGTHS {
            let shapes = shapes_at(tokens);
            crate::testing::warmed(|| {
                shapes[0].costs(&device, &shipped, &grouping);
            });
            eprintln!("\n  a prefill of {tokens} tokens");
            eprintln!(
                "  {:<38}{:>10}{:>16}{:>22}",
                "", "device", "achieved", "of the instruction"
            );
            for shape in &shapes {
                let taken = shape.costs(&device, &shipped, &grouping);
                let rate = shape.flops() / taken.as_secs_f64();
                assert!(
                    rate < ceiling,
                    "{} at {tokens} tokens reads {:.1} TFLOP/s against a matrix instruction that \
                     issues at {:.1} — a kernel cannot outrun the instruction it is made of, so \
                     the clock is measuring something other than this dispatch",
                    shape.what,
                    1e-12 * rate,
                    1e-12 * ceiling,
                );
                eprintln!(
                    "  {:<38}{:>10}{:>16}{:>22}",
                    shape.what,
                    format!("{:.2}ms", 1e3 * taken.as_secs_f64()),
                    format!("{:.1} TFLOP/s", 1e-12 * rate),
                    format!("{:.0}%", 1e2 * rate / ceiling),
                );
            }
        }
    }

    /// **Whether the matmul is bandwidth-bound now that it is 2.85× faster, and
    /// what it is waiting on if it is not** — the first question of this
    /// milestone, because every closed door in this repo was closed by a
    /// measurement on the kernel this one replaced.
    ///
    /// A3 priced the expert weight's 96-fold re-reading at 10.4%, A3's query
    /// block at 23%, A4's whole limiter table: **all of them on the reference
    /// tile.** The block is 2.85× faster than that tile, and a kernel that was
    /// issue-bound can become bandwidth-bound when it gets faster — at which
    /// moment every byte-count lever those findings retired comes back.
    ///
    /// Two readings, and they answer different questions:
    ///
    /// - **The roofline**, which is [`Blocked::compulsory`] over the device's own
    ///   clock against the 725 GB/s `what_a_streaming_read_achieves_on_this
    ///   _machine` measures. This says how far the dispatch is from a kernel
    ///   that fetched each byte once — the floor no arrangement of this
    ///   arithmetic gets under.
    /// - **The limiter arms**, which say what it is actually waiting on. The
    ///   roofline alone cannot separate a kernel bound by traffic it must move
    ///   from one bound by traffic it re-reads, and those two have opposite
    ///   consequences for the rest of this brief: the first closes the door on
    ///   fragment reuse and a taller block, the second is exactly what they buy.
    ///
    /// **Every arm answers wrongly and the case asserts that it does**, the way
    /// [`what_a_prefills_packed_matmul_is_bound_by`] asserts it: a replacement
    /// that matched nothing would report the shipped kernel under another name.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_prefills_blocked_matmul_is_bound_by() {
        let Some(device) = device() else { return };

        let shipped =
            PackedMatmul::under(&device, Numerics::Production).expect("the block compiles");
        let want = a_tiled_call_answers(&device, &shipped);
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");

        let arms = without_each_term_of_a_block();
        let mutants: Vec<(&str, PackedMatmul)> = arms
            .iter()
            .map(|(what, written)| {
                let arm = PackedMatmul::blocked_from_source(&device, written, Block::SHIPPED)
                    .expect("the mutant compiles");
                assert_ne!(
                    a_tiled_call_answers(&device, &arm),
                    want,
                    "{what}: the mutation answered what the block answers"
                );
                (*what, arm)
            })
            .collect();

        for tokens in BLOCKED_LENGTHS {
            let shapes = shapes_at(tokens);

            eprintln!("\n  a prefill of {tokens} tokens");
            eprintln!(
                "  {:<42}{}",
                "without",
                shapes
                    .iter()
                    .map(|shape| format!("{:>26}", shape.what))
                    .collect::<String>()
            );

            for shape in &shapes {
                let entries = entries_dispatched(&device, |batch| {
                    let bank = shape.upload(&device, &shipped);
                    let mut x = device.buffer(&shape.case.x).expect("the rows upload");
                    shape.encode(batch, &device, &bank, &mut x, &grouping);
                });
                let blocked = match shape.grouped {
                    false => MMA_TILED_ENTRY,
                    true => MMA_GROUPED_ENTRY,
                };
                assert!(
                    entries.iter().any(|entry| entry == blocked),
                    "{} at {tokens} tokens dispatched {entries:?} and not {blocked}, so this \
                     column is about the reference tile",
                    shape.what
                );
            }

            // **Warm, and swept both ways.** The arms this table separates are
            // four and five percent apart, which is inside what a clock climbing
            // over a sweep manufactures — and this crate has the case on record:
            // external work reproducing the threadgroup-memory experiment
            // reported a 1.7× win that vanished entirely once thirty warm-up
            // dispatches were added. Two seconds of load puts every arm on the
            // sustained clock, which is the one a prefill runs at, and running
            // the list backwards is what separates a term the kernel has from a
            // ramp that follows the order.
            //
            // The shipped kernel is arm zero rather than a figure taken before
            // the sweep, so that what the percentages divide by was measured
            // under the same conditions as what divides it — including its own
            // place at each end of the list.
            crate::testing::warmed(|| {
                shapes[0].costs(&device, &shipped, &grouping);
            });
            let taken = |at: usize| -> Vec<Duration> {
                let arm = match at {
                    0 => &shipped,
                    at => &mutants[at - 1].1,
                };
                shapes
                    .iter()
                    .map(|shape| shape.costs(&device, arm, &grouping))
                    .collect()
            };
            let listed: Vec<usize> = (0..=mutants.len()).collect();
            let (up, down) = crate::testing::both_ways(&listed, taken);

            for (opened, passes) in [("swept up the list", &up), ("and down it", &down)] {
                eprintln!("  {opened}");
                let whole = &passes[0];
                for (at, taken) in passes.iter().enumerate() {
                    let what = match at {
                        0 => "nothing — the kernel",
                        at => mutants[at - 1].0,
                    };
                    let cells: String = taken
                        .iter()
                        .zip(whole)
                        .map(|(each, whole)| {
                            format!(
                                "{:>17}{:>9}",
                                format!("{:.2}ms", 1e3 * each.as_secs_f64()),
                                format!("{:.0}%", 1e2 * each.as_secs_f64() / whole.as_secs_f64()),
                            )
                        })
                        .collect();
                    eprintln!("  {what:<42}{cells}");
                }
            }
            let whole = up[0].as_slice();

            eprintln!(
                "  {:<42}{}",
                "it must move, once each",
                shapes
                    .iter()
                    .map(|shape| format!(
                        "{:>26}",
                        format!("{:.2} GB", shape.compulsory() as f64 / 1e9)
                    ))
                    .collect::<String>()
            );
            eprintln!(
                "  {:<42}{}",
                "which at 725 GB/s would take",
                shapes
                    .iter()
                    .zip(whole)
                    .map(|(shape, taken)| {
                        let floor = shape.compulsory() as f64 / MEMORY_BANDWIDTH;
                        format!(
                            "{:>17}{:>9}",
                            format!("{:.2}ms", 1e3 * floor),
                            format!("{:.0}%", 1e2 * floor / taken.as_secs_f64())
                        )
                    })
                    .collect::<String>()
            );

            for (shape, taken) in shapes.iter().zip(whole) {
                let achieved = shape.compulsory() as f64 / taken.as_secs_f64();
                assert!(
                    achieved < MEMORY_BANDWIDTH,
                    "{} at {tokens} tokens moved its {:.2} GB at {:.0} GB/s, which is past what \
                     this machine streams — so either the clock or the byte count is wrong",
                    shape.what,
                    shape.compulsory() as f64 / 1e9,
                    achieved / 1e9,
                );
            }
        }
    }

    /// The shipped source with both decodes taken through one gather into a
    /// 256-entry table of pairs, one entry per packed byte.
    ///
    /// **A byte is the unit the kernel reads and a byte is two codes**, so a
    /// table indexed by the whole byte is one load where the table above is two,
    /// against 2 KiB of constant memory rather than 64 bytes. Both call sites
    /// move, which is what the two replacements are.
    fn through_a_table_of_pairs(shipped: &str) -> String {
        let pairs: Vec<String> = (0..1usize << u8::BITS)
            .map(|byte| {
                let (low, high) = (byte & (ELEMENTS.len() - 1), byte >> BITS);
                format!("float2({:?}f, {:?}f)", ELEMENTS[low], ELEMENTS[high])
            })
            .collect();
        let carried = crate::testing::instead_of(
            shipped,
            "using namespace metal;",
            &format!(
                "using namespace metal;\nconstant float2 ELEMENT_PAIRS[] = {{ {} }};",
                pairs.join(", ")
            ),
        );
        let walked = crate::testing::instead_of(
            &carried,
            "        dot += element(low) * v[0] + element(high) * v[1];",
            "        const float2 pair = ELEMENT_PAIRS[code];\n        dot += pair.x * v[0] + \
             pair.y * v[1];",
        );
        crate::testing::instead_of(
            &walked,
            "                low[c] = element(code & CODE_MASK);\n                high[c] = \
             element((code >> BITS) & CODE_MASK);",
            "                const float2 pair = ELEMENT_PAIRS[code];\n                low[c] = \
             pair.x;\n                high[c] = pair.y;",
        )
    }

    /// The ways a packed byte's two codes can be turned into the two floats they
    /// stand for, each of which answers the same bits.
    ///
    /// **The arms differ in what the decode costs and in nothing else.** Every
    /// one of them produces [`ELEMENTS`] for every code — the table arms by
    /// holding it and the arithmetic arm by assembling it — so what the table below
    /// ranks is three ways of spending time on one answer, and the case asserts
    /// the answer rather than assuming it.
    fn each_way_of_decoding_a_byte() -> Vec<(&'static str, String)> {
        let shipped = source();
        vec![
            ("two gathers into a table — shipped", shipped.clone()),
            (
                "the field its bits assemble",
                assembled_from_the_bits(&shipped),
            ),
            (
                "one gather into a table of pairs",
                through_a_table_of_pairs(&shipped),
            ),
        ]
    }

    /// **What each way of decoding a packed byte costs**, over the two shapes a
    /// prefill gives this kernel.
    ///
    /// A4 priced the decode by removing it — an integer-to-float conversion in
    /// its place read 70 and 71% of the kernel — and read that as a table
    /// lookup's latency, "a memory access nobody counts". This asks the question
    /// the ablation could not: what the *cheapest* decode is, among decodes that
    /// answer the same sixteen floats. A term worth 30% is only worth taking if
    /// something else can produce the same bits for less.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_each_way_of_decoding_a_packed_byte_costs() {
        let Some(device) = device() else { return };
        let want = bit_patterns(&a_tiled_call_answers(&device, &matmul(&device)));
        eprintln!("  {:<34}{}", "the decode", bound_header());

        for (what, written) in each_way_of_decoding_a_byte() {
            let arm = PackedMatmul::from_source(&device, &written).expect("the arm compiles");
            assert_eq!(
                bit_patterns(&a_tiled_call_answers(&device, &arm)),
                want,
                "{what}: the arm answered other bits than the shipped decode"
            );
            let cells: String = a_prefills_shapes_cost(&device, &arm)
                .iter()
                .map(|taken| format!("{:>26}", format!("{:.2}ms", 1e3 * taken.as_secs_f64())))
                .collect();
            eprintln!("  {what:<34}{cells}");
        }
    }

    /// The shipped source with a threadgroup array of `floats` nobody reads for
    /// anything, which is the occupancy knob the attention kernel's own table
    /// turns — met on the kernel that declares none at all.
    ///
    /// **A tile holds no threadgroup memory, so this can only lower how many
    /// threadgroups a core holds and never raise it.** That is the whole of the
    /// question here: the shipped dispatch is already at whatever residency this
    /// part gives a kernel that asks for nothing, and the attention table says a
    /// row can be a quarter faster on the other side of a turn.
    ///
    /// Every lane fills and reads its own entry, so no barrier stands between
    /// the two and the value added to the output is a zero. **One store a lane
    /// at every size**, which is what keeps this a knob on the memory rather
    /// than on the work. Whether the array survived the compiler is read off
    /// `staticThreadgroupMemoryLength` rather than hoped for.
    fn at_residency(floats: usize) -> String {
        assert!(floats >= THREADS_PER_GROUP, "a thread fills its own entry");
        crate::testing::instead_of(
            &source(),
            &format!("constant uint RESIDENCY = {RESIDENCY};"),
            &format!("constant uint RESIDENCY = {floats};"),
        )
    }

    /// **The memory the turn rests on is memory a tile declares, and it reaches
    /// no dispatch a decode step makes** — the two things about [`RESIDENCY`]
    /// this side can check without a clock.
    ///
    /// A store of a zero followed by a load of it is what a forwarding pass
    /// exists to remove, and a compiler that removed this one would put the
    /// dispatches back on the wrong side of the turn with nothing else to show
    /// for it. What says it did not is `staticThreadgroupMemoryLength`, which is
    /// a figure the pipeline reports after it was compiled.
    #[test]
    fn a_tile_declares_the_memory_its_occupancy_turn_rests_on() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        for tiled in [&matmul.tiled, &matmul.grouped] {
            assert_eq!(
                tiled.threadgroup_memory(),
                size_of::<f32>() * RESIDENCY,
                "a tiled entry declares neither more nor less than the turn wants"
            );
        }
        assert_eq!(
            matmul.kernel.threadgroup_memory(),
            0,
            "the entry a decode step dispatches declares memory it has no use for"
        );

        // Three threadgroups a core is every declaration in (20, 26.67] KiB by
        // the sweep below, which is where a shipped figure has to sit for a
        // compiler's own rounding not to move it off the plateau.
        let held = matmul.tiled.threadgroup_memory();
        let (least, most) = (20 * 1024 + 1, 26 * 1024 + 512);
        assert!(
            (least..=most).contains(&held),
            "{held} bytes is off the plateau three threadgroups a core sit on, {least}..={most}"
        );
    }

    /// **How many threadgroups of this kernel a core holds, and what holding
    /// fewer is worth** — the occupancy term, on the two rows that are 54% of a
    /// long prefill.
    ///
    /// A tile of this kernel is a simdgroup and wants no threadgroup memory at
    /// all, so it ran at whatever residency this part gives a kernel that asks
    /// for nothing — and the row says that is too many threadgroups a core
    /// rather than too few. There is nothing here a declaration could hold, so
    /// the memory it declares is memory nobody reads and the row is what says
    /// how much of it to take.
    ///
    /// **Every arm answers what the shipped kernel answers, and that is
    /// asserted rather than argued**: the declaration reaches no operand and no
    /// order, so a row that moved the answer would be a row about something
    /// other than residency.
    ///
    /// **Warm and swept both ways**, for the reasons [`crate::testing::warmed`]
    /// and [`crate::testing::both_ways`] give. Here the two passes agree to a
    /// tenth of a percent at every arm, so the row is the kernel's.
    ///
    /// **The warm-up is what decides which clock this table is about, and this
    /// is the sweep it reaches.** An arm here is about 300 ms and there are
    /// eleven of them, which is short enough to sit inside this part's boost
    /// window end to end; two seconds of load first puts every arm on the
    /// sustained clock, which is the one a prefill of any length runs at. The
    /// turn, its declaration and its far edge are the same on either clock and
    /// only the size of the win moves.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn how_many_threadgroups_of_a_prefills_packed_matmul_a_core_holds() {
        let Some(device) = device() else { return };
        /// Floats a tile declares, from a kernel that declares nothing at all
        /// past the turn to the most a threadgroup may take.
        const DECLARED: [usize; 11] =
            [0, 256, 1024, 2048, 3072, 4096, 5120, 6144, 6656, 7168, 8192];

        let shipped = matmul(&device);
        let want = a_tiled_call_answers(&device, &shipped);
        eprintln!(
            "  a threadgroup may declare {} KiB of a core's own memory",
            device.most_threadgroup_bytes() / 1024
        );
        crate::testing::warmed(|| {
            a_prefills_shapes_cost(&device, &shipped);
        });

        let (up, down) = crate::testing::both_ways(&DECLARED, |floats| {
            // A tile that declares nothing is the entry a decode step
            // dispatches, reached by taking the residency out of the walk
            // rather than by shrinking it: an array of no floats is not a
            // declaration a kernel may make.
            let arm = match floats {
                0 => crate::testing::instead_of(
                    &crate::testing::instead_of(
                        &source(),
                        "    threadgroup float residency[RESIDENCY];\n    threadgroup volatile float *held = residency;\n    held[local] = 0.0f;\n",
                        "",
                    ),
                    "sums[r][c] = held[local];",
                    "sums[r][c] = 0.0f;",
                ),
                floats => at_residency(floats),
            };
            let matmul = PackedMatmul::from_source(&device, &arm).expect("the arm compiles");
            assert_eq!(
                a_tiled_call_answers(&device, &matmul),
                want,
                "{floats} floats of residency moved the answer"
            );
            (
                matmul.tiled.threadgroup_memory(),
                a_prefills_shapes_cost(&device, &matmul),
            )
        });

        for (opened, rows) in [("up the list", &up), ("down it", &down)] {
            eprintln!("  swept {opened}");
            eprintln!("  {:<32}{}", "a threadgroup", bound_header());
            for (held, taken) in rows {
                let cells: String = taken
                    .iter()
                    .map(|each| format!("{:>26}", format!("{:.2}ms", 1e3 * each.as_secs_f64())))
                    .collect();
                eprintln!(
                    "  {:<32}{cells}",
                    format!("{:.0} KiB", *held as f64 / 1024.0)
                );
            }
        }
    }

    /// What one small tiled call answers, which is how an arm of the table above
    /// is held to having changed the kernel rather than only its source.
    ///
    /// Two experts and thirty-seven rows apiece, which is the shape
    /// `a_tiled_dispatch_answers_row_for_row_what_the_untiled_one_answers`
    /// establishes the tiled entry over — a partial last tile, a run that
    /// straddles one, and columns that do not fill theirs — at a height that
    /// also reaches the entries behind [`Numerics::Production`].
    ///
    /// **The height is the flag's own line and not a spare row.** A call under
    /// [`MMA_ROWS_A_BLOCK`] rows stays on the reference tile whichever numerics
    /// asked, so a helper eleven rows tall would run the same kernel under both
    /// words and `a_call_under_either_numerics_answers_what_the_other_answers`
    /// would pass by comparing a thing to itself.
    fn a_tiled_call_answers(device: &Device, matmul: &PackedMatmul) -> Vec<f32> {
        const EXPERTS: usize = 2;
        // Rows enough that the tallest shape [`SWEPT_BLOCKS`] carries reaches
        // the block rather than falling back to the reference tile: a block is
        // dispatched only from `MMA_BLOCKS_A_CALL` of its own rows, so a run
        // shorter than that answers the tile's bits and the sweep compares two
        // different computations. Prime, so that no divisor of a swept shape
        // lines up with it.
        const SOURCES: usize = 71;
        let case = Case::noisy(IN_DIM, EXPERTS * OUT_DIM, SOURCES);
        let bank = PackedBank::upload(
            device,
            matmul,
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
        assert!(tiles(&chosen, ROWS_A_TILE), "the call was not tiled");

        let mut batch = device.batch().expect("a command buffer opens");
        let mut input = device.buffer(&case.x).expect("the input uploads");
        let got = bank
            .encode_repeating(&mut batch, &chosen, &mut input)
            .expect("the dispatch encodes");
        batch.wait().expect("the dispatch completes");
        got.take()
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

    /// The shipped kernel with a different [`CHUNKS_IN_FLIGHT`] written into its
    /// prelude, which is the one thing that sweep varies.
    ///
    /// The declaration is named in full on both sides of the rewrite, for the
    /// reason [`a_lane_reading`] gives: the prelude holds several `uint`
    /// constants at small powers of two and a check for the value alone would
    /// pass against one of those.
    fn a_lane_holding(chunks: usize) -> String {
        let wanted = format!("constant uint CHUNKS_IN_FLIGHT = {chunks};");
        let source = source().replace(
            &format!("constant uint CHUNKS_IN_FLIGHT = {CHUNKS_IN_FLIGHT};"),
            &wanted,
        );
        assert!(source.contains(&wanted), "the prelude declares {wanted}");
        source
    }

    /// The chunk counts the sweep runs and the bit-for-bit case holds, one of
    /// which is the shipped one.
    const CHUNKS_HELD: [usize; 5] = [1, 2, 4, 8, 16];

    /// **A lane that holds several chunks answers the bits a lane holding one
    /// answers**, which is what makes [`CHUNKS_IN_FLIGHT`] a constant rather than
    /// a second numerics.
    ///
    /// Every other lever this module has taken on the innermost walk changed the
    /// order the products are summed in and had to go behind [`Numerics`] for it.
    /// This one does not: the held chunks are read in the order the walk visits
    /// them and added in that order, so the sum is the same expression associated
    /// the same way. The claim is cheap to make and worth nothing unheld, so it is
    /// held here against the shipped kernel's own bit patterns.
    ///
    /// **The reductions are the checkpoint's and two that do not divide.** A walk
    /// of `steps` chunks runs `steps / chunks` held iterations and the rest one at
    /// a time, so what has to be checked is both paths and the seam between them —
    /// and 1536 over 16 held chunks is no held iteration at all, which is the arm
    /// that says the remainder walk alone still answers.
    #[test]
    fn a_walk_that_holds_several_chunks_answers_the_bits_one_holding_one_answers() {
        let Some(device) = device() else { return };
        // 2048 and 8192 are `shared down` and `dense down`; 4096 is every other
        // reduction in the checkpoint; 1536 and 2560 divide neither the lane
        // stride nor any chunk count above one.
        const REDUCTIONS: [usize; 5] = [1536, 2048, 2560, 4096, 8192];

        for in_dim in REDUCTIONS {
            let case = Case::noisy(in_dim, 71, 3);
            let answers = |chunks: usize| {
                let matmul = PackedMatmul::from_source(&device, &a_lane_holding(chunks))
                    .unwrap_or_else(|err| panic!("{chunks} chunks compile: {err}"));
                bit_patterns(
                    &case
                        .upload(&device, &matmul)
                        .multiply(&case.x)
                        .expect("the dispatch runs"),
                )
            };
            let want = answers(1);
            for chunks in CHUNKS_HELD {
                assert_eq!(
                    answers(chunks),
                    want,
                    "a reduction of {in_dim} answered other bits with {chunks} chunks held"
                );
            }
        }
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
