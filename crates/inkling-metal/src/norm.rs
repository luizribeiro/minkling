//! `x * rsqrt(mean(x²) + eps) * weight`, on the device.
//!
//! The first operation here that is not a multiply against a weight. Every
//! kernel before it consumed an MXFP4 tensor and produced activations; this one
//! consumes activations, which is what an engine that means to stop returning to
//! the CPU between layers has to be able to do.
//!
//! # Apple GPUs have no double, and this reduction needs the range
//!
//! [`inkling_core::ops::rms_norm`] accumulates its sum of squares in f64, and
//! the comment there says why: values above roughly 3e18 square past f32's
//! maximum, and a row that overflows there normalises to a row of zeros — a
//! wrong answer that looks like a well-behaved one. There is no f64 on this
//! hardware to borrow.
//!
//! The range is bought instead by **scaling each row by a power of two before
//! the squares are formed**, which is the same trick
//! [`softmax`](inkling_core::ops::softmax) uses over its peak, with one
//! difference that matters: the factor is `2^e` for `e = ilogb(max|x|)`, so
//! dividing by it is exact and costs no bits at all. What is summed is
//! `y = x·2^-e`, whose largest entry lies in `[1, 2)`, so the sum over a
//! 4096-wide row cannot exceed 2^14 whatever the row held.
//!
//! **The exponent then cancels.** The answer is `x·rsqrt(mean(x²) + eps)·w`,
//! and with `x = y·2^e` and
//! `rsqrt(mean(x²) + eps) = 2^-e·rsqrt(mean(y²) + eps·2^-2e)` the two powers of
//! two multiply out — so the kernel never forms the large factor it divided by,
//! and nothing it computes can overflow or fall into the subnormals on the way
//! back. A row of 1e20 normalises to its weight here, and the kernel that sums
//! `x²` directly returns zeros;
//! `a_row_that_squares_past_f32_still_normalises_on_the_device` is that pair.
//!
//! The one clamp is at the small end. `2^-e` is never formed, but `eps·2^-2e`
//! is, and a row whose entries are all around 1e-30 would ask for an `eps`
//! scaled past f32's maximum. [`LEAST_EXPONENT`] stops `e` at -60, which bites
//! only on rows whose mean square is below `2^-120`, or 7.5e-37 — thirty
//! decades under any `rms_norm_eps` a config carries, and so squarely inside
//! the range where `eps` is the whole of the answer on both paths anyway.
//!
//! # It rounds where the CPU rounds, which is nowhere
//!
//! MLX's own RMSNorm rounds twice: it normalises in float32, rounds that
//! intermediate back to the input's dtype, and only then multiplies by the
//! weight and rounds again — which is what [`inkling_core::embed`]'s tolerance
//! is about, since modelling both roundings reproduces the recorded
//! `embed_norm_out` bit for bit. The CPU path here models neither: it
//! normalises in float32 and rounds nowhere, and lands within one bfloat16
//! quantum of MLX because of it.
//!
//! **This kernel matches the CPU path and not MLX**, in the same association —
//! `(x·scale)·w`, one rounding each. The CPU is the oracle every kernel in this
//! tree is checked against, and a kernel that reproduced MLX's extra rounding
//! would be the one operation in the model whose backend changed the answer.

use std::borrow::Cow;
use std::cell::RefCell;

use inkling_core::profile::{self, Op};

use crate::buffer::{Buffer, Landing};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "rms_norm";

/// The entry that runs two of those calls as one dispatch — see [`encode_pair`].
const PAIRED_ENTRY: &str = "rms_norm_pair";

/// How many `uint`s the kernel's `Shape` struct declares.
const FIELDS: usize = 7;

/// The widest threadgroup a dispatch here uses, and — unlike [`crate::matmul`]'s
/// — all of it works on one group of one row.
///
/// **A norm is far too small to be given a simdgroup.** The packed matmul gives
/// one simdgroup to each output element and has 201024 of them to dispatch; a
/// decode-shaped norm has one row, so a simdgroup apiece is 32 threads for the
/// whole kernel and nothing to hide a memory latency behind. Measured at that
/// width it was 250 microseconds of device time for 16 KB read three times,
/// which is more than the submission around it. Eight simdgroups on the same row
/// is `a_decode_shaped_norm_costs_less_than_the_submission_around_it`.
///
/// It is a ceiling rather than the count: [`LayerNorm::threads_per_group`] takes
/// the narrower of this and what the group has lanes for.
const THREADS_PER_GROUP: usize = 256;

/// Floats one thread reads at a stride, where the group's width is a multiple of
/// it.
///
/// **What a thread reads at once is what the kernel's occupancy is really made
/// of.** A group is reduced by one threadgroup, which is one core of eighty, so
/// what is left to hide the memory latency behind is the requests one thread can
/// have outstanding. A `[1, 4096]` group over 256 threads is sixteen dependent
/// strides a thread when they are floats and four when they are `float4`, and
/// the device's own clock puts that difference at 17.6 microseconds against 8.0.
/// See [`BODY`] for the alignment this rests on, and
/// `what_a_decode_steps_norms_cost_the_device` for the rest of the figures.
const LANE: usize = 4;

/// Entries the kernel's threadgroup arrays hold, which has to be a constant
/// where the number of simdgroups is not: 1024 threads is the widest threadgroup
/// any Apple GPU allows and 32 the narrowest simdgroup any reports, so 32
/// partials is the most a threadgroup can produce.
const MOST_SIMDGROUPS: usize = 32;

/// How far down `ilogb` of a row's peak is allowed to go, which bounds the
/// `eps·2^-2e` the kernel forms. See the module documentation: at -60 the clamp
/// reaches only rows whose mean square is decades below any `eps`.
const LEAST_EXPONENT: i32 = -60;

/// The compiled kernel, which every norm on a device shares.
///
/// Per source string rather than per weight, like [`crate::PackedMatmul`]: the
/// source names no shape, so one of these serves every norm in the model.
#[derive(Debug)]
pub struct RmsNorm {
    kernel: Kernel,
    /// The same source's paired entry, compiled beside it because a model
    /// wanting one wants the other: a layer's head norms pair and its input
    /// layernorm has nothing to pair with.
    paired: Kernel,
}

impl RmsNorm {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`RmsNorm::new`] out of a source string of the caller's own, which is how
    /// a test puts a deliberately wrong kernel through the same plumbing as the
    /// right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
            paired: device.compile(source, PAIRED_ENTRY)?,
        })
    }
}

/// One RMSNorm's weight on the device, and the normalisation through it.
///
/// The weight is `[width]` and is copied rather than wrapped: a norm weight is
/// 16 KB of bfloat16 in the checkpoint and the kernel wants float32, so there
/// are no packed bytes here to hand over in place. Copied **once**, which is the
/// point — the CPU path widens the same tensor out of the mapping again on every
/// layer of every step.
#[derive(Debug)]
pub struct LayerNorm<'a> {
    device: &'a Device,
    norm: &'a RmsNorm,
    /// Held behind a cell for the reason [`crate::PackedBank`]'s resident
    /// tensors are: binding a buffer to a dispatch borrows it exclusively, and
    /// a norm is bound once per call while the weight belongs to the norm.
    weight: RefCell<Buffer<f32>>,
    width: usize,
    /// What [`LayerNorm::threads_per_group`] worked out, held rather than asked
    /// again: it is decided by the weight's width and the kernel's simdgroup,
    /// and neither moves once a norm exists, where a decode step encodes four of
    /// these a layer.
    threads: usize,
    eps: f32,
}

impl<'a> LayerNorm<'a> {
    pub fn new(
        device: &'a Device,
        norm: &'a RmsNorm,
        weight: &[f32],
        eps: f32,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            weight: RefCell::new(device.buffer(weight)?),
            width: weight.len(),
            threads: Self::threads_per_group(weight.len(), norm.kernel.simd_width()),
            device,
            norm,
            eps,
        })
    }

    /// The width a row has to be, which is the weight's.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The `rms_norm_eps` this was wrapped with.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// `[rows, width]` in, `[rows, width]` out, submitted on its own.
    ///
    /// What a caller with nothing to batch it against wants, and what the cases
    /// here drive. A caller that has something to batch it against — the four
    /// projections that consume what this produced — reaches for
    /// [`LayerNorm::encode`] instead, because a submission costs 206
    /// microseconds and this arithmetic does not.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, MetalError> {
        let mut batch = self.device.batch()?;
        let mut input = self.device.buffer(x)?;
        let normed = self.encode(&mut batch, &mut input)?;
        batch.wait()?;
        Ok(normed.to_vec())
    }

    /// The same normalisation over rows a dispatch already left on the device,
    /// encoded into `batch` and leaving its own rows there in turn.
    ///
    /// Both ends are a buffer, and that is the whole point of this method
    /// existing beside [`LayerNorm::forward`]: neither a normed hidden state nor
    /// the value it was normed from is something the CPU wants — they are what
    /// the dispatch before and the dispatch after want — and copying either back
    /// to be copied over again is two crossings of a seam that nothing asked
    /// for. A layer's *second* norm is the case that made the input end of it
    /// matter: it reads the residual add before it, which is a dispatch now.
    ///
    /// A call over no rows is the device's refusal of a zero-length allocation
    /// rather than an empty answer, which is where this parts company with
    /// [`PackedBank::encode`](crate::PackedBank::encode). That one answers a
    /// gather of no rows with no values, because a bank nothing routed to is an
    /// ordinary step of the router's; this returns the buffer the *next*
    /// dispatch reads, and there is no empty buffer to hand it — a norm over no
    /// tokens is a forward pass over no tokens.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Buffer<f32>, MetalError> {
        self.encode_last(batch, x, self.rows(x.len()))
    }

    /// The same normalisation over the **last** `rows` rows of what a dispatch
    /// left on the device, leaving those rows alone on the other side.
    ///
    /// **A suffix rather than a range, because that is the shape the back of
    /// the model asks for.** A pass produces a hidden state per token and the
    /// head reads the last of them — one row at a prefill, and the block a
    /// speculative round proposed at a decode step — so what the tail needs of
    /// a buffer forty-two layers wrote is its final rows and nothing before
    /// them. A norm is over a row and reads no other, so the answer for those
    /// rows is what a norm over the whole call would have left in the same
    /// places.
    ///
    /// The alternative was an offset on whatever reads the normed rows next,
    /// which is [`crate::PackedMatmul`] — 70% of a decode step's device time,
    /// against this kernel's 8.7%.
    pub fn encode_last(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        rows: usize,
    ) -> Result<Buffer<f32>, MetalError> {
        let _timed = profile::scope(Op::Encode);
        let from = self.rows(x.len()) - rows;
        let mut out = self.device.zeroed::<f32>(rows * self.width)?;
        self.encoding(
            batch,
            x,
            Reading { from, rows },
            None,
            Landing {
                out: &mut out,
                groups: 1,
                stride: rows,
                base: 0,
            },
        )?;
        Ok(out)
    }

    /// The same normalisation over each `width`-wide group of rows a dispatch
    /// already left on the device, scattered into `landing`.
    ///
    /// **This is the head norm.** `InklingAttention` reshapes a projection's
    /// output into heads, normalises the last axis and then transposes; the
    /// reduction is this one over a row cut into `landing.groups` groups, and
    /// the transpose is where the rows land. The key's goes one step further and
    /// lands in the span the layer is keeping, so a call's keys are normed and
    /// appended by one dispatch.
    ///
    /// `scale` is one value per row — log scaling's `tau`, which multiplies the
    /// query and nothing else. `None` is a row of ones.
    pub fn encode_over(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        scale: Option<&[f32]>,
        landing: Landing<'_>,
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        let rows = self.rows(x.len() / landing.groups.max(1));
        self.encoding(batch, x, Reading { from: 0, rows }, scale, landing)
    }

    /// The same, over a run of the rows a dispatch left rather than all of them.
    ///
    /// **A run because a batch's rows are one buffer.** The projections of a
    /// batched step produce every sequence's rows together, and a head norm
    /// writes into the span of the sequence whose rows it is norming — so what
    /// a call reads is the run that is its own. A sequence advancing alone
    /// reads the whole call, which is [`Reading::whole`] and is what
    /// [`LayerNorm::encode_over`] passes.
    pub fn encode_rows(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        reading: Reading,
        scale: Option<&[f32]>,
        landing: Landing<'_>,
    ) -> Result<(), MetalError> {
        let _timed = profile::scope(Op::Encode);
        self.encoding(batch, x, reading, scale, landing)
    }

    /// Threads the threadgroup over one group holds: whole simdgroups enough to
    /// give each thread a lane of the group, and never more than
    /// [`THREADS_PER_GROUP`].
    ///
    /// **A threadgroup wider than its group is threads that read nothing and
    /// wait at both barriers anyway.** A head norm is 128 wide, which is 32
    /// lanes of [`LANE`], so seven of every eight of [`THREADS_PER_GROUP`] have
    /// no stride to run — and a prefill gives one threadgroup to each (row,
    /// head), so it pays for all of them: 33 microseconds a dispatch against 10
    /// over a 97-token prompt's query norm.
    ///
    /// The simdgroup is asked of the kernel rather than assumed, for
    /// [`Kernel::simd_width`](crate::kernel::Kernel::simd_width)'s reason: both
    /// reductions inside are `simd_`-prefixed and their partials are indexed by
    /// simdgroup, so a threadgroup that was not whole simdgroups would leave one
    /// of them holding a partial answer nothing reduces. The cap is the one
    /// place that can still round to a part of one, and a group wide enough to
    /// reach it has lanes for every thread anyway.
    fn threads_per_group(width: usize, simd: usize) -> usize {
        let lanes = match width % LANE {
            0 => width / LANE,
            _ => width,
        };
        (lanes.div_ceil(simd) * simd).min(THREADS_PER_GROUP)
    }

    /// How many rows of this norm's width `values` is.
    fn rows(&self, values: usize) -> usize {
        assert_eq!(
            values % self.width,
            0,
            "{values} values are not whole rows of {}",
            self.width
        );
        values / self.width
    }

    /// One dispatch encoded, without the scope its two callers each open — so
    /// that the profile counts a dispatch once however it was reached.
    fn encoding(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        reading: Reading,
        scale: Option<&[f32]>,
        landing: Landing<'_>,
    ) -> Result<(), MetalError> {
        let (call, scale) = self.described(x.len(), reading, scale, &landing);
        let mut shape = self.device.inline(&call.fields)?;
        let mut weight = self.weight.borrow_mut();
        let mut scales = self.device.inline(&scale)?;

        batch.add(
            &self.norm.kernel,
            &[
                shape.arg(),
                x.arg(),
                weight.arg(),
                scales.arg(),
                landing.out.arg(),
            ],
            // A threadgroup to each group of each row, which is what makes the
            // pair the threadgroup's own position and so what makes the barriers
            // inside uniform: a threadgroup either runs a group or returns from
            // one, and never splits over the question.
            Grid::new(call.slots * self.threads, self.threads),
            call.moves,
        )
    }

    /// Everything about a call that does not depend on whether it has a
    /// dispatch to itself: the seven fields the kernel reads, the threadgroups
    /// it needs, and the bytes it moves.
    ///
    /// Here rather than inside [`LayerNorm::encoding`] because a paired dispatch
    /// asks the same three questions of each half, and a second spelling of the
    /// shape struct is one that could drift from the kernel's own.
    fn described<'s>(
        &self,
        values: usize,
        reading: Reading,
        scale: Option<&'s [f32]>,
        landing: &Landing<'_>,
    ) -> (Call, Cow<'s, [f32]>) {
        let Reading { from, rows } = reading;
        assert!(landing.groups > 0, "a row has groups");
        assert_eq!(
            values % landing.groups,
            0,
            "{values} values are not {} groups",
            landing.groups
        );
        assert!(
            from + rows <= self.rows(values / landing.groups),
            "{rows} rows at {from} of a call holding {}",
            self.rows(values / landing.groups)
        );
        let scale = match scale {
            Some(scale) => {
                assert_eq!(scale.len(), rows, "a scale a row");
                Cow::Borrowed(scale)
            }
            None => Cow::Owned(vec![1.0; rows]),
        };
        landing.fits(rows, self.width);

        let call = Call {
            fields: [
                extent(rows, "rows of a call"),
                extent(landing.groups, "the groups of a row"),
                extent(self.width, "the width of a norm"),
                extent(landing.stride, "the rows a group has room for"),
                extent(landing.base, "where a call's rows start"),
                extent(from, "where a call's rows are read from"),
                self.eps.to_bits(),
            ],
            slots: rows * landing.groups,
            // The rows in, the same rows out, the one weight every row of every
            // call reads, and a scale for each row. A call over a suffix reads
            // and writes its own rows and not the ones in front of them.
            moves: size_of::<f32>() * (2 * rows * landing.groups * self.width + self.width + rows),
        };
        (call, scale)
    }
}

/// Which rows of what a dispatch left on the device a norm reads.
///
/// **A run rather than a buffer because a batch's rows are one buffer**, for
/// the reason a convolution's [`Reading`](crate::sconv::Reading) is one. The
/// two are separate types because the two kernels index a row differently —
/// this one by group and that one by channel — and one type over both would be
/// a field each of them has to be told to ignore.
#[derive(Debug, Clone, Copy)]
pub struct Reading {
    /// The row of `x` this call's first row is.
    pub from: usize,
    /// How many rows it takes from there.
    pub rows: usize,
}

impl Reading {
    /// Every row of what the call was handed, which is what a sequence
    /// advancing alone reads.
    pub fn whole(rows: usize) -> Self {
        Self { from: 0, rows }
    }
}

/// One norm's call as a paired dispatch describes it.
struct Call {
    fields: [u32; FIELDS],
    /// Threadgroups it wants, which is a group of a row each.
    slots: usize,
    moves: usize,
}

/// One half of a paired normalisation: the rows it reads, what multiplies them,
/// and where its own rows land.
///
/// A struct rather than four more arguments because [`encode_pair`] takes two of
/// everything, and eight positional arguments of which four repeat is a call
/// nobody can read.
pub struct Normalising<'a> {
    pub norm: &'a LayerNorm<'a>,
    pub x: &'a mut Buffer<f32>,
    /// Which rows of `x` this half reads — see [`LayerNorm::encode_rows`].
    pub reading: Reading,
    /// Log scaling's `tau`, one value per row. `None` is a row of ones.
    pub scale: Option<&'a [f32]>,
    pub landing: Landing<'a>,
}

/// **Two norms as one dispatch**, where the two ask for the same threadgroup —
/// and as the two dispatches they have always been where they do not.
///
/// A layer's query norm and key norm are the pair this exists for: they read
/// different rows against different weights into different landings, neither
/// reads what the other writes, and a decode step encoded 42 of each. What they
/// have to share is the threadgroup, which is a property of their widths — both
/// are `head_dim` in this checkpoint's architecture — and nothing else.
///
/// **The fallback is not a fault.** A checkpoint whose query heads and key heads
/// were different widths would ask for two threadgroup shapes out of one grid,
/// which Metal has no way to dispatch; that is a pair this cannot make rather
/// than a checkpoint this cannot run, and the two dispatches it makes instead are
/// the two it always made.
///
/// The answer is the same bits either way, which
/// `a_paired_norm_answers_what_the_two_norms_it_replaces_answer` holds exactly:
/// a threadgroup runs the walk it ran alone, over the same values, in the same
/// order, against the same weight.
pub fn encode_pair(
    batch: &mut Batch<'_>,
    first: Normalising<'_>,
    second: Normalising<'_>,
) -> Result<(), MetalError> {
    let _timed = profile::scope(Op::Encode);
    let Normalising {
        norm: one,
        x: first_x,
        reading: first_reading,
        scale: first_scale,
        landing: first_landing,
    } = first;
    let Normalising {
        norm: other,
        x: second_x,
        reading: second_reading,
        scale: second_scale,
        landing: second_landing,
    } = second;

    if one.threads != other.threads {
        one.encoding(batch, first_x, first_reading, first_scale, first_landing)?;
        return other.encoding(
            batch,
            second_x,
            second_reading,
            second_scale,
            second_landing,
        );
    }

    let (first_call, first_scale) =
        one.described(first_x.len(), first_reading, first_scale, &first_landing);
    let (second_call, second_scale) = other.described(
        second_x.len(),
        second_reading,
        second_scale,
        &second_landing,
    );
    let device = one.device;
    let mut first_shape = device.inline(&first_call.fields)?;
    let mut second_shape = device.inline(&second_call.fields)?;
    let mut first_scales = device.inline(&first_scale)?;
    let mut second_scales = device.inline(&second_scale)?;
    let mut first_weight = one.weight.borrow_mut();
    let mut second_weight = other.weight.borrow_mut();

    batch.add(
        &one.norm.paired,
        &[
            first_shape.arg(),
            first_x.arg(),
            first_weight.arg(),
            first_scales.arg(),
            first_landing.out.arg(),
            second_shape.arg(),
            second_x.arg(),
            second_weight.arg(),
            second_scales.arg(),
            second_landing.out.arg(),
        ],
        Grid::new(
            (first_call.slots + second_call.slots) * one.threads,
            one.threads,
        ),
        first_call.moves + second_call.moves,
    )
}

/// The kernel, with the one constant the range argument rests on written into
/// its prelude rather than spelled twice.
fn source() -> String {
    format!(
        "constant int LEAST_EXPONENT = {LEAST_EXPONENT};\n\
         constant uint MOST_SIMDGROUPS = {MOST_SIMDGROUPS};\n\
         constant uint LANE = {LANE};\n{BODY}"
    )
}

/// Everything of the kernel that the clamp does not decide.
///
/// `eps` arrives as the bits of a float in a `uint` field rather than as a
/// float, because the three scalars are one buffer and a struct mixing the two
/// types is a layout the Rust side and the source would each have to get right
/// independently.
const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint rows;
    uint groups;
    uint width;
    uint stride;
    uint base;
    uint source;
    uint eps_bits;
};

/// What a lane contributes to each of the two reductions, and how it is brought
/// into the row's scale — the three places the body below touches a value rather
/// than an index, overloaded so that it is written once for a lane of one float
/// and a lane of `LANE`.
///
/// `ldexp` is where the two genuinely differ: Metal spells the vector one with a
/// vector of exponents, and every lane of a group shares the group's.
///
/// **A square is rounded before it is summed, and the pragma is what says so.**
/// Metal contracts a multiply feeding an add into `fma` by default, which keeps
/// the product's low bits and is the more accurate reduction of the two — and it
/// is the one that costs this kernel the property it exists for. A uniform row
/// normalises to its weight exactly because the row's own factor cancels: the
/// sum is a whole multiple of one rounded square, and the reciprocal square root
/// of its mean multiplies back against the value it was formed from. Summing
/// unrounded products leaves that mean a bit low and the answer an ulp under the
/// weight, which is exactly where the CPU's own narrowing leaves it —
/// `a_row_at_the_top_of_f32_normalises_where_the_cpus_own_scale_narrows` is the
/// pair, and it is a lane of `LANE` that can be measured saying so.
///
/// A lane of one is held to the same rule and cannot demonstrate it: a thread
/// with a single value adds its square to a zero, and `fma(x, x, 0)` rounds
/// where `x * x` rounds. It is written this way because the two overloads are
/// one reduction with one rounding rule and not two, which is what
/// `the_vectorised_lane_and_the_ragged_one_reduce_alike` compares.
static inline float peak_of(float value) {
    return fabs(value);
}
static inline float peak_of(float4 values) {
    const float4 magnitude = fabs(values);
    return fmax(fmax(magnitude.x, magnitude.y), fmax(magnitude.z, magnitude.w));
}
static inline float squares_of(float value) {
    #pragma clang fp contract(off)
    return value * value;
}
static inline float squares_of(float4 values) {
    #pragma clang fp contract(off)
    const float4 squares = values * values;
    return (squares.x + squares.y) + (squares.z + squares.w);
}
static inline float scaled_by(float value, int exponent) {
    return ldexp(value, exponent);
}
static inline float4 scaled_by(float4 values, int exponent) {
    return ldexp(values, int4(exponent));
}

/// One group of one row normalised, by the whole threadgroup that holds it.
///
/// Three passes over the row rather than one: the peak, the sum of the scaled
/// squares, and the write. A row of the model's width is 16 KB and the second
/// and third passes read what the first pulled in, where holding the row in
/// registers would fit in none of them.
///
/// Each of the two reductions is a simdgroup reduction and then a pass over what
/// the simdgroups left, which every thread does for itself — cheaper than
/// reducing the partials once and broadcasting, at eight entries, and it needs
/// one barrier rather than two.
///
/// **A lane is `LANE` floats where the group's width is a multiple of it**, and
/// this is where the two instantiations below come from. A width `LANE` divides
/// is both what makes `width / LANE` the whole of the group and what makes the
/// group's pointers safe to read as vectors: `values` and `result` are the
/// buffer's base plus a multiple of `width`, so nothing else bounds their
/// alignment. A norm over a ragged width takes the same body a float at a time.
template <typename Lane>
static void normalise(
    constant Shape &shape,
    device const Lane *values,
    device const Lane *weight,
    device Lane *result,
    uint lanes,
    float row_scale,
    threadgroup float *peaks,
    threadgroup float *sums,
    uint local,
    uint threads,
    uint lane,
    uint simd,
    uint simds
) {
    float peak = 0.0f;
    for (uint i = local; i < lanes; i += threads) {
        peak = fmax(peak, peak_of(values[i]));
    }
    peak = simd_max(peak);
    if (lane == 0) {
        peaks[simd] = peak;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < simds; ++s) {
        peak = fmax(peak, peaks[s]);
    }

    // `ilogb` of zero is a value the language leaves to the implementation, and
    // an all-zero row is the case `eps` exists for — so the exponent is chosen
    // without asking about it. Every thread reaches the same answer, `peak`
    // having been reduced across the whole threadgroup first.
    const int exponent = peak > 0.0f ? max(ilogb(peak), LEAST_EXPONENT) : LEAST_EXPONENT;

    float sum = 0.0f;
    for (uint i = local; i < lanes; i += threads) {
        sum += squares_of(scaled_by(values[i], -exponent));
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        sums[simd] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint s = 0; s < simds; ++s) {
        sum += sums[s];
    }

    // The scaled mean, against an `eps` brought into the same scale — which is
    // what lets the row's own factor cancel below rather than be multiplied
    // back in. `precise` because the default is a hardware approximation, and
    // this is the one reciprocal square root the whole answer rests on.
    const float eps = as_type<float>(shape.eps_bits);
    const float inverse = precise::rsqrt(
        sum / (float)shape.width + ldexp(eps, -2 * exponent)
    );

    for (uint i = local; i < lanes; i += threads) {
        result[i] = scaled_by(values[i], -exponent) * inverse * weight[i] * row_scale;
    }
}

/// One threadgroup's share of a call: the group of the row `slot` names,
/// normalised where the call's landing puts it.
///
/// Here rather than inside the entry point because two entry points run it —
/// the call on its own, and either half of a pair. What a paired dispatch
/// changes is which of two shapes a threadgroup reads and nothing else, which is
/// what makes the pair bit-safe by construction rather than within a tolerance.
static void normalise_slot(
    constant Shape &shape,
    device const float *x,
    device const float *weight,
    device const float *scale,
    device float *out,
    uint slot,
    threadgroup float *peaks,
    threadgroup float *sums,
    uint local,
    uint threads,
    uint lane,
    uint simd,
    uint simds
) {
    const uint row = slot / shape.groups;
    const uint group = slot % shape.groups;

    device const float *values =
        x + ((ulong)shape.source * shape.groups + slot) * shape.width;
    device float *result =
        out + ((ulong)group * shape.stride + shape.base + row) * shape.width;
    const float row_scale = scale[row];

    // The width is a shape and not a thread's own, so both barriers inside stay
    // uniform however this branches.
    if (shape.width % LANE == 0) {
        normalise(
            shape,
            (device const float4 *)values,
            (device const float4 *)weight,
            (device float4 *)result,
            shape.width / LANE,
            row_scale, peaks, sums, local, threads, lane, simd, simds
        );
    } else {
        normalise(
            shape, values, weight, result, shape.width,
            row_scale, peaks, sums, local, threads, lane, simd, simds
        );
    }
}

/// `out = x * rsqrt(mean(x^2) + eps) * weight * scale`, one threadgroup to each
/// group of each row.
///
/// **A row is `groups` groups of `width`, and the norm is over a group.** A
/// layer's input layernorm is one group of the hidden width; a head norm is
/// `heads` groups of `head_dim`, over the same weight, which is what
/// `InklingAttention` means by reshaping into heads and normalising the last
/// axis. The two are the same reduction over a different divisor.
///
/// Where a group's rows land is [`Landing`](crate::Landing)'s three numbers —
/// see there for why the transpose and the append are indexing rather than
/// passes over a tensor. `source` is the other end of the same idea: the row of
/// `x` this call's first row is, so that a caller wanting the rows the head
/// reads dispatches over those rows rather than over the whole call.
///
/// `scale` is one value per row, multiplied into every channel of it. Log
/// scaling's `tau` is the only thing that ever sets it and everywhere else it is
/// a row of ones, which multiplies exactly.
kernel void rms_norm(
    constant Shape &shape [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    device float *out [[buffer(4)]],
    uint slot [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]
) {
    threadgroup float peaks[MOST_SIMDGROUPS];
    threadgroup float sums[MOST_SIMDGROUPS];

    // Unreachable under the grid this is dispatched over, which gives exactly
    // one threadgroup to each group of each row. It is here for what it would
    // have to be if that ever stopped being true: the pair is the threadgroup's
    // own position, so this turns away a whole group and never splits one — and
    // a bounds check on `local` instead would leave some threads at the barriers
    // below and others past them, which is undefined rather than slow.
    if (slot >= shape.rows * shape.groups) {
        return;
    }
    normalise_slot(
        shape, x, weight, scale, out, slot,
        peaks, sums, local, threads, lane, simd, simds
    );
}

/// Two norms as one dispatch: a grid of both calls' threadgroups, the first
/// call's at the front of it.
///
/// **A layer's query norm and key norm were two dispatches because they are two
/// tensors and for no other reason.** They read different rows against different
/// weights into different landings, and neither reads anything the other writes,
/// so what separates them inside a dispatch is which shape a threadgroup reads.
///
/// **The branch is on the threadgroup's own position and so is uniform**, which
/// is what keeps the two barriers inside `normalise` legal: a threadgroup takes
/// one side whole. It is also what makes the answer the same bits — the walk,
/// the rounding rule and the reduction order a threadgroup runs are the ones it
/// ran when the call was alone, over the same values in the same order.
///
/// The two halves share one threadgroup width, which is why a pair is only
/// formed where both call for the same one — see `norm::encode_pair`.
kernel void rms_norm_pair(
    constant Shape &first [[buffer(0)]],
    device const float *first_x [[buffer(1)]],
    device const float *first_weight [[buffer(2)]],
    device const float *first_scale [[buffer(3)]],
    device float *first_out [[buffer(4)]],
    constant Shape &second [[buffer(5)]],
    device const float *second_x [[buffer(6)]],
    device const float *second_weight [[buffer(7)]],
    device const float *second_scale [[buffer(8)]],
    device float *second_out [[buffer(9)]],
    uint slot [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]
) {
    threadgroup float peaks[MOST_SIMDGROUPS];
    threadgroup float sums[MOST_SIMDGROUPS];

    const uint firsts = first.rows * first.groups;
    if (slot < firsts) {
        normalise_slot(
            first, first_x, first_weight, first_scale, first_out, slot,
            peaks, sums, local, threads, lane, simd, simds
        );
    } else if (slot < firsts + second.rows * second.groups) {
        normalise_slot(
            second, second_x, second_weight, second_scale, second_out,
            slot - firsts,
            peaks, sums, local, threads, lane, simd, simds
        );
    }
}
"#;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use inkling_core::fixture::{NORM_CASES, OPS, deviation, norm_case, norm_eps};
    use inkling_core::ops::rms_norm;
    use inkling_core::split_heads;

    use crate::testing::device;

    /// The bound `inkling_core::ops` states for the same cases, for the same
    /// reason: both sides reduce over the whole feature axis, so summation order
    /// alone moves the last bits.
    ///
    /// It is not loosened for the scaling. Dividing by a power of two is exact,
    /// so what the kernel sums are the same products the CPU sums with their
    /// exponents shifted, and a tree of 32 lanes is the better-conditioned order
    /// of the two — which is what the measurement says: 1.2e-7 worst across the
    /// five cases against the CPU's own 1.8e-7 from the same fixture, and a
    /// 4096-wide row lands at 1.3e-7. The kernel is the closer of the two to
    /// MLX despite having no double to accumulate in.
    const TOLERANCE: f32 = 1e-6;

    /// The same normalisation on the CPU, which every case here is measured
    /// against.
    ///
    /// Named rather than written out, so that what the kernel is checked against
    /// is visibly [`inkling_core::ops::rms_norm`] — the f64 accumulator and all
    /// — and not a second implementation of the formula living in this file.
    fn on_the_cpu(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        rms_norm(x, weight, eps)
    }

    /// Every shape MLX was asked about, dispatched.
    ///
    /// The cases are [`inkling_core::fixture`]'s rather than this module's, and
    /// deliberately: they are the ones the CPU path is pinned to, so what this
    /// says is that both backends answer the same five questions and not that
    /// each answers its own.
    ///
    /// The ragged case is the one worth naming: its width is not a multiple of
    /// eight and so not a multiple of the simdgroup, which is what leaves the
    /// lanes of the last stride with nothing to read.
    #[test]
    fn the_kernel_reproduces_mlx_for_every_shape() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let ckpt = inkling_core::fixture::open(OPS);
        let eps = norm_eps(&ckpt);
        let mut worst = 0.0f32;

        for name in NORM_CASES {
            let (x, weight, want) = norm_case(&ckpt, name);
            let got = LayerNorm::new(&device, &norm, &weight, eps)
                .expect("the weight uploads")
                .forward(&x)
                .expect("the dispatch completes");

            assert_eq!(got.len(), x.len(), "{name}");
            let deviation = deviation(&got, &want);
            assert!(deviation <= TOLERANCE, "{name}: deviation {deviation:e}");
            worst = worst.max(deviation);
        }
        eprintln!(
            "worst deviation from mlx over {} cases: {worst:e}",
            NORM_CASES.len()
        );
    }

    /// **A suffix answers for its own rows and reads none in front of them.**
    /// The back of the model norms the rows the head reads out of a buffer
    /// forty-two layers wrote, so what says this is indexing rather than a
    /// second normalisation is that the values land where a norm over the whole
    /// call left them — and that the rows before the suffix are not what came
    /// back.
    ///
    /// The rows are all different, so a suffix taken from the wrong end is a
    /// different answer rather than the same one.
    #[test]
    fn a_norm_over_a_suffix_answers_what_the_whole_call_answers_for_those_rows() {
        let Some(device) = device() else { return };
        const WIDTH: usize = 128;
        const ROWS: usize = 5;
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let weight: Vec<f32> = (0..WIDTH).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let layer = LayerNorm::new(&device, &norm, &weight, 1e-6).expect("the weight uploads");
        let rows: Vec<f32> = (0..ROWS * WIDTH)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * (1 + i / WIDTH) as f32)
            .collect();

        let whole = layer.forward(&rows).expect("the dispatch completes");
        for last in 1..=ROWS {
            let mut x = device.buffer(&rows).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            let suffix = layer
                .encode_last(&mut batch, &mut x, last)
                .expect("the dispatch encodes");
            batch.wait().expect("the dispatch completes");

            assert_eq!(suffix.len(), last * WIDTH, "{last} rows");
            assert_eq!(
                suffix.to_vec(),
                whole[(ROWS - last) * WIDTH..],
                "the last {last} rows"
            );
        }
    }

    /// **A norm over a run of a call's rows answers what the whole call answers
    /// for those rows**, and writes only its own.
    ///
    /// This is what a batched step needs of the head norm: one `q_proj`
    /// produces every sequence's rows into one buffer, and the norm in front of
    /// a sequence's attention step reads the run that is its own. The suffix
    /// above is the same indexing with the run pinned to the end, which is what
    /// the back of the model wants; neither end is where a batch's middle
    /// sequence sits.
    #[test]
    fn a_norm_over_a_run_of_rows_answers_what_the_whole_call_answers_for_them() {
        let Some(device) = device() else { return };
        const WIDTH: usize = 128;
        const ROWS: usize = 5;
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let weight: Vec<f32> = (0..WIDTH).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let layer = LayerNorm::new(&device, &norm, &weight, 1e-6).expect("the weight uploads");
        let rows: Vec<f32> = (0..ROWS * WIDTH)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * (1 + i / WIDTH) as f32)
            .collect();
        let whole = layer.forward(&rows).expect("the dispatch completes");

        for from in 0..ROWS {
            for taken in 1..=ROWS - from {
                let mut x = device.buffer(&rows).expect("the rows upload");
                let mut out = device
                    .zeroed::<f32>(ROWS * WIDTH)
                    .expect("the landing allocates");
                let mut batch = device.batch().expect("a command buffer opens");
                layer
                    .encode_rows(
                        &mut batch,
                        &mut x,
                        Reading { from, rows: taken },
                        None,
                        Landing {
                            out: &mut out,
                            groups: 1,
                            stride: ROWS,
                            base: from,
                        },
                    )
                    .expect("the dispatch encodes");
                batch.wait().expect("the dispatch completes");

                let out = out.to_vec();
                assert_eq!(
                    out[from * WIDTH..(from + taken) * WIDTH],
                    whole[from * WIDTH..(from + taken) * WIDTH],
                    "{taken} rows from {from}"
                );
                for at in (0..ROWS).filter(|at| !(from..from + taken).contains(at)) {
                    assert!(
                        out[at * WIDTH..][..WIDTH].iter().all(|value| *value == 0.0),
                        "row {at} was written by a call over {taken} rows from {from}"
                    );
                }
            }
        }
    }

    /// What the bandwidth column divides by, against what the kernel reads: the
    /// rows in, the same rows out, the one weight every row of every call
    /// reduces against, and a scale a row.
    ///
    /// A suffix is charged its own rows, which is what keeps the tail's one-row
    /// norm out of a table that would otherwise read it as a pass over the whole
    /// prompt.
    #[test]
    fn a_dispatch_declares_the_rows_and_the_weight_it_reads() {
        let Some(device) = device() else { return };
        const WIDTH: usize = 128;
        const ROWS: usize = 3;
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let layer =
            LayerNorm::new(&device, &norm, &vec![1.0; WIDTH], 1e-6).expect("the weight uploads");
        let mut x = device
            .buffer(&vec![0.5f32; ROWS * WIDTH])
            .expect("the rows upload");

        let moved = crate::testing::moved(&device, |batch| {
            layer.encode(batch, &mut x).expect("the dispatch encodes");
        });

        assert_eq!(
            moved as usize,
            size_of::<f32>() * (2 * ROWS * WIDTH + WIDTH + ROWS)
        );
        // **The weight is one row and not one a row.** Every row of a call
        // reduces against the same `width` values, so a figure that charged
        // them per row would scale with the call and read as a kernel three
        // times further from the machine than it is.
        assert!(
            (moved as usize) < size_of::<f32>() * (2 * ROWS * WIDTH + ROWS * WIDTH),
            "the weight was charged once a row"
        );
    }

    /// **The reason this kernel is not the CPU's loop transcribed.** A row of
    /// 1e20 squares to 1e40, which is past f32's maximum, and a kernel that
    /// summed those squares directly returns a row of zeros — normalised,
    /// finite and wrong.
    ///
    /// `inkling_core::ops`'s own `a_row_that_squares_past_f32_still_normalises`
    /// is the same case against the f64 accumulator this hardware does not have.
    /// The mutation below is what says the scaling is what buys the range and
    /// not something else: the same source with the row's exponent taken out of
    /// it, through the same plumbing, on the same input.
    ///
    /// **Both read widths, because the reduction is written once and
    /// instantiated twice.** A width [`LANE`] divides reads four values to a
    /// lane and a ragged one reads them a float at a time, so a scaling that
    /// survived only one of those would be a range this kernel has at the
    /// model's widths and not at the fixtures'.
    ///
    /// The bound each is held to is the width's own arithmetic and not the two
    /// paths'. A uniform row sums one rounded square as many times as it is
    /// wide, and 32 of anything doubles five times and rounds nowhere where 33
    /// of it needs thirty bits — so the first agrees with the f64 oracle bit for
    /// bit and the second is an ulp under it, on either path.
    #[test]
    fn a_row_that_squares_past_f32_still_normalises_on_the_device() {
        let Some(device) = device() else { return };
        for (width, exactly) in [(32, 0.0), (33, TOLERANCE)] {
            let x = vec![1e20; width];
            let weight = vec![1.0; width];
            let normed = |norm: &RmsNorm| {
                LayerNorm::new(&device, norm, &weight, 1e-6)
                    .expect("the weight uploads")
                    .forward(&x)
                    .expect("the dispatch completes")
            };

            let got = normed(&RmsNorm::new(&device).expect("the norm compiles"));
            assert!(
                got.iter().all(|y| (y - 1.0).abs() <= TOLERANCE),
                "a uniform row of {width} should normalise to its weight: {got:?}"
            );
            let deviation = deviation(&got, &on_the_cpu(&x, &weight, 1e-6));
            assert!(deviation <= exactly, "{width}: deviation {deviation:e}");

            let unscaled = source()
                .replace("scaled_by(values[i], -exponent)", "values[i]")
                .replace("ldexp(eps, -2 * exponent)", "eps");
            assert_ne!(unscaled, source(), "the mutation changed nothing");
            let flushed =
                normed(&RmsNorm::from_source(&device, &unscaled).expect("the mutant compiles"));
            assert_eq!(
                flushed,
                vec![0.0; width],
                "summing a row of {width} unscaled did not flush it, so this proves nothing"
            );
        }
    }

    /// **The two read widths are one reduction**, and this is what says so: the
    /// same values through the shipped kernel and through one whose width test
    /// can never pass, so the float-at-a-time body runs where the `float4` one
    /// would have.
    ///
    /// Within the same bound as the CPU rather than bit for bit, and the reason
    /// is the reason the bound exists at all. The two paths group a group's
    /// values differently — a lane of four holds four neighbours where a lane of
    /// one holds every `threads`th value — so what they hand the simdgroup tree
    /// are different partial sums of the same terms, and summation order alone
    /// moves the last bits. What is being asked here is whether the vectorised
    /// read reaches the same values, which a stride or a cast that was wrong
    /// would answer by decades rather than by ulps.
    #[test]
    fn the_vectorised_lane_and_the_ragged_one_reduce_alike() {
        let Some(device) = device() else { return };
        let scalar = source().replace("shape.width % LANE == 0", "false");
        assert_ne!(scalar, source(), "the mutation changed nothing");
        let (rows, width) = (3, 256);
        let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 12.0)
            .collect();

        let normed = |source: &str| {
            let norm = RmsNorm::from_source(&device, source).expect("the norm compiles");
            LayerNorm::new(&device, &norm, &weight, 1e-6)
                .expect("the weight uploads")
                .forward(&x)
                .expect("the dispatch completes")
        };

        let want = on_the_cpu(&x, &weight, 1e-6);
        for (path, got) in [
            ("a lane of LANE", normed(&source())),
            ("a lane of one", normed(&scalar)),
        ] {
            let deviation = deviation(&got, &want);
            assert!(deviation <= TOLERANCE, "{path}: deviation {deviation:e}");
        }
    }

    /// `eps` sits under the square root, which is the one place the
    /// normalisation divides by zero — and on this path it is also the row whose
    /// exponent is chosen from a peak that is not there to be read.
    #[test]
    fn an_all_zero_row_normalises_to_zero_rather_than_nan() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let width = 40;
        let weight: Vec<f32> = (0..width).map(|i| 1.0 + i as f32).collect();

        let got = LayerNorm::new(&device, &norm, &weight, 1e-6)
            .expect("the weight uploads")
            .forward(&vec![0.0; width])
            .expect("the dispatch completes");

        assert!(got.iter().all(|y| *y == 0.0), "{got:?}");
    }

    /// The top of the range, which the overflow case above does not reach and
    /// where the *oracle* is the one that gives ground.
    ///
    /// A row near f32's maximum normalises by a scale of about 6e-39, and the
    /// smallest normal float is 1.2e-38 — so the CPU's `1/sqrt(mean(x²) + eps)`
    /// is formed in f64 and then narrowed into the subnormals, where it gives up
    /// the low bits of its mantissa. This kernel never forms that number at all:
    /// the row's exponent cancels, so what it multiplies is a value in `[1, 2)`
    /// against a reciprocal square root of the same order.
    ///
    /// A uniform row has to normalise to its weight exactly, and the assertion
    /// is that only one of the two does. One ulp is all it costs the CPU here,
    /// two decades further up it would be the whole answer, and neither is a
    /// magnitude a hidden state reaches — what this pins is that the device path
    /// has no such ceiling to be near.
    ///
    /// **And it is the rounded square that keeps it there.** The mutation is the
    /// pragma taken out of the kernel, which lets Metal fuse the multiply that
    /// forms a square into the add that accumulates it — the more accurate sum
    /// of the two, and the one that lands the answer on the CPU's own narrowed
    /// value. That is the whole of what the pragma is for, on the only lane
    /// width that can be measured saying so.
    #[test]
    fn a_row_at_the_top_of_f32_normalises_where_the_cpus_own_scale_narrows() {
        let Some(device) = device() else { return };
        let width = 32;
        let x = vec![f32::MAX / 2.0; width];
        let weight = vec![1.0; width];
        let normed = |norm: &RmsNorm| {
            LayerNorm::new(&device, norm, &weight, 1e-6)
                .expect("the weight uploads")
                .forward(&x)
                .expect("the dispatch completes")
        };

        let got = normed(&RmsNorm::new(&device).expect("the norm compiles"));
        let theirs = on_the_cpu(&x, &weight, 1e-6);
        eprintln!(
            "a row at {:e}: the device gives {}, the CPU {}",
            x[0], got[0], theirs[0]
        );

        assert_eq!(got, weight, "a uniform row normalises to its weight");
        assert_ne!(theirs, weight, "the CPU's scale did not narrow after all");

        let fused = source().replace("#pragma clang fp contract(off)\n", "");
        assert_ne!(fused, source(), "the mutation changed nothing");
        assert_eq!(
            normed(&RmsNorm::from_source(&device, &fused).expect("the mutant compiles")),
            theirs,
            "summing unrounded squares did not narrow the answer, so this proves nothing"
        );
    }

    /// The other end of the range, where the clamp is: a row far below where
    /// `eps` stops mattering still normalises rather than dividing by an `eps`
    /// scaled past f32's maximum.
    ///
    /// `eps` is the whole answer at this magnitude on either path — 1e-30
    /// squares to 1e-60, which is fifty decades under it — so what this asserts
    /// is that the two agree there, not that either is interesting.
    #[test]
    fn a_row_far_below_eps_normalises_to_what_the_cpu_gives() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let width = 64;
        let x: Vec<f32> = (0..width).map(|i| 1e-30 * (i as f32 - 32.0)).collect();
        let weight = vec![2.0; width];

        let got = LayerNorm::new(&device, &norm, &weight, 1e-6)
            .expect("the weight uploads")
            .forward(&x)
            .expect("the dispatch completes");

        assert!(got.iter().all(|y| y.is_finite()), "{got:?}");
        let deviation = deviation(&got, &on_the_cpu(&x, &weight, 1e-6));
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    /// Each row is normalised by its own RMS, which is what a norm over the last
    /// axis means and what a kernel reducing over the whole buffer would get
    /// wrong while still producing a plausible tensor.
    ///
    /// The two rows differ by 2^10, so a shared reduction leaves one of them
    /// three decades out — and each is checked against the same row dispatched
    /// alone, which is the comparison a shared reduction fails.
    ///
    /// That the two normalise to nearly the same values is the second half, and
    /// "nearly" is exact rather than sloppy: `eps` sits under the square root as
    /// an absolute term, so a norm is *not* scale-invariant and a row scaled by
    /// 2^10 normalises to `x/sqrt(mean + eps/2^20)` where the small one gives
    /// `x/sqrt(mean + eps)`. At this magnitude that is a couple of ulps, and at
    /// zero it would be the whole answer.
    #[test]
    fn each_row_is_normalised_by_its_own_root_mean_square() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let width = 48;
        let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i % 5) as f32).collect();
        let small: Vec<f32> = (0..width).map(|i| (i as f32 - 24.0) / 8.0).collect();
        let large: Vec<f32> = small.iter().map(|x| x * 1024.0).collect();

        let layer = LayerNorm::new(&device, &norm, &weight, 1e-6).expect("the weight uploads");
        let together = layer
            .forward(&[small.clone(), large.clone()].concat())
            .expect("the dispatch completes");

        for (row, x) in [small, large].iter().enumerate() {
            let alone = layer.forward(x).expect("the dispatch completes");
            assert_eq!(&together[row * width..][..width], &alone[..], "row {row}");
        }
        let apart = deviation(&together[..width], &together[width..]);
        assert!(
            apart <= TOLERANCE,
            "a factor of 2^10 between the rows survived the normalisation: {apart:e}"
        );
    }

    /// The weight multiplies by channel and the scale by row, and exchanging the
    /// two axes is the mistake that still fills the buffer. A weight of ones
    /// would hide it, so this one is not.
    #[test]
    fn the_weight_multiplies_the_channel_it_belongs_to() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let width = 33;
        let x: Vec<f32> = (0..width).map(|i| ((i % 7) as f32 - 3.0) / 2.0).collect();
        let weight: Vec<f32> = (0..width).map(|i| 1.0 + i as f32 / 16.0).collect();
        let normed = |weight: &[f32]| {
            LayerNorm::new(&device, &norm, weight, 1e-6)
                .expect("the weight uploads")
                .forward(&x)
                .expect("the dispatch completes")
        };

        let want = on_the_cpu(&x, &weight, 1e-6);
        let agreed = deviation(&normed(&weight), &want);
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");

        let mut reversed = weight.clone();
        reversed.reverse();
        assert!(
            deviation(&normed(&reversed), &want) > TOLERANCE,
            "the weight reached the answer in an order it cannot see"
        );
    }

    /// A width no power of two divides leaves threads idle on the last stride
    /// of every pass, and it leaves the two reductions with simdgroups that
    /// contributed nothing — 37 values are read a float at a time, and the two
    /// simdgroups the threadgroup is sized to hold 32 of them and 5.
    #[test]
    fn a_width_narrower_than_the_threadgroup_still_reproduces_the_cpu() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let (width, rows) = (37, 19);
        let weight: Vec<f32> = (0..width).map(|i| 1.0 + (i % 3) as f32).collect();
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i % 23) as f32 - 11.0) / 4.0)
            .collect();
        let got = LayerNorm::new(&device, &norm, &weight, 1e-6)
            .expect("the weight uploads")
            .forward(&x)
            .expect("the dispatch completes");

        let deviation = deviation(&got, &on_the_cpu(&x, &weight, 1e-6));
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    /// **The head norm**: a row cut into groups, each normalised over its own
    /// channels by the same weight, and written where the attention step reads
    /// it.
    ///
    /// The reduction is checked against [`inkling_core::ops::rms_norm`] over a
    /// weight of the head's width, which is the same function this module's
    /// other cases use and which already normalises a chunk per weight — so
    /// what is new here is the divisor and the layout, not the arithmetic.
    ///
    /// The layout is the claim: the input is `[rows, groups * width]` and the
    /// output is `[groups, rows, width]`, which is
    /// [`split_heads`](inkling_core::split_heads) done by choosing an index.
    /// Two rows and three groups, because a single row makes the transpose the
    /// identity and a single group makes the divisor the row.
    #[test]
    fn a_rows_groups_are_normalised_apart_and_land_where_the_step_reads_them() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let (rows, groups, width) = (2, 3, 40);
        let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let x: Vec<f32> = (0..rows * groups * width)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 16.0)
            .collect();

        let layer = LayerNorm::new(&device, &norm, &weight, 1e-6).expect("the weight uploads");
        let mut input = device.buffer(&x).expect("the row uploads");
        let mut out = device
            .zeroed::<f32>(x.len())
            .expect("the landing allocates");
        let mut batch = device.batch().expect("a command buffer opens");
        layer
            .encode_over(
                &mut batch,
                &mut input,
                None,
                Landing {
                    out: &mut out,
                    groups,
                    stride: rows,
                    base: 0,
                },
            )
            .expect("the norm encodes");
        batch.wait().expect("the batch completes");

        let apart = on_the_cpu(&x, &weight, 1e-6);
        let agreed = deviation(&out.to_vec(), &split_heads(&apart, groups, width));
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");

        // And the transpose is load-bearing: the same values left where they
        // were computed are a different tensor, which is what a landing that
        // ignored its groups would produce.
        assert!(
            deviation(&out.to_vec(), &apart) > TOLERANCE,
            "the groups landed in the order they were read"
        );
    }

    /// **A paired dispatch answers what the two dispatches it replaces answer,
    /// exactly** — which is the whole of what a merge is allowed to be, and the
    /// one thing a tolerance would be the wrong instrument for.
    ///
    /// The two halves are made as unalike as a layer's query norm and key norm
    /// are: different rows, different values, different weights, one with a
    /// per-row scale and one without, different group counts, and landings of
    /// different strides. What they share is the width, which is what makes the
    /// threadgroup one threadgroup.
    ///
    /// **And it costs the pipeline nothing**, which is asserted beside the
    /// answer: a merge that halved the widest threadgroup this kernel can be
    /// dispatched in would be giving back on one axis what it took on another.
    #[test]
    fn a_paired_norm_answers_what_the_two_norms_it_replaces_answer() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        assert!(
            norm.paired.max_threads_per_group() >= norm.kernel.max_threads_per_group(),
            "the paired entry is dispatched in a narrower threadgroup than the plain one"
        );

        let width = 32;
        let query = weight_of(width, 3);
        let key = weight_of(width, 5);
        let (queries, heads, kv_heads) = (3, 4, 2);
        let q = values(queries * heads * width, 11);
        let k = values(queries * kv_heads * width, 23);
        let taus: Vec<f32> = (0..queries).map(|i| 0.5 + i as f32 / 4.0).collect();

        let one = LayerNorm::new(&device, &norm, &query, 1e-6).expect("the weight uploads");
        let other = LayerNorm::new(&device, &norm, &key, 1e-5).expect("the weight uploads");
        assert_eq!(
            one.threads, other.threads,
            "the two ask for one threadgroup"
        );

        // The strides differ so that a half reading the other's landing would
        // write somewhere neither of them wrote alone.
        let landings = (queries, queries + 4);
        let together = normed_pair(
            &device,
            &one,
            &q,
            Some(&taus),
            heads,
            landings.0,
            &other,
            &k,
            None,
            kv_heads,
            landings.1,
        );
        let apart = (
            normed_alone(&device, &one, &q, Some(&taus), heads, landings.0),
            normed_alone(&device, &other, &k, None, kv_heads, landings.1),
        );

        assert_eq!(together.0, apart.0, "the query norm");
        assert_eq!(together.1, apart.1, "the key norm");
        assert!(
            together.0.iter().any(|value| *value != 0.0),
            "a pair that wrote nothing would compare equal to another that did"
        );
    }

    /// **Two norms of different widths are two dispatches, and answer the same.**
    ///
    /// A pair shares one threadgroup and a threadgroup is decided by the width,
    /// so halves that disagree about it cannot come out of one grid. Nothing in
    /// this checkpoint's architecture makes them disagree — a query head and a
    /// key head are both `head_dim` — so this is the arm that says the fallback
    /// is a fallback rather than a wrong answer.
    #[test]
    fn norms_that_cannot_share_a_threadgroup_are_dispatched_apart() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let (narrow, wide) = (32, 1024);
        let one = LayerNorm::new(&device, &norm, &weight_of(narrow, 3), 1e-6)
            .expect("the weight uploads");
        let other =
            LayerNorm::new(&device, &norm, &weight_of(wide, 5), 1e-6).expect("the weight uploads");
        assert_ne!(one.threads, other.threads, "the two widths part company");

        let (x, y) = (values(2 * narrow, 7), values(2 * wide, 9));
        let dispatches = device.dispatches();
        let together = normed_pair(&device, &one, &x, None, 1, 2, &other, &y, None, 1, 2);
        assert_eq!(device.dispatches() - dispatches, 2, "two dispatches");

        assert_eq!(together.0, normed_alone(&device, &one, &x, None, 1, 2));
        assert_eq!(together.1, normed_alone(&device, &other, &y, None, 1, 2));
    }

    /// A weight that is not all one value, so a norm that dropped it or read it
    /// at the wrong channel would answer differently.
    fn weight_of(width: usize, salt: usize) -> Vec<f32> {
        (0..width)
            .map(|i| 0.5 + ((i * salt) % 11) as f32 / 8.0)
            .collect()
    }

    /// Values spread over both signs and several magnitudes, so that the peak
    /// and the reduction each have something to do.
    fn values(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 37 + salt) % 89) as f32 / 8.0 - 5.5)
            .collect()
    }

    /// One norm dispatched on its own, into a landing of `stride` rows.
    fn normed_alone(
        device: &Device,
        norm: &LayerNorm<'_>,
        x: &[f32],
        scale: Option<&[f32]>,
        groups: usize,
        stride: usize,
    ) -> Vec<f32> {
        let mut input = device.buffer(x).expect("the rows upload");
        let mut out = device
            .zeroed::<f32>(groups * stride * norm.width())
            .expect("the landing allocates");
        let mut batch = device.batch().expect("a command buffer opens");
        norm.encode_over(
            &mut batch,
            &mut input,
            scale,
            Landing {
                out: &mut out,
                groups,
                stride,
                base: 0,
            },
        )
        .expect("the norm encodes");
        batch.wait().expect("the batch completes");
        out.to_vec()
    }

    /// The same two, through one [`encode_pair`].
    #[expect(
        clippy::too_many_arguments,
        reason = "two of everything a norm's call is, which is what the pair takes"
    )]
    fn normed_pair(
        device: &Device,
        one: &LayerNorm<'_>,
        x: &[f32],
        scale: Option<&[f32]>,
        groups: usize,
        stride: usize,
        other: &LayerNorm<'_>,
        y: &[f32],
        other_scale: Option<&[f32]>,
        other_groups: usize,
        other_stride: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut first_x = device.buffer(x).expect("the rows upload");
        let mut second_x = device.buffer(y).expect("the rows upload");
        let mut first_out = device
            .zeroed::<f32>(groups * stride * one.width())
            .expect("the landing allocates");
        let mut second_out = device
            .zeroed::<f32>(other_groups * other_stride * other.width())
            .expect("the landing allocates");
        let mut batch = device.batch().expect("a command buffer opens");
        super::encode_pair(
            &mut batch,
            Normalising {
                norm: one,
                x: &mut first_x,
                reading: Reading::whole(x.len() / (groups * one.width())),
                scale,
                landing: Landing {
                    out: &mut first_out,
                    groups,
                    stride,
                    base: 0,
                },
            },
            Normalising {
                norm: other,
                x: &mut second_x,
                reading: Reading::whole(y.len() / (other_groups * other.width())),
                scale: other_scale,
                landing: Landing {
                    out: &mut second_out,
                    groups: other_groups,
                    stride: other_stride,
                    base: 0,
                },
            },
        )
        .expect("the pair encodes");
        batch.wait().expect("the batch completes");
        (first_out.to_vec(), second_out.to_vec())
    }

    /// A landing writes its call's rows into a span with room for more, and
    /// touches nothing else — which is what makes a key's head norm the append
    /// into the cache rather than a step before one.
    ///
    /// Two calls at different offsets into one span, with the slots between and
    /// after them left at zero, and each group's own prefix checked against the
    /// same call dispatched into a span of its own.
    #[test]
    fn a_landing_writes_its_own_rows_of_a_span_and_no_others() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let (groups, width, stride) = (2, 8, 6);
        let weight: Vec<f32> = (0..width).map(|i| 1.0 + (i % 3) as f32).collect();
        let layer = LayerNorm::new(&device, &norm, &weight, 1e-6).expect("the weight uploads");

        let mut span = device
            .zeroed::<f32>(groups * stride * width)
            .expect("the span allocates");
        let call = |rows: usize, salt: usize| -> Vec<f32> {
            (0..rows * groups * width)
                .map(|i| ((i * 13 + salt) % 29) as f32 - 14.0)
                .collect()
        };
        let normed = |x: &[f32], base: usize, out: &mut Buffer<f32>, stride: usize| {
            let mut input = device.buffer(x).expect("the rows upload");
            let mut batch = device.batch().expect("a command buffer opens");
            layer
                .encode_over(
                    &mut batch,
                    &mut input,
                    None,
                    Landing {
                        out,
                        groups,
                        stride,
                        base,
                    },
                )
                .expect("the norm encodes");
            batch.wait().expect("the batch completes");
        };

        let (first, second) = (call(3, 0), call(2, 7));
        normed(&first, 0, &mut span, stride);
        normed(&second, 3, &mut span, stride);

        // The same two calls into a span cut to exactly what they fill, which
        // has no room for a row to land in the wrong place.
        let mut want = device
            .zeroed::<f32>(groups * 5 * width)
            .expect("the span allocates");
        normed(&first, 0, &mut want, 5);
        normed(&second, 3, &mut want, 5);

        let (span, want) = (span.to_vec(), want.to_vec());
        for group in 0..groups {
            let (filled, held) = (5 * width, stride * width);
            assert_eq!(
                span[group * held..][..filled],
                want[group * filled..][..filled],
                "group {group}"
            );
            assert!(
                span[group * held + filled..][..held - filled]
                    .iter()
                    .all(|slot| *slot == 0.0),
                "group {group} was written past its own rows"
            );
        }
    }

    /// One scale per row, multiplied into every channel of it — which is log
    /// scaling's `tau` and nothing else, and which is why `None` has to be
    /// exactly a row of ones rather than nearly one.
    #[test]
    fn the_row_scale_multiplies_every_channel_of_the_row_it_belongs_to() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let (rows, width) = (3, 16);
        let weight: Vec<f32> = (0..width).map(|i| 0.5 + (i % 5) as f32).collect();
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 19 % 31) as f32 - 15.0) / 4.0)
            .collect();
        let scales = [0.5, 2.0, 4.0];

        let layer = LayerNorm::new(&device, &norm, &weight, 1e-6).expect("the weight uploads");
        let scaled = |scale: Option<&[f32]>| {
            let mut input = device.buffer(&x).expect("the rows upload");
            let mut out = device
                .zeroed::<f32>(x.len())
                .expect("the landing allocates");
            let mut batch = device.batch().expect("a command buffer opens");
            layer
                .encode_over(
                    &mut batch,
                    &mut input,
                    scale,
                    Landing {
                        out: &mut out,
                        groups: 1,
                        stride: rows,
                        base: 0,
                    },
                )
                .expect("the norm encodes");
            batch.wait().expect("the batch completes");
            out.to_vec()
        };

        let plain = scaled(None);
        assert_eq!(plain, on_the_cpu(&x, &weight, 1e-6), "a row of ones");

        let want: Vec<f32> = plain
            .chunks_exact(width)
            .zip(scales)
            .flat_map(|(row, scale)| row.iter().map(move |value| value * scale))
            .collect();
        assert_eq!(scaled(Some(&scales)), want);
    }

    /// What a decode-shaped norm costs the device to run, which is the figure
    /// that decides whether moving one here is worth a dispatch at all.
    ///
    /// A `[1, 4096]` norm is 16 KB read three times and nothing else, against a
    /// `[1, 4096] @ [4096, 4096]ᵀ` projection's 8 MB of packed weights — so the
    /// arithmetic is four hundred times smaller and the question is entirely
    /// whether the grid is wide enough to hide the latency. Nothing asserts a
    /// ratio; the numbers go to stderr for the commit message to quote.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn a_decode_shaped_norm_costs_less_than_the_submission_around_it() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let width = 4096;
        let x: Vec<f32> = (0..width).map(|i| (i % 37) as f32 - 18.0).collect();
        let layer =
            LayerNorm::new(&device, &norm, &vec![1.0; width], 1e-6).expect("the weight uploads");

        // Warm: the first dispatch of a fresh pipeline pays for the driver's
        // first look at these buffers, which a decode loop pays once.
        for _ in 0..2 {
            layer.forward(&x).expect("the dispatch completes");
        }

        const CALLS: u32 = 64;
        let started = std::time::Instant::now();
        for _ in 0..CALLS {
            layer.forward(&x).expect("the dispatch completes");
        }
        let each = started.elapsed() / CALLS;

        eprintln!("a [1, {width}] norm submitted on its own: {each:.2?}");
        assert!(each < std::time::Duration::from_millis(2), "{each:?}");
    }

    /// **A group's threadgroup is the simdgroups its lanes need**, which is the
    /// one thing about this dispatch that no other case here can see: the kernel
    /// strides over the group by `threads_per_threadgroup` whatever that is, so
    /// a threadgroup of the wrong width still normalises correctly and every
    /// other test in this module passes.
    ///
    /// The two anchors are the shapes the model runs — a head norm's 128 and the
    /// hidden width's 4096 — and the rest is the two ways the sizing can be
    /// wrong. A threadgroup narrower than the group's lanes is a thread doing
    /// two strides where the cap did not ask it to; a threadgroup a whole
    /// simdgroup wider is 32 threads that read nothing and wait at both
    /// barriers.
    #[test]
    fn a_groups_threadgroup_is_the_simdgroups_its_lanes_need() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let simd = norm.kernel.simd_width();
        let sized = |width: usize| {
            LayerNorm::new(&device, &norm, &vec![1.0; width], 1e-6)
                .expect("the weight uploads")
                .threads
        };

        assert_eq!(sized(128), simd, "a head norm is one simdgroup of lanes");
        assert_eq!(sized(4096), THREADS_PER_GROUP, "the hidden width fills one");

        for width in [1, 2, 16, 33, 37, 40, 48, 128, 256, 1024, 4096] {
            let lanes = if width % LANE == 0 {
                width / LANE
            } else {
                width
            };
            let threads = sized(width);
            assert!(threads <= THREADS_PER_GROUP, "{width}: {threads}");
            assert_eq!(
                threads % simd,
                0,
                "{width}: {threads} is a partial simdgroup"
            );
            assert!(
                threads >= lanes.min(THREADS_PER_GROUP),
                "{width}: {lanes} lanes over {threads} threads"
            );
            assert!(
                threads == THREADS_PER_GROUP || threads - simd < lanes,
                "{width}: a simdgroup of {threads} has no lane to read"
            );
        }
    }

    /// **What each norm a layer runs costs the device**, at the shape a decode
    /// step gives it and at the three prefill lengths M9 measured.
    ///
    /// The per-kernel table says `rms_norm` is 8.6% of a decode step's device
    /// time for 0.3% of its bandwidth, which is a ranking and not a diagnosis.
    /// This is the diagnosis: the same 4096 values, as one group and as the 32 a
    /// query head norm cuts them into, are 18.0 microseconds and 5.4 — so what
    /// the kernel waits on is the memory latency one core can have outstanding,
    /// and not the memory. An empty dispatch of the same grid is 1.5, which is
    /// the floor under every row here.
    ///
    /// Read off the device's own clock over a command buffer of `CALLS`
    /// dispatches, rather than off the wall clock around one: a submission is
    /// 225 microseconds and every figure here is under forty. Sampling stays off
    /// — a timed dispatch is a compute pass of its own and that boundary is
    /// worth a couple of microseconds against numbers this size.
    ///
    /// The shapes are interleaved and repeated rather than run one after the
    /// other, because T1 established that this machine's own state moves a
    /// measurement by more than the differences here are made of. Nothing
    /// asserts a duration; the numbers go to stderr for the commit message to
    /// quote.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_a_decode_steps_norms_cost_the_device() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        const CALLS: usize = 256;
        const ROUNDS: usize = 5;
        // The checkpoint's own attention shape, so the head norms below are the
        // ones a layer runs and not a plausible pair.
        const HIDDEN: usize = 4096;
        const HEAD: usize = 128;
        const HEADS: usize = 32;
        const KV_HEADS: usize = 8;

        let cost = |rows: usize, groups: usize, width: usize| -> Duration {
            let layer = LayerNorm::new(&device, &norm, &vec![1.0; width], 1e-6)
                .expect("the weight uploads");
            let x: Vec<f32> = (0..rows * groups * width)
                .map(|i| (i % 37) as f32 - 18.0)
                .collect();
            let mut input = device.buffer(&x).expect("the rows upload");
            let mut out = device
                .zeroed::<f32>(x.len())
                .expect("the landing allocates");

            let mut batch = device.batch().expect("a command buffer opens");
            for _ in 0..CALLS {
                layer
                    .encode_over(
                        &mut batch,
                        &mut input,
                        None,
                        Landing {
                            out: &mut out,
                            groups,
                            stride: rows,
                            base: 0,
                        },
                    )
                    .expect("the norm encodes");
            }
            profile::take();
            batch.wait().expect("the batch completes");
            profile::take().gpu() / CALLS as u32
        };

        let shapes = [
            ("input layernorm, decode", 1, 1, HIDDEN),
            ("query head norm, decode", 1, HEADS, HEAD),
            ("key head norm, decode", 1, KV_HEADS, HEAD),
            ("input layernorm, 97", 97, 1, HIDDEN),
            ("query head norm, 97", 97, HEADS, HEAD),
            ("input layernorm, 385", 385, 1, HIDDEN),
            ("input layernorm, 769", 769, 1, HIDDEN),
        ];
        // Warm: the first dispatch of a fresh pipeline pays for the driver's
        // first look at these buffers, which a decode loop pays once.
        let mut taken = shapes.map(|(_, rows, groups, width)| {
            cost(rows, groups, width);
            Vec::with_capacity(ROUNDS)
        });
        for _ in 0..ROUNDS {
            for (each, (_, rows, groups, width)) in taken.iter_mut().zip(shapes) {
                each.push(cost(rows, groups, width));
            }
        }

        for ((name, rows, groups, width), each) in shapes.iter().zip(&taken) {
            let mean = each.iter().sum::<Duration>() / each.len() as u32;
            eprintln!(
                "{name:<26} [{rows}, {}] in {groups} of {width}, one threadgroup each: \
                 {mean:.2?} a dispatch, best {:.2?}",
                groups * width,
                each.iter().min().expect("a round ran"),
            );
        }
    }

    /// A row of the width the model actually runs, which is what says whether a
    /// 4096-long reduction split across 32 lanes agrees with a serial one.
    #[test]
    fn a_hidden_state_sized_row_agrees_with_the_cpu() {
        let Some(device) = device() else { return };
        let norm = RmsNorm::new(&device).expect("the norm compiles");
        let width = 4096;
        let x: Vec<f32> = (0..width)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 12.0)
            .collect();
        let weight: Vec<f32> = (0..width).map(|i| 0.75 + (i % 11) as f32 / 32.0).collect();

        let got = LayerNorm::new(&device, &norm, &weight, 1e-5)
            .expect("the weight uploads")
            .forward(&x)
            .expect("the dispatch completes");

        let deviation = deviation(&got, &on_the_cpu(&x, &weight, 1e-5));
        eprintln!("a [1, {width}] norm: deviation {deviation:e}");
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }
}
