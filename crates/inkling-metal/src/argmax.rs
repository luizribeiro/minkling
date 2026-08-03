//! The last thing a step asks this process for: which id a row of logits names.
//!
//! Every other kernel in this crate is held to the CPU *within a tolerance*,
//! because both sides reduce the same products in different orders and float
//! addition does not associate. **This one is held to it exactly**, and it has
//! to be: [`greedy`](inkling_core::generate::greedy) is `top_k(logits, 1)` and
//! `top_k` takes the lower id where two logits agree, so the whole engine's
//! token identity rests on a rule about *equal* values. A reduction that broke a
//! tie the other way round would move a token while every aggregate metric in
//! this file stayed where it was.
//!
//! # The tie rule survives a tree because of what the tree combines
//!
//! A candidate is a value and the id holding it, and the operator that combines
//! two of them keeps the larger value and, where the two agree, the lower id.
//! **That operator is a maximum under a total order** — candidates ranked by
//! value ascending and, within a value, by id descending — and a maximum over a
//! set is the same element whatever order the set is folded in. So the tie rule
//! is not a property this kernel has to re-establish at every combining step: it
//! is a property of the operator, and every step is the same operator. Eighty
//! cores combining in an order the hardware chooses reach the answer a serial
//! scan reaches, and `the_answer_does_not_depend_on_how_many_threadgroups_
//! reduced_it` is that stated as a measurement rather than as this paragraph.
//!
//! # The order is `total_cmp`'s and not `>`'s
//!
//! `top_k` ranks with [`f32::total_cmp`], which is a *total* order over every
//! float: it separates `-0.0` from `0.0` and it places a NaN at one end or the
//! other by its sign. A kernel comparing with `>` agrees with it everywhere the
//! two are distinct values and disagrees exactly where a tie rule is what is
//! being tested — `-0.0 > 0.0` and `0.0 > -0.0` are both false, so a float
//! comparison calls them tied and takes the lower id where `total_cmp` takes the
//! `0.0`.
//!
//! So nothing here compares floats at all. [`ORDERED`] maps a float's bits to
//! the unsigned integer that ranks it, which is `total_cmp`'s own key with the
//! sign of the comparison moved from `i32` to `u32`, and the reduction compares
//! integers. That is what makes the equality this rests on bit equality rather
//! than numeric equality, and it is what makes NaN and `-0.0` answers rather
//! than caveats.
//!
//! # A padded id is not a token
//!
//! `lm_head` is `[201024, 4096]` and 200058 of those rows are vocabulary. The
//! projection is already cut there — see
//! [`PackedProjection::wrap_packed`](crate::PackedProjection::wrap_packed) — so
//! the 966 padded rows are never multiplied here. This kernel is told the
//! vocabulary anyway and ranks nothing past it, because a cut made in one place
//! is a cut that stops being made the day something else feeds this: an argmax
//! that ranked what it was handed would name an id the tokenizer does not spell,
//! out of a row whose padded entries are whatever the head's untrained rows
//! produce. `the_padding_past_the_vocabulary_never_wins` fills those slots with
//! infinities and asks.
//!
//! # Two dispatches, which is the opposite of what the norm decided
//!
//! [`crate::norm`] measured a split across threadgroups and declined it: a
//! second dispatch costs about four microseconds to encode against the six the
//! split would save over a 4096-wide row. **The same arithmetic at fifty times
//! the width goes the other way.** A row of the vocabulary is 800 KB, one
//! threadgroup is one core of eighty, and what a single group would spend
//! walking it is far past the launch a second dispatch costs. So a row is cut
//! into a run of threadgroups that each reduce a stripe of it, and a second
//! dispatch reduces what they left — with the same operator, which is why the
//! cut cannot move the answer.

use inkling_core::profile::{self, Op};

use crate::buffer::{Buffer, Element};
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "argmax";
const COMBINE_ENTRY: &str = "argmax_combine";

/// Threads one threadgroup of either dispatch holds.
///
/// [`crate::norm`]'s width rather than [`crate::router`]'s, and for the same
/// reason: what a group of this reduction waits on is the memory requests one
/// core can have outstanding, and 256 threads a stripe is what keeps a run of
/// them in flight. The second dispatch takes the same width over far less work,
/// which costs the threads that find nothing a barrier each and no reads at all.
const THREADS_PER_GROUP: usize = 256;

/// Entries the kernels' per-simdgroup arrays hold, which has to be a constant
/// where the number of simdgroups is not: 1024 threads is the widest threadgroup
/// any Apple GPU allows and 32 the narrowest simdgroup any reports, so 32
/// partials is the most a threadgroup can produce.
const MOST_SIMDGROUPS: usize = 32;

/// The most threadgroups one row's reduction is cut into, which is this
/// machine's own core count.
///
/// **The sweep `what_an_argmax_over_the_vocabulary_costs` prints is why the cut
/// exists at all, and it is emphatic.** A row of the vocabulary reduced by one
/// threadgroup is 284 microseconds — which is the figure this process's own
/// argmax was measured at, to within its spread, and is what says a device
/// argmax on one core buys exactly nothing. The same row at 80 is 10.9, at 128
/// is 9.8 and at 512 is 12.0.
///
/// **A number taken from the hardware rather than fitted to a row count.** The
/// best cut is not the same at every block: 128 at one and two rows, 80 at four
/// and nine, and the curve between 64 and 128 is flat at all four — so 80 is
/// never more than 11% off the best of them, where 128 is 18% off at nine rows.
/// A rule that tracked the block would be fitted to a sweep of four points on a
/// surface that is flat where it matters.
///
/// It is a ceiling and not the count: [`groups_a_row`] gives a row no more
/// groups than it has threads' worth of ids, so a short row is not cut into
/// stripes most of which are empty.
const GROUPS_A_ROW: usize = 80;

/// Threadgroups the first dispatch gives each row of `ids` values.
///
/// Two bounds and nothing between them. A row is never cut into more groups than
/// it has threadgroups' worth of ids, because a stripe shorter than a
/// threadgroup is threads that read nothing and wait at a barrier anyway; and it
/// is never cut into more than [`GROUPS_A_ROW`], which is where the sweep goes
/// flat. A cut of nothing is a cut of one, since a row is always reduced by
/// somebody.
fn groups_a_row(ids: usize) -> usize {
    ids.div_ceil(THREADS_PER_GROUP).clamp(1, GROUPS_A_ROW)
}

/// What of a buffer of logits an argmax ranks.
///
/// Two extents rather than one, and named rather than positional, because
/// exchanging them is an argmax that still answers: `stride` is how far apart
/// two rows of the buffer are and `ids` is how many of a row are vocabulary. The
/// two are equal for everything this engine dispatches today — the head is cut
/// at the unpadded vocabulary before a logit is ever formed — and they are two
/// numbers so that a caller which ever hands over a padded row is ranking the
/// tokens rather than the padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vocabulary {
    /// Ids a row holds, which is `unpadded_vocab_size` and is where the
    /// reference cuts.
    pub ids: usize,
    /// Values between one row of the buffer and the next.
    pub stride: usize,
}

impl Vocabulary {
    /// A row that is all vocabulary, which is every row this engine forms.
    pub fn of(ids: usize) -> Self {
        Self { ids, stride: ids }
    }
}

/// The largest value a reduction has seen and the id holding it.
///
/// **The value travels as the integer that orders it** rather than as the float
/// it came from — see [`ORDERED`] — so the equality the tie rule turns on is bit
/// equality and a NaN compares like anything else.
///
/// `#[repr(C)]` because the same two fields are declared in the source, and the
/// two declarations have to be one layout. `a_candidate_is_the_pair_the_source_
/// declares` is that asserted rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct Candidate {
    key: u32,
    at: u32,
}

// SAFETY: two `u32`s in a `#[repr(C)]` struct, which is eight bytes of any bit
// pattern with no padding between or after them.
unsafe impl Element for Candidate {}

/// The id an empty candidate carries, which is the identity of the operator that
/// combines two of them: it loses to every real candidate and beats none.
///
/// **A sentinel rather than a branch**, which is what a reduction over stripes
/// most of which may be empty wants — a threadgroup with nothing to offer offers
/// this and takes part in every step like any other. It is an id no row holds,
/// which is what [`GreedyArgmax::encoding`] refuses a vocabulary that would
/// reach.
const NONE: u32 = u32::MAX;

/// The two compiled kernels, which every argmax on a device shares.
///
/// One struct and not two beside each other, where [`crate::Router`] and
/// [`crate::RouterWeights`] are two: those are separate because they are
/// separate mistakes — one reads the biased scores and the other the raw logits
/// — and these two are the same reduction run twice, over the row and then over
/// what the row's stripes left. A test that puts a mutant tie rule through the
/// plumbing has to mutate both, because both combine candidates.
#[derive(Debug)]
pub struct GreedyArgmax {
    over: Kernel,
    combine: Kernel,
}

impl GreedyArgmax {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_sources(device, &source(), &combine_source())
    }

    /// [`GreedyArgmax::new`] out of source strings of the caller's own, which is
    /// how a test puts a deliberately wrong tie rule through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_sources(
        device: &Device,
        over: &str,
        combine: &str,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            over: device.compile(over, ENTRY)?,
            combine: device.compile(combine, COMBINE_ENTRY)?,
        })
    }

    /// `[rows, vocab.stride]` logits in, `[rows]` ids out, encoded into `batch`
    /// over a buffer a dispatch already left there.
    ///
    /// The output is a buffer of ids and not of logits, and that is the whole
    /// point: what crosses back is four bytes a row where the row itself is
    /// 800 KB, and the pass that ranks those 800 KB is a dispatch in the command
    /// buffer that wrote them.
    pub fn encode(
        &self,
        batch: &mut Batch<'_>,
        logits: &mut Buffer<f32>,
        vocab: Vocabulary,
    ) -> Result<Buffer<u32>, MetalError> {
        self.encoding(batch, logits, vocab, groups_a_row(vocab.ids))
    }

    /// The same argmax, cut into a stated number of threadgroups a row.
    ///
    /// Here for the cases that are about the cut rather than about the answer:
    /// the tie ones, which need a boundary to place a tie across, and the sweep
    /// [`GROUPS_A_ROW`] was chosen from. Nothing in the engine states its own.
    pub(crate) fn encoding(
        &self,
        batch: &mut Batch<'_>,
        logits: &mut Buffer<f32>,
        vocab: Vocabulary,
        groups: usize,
    ) -> Result<Buffer<u32>, MetalError> {
        let _timed = profile::scope(Op::Encode);
        let Vocabulary { ids, stride } = vocab;
        assert!(ids > 0, "an argmax over no ids");
        assert!(ids <= stride, "{ids} ids do not fit a row of {stride}");
        // The one bound the sentinel needs, and it is checked before the buffer
        // is measured so that a caller naming an impossible vocabulary is told
        // which of the two it got wrong. Unreachable through any real call —
        // four billion logits is 16 GB of one row — and a vocabulary that
        // reached it would make the empty candidate an id.
        assert!(
            ids < NONE as usize,
            "{ids} ids reach the id an empty candidate carries"
        );
        assert!(groups > 0, "a row is reduced by some threadgroups");
        assert_eq!(
            logits.len() % stride,
            0,
            "{} logits are not whole rows of {stride}",
            logits.len()
        );
        let rows = logits.len() / stride;
        assert!(rows > 0, "an argmax over no rows");

        let fields = [
            extent(rows, "the rows of a call"),
            extent(ids, "the ids a row holds"),
            extent(stride, "the width of a row of logits"),
            extent(groups, "the threadgroups a row is cut into"),
        ];
        let mut shape = batch.device().inline(&fields)?;
        let mut partials = batch.device().zeroed::<Candidate>(rows * groups)?;
        let mut picks = batch.device().zeroed::<u32>(rows)?;

        // **The row and not the buffer.** A padded row is bound whole and the
        // ids of it are read, which is the same distinction the router's two
        // dispatches declare over a gate they are bound all 258 of.
        let stripes = size_of::<Candidate>() * partials.len();
        let ranked = size_of::<f32>() * rows * ids;
        let taken = size_of::<u32>() * picks.len();
        batch.add(
            &self.over,
            &[shape.arg(), logits.arg(), partials.arg()],
            Grid::new(rows * groups * THREADS_PER_GROUP, THREADS_PER_GROUP),
            ranked + stripes,
        )?;
        batch.add(
            &self.combine,
            &[shape.arg(), partials.arg(), picks.arg()],
            Grid::new(rows * THREADS_PER_GROUP, THREADS_PER_GROUP),
            stripes + taken,
        )?;
        Ok(picks)
    }

    /// The same argmax submitted on its own, for a caller with nothing to batch
    /// it against — which is the cases here and nothing in the engine.
    pub fn picks(
        &self,
        device: &Device,
        logits: &[f32],
        vocab: Vocabulary,
    ) -> Result<Vec<u32>, MetalError> {
        self.picking(device, logits, vocab, groups_a_row(vocab.ids))
    }

    /// The same, cut into a stated number of threadgroups a row.
    pub(crate) fn picking(
        &self,
        device: &Device,
        logits: &[f32],
        vocab: Vocabulary,
        groups: usize,
    ) -> Result<Vec<u32>, MetalError> {
        let mut input = device.buffer(logits)?;
        let mut batch = device.batch()?;
        let picks = self.encoding(&mut batch, &mut input, vocab, groups)?;
        batch.wait()?;
        Ok(picks.to_vec())
    }
}

/// The first dispatch, with the bound its threadgroup array is sized by and the
/// ranking it compares under written into its prelude rather than spelled twice.
fn source() -> String {
    format!("{}{BODY}", prelude())
}

/// The second, over the same prelude — the same operator, reducing what the
/// stripes of a row left rather than the row.
fn combine_source() -> String {
    format!("{}{COMBINE}", prelude())
}

/// Everything the two kernels share, with the two constants substituted in.
///
/// [`ORDERED`] arrives by substitution rather than by being written in
/// [`PRELUDE`] so that a case mutating the ranking can name the string it is
/// replacing instead of copying it — a mutation that silently stopped matching
/// would be a case asserting a difference between one kernel and itself.
fn prelude() -> String {
    format!(
        "constant uint MOST_SIMDGROUPS = {MOST_SIMDGROUPS};\n{}",
        PRELUDE.replace(ORDERED_KEY, ORDERED)
    )
}

/// Where [`ORDERED`] goes in [`PRELUDE`].
const ORDERED_KEY: &str = "ORDERED_KEY";

/// `f32::total_cmp`'s ranking, as an unsigned integer, written where both
/// kernels and the case that pins it can name the same string.
///
/// The map is: a float with its sign bit set is every bit of it flipped, and one
/// without is the sign bit set. That is the standard order-preserving map from
/// float bits to unsigned integers, and what makes it *this* engine's order is
/// what it agrees with. `total_cmp` builds `bits ^ ((bits >> 31) >> 1)` and
/// compares the result as a signed integer; this is that value exclusive-ored
/// with the sign bit, and exclusive-oring the sign bit is exactly the map from
/// the signed order to the unsigned one. So the two rank every pair of floats
/// the same way, `-0.0` under `0.0` and a NaN at whichever end its sign puts it.
const ORDERED: &str = "(bits & 0x80000000u) ? ~bits : (bits | 0x80000000u)";

/// Everything both kernels share: the shape they take, the candidate they
/// combine, and the operator and the threadgroup reduction that combine it.
const PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint rows;
    uint ids;
    uint stride;
    uint groups;
};

/// The largest value a reduction has seen and the id holding it.
struct Candidate {
    uint key;
    uint at;
};

/// The id an empty candidate carries, which no row holds.
constant uint NONE = 0xffffffffu;

/// `f32::total_cmp`'s ranking of a float, as an unsigned integer — see the
/// module documentation for why nothing here compares floats.
static inline uint ordered(float value) {
    const uint bits = as_type<uint>(value);
    return ORDERED_KEY;
}

/// Whether a reduction combining `contender` with `held` keeps the contender:
/// the larger value, and where the two values agree the lower id.
///
/// **This is the tie rule and it is the whole of it.** It is a maximum under the
/// total order that ranks a candidate by `key` ascending and by `at` descending,
/// which is a total order because `(key, at)` pairs are distinct in `at`; so the
/// operator is associative and commutative, and a fold over a set of candidates
/// is the same candidate whatever order the fold takes them in. Every step of
/// every reduction below — a thread's own stripe, a simdgroup, a threadgroup,
/// and the second dispatch over what the threadgroups left — is this operator.
///
/// An empty candidate is `{0, NONE}` and is the identity: no real candidate has
/// an id of `NONE`, so a real candidate always beats it and it beats no real
/// candidate. That is what lets a stripe with nothing in it take part in the
/// reduction rather than be branched around.
static inline bool beats(Candidate contender, Candidate held) {
    return contender.key > held.key
        || (contender.key == held.key && contender.at < held.at);
}

/// One threadgroup's candidates reduced to one, by the same operator.
///
/// Two simdgroup reductions rather than one, for [`router_top_k`]'s reason: a
/// maximum over the keys alone cannot express which id holds it, so the largest
/// key comes back first and the lowest id holding it second. Every thread of the
/// group then walks the simdgroups' partials for itself, which is cheaper than
/// reducing them once and broadcasting at 32 entries and needs one barrier
/// rather than two.
///
/// Every thread of the threadgroup reaches this, whatever it found: a thread
/// with nothing offers the empty candidate, which is what keeps both simdgroup
/// reductions over the whole simdgroup and the barrier uniform.
static Candidate reduced(
    Candidate mine,
    threadgroup Candidate *partials,
    uint lane,
    uint simd,
    uint simds
) {
    const uint top = simd_max(mine.key);
    const uint first = simd_min(mine.key == top ? mine.at : NONE);
    if (lane == 0) {
        partials[simd].key = top;
        partials[simd].at = first;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Candidate best = { 0u, NONE };
    for (uint s = 0; s < simds; ++s) {
        if (beats(partials[s], best)) {
            best = partials[s];
        }
    }
    return best;
}
"#;

/// The first dispatch: a stripe of a row to each threadgroup.
const BODY: &str = r#"
/// The largest candidate of each threadgroup's stripe of each row.
///
/// **A stripe is contiguous**, which is what makes a threadgroup boundary a
/// place a tie can be put: the ids one group reduces are a run of the row, so a
/// tie whose two holders straddle `span` is one only the second dispatch can
/// break. `a_tie_that_straddles_a_threadgroup_is_taken_at_its_lowest_id` is
/// where that is placed on purpose.
///
/// The row is `stride` wide and the first `ids` of it are ranked. A padded row's
/// last entries are bound and never read, which is what keeps an id past the
/// vocabulary out of the answer rather than out of the buffer.
kernel void argmax(
    constant Shape &shape [[buffer(0)]],
    device const float *logits [[buffer(1)]],
    device Candidate *partials [[buffer(2)]],
    uint slot [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]
) {
    threadgroup Candidate seen[MOST_SIMDGROUPS];

    // A whole threadgroup turns away or none of it does, which is what makes the
    // barrier inside `reduced` uniform. The grid gives one threadgroup to each
    // stripe of each row, so this is unreachable; a bounds check on `local`
    // instead would leave some threads at a barrier and others past it, which is
    // undefined rather than slow.
    if ((ulong)slot >= (ulong)shape.rows * shape.groups) {
        return;
    }
    const uint row = slot / shape.groups;
    const uint group = slot % shape.groups;
    device const float *values = logits + (ulong)row * shape.stride;

    // In `ulong` for the reason the pointer above is: `ids` is a `uint` and the
    // last stripe of a cut reaches past it, so a row wide enough would wrap the
    // multiply, the sum, or the stride the loop below adds. Nothing this engine
    // dispatches is within a threadgroup of that width; what the wider type buys
    // is that the bound does not have to be argued.
    const ulong span = ((ulong)shape.ids + shape.groups - 1) / shape.groups;
    const uint from = (uint)min((ulong)group * span, (ulong)shape.ids);
    const uint to = (uint)min((ulong)from + span, (ulong)shape.ids);

    Candidate mine = { 0u, NONE };
    for (ulong at = (ulong)from + local; at < to; at += threads) {
        const Candidate here = { ordered(values[at]), (uint)at };
        if (beats(here, mine)) {
            mine = here;
        }
    }

    const Candidate best = reduced(mine, seen, lane, simd, simds);
    if (local == 0) {
        partials[slot] = best;
    }
}
"#;

/// The second dispatch: a row's stripes reduced to the row's own id.
const COMBINE: &str = r#"
/// The id each row names, out of what the first dispatch's threadgroups left.
///
/// The same operator over `groups` candidates a row, which is what says the cut
/// cannot move the answer: this fold and the folds inside every stripe are one
/// associative operator, so the whole is a maximum over the row's candidates
/// however it was divided.
///
/// A row of no candidates cannot arise — a row is always cut into at least one
/// stripe — and a row whose every stripe was empty would answer `NONE`, which is
/// the vocabulary's own width away from any id and so a failure that is visible
/// rather than a plausible token.
kernel void argmax_combine(
    constant Shape &shape [[buffer(0)]],
    device const Candidate *partials [[buffer(1)]],
    device uint *picks [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]
) {
    threadgroup Candidate seen[MOST_SIMDGROUPS];

    if (row >= shape.rows) {
        return;
    }
    device const Candidate *stripes = partials + (ulong)row * shape.groups;

    Candidate mine = { 0u, NONE };
    for (uint stripe = local; stripe < shape.groups; stripe += threads) {
        if (beats(stripes[stripe], mine)) {
            mine = stripes[stripe];
        }
    }

    const Candidate best = reduced(mine, seen, lane, simd, simds);
    if (local == 0) {
        picks[row] = best.at;
    }
}
"#;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::testing::device;
    use inkling_core::greedy;

    /// The vocabulary's own width, which is what every case that can afford it
    /// is measured at.
    ///
    /// **A row 800 KB wide costs microseconds and buys the shapes.** The stripes
    /// a cut leaves, the simdgroups inside one and the strides inside a thread
    /// are all places a tie can sit, and at a toy width most of them are the same
    /// place — so the tie cases below place their ties by arithmetic on this
    /// number rather than on a width chosen to make a test quick.
    const IDS: usize = 200_058;

    /// The padded head's own width, which is 966 rows further on and is what a
    /// row of this engine's logits is *not*.
    const PADDED: usize = 201_024;

    /// The id the host takes, which is what every case here is measured against.
    ///
    /// [`greedy`] and not an argmax written out beside it: the rule under test is
    /// `top_k`'s own — the largest under `total_cmp`, ties to the lower id — and a
    /// second spelling of it here would hold this kernel to a rule the engine
    /// does not use.
    fn on_the_cpu(logits: &[f32]) -> u32 {
        greedy(logits) as u32
    }

    /// A row whose values are all different and none of them tied, so that a
    /// case about something else is not quietly a case about the tie rule.
    fn noisy(ids: usize, seed: usize) -> Vec<f32> {
        (0..ids)
            .map(|i| ((i * 37 + seed * 101) % 199_999) as f32 / 512.0 - 195.0)
            .collect()
    }

    /// A row of `ids` values whose largest is `tied` at every id in `at`, and
    /// nowhere else.
    ///
    /// The values under it are not constant, so a kernel that read the wrong
    /// entry is a different answer rather than the same one.
    fn tied_at(ids: usize, at: &[usize]) -> Vec<f32> {
        let mut row: Vec<f32> = (0..ids).map(|i| -1.0 - (i % 7) as f32).collect();
        for id in at {
            row[*id] = 2.5;
        }
        row
    }

    /// One place a tie can sit in this reduction: how the row is cut, where the
    /// two holders are, and which of the two reductions has to break it.
    struct Straddle {
        what: &'static str,
        groups: usize,
        at: [usize; 2],
        decided_by: Tie,
    }

    /// Which of the two reductions breaks a tie: the one inside a simdgroup,
    /// which is `simd_min` over the lanes holding the largest key, or every other
    /// one, which is `beats`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Tie {
        WithinASimdgroup,
        Between,
    }

    /// The stripe a threadgroup owns at a given cut.
    fn span(groups: usize) -> usize {
        IDS.div_ceil(groups)
    }

    /// **Every place a tie can sit in this reduction**, and the cut that puts it
    /// there.
    ///
    /// The offsets are arithmetic on the cut rather than round numbers. A
    /// threadgroup's stripe is `span(groups)` ids and a thread of it takes every
    /// [`THREADS_PER_GROUP`]th of them, so id `from + local` is lane `local % 32`
    /// of simdgroup `local / 32` — which is what makes 7 and 8 one simdgroup, 7
    /// and 39 two, and 7 and 263 one thread twice.
    ///
    /// **The last three are the cross-threadgroup tie and there are three of them
    /// because the second dispatch is a reduction too.** It gives a thread to
    /// each stripe, so where the two holders land in it depends on how many
    /// stripes there are: at a cut of four they are two lanes of one simdgroup, at
    /// 64 they are two simdgroups, and at 512 they are one thread on two of its
    /// own strides. A table that only had the first of those would say the tie
    /// rule holds across cores at one cut and never ask about the others.
    fn straddles() -> [Straddle; 6] {
        [
            Straddle {
                what: "within a simdgroup",
                groups: 4,
                at: [7, 8],
                decided_by: Tie::WithinASimdgroup,
            },
            Straddle {
                what: "across a thread's own strides",
                groups: 4,
                at: [7, 7 + THREADS_PER_GROUP],
                decided_by: Tie::Between,
            },
            Straddle {
                what: "across simdgroups",
                groups: 4,
                at: [7, 39],
                decided_by: Tie::Between,
            },
            Straddle {
                what: "across threadgroups, one simdgroup of the second dispatch",
                groups: 4,
                at: [7, span(4) + 3],
                decided_by: Tie::WithinASimdgroup,
            },
            Straddle {
                what: "across threadgroups, two simdgroups of the second dispatch",
                groups: 64,
                at: [7, 40 * span(64) + 3],
                decided_by: Tie::Between,
            },
            Straddle {
                what: "across threadgroups, two strides of the second dispatch",
                groups: 512,
                at: [7, THREADS_PER_GROUP * span(512) + 3],
                decided_by: Tie::Between,
            },
        ]
    }

    /// The shipped pair of kernels.
    fn kernels(device: &Device) -> GreedyArgmax {
        GreedyArgmax::new(device).expect("the argmax compiles")
    }

    /// The same pair with `mutate` applied to both sources, which is how a case
    /// puts a deliberately wrong tie rule through the same plumbing as the right
    /// one.
    ///
    /// **Both, because both combine candidates.** The operator is in the prelude
    /// the two entries share, so a mutation reaching only the first dispatch
    /// would leave the second breaking ties correctly and the case would be
    /// asserting less than it reads.
    fn mutant(device: &Device, mutate: impl Fn(&str) -> String) -> GreedyArgmax {
        let (over, combine) = (mutate(&source()), mutate(&combine_source()));
        assert_ne!(over, source(), "the mutation changed nothing");
        assert_ne!(combine, combine_source(), "the mutation changed nothing");
        GreedyArgmax::from_sources(device, &over, &combine).expect("the mutant compiles")
    }

    /// The tie broken the other way inside a simdgroup: the highest lane of a
    /// tied run rather than the lowest.
    fn taking_the_last_lane(source: &str) -> String {
        source.replace(
            "simd_min(mine.key == top ? mine.at : NONE)",
            "simd_max(mine.key == top ? mine.at : 0u)",
        )
    }

    /// The tie broken the other way everywhere else: a thread's own stripe, the
    /// pass over the simdgroups' partials, and the second dispatch over the
    /// threadgroups'.
    fn taking_the_last_candidate(source: &str) -> String {
        source.replace("contender.at < held.at", "contender.at > held.at")
    }

    /// **The whole claim, at the width the engine runs it at**: the id a
    /// dispatch names is the id `greedy` names, over rows with no tie in them at
    /// all.
    ///
    /// A row of a distribution rather than of a tie, which is deliberately the
    /// weakest case in this module: a reduction that is correct on distinct
    /// values proves nothing about the rule the engine's token identity rests
    /// on, and every case under this one is about that rule.
    #[test]
    fn a_dispatch_takes_the_id_the_host_takes() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);

        for seed in 0..4 {
            let row = noisy(IDS, seed);
            let got = argmax
                .picks(&device, &row, Vocabulary::of(IDS))
                .expect("the dispatch completes");
            assert_eq!(got, vec![on_the_cpu(&row)], "seed {seed}");
        }
    }

    /// Every id of a row carries the same value, so the answer is entirely the
    /// tie rule and nothing else — 200058 ids agreeing bit for bit across every
    /// stripe, simdgroup and stride of the reduction.
    #[test]
    fn a_row_of_ties_is_taken_at_its_lowest_id() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let row = vec![0.25; IDS];

        let got = argmax
            .picks(&device, &row, Vocabulary::of(IDS))
            .expect("the dispatch completes");

        assert_eq!(got, vec![0]);
        assert_eq!(
            got,
            vec![on_the_cpu(&row)],
            "the host breaks it the same way"
        );
    }

    /// **A tie in each of the four places this reduction has to break one**, and
    /// the cut is stated so that a threadgroup boundary is a number a case can
    /// place a value either side of.
    ///
    /// The last case is the one the whole milestone turns on: its two holders are
    /// reduced by different threadgroups, which are different cores, and nothing
    /// but the second dispatch ever sees both.
    #[test]
    fn a_tie_that_straddles_a_threadgroup_is_taken_at_its_lowest_id() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);

        for straddle in straddles() {
            let Straddle {
                what, groups, at, ..
            } = straddle;
            let row = tied_at(IDS, &at);
            let got = argmax
                .picking(&device, &row, Vocabulary::of(IDS), groups)
                .expect("the dispatch completes");

            assert_eq!(got, vec![at[0] as u32], "{what}");
            assert_eq!(got, vec![on_the_cpu(&row)], "{what}: the host disagrees");
        }
    }

    /// The lower id of a tie is not always the one in the first stripe, and a
    /// second dispatch that took the first non-empty candidate rather than the
    /// lowest would pass every case above.
    #[test]
    fn a_tie_between_two_later_threadgroups_is_taken_at_its_lowest_id() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let span = span(4);
        let row = tied_at(IDS, &[2 * span + 5, span + 5, 3 * span + 1]);

        let got = argmax
            .picking(&device, &row, Vocabulary::of(IDS), 4)
            .expect("the dispatch completes");

        assert_eq!(got, vec![(span + 5) as u32]);
        assert_eq!(got, vec![on_the_cpu(&row)]);
    }

    /// **Both mutations, and each against the ties it owns.** A tie inside a
    /// simdgroup is broken by `simd_min` and every other tie by `beats`, so a
    /// case that only showed *some* answer moving would not say the two
    /// reductions each carry the rule.
    ///
    /// The assertion is two-sided: the mutation the case names has to move the
    /// answer, and the other one has to leave it where it was. A mutation that
    /// moved everything would mean the cases above are pinned by something
    /// coarser than the rule they are written for.
    #[test]
    fn taking_the_last_of_a_tied_run_takes_a_different_id() {
        let Some(device) = device() else { return };
        let mutants = [
            (Tie::WithinASimdgroup, mutant(&device, taking_the_last_lane)),
            (Tie::Between, mutant(&device, taking_the_last_candidate)),
        ];

        for (broken, argmax) in &mutants {
            for straddle in straddles() {
                let Straddle {
                    what,
                    groups,
                    at,
                    decided_by,
                } = straddle;
                let row = tied_at(IDS, &at);
                let got = argmax
                    .picking(&device, &row, Vocabulary::of(IDS), groups)
                    .expect("the dispatch completes");

                match decided_by == *broken {
                    true => assert_eq!(
                        got,
                        vec![at[1] as u32],
                        "{what}: {broken:?} took neither end of the tied run"
                    ),
                    false => assert_eq!(
                        got,
                        vec![at[0] as u32],
                        "{what}: {broken:?} reached a tie it does not decide"
                    ),
                }
            }
        }
    }

    /// **A tie in a row that is not the first**, which every case above is
    /// silent about: all of them rank one row, and a first dispatch that wrote
    /// its stripes at `group * rows + row` rather than at `row * groups + group`
    /// would answer all of them correctly and mix two rows' candidates on any
    /// block.
    ///
    /// Each row of the block ties in a different place and at a different pair
    /// of ids, so a row that read another's stripes is a different id rather
    /// than the same one.
    #[test]
    fn a_tie_in_a_later_row_of_a_block_is_taken_at_its_lowest_id() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let groups = 8;
        let stripe = span(groups);
        let rows: [Vec<usize>; 3] = [
            vec![11, 12, 3 * stripe + 4],
            vec![5 * stripe + 9, 6 * stripe + 40, 7 * stripe + 1],
            vec![stripe + 2, stripe + 3, 4 * stripe + 7],
        ];

        let block: Vec<f32> = rows.iter().flat_map(|at| tied_at(IDS, at)).collect();
        let got = argmax
            .picking(&device, &block, Vocabulary::of(IDS), groups)
            .expect("the dispatch completes");

        let want: Vec<u32> = rows
            .iter()
            .map(|at| *at.iter().min().expect("a row ties somewhere") as u32)
            .collect();
        assert_eq!(got, want);
        assert!(
            want[0] != want[1] && want[1] != want[2],
            "two rows that answered alike would prove nothing"
        );
        let apart: Vec<u32> = block.chunks_exact(IDS).map(on_the_cpu).collect();
        assert_eq!(got, apart, "the rows agree with the host one at a time");
    }

    /// The two magnitudes the bit map has to carry that no other case here
    /// reaches: a subnormal, whose exponent field is empty, and an infinity,
    /// whose mantissa is.
    ///
    /// **Both as the peak of the ranked region rather than beside it.** The
    /// padding case above fills its excluded slots with infinities, which says
    /// nothing about whether an infinity *wins* correctly; and a subnormal is
    /// where a ranking built on anything but the bits — a comparison that
    /// flushed one to zero, say — would part company with `total_cmp` while
    /// looking right everywhere else.
    #[test]
    fn a_subnormal_and_an_infinity_rank_where_the_host_ranks_them() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let taken = |row: &[f32]| {
            argmax
                .picks(&device, row, Vocabulary::of(row.len()))
                .expect("the dispatch completes")
        };

        // A row of subnormals, whose largest is the peak — and two of them tied
        // beneath it, so the rule is exercised at a magnitude a flush would
        // erase.
        let mut small: Vec<f32> = (0..4096)
            .map(|i| f32::MIN_POSITIVE * (i % 5) as f32 / 8.0)
            .collect();
        small[2001] = f32::MIN_POSITIVE / 2.0;
        assert_eq!(taken(&small), vec![on_the_cpu(&small)]);
        assert_eq!(taken(&small), vec![4], "the largest subnormal is at 4");

        let mut vast = noisy(4096, 3);
        vast[77] = f32::INFINITY;
        vast[1200] = f32::INFINITY;
        assert_eq!(taken(&vast), vec![77], "an infinity outranks every finite");
        assert_eq!(taken(&vast), vec![on_the_cpu(&vast)]);

        let mut below = vec![f32::NEG_INFINITY; 4096];
        below[9] = -f32::MAX;
        assert_eq!(
            taken(&below),
            vec![9],
            "a negative infinity is the smallest"
        );
        assert_eq!(taken(&below), vec![on_the_cpu(&below)]);
    }

    /// **966 slots that must never win a reduction.** `lm_head` is 201024 rows
    /// and 200058 of them are vocabulary; the projection is cut there, and this
    /// is what says the argmax is cut there too.
    ///
    /// The padding is filled with infinities rather than with plausible values,
    /// because what is being asked is whether those entries are read at all —
    /// and an argmax that ranked them would answer with an id no tokenizer
    /// spells rather than with a token that is merely wrong.
    #[test]
    fn the_padding_past_the_vocabulary_never_wins() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let vocab = Vocabulary {
            ids: IDS,
            stride: PADDED,
        };

        let mut row = noisy(PADDED, 5);
        row[IDS..].fill(f32::INFINITY);
        let got = argmax
            .picks(&device, &row, vocab)
            .expect("the dispatch completes");

        assert!((got[0] as usize) < IDS, "a padded id was taken: {}", got[0]);
        assert_eq!(got, vec![on_the_cpu(&row[..IDS])]);
        // And the infinities are there to be found, so a case that passed
        // because the padding held nothing would fail here.
        assert_eq!(on_the_cpu(&row), IDS as u32, "the padding was not the peak");
    }

    /// A tie whose two holders straddle the cut at the vocabulary, which is the
    /// one boundary in this kernel where the lower id is not the rule — the
    /// padded id loses because it is padding and not because it is higher.
    #[test]
    fn a_tie_at_the_padding_edge_is_taken_inside_the_vocabulary() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let vocab = Vocabulary {
            ids: IDS,
            stride: PADDED,
        };

        let mut row = tied_at(PADDED, &[IDS - 1, IDS, PADDED - 1]);
        // The padded holders carry more than the tie, so a kernel reaching past
        // the vocabulary loses on the value rather than on the id.
        row[IDS] = 3.5;
        let got = argmax
            .picks(&device, &row, vocab)
            .expect("the dispatch completes");

        assert_eq!(got, vec![(IDS - 1) as u32]);
        assert_eq!(got, vec![on_the_cpu(&row[..IDS])]);
    }

    /// **`-0.0` and `0.0` are one number to a float comparison and two to the
    /// ranking this engine uses.** `top_k` ranks with `total_cmp`, which puts
    /// `-0.0` below `0.0`; a kernel comparing with `>` would call them tied and
    /// take the lower id, which here is the `-0.0`.
    ///
    /// So this case is the one that separates the ranking from the tie rule: the
    /// answer is the *higher* of the two ids, and no reduction that got the tie
    /// rule right and the order wrong can reach it.
    #[test]
    fn minus_zero_ranks_under_zero_the_way_the_host_ranks_it() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let mut row = vec![-1.0f32; 4096];
        row[5] = -0.0;
        row[9] = 0.0;

        let got = argmax
            .picks(&device, &row, Vocabulary::of(row.len()))
            .expect("the dispatch completes");

        assert_eq!(got, vec![9], "the zeros were ranked as one value");
        assert_eq!(got, vec![on_the_cpu(&row)]);
        // Both zeros are the peak by any float comparison, so a kernel using one
        // would answer 5 — which is what says this case is about the order.
        assert!(row.iter().all(|value| *value <= 0.0));
    }

    /// A NaN ranks at whichever end of `total_cmp` its sign puts it, and both
    /// ends are asked.
    ///
    /// **Not a value a logit reaches, and that is not the point.** The claim
    /// this kernel makes is that it takes the id `greedy` takes for any row it
    /// can be handed, and a NaN is where an ordering that is "close enough on
    /// real data" and one that is `total_cmp` part company by the whole
    /// vocabulary.
    #[test]
    fn a_nan_ranks_where_its_sign_puts_it() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let taken = |row: &[f32]| {
            argmax
                .picks(&device, row, Vocabulary::of(row.len()))
                .expect("the dispatch completes")
        };

        let mut above = noisy(4096, 1);
        above[11] = f32::NAN;
        assert_eq!(taken(&above), vec![11], "a positive NaN is the largest");
        assert_eq!(taken(&above), vec![on_the_cpu(&above)]);

        let mut below = vec![-1e30f32; 4096];
        below[3] = -f32::NAN;
        below[100] = -1e29;
        assert_eq!(taken(&below), vec![100], "a negative NaN is the smallest");
        assert_eq!(taken(&below), vec![on_the_cpu(&below)]);
    }

    /// The bits of a float ordered as they lie rank every negative value above
    /// every positive one, which is the mutation that says the map in [`ORDERED`]
    /// is doing something rather than being a spelling of the identity.
    #[test]
    fn ranking_the_bits_as_they_lie_puts_the_negatives_at_the_top() {
        let Some(device) = device() else { return };
        let row = noisy(4096, 2);
        let vocab = Vocabulary::of(row.len());

        let taken = kernels(&device)
            .picks(&device, &row, vocab)
            .expect("the dispatch completes");
        let naive = mutant(&device, |source| source.replace(ORDERED, "bits"))
            .picks(&device, &row, vocab)
            .expect("the mutant completes");

        assert_eq!(taken, vec![on_the_cpu(&row)]);
        assert_ne!(naive, taken, "the ranking was the bits all along");
        assert!(
            row[naive[0] as usize] < 0.0 && row[taken[0] as usize] > 0.0,
            "the naive order did not rank a negative value at the top"
        );
    }

    /// **The associativity claim, measured rather than argued.** The same row
    /// cut into every number of threadgroups from one to more than it has ids
    /// gives one answer, because the operator every one of those folds uses is
    /// the same associative one.
    ///
    /// Cuts that do not divide the row, cuts of one — where there is nothing for
    /// the second dispatch to combine — and cuts past the row, where most stripes
    /// are empty and the identity candidate is the whole of what they contribute.
    #[test]
    fn the_answer_does_not_depend_on_how_many_threadgroups_reduced_it() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let ids = 5_000;
        let row = tied_at(ids, &[97, 1_499, 1_500, 3_001, 4_999]);
        let vocab = Vocabulary::of(ids);

        for groups in [1usize, 2, 3, 5, 7, 8, 16, 64, 128, 999, 6_000] {
            let got = argmax
                .picking(&device, &row, vocab, groups)
                .expect("the dispatch completes");
            assert_eq!(got, vec![97], "cut into {groups} threadgroups");
        }
        assert_eq!(on_the_cpu(&row), 97);
    }

    /// A call is rows and each of them names its own id, which is what a
    /// speculative round's block asks for — and what a reduction over the whole
    /// buffer would answer with one number for.
    ///
    /// The rows are all different and their peaks are in different places, so a
    /// row read from the wrong offset is a different id rather than the same one.
    #[test]
    fn each_row_of_a_block_names_its_own_id() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let ids = 3_000;
        let rows = 9;

        let mut block: Vec<f32> = Vec::new();
        for row in 0..rows {
            let mut values = noisy(ids, row);
            values[row * 211 + 13] = 1e6;
            block.extend(values);
        }

        let got = argmax
            .picks(&device, &block, Vocabulary::of(ids))
            .expect("the dispatch completes");

        let want: Vec<u32> = (0..rows).map(|row| (row * 211 + 13) as u32).collect();
        assert_eq!(got, want);
        let apart: Vec<u32> = block.chunks_exact(ids).map(on_the_cpu).collect();
        assert_eq!(got, apart, "the rows agree with the host one at a time");
    }

    /// What the bandwidth column divides by, against what the two dispatches
    /// read: the ids of every row, the stripes the first left, and the ids the
    /// second names.
    ///
    /// **The padding is bound and not charged**, which is the same distinction
    /// the router's selection makes over the two shared logits it never ranks: a
    /// dispatch is charged what it moves and not what it was handed.
    #[test]
    fn a_dispatch_declares_the_ids_it_ranks() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        let (ids, stride, rows) = (3_000usize, 3_200usize, 4usize);
        let groups = groups_a_row(ids);
        let mut logits = device
            .buffer(&vec![0.5f32; rows * stride])
            .expect("the logits upload");

        let moved = crate::testing::moved(&device, |batch| {
            argmax
                .encode(batch, &mut logits, Vocabulary { ids, stride })
                .expect("the argmax encodes");
        });

        let stripes = size_of::<Candidate>() * rows * groups;
        assert_eq!(
            moved as usize,
            size_of::<f32>() * rows * ids + 2 * stripes + size_of::<u32>() * rows,
        );
        assert!(
            (moved as usize) < size_of::<f32>() * rows * stride + 2 * stripes + 4 * rows,
            "the padding this row was bound was charged as ids"
        );
    }

    /// The two fields the source declares, which the Rust side allocates a
    /// buffer of and never reads: a layout the two disagreed about would be a
    /// second dispatch reducing candidates that are not the ones the first
    /// wrote.
    #[test]
    fn a_candidate_is_the_pair_the_source_declares() {
        assert_eq!(size_of::<Candidate>(), 2 * size_of::<u32>());
        assert_eq!(align_of::<Candidate>(), align_of::<u32>());
        assert_eq!(
            Candidate {
                key: 0,
                at: u32::MAX
            }
            .at,
            NONE,
            "the empty candidate's id is the one the source spells"
        );
    }

    /// The rule a row's cut is made by, at the shapes it is made for and at the
    /// two ends it has to be defensive about.
    ///
    /// A row is reduced by somebody, however short; it is never cut into more
    /// stripes than it has threadgroups' worth of ids; and a row of the
    /// vocabulary reaches the ceiling rather than sitting under it.
    #[test]
    fn a_row_is_cut_into_the_threadgroups_its_ids_fill() {
        assert_eq!(
            groups_a_row(IDS),
            GROUPS_A_ROW,
            "the vocabulary fills the cut"
        );
        for ids in [1usize, 2, 255, 256, 257, 4_096, IDS, PADDED, usize::MAX] {
            let groups = groups_a_row(ids);
            assert!((1..=GROUPS_A_ROW).contains(&groups), "{ids}: {groups}");
            assert!(
                groups == GROUPS_A_ROW || groups * THREADS_PER_GROUP >= ids,
                "{ids}: {groups} stripes leave threads with nothing to read"
            );
        }
        assert_eq!(groups_a_row(1), 1);
        assert_eq!(groups_a_row(THREADS_PER_GROUP + 1), 2);
    }

    /// A vocabulary wider than the row it is cut from, which is an argmax that
    /// would read past the end of a buffer rather than answer wrongly.
    #[test]
    #[should_panic(expected = "ids do not fit a row of")]
    fn a_vocabulary_wider_than_its_row_is_refused() {
        let Some(device) = device() else {
            panic!("ids do not fit a row of: no device to ask")
        };
        let _ = kernels(&device).picks(
            &device,
            &[0.0; 64],
            Vocabulary {
                ids: 65,
                stride: 64,
            },
        );
    }

    /// A vocabulary reaching the id an empty candidate carries, which would make
    /// the identity of the reduction a token.
    ///
    /// Unreachable through any real call and checked before the buffer is
    /// measured, so that a caller naming it is told which of the two it got
    /// wrong rather than that its buffer is the wrong length.
    #[test]
    #[should_panic(expected = "reach the id an empty candidate carries")]
    fn a_vocabulary_reaching_the_empty_candidates_id_is_refused() {
        let Some(device) = device() else {
            panic!("reach the id an empty candidate carries: no device to ask")
        };
        let _ = kernels(&device).picks(
            &device,
            &[0.0; 64],
            Vocabulary {
                ids: NONE as usize,
                stride: NONE as usize,
            },
        );
    }

    /// **What an argmax over the vocabulary costs the device at each cut**, and
    /// the sweep [`GROUPS_A_ROW`] was chosen from.
    ///
    /// One threadgroup is one core of eighty, which is where a reduction like
    /// this one starts and is what [`crate::norm`] measured at a fiftieth of the
    /// width; the row this walks is 800 KB. The second dispatch is in every
    /// figure here, so what the table ranks is the whole argmax and not the part
    /// of it a cut makes cheaper.
    ///
    /// Read off the device's own clock over a command buffer of `CALLS`
    /// argmaxes, for [`crate::norm`]'s reason: a submission is 225 microseconds
    /// and most of these are under a hundred. Nothing asserts a duration; the
    /// numbers go to stderr for the commit message to quote.
    #[test]
    #[ignore = "a measurement: `just test-timing`, or `just test-full`"]
    fn what_an_argmax_over_the_vocabulary_costs() {
        let Some(device) = device() else { return };
        let argmax = kernels(&device);
        const CALLS: usize = 32;
        const ROUNDS: usize = 5;

        let cost = |rows: usize, groups: usize| -> Duration {
            let row = noisy(IDS, 1);
            let block: Vec<f32> = row.repeat(rows);
            let mut logits = device.buffer(&block).expect("the logits upload");
            crate::testing::device_time(&device, CALLS, |batch| {
                argmax
                    .encoding(batch, &mut logits, Vocabulary::of(IDS), groups)
                    .expect("the argmax encodes");
            })
        };

        let cuts = [1usize, 2, 4, 8, 16, 32, 64, 80, 128, 256, 512];
        let shapes = [1usize, 2, 4, 9];
        let mut taken = vec![vec![Vec::new(); cuts.len()]; shapes.len()];
        for round in 0..=ROUNDS {
            for (s, rows) in shapes.iter().enumerate() {
                for (c, groups) in cuts.iter().enumerate() {
                    let each = cost(*rows, *groups);
                    if round > 0 {
                        taken[s][c].push(each);
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
                .expect("a cut")
                .0;
            eprintln!(
                "[{rows}, {IDS}] over {cuts:?}: {}, best {} at {:.2?}, the rule chose {} at {:.2?}",
                means
                    .iter()
                    .map(|each| format!("{each:.2?}"))
                    .collect::<Vec<String>>()
                    .join(" "),
                cuts[best],
                means[best],
                groups_a_row(IDS),
                means[cuts
                    .iter()
                    .position(|cut| *cut == groups_a_row(IDS))
                    .expect("the rule's cut is in the sweep")],
            );
        }
    }
}
