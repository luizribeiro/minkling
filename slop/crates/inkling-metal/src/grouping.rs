//! The order a call's rows would be in if they were laid out expert by expert.
//!
//! **This is the one thing a prefill's routed bank was missing.** A tile of
//! [`crate::PackedMatmul`]'s second entry reads a weight row once for every row
//! of the tile that named the same expert, and a routed bank's rows name six
//! different experts a token by construction — so consecutive rows never share a
//! read however long the prompt, and the 59% of a prefill's bytes those banks
//! are went untiled. Sorting the rows by expert is what puts rows that could
//! share a read beside each other.
//!
//! **It is a permutation and nothing else.** What comes out is where each row
//! went, not which expert any row goes through: `experts[i]` is
//! `chosen[order[i]]` by construction, so a token still reads exactly the six
//! experts its router named. That is the property the whole change rests on and
//! `a_grouping_moves_rows_and_never_the_expert_a_row_named` is where it is held.
//!
//! **A counting sort in one threadgroup**, because the buckets are the layer's
//! own expert count and the rows are the tokens six times over — 256 and 4614 on
//! a 769-token prefill. The rows are counted in parallel with a threadgroup
//! atomic per expert, which is order-independent and so gives the same counts
//! whatever order the threads arrive in; the placement is then a thread to an
//! expert, walking the rows in order and writing out the ones that named it. So
//! the answer is the same on every run — a stable sort, ties by row — where an
//! atomic placement would have been a different permutation each time. Nothing
//! downstream would notice, since a row's own answer does not depend on which
//! tile it lands in, but a permutation that changed under a repeated measurement
//! would make the tiles a call ends up with unrepeatable.

use inkling_core::profile::{self, Op};

use crate::buffer::Buffer;
use crate::device::{Device, MetalError};
use crate::kernel::{Batch, Grid, Kernel, extent};

const ENTRY: &str = "group_by_expert";

/// Threads the one threadgroup of a dispatch holds, which is a thread to an
/// expert while the layer has no more experts than this.
///
/// One threadgroup and not a grid of them: a counting sort's placement needs
/// every bucket's offset, which is a reduction over all of the counts, and there
/// is no barrier across threadgroups inside a dispatch. Two dispatches would buy
/// the parallelism at the price of a second launch and a buffer between them,
/// and one core is enough for what this is: 40 of these are 25.9 ms of a
/// 769-token prefill's 5.11 s of device time, 0.5% of it and the smallest row of
/// the table `where_a_prefill_spends_its_time` prints that is not a rounding.
/// Its cost is the rows rather than the experts — a thread walks the selection
/// once to place its own bucket — so it grows with the prompt and not with the
/// bank.
const THREADS_PER_GROUP: usize = 256;

/// Experts a grouping can sort into.
///
/// The bound is the threadgroup array of counts, which is `MOST_EXPERTS + 1`
/// entries of four bytes — the last of them for an index no bank holds, which is
/// what keeps the answer a permutation whatever it is handed. Inkling routes
/// over 256; a model with more would need the array widened rather than a
/// fallback, which is why it is refused where a grouping is asked for.
const MOST_EXPERTS: usize = 512;

/// The `uint`s one block of the plan is written as: where its rows start, and
/// how many of them it holds.
pub(crate) const PLAN_FIELDS: usize = 2;

/// Blocks a plan of this shape can hold, which is what the grid a dispatch
/// through it is covered by has to be.
///
/// **A bound rather than a count, because the counts are on the device.** A run
/// of `L` rows is `ceil(L / rows_a_block)` blocks, and the worst the runs can
/// do is leave every one of them a part-block short — so summing the ceilings
/// is at most the rows over the height plus a block a bucket. The plan is zeroed
/// and the blocks past the last run hold nothing, which is what the dispatch
/// reads them as.
///
/// The bucket count is the bank's experts and one more, for the reason
/// `bucket_of` gives: an index past the bank goes in a bucket of its own so that
/// the layout stays a rearrangement of the rows.
pub(crate) fn blocks_a_plan(rows: usize, experts: usize, rows_a_block: usize) -> usize {
    match rows_a_block {
        0 => 0,
        height => rows.div_ceil(height) + experts + 1,
    }
}

/// The compiled kernel, which every MoE layer on a device shares.
///
/// Per source string rather than per layer, like [`crate::PackedMatmul`] and
/// [`crate::MoeCombine`]: the source names no shape, so one of these serves all
/// forty layers that route.
#[derive(Debug)]
pub struct ExpertGrouping {
    kernel: Kernel,
}

/// One call's rows laid out expert by expert, as the dispatch that sorted them
/// left them.
///
/// Both halves together, because neither says anything alone: `order` is where
/// the rows went and `experts` is what they named, and a caller that took one
/// without the other would be dispatching a bank over a list that did not
/// describe its rows.
#[derive(Debug)]
pub struct Grouped {
    /// Row `i` of the grouped call is row `order[i]` of the call the router
    /// named — so this is what a dispatch reads its input through and what it
    /// scatters its output through.
    pub(crate) order: Buffer<u32>,
    /// The expert row `i` of the grouped call goes through, which is
    /// `chosen[order[i]]` and nothing else.
    pub(crate) experts: Buffer<u32>,
    /// Where each block of a blocked dispatch starts and how many of the call's
    /// rows it holds, a pair of `uint` apiece.
    ///
    /// **This is the sort's answer to a question only the sort can answer
    /// cheaply.** A block whose rows name several experts runs the whole
    /// reduction once per expert they name, and where the boundaries fall is
    /// decided by counts that live on the device and are never read back. The
    /// counting sort has every one of them in a threadgroup array by the time it
    /// places a row, so cutting the blocks at the boundaries costs it a few
    /// hundred writes and costs the block that reads it one load.
    ///
    /// A pair of zeros is a block with nothing in it, which is what the entries
    /// past the last run hold: the count is bounded above rather than known, so
    /// the grid covers the bound and the threadgroups over it return.
    pub(crate) plan: Buffer<u32>,
    /// The height the plan was cut against, which a dispatch through it has to
    /// be the block of.
    ///
    /// **Carried so that a mismatch is a panic rather than an answer.** A plan
    /// cut at 32 rows and read by a block of 16 would leave every second block's
    /// rows unreached — a wrong answer and not a slow one — and the height is a
    /// value the block carries, so a sweep can hold one of each in a process.
    pub(crate) rows_a_block: usize,
}

/// What a sort leaves behind, read back: where each row went, what each of them
/// named, and the blocks the plan cut the runs into.
///
/// The three together because none of them says anything alone — see
/// [`Grouped`], which is the same three where a dispatch reads them.
pub type Sorted = (Vec<u32>, Vec<u32>, Vec<u32>);

impl ExpertGrouping {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`ExpertGrouping::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }

    /// The rows a selection named, sorted by the expert each of them named,
    /// encoded into `batch` over a buffer a dispatch already left there.
    ///
    /// **Nothing here waits for this side**, which is the whole reason it can be
    /// done at all: the selection is what the top-k wrote two dispatches back and
    /// the order is what the bank behind this reads, so the router, the sort and
    /// the bank stay in the one command buffer a MoE layer is.
    pub(crate) fn encode(
        &self,
        batch: &mut Batch<'_>,
        chosen: &mut Buffer<u32>,
        experts: usize,
        rows_a_block: usize,
    ) -> Result<Grouped, MetalError> {
        let _timed = profile::scope(Op::Encode);
        assert!(
            (1..=MOST_EXPERTS).contains(&experts),
            "{experts} experts are not between one and the {MOST_EXPERTS} a threadgroup counts"
        );
        let rows = chosen.len();
        assert!(rows > 0, "a grouping sorts some rows");

        let fields = [
            extent(rows, "the rows of a call"),
            extent(experts, "the experts a bank holds"),
            extent(
                rows_a_block,
                "the rows a block of the dispatch behind this holds",
            ),
        ];
        let mut shape = batch.device().inline(&fields)?;
        let mut grouped = Grouped {
            order: batch.device().zeroed::<u32>(rows)?,
            experts: batch.device().zeroed::<u32>(rows)?,
            // A call with no block behind it plans nothing and is still handed a
            // buffer, because a binding is what a kernel takes and an
            // allocation of no bytes is what Metal refuses.
            plan: batch
                .device()
                .zeroed::<u32>(PLAN_FIELDS * blocks_a_plan(rows, experts, rows_a_block).max(1))?,
            rows_a_block,
        };

        // **The selection is read twice and charged once**, which is the same
        // reading the matmul takes of an input every output column re-reads: the
        // count pass and the placement pass walk the same `rows` indices, and
        // the second walk is over what the first left in cache.
        let moves = size_of::<u32>() * 3 * rows;
        batch.add(
            &self.kernel,
            &[
                shape.arg(),
                chosen.arg(),
                grouped.order.arg(),
                grouped.experts.arg(),
                grouped.plan.arg(),
            ],
            Grid::new(THREADS_PER_GROUP, THREADS_PER_GROUP),
            moves,
        )?;
        Ok(grouped)
    }

    /// The same sort submitted on its own, over a selection this side holds —
    /// which is the cases here and nothing in the engine.
    pub fn group(
        &self,
        device: &Device,
        chosen: &[u32],
        experts: usize,
        rows_a_block: usize,
    ) -> Result<Sorted, MetalError> {
        let mut chosen = device.buffer(chosen)?;
        let mut batch = device.batch()?;
        let grouped = self.encode(&mut batch, &mut chosen, experts, rows_a_block)?;
        batch.wait()?;
        Ok((
            grouped.order.to_vec(),
            grouped.experts.to_vec(),
            grouped.plan.to_vec(),
        ))
    }
}

/// The kernel, with the bound its threadgroup array of counts is sized by
/// written into its prelude rather than spelled twice.
pub(crate) fn source() -> String {
    format!("constant uint MOST_EXPERTS = {MOST_EXPERTS};\n{BODY}")
}

/// Everything of the kernel that that bound does not decide.
pub(crate) const BODY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint rows;
    uint experts;
    /// Rows one block of the dispatch behind this covers, which is what the
    /// runs are cut into blocks against — or zero where nothing behind this
    /// blocks its rows and the plan is not written.
    uint rows_a_block;
};

/// The bucket a row falls in, which is its expert unless the expert is one the
/// bank does not hold.
///
/// **An index past the bank goes in a bucket of its own rather than nowhere.**
/// Nothing on this side has seen these indices — they are what a top-k wrote
/// where no readback looks — and a row dropped for being out of range would
/// leave a hole in the order and a row written twice, which is a permutation
/// that is not one: two of the call's rows would land on the same output row and
/// one would never be written at all. The bank still reads whatever such an
/// index addresses, exactly as it does today; what this guarantees is only that
/// the layout stays a rearrangement of the rows.
inline uint bucket_of(uint expert, uint experts) {
    return min(expert, experts);
}

/// `order` is the rows `0..rows` sorted by `chosen`, stably; `grouped[i]` is
/// `chosen[order[i]]`.
///
/// One threadgroup, three passes and two barriers. The counts are taken in
/// parallel over the rows, because a sum does not care what order it is added
/// in. The offsets are a prefix sum a thread walks for its own bucket, which at
/// a few hundred buckets is cheaper than a scan and needs no second array. The
/// placement is a thread to a bucket walking every row in order, which is what
/// makes the sort stable — an atomic claim on a slot would be a different
/// permutation on every run.
kernel void group_by_expert(
    constant Shape &shape [[buffer(0)]],
    device const uint *chosen [[buffer(1)]],
    device uint *order [[buffer(2)]],
    device uint *grouped [[buffer(3)]],
    device uint *plan [[buffer(4)]],
    uint local [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint counts[MOST_EXPERTS + 1];

    const uint buckets = min(shape.experts, MOST_EXPERTS) + 1;
    for (uint bucket = local; bucket < buckets; bucket += threads) {
        atomic_store_explicit(&counts[bucket], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = local; row < shape.rows; row += threads) {
        const uint bucket = bucket_of(chosen[row], shape.experts);
        atomic_fetch_add_explicit(&counts[bucket], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bucket = local; bucket < buckets; bucket += threads) {
        uint at = 0;
        for (uint before = 0; before < bucket; ++before) {
            at += atomic_load_explicit(&counts[before], memory_order_relaxed);
        }
        const uint start = at;
        for (uint row = 0; row < shape.rows; ++row) {
            if (bucket_of(chosen[row], shape.experts) == bucket) {
                order[at] = row;
                grouped[at] = chosen[row];
                ++at;
            }
        }

        // **This bucket's run cut into blocks**, which is the whole of what the
        // plan is: a block of the dispatch behind this gets one bucket's rows
        // and however few of them are left over, so no block of it ever holds
        // rows naming two experts and no block runs the reduction twice.
        //
        // The last block of a run holds what the run had left, which is what
        // lets the rows stay where the sort put them — a layout padded to whole
        // blocks would move every row after the first short run and every reader
        // of `order` would have to know it.
        // **A second walk of the same counts rather than a second accumulator in
        // the one above**, which is the shape a cost decides. The prefix sum
        // this needs is over the *blocks* each bucket takes rather than over its
        // rows, and carrying both through one loop puts a division and a branch
        // in a loop every dispatch runs — where a grouped call whose runs are
        // long reads no plan at all and would pay for one anyway. That is 34 ms
        // of a 16384-token prefill, which is 0.1% of it and was measured.
        //
        // Skipped entirely rather than reached with a stride of zero, which
        // would be a loop that never ends.
        if (shape.rows_a_block > 0) {
            uint planned = 0;
            for (uint before = 0; before < bucket; ++before) {
                planned += (atomic_load_explicit(&counts[before], memory_order_relaxed)
                            + shape.rows_a_block - 1) / shape.rows_a_block;
            }
            for (uint held = start; held < at; held += shape.rows_a_block) {
                plan[2 * planned] = held;
                plan[2 * planned + 1] = min(shape.rows_a_block, at - held);
                ++planned;
            }
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ROUTING, device};

    /// Tokens enough that every expert of the shape below is named several times
    /// over, and not a multiple of anything here.
    const TOKENS: usize = 131;

    /// The height these cases cut a plan against.
    ///
    /// **A number of this module's own rather than the matmul's**, because what
    /// is being checked here is that the plan describes the runs — which is true
    /// of any height, and is most easily got wrong at one that divides nothing.
    const ROWS_A_BLOCK: usize = 5;

    /// A selection of the shape a router writes: `top_k` experts a token, no two
    /// of a token's the same, and spread over the whole bank rather than over its
    /// first rows.
    ///
    /// Not the real router's, because what is being sorted here is a list of
    /// indices and the only thing about it that matters is which indices they
    /// are — `experts::tests` is where a grouping meets a selection a dispatch
    /// made.
    fn selection(tokens: usize, experts: usize, top_k: usize, seed: usize) -> Vec<u32> {
        (0..tokens)
            .flat_map(|token| {
                let first = (token * 37 + seed) % experts;
                (0..top_k).map(move |slot| ((first + slot * 29) % experts) as u32)
            })
            .collect()
    }

    /// The whole claim of this module, and the one the change it exists for
    /// rests on: **the rows move and what they named does not.**
    ///
    /// Three properties, and none of them is implied by the others. The order is
    /// a permutation of the rows — every row once, no row twice — which is what
    /// makes reading through it and writing back through it lossless. The expert
    /// list that comes out is the one that went in, read through that
    /// permutation, so no row changed which weight it goes through. And a token
    /// still reads the set its router named, which is the two together said in
    /// the terms the model is about.
    #[test]
    fn a_grouping_moves_rows_and_never_the_expert_a_row_named() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let chosen = selection(TOKENS, ROUTING.n_routed, ROUTING.top_k, 0);

        let (order, grouped, _) = grouping
            .group(&device, &chosen, ROUTING.n_routed, ROWS_A_BLOCK)
            .expect("the dispatch completes");

        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..chosen.len() as u32).collect::<Vec<u32>>(),
            "the order is not a permutation of the call's rows"
        );
        for (row, at) in order.iter().enumerate() {
            assert_eq!(
                grouped[row], chosen[*at as usize],
                "row {row} came from row {at}"
            );
        }

        let of_token = |list: &[u32], token: usize| {
            let mut row = list[token * ROUTING.top_k..][..ROUTING.top_k].to_vec();
            row.sort_unstable();
            row
        };
        for token in 0..TOKENS {
            let mut moved: Vec<u32> = order
                .iter()
                .zip(&grouped)
                .filter(|(at, _)| **at as usize / ROUTING.top_k == token)
                .map(|(_, expert)| *expert)
                .collect();
            moved.sort_unstable();
            assert_eq!(moved, of_token(&chosen, token), "token {token}");
        }
    }

    /// **The rows come out grouped**, which is the whole point of moving them:
    /// every row naming one expert is one run, so the tile behind this has a run
    /// to share a weight read across.
    ///
    /// The run lengths go to stderr beside it, because what a tile of four is
    /// worth is decided by them and not by the row count — see
    /// [`crate::matmul`]'s own `ROWS_A_TILE`.
    #[test]
    fn a_groupings_rows_are_one_run_per_expert() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let chosen = selection(TOKENS, ROUTING.n_routed, ROUTING.top_k, 3);

        let (_, grouped, _) = grouping
            .group(&device, &chosen, ROUTING.n_routed, ROWS_A_BLOCK)
            .expect("the dispatch completes");

        assert!(
            grouped.windows(2).all(|pair| pair[0] <= pair[1]),
            "the grouped experts are not in order"
        );
        let runs: Vec<usize> = grouped.chunk_by(|a, b| a == b).map(<[u32]>::len).collect();
        let named = chosen.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(runs.len(), named.len(), "one run per expert the rows named");
        eprintln!(
            "{} rows over {} experts: runs of {} to {}",
            chosen.len(),
            runs.len(),
            runs.iter().min().expect("a run"),
            runs.iter().max().expect("a run"),
        );
    }

    /// **The plan covers every row of the call exactly once and never covers two
    /// experts' rows in one block**, which is the whole of what a dispatch
    /// through it rests on.
    ///
    /// Three properties, and none implies the others. Every row is in some block,
    /// which is what says no output row goes unwritten. No row is in two, which
    /// is what says none is written twice — a block laid over another's rows
    /// would answer them under one expert and then the other, and the second
    /// write would win. And no block spans a run boundary, which is what the
    /// plan is *for*: a block whose rows disagree still answers them, so this is
    /// the rate the plan buys rather than the answer it keeps.
    ///
    /// **A height that divides nothing here**, so a plan that happened to be
    /// right at 32 rows over runs that were multiples of it would not be right
    /// at this one.
    #[test]
    fn a_plan_covers_every_row_once_and_never_two_experts_in_one_block() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        // A bank narrow enough that the runs are dozens of rows and wide enough
        // that there are boundaries for a block to land on, with rows naming an
        // index it does not hold — because those go in a bucket of their own and
        // the plan has to cover that run like any other.
        let mut chosen = selection(TOKENS, 11, ROUTING.top_k, 7);
        for row in chosen.iter_mut().step_by(17) {
            *row = 11;
        }

        let (_, grouped, plan) = grouping
            .group(&device, &chosen, 11, ROWS_A_BLOCK)
            .expect("the dispatch completes");
        assert!(
            grouped.contains(&11),
            "no row named an index past the bank, so the bucket that holds them is empty"
        );

        let mut covered = vec![0usize; chosen.len()];
        let mut blocks = 0;
        for held in plan.chunks_exact(PLAN_FIELDS) {
            let (first, rows) = (held[0] as usize, held[1] as usize);
            if rows == 0 {
                continue;
            }
            blocks += 1;
            assert!(
                rows <= ROWS_A_BLOCK,
                "a block of {rows} rows is over the {ROWS_A_BLOCK} it was cut at"
            );
            assert!(
                grouped[first..first + rows]
                    .iter()
                    .all(|e| *e == grouped[first]),
                "a block at {first} of {rows} rows spans a run boundary"
            );
            covered[first..first + rows]
                .iter_mut()
                .for_each(|times| *times += 1);
        }
        assert!(
            covered.iter().all(|times| *times == 1),
            "the plan covers {} rows once, {} never and {} twice over",
            covered.iter().filter(|times| **times == 1).count(),
            covered.iter().filter(|times| **times == 0).count(),
            covered.iter().filter(|times| **times > 1).count(),
        );
        let runs = grouped.chunk_by(|a, b| a == b).count();
        eprintln!(
            "{} rows over {runs} runs: {blocks} blocks of at most {ROWS_A_BLOCK}",
            chosen.len()
        );
        assert!(
            blocks >= runs && blocks <= chosen.len().div_ceil(ROWS_A_BLOCK) + runs,
            "{blocks} blocks over {runs} runs is neither a block a run nor the rows cut into them"
        );
    }

    /// A stable sort, which is what says the same call gives the same layout
    /// twice — so a measurement repeated over one prompt is repeated over one
    /// set of tiles.
    ///
    /// The rows of a bucket come out in the order they went in, which an atomic
    /// claim on a slot would not promise: it would still be a permutation, and a
    /// different one each run.
    #[test]
    fn a_grouping_keeps_the_rows_of_one_expert_in_the_order_they_came() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let chosen = selection(TOKENS, 8, ROUTING.top_k, 11);

        let (order, ..) = grouping
            .group(&device, &chosen, 8, ROWS_A_BLOCK)
            .expect("the dispatch completes");

        let mut want: Vec<u32> = (0..chosen.len() as u32).collect();
        want.sort_by_key(|row| chosen[*row as usize]);
        assert_eq!(order, want, "the rows of an expert are not in row order");
        assert_eq!(
            grouping
                .group(&device, &chosen, 8, ROWS_A_BLOCK)
                .expect("the dispatch completes")
                .0,
            order,
            "two runs of one selection disagreed"
        );
    }

    /// An index no bank holds is a row like any other here, and it has to be:
    /// the selection is in device memory and this side has not seen it, so a
    /// grouping that dropped such a row would leave a hole in the order and a
    /// row written twice — the one way a permutation stops being one.
    ///
    /// What the bank then reads through that index is what it reads today. This
    /// is about the layout and nothing else.
    #[test]
    fn a_row_naming_an_expert_the_bank_does_not_hold_still_lands_somewhere() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let chosen = [3u32, 9, 0, 9, 1, 3];

        let (order, grouped, _) = grouping
            .group(&device, &chosen, 4, ROWS_A_BLOCK)
            .expect("the dispatch completes");

        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..chosen.len() as u32).collect::<Vec<u32>>());
        assert_eq!(
            grouped,
            vec![0, 1, 3, 3, 9, 9],
            "the rows past the bank last"
        );
        for (row, at) in order.iter().enumerate() {
            assert_eq!(grouped[row], chosen[*at as usize]);
        }
    }

    /// A bank with more experts than a threadgroup counts is refused where the
    /// grouping is asked for, rather than sorting into a bucket array it would
    /// read past the end of.
    #[test]
    #[should_panic(expected = "experts are not between one and the")]
    fn a_bank_of_more_experts_than_a_threadgroup_counts_is_refused() {
        let Some(device) = device() else {
            panic!("experts are not between one and the: no device to ask")
        };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let _ = grouping.group(&device, &[0u32, 1], MOST_EXPERTS + 1, ROWS_A_BLOCK);
    }

    /// What the bandwidth column divides by, against what the kernel reads: the
    /// selection once, and the two lists it writes.
    #[test]
    fn a_dispatch_declares_the_selection_it_reads_and_the_order_it_writes() {
        let Some(device) = device() else { return };
        let grouping = ExpertGrouping::new(&device).expect("the grouping compiles");
        let chosen = selection(TOKENS, ROUTING.n_routed, ROUTING.top_k, 0);
        let mut on_the_device = device.buffer(&chosen).expect("the selection uploads");

        let moved = crate::testing::moved(&device, |batch| {
            grouping
                .encode(batch, &mut on_the_device, ROUTING.n_routed, ROWS_A_BLOCK)
                .expect("the grouping encodes");
        });

        assert_eq!(moved as usize, size_of::<u32>() * 3 * chosen.len());
    }
}
