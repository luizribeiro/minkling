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
}

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
        ];
        let mut shape = batch.device().inline(&fields)?;
        let mut grouped = Grouped {
            order: batch.device().zeroed::<u32>(rows)?,
            experts: batch.device().zeroed::<u32>(rows)?,
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
    ) -> Result<(Vec<u32>, Vec<u32>), MetalError> {
        let mut chosen = device.buffer(chosen)?;
        let mut batch = device.batch()?;
        let grouped = self.encode(&mut batch, &mut chosen, experts)?;
        batch.wait()?;
        Ok((grouped.order.to_vec(), grouped.experts.to_vec()))
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
        for (uint row = 0; row < shape.rows; ++row) {
            if (bucket_of(chosen[row], shape.experts) == bucket) {
                order[at] = row;
                grouped[at] = chosen[row];
                ++at;
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

        let (order, grouped) = grouping
            .group(&device, &chosen, ROUTING.n_routed)
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

        let (_, grouped) = grouping
            .group(&device, &chosen, ROUTING.n_routed)
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

        let (order, _) = grouping
            .group(&device, &chosen, 8)
            .expect("the dispatch completes");

        let mut want: Vec<u32> = (0..chosen.len() as u32).collect();
        want.sort_by_key(|row| chosen[*row as usize]);
        assert_eq!(order, want, "the rows of an expert are not in row order");
        assert_eq!(
            grouping
                .group(&device, &chosen, 8)
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

        let (order, grouped) = grouping
            .group(&device, &chosen, 4)
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
        let _ = grouping.group(&device, &[0u32, 1], MOST_EXPERTS + 1);
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
                .encode(batch, &mut on_the_device, ROUTING.n_routed)
                .expect("the grouping encodes");
        });

        assert_eq!(moved as usize, size_of::<u32>() * 3 * chosen.len());
    }
}
