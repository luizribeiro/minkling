//! What a dispatch has to wait for, and what it does not.
//!
//! **Metal's default dispatch type is serial**, so every dispatch in a pass
//! waits for the one before it whether or not it reads what that one wrote. A
//! decode step's dispatches are mostly a chain — a norm feeds four projections, a
//! convolution feeds a norm, an activation feeds a `down` — but not entirely: the
//! four projections read the same rows and write four different buffers, and
//! nothing orders them but the encoder.
//!
//! `computeCommandEncoderWithDispatchType:` takes that ordering off and asks for
//! it back one `memoryBarrierWithScope:` at a time. What decides whether that is
//! worth doing is a count — how many barriers a sequence still needs — and
//! `what_a_barrier_costs_the_device_against_the_dispatches_it_separates` is what
//! turns that count into microseconds.
//!
//! **The count has to be derived rather than read off the code.** A barrier
//! removed by inspection is a race that is correct most of the time, and the
//! kernels this engine dispatches carry state between calls — a convolution's
//! window, the attention span, the router's selection — so the step a mistake
//! shows up on is not the step it was made on. So nothing here asks what a
//! dispatch is *for*. It asks which allocations filled its slots, which of those
//! slots its own Metal source declares writable, and nothing else.
//!
//! # What makes the answer sound
//!
//! Two dispatches may share a group when no allocation they both name is
//! written by either. That is the standard read-write hazard, over an identity —
//! the allocation's address — that holds for exactly as long as it needs to:
//! **a command buffer retains everything bound into it**, so no allocation named
//! in a run can be freed while the run is open, and an address that appears
//! twice in one sequence is one allocation both times.
//!
//! The error the identity admits goes one way. An address freed after one run
//! and handed back in the next reads as one allocation where there were two,
//! which is a hazard reported where there is none — a barrier kept, not a
//! barrier dropped. [`Slot`](crate::trace::Slot) says the same thing about the
//! same addresses for the same reason.
//!
//! **The identity is the binding and not the memory**, which is the one way it
//! could go the other way: two `MTLBuffer`s over one region would read as two
//! allocations, and a hazard between them would be missed. This engine makes
//! exactly one kind of second binding over memory it did not allocate —
//! [`Device::wrap`](crate::Device::wrap), which hands the GPU a checkpoint's own
//! pages where they lie — and those pages are a read-only mapping. Nothing
//! writes them, and a kernel that tried would take the process down with a bus
//! error rather than race, which is what [`Mapped`](crate::Mapped) says of
//! itself. Every allocation any dispatch here writes came from
//! [`Device::zeroed`](crate::Device::zeroed) and is its own region.
//!
//! **Whether a slot is written is the kernel's own statement**, parsed out of
//! the source string it was compiled from: `device const float *x` cannot be
//! written and `device float *out` may be. A slot the parse does not recognise
//! is taken as written, which is a barrier kept. See
//! [`Kernel::writes`](crate::Kernel::writes).

use std::time::Duration;

use crate::trace::{Encoded, Slot};

/// What Metal's serial dispatch type costs the device per dispatch, over and
/// above what the same dispatch costs in a pass that orders nothing.
///
/// **This is the whole of what a concurrent pass has to sell.** Measured by
/// `what_a_barrier_costs_the_device_against_the_dispatches_it_separates` at 1.554
/// microseconds a serial dispatch against 0.339 for the same dispatch with the
/// ordering off — a thousand empty dispatches, one grid, the driver's own clock.
const SERIAL_ORDERING: Duration = Duration::from_nanos(1215);

/// What one `memoryBarrierWithScope:` costs the device, from the same sweep:
/// 2.401 microseconds a dispatch with a barrier after every one of them, less
/// the 0.339 the dispatch costs alone.
///
/// **It is larger than the ordering it replaces**, which is the fact the whole
/// arithmetic turns on: a barrier kept is dearer than the serial dispatch type,
/// so only a barrier *removed* is worth anything, and a sequence pays for the
/// mechanism unless its groups average about 1.7 dispatches.
///
/// Confirmed on a real decode step rather than only on empty dispatches: every
/// pass made concurrent with a barrier after every dispatch — which is the same
/// ordering the serial type gives, and so the same tokens — reads 16.121 ms of
/// device against 16.825, seven alternating pairs, every pair the same way. That
/// is 0.804 microseconds a dispatch over the 876 a step at that context makes,
/// against the 0.847 these two figures predict.
const A_BARRIER: Duration = Duration::from_nanos(2062);

/// A sequence of dispatches divided into groups whose members can run at the
/// same time as each other.
///
/// The division is the coarsest one that keeps every hazard: a group is extended
/// until the next dispatch touches something a dispatch already in it wrote, and
/// a barrier goes between two groups. **Greedy is optimal here** — the groups
/// are contiguous, so a group that stopped early can only be a group that some
/// later group has to make up for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Groups {
    /// What each group holds, as the entries of its dispatches in the order
    /// they were encoded.
    ///
    /// The entries rather than a count, because a width on its own does not say
    /// *what* a step found to run at the same time — and a table that could not
    /// name the four projections could not be read against the layer they come
    /// from. See [`Groups::shapes`].
    groups: Vec<Vec<String>>,
    /// How many of those groups each command buffer holds, which is what says
    /// how many of the gaps between them are barriers somebody has to encode:
    /// the gap at a submission is one the queue already keeps.
    passes: Vec<usize>,
}

impl Groups {
    /// The sequence divided, with `boundaries` the dispatch indices at which a
    /// command buffer was committed.
    ///
    /// **A group never spans a boundary.** Two command buffers are two passes
    /// and a barrier cannot reach across one, so a boundary closes whatever
    /// group is open — which is also why it costs nothing: the queue already
    /// orders one buffer after another.
    pub fn over(dispatches: &[Encoded], boundaries: &[usize]) -> Self {
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut passes: Vec<usize> = Vec::new();
        let mut open: Vec<&Encoded> = Vec::new();
        let mut in_pass = 0usize;
        let close =
            |open: &mut Vec<&Encoded>, groups: &mut Vec<Vec<String>>, in_pass: &mut usize| {
                if !open.is_empty() {
                    groups.push(open.iter().map(|held| held.entry.clone()).collect());
                    *in_pass += 1;
                    open.clear();
                }
            };
        for (at, dispatch) in dispatches.iter().enumerate() {
            if at > 0 && boundaries.contains(&at) {
                close(&mut open, &mut groups, &mut in_pass);
                passes.push(std::mem::take(&mut in_pass));
            }
            if open.iter().any(|held| hazard(held, dispatch)) {
                close(&mut open, &mut groups, &mut in_pass);
            }
            open.push(dispatch);
        }
        close(&mut open, &mut groups, &mut in_pass);
        if in_pass > 0 {
            passes.push(in_pass);
        }
        Self { groups, passes }
    }

    pub fn dispatches(&self) -> usize {
        self.groups.iter().map(Vec::len).sum()
    }

    pub fn groups(&self) -> usize {
        self.groups.len()
    }

    /// How many barriers a concurrent pass would have to encode, which is one
    /// fewer than the groups in each command buffer rather than one fewer than
    /// the groups in the whole sequence.
    pub fn barriers(&self) -> usize {
        self.passes.iter().map(|groups| groups - 1).sum()
    }

    /// How many command buffers the sequence was submitted in, which is how
    /// many of the gaps between its groups cost nothing.
    pub fn passes(&self) -> usize {
        self.passes.len()
    }

    /// The widest group, which is what says whether the sequence has any
    /// concurrency in it at all.
    pub fn widest(&self) -> usize {
        self.groups.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Dispatches to a group, which is the figure the break-even in
    /// `what_a_barrier_costs_the_device_against_the_dispatches_it_separates` is
    /// stated in.
    pub fn average(&self) -> f64 {
        match self.groups.is_empty() {
            true => 0.0,
            false => self.dispatches() as f64 / self.groups.len() as f64,
        }
    }

    /// What encoding this sequence concurrently would be worth: the ordering it
    /// would stop paying for, less the barriers it would still have to encode.
    ///
    /// Negative where the barriers cost more than the ordering, which is what a
    /// sequence of mostly-chained dispatches comes to — see [`A_BARRIER`], which
    /// is dearer than [`SERIAL_ORDERING`] and so makes the sign a question about
    /// the count rather than about the mechanism.
    pub fn worth(&self) -> f64 {
        SERIAL_ORDERING.as_secs_f64() * self.dispatches() as f64
            - A_BARRIER.as_secs_f64() * self.barriers() as f64
    }

    /// Dispatches to a group at which encoding concurrently starts to pay,
    /// which is what [`Groups::average`] has to be above.
    ///
    /// One barrier separates two groups, so a sequence of `g` dispatches to a
    /// group pays `A_BARRIER / g` where it saves `SERIAL_ORDERING` — and the
    /// break-even is their ratio.
    pub fn break_even() -> f64 {
        A_BARRIER.as_secs_f64() / SERIAL_ORDERING.as_secs_f64()
    }

    /// Every distinct group the sequence divided into, widest first, with how
    /// many of each — which is what turns a width into a finding: 42 groups of
    /// `packed_matmul, packed_matmul, packed_matmul, packed_matmul` is one a
    /// reader can hold against the four projections a layer has, where "42 of
    /// width 4" is not.
    pub fn shapes(&self) -> Vec<(usize, &[String])> {
        let mut counted: Vec<(usize, &[String])> = Vec::new();
        for group in &self.groups {
            match counted
                .iter_mut()
                .find(|(_, held)| *held == group.as_slice())
            {
                Some((count, _)) => *count += 1,
                None => counted.push((1, group.as_slice())),
            }
        }
        counted.sort_unstable_by_key(|(count, held)| (std::cmp::Reverse(held.len()), *count));
        counted
    }
}

/// Whether two dispatches have to be ordered: an allocation both name, that at
/// least one of them may write.
///
/// **An inline argument is never a hazard.** `setBytes:` copies the value into
/// the command buffer as the dispatch is encoded, so there is no memory two
/// dispatches could disagree about — which is what makes a shape struct free to
/// share a group with anything.
fn hazard(before: &Encoded, after: &Encoded) -> bool {
    before.slots.iter().any(|held| match held {
        Slot::Inline(_) => false,
        Slot::Bound { at, written } => after.slots.iter().any(|slot| match slot {
            Slot::Inline(_) => false,
            Slot::Bound {
                at: other,
                written: writes,
            } => at == other && (*written || *writes),
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A dispatch naming `reads` and `writes`, which is everything the division
    /// looks at.
    fn dispatch(reads: &[usize], writes: &[usize]) -> Encoded {
        named("saxpy", reads, writes)
    }

    fn named(entry: &str, reads: &[usize], writes: &[usize]) -> Encoded {
        Encoded {
            entry: entry.to_owned(),
            pipeline: 1,
            slots: reads
                .iter()
                .map(|at| Slot::Bound {
                    at: *at,
                    written: false,
                })
                .chain(writes.iter().map(|at| Slot::Bound {
                    at: *at,
                    written: true,
                }))
                .collect(),
            threads: 1,
            threads_per_group: 1,
            encoding: Duration::ZERO,
        }
    }

    /// The shape a layer's four projections are: one buffer read by all of
    /// them, four outputs nobody else touches. Nothing orders them, so they are
    /// one group.
    #[test]
    fn dispatches_that_only_share_what_they_read_are_one_group() {
        let normed = 100;
        let sequence: Vec<Encoded> = (0..4).map(|out| dispatch(&[normed], &[out])).collect();

        let groups = Groups::over(&sequence, &[]);

        assert_eq!(groups.groups(), 1);
        assert_eq!((groups.dispatches(), groups.widest()), (4, 4));
        assert_eq!(groups.barriers(), 0);
        assert_eq!(groups.average(), 4.0);
    }

    /// The shape the rest of a layer is: each dispatch reads what the one
    /// before it wrote, so every group is one dispatch and every gap is a
    /// barrier.
    #[test]
    fn a_chain_is_a_group_a_dispatch() {
        let sequence: Vec<Encoded> = (0..5).map(|at| dispatch(&[at], &[at + 1])).collect();

        let groups = Groups::over(&sequence, &[]);

        assert_eq!((groups.groups(), groups.widest()), (5, 1));
        assert_eq!(groups.barriers(), 4);
        assert_eq!(groups.shapes(), [(5, ["saxpy".to_owned()].as_slice())]);
    }

    /// **A write after a read is a hazard as much as a read after a write.** A
    /// dispatch that overwrites what an earlier one is still reading would
    /// answer with whichever of the two the hardware ran first — which is the
    /// case a rule that looked only for reads-after-writes would let through.
    #[test]
    fn a_dispatch_that_overwrites_what_an_open_group_reads_closes_it() {
        let sequence = [dispatch(&[7], &[8]), dispatch(&[9], &[7])];

        let groups = Groups::over(&sequence, &[]);

        assert_eq!((groups.groups(), groups.barriers()), (2, 1));
    }

    /// Two writes to one allocation are a hazard even where neither reads it,
    /// which is the last of the three orderings a group has to keep.
    #[test]
    fn two_dispatches_writing_one_allocation_are_two_groups() {
        let sequence = [dispatch(&[], &[3]), dispatch(&[], &[3])];

        assert_eq!(Groups::over(&sequence, &[]).groups(), 2);
    }

    /// A group is closed by anything already in it and not only by the dispatch
    /// before it — the case a rule comparing neighbours would get wrong, and the
    /// one that decides whether a group may be extended past a dispatch it
    /// happens not to touch.
    #[test]
    fn a_group_is_closed_by_anything_in_it_rather_than_by_its_last() {
        let sequence = [
            dispatch(&[], &[1]),
            dispatch(&[], &[2]),
            dispatch(&[1], &[3]),
        ];

        let groups = Groups::over(&sequence, &[]);

        assert_eq!(groups.widest(), 2);
        assert_eq!(groups.barriers(), 1);
    }

    /// An inline argument is a copy in the command buffer, so two dispatches
    /// carrying the same bytes are not two dispatches sharing memory.
    #[test]
    fn inline_arguments_order_nothing() {
        let inline = |bytes: u8| Encoded {
            slots: vec![Slot::Inline(vec![bytes])],
            ..dispatch(&[], &[])
        };
        let sequence = [inline(1), inline(1)];

        assert_eq!(Groups::over(&sequence, &[]).groups(), 1);
    }

    /// **A group cannot span a command buffer**, so a boundary closes one —
    /// and the barrier it stands in for is not one anybody encodes, which is
    /// what keeps `barriers` below `groups - 1` where a sequence was submitted
    /// part way through.
    #[test]
    fn a_submission_closes_a_group_without_costing_a_barrier() {
        let sequence: Vec<Encoded> = (0..4).map(|out| dispatch(&[100], &[out])).collect();

        let whole = Groups::over(&sequence, &[]);
        let split = Groups::over(&sequence, &[2]);

        assert_eq!((whole.groups(), whole.barriers()), (1, 0));
        assert_eq!(
            (split.groups(), split.barriers()),
            (2, 0),
            "two passes of two, and no barrier inside either"
        );
    }

    /// **The arithmetic, at the break-even the two measured constants imply.**
    /// A sequence whose groups sit exactly on it is worth nothing either way,
    /// and the two either side of it are worth what their counts say — which is
    /// the whole of how a dispatch count and a barrier count become a verdict.
    #[test]
    fn what_a_sequence_is_worth_follows_its_groups_against_the_break_even() {
        // Two dispatches to a group is above the break-even of about 1.7, and a
        // group a dispatch is below it. A pair writes two buffers and the pair
        // behind it reads the first of them, which is what closes each group
        // after exactly two.
        let paired: Vec<Encoded> = (0..8)
            .map(|at| match at {
                0 | 1 => dispatch(&[], &[10 + at]),
                _ => dispatch(&[10 + 2 * (at / 2 - 1)], &[10 + at]),
            })
            .collect();
        let chained: Vec<Encoded> = (0..8).map(|at| dispatch(&[at], &[at + 1])).collect();

        let paired = Groups::over(&paired, &[]);
        let chained = Groups::over(&chained, &[]);

        assert_eq!((paired.groups(), paired.barriers()), (4, 3));
        assert!(paired.average() > Groups::break_even(), "{paired:?}");
        assert!(paired.worth() > 0.0, "{}", paired.worth());

        assert_eq!((chained.groups(), chained.barriers()), (8, 7));
        assert!(chained.average() < Groups::break_even(), "{chained:?}");
        assert!(
            chained.worth() < 0.0,
            "a chain pays for the mechanism: {}",
            chained.worth()
        );
    }

    /// A sequence already in one group is worth its whole ordering, and it is
    /// the ceiling every other answer sits under.
    #[test]
    fn a_sequence_with_no_barriers_left_is_worth_its_whole_ordering() {
        let sequence: Vec<Encoded> = (0..10).map(|out| dispatch(&[100], &[out])).collect();

        let groups = Groups::over(&sequence, &[]);

        assert_eq!(groups.barriers(), 0);
        assert!((groups.worth() - 10.0 * SERIAL_ORDERING.as_secs_f64()).abs() < 1e-12);
    }

    #[test]
    fn a_sequence_of_nothing_is_no_groups_and_no_barriers() {
        let groups = Groups::over(&[], &[]);

        assert_eq!((groups.groups(), groups.dispatches()), (0, 0));
        assert_eq!((groups.barriers(), groups.widest()), (0, 0));
        assert_eq!(groups.average(), 0.0);
    }
}
