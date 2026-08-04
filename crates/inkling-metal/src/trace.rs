//! What a step encoded, dispatch by dispatch, so that two steps can be
//! compared.
//!
//! **A decode step is believed to encode the same sequence every step**, and a
//! sequence encoded once and reused rests entirely on that being true. Believed
//! is not measured: the belief is about a thousand dispatches spread over
//! fourteen files, and the thing that would break it — one call whose grid or
//! whose bindings follow the context — would break it silently, as a step
//! replaying the wrong shape rather than as a failure.
//!
//! So a step can be asked to write down what it encoded. What comes back is one
//! [`Encoded`] a dispatch, holding the entry, the pipeline, the grid and what
//! filled each argument slot — everything an indirect command would have to
//! carry — and [`Difference`] is two of those held against each other.
//!
//! **Off by default and for a measurement rather than a run.** A decode loop
//! nobody is asking would otherwise grow a record per dispatch for as long as it
//! runs, which is the reason [`Device::record_round_trips`](crate::Device) gives
//! for the same decision.

use std::cell::RefCell;
use std::time::Duration;

/// What filled one of a dispatch's argument slots.
///
/// An allocation is recorded as its address rather than its contents: what an
/// indirect command binds is the buffer, so what matters between two steps is
/// whether the *same* buffer was named, not whether the bytes in it changed. An
/// inline argument is the opposite — it is copied into the command buffer as the
/// dispatch is encoded, so its bytes are the argument and are what has to be
/// compared.
///
/// **An address is not a durable identity and the count it feeds is a lower
/// bound.** This engine allocates most of a step's activations fresh and drops
/// them at the end of it, so an address freed on one step can be handed back on
/// the next — and two different allocations that landed at one address read here
/// as one that did not change. The error only ever goes one way: it can call a
/// slot unchanged that changed, never the reverse. So
/// [`Difference::patches`] is a floor on what a reused sequence would have to
/// write, which is the direction a decision against reusing one can be made
/// under. What it cannot understate is [`Difference::reusable`], which is about
/// entries, pipelines and grids and reads no address at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot {
    Bound(usize),
    Inline(Vec<u8>),
}

impl Slot {
    pub(crate) fn of(arg: &crate::buffer::Arg<'_>) -> Self {
        match arg {
            crate::buffer::Arg::Bound(buffer) => {
                Self::Bound(std::ptr::from_ref(*buffer) as *const () as usize)
            }
            crate::buffer::Arg::Inline(bytes) => Self::Inline(bytes.to_vec()),
        }
    }
}

/// One dispatch, as everything about it that an indirect command would have to
/// carry.
#[derive(Debug, Clone)]
pub struct Encoded {
    /// What the profile calls the entry that ran.
    pub entry: String,
    /// The compute pipeline's own address, which is what an indirect command
    /// names and what a kernel chosen by shape would change.
    pub pipeline: usize,
    pub slots: Vec<Slot>,
    pub threads: usize,
    pub threads_per_group: usize,
    /// What the Metal calls encoding this dispatch took, which is the part of a
    /// step's encode an indirect command would remove and the part above it is
    /// the part it would not.
    pub encoding: Duration,
}

impl Encoded {
    /// Whether two dispatches are the same command but for what fills its
    /// slots — same entry, same pipeline, same grid, same number of slots.
    ///
    /// **This is the predicate a reusable sequence turns on.** A slot that
    /// changed is a patch; anything here that changed is a command that has to
    /// be written whole, and a *count* that changed is a sequence that cannot be
    /// reused at all.
    pub fn same_command(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
            && self.entry == other.entry
            && self.threads == other.threads
            && self.threads_per_group == other.threads_per_group
            && self.slots.len() == other.slots.len()
    }
}

/// How one step's sequence differs from the step before it.
#[derive(Debug, Default, Clone)]
pub struct Difference {
    /// Dispatches in each, which have to agree before anything else means
    /// anything.
    pub dispatches: (usize, usize),
    /// Dispatches at the same index that are not the same command — a different
    /// entry, a different pipeline or a different grid.
    pub commands_changed: Vec<usize>,
    /// Slots holding a different allocation, as `(dispatch, slot)`.
    pub bound_changed: Vec<(usize, usize)>,
    /// Slots holding different inline bytes, as `(dispatch, slot)`.
    pub inline_changed: Vec<(usize, usize)>,
    /// Slots that changed which *kind* of argument they are, which is a command
    /// that has to be written rather than patched.
    pub kind_changed: Vec<(usize, usize)>,
    /// Slots that were named at all, which is what the changed counts are a
    /// fraction of.
    ///
    /// **Of the dispatches the two sequences have in common**, which for two
    /// sequences of different lengths is the shorter — so a reader that has not
    /// checked [`Difference::reusable`] is reading a prefix and this says so
    /// rather than the counts silently doing it.
    pub slots: usize,
}

impl Difference {
    /// The two sequences held against each other.
    pub fn between(before: &[Encoded], after: &[Encoded]) -> Self {
        let mut difference = Self {
            dispatches: (before.len(), after.len()),
            ..Self::default()
        };
        for (at, (before, after)) in before.iter().zip(after).enumerate() {
            difference.slots += after.slots.len();
            if !before.same_command(after) {
                difference.commands_changed.push(at);
                continue;
            }
            for (slot, (before, after)) in before.slots.iter().zip(&after.slots).enumerate() {
                match (before, after) {
                    (Slot::Bound(before), Slot::Bound(after)) if before != after => {
                        difference.bound_changed.push((at, slot));
                    }
                    (Slot::Inline(before), Slot::Inline(after)) if before != after => {
                        difference.inline_changed.push((at, slot));
                    }
                    (Slot::Bound(_), Slot::Inline(_)) | (Slot::Inline(_), Slot::Bound(_)) => {
                        difference.kind_changed.push((at, slot));
                    }
                    _ => {}
                }
            }
        }
        difference
    }

    /// Whether the two are the same sequence of the same commands, whatever
    /// fills their slots — which is the whole condition for encoding it once.
    pub fn reusable(&self) -> bool {
        self.dispatches.0 == self.dispatches.1
            && self.commands_changed.is_empty()
            && self.kind_changed.is_empty()
    }

    /// Slots that would have to be written between two runs of one sequence.
    pub fn patches(&self) -> usize {
        self.bound_changed.len() + self.inline_changed.len()
    }
}

thread_local! {
    /// The dispatches encoded since somebody asked, on the thread that encoded
    /// them. A thread local rather than a field of the device because
    /// [`Batch::add`](crate::kernel::Batch) is where a dispatch is described and
    /// a batch reaches its device only to allocate.
    static ENCODED: RefCell<Option<Vec<Encoded>>> = const { RefCell::new(None) };
}

/// Keep an [`Encoded`] for every dispatch from here, or stop and discard what
/// was kept.
pub fn record(recording: bool) {
    ENCODED.with(|encoded| *encoded.borrow_mut() = recording.then(Vec::new));
}

/// Every dispatch since [`record`] was switched on, in the order they were
/// encoded, and the record cleared — so that a caller measuring one step of a
/// loop is handed that step rather than the run so far.
pub fn take() -> Vec<Encoded> {
    ENCODED.with(|encoded| match encoded.borrow_mut().as_mut() {
        None => Vec::new(),
        Some(taken) => std::mem::take(taken),
    })
}

/// Whether anybody is recording, which is what keeps an unmeasured run from
/// building a description of a dispatch it will throw away.
pub(crate) fn recording() -> bool {
    ENCODED.with(|encoded| encoded.borrow().is_some())
}

pub(crate) fn encoded(describe: impl FnOnce() -> Encoded) {
    ENCODED.with(|encoded| {
        if let Some(taken) = encoded.borrow_mut().as_mut() {
            taken.push(describe());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch(pipeline: usize, slots: Vec<Slot>) -> Encoded {
        Encoded {
            entry: "saxpy".to_owned(),
            pipeline,
            slots,
            threads: 4096,
            threads_per_group: 64,
            encoding: Duration::ZERO,
        }
    }

    /// The case a reusable sequence is: the same commands, one slot of one of
    /// them naming a different allocation and one holding different bytes.
    #[test]
    fn two_steps_differing_only_in_what_fills_their_slots_are_one_sequence() {
        let before = [
            dispatch(1, vec![Slot::Bound(10), Slot::Inline(vec![0, 0])]),
            dispatch(2, vec![Slot::Bound(20)]),
        ];
        let after = [
            dispatch(1, vec![Slot::Bound(11), Slot::Inline(vec![0, 1])]),
            dispatch(2, vec![Slot::Bound(20)]),
        ];

        let difference = Difference::between(&before, &after);

        assert!(difference.reusable());
        assert_eq!(difference.bound_changed, [(0, 0)]);
        assert_eq!(difference.inline_changed, [(0, 1)]);
        assert_eq!((difference.patches(), difference.slots), (2, 3));
    }

    /// A dispatch that ran another pipeline, another grid or another number of
    /// slots is a command rather than a patch — and a step with a different
    /// number of them is not the same sequence at all.
    #[test]
    fn a_step_that_changed_a_pipeline_a_grid_or_a_count_is_not_the_same_sequence() {
        let one = [dispatch(1, vec![Slot::Bound(10)])];

        let other_pipeline = [dispatch(2, vec![Slot::Bound(10)])];
        assert_eq!(
            Difference::between(&one, &other_pipeline).commands_changed,
            [0]
        );

        let mut other_grid = one.clone();
        other_grid[0].threads = 8192;
        assert_eq!(Difference::between(&one, &other_grid).commands_changed, [0]);

        let other_slots = [dispatch(1, vec![Slot::Bound(10), Slot::Bound(11)])];
        assert_eq!(
            Difference::between(&one, &other_slots).commands_changed,
            [0]
        );

        let longer = [
            dispatch(1, vec![Slot::Bound(10)]),
            dispatch(1, vec![Slot::Bound(10)]),
        ];
        let difference = Difference::between(&one, &longer);
        assert!(!difference.reusable(), "{difference:?}");
        assert_eq!(difference.dispatches, (1, 2));
    }

    /// A slot that was inline bytes and became an allocation is neither a patch
    /// nor the same command: an indirect command has no inline binding at all,
    /// so which arm a slot is in is part of the command's shape.
    #[test]
    fn a_slot_that_changed_which_kind_of_argument_it_is_refuses_the_sequence() {
        let before = [dispatch(1, vec![Slot::Inline(vec![7])])];
        let after = [dispatch(1, vec![Slot::Bound(30)])];

        let difference = Difference::between(&before, &after);

        assert_eq!(difference.kind_changed, [(0, 0)]);
        assert!(!difference.reusable(), "{difference:?}");
    }

    /// **One buffer read twice is one slot and two buffers are two**, which is
    /// what every `bound` figure this module reports rests on and which the
    /// cases above assume rather than check: they build `Slot`s by hand.
    ///
    /// It cannot check the hazard [`Slot`] declares — an address handed back to
    /// a later allocation — because a test that arranged one would be asserting
    /// the allocator's behaviour rather than this crate's.
    #[test]
    fn a_slot_names_the_allocation_it_was_taken_from() {
        let Some(device) = crate::testing::device() else {
            return;
        };
        let mut one: crate::Buffer<f32> = device.zeroed(4).expect("the buffer allocates");
        let mut other: crate::Buffer<f32> = device.zeroed(4).expect("the buffer allocates");

        assert_eq!(Slot::of(&one.arg()), Slot::of(&one.arg()));
        assert_ne!(Slot::of(&one.arg()), Slot::of(&other.arg()));

        let mut inline = device.inline(&[7u32, 9]).expect("the values are inline");
        assert!(matches!(Slot::of(&inline.arg()), Slot::Inline(bytes) if bytes.len() == 8));
    }

    /// Nobody is recording unless somebody asked, and the reading clears what
    /// was kept.
    #[test]
    fn a_trace_is_kept_only_while_it_is_asked_for() {
        assert!(!recording());
        encoded(|| unreachable!("nobody is recording"));

        record(true);
        assert!(recording());
        encoded(|| dispatch(1, Vec::new()));
        assert_eq!(take().len(), 1);
        assert!(take().is_empty(), "the reading clears");

        record(false);
        encoded(|| unreachable!("recording stopped"));
        assert!(take().is_empty());
    }
}
