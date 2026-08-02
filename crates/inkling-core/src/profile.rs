//! Where a step's time went, by operation.
//!
//! Every prediction this project has made about its own cost has been wrong in
//! an instructive way — `lm_head` was 7.6% of a step and not the 54% the
//! parameter count implied, the bandwidth model died against a kernel that runs
//! at 4% of it, and M5 found that percentage shares are not fixed costs because
//! removing a large term shrinks the denominator under everything else. So the
//! next thing to move is measured rather than reasoned about, and this is what
//! measures it.
//!
//! # Self time, so that the rows add up
//!
//! A scope is charged the time inside it that no scope inside *it* claimed.
//! Without that a coarse scope and a fine one nested in it would both count the
//! same microseconds and the table would sum past the step it describes; with
//! it, every microsecond lands in exactly one row and what the rows leave over
//! is a number a caller can subtract and name.
//!
//! The consequence worth stating is that a row is not "what this operation
//! costs" but "what this operation costs *here*". [`Op::Router`] holds the
//! top-k and the softmax and not the gate's own multiply, which is an
//! [`Op::Linear`] inside it; [`Op::Submit`] holds the wait and not the buffers
//! a dispatch filled first, which are an [`Op::Encode`] beside it.
//!
//! # What it costs to know
//!
//! A scope is two clock reads and two borrows of a thread-local, which
//! `a_scope_costs_less_than_the_ops_it_times` below measures at 37 nanoseconds
//! on this machine. How many a step opens is the model's shape, and the profile
//! itself is what reports that — so what this costs is a number the same table
//! carries rather than an estimate anyone has to take on trust.
//!
//! Left on rather than put behind a feature, because a profile that has to be
//! switched on is one that is never on when the surprising run happens.
//!
//! # Two clocks, and only one of them is a scope
//!
//! Everything above is this process's clock around work it asked for. What the
//! *device* spent is a second account, reported by a backend rather than
//! measured here: [`ran_on_the_gpu`] for a whole command buffer, and
//! [`dispatched`] for one kernel's share of one. Neither adds to [`total`] —
//! both are subdivisions of [`Op::Submit`], which is the wall time around them
//! — and the per-kernel rows are what say which of the dispatches inside a
//! submission owns it. Which kernel owns which milliseconds is the one thing a
//! `submit and wait` row cannot answer, and it is what decides which kernel is
//! worth rewriting.
//!
//! # One thread's
//!
//! The accounts are thread-local, which is what a sequence is: everything below
//! [`Generator::stream`](crate::Generator::stream) runs on the thread that
//! called it. A server decoding two sequences on two threads has two sets of
//! accounts, and summing them is a question that can be answered when there is
//! something to sum.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, Instant};

/// One piece of a forward pass, as the thing time is charged to.
///
/// The list is the operations a decode step is made of rather than the modules
/// it passes through, because what it exists to order is which one to move next.
/// Discriminants index the accounts, which is what [`Op::ALL`] is checked
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Op {
    /// A row of the embedding table decoded for a token that asked for it.
    Embedding,
    /// A weight read: a bfloat16 tensor widened, or an MXFP4 one decoded to
    /// float32. A layer's bfloat16 tensors are widened once, when the weights
    /// are opened, so what is charged here on a step is the packed weights the
    /// CPU path decodes to multiply against.
    Decode,
    /// Every RMSNorm: a layer's two, attention's query and key norms, and the
    /// final one before the head.
    RmsNorm,
    /// The four depthwise causal short convolutions.
    Sconv,
    /// The banded relative-position bias, materialised.
    Mask,
    /// The attention step itself: scores, softmax and the weighted sum.
    Sdpa,
    /// The router's top-k over 256 sigmoid-corrected scores and the softmax
    /// over what it picked. Its `[258, 4096]` gate is an [`Op::Linear`] inside
    /// this.
    Router,
    /// The rows a bank's call reads, gathered out of the hidden state, and what
    /// it answered scattered back into place.
    Gather,
    /// `x @ wᵀ` over a weight already in memory, which on the CPU path is every
    /// projection in the model and on the device path is the router's gate
    /// alone.
    Linear,
    /// `silu(gate) * up`, and the scale a dense layer multiplies its output by.
    Swiglu,
    /// The two residual adds.
    Residual,
    /// The mup divide and the argmax over the vocabulary.
    Sample,
    /// A dispatch's buffers: the input copied over and the output allocated,
    /// and the bindings encoded. A shape and the scalars beside it are in the
    /// command buffer rather than in an allocation, so what is charged here for
    /// them is the copy and not a `newBufferWithLength:`.
    Encode,
    /// A command buffer committed, and a command buffer waited for. This is the
    /// round trip [`Profile::gpu`] is the inner part of.
    ///
    /// **Two calls a buffer where the two happen apart**, which is a backend
    /// that commits one and keeps encoding into the next: the commit and the
    /// wait are then two intervals with the caller's own work between them, and
    /// a single scope around both would charge that work here.
    Submit,
    /// What a dispatch produced, copied off the device.
    Readback,
}

impl Op {
    /// Every op, in discriminant order — which is the order the accounts are
    /// indexed in, and what `the_ops_index_their_own_accounts` is about: a
    /// variant added here out of order would charge one op's time to another's
    /// row.
    pub const ALL: [Op; 15] = [
        Op::Embedding,
        Op::Decode,
        Op::RmsNorm,
        Op::Sconv,
        Op::Mask,
        Op::Sdpa,
        Op::Router,
        Op::Gather,
        Op::Linear,
        Op::Swiglu,
        Op::Residual,
        Op::Sample,
        Op::Encode,
        Op::Submit,
        Op::Readback,
    ];

    /// What a table calls this row.
    pub fn name(self) -> &'static str {
        match self {
            Op::Embedding => "embedding",
            Op::Decode => "weights decoded",
            Op::RmsNorm => "rms_norm",
            Op::Sconv => "sconv",
            Op::Mask => "mask",
            Op::Sdpa => "sdpa",
            Op::Router => "router",
            Op::Gather => "moe gather",
            Op::Linear => "linear",
            Op::Swiglu => "swiglu",
            Op::Residual => "residual add",
            Op::Sample => "sample",
            Op::Encode => "dispatch encode",
            Op::Submit => "submit and wait",
            Op::Readback => "readback",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

const OPS: usize = Op::ALL.len();

/// What one kernel's dispatches cost on the device, from the GPU's own clock.
///
/// A row of the table [`Profile::kernels`] returns, and empty unless a backend
/// is sampling — the timestamps behind it are hardware counters somebody has to
/// have asked for.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Dispatches {
    /// How many dispatches of this kernel the device timed.
    pub calls: u64,
    /// What it was executing them for, summed.
    pub elapsed: Duration,
    /// The bytes they said they move between memory and the GPU, summed.
    ///
    /// Declared by each dispatch rather than derived from what it bound: a
    /// bank's dispatch binds all 256 experts and reads six of them, and a
    /// layer's attention binds a span with room for a thousand keys and reads
    /// the eight there are. What a buffer *is* is not what a dispatch *moves*,
    /// and only the caller knows the difference.
    pub bytes: u64,
}

impl Dispatches {
    /// What this kernel moved per second of the device's own time — the figure
    /// a memory bandwidth is compared against.
    ///
    /// Zero for a kernel the device reported no time for, which is a kernel
    /// nobody sampled rather than an infinitely fast one.
    pub fn bytes_per_second(&self) -> f64 {
        match self.elapsed.as_secs_f64() {
            0.0 => 0.0,
            elapsed => self.bytes as f64 / elapsed,
        }
    }
}

/// What the calling thread has spent, and where.
#[derive(Debug, Default, Clone)]
struct Accounts {
    /// Time charged to each op that no op inside it claimed.
    elapsed: [Duration; OPS],
    calls: [u64; OPS],
    /// What the scopes opened inside the innermost open one have taken, which
    /// is what its own charge has subtracted from it when it closes.
    children: Duration,
    /// The same figure for each scope further out, to be restored as they close.
    outer: Vec<Duration>,
    /// What the device reported it was executing for, which is inside
    /// [`Op::Submit`] rather than beside it.
    gpu: Duration,
    /// The same figure split by the kernel that ran, which is inside [`gpu`]
    /// rather than beside it.
    ///
    /// [`gpu`]: Accounts::gpu
    kernels: BTreeMap<String, Dispatches>,
}

thread_local! {
    static ACCOUNTS: RefCell<Accounts> = RefCell::new(Accounts::default());
}

/// An open scope, which charges [`Op`] when it is dropped.
///
/// It has to be held in a *named* binding. `#[must_use]` catches the bare
/// `scope(op);` and nothing else — `let _ = scope(op)` is the compiler's own
/// suggestion for silencing that lint, and it drops the scope on the same line
/// and charges an empty interval. Nothing in the language can catch that one,
/// so it is written down here instead.
#[must_use = "a scope charges its op when it is dropped, so it has to be held in a named binding"]
#[derive(Debug)]
pub struct Scope {
    op: Op,
    started: Instant,
}

/// Charge everything until this is dropped to `op`.
pub fn scope(op: Op) -> Scope {
    ACCOUNTS.with_borrow_mut(|accounts| {
        accounts.outer.push(accounts.children);
        accounts.children = Duration::ZERO;
    });
    Scope {
        op,
        started: Instant::now(),
    }
}

/// [`scope`] around a call, for the call in the middle of an expression that a
/// binding would have to be lifted out of.
pub fn timed<T>(op: Op, run: impl FnOnce() -> T) -> T {
    let _scope = scope(op);
    run()
}

impl Drop for Scope {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        let index = self.op.index();
        ACCOUNTS.with_borrow_mut(|accounts| {
            // Saturating because a nested scope cannot have run longer than the
            // one around it and a clock that said otherwise is not a reason to
            // panic in a destructor.
            accounts.elapsed[index] += elapsed.saturating_sub(accounts.children);
            accounts.calls[index] += 1;
            accounts.children = accounts.outer.pop().unwrap_or_default() + elapsed;
        });
    }
}

/// What the device reported one command buffer executed for, which only a
/// backend can know: it is the GPU's own clock rather than this one's, and the
/// wall time around it is [`Op::Submit`].
pub fn ran_on_the_gpu(elapsed: Duration) {
    ACCOUNTS.with_borrow_mut(|accounts| accounts.gpu += elapsed);
}

/// What the device reported one kernel's `calls` dispatches executed for, which
/// is a share of the [`ran_on_the_gpu`] figure for the command buffer they were
/// in rather than a figure beside it.
///
/// Charged per kernel per command buffer rather than per dispatch, so that a
/// submission holding a thousand of them costs a handful of lookups: a backend
/// that has timestamps for each one sums them by kernel first, which is the
/// grain this is read at anyway.
pub fn dispatched(kernel: &str, calls: u64, elapsed: Duration, bytes: u64) {
    ACCOUNTS.with_borrow_mut(|accounts| {
        let account = match accounts.kernels.get_mut(kernel) {
            Some(account) => account,
            None => accounts.kernels.entry(kernel.to_owned()).or_default(),
        };
        account.calls += calls;
        account.elapsed += elapsed;
        account.bytes += bytes;
    });
}

/// Everything charged since the last [`take`], and the accounts cleared.
///
/// Cleared rather than read, so that a caller measuring one step of a loop
/// takes the step rather than the run so far.
///
/// Between steps and not inside one. A scope still open when this is called has
/// charged nothing yet, and would charge the step it started in to the step it
/// ends in — a step short by a term and the next one long by the same, which is
/// worth a panic rather than a plausible table.
pub fn take() -> Profile {
    ACCOUNTS.with_borrow_mut(|accounts| {
        assert!(
            accounts.outer.is_empty(),
            "{} scopes are still open",
            accounts.outer.len()
        );
        let taken = Profile {
            elapsed: accounts.elapsed,
            calls: accounts.calls,
            gpu: accounts.gpu,
            kernels: std::mem::take(&mut accounts.kernels),
        };
        accounts.elapsed = [Duration::ZERO; OPS];
        accounts.calls = [0; OPS];
        accounts.gpu = Duration::ZERO;
        taken
    })
}

/// What a run cost, by operation, from [`take`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Profile {
    elapsed: [Duration; OPS],
    calls: [u64; OPS],
    gpu: Duration,
    kernels: BTreeMap<String, Dispatches>,
}

impl Profile {
    /// How long was charged to `op`, not counting what the ops inside it took.
    pub fn elapsed(&self, op: Op) -> Duration {
        self.elapsed[op.index()]
    }

    /// How many scopes of `op` closed.
    pub fn calls(&self, op: Op) -> u64 {
        self.calls[op.index()]
    }

    /// Everything charged, which is what a caller subtracts from a measured
    /// wall time to see what nothing here accounts for.
    pub fn total(&self) -> Duration {
        self.elapsed.iter().sum()
    }

    /// What the device said it was executing for. Inside [`Op::Submit`] and not
    /// beside it: the rest of that row is the round trip around the work.
    pub fn gpu(&self) -> Duration {
        self.gpu
    }

    /// Every op with what it took and how often, heaviest first, leaving out
    /// the ops this run never reached.
    pub fn rows(&self) -> Vec<(Op, u64, Duration)> {
        let mut rows: Vec<(Op, u64, Duration)> = Op::ALL
            .iter()
            .map(|op| (*op, self.calls(*op), self.elapsed(*op)))
            .filter(|(_, calls, _)| *calls > 0)
            .collect();
        rows.sort_by_key(|(_, _, elapsed)| std::cmp::Reverse(*elapsed));
        rows
    }

    /// Each kernel the device timed dispatches of, heaviest first.
    ///
    /// Empty unless a backend was sampling: these come from
    /// [`dispatched`] rather than from a scope, and what a backend has to do to
    /// know them is not free.
    pub fn kernels(&self) -> Vec<(&str, Dispatches)> {
        let mut rows: Vec<(&str, Dispatches)> = self
            .kernels
            .iter()
            .map(|(kernel, account)| (kernel.as_str(), *account))
            .collect();
        rows.sort_by_key(|(_, account)| std::cmp::Reverse(account.elapsed));
        rows
    }

    /// What the device was executing for over every kernel [`Profile::kernels`]
    /// holds, which is the part of [`Profile::gpu`] the sampling accounted for.
    ///
    /// The two are not the same number and the gap is the finding rather than
    /// an error: a command buffer's own clock runs from before its first
    /// dispatch to after its last, and what the passes inside it do not claim is
    /// the device's own gaps between them.
    pub fn dispatched(&self) -> Duration {
        self.kernels.values().map(|account| account.elapsed).sum()
    }

    /// Each figure divided by the `steps` that produced it, for a profile taken
    /// over a run rather than over one step.
    ///
    /// The call counts divide as integers, which is exact for every op here —
    /// each runs a number of times the model's shape decides, the same on every
    /// step — and would round a fractional average down if one ever did not.
    pub fn per_step(&self, steps: u32) -> Self {
        assert!(steps > 0, "a profile over no steps");
        Self {
            elapsed: self.elapsed.map(|elapsed| elapsed / steps),
            calls: self.calls.map(|calls| calls / u64::from(steps)),
            gpu: self.gpu / steps,
            kernels: self
                .kernels
                .iter()
                .map(|(kernel, account)| {
                    (
                        kernel.clone(),
                        Dispatches {
                            calls: account.calls / u64::from(steps),
                            elapsed: account.elapsed / steps,
                            bytes: account.bytes / u64::from(steps),
                        },
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Long enough that a clock of any resolution tells the two scopes apart,
    /// and short enough that the suite does not notice.
    const SPIN: Duration = Duration::from_millis(4);

    /// Busy rather than asleep, because what a scope charges is wall time and a
    /// sleep is the one way to spend it that a profiler would rather not be
    /// pinned to the scheduler for.
    fn spin(how_long: Duration) {
        let started = Instant::now();
        while started.elapsed() < how_long {
            std::hint::spin_loop();
        }
    }

    /// Each thread has its own accounts, so a test that takes them is only
    /// racing itself — but two tests on one thread are not, and `take` clears.
    fn fresh() {
        take();
    }

    /// The discriminants index the accounts, so a variant added to the enum
    /// without being added to [`Op::ALL`] — or added in the wrong place — would
    /// charge one op's time to another's row.
    #[test]
    fn the_ops_index_their_own_accounts() {
        for (index, op) in Op::ALL.iter().enumerate() {
            assert_eq!(op.index(), index, "{op}");
        }
    }

    #[test]
    fn a_scope_charges_what_ran_inside_it() {
        fresh();
        {
            let _scope = scope(Op::Sdpa);
            spin(SPIN);
        }

        let profile = take();
        assert_eq!(profile.calls(Op::Sdpa), 1);
        assert!(profile.elapsed(Op::Sdpa) >= SPIN, "{profile:?}");
        assert_eq!(profile.elapsed(Op::Mask), Duration::ZERO);
    }

    /// The whole of what makes the rows add up: an op inside another is charged
    /// once, to itself, and the one around it keeps only what it spent on its
    /// own. Two scopes that both counted the same milliseconds would leave a
    /// table summing to twice the step it describes.
    #[test]
    fn a_nested_scope_is_charged_to_itself_and_taken_off_the_one_around_it() {
        fresh();
        {
            let _outer = scope(Op::Router);
            spin(SPIN);
            {
                let _inner = scope(Op::Linear);
                spin(2 * SPIN);
            }
        }

        let profile = take();
        let (outer, inner) = (profile.elapsed(Op::Router), profile.elapsed(Op::Linear));
        assert!(inner >= 2 * SPIN, "{inner:?}");
        assert!(outer >= SPIN, "{outer:?}");
        assert!(
            outer < 2 * SPIN,
            "the inner scope's time was charged twice: {outer:?}"
        );
        assert!(profile.total() >= 3 * SPIN, "{profile:?}");
    }

    /// Siblings do not take each other's time off, which is the mistake the
    /// accounting above could make in the other direction: what the *previous*
    /// scope took is not what this one contains.
    #[test]
    fn two_scopes_beside_each_other_are_each_charged_in_full() {
        fresh();
        {
            let _outer = scope(Op::Sdpa);
            {
                let _first = scope(Op::Mask);
                spin(SPIN);
            }
            {
                let _second = scope(Op::Sconv);
                spin(SPIN);
            }
        }

        let profile = take();
        assert!(profile.elapsed(Op::Mask) >= SPIN, "{profile:?}");
        assert!(profile.elapsed(Op::Sconv) >= SPIN, "{profile:?}");
        assert!(
            profile.elapsed(Op::Sdpa) < SPIN,
            "the siblings' time reached the scope around them: {profile:?}"
        );
    }

    /// Taken between steps and not inside one. A scope left open would charge
    /// the step it started in to the step it ends in, and both tables would
    /// look like tables.
    #[test]
    #[should_panic(expected = "1 scopes are still open")]
    fn taking_the_accounts_with_a_scope_still_open_is_refused() {
        fresh();
        let _open = scope(Op::Sdpa);
        take();
    }

    /// What a scope costs, which is what says the profile is worth leaving on.
    ///
    /// The comparison that matters is against the cheapest thing it times: a
    /// residual add over a 4096-wide row is a couple of microseconds, and a
    /// scope has to be decades below that or the table would be describing
    /// itself. Nothing asserts a ratio — the number goes to stderr for the
    /// commit message to quote — and what is asserted is that it stays under a
    /// tenth of the cheapest row.
    #[test]
    fn a_scope_costs_less_than_the_ops_it_times() {
        fresh();
        const SCOPES: u32 = 200_000;

        for _ in 0..SCOPES / 10 {
            timed(Op::Residual, || ());
        }
        take();

        let started = Instant::now();
        for _ in 0..SCOPES {
            timed(Op::Residual, || ());
        }
        let each = started.elapsed() / SCOPES;

        eprintln!("a scope opened and closed: {each:?}");
        assert_eq!(take().calls(Op::Residual), u64::from(SCOPES));
        assert!(each < Duration::from_nanos(200), "{each:?}");
    }

    #[test]
    fn taking_the_accounts_clears_them() {
        fresh();
        timed(Op::Residual, || spin(SPIN));

        assert!(take().calls(Op::Residual) > 0);
        assert_eq!(take(), Profile::default());
    }

    /// The device's own clock, which no scope here can read: it is reported
    /// rather than measured, and it sits inside [`Op::Submit`] rather than
    /// adding to the total.
    #[test]
    fn the_gpus_own_time_is_reported_inside_the_submission_rather_than_beside_it() {
        fresh();
        timed(Op::Submit, || {
            spin(SPIN);
            ran_on_the_gpu(SPIN / 2);
        });

        let profile = take();
        assert_eq!(profile.gpu(), SPIN / 2);
        assert!(profile.gpu() < profile.elapsed(Op::Submit));
        assert_eq!(
            profile.total(),
            profile.elapsed(Op::Submit),
            "the device's time was added to the wall time around it"
        );
    }

    /// The per-kernel rows are a subdivision of the device's own clock and not
    /// a second charge against it, the same way that clock is a subdivision of
    /// the submission around it. Three figures nested, and the total is still
    /// the wall time.
    #[test]
    fn a_kernels_device_time_is_reported_inside_the_gpus_rather_than_beside_it() {
        fresh();
        timed(Op::Submit, || {
            spin(SPIN);
            ran_on_the_gpu(SPIN / 2);
            dispatched("packed_matmul", 3, SPIN / 4, 3_000);
            dispatched("rms_norm", 1, SPIN / 8, 1_000);
        });

        let profile = take();
        assert_eq!(profile.dispatched(), SPIN / 4 + SPIN / 8);
        assert!(profile.dispatched() < profile.gpu());
        assert_eq!(
            profile.total(),
            profile.elapsed(Op::Submit),
            "a kernel's device time was added to the wall time around it"
        );
    }

    /// One kernel is dispatched from several command buffers a step, so what a
    /// row has to be is the sum over all of them rather than the last one seen.
    #[test]
    fn a_kernel_dispatched_from_several_command_buffers_is_one_row() {
        fresh();
        dispatched("packed_matmul", 26, SPIN, 26);
        dispatched("rms_norm", 2, 3 * SPIN, 2);
        dispatched("packed_matmul", 4, SPIN, 4);

        let profile = take();
        assert_eq!(
            profile.kernels(),
            [
                (
                    "rms_norm",
                    Dispatches {
                        calls: 2,
                        elapsed: 3 * SPIN,
                        bytes: 2
                    }
                ),
                (
                    "packed_matmul",
                    Dispatches {
                        calls: 30,
                        elapsed: 2 * SPIN,
                        bytes: 30
                    }
                ),
            ],
            "heaviest first, and one row a kernel"
        );
    }

    /// What a row is read for: the bytes a kernel moved against the seconds it
    /// took to move them, which is the figure this machine's 819 GB/s is a
    /// ceiling on. A kernel nobody timed is nought and not a division by one.
    #[test]
    fn a_kernels_bandwidth_is_what_it_moved_over_what_it_took() {
        let moved = Dispatches {
            calls: 2,
            elapsed: Duration::from_millis(4),
            bytes: 8_000_000,
        };
        assert_eq!(moved.bytes_per_second(), 2e9);

        let untimed = Dispatches {
            bytes: 8_000_000,
            ..Dispatches::default()
        };
        assert_eq!(untimed.bytes_per_second(), 0.0);
    }

    /// A profile with nothing sampled has no kernel rows at all rather than a
    /// row of zeroes for every kernel that ran — which is what makes an empty
    /// table mean "nobody was sampling" and not "the device did nothing".
    #[test]
    fn a_profile_taken_without_sampling_has_no_kernel_rows() {
        fresh();
        timed(Op::Submit, || ran_on_the_gpu(SPIN));

        let profile = take();
        assert!(profile.kernels().is_empty());
        assert_eq!(profile.dispatched(), Duration::ZERO);
        assert_eq!(profile.gpu(), SPIN);
    }

    /// A table leaves out what a run never reached, and puts the heaviest first
    /// — which is the whole use it is put to.
    #[test]
    fn the_rows_are_the_ops_that_ran_heaviest_first() {
        fresh();
        timed(Op::Sdpa, || spin(2 * SPIN));
        timed(Op::Mask, || spin(SPIN));

        let rows = take().rows();
        assert_eq!(
            rows.iter().map(|(op, ..)| *op).collect::<Vec<Op>>(),
            [Op::Sdpa, Op::Mask]
        );
        assert!(rows.iter().all(|(_, calls, _)| *calls == 1));
    }

    #[test]
    fn a_profile_divides_by_the_steps_that_produced_it() {
        fresh();
        for _ in 0..2 {
            timed(Op::Sdpa, || spin(SPIN));
        }
        ran_on_the_gpu(SPIN);

        let each = take().per_step(2);
        assert_eq!(each.calls(Op::Sdpa), 1);
        assert!(each.elapsed(Op::Sdpa) >= SPIN / 2);
        assert!(each.elapsed(Op::Sdpa) < 2 * SPIN);
        assert_eq!(each.gpu(), SPIN / 2);
    }

    #[test]
    fn a_profile_divides_its_kernel_rows_by_the_steps_too() {
        fresh();
        for _ in 0..2 {
            dispatched("packed_matmul", 26, SPIN, 1_024);
        }

        let each = take().per_step(2);
        assert_eq!(
            each.kernels(),
            [(
                "packed_matmul",
                Dispatches {
                    calls: 26,
                    elapsed: SPIN,
                    bytes: 1_024
                }
            )]
        );
    }

    /// A panic unwinding through a scope still closes it, which is what keeps
    /// the stack of open scopes from being left one deep — and the next step's
    /// accounts from having the panicking one's time subtracted from them.
    #[test]
    fn a_scope_unwound_through_is_still_closed() {
        fresh();
        let panicked = std::panic::catch_unwind(|| {
            let _outer = scope(Op::Sdpa);
            let _inner = scope(Op::Mask);
            panic!("the layer did not run");
        });
        assert!(panicked.is_err());

        timed(Op::Residual, || spin(SPIN));
        let profile = take();
        assert!(
            profile.elapsed(Op::Residual) >= SPIN,
            "the unwound scopes were charged the next one's time: {profile:?}"
        );
    }
}
