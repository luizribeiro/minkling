//! A slot that fills when a request arrives.
//!
//! [`Generator::generate_batch`](crate::generate::Generator::generate_batch) is
//! the *static* arrangement: N prompts prefilled a sequence at a time, then all
//! of them decoded together until the last one is done, and a slot a sequence
//! vacates stays empty for the rest of the run. That is the shape every batching
//! figure in this repo was taken at, and what it is worth is settled — 2.21× at
//! width 32, and a ceiling of 1.08× above it.
//!
//! **What is not settled, and is what this is for, is latency.** A request that
//! arrives while a batch is running waits for the batch to drain before it is
//! prefilled at all. For a fleet of agents asking at irregular times that wait
//! is the whole of what a user feels, and no figure here had ever measured it.
//!
//! # What a slot is doing, and why a prompt is fed short of its last token
//!
//! A seated request is in one of two states, and the difference is which rows
//! the call it rides in asks a question about:
//!
//! - **Filling.** Its prompt is going into its cache, a share of the step's
//!   prompt budget at a time, in the same forward pass the sequences already in
//!   flight take their next token out of. None of those rows is a question.
//! - **Decoding.** It feeds one row a step — the token it owes — and the tail is
//!   asked what follows it.
//!
//! A prompt is filled to `prompt[..n-1]` and no further. The token that follows
//! a prompt is decided by the prompt's *last* row, and that row is a question —
//! so the last token is fed as the sequence's first decode row, where every
//! other sequence's row already is. That is the split
//! `prefilling_then_decoding_matches_one_prefill_over_the_whole_sequence` states
//! as an invariant: prefilling `k` and decoding the rest is one prefill over all
//! of it, exactly and not nearly. **So the chunk size changes no token**, which
//! is asserted here rather than argued.
//!
//! # What the chunk buys and what it costs
//!
//! A joining prompt fed whole is one call of `n + d` rows, and the `d`
//! sequences already decoding wait the whole of it — a 385-token prompt riding
//! with seven decoders is a **2.75 s** step against their own **73.6 ms**, so
//! one arrival costs every sequence in flight thirty-seven steps' worth of
//! jitter on a single token. Given a budget of `b` prompt rows a step, the same
//! prompt is spread over `ceil((n-1)/b)` steps and the worst any one of them
//! costs them is bounded by that budget: 725 ms at 128 rows, 199 at 16.
//!
//! **The budget is the call's and not the seat's** — see [`Continuous::new`],
//! where the measurement that decided it is. A per-seat number bounds nothing:
//! eight slots filling 128 rows apiece is a call of 1024 rows.
//!
//! **What the budget does not buy is the total.** Summed over the decoders the
//! delay is 1.96 to 2.99 s whatever it is, because it is the prefill's own work
//! and somebody has to do it. The budget is a bound on the jitter of one token
//! and not a reduction in what the prompt costs.
//!
//! **And a narrow budget is not free either.** It pays a call's overhead once
//! per chunk rather than once per prompt, and that is most of what the sweep
//! measures: a 385-token prompt in 24 chunks of 16 costs 4.54 s of device where
//! the same prompt whole costs 1.78.
//!
//! **What it is not is the query block.** `FusedAttention::blocked` does refuse
//! a call carrying more than one sequence — a block stages one tile of keys for
//! the 64 query rows sharing it, and rows attending over different spans cannot
//! share one — so a chunk riding beside decoders can never be given one. It
//! turns out to cost nothing measurable at these shapes, and it is worth saying
//! why twice over: the entry is behind `--numerics production` and the figures
//! here are taken under the default word, which has no block to lose; and the
//! device time a second sequence in the call adds is **15 to 22 ms a step**, of
//! which 8.65 is the decode row's own marginal cost — so what is left is 6 to
//! 13 ms of a second seat's overhead, and it does not scale with the rows the
//! way a lost block would. These are differences of a few percent between two
//! large numbers and they bound rather than resolve; the README has the
//! arithmetic.
//!
//! # Starvation is a correctness property
//!
//! A scheduler that never admits a waiting request is wrong however fast its
//! steps are. The queue here is first in, first out, and every free slot is
//! filled from the front of it before a step is built — so a request at position
//! `p` waits for `p` slots to free and for nothing else. Both halves of that are
//! asserted: no request waits while a slot is free, and a stream of requests
//! through one slot comes back in the order it was submitted.
//!
//! # A request nobody is waiting for any more
//!
//! Every request here finishes by producing the tokens it asked for. A client
//! does not: it hangs up, and the seat it left decodes a budget nobody will read
//! while the request behind it waits for a slot. That is the same starvation the
//! queue is careful about, arriving through the one door the queue does not
//! watch — so [`Continuous::release`] is the other half of it, and a fleet is
//! precisely the workload that produces abandoned requests.
//!
//! # A conversation that comes back to its own keys
//!
//! Everything above is about a slot as a *seat*: whoever is in it now, and how
//! quickly the next one gets it. What a slot also is, and what nothing here used
//! to read, is a **cache with a position** — the keys and windows the last
//! sequence in it left behind.
//!
//! Handing those on unexamined is the one thing this engine has never done. A
//! joining sequence attending over its predecessor's keys answers a row of the
//! right shape out of a plausible softmax and goes on generating fluent text,
//! and `a_request_that_takes_a_finished_ones_slot_produces_what_it_produces_alone`
//! exists to say it does not. So the freshness is not being given up. What is
//! added is the one case where the keys in a slot **are** the arriving
//! sequence's own:
//!
//! - Every slot carries a [`Kept`] — the ids its cache is sitting at the end of,
//!   and nothing where it holds none.
//! - A prompt is admitted into a free slot whose kept ids it **starts with**,
//!   and it prefills only the rest. That is a comparison against the tokens
//!   themselves rather than a name a client asserted: **a slot's contents are
//!   proved to belong to the arriving conversation, not claimed to.**
//! - A prompt matching nothing is admitted into a slot holding nothing, or, if
//!   every free slot holds someone else's conversation, into the one that last
//!   held a sequence longest ago — whose conversation is forgotten, which is to
//!   say whose cache is replaced by one holding no keys. **An evicted
//!   conversation coming back is a miss like any other**, and a miss is exactly
//!   the fresh cache this engine always gave.
//!
//! So there is one new sentence to be wrong about and it is the matching. Below
//! it, nothing changed: a slot that did not match hands over a cache at no keys,
//! and a slot that did hands over one whose every key the arriving prompt named.
//!
//! **The position a conversation is kept at is the end of its prompt**, which is
//! not where its generation left the cache — see [`Kept::turn`], which resumes
//! the same mark for the same reason. So a seated sequence takes a mark on the
//! step its prompt goes in, and the slot is put back to it when the sequence
//! leaves. A sequence that never got that far leaves no position anything could
//! be resumed to, and its slot is forgotten rather than kept.
//!
//! [`Continuous::new`] keeps nothing, and [`Continuous::keeping`] is the arm that
//! does. Every batching figure in this repo was taken on the first of those and
//! is a figure about seats.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::config::TextConfig;
use crate::generate::Generator;
use crate::keep::Kept;
use crate::model::{Batched, Mark, ModelWeights};

/// What a caller asks the engine for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The prompt, which may not be empty: a forward pass over no tokens is
    /// refused below this line and the refusal would land a step later, on a
    /// batch that has nothing to do with it.
    pub prompt: Vec<usize>,
    /// How many tokens to decode.
    ///
    /// **A request that wants none is dropped from the queue and never
    /// answered**, which is worth stating rather than leaving to be discovered:
    /// its ticket produces no [`Answered`] and a caller waiting on one waits
    /// forever. What it must not do is take a slot, because a slot held by a
    /// request that is already finished is a slot standing empty in front of the
    /// one behind it.
    pub count: usize,
}

/// One request, put in a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admitted {
    pub ticket: usize,
    /// The slot it was given, which for a returning conversation is the slot its
    /// own keys are in — see [`Continuous::seat_for`].
    pub slot: usize,
    /// Tokens of its prompt that slot already held, and zero where it held none
    /// of them: a fresh conversation, an evicted one coming back, and every
    /// request of an engine keeping nothing.
    pub reused: usize,
}

/// One request, answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answered {
    pub ticket: usize,
    pub produced: Vec<usize>,
}

/// What one sequence in one slot carries between steps.
///
/// **The cache is not here, it is the slot's** — see [`Slot`]. A sequence is a
/// tenant of a cache that outlives it, which is the whole of what lets the next
/// tenant be the same conversation.
#[derive(Debug)]
struct Resident {
    ticket: usize,
    prompt: Vec<usize>,
    count: usize,
    /// Tokens of `prompt` in the slot's cache, which starts at whatever the slot
    /// already held of this conversation and never reaches the prompt's last
    /// token: see the module documentation.
    filled: usize,
    produced: Vec<usize>,
    /// Where the cache stood when this sequence's prompt was in and before its
    /// first row that is a question — the position the slot is put back to when
    /// it leaves, and `None` until its prompt is in.
    mark: Option<Mark>,
}

impl Resident {
    /// Whether this sequence's next rows are prompt rather than a token it owes.
    fn filling(&self) -> bool {
        self.filled + 1 < self.prompt.len()
    }

    /// What the slot holds once this has left it, which is its prompt short of
    /// the last token — the position [`Resident::mark`] was taken at, and the
    /// same one [`Kept::turn`] records on the serial path.
    fn kept(&self) -> &[usize] {
        &self.prompt[..self.prompt.len() - 1]
    }

    /// The prompt rows it feeds next, given the `share` of the step's budget it
    /// was allowed — which is never empty while it is filling, because a seat
    /// allowed no rows would sit in a slot making no progress and never leave
    /// it.
    fn chunk(&self, share: usize) -> &[usize] {
        &self.prompt[self.filled..(self.filled + share.max(1)).min(self.prompt.len() - 1)]
    }

    /// The one row it feeds when it is decoding: the last token of its prompt
    /// until it has produced something, and what it produced last after that.
    fn owed(&self) -> usize {
        *self
            .produced
            .last()
            .unwrap_or_else(|| self.prompt.last().expect("a request carries a prompt"))
    }

    fn done(&self) -> bool {
        self.produced.len() >= self.count
    }
}

/// What one step of the engine did.
///
/// **Rows rather than sequences on both counts**, because that is what the call
/// cost: a filling seat feeds a chunk and a decoding one feeds a token, and a
/// step's price is the sum.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stepped {
    /// Prompt rows the step carried for sequences filling.
    pub filling: usize,
    /// Rows it carried for sequences decoding, which is one apiece and so is
    /// also how many sequences took a token out of it.
    pub decoding: usize,
    /// Sequences the step carried at all, filling and decoding together — which
    /// is what says the engine kept its slots full and not only that it ran.
    pub seats: usize,
    /// Tickets whose **first** token this step produced, which is what a
    /// time-to-first-token is measured to.
    pub first: Vec<usize>,
    /// The requests this step put in a slot.
    ///
    /// **What a caller cannot work out for itself.** Which slot a request goes
    /// into is the scheduler's decision and what it reused follows from it, so a
    /// server or a measurement that wants to say a turn started warm has to be
    /// told here or not at all — and told at the step it was decided, since by
    /// the time the request is answered its slot has been given to somebody
    /// else.
    pub admitted: Vec<Admitted>,
    /// Every token this step decoded, against the ticket that owes it.
    ///
    /// **What a caller streaming its answers needs and a caller collecting them
    /// does not.** [`Answered`] arrives once, on the step a request finishes, so
    /// a server reading only that holds every token back until the last one —
    /// which is the whole reply buffered behind a budget that is seconds long.
    /// The same ids are in both: see
    /// `every_token_is_reported_against_the_ticket_that_produced_it`.
    pub produced: Vec<(usize, usize)>,
    /// Requests this step finished.
    pub done: Vec<Answered>,
    /// **What keeping conversations costs this step whether anything matched or
    /// not**, which is the question a feature like this owes about its own bad
    /// case: the matching against every free slot, the mark a seat takes when
    /// its prompt goes in, and the resume and the recording a departure pays.
    ///
    /// It does not include the forward pass, and it does include the fresh cache
    /// a miss builds — which is what this engine allocated at every admission
    /// before there was a conversation to keep. The mirror of
    /// [`Served::bookkeeping`](crate::keep::Served::bookkeeping) at the other
    /// width.
    pub bookkeeping: Duration,
}

impl Stepped {
    /// Rows the call carried, which is what it was charged for.
    pub fn rows(&self) -> usize {
        self.filling + self.decoding
    }

    /// Whether the engine had anything to do at all.
    pub fn empty(&self) -> bool {
        self.rows() == 0
    }
}

/// One of the engine's slots: the conversation its cache is sitting at the end
/// of, and whoever is seated in it now.
///
/// **The two have different lifetimes and that is the point.** A resident is one
/// request; the conversation outlives it, so the next request that is the same
/// conversation finds its own keys where it left them.
#[derive(Debug)]
struct Slot<'a> {
    kept: Kept<'a>,
    held: Option<Resident>,
    /// When this slot last had a sequence leave it, which is the order slots are
    /// evicted in — see [`Continuous::seat_for`]. Zero for a slot nothing has
    /// ever sat in, which is also a slot holding no conversation.
    vacated: u64,
}

impl Slot<'_> {
    /// Whether the sequence seated here is feeding prompt rather than a token it
    /// owes, and `false` for an empty slot.
    fn filling(&self) -> bool {
        self.held.as_ref().is_some_and(Resident::filling)
    }
}

/// The smallest scheduler that can answer what a request joining a running
/// batch waits: a fixed set of slots, a queue in front of them, and one step
/// that admits, fills, decodes and evicts.
#[derive(Debug)]
pub struct Continuous<'a> {
    slots: Vec<Slot<'a>>,
    waiting: VecDeque<(usize, Request)>,
    /// Prompt rows one step carries at most, **over every seat that is
    /// filling** — see [`Continuous::new`].
    chunk: usize,
    tickets: usize,
    /// Sequences that have left a slot, which is what stamps [`Slot::vacated`].
    /// A counter rather than a clock, because the order is all the eviction
    /// wants and a clock would make the choice depend on how fast the host is.
    vacancies: u64,
}

impl<'a> Continuous<'a> {
    /// `slots` sequences in flight at once, a step carrying at most `chunk` rows
    /// of prompt between all of them, and a slot keeping its conversation at up
    /// to `bound` positions once the sequence holding it leaves.
    ///
    /// **A returning conversation is recognised by its tokens**, which is the
    /// decision this milestone turns on and is stated here because this is where
    /// the bound is asked for. A slot records the ids its cache sits at the end
    /// of; an arriving prompt is given that slot only if it *starts with* those
    /// ids, so what admits it is a comparison against the keys themselves. A
    /// session id would be cheaper and would be a claim: a client that reused
    /// one, or two clients that collided on one, would put a sequence on top of
    /// keys that are not its own — which is the one failure this engine has
    /// never had and which no oracle over a single sequence can see. It would
    /// also need a field a real OpenAI client has no way to send.
    ///
    /// `bound` of zero keeps nothing, which is [`Continuous::new`] and is what
    /// every batching figure in this repo was taken under.
    pub fn keeping(config: &'a TextConfig, slots: usize, chunk: usize, bound: usize) -> Self {
        assert!(slots > 0, "an engine with no slots");
        assert!(
            chunk > 0,
            "a prompt fed no rows a step never enters a cache"
        );
        Self {
            slots: (0..slots)
                .map(|slot| Slot {
                    kept: Kept::in_slot(config, bound, slot),
                    held: None,
                    vacated: 0,
                })
                .collect(),
            waiting: VecDeque::new(),
            chunk,
            tickets: 0,
            vacancies: 0,
        }
    }

    /// `slots` sequences in flight at once, and a step carrying at most `chunk`
    /// rows of prompt between all of them.
    ///
    /// **A budget on the call and not on the seat**, which is the only reading
    /// under which the number bounds anything. Eight slots filling 128 rows
    /// apiece is a call of 1024 rows — measured at 46 tokens a second against
    /// the 119 the same width decodes at — so a per-seat chunk names a figure
    /// that is not the cost of a step and cannot be read as one. Split, it is
    /// what a step's prompt rows are capped at, and that is the sentence a
    /// scheduler needs.
    ///
    /// A seat always takes at least one row, so a budget smaller than the seats
    /// filling is exceeded rather than starving one of them. **A seat with less
    /// prompt left than its share takes what it has and the difference is not
    /// handed to a seat that could use it**, so a call comes in under the budget
    /// where the seats' prompts run out unevenly — which is a step this leaves
    /// narrower than it had to be and never wider than it said.
    ///
    /// `slots` is the width the backend was wrapped for and not a number this
    /// picks: a slot is a span and four convolution windows in every layer,
    /// allocated when the stack is wrapped rather than when a sequence sits in
    /// one. Naming more here than the backend holds is refused where the slot
    /// is handed over, which is one layer down and is where every other slot
    /// mistake is refused too.
    ///
    /// **This keeps nothing between requests**, so every sequence admitted here
    /// starts from a cache holding no keys. That is the engine every batching
    /// figure in this repo was taken on and it is the arm a kept conversation is
    /// measured against; [`Continuous::keeping`] is the other one.
    pub fn new(config: &'a TextConfig, slots: usize, chunk: usize) -> Self {
        Self::keeping(config, slots, chunk, 0)
    }

    /// Queue a request, and the ticket it will be answered under.
    ///
    /// The ticket is handed back rather than the caller naming one, so that two
    /// requests cannot share an identity and a latency cannot be attributed to
    /// the wrong arrival.
    pub fn submit(&mut self, request: Request) -> usize {
        assert!(!request.prompt.is_empty(), "a request with no prompt");
        let ticket = self.tickets;
        self.tickets += 1;
        self.waiting.push_back((ticket, request));
        ticket
    }

    /// Give a request up, seated or still queued, and answer whether it was
    /// there to give up.
    ///
    /// **The seat is free before the next step is built**, which is the property
    /// a server needs and a benchmark never did. A benchmark's requests all want
    /// their tokens; a fleet's client hangs up, and a slot held for one that
    /// hung up decodes a whole budget nobody reads while the request behind it
    /// waits. That is a leak that only appears under the load a scheduler exists
    /// for, so it is answered here rather than left to a caller polling
    /// [`Continuous::seated`].
    ///
    /// **A ticket released twice is not an error.** A request that finished on
    /// the same step its client hung up is one request ending for two reasons,
    /// and the second has nothing left to undo — so this reports rather than
    /// asserts, and a caller that cares can read the answer.
    ///
    /// The slot is left at the position its conversation ends at, which for an
    /// engine keeping nothing is a cache holding no keys — see
    /// [`Continuous::vacate`], where both of those are one line.
    ///
    /// **The weights are asked for because a sequence's state is in two places.**
    /// Putting a slot back where its conversation ends moves the cache on this
    /// side and the backend's own span and windows, and a caller that moved one
    /// and not the other would leave a slot whose position is one thing here and
    /// another there — and which still answers. It is the same argument
    /// [`ModelWeights::rewind`](crate::ModelWeights::rewind) is one call rather
    /// than two under.
    pub fn release(&mut self, ticket: usize, weights: &impl ModelWeights) -> bool {
        let seat = self
            .slots
            .iter()
            .position(|slot| slot.held.as_ref().is_some_and(|held| held.ticket == ticket));
        if let Some(at) = seat {
            self.vacate(at, weights);
            return true;
        }
        let queued = self.waiting.len();
        self.waiting.retain(|(waiting, _)| *waiting != ticket);
        self.waiting.len() < queued
    }

    /// How many requests are queued and not yet in a slot.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// How many slots hold a sequence.
    pub fn seated(&self) -> usize {
        self.slots.iter().filter(|slot| slot.held.is_some()).count()
    }

    /// Positions the conversation in slot `at` is being kept at, and zero for a
    /// slot holding none — which is every slot of an engine keeping nothing.
    pub fn kept_at(&self, at: usize) -> usize {
        self.slots[at].kept.held()
    }

    /// How many slots there are.
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Whether there is nothing seated and nothing queued.
    pub fn idle(&self) -> bool {
        self.seated() == 0 && self.waiting.is_empty()
    }

    /// One step of the whole engine: admit what is waiting, one forward pass
    /// carrying every seated sequence's next rows, and out of its slot with
    /// whatever that pass finished.
    ///
    /// **A sequence leaves its slot in the step that finished it**, so the slot
    /// is carrying the next request's prompt in the very next step — where an
    /// engine that held a finished sequence until somebody collected it would
    /// leave a slot idle at every handover.
    ///
    /// The cache a joining sequence is given holds either **nothing** or
    /// **tokens the sequence's own prompt names**, and there is no third
    /// arrangement. A cache at no keys is what puts the device's span and its
    /// four convolution windows back to the start of a sequence — a span is
    /// restarted where a sequence at no keys is handed over, see
    /// `LayerAttention::hold` — and a cache carried over from a *different*
    /// sequence would have the joiner attending over its keys, plausibly and
    /// quietly. See [`Continuous::seat_for`], which is the whole of the choice.
    pub fn step(&mut self, generator: &Generator<'_>, weights: &impl ModelWeights) -> Stepped {
        let mut stepped = Stepped::default();
        let at = Instant::now();
        self.admit(&mut stepped);
        stepped.bookkeeping = at.elapsed();
        // **Starvation, asserted where it can be seen.** A scheduler that never
        // admits a waiting request is wrong however fast its steps are, and the
        // shape that would be is a queue with a slot standing empty in front of
        // it. Checked every step rather than stated in a comment, because what
        // it costs is one comparison against a forward pass.
        assert!(
            self.waiting.is_empty() || self.slots.iter().all(|slot| slot.held.is_some()),
            "{} requests waiting while {} of {} slots stand empty",
            self.waiting.len(),
            self.slots.iter().filter(|slot| slot.held.is_none()).count(),
            self.slots.len(),
        );

        // **The seats are ordered before anything is borrowed**, filling in
        // front and decoding behind, because that order is what
        // `Generator::step_admitting` asks the tail for — the block is the last
        // rows of the call and the decoders are the rows that are questions.
        // Stable, so that within each group a seat is in slot order and a
        // reading is attributable to a slot.
        let mut seated: Vec<&mut Slot<'a>> = self
            .slots
            .iter_mut()
            .filter(|slot| slot.held.is_some())
            .collect();
        seated.sort_by_key(|slot| !slot.filling());
        if seated.is_empty() {
            return stepped;
        }

        // **The position a conversation will be kept at, taken before the first
        // row that is a question.** A prompt is in once a seat stops filling,
        // and one step later the generation has moved the cache past it — so
        // this is the only moment the slot stands where the next turn's prompt
        // will start, and a mark taken at the end of a *generation* would put
        // the cache somewhere in the middle of a reply the client never sends
        // back. It is where `Kept::turn` takes its own, for the same reason.
        //
        // Between runs, which is what a mark needs of its caller on a device:
        // nothing of this step has been encoded yet.
        let at = Instant::now();
        for slot in seated.iter_mut() {
            let Slot { kept, held, .. } = &mut **slot;
            let held = held.as_mut().expect("a seated slot");
            if !held.filling() && held.mark.is_none() {
                held.mark = Some(weights.mark(kept.cache()));
            }
        }
        stepped.bookkeeping += at.elapsed();

        // The step's prompt budget, split over the seats that are filling — see
        // [`Continuous::new`] for why the number is the call's and not the
        // seat's. **The remainder is handed out rather than dropped**: a budget
        // of seven over three seats is 3, 2, 2 and not 2, 2, 2, because the
        // rounding is otherwise a row a step that nobody may use and every
        // prompt takes longer to enter for it.
        let filling = seated.iter().filter(|slot| slot.filling()).count();
        let (share, mut spare) = match filling {
            0 => (0, 0),
            filling => (self.chunk / filling, self.chunk % filling),
        };

        let fed: Vec<Vec<usize>> = seated
            .iter()
            .map(|slot| {
                let held = slot.held.as_ref().expect("a seated slot");
                match held.filling() {
                    true => {
                        let over = usize::from(spare > 0);
                        spare -= over;
                        held.chunk(share + over).to_vec()
                    }
                    false => vec![held.owed()],
                }
            })
            .collect();
        stepped.seats = seated.len();
        stepped.decoding = seated.len() - filling;
        stepped.filling = fed.iter().map(Vec::len).sum::<usize>() - stepped.decoding;

        let mut batch: Vec<Batched<'_>> = seated
            .iter_mut()
            .zip(&fed)
            .map(|(slot, ids)| Batched {
                cache: slot.kept.cache_mut(),
                ids,
            })
            .collect();
        let picked = generator.step_admitting(&mut batch, stepped.decoding, weights);
        drop(batch);

        let mut answered = picked.into_iter();
        for (slot, ids) in seated.iter_mut().zip(&fed) {
            let held = slot.held.as_mut().expect("a seated slot");
            match held.filling() {
                true => held.filled += ids.len(),
                false => {
                    let id = answered.next().expect("an id per decoding seat");
                    if held.produced.is_empty() {
                        stepped.first.push(held.ticket);
                    }
                    held.produced.push(id);
                    stepped.produced.push((held.ticket, id));
                    if held.done() {
                        stepped.done.push(Answered {
                            ticket: held.ticket,
                            produced: held.produced.clone(),
                        });
                    }
                }
            }
        }

        let took = Instant::now();
        for at in 0..self.slots.len() {
            if self.slots[at].held.as_ref().is_some_and(Resident::done) {
                self.vacate(at, weights);
            }
        }
        stepped.bookkeeping += took.elapsed();
        stepped
    }

    /// The sequence in slot `at` out of it, and the slot left where its
    /// conversation ends.
    ///
    /// **The mark is what makes this two lines rather than one.** A slot put back
    /// to where its prompt ended is a slot the next turn of that conversation can
    /// be resumed into; a slot left where the *generation* ended is a slot
    /// standing in the middle of a reply, and the reply a client sends back is
    /// not the ids the model streamed — the turn structure renders a thinking
    /// channel as a message of its own where the model emits it inside one. So a
    /// prompt matching that position is a prompt nothing could honour, which is
    /// why the ids recorded are the ones the mark stands at.
    ///
    /// **A sequence with no mark has not generated anything, which is two
    /// different states and not one.** The mark is taken at the top of the first
    /// step a seat is not filling in, so there is an inter-step window — after
    /// the step that fed a prompt's last chunk, before the step that feeds its
    /// first row that is a question — where the cache stands exactly at
    /// [`Resident::kept`] and nothing has been marked. A client that hangs up
    /// waiting for a first token leaves its seat precisely there, and a prompt
    /// short enough to fill in one step passes through that window on the way to
    /// every reply. So what decides is whether the prompt is *in*, and not
    /// whether a mark was taken: a seat that is no longer filling is a seat whose
    /// cache is already where a mark would have put it, and the only thing left
    /// to do with it is record the ids.
    ///
    /// A sequence that never finished filling is the other state, and there is no
    /// position in that cache anything could be resumed to — so its slot is
    /// forgotten, which is to say handed on holding no keys at all. **That is
    /// this engine's own behaviour before there was a conversation to keep**, and
    /// it is what an engine keeping nothing does at every departure.
    fn vacate(&mut self, at: usize, weights: &impl ModelWeights) {
        self.vacancies += 1;
        let vacancy = self.vacancies;
        let slot = &mut self.slots[at];
        slot.vacated = vacancy;
        let Some(held) = slot.held.take() else { return };
        match &held.mark {
            Some(mark) => {
                weights.resume(slot.kept.cache_mut(), mark);
                slot.kept.keep(held.kept());
            }
            None if !held.filling() => slot.kept.keep(held.kept()),
            None => slot.kept.forget(),
        }
    }

    /// The slot an arriving prompt is given, and `None` where every one of them
    /// holds a sequence.
    ///
    /// **Three rules, in order, and the first is the milestone.**
    ///
    /// - A free slot whose kept ids the prompt *starts with*, longest match
    ///   first. That is a returning conversation finding its own keys, and what
    ///   it is admitted on is a comparison against the tokens themselves.
    /// - Failing that, a free slot holding no conversation, lowest first — so a
    ///   server with slots to spare evicts nobody.
    /// - Failing that, the free slot that last had a sequence leave it,
    ///   **longest ago**. That is the eviction, and it is where a fleet lives: N
    ///   agents over N slots evict nothing at all, and past that the
    ///   conversation that has gone longest without a turn is the one likeliest
    ///   to have ended.
    ///
    /// A slot with a sequence in it is not a candidate under any of the three,
    /// which is the admission rule this engine always had. **Nothing here can
    /// leave a request waiting while a slot is free**: every rule after the
    /// first ranks the same set, and the last of them is a total order over it.
    ///
    /// **So a conversation whose own slot is busy does not wait for it.** It is
    /// admitted somewhere else and prefills from the beginning, because waiting
    /// for a cache means waiting out another request's whole generation — a
    /// second turn of one conversation arriving while its first is still
    /// decoding would sit in the queue behind seconds of somebody else's budget
    /// to save a prefill. The starvation guarantee is the stronger promise and
    /// this is where the two meet.
    ///
    /// The eviction itself is not here and is not a step. A slot chosen by the
    /// third rule is one the prompt matches nothing of, and
    /// [`Kept::opened`] answers a prompt that matches nothing by forgetting —
    /// so what an evicted conversation costs its owner is a cold prefill, on the
    /// turn after the eviction, and nothing else.
    fn seat_for(&self, prompt: &[usize]) -> Option<usize> {
        let free = || {
            self.slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.held.is_none())
        };
        let resumed = free()
            .map(|(at, slot)| (at, slot.kept.matching(prompt)))
            .filter(|(_, matching)| *matching > 0)
            .max_by_key(|(at, matching)| (*matching, std::cmp::Reverse(*at)));
        if let Some((at, _)) = resumed {
            return Some(at);
        }
        free()
            .find(|(_, slot)| slot.kept.held() == 0)
            .or_else(|| free().min_by_key(|(_, slot)| slot.vacated))
            .map(|(at, _)| at)
    }

    /// Every free slot filled from the front of the queue.
    ///
    /// **First in, first out, and every free slot filled**, which is the whole
    /// of the starvation guarantee: a request at position `p` of the queue waits
    /// for `p` slots to free and for nothing else, and no slot is left empty
    /// while anything is waiting.
    ///
    /// The queue is walked in order and each request takes the slot
    /// [`Continuous::seat_for`] names, rather than the slots being walked in
    /// order and each taking the request in front of it. **That is the whole of
    /// what the matching changes about admission**: which free slot a request
    /// goes into is now a question about the request, and the order it is
    /// answered in is unmoved.
    fn admit(&mut self, stepped: &mut Stepped) {
        loop {
            let Some((_, request)) = self.waiting.front() else {
                return;
            };
            // A request wanting no tokens is answered by taking no slot at all,
            // and the queue is walked past it — a slot left to it would be a
            // slot standing empty in front of the request behind it.
            if request.count == 0 {
                self.waiting.pop_front();
                continue;
            }
            let Some(at) = self.seat_for(&request.prompt) else {
                return;
            };
            let (ticket, request) = self.waiting.pop_front().expect("the request just read");
            let slot = &mut self.slots[at];
            let (_, reused) = slot.kept.opened(&request.prompt);
            stepped.admitted.push(Admitted {
                ticket,
                slot: at,
                reused,
            });
            slot.held = Some(Resident {
                ticket,
                prompt: request.prompt,
                count: request.count,
                filled: reused,
                produced: Vec::new(),
                mark: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::Stack;
    use crate::head::LmHead;
    use crate::keep::DEFAULT_BOUND;
    use crate::model::ModelCache;
    use crate::ops::DenseProjection;
    use crate::workload::{Turned, turned};

    /// The head the synthetic stack's generator reads, which is its embedding
    /// table — see `generate`'s own tests, where the same tie is spelled out.
    fn generator<'a>(stack: &'a Stack, head: &'a DenseProjection<'a>) -> Generator<'a> {
        Generator::new(stack.model(), LmHead::for_config(&stack.config), head)
    }

    /// How many tokens the cases decode. Enough that a step reads what more than
    /// one step before it left — a short convolution's window is three inputs
    /// deep.
    const COUNT: usize = 4;

    /// Everything a run of the engine did: what each request was answered, and
    /// the rows the steps carried on each of the two accounts.
    ///
    /// **The row counts are the half of this that a token comparison cannot
    /// state.** The synthetic stack settles onto a repeated id within a couple
    /// of steps, so a sequence handed one token twice can produce the same
    /// continuation as one handed it once — which is a property of this fixture
    /// and not of the engine, and it is exactly what
    /// [`Ran::against`] is here to catch instead.
    #[derive(Debug, Default)]
    struct Ran {
        answered: Vec<Answered>,
        filling: usize,
        decoding: usize,
        /// Calls that carried a prompt row and a decode row at once, which is
        /// the shape a static batch never has and the one every case here is
        /// about. Counted rather than assumed: a chunk wide enough to swallow
        /// each prompt in one step would leave the engine alternating between
        /// prefills and decode steps and never reach it.
        mixed: usize,
        /// The most prompt rows any one step of the run carried, which is what
        /// the budget is a budget on.
        widest: usize,
    }

    impl Ran {
        fn took(&mut self, stepped: Stepped) {
            assert!(!stepped.empty(), "a step that carried no rows");
            self.filling += stepped.filling;
            self.decoding += stepped.decoding;
            self.mixed += usize::from(stepped.filling > 0 && stepped.decoding > 0);
            self.widest = self.widest.max(stepped.filling);
            self.answered.extend(stepped.done);
        }

        /// The engine run until it has nothing left, from wherever it is.
        fn on(&mut self, engine: &mut Continuous<'_>, stack: &Stack, head: &DenseProjection<'_>) {
            let generator = generator(stack, head);
            while !engine.idle() {
                self.took(engine.step(&generator, stack));
            }
        }

        /// **The rows a set of requests is worth, and no others.** Every prompt
        /// but its last token is filled once and every token asked for is
        /// decoded once — so a prompt fed twice, a prompt fed to its last token
        /// as well, or a decode step run for a sequence that had finished is a
        /// row this does not account for.
        ///
        /// The answers come back keyed by ticket rather than in the order they
        /// finished, because a continuous engine's whole point is that they do
        /// not finish in the order they arrived.
        fn against(&mut self, asked: &[Request]) -> &[Answered] {
            let wanted: usize = asked.iter().map(|request| request.count).sum();
            let prompted: usize = asked
                .iter()
                .filter(|request| request.count > 0)
                .map(|request| request.prompt.len() - 1)
                .sum();
            assert_eq!(self.filling, prompted, "prompt rows fed");
            assert_eq!(self.decoding, wanted, "rows decoded");
            self.answered.sort_by_key(|answer| answer.ticket);
            &self.answered
        }
    }

    /// The two of [`requests`] that differ in length, content and budget, which
    /// is the pair every release case drives: one request abandoned and one
    /// behind it, told apart by what they generate.
    fn a_pair(stack: &Stack) -> Vec<Request> {
        let mut asked = requests(stack);
        asked.truncate(2);
        asked
    }

    /// The same generation run alone, which is what every case below is held
    /// against.
    fn alone(stack: &Stack, prompt: &[usize], count: usize) -> Vec<usize> {
        let head = stack.head();
        generator(stack, &head).generate(&mut ModelCache::new(&stack.config), prompt, count, stack)
    }

    /// Three prompts that differ in every way a neighbour can: length, content,
    /// and how many tokens they ask for.
    fn requests(stack: &Stack) -> Vec<Request> {
        let sequence = stack.sequence();
        vec![
            Request {
                prompt: sequence[..3].to_vec(),
                count: COUNT,
            },
            Request {
                prompt: sequence[3..].to_vec(),
                count: COUNT,
            },
            Request {
                prompt: sequence[..4].to_vec(),
                count: COUNT - 2,
            },
        ]
    }

    /// **The contamination case for a slot that fills while its neighbours are
    /// decoding**: a request admitted into a running batch produces exactly what
    /// it produces alone, and so do the sequences it joined.
    ///
    /// This is the case `generate_batch` structurally cannot state. There, every
    /// sequence is prefilled before any of them decodes and the membership of
    /// the batch never grows — so a joining prompt's rows have never been in a
    /// call beside a decode row, and the seat that carries them has never been
    /// longer than its neighbours'. Here they are, at every arrival offset the
    /// budget allows.
    ///
    /// **Both halves are asserted and the second is the one that is new.** A
    /// joiner that read a neighbour's span would produce plausible tokens; so
    /// would a neighbour whose window the joiner's chunk wrote over. Exact
    /// equality against each sequence run alone is the only thing that says
    /// otherwise, and it is exact because every arm multiplies the same numbers
    /// in the same order.
    #[test]
    fn a_request_that_joins_a_running_batch_produces_what_it_produces_alone() {
        let stack = Stack::load();
        let head = stack.head();
        let asked = requests(&stack);
        let want: Vec<Vec<usize>> = asked
            .iter()
            .map(|request| alone(&stack, &request.prompt, request.count))
            .collect();
        assert_ne!(want[0], want[1], "two generations to tell apart");

        // The third arrives after the first two have been decoding for `late`
        // steps, which is a slot filling beside neighbours mid-generation.
        const CHUNKS: [usize; 3] = [1, 2, 8];
        let mut mixed = [0usize; CHUNKS.len()];
        for late in 0..=COUNT {
            for (which, chunk) in CHUNKS.into_iter().enumerate() {
                let mut engine = Continuous::new(&stack.config, asked.len(), chunk);
                engine.submit(asked[0].clone());
                engine.submit(asked[1].clone());

                let generator = generator(&stack, &head);
                let mut ran = Ran::default();
                for _ in 0..late {
                    ran.took(engine.step(&generator, &stack));
                }
                engine.submit(asked[2].clone());
                ran.on(&mut engine, &stack, &head);
                mixed[which] += ran.mixed;
                let answered = ran.against(&asked);

                assert_eq!(answered.len(), asked.len(), "late {late}, chunk {chunk}");
                for (at, answer) in answered.iter().enumerate() {
                    assert_eq!(answer.ticket, at);
                    assert_eq!(
                        answer.produced, want[at],
                        "sequence {at} joining {late} steps in, {chunk} rows a chunk"
                    );
                }
            }
        }
        // **That the case reached the shape it is about**, at every chunk. A
        // chunk wide enough to swallow each prompt in one step leaves the engine
        // alternating prefills with decode steps and never puts a prompt row in
        // a call beside one — which is a case that passes without asserting
        // anything, and the widest chunk here reaches the shape only through the
        // request that arrives late.
        for (chunk, mixed) in CHUNKS.into_iter().zip(mixed) {
            assert!(mixed > 0, "no mixed call at {chunk} rows a chunk");
        }
    }

    /// **The contamination case for a slot the sequence before it left**: a
    /// request that takes a finished one's slot produces what it produces alone.
    ///
    /// This is the other half of what a continuous engine does and the half
    /// where the leak is invisible. The device's span and its four convolution
    /// windows are the slot's rather than the sequence's — a slot handed on with
    /// the previous sequence's keys still counted would have the joiner attend
    /// over them, answer a row of the right shape out of a plausible softmax,
    /// and go on generating fluent text.
    ///
    /// One slot and four requests, so that every request but the first is a
    /// handover and the fourth is the third's slot at second hand.
    #[test]
    fn a_request_that_takes_a_finished_ones_slot_produces_what_it_produces_alone() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let asked: Vec<Request> = [0usize, 1, 2, 3]
            .iter()
            .map(|at| Request {
                prompt: sequence[*at..*at + 4].to_vec(),
                count: COUNT,
            })
            .collect();
        let want: Vec<Vec<usize>> = asked
            .iter()
            .map(|request| alone(&stack, &request.prompt, request.count))
            .collect();
        assert_ne!(want[0], want[1], "two generations to tell apart");

        let mut engine = Continuous::new(&stack.config, 1, 2);
        for request in &asked {
            engine.submit(request.clone());
        }
        let mut ran = Ran::default();
        ran.on(&mut engine, &stack, &head);
        let answered = ran.against(&asked);

        assert_eq!(answered.len(), asked.len());
        for (at, answer) in answered.iter().enumerate() {
            assert_eq!(answer.produced, want[at], "the {at}th sequence in the slot");
        }
    }

    /// **How many rows a prompt enters a step in changes no token.** A prompt
    /// filled two rows at a time and one filled whole are the same keys in the
    /// same order, so the tokens after them are the same tokens — exactly, which
    /// is what makes the chunk a latency knob rather than an approximation.
    #[test]
    fn the_rows_a_prompt_is_fed_in_change_no_token() {
        let stack = Stack::load();
        let head = stack.head();
        let asked = requests(&stack);

        let mut whole: Option<Vec<Answered>> = None;
        for chunk in [1, 2, 3, 64] {
            let mut engine = Continuous::new(&stack.config, asked.len(), chunk);
            for request in &asked {
                engine.submit(request.clone());
            }
            let mut ran = Ran::default();
            ran.on(&mut engine, &stack, &head);
            let answered = ran.against(&asked).to_vec();
            match &whole {
                None => whole = Some(answered),
                Some(want) => assert_eq!(&answered, want, "{chunk} rows a chunk"),
            }
        }
    }

    /// A request's first token arrives when its prompt is in and one row has
    /// been asked what follows it, and never before — so the step it is reported
    /// at is the step it was produced at.
    ///
    /// Stated in steps rather than in time, because what a chunk costs is the
    /// steps it spreads a prompt over and a clock would measure the host as
    /// well.
    #[test]
    fn a_first_token_is_reported_at_the_step_that_produced_it() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let sequence = stack.sequence();

        for chunk in [1, 2, 8] {
            let mut engine = Continuous::new(&stack.config, 1, chunk);
            let ticket = engine.submit(Request {
                prompt: sequence.clone(),
                count: COUNT,
            });
            // The prompt short of its last token, spread over the chunk, and one
            // decode step behind it.
            let want = (sequence.len() - 1).div_ceil(chunk) + 1;

            let mut steps = 0;
            let mut first = None;
            while first.is_none() {
                steps += 1;
                let stepped = engine.step(&generator, &stack);
                assert!(steps <= want, "no first token in {steps} steps");
                if stepped.first.contains(&ticket) {
                    first = Some(steps);
                }
            }
            assert_eq!(first, Some(want), "{chunk} rows a chunk");
        }
    }

    /// **Starvation, as the correctness property it is.** Every request
    /// submitted is answered, in the order it was submitted, and **every step
    /// carries as many sequences as there are requests outstanding to carry**,
    /// up to the slots there are.
    ///
    /// The last of those is what the assertion inside the engine cannot say on
    /// its own: an engine that admitted one request a step, or only ever into
    /// slot zero, would leave the queue draining and every step of it would
    /// still have a full complement of empty slots to point at. Counted against
    /// what the test knows is outstanding, a step that carried fewer is a
    /// request that waited for nothing.
    ///
    /// Two slots against five requests, so that every request but the first two
    /// waits for one to free; and the queue is grown while the engine runs, so
    /// that a request arriving behind a full queue is driven as well as one
    /// arriving into an empty engine.
    #[test]
    fn no_request_waits_while_a_slot_is_free() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let sequence = stack.sequence();
        let asked: Vec<Request> = (0..5)
            .map(|at| Request {
                prompt: sequence[at % 3..][..3].to_vec(),
                count: COUNT,
            })
            .collect();

        let mut engine = Continuous::new(&stack.config, 2, 2);
        let mut queued = asked.iter();
        for _ in 0..3 {
            engine.submit(queued.next().expect("a request").clone());
        }

        let mut answered: Vec<Answered> = Vec::new();
        let mut submitted = 3;
        while !engine.idle() {
            let outstanding = submitted - answered.len();
            let stepped = engine.step(&generator, &stack);
            assert_eq!(
                stepped.seats,
                outstanding.min(engine.slots()),
                "{} outstanding against {} slots",
                outstanding,
                engine.slots(),
            );
            answered.extend(stepped.done);
            if let Some(request) = queued.next() {
                engine.submit(request.clone());
                submitted += 1;
            }
        }

        assert_eq!(answered.len(), asked.len(), "every request answered");
        let served: Vec<usize> = answered.iter().map(|answer| answer.ticket).collect();
        let mut order = served.clone();
        order.sort_unstable();
        assert_eq!(
            served, order,
            "answered out of the order they were admitted"
        );
    }

    /// **A step's prompt rows are the call's budget and not the seat's.** Three
    /// prompts filling at once through a budget of six is six rows a step and
    /// not eighteen, which is the only reading under which the number bounds
    /// what a call costs — and a call is what the sequences already decoding
    /// wait for.
    ///
    /// The two ends are asserted as well as the middle: a budget smaller than
    /// the seats filling gives each of them one row rather than starving one,
    /// and a budget wider than the prompts have left comes in under itself.
    /// **And a budget that does not divide is met anyway**, which is what says
    /// the remainder is handed out rather than truncated away.
    #[test]
    fn a_steps_prompt_rows_are_the_calls_budget_and_not_a_seats() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let seats = 3;
        let asked: Vec<Request> = (0..seats)
            .map(|_| Request {
                prompt: sequence.clone(),
                count: COUNT,
            })
            .collect();

        for (chunk, widest) in [
            // Met exactly: two rows to each of three seats.
            (6, 6),
            (3, 3),
            // **Met exactly where it does not divide**, which is the remainder
            // being handed out rather than dropped: 3, 2, 2 and not 2, 2, 2.
            (7, 7),
            (5, 5),
            // Over the budget rather than starving a seat, which is the one
            // direction it may be missed in and is bounded by the seats.
            (2, 3),
            (1, 3),
            // Under it, the other way: the three prompts have eight rows left
            // apiece and there is nothing else to fill the budget with.
            (60, seats * (sequence.len() - 1)),
        ] {
            let mut engine = Continuous::new(&stack.config, seats, chunk);
            for request in &asked {
                engine.submit(request.clone());
            }
            let mut ran = Ran::default();
            ran.on(&mut engine, &stack, &head);
            ran.against(&asked);
            assert_eq!(ran.widest, widest, "{chunk} rows a step over {seats} seats");
            assert!(
                ran.widest <= chunk.max(seats),
                "{} rows in one call against a budget of {chunk}",
                ran.widest
            );
        }
    }

    /// **Seats whose prompts run out at different steps still never take more
    /// than the budget between them**, which is the case every other one here
    /// misses: they all give their concurrently-filling seats the same prompt,
    /// so a share nobody could use has never been left on the table.
    ///
    /// Three prompts of three lengths through a budget that does not divide
    /// them. What this says is the direction the miss goes in — **under the
    /// budget and never over it** — because a seat with less prompt left than
    /// its share takes what it has and the difference is not handed on.
    #[test]
    fn seats_whose_prompts_run_out_unevenly_stay_inside_the_budget() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let asked: Vec<Request> = [2usize, 5, sequence.len()]
            .iter()
            .map(|len| Request {
                prompt: sequence[..*len].to_vec(),
                count: COUNT,
            })
            .collect();
        let want: Vec<Vec<usize>> = asked
            .iter()
            .map(|request| alone(&stack, &request.prompt, request.count))
            .collect();

        for chunk in [1, 2, 5, 7, 11] {
            let mut engine = Continuous::new(&stack.config, asked.len(), chunk);
            for request in &asked {
                engine.submit(request.clone());
            }
            let mut ran = Ran::default();
            ran.on(&mut engine, &stack, &head);
            let answered = ran.against(&asked).to_vec();
            assert!(
                ran.widest <= chunk.max(asked.len()),
                "{} rows in one call against a budget of {chunk}",
                ran.widest
            );
            for (at, answer) in answered.iter().enumerate() {
                assert_eq!(answer.produced, want[at], "request {at}, {chunk} a step");
            }
        }
    }

    /// **A request whose prompt is one token never fills at all.** Its whole
    /// prompt is the row that is a question, so it goes straight to decoding on
    /// the step it is admitted — which is a branch of [`Resident::filling`] that
    /// no other case here reaches, and the one where an engine that filled
    /// `prompt[..n-1]` unconditionally would ask a call for zero rows.
    ///
    /// Driven beside a neighbour that *is* filling, so the step it is admitted
    /// on is a mixed call with a decoding seat that has produced nothing yet.
    #[test]
    fn a_request_whose_prompt_is_one_token_is_decoding_from_the_step_it_arrives_on() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let sequence = stack.sequence();
        let asked = [
            Request {
                prompt: sequence.clone(),
                count: COUNT,
            },
            Request {
                prompt: sequence[..1].to_vec(),
                count: COUNT,
            },
        ];
        let want: Vec<Vec<usize>> = asked
            .iter()
            .map(|request| alone(&stack, &request.prompt, request.count))
            .collect();
        assert_ne!(want[0], want[1], "two generations to tell apart");

        let mut engine = Continuous::new(&stack.config, 2, 2);
        for request in &asked {
            engine.submit(request.clone());
        }
        let mut ran = Ran::default();
        let first = engine.step(&generator, &stack);
        assert_eq!(first.decoding, 1, "the one-token prompt fed a question");
        assert!(first.filling > 0, "its neighbour was still filling");
        ran.took(first);
        ran.on(&mut engine, &stack, &head);

        let answered = ran.against(&asked);
        for (at, answer) in answered.iter().enumerate() {
            assert_eq!(answer.produced, want[at], "request {at}");
        }
    }

    /// A request that wants no tokens never takes a slot, and does not hold up
    /// the one behind it — which is the queue's one shape that could otherwise
    /// wedge it.
    #[test]
    fn a_request_wanting_no_tokens_takes_no_slot() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();

        let asked = [
            Request {
                prompt: sequence.clone(),
                count: 0,
            },
            Request {
                prompt: sequence[..3].to_vec(),
                count: COUNT,
            },
        ];
        let mut engine = Continuous::new(&stack.config, 1, 2);
        engine.submit(asked[0].clone());
        let wanted = engine.submit(asked[1].clone());

        let mut ran = Ran::default();
        ran.on(&mut engine, &stack, &head);
        let answered = ran.against(&asked);
        assert_eq!(answered.len(), 1);
        assert_eq!(answered[0].ticket, wanted);
        assert_eq!(
            answered[0].produced,
            alone(&stack, &sequence[..3], COUNT),
            "the request behind the empty one"
        );
    }

    /// **The ids a caller streams are the ids it would have collected.** A
    /// server that sends tokens as they arrive reads [`Stepped::produced`] and
    /// never sees an [`Answered`] until the last step, so the two have to carry
    /// the same generation — and they are filled by two different lines, which
    /// is exactly the pair that drifts.
    ///
    /// Held against `alone` as well, so this says the streamed ids are right and
    /// not only that the two disagree about nothing.
    ///
    /// **A ticket must not be readable as a slot index here**, which is the one
    /// way this could pass while the attribution is wrong: every other case in
    /// this file hands out tickets into an engine wide enough to seat all of
    /// them at once, so ticket `n` sits in slot `n` and an engine reporting the
    /// slot would be right by accident. Two slots for three requests puts the
    /// third in a slot it does not number, and the empty request in front —
    /// which takes ticket 0 and no slot at all — is what makes slot 0 a ticket
    /// nothing may be attributed to.
    #[test]
    fn every_token_is_reported_against_the_ticket_that_produced_it() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let asked = requests(&stack);

        let mut engine = Continuous::new(&stack.config, 2, 2);
        let empty = engine.submit(Request {
            prompt: stack.sequence(),
            count: 0,
        });
        let tickets: Vec<usize> = asked
            .iter()
            .map(|request| engine.submit(request.clone()))
            .collect();

        let mut streamed: Vec<Vec<usize>> = vec![Vec::new(); tickets.len() + 1];
        let mut collected: Vec<Answered> = Vec::new();
        while !engine.idle() {
            let stepped = engine.step(&generator, &stack);
            for (ticket, id) in &stepped.produced {
                streamed[*ticket].push(*id);
            }
            collected.extend(stepped.done);
        }

        assert!(
            streamed[empty].is_empty(),
            "a token was attributed to the request that took no slot: {:?}",
            streamed[empty]
        );
        collected.sort_by_key(|answer| answer.ticket);
        assert_eq!(collected.len(), asked.len());
        for (at, answer) in collected.iter().enumerate() {
            assert_eq!(answer.ticket, tickets[at]);
            assert_eq!(
                streamed[answer.ticket], answer.produced,
                "ticket {} streamed",
                answer.ticket
            );
            assert_eq!(
                answer.produced,
                alone(&stack, &asked[at].prompt, asked[at].count),
                "ticket {} against the same generation alone",
                answer.ticket
            );
        }
    }

    /// **A seat given up is a seat the request behind it takes.** One slot and
    /// two requests: the first is released a step into its generation, and the
    /// second is admitted into the slot it left — producing what it produces
    /// alone, which is what says the released sequence's keys did not come with
    /// the slot.
    ///
    /// The whole of what a client hanging up costs, stated as steps: the
    /// released request's remaining budget is never decoded, so the run is
    /// shorter than the two requests together asked for.
    #[test]
    fn a_released_seat_is_taken_by_the_request_behind_it() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let asked = a_pair(&stack);
        let want = alone(&stack, &asked[1].prompt, asked[1].count);

        let mut engine = Continuous::new(&stack.config, 1, 2);
        let abandoned = engine.submit(asked[0].clone());
        let behind = engine.submit(asked[1].clone());

        // Far enough in that the seat holds a prompt and a token of its own,
        // which is the state an abandoned seat is actually abandoned in.
        let mut decoded = 0;
        while decoded == 0 {
            decoded = engine.step(&generator, &stack).produced.len();
        }
        assert_eq!(engine.seated(), 1, "the first request is in the slot");
        assert_eq!(engine.waiting(), 1, "the second is behind it");

        assert!(
            engine.release(abandoned, &stack),
            "the seat was there to give up"
        );
        assert_eq!(engine.seated(), 0, "the slot is free in the same breath");

        let mut streamed = Vec::new();
        let mut answered = Vec::new();
        while !engine.idle() {
            let stepped = engine.step(&generator, &stack);
            streamed.extend(stepped.produced);
            answered.extend(stepped.done);
        }

        assert_eq!(answered.len(), 1, "the abandoned request is not answered");
        assert_eq!(answered[0].ticket, behind);
        assert_eq!(answered[0].produced, want, "the slot came with keys");
        assert!(
            streamed.iter().all(|(ticket, _)| *ticket == behind),
            "the abandoned seat decoded after it was released: {streamed:?}"
        );
    }

    /// A request released before it was ever seated never takes a slot at all,
    /// and the one behind it does not wait for it.
    ///
    /// Which is the queue's half of the same property: a client that hangs up
    /// while waiting is the commonest one there is — it hung up *because* it was
    /// waiting.
    #[test]
    fn a_request_released_while_it_waits_is_never_admitted() {
        let stack = Stack::load();
        let head = stack.head();
        let asked = a_pair(&stack);

        let mut engine = Continuous::new(&stack.config, 1, 2);
        let served = engine.submit(asked[0].clone());
        let abandoned = engine.submit(asked[1].clone());

        let generator = generator(&stack, &head);
        let mut ran = Ran::default();
        // The one slot taken, which is what leaves the second request waiting —
        // and waiting is why its client hung up.
        ran.took(engine.step(&generator, &stack));
        assert_eq!(engine.seated(), 1);
        assert_eq!(engine.waiting(), 1);

        assert!(
            engine.release(abandoned, &stack),
            "the queued request was there"
        );
        assert_eq!(engine.waiting(), 0);

        ran.on(&mut engine, &stack, &head);
        let answered = ran.against(&asked[..1]);
        assert_eq!(answered.len(), 1);
        assert_eq!(answered[0].ticket, served);
    }

    /// **Releasing a ticket that is not there is not an error.** A request that
    /// finished on the step its client hung up is one request ending twice, and
    /// a server that had to tell the two apart before it could clean up would
    /// have to win that race.
    #[test]
    fn releasing_a_ticket_the_engine_does_not_hold_changes_nothing() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let asked = [Request {
            prompt: sequence[..3].to_vec(),
            count: COUNT,
        }];

        let mut engine = Continuous::new(&stack.config, 2, 2);
        let ticket = engine.submit(asked[0].clone());
        assert!(
            !engine.release(ticket + 1, &stack),
            "a ticket never handed out"
        );

        let mut ran = Ran::default();
        ran.on(&mut engine, &stack, &head);
        assert_eq!(ran.against(&asked).len(), 1);

        assert!(
            !engine.release(ticket, &stack),
            "a request already answered"
        );
        assert!(engine.idle());
    }

    #[test]
    #[should_panic(expected = "a request with no prompt")]
    fn a_request_with_no_prompt_is_refused() {
        let stack = Stack::load();
        Continuous::new(&stack.config, 1, 1).submit(Request {
            prompt: Vec::new(),
            count: 1,
        });
    }

    /// How many turns a conversation below takes.
    ///
    /// Three, because two cannot tell a conversation returning to its own slot
    /// from one that happened to be given the same slot twice: the third turn is
    /// the one that has to find keys a *second* departure left, in a slot two
    /// other sequences have been through.
    const TURNS: usize = 3;

    /// How many tokens of its own a turn adds on the end of what it was
    /// answered, which is the "and a little more" a coding turn is.
    const ADDED: usize = 2;

    /// One turn of one conversation, as the engine answered it.
    #[derive(Debug, Clone)]
    struct Turn {
        prompt: Vec<usize>,
        produced: Vec<usize>,
        slot: usize,
        reused: usize,
    }

    /// Several conversations through one engine, each sending its whole history
    /// back every turn — **which is the workload the whole arrangement is for,
    /// and the one nothing here had ever driven.**
    ///
    /// A turn's prompt is the turn before it, the reply it was given, and a
    /// couple of tokens of its own, so every turn but the first is an exact
    /// extension of the one before it — the only shape [`Kept::matching`] can
    /// serve, and the shape a client sending a conversation back produces.
    ///
    /// A turn of every conversation is submitted together and the engine is
    /// drained before the next, so the slots are free when each turn arrives.
    /// That is what makes the seating a statement about the *matching*: a
    /// conversation given its own slot here was given it because its tokens said
    /// so, not because it was the only slot going.
    fn talking(
        engine: &mut Continuous<'_>,
        stack: &Stack,
        head: &DenseProjection<'_>,
        openings: &[Vec<usize>],
    ) -> Vec<Vec<Turn>> {
        let generator = generator(stack, head);
        let sequence = stack.sequence();
        let mut prompts: Vec<Vec<usize>> = openings.to_vec();
        let mut taken: Vec<Vec<Turn>> = vec![Vec::new(); openings.len()];

        for turn in 0..TURNS {
            let asking: Vec<Request> = prompts
                .iter()
                .map(|prompt| Request {
                    prompt: prompt.clone(),
                    count: COUNT,
                })
                .collect();
            let answered = turned(engine, &generator, stack, &asking, |_| {});

            for (which, prompt) in prompts.iter_mut().enumerate() {
                let Turned { seat, produced } = &answered[which];
                taken[which].push(Turn {
                    prompt: prompt.clone(),
                    produced: produced.clone(),
                    slot: seat.slot,
                    reused: seat.reused,
                });
                prompt.extend_from_slice(produced);
                prompt.extend(sequence.iter().copied().cycle().skip(turn).take(ADDED));
            }
        }
        taken
    }

    /// Two conversations that differ from their first token, which is what makes
    /// the tokens each is answered with a thing the other could not have
    /// produced.
    fn two_conversations(stack: &Stack) -> Vec<Vec<usize>> {
        let sequence = stack.sequence();
        vec![sequence[..3].to_vec(), sequence[3..].to_vec()]
    }

    /// One request through an engine until it is answered, and the seat it was
    /// given.
    fn once(
        engine: &mut Continuous<'_>,
        stack: &Stack,
        head: &DenseProjection<'_>,
        prompt: &[usize],
    ) -> (Admitted, Vec<usize>) {
        let generator = generator(stack, head);
        let ticket = engine.submit(Request {
            prompt: prompt.to_vec(),
            count: COUNT,
        });
        let (mut seat, mut produced) = (None, None);
        while !engine.idle() {
            let stepped = engine.step(&generator, stack);
            seat = seat.or_else(|| {
                stepped
                    .admitted
                    .iter()
                    .copied()
                    .find(|at| at.ticket == ticket)
            });
            produced = produced.or_else(|| {
                stepped
                    .done
                    .iter()
                    .find(|answer| answer.ticket == ticket)
                    .map(|answer| answer.produced.clone())
            });
        }
        (
            seat.expect("the request was seated"),
            produced.expect("the request was answered"),
        )
    }

    /// **The contamination case this milestone is about**, and the one the two
    /// above cannot state: two conversations coming back to slots that hold
    /// their own keys, turn after turn, each answered exactly what it is
    /// answered alone.
    ///
    /// The invariant that made batching safe was that a joining sequence never
    /// attends over what it found in the slot. This relaxes it — a returning
    /// sequence attends over precisely what it found — so what has to hold in
    /// its place is that the keys it found are the ones its own prompt named.
    /// Nothing about the tokens of a *single* conversation can say that: a
    /// sequence resumed onto its neighbour's keys produces fluent text, and so
    /// does one resumed onto its own from the wrong position. **Two
    /// conversations interleaved through the same slots, held against
    /// themselves run alone, is the only arrangement where being wrong shows
    /// up.**
    ///
    /// Three claims, and each is a different way to be wrong:
    ///
    /// - Every turn's ids are the ids that turn's whole prompt produces from
    ///   nothing. That is the leak.
    /// - Every turn after the first came back to the slot the one before it
    ///   used. That is the matching doing the choosing.
    /// - What it reused is exactly the tokens the last turn's prompt left
    ///   behind, so the delta prefilled is the addition and not a token more or
    ///   less.
    #[test]
    fn two_conversations_coming_back_to_their_own_slots_each_get_their_own_tokens() {
        let stack = Stack::load();
        let head = stack.head();
        let openings = two_conversations(&stack);

        let mut engine = Continuous::keeping(&stack.config, openings.len(), 2, DEFAULT_BOUND);
        let talked = talking(&mut engine, &stack, &head, &openings);

        assert_ne!(
            talked[0][0].produced, talked[1][0].produced,
            "two conversations to tell apart"
        );
        assert_ne!(
            talked[0][0].slot, talked[1][0].slot,
            "both conversations opened in one slot"
        );

        for (which, turns) in talked.iter().enumerate() {
            assert_eq!(turns.len(), TURNS);
            assert_eq!(turns[0].reused, 0, "conversation {which} opened warm");
            for (at, turn) in turns.iter().enumerate() {
                assert_eq!(
                    turn.produced,
                    alone(&stack, &turn.prompt, COUNT),
                    "conversation {which}, turn {at}, against the whole prompt alone"
                );
                if at == 0 {
                    continue;
                }
                assert_eq!(
                    turn.slot,
                    turns[at - 1].slot,
                    "conversation {which} came back to another slot at turn {at}"
                );
                assert_eq!(
                    turn.reused,
                    turns[at - 1].prompt.len() - 1,
                    "conversation {which}, turn {at}, resumed to the wrong position"
                );
            }
        }
    }

    /// **Cold and warm are the same tokens at width greater than one**, which is
    /// what a kept conversation has to be before it is a latency optimisation at
    /// all: the same fleet through an engine that keeps nothing and one that
    /// keeps everything, id for id.
    ///
    /// Held against `alone` in the case above as well, so the two arms agreeing
    /// is not the whole of the claim — two arms can agree by being wrong the
    /// same way, and only one of them prefills the whole prompt every turn.
    #[test]
    fn an_engine_keeping_conversations_answers_what_one_keeping_nothing_answers() {
        let stack = Stack::load();
        let head = stack.head();
        let openings = two_conversations(&stack);

        let mut cold = Continuous::new(&stack.config, openings.len(), 2);
        let mut warm = Continuous::keeping(&stack.config, openings.len(), 2, DEFAULT_BOUND);
        let cold = talking(&mut cold, &stack, &head, &openings);
        let warm = talking(&mut warm, &stack, &head, &openings);

        assert!(
            cold.iter().flatten().all(|turn| turn.reused == 0),
            "the arm that keeps nothing reused something"
        );
        assert!(
            warm.iter()
                .all(|turns| turns[1..].iter().all(|t| t.reused > 0)),
            "the arm that keeps everything reused nothing"
        );
        for (which, (cold, warm)) in cold.iter().zip(&warm).enumerate() {
            for (at, (cold, warm)) in cold.iter().zip(warm).enumerate() {
                assert_eq!(cold.prompt, warm.prompt, "conversation {which}, turn {at}");
                assert_eq!(
                    cold.produced, warm.produced,
                    "conversation {which}, turn {at}"
                );
            }
        }
    }

    /// **A conversation returning to a slot that was evicted prefills from the
    /// beginning, and is right.** This is the case that decides whether eviction
    /// is a policy or a bug: a slot handed to somebody else and then handed back
    /// holds a stranger's keys, and a returning conversation that matched it on
    /// anything but its own tokens would attend over them.
    ///
    /// One slot and three conversations. The second evicts the first, and the
    /// first comes back to find its own slot holding the second — which is a
    /// miss, a prefill of the whole prompt, and the same ids the prompt produces
    /// alone.
    #[test]
    fn a_conversation_returning_to_an_evicted_slot_prefills_it_again_and_is_right() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let opening = sequence[..3].to_vec();
        let other = sequence[3..].to_vec();

        let mut engine = Continuous::keeping(&stack.config, 1, 2, DEFAULT_BOUND);
        let (_, first) = once(&mut engine, &stack, &head, &opening);
        assert_eq!(engine.kept_at(0), opening.len() - 1, "nothing was kept");

        // The turn that would have been resumed, had anything else not taken the
        // slot in between.
        let mut returning = opening.clone();
        returning.extend_from_slice(&first);
        returning.extend_from_slice(&sequence[..ADDED]);

        let (evicting, _) = once(&mut engine, &stack, &head, &other);
        assert_eq!(
            evicting.reused, 0,
            "a stranger matched the kept conversation"
        );
        assert_eq!(
            engine.kept_at(0),
            other.len() - 1,
            "the slot kept the evicted conversation"
        );

        let (back, produced) = once(&mut engine, &stack, &head, &returning);
        assert_eq!(back.reused, 0, "an evicted conversation was resumed");
        assert_eq!(
            produced,
            alone(&stack, &returning, COUNT),
            "the returning conversation read what evicted it"
        );
    }

    /// **The slot a new conversation takes is the one that went longest without
    /// a sequence in it**, which is the whole of the eviction policy and is
    /// where a fleet lives: N agents through N slots evict nothing at all, and
    /// past that the conversation that has gone longest without a turn is the
    /// one likeliest to have ended.
    ///
    /// Two slots and three conversations, so exactly one of the two has to go
    /// and which one is the claim. The first conversation's slot is vacated
    /// before the second's, so the first is the one that goes — and the second's
    /// next turn still resumes, which is what says the eviction took one slot
    /// rather than the pair.
    #[test]
    fn a_new_conversation_evicts_the_slot_that_went_longest_without_a_sequence() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let openings = [
            sequence[..3].to_vec(),
            sequence[3..].to_vec(),
            sequence[1..5].to_vec(),
        ];

        let mut engine = Continuous::keeping(&stack.config, 2, 2, DEFAULT_BOUND);
        let (first, _) = once(&mut engine, &stack, &head, &openings[0]);
        let (second, _) = once(&mut engine, &stack, &head, &openings[1]);
        assert_ne!(first.slot, second.slot, "both took one slot");

        let (third, _) = once(&mut engine, &stack, &head, &openings[2]);
        assert_eq!(
            third.slot, first.slot,
            "the newest conversation was evicted rather than the oldest"
        );
        assert_eq!(third.reused, 0);

        // The conversation that was not evicted still comes back to its own
        // keys, which is what says one slot went and not the pair.
        let mut returning = openings[1].clone();
        returning.extend_from_slice(&sequence[..ADDED]);
        let (back, produced) = once(&mut engine, &stack, &head, &returning);
        assert_eq!(back.slot, second.slot);
        assert_eq!(back.reused, openings[1].len() - 1);
        assert_eq!(produced, alone(&stack, &returning, COUNT));
    }

    /// **A seat given up before its prompt was in leaves no conversation
    /// behind.** The position a slot is kept at is where a prompt ends, and a
    /// sequence still filling has never stood there — so there is nothing to
    /// resume to and the slot is handed on holding no keys, which is exactly
    /// what it was handed on holding before any of this existed.
    ///
    /// The request behind it is what says so: it produces what it produces
    /// alone, from a slot whose predecessor left mid-prefill.
    #[test]
    fn a_seat_given_up_before_its_prompt_was_in_keeps_no_conversation() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let asked = a_pair(&stack);
        let want = alone(&stack, &asked[1].prompt, asked[1].count);

        let mut engine = Continuous::keeping(&stack.config, 1, 1, DEFAULT_BOUND);
        let abandoned = engine.submit(asked[0].clone());
        engine.submit(asked[1].clone());

        // One row of a prompt longer than one row, so the seat is given up
        // while it is still filling.
        let stepped = engine.step(&generator, &stack);
        assert_eq!(stepped.decoding, 0, "the seat had already stopped filling");
        assert!(engine.release(abandoned, &stack), "the seat was there");
        assert_eq!(engine.kept_at(0), 0, "a half-filled prompt was kept");

        let (behind, produced) = {
            let mut seat = None;
            let mut produced = None;
            while !engine.idle() {
                let stepped = engine.step(&generator, &stack);
                seat = seat.or_else(|| stepped.admitted.first().copied());
                produced =
                    produced.or_else(|| stepped.done.first().map(|answer| answer.produced.clone()));
            }
            (seat.expect("the request behind was seated"), produced)
        };
        assert_eq!(behind.reused, 0);
        assert_eq!(
            produced.expect("the request behind was answered"),
            want,
            "the slot came with a half-filled prompt"
        );
    }

    /// **A client that hangs up after its prompt is in still leaves its
    /// conversation behind**, and the turn that comes back finds it.
    ///
    /// Which is the case a fleet actually produces: an agent cancels a turn —
    /// the user typed something else, the tool call came back — and asks again
    /// over the same context. The keys are as real as if the turn had finished,
    /// because the position kept is where the *prompt* ended and the generation
    /// is resumed away either way.
    #[test]
    fn a_seat_given_up_after_its_prompt_was_in_keeps_the_conversation() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let sequence = stack.sequence();
        let opening = sequence[..4].to_vec();

        let mut engine = Continuous::keeping(&stack.config, 1, 2, DEFAULT_BOUND);
        let abandoned = engine.submit(Request {
            prompt: opening.clone(),
            count: COUNT,
        });
        let mut produced = Vec::new();
        while produced.is_empty() {
            produced = engine.step(&generator, &stack).produced;
        }
        assert!(engine.release(abandoned, &stack), "the seat was there");
        assert_eq!(
            engine.kept_at(0),
            opening.len() - 1,
            "an abandoned turn's conversation was thrown away"
        );

        let mut returning = opening.clone();
        returning.extend_from_slice(&sequence[..ADDED]);
        let (back, produced) = once(&mut engine, &stack, &head, &returning);
        assert_eq!(back.reused, opening.len() - 1);
        assert_eq!(
            produced,
            alone(&stack, &returning, COUNT),
            "the resumed turn read the cancelled generation's keys"
        );
    }

    /// **A seat given up between its prompt going in and its first token coming
    /// out keeps the conversation**, which is a window one step wide and is
    /// where a client waiting on a first token actually hangs up.
    ///
    /// The mark is taken at the top of the first step a seat is *not* filling
    /// in, so a seat that finished filling in the step just gone has no mark and
    /// a cache standing exactly where one would have put it. Deciding on the
    /// mark rather than on the prompt threw that conversation away — silently,
    /// and for every prompt short enough to fill in one step, which is every
    /// prompt under the server's own budget of 128 rows.
    ///
    /// Two turns rather than a count of positions, because the position is what
    /// the bug hid behind: the second turn resumes and produces what its whole
    /// prompt produces alone.
    #[test]
    fn a_seat_given_up_between_its_prompt_and_its_first_token_keeps_the_conversation() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let sequence = stack.sequence();
        let opening = sequence[..4].to_vec();

        // A budget wide enough to swallow the prompt in one step, which is what
        // makes the window the step after admission.
        let mut engine = Continuous::keeping(&stack.config, 1, 64, DEFAULT_BOUND);
        let abandoned = engine.submit(Request {
            prompt: opening.clone(),
            count: COUNT,
        });
        let stepped = engine.step(&generator, &stack);
        assert_eq!(
            stepped.filling,
            opening.len() - 1,
            "the prompt went in whole"
        );
        assert_eq!(
            stepped.decoding, 0,
            "a token was produced before the window"
        );

        assert!(engine.release(abandoned, &stack), "the seat was there");
        assert_eq!(
            engine.kept_at(0),
            opening.len() - 1,
            "a prompt that was fully in was thrown away"
        );

        let mut returning = opening.clone();
        returning.extend_from_slice(&sequence[..ADDED]);
        let (back, produced) = once(&mut engine, &stack, &head, &returning);
        assert_eq!(back.reused, opening.len() - 1);
        assert_eq!(
            produced,
            alone(&stack, &returning, COUNT),
            "the resumed turn read a cache that was not where it was recorded"
        );
    }

    /// A conversation grown past what a slot will keep is served and then
    /// forgotten, so the turn after it prefills from the beginning — the bound
    /// declines to keep rather than declining to answer.
    #[test]
    fn a_conversation_past_the_slots_bound_is_served_and_then_forgotten() {
        let stack = Stack::load();
        let head = stack.head();
        let sequence = stack.sequence();
        let opening = sequence[..4].to_vec();

        let mut engine = Continuous::keeping(&stack.config, 1, 2, opening.len() - 2);
        let (_, produced) = once(&mut engine, &stack, &head, &opening);
        assert_eq!(
            engine.kept_at(0),
            0,
            "the bound kept a conversation past it"
        );

        let mut returning = opening.clone();
        returning.extend_from_slice(&produced);
        let (back, produced) = once(&mut engine, &stack, &head, &returning);
        assert_eq!(back.reused, 0);
        assert_eq!(produced, alone(&stack, &returning, COUNT));
    }
}
