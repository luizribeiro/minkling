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
//! - **Filling.** Its prompt is going into its cache, [`Continuous::chunk`]
//!   tokens of it a step, in the same forward pass the sequences already in
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
//! sequences already decoding wait the whole of it — a 385-token prompt is about
//! a second on this part against a 15 ms decode step, so one arrival stalls
//! every sequence in flight for sixty steps' worth of time. Fed
//! [`Continuous::chunk`] rows a step, the same prompt is spread over
//! `ceil((n-1)/chunk)` steps and what any one of them costs the sequences in
//! flight is bounded by the chunk.
//!
//! **What it costs is that the joiner's rows lose the query block.**
//! `FusedAttention::blocked` refuses a call carrying more than one sequence —
//! a block stages one tile of keys for the 64 query rows sharing it, and rows
//! attending over different spans cannot share one — so a chunk riding beside
//! decoders walks its keys a query row at a time where a prefill of its own
//! would have blocked them. That is a real price and it is measured rather than
//! assumed; see the README.
//!
//! # Starvation is a correctness property
//!
//! A scheduler that never admits a waiting request is wrong however fast its
//! steps are. The queue here is first in, first out, and every free slot is
//! filled from the front of it before a step is built — so a request at position
//! `p` waits for `p` slots to free and for nothing else. Both halves of that are
//! asserted: no request waits while a slot is free, and a stream of requests
//! through one slot comes back in the order it was submitted.

use std::collections::VecDeque;

use crate::config::TextConfig;
use crate::generate::Generator;
use crate::model::{Batched, ModelCache, ModelWeights};

/// What a caller asks the engine for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The prompt, which may not be empty: a forward pass over no tokens is
    /// refused below this line and the refusal would land a step later, on a
    /// batch that has nothing to do with it.
    pub prompt: Vec<usize>,
    /// How many tokens to decode. A request that wants none is answered
    /// immediately and never takes a slot.
    pub count: usize,
}

/// One request, answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answered {
    pub ticket: usize,
    pub produced: Vec<usize>,
}

/// What one sequence in one slot carries between steps.
#[derive(Debug)]
struct Resident {
    ticket: usize,
    prompt: Vec<usize>,
    count: usize,
    /// Tokens of `prompt` already in `cache`, which never reaches its last one:
    /// see the module documentation.
    filled: usize,
    produced: Vec<usize>,
    cache: ModelCache,
}

impl Resident {
    /// Whether this sequence's next rows are prompt rather than a token it owes.
    fn filling(&self) -> bool {
        self.filled + 1 < self.prompt.len()
    }

    /// The prompt rows it feeds next, which is never empty while it is filling.
    fn chunk(&self, chunk: usize) -> &[usize] {
        &self.prompt[self.filled..(self.filled + chunk).min(self.prompt.len() - 1)]
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
    /// Requests this step finished.
    pub done: Vec<Answered>,
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

/// The smallest scheduler that can answer what a request joining a running
/// batch waits: a fixed set of slots, a queue in front of them, and one step
/// that admits, fills, decodes and evicts.
#[derive(Debug)]
pub struct Continuous<'a> {
    config: &'a TextConfig,
    slots: Vec<Option<Resident>>,
    waiting: VecDeque<(usize, Request)>,
    /// Prompt rows one filling seat feeds in one step.
    chunk: usize,
    tickets: usize,
}

impl<'a> Continuous<'a> {
    /// `slots` sequences in flight at once, a joining prompt entering `chunk`
    /// rows a step.
    ///
    /// `slots` is the width the backend was wrapped for and not a number this
    /// picks: a slot is a span and four convolution windows in every layer,
    /// allocated when the stack is wrapped rather than when a sequence sits in
    /// one. Naming more here than the backend holds is refused where the slot
    /// is handed over, which is one layer down and is where every other slot
    /// mistake is refused too.
    pub fn new(config: &'a TextConfig, slots: usize, chunk: usize) -> Self {
        assert!(slots > 0, "an engine with no slots");
        assert!(
            chunk > 0,
            "a prompt fed no rows a step never enters a cache"
        );
        Self {
            config,
            slots: (0..slots).map(|_| None).collect(),
            waiting: VecDeque::new(),
            chunk,
            tickets: 0,
        }
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

    /// How many requests are queued and not yet in a slot.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// How many slots hold a sequence.
    pub fn seated(&self) -> usize {
        self.slots.iter().flatten().count()
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
    /// The cache a joining sequence is given is a **fresh** one in that slot,
    /// which is what puts the device's span and its four convolution windows
    /// back to nothing: a span is restarted where a sequence at no keys is
    /// handed over — see `LayerAttention::hold` — and a cache carried over from
    /// the sequence before would have the joiner attending over its keys,
    /// plausibly and quietly.
    pub fn step(&mut self, generator: &Generator<'_>, weights: &impl ModelWeights) -> Stepped {
        let mut stepped = Stepped::default();
        self.admit();
        // **Starvation, asserted where it can be seen.** A scheduler that never
        // admits a waiting request is wrong however fast its steps are, and the
        // shape that would be is a queue with a slot standing empty in front of
        // it. Checked every step rather than stated in a comment, because what
        // it costs is one comparison against a forward pass.
        assert!(
            self.waiting.is_empty() || self.slots.iter().all(Option::is_some),
            "{} requests waiting while {} of {} slots stand empty",
            self.waiting.len(),
            self.slots.iter().filter(|slot| slot.is_none()).count(),
            self.slots.len(),
        );

        // **The seats are ordered before anything is borrowed**, filling in
        // front and decoding behind, because that order is what
        // `Generator::step_admitting` asks the tail for — the block is the last
        // rows of the call and the decoders are the rows that are questions.
        // Stable, so that within each group a seat is in slot order and a
        // reading is attributable to a slot.
        let mut seated: Vec<&mut Resident> = self.slots.iter_mut().flatten().collect();
        seated.sort_by_key(|held| !held.filling());
        if seated.is_empty() {
            return stepped;
        }

        let fed: Vec<Vec<usize>> = seated
            .iter()
            .map(|held| match held.filling() {
                true => held.chunk(self.chunk).to_vec(),
                false => vec![held.owed()],
            })
            .collect();
        stepped.seats = seated.len();
        stepped.decoding = seated.iter().filter(|held| !held.filling()).count();
        stepped.filling = fed.iter().map(Vec::len).sum::<usize>() - stepped.decoding;

        let mut batch: Vec<Batched<'_>> = seated
            .iter_mut()
            .zip(&fed)
            .map(|(held, ids)| Batched {
                cache: &mut held.cache,
                ids,
            })
            .collect();
        let picked = generator.step_admitting(&mut batch, stepped.decoding, weights);
        drop(batch);

        let mut answered = picked.into_iter();
        for (held, ids) in seated.iter_mut().zip(&fed) {
            match held.filling() {
                true => held.filled += ids.len(),
                false => {
                    let id = answered.next().expect("an id per decoding seat");
                    if held.produced.is_empty() {
                        stepped.first.push(held.ticket);
                    }
                    held.produced.push(id);
                    if held.done() {
                        stepped.done.push(Answered {
                            ticket: held.ticket,
                            produced: held.produced.clone(),
                        });
                    }
                }
            }
        }

        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(Resident::done) {
                *slot = None;
            }
        }
        stepped
    }

    /// Every free slot filled from the front of the queue.
    ///
    /// **First in, first out, and every free slot filled**, which is the whole
    /// of the starvation guarantee: a request at position `p` of the queue waits
    /// for `p` slots to free and for nothing else, and no slot is left empty
    /// while anything is waiting.
    fn admit(&mut self) {
        for slot in 0..self.slots.len() {
            if self.slots[slot].is_some() {
                continue;
            }
            // A request wanting no tokens is answered by taking no slot at all,
            // and the queue is walked past it — a slot left to it would be a
            // slot standing empty in front of the request behind it.
            while let Some((ticket, request)) = self.waiting.pop_front() {
                if request.count == 0 {
                    continue;
                }
                self.slots[slot] = Some(Resident {
                    ticket,
                    prompt: request.prompt,
                    count: request.count,
                    filled: 0,
                    produced: Vec::new(),
                    cache: ModelCache::in_slot(self.config, 0, slot),
                });
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::Stack;
    use crate::head::LmHead;
    use crate::ops::DenseProjection;

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
    }

    impl Ran {
        fn took(&mut self, stepped: Stepped) {
            assert!(!stepped.empty(), "a step that carried no rows");
            self.filling += stepped.filling;
            self.decoding += stepped.decoding;
            self.mixed += usize::from(stepped.filling > 0 && stepped.decoding > 0);
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

    #[test]
    #[should_panic(expected = "a request with no prompt")]
    fn a_request_with_no_prompt_is_refused() {
        let stack = Stack::load();
        Continuous::new(&stack.config, 1, 1).submit(Request {
            prompt: Vec::new(),
            count: 1,
        });
    }
}
