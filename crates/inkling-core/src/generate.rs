//! Generation: a prompt in, tokens out, one at a time.
//!
//! Nothing here is a new op. What this is, is the loop — and the loop is where
//! the second of the model's two regimes finally runs.
//!
//! # The two regimes
//!
//! Everything below this line has been driven in **prefill**: some number of
//! tokens at once, against caches that start empty. Every fixture in the tree is
//! that, or a continuation of it. **Decode** is one token against caches that
//! carry every prior step, and it is where a cache mistake lives — a window
//! written but never read, a convolution slot exchanged with its neighbour, a
//! query offset that never advanced.
//!
//! The invariant is that the two agree:
//!
//! > Prefilling N tokens and then decoding the (N+1)th gives the same logits as
//! > one prefill over all N+1.
//!
//! That is the property the engine rests on, and on the synthetic stack it holds
//! *exactly* rather than within a tolerance — both paths multiply the same
//! numbers in the same order, and the only thing a split changes is where they
//! come from. It is the same claim [`ShortConv`](crate::sconv::ShortConv) and
//! [`DecoderLayer`](crate::layer::DecoderLayer) already make about themselves,
//! made once about the whole model.
//!
//! # The cache grows and is never trimmed
//!
//! mlx-vlm gives **every** layer a plain `KVCache`, sliding layers included, and
//! enforces the 512-token window entirely through the mask. So does this. The
//! keys of a sliding layer accumulate for the whole sequence and all but the
//! last 512 of them are masked out on every step, which is waste rather than
//! error.
//!
//! Trimming them is the optimisation that makes a 1M-token context affordable —
//! 35 of 42 layers would stop growing, and the README's 30 GiB figure assumes
//! it. It is deliberately not here: it is a departure from the reference rather
//! than a port of it, and it needs its own equivalence argument, against the one
//! thing that cannot be trimmed alongside it. The short convolution's state is
//! the last `K-1` inputs and holds no positions at all, so a rotated KV cache
//! and a convolution window have to agree about what "already seen" means.
//!
//! # Greedy, and only greedy
//!
//! [`greedy`] takes the argmax and there is no sampler here, which is a claim
//! about what can be validated rather than a limitation of the loop. What
//! survives forty-two layers of accumulated bfloat16 is the argmax: `B3` measured
//! the tail reordering at six of eight positions on distinct values once the
//! model is fed its own hidden state, so a temperature or a nucleus cut over
//! this port's logits would draw from an order that is legitimately not the
//! reference's. A sampler belongs in a later commit with its own honest
//! statement about where it diverges — not smuggled in beside a loop that can be
//! checked token for token.
//!
//! # A token at a time, to whoever asked for it
//!
//! [`Generator::stream`] hands each id to a sink as it is decided rather than
//! returning them all at the end, and that is a decision the price of a step
//! forces: a decode step costs 9.0 s on the CPU path, so a caller that waits for
//! the last one waits minutes for the first. Nothing here writes anything —
//! whether a token becomes text, a line on a terminal or a chunk on a socket is
//! the sink's, and [`crate::detokenize`] is what makes a token's worth of text
//! well-defined at all.
//!
//! Three things end a generation, and [`Stop`] says which: the end-of-sequence
//! id, the budget, or the sink declining the next one. The last exists because a
//! sink can fail — a closed pipe, a client that hung up — and discovering that
//! only after the budget runs out is a budget of arithmetic nobody will read.

use std::ops::ControlFlow;

use crate::head::{LmHead, Tail, Tailed};
use crate::layer::Passed;
use crate::model::{Batched, Model, ModelCache, ModelWeights};
use crate::ops::{Projection, top_k};
use crate::profile::{self, Op};

/// The id a greedy sampler takes.
///
/// [`top_k`] rather than a fresh argmax, so that the rule for a tie — the lower
/// id — is the one every ranking in the tree already uses, and a position whose
/// top two logits bfloat16 cannot tell apart resolves the same way here as it
/// does in a comparison against the oracle.
pub fn greedy(logits: &[f32]) -> usize {
    let _timed = profile::scope(Op::Sample);
    *top_k(logits, 1).first().expect("logits to sample from")
}

/// What ends a generation, as the caller states it beforehand.
///
/// The end-of-sequence id is an `Option` because it is the *config's* answer and
/// not the tokenizer's — see [`Tokenizer`](crate::tokenizer::Tokenizer), whose
/// files name none — so a caller driving a checkpoint that declares one has it
/// and a synthetic stack has nothing to put here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ending {
    /// How many tokens may be decoded at most.
    pub budget: usize,
    /// The id that ends the generation as soon as it arrives.
    pub eos: Option<usize>,
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The end-of-sequence id arrived.
    ///
    /// It reaches the sink like any other token. The vocabulary spells it, a
    /// detokenizer renders it, and a caller that wanted it dropped can drop the
    /// one id it named — where a loop that swallowed it would leave nothing able
    /// to tell an ended message from a truncated one by reading the stream.
    EndOfSequence,
    /// The budget ran out with the model still going.
    Budget,
    /// The sink stopped taking tokens.
    Sink,
}

/// Where a round's guesses come from.
///
/// **A guess is never load-bearing**, and that is the whole of what this trait
/// promises the loop above: whatever it answers, the model is asked what
/// follows each of those tokens and only the prefix it agrees with is kept. So
/// a proposer that is wrong costs a round its speedup and cannot cost it a
/// token — which is what lets [`crate::mtp`]'s heads guess from their own
/// guesses, and what lets the tests here drive this loop with a proposer that
/// is deliberately, systematically wrong.
///
/// It is `&mut self` because a proposer has state of its own: the MTP heads
/// carry a cache apiece, chained across rounds, and a round is where they are
/// advanced.
pub trait Proposer {
    /// How many tokens this can guess at most, which the loop takes as the
    /// ceiling on a round's block.
    fn depth(&self) -> usize;

    /// Up to `round.depth` tokens after the last row of `round`.
    ///
    /// Fewer is allowed and none is allowed: a round of one token is an
    /// ordinary decode step, which is what a proposer that has nothing to say
    /// leaves behind.
    fn propose(&mut self, round: Round<'_>) -> &[usize];
}

/// Where a batch's rounds come from.
///
/// **The mirror of [`Proposer`] one seat out, and a trait of its own rather than
/// a default on that one**, for the reason a batch is not a loop over one
/// sequence: a proposer carries state per sequence — the MTP heads carry a chain
/// of caches apiece — so an implementation asked for N rounds has to know they
/// are N sequences' and not one sequence's N times. A default that walked the
/// rounds through [`Proposer::propose`] would give every sequence in the batch
/// the state of the one before it, silently.
///
/// A guess is never load-bearing here either: [`Generator::speculate_batch`]
/// verifies every one of them against the model's own answer, per sequence.
pub trait BatchProposer {
    /// How many tokens this can guess at most for any one sequence, which the
    /// loop takes as the ceiling on a round's block.
    fn depth(&self) -> usize;

    /// Up to `round.depth` tokens after the last row of each round.
    ///
    /// One entry per round handed in, in the order they were handed in — which
    /// is the sequences' slot order, since a batch's rows are laid that way.
    fn propose_batch(&mut self, rounds: &[Seated<'_>]) -> Vec<Vec<usize>>;
}

/// One sequence's round inside a batch of them.
///
/// `at` is which of the proposer's sequences it belongs to, because a batch's
/// live members are a subset: a sequence that has produced everything it was
/// asked for drops out and the rest go on without it, so the `n`th round of a
/// call is not the `n`th sequence.
pub struct Seated<'a> {
    pub at: usize,
    pub round: Round<'a>,
}

impl BatchProposer for Alone {
    fn depth(&self) -> usize {
        0
    }

    fn propose_batch(&mut self, rounds: &[Seated<'_>]) -> Vec<Vec<usize>> {
        rounds.iter().map(|_| Vec::new()).collect()
    }
}

/// What a round hands its proposer: the rows it committed, and what follows
/// each of them.
///
/// **Every row, not only the last.** A proposer that carries state — the heads
/// do — has to see the positions the round accepted, or its own history skips
/// the tokens speculation bought and stops being the history the model has. The
/// rows a rejection took back are not here, because they are not the sequence.
#[derive(Debug, Clone, Copy)]
pub struct Round<'a> {
    /// `[rows, hidden]`, the **post-final-norm** hidden state of each row the
    /// round committed — which is what an MTP head is chained from, and is
    /// already normed here because that is the tensor the model's own logits
    /// come out of.
    ///
    /// Empty for a proposer of no depth. Nothing but a proposer reads this, so
    /// a round that will not be asked for guesses does not ask for it either —
    /// see [`Tail::chained`], where on a device that is a dispatch and a
    /// crossing rather than a slice.
    pub hidden: &'a [f32],
    /// The token that follows each of those rows. The last is the one the model
    /// has just produced and the loop has yet to feed, so a proposer guessing
    /// what comes after it is guessing past the end of the sequence rather than
    /// alongside it.
    pub next: &'a [usize],
    /// How many tokens this round can use, which is the proposer's own depth
    /// capped by the budget the generation has left. Zero is a generation about
    /// to end.
    pub depth: usize,
}

/// What one sequence of a batch was answered: the row of the call its own rows
/// started at, and the id the model produced for each of them.
///
/// **The ids of every row and not of the last**, for the reason
/// [`Generator::step_batch`] runs the tail over the whole call: a sequence
/// feeding more than one row is a speculative block, and every row of a block
/// is a question. A sequence feeding one row has one id, which is a decode step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picked {
    /// The row of the call this sequence's rows started at, which is what a
    /// caller reading anything else the call produced indexes by.
    pub first: usize,
    pub picks: Vec<usize>,
}

impl Picked {
    /// The id the model produced for this sequence's last row, which for a
    /// sequence feeding one row is the token a decode step decided.
    pub fn last(&self) -> usize {
        *self.picks.last().expect("a row per sequence of a batch")
    }
}

/// What a batched step answered: each sequence's ids, and — where the caller
/// asked for them — the normed rows a proposer is chained from.
///
/// The rows are the whole call's, laid sequence by sequence in the order the
/// batch names them, so [`Picked::first`] is what indexes into them.
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    pub picked: Vec<Picked>,
    /// `[rows, hidden]` post-final-norm, and empty where nothing asked.
    pub normed: Vec<f32>,
}

/// Where each sequence of a batched speculative run is: what it has produced,
/// the token it owes the next call, and what its chain guessed after that.
///
/// One struct rather than three parallel vectors passed around, because they
/// are indexed by the same thing and a caller holding two of them in step by
/// hand is a caller that can stop.
struct Running {
    produced: Vec<Vec<usize>>,
    pending: Vec<usize>,
    guesses: Vec<Vec<usize>>,
}

/// The proposer of a generation that does not speculate, which guesses nothing
/// and is what [`Generator::stream`] runs.
#[derive(Debug, Clone, Copy)]
pub struct Alone;

impl Proposer for Alone {
    fn depth(&self) -> usize {
        0
    }

    fn propose(&mut self, _: Round<'_>) -> &[usize] {
        &[]
    }
}

/// The model and its head, as the thing that turns ids into ids.
///
/// `LanguageModel` is the two of them together — the stack, its final norm, and
/// `lm_head` — and generation is the first caller that needs all of it at once.
#[derive(Clone, Copy)]
pub struct Generator<'a> {
    model: Model<'a>,
    head: LmHead,
    head_weights: &'a dyn Projection,
}

impl<'a> Generator<'a> {
    /// `head_weights` is `lm_head` itself, as the projection the head
    /// multiplies through — 200058 rows of 4096, held by whoever knows how they
    /// are stored.
    ///
    /// Taken once, here, rather than per call, and taken as a
    /// [`Projection`] rather than as weights: which backend runs the largest
    /// operation in a decode step is settled when a generator is built, and
    /// nothing below this line — not [`Generator::stream`], not a sink, not a
    /// server — is told which it was.
    pub fn new(model: Model<'a>, head: LmHead, head_weights: &'a dyn Projection) -> Self {
        Self {
            model,
            head,
            head_weights,
        }
    }

    pub fn model(&self) -> Model<'a> {
        self.model
    }

    pub fn head(&self) -> LmHead {
        self.head
    }

    /// The `[vocab]` logits of the **last** of `ids`, leaving every one of them
    /// behind in `cache`.
    ///
    /// The last alone because it is the only one that decides anything: at a
    /// prompt's end it is the next token, and at a decode step it is all there
    /// is. Every earlier position's logits would be a full pass over the head —
    /// 201024 rows — to answer a question nothing asks.
    ///
    /// The norm is applied to that one row rather than to the whole state, which
    /// is the same value: RMSNorm divides a row by its own RMS and reads no
    /// other.
    pub fn logits(
        &self,
        cache: &mut ModelCache,
        ids: &[usize],
        weights: &impl ModelWeights,
    ) -> Vec<f32> {
        self.tailed(
            cache,
            ids,
            Tail {
                block: 1,
                chained: false,
                logits: true,
            },
            weights,
        )
        .logits
    }

    /// The stack over `ids`, and the back of the model behind it: the final
    /// norm, the muP divide and `lm_head` over the last `want.block` rows.
    ///
    /// **One call rather than three, because the three may be one command
    /// buffer.** A backend that ran the last layer can run the norm and the
    /// projection where that layer's rows already are — see
    /// [`ModelWeights::tail`] — so what crosses back is what a token is taken
    /// from, and asking the stack and then the head would be the round trip in
    /// between. A backend that answered with rows is answered here, in the
    /// three lines the reference writes.
    pub fn tailed(
        &self,
        cache: &mut ModelCache,
        ids: &[usize],
        want: Tail,
        weights: &impl ModelWeights,
    ) -> Tailed {
        assert!(!ids.is_empty(), "a forward pass over no tokens");
        assert!(want.block <= ids.len(), "a block longer than the pass");

        match self.model.forward(cache, ids, weights) {
            Passed::Carried(rows) => weights
                .tail(rows, want)
                .expect("a stack that carried its last layer's rows holds the tail behind them"),
            Passed::Rows(h) => self.on_this_side(&h, want),
        }
    }

    /// The same tail, over rows this side is holding.
    ///
    /// The whole state is normed rather than the block alone, because the norm
    /// is what a proposer is chained from and a row costs a pass over its own
    /// 4096 values. What the head is then given is the block, which is where
    /// the 200058-row projection is worth not running.
    ///
    /// **The logits are formed whatever [`Tail::logits`] says**, because the ids
    /// come out of them and there is nowhere else on this side for them to come
    /// from. What the flag decides here is whether they are handed on, which is
    /// a `Vec` moved against a `Vec` dropped — where on a device it is 800 KB a
    /// row not crossing a seam.
    fn on_this_side(&self, h: &[f32], want: Tail) -> Tailed {
        let hidden = self.model.hidden();
        let normed = self.model.final_norm(h);
        let block = &normed[normed.len() - want.block * hidden..];
        let logits = self.head.forward(block, self.head_weights);
        Tailed {
            normed: match want.chained {
                true => normed,
                false => Vec::new(),
            },
            picks: Self::picks(&logits, want.block),
            logits: match want.logits {
                true => logits,
                false => Vec::new(),
            },
        }
    }

    /// The whole stack and the back of the model over a batch of sequences
    /// advancing together, and the id each of them produced.
    ///
    /// **The one call in this file whose subject is more than one request.**
    /// Every other measure of a decode step is what it costs to produce one
    /// token, and the weights it reads to produce that one — 5.9 GB of them —
    /// are the same weights every other sequence in flight would read. So N
    /// sequences fed together are one set of those reads and N tokens, and
    /// nothing in a kernel had to change for it: what changed is that a
    /// sequence's span and its four convolution windows are a slot's rather
    /// than the layer's.
    ///
    /// **Every row of the call is asked, not the last of each sequence.** A
    /// sequence feeding more than one row is a speculative block, and every row
    /// of a block is a question the round has to have the answer to — so the
    /// tail runs over the whole call and each sequence is handed back its own
    /// rows' answers. That is also why this does not batch a *prefill*: a
    /// prompt's every position would be a pass over 200058 rows of the head to
    /// answer a question nothing asks, and a prompt already fills the machine
    /// on its own.
    ///
    /// The ids each sequence fed are left in its own cache, as a step of its
    /// own would leave them.
    pub fn step_batch(
        &self,
        batch: &mut [Batched<'_>],
        weights: &impl ModelWeights,
    ) -> Vec<Picked> {
        self.verify_batch(batch, false, weights).picked
    }

    /// The same batched step with the rows a proposer is chained from carried
    /// back beside the ids.
    ///
    /// **The one call a batched speculative round makes.** Every row of the
    /// call is a question — a sequence's rows are the token it owes plus the
    /// guesses a chain made after it — and what the round then needs of the
    /// rows it keeps is the *post*-final-norm hidden state each of them
    /// produced, because that is what an MTP head is chained from. Asking for
    /// the two separately would be the pass run twice.
    ///
    /// `chained` is what decides whether those rows cross back at all, for the
    /// reason [`Tail::chained`](crate::Tail::chained) exists: a batch that will
    /// not be asked for guesses does not pay a dispatch and a crossing to carry
    /// them.
    pub fn verify_batch(
        &self,
        batch: &mut [Batched<'_>],
        chained: bool,
        weights: &impl ModelWeights,
    ) -> Verified {
        let counts: Vec<usize> = batch.iter().map(|sequence| sequence.ids.len()).collect();
        let rows: usize = counts.iter().sum();
        let tailed = self.asked_of(
            batch,
            Tail {
                block: rows,
                chained,
                logits: false,
            },
            weights,
        );

        let mut from = 0;
        let picked = counts
            .into_iter()
            .map(|queries| {
                let picked = Picked {
                    first: from,
                    picks: tailed.picks[from..from + queries].to_vec(),
                };
                from += queries;
                picked
            })
            .collect();
        Verified {
            picked,
            normed: tailed.normed,
        }
    }

    /// The same step over a batch whose **leading seats are feeding a prompt
    /// and whose trailing `decoding` seats are decoding a row apiece**, and the
    /// id each of those decoders produced.
    ///
    /// **This is what a slot filling while its neighbours decode looks like from
    /// here**, and it is one call rather than two: a request joining a running
    /// batch feeds its prompt in the same forward pass the sequences already in
    /// flight take their next token out of, so the weights that pass reads are
    /// read once for the joiner and the decoders together. That is the seat path
    /// `B3` built and nothing reached — a seat is a *stride* of the grid rather
    /// than a run of it precisely so that one call can carry rows of unequal
    /// counts.
    ///
    /// **The order of the seats is the interface and it is asserted.** A tail is
    /// asked for the last `block` rows of a call — see [`Tail::block`] — so the
    /// rows that are questions have to be the last of them. A joining prompt's
    /// rows are not questions: every position but its last would be a pass over
    /// 200058 rows of the head to answer what nothing asks. So the fillers go in
    /// front, the decoders behind, and the block is exactly the decoders.
    ///
    /// **Which is why a prompt is filled to its last token and no further.**
    /// The token that follows a prompt is decided by the prompt's last row, and
    /// that row is a question — so a joining sequence feeds `prompt[..n-1]`
    /// here and then feeds `prompt[n-1]` as its first decode row, which is the
    /// split `prefilling_then_decoding_matches_one_prefill_over_the_whole_sequence`
    /// says produces the same logits as one pass over all of it.
    pub fn step_admitting(
        &self,
        batch: &mut [Batched<'_>],
        decoding: usize,
        weights: &impl ModelWeights,
    ) -> Vec<usize> {
        assert!(
            decoding <= batch.len(),
            "{decoding} decoding seats in a batch of {}",
            batch.len()
        );
        let filling = batch.len() - decoding;
        assert!(
            batch[..filling].iter().all(|seat| !seat.ids.is_empty()),
            "a seat of the batch feeding no rows"
        );
        assert!(
            batch[filling..].iter().all(|seat| seat.ids.len() == 1),
            "a decoding seat feeding more than the token it owes"
        );

        // **A step with nothing to ask still asks for one row**, which is a
        // batch every seat of which is filling: a backend that carried the last
        // layer's rows has a run still open and the tail is what closes it, so
        // the smallest tail there is is the one [`Generator::prefill`] takes and
        // for the same reason. The id it names is a mid-prompt position's and is
        // dropped here.
        let picks = self
            .asked_of(
                batch,
                Tail {
                    block: decoding.max(1),
                    chained: false,
                    logits: false,
                },
                weights,
            )
            .picks;
        picks[picks.len() - decoding..].to_vec()
    }

    /// The stack and the tail over a batch, and the ids of the **last `block`
    /// rows** of the call.
    ///
    /// The one place the two batched steps share, because what differs between
    /// them is which rows are questions and nothing else: a batch every row of
    /// which is a speculative block's asks for all of them, and a batch that is
    /// admitting a sequence asks for the decoders behind the prompt being fed.
    fn asked_of(
        &self,
        batch: &mut [Batched<'_>],
        want: Tail,
        weights: &impl ModelWeights,
    ) -> Tailed {
        let tailed = match self.model.forward_batch(batch, weights) {
            Passed::Carried(carried) => weights
                .tail(carried, want)
                .expect("a stack that carried its last layer's rows holds the tail behind them"),
            Passed::Rows(h) => self.on_this_side(&h, want),
        };
        assert_eq!(tailed.picks.len(), want.block, "an id per row of the block");
        tailed
    }

    /// `ids` into `cache`, with no token taken back out of it.
    ///
    /// **What [`Generator::stream`] does not leave room for**: a prompt whose
    /// keys are wanted without the token that follows them. A server keeping a
    /// conversation across requests wants exactly that — the cache at the end of
    /// the prompt, marked there, and the generation after the mark — and a
    /// stream that prefills and samples in one pass has no seam to mark at.
    ///
    /// The block of one is the seam **and it is not free of purpose**: a backend
    /// that carried the last layer's rows has a run still open, and the tail is
    /// what closes it. Marking a window a dispatch has not written is reading
    /// whatever was there before. So this asks for the smallest tail there is —
    /// one row, no logits handed back, nothing chained — and drops the id it
    /// names.
    ///
    /// **Smallest is not free, and it is worth saying what it costs.**
    /// [`Tail::logits`] decides whether the logits cross back, not whether they
    /// are formed: a backend holding the tail runs the norm, the muP divide and
    /// the 200058-row projection either way. So a prompt split into a prefill
    /// and a generation pays one more of those than a prompt fed whole — about
    /// a decode step's head, once a turn, against a turn that is seconds.
    pub fn prefill(&self, cache: &mut ModelCache, ids: &[usize], weights: &impl ModelWeights) {
        self.tailed(
            cache,
            ids,
            Tail {
                block: 1,
                chained: false,
                logits: false,
            },
            weights,
        );
    }

    /// The ids a greedy sampler takes from `rows` rows of logits.
    ///
    /// Every row of a speculative block is a question — "what does the model
    /// say follows what was fed up to here" — and the answers are what the
    /// block is checked against. The rows in front of it are not: at a prefill
    /// they are the prompt, whose every position would be a pass over 200058
    /// rows of the head to answer a question nothing asks, and whose logits at
    /// a 769-token prompt would be 615 MB. So what a block asks for is
    /// [`Tail::block`] and never the pass.
    pub fn picks(logits: &[f32], rows: usize) -> Vec<usize> {
        logits
            .chunks_exact(logits.len() / rows)
            .map(greedy)
            .collect()
    }

    /// The id a greedy sampler takes from a hidden state the stack has already
    /// produced, which is `speculative_argmax_from_hidden` in the reference.
    ///
    /// The state is the *pre*-final-norm one — what a layer of the stack
    /// answers with, and what an MTP head answers with — so the norm is applied
    /// here. What it is for is a head's guess: a head produces a hidden state
    /// and the only way to learn which token that is is to put it through the
    /// same final norm and the same 200058-row projection the model's own
    /// answer goes through.
    ///
    /// This is the tail run *here*, on rows that have already crossed back. A
    /// head whose backend holds the tail never reaches it — the guess comes
    /// back beside the head's own rows, out of the head's own command buffer.
    pub fn id_from_hidden(&self, hidden: &[f32]) -> usize {
        let normed = self.model.final_norm(hidden);
        greedy(&self.head.forward(&normed, self.head_weights))
    }

    /// `prompt` prefilled, then tokens decoded greedily until `ending` says to
    /// stop, each handed to `sink` as it is decided and then fed back through
    /// the caches the step before it left behind.
    ///
    /// This is [`Generator::speculate`] against a proposer that guesses
    /// nothing, and it is *the same loop* rather than a simpler one beside it —
    /// which is what makes "speculation changes no token" a claim about the
    /// proposer rather than about two implementations of a generation.
    ///
    /// The prompt is prefilled by the first step rather than by a step of its
    /// own, so that step's cost is a prefill's and every later one's is a
    /// decode's. Against this checkpoint those are 0.99 s and 0.055 s on the
    /// device path and 9.0 s a decode step on the CPU's — the two
    /// regimes are worth telling apart in anything that reports timings, and a
    /// mean over the steps of one call describes neither.
    ///
    /// The last token generated is never fed back, so its keys are not in
    /// `cache` when this returns. That is what makes stopping on an
    /// end-of-sequence id leave a cache holding the message and not its
    /// terminator, and what makes generating `n` then `m` more the same sequence
    /// as generating `n + m`.
    ///
    /// The sink is offered every token, the end-of-sequence id included, and is
    /// asked before that id is checked for — so a sink that declines the token
    /// which is *also* the eos ends the generation as [`Stop::Sink`]. A sink
    /// that could not take a token did not take that one either, and reporting
    /// a clean end for a message the caller never received would be a worse
    /// answer than reporting the refusal.
    ///
    /// A budget of zero prefills nothing and leaves `cache` untouched: the
    /// prompt enters it on the first step, because a prompt whose next token is
    /// never asked for is a forward pass with no answer.
    pub fn stream(
        &self,
        cache: &mut ModelCache,
        prompt: &[usize],
        ending: Ending,
        weights: &impl ModelWeights,
        sink: impl FnMut(usize) -> ControlFlow<()>,
    ) -> Stop {
        self.speculate(cache, prompt, ending, weights, &mut Alone, sink)
    }

    /// The same generation, with `proposer` allowed to guess ahead.
    ///
    /// # A round, and why it is the same tokens
    ///
    /// A round feeds the model the token it owes plus the `k` a proposer
    /// guessed after it, and reads the argmax at every one of those `k + 1`
    /// positions. Position `i` of that block answers *what follows everything
    /// fed up to and including row `i`* — which, for as long as the guesses
    /// were right, is exactly the question the next decode step would have
    /// asked. So the block's answers are checked against the guesses in order,
    /// the longest agreeing prefix is kept, and the first answer that disagreed
    /// is kept too: it is the model's own next token, arrived at from a prefix
    /// the model agrees with.
    ///
    /// **Nothing here trusts a guess.** A round emits the tokens the model
    /// produced, in the order it produces them, and a proposer that guessed
    /// nothing right still emits one — the same one a decode step would have.
    /// What a good guess buys is that several of them came out of one forward
    /// pass. That is the whole of the argument that this is a latency
    /// optimisation rather than an approximation, and
    /// `speculation_changes_no_token` is it stated as a test.
    ///
    /// # What the rejected tokens leave behind
    ///
    /// The block was *fed*: every layer's keys, and every one of its four
    /// convolution windows, moved for tokens that turned out to be wrong. They
    /// are taken back before the round ends — see
    /// [`ModelWeights::rewind`](crate::model::ModelWeights::rewind) — which is
    /// what makes the cache this leaves the cache a generation of the same
    /// tokens would have left, and what the caller has to have asked for by
    /// building a cache that keeps enough slack to give them back.
    ///
    /// A token that reaches the sink is never in the cache when this returns
    /// with it, which is the property [`Generator::stream`] already had: a
    /// generation that stops on an end-of-sequence id leaves a cache holding the
    /// message and not its terminator, whether or not the terminator was
    /// guessed.
    pub fn speculate(
        &self,
        cache: &mut ModelCache,
        prompt: &[usize],
        ending: Ending,
        weights: &impl ModelWeights,
        proposer: &mut impl Proposer,
        mut sink: impl FnMut(usize) -> ControlFlow<()>,
    ) -> Stop {
        let hidden = self.model.hidden();
        let mut ids = prompt.to_vec();
        let mut guesses: Vec<usize> = Vec::new();
        let mut left = ending.budget;

        while left > 0 {
            assert!(!ids.is_empty(), "a forward pass over no tokens");
            let (rows, block) = (ids.len(), guesses.len() + 1);
            let base = rows - block;

            let tailed = self.tailed(
                cache,
                &ids,
                Tail {
                    block,
                    chained: proposer.depth() > 0,
                    logits: false,
                },
                weights,
            );
            let picks = &tailed.picks;
            assert_eq!(picks.len(), block, "an id per row of the block");
            let agreed = guesses
                .iter()
                .zip(picks)
                .take_while(|(guess, pick)| guess == pick)
                .count();

            // What the round produced: the guesses the model agreed with, and
            // its own answer to the first position it did not.
            let mut produced = guesses[..agreed].to_vec();
            produced.push(picks[agreed]);

            let mut stop = None;
            let mut kept = base + 1;
            for (at, id) in produced.iter().enumerate() {
                kept = base + 1 + at;
                left -= 1;
                stop = match () {
                    _ if sink(*id).is_break() => Some(Stop::Sink),
                    _ if Some(*id) == ending.eos => Some(Stop::EndOfSequence),
                    _ if left == 0 => Some(Stop::Budget),
                    _ => None,
                };
                if stop.is_some() {
                    break;
                }
            }
            weights.rewind(cache, rows - kept);
            if let Some(stop) = stop {
                return stop;
            }

            // The rows the round kept, and the token that follows each of them
            // — which for the last is the one the model has just produced and
            // this loop has yet to feed.
            let pending = produced[produced.len() - 1];
            let mut next = ids[1..kept].to_vec();
            next.push(pending);
            guesses = proposer
                .propose(Round {
                    hidden: match tailed.normed.is_empty() {
                        true => &[],
                        false => &tailed.normed[..kept * hidden],
                    },
                    next: &next,
                    depth: proposer.depth().min(left.saturating_sub(1)),
                })
                .to_vec();

            ids = std::iter::once(pending)
                .chain(guesses.iter().copied())
                .collect();
        }
        Stop::Budget
    }

    /// Several generations at once: each prompt prefilled on its own, then
    /// every sequence decoded together until it has the tokens `counts` asked
    /// it for.
    ///
    /// **The prompts are prefilled a sequence at a time and the tokens are
    /// decoded together**, which is where the two regimes differ: a prefill of
    /// any length already fills the machine, and a decode step's dispatches
    /// carry a thousand output elements against eighty cores. So what a batch
    /// is for is the second, and the first is left where it was.
    ///
    /// A sequence whose count runs out drops out of the batch and the rest go
    /// on without it, which is what the tail of any real batch looks like: the
    /// sequences do not finish together.
    ///
    /// `caches` is a cache a prompt, each built for its own slot of whatever
    /// backend runs the layers — see [`ModelCache::in_slot`].
    pub fn generate_batch(
        &self,
        caches: &mut [ModelCache],
        prompts: &[&[usize]],
        counts: &[usize],
        weights: &impl ModelWeights,
    ) -> Vec<Vec<usize>> {
        assert_eq!(caches.len(), prompts.len(), "a cache a prompt");
        assert_eq!(counts.len(), prompts.len(), "a budget a prompt");

        let want = Tail {
            block: 1,
            chained: false,
            logits: false,
        };
        let mut produced: Vec<Vec<usize>> = prompts.iter().map(|_| Vec::new()).collect();
        for (at, (cache, prompt)) in caches.iter_mut().zip(prompts).enumerate() {
            if counts[at] == 0 {
                continue;
            }
            let picks = self.tailed(cache, prompt, want, weights).picks;
            produced[at].push(*picks.first().expect("an id for the prompt's last row"));
        }

        loop {
            let live: Vec<usize> = (0..prompts.len())
                .filter(|at| produced[*at].len() < counts[*at])
                .collect();
            if live.is_empty() {
                return produced;
            }
            let feeding: Vec<[usize; 1]> = live
                .iter()
                .map(|at| [*produced[*at].last().expect("a token to feed back")])
                .collect();
            let mut batch: Vec<Batched<'_>> = caches
                .iter_mut()
                .enumerate()
                .filter(|(at, _)| live.contains(at))
                .zip(&feeding)
                .map(|((_, cache), ids)| Batched { cache, ids })
                .collect();
            for (at, picked) in live.iter().zip(self.step_batch(&mut batch, weights)) {
                produced[*at].push(picked.last());
            }
        }
    }

    /// The same several generations, with `proposer` allowed to guess ahead for
    /// every one of them at once.
    ///
    /// # The verify is one call and it is ragged
    ///
    /// Each live sequence feeds the token it owes plus the guesses its own
    /// chain made after it, and all of those rows go through the stack
    /// together — so the weights a round reads are read once for the whole
    /// batch, as a decode step's already are, and the block a round verifies is
    /// N sequences' blocks rather than one. **The rows are unequal by
    /// construction**: a round accepts as many guesses as the model agreed
    /// with, which is a different number per sequence, so the next call's seats
    /// differ in length and each sequence rewinds its own cache by its own
    /// amount.
    ///
    /// Everything a round does to one sequence is what
    /// [`Generator::speculate`] does to it alone — the longest agreeing prefix
    /// is kept, the first disagreeing answer is kept too, and the rows behind
    /// that are taken back out of the cache *and* out of whatever backend holds
    /// its span — and nothing about it is shared between two sequences of the
    /// batch. That is what makes "speculation at batch changes no token" a
    /// claim a caller can check by running the same sequence alone.
    ///
    /// `caches` is a cache a prompt, each built for its own slot — see
    /// [`ModelCache::in_slot`] — and each with slack enough for the depth, since
    /// a rejected token is taken back out of a window that had to have room for
    /// it.
    ///
    /// A sequence whose count runs out drops out of the batch and the rest go
    /// on without it, which is what the tail of any real batch looks like.
    pub fn speculate_batch(
        &self,
        caches: &mut [ModelCache],
        prompts: &[&[usize]],
        counts: &[usize],
        weights: &impl ModelWeights,
        proposer: &mut impl BatchProposer,
    ) -> Vec<Vec<usize>> {
        assert_eq!(caches.len(), prompts.len(), "a cache a prompt");
        assert_eq!(counts.len(), prompts.len(), "a budget a prompt");
        let hidden = self.model.hidden();
        let speculating = proposer.depth() > 0;

        let mut run = Running {
            produced: prompts.iter().map(|_| Vec::new()).collect(),
            pending: vec![0; prompts.len()],
            guesses: prompts.iter().map(|_| Vec::new()).collect(),
        };

        // **The prompts are prefilled a sequence at a time**, for
        // [`Generator::generate_batch`]'s reason: a prefill of any length
        // already fills the machine, and what a batch is for is the decode.
        // The round each prompt opens is the prompt itself — every position of
        // it, and the token that follows each — which is the round
        // [`Generator::speculate`] asks for out of its own first pass.
        let mut opened: Vec<(Vec<f32>, Vec<usize>)> = Vec::with_capacity(prompts.len());
        for (at, (cache, prompt)) in caches.iter_mut().zip(prompts).enumerate() {
            if counts[at] == 0 {
                opened.push(Default::default());
                continue;
            }
            let tailed = self.tailed(
                cache,
                prompt,
                Tail {
                    block: 1,
                    chained: speculating,
                    logits: false,
                },
                weights,
            );
            let first = *tailed
                .picks
                .first()
                .expect("an id for the prompt's last row");
            run.produced[at].push(first);
            run.pending[at] = first;
            let mut next = prompt[1..].to_vec();
            next.push(first);
            opened.push((tailed.normed, next));
        }
        let opening: Vec<(usize, &[f32], &[usize])> = opened
            .iter()
            .enumerate()
            .map(|(at, (normed, next))| (at, normed.as_slice(), next.as_slice()))
            .collect();
        self.asking(&opening, counts, &mut run, proposer);

        loop {
            let live: Vec<usize> = (0..prompts.len())
                .filter(|at| run.produced[*at].len() < counts[*at])
                .collect();
            if live.is_empty() {
                return run.produced;
            }

            // The rows each live sequence feeds: the token it owes, and the
            // guesses its own chain made after it.
            let feeding: Vec<Vec<usize>> = live
                .iter()
                .map(|at| {
                    std::iter::once(run.pending[*at])
                        .chain(run.guesses[*at].iter().copied())
                        .collect()
                })
                .collect();
            let verified = {
                let mut batch: Vec<Batched<'_>> = caches
                    .iter_mut()
                    .enumerate()
                    .filter(|(at, _)| live.contains(at))
                    .zip(&feeding)
                    .map(|((_, cache), ids)| Batched { cache, ids })
                    .collect();
                self.verify_batch(&mut batch, speculating, weights)
            };

            // What each sequence's round produced, and what its cache has to
            // give back — which is its own number and not the call's, because a
            // round accepts what the model agreed with for that sequence alone.
            let mut kept: Vec<(usize, Vec<usize>)> = Vec::with_capacity(live.len());
            for ((at, ids), picked) in live.iter().zip(&feeding).zip(&verified.picked) {
                let agreed = run.guesses[*at]
                    .iter()
                    .zip(&picked.picks)
                    .take_while(|(guess, pick)| guess == pick)
                    .count();
                let mut round = run.guesses[*at][..agreed].to_vec();
                round.push(picked.picks[agreed]);
                round.truncate(counts[*at] - run.produced[*at].len());

                let held = round.len();
                run.produced[*at].extend(&round);
                run.pending[*at] = *round.last().expect("a round commits a token");
                // The rows this sequence keeps, and the token that follows each
                // — which for the last is the one the model has just produced
                // and this loop has yet to feed.
                let mut next = ids[1..held].to_vec();
                next.push(run.pending[*at]);
                kept.push((held, next));
                weights.rewind(&mut caches[*at], ids.len() - held);
            }

            let rounds: Vec<(usize, &[f32], &[usize])> = live
                .iter()
                .zip(&verified.picked)
                .zip(&kept)
                .map(|((at, picked), (held, next))| {
                    // This sequence's own rows out of the call's, which is what
                    // `Picked::first` is for: a seat's rows begin where the
                    // seats in front of it end, and a round chained from the
                    // call's first rows would chain every sequence from
                    // whichever one came first.
                    let rows = match speculating {
                        true => {
                            &verified.normed[picked.first * hidden..(picked.first + held) * hidden]
                        }
                        false => &[][..],
                    };
                    (*at, rows, next.as_slice())
                })
                .collect();
            self.asking(&rounds, counts, &mut run, proposer);
        }
    }

    /// The rounds that still have budget to guess into, handed to the proposer
    /// in one call, and the guesses it answered with put where the next verify
    /// reads them.
    ///
    /// **One call and not one per sequence**, which is the whole of what this
    /// milestone is: a chain of heads over N sequences reads its eight weights
    /// once for all of them.
    ///
    /// A sequence with nothing left to produce is not asked at all — its round
    /// would be a chain run for guesses nobody will verify.
    fn asking(
        &self,
        rounds: &[(usize, &[f32], &[usize])],
        counts: &[usize],
        run: &mut Running,
        proposer: &mut impl BatchProposer,
    ) {
        for guesses in &mut run.guesses {
            guesses.clear();
        }
        let speculating = proposer.depth() > 0;
        let seated: Vec<Seated<'_>> = rounds
            .iter()
            .filter(|(at, _, _)| run.produced[*at].len() < counts[*at])
            .map(|(at, hidden, next)| Seated {
                at: *at,
                round: Round {
                    hidden: match speculating {
                        true => hidden,
                        false => &[],
                    },
                    next,
                    depth: proposer
                        .depth()
                        .min((counts[*at] - run.produced[*at].len()).saturating_sub(1)),
                },
            })
            .collect();
        if seated.is_empty() {
            return;
        }
        for (seated, proposed) in seated.iter().zip(proposer.propose_batch(&seated)) {
            run.guesses[seated.at] = proposed;
        }
    }

    /// [`stream`](Self::stream) collected: `count` tokens decoded greedily, with
    /// no id ending it early.
    pub fn generate(
        &self,
        cache: &mut ModelCache,
        prompt: &[usize],
        count: usize,
        weights: &impl ModelWeights,
    ) -> Vec<usize> {
        let mut generated = Vec::with_capacity(count);
        let ending = Ending {
            budget: count,
            eos: None,
        };
        self.stream(cache, prompt, ending, weights, |id| {
            generated.push(id);
            ControlFlow::Continue(())
        });
        generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::Stack;
    use crate::ops::DenseProjection;

    /// The synthetic stack carries no `lm_head` — `InklingModel` does not hold
    /// one — so the head here reads the embedding table, which is exactly what
    /// `tie_word_embeddings` decides and what
    /// [`head_module`](crate::weights) resolves for a tied checkpoint:
    /// `nn.Embedding.as_linear` is `h @ Wᵀ` over the same `[vocab, hidden]` rows
    /// the lookup returns.
    ///
    /// Which table the rows come from decides nothing here. What is under test
    /// is the loop, and the loop needs a head that ranks 48 ids rather than a
    /// particular one. Held whole rather than reached a row at a time, which 48
    /// rows of 32 allow and the checkpoint's 200058 do not.
    fn generator<'a>(stack: &'a Stack, head: &'a DenseProjection<'a>) -> Generator<'a> {
        Generator::new(stack.model(), LmHead::for_config(&stack.config), head)
    }

    fn logits(stack: &Stack, cache: &mut ModelCache, ids: &[usize]) -> Vec<f32> {
        let head = stack.head();
        generator(stack, &head).logits(cache, ids, stack)
    }

    fn generate(
        stack: &Stack,
        cache: &mut ModelCache,
        prompt: &[usize],
        count: usize,
    ) -> Vec<usize> {
        let head = stack.head();
        generator(stack, &head).generate(cache, prompt, count, stack)
    }

    /// What a caller streaming a generation sees: the ids that reached the sink,
    /// and why the loop ended.
    ///
    /// `take` is how many the sink accepts before declining the next, which is
    /// the one of the three endings the model itself has no say in.
    fn stream(
        stack: &Stack,
        cache: &mut ModelCache,
        prompt: &[usize],
        ending: Ending,
        take: usize,
    ) -> (Vec<usize>, Stop) {
        let head = stack.head();
        let mut streamed = Vec::new();
        let stop = generator(stack, &head).stream(cache, prompt, ending, stack, |id| {
            streamed.push(id);
            match streamed.len() < take {
                true => ControlFlow::Continue(()),
                false => ControlFlow::Break(()),
            }
        });
        (streamed, stop)
    }

    /// What the loop generates when nothing but the budget ends it, which is
    /// what every case below states its own ending against.
    fn baseline(stack: &Stack) -> Vec<usize> {
        generate(
            stack,
            &mut ModelCache::new(&stack.config),
            &stack.ids,
            COUNT,
        )
    }

    /// A sink that takes everything the budget allows.
    fn streamed(stack: &Stack, ending: Ending) -> (Vec<usize>, Stop) {
        let cache = &mut ModelCache::new(&stack.config);
        stream(stack, cache, &stack.ids, ending, usize::MAX)
    }

    /// One prefill over the whole sequence, which is what every split below has
    /// to reproduce.
    fn whole(stack: &Stack, ids: &[usize]) -> Vec<f32> {
        logits(stack, &mut ModelCache::new(&stack.config), ids)
    }

    /// The invariant the engine rests on, at every split point the fixture's
    /// nine ids allow: prefill `k` of them, decode the rest one at a time, and
    /// arrive at the same logits as one pass over all nine.
    ///
    /// Exact equality rather than a tolerance, for the reason the short
    /// convolution's own split test demands it: both paths multiply the same
    /// numbers in the same order, and the only thing a split changes is where
    /// they come from. A tolerance here would absorb precisely the cache
    /// mistakes this exists to catch — a query offset off by one moves a mask
    /// entry by one distance, and a trained band is smooth enough that the
    /// answer would only drift.
    ///
    /// `k = 1` is the extreme worth naming: eight consecutive decode steps, each
    /// reading what all the ones before it left, which is a generation loop with
    /// its sampler replaced by the recorded ids.
    #[test]
    fn prefilling_then_decoding_matches_one_prefill_over_the_whole_sequence() {
        let stack = Stack::load();
        let sequence = stack.sequence();
        let want = whole(&stack, &sequence);

        for split in 1..sequence.len() {
            let cache = &mut ModelCache::new(&stack.config);
            let mut got = logits(&stack, cache, &sequence[..split]);
            for id in &sequence[split..] {
                got = logits(&stack, cache, &[*id]);
            }
            assert_eq!(got, want, "prefilled {split} of {}", sequence.len());
        }
    }

    /// The window a decode step attends over has to be the whole cache, not the
    /// step's own token. A model handed a fresh cache per step still generates —
    /// every token sees itself and nothing else — so this is the mistake that a
    /// self-consistent loop would hide.
    #[test]
    fn decoding_from_a_fresh_cache_each_step_changes_the_answer() {
        let stack = Stack::load();
        let sequence = stack.sequence();

        let mut got = Vec::new();
        for id in &sequence {
            got = logits(&stack, &mut ModelCache::new(&stack.config), &[*id]);
        }
        assert_ne!(got, whole(&stack, &sequence));
    }

    /// What generation is, stated against the prefill it has to agree with: the
    /// tokens the loop produced, fed back to a *fresh* model as if they had been
    /// the prompt all along, predict the next one it produced.
    ///
    /// This is the invariant above with the sampler in the loop rather than the
    /// recorded ids, which is what a generator actually does — and it is the
    /// claim that fails if `generate` forgot to feed a token back, sampled from
    /// the wrong position, or advanced the cache twice.
    #[test]
    fn each_generated_token_is_what_a_prefill_of_the_ones_before_it_predicts() {
        let stack = Stack::load();
        let prompt = &stack.ids;
        let generated = generate(&stack, &mut ModelCache::new(&stack.config), prompt, COUNT);
        assert_eq!(generated.len(), COUNT);

        for step in 0..generated.len() {
            let so_far = [prompt.clone(), generated[..step].to_vec()].concat();
            assert_eq!(
                greedy(&whole(&stack, &so_far)),
                generated[step],
                "token {step} of {generated:?}"
            );
        }
    }

    /// How many tokens the generation cases decode. Enough that a step reads
    /// what more than one step before it left — a short convolution's window is
    /// three inputs deep — and few enough that the whole-sequence prefill each
    /// one is checked against stays a test.
    const COUNT: usize = 4;

    /// Generation resumes: `n` tokens and then `m` more, against one cache, is
    /// the sequence `n + m` would have been. This is what says the loop leaves a
    /// cache a caller can carry, which is the whole reason a decode step is
    /// cheaper than a prefill.
    #[test]
    fn generating_in_two_calls_matches_generating_in_one() {
        let stack = Stack::load();
        let at_once = baseline(&stack);

        let cache = &mut ModelCache::new(&stack.config);
        let mut split = generate(&stack, cache, &stack.ids, 1);
        split.extend(generate(&stack, cache, &split, COUNT - 1));
        assert_eq!(split, at_once);
    }

    /// The loop samples from the last position of what it was handed. A
    /// generator that read the first instead would still generate, and on a
    /// one-token decode step the two are the same — so only the prompt can
    /// settle it.
    #[test]
    fn the_prompt_is_sampled_from_its_last_position() {
        let stack = Stack::load();
        let prompt = &stack.ids;
        let first = generate(&stack, &mut ModelCache::new(&stack.config), prompt, 1);

        assert_eq!(first, vec![greedy(&whole(&stack, prompt))]);
        assert_ne!(
            first,
            vec![greedy(&whole(&stack, &prompt[..1]))],
            "the first and last positions of the prompt agree, so this settles nothing"
        );
    }

    /// Every id is a real one. The synthetic config states no unpadded
    /// vocabulary, so the head is the whole table here; against the checkpoint
    /// the cut is what keeps the 966 padding rows — all-zero, and so all logits
    /// of exactly 0.0 — out of the ranking a sampler reads.
    #[test]
    fn every_generated_id_is_in_the_vocabulary() {
        let stack = Stack::load();
        let head = LmHead::for_config(&stack.config);
        let generated = baseline(&stack);
        assert!(
            generated.iter().all(|id| *id < head.vocab()),
            "{generated:?}"
        );
    }

    /// Streaming and collecting are the same generation. Everything below states
    /// what a sink can see that a returned vector cannot; this is what says the
    /// two are the same loop.
    #[test]
    fn streaming_surfaces_the_tokens_generating_returns() {
        let stack = Stack::load();
        let ending = Ending {
            budget: COUNT,
            eos: None,
        };
        let (streamed, stop) = streamed(&stack, ending);

        assert_eq!(
            streamed,
            generate(
                &stack,
                &mut ModelCache::new(&stack.config),
                &stack.ids,
                COUNT
            )
        );
        assert_eq!(stop, Stop::Budget);
    }

    /// How many tokens a generation that ends on `id` produces. The synthetic
    /// stack repeats itself, so the id a case names may arrive earlier than the
    /// position it was taken from, and every eos case below has to say where it
    /// first arrives rather than assume.
    fn ends_after(generated: &[usize], id: usize) -> usize {
        generated
            .iter()
            .position(|got| *got == id)
            .expect("an id this generation reaches")
            + 1
    }

    /// The end-of-sequence id ends the generation and reaches the sink, which is
    /// what lets a caller that renders special tokens render this one.
    #[test]
    fn a_generation_ends_on_the_end_of_sequence_id_and_surfaces_it() {
        let stack = Stack::load();
        let generated = baseline(&stack);
        let eos = generated[COUNT - 2];
        let ending = Ending {
            budget: COUNT,
            eos: Some(eos),
        };

        let (streamed, stop) = streamed(&stack, ending);
        assert_eq!(stop, Stop::EndOfSequence);
        assert_eq!(streamed, generated[..ends_after(&generated, eos)]);
        assert!(streamed.len() < COUNT, "the budget would have ended it");
    }

    /// Which of the two ends a generation that meets both at once. The model
    /// stopped, so a caller told it ran out of budget would resume a message the
    /// model considers finished.
    #[test]
    fn an_end_of_sequence_id_on_the_last_token_the_budget_allows_still_ends_it() {
        let stack = Stack::load();
        let generated = baseline(&stack);
        let eos = generated[COUNT - 1];
        let ending = Ending {
            budget: ends_after(&generated, eos),
            eos: Some(eos),
        };

        let (streamed, stop) = streamed(&stack, ending);
        assert_eq!(stop, Stop::EndOfSequence);
        assert_eq!(streamed, generated[..ending.budget]);
    }

    /// An id this generation never reaches leaves the budget to end it, which is
    /// the case a length cap exists for at all.
    #[test]
    fn an_end_of_sequence_id_the_generation_never_reaches_ends_nothing() {
        let stack = Stack::load();
        let generated = baseline(&stack);
        let unreachable = (0..)
            .find(|id| !generated.contains(id))
            .expect("an unused id");

        let (streamed, stop) = streamed(
            &stack,
            Ending {
                budget: COUNT,
                eos: Some(unreachable),
            },
        );
        assert_eq!(stop, Stop::Budget);
        assert_eq!(streamed, generated);
    }

    /// A sink that stops taking tokens stops the model too. A closed pipe
    /// discovered only when the budget runs out is a budget of arithmetic nobody
    /// will read — so what says this works is the count of tokens the sink was
    /// offered, not the ones it kept.
    #[test]
    fn a_sink_that_declines_a_token_ends_the_generation() {
        let stack = Stack::load();
        const TAKE: usize = 2;

        let (streamed, stop) = stream(
            &stack,
            &mut ModelCache::new(&stack.config),
            &stack.ids,
            Ending {
                budget: COUNT,
                eos: None,
            },
            TAKE,
        );
        assert_eq!(stop, Stop::Sink);
        assert_eq!(streamed.len(), TAKE, "the model ran past the sink");
    }

    /// The token a generation ended on is not fed back, so the cache it leaves
    /// holds the message and not its terminator. What that buys is a caller who
    /// can carry the cache on — a served conversation appends to it rather than
    /// prefilling the turn again.
    #[test]
    fn the_end_of_sequence_token_is_not_fed_back() {
        let stack = Stack::load();
        let generated = baseline(&stack);
        let eos = generated[COUNT - 2];

        let cache = &mut ModelCache::new(&stack.config);
        stream(
            &stack,
            cache,
            &stack.ids,
            Ending {
                budget: COUNT,
                eos: Some(eos),
            },
            usize::MAX,
        );

        let so_far = [
            stack.ids.clone(),
            generated[..ends_after(&generated, eos)].to_vec(),
        ]
        .concat();
        assert_eq!(logits(&stack, cache, &[eos]), whole(&stack, &so_far));
    }

    /// A budget of nothing is a prompt nobody asked a question about, so the
    /// pass that would have prefilled it never runs.
    ///
    /// Stated on `stream` alone, for both entry points: `generate` is this loop
    /// collected, and the case above says the two produce the same tokens.
    #[test]
    fn generating_nothing_leaves_the_cache_alone() {
        let stack = Stack::load();
        let cache = &mut ModelCache::new(&stack.config);
        let ending = Ending {
            budget: 0,
            eos: None,
        };

        let (streamed, stop) = stream(&stack, cache, &stack.ids, ending, usize::MAX);
        assert!(streamed.is_empty());
        assert_eq!(stop, Stop::Budget);
        assert_eq!(logits(&stack, cache, &stack.ids), whole(&stack, &stack.ids));
    }

    /// **The same generation, run alone and run inside a batch, produces
    /// identical tokens** — at every position of the batch, beside neighbours
    /// with prompts of different lengths, and beside a neighbour that finishes
    /// early.
    ///
    /// This is the loop's half of the claim; the kernels' half is
    /// `a_sequence_in_a_batch_produces_what_it_produces_alone` in
    /// `inkling_metal::projections`, where the state a batch splits actually
    /// lives. What this can fail on is the bookkeeping around it: a sequence
    /// handed another's token to feed back, a sequence dropped from the batch
    /// whose neighbours then read the wrong rows, or an id taken from the wrong
    /// row of the call.
    ///
    /// A batch is exactly the tokens, not close to them: every arm multiplies
    /// the same numbers in the same order.
    #[test]
    fn a_generation_in_a_batch_produces_what_it_produces_alone() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let sequence = stack.sequence();
        let prompts: [&[usize]; 3] = [&sequence[..3], &sequence[3..], &sequence[..1]];
        // The third finishes two tokens before the others, which is what a
        // batch's tail looks like: the sequences do not stop together.
        let counts = [COUNT, COUNT, COUNT - 2];

        let alone: Vec<Vec<usize>> = prompts
            .iter()
            .zip(&counts)
            .map(|(ids, count)| {
                generator.generate(&mut ModelCache::new(&stack.config), ids, *count, &stack)
            })
            .collect();
        assert_ne!(alone[0], alone[1], "two generations to tell apart");

        for order in [vec![0, 1, 2], vec![2, 1, 0], vec![1, 2], vec![2, 0]] {
            let mut caches: Vec<ModelCache> = (0..order.len())
                .map(|slot| ModelCache::in_slot(&stack.config, 0, slot))
                .collect();
            let ids: Vec<&[usize]> = order.iter().map(|at| prompts[*at]).collect();
            let budgets: Vec<usize> = order.iter().map(|at| counts[*at]).collect();
            let batched = generator.generate_batch(&mut caches, &ids, &budgets, &stack);

            for (at, seq) in order.iter().enumerate() {
                assert_eq!(
                    batched[at], alone[*seq],
                    "sequence {seq} at position {at} of {order:?}"
                );
            }
        }
    }

    /// A proposer that guesses from the continuation the model actually
    /// produces, and lies from a stated depth onwards.
    ///
    /// One proposer rather than three, because what the cases below vary is one
    /// number: `right` of 8 guesses everything correctly, 0 nothing, and 1 the
    /// first of each round and no more — which is the case that exercises a
    /// round accepting *some* of its block, where the two extremes exercise the
    /// ends. A liar is not a hypothetical here either: at 44.9% acceptance on
    /// prose, the second head of a real round is wrong more often than not.
    struct Guesser {
        /// The prompt and everything a generation of it produces, which is what
        /// a guess is measured against — and, shifted by one, what a wrong
        /// guess is drawn from.
        truth: Vec<usize>,
        /// How many of a round's guesses are the truth before the rest are not.
        right: usize,
        /// Tokens of `truth` the loop has fed, which is where this proposer's
        /// guesses start: the token after the one it was handed last.
        at: usize,
        guesses: Vec<usize>,
        /// How many rounds it has been asked for, which is what says
        /// speculation banked anything at all.
        rounds: usize,
        /// The rows every round handed this sequence to be chained from, which
        /// a real proposer reads and this one only records.
        ///
        /// **Recorded because nothing else at this scale looks at them.** The
        /// hidden state is what an MTP head is chained from, so a batched round
        /// that handed a sequence its neighbour's rows would guess from the
        /// wrong trajectory — and a proposer built to be deliberately wrong
        /// cannot notice, because its guesses do not come from the rows at all.
        chained: Vec<Vec<f32>>,
    }

    impl Guesser {
        fn new(truth: Vec<usize>, right: usize) -> Self {
            Self {
                truth,
                right,
                at: 0,
                guesses: Vec::new(),
                rounds: 0,
                chained: Vec::new(),
            }
        }
    }

    impl Proposer for Guesser {
        fn depth(&self) -> usize {
            DEPTH
        }

        fn propose(&mut self, round: Round<'_>) -> &[usize] {
            self.rounds += 1;
            self.at += round.next.len();
            assert_eq!(
                round.hidden.len(),
                round.next.len() * HIDDEN,
                "a hidden state per row the round kept"
            );
            self.chained.push(round.hidden.to_vec());
            assert_eq!(
                round.next.last().copied(),
                self.truth.get(self.at).copied(),
                "the last row's own next token"
            );

            let ahead = self.truth.get(self.at + 1..).unwrap_or_default();
            self.guesses = ahead
                .iter()
                .take(round.depth)
                .enumerate()
                .map(|(at, id)| match at < self.right {
                    true => *id,
                    false => (id + 1) % VOCAB,
                })
                .collect();
            &self.guesses
        }
    }

    /// How far ahead the cases guess, which is more than the four tokens they
    /// generate — so a round runs out of sequence to guess from, and the loop
    /// meets a block it cannot fill.
    const DEPTH: usize = 3;

    /// The synthetic stack's vocabulary, which a wrong guess has to stay inside
    /// of: an id past the head's rows is not a token the model could produce.
    const VOCAB: usize = 48;

    /// The synthetic stack's hidden width, which a round's chained rows are
    /// counted in.
    const HIDDEN: usize = 32;

    /// Everything a generation is: the tokens, why it ended, and the logits the
    /// cache it left behind produces for the next token.
    ///
    /// The last is what says the *state* is the same and not only the answer. A
    /// loop that emitted the right tokens over a cache holding the rejected ones
    /// would be a generation that is right until it is continued.
    fn ran(
        stack: &Stack,
        ending: Ending,
        take: usize,
        proposer: &mut impl Proposer,
    ) -> (Vec<usize>, Stop, Vec<f32>) {
        let head = stack.head();
        let cache = &mut ModelCache::speculating(&stack.config, proposer.depth());
        let mut streamed = Vec::new();
        let stop =
            generator(stack, &head).speculate(cache, &stack.ids, ending, stack, proposer, |id| {
                streamed.push(id);
                match streamed.len() < take {
                    true => ControlFlow::Continue(()),
                    false => ControlFlow::Break(()),
                }
            });
        let last = streamed.last().copied().unwrap_or(stack.ids[0]);
        (streamed, stop, logits(stack, cache, &[last]))
    }

    /// **The property speculation exists to keep**: the tokens are the tokens,
    /// however far ahead a round guessed and however wrong it was.
    ///
    /// Exact equality on all three of what a generation is — the tokens, the
    /// ending, and the logits the cache goes on to produce — against the same
    /// generation with no proposer at all. What a wrong guess must not leave
    /// behind is a token in the stream *or* a key in the cache, and only the
    /// third of those says the second.
    ///
    /// Every ending the loop has, because each ends a round in a different
    /// place: the budget between two of a round's tokens, the sink between two,
    /// and an id the model produces mid-block.
    #[test]
    fn speculation_changes_no_token() {
        let stack = Stack::load();
        let truth = [stack.ids.clone(), baseline(&stack)].concat();
        let eos = truth[stack.ids.len() + COUNT - 2];

        for (what, ending, take) in [
            (
                "the budget",
                Ending {
                    budget: COUNT,
                    eos: None,
                },
                usize::MAX,
            ),
            (
                "a shorter budget",
                Ending {
                    budget: COUNT - 1,
                    eos: None,
                },
                usize::MAX,
            ),
            (
                "an end-of-sequence id",
                Ending {
                    budget: COUNT,
                    eos: Some(eos),
                },
                usize::MAX,
            ),
            (
                "a sink that stops",
                Ending {
                    budget: COUNT,
                    eos: None,
                },
                2,
            ),
        ] {
            let alone = ran(&stack, ending, take, &mut Alone);
            for right in [0, 1, DEPTH] {
                let mut guesser = Guesser::new(truth.clone(), right);
                let speculated = ran(&stack, ending, take, &mut guesser);
                assert_eq!(speculated.0, alone.0, "{what}, {right} right: the tokens");
                assert_eq!(speculated.1, alone.1, "{what}, {right} right: the ending");
                assert_eq!(speculated.2, alone.2, "{what}, {right} right: the cache");
            }
        }
    }

    /// And that it banks anything at all, which the case above cannot say: a
    /// loop that ignored every guess would pass it.
    ///
    /// Stated in rounds rather than in time, because what a round is worth is
    /// the forward pass it did not run. Counted as the rounds that went on to
    /// ask for another guess: four tokens are four rounds when every guess is
    /// wrong — a round a token, which is decoding — and two when none is, since
    /// the prefill has nothing to guess from yet and the round after it banks
    /// the other three.
    #[test]
    fn a_round_that_guessed_right_is_a_forward_pass_nobody_ran() {
        let stack = Stack::load();
        let truth = [stack.ids.clone(), baseline(&stack)].concat();
        let ending = Ending {
            budget: COUNT,
            eos: None,
        };

        let rounds = |right: usize| {
            let mut guesser = Guesser::new(truth.clone(), right);
            ran(&stack, ending, usize::MAX, &mut guesser);
            guesser.rounds
        };
        let (perfect, wrong) = (rounds(DEPTH), rounds(0));
        assert!(
            perfect < wrong,
            "{perfect} rounds guessing right against {wrong} guessing wrong"
        );
        assert_eq!(wrong, COUNT - 1, "a round a token, which is decoding");
        assert_eq!(perfect, 1, "one round bought the rest of the budget");
    }

    /// A cache that kept no slack cannot give a rejected token back, and says so
    /// where the round would have taken it — rather than answering out of a
    /// window that no longer holds what it claims to.
    #[test]
    #[should_panic(expected = "the window can give back")]
    fn speculating_against_a_cache_that_kept_no_slack_is_refused() {
        let stack = Stack::load();
        let truth = [stack.ids.clone(), baseline(&stack)].concat();
        let head = stack.head();
        let ending = Ending {
            budget: COUNT,
            eos: None,
        };

        generator(&stack, &head).speculate(
            &mut ModelCache::new(&stack.config),
            &stack.ids,
            ending,
            &stack,
            &mut Guesser::new(truth, 0),
            |_| ControlFlow::Continue(()),
        );
    }

    #[test]
    #[should_panic(expected = "a forward pass over no tokens")]
    fn a_forward_pass_over_no_tokens_is_refused() {
        let stack = Stack::load();
        logits(&stack, &mut ModelCache::new(&stack.config), &[]);
    }

    /// A chain a sequence, each guessing right up to its own depth, and the
    /// rows each round handed every one of them.
    ///
    /// **One proposer per sequence rather than one shared**, which is what
    /// [`BatchProposer`] exists to say: a chain carries state per sequence, and
    /// a batch is N of them asked at once.
    struct Guessers {
        chains: Vec<Guesser>,
        /// Per round: the rows each sequence fed, in the order they were
        /// handed over. What the ragged case is checked *against*, since a
        /// batch whose seats are all the same length is not the shape this
        /// milestone is about — see
        /// `a_batched_verify_is_ragged_and_the_seats_are_not_the_same_length`.
        seats: Vec<Vec<usize>>,
    }

    impl Guessers {
        fn new(right: &[usize]) -> Self {
            Self {
                chains: right
                    .iter()
                    .map(|right| Guesser::new(Vec::new(), *right))
                    .collect(),
                seats: Vec::new(),
            }
        }
    }

    impl BatchProposer for Guessers {
        fn depth(&self) -> usize {
            DEPTH
        }

        fn propose_batch(&mut self, rounds: &[Seated<'_>]) -> Vec<Vec<usize>> {
            self.seats.push(
                rounds
                    .iter()
                    .map(|seated| seated.round.next.len())
                    .collect(),
            );
            rounds
                .iter()
                .map(|seated| self.chains[seated.at].propose(seated.round).to_vec())
                .collect()
        }
    }

    /// The same generation run alone, for the sequence a batched case is
    /// checking one seat of: the tokens it produced, and the rows every round
    /// handed it to be chained from.
    fn alone(stack: &Stack, prompt: &[usize], count: usize, right: usize) -> (Vec<usize>, Guesser) {
        let head = stack.head();
        let truth = [
            prompt.to_vec(),
            generator(stack, &head).generate(
                &mut ModelCache::new(&stack.config),
                prompt,
                count,
                stack,
            ),
        ]
        .concat();
        let cache = &mut ModelCache::speculating(&stack.config, DEPTH);
        let mut generated = Vec::new();
        let mut guesser = Guesser::new(truth, right);
        generator(stack, &head).speculate(
            cache,
            prompt,
            Ending {
                budget: count,
                eos: None,
            },
            stack,
            &mut guesser,
            |id| {
                generated.push(id);
                ControlFlow::Continue(())
            },
        );
        (generated, guesser)
    }

    /// The prompts a batched speculative case drives, which are three lengths
    /// so that no two sequences' rows line up by accident.
    fn prompts(stack: &Stack) -> Vec<Vec<usize>> {
        let sequence = stack.sequence();
        vec![
            sequence[..3].to_vec(),
            sequence[3..].to_vec(),
            sequence[..1].to_vec(),
        ]
    }

    /// How many of its round's guesses each of the three sequences gets right,
    /// which is what makes the seats of one call unequal: at `DEPTH` the model
    /// agrees with all three and the sequence banks four rows, at 1 it banks
    /// two, and at 0 it banks one.
    const RIGHT: [usize; 3] = [DEPTH, 0, 1];

    /// How many tokens a batched speculative case asks each sequence for.
    ///
    /// **More than [`COUNT`], and the difference is load-bearing**: a round that
    /// accepts everything banks the whole of a four-token budget in one go and
    /// never rejects a guess, so a budget that short leaves the rewind — which
    /// is the one thing a ragged verify does per sequence — unexercised.
    const ROUNDS: usize = COUNT + 4;

    /// Every sequence's round through a batched proposer, and every seat's own
    /// chain beside it.
    ///
    /// Returns what each seat produced and what each seat's chain recorded,
    /// which is the pair every case below reads: a token comparison says the
    /// answer is the same and the chained rows say the *trajectory* was.
    fn batched(
        stack: &Stack,
        prompts: &[Vec<usize>],
        order: &[usize],
        counts: &[usize],
        truths: &[Vec<usize>],
    ) -> (Vec<Vec<usize>>, Guessers) {
        let head = stack.head();
        let generator = generator(stack, &head);
        let mut caches: Vec<ModelCache> = (0..order.len())
            .map(|slot| ModelCache::in_slot(&stack.config, DEPTH, slot))
            .collect();
        let ids: Vec<&[usize]> = order.iter().map(|at| prompts[*at].as_slice()).collect();
        let budgets: Vec<usize> = order.iter().map(|at| counts[*at]).collect();
        let mut proposer = Guessers::new(&order.iter().map(|at| RIGHT[*at]).collect::<Vec<_>>());
        for (chain, at) in proposer.chains.iter_mut().zip(order) {
            chain.truth.clone_from(&truths[*at]);
        }
        let produced = generator.speculate_batch(&mut caches, &ids, &budgets, stack, &mut proposer);
        (produced, proposer)
    }

    /// **A generation that speculates inside a batch produces what it produces
    /// alone**, at every position of the batch and beside neighbours whose
    /// rounds accept a different number of guesses from its own.
    ///
    /// This is the loop's half of the milestone's claim, and what it can fail
    /// on is the bookkeeping a ragged verify needs: a sequence rewound by the
    /// call's rows rather than by its own, a sequence chained from the rows of
    /// whichever seat came first, or a sequence handed its neighbour's guesses.
    ///
    /// **Both halves of a round are checked and the second is not the tokens.**
    /// The rows a round hands its proposer are what a real chain is chained
    /// from, and a proposer built to be deliberately wrong guesses without
    /// reading them — so a batch that handed a sequence its neighbour's rows
    /// would produce the right tokens here and the wrong ones with the heads in
    /// place. Asserting the recorded rows is what closes that.
    ///
    /// Exactly the tokens and not nearly: every arm multiplies the same numbers
    /// in the same order, so a batch that has not leaked is a batch that agrees
    /// bit for bit.
    #[test]
    fn a_speculating_generation_in_a_batch_produces_what_it_produces_alone() {
        let stack = Stack::load();
        let prompts = prompts(&stack);
        // **The first finishes four tokens before the others**, which is what a
        // batch's tail looks like: the sequences do not stop together. The
        // *first* rather than the last, because a live set that only ever
        // shrinks from its end is one whose members are still their own
        // positions — and a round handed back by position rather than by slot
        // would be right in every batch that drains that way.
        let counts = [ROUNDS - 4, ROUNDS, ROUNDS];

        let apart: Vec<(Vec<usize>, Guesser)> = (0..prompts.len())
            .map(|at| alone(&stack, &prompts[at], counts[at], RIGHT[at]))
            .collect();
        assert_ne!(apart[0].0, apart[1].0, "two generations to tell apart");
        let truths: Vec<Vec<usize>> = (0..prompts.len())
            .map(|at| [prompts[at].clone(), apart[at].0.clone()].concat())
            .collect();

        for order in [vec![0, 1, 2], vec![2, 1, 0], vec![1, 2], vec![2, 0]] {
            let (produced, proposer) = batched(&stack, &prompts, &order, &counts, &truths);
            for (position, at) in order.iter().enumerate() {
                assert_eq!(
                    produced[position], apart[*at].0,
                    "sequence {at} at position {position} of {order:?}: the tokens"
                );
                assert_eq!(
                    proposer.chains[position].chained, apart[*at].1.chained,
                    "sequence {at} at position {position} of {order:?}: the chained rows"
                );
            }
        }
    }

    /// **And that the seats of one call are actually unequal**, which the case
    /// above cannot say: a batch whose sequences all accept the same number of
    /// guesses is the one shape a ragged verify is not being tested by, and it
    /// would pass every assertion above.
    ///
    /// Stated on the rows the chain was handed rather than on the verify's,
    /// because the two are the same rows — a round runs its heads over what it
    /// committed — and this is the side a proposer can see.
    #[test]
    fn a_batched_verify_is_ragged_and_the_seats_are_not_the_same_length() {
        let stack = Stack::load();
        let prompts = prompts(&stack);
        let counts = [ROUNDS; 3];
        let order: Vec<usize> = (0..prompts.len()).collect();

        let truths: Vec<Vec<usize>> = (0..prompts.len())
            .map(|at| {
                let (produced, _) = alone(&stack, &prompts[at], counts[at], RIGHT[at]);
                [prompts[at].clone(), produced].concat()
            })
            .collect();
        let (_, proposer) = batched(&stack, &prompts, &order, &counts, &truths);

        let ragged = proposer
            .seats
            .iter()
            // The prompts open the rounds and are three lengths of their own,
            // so the round that says something about *acceptance* is one where
            // every seat is feeding a block rather than a prompt.
            .skip(1)
            .filter(|seats| seats.len() > 1)
            .filter(|seats| seats.iter().any(|rows| *rows != seats[0]))
            .count();
        assert!(
            ragged > 0,
            "every round's seats were the same length: {:?}",
            proposer.seats
        );
    }

    /// **A batch that speculates nothing is the batch that always ran.** The
    /// loop above is [`Generator::generate_batch`] with a proposer in it, and
    /// this is what says so: the same tokens, from the same prompts, at the
    /// same budgets.
    #[test]
    fn a_batch_that_guesses_nothing_generates_what_a_plain_batch_generates() {
        let stack = Stack::load();
        let head = stack.head();
        let generator = generator(&stack, &head);
        let prompts = prompts(&stack);
        let counts = [COUNT, COUNT, COUNT - 2];
        let ids: Vec<&[usize]> = prompts.iter().map(Vec::as_slice).collect();

        let mut plain: Vec<ModelCache> = (0..prompts.len())
            .map(|slot| ModelCache::in_slot(&stack.config, 0, slot))
            .collect();
        let mut speculating: Vec<ModelCache> = (0..prompts.len())
            .map(|slot| ModelCache::in_slot(&stack.config, 0, slot))
            .collect();

        assert_eq!(
            generator.speculate_batch(&mut speculating, &ids, &counts, &stack, &mut Alone),
            generator.generate_batch(&mut plain, &ids, &counts, &stack)
        );
    }
}
