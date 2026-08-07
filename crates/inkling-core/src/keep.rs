//! What one request leaves behind for the next one.
//!
//! # The lever
//!
//! A coding session sends the same context back turn after turn with a little
//! added each time. A server that keeps nothing re-prefills the whole
//! conversation every turn, which at 16384 tokens is 33 s of work to re-derive a
//! cache it computed thirty seconds earlier and threw away. Keeping it and
//! prefilling only what is new is worth more than any kernel in this tree.
//!
//! # One conversation a slot, because a slot holds one sequence
//!
//! A backend holds one span and one window pair per layer *per slot*, so two
//! conversations interleaved through the same slot would overwrite each other's
//! keys. That is not a policy decision here, it is the shape of the backend: a
//! prompt that does not extend what a slot holds replaces it, and there is no
//! arrangement of this file that could keep both in one slot.
//!
//! So one of these is not a prefix-cache service and does not pretend to be one.
//! It is one entry, and the entry is a conversation. A server answering one
//! request at a time holds one, in slot zero; [`Continuous`](crate::Continuous)
//! holds one per slot and decides which slot an arriving prompt is given — see
//! its `admit`, where the matching below is what a returning conversation is
//! recognised by.
//!
//! # The invariant, which is what makes the matching a comparison
//!
//! **The cache sits at the end of the ids this records.** Nothing else about it
//! has to be known: a prompt whose leading tokens are those ids is a prompt the
//! cache is a prefix of, and what is left to prefill is the rest of it.
//!
//! Holding that invariant is why a request marks its cache before it generates
//! and resumes the mark afterwards — see [`Kept::keep`]. A cache left where the
//! generation put it would record the reply as well, and the reply a client
//! sends back is not the reply the model streamed: the turn structure renders a
//! thinking channel as a message of its own where the model emits it inside one.
//! A cache that recorded them would match to somewhere in the middle of the
//! reply, and there is no mark there.
//!
//! # What it costs, and what it does not
//!
//! **The keys were already held between requests before this file existed.** A
//! layer's span is allocated on the device, grows by doubling, and is never
//! given back — a fresh cache sets `held` to zero and leaves the buffer where it
//! is. So what a kept conversation adds is not the KV: it is the ids, and the
//! mark a request takes and drops inside its own scope.
//!
//! On `--backend cpu` that is not true and the bound is what stands in the way:
//! there the keys are the cache's own vectors, so a kept cache is a kept 2688
//! MiB at 8192 tokens where before it was freed when the request ended.

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use crate::config::TextConfig;
use crate::generate::{Ending, Generator, Stop};
use crate::model::{ModelCache, ModelWeights};

/// What one turn came to, beside the reply it streamed.
#[derive(Debug, Clone)]
pub struct Served {
    pub stop: Stop,
    /// Tokens of the prompt the cache already held, and zero on a miss.
    pub reused: usize,
    /// **What the arrangement costs whether it hits or misses**, which is the
    /// question a feature like this has to answer about its own bad case: the
    /// matching, the mark, the resume and the recording. It does not include the
    /// prefill or a single decode step, and it does include the fresh cache a
    /// miss builds — which is what the server allocated on every request before
    /// any of this existed.
    pub bookkeeping: Duration,
}

/// The kept conversation: the tokens the cache was built from, and the cache
/// sitting at the end of them.
#[derive(Debug)]
pub struct Kept<'a> {
    config: &'a TextConfig,
    bound: usize,
    /// Which of a backend's slots the cache's keys and windows live in, which a
    /// forgotten conversation is rebuilt into — see [`Kept::in_slot`].
    slot: usize,
    /// The tokens the cache holds, and nothing where it holds none.
    ids: Vec<usize>,
    cache: ModelCache,
}

/// The most positions a conversation may be kept at.
///
/// **What it bounds is different on the two paths**, which is why it is a count
/// of positions rather than of bytes. On the device the spans are the backend's
/// and are held whether or not anything is kept here, so what this bounds is a
/// conversation growing without end. On the CPU path the keys are in the cache
/// itself and this is the whole of what stands between a server and a resident
/// set that only grows: the README's own table puts 32768 tokens at 10752 MiB
/// held.
///
/// 32768 because that is the largest context this repo's KV table prices at
/// under a fifth of the host. A conversation past it is served and forgotten,
/// which costs the turn after it a cold prefill and costs nothing else.
pub const DEFAULT_BOUND: usize = 32768;

impl<'a> Kept<'a> {
    /// A server holding nothing yet. `bound` of zero keeps nothing at all, which
    /// is the arm every measurement of this is taken against.
    pub fn new(config: &'a TextConfig, bound: usize) -> Self {
        Self::in_slot(config, bound, 0)
    }

    /// The same, for a conversation whose keys and windows are a backend's slot
    /// `slot` rather than its first.
    ///
    /// **A slot is where the state actually is, so it has to survive
    /// forgetting.** [`Kept::forget`] builds a cache that holds nothing, and one
    /// built without the slot would name slot zero — which is another
    /// conversation's span in a batch, and a sequence handed it would write its
    /// keys over a neighbour's. See
    /// [`ModelCache::in_slot`](crate::ModelCache::in_slot).
    pub fn in_slot(config: &'a TextConfig, bound: usize, slot: usize) -> Self {
        Self {
            config,
            bound,
            slot,
            ids: Vec::new(),
            cache: ModelCache::in_slot(config, 0, slot),
        }
    }

    /// How many of `ids` the cache in hand already holds.
    ///
    /// **All of them or none**, which is what "exact extension" means: this
    /// keeps one position it can resume to and that position is the end of what
    /// it recorded, so a prompt that agrees for a while and then parts company
    /// is a prompt with nothing here to start from. A coding turn is an exact
    /// extension of the turn before it, which is the case worth having before
    /// any cleverness is.
    ///
    /// Never the whole of `ids`: a forward pass over no tokens is not a forward
    /// pass, so the last token of a prompt is always one this leaves to be fed.
    pub fn matching(&self, ids: &[usize]) -> usize {
        match ids.len() > self.ids.len() && ids.starts_with(&self.ids) {
            true => self.ids.len(),
            false => 0,
        }
    }

    /// The cache to serve `ids` against, and how many of them it already holds.
    ///
    /// A prompt that does not extend what is kept gets a cache that holds
    /// nothing — which is also what puts the device's spans and windows back to
    /// the start of a sequence, since a cache that has seen no keys is what they
    /// read as one beginning.
    pub fn opened(&mut self, ids: &[usize]) -> (&mut ModelCache, usize) {
        let matching = self.matching(ids);
        if matching == 0 {
            self.forget();
        }
        (&mut self.cache, matching)
    }

    /// Record `ids` as what the cache now sits at the end of.
    ///
    /// The caller's part of the invariant, and the whole of it: it has resumed
    /// the mark it took at the end of `ids`, so the cache is where this says it
    /// is. A caller that recorded a longer sequence than it resumed to would
    /// hand the next request a match it cannot honour.
    ///
    /// A conversation past the bound is forgotten rather than trimmed. There is
    /// no position between the two to resume to — the mark is at the end of the
    /// prompt or nowhere — so what a bound can do is decline to keep.
    pub fn keep(&mut self, ids: &[usize]) {
        match ids.len() <= self.bound {
            true => {
                self.ids.clear();
                self.ids.extend_from_slice(ids);
            }
            false => self.forget(),
        }
    }

    /// One turn served against what this holds: the part of `ids` it does not
    /// already have prefilled, the reply streamed to `sink`, and the cache put
    /// back at the end of the prompt. Answers with the stop and how many of
    /// `ids` it did not have to prefill.
    ///
    /// **The whole of the arrangement is here rather than at the server**, so
    /// that what a measurement of this drives is what a request drives. The
    /// three things it has to get right are the three lines: the last token of
    /// the prompt is fed by the generation rather than the prefill, because a
    /// forward pass over no tokens is not one; the mark is taken where the
    /// prompt ends and not where the reply does; and what is recorded is what
    /// the cache was resumed to.
    pub fn turn(
        &mut self,
        generator: &Generator<'_>,
        weights: &impl ModelWeights,
        ids: &[usize],
        ending: Ending,
        sink: impl FnMut(usize) -> ControlFlow<()>,
    ) -> Served {
        assert!(!ids.is_empty(), "a turn over no tokens");
        let held = ids.len() - 1;

        let at = Instant::now();
        let (cache, reused) = self.opened(ids);
        let mut bookkeeping = at.elapsed();
        if reused < held {
            generator.prefill(cache, &ids[reused..held], weights);
        }

        // Where the next turn's prompt starts, taken before the generation moves
        // the cache past it and put back once the reply is done.
        let at = Instant::now();
        let mark = weights.mark(cache);
        bookkeeping += at.elapsed();

        let stop = generator.stream(cache, &ids[held..], ending, weights, sink);

        let at = Instant::now();
        weights.resume(cache, &mark);
        self.keep(&ids[..held]);
        bookkeeping += at.elapsed();

        Served {
            stop,
            reused,
            bookkeeping,
        }
    }

    /// Keep nothing, and start the next turn from a cache that holds nothing.
    ///
    /// **Both of the ways a conversation ends here**: a prompt that does not
    /// extend the kept one, and a conversation that has grown past the bound.
    /// Either way what the next turn needs is a cache whose count is zero, which
    /// is what the device reads as a sequence beginning — see
    /// [`Kept::opened`].
    ///
    /// It is not a recovery path and there is nothing here that could be one: a
    /// failure inside a turn is a panic in this tree, and a panic takes the
    /// process with it rather than leaving a server holding a cache nobody can
    /// say the position of.
    pub fn forget(&mut self) {
        self.ids.clear();
        self.cache = ModelCache::in_slot(self.config, 0, self.slot);
    }

    /// Positions this is holding a conversation at.
    pub fn held(&self) -> usize {
        self.ids.len()
    }

    /// The cache itself, for a caller that has already decided this is the
    /// conversation it is serving — which [`Kept::opened`] is the deciding of.
    ///
    /// **Not a way round the matching.** A caller that fed rows through here
    /// without opening first would be feeding them into whatever the last
    /// conversation left, which is precisely the contamination the ids exist to
    /// rule out. What it is for is a scheduler, which opens a slot once when a
    /// sequence is admitted and then advances it a step at a time for as long as
    /// that sequence is seated.
    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }

    /// The same, to feed rows through. See [`Kept::cache`].
    pub fn cache_mut(&mut self) -> &mut ModelCache {
        &mut self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::Stack;

    /// A conversation and the turn after it: the same tokens, with more on the
    /// end. Realistic in the one way that matters here — the prefix is long and
    /// the addition is short, which is the shape the whole thing exists for.
    const TURN: [usize; 6] = [11, 22, 33, 44, 55, 66];

    fn kept(config: &TextConfig) -> Kept<'_> {
        Kept::new(config, DEFAULT_BOUND)
    }

    /// A server holding nothing matches nothing, however familiar the prompt
    /// looks.
    #[test]
    fn a_server_that_has_kept_nothing_starts_every_prompt_from_the_beginning() {
        let stack = Stack::load();
        let mut kept = kept(&stack.config);
        assert_eq!(kept.held(), 0);
        assert_eq!(kept.opened(&TURN).1, 0);
    }

    /// The case the whole file is for: turn two is turn one with more on the
    /// end, and what it has to prefill is the more.
    #[test]
    fn a_turn_that_extends_the_kept_one_starts_where_it_left_off() {
        let stack = Stack::load();
        let mut kept = kept(&stack.config);
        kept.keep(&TURN[..4]);

        assert_eq!(kept.held(), 4);
        assert_eq!(kept.opened(&TURN).1, 4);
    }

    /// A prompt that agrees for a while and then does not has nothing here to
    /// start from — there is one position kept and it is not inside the
    /// disagreement.
    #[test]
    fn a_turn_that_parts_company_with_the_kept_one_starts_from_the_beginning() {
        let stack = Stack::load();
        let mut kept = kept(&stack.config);
        kept.keep(&TURN[..4]);

        let diverged = [TURN[0], TURN[1], 99, TURN[3], TURN[4], TURN[5]];
        assert_eq!(kept.matching(&diverged), 0);
        assert_eq!(kept.opened(&diverged).1, 0);
    }

    /// A prompt this holds every token of would leave nothing to feed, and a
    /// forward pass over no tokens is not one. The same prompt asked twice is
    /// the commonest way for a client to produce that.
    #[test]
    fn a_prompt_the_cache_already_holds_whole_still_leaves_a_token_to_feed() {
        let stack = Stack::load();
        let mut kept = kept(&stack.config);
        kept.keep(&TURN);

        assert_eq!(kept.matching(&TURN), 0);
        assert_eq!(
            kept.matching(&TURN[..3]),
            0,
            "a prompt shorter than the kept"
        );
    }

    /// The bound refuses to *keep*, rather than refusing to serve: the request
    /// that reached it is answered, and the turn after it pays a cold prefill.
    #[test]
    fn a_conversation_past_the_bound_is_served_and_then_forgotten() {
        let stack = Stack::load();
        let mut kept = Kept::new(&stack.config, 4);
        kept.keep(&TURN[..4]);
        assert_eq!(kept.held(), 4);

        kept.keep(&TURN[..5]);
        assert_eq!(kept.held(), 0, "the bound kept a conversation past it");
        assert_eq!(kept.opened(&TURN).1, 0);
    }

    /// A bound of zero is the off switch, and it is what every arm this is
    /// measured against runs under.
    #[test]
    fn a_bound_of_no_positions_keeps_nothing() {
        let stack = Stack::load();
        let mut kept = Kept::new(&stack.config, 0);
        kept.keep(&TURN[..1]);
        assert_eq!(kept.held(), 0);
    }

    /// **A conversation kept in a slot stays in that slot, forgotten or not.**
    /// The keys and the windows are the slot's, so a cache rebuilt into slot
    /// zero would have the next sequence writing over whoever is seated there —
    /// which is the one way a per-slot conversation could reintroduce exactly
    /// the contamination it exists to rule out.
    #[test]
    fn a_conversation_forgotten_in_a_slot_is_rebuilt_in_the_same_slot() {
        let stack = Stack::load();
        let mut kept = Kept::in_slot(&stack.config, DEFAULT_BOUND, 3);
        assert_eq!(kept.opened(&TURN).0.slot(), 3);

        kept.keep(&TURN[..4]);
        kept.forget();
        assert_eq!(kept.held(), 0);
        assert_eq!(kept.opened(&TURN).0.slot(), 3, "forgetting moved the slot");
    }

    /// Forgetting is both of the ways a conversation ends here, and what it has
    /// to leave is a cache the next turn reads as a sequence beginning — not
    /// merely an empty list of ids beside a cache still holding keys.
    #[test]
    fn forgetting_a_conversation_leaves_the_next_turn_nothing_to_start_from() {
        let stack = Stack::load();
        let mut kept = kept(&stack.config);
        kept.keep(&TURN[..4]);
        kept.forget();

        assert_eq!(kept.held(), 0);
        assert_eq!(kept.opened(&TURN).1, 0);
    }
}
