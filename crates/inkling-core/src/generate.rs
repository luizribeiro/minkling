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
//! forces: a decode step costs 9.2 s on the CPU path, so a caller that waits for
//! the last one waits minutes for the first. Nothing here writes anything —
//! whether a token becomes text, a line on a terminal or a chunk on a socket is
//! the sink's, and [`crate::detokenize`] is what makes a token's worth of text
//! well-defined at all.
//!
//! Three things end a generation, and [`Stop`] says which: the end-of-sequence
//! id, the budget, or the sink declining the next one. The last exists because a
//! sink can fail — a closed pipe, a client that hung up — and at 9.2 s a step,
//! discovering that only after the budget runs out is minutes of arithmetic
//! nobody will read.

use std::ops::ControlFlow;

use crate::head::LmHead;
use crate::model::{Model, ModelCache, ModelWeights};
use crate::ops::top_k;

/// The id a greedy sampler takes.
///
/// [`top_k`] rather than a fresh argmax, so that the rule for a tie — the lower
/// id — is the one every ranking in the tree already uses, and a position whose
/// top two logits bfloat16 cannot tell apart resolves the same way here as it
/// does in a comparison against the oracle.
pub fn greedy(logits: &[f32]) -> usize {
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

/// The model and its head, as the thing that turns ids into ids.
///
/// `LanguageModel` is the two of them together — the stack, its final norm, and
/// `lm_head` — and generation is the first caller that needs all of it at once.
#[derive(Debug, Clone, Copy)]
pub struct Generator<'a> {
    model: Model<'a>,
    head: LmHead,
}

impl<'a> Generator<'a> {
    pub fn new(model: Model<'a>, head: LmHead) -> Self {
        Self { model, head }
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
    ///
    /// `head_row` is asked for row `id` of the final projection, `[hidden]`
    /// long, the same way [`LmHead::forward`] asks — which is what keeps a
    /// 3.3 GB table from having to be held to generate one token.
    pub fn logits(
        &self,
        cache: &mut ModelCache,
        ids: &[usize],
        weights: &impl ModelWeights,
        head_row: impl Fn(usize) -> Vec<f32>,
    ) -> Vec<f32> {
        assert!(!ids.is_empty(), "a forward pass over no tokens");

        let h = self.model.forward(cache, ids, weights);
        let hidden = h.len() / ids.len();
        let last = &h[h.len() - hidden..];
        self.head.forward(&self.model.final_norm(last), head_row)
    }

    /// `prompt` prefilled, then tokens decoded greedily until `ending` says to
    /// stop, each handed to `sink` as it is decided and then fed back through
    /// the caches the step before it left behind.
    ///
    /// The prompt is prefilled by the first step rather than by a step of its
    /// own, so that step's cost is a prefill's and every later one's is a
    /// decode's. Against this checkpoint those are 54.7 s and 9.2 s — the two
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
        head_row: impl Fn(usize) -> Vec<f32>,
        mut sink: impl FnMut(usize) -> ControlFlow<()>,
    ) -> Stop {
        let mut ids = prompt.to_vec();
        for _ in 0..ending.budget {
            let id = greedy(&self.logits(cache, &ids, weights, &head_row));
            if sink(id).is_break() {
                return Stop::Sink;
            }
            if Some(id) == ending.eos {
                return Stop::EndOfSequence;
            }
            ids = vec![id];
        }
        Stop::Budget
    }

    /// [`stream`](Self::stream) collected: `count` tokens decoded greedily, with
    /// no id ending it early.
    pub fn generate(
        &self,
        cache: &mut ModelCache,
        prompt: &[usize],
        count: usize,
        weights: &impl ModelWeights,
        head_row: impl Fn(usize) -> Vec<f32>,
    ) -> Vec<usize> {
        let mut generated = Vec::with_capacity(count);
        let ending = Ending {
            budget: count,
            eos: None,
        };
        self.stream(cache, prompt, ending, weights, head_row, |id| {
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

    /// The synthetic stack carries no `lm_head` — `InklingModel` does not hold
    /// one — so the head here reads the embedding table, which is exactly what
    /// `tie_word_embeddings` decides and what
    /// [`head_module`](crate::weights) resolves for a tied checkpoint:
    /// `nn.Embedding.as_linear` is `h @ Wᵀ` over the same `[vocab, hidden]` rows
    /// the lookup returns.
    ///
    /// Which table the rows come from decides nothing here. What is under test
    /// is the loop, and the loop needs a head that ranks 48 ids rather than a
    /// particular one.
    fn generator(stack: &Stack) -> Generator<'_> {
        Generator::new(stack.model(), LmHead::for_config(&stack.config))
    }

    fn logits(stack: &Stack, cache: &mut ModelCache, ids: &[usize]) -> Vec<f32> {
        generator(stack).logits(cache, ids, stack, |id| stack.embedding_row(id))
    }

    fn generate(
        stack: &Stack,
        cache: &mut ModelCache,
        prompt: &[usize],
        count: usize,
    ) -> Vec<usize> {
        generator(stack).generate(cache, prompt, count, stack, |id| stack.embedding_row(id))
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
        let mut streamed = Vec::new();
        let stop = generator(stack).stream(
            cache,
            prompt,
            ending,
            stack,
            |id| stack.embedding_row(id),
            |id| {
                streamed.push(id);
                match streamed.len() < take {
                    true => ControlFlow::Continue(()),
                    false => ControlFlow::Break(()),
                }
            },
        );
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

    /// A sink that stops taking tokens stops the model too. At 9.2 s a decode
    /// step, a closed pipe discovered only when the budget runs out is minutes
    /// of arithmetic nobody will read — so what says this works is the count of
    /// tokens the sink was offered, not the ones it kept.
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

    #[test]
    #[should_panic(expected = "a forward pass over no tokens")]
    fn a_forward_pass_over_no_tokens_is_refused() {
        let stack = Stack::load();
        logits(&stack, &mut ModelCache::new(&stack.config), &[]);
    }
}
