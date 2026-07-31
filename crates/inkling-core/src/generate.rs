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

    /// `prompt` prefilled, then `count` tokens decoded greedily, each fed back
    /// through the caches the step before it left behind.
    ///
    /// The last token generated is returned without ever being fed back, so its
    /// keys are not in `cache` when this returns — which is what a generator
    /// that stopped on an end-of-sequence id would want, and what makes
    /// generating `n` then `m` more the same sequence as generating `n + m`.
    ///
    /// A `count` of zero prefills nothing and leaves `cache` untouched: the
    /// prompt enters it on the first step, because a prompt whose next token is
    /// never asked for is a forward pass with no answer.
    pub fn generate(
        &self,
        cache: &mut ModelCache,
        prompt: &[usize],
        count: usize,
        weights: &impl ModelWeights,
        head_row: impl Fn(usize) -> Vec<f32>,
    ) -> Vec<usize> {
        let mut generated = Vec::with_capacity(count);
        let mut ids = prompt.to_vec();
        for _ in 0..count {
            ids = vec![greedy(&self.logits(cache, &ids, weights, &head_row))];
            generated.extend_from_slice(&ids);
        }
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
        let at_once = generate(
            &stack,
            &mut ModelCache::new(&stack.config),
            &stack.ids,
            COUNT,
        );

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
        let generated = generate(
            &stack,
            &mut ModelCache::new(&stack.config),
            &stack.ids,
            COUNT,
        );
        assert!(
            generated.iter().all(|id| *id < head.vocab()),
            "{generated:?}"
        );
    }

    #[test]
    fn generating_nothing_leaves_the_cache_alone() {
        let stack = Stack::load();
        let cache = &mut ModelCache::new(&stack.config);
        assert!(generate(&stack, cache, &stack.ids, 0).is_empty());
        assert_eq!(logits(&stack, cache, &stack.ids), whole(&stack, &stack.ids));
    }

    #[test]
    #[should_panic(expected = "a forward pass over no tokens")]
    fn a_forward_pass_over_no_tokens_is_refused() {
        let stack = Stack::load();
        logits(&stack, &mut ModelCache::new(&stack.config), &[]);
    }
}
