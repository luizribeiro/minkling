//! The back of the model: the normed hidden state to logits.
//!
//! `LanguageModel._logits_from_norm` is three lines, and each of them is a way
//! to be wrong that reading generated text cannot show.
//!
//! **The muP divide comes before the projection.** `h` is divided by
//! `logits_mup_width_multiplier` — 16.0 for Inkling-Small — and only then
//! projected. Dropping it multiplies every logit by sixteen, which moves no
//! argmax at all and rescales every distribution a sampler will draw from: a
//! temperature of 0.7 becomes 0.044, and greedy decoding is bit-identical. No
//! test that asserts on ordering can see this, which is why the hermetic case
//! asserts on values.
//!
//! **The head is wider than the vocabulary.** Its 201024 rows carry 200058 real
//! ones, and `unpadded_vocab_size` is where the reference cuts. What the
//! remaining 966 hold decides whether that cut matters, and here it does:
//! `lm_head`'s padding is all-zero MXFP4 codes under all-zero scales — the
//! quantisation fixture's `vocab_padding` slice pins that — so every one of them
//! decodes to exactly 0.0 and produces a logit of exactly 0.0. That is not a
//! harmless nothing. It outranks every real logit that came out negative, which
//! at a given position is most of them, and if the position's best real logit is
//! itself negative it outranks all of them and takes the argmax. The other end
//! of the model does not agree about this — `embed_tokens`' padding rows are
//! small but nonzero, which [`crate::embed`] states — so neither end can be read
//! off the other.
//!
//! **A row is asked for, not handed over.** The head is `[201024, 4096]`, which
//! is 3.3 GB decoded, and [`crate::model`] gives the argument for why a weight
//! that size arrives through an index. It buys something specific here: the
//! truncation costs nothing to honour, because a row past the vocabulary is
//! never decoded at all.
//!
//! `tie_word_embeddings` decides only *which* table those rows come from —
//! `lm_head` when it is clear, `embed_tokens` read as a linear when it is set,
//! and `nn.Embedding.as_linear` is `h @ Wᵀ` over the same `[vocab, hidden]`
//! rows either way. So it is not a branch in the arithmetic and there is none
//! here; see [`CheckpointWeights`](crate::weights::CheckpointWeights) for where
//! the checkpoint answers it. Inkling-Small clears it and carries a
//! `language_model.lm_head`.
//!
//! Nothing here is new arithmetic — a logit is one row of `nn.Linear`, which
//! `reference/fixtures/ops.safetensors` already pins — so what the hermetic
//! cases test is the wiring, and the real head is left to
//! `tests/real_checkpoint.rs`.

use crate::config::TextConfig;
use crate::ops::linear;

/// `_logits_from_norm`: the muP divide, the projection, and the cut at the
/// unpadded vocabulary.
#[derive(Debug, Clone, Copy)]
pub struct LmHead {
    hidden: usize,
    vocab: usize,
    mup: f32,
}

impl LmHead {
    /// `vocab` is how many of the head's rows are vocabulary, which is where
    /// the reference truncates and so how many logits a token gets.
    pub fn new(hidden: usize, vocab: usize, mup: f32) -> Self {
        assert!(mup != 0.0, "the muP width multiplier divides");
        Self { hidden, vocab, mup }
    }

    /// The head this config asks for. `unpadded_vocab_size` is optional in the
    /// reference and absent means the head is all vocabulary, which is the one
    /// case where the padded width is the right one to project to.
    pub fn for_config(config: &TextConfig) -> Self {
        Self::new(
            config.hidden_size,
            config.unpadded_vocab_size.unwrap_or(config.vocab_size),
            config.logits_mup_width_multiplier,
        )
    }

    /// How many logits a token gets.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// `[tokens, hidden]` in, `[tokens, vocab]` out.
    ///
    /// `row` is asked for row `id` of the head, `[hidden]` long — the same shape
    /// of question [`Embed::forward`](crate::embed::Embed::forward) asks of the
    /// embedding table, and asked here only for the ids that survive the
    /// truncation.
    pub fn forward(&self, h: &[f32], row: impl Fn(usize) -> Vec<f32>) -> Vec<f32> {
        assert_eq!(
            h.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            h.len(),
            self.hidden
        );
        let tokens = h.len() / self.hidden;
        let scaled: Vec<f32> = h.iter().map(|x| x / self.mup).collect();

        let mut logits = vec![0.0; tokens * self.vocab];
        for id in 0..self.vocab {
            let column = linear(&scaled, &row(id), self.hidden);
            assert_eq!(column.len(), tokens, "row {id} of the head");
            for (token, logit) in column.into_iter().enumerate() {
                logits[token * self.vocab + id] = logit;
            }
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::fixture;
    use std::cell::RefCell;

    const HIDDEN: usize = 4;
    const VOCAB: usize = 3;
    const PADDING: usize = 2;
    const MUP: f32 = 16.0;

    /// The config the synthetic stack was built from, which states no unpadded
    /// vocabulary — the branch a padded checkpoint never exercises.
    const STACK_CONFIG: &str = "stack.json";

    /// A head whose real rows are all negative and whose padding rows are the
    /// all-zero ones the checkpoint carries.
    ///
    /// Negative on purpose: a padding logit is exactly 0.0, so what decides
    /// whether truncation is load-bearing is whether anything real beats zero.
    /// This is the position where nothing does.
    struct Head {
        rows: Vec<f32>,
        asked: RefCell<Vec<usize>>,
    }

    impl Head {
        fn new() -> Self {
            let real = [
                [-0.1, -0.2, -0.3, -0.4],
                [-0.2, -0.1, -0.1, -0.1],
                [-0.3, -0.3, -0.3, -0.3],
            ];
            Self {
                rows: real
                    .into_iter()
                    .flatten()
                    .chain(std::iter::repeat_n(0.0, PADDING * HIDDEN))
                    .collect(),
                asked: RefCell::new(Vec::new()),
            }
        }

        fn row(&self, id: usize) -> Vec<f32> {
            self.asked.borrow_mut().push(id);
            self.rows[id * HIDDEN..][..HIDDEN].to_vec()
        }

        fn logits(&self, head: &LmHead, h: &[f32]) -> Vec<f32> {
            head.forward(h, |id| self.row(id))
        }
    }

    fn hidden_state() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0]
    }

    fn argmax(logits: &[f32]) -> usize {
        crate::ops::top_k(logits, 1)[0]
    }

    /// The divide is the one step of the three that ordering cannot see, so it
    /// is asserted on values: two sets of logits a factor of the multiplier
    /// apart, with the same argmax.
    #[test]
    fn the_head_divides_by_the_mup_multiplier_before_projecting() {
        let (head, h) = (Head::new(), hidden_state());
        let divided = head.logits(&LmHead::new(HIDDEN, VOCAB, MUP), &h);
        let undivided = head.logits(&LmHead::new(HIDDEN, VOCAB, 1.0), &h);

        assert_eq!(
            divided,
            linear(&h, &head.rows[..VOCAB * HIDDEN], HIDDEN)
                .iter()
                .map(|logit| logit / MUP)
                .collect::<Vec<f32>>()
        );
        for (divided, undivided) in divided.iter().zip(&undivided) {
            assert_eq!(divided * MUP, *undivided);
        }
        assert_eq!(
            argmax(&divided),
            argmax(&undivided),
            "the multiplier is exactly what an ordering test cannot catch"
        );
    }

    /// Truncation, stated as the thing it prevents: the padding rows decode to
    /// zero, and at a position whose real logits are all negative a zero wins.
    #[test]
    fn an_untruncated_head_takes_its_argmax_from_the_padding() {
        let (head, h) = (Head::new(), hidden_state());

        let truncated = head.logits(&LmHead::new(HIDDEN, VOCAB, MUP), &h);
        assert_eq!(truncated.len(), VOCAB);
        assert!(
            truncated.iter().all(|logit| *logit < 0.0),
            "a position with a positive logit in it would not settle this"
        );
        assert_eq!(argmax(&truncated), 1);

        let untruncated = head.logits(&LmHead::new(HIDDEN, VOCAB + PADDING, MUP), &h);
        assert_eq!(untruncated[VOCAB..], [0.0; PADDING]);
        assert_eq!(argmax(&untruncated), VOCAB, "the first padding id");
    }

    /// Honouring the truncation by never decoding the row, which is what makes
    /// it free rather than a slice taken afterwards.
    #[test]
    fn a_truncated_head_never_asks_for_a_padding_row() {
        let (head, h) = (Head::new(), hidden_state());
        head.logits(&LmHead::new(HIDDEN, VOCAB, MUP), &h);
        assert_eq!(*head.asked.borrow(), (0..VOCAB).collect::<Vec<usize>>());
    }

    #[test]
    fn every_token_gets_its_own_row_of_logits() {
        let (head, h) = (Head::new(), hidden_state());
        let two = [h.clone(), h.clone()].concat();
        let logits = head.logits(&LmHead::new(HIDDEN, VOCAB, MUP), &two);

        assert_eq!(logits.len(), 2 * VOCAB);
        assert_eq!(logits[..VOCAB], logits[VOCAB..]);
        assert_eq!(
            logits[..VOCAB],
            head.logits(&LmHead::new(HIDDEN, VOCAB, MUP), &h)[..]
        );
    }

    /// A config that states no unpadded vocabulary projects to the whole head,
    /// which is the branch Inkling-Small's own config never reaches.
    #[test]
    fn a_config_without_an_unpadded_vocabulary_keeps_the_whole_head() {
        let mut config = serde_json::from_str::<Config>(&fixture::read(STACK_CONFIG))
            .expect("the recorded config parses")
            .text_config;
        assert_eq!(config.unpadded_vocab_size, None, "the fixture is unpadded");
        assert_eq!(LmHead::for_config(&config).vocab(), config.vocab_size);

        config.unpadded_vocab_size = Some(config.vocab_size - 1);
        assert_eq!(
            LmHead::for_config(&config).vocab(),
            config.vocab_size - 1,
            "a stated unpadded vocabulary is where the head stops"
        );
    }

    #[test]
    #[should_panic(expected = "the muP width multiplier divides")]
    fn a_zero_mup_multiplier_is_refused() {
        LmHead::new(HIDDEN, VOCAB, 0.0);
    }

    #[test]
    #[should_panic(expected = "are not whole rows of")]
    fn a_hidden_state_that_is_not_whole_rows_is_refused() {
        let (head, h) = (Head::new(), hidden_state());
        head.logits(&LmHead::new(HIDDEN, VOCAB, MUP), &h[1..]);
    }

    #[test]
    #[should_panic(expected = "row 0 of the head")]
    fn a_head_row_of_the_wrong_width_is_refused() {
        let head = LmHead::new(HIDDEN, VOCAB, MUP);
        head.forward(&hidden_state(), |_| vec![0.0; 2 * HIDDEN]);
    }
}
