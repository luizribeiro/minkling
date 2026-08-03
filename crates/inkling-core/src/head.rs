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
//! **The weights are named as a [`Projection`], not handed over.** The head is
//! `[201024, 4096]`, which is 3.3 GB decoded, and [`crate::model`] gives the
//! argument for why a weight that size is never a slice. What this asks for
//! instead is the operation — `h @ Wᵀ` over a weight whose storage it cannot see
//! — so the same three lines run against rows the CPU decodes one at a time
//! ([`PackedRows`](crate::weights::PackedRows)) and against codes a Metal
//! dispatch multiplies without decoding at all.
//!
//! The truncation is then a shape rather than a slice taken afterwards. The
//! projection is built to `vocab` outputs and the head refuses one that is not,
//! so a row past the vocabulary is never decoded, never uploaded and never
//! multiplied — which is what makes honouring the cut cost nothing on either
//! backend.
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
use crate::ops::Projection;
use crate::profile::{self, Op};

/// What a caller wants of the back of the model, which is not the same two
/// things every time.
///
/// **The logits are wanted for the rows a token is taken from and the normed
/// state for the rows something is chained from**, and those are different
/// rows: a prefill takes one token out of a prompt's worth of positions, and a
/// speculative round hands every row it committed to the heads. So a caller
/// says which of the two it is asking for rather than being handed both and
/// throwing one away — where a backend that runs the tail on a device would be
/// throwing away a dispatch and a crossing rather than a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tail {
    /// How many of the pass's last rows a token is taken from.
    pub block: usize,
    /// Whether the `[rows, hidden]` normed state is wanted beside them.
    ///
    /// Only a proposer reads it — an MTP head is chained from the model's own
    /// final norm — so a run that speculates nothing asks for none and the
    /// norm behind it is never dispatched. See
    /// [`Round::hidden`](crate::generate::Round::hidden).
    pub chained: bool,
    /// Whether the `[block, vocab]` logits are wanted beside the ids taken from
    /// them.
    ///
    /// **The ids are not optional and these are**, which is the shape a
    /// generation actually has: what a loop does with a row of logits is take
    /// its argmax, and both backends answer with the argmax. So a run that only
    /// decodes asks for none of them and 800 KB a row stays where it was
    /// written — where a caller comparing this port against the reference wants
    /// the distribution itself and says so.
    pub logits: bool,
}

/// What the back of the model answers with, wherever it ran.
///
/// One value rather than two calls because the two halves come out of one
/// command buffer on a backend that holds the tail: the same normed rows the
/// projection reads are the rows a head is chained from, and asking for them
/// separately is the round trip this exists to remove.
#[derive(Debug, Clone, PartialEq)]
pub struct Tailed {
    /// `[rows, hidden]` through the final norm, and empty where
    /// [`Tail::chained`] did not ask for it.
    pub normed: Vec<f32>,
    /// `[block, vocab]`, the muP divide and the projection behind that norm.
    pub logits: Vec<f32>,
    /// The id a greedy sampler takes from each of those rows.
    ///
    /// **Beside the logits rather than derived from them by the caller**, and
    /// the reason is where the argmax runs. On this side it is a pass over the
    /// row that produced it and nothing is gained by naming it here; on a
    /// backend that holds the tail it is a dispatch in the command buffer that
    /// wrote the row, and what crosses back is four bytes where the row is
    /// 800 KB. Both fill it, so the loop above reads one field rather than
    /// asking which backend answered — see
    /// [`Generator::picks`](crate::Generator::picks), which is this on the CPU.
    pub picks: Vec<usize>,
}

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

    /// `logits_mup_width_multiplier`, which the hidden state is divided by
    /// before it is projected.
    ///
    /// Asked for by a backend that means to run the divide where the norm in
    /// front of it ran — see [`Tail`]. It is the multiplier rather than its
    /// reciprocal because what the reference does is a division, and whether
    /// that can be folded into something else without moving a bit is the
    /// backend's question to answer about its own arithmetic.
    pub fn mup(&self) -> f32 {
        self.mup
    }

    /// Whether `weights` is this head's `[vocab, hidden]` projection, which the
    /// two shapes it reports are the whole of the answer to: a projection to the
    /// padded width is the truncation left undone, and one from a different
    /// width is another tensor entirely.
    ///
    /// Public because it is asked twice — once by [`LmHead::forward`], and once
    /// by whoever chooses the backend, so that a projection built to the wrong
    /// shape is refused when it is handed over rather than one prefill later.
    pub fn expects(&self, weights: &dyn Projection) {
        assert_eq!(
            weights.in_dim(),
            self.hidden,
            "the head projects from {}",
            self.hidden
        );
        assert_eq!(
            weights.out_dim(),
            self.vocab,
            "the head projects to {} logits",
            self.vocab
        );
    }

    /// `[tokens, hidden]` in, `[tokens, vocab]` out.
    pub fn forward(&self, h: &[f32], weights: &dyn Projection) -> Vec<f32> {
        assert_eq!(
            h.len() % self.hidden,
            0,
            "{} values are not whole rows of {}",
            h.len(),
            self.hidden
        );
        self.expects(weights);

        let scaled = profile::timed(Op::Sample, || {
            h.iter().map(|x| x / self.mup).collect::<Vec<f32>>()
        });
        weights.forward(&scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::fixture;
    use crate::ops::{DenseProjection, linear};

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
    fn rows() -> Vec<f32> {
        [
            [-0.1, -0.2, -0.3, -0.4],
            [-0.2, -0.1, -0.1, -0.1],
            [-0.3, -0.3, -0.3, -0.3],
        ]
        .into_iter()
        .flatten()
        .chain(std::iter::repeat_n(0.0, PADDING * HIDDEN))
        .collect()
    }

    /// The head's weights cut where a vocabulary of `vocab` ends, which is what
    /// a backend builds its projection to and what the head then checks.
    fn projection(rows: &[f32], vocab: usize) -> DenseProjection<'_> {
        DenseProjection::new(HIDDEN, &rows[..vocab * HIDDEN])
    }

    /// One position through a head and the weights it was cut for.
    fn logits(rows: &[f32], head: &LmHead) -> Vec<f32> {
        head.forward(&hidden_state(), &projection(rows, head.vocab()))
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
        let rows = rows();
        let divided = logits(&rows, &LmHead::new(HIDDEN, VOCAB, MUP));
        let undivided = logits(&rows, &LmHead::new(HIDDEN, VOCAB, 1.0));

        assert_eq!(
            divided,
            linear(&hidden_state(), &rows[..VOCAB * HIDDEN], HIDDEN)
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
        let rows = rows();

        let truncated = logits(&rows, &LmHead::new(HIDDEN, VOCAB, MUP));
        assert_eq!(truncated.len(), VOCAB);
        assert!(
            truncated.iter().all(|logit| *logit < 0.0),
            "a position with a positive logit in it would not settle this"
        );
        assert_eq!(argmax(&truncated), 1);

        let untruncated = logits(&rows, &LmHead::new(HIDDEN, VOCAB + PADDING, MUP));
        assert_eq!(untruncated[VOCAB..], [0.0; PADDING]);
        assert_eq!(argmax(&untruncated), VOCAB, "the first padding id");
    }

    #[test]
    fn every_token_gets_its_own_row_of_logits() {
        let rows = rows();
        let head = LmHead::new(HIDDEN, VOCAB, MUP);
        let h = hidden_state();
        let two = head.forward(
            &[h.clone(), h.clone()].concat(),
            &projection(&rows, head.vocab()),
        );

        assert_eq!(two.len(), 2 * VOCAB);
        assert_eq!(two[..VOCAB], two[VOCAB..]);
        assert_eq!(two[..VOCAB], logits(&rows, &head)[..]);
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
        let rows = rows();
        let head = LmHead::new(HIDDEN, VOCAB, MUP);
        head.forward(&hidden_state()[1..], &projection(&rows, head.vocab()));
    }

    /// The truncation, now that it is the projection's shape rather than a loop
    /// bound: a head handed the whole padded table would produce the 966 logits
    /// of exactly zero the case above ranks, and it is refused instead.
    #[test]
    #[should_panic(expected = "the head projects to 3 logits")]
    fn a_projection_that_kept_the_padding_rows_is_refused() {
        let rows = rows();
        LmHead::new(HIDDEN, VOCAB, MUP)
            .forward(&hidden_state(), &projection(&rows, VOCAB + PADDING));
    }

    #[test]
    #[should_panic(expected = "the head projects from 4")]
    fn a_projection_from_the_wrong_width_is_refused() {
        let rows = rows();
        let narrow = DenseProjection::new(HIDDEN / 2, &rows[..VOCAB * HIDDEN]);
        LmHead::new(HIDDEN, VOCAB, MUP).forward(&hidden_state(), &narrow);
    }
}
