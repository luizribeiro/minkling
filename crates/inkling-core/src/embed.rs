//! The front of the model: token ids to the tensor layer 0 consumes.
//!
//! `InklingModel.embed` is two lines — a table lookup, then an RMSNorm the
//! config can switch off — and neither is new arithmetic. What is worth writing
//! down is the table.
//!
//! **The embedding table is quantised.** `embed_tokens.weight` is
//! `[201024, 512]` of `U32` beside `[201024, 128]` of `U8` scales: the same
//! MXFP4 pair as every projection, not the float array a lookup usually slices.
//! Decoded whole it is 3.3 GB, so a row is asked for when a token wants it and
//! dropped again — the bargain [`crate::layer::Experts`] makes for the same
//! reason.
//!
//! **The rows past `unpadded_vocab_size` are not zero.** 200058 of the table's
//! 201024 rows are vocabulary and the other 966 are padding, and every one of
//! them holds small nonzero values: scale bytes 0x70..0x72 against a trained
//! row's 0x7a..0x7d, about 1.5e-4 RMS against 0.17. `lm_head`'s padding at the
//! other end of the model *is* all-zero codes under all-zero scales, which the
//! MXFP4 fixture's `vocab_padding` slice already pins, so the two ends of the
//! model do not agree about what padding means and reading one off the other
//! would be wrong.
//!
//! Nothing here guards against an id landing in that range. The reference does
//! not, and a tokenizer cannot emit one; the point of stating it is that the
//! answer to what such an id would do is neither "nothing" nor "a vector a
//! thousandth the size". `embed_norm` divides by the row's own RMS, which would
//! erase the difference entirely — except that at 1.5e-4 the padding row's mean
//! square is 46 times *below* `rms_norm_eps`, so the epsilon dominates the
//! divide and the row does not normalise to unit RMS. What comes out is about
//! an eighth of a real token's magnitude: attenuated, but well inside the range
//! layer 0 will treat as an ordinary input.
//!
//! Pinned by the trained weight in `reference/fixtures/embed.safetensors`
//! against the recorded `embed_out` and `embed_norm_out` in
//! `layer_activations.safetensors`. The lookup itself needs the real table and
//! is left to `tests/real_checkpoint.rs`.

use crate::ops::rms_norm;

/// `embed_tokens` followed by the optional `embed_norm`.
#[derive(Debug, Clone, Copy)]
pub struct Embed<'a> {
    norm: Option<&'a [f32]>,
    eps: f32,
}

impl<'a> Embed<'a> {
    /// `norm` is `embed_norm`'s weight, absent when the config clears
    /// `use_embed_norm` — which Inkling-Small sets, so absent here means a
    /// different checkpoint rather than a shortcut.
    pub fn new(norm: Option<&'a [f32]>, eps: f32) -> Self {
        Self { norm, eps }
    }

    /// `[tokens]` ids in, `[tokens, hidden]` out.
    ///
    /// The table arrives as a function of an id rather than as a slice: at
    /// Inkling-Small's size it is 3.3 GB of float32, and one forward pass wants
    /// as many rows as it has tokens.
    pub fn forward(&self, ids: &[usize], row: impl Fn(usize) -> Vec<f32>) -> Vec<f32> {
        let h: Vec<f32> = ids.iter().flat_map(|id| row(*id)).collect();
        match self.norm {
            Some(weight) => rms_norm(&h, weight, self.eps),
            None => h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::fixture::{self, ACTIVATIONS, deviation};

    /// `embed_norm`'s trained weight, from `just dump-embed-fixture`.
    const FIXTURE: &str = "embed.safetensors";

    /// One RMSNorm over 4096 recorded values, and still four thousand times
    /// looser than the 1e-6 the synthetic RMSNorm cases hold. The reason is not
    /// the reduction: it is that this norm ran in bfloat16 and **rounds twice**.
    ///
    /// MLX normalises in float32, rounds that intermediate to the input's dtype,
    /// and only then multiplies by the weight and rounds again. Modelling those
    /// two roundings over the recorded `embed_out` reproduces `embed_norm_out`
    /// bit for bit across all 32768 values, which is what identifies the gap;
    /// this port normalises in float32 and rounds nowhere, so it lands within
    /// one bfloat16 quantum of MLX and cannot land closer.
    ///
    /// That quantum is 2^-9 of the tensor's peak, so 4e-3 is two of them.
    /// Measured when this landed: 1.9e-3, which is 0.99 quanta — the bound is a
    /// factor of two above an error that is already at its floor, and a port
    /// that needed more than one quantum would be wrong in some other way.
    /// Against the weakest mutation it has to catch — the norm skipped
    /// altogether — 9.4e-1, two and a half orders of magnitude above it.
    const TOLERANCE: f32 = 4e-3;

    struct Recorded {
        hidden: usize,
        weight: Vec<f32>,
        eps: f32,
        embed_out: Vec<f32>,
        embed_norm_out: Vec<f32>,
    }

    impl Recorded {
        fn load() -> Self {
            let weights = fixture::open(FIXTURE);
            let activations = fixture::open(ACTIVATIONS);
            let of = |ckpt: &Checkpoint, name: &str| fixture::f32s(&fixture::tensor(ckpt, name));
            let embed_out = fixture::tensor(&activations, "embed_out");
            Self {
                hidden: *embed_out.shape().last().expect("embed_out has a last axis"),
                weight: of(&weights, "embed_norm.weight"),
                eps: of(&weights, "rms_norm_eps")[0],
                embed_out: fixture::f32s(&embed_out),
                embed_norm_out: of(&activations, "embed_norm_out"),
            }
        }

        fn tokens(&self) -> Vec<usize> {
            (0..self.embed_out.len() / self.hidden).collect()
        }

        /// The recorded lookup, standing in for the table the checkpoint holds:
        /// `embed_out` is what `embed_tokens` produced for the fixture's ids, so
        /// row `i` of it is the row token `i` looked up.
        fn forward(&self, norm: Option<&[f32]>) -> Vec<f32> {
            Embed::new(norm, self.eps).forward(&self.tokens(), |token| {
                self.embed_out[token * self.hidden..][..self.hidden].to_vec()
            })
        }
    }

    #[test]
    fn embed_norm_reproduces_the_reference() {
        let recorded = Recorded::load();
        let deviation = deviation(
            &recorded.forward(Some(&recorded.weight)),
            &recorded.embed_norm_out,
        );
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    /// `use_embed_norm` is a config flag, so passing the raw embedding through
    /// is a live way to be wrong rather than a hypothetical one. It has to be
    /// caught by the numbers: the norm changes the scale of every row without
    /// changing its shape, so a pass that skipped it still runs and still
    /// generates text.
    #[test]
    fn skipping_embed_norm_changes_the_answer() {
        let recorded = Recorded::load();
        let raw = recorded.forward(None);
        assert_eq!(raw, recorded.embed_out, "with no weight this is the lookup");

        let deviation = deviation(&raw, &recorded.embed_norm_out);
        assert!(
            deviation > TOLERANCE,
            "the unnormalised embedding deviates by only {deviation:e}"
        );
    }
}
