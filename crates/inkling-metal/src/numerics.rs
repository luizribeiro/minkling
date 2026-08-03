//! Which arithmetic a kernel's innermost accumulation is allowed to use.
//!
//! Every kernel in this tree has, until now, been held to one standard: the
//! answer is the CPU path's bit for bit, and the recorded continuation
//! `[656, 13, 623, 180069, 86333, 60500, 220, 23]` has never moved. That
//! standard is what makes a mutation falsifiable — a sweep can say a form is
//! 12% faster and the case beside it can say the form is the same floats — and
//! it is also a ceiling. The structure that carries the reference runtime's 51×
//! on attention is `simdgroup_matrix`, whose `k` dimension is summed in an order
//! the instruction defines and this side does not choose. **A hardware matrix
//! multiply cannot be bit-identical to a `simd_sum` over lanes**, so a kernel
//! built on one is a kernel the standard above rules out before it is written.
//!
//! This is the switch that lets it be written and measured anyway.
//!
//! # Neither of the two is "more accurate"
//!
//! [`Numerics::Reference`] is verifiable: its answer is a fixed order of
//! operations that the CPU path reproduces exactly, so a disagreement anywhere
//! in the engine is a bug with a witness. [`Numerics::Production`] is not — its
//! order is the instruction's — and that is the whole of the difference. Both
//! sum the same exact products of the same exactly-decoded values; a matrix
//! instruction's accumulation is not less precise than a lane-strided one and on
//! a long reduction is usually a little more so. What is lost is not accuracy,
//! it is the *oracle*: nothing on the other side of this flag can be checked
//! against a recorded array of bits.
//!
//! So the chain a disagreement is bisected through gains a link. It was
//! CPU → Metal, one arrow, settled by rerunning a command with `--backend cpu`.
//! It is now CPU → Metal under the reference → Metal under the production
//! numerics, and the middle of those three is what says whether a wrong token
//! came from the kernel structure or from the arithmetic inside it. `bench
//! divergence` is the instrument that walks the last arrow: the same prompts
//! through both paths, and where and how often their tokens part company.
//!
//! # What it is allowed to select
//!
//! **The innermost compute and nothing else.** Tiling, submission structure, the
//! grouping, KV handling, `splits_for`, the occupancy declarations — all shared,
//! all exercised whichever way this reads. A kernel behind this flag is a
//! different accumulation over the same dispatch, taking the same bindings from
//! the same encoder, at the same shapes the same predicates chose.
//!
//! That bound is the point rather than a tidiness preference. `attention.rs` and
//! `matmul.rs` are the two most-edited files in this repo; a fork of the engine
//! at any level above the accumulate would have two of everything that has moved
//! in the last four milestones — the occupancy turns, `splits_for`, the
//! grouping's two ends — and would rot inside two more.

/// Which arithmetic the two dominant kernels accumulate with.
///
/// **The default is [`Numerics::Reference`] and that is not a placeholder.**
/// Nothing changes for a caller who does not ask: whatever entries stand behind
/// this flag are to be compiled under [`Numerics::Production`] and nowhere else,
/// so a reference run has them neither in its pipeline cache nor in its
/// dispatches. **As this lands, nothing stands behind it at all** — the two
/// answer the same bits because they are the same three kernels — and the
/// commit that puts a kernel there is the one that makes the sentence above
/// bite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Numerics {
    /// Every product summed in an order this side picked and the CPU path
    /// reproduces: a lane-strided walk of the reduction under one `simd_sum`.
    /// **Bit-for-bit checkable, and every gated case in this tree pins it.**
    #[default]
    Reference,
    /// The reduction carried by `simdgroup_matrix`, whose summation order the
    /// instruction defines. Faster or not — that is what a milestone measures —
    /// and never bit-comparable to the above.
    Production,
}

impl Numerics {
    /// Whether a kernel behind this flag should be compiled at all.
    ///
    /// Read at compile time rather than at dispatch time, which is what keeps a
    /// reference run free of the production entries entirely rather than merely
    /// free of dispatching them.
    pub fn is_production(self) -> bool {
        self == Self::Production
    }

    /// The word a command line spells it with, and the word a report prints
    /// back. One reading of the two names, so that a flag parsed here and echoed
    /// there cannot drift into two spellings.
    pub const fn named(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Production => "production",
        }
    }

    /// The flag's value as a command line gives it.
    pub fn parse(name: &str) -> Option<Self> {
        [Self::Reference, Self::Production]
            .into_iter()
            .find(|numerics| numerics.named() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_reference() {
        assert_eq!(Numerics::default(), Numerics::Reference);
        assert!(!Numerics::default().is_production());
    }

    #[test]
    fn every_numerics_parses_back_from_the_name_it_prints() {
        for numerics in [Numerics::Reference, Numerics::Production] {
            assert_eq!(Numerics::parse(numerics.named()), Some(numerics));
        }
    }

    #[test]
    fn a_word_that_is_neither_is_no_numerics() {
        for word in ["", "metal", "fast", "Reference", "production "] {
            assert_eq!(Numerics::parse(word), None, "{word:?}");
        }
    }
}
