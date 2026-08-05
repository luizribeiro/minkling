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
//! # Neither of the first two is "more accurate"
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
//! # The third is, and that is why it is a third word
//!
//! [`Numerics::Rounded`] is the one place in this tree where a product is not
//! exact. It stages the packed matmul's two tiles as 16-bit floats, so
//! `element × scale` carries eleven bits of significand where every other path
//! here carries twenty-four. That is a different claim from the one above and
//! it is deliberately not folded into it: the sentence
//! [`Numerics::Production`] is documented on — that both sides form the same
//! exact products and differ only in the order they are summed — stays true of
//! `production` because this is not `production`.
//!
//! So the chain a disagreement is bisected through gains a link. It was
//! CPU → Metal, one arrow, settled by rerunning a command with `--backend cpu`.
//! It is now CPU → Metal under the reference → Metal behind the flag, and the
//! middle of those three is what says whether a wrong token came from the
//! kernel structure or from the arithmetic inside it. `bench diverge` is the
//! instrument that walks the last arrow: the same prompts through the reference
//! and one word behind the flag, and where and how often their tokens part
//! company. **It is pointed at a word rather than run over all of them**, since
//! "what is a summation order worth" and "what is a rounded operand worth" are
//! two questions and one line cannot answer both.
//!
//! # What it is allowed to select
//!
//! **The innermost compute and nothing else.** Tiling, submission structure, the
//! grouping, KV handling, `splits_for`, the occupancy declarations — all shared,
//! all exercised whichever way this reads. A kernel behind this flag is a
//! different accumulation over the same dispatch, taking the same bindings from
//! the same encoder, at the same shapes the same predicates chose.
//!
//! The operand word moves one more thing and it is inside that bound: the
//! padding between two staged rows, which exists to keep a fragment's eight rows
//! on eight distinct banks and is derived against how wide a staged element is.
//! It is a threadgroup-memory layout the kernel declares, not a decision about
//! what to dispatch — the grid, the block's extents and the predicates are the
//! shipped ones to the value.
//!
//! That bound is the point rather than a tidiness preference. `attention.rs` and
//! `matmul.rs` are the two most-edited files in this repo; a fork of the engine
//! at any level above the accumulate would have two of everything that has moved
//! in the last four milestones — the occupancy turns, `splits_for`, the
//! grouping's two ends — and would rot inside two more.

/// Which arithmetic the two dominant kernels accumulate with.
///
/// **The default is [`Numerics::Reference`] and that is not a placeholder.**
/// Nothing changes for a caller who does not ask: the entries behind this flag
/// are compiled under the two words below it and nowhere else, so a reference
/// run has them neither in its pipeline cache nor in its dispatches — and the
/// source string it hands the compiler is byte for byte the one it handed before
/// the flag existed. `the_reference_source_does_not_carry_the_production_entries`
/// is where that is held.
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
    ///
    /// **Every product it forms is still exact.** A code is one of sixteen
    /// table values and a group scale is a power of two, so `element × scale`
    /// is exact in f32 and this word differs from the reference by summation
    /// order alone.
    Production,
    /// The same reduction over operands rounded to 16 bits, which is the one
    /// word here whose products are not exact.
    ///
    /// **A third word rather than a faster production path**, because what it
    /// gives up is the sentence above rather than the summation order — see the
    /// module documentation. The accumulator stays `simdgroup_float8x8`; what is
    /// narrowed is the two staged tiles the instruction reads.
    Rounded,
}

impl Numerics {
    /// Every word, in the order a report walks them: the checkable one, then
    /// the two behind the flag.
    ///
    /// One reading of the list, so that a case sweeping the words and a table
    /// printing them cannot come to hold different ideas of how many there are.
    pub const EVERY: [Self; 3] = [Self::Reference, Self::Production, Self::Rounded];

    /// Whether a kernel behind this flag should be compiled at all.
    ///
    /// Read at compile time rather than at dispatch time, which is what keeps a
    /// reference run free of the entries behind the flag entirely rather than
    /// merely free of dispatching them.
    pub const fn compiles_the_entries(self) -> bool {
        !matches!(self, Self::Reference)
    }

    /// Whether the packed matmul's staged tiles are narrowed to 16 bits, which
    /// is the one thing [`Numerics::Rounded`] does that
    /// [`Numerics::Production`] does not.
    ///
    /// **Separate from [`Numerics::compiles_the_entries`] on purpose.** The two
    /// words behind the flag select the same entries at the same shapes through
    /// the same predicates, so everything that asks "is this the fast path"
    /// wants the first of these and only the source generator wants this one.
    pub const fn rounds_operands(self) -> bool {
        matches!(self, Self::Rounded)
    }

    /// The word a command line spells it with, and the word a report prints
    /// back. One reading of the three names, so that a flag parsed here and
    /// echoed there cannot drift into two spellings.
    pub const fn named(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Production => "production",
            Self::Rounded => "rounded",
        }
    }

    /// The flag's value as a command line gives it.
    pub fn parse(name: &str) -> Option<Self> {
        Self::EVERY
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
        assert!(!Numerics::default().compiles_the_entries());
        assert!(!Numerics::default().rounds_operands());
    }

    /// **Only the third word rounds anything**, which is what keeps the
    /// property `production` is documented on a fact about this enum rather
    /// than about the two call sites that read it.
    #[test]
    fn the_operand_word_is_the_only_one_that_rounds_a_product() {
        let rounds: Vec<&str> = Numerics::EVERY
            .into_iter()
            .filter(|numerics| numerics.rounds_operands())
            .map(Numerics::named)
            .collect();
        assert_eq!(rounds, ["rounded"]);
        assert!(Numerics::Production.compiles_the_entries());
        assert!(Numerics::Rounded.compiles_the_entries());
    }

    #[test]
    fn every_numerics_parses_back_from_the_name_it_prints() {
        for numerics in Numerics::EVERY {
            assert_eq!(Numerics::parse(numerics.named()), Some(numerics));
        }
    }

    /// Every word is spelled once, so that two of them cannot answer the same
    /// command line.
    #[test]
    fn no_two_words_are_spelled_alike() {
        let mut named: Vec<&str> = Numerics::EVERY.into_iter().map(Numerics::named).collect();
        named.sort_unstable();
        let spelled = named.len();
        named.dedup();
        assert_eq!(named.len(), spelled, "two words share a spelling");
    }

    #[test]
    fn a_word_that_is_none_of_them_is_no_numerics() {
        for word in ["", "metal", "fast", "Reference", "production ", "half"] {
            assert_eq!(Numerics::parse(word), None, "{word:?}");
        }
    }
}
