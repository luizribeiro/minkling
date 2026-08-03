//! What this repo's own measurements are taken over.
//!
//! Not engine code, and here anyway. **Acceptance is a property of the text and
//! not of the engine** — the study measured 99.7% at the first head on
//! enumeration against 44.9% on prose — so two measurements of the heads taken
//! over two prompts are two measurements, however alike the tables look. The
//! timing tier and `just bench` both quote acceptance and both have to be
//! quoting it about the same workload, and a constant copied into each of them
//! is a constant that can drift in one.
//!
//! What is not here is anything a measurement *decides*: how many pairs, how the
//! ranges are compared, what a table prints. Those belong to whoever is
//! measuring. This is the input.

/// The prompt every multi-token prediction figure in this repo is taken over.
///
/// Structured rather than prose, and enumeration rather than either: the
/// acceptance study measured six regimes and found the spread between them
/// larger than the spread between depths, so what a figure taken here says is
/// "on text like this" and nothing wider.
///
/// **It is a file rather than a literal because the other engine reads it too.**
/// A cross-engine sitting is only a comparison of engines if both were given the
/// same tokens, and `reference/scripts/bench_engines.py` reads these same bytes
/// — where a copy on each side is a copy that can drift in one. The trailing
/// newline `end-of-file-fixer` insists on is not part of the prompt.
pub const STRUCTURED_PROMPT: &str = include_str!("workload.txt").trim_ascii_end();

/// The (prompt, generated) pairs a cross-engine sitting is taken over.
///
/// **This is the number a user feels**, and prefill and decode trade against
/// each other inside it: a long prompt is where this engine loses and a long
/// generation is where it wins. So the pairs are chosen to straddle that — the
/// three prompt lengths this repo quotes at one generation length, and the
/// shortest prompt again at four times the generation.
pub const REALISTIC: [(usize, usize); 4] = [(97, 128), (385, 128), (769, 128), (97, 512)];

/// How many tokens a decode figure is the mean of, and how many each depth of a
/// sweep decodes.
pub const DECODED: usize = 64;

/// The speculation depth this repo's own sweep says pays best, and so the depth
/// a cross-engine table quotes beside `k = 0`.
///
/// Measured rather than derived — see the sweep under "Sampling on the device" —
/// so a sitting that moves it moves this. **And moving it means moving
/// `reference/scripts/bench_engines.py`'s own default with it**, since the two
/// arms of a cross-engine sitting name their rows after their own depth and the
/// harness refuses arms whose rows do not line up.
pub const BEST: usize = 2;

/// How deep a sweep of real generations goes.
///
/// Four, where a verify block is priced to eight: the study's pooled optimum was
/// 2 and its deepest paying depth 6, and every depth here is a whole generation
/// rather than a repeat of one block.
pub const SWEPT: usize = 4;

/// `ids` repeated up to `tokens` and cut there, which is how a prefill of a
/// given length gets a prompt.
///
/// Real ids repeated rather than one id repeated, because which experts a token
/// routes to is decided by the token: a prompt of one id would send every row of
/// every bank through the same six of 256 and measure a stack nobody runs.
pub fn tiled(ids: &[usize], tokens: usize) -> Vec<usize> {
    ids.iter().copied().cycle().take(tokens).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cut where it was asked for and not at the end of a repeat, so that a
    /// prefill of 97 tokens is 97 tokens and not 96 or 104.
    #[test]
    fn tiling_repeats_the_prompt_and_cuts_where_the_length_says() {
        assert_eq!(tiled(&[1, 2, 3], 7), [1, 2, 3, 1, 2, 3, 1]);
        assert_eq!(tiled(&[1, 2, 3], 3), [1, 2, 3]);
        assert_eq!(tiled(&[1, 2, 3], 2), [1, 2]);
        assert!(tiled(&[1, 2, 3], 0).is_empty());
    }

    /// A prompt with no ids in it cannot be tiled to any length, and a `cycle`
    /// over nothing yields nothing rather than looping — which is the one way
    /// this could have hung.
    #[test]
    fn tiling_nothing_is_nothing() {
        assert!(tiled(&[], 8).is_empty());
    }

    /// The prompt is a file, and a file is edited by tools that do not know what
    /// is in it: `end-of-file-fixer` appends a newline and
    /// `trim-trailing-whitespace` takes spaces off a line. A prompt that gained
    /// either would tokenize differently and quietly move every acceptance
    /// figure this repo has.
    #[test]
    fn the_prompt_carries_no_whitespace_from_the_file_it_is_held_in() {
        assert!(!STRUCTURED_PROMPT.contains('\n'), "{STRUCTURED_PROMPT}");
        assert!(!STRUCTURED_PROMPT.contains("  "), "{STRUCTURED_PROMPT}");
        assert!(
            STRUCTURED_PROMPT.starts_with("<|message_user|><|content_text|>Count from 1 to 30."),
            "{STRUCTURED_PROMPT}"
        );
        assert!(
            STRUCTURED_PROMPT.ends_with("No commentary.<|end_message|><|message_model|>"),
            "{STRUCTURED_PROMPT}"
        );
    }

    /// A cross-engine sitting says where prefill and decode cross over, and it
    /// can only say it from pairs that fall either side: one prompt length at
    /// two generation lengths says what a longer generation buys, and three
    /// prompt lengths at one says what a longer prompt costs.
    #[test]
    fn the_realistic_pairs_vary_each_half_against_a_fixed_other() {
        let generated: Vec<usize> = REALISTIC
            .iter()
            .filter(|(prompt, _)| *prompt == 97)
            .map(|(_, generated)| *generated)
            .collect();
        assert!(generated.len() > 1, "{REALISTIC:?}");

        let prompts: Vec<usize> = REALISTIC
            .iter()
            .filter(|(_, generated)| *generated == 128)
            .map(|(prompt, _)| *prompt)
            .collect();
        assert_eq!(prompts, [97, 385, 769]);
    }
}
