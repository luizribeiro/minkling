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
pub const STRUCTURED_PROMPT: &str = "<|message_user|><|content_text|>Count from 1 to 30. Put each on its own line in exactly \
     the form 'Line N: N squared is M'. No commentary.<|end_message|><|message_model|>";

/// How many tokens a decode figure is the mean of, and how many each depth of a
/// sweep decodes.
pub const DECODED: usize = 64;

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

    /// The prompt is written across three source lines and is one line of text.
    /// A continuation whose backslash went missing, or whose indentation landed
    /// inside the string, would tokenize differently and quietly move every
    /// acceptance figure this repo has.
    #[test]
    fn the_prompt_carries_no_whitespace_from_the_source_it_is_written_in() {
        assert!(!STRUCTURED_PROMPT.contains('\n'), "{STRUCTURED_PROMPT}");
        assert!(!STRUCTURED_PROMPT.contains("  "), "{STRUCTURED_PROMPT}");
        assert!(
            STRUCTURED_PROMPT.ends_with("No commentary.<|end_message|><|message_model|>"),
            "{STRUCTURED_PROMPT}"
        );
    }
}
