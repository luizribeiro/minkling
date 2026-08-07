//! The `stop` a client sends, matched against the text it will see.
//!
//! Without it a model that runs on costs a client tokens it pays for and
//! returns junk past the answer, and every OpenAI client sends one. What makes
//! it more than a `contains` is that this server streams: a frame is gone as
//! soon as it is written, so a stop sequence has to be recognised *before* the
//! text that would spell it goes out rather than after.
//!
//! # Over decoded text, because a stop sequence is not a token
//!
//! A client writes `"\nUser:"` and the vocabulary has no such id. That string
//! arrives spread over however many tokens the merges happened to produce, and
//! nothing says a boundary falls where the client's string starts or ends — so
//! matching on ids would answer a question nobody asked. What is matched here is
//! the text [`Utf8Stream`](inkling_core::Utf8Stream) has already decoded, which
//! is the only place the sequence exists at all.
//!
//! # Against what the client sees, which is `content`
//!
//! **This is a decision and the two answers differ.** What the model emits
//! carries `<|content_thinking|>`, `<|content_text|>` and `<|end_message|>`, and
//! [`crate::chat`] strips all three and splits the thinking channel out into
//! `reasoning_content`. A client's `stop` is written against the field it
//! renders, so it is matched against **`content` and nothing else**:
//!
//! - **Not the markers.** A client that has never seen `<|content_text|>` cannot
//!   have written a `stop` that means to match one, and a sequence that spanned
//!   a marker would fire on text the client would have read as contiguous.
//! - **Not `reasoning_content`.** The model opens a thinking channel unprompted
//!   and reasons in the client's own words, so a `stop` of `"\nUser:"` would
//!   fire inside the reasoning of half the requests that carry one — cutting the
//!   turn off before the answer had started. A client that wanted that would have
//!   no way to ask for the other.
//!
//! What that costs is a turn that never leaves the thinking channel, which no
//! `stop` can end and the budget ends instead. That is the right failure: the
//! answer the client's rule was written about was never reached.
//!
//! **Content is matched as one string across a thinking interruption.** The
//! model can emit content, open a thinking channel, and return to content inside
//! one turn, and what the client's `content` holds is the two pieces joined —
//! so what is held back is held across the interruption rather than flushed at
//! it, because the join is where the client would see the sequence.
//!
//! # Nothing is emitted that would have to be retracted
//!
//! A stop that straddles a token boundary is only recognisable once the second
//! half arrives, and by then the first half has already been framed — unless it
//! is held. So text whose tail could still turn out to begin a stop sequence is
//! kept until the next token says whether it did: at most one byte short of the
//! longest sequence, released the moment the ambiguity resolves, and released
//! whole at the end of a generation nothing matched in.
//!
//! What that costs is that the last few bytes of a reply arrive one token late.
//! The alternative is a frame a client has already rendered and this would have
//! to take back, which the wire has no shape for.

/// The sequences a request named, and the text held back because it might yet
/// begin one.
#[derive(Debug, Default)]
pub struct Stops {
    sequences: Vec<String>,
    held: String,
}

/// What of the text handed over may be shown, and whether the turn is over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Taken {
    /// Text safe to frame now, which is empty where all of it is still
    /// ambiguous.
    pub shown: String,
    /// Whether a sequence matched, which ends the turn and is reported as
    /// `finish_reason: "stop"`.
    pub struck: bool,
}

/// The most sequences a request may name, which is OpenAI's own limit.
///
/// Refused rather than truncated: a client that named five and had the fifth
/// dropped would get a reply running past a stop it asked for, which is the
/// failure `stop` exists to prevent.
pub const MOST_SEQUENCES: usize = 4;

impl Stops {
    /// A request's sequences, or nothing at all for a request that named none —
    /// which is the path every request took before this existed and the one that
    /// holds nothing back.
    pub fn new(sequences: Vec<String>) -> Self {
        Self {
            sequences,
            held: String::new(),
        }
    }

    /// Whether there is anything to match, which is what lets a request that
    /// named no `stop` pay nothing for the ones that did.
    pub fn idle(&self) -> bool {
        self.sequences.is_empty()
    }

    /// `text` added to what was held, and what of the two may now be shown.
    ///
    /// The match is searched for in the join rather than in `text`, which is the
    /// whole point: a sequence spread over two tokens exists in neither of them.
    pub fn take(&mut self, text: &str) -> Taken {
        if self.idle() {
            return Taken {
                shown: text.to_string(),
                struck: false,
            };
        }
        self.held.push_str(text);

        if let Some(at) = self.earliest() {
            self.held.truncate(at);
            return Taken {
                // The stop text itself is never shown, and neither is anything
                // behind it: the client asked for the reply up to here.
                shown: std::mem::take(&mut self.held),
                struck: true,
            };
        }

        let ambiguous = self.ambiguous();
        let shown = self.held.drain(..self.held.len() - ambiguous).collect();
        Taken {
            shown,
            struck: false,
        }
    }

    /// Everything still held, which a generation that ended without a match owes
    /// the client.
    ///
    /// Only ever a proper prefix of some sequence — the ambiguity the last token
    /// left — and there is nothing after it to resolve it, so it was text all
    /// along.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Where the first sequence to match starts, which is where the reply ends.
    ///
    /// The earliest rather than the longest: two sequences that both match are
    /// two rules the client asked for, and the one that fires is the one that
    /// fired first.
    fn earliest(&self) -> Option<usize> {
        self.sequences
            .iter()
            .filter_map(|sequence| self.held.find(sequence.as_str()))
            .min()
    }

    /// How many bytes off the end of the held text could still be the beginning
    /// of a sequence, and so may not be shown yet.
    ///
    /// Walked from the longest suffix down, so the first hit is the most that
    /// has to be kept. Char boundaries rather than bytes, because half a
    /// character is not text a frame may carry — and a sequence is valid UTF-8,
    /// so a suffix that begins mid-character can be no prefix of one anyway.
    fn ambiguous(&self) -> usize {
        for (at, _) in self.held.char_indices() {
            let suffix = &self.held[at..];
            let begins = self
                .sequences
                .iter()
                .any(|sequence| sequence.len() > suffix.len() && sequence.starts_with(suffix));
            if begins {
                return self.held.len() - at;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(sequences: &[&str]) -> Stops {
        Stops::new(sequences.iter().map(|word| word.to_string()).collect())
    }

    /// Everything a run of tokens showed the client, and whether it was cut
    /// short. What a client's `content` field ends up holding, in other words,
    /// which is the only thing any of this is about.
    fn shown(stops: &mut Stops, tokens: &[&str]) -> (String, bool) {
        let mut shown = String::new();
        for token in tokens {
            let taken = stops.take(token);
            shown.push_str(&taken.shown);
            if taken.struck {
                return (shown, true);
            }
        }
        shown.push_str(&stops.finish());
        (shown, false)
    }

    /// A request that named no `stop` is the path every request took before this
    /// existed: nothing matched, nothing held, and every token shown as it
    /// arrives.
    #[test]
    fn a_request_that_named_no_stop_holds_nothing_back() {
        let mut stops = stops(&[]);
        assert!(stops.idle());
        for token in ["Hel", "lo", "."] {
            let taken = stops.take(token);
            assert_eq!(taken.shown, token, "{token} was held");
            assert!(!taken.struck);
        }
        assert_eq!(stops.finish(), "");
    }

    /// **The case the whole module is for.** No token spells `\nUser:` — it
    /// arrives over four of them — so a match on ids finds nothing and a match
    /// on the decoded join finds it.
    #[test]
    fn a_stop_that_straddles_token_boundaries_is_matched_in_the_join() {
        let (shown, struck) = shown(
            &mut stops(&["\nUser:"]),
            &["The", " answer.", "\nUs", "er:"],
        );
        assert!(struck, "the sequence spans four tokens and none spells it");
        assert_eq!(shown, "The answer.");
    }

    /// The same sequence arriving a byte at a time, which is the worst split
    /// there is. Nothing about the matching may depend on where the merges fell.
    #[test]
    fn a_stop_is_matched_however_the_tokens_fell() {
        let split: Vec<String> = "The answer.\nUser:"
            .chars()
            .map(|char| char.to_string())
            .collect();
        let tokens: Vec<&str> = split.iter().map(String::as_str).collect();

        let (shown, struck) = shown(&mut stops(&["\nUser:"]), &tokens);
        assert!(struck);
        assert_eq!(shown, "The answer.");
    }

    /// **The stop text never reaches the client, and neither does anything
    /// behind it.** A client that asked for the reply up to a sequence is asking
    /// not to be shown the sequence.
    #[test]
    fn the_stop_text_is_not_in_the_output() {
        let (shown, struck) = shown(&mut stops(&["END"]), &["Done. ", "END", " and more"]);
        assert!(struck);
        assert_eq!(shown, "Done. ");
        assert!(!shown.contains("END"), "{shown:?}");
        assert!(!shown.contains("and more"), "{shown:?}");
    }

    /// **Nothing is shown that a match would have to take back.** The frame is
    /// gone the moment it is written, so text whose tail could still begin a
    /// sequence is held until the next token resolves it — and this asserts the
    /// holding rather than only the outcome, because the outcome is the same
    /// either way once the reply is added up.
    #[test]
    fn a_partial_match_is_held_back_rather_than_shown_and_retracted() {
        let mut stops = stops(&["\nUser:"]);
        assert_eq!(stops.take("Answer.").shown, "Answer.");

        // Every one of these could still turn into the sequence, so none of them
        // may go out.
        for token in ["\n", "Us", "er"] {
            assert_eq!(stops.take(token).shown, "", "{token:?} was shown too early");
        }
        assert!(stops.take(":").struck);
    }

    /// The other half of holding back: an ambiguity the next token settles the
    /// other way is released, whole and in order, rather than lost.
    #[test]
    fn text_that_only_looked_like_a_stop_is_released_once_it_is_settled() {
        let mut stops = stops(&["\nUser:"]);
        assert_eq!(stops.take("Answer.\nUse").shown, "Answer.");
        assert_eq!(stops.take("ful.").shown, "\nUseful.");
        assert_eq!(stops.finish(), "");
    }

    /// A generation that ended with an ambiguity outstanding owes the client the
    /// text, not silence. There is nothing left to resolve it, so it was text all
    /// along — and a reply quietly missing its last few bytes is worse than a
    /// stop that did not fire.
    #[test]
    fn text_held_at_the_end_of_a_generation_still_reaches_the_client() {
        let (shown, struck) = shown(&mut stops(&["\nUser:"]), &["Answer.", "\nUser"]);
        assert!(!struck);
        assert_eq!(shown, "Answer.\nUser");
    }

    /// A sequence wholly inside one token is the ordinary case, and it still
    /// cuts at the sequence rather than at the token.
    #[test]
    fn a_stop_inside_a_single_token_cuts_at_the_sequence() {
        let (shown, struck) = shown(&mut stops(&["END"]), &["before END after"]);
        assert!(struck);
        assert_eq!(shown, "before ");
    }

    /// Several sequences, and the one that fires is the one that fires first —
    /// which is not the one named first, and not the longest.
    #[test]
    fn the_earliest_of_several_sequences_is_the_one_that_ends_the_reply() {
        let (shown, struck) = shown(&mut stops(&["ZZZ", "b"]), &["a", "b", "c ZZZ"]);
        assert!(struck);
        assert_eq!(shown, "a");
    }

    /// Two sequences held at once: text ambiguous against either is kept until
    /// both are settled, and the amount kept is the longer one's.
    #[test]
    fn text_ambiguous_against_any_sequence_is_held_for_all_of_them() {
        let mut stops = stops(&["abcd", "xy"]);
        assert_eq!(stops.take("hello abc").shown, "hello ");
        assert_eq!(stops.take("!").shown, "abc!");
        assert_eq!(stops.take("ab").shown, "");
        assert!(stops.take("cd").struck);
    }

    /// **A sequence must not be cut apart mid-character.** A suffix that begins
    /// inside a character is no prefix of any sequence — sequences are text —
    /// and a frame carrying half a character is a frame a client renders as a
    /// replacement.
    #[test]
    fn a_multi_byte_character_is_never_split_by_the_hold_back() {
        let mut held = stops(&["日本"]);
        let taken = held.take("見て日");
        assert_eq!(taken.shown, "見て");
        assert!(!taken.struck);
        assert!(held.take("本").struck);
    }

    /// The same, where the character is four bytes and the sequence begins with
    /// it — the split a byte-level vocabulary produces whenever no merge covers
    /// an emoji.
    #[test]
    fn a_sequence_beginning_with_a_wide_character_still_matches() {
        let (shown, struck) = shown(&mut stops(&["🙂!"]), &["Café, ", "🙂", "!"]);
        assert!(struck);
        assert_eq!(shown, "Café, ");
    }

    /// A sequence the reply never contains changes nothing about the reply.
    #[test]
    fn a_sequence_that_never_matches_leaves_the_reply_whole() {
        let (shown, struck) = shown(&mut stops(&["\nUser:"]), &["Just ", "an ", "answer."]);
        assert!(!struck);
        assert_eq!(shown, "Just an answer.");
    }

    /// A reply that *is* the sequence, from its first byte, shows nothing at all
    /// — which is a client's rule firing immediately rather than a bug.
    #[test]
    fn a_reply_that_begins_with_the_sequence_shows_nothing() {
        let (shown, struck) = shown(&mut stops(&["END"]), &["END", "more"]);
        assert!(struck);
        assert_eq!(shown, "");
    }
}
