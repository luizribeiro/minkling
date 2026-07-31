//! Assembling text from the bytes tokens carry, as they arrive.
//!
//! Inkling's vocabulary is byte-level: a piece is not text but a spelling of
//! the bytes that piece contributes, in an alphabet where every byte has a
//! printable character. Nothing makes those bytes a whole character. A
//! character wider than one byte is split across tokens whenever no merge
//! covers it — which is ordinary for anything the training data saw rarely —
//! so decoding each token on its own and concatenating puts a replacement
//! character wherever a character crossed a token boundary.
//!
//! [`Utf8Stream`] is what keeps a stream equal to decoding the whole sequence
//! at once: bytes that do not yet complete a character are held back until the
//! bytes that finish them arrive.

/// Whether a byte spells itself in the byte-level alphabet.
///
/// The printable Latin-1 bytes do. The other 68 are whitespace, control
/// characters or unassigned code points, and the alphabet lifts them out of
/// the way rather than spell a piece with a literal newline.
const fn spells_itself(byte: u8) -> bool {
    matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff)
}

const LIFTED_COUNT: usize = 68;

/// The bytes that do not spell themselves, in ascending order. The alphabet
/// gives them U+0100 and up in exactly this order.
const LIFTED: [u8; LIFTED_COUNT] = lifted_bytes();

const fn lifted_bytes() -> [u8; LIFTED_COUNT] {
    let mut lifted = [0; LIFTED_COUNT];
    let (mut byte, mut found) = (0, 0);
    while byte < 256 {
        if !spells_itself(byte as u8) {
            lifted[found] = byte as u8;
            found += 1;
        }
        byte += 1;
    }
    assert!(found == LIFTED_COUNT);
    lifted
}

/// The byte a character of the alphabet stands for, or `None` for a character
/// that is not in it.
pub fn char_byte(spelling: char) -> Option<u8> {
    let code = spelling as u32;
    if code < 0x100 && spells_itself(code as u8) {
        return Some(code as u8);
    }
    LIFTED.get(code.checked_sub(0x100)? as usize).copied()
}

/// The bytes a vocabulary piece contributes, or `None` if it is spelled with
/// anything outside the alphabet.
///
/// A special token is stored as literal text rather than as a byte-level
/// spelling, but every character Inkling spells one with — `<`, `|`, letters,
/// `_` — stands for itself here, so one mapping serves both.
pub fn piece_bytes(piece: &str) -> Option<Vec<u8>> {
    piece.chars().map(char_byte).collect()
}

/// Text assembled from bytes that arrive a few at a time.
///
/// [`push`](Self::push) returns the text those bytes completed, which is empty
/// for as long as a character is still missing bytes. [`finish`](Self::finish)
/// gives up on whatever is left, exactly as decoding truncated bytes in one go
/// would: one replacement character.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    /// A prefix of one character, never more: everything a character completes
    /// leaves on the next push.
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut text = String::new();
        loop {
            let error = match std::str::from_utf8(&self.pending) {
                Ok(complete) => {
                    text.push_str(complete);
                    self.pending.clear();
                    return text;
                }
                Err(error) => error,
            };
            let valid = error.valid_up_to();
            text.push_str(std::str::from_utf8(&self.pending[..valid]).expect("valid up to here"));
            let Some(invalid) = error.error_len() else {
                self.pending.drain(..valid);
                return text;
            };
            // Bytes that no continuation can rescue, which a lossy decode of
            // the whole sequence would replace one run at a time.
            text.push(char::REPLACEMENT_CHARACTER);
            self.pending.drain(..valid + invalid);
        }
    }

    /// The text of an incomplete character left at the end of a stream.
    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::TokenizerFixture;

    /// Every character of the alphabet stands for a different byte, and
    /// between them they stand for all 256.
    #[test]
    fn the_alphabet_spells_every_byte_exactly_once() {
        let alphabet = (0x21..=0x7e)
            .chain(0xa1..=0xac)
            .chain(0xae..=0xff)
            .chain(0x100..0x100 + LIFTED_COUNT as u32);

        let mut spelled = [false; 256];
        for code in alphabet {
            let spelling = char::from_u32(code).expect("a code point");
            let byte = char_byte(spelling).expect("in the alphabet");
            assert!(!spelled[byte as usize], "{spelling:?} respells {byte:#04x}");
            spelled[byte as usize] = true;
        }
        assert!(spelled.iter().all(|&hit| hit), "not every byte is spelled");
    }

    #[test]
    fn characters_outside_the_alphabet_spell_no_byte() {
        for outside in [' ', '\n', '\u{ad}', '\u{144}', '日', '\u{0}'] {
            assert_eq!(char_byte(outside), None, "{outside:?}");
        }
    }

    /// A piece is the bytes it spells, not the characters it is written with:
    /// `æĹ¥` is three bytes of one character, and a special token is its own
    /// literal text.
    #[test]
    fn a_piece_spells_the_bytes_it_carries() {
        assert_eq!(piece_bytes("æĹ¥"), Some("日".as_bytes().to_vec()));
        assert_eq!(piece_bytes("Ġthe"), Some(b" the".to_vec()));
        assert_eq!(
            piece_bytes("<|content_thinking|>"),
            Some(b"<|content_thinking|>".to_vec())
        );
        assert_eq!(piece_bytes("日"), None, "text is not a byte-level spelling");
    }

    fn token_bytes(case: &crate::fixture::TokenizerCase) -> Vec<Vec<u8>> {
        case.pieces
            .iter()
            .map(|piece| piece_bytes(piece).expect("the fixture's pieces are byte-level"))
            .collect()
    }

    /// The reference's streaming detokenizer and this one surface the same
    /// text after the same token — including nothing at all while a character
    /// is still incomplete, and a special token rendered where it arrived.
    #[test]
    fn streaming_a_case_matches_what_the_reference_surfaced_per_token() {
        let fixture = TokenizerFixture::load();
        for (name, case) in &fixture.cases {
            let mut stream = Utf8Stream::new();
            let mut segments: Vec<String> = token_bytes(case)
                .iter()
                .map(|bytes| stream.push(bytes))
                .collect();
            segments.push(stream.finish());

            assert_eq!(segments, case.segments, "{name}");
            assert_eq!(segments.concat(), case.text, "{name}");
        }
    }

    /// That the case above is worth holding: the same tokens decoded one at a
    /// time and concatenated, which is what a stream without a buffer does and
    /// is fluent right up until a character spans tokens.
    #[test]
    fn decoding_each_token_alone_mangles_a_split_character() {
        let fixture = TokenizerFixture::load();
        let case = fixture.case("split_characters");

        let naive: String = token_bytes(case)
            .iter()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .collect();

        assert_ne!(naive, case.text);
        assert!(naive.contains(char::REPLACEMENT_CHARACTER), "{naive:?}");
    }

    /// Where the bytes are cut apart cannot change the text they make, which
    /// is the whole claim a streaming decoder makes.
    #[test]
    fn any_splitting_of_the_same_bytes_decodes_the_same() {
        let fixture = TokenizerFixture::load();
        let bytes: Vec<u8> = token_bytes(fixture.case("turn")).concat();

        for cut in 0..=bytes.len() {
            let mut stream = Utf8Stream::new();
            let mut text = stream.push(&bytes[..cut]);
            text.push_str(&stream.push(&bytes[cut..]));
            text.push_str(&stream.finish());
            assert_eq!(text, fixture.case("turn").text, "cut at {cut}");
        }
    }

    /// Bytes that are no character at all still decode the way decoding them
    /// in one go does, rather than stall the stream waiting for a completion
    /// that cannot come.
    #[test]
    fn invalid_bytes_decode_as_a_lossy_decode_of_the_whole_does() {
        let cases: [&[u8]; 5] = [
            b"\xff",
            b"a\xffb",
            b"\xe6\x97",             // truncated
            b"\xe6\x97\x41",         // abandoned halfway
            b"\xf0\x9f\x99\x82\xc3", // complete, then truncated
        ];
        for bytes in cases {
            for cut in 0..=bytes.len() {
                let mut stream = Utf8Stream::new();
                let mut text = stream.push(&bytes[..cut]);
                text.push_str(&stream.push(&bytes[cut..]));
                text.push_str(&stream.finish());
                assert_eq!(
                    text,
                    String::from_utf8_lossy(bytes),
                    "{bytes:?} cut at {cut}"
                );
            }
        }
    }
}
