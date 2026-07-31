//! The checkpoint's tokenizer: text to ids, and ids back to text.
//!
//! `tokenizer_config.json` names `TokenizersBackend`, so `tokenizer.json` is
//! what the reference loads and what this loads — the `tiktoken/tokenizer.model`
//! beside it is a second copy of the same vocabulary, id for id, which
//! `just compare-tokenizers` checks rather than assumes.
//!
//! Decoding is this crate's own, over the bytes the vocabulary pieces spell,
//! so that one code path serves both a whole sequence and a stream arriving a
//! token at a time. See [`crate::detokenize`] for why that matters.

use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::detokenize::{Utf8Stream, piece_bytes};

const TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("{} is not a readable tokenizer: {message}", .path.display())]
    Malformed { path: PathBuf, message: String },

    #[error("config.json names no eos_token_id, and the tokenizer names no eos of its own")]
    NoEos,

    #[error("no token with id {0} in this vocabulary")]
    UnknownToken(u32),

    #[error("token {id} is spelled {piece:?}, which is not a byte-level spelling")]
    NotByteLevel { id: u32, piece: String },

    #[error("cannot encode: {0}")]
    Encode(String),
}

/// A loaded `tokenizer.json`, and the id that ends a generation.
///
/// The end-of-sequence id comes from the model config rather than from the
/// tokenizer's own files, and it is the caller's [`Config`] that supplies it.
/// Inkling's checkpoints declare no `eos_token` at all — every special token
/// is listed under `additional_special_tokens`, so anything reading the
/// tokenizer alone finds none, and a generator that then settles for
/// `<|endoftext|>` waits for a token this model does not emit.
pub struct Tokenizer {
    vocabulary: tokenizers::Tokenizer,
    eos: u32,
}

impl Tokenizer {
    /// Opens the `tokenizer.json` of a checkpoint directory.
    pub fn open(dir: &Path, config: &Config) -> Result<Self, TokenizerError> {
        let path = dir.join(TOKENIZER_FILE);
        let vocabulary =
            tokenizers::Tokenizer::from_file(&path).map_err(|err| TokenizerError::Malformed {
                path: path.clone(),
                message: err.to_string(),
            })?;

        let eos = config.eos_token_id.ok_or(TokenizerError::NoEos)?;
        let tokenizer = Self { vocabulary, eos };
        // A config and a vocabulary that disagree about which ids exist are
        // two halves of different checkpoints.
        tokenizer.token_bytes(eos)?;
        Ok(tokenizer)
    }

    /// The id whose arrival ends a generation.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.vocabulary
            .encode(text, false)
            .map(|encoded| encoded.get_ids().to_vec())
            .map_err(|err| TokenizerError::Encode(err.to_string()))
    }

    /// The text a whole sequence of ids spells.
    ///
    /// Special tokens are rendered rather than dropped, which is what the
    /// reference's streaming detokenizer does and the only reading under which
    /// a `<|content_thinking|>` the model emitted unprompted is visible to
    /// whatever has to act on it.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut stream = self.stream();
        let mut text = String::new();
        for &id in ids {
            text.push_str(&stream.push(id)?);
        }
        text.push_str(&stream.finish());
        Ok(text)
    }

    /// Decoding as the tokens arrive, one at a time.
    pub fn stream(&self) -> Detokenizer<'_> {
        Detokenizer {
            tokenizer: self,
            text: Utf8Stream::new(),
        }
    }

    /// How a token is spelled in the vocabulary, which for an ordinary token
    /// is a byte-level spelling rather than the text it stands for.
    pub fn piece(&self, id: u32) -> Option<String> {
        self.vocabulary.id_to_token(id)
    }

    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.vocabulary.token_to_id(piece)
    }

    /// The bytes one token contributes to the text.
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
        let piece = self.piece(id).ok_or(TokenizerError::UnknownToken(id))?;
        piece_bytes(&piece).ok_or(TokenizerError::NotByteLevel { id, piece })
    }
}

/// A decode in progress, fed one token at a time.
///
/// [`push`](Self::push) returns the text that token completed — nothing at all
/// while a character is still missing bytes — so the text a caller has seen
/// after the last token is exactly what [`Tokenizer::decode`] would have made
/// of the whole sequence.
pub struct Detokenizer<'a> {
    tokenizer: &'a Tokenizer,
    text: Utf8Stream,
}

impl Detokenizer<'_> {
    pub fn push(&mut self, id: u32) -> Result<String, TokenizerError> {
        let bytes = self.tokenizer.token_bytes(id)?;
        Ok(self.text.push(&bytes))
    }

    /// The text of a character the stream ended in the middle of.
    pub fn finish(&mut self) -> String {
        self.text.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;

    use tempfile::TempDir;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    use super::*;
    use crate::config::INKLING_SMALL;
    use crate::detokenize::char_byte;
    use crate::fixture::TokenizerFixture;

    /// A checkpoint's tokenizer in miniature, shaped like Inkling's: byte-level
    /// pieces, and special tokens added past the end of the vocabulary. Six
    /// pieces stand in for 199998 — enough to spell `日` a byte at a time —
    /// because what the tests below turn on is which files an id comes from,
    /// not which merges produced it. One piece is the odd one out, spelled in
    /// text rather than in the alphabet, which no tokenizer of this family has
    /// and which decoding therefore has to refuse.
    const TINY_TOKENIZER: &str = r#"{
      "version": "1.0", "truncation": null, "padding": null,
      "added_tokens": [
        {"id": 7, "content": "<|endoftext|>", "single_word": false, "lstrip": false,
         "rstrip": false, "normalized": false, "special": true},
        {"id": 8, "content": "<|content_model_end_sampling|>", "single_word": false,
         "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                        "trim_offsets": true, "use_regex": true},
      "post_processor": null,
      "decoder": {"type": "ByteLevel", "add_prefix_space": true,
                  "trim_offsets": true, "use_regex": true},
      "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                "vocab": {"a": 0, "b": 1, "Ġ": 2, "æ": 3, "Ĺ": 4, "¥": 5, "日": 6},
                "merges": []}
    }"#;

    const ENDOFTEXT: u32 = 7;
    const END_SAMPLING: u32 = 8;
    /// The piece written in text rather than in the alphabet.
    const AS_TEXT: u32 = 6;

    /// A checkpoint directory holding nothing but the tokenizer.
    fn checkpoint() -> TempDir {
        let dir = TempDir::new().expect("a temporary directory");
        fs::write(dir.path().join(TOKENIZER_FILE), TINY_TOKENIZER).expect("writes");
        dir
    }

    fn config(eos: Option<u32>) -> Config {
        let mut config: Config = serde_json::from_str(INKLING_SMALL).expect("parses");
        config.eos_token_id = eos;
        config
    }

    fn open(dir: &TempDir, eos: Option<u32>) -> Result<Tokenizer, TokenizerError> {
        Tokenizer::open(dir.path(), &config(eos))
    }

    /// The alphabet this crate maps pieces out of is the one the vocabulary is
    /// written in, which the loader itself can be asked.
    #[test]
    fn the_byte_level_alphabet_is_the_loaders_own() {
        let ours: HashSet<char> = (0..=0xffffu32)
            .filter_map(char::from_u32)
            .filter(|&c| char_byte(c).is_some())
            .collect();
        let theirs: HashSet<char> = ByteLevel::alphabet().into_iter().collect();
        assert_eq!(ours, theirs);
    }

    /// The one fact the tokenizer's own files cannot supply. `<|endoftext|>` is
    /// what a port that asked them settles for, and it is not this id.
    #[test]
    fn the_eos_id_comes_from_the_config() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        assert_eq!(tokenizer.eos(), END_SAMPLING);
        assert_eq!(
            tokenizer.piece(tokenizer.eos()).as_deref(),
            Some("<|content_model_end_sampling|>")
        );
        assert_eq!(tokenizer.id_of("<|endoftext|>"), Some(ENDOFTEXT));
        assert_ne!(tokenizer.eos(), ENDOFTEXT, "the tokenizer's own guess");
    }

    #[test]
    fn a_config_that_names_no_eos_is_refused() {
        let dir = checkpoint();
        assert!(matches!(open(&dir, None), Err(TokenizerError::NoEos)));
    }

    /// A config paired with the wrong tokenizer, which the real checkpoint's
    /// eos makes of this six-piece one.
    #[test]
    fn an_eos_the_vocabulary_does_not_hold_is_refused() {
        let dir = checkpoint();
        let recorded = TokenizerFixture::load().eos_token_id;
        assert!(matches!(
            open(&dir, Some(recorded)),
            Err(TokenizerError::UnknownToken(id)) if id == recorded
        ));
    }

    /// A vocabulary written in text rather than in the byte-level alphabet is
    /// some other tokenizer, and pieces of it cannot be reassembled as bytes.
    #[test]
    fn a_piece_that_is_not_a_byte_level_spelling_is_refused() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        assert!(matches!(
            tokenizer.token_bytes(AS_TEXT),
            Err(TokenizerError::NotByteLevel { id: AS_TEXT, .. })
        ));
    }

    #[test]
    fn a_directory_without_a_tokenizer_is_refused() {
        let dir = TempDir::new().expect("a temporary directory");
        assert!(matches!(
            Tokenizer::open(dir.path(), &config(Some(END_SAMPLING))),
            Err(TokenizerError::Malformed { .. })
        ));
    }

    #[test]
    fn text_encodes_to_ids_and_decodes_back() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        let ids = tokenizer.encode("ab a").expect("encodes");
        assert_eq!(ids, [0, 1, 2, 0]);
        assert_eq!(tokenizer.decode(&ids).expect("decodes"), "ab a");
    }

    /// The model emits `<|content_thinking|>` unprompted, so a special token
    /// in mid-stream is ordinary. The reference renders them, and so does this.
    #[test]
    fn special_tokens_are_rendered_where_they_arrive() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        let ids = [ENDOFTEXT, 0, END_SAMPLING];
        assert_eq!(
            tokenizer.decode(&ids).expect("decodes"),
            "<|endoftext|>a<|content_model_end_sampling|>"
        );
    }

    /// One character across three tokens, which is what a stream that decoded
    /// each token on its own would render as three replacement characters.
    #[test]
    fn a_character_split_across_tokens_surfaces_only_when_it_is_whole() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");
        let ids = [0, 3, 4, 5, 1];

        let mut stream = tokenizer.stream();
        let segments: Vec<String> = ids
            .iter()
            .map(|&id| stream.push(id).expect("decodes"))
            .collect();

        assert_eq!(segments, ["a", "", "", "日", "b"]);
        assert_eq!(stream.finish(), "");
        assert_eq!(segments.concat(), tokenizer.decode(&ids).expect("decodes"));
    }

    /// A generation cut off mid-character decodes to what a lossy decode of
    /// those bytes gives, rather than swallowing the bytes or waiting forever.
    #[test]
    fn a_sequence_ending_mid_character_flushes_a_replacement() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        assert_eq!(tokenizer.decode(&[3, 4]).expect("decodes"), "\u{fffd}");
    }

    /// What the held-back bytes were waiting for never arrives. Holding them
    /// any longer would keep the special token that ended the message — and
    /// everything after it — out of the stream too.
    #[test]
    fn a_special_token_after_a_broken_character_releases_it() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        let mut stream = tokenizer.stream();
        assert_eq!(stream.push(3).expect("decodes"), "");
        assert_eq!(
            stream.push(ENDOFTEXT).expect("decodes"),
            "\u{fffd}<|endoftext|>"
        );
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn an_id_outside_the_vocabulary_is_refused() {
        let dir = checkpoint();
        let tokenizer = open(&dir, Some(END_SAMPLING)).expect("opens");

        assert!(matches!(
            tokenizer.decode(&[0, 4096]),
            Err(TokenizerError::UnknownToken(4096))
        ));
    }
}
