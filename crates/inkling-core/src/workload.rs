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

/// The prompts two numerics are held against each other over.
///
/// **A differential run's corpus has one job the timing corpus does not: to be
/// several distributions rather than one length.** Whether two accumulations
/// name the same token is decided by how close the top two logits are, and how
/// close those are is a property of the text — enumeration puts a near-certain
/// token in front of the model where prose puts a dozen plausible ones. A corpus
/// of one prompt tiled to four lengths is one distribution measured four times,
/// and would report the agreement of whichever regime it happened to be.
///
/// So these are six texts and not six lengths: enumeration, prose mid-sentence,
/// code, a chat turn written out the way [`STRUCTURED_PROMPT`] is, a list of
/// numerals, and a factual question with one likely answer.
///
/// **Length is a second axis and it is deliberate here too.** The two entries
/// this flag reaches are the tiled ones, and which of them a call goes through
/// is decided by how many rows it has: every prompt below reaches the tiled
/// projections, and only one long enough for a routed bank's rows to outnumber
/// its experts reaches the grouped entry. The list of primes is that one, and
/// `a_corpus_reaches_both_of_the_entries_this_flag_selects` is what holds it to
/// a length rather than leaving it to whoever next edits the text.
pub const CORPUS: [&str; 6] = [
    "<|message_user|><|content_text|>Count from 1 to 30. Separate them with commas. No \
     commentary.<|end_message|><|message_model|>",
    "The lighthouse keeper had not spoken to another person in nine weeks, and when the supply \
     boat finally rounded the headland he found that he",
    "fn merge(left: &[u32], right: &[u32]) -> Vec<u32> {\n    let mut out = Vec::with_capacity(",
    "<|message_user|><|content_text|>Explain why a hash map's worst case is linear, in two \
     sentences.<|end_message|><|message_model|>",
    "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, \
     97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, \
     191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, \
     283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, \
     401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, \
     509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, \
     631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, \
     751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, \
     877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, \
     1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, \
     1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, \
     1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, \
     1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, \
     1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, \
     1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, \
     1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, \
     1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, \
     1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, \
     2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, \
     2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, \
     2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, \
     2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, \
     2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, \
     2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, \
     2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, \
     2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, \
     2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, \
     3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, \
     3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, \
     3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, \
     3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, \
     3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, \
     3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, \
     3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, \
     3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, \
     4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159,",
    "<|message_user|><|content_text|>What is the capital of \
     France?<|end_message|><|message_model|>",
];

/// How many tokens a differential run generates from each of [`CORPUS`]'s
/// prompts.
///
/// **Long enough for a disagreement to have somewhere to happen.** Two paths
/// that name the same token at every step of a generation say something only if
/// the generation had steps; 64 of them across six prompts is 384 sampled
/// argmaxes, and a per-step disagreement rate under about a third of a percent
/// is what "no divergence" over that corpus can bound.
pub const DIFFERENTIAL: usize = 64;

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

    /// **A differential corpus of one length is a corpus that reaches one
    /// entry.** Which of the packed matmul's entries a call is dispatched
    /// through is decided by its rows: every prompt here reaches the tiled
    /// projections, and only a prompt whose routed-bank rows outnumber the
    /// bank's 256 experts reaches the grouped one — six rows a token against a
    /// threshold of two runs an expert, which is 86 tokens. A corpus of six
    /// short prompts would report the agreement of one entry and say nothing
    /// about the other.
    ///
    /// Bytes rather than tokens, because a tokenizer is a checkpoint away from
    /// here and the margin does not need one: 1500 bytes of comma-separated
    /// numerals is several hundred tokens however they are split, where 86 is
    /// the line.
    #[test]
    fn a_corpus_reaches_both_of_the_entries_this_flag_selects() {
        let longest = CORPUS.iter().map(|prompt| prompt.len()).max();
        assert!(longest > Some(1500), "{longest:?} bytes is the longest");
    }

    /// Two copies of one prompt are one prompt measured twice, and the whole
    /// point of the corpus is that it is several distributions.
    #[test]
    fn no_two_prompts_of_the_corpus_are_the_same_prompt() {
        for (at, prompt) in CORPUS.iter().enumerate() {
            assert!(!prompt.is_empty(), "prompt {at} is empty");
            assert!(
                !CORPUS[..at].contains(prompt),
                "prompt {at} is one of the ones before it"
            );
        }
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
