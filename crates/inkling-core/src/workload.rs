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

/// The simulated coding session every figure for a kept cache is taken over.
///
/// **This is the workload the architectural win is a claim about**, and it is
/// not a microbenchmark: a prefill of a given length says what one prompt costs,
/// where what a user feels is the same conversation coming back turn after turn
/// with a little added each time. A figure taken on a single prompt cannot say
/// anything at all about keeping one between requests, because there is no
/// "between".
///
/// The shape is a coding session's: an opening that is already thousands of
/// tokens — a file, a task, a directory listing — and then turns that each add a
/// question and are each answered.
///
/// [`Session::OPENING`] is where the sitting's length is decided, and **it is
/// the workload's number rather than the cheapest one to measure**: a coding
/// turn opens nearer 8192 than 2048, and what a kept cache is worth moves with
/// it — 72.6% off the session at 8192 against 61.6% at 2048. A shorter sitting
/// is `--tokens 2048`, and it understates the effect rather than misrepresenting
/// it.
///
/// What the default costs is a sitting of about seventeen minutes over three
/// pairs, most of it the arm that keeps nothing re-prefilling nine thousand
/// tokens five times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Session {
    /// Tokens the conversation opens at, before any turn is taken.
    pub opening: usize,
    /// Turns, each of them a prompt and a generation.
    pub turns: usize,
    /// Tokens the user adds at the start of every turn after the first.
    pub added: usize,
    /// Tokens the model produces in each turn, which the next turn's prompt
    /// carries back — a client that sends the conversation back sends the reply
    /// with it.
    pub generated: usize,
}

impl Session {
    pub const OPENING: usize = 8192;

    /// The default session: [`Session::OPENING`] tokens opened with, five
    /// turns, a few hundred tokens added and decoded each.
    pub const fn new(opening: usize) -> Self {
        Self {
            opening,
            turns: 5,
            added: 256,
            generated: 64,
        }
    }

    /// The prompt of turn `turn`, given what the model produced in the turns
    /// before it.
    ///
    /// **A turn's prompt is the last one, the reply to it, and what the user
    /// added** — which is what makes a coding turn an exact extension of the
    /// turn before it, and is the whole reason a kept cache pays here. The added
    /// tokens come from a different place in `ids` each turn, so that no two
    /// turns add the same text and the routing sees a real spread of tokens.
    pub fn prompt(&self, ids: &[usize], turn: usize, produced: &[Vec<usize>]) -> Vec<usize> {
        let mut prompt = tiled(ids, self.opening);
        for (at, reply) in produced.iter().enumerate().take(turn) {
            prompt.extend_from_slice(reply);
            prompt.extend(
                ids.iter()
                    .copied()
                    .cycle()
                    .skip(self.opening + at * self.added)
                    .take(self.added),
            );
        }
        prompt
    }
}

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
/// **Length is a second axis and it is what decides which entries a prompt
/// reaches.** A call is given a blocked entry only where its rows are two
/// blocks' worth — 64 of them — so a prompt shorter than that runs the same
/// kernels under both words and its agreement would be a check on this harness
/// rather than on any arithmetic. Every prompt below clears that bar on the
/// projections.
///
/// **A routed bank needs two lengths and not one, and that is a change.** Its
/// blocks are cut at the sort's run boundaries while an expert's run is under
/// four blocks and laid over the rows above it, which are two entries at two
/// lengths: about 171 tokens and about 5462. The code listing is the first and
/// the list of primes is the second, and `bench diverge` holds the corpus to
/// reaching every entry the flag selects rather than leaving it to whoever next
/// edits the text — which is how the second length came to be missing here and
/// how it was caught.
pub const CORPUS: [&str; 6] = [
    "<|message_user|><|content_text|>Count from 1 to 30. Separate them with commas, with no \
     commentary before or after them, and stop as soon as you reach thirty rather than carrying \
     on into the thirties. Do not number the lines, do not explain what you are about to do, and \
     do not add a closing remark once you have finished \
     counting.<|end_message|><|message_model|>",
    "The lighthouse keeper had not spoken to another person in nine weeks, and when the supply \
     boat finally rounded the headland he found that he had forgotten which of the several things \
     he had been saving up to say was the one that had seemed urgent, so he stood on the jetty \
     with his hands in his pockets and said nothing at all until",
    "fn merge(left: &[u32], right: &[u32]) -> Vec<u32> {\n    let mut out = \
     Vec::with_capacity(left.len() + right.len());\n    let (mut i, mut j) = (0, 0);\n    while \
     i < left.len() && j < right.len() {\n        if left[i] <= right[j] {\n            \
     out.push(left[i]);\n            i += 1;\n        } else {\n            \
     out.push(right[j]);\n            j += 1;\n        }\n    }\n    \
     out.extend_from_slice(&left[i..]);\n    out.extend_from_slice(&right[j..]);\n    \
     out\n}\n\nfn sort(values: &[u32]) -> Vec<u32> {\n    if values.len() <= 1 {\n        return \
     values.to_vec();\n    }\n    let middle = values.len() / 2;\n    \
     merge(&sort(&values[..middle]), &sort(&values[middle..]))\n}\n\n/// A run-length encoding \
     of a sorted slice, as (value, count) pairs.\nfn runs(values: &[u32]) -> Vec<(u32, usize)> \
     {\n    let mut out: Vec<(u32, usize)> = Vec::new();\n    for value in values {\n        \
     match out.last_mut() {\n            Some((held, count)) if held == value => *count += 1,\n  \
     _ => out.push((*value, 1)),\n        }\n    }\n    out\n}\n\n/// The offset each run would \
     start at if the runs were laid end to end, which\n/// is the exclusive prefix sum of the \
     counts.\nfn offsets(runs: &[(u32, usize)]) -> Vec<usize> {\n    let mut at = 0;\n    let \
     mut out = Vec::with_capacity(runs.len());\n    for (_, count) in runs {\n        \
     out.push(at);\n        at += count;\n    }\n    out\n}\n\n/// Each run cut into blocks of \
     at most `height` rows, as (first, rows) pairs.\n///\n/// A block never spans two runs, so \
     the caller can treat every block as\n/// belonging to exactly one value — which is the \
     whole reason to cut them this\n/// way rather than laying a fixed grid over the rows.\nfn \
     blocks(runs: &[(u32, usize)], height: usize) -> Vec<(usize, usize)> {\n    assert!(height > \
     0, \"a block holds some rows\");\n    let mut out = Vec::new();\n    for (at, (_, count)) \
     in offsets(runs).into_iter().zip(runs) {\n        let mut held = 0;\n        while held < \
     *count {\n            out.push((at + held, (count - held).min(height)));\n            held \
     += height;\n        }\n    }\n    out\n}\n\nfn histogram(values: &[u32], buckets: usize) -> \
     Vec<usize> {\n    let mut counts = vec![0; buckets];\n    let widest = \
     values.iter().copied().max().unwrap_or(0) as usize;\n    for value in values {\n        let \
     bucket = match widest {\n            0 => 0,\n            widest => (*value as usize * \
     (buckets - 1)) / widest,\n        };\n        counts[bucket] += 1;\n    }\n    \
     counts\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    fn seeded(count: usize, \
     seed: u32) -> Vec<u32> {\n        let mut state = seed | 1;\n        (0..count)\n           \
     .map(|_| {\n                state = \
     state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);\n                state >> 20\n   \
     })\n            .collect()\n    }\n\n    #[test]\n    fn \
     a_sort_is_a_permutation_of_what_it_was_given() {\n        for seed in [1u32, 7, 99, \
     4_294_967_291] {\n            let values = seeded(257, seed);\n            let sorted = \
     sort(&values);\n            assert_eq!(sorted.len(), values.len());\n            let mut \
     want = values.clone();\n            want.sort_unstable();\n            assert_eq!(sorted, \
     want, \"seed {seed}\");\n        }\n    }\n\n    #[test]\n    fn \
     the_runs_of_a_sorted_slice_sum_to_its_length() {\n        let sorted = sort(&seeded(1024, \
     3));\n        let counted: usize = runs(&sorted).iter().map(|(_, count)| count).sum();\n    \
     assert_eq!(counted, sorted.len());\n    }\n\n    #[test]\n    fn \
     every_block_lies_inside_one_run() {\n        let sorted = sort(&seeded(1024, 11));\n        \
     let held = runs(&sorted);\n        for height in [1usize, 3, 8, 32, 4096] {\n            \
     let mut covered = vec![0usize; sorted.len()];\n            for (first, rows) in \
     blocks(&held, height) {\n                assert!(rows <= height, \"a block of {rows} rows \
     at height {height}\");\n                assert!(\n                    sorted[first..first + \
     rows].iter().all(|v| *v == sorted[first]),\n                    \"a block at {first} of \
     {rows} rows spans two runs\"\n                );\n                for slot in &mut \
     covered[first..first + rows] {\n                    *slot += 1;\n                }\n        \
     }\n            assert!(\n                covered.iter().all(|times| *times == 1),\n         \
     \"the blocks do not cover every row exactly once at height {height}\"\n            );\n     \
     }\n    }\n\n    #[test]\n    fn a_histogram_counts_every_value_once() {\n        let values \
     = seeded(512, 17);\n        assert_eq!(histogram(&values, 16).iter().sum::<usize>(), \
     values.len());\n    }\n}\n\nfn main() {\n",
    "<|message_user|><|content_text|>Explain why a hash map's worst case is linear rather than \
     constant, why that almost never happens with a good hash function, what an attacker who \
     controls the keys can do about it, and which of the usual mitigations — a randomised seed, a \
     tree fallback for long chains, or a keyed hash — actually removes the problem rather than \
     making it less likely. Four sentences, no \
     bullets.<|end_message|><|message_model|>",
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
     4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, \
     4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, \
     4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, \
     4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, \
     4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, \
     4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, \
     4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, \
     4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, \
     5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, \
     5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, \
     5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, \
     5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, \
     5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, \
     5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, \
     5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, \
     6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, \
     6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, \
     6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, \
     6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, \
     6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, \
     6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, \
     6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, \
     6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, \
     7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, \
     7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, \
     7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, \
     7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, \
     7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, \
     7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, \
     7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, \
     8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147, \
     8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, \
     8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, \
     8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, \
     8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, \
     8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, \
     8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933, 8941, 8951, \
     8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, \
     9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221, \
     9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, \
     9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, \
     9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, \
     9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, \
     9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, \
     9883, 9887, 9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973, 10007, 10009, 10037, \
     10039, 10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111, 10133, 10139, 10141, \
     10151, 10159, 10163, 10169, 10177, 10181, 10193, 10211, 10223, 10243, 10247, 10253, 10259, \
     10267, 10271, 10273, 10289, 10301, 10303, 10313, 10321, 10331, 10333, 10337, 10343, 10357, \
     10369, 10391, 10399, 10427, 10429, 10433, 10453, 10457, 10459, 10463, 10477, 10487, 10499, \
     10501, 10513, 10529, 10531, 10559, 10567, 10589, 10597, 10601, 10607, 10613, 10627, 10631, \
     10639, 10651, 10657, 10663, 10667, 10687, 10691, 10709, 10711, 10723, 10729, 10733, 10739, \
     10753, 10771, 10781, 10789, 10799, 10831, 10837, 10847, 10853, 10859, 10861, 10867, 10883, \
     10889, 10891, 10903, 10909, 10937, 10939, 10949, 10957, 10973, 10979, 10987, 10993, 11003, \
     11027, 11047, 11057, 11059, 11069, 11071, 11083, 11087, 11093, 11113, 11117, 11119, 11131, \
     11149, 11159, 11161, 11171, 11173, 11177, 11197, 11213, 11239, 11243, 11251, 11257, 11261, \
     11273, 11279, 11287, 11299, 11311, 11317, 11321, 11329, 11351, 11353, 11369, 11383, 11393, \
     11399, 11411, 11423, 11437, 11443, 11447, 11467, 11471, 11483, 11489, 11491, 11497, 11503, \
     11519, 11527, 11549, 11551, 11579, 11587, 11593, 11597, 11617, 11621, 11633, 11657, 11677, \
     11681, 11689, 11699, 11701, 11717, 11719, 11731, 11743, 11777, 11779, 11783, 11789, 11801, \
     11807, 11813, 11821, 11827, 11831, 11833, 11839, 11863, 11867, 11887, 11897, 11903, 11909, \
     11923, 11927, 11933, 11939, 11941, 11953, 11959, 11969, 11971, 11981, 11987, 12007, 12011, \
     12037, 12041, 12043, 12049, 12071, 12073, 12097, 12101, 12107, 12109, 12113, 12119, 12143, \
     12149, 12157, 12161, 12163, 12197, 12203, 12211, 12227, 12239, 12241, 12251, 12253, 12263, \
     12269, 12277, 12281, 12289, 12301, 12323, 12329, 12343, 12347, 12373, 12377, 12379, 12391, \
     12401, 12409, 12413, 12421, 12433, 12437, 12451, 12457, 12473, 12479, 12487, 12491, 12497, \
     12503, 12511, 12517, 12527, 12539, 12541, 12547, 12553, 12569, 12577, 12583, 12589, 12601, \
     12611, 12613, 12619, 12637, 12641, 12647, 12653, 12659, 12671, 12689, 12697, 12703, 12713, \
     12721, 12739, 12743, 12757, 12763, 12781, 12791, 12799, 12809, 12821, 12823, 12829, 12841, \
     12853, 12889, 12893, 12899, 12907, 12911, 12917, 12919, 12923, 12941, 12953, 12959, 12967, \
     12973, 12979, 12983, 13001, 13003, 13007, 13009, 13033, 13037, 13043, 13049, 13063, 13093, \
     13099, 13103, 13109, 13121, 13127, 13147, 13151, 13159, 13163, 13171, 13177, 13183, 13187, \
     13217, 13219, 13229, 13241, 13249, 13259, 13267, 13291, 13297, 13309, 13313, 13327, 13331, \
     13337, 13339, 13367, 13381, 13397, 13399, 13411, 13417, 13421, 13441, 13451, 13457, 13463, \
     13469, 13477, 13487, 13499, 13513, 13523, 13537, 13553, 13567, 13577, 13591, 13597, 13613, \
     13619, 13627, 13633, 13649, 13669, 13679, 13681, 13687, 13691, 13693, 13697, 13709, 13711, \
     13721, 13723, 13729, 13751, 13757, 13759, 13763, 13781, 13789, 13799, 13807, 13829, 13831, \
     13841, 13859, 13873, 13877, 13879, 13883, 13901, 13903, 13907, 13913, 13921, 13931, 13933, \
     13963, 13967, 13997, 13999,",
    "<|message_user|><|content_text|>What is the capital of France, what river runs through it, \
     roughly how many people live in the city proper as against the wider metropolitan area, and \
     which of its railway termini would you leave from for Brussels, for Bordeaux and for \
     Marseille? Answer in one sentence each and do not pad them \
     out.<|end_message|><|message_model|>",
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

    /// **A differential corpus of short prompts is a corpus that reaches neither
    /// entry.** Which of the packed matmul's entries a call is dispatched
    /// through is decided by its rows, and the entries behind `--numerics
    /// production` are given a call only where its rows are two 32-row blocks'
    /// worth — so a prompt under 64 tokens runs the same kernels under both
    /// words and reports nothing at all. Above that the projections reach the
    /// tiled entry, and a prompt whose routed-bank rows outnumber the bank's 256
    /// experts by a block's worth of runs — six rows a token against 32 runs an
    /// expert, which is about 1366 tokens — reaches the grouped one.
    ///
    /// **A coarse guard and not a derivation**, and saying which is the point.
    /// A tokenizer is a checkpoint away from here, so this can only bound the
    /// bytes — and bytes do not give tokens: the code member below is 3.1 bytes
    /// a token and the chat-tagged one 5.7, which is nearly a factor of two.
    /// What this catches is a member deleted down to a line; what it cannot
    /// catch is one that stays long in bytes and short in tokens.
    ///
    /// **The check with teeth is in `bench diverge`**, which holds every
    /// prompt's *token* count against `PackedMatmul::SHORTEST_BLOCKED_CALL`
    /// before it runs anything — the tokenizer is open by then, and a length in
    /// tokens is the only length that decides which entry a call reaches.
    #[test]
    fn no_prompt_of_the_corpus_has_been_cut_down_to_a_line() {
        let shortest = CORPUS.iter().map(|prompt| prompt.len()).min();
        assert!(shortest > Some(200), "{shortest:?} bytes is the shortest");
        let longest = CORPUS.iter().map(|prompt| prompt.len()).max();
        assert!(longest > Some(9000), "{longest:?} bytes is the longest");
        // The second-longest is the one that reaches the entry the longest does
        // not, so a corpus cut down to one long member fails here.
        let mut lengths: Vec<usize> = CORPUS.iter().map(|prompt| prompt.len()).collect();
        lengths.sort_unstable();
        let second = lengths[lengths.len() - 2];
        assert!(second > 4000, "{second} bytes is the second longest");
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

    /// The property the whole of a kept cache rests on, stated about the
    /// workload rather than about the engine: **turn `n + 1`'s prompt starts
    /// with turn `n`'s.** A session that did not have it would measure a cache
    /// that never matched, and would report the miss path as the feature.
    #[test]
    fn each_turn_of_the_session_is_an_exact_extension_of_the_turn_before_it() {
        let session = Session::new(64);
        let ids: Vec<usize> = (1..=7).collect();
        let produced: Vec<Vec<usize>> = (0..session.turns)
            .map(|turn| vec![900 + turn; session.generated])
            .collect();

        let prompts: Vec<Vec<usize>> = (0..session.turns)
            .map(|turn| session.prompt(&ids, turn, &produced))
            .collect();
        for (turn, pair) in prompts.windows(2).enumerate() {
            let [before, after] = [&pair[0], &pair[1]];
            assert!(after.starts_with(before), "turn {turn} is not extended");
            assert_eq!(
                after.len() - before.len(),
                session.generated + session.added,
                "turn {turn} added something other than a reply and a question"
            );
        }
        assert_eq!(prompts[0].len(), session.opening);
    }

    /// Two turns that added the same tokens would put the same text in front of
    /// the model twice, which is a session of one distribution measured five
    /// times — the mistake [`CORPUS`] exists to avoid on the other axis.
    #[test]
    fn no_two_turns_of_the_session_add_the_same_tokens() {
        let session = Session::new(64);
        let ids: Vec<usize> = (1..=7).collect();
        let produced: Vec<Vec<usize>> = (0..session.turns).map(|_| vec![900]).collect();

        let added: Vec<Vec<usize>> = (1..session.turns)
            .map(|turn| {
                let (before, after) = (
                    session.prompt(&ids, turn - 1, &produced),
                    session.prompt(&ids, turn, &produced),
                );
                after[before.len() + produced[turn - 1].len()..].to_vec()
            })
            .collect();
        for (at, tokens) in added.iter().enumerate() {
            assert!(
                !added[..at].contains(tokens),
                "turn {at} repeats a question"
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
