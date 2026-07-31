"""Check that the two tokenizers a checkpoint ships describe the same thing.

`tokenizer_config.json` names `TokenizersBackend`, so `tokenizer.json` is what
the reference actually loads and what the Rust side should load too. The
`tiktoken/tokenizer.model` beside it is a second copy of the same vocabulary in
a different format, and nothing in the checkpoint says the two agree.

Two comparisons say it. The vocabularies are compared id by id, after mapping
the HF pieces out of the byte-level alphabet and back to the bytes tiktoken
stores base64-encoded. Then a tiktoken-style encoder — the split regex read out
of `tokenizer.json`, and merges chosen by rank rather than by a merge list — is
run over a corpus and compared against the HF tokenizer's own output, which is
what says the two agree on more than which strings exist.

Loads no model: both files are read straight off disk.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import regex
from inkling_ref import piece_bytes
from tokenizers import Tokenizer

# Text chosen for the places byte-level BPE has a choice to make: the recorded
# fixture prompt, multi-byte characters at every UTF-8 width, whitespace runs
# and newlines, digit groups past the regex's three-digit cap, contractions the
# case-insensitive branch of the pattern matches, and code punctuation.
CORPUS = [
    "The lighthouse keeper counted the ships that passed the headland.",
    "café naïve résumé Æthelred",
    "日本語のテキストと한국어 混在",
    "🙂🙃 emoji  runs\tand\ttabs",
    "trailing spaces   \n\n  and newlines\r\n",
    "1234567890 and 0x1F and 3.14159 and 1,000,000",
    "It's don't they've we'll I'm he'd she's — dashes",
    "def f(x: int) -> list[str]: return [chr(x)] * 2  # comment",
    "МОСКВА москва ΑΘΗΝΑ αθήνα",
    "a" * 200,
    "",
    " ",
    "\n",
]


def read_tiktoken(path):
    ranks = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks


def compare_vocabularies(hf, ranks):
    """Every id in one file, spelled as bytes, is that id in the other."""
    hf_bytes = {}
    unmappable = []
    for piece, token_id in hf.get_vocab(with_added_tokens=True).items():
        try:
            hf_bytes[piece_bytes(piece)] = token_id
        except KeyError:
            unmappable.append(piece)

    problems = []
    if unmappable:
        problems.append(f"{len(unmappable)} pieces outside the byte-level alphabet")
    if len(hf_bytes) != hf.get_vocab_size(True):
        problems.append("HF pieces collide once mapped to bytes")

    only_hf = sorted(hf_bytes.keys() - ranks.keys())
    only_tiktoken = sorted(ranks.keys() - hf_bytes.keys())
    disagreed = sorted(
        t for t in hf_bytes.keys() & ranks.keys() if hf_bytes[t] != ranks[t]
    )
    for label, tokens in (
        ("only in tokenizer.json", only_hf),
        ("only in tiktoken", only_tiktoken),
        ("at different ids", disagreed),
    ):
        if tokens:
            problems.append(f"{len(tokens)} tokens {label}, e.g. {tokens[:3]}")

    print(f"vocabulary: {len(hf_bytes)} HF pieces, {len(ranks)} tiktoken ranks")
    return problems


def split_pattern(tokenizer_json):
    """The pre-tokenizer's split regex, read out of the file being checked
    rather than hardcoded — a tiktoken encoder that split differently would
    prove nothing about this checkpoint."""
    for pre in tokenizer_json["pre_tokenizer"]["pretokenizers"]:
        if pre["type"] == "Split":
            return pre["pattern"]["Regex"]
    raise SystemExit("tokenizer.json has no Split pre-tokenizer")


def merge_by_rank(piece, ranks):
    """tiktoken's byte-pair encoding: repeatedly merge the adjacent pair with
    the lowest rank. No merge list — the ranks are the merge order."""
    if piece in ranks:
        return [ranks[piece]]
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        merges = (
            (ranks[parts[i] + parts[i + 1]], i)
            for i in range(len(parts) - 1)
            if parts[i] + parts[i + 1] in ranks
        )
        best = min(merges, default=None)
        if best is None:
            break
        _, i = best
        parts[i : i + 2] = [parts[i] + parts[i + 1]]
    return [ranks[part] for part in parts]


def tiktoken_encode(text, pattern, ranks):
    return [
        token_id
        for piece in regex.findall(pattern, text)
        for token_id in merge_by_rank(piece.encode(), ranks)
    ]


def compare_encodings(hf, pattern, ranks, corpus):
    problems = []
    for text in corpus:
        want = hf.encode(text, add_special_tokens=False).ids
        got = tiktoken_encode(text, pattern, ranks)
        if want != got:
            problems.append(f"{text[:40]!r}: HF {want[:8]} vs tiktoken {got[:8]}")
    print(
        f"encoding: {len(corpus)} texts agree" if not problems else "encoding: differs"
    )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path, nargs="?", default=Path("models/Inkling-Small-mxfp4")
    )
    args = parser.parse_args()

    hf_path = args.model / "tokenizer.json"
    tiktoken_path = args.model / "tiktoken" / "tokenizer.model"
    config = json.loads((args.model / "tokenizer_config.json").read_text())
    print(f"tokenizer_config.json declares backend {config['backend']!r}")

    hf = Tokenizer.from_file(str(hf_path))
    ranks = read_tiktoken(tiktoken_path)
    problems = compare_vocabularies(hf, ranks)
    problems += compare_encodings(
        hf, split_pattern(json.loads(hf_path.read_text())), ranks, CORPUS
    )

    for problem in problems:
        print(f"  disagreement: {problem}")
    print(
        "the two files describe the same tokenizer" if not problems else "they differ"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
