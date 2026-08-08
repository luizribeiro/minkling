"""Dump the text/id pairs the Rust tokenizer is tested against.

`tokenizer.json` is 27 MB and cannot be committed, so the hermetic tests get a
few kilobytes of it instead: for each case the ids, the text the reference
decodes them to, and the vocabulary pieces those ids name. The pieces are what
make the streaming decoder testable without a vocabulary — a piece is a
byte-level spelling, so the Rust side can reconstruct the bytes each token
contributes and assemble them itself.

Recorded beside them is what the reference's own streaming detokenizer emits
per token, which is the thing a naive port gets wrong: a character split across
tokens must not surface until its last byte arrives, and a special token in
mid-stream is rendered rather than dropped.

Loads no model. The tokenizer and `config.json` are read straight off disk.
"""

import argparse
import json
from pathlib import Path

import mlx_lm.tokenizer_utils as tokenizer_utils
from inkling_ref import byte_pieces, piece_bytes
from tokenizers import Tokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = FIXTURES / "tokenizer_cases.json"

# The sentence `dump_activations.py` records its forward pass over. Committing
# the pair here is what lets a test say the Rust tokenizer produces the ids the
# rest of the fixtures were captured from.
PROMPT = "The lighthouse keeper counted the ships that passed the headland."

# A model turn as `chat_template.jinja` lays one out. The model emits
# `<|content_thinking|>` unprompted, so decoding has to survive special tokens
# arriving in the middle of a stream rather than only at its end.
TURN = (
    "<|message_model|><|content_thinking|>Weigh it up.<|content_text|>"
    "Café, 日本語, 🙂.<|end_message|>"
)


def by_pieces(tokenizer, pieces):
    """Ids named by the pieces that spell them, for a case whose point is how
    the tokens are split rather than how the text encodes."""
    return [tokenizer.token_to_id(piece) for piece in pieces]


def cases(tokenizer):
    """Each case is ids first: what a decoder is handed is a token sequence,
    and the text is whatever the reference makes of it."""

    def encoded(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    return {
        "prompt": encoded(PROMPT),
        "turn": encoded(TURN),
        # 日 and 🙂 spelled one byte at a time, which is the arrangement that
        # tells a streaming decoder that holds partial UTF-8 apart from one
        # that emits each token as it lands: decoded singly these are eight
        # replacement characters, and decoded together they are two characters.
        "split_characters": by_pieces(
            tokenizer,
            [*byte_pieces("The"), *byte_pieces("日"), *byte_pieces("🙂"), "."],
        ),
        # A character the model never finished, and then a special token. What
        # the held-back bytes were waiting for never arrives, and holding them
        # any longer would keep everything after them out of the stream too.
        "interrupted_character": by_pieces(
            tokenizer,
            [*byte_pieces("The"), *byte_pieces("日")[:2], "<|end_message|>", "."],
        ),
    }


def reference_segments(detokenizer, ids):
    """What the reference's streaming detokenizer surfaces as each token
    arrives, and then whatever its flush leaves over.

    mlx_lm's BPE detokenizer holds a lone space token back for a token and
    trims a leading space off the first segment; both are cosmetic and both
    would put its text at odds with `decode`, so the cases stay clear of them
    and this refuses one that would not."""
    for token_id in ids:
        piece = detokenizer.tokenmap[token_id]
        assert piece_bytes(piece) != b" ", (
            f"token {token_id} is a lone space, which the reference delays"
        )

    detokenizer.reset()
    segments = []
    for token_id in ids:
        detokenizer.add_token(token_id)
        segments.append(detokenizer.last_segment)
    detokenizer.finalize()
    segments.append(detokenizer.last_segment)
    return segments


def build(model_path):
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    detokenizer = tokenizer_utils.load(model_path).detokenizer
    config = json.loads((model_path / "config.json").read_text())
    eos = config["eos_token_id"]

    fixture = {
        "checkpoint": str(model_path),
        "eos_token_id": eos,
        "eos_token": tokenizer.id_to_token(eos),
        "cases": {},
    }
    for name, ids in cases(tokenizer).items():
        text = tokenizer.decode(ids, skip_special_tokens=False)
        segments = reference_segments(detokenizer, ids)
        assert "".join(segments) == text, f"{name}: streamed text is not decoded text"
        fixture["cases"][name] = {
            "ids": ids,
            "text": text,
            "pieces": [tokenizer.id_to_token(token_id) for token_id in ids],
            "segments": segments,
            # Whether the text encodes back to these ids. A case built from
            # pieces splits characters no encoder would split, and says so.
            "round_trips": tokenizer.encode(text, add_special_tokens=False).ids == ids,
        }
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path, nargs="?", default=Path("models/Inkling-Small-mxfp4")
    )
    args = parser.parse_args()

    fixture = build(args.model)
    FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    for name, case in fixture["cases"].items():
        print(f"  {name}: {len(case['ids'])} ids, {case['text'][:48]!r}")


if __name__ == "__main__":
    main()
