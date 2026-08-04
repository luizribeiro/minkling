"""The reference's half of a session sitting: does mlx-vlm keep a cache between
requests, and what is it worth when it does.

# It does, and the answer is worth stating precisely

mlx-vlm ships Automatic Prefix Caching. For a plain attention model it is
block-level and pageable; for one whose cache is not block-concatenable — which
Inkling's is not, because every layer carries four short-convolution windows
beside its keys — it falls to the *exact* path: a whole prompt-cache snapshot
taken at a prefix boundary, matched by the token ids that produced it. That is
the same shape this repo's own `Kept` has, arrived at from the same constraint,
and it is what this arm drives.

**It is off unless asked for.** `APC_ENABLED` defaults to `0` and the manager is
wired through mlx-vlm's server rather than through `generate_step`, so a caller
of the generation API gets no reuse at all. This builds the manager directly,
which is the same object the server would have built.

**And its cache is not trimmable.** `ArraysCache` — the windows — declares no
`trim`, so `CacheList.is_trimmable()` is false for this architecture and the
ordinary "keep the cache and drop the reply" move is not available: the reply
cannot be taken back out. The snapshot is what stands in for it, and a snapshot
is a copy of the whole KV rather than a count. What that costs is measured here
rather than argued about — it lands inside the turn that takes it.

# What it measures

The same session the Rust arm runs, printed under the same names: per turn the
wall, the time to the first token and how many tokens had to be prefilled, and
the whole conversation's wall beside them. Two arms, `--reuse-tokens 0` and a
bound, exactly as on the other side.

**The prompts are the same lengths and not the same tokens.** A turn's prompt
carries the previous turn's reply, and the two engines write their own replies —
so what is held equal here is the shape of the session, which is what the
timings depend on.
"""

import argparse
import sys
import time

import mlx.core as mx
from bench_common import PROMPT, tokenizer
from inkling_ref import load_model
from mlx_vlm.apc import APCManager
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.common import generation_stream, wired_limit
from mlx_vlm.models import cache

# `inkling_core::workload::Session`'s own shape. A copy rather than a shared
# file, and the harness is what checks it: two arms that do not report the same
# rows are refused by name, so a session the two disagreed about would fail to
# run rather than publish a table.
TURNS = 5
ADDED = 256
GENERATED = 64
OPENING = 2048


def prompt_for(ids, turn, produced, opening):
    """`Session::prompt`, in Python.

    A turn's prompt is the last one, the reply to it, and what the user added —
    which is what makes a coding turn an exact extension of the turn before it.
    The added tokens come from a different place in `ids` each turn, so no two
    turns add the same text.
    """
    repeats = -(-opening // len(ids))
    prompt = (ids * repeats)[:opening]
    for at, reply in enumerate(produced[:turn]):
        prompt = prompt + list(reply)
        start = opening + at * ADDED
        wrapped = (ids * (-(-(start + ADDED) // len(ids)) + 1))[start : start + ADDED]
        prompt = prompt + wrapped
    return prompt


def turn(model, ids, generated, manager):
    """One turn, timed: the steps, the tokens it produced, and how many of its
    prompt had to be prefilled.

    The snapshot is taken after the *first* token, which is where the cache holds
    exactly the prompt — the next step appends to it. That is the same position
    the Rust arm marks at, and for the same reason: what a client sends back next
    turn starts with this turn's prompt.
    """
    prompt_cache, prefix = (None, 0)
    if manager is not None:
        prompt_cache, prefix = manager.lookup_exact_cache(ids)
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model.language_model)
        prefix = 0

    steps, tokens = [], []
    fed = mx.array([ids[prefix:]])
    with wired_limit(model, [generation_stream]):
        at = time.perf_counter()
        for token, _ in generate_step(
            fed, model, None, None, max_tokens=generated, prompt_cache=prompt_cache
        ):
            now = time.perf_counter()
            steps.append(now - at)
            tokens.append(int(token))
            if len(tokens) == 1 and manager is not None:
                manager.store_exact_cache(ids, prompt_cache)
            at = time.perf_counter()
    return steps, tokens, len(ids) - prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurement", choices=["session"])
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--tokens", type=int, default=OPENING, help="the opening prompt")
    ap.add_argument(
        "--reuse-tokens",
        type=int,
        default=1 << 15,
        help="positions kept between turns; 0 keeps nothing",
    )
    args = ap.parse_args()

    model, processor = load_model(args.model)
    ids = tokenizer(processor).encode(
        PROMPT.read_text().rstrip("\n"), add_special_tokens=False
    )

    # The manager the server would have built, or nothing at all for the arm that
    # keeps nothing. `APC_EXACT_CACHE_ENTRIES` is what bounds it, and one entry
    # is what a single conversation needs.
    manager = APCManager() if args.reuse_tokens > 0 else None

    walls, produced = [], []
    for at in range(TURNS):
        prompt = prompt_for(ids, at, produced, args.tokens)
        if len(prompt) > args.reuse_tokens and manager is not None:
            # The bound, as the other arm applies it: a conversation past it is
            # served and then forgotten.
            manager = APCManager()
        steps, tokens, prefilled = turn(model, prompt, GENERATED, manager)
        produced.append(tokens)

        wall, first = sum(steps), steps[0]
        walls.append(wall)
        print(
            f"  turn {at}: {len(prompt)} tokens, {len(prompt) - prefilled} reused, "
            f"{wall:.2f} s wall, {first:.2f} s to first",
            file=sys.stderr,
        )
        print(f"turn{at}.wall {wall * 1e3:.4f} ms")
        print(f"turn{at}.first {first * 1e3:.4f} ms")
        print(f"turn{at}.prefilled {prefilled}.0000 tokens")

    print(f"session {sum(walls) * 1e3:.4f} ms")
    print(
        f"session: {TURNS} turns keeping {args.reuse_tokens}, "
        f"first eight {produced[0][:8]}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
