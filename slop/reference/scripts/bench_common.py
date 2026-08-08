"""What every reference script that times a generation needs.

Two scripts time mlx-vlm over this repo's own prompt — `bench_engines.py`, which
is one arm of a paired cross-engine sitting, and `context_sweep.py`, which is the
shape of a curve over contexts that sitting does not reach. **They have to build
the prompt the same way and enter the wired limit the same way**, and the second
of those is not a detail: at the default limit of 0 nothing is guaranteed
GPU-resident and a decode step costs 2.6 s instead of 32 ms, which is a mistake
that reads as a finding rather than as a bug.
"""

import time
from pathlib import Path

import mlx.core as mx
from inkling_ref import tokenizer
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.common import generation_stream, wired_limit
from mlx_vlm.models import cache

# The prompt both engines are given, which is `inkling_core::workload`'s own.
# **A comparison of two engines is only that if both were given the same
# tokens**, so it is read from the Rust side rather than written out here.
PROMPT = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "inkling-core"
    / "src"
    / "workload.txt"
)


def prompt_ids(processor, tokens):
    """The shared prompt, tiled to `tokens` and cut there.

    Real ids repeated rather than one id repeated, for the reason
    `inkling_core::workload::tiled` gives: which experts a token routes to is
    decided by the token, so a prompt of one id would measure a stack nobody
    runs.
    """
    ids = tokenizer(processor).encode(
        PROMPT.read_text().rstrip("\n"), add_special_tokens=False
    )
    repeats = -(-tokens // len(ids))
    return mx.array([(ids * repeats)[:tokens]])


def timed(model, ids, generated):
    """One generation of exactly `generated` tokens, timed a step at a time.

    `generate_step` is mlx-vlm's own loop and stops on its token budget rather
    than on an end-of-message token, which is what the Rust arm's `Ending` does
    with no eos. Greedy by default — temperature 0 — as both sides are.

    The wired limit is entered per generation rather than around the sitting,
    because that is where `stream_generate` enters it. See `results/prefill.md`.

    Returns the per-step durations, the prefill first, and the tokens.
    """
    steps, tokens = [], []
    with wired_limit(model, [generation_stream]):
        prompt_cache = cache.make_prompt_cache(model.language_model)
        at = time.perf_counter()
        for token, _ in generate_step(
            ids, model, None, None, max_tokens=generated, prompt_cache=prompt_cache
        ):
            now = time.perf_counter()
            steps.append(now - at)
            tokens.append(token)
            at = now
    return steps, tokens
