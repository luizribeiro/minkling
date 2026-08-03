"""The reference's half of a cross-engine sitting.

# What it is

`bench alternate` runs two executables against each other with the order flipped
each pair and reports whether the ranges overlap. Its arms have always been two
builds of this repo's own engine — but an arm is an executable that prints
`name value unit` lines, and nothing in that contract says which engine produced
them. This is mlx-vlm behind the same contract, so that a figure against the
reference is taken the way every figure between two of our own builds is: one
sitting, alternating, both ranges reported.

**That is the whole reason this file exists.** A cross-engine number measured by
running one engine and then the other carries the drift of the machine between
them, and this host has moved 1.7% inside a single sitting. Two figures taken an
hour apart carry that drift in whichever direction the order chose.

# What it measures

One generation per (prompt, generated) pair, timed a step at a time, which is
three readings rather than three runs: the first step is the prefill, the mean of
the ones after it is a decode step, and the sum is the wall a user waits.

**The prompt is read from the Rust side rather than written out here.** A
comparison of two engines is only that if both were given the same tokens.

# What it does not do

It does not speculate: mlx-vlm drops every `model.mtp.*` tensor at load, which is
the reason this repo's speculation exists at all. So the `k = 0` and `k = 2` rows
are one measurement printed under both names, and the reference's column is flat
across the two by construction rather than by measurement.
"""

import argparse
import sys
import time
from pathlib import Path

from inkling_ref import load_model, tokenizer
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.common import generation_stream, wired_limit
from mlx_vlm.models import cache

import mlx.core as mx

# The (prompt, generated) pairs, which are `inkling_core::workload::REALISTIC`.
# A copy rather than a shared file, and the harness is what checks it: two arms
# that do not report the same readings are refused by name, so a table this
# disagreed about would not be published, it would fail to run.
REALISTIC = ((97, 128), (385, 128), (769, 128), (97, 512))

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
    because that is where `stream_generate` enters it; at the default limit of 0
    nothing is guaranteed GPU-resident and a decode step costs 2.6 s instead of
    32 ms. See `results/prefill.md`.
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


def readings(steps, prompted, generated, depth):
    """What one generation says, under the names the Rust arm prints."""
    after = steps[1:]
    pair = f"{prompted}x{generated}.k{depth}"
    return [
        (f"{pair}.wall", sum(steps)),
        (f"{pair}.first", steps[0]),
        (f"{pair}.token", sum(after) / len(after) if after else 0.0),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurement", choices=["engines"])
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument(
        "--depth",
        type=int,
        # `inkling_core::workload::BEST`, which is where this has to be kept: an
        # arm names its rows after its own depth, so the two defaulting apart is
        # a sitting the harness refuses by name rather than one that runs and
        # compares the wrong pair of columns.
        default=2,
        help="the depth the other arm speculates at; this one names its rows "
        "after it and measures once",
    )
    args = ap.parse_args()

    model, processor = load_model(args.model)

    taken = {}
    # The whole pass twice, the first thrown away — what a pair costs depends on
    # what ran before it, so two passes are what give every pair the same
    # predecessor in the pass that counts. The Rust arm does the same.
    for measured in (False, True):
        for prompted, generated in REALISTIC:
            steps, tokens = timed(model, prompt_ids(processor, prompted), generated)
            if not measured:
                continue
            taken[(prompted, generated)] = steps
            # The spread and not just the mean: a decode step here is flat in
            # the context, so a mean that is not the median is a run something
            # else was happening during rather than a slower engine.
            after = sorted(steps[1:])
            print(
                f"{prompted}x{generated}: {len(steps)} tokens, "
                f"steps min {after[0] * 1e3:.2f} "
                f"p50 {after[len(after) // 2] * 1e3:.2f} "
                f"max {after[-1] * 1e3:.2f} ms, "
                f"first eight {[int(token) for token in tokens[:8]]}",
                file=sys.stderr,
            )

    depths = [0, args.depth] if args.depth > 0 else [0]
    for depth in depths:
        for prompted, generated in REALISTIC:
            for name, seconds in readings(
                taken[(prompted, generated)], prompted, generated, depth
            ):
                print(f"{name} {seconds * 1e3:.4f} ms")


if __name__ == "__main__":
    main()
