"""What the reference's decode step and its memory cost as the context grows.

# Why this is not `bench_engines.py`

That script is one half of a paired, alternating cross-engine sitting, and its
four (prompt, generated) pairs top out at a 769-token prompt. **Every decode
figure either engine has here was taken inside that range**, and a coding turn
opens at thousands of tokens and grows all session — so "the reference is flat
in the context" is a claim about eightfold, established where a coding workload
has not yet begun.

Flat to 769 does not prove flat to 32768, and the reason to doubt it is in the
reference's own source: `InklingModel.make_cache` returns a plain `KVCache` for
every one of the 42 layers, including the 35 whose attention is capped at a
512-token window. So the reference retains every key it has ever seen and its
attention reads them, masked. Whether that is visible at 769 tokens and whether
it is visible at 32768 are two different questions, and only the second one is
about a coding context.

# What it measures

One generation per context: a prompt tiled to that length, then a handful of
decode steps. The decode figure is the mean of the steps after the first, for
the reason `bench_engines.py` gives — the first step is the prefill. Peak memory
is MLX's own high-water mark, reset per context, which is the figure the
architecture's KV arithmetic is a prediction about.

**Not paired and not alternating, and that is deliberate.** What this is for is
the shape of a curve over a 340-fold range, where the effect is orders of
magnitude and the machine's 1.7% drift cannot reach it. The paired sitting stays
`just bench-engines`, and no headline figure should be taken from here.
"""

import argparse
import gc
import traceback
from pathlib import Path

import mlx.core as mx
from bench_common import prompt_ids, timed
from inkling_ref import load_model

# The contexts, out to where a coding session lives. The first three are
# `inkling_core::workload::REALISTIC`'s own prompt lengths, so this table and
# the paired one meet at three points rather than at none.
CONTEXTS = (97, 385, 769, 2048, 4096, 8192, 16384, 32768)

# Decode steps after the prefill. Enough to average, few enough that the prefill
# is what a long context costs here rather than the generation.
GENERATED = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=CONTEXTS,
        help="the prompt lengths to sweep, longest last",
    )
    args = ap.parse_args()

    model, processor = load_model(Path(args.model))
    print(
        f"{'context':>8}{'prefill':>12}{'a token':>12}{'tokens/s':>11}{'peak':>12}",
        flush=True,
    )
    refused = 0
    for context in args.contexts:
        ids = prompt_ids(processor, context)
        # Per context rather than once, so a row's peak is that row's rather
        # than the sweep's running maximum.
        mx.clear_cache()
        gc.collect()
        mx.reset_peak_memory()
        try:
            steps, _tokens = timed(model, ids, GENERATED)
        except MemoryError:
            # **The refusal is the reading, and only this one is.** A context
            # the reference cannot allocate for is what the sweep is asking
            # about; anything else is a bug in this script or in mlx-vlm, and a
            # sweep that printed it in the same shape as a row would invite it
            # to be read as one. So the traceback goes out and the exit code
            # carries it.
            refused += 1
            print(f"{context:>8}  refused, out of memory", flush=True)
            traceback.print_exc()
            continue
        after = steps[1:]
        a_token = sum(after) / len(after) if after else float("nan")
        print(
            f"{context:>8}{steps[0] * 1e3:>10.0f}ms{a_token * 1e3:>10.2f}ms"
            f"{1.0 / a_token:>11.1f}{mx.get_peak_memory() / (1 << 30):>10.2f}GiB",
            flush=True,
        )
    raise SystemExit(1 if refused else 0)


if __name__ == "__main__":
    main()
