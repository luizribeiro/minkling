"""What mlx's attention kernel costs at the shapes this engine's prefill gives its own.

**Not a benchmark of mlx-vlm and not paired against anything.** `bench-engines`
is where a cross-engine claim is made and this is not one: it is one dispatch of
`mx.fast.scaled_dot_product_attention` at Inkling's `[1, 32, n, 128]` over 8 KV
heads, so that the reference's attention kernel can be read beside a number
rather than only beside its source.

Three arms, and which of them is which is the point:

- **no mask** walks the whole `n × n` rectangle.
- **causal** walks the triangle, which is the work this engine's global layers do.
- **an additive mask** is what `mlx_vlm.models.inkling` actually passes — a
  materialised `[B, H, LQ, S]` float tensor carrying the banded relative-position
  bias. It is the arm that says whether the mask puts the call on a slower path,
  and whether the kernel still bounds its loop when one is given.

The rate column divides by the work the arm's own bound leaves, so the causal
row is over the triangle and the other two over the rectangle.
"""

import argparse
import time

import mlx.core as mx

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128

# 32 heads of 16384 squared is 34 GB of float32, which is a mask to decline to
# build rather than one to wait for.
MOST_MATERIALISED = 8192


def best_of(fn, rounds):
    """The fastest of `rounds` evaluations, after one that only warms."""
    mx.eval(fn())
    mx.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def arms_over(q, k, v, scale, n, dtype):
    """The three calls, and the fraction of the rectangle each of them walks."""
    arms = {
        "no mask": (
            lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
            1.0,
        ),
        "causal": (
            lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal"
            ),
            0.5,
        ),
    }
    if n <= MOST_MATERIALISED:
        rows = mx.arange(n).reshape(n, 1)
        cols = mx.arange(n).reshape(1, n)
        banded = mx.where(cols <= rows, mx.zeros((n, n)), mx.array(-1e30))
        banded = mx.broadcast_to(banded, (1, HEADS, n, n)).astype(dtype)
        mx.eval(banded)
        arms["an additive mask"] = (
            lambda: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=banded
            ),
            1.0,
        )
    return arms


def run(n, dtype, rounds):
    scale = HEAD_DIM**-0.5
    q = mx.random.normal((1, HEADS, n, HEAD_DIM)).astype(dtype)
    k = mx.random.normal((1, KV_HEADS, n, HEAD_DIM)).astype(dtype)
    v = mx.random.normal((1, KV_HEADS, n, HEAD_DIM)).astype(dtype)
    mx.eval(q, k, v)

    print(f"  {n} tokens, {dtype}, {HEADS} heads over {KV_HEADS}")
    for what, (fn, walked) in arms_over(q, k, v, scale, n, dtype).items():
        taken = best_of(fn, rounds)
        # Two multiply-adds a channel, twice: the scores and the weighted values.
        flop = 2 * 2 * HEADS * n * n * HEAD_DIM * walked
        print(f"    {what:<18}{taken * 1e3:9.2f}ms{flop / taken / 1e12:9.2f} TFLOP/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", nargs="*", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--dtype",
        default="both",
        choices=["both", "bfloat16", "float32"],
        help="float32 is what this engine's own kernel is in and is the comparable arm",
    )
    args = parser.parse_args()

    wanted = {"bfloat16": mx.bfloat16, "float32": mx.float32}
    dtypes = wanted.values() if args.dtype == "both" else [wanted[args.dtype]]
    for n in args.tokens:
        for dtype in dtypes:
            run(n, dtype, args.rounds)


if __name__ == "__main__":
    main()
