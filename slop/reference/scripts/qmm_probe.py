"""What mlx's quantised matmul costs at the two shapes this engine's prefill gives its own.

**Not a benchmark of mlx-vlm and not paired against anything.** `bench-engines`
is where a cross-engine claim is made and this is not one. It is one dispatch
each of `mx.quantized_matmul` and `mx.gather_qmm` at the shapes
`what_a_prefills_blocked_matmul_is_bound_by` reads its tables across — a
projection of `tokens` rows through 4096 to 4096, and a routed bank of six rows
a token through 4096 to 2048 over 256 experts — so that mlx's quantised kernels
can be read beside a number rather than only beside their source.

**MXFP4 is the mode to read**, because it is this checkpoint's: four-bit codes,
groups of 32, an E8M0 scale byte and no bias. `affine` is beside it because it
is what most quantised mlx checkpoints are, and `--dense` is the unquantised
GEMM, which is the only arm here that says what the part does when nothing is
being decoded.

The `sorted` column is what decides which kernel a gather reaches:
`gather_qmm_rhs` takes one expert per row of a plain `[M, K]` input and is
reached only when the indices are declared sorted. Unsorted is the same call
through the matrix-per-row path, and the gap between them is not a claim about
this engine — it is what mlx's own routing pays for not sorting.

**`ragged` is the row this probe was missing.** Every figure it has ever printed
was taken over `arange(rows) * experts // rows`, which gives every expert the
same run of `6 * tokens / 256` rows — 48 at 2048, 96 at 4096, 192 at 8192, and
each of them a whole number of `gather_qmm_rhs`'s own 16-row tile. So the even
column asks that kernel only about run lengths its tile divides. A router's
counts are not those: `fp_gather_qmm_rhs` cuts a tile at each boundary the
sorted indices hold and runs a *whole* gemm over `K` for every run it finds, so
a boundary landing inside a tile costs it a second walk exactly as it costs this
engine's block a second pass. The `ragged` arm is the same rows and the same
experts with the boundaries where a router puts them, and it is the arm that says
whether 79% is the kernel or the fixture.
"""

import argparse
import time

import mlx.core as mx

# (what, rows a token, in_dim, out_dim, experts) — `BOUND_SHAPES` in
# `crates/inkling-metal/src/matmul.rs`, which is where the figures these sit
# beside are taken.
SHAPES = (
    ("q_proj", 1, 4096, 4096, 1),
    ("a routed bank", 6, 4096, 2048, 256),
)


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


def ragged(rows, experts, block=32):
    """The same rows over the same experts with the run boundaries where a
    router's own counts put them, rather than every run the same length, or
    `None` where the mean run is no longer than the tile and so holds no
    boundary that could land inside one.

    `each_way_runs_can_land` in `crates/inkling-metal/src/matmul.rs`, written
    out again here so that both kernels are asked the identical question — seven
    is coprime with `block`, so the offsets walk every alignment instead of
    alternating between two of them.
    """
    if rows // experts <= block:
        return None
    at = (
        [0]
        + [
            rows * e // experts + (e * 7 % block) - block // 2
            for e in range(1, experts)
        ]
        + [rows]
    )
    return [e for e in range(experts) for _ in range(at[e + 1] - at[e])]


def quantised(rows, in_dim, out_dim, experts, dtype, mode, group, bits, sort, runs):
    """One dispatch of the call this shape gives mlx, and what it took."""
    x = mx.random.normal((rows, 1, in_dim) if experts > 1 else (rows, in_dim))
    x = x.astype(dtype)
    w = mx.random.normal(
        (experts, out_dim, in_dim) if experts > 1 else (out_dim, in_dim)
    )
    packed = mx.quantize(w.astype(dtype), group_size=group, bits=bits, mode=mode)
    del w
    codes, scales, biases = (*packed, None)[:3]
    if experts == 1:
        mx.eval(x, codes, scales)
        return best_of(
            lambda: mx.quantized_matmul(
                x,
                codes,
                scales,
                biases,
                transpose=True,
                group_size=group,
                bits=bits,
                mode=mode,
            ),
            ROUNDS,
        )
    spread = ragged(rows, experts) if runs == "ragged" else None
    if runs == "ragged" and spread is None:
        return None
    at = (
        mx.array(spread, dtype=mx.uint32)
        if spread is not None
        else mx.arange(rows, dtype=mx.uint32) * experts // rows
    )
    mx.eval(x, codes, scales, at)
    return best_of(
        lambda: mx.gather_qmm(
            x,
            codes,
            scales,
            biases,
            rhs_indices=at,
            transpose=True,
            group_size=group,
            bits=bits,
            mode=mode,
            sorted_indices=sort,
        ),
        ROUNDS,
    )


def dense(rows, in_dim, out_dim, experts, dtype):
    """The same shape with nothing quantised, which no engine here runs and
    which is what says how much of the cost is the decode."""
    if experts > 1:
        return None
    x = mx.random.normal((rows, in_dim)).astype(dtype)
    w = mx.random.normal((in_dim, out_dim)).astype(dtype)
    mx.eval(x, w)
    return best_of(lambda: x @ w, ROUNDS)


ROUNDS = 3


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
    parser.add_argument(
        "--mode",
        default="mxfp4",
        choices=["mxfp4", "affine", "both"],
        help="mxfp4 is this checkpoint's format",
    )
    parser.add_argument("--dense", action="store_true", help="add the unquantised arm")
    args = parser.parse_args()

    global ROUNDS
    ROUNDS = args.rounds
    wanted = {"bfloat16": mx.bfloat16, "float32": mx.float32}
    dtypes = list(wanted.values()) if args.dtype == "both" else [wanted[args.dtype]]
    modes = ["mxfp4", "affine"] if args.mode == "both" else [args.mode]

    for tokens in args.tokens:
        print(f"\n  a prefill of {tokens} tokens")
        print(
            f"  {'':<16}{'dtype':<11}{'mode':<9}{'indices':<16}"
            f"{'device':>10}{'rate':>16}"
        )
        for what, a_token, in_dim, out_dim, experts in SHAPES:
            rows = tokens * a_token
            flop = 2 * rows * in_dim * out_dim
            for dtype in dtypes:
                name = "bfloat16" if dtype == mx.bfloat16 else "float32"
                for mode in modes:
                    sorts = (
                        [(True, "even"), (True, "ragged"), (False, "even")]
                        if experts > 1
                        else [(True, "even")]
                    )
                    for sort, runs in sorts:
                        taken = quantised(
                            rows,
                            in_dim,
                            out_dim,
                            experts,
                            dtype,
                            mode,
                            32,
                            4,
                            sort,
                            runs,
                        )
                        # A mean run no longer than the tile holds no
                        # boundary that could land inside one, so the arm is not
                        # a question at that length.
                        if taken is None:
                            continue
                        told = f"sorted, {runs}" if sort else "unsorted"
                        print(
                            f"  {what:<16}{name:<11}{mode:<9}{told:<16}"
                            f"{taken * 1e3:9.2f}ms{flop / taken / 1e12:11.1f} TFLOP/s"
                        )
                if args.dense:
                    taken = dense(rows, in_dim, out_dim, experts, dtype)
                    if taken is not None:
                        print(
                            f"  {what:<16}{name:<11}{'dense':<9}{'':<16}"
                            f"{taken * 1e3:9.2f}ms{flop / taken / 1e12:11.1f} TFLOP/s"
                        )


if __name__ == "__main__":
    main()
