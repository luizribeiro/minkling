# Prefill sweep — Inkling-Small-mxfp4

Measured on a 512 GiB M-series host via `just prefill-bench`, one
`language_model(..., skip_logits=True)` forward pass per row, `mx.eval`'d.
Model resident 130.6 GiB (140.27 GB) in every row. Raw numbers in
`prefill.json`.

| tokens | wall s | tok/s | peak GB | Δ over resident GB | mask GB | mask / Δ |
| -----: | -----: | ----: | ------: | -----------------: | ------: | -------: |
|   1024 |   1.53 | 667.4 |   143.1 |                2.8 |    0.07 |     2.4% |
|   4096 |   5.64 | 726.0 |   151.6 |               11.4 |    1.07 |     9.4% |
|   8192 |  13.04 | 628.2 |   167.2 |               27.0 |    4.29 |    15.9% |
|  16384 |  37.72 | 434.4 |   215.6 |               75.3 |   17.18 |    22.8% |
|  32768 |      — |     — |       — |                  — |   68.72 |        — |

32768 was refused unmeasured by the memory guard: projected peak 406 GiB
against a 350 GiB budget. See "Why 32768 is not here" below.

## Scaling is not yet quadratic, but it is heading there

|          step | wall time | exponent | Δ memory | exponent |
| ------------: | --------: | -------: | -------: | -------: |
|  1024 → 4096  |    ×3.68  |     0.94 |    ×4.07 |     1.01 |
|  4096 → 8192  |    ×2.31  |     1.21 |    ×2.37 |     1.25 |
| 8192 → 16384  |    ×2.89  |     1.53 |    ×2.79 |     1.48 |

The mask grows as L², but neither time nor memory does — the exponent climbs
from ~1.0 to ~1.5 across the sweep. This follows directly from the 5:1 layer
split: 35 of 42 layers are capped at a 512-token window and cost linear time,
only the 7 global layers cost L². Fitting `c1*L + c2*L**2` to wall time over
the last two points gives 881 µs/token and 86.7 ns/token², so the quadratic
share of runtime is 9% at 1024, 45% at 8192, 62% at 16384, and a projected 76%
at 32768. Crossover sits near 8192.

## The materialised mask does not dominate

The same fit over the memory deltas gives **1.99 MB/token linear** and
**159.3 B/token² quadratic**. The mask's own quadratic coefficient is
`32 heads × 2 bytes = 64 B/token²`, so the mask is only **2.49×** smaller than
the whole quadratic term — and the quadratic term is itself only part of the
gap.

At 16384, where peak sat 75.3 GB above resident against a 17.18 GB mask:

| component               | GB   | share of the gap |
| ----------------------- | ---- | ---------------: |
| linear, 1.99 MB/token   | 32.5 |              43% |
| mask, [B, H, L, L]      | 17.2 |              23% |
| other quadratic buffers | 25.6 |              34% |

The mask is under a quarter of it. The other two thirds:

- **~25.6 GB of non-mask L² buffers.** An explicit additive mask of shape
  [B, H, LQ, S] disqualifies MLX's fused SDPA kernel, so attention materialises
  the score matrix and its softmax as well. The residual 95.3 B/token² is
  roughly one more score-sized buffer plus a softmax transient — the mask does
  not just cost its own bytes, it forces the scores to be spelled out too.
- **~32.5 GB of linear working set**, at ~2 MB/token: the sliding-window masks
  ([H, L, 512] is 32 KB/token), KV for the 7 global layers at 28 KiB/token,
  hidden/residual transients, router logits over 256 experts, and the MoE
  expert activations.

Peak is a maximum over time, not a sum over layers, so these are one layer's
simultaneous working set rather than 42 layers accumulated.

**Read for the Rust engine:** removing the materialised mask alone buys 23% at
16384 and would not change the shape of the curve. A fused attention that never
forms [H, L, L] at all removes the mask *and* the score buffers — 57% of the gap
at 16384, a projected 72% at 32768. That is the first thing worth writing.
The remaining 2 MB/token linear term is what then sets the ceiling, and it wants
chunked prefill rather than a better kernel.

## Why 32768 is not here

Projection from the 8192/16384 pair puts the peak at 351 GiB raw, 406 GiB with
the guard's 1.25 margin. The margin is calibrated, not arbitrary: predicting
16384 from the 4096/8192 pair gave 170 GiB raw against 200.8 GiB actual, a 15%
under-read that the margin converts into a 6% over-read. So ~400–410 GiB is the
honest expectation for 32768, against roughly 445 GiB actually available with
the vllm-mlx daemons resident — about 8% of machine RAM as slack.

An earlier uncapped attempt at 65536 (projected ~1040 GiB) took the host down.
Swap does not rescue an overshoot at this scale: it is grown dynamically, reads
0 at rest, and reached only ~26 GB during that crash — orders of magnitude
short of the overshoot, and far too slow to matter against an allocation rate
like this.

`mx.set_memory_limit` is **not** a cap and must not be trusted as one. Its own
documentation calls it a guideline that raises only once the system is out of
RAM and swap; measured directly, an 8 GiB allocation against a 4 GiB limit
succeeds silently. The guard in `prefill_bench.py` is therefore predictive — it
extrapolates from rows already measured and refuses a length before allocating
anything — with `set_memory_limit` kept only as a last-ditch net.
