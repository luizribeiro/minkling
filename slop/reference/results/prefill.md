# Prefill sweep — Inkling-Small-mxfp4

Measured on a 512 GiB M3 Ultra host via `just prefill-bench`, one
`language_model(..., skip_logits=True)` forward pass per row, `mx.eval`'d.
Model resident 130.6 GiB (140.27 GB) in every row. Raw numbers in
`prefill.json`.

**This sweep was re-measured.** The first pass ran without `mx.set_wired_limit`
held, which inflated the 16384 row by 2–3 s and pushed the scaling fit toward
the quadratic. Both sets of numbers are below; the memory numbers are unaffected
and unchanged.

| tokens | wall s | tok/s | peak GB | Δ over resident GB | mask GB | mask / Δ |
| -----: | -----: | ----: | ------: | -----------------: | ------: | -------: |
|   1024 |   1.41 | 728.2 |   143.0 |                2.8 |    0.07 |     2.4% |
|   4096 |   5.62 | 729.2 |   151.6 |               11.4 |    1.07 |     9.4% |
|   8192 |  13.08 | 626.4 |   167.2 |               27.0 |    4.29 |    15.9% |
|  16384 |  34.34 | 477.1 |   215.6 |               75.3 |   17.18 |    22.8% |
|  32768 |      — |     — |       — |                  — |   68.72 |        — |

32768 was refused unmeasured by the memory guard: projected peak 406 GiB
against a 350 GiB budget. See "Why 32768 is not here" below.

## What the wired limit was worth

`generate` raises the wired limit to the device's
`max_recommended_working_set_size` (464 GiB here) and restores it on exit.
`prefill_bench.py` drove `model.language_model(...)` directly and never set it,
so the published sweep ran at the default limit of 0 — nothing guaranteed
GPU-resident. The same omission costs a *decode* step 2.6 s instead of 32 ms
(`mtp_acceptance.md`), flat in token count, which is why the sweep was suspect.

Measured directly, by alternating `mx.set_wired_limit(max_rec)` and
`mx.set_wired_limit(0)` around repeats of the same prompt inside one process,
best of 4 (best of 2 at 16384):

| tokens | held s | default s | penalty | peak GiB |
| -----: | -----: | --------: | ------: | -------: |
|    128 |  0.313 |     0.314 |    1 ms |    131.0 |
|   1024 |  1.411 |     1.479 |   68 ms |    133.2 |
|   4096 |  5.562 |     5.656 |   94 ms |    141.2 |
|   8192 | 13.038 |    13.168 |  130 ms |    155.7 |
|  16384 | 34.334 |    36.556 | 2222 ms |    200.8 |

**The penalty is real but it is not the flat per-pass cost decode pays.** It
sits at 1–130 ms through 8192 and then jumps 17× to 2.2 s for a 2× step in
length. Held and default repeats do not overlap at any length, so it is a real
effect everywhere — but as a share of runtime it is 4.8% at 1024 and 1.0% at
8192 against 6.5% at 16384, and in absolute terms only the last row is worth
anything. The flat-overhead worry was wrong in shape: short prefills were never
meaningfully distorted. The decode penalty is a different mechanism (a 267 GiB
checkpoint against the same limit) and does not transfer to this one.

What the penalty tracks is peak memory, not token count — ~0 while the
transients are a few GB, seconds once they reach 75 GB. That is consistent with
a prefill's own transients evicting unwired weight pages: it allocates and frees
tens of GB per layer, and at the default limit nothing protects the 130 GiB
being read underneath it. Not directly instrumented, though, and a 17× jump
after three gently growing points would fit a threshold being crossed just as
well as a continuous effect.

Holding the limit also makes the measurement reproducible where it matters: the
two 16384 repeats agreed to 1 ms held, and spread over 36.6–38.0 s unheld.

**The published sweep was reproducible, not noisy**, so the correction is not an
artefact of re-running: a fresh unheld process reproduces its 1024 and 4096
numbers to within 1% (1.528 s and 5.590 s against 1.534 s and 5.642 s). But
sweep-to-sweep differences are the wrong thing to read the penalty off — the
old and new sweeps differ by 8.4% at 1024, where the controlled measurement puts
the penalty at 4.8%, and by +0.3% at 8192, where it should have got faster. Run
variance of order 0.1 s sits on top. Only at 16384 does the penalty clear that
by an order of magnitude, and 16384 is the row the scaling fit is anchored on.

## Scaling is not yet quadratic, and less close than first reported

|          step | wall ×, corrected | exponent | (as published) | Δ memory | exponent |
| ------------: | ----------------: | -------: | -------------: | -------: | -------: |
|  1024 → 4096  |             ×3.99 |     1.00 |  ×3.68 / 0.94  |    ×4.12 |     1.02 |
|  4096 → 8192  |             ×2.33 |     1.22 |  ×2.31 / 1.21  |    ×2.37 |     1.25 |
| 8192 → 16384  |             ×2.63 |     1.39 |  ×2.89 / 1.53  |    ×2.79 |     1.48 |

The exponent still climbs — 35 of 42 layers are capped at a 512-token window and
cost linear time, only the 7 global layers cost L² — but it climbs from 1.00 to
1.39, not to 1.53. The first step is now indistinguishable from linear.

Fitting `c1*L + c2*L**2` to wall time over the last two points gives **1097
µs/token and 60.98 ns/token²**, against 881 µs and 86.7 ns before — the linear
coefficient up 25%, the quadratic one down 30%, off one corrected row:

| quadratic share of runtime | 1024 | 4096 | 8192 | 16384 | 32768 (proj) |
| -------------------------- | ---: | ---: | ---: | ----: | -----------: |
| corrected                  |   5% |  19% |  31% |   48% |          65% |
| as published               |   9% |  29% |  45% |   62% |          76% |

**Crossover moves from ~10200 tokens to ~18000** — from just past the middle of
the sweep to just past its end. Attention's L² term does not become the majority
of prefill until beyond 16384.

A two-point fit is worth cross-checking now that all four rows are trustworthy.
Least squares over all four with a constant term gives `c0 = 0.267 s`,
`c1 = 1048 µs/token`, `c2 = 62.96 ns/token²`, and reproduces every measured row
to 0.01 s — crossover 16600, quadratic share 49% at 16384. The constant is not a
fitting artefact: a 128-token prefill measures 0.313 s where `c1*L` alone would
put it at 0.13 s. 42 layers of kernel launches and MoE gathers cost a quarter of
a second before any token-dependent work happens.

## The materialised mask does not dominate

**Unchanged by the re-measurement.** Peak memory came back byte-identical at
4096, 8192 and 16384 and within 0.03 GB at 1024 — the wired limit governs where
pages live, not how many are allocated. Everything in this section stands as
first measured.

The fit over the memory deltas gives **1.99 MB/token linear** and
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

## Read for the Rust engine: the conclusion survives, the argument for it does not

A fused attention that never forms [H, L, L] is still the first thing worth
writing — but the time-scaling case that was made for it was overstated, and it
is not what carries the decision.

**What no longer holds.** Prefill is not on the verge of going quadratic at the
lengths this host can reach. The quadratic share at 16384 is 48%, not 62%;
crossover is ~18000 tokens, not ~10200; the top exponent is 1.39, not 1.53. Up
to about 16k tokens prefill time is majority *linear* — MoE expert reads and 35
sliding-window layers — and a kernel that only attacks the L² term is attacking
the minority of the runtime. The corrected linear coefficient is 25% larger than
first reported (1097 vs 881 µs/token), so chunked prefill and the MoE path
matter more than the earlier numbers implied, and sooner.

**What holds, and decides it.** The memory argument is untouched. Removing the
materialised mask alone buys 23% of the gap at 16384; a fused attention removes
the mask *and* the score buffers it forces into existence — 57% of the gap at
16384, a projected 72% at 32768. And memory, not time, is the binding
constraint: 32768 is refused because it needs ~406 GiB, not because it would
take too long. Corrected, it projects to **101 s** of wall time against 122 s
before — same two-point fit both sides — which is unremarkable either way. The
reason this machine cannot prefill 32k tokens is bytes, and the mask-plus-scores
term is most of the bytes that grow.

So: build the fused kernel first because it is what makes long prefill *reachable
at all*, not because prefill time is about to go quadratic. The second thing is
the ~2 MB/token linear term, and it wants chunked prefill rather than a better
kernel.

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

## Which other scripts had the same exposure

Only three scripts under `reference/` report timings, and the other two were
already correct: `smoke.py` goes through mlx-vlm's `generate`, which raises and
restores the limit itself, and `mtp_acceptance.py` holds it explicitly.

The `dump_*.py` fixture scripts do drive the modules directly without it, but
they write tensors and a manifest of shapes — no timings — and the wired limit
governs page residency, not arithmetic. `dump_activations.py` included: it is
slower without the limit held and produces the same bytes. The committed
fixtures need no regeneration.
