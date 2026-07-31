# MTP acceptance — Inkling-Small-8bit

Measured via `just mtp-acceptance`, greedy decoding over 6 prompts, **2219
generated tokens** scored at **2171 positions** per depth. Model resident
266.7 GiB (262.5 GiB main stack + 4.2 GiB of BF16 MTP weights). Raw numbers in
`mtp_acceptance.json`.

The headline: **the heads are good, the verify pass is expensive, and the two
roughly cancel.** Pooled speedup peaks at **1.32x at 2 speculated tokens**, and
ranges from 1.04x on prose to 1.98x on enumeration. This is greedy-decoding
agreement, an upper bound (see "What this is not" below).

## The architecture, read off the tensors

160 tensors under `model.mtp.*`, 20 per head across 8 heads, all BF16 — the
8-bit quantiser left them alone. They live in their own `mtp.safetensors`
shard, which is why `sanitize`'s `if ".mtp" in k` drop can be sidestepped by
loading the shard directly rather than patching mlx-vlm.

One head is:

    hidden ─→ hidden_norm ─┐
                           ├─→ concat[hidden;embed] ─→ input_proj ─→ block
    embed  ─→ embed_norm  ─┘         [4096, 8192]      → 4096

    block = a decoder layer: attn (q/k/v sconv, banded rel bias)
            + dense SwiGLU MLP at 16384, times a learned global_scale

Head `d` at position `i` consumes the chained hidden state and the embedding of
token `i+d+1`, and predicts token `i+d+2`. Head 0's hidden input is the main
stack's; head `d`'s is head `d-1`'s output, passed raw.

Three things the names and shapes fix outright:

- **Dense, not MoE.** `mlp.w13_dn` is `[32768, 4096]` and `mlp.w2_md` is
  `[4096, 16384]` — the fused gate/up and down of a dense SwiGLU at
  `dense_intermediate_size`. No `gate_weight`, no `switch_mlp`, no
  `e_score_correction_bias`. A head is ~245M parameters against the main
  stack's ~12B active, so all 8 cost 4.2 GiB of BF16 reads.
- **Their own local/global split.** `mtp_config.local_layer_ids` is
  `[0, 2, 4, 5, 6, 7]`, so heads 1 and 3 are global. Confirmed by
  `rel_logits_proj.proj`: `[16, 1024]` on heads 1 and 3, `[16, 512]` on the
  other six, exactly as `InklingAttention` picks `rel_extent` from
  `sliding_window_size` vs `rel_extent`.
- **Upstream names.** The main stack was rewritten to mlx-vlm's names by the
  quantiser but the MTP tensors kept theirs (`attn.wq_du`, `attn_norm`,
  `mlp.w13_dn`), so `Model._map_llm_layer` maps a head's `transformer_block`
  unchanged.

### Two things the tensors do not fix, resolved by measurement

`input_proj` being `[4096, 8192]` says it eats two normed 4096-wide vectors but
not in which order, and the checkpoint's embedding is already normed by the main
stack's `embed_norm`, so whether a head's own `embed_norm` stacks on that or
replaces it is undetermined. All 8 combinations were scored:

| concat order      | embedding fed to head | main hidden | depth-1 |
| ----------------- | --------------------- | ----------- | ------: |
| `[hidden; embed]` | `embed_norm(E)`       | post-norm   |  76.4% |
| `[hidden; embed]` | `embed_norm(E)`       | pre-norm    |  73.2% |
| `[hidden; embed]` | raw `E`               | post-norm   |  66.9% |
| `[hidden; embed]` | raw `E`               | pre-norm    |  48.8% |
| `[embed; hidden]` | either                | either      |  ≤0.8% |

Concat order is settled by one head: reversed, the projection sees the hidden
state in the half of the weight trained for embeddings and agrees with the model
on **nothing**. The two normalisation choices degrade gently instead and needed
the full 2171 positions to separate — at one head and 128 tokens they sat within
noise of each other. Pooled over the real measurement:

| wiring                                       | depth 1 | depth 8 joint | tok/round |
| -------------------------------------------- | ------: | ------------: | --------: |
| `[h;e]`, `embed_norm(E)`, **post-norm hidden** |  77.7% |         29.4% |     4.600 |
| `[h;e]`, `embed_norm(E)`, pre-norm hidden      |  74.3% |         25.0% |     4.247 |
| `[h;e]`, raw `E`, post-norm hidden             |  65.6% |          0.3% |     2.239 |
| `[h;e]`, raw `E`, pre-norm hidden              |  53.9% |          0.1% |     1.906 |

So the heads consume the **post-final-norm** hidden state and the **doubly
normed** embedding. The post-norm result is worth stating carefully, because
`mtp_config` contains `chain_hidden_post_norm: false` and that reads like a
contradiction. It is not: the flag governs the links *between* heads, which are
raw here, and the winning configuration leaves them raw. It says nothing about
the main stack → head 0 link, which is where the final norm is applied. Both
halves of the config are satisfied.

The 3.4-point depth-1 gap between the top two rows is ~4 standard errors on
2171 paired positions, so it is a real effect rather than a coin flip — but it
is small, and an implementation that used the pre-norm hidden would lose about
8% of its accepted tokens rather than break.

## Per-depth acceptance

**Marginal** is head `d` alone; **joint** is depths 1..d all correct, which is
what a round actually banks, since a rejected head discards everything after it.

| prompt      | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | regime |
| ----------- | --: | --: | --: | --: | --: | --: | --: | --: | --- |
| enumeration | 99.7 | 98.1 | 97.6 | 97.3 | 97.3 | 97.1 | 96.8 | 96.8 | rigid repetition |
| json        | 91.8 | 82.7 | 75.0 | 67.8 | 58.8 | 49.7 | 43.6 | 37.5 | structured |
| code        | 85.4 | 66.0 | 55.6 | 46.3 | 39.6 | 36.4 | 33.8 | 31.4 | code |
| table       | 80.4 | 56.0 | 41.6 | 29.2 | 19.9 | 12.7 |  7.6 |  4.5 | structured + facts |
| technical   | 64.4 | 32.7 | 19.7 |  9.3 |  6.1 |  3.5 |  1.6 |  0.5 | expository prose |
| prose       | 44.9 | 13.3 |  4.0 |  1.1 |  0.3 |  0.0 |  0.0 |  0.0 | free prose |
| **pooled**  | **77.7** | **58.2** | **49.2** | **42.3** | **37.7** | **34.0** | **31.5** | **29.4** | |

Marginal acceptance, pooled: 77.7, 69.5, 69.3, 67.9, 68.1, 67.9, 66.7, 67.3.

**The decay is not in the heads.** Marginal acceptance is flat from depth 2
onward — head 8 is as good at its job (67.3%) as head 2 is at its own (69.5%).
The joint curve falls only because independent-ish 68% events multiply. Head 8
is not a worse head; it is a head that rarely gets asked. That matters for the
engine: there is no depth past which the heads stop being worth loading, so the
speculation depth is decided purely by the cost of the verify pass.

**The spread between regimes is larger than the spread between depths.** Joint
acceptance at depth 8 ranges over three orders of magnitude, 96.8% to 0.0%. Any
single-prompt measurement of this would have been worthless — prose alone says
MTP is dead, enumeration alone says it triples throughput.

## What a round costs

Timed against a warm 512-token cache on real decoded tokens, 12 repeats.
Decode step: **31.8 ms** (31.5 tok/s, consistent with the 33.4 tok/s baseline).

| speculated k | verify k+1 tok | ×decode | MTP chain | ×decode | round ×decode |
| -----------: | -------------: | ------: | --------: | ------: | ------------: |
| 0 | 31.8 ms | 1.00 | — | — | 1.00 |
| 1 | 40.3 ms | 1.27 | 3.9 ms | 0.12 | 1.39 |
| 2 | 49.1 ms | 1.55 | 7.7 ms | 0.24 | 1.79 |
| 3 | 59.0 ms | 1.86 | 12.0 ms | 0.38 | 2.24 |
| 4 | 68.7 ms | 2.16 | 16.0 ms | 0.50 | 2.67 |
| 6 | 88.7 ms | 2.79 | 21.5 ms | 0.68 | 3.47 |
| 8 | 115.9 ms | 3.65 | 28.7 ms | 0.90 | 4.55 |

**A verify pass is not free the way it is on a dense model.** Each extra token
in the block costs ~10.5 ms, a third of a full decode step. This is the MoE: one
token reads 6 routed experts per layer, nine tokens read up to 54, and the
expert bank is where the weight is. Speculation's usual bargain — that verifying
k tokens costs about what decoding one does, because you re-read the same
weights — does not hold under top-6-of-256 routing.

That sets a hard floor. An extra speculated token pays only if its joint
acceptance exceeds roughly the ~0.44 decode-steps it costs (0.33 verify + 0.11
chain). Pooled joint acceptance crosses that between depth 3 (49.2%) and depth 4
(42.3%), which is exactly where the measured optimum sits.

The 4.2 GiB of MTP weights should cost ~0.39 of a decode step to read at all 8
heads; the chain measures 0.90. The gap is per-head launch overhead in the
reference, and is the one number here a Rust engine should straightforwardly
beat.

## Expected speedup

| k | tok/round | round ×decode | speedup |
| -: | --------: | ------------: | ------: |
| 0 | 1.000 | 1.000 | 1.000 |
| 1 | 1.777 | 1.392 | 1.276 |
| **2** | **2.359** | **1.789** | **1.318** |
| 3 | 2.851 | 2.237 | 1.275 |
| 4 | 3.274 | 2.666 | 1.228 |
| 6 | 3.991 | 3.472 | 1.150 |
| 8 | 4.600 | 4.553 | 1.010 |

Pooled optimum **1.32x at k=2**. Speculating all 8 heads is worth **1.01x** —
the checkpoint ships five more heads than are worth running on a mixed
workload. Per prompt, at each one's own best depth:

| prompt | best k | tok/round | speedup |
| --- | --: | --: | --: |
| enumeration | 6 | 6.87 | **1.98x** |
| json | 4 | 4.17 | **1.57x** |
| code | 2 | 2.51 | **1.40x** |
| table | 2 | 2.36 | 1.32x |
| technical | 1 | 1.64 | 1.18x |
| prose | 1 | 1.45 | 1.04x |

The optimal depth is workload-dependent and varies by 6x in payoff, which
argues for choosing k adaptively from a running acceptance estimate rather than
fixing it.

## What this is not

- **Greedy agreement, not accept-reject.** Acceptance here is "head's argmax ==
  the argmax the model produced". A real sampling scheme must agree on the
  *distribution* and rejects on a random draw, so these are ceilings. At
  temperature 0 they are exact.
- **Reference costs, not engine costs.** The verify timings are mlx-vlm's:
  materialised `[B, H, LQ, S]` masks and MLX's `SwitchGLU` gather. The MoE
  expert-bandwidth growth is fundamental; the per-head chain overhead is not.
  A better engine moves the cost column, not the acceptance column.
- **Single sequence, no batching.** Under continuous batching a verify block
  competes with other sequences' tokens for the same expert reads, which likely
  makes speculation *cheaper* per token at high load — untested here, and the
  interaction is the open question.
- **Short contexts.** 512-token cache, ≤440-token sequences. Acceptance is
  plausibly context-dependent and this says nothing about 100k.

## Read: worth building on, not worth building around

The heads work. 77.7% at depth 1 clears the 60% bar comfortably, and marginal
acceptance stays near 68% all the way to head 8 — this is a well-trained MTP
stack, not a vestigial one. Standing it up cost no changes to mlx-vlm at all.

But acceptance was the wrong thing to be nervous about. **The cost of the verify
pass is what decides this, and fine-grained MoE makes it expensive.** A 1.32x
pooled speedup is a real win and worth having; it is not a differentiator you
design a scheduler around, and it does not carry the project. Continuous
batching has to.

Two things follow for the engine. First, build MTP as an *optional shallow*
mode — k=2 or 3, adaptive on a running acceptance estimate — rather than a
fixed 8-deep pipeline; the depth that pays varies 6x across workloads and 8 deep
never wins. Second, and more consequential: MTP rejection and continuous
batching contend for the same short-conv restore-and-replay path (README,
"Short-conv state cannot be trimmed"). Given that speculation buys 1.3x while
batching should buy several times that, **if the two designs conflict, batching
wins.** Do not let a speculative-decoding design constrain the batching
scheduler.
