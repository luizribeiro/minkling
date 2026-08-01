# inklingrs

An inference engine for [Inkling-Small](https://thinkingmachines.ai/news/inkling-small/)
on Apple Silicon, in Rust + Metal. The two things it aims to do that existing
runtimes do not: **multi-token prediction** and **continuous batching**.

## Layout

    crates/inkling-core     config, checkpoint loading, architecture
    crates/inkling-metal    Metal backend; kernels compiled at runtime
    crates/inkling-cli      binary
    reference/              Python mlx-vlm oracle, kept out of the Rust tree
    models/                 weights (gitignored)

`inkling-serve` splits out of `inkling-cli` once the batching scheduler is more
than a request loop.

## Getting started

    direnv allow          # or: nix develop
    just sync             # reference venv + mlx-vlm patches
    just test

Text in, text out, streamed to stdout as each token is decoded:

    inklingrs generate models/Inkling-Small-mxfp4 --prompt 'The lighthouse keeper' -n 4

A decode step is about 75 ms against mlx-vlm's 32 ms, and the timings go to
stderr so stdout stays pipeable. The prompt reaches the tokenizer as it stands,
so the model *continues* it rather than answering it. A chat turn is written out
in full — `<|message_user|><|content_text|>…<|end_message|><|message_model|>` —
rather than applied by a template this does not implement.

**Every matmul in the model runs on the GPU, and no weight one of them reads is
ever decoded to memory** — the MXFP4 ones in registers a nibble at a time, the
routers' bfloat16 gates by a shift — and `--backend cpu` puts them all back:
0.075 s a token against the CPU's 9.0. The experts were the first two thirds of
that. A token reads 6 of each MoE layer's 256 experts and both of its shared
ones, which is 32 GB of float32 the CPU path decodes to multiply against and 4.3
GB of packed bytes the GPU path indexes into and never decodes at all. The rest
is every layer's own projections — five for attention, three more on each of the
two dense layers — which are 9 GB of float32 that *every* token reads all of.

**Nothing is copied onto the device.** The forty layers' banks are 137 GB, which
is the whole checkpoint but for its two ends, and they are handed to the GPU
where the checkpoint mapped them — `newBufferWithBytesNoCopy` over all of it in
6 ms. So the resident set goes *down* — 20.8 GiB with only the head there, 2.4
GiB once the banks are, and 0.12 GiB once the layers' own projections, norms and
gates are too — and a bank nobody routes to costs nothing to have wrapped. Note
what those numbers stop meaning: the pages are still in the unified buffer
cache, they are simply no longer this process's.

The one thing here that is allocated rather than mapped is the keys and values,
and it is the only part of the footprint that grows with the sequence: each
layer keeps its own `[kv_heads, capacity, head_dim]` span of each and doubles it
when a sequence outruns it, which at 64 slots is 21 MB across the stack and puts
an eight-token generation at 0.14 GiB.

**What a step costs is now mostly the asking, and that is measured rather than
inferred.** Every operation a forward pass runs opens a scope charged the time
inside it that no scope inside *it* claimed, so the rows of a decode step sum to
the step and what they leave over is a number rather than a shrug:

    submit and wait     167    81%      of which the device executed for 25 ms
    dispatch encode     749    10%
    readback            329     3%
    everything else                     6%

**Six rows have left it since that table was first written, and the shape of
what is left has not changed.** The routers' gates were 19% and every layer's
bfloat16 tensors widened again on every step were 8%; the first is a dispatch now
and the second happens once, at load. The attention step and the mask it added
were 1.4% between them, and the two are one dispatch now. The two short
convolutions inside attention and the two head norms beside them were 0.8% by
the rows that named them, and they are four more dispatches — what they cost was
never their own time. Four fifths of a step is a round trip, and the device is
executing for two fifths of that — so the rest is 167 command buffers submitted
and waited for around work that was already done. Every activation op left is
3.8% of a step together.

**A dispatch's shape is not an allocation.** Each of the 749 took its dimensions,
its offsets and the expert its rows go through in small `MTLBuffer`s of its own —
1374 a step, made and freed between two steps that wanted the same values — where
`setBytes:` puts the same bytes in the command buffer as the dispatch is encoded.
That took the encode row from 9.35 ms to 7.28 ms, measured against the commit
before it and alternating between the two over seven pairs, with the wait row and
the device's own clock where they were. It stays a *copy* per dispatch rather than
a buffer the layer holds, which is what lets two calls of different heights share
a command buffer — and 994 allocations are left in that row, every one of them an
output or a row gathered for a bank.

Multiplies that share an input share a command buffer, and so do multiplies that
share nothing: the four projections a layer's normed hidden state feeds, the norm
that makes it, the two convolutions and two head norms behind two of them, the
attention step beside the projection it feeds, each expert bank's gate and up,
the router's own gate beside the shared bank it weights, and that bank's last
multiply beside the routed bank's first — 749 dispatches in 167 submissions.
**What those last two have in common
is that a seam had to be able to express them.** The gate's answer decides how
heavily each shared expert counts and not which rows it runs; and by the time the
shared bank's `down` is encoded, the routing that names the routed bank's rows
has already been taken from logits the gate handed over — so neither pair waits
for itself, and 40 gates and 40 round trips cost nothing. Handing a backend one
bank at a time, neither is visible.

Removing those 40 submissions took the wait row from 249 to 209 and about 6.3 ms
off it, measured against the commit before it and alternating between the two
over seven pairs: the device is executing for the same 23 ms and the dispatch
count does not move, so what is left is the round trip. A submission is 206
microseconds measured on its own and about 157 measured as the fortieth taken
out of a stream of 249 — two numbers this project has to keep apart, since only
the second says what merging a command buffer is worth.

**A layer's attention is now one submission rather than two**, and the same
alternating measurement says what that was worth: 209 command buffers to 167,
7.2 ms off the wait row and 8.8 ms off the step over seven pairs. The dispatch
count went the other way — 581 to 749, four more a layer — and cost nothing to
encode, because what came off beside them is a copy of the whole cached span
onto the device per layer per step. At 42 round trips the wait says a marginal
submission is about 172 microseconds here, a little above the 157 the fortieth
of 249 measured.

**The attention step is a dispatch even though it has no weight to hand over**,
and what it hands over instead is a tensor nobody builds. The reference adds a
materialised `[B, H, LQ, S]` mask to its logits; the kernel derives each entry
from the backward distance where it scores the key it belongs to, so the mask and
the scores it forces alongside it are never allocated. Over the eight-token
context that profile is taken across, that is a wash — 42 more dispatches cost
about the millisecond the CPU's own scores and mask cost — and it is not what the
kernel is for. A 769-token prompt prefills in 13.6 s against 55.4 s, and the gap
widens with the prompt: ×1.3 at 97 tokens, ×2.4 at 385, ×4.1 at 769.

**Everything a layer's attention does is now one command buffer**: the input
layernorm, the four projections that read it, the two short convolutions behind
the key and the value, the two head norms over the query and the convolved key,
the attention step and `o_proj` — eleven dispatches and one submission, with
every value between them a buffer the next dispatch reads. Three of them write
state that outlives the call — a convolution leaves its window, and between them
the key's norm and the value's convolution leave the span the step attends over
— so where those three ran decided where that state lives, and holding it is
what let the rest follow.

What is left on the CPU is each layer's second norm, the two short convolutions
on its residual path, and the router's top-k and softmax over eight numbers.
There is no matmul left outside the GPU. Both backends generate the same tokens,
and the CPU one stays the oracle every kernel here is validated against.

Or the same model behind an OpenAI-compatible endpoint, loaded once:

    inklingrs serve models/Inkling-Small-mxfp4

    curl -sN http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}'

`POST /v1/chat/completions`, streaming and collected, plus `GET /v1/models`.
Here the turn structure *is* applied — hard-coded rather than interpreted from
`chat_template.jinja`, and checked against what that template renders — because
without it nothing puts the model in a turn it could end and every request runs
to `max_tokens`. The model's thinking channel arrives in `reasoning_content` and
its answer in `content`, with the markers themselves in neither.

One request at a time; a second client waits. Batching is the scheduler's job
and the scheduler is the reason this engine exists.

## Why the reference directory exists

`sconv`, the banded relative-position bias, and sigmoid-gated top-6-of-256
routing cannot be validated by reading generated text. `reference/` is a
patched mlx-vlm used for layer-by-layer tensor comparison — an oracle, not a
dependency of the engine.

Two patches are needed before it loads Inkling-Small at all:

- `03-config-field-names.patch` — mlx-vlm reads `intermediate_size` as the dense
  FFN width, but Inkling calls the dense width `dense_intermediate_size` and uses
  `intermediate_size` for the per-expert width. Unpatched, both are wrong for
  Inkling-Small, and Inkling-975B's two dense layers load at 3072 instead of
  24576.
- `04-drop-identity-expert-scales.patch` — the MXFP4 quant carries identity
  `switch_mlp.{gate,out}_scale` tensors with no counterpart in the model, which
  abort a strict load. Dropped, with a guard that refuses any non-identity value.

The other three — a model-type remap, submodule configs exported for the dump
scripts, and a tap on the MoE router — are what the fixtures are captured
through rather than what makes the checkpoint load.

## Architecture notes

42 layers, hidden 4096, 256 routed experts (top-6) plus 2 shared, 276B total /
12B active. No RoPE — position comes from depthwise causal short convolutions
(kernel 4, on the key and value inside attention and on what attention and the
MLP produced, before each residual add) plus a learned relative logit bias over
a 1024-token extent.

Three properties drive the design:

**Attention is 5:1 local:global.** Layers 5, 11, 17, 23, 29, 35 and 41 are full
attention; the other 35 are capped at a 512-token window. Only the 7 global
layers grow with sequence length, so KV costs 28 KiB/token plus a fixed 70 MiB
per sequence — a 1M-token context fits in under 30 GiB. This is what makes deep
batching plausible on one machine.

**Short-conv state cannot be trimmed.** It keeps only the last `K-1` inputs, so
rejected speculative tokens need restore-and-replay rather than truncation.
Reordering along the batch dimension is fine, so continuous batching works, but
MTP rejection and batching meet here and this is the hard part of the engine.

**The reference materialises the mask.** It builds a full `[B, H, LQ, S]`
additive tensor — acceptable when decoding, quadratic when prefilling, and an
explicit additive mask of that shape also disqualifies MLX's own fused SDPA, so
the scores get spelled out beside it. Together they are 57% of what a
16384-token prefill allocates over the resident weights, and 32768 tokens are
refused at a projected 406 GiB. `--backend metal` builds neither: the
relative-position bias is computed per element inside the attention kernel,
which is where a custom engine wins outright.

## Weights

The MXFP4 quant (`mlx-community/Inkling-Small-mxfp4`, 140 GB) has **no MTP
tensors** — they were stripped during quantisation. It is fine for text, vision,
audio, batching and perf work, but MTP requires the BF16 original
(`thinkingmachines/Inkling-Small`, 532 GB). The official NVFP4 keeps its MTP
weights but is in ModelOpt format, which mlx-vlm cannot read.
