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

A decode step is about 114 ms against mlx-vlm's 32 ms, and the timings go to
stderr so stdout stays pipeable. The prompt reaches the tokenizer as it stands,
so the model *continues* it rather than answering it. A chat turn is written out
in full — `<|message_user|><|content_text|>…<|end_message|><|message_model|>` —
rather than applied by a template this does not implement.

Every weight the model multiplies against runs on the GPU, against MXFP4 codes
it never decodes, and `--backend cpu` puts them all back: 0.114 s a token
against the CPU's 8.9. The experts were the first two thirds of that. A token reads 6 of
each MoE layer's 256 experts and both of its shared ones, which is 32 GB of
float32 the CPU path decodes to multiply against and 4.3 GB of packed bytes the
GPU path indexes into and never decodes at all. The rest is every layer's own
projections — five for attention, three more on each of the two dense layers —
which are 9 GB of float32 that *every* token reads all of.

**Nothing is copied onto the device.** The forty layers' banks are 137 GB, which
is the whole checkpoint but for its two ends, and they are handed to the GPU
where the checkpoint mapped them — `newBufferWithBytesNoCopy` over all of it in
6 ms. So the resident set goes *down* — 20.8 GiB with only the head there, 2.4
GiB once the banks are, and 0.19 GiB once the layers' own projections and norms
are too — and a bank nobody routes to costs nothing to have wrapped. Note what those
numbers stop meaning: the pages are still in the unified buffer cache, they are
simply no longer this process's.

**What a step costs is now mostly the asking, and that is measured rather than
inferred.** Every operation a forward pass runs opens a scope charged the time
inside it that no scope inside *it* claimed, so the rows of a decode step sum to
the step and what they leave over is a number rather than a shrug:

    submit and wait     249    60%      of which the device executed for 22 ms
    linear               40    19%      the routers' [258, 4096] gates
    weights decoded     500     8%      every layer's bfloat16 tensors, again
    dispatch encode     499     6%
    readback            457     2%
    everything else                     3%

Two thirds of the step is a round trip, and the device is executing for a
quarter of that — so the rest is 249 command buffers submitted and waited for
around work that was already done. Every activation op this engine has, the
attention step included, is 3% of a step together.

Multiplies that share an input already share a command buffer: the four
projections a layer's normed hidden state feeds, each expert bank's gate and up,
and now the norm that makes that hidden state — 499 dispatches in 249
submissions. **The norm is what a device-resident activation looks like.** Its
output is a buffer the four projections read directly, so the normed state is
never a `Vec<f32>` anywhere and the step costs one dispatch a layer more and not
one submission.

What is left on the CPU is the attention step itself — whose scores and softmax
multiply activations against activations and have no weight to hand over — plus
each layer's second norm, its four short convolutions, and the routers' own
`[258, 4096]` gate, which is the last matmul in the model running as a loop.
Both backends generate the same tokens, and the CPU one stays the oracle every
kernel here is validated against.

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

**The mask is materialised.** The reference builds a full `[B, H, LQ, S]`
additive mask. Acceptable when decoding, quadratic when prefilling. Fusing the
relative-position bias into a flash-attention kernel is where a custom engine
wins outright.

## Weights

The MXFP4 quant (`mlx-community/Inkling-Small-mxfp4`, 140 GB) has **no MTP
tensors** — they were stripped during quantisation. It is fine for text, vision,
audio, batching and perf work, but MTP requires the BF16 original
(`thinkingmachines/Inkling-Small`, 532 GB). The official NVFP4 keeps its MTP
weights but is in ModelOpt format, which mlx-vlm cannot read.
