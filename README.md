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

Keep the budget small: a decode step is about 3.2 s against mlx-vlm's 32 ms, and
the timings go to stderr so stdout stays pipeable. The prompt reaches the
tokenizer as it stands, so the model *continues* it rather
than answering it. A chat turn is written out in full —
`<|message_user|><|content_text|>…<|end_message|><|message_model|>` — rather than
applied by a template this does not implement.

The routed experts and `lm_head` run on the GPU, against MXFP4 codes they never
decode, and `--backend cpu` puts them back: 3.2 s a token against the CPU's 8.9.
The experts are where that comes from. A token reads 6 of each MoE layer's 256
experts and both of its shared ones, which is 32 GB of float32 the CPU path
decodes to multiply against and 4.3 GB of packed bytes the GPU path indexes into
and never decodes at all — 73% of a decode step, gone in one dispatch per
projection per layer.

**Nothing is copied onto the device.** The forty layers' banks are 137 GB, which
is the whole checkpoint but for its two ends, and they are handed to the GPU
where the checkpoint mapped them — `newBufferWithBytesNoCopy` over all of it in
6 ms. So the resident set goes *down*, 20.8 GiB to 2.4 GiB, and a bank nobody
routes to costs nothing to have wrapped. Note what that number stops meaning:
those pages are still in the unified buffer cache, they are simply no longer
this process's.

What is left is the 9 GB of attention and dense-FFN projections the CPU still
decodes, which is 78% of a step — and of that, two thirds is the *multiply*
rather than the decode, a serial f32 dot product no compiler may vectorise. Both
backends generate the same tokens, and the CPU one stays the oracle every kernel
here is validated against.

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

- `inkling-config.py.patch` — mlx-vlm reads `intermediate_size` as the dense FFN
  width, but Inkling calls the dense width `dense_intermediate_size` and uses
  `intermediate_size` for the per-expert width. Unpatched, both are wrong for
  Inkling-Small, and Inkling-975B's two dense layers load at 3072 instead of
  24576.
- `inkling-inkling.py.patch` — the MXFP4 quant carries identity
  `switch_mlp.{gate,out}_scale` tensors with no counterpart in the model, which
  abort a strict load. Dropped, with a guard that refuses any non-identity value.

## Architecture notes

42 layers, hidden 4096, 256 routed experts (top-6) plus 2 shared, 276B total /
12B active. No RoPE — position comes from depthwise causal short convolutions
(kernel 4, on q/k/v and on the attention/MLP inputs) plus a learned relative
logit bias over a 1024-token extent.

Three properties drive the design:

**Attention is 5:1 local:global.** Layers 5, 11, 17, 23, 29, 35 and 41 are full
attention; the other 35 are capped at a 512-token window. Only the 7 global
layers grow with sequence length, so KV costs 28 KiB/token plus a fixed 73 MiB
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
