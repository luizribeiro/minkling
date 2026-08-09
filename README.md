<p align="center">
  <img src="slop/minkling.png" alt="minkling" width="480">
</p>

Minkling is an AI-generated inference engine for
[Thinking Machine](https://thinkingmachines.ai/)'s
[Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small) model,
focused on fast inference on Apple Silicon with Metal kernels.

It has the following features:

* Packed MXFP4 inference on Metal, without expanding the model's weights in
  memory
* Speculative decoding using Inkling's native multi-token prediction heads
* Continuous batching, including admitting new requests into a running batch
* KV-cache reuse across multi-turn conversations, both serially and in batches
* An OpenAI-compatible chat completions server with streaming, tool calling,
  stop sequences, and token usage
* CPU and Metal backends, with selectable Metal numerics for validation and
  performance tuning

## Running

Minkling runs on Apple Silicon and requires the Xcode Command Line Tools and a
Rust toolchain. The provided Nix development shell includes Rust and the other
development tools:

```sh
nix develop
```

Download the roughly 140 GB
[MXFP4 checkpoint](https://huggingface.co/mlx-community/Inkling-Small-mxfp4)
from Hugging Face:

```sh
mkdir -p slop/models
uvx --from huggingface-hub hf download mlx-community/Inkling-Small-mxfp4 \
  --local-dir slop/models/Inkling-Small-mxfp4
```

Build the reviewed server and load the checkpoint:

```sh
cargo build --release --bin minkling

target/release/minkling serve slop/models/Inkling-Small-mxfp4 \
  --max-tokens 64 \
  --numerics production
```

`--numerics production` selects the faster prefill kernels. Omit it to use the
bit-reproducible reference numerics. `--max-tokens` is both the default and the
largest budget a request may ask for.

Then send an OpenAI-compatible request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64}'
```

The host listens on loopback by default, limits request bodies to 1 MiB, and
queues at most 16 requests for its single model-owning inference worker. It
also exposes `GET /healthz` and `GET /v1/models`.

This first host milestone returns collected completions. It explicitly rejects
`"stream": true`; streaming needs cancellation and backpressure at the worker
boundary and will be migrated separately. Continuous batching, streaming, and
speculative generation remain available in the quarantined `inklingrs` CLI and
are documented in [`slop/README.md`](slop/README.md).
