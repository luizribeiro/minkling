<p align="center">
  <img src="slop/minkling.png" alt="minkling" width="480">
</p>

Minkling is an AI-generated inference engine for
[Thinking Machine](https://thinkingmachines.ai/)'s
[Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small) model,
served through an OpenAI-compatible HTTP API on Apple Silicon.

## Features

* Packed MXFP4 inference on Metal
* Native multi-token prediction and speculative decoding
* Continuous batching and KV-cache reuse
* A CPU reference backend and selectable Metal numerics
* Collected and streaming OpenAI-compatible chat completions
* Tool calling, stop sequences, and token usage
* Bounded request queues and cancellation on client disconnect

`minkling serve` currently uses the single-request, single-token decode path;
MTP and continuous batching are engine capabilities still to be wired into the
new host.

## Run

Install the Xcode Command Line Tools and [Nix](https://nixos.org/), then download
the roughly 140 GB MXFP4 checkpoint:

```sh
nix develop

mkdir -p slop/models
uvx --from huggingface-hub hf download \
  mlx-community/Inkling-Small-mxfp4 \
  --local-dir slop/models/Inkling-Small-mxfp4
```

Start Minkling:

```sh
cargo run --release --bin minkling -- \
  serve slop/models/Inkling-Small-mxfp4 \
  --numerics production
```

Send a streaming chat completion:

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64,"stream":true}'
```

Set `"stream": false` for a collected response. The server also exposes
`GET /healthz` and `GET /v1/models`.
