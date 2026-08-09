<p align="center">
  <img src="slop/minkling.png" alt="minkling" width="480">
</p>

Minkling is an AI-generated inference engine for
[Thinking Machine](https://thinkingmachines.ai/)'s
[Inkling Small](https://huggingface.co/thinkingmachines/Inkling-Small) model,
served through an OpenAI-compatible HTTP API on Apple Silicon.

## Run

Install the Xcode Command Line Tools and [Nix](https://nixos.org/), then download
the roughly 140 GB MXFP4 checkpoint:

```sh
mkdir -p slop/models
nix develop --command uvx --from huggingface-hub hf download \
  mlx-community/Inkling-Small-mxfp4 \
  --local-dir slop/models/Inkling-Small-mxfp4
```

Start Minkling:

```sh
nix develop --command cargo run --release --bin minkling -- \
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
