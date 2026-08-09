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

Build the release binary and continue a prompt:

```sh
cargo build --release --bin inklingrs

target/release/inklingrs generate slop/models/Inkling-Small-mxfp4 \
  --prompt "The lighthouse keeper" \
  --max-tokens 64 \
  --numerics production
```

`--numerics production` selects the faster prefill kernels. Omit it to use the
bit-reproducible reference numerics.

`generate` continues the prompt exactly as provided; it does not apply the
model's chat template. To serve templated chat completions instead, start the
HTTP server:

```sh
target/release/inklingrs serve slop/models/Inkling-Small-mxfp4 \
  --numerics production \
  --slots 4
```

Then send an OpenAI-compatible request:

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64,"stream":true}'
```

The server uses one slot by default. Set `--slots N` to advance up to `N`
requests together, or omit it for the lowest memory use. You can also inspect a
checkpoint without loading its weights:

```sh
target/release/inklingrs inspect slop/models/Inkling-Small-mxfp4/config.json
```

### Speculative decoding

The published MXFP4 checkpoint declares the model's eight MTP heads but omits
their weights. Download the original BF16 head shard into the checkpoint:

```sh
uvx --from huggingface-hub hf download thinkingmachines/Inkling-Small \
  mtp.safetensors \
  --local-dir slop/models/Inkling-Small-mxfp4
```

Minkling can use that shard directly, but it is roughly 4.46 GB. The repository
includes a script that packs its matrix weights into MXFP4 and produces a
roughly 1.19 GB shard instead:

```sh
cd slop
just sync
just quantize-mtp
cd ..
```

This creates `slop/models/Inkling-Small-mxfp4-mtp4`. Its model files are
symlinks to the downloaded MXFP4 checkpoint, while its `mtp.safetensors` is the
locally packed shard. Use that checkpoint with `--speculate N`:

```sh
target/release/inklingrs generate slop/models/Inkling-Small-mxfp4-mtp4 \
  --prompt "The lighthouse keeper" \
  --max-tokens 64 \
  --numerics production \
  --speculate 2
```

Speculation is currently available for `generate`; the HTTP server decodes
without the MTP heads. See the [weights documentation](slop/README.md#weights)
for how the two head formats are validated against each other.
