"""Dump `embed_norm`'s trained weight, so the normalisation at the front of the
model can be tested without the checkpoint.

`layer_activations.safetensors` already carries `embed_out` and
`embed_norm_out`, which is the oracle for this step; what it does not carry is
the weight between them, and a test that recovered it by dividing one recorded
tensor by the other would agree with any implementation. The weight is `[4096]`,
so committing it costs 16 KiB and makes the step hermetic.

The lookup that produces `embed_out` cannot follow it here. `embed_tokens` is
`[201024, 4096]` of MXFP4 — 3.3 GB decoded — so it stays in the gated tier that
reads a real checkpoint.

Read straight out of the shard that owns it rather than through `mlx_vlm.load`:
the fixture needs one tensor."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from inkling_ref import checkpoint_tensor

EMBED_NORM = "language_model.model.embed_norm.weight"


def collect(model_path):
    config = json.loads((model_path / "config.json").read_text())["text_config"]
    if not config["use_embed_norm"]:
        raise SystemExit(f"{model_path} sets use_embed_norm false; there is no weight")
    return {
        "embed_norm.weight": checkpoint_tensor(model_path, EMBED_NORM).astype(
            mx.float32
        ),
        "rms_norm_eps": mx.array([config["rms_norm_eps"]], dtype=mx.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    tensors = collect(Path(args.model))
    mx.eval(*tensors.values())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "embed.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    weight = tensors["embed_norm.weight"]
    print(
        f"embed_norm.weight {list(weight.shape)} in {weight.min():.4f}..{weight.max():.4f}"
    )
    print(
        f"{len(tensors)} tensors, {bundle.stat().st_size / (1 << 10):.2f} KiB -> {bundle}"
    )


if __name__ == "__main__":
    main()
