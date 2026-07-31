"""Dump synthetic tensors through mlx-vlm's own RMSNorm and dense MLP, as the
oracle the Rust CPU ops are tested against.

Inkling's dense feed-forward weights are `[16384, 4096]` — 268 MB in float32 —
so unlike the activation and MXFP4 fixtures this one cannot be cut out of the
checkpoint. It uses deterministic micro-tensors instead: a fixed seed, and
dimensions in the tens. What it pins is not Inkling's trained numbers but
mlx-vlm's arithmetic, which is the part a port gets wrong.

Nothing here reimplements an op. `nn.RMSNorm` is constructed the way
`InklingModel` constructs it, and `InklingDenseMLP` is loaded with weights and
called, so the recorded outputs come from the classes the model runs — down to
the trailing `global_scale` multiply, which lives in `InklingDenseMLP` rather
than in the shared `SwiGLUMLP` and so is easy to drop.

Everything is stored and computed in float32, so a comparison measures the
port's arithmetic rather than bf16 rounding."""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from inkling_ref import f32, gamma, projection
from mlx.utils import tree_flatten
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import InklingDenseMLP

SEED = 20260730

DIM = 40
HIDDEN_DIM = 96
MLP_INPUT_SHAPE = (2, 3, DIM)

# Anything but 1.0, or `InklingDenseMLP`'s trailing multiply is untestable.
GLOBAL_SCALE = 1.7

# Row magnitude for the overflow case. MLX accumulates the sum of squares in
# float32, so a 32-wide row flushes to zero above ~3e18; this stays a decade
# under that, where the oracle is still an oracle.
LARGE = 1e18

NORM_SHAPES = {
    "norm_wide": (4, 64),
    # A last axis that is not a multiple of 8, so a future SIMD path cannot
    # quietly assume one.
    "norm_odd": (3, 37),
    "norm_batched": (2, 3, 24),
    "norm_zero_row": (3, 32),
    "norm_large": (2, 32),
}


def norm_input(rng, name, shape):
    x = rng.standard_normal(shape)
    if name == "norm_zero_row":
        # eps exists for exactly this row, and it is where a port divides by zero.
        x[1] = 0.0
    if name == "norm_large":
        x[0] *= LARGE
    return x


def rms_norm_case(rng, name, shape, eps):
    x = f32(norm_input(rng, name, shape))
    norm = nn.RMSNorm(shape[-1], eps=eps)
    norm.load_weights([("weight", gamma(rng, shape[-1]))])
    return {
        f"{name}.input": x,
        f"{name}.weight": norm.weight,
        f"{name}.output": norm(x),
    }


def dense_mlp_case(rng, config):
    weights = {
        "gate_proj.weight": projection(rng, HIDDEN_DIM, DIM),
        "up_proj.weight": projection(rng, HIDDEN_DIM, DIM),
        "down_proj.weight": projection(rng, DIM, HIDDEN_DIM),
        "global_scale": f32([GLOBAL_SCALE]),
    }
    mlp = InklingDenseMLP(config)
    mlp.load_weights(list(weights.items()))

    x = f32(rng.standard_normal(MLP_INPUT_SHAPE))
    tensors = {f"mlp.{name}": value for name, value in tree_flatten(mlp.parameters())}
    tensors["mlp.input"] = x
    tensors["mlp.output"] = mlp(x)
    return tensors


def collect():
    rng = np.random.default_rng(SEED)
    config = TextConfig(hidden_size=DIM, intermediate_size=HIDDEN_DIM)

    tensors = {"rms_norm_eps": f32([config.rms_norm_eps])}
    for name, shape in NORM_SHAPES.items():
        tensors.update(rms_norm_case(rng, name, shape, config.rms_norm_eps))
    tensors.update(dense_mlp_case(rng, config))

    mx.eval(list(tensors.values()))
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    tensors = collect()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "ops.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    for name, value in tensors.items():
        peak = float(mx.abs(value).max())
        print(f"{name:<26}  {str(list(value.shape)):<14}  max |x| {peak:.4g}")
    print(
        f"\n{len(tensors)} tensors, {bundle.stat().st_size / 1024:.1f} KiB -> {bundle}"
    )


if __name__ == "__main__":
    main()
