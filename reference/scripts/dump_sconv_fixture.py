"""Dump what the Rust short convolution is tested against: the checkpoint's own
`sconv` weights for the layers `dump_activations.py` captured, and a set of
synthetic float32 cases run through mlx-vlm's `InklingShortConvolution`.

Two kinds of case, because the two things worth pinning can be held to very
different precisions.

The real weights pair with activations that are already committed —
`layer0.k_proj_out` is what `layer0.k_sconv_out` was computed from — so the op
can be checked end to end against trained numbers. That check is coarse: the
model runs in bfloat16, and `InklingShortConvolution` casts its padded input to
the weight's dtype, so the recorded output has been rounded to bfloat16 once
after the convolution and again after the residual add. What it settles is the
weight layout and the trained kernel, not the last bits.

The synthetic cases are float32 throughout and settle the arithmetic: which end
of the kernel meets the current timestep, that the residual is added, that a
mask zeroes the convolution's input but not the residual, and that a cache
carries the same answer across a split. Dimensions in the single digits, so the
tensors stay readable in a hex dump.

Nothing here reimplements the op. Every synthetic output comes from calling the
reference module, with the reference `ArraysCache` the model itself allocates."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import checkpoint_tensor, index_of
from mlx_lm.models.cache import ArraysCache
from mlx_vlm.models.inkling.language import InklingShortConvolution

# The layers dump_activations.py captured, and the four conv slots each one
# holds. The index is the module's `conv_idx`: its slot in the layer's shared
# four-entry conv cache.
CAPTURED_LAYERS = (0, 2)
CONVS = (
    ("k_sconv", "self_attn.k_sconv"),
    ("v_sconv", "self_attn.v_sconv"),
    ("attn_sconv", "attn_sconv"),
    ("mlp_sconv", "mlp_sconv"),
)

SEED = 20260731

BATCH = 2
LENGTH = 6
# Not a multiple of four, so a future SIMD path cannot quietly assume one.
CHANNELS = 5

# Shorter than the K-1 timesteps a cache holds, so the state left behind has to
# mix what was already there with what just arrived.
SHORT_LENGTH = 2


def sconv(kernel_size, weight):
    conv = InklingShortConvolution(CHANNELS, kernel_size, conv_idx=0)
    conv.conv.load_weights([("weight", weight)])
    return conv


def taps(rng, kernel_size):
    """`[channels, kernel_size, 1]`, the layout `nn.Conv1d` expects and the
    layout the checkpoint stores.

    Drawn so no two taps of a channel are close in magnitude: a kernel read in
    reversed time order still produces plausible numbers, and only an asymmetric
    kernel makes it produce different ones. The ramp is about the four-fold one
    the trained kernels show, which also keeps the convolution and the residual
    within a factor of a few of each other — a residual lost in the noise would
    be a residual nothing tests."""
    magnitude = 0.4 * 1.6 ** np.arange(kernel_size)
    signs = rng.choice([-1.0, 1.0], size=(CHANNELS, kernel_size))
    spread = 1.0 + 0.25 * rng.standard_normal((CHANNELS, kernel_size))
    return mx.array((signs * spread * magnitude)[..., None], dtype=mx.float32)


def conv_mask():
    """Row 0 drops timesteps in the middle, so the zeroing has to propagate
    through the following windows. Row 1 drops everything, which leaves the
    residual as the whole answer and is the case that separates a mask on the
    convolution's input from a mask on the output.

    Saved as float32 like every other fixture tensor; the module wants bool."""
    mask = np.ones((BATCH, LENGTH), dtype=np.float32)
    mask[0, 2:4] = 0.0
    mask[1, :] = 0.0
    return mx.array(mask)


def streamed(conv, x):
    """One timestep at a time through a fresh cache, as decoding runs it."""
    cache = ArraysCache(4)
    steps = [conv(x[:, t : t + 1, :], cache=cache) for t in range(x.shape[1])]
    return mx.concatenate(steps, axis=1), cache[conv.conv_idx]


def synthetic_cases(rng, kernel_size):
    weight = taps(rng, kernel_size)
    conv = sconv(kernel_size, weight)

    def normal(*shape):
        return mx.array(rng.standard_normal(shape), dtype=mx.float32)

    x = normal(BATCH, LENGTH, CHANNELS)
    cases = {"weight": weight, "input": x, "whole": conv(x)}

    cases["streamed"], cases["streamed_state"] = streamed(conv, x)

    primed = ArraysCache(4)
    primed[conv.conv_idx] = normal(BATCH, kernel_size - 1, CHANNELS)
    cases["primed_state"] = primed[conv.conv_idx]
    cases["short_input"] = normal(BATCH, SHORT_LENGTH, CHANNELS)
    cases["primed_output"] = conv(cases["short_input"], cache=primed)
    cases["primed_final_state"] = primed[conv.conv_idx]

    masked = ArraysCache(4)
    cases["mask"] = conv_mask()
    cases["masked_output"] = conv(x, cache=masked, mask=cases["mask"] != 0)
    cases["masked_state"] = masked[conv.conv_idx]

    return {f"synthetic.{name}": value for name, value in cases.items()}


def real_weights(model_path):
    """The trained kernels, exactly as the checkpoint stores them.

    `Model.sanitize` transposes `model.llm.layers.N.attn.k_sconv.weight` through
    `(0, 2, 1)` on the way to `nn.Conv1d`, so the two published layouts are
    `[channels, 1, kernel]` and `[channels, kernel, 1]`. Both are one contiguous
    run of `kernel` taps per channel; the transpose only moves a length-1 axis
    and never touches an element's position. Recorded here in whichever of the
    two this checkpoint uses, which is what Rust reads."""
    return {
        f"layer{layer}.{name}.weight": checkpoint_tensor(
            model_path, f"language_model.model.layers.{layer}.{path}.conv.weight"
        ).astype(mx.float32)
        for layer in CAPTURED_LAYERS
        for name, path in CONVS
    }


def check_reversal_is_visible(tensors):
    """A synthetic kernel that happened to be near-symmetric would let a
    reversed-time-order convolution reproduce the fixture."""
    weight = np.asarray(tensors["synthetic.weight"]).reshape(CHANNELS, -1)
    if np.allclose(weight, weight[:, ::-1], rtol=0.5):
        raise SystemExit(f"synthetic kernel is near-symmetric:\n{weight}")


def check_streaming_agrees(tensors):
    """The property the engine's decode loop rests on. If the reference itself
    did not have it there would be nothing to port."""
    whole = np.asarray(tensors["synthetic.whole"])
    step = np.asarray(tensors["synthetic.streamed"])
    gap = np.abs(whole - step).max() / np.abs(whole).max()
    if gap > 1e-6:
        raise SystemExit(
            f"reference streaming disagrees with whole-sequence: {gap:.3e}"
        )
    return gap


def collect(model_path, kernel_size):
    rng = np.random.default_rng(SEED)
    tensors = {"kernel_size": mx.array([kernel_size], dtype=mx.float32)}
    tensors.update(synthetic_cases(rng, kernel_size))
    tensors.update(real_weights(model_path))
    mx.eval(list(tensors.values()))
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    model_path = Path(args.model)
    config = json.loads((model_path / "config.json").read_text())
    tensors = collect(model_path, config["text_config"]["sconv_kernel_size"])

    check_reversal_is_visible(tensors)
    gap = check_streaming_agrees(tensors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "sconv.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    for name, value in tensors.items():
        print(
            f"{name:<28}  {str(list(value.shape)):<14}  max |x| {float(mx.abs(value).max()):.4g}"
        )
    print(f"\nreference streaming vs whole-sequence: {gap:.3e}")
    print(f"checkpoint total_size {index_of(model_path)['metadata']['total_size']}")
    print(f"{len(tensors)} tensors, {bundle.stat().st_size / 1024:.1f} KiB -> {bundle}")


if __name__ == "__main__":
    main()
