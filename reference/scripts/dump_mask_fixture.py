"""Dump what the Rust banded relative-position mask is tested against: the
checkpoint's own `rel_proj` for the layers `dump_activations.py` captured, and a
set of synthetic float32 cases run through mlx-vlm's `banded_additive_mask`.

The trained projections pair with activations that are already committed —
`layer0.r_proj_out` is what `layer0.mask` was computed from — so the op can be
checked end to end against trained numbers. That check is coarse: the model runs
in bfloat16, so the recorded mask was rounded once after the kernel returned.

The synthetic cases exist because the committed masks are eight tokens wide at
offset zero, where nothing is far enough back to be capped by the 512-token
window or to fall outside the 1024-token band. Two of the kernel's four branches
never run there. A longer capture is not an option — `[1, 32, 600, 600]` in
float32 is 46 MB — so these are short query windows against a long key span,
placed by `q_offset` so the branch boundaries land inside the span. Eight
queries over 1200 keys at two heads is 77 KB.

Nothing here reimplements the op: every mask comes from calling
`banded_additive_mask`. `branches` classifies positions by backward distance,
which is how the coverage checks know a case reaches the branch it was written
for, and it decides no value."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import CAPTURED_LAYERS, checkpoint_tensor, index_of
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import banded_additive_mask

# The prompt length the captured layers' masks were recorded over — the key
# span of the masks in layer_activations.safetensors, which these synthetic
# cases surround.
ACTIVATION_SEQ_LEN = 8

SEED = 20260801

# The checkpoint's own d_rel. Two heads is the fewest that can tell the
# head-minor `[B, LQ, H, d_rel]` layout of `rel` from a head-major one.
D_REL = 16
HEADS = 2

# Each case's shape and the configuration InklingAttention would have passed.
# `keys` is the whole KV span; `q_offset` places the query window inside it,
# which is what moves the branch boundaries around.
CASES = {
    # A sliding layer's own configuration, with the window edge strictly inside
    # the key span so a single row holds causally masked, window-capped and
    # in-band entries at once.
    "sliding_window": dict(
        batch=1, queries=8, keys=1200, q_offset=800, sliding=512, rel_extent=512
    ),
    # A global layer's. Nothing is window-capped, and the span reaches past the
    # 1024-token band — the only configuration in which a position is in
    # context, outside the band, and therefore exactly zero.
    "global_band": dict(
        batch=1, queries=8, keys=1200, q_offset=1100, sliding=0, rel_extent=1024
    ),
    # One query against a long cache: the shape decode runs, and the shape in
    # which a `q_offset` that is dropped is visible. During prefill it is zero
    # and dropping it changes nothing.
    "decode": dict(
        batch=1, queries=1, keys=1200, q_offset=1199, sliding=512, rel_extent=512
    ),
    # Prefill: a whole prompt against itself from position zero, batched, which
    # is the only case here that exercises the batch stride of `rel`.
    "prefill": dict(
        batch=2, queries=64, keys=64, q_offset=0, sliding=512, rel_extent=512
    ),
    # A window narrower than the band. No Inkling layer is configured this way:
    # sliding layers set both to 512 and global layers set the window to zero,
    # so the window cap and the band overlap nowhere in the real model. This is
    # the only arrangement that can show which of the two the kernel applies
    # first.
    "narrow_window": dict(
        batch=1, queries=8, keys=1200, q_offset=1100, sliding=256, rel_extent=1024
    ),
}


def distances(spec):
    """`(i + q_offset) - j` for every query `i` and key `j` — the backward
    distance the kernel branches on."""
    qp = np.arange(spec["queries"])[:, None] + spec["q_offset"]
    return qp - np.arange(spec["keys"])[None, :]


def branches(spec):
    """Which of the kernel's four branches each position falls in: causal,
    window cap, learned bias, or in-context outside the band.

    Written as the kernel's if-chain in reverse, so the earlier a branch is in
    the chain the later it is applied here and the higher it sits."""
    dist = distances(spec)
    branch = np.full(dist.shape, 4)
    branch[dist < spec["rel_extent"]] = 3
    if spec["sliding"] > 0:
        branch[dist >= spec["sliding"]] = 2
    branch[dist < 0] = 1
    return branch


def config_tensor(spec):
    """`[q_offset, keys, sliding, rel_extent]`, the four scalars the shapes do
    not carry."""
    return mx.array(
        [spec[field] for field in ("q_offset", "keys", "sliding", "rel_extent")],
        dtype=mx.float32,
    )


def projection(rng, rel_extent):
    """`[d_rel, rel_extent]`, scaled by fan-in so a bias lands in the same
    handful-of-units range the trained masks do.

    Drawn independently per column, so the bias varies sharply with distance and
    an off-by-one in the distance index cannot pass for the right answer."""
    return mx.array(
        rng.standard_normal((D_REL, rel_extent)) / np.sqrt(D_REL), dtype=mx.float32
    )


def synthetic_cases(rng):
    extents = sorted({spec["rel_extent"] for spec in CASES.values()})
    projections = {extent: projection(rng, extent) for extent in extents}
    tensors = {f"proj{extent}": proj for extent, proj in projections.items()}

    for name, spec in CASES.items():
        rel = mx.array(
            rng.standard_normal((spec["batch"], spec["queries"], HEADS, D_REL)),
            dtype=mx.float32,
        )
        tensors[f"{name}.rel"] = rel
        tensors[f"{name}.config"] = config_tensor(spec)
        tensors[f"{name}.mask"] = banded_additive_mask(
            rel,
            projections[spec["rel_extent"]],
            spec["q_offset"],
            spec["keys"],
            spec["sliding"],
            spec["rel_extent"],
        )
    return tensors


def trained_projections(model_path):
    """`rel_proj` as the checkpoint stores it, `[d_rel, rel_extent]`, alongside
    the configuration `InklingAttention.__init__` derives for that layer."""
    config = TextConfig.from_dict(
        json.loads((model_path / "config.json").read_text())["text_config"]
    )
    tensors = {}
    for layer in CAPTURED_LAYERS:
        sliding = config.layer_is_sliding(layer)
        spec = dict(
            q_offset=0,
            keys=ACTIVATION_SEQ_LEN,
            sliding=config.sliding_window_size if sliding else 0,
            rel_extent=config.sliding_window_size if sliding else config.rel_extent,
        )
        tensors[f"layer{layer}.config"] = config_tensor(spec)
        tensors[f"layer{layer}.rel_proj"] = checkpoint_tensor(
            model_path, f"language_model.model.layers.{layer}.self_attn.rel_proj"
        ).astype(mx.float32)
    return tensors


def check_branch_coverage():
    """The synthetic cases exist to reach the branches the trained masks cannot.
    A case whose offsets were retuned until it no longer did would leave a test
    that passes by never running the code it names."""
    covered = {}
    for name, spec in CASES.items():
        reached = set(np.unique(branches(spec)).tolist())
        covered[name] = sorted(reached)
        if spec["rel_extent"] == spec["sliding"] and 4 in reached:
            raise SystemExit(f"{name}: a window as wide as the band cannot reach 4")
    missing = {1, 2, 3, 4} - {b for reached in covered.values() for b in reached}
    if missing:
        raise SystemExit(f"no case reaches branch(es) {sorted(missing)}")
    return covered


def check_bias_is_never_zero(tensors):
    """Branch 4 is the value zero, so a learned bias that happened to be zero
    would make the two indistinguishable."""
    for name, spec in CASES.items():
        mask = np.asarray(tensors[f"{name}.mask"])
        in_band = np.broadcast_to(branches(spec) == 3, mask.shape)
        if (mask[in_band] == 0.0).any():
            raise SystemExit(f"{name}: a learned bias is exactly zero")


def check_masked_constant(tensors):
    """Float32 puts `-1e30` in the output exactly, so the synthetic cases pin
    the constant itself. The committed bfloat16 masks cannot: `-1e30` does not
    survive the round trip, and lands at `-1.0002555517425873e30`."""
    values = set()
    for name in CASES:
        mask = np.asarray(tensors[f"{name}.mask"])
        values.update(np.unique(mask[mask <= -1e29]))
    if values != {np.float32(-1e30)}:
        raise SystemExit(f"masked entries are not a single -1e30: {sorted(values)}")


def collect(model_path):
    rng = np.random.default_rng(SEED)
    tensors = synthetic_cases(rng)
    tensors.update(trained_projections(model_path))
    mx.eval(list(tensors.values()))
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    model_path = Path(args.model)
    covered = check_branch_coverage()
    tensors = collect(model_path)
    check_bias_is_never_zero(tensors)
    check_masked_constant(tensors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "mask.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    for name, value in sorted(tensors.items()):
        peak = float(mx.abs(mx.where(value > -1e29, value, 0)).max())
        print(f"{name:<26}  {str(list(value.shape)):<18}  max |x| {peak:.4g}")
    print("\nbranches reached:")
    for name, reached in covered.items():
        print(f"  {name:<16}  {reached}")
    print(f"checkpoint total_size {index_of(model_path)['metadata']['total_size']}")
    print(f"{len(tensors)} tensors, {bundle.stat().st_size / 1024:.1f} KiB -> {bundle}")


if __name__ == "__main__":
    main()
