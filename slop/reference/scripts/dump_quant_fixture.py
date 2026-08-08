"""Dump packed MXFP4 slices next to MLX's own dequantisation of them, as the
oracle the Rust decoder is tested against.

Every weight in the checkpoint but the norms arrives as a `U32` block of 4-bit
codes plus a `U8` block scale, so a decoder that gets the nibble order, the sign
bit or the scale's extremes wrong still produces plausible-looking numbers and
wrong ones. Only MLX's answer settles it.

The three real slices cover the shapes the decoder meets — a 3-D routed-expert
stack, a 2-D dense projection, and the vocabulary padding where the scale bytes
go to zero. They do not settle the format on their own: real scale bytes span
0x00 and 0x6e..0x82, and the padding rows that carry 0x00 carry only zero codes
with it, so the checkpoint never shows what a zero scale does to a nonzero code.
The synthetic grid closes that gap by putting all 16 codes under all 256 scale
bytes.

The recorded answer is the CPU backend's. Metal's differs, for the two smallest
scale bytes only: it builds the block scale by shifting the byte into an
exponent field, which turns 0x00 into zero rather than 2^-127, and it flushes
subnormal products to zero. `survey` establishes that no such input exists in
the checkpoint, so the backends agree on everything the engine will ever decode
and the exact answer is the one worth pinning.

Slices are read straight out of one shard rather than through `mlx_vlm.load`,
because the fixture needs a few hundred KiB."""

import argparse
import json
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import index_of, load_shard

GROUP_SIZE = 32
BITS = 4
MODE = "mxfp4"

CODES_PER_WORD = 32 // BITS
WORDS_PER_GROUP = GROUP_SIZE // CODES_PER_WORD
CODE_COUNT = 1 << BITS

# Above this byte, 2^(byte-127) is normal in f32 and both backends agree.
SUBNORMAL_SCALE_BYTES = 0x02

ROUTED_EXPERT = "language_model.model.layers.3.mlp.switch_mlp.gate_proj"
DENSE_FFN = "language_model.model.layers.0.mlp.gate_proj"
VOCAB = "language_model.lm_head"
PADDING_CONTEXT = 32


def slice_repr(index):
    return "[" + ", ".join(f"{s.start}:{s.stop}" for s in index) + "]"


def codes_in(packed):
    shifts = BITS * np.arange(CODES_PER_WORD, dtype=np.uint32)
    return (np.asarray(packed).reshape(-1, 1) >> shifts) & (CODE_COUNT - 1)


def dequantize(packed, scales, stream):
    return mx.dequantize(
        packed,
        scales,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
        dtype=mx.float32,
        stream=stream,
    )


def code_grid():
    """One row of 256 consecutive groups, group `s` holding all 16 codes twice
    under scale byte `s`, so decoded value `s * 32 + c` is code `c % 16` scaled
    by `s`.

    This is the only part of the fixture that pins the element table, the sign
    bit and the meaning of the extreme scale bytes, because the checkpoint's own
    scale bytes never leave a narrow band around 0x7f. Laying the groups out
    along one row also makes every group boundary a change of scale."""
    codes = np.arange(GROUP_SIZE, dtype=np.uint32) % CODE_COUNT
    words = np.zeros(WORDS_PER_GROUP, dtype=np.uint32)
    for i, code in enumerate(codes):
        words[i // CODES_PER_WORD] |= code << np.uint32(BITS * (i % CODES_PER_WORD))
    packed = np.tile(words, 256).reshape(1, -1)
    scales = np.arange(256, dtype=np.uint8).reshape(1, -1)
    return mx.array(packed), mx.array(scales)


def real_slices(model_path):
    """`lm_head`'s rows past `unpadded_vocab_size` are padding: all-zero codes
    under all-zero scales. Straddling that row is how the fixture gets a scale
    byte outside the narrow band the trained weights use."""
    config = json.loads((model_path / "config.json").read_text())
    vocab = config["text_config"]["unpadded_vocab_size"]
    return (
        ("routed_expert", ROUTED_EXPERT, (slice(0, 2), slice(0, 32))),
        ("dense_ffn", DENSE_FFN, (slice(0, 64),)),
        (
            "vocab_padding",
            VOCAB,
            (slice(vocab - PADDING_CONTEXT, vocab + PADDING_CONTEXT),),
        ),
    )


def quantized_tensors(model_path):
    """Every `weight`/`scales` pair in the checkpoint, grouped by shard so a
    sweep evaluates one shard's worth of arrays at a time."""
    weight_map = index_of(model_path)["weight_map"]
    pairs = sorted(
        (weight_map[name], name[: -len(".scales")])
        for name in weight_map
        if name.endswith(".scales")
    )
    return groupby(pairs, key=itemgetter(0))


def load_pair(shard, tensor, index=...):
    return shard[tensor + ".weight"][index], shard[tensor + ".scales"][index]


def check_codes_covered(name, packed):
    """A slice that happens to miss a code silently stops testing it."""
    seen = np.unique(codes_in(packed))
    if len(seen) != CODE_COUNT:
        raise SystemExit(f"{name}: exercises only codes {seen.tolist()}")


def check_group_boundary(name, scales):
    """The decoder must switch scales every 32 values. A slice whose groups all
    share one scale byte cannot tell a correct implementation from one that
    reads the first scale and holds it."""
    rows = np.asarray(scales).reshape(-1, np.asarray(scales).shape[-1])
    if not (rows[:, :-1] != rows[:, 1:]).any():
        raise SystemExit(f"{name}: no two adjacent groups differ in scale")


def metal_divergence(name, packed, scales, values):
    """Where Metal's answer differs from the recorded CPU one. Divergence is
    only understood under the two smallest scale bytes, and only towards zero;
    anything else means the backends have moved apart somewhere that matters."""
    cpu = np.asarray(values)
    metal = np.asarray(dequantize(packed, scales, mx.gpu))
    per_value = np.repeat(np.asarray(scales), GROUP_SIZE, axis=-1)
    differs = cpu.view(np.uint32) != metal.view(np.uint32)
    explained = (per_value < SUBNORMAL_SCALE_BYTES) & (metal == 0)
    if (differs & ~explained).any():
        raise SystemExit(
            f"{name}: unexplained cpu/metal divergence at {np.argwhere(differs & ~explained)}"
        )
    return int(differs.sum())


def survey(model_path):
    """Every scale byte the checkpoint actually contains, and whether a zero
    scale ever meets a nonzero code.

    This is what licenses recording the CPU backend's answer: the two backends
    only part company at scale bytes 0x00 and 0x01, and the survey shows the
    checkpoint reaches neither in a way that decodes to anything but zero. Read
    on every regeneration rather than asserted once, because a re-quantised
    checkpoint could invalidate it silently."""
    seen, disputed, count = set(), 0, 0
    for shard_name, pairs in quantized_tensors(model_path):
        shard = load_shard(model_path, shard_name)
        for _, tensor in pairs:
            count += 1
            packed, scales = load_pair(shard, tensor)
            mx.eval(scales)
            scales = np.asarray(scales)
            seen.update(np.unique(scales).tolist())
            subnormal = scales < SUBNORMAL_SCALE_BYTES
            if not subnormal.any():
                continue
            mx.eval(packed)
            groups = np.asarray(packed).reshape(*scales.shape, WORDS_PER_GROUP)
            disputed += int((subnormal & (groups != 0).any(axis=-1)).sum())

    return {
        "tensors": count,
        "scale_bytes": sorted(int(b) for b in seen),
        "groups_where_the_backends_would_disagree": disputed,
    }


def collect(model_path):
    weight_map = index_of(model_path)["weight_map"]
    entries = [
        (name, tensor, slice_repr(index))
        + load_pair(
            load_shard(model_path, weight_map[tensor + ".weight"]), tensor, index
        )
        for name, tensor, index in real_slices(model_path)
    ]
    entries.append(("code_grid", None, None) + code_grid())

    tensors, manifest = {}, []
    for name, tensor, sliced, packed, scales in entries:
        values = dequantize(packed, scales, mx.cpu)
        mx.eval(packed, scales, values)
        check_codes_covered(name, packed)
        check_group_boundary(name, scales)
        tensors[f"{name}.weight"] = packed
        tensors[f"{name}.scales"] = scales
        tensors[f"{name}.dequantized"] = values
        manifest.append(
            {
                "name": name,
                "tensor": tensor,
                "slice": sliced,
                "shape": list(packed.shape),
                "dequantized_shape": list(values.shape),
                "scale_byte_range": [int(scales.min()), int(scales.max())],
                "values_metal_would_flush_to_zero": metal_divergence(
                    name, packed, scales, values
                ),
            }
        )
    return tensors, manifest


def build_manifest(model_path, slices, checkpoint_survey):
    return {
        "quantization": {"group_size": GROUP_SIZE, "bits": BITS, "mode": MODE},
        "checkpoint": {
            "path": str(model_path),
            "total_size": index_of(model_path)["metadata"]["total_size"],
            "quantized": checkpoint_survey,
        },
        "dequantized_by": "mx.dequantize on the cpu stream",
        "slices": slices,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    model_path = Path(args.model)
    tensors, slices = collect(model_path)
    manifest = build_manifest(model_path, slices, survey(model_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "mxfp4_dequant.safetensors"
    mx.save_safetensors(str(bundle), tensors)
    with open(out_dir / "mxfp4_dequant.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    for entry in slices:
        source = (entry["tensor"] or "synthetic") + (entry["slice"] or "")
        lo, hi = entry["scale_byte_range"]
        print(
            f"{entry['name']:<14}  {str(entry['shape']):<16}  scale bytes {lo:#04x}..{hi:#04x}  {source}"
        )
    survey_result = manifest["checkpoint"]["quantized"]
    print(
        f"\ncheckpoint: {survey_result['tensors']} quantized tensors, scale bytes "
        f"{survey_result['scale_bytes'][0]:#04x}..{survey_result['scale_bytes'][-1]:#04x}, "
        f"{survey_result['groups_where_the_backends_would_disagree']} groups where "
        f"cpu and metal would disagree"
    )
    print(
        f"{len(tensors)} tensors, {bundle.stat().st_size / (1 << 20):.2f} MiB -> {bundle}"
    )


if __name__ == "__main__":
    main()
