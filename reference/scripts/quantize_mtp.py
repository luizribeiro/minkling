#!/usr/bin/env python3
"""Quantise the multi-token prediction heads to MXFP4, beside a stack that is
already MXFP4.

Every quantiser left `model.mtp.*` in bfloat16 -- the 8-bit quant's
`mtp.safetensors` is the BF16 original's 160 tensors byte for byte -- so the
heads are the one part of this checkpoint nobody has ever packed, and two thirds
of a chain's device time is one kernel reading them. This is the forward
direction of `inkling_core::quant`, which the engine has only ever read: MLX's
own `mx.quantize(mode="mxfp4")`, which is what produced the stack these heads sit
beside, so the codes this writes are the codes that dequantiser is pinned to
rather than a second implementation of the same table.

**Eight tensors a head and not twenty.** The rest are norms, convolution kernels,
the relative-position table and a scalar -- 260 KB a head against 532 MiB, none
of it read by a matmul, and `quantize.py` leaves the stack's own counterparts
alone for the same reason.

**The fused `w13_dn` is written as two tensors.** A head's SwiGLU keeps its gate
and its up projection interleaved row by row in one `[2 * dense, hidden]` tensor,
and the two are read separately -- which for bfloat16 is a row stride the kernel
takes and for a packed tensor would be a stride through codes, scales and the
group boundaries between them. Splitting here costs nothing and changes nothing:
a group spans 32 values of a row, so which rows are in a tensor is not something
quantisation can see.

The bfloat16 original is not touched. It is the oracle the quantised heads' own
guesses are held against, and it is what `--speculate` reads for as long as it
sits beside the stack.
"""

import argparse
import json
import struct
import time
from pathlib import Path

import mlx.core as mx

GIB = 1 << 30

# What MXFP4 is, as the checkpoint's `quantization` block spells it and as
# `inkling_core::quant` decodes it: E2M1 codes in groups of 32 under an E8M0
# scale byte apiece.
BITS = 4
GROUP_SIZE = 32
MODE = "mxfp4"

# The eight matmul weights of a head, by the suffix each is stored under. Every
# one is `[out, in]` with `in` a multiple of the group size, which is what
# `mx.quantize` needs and what makes this list a rule rather than a table.
QUANTIZED = (
    "input_proj.weight",
    "transformer_block.attn.wq_du.weight",
    "transformer_block.attn.wk_dv.weight",
    "transformer_block.attn.wv_dv.weight",
    "transformer_block.attn.wr_du.weight",
    "transformer_block.attn.wo_ud.weight",
    "transformer_block.mlp.w13_dn.weight",
    "transformer_block.mlp.w2_md.weight",
)

# The one of those that is two weights. Its even rows are the SwiGLU's gate and
# its odd ones the up projection -- see `inkling_core::mtp`'s `Mlp::split`.
FUSED = "transformer_block.mlp.w13_dn.weight"
HALVES = ("gate", "up")


def head_of(name):
    """The head a tensor belongs to, or `None` for one that names no head."""
    parts = name.split(".")
    if len(parts) < 4 or parts[:3] != ["model", "mtp", "layers"]:
        return None
    return int(parts[3])


def suffix_of(name):
    return name.split(".", 4)[4]


def packed_names(name):
    """What one source tensor is written as: a packed pair, or two of them."""
    stem = name[: -len(".weight")]
    if suffix_of(name) == FUSED:
        return [f"{stem}.{half}" for half in HALVES]
    return [stem]


def halves(value, name):
    """The tensors one source tensor quantises to, in `packed_names` order."""
    if suffix_of(name) == FUSED:
        return [value[first :: len(HALVES)] for first in range(len(HALVES))]
    return [value]


def error(original, packed, scales):
    """How far the codes are from the weight they stand for, relative to the
    weight's own size. What acceptance is at risk from is this, and a number per
    tensor is what says whether one of them took it far worse than the rest.

    Undefined for a weight that is all zeros, which is a weight nothing here
    has: the divisor is the weight's own norm."""
    back = mx.dequantize(packed, scales, group_size=GROUP_SIZE, bits=BITS, mode=MODE)
    original = original.astype(mx.float32)
    return float(
        mx.sqrt(mx.sum(mx.square(back - original)) / mx.sum(mx.square(original)))
    )


def read_header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def quantize(src, dst, opts):
    weights = mx.load(str(src))
    out = {}
    worst = ("", 0.0)
    quantized = 0
    for name in sorted(weights):
        value = weights.pop(name)
        head = head_of(name)
        if head is None or suffix_of(name) not in QUANTIZED:
            out[name] = value
            continue

        for packed_name, half in zip(packed_names(name), halves(value, name)):
            codes, scales = mx.quantize(
                half, group_size=GROUP_SIZE, bits=BITS, mode=MODE
            )
            mx.eval(codes, scales)
            out[f"{packed_name}.weight"] = codes
            out[f"{packed_name}.scales"] = scales
            if opts.check:
                drift = error(half, codes, scales)
                if drift > worst[1]:
                    worst = (packed_name, drift)
                print(f"  {packed_name:<58} {drift:.5f}", flush=True)
            quantized += 1
        del value

    # Staged and renamed, so that a run interrupted part way through does not
    # leave something a loader would map as a checkpoint's heads.
    tmp = dst.parent / f".partial-{dst.name}"
    mx.save_safetensors(str(tmp), out)
    tmp.rename(dst)
    return quantized, worst


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="the bfloat16 mtp.safetensors")
    p.add_argument("dst", type=Path, help="where the packed one goes")
    p.add_argument(
        "--check",
        action="store_true",
        help="dequantise each tensor back and print how far the codes fell",
    )
    return p.parse_args()


def main():
    opts = parse_args()
    # On the CPU, for the reason `quantize.py` gives: the source is mapped rather
    # than read, and a Metal kernel over a tensor arriving off disk trips the GPU
    # watchdog long before the tensor is whole.
    mx.set_default_device(mx.cpu)

    at = time.monotonic()
    quantized, worst = quantize(opts.src, opts.dst, opts)
    was = opts.src.stat().st_size
    now = opts.dst.stat().st_size
    print(
        f"{quantized} tensors packed in {time.monotonic() - at:.0f}s: "
        f"{was / GIB:.2f} GiB to {now / GIB:.2f} GiB, {was / now:.2f}x"
    )
    if opts.check:
        print(f"worst tensor {worst[0]} at {worst[1]:.5f}")

    header = read_header(opts.dst)
    dtypes = sorted({info["dtype"] for info in header.values()})
    print(f"{len(header)} tensors, dtypes {dtypes}")


if __name__ == "__main__":
    main()
