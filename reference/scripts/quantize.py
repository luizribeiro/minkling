#!/usr/bin/env python3
"""Quantise a BF16 Inkling checkpoint to 8-bit, one shard at a time.

mlx-vlm's own converter materialises the whole model before quantising it,
which needs far more memory than this host has, and its `sanitize` drops
`model.mtp.*` on the way in -- which is why `mlx-community/Inkling-Small-mxfp4`
has no MTP weights at all. This reads one shard, quantises it, writes it and
releases it, and copies the MTP tensors through untouched, so the result both
fits in memory and can still be used to measure multi-token-prediction
acceptance rates.

Output tensors carry the loader's post-`sanitize` names. Writing the source
names instead is not an option: `sanitize` maps `attn.wq_du.*` to
`self_attn.q_proj.weight` on the leaf-insensitive path, so a `.scales` sibling
would land on top of the weight it belongs to.

Resumable at shard granularity -- an output shard that already exists is left
alone -- and `--time-budget` stops it cleanly between shards, which is how a
two-hour job is run under a ten-minute command timeout.
"""

import argparse
import json
import os
import resource
import shutil
import struct
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.inkling.inkling import Model

GIB = 1 << 30

# `Model.sanitize` is the loader's own renaming pass, and it reads no instance
# state, so calling it unbound keeps the output names correct by construction
# rather than by a second implementation of the same table that can drift.
RENAMER = Model.__new__(Model)

LANGUAGE_PREFIX = "language_model."

# Quantise the language model's matmul weights and nothing else. Every
# `nn.Linear` and `SwitchLinear` in the decoder is named `*_proj.weight` once
# renamed -- attention q/k/v/r/o, the dense MLP, the shared experts and the
# 256-expert bank -- and nothing else is.
QUANTIZED_SUFFIX = "_proj.weight"

# What that rule deliberately leaves in its original precision, all of it
# matmul-shaped enough to be quantised by accident. The router gate and its
# bias choose which experts run, so error there changes the computation rather
# than its precision; `rel_proj` is a table of relative-position biases read by
# index; the short convolutions hold four taps a channel; and the embedding and
# unembedding sit at the two ends of the logit path whose fidelity is the thing
# this checkpoint exists to measure, for 3.1 GiB out of ~270.
UNQUANTIZED_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
    "sconv.conv.weight",
    "mlp.gate_weight",
    "self_attn.rel_proj",
)


def preserved_verbatim(name):
    """Tensors mlx-vlm's `sanitize` drops at load. Copying them through rather
    than renaming them is what keeps the MTP weights in the output at all."""
    return (
        ".mtp" in name or name.startswith("model.mtp") or name.endswith("training_args")
    )


def should_quantize(name):
    return name.startswith(LANGUAGE_PREFIX) and name.endswith(QUANTIZED_SUFFIX)


def check_left_alone(name, value):
    """Refuse to pass a matmul-shaped language-model tensor through unquantised
    unless it is one we named. Silently leaving an 8 GiB expert bank in BF16 is
    as much a bug as quantising a norm, and neither shows up in a load test."""
    if not name.startswith(LANGUAGE_PREFIX):
        return
    if value.ndim >= 2 and not name.endswith(UNQUANTIZED_SUFFIXES):
        raise ValueError(f"{name} is matmul-shaped and matches no rule")


def quantize_shard(src_file, dst_file, opts):
    weights = mx.load(str(src_file))
    verbatim = {k: v for k, v in weights.items() if preserved_verbatim(k)}
    renamable = {k: v for k, v in weights.items() if k not in verbatim}
    del weights

    out = dict(verbatim)
    renamed = Model.sanitize(RENAMER, renamable)
    del renamable

    for name in sorted(renamed):
        value = renamed.pop(name)
        if should_quantize(name):
            stem = name[: -len(".weight")]
            # `affine` returns scales and biases, the float4 modes only scales.
            fields = mx.quantize(
                value,
                group_size=opts.group_size,
                bits=opts.bits,
                mode=opts.mode,
            )
            names = [stem + s for s in (".weight", ".scales", ".biases")]
        else:
            check_left_alone(name, value)
            fields = (value,)
            names = [name]
        out.update(zip(names, fields))
        # Evaluate here, not at save time, so the shard's inputs are released as
        # they are consumed rather than all being live at once.
        mx.eval(*fields)
        del value, fields

    # `mx.save_safetensors` insists on the extension, and a half-written shard
    # must not look like a finished one to the next run, so stage it as a
    # dotfile: neither the index nor a `*.safetensors` glob will pick it up.
    tmp = dst_file.parent / f".partial-{dst_file.name}"
    mx.save_safetensors(str(tmp), out)
    tmp.rename(dst_file)


def read_header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def source_shards(src):
    index = json.loads((src / "model.safetensors.index.json").read_text())
    return sorted(set(index["weight_map"].values()))


def write_index(dst, shards):
    weight_map = {}
    total = 0
    for shard in shards:
        for name, info in read_header(dst / shard).items():
            weight_map[name] = shard
            start, end = info["data_offsets"]
            total += end - start
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map},
            indent=2,
        )
    )
    return weight_map, total


def copy_auxiliary(src, dst, opts):
    """Everything that is not a tensor. Dotfiles are HuggingFace bookkeeping,
    and the source index names tensors this conversion renames and splits, so
    it is rebuilt from the output rather than copied."""
    for path in sorted(src.iterdir()):
        if path.name.startswith(".") or path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        target = dst / path.name
        if target.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)

    config = json.loads((src / "config.json").read_text())
    quantization = {
        "group_size": opts.group_size,
        "bits": opts.bits,
        "mode": opts.mode,
    }
    # mlx-vlm reads `quantization` and re-exports `quantization_config`; the
    # mxfp4 checkpoint carries both and so must this one.
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    (dst / "config.json").write_text(json.dumps(config, indent=2))


def rss_bytes():
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip()) * 1024


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    # `affine` is the only 8-bit mode with a per-group bias as well as a scale,
    # so it fits the weight distribution instead of hanging it off a shared
    # power-of-two exponent the way `mxfp8` does, and it is what mlx-vlm
    # quantises and loads by default. Group 64 is mlx's own default: 6.25%
    # overhead against 12.5% at group 32, over groups still short enough that
    # one outlier cannot flatten the rest.
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--mode", default="affine")
    p.add_argument(
        "--time-budget",
        type=float,
        help="stop cleanly once another shard would not fit in this many seconds",
    )
    p.add_argument("--rss-limit-gib", type=float, default=96.0)
    return p.parse_args()


def main():
    opts = parse_args()
    # Quantise on the CPU. mlx maps the shard rather than reading it, so a Metal
    # kernel over a tensor on this NFS mount faults on pages arriving at 67 MB/s
    # and trips the GPU watchdog long before the tensor is whole. The CPU
    # quantiser sustains ~1.4 GB/s, twenty times what the mount delivers.
    mx.set_default_device(mx.cpu)

    opts.dst.mkdir(parents=True, exist_ok=True)
    copy_auxiliary(opts.src, opts.dst, opts)

    shards = source_shards(opts.src)
    started = time.monotonic()
    slowest = 0.0
    for shard in shards:
        target = opts.dst / shard
        if target.exists():
            continue
        elapsed = time.monotonic() - started
        if opts.time_budget and elapsed + slowest > opts.time_budget:
            print(f"time budget reached after {elapsed:.0f}s; re-run to continue")
            break

        at = time.monotonic()
        quantize_shard(opts.src / shard, target, opts)
        mx.clear_cache()
        slowest = max(slowest, time.monotonic() - at)

        rss = rss_bytes()
        print(
            f"{shard}  {time.monotonic() - at:6.0f}s  "
            f"{target.stat().st_size / GIB:6.2f} GiB out  "
            f"rss {rss / GIB:5.1f} GiB  peak {peak_rss_bytes() / GIB:5.1f} GiB",
            flush=True,
        )
        if rss > opts.rss_limit_gib * GIB:
            raise SystemExit(
                f"resident set reached {rss / GIB:.1f} GiB between shards, over "
                f"the {opts.rss_limit_gib:.0f} GiB limit; stopping before the "
                "host does"
            )

    remaining = [s for s in shards if not (opts.dst / s).exists()]
    if remaining:
        print(f"{len(remaining)}/{len(shards)} shards still to do")
        return

    weight_map, total = write_index(opts.dst, shards)
    mtp = sum(1 for name in weight_map if preserved_verbatim(name))
    print(
        f"{len(weight_map)} tensors, {mtp} of them MTP, "
        f"{total / GIB:.1f} GiB across {len(shards)} shards"
    )
    print(f"peak rss {peak_rss_bytes() / GIB:.1f} GiB")


if __name__ == "__main__":
    main()
