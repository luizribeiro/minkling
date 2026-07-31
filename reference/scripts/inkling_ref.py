"""Shared setup for the reference scripts: checkpoint loading with the
eos/pad-token fix-up every Inkling checkpoint needs, shard-level access for the
scripts that want a few tensors rather than a model, the seeded draws the
synthetic fixtures stand a module up from, and byte formatting."""

import json
from functools import lru_cache

import mlx.core as mx
import numpy as np
from mlx_vlm import load

# The decoder layers `dump_activations.py` captures, and so the layers every
# trained fixture is cut from. Three scripts have to agree on them — the
# activations, the sconv kernels and the mask projections are all indexed by
# layer, and a bundle that names a layer the others do not is a fixture the
# Rust side cannot open.
#
# `check_layer_coverage` refuses a set that stops covering what the fixtures
# exist to pin.
CAPTURED_LAYERS = (0, 2)


def gib(n_bytes):
    return n_bytes / (1 << 30)


@lru_cache(maxsize=1)
def index_of(model_path):
    return json.loads((model_path / "model.safetensors.index.json").read_text())


def load_shard(model_path, shard):
    """`mx.load` is lazy, so a shard costs only the tensors actually evaluated."""
    return mx.load(str(model_path / shard))


def checkpoint_tensor(model_path, name):
    """One named tensor, straight out of the shard that owns it. Materialising
    the 130 GiB model to read a few hundred KiB has crashed this host before."""
    return load_shard(model_path, index_of(model_path)["weight_map"][name])[name]


def f32(values):
    return mx.array(values, dtype=mx.float32)


def projection(rng, out_dim, in_dim):
    """An `nn.Linear` weight, `[out, in]`, scaled by fan-in so what it produces
    lands in the same range the trained activations do — where an activation is
    still curved, and where a bias is still comparable to a logit."""
    return f32(rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim))


def gamma(rng, dims):
    """A trained RMSNorm weight sits near 1 and is not uniform, so reading it as
    a scalar — or dropping it — has to change the answer."""
    return f32(1.0 + 0.5 * rng.standard_normal(dims))


def taps(rng, channels, kernel_size):
    """A short convolution's kernel, `[channels, kernel_size, 1]` as `nn.Conv1d`
    and the checkpoint both store it.

    Drawn so no two taps of a channel are close in magnitude: a kernel read in
    reversed time order still produces plausible numbers, and only an asymmetric
    kernel makes it produce different ones. The ramp is about the four-fold one
    the trained kernels show, which also keeps the convolution and the residual
    within a factor of a few of each other — a residual lost in the noise would
    be a residual nothing tests."""
    magnitude = 0.4 * 1.6 ** np.arange(kernel_size)
    signs = rng.choice([-1.0, 1.0], size=(channels, kernel_size))
    spread = 1.0 + 0.25 * rng.standard_normal((channels, kernel_size))
    return f32((signs * spread * magnitude)[..., None])


def tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _resolve_eos_token(model, processor):
    # The checkpoint lists every special token under additional_special_tokens
    # and names no eos, so mlx-vlm's `pad_token = eos_token` fallback assigns
    # None and padding fails. Resolve eos from the model config instead.
    tok = tokenizer(processor)
    if tok.eos_token is None:
        tok.eos_token = tok.convert_ids_to_tokens(model.config.eos_token_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token


def load_model(path):
    model, processor = load(path)
    mx.eval(model.parameters())
    _resolve_eos_token(model, processor)
    return model, processor
