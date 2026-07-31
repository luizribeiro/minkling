"""Shared setup for the reference scripts: checkpoint loading with the
eos/pad-token fix-up every Inkling checkpoint needs, shard-level access for the
scripts that want a few tensors rather than a model, and byte formatting."""

import json
from functools import lru_cache

import mlx.core as mx
from mlx_vlm import load


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
