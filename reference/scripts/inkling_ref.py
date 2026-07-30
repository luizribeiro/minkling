"""Shared setup for the reference scripts: checkpoint loading with the
eos/pad-token fix-up every Inkling checkpoint needs, and byte formatting."""

import mlx.core as mx
from mlx_vlm import load


def gib(n_bytes):
    return n_bytes / (1 << 30)


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
