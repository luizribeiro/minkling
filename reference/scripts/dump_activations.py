"""Dump the intermediate activations of a fixed 8-token forward pass through one
dense layer and one MoE layer, as float32, for the Rust engine to compare
against.

Reading generated text cannot tell a correct port from one that uses
1/sqrt(head_dim) as the attention scale, runs the short convolutions in bf16,
folds the router's correction bias into the expert weights, or transposes the
relative-position bias. Each of those produces fluent output and wrong numbers.
Recorded tensors are the only cheap way to catch them.

Eight tokens keeps the bundle small enough to commit, so `cargo test` compares
against it without the 131 GB checkpoint anywhere in sight.

Everything is captured by wrapping the reference's own modules and functions,
never by recomputing: a recomputation that drifts from the reference would make
the fixture agree with the bug it is supposed to catch."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from inkling_ref import CAPTURED_LAYERS, load_model, tokenizer
from mlx_vlm.models.inkling import language

PROMPT = "The lighthouse keeper counted the ships that passed the headland."
SEQ_LEN = 8


class Capture:
    """Collects named tensors from a forward pass. Names are prefixed with the
    decoder layer in flight, and layers outside the captured set record
    nothing, so the other 40 cost only a dictionary lookup."""

    def __init__(self, layer_indices):
        self.layer_indices = tuple(layer_indices)
        self.layer = None
        self.tensors = {}

    def record(self, name, value):
        if self.layer is None:
            self.tensors[name] = value
        elif self.layer in self.layer_indices:
            self.tensors[f"layer{self.layer}.{name}"] = value


def _tap(capture, output_as=None, input_as=None):
    def hook(args, out):
        if input_as is not None:
            capture.record(input_as, args[0])
        if output_as is not None:
            capture.record(output_as, out)

    return hook


def _instrument_class(cls, hooks):
    """mlx modules are dicts and dispatch ``__call__`` through the type, so a
    hook cannot be attached to one instance; it lives on the class and selects
    by identity."""
    inner = cls.__call__

    def wrapped(self, *args, **kwargs):
        out = inner(self, *args, **kwargs)
        hook = hooks.get(id(self))
        if hook is not None:
            hook(args, out)
        return out

    cls.__call__ = wrapped


def _instrument_modules(taps):
    hooks = {id(module): hook for module, hook in taps}
    for cls in {type(module) for module, _ in taps}:
        _instrument_class(cls, hooks)


def _instrument_layers(capture, layers):
    """Track which decoder layer is executing, and record what it consumed and
    what it returned.

    The input is recorded even though the first captured layer's is
    ``embed_norm_out``, because the layers between two captures record nothing
    and the second one's input appears nowhere else. It can be recovered as
    ``h - attn_sconv_out``, but only by assuming the residual wiring that is
    itself under test."""
    inner = type(layers[0]).__call__
    index = {id(layer): i for i, layer in enumerate(layers)}

    def wrapped(self, *args, **kwargs):
        capture.layer = index[id(self)]
        capture.record("input", args[0])
        out = inner(self, *args, **kwargs)
        capture.record("out", out)
        capture.layer = None
        return out

    type(layers[0]).__call__ = wrapped


def _instrument_attention(capture):
    """Take q, k and the mask as attention receives them, and the relative-position
    input as the mask kernel receives it: those call sites carry the norms,
    reshapes and transposes the reference applies inline, and on a global layer
    they also carry the log-scaling that rescales q and the mask after the
    kernel returns. Pinning what attention consumed is the point."""
    banded, sdpa = language.banded_additive_mask, language.scaled_dot_product_attention

    def _banded_additive_mask(rel, *args, **kwargs):
        capture.record("r_proj_out", rel)
        return banded(rel, *args, **kwargs)

    def _scaled_dot_product_attention(q, k, *args, mask, **kwargs):
        out = sdpa(q, k, *args, mask=mask, **kwargs)
        capture.record("q_norm_out", q)
        capture.record("k_norm_out", k)
        capture.record("mask", mask)
        capture.record("sdpa_out", out)
        return out

    language.banded_additive_mask = _banded_additive_mask
    language.scaled_dot_product_attention = _scaled_dot_product_attention


def _instrument_moe(capture):
    if not hasattr(language, "moe_tap"):
        raise SystemExit(
            "the installed mlx-vlm has no MoE observation tap; run 'just sync'"
        )

    def tap(module, tensors):
        for name, value in tensors.items():
            capture.record(name, value)

    language.moe_tap = tap


def instrument(capture, model):
    lm = model.language_model.model
    taps = [
        (lm.embed_tokens, _tap(capture, "embed_out")),
        (lm.embed_norm, _tap(capture, "embed_norm_out")),
    ]
    for i in capture.layer_indices:
        layer = lm.layers[i]
        attn = layer.self_attn
        taps += [
            (layer.input_layernorm, _tap(capture, "input_layernorm_out")),
            (attn.q_proj, _tap(capture, "q_proj_out")),
            (attn.k_proj, _tap(capture, "k_proj_out")),
            (attn.v_proj, _tap(capture, "v_proj_out")),
            (attn.k_sconv, _tap(capture, "k_sconv_out")),
            (attn.v_sconv, _tap(capture, "v_sconv_out")),
            (attn.o_proj, _tap(capture, "o_proj_out")),
            (layer.attn_sconv, _tap(capture, "attn_sconv_out")),
            (
                layer.post_attention_layernorm,
                _tap(capture, "post_attention_ln_out", input_as="h"),
            ),
            (layer.mlp, _tap(capture, "mlp_out")),
            (layer.mlp_sconv, _tap(capture, "mlp_sconv_out")),
        ]
    _instrument_modules(taps)
    _instrument_layers(capture, lm.layers)
    _instrument_attention(capture)
    _instrument_moe(capture)


def check_layer_coverage(config, layer_indices):
    """Which layer is dense and which is MoE comes from the checkpoint, so a
    hard-coded pair can quietly stop covering the router — the one piece the
    fixture exists to pin — without anything looking wrong."""
    if {config.layer_is_dense(i) for i in layer_indices} != {True, False}:
        raise SystemExit(
            f"layers {list(layer_indices)} do not cover both a dense and a MoE MLP"
        )


def token_ids(processor):
    ids = tokenizer(processor).encode(PROMPT)[:SEQ_LEN]
    if len(ids) < SEQ_LEN:
        raise SystemExit(f"prompt tokenises to {len(ids)} ids, need {SEQ_LEN}")
    return ids


def as_fixture_dtype(value):
    """Everything float lands as float32 whatever it was computed in, so a
    comparison never has to reason about the reference's dtype choices."""
    dtype = mx.int32 if mx.issubdtype(value.dtype, mx.integer) else mx.float32
    return value.astype(dtype)


def dtype_name(value):
    return str(value.dtype).rsplit(".", 1)[-1]


def build_manifest(model, model_path, input_ids, layer_indices, tensors):
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    config = model.config.text_config
    return {
        "prompt": PROMPT,
        "input_ids": input_ids,
        "checkpoint": {
            "path": str(model_path),
            "total_size": index["metadata"]["total_size"],
        },
        "layers": [
            {
                "index": i,
                "attention": "sliding" if config.layer_is_sliding(i) else "global",
                "mlp": "dense" if config.layer_is_dense(i) else "moe",
            }
            for i in layer_indices
        ],
        "tensors": [
            {"name": name, "shape": list(value.shape), "dtype": dtype_name(value)}
            for name, value in tensors.items()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    model, processor = load_model(args.model)
    check_layer_coverage(model.config.text_config, CAPTURED_LAYERS)
    ids = token_ids(processor)
    inputs = mx.array(ids, dtype=mx.int32)[None, :]

    capture = Capture(CAPTURED_LAYERS)
    instrument(capture, model)
    capture.record("input_ids", inputs)
    model.language_model(
        inputs=inputs, cache=model.language_model.make_cache(), skip_logits=True
    )

    tensors = {
        name: as_fixture_dtype(value) for name, value in sorted(capture.tensors.items())
    }
    mx.eval(list(tensors.values()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "layer_activations.safetensors"), tensors)
    manifest = build_manifest(model, args.model, ids, CAPTURED_LAYERS, tensors)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    for entry in manifest["tensors"]:
        print(f"{entry['name']:<34}  {entry['dtype']:<8}  {entry['shape']}")
    size = (out_dir / "layer_activations.safetensors").stat().st_size
    print(f"\n{len(tensors)} tensors, {size / (1 << 20):.2f} MiB -> {out_dir}")


if __name__ == "__main__":
    main()
