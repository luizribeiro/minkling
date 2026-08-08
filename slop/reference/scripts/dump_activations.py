"""Dump the intermediate activations of a fixed forward pass through a few
decoder layers, as float32, for the Rust engine to compare against.

Reading generated text cannot tell a correct port from one that uses
1/sqrt(head_dim) as the attention scale, runs the short convolutions in bf16,
folds the router's correction bias into the expert weights, or transposes the
relative-position bias. Each of those produces fluent output and wrong numbers.
Recorded tensors are the only cheap way to catch them.

Two captures come out of this, and the length is what separates them.

Eight tokens keeps a bundle small enough to commit, so `cargo test` compares
against it without the 131 GB checkpoint anywhere in sight. What it cannot
reach is anything that needs distance: at eight tokens no key is old enough to
be capped by a sliding layer's 512-token window or to fall outside a global
layer's 1024-token band.

Past the band the mask alone is `[1, 32, L, L]` — 210 MB a layer at 1280 tokens
— so that capture is generated on demand by `just dump-long-activations` and
gitignored rather than committed. It is reproducible from this file: the prompt
is one sentence repeated.

Beside the captured layers' intermediates, both ends of the model are recorded:
the embedding at the front, and at the back the final hidden state — before the
final norm and after it, because the reference distinguishes them — and the
logits the head makes of it. The forty-two layers' own intermediates are not, and
could not be committed; the end-to-end comparison is what tests the stack.

The oracle's greedy continuation of the same ids is recorded too, which is the
one thing here that is not a tensor of a single forward pass. It costs one
argmax and one cached decode step per token, and it buys a comparison the Rust
tests cannot make for themselves: the reference and this port producing the same
tokens, without a Python round trip in `cargo test`.

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

# How deep the recorded ranking goes. Deeper than any assertion is expected to
# hold — 32 ids and values per position is 2 KB — so that where the ordering
# stops surviving forty-two layers of bfloat16 is something the fixture can be
# asked rather than something the fixture has to be regenerated to answer.
TOP_K = 32

# How many tokens of greedy continuation are recorded. Longer than any test is
# expected to run, for the same reason `TOP_K` is deeper than any assertion:
# decoding a token costs the Rust engine 41 GB of dequantisation and a minute
# and a half, so how far the two agree is a number a test spends deliberately,
# and lengthening the fixture to find out should not mean loading 131 GB again.
CONTINUATION = 8


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


def record_final_hidden_state(capture, lm):
    """The last two lines of `LanguageModel.__call__`, which are the two ends of
    the model and are not interchangeable.

    `InklingModel.__call__` is called with `skip_final_norm=True` — its only
    caller always sets it — so what `layers_out` records is the state *before*
    the final norm: what `return_hidden` gives the MTP heads and what
    `speculative_verify_hidden` returns. The norm is applied one line later, on
    the way to the logits, and `norm_out` is that. A port that returned either
    where the other belonged would still generate.

    The norm is called here rather than tapped because `skip_logits=True` means
    the reference never reaches it. It is the reference's own module, over the
    reference's own tensor; nothing is recomputed."""
    capture.record("norm_out", lm.norm(capture.tensors["layers_out"]))


def _untruncated_logits(language_model, h):
    """`_logits_from_norm` with its last two lines disabled, which is where the
    padding logits are.

    `unpadded_vocab_size` is read off the config at call time, so clearing it is
    the reference's own method answering what it would answer for a checkpoint
    that stated no truncation. Nothing is reimplemented, which matters most for
    the one tensor recorded so that a port can be caught skipping the cut."""
    config = language_model.config
    stated = config.unpadded_vocab_size
    config.unpadded_vocab_size = None
    try:
        return language_model._logits_from_norm(h)
    finally:
        config.unpadded_vocab_size = stated


def record_logits(capture, language_model, top_k):
    """What `_logits_from_norm` makes of the normed state: the muP divide, the
    projection through `lm_head`, and the cut at `unpadded_vocab_size`.

    All 200058 logits of all eight positions is 6.4 MB, against a bundle that has
    been kept to a few megabytes throughout, and most of it would pin nothing:
    what survives forty-two layers of accumulated bfloat16 is the *ordering*, and
    the ordering that decides anything is at the top. So two things are recorded
    instead.

    `logits_topk_ids` and `logits_topk_values` are the top `top_k` of every
    position, which is what an argmax and a top-k comparison need, and what a
    port that dropped the muP divide fails against on values.

    `logits_untruncated` is every logit of the *last* position — the one that
    decides the next token — and it is recorded before the cut rather than after.
    That costs 966 float32 values over the truncated tensor and buys the only
    committed evidence of what the padding rows produce, which is the question
    truncation exists to answer."""
    normed = capture.tensors["norm_out"]
    logits = language_model._logits_from_norm(normed)
    untruncated = _untruncated_logits(language_model, normed)

    vocab = logits.shape[-1]
    if not mx.array_equal(untruncated[..., :vocab], logits):
        raise SystemExit("the untruncated logits do not extend the truncated ones")

    order = mx.argsort(-logits, axis=-1)[..., :top_k]
    capture.record("logits_topk_ids", order)
    capture.record("logits_topk_values", mx.take_along_axis(logits, order, axis=-1))
    capture.record("logits_untruncated", untruncated[0, -1])


def greedy_continuation(language_model, inputs, count):
    """The ids the reference generates from `inputs` by taking the argmax, one
    token at a time, against the caches `make_cache` allocates.

    This is the milestone the Rust engine's decode loop is measured against, so
    every step of it is the reference's own: `LanguageModel.__call__` produces
    the logits, `_logits_from_norm` truncates them at `unpadded_vocab_size`, and
    `mx.argmax` breaks a tie towards the lower id — which is the rule the Rust
    ranking uses too, and one that matters, because three significant digits of
    bfloat16 over 200058 logits leaves ties everywhere.

    Only the sampled token is fed back. The prompt enters the caches on the first
    call and every step after it is one token against everything before it, which
    is the decode regime rather than a prefill repeated.

    Run before the capture is instrumented, because the taps are installed on
    module *classes* and would otherwise overwrite every recorded tensor with the
    last decode step's."""
    cache = language_model.make_cache()
    ids, generated = inputs, []
    for _ in range(count):
        logits = language_model(inputs=ids, cache=cache).logits
        ids = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(ids)
        generated.append(int(ids[0, 0]))
    return mx.array([generated], dtype=mx.int32)


def instrument(capture, model):
    lm = model.language_model.model
    taps = [
        (lm, _tap(capture, "layers_out")),
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
    """Which layer is dense and which is MoE, and which is sliding and which is
    global, both come from the checkpoint, so a hard-coded set can quietly stop
    covering the router or the full-attention path — the pieces the fixture
    exists to pin — without anything looking wrong."""
    for both, holds in (
        ("a dense and a MoE MLP", config.layer_is_dense),
        ("a sliding and a global attention", config.layer_is_sliding),
    ):
        if {holds(i) for i in layer_indices} != {True, False}:
            raise SystemExit(f"layers {list(layer_indices)} do not cover both {both}")


def token_ids(processor, seq_len):
    """`seq_len` ids, and how many times the prompt had to be repeated to reach
    them.

    What the tokens say decides nothing — every layer computes the same
    arithmetic over whatever it is handed — so a capture long enough to reach
    past the relative-position band repeats one sentence rather than carrying a
    thousand tokens of prose. Nothing outside this file is needed to reproduce
    the input, and a capture short enough for one copy tokenises exactly as it
    did before there was a second."""
    tokenize = tokenizer(processor).encode
    text, repeats = PROMPT, 1
    ids = tokenize(text)
    while len(ids) < seq_len:
        text, repeats = f"{text} {PROMPT}", repeats + 1
        ids = tokenize(text)
    return ids[:seq_len], repeats


def as_fixture_dtype(value):
    """Everything float lands as float32 whatever it was computed in, so a
    comparison never has to reason about the reference's dtype choices."""
    dtype = mx.int32 if mx.issubdtype(value.dtype, mx.integer) else mx.float32
    return value.astype(dtype)


def dtype_name(value):
    return str(value.dtype).rsplit(".", 1)[-1]


def build_manifest(
    model, model_path, input_ids, repeats, layer_indices, tensors, continuation
):
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    config = model.config.text_config
    return {
        "prompt": PROMPT,
        "prompt_repeats": repeats,
        "input_ids": input_ids,
        "greedy_continuation": continuation,
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
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--layers", type=int, nargs="+", default=CAPTURED_LAYERS)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--continuation", type=int, default=CONTINUATION)
    ap.add_argument("--name", default="layer_activations")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    model, processor = load_model(args.model)
    check_layer_coverage(model.config.text_config, args.layers)
    ids, repeats = token_ids(processor, args.seq_len)
    inputs = mx.array(ids, dtype=mx.int32)[None, :]
    continuation = greedy_continuation(model.language_model, inputs, args.continuation)

    capture = Capture(args.layers)
    instrument(capture, model)
    capture.record("input_ids", inputs)
    capture.record("greedy_continuation", continuation)
    model.language_model(
        inputs=inputs, cache=model.language_model.make_cache(), skip_logits=True
    )
    record_final_hidden_state(capture, model.language_model.model)
    record_logits(capture, model.language_model, args.top_k)

    tensors = {
        name: as_fixture_dtype(value) for name, value in sorted(capture.tensors.items())
    }
    mx.eval(list(tensors.values()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"{args.name}.safetensors"
    mx.save_safetensors(str(bundle), tensors)
    manifest = build_manifest(
        model,
        args.model,
        ids,
        repeats,
        args.layers,
        tensors,
        continuation.tolist()[0],
    )
    with open(out_dir / f"{args.name}.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    for entry in manifest["tensors"]:
        print(f"{entry['name']:<34}  {entry['dtype']:<8}  {entry['shape']}")
    size = bundle.stat().st_size
    print(f"\n{len(tensors)} tensors, {size / (1 << 20):.2f} MiB -> {bundle}")


if __name__ == "__main__":
    main()
