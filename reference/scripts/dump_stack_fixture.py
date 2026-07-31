"""Dump a synthetic model stack through mlx-vlm's own `InklingModel`, as the
oracle the assembled Rust stack is tested against.

Every piece the stack is built from already has a fixture of its own, and the
decoder layer has one for its assembly. What this pins is the assembly *of the
layers*: that they run in order, that each gets the attention config and the MLP
its index calls for, that each keeps its own cache across calls, and where the
final norm sits.

Three of those are invisible in anything but a multi-layer model.

- **Layer order.** Exchanging two layers leaves a stack that runs and generates.
  Only a model whose layers differ from each other can say otherwise, so the
  five here differ in both ways a real layer can: 0 and 1 are dense and the rest
  MoE, and 2 is the only global one.
- **One cache per layer.** `make_cache` allocates a `CacheList(KVCache(),
  ArraysCache(4))` for each, and a stack that threaded one cache through all of
  them still answers, because the sliding layers here are all the same shape.
  Each call is therefore run twice, and the second is where a shared cache
  shows.
- **The final norm.** `InklingModel.__call__` applies it unless
  `skip_final_norm`, and its only caller sets that flag and applies
  `self.model.norm` separately on the way to the logits. Both answers are
  recorded, because a port that returned the wrong one still makes logits.

The two sets of attention head fields are set *apart* here — a sliding layer
gets 2 KV heads of width 8 and a global one 2 of width 16 — which
Inkling-Small's config cannot do, because it sets both sets to the same numbers.
A stack that read one set for every layer builds a cache of the wrong stride,
and only a config whose sets differ makes that visible.

Inkling's own stack is 42 layers of `[4096, 4096]` projections over 256 experts,
so like the attention, MoE and layer fixtures this one stands the reference
module up on seeded micro-weights instead — float32 throughout, where a
comparison measures the port rather than bfloat16 rounding. The trained stack is
left to the checkpoint-gated tests.

The config is written out beside the bundle, in the spelling a checkpoint's
`config.json` uses, and is what both sides are built from: `TextConfig.from_dict`
reads it here and serde reads it in the Rust tests, so the two cannot drift.

Nothing here reimplements an op. Every recorded value comes from calling
`InklingModel` with the caches `LanguageModel.make_cache` allocates."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import f32, gamma, layer_parameters
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import (
    InklingDenseMLP,
    InklingSparseMoE,
    LanguageModel,
)

SEED = 20260731

VOCAB = 48
HIDDEN = 32
D_REL = 3
SCONV_KERNEL = 4

# A sliding layer's keys are 2 x 8 and a global layer's are 2 x 16, so the two
# sets of head fields cannot be confused for one another. Both give 32 query
# channels, which `o_proj` maps back to `hidden`.
SWA_HEADS, SWA_KV_HEADS, SWA_HEAD_DIM = 4, 2, 8
HEADS, KV_HEADS, HEAD_DIM = 2, 2, 16

# Narrow enough that a prefill of six tokens reaches past it, so the window cap
# runs on the sliding layers. A sliding layer's band is its window; the global
# layer's is `rel_extent`, and 8 puts the continuation's last key outside it.
SLIDING_WINDOW = 4
REL_EXTENT = 8

# Layers 0 and 1 are dense and the other three MoE; 2 is the only global one.
# That covers both MLPs and both attentions, and puts the global layer in the
# middle so an order mistake cannot be a boundary effect.
#
# Five rather than four because the order check needs a pair of layers that
# could be exchanged at all: two layers of different attention configs or
# different MLPs have different shapes, so exchanging them fails to load rather
# than answering differently. This leaves one exchangeable dense pair (0, 1) and
# one MoE pair (3, 4).
LAYERS = 5
DENSE_MLP_IDX = 2
LOCAL_LAYER_IDS = [0, 1, 3, 4]

DENSE_INTERMEDIATE = 48
MOE_INTERMEDIATE = 16
N_ROUTED = 16
N_SHARED = 2
TOP_K = 3
ROUTE_SCALE = 8.0

PREFILL = 6
CONTINUE = 3

CALLS = ("prefill", "continue")


def config_dict():
    """The config as a checkpoint's `config.json` spells it: `intermediate_size`
    is the per-expert width and `dense_intermediate_size` the dense one, which
    is the transposition the README calls load-bearing. `TextConfig.__post_init__`
    rebinds them for the reference, and serde reads the same two names."""
    return {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "unpadded_vocab_size": None,
        "model_max_length": 1024,
        "rms_norm_eps": 1e-6,
        "use_embed_norm": True,
        "logits_mup_width_multiplier": 1.0,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "swa_num_attention_heads": SWA_HEADS,
        "swa_num_key_value_heads": SWA_KV_HEADS,
        "swa_head_dim": SWA_HEAD_DIM,
        "sliding_window_size": SLIDING_WINDOW,
        "local_layer_ids": LOCAL_LAYER_IDS,
        "d_rel": D_REL,
        "rel_extent": REL_EXTENT,
        "log_scaling_n_floor": None,
        "log_scaling_alpha": 0.1,
        "use_sconv": True,
        "sconv_kernel_size": SCONV_KERNEL,
        "dense_mlp_idx": DENSE_MLP_IDX,
        "dense_intermediate_size": DENSE_INTERMEDIATE,
        "intermediate_size": MOE_INTERMEDIATE,
        "n_routed_experts": N_ROUTED,
        "num_experts_per_tok": TOP_K,
        "n_shared_experts": N_SHARED,
        "route_scale": ROUTE_SCALE,
        "use_gate_bias": True,
        "norm_after_topk": True,
        "shared_expert_sink": True,
    }


def text_config():
    return TextConfig.from_dict(config_dict())


def stack_parameters(rng, config):
    """Every tensor `InklingModel` holds: the embedding table, the two norms
    outside the layers, and each layer's own.

    `inkling_ref.layer_parameters` draws a layer's, reading every width off the
    config, which is what lets the five layers here differ from each other while
    the layer fixture's two do not."""
    weights = {
        "embed_tokens.weight": f32(rng.standard_normal((config.vocab_size, HIDDEN))),
        "embed_norm.weight": gamma(rng, HIDDEN),
        "norm.weight": gamma(rng, HIDDEN),
    }
    for layer in range(config.num_hidden_layers):
        for name, value in layer_parameters(rng, config, layer).items():
            weights[f"layers.{layer}.{name}"] = value
    return weights


def model(config, weights):
    """`InklingModel`, by way of the `LanguageModel` that owns `make_cache`, so
    the caches driven here are the ones a real decode allocates."""
    lm = LanguageModel(config)
    lm.model.load_weights([(name, value) for name, value in weights.items()])
    return lm


def run(lm, calls):
    """Each call's two answers against one set of caches: the hidden state the
    layers produced, and what the final norm made of it.

    `skip_final_norm=True` and a separate `self.model.norm` is not a choice made
    here — it is verbatim what `LanguageModel.__call__` does one line above the
    logits."""
    caches = lm.make_cache()
    for ids in calls:
        layers_out = lm.model(ids, caches, skip_final_norm=True)
        norm_out = lm.model.norm(layers_out)
        mx.eval(layers_out, norm_out)
        yield layers_out, norm_out


def check_the_layers_are_the_shapes_the_config_asks_for(lm, config):
    """The layer weights come from another dump's constants and the config from
    this one's. A stack whose `q_proj` disagreed with its `num_attention_heads`
    would fail somewhere far less legible than here."""
    for i, layer in enumerate(lm.model.layers):
        attn = layer.self_attn
        sliding = config.layer_is_sliding(i)
        want = {
            "n_heads": SWA_HEADS if sliding else HEADS,
            "n_kv": SWA_KV_HEADS if sliding else KV_HEADS,
            "head_dim": SWA_HEAD_DIM if sliding else HEAD_DIM,
        }
        got = {name: getattr(attn, name) for name in want}
        if got != want:
            raise SystemExit(f"layer {i}: attention is {got}, not {want}")
        if attn.q_proj.weight.shape != (want["n_heads"] * want["head_dim"], HIDDEN):
            raise SystemExit(f"layer {i}: q_proj is {attn.q_proj.weight.shape}")


def check_the_stack_covers_both_of_everything(lm, config):
    """Which MLP and which attention a layer gets both come from the config, so
    a layer list that quietly stopped covering one of them would leave the
    per-layer wiring untested — and would make the order test pass for the wrong
    reason, because identical layers commute."""
    mlps = {type(layer.mlp) for layer in lm.model.layers}
    if mlps != {InklingDenseMLP, InklingSparseMoE}:
        raise SystemExit(f"the stack does not cover both MLPs: {mlps}")
    slidings = {config.layer_is_sliding(i) for i in range(config.num_hidden_layers)}
    if slidings != {True, False}:
        raise SystemExit("the stack does not cover both attentions")
    if SWA_KV_HEADS * SWA_HEAD_DIM == KV_HEADS * HEAD_DIM:
        raise SystemExit("the two sets of head fields agree on the key width")


def check_the_caches_are_one_per_layer(lm, config):
    caches = lm.make_cache()
    if len(caches) != config.num_hidden_layers:
        raise SystemExit(f"{len(caches)} caches for {config.num_hidden_layers} layers")


def interchangeable(config):
    """Adjacent layers a stack could exchange and still run: same attention
    config and same MLP, and so the same shapes throughout.

    Any other pair differs in shape, so exchanging it fails to load rather than
    answering differently — which would say nothing about order. Derived rather
    than listed, so that a change to `local_layer_ids` or `dense_mlp_idx` cannot
    leave the pairs behind."""
    pairs = [
        (a, a + 1)
        for a in range(config.num_hidden_layers - 1)
        if config.layer_is_sliding(a) == config.layer_is_sliding(a + 1)
        and config.layer_is_dense(a) == config.layer_is_dense(a + 1)
    ]
    kinds = {config.layer_is_dense(a) for a, _ in pairs}
    if kinds != {True, False}:
        raise SystemExit(f"the exchangeable pairs {pairs} cover only {kinds}")
    return pairs


def check_exchanging_two_layers_moves_the_answer(config, weights, calls, recorded):
    """The mutation the Rust test's order case makes. If two layers commuted,
    that test would pass without the stack running them in order at all."""
    for a, b in interchangeable(config):
        swapped = dict(weights)
        for name, value in weights.items():
            for src, dst in ((a, b), (b, a)):
                if name.startswith(f"layers.{src}."):
                    swapped[name.replace(f"layers.{src}.", f"layers.{dst}.", 1)] = value
        out = list(run(model(config, swapped), calls))
        want = np.asarray(recorded[-1][0])
        gap = np.abs(np.asarray(out[-1][0]) - want).max() / np.abs(want).max()
        if gap < 1e-2:
            raise SystemExit(f"layers {a} and {b} commute, at {gap:.3e}")
        yield (a, b), gap


def check_the_continuation_reads_the_caches(config, weights, calls, recorded):
    """The second call has to depend on the first, or a fixture that only ever
    prefilled would let a port drop the caches entirely."""
    fresh = list(run(model(config, weights), calls[1:]))
    want = np.asarray(recorded[1][0])
    gap = np.abs(np.asarray(fresh[0][0]) - want).max() / np.abs(want).max()
    if gap < 1e-2:
        raise SystemExit(f"the caches move the continuation by only {gap:.3e}")
    return gap


def check_the_final_norm_moves_the_answer(recorded):
    """The stack returns the pre-norm state and the norm is applied separately.
    If the norm barely moved the answer, a port that returned the wrong one
    would sit inside the Rust test's tolerance."""
    for call, (layers_out, norm_out) in zip(CALLS, recorded):
        layers_out, norm_out = np.asarray(layers_out), np.asarray(norm_out)
        gap = np.abs(norm_out - layers_out).max() / np.abs(layers_out).max()
        if gap < 1e-2:
            raise SystemExit(f"{call}: the final norm moves the answer by {gap:.3e}")
        yield call, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    config = text_config()
    weights = stack_parameters(rng, config)
    ids = [
        mx.array(rng.integers(0, VOCAB, size=(1, length)), dtype=mx.int32)
        for length in (PREFILL, CONTINUE)
    ]

    lm = model(config, weights)
    check_the_layers_are_the_shapes_the_config_asks_for(lm, config)
    check_the_stack_covers_both_of_everything(lm, config)
    check_the_caches_are_one_per_layer(lm, config)
    recorded = list(run(lm, ids))

    tensors = {
        **weights,
        "input_ids": ids[0],
        "continue_ids": ids[1],
        **{
            f"{call}.{name}": value
            for call, answers in zip(CALLS, recorded)
            for name, value in zip(("layers_out", "norm_out"), answers)
        },
    }
    # A layer with a router records its shape, and a dense layer records none,
    # which is the fixture saying which MLP the layer index called for.
    for layer in range(config.num_hidden_layers):
        if not config.layer_is_dense(layer):
            tensors[f"layers.{layer}.moe_config"] = f32(
                [N_ROUTED, N_SHARED, TOP_K, ROUTE_SCALE]
            )
    mx.eval(list(tensors.values()))

    exchanged = list(
        check_exchanging_two_layers_moves_the_answer(config, weights, ids, recorded)
    )
    cached = check_the_continuation_reads_the_caches(config, weights, ids, recorded)
    normed = list(check_the_final_norm_moves_the_answer(recorded))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "stack.safetensors"
    mx.save_safetensors(str(bundle), tensors)
    with open(out_dir / "stack.json", "w") as f:
        json.dump({"text_config": config_dict()}, f, indent=2)
        f.write("\n")

    for name, value in sorted(tensors.items()):
        peak = float(mx.abs(value).max())
        print(f"{name:<46}  {str(list(value.shape)):<14}  max |x| {peak:.4g}")
    print("\nexchanging two layers moves the answer by:")
    for (a, b), gap in exchanged:
        print(f"  {a} <-> {b}  {gap:.3e}")
    print(f"the caches move the continuation by: {cached:.3e}")
    print("the final norm moves the answer by:")
    for call, gap in normed:
        print(f"  {call:<9}  {gap:.3e}")
    print(f"{len(tensors)} tensors, {bundle.stat().st_size / 1024:.1f} KiB -> {bundle}")


if __name__ == "__main__":
    main()
