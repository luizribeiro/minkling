"""Dump synthetic sequences through one multi-token prediction head, as the
oracle the Rust head is tested against.

A head is a decoder layer with three tensors in front of it: an RMSNorm over the
hidden state it is chained from, another over the embedding of the token one
position further ahead, and a `[hidden, 2 * hidden]` projection that takes the
pair back to model width. The layer itself already has a fixture; what this one
pins is the three in front of it, and two things about them that nothing in the
shapes settles:

- **The concatenation order.** `input_proj` is `[hidden, 2 * hidden]`, which
  fixes that it eats two normed vectors and not which half is which. Reversed,
  the projection reads the hidden state through the half of the weight trained
  for embeddings — a head that still runs and agrees with the model on nothing.
- **Which layer plan the block reads.** The heads carry their own
  `local_layer_ids` and are all dense, so head 1 here is a global layer where
  head 0 is a sliding one. A head built against the main stack's plan gets the
  wrong band, the wrong window and the wrong log scaling, and still runs.

Both are recorded rather than argued: the two cases differ in the layer plan,
and the reversed concatenation is checked to move the answer before the fixture
is written.

`InklingMTPLayer` is imported from the acceptance study rather than restated —
it is the module that produced the study's numbers, and a second spelling of it
here could drift from the one those were measured with.

Inkling's own head is `[4096, 8192]` in front of `[4096, 4096]` projections, so
like the layer and attention fixtures this stands the reference module up on
seeded micro-weights instead — float32 throughout, where a comparison measures
the port rather than bfloat16 rounding."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import f32, gamma, layer_parameters, projection
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.inkling.config import TextConfig
from mtp_acceptance import InklingMTPLayer

SEED = 20260901

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
D_REL = 3
SCONV_KERNEL = 4

# The window a sliding head is capped at, and the wider band a global one
# contracts against — the two apart, the way the checkpoint has them at 512 and
# 1024, so a head that read the wrong one is a head with a different mask.
SLIDING_WINDOW = 4
REL_EXTENT = 6

DENSE_INTERMEDIATE = 48

# Long enough that the prefill runs past both the window and the band, so the
# capped entries and the in-context-but-outside-band ones are both reached.
PREFILL = 8
CONTINUE = 3

# The heads' own layer plan: every head is dense, and `local_layer_ids` is
# theirs rather than the main stack's — [0, 2, 4, 5, 6, 7] in the checkpoint, so
# heads 1 and 3 are the global ones. Here head 0 is sliding and head 1 global.
HEAD_COUNT = 2
LOCAL_HEAD_IDS = [0]
CASES = {"local": 0, "global": 1}

# The slot each of the block's four short convolutions writes, by `conv_idx`.
CONV_SLOTS = ("k_sconv", "v_sconv", "attn_sconv", "mlp_sconv")


def config():
    """`mtp_text_config` applied to a small stack: the heads' own local/global
    split, and `dense_mlp_idx` past every head so none of them routes."""
    return TextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=HEAD_COUNT,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV_HEADS,
        swa_head_dim=HEAD_DIM,
        sliding_window_size=SLIDING_WINDOW,
        local_layer_ids=LOCAL_HEAD_IDS,
        d_rel=D_REL,
        rel_extent=REL_EXTENT,
        sconv_kernel_size=SCONV_KERNEL,
        dense_mlp_idx=HEAD_COUNT,
        intermediate_size=DENSE_INTERMEDIATE,
    )


def head_parameters(rng, head_index):
    """Every tensor one head holds: the block's, and the three in front of it."""
    return {
        "embed_norm.weight": gamma(rng, HIDDEN),
        "hidden_norm.weight": gamma(rng, HIDDEN),
        "input_proj.weight": projection(rng, HIDDEN, 2 * HIDDEN),
        **{
            f"transformer_block.{name}": w
            for name, w in layer_parameters(rng, config(), head_index).items()
        },
    }


def head(head_index, weights):
    layer = InklingMTPLayer(config(), head_index)
    layer.load_weights(list(weights.items()))
    return layer


def attention_config_tensor(attn):
    """The scalars `InklingAttention.__init__` derives from the config and the
    head index, in the layout the attention and layer fixtures record them in.
    A `log_floor` of zero is the None a sliding layer gets."""
    return f32(
        [
            attn.n_heads,
            attn.n_kv,
            attn.head_dim,
            attn.d_rel,
            attn.sliding,
            attn.rel_extent,
            attn.q_norm.eps,
            attn.log_floor or 0,
            attn.log_alpha,
        ]
    )


def run(layer, calls):
    """A prefill and a continuation against one cache, as a speculative round
    runs it, with the four convolution slots as the prefill left them."""
    cache = CacheList(KVCache(), ArraysCache(4))
    (hidden, embed), (continue_hidden, continue_embed) = calls
    prefill = layer(hidden, embed, cache=cache)
    mx.eval(prefill)
    conv_state = {name: cache[1][i] for i, name in enumerate(CONV_SLOTS)}
    mx.eval(list(conv_state.values()))
    return prefill, layer(continue_hidden, continue_embed, cache=cache), conv_state


def case(head_index, weights, calls):
    layer = head(head_index, weights)
    prefill, rest, conv_state = run(layer, calls)
    return layer, {
        **weights,
        "config": attention_config_tensor(layer.transformer_block.self_attn),
        "prefill_out": prefill,
        "continue_out": rest,
        **{f"conv_state.{name}": value for name, value in conv_state.items()},
    }


def check_the_cases_cover_both_layer_plans(layers):
    """Which band and window a head gets comes from the heads' own
    `local_layer_ids`, so a case list covering one kind would leave a head built
    against the main stack's plan indistinguishable from one built against
    theirs."""
    windows = {
        name: layer.transformer_block.self_attn.sliding
        for name, layer in layers.items()
    }
    if len(set(windows.values())) != 2:
        raise SystemExit(f"the cases do not cover both layer plans: {windows}")


def check_the_concatenation_order_matters(layers, weights, calls):
    """Feeding `input_proj` its two halves the other way round has to move the
    answer, or the fixture says nothing about which half is which."""
    (hidden, embed), _ = calls
    for name, layer in layers.items():
        want = np.asarray(
            layer(hidden, embed, cache=CacheList(KVCache(), ArraysCache(4)))
        )
        reversed_ = np.asarray(
            head(CASES[name], weights[name])(
                hidden,
                embed,
                cache=CacheList(KVCache(), ArraysCache(4)),
                hidden_first=False,
            )
        )
        gap = np.abs(reversed_ - want).max() / np.abs(want).max()
        if gap < 1e-2:
            raise SystemExit(
                f"{name}: reversing the concatenation moves it by {gap:.3e}"
            )
        yield name, gap


def check_the_head_moves_its_hidden_input(cases, calls):
    """A head is not a small perturbation of what it was chained from: nothing
    of the hidden state reaches the output except through `input_proj`, so a
    port that passed it straight through has to be far outside any tolerance."""
    (hidden, _), _ = calls
    for name, values in cases.items():
        out = np.asarray(values["prefill_out"])
        moved = np.abs(out - np.asarray(hidden)).max() / np.abs(out).max()
        if moved < 1e-2:
            raise SystemExit(f"{name}: the head moves its hidden input by {moved:.3e}")
        yield name, moved


def collect():
    rng = np.random.default_rng(SEED)
    sequence = lambda rows: f32(rng.standard_normal((1, rows, HIDDEN)))  # noqa: E731
    calls = (
        (sequence(PREFILL), sequence(PREFILL)),
        (sequence(CONTINUE), sequence(CONTINUE)),
    )

    weights, layers, cases = {}, {}, {}
    for name, head_index in CASES.items():
        weights[name] = head_parameters(rng, head_index)
        layers[name], cases[name] = case(head_index, weights[name], calls)

    (hidden, embed), (continue_hidden, continue_embed) = calls
    tensors = {
        "hidden": hidden,
        "embed": embed,
        "continue_hidden": continue_hidden,
        "continue_embed": continue_embed,
    }
    for name, values in cases.items():
        tensors.update({f"{name}.{field}": v for field, v in values.items()})
    mx.eval(list(tensors.values()))
    return tensors, layers, cases, weights, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    tensors, layers, cases, weights, calls = collect()
    check_the_cases_cover_both_layer_plans(layers)
    reversed_ = list(check_the_concatenation_order_matters(layers, weights, calls))
    moved = list(check_the_head_moves_its_hidden_input(cases, calls))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "mtp.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    print(f"{bundle}  {len(tensors)} tensors")
    for name, gap in reversed_:
        print(f"  {name}: reversing the concatenation moves the answer by {gap:.3e}")
    for name, gap in moved:
        print(f"  {name}: the head moves its hidden input by {gap:.3e}")


if __name__ == "__main__":
    main()
