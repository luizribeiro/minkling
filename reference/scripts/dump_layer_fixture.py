"""Dump synthetic sequences through mlx-vlm's own `InklingDecoderLayer`, as the
oracle the assembled Rust layer is tested against.

Every op the layer is built from already has a fixture of its own. What this one
pins is the assembly: the order the pieces run in, the two residual adds, and
the single four-slot convolution cache the layer threads through all four of its
short convolutions.

Two of those are invisible in any single forward pass.

- A residual is added to the value *before* its norm — `x` before
  `input_layernorm`, `h` before `post_attention_layernorm`. Adding what the norm
  produced instead is the ordinary pre-norm/post-norm slip and leaves a layer
  that still runs.
- `conv_idx` picks each convolution's slot: 0 for the key's and 1 for the
  value's inside attention, 2 for `attn_sconv` and 3 for `mlp_sconv`. A slot is
  written at the end of a call and read at the start of the next, so a port that
  exchanged the last two agrees for exactly one call. Each case is therefore run
  twice against one cache, and the four slots are recorded between the calls.

Both MLPs are covered, because the layer is one structure whose only variation
is that slot: the dense case is a layer below `dense_mlp_idx` and the MoE case
one above it.

Inkling's own layer is `[4096, 4096]` projections over 256 experts, so like the
attention and MoE fixtures this one stands the reference module up on seeded
micro-weights instead — float32 throughout, where a comparison measures the port
rather than bfloat16 rounding. The trained layer is left to the
checkpoint-gated tests.

Nothing here reimplements an op. Every recorded value comes from calling
`InklingDecoderLayer` with the cache `LanguageModel.make_cache` allocates."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import f32, layer_parameters
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import (
    InklingDecoderLayer,
    InklingDenseMLP,
    InklingSparseMoE,
)

SEED = 20260804

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
D_REL = 3
SCONV_KERNEL = 4

# Narrow enough that a prefill of eight tokens reaches past it, so the window
# cap runs. A sliding layer's band is its window, so this is the mask's extent
# too.
SLIDING_WINDOW = 4

DENSE_INTERMEDIATE = 48
MOE_INTERMEDIATE = 16
N_ROUTED = 16
N_SHARED = 2
TOP_K = 3

# The checkpoint's own route_scale. The two global scales the layers are drawn
# with are `inkling_ref`'s, shared with the stack fixture.
ROUTE_SCALE = 8.0

PREFILL = 8
CONTINUE = 3

# Layers below `dense_mlp_idx` are dense and the rest are MoE. Both cases here
# are sliding: what a global layer does differently sits in attention, where the
# activation dump covers one against real weights and the attention fixture pins
# log scaling.
DENSE_LAYER = 0
MOE_LAYER = 1
CASES = {"dense": DENSE_LAYER, "moe": MOE_LAYER}

# The slot each of the layer's four short convolutions writes, by `conv_idx`.
CONV_SLOTS = ("k_sconv", "v_sconv", "attn_sconv", "mlp_sconv")


def config():
    return TextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV_HEADS,
        swa_head_dim=HEAD_DIM,
        sliding_window_size=SLIDING_WINDOW,
        local_layer_ids=[DENSE_LAYER, MOE_LAYER],
        d_rel=D_REL,
        rel_extent=SLIDING_WINDOW,
        sconv_kernel_size=SCONV_KERNEL,
        dense_mlp_idx=MOE_LAYER,
        intermediate_size=DENSE_INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=N_ROUTED,
        num_experts_per_tok=TOP_K,
        n_shared_experts=N_SHARED,
        route_scale=ROUTE_SCALE,
    )


def decoder_layer(layer_index, weights):
    layer = InklingDecoderLayer(config(), layer_index)
    layer.load_weights(list(weights.items()))
    return layer


def attention_config_tensor(attn):
    """The scalars `InklingAttention.__init__` derives from the config and the
    layer index, in the layout the attention fixture records them in. A
    `log_floor` of zero is the None a sliding layer gets."""
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


def run(layer, x, x_continue):
    """A prefill and a continuation against one cache, as decoding runs it, with
    the four convolution slots as the prefill left them.

    The slots are copied out between the calls because the continuation
    overwrites them, and that overwriting is the point: what it reads first is
    what is recorded here."""
    cache = CacheList(KVCache(), ArraysCache(4))
    prefill = layer(x, cache=cache)
    mx.eval(prefill)
    conv_state = {name: cache[1][i] for i, name in enumerate(CONV_SLOTS)}
    mx.eval(list(conv_state.values()))
    return prefill, layer(x_continue, cache=cache), conv_state


def case(layer_index, weights, x, x_continue):
    layer = decoder_layer(layer_index, weights)
    prefill, rest, conv_state = run(layer, x, x_continue)

    tensors = {
        **weights,
        "config": attention_config_tensor(layer.self_attn),
        "prefill_out": prefill,
        "continue_out": rest,
        **{f"conv_state.{name}": value for name, value in conv_state.items()},
    }
    # A layer with a router records its shape, and a dense layer records none,
    # which is the fixture saying which MLP the layer index called for.
    if isinstance(layer.mlp, InklingSparseMoE):
        tensors["moe_config"] = f32([N_ROUTED, N_SHARED, TOP_K, ROUTE_SCALE])
    return layer, tensors


def check_the_cases_cover_both_mlps(layers):
    """Which MLP a layer gets comes from `dense_mlp_idx`, so a case list that
    quietly stopped covering one of them would leave half the layer untested."""
    kinds = {name: type(layer.mlp) for name, layer in layers.items()}
    if set(kinds.values()) != {InklingDenseMLP, InklingSparseMoE}:
        raise SystemExit(f"the cases do not cover both MLPs: {kinds}")


def check_the_slots_are_the_two_widths(tensors):
    """The four slots are attention's two and the layer's two, and the recorded
    widths are what says which pair is which. Equal widths would leave a port
    free to exchange the pairs."""
    for name, values in tensors.items():
        widths = {
            slot: np.asarray(values[f"conv_state.{slot}"]).shape[-1]
            for slot in CONV_SLOTS
        }
        want = {
            "k_sconv": KV_HEADS * HEAD_DIM,
            "v_sconv": KV_HEADS * HEAD_DIM,
            "attn_sconv": HIDDEN,
            "mlp_sconv": HIDDEN,
        }
        if widths != want:
            raise SystemExit(f"{name}: conv slots are {widths}, not {want}")
        for slot, value in ((s, values[f"conv_state.{s}"]) for s in CONV_SLOTS):
            if value.shape[-2] != SCONV_KERNEL - 1:
                raise SystemExit(f"{name}: {slot} keeps {value.shape[-2]} timesteps")


def check_the_continuation_reads_the_cache(cases, weights, x_continue):
    """The second call has to depend on the first, or a fixture that only ever
    prefilled would let a port drop the cache entirely."""
    for name, values in cases.items():
        fresh = decoder_layer(CASES[name], weights[name])
        alone = fresh(x_continue, cache=CacheList(KVCache(), ArraysCache(4)))
        want = np.asarray(values["continue_out"])
        gap = np.abs(np.asarray(alone) - want).max() / np.abs(want).max()
        if gap < 1e-2:
            raise SystemExit(f"{name}: the cache moves the continuation by {gap:.3e}")
        yield name, gap


def check_the_layer_moves_its_input(cases, x):
    """The residual carries the input straight through, so most of the output is
    the input. What has to stay visible is the rest: a layer that moved its input
    by almost nothing would put every mutation the Rust tests make inside their
    tolerance."""
    for name, values in cases.items():
        out = np.asarray(values["prefill_out"])
        moved = np.abs(out - np.asarray(x)).max() / np.abs(out).max()
        if moved < 1e-2:
            raise SystemExit(f"{name}: the layer moves its input by only {moved:.3e}")
        yield name, moved


def collect():
    rng = np.random.default_rng(SEED)
    x = f32(rng.standard_normal((1, PREFILL, HIDDEN)))
    x_continue = f32(rng.standard_normal((1, CONTINUE, HIDDEN)))

    weights, layers, cases = {}, {}, {}
    for name, layer_index in CASES.items():
        weights[name] = layer_parameters(rng, config(), layer_index)
        layers[name], cases[name] = case(layer_index, weights[name], x, x_continue)

    tensors = {"x": x, "continue_x": x_continue}
    for name, values in cases.items():
        tensors.update({f"{name}.{field}": v for field, v in values.items()})
    mx.eval(list(tensors.values()))
    return tensors, layers, cases, weights, x, x_continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    tensors, layers, cases, weights, x, x_continue = collect()
    check_the_cases_cover_both_mlps(layers)
    check_the_slots_are_the_two_widths(cases)
    cached = list(check_the_continuation_reads_the_cache(cases, weights, x_continue))
    moved = list(check_the_layer_moves_its_input(cases, x))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "layer.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    for name, value in sorted(tensors.items()):
        peak = float(mx.abs(value).max())
        print(f"{name:<42}  {str(list(value.shape)):<16}  max |x| {peak:.4g}")
    print("\nthe cache moves the continuation by:")
    for name, gap in cached:
        print(f"  {name:<6}  {gap:.3e}")
    print("the layer moves its input by:")
    for name, gap in moved:
        print(f"  {name:<6}  {gap:.3e}")
    print(f"{len(tensors)} tensors, {bundle.stat().st_size / 1024:.1f} KiB -> {bundle}")


if __name__ == "__main__":
    main()
