"""Dump synthetic sequences through mlx-vlm's own `InklingAttention`, as the
oracle the whole Rust attention layer is tested against.

Inkling's attention projections are `[4096, 4096]` — 67 MB each in float32 — so
like the RMSNorm and MLP fixture this one cannot be cut out of the checkpoint.
It stands the reference module up on seeded micro-weights instead: four query
heads over two KV heads, eight channels each. What it pins is not Inkling's
trained numbers but mlx-vlm's arithmetic, in float32 throughout, where a
comparison measures the port rather than bfloat16 rounding.

Log scaling is the reason this fixture has to exist at all. It fires only on
global layers and only past `log_scaling_n_floor`, which the checkpoint sets to
128000: `tau` is exactly 1 at every position any recorded forward pass reaches,
so the branch is dead in the committed activations and a port could leave it
unwritten, or write it wrong, and pass everything. Here the floor is small
enough that positions cross it. Two global cases share one set of weights and
differ only in that floor — one below it end to end, one above it — so their
outputs differ by log scaling and nothing else, and the difference is what a
port has to reproduce.

Each case is run twice against one cache, a prefill and then a shorter
continuation, because half of what log scaling does is read the KV cache's
offset. So do the short convolutions and the mask; a fixture that only ever
prefilled would let all three drop it.

Nothing here reimplements an op. Every output comes from calling
`InklingAttention` with the caches `LanguageModel.make_cache` allocates."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import f32, gamma, projection, taps
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import InklingAttention

SEED = 20260802

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
D_REL = 3
SCONV_KERNEL = 4

# Narrow enough that a prefill of eight tokens reaches past it, so the window
# cap runs. A global layer's band is wider than that but still inside the span,
# so the outside-the-band case runs too.
SLIDING_WINDOW = 4
REL_EXTENT = 6

# The checkpoint's own alpha. A larger one would make log scaling easier to see
# and would be testing a model that does not exist.
LOG_ALPHA = 0.1

# Above every position these cases reach, and below almost all of them. The
# checkpoint's floor is 128000, which no committed capture comes near.
INERT_FLOOR = 4096
SCALED_FLOOR = 4

PREFILL = 8
CONTINUE = 3

SLIDING_LAYER = 0
GLOBAL_LAYER = 1

# Each case's layer and the floor its config carries. `sliding` takes no floor
# at all: `InklingAttention` sets `log_floor` to None on a sliding layer
# whatever the config says, which is itself worth pinning.
CASES = {
    "sliding": (SLIDING_LAYER, SCALED_FLOOR),
    "global_inert": (GLOBAL_LAYER, INERT_FLOOR),
    "global_scaled": (GLOBAL_LAYER, SCALED_FLOOR),
}

# The two cases that share weights, and so differ only by their floor.
LOG_SCALING_PAIR = ("global_inert", "global_scaled")

# Where the floor sits in the config tensor `config_tensor` writes.
LOG_FLOOR_FIELD = 7


def config(log_floor):
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
        local_layer_ids=[SLIDING_LAYER],
        d_rel=D_REL,
        rel_extent=REL_EXTENT,
        sconv_kernel_size=SCONV_KERNEL,
        log_scaling_n_floor=log_floor,
        log_scaling_alpha=LOG_ALPHA,
    )


def parameters(rng, rel_extent):
    """Every tensor `InklingAttention` holds, drawn once per layer kind.

    `rel_proj` is contracted over `d_rel` rather than over its own width, so it
    is scaled by that; `nn.Linear` leaves it zeroed, which would make the mask a
    plain causal one and half of this fixture inert."""
    kv_width = KV_HEADS * HEAD_DIM
    return {
        "q_proj.weight": projection(rng, HEADS * HEAD_DIM, HIDDEN),
        "k_proj.weight": projection(rng, kv_width, HIDDEN),
        "v_proj.weight": projection(rng, kv_width, HIDDEN),
        "r_proj.weight": projection(rng, HEADS * D_REL, HIDDEN),
        "o_proj.weight": projection(rng, HIDDEN, HEADS * HEAD_DIM),
        "q_norm.weight": gamma(rng, HEAD_DIM),
        "k_norm.weight": gamma(rng, HEAD_DIM),
        "k_sconv.conv.weight": taps(rng, kv_width, SCONV_KERNEL),
        "v_sconv.conv.weight": taps(rng, kv_width, SCONV_KERNEL),
        "rel_proj": f32(rng.standard_normal((D_REL, rel_extent)) / np.sqrt(D_REL)),
    }


def attention(log_floor, layer, weights):
    attn = InklingAttention(config(log_floor), layer)
    attn.load_weights(list(weights.items()))
    return attn


def config_tensor(attn):
    """The scalars `InklingAttention.__init__` derives from the config and the
    layer index, which the recorded shapes do not carry. A `log_floor` of zero
    is the None a sliding layer gets."""
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


def run(attn, x, x_continue):
    """A prefill and a continuation against one cache, as decoding runs it."""
    cache = CacheList(KVCache(), ArraysCache(4))
    return attn(x, cache=cache), attn(x_continue, cache=cache)


def synthetic_cases(rng, x, x_continue):
    weights = {
        "sliding": parameters(rng, SLIDING_WINDOW),
        "global": parameters(rng, REL_EXTENT),
    }
    tensors = {
        f"{kind}.{name}": value
        for kind, params in weights.items()
        for name, value in params.items()
    }

    for case, (layer, floor) in CASES.items():
        kind = "sliding" if layer == SLIDING_LAYER else "global"
        attn = attention(floor, layer, weights[kind])
        tensors[f"{case}.config"] = config_tensor(attn)
        prefill, rest = run(attn, x, x_continue)
        tensors[f"{case}.prefill_out"] = prefill
        tensors[f"{case}.continue_out"] = rest
    return tensors


def tau(spec, offset, length):
    """`1 + alpha * log(max(qpos / floor, 1))` over one call's query positions,
    written out of the config alone so the coverage checks below agree with
    `InklingAttention` only where it is right. It decides no recorded value."""
    _, floor = spec
    qpos = np.arange(length) + offset + 1
    return 1.0 + LOG_ALPHA * np.log(np.maximum(qpos / floor, 1.0))


def check_log_scaling_coverage():
    """The one branch the committed activations cannot reach. If the floors
    stopped straddling the positions these calls run over, the pair below would
    still agree and would be agreeing about nothing."""
    inert, scaled = (CASES[case] for case in LOG_SCALING_PAIR)
    calls = ((0, PREFILL), (PREFILL, CONTINUE))
    if any((tau(inert, *call) != 1.0).any() for call in calls):
        raise SystemExit(
            f"{LOG_SCALING_PAIR[0]}: its floor is not above every position"
        )
    if not (tau(scaled, 0, PREFILL) == 1.0).any():
        raise SystemExit(
            f"{LOG_SCALING_PAIR[1]}: no prefill position is below its floor"
        )
    if (tau(scaled, PREFILL, CONTINUE) <= 1.0).any():
        raise SystemExit(
            f"{LOG_SCALING_PAIR[1]}: the continuation does not clear its floor"
        )


def check_log_scaling_is_visible(tensors):
    """The pair shares its weights and its input, so anything they disagree
    about is log scaling. A disagreement lost in the noise would leave a port
    free to drop the branch."""
    for field in ("prefill_out", "continue_out"):
        inert, scaled = (
            np.asarray(tensors[f"{case}.{field}"]) for case in LOG_SCALING_PAIR
        )
        gap = np.abs(inert - scaled).max() / np.abs(inert).max()
        if gap < 1e-2:
            raise SystemExit(f"{field}: log scaling moves the answer by only {gap:.3e}")
        yield field, gap


def check_mask_branches():
    """The two branches a short prefill at offset zero cannot reach on its own:
    a key older than the window, and a key in context but outside the band."""
    distance = PREFILL + CONTINUE - 1
    if distance < SLIDING_WINDOW:
        raise SystemExit("no key is old enough for the sliding window to cap")
    if distance < REL_EXTENT:
        raise SystemExit("no key falls outside the global band")


def check_the_pair_differs_only_in_its_floor(tensors):
    """`global_inert` and `global_scaled` are one weight set and one input under
    two floors, so anything they disagree about is log scaling. A second
    difference between them would be a second explanation."""
    inert, scaled = (np.asarray(tensors[f"{case}.config"]) for case in LOG_SCALING_PAIR)
    rest = [at for at in range(len(inert)) if at != LOG_FLOOR_FIELD]
    if inert[LOG_FLOOR_FIELD] == scaled[LOG_FLOOR_FIELD]:
        raise SystemExit(f"{LOG_SCALING_PAIR} share a floor")
    if not (inert[rest] == scaled[rest]).all():
        raise SystemExit(f"{LOG_SCALING_PAIR} differ in more than their floor")


def collect():
    rng = np.random.default_rng(SEED)
    x = f32(rng.standard_normal((1, PREFILL, HIDDEN)))
    x_continue = f32(rng.standard_normal((1, CONTINUE, HIDDEN)))

    tensors = {"x": x, "continue_x": x_continue}
    tensors.update(synthetic_cases(rng, x, x_continue))
    mx.eval(list(tensors.values()))
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    check_log_scaling_coverage()
    check_mask_branches()
    tensors = collect()
    check_the_pair_differs_only_in_its_floor(tensors)
    gaps = list(check_log_scaling_is_visible(tensors))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "attention.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    for name, value in tensors.items():
        peak = float(mx.abs(value).max())
        print(f"{name:<30}  {str(list(value.shape)):<14}  max |x| {peak:.4g}")
    print("\nlog scaling moves:")
    for field, gap in gaps:
        print(f"  {field:<14}  {gap:.3e}")
    print(f"{len(tensors)} tensors, {bundle.stat().st_size / 1024:.1f} KiB -> {bundle}")


if __name__ == "__main__":
    main()
