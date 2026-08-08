"""Dump the sigmoid-gated router — the trained gate of the captured MoE layer,
and synthetic sequences through mlx-vlm's own `InklingSparseMoE` — as the oracle
the Rust mixture-of-experts layer is tested against.

The routed experts are `[256, 2048, 4096]` and the shared pair `[2, 2048, 4096]`,
about 25 GB in float32, so no fixture can carry them and the trained expert
arithmetic is left to the checkpoint-gated tests. What can be committed is the
gate — `[258, 4096]` plus a bias and two scalars, under 5 MB — and the gate is
where every way to get this layer wrong lives.

There are three of them, and each leaves a model that runs and generates:

- The correction bias picks the experts and does not weight them. `sigmoid(logits)
  + e_score_correction_bias` decides the top-k; the weights are a softmax over
  the *raw* logits of the experts it chose. Adding the bias twice moves the
  weights by a fifth, carrying the biased score into them by a half.
- The weights are one softmax over the top-k routed logits and both shared
  logits together — eight numbers at Inkling-Small's shape. Normalising the six
  alone leaves the shared experts three times too heavy.
- Two scale factors multiply the result: `route_scale` from the config, and a
  learned `global_scale` from the checkpoint. Layer 2's `global_scale` is
  0.00704, so a port that applies `route_scale` alone runs 142x hot.

The trained gate is recorded in the checkpoint's own dtypes rather than widened
to float32 like the activation captures: it is a weight and not an answer, so
reading it runs the same widening the engine will, and bfloat16 halves what has
to be committed.

The synthetic cases stand the reference module up on seeded micro-weights, in
float32 throughout, where a comparison measures the port rather than bfloat16
rounding. They also reach the one thing no trained capture does: a genuine tie
in the biased score, which is where `mx.argpartition` promises nothing.

Nothing here reimplements the router. Every recorded value comes from calling
`InklingSparseMoE` through the `moe_tap` the reference is patched with."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import checkpoint_tensor, f32, projection
from mlx_vlm.models.inkling import language
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import InklingSparseMoE

# The layer the committed activations captured a router for, and the bundle
# holding that capture. This fixture is read beside it.
ACTIVATIONS = "layer_activations.safetensors"
MOE_LAYER = 2
GATE = f"language_model.model.layers.{MOE_LAYER}.mlp"

SEED = 20260803

HIDDEN = 32
MOE_INTERMEDIATE = 16
N_ROUTED = 16
N_SHARED = 2
TOP_K = 3
TOKENS = 8

# The checkpoint's own route_scale. The global scale is not the checkpoint's —
# 0.007 would put every recorded output two decades below its input — but it is
# far enough from one, and from 1/route_scale, that dropping either factor shows.
ROUTE_SCALE = 8.0
GLOBAL_SCALE = 0.35

# The two routed experts the tie case gives one gate row and one bias, so their
# biased scores agree bit for bit. Their ranks straddle the top-k boundary, so
# exactly one of them is selected and which one is the whole question.
TIED = (2, 9)

# What the tap hands back, all of which every case records.
TAPPED = (
    "gate_logits",
    "gate_scores_biased",
    "topk_idx",
    "topk_w",
    "shared_gammas",
    "routed_out",
    "shared_out",
)


def config():
    return TextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=N_ROUTED,
        num_experts_per_tok=TOP_K,
        n_shared_experts=N_SHARED,
        route_scale=ROUTE_SCALE,
    )


def expert_stack(rng, count, out_dim, in_dim):
    """A `SwitchLinear` weight, `[experts, out, in]`: one `nn.Linear` per expert,
    drawn the way every other synthetic projection here is."""
    return mx.stack([projection(rng, out_dim, in_dim) for _ in range(count)])


def bank(rng, count):
    """The three projections one `SwitchGLU` holds."""
    return {
        f"{name}.weight": expert_stack(rng, count, out_dim, in_dim)
        for name, (out_dim, in_dim) in (
            ("gate_proj", (MOE_INTERMEDIATE, HIDDEN)),
            ("up_proj", (MOE_INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, MOE_INTERMEDIATE)),
        )
    }


def parameters(rng, gate_weight, correction_bias):
    """Every tensor `InklingSparseMoE` holds. The gate and the bias are handed in
    because the tie case builds them rather than drawing them."""
    return {
        "gate_weight": gate_weight,
        "e_score_correction_bias": correction_bias,
        "global_scale": f32([GLOBAL_SCALE]),
        **{f"switch_mlp.{name}": w for name, w in bank(rng, N_ROUTED).items()},
        **{f"shared_experts.{name}": w for name, w in bank(rng, N_SHARED).items()},
    }


def spread_parameters(rng):
    """A bias over the range the trained one spans, 0.05 to 0.77 — comparable to
    the spread of `sigmoid`, which is what makes it move the selection rather
    than decorate it."""
    gate = projection(rng, N_ROUTED + N_SHARED, HIDDEN)
    return parameters(rng, gate, f32(rng.uniform(0.05, 0.8, N_ROUTED)))


def tie_parameters(rng):
    """A gate whose biased scores tie at the top-k boundary.

    The bias descends by a whole unit per expert and `sigmoid` spans less than
    one, so the ranking is the bias's and is the same for every token: experts 0
    and 1 take the first two slots and the tied pair contests the third. Giving
    the pair one gate row makes their logits, and so their scores, identical bit
    for bit rather than merely close."""
    ramp = np.arange(N_ROUTED, 0, -1, dtype=np.float64)
    lo, hi = TIED
    ramp[hi] = ramp[lo]

    gate = np.asarray(projection(rng, N_ROUTED + N_SHARED, HIDDEN))
    gate[hi] = gate[lo]
    return parameters(rng, f32(gate), f32(ramp))


def run(params, x):
    """One case's forward pass, and everything the tap saw on the way."""
    moe = InklingSparseMoE(config())
    moe.load_weights(list(params.items()))

    captured = {}
    language.moe_tap = lambda _module, tensors: captured.update(tensors)
    try:
        out = moe(x)
    finally:
        language.moe_tap = None
    return out, captured


def case(params, x):
    out, captured = run(params, x)
    tapped = {name: captured[name] for name in TAPPED}
    # Selected experts land as int32, the dtype `dump_activations` records them
    # in, so one reader serves both bundles.
    tapped["topk_idx"] = tapped["topk_idx"].astype(mx.int32)
    return {
        **params,
        "config": f32([N_ROUTED, N_SHARED, TOP_K, ROUTE_SCALE]),
        **tapped,
        "out": out,
    }


def trained_config(model_path):
    text = json.loads((model_path / "config.json").read_text())["text_config"]
    return [
        text["n_routed_experts"],
        text["n_shared_experts"],
        text["num_experts_per_tok"],
        text["route_scale"],
    ]


def trained_gate(model_path):
    """The captured layer's router, straight out of the shard that owns it. The
    expert banks beside it in that layer are a thousand times larger and stay
    where they are."""
    return {
        "gate_weight": checkpoint_tensor(model_path, f"{GATE}.gate_weight"),
        "e_score_correction_bias": checkpoint_tensor(
            model_path, f"{GATE}.e_score_correction_bias"
        ),
        "global_scale": checkpoint_tensor(model_path, f"{GATE}.global_scale"),
        "config": f32(trained_config(model_path)),
        "layer": f32([MOE_LAYER]),
    }


def biased_scores(logits, bias, n_routed):
    return 1.0 / (1.0 + np.exp(-logits[:, :n_routed])) + bias


def selected(scores, top_k):
    """The set `mx.argpartition` picks, as a stable descending sort picks it, in
    ascending expert order. Used by the checks below; no recorded value goes
    through it."""
    return np.sort(np.argsort(-scores, axis=-1, kind="stable")[:, :top_k], axis=-1)


def check_the_bias_moves_the_selection(tensors):
    """The correction bias is a selection-only term, which is a claim worth
    making only if it selects something else. A bias too small to reorder any
    token's top-k would leave the first trap untested."""
    logits = np.asarray(tensors["gate_logits"])
    bias = np.asarray(tensors["e_score_correction_bias"])
    with_bias = selected(biased_scores(logits, bias, N_ROUTED), TOP_K)
    without = selected(biased_scores(logits, 0.0, N_ROUTED), TOP_K)
    moved = int((with_bias != without).any(axis=-1).sum())
    if moved == 0:
        raise SystemExit("the correction bias reorders no token's top-k")
    return moved


def check_the_weights_carry_both_scales(tensors):
    """A joint softmax sums to one over the routed and the shared weights
    together, so every row of the concatenation sums to `route_scale *
    global_scale`. That sum is the signature of the joint normalisation and of
    the second scale at once."""
    rows = np.concatenate(
        [np.asarray(tensors["topk_w"]), np.asarray(tensors["shared_gammas"])], axis=-1
    ).sum(axis=-1)
    want = ROUTE_SCALE * GLOBAL_SCALE
    if not np.allclose(rows, want, rtol=1e-6):
        raise SystemExit(f"weight rows sum to {rows}, not {want}")


def check_the_tie_is_exact_and_at_the_boundary(tensors):
    """A tie that were merely close would be settled by whichever expert rounded
    higher, and a tie away from the boundary by taking both."""
    scores = np.asarray(tensors["gate_scores_biased"])
    lo, hi = TIED
    if not (scores[:, lo] == scores[:, hi]).all():
        raise SystemExit(f"experts {TIED} do not tie: {scores[:, lo] - scores[:, hi]}")

    above = (scores > scores[:, [lo]]).sum(axis=-1)
    if not (above == TOP_K - 1).all():
        raise SystemExit(f"the tie is not at slot {TOP_K}: {above} experts score above")

    selected = {int(e) for row in np.asarray(tensors["topk_idx"]) for e in row}
    picked = selected & set(TIED)
    if len(picked) != 1:
        raise SystemExit(f"the tie was not broken: experts {sorted(picked)} selected")
    return picked.pop()


def check_the_order_is_the_kernels_and_not_the_references(scores, top_k):
    """`mx.argpartition` promises only that the k-th element lands where a sort
    would put it; the k before it are a set, in no stated order.

    Two MLX streams do return them in two different orders for one input, so the
    order the recorded `topk_idx` carries belongs to the kernel that ran and not
    to the reference. What the streams agree on is the *set*, which is all a port
    can be held to — and all the routed sum depends on, each expert carrying its
    own weight into it.

    Raised rather than printed: if the two ever agree, this claim is no longer
    established and the recorded order stops being an artefact."""
    picks = {
        name: np.asarray(
            mx.argpartition(-mx.array(scores), top_k - 1, axis=-1, stream=stream)[
                :, :top_k
            ]
        )
        for name, stream in (("gpu", mx.gpu), ("cpu", mx.cpu))
    }
    if not (np.sort(picks["gpu"], -1) == np.sort(picks["cpu"], -1)).all():
        raise SystemExit("the two streams disagree on which experts are selected")
    if (picks["gpu"] == picks["cpu"]).all():
        raise SystemExit("the two streams now agree on the order; recheck the claim")


def check_the_trained_gate_reproduces_the_recorded_selection(gate, recorded, top_k):
    """The committed gate is float32 and the recorded logits were computed in
    bfloat16, so a port that recomputes them parts company with the capture by
    about a thousandth. Selection survives that only while the gap between the
    k-th and the (k+1)-th score stays well clear of it — a property of these
    eight tokens rather than a guarantee, and one a regenerated capture could
    quietly lose."""
    weight = np.asarray(gate["gate_weight"].astype(mx.float32))
    bias = np.asarray(gate["e_score_correction_bias"].astype(mx.float32))
    n_routed = int(np.asarray(gate["config"])[0])

    x = np.asarray(recorded["post_attention_ln_out"]).reshape(-1, weight.shape[-1])
    recomputed = biased_scores(x @ weight.T, bias, n_routed)
    captured = np.asarray(recorded["gate_scores_biased"])

    want = np.sort(np.asarray(recorded["topk_idx"]), axis=-1)
    if not (selected(recomputed, top_k) == want).all():
        raise SystemExit("the committed gate selects different experts")

    ranked = np.sort(captured, axis=-1)[:, ::-1]
    margin = float((ranked[:, top_k - 1] - ranked[:, top_k]).min())
    drift = float(np.abs(recomputed - captured).max())
    if margin < 4 * drift:
        raise SystemExit(f"selection margin {margin:.3e} is not clear of {drift:.3e}")
    return margin, drift


def recorded_layer(out_dir):
    """The MoE layer's tensors out of the committed activations, unprefixed."""
    bundle = mx.load(str(Path(out_dir) / ACTIVATIONS))
    prefix = f"layer{MOE_LAYER}."
    return {
        name[len(prefix) :]: value
        for name, value in bundle.items()
        if name.startswith(prefix)
    }


def collect(model_path):
    rng = np.random.default_rng(SEED)
    x = f32(rng.standard_normal((1, TOKENS, HIDDEN)))

    cases = {
        "main": case(spread_parameters(rng), x),
        "tie": case(tie_parameters(rng), x),
    }
    gate = trained_gate(model_path)

    tensors = {"x": x}
    for name, values in cases.items():
        tensors.update({f"{name}.{field}": v for field, v in values.items()})
    tensors.update({f"trained.{field}": v for field, v in gate.items()})
    mx.eval(list(tensors.values()))
    return tensors, cases, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--out-dir", default="reference/fixtures")
    args = ap.parse_args()

    tensors, cases, gate = collect(Path(args.model))
    recorded = recorded_layer(args.out_dir)
    trained_top_k = int(np.asarray(gate["config"])[2])

    moved = check_the_bias_moves_the_selection(cases["main"])
    check_the_weights_carry_both_scales(cases["main"])
    winner = check_the_tie_is_exact_and_at_the_boundary(cases["tie"])
    check_the_order_is_the_kernels_and_not_the_references(
        np.asarray(recorded["gate_scores_biased"]), trained_top_k
    )
    margin, drift = check_the_trained_gate_reproduces_the_recorded_selection(
        gate, recorded, trained_top_k
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "moe.safetensors"
    mx.save_safetensors(str(bundle), tensors)

    for name, value in sorted(tensors.items()):
        dtype = str(value.dtype).rsplit(".", 1)[-1]
        print(f"{name:<38}  {str(list(value.shape)):<16}  {dtype}")
    print(f"\nthe correction bias reorders {moved} of {TOKENS} tokens' top-{TOP_K}")
    print(f"the tie at slot {TOP_K} went to expert {winner} of {TIED}")
    print(f"trained margin {margin:.3e} against a float32 drift of {drift:.3e}")
    print(
        f"{len(tensors)} tensors, {bundle.stat().st_size / (1 << 20):.2f} MiB -> {bundle}"
    )


if __name__ == "__main__":
    main()
