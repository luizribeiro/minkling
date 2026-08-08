"""Dump the routed bank's run lengths as the model itself routes them, at each
prompt length this repo's block tables are read at.

**A grouped call's cost is decided by where its run boundaries fall against a
block, and until now every table on that kernel invented the runs.** The Rust
side's fixture lays 256 experts over the rows as `row * experts / rows`, which
gives every expert the same run and puts every boundary on a block edge at any
length that divides; `qmm_probe.py` built the same thing with `arange(rows) *
experts // rows`. Both are one layout in eight and it is the one no router
produces — the milestone before this found the flat 79% column that layout
produced for `gather_qmm`, and the 13 points our own block appeared to gain
between 2048 and 8192, were both that divisibility rather than anything about a
length.

So this asks the model. `moe_tap` is patched into `InklingSparseMoE` for the
numerical oracle and hands out `topk_idx` — the experts the top-k actually
chose — and one prefill of `n` tokens through the stack produces one selection
per MoE layer. What is written out is the count per expert, per layer, per
length: the run lengths a sort would cut, with no construction of ours between
the router and the number.

**Every MoE layer and not one.** The routing is a property of the layer as much
as of the text — a layer whose gate is flatter spreads its rows further — and a
table quoting one layer would be quoting a sample of forty. The Rust side reads
the layer whose counts are the median spread and says so.

The prompt is `bench_common.prompt_ids`, which is `inkling_core::workload`'s own
tiled to length: the same tokens the engine's own prefill is measured over, so
the runs recorded here are the runs those measurements ran against.
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from bench_common import prompt_ids
from inkling_ref import load_model
from mlx_vlm.models.inkling import language

# The lengths the block's tables are read at: a coding session's per-turn delta
# prefill at the short end, and the two the engine's long-prefill rows are
# quoted at above them.
LENGTHS = [321, 512, 769, 1024, 2048, 4096, 8192, 16384]


def routing_at(model, prompt):
    """One prefill's selection, as counts per expert per MoE layer.

    The tap fires once per MoE layer per forward pass, in layer order, so the
    list this returns is indexed the way the stack is. `topk_idx` is
    `[batch, tokens, top_k]` and every one of its entries is a row the routed
    bank will run.
    """
    seen = []

    def tap(_module, values):
        idx = np.array(values["topk_idx"], copy=False).reshape(-1)
        seen.append(
            np.bincount(idx, minlength=model.config.text_config.n_routed_experts)
        )

    language.moe_tap = tap
    try:
        cache = model.language_model.make_cache()
        out = model.language_model(
            inputs=prompt, cache=cache, return_hidden=True, skip_logits=True
        )
        mx.eval(out.hidden_states[-1])
        del out, cache
    finally:
        language.moe_tap = None
    mx.clear_cache()
    return [counts.tolist() for counts in seen]


def spread(counts):
    """What a table wants to say about one layer's runs in three numbers: how
    far the busiest expert is above the mean, how many get nothing, and the
    blocks a 32-row cut would make of it against the rows it holds."""
    counts = np.array(counts)
    rows = int(counts.sum())
    mean = rows / len(counts)
    blocks = int(np.ceil(counts[counts > 0] / 32).sum())
    return {
        "rows": rows,
        "hottest": int(counts.max()),
        "hottest_over_mean": float(counts.max() / mean),
        "empty": int((counts == 0).sum()),
        "blocks_at_32": blocks,
        "waste_at_32": float(blocks / (rows / 32) - 1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=LENGTHS,
        help="prompt lengths to route, defaulting to the block tables' own",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "fixtures"),
        help="where the bundle is written",
    )
    args = ap.parse_args()

    model, processor = load_model(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "routing.safetensors"

    tensors = {}
    for tokens in args.tokens:
        layers = routing_at(model, prompt_ids(processor, tokens))
        # Float32 like every other fixture here, and exact: the largest count a
        # 16384-token prompt produces is about 14000, decades under the 2^24 an
        # f32 holds integers to.
        tensors[f"routing_{tokens}"] = mx.array(layers, dtype=mx.float32)
        waste = [spread(counts)["waste_at_32"] for counts in layers]
        hot = [spread(counts)["hottest_over_mean"] for counts in layers]
        print(
            f"{tokens:>6} tokens  {len(layers)} MoE layers  "
            f"hottest/mean {min(hot):.2f}-{max(hot):.2f}  "
            f"part-empty blocks at 32 rows {1e2 * min(waste):.1f}-{1e2 * max(waste):.1f}%",
            flush=True,
        )
        # Written as each length lands rather than at the end: the longest
        # prompts are minutes apiece and a sweep that fell over on the last of
        # them would otherwise have nothing to show for the ones before it.
        mx.save_safetensors(str(bundle), tensors)

    print(f"\nwrote {bundle}")
    for name, value in sorted(tensors.items()):
        print(f"  {name:<20}  {list(value.shape)}")


if __name__ == "__main__":
    main()
