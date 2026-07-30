"""Sweep prompt lengths through a single prefill forward pass and report wall
time, throughput and peak memory for each. The smoke prompt is 27 tokens, far
too short to say anything about prefill; these numbers decide whether attention
or the MoE path is the first thing worth writing in Rust.

Only the transformer stack is timed. The lm_head is skipped: a real decoder
projects logits for the last position alone, so including a full
[1, L, vocab] projection would measure work no engine performs."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from inkling_ref import gib, load_model, tokenizer

GIB = 1 << 30

# A two-point fit reads the quadratic term low, and the further the
# extrapolation reaches past the points it was fitted on, the lower it reads.
# Doubling the prompt each step keeps the error inside this margin; larger jumps
# do not, so keep the sweep dense rather than widening the margin.
PROJECTION_MARGIN = 1.25

FILLER = (
    "The lighthouse keeper counted the ships that passed the headland each "
    "evening, noting the weather, the tide, and whatever else seemed worth "
    "remembering by lamplight. "
)


def build_prompt(processor, n_tokens):
    ids = tokenizer(processor).encode(FILLER)
    tiled = np.tile(ids, -(-n_tokens // len(ids)))[:n_tokens]
    return mx.array(tiled)[None, :]


def projected_mask_bytes(model, n_tokens):
    """banded_additive_mask materialises [B, H, LQ, S] per layer, so a prefill
    pays the full square even on the 512-token sliding layers."""
    config = model.config.text_config
    heads = max(config.num_attention_heads, config.swa_num_attention_heads)
    return heads * n_tokens * n_tokens * 2


class BudgetExceeded(RuntimeError):
    pass


def projected_peak_bytes(measured, n_tokens, mask_bytes):
    """Extrapolate this prompt's peak from the deltas already measured. The
    attention path materialises several [H, L, L] intermediates, so peak grows
    quadratically and a two-point fit of a*L + b*L**2 over the measured deltas
    recovers the term that decides whether a longer prompt fits at all.

    With fewer than two points to fit, the mask alone is the only bound
    available; it holds for the shortest prompts of a sweep, which are the ones
    never in question."""
    resident = mx.get_active_memory()
    if len(measured) < 2:
        return resident + mask_bytes
    (l1, d1), (l2, d2) = measured[-2:]
    b = (d2 / l2 - d1 / l1) / (l2 - l1)
    a = d1 / l1 - b * l1
    delta = max(a * n_tokens + b * n_tokens**2, mask_bytes)
    return resident + PROJECTION_MARGIN * delta


def check_budget(measured, n_tokens, mask_bytes, budget_bytes):
    projected = projected_peak_bytes(measured, n_tokens, mask_bytes)
    if projected > budget_bytes:
        raise BudgetExceeded(
            f"projected peak {gib(projected):.0f} GiB over the "
            f"{gib(budget_bytes):.0f} GiB budget"
        )


def measure_prefill(model, prompt):
    cache = model.language_model.make_cache()
    mx.clear_cache()
    mx.reset_peak_memory()
    resident = mx.get_active_memory()

    t0 = time.perf_counter()
    out = model.language_model(
        inputs=prompt, cache=cache, return_hidden=True, skip_logits=True
    )
    mx.eval(out.hidden_states[-1])
    wall_s = time.perf_counter() - t0

    peak = mx.get_peak_memory()
    del out, cache
    mx.clear_cache()
    return {"wall_s": wall_s, "resident_bytes": resident, "peak_bytes": peak}


def format_row(row):
    if "error" in row:
        return f"{row['tokens']:>8}  {row['error']}"
    return (
        f"{row['tokens']:>8}  {row['wall_s']:>9.2f}  {row['tok_s']:>9.1f}  "
        f"{gib(row['peak_bytes']):>9.1f}  "
        f"{gib(row['peak_bytes'] - row['resident_bytes']):>9.1f}  "
        f"{gib(row['mask_bytes']):>9.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument(
        "--lengths", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768]
    )
    ap.add_argument(
        "--memory-limit-gib",
        type=float,
        default=350.0,
        help="peak-memory budget; a prompt whose projected peak exceeds it is "
        "refused unmeasured rather than risking the host",
    )
    ap.add_argument("--warmup-tokens", type=int, default=64)
    ap.add_argument(
        "--stop-after-seconds",
        type=float,
        default=900.0,
        help="skip the remaining longer prompts once one prefill exceeds this",
    )
    ap.add_argument("--json", help="write the results table here as they are measured")
    args = ap.parse_args()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)

    budget_bytes = int(args.memory_limit_gib * GIB)
    # mx.set_memory_limit is a guideline, not a cap: it raises only once the
    # system has no RAM or swap left, by which point the host is already
    # thrashing. It is kept as a last-ditch net; check_budget is the guard.
    mx.set_memory_limit(budget_bytes)

    t0 = time.perf_counter()
    model, processor = load_model(args.model)
    print(f"load            {time.perf_counter() - t0:.1f} s")
    print(f"resident        {gib(mx.get_active_memory()):.1f} GiB")

    measure_prefill(model, build_prompt(processor, args.warmup_tokens))

    header = f"{'tokens':>8}  {'wall s':>9}  {'tok/s':>9}  {'peak GiB':>9}  {'Δ GiB':>9}  {'mask GiB':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    measured = []
    budget_spent = False
    for n_tokens in sorted(args.lengths):
        row = {"tokens": n_tokens, "mask_bytes": projected_mask_bytes(model, n_tokens)}
        if budget_spent:
            row["error"] = (
                f"skipped after a prefill exceeded {args.stop_after_seconds:.0f} s"
            )
        else:
            try:
                check_budget(measured, n_tokens, row["mask_bytes"], budget_bytes)
                prompt = build_prompt(processor, n_tokens)
                mx.eval(prompt)
                row.update(measure_prefill(model, prompt))
                row["tok_s"] = n_tokens / row["wall_s"]
                measured.append((n_tokens, row["peak_bytes"] - row["resident_bytes"]))
                budget_spent = row["wall_s"] > args.stop_after_seconds
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                mx.clear_cache()
        rows.append(row)
        print(format_row(row), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
                f.write("\n")


if __name__ == "__main__":
    main()
