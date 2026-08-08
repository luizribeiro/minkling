"""Load Inkling-Small through the patched mlx-vlm and report load cost and
decode throughput. This is the baseline the Rust engine has to beat, and the
oracle its layer outputs get compared against."""

import argparse
import time

import mlx.core as mx
from inkling_ref import gib, load_model
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument(
        "--prompt", default="Explain what a mixture-of-experts layer does, briefly."
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, processor = load_model(args.model)
    load_s = time.perf_counter() - t0

    print(f"load            {load_s:.1f} s")
    print(f"resident        {gib(mx.get_active_memory()):.1f} GiB")
    print(f"peak            {gib(mx.get_peak_memory()):.1f} GiB")

    formatted = apply_chat_template(processor, model.config, args.prompt)

    t0 = time.perf_counter()
    result = generate(
        model, processor, formatted, max_tokens=args.max_tokens, verbose=False
    )
    gen_s = time.perf_counter() - t0

    print(f"wall            {gen_s:.1f} s")
    print(f"prompt          {result.prompt_tokens} tok @ {result.prompt_tps:.1f} tok/s")
    print(
        f"decode          {result.generation_tokens} tok @ "
        f"{result.generation_tps:.1f} tok/s"
    )
    print(f"finish          {result.finish_reason}")
    print(f"peak after gen  {gib(mx.get_peak_memory()):.1f} GiB")
    print("-" * 60)
    print(result.text)


if __name__ == "__main__":
    main()
