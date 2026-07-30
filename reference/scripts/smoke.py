"""Load Inkling-Small through the patched mlx-vlm and report load cost and
decode throughput. This is the baseline the Rust engine has to beat, and the
oracle its layer outputs get compared against."""

import argparse
import time

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template


def gib(n_bytes):
    return n_bytes / (1 << 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument(
        "--prompt", default="Explain what a mixture-of-experts layer does, briefly."
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, processor = load(args.model)
    mx.eval(model.parameters())
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

    text = getattr(result, "text", str(result))
    n = getattr(result, "generation_tokens", None) or len(
        processor.tokenizer.encode(text)
    )

    print(f"generated       {n} tokens in {gen_s:.1f} s  ({n / gen_s:.1f} tok/s)")
    print(f"peak after gen  {gib(mx.get_peak_memory()):.1f} GiB")
    print("-" * 60)
    print(text)


if __name__ == "__main__":
    main()
