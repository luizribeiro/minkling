"""Measure what Inkling-Small's 8 multi-token-prediction heads would accept
under self-speculative decoding.

mlx-vlm drops every `model.mtp.*` tensor at load, so nothing in the reference
has ever run these heads. This stands them up on the existing decoder modules
and asks the one question the engine's design rests on: how far ahead can the
heads guess before they stop agreeing with what the model actually decodes.

## Replay is teacher-forced, and that is exact

A real self-speculative round chains the heads on their own guesses: head d is
fed the token head d-1 proposed. Replaying that faithfully would mean a
sequential forward per (head, position). It is not necessary.

Head d's proposal is only ever *used* when heads 0..d-1 were all accepted, and
on that event the chain fed head d the true tokens — so the speculative
trajectory and the teacher-forced one coincide, hidden states included. The
same holds for the attention history: a correct engine replays rejected
positions against the accepted tokens, so a head's cache always holds the
teacher-forced trajectory. Feeding each head the true tokens therefore recovers
the joint acceptance exactly, at one full-sequence forward per head instead of
one per position.

What that does *not* give is the marginal quality of head d on its own after an
earlier head has already gone wrong. That number does not enter the speedup, so
it is not measured.

## What is and is not being measured

Greedy-decoding agreement: the fraction of steps where a head's argmax equals
the argmax the model itself produced. That is an upper bound on a real
accept-reject scheme under sampling, which must also agree on the *distribution*
and rejects on a random draw. Read the numbers here as a ceiling.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from inkling_ref import gib, load_model, tokenizer
from mlx_vlm.generate import wired_limit
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from mlx_vlm.models.inkling.language import InklingDecoderLayer
from mlx_vlm.prompt_utils import apply_chat_template

GIB = 1 << 30

# Set from --timing. Every model-touching run here has to stay inside a command
# timeout, and a stage that silently costs minutes is what blows one.
TIMING = False

# Regimes acceptance is known to differ wildly across. A single prompt would put
# the decay curve anywhere between these, so the spread is part of the result.
PROMPTS = {
    "prose": "Write three paragraphs of narrative prose about a lighthouse "
    "keeper who has begun to distrust the log he keeps.",
    "code": "Write a Python function that parses an ISO-8601 duration string "
    "into a number of seconds. Include a docstring and handle the fractional "
    "seconds case.",
    "json": "Output a JSON array of 12 objects, each with the fields id, name, "
    "city, country and population. No commentary, just the JSON.",
    "enumeration": "Count from 1 to 60. Put each on its own line in exactly "
    "the form 'Line N: N squared is M'. No commentary.",
    "technical": "Explain how a mixture-of-experts layer routes tokens, what "
    "the router's top-k selection costs, and why shared experts are used.",
    "table": "Produce a markdown table of the 15 largest countries by land "
    "area, with columns for rank, country, area in km2, and continent.",
}

# The template opens the model's turn with a content-type marker, so the first
# token generated is itself a special token and "stop on any special token"
# stops immediately. Only these two end a turn.
TURN_END_TOKENS = ("<|end_message|>", "<|content_model_end_sampling|>")

# Inkling reasons by default, which would measure acceptance over chain of
# thought rather than over the regimes the prompts were chosen for.
REASONING_EFFORT = "none"

# Ambiguities the tensor names and shapes cannot settle: input_proj is
# [4096, 8192], which fixes that it consumes two normed 4096-wide vectors but
# not their order; and the checkpoint's shared embedding is already normed by
# the main stack's embed_norm, so whether the head's own embed_norm stacks on
# top of that or replaces it is undetermined. Resolved by measurement.
VARIANTS = [
    {"hidden_first": h, "embed_prenormed": e, "hidden_post_norm": n}
    for h in (True, False)
    for e in (False, True)
    for n in (False, True)
]


class InklingMTPLayer(nn.Module):
    """One next-n-token head: norm the incoming hidden state and the embedding
    of the token one position further ahead, project the pair back to model
    width, and run a decoder layer over it."""

    def __init__(self, config, layer_idx):
        super().__init__()
        hidden = config.hidden_size
        self.embed_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.transformer_block = InklingDecoderLayer(config, layer_idx)

    def __call__(self, hidden, embed, cache=None, hidden_first=True):
        parts = [self.hidden_norm(hidden), self.embed_norm(embed)]
        if not hidden_first:
            parts.reverse()
        x = self.input_proj(mx.concatenate(parts, axis=-1))
        return self.transformer_block(x, cache=cache)


class InklingMTPStack(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.layers = [InklingMTPLayer(config, i) for i in range(num_layers)]

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(4)) for _ in self.layers]


def mtp_text_config(text_config, mtp_config):
    """The heads are decoder layers indexed 0..7 in their own space: they carry
    their own local/global split, and every one of them is dense. A shallow copy
    keeps the field rebinding `TextConfig.__post_init__` already did."""
    config = copy.copy(text_config)
    config.local_layer_ids = mtp_config["local_layer_ids"]
    config.layer_types = None
    config.mlp_layer_types = None
    config.dense_mlp_idx = mtp_config["num_nextn_predict_layers"]
    return config


def mtp_weights(model, raw):
    """The 8-bit quantiser kept the MTP tensors under their upstream names while
    every other tensor was rewritten to mlx-vlm's. `_map_llm_layer` is the
    rename the loader applies to a decoder layer, and the head's
    `transformer_block` is one, so it is reused rather than restated."""
    out = {}
    for key, value in raw.items():
        index, rest = key[len("model.mtp.layers.") :].split(".", 1)
        base = f"layers.{index}."
        block = "transformer_block."
        if rest.startswith(block):
            out.update(model._map_llm_layer(base + block, rest[len(block) :], value))
        else:
            out[base + rest] = value
    return out


def load_mtp(model, model_path, config_json):
    mtp_config = config_json["mtp_config"]
    config = mtp_text_config(model.config.text_config, mtp_config)
    mtp = InklingMTPStack(config, mtp_config["num_nextn_predict_layers"])
    raw = mx.load(str(Path(model_path) / "mtp.safetensors"))
    mtp.load_weights(list(mtp_weights(model, raw).items()))
    mx.eval(mtp.parameters())
    return mtp


def stop_ids(model, processor):
    tok = tokenizer(processor)
    ids = {model.config.eos_token_id}
    for name in TURN_END_TOKENS:
        token_id = tok.convert_tokens_to_ids(name)
        if token_id is not None:
            ids.add(token_id)
    return ids


def encode_prompt(model, processor, text):
    formatted = apply_chat_template(
        processor, model.config, text, reasoning_effort=REASONING_EFFORT
    )
    return tokenizer(processor).encode(formatted, add_special_tokens=False)


def greedy_decode(model, prompt_ids, max_tokens, eos_ids):
    """Decode greedily, keeping the pre-final-norm hidden state at every
    position. Position i's hidden is what predicted token i+1 and what the MTP
    heads hang off, so the prompt's are kept too — a head's attention has to see
    the prompt for its cache to mean anything."""
    lm = model.language_model
    cache = lm.make_cache()
    out = lm(inputs=prompt_ids, cache=cache, return_hidden=True, skip_logits=True)
    hiddens = [out.hidden_states[-1]]
    current = hiddens[0][:, -1:, :]

    tokens = []
    for _ in range(max_tokens):
        nxt = lm.speculative_argmax_from_hidden(current)
        mx.eval(nxt)
        token = int(nxt[0, 0].item())
        tokens.append(token)
        if token in eos_ids or len(tokens) == max_tokens:
            break
        out = lm(inputs=nxt, cache=cache, return_hidden=True, skip_logits=True)
        current = out.hidden_states[-1]
        hiddens.append(current)

    return tokens, mx.concatenate(hiddens, axis=1)


def head_embedding(lm, ids, embed_prenormed):
    return lm.model.embed(ids) if embed_prenormed else lm.model.embed_tokens(ids)


def mtp_predictions(model, mtp, seq, hidden, variant, depths=None):
    """Teacher-forced replay. Head d at position i consumes the chained hidden
    state and the embedding of token i+d+1, and predicts token i+d+2, so each
    head runs over one position fewer than the one before it."""
    lm = model.language_model
    total = len(seq)
    chained = lm.model.norm(hidden) if variant["hidden_post_norm"] else hidden

    layers = mtp.layers[: depths or len(mtp.layers)]
    predictions = []
    for depth, layer in enumerate(layers):
        count = total - depth - 2
        if count <= 0:
            break
        t0 = time.perf_counter()
        ids = mx.array(seq[depth + 1 : depth + 1 + count])[None, :]
        embed = head_embedding(lm, ids, variant["embed_prenormed"])
        chained = layer(
            chained[:, :count, :], embed, hidden_first=variant["hidden_first"]
        )
        guess = lm.speculative_argmax_from_hidden(chained)
        mx.eval(guess, chained)
        predictions.append(np.array(guess[0], copy=True))
        if TIMING:
            print(
                f"    head {depth} {count}pos {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
    return predictions


def acceptance(predictions, seq, start):
    """Per-depth hit masks over the decoded region, on the position set every
    depth can be scored on so the marginal and joint curves are comparable."""
    count = min(p.shape[0] for p in predictions)
    hits = [
        np.asarray(p[:count] == np.asarray(seq[depth + 2 : depth + 2 + count]))
        for depth, p in enumerate(predictions)
    ]
    return [h[start:] for h in hits]


def curves(hits):
    joint = np.ones_like(hits[0])
    marginal_out, joint_out = [], []
    for hit in hits:
        joint = joint & hit
        marginal_out.append(float(hit.mean()))
        joint_out.append(float(joint.mean()))
    return marginal_out, joint_out


SURVIVOR_FRACTION = 0.5


def probe(model, mtp, seq, hidden, prompt_len):
    """Identify the wiring in two stages, because the axes do not separate
    equally. Feeding `input_proj` its two halves the wrong way round puts the
    hidden state in the half of the weight trained for embeddings, and head 1
    then agrees with the model on nothing at all — one head settles that.

    The two normalisation choices degrade gently instead, and at one head they
    land within a few points of each other. Running the survivors to full depth
    is what separates them: a head fed a slightly wrong input still guesses the
    easy tokens, and only compounding over eight of them shows the cost."""
    rows = []
    for variant in VARIANTS:
        predictions = mtp_predictions(model, mtp, seq, hidden, variant, depths=1)
        hits = acceptance(predictions, seq, prompt_len - 1)
        rows.append({**variant, "depth_1_acceptance": float(hits[0].mean())})
        print(
            f"  depth 1  {variant} -> {rows[-1]['depth_1_acceptance']:.4f}", flush=True
        )

    cutoff = SURVIVOR_FRACTION * max(r["depth_1_acceptance"] for r in rows)
    for row in rows:
        if row["depth_1_acceptance"] < cutoff:
            continue
        predictions = mtp_predictions(model, mtp, seq, hidden, row)
        marginal, joint = curves(acceptance(predictions, seq, prompt_len - 1))
        row["marginal"] = marginal
        row["joint"] = joint
        row["expected_tokens_per_round"] = 1.0 + float(np.sum(joint))
        print(
            f"  full     {ordered_variant(row)} -> "
            f"{row['expected_tokens_per_round']:.3f} tok/round  "
            + " ".join(f"{v * 100:.0f}" for v in marginal),
            flush=True,
        )

    survivors = [r for r in rows if "joint" in r]
    return max(survivors, key=lambda r: r["expected_tokens_per_round"]), rows


def ordered_variant(row):
    return {k: row[k] for k in VARIANTS[0]}


def time_step(fn, repeats):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def forward_hidden(lm, tokens, cache):
    """Always evaluate. An unevaluated forward costs the graph-building time and
    nothing else, which reads as an impossibly fast decode step and leaves the
    real work pending — enough of them and the flush evicts the mapped weights,
    after which everything downstream runs at disk speed."""
    out = lm(inputs=tokens, cache=cache, return_hidden=True, skip_logits=True)
    hidden = out.hidden_states[-1]
    mx.eval(hidden)
    return hidden


def warm_up(model, tokens):
    """MLX maps the checkpoint rather than reading it, so a weight is faulted off
    disk by the first pass that reaches it. Timing anything before that measures
    NVMe bandwidth.

    Random ids rather than text: what has to be faulted in is the expert bank,
    all 256 per layer, and real tokens route to the same few. One wide prefill
    over uniform ids reaches them in a single pass, where sequential decoding
    takes thousands of tokens to cover the same ground."""
    vocab = model.config.text_config.vocab_size
    ids = mx.random.randint(0, vocab, (1, tokens))
    lm = model.language_model
    t0 = time.perf_counter()
    forward_hidden(lm, ids, lm.make_cache())
    mx.clear_cache()
    return time.perf_counter() - t0


def cost_tokens(processor, results, needed):
    """Real decoded tokens, not filler. What a verify block costs is set by how
    many distinct experts its tokens reach, and a block of one repeated token
    reaches the same six the single-token step already read — which prices the
    block at a decode step and flatters speculation by several fold."""
    tok = tokenizer(processor)
    ids = []
    for row in results["prompts"]:
        ids.extend(tok.encode(row["text"], add_special_tokens=False))
        if len(ids) >= needed:
            return ids[:needed]
    raise ValueError(
        f"need {needed} decoded tokens to price a verify block, have {len(ids)}; "
        "measure more prompts or lower --cost-context"
    )


def measure_costs(model, mtp, ids, depths, context_tokens, repeats):
    """A verify round replaces one decode step with a `k + 1`-token forward plus
    k passes through the heads, measured against a warmed cache at a realistic
    context length. The block costs more than a decode step because its tokens
    route to different experts, so the MoE reads several times the weight a
    single token does — the effect that decides whether speculation pays here."""
    lm = model.language_model
    prompt = mx.array(ids[:context_tokens])[None, :]
    follow = ids[context_tokens:]
    one = mx.array([follow[:1]])

    def against_warm_cache(tokens):
        cache = lm.make_cache()
        forward_hidden(lm, prompt, cache)
        return lambda: forward_hidden(lm, tokens, cache)

    # Swept rather than measured at full depth: the block forward is superlinear
    # in its token count because the MoE reads a wider slice of the expert bank
    # per extra token, so the depth worth speculating to is the one that pays for
    # itself, not the one the checkpoint happens to ship heads for.
    verify_s = [
        time_step(against_warm_cache(mx.array([follow[: k + 1]])), repeats)
        for k in range(depths + 1)
    ]

    caches = mtp.make_cache()
    hidden = forward_hidden(lm, prompt, lm.make_cache())
    embed = head_embedding(lm, prompt, True)
    for layer, cache in zip(mtp.layers, caches):
        hidden = layer(hidden, embed, cache=cache)
    mx.eval(hidden)
    last = hidden[:, -1:, :]
    first_embed = head_embedding(lm, one, True)

    def chain_upto(k):
        """Head d+1 embeds the token head d proposed, so the argmax and the
        lookup are part of the chain's cost. They stay on the GPU: an engine has
        no reason to sync each head back to the host, so only the final result
        is evaluated."""

        def run():
            h, e = last, first_embed
            for layer, cache in zip(mtp.layers[:k], caches[:k]):
                h = layer(h, e, cache=cache)
                e = head_embedding(lm, lm.speculative_argmax_from_hidden(h), True)
            mx.eval(e)

        return run

    chain_s = [0.0] + [time_step(chain_upto(k), repeats) for k in range(1, depths + 1)]
    mx.clear_cache()
    return {
        "decode_s": verify_s[0],
        "verify_block_s": verify_s,
        "mtp_chain_s": chain_s,
        "context_tokens": context_tokens,
        "repeats": repeats,
    }


def speedup_by_depth(joint, costs):
    """A round that speculates k tokens costs one k+1-token verify plus k heads
    and returns 1 + the joint acceptance through depth k. k = 0 is ordinary
    decoding and has to come out at 1.0."""
    rows = []
    for k in range(len(joint) + 1):
        tokens = 1.0 + float(np.sum(joint[:k]))
        ratio = (costs["verify_block_s"][k] + costs["mtp_chain_s"][k]) / costs[
            "decode_s"
        ]
        rows.append(
            {
                "speculated": k,
                "expected_tokens_per_round": tokens,
                "round_over_decode": ratio,
                "speedup": tokens / ratio,
            }
        )
    return rows


def best_depth(joint, costs):
    return max(speedup_by_depth(joint, costs), key=lambda r: r["speedup"])


def run_prompt(model, mtp, processor, name, text, args, variants):
    """Decoding is what a prompt costs; replaying the heads over the result is
    under a second. Every surviving wiring is scored on the same decode so the
    comparison between them is paired and effectively free."""
    tok = tokenizer(processor)
    prompt_ids = encode_prompt(model, processor, text)
    prompt = mx.array(prompt_ids)[None, :]

    t0 = time.perf_counter()
    generated, hidden = greedy_decode(
        model, prompt, args.max_tokens, stop_ids(model, processor)
    )
    decode_s = time.perf_counter() - t0

    seq = list(prompt_ids) + generated
    scored, positions = {}, 0
    for variant in variants:
        predictions = mtp_predictions(model, mtp, seq, hidden, variant)
        hits = acceptance(predictions, seq, len(prompt_ids) - 1)
        marginal, joint = curves(hits)
        scored[variant_key(variant)] = {"marginal": marginal, "joint": joint}
        positions = int(hits[0].size)

    return {
        "prompt": name,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "scored_positions": positions,
        "decode_s": decode_s,
        "decode_tok_s": len(generated) / decode_s,
        "variants": scored,
        "text": tok.decode(generated),
    }


def variant_key(variant):
    return ",".join(f"{k}={int(variant[k])}" for k in VARIANTS[0])


def pooled(rows, key):
    """Pool by scored position rather than averaging the per-prompt rates, so a
    prompt that hit EOS early does not weigh the same as one that ran to the
    limit."""
    weights = np.array([r["scored_positions"] for r in rows], dtype=np.float64)
    out = {}
    for curve in ("marginal", "joint"):
        values = np.array([r["variants"][key][curve] for r in rows])
        out[curve] = list(np.average(values, axis=0, weights=weights))
    out["scored_positions"] = int(weights.sum())
    out["expected_tokens_per_round"] = 1.0 + float(np.sum(out["joint"]))
    return out


def format_curve(label, values):
    return f"{label:<12}" + "".join(f"{v * 100:>7.1f}" for v in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a local checkpoint directory")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--prompts", nargs="+", default=sorted(PROMPTS))
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--probe-tokens", type=int, default=320)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--warmup-tokens", type=int, default=2048)
    ap.add_argument("--skip-costs", action="store_true")
    ap.add_argument("--cost-context", type=int, default=512)
    ap.add_argument("--cost-repeats", type=int, default=12)
    ap.add_argument("--memory-limit-gib", type=float, default=380.0)
    ap.add_argument("--json", default="reference/results/mtp_acceptance.json")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep the prompts already in --json and only decode the rest, so a "
        "run can be split across commands without repeating the expensive part",
    )
    args = ap.parse_args()

    global TIMING
    TIMING = args.timing
    mx.set_memory_limit(int(args.memory_limit_gib * GIB))
    config_json = json.loads((Path(args.model) / "config.json").read_text())

    t0 = time.perf_counter()
    model, processor = load_model(args.model)
    print(f"load            {time.perf_counter() - t0:.1f} s")
    mtp = load_mtp(model, args.model, config_json)
    print(f"resident        {gib(mx.get_active_memory()):.1f} GiB", flush=True)

    # Every forward here has to run under the raised wired limit. This model's
    # working set is far past the default, and unwired each kernel stalls paging
    # into GPU-reachable memory: a decode step costs 2.6 s rather than 33 ms,
    # flat in token count. `generate` sets this and restores it on exit, which
    # is why a loop written against the same modules is 80x slower on its own.
    with wired_limit(model):
        print(f"warmup          {warm_up(model, args.warmup_tokens):.1f} s", flush=True)
        measure(args, model, processor, mtp, config_json)


def measure(args, model, processor, mtp, config_json):
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if args.resume and out.exists() else {}
    results.update(
        {
            "model": args.model,
            "mtp_config": config_json["mtp_config"],
            "max_tokens": args.max_tokens,
        }
    )
    results.setdefault("prompts", [])

    def persist():
        out.write_text(json.dumps(results, indent=2) + "\n")

    if "probe" not in results:
        probe_name = args.prompts[0]
        probe_ids = encode_prompt(model, processor, PROMPTS[probe_name])
        generated, hidden = greedy_decode(
            model,
            mx.array(probe_ids)[None, :],
            args.probe_tokens,
            stop_ids(model, processor),
        )
        print(f"probe on '{probe_name}': {len(generated)} tokens", flush=True)
        best, rows = probe(
            model, mtp, list(probe_ids) + generated, hidden, len(probe_ids)
        )
        results["probe"] = {
            "prompt": probe_name,
            "variants": rows,
            "best": ordered_variant(best),
        }
        persist()
    if args.probe_only:
        return

    # Every wiring the probe could not rule out is carried into the measurement:
    # a few points at one head is not enough to pick between them, and scoring
    # them all on the same decodes settles it on ~20x the positions for free.
    survivors = [
        ordered_variant(r) for r in results["probe"]["variants"] if "joint" in r
    ]
    print(f"carrying {len(survivors)} wiring(s) into the measurement", flush=True)

    done = {r["prompt"] for r in results["prompts"]}
    for name in args.prompts:
        if name in done:
            print(f"{name}: already measured, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        row = run_prompt(model, mtp, processor, name, PROMPTS[name], args, survivors)
        results["prompts"].append(row)
        persist()
        print(
            f"{name}: {row['generated_tokens']} tok, "
            f"{row['scored_positions']} scored, "
            f"{row['decode_tok_s']:.1f} tok/s, {time.perf_counter() - t0:.0f} s",
            flush=True,
        )

    # After the prompts, so the block can be priced on tokens the model actually
    # produced rather than on filler.
    if not args.skip_costs and "costs" not in results:
        needed = args.cost_context + len(mtp.layers) + 1
        results["costs"] = measure_costs(
            model,
            mtp,
            cost_tokens(processor, results, needed),
            len(mtp.layers),
            args.cost_context,
            args.cost_repeats,
        )
        print(f"costs           {results['costs']}", flush=True)
        persist()

    summarise(results, survivors, len(mtp.layers))
    persist()


def summarise(results, survivors, depths):
    header = "depth       " + "".join(f"{d + 1:>7}" for d in range(depths))
    results["pooled"] = {
        variant_key(v): pooled(results["prompts"], variant_key(v)) for v in survivors
    }
    best = max(
        results["pooled"],
        key=lambda k: results["pooled"][k]["expected_tokens_per_round"],
    )
    results["chosen_wiring"] = best

    for key, curves_ in results["pooled"].items():
        mark = "*" if key == best else " "
        print(f"\n{mark} {key}  ({curves_['scored_positions']} positions)")
        print(header)
        print(format_curve("marginal %", curves_["marginal"]))
        print(format_curve("joint %", curves_["joint"]))
        print(f"expected tokens/round {curves_['expected_tokens_per_round']:.3f}")

    costs = results.get("costs")
    for row in results["prompts"]:
        joint = row["variants"][best]["joint"]
        print(f"\n{row['prompt']}  ({row['scored_positions']} positions)")
        print(header)
        print(format_curve("marginal %", row["variants"][best]["marginal"]))
        print(format_curve("joint %", joint))
        if costs:
            row["best"] = best_depth(joint, costs)
            print(
                f"best k={row['best']['speculated']}  "
                f"{row['best']['expected_tokens_per_round']:.2f} tok/round  "
                f"{row['best']['speedup']:.2f}x"
            )

    if not costs:
        return
    results["speedup_by_depth"] = speedup_by_depth(
        results["pooled"][best]["joint"], costs
    )
    results["speedup"] = best_depth(results["pooled"][best]["joint"], costs)
    print(f"\n{'k':>3}  {'tok/round':>10}  {'round/decode':>13}  {'speedup':>8}")
    for r in results["speedup_by_depth"]:
        print(
            f"{r['speculated']:>3}  {r['expected_tokens_per_round']:>10.3f}  "
            f"{r['round_over_decode']:>13.3f}  {r['speedup']:>8.3f}"
        )


if __name__ == "__main__":
    main()
