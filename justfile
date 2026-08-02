checkpoint := "models/Inkling-Small-mxfp4"

# How many pairs `just bench` alternates over. Seven is what most of the paired
# figures in the README were taken over; `BENCH_PAIRS=3 just bench …` is the
# shorter sitting a large effect does not need seven of.
pairs := env("BENCH_PAIRS", "7")

default:
    @just --list

# Build the Python reference env and patch mlx-vlm into a loadable state
sync:
    cd reference && uv sync
    reference/scripts/apply_patches.sh

# Every assertion that needs no checkpoint, in about twelve seconds. **What to
# run while iterating**, and what to run after every edit.
#
# One process a crate, because opening a Metal device costs a second and the 110
# kernel tests would each pay it. Nothing here measures the process it runs in,
# which is what makes sharing one free; `test-timing` is where that stops being
# true. The README's test section has the figures.
#
# The checkpoint is unset rather than left to the environment, so a shell that
# exports it does not turn this into `test-full` by surprise.
test:
    env -u INKLINGRS_CHECKPOINT cargo test --workspace

# The measurements — a duration, a resident set, a profile table — one at a time
# with nothing beside them. **What to run for anything timed**, and the only
# thing that runs the measurements at all.
#
# `#[ignore]` is what keeps these out of every run that has tests beside them,
# and what selects them here; `.config/nextest.toml` says what a number taken
# beside another test is worth. What this does not answer is whether a number
# moved between two builds — see `just bench`, which is the paired arrangement
# and does not rebuild between pairs.
test-timing checkpoint=checkpoint:
    INKLINGRS_CHECKPOINT={{ absolute_path(checkpoint) }} \
        cargo nextest run --profile timing --run-ignored only

# Everything against a real checkpoint, a process a test, then the measurements
# on their own. About three minutes forty, most of it the CPU oracle at 9.0 s a
# decoded token.
#
# **What it is for is the ends of a series, not every commit in one.** The CPU
# oracle is most of those minutes and it cannot have changed between two commits
# that never touched the CPU path, so an agent landing six commits paid twenty
# minutes to re-prove it five times over. Run it before the first commit of a
# series and again before the last; run `just test` in between, and
# `just test-timing` for anything that reports a number. A commit that touches no
# `.rs` file needs none of the three — the pre-commit hooks already skip clippy
# on those by config.
test-full checkpoint=checkpoint: && (test-timing checkpoint)
    INKLINGRS_CHECKPOINT={{ absolute_path(checkpoint) }} cargo nextest run

# Weigh two refs against each other in one sitting: one of
#
#   just bench HEAD~1 HEAD decode
#   just bench HEAD~1 .    prefill --tokens 769
#   just bench v1 v2       sweep --depth 4
#
# `.` is the working tree, which is the arm a change is measured from before it
# is a commit at all.
#
# **Each ref is built once and kept.** The binaries land in `target/bench/bin`
# under the commit they were built from, so a second sitting against the same
# pair builds nothing — and no pair of the seven rebuilds anything, which is what
# a flip used to cost. The alternation is what the pairs are for: same sitting,
# order flipped each pair, and the report says whether the ranges overlap.
#
# One arm at a time and one Metal device apiece, for the reason
# `.config/nextest.toml` gives: a number taken beside another measurement is a
# number about the other measurement.
#
# `rm -rf target/bench` is how the kept binaries are given back; nothing here
# evicts them, because what they cost is a build each and what they buy is every
# later sitting against the same commit.
bench a b *measurement:
    #!/usr/bin/env bash
    set -euo pipefail
    root="$(git rev-parse --show-toplevel)"
    bin="$root/target/bench/bin"
    mkdir -p "$bin"
    git worktree prune

    # A ref's binary, built if this is the first sitting that has asked for it.
    arm() {
        if [ "$1" = "." ]; then
            cargo build --quiet --bin bench
            # Copied out rather than run in place: the next `cargo build` would
            # otherwise swap the binary under a sitting that is still running.
            cp "$root/target/debug/bench" "$bin/working-tree"
            echo "$bin/working-tree"
            return
        fi
        local sha
        sha="$(git rev-parse --verify "$1^{commit}")"
        if [ ! -x "$bin/$sha" ]; then
            local tree="$root/target/bench/tree-$$"
            rm -rf "$tree"
            git worktree add --detach --quiet "$tree" "$sha"
            # A target directory of its own, so that building the other ref does
            # not invalidate this one and turn the pairs back into rebuilds.
            CARGO_TARGET_DIR="$root/target/bench/target" \
                cargo build --quiet --manifest-path "$tree/Cargo.toml" --bin bench
            cp "$root/target/bench/target/debug/bench" "$bin/$sha"
            git worktree remove --force "$tree"
        fi
        echo "$bin/$sha"
    }

    a="$(arm '{{ a }}')"
    b="$(arm '{{ b }}')"
    # The harness is the working tree's own, whichever refs the arms are.
    cargo build --quiet --bin bench
    # The measurement is deliberately unquoted: `sweep --depth 4` is three
    # arguments to each arm and quoting it would hand them one.
    "$root/target/debug/bench" alternate --pairs {{ pairs }} "$a" "$b" \
        -- {{ measurement }} "{{ absolute_path(checkpoint) }}"

# Weigh two checkpoints against each other with one build, which is the shape a
# change to the weights has:
#
#   just bench-weights models/Inkling-Small-mxfp4 models/Inkling-Small-mxfp4-mtp4 sweep
#
# The same alternation and the same report as `just bench` — an arm is a command
# line, so what differs between the two can be an argument as readily as an
# executable. The build is the working tree's, once, for both arms.
bench-weights a b *measurement:
    #!/usr/bin/env bash
    set -euo pipefail
    root="$(git rev-parse --show-toplevel)"
    cargo build --quiet --bin bench
    bin="$root/target/bench/bin/working-tree"
    mkdir -p "$(dirname "$bin")"
    cp "$root/target/debug/bench" "$bin"
    "$root/target/debug/bench" alternate --pairs {{ pairs }} \
        "$bin {{ absolute_path(a) }}" "$bin {{ absolute_path(b) }}" -- {{ measurement }}

fmt:
    cargo fmt --all
    cargo clippy --all-targets -- -D warnings

# Summarise a checkpoint's architecture and KV cost
inspect config:
    cargo run -q --bin inklingrs -- inspect {{ config }}

# Baseline load cost and decode throughput via the reference implementation
smoke model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/smoke.py {{ model }}

# Prefill wall time, throughput and peak memory across a prompt-length sweep
prefill-bench model="models/Inkling-Small-mxfp4" *args:
    reference/.venv/bin/python reference/scripts/prefill_bench.py {{ model }} {{ args }}

# Per-depth MTP acceptance and the speculation depth that pays for itself.
# Needs a checkpoint with the MTP tensors, so not the mxfp4 quant.
mtp-acceptance model="models/Inkling-Small-8bit" *args:
    reference/.venv/bin/python reference/scripts/mtp_acceptance.py {{ model }} \
        --json reference/results/mtp_acceptance.json {{ args }}

# Check that the checkpoint's tiktoken export and tokenizer.json agree, id for id
compare-tokenizers model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/compare_tokenizers.py {{ model }}

# Regenerate the committed reference activations the Rust layers are tested against
dump-activations model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_activations.py {{ model }}

# Recapture one sliding and one global layer over a sequence long enough to reach
# past the 1024-token relative-position band. Gitignored rather than committed —
# the masks alone are 210 MB a layer — and the tests that read it skip when it is
# absent, so regenerate it before running them.
dump-long-activations model="models/Inkling-Small-mxfp4" tokens="1280":
    reference/.venv/bin/python reference/scripts/dump_activations.py {{ model }} \
        --seq-len {{ tokens }} --layers 0 5 --name long_activations

# Regenerate the committed embed_norm weight the Rust embedding is tested against
dump-embed-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_embed_fixture.py {{ model }}

# Regenerate the committed MXFP4 slices the Rust dequantiser is tested against
dump-quant-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_quant_fixture.py {{ model }}

# Regenerate the committed sconv kernels and cases the Rust short conv is tested against
dump-sconv-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_sconv_fixture.py {{ model }}

# Regenerate the committed relative-position cases the Rust mask is tested against
dump-mask-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_mask_fixture.py {{ model }}

# Regenerate the committed synthetic attention cases the Rust layer is tested against
dump-attention-fixture:
    reference/.venv/bin/python reference/scripts/dump_attention_fixture.py

# Regenerate the committed synthetic decoder layers the Rust layer is tested against
dump-layer-fixture:
    reference/.venv/bin/python reference/scripts/dump_layer_fixture.py

# Regenerate the committed synthetic MTP heads the Rust head is tested against
dump-mtp-fixture:
    reference/.venv/bin/python reference/scripts/dump_mtp_fixture.py

# Regenerate the committed synthetic model stack the Rust stack is tested against
dump-stack-fixture:
    reference/.venv/bin/python reference/scripts/dump_stack_fixture.py

# Regenerate the committed router gate and synthetic cases the Rust MoE is tested against
dump-moe-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_moe_fixture.py {{ model }}

# Regenerate the committed synthetic tensors the Rust CPU ops are tested against
dump-op-fixture:
    reference/.venv/bin/python reference/scripts/dump_op_fixture.py

# Regenerate the committed text/id pairs the Rust tokenizer is tested against
dump-tokenizer-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_tokenizer_fixture.py {{ model }}

# Regenerate the committed prompts the Rust turn structure is tested against.
# The server writes the structure out by hand rather than interpreting
# chat_template.jinja, and this is what says the two agree.
dump-chat-template-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_chat_template_cases.py {{ model }}

fetch repo="thinkingmachines/Inkling-Small":
    hf download {{ repo }}

# Put the multi-token prediction heads beside a checkpoint that was quantised
# without them.
#
# Every quantiser left `model.mtp.*` in bfloat16 — the 8-bit quant's
# mtp.safetensors is the BF16 original's 160 tensors byte for byte — so the
# heads are not a quantisation of anything and pair with any stack quantised
# from the same original. That makes this a 4.5 GB copy where re-quantising the
# 532 GB original to keep them would be hours. No index names the shard in
# either quant; the loader maps it because it is there.
mtp-shard src="models/Inkling-Small-8bit" dst="models/Inkling-Small-mxfp4":
    cp {{ src }}/mtp.safetensors {{ dst }}/mtp.safetensors

# The same checkpoint with its heads packed into the format its stack is already
# in: a directory of symlinks to `src`, and one shard of its own.
#
# **The bfloat16 heads are not touched and must not be.** They are the oracle the
# packed ones' guesses are held against — `just bench-weights <src> <dst> guesses`
# — and the two checkpoints are the same 140 GB stack read twice, so what this
# costs on disk is the 1.1 GiB shard and forty symlinks.
#
# A loader maps every `*.safetensors` in a directory, which is why this is a
# directory rather than a second shard beside the first: two shards naming the
# same tensors is a checkpoint that holds each of them twice.
quantize-mtp src="models/Inkling-Small-mxfp4" dst="models/Inkling-Small-mxfp4-mtp4":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{ dst }}
    for path in {{ src }}/*; do
        name="$(basename "$path")"
        [ "$name" = "mtp.safetensors" ] && continue
        [ -e "{{ dst }}/$name" ] || ln -s "{{ absolute_path(src) }}/$name" "{{ dst }}/$name"
    done
    reference/.venv/bin/python reference/scripts/quantize_mtp.py \
        {{ src }}/mtp.safetensors {{ dst }}/mtp.safetensors --check

# Quantise the BF16 original to 8-bit, keeping the MTP tensors the mxfp4 quant
# dropped. Streams a shard at a time and resumes from what it has already
# written, so it can be run in chunks and re-run until it prints an index:
#   just quantize "$src" "$dst" --time-budget 480
quantize src="/mnt/truenas/models/thinkingmachines/Inkling-Small" dst="models/Inkling-Small-8bit" *args:
    reference/.venv/bin/python reference/scripts/quantize.py {{ src }} {{ dst }} {{ args }}
