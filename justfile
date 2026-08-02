checkpoint := "models/Inkling-Small-mxfp4"

default:
    @just --list

# Build the Python reference env and patch mlx-vlm into a loadable state
sync:
    cd reference && uv sync
    reference/scripts/apply_patches.sh

# Every assertion that needs no checkpoint, in about twelve seconds. What to run
# while iterating.
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
# with nothing beside them. `#[ignore]` is what keeps these out of every run that
# has tests beside them, and what selects them here; `.config/nextest.toml` says
# what a number taken beside another test is worth.
test-timing checkpoint=checkpoint:
    INKLINGRS_CHECKPOINT={{ absolute_path(checkpoint) }} \
        cargo nextest run --profile timing --run-ignored only

# Everything, and what has to pass before a commit lands: the whole suite
# against a real checkpoint, a process a test, then the measurements on their
# own. About three minutes forty, most of it the CPU oracle at 9.0 s a decoded
# token.
test-full checkpoint=checkpoint: && (test-timing checkpoint)
    INKLINGRS_CHECKPOINT={{ absolute_path(checkpoint) }} cargo nextest run

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

# Quantise the BF16 original to 8-bit, keeping the MTP tensors the mxfp4 quant
# dropped. Streams a shard at a time and resumes from what it has already
# written, so it can be run in chunks and re-run until it prints an index:
#   just quantize "$src" "$dst" --time-budget 480
quantize src="/mnt/truenas/models/thinkingmachines/Inkling-Small" dst="models/Inkling-Small-8bit" *args:
    reference/.venv/bin/python reference/scripts/quantize.py {{ src }} {{ dst }} {{ args }}
