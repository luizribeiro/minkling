default:
    @just --list

# Build the Python reference env and patch mlx-vlm into a loadable state
sync:
    cd reference && uv sync
    reference/scripts/apply_patches.sh

test:
    cargo nextest run

fmt:
    cargo fmt --all
    cargo clippy --all-targets -- -D warnings

# Summarise a checkpoint's architecture and KV cost
inspect config:
    cargo run -q --bin inklingrs -- {{ config }}

# Baseline load cost and decode throughput via the reference implementation
smoke model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/smoke.py {{ model }}

# Prefill wall time, throughput and peak memory across a prompt-length sweep
prefill-bench model="models/Inkling-Small-mxfp4" *args:
    reference/.venv/bin/python reference/scripts/prefill_bench.py {{ model }} {{ args }}

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

# Regenerate the committed synthetic model stack the Rust stack is tested against
dump-stack-fixture:
    reference/.venv/bin/python reference/scripts/dump_stack_fixture.py

# Regenerate the committed router gate and synthetic cases the Rust MoE is tested against
dump-moe-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_moe_fixture.py {{ model }}

# Regenerate the committed synthetic tensors the Rust CPU ops are tested against
dump-op-fixture:
    reference/.venv/bin/python reference/scripts/dump_op_fixture.py

fetch repo="thinkingmachines/Inkling-Small":
    hf download {{ repo }}

# Quantise the BF16 original to 8-bit, keeping the MTP tensors the mxfp4 quant
# dropped. Streams a shard at a time and resumes from what it has already
# written, so it can be run in chunks and re-run until it prints an index:
#   just quantize "$src" "$dst" --time-budget 480
quantize src="/mnt/truenas/models/thinkingmachines/Inkling-Small" dst="models/Inkling-Small-8bit" *args:
    reference/.venv/bin/python reference/scripts/quantize.py {{ src }} {{ dst }} {{ args }}
