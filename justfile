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

# Regenerate the committed MXFP4 slices the Rust dequantiser is tested against
dump-quant-fixture model="models/Inkling-Small-mxfp4":
    reference/.venv/bin/python reference/scripts/dump_quant_fixture.py {{ model }}

fetch repo="thinkingmachines/Inkling-Small":
    hf download {{ repo }}
