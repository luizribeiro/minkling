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

fetch repo="thinkingmachines/Inkling-Small":
    hf download {{ repo }}
