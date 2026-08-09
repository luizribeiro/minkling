default:
    @just --list

# Check the root-owned boundaries and every Cargo workspace member.
check:
    cargo deny --frozen check bans sources
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --locked -- -D warnings

# Run the legacy suite without accidentally activating checkpoint tests.
test: check
    env -u INKLINGRS_CHECKPOINT cargo test --workspace --locked

# TODO(slop-features): Add `--all-features` to the Clippy and test commands
# after reviewing the legacy feature-only paths; feature-gated code can
# otherwise evade both checks.
