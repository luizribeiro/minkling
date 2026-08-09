# minkling

![minkling](slop/minkling.png)

This repository is being rebuilt into a small, human-reviewed server for
running Inkling models on Apple Silicon.

The working but unreviewed generated implementation lives in `slop/`, behind
root-owned Cargo, Clippy, cargo-deny, and Nix policies. New server code will
live outside that quarantine and reach the model through a narrow interface.
