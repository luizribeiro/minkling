#!/usr/bin/env bash
# Patch the mlx-vlm installed in the reference venv so it can load Inkling
# checkpoints. Idempotent: a patch that reverse-applies cleanly is already in
# place. Patches are rooted at site-packages and applied with -p1.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${1:-$here/../.venv}"
root="$venv/lib/python3.12/site-packages"

[ -d "$root/mlx_vlm" ] || {
  echo "no mlx-vlm in $venv; run 'uv sync' first" >&2
  exit 1
}

for p in "$here"/../patches/*.patch; do
  name="$(basename "$p")"
  if patch -s -p1 -R --dry-run -f -d "$root" <"$p" >/dev/null 2>&1; then
    echo "  already applied  $name"
  elif patch -s -p1 -d "$root" <"$p"; then
    echo "  applied          $name"
  else
    echo "  FAILED           $name" >&2
    exit 1
  fi
done
