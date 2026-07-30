#!/usr/bin/env bash
# Patch the mlx-vlm installed in the reference venv. Idempotent: a patch that
# reverse-applies cleanly is already in place.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${1:-$here/../.venv}"
pkg="$venv/lib/python3.12/site-packages/mlx_vlm/models/inkling"

[ -d "$pkg" ] || { echo "no mlx-vlm in $venv; run 'uv sync' first" >&2; exit 1; }

for p in "$here"/../patches/inkling-*.patch; do
  name="$(basename "$p")"
  target="$pkg/${name#inkling-}"
  target="${target%.patch}"

  if patch -s -R --dry-run -f "$target" "$p" >/dev/null 2>&1; then
    echo "  already applied  $name"
  elif patch -s "$target" "$p"; then
    echo "  applied          $name"
  else
    echo "  FAILED           $name" >&2
    exit 1
  fi
done
