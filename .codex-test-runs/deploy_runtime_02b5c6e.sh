#!/usr/bin/env bash
set -euo pipefail

SHA=02b5c6e4e92bd5eb38619376e273c75136c1f5c1
ROOT=/home/cereal/model-surgery-lab/runtime
ARCHIVE="$ROOT/neural-forge-$SHA.tar"
TARGET="$ROOT/neural-forge-$SHA"

test -f "$ARCHIVE"
test ! -e "$TARGET"
mkdir -p "$TARGET"
tar -xf "$ARCHIVE" -C "$TARGET"
test -f "$TARGET/pyproject.toml"
printf '%s\n' "$TARGET"
