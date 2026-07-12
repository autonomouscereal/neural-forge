#!/usr/bin/env bash
set -euo pipefail

TARGET=/home/cereal/model-surgery-lab/neural-forge-unified-20260710
EXPECTED_PARENT=/home/cereal/model-surgery-lab
resolved=$(realpath -m -- "$TARGET")
parent=$(dirname -- "$resolved")
test "$resolved" = "$TARGET"
test "$parent" = "$EXPECTED_PARENT"
test -d "$TARGET"
test -f "$TARGET/pyproject.toml"
rm -rf -- "$TARGET"
test ! -e "$TARGET"
printf 'removed failed deployment target=%s\n' "$TARGET"
