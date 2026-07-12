#!/usr/bin/env bash
set -euo pipefail

BASE=/home/cereal/model-surgery-lab/neural-forge-pivot
TARGET=/home/cereal/model-surgery-lab/neural-forge-unified-20260710
ARCHIVE=/tmp/neural-forge-unified-20260711.zip
PY=/home/cereal/model-surgery-lab/venv/bin/python

test -d "$BASE"
test -f "$ARCHIVE"
test ! -e "$TARGET"
cp -a "$BASE" "$TARGET"

python3 - "$ARCHIVE" "$TARGET" <<'PY'
from pathlib import Path
import sys
import zipfile

archive = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive) as handle:
    for member in handle.infolist():
        relative = Path(member.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe archive member: {member.filename!r}")
        destination = (target / relative).resolve()
        if target != destination and target not in destination.parents:
            raise SystemExit(f"archive member escapes target: {member.filename!r}")
    handle.extractall(target)
PY

cd "$TARGET"
PYTHONPATH="$TARGET/src" "$PY" -m compileall -q src experiments tests
if "$PY" -c 'import ruff' >/dev/null 2>&1; then
  PYTHONPATH="$TARGET/src" "$PY" -m ruff check \
    src/neural_forge/hf_evaluation_provenance.py \
    src/neural_forge/hf_multi_anchor_cluster.py \
    src/neural_forge/hf_parent_compiler.py \
    src/neural_forge/hf_qwen_codebook_model.py \
    src/neural_forge/hf_qwen_sensitivity.py \
    src/neural_forge/hf_qwen_surgery_allocator.py \
    experiments/prepare_qwen_unified_splits.py \
    experiments/run_qwen_unified_surgery.py
else
  printf 'remote ruff unavailable; local focused lint is authoritative\n'
fi
PYTHONPATH="$TARGET/src" "$PY" -m pytest -q \
  tests/test_hf_evaluation_provenance.py \
  tests/test_run_qwen_unified_surgery.py \
  tests/test_prepare_qwen_unified_splits.py \
  tests/test_hf_qwen_codebook_model.py \
  tests/test_hf_qwen_surgery_allocator.py

printf 'deployed target=%s archive=%s\n' "$TARGET" "$ARCHIVE"
