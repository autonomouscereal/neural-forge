#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab
PY="$ROOT/venv/bin/python"
SCRIPT="$ROOT/neural-forge-src/upload_staging/run_lfm2_final_review.py"
OUT="$ROOT/artifacts/lfm2_dynamic_fold_20260716/final-only-whole-output-review-midfrontier-20260716"

"$PY" "$SCRIPT" \
  --output "$OUT" \
  prepare \
  --label m2556 \
  --label m2552 \
  --label m2544

launch() {
  local label="$1"
  local cuda_ordinal="$2"
  nohup env \
    CUDA_VISIBLE_DEVICES="$cuda_ordinal" \
    PYTHONPATH="$ROOT/neural-forge-src/src" \
    "$PY" "$SCRIPT" \
      --output "$OUT" \
      generate \
      --label "$label" \
      --device cuda:0 \
      >"$OUT/$label-generation.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$OUT/$label-generation.pid"
  echo "$label pid=$pid cuda_ordinal=$cuda_ordinal"
}

launch m2556 0
launch m2552 1
launch m2544 2
