#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab
PY="$ROOT/venv/bin/python"
SCRIPT="$ROOT/neural-forge-src/upload_staging/run_lfm2_final_review.py"
OUT="$ROOT/artifacts/lfm2_dynamic_fold_20260716/final-only-whole-output-review-round3-20260716"

"$PY" "$SCRIPT" \
  --output "$OUT" \
  prepare \
  --label m2536 \
  --label m2528 \
  --label m2496

launch() {
  local label="$1"
  local gpu="$2"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$ROOT/neural-forge-src/src" \
    "$PY" "$SCRIPT" \
      --output "$OUT" \
      generate \
      --label "$label" \
      --device cuda:0 \
      >"$OUT/$label-generation.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$OUT/$label-generation.pid"
  echo "$label pid=$pid physical_gpu=$gpu"
}

launch m2536 0
launch m2528 4
launch m2496 6
