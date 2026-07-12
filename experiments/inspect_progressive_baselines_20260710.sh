#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/ai_data/neural-forge/progressive_stage1_20260709
for NAME in \
  qwen4b_to_blank1p410b_adaptive_observable_path_gpu6_r2_20260710 \
  qwen4b_to_blank1p410b_adaptive_global_native_gpu0_20260710 \
  qwen4b_to_blank1p410b_adaptive_observable_native_gpu4_20260710
do
  echo "=== $NAME ==="
  LOG="$ROOT/processes/$NAME.log"
  if [[ -f "$LOG" ]]; then
    tail -n 100 "$LOG"
  fi
  echo "=== $NAME artifacts ==="
  if [[ -d "$ROOT/$NAME" ]]; then
    find "$ROOT/$NAME" -maxdepth 4 -type f \
      \( -name '*.json' -o -name '*.log' -o -name '*.txt' \) \
      -printf '%P %s bytes\n' | sort | tail -n 100
  fi
done
