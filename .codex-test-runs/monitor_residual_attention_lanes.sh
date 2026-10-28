#!/usr/bin/env bash
set -u

BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0003
for lane in fit_guided full_strength_structural residual_writer; do
  pid=$(cat "$BASE/$lane/postfold.pid")
  if kill -0 "$pid" 2>/dev/null; then
    state=running
  else
    state=exited
  fi
  receipt=absent
  if [[ -s "$BASE/$lane/postfold-receipt.json" ]]; then
    receipt=present
  fi
  printf '\n%s pid=%s state=%s receipt=%s\n' "$lane" "$pid" "$state" "$receipt"
  tail -n 12 "$BASE/$lane/postfold.log"
done

printf '\nGPU index,memory MiB,utilization percent\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
