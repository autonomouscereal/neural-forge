#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-unified-20260710
RUN_ROOT=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711
TAG=p35_mixed_drop16_pareto_v7_20260711
GPU_UUID=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582

test -d "$ROOT"
test ! -e "$RUN_ROOT/$TAG"
test ! -e "$RUN_ROOT/$TAG.log"
test ! -e "$RUN_ROOT/$TAG.pid"
test ! -e "$RUN_ROOT/$TAG.identity.json"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU_UUID" '$1 == uuid')
test -n "$GPU_ROW"
GPU_MEMORY_USED=$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')
test "$GPU_MEMORY_USED" -lt 1024

export ROOT RUN_ROOT TAG
export GPU="$GPU_UUID"
export MAX_CANDIDATE_VALIDATIONS=8
unset INITIAL_CANDIDATE_VALIDATION_SUMMARY
bash "$ROOT/experiments/launch_qwen3_4b_unified_surgery_20260711.sh"

printf 'prelaunch_gpu=%s validation_source=fresh-independent\n' "$GPU_ROW"
