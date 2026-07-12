#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-src
RUN_ROOT=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711
TAG=p35_mixed_drop16_pareto_v4_20260711
SOURCE_ROUND="$RUN_ROOT/p35_mixed_drop16_pareto_v3_20260711/candidate_validation/round-000001"
GPU_UUID=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582

test -d "$SOURCE_ROUND"
test -f "$SOURCE_ROUND/round_claim.json"
test ! -e "$RUN_ROOT/$TAG"
test ! -e "$RUN_ROOT/$TAG.pid"
test ! -e "$RUN_ROOT/$TAG.identity.json"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU_UUID" '$1 == uuid')
test -n "$GPU_ROW"

export ROOT RUN_ROOT TAG
export GPU="$GPU_UUID"
export MAX_CANDIDATE_VALIDATIONS=8
export INITIAL_CANDIDATE_VALIDATION_SUMMARY="$SOURCE_ROUND"
bash "$ROOT/experiments/launch_qwen3_4b_unified_surgery_20260711.sh"

printf 'prelaunch_gpu=%s source_round=%s\n' "$GPU_ROW" "$SOURCE_ROUND"
