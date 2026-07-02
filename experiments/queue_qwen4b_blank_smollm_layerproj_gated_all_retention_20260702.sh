#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"

export TAG="${TAG:-qwen4b_blank_smollm_cka_weighted_block_weighted_layerproj_gated_all_full_gpu4_20260702}"
export MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
export QUEUE_PID="${QUEUE_PID:-3634313}"
export PARENT_WAIT_PIDS="${PARENT_WAIT_PIDS:-3345256}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
export MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
export MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-1000}"
export MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"

exec bash "$SCRIPT_DIR/queue_eval_retention_compare.sh"
