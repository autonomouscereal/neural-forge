#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals}"

export TAG="${TAG:-lfm230m_blank_smollm_trainable_byteprobe_full_gpu6_20260702}"
export MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
export QUEUE_PID="${QUEUE_PID:-0}"
export PARENT_WAIT_PIDS="${PARENT_WAIT_PIDS:-0}"
export PARENT_REPORT_CANDIDATES="${PARENT_REPORT_CANDIDATES:-$EVAL_DIR/eval_donor_lfm230m_exactgen_20260702_1000x1000x250.json}"
export GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
export EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
export MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
export MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-1000}"
export MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"
export BASELINE_REPORTS="${BASELINE_REPORTS:-blank_random=$EVAL_DIR/eval_blank_smollm135m_random_1000g_20260701_1000x1000x250.json;cleanopen_l4=$EVAL_DIR/eval_cleanopen_l4_alm_1000g_20260701_1000x1000x250.json;mathdrill_l4=$EVAL_DIR/eval_mathdrill_l4_alm_1000g_20260701_1000x1000x250.json}"

exec bash "$SCRIPT_DIR/queue_eval_retention_compare.sh"
