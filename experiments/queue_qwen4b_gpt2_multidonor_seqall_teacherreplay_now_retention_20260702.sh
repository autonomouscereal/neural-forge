#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals}"

export TAG="${TAG:-qwen4b_gpt2_multidonor_seqall_teacherreplay_now_gpu4_20260702}"
export MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
export QUEUE_PID="${QUEUE_PID:-$(cat "$RUN_ROOT/logs/$TAG.pid" 2>/dev/null || echo 0)}"
export QUEUE_PID_FILE="${QUEUE_PID_FILE:-$RUN_ROOT/logs/$TAG.pid}"
export PARENT_WAIT_PIDS="${PARENT_WAIT_PIDS:-3345256 3692476}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
export MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
export MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-1000}"
export MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"
export BASELINE_REPORTS="${BASELINE_REPORTS:-qwen_parent_cached=$EVAL_DIR/eval_parent_qwen3_4b_base_1000g_20260701_1000x1000x250.json;gpt2_donor=$EVAL_DIR/eval_donor_gpt2_exactgen_20260702_1000x1000x250.json;blank_random=$EVAL_DIR/eval_blank_smollm135m_random_1000g_20260701_1000x1000x250.json;cleanopen_l4=$EVAL_DIR/eval_cleanopen_l4_alm_1000g_20260701_1000x1000x250.json;mathdrill_l4=$EVAL_DIR/eval_mathdrill_l4_alm_1000g_20260701_1000x1000x250.json;cleanopen_full=$EVAL_DIR/eval_cleanopen_full_02_alm_exactgen_20260702_1000x1000x250.json;topneg_margin=$EVAL_DIR/eval_qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702_exactgen_1000x1000x250.json}"

exec bash "$SCRIPT_DIR/queue_eval_retention_compare.sh"
