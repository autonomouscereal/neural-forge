#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"

export TAG="${TAG:-qwen4b_gpt2_multidonor_seqall_teacherreplay_now_gpu4_20260702}"
export MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
export WAIT_PIDS="${WAIT_PIDS:-0}"
export GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
export TASKS="${TASKS:-arc_easy,hellaswag,piqa,winogrande}"
export LIMIT="${LIMIT:-50}"
export MODEL_ENTRIES="${MODEL_ENTRIES:-child=$MODEL_DIR;qwen_parent=Qwen/Qwen3-4B-Base;gpt2_donor=openai-community/gpt2;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;cleanopen_l4=$RUN_ROOT/qwen4b_cleanopen_anchor_newloss_l4_gpu0_20260701;mathdrill_l4=$RUN_ROOT/qwen4b_mathdrill_anchor_newloss_l4_gpu6_20260701;topneg_margin=$RUN_ROOT/qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702}"

exec bash "$SCRIPT_DIR/queue_lm_eval_after_retention.sh"
