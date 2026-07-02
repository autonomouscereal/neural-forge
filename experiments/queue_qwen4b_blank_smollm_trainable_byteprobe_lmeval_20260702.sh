#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"

export TAG="${TAG:-qwen4b_blank_smollm_cka_weighted_block_weighted_trainable_byteprobe_full_gpu4_20260702}"
export MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
export WAIT_PIDS="${WAIT_PIDS:-3775678}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export TASKS="${TASKS:-arc_easy,hellaswag,piqa,winogrande}"
export LIMIT="${LIMIT:-50}"
export MODEL_ENTRIES="${MODEL_ENTRIES:-child=$MODEL_DIR;qwen_parent=Qwen/Qwen3-4B-Base;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;cleanopen_l4=$RUN_ROOT/qwen4b_cleanopen_anchor_newloss_l4_gpu0_20260701;mathdrill_l4=$RUN_ROOT/qwen4b_mathdrill_anchor_newloss_l4_gpu6_20260701;cleanopen_full=$RUN_ROOT/qwen4b_cleanopen_full_02_alm_exactgen_20260702;topneg_margin=$RUN_ROOT/qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702}"

exec bash "$SCRIPT_DIR/queue_lm_eval_after_retention.sh"
