#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"

export VARIANT="${VARIANT:-qwen}"
export TAG="${TAG:-qwen4b_basisaligned_anchor_beststack_recover_gpu6_20260707}"
export GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
export WAIT_PIDS="${WAIT_PIDS:-3436884 3436885}"
export POLL_SECONDS="${POLL_SECONDS:-120}"
export ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/qwen4b_blank_smollm_basisaligned_initonly_qrsinkhorn_gpu0wait_20260707}"
export STAGES="${STAGES:-block,alm}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-quantile-medoid}"
export BLOCK_NONFINITE_STEP_POLICY="${BLOCK_NONFINITE_STEP_POLICY:-skip-step}"
export BLOCK_MAX_NONFINITE_STEPS_PER_LAYER="${BLOCK_MAX_NONFINITE_STEPS_PER_LAYER:-12}"
export BLOCK_LR="${BLOCK_LR:-2e-7}"
export RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.005}"
export ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT="${ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT:-0.08}"
export FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"
export LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
