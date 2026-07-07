#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
VARIANT="${VARIANT:-dualalm29}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
WAIT_PIDS="${WAIT_PIDS:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"

SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base}"
CALIBRATION_FILE="${CALIBRATION_FILE:-$RUN_ROOT/calibration_mix_plus_recovery_drills_20260702.jsonl}"
EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-640}"

case "$VARIANT" in
  dualalm29)
    TAG="${TAG:-qwen4b_dualalm29_anchor_topneg_exact_gpu6_20260707}"
    CANDIDATE_MODEL="${CANDIDATE_MODEL:-$REPO_ROOT/runs/qwen4b_to_smollm135m_dualspace_alm_20260627/tied29x_dualalm240_last4}"
    TIE_CANDIDATE_BEFORE_RECOVERY="${TIE_CANDIDATE_BEFORE_RECOVERY:-1}"
    FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"
    ALM_STEPS="${ALM_STEPS:-180}"
    ALM_LR="${ALM_LR:-5e-7}"
    ;;
  ffn1496)
    TAG="${TAG:-qwen4b_ffn1496_strict_anchor_topneg_gpu6_20260707}"
    CANDIDATE_MODEL="${CANDIDATE_MODEL:-$REPO_ROOT/runs/qwen4b_to_smollm135m_strict25_ffnfold_20260627_191250/alpha05_ffn1496_folded_xreconstruct_mlp_a05}"
    # The FFN1496 artifact is itself the strict >=25x anchor; preserve its head
    # shape for the recovery run unless a caller explicitly asks to tie it.
    TIE_CANDIDATE_BEFORE_RECOVERY="${TIE_CANDIDATE_BEFORE_RECOVERY:-0}"
    FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-preserve}"
    ALM_STEPS="${ALM_STEPS:-140}"
    ALM_LR="${ALM_LR:-5e-7}"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Expected dualalm29 or ffn1496." >&2
    exit 2
    ;;
esac

export REPO_ROOT RUN_ROOT TAG SOURCE_MODEL CANDIDATE_MODEL
export TIE_CANDIDATE_BEFORE_RECOVERY FINAL_OUTPUT_HEAD_POLICY
export GPU_UUID WAIT_PIDS POLL_SECONDS
export CALIBRATION_FILE EVAL_SUITE_FILE MAX_CALIBRATION_TEXTS
export ALM_STEPS ALM_LR

exec bash "$SCRIPT_DIR/queue_qwen4b_blank_beststack_topneg_recover_20260707.sh"
