#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_blank_smollm_balanced_signed_cka_qmedoid_lrcguard_full_gpu4_20260702}"
export WAIT_PIDS="${WAIT_PIDS:-3325882}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT=balanced-signed-similarity
export INIT_TRANSPORT_SUPPORT=quantile-medoid
export INIT_FOLD_STAT_REPAIR=rms
export LRC_MAX_DELTA_RMS_RATIO=1.5
export BLOCK_NONFINITE_STEP_POLICY=skip-step
export BLOCK_MAX_NONFINITE_STEPS_PER_LAYER=3

exec bash "$SCRIPT_DIR/queue_qwen4b_blank_smollm_signed_cka_rms_lrcguard_geomean_20260702.sh"
