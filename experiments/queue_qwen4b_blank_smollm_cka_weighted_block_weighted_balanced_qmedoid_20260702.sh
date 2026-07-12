#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_blank_smollm_cka_weighted_block_weighted_balanced_qmedoid_full_gpu4_20260702}"
export WAIT_PIDS="${WAIT_PIDS:-3452698}"
export INIT_DONOR_LAYER_GROUP_STRATEGY=activation-cka-weighted
export BLOCK_DONOR_LAYER_GROUP_STRATEGY=activation-cka-weighted

exec bash "$SCRIPT_DIR/queue_qwen4b_blank_smollm_balanced_signed_cka_qmedoid_lrcguard_20260702.sh"
