#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_blank_smollm_cka_weighted_block_weighted_layerproj_gated_full_gpu4_20260702}"
export WAIT_PIDS="${WAIT_PIDS:-3511265}"
export BLOCK_HIDDEN_PROJECTION_SCOPE="${BLOCK_HIDDEN_PROJECTION_SCOPE:-layer-input-gated}"

exec bash "$SCRIPT_DIR/queue_qwen4b_blank_smollm_cka_weighted_block_weighted_balanced_qmedoid_20260702.sh"
