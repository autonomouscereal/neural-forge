#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_blank_smollm_cka_weighted_block_weighted_trainable_byteprobe_full_gpu4_20260702}"
export WAIT_PIDS="${WAIT_PIDS:-3720635}"
export BLOCK_HIDDEN_PROJECTION_SCOPE="${BLOCK_HIDDEN_PROJECTION_SCOPE:-layer-input-gated}"
export PRE_RECONSTRUCTION_TARGET="${PRE_RECONSTRUCTION_TARGET:-all}"
export BLOCK_BYTE_HIDDEN_KD_WEIGHT="${BLOCK_BYTE_HIDDEN_KD_WEIGHT:-0.004}"
export BLOCK_BYTE_HIDDEN_PROBE_MODE="${BLOCK_BYTE_HIDDEN_PROBE_MODE:-trainable}"
export BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT="${BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT:-0.002}"

exec bash "$SCRIPT_DIR/queue_qwen4b_blank_smollm_cka_weighted_block_weighted_layerproj_gated_all_20260702.sh"
