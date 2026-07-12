#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Diverse-medoid Qwen+GPT-2 blank-target A/B with the strongest old-best
# recovery ingredients restored. This is intended to run after the plain
# diverse-medoid lane so the transport support change is still comparable.
export TAG="${TAG:-qwen4b_gpt2_multidonor_seqall_diversemedoid_oldbestmix_gpu4_20260702}"

export INIT_TRANSPORT_SUPPORT="${INIT_TRANSPORT_SUPPORT:-diverse-medoid}"
export RESIDUAL_MODE="${RESIDUAL_MODE:-lowrank-pre}"
export PRE_RECONSTRUCTION_TARGET="${PRE_RECONSTRUCTION_TARGET:-block-interface}"
export BLOCK_HIDDEN_METRIC="${BLOCK_HIDDEN_METRIC:-both}"
export FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"
export ALM_ATTENTION_INFLUENCE_WEIGHT="${ALM_ATTENTION_INFLUENCE_WEIGHT:-0.04}"
export ALM_ATTENTION_INFLUENCE_LAYER="${ALM_ATTENTION_INFLUENCE_LAYER:-middle}"

exec bash "$SCRIPT_DIR/queue_qwen4b_gpt2_multidonor_seqall_diverse_medoid_20260702.sh"
