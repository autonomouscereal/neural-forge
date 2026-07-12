#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_gpt2_multidonor_seqall_teacherreplay_aim004mid_gpu4_20260702}"
export ALM_ATTENTION_INFLUENCE_WEIGHT="${ALM_ATTENTION_INFLUENCE_WEIGHT:-0.04}"
export ALM_ATTENTION_INFLUENCE_LAYER="${ALM_ATTENTION_INFLUENCE_LAYER:-middle}"

exec bash "$SCRIPT_DIR/queue_qwen4b_gpt2_multidonor_seqall_teacherreplay_now_20260702.sh"
