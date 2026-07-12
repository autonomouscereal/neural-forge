#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-lfm230m_blank_smollm_trainable_byteprobe_teacherreplay_aim004mid_gpu0_20260702}"
export GPU_UUID="${GPU_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
exec bash "$SCRIPT_DIR/queue_lfm230m_blank_smollm_trainable_byteprobe_retention_20260702.sh"
