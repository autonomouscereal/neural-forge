#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TAG="${TAG:-qwen4b_gpt2_multidonor_seqall_teacherreplay_full_gpu4_20260702}"

exec bash "$SCRIPT_DIR/queue_qwen4b_gpt2_multidonor_seqall_retention_20260702.sh"
