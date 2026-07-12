#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"

export TAG="${TAG:-donor_gpt2_exactgen_uuid_20260702}"
export MODEL="${MODEL:-openai-community/gpt2}"
export REPORT="${REPORT:-$RUN_ROOT/evals/eval_donor_gpt2_exactgen_20260702_1000x1000x250.json}"
export GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
export WAIT_PIDS="${WAIT_PIDS:-3345256}"

exec bash "$SCRIPT_DIR/queue_parent_qwen3_4b_exactgen_uuid_20260702.sh"
