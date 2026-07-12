#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First-principles A/B: deterministic CountSketch signed hashing for donor-axis
# collisions. This keeps the beststack recovery stack and uses centroid
# transport so the countsketch assignment/signs define the folded basis.
export VARIANT="${VARIANT:-qwen_lfm}"
export TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_countsketch_gpu0_20260708}"
export GPU_UUID="${GPU_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
export WAIT_PIDS="${WAIT_PIDS:-0}"
export POLL_SECONDS="${POLL_SECONDS:-120}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-centroid}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-countsketch}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.70,0.30}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"
export LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
