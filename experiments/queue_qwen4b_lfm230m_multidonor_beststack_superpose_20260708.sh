#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First-principles superposition fold. This disables activation-PCA projection
# for init so signed-superpose assignment directly packs donor axes into the
# blank child. The bet is controlled superposition beats similarity clustering
# when the compression ratio is high.
export VARIANT="${VARIANT:-qwen_lfm}"
export TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_superpose_gpu6_20260708}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.70,0.30}"
export GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export INIT_FOLD_MODE="${INIT_FOLD_MODE:-mean}"
export INIT_HIDDEN_BASIS_MODE="${INIT_HIDDEN_BASIS_MODE:-none}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-sparse}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-centroid}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
