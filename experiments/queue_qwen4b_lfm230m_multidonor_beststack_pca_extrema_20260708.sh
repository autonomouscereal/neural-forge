#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PCA-extrema transport keeps the current transport/recovery stack, but chooses
# real donor axes at principal activation extremes before filling coverage. This
# tests whether rare boundary directions are being averaged out by medoid-only
# selectors.
export VARIANT="${VARIANT:-qwen_lfm}"
export TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_pca_extrema_gpu6_20260708}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.70,0.30}"
export GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-pca-extrema-medoid}"
export INIT_ATTENTION_HEAD_DIM_TRANSPORT="${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-qk-vo}"
export ACT_TARGET="${ACT_TARGET:-all}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
