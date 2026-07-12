#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pure fold-only rare-direction lane for comparison with volume support.
export VARIANT="${VARIANT:-qwen_lfm}"
export TAG="${TAG:-qwen4b_lfm230m_beststack_pca_operator_initonly_gpu0_20260709}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.70,0.30}"
export GPU_UUID="${GPU_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
export STAGES="${STAGES:-init}"
export INIT_OPERATOR_PRODUCT_AXIS_WEIGHTS="${INIT_OPERATOR_PRODUCT_AXIS_WEIGHTS:-1}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-pca-extrema-medoid}"
export INIT_ATTENTION_HEAD_DIM_TRANSPORT="${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-qk-vo}"
export ACT_TARGET="${ACT_TARGET:-all}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
