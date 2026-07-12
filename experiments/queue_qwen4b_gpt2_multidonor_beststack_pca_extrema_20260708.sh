#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cross-family PCA-extrema transport A/B against the domain-diverse and
# countsketch lanes.
export VARIANT="${VARIANT:-qwen_gpt2}"
export TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_pca_extrema_gpu4_20260708}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.85,0.15}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-pca-extrema-medoid}"
export INIT_ATTENTION_HEAD_DIM_TRANSPORT="${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-qk-vo}"
export ACT_TARGET="${ACT_TARGET:-all}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
