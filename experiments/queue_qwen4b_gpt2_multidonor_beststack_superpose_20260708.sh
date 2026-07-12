#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same pure superposition fold as the Qwen+LFM lane, but with a tokenizer and
# architecture-mismatched GPT-2 donor to test whether the collision strategy
# survives a more hostile cross-family setup.
export VARIANT="${VARIANT:-qwen_gpt2}"
export TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_superpose_gpu4_20260708}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.85,0.15}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export INIT_FOLD_MODE="${INIT_FOLD_MODE:-mean}"
export INIT_HIDDEN_BASIS_MODE="${INIT_HIDDEN_BASIS_MODE:-none}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-signed-superpose}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-sparse}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-centroid}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
