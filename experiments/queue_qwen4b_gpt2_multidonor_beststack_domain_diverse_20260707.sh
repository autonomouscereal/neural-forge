#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A/B against the current Qwen+GPT-2 quantile-medoid beststack lane. This uses
# the same recovery stack and switches the structural support selector to
# domain-diverse real donor medoids.
export VARIANT="${VARIANT:-qwen_gpt2}"
export TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_seqall_domain_diverse_gpu4_20260707}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export WAIT_PIDS="${WAIT_PIDS:-0}"
export POLL_SECONDS="${POLL_SECONDS:-120}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-domain-diverse-medoid}"
export SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.85,0.15}"
export RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
export FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"
export LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
