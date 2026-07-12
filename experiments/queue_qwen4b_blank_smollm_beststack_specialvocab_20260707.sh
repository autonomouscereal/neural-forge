#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VARIANT="${VARIANT:-qwen}"
export TAG="${TAG:-qwen4b_blank_smollm_beststack_specialvocab_qrsinkhorn_gpu4_20260707}"
export GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
export WAIT_PIDS="${WAIT_PIDS:-1431696}"
export POLL_SECONDS="${POLL_SECONDS:-120}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}"
export TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-quantile-medoid}"
export FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"
export LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"

exec bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
