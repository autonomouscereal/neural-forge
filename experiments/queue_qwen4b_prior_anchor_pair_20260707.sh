#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
POLL_SECONDS="${POLL_SECONDS:-60}"
UPSTREAM_WAIT_PIDS="${WAIT_PIDS:-}"
QUEUE_DUALALM29="${QUEUE_DUALALM29:-1}"
QUEUE_FFN1496="${QUEUE_FFN1496:-1}"
DUALALM29_TAG="${DUALALM29_TAG:-qwen4b_dualalm29_anchor_topneg_exact_gpu6_20260707}"
FFN1496_TAG="${FFN1496_TAG:-qwen4b_ffn1496_strict_anchor_topneg_gpu6_20260707}"

mkdir -p "$RUN_ROOT/logs"

append_pid() {
  local existing="$1"
  local pid="$2"
  if [[ -z "$existing" ]]; then
    echo "$pid"
  else
    echo "$existing $pid"
  fi
}

launch_prior_anchor_variant() {
  local variant="$1"
  local tag="$2"
  local wait_pids="$3"
  local log="$RUN_ROOT/logs/${tag}.queue.log"
  local pid_file="$RUN_ROOT/logs/${tag}.queue.pid"

  echo "prior_anchor_pair_launch=$(date -Is) variant=$variant tag=$tag wait_pids=${wait_pids:-none} gpu=$GPU_UUID" | tee -a "$log"
  VARIANT="$variant" \
  TAG="$tag" \
  WAIT_PIDS="$wait_pids" \
  GPU_UUID="$GPU_UUID" \
  POLL_SECONDS="$POLL_SECONDS" \
  REPO_ROOT="$REPO_ROOT" \
  RUN_ROOT="$RUN_ROOT" \
    nohup bash "$SCRIPT_DIR/queue_qwen4b_prior_anchor_topneg_recover_20260707.sh" >> "$log" 2>&1 &
  local pid="$!"
  printf '%s\n' "$pid" > "$pid_file"
  echo "prior_anchor_pair_pid variant=$variant tag=$tag pid=$pid log=$log pid_file=$pid_file"
}

CURRENT_WAIT_PIDS="$UPSTREAM_WAIT_PIDS"

if [[ "$QUEUE_DUALALM29" = "1" ]]; then
  launch_prior_anchor_variant "dualalm29" "$DUALALM29_TAG" "$CURRENT_WAIT_PIDS"
  DUAL_PID="$(cat "$RUN_ROOT/logs/${DUALALM29_TAG}.queue.pid")"
  CURRENT_WAIT_PIDS="$(append_pid "$CURRENT_WAIT_PIDS" "$DUAL_PID")"
fi

if [[ "$QUEUE_FFN1496" = "1" ]]; then
  launch_prior_anchor_variant "ffn1496" "$FFN1496_TAG" "$CURRENT_WAIT_PIDS"
fi
