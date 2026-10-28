#!/usr/bin/env bash
set -euo pipefail

SHA=02cec5c82b6820a77e273af38b78cc0b0880285a
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0004
VALIDATOR=/tmp/validate_lfm2_postfold_publications.py
FRONTIER_LAUNCHER=/tmp/launch_terminal_frontier_lanes_a0004.sh
REVIEW_LAUNCHER=/tmp/launch_whole_response_review_lanes_a0004.sh
POLL_SECONDS=30

LANES=(fit_guided full_strength_structural residual_writer)

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

require_runtime_inputs() {
  [[ -d "$CODE" ]] || fail "runtime is missing: $CODE"
  [[ -x "$PYTHON" ]] || fail "Python runtime is not executable: $PYTHON"
  [[ -s "$VALIDATOR" ]] || fail "validator is missing: $VALIDATOR"
  [[ -s "$FRONTIER_LAUNCHER" ]] || fail \
    "terminal-frontier launcher is missing: $FRONTIER_LAUNCHER"
  [[ -s "$REVIEW_LAUNCHER" ]] || fail \
    "whole-response launcher is missing: $REVIEW_LAUNCHER"
  [[ -d "$BASE" ]] || fail "a0004 attempt root is missing: $BASE"
}

pid_is_expected_process() {
  local pid=$1
  local marker=$2
  local cmdline

  kill -0 "$pid" 2>/dev/null || return 1
  if [[ ! -r "/proc/$pid/cmdline" ]]; then
    return 0
  fi
  cmdline=$(tr '\000' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ -n "$cmdline" && "$cmdline" == *"$marker"* ]]
}

poll_stage() {
  local stage=$1
  local subdirectory=$2
  local pid_name=$3
  local artifact_name=$4
  local process_marker=$5
  local lane
  local root
  local pidfile
  local artifact
  local pid
  local state
  local all_complete
  local states

  while true; do
    all_complete=1
    states=()
    for lane in "${LANES[@]}"; do
      root="$BASE/$lane"
      if [[ -n "$subdirectory" ]]; then
        root="$root/$subdirectory"
      fi
      pidfile="$root/$pid_name"
      artifact="$root/$artifact_name"

      [[ -s "$pidfile" ]] || fail "$stage $lane PID file is missing: $pidfile"
      IFS= read -r pid <"$pidfile" || fail "$stage $lane PID cannot be read"
      [[ "$pid" =~ ^[1-9][0-9]*$ ]] || fail "$stage $lane PID is invalid: $pid"

      if pid_is_expected_process "$pid" "$process_marker"; then
        state=running
        all_complete=0
      elif [[ -s "$artifact" ]]; then
        state=complete
      else
        fail "$stage $lane PID $pid exited without $artifact_name"
      fi
      states+=("$lane=$state")
    done

    log "$stage ${states[*]}"
    if (( all_complete )); then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

validate_postfold_receipts() {
  local output
  local status

  if output=$(env \
    PYTHONPATH="$CODE/src" \
    PYTHONHASHSEED=0 \
    "$PYTHON" -u "$VALIDATOR" \
    --base-directory "$BASE" 2>&1); then
    log "postfold structural validator succeeded"
  else
    status=$?
    printf '%s\n' "$output" >&2
    fail "postfold structural validator failed with exit $status"
  fi
}

invoke_launcher() {
  local stage=$1
  local launcher=$2
  local output
  local status
  local summary

  if output=$(bash "$launcher" 2>&1); then
    summary=$(printf '%s' "$output" | tr '\n' ';')
    log "$stage launcher succeeded${summary:+: $summary}"
  else
    status=$?
    printf '%s\n' "$output" >&2
    fail "$stage launcher failed with exit $status"
  fi
}

require_runtime_inputs
log "watching a0004 pipeline at $BASE"

poll_stage \
  body \
  "" \
  postfold.pid \
  postfold-receipt.json \
  run_lfm2_semantic_postfold.py

validate_postfold_receipts
invoke_launcher terminal-frontier "$FRONTIER_LAUNCHER"

poll_stage \
  terminal-frontier \
  terminal-frontier \
  frontier.pid \
  frontier-receipt.json \
  run_lfm2_semantic_terminal_frontier.py

invoke_launcher whole-response-review "$REVIEW_LAUNCHER"

poll_stage \
  whole-response-review \
  terminal-frontier \
  review.pid \
  whole-response-dossier.json \
  review_lfm2_semantic_frontier.py

log "a0004 pipeline completed with three nonempty whole-response dossiers"
exit 0
