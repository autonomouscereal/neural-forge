#!/usr/bin/env bash
set -euo pipefail

sharded_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-sharded-v1
script_path="$sharded_root/control/orchestrate.sh"
log_path="$sharded_root/logs/orchestrator.log"
pid_path="$sharded_root/logs/orchestrator.pid"

if [[ -f "$pid_path" ]]; then
  existing_pid="$(tr -dc '0-9' < "$pid_path")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "orchestrator_already_live pid=$existing_pid"
    exit 0
  fi
fi

nohup bash "$script_path" >"$log_path" 2>&1 &
orchestrator_pid=$!
printf '%s\n' "$orchestrator_pid" > "$pid_path"
echo "orchestrator_started pid=$orchestrator_pid log=$log_path"
