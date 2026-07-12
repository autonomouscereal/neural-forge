#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' GPU_STATUS
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

printf '%s\n' NEURAL_FORGE_PROCESSES
pgrep -af 'neural-forge|qwen3_4b|run_qwen' || true

printf '%s\n' REPO_STATUS
for directory in \
  /home/cereal/model-surgery-lab/neural-forge-pivot \
  /home/cereal/model-surgery-lab/neural-forge-src; do
  if [[ -d "$directory/.git" ]]; then
    head=$(git -C "$directory" rev-parse --short HEAD)
    branch=$(git -C "$directory" branch --show-current)
    printf 'repo=%s head=%s branch=%s\n' "$directory" "$head" "$branch"
    git -C "$directory" status --short --untracked-files=no | head -80
  fi
done

printf '%s\n' WATCHED_RUNS
base=/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710
for tag in \
  qwen3_4b_pair_drop1_layer35_dev_reused_eval_20260710 \
  qwen3_4b_pair_drop1_layer18_dev_reused_eval_20260710; do
  directory="$base/$tag"
  printf 'tag=%s\n' "$tag"
  if [[ -f "$directory/run_state.json" ]]; then
    python3 - "$directory/run_state.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(payload, sort_keys=True))
PY
  fi
  find "$directory" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort
  if [[ -f "$directory/transaction_report.json" ]]; then
    python3 - "$directory/transaction_report.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
keys = ("status", "target", "accepted_count", "rejected_count", "stop_reason")
print(json.dumps({key: payload.get(key) for key in keys}, sort_keys=True))
PY
  fi
done

printf '%s\n' DISK
df -h /home/cereal/model-surgery-lab /mnt/ai_data
