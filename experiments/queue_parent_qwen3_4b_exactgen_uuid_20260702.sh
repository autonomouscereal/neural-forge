#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
TAG="${TAG:-parent_qwen3_4b_base_exactgen_uuid_20260702}"
WAIT_PIDS="${WAIT_PIDS:-}"
MEM_THRESHOLD="${MEM_THRESHOLD:-2000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

SUITE="${SUITE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
REPORT="${REPORT:-$RUN_ROOT/evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/evals" "$RUN_ROOT/logs"

echo "queue_start=$(date -Is) tag=$TAG gpu_uuid=$GPU_UUID wait_pids=${WAIT_PIDS:-none} revision=$(cat .deployed_revision 2>/dev/null || true)"

for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date -Is) waiting_for_prior_pid=$pid"
    sleep "$POLL_SECONDS"
  done
done

if [ -s "$REPORT" ]; then
  if /home/cereal/model-surgery-lab/venv/bin/python - "$REPORT" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
if "behavior_suite" not in data or "choice_accuracy" not in data:
    raise SystemExit(1)
PY
  then
    echo "already_done report=$REPORT"
    exit 0
  fi
  invalid="$REPORT.invalid.$(date +%Y%m%dT%H%M%S)"
  mv "$REPORT" "$invalid"
  echo "moved_invalid_report=$invalid"
fi

while true; do
  used=$(nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || echo 999999)
  busy=$(
    nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader 2>/dev/null |
      awk -F, -v gpu="$GPU_UUID" '$2 ~ gpu && $3 ~ /python/ {print $1}' |
      wc -l
  )
  if [ -n "$used" ] && [ "$used" -le "$MEM_THRESHOLD" ] && [ "$busy" = "0" ]; then
    break
  fi
  echo "$(date -Is) waiting_for_gpu_uuid=$GPU_UUID memory_used_mb=${used:-unknown} busy_python=$busy threshold_mb=$MEM_THRESHOLD"
  sleep "$POLL_SECONDS"
done

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
echo "eval_start=$(date -Is) tag=$TAG model=$MODEL report=$REPORT suite=$SUITE gpu_uuid=$GPU_UUID revision=$(cat .deployed_revision 2>/dev/null || true)"
/home/cereal/model-surgery-lab/venv/bin/python -m neural_forge.cli hf-eval "$MODEL" "$REPORT" \
  --dtype float16 \
  --device cuda:0 \
  --max-new-tokens 96 \
  --eval-suite "$SUITE" \
  --max-text-evals 128 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000
echo "eval_done=$(date -Is) tag=$TAG report=$REPORT"
