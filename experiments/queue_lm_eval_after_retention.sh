#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
TAG="${TAG:?Set TAG to the candidate tag for the lm-eval slice.}"
MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
WAIT_PIDS="${WAIT_PIDS:-}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
MEM_THRESHOLD="${MEM_THRESHOLD:-2000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TASKS="${TASKS:-arc_easy,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-50}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda:0}"
LM_EVAL_COMMAND="${LM_EVAL_COMMAND:-/home/cereal/ai-benchmark-lab/envs/lm-eval/bin/python -m lm_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/lm_eval_${TAG}_20260702}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/lm_eval_cache_20260702}"
MODEL_ENTRIES="${MODEL_ENTRIES:-child=$MODEL_DIR;qwen_parent=Qwen/Qwen3-4B-Base;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;cleanopen_l4=$RUN_ROOT/qwen4b_cleanopen_anchor_newloss_l4_gpu0_20260701;mathdrill_l4=$RUN_ROOT/qwen4b_mathdrill_anchor_newloss_l4_gpu6_20260701}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$RUN_ROOT/logs"

echo "lm_eval_queue_start=$(date -Is) tag=$TAG wait_pids=${WAIT_PIDS:-none} gpu_uuid=$GPU_UUID revision=$(cat .deployed_revision 2>/dev/null || true)"

for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date -Is) waiting_for_prior_pid=$pid"
    sleep "$POLL_SECONDS"
  done
done

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "missing_model_dir=$MODEL_DIR" >&2
  exit 2
fi

while true; do
  used=$(nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || echo 999999)
  busy=$(
    nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader 2>/dev/null |
      awk -F, -v gpu="$GPU_UUID" '$2 ~ gpu && $3 ~ /python/ {print $1}' |
      wc -l
  )
  if [[ -n "$used" && "$used" -le "$MEM_THRESHOLD" && "$busy" = "0" ]]; then
    break
  fi
  echo "$(date -Is) waiting_for_gpu_uuid=$GPU_UUID memory_used_mb=${used:-unknown} busy_python=$busy threshold_mb=$MEM_THRESHOLD"
  sleep "$POLL_SECONDS"
done

MODEL_ARGS=()
IFS=';' read -ra ENTRIES <<< "$MODEL_ENTRIES"
for entry in "${ENTRIES[@]}"; do
  [[ -n "$entry" ]] || continue
  label="${entry%%=*}"
  model="${entry#*=}"
  if [[ "$label" == "$model" || -z "$label" || -z "$model" ]]; then
    echo "invalid_model_entry=$entry" >&2
    exit 3
  fi
  if [[ "$model" == /* && ! -e "$model" ]]; then
    echo "skip_missing_model_entry=$entry"
    continue
  fi
  MODEL_ARGS+=(--model "$label=$model")
done

if [[ "${#MODEL_ARGS[@]}" -eq 0 ]]; then
  echo "no_valid_model_entries" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
echo "lm_eval_start=$(date -Is) tag=$TAG output_dir=$OUTPUT_DIR cache_dir=$CACHE_DIR tasks=$TASKS limit=$LIMIT"
/home/cereal/model-surgery-lab/venv/bin/python experiments/run_lm_eval_comparison.py \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --lm-eval-command "$LM_EVAL_COMMAND" \
  "${MODEL_ARGS[@]}"
echo "lm_eval_done=$(date -Is) tag=$TAG summary=$OUTPUT_DIR/summary.json"
