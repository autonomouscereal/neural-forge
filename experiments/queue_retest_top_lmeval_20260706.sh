#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals/retest_matrix_20260705_220140}"
STATE="${STATE:-$EVAL_DIR/queue_state_c39ad2f_poll30.json}"
WAIT_PIDS="${WAIT_PIDS:-}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_THRESHOLD="${MEM_THRESHOLD:-2000}"
IGNORE_GPU_PIDS="${IGNORE_GPU_PIDS:-12681}"
TASKS="${TASKS:-arc_easy,hellaswag,piqa,winogrande}"
LIMIT="${LIMIT:-100}"
MAX_MODELS="${MAX_MODELS:-24}"
TAG="${TAG:-retest_top_lmeval_20260706}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/lm_eval_${TAG}}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/lm_eval_cache_20260702}"
SUMMARY_JSON="${SUMMARY_JSON:-$EVAL_DIR/summary_${TAG}.json}"
SUMMARY_MD="${SUMMARY_MD:-$EVAL_DIR/summary_${TAG}.md}"
MODEL_ENTRIES_FILE="${MODEL_ENTRIES_FILE:-$EVAL_DIR/${TAG}_model_entries.env}"
LM_EVAL_COMMAND="${LM_EVAL_COMMAND:-/home/cereal/ai-benchmark-lab/envs/lm-eval/bin/python -m lm_eval}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$RUN_ROOT/logs" "$EVAL_DIR"

echo "retest_top_lmeval_queue_start=$(date -Is) tag=$TAG wait_pids=${WAIT_PIDS:-none} gpu_uuid=$GPU_UUID revision=$(cat .deployed_revision 2>/dev/null || true)"

for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date -Is) waiting_for_prior_pid=$pid"
    sleep "$POLL_SECONDS"
  done
done

/home/cereal/model-surgery-lab/venv/bin/python experiments/retest_matrix.py summarize \
  --queue-state "$STATE" \
  --out-json "$SUMMARY_JSON" \
  --out-md "$SUMMARY_MD" >/dev/null

/home/cereal/model-surgery-lab/venv/bin/python - "$SUMMARY_JSON" "$MAX_MODELS" "$MODEL_ENTRIES_FILE" "$RUN_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
max_models = int(sys.argv[2])
out_path = Path(sys.argv[3])
run_root = Path(sys.argv[4])
data = json.loads(summary_path.read_text(encoding="utf-8"))
rows = [row for row in data.get("rows", []) if row.get("model_dir")]

def ratio(row, key):
    correct = row.get(f"{key}_correct")
    total = row.get(f"{key}_total")
    if not total:
        return 0.0
    return float(correct or 0) / float(total)

def ppl(row):
    value = row.get("perplexity_mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return value if value == value else float("inf")

def label_slug(label):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("._-")
    return slug[:64] or "candidate"

selected = []
seen = set()

def add(label, model, reason):
    if not label or not model:
        return
    model = str(model)
    key = (label, model)
    if key in seen:
        return
    if model.startswith("/") and not (Path(model) / "config.json").exists() and not Path(model).exists():
        return
    seen.add(key)
    selected.append({"label": label_slug(label), "model": model, "reason": reason})

add("qwen_parent", "Qwen/Qwen3-4B-Base", "baseline")
add("gpt2_donor", "openai-community/gpt2", "baseline")
add("lfm230m_donor", "LiquidAI/LFM2.5-230M", "baseline")
add("blank_random", run_root / "blank_smollm2_135m_seed0", "baseline")

ranked = sorted(
    rows,
    key=lambda row: (
        -ratio(row, "behavior"),
        -ratio(row, "generation"),
        -(row.get("behavior_parent_retention") or 0.0),
        -(row.get("generation_parent_retention") or 0.0),
        ppl(row),
    ),
)
for row in ranked[:12]:
    add(row.get("label"), row.get("model_dir"), "top_behavior")

by_category = {}
for row in ranked:
    category = row.get("category") or "unknown"
    by_category.setdefault(category, row)
for category, row in sorted(by_category.items()):
    add(row.get("label"), row.get("model_dir"), f"best_category_{category}")

low_ppl = sorted(rows, key=ppl)
for row in low_ppl[:8]:
    add(row.get("label"), row.get("model_dir"), "low_ppl_trap_check")

selected = selected[:max_models]
entries = ";".join(f"{item['label']}={item['model']}" for item in selected)
out_path.write_text(
    "MODEL_ENTRIES=" + json.dumps(entries) + "\n"
    + "SELECTED_MODELS_JSON=" + json.dumps(selected, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"selected_count": len(selected), "entries_file": str(out_path), "selected": selected}, indent=2))
PY

source "$MODEL_ENTRIES_FILE"
echo "selected_model_entries=$MODEL_ENTRIES"

while true; do
  used=$(nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || echo 999999)
  busy=$(
    nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader 2>/dev/null |
      awk -F, -v gpu="$GPU_UUID" -v ignore="$IGNORE_GPU_PIDS" '
        BEGIN {
          split(ignore, ignored, /[ ,]+/)
          for (idx in ignored) {
            if (ignored[idx] != "") {
              ignored_pid[ignored[idx]] = 1
            }
          }
        }
        {
          pid = $1
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
          if ($2 ~ gpu && $3 ~ /python/ && !(pid in ignored_pid)) {
            print pid
          }
        }' |
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
echo "retest_top_lmeval_start=$(date -Is) output_dir=$OUTPUT_DIR cache_dir=$CACHE_DIR tasks=$TASKS limit=$LIMIT"
/home/cereal/model-surgery-lab/venv/bin/python experiments/run_lm_eval_comparison.py \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --batch-size auto \
  --dtype float16 \
  --device cuda:0 \
  --lm-eval-command "$LM_EVAL_COMMAND" \
  "${MODEL_ARGS[@]}"
echo "retest_top_lmeval_done=$(date -Is) summary=$OUTPUT_DIR/summary.json"
