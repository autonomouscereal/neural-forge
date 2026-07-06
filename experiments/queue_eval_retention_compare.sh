#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals}"
TAG="${TAG:?Set TAG to the completed run tag to evaluate.}"
MODEL_DIR_EXPLICIT="${MODEL_DIR+x}"
MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/$TAG}"
QUEUE_PID="${QUEUE_PID:-0}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-}"
PARENT_WAIT_PIDS="${PARENT_WAIT_PIDS:-}"
GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
MEM_THRESHOLD="${MEM_THRESHOLD:-2000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-1000}"
MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"
MAX_TEXT_EVALS="${MAX_TEXT_EVALS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
REPORT="${REPORT:-$EVAL_DIR/eval_${TAG}_exactgen_20260702_${MAX_BEHAVIOR_PROBES}x${MAX_GENERATION_PROMPTS}x${MAX_CHOICE_PROBES}.json}"
MANUAL_JSON="${MANUAL_JSON:-$EVAL_DIR/manual_${TAG}_exactgen_20260702.json}"
OUT_DIR="${OUT_DIR:-$EVAL_DIR/retention_${TAG}_20260702}"
PARENT_REPORT="${PARENT_REPORT:-}"
PARENT_REPORT_CANDIDATES="${PARENT_REPORT_CANDIDATES:-$EVAL_DIR/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json;$EVAL_DIR/eval_parent_qwen3_4b_base_1000g_20260701_1000x1000x250.json}"
BASELINE_REPORTS="${BASELINE_REPORTS:-blank_random=$EVAL_DIR/eval_blank_smollm135m_random_1000g_20260701_1000x1000x250.json;cleanopen_l4=$EVAL_DIR/eval_cleanopen_l4_alm_1000g_20260701_1000x1000x250.json;mathdrill_l4=$EVAL_DIR/eval_mathdrill_l4_alm_1000g_20260701_1000x1000x250.json;cleanopen_full=$EVAL_DIR/eval_cleanopen_full_02_alm_exactgen_20260702_1000x1000x250.json;topneg_margin=$EVAL_DIR/eval_qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702_exactgen_1000x1000x250.json}"

cd "$REPO_ROOT"
mkdir -p "$EVAL_DIR" "$OUT_DIR" "$RUN_ROOT/logs"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "retention_queue_start=$(date -Is) tag=$TAG queue_pid=$QUEUE_PID queue_pid_file=${QUEUE_PID_FILE:-none} gpu_uuid=$GPU_UUID revision=$(cat .deployed_revision 2>/dev/null || true)"

wait_for_pids() {
  local label="$1"
  shift || true
  for pid in "$@"; do
    if [[ -z "$pid" || "$pid" == "0" ]]; then
      continue
    fi
    while kill -0 "$pid" 2>/dev/null; do
      echo "$(date -Is) waiting_for_${label}_pid=$pid"
      sleep "$POLL_SECONDS"
    done
  done
}

wait_for_pid_file() {
  local path="$1"
  local run_pid=""
  if [[ -z "$path" ]]; then
    return 0
  fi
  while true; do
    if [[ -s "$path" ]]; then
      run_pid=$(tr -dc '0-9\n' < "$path" | head -n 1 || true)
      if [[ "$run_pid" =~ ^[0-9]+$ && "$run_pid" != "0" ]]; then
        echo "$(date -Is) selected_run_pid_from_file=$run_pid path=$path"
        wait_for_pids "run" "$run_pid"
        return 0
      fi
    fi
    echo "$(date -Is) waiting_for_run_pid_file=$path"
    sleep "$POLL_SECONDS"
  done
}

valid_eval_report() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  /home/cereal/model-surgery-lab/venv/bin/python - "$path" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
if "behavior_suite" not in data or "choice_accuracy" not in data:
    raise SystemExit(1)
PY
}

resolve_model_dir() {
  local candidate="$1"
  if [[ -f "$candidate/config.json" ]]; then
    echo "$candidate"
    return 0
  fi
  local summary="$candidate/reports/chain_anchor_summary.json"
  if [[ -s "$summary" ]]; then
    local resolved
    if resolved=$(/home/cereal/model-surgery-lab/venv/bin/python - "$summary" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
data = json.loads(summary.read_text(encoding="utf-8"))
for key in ("final_model_dir", "best_model_dir"):
    value = data.get(key)
    if value and (Path(value) / "config.json").exists():
        print(value)
        raise SystemExit(0)
raise SystemExit(1)
PY
    ); then
      echo "$resolved"
      return 0
    fi
  fi
  echo "$candidate"
}

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || echo 999999)
    busy=$(
      nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader 2>/dev/null |
        awk -F, -v gpu="$GPU_UUID" -v ignore="${IGNORE_GPU_PIDS:-}" '
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
}

safe_label() {
  echo "$1" | tr -cs 'A-Za-z0-9_.-' '_'
}

wait_for_pids "upstream" $QUEUE_PID
wait_for_pid_file "$QUEUE_PID_FILE"

MODEL_DIR="$(resolve_model_dir "$MODEL_DIR")"
echo "resolved_model_dir=$MODEL_DIR explicit_model_dir=${MODEL_DIR_EXPLICIT:-0}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "missing_model_dir=$MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "missing_model_config=$MODEL_DIR/config.json" >&2
  exit 2
fi

if valid_eval_report "$REPORT"; then
  echo "child_eval_cache_hit=$REPORT"
else
  wait_for_gpu
  echo "child_eval_start=$(date -Is) tag=$TAG model=$MODEL_DIR report=$REPORT suite=$EVAL_SUITE_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" /home/cereal/model-surgery-lab/venv/bin/python -m neural_forge.cli hf-eval \
    "$MODEL_DIR" \
    "$REPORT" \
    --eval-suite "$EVAL_SUITE_FILE" \
    --max-behavior-probes "$MAX_BEHAVIOR_PROBES" \
    --max-generation-prompts "$MAX_GENERATION_PROMPTS" \
    --max-choice-probes "$MAX_CHOICE_PROBES" \
    --max-text-evals "$MAX_TEXT_EVALS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
  echo "child_eval_done=$(date -Is) report=$REPORT"
fi

if [[ ! -s "$MANUAL_JSON" ]]; then
  wait_for_gpu
  echo "manual_eval_start=$(date -Is) tag=$TAG manual=$MANUAL_JSON"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" /home/cereal/model-surgery-lab/venv/bin/python - "$MODEL_DIR" "$MANUAL_JSON" "$DEVICE" "$DTYPE" <<'PY'
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_path, device, dtype_name = sys.argv[1:5]
prompts = [
    "Q: What is 6+4? A:",
    "Q: If a box has 3 red balls and 2 blue balls, how many balls are there? A:",
    "Write a Python function that adds two numbers.",
    "Briefly explain tokenizer mismatch in cross-family model merging.",
    "What should happen before pruning neurons in a merge-first compression pipeline?",
    "A person hears thunder after seeing lightning. What should they do?",
    "Explain one risk when merging two neural models.",
]
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if dtype_name == "float16" else torch.float32,
    device_map={"": device},
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
items = []
for prompt in prompts:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    items.append({"prompt": prompt, "output": text})
Path(out_path).write_text(
    json.dumps({"model": model_dir, "manual_generations": items}, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({"manual_eval": out_path}, indent=2))
PY
  echo "manual_eval_done=$(date -Is) manual=$MANUAL_JSON"
else
  echo "manual_eval_cache_hit=$MANUAL_JSON"
fi

wait_for_pids "parent" $PARENT_WAIT_PIDS

SELECTED_PARENT=$(/home/cereal/model-surgery-lab/venv/bin/python - "$PARENT_REPORT" "$PARENT_REPORT_CANDIDATES" <<'PY'
import json
import sys
from pathlib import Path

explicit = sys.argv[1].strip()
candidates = []
if explicit:
    candidates.append(explicit)
candidates.extend(item for item in sys.argv[2].split(";") if item.strip())
for raw in candidates:
    path = Path(raw)
    if not path.exists() or path.stat().st_size <= 0:
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if "behavior_suite" in data and "choice_accuracy" in data:
        print(path)
        raise SystemExit(0)
raise SystemExit(1)
PY
) || SELECTED_PARENT=""

if [[ -n "$SELECTED_PARENT" ]]; then
  echo "selected_parent_report=$SELECTED_PARENT"
  /home/cereal/model-surgery-lab/venv/bin/python experiments/compare_eval_retention.py \
    "$SELECTED_PARENT" \
    "$REPORT" \
    --out "$OUT_DIR/parent_vs_child.json" \
    --max-samples 12
else
  echo "missing_valid_parent_report candidates=$PARENT_REPORT_CANDIDATES"
fi

IFS=';' read -ra BASELINES <<< "$BASELINE_REPORTS"
for item in "${BASELINES[@]}"; do
  [[ -n "$item" ]] || continue
  label="${item%%=*}"
  path="${item#*=}"
  [[ "$label" != "$path" ]] || label="$(basename "$path" .json)"
  if valid_eval_report "$path"; then
    out_label=$(safe_label "$label")
    /home/cereal/model-surgery-lab/venv/bin/python experiments/compare_eval_retention.py \
      "$path" \
      "$REPORT" \
      --out "$OUT_DIR/${out_label}_vs_child.json" \
      --max-samples 8
    echo "baseline_compare_done label=$label path=$path"
  else
    echo "baseline_compare_skipped label=$label missing_or_invalid=$path"
  fi
done

/home/cereal/model-surgery-lab/venv/bin/python - "$OUT_DIR" "$TAG" "$REPORT" "$MANUAL_JSON" "$SELECTED_PARENT" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary = {
    "operation": "eval_retention_compare",
    "tag": sys.argv[2],
    "child_report": sys.argv[3],
    "manual_report": sys.argv[4],
    "selected_parent_report": sys.argv[5],
    "comparisons": {},
}
for path in sorted(out_dir.glob("*_vs_child.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary["comparisons"][path.stem] = {
        "path": str(path),
        "parent": data.get("parent"),
        "child": data.get("child"),
        "param_compression_ratio": data.get("param_compression_ratio"),
        "behavior": data.get("behavior", {}).get("overall"),
        "generation": data.get("generation", {}).get("overall"),
        "choice": data.get("choice", {}).get("overall"),
        "manual_behavior_samples": data.get("manual_samples", {}).get("behavior", [])[:3],
        "manual_generation_samples": data.get("manual_samples", {}).get("generation", [])[:3],
    }
summary_path = out_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "retention_queue_done=$(date -Is) tag=$TAG out_dir=$OUT_DIR"
