#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals}"
MODEL_ID="${MODEL_ID:?Set MODEL_ID to the HF model id or checkpoint path to evaluate.}"
LABEL="${LABEL:-$(echo "$MODEL_ID" | tr -cs 'A-Za-z0-9_.-' '_')}"
WAIT_PIDS="${WAIT_PIDS:-}"
GPU_UUID="${GPU_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
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
REPORT="${REPORT:-$EVAL_DIR/eval_${LABEL}_exactgen_20260702_${MAX_BEHAVIOR_PROBES}x${MAX_GENERATION_PROMPTS}x${MAX_CHOICE_PROBES}.json}"
MANUAL_JSON="${MANUAL_JSON:-$EVAL_DIR/manual_${LABEL}_exactgen_20260702.json}"

cd "$REPO_ROOT"
mkdir -p "$EVAL_DIR" "$RUN_ROOT/logs"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "hf_eval_queue_start=$(date -Is) label=$LABEL model=$MODEL_ID wait_pids=${WAIT_PIDS:-none} gpu_uuid=$GPU_UUID revision=$(cat .deployed_revision 2>/dev/null || true)"

for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date -Is) waiting_for_prior_pid=$pid"
    sleep "$POLL_SECONDS"
  done
done

valid_eval_report() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  /home/cereal/model-surgery-lab/venv/bin/python - "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
if "behavior_suite" not in data or "choice_accuracy" not in data:
    raise SystemExit(1)
PY
}

wait_for_gpu() {
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
}

if valid_eval_report "$REPORT"; then
  echo "eval_cache_hit=$REPORT"
else
  wait_for_gpu
  echo "eval_start=$(date -Is) label=$LABEL report=$REPORT suite=$EVAL_SUITE_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" /home/cereal/model-surgery-lab/venv/bin/python -m neural_forge.cli hf-eval \
    "$MODEL_ID" \
    "$REPORT" \
    --eval-suite "$EVAL_SUITE_FILE" \
    --max-behavior-probes "$MAX_BEHAVIOR_PROBES" \
    --max-generation-prompts "$MAX_GENERATION_PROMPTS" \
    --max-choice-probes "$MAX_CHOICE_PROBES" \
    --max-text-evals "$MAX_TEXT_EVALS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
  echo "eval_done=$(date -Is) report=$REPORT"
fi

if [[ -s "$MANUAL_JSON" ]]; then
  echo "manual_cache_hit=$MANUAL_JSON"
else
  wait_for_gpu
  echo "manual_start=$(date -Is) label=$LABEL manual=$MANUAL_JSON"
  CUDA_VISIBLE_DEVICES="$GPU_UUID" /home/cereal/model-surgery-lab/venv/bin/python - "$MODEL_ID" "$MANUAL_JSON" "$DEVICE" "$DTYPE" <<'PY'
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, out_path, device, dtype_name = sys.argv[1:5]
prompts = [
    "Q: What is 6+4? A:",
    "Q: If a box has 3 red balls and 2 blue balls, how many balls are there? A:",
    "Write a Python function that adds two numbers.",
    "Briefly explain tokenizer mismatch in cross-family model merging.",
    "What should happen before pruning neurons in a merge-first compression pipeline?",
    "A person hears thunder after seeing lightning. What should they do?",
    "Explain one risk when merging two neural models.",
]
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
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
    json.dumps({"model": model_id, "manual_generations": items}, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({"manual_eval": out_path}, indent=2))
PY
  echo "manual_done=$(date -Is) manual=$MANUAL_JSON"
fi
