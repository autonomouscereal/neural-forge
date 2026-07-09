#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the checkpoint directory to evaluate.}"
TAG="${TAG:-$(basename "$MODEL_DIR")}"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-200}"
MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"
MAX_TEXT_EVALS="${MAX_TEXT_EVALS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite.json}"
FORCE_EVAL="${FORCE_EVAL:-0}"
FORCE_MANUAL="${FORCE_MANUAL:-0}"
CPU_THREAD_CAP="${CPU_THREAD_CAP:-4}"
TORCH_INTEROP_THREAD_CAP="${TORCH_INTEROP_THREAD_CAP:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREAD_CAP}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREAD_CAP}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREAD_CAP}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREAD_CAP}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREAD_CAP}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPU_THREAD_CAP}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-$TORCH_INTEROP_THREAD_CAP}"

cd "$REPO_ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate

OUT_DIR="${OUT_DIR:-$RUN_ROOT/evals}"
mkdir -p "$OUT_DIR"
BROAD_JSON="${BROAD_JSON:-$OUT_DIR/eval_${TAG}_${MAX_BEHAVIOR_PROBES}x${MAX_GENERATION_PROMPTS}x${MAX_CHOICE_PROBES}.json}"
MANUAL_JSON="${MANUAL_JSON:-$OUT_DIR/manual_${TAG}.json}"

valid_eval_report() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
if "behavior_suite" not in data or "choice_accuracy" not in data:
    raise SystemExit(1)
PY
}

if [[ "$FORCE_EVAL" != "1" ]] && valid_eval_report "$BROAD_JSON"; then
  echo "broad_eval_cache_hit=$BROAD_JSON"
else
  CUDA_VISIBLE_DEVICES="$GPU_UUID" python -m neural_forge.cli hf-eval \
    "$MODEL_DIR" \
    "$BROAD_JSON" \
    --eval-suite "$EVAL_SUITE_FILE" \
    --max-behavior-probes "$MAX_BEHAVIOR_PROBES" \
    --max-generation-prompts "$MAX_GENERATION_PROMPTS" \
    --max-choice-probes "$MAX_CHOICE_PROBES" \
    --max-text-evals "$MAX_TEXT_EVALS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
fi

if [[ "$FORCE_MANUAL" != "1" ]] && [[ -s "$MANUAL_JSON" ]]; then
  echo "manual_eval_cache_hit=$MANUAL_JSON"
else
  CUDA_VISIBLE_DEVICES="$GPU_UUID" python - <<PY
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "$MODEL_DIR"
out = Path("$MANUAL_JSON")
prompts = [
    "Q: What is 6+4? A:",
    "Write a Python function that adds two numbers.",
    "Explain one risk when merging two neural models.",
    "A person hears thunder after seeing lightning. What should they do?",
    "In model surgery, why is behavior testing better than only checking perplexity?",
    "Briefly explain tokenizer mismatch in cross-family model merging.",
    "What should happen before pruning neurons in a merge-first compression pipeline?",
]

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if "$DTYPE" == "float16" else torch.float32,
    device_map={"": "$DEVICE"},
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
items = []
for prompt in prompts:
    encoded = tokenizer(prompt, return_tensors="pt").to("$DEVICE")
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
out.write_text(json.dumps({"model": model_dir, "manual_generations": items}, indent=2), encoding="utf-8")
print(json.dumps({"broad_eval": "$BROAD_JSON", "manual_eval": "$MANUAL_JSON"}, indent=2))
PY
fi
