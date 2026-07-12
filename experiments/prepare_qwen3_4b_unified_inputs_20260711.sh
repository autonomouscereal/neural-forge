#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-unified-20260710}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
RUN_ROOT=${RUN_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711}
SPLIT_ROOT=${SPLIT_ROOT:-$RUN_ROOT/splits_v1}

CALIBRATION=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl
EVAL_SUITE=/home/cereal/model-surgery-lab/neural-forge-pivot/runs/qwen3_compensated_fragile_selection_v5c_20260711.json
MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539

test -f "$CALIBRATION"
test -f "$EVAL_SUITE"
test ! -e "$SPLIT_ROOT"
mkdir -p "$RUN_ROOT"

PYTHONPATH="$ROOT/src" "$PY" "$ROOT/experiments/prepare_qwen_unified_splits.py" \
  --calibration-source "$CALIBRATION" \
  --eval-suite-source "$EVAL_SUITE" \
  --output-dir "$SPLIT_ROOT" \
  --seed qwen3-4b-unified-surgery-20260711-v1 \
  --calibration-count 8 \
  --sensitivity-gate-count 8 \
  --candidate-text-count 2 \
  --candidate-choice-count 16 \
  --candidate-generation-count 32 \
  --candidate-behavior-count 32 \
  --transaction-text-count 2 \
  --transaction-choice-count 48 \
  --transaction-generation-count 64 \
  --transaction-behavior-count 64

PYTHONPATH="$ROOT/src" "$PY" - "$ROOT" "$MODEL" "$SPLIT_ROOT" <<'PY'
from pathlib import Path
import json
import sys

from transformers import AutoTokenizer

root = Path(sys.argv[1])
model = sys.argv[2]
split_root = Path(sys.argv[3])
sys.path.insert(0, str(root / "experiments"))
import run_qwen_unified_surgery as runner

tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
calibration_prompts = runner._load_prompts(split_root / "calibration.jsonl", 8)
sensitivity_prompts = runner._load_prompts(split_root / "sensitivity_gate.jsonl", 8)
calibration = runner.encode_prompt_rows(
    tokenizer, calibration_prompts, max_length=256, name="sensitivity-calibration"
)
sensitivity = runner.encode_prompt_rows(
    tokenizer, sensitivity_prompts, max_length=256, name="sensitivity-gate"
)
candidate = runner.bind_evaluation_suite(
    tokenizer,
    split_root / "candidate_validation_suite.json",
    name="candidate-validation",
    max_new_tokens=96,
)
transaction = runner.bind_evaluation_suite(
    tokenizer,
    split_root / "transaction_gate_suite.json",
    name="transaction-gate",
    max_new_tokens=96,
)
report = runner.build_split_provenance(
    calibration,
    sensitivity,
    transaction,
    candidate,
    mode="independent",
)
(split_root / "split_path_provenance.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({
    "all_token_paths_disjoint": report["all_token_paths_disjoint"],
    "provenance_sha256": report["provenance_sha256"],
}, sort_keys=True))
PY

printf 'prepared split_root=%s\n' "$SPLIT_ROOT"
