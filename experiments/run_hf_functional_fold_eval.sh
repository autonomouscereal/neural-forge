#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/cereal/model-surgery-lab/neural-forge-src}"
PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the experiment root.}"
TAG="${TAG:?Set TAG to a unique run name.}"
OUT="${OUT:-$RUN_ROOT/$TAG}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:?Set CANDIDATE_MODEL to a completed folded checkpoint.}"
GPU_UUID="${GPU_UUID:?Set GPU_UUID to the physical GPU UUID.}"
FUNCTIONAL_TARGET="${FUNCTIONAL_TARGET:-module-output}"
RIDGE="${RIDGE:-0.001}"
MAX_LENGTH="${MAX_LENGTH:-160}"
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-4096}"
MAX_ANCHORS="${MAX_ANCHORS:-20000}"
MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-1024}"
CALIBRATION_FILE="${CALIBRATION_FILE:-$RUN_ROOT/calibration_mix_plus_recovery_drills_20260702.jsonl}"
FIT_GATE_VALIDATION_FRACTION="${FIT_GATE_VALIDATION_FRACTION:-0.25}"
FIT_GATE_MIN_VALIDATION_ROWS="${FIT_GATE_MIN_VALIDATION_ROWS:-128}"
FIT_GATE_MIN_RELATIVE_IMPROVEMENT="${FIT_GATE_MIN_RELATIVE_IMPROVEMENT:-0.02}"
FIT_GATE_MAX_COSINE_DROP="${FIT_GATE_MAX_COSINE_DROP:-0.0}"
PARENT_EVAL="${PARENT_EVAL:-$RUN_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json}"
EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"

mkdir -p "$OUT/reports"
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "functional_fold_start=$(date -Is) tag=$TAG target=$FUNCTIONAL_TARGET ridge=$RIDGE gpu_uuid=$GPU_UUID"
"$PY" -m neural_forge.cli hf-functional-fold \
  "$SOURCE_MODEL" \
  "$CANDIDATE_MODEL" \
  "$OUT/model" \
  --functional-target "$FUNCTIONAL_TARGET" \
  --ridge "$RIDGE" \
  --max-length "$MAX_LENGTH" \
  --max-tokens-per-layer "$MAX_TOKENS_PER_LAYER" \
  --max-anchors "$MAX_ANCHORS" \
  --fit-gate-validation-fraction "$FIT_GATE_VALIDATION_FRACTION" \
  --fit-gate-min-validation-rows "$FIT_GATE_MIN_VALIDATION_ROWS" \
  --fit-gate-min-relative-improvement "$FIT_GATE_MIN_RELATIVE_IMPROVEMENT" \
  --fit-gate-max-cosine-drop "$FIT_GATE_MAX_COSINE_DROP" \
  --calibration-file "$CALIBRATION_FILE" \
  --max-calibration-texts "$MAX_CALIBRATION_TEXTS" \
  --dtype float16 \
  --candidate-dtype float32 \
  --device cuda:0
echo "functional_fold_done=$(date -Is) tag=$TAG"

REPO_ROOT="$REPO" \
RUN_ROOT="$RUN_ROOT" \
EVAL_DIR="$OUT/reports" \
MODEL_ID="$OUT/model" \
LABEL="$TAG" \
GPU_UUID="$GPU_UUID" \
REPORT="$OUT/reports/eval_1000x1000x250.json" \
MANUAL_JSON="$OUT/reports/manual_generations.json" \
EVAL_SUITE_FILE="$EVAL_SUITE_FILE" \
MAX_BEHAVIOR_PROBES=1000 \
MAX_GENERATION_PROMPTS=1000 \
MAX_CHOICE_PROBES=250 \
MAX_TEXT_EVALS=128 \
MAX_NEW_TOKENS=96 \
bash "$REPO/experiments/queue_hf_eval_model.sh"

if [[ -f "$PARENT_EVAL" && -f "$OUT/reports/eval_1000x1000x250.json" ]]; then
  "$PY" "$REPO/experiments/compare_eval_retention.py" \
    "$PARENT_EVAL" \
    "$OUT/reports/eval_1000x1000x250.json" \
    --out "$OUT/reports/retention_vs_parent.json" \
    --max-samples 12 >/dev/null
fi

echo "functional_fold_eval_done=$(date -Is) tag=$TAG"
