#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-unified-20260710}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
RUN_ROOT=${RUN_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711}
SPLIT_ROOT=${SPLIT_ROOT:-$RUN_ROOT/splits_v1}
TAG=${TAG:-p35_mixed_drop16_pareto_v6_20260711}
GPU=${GPU:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}
MAX_CANDIDATE_VALIDATIONS=${MAX_CANDIDATE_VALIDATIONS:-8}
INITIAL_CANDIDATE_VALIDATION_SUMMARY=${INITIAL_CANDIDATE_VALIDATION_SUMMARY:-}

IMPORT_VALIDATION_ARGS=()
if [[ -n "$INITIAL_CANDIDATE_VALIDATION_SUMMARY" ]]; then
  test -e "$INITIAL_CANDIDATE_VALIDATION_SUMMARY"
  IMPORT_VALIDATION_ARGS=(
    --initial-candidate-validation-summary
    "$INITIAL_CANDIDATE_VALIDATION_SUMMARY"
  )
fi

MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
INCUMBENT=${INCUMBENT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT=$RUN_ROOT/$TAG
LOG=$RUN_ROOT/$TAG.log
PID_FILE=$RUN_ROOT/$TAG.pid
IDENTITY_FILE=$RUN_ROOT/$TAG.identity.json

for path in \
  "$SPLIT_ROOT/calibration.jsonl" \
  "$SPLIT_ROOT/sensitivity_gate.jsonl" \
  "$SPLIT_ROOT/candidate_validation_suite.json" \
  "$SPLIT_ROOT/transaction_gate_suite.json"; do
  test -f "$path"
done
test -d "$INCUMBENT"
test ! -e "$OUTPUT"
test ! -e "$PID_FILE"
test ! -e "$IDENTITY_FILE"
mkdir -p "$RUN_ROOT"

"$PY" "$ROOT/experiments/launch_detached.py" \
  --log "$LOG" \
  --pid-file "$PID_FILE" \
  --identity-file "$IDENTITY_FILE" \
  --cwd "$ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$GPU" \
  --env "PYTHONPATH=$ROOT/src" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- \
  "$PY" "$ROOT/experiments/run_qwen_unified_surgery.py" \
  --input-codebook-checkpoint "$INCUMBENT" \
  --tokenizer "$MODEL" \
  --output "$OUTPUT" \
  --checkpoint-artifact-savings-bytes 245760 \
  --calibration-file "$SPLIT_ROOT/calibration.jsonl" \
  --sensitivity-gate-file "$SPLIT_ROOT/sensitivity_gate.jsonl" \
  --candidate-validation-suite "$SPLIT_ROOT/candidate_validation_suite.json" \
  --transaction-gate-suite "$SPLIT_ROOT/transaction_gate_suite.json" \
  --candidate-validation-mode independent \
  --device cuda:0 \
  --expected-layers 36 \
  --max-length 256 \
  --calibration-count 8 \
  --sensitivity-gate-count 8 \
  --neuron-sensitivity-candidates 256 \
  --attention-sensitivity-candidates 128 \
  --sensitivity-candidate-chunk-size 4 \
  --max-candidate-forward-rows 256 \
  --sensitivity-sketch-dim 16 \
  --neuron-high-damage-anchors 4 \
  --attention-high-damage-anchors 4 \
  --max-candidate-validations "$MAX_CANDIDATE_VALIDATIONS" \
  --max-proposals-per-modality 8 \
  --max-batch-size 8 \
  --max-attempts 8 \
  --max-overshoot-bytes 16384 \
  --max-overshoot-ratio 0.10 \
  --max-new-tokens 96 \
  --min-behavior-retention 0.90 \
  --min-choice-retention 0.90 \
  --min-generation-retention 0.90 \
  --max-frontier-checkpoints 3 \
  --local-files-only \
  "${IMPORT_VALIDATION_ARGS[@]}"

printf 'launched tag=%s gpu=%s output=%s log=%s\n' "$TAG" "$GPU" "$OUTPUT" "$LOG"
