#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/cereal/model-surgery-lab/neural-forge-src}"
PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$REPO/runs/qwen4b_to_smollm135m_transport_random_20260627}"
TARGET_MODEL="${TARGET_MODEL:-$RUN_ROOT/blank_qwen3_146m_h512_i2048_l18_h4kv1_seed0}"
TAG="${TAG:-qwen4b_to_blank_qwen3_146m_path_conjugate_initonly_20260709}"
OUT="${OUT:-$RUN_ROOT/$TAG}"
GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-volume-medoid}"
HIDDEN_BASIS_MODE="${HIDDEN_BASIS_MODE:-activation-coordinate}"
DONOR_LAYER_FOLD_MODE="${DONOR_LAYER_FOLD_MODE:-representative}"
DONOR_LAYER_OUTPUT_SCALE="${DONOR_LAYER_OUTPUT_SCALE:-unit}"
PARENT_EVAL="${PARENT_EVAL:-$RUN_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json}"

mkdir -p "$OUT/reports"
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -m neural_forge.cli hf-donor-project-init "$OUT/model" \
  --target-config-model "$TARGET_MODEL" \
  --target-base-model "$TARGET_MODEL" \
  --target-tokenizer-model "$TARGET_MODEL" \
  --donor-model Qwen/Qwen3-4B-Base \
  --seed 0 \
  --max-anchors 151936 \
  --fold-mode path-conjugate \
  --hidden-basis-mode "$HIDDEN_BASIS_MODE" \
  --hidden-basis-scope global \
  --hidden-basis-max-rows 4096 \
  --activation-fold-assignment balanced-similarity \
  --fold-stat-repair none \
  --transport-backend qr-sinkhorn \
  --transport-support "$TRANSPORT_SUPPORT" \
  --donor-layer-fold-mode "$DONOR_LAYER_FOLD_MODE" \
  --donor-layer-output-scale "$DONOR_LAYER_OUTPUT_SCALE" \
  --donor-layer-group-strategy activation-cka-weighted \
  --attention-fold-mode activation-coupled \
  --attention-head-dim-transport none \
  --token-init-mode shared \
  --token-fallback-mode none \
  --projection-scale none \
  --projection-blend-alpha 1.0 \
  --consumer-hidden-reduction sum \
  --operator-product-axis-weights \
  --activation-feature-mode covariance \
  --activation-subspace-rank 12 \
  --activation-domain-balance keyword \
  --activation-calibration-file "$RUN_ROOT/calibration_mix_plus_recovery_drills_20260702.jsonl" \
  --activation-max-calibration-texts 1024 \
  --activation-max-length 160 \
  --activation-device cuda:0 \
  --post-init-eval-suite "$RUN_ROOT/behavior_suite_exactgen_20260702.json" \
  --post-init-eval-output "$OUT/reports/eval_1000x1000x250.json" \
  --post-init-max-text-evals 64 \
  --post-init-max-choice-probes 250 \
  --post-init-max-generation-prompts 1000 \
  --post-init-max-behavior-probes 1000 \
  --post-init-max-new-tokens 96 \
  --post-init-eval-device cuda:0 \
  --post-init-eval-dtype float16 \
  --dtype float16

if [[ -f "$PARENT_EVAL" && -f "$OUT/reports/eval_1000x1000x250.json" ]]; then
  "$PY" "$REPO/experiments/compare_eval_retention.py" \
    "$PARENT_EVAL" \
    "$OUT/reports/eval_1000x1000x250.json" \
    --out "$OUT/reports/retention_vs_qwen_parent.json" \
    --max-samples 12 >/dev/null
fi
