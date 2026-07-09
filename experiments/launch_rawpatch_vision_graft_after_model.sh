#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
MODEL_DIR="${MODEL_DIR:-$RUN_ROOT/qwen4b_mathdrill_preserve_hard_s300_gpu0}"
TAG="${TAG:-rawpatch_graft_after_text_model_gpu6_20260701}"
TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-4}"
TEXT_RETENTION_WEIGHT="${TEXT_RETENTION_WEIGHT:-1.0}"
STEPS="${STEPS:-600}"

cd "$REPO_ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate
mkdir -p "$RUN_ROOT/logs"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "missing_model_dir=$MODEL_DIR" >&2
  exit 0
fi

OUT="$RUN_ROOT/$TAG"
LOG="$RUN_ROOT/logs/$TAG.log"
PID="$RUN_ROOT/logs/$TAG.pid"

CUDA_VISIBLE_DEVICES="$GPU_UUID" nohup python -m neural_forge.cli hf-vision-graft \
  "$MODEL_DIR" \
  "$OUT" \
  --steps "$STEPS" \
  --batch-size 8 \
  --eval-batches 24 \
  --train-last-layers "$TRAIN_LAST_LAYERS" \
  --text-retention-file "$RUN_ROOT/calibration_mix.jsonl" \
  --text-retention-weight "$TEXT_RETENTION_WEIGHT" \
  --text-retention-batch-size 2 \
  --text-retention-max-length 128 \
  --device cuda:0 \
  --dtype float16 > "$LOG" 2>&1 &

echo $! > "$PID"
echo "$TAG"
echo "model=$MODEL_DIR"
echo "pid=$(cat "$PID")"
echo "log=$LOG"
