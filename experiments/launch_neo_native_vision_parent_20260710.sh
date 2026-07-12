#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-src
RUN_ROOT=/mnt/ai_data/neural-forge/native_vision_parent_20260710
TAG=${TAG:-neo1p5_2b_sft_probe256_bf16_gpu0_r6_eagerprefill_tf4573_20260710}
GPU=${GPU:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}
PY=/home/cereal/model-surgery-lab/venv/bin/python
OUTPUT_DIR="$RUN_ROOT/$TAG"

mkdir -p "$RUN_ROOT"

"$PY" "$ROOT/experiments/launch_detached.py" \
  --log "$RUN_ROOT/$TAG.log" \
  --pid-file "$RUN_ROOT/$TAG.pid" \
  --identity-file "$RUN_ROOT/$TAG.identity.json" \
  --cwd "$ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$GPU" \
  --env "PYTHONPATH=/mnt/ai_data/neural-forge/python_overlays/transformers_4_57_3:$ROOT/src" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- \
  "$PY" "$ROOT/experiments/eval_neo_native_vision.py" \
  --manifest /mnt/ai_data/neural-forge/native_vision_probe_v1_20260710/manifest.json \
  --manifest-sha256 76152804adba32da5e51b4ffccf5b9fe18c9b079dec8a6ee72553e88e1ac4811 \
  --model-snapshot /mnt/ai_data/neural-forge/native_parent_cache_20260710/neo1_5_2b_materialized_105970970eccd7cddf105224a4930342c31733e4 \
  --output-dir "$OUTPUT_DIR" \
  --dtype bf16 \
  --device cuda:0 \
  --max-new-tokens 16 \
  --seed 0 \
  --manual-samples-per-group 4

echo "launched tag=$TAG output_dir=$OUTPUT_DIR gpu=$GPU"
