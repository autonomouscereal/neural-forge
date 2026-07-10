#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-pivot
RUN_ROOT=/mnt/ai_data/neural-forge/cross_family_blank_20260710
TAG=${TAG:-qwen3_4b_to_blank_llama88m_structuralprior_gpu0_r3_20260710}
RUN_DIR="$RUN_ROOT/$TAG"
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=${GPU:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}
HIDDEN_SIZE=${HIDDEN_SIZE:-384}
INTERMEDIATE_SIZE=${INTERMEDIATE_SIZE:-768}
TARGET_FAMILY=${TARGET_FAMILY:-llama}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-24}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-6}
NUM_KEY_VALUE_HEADS=${NUM_KEY_VALUE_HEADS:-1}
HEAD_DIM=${HEAD_DIM:-64}
EXACT_PARAMETER_BUDGET=${EXACT_PARAMETER_BUDGET:-87853440}
MAX_OPERATOR_DELTA_RATIO=${MAX_OPERATOR_DELTA_RATIO:-4}
RESIDUAL_MAP_METHOD=${RESIDUAL_MAP_METHOD:-sparse_fold}
VOCAB_ROW_NORM_REPAIR=${VOCAB_ROW_NORM_REPAIR:-none}

mkdir -p "$RUN_ROOT"

"$PY" "$ROOT/experiments/launch_detached.py" \
  --log "$RUN_ROOT/$TAG.log" \
  --pid-file "$RUN_ROOT/$TAG.pid" \
  --identity-file "$RUN_ROOT/$TAG.identity.json" \
  --cwd "$ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$GPU" \
  --env "PYTHONPATH=$ROOT/src" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- \
  "$PY" "$ROOT/experiments/launch_cross_family_blank_run.py" \
  --source-model /home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --run-root "$RUN_ROOT" \
  --tag "$TAG" \
  --calibration-file /home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl \
  --eval-suite /home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json \
  --cached-parent-eval /home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json \
  --calibration-count 64 \
  --validation-count 16 \
  --target-family "$TARGET_FAMILY" \
  --hidden-size "$HIDDEN_SIZE" \
  --intermediate-size "$INTERMEDIATE_SIZE" \
  --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
  --num-attention-heads "$NUM_ATTENTION_HEADS" \
  --num-key-value-heads "$NUM_KEY_VALUE_HEADS" \
  --head-dim "$HEAD_DIM" \
  --max-position-embeddings 32768 \
  --exact-parameter-budget "$EXACT_PARAMETER_BUDGET" \
  --seed 0 \
  --dtype float32 \
  --donor-dtype float16 \
  --target-dtype float32 \
  --device cuda:0 \
  --local-files-only \
  --trace-pooling token_rows \
  --residual-map-method "$RESIDUAL_MAP_METHOD" \
  --ridge 1e-3 \
  --max-operator-delta-ratio "$MAX_OPERATOR_DELTA_RATIO" \
  --min-final-state-improvement 0 \
  --min-logit-improvement 0 \
  --min-top1-gain 0 \
  --max-coordinate-rows 4096 \
  --vocab-projection-chunk-size 2048 \
  --vocab-row-norm-repair "$VOCAB_ROW_NORM_REPAIR" \
  --max-new-tokens 96 \
  --max-text-evals 1000 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000 \
  --manual-sample-count 64

echo "launched tag=$TAG run_dir=$RUN_DIR gpu=$GPU"
