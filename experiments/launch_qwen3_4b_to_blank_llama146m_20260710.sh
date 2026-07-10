#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-pivot
RUN_ROOT=/mnt/ai_data/neural-forge/cross_family_blank_20260710
TAG=qwen3_4b_to_blank_llama146m_sparsefold_gpu0_r1_20260710
RUN_DIR="$RUN_ROOT/$TAG"
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561

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
  --target-family llama \
  --hidden-size 384 \
  --intermediate-size 768 \
  --num-hidden-layers 24 \
  --num-attention-heads 6 \
  --num-key-value-heads 1 \
  --head-dim 64 \
  --max-position-embeddings 32768 \
  --exact-parameter-budget 146196864 \
  --seed 0 \
  --dtype float16 \
  --device cuda:0 \
  --local-files-only \
  --trace-pooling token_rows \
  --residual-map-method sparse_fold \
  --ridge 1e-5 \
  --min-final-state-improvement 0 \
  --min-logit-improvement 0 \
  --min-top1-gain 0 \
  --max-coordinate-rows 4096 \
  --vocab-projection-chunk-size 2048 \
  --max-new-tokens 96 \
  --max-text-evals 1000 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000 \
  --manual-sample-count 64

echo "launched tag=$TAG run_dir=$RUN_DIR gpu=$GPU"
