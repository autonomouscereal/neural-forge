#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-pivot
RUN_ROOT=/mnt/ai_data/neural-forge/real_medoid_dual_20260710
TAG=${TAG:-qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_gpu4_r2_20260710}
PY=/home/cereal/model-surgery-lab/venv/bin/python
GPU=${GPU:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}
MAX_OPERATOR_DELTA_RATIO=${MAX_OPERATOR_DELTA_RATIO:-4.0}

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
  --calibration-count 256 \
  --validation-count 64 \
  --target-family qwen3 \
  --hidden-size 1536 \
  --intermediate-size 5120 \
  --num-hidden-layers 30 \
  --num-attention-heads 12 \
  --num-key-value-heads 4 \
  --head-dim 128 \
  --max-position-embeddings 32768 \
  --exact-parameter-budget 1130007552 \
  --seed 0 \
  --dtype float32 \
  --donor-dtype float16 \
  --target-dtype float32 \
  --device cuda:0 \
  --local-files-only \
  --trace-pooling last_valid \
  --residual-map-method dual_orthonormal \
  --nonlinear-merge-method real_medoid \
  --ridge 1e-3 \
  --max-operator-delta-ratio "$MAX_OPERATOR_DELTA_RATIO" \
  --min-final-state-improvement 0 \
  --min-logit-improvement 0 \
  --min-top1-gain 0 \
  --max-coordinate-rows 1536 \
  --vocab-projection-chunk-size 2048 \
  --vocab-row-norm-repair none \
  --dual-alpha 1 \
  --dual-beta 1 \
  --dual-gamma 2 \
  --dual-margin-topk 5 \
  --dual-oracle-vocab-rows 4096 \
  --max-new-tokens 96 \
  --max-text-evals 1000 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000 \
  --manual-sample-count 64

echo "launched tag=$TAG gpu=$GPU"
