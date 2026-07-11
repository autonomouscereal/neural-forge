#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-pivot}
RUN_ROOT=${RUN_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
GPU=${GPU:?Set GPU to a physical GPU UUID.}
TARGET_UNIQUE_PARAMETERS=${TARGET_UNIQUE_PARAMETERS:?Set the exact target parameter count.}
TAG=${TAG:-qwen3_4b_compensated_target${TARGET_UNIQUE_PARAMETERS}_r1_20260710}
CHECKPOINT_STAGING_ROOT=${CHECKPOINT_STAGING_ROOT:-/home/cereal/model-surgery-lab/neural-forge-checkpoint-staging}
MERGES_PER_TRANSACTION=${MERGES_PER_TRANSACTION:-64}
PROPOSAL_LAYER_LIMIT=${PROPOSAL_LAYER_LIMIT:-4}
FULL_MILESTONE_INTERVAL=${FULL_MILESTONE_INTERVAL:-8}
FOLD_STRATEGY=${FOLD_STRATEGY:-pair}
SELECTION_EVAL_SUITE=${SELECTION_EVAL_SUITE:?Set a token-disjoint adaptive-selection suite.}
PARENT_SELECTION_EVAL=${PARENT_SELECTION_EVAL:?Set its exact identity-bound cached parent evaluation.}
PARENT_CERTIFICATION_EVAL=${PARENT_CERTIFICATION_EVAL:?Set the exact identity-bound 1000/1000/250 parent evaluation.}
MANUAL_REVIEW_TIMEOUT_SECONDS=${MANUAL_REVIEW_TIMEOUT_SECONDS:-86400}

MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
CALIBRATION=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl
EVAL_SUITE=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json
SOURCE_REVISION=906bfd4b4dc7f14ee4320094d8b41684abff8539

mkdir -p "$RUN_ROOT" "$CHECKPOINT_STAGING_ROOT"

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
  "$PY" "$ROOT/experiments/run_qwen_compensated_neuron_fold.py" \
  --source-model "$MODEL" \
  --source-tokenizer "$MODEL" \
  --source-revision "$SOURCE_REVISION" \
  --output-dir "$RUN_ROOT/$TAG" \
  --calibration-jsonl "$CALIBRATION" \
  --eval-suite "$EVAL_SUITE" \
  --cached-parent-eval "$PARENT_CERTIFICATION_EVAL" \
  --selection-eval-suite "$SELECTION_EVAL_SUITE" \
  --cached-parent-selection-eval "$PARENT_SELECTION_EVAL" \
  --target-unique-parameters "$TARGET_UNIQUE_PARAMETERS" \
  --checkpoint-staging-root "$CHECKPOINT_STAGING_ROOT" \
  --calibration-count 96 \
  --validation-count 96 \
  --calibration-max-length 160 \
  --activation-rows 256 \
  --anchor-fraction 0.05 \
  --minimum-anchor-count 64 \
  --projection-count 8 \
  --neighbor-window 4 \
  --max-pair-candidates 32768 \
  --max-generated-pairs 65536 \
  --score-chunk-size 512 \
  --proposal-layer-limit "$PROPOSAL_LAYER_LIMIT" \
  --prefilter-columns 256 \
  --fold-strategy "$FOLD_STRATEGY" \
  --multi-anchor-candidate-pool-size 512 \
  --multi-anchor-max-anchors 4 \
  --multi-anchor-similarity-chunk-size 64 \
  --multi-anchor-max-similarity-products 16000000 \
  --multi-anchor-max-projection-fits 131072 \
  --merges-per-transaction "$MERGES_PER_TRANSACTION" \
  --max-transactions 128 \
  --full-milestone-interval "$FULL_MILESTONE_INTERVAL" \
  --gate-text-count 16 \
  --gate-behavior-count 64 \
  --gate-choice-count 48 \
  --gate-generation-count 64 \
  --selection-max-text-evals 32 \
  --selection-max-choice-probes 128 \
  --selection-max-generation-prompts 256 \
  --selection-max-behavior-probes 256 \
  --max-gate-behavior-losses 0 \
  --max-gate-choice-losses 0 \
  --max-gate-generation-losses 0 \
  --max-full-behavior-losses 0 \
  --max-full-choice-losses 0 \
  --max-full-generation-losses 0 \
  --dtype float16 \
  --device cuda:0 \
  --max-new-tokens 96 \
  --max-text-evals 1000 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000 \
  --manual-sample-count 128 \
  --manual-review-verdict-file "$RUN_ROOT/$TAG/manual_review_verdict.json" \
  --manual-review-timeout-seconds "$MANUAL_REVIEW_TIMEOUT_SECONDS" \
  --reload-equivalence-generation-tokens 8 \
  --max-host-activation-bytes 8589934592 \
  --max-host-rss-bytes 274877906944 \
  --max-cuda-allocated-bytes 24696061952 \
  --max-cuda-reserved-bytes 24696061952 \
  --max-wall-seconds 172800 \
  --max-artifact-bytes 17179869184 \
  --max-history-records 256 \
  --max-history-json-bytes 67108864 \
  --local-files-only

echo "launched tag=$TAG gpu=$GPU target=$TARGET_UNIQUE_PARAMETERS"
