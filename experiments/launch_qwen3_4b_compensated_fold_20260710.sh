#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
RUN_ROOT=${RUN_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
GPU=${GPU:?Set GPU to a physical GPU UUID.}
if [[ ! "$GPU" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]; then
  echo "GPU must be an exact physical GPU UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)" >&2
  exit 2
fi
PREPARE_PARENT_CACHES=${PREPARE_PARENT_CACHES:-0}
TARGET_UNIQUE_PARAMETERS=${TARGET_UNIQUE_PARAMETERS:-}
FOLD_STRATEGY=${FOLD_STRATEGY:-pair}
COMPUTE_SHAPE_MODE=${COMPUTE_SHAPE_MODE:-compact}
MAX_TRANSIENT_COMPUTE_BYTES=${MAX_TRANSIENT_COMPUTE_BYTES:-2147483648}
if [[ "$FOLD_STRATEGY" == "multi_anchor" ]]; then
  FOLD_STRATEGY=multi-anchor
fi
case "$FOLD_STRATEGY" in
  pair|multi-anchor) ;;
  *)
    echo "FOLD_STRATEGY must be pair or multi-anchor" >&2
    exit 2
    ;;
esac
case "$COMPUTE_SHAPE_MODE" in
  compact|source-padded) ;;
  *)
    echo "COMPUTE_SHAPE_MODE must be compact or source-padded" >&2
    exit 2
    ;;
esac
if [[ "$PREPARE_PARENT_CACHES" != "1" && -z "$TARGET_UNIQUE_PARAMETERS" ]]; then
  echo "TARGET_UNIQUE_PARAMETERS is required unless PREPARE_PARENT_CACHES=1" >&2
  exit 2
fi
TAG=${TAG:-qwen3_4b_compensated_${TARGET_UNIQUE_PARAMETERS:-parent_caches}_${FOLD_STRATEGY}_r1_20260710}
CHECKPOINT_STAGING_ROOT=${CHECKPOINT_STAGING_ROOT:-/home/cereal/model-surgery-lab/neural-forge-checkpoint-staging}
MERGES_PER_TRANSACTION=${MERGES_PER_TRANSACTION:-64}
PROPOSAL_LAYER_LIMIT=${PROPOSAL_LAYER_LIMIT:-4}
FULL_MILESTONE_INTERVAL=${FULL_MILESTONE_INTERVAL:-8}
CALIBRATION_COUNT=${CALIBRATION_COUNT:-96}
VALIDATION_COUNT=${VALIDATION_COUNT:-96}
CALIBRATION_MAX_LENGTH=${CALIBRATION_MAX_LENGTH:-160}
ACTIVATION_ROWS=${ACTIVATION_ROWS:-256}
ANCHOR_FRACTION=${ANCHOR_FRACTION:-0.05}
MINIMUM_ANCHOR_COUNT=${MINIMUM_ANCHOR_COUNT:-64}
EXCLUDE_LAYERS=${EXCLUDE_LAYERS:-}
PROJECTION_COUNT=${PROJECTION_COUNT:-8}
NEIGHBOR_WINDOW=${NEIGHBOR_WINDOW:-4}
MAX_PAIR_CANDIDATES=${MAX_PAIR_CANDIDATES:-32768}
MAX_GENERATED_PAIRS=${MAX_GENERATED_PAIRS:-65536}
SCORE_CHUNK_SIZE=${SCORE_CHUNK_SIZE:-512}
PREFILTER_COLUMNS=${PREFILTER_COLUMNS:-256}
MULTI_ANCHOR_CANDIDATE_POOL_SIZE=${MULTI_ANCHOR_CANDIDATE_POOL_SIZE:-512}
MULTI_ANCHOR_MAX_ANCHORS=${MULTI_ANCHOR_MAX_ANCHORS:-4}
MULTI_ANCHOR_SIMILARITY_CHUNK_SIZE=${MULTI_ANCHOR_SIMILARITY_CHUNK_SIZE:-64}
MULTI_ANCHOR_MAX_SIMILARITY_PRODUCTS=${MULTI_ANCHOR_MAX_SIMILARITY_PRODUCTS:-16000000}
MULTI_ANCHOR_MAX_PROJECTION_FITS=${MULTI_ANCHOR_MAX_PROJECTION_FITS:-131072}
MULTI_ANCHOR_RIDGE=${MULTI_ANCHOR_RIDGE:-0.000001}
MULTI_ANCHOR_MAX_COEFFICIENT_L1=${MULTI_ANCHOR_MAX_COEFFICIENT_L1:-16.0}
MAX_CANDIDATE_ATTEMPTS=${MAX_CANDIDATE_ATTEMPTS:-128}
MAX_TRANSACTIONS=${MAX_TRANSACTIONS:-128}
SELECTION_MAX_TEXT_EVALS=${SELECTION_MAX_TEXT_EVALS:-32}
SELECTION_MAX_CHOICE_PROBES=${SELECTION_MAX_CHOICE_PROBES:-128}
SELECTION_MAX_GENERATION_PROMPTS=${SELECTION_MAX_GENERATION_PROMPTS:-256}
SELECTION_MAX_BEHAVIOR_PROBES=${SELECTION_MAX_BEHAVIOR_PROBES:-256}
MAX_GATE_BEHAVIOR_LOSSES=${MAX_GATE_BEHAVIOR_LOSSES:-0}
MAX_GATE_CHOICE_LOSSES=${MAX_GATE_CHOICE_LOSSES:-0}
MAX_GATE_GENERATION_LOSSES=${MAX_GATE_GENERATION_LOSSES:-0}
MAX_FULL_BEHAVIOR_LOSSES=${MAX_FULL_BEHAVIOR_LOSSES:-0}
MAX_FULL_CHOICE_LOSSES=${MAX_FULL_CHOICE_LOSSES:-0}
MAX_FULL_GENERATION_LOSSES=${MAX_FULL_GENERATION_LOSSES:-0}
SELECTION_EVAL_SUITE=${SELECTION_EVAL_SUITE:?Set a token-disjoint adaptive-selection suite.}
PARENT_SELECTION_EVAL=${PARENT_SELECTION_EVAL:?Set its exact identity-bound cached parent evaluation.}
PARENT_CERTIFICATION_EVAL=${PARENT_CERTIFICATION_EVAL:?Set the exact identity-bound 1000/1000/250 parent evaluation.}
MANUAL_REVIEW_TIMEOUT_SECONDS=${MANUAL_REVIEW_TIMEOUT_SECONDS:-86400}

MODEL=${MODEL:-/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
CALIBRATION=${CALIBRATION:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl}
EVAL_SUITE=${EVAL_SUITE:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json}
SOURCE_REVISION=${SOURCE_REVISION:-906bfd4b4dc7f14ee4320094d8b41684abff8539}

mkdir -p "$RUN_ROOT" "$CHECKPOINT_STAGING_ROOT"

MODE_ARGS=()
if [[ "$PREPARE_PARENT_CACHES" == "1" ]]; then
  MODE_ARGS+=(--prepare-parent-caches)
else
  MODE_ARGS+=(--target-unique-parameters "$TARGET_UNIQUE_PARAMETERS")
fi

EXCLUDE_ARGS=()
if [[ -n "$EXCLUDE_LAYERS" ]]; then
  IFS=',' read -r -a EXCLUDED <<< "$EXCLUDE_LAYERS"
  for layer in "${EXCLUDED[@]}"; do
    if [[ ! "$layer" =~ ^[0-9]+$ ]]; then
      echo "EXCLUDE_LAYERS must be a comma-separated list of non-negative integers" >&2
      exit 2
    fi
    EXCLUDE_ARGS+=(--exclude-layer "$layer")
  done
fi

"$PY" "$ROOT/experiments/launch_detached.py" \
  --log "$RUN_ROOT/$TAG.log" \
  --pid-file "$RUN_ROOT/$TAG.pid" \
  --identity-file "$RUN_ROOT/$TAG.identity.json" \
  --cwd "$ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$GPU" \
  --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
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
  "${MODE_ARGS[@]}" \
  --checkpoint-staging-root "$CHECKPOINT_STAGING_ROOT" \
  --calibration-count "$CALIBRATION_COUNT" \
  --validation-count "$VALIDATION_COUNT" \
  --calibration-max-length "$CALIBRATION_MAX_LENGTH" \
  --activation-rows "$ACTIVATION_ROWS" \
  --anchor-fraction "$ANCHOR_FRACTION" \
  --minimum-anchor-count "$MINIMUM_ANCHOR_COUNT" \
  --projection-count "$PROJECTION_COUNT" \
  --neighbor-window "$NEIGHBOR_WINDOW" \
  --max-pair-candidates "$MAX_PAIR_CANDIDATES" \
  --max-generated-pairs "$MAX_GENERATED_PAIRS" \
  --score-chunk-size "$SCORE_CHUNK_SIZE" \
  --proposal-layer-limit "$PROPOSAL_LAYER_LIMIT" \
  --prefilter-columns "$PREFILTER_COLUMNS" \
  "${EXCLUDE_ARGS[@]}" \
  --fold-strategy "$FOLD_STRATEGY" \
  --compute-shape-mode "$COMPUTE_SHAPE_MODE" \
  --max-transient-compute-bytes "$MAX_TRANSIENT_COMPUTE_BYTES" \
  --multi-anchor-candidate-pool-size "$MULTI_ANCHOR_CANDIDATE_POOL_SIZE" \
  --multi-anchor-max-anchors "$MULTI_ANCHOR_MAX_ANCHORS" \
  --multi-anchor-similarity-chunk-size "$MULTI_ANCHOR_SIMILARITY_CHUNK_SIZE" \
  --multi-anchor-max-similarity-products "$MULTI_ANCHOR_MAX_SIMILARITY_PRODUCTS" \
  --multi-anchor-max-projection-fits "$MULTI_ANCHOR_MAX_PROJECTION_FITS" \
  --multi-anchor-ridge "$MULTI_ANCHOR_RIDGE" \
  --multi-anchor-max-coefficient-l1 "$MULTI_ANCHOR_MAX_COEFFICIENT_L1" \
  --merges-per-transaction "$MERGES_PER_TRANSACTION" \
  --max-candidate-attempts "$MAX_CANDIDATE_ATTEMPTS" \
  --max-transactions "$MAX_TRANSACTIONS" \
  --full-milestone-interval "$FULL_MILESTONE_INTERVAL" \
  --gate-text-count 16 \
  --gate-behavior-count 64 \
  --gate-choice-count 48 \
  --gate-generation-count 64 \
  --selection-max-text-evals "$SELECTION_MAX_TEXT_EVALS" \
  --selection-max-choice-probes "$SELECTION_MAX_CHOICE_PROBES" \
  --selection-max-generation-prompts "$SELECTION_MAX_GENERATION_PROMPTS" \
  --selection-max-behavior-probes "$SELECTION_MAX_BEHAVIOR_PROBES" \
  --max-gate-behavior-losses "$MAX_GATE_BEHAVIOR_LOSSES" \
  --max-gate-choice-losses "$MAX_GATE_CHOICE_LOSSES" \
  --max-gate-generation-losses "$MAX_GATE_GENERATION_LOSSES" \
  --max-full-behavior-losses "$MAX_FULL_BEHAVIOR_LOSSES" \
  --max-full-choice-losses "$MAX_FULL_CHOICE_LOSSES" \
  --max-full-generation-losses "$MAX_FULL_GENERATION_LOSSES" \
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

echo "launched tag=$TAG gpu=$GPU prepare_parent_caches=$PREPARE_PARENT_CACHES target=$TARGET_UNIQUE_PARAMETERS fold_strategy=$FOLD_STRATEGY compute_shape_mode=$COMPUTE_SHAPE_MODE"
