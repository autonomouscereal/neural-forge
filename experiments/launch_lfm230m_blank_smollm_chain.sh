#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
SOURCE_MODEL="${SOURCE_MODEL:-LiquidAI/LFM2.5-230M}"
ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/blank_smollm2_135m_seed0}"
TAG="${TAG:-lfm230m_blank_smollm_chain_true_lrc_rank16_gpu6_20260629}"
ACT_TARGET="${ACT_TARGET:-mlp-output}"
LRC_ALPHA="${LRC_ALPHA:-0.25}"
LRC_RANK="${LRC_RANK:-16}"
LRC_FACTOR_STEPS="${LRC_FACTOR_STEPS:-40}"
RESIDUAL_MODE="${RESIDUAL_MODE:-lowrank-pre}"
RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.01}"
RESIDUAL_RANK="${RESIDUAL_RANK:-16}"
INIT_PROJECTION_SCALE="${INIT_PROJECTION_SCALE:-target-rms}"
INIT_PROJECTION_BLEND_ALPHA="${INIT_PROJECTION_BLEND_ALPHA:-1.0}"
INIT_TRANSPORT_SUPPORT="${INIT_TRANSPORT_SUPPORT:-centroid}"
BLOCK_STEPS_PER_LAYER="${BLOCK_STEPS_PER_LAYER:-4}"
BLOCK_LR="${BLOCK_LR:-2e-6}"
BLOCK_ALM_KD_WEIGHT="${BLOCK_ALM_KD_WEIGHT:-0.0}"
BLOCK_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_PROJECTED_VOCAB_KD_WEIGHT:-0.0}"
BLOCK_BYTE_KD_WEIGHT="${BLOCK_BYTE_KD_WEIGHT:-0.0}"
BLOCK_BLOCK_IO_KD_WEIGHT="${BLOCK_BLOCK_IO_KD_WEIGHT:-0.02}"
BLOCK_ON_POLICY_ANCHOR_KD_WEIGHT="${BLOCK_ON_POLICY_ANCHOR_KD_WEIGHT:-0.0}"
BLOCK_ON_POLICY_UNLIKELIHOOD_WEIGHT="${BLOCK_ON_POLICY_UNLIKELIHOOD_WEIGHT:-0.0}"
BLOCK_ON_POLICY_HIDDEN_KD_WEIGHT="${BLOCK_ON_POLICY_HIDDEN_KD_WEIGHT:-0.0}"
BLOCK_ON_POLICY_MAX_NEW_TOKENS="${BLOCK_ON_POLICY_MAX_NEW_TOKENS:-24}"
BLOCK_ON_POLICY_INTERVAL="${BLOCK_ON_POLICY_INTERVAL:-10}"
BLOCK_ON_POLICY_RESPONSE_LOSS_MODE="${BLOCK_ON_POLICY_RESPONSE_LOSS_MODE:-auto}"
BLOCK_ON_POLICY_UNLIKELIHOOD_WINDOW="${BLOCK_ON_POLICY_UNLIKELIHOOD_WINDOW:-16}"
BLOCK_STEPWISE_DONOR_KD_WEIGHT="${BLOCK_STEPWISE_DONOR_KD_WEIGHT:-0.0}"
BLOCK_STEPWISE_DONOR_MAX_NEW_TOKENS="${BLOCK_STEPWISE_DONOR_MAX_NEW_TOKENS:-8}"
BLOCK_STEPWISE_DONOR_INTERVAL="${BLOCK_STEPWISE_DONOR_INTERVAL:-4}"
BLOCK_TEACHER_GENERATE_LM_WEIGHT="${BLOCK_TEACHER_GENERATE_LM_WEIGHT:-0.0}"
BLOCK_TEACHER_GENERATE_HIDDEN_KD_WEIGHT="${BLOCK_TEACHER_GENERATE_HIDDEN_KD_WEIGHT:-0.0}"
BLOCK_TEACHER_GENERATE_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_TEACHER_GENERATE_PROJECTED_VOCAB_KD_WEIGHT:-0.0}"
BLOCK_TEACHER_GENERATE_BYTE_KD_WEIGHT="${BLOCK_TEACHER_GENERATE_BYTE_KD_WEIGHT:-0.0}"
BLOCK_TEACHER_GENERATE_MAX_NEW_TOKENS="${BLOCK_TEACHER_GENERATE_MAX_NEW_TOKENS:-64}"
BLOCK_ALM_MAX_CHUNKS="${BLOCK_ALM_MAX_CHUNKS:-64}"
BLOCK_ALM_MIN_CHARS="${BLOCK_ALM_MIN_CHARS:-2}"
BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT="${BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT:-0.0}"
BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT:-0.0}"
BLOCK_FOCUSED_LM_WEIGHT="${BLOCK_FOCUSED_LM_WEIGHT:-0.0}"
BLOCK_TOKEN_FOCUS="${BLOCK_TOKEN_FOCUS:-auto}"
BLOCK_PROJECTED_VOCAB_KD_MODE="${BLOCK_PROJECTED_VOCAB_KD_MODE:-teacher-to-student}"
BLOCK_BYTE_HIDDEN_KD_WEIGHT="${BLOCK_BYTE_HIDDEN_KD_WEIGHT:-0.0}"
BLOCK_BLOCK_IO_RIDGE="${BLOCK_BLOCK_IO_RIDGE:-1e-3}"
BLOCK_BLOCK_IO_LOSS="${BLOCK_BLOCK_IO_LOSS:-both}"
BLOCK_MODULE_OUTPUT_KD_WEIGHT="${BLOCK_MODULE_OUTPUT_KD_WEIGHT:-0.02}"
BLOCK_MODULE_OUTPUT_TARGETS="${BLOCK_MODULE_OUTPUT_TARGETS:-both}"
BLOCK_MODULE_OUTPUT_TRANSPORT="${BLOCK_MODULE_OUTPUT_TRANSPORT:-both}"
BLOCK_MODULE_OUTPUT_TEMPERATURE="${BLOCK_MODULE_OUTPUT_TEMPERATURE:-1.0}"
BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS="${BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS:-12}"
BLOCK_PROJECTED_VOCAB_MAX_SOURCE_TOKENS="${BLOCK_PROJECTED_VOCAB_MAX_SOURCE_TOKENS:-8}"
ALM_STEPS="${ALM_STEPS:-180}"
ALM_LR="${ALM_LR:-2e-6}"
ALM_LM_WEIGHT="${ALM_LM_WEIGHT:-0.10}"
ALM_KD_WEIGHT="${ALM_KD_WEIGHT:-0.05}"
ALM_PROJECTED_VOCAB_KD_WEIGHT="${ALM_PROJECTED_VOCAB_KD_WEIGHT:-0.04}"
ALM_SPAN_HIDDEN_KD_WEIGHT="${ALM_SPAN_HIDDEN_KD_WEIGHT:-0.08}"
ALM_TRAIN_SCOPE="${ALM_TRAIN_SCOPE:-all}"

BLOCK_OUTPUT_HEAD_ARGS=()
if [[ "${BLOCK_TRAIN_OUTPUT_HEAD_WITH_KD:-0}" == "1" ]]; then
  BLOCK_OUTPUT_HEAD_ARGS+=(--block-train-output-head-with-kd)
fi
if [[ "${BLOCK_UNTIE_OUTPUT_HEAD_FOR_KD:-0}" == "1" ]]; then
  BLOCK_OUTPUT_HEAD_ARGS+=(--block-untie-output-head-for-kd)
fi
if [[ "${BLOCK_ALLOW_TIED_OUTPUT_HEAD_TRAINING:-0}" == "1" ]]; then
  BLOCK_OUTPUT_HEAD_ARGS+=(--block-allow-tied-output-head-training)
fi

cd "$REPO_ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate
mkdir -p "$RUN_ROOT/logs"

OUT="$RUN_ROOT/$TAG"
LOG="$RUN_ROOT/logs/$TAG.log"
PID="$RUN_ROOT/logs/$TAG.pid"

CUDA_VISIBLE_DEVICES="$GPU_UUID" nohup python -m neural_forge.cli hf-xfamily-chain-anchor-compress \
  "$SOURCE_MODEL" \
  "$ANCHOR_MODEL" \
  "$OUT" \
  --stages init,block,alm \
  --calibration-file "$RUN_ROOT/calibration_mix.jsonl" \
  --max-calibration-texts 384 \
  --eval-suite "$RUN_ROOT/behavior_suite.json" \
  --max-behavior-probes 200 \
  --max-generation-prompts 100 \
  --max-choice-probes 100 \
  --max-text-evals 96 \
  --max-new-tokens 96 \
  --stage-ppl-gate-mode advisory \
  --promote-policy always \
  --init-fold-mode transport \
  --init-hidden-basis-mode activation-pca \
  --init-hidden-basis-scope layer \
  --init-hidden-basis-max-rows 4096 \
  --init-hidden-basis-preprocess uncentered-rowrms \
  --init-activation-fold-assignment similarity \
  --init-transport-backend sinkhorn \
  --init-transport-support "$INIT_TRANSPORT_SUPPORT" \
  --init-donor-layer-fold-mode fold \
  --init-donor-layer-group-strategy activation-energy \
  --init-attention-fold-mode activation-coupled \
  --init-attention-head-dim-transport none \
  --init-token-init-mode donor-tokenized-weighted \
  --init-projection-scale "$INIT_PROJECTION_SCALE" \
  --init-projection-blend-alpha "$INIT_PROJECTION_BLEND_ALPHA" \
  --init-consumer-hidden-reduction sqrt-sum \
  --init-activation-fit-linears \
  --init-activation-fit-target "$ACT_TARGET" \
  --init-activation-fit-max-rows 2048 \
  --init-activation-fit-alpha "$LRC_ALPHA" \
  --init-activation-fit-ridge 0.03 \
  --init-activation-fit-rank "$LRC_RANK" \
  --init-activation-fit-rank-mode delta \
  --init-activation-fit-method lrc \
  --init-activation-fit-factor-steps "$LRC_FACTOR_STEPS" \
  --init-activation-fit-factor-lr 0.03 \
  --init-activation-fit-factor-l2 0.0001 \
  --init-activation-fit-gate validation \
  --init-activation-fit-validation-fraction 0.20 \
  --init-activation-fit-min-validation-rows 16 \
  --init-activation-fit-min-relative-improvement 0.01 \
  --init-activation-fit-max-cosine-drop 0.02 \
  --init-activation-feature-mode covariance \
  --init-activation-subspace-rank 12 \
  --init-activation-max-length 160 \
  --block-steps-per-layer "$BLOCK_STEPS_PER_LAYER" \
  --block-batch-size 1 \
  --block-max-length 128 \
  --block-lr "$BLOCK_LR" \
  --block-hidden-metric projected-geometry \
  --block-span-pair-mode both \
  --block-target delta \
  --block-residual-magnitude-compensation "$RESIDUAL_MODE" \
  --block-residual-compensation-alpha "$RESIDUAL_ALPHA" \
  --block-residual-compensation-linear-rank "$RESIDUAL_RANK" \
  --block-pre-reconstruction-target block-interface \
  --block-pre-reconstruction-alpha 0.02 \
  --block-pre-reconstruction-fit-gate validation \
  --block-donor-layer-fold-mode fold \
  --block-donor-layer-group-strategy activation-energy \
  --block-layer-gate-metric behavior \
  --block-layer-gate-max-texts 160 \
  --block-layer-gate-max-generation-prompts 80 \
  --block-layer-gate-max-behavior-drop 0.02 \
  --block-alm-kd-weight "$BLOCK_ALM_KD_WEIGHT" \
  --block-projected-vocab-kd-weight "$BLOCK_PROJECTED_VOCAB_KD_WEIGHT" \
  --block-byte-kd-weight "$BLOCK_BYTE_KD_WEIGHT" \
  --block-block-io-kd-weight "$BLOCK_BLOCK_IO_KD_WEIGHT" \
  --block-on-policy-anchor-kd-weight "$BLOCK_ON_POLICY_ANCHOR_KD_WEIGHT" \
  --block-on-policy-unlikelihood-weight "$BLOCK_ON_POLICY_UNLIKELIHOOD_WEIGHT" \
  --block-on-policy-hidden-kd-weight "$BLOCK_ON_POLICY_HIDDEN_KD_WEIGHT" \
  --block-on-policy-max-new-tokens "$BLOCK_ON_POLICY_MAX_NEW_TOKENS" \
  --block-on-policy-interval "$BLOCK_ON_POLICY_INTERVAL" \
  --block-on-policy-response-loss-mode "$BLOCK_ON_POLICY_RESPONSE_LOSS_MODE" \
  --block-on-policy-unlikelihood-window "$BLOCK_ON_POLICY_UNLIKELIHOOD_WINDOW" \
  --block-stepwise-donor-kd-weight "$BLOCK_STEPWISE_DONOR_KD_WEIGHT" \
  --block-stepwise-donor-max-new-tokens "$BLOCK_STEPWISE_DONOR_MAX_NEW_TOKENS" \
  --block-stepwise-donor-interval "$BLOCK_STEPWISE_DONOR_INTERVAL" \
  --block-teacher-generate-lm-weight "$BLOCK_TEACHER_GENERATE_LM_WEIGHT" \
  --block-teacher-generate-hidden-kd-weight "$BLOCK_TEACHER_GENERATE_HIDDEN_KD_WEIGHT" \
  --block-teacher-generate-projected-vocab-kd-weight "$BLOCK_TEACHER_GENERATE_PROJECTED_VOCAB_KD_WEIGHT" \
  --block-teacher-generate-byte-kd-weight "$BLOCK_TEACHER_GENERATE_BYTE_KD_WEIGHT" \
  --block-teacher-generate-max-new-tokens "$BLOCK_TEACHER_GENERATE_MAX_NEW_TOKENS" \
  --block-alm-max-chunks "$BLOCK_ALM_MAX_CHUNKS" \
  --block-alm-min-chars "$BLOCK_ALM_MIN_CHARS" \
  --block-focused-block-hidden-kd-weight "$BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT" \
  --block-focused-projected-vocab-kd-weight "$BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT" \
  --block-focused-lm-weight "$BLOCK_FOCUSED_LM_WEIGHT" \
  --block-token-focus "$BLOCK_TOKEN_FOCUS" \
  --block-projected-vocab-kd-mode "$BLOCK_PROJECTED_VOCAB_KD_MODE" \
  --block-byte-hidden-kd-weight "$BLOCK_BYTE_HIDDEN_KD_WEIGHT" \
  --block-block-io-ridge "$BLOCK_BLOCK_IO_RIDGE" \
  --block-block-io-loss "$BLOCK_BLOCK_IO_LOSS" \
  "${BLOCK_OUTPUT_HEAD_ARGS[@]}" \
  --block-module-output-kd-weight "$BLOCK_MODULE_OUTPUT_KD_WEIGHT" \
  --block-module-output-targets "$BLOCK_MODULE_OUTPUT_TARGETS" \
  --block-module-output-transport "$BLOCK_MODULE_OUTPUT_TRANSPORT" \
  --block-module-output-temperature "$BLOCK_MODULE_OUTPUT_TEMPERATURE" \
  --block-module-output-sinkhorn-iterations "$BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS" \
  --block-projected-vocab-max-source-tokens "$BLOCK_PROJECTED_VOCAB_MAX_SOURCE_TOKENS" \
  --block-save-best-layer-behavior \
  --block-layer-behavior-eval-interval 2 \
  --alm-steps "$ALM_STEPS" \
  --alm-batch-size 1 \
  --alm-max-length 128 \
  --alm-lr "$ALM_LR" \
  --alm-lm-weight "$ALM_LM_WEIGHT" \
  --alm-kd-weight "$ALM_KD_WEIGHT" \
  --alm-projected-vocab-kd-weight "$ALM_PROJECTED_VOCAB_KD_WEIGHT" \
  --alm-span-hidden-kd-weight "$ALM_SPAN_HIDDEN_KD_WEIGHT" \
  --alm-train-scope "$ALM_TRAIN_SCOPE" \
  --alm-save-best-behavior \
  --alm-behavior-eval-interval 40 \
  --dtype float16 \
  --candidate-dtype float32 \
  --device cuda:0 > "$LOG" 2>&1 &

echo $! > "$PID"
echo "$TAG"
echo "source=$SOURCE_MODEL"
echo "anchor=$ANCHOR_MODEL"
echo "pid=$(cat "$PID")"
echo "log=$LOG"
