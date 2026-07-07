#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
SOURCE_MODEL="${SOURCE_MODEL:-LiquidAI/LFM2.5-230M}"
ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/blank_smollm2_135m_seed0}"
TAG="${TAG:-lfm230m_blank_smollm_chain_true_lrc_rank16_gpu6_20260629}"
CALIBRATION_FILE="${CALIBRATION_FILE:-$RUN_ROOT/calibration_mix.jsonl}"
EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite.json}"
MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-384}"
MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-200}"
MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-100}"
MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-100}"
MAX_TEXT_EVALS="${MAX_TEXT_EVALS:-96}"
ACT_TARGET="${ACT_TARGET:-mlp-output}"
LRC_ALPHA="${LRC_ALPHA:-0.25}"
LRC_RANK="${LRC_RANK:-16}"
LRC_FACTOR_STEPS="${LRC_FACTOR_STEPS:-40}"
LRC_MAX_DELTA_RMS_RATIO="${LRC_MAX_DELTA_RMS_RATIO:-0}"
INIT_ACTIVATION_FIT_LINEARS="${INIT_ACTIVATION_FIT_LINEARS:-1}"
INIT_ACTIVATION_FIT_METHOD="${INIT_ACTIVATION_FIT_METHOD:-lrc}"
RESIDUAL_MODE="${RESIDUAL_MODE:-lowrank-pre}"
RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.01}"
RESIDUAL_RANK="${RESIDUAL_RANK:-16}"
INIT_PROJECTION_SCALE="${INIT_PROJECTION_SCALE:-target-rms}"
INIT_PROJECTION_BLEND_ALPHA="${INIT_PROJECTION_BLEND_ALPHA:-1.0}"
INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-similarity}"
INIT_FOLD_STAT_REPAIR="${INIT_FOLD_STAT_REPAIR:-none}"
INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-sinkhorn}"
INIT_TRANSPORT_SUPPORT="${INIT_TRANSPORT_SUPPORT:-centroid}"
INIT_ATTENTION_HEAD_DIM_TRANSPORT="${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-none}"
INIT_ACTIVATION_DOMAIN_BALANCE="${INIT_ACTIVATION_DOMAIN_BALANCE:-none}"
INIT_DONOR_LAYER_GROUP_STRATEGY="${INIT_DONOR_LAYER_GROUP_STRATEGY:-activation-geometry}"
BLOCK_DONOR_LAYER_GROUP_STRATEGY="${BLOCK_DONOR_LAYER_GROUP_STRATEGY:-activation-geometry}"
INIT_TOKEN_INIT_MODE="${INIT_TOKEN_INIT_MODE:-donor-tokenized-weighted}"
INIT_TOKEN_FALLBACK_MODE="${INIT_TOKEN_FALLBACK_MODE:-donor-mean}"
INIT_UNTIE_OUTPUT_HEAD="${INIT_UNTIE_OUTPUT_HEAD:-1}"
FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-preserve}"
PROMOTE_POLICY="${PROMOTE_POLICY:-always}"
STAGE_INPUT_POLICY="${STAGE_INPUT_POLICY:-sequential}"
BLOCK_STEPS_PER_LAYER="${BLOCK_STEPS_PER_LAYER:-4}"
BLOCK_LR="${BLOCK_LR:-2e-6}"
BLOCK_HIDDEN_PROJECTION_SCOPE="${BLOCK_HIDDEN_PROJECTION_SCOPE:-layer-input}"
BLOCK_CALIBRATION_ORDER="${BLOCK_CALIBRATION_ORDER:-original}"
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
BLOCK_TEACHER_GENERATE_LOSS_MODE="${BLOCK_TEACHER_GENERATE_LOSS_MODE:-none}"
BLOCK_ALM_MAX_CHUNKS="${BLOCK_ALM_MAX_CHUNKS:-64}"
BLOCK_ALM_MIN_CHARS="${BLOCK_ALM_MIN_CHARS:-2}"
BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT="${BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT:-0.0}"
BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT:-0.0}"
BLOCK_FOCUSED_LM_WEIGHT="${BLOCK_FOCUSED_LM_WEIGHT:-0.0}"
BLOCK_TOKEN_FOCUS="${BLOCK_TOKEN_FOCUS:-auto}"
BLOCK_PROJECTED_VOCAB_KD_MODE="${BLOCK_PROJECTED_VOCAB_KD_MODE:-teacher-to-student}"
BLOCK_BYTE_HIDDEN_KD_WEIGHT="${BLOCK_BYTE_HIDDEN_KD_WEIGHT:-0.0}"
BLOCK_BYTE_HIDDEN_PROBE_MODE="${BLOCK_BYTE_HIDDEN_PROBE_MODE:-fixed}"
BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT="${BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT:-0.0}"
BLOCK_BLOCK_IO_RIDGE="${BLOCK_BLOCK_IO_RIDGE:-1e-3}"
BLOCK_BLOCK_IO_LOSS="${BLOCK_BLOCK_IO_LOSS:-both}"
BLOCK_MODULE_OUTPUT_KD_WEIGHT="${BLOCK_MODULE_OUTPUT_KD_WEIGHT:-0.02}"
BLOCK_MODULE_OUTPUT_TARGETS="${BLOCK_MODULE_OUTPUT_TARGETS:-both}"
BLOCK_MODULE_OUTPUT_TRANSPORT="${BLOCK_MODULE_OUTPUT_TRANSPORT:-both}"
BLOCK_MODULE_OUTPUT_TEMPERATURE="${BLOCK_MODULE_OUTPUT_TEMPERATURE:-1.0}"
BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS="${BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS:-12}"
BLOCK_PROJECTED_VOCAB_MAX_SOURCE_TOKENS="${BLOCK_PROJECTED_VOCAB_MAX_SOURCE_TOKENS:-8}"
BLOCK_NONFINITE_STEP_POLICY="${BLOCK_NONFINITE_STEP_POLICY:-raise}"
BLOCK_MAX_NONFINITE_STEPS_PER_LAYER="${BLOCK_MAX_NONFINITE_STEPS_PER_LAYER:-0}"
PRE_RECONSTRUCTION_TARGET="${PRE_RECONSTRUCTION_TARGET:-block-interface}"
PRE_RECONSTRUCTION_ALPHA="${PRE_RECONSTRUCTION_ALPHA:-0.02}"
PRE_RECONSTRUCTION_FIT_GATE="${PRE_RECONSTRUCTION_FIT_GATE:-validation}"
ALM_STEPS="${ALM_STEPS:-180}"
ALM_LR="${ALM_LR:-2e-6}"
ALM_LM_WEIGHT="${ALM_LM_WEIGHT:-0.10}"
ALM_FIRST_TOKEN_LM_WEIGHT="${ALM_FIRST_TOKEN_LM_WEIGHT:-0.0}"
ALM_RESPONSE_END_LM_WEIGHT="${ALM_RESPONSE_END_LM_WEIGHT:-0.0}"
ALM_FOCUSED_LM_WEIGHT="${ALM_FOCUSED_LM_WEIGHT:-0.0}"
ALM_KD_WEIGHT="${ALM_KD_WEIGHT:-0.05}"
ALM_PROJECTED_VOCAB_KD_WEIGHT="${ALM_PROJECTED_VOCAB_KD_WEIGHT:-0.04}"
ALM_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT="${ALM_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT:-0.0}"
ALM_SPAN_HIDDEN_KD_WEIGHT="${ALM_SPAN_HIDDEN_KD_WEIGHT:-0.08}"
ALM_ATTENTION_INFLUENCE_WEIGHT="${ALM_ATTENTION_INFLUENCE_WEIGHT:-0.0}"
ALM_ATTENTION_INFLUENCE_LAYER="${ALM_ATTENTION_INFLUENCE_LAYER:-last}"
ALM_TOKEN_KD_WEIGHT="${ALM_TOKEN_KD_WEIGHT:-0.0}"
ALM_TOKEN_FOCUS="${ALM_TOKEN_FOCUS:-auto}"
ALM_TOKEN_MAX_PAIRS="${ALM_TOKEN_MAX_PAIRS:-512}"
ALM_TOKEN_TEACHER_CONFIDENCE_FLOOR="${ALM_TOKEN_TEACHER_CONFIDENCE_FLOOR:-0.0}"
ALM_ANSWER_CONTRASTIVE_WEIGHT="${ALM_ANSWER_CONTRASTIVE_WEIGHT:-0.0}"
ALM_ANSWER_CONTRASTIVE_MAX_NEGATIVES="${ALM_ANSWER_CONTRASTIVE_MAX_NEGATIVES:-4}"
ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT="${ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT:-0.0}"
ALM_ANSWER_FIRST_TOKEN_MARGIN="${ALM_ANSWER_FIRST_TOKEN_MARGIN:-1.0}"
ALM_ANSWER_FIRST_TOKEN_MAX_NEGATIVES="${ALM_ANSWER_FIRST_TOKEN_MAX_NEGATIVES:-8}"
ALM_ANSWER_FIRST_TOKEN_TOP_NEGATIVES="${ALM_ANSWER_FIRST_TOKEN_TOP_NEGATIVES:-0}"
ALM_TEACHER_RANK_WEIGHT="${ALM_TEACHER_RANK_WEIGHT:-0.0}"
ALM_TEACHER_RANK_TEMPERATURE="${ALM_TEACHER_RANK_TEMPERATURE:-1.0}"
ALM_TEACHER_RANK_MAX_NEGATIVES="${ALM_TEACHER_RANK_MAX_NEGATIVES:-4}"
ALM_TEACHER_GENERATE_LM_WEIGHT="${ALM_TEACHER_GENERATE_LM_WEIGHT:-0.0}"
ALM_TEACHER_GENERATE_MAX_NEW_TOKENS="${ALM_TEACHER_GENERATE_MAX_NEW_TOKENS:-64}"
ALM_TEACHER_GENERATE_LOSS_MODE="${ALM_TEACHER_GENERATE_LOSS_MODE:-none}"
ALM_STEPWISE_DONOR_KD_WEIGHT="${ALM_STEPWISE_DONOR_KD_WEIGHT:-0.0}"
ALM_STEPWISE_DONOR_MAX_NEW_TOKENS="${ALM_STEPWISE_DONOR_MAX_NEW_TOKENS:-16}"
ALM_ON_POLICY_ANCHOR_KD_WEIGHT="${ALM_ON_POLICY_ANCHOR_KD_WEIGHT:-0.0}"
ALM_ON_POLICY_UNLIKELIHOOD_WEIGHT="${ALM_ON_POLICY_UNLIKELIHOOD_WEIGHT:-0.0}"
ALM_ON_POLICY_HIDDEN_KD_WEIGHT="${ALM_ON_POLICY_HIDDEN_KD_WEIGHT:-0.0}"
ALM_ON_POLICY_MAX_NEW_TOKENS="${ALM_ON_POLICY_MAX_NEW_TOKENS:-24}"
ALM_ON_POLICY_INTERVAL="${ALM_ON_POLICY_INTERVAL:-10}"
ALM_TRAIN_SCOPE="${ALM_TRAIN_SCOPE:-all}"
ALM_RESPONSE_LOSS_MODE="${ALM_RESPONSE_LOSS_MODE:-none}"
ALM_CALIBRATION_ORDER="${ALM_CALIBRATION_ORDER:-original}"

BLOCK_OUTPUT_HEAD_ARGS=()
INIT_OUTPUT_HEAD_ARGS=()
INIT_ACTIVATION_FIT_ARGS=()
if [[ "$INIT_UNTIE_OUTPUT_HEAD" == "1" ]]; then
  INIT_OUTPUT_HEAD_ARGS+=(--init-untie-output-head)
fi
if [[ "$INIT_ACTIVATION_FIT_LINEARS" == "1" ]]; then
  INIT_ACTIVATION_FIT_ARGS+=(--init-activation-fit-linears)
fi
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
  --calibration-file "$CALIBRATION_FILE" \
  --max-calibration-texts "$MAX_CALIBRATION_TEXTS" \
  --eval-suite "$EVAL_SUITE_FILE" \
  --max-behavior-probes "$MAX_BEHAVIOR_PROBES" \
  --max-generation-prompts "$MAX_GENERATION_PROMPTS" \
  --max-choice-probes "$MAX_CHOICE_PROBES" \
  --max-text-evals "$MAX_TEXT_EVALS" \
  --max-new-tokens 96 \
  --stage-ppl-gate-mode advisory \
  --promote-policy "$PROMOTE_POLICY" \
  --stage-input-policy "$STAGE_INPUT_POLICY" \
  --final-output-head-policy "$FINAL_OUTPUT_HEAD_POLICY" \
  --init-fold-mode transport \
  --init-hidden-basis-mode activation-pca \
  --init-hidden-basis-scope layer \
  --init-hidden-basis-max-rows 4096 \
  --init-hidden-basis-preprocess uncentered-rowrms \
  --init-activation-fold-assignment "$INIT_ACTIVATION_FOLD_ASSIGNMENT" \
  --init-fold-stat-repair "$INIT_FOLD_STAT_REPAIR" \
  --init-transport-backend "$INIT_TRANSPORT_BACKEND" \
  --init-transport-support "$INIT_TRANSPORT_SUPPORT" \
  --init-donor-layer-fold-mode fold \
  --init-donor-layer-group-strategy "$INIT_DONOR_LAYER_GROUP_STRATEGY" \
  --init-attention-fold-mode activation-coupled \
  --init-attention-head-dim-transport "$INIT_ATTENTION_HEAD_DIM_TRANSPORT" \
  --init-token-init-mode "$INIT_TOKEN_INIT_MODE" \
  --init-token-fallback-mode "$INIT_TOKEN_FALLBACK_MODE" \
  "${INIT_OUTPUT_HEAD_ARGS[@]}" \
  --init-projection-scale "$INIT_PROJECTION_SCALE" \
  --init-projection-blend-alpha "$INIT_PROJECTION_BLEND_ALPHA" \
  --init-consumer-hidden-reduction sqrt-sum \
  "${INIT_ACTIVATION_FIT_ARGS[@]}" \
  --init-activation-fit-target "$ACT_TARGET" \
  --init-activation-fit-max-rows 2048 \
  --init-activation-fit-alpha "$LRC_ALPHA" \
  --init-activation-fit-ridge 0.03 \
  --init-activation-fit-rank "$LRC_RANK" \
  --init-activation-fit-rank-mode delta \
  --init-activation-fit-method "$INIT_ACTIVATION_FIT_METHOD" \
  --init-activation-fit-factor-steps "$LRC_FACTOR_STEPS" \
  --init-activation-fit-factor-lr 0.03 \
  --init-activation-fit-factor-l2 0.0001 \
  --init-activation-fit-max-delta-rms-ratio "$LRC_MAX_DELTA_RMS_RATIO" \
  --init-activation-fit-gate validation \
  --init-activation-fit-validation-fraction 0.20 \
  --init-activation-fit-min-validation-rows 16 \
  --init-activation-fit-min-relative-improvement 0.01 \
  --init-activation-fit-max-cosine-drop 0.02 \
  --init-activation-feature-mode covariance \
  --init-activation-subspace-rank 12 \
  --init-activation-domain-balance "$INIT_ACTIVATION_DOMAIN_BALANCE" \
  --init-activation-max-length 160 \
  --block-steps-per-layer "$BLOCK_STEPS_PER_LAYER" \
  --block-batch-size 1 \
  --block-max-length 128 \
  --block-lr "$BLOCK_LR" \
  --block-hidden-projection-scope "$BLOCK_HIDDEN_PROJECTION_SCOPE" \
  --block-hidden-metric projected-geometry \
  --block-span-pair-mode both \
  --block-target delta \
  --block-residual-magnitude-compensation "$RESIDUAL_MODE" \
  --block-residual-compensation-alpha "$RESIDUAL_ALPHA" \
  --block-residual-compensation-linear-rank "$RESIDUAL_RANK" \
  --block-pre-reconstruction-target "$PRE_RECONSTRUCTION_TARGET" \
  --block-pre-reconstruction-alpha "$PRE_RECONSTRUCTION_ALPHA" \
  --block-pre-reconstruction-fit-gate "$PRE_RECONSTRUCTION_FIT_GATE" \
  --block-donor-layer-fold-mode fold \
  --block-donor-layer-group-strategy "$BLOCK_DONOR_LAYER_GROUP_STRATEGY" \
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
  --block-teacher-generate-loss-mode "$BLOCK_TEACHER_GENERATE_LOSS_MODE" \
  --block-alm-max-chunks "$BLOCK_ALM_MAX_CHUNKS" \
  --block-alm-min-chars "$BLOCK_ALM_MIN_CHARS" \
  --block-focused-block-hidden-kd-weight "$BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT" \
  --block-focused-projected-vocab-kd-weight "$BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT" \
  --block-focused-lm-weight "$BLOCK_FOCUSED_LM_WEIGHT" \
  --block-token-focus "$BLOCK_TOKEN_FOCUS" \
  --block-projected-vocab-kd-mode "$BLOCK_PROJECTED_VOCAB_KD_MODE" \
  --block-byte-hidden-kd-weight "$BLOCK_BYTE_HIDDEN_KD_WEIGHT" \
  --block-byte-hidden-probe-mode "$BLOCK_BYTE_HIDDEN_PROBE_MODE" \
  --block-byte-hidden-probe-anchor-weight "$BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT" \
  --block-block-io-ridge "$BLOCK_BLOCK_IO_RIDGE" \
  --block-block-io-loss "$BLOCK_BLOCK_IO_LOSS" \
  "${BLOCK_OUTPUT_HEAD_ARGS[@]}" \
  --block-module-output-kd-weight "$BLOCK_MODULE_OUTPUT_KD_WEIGHT" \
  --block-module-output-targets "$BLOCK_MODULE_OUTPUT_TARGETS" \
  --block-module-output-transport "$BLOCK_MODULE_OUTPUT_TRANSPORT" \
  --block-module-output-temperature "$BLOCK_MODULE_OUTPUT_TEMPERATURE" \
  --block-module-output-sinkhorn-iterations "$BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS" \
  --block-projected-vocab-max-source-tokens "$BLOCK_PROJECTED_VOCAB_MAX_SOURCE_TOKENS" \
  --block-nonfinite-step-policy "$BLOCK_NONFINITE_STEP_POLICY" \
  --block-max-nonfinite-steps-per-layer "$BLOCK_MAX_NONFINITE_STEPS_PER_LAYER" \
  --block-save-best-layer-behavior \
  --block-layer-behavior-eval-interval 2 \
  --block-calibration-order "$BLOCK_CALIBRATION_ORDER" \
  --alm-steps "$ALM_STEPS" \
  --alm-batch-size 1 \
  --alm-max-length 128 \
  --alm-lr "$ALM_LR" \
  --alm-lm-weight "$ALM_LM_WEIGHT" \
  --alm-first-token-lm-weight "$ALM_FIRST_TOKEN_LM_WEIGHT" \
  --alm-response-end-lm-weight "$ALM_RESPONSE_END_LM_WEIGHT" \
  --alm-focused-lm-weight "$ALM_FOCUSED_LM_WEIGHT" \
  --alm-kd-weight "$ALM_KD_WEIGHT" \
  --alm-projected-vocab-kd-weight "$ALM_PROJECTED_VOCAB_KD_WEIGHT" \
  --alm-focused-projected-vocab-kd-weight "$ALM_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT" \
  --alm-span-hidden-kd-weight "$ALM_SPAN_HIDDEN_KD_WEIGHT" \
  --alm-attention-influence-weight "$ALM_ATTENTION_INFLUENCE_WEIGHT" \
  --alm-attention-influence-layer "$ALM_ATTENTION_INFLUENCE_LAYER" \
  --alm-token-kd-weight "$ALM_TOKEN_KD_WEIGHT" \
  --alm-token-focus "$ALM_TOKEN_FOCUS" \
  --alm-token-max-pairs "$ALM_TOKEN_MAX_PAIRS" \
  --alm-token-teacher-confidence-floor "$ALM_TOKEN_TEACHER_CONFIDENCE_FLOOR" \
  --alm-answer-contrastive-weight "$ALM_ANSWER_CONTRASTIVE_WEIGHT" \
  --alm-answer-contrastive-max-negatives "$ALM_ANSWER_CONTRASTIVE_MAX_NEGATIVES" \
  --alm-answer-first-token-margin-weight "$ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT" \
  --alm-answer-first-token-margin "$ALM_ANSWER_FIRST_TOKEN_MARGIN" \
  --alm-answer-first-token-max-negatives "$ALM_ANSWER_FIRST_TOKEN_MAX_NEGATIVES" \
  --alm-answer-first-token-top-negatives "$ALM_ANSWER_FIRST_TOKEN_TOP_NEGATIVES" \
  --alm-teacher-rank-weight "$ALM_TEACHER_RANK_WEIGHT" \
  --alm-teacher-rank-temperature "$ALM_TEACHER_RANK_TEMPERATURE" \
  --alm-teacher-rank-max-negatives "$ALM_TEACHER_RANK_MAX_NEGATIVES" \
  --alm-teacher-generate-lm-weight "$ALM_TEACHER_GENERATE_LM_WEIGHT" \
  --alm-teacher-generate-max-new-tokens "$ALM_TEACHER_GENERATE_MAX_NEW_TOKENS" \
  --alm-teacher-generate-loss-mode "$ALM_TEACHER_GENERATE_LOSS_MODE" \
  --alm-stepwise-donor-kd-weight "$ALM_STEPWISE_DONOR_KD_WEIGHT" \
  --alm-stepwise-donor-max-new-tokens "$ALM_STEPWISE_DONOR_MAX_NEW_TOKENS" \
  --alm-on-policy-anchor-kd-weight "$ALM_ON_POLICY_ANCHOR_KD_WEIGHT" \
  --alm-on-policy-unlikelihood-weight "$ALM_ON_POLICY_UNLIKELIHOOD_WEIGHT" \
  --alm-on-policy-hidden-kd-weight "$ALM_ON_POLICY_HIDDEN_KD_WEIGHT" \
  --alm-on-policy-max-new-tokens "$ALM_ON_POLICY_MAX_NEW_TOKENS" \
  --alm-on-policy-interval "$ALM_ON_POLICY_INTERVAL" \
  --alm-train-scope "$ALM_TRAIN_SCOPE" \
  --alm-response-loss-mode "$ALM_RESPONSE_LOSS_MODE" \
  --alm-save-best-behavior \
  --alm-behavior-eval-interval 40 \
  --alm-calibration-order "$ALM_CALIBRATION_ORDER" \
  --dtype float16 \
  --candidate-dtype float32 \
  --device cuda:0 > "$LOG" 2>&1 &

echo $! > "$PID"
echo "$TAG"
echo "source=$SOURCE_MODEL"
echo "anchor=$ANCHOR_MODEL"
echo "pid=$(cat "$PID")"
echo "log=$LOG"
