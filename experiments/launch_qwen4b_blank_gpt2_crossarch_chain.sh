#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base}"
ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/blank_gpt2_124m_seed0}"
TAG="${TAG:-qwen4b_blank_gpt2_crossarch_lrc_alm_gpu6_20260629}"
INIT_TRANSPORT_SUPPORT="${INIT_TRANSPORT_SUPPORT:-centroid}"
INIT_ATTENTION_HEAD_DIM_TRANSPORT="${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-none}"
INIT_ACTIVATION_DOMAIN_BALANCE="${INIT_ACTIVATION_DOMAIN_BALANCE:-none}"
INIT_DONOR_LAYER_GROUP_STRATEGY="${INIT_DONOR_LAYER_GROUP_STRATEGY:-activation-geometry}"
BLOCK_DONOR_LAYER_GROUP_STRATEGY="${BLOCK_DONOR_LAYER_GROUP_STRATEGY:-activation-geometry}"
INIT_UNTIE_OUTPUT_HEAD="${INIT_UNTIE_OUTPUT_HEAD:-1}"
INIT_ACTIVATION_FIT_METHOD="${INIT_ACTIVATION_FIT_METHOD:-lrc}"
INIT_TOKEN_INIT_MODE="${INIT_TOKEN_INIT_MODE:-donor-tokenized-weighted}"
INIT_TOKEN_FALLBACK_MODE="${INIT_TOKEN_FALLBACK_MODE:-donor-mean}"
PROMOTE_POLICY="${PROMOTE_POLICY:-always}"
STAGE_INPUT_POLICY="${STAGE_INPUT_POLICY:-incumbent}"
BLOCK_CALIBRATION_ORDER="${BLOCK_CALIBRATION_ORDER:-original}"
ALM_CALIBRATION_ORDER="${ALM_CALIBRATION_ORDER:-original}"
ALM_FIRST_TOKEN_LM_WEIGHT="${ALM_FIRST_TOKEN_LM_WEIGHT:-0.0}"
ALM_RESPONSE_END_LM_WEIGHT="${ALM_RESPONSE_END_LM_WEIGHT:-0.0}"
ALM_FOCUSED_LM_WEIGHT="${ALM_FOCUSED_LM_WEIGHT:-0.0}"
ALM_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT="${ALM_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT:-0.0}"
ALM_ATTENTION_INFLUENCE_WEIGHT="${ALM_ATTENTION_INFLUENCE_WEIGHT:-0.0}"
ALM_ATTENTION_INFLUENCE_LAYER="${ALM_ATTENTION_INFLUENCE_LAYER:-last}"
ALM_TOKEN_KD_WEIGHT="${ALM_TOKEN_KD_WEIGHT:-0.0}"
ALM_TOKEN_FOCUS="${ALM_TOKEN_FOCUS:-auto}"
ALM_TOKEN_MAX_PAIRS="${ALM_TOKEN_MAX_PAIRS:-512}"
ALM_TOKEN_TEACHER_CONFIDENCE_FLOOR="${ALM_TOKEN_TEACHER_CONFIDENCE_FLOOR:-0.0}"
ALM_ANSWER_CONTRASTIVE_WEIGHT="${ALM_ANSWER_CONTRASTIVE_WEIGHT:-0.0}"
ALM_ANSWER_CONTRASTIVE_MAX_NEGATIVES="${ALM_ANSWER_CONTRASTIVE_MAX_NEGATIVES:-4}"
ALM_TEACHER_RANK_WEIGHT="${ALM_TEACHER_RANK_WEIGHT:-0.0}"
ALM_TEACHER_RANK_TEMPERATURE="${ALM_TEACHER_RANK_TEMPERATURE:-1.0}"
ALM_TEACHER_RANK_MAX_NEGATIVES="${ALM_TEACHER_RANK_MAX_NEGATIVES:-4}"
ALM_TEACHER_GENERATE_LM_WEIGHT="${ALM_TEACHER_GENERATE_LM_WEIGHT:-0.0}"
ALM_TEACHER_GENERATE_MAX_NEW_TOKENS="${ALM_TEACHER_GENERATE_MAX_NEW_TOKENS:-64}"
ALM_STEPWISE_DONOR_KD_WEIGHT="${ALM_STEPWISE_DONOR_KD_WEIGHT:-0.0}"
ALM_STEPWISE_DONOR_MAX_NEW_TOKENS="${ALM_STEPWISE_DONOR_MAX_NEW_TOKENS:-16}"
ALM_ON_POLICY_ANCHOR_KD_WEIGHT="${ALM_ON_POLICY_ANCHOR_KD_WEIGHT:-0.0}"
ALM_ON_POLICY_UNLIKELIHOOD_WEIGHT="${ALM_ON_POLICY_UNLIKELIHOOD_WEIGHT:-0.0}"
ALM_ON_POLICY_HIDDEN_KD_WEIGHT="${ALM_ON_POLICY_HIDDEN_KD_WEIGHT:-0.0}"
ALM_ON_POLICY_MAX_NEW_TOKENS="${ALM_ON_POLICY_MAX_NEW_TOKENS:-24}"
ALM_ON_POLICY_INTERVAL="${ALM_ON_POLICY_INTERVAL:-10}"
ALM_RESPONSE_LOSS_MODE="${ALM_RESPONSE_LOSS_MODE:-none}"

INIT_OUTPUT_HEAD_ARGS=()
if [[ "$INIT_UNTIE_OUTPUT_HEAD" == "1" ]]; then
  INIT_OUTPUT_HEAD_ARGS+=(--init-untie-output-head)
fi

cd "$REPO_ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate
mkdir -p "$RUN_ROOT/logs"

if [ ! -f "$ANCHOR_MODEL/config.json" ]; then
  python -m neural_forge.cli hf-random-init \
    "$ANCHOR_MODEL" \
    --config-model openai-community/gpt2 \
    --tokenizer-model openai-community/gpt2 \
    --seed 0 \
    --dtype float16
fi

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
  --promote-policy "$PROMOTE_POLICY" \
  --stage-input-policy "$STAGE_INPUT_POLICY" \
  --init-fold-mode transport \
  --init-hidden-basis-mode activation-pca \
  --init-hidden-basis-scope layer \
  --init-hidden-basis-max-rows 4096 \
  --init-hidden-basis-preprocess uncentered-rowrms \
  --init-activation-fold-assignment similarity \
  --init-transport-backend sinkhorn \
  --init-transport-support "$INIT_TRANSPORT_SUPPORT" \
  --init-donor-layer-fold-mode fold \
  --init-donor-layer-group-strategy "$INIT_DONOR_LAYER_GROUP_STRATEGY" \
  --init-attention-fold-mode activation-coupled \
  --init-attention-head-dim-transport "$INIT_ATTENTION_HEAD_DIM_TRANSPORT" \
  --init-token-init-mode "$INIT_TOKEN_INIT_MODE" \
  --init-token-fallback-mode "$INIT_TOKEN_FALLBACK_MODE" \
  "${INIT_OUTPUT_HEAD_ARGS[@]}" \
  --init-projection-scale target-rms \
  --init-projection-blend-alpha 1.0 \
  --init-consumer-hidden-reduction sqrt-sum \
  --init-activation-fit-linears \
  --init-activation-fit-target all \
  --init-activation-fit-max-rows 2048 \
  --init-activation-fit-alpha 0.20 \
  --init-activation-fit-ridge 0.03 \
  --init-activation-fit-rank 12 \
  --init-activation-fit-rank-mode delta \
  --init-activation-fit-method "$INIT_ACTIVATION_FIT_METHOD" \
  --init-activation-fit-factor-steps 36 \
  --init-activation-fit-factor-lr 0.03 \
  --init-activation-fit-factor-l2 0.0001 \
  --init-activation-fit-gate validation \
  --init-activation-fit-validation-fraction 0.20 \
  --init-activation-fit-min-validation-rows 16 \
  --init-activation-fit-min-relative-improvement 0.01 \
  --init-activation-fit-max-cosine-drop 0.02 \
  --init-activation-feature-mode covariance \
  --init-activation-subspace-rank 12 \
  --init-activation-domain-balance "$INIT_ACTIVATION_DOMAIN_BALANCE" \
  --init-activation-max-length 160 \
  --block-steps-per-layer 4 \
  --block-batch-size 1 \
  --block-max-length 128 \
  --block-lr 2e-6 \
  --block-hidden-metric projected-geometry \
  --block-span-pair-mode both \
  --block-target delta \
  --block-residual-magnitude-compensation lowrank-pre \
  --block-residual-compensation-alpha 0.01 \
  --block-residual-compensation-linear-rank 16 \
  --block-pre-reconstruction-target block-interface \
  --block-pre-reconstruction-alpha 0.02 \
  --block-pre-reconstruction-fit-gate validation \
  --block-donor-layer-fold-mode fold \
  --block-donor-layer-group-strategy "$BLOCK_DONOR_LAYER_GROUP_STRATEGY" \
  --block-layer-gate-metric behavior \
  --block-layer-gate-max-texts 160 \
  --block-layer-gate-max-generation-prompts 80 \
  --block-layer-gate-max-behavior-drop 0.02 \
  --block-block-io-kd-weight 0.02 \
  --block-module-output-kd-weight 0.02 \
  --block-module-output-targets both \
  --block-module-output-transport both \
  --block-save-best-layer-behavior \
  --block-layer-behavior-eval-interval 2 \
  --block-calibration-order "$BLOCK_CALIBRATION_ORDER" \
  --alm-steps 220 \
  --alm-batch-size 1 \
  --alm-max-length 128 \
  --alm-lr 2e-6 \
  --alm-lm-weight 0.10 \
  --alm-first-token-lm-weight "$ALM_FIRST_TOKEN_LM_WEIGHT" \
  --alm-response-end-lm-weight "$ALM_RESPONSE_END_LM_WEIGHT" \
  --alm-focused-lm-weight "$ALM_FOCUSED_LM_WEIGHT" \
  --alm-kd-weight 0.05 \
  --alm-projected-vocab-kd-weight 0.04 \
  --alm-focused-projected-vocab-kd-weight "$ALM_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT" \
  --alm-span-hidden-kd-weight 0.08 \
  --alm-attention-influence-weight "$ALM_ATTENTION_INFLUENCE_WEIGHT" \
  --alm-attention-influence-layer "$ALM_ATTENTION_INFLUENCE_LAYER" \
  --alm-token-kd-weight "$ALM_TOKEN_KD_WEIGHT" \
  --alm-token-focus "$ALM_TOKEN_FOCUS" \
  --alm-token-max-pairs "$ALM_TOKEN_MAX_PAIRS" \
  --alm-token-teacher-confidence-floor "$ALM_TOKEN_TEACHER_CONFIDENCE_FLOOR" \
  --alm-answer-contrastive-weight "$ALM_ANSWER_CONTRASTIVE_WEIGHT" \
  --alm-answer-contrastive-max-negatives "$ALM_ANSWER_CONTRASTIVE_MAX_NEGATIVES" \
  --alm-teacher-rank-weight "$ALM_TEACHER_RANK_WEIGHT" \
  --alm-teacher-rank-temperature "$ALM_TEACHER_RANK_TEMPERATURE" \
  --alm-teacher-rank-max-negatives "$ALM_TEACHER_RANK_MAX_NEGATIVES" \
  --alm-teacher-generate-lm-weight "$ALM_TEACHER_GENERATE_LM_WEIGHT" \
  --alm-teacher-generate-max-new-tokens "$ALM_TEACHER_GENERATE_MAX_NEW_TOKENS" \
  --alm-stepwise-donor-kd-weight "$ALM_STEPWISE_DONOR_KD_WEIGHT" \
  --alm-stepwise-donor-max-new-tokens "$ALM_STEPWISE_DONOR_MAX_NEW_TOKENS" \
  --alm-on-policy-anchor-kd-weight "$ALM_ON_POLICY_ANCHOR_KD_WEIGHT" \
  --alm-on-policy-unlikelihood-weight "$ALM_ON_POLICY_UNLIKELIHOOD_WEIGHT" \
  --alm-on-policy-hidden-kd-weight "$ALM_ON_POLICY_HIDDEN_KD_WEIGHT" \
  --alm-on-policy-max-new-tokens "$ALM_ON_POLICY_MAX_NEW_TOKENS" \
  --alm-on-policy-interval "$ALM_ON_POLICY_INTERVAL" \
  --alm-train-scope all \
  --alm-response-loss-mode "$ALM_RESPONSE_LOSS_MODE" \
  --alm-save-best-behavior \
  --alm-behavior-eval-interval 40 \
  --alm-calibration-order "$ALM_CALIBRATION_ORDER" \
  --dtype float16 \
  --candidate-dtype float32 \
  --device cuda:0 < /dev/null > "$LOG" 2>&1 &

echo $! > "$PID"
echo "$TAG"
echo "source=$SOURCE_MODEL"
echo "anchor=$ANCHOR_MODEL"
echo "pid=$(cat "$PID")"
echo "log=$LOG"
