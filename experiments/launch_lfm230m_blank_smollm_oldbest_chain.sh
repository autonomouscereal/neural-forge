#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
SOURCE_MODEL="${SOURCE_MODEL:-LiquidAI/LFM2.5-230M}"
ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/blank_smollm2_135m_seed0}"
TAG="${TAG:-lfm230m_blank_smollm_chain_oldbest_noactfit_gpu6_20260629}"
INIT_ATTENTION_HEAD_DIM_TRANSPORT="${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-none}"
INIT_TOKEN_FALLBACK_MODE="${INIT_TOKEN_FALLBACK_MODE:-donor-mean}"

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
  --init-donor-layer-fold-mode fold \
  --init-donor-layer-group-strategy activation-energy \
  --init-attention-fold-mode activation-coupled \
  --init-attention-head-dim-transport "$INIT_ATTENTION_HEAD_DIM_TRANSPORT" \
  --init-token-init-mode donor-tokenized-weighted \
  --init-token-fallback-mode "$INIT_TOKEN_FALLBACK_MODE" \
  --init-projection-scale target-rms \
  --init-projection-blend-alpha 1.0 \
  --init-consumer-hidden-reduction sqrt-sum \
  --init-activation-feature-mode covariance \
  --init-activation-subspace-rank 12 \
  --init-activation-max-length 160 \
  --block-steps-per-layer 6 \
  --block-batch-size 1 \
  --block-max-length 128 \
  --block-lr 2e-6 \
  --block-hidden-metric both \
  --block-span-pair-mode both \
  --block-target delta \
  --block-residual-magnitude-compensation lowrank-pre \
  --block-residual-compensation-alpha 0.01 \
  --block-residual-compensation-linear-rank 16 \
  --block-pre-reconstruction-target block-interface \
  --block-pre-reconstruction-alpha 0.02 \
  --block-pre-reconstruction-fit-gate validation \
  --block-donor-layer-fold-mode fold \
  --block-donor-layer-group-strategy activation-energy \
  --block-layer-gate-metric behavior \
  --block-layer-gate-max-texts 160 \
  --block-layer-gate-max-generation-prompts 80 \
  --block-layer-gate-max-behavior-drop 0.02 \
  --block-alm-kd-weight 0.02 \
  --block-projected-vocab-kd-weight 0.02 \
  --block-byte-kd-weight 0.01 \
  --block-block-io-kd-weight 0.02 \
  --block-module-output-kd-weight 0.02 \
  --block-module-output-targets both \
  --block-module-output-transport both \
  --block-save-best-layer-behavior \
  --block-layer-behavior-eval-interval 2 \
  --alm-steps 220 \
  --alm-batch-size 1 \
  --alm-max-length 128 \
  --alm-lr 2e-6 \
  --alm-anchor-kd-weight 0.08 \
  --alm-lm-weight 0.12 \
  --alm-dual-space-kd-weight 0.12 \
  --alm-kd-weight 0.08 \
  --alm-projected-vocab-kd-weight 0.04 \
  --alm-span-hidden-kd-weight 0.10 \
  --alm-train-scope all \
  --alm-response-loss-mode auto \
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
