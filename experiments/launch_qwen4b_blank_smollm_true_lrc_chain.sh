#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
TAG="${TAG:-qwen4b_blank_smollm_chain_mlpout_true_lrc_rank16_blockio_alm_gpu4_20260629}"

cd "$REPO_ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate
mkdir -p "$RUN_ROOT/logs"

OUT="$RUN_ROOT/$TAG"
LOG="$RUN_ROOT/logs/$TAG.log"
PID="$RUN_ROOT/logs/$TAG.pid"

CUDA_VISIBLE_DEVICES="$GPU_UUID" nohup python -m neural_forge.cli hf-xfamily-chain-anchor-compress \
  Qwen/Qwen3-4B-Base \
  "$RUN_ROOT/blank_smollm2_135m_seed0" \
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
  --init-attention-head-dim-transport none \
  --init-token-init-mode donor-tokenized-weighted \
  --init-projection-scale target-rms \
  --init-projection-blend-alpha 1.0 \
  --init-consumer-hidden-reduction sqrt-sum \
  --init-activation-fit-linears \
  --init-activation-fit-target mlp-output \
  --init-activation-fit-max-rows 2048 \
  --init-activation-fit-alpha 0.25 \
  --init-activation-fit-ridge 0.03 \
  --init-activation-fit-rank 16 \
  --init-activation-fit-rank-mode delta \
  --init-activation-fit-method lrc \
  --init-activation-fit-factor-steps 40 \
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
  --block-donor-layer-group-strategy activation-energy \
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
  --alm-steps 220 \
  --alm-batch-size 1 \
  --alm-max-length 128 \
  --alm-lr 2e-6 \
  --alm-lm-weight 0.10 \
  --alm-kd-weight 0.05 \
  --alm-projected-vocab-kd-weight 0.04 \
  --alm-span-hidden-kd-weight 0.08 \
  --alm-train-scope all \
  --alm-save-best-behavior \
  --alm-behavior-eval-interval 40 \
  --dtype float16 \
  --candidate-dtype float32 \
  --device cuda:0 > "$LOG" 2>&1 &

echo $! > "$PID"
echo "$TAG"
echo "pid=$(cat "$PID")"
echo "log=$LOG"
