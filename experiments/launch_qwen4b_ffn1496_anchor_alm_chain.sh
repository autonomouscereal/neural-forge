#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base}"
ANCHOR_MODEL="${ANCHOR_MODEL:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_strict25_ffnfold_20260627_191250/alpha05_ffn1496_folded_xreconstruct_mlp_a05}"
TAG="${TAG:-qwen4b_ffn1496_anchor_blockalm_gpu4_20260629}"

cd "$REPO_ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate
mkdir -p "$RUN_ROOT/logs"

if [ ! -f "$ANCHOR_MODEL/config.json" ]; then
  echo "missing_anchor=$ANCHOR_MODEL" >&2
  exit 0
fi

OUT="$RUN_ROOT/$TAG"
LOG="$RUN_ROOT/logs/$TAG.log"
PID="$RUN_ROOT/logs/$TAG.pid"

CUDA_VISIBLE_DEVICES="$GPU_UUID" nohup python -m neural_forge.cli hf-xfamily-chain-anchor-compress \
  "$SOURCE_MODEL" \
  "$ANCHOR_MODEL" \
  "$OUT" \
  --stages block,alm \
  --calibration-file "$RUN_ROOT/calibration_mix.jsonl" \
  --max-calibration-texts 512 \
  --eval-suite "$RUN_ROOT/behavior_suite.json" \
  --max-behavior-probes 200 \
  --max-generation-prompts 100 \
  --max-choice-probes 100 \
  --max-text-evals 96 \
  --max-new-tokens 96 \
  --stage-ppl-gate-mode advisory \
  --promote-policy always \
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
  --alm-steps 260 \
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
