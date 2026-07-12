#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
TAG="${TAG:-qwen4b_blank_smollm_signed_cka_geomean_full_gpu4_20260702}"
WAIT_PIDS="${WAIT_PIDS:-}"

echo "queue_start=$(date -Is) tag=$TAG gpu=$GPU_UUID wait_pids=${WAIT_PIDS:-none}"

if [[ -n "$WAIT_PIDS" ]]; then
  for pid in $WAIT_PIDS; do
    while kill -0 "$pid" 2>/dev/null; do
      echo "$(date -Is) waiting_for_prior_pid=$pid"
      sleep 300
    done
  done
fi

while true; do
  busy=$(
    nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader |
      awk -F, -v gpu="$GPU_UUID" '$2 ~ gpu && $3 ~ /python/ {print $1}' |
      wc -l
  )
  if [[ "$busy" = "0" ]]; then
    break
  fi
  echo "$(date -Is) waiting_for_gpu=$GPU_UUID"
  sleep 300
done

cd "$REPO_ROOT"
export RUN_ROOT="$ROOT"
export REPO_ROOT
export GPU_UUID
export TAG

export INIT_PROJECTION_SCALE=geomean-rms
export INIT_PROJECTION_BLEND_ALPHA=1.0
export INIT_ACTIVATION_FOLD_ASSIGNMENT=signed-similarity
export INIT_TRANSPORT_SUPPORT=centroid
export INIT_ACTIVATION_DOMAIN_BALANCE=keyword
export INIT_DONOR_LAYER_GROUP_STRATEGY=activation-cka
export BLOCK_DONOR_LAYER_GROUP_STRATEGY=activation-cka

export RESIDUAL_MODE=eora-pre
export RESIDUAL_ALPHA=0.01
export RESIDUAL_RANK=16
export RESIDUAL_MIN_SCALE=0.45
export RESIDUAL_MAX_SCALE=1.65
export RESIDUAL_LINEAR_RIDGE=1e-3

export PRE_RECONSTRUCTION_TARGET=block-interface
export PRE_RECONSTRUCTION_RIDGE=0.006
export PRE_RECONSTRUCTION_ALPHA=0.02
export PRE_RECONSTRUCTION_OBJECTIVE=cosine
export PRE_RECONSTRUCTION_COSINE_STEPS=24
export PRE_RECONSTRUCTION_COSINE_LR=0.01
export PRE_RECONSTRUCTION_COSINE_L2=0.0001
export PRE_RECONSTRUCTION_FIT_GATE=validation
export PRE_RECONSTRUCTION_VALIDATION_FRACTION=0.25
export PRE_RECONSTRUCTION_MIN_VALIDATION_ROWS=16
export PRE_RECONSTRUCTION_MAX_TOKENS_PER_LAYER=768
export PRE_RECONSTRUCTION_MIN_RELATIVE_IMPROVEMENT=0.01
export PRE_RECONSTRUCTION_MAX_COSINE_DROP=0.02

export BLOCK_STEPS_PER_LAYER=5
export BLOCK_LR=5e-7
export BLOCK_ALM_KD_WEIGHT=0.02
export BLOCK_PROJECTED_VOCAB_KD_WEIGHT=0.02
export BLOCK_BYTE_KD_WEIGHT=0.01
export BLOCK_BLOCK_IO_KD_WEIGHT=0.02
export BLOCK_MODULE_OUTPUT_KD_WEIGHT=0.02
export BLOCK_MODULE_OUTPUT_TARGETS=both
export BLOCK_MODULE_OUTPUT_TRANSPORT=both
export BLOCK_MODULE_OUTPUT_SINKHORN_ITERATIONS=12

export BLOCK_ON_POLICY_ANCHOR_KD_WEIGHT=0.005
export BLOCK_ON_POLICY_UNLIKELIHOOD_WEIGHT=0.001
export BLOCK_ON_POLICY_HIDDEN_KD_WEIGHT=0.015
export BLOCK_ON_POLICY_INTERVAL=3
export BLOCK_ON_POLICY_MAX_NEW_TOKENS=16
export BLOCK_STEPWISE_DONOR_KD_WEIGHT=0.005
export BLOCK_STEPWISE_DONOR_INTERVAL=4
export BLOCK_TEACHER_GENERATE_LM_WEIGHT=0.003
export BLOCK_TEACHER_GENERATE_HIDDEN_KD_WEIGHT=0.015
export BLOCK_TEACHER_GENERATE_PROJECTED_VOCAB_KD_WEIGHT=0.003
export BLOCK_TEACHER_GENERATE_BYTE_KD_WEIGHT=0.003
export BLOCK_TEACHER_GENERATE_MAX_NEW_TOKENS=48
export BLOCK_TOKEN_FOCUS=numbers
export BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT=0.01
export BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT=0.003
export BLOCK_FOCUSED_LM_WEIGHT=0.005
export BLOCK_BYTE_HIDDEN_KD_WEIGHT=0.003

export ALM_STEPS=260
export ALM_LR=5e-7
export ALM_LM_WEIGHT=0.12
export ALM_KD_WEIGHT=0.08
export ALM_PROJECTED_VOCAB_KD_WEIGHT=0.04
export ALM_SPAN_HIDDEN_KD_WEIGHT=0.10
export ALM_TRAIN_SCOPE=all
export ALM_RESPONSE_LOSS_MODE=auto

bash experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh
