#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
TAG="${TAG:-qwen4b_blank_beststack_02block_topneg_recover_gpu6_20260707}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-$RUN_ROOT/qwen4b_blank_smollm_beststack_qrsinkhorn_module_eora_qmedoid_gpu4_20260706/models/02_block}"
GPU_UUID="${GPU_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"
WAIT_PIDS="${WAIT_PIDS:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_THRESHOLD="${MEM_THRESHOLD:-2000}"
MIN_CUDA_COMPUTE_CAPABILITY="${MIN_CUDA_COMPUTE_CAPABILITY:-7.0}"

CALIBRATION_FILE="${CALIBRATION_FILE:-$RUN_ROOT/calibration_mix_plus_recovery_drills_20260702.jsonl}"
EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-640}"
ALM_STEPS="${ALM_STEPS:-180}"
ALM_LR="${ALM_LR:-5e-7}"
ALM_LM_WEIGHT="${ALM_LM_WEIGHT:-0.08}"
ALM_FIRST_TOKEN_LM_WEIGHT="${ALM_FIRST_TOKEN_LM_WEIGHT:-0.05}"
ALM_RESPONSE_END_LM_WEIGHT="${ALM_RESPONSE_END_LM_WEIGHT:-0.05}"
ALM_FOCUSED_LM_WEIGHT="${ALM_FOCUSED_LM_WEIGHT:-0.04}"
ALM_ANSWER_CONTRASTIVE_WEIGHT="${ALM_ANSWER_CONTRASTIVE_WEIGHT:-0.10}"
ALM_ANSWER_MARGIN_WEIGHT="${ALM_ANSWER_MARGIN_WEIGHT:-0.08}"
ALM_ANSWER_MARGIN="${ALM_ANSWER_MARGIN:-1.5}"
ALM_STEPWISE_DONOR_KD_WEIGHT="${ALM_STEPWISE_DONOR_KD_WEIGHT:-0.025}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/logs"

wait_for_pids() {
  for pid in "$@"; do
    if [[ -z "$pid" || "$pid" == "0" ]]; then
      continue
    fi
    while kill -0 "$pid" 2>/dev/null; do
      echo "$(date -Is) waiting_for_prior_pid=$pid"
      sleep "$POLL_SECONDS"
    done
  done
}

require_supported_cuda_gpu() {
  local compute_cap
  compute_cap="$(nvidia-smi -i "$GPU_UUID" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  if [[ -z "$compute_cap" ]]; then
    echo "missing_cuda_gpu uuid=$GPU_UUID" >&2
    exit 3
  fi
  awk -v got="$compute_cap" -v need="$MIN_CUDA_COMPUTE_CAPABILITY" 'BEGIN { exit !(got + 0 >= need + 0) }' || {
    echo "unsupported_cuda_gpu uuid=$GPU_UUID compute_cap=$compute_cap min_compute_cap=$MIN_CUDA_COMPUTE_CAPABILITY" >&2
    exit 3
  }
  echo "cuda_gpu_ok uuid=$GPU_UUID compute_cap=$compute_cap min_compute_cap=$MIN_CUDA_COMPUTE_CAPABILITY"
}

wait_for_gpu() {
  while true; do
    local used busy
    used="$(nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || true)"
    busy="$(
      nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader 2>/dev/null |
        awk -F, -v gpu="$GPU_UUID" '$2 ~ gpu && $3 ~ /python/ {print $1}' |
        wc -l
    )"
    if [[ -n "$used" && "$used" -le "$MEM_THRESHOLD" && "$busy" = "0" ]]; then
      break
    fi
    echo "$(date -Is) waiting_for_gpu_uuid=$GPU_UUID memory_used_mb=${used:-unknown} busy_python=$busy threshold_mb=$MEM_THRESHOLD"
    sleep "$POLL_SECONDS"
  done
}

if [[ ! -d "$CANDIDATE_MODEL" ]]; then
  echo "missing_candidate_model=$CANDIDATE_MODEL" >&2
  exit 4
fi
if [[ ! -f "$CALIBRATION_FILE" ]]; then
  echo "missing_calibration_file=$CALIBRATION_FILE" >&2
  exit 4
fi
if [[ ! -f "$EVAL_SUITE_FILE" ]]; then
  echo "missing_eval_suite=$EVAL_SUITE_FILE" >&2
  exit 4
fi

echo "topneg_recover_queue_start=$(date -Is) tag=$TAG source=$SOURCE_MODEL candidate=$CANDIDATE_MODEL gpu=$GPU_UUID wait_pids=${WAIT_PIDS:-none} revision=$(cat .deployed_revision 2>/dev/null || true)"
wait_for_pids $WAIT_PIDS
require_supported_cuda_gpu
wait_for_gpu

source /home/cereal/model-surgery-lab/venv/bin/activate
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"

OUT="$RUN_ROOT/$TAG"
mkdir -p "$OUT"

python -m neural_forge.cli hf-xfamily-chain-anchor-compress \
  "$SOURCE_MODEL" \
  "$CANDIDATE_MODEL" \
  "$OUT" \
  --stages alm \
  --calibration-file "$CALIBRATION_FILE" \
  --max-calibration-texts "$MAX_CALIBRATION_TEXTS" \
  --eval-suite "$EVAL_SUITE_FILE" \
  --max-behavior-probes 1000 \
  --max-generation-prompts 1000 \
  --max-choice-probes 250 \
  --max-text-evals 128 \
  --max-new-tokens 96 \
  --stage-ppl-gate-mode advisory \
  --promote-policy always \
  --stage-input-policy sequential \
  --alm-steps "$ALM_STEPS" \
  --alm-batch-size 1 \
  --alm-max-length 128 \
  --alm-lr "$ALM_LR" \
  --alm-anchor-kd-weight 0.04 \
  --alm-lm-weight "$ALM_LM_WEIGHT" \
  --alm-first-token-lm-weight "$ALM_FIRST_TOKEN_LM_WEIGHT" \
  --alm-response-end-lm-weight "$ALM_RESPONSE_END_LM_WEIGHT" \
  --alm-focused-lm-weight "$ALM_FOCUSED_LM_WEIGHT" \
  --alm-dual-space-kd-weight 0.08 \
  --alm-attention-influence-weight 0.02 \
  --alm-attention-influence-layer middle \
  --alm-kd-weight 0.06 \
  --alm-projected-vocab-kd-weight 0.04 \
  --alm-focused-projected-vocab-kd-weight 0.04 \
  --alm-span-hidden-kd-weight 0.10 \
  --alm-token-kd-weight 0.05 \
  --alm-token-focus numbers \
  --alm-token-max-pairs 512 \
  --alm-token-teacher-confidence-floor 0.05 \
  --alm-answer-contrastive-weight "$ALM_ANSWER_CONTRASTIVE_WEIGHT" \
  --alm-answer-contrastive-max-negatives 8 \
  --alm-answer-first-token-margin-weight "$ALM_ANSWER_MARGIN_WEIGHT" \
  --alm-answer-first-token-margin "$ALM_ANSWER_MARGIN" \
  --alm-answer-first-token-max-negatives 10 \
  --alm-answer-first-token-top-negatives 24 \
  --alm-teacher-rank-weight 0.08 \
  --alm-teacher-rank-temperature 0.7 \
  --alm-teacher-rank-max-negatives 8 \
  --alm-teacher-generate-lm-weight 0.015 \
  --alm-teacher-generate-max-new-tokens 48 \
  --alm-teacher-generate-loss-mode auto \
  --alm-stepwise-donor-kd-weight "$ALM_STEPWISE_DONOR_KD_WEIGHT" \
  --alm-stepwise-donor-max-new-tokens 16 \
  --alm-on-policy-anchor-kd-weight 0.01 \
  --alm-on-policy-unlikelihood-weight 0.002 \
  --alm-on-policy-hidden-kd-weight 0.02 \
  --alm-on-policy-max-new-tokens 24 \
  --alm-on-policy-interval 8 \
  --alm-train-scope all \
  --alm-response-loss-mode auto \
  --alm-save-best-behavior \
  --alm-behavior-eval-interval 40 \
  --dtype float16 \
  --candidate-dtype float32 \
  --device cuda:0

FINAL_MODEL="$(
  python - <<'PY' "$OUT"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = out / "reports" / "chain_anchor_summary.json"
model = ""
if summary.exists():
    data = json.loads(summary.read_text(encoding="utf-8"))
    model = str(data.get("best_model_dir") or "")
if not model:
    for rel in ("models/01_alm", "models/00_alm", "models/02_alm"):
        candidate = out / rel
        if candidate.is_dir():
            model = str(candidate)
            break
print(model)
PY
)"
if [[ -z "$FINAL_MODEL" || ! -d "$FINAL_MODEL" ]]; then
  echo "missing_final_model=$FINAL_MODEL out=$OUT" >&2
  exit 5
fi

wait_for_gpu
MODEL_DIR="$FINAL_MODEL" \
TAG="${TAG}_exactgen" \
RUN_ROOT="$RUN_ROOT" \
REPO_ROOT="$REPO_ROOT" \
GPU_UUID="$GPU_UUID" \
DEVICE=cuda:0 \
DTYPE=float16 \
MAX_BEHAVIOR_PROBES=1000 \
MAX_GENERATION_PROMPTS=1000 \
MAX_CHOICE_PROBES=250 \
MAX_TEXT_EVALS=128 \
MAX_NEW_TOKENS=96 \
bash "$REPO_ROOT/experiments/eval_model_broad_and_manual.sh"

echo "topneg_recover_queue_done=$(date -Is) tag=$TAG final_model=$FINAL_MODEL"
