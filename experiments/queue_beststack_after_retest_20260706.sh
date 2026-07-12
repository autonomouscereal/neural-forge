#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
VARIANT="${VARIANT:-qwen}"
GPU_UUID="${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
WAIT_PIDS="${WAIT_PIDS:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MEM_THRESHOLD="${MEM_THRESHOLD:-2000}"
IGNORE_GPU_PIDS="${IGNORE_GPU_PIDS:-12681}"
MIN_CUDA_COMPUTE_CAPABILITY="${MIN_CUDA_COMPUTE_CAPABILITY:-7.0}"
BESTSTACK_EVAL_DIR="${BESTSTACK_EVAL_DIR:-$RUN_ROOT/evals/beststack_20260706}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-$RUN_ROOT/locks}"
GPU_STARTUP_SETTLE_SECONDS="${GPU_STARTUP_SETTLE_SECONDS:-30}"
MIN_HOST_MEM_AVAILABLE_GB="${MIN_HOST_MEM_AVAILABLE_GB:-48}"
MAX_HOST_SWAP_USED_GB="${MAX_HOST_SWAP_USED_GB:-1}"
MAX_CONCURRENT_HEAVY_FOLDS="${MAX_CONCURRENT_HEAVY_FOLDS:-2}"
TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-quantile-medoid}"
RESIDUAL_MODE="${RESIDUAL_MODE:-module-eora-pre}"
ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT="${ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT:-0.05}"
ALM_ANSWER_FIRST_TOKEN_TOP_NEGATIVES="${ALM_ANSWER_FIRST_TOKEN_TOP_NEGATIVES:-24}"
# Best-stack proof lanes target the requested child architecture size. Init can
# temporarily untie the head for cross-tokenizer folding, but the promoted
# artifact should be tied and re-evaluated unless a caller explicitly overrides.
FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}"

case "$VARIANT" in
  qwen)
    TAG="${TAG:-qwen4b_blank_smollm_beststack_module_eora_qmedoid_gpu4_20260706}"
    SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base}"
    SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-}"
    RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-primary}"
    LAUNCH_SCRIPT="experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh"
    PARENT_REPORT_CANDIDATES="$RUN_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json;$RUN_ROOT/evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json"
    MODEL_ENTRIES="child=$RUN_ROOT/$TAG;qwen_parent=Qwen/Qwen3-4B-Base;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;mathdrill_full=$RUN_ROOT/qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701;cleanopen_full=$RUN_ROOT/qwen4b_cleanopen_full_02_alm_exactgen_20260702"
    ;;
  qwen_gpt2)
    TAG="${TAG:-qwen4b_gpt2_multidonor_beststack_module_eora_qmedoid_gpu4_20260706}"
    SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,openai-community/gpt2}"
    SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.85,0.15}"
    RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
    LAUNCH_SCRIPT="experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh"
    PARENT_REPORT_CANDIDATES="$RUN_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json;$RUN_ROOT/evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json"
    MODEL_ENTRIES="child=$RUN_ROOT/$TAG;qwen_parent=Qwen/Qwen3-4B-Base;gpt2_donor=openai-community/gpt2;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;mathdrill_full=$RUN_ROOT/qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701"
    ;;
  qwen_lfm)
    TAG="${TAG:-qwen4b_lfm230m_multidonor_beststack_module_eora_qmedoid_gpu0_20260707}"
    SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-4B-Base,LiquidAI/LFM2.5-230M}"
    SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-0.70,0.30}"
    RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}"
    LAUNCH_SCRIPT="experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh"
    PARENT_REPORT_CANDIDATES="$RUN_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json;$RUN_ROOT/evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json"
    MODEL_ENTRIES="child=$RUN_ROOT/$TAG;qwen_parent=Qwen/Qwen3-4B-Base;lfm230m_donor=LiquidAI/LFM2.5-230M;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;mathdrill_full=$RUN_ROOT/qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701"
    ;;
  lfm)
    TAG="${TAG:-lfm230m_blank_smollm_beststack_module_eora_qmedoid_gpu6_20260706}"
    SOURCE_MODEL="${SOURCE_MODEL:-LiquidAI/LFM2.5-230M}"
    ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/blank_smollm2_135m_seed0}"
    SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-}"
    RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-primary}"
    LAUNCH_SCRIPT="experiments/launch_lfm230m_blank_smollm_chain.sh"
    PARENT_REPORT_CANDIDATES="$RUN_ROOT/evals/eval_donor_lfm230m_strict_20260705_1000x1000x250.json;$RUN_ROOT/evals/eval_donor_lfm230m_exactgen_20260702_1000x1000x250.json"
    MODEL_ENTRIES="child=$RUN_ROOT/$TAG;lfm230m_donor=LiquidAI/LFM2.5-230M;blank_random=$RUN_ROOT/blank_smollm2_135m_seed0;mathdrill_full=$RUN_ROOT/qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Expected qwen, qwen_gpt2, qwen_lfm, or lfm." >&2
    exit 2
    ;;
esac

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
  if [[ "${MIN_CUDA_COMPUTE_CAPABILITY:-0}" == "0" ]]; then
    return 0
  fi
  local row cc name
  row=$(
    nvidia-smi -i "$GPU_UUID" --query-gpu=compute_cap,name --format=csv,noheader,nounits 2>/dev/null |
      head -n 1 || true
  )
  cc="${row%%,*}"
  name="${row#*,}"
  cc="${cc//[[:space:]]/}"
  name="${name#"${name%%[![:space:]]*}"}"
  if [[ -z "$row" || -z "$cc" || "$cc" == "$row" ]]; then
    echo "unsupported_cuda_gpu_check_failed uuid=$GPU_UUID min_compute_cap=$MIN_CUDA_COMPUTE_CAPABILITY" >&2
    exit 78
  fi
  if ! awk -v cc="$cc" -v min="$MIN_CUDA_COMPUTE_CAPABILITY" 'BEGIN { exit !((cc + 0) >= (min + 0)) }'; then
    echo "unsupported_cuda_gpu uuid=$GPU_UUID name=$name compute_cap=$cc min_compute_cap=$MIN_CUDA_COMPUTE_CAPABILITY" >&2
    exit 78
  fi
  echo "cuda_gpu_ok uuid=$GPU_UUID name=$name compute_cap=$cc min_compute_cap=$MIN_CUDA_COMPUTE_CAPABILITY"
}

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || echo 999999)
    busy=$(
      nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader 2>/dev/null |
        awk -F, -v gpu="$GPU_UUID" -v ignore="$IGNORE_GPU_PIDS" '
          BEGIN {
            split(ignore, ignored, /[ ,]+/)
            for (idx in ignored) {
              if (ignored[idx] != "") {
                ignored_pid[ignored[idx]] = 1
              }
            }
          }
          {
            pid = $1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
            if ($2 ~ gpu && $3 ~ /python/ && !(pid in ignored_pid)) {
              print pid
            }
          }' |
        wc -l
    )
    if [[ -n "$used" && "$used" -le "$MEM_THRESHOLD" && "$busy" = "0" ]]; then
      break
    fi
    echo "$(date -Is) waiting_for_gpu_uuid=$GPU_UUID memory_used_mb=${used:-unknown} busy_python=$busy threshold_mb=$MEM_THRESHOLD"
    sleep "$POLL_SECONDS"
  done
}

host_capacity_status() {
  awk -v min_gb="$MIN_HOST_MEM_AVAILABLE_GB" -v max_swap_gb="$MAX_HOST_SWAP_USED_GB" '
    /MemAvailable:/ { mem_available_kb = $2 }
    /SwapTotal:/ { swap_total_kb = $2 }
    /SwapFree:/ { swap_free_kb = $2 }
    END {
      min_available_kb = min_gb * 1024 * 1024
      max_swap_used_kb = max_swap_gb * 1024 * 1024
      swap_used_kb = swap_total_kb - swap_free_kb
      printf "mem_available_gb=%.1f min_mem_available_gb=%s swap_used_gb=%.1f max_swap_used_gb=%s\n",
        mem_available_kb / 1024 / 1024, min_gb, swap_used_kb / 1024 / 1024, max_swap_gb
      if (mem_available_kb < min_available_kb || swap_used_kb > max_swap_used_kb) {
        exit 1
      }
    }
  ' /proc/meminfo
}

active_heavy_fold_count() {
  ps -eo pid=,ppid=,comm=,args= |
    awk -v self="$$" -v parent="$PPID" '
      {
        pid = $1
        ppid = $2
        comm = $3
        args = $0
        sub(/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", args)
        if (pid == self || pid == parent) {
          next
        }
        if (comm !~ /python/) {
          next
        }
        if (args ~ /[[:space:]]-m[[:space:]]+neural_forge\.cli[[:space:]]+hf-xfamily-chain-anchor-compress/) {
          count += 1
        }
      }
      END { print count + 0 }
    '
}

wait_for_host_capacity() {
  local status active mem_ok
  while true; do
    if status="$(host_capacity_status 2>&1)"; then
      mem_ok=1
    else
      mem_ok=0
    fi
    active="$(active_heavy_fold_count)"
    if [[ "$active" -lt "$MAX_CONCURRENT_HEAVY_FOLDS" && "$mem_ok" == "1" ]]; then
      echo "host_capacity_ok active_heavy_folds=$active max_concurrent_heavy_folds=$MAX_CONCURRENT_HEAVY_FOLDS $status"
      break
    fi
    echo "$(date -Is) waiting_for_host_capacity active_heavy_folds=$active max_concurrent_heavy_folds=$MAX_CONCURRENT_HEAVY_FOLDS $status"
    sleep "$POLL_SECONDS"
  done
}

sanitize_lock_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

wait_for_launched_gpu_process() {
  local pid="$1"
  local deadline=$((SECONDS + GPU_STARTUP_SETTLE_SECONDS))
  if [[ -z "$pid" || "$pid" == "0" ]]; then
    echo "gpu_launch_lock_settle_no_pid seconds=$GPU_STARTUP_SETTLE_SECONDS"
    sleep "$GPU_STARTUP_SETTLE_SECONDS"
    return 0
  fi
  while (( SECONDS < deadline )); do
    if nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null |
      awk -F, -v pid="$pid" -v gpu="$GPU_UUID" '
        {
          p = $1
          g = $2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", p)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", g)
          if (p == pid && g == gpu) {
            found = 1
          }
        }
        END { exit !found }
      '; then
      echo "gpu_launch_process_visible pid=$pid gpu=$GPU_UUID"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "gpu_launch_process_exited_before_visible pid=$pid gpu=$GPU_UUID"
      return 0
    fi
    sleep 2
  done
  echo "gpu_launch_lock_settle_timeout pid=$pid gpu=$GPU_UUID seconds=$GPU_STARTUP_SETTLE_SECONDS"
}

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/logs" "$BESTSTACK_EVAL_DIR"

echo "beststack_queue_start=$(date -Is) variant=$VARIANT tag=$TAG gpu=$GPU_UUID wait_pids=${WAIT_PIDS:-none} revision=$(cat .deployed_revision 2>/dev/null || true)"
wait_for_pids $WAIT_PIDS
mkdir -p "$GPU_LOCK_DIR"
GPU_LOCK_FILE="$GPU_LOCK_DIR/gpu_$(sanitize_lock_name "$GPU_UUID").launch.lock"
exec 9>"$GPU_LOCK_FILE"
echo "waiting_for_gpu_launch_lock file=$GPU_LOCK_FILE"
flock 9
echo "acquired_gpu_launch_lock file=$GPU_LOCK_FILE"
require_supported_cuda_gpu
wait_for_host_capacity
wait_for_gpu

export RUN_ROOT
export REPO_ROOT
export GPU_UUID
export TAG
export MEM_THRESHOLD
export SOURCE_MODEL
export SOURCE_WEIGHTS
export RECOVERY_SOURCE_MODE
export ANCHOR_MODEL="${ANCHOR_MODEL:-$RUN_ROOT/blank_smollm2_135m_seed0}"

export CALIBRATION_FILE="${CALIBRATION_FILE:-$RUN_ROOT/calibration_mix_plus_recovery_drills_20260702.jsonl}"
export EVAL_SUITE_FILE="${EVAL_SUITE_FILE:-$RUN_ROOT/behavior_suite_exactgen_20260702.json}"
export MAX_CALIBRATION_TEXTS="${MAX_CALIBRATION_TEXTS:-1024}"
export MAX_BEHAVIOR_PROBES="${MAX_BEHAVIOR_PROBES:-1000}"
export MAX_GENERATION_PROMPTS="${MAX_GENERATION_PROMPTS:-1000}"
export MAX_CHOICE_PROBES="${MAX_CHOICE_PROBES:-250}"
export MAX_TEXT_EVALS="${MAX_TEXT_EVALS:-128}"
export FINAL_OUTPUT_HEAD_POLICY
# CoM-style folding works best when each repair stage sees the model state
# produced by the previous stage, so default to sequential rather than
# repeatedly re-anchoring to the incumbent.
export STAGE_INPUT_POLICY="${STAGE_INPUT_POLICY:-sequential}"
export PROMOTE_POLICY="${PROMOTE_POLICY:-always}"

export INIT_PROJECTION_SCALE="${INIT_PROJECTION_SCALE:-geomean-rms}"
export INIT_PROJECTION_BLEND_ALPHA="${INIT_PROJECTION_BLEND_ALPHA:-1.0}"
export INIT_ACTIVATION_FOLD_ASSIGNMENT="${INIT_ACTIVATION_FOLD_ASSIGNMENT:-balanced-signed-similarity}"
export INIT_FOLD_STAT_REPAIR="${INIT_FOLD_STAT_REPAIR:-rms}"
export INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-sinkhorn}"
export INIT_TRANSPORT_SUPPORT="${INIT_TRANSPORT_SUPPORT:-$TRANSPORT_SUPPORT}"
if [[ -z "${INIT_ATTENTION_HEAD_DIM_TRANSPORT:-}" ]]; then
  if [[ "$INIT_TRANSPORT_SUPPORT" == "domain-quantile-medoid" || "$INIT_TRANSPORT_SUPPORT" == "domain-diverse-medoid" ]]; then
    INIT_ATTENTION_HEAD_DIM_TRANSPORT="qk-vo"
  else
    INIT_ATTENTION_HEAD_DIM_TRANSPORT="none"
  fi
fi
export INIT_ATTENTION_HEAD_DIM_TRANSPORT
if [[ -z "${ACT_TARGET:-}" && ( "$INIT_TRANSPORT_SUPPORT" == "domain-quantile-medoid" || "$INIT_TRANSPORT_SUPPORT" == "domain-diverse-medoid" ) ]]; then
  ACT_TARGET="all"
fi
export ACT_TARGET="${ACT_TARGET:-mlp-output}"
export INIT_ACTIVATION_FIT_LINEARS="${INIT_ACTIVATION_FIT_LINEARS:-1}"
export INIT_TOKEN_INIT_MODE="${INIT_TOKEN_INIT_MODE:-donor-tokenized-weighted}"
export INIT_TOKEN_FALLBACK_MODE="${INIT_TOKEN_FALLBACK_MODE:-donor-mean}"
export INIT_ACTIVATION_DOMAIN_BALANCE="${INIT_ACTIVATION_DOMAIN_BALANCE:-keyword}"
export INIT_DONOR_LAYER_GROUP_STRATEGY="${INIT_DONOR_LAYER_GROUP_STRATEGY:-activation-cka-weighted}"
export INIT_DONOR_LAYER_OUTPUT_SCALE="${INIT_DONOR_LAYER_OUTPUT_SCALE:-group-count}"
export BLOCK_DONOR_LAYER_GROUP_STRATEGY="${BLOCK_DONOR_LAYER_GROUP_STRATEGY:-activation-cka-weighted}"
export LRC_MAX_DELTA_RMS_RATIO="${LRC_MAX_DELTA_RMS_RATIO:-1.5}"

export BLOCK_HIDDEN_PROJECTION_SCOPE="${BLOCK_HIDDEN_PROJECTION_SCOPE:-layer-input-gated}"
export PRE_RECONSTRUCTION_TARGET="${PRE_RECONSTRUCTION_TARGET:-all}"
export PRE_RECONSTRUCTION_OBJECTIVE="${PRE_RECONSTRUCTION_OBJECTIVE:-cosine}"
export PRE_RECONSTRUCTION_ALPHA="${PRE_RECONSTRUCTION_ALPHA:-0.02}"
export PRE_RECONSTRUCTION_RIDGE="${PRE_RECONSTRUCTION_RIDGE:-0.006}"
export PRE_RECONSTRUCTION_FIT_GATE="${PRE_RECONSTRUCTION_FIT_GATE:-validation}"
export PRE_RECONSTRUCTION_MIN_FITTED_COSINE="${PRE_RECONSTRUCTION_MIN_FITTED_COSINE:-0.05}"
export RESIDUAL_MODE
export RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.01}"
export RESIDUAL_RANK="${RESIDUAL_RANK:-16}"
export RESIDUAL_MIN_SCALE="${RESIDUAL_MIN_SCALE:-0.45}"
export RESIDUAL_MAX_SCALE="${RESIDUAL_MAX_SCALE:-1.65}"
export RESIDUAL_LINEAR_RIDGE="${RESIDUAL_LINEAR_RIDGE:-1e-3}"

export BLOCK_STEPS_PER_LAYER="${BLOCK_STEPS_PER_LAYER:-5}"
export BLOCK_LR="${BLOCK_LR:-5e-7}"
export BLOCK_ALM_KD_WEIGHT="${BLOCK_ALM_KD_WEIGHT:-0.02}"
export BLOCK_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_PROJECTED_VOCAB_KD_WEIGHT:-0.02}"
export BLOCK_BYTE_KD_WEIGHT="${BLOCK_BYTE_KD_WEIGHT:-0.01}"
export BLOCK_BLOCK_IO_KD_WEIGHT="${BLOCK_BLOCK_IO_KD_WEIGHT:-0.02}"
export BLOCK_MODULE_OUTPUT_KD_WEIGHT="${BLOCK_MODULE_OUTPUT_KD_WEIGHT:-0.02}"
export BLOCK_MODULE_OUTPUT_TARGETS="${BLOCK_MODULE_OUTPUT_TARGETS:-both}"
export BLOCK_MODULE_OUTPUT_TRANSPORT="${BLOCK_MODULE_OUTPUT_TRANSPORT:-both}"
export BLOCK_BYTE_HIDDEN_KD_WEIGHT="${BLOCK_BYTE_HIDDEN_KD_WEIGHT:-0.004}"
export BLOCK_BYTE_HIDDEN_PROBE_MODE="${BLOCK_BYTE_HIDDEN_PROBE_MODE:-trainable}"
export BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT="${BLOCK_BYTE_HIDDEN_PROBE_ANCHOR_WEIGHT:-0.002}"
export BLOCK_NONFINITE_STEP_POLICY="${BLOCK_NONFINITE_STEP_POLICY:-skip-step}"
export BLOCK_MAX_NONFINITE_STEPS_PER_LAYER="${BLOCK_MAX_NONFINITE_STEPS_PER_LAYER:-3}"
export BLOCK_TEACHER_GENERATE_LM_WEIGHT="${BLOCK_TEACHER_GENERATE_LM_WEIGHT:-0.02}"
export BLOCK_TEACHER_GENERATE_HIDDEN_KD_WEIGHT="${BLOCK_TEACHER_GENERATE_HIDDEN_KD_WEIGHT:-0.02}"
export BLOCK_TEACHER_GENERATE_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_TEACHER_GENERATE_PROJECTED_VOCAB_KD_WEIGHT:-0.006}"
export BLOCK_TEACHER_GENERATE_BYTE_KD_WEIGHT="${BLOCK_TEACHER_GENERATE_BYTE_KD_WEIGHT:-0.006}"
export BLOCK_TEACHER_GENERATE_MAX_NEW_TOKENS="${BLOCK_TEACHER_GENERATE_MAX_NEW_TOKENS:-64}"
export BLOCK_TEACHER_GENERATE_LOSS_MODE="${BLOCK_TEACHER_GENERATE_LOSS_MODE:-auto}"
export BLOCK_TOKEN_FOCUS="${BLOCK_TOKEN_FOCUS:-numbers}"
export BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT="${BLOCK_FOCUSED_BLOCK_HIDDEN_KD_WEIGHT:-0.01}"
export BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT="${BLOCK_FOCUSED_PROJECTED_VOCAB_KD_WEIGHT:-0.003}"
export BLOCK_FOCUSED_LM_WEIGHT="${BLOCK_FOCUSED_LM_WEIGHT:-0.005}"
# Blank-target beststack lanes need the folded child logits trained, not only
# hidden geometry. Keep this at the queue layer so callers can still run
# no-head ablations by exporting 0.
export BLOCK_TRAIN_OUTPUT_HEAD_WITH_KD="${BLOCK_TRAIN_OUTPUT_HEAD_WITH_KD:-1}"
export BLOCK_UNTIE_OUTPUT_HEAD_FOR_KD="${BLOCK_UNTIE_OUTPUT_HEAD_FOR_KD:-1}"
export BLOCK_ALLOW_TIED_OUTPUT_HEAD_TRAINING="${BLOCK_ALLOW_TIED_OUTPUT_HEAD_TRAINING:-0}"

export ALM_STEPS="${ALM_STEPS:-260}"
export ALM_LR="${ALM_LR:-5e-7}"
export ALM_LM_WEIGHT="${ALM_LM_WEIGHT:-0.12}"
export ALM_ANCHOR_KD_WEIGHT="${ALM_ANCHOR_KD_WEIGHT:-0.08}"
export ALM_DUAL_SPACE_KD_WEIGHT="${ALM_DUAL_SPACE_KD_WEIGHT:-0.12}"
export ALM_KD_WEIGHT="${ALM_KD_WEIGHT:-0.08}"
export ALM_PROJECTED_VOCAB_KD_WEIGHT="${ALM_PROJECTED_VOCAB_KD_WEIGHT:-0.04}"
export ALM_SPAN_HIDDEN_KD_WEIGHT="${ALM_SPAN_HIDDEN_KD_WEIGHT:-0.10}"
export ALM_SPAN_KD_WEIGHT="${ALM_SPAN_KD_WEIGHT:-0.04}"
export ALM_ATTENTION_INFLUENCE_WEIGHT="${ALM_ATTENTION_INFLUENCE_WEIGHT:-0.04}"
export ALM_ATTENTION_INFLUENCE_LAYER="${ALM_ATTENTION_INFLUENCE_LAYER:-middle}"
export ALM_TOKEN_FOCUS="${ALM_TOKEN_FOCUS:-numbers}"
export ALM_ANSWER_FIRST_TOKEN_MARGIN_WEIGHT
export ALM_ANSWER_FIRST_TOKEN_MARGIN="${ALM_ANSWER_FIRST_TOKEN_MARGIN:-1.0}"
export ALM_ANSWER_FIRST_TOKEN_TOP_NEGATIVES
export ALM_ANSWER_FIRST_TOKEN_MAX_NEGATIVES="${ALM_ANSWER_FIRST_TOKEN_MAX_NEGATIVES:-8}"
export ALM_TEACHER_GENERATE_LM_WEIGHT="${ALM_TEACHER_GENERATE_LM_WEIGHT:-0.04}"
export ALM_TEACHER_GENERATE_MAX_NEW_TOKENS="${ALM_TEACHER_GENERATE_MAX_NEW_TOKENS:-64}"
export ALM_TEACHER_GENERATE_LOSS_MODE="${ALM_TEACHER_GENERATE_LOSS_MODE:-auto}"
export ALM_TRAIN_SCOPE="${ALM_TRAIN_SCOPE:-all}"
export ALM_RESPONSE_LOSS_MODE="${ALM_RESPONSE_LOSS_MODE:-auto}"

bash "$LAUNCH_SCRIPT"
RUN_PID_FILE="$RUN_ROOT/logs/$TAG.pid"
RUN_PID="$(tr -dc '0-9\n' < "$RUN_PID_FILE" | head -n 1 || true)"
echo "beststack_model_launched tag=$TAG run_pid=${RUN_PID:-unknown} pid_file=$RUN_PID_FILE"
wait_for_launched_gpu_process "${RUN_PID:-}"
exec 9>&-

RETENTION_LOG="$RUN_ROOT/logs/${TAG}_beststack_retention.queue.log"
(
  export TAG
  export MODEL_DIR="$RUN_ROOT/$TAG"
  export QUEUE_PID_FILE="$RUN_PID_FILE"
  export GPU_UUID
  export EVAL_DIR="$BESTSTACK_EVAL_DIR"
  export EVAL_SUITE_FILE
  export PARENT_REPORT_CANDIDATES
  export BASELINE_REPORTS="qwen_parent_strict=$RUN_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json;gpt2_donor_strict=$RUN_ROOT/evals/eval_donor_gpt2_strict_20260705_1000x1000x250.json;lfm230m_donor_strict=$RUN_ROOT/evals/eval_donor_lfm230m_strict_20260705_1000x1000x250.json;blank_random_strict=$RUN_ROOT/evals/eval_blank_smollm135m_random_strict_20260705_1000x1000x250.json;mathdrill_full=$RUN_ROOT/evals/eval_mathdrill_eora_geom_full_fix_best_20260701_1000x1000x250.json;cleanopen_full=$RUN_ROOT/evals/eval_cleanopen_full_02_alm_exactgen_20260702_1000x1000x250.json"
  export REPORT="$BESTSTACK_EVAL_DIR/eval_${TAG}_strict_20260706_1000x1000x250.json"
  export MANUAL_JSON="$BESTSTACK_EVAL_DIR/manual_${TAG}_strict_20260706.json"
  export OUT_DIR="$BESTSTACK_EVAL_DIR/retention_${TAG}_strict_20260706"
  export POLL_SECONDS
  export IGNORE_GPU_PIDS
  bash experiments/queue_eval_retention_compare.sh
) < /dev/null > "$RETENTION_LOG" 2>&1 &
RETENTION_PID=$!
echo "$RETENTION_PID" > "$RUN_ROOT/logs/${TAG}_beststack_retention.queue.pid"
echo "beststack_retention_queued pid=$RETENTION_PID log=$RETENTION_LOG"

LMEVAL_LOG="$RUN_ROOT/logs/${TAG}_beststack_lmeval.queue.log"
(
  export TAG
  export MODEL_DIR="$RUN_ROOT/$TAG"
  export WAIT_PIDS="$RETENTION_PID"
  export GPU_UUID
  export OUTPUT_DIR="$RUN_ROOT/lm_eval_${TAG}_20260706"
  export LIMIT="$LM_EVAL_LIMIT"
  export MODEL_ENTRIES
  export POLL_SECONDS
  export IGNORE_GPU_PIDS
  bash experiments/queue_lm_eval_after_retention.sh
) < /dev/null > "$LMEVAL_LOG" 2>&1 &
LMEVAL_PID=$!
echo "$LMEVAL_PID" > "$RUN_ROOT/logs/${TAG}_beststack_lmeval.queue.pid"
echo "beststack_lmeval_queued pid=$LMEVAL_PID log=$LMEVAL_LOG"
echo "beststack_queue_done=$(date -Is) tag=$TAG"
