#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
BRANCH="${BRANCH:-codex/lrc-blank-compression}"
FETCH_BRANCH="${FETCH_BRANCH:-1}"
START_LANES="${START_LANES:-true_lrc,oldbest,lfm_blank}"

GPU0_UUID="${GPU0_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
GPU4_UUID="${GPU4_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
GPU6_UUID="${GPU6_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"

cd "$REPO_ROOT"

if [ "$FETCH_BRANCH" = "1" ]; then
  git fetch origin "$BRANCH"
  if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git switch "$BRANCH"
  else
    git switch -c "$BRANCH" --track "origin/$BRANCH"
  fi
  git pull --ff-only
fi

echo "==== validate"
bash experiments/validate_lrc_patch_remote.sh

echo "==== current lanes"
bash experiments/poll_qwen4b_blank_smollm_chain_lanes.sh || true

gpu_free() {
  local uuid="$1"
  local used
  used="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid" || true)"
  [ "$used" = "0" ]
}

tag_running_or_done() {
  local tag="$1"
  local pid_file="$RUN_ROOT/logs/$tag.pid"
  local summary="$RUN_ROOT/$tag/reports/chain_anchor_summary.json"
  [ -f "$summary" ] && return 0
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    kill -0 "$pid" 2>/dev/null && return 0
  fi
  return 1
}

contains_lane() {
  local lane="$1"
  case ",$START_LANES," in
    *",$lane,"*) return 0 ;;
    *) return 1 ;;
  esac
}

launch_if_possible() {
  local lane="$1"
  local tag="$2"
  local gpu="$3"
  local script="$4"
  if ! contains_lane "$lane"; then
    echo "skip $lane: not in START_LANES=$START_LANES"
    return 0
  fi
  if tag_running_or_done "$tag"; then
    echo "skip $lane: existing pid or summary for $tag"
    return 0
  fi
  if ! gpu_free "$gpu"; then
    echo "skip $lane: gpu busy $gpu"
    return 0
  fi
  echo "launch $lane on $gpu"
  GPU_UUID="$gpu" TAG="$tag" bash "$script"
}

launch_if_possible \
  true_lrc \
  qwen4b_blank_smollm_chain_mlpout_true_lrc_rank16_blockio_alm_gpu4_20260629 \
  "$GPU4_UUID" \
  experiments/launch_qwen4b_blank_smollm_true_lrc_chain.sh

launch_if_possible \
  oldbest \
  qwen4b_blank_smollm_chain_oldbest_noactfit_cka_alm_gpu0_20260629 \
  "$GPU0_UUID" \
  experiments/launch_qwen4b_blank_smollm_oldbest_chain.sh

launch_if_possible \
  lfm_blank \
  lfm230m_blank_smollm_chain_true_lrc_rank16_gpu6_20260629 \
  "$GPU6_UUID" \
  experiments/launch_lfm230m_blank_smollm_chain.sh

echo "==== after launch"
bash experiments/poll_qwen4b_blank_smollm_chain_lanes.sh || true
