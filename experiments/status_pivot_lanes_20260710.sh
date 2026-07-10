#!/usr/bin/env bash
set -euo pipefail

show_run() {
  local root=$1
  local tag=$2
  local pid_file="$root/$tag.pid"
  local log_file="$root/$tag.log"
  local run_dir="$root/$tag"
  echo "=== $tag ==="
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(tr -dc '0-9' < "$pid_file")
    ps -p "$pid" -o pid=,etime=,pcpu=,nlwp=,rss=,stat=,cmd= || true
  else
    echo "pid_file_missing"
  fi
  if [[ -f "$log_file" ]]; then
    grep 'neural_forge_progress' "$log_file" | tail -n 20 || true
    tail -n 10 "$log_file"
  else
    echo "log_missing"
  fi
  if [[ -d "$run_dir" ]]; then
    find "$run_dir" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
  fi
}

show_run \
  /mnt/ai_data/neural-forge/progressive_closed_form_20260710 \
  qwen3_4b_to_blank_qwen1p13b_stage1_gpu6_r2_20260710
show_run \
  /mnt/ai_data/neural-forge/cross_family_blank_20260710 \
  qwen3_4b_to_blank_qwen134m_headgqa_producttrust_gpu0_r1_20260710
show_run \
  /mnt/ai_data/neural-forge/cross_family_blank_20260710 \
  qwen3_4b_to_blank_llama131m_headgqa_producttrust_gpu4_r1_20260710

echo "=== NEO1.5 download ==="
DOWNLOAD_ROOT=/mnt/ai_data/neural-forge/native_parent_cache_20260710
if [[ -f "$DOWNLOAD_ROOT/neo1_5_2b_download.pid" ]]; then
  DOWNLOAD_PID=$(tr -dc '0-9' < "$DOWNLOAD_ROOT/neo1_5_2b_download.pid")
  ps -p "$DOWNLOAD_PID" -o pid=,etime=,pcpu=,rss=,stat=,cmd= || true
fi
if [[ -f "$DOWNLOAD_ROOT/neo1_5_2b_download.log" ]]; then
  tail -n 15 "$DOWNLOAD_ROOT/neo1_5_2b_download.log"
fi

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader
echo "SSH_SYN_RECV $(ss -Htan state syn-recv sport = :22 | wc -l)"
