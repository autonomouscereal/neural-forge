#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/mnt/ai_data/neural-forge/cross_family_blank_20260710
for TAG in \
  qwen3_4b_to_blank_llama88m_structuralprior_gpu0_r3_20260710 \
  qwen3_4b_to_blank_llama130m_structuralprior_gpu4_r2_20260710
do
  PID_FILE="$RUN_ROOT/$TAG.pid"
  LOG_FILE="$RUN_ROOT/$TAG.log"
  RUN_DIR="$RUN_ROOT/$TAG"
  echo "=== $TAG process ==="
  if [[ -f "$PID_FILE" ]]; then
    PID=$(tr -dc '0-9' < "$PID_FILE")
    ps -p "$PID" -o pid=,etime=,stat=,rss=,cmd= || true
  else
    echo "pid_file_missing"
  fi
  echo "=== $TAG progress ==="
  if [[ -f "$LOG_FILE" ]]; then
    grep 'neural_forge_progress' "$LOG_FILE" | tail -n 30 || true
    echo "=== $TAG log_tail ==="
    tail -n 30 "$LOG_FILE"
  else
    echo "log_missing"
  fi
  echo "=== $TAG artifacts ==="
  if [[ -d "$RUN_DIR" ]]; then
    find "$RUN_DIR" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
  else
    echo "run_dir_missing"
  fi
done

echo "=== gpus ==="
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
echo "SSH_SYN_RECV $(ss -Htan state syn-recv sport = :22 | wc -l)"
