#!/usr/bin/env bash
set -u

root=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710

for tag in \
  qwen3_4b_codebook_pn36_pa36_gpu0_r1_20260710 \
  qwen3_4b_codebook_pn36_pa36_gpu0_r2_20260710 \
  qwen3_4b_codebook_pn36_pa36_gpu0_r3_20260710 \
  qwen3_4b_codebook_pn34_pa34_gpu6_r1_20260710 \
  qwen3_4b_codebook_pn32_pa32_gpu4_r1_20260710 \
  qwen3_4b_codebook_pn28_pa28_gpu0_r1_20260710 \
  qwen3_4b_codebook_pn35_pa35_gpu4_r1_20260710 \
  qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710 \
  qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710 \
  qwen3_4b_codebook_pn34_pa36_gpu6_r1_20260710
do
  echo "=== ${tag} ==="
  pid_file="${root}/${tag}.pid"
  log_file="${root}/${tag}.log"
  run_dir="${root}/${tag}"

  if [ -f "$pid_file" ]; then
    pid=$(tr -d '[:space:]' < "$pid_file")
    echo "pid=${pid}"
    ps -p "$pid" -o pid=,stat=,etime=,pcpu=,pmem=,cmd= || true
  else
    echo "pid_file=missing"
  fi

  if [ -f "$log_file" ]; then
    grep -a "neural_forge_progress" "$log_file" | tail -n 8 || true
    tail -n 8 "$log_file" || true
  fi

  for artifact in \
    calibration_report.json \
    alignment_report.json \
    runtime_verification.json \
    compact_checkpoint_report.json \
    candidate_eval.json \
    parent_candidate_comparison.json \
    run_report.json
  do
    if [ -f "${run_dir}/${artifact}" ]; then
      echo "FOUND ${run_dir}/${artifact}"
      stat -c '%s bytes %y' "${run_dir}/${artifact}"
    fi
  done
done

echo "=== GPU ==="
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
