#!/usr/bin/env bash
set -u

root=/mnt/ai_data/neural-forge/depth_survival_20260710

for spec in \
  "1634333:qwen3_4b_depth34_realmedoid_dual_gpu6_r1_20260710" \
  "1812477:qwen3_4b_depth32_realmedoid_dual_gpu0_r4_c128_20260710" \
  "1336854:qwen3_4b_depth28_realmedoid_dual_gpu4_r2_20260710"
do
  pid="${spec%%:*}"
  tag="${spec#*:}"
  echo "=== ${tag} pid=${pid} ==="
  ps -p "${pid}" -o pid=,stat=,etime=,pcpu=,pmem=,cmd= || true

  if [ -f "${root}/${tag}.log" ]; then
    grep -a "neural_forge_progress" "${root}/${tag}.log" | tail -n 4 || true
    tail -n 5 "${root}/${tag}.log" || true
  fi

  for artifact in run_report.json parent_candidate_comparison.json; do
    if [ -f "${root}/${tag}/${artifact}" ]; then
      echo "FOUND ${root}/${tag}/${artifact}"
      stat -c '%s bytes %y' "${root}/${tag}/${artifact}"
    fi
  done
done

echo "=== GPU ==="
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
