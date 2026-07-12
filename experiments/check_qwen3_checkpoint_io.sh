#!/usr/bin/env bash
set -u

root=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710
tags=(
  qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710
  qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710
  qwen3_4b_codebook_pn34_pa36_gpu6_r1_20260710
)

date --iso-8601=seconds
for tag in "${tags[@]}"; do
  echo "=== $tag ==="
  pid=$(tr -d '[:space:]' < "$root/$tag.pid")
  ps -p "$pid" -o pid=,stat=,etime=,rss=,vsz=,pcpu=,pmem= || true
  if [ -r "/proc/$pid/wchan" ]; then
    printf 'wchan='
    cat "/proc/$pid/wchan"
  fi
  if [ -r "/proc/$pid/io" ]; then
    grep -E '^(rchar|wchar|read_bytes|write_bytes|cancelled_write_bytes):' "/proc/$pid/io"
  fi
  find "$root/$tag/compact_child" -maxdepth 1 -type f \
    -printf '%f %s bytes %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort || true
done

echo '=== memory ==='
free -h
echo '=== filesystem ==='
df -h /mnt/ai_data
df -i /mnt/ai_data
echo '=== vmstat ==='
vmstat 1 3
