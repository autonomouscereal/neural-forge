set -u
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_producer_aware_v2_4x_20260723/d622207
printf 'UTC '
date -u +'%Y-%m-%dT%H:%M:%SZ'
printf '\nPROCESSES\n'
ps -p 559294,559295 -o pid=,stat=,etime=,pcpu=,pmem=,rss=,vsz=,time= || true
printf '\nGPU\n'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null | grep -E '^(559294|559295),' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
printf '\nRECENT_FILES\n'
if [ -d "$BASE" ]; then
  find "$BASE" -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -40
else
  echo "MISSING_BASE $BASE"
fi
printf '\nLOG_TAILS\n'
if [ -d "$BASE" ]; then
  while IFS= read -r f; do
    printf '\n--- %s ---\n' "$f"
    tail -n 80 "$f" 2>/dev/null || true
  done < <(find "$BASE" -type f \( -name '*.log' -o -name 'stdout*' -o -name 'stderr*' \) -print 2>/dev/null | sort)
fi
