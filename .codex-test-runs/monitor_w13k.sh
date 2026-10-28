set -u
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0025-exact4x-h816-mgv3-r12-w13k-cold
PID=3262674
echo "REMOTE_TIME $(date -Ins)"
echo "PROCESS"
ps -p "$PID" -o pid=,ppid=,stat=,etime=,etimes=,%cpu=,%mem=,rss=,vsz=,lstart=,cmd= || true
echo "PROC_IO"
if [ -r "/proc/$PID/io" ]; then
  sed -n '1,20p' "/proc/$PID/io"
fi
echo "GPU_SUMMARY"
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true
echo "GPU_PROCESSES"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null | grep -F "$PID" || true
echo "RUN_ROOT"
if [ -d "$RUN" ]; then
  find "$RUN" -maxdepth 4 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %s %p\n' | sort | tail -80
else
  echo MISSING
fi
echo "LOG_TAILS"
if [ -d "$RUN" ]; then
  while IFS= read -r f; do
    case "$f" in
      *.log|*.out|*.err|*.jsonl)
        echo "--- $f"
        tail -40 "$f" 2>/dev/null || true
        ;;
    esac
  done < <(find "$RUN" -maxdepth 3 -type f | sort)
fi
