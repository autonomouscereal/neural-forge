set -u
pid=2909536
run=/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_causal_head_frontier_20260724/9100d17b/six-head-control-a0001
log=${run}.log
echo PROCESS
ps -p "$pid" -o pid=,stat=,etime=,pcpu=,pmem=,rss=,vsz=,args= || true
echo LOG_META
if [ -f "$log" ]; then
  stat -c 'size=%s mtime=%y path=%n' "$log"
  echo LOG_TAIL
  tail -n 100 "$log"
else
  echo "missing $log"
fi
echo ARTIFACTS
if [ -d "$run" ]; then
  find "$run" -maxdepth 4 -type f -printf '%T@ %s %p\n' | sort -n | tail -n 100
else
  echo "missing $run"
fi
