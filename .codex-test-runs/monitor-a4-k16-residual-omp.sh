set -u

RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a4-k16-residual-omp
if [ ! -f "$RUN/driver.pid" ]; then
  echo "state=missing-pid"
  exit 0
fi

read -r pid < "$RUN/driver.pid"
if kill -0 "$pid" 2>/dev/null; then
  state=running
else
  state=stopped
fi
echo "state=$state pid=$pid"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader -i 4
wc -c "$RUN/driver.log"
tail -n 30 "$RUN/driver.log"
if [ -d "$RUN/events" ]; then
  find "$RUN/events" -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -n 5
fi
