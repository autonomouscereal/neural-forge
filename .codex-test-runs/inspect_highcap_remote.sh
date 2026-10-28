set -euo pipefail

launch=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730/_launch/a9-a11-highcap-3090-r2
pid=$(cat "$launch/job.pid")
echo "PID=$pid"
if kill -0 "$pid" 2>/dev/null; then
  ps -p "$pid" -o pid,ppid,etime,%cpu,%mem,rss,vsz,stat,args
else
  echo PROCESS_NOT_RUNNING
fi
echo IDENTITY
cat "$launch/job.identity.json"
echo LOG
tail -80 "$launch/job.log"
echo GPU
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | grep 'GPU-2ce6a95c-7b1f-644a-3808-d781732a0582' || true
echo OUTPUTS
find /home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730 -maxdepth 4 -type f -printf '%p %s\n' | sort | head -120
