set -euo pipefail

run=/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a10-k32-cout-lambda0001
pid=$(cat "$run/job.pid" 2>/dev/null || printf missing)
status=$(cat "$run/job.status" 2>/dev/null || printf running)
printf 'pid=%s status=%s\n' "$pid" "$status"
if [ "$pid" != missing ] && kill -0 "$pid" 2>/dev/null; then
  ps -p "$pid" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=
fi
printf 'artifacts:\n'
find "$run" -maxdepth 3 -type f -printf '%P %s bytes\n' 2>/dev/null | sort
printf 'latest-log:\n'
tail -n 20 "$run/job.log" 2>/dev/null || true
printf 'gpu-processes:\n'
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
