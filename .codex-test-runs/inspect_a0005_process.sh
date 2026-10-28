set -euo pipefail

PID=$(cat /home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_heterogeneous_20260729/neural-forge-07fafdb0-3a9bc301/a0005-aggressive-h784-exact57420144/run.pid)
echo "PID=$PID"
readlink -f "/proc/$PID/exe" || true
tr '\0' ' ' < "/proc/$PID/cmdline" || true
echo
tr '\0' '\n' < "/proc/$PID/environ" \
  | grep -E '^(PYTHONPATH|VIRTUAL_ENV|CUDA_VISIBLE_DEVICES|PATH)=' || true
ps -o pid,ppid,stat,etime,cmd --forest -p "$PID" --ppid "$PID" || true
