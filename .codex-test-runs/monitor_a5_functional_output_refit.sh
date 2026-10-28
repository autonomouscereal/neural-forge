set -euo pipefail

run_dir=/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a5-fixed-a4-support
printf 'pid=%s\n' "$(cat "$run_dir/job.pid" 2>/dev/null || printf missing)"
printf 'status=%s\n' "$(cat "$run_dir/job.status" 2>/dev/null || printf running)"
printf 'gpu_processes:\n'
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
printf 'latest_log:\n'
tail -n 25 "$run_dir/job.log" 2>/dev/null || true
