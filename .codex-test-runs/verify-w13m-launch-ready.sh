#!/usr/bin/env bash
set -euo pipefail

old_pid=3551402
old_capsules=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0026-exact4x-h816-mgv3-r12-w13l-cold/restart-capsules
new_run=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0027-exact4x-h816-mgv3-r12-w13m-resume-d296b602

if kill -0 "$old_pid" 2>/dev/null; then
  echo "Old W13L PID is still live" >&2
  exit 93
fi
if [ -e "$new_run" ]; then
  echo "W13M run root unexpectedly exists" >&2
  exit 94
fi

mapfile -t capsules < <(find "$old_capsules" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
printf 'OLD_PID_DEAD=%s\n' "$old_pid"
printf 'NEW_RUN_ABSENT=%s\n' "$new_run"
printf 'CAPSULES=%s\n' "${capsules[*]}"
printf 'GPU_PIDS=' 
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader || true
