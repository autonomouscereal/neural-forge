set -euo pipefail

run_dir=/home/cereal/model-surgery-lab/artifacts/neural_forge_carrier_replant_20260730/a8-natural-language-2025-2026-corpus356-cout-lambda0001
printf 'pid=%s\n' "$(cat "$run_dir/job.pid" 2>/dev/null || printf missing)"
printf 'status=%s\n' "$(cat "$run_dir/job.status" 2>/dev/null || printf running)"
printf 'completed_prompts=%s\n' "$(find "$run_dir/prompt-results" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
printf 'carrier_artifact=%s\n' "$(test -f "$run_dir/carrier-artifact.json" && printf complete || printf missing)"
printf 'gpu_processes:\n'
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
printf 'latest_log:\n'
tail -n 40 "$run_dir/job.log" 2>/dev/null || true
