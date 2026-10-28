set -euo pipefail

run_dir=/home/cereal/model-surgery-lab/artifacts/neural_forge_carrier_replant_20260730/a7-repository-corpus356-cout-lambda0001
mkdir -p "$run_dir"
rm -f "$run_dir/job.status"
(
  set +e
  printf '{"stage":"job-start","utc":"%s","host":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)"
  NF_CARRIER_RUN_DIRECTORY="$run_dir" \
  NF_CARRIER_SELECTION_MODE=repository-corpus-v1 \
  NF_CARRIER_BUDGET=356 \
  NF_RUNTIME_DEVICE=cuda:0 \
  NF_RUNTIME_DTYPE=bfloat16 \
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4 timeout 3600s \
    /home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python \
    /home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/.codex-test-runs/a6_prompt_carrier_replant.py
  rc=$?
  printf '{"stage":"job-stop","utc":"%s","exit_code":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc"
  printf '%d\n' "$rc" > "$run_dir/job.status"
) > "$run_dir/job.log" 2>&1 &
pid=$!
printf '%d\n' "$pid" > "$run_dir/job.pid"
printf 'pid=%d\nlog=%s\n' "$pid" "$run_dir/job.log"
