set -euo pipefail

run_dir=/home/cereal/model-surgery-lab/artifacts/neural_forge_token_bridge_isolation_20260730/a4-k16
mkdir -p "$run_dir"
printf 'stage=start\nutc=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" | tee "$run_dir/launch.log"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4 timeout 600s \
  /home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python \
  /home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2/.codex-test-runs/a4_first_token_isolation.py \
  2>&1 | tee "$run_dir/result.jsonl"
printf 'stage=complete\nutc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$run_dir/launch.log"
