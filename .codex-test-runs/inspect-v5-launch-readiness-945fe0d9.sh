set -euo pipefail

root=/home/cereal/model-surgery-lab/deployments/neural-forge-945fe0d9
artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
receipt="$artifact_root/candidate-packs/r12-v5-launch-receipt.json"

test -f "$root/experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py"
if [ -e "$receipt" ]; then
    printf 'receipt=PRESENT\n'
else
    printf 'receipt=ABSENT\n'
fi
printf 'artifact_directories\n'
find "$artifact_root" -maxdepth 2 -mindepth 1 -type d -printf '%p\n' 2>/dev/null | sort || true
printf 'gpu_inventory\n'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
printf 'compute_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
printf 'matching_processes\n'
pgrep -af 'run_lfm2_exact4x_moving_gauge_v5_r12_live|neural_forge_exact4x_moving_gauge_v5_20260804' || true
printf 'disk_free\n'
df -h /home/cereal/model-surgery-lab/artifacts
