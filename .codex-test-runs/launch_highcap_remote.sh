set -euo pipefail

root=/home/cereal/model-surgery-lab/neural-forge-highcap-eb0a5460
ROOT="$root" \
RUN_TAG=a9-a11-highcap-3090-r2 \
bash "$root/experiments/launch_lfm2_a9_a11_high_cap_reuse_20260730.sh"
