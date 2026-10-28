set -euo pipefail

root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
printf 'processes\n'
pgrep -af 'run_lfm2_exact4x_moving_gauge_v5_r12_live.py' || true
printf 'locks_and_incomplete\n'
find "$root" -maxdepth 8 \( -name '*.lock' -o -name '*incomplete*' -o -name '.incomplete' \) -printf '%y %p\n' 2>/dev/null | sort || true
printf 'checkpoint_tree\n'
find "$root/checkpoints" -maxdepth 5 -printf '%y %s %p\n' 2>/dev/null | sort || true
