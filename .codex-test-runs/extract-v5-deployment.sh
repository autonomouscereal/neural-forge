set -euo pipefail
root=/home/cereal/model-surgery-lab/deployments/neural-forge-f7bbf16d
cd "$root"
printf '%s  %s\n' '1440b188d792d68a339c4891fb17d3627f2330be95275bc621ce12247de1cab0' source.zip | sha256sum --check --strict
unzip -q source.zip -d .
test -f experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py
test -f experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json
test -f src/neural_forge/hf_tokenizer_bridge.py
PYTHONPATH="$root/src:$root" /home/cereal/model-surgery-lab/venv/bin/python -m py_compile \
    experiments/run_lfm2_exact4x_moving_gauge_v5.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_live.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py
PYTHONPATH="$root/src:$root" /home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
from pathlib import Path
from experiments.run_lfm2_exact4x_moving_gauge_v5_r12_live import read_r12_v5_launch_manifest

manifest = read_r12_v5_launch_manifest(
    Path("experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json")
)
print("manifest", manifest["launch_manifest_sha256"])
print("bridge_root", manifest["tokenizer_bridge_root"])
print("receipt", manifest["receipt_path"])
PY
