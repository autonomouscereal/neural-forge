set -euo pipefail

root=/home/cereal/model-surgery-lab/deployments/neural-forge-c8cfa511
cd "$root"
printf '%s  %s\n' \
    'c92d6d92da5f9e205871b50f7a017eaac1c045eb10f364a2409750c8fb4dea0f' \
    source.zip | sha256sum --check --strict
unzip -q source.zip -d .
test -f experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py
test -f experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json
test -f src/neural_forge/lfm2_exact4x_moving_gauge_v5_candidate_materializer.py
test -f src/neural_forge/hf_tokenizer_bridge.py
test -f research/lfm2_semantic_whole_response_prompts_v1.json
test -f reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md
export PYTHONPATH="$root/src:$root"
python=/home/cereal/model-surgery-lab/venv/bin/python
"$python" -m py_compile \
    experiments/run_lfm2_exact4x_moving_gauge_v5.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_live.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py \
    src/neural_forge/lfm2_exact4x_moving_gauge_v5_candidate_materializer.py
"$python" - <<'PY'
from pathlib import Path
from experiments.run_lfm2_exact4x_moving_gauge_v5_r12_live import (
    read_r12_v5_launch_manifest,
)

manifest = read_r12_v5_launch_manifest(
    Path("experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json")
)
assert manifest["launch_manifest_sha256"] == (
    "9ce176b9650296ad0c99658ae46ccb3118bdb9800c91d4d820b1d316989dbb58"
)
assert manifest["receipt_path"].endswith("r12-v5-launch-receipt-v2.json")
print("manifest", manifest["launch_manifest_sha256"])
print("bridge_root", manifest["tokenizer_bridge_root"])
print("receipt", manifest["receipt_path"])
PY
