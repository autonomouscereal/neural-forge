set -euo pipefail

umask 077
upload_root=/home/cereal/model-surgery-lab/upload_staging
archive="$upload_root/neural-forge-a884b2d1.zip"
review="$upload_root/review-v5-c8cfa511.sh"
prompts="$upload_root/lfm2_semantic_whole_response_prompts_v1.a884b2d1.raw.json"
destination=/home/cereal/model-surgery-lab/deployments/neural-forge-a884b2d1
staging="$destination.staging.$$"
python=/home/cereal/model-surgery-lab/venv/bin/python

test ! -e "$destination"
test ! -e "$staging"
printf '%s  %s\n' \
    '6b985ff160054bafb135430dcfe80253b9f1fcdf6b394a627c42a18999a3b311' \
    "$archive" | sha256sum --check --strict
printf '%s  %s\n' \
    '30018c3128122edabefe1431369bc1018c1953412d9b7f9730b7cdbff5c559c4' \
    "$review" | sha256sum --check --strict
printf '%s  %s\n' \
    '0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b' \
    "$prompts" | sha256sum --check --strict

install -d -m 700 "$staging"
mv "$archive" "$staging/source.zip"
unzip -q "$staging/source.zip" -d "$staging"
install -m 600 \
    "$prompts" \
    "$staging/research/lfm2_semantic_whole_response_prompts_v1.json"
test -f "$staging/src/neural_forge/lfm2_carrier_review_cli.py"
test -f "$staging/src/neural_forge/lfm2_exact4x_moving_gauge_v5_candidate_materializer.py"
test -f "$staging/experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py"
test -f "$staging/experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json"
test -f "$staging/research/lfm2_semantic_whole_response_prompts_v1.json"
printf '%s  %s\n' \
    '0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b' \
    "$staging/research/lfm2_semantic_whole_response_prompts_v1.json" | \
    sha256sum --check --strict

cd "$staging"
export PYTHONPATH="$staging/src:$staging"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$python" -m py_compile \
    src/neural_forge/lfm2_carrier_review_cli.py \
    src/neural_forge/lfm2_exact4x_moving_gauge_v5_candidate_materializer.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5.py \
    experiments/run_lfm2_exact4x_moving_gauge_v5_r12_live.py
"$python" - <<'PY'
from pathlib import Path

from experiments.run_lfm2_exact4x_moving_gauge_v5_r12_live import (
    read_r12_v5_launch_manifest,
)
from neural_forge.lfm2_carrier_review_cli import (
    Lfm2CarrierReviewRecoveryConfig,
)

field = Lfm2CarrierReviewRecoveryConfig.__dataclass_fields__[
    "maximum_prompt_tokens"
]
assert field.default == 8_192
manifest = read_r12_v5_launch_manifest(
    Path("experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json")
)
assert manifest["launch_manifest_sha256"] == (
    "9ce176b9650296ad0c99658ae46ccb3118bdb9800c91d4d820b1d316989dbb58"
)
assert manifest["receipt_path"].endswith("r12-v5-launch-receipt-v2.json")
PY

printf '%s\n' 'a884b2d1' > SOURCE_COMMIT
chmod -R u=rwX,go= "$staging"
mv "$staging" "$destination"
chmod 700 "$review"
grep -Fq \
    'deployment_root=/home/cereal/model-surgery-lab/deployments/neural-forge-a884b2d1' \
    "$review"
printf 'deployment_root=%s\nreview_launcher=%s\n' "$destination" "$review"
sha256sum "$destination/source.zip" "$review"
