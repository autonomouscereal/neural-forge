set -euo pipefail
runtime=/home/cereal/model-surgery-lab/runtimes/neural-forge-9100d17b
source_run=/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/d6f5e1c8/structural-first-effectful-a0005
echo RUNNER_PREDICATE
sed -n '742,779p' "$runtime/experiments/run_lfm2_exact4x_causal_head_frontier.py"
echo SOURCE_FILES
find "$source_run" -maxdepth 2 -type f -printf '%s %p\n' | sort
echo RECEIPT_HASH_FIELDS
python3 - "$source_run" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "structural-interface-install.json",
    "semantic-layer-stack.json",
    "deferred-final-w2-install.json",
    "structural-body-receipt.json",
):
    path = root / name
    print(f"FILE {name} exists={path.is_file()}")
    if not path.is_file():
        continue
    value = json.loads(path.read_text())
    for key in (
        "model_state_before_sha256",
        "model_state_after_sha256",
        "child_state_before_sha256",
        "child_state_after_sha256",
        "body_model_state_sha256",
        "body_lfm2_identity_sha256",
        "causal_body_inventory_sha256",
        "exact_target_unique_parameters",
        "target_unique_parameters",
        "installed_input_table_sha256",
        "installed_output_baseline_sha256",
        "result_sha256",
        "receipt_sha256",
    ):
        if key in value:
            print(f"  {key}={value[key]}")
PY
