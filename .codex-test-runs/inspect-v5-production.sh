set -euo pipefail
cd /home/cereal/model-surgery-lab
echo HOST
hostname
echo GIT
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
else
    echo unversioned-remote-deployment
fi
echo GPU
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo PROCESSES
pgrep -af "moving_gauge_v5|exact4x.*v5" || true
echo ARTIFACTS
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import json
from pathlib import Path

roots = {
    "child": Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12/published-child/child"),
    "v3": Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_first_divergence_20260804/a0023-exact4x-h816-mgv3-r12-layer0-v3"),
    "parent_cache": Path("/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"),
    "basis": Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1"),
    "bridge": Path("/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2"),
    "target": Path("/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/target_tokenized_calibration.json"),
}
for key, path in roots.items():
    print(key, path.exists(), path)
print("child files", sorted(path.name for path in roots["child"].iterdir()))
print("v3 files", sorted(path.name for path in roots["v3"].iterdir()))
for label, path in (
    ("child_receipt", roots["child"] / "neural_forge_child_package_receipt.json"),
    ("child_bridge_ref", roots["child"] / "neural_forge_tokenizer_bridge_reference.json"),
    ("v3_request", roots["v3"] / "request.json"),
    ("v3_ledger", roots["v3"] / "first-divergence-ledger.json"),
):
    if path.exists():
        value = json.loads(path.read_text())
        print(label, {key: value.get(key) for key in value if "sha256" in key or key in {"schema", "batch_count", "runtime_device", "runtime_dtype", "trace_cache_key"}})
for path in roots["basis"].rglob("*.json"):
    try:
        value = json.loads(path.read_text())
    except Exception:
        continue
    if isinstance(value, dict) and ("source_anchor" in value or "anchor_key_id" in value or "key_id" in value):
        print("basis_authority_file", path)
        print({key: value.get(key) for key in value if key in {"source_anchor", "anchor_key_id", "key_id", "phase_key", "completion_sha256", "map_sha256", "schema"}})
PY
