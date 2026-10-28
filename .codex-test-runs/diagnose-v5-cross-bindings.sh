set -euo pipefail
root=/home/cereal/model-surgery-lab/deployments/neural-forge-c546030f
cd "$root"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$root/src:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
from pathlib import Path

from experiments import run_lfm2_exact4x_moving_gauge_v5_r12_live as launch


class DiagnosticComplete(RuntimeError):
    pass


def diagnose(**values):
    lineage = values["lineage"]
    provenance = values["provenance"]
    package = values["package"]
    counted_plan = values["counted_plan"]
    request = values["v3_request"]
    ledger = values["v3_ledger"]
    basis = values["basis_phase"]
    bridge = values["bridge"]
    adapter = values["adapter"]
    checks = {
        "lineage.child_package": (lineage["r12_child_package"], provenance["child_package"]),
        "lineage.child_identity": (lineage["r12_child_identity"], provenance["child_identity"]),
        "lineage.child_state_dict": (lineage["r12_state_dict"], provenance["child_state_dict"]),
        "lineage.v3_request": (lineage["v3_request"], provenance["v3_request"]),
        "lineage.v3_ledger": (lineage["v3_ledger"], provenance["v3_ledger"]),
        "lineage.parent_completion": (lineage["parent_trace_cache_completion"], provenance["parent_completion"]),
        "lineage.child_completion": (lineage["child_trace_cache_completion"], provenance["child_completion"]),
        "lineage.basis_key": (lineage["basis_phase_key"], provenance["basis_phase_key"]),
        "lineage.basis_completion": (lineage["basis_phase_completion"], provenance["basis_completion"]),
        "lineage.bridge_artifact": (lineage["tokenizer_bridge"], provenance["bridge_artifact"]),
        "package.package": (package.package_sha256, lineage["r12_child_package"]),
        "package.bridge": (package.receipt.get("source_bridge_artifact_sha256"), provenance["bridge_artifact"]),
        "counted_plan": (counted_plan.fingerprint, lineage["counted_child_state_plan"]),
        "request.request": (request.get("request_sha256"), lineage["v3_request"]),
        "ledger.ledger": (ledger.get("ledger_sha256"), lineage["v3_ledger"]),
        "ledger.batch_count": (ledger.get("batch_count"), 256),
        "request.package": (request.get("child_package_sha256"), package.package_sha256),
        "request.child_identity": (request.get("child_state_sha256"), provenance["child_identity"]),
        "request.route_package": (request.get("route_package_sha256"), provenance["route_package"]),
        "request.route_runtime": (request.get("route_runtime_sha256"), provenance["route_runtime"]),
        "request.portable_runtime": (request.get("portable_runtime_receipt_sha256"), provenance["portable_runtime_receipt"]),
        "request.basis_key": (request.get("basis_phase_key_sha256"), basis.phase_key.phase_key_sha256),
        "request.basis_completion": (request.get("basis_completion_sha256"), basis.completion_sha256),
        "request.basis_map": (request.get("basis_map_sha256"), basis.map_sha256),
        "basis.key": (basis.phase_key.phase_key_sha256, lineage["basis_phase_key"]),
        "basis.completion": (basis.completion_sha256, lineage["basis_phase_completion"]),
        "basis.map": (basis.map_sha256, provenance["basis_map"]),
        "basis.source_anchor": (basis.source_anchor["hmac_sha256"], lineage["basis_source_anchor"]),
        "bridge.artifact": (bridge.artifact_sha256, provenance["bridge_artifact"]),
        "bridge.profile": (bridge.profile_fingerprint, provenance["bridge_profile"]),
        "adapter.evidence": (adapter.evidence_sha256, provenance["bridge_trace_adapter"]),
        "adapter.bridge": (adapter.bridge_artifact_sha256, bridge.artifact_sha256),
        "adapter.profile": (adapter.transport_profile_fingerprint, bridge.profile_fingerprint),
    }
    mismatches = 0
    for name, (actual, expected) in checks.items():
        equal = actual == expected
        print(name, "OK" if equal else "MISMATCH", repr(actual), repr(expected))
        mismatches += int(not equal)
    print("mismatch_count", mismatches)
    raise DiagnosticComplete


manifest = launch.read_r12_v5_launch_manifest(
    Path("experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json")
)
launch._assert_lineage_cross_bindings = diagnose
try:
    launch.execute_r12_v5_launch(
        manifest,
        authority_key_path=Path(
            "/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key"
        ),
    )
except DiagnosticComplete:
    pass
else:
    raise AssertionError("Cross-binding diagnostic did not stop")
PY
