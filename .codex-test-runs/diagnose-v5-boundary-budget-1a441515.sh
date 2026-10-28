set -euo pipefail

root=/home/cereal/model-surgery-lab/deployments/neural-forge-1a441515
cd "$root"
export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd
export PYTHONPATH="$root/src:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
from pathlib import Path

from experiments import run_lfm2_exact4x_moving_gauge_v5_live as live
from experiments import run_lfm2_exact4x_moving_gauge_v5_r12_live as launch
from neural_forge import hf_lfm2_moving_gauge_boundary_phase_cache as boundary
from neural_forge.hf_lfm2_exact4x_causal_diagnostic import (
    authenticate_v3_first_divergence_ledger,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (
    authenticate_lfm2_exact4x_first_divergence_request,
)


class DiagnosticComplete(RuntimeError):
    pass


def diagnose(_cache, request, *, authority, checkpoint_root, source_provider):
    del authority, checkpoint_root
    sources = source_provider()
    basis = boundary._authenticate_balanced_basis_budget_authority(
        sources.basis_phase,
        sources.basis_phase_authority,
        sources.counted_child_state_plan,
        sources.basis_phase_cache,
    )
    identity = basis["identity"]
    v3_request = authenticate_lfm2_exact4x_first_divergence_request(
        boundary._deep_thaw(sources.v3_first_divergence_request)
    )
    v3_ledger = authenticate_v3_first_divergence_ledger(
        boundary._deep_thaw(sources.v3_first_divergence_ledger),
        expected_request=v3_request,
    )
    provenance = request.provenance_sha256s
    budget = v3_request.get("budget_observation")
    checks = {
        "basis.phase_key": (identity.get("phase_key_sha256"), provenance["basis_phase_key"]),
        "basis.completion": (identity.get("completion_sha256"), provenance["basis_completion"]),
        "basis.map": (identity.get("basis_map_sha256"), provenance["basis_map"]),
        "basis.source_width": (identity.get("source_width"), request.parent_width),
        "basis.target_width": (identity.get("target_width"), request.child_width),
        "basis.parent_count": (basis["parent_parameter_count"], request.parent_parameter_count),
        "basis.child_count": (basis["child_parameter_count"], request.child_parameter_count),
        "basis.route": (basis["route_artifact_sha256"], provenance["route_package"]),
        "v3.request": (v3_request.get("request_sha256"), provenance["v3_request"]),
        "v3.ledger": (v3_ledger.get("ledger_sha256"), provenance["v3_ledger"]),
        "v3.batch_count": (v3_ledger.get("batch_count"), request.expected_batch_count),
        "v3.child_package": (v3_request.get("child_package_sha256"), provenance["child_package"]),
        "v3.child_identity": (v3_request.get("child_state_sha256"), provenance["child_identity"]),
        "v3.route_package": (v3_request.get("route_package_sha256"), provenance["route_package"]),
        "v3.route_runtime": (v3_request.get("route_runtime_sha256"), provenance["route_runtime"]),
        "v3.portable_runtime": (v3_request.get("portable_runtime_receipt_sha256"), provenance["portable_runtime_receipt"]),
        "v3.parent_cache_key": (v3_request.get("trace_cache_key"), request.parent_cache_key),
        "v3.parent_session": (v3_request.get("trace_selective_session_sha256"), provenance["parent_selective_session"]),
        "v3.basis_key": (v3_request.get("basis_phase_key_sha256"), provenance["basis_phase_key"]),
        "v3.basis_completion": (v3_request.get("basis_completion_sha256"), provenance["basis_completion"]),
        "v3.basis_map": (v3_request.get("basis_map_sha256"), provenance["basis_map"]),
        "budget.parent": (budget.get("parent_parameter_elements"), request.parent_parameter_count),
        "budget.dense": (budget.get("physical_dense_parameter_elements"), basis["dense_parameter_count"]),
        "budget.routes": (budget.get("continuous_route_parameter_elements"), sources.counted_child_state_plan.continuous_parameter_elements),
        "budget.counted_total": (budget.get("counted_total_parameter_elements"), request.child_parameter_count),
        "budget.target_total": (budget.get("target_total_parameter_elements"), request.child_parameter_count),
        "budget.complete_child": (budget.get("one_complete_child_observed"), True),
        "budget.partial": (budget.get("partial_component_variant"), False),
        "budget.pruning": (budget.get("pruning_credit"), False),
        "budget.quantization": (budget.get("quantization_credit"), False),
        "budget.sharing": (budget.get("parameter_sharing_credit"), False),
    }
    mismatches = 0
    for name, (actual, expected) in checks.items():
        equal = actual == expected and type(actual) is type(expected)
        print(name, "OK" if equal else "MISMATCH", repr(actual), repr(expected))
        mismatches += int(not equal)
    print("mismatch_count", mismatches)
    raise DiagnosticComplete


live.load_or_extract_lfm2_moving_gauge_boundary_phase = diagnose
manifest = launch.read_r12_v5_launch_manifest(
    Path("experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json")
)
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
    raise AssertionError("Boundary budget diagnostic did not stop")
PY
