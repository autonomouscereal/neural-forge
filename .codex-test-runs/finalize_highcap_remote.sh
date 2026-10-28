set -euo pipefail

root=/home/cereal/model-surgery-lab/neural-forge-highcap-eb0a5460
base=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src:$root"

/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

from neural_forge.dynamic_circuit_ir import canonical_sha256

base = Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730")
zero_names = {
    "parent_model_load_calls",
    "parent_forward_calls",
    "parent_generate_calls",
    "parent_trace_recaptures",
    "fold_compilations",
    "support_discovery_calls",
    "coefficient_solve_calls",
    "training_steps",
    "optimizer_steps",
    "backward_calls",
    "pruned_parameters",
    "quantized_parameters",
}
for path in sorted(base.glob("*/high-cap-dossier.json")):
    dossier = json.loads(path.read_text())
    body = dict(dossier)
    claimed = body.pop("result_sha256")
    attempts = [
        prompt["authoritative_high_cap_attempt"]
        for prompt in dossier["prompt_results"]
    ]
    statuses = {}
    for attempt in attempts:
        status = attempt["termination_evidence"]["completion_status"]
        statuses[status] = statuses.get(status, 0) + 1
    audits_zero = all(
        int(dossier["operation_audit"][name]) == 0 for name in zero_names
    ) and all(
        all(int(attempt["operation_audit"][name]) == 0 for name in zero_names)
        for attempt in attempts
    )
    print(json.dumps({
        "lane": dossier["lane_id"],
        "result_sha256": claimed,
        "seal_valid": claimed == canonical_sha256(body),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_count": dossier["prompt_count"],
        "authoritative_completion_counts": statuses,
        "runtime_error_count": sum(a["runtime_error"] is not None for a in attempts),
        "all_target_roundtrips_exact": all(
            a["response"] is not None and a["response"]["target_roundtrip_exact"]
            for a in attempts
        ),
        "all_forbidden_operation_counters_zero": audits_zero,
        "network_access": dossier["operation_audit"]["network_access"],
        "physical_child_parameter_count": dossier["physical_child_parameter_count"],
        "routing_continuous_state_count": dossier["routing_receipt"].get(
            "continuous_state_count"
        ),
        "checkpoint_inventory_sha256": dossier[
            "checkpoint_artifact_inventory_sha256"
        ],
        "child_state_sha256": dossier["child_state_sha256"],
        "routing_contract_sha256": dossier["routing_contract_sha256"],
        "functional_output_authenticated": (
            dossier["functional_output_authentication"] is not None
        ),
        "safety_separate": dossier["safety_is_separate_from_folding_quality"],
        "cap_censored_not_failed": dossier[
            "generation_limit_hit_is_censored_not_failed"
        ],
        "automatic_acceptance_or_rejection_performed": dossier[
            "automatic_acceptance_or_rejection_performed"
        ],
    }, sort_keys=True))
PY

echo LOG_COMPLETE
tail -3 "$base/_launch/a9-a11-highcap-3090-r2/job.log"
echo GPU4
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5p'
