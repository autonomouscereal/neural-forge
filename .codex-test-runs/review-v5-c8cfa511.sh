#!/usr/bin/env bash
set -euo pipefail

umask 077

deployment_root=/home/cereal/model-surgery-lab/deployments/neural-forge-a884b2d1
artifact_root=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804
python_bin=/home/cereal/model-surgery-lab/venv/bin/python
materialization_index="$artifact_root/materialization-index-c8cfa511.jsonl"
v5_pid_file="$artifact_root/v5-r12-c8cfa511-rtx8000.pid"
v5_receipt="$artifact_root/candidate-packs/r12-v5-launch-receipt-v2.json"
launch_manifest="$deployment_root/experiments/manifests/lfm2-exact4x-moving-gauge-v5-r12-launch.json"
authority_key=/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key
prompts_json="$deployment_root/research/lfm2_semantic_whole_response_prompts_v1.json"
prompts_sha256=0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b
review_gpu_index=6
review_gpu_uuid=GPU-f5a09535-fc85-a027-50f7-734b344747dd
review_root="$artifact_root/whole-response-reviews-c8cfa511"
machine_root="$review_root/machine-gpu-f5a09535"
manual_root="$review_root/manual-intelligence-judgments"

fail() {
    printf 'v5-review-launcher-error=%s\n' "$*" >&2
    exit 2
}

test -d "$deployment_root" || fail "deployment-root-missing"
test -x "$python_bin" || fail "python-runtime-missing"
test -s "$v5_pid_file" || fail "v5-pid-file-missing"
test -s "$v5_receipt" || fail "authenticated-v5-receipt-missing"
test -s "$materialization_index" || fail "materialization-index-missing"
test -s "$prompts_json" || fail "frozen-prompt-pack-missing"
test -s "$launch_manifest" || fail "launch-manifest-missing"
test -s "$authority_key" || fail "launch-authority-key-missing"

v5_pid=$(tr -d '[:space:]' < "$v5_pid_file")
[[ "$v5_pid" =~ ^[1-9][0-9]*$ ]] || fail "v5-pid-invalid"
if kill -0 "$v5_pid" 2>/dev/null; then
    printf 'v5-still-live=%s\n' "$v5_pid" >&2
    exit 75
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi-missing"
command -v flock >/dev/null 2>&1 || fail "flock-missing"
gpu6_uuid=$(
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
        awk -F ',[[:space:]]*' -v expected="$review_gpu_index" \
            '$1 == expected { print $2 }'
)
test "$gpu6_uuid" = "$review_gpu_uuid" || fail "gpu6-uuid-mismatch"

mkdir -p "$review_root"
exec 9>"$review_root/.review-v5-c8cfa511.lock"
flock -n 9 || fail "another-v5-review-launcher-is-live"

require_idle_gpu6() {
    local compute_rows
    compute_rows=$(
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name \
            --format=csv,noheader,nounits 2>/dev/null
    ) || fail "gpu6-compute-process-query-failed"
    if printf '%s\n' "$compute_rows" | grep -Fq "$review_gpu_uuid"; then
        fail "gpu6-has-active-compute-process"
    fi
}

require_idle_gpu6

cd "$deployment_root"
export PYTHONPATH="$deployment_root/src:$deployment_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# UUID selection makes the physical RTX 8000 appear to the review CLI as cuda:0.
export CUDA_VISIBLE_DEVICES="$review_gpu_uuid"

plan_file=$(mktemp "$artifact_root/.review-v5-c8cfa511.XXXXXX")
trap 'rm -f "$plan_file"' EXIT

# Authenticate every index row and child-local package before any generation.
# The plan is written in manual-review priority order, independent of index order.
"$python_bin" - "$materialization_index" "$prompts_json" \
    "$prompts_sha256" "$plan_file" "$launch_manifest" \
    "$authority_key" "$v5_receipt" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from neural_forge.lfm2_carrier_child_package import (
    load_authenticated_lfm2_carrier_child_package,
)
from neural_forge.lfm2_exact4x_moving_gauge_v5_candidate_materializer import (
    _load_materialization_receipt,
    load_authenticated_lfm2_exact4x_v5_materialization_provenance,
)


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} is not a lowercase SHA-256")
    return value


def ordinary_file(path: Path, name: str) -> Path:
    absolute = path.expanduser().absolute()
    resolved = absolute.resolve(strict=True)
    if (
        absolute.is_symlink()
        or not resolved.is_file()
        or getattr(resolved.stat(), "st_nlink", 1) != 1
    ):
        raise RuntimeError(f"{name} is not one ordinary file")
    return resolved


index_path = ordinary_file(Path(sys.argv[1]), "materialization index")
prompt_path = ordinary_file(Path(sys.argv[2]), "frozen prompt pack")
expected_prompt_sha256 = digest(sys.argv[3], "frozen prompt pack digest")
plan_path = Path(sys.argv[4]).expanduser().absolute()
manifest_path = ordinary_file(Path(sys.argv[5]), "V5 launch manifest")
authority_key_path = ordinary_file(Path(sys.argv[6]), "V5 authority key")
receipt_path = ordinary_file(Path(sys.argv[7]), "V5 final receipt")

provenance = load_authenticated_lfm2_exact4x_v5_materialization_provenance(
    launch_manifest_json=manifest_path,
    authority_key_path=authority_key_path,
    chart_mode="E0A",
    target_arm="sequential_conditioned",
)
if (
    Path(provenance.launch_manifest["receipt_path"])
    .expanduser()
    .absolute()
    .resolve(strict=True)
    != receipt_path
):
    raise RuntimeError("authenticated launch points to a different final receipt")
expected_output_root = provenance.output_root.resolve(strict=True)
expected_manifest_sha256 = provenance.launch_manifest["launch_manifest_sha256"]
expected_receipt_sha256 = provenance.final_receipt["receipt_sha256"]
expected_receipt_hmac_sha256 = provenance.final_receipt["receipt_hmac_sha256"]
expected_receipt_hmac_key_id = provenance.final_receipt["receipt_hmac_key_id"]
expected_phase8_sha256 = provenance.phase8_outcome["artifact_sha256"]
expected_source_package_sha256 = provenance.source_package.package_sha256
expected_source_state_sha256 = provenance.source_fingerprint.state_dict_sha256
expected_source_identity_sha256 = provenance.source_fingerprint.identity_sha256
expected_plan_sha256 = provenance.counted_plan.fingerprint
expected_request_sha256 = provenance.request.request_sha256

if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != expected_prompt_sha256:
    raise RuntimeError("frozen prompt pack digest mismatch")
prompt_payload = json.loads(
    prompt_path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
prompt_rows = prompt_payload.get("prompts") if isinstance(prompt_payload, dict) else None
if (
    prompt_payload.get("schema")
    != "neural-forge-whole-response-review-prompts-v1"
    or not isinstance(prompt_rows, list)
    or len(prompt_rows) != 16
    or len({row.get("id") for row in prompt_rows if isinstance(row, dict)}) != 16
):
    raise RuntimeError("frozen prompt pack is not the sealed 16-prompt protocol")

raw_lines = index_path.read_bytes().splitlines()
if len(raw_lines) != 6 or any(not line.strip() for line in raw_lines):
    raise RuntimeError("materialization index must contain exactly six nonempty lines")

required_fields = {
    "child_directory",
    "child_package_sha256",
    "published_child_artifact_sha256",
    "candidate_materialization_sha256",
    "candidate_pack_sha256",
}
by_lane: dict[tuple[str, str], tuple[Path, str, str]] = {}
seen_children: set[Path] = set()
seen_packages: set[str] = set()
seen_materializations: set[str] = set()
seen_candidate_packs: set[str] = set()

for line_number, raw_line in enumerate(raw_lines, start=1):
    try:
        row = json.loads(raw_line.decode("ascii"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"index line {line_number} is not strict ASCII JSON") from exc
    if not isinstance(row, dict) or set(row) != required_fields:
        raise RuntimeError(f"index line {line_number} has an unexpected field set")

    package_sha256 = digest(row["child_package_sha256"], "child package digest")
    artifact_sha256 = digest(
        row["published_child_artifact_sha256"],
        "published child artifact digest",
    )
    materialization_sha256 = digest(
        row["candidate_materialization_sha256"],
        "candidate materialization digest",
    )
    candidate_pack_sha256 = digest(
        row["candidate_pack_sha256"],
        "candidate pack digest",
    )
    if artifact_sha256 != package_sha256:
        raise RuntimeError("V5 published artifact digest differs from its child package")

    child_text = row["child_directory"]
    if (
        not isinstance(child_text, str)
        or not child_text
        or any(character in child_text for character in "\t\r\n")
    ):
        raise RuntimeError("child directory is not a safe plan field")
    child = Path(child_text).expanduser().absolute()
    resolved_child = child.resolve(strict=True)
    if (
        child != resolved_child
        or child.is_symlink()
        or not child.is_dir()
        or child.name != package_sha256
    ):
        raise RuntimeError("child directory is not the canonical package directory")

    package = load_authenticated_lfm2_carrier_child_package(
        child,
        expected_package_sha256=package_sha256,
    )
    if package.root != resolved_child or package.package_sha256 != package_sha256:
        raise RuntimeError("fresh child-local package authentication changed identity")
    materialization = _load_materialization_receipt(
        child,
        expected_materialization_sha256=materialization_sha256,
    )
    chart = materialization.get("chart_mode")
    arm = materialization.get("target_arm")
    lane = (chart, arm)
    candidate = provenance.candidates.get(f"{chart}:{arm}")
    if (
        child.parent != expected_output_root
        or candidate is None
        or candidate.candidate_pack_sha256 != candidate_pack_sha256
        or materialization.get("v5_launch_manifest_sha256")
        != expected_manifest_sha256
        or materialization.get("v5_launch_receipt_sha256")
        != expected_receipt_sha256
        or materialization.get("v5_launch_receipt_hmac_sha256")
        != expected_receipt_hmac_sha256
        or materialization.get("v5_launch_receipt_hmac_key_id")
        != expected_receipt_hmac_key_id
        or materialization.get("v5_phase8_artifact_sha256")
        != expected_phase8_sha256
        or materialization.get("source_child_package_sha256")
        != expected_source_package_sha256
        or materialization.get("source_child_state_dict_sha256")
        != expected_source_state_sha256
        or materialization.get("source_child_identity_sha256")
        != expected_source_identity_sha256
        or materialization.get("counted_child_state_plan_fingerprint")
        != expected_plan_sha256
        or materialization.get("request_sha256") != expected_request_sha256
        or materialization.get("candidate_pack_sha256") != candidate_pack_sha256
        or materialization.get("standalone_child_local_runtime") is not True
        or materialization.get("installed") is not True
        or materialization.get("exactly_two_tensors_replaced") is not True
        or materialization.get("noncandidate_state_bit_identical") is not True
        or materialization.get("config_topology_and_tokenizers_unchanged")
        is not True
        or materialization.get("carrier_routes_and_lexicon_unchanged") is not True
        or materialization.get("carrier_route_package_freshly_reauthenticated")
        is not True
        or materialization.get("manual_semantic_review_status") != "pending"
        or materialization.get("publication_authorized") is not False
        or materialization.get("accepted_artifact") is not False
        or materialization.get("parent_model_load_count") != 0
        or materialization.get("parent_forward_count") != 0
        or materialization.get("child_forward_count") != 0
        or materialization.get("child_generation_count") != 0
        or materialization.get("training_steps") != 0
        or materialization.get("gradient_steps") != 0
        or materialization.get("backward_calls") != 0
        or materialization.get("optimizer_steps") != 0
        or materialization.get("quantization_calls") != 0
        or materialization.get("pruning_calls") != 0
        or materialization.get("parameter_sharing_credit") is not False
        or materialization.get("runtime_sidecar_parameter_elements") != 0
    ):
        raise RuntimeError(
            "candidate materialization is not fully bound child-only fold-only provenance"
        )

    if chart not in {"E0", "E0R", "E0A"} or arm not in {
        "joint",
        "sequential_conditioned",
    }:
        raise RuntimeError("candidate materialization has an unknown review lane")
    if lane in by_lane:
        raise RuntimeError(f"duplicate materialized lane: {chart}:{arm}")
    if (
        child in seen_children
        or package_sha256 in seen_packages
        or materialization_sha256 in seen_materializations
        or candidate_pack_sha256 in seen_candidate_packs
    ):
        raise RuntimeError("materialization index reuses a candidate identity")

    seen_children.add(child)
    seen_packages.add(package_sha256)
    seen_materializations.add(materialization_sha256)
    seen_candidate_packs.add(candidate_pack_sha256)
    by_lane[lane] = (child, package_sha256, artifact_sha256)

priority = (
    ("E0A", "sequential_conditioned"),
    ("E0A", "joint"),
    ("E0R", "sequential_conditioned"),
    ("E0R", "joint"),
    ("E0", "sequential_conditioned"),
    ("E0", "joint"),
)
if set(by_lane) != set(priority):
    raise RuntimeError("authenticated materializations do not cover all six V5 lanes")

with plan_path.open("w", encoding="ascii", newline="\n") as handle:
    for chart, arm in priority:
        child, package_sha256, artifact_sha256 = by_lane[(chart, arm)]
        label = f"{chart}-{arm}"
        handle.write(
            f"{label}\t{child}\t{package_sha256}\t{artifact_sha256}\n"
        )
    handle.flush()
    os.fsync(handle.fileno())
PY

test "$(wc -l < "$plan_file")" -eq 6 || fail "authenticated-review-plan-incomplete"
mkdir -p "$machine_root"

# Exit 0 only for a complete, self-sealed machine dossier bound to this exact
# child, published artifact, prompt pack, runtime policy, and review protocol.
# Exit 10 means no dossier yet; every other result is a hard mismatch.
dossier_matches() {
    "$python_bin" - "$1" "$2" "$3" "$prompts_sha256" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.lfm2_carrier_review_cli import SCHEMA


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate dossier key: {key!r}")
        result[key] = value
    return result


path = Path(sys.argv[1]).expanduser().absolute()
package_sha256, artifact_sha256, prompts_sha256 = sys.argv[2:5]
if not path.exists() and not path.is_symlink():
    raise SystemExit(10)
if path.is_symlink() or not path.is_file() or getattr(path.stat(), "st_nlink", 1) != 1:
    print("existing dossier is not one ordinary file", file=sys.stderr)
    raise SystemExit(11)
try:
    dossier = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
    )
except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
    print(f"existing dossier is not strict JSON: {exc}", file=sys.stderr)
    raise SystemExit(11)
if not isinstance(dossier, dict):
    print("existing dossier is not a JSON object", file=sys.stderr)
    raise SystemExit(11)
body = dict(dossier)
seal = body.pop("dossier_sha256", None)
policy = dossier.get("evaluation_policy")
valid = (
    isinstance(seal, str)
    and seal == canonical_sha256(body)
    and dossier.get("schema") == SCHEMA
    and dossier.get("status") == "complete-ready-for-manual-intelligence-review"
    and dossier.get("child_package_sha256") == package_sha256
    and dossier.get("published_child_artifact_sha256") == artifact_sha256
    and dossier.get("prompts_file_sha256") == prompts_sha256
    and dossier.get("prompt_count") == 16
    and dossier.get("response_count") == 16
    and dossier.get("runtime_device") == "cuda:0"
    and dossier.get("runtime_dtype") == "float32"
    and dossier.get("sealed_child_table_dtype") == "float32"
    and dossier.get("direct_local_model_loader_used") is True
    and dossier.get("hugging_face_auto_loader_used") is False
    and dossier.get("network_access") is False
    and dossier.get("original_route_publication_loaded") is False
    and dossier.get("original_span_evidence_loaded") is False
    and dossier.get("original_tokenizer_source_loaded") is False
    and dossier.get("maximum_prompt_tokens") == 8192
    and dossier.get("initial_maximum_new_tokens") == 1024
    and dossier.get("retry_maximum_new_tokens") == 2048
    and dossier.get("adaptive_retry_response_count")
    == dossier.get("initial_cap_censored_response_count")
    and dossier.get("safety_observations_counted_as_folding_failures") is False
    and dossier.get("safety_observations_counted_as_intelligence_failures") is False
    and dossier.get("cap_censored_responses_counted_as_folding_failures") is False
    and dossier.get("manual_whole_response_intelligence_review_is_authoritative")
    is True
    and dossier.get("automatic_intelligence_verdict_performed") is False
    and dossier.get("numeric_metrics_have_acceptance_authority") is False
    and isinstance(policy, dict)
    and policy.get("safety_prompts_excluded_from_folding_failure") is True
    and policy.get("safety_prompts_excluded_from_intelligence_failure") is True
    and policy.get("retry_scope")
    == "only-initial-output-cap-censored-responses"
    and policy.get("remaining_output_cap_censored_is_failure") is False
)
if not valid:
    print("existing dossier is incomplete or identity/policy mismatched", file=sys.stderr)
    raise SystemExit(11)
PY
}

while IFS=$'\t' read -r label child_dir package_sha256 artifact_sha256; do
    test -n "$label" || fail "empty-review-label"
    candidate_root="$machine_root/$label"
    checkpoint_dir="$candidate_root/checkpoints"
    dossier_root="$candidate_root/dossier"
    dossier_json="$dossier_root/carrier-review.json"
    mkdir -p "$checkpoint_dir" "$dossier_root"

    printf 'review-start=%s child=%s gpu_uuid=%s\n' \
        "$label" "$package_sha256" "$review_gpu_uuid"
    require_idle_gpu6
    "$python_bin" -u -m neural_forge.lfm2_carrier_review_cli \
        --child-dir "$child_dir" \
        --expected-child-package-sha256 "$package_sha256" \
        --published-child-artifact-sha256 "$artifact_sha256" \
        --prompts-json "$prompts_json" \
        --expected-prompts-sha256 "$prompts_sha256" \
        --checkpoint-dir "$checkpoint_dir" \
        --dossier-json "$dossier_json" \
        --device cuda:0 \
        --dtype float32 \
        --maximum-prompt-tokens 8192 \
        --initial-maximum-new-tokens 1024 \
        --retry-maximum-new-tokens 2048

    dossier_matches \
        "$dossier_json" "$package_sha256" "$artifact_sha256" || \
        fail "completed-dossier-failed-reauthentication:$label"
    printf 'review-complete=%s dossier=%s\n' "$label" "$dossier_json"
done < "$plan_file"

printf 'machine_dossier_root=%s\n' "$machine_root"
printf 'manual_judgment_root=%s\n' "$manual_root"
printf '%s\n' \
    'manual_whole_response_intelligence_review_required=true' \
    'safety_refusals_are_not_model_failures=true' \
    'output_cap_censoring_is_not_model_failure=true'
