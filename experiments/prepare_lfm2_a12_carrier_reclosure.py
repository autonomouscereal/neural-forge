"""Prepare A12 fit-only evidence for the existing carrier-reclosure runner.

This is deliberately an adapter, not another carrier compiler.  It converts
the authenticated A12 prompt manifest, A12 trajectory audit, and the freshly
reloaded independent C_in/C_out support publication into the exact artifacts
consumed by :mod:`run_lfm2_carrier_reclosure`.  It executes no model and no
training.  Review prompts are bound only as a future evaluation inventory and
are never read for evidence production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = Path(__file__).resolve().parent
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_causal_semantic_support import (  # noqa: E402
    Lfm2CausalSupportFeaturePack,
)
from neural_forge.hf_lfm2_semantic_carrier_allocation import (  # noqa: E402
    build_lfm2_semantic_carrier_demand_evidence,
    build_lfm2_semantic_carrier_inventory,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_lfm2_sparse_semantic_codebook,
)
from neural_forge.tokenizer_control_contract import (  # noqa: E402
    compile_local_tokenizer_control_contract,
)
import run_lfm2_carrier_reclosure as carrier_runner  # noqa: E402
import run_lfm2_causal_semantic_support as support_runner  # noqa: E402


PREPARATION_SCHEMA = "neural-forge-lfm2-a12-carrier-reclosure-preparation-v1"
PREPARATION_COMPLETION_SCHEMA = (
    "neural-forge-lfm2-a12-carrier-reclosure-preparation-completion-v1"
)
_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "model_objects_constructed": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2A12CarrierPreparationError(RuntimeError):
    """Raised when A12 carrier preparation cannot authenticate its inputs."""


def _plain_json(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2A12CarrierPreparationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2A12CarrierPreparationError(f"{name} must be positive.")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2A12CarrierPreparationError(f"Cannot read {name} JSON.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2A12CarrierPreparationError(f"{name} must be a JSON object.")
    return _plain_json(value)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".safetensors", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in tensors.items()
            },
            str(temporary_path),
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_pinned_matrix(
    path: Path, *, key: str, expected_file_sha256: str, name: str
) -> torch.Tensor:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise Lfm2A12CarrierPreparationError(f"{name} must be an ordinary file.")
    if _file_sha256(resolved) != _digest(expected_file_sha256, f"{name} file pin"):
        raise Lfm2A12CarrierPreparationError(f"{name} file hash differs.")
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise Lfm2A12CarrierPreparationError(
                f"{name} tensor key {key!r} is absent."
            )
        tensor = handle.get_tensor(key).detach().cpu().contiguous()
    if tensor.ndim != 2 or not torch.is_floating_point(tensor):
        raise Lfm2A12CarrierPreparationError(f"{name} must be a floating matrix.")
    return tensor


def _authenticate_feature_pack(
    tensor_path: Path,
    receipt_path: Path,
    *,
    expected_tensor_file_sha256: str,
    expected_artifact_sha256: str,
) -> Lfm2CausalSupportFeaturePack:
    if _file_sha256(tensor_path) != _digest(
        expected_tensor_file_sha256, "feature tensor file pin"
    ):
        raise Lfm2A12CarrierPreparationError("Feature tensor file hash differs.")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {
            "input_feature_transform",
            "output_feature_transform",
        }:
            raise Lfm2A12CarrierPreparationError(
                "Feature tensor inventory is not exact."
            )
        value = Lfm2CausalSupportFeaturePack(
            handle.get_tensor("input_feature_transform"),
            handle.get_tensor("output_feature_transform"),
            _read_json(receipt_path, "feature receipt"),
        )
    if value.artifact_sha256 != _digest(
        expected_artifact_sha256, "feature artifact pin"
    ):
        raise Lfm2A12CarrierPreparationError("Feature artifact seal differs.")
    return value


def _presence_counts(
    *,
    audit: Mapping[str, Any],
    prompt_manifest: Mapping[str, Any],
    donor_count: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Count per-prompt structural presence, never raw occurrence frequency."""

    fit_ids = prompt_manifest["fit_prompt_ids"]
    token_rows = prompt_manifest["prompt_token_ids_by_prompt"]
    results = {row["prompt_id"]: row for row in audit["prompt_results"]}
    occurrence = torch.zeros(donor_count, dtype=torch.int64)
    winners = torch.zeros(donor_count, dtype=torch.int64)
    for prompt_id in fit_ids:
        result = results[prompt_id]
        forced = {
            int(row["next_forced_token_id"])
            for row in result["parent_reference_tokens"][:-1]
        }
        present = {int(value) for value in token_rows[prompt_id]}.union(forced)
        winner_set = {
            int(row["parent_winner_token_id"])
            for row in result["lanes"]["exact_exact"]["tokens"]
        }
        if any(value < 0 or value >= donor_count for value in present | winner_set):
            raise Lfm2A12CarrierPreparationError(
                "A12 carrier demand contains an undefined donor ID."
            )
        occurrence[torch.tensor(sorted(present), dtype=torch.int64)] += 1
        winners[torch.tensor(sorted(winner_set), dtype=torch.int64)] += 1
    diagnostics = {
        "fit_prompt_count": len(fit_ids),
        "donor_prompt_presence_nonzero_count": int((occurrence > 0).sum().item()),
        "donor_prompt_winner_presence_nonzero_count": int((winners > 0).sum().item()),
        "raw_occurrence_frequency_used": False,
        "aggregation": "maximum-one-count-per-donor-per-prompt-per-role",
    }
    return occurrence, winners, diagnostics


def _route_row(
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    coefficients: torch.Tensor,
    donor_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(offsets[donor_id].item())
    stop = int(offsets[donor_id + 1].item())
    return target_ids[start:stop], coefficients[start:stop]


def _canonicalize_csr_target_order(
    offsets: torch.Tensor,
    target_ids: torch.Tensor,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort each support row without changing its linear combination."""

    ordered_ids = target_ids.clone()
    ordered_coefficients = coefficients.clone()
    for donor_id in range(int(offsets.numel()) - 1):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        if stop - start <= 1:
            continue
        permutation = torch.argsort(ordered_ids[start:stop], stable=True)
        ordered_ids[start:stop] = ordered_ids[start:stop].index_select(0, permutation)
        ordered_coefficients[start:stop] = ordered_coefficients[
            start:stop
        ].index_select(0, permutation)
    return offsets.clone(), ordered_ids.contiguous(), ordered_coefficients.contiguous()


def _relative_feature_error(
    source: torch.Tensor, reconstruction: torch.Tensor, transform: torch.Tensor
) -> float:
    source_feature = source @ transform.transpose(0, 1)
    residual_feature = (source - reconstruction) @ transform.transpose(0, 1)
    numerator = torch.linalg.vector_norm(residual_feature)
    denominator = torch.linalg.vector_norm(source_feature).clamp_min(1e-300)
    return float((numerator / denominator).item())


def _demand_diagnostics(
    *,
    occurrence: torch.Tensor,
    winners: torch.Tensor,
    parent_table: torch.Tensor,
    target_table: torch.Tensor,
    input_transform: torch.Tensor,
    output_transform: torch.Tensor,
    input_offsets: torch.Tensor,
    input_target_ids: torch.Tensor,
    input_coefficients: torch.Tensor,
    output_offsets: torch.Tensor,
    output_target_ids: torch.Tensor,
    output_coefficients: torch.Tensor,
    target_to_donor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    donor_count = int(parent_table.shape[0])
    errors = torch.zeros(donor_count, dtype=torch.float64)
    costs = torch.zeros(donor_count, dtype=torch.float64)
    parent = parent_table.to(torch.float64)
    target = target_table.to(torch.float64)
    input_view = input_transform.to(torch.float64)
    output_view = output_transform.to(torch.float64)
    demanded = torch.nonzero((occurrence > 0) | (winners > 0), as_tuple=False).reshape(
        -1
    )
    for donor_id in demanded.tolist():
        input_ids, input_values = _route_row(
            input_offsets, input_target_ids, input_coefficients, donor_id
        )
        output_ids, output_values = _route_row(
            output_offsets, output_target_ids, output_coefficients, donor_id
        )
        input_reconstruction = input_values.to(torch.float64) @ target.index_select(
            0, input_ids
        )
        output_reconstruction = output_values.to(torch.float64) @ target.index_select(
            0, output_ids
        )
        input_error = _relative_feature_error(
            parent[donor_id], input_reconstruction, input_view
        )
        output_error = _relative_feature_error(
            parent[donor_id], output_reconstruction, output_view
        )
        errors[donor_id] = max(input_error, output_error)
    direct_rows = parent.index_select(0, target_to_donor)
    direct_norms = torch.linalg.vector_norm(direct_rows, dim=1)
    costs[target_to_donor] = direct_norms
    return errors, costs


def _sealed_artifact(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {**_plain_json(body), "artifact_sha256": canonical_sha256(body)}
    _atomic_write_json(path, receipt)
    return receipt


def _file_spec(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


# Scoped, not process-global: a bare ``torch.set_grad_enabled(False)`` here
# leaked into the calling thread for the rest of the interpreter and silently
# disabled autograd for everything that ran afterwards.
@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.trajectory_audit_json.expanduser().resolve()
    audit_completion_path = args.trajectory_completion_json.expanduser().resolve()
    audit, _audit_completion = support_runner.authenticate_completed_trajectory_audit(
        audit_path=audit_path,
        completion_path=audit_completion_path,
        expected_audit_file_sha256=args.expected_trajectory_audit_file_sha256,
        expected_audit_sha256=args.expected_trajectory_audit_sha256,
        expected_completion_sha256=args.expected_trajectory_completion_sha256,
    )
    feature_pack = _authenticate_feature_pack(
        args.feature_tensors_safetensors.expanduser().resolve(),
        args.feature_receipt_json.expanduser().resolve(),
        expected_tensor_file_sha256=args.expected_feature_tensor_file_sha256,
        expected_artifact_sha256=args.expected_feature_artifact_sha256,
    )
    manifest = support_runner.authenticate_fit_prompt_manifest(
        _read_json(args.fit_prompt_manifest_json, "fit prompt manifest"),
        expected_manifest_sha256=args.expected_fit_prompt_manifest_sha256,
        expected_audit_sha256=audit["audit_sha256"],
        feature_pack=feature_pack,
    )
    if manifest["manual_review_prompt_ids_sha256"] != _digest(
        args.expected_manual_review_prompt_ids_sha256,
        "manual-review prompt-ID inventory pin",
    ):
        raise Lfm2A12CarrierPreparationError(
            "Carrier preparation is bound to a different future review inventory."
        )
    support_directory = args.support_publication_directory.expanduser().resolve()
    support = support_runner._load_cached_selection(
        support_directory,
        expected_cache_key=_digest(
            args.expected_support_cache_key, "support cache key"
        ),
    )
    if support is None or support.result_sha256 != _digest(
        args.expected_support_result_sha256, "support result pin"
    ):
        raise Lfm2A12CarrierPreparationError(
            "Independent support publication authentication failed."
        )
    # This fresh-reload observer is intentionally executed even though the
    # selection was already authenticated above.
    allocations = support_runner.causal_support_allocations_from_publication(
        support_directory,
        expected_cache_key=args.expected_support_cache_key,
    )
    if len(allocations) != 2:
        raise Lfm2A12CarrierPreparationError(
            "Independent support publication has no exact two-route allocation."
        )
    input_route = _canonicalize_csr_target_order(
        support.input_offsets,
        support.input_target_ids,
        support.input_coefficients,
    )
    output_route = _canonicalize_csr_target_order(
        support.output_offsets,
        support.output_target_ids,
        support.output_coefficients,
    )
    codebook = load_lfm2_sparse_semantic_codebook(
        args.input_codebook_directory.expanduser().resolve(),
        expected_artifact_sha256=_digest(
            args.expected_input_codebook_artifact_sha256,
            "input codebook artifact pin",
        ),
    )
    donor_count = int(codebook.receipt["donor_row_count"])
    target_count = int(codebook.receipt["target_row_count"])
    hidden_size = int(codebook.receipt["hidden_size"])
    parent_table = _load_pinned_matrix(
        args.parent_table_safetensors,
        key=args.parent_table_key,
        expected_file_sha256=args.expected_parent_table_file_sha256,
        name="parent table",
    )[:donor_count, :]
    target_table = _load_pinned_matrix(
        args.target_table_safetensors,
        key=args.target_table_key,
        expected_file_sha256=args.expected_target_table_file_sha256,
        name="target table",
    )[:target_count, :]
    if tuple(parent_table.shape) != (donor_count, hidden_size) or tuple(
        target_table.shape
    ) != (target_count, hidden_size):
        raise Lfm2A12CarrierPreparationError(
            "Parent/target carrier table geometry differs from the semantic base."
        )
    solver = {
        "support_size": _positive_int(args.support_size, "support size"),
        "rows_per_chunk": _positive_int(args.rows_per_chunk, "rows per chunk"),
        "ridge": float(args.ridge),
        "selection_device": args.selection_device,
        "selection_dtype": args.selection_dtype,
        "solve_device": "cpu",
        "solve_dtype": "float64",
    }
    numeric_contract = carrier_runner._solver_math_contract(solver)
    numeric_contract_sha256 = canonical_sha256(numeric_contract)
    tokenizer_hashes = {
        "tokenizer.json": _digest(
            args.expected_tokenizer_json_sha256, "tokenizer.json pin"
        ),
        "tokenizer_config.json": _digest(
            args.expected_tokenizer_config_sha256, "tokenizer config pin"
        ),
        "chat_template.jinja": _digest(
            args.expected_chat_template_sha256, "chat template pin"
        ),
    }
    tokenizer_directory = args.tokenizer_directory.expanduser().resolve()
    tokenizer_contract = compile_local_tokenizer_control_contract(
        tokenizer_directory,
        expected_file_sha256=tokenizer_hashes,
    )
    base_tensors = {
        "target_table": target_table,
        "target_to_donor": codebook.target_to_donor,
        "input_offsets": input_route[0],
        "input_target_ids": input_route[1],
        "input_coefficients": input_route[2],
        "output_offsets": output_route[0],
        "output_target_ids": output_route[1],
        "output_coefficients": output_route[2],
    }
    inventory = build_lfm2_semantic_carrier_inventory(
        input_offsets=input_route[0],
        input_target_ids=input_route[1],
        input_coefficients=input_route[2],
        output_offsets=output_route[0],
        output_target_ids=output_route[1],
        output_coefficients=output_route[2],
        target_to_donor=codebook.target_to_donor,
    )
    base_binding_sha256 = canonical_sha256(inventory.as_dict()["base_binding"])
    occurrence, winners, presence_diagnostics = _presence_counts(
        audit=audit,
        prompt_manifest=manifest,
        donor_count=donor_count,
    )
    errors, costs = _demand_diagnostics(
        occurrence=occurrence,
        winners=winners,
        parent_table=parent_table,
        target_table=target_table,
        input_transform=feature_pack.input_feature_transform,
        output_transform=feature_pack.output_feature_transform,
        input_offsets=input_route[0],
        input_target_ids=input_route[1],
        input_coefficients=input_route[2],
        output_offsets=output_route[0],
        output_target_ids=output_route[1],
        output_coefficients=output_route[2],
        target_to_donor=codebook.target_to_donor,
    )
    direct_set = set(codebook.target_to_donor.tolist())
    protected_set = set(tokenizer_contract.protected_id_values)
    demanded_non_direct = [
        donor_id
        for donor_id in torch.nonzero((occurrence > 0) | (winners > 0), as_tuple=False)
        .reshape(-1)
        .tolist()
        if donor_id not in direct_set and donor_id not in protected_set
    ]
    promotion_count = min(
        len(demanded_non_direct),
        _positive_int(args.maximum_promotion_count, "maximum promotion count"),
    )
    if promotion_count <= 0:
        raise Lfm2A12CarrierPreparationError(
            "A12 contains no demanded unprotected non-direct carrier candidates."
        )
    request_identity_body = {
        "schema": PREPARATION_SCHEMA,
        "trajectory_audit_sha256": audit["audit_sha256"],
        "fit_prompt_manifest_sha256": manifest["manifest_sha256"],
        "manual_review_prompt_ids_sha256": manifest["manual_review_prompt_ids_sha256"],
        "feature_pack_artifact_sha256": feature_pack.artifact_sha256,
        "support_result_sha256": support.result_sha256,
        "support_allocation_sha256s": [value.artifact_sha256 for value in allocations],
        "input_codebook_artifact_sha256": codebook.artifact_sha256,
        "parent_table_file_sha256": args.expected_parent_table_file_sha256,
        "parent_identity_sha256": _digest(
            args.parent_identity_sha256, "parent identity pin"
        ),
        "target_table_file_sha256": args.expected_target_table_file_sha256,
        "tokenizer_control_receipt_sha256": tokenizer_contract.receipt_sha256,
        "numeric_contract": numeric_contract,
        "numeric_contract_sha256": numeric_contract_sha256,
        "promotion_count": promotion_count,
        "winner_weight": _positive_int(args.winner_weight, "winner weight"),
        "allow_positive_incidence": True,
        "producer_implementation_sha256": _file_sha256(Path(__file__).resolve()),
        "review_prompts_consumed": 0,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    preparation_cache_key = canonical_sha256(request_identity_body)
    directory = args.output_directory.expanduser().resolve() / preparation_cache_key
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = _read_json(complete_path, "carrier preparation completion")
        supplied = complete.pop("completion_sha256", None)
        request_path = directory / "carrier-request.json"
        if (
            supplied != canonical_sha256(complete)
            or complete.get("schema") != PREPARATION_COMPLETION_SCHEMA
            or complete.get("preparation_cache_key") != preparation_cache_key
            or complete.get("request_file_sha256") != _file_sha256(request_path)
        ):
            raise Lfm2A12CarrierPreparationError(
                "Cached carrier preparation completion failed authentication."
            )
        return {**complete, "completion_sha256": supplied, "cache_hit": True}
    base_path = directory / "base-bridge.safetensors"
    parent_path = directory / "parent-table.safetensors"
    evidence_tensor_path = directory / "carrier-evidence.safetensors"
    evidence_receipt_path = directory / "carrier-evidence.json"
    input_metric_tensor_path = directory / "input-metric.safetensors"
    input_metric_receipt_path = directory / "input-metric.json"
    output_metric_tensor_path = directory / "output-metric.safetensors"
    output_metric_receipt_path = directory / "output-metric.json"
    _atomic_save_tensors(base_path, base_tensors)
    _atomic_save_tensors(parent_path, {"parent_table": parent_table})
    evidence_tensors = {
        "donor_occurrence_counts": occurrence,
        "donor_winner_counts": winners,
        "donor_reconstruction_errors": errors,
        "donor_eviction_costs": costs,
    }
    _atomic_save_tensors(evidence_tensor_path, evidence_tensors)
    producer_sha = request_identity_body["producer_implementation_sha256"]
    compiled_evidence = build_lfm2_semantic_carrier_demand_evidence(
        occurrence,
        winners,
        errors,
        costs,
        source_artifact_sha256=support.result_sha256,
        source_corpus_sha256=audit["audit_sha256"],
        parent_identity_sha256=_digest(
            args.parent_identity_sha256, "parent identity pin"
        ),
        fit_partition_manifest_sha256=manifest["manifest_sha256"],
        producer_implementation_sha256=producer_sha,
        numeric_contract_sha256=numeric_contract_sha256,
        base_semantic_binding_sha256=base_binding_sha256,
        partition_role="fold-fit",
        selection_allowed=True,
    )
    evidence_body = {
        "schema": carrier_runner.CARRIER_EVIDENCE_INPUT_SCHEMA,
        "partition_role": "fold-fit",
        "selection_allowed": True,
        "review_or_evaluation_evidence_allowed": False,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "tensor_file_sha256": _file_sha256(evidence_tensor_path),
        "source_artifact_sha256": support.result_sha256,
        "source_corpus_sha256": audit["audit_sha256"],
        "parent_identity_sha256": args.parent_identity_sha256,
        "fit_partition_manifest_sha256": manifest["manifest_sha256"],
        "producer_implementation_sha256": producer_sha,
        "numeric_contract_sha256": numeric_contract_sha256,
        "base_semantic_binding_sha256": base_binding_sha256,
        "compiled_evidence_artifact_sha256": compiled_evidence.artifact_sha256,
        "presence_diagnostics": presence_diagnostics,
        **{
            f"{name}_sha256": carrier_runner._tensor_sha256(tensor)
            for name, tensor in evidence_tensors.items()
        },
    }
    _sealed_artifact(evidence_receipt_path, evidence_body)
    input_transform = feature_pack.input_feature_transform.to(torch.float64)
    output_transform = feature_pack.output_feature_transform.to(torch.float64)
    input_metric = (input_transform.transpose(0, 1) @ input_transform).contiguous()
    output_metric = (
        output_transform.transpose(0, 1)
        @ output_transform
        / float(output_transform.shape[0])
    ).contiguous()
    metric_specs: dict[str, dict[str, dict[str, str]]] = {}
    for channel, metric, tensor_path, receipt_path in (
        (
            "input",
            input_metric,
            input_metric_tensor_path,
            input_metric_receipt_path,
        ),
        (
            "output",
            output_metric,
            output_metric_tensor_path,
            output_metric_receipt_path,
        ),
    ):
        _atomic_save_tensors(tensor_path, {"metric": metric})
        metric_body = {
            "schema": carrier_runner.SOLVER_EVIDENCE_SCHEMA,
            "channel": channel,
            "partition_role": "fold-fit",
            "cached_trace_evidence": True,
            "review_or_evaluation_evidence_allowed": False,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "metric_sha256": carrier_runner._tensor_sha256(metric),
            "numeric_contract_sha256": numeric_contract_sha256,
            "trace_cache_artifact_sha256": audit["trace_selective_session_sha256"],
            "fit_partition_manifest_sha256": manifest["manifest_sha256"],
        }
        _sealed_artifact(receipt_path, metric_body)
        metric_specs[channel] = {
            "tensors": _file_spec(tensor_path),
            "receipt": _file_spec(receipt_path),
        }
    request = {
        "schema": carrier_runner.REQUEST_SCHEMA,
        "run_id": f"a12-carrier-reclosure-{preparation_cache_key[:12]}",
        "phase_cache_directory": str((directory / "phase-cache").resolve()),
        "event_log": str((directory / "events.jsonl").resolve()),
        "output_directory": str((directory / "bridge").resolve()),
        "base_bridge": _file_spec(base_path),
        "parent_table": _file_spec(parent_path),
        "tokenizer_control": {
            "artifact_directory": str(tokenizer_directory),
            "expected_file_sha256": tokenizer_hashes,
            "manifest_files": [],
        },
        "carrier_evidence": {
            "tensors": _file_spec(evidence_tensor_path),
            "receipt": _file_spec(evidence_receipt_path),
        },
        "input_solver_evidence": metric_specs["input"],
        "output_solver_evidence": metric_specs["output"],
        "selection": {
            "promotion_count": promotion_count,
            "winner_weight": args.winner_weight,
            "allow_positive_incidence": True,
        },
        "solver": solver,
    }
    request_path = directory / "carrier-request.json"
    _atomic_write_json(request_path, request)
    completion_body = {
        "schema": PREPARATION_COMPLETION_SCHEMA,
        "preparation_cache_key": preparation_cache_key,
        "request_file": str(request_path),
        "request_file_sha256": _file_sha256(request_path),
        "promotion_count": promotion_count,
        "demanded_unprotected_non_direct_count": len(demanded_non_direct),
        "carrier_evidence_artifact_sha256": compiled_evidence.artifact_sha256,
        "base_semantic_binding_sha256": base_binding_sha256,
        "numeric_contract_sha256": numeric_contract_sha256,
        "manual_review_prompt_ids_sha256": manifest["manual_review_prompt_ids_sha256"],
        "review_prompts_consumed": 0,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
    }
    complete = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    _atomic_write_json(complete_path, complete)
    return {**complete, "cache_hit": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-audit-json", type=Path, required=True)
    parser.add_argument("--trajectory-completion-json", type=Path, required=True)
    parser.add_argument("--expected-trajectory-audit-file-sha256", required=True)
    parser.add_argument("--expected-trajectory-audit-sha256", required=True)
    parser.add_argument("--expected-trajectory-completion-sha256", required=True)
    parser.add_argument("--feature-tensors-safetensors", type=Path, required=True)
    parser.add_argument("--feature-receipt-json", type=Path, required=True)
    parser.add_argument("--expected-feature-tensor-file-sha256", required=True)
    parser.add_argument("--expected-feature-artifact-sha256", required=True)
    parser.add_argument("--fit-prompt-manifest-json", type=Path, required=True)
    parser.add_argument("--expected-fit-prompt-manifest-sha256", required=True)
    parser.add_argument("--expected-manual-review-prompt-ids-sha256", required=True)
    parser.add_argument("--support-publication-directory", type=Path, required=True)
    parser.add_argument("--expected-support-cache-key", required=True)
    parser.add_argument("--expected-support-result-sha256", required=True)
    parser.add_argument("--input-codebook-directory", type=Path, required=True)
    parser.add_argument("--expected-input-codebook-artifact-sha256", required=True)
    parser.add_argument("--parent-table-safetensors", type=Path, required=True)
    parser.add_argument("--parent-table-key", required=True)
    parser.add_argument("--expected-parent-table-file-sha256", required=True)
    parser.add_argument("--parent-identity-sha256", required=True)
    parser.add_argument("--target-table-safetensors", type=Path, required=True)
    parser.add_argument("--target-table-key", required=True)
    parser.add_argument("--expected-target-table-file-sha256", required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-json-sha256", required=True)
    parser.add_argument("--expected-tokenizer-config-sha256", required=True)
    parser.add_argument("--expected-chat-template-sha256", required=True)
    parser.add_argument("--maximum-promotion-count", type=int, default=65536)
    parser.add_argument("--winner-weight", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--selection-device", default="cuda:0")
    parser.add_argument(
        "--selection-dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
