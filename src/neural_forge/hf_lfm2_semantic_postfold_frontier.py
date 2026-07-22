"""Durable terminal-frontier banks and exact-W2 child materialization.

The expensive semantic body fold ends at an authenticated preterminal capsule
whose final MLP down projection is deliberately zero.  The fused terminal
compiler derives a bounded bank of exact-BF16 down-projection candidates in
memory.  This module gives that bank a strict, nonpromotable disk boundary and
materializes any retained candidate into a fresh standalone child without
rerunning the body fold.

Numerical terminal diagnostics are preserved as provenance.  They never
eliminate a retained candidate and they never authorize intelligence or
promotion.  Candidate quality is decided only after complete whole-response
evaluation.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_state,
    fingerprint_model_state_entries,
)
from neural_forge.hf_lfm2_cpcf_model_fold import independent_parameter_inventory
from neural_forge.hf_lfm2_circuit_trace_cache import open_lfm2_circuit_trace_cache
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    _atomic_rename_noreplace,
    _canonical_mapping,
    _deep_freeze,
    _deep_thaw,
    _file_record,
    _fsync_directory,
    _is_link_or_junction,
    _lexists,
    _load_safetensors_path,
    _normalize_tensors,
    _read_canonical_json_with_raw,
    _real_directory,
    _regular_file,
    _save_safetensors,
    _stat_identity,
    _write_canonical_json,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA,
    LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionResidualResult,
    _tensor_sha256 as _terminal_tensor_sha256,
)
from neural_forge.hf_lfm2_long_horizon_terminal_refold import (
    Lfm2LongHorizonTrajectorySplitPlan,
    load_lfm2_parent_final_rmsnorm_from_local_safetensors,
    preflight_lfm2_long_horizon_terminal_cache,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA,
    _ZERO_OPERATIONS as _SEMANTIC_ZERO_OPERATIONS,
    _authenticate_fused_terminal_cache_evidence,
    _authenticate_fused_terminal_tied_interface,
    _authenticate_terminal_frontier_export_context,
    _final_w2_parameter,
    _parameter_count,
    _tensor_sha256,
    compile_install_lfm2_fused_semantic_terminal_refold,
    load_lfm2_cpcf_v5_standalone_child,
    load_lfm2_mapped_parent_tied_table_from_local_safetensors,
    publish_lfm2_semantic_postfold_checkpoint,
)
from neural_forge.hf_lfm2_semantic_preterminal_capsule import (
    Lfm2SemanticPreterminalCapsule,
    Lfm2SemanticPreterminalCapsuleReference,
    _open_outer_capsule,
    load_lfm2_semantic_preterminal_capsule,
)
from neural_forge.hf_tokenizer_bridge import load_strict_tokenizer_bridge_artifact


LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-semantic-terminal-frontier-bank-manifest-v1"
)
LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-semantic-terminal-frontier-bank-receipt-v1"
)
LFM2_SEMANTIC_TERMINAL_FRONTIER_MATERIALIZATION_SCHEMA = (
    "neural-forge-lfm2-semantic-terminal-frontier-materialization-v1"
)
LFM2_SEMANTIC_TERMINAL_FRONTIER_CONTEXT_SCHEMA = (
    "neural-forge-lfm2-semantic-terminal-frontier-context-v1"
)
LFM2_SEMANTIC_TERMINAL_FRONTIER_SOURCE_ROOTS_SCHEMA = (
    "neural-forge-lfm2-semantic-terminal-frontier-source-roots-v1"
)

_BANK_ARTIFACT_KIND = "lfm2-semantic-terminal-frontier-bank"
_BANK_PAYLOAD_NAME = "terminal-frontier.safetensors"
_BANK_MANIFEST_NAME = "frontier-manifest.json"
_BANK_RECEIPT_NAME = "frontier-receipt.json"
_BANK_ROOT_INVENTORY = frozenset(
    {_BANK_PAYLOAD_NAME, _BANK_MANIFEST_NAME, _BANK_RECEIPT_NAME}
)
_MATERIALIZATION_RECEIPT_NAME = "materialization-receipt.json"
_MATERIALIZATION_CHECKPOINT_NAME = "checkpoint"
_MATERIALIZATION_ROOT_INVENTORY = frozenset(
    {_MATERIALIZATION_CHECKPOINT_NAME, _MATERIALIZATION_RECEIPT_NAME}
)
_FINAL_W2_NAME = "model.layers.13.feed_forward.w2.weight"
_NONPROMOTABLE_FLAGS = {
    "promotion_authorized": False,
    "accepted_artifact": False,
    "behavior_capability_proven": False,
    "intelligence_retention_proven": False,
}
_ZERO_OPERATION_AUDIT = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "model_generate_calls": 0,
    "trace_recaptures": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parameters_sparsified": 0,
    "network_access_performed": False,
}


class Lfm2SemanticTerminalFrontierError(RuntimeError):
    """A terminal frontier bank or materialization failed authentication."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticTerminalFrontierError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _sealed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    normalized = _canonical_mapping(
        _deep_thaw(body),
        "terminal frontier receipt body",
    )
    if field_name in normalized:
        raise Lfm2SemanticTerminalFrontierError(
            f"Terminal frontier body already contains {field_name!r}."
        )
    return {**normalized, field_name: canonical_sha256(normalized)}


def _verify_seal(
    value: Mapping[str, Any],
    *,
    field_name: str,
    label: str,
) -> dict[str, Any]:
    normalized = _canonical_mapping(_deep_thaw(value), label)
    body = dict(normalized)
    supplied = body.pop(field_name, None)
    if supplied != canonical_sha256(body):
        raise Lfm2SemanticTerminalFrontierError(f"{label} self-seal is invalid.")
    return normalized


def _absolute_new_output(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    output = Path(value).expanduser().absolute()
    if _lexists(output):
        raise Lfm2SemanticTerminalFrontierError(
            f"Refusing to replace existing {label} {output}."
        )
    parent = output.parent
    _real_directory(parent, name=f"{label} parent")
    if _is_link_or_junction(parent) or parent.resolve(strict=True) != parent:
        raise Lfm2SemanticTerminalFrontierError(
            f"{label.capitalize()} parent cannot be or traverse a link."
        )
    return output


def _safe_remove_stage(
    stage: Path,
    *,
    parent: Path,
    identity: tuple[int, int],
    prefix: str,
) -> None:
    if not _lexists(stage):
        return
    current = stage.lstat()
    if (
        _stat_identity(current) != identity
        or not stat.S_ISDIR(current.st_mode)
        or _is_link_or_junction(stage)
        or stage.resolve(strict=True).parent != parent.resolve(strict=True)
        or not stage.name.startswith(prefix)
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Refusing to remove unauthenticated terminal-frontier staging."
        )
    shutil.rmtree(stage)
    _fsync_directory(parent)


def _root_with_exact_inventory(
    value: str | os.PathLike[str],
    *,
    inventory: frozenset[str],
    directory_names: frozenset[str],
    label: str,
) -> tuple[Path, tuple[int, int]]:
    lexical = Path(value).expanduser().absolute()
    if _is_link_or_junction(lexical):
        raise Lfm2SemanticTerminalFrontierError(f"{label} root is a link.")
    try:
        root = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticTerminalFrontierError(f"{label} root is missing.") from exc
    if root != lexical:
        raise Lfm2SemanticTerminalFrontierError(
            f"{label} root path cannot traverse a link."
        )
    root_stat = _real_directory(root, name=f"{label} root")
    entries = tuple(os.scandir(root))
    if {entry.name for entry in entries} != set(inventory):
        raise Lfm2SemanticTerminalFrontierError(
            f"{label} has missing or unexpected root entries."
        )
    for entry in entries:
        path = root / entry.name
        if entry.name in directory_names:
            _real_directory(path, name=f"{label} {entry.name}")
        else:
            _regular_file(path, name=f"{label} {entry.name}")
    return root, _stat_identity(root_stat)


def _recheck_root(
    root: Path,
    identity: tuple[int, int],
    *,
    inventory: frozenset[str],
    directory_names: frozenset[str],
    label: str,
) -> None:
    current, current_identity = _root_with_exact_inventory(
        root,
        inventory=inventory,
        directory_names=directory_names,
        label=label,
    )
    if current != root or current_identity != identity:
        raise Lfm2SemanticTerminalFrontierError(
            f"{label} root changed during authentication."
        )


def _terminal_report(result: Lfm2CpcfTerminalActionResidualResult) -> dict[str, Any]:
    if (
        not isinstance(result, Lfm2CpcfTerminalActionResidualResult)
        or not result.verify_authentication()
        or getattr(result, "_compiler_issued", False) is not True
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier export requires an authenticated compiler-issued result."
        )
    report = _verify_seal(
        result.report,
        field_name="result_sha256",
        label="terminal action result",
    )
    if report.get("schema") != LFM2_CPCF_TERMINAL_ACTION_RESULT_SCHEMA:
        raise Lfm2SemanticTerminalFrontierError("Terminal result has the wrong schema.")
    return report


def _frontier_records(
    report: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    frontier = report.get("frontier")
    if (
        not isinstance(frontier, list)
        or len(frontier) < 2
        or report.get("frontier_count") != len(frontier)
        or report.get("expected_frontier_count") != len(frontier)
        or not all(isinstance(record, Mapping) for record in frontier)
        or frontier[0].get("candidate_id") != "incumbent"
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal result has no complete retained frontier."
        )
    records = {
        str(record.get("candidate_id")): _canonical_mapping(
            record,
            "terminal frontier record",
        )
        for record in frontier[1:]
    }
    ids = tuple(str(record.get("candidate_id")) for record in frontier[1:])
    if (
        any(not candidate_id or candidate_id == "incumbent" for candidate_id in ids)
        or len(set(ids)) != len(ids)
        or set(records) != set(ids)
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier candidate identities are invalid."
        )
    for candidate_id in ids:
        _digest(records[candidate_id].get("state_sha256"), "candidate state SHA-256")
    return ids, records


def _authenticated_export_context(
    value: Mapping[str, Any],
    *,
    terminal_result: Lfm2CpcfTerminalActionResidualResult,
    capsule_reference: Lfm2SemanticPreterminalCapsuleReference,
    capsule_manifest: Mapping[str, Any],
    terminal_report: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        context = dict(
            _authenticate_terminal_frontier_export_context(
                terminal_result,
                value,
            )
        )
    except Exception as exc:
        raise Lfm2SemanticTerminalFrontierError(
            "Fused terminal export context failed authentication."
        ) from exc
    context = _canonical_mapping(
        _deep_thaw(context),
        "terminal frontier export context",
    )
    bindings = capsule_manifest.get("bindings")
    model = capsule_manifest.get("model")
    layer_receipt = capsule_manifest.get("semantic_layer_stack_receipt")
    interface_authentication = capsule_manifest.get("active_interface_authentication")
    if (
        context.get("schema") != LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA
        or context.get("terminal_result_sha256") != terminal_report.get("result_sha256")
        or context.get("terminal_seed_sha256") != terminal_report.get("seed_sha256")
        or context.get("terminal_incumbent_down_weight_terminal_action_sha256")
        != terminal_report.get("incumbent_seed_writer_sha256")
        or not isinstance(bindings, Mapping)
        or context.get("cache_key") != bindings.get("semantic_cache_key")
        or context.get("cache_parent_fingerprint")
        != bindings.get("cache_parent_fingerprint")
        or context.get("bridge_artifact_sha256")
        != bindings.get("bridge_artifact_sha256")
        or context.get("parent_final_norm_receipt_sha256")
        != bindings.get("parent_final_norm_receipt_sha256")
        or not isinstance(model, Mapping)
        or context.get("terminal_incumbent_down_weight_terminal_action_sha256")
        != model.get("final_w2_terminal_action_sha256")
        or not isinstance(layer_receipt, Mapping)
        or context.get("semantic_layer_stack_result_sha256")
        != layer_receipt.get("result_sha256")
        or context.get("layer_stack_cache_session_sha256")
        != layer_receipt.get("cache_selective_session_sha256")
        or context.get("terminal_cache_session_sha256")
        != context.get("layer_stack_cache_session_sha256")
        or not isinstance(interface_authentication, Mapping)
        or context.get("mapped_parent_tied_table_interface_report_sha256")
        != interface_authentication.get("interface_report_sha256")
        or context.get("cache_content_identity_matches_layer_stack") is not True
        or type(context.get("preterminal_resume_mode")) is not bool
        or type(context.get("same_live_cache_session_as_layer_stack")) is not bool
        or (
            context.get("preterminal_resume_mode") is True
            and context.get("same_live_cache_session_as_layer_stack") is not False
        )
        or (
            context.get("preterminal_resume_mode") is False
            and context.get("same_live_cache_session_as_layer_stack") is not True
        )
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Fused terminal export context is not bound to this capsule and solve."
        )
    for name in (
        "terminal_cache_compatibility_sha256",
        "terminal_cache_payload_binding_sha256",
        "terminal_cache_session_sha256",
        "mapped_parent_tied_table_sha256",
        "tied_interface_authentication_sha256",
        "context_sha256",
    ):
        _digest(context.get(name), name)
    return context


def _authenticate_exact_zero_capsule_frontier(
    *,
    capsule_manifest: Mapping[str, Any],
    terminal_report: Mapping[str, Any],
    candidate_weights: Mapping[str, Any],
) -> None:
    model = capsule_manifest.get("model")
    output_hashes = terminal_report.get("output_state_sha256")
    split_roles = terminal_report.get("split_roles")
    validation = terminal_report.get("validation")
    frontier = terminal_report.get("frontier")
    if not isinstance(model, Mapping) or not isinstance(output_hashes, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Capsule or terminal result omitted exact-zero state bindings."
        )
    expected_shape = model.get("final_w2_shape")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise RuntimeError("PyTorch is required for frontier banks.") from exc
    if (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in expected_shape
        )
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal capsule final-W2 shape is invalid."
        )
    terminal_zero_sha = _terminal_tensor_sha256(
        torch.zeros(tuple(expected_shape), dtype=torch.bfloat16),
        "capsule.exact_zero_final_w2",
    )
    semantic_zero_sha = _tensor_sha256(
        torch.zeros(tuple(expected_shape), dtype=torch.bfloat16)
    )
    if (
        model.get("final_w2_exact_zero") is not True
        or model.get("final_w2_nonzero_element_count") != 0
        or model.get("final_w2_dtype") != "bfloat16"
        or model.get("final_w2_semantic_sha256") != semantic_zero_sha
        or model.get("final_w2_terminal_action_sha256") != terminal_zero_sha
        or terminal_report.get("exact_zero_cold_start") is not True
        or terminal_report.get("incumbent_all_elements_zero") is not True
        or terminal_report.get("incumbent_nonzero_element_count") != 0
        or terminal_report.get("incumbent_seed_writer_sha256") != terminal_zero_sha
        or output_hashes.get("incumbent") != terminal_zero_sha
        or terminal_report.get("structural_nonveto_schema")
        != LFM2_CPCF_TERMINAL_ACTION_STRUCTURAL_NONVETO_SCHEMA
        or terminal_report.get("candidate_eligibility_basis")
        != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
        or terminal_report.get("structural_safety_is_only_candidate_veto") is not True
        or terminal_report.get("numeric_metrics_used_for_candidate_ordering")
        is not True
        or terminal_report.get("numeric_metrics_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("numeric_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("numeric_thresholds_used_for_publication") is not False
        or terminal_report.get("weight_delta_threshold_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("fit_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("fit_check_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("selection_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("validation_thresholds_used_for_publication")
        is not False
        or terminal_report.get("validation_publication_veto_applied") is not False
        or terminal_report.get("rollback_authorized") is not False
        or terminal_report.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or terminal_report.get(
            "behavioral_intelligence_review_outside_terminal_compiler"
        )
        is not True
        or terminal_report.get(
            "published_writer_is_structurally_safe_effectful_nonincumbent"
        )
        is not True
        or terminal_report.get("selected_nonincumbent") is not True
        or terminal_report.get("accepted") is not True
        or terminal_report.get("rollback_performed") is not False
        or terminal_report.get("published_is_incumbent") is not False
        or terminal_report.get("fit_metrics_used_for_structural_eligibility")
        is not False
        or terminal_report.get("fit_metric_thresholds_used_for_candidate_eligibility")
        is not False
        or terminal_report.get("fit_metric_thresholds_used_for_publication")
        is not False
        or terminal_report.get("fit_metrics_have_final_intelligence_authority")
        is not False
        or not isinstance(split_roles, Mapping)
        or split_roles.get("candidate_eligibility_uses_only_finite_dense_structure")
        is not True
        or split_roles.get("candidate_eligibility_uses_only_structural_safety")
        is not True
        or split_roles.get("fit_used_for_candidate_eligibility") is not False
        or not isinstance(validation, Mapping)
        or validation.get("heldout_diagnostic_only") is not True
        or validation.get("publication_gate_applied") is not False
        or validation.get("used_only_for_frozen_candidate_acceptance") is not False
        or not isinstance(frontier, list)
        or not frontier
        or not isinstance(frontier[0], Mapping)
        or frontier[0].get("candidate_id") != "incumbent"
        or frontier[0].get("eligible") is not False
        or frontier[0].get("structurally_eligible") is not False
        or frontier[0].get("effectful_nonincumbent") is not False
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal bank export is restricted to the capsule's exact-zero, "
            "structural-nonveto cold-start frontier."
        )
    records = {
        str(record.get("candidate_id")): record
        for record in frontier[1:]
        if isinstance(record, Mapping)
    }
    structurally_eligible_count = 0
    for candidate_id, tensor in candidate_weights.items():
        record = records.get(candidate_id)
        coverage_value = record.get("structural_coverage") if record else None
        if not isinstance(record, Mapping) or not isinstance(coverage_value, Mapping):
            raise Lfm2SemanticTerminalFrontierError(
                f"Terminal frontier writer {candidate_id!r} omitted structural proof."
            )
        coverage = dict(coverage_value)
        coverage_sha256 = coverage.pop("coverage_sha256", None)
        actual_nonzero = tensor.ne(0)
        actual_finite_count = int(torch.isfinite(tensor).sum().item())
        actual_nonzero_count = int(actual_nonzero.sum().item())
        actual_output_coverage = int(actual_nonzero.any(dim=1).sum().item())
        actual_source_coverage = int(actual_nonzero.any(dim=0).sum().item())
        expected_storage_bytes = int(tensor.numel() * tensor.element_size())
        exact_private_storage = bool(
            tensor.device.type == "cpu"
            and tensor.layout == torch.strided
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and int(tensor.storage_offset()) == 0
            and int(tensor.untyped_storage().nbytes()) == expected_storage_bytes
            and tensor.requires_grad is False
            and tensor.grad_fn is None
        )
        coverage_complete = bool(
            [int(size) for size in tensor.shape] == expected_shape
            and actual_finite_count == int(tensor.numel())
            and actual_nonzero_count > 0
            and actual_output_coverage == expected_shape[0]
            and actual_source_coverage == expected_shape[1]
            and exact_private_storage
        )
        effectful_nonincumbent = record.get("state_sha256") != terminal_zero_sha
        structurally_eligible = bool(coverage_complete and effectful_nonincumbent)
        structurally_eligible_count += int(structurally_eligible)
        if (
            coverage_sha256 != canonical_sha256(coverage)
            or record.get("structural_coverage_sha256") != coverage_sha256
            or coverage_value.get("schema")
            != "neural-forge-lfm2-terminal-w2-cold-start-coverage-v1"
            or coverage_value.get("tensor_sha256") != record.get("state_sha256")
            or coverage_value.get("shape") != expected_shape
            or coverage_value.get("dtype") != "bfloat16"
            or coverage_value.get("element_count") != int(tensor.numel())
            or coverage_value.get("contiguous_cpu_storage")
            is not (tensor.device.type == "cpu" and tensor.is_contiguous())
            or coverage_value.get("storage_offset") != int(tensor.storage_offset())
            or coverage_value.get("storage_bytes")
            != int(tensor.untyped_storage().nbytes())
            or coverage_value.get("expected_storage_bytes") != expected_storage_bytes
            or coverage_value.get("detached_without_grad")
            is not (tensor.requires_grad is False and tensor.grad_fn is None)
            or coverage_value.get("owns_exact_storage_allocation")
            is not exact_private_storage
            or coverage_value.get("finite_element_count") != actual_finite_count
            or coverage_value.get("nonzero_element_count") != actual_nonzero_count
            or coverage_value.get("all_elements_finite")
            is not (actual_finite_count == int(tensor.numel()))
            or coverage_value.get("all_elements_nonzero")
            is not (actual_nonzero_count == int(tensor.numel()))
            or coverage_value.get("target_output_axis_size") != expected_shape[0]
            or coverage_value.get("covered_target_output_count")
            != actual_output_coverage
            or coverage_value.get("all_target_outputs_covered")
            is not (actual_output_coverage == expected_shape[0])
            or coverage_value.get("source_feature_axis_size") != expected_shape[1]
            or coverage_value.get("covered_source_feature_count")
            != actual_source_coverage
            or coverage_value.get("all_source_features_covered")
            is not (actual_source_coverage == expected_shape[1])
            or coverage_value.get("complete_dense_writer_coverage")
            is not coverage_complete
            or coverage_value.get("complete_dense_writer_structural_safety")
            is not coverage_complete
            or coverage_value.get("numeric_parent_child_weight_equality_required")
            is not False
            or record.get("effectful_nonincumbent") is not effectful_nonincumbent
            or record.get("structurally_eligible") is not structurally_eligible
            or record.get("eligible") is not structurally_eligible
            or record.get("structural_safety_passed") is not coverage_complete
            or record.get("weight_norm_gate_passed") is not coverage_complete
            or record.get("candidate_eligibility_basis")
            != "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
            or record.get("weight_delta_threshold_used_for_eligibility") is not False
            or record.get("fit_thresholds_used_for_eligibility") is not False
            or record.get("fit_check_thresholds_used_for_eligibility") is not False
            or record.get("selection_thresholds_used_for_eligibility") is not False
            or record.get("numeric_thresholds_used_for_publication") is not False
            or record.get("numeric_metrics_used_for_candidate_ordering") is not True
            or record.get("numeric_metrics_have_final_intelligence_authority")
            is not False
        ):
            raise Lfm2SemanticTerminalFrontierError(
                f"Terminal frontier writer {candidate_id!r} has invalid structural "
                "nonveto provenance."
            )
    selected_id = terminal_report.get("selected_candidate_id")
    selected_record = records.get(str(selected_id))
    if (
        len(records) != len(candidate_weights)
        or terminal_report.get("structurally_eligible_frontier_count")
        != structurally_eligible_count
        or not isinstance(selected_record, Mapping)
        or selected_record.get("eligible") is not True
        or selected_record.get("effectful_nonincumbent") is not True
        or selected_record.get("structural_safety_passed") is not True
        or output_hashes.get("selected") != selected_record.get("state_sha256")
        or output_hashes.get("published") != selected_record.get("state_sha256")
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier selected writer is not the authenticated structural "
            "nonveto publication."
        )


def _canonical_existing_source_directory(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    lexical = Path(value).expanduser().absolute()
    if _is_link_or_junction(lexical):
        raise Lfm2SemanticTerminalFrontierError(f"{label} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticTerminalFrontierError(f"{label} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2SemanticTerminalFrontierError(
            f"{label} must be one canonical real directory."
        )
    return resolved


@dataclass(frozen=True, slots=True)
class Lfm2SemanticTerminalFrontierSourceRoots:
    """Complete role-labeled local source roots for one frontier solve."""

    preterminal_capsule_directory: Path
    semantic_cache_directory: Path
    tokenizer_bridge_directory: Path
    parent_model_directory: Path
    donor_tokenizer_directories: tuple[Path, ...]
    target_tokenizer_directories: tuple[Path, ...]

    def __post_init__(self) -> None:
        for name in (
            "preterminal_capsule_directory",
            "semantic_cache_directory",
            "tokenizer_bridge_directory",
            "parent_model_directory",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_existing_source_directory(
                    getattr(self, name),
                    label=name.replace("_", " "),
                ),
            )
        for name in (
            "donor_tokenizer_directories",
            "target_tokenizer_directories",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, tuple) or not raw:
                raise Lfm2SemanticTerminalFrontierError(
                    f"{name} must be a nonempty tuple of canonical directories."
                )
            normalized = tuple(
                _canonical_existing_source_directory(
                    value,
                    label=name.replace("_", " "),
                )
                for value in raw
            )
            if len(set(normalized)) != len(normalized):
                raise Lfm2SemanticTerminalFrontierError(
                    f"{name} contains duplicate roots."
                )
            object.__setattr__(self, name, tuple(sorted(normalized, key=str)))

    def authentication_body(self) -> dict[str, Any]:
        role_paths = {
            "preterminal_capsule": [str(self.preterminal_capsule_directory)],
            "semantic_cache": [str(self.semantic_cache_directory)],
            "tokenizer_bridge": [str(self.tokenizer_bridge_directory)],
            "parent_model": [str(self.parent_model_directory)],
            "donor_tokenizer": [
                str(value) for value in self.donor_tokenizer_directories
            ],
            "target_tokenizer": [
                str(value) for value in self.target_tokenizer_directories
            ],
        }
        return {
            "schema": LFM2_SEMANTIC_TERMINAL_FRONTIER_SOURCE_ROOTS_SCHEMA,
            "role_paths": role_paths,
            "required_roles": list(role_paths),
            "role_count": len(role_paths),
            "canonical_root_count": sum(len(value) for value in role_paths.values()),
            "all_roles_explicit_and_nonempty": True,
        }

    @property
    def source_roots_sha256(self) -> str:
        return canonical_sha256(self.authentication_body())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "source_roots_sha256": self.source_roots_sha256,
        }

    def all_paths(self) -> tuple[Path, ...]:
        body = self.authentication_body()["role_paths"]
        return tuple(Path(value) for role in body for value in body[role])


def _authenticate_source_roots_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    roots = _canonical_mapping(_deep_thaw(value), "terminal frontier source roots")
    body = dict(roots)
    supplied_sha = body.pop("source_roots_sha256", None)
    role_paths = body.get("role_paths")
    required_roles = (
        "preterminal_capsule",
        "semantic_cache",
        "tokenizer_bridge",
        "parent_model",
        "donor_tokenizer",
        "target_tokenizer",
    )
    if (
        roots.get("schema") != LFM2_SEMANTIC_TERMINAL_FRONTIER_SOURCE_ROOTS_SCHEMA
        or supplied_sha != canonical_sha256(body)
        or not isinstance(role_paths, Mapping)
        or set(role_paths) != set(required_roles)
        or body.get("required_roles") != list(required_roles)
        or body.get("role_count") != len(required_roles)
        or body.get("all_roles_explicit_and_nonempty") is not True
        or any(
            not isinstance(role_paths.get(role), list)
            or not role_paths[role]
            or any(
                not isinstance(path, str)
                or not path
                or not Path(path).is_absolute()
                or Path(path).absolute() != Path(path)
                for path in role_paths[role]
            )
            or len(set(role_paths[role])) != len(role_paths[role])
            for role in required_roles
        )
        or body.get("canonical_root_count")
        != sum(len(role_paths[role]) for role in required_roles)
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier source-root inventory is invalid."
        )
    return roots


def _source_root_paths(value: Mapping[str, Any]) -> tuple[Path, ...]:
    roots = _authenticate_source_roots_mapping(value)
    return tuple(
        Path(path)
        for role in roots["required_roles"]
        for path in roots["role_paths"][role]
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticTerminalFrontierBankReference:
    path: Path
    receipt_sha256: str
    receipt_file_sha256: str
    manifest_sha256: str
    preterminal_capsule_receipt_sha256: str
    terminal_result_sha256: str
    source_roots_sha256: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "receipt_sha256",
            "receipt_file_sha256",
            "manifest_sha256",
            "preterminal_capsule_receipt_sha256",
            "terminal_result_sha256",
            "source_roots_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            not self.candidate_ids
            or len(set(self.candidate_ids)) != len(self.candidate_ids)
            or any(
                not isinstance(value, str) or not value for value in self.candidate_ids
            )
        ):
            raise Lfm2SemanticTerminalFrontierError(
                "Frontier bank reference candidate IDs are invalid."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "preterminal_capsule_receipt_sha256": (
                self.preterminal_capsule_receipt_sha256
            ),
            "terminal_result_sha256": self.terminal_result_sha256,
            "source_roots_sha256": self.source_roots_sha256,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticTerminalFrontierBank:
    path: Path
    candidate_weights: Mapping[str, Any] = field(repr=False, compare=False)
    terminal_result_report: Mapping[str, Any]
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    reference: Lfm2SemanticTerminalFrontierBankReference


def publish_lfm2_semantic_terminal_frontier_bank(
    result: Lfm2CpcfTerminalActionResidualResult,
    export_context: Mapping[str, Any],
    *,
    preterminal_capsule_reference: Lfm2SemanticPreterminalCapsuleReference,
    source_roots: Lfm2SemanticTerminalFrontierSourceRoots,
    output_directory: str | os.PathLike[str],
) -> Lfm2SemanticTerminalFrontierBankReference:
    """Atomically export every authenticated terminal writer before installation."""

    if not isinstance(
        preterminal_capsule_reference,
        Lfm2SemanticPreterminalCapsuleReference,
    ):
        raise TypeError(
            "preterminal_capsule_reference must be an authenticated capsule reference."
        )
    if not isinstance(source_roots, Lfm2SemanticTerminalFrontierSourceRoots):
        raise TypeError("source_roots must be complete role-labeled source roots.")
    if source_roots.preterminal_capsule_directory != preterminal_capsule_reference.path:
        raise Lfm2SemanticTerminalFrontierError(
            "Source roots name a different preterminal capsule."
        )
    output = _absolute_new_output(output_directory, label="terminal frontier bank")
    _require_frontier_output_disjoint(
        output,
        *source_roots.all_paths(),
    )
    opened_capsule = _open_outer_capsule(
        preterminal_capsule_reference.path,
        expected_receipt_file_sha256=(
            preterminal_capsule_reference.receipt_file_sha256
        ),
        expected_receipt_sha256=preterminal_capsule_reference.receipt_sha256,
        expected_bindings_sha256=preterminal_capsule_reference.bindings_sha256,
        load_payload=False,
    )
    report = _terminal_report(result)
    candidate_ids, records = _frontier_records(report)
    if tuple(result.frontier_down_weights) != candidate_ids:
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal result tensor bank order differs from its report."
        )
    payload, entries = _normalize_tensors(
        result.frontier_down_weights,
        name="terminal frontier writer bank",
        clone=True,
    )
    if set(payload) != set(candidate_ids):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier tensor bank is incomplete."
        )
    for candidate_id in candidate_ids:
        if (
            _terminal_tensor_sha256(payload[candidate_id])
            != records[candidate_id]["state_sha256"]
        ):
            raise Lfm2SemanticTerminalFrontierError(
                f"Terminal candidate {candidate_id!r} differs from its result report."
            )
    capsule_manifest = opened_capsule["manifest"]
    _authenticate_exact_zero_capsule_frontier(
        capsule_manifest=capsule_manifest,
        terminal_report=report,
        candidate_weights=payload,
    )
    authenticated_export_context = _authenticated_export_context(
        export_context,
        terminal_result=result,
        capsule_reference=preterminal_capsule_reference,
        capsule_manifest=capsule_manifest,
        terminal_report=report,
    )
    compiler_binding = getattr(result, "_compiler_input_binding", None)
    if not isinstance(compiler_binding, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Compiler-issued terminal result omitted its private input binding."
        )
    compiler_binding_sha = _digest(
        compiler_binding.get("binding_sha256"),
        "terminal compiler input binding SHA-256",
    )
    authenticated_source_roots = source_roots.as_dict()
    stage = Path(
        tempfile.mkdtemp(prefix=".semantic-frontier-bank-", dir=str(output.parent))
    ).absolute()
    stage_identity = _stat_identity(_real_directory(stage, name="frontier bank stage"))
    moved = False
    try:
        payload_path = stage / _BANK_PAYLOAD_NAME
        _save_safetensors(
            payload_path,
            payload,
            artifact_kind=_BANK_ARTIFACT_KIND,
            file_role="terminal-w2-frontier",
        )
        payload_record = _file_record(payload_path, tensor_entries=entries)
        manifest_body = {
            "schema": LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_MANIFEST_SCHEMA,
            "artifact_kind": _BANK_ARTIFACT_KIND,
            "status": "quarantined-terminal-frontier-for-whole-response-review",
            **_NONPROMOTABLE_FLAGS,
            "preterminal_capsule": preterminal_capsule_reference.as_dict(),
            "authenticated_source_roots": authenticated_source_roots,
            "authenticated_source_roots_sha256": source_roots.source_roots_sha256,
            "terminal_frontier_export_context": authenticated_export_context,
            "terminal_frontier_export_context_sha256": authenticated_export_context[
                "context_sha256"
            ],
            "terminal_result_report": report,
            "terminal_result_sha256": result.result_sha256,
            "terminal_policy_sha256": result.policy.authentication_sha256,
            "terminal_compiler_input_binding_sha256": compiler_binding_sha,
            "frontier_payload": payload_record,
            "candidate_ids": list(candidate_ids),
            "candidate_state_sha256": {
                candidate_id: records[candidate_id]["state_sha256"]
                for candidate_id in candidate_ids
            },
            "candidate_count": len(candidate_ids),
            "incumbent_exported": False,
            "exact_zero_cold_start": True,
            "incumbent_matches_preterminal_capsule_final_w2": True,
            "all_retained_candidates_exported": True,
            "candidate_metric_threshold_used_for_export": False,
            "candidate_metric_threshold_used_for_eligibility": False,
            "numeric_metrics_preserved_as_diagnostics": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "holistic_whole_response_review_required": True,
            "exported_before_any_frontier_writer_installation": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        manifest = _sealed(manifest_body, "manifest_sha256")
        manifest_path = stage / _BANK_MANIFEST_NAME
        manifest_raw = _write_canonical_json(manifest_path, manifest)
        receipt_body = {
            "schema": LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_RECEIPT_SCHEMA,
            "artifact_kind": _BANK_ARTIFACT_KIND,
            "status": "quarantined-terminal-frontier-for-whole-response-review",
            **_NONPROMOTABLE_FLAGS,
            "manifest_file": {
                "path": _BANK_MANIFEST_NAME,
                "size_bytes": len(manifest_raw),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            },
            "manifest_sha256": manifest["manifest_sha256"],
            "frontier_payload_file_sha256": payload_record["sha256"],
            "preterminal_capsule_receipt_sha256": (
                preterminal_capsule_reference.receipt_sha256
            ),
            "preterminal_capsule_receipt_file_sha256": (
                preterminal_capsule_reference.receipt_file_sha256
            ),
            "preterminal_capsule_bindings_sha256": (
                preterminal_capsule_reference.bindings_sha256
            ),
            "authenticated_source_roots": authenticated_source_roots,
            "authenticated_source_roots_sha256": source_roots.source_roots_sha256,
            "terminal_frontier_export_context_sha256": authenticated_export_context[
                "context_sha256"
            ],
            "terminal_result_sha256": result.result_sha256,
            "terminal_policy_sha256": result.policy.authentication_sha256,
            "terminal_compiler_input_binding_sha256": compiler_binding_sha,
            "candidate_ids": list(candidate_ids),
            "candidate_count": len(candidate_ids),
            "all_retained_candidates_exported": True,
            "exact_zero_cold_start": True,
            "incumbent_matches_preterminal_capsule_final_w2": True,
            "candidate_metric_threshold_used_for_export": False,
            "numeric_metrics_have_final_intelligence_authority": False,
            "holistic_whole_response_review_required": True,
            "exported_before_any_frontier_writer_installation": True,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        receipt = _sealed(receipt_body, "receipt_sha256")
        receipt_path = stage / _BANK_RECEIPT_NAME
        receipt_raw = _write_canonical_json(receipt_path, receipt)
        _fsync_directory(stage)
        expected_file_sha = hashlib.sha256(receipt_raw).hexdigest()
        _open_terminal_frontier_bank(
            stage,
            expected_receipt_file_sha256=expected_file_sha,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_preterminal_capsule_receipt_sha256=(
                preterminal_capsule_reference.receipt_sha256
            ),
            expected_terminal_result_sha256=result.result_sha256,
            load_payload=False,
        )
        _atomic_rename_noreplace(stage, output)
        moved = True
        _fsync_directory(output.parent)
        opened = _open_terminal_frontier_bank(
            output,
            expected_receipt_file_sha256=expected_file_sha,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_preterminal_capsule_receipt_sha256=(
                preterminal_capsule_reference.receipt_sha256
            ),
            expected_terminal_result_sha256=result.result_sha256,
            load_payload=False,
        )
        return Lfm2SemanticTerminalFrontierBankReference(
            path=output,
            receipt_sha256=receipt["receipt_sha256"],
            receipt_file_sha256=opened["receipt_file_sha256"],
            manifest_sha256=manifest["manifest_sha256"],
            preterminal_capsule_receipt_sha256=(
                preterminal_capsule_reference.receipt_sha256
            ),
            terminal_result_sha256=result.result_sha256,
            source_roots_sha256=source_roots.source_roots_sha256,
            candidate_ids=candidate_ids,
        )
    except BaseException:
        if not moved and _lexists(stage):
            _safe_remove_stage(
                stage,
                parent=output.parent,
                identity=stage_identity,
                prefix=".semantic-frontier-bank-",
            )
        raise


def _open_terminal_frontier_bank(
    directory: str | os.PathLike[str],
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_preterminal_capsule_receipt_sha256: str,
    expected_terminal_result_sha256: str,
    load_payload: bool,
) -> dict[str, Any]:
    expected_file_sha = _digest(
        expected_receipt_file_sha256,
        "expected frontier receipt file SHA-256",
    )
    expected_receipt_sha = _digest(
        expected_receipt_sha256,
        "expected frontier receipt SHA-256",
    )
    expected_capsule_sha = _digest(
        expected_preterminal_capsule_receipt_sha256,
        "expected preterminal capsule receipt SHA-256",
    )
    expected_result_sha = _digest(
        expected_terminal_result_sha256,
        "expected terminal result SHA-256",
    )
    root, root_identity = _root_with_exact_inventory(
        directory,
        inventory=_BANK_ROOT_INVENTORY,
        directory_names=frozenset(),
        label="terminal frontier bank",
    )
    receipt, receipt_raw = _read_canonical_json_with_raw(
        root / _BANK_RECEIPT_NAME,
        name=_BANK_RECEIPT_NAME,
    )
    receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt = _verify_seal(
        receipt,
        field_name="receipt_sha256",
        label="terminal frontier bank receipt",
    )
    receipt_source_roots_value = receipt.get("authenticated_source_roots")
    if not isinstance(receipt_source_roots_value, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier receipt omitted its source-root inventory."
        )
    receipt_source_roots = _authenticate_source_roots_mapping(
        receipt_source_roots_value
    )
    if (
        receipt_file_sha != expected_file_sha
        or receipt.get("receipt_sha256") != expected_receipt_sha
        or receipt.get("schema") != LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != _BANK_ARTIFACT_KIND
        or receipt.get("preterminal_capsule_receipt_sha256") != expected_capsule_sha
        or receipt.get("terminal_result_sha256") != expected_result_sha
        or receipt.get("authenticated_source_roots_sha256")
        != receipt_source_roots.get("source_roots_sha256")
        or any(
            receipt.get(name) != expected
            for name, expected in _NONPROMOTABLE_FLAGS.items()
        )
        or receipt.get("all_retained_candidates_exported") is not True
        or receipt.get("exact_zero_cold_start") is not True
        or receipt.get("incumbent_matches_preterminal_capsule_final_w2") is not True
        or receipt.get("candidate_metric_threshold_used_for_export") is not False
        or receipt.get("numeric_metrics_have_final_intelligence_authority") is not False
        or receipt.get("holistic_whole_response_review_required") is not True
        or receipt.get("exported_before_any_frontier_writer_installation") is not True
        or receipt.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier bank receipt differs from its external pins."
        )
    manifest_record = receipt.get("manifest_file")
    if not isinstance(manifest_record, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier receipt omitted its manifest record."
        )
    manifest, manifest_raw = _read_canonical_json_with_raw(
        root / _BANK_MANIFEST_NAME,
        name=_BANK_MANIFEST_NAME,
    )
    if (
        manifest_record.get("path") != _BANK_MANIFEST_NAME
        or manifest_record.get("size_bytes") != len(manifest_raw)
        or manifest_record.get("sha256") != hashlib.sha256(manifest_raw).hexdigest()
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier manifest differs from its receipt."
        )
    manifest = _verify_seal(
        manifest,
        field_name="manifest_sha256",
        label="terminal frontier bank manifest",
    )
    report = _verify_seal(
        manifest.get("terminal_result_report"),
        field_name="result_sha256",
        label="persisted terminal action result",
    )
    candidate_ids, records = _frontier_records(report)
    export_context = manifest.get("terminal_frontier_export_context")
    manifest_source_roots_value = manifest.get("authenticated_source_roots")
    if not isinstance(manifest_source_roots_value, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier manifest omitted its source-root inventory."
        )
    manifest_source_roots = _authenticate_source_roots_mapping(
        manifest_source_roots_value
    )
    if (
        manifest.get("schema") != LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_MANIFEST_SCHEMA
        or manifest.get("artifact_kind") != _BANK_ARTIFACT_KIND
        or manifest.get("manifest_sha256") != receipt.get("manifest_sha256")
        or manifest.get("terminal_result_sha256") != expected_result_sha
        or report.get("result_sha256") != expected_result_sha
        or manifest_source_roots != receipt_source_roots
        or manifest.get("authenticated_source_roots_sha256")
        != manifest_source_roots.get("source_roots_sha256")
        or manifest.get("candidate_ids") != list(candidate_ids)
        or manifest.get("candidate_count") != len(candidate_ids)
        or receipt.get("candidate_ids") != list(candidate_ids)
        or receipt.get("candidate_count") != len(candidate_ids)
        or manifest.get("candidate_state_sha256")
        != {
            candidate_id: records[candidate_id]["state_sha256"]
            for candidate_id in candidate_ids
        }
        or any(
            manifest.get(name) != expected
            for name, expected in _NONPROMOTABLE_FLAGS.items()
        )
        or manifest.get("incumbent_exported") is not False
        or manifest.get("exact_zero_cold_start") is not True
        or manifest.get("incumbent_matches_preterminal_capsule_final_w2") is not True
        or manifest.get("all_retained_candidates_exported") is not True
        or manifest.get("candidate_metric_threshold_used_for_export") is not False
        or manifest.get("candidate_metric_threshold_used_for_eligibility") is not False
        or manifest.get("numeric_metrics_preserved_as_diagnostics") is not True
        or manifest.get("numeric_metrics_have_final_intelligence_authority")
        is not False
        or manifest.get("holistic_whole_response_review_required") is not True
        or manifest.get("exported_before_any_frontier_writer_installation") is not True
        or manifest.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier bank manifest contract is invalid."
        )
    capsule_reference = manifest.get("preterminal_capsule")
    if (
        not isinstance(capsule_reference, Mapping)
        or capsule_reference.get("receipt_sha256") != expected_capsule_sha
        or capsule_reference.get("receipt_file_sha256")
        != receipt.get("preterminal_capsule_receipt_file_sha256")
        or capsule_reference.get("bindings_sha256")
        != receipt.get("preterminal_capsule_bindings_sha256")
        or manifest_source_roots["role_paths"]["preterminal_capsule"]
        != [capsule_reference.get("path")]
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier bank lost its preterminal capsule binding."
        )
    if (
        not isinstance(export_context, Mapping)
        or manifest.get("terminal_frontier_export_context_sha256")
        != export_context.get("context_sha256")
        or receipt.get("terminal_frontier_export_context_sha256")
        != export_context.get("context_sha256")
        or export_context.get("schema")
        != LFM2_SEMANTIC_TERMINAL_FRONTIER_EXPORT_CONTEXT_SCHEMA
        or export_context.get("terminal_result_sha256") != expected_result_sha
        or export_context.get("terminal_seed_sha256") != report.get("seed_sha256")
        or export_context.get("terminal_incumbent_down_weight_terminal_action_sha256")
        != report.get("incumbent_seed_writer_sha256")
        or export_context.get("cache_content_identity_matches_layer_stack") is not True
        or export_context.get("terminal_cache_session_sha256")
        != export_context.get("layer_stack_cache_session_sha256")
        or export_context.get("frontier_export_occurs_before_final_w2_installation")
        is not True
        or type(export_context.get("preterminal_resume_mode")) is not bool
        or type(export_context.get("same_live_cache_session_as_layer_stack"))
        is not bool
        or (
            export_context.get("preterminal_resume_mode") is True
            and export_context.get("same_live_cache_session_as_layer_stack") is True
        )
        or export_context.get("operation_audit") != dict(_SEMANTIC_ZERO_OPERATIONS)
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier bank export context is invalid."
        )
    export_body = dict(export_context)
    export_sha = export_body.pop("context_sha256", None)
    if export_sha != canonical_sha256(export_body):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier bank export context seal is invalid."
        )
    payload_record = manifest.get("frontier_payload")
    if not isinstance(payload_record, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier manifest omitted its tensor payload."
        )
    tensors = _load_safetensors_path(
        root / _BANK_PAYLOAD_NAME,
        name=_BANK_PAYLOAD_NAME,
        artifact_kind=_BANK_ARTIFACT_KIND,
        file_role="terminal-w2-frontier",
        record=payload_record,
        clone_tensors=load_payload,
    )
    if (
        payload_record.get("path") != _BANK_PAYLOAD_NAME
        or payload_record.get("sha256") != receipt.get("frontier_payload_file_sha256")
        or [entry.get("name") for entry in payload_record.get("tensor_entries", [])]
        != list(sorted(candidate_ids))
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier tensor inventory is incomplete."
        )
    if load_payload:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional boundary
            raise RuntimeError("PyTorch is required for frontier banks.") from exc
        if set(tensors) != set(candidate_ids):
            raise Lfm2SemanticTerminalFrontierError(
                "Loaded terminal frontier tensor names are incomplete."
            )
        for candidate_id in candidate_ids:
            tensor = tensors[candidate_id]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.bfloat16
                or tensor.layout != torch.strided
                or tensor.ndim != 2
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not bool(torch.isfinite(tensor).all().item())
                or _terminal_tensor_sha256(tensor)
                != records[candidate_id]["state_sha256"]
            ):
                raise Lfm2SemanticTerminalFrontierError(
                    f"Loaded terminal candidate {candidate_id!r} is invalid."
                )
    _recheck_root(
        root,
        root_identity,
        inventory=_BANK_ROOT_INVENTORY,
        directory_names=frozenset(),
        label="terminal frontier bank",
    )
    return {
        "root": root,
        "receipt": receipt,
        "receipt_file_sha256": receipt_file_sha,
        "manifest": manifest,
        "report": report,
        "candidate_ids": candidate_ids,
        "tensors": tensors,
        "source_roots": manifest_source_roots,
    }


def load_lfm2_semantic_terminal_frontier_bank(
    directory: str | os.PathLike[str],
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_preterminal_capsule_receipt_sha256: str,
    expected_terminal_result_sha256: str,
) -> Lfm2SemanticTerminalFrontierBank:
    """Strictly load one externally pinned, complete terminal writer bank."""

    opened = _open_terminal_frontier_bank(
        directory,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_preterminal_capsule_receipt_sha256=(
            expected_preterminal_capsule_receipt_sha256
        ),
        expected_terminal_result_sha256=expected_terminal_result_sha256,
        load_payload=True,
    )
    manifest = opened["manifest"]
    capsule = manifest["preterminal_capsule"]
    reference = Lfm2SemanticTerminalFrontierBankReference(
        path=opened["root"],
        receipt_sha256=opened["receipt"]["receipt_sha256"],
        receipt_file_sha256=opened["receipt_file_sha256"],
        manifest_sha256=manifest["manifest_sha256"],
        preterminal_capsule_receipt_sha256=capsule["receipt_sha256"],
        terminal_result_sha256=manifest["terminal_result_sha256"],
        source_roots_sha256=manifest["authenticated_source_roots_sha256"],
        candidate_ids=opened["candidate_ids"],
    )
    return Lfm2SemanticTerminalFrontierBank(
        path=opened["root"],
        candidate_weights=MappingProxyType(dict(opened["tensors"])),
        terminal_result_report=_deep_freeze(dict(opened["report"])),
        manifest=_deep_freeze(dict(manifest)),
        receipt=_deep_freeze(dict(opened["receipt"])),
        reference=reference,
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticTerminalFrontierCompilation:
    """Authenticated result of resuming one capsule through frontier export."""

    preterminal_capsule_reference: Lfm2SemanticPreterminalCapsuleReference
    bank_reference: Lfm2SemanticTerminalFrontierBankReference
    terminal_receipt: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        receipt = _verify_seal(
            self.terminal_receipt,
            field_name="result_sha256",
            label="fused semantic terminal receipt",
        )
        terminal_result = receipt.get("terminal_result")
        if (
            not isinstance(terminal_result, Mapping)
            or self.bank_reference.preterminal_capsule_receipt_sha256
            != self.preterminal_capsule_reference.receipt_sha256
            or terminal_result.get("result_sha256")
            != self.bank_reference.terminal_result_sha256
        ):
            raise Lfm2SemanticTerminalFrontierError(
                "Terminal frontier compilation references different capsule phases."
            )
        object.__setattr__(
            self,
            "terminal_receipt",
            _deep_freeze(dict(receipt)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "preterminal_capsule": self.preterminal_capsule_reference.as_dict(),
            "terminal_frontier_bank": self.bank_reference.as_dict(),
            "terminal_receipt_sha256": self.terminal_receipt["result_sha256"],
            "terminal_result_sha256": self.bank_reference.terminal_result_sha256,
            "preterminal_resume_mode": True,
            "same_live_cache_session_as_layer_stack": False,
            "cache_content_identity_matches_layer_stack": True,
            "frontier_exported_before_final_w2_installation": True,
            "candidate_metric_threshold_used_for_export": False,
            "holistic_whole_response_review_required": True,
        }


def _require_frontier_output_disjoint(
    output: Path,
    *inputs: str | os.PathLike[str],
) -> None:
    for value in inputs:
        lexical = Path(value).expanduser().absolute()
        try:
            root = lexical.resolve(strict=True)
        except OSError as exc:
            raise Lfm2SemanticTerminalFrontierError(
                f"Frontier resume input is missing: {lexical}."
            ) from exc
        if root != lexical or _is_link_or_junction(lexical):
            raise Lfm2SemanticTerminalFrontierError(
                "Frontier resume inputs cannot be or traverse links."
            )
        if output == root or root in output.parents or output in root.parents:
            raise Lfm2SemanticTerminalFrontierError(
                "Terminal frontier output must be disjoint from every input root."
            )


def _local_tokenizer_source_directories(tokenizer: Any) -> tuple[Path, ...]:
    """Return existing local roots used to construct a tokenizer, if exposed."""

    candidates: list[Any] = [
        getattr(tokenizer, "name_or_path", None),
        getattr(tokenizer, "_name_or_path", None),
    ]
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        candidates.extend(
            init_kwargs.get(name)
            for name in ("name_or_path", "tokenizer_file", "vocab_file")
        )
    roots: dict[str, Path] = {}
    for value in candidates:
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            continue
        lexical = Path(value).expanduser().absolute()
        if not _lexists(lexical):
            continue
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as exc:
            raise Lfm2SemanticTerminalFrontierError(
                "Local tokenizer source changed during path authentication."
            ) from exc
        root = resolved if resolved.is_dir() else resolved.parent
        roots[str(root)] = root
    return tuple(roots[name] for name in sorted(roots))


def compile_lfm2_semantic_terminal_frontier_bank_from_capsule(
    *,
    preterminal_capsule_directory: str | os.PathLike[str],
    expected_preterminal_receipt_file_sha256: str,
    expected_preterminal_receipt_sha256: str,
    expected_preterminal_bindings_sha256: str,
    cache_directory: str | os.PathLike[str],
    expected_cache_key: str,
    bridge_directory: str | os.PathLike[str],
    expected_bridge_artifact_sha256: str,
    donor_tokenizer: Any,
    parent_model_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    device: str,
    dtype: str = "bfloat16",
    parent_final_norm_tensor_key: str = "model.embedding_norm.weight",
    parent_norm_epsilon_config_key: str = "norm_eps",
    parent_tied_tensor_key: str = "model.embed_tokens.weight",
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
    policy: Lfm2CpcfTerminalActionPolicy | None = None,
    fused_terminal_compiler: Callable[..., Mapping[str, Any]] = (
        compile_install_lfm2_fused_semantic_terminal_refold
    ),
) -> Lfm2SemanticTerminalFrontierCompilation:
    """Resume an authenticated capsule and durably export every terminal W2.

    The fused compiler opens a distinct authenticated cache session itself.  Its
    two-argument exporter is the only publication seam, and it runs before the
    compiler installs any final W2 into the in-memory capsule child.
    """

    if not callable(fused_terminal_compiler):
        raise TypeError("fused_terminal_compiler must be callable.")
    expected_cache = _digest(expected_cache_key, "expected semantic cache key")
    expected_bridge = _digest(
        expected_bridge_artifact_sha256,
        "expected tokenizer bridge artifact SHA-256",
    )
    output = _absolute_new_output(
        output_directory,
        label="terminal frontier bank",
    )
    donor_tokenizer_sources = _local_tokenizer_source_directories(donor_tokenizer)
    if not donor_tokenizer_sources:
        donor_tokenizer_sources = (
            _canonical_existing_source_directory(
                parent_model_directory,
                label="donor tokenizer fallback parent model directory",
            ),
        )
    _require_frontier_output_disjoint(
        output,
        preterminal_capsule_directory,
        cache_directory,
        bridge_directory,
        parent_model_directory,
        *donor_tokenizer_sources,
    )

    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    if cache.cache_key != expected_cache:
        raise Lfm2SemanticTerminalFrontierError(
            "Semantic cache identity differs from its external pin."
        )
    terminal_cache_compatibility = preflight_lfm2_long_horizon_terminal_cache(cache)
    capsule = load_lfm2_semantic_preterminal_capsule(
        preterminal_capsule_directory,
        expected_receipt_file_sha256=expected_preterminal_receipt_file_sha256,
        expected_receipt_sha256=expected_preterminal_receipt_sha256,
        expected_bindings_sha256=expected_preterminal_bindings_sha256,
        device=device,
        dtype=dtype,
    )
    capsule_tokenizer_sources = _local_tokenizer_source_directories(
        capsule.target_tokenizer
    )
    if not capsule_tokenizer_sources:
        capsule_tokenizer_sources = (
            _canonical_existing_source_directory(
                bridge_directory,
                label="target tokenizer fallback bridge directory",
            ),
        )
    source_roots = Lfm2SemanticTerminalFrontierSourceRoots(
        preterminal_capsule_directory=Path(preterminal_capsule_directory),
        semantic_cache_directory=Path(cache_directory),
        tokenizer_bridge_directory=Path(bridge_directory),
        parent_model_directory=Path(parent_model_directory),
        donor_tokenizer_directories=donor_tokenizer_sources,
        target_tokenizer_directories=capsule_tokenizer_sources,
    )
    _require_frontier_output_disjoint(
        output,
        *source_roots.all_paths(),
    )
    bindings = capsule.bindings
    cache_manifest = cache.manifest
    if (
        bindings.semantic_cache_key != cache.cache_key
        or bindings.semantic_cache_manifest_sha256
        != cache_manifest.get("manifest_sha256")
        or bindings.cache_parent_fingerprint != cache.parent_fingerprint
        or capsule.layer_stack.receipt.get("cache_selective_session_sha256")
        != bindings.cache_selective_session_sha256
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal capsule is not bound to the supplied semantic cache."
        )
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=capsule.target_tokenizer,
        expected_artifact_sha256=expected_bridge,
    )
    if (
        bridge.artifact_sha256 != bindings.bridge_artifact_sha256
        or bridge.artifact_sha256 != expected_bridge
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal capsule is not bound to the supplied tokenizer bridge."
        )
    parent_final_norm = load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        cache,
        parent_model_directory,
        tensor_key=parent_final_norm_tensor_key,
        epsilon_config_key=parent_norm_epsilon_config_key,
    )
    if parent_final_norm.result_sha256 != bindings.parent_final_norm_receipt_sha256:
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal capsule is not bound to the supplied parent final norm."
        )
    mapped_parent_tied_table, mapped_parent_tied_table_receipt = (
        load_lfm2_mapped_parent_tied_table_from_local_safetensors(
            parent_model_directory,
            cache=cache,
            bridge=bridge,
            authenticated_parent_final_norm=parent_final_norm,
            tensor_key=parent_tied_tensor_key,
        )
    )
    if (
        mapped_parent_tied_table_receipt.get("result_sha256")
        != bindings.mapped_parent_tied_table_receipt_sha256
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal capsule is not bound to the supplied mapped tied table."
        )
    try:
        expected_tied_interface_authentication = dict(
            _authenticate_fused_terminal_tied_interface(
                target_model=capsule.target_model,
                interface=capsule.interface,
                mapped_parent_tied_table=mapped_parent_tied_table,
                layer_receipt=capsule.layer_stack.receipt,
            )
        )
    except Exception as exc:
        raise Lfm2SemanticTerminalFrontierError(
            "Capsule tied interface could not be reauthenticated for terminal resume."
        ) from exc
    expected_live_context_facts = {
        "mapped_parent_tied_table_sha256": _tensor_sha256(mapped_parent_tied_table),
        "tied_interface_authentication_sha256": (
            expected_tied_interface_authentication["result_sha256"]
        ),
        "terminal_cache_compatibility_sha256": terminal_cache_compatibility[
            "authentication_sha256"
        ],
    }

    exported: list[
        tuple[Lfm2SemanticTerminalFrontierBankReference, Mapping[str, Any]]
    ] = []

    def exporter(
        result: Lfm2CpcfTerminalActionResidualResult,
        export_context: Mapping[str, Any],
    ) -> Lfm2SemanticTerminalFrontierBankReference:
        if exported:
            raise Lfm2SemanticTerminalFrontierError(
                "Fused terminal attempted to publish its frontier more than once."
            )
        if any(
            export_context.get(name) != value
            for name, value in expected_live_context_facts.items()
        ):
            raise Lfm2SemanticTerminalFrontierError(
                "Fused terminal export context differs from live resume inputs."
            )
        reference = publish_lfm2_semantic_terminal_frontier_bank(
            result,
            export_context,
            preterminal_capsule_reference=capsule.reference,
            source_roots=source_roots,
            output_directory=output,
        )
        exported.append((reference, dict(export_context)))
        return reference

    terminal_receipt = fused_terminal_compiler(
        cache=cache,
        bridge=bridge,
        target_model=capsule.target_model,
        layer_stack=capsule.layer_stack,
        interface=capsule.interface,
        source_selected_tied_table=mapped_parent_tied_table,
        parent_final_norm=parent_final_norm,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=capsule.target_tokenizer,
        split_plan=split_plan,
        policy=policy,
        terminal_cache_compatibility=terminal_cache_compatibility,
        authenticated_cache_session=None,
        preterminal_resume_mode=True,
        terminal_frontier_exporter=exporter,
    )
    if not isinstance(terminal_receipt, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Fused terminal returned no authenticated receipt."
        )
    terminal_receipt = _verify_seal(
        terminal_receipt,
        field_name="result_sha256",
        label="fused semantic terminal receipt",
    )
    try:
        authenticated_cache_facts = dict(
            _authenticate_fused_terminal_cache_evidence(terminal_receipt)
        )
    except Exception as exc:
        raise Lfm2SemanticTerminalFrontierError(
            "Fused terminal cache evidence failed post-export authentication."
        ) from exc
    if len(exported) != 1:
        raise Lfm2SemanticTerminalFrontierError(
            "Fused terminal did not publish exactly one complete frontier bank."
        )
    bank_reference, callback_context = exported[0]
    receipt_context = terminal_receipt.get("terminal_frontier_export_context")
    receipt_export = terminal_receipt.get("terminal_frontier_export")
    terminal_result = terminal_receipt.get("terminal_result")
    receipt_tied_interface = terminal_receipt.get("tied_interface_authentication")
    if (
        not isinstance(receipt_context, Mapping)
        or not isinstance(receipt_export, Mapping)
        or not isinstance(terminal_result, Mapping)
        or not isinstance(receipt_tied_interface, Mapping)
        or dict(receipt_context) != dict(callback_context)
        or receipt_context.get("context_sha256")
        != terminal_receipt.get("terminal_frontier_export_context_sha256")
        or receipt_context.get("preterminal_resume_mode") is not True
        or receipt_context.get("same_live_cache_session_as_layer_stack") is not False
        or receipt_context.get("cache_content_identity_matches_layer_stack") is not True
        or receipt_context.get("frontier_export_occurs_before_final_w2_installation")
        is not True
        or any(
            receipt_context.get(name) != value
            for name, value in expected_live_context_facts.items()
        )
        or receipt_context.get("terminal_cache_compatibility_sha256")
        != authenticated_cache_facts.get("terminal_cache_compatibility_sha256")
        or receipt_context.get("terminal_cache_payload_binding_sha256")
        != authenticated_cache_facts.get("terminal_cache_payload_binding_sha256")
        or receipt_context.get("terminal_cache_session_sha256")
        != authenticated_cache_facts.get("terminal_cache_session_sha256")
        or dict(receipt_tied_interface) != expected_tied_interface_authentication
        or terminal_receipt.get("tied_interface_authentication_sha256")
        != expected_tied_interface_authentication.get("result_sha256")
        or dict(receipt_export) != bank_reference.as_dict()
        or terminal_receipt.get("terminal_frontier_export_sha256")
        != canonical_sha256(bank_reference.as_dict())
        or terminal_receipt.get("terminal_frontier_exported_before_w2_installation")
        is not True
        or terminal_result.get("result_sha256") != bank_reference.terminal_result_sha256
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Fused terminal receipt lost its authenticated frontier publication."
        )
    return Lfm2SemanticTerminalFrontierCompilation(
        preterminal_capsule_reference=capsule.reference,
        bank_reference=bank_reference,
        terminal_receipt=terminal_receipt,
    )


def _live_identity_inventory(model: Any) -> tuple[tuple[Any, ...], ...]:
    """Capture identities/storage so an in-place install cannot replace topology."""

    values = []
    for role, method_name in (
        ("parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    ):
        method = getattr(model, method_name, None)
        if not callable(method):
            raise Lfm2SemanticTerminalFrontierError(
                f"Materialized child has no {method_name}() method."
            )
        try:
            named = tuple(method(remove_duplicate=False))
        except TypeError:  # pragma: no cover - older torch boundary
            named = tuple(method())
        for name, tensor in named:
            storage = tensor.untyped_storage()
            values.append(
                (
                    role,
                    str(name),
                    id(tensor),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                    int(tensor.storage_offset()),
                    tuple(int(size) for size in tensor.shape),
                    tuple(int(size) for size in tensor.stride()),
                    str(tensor.dtype),
                    str(tensor.device),
                )
            )
    return tuple(sorted(values, key=lambda value: (value[0], value[1])))


def _stable_topology_inventory(model: Any) -> dict[str, Any]:
    live = _live_identity_inventory(model)
    records = [
        {
            "role": role,
            "name": name,
            "storage_nbytes": storage_nbytes,
            "storage_offset": storage_offset,
            "shape": list(shape),
            "stride": list(stride),
            "dtype": dtype,
        }
        for (
            role,
            name,
            _object_id,
            _storage_pointer,
            storage_nbytes,
            storage_offset,
            shape,
            stride,
            dtype,
            _device,
        ) in live
    ]
    body = {
        "schema": "neural-forge-lfm2-frontier-stable-topology-v1",
        "tensor_count": len(records),
        "tensors": records,
        "parameter_inventory_sha256": independent_parameter_inventory(model)[
            "inventory_sha256"
        ],
    }
    return {**body, "topology_sha256": canonical_sha256(body)}


def _state_change_proof(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    before_dict = dict(before)
    after_dict = dict(after)
    if set(before_dict) != set(after_dict):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal materialization changed model state names."
        )
    changed = sorted(
        name for name in before_dict if before_dict[name] != after_dict[name]
    )
    unchanged_before = {
        name: before_dict[name] for name in sorted(before_dict) if name not in changed
    }
    unchanged_after = {
        name: after_dict[name] for name in sorted(after_dict) if name not in changed
    }
    body = {
        "schema": "neural-forge-lfm2-frontier-state-change-proof-v1",
        "state_entry_count": len(before_dict),
        "changed_state_keys": changed,
        "changed_state_count": len(changed),
        "before_entry_sha256": canonical_sha256(before_dict),
        "after_entry_sha256": canonical_sha256(after_dict),
        "unchanged_entry_sha256_before": canonical_sha256(unchanged_before),
        "unchanged_entry_sha256_after": canonical_sha256(unchanged_after),
        "all_nonterminal_state_entries_unchanged": (
            unchanged_before == unchanged_after
        ),
    }
    return {**body, "proof_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2SemanticTerminalFrontierMaterializationReference:
    path: Path
    checkpoint_path: Path
    receipt_sha256: str
    receipt_file_sha256: str
    checkpoint_artifact_sha256: str
    model_state_sha256: str
    candidate_id: str
    candidate_state_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "receipt_sha256",
            "receipt_file_sha256",
            "checkpoint_artifact_sha256",
            "model_state_sha256",
            "candidate_state_sha256",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise Lfm2SemanticTerminalFrontierError(
                "Materialized candidate ID must be nonempty text."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "checkpoint_path": str(self.checkpoint_path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_file_sha256": self.receipt_file_sha256,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "model_state_sha256": self.model_state_sha256,
            "candidate_id": self.candidate_id,
            "candidate_state_sha256": self.candidate_state_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticTerminalFrontierMaterialization:
    path: Path
    checkpoint_path: Path
    target_model: Any = field(repr=False, compare=False)
    target_tokenizer: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    reference: Lfm2SemanticTerminalFrontierMaterializationReference


def _capsule_bank_binding(
    capsule: Lfm2SemanticPreterminalCapsule,
    bank: Lfm2SemanticTerminalFrontierBank,
) -> None:
    source = bank.manifest.get("preterminal_capsule")
    if (
        not isinstance(source, Mapping)
        or source.get("receipt_sha256") != capsule.reference.receipt_sha256
        or source.get("receipt_file_sha256") != capsule.reference.receipt_file_sha256
        or source.get("manifest_sha256") != capsule.reference.manifest_sha256
        or source.get("bindings_sha256") != capsule.reference.bindings_sha256
        or source.get("context_binding_sha256")
        != capsule.reference.context_binding_sha256
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier bank was not derived from this preterminal capsule."
        )


def _candidate_record(
    bank: Lfm2SemanticTerminalFrontierBank,
    candidate_id: str,
) -> dict[str, Any]:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise Lfm2SemanticTerminalFrontierError("candidate_id must be nonempty text.")
    if candidate_id not in bank.reference.candidate_ids:
        raise Lfm2SemanticTerminalFrontierError(
            f"Candidate {candidate_id!r} is not in the authenticated retained bank."
        )
    records = {
        str(value.get("candidate_id")): value
        for value in bank.terminal_result_report["frontier"][1:]
    }
    record = records.get(candidate_id)
    if not isinstance(record, Mapping):
        raise Lfm2SemanticTerminalFrontierError(
            "Authenticated terminal report omitted the requested candidate."
        )
    return _canonical_mapping(
        _deep_thaw(record),
        "requested terminal frontier candidate",
    )


def _materialize_loaded_candidate(
    *,
    capsule: Lfm2SemanticPreterminalCapsule,
    bank: Lfm2SemanticTerminalFrontierBank,
    candidate_id: str,
    output_directory: str | os.PathLike[str],
    device: str,
    dtype: str,
) -> Lfm2SemanticTerminalFrontierMaterializationReference:
    bank_source_roots = _authenticate_source_roots_mapping(
        bank.manifest.get("authenticated_source_roots")
    )
    if bank_source_roots["role_paths"]["preterminal_capsule"] != [str(capsule.path)]:
        raise Lfm2SemanticTerminalFrontierError(
            "Loaded capsule path differs from sealed bank source provenance."
        )
    output = _absolute_new_output(
        output_directory,
        label="terminal frontier materialization",
    )
    _require_frontier_output_disjoint(
        output,
        capsule.path,
        bank.path,
        *_source_root_paths(bank_source_roots),
        *_local_tokenizer_source_directories(capsule.target_tokenizer),
    )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional boundary
        raise RuntimeError("PyTorch is required for frontier materialization.") from exc
    _capsule_bank_binding(capsule, bank)
    record = _candidate_record(bank, candidate_id)
    candidate = bank.candidate_weights[candidate_id]
    candidate_sha = _digest(record.get("state_sha256"), "candidate state SHA-256")
    if _terminal_tensor_sha256(candidate) != candidate_sha:
        raise Lfm2SemanticTerminalFrontierError(
            "Requested candidate tensor changed after bank authentication."
        )
    model = capsule.target_model
    tokenizer = capsule.target_tokenizer
    final_w2 = _final_w2_parameter(model)
    if (
        final_w2.dtype != torch.bfloat16
        or final_w2.layout != torch.strided
        or final_w2.device.type == "meta"
        or tuple(final_w2.shape) != tuple(candidate.shape)
        or int(torch.count_nonzero(final_w2).item()) != 0
        or candidate.dtype != torch.bfloat16
        or candidate.device.type != "cpu"
        or candidate.layout != torch.strided
        or not candidate.is_contiguous()
        or not bool(torch.isfinite(candidate).all().item())
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Capsule final W2 or retained candidate has invalid dense BF16 geometry."
        )
    before_model_state = fingerprint_model_state(model)
    if before_model_state != capsule.manifest["model"]["model_state_sha256"]:
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal child changed after capsule authentication."
        )
    before_entries = fingerprint_model_state_entries(model)
    if _FINAL_W2_NAME not in before_entries:
        raise Lfm2SemanticTerminalFrontierError(
            "Preterminal child omitted the canonical final W2 state key."
        )
    before_w2_sha = _tensor_sha256(final_w2)
    if torch.equal(candidate, final_w2.detach().to(device="cpu")):
        raise Lfm2SemanticTerminalFrontierError(
            "Requested retained candidate is an exact no-op at the preterminal seam."
        )
    before_live_identity = _live_identity_inventory(model)
    before_topology = _stable_topology_inventory(model)
    before_parameter_inventory = _canonical_mapping(
        _deep_thaw(independent_parameter_inventory(model)),
        "preterminal parameter inventory",
    )
    with torch.no_grad():
        final_w2.copy_(candidate.to(device=final_w2.device, dtype=final_w2.dtype))
    if _terminal_tensor_sha256(candidate) != candidate_sha:
        raise Lfm2SemanticTerminalFrontierError(
            "Materialization mutated the retained source candidate."
        )
    after_live_identity = _live_identity_inventory(model)
    after_topology = _stable_topology_inventory(model)
    after_parameter_inventory = _canonical_mapping(
        _deep_thaw(independent_parameter_inventory(model)),
        "materialized parameter inventory",
    )
    after_entries = fingerprint_model_state_entries(model)
    state_change = _state_change_proof(before_entries, after_entries)
    if (
        state_change["changed_state_keys"] != [_FINAL_W2_NAME]
        or state_change["all_nonterminal_state_entries_unchanged"] is not True
        or before_live_identity != after_live_identity
        or before_topology != after_topology
        or before_parameter_inventory != after_parameter_inventory
        or _terminal_tensor_sha256(final_w2) != candidate_sha
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal materialization changed state or topology outside final W2."
        )
    after_model_state = fingerprint_model_state(model)
    if after_model_state == before_model_state:
        raise Lfm2SemanticTerminalFrontierError(
            "Effectful final-W2 materialization did not change model state."
        )
    context_body = {
        "schema": LFM2_SEMANTIC_TERMINAL_FRONTIER_CONTEXT_SCHEMA,
        "execution_mode": "two-session-preterminal-capsule-resume",
        "same_live_cache_session_claimed": False,
        "preterminal_capsule_receipt_sha256": capsule.reference.receipt_sha256,
        "preterminal_capsule_manifest_sha256": capsule.reference.manifest_sha256,
        "preterminal_capsule_context_binding_sha256": (
            capsule.reference.context_binding_sha256
        ),
        "terminal_frontier_bank_receipt_sha256": bank.reference.receipt_sha256,
        "terminal_frontier_bank_manifest_sha256": bank.reference.manifest_sha256,
        "authenticated_source_roots_sha256": bank.reference.source_roots_sha256,
        "terminal_result_sha256": bank.reference.terminal_result_sha256,
        "candidate_id": candidate_id,
        "candidate_state_sha256": candidate_sha,
        "preterminal_model_state_sha256": before_model_state,
        "materialized_model_state_sha256": after_model_state,
        "final_w2_parameter_name": _FINAL_W2_NAME,
        "holistic_whole_response_review_pending": True,
    }
    context_sha = canonical_sha256(context_body)
    stage = Path(
        tempfile.mkdtemp(
            prefix=".semantic-frontier-child-",
            dir=str(output.parent),
        )
    ).absolute()
    stage_identity = _stat_identity(
        _real_directory(stage, name="frontier materialization stage")
    )
    moved = False
    try:
        reloaded, checkpoint = publish_lfm2_semantic_postfold_checkpoint(
            target_model=model,
            target_tokenizer=tokenizer,
            output_directory=stage / _MATERIALIZATION_CHECKPOINT_NAME,
            context_binding_sha256=context_sha,
            device=device,
            dtype=dtype,
        )
        reloaded_entries = fingerprint_model_state_entries(reloaded)
        reload_change = _state_change_proof(before_entries, reloaded_entries)
        reloaded_topology = _stable_topology_inventory(reloaded)
        if (
            reload_change["changed_state_keys"] != [_FINAL_W2_NAME]
            or reload_change["all_nonterminal_state_entries_unchanged"] is not True
            or reloaded_topology != after_topology
            or fingerprint_model_state(reloaded) != after_model_state
            or _terminal_tensor_sha256(_final_w2_parameter(reloaded)) != candidate_sha
            or _parameter_count(reloaded) != _parameter_count(model)
        ):
            raise Lfm2SemanticTerminalFrontierError(
                "Fresh standalone reload differs from exact-W2 materialization."
            )
        receipt_body = {
            "schema": LFM2_SEMANTIC_TERMINAL_FRONTIER_MATERIALIZATION_SCHEMA,
            "status": "standalone-dense-candidate-ready-for-whole-response-review",
            **_NONPROMOTABLE_FLAGS,
            "preterminal_capsule": capsule.reference.as_dict(),
            "terminal_frontier_bank": bank.reference.as_dict(),
            "terminal_result_report": _deep_thaw(bank.terminal_result_report),
            "candidate_id": candidate_id,
            "candidate_state_sha256": candidate_sha,
            "candidate_frontier_record": record,
            "all_authenticated_candidate_ids": list(bank.reference.candidate_ids),
            "all_authenticated_candidate_count": len(bank.reference.candidate_ids),
            "candidate_materialized_without_metric_threshold": True,
            "candidate_metric_fields_read_for_materialization": False,
            "numeric_metrics_preserved_as_diagnostics": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "holistic_all_64_whole_response_review_required": True,
            "holistic_review_complete": False,
            "materialization_is_not_intelligence_acceptance": True,
            "source_model_state_sha256": before_model_state,
            "materialized_model_state_sha256": after_model_state,
            "source_final_w2_semantic_sha256": before_w2_sha,
            "materialized_final_w2_terminal_action_sha256": candidate_sha,
            "materialized_final_w2_semantic_sha256": _tensor_sha256(final_w2),
            "final_w2_parameter_name": _FINAL_W2_NAME,
            "state_change_proof": state_change,
            "fresh_reload_state_change_proof": reload_change,
            "topology_before": before_topology,
            "topology_after": after_topology,
            "fresh_reload_topology": reloaded_topology,
            "parameter_inventory_before": before_parameter_inventory,
            "parameter_inventory_after": after_parameter_inventory,
            "parameter_identity_and_storage_preserved_during_install": True,
            "exactly_one_state_tensor_changed": True,
            "only_final_layer_w2_changed": True,
            "parameter_count_changed": False,
            "checkpoint": dict(checkpoint),
            "fresh_reload_parent_free_state_sha256": fingerprint_model_state(reloaded),
            "context_binding": {**context_body, "context_sha256": context_sha},
            "standalone_child": True,
            "parent_model_required_for_reload": False,
            "trace_cache_required_for_reload": False,
            "frontier_bank_required_for_reload": False,
            "preterminal_capsule_required_for_reload": False,
            "sidecar_required_for_inference": False,
            "training_performed": False,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        receipt = _sealed(receipt_body, "receipt_sha256")
        receipt_path = stage / _MATERIALIZATION_RECEIPT_NAME
        receipt_raw = _write_canonical_json(receipt_path, receipt)
        receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
        _fsync_directory(stage)
        _open_terminal_frontier_materialization(
            stage,
            expected_receipt_file_sha256=receipt_file_sha,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_candidate_id=candidate_id,
            expected_candidate_state_sha256=candidate_sha,
        )
        if _lexists(output):
            raise Lfm2SemanticTerminalFrontierError(
                "Terminal frontier child destination appeared during publication."
            )
        _atomic_rename_noreplace(stage, output)
        moved = True
        _fsync_directory(output.parent)
        opened = _open_terminal_frontier_materialization(
            output,
            expected_receipt_file_sha256=receipt_file_sha,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_candidate_id=candidate_id,
            expected_candidate_state_sha256=candidate_sha,
        )
        return Lfm2SemanticTerminalFrontierMaterializationReference(
            path=output,
            checkpoint_path=output / _MATERIALIZATION_CHECKPOINT_NAME,
            receipt_sha256=receipt["receipt_sha256"],
            receipt_file_sha256=opened["receipt_file_sha256"],
            checkpoint_artifact_sha256=checkpoint["artifact_sha256"],
            model_state_sha256=after_model_state,
            candidate_id=candidate_id,
            candidate_state_sha256=candidate_sha,
        )
    except BaseException:
        if not moved and _lexists(stage):
            _safe_remove_stage(
                stage,
                parent=output.parent,
                identity=stage_identity,
                prefix=".semantic-frontier-child-",
            )
        raise


def materialize_lfm2_semantic_terminal_frontier_candidate(
    *,
    preterminal_capsule_directory: str | os.PathLike[str],
    expected_preterminal_receipt_file_sha256: str,
    expected_preterminal_receipt_sha256: str,
    expected_preterminal_bindings_sha256: str,
    terminal_frontier_bank_directory: str | os.PathLike[str],
    expected_terminal_frontier_bank_receipt_file_sha256: str,
    expected_terminal_frontier_bank_receipt_sha256: str,
    expected_terminal_result_sha256: str,
    expected_source_roots: Lfm2SemanticTerminalFrontierSourceRoots,
    candidate_id: str,
    output_directory: str | os.PathLike[str],
    device: str,
    dtype: str = "bfloat16",
) -> Lfm2SemanticTerminalFrontierMaterializationReference:
    """Load both phases and materialize one retained W2 into a standalone child."""

    if not isinstance(
        expected_source_roots,
        Lfm2SemanticTerminalFrontierSourceRoots,
    ):
        raise TypeError(
            "expected_source_roots must explicitly pin every role-labeled source."
        )
    if (
        expected_source_roots.preterminal_capsule_directory
        != Path(preterminal_capsule_directory).expanduser().absolute()
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Explicit source-root pins name a different capsule directory."
        )
    output = _absolute_new_output(
        output_directory,
        label="terminal frontier materialization",
    )
    _require_frontier_output_disjoint(
        output,
        terminal_frontier_bank_directory,
        *expected_source_roots.all_paths(),
    )
    metadata_only_bank = _open_terminal_frontier_bank(
        terminal_frontier_bank_directory,
        expected_receipt_file_sha256=(
            expected_terminal_frontier_bank_receipt_file_sha256
        ),
        expected_receipt_sha256=expected_terminal_frontier_bank_receipt_sha256,
        expected_preterminal_capsule_receipt_sha256=(
            expected_preterminal_receipt_sha256
        ),
        expected_terminal_result_sha256=expected_terminal_result_sha256,
        load_payload=False,
    )
    if metadata_only_bank["source_roots"] != expected_source_roots.as_dict():
        raise Lfm2SemanticTerminalFrontierError(
            "Explicit source-root pins differ from sealed bank provenance."
        )
    capsule = load_lfm2_semantic_preterminal_capsule(
        preterminal_capsule_directory,
        expected_receipt_file_sha256=expected_preterminal_receipt_file_sha256,
        expected_receipt_sha256=expected_preterminal_receipt_sha256,
        expected_bindings_sha256=expected_preterminal_bindings_sha256,
        device=device,
        dtype=dtype,
    )
    bank = load_lfm2_semantic_terminal_frontier_bank(
        terminal_frontier_bank_directory,
        expected_receipt_file_sha256=(
            expected_terminal_frontier_bank_receipt_file_sha256
        ),
        expected_receipt_sha256=expected_terminal_frontier_bank_receipt_sha256,
        expected_preterminal_capsule_receipt_sha256=(
            expected_preterminal_receipt_sha256
        ),
        expected_terminal_result_sha256=expected_terminal_result_sha256,
    )
    return _materialize_loaded_candidate(
        capsule=capsule,
        bank=bank,
        candidate_id=candidate_id,
        output_directory=output,
        device=device,
        dtype=dtype,
    )


def _open_terminal_frontier_materialization(
    directory: str | os.PathLike[str],
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_candidate_id: str,
    expected_candidate_state_sha256: str,
) -> dict[str, Any]:
    expected_file_sha = _digest(
        expected_receipt_file_sha256,
        "expected materialization receipt file SHA-256",
    )
    expected_receipt_sha = _digest(
        expected_receipt_sha256,
        "expected materialization receipt SHA-256",
    )
    expected_state_sha = _digest(
        expected_candidate_state_sha256,
        "expected candidate state SHA-256",
    )
    if not isinstance(expected_candidate_id, str) or not expected_candidate_id:
        raise Lfm2SemanticTerminalFrontierError(
            "expected_candidate_id must be nonempty text."
        )
    root, root_identity = _root_with_exact_inventory(
        directory,
        inventory=_MATERIALIZATION_ROOT_INVENTORY,
        directory_names=frozenset({_MATERIALIZATION_CHECKPOINT_NAME}),
        label="terminal frontier materialization",
    )
    receipt, raw = _read_canonical_json_with_raw(
        root / _MATERIALIZATION_RECEIPT_NAME,
        name=_MATERIALIZATION_RECEIPT_NAME,
    )
    file_sha = hashlib.sha256(raw).hexdigest()
    receipt = _verify_seal(
        receipt,
        field_name="receipt_sha256",
        label="terminal frontier materialization receipt",
    )
    checkpoint = receipt.get("checkpoint")
    context = receipt.get("context_binding")
    state_change = receipt.get("state_change_proof")
    reload_change = receipt.get("fresh_reload_state_change_proof")
    bank = receipt.get("terminal_frontier_bank")
    capsule = receipt.get("preterminal_capsule")
    if (
        file_sha != expected_file_sha
        or receipt.get("receipt_sha256") != expected_receipt_sha
        or receipt.get("schema")
        != LFM2_SEMANTIC_TERMINAL_FRONTIER_MATERIALIZATION_SCHEMA
        or receipt.get("status")
        != "standalone-dense-candidate-ready-for-whole-response-review"
        or receipt.get("candidate_id") != expected_candidate_id
        or receipt.get("candidate_state_sha256") != expected_state_sha
        or receipt.get("materialized_final_w2_terminal_action_sha256")
        != expected_state_sha
        or receipt.get("final_w2_parameter_name") != _FINAL_W2_NAME
        or any(
            receipt.get(name) != expected
            for name, expected in _NONPROMOTABLE_FLAGS.items()
        )
        or receipt.get("candidate_materialized_without_metric_threshold") is not True
        or receipt.get("candidate_metric_fields_read_for_materialization") is not False
        or receipt.get("numeric_metrics_have_final_intelligence_authority") is not False
        or receipt.get("holistic_all_64_whole_response_review_required") is not True
        or receipt.get("holistic_review_complete") is not False
        or receipt.get("materialization_is_not_intelligence_acceptance") is not True
        or receipt.get("exactly_one_state_tensor_changed") is not True
        or receipt.get("only_final_layer_w2_changed") is not True
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("standalone_child") is not True
        or receipt.get("parent_model_required_for_reload") is not False
        or receipt.get("trace_cache_required_for_reload") is not False
        or receipt.get("frontier_bank_required_for_reload") is not False
        or receipt.get("preterminal_capsule_required_for_reload") is not False
        or receipt.get("sidecar_required_for_inference") is not False
        or receipt.get("training_performed") is not False
        or receipt.get("optimizer_steps") != 0
        or receipt.get("parameters_pruned") != 0
        or receipt.get("parameters_quantized") != 0
        or receipt.get("parameters_sparsified") != 0
        or receipt.get("operation_audit") != _ZERO_OPERATION_AUDIT
        or not isinstance(checkpoint, Mapping)
        or not isinstance(context, Mapping)
        or not isinstance(state_change, Mapping)
        or not isinstance(reload_change, Mapping)
        or not isinstance(bank, Mapping)
        or not isinstance(capsule, Mapping)
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier materialization receipt is invalid."
        )
    context_body = dict(context)
    context_sha = context_body.pop("context_sha256", None)
    if (
        context_sha != canonical_sha256(context_body)
        or context.get("schema") != LFM2_SEMANTIC_TERMINAL_FRONTIER_CONTEXT_SCHEMA
        or context.get("execution_mode") != "two-session-preterminal-capsule-resume"
        or context.get("same_live_cache_session_claimed") is not False
        or context.get("candidate_id") != expected_candidate_id
        or context.get("candidate_state_sha256") != expected_state_sha
        or context.get("authenticated_source_roots_sha256")
        != bank.get("source_roots_sha256")
        or checkpoint.get("context_binding_sha256") != context_sha
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Terminal frontier materialization context is invalid."
        )
    for proof in (state_change, reload_change):
        proof_body = dict(proof)
        proof_sha = proof_body.pop("proof_sha256", None)
        if (
            proof_sha != canonical_sha256(proof_body)
            or proof.get("changed_state_keys") != [_FINAL_W2_NAME]
            or proof.get("changed_state_count") != 1
            or proof.get("all_nonterminal_state_entries_unchanged") is not True
            or proof.get("unchanged_entry_sha256_before")
            != proof.get("unchanged_entry_sha256_after")
        ):
            raise Lfm2SemanticTerminalFrontierError(
                "Exact final-W2 state-change proof is invalid."
            )
    terminal_report = _verify_seal(
        receipt.get("terminal_result_report"),
        field_name="result_sha256",
        label="materialized terminal result report",
    )
    candidate_ids, records = _frontier_records(terminal_report)
    if (
        receipt.get("all_authenticated_candidate_ids") != list(candidate_ids)
        or receipt.get("all_authenticated_candidate_count") != len(candidate_ids)
        or expected_candidate_id not in records
        or records[expected_candidate_id].get("state_sha256") != expected_state_sha
        or receipt.get("candidate_frontier_record") != records[expected_candidate_id]
        or bank.get("terminal_result_sha256") != terminal_report["result_sha256"]
        or bank.get("preterminal_capsule_receipt_sha256")
        != capsule.get("receipt_sha256")
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Materialization lost its complete terminal frontier provenance."
        )
    _recheck_root(
        root,
        root_identity,
        inventory=_MATERIALIZATION_ROOT_INVENTORY,
        directory_names=frozenset({_MATERIALIZATION_CHECKPOINT_NAME}),
        label="terminal frontier materialization",
    )
    return {
        "root": root,
        "receipt": receipt,
        "receipt_file_sha256": file_sha,
    }


def load_lfm2_semantic_terminal_frontier_materialization(
    directory: str | os.PathLike[str],
    *,
    expected_receipt_file_sha256: str,
    expected_receipt_sha256: str,
    expected_candidate_id: str,
    expected_candidate_state_sha256: str,
    device: str,
    dtype: str = "bfloat16",
) -> Lfm2SemanticTerminalFrontierMaterialization:
    """Authenticate provenance and reload the child via the existing loader."""

    opened = _open_terminal_frontier_materialization(
        directory,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_candidate_id=expected_candidate_id,
        expected_candidate_state_sha256=expected_candidate_state_sha256,
    )
    receipt = opened["receipt"]
    checkpoint = receipt["checkpoint"]
    checkpoint_path = opened["root"] / _MATERIALIZATION_CHECKPOINT_NAME
    model, tokenizer, metadata = load_lfm2_cpcf_v5_standalone_child(
        checkpoint_path,
        expected_artifact_sha256=checkpoint["artifact_sha256"],
        device=device,
        dtype=dtype,
    )
    candidate_sha = _digest(
        expected_candidate_state_sha256,
        "expected candidate state SHA-256",
    )
    if (
        metadata.get("model_state_sha256")
        != receipt.get("materialized_model_state_sha256")
        or fingerprint_model_state(model)
        != receipt.get("fresh_reload_parent_free_state_sha256")
        or _terminal_tensor_sha256(_final_w2_parameter(model)) != candidate_sha
        or checkpoint.get("model_state_sha256")
        != receipt.get("materialized_model_state_sha256")
        or checkpoint.get("parameter_inventory_sha256")
        != metadata.get("parameter_inventory_sha256")
        or checkpoint.get("tokenizer_semantics_sha256")
        != metadata.get("tokenizer_semantics_sha256")
        or checkpoint.get("staged_reload_verified") is not True
        or checkpoint.get("published_reload_verified") is not True
        or checkpoint.get("standalone_dense_safetensors") is not True
        or checkpoint.get("parent_model_required_for_reload") is not False
        or checkpoint.get("cache_required_for_reload") is not False
    ):
        raise Lfm2SemanticTerminalFrontierError(
            "Strict-local materialized child differs from its provenance."
        )
    reference = Lfm2SemanticTerminalFrontierMaterializationReference(
        path=opened["root"],
        checkpoint_path=checkpoint_path,
        receipt_sha256=receipt["receipt_sha256"],
        receipt_file_sha256=opened["receipt_file_sha256"],
        checkpoint_artifact_sha256=checkpoint["artifact_sha256"],
        model_state_sha256=receipt["materialized_model_state_sha256"],
        candidate_id=expected_candidate_id,
        candidate_state_sha256=candidate_sha,
    )
    return Lfm2SemanticTerminalFrontierMaterialization(
        path=opened["root"],
        checkpoint_path=checkpoint_path,
        target_model=model,
        target_tokenizer=tokenizer,
        receipt=_deep_freeze(dict(receipt)),
        reference=reference,
    )


__all__ = [
    "LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_MANIFEST_SCHEMA",
    "LFM2_SEMANTIC_TERMINAL_FRONTIER_BANK_RECEIPT_SCHEMA",
    "LFM2_SEMANTIC_TERMINAL_FRONTIER_CONTEXT_SCHEMA",
    "LFM2_SEMANTIC_TERMINAL_FRONTIER_MATERIALIZATION_SCHEMA",
    "LFM2_SEMANTIC_TERMINAL_FRONTIER_SOURCE_ROOTS_SCHEMA",
    "Lfm2SemanticTerminalFrontierBank",
    "Lfm2SemanticTerminalFrontierBankReference",
    "Lfm2SemanticTerminalFrontierCompilation",
    "Lfm2SemanticTerminalFrontierError",
    "Lfm2SemanticTerminalFrontierMaterialization",
    "Lfm2SemanticTerminalFrontierMaterializationReference",
    "Lfm2SemanticTerminalFrontierSourceRoots",
    "compile_lfm2_semantic_terminal_frontier_bank_from_capsule",
    "load_lfm2_semantic_terminal_frontier_bank",
    "load_lfm2_semantic_terminal_frontier_materialization",
    "materialize_lfm2_semantic_terminal_frontier_candidate",
    "publish_lfm2_semantic_terminal_frontier_bank",
]
