"""Strict causal-pair verifier for the real H776 attention diagnostic.

The verifier is deliberately downstream-only.  It accepts two completed,
self-contained diagnostics, authenticates their proof chains, and compares
the already-frozen selection and validation evidence.  It has no model,
tensor, cache, fitting, retry, or publication API.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .hf_lfm2_cpcf_v5_real_attention_probe import (
    H776_ATTENTION_CANDIDATE_IDS,
    H776_BLANK_REPORT_SHA256,
    H776_BLANK_STATE_SHA256,
    H776_CACHE_SHA256,
    H776_FIT_QUERY_ROWS,
    H776_FULL_POLICY_SHA256,
    H776_HEAD_DIM,
    H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256,
    H776_LAYER_COUNT,
    H776_LAYER_INDEX,
    H776_LAYER_TYPES,
    H776_MAXIMUM_CUDA_BYTES_PER_DEVICE,
    H776_MAXIMUM_PROCESS_RSS_BYTES,
    H776_OPERATOR_KINDS,
    H776_ORIGIN_COMPILER_SHA256,
    H776_PARAMETER_COMPRESSION_RATIO,
    H776_PARENT_UNIQUE_PARAMETERS,
    H776_PHASE_COUNT,
    H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
    H776_SEQUENCE_SOLVER_PLAN_SHA256,
    H776_SELECTIVE_CATALOG_SHA256,
    H776_SELECTIVE_PROVIDER_SHA256,
    H776_SELECTIVE_SOURCE_SHA256,
    H776_SOURCE_HIDDEN_SIZE,
    H776_SOURCE_INVENTORY_SHA256,
    H776_SOURCE_STATE_SHA256,
    H776_SOURCE_TIED_TABLE_SHA256,
    H776_SPLIT_PHASE_COUNTS,
    H776_SPLIT_TRAJECTORY_COUNTS,
    H776_TARGET_HIDDEN_SIZE,
    H776_TARGET_INTERMEDIATE_SIZE,
    H776_TARGET_KEY_VALUE_HEADS,
    H776_TARGET_PLAN_FINGERPRINT,
    H776_TARGET_QUERY_HEADS,
    H776_TARGET_TOPOLOGY_SHA256,
    H776_TARGET_UNIQUE_PARAMETERS,
    H776_TOKEN_BRIDGE_SHA256,
    H776_TOKENIZER_SHA256,
    H776_TRAJECTORY_COUNT,
    H776_VALIDATION_QUERY_ROWS,
    LFM2_CPCF_V5_REAL_ATTENTION_CHECKPOINT_AUDIT_SCHEMA,
    LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA,
    LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA,
    LFM2_CPCF_V5_REAL_ATTENTION_RESOURCE_REPORT_SCHEMA,
    Lfm2SequenceEffectivePcgCollector,
)
from .hf_lfm2_sequence_effective_attention_fold import (
    LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA,
)


LFM2_CPCF_V5_REAL_ATTENTION_PROBE_RUN_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-probe-run-v1"
)
LFM2_CPCF_V5_REAL_ATTENTION_CAUSAL_PAIR_VERDICT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-causal-pair-verdict-v1"
)

_EXPECTED_PHASE_NAMES = (
    "prefill",
    "decode.000",
    "decode.001",
    "decode.002",
    "decode.003",
    "decode.004",
    "decode.005",
    "decode.006",
    "decode.007",
)
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")
_RUN_FIELDS = frozenset(
    {
        "schema",
        "status",
        "probe",
        "runtime_guard",
        "structural_parameter_compaction",
        "parent_parameter_inventory",
        "target_parameter_inventory",
        "local_parent_io_audit",
        "catalog_preparation_receipt_sha256",
        "catalog_reauthenticated_after_provider_close",
        "post_operator_chart_transport_shadow_enabled",
        "attention_sequential_transport_enabled",
        "diagnostic_only",
        "child_checkpoint_saved",
        "publication_attempted",
        "promotion_authorized",
        "receipt_sha256",
    }
)
_PROBE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "pins",
        "runtime_compiler_implementation",
        "runtime_sequence_source_sha256",
        "cross_runtime_source_inventory",
        "target_geometry",
        "parameter_compaction",
        "checkpoint_audit",
        "frontier_audit",
        "phase_audit",
        "prefix_install_report",
        "layer_2_result",
        "post_operator_chart_transport_shadow",
        "attention_sequential_transport",
        "resource_report",
        "stop_conditions",
        "promotion_authorized",
        "accepted_child_artifact",
        "whole_model_intelligence_retention_proven",
        "non_promotable_probe_only",
        "report_sha256",
    }
)
_PIN_FIELDS = frozenset(
    {
        "checkpoint_store_directory",
        "checkpoint_key",
        "checkpoint_manifest_sha256",
        "expected_trace_sha256",
        "expected_runtime_compiler_implementation_sha256",
        "expected_sequence_source_sha256",
        "expected_target_plan_fingerprint",
        "expected_target_topology_sha256",
        "expected_selective_catalog_sha256",
        "expected_selective_source_sha256",
        "expected_selective_provider_sha256",
        "expected_cache_sha256",
        "expected_token_bridge_sha256",
        "expected_tokenizer_sha256",
        "expected_source_tied_table_sha256",
        "expected_source_state_sha256",
        "expected_source_inventory_sha256",
        "expected_blank_report_sha256",
        "expected_blank_state_sha256",
        "expected_full_policy_sha256",
        "expected_origin_compiler_sha256",
        "expected_sequence_solver_plan_sha256",
        "layer_index",
        "expected_target_unique_parameters",
        "maximum_process_rss_bytes",
        "maximum_cuda_bytes_per_device",
        "require_zero_swap",
        "expected_attention_sequential_transport_enabled",
    }
)
_PIN_CONSTANTS = MappingProxyType(
    {
        "expected_target_plan_fingerprint": H776_TARGET_PLAN_FINGERPRINT,
        "expected_target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        "expected_selective_catalog_sha256": H776_SELECTIVE_CATALOG_SHA256,
        "expected_selective_source_sha256": H776_SELECTIVE_SOURCE_SHA256,
        "expected_selective_provider_sha256": H776_SELECTIVE_PROVIDER_SHA256,
        "expected_cache_sha256": H776_CACHE_SHA256,
        "expected_token_bridge_sha256": H776_TOKEN_BRIDGE_SHA256,
        "expected_tokenizer_sha256": H776_TOKENIZER_SHA256,
        "expected_source_tied_table_sha256": H776_SOURCE_TIED_TABLE_SHA256,
        "expected_source_state_sha256": H776_SOURCE_STATE_SHA256,
        "expected_source_inventory_sha256": H776_SOURCE_INVENTORY_SHA256,
        "expected_blank_report_sha256": H776_BLANK_REPORT_SHA256,
        "expected_blank_state_sha256": H776_BLANK_STATE_SHA256,
        "expected_full_policy_sha256": H776_FULL_POLICY_SHA256,
        "expected_origin_compiler_sha256": H776_ORIGIN_COMPILER_SHA256,
        "expected_sequence_solver_plan_sha256": (H776_SEQUENCE_SOLVER_PLAN_SHA256),
        "layer_index": H776_LAYER_INDEX,
        "expected_target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
        "maximum_process_rss_bytes": H776_MAXIMUM_PROCESS_RSS_BYTES,
        "maximum_cuda_bytes_per_device": H776_MAXIMUM_CUDA_BYTES_PER_DEVICE,
        "require_zero_swap": True,
    }
)
_METRIC_FIELDS = frozenset(
    {
        "rows",
        "width",
        "rmse",
        "relative_rmse",
        "cosine_similarity",
        "maximum_absolute_error",
        "prediction_sha256",
        "target_sha256",
    }
)
_INVENTORY_FIELDS = frozenset(
    {
        "method",
        "unique_parameters",
        "raw_named_parameter_occurrences",
        "raw_named_parameter_numel",
        "parameter_iterator_unique_numel",
        "independent_methods_agree",
        "alias_group_count",
        "alias_groups",
        "checkpoint_byte_size_used",
        "quantization_adjustment_used",
        "pruning_mask_adjustment_used",
        "inventory_sha256",
    }
)
_CHECKPOINT_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_reference",
        "boundary_reference",
        "next_layer_index",
        "completed_layer_indices",
        "completed_operator_kinds",
        "incoming_boundary",
        "incoming_state_bank_sha256",
        "target_topology_sha256",
        "full_policy_sha256",
        "origin_compiler_sha256",
        "k1_rejected_as_insufficient",
        "k3_not_opened",
        "audit_sha256",
    }
)
_FRONTIER_AUDIT_FIELDS = frozenset(
    {
        "frontier_sha256",
        "context_sha256",
        "next_layer_index",
        "expected_layer_count",
        "source_width",
        "target_width",
        "phase_count",
        "split_row_counts",
        "layer_operator_kinds",
        "cross_runtime_probe_only",
        "ordinary_resume_gate_weakened",
        "audit_sha256",
    }
)
_PHASE_AUDIT_FIELDS = frozenset(
    {
        "phase_count",
        "trajectory_count",
        "phases_per_trajectory",
        "ordered_phase_names",
        "split_phase_counts",
        "split_trajectory_counts",
        "split_query_rows",
        "complete_prefill_plus_eight_decode_phases",
        "audit_sha256",
    }
)
_PREFIX_INSTALL_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_reference",
        "target_topology_sha256",
        "blank_state_sha256_before",
        "installed_state_sha256_after",
        "tensor_count",
        "next_layer_index",
        "exact_inventory_verified",
        "exact_object_aliases_verified",
        "exact_storage_aliases_verified",
        "exact_shape_dtype_stride_offset_device_verified",
        "tensor_identities_preserved",
        "exact_prefix_tensor_bytes_installed",
        "loaders_used",
        "parent_evidence_tensors_copied",
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
        "promotion_claimed",
        "report_sha256",
    }
)
_CHART_TRANSPORT_BINDING_FIELDS = frozenset(
    {
        "enabled",
        "mode",
        "source_layer_charts_sha256",
        "compiler_input_layer_charts_sha256",
        "controller_report_sha256",
        "transport_report_sha256",
        "selection_used_by_controller_solver",
        "validation_rows_read_by_controller",
        "default_off_semantics_preserved",
        "controller_report",
        "binding_sha256",
    }
)
_SEQUENTIAL_TRANSPORT_BINDING_FIELDS = frozenset(
    {
        "enabled",
        "mode",
        "runtime_matches_independent_invocation_pin",
        "layer_call_only_diagnostic",
        "authenticated_prefix_full_policy_sha256",
        "authenticated_prefix_full_policy_unchanged",
        "prefix_runtime_policy_mutated",
        "binding_sha256",
    }
)
_SEQUENTIAL_TRANSPORT_AUDIT_FIELDS = frozenset(
    {
        "enabled",
        "policy_report_sha256",
        "candidate_selection_report_sha256",
        "selected_candidate_id",
        "selected_candidate_report_sha256",
        "selected_effective_boundary_charts_sha256",
        "selected_operator_transport_mlp_bundle_sha256",
        "source_boundary_charts_sha256",
        "source_boundary_charts_payload_sha256",
        "candidate_effective_boundary_charts_sha256",
        "candidate_operator_transport_mlp_bundle_sha256",
        "fit_only_candidate_local_transport_authenticated",
        "selection_selects_frozen_bundle_without_refit",
        "validation_rows_used_by_transport_solver",
        "winning_chart_used_for_final_metrics",
        "audit_sha256",
    }
)
_RESOURCE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "sample_interval_seconds",
        "sample_count",
        "maximum_process_rss_bytes",
        "maximum_cuda_bytes_per_device",
        "peak_vm_rss_bytes",
        "peak_vm_hwm_bytes",
        "peak_vm_swap_bytes",
        "peak_cuda_reserved_bytes",
        "samples",
        "rss_cap_passed",
        "cuda_cap_passed",
        "zero_swap_passed",
        "report_sha256",
    }
)
_RUNTIME_GUARD_FIELDS = frozenset(
    {
        "schema",
        "cuda_allocator_policy",
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
        "pruning_operations",
        "quantization_operations",
        "parent_module_forward_calls",
        "parent_root_forward_calls",
        "parent_inference_entrypoint_calls",
        "target_module_forward_calls",
        "target_root_forward_calls",
        "target_inference_entrypoint_calls",
        "training_steps",
        "ordinary_bfloat16_and_float32_casts_blocked",
        "scope_started",
        "parent_model_load_guarded",
        "parent_model_observer_started_immediately_after_load",
        "blank_target_synthesis_guarded",
        "target_model_observer_started_immediately_after_synthesis",
        "selective_provider_open_and_close_guarded",
        "source_tied_table_mapping_guarded",
        "layer_2_probe_guarded",
        "all_model_touching_operations_after_parent_load_guarded",
        "clean",
        "report_sha256",
    }
)
_PCG_AUDIT_BASE_FIELDS = frozenset(
    {
        "schema",
        "event_count",
        "multi_rhs_solve_count",
        "single_tensor_solve_count",
        "als_iteration_count",
        "expected_roles",
        "observed_roles",
        "events",
        "event_to_artifact_solve_bindings",
        "sequence_artifact_report_sha256",
        "sequence_candidate_frontier_sha256",
        "sequence_solve_report_sha256",
        "sequence_solve_report_inventory_sha256",
        "artifact_value_solve_report_sha256",
        "all_roles_unique",
        "all_event_report_hashes_unique",
        "all_observation_hashes_unique",
        "all_iterations_at_most",
        "all_final_relative_normal_residuals_at_most",
        "solver_results_or_arguments_modified",
        "audit_sha256",
    }
)
_PCG_RESTARTED_FIELDS = frozenset(
    {
        "legacy_solver_event_count",
        "restarted_solver_event_count",
        "original_lane_maximum_iterations",
        "restart_block_iterations",
        "hard_maximum_iterations",
        "exact_final_residual_gate",
        "all_extended_final_residuals_exactly_recomputed",
        "all_restart_schedules_authenticated",
        "all_matvec_counts_authenticated",
        "restarted_extended_solver_events_authenticated",
    }
)
_PCG_ROW_BLOCK_FIELDS = frozenset(
    {
        "legacy_solver_event_count",
        "restarted_solver_event_count",
        "continuous_row_block_solver_event_count",
        "original_lane_maximum_iterations",
        "row_block_hard_maximum_iterations",
        "exact_final_residual_gate",
        "all_extended_final_residuals_exactly_recomputed",
        "all_matvec_counts_authenticated",
        "all_continuous_row_block_lanes_authenticated",
        "all_row_block_preconditioner_receipts_authenticated",
        "continuous_row_block_solver_events_authenticated",
    }
)
_PCG_HIDDEN_PARTIAL_TRACE_FIELDS = frozenset(
    {
        "legacy_solver_event_count",
        "restarted_solver_event_count",
        "continuous_hidden_partial_trace_solver_event_count",
        "original_lane_maximum_iterations",
        "hidden_partial_trace_hard_maximum_iterations",
        "exact_final_residual_gate",
        "all_extended_final_residuals_exactly_recomputed",
        "all_matvec_counts_authenticated",
        "all_continuous_hidden_partial_trace_lanes_authenticated",
        "all_hidden_partial_trace_preconditioner_receipts_authenticated",
        "continuous_hidden_partial_trace_solver_events_authenticated",
        "hidden_partial_trace_pcg_execution_profile_sha256",
        "all_hidden_partial_trace_execution_profiles_authenticated",
        "stale_row_block_event_count",
    }
)
_ZERO_GUARD_FIELDS = (
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
    "pruning_operations",
    "quantization_operations",
    "parent_module_forward_calls",
    "parent_root_forward_calls",
    "parent_inference_entrypoint_calls",
    "target_root_forward_calls",
    "target_inference_entrypoint_calls",
    "training_steps",
)
_ZERO_PREFIX_FIELDS = (
    "loaders_used",
    "parent_evidence_tensors_copied",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "parameters_pruned",
    "parameters_quantized",
)
_ZERO_CONTROLLER_FIELDS = (
    "parent_model_forward_calls",
    "target_model_forward_calls",
    "target_layer_forward_calls",
    "training_steps",
    "optimizer_steps",
    "backward_calls",
    "parameters_pruned",
    "parameters_quantized",
)
_SEQUENCE_ARTIFACT_ZERO_EXECUTION_FIELDS = (
    "parent_model_forward_calls",
    "target_model_forward_calls",
    "target_layer_forward_calls",
    "generation_calls",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "parameters_pruned",
    "parameters_quantized",
    "autograd_graphs",
)
_RESOURCE_CUDA_DEVICE_KEYS = frozenset({"cuda:0", "cuda:1"})


class Lfm2CpcfV5RealAttentionCausalPairError(RuntimeError):
    """One causal-pair input failed authentication."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            "Causal-pair metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = {str(name): _plain(child) for name, child in value.items()}
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        value = [_plain(child) for child in value]
    return json.loads(_canonical_json(value).decode("ascii"))


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5RealAttentionCausalPairError(f"{name} must be a mapping.")
    return _plain(value)


def _sealed(
    value: Any,
    *,
    seal_field: str,
    name: str,
    exact_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    result = _mapping(value, name)
    if exact_fields is not None and set(result) != exact_fields:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has an unexpected field inventory."
        )
    body = dict(result)
    claimed = _digest(body.pop(seal_field, None), f"{name}.{seal_field}")
    if _sha256(body) != claimed:
        raise Lfm2CpcfV5RealAttentionCausalPairError(f"{name} is not self-sealed.")
    return result


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} must be a finite"
            + (" nonnegative" if nonnegative else "")
            + " number."
        )
    return float(value)


def _zero_integer_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    name: str,
) -> None:
    for field in fields:
        if type(value.get(field)) is not int or value[field] != 0:
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name}.{field} does not prove zero execution."
            )


def _authenticate_inventory(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="inventory_sha256",
        name=name,
        exact_fields=_INVENTORY_FIELDS,
    )
    if (
        report.get("method")
        != "unique_materialized_parameter_storage_offset_numel_identity"
        or report.get("independent_methods_agree") is not True
        or report.get("checkpoint_byte_size_used") is not False
        or report.get("quantization_adjustment_used") is not False
        or report.get("pruning_mask_adjustment_used") is not False
        or type(report.get("unique_parameters")) is not int
        or report["unique_parameters"] <= 0
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not an independent dense parameter inventory."
        )
    return report


def _authenticate_pins(
    value: Any,
    *,
    expected_transport: bool,
    name: str,
) -> dict[str, Any]:
    pins = _mapping(value, name)
    if set(pins) != _PIN_FIELDS:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has an unexpected field inventory."
        )
    if (
        not isinstance(pins.get("checkpoint_store_directory"), str)
        or not pins["checkpoint_store_directory"]
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name}.checkpoint_store_directory is invalid."
        )
    for field in (
        "checkpoint_key",
        "checkpoint_manifest_sha256",
        "expected_trace_sha256",
        "expected_runtime_compiler_implementation_sha256",
        "expected_sequence_source_sha256",
        *(field for field in _PIN_CONSTANTS if isinstance(_PIN_CONSTANTS[field], str)),
    ):
        _digest(pins.get(field), f"{name}.{field}")
    changed = [
        field
        for field, expected in _PIN_CONSTANTS.items()
        if pins.get(field) != expected
    ]
    if (
        changed
        or pins.get("expected_attention_sequential_transport_enabled")
        is not expected_transport
        or pins["expected_runtime_compiler_implementation_sha256"]
        == H776_ORIGIN_COMPILER_SHA256
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} differs from the exact H776 invocation"
            + ("" if not changed else ": " + ", ".join(changed))
            + "."
        )
    return pins


def _authenticate_runtime_implementation(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="implementation_sha256",
        name=name,
    )
    _zero_integer_fields(
        report,
        (
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
        ),
        name=name,
    )
    files = report.get("ordered_source_files")
    if (
        not isinstance(files, list)
        or not files
        or files != sorted(files, key=lambda item: item.get("module", ""))
        or any(
            not isinstance(item, dict) or set(item) != {"module", "filename", "sha256"}
            for item in files
        )
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has no deterministic source inventory."
        )
    for index, item in enumerate(files):
        _digest(item.get("sha256"), f"{name}.ordered_source_files[{index}]")
    return report


def _authenticate_checkpoint_audit(
    value: Any,
    *,
    pins: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="audit_sha256",
        name=name,
        exact_fields=_CHECKPOINT_AUDIT_FIELDS,
    )
    reference = report.get("checkpoint_reference")
    boundary = report.get("boundary_reference")
    if (
        report.get("schema") != LFM2_CPCF_V5_REAL_ATTENTION_CHECKPOINT_AUDIT_SCHEMA
        or reference
        != {
            "artifact_kind": "checkpoint",
            "artifact_key": pins["checkpoint_key"],
            "manifest_sha256": pins["checkpoint_manifest_sha256"],
        }
        or not isinstance(boundary, dict)
        or boundary.get("artifact_kind") != "boundary"
        or report.get("next_layer_index") != H776_LAYER_INDEX
        or report.get("completed_layer_indices") != [0, 1]
        or report.get("completed_operator_kinds") != ["short_conv", "short_conv"]
        or report.get("incoming_boundary") != "layer.001.post_layer"
        or report.get("target_topology_sha256") != H776_TARGET_TOPOLOGY_SHA256
        or report.get("full_policy_sha256") != H776_FULL_POLICY_SHA256
        or report.get("origin_compiler_sha256") != H776_ORIGIN_COMPILER_SHA256
        or report.get("k1_rejected_as_insufficient") is not True
        or report.get("k3_not_opened") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not the exact authenticated k=2 prefix."
        )
    _digest(report.get("incoming_state_bank_sha256"), f"{name}.incoming state")
    return report


def _authenticate_frontier_audit(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="audit_sha256",
        name=name,
        exact_fields=_FRONTIER_AUDIT_FIELDS,
    )
    rows = report.get("split_row_counts")
    if (
        report.get("next_layer_index") != H776_LAYER_INDEX
        or report.get("expected_layer_count") != H776_LAYER_COUNT
        or report.get("source_width") != H776_SOURCE_HIDDEN_SIZE
        or report.get("target_width") != H776_TARGET_HIDDEN_SIZE
        or report.get("phase_count") != H776_PHASE_COUNT
        or report.get("layer_operator_kinds") != list(H776_OPERATOR_KINDS)
        or report.get("cross_runtime_probe_only") is not True
        or report.get("ordinary_resume_gate_weakened") is not False
        or not isinstance(rows, dict)
        or rows.get("fit") != H776_FIT_QUERY_ROWS
        or rows.get("validation") != H776_VALIDATION_QUERY_ROWS
        or type(rows.get("selection")) is not int
        or rows["selection"] <= 0
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} differs from the H776 frontier."
        )
    for field in ("frontier_sha256", "context_sha256"):
        _digest(report.get(field), f"{name}.{field}")
    return report


def _authenticate_phase_audit(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="audit_sha256",
        name=name,
        exact_fields=_PHASE_AUDIT_FIELDS,
    )
    if (
        report.get("phase_count") != H776_PHASE_COUNT
        or report.get("trajectory_count") != H776_TRAJECTORY_COUNT
        or report.get("phases_per_trajectory") != len(_EXPECTED_PHASE_NAMES)
        or report.get("ordered_phase_names") != list(_EXPECTED_PHASE_NAMES)
        or report.get("split_phase_counts") != dict(H776_SPLIT_PHASE_COUNTS)
        or report.get("split_trajectory_counts") != dict(H776_SPLIT_TRAJECTORY_COUNTS)
        or report.get("split_query_rows", {}).get("fit") != H776_FIT_QUERY_ROWS
        or report.get("split_query_rows", {}).get("validation")
        != H776_VALIDATION_QUERY_ROWS
        or report.get("complete_prefill_plus_eight_decode_phases") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not the complete causal phase bank."
        )
    return report


def _authenticate_prefix_install(
    value: Any,
    *,
    checkpoint: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="report_sha256",
        name=name,
        exact_fields=_PREFIX_INSTALL_FIELDS,
    )
    required_true = (
        "exact_inventory_verified",
        "exact_object_aliases_verified",
        "exact_storage_aliases_verified",
        "exact_shape_dtype_stride_offset_device_verified",
        "tensor_identities_preserved",
        "exact_prefix_tensor_bytes_installed",
    )
    if (
        report.get("checkpoint_reference") != checkpoint["checkpoint_reference"]
        or report.get("target_topology_sha256") != H776_TARGET_TOPOLOGY_SHA256
        or report.get("blank_state_sha256_before") != H776_BLANK_STATE_SHA256
        or report.get("next_layer_index") != H776_LAYER_INDEX
        or any(report.get(field) is not True for field in required_true)
        or report.get("promotion_claimed") is not False
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not an exact k=2 prefix installation."
        )
    _zero_integer_fields(report, _ZERO_PREFIX_FIELDS, name=name)
    _digest(
        report.get("installed_state_sha256_after"),
        f"{name}.installed_state_sha256_after",
    )
    return report


def _authenticate_metric(
    value: Any,
    *,
    expected_width: int,
    name: str,
) -> dict[str, Any]:
    metric = _mapping(value, name)
    if (
        set(metric) != _METRIC_FIELDS
        or type(metric.get("rows")) is not int
        or metric["rows"] <= 0
        or metric.get("width") != expected_width
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has invalid endpoint geometry."
        )
    for field in ("rmse", "relative_rmse", "maximum_absolute_error"):
        _finite(metric.get(field), f"{name}.{field}", nonnegative=True)
    cosine = _finite(metric.get("cosine_similarity"), f"{name}.cosine")
    if not -1.0 <= cosine <= 1.0:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name}.cosine_similarity is outside [-1, 1]."
        )
    for field in ("prediction_sha256", "target_sha256"):
        _digest(metric.get(field), f"{name}.{field}")
    return metric


def _authenticate_validation_metrics(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="report_sha256",
        name=name,
        exact_fields=frozenset(
            {
                "split",
                "phase_count",
                "phase_sha256",
                "post_operator",
                "post_layer",
                "phase_metrics",
                "report_sha256",
            }
        ),
    )
    phases = report.get("phase_metrics")
    phase_seals = report.get("phase_sha256")
    if (
        report.get("split") != "validation"
        or report.get("phase_count") != H776_SPLIT_PHASE_COUNTS["validation"]
        or not isinstance(phase_seals, list)
        or len(phase_seals) != report["phase_count"]
        or len(set(phase_seals)) != len(phase_seals)
        or not isinstance(phases, dict)
        or set(phases) != set(_EXPECTED_PHASE_NAMES)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not the complete blinded validation split."
        )
    for index, seal in enumerate(phase_seals):
        _digest(seal, f"{name}.phase_sha256[{index}]")
    aggregate_operator = _authenticate_metric(
        report["post_operator"],
        expected_width=H776_TARGET_HIDDEN_SIZE,
        name=f"{name}.post_operator",
    )
    aggregate_layer = _authenticate_metric(
        report["post_layer"],
        expected_width=H776_TARGET_HIDDEN_SIZE,
        name=f"{name}.post_layer",
    )
    if (
        aggregate_operator["rows"] != H776_VALIDATION_QUERY_ROWS
        or aggregate_layer["rows"] != H776_VALIDATION_QUERY_ROWS
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} aggregate rows differ from the sealed validation bank."
        )
    grouped_seals = []
    operator_rows = 0
    layer_rows = 0
    for phase_name in _EXPECTED_PHASE_NAMES:
        phase = _sealed(
            phases[phase_name],
            seal_field="report_sha256",
            name=f"{name}.phase_metrics[{phase_name!r}]",
            exact_fields=frozenset(
                {
                    "phase",
                    "phase_count",
                    "phase_sha256",
                    "post_operator",
                    "post_layer",
                    "report_sha256",
                }
            ),
        )
        seals = phase.get("phase_sha256")
        if (
            phase.get("phase") != phase_name
            or phase.get("phase_count") != H776_SPLIT_TRAJECTORY_COUNTS["validation"]
            or not isinstance(seals, list)
            or len(seals) != phase["phase_count"]
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name} phase {phase_name!r} is incomplete."
            )
        grouped_seals.extend(seals)
        operator = _authenticate_metric(
            phase["post_operator"],
            expected_width=H776_TARGET_HIDDEN_SIZE,
            name=f"{name}.{phase_name}.post_operator",
        )
        layer = _authenticate_metric(
            phase["post_layer"],
            expected_width=H776_TARGET_HIDDEN_SIZE,
            name=f"{name}.{phase_name}.post_layer",
        )
        operator_rows += operator["rows"]
        layer_rows += layer["rows"]
    if (
        sorted(grouped_seals) != sorted(phase_seals)
        or operator_rows != aggregate_operator["rows"]
        or layer_rows != aggregate_layer["rows"]
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} phase metrics do not partition aggregate evidence."
        )
    return report


def _authenticate_solver_plan(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="report_sha256",
        name=name,
    )
    if (
        report.get("strategy") != "streamed_cross_head_statistics_pcg"
        or report.get("report_sha256") != H776_SEQUENCE_SOLVER_PLAN_SHA256
        or report.get("fit_rows") != H776_FIT_QUERY_ROWS
        or report.get("target_query_heads") != H776_TARGET_QUERY_HEADS
        or report.get("target_hidden_size") != H776_TARGET_HIDDEN_SIZE
        or report.get("target_output_size") != H776_TARGET_HIDDEN_SIZE
        or report.get("head_dim") != H776_HEAD_DIM
        or report.get("forbidden_value_coefficient_materialized") is not False
        or report.get("forbidden_value_dense_gram_materialized") is not False
        or report.get("complete_operator_dense_factorization_avoided") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not the pinned scalable H776 solve plan."
        )
    return report


def _authenticate_pcg_audit(
    value: Any,
    *,
    sequence_artifact: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    report = _sealed(value, seal_field="audit_sha256", name=name)
    events = report.get("events")
    bindings = report.get("event_to_artifact_solve_bindings")
    iterations = report.get("als_iteration_count")
    if (
        report.get("schema") != LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA
        or iterations not in {1, 2}
        or not isinstance(events, list)
        or not isinstance(bindings, list)
        or report.get("event_count") != len(events)
        or report.get("single_tensor_solve_count") != len(events)
        or report.get("multi_rhs_solve_count") != 0
        or len(events) != iterations * H776_TARGET_KEY_VALUE_HEADS
        or len(bindings) != len(events)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has an invalid PCG event inventory."
        )
    collector = Lfm2SequenceEffectivePcgCollector()
    try:
        for event in events:
            collector(event)
    except Exception as exc:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} contains unauthenticated PCG telemetry."
        ) from exc
    roles = [
        (f"sequence_effective_structured_value_iteration_{iteration}_group_{group}")
        for iteration in range(1, iterations + 1)
        for group in range(H776_TARGET_KEY_VALUE_HEADS)
    ]
    if (
        report.get("expected_roles") != roles
        or report.get("observed_roles") != roles
        or report.get("all_roles_unique") is not True
        or report.get("all_event_report_hashes_unique") is not True
        or report.get("all_observation_hashes_unique") is not True
        or report.get("all_final_relative_normal_residuals_at_most") != 1e-8
        or report.get("solver_results_or_arguments_modified") is not False
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove exact ordered PCG convergence."
        )
    outer_hashes = report.get("artifact_value_solve_report_sha256")
    solve_hashes = report.get("sequence_solve_report_sha256")
    if (
        not isinstance(outer_hashes, list)
        or len(outer_hashes) != len(events)
        or not isinstance(solve_hashes, list)
        or len(solve_hashes) != 2 + iterations * 6 + 1
        or report.get("sequence_solve_report_inventory_sha256") != _sha256(solve_hashes)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has an invalid artifact solve inventory."
        )
    expected_outer = []
    for iteration in range(iterations):
        expected_outer.extend(solve_hashes[2 + iteration * 6 : 7 + iteration * 6])
    if outer_hashes != expected_outer:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} PCG solves are not ordered within the ALS inventory."
        )
    for index, (event, binding, role, outer_hash) in enumerate(
        zip(events, bindings, roles, outer_hashes, strict=True)
    ):
        if (
            not isinstance(binding, dict)
            or binding.get("role") != role
            or binding.get("iteration") != index // 5 + 1
            or binding.get("group") != index % 5
            or binding.get("observation_sha256") != event.get("observation_sha256")
            or binding.get("pcg_report_sha256") != event.get("report_sha256")
            or binding.get("artifact_value_solve_report_sha256") != outer_hash
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name} event {index} is not cross-bound to its artifact solve."
            )
    solvers = [event["report"]["solver"] for event in events]
    restarted = sum(
        solver == "deterministic_restarted_jacobi_pcg_single_tensor"
        for solver in solvers
    )
    row_block = sum(
        solver == "deterministic_continuous_row_block_jacobi_pcg_single_tensor"
        for solver in solvers
    )
    hidden_partial_trace = sum(
        solver
        == ("deterministic_continuous_hidden_partial_trace_cholesky_pcg_single_tensor")
        for solver in solvers
    )
    legacy = len(solvers) - restarted - row_block - hidden_partial_trace
    maximum = 4096 if row_block or hidden_partial_trace else (512 if restarted else 64)
    expected_fields = set(_PCG_AUDIT_BASE_FIELDS)
    if restarted:
        expected_fields.update(_PCG_RESTARTED_FIELDS)
    if row_block:
        expected_fields.update(_PCG_ROW_BLOCK_FIELDS)
    if hidden_partial_trace:
        expected_fields.update(_PCG_HIDDEN_PARTIAL_TRACE_FIELDS)
    if (
        set(report) != expected_fields
        or report.get("all_iterations_at_most") != maximum
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} reports the wrong authenticated solver contract."
        )
    if restarted and (
        report.get("restarted_solver_event_count") != restarted
        or report.get("legacy_solver_event_count") != legacy
        or report.get("original_lane_maximum_iterations") != 64
        or report.get("restart_block_iterations") != 64
        or report.get("hard_maximum_iterations") != 512
        or report.get("exact_final_residual_gate") != 1e-8
        or report.get("all_extended_final_residuals_exactly_recomputed") is not True
        or report.get("all_restart_schedules_authenticated") is not True
        or report.get("all_matvec_counts_authenticated") is not True
        or report.get("restarted_extended_solver_events_authenticated") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} restarted lanes are not authenticated."
        )
    if row_block and (
        report.get("continuous_row_block_solver_event_count") != row_block
        or report.get("restarted_solver_event_count") != restarted
        or report.get("legacy_solver_event_count") != legacy
        or report.get("original_lane_maximum_iterations") != 64
        or report.get("row_block_hard_maximum_iterations") != 4096
        or report.get("exact_final_residual_gate") != 1e-8
        or report.get("all_extended_final_residuals_exactly_recomputed") is not True
        or report.get("all_matvec_counts_authenticated") is not True
        or report.get("all_continuous_row_block_lanes_authenticated") is not True
        or report.get("all_row_block_preconditioner_receipts_authenticated") is not True
        or report.get("continuous_row_block_solver_events_authenticated") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} row-block lanes are not authenticated."
        )
    if hidden_partial_trace and (
        report.get("continuous_hidden_partial_trace_solver_event_count")
        != hidden_partial_trace
        or report.get("restarted_solver_event_count") != restarted
        or report.get("legacy_solver_event_count") != legacy
        or report.get("original_lane_maximum_iterations") != 64
        or report.get("hidden_partial_trace_hard_maximum_iterations") != 4096
        or report.get("exact_final_residual_gate") != 1e-8
        or report.get("all_extended_final_residuals_exactly_recomputed") is not True
        or report.get("all_matvec_counts_authenticated") is not True
        or report.get("all_continuous_hidden_partial_trace_lanes_authenticated")
        is not True
        or report.get("all_hidden_partial_trace_preconditioner_receipts_authenticated")
        is not True
        or report.get("continuous_hidden_partial_trace_solver_events_authenticated")
        is not True
        or report.get("hidden_partial_trace_pcg_execution_profile_sha256")
        != H776_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
        or report.get("all_hidden_partial_trace_execution_profiles_authenticated")
        is not True
        or report.get("stale_row_block_event_count") != 0
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} hidden partial-trace lanes are not authenticated."
        )
    for field in (
        "sequence_artifact_report_sha256",
        "sequence_candidate_frontier_sha256",
        "sequence_solve_report_inventory_sha256",
    ):
        _digest(report.get(field), f"{name}.{field}")
    for index, digest in enumerate(solve_hashes):
        _digest(digest, f"{name}.sequence_solve_report_sha256[{index}]")
    replayed = Lfm2SequenceEffectivePcgCollector()
    try:
        for event in events:
            replayed(event)
        reproduced = _plain(replayed.finalize(sequence_artifact=sequence_artifact))
    except Exception as exc:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} cannot be reproduced from its sealed sequence artifact."
        ) from exc
    if reproduced != report:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} differs from collector.finalize(sequence_artifact=...)."
        )
    return report


def _authenticate_sequence_artifact(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    artifact = _sealed(
        value,
        seal_field="report_sha256",
        name=name,
    )
    frontier = artifact.get("candidate_frontier")
    solve_hashes = artifact.get("solve_report_sha256")
    solver_plan = artifact.get("solver_plan")
    execution = artifact.get("execution")
    producer_before = _digest(
        artifact.get("producer_state_sha256_before"),
        f"{name}.producer_state_sha256_before",
    )
    producer_after = _digest(
        artifact.get("producer_state_sha256_after"),
        f"{name}.producer_state_sha256_after",
    )
    if (
        artifact.get("schema") != LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA
        or artifact.get("candidate_id") != H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
        or artifact.get("fixed_query_key_state_reused_exactly") is not True
        or producer_before != producer_after
        or artifact.get("producer_state_unchanged") is not True
        or artifact.get("fit_only_compilation") is not True
        or artifact.get("compiled_from_splits") != ["fit"]
        or type(artifact.get("selection_case_count_consumed")) is not int
        or artifact["selection_case_count_consumed"] != 0
        or type(artifact.get("validation_case_count_consumed")) is not int
        or artifact["validation_case_count_consumed"] != 0
        or artifact.get("selection_inputs_accessible") is not False
        or artifact.get("validation_inputs_accessible") is not False
        or artifact.get("blank_target_values_read") is not False
        or artifact.get("dense_target_native_qkvo") is not True
        or artifact.get("ordinary_dense_target_value_output") is not True
        or artifact.get("forbidden_value_coefficient_tensor_materialized") is not False
        or artifact.get("forbidden_hidden_head_dim_gram_materialized") is not False
        or not isinstance(frontier, list)
        or not frontier
        or artifact.get("candidate_frontier_sha256") != _sha256(frontier)
        or type(artifact.get("bounded_proposal_count")) is not int
        or artifact["bounded_proposal_count"] != len(frontier)
        or type(artifact.get("selected_proposal_index")) is not int
        or artifact["selected_proposal_index"] != len(frontier) - 1
        or not isinstance(solve_hashes, list)
        or not isinstance(execution, dict)
        or set(execution) != set(_SEQUENCE_ARTIFACT_ZERO_EXECUTION_FIELDS)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not the sealed fit-only sequence-effective artifact."
        )
    _zero_integer_fields(
        execution,
        _SEQUENCE_ARTIFACT_ZERO_EXECUTION_FIELDS,
        name=f"{name}.execution",
    )
    _authenticate_solver_plan(solver_plan, name=f"{name}.solver_plan")
    return artifact


def _authenticate_behavioral_selection(
    report: Mapping[str, Any],
    *,
    name: str,
) -> tuple[dict[str, tuple[float, ...]], str]:
    candidate_ids = report.get("candidate_ids")
    selection = report.get("selection_keys")
    behavioral = report.get("behavioral_selection_keys")
    selected = report.get("selected_candidate_id")
    if (
        candidate_ids != list(H776_ATTENTION_CANDIDATE_IDS)
        or not isinstance(selection, dict)
        or not isinstance(behavioral, dict)
        or set(selection) != set(H776_ATTENTION_CANDIDATE_IDS)
        or set(behavioral) != set(H776_ATTENTION_CANDIDATE_IDS)
        or selected not in H776_ATTENTION_CANDIDATE_IDS
        or report.get("lexical_tie_break_excluded_from_strict_improvement") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has an invalid attention tournament."
        )
    normalized = {}
    full = {}
    for candidate_id in H776_ATTENTION_CANDIDATE_IDS:
        key = selection[candidate_id]
        numeric = behavioral[candidate_id]
        if (
            not isinstance(key, list)
            or len(key) != 11
            or key[-1] != candidate_id
            or not isinstance(numeric, list)
            or len(numeric) != 10
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name} candidate {candidate_id!r} has an invalid selection key."
            )
        values = tuple(
            _finite(value, f"{name}.{candidate_id}.selection_key") for value in key[:-1]
        )
        behavioral_values = tuple(
            _finite(value, f"{name}.{candidate_id}.behavioral_selection_key")
            for value in numeric
        )
        if values != behavioral_values:
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name} candidate {candidate_id!r} lexical exclusion changed."
            )
        normalized[candidate_id] = behavioral_values
        full[candidate_id] = (*values, candidate_id)
    if selected != min(full, key=full.get):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} selected candidate differs from its sealed key ordering."
        )
    sequence_won = selected == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
    sequence_strict = all(
        normalized[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID] < normalized[candidate_id]
        for candidate_id in H776_ATTENTION_CANDIDATE_IDS[:-1]
    )
    expected_outcome = (
        "positive_sequence_effective_selection"
        if sequence_won and sequence_strict
        else "honest_negative_or_control_selected"
    )
    if (
        report.get("sequence_effective_won") is not sequence_won
        or report.get("sequence_effective_strictly_better_than_both_controls")
        is not sequence_strict
        or report.get("scientific_outcome") != expected_outcome
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} precomputed selection outcome is inconsistent."
        )
    return normalized, selected


def _authenticate_sequential_transport_audit(
    value: Any,
    *,
    selected_id: str,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="audit_sha256",
        name=name,
        exact_fields=_SEQUENTIAL_TRANSPORT_AUDIT_FIELDS,
    )
    candidate_charts = report.get("candidate_effective_boundary_charts_sha256")
    candidate_bundles = report.get("candidate_operator_transport_mlp_bundle_sha256")
    if (
        report.get("enabled") is not True
        or report.get("selected_candidate_id") != selected_id
        or not isinstance(candidate_charts, dict)
        or set(candidate_charts) != set(H776_ATTENTION_CANDIDATE_IDS)
        or not isinstance(candidate_bundles, dict)
        or set(candidate_bundles) != set(H776_ATTENTION_CANDIDATE_IDS)
        or report.get("fit_only_candidate_local_transport_authenticated") is not True
        or report.get("selection_selects_frozen_bundle_without_refit") is not True
        or report.get("validation_rows_used_by_transport_solver") != 0
        or report.get("winning_chart_used_for_final_metrics") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove fit-only frozen candidate-local transport."
        )
    for candidate_id in H776_ATTENTION_CANDIDATE_IDS:
        _digest(
            candidate_charts.get(candidate_id),
            f"{name}.candidate_effective_boundary_charts_sha256[{candidate_id!r}]",
        )
        _digest(
            candidate_bundles.get(candidate_id),
            f"{name}.candidate_operator_transport_mlp_bundle_sha256[{candidate_id!r}]",
        )
    for field in (
        "policy_report_sha256",
        "candidate_selection_report_sha256",
        "selected_candidate_report_sha256",
        "selected_effective_boundary_charts_sha256",
        "selected_operator_transport_mlp_bundle_sha256",
        "source_boundary_charts_sha256",
        "source_boundary_charts_payload_sha256",
    ):
        _digest(report.get(field), f"{name}.{field}")
    if (
        candidate_charts[selected_id]
        != report["selected_effective_boundary_charts_sha256"]
        or candidate_bundles[selected_id]
        != report["selected_operator_transport_mlp_bundle_sha256"]
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} selected chart or bundle differs from its candidate map."
        )
    return report


def _authenticate_layer_result(
    value: Any,
    *,
    expected_transport: bool,
    incoming_state_bank_sha256: str,
    name: str,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[float, ...]],
    str,
    dict[str, Any],
]:
    report = _sealed(value, seal_field="audit_sha256", name=name)
    base_fields = {
        "layer_result_sha256",
        "input_bank_sha256",
        "next_bank_sha256",
        "candidate_ids",
        "selected_candidate_id",
        "selection_keys",
        "behavioral_selection_keys",
        "lexical_tie_break_excluded_from_strict_improvement",
        "sequence_effective_won",
        "sequence_effective_strictly_better_than_both_controls",
        "scientific_outcome",
        "full_validation_metrics",
        "sequence_effective_artifact",
        "sequence_solver_plan",
        "pcg_convergence",
        "in_memory_next_bank_created_by_layer_replay",
        "k3_checkpoint_published",
        "layer_3_compiled",
        "automatic_retry_or_retuning_performed",
        "audit_sha256",
    }
    if expected_transport:
        base_fields.add("attention_sequential_transport")
    if set(report) != base_fields:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} has an unexpected authenticated field inventory."
        )
    if (
        report.get("input_bank_sha256") != incoming_state_bank_sha256
        or report.get("in_memory_next_bank_created_by_layer_replay") is not True
        or report.get("k3_checkpoint_published") is not False
        or report.get("layer_3_compiled") is not False
        or report.get("automatic_retry_or_retuning_performed") is not False
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove one frozen layer-2 replay."
        )
    for field in (
        "layer_result_sha256",
        "input_bank_sha256",
        "next_bank_sha256",
    ):
        _digest(report.get(field), f"{name}.{field}")
    keys, selected = _authenticate_behavioral_selection(report, name=name)
    validation = _authenticate_validation_metrics(
        report.get("full_validation_metrics"),
        name=f"{name}.full_validation_metrics",
    )
    artifact = _authenticate_sequence_artifact(
        report.get("sequence_effective_artifact"),
        name=f"{name}.sequence_effective_artifact",
    )
    solver_plan = _authenticate_solver_plan(
        report.get("sequence_solver_plan"),
        name=f"{name}.sequence_solver_plan",
    )
    if artifact.get("solver_plan") != solver_plan:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} solver plan differs from its sealed sequence artifact."
        )
    _authenticate_pcg_audit(
        report.get("pcg_convergence"),
        sequence_artifact=artifact,
        name=f"{name}.pcg_convergence",
    )
    if expected_transport:
        _authenticate_sequential_transport_audit(
            report.get("attention_sequential_transport"),
            selected_id=selected,
            name=f"{name}.attention_sequential_transport",
        )
    return report, keys, selected, validation


def _authenticate_chart_transport_binding(
    value: Any,
    *,
    enabled: bool,
    name: str,
) -> dict[str, Any]:
    binding = _sealed(
        value,
        seal_field="binding_sha256",
        name=name,
        exact_fields=_CHART_TRANSPORT_BINDING_FIELDS,
    )
    controller = binding.get("controller_report")
    if not enabled:
        if (
            binding.get("enabled") is not False
            or binding.get("mode") != "off"
            or binding.get("source_layer_charts_sha256")
            != binding.get("compiler_input_layer_charts_sha256")
            or binding.get("controller_report_sha256") is not None
            or binding.get("transport_report_sha256") is not None
            or binding.get("selection_used_by_controller_solver") is not None
            or binding.get("validation_rows_read_by_controller") is not None
            or binding.get("default_off_semantics_preserved") is not True
            or controller is not None
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name} does not prove exact default-off semantics."
            )
        return binding
    report = _sealed(
        controller,
        seal_field="report_sha256",
        name=f"{name}.controller_report",
    )
    required_true = (
        "selection_read_after_transport_freeze",
        "fit_only_closed_form_orthogonal_procrustes",
        "source_bank_unchanged",
        "source_charts_unchanged",
        "source_evidence_unchanged",
        "post_layer_chart_same_object",
        "decoded_state_predictions_invariant",
        "feature_predictions_invariant",
        "forced_shadow_for_controlled_comparison",
    )
    required_false = (
        "selection_used_by_solver",
        "validation_used_by_solver",
        "numeric_parent_weight_matching_objective",
        "sparsity_introduced",
        "parent_sidecar_required",
        "automatic_retry_or_retuning_performed",
        "automatic_accept_reject_controller",
    )
    if (
        binding.get("enabled") is not True
        or binding.get("mode") != "fit_only_post_operator_orthogonal_chart_transport"
        or binding.get("source_layer_charts_sha256")
        == binding.get("compiler_input_layer_charts_sha256")
        or binding.get("controller_report_sha256") != report["report_sha256"]
        or binding.get("transport_report_sha256")
        != report.get("transport_report_sha256")
        or binding.get("selection_used_by_controller_solver") is not False
        or binding.get("validation_rows_read_by_controller") != 0
        or binding.get("default_off_semantics_preserved") is not False
        or report.get("component") != "fit_only_post_operator_chart_transport_shadow"
        or report.get("layer_index") != H776_LAYER_INDEX
        or report.get("operator_kind") != "full_attention"
        or report.get("source_width") != H776_SOURCE_HIDDEN_SIZE
        or report.get("target_width") != H776_TARGET_HIDDEN_SIZE
        or report.get("source_layer_charts_sha256")
        != binding.get("source_layer_charts_sha256")
        or report.get("transported_layer_charts_sha256")
        != binding.get("compiler_input_layer_charts_sha256")
        or report.get("validation_rows_read") != 0
        or any(report.get(field) is not True for field in required_true)
        or any(report.get(field) is not False for field in required_false)
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove frozen fit-only post-operator transport."
        )
    _zero_integer_fields(report, _ZERO_CONTROLLER_FIELDS, name=name)
    return binding


def _authenticate_sequential_transport_binding(
    value: Any,
    *,
    enabled: bool,
    full_policy_sha256: str,
    name: str,
) -> dict[str, Any]:
    binding = _sealed(
        value,
        seal_field="binding_sha256",
        name=name,
        exact_fields=_SEQUENTIAL_TRANSPORT_BINDING_FIELDS,
    )
    if (
        binding.get("enabled") is not enabled
        or binding.get("mode")
        != ("candidate_local_fit_only_post_layer_transport" if enabled else "off")
        or binding.get("runtime_matches_independent_invocation_pin") is not True
        or binding.get("layer_call_only_diagnostic") is not True
        or binding.get("authenticated_prefix_full_policy_sha256") != full_policy_sha256
        or binding.get("authenticated_prefix_full_policy_unchanged") is not True
        or binding.get("prefix_runtime_policy_mutated") is not False
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not match its independent invocation pin."
        )
    return binding


def _authenticate_resource_report(
    value: Any,
    *,
    enabled_chart_transport: bool,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="report_sha256",
        name=name,
        exact_fields=_RESOURCE_REPORT_FIELDS,
    )
    samples = report.get("samples")
    peaks = report.get("peak_cuda_reserved_bytes")
    if (
        report.get("schema") != LFM2_CPCF_V5_REAL_ATTENTION_RESOURCE_REPORT_SCHEMA
        or type(report.get("sample_interval_seconds")) is not float
        or report["sample_interval_seconds"] != 0.05
        or report.get("maximum_process_rss_bytes") != H776_MAXIMUM_PROCESS_RSS_BYTES
        or report.get("maximum_cuda_bytes_per_device")
        != H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
        or report.get("rss_cap_passed") is not True
        or report.get("cuda_cap_passed") is not True
        or report.get("zero_swap_passed") is not True
        or report.get("peak_vm_swap_bytes") != 0
        or not isinstance(samples, list)
        or not samples
        or report.get("sample_count") != len(samples)
        or not isinstance(peaks, dict)
        or set(peaks) != _RESOURCE_CUDA_DEVICE_KEYS
        or samples[0].get("label") != "enter"
        or samples[-1].get("label") != "exit"
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove bounded complete resource monitoring."
        )
    _finite(
        report.get("sample_interval_seconds"),
        f"{name}.sample_interval_seconds",
        nonnegative=True,
    )
    monotonic = []
    for index, sample in enumerate(samples):
        if (
            not isinstance(sample, dict)
            or set(sample)
            != {
                "label",
                "monotonic_ns",
                "process",
                "cuda_peak_reserved_bytes",
            }
            or not isinstance(sample.get("label"), str)
            or type(sample.get("monotonic_ns")) is not int
            or not isinstance(sample.get("process"), dict)
            or set(sample["process"]) != {"VmRSS", "VmHWM", "VmSwap"}
            or not isinstance(sample.get("cuda_peak_reserved_bytes"), dict)
            or set(sample["cuda_peak_reserved_bytes"]) != _RESOURCE_CUDA_DEVICE_KEYS
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name}.samples[{index}] is invalid."
            )
        monotonic.append(sample["monotonic_ns"])
        for field in ("VmRSS", "VmHWM", "VmSwap"):
            observed = sample["process"][field]
            if type(observed) is not int or observed < 0:
                raise Lfm2CpcfV5RealAttentionCausalPairError(
                    f"{name}.samples[{index}].process.{field} is invalid."
                )
        if (
            sample["process"]["VmRSS"] > H776_MAXIMUM_PROCESS_RSS_BYTES
            or sample["process"]["VmHWM"] > H776_MAXIMUM_PROCESS_RSS_BYTES
            or sample["process"]["VmSwap"] != 0
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                f"{name}.samples[{index}] exceeded the process resource gate."
            )
        for observed in sample["cuda_peak_reserved_bytes"].values():
            if (
                type(observed) is not int
                or observed < 0
                or observed > H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
            ):
                raise Lfm2CpcfV5RealAttentionCausalPairError(
                    f"{name}.samples[{index}] exceeded the CUDA gate."
                )
    if monotonic != sorted(monotonic) or len(set(monotonic)) != len(monotonic):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} sample chronology is invalid."
        )
    if not any(sample["label"] == "periodic" for sample in samples):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} lacks a required periodic resource sample."
        )
    if (
        report.get("peak_vm_rss_bytes")
        != max(sample["process"]["VmRSS"] for sample in samples)
        or report.get("peak_vm_hwm_bytes")
        != max(sample["process"]["VmHWM"] for sample in samples)
        or report.get("peak_vm_swap_bytes")
        != max(sample["process"]["VmSwap"] for sample in samples)
        or peaks
        != {
            device: max(
                sample["cuda_peak_reserved_bytes"][device] for sample in samples
            )
            for device in sorted(_RESOURCE_CUDA_DEVICE_KEYS)
        }
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} peak summary does not reproduce from its samples."
        )
    labels = [sample["label"] for sample in samples if sample["label"] != "periodic"]
    expected = [
        "enter",
        "authenticated_inputs_ready",
        "interface_recomputed_final_norm_released",
        "boundary_frontier_hydrated",
        "k2_prefix_installed",
    ]
    if enabled_chart_transport:
        expected.append("layer_2_post_operator_chart_transport_frozen")
    expected.extend(
        [
            "layer_2_evidence_and_charts_ready",
            "layer_2_compiled_and_validated",
            "layer_2_evidence_released",
            "exit",
        ]
    )
    if labels != expected:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} checkpoint coverage is incomplete."
        )
    return report


def _authenticate_runtime_guard(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    report = _sealed(
        value,
        seal_field="report_sha256",
        name=name,
        exact_fields=_RUNTIME_GUARD_FIELDS,
    )
    allocator = _sealed(
        report.get("cuda_allocator_policy"),
        seal_field="receipt_sha256",
        name=f"{name}.cuda_allocator_policy",
    )
    _zero_integer_fields(report, _ZERO_GUARD_FIELDS, name=name)
    if (
        report.get("clean") is not True
        or type(report.get("target_module_forward_calls")) is not int
        or report["target_module_forward_calls"] < 0
        or report.get("scope_started") != "immediately_after_parent_model_load"
        or report.get("parent_model_load_guarded") is not False
        or report.get("parent_model_observer_started_immediately_after_load")
        is not True
        or report.get("blank_target_synthesis_guarded") is not True
        or report.get("target_model_observer_started_immediately_after_synthesis")
        is not True
        or report.get("selective_provider_open_and_close_guarded") is not True
        or report.get("source_tied_table_mapping_guarded") is not True
        or report.get("layer_2_probe_guarded") is not True
        or report.get("all_model_touching_operations_after_parent_load_guarded")
        is not True
        or allocator.get("requested_maximum_cuda_bytes_per_device")
        != H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
        or allocator.get("logical_devices") != [0, 1]
        or allocator.get("all_devices_enforced") is not True
        or allocator.get("enforced_before_substantive_cuda_allocations") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove a clean fold-only runtime."
        )
    return report


def _authenticate_compaction(
    run: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _authenticate_inventory(
        run.get("parent_parameter_inventory"),
        name=f"{name}.parent_parameter_inventory",
    )
    target = _authenticate_inventory(
        run.get("target_parameter_inventory"),
        name=f"{name}.target_parameter_inventory",
    )
    structural = _sealed(
        run.get("structural_parameter_compaction"),
        seal_field="receipt_sha256",
        name=f"{name}.structural_parameter_compaction",
    )
    probe_compaction = probe.get("parameter_compaction")
    if (
        source.get("unique_parameters") != H776_PARENT_UNIQUE_PARAMETERS
        or target.get("unique_parameters") != H776_TARGET_UNIQUE_PARAMETERS
        or structural.get("source_inventory") != source
        or structural.get("target_inventory") != target
        or structural.get("source_unique_parameters") != H776_PARENT_UNIQUE_PARAMETERS
        or structural.get("target_unique_parameters") != H776_TARGET_UNIQUE_PARAMETERS
        or structural.get("planned_target_unique_parameters")
        != H776_TARGET_UNIQUE_PARAMETERS
        or structural.get("parameters_removed_from_topology")
        != H776_PARENT_UNIQUE_PARAMETERS - H776_TARGET_UNIQUE_PARAMETERS
        or structural.get("realized_compression_ratio")
        != H776_PARAMETER_COMPRESSION_RATIO
        or structural.get("promotion_ratio_met") is not True
        or structural.get("parameter_count_compaction_only") is not True
        or structural.get("checkpoint_byte_ratio_used") is not False
        or structural.get("quantization_used") is not False
        or structural.get("pruning_used") is not False
        or not isinstance(probe_compaction, dict)
        or probe_compaction.get("parent_unique_parameters")
        != H776_PARENT_UNIQUE_PARAMETERS
        or probe_compaction.get("target_unique_parameters")
        != H776_TARGET_UNIQUE_PARAMETERS
        or probe_compaction.get("compression_ratio") != H776_PARAMETER_COMPRESSION_RATIO
        or probe_compaction.get("parent_inventory_sha256") != source["inventory_sha256"]
        or probe_compaction.get("target_inventory_sha256") != target["inventory_sha256"]
        or probe_compaction.get("quantization_used") is not False
        or probe_compaction.get("pruning_used") is not False
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} does not prove exact dense 4x parameter compaction."
        )
    return source, target, structural


def _authenticate_stop_conditions(
    run: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    name: str,
) -> None:
    stop = probe.get("stop_conditions")
    required_true = ("exactly_one_layer_compiled",)
    required_false = (
        "next_layer_compilation_attempted",
        "k3_checkpoint_published",
        "child_checkpoint_saved",
        "production_publication_attempted",
        "ordinary_resume_gate_weakened",
        "parent_trace_recapture_used",
        "training_used",
        "optimizer_used",
        "backward_used",
        "pruning_used",
        "quantization_used",
        "automatic_retry_or_retuning_used",
    )
    if (
        not isinstance(stop, dict)
        or stop.get("compiled_layer_indices") != [H776_LAYER_INDEX]
        or any(stop.get(field) is not True for field in required_true)
        or any(stop.get(field) is not False for field in required_false)
        or probe.get("promotion_authorized") is not False
        or probe.get("accepted_child_artifact") is not False
        or probe.get("whole_model_intelligence_retention_proven") is not False
        or probe.get("non_promotable_probe_only") is not True
        or run.get("diagnostic_only") is not True
        or run.get("child_checkpoint_saved") is not False
        or run.get("publication_attempted") is not False
        or run.get("promotion_authorized") is not False
        or run.get("catalog_reauthenticated_after_provider_close") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} crossed a diagnostic-only or fold-only stop condition."
        )


@dataclass(frozen=True, slots=True)
class _AuthenticatedDiagnostic:
    report: Mapping[str, Any]
    probe: Mapping[str, Any]
    pins: Mapping[str, Any]
    runtime_guard: Mapping[str, Any]
    resource_report: Mapping[str, Any]
    checkpoint_audit: Mapping[str, Any]
    frontier_audit: Mapping[str, Any]
    phase_audit: Mapping[str, Any]
    prefix_install_report: Mapping[str, Any]
    layer_result: Mapping[str, Any]
    behavioral_keys: Mapping[str, tuple[float, ...]]
    selected_candidate_id: str
    validation_metrics: Mapping[str, Any]
    parent_inventory: Mapping[str, Any]
    target_inventory: Mapping[str, Any]
    structural_compaction: Mapping[str, Any]


def _validation_target_sha256_inventory(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "aggregate": {
            endpoint: validation[endpoint]["target_sha256"]
            for endpoint in ("post_operator", "post_layer")
        },
        "phase_metrics": {
            phase: {
                endpoint: validation["phase_metrics"][phase][endpoint]["target_sha256"]
                for endpoint in ("post_operator", "post_layer")
            }
            for phase in _EXPECTED_PHASE_NAMES
        },
    }


def _authenticate_diagnostic(
    value: Any,
    *,
    expected_chart_transport: bool,
    expected_sequential_transport: bool,
    name: str,
) -> _AuthenticatedDiagnostic:
    run = _sealed(
        value,
        seal_field="receipt_sha256",
        name=name,
        exact_fields=_RUN_FIELDS,
    )
    probe = _sealed(
        run.get("probe"),
        seal_field="report_sha256",
        name=f"{name}.probe",
        exact_fields=_PROBE_FIELDS,
    )
    if (
        run.get("schema") != LFM2_CPCF_V5_REAL_ATTENTION_PROBE_RUN_SCHEMA
        or probe.get("schema") != LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA
        or run.get("status") != probe.get("status")
        or run.get("post_operator_chart_transport_shadow_enabled")
        is not expected_chart_transport
        or run.get("attention_sequential_transport_enabled")
        is not expected_sequential_transport
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} is not the requested completed probe arm."
        )
    pins = _authenticate_pins(
        probe.get("pins"),
        expected_transport=expected_sequential_transport,
        name=f"{name}.probe.pins",
    )
    implementation = _authenticate_runtime_implementation(
        probe.get("runtime_compiler_implementation"),
        name=f"{name}.probe.runtime_compiler_implementation",
    )
    if (
        implementation.get("implementation_sha256")
        != pins["expected_runtime_compiler_implementation_sha256"]
        or probe.get("runtime_sequence_source_sha256")
        != pins["expected_sequence_source_sha256"]
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} runtime source differs from its independent pins."
        )
    runtime_guard = _authenticate_runtime_guard(
        run.get("runtime_guard"),
        name=f"{name}.runtime_guard",
    )
    checkpoint = _authenticate_checkpoint_audit(
        probe.get("checkpoint_audit"),
        pins=pins,
        name=f"{name}.probe.checkpoint_audit",
    )
    frontier = _authenticate_frontier_audit(
        probe.get("frontier_audit"),
        name=f"{name}.probe.frontier_audit",
    )
    phase = _authenticate_phase_audit(
        probe.get("phase_audit"),
        name=f"{name}.probe.phase_audit",
    )
    prefix = _authenticate_prefix_install(
        probe.get("prefix_install_report"),
        checkpoint=checkpoint,
        name=f"{name}.probe.prefix_install_report",
    )
    layer, keys, selected, validation = _authenticate_layer_result(
        probe.get("layer_2_result"),
        expected_transport=expected_sequential_transport,
        incoming_state_bank_sha256=checkpoint["incoming_state_bank_sha256"],
        name=f"{name}.probe.layer_2_result",
    )
    expected_status = (
        "completed_positive_sequence_effective_selection"
        if layer["sequence_effective_won"]
        and layer["sequence_effective_strictly_better_than_both_controls"]
        else "completed_honest_negative_or_control_selected"
    )
    if run.get("status") != expected_status:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} completion status differs from its authenticated result."
        )
    chart_binding = _authenticate_chart_transport_binding(
        probe.get("post_operator_chart_transport_shadow"),
        enabled=expected_chart_transport,
        name=f"{name}.probe.post_operator_chart_transport_shadow",
    )
    _authenticate_sequential_transport_binding(
        probe.get("attention_sequential_transport"),
        enabled=expected_sequential_transport,
        full_policy_sha256=pins["expected_full_policy_sha256"],
        name=f"{name}.probe.attention_sequential_transport",
    )
    if expected_sequential_transport and (
        layer["attention_sequential_transport"]["source_boundary_charts_sha256"]
        != chart_binding["compiler_input_layer_charts_sha256"]
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} candidate-local transport is not bound to the global chart."
        )
    resource = _authenticate_resource_report(
        probe.get("resource_report"),
        enabled_chart_transport=expected_chart_transport,
        name=f"{name}.probe.resource_report",
    )
    parent, target, structural = _authenticate_compaction(
        run,
        probe,
        name=name,
    )
    geometry = probe.get("target_geometry")
    expected_geometry = {
        "hidden_size": H776_TARGET_HIDDEN_SIZE,
        "intermediate_size": H776_TARGET_INTERMEDIATE_SIZE,
        "query_heads": H776_TARGET_QUERY_HEADS,
        "key_value_heads": H776_TARGET_KEY_VALUE_HEADS,
        "head_dim": H776_HEAD_DIM,
        "layer_count": H776_LAYER_COUNT,
        "layer_types": list(H776_LAYER_TYPES),
        "layer_2_attention_shapes": {
            "q_proj.weight": [640, 776],
            "k_proj.weight": [320, 776],
            "v_proj.weight": [320, 776],
            "out_proj.weight": [776, 640],
        },
    }
    local_io = run.get("local_parent_io_audit")
    if (
        geometry != expected_geometry
        or not isinstance(local_io, dict)
        or local_io.get("auto_loader_used") is not False
        or local_io.get("from_pretrained_used") is not False
        or local_io.get("hub_download_used") is not False
        or local_io.get("network_allowed") is not False
        or local_io.get("hf_hub_offline") is not True
        or local_io.get("transformers_offline") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} topology or local-only input contract changed."
        )
    _digest(
        run.get("catalog_preparation_receipt_sha256"),
        f"{name}.catalog_preparation_receipt_sha256",
    )
    source_inventory = probe.get("cross_runtime_source_inventory")
    if (
        not isinstance(source_inventory, dict)
        or source_inventory.get("checkpoint_origin_inventory_sha256")
        != H776_SOURCE_INVENTORY_SHA256
        or source_inventory.get("origin_digest_reproduction_required") is not False
        or source_inventory.get("live_inventory_unchanged_before_after") is not True
    ):
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            f"{name} source inventory contract changed."
        )
    _digest(
        source_inventory.get("live_process_inventory_sha256"),
        f"{name}.live_process_inventory_sha256",
    )
    _authenticate_stop_conditions(run, probe, name=name)
    return _AuthenticatedDiagnostic(
        report=MappingProxyType(run),
        probe=MappingProxyType(probe),
        pins=MappingProxyType(pins),
        runtime_guard=MappingProxyType(runtime_guard),
        resource_report=MappingProxyType(resource),
        checkpoint_audit=MappingProxyType(checkpoint),
        frontier_audit=MappingProxyType(frontier),
        phase_audit=MappingProxyType(phase),
        prefix_install_report=MappingProxyType(prefix),
        layer_result=MappingProxyType(layer),
        behavioral_keys=MappingProxyType(keys),
        selected_candidate_id=selected,
        validation_metrics=MappingProxyType(validation),
        parent_inventory=MappingProxyType(parent),
        target_inventory=MappingProxyType(target),
        structural_compaction=MappingProxyType(structural),
    )


def _common_pair_contract(
    baseline: _AuthenticatedDiagnostic,
    combined: _AuthenticatedDiagnostic,
) -> Mapping[str, Any]:
    baseline_pins = dict(baseline.pins)
    combined_pins = dict(combined.pins)
    baseline_pins.pop("expected_attention_sequential_transport_enabled")
    combined_pins.pop("expected_attention_sequential_transport_enabled")
    baseline_validation_targets = _validation_target_sha256_inventory(
        baseline.validation_metrics
    )
    combined_validation_targets = _validation_target_sha256_inventory(
        combined.validation_metrics
    )
    comparisons = {
        "all_nonintervention_pins_equal": baseline_pins == combined_pins,
        "runtime_compiler_implementation_equal": (
            baseline.probe["runtime_compiler_implementation"]
            == combined.probe["runtime_compiler_implementation"]
        ),
        "runtime_sequence_source_equal": (
            baseline.probe["runtime_sequence_source_sha256"]
            == combined.probe["runtime_sequence_source_sha256"]
        ),
        "target_geometry_equal": (
            baseline.probe["target_geometry"] == combined.probe["target_geometry"]
        ),
        "parent_parameter_inventory_equal": (
            baseline.parent_inventory == combined.parent_inventory
        ),
        "target_parameter_inventory_equal": (
            baseline.target_inventory == combined.target_inventory
        ),
        "structural_compaction_equal": (
            baseline.structural_compaction == combined.structural_compaction
        ),
        "catalog_preparation_receipt_equal": (
            baseline.report["catalog_preparation_receipt_sha256"]
            == combined.report["catalog_preparation_receipt_sha256"]
        ),
        "checkpoint_audit_equal": (
            baseline.checkpoint_audit == combined.checkpoint_audit
        ),
        "frontier_audit_equal": baseline.frontier_audit == combined.frontier_audit,
        "phase_audit_equal": baseline.phase_audit == combined.phase_audit,
        "prefix_install_report_equal": (
            baseline.prefix_install_report == combined.prefix_install_report
        ),
        "layer_input_bank_equal": (
            baseline.layer_result["input_bank_sha256"]
            == combined.layer_result["input_bank_sha256"]
        ),
        "validation_phase_evidence_equal": (
            baseline.validation_metrics["phase_sha256"]
            == combined.validation_metrics["phase_sha256"]
            and all(
                baseline.validation_metrics["phase_metrics"][phase]["phase_sha256"]
                == combined.validation_metrics["phase_metrics"][phase]["phase_sha256"]
                for phase in _EXPECTED_PHASE_NAMES
            )
        ),
        "validation_target_sha256_inventory_equal": (
            baseline_validation_targets == combined_validation_targets
        ),
        "runtime_allocator_policy_equal": (
            baseline.runtime_guard["cuda_allocator_policy"]
            == combined.runtime_guard["cuda_allocator_policy"]
        ),
        "target_module_forward_counter_equal": (
            baseline.runtime_guard["target_module_forward_calls"]
            == combined.runtime_guard["target_module_forward_calls"]
        ),
        "resource_policy_equal": (
            baseline.resource_report["sample_interval_seconds"]
            == combined.resource_report["sample_interval_seconds"]
            and baseline.resource_report["maximum_process_rss_bytes"]
            == combined.resource_report["maximum_process_rss_bytes"]
            and baseline.resource_report["maximum_cuda_bytes_per_device"]
            == combined.resource_report["maximum_cuda_bytes_per_device"]
            and set(baseline.resource_report["peak_cuda_reserved_bytes"])
            == set(combined.resource_report["peak_cuda_reserved_bytes"])
        ),
    }
    failed = [name for name, passed in comparisons.items() if not passed]
    if failed:
        raise Lfm2CpcfV5RealAttentionCausalPairError(
            "Baseline and combined diagnostics are not one causal pair: "
            + ", ".join(failed)
            + "."
        )
    body = {
        **comparisons,
        "checkpoint_key": baseline.pins["checkpoint_key"],
        "checkpoint_manifest_sha256": (baseline.pins["checkpoint_manifest_sha256"]),
        "trace_sha256": baseline.pins["expected_trace_sha256"],
        "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        "target_plan_fingerprint": H776_TARGET_PLAN_FINGERPRINT,
        "validation_phase_inventory_sha256": _sha256(
            baseline.validation_metrics["phase_sha256"]
        ),
        "validation_target_sha256_inventory": baseline_validation_targets,
        "validation_target_sha256_inventory_sha256": _sha256(
            baseline_validation_targets
        ),
        "parameter_compression_ratio": H776_PARAMETER_COMPRESSION_RATIO,
        "no_training_pruning_quantization_or_publication": True,
    }
    return MappingProxyType({**body, "contract_sha256": _sha256(body)})


def _post_layer_metric(
    validation: Mapping[str, Any],
    phase: str | None = None,
) -> tuple[float, float]:
    source = (
        validation["post_layer"]
        if phase is None
        else validation["phase_metrics"][phase]["post_layer"]
    )
    return (
        float(source["relative_rmse"]),
        float(source["cosine_similarity"]),
    )


def _validation_comparison(
    baseline: Mapping[str, Any],
    combined: Mapping[str, Any],
) -> Mapping[str, Any]:
    baseline_by_phase = {
        phase: _post_layer_metric(baseline, phase) for phase in _EXPECTED_PHASE_NAMES
    }
    combined_by_phase = {
        phase: _post_layer_metric(combined, phase) for phase in _EXPECTED_PHASE_NAMES
    }
    baseline_aggregate = _post_layer_metric(baseline)
    combined_aggregate = _post_layer_metric(combined)
    baseline_decode = baseline_by_phase["decode.007"]
    combined_decode = combined_by_phase["decode.007"]
    baseline_worst_rmse = max(value[0] for value in baseline_by_phase.values())
    combined_worst_rmse = max(value[0] for value in combined_by_phase.values())
    baseline_min_cosine = min(value[1] for value in baseline_by_phase.values())
    combined_min_cosine = min(value[1] for value in combined_by_phase.values())
    per_phase = {
        phase: {
            "baseline_relative_rmse": baseline_by_phase[phase][0],
            "combined_relative_rmse": combined_by_phase[phase][0],
            "relative_rmse_delta": (
                combined_by_phase[phase][0] - baseline_by_phase[phase][0]
            ),
            "relative_rmse_nonregression": (
                combined_by_phase[phase][0] <= baseline_by_phase[phase][0]
            ),
            "baseline_cosine_similarity": baseline_by_phase[phase][1],
            "combined_cosine_similarity": combined_by_phase[phase][1],
            "cosine_similarity_delta": (
                combined_by_phase[phase][1] - baseline_by_phase[phase][1]
            ),
            "cosine_similarity_nonregression": (
                combined_by_phase[phase][1] >= baseline_by_phase[phase][1]
            ),
        }
        for phase in _EXPECTED_PHASE_NAMES
    }
    gates = {
        "aggregate_relative_rmse_strict_improvement": (
            combined_aggregate[0] < baseline_aggregate[0]
        ),
        "aggregate_cosine_strict_improvement": (
            combined_aggregate[1] > baseline_aggregate[1]
        ),
        "decode_007_relative_rmse_strict_improvement": (
            combined_decode[0] < baseline_decode[0]
        ),
        "decode_007_cosine_strict_improvement": (
            combined_decode[1] > baseline_decode[1]
        ),
        "worst_phase_relative_rmse_strict_improvement": (
            combined_worst_rmse < baseline_worst_rmse
        ),
        "minimum_phase_cosine_strict_improvement": (
            combined_min_cosine > baseline_min_cosine
        ),
        "every_phase_relative_rmse_nonregression": all(
            value["relative_rmse_nonregression"] for value in per_phase.values()
        ),
        "every_phase_cosine_nonregression": all(
            value["cosine_similarity_nonregression"] for value in per_phase.values()
        ),
    }
    body = {
        "metric_scope": "blinded_validation_post_layer",
        "phase_order": list(_EXPECTED_PHASE_NAMES),
        "baseline": {
            "aggregate_relative_rmse": baseline_aggregate[0],
            "aggregate_cosine_similarity": baseline_aggregate[1],
            "decode_007_relative_rmse": baseline_decode[0],
            "decode_007_cosine_similarity": baseline_decode[1],
            "worst_phase_relative_rmse": baseline_worst_rmse,
            "minimum_phase_cosine_similarity": baseline_min_cosine,
        },
        "combined": {
            "aggregate_relative_rmse": combined_aggregate[0],
            "aggregate_cosine_similarity": combined_aggregate[1],
            "decode_007_relative_rmse": combined_decode[0],
            "decode_007_cosine_similarity": combined_decode[1],
            "worst_phase_relative_rmse": combined_worst_rmse,
            "minimum_phase_cosine_similarity": combined_min_cosine,
        },
        "per_phase": per_phase,
        "gates": gates,
        "strict_improvement_uses_exact_finite_reported_values": True,
        "nonregression_uses_exact_pareto_inequalities": True,
    }
    return MappingProxyType({**body, "comparison_sha256": _sha256(body)})


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RealAttentionCausalPairVerdict:
    """One immutable, self-authenticating causal-pair verdict."""

    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        authenticated = _sealed(
            self.report,
            seal_field="report_sha256",
            name="causal-pair verdict",
        )
        if (
            authenticated.get("schema")
            != LFM2_CPCF_V5_REAL_ATTENTION_CAUSAL_PAIR_VERDICT_SCHEMA
            or type(authenticated.get("passed")) is not bool
            or authenticated.get("verdict")
            != (
                "positive_residual_attention_causal_improvement"
                if authenticated["passed"]
                else "authenticated_pair_failed_scientific_gates"
            )
        ):
            raise Lfm2CpcfV5RealAttentionCausalPairError(
                "Causal-pair verdict type or status is invalid."
            )
        object.__setattr__(self, "report", MappingProxyType(authenticated))

    @property
    def passed(self) -> bool:
        return bool(self.report["passed"])

    def as_dict(self) -> dict[str, Any]:
        return _plain(self.report)


def verify_lfm2_cpcf_v5_real_attention_causal_pair(
    baseline_diagnostic: Mapping[str, Any],
    combined_diagnostic: Mapping[str, Any],
) -> Lfm2CpcfV5RealAttentionCausalPairVerdict:
    """Authenticate and score the frozen off/off versus on/on H776 pair."""

    baseline = _authenticate_diagnostic(
        baseline_diagnostic,
        expected_chart_transport=False,
        expected_sequential_transport=False,
        name="baseline_diagnostic",
    )
    combined = _authenticate_diagnostic(
        combined_diagnostic,
        expected_chart_transport=True,
        expected_sequential_transport=True,
        name="combined_diagnostic",
    )
    common = _common_pair_contract(baseline, combined)
    validation = _validation_comparison(
        baseline.validation_metrics,
        combined.validation_metrics,
    )
    baseline_winning_key = baseline.behavioral_keys[baseline.selected_candidate_id]
    combined_winning_key = combined.behavioral_keys[combined.selected_candidate_id]
    behavioral = {
        "key_order": [
            "final_causal_phase_post_layer_relative_rmse",
            "maximum_consecutive_causal_phase_relative_rmse_drift",
            "worst_causal_phase_post_layer_relative_rmse",
            "aggregate_post_layer_relative_rmse",
            "negative_final_causal_phase_post_layer_cosine_similarity",
            "negative_worst_causal_phase_post_layer_cosine_similarity",
            "selection_context_relative_rmse",
            "negative_selection_context_cosine_similarity",
            "query_key_js_divergence_worst",
            "negative_query_key_top1_agreement_fraction",
        ],
        "candidate_id_tie_break_excluded": True,
        "baseline_selected_candidate_id": baseline.selected_candidate_id,
        "combined_selected_candidate_id": combined.selected_candidate_id,
        "baseline_winning_behavioral_key": list(baseline_winning_key),
        "combined_winning_behavioral_key": list(combined_winning_key),
        "combined_winning_key_strictly_improves_baseline": (
            combined_winning_key < baseline_winning_key
        ),
        "combined_sequence_candidate_won": (
            combined.selected_candidate_id == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
        ),
        "combined_sequence_candidate_strictly_better_than_controls": (
            combined.layer_result[
                "sequence_effective_strictly_better_than_both_controls"
            ]
        ),
        "same_candidate_sequence_key_diagnostic": {
            "baseline": list(
                baseline.behavioral_keys[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]
            ),
            "combined": list(
                combined.behavioral_keys[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]
            ),
            "combined_strictly_improves_baseline": (
                combined.behavioral_keys[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]
                < baseline.behavioral_keys[H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]
            ),
        },
    }
    gates = {
        "combined_sequence_candidate_won_strictly": (
            behavioral["combined_sequence_candidate_won"]
            and behavioral["combined_sequence_candidate_strictly_better_than_controls"]
        ),
        "combined_behavioral_selection_key_strictly_improved": (
            behavioral["combined_winning_key_strictly_improves_baseline"]
        ),
        **dict(validation["gates"]),
    }
    passed = all(gates.values())
    body = {
        "schema": LFM2_CPCF_V5_REAL_ATTENTION_CAUSAL_PAIR_VERDICT_SCHEMA,
        "verdict": (
            "positive_residual_attention_causal_improvement"
            if passed
            else "authenticated_pair_failed_scientific_gates"
        ),
        "passed": passed,
        "baseline_receipt_sha256": baseline.report["receipt_sha256"],
        "combined_receipt_sha256": combined.report["receipt_sha256"],
        "common_pair_contract": dict(common),
        "transport_intervention": {
            "baseline_post_operator_chart_transport": False,
            "baseline_candidate_local_sequential_transport": False,
            "combined_post_operator_chart_transport": True,
            "combined_candidate_local_sequential_transport": True,
            "combined_post_operator_fit_only_frozen": True,
            "combined_candidate_local_fit_only_frozen": True,
            "selection_refit_or_transport_mutation": False,
            "validation_rows_used_by_transport_solvers": 0,
            "automatic_retry_or_retuning_performed": False,
        },
        "behavioral_selection_comparison": behavioral,
        "blinded_validation_comparison": dict(validation),
        "gates": gates,
        "model_or_diagnostic_mutation_performed": False,
        "training_or_optimization_performed": False,
        "pruning_or_quantization_performed": False,
        "validation_driven_tuning_performed": False,
        "publication_or_child_checkpoint_performed": False,
    }
    return Lfm2CpcfV5RealAttentionCausalPairVerdict(
        {**body, "report_sha256": _sha256(body)}
    )


__all__ = [
    "LFM2_CPCF_V5_REAL_ATTENTION_CAUSAL_PAIR_VERDICT_SCHEMA",
    "Lfm2CpcfV5RealAttentionCausalPairError",
    "Lfm2CpcfV5RealAttentionCausalPairVerdict",
    "verify_lfm2_cpcf_v5_real_attention_causal_pair",
]
