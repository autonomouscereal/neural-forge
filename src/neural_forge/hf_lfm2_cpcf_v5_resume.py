"""Authenticated partial-frontier resume core for selective LFM2 CPCF v5.

The v5 compiler fits boundary charts one layer at a time, so the complete v4
boundary bundle does not exist at an intermediate continuation point.  This
module persists the smaller exact object that *does* exist:

* the installed tied interface and its lineage;
* the contiguous chart prefix ``range(next_layer_index)``;
* the exact source-observation hashes used to fit that prefix;
* the actual folded child state bank at the next layer; and
* a complete target-prefix snapshot in the existing immutable CPCF store.

Only child tensors are archived.  Parent/cache tensors remain addressed by
their authenticated scalar identities.  Publication, restoration, and
admission perform no model execution, training, pruning, or quantization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import (
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_closed_form_causal_circuit_fold import (
    PredictiveBoundaryMap,
)
from neural_forge.hf_cpcf_boundary_chart import CpcfBoundaryChart
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    Lfm2CpcfCompletedLayerReceipt,
)
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    Lfm2CpcfArtifactReference,
    Lfm2CpcfBoundaryBindings,
    Lfm2CpcfResumeArtifactIntegrityError,
    Lfm2CpcfResumeArtifactStore,
    VerifiedLfm2CpcfPrefixCheckpoint,
    build_lfm2_cpcf_checkpoint_bindings,
    build_lfm2_cpcf_completed_layers_document,
    build_lfm2_cpcf_installed_parameter_paths,
    build_lfm2_cpcf_target_topology_snapshot,
    flatten_lfm2_cpcf_incoming_state_bank,
)
from neural_forge.hf_lfm2_cpcf_resume_bridge import (
    install_lfm2_cpcf_prefix_checkpoint,
    reconstruct_lfm2_cpcf_phase_state_bank,
)
from neural_forge import hf_lfm2_cpcf_model_fold as model_fold
from neural_forge import hf_lfm2_cpcf_resume_artifact as resume_artifact
from neural_forge import hf_lfm2_cpcf_resume_bridge as resume_bridge

LFM2_CPCF_V5_RESUME_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-resume-v1"
LFM2_CPCF_V5_BOUNDARY_FRONTIER_SCHEMA = "neural-forge-lfm2-cpcf-v5-boundary-frontier-v1"
LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA = "neural-forge-lfm2-cpcf-v5-progress-event-v1"
LFM2_CPCF_V5_ADMISSION_SCHEMA = "neural-forge-lfm2-cpcf-v5-prefix-admission-v1"
LFM2_CPCF_V5_REJECTION_SCHEMA = "neural-forge-lfm2-cpcf-v5-prefix-rss-rejection-v1"
LFM2_CPCF_V5_PENDING_SCHEMA = "neural-forge-lfm2-cpcf-v5-pending-checkpoint-v1"
LFM2_CPCF_V5_TERMINAL_SCHEMA = "neural-forge-lfm2-cpcf-v5-terminal-outcome-v1"
LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-physical-rss-checkpoint-v1"
)
LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-physical-rss-expectation-v1"
)
LFM2_CPCF_V5_PHYSICAL_RSS_POLICY_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-physical-rss-policy-v1"
)
LFM2_CPCF_V5_RESUME_IMPLEMENTATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-resume-implementation-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPLITS = ("selection", "fit", "validation")
_BOUNDARIES = ("residual.post_operator", "residual.post_layer")
_V5_DOCUMENT_COLLECTIONS = frozenset({"v5-progress", "v5-pending", "v5-terminals"})
_V5_DOCUMENT_TEMP = re.compile(r"\.v5-document-[0-9a-f]{64}\.tmp\Z")
_V5_TERMINAL_STAGE = re.compile(r"\.v5-terminal-[0-9a-f]{64}\.tmp\Z")
_RSS_EXPECTATION_FIELDS = frozenset(
    {
        "schema",
        "label",
        "rss_guard_id_sha256",
        "checkpoint_ordinal",
        "previous_checkpoint_report_sha256",
        "start_rss_bytes",
        "start_lifetime_peak_bytes",
        "full_phase_tree_logical_bytes",
        "minimum_reduction_fraction",
        "relative_maximum_increase_bytes",
        "maximum_process_rss_bytes",
        "absolute_headroom_bytes",
        "maximum_increase_bytes",
        "expectation_sha256",
    }
)
_RSS_CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "label",
        "start_rss_bytes",
        "current_rss_bytes",
        "peak_rss_bytes",
        "start_lifetime_peak_bytes",
        "current_lifetime_peak_bytes",
        "current_increase_bytes",
        "peak_current_rss_increase_bytes",
        "lifetime_peak_increase_bytes",
        "peak_increase_bytes",
        "full_phase_tree_logical_bytes",
        "minimum_reduction_fraction",
        "relative_maximum_increase_bytes",
        "maximum_process_rss_bytes",
        "absolute_headroom_bytes",
        "maximum_increase_bytes",
        "rss_guard_id_sha256",
        "checkpoint_ordinal",
        "previous_checkpoint_report_sha256",
        "physical_rss_gate_passed",
        "report_sha256",
    }
)
_PROGRESS_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "frontier_sha256",
        "context_sha256",
        "next_layer_index",
        "incoming_state_bank_sha256",
        "completed_layer_result_sha256",
        "target_prefix_snapshot_receipt",
        "pending_rss_admission",
        "rss_admission_claimed",
        "parent_or_cache_tensors_archived",
        "operation_audit",
        "report_sha256",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "pending_sha256",
        "outcome_kind",
        "outcome_sha256",
        "outcome_filename",
        "exactly_one_terminal_outcome",
        "operation_audit",
        "terminal_sha256",
    }
)
_ADMISSION_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "pending_sha256",
        "frontier_reference",
        "checkpoint_reference",
        "frontier_sha256",
        "event_report_sha256",
        "next_layer_index",
        "rss_checkpoint_expectation_sha256",
        "rss_checkpoint_policy_binding",
        "rss_checkpoint_receipt",
        "rss_checkpoint_report_sha256",
        "pending_checkpoint_reauthenticated",
        "persisted_progress_event_recomputed",
        "original_policy_admitted",
        "readmission_used",
        "promotion_claimed",
        "operation_audit",
        "admission_sha256",
    }
)
_REJECTION_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "pending_sha256",
        "frontier_reference",
        "checkpoint_reference",
        "frontier_sha256",
        "event_report_sha256",
        "next_layer_index",
        "rss_checkpoint_expectation_sha256",
        "rss_checkpoint_policy_binding",
        "rss_violation_report",
        "rss_violation_report_sha256",
        "pending_checkpoint_reauthenticated",
        "persisted_progress_event_recomputed",
        "original_policy_admitted",
        "promotable",
        "operation_audit",
        "rejection_sha256",
    }
)
_FRONTIER_FIELDS = frozenset(
    {
        "schema",
        "resume_schema",
        "context",
        "next_layer_index",
        "expected_layer_count",
        "source_width",
        "target_width",
        "ordered_phase_sha256",
        "split_phase_sha256",
        "split_row_sha256",
        "split_row_counts",
        "layer_operator_kinds",
        "interface",
        "completed_layer_charts",
        "source_observations",
        "phase_weighting",
        "operation_audit",
        "tensor_inventory",
        "frontier_sha256",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "schema",
        "source_model_state_sha256",
        "source_model_inventory_sha256",
        "cache_sha256",
        "selective_catalog_sha256",
        "selective_source_sha256",
        "selective_provider_sha256",
        "trace_sha256",
        "tokenizer_sha256",
        "token_bridge_sha256",
        "source_tied_table_sha256",
        "blank_target_report_sha256",
        "blank_target_state_sha256",
        "target_topology_sha256",
        "planned_target_unique_parameters",
        "runtime_policy_sha256",
        "physical_rss_policy_sha256",
        "compiler_implementation_sha256",
        "resume_implementation_sha256",
        "selector_contract_sha256",
        "numerical_contract_sha256",
        "gpu_contract_sha256",
        "bootstrap_authentication_sha256",
        "final_norm_authentication_sha256",
        "context_sha256",
    }
)
_INTERFACE_FIELDS = frozenset(
    {
        "target_weight_tensor",
        "target_weight_sha256",
        "producer_encoder_sha256",
        "fit_future_sha256",
        "validation_future_sha256",
        "active_interface_report",
        "base_interface_report",
        "causal_anchor_report",
        "anchor_split_reports",
    }
)
_PHASE_WEIGHTING_FIELDS = frozenset(
    {
        "fit_weights_sha256",
        "validation_weights_sha256",
        "fit_weight_summary",
        "validation_weight_summary",
    }
)
_ZERO_AUDIT = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "generation_calls": 0,
    "parent_cache_generation_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "fallback_paths": 0,
}


class Lfm2CpcfV5ResumeError(Lfm2CpcfResumeArtifactIntegrityError):
    """A v5 continuation object failed exact authentication."""


_EXPECTED_RSS_GUARD_MODULE = (
    "neural_forge.hf_lfm2_cpcf_v5_model_compiler"
)
_EXPECTED_RSS_GUARD_QUALNAME = "_PhysicalRssGuard"


def _live_rss_guard_authorizer(
    guard: Any,
    *,
    method_name: str,
) -> Any:
    guard_type = type(guard)
    method = getattr(guard, method_name, None)
    if (
        guard_type.__module__ != _EXPECTED_RSS_GUARD_MODULE
        or guard_type.__qualname__ != _EXPECTED_RSS_GUARD_QUALNAME
        or not callable(method)
    ):
        raise Lfm2CpcfV5ResumeError(
            "RSS authority must be the exact live CPCF-v5 physical RSS guard."
        )
    return method


class Lfm2CpcfV5PhysicalRssCheckpointExpectation(Mapping[str, Any]):
    """One live guard's exact one-shot authorization for its next checkpoint."""

    __slots__ = (
        "_expectation",
        "_guard_authority",
        "_sequence_capability",
    )

    def __init__(
        self,
        expectation: Mapping[str, Any],
        *,
        _guard: Any,
        _sequence_capability: object,
    ) -> None:
        canonical = _mapping(expectation, "typed RSS checkpoint expectation")
        validated = _rss_checkpoint_expectation(
            canonical,
            next_layer_index=_nonnegative_int(
                getattr(_guard, "_authorized_next_layer_index", None),
                "guard-authorized next layer index",
            ),
        )
        authorizer = _live_rss_guard_authorizer(
            _guard,
            method_name="_authorize_rss_expectation_construction",
        )
        guard_authority = authorizer(
            validated,
            sequence_capability=_sequence_capability,
        )
        self._expectation = MappingProxyType(validated)
        self._guard_authority = guard_authority
        self._sequence_capability = _sequence_capability

    def __getitem__(self, key: str) -> Any:
        return self._expectation[key]

    def __iter__(self):
        return iter(self._expectation)

    def __len__(self) -> int:
        return len(self._expectation)


class Lfm2CpcfV5PhysicalRssCheckpointReceipt(Mapping[str, Any]):
    """Exact typed RSS receipt issued only by the authenticated guard path."""

    __slots__ = (
        "_report",
        "_guard_authority",
        "_sequence_capability",
    )

    def __init__(
        self,
        report: Mapping[str, Any],
        *,
        _guard: Any,
        _sequence_capability: object,
    ) -> None:
        canonical = _mapping(report, "typed RSS checkpoint receipt")
        body = dict(canonical)
        claimed = _digest(
            body.pop("report_sha256", None),
            "typed RSS checkpoint report SHA-256",
        )
        if (
            set(canonical) != _RSS_CHECKPOINT_FIELDS
            or canonical.get("schema") != LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA
            or canonical_sha256(body) != claimed
        ):
            raise Lfm2CpcfV5ResumeError(
                "Typed RSS checkpoint receipt schema or seal is invalid."
            )
        authorizer = _live_rss_guard_authorizer(
            _guard,
            method_name="_authorize_rss_receipt_construction",
        )
        guard_authority = authorizer(
            canonical,
            sequence_capability=_sequence_capability,
        )
        self._report = MappingProxyType(canonical)
        self._guard_authority = guard_authority
        self._sequence_capability = _sequence_capability

    def __getitem__(self, key: str) -> Any:
        return self._report[key]

    def __iter__(self):
        return iter(self._report)

    def __len__(self) -> int:
        return len(self._report)


def _canonical(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(_thaw(value)).decode("ascii"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5ResumeError(f"{name} must be finite canonical JSON.") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    result = _canonical(value, name)
    if not isinstance(result, dict):
        raise Lfm2CpcfV5ResumeError(f"{name} must be a mapping.")
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfV5ResumeError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfV5ResumeError(f"{name} must be a nonnegative integer.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise Lfm2CpcfV5ResumeError(f"{name} must be positive.")
    return result


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfV5ResumeError(f"{name} must be within [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfV5ResumeError(f"{name} must be within [0, 1].") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise Lfm2CpcfV5ResumeError(f"{name} must be within [0, 1].")
    return result


def _expected_prefix_rss_label(next_layer_index: int) -> str:
    index = _nonnegative_int(next_layer_index, "next_layer_index")
    if index == 0:
        return "prefix_000000_persisted_before_residency"
    return f"layer_{index - 1:03d}_compiled_checkpoint_persisted_before_release"


def _rss_checkpoint_expectation(
    value: Any,
    *,
    next_layer_index: int,
) -> dict[str, Any]:
    expectation = _mapping(value, "RSS checkpoint expectation")
    body = dict(expectation)
    claimed = _digest(
        body.pop("expectation_sha256", None),
        "RSS checkpoint expectation SHA-256",
    )
    previous = expectation.get("previous_checkpoint_report_sha256")
    ordinal = _nonnegative_int(
        expectation.get("checkpoint_ordinal"),
        "RSS checkpoint ordinal",
    )
    start = _nonnegative_int(
        expectation.get("start_rss_bytes"),
        "RSS start bytes",
    )
    start_hwm = _nonnegative_int(
        expectation.get("start_lifetime_peak_bytes"),
        "RSS start lifetime HWM bytes",
    )
    logical = _positive_int(
        expectation.get("full_phase_tree_logical_bytes"),
        "RSS full phase-tree logical bytes",
    )
    reduction = _fraction(
        expectation.get("minimum_reduction_fraction"),
        "RSS minimum reduction fraction",
    )
    maximum = _nonnegative_int(
        expectation.get("maximum_increase_bytes"),
        "RSS maximum increase bytes",
    )
    relative_maximum = _nonnegative_int(
        expectation.get("relative_maximum_increase_bytes"),
        "RSS relative maximum increase bytes",
    )
    maximum_process = _positive_int(
        expectation.get("maximum_process_rss_bytes"),
        "RSS maximum process bytes",
    )
    absolute_headroom = _nonnegative_int(
        expectation.get("absolute_headroom_bytes"),
        "RSS absolute headroom bytes",
    )
    logical_floor = int(math.floor(logical * (1.0 - reduction)))
    if (
        set(expectation) != _RSS_EXPECTATION_FIELDS
        or expectation.get("schema") != LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA
        or canonical_sha256(body) != claimed
        or expectation.get("label") != _expected_prefix_rss_label(next_layer_index)
        or (
            previous is not None
            and _digest(
                previous,
                "previous RSS checkpoint report SHA-256",
            )
            != previous
        )
        or (ordinal == 0) != (previous is None)
        or start_hwm < start
        or start > maximum_process
        or absolute_headroom != maximum_process - start
        or relative_maximum > logical_floor
        or maximum != min(relative_maximum, absolute_headroom)
    ):
        raise Lfm2CpcfV5ResumeError(
            "RSS expectation schema, lineage, baseline, label, or "
            "authenticated logical cap is invalid."
        )
    _digest(
        expectation.get("rss_guard_id_sha256"),
        "RSS guard identity SHA-256",
    )
    return expectation


def build_lfm2_cpcf_v5_physical_rss_policy_sha256(
    *,
    full_phase_tree_logical_bytes: int,
    minimum_reduction_fraction: float,
    maximum_increase_bytes: int,
    maximum_process_rss_bytes: int = (1 << 63) - 1,
) -> str:
    """Seal the exact RSS policy fields authorized by one resume context."""

    logical = _positive_int(
        full_phase_tree_logical_bytes,
        "RSS policy full phase-tree logical bytes",
    )
    reduction = _fraction(
        minimum_reduction_fraction,
        "RSS policy minimum reduction fraction",
    )
    maximum = _nonnegative_int(
        maximum_increase_bytes,
        "RSS policy maximum increase bytes",
    )
    maximum_process = _positive_int(
        maximum_process_rss_bytes,
        "RSS policy maximum process bytes",
    )
    logical_floor = int(math.floor(logical * (1.0 - reduction)))
    if maximum > logical_floor:
        raise Lfm2CpcfV5ResumeError(
            "RSS policy maximum increase exceeds its authenticated logical "
            "reduction floor."
        )
    return canonical_sha256(
        {
            "schema": LFM2_CPCF_V5_PHYSICAL_RSS_POLICY_SCHEMA,
            "full_phase_tree_logical_bytes": logical,
            "minimum_reduction_fraction": reduction,
            "maximum_increase_bytes": maximum,
            "maximum_process_rss_bytes": maximum_process,
        }
    )


def _rss_expectation_policy_sha256(
    expectation: Mapping[str, Any],
) -> str:
    return build_lfm2_cpcf_v5_physical_rss_policy_sha256(
        full_phase_tree_logical_bytes=expectation["full_phase_tree_logical_bytes"],
        minimum_reduction_fraction=expectation["minimum_reduction_fraction"],
        maximum_increase_bytes=expectation["relative_maximum_increase_bytes"],
        maximum_process_rss_bytes=expectation["maximum_process_rss_bytes"],
    )


def _rss_checkpoint_policy_binding(
    expectation: Mapping[str, Any],
) -> dict[str, int]:
    return {
        "start_rss_bytes": _nonnegative_int(
            expectation.get("start_rss_bytes"),
            "RSS policy binding start bytes",
        ),
        "start_lifetime_peak_bytes": _nonnegative_int(
            expectation.get("start_lifetime_peak_bytes"),
            "RSS policy binding start lifetime peak bytes",
        ),
        "relative_maximum_increase_bytes": _nonnegative_int(
            expectation.get("relative_maximum_increase_bytes"),
            "RSS policy binding relative maximum increase bytes",
        ),
        "maximum_process_rss_bytes": _positive_int(
            expectation.get("maximum_process_rss_bytes"),
            "RSS policy binding maximum process bytes",
        ),
        "absolute_headroom_bytes": _nonnegative_int(
            expectation.get("absolute_headroom_bytes"),
            "RSS policy binding absolute headroom bytes",
        ),
        "maximum_increase_bytes": _nonnegative_int(
            expectation.get("maximum_increase_bytes"),
            "RSS policy binding maximum increase bytes",
        ),
    }


def _authenticate_rss_checkpoint(
    value: Any,
    *,
    expectation: Mapping[str, Any],
    passed: bool,
    require_typed: bool = True,
) -> dict[str, Any]:
    if require_typed and type(value) is not (Lfm2CpcfV5PhysicalRssCheckpointReceipt):
        raise Lfm2CpcfV5ResumeError(
            "RSS admission requires the exact typed V5 guard receipt; "
            "arbitrary self-sealed mappings fail closed."
        )
    receipt = _mapping(value, "RSS checkpoint receipt")
    body = dict(receipt)
    claimed = _digest(
        body.pop("report_sha256", None),
        "RSS checkpoint report SHA-256",
    )
    integer_fields = (
        "start_rss_bytes",
        "current_rss_bytes",
        "peak_rss_bytes",
        "start_lifetime_peak_bytes",
        "current_lifetime_peak_bytes",
        "current_increase_bytes",
        "peak_current_rss_increase_bytes",
        "lifetime_peak_increase_bytes",
        "peak_increase_bytes",
        "full_phase_tree_logical_bytes",
        "relative_maximum_increase_bytes",
        "maximum_process_rss_bytes",
        "absolute_headroom_bytes",
        "maximum_increase_bytes",
        "checkpoint_ordinal",
    )
    numbers = {
        name: _nonnegative_int(receipt.get(name), f"RSS receipt {name}")
        for name in integer_fields
    }
    reduction = _fraction(
        receipt.get("minimum_reduction_fraction"),
        "RSS receipt minimum reduction fraction",
    )
    current_growth = max(
        0,
        numbers["current_rss_bytes"] - numbers["start_rss_bytes"],
    )
    lifetime_growth = max(
        0,
        numbers["current_lifetime_peak_bytes"] - numbers["start_lifetime_peak_bytes"],
    )
    peak_growth = max(
        numbers["peak_current_rss_increase_bytes"],
        numbers["lifetime_peak_increase_bytes"],
    )
    peak_current_growth = max(
        0,
        numbers["peak_rss_bytes"] - numbers["start_rss_bytes"],
    )
    absolute_headroom = max(
        0,
        numbers["maximum_process_rss_bytes"] - numbers["start_rss_bytes"],
    )
    expected_fields = (
        "label",
        "rss_guard_id_sha256",
        "checkpoint_ordinal",
        "previous_checkpoint_report_sha256",
        "start_rss_bytes",
        "start_lifetime_peak_bytes",
        "full_phase_tree_logical_bytes",
        "minimum_reduction_fraction",
        "relative_maximum_increase_bytes",
        "maximum_process_rss_bytes",
        "absolute_headroom_bytes",
        "maximum_increase_bytes",
    )
    if (
        set(receipt) != _RSS_CHECKPOINT_FIELDS
        or receipt.get("schema") != LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA
        or canonical_sha256(body) != claimed
        or any(receipt.get(name) != expectation.get(name) for name in expected_fields)
        or receipt.get("physical_rss_gate_passed") is not passed
        or numbers["peak_rss_bytes"] < numbers["current_rss_bytes"]
        or numbers["current_lifetime_peak_bytes"] < numbers["peak_rss_bytes"]
        or numbers["current_lifetime_peak_bytes"] < numbers["start_lifetime_peak_bytes"]
        or numbers["current_increase_bytes"] != current_growth
        or numbers["peak_current_rss_increase_bytes"] != peak_current_growth
        or numbers["lifetime_peak_increase_bytes"] != lifetime_growth
        or numbers["peak_increase_bytes"] != peak_growth
        or numbers["absolute_headroom_bytes"] != absolute_headroom
        or numbers["start_rss_bytes"] > numbers["maximum_process_rss_bytes"]
        or numbers["maximum_increase_bytes"]
        != min(
            numbers["relative_maximum_increase_bytes"],
            numbers["absolute_headroom_bytes"],
        )
        or reduction != float(expectation["minimum_reduction_fraction"])
        or (
            passed
            and (
                numbers["current_increase_bytes"] > numbers["maximum_increase_bytes"]
                or numbers["peak_increase_bytes"] > numbers["maximum_increase_bytes"]
                or numbers["current_rss_bytes"]
                > numbers["maximum_process_rss_bytes"]
                or numbers["peak_rss_bytes"]
                > numbers["maximum_process_rss_bytes"]
                or numbers["current_lifetime_peak_bytes"]
                > numbers["maximum_process_rss_bytes"]
            )
        )
    ):
        raise Lfm2CpcfV5ResumeError(
            "RSS checkpoint schema, seal, lineage, baseline, peak math, "
            "policy cap, or outcome is invalid."
        )
    return receipt


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(child) for child in value]
    return value


def _cpu_tensor(value: Any, name: str) -> Any:
    torch = model_fold._torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfV5ResumeError(
            f"{name} must be a detached dense materialized tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise Lfm2CpcfV5ResumeError(f"{name} contains NaN or infinity.")
    return result


def _self_sealed_report(value: Any, name: str) -> dict[str, Any]:
    report = _mapping(value, name)
    body = dict(report)
    claimed = _digest(body.pop("report_sha256", None), f"{name}.report_sha256")
    if canonical_sha256(body) != claimed:
        raise Lfm2CpcfV5ResumeError(f"{name} report seal is invalid.")
    return report


def _context(value: Any) -> dict[str, Any]:
    context = _mapping(value, "v5 resume context")
    if set(context) != _CONTEXT_FIELDS:
        raise Lfm2CpcfV5ResumeError("V5 resume context has an unexpected field set.")
    body = dict(context)
    claimed = _digest(body.pop("context_sha256", None), "context_sha256")
    if (
        context.get("schema") != LFM2_CPCF_V5_RESUME_SCHEMA
        or canonical_sha256(body) != claimed
    ):
        raise Lfm2CpcfV5ResumeError("V5 resume context schema or seal is invalid.")
    for name in _CONTEXT_FIELDS - {
        "schema",
        "planned_target_unique_parameters",
        "context_sha256",
    }:
        _digest(context[name], f"context.{name}")
    _positive_int(
        context["planned_target_unique_parameters"],
        "context.planned_target_unique_parameters",
    )
    return context


def build_lfm2_cpcf_v5_resume_context(
    *,
    source_model_state_sha256: str,
    source_model_inventory_sha256: str,
    cache_sha256: str,
    selective_catalog_sha256: str,
    selective_source_sha256: str,
    selective_provider_sha256: str,
    trace_sha256: str,
    tokenizer_sha256: str,
    token_bridge_sha256: str,
    source_tied_table_sha256: str,
    blank_target_report_sha256: str,
    blank_target_state_sha256: str,
    target_topology_sha256: str,
    planned_target_unique_parameters: int,
    runtime_policy_sha256: str,
    physical_rss_policy_sha256: str,
    compiler_implementation_sha256: str,
    resume_implementation_sha256: str,
    selector_contract_sha256: str,
    numerical_contract_sha256: str,
    gpu_contract_sha256: str,
    bootstrap_authentication_sha256: str,
    final_norm_authentication_sha256: str,
) -> Mapping[str, Any]:
    """Build the exact live-input identity accepted by a v5 frontier."""

    body = {
        "schema": LFM2_CPCF_V5_RESUME_SCHEMA,
        "source_model_state_sha256": source_model_state_sha256,
        "source_model_inventory_sha256": source_model_inventory_sha256,
        "cache_sha256": cache_sha256,
        "selective_catalog_sha256": selective_catalog_sha256,
        "selective_source_sha256": selective_source_sha256,
        "selective_provider_sha256": selective_provider_sha256,
        "trace_sha256": trace_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "token_bridge_sha256": token_bridge_sha256,
        "source_tied_table_sha256": source_tied_table_sha256,
        "blank_target_report_sha256": blank_target_report_sha256,
        "blank_target_state_sha256": blank_target_state_sha256,
        "target_topology_sha256": target_topology_sha256,
        "planned_target_unique_parameters": planned_target_unique_parameters,
        "runtime_policy_sha256": runtime_policy_sha256,
        "physical_rss_policy_sha256": physical_rss_policy_sha256,
        "compiler_implementation_sha256": compiler_implementation_sha256,
        "resume_implementation_sha256": resume_implementation_sha256,
        "selector_contract_sha256": selector_contract_sha256,
        "numerical_contract_sha256": numerical_contract_sha256,
        "gpu_contract_sha256": gpu_contract_sha256,
        "bootstrap_authentication_sha256": bootstrap_authentication_sha256,
        "final_norm_authentication_sha256": final_norm_authentication_sha256,
    }
    return MappingProxyType(
        _context({**body, "context_sha256": canonical_sha256(body)})
    )


def lfm2_cpcf_v5_resume_implementation() -> Mapping[str, Any]:
    """Return the source-bound identity of this resume implementation."""

    path = Path(__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {
        "schema": LFM2_CPCF_V5_RESUME_IMPLEMENTATION_SCHEMA,
        "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
        "frontier_schema": LFM2_CPCF_V5_BOUNDARY_FRONTIER_SCHEMA,
        "progress_schema": LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA,
        "source_filename": path.name,
        "source_sha256": digest,
        "existing_resume_artifact_store_reused": True,
        "fresh_blank_prefix_installer_reused": True,
        "parent_or_cache_tensor_sidecars": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "implementation_sha256": canonical_sha256(body)})


def _chart_from_payload(
    payload: Mapping[str, Any],
    *,
    tensors: Mapping[str, Any],
) -> CpcfBoundaryChart:
    predictors = {
        item["name"]: tensors[item["tensor"]] for item in payload["feature_predictors"]
    }
    mapping = PredictiveBoundaryMap(
        encoder=tensors[payload["encoder_tensor"]],
        decoder=tensors[payload["decoder_tensor"]],
        feature_predictors=predictors,
        report=_thaw(payload["mapping_report"]),
    )
    return CpcfBoundaryChart(
        name=payload["name"],
        mapping=mapping,
        report=_thaw(payload["report"]),
    )


def _frontier_payload(
    compiler: Any,
    *,
    context: Mapping[str, Any],
    next_layer_index: int,
    expected_layer_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = _nonnegative_int(next_layer_index, "next_layer_index")
    depth = _positive_int(expected_layer_count, "expected_layer_count")
    if (
        index > depth
        or getattr(compiler, "interface", None) is None
        or getattr(compiler, "base_interface", None) is None
        or getattr(compiler, "anchor_fold", None) is None
        or getattr(compiler, "anchor_split_reports", None) is None
        or getattr(compiler, "fit_future", None) is None
        or getattr(compiler, "validation_future", None) is None
        or getattr(compiler, "fit_weights", None) is None
        or getattr(compiler, "validation_weights", None) is None
        or getattr(compiler, "fit_weight_summary", None) is None
        or getattr(compiler, "validation_weight_summary", None) is None
    ):
        raise Lfm2CpcfV5ResumeError(
            "Selective boundary compiler is not snapshot-ready."
        )
    phases = tuple(getattr(compiler, "phases", ()))
    layer_kinds = tuple(getattr(compiler, "layer_kinds", ()))
    layers = tuple(getattr(compiler, "layer_results", ()))
    if (
        not phases
        or len(layer_kinds) != depth
        or len(layers) != index
        or tuple(layer.layer_index for layer in layers) != tuple(range(index))
    ):
        raise Lfm2CpcfV5ResumeError(
            "V5 frontier layers are not the exact contiguous accepted prefix."
        )
    target_weight_tensor = "interface.target_weight"
    tensors: dict[str, Any] = {
        target_weight_tensor: _cpu_tensor(
            compiler.interface.target_weight,
            target_weight_tensor,
        )
    }
    chart_payloads = []
    for layer in layers:
        chart_payloads.append(
            {
                "layer_index": layer.layer_index,
                "operator_kind": layer.operator_kind,
                "layer_charts_sha256": layer.layer_charts_sha256,
                "post_operator": resume_bridge._chart_payload(
                    layer.post_operator,
                    layer_index=layer.layer_index,
                    slot="post_operator",
                    tensors=tensors,
                ),
                "post_layer": resume_bridge._chart_payload(
                    layer.post_layer,
                    layer_index=layer.layer_index,
                    slot="post_layer",
                    tensors=tensors,
                ),
            }
        )
    observations = []
    observation_sha = dict(getattr(compiler, "_observation_sha", {}))
    for phase_index, phase in enumerate(phases):
        expected_fields = ["output.final_norm_output"]
        for layer_index in range(index):
            expected_fields.extend(
                f"layer.{layer_index:03d}.{boundary_name}"
                for boundary_name in _BOUNDARIES
            )
        for field in expected_fields:
            try:
                tensor_sha = observation_sha[(phase_index, field)]
            except KeyError as exc:
                raise Lfm2CpcfV5ResumeError(
                    "V5 frontier source-observation prefix is incomplete."
                ) from exc
            observations.append(
                {
                    "phase_index": phase_index,
                    "phase_sha256": phase.phase_sha256,
                    "field": field,
                    "tensor_sha256": _digest(
                        tensor_sha,
                        "source observation tensor SHA-256",
                    ),
                }
            )
    body = {
        "schema": LFM2_CPCF_V5_BOUNDARY_FRONTIER_SCHEMA,
        "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
        "context": _context(context),
        "next_layer_index": index,
        "expected_layer_count": depth,
        "source_width": int(compiler.source_width),
        "target_width": int(compiler.target_width),
        "ordered_phase_sha256": [phase.phase_sha256 for phase in phases],
        "split_phase_sha256": dict(compiler.split_phase_sha256),
        "split_row_sha256": dict(compiler.split_row_sha256),
        "split_row_counts": dict(compiler.split_rows),
        "layer_operator_kinds": list(layer_kinds),
        "interface": {
            "target_weight_tensor": target_weight_tensor,
            "target_weight_sha256": resume_bridge._runtime_tensor_sha256(
                compiler.interface.target_weight
            ),
            "producer_encoder_sha256": (
                resume_bridge._runtime_tensor_sha256(
                    compiler.interface.producer_encoder
                )
            ),
            "fit_future_sha256": resume_bridge._runtime_tensor_sha256(
                compiler.fit_future
            ),
            "validation_future_sha256": (
                resume_bridge._runtime_tensor_sha256(compiler.validation_future)
            ),
            "active_interface_report": _thaw(compiler.interface.report),
            "base_interface_report": _thaw(compiler.base_interface.report),
            "causal_anchor_report": _thaw(compiler.anchor_fold.report),
            "anchor_split_reports": _thaw(compiler.anchor_split_reports),
        },
        "completed_layer_charts": chart_payloads,
        "source_observations": observations,
        "phase_weighting": {
            "fit_weights_sha256": resume_bridge._runtime_tensor_sha256(
                compiler.fit_weights
            ),
            "validation_weights_sha256": (
                resume_bridge._runtime_tensor_sha256(compiler.validation_weights)
            ),
            "fit_weight_summary": _thaw(compiler.fit_weight_summary),
            "validation_weight_summary": _thaw(compiler.validation_weight_summary),
        },
        "operation_audit": dict(_ZERO_AUDIT),
        "tensor_inventory": [
            resume_bridge._tensor_record(name, tensors[name])
            for name in sorted(tensors)
        ],
    }
    return (
        {**body, "frontier_sha256": canonical_sha256(body)},
        tensors,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5BoundaryFrontier:
    """Typed exact partial boundary state at one accepted layer boundary."""

    report: Mapping[str, Any]
    tensors: Mapping[str, Any]

    def __post_init__(self) -> None:
        report = _mapping(self.report, "v5 boundary frontier")
        if set(report) != _FRONTIER_FIELDS:
            raise Lfm2CpcfV5ResumeError(
                "V5 boundary frontier has an unexpected field set."
            )
        body = dict(report)
        claimed = _digest(
            body.pop("frontier_sha256", None),
            "frontier_sha256",
        )
        if (
            report.get("schema") != LFM2_CPCF_V5_BOUNDARY_FRONTIER_SCHEMA
            or report.get("resume_schema") != LFM2_CPCF_V5_RESUME_SCHEMA
            or canonical_sha256(body) != claimed
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 boundary frontier schema or seal is invalid."
            )
        _context(report.get("context"))
        next_index = _nonnegative_int(
            report.get("next_layer_index"),
            "frontier.next_layer_index",
        )
        depth = _positive_int(
            report.get("expected_layer_count"),
            "frontier.expected_layer_count",
        )
        source_width = _positive_int(
            report.get("source_width"),
            "frontier.source_width",
        )
        target_width = _positive_int(
            report.get("target_width"),
            "frontier.target_width",
        )
        if next_index > depth or target_width >= source_width:
            raise Lfm2CpcfV5ResumeError(
                "V5 boundary frontier depth or widths are invalid."
            )
        phase_sha = report.get("ordered_phase_sha256")
        layer_kinds = report.get("layer_operator_kinds")
        if (
            not isinstance(phase_sha, list)
            or not phase_sha
            or any(
                _SHA256.fullmatch(value) is None
                for value in phase_sha
                if isinstance(value, str)
            )
            or any(not isinstance(value, str) for value in phase_sha)
            or not isinstance(layer_kinds, list)
            or len(layer_kinds) != depth
            or any(kind not in {"short_conv", "full_attention"} for kind in layer_kinds)
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier phase or layer inventory is invalid."
            )
        for field in ("split_phase_sha256", "split_row_sha256"):
            values = report.get(field)
            if not isinstance(values, dict) or set(values) != set(_SPLITS):
                raise Lfm2CpcfV5ResumeError(f"V5 frontier {field} is incomplete.")
            for split in _SPLITS:
                _digest(values[split], f"{field}.{split}")
        row_counts = report.get("split_row_counts")
        if not isinstance(row_counts, dict) or set(row_counts) != set(_SPLITS):
            raise Lfm2CpcfV5ResumeError("V5 frontier split row counts are incomplete.")
        for split in _SPLITS:
            _nonnegative_int(row_counts[split], f"split_row_counts.{split}")
        if report.get("operation_audit") != _ZERO_AUDIT:
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier does not prove zero prohibited operations."
            )
        tensors = {
            str(name): _cpu_tensor(value, f"frontier tensor {name!r}")
            for name, value in self.tensors.items()
        }
        inventory = report.get("tensor_inventory")
        if (
            not isinstance(inventory, list)
            or [item.get("name") for item in inventory] != sorted(tensors)
            or set(tensors) != {item.get("name") for item in inventory}
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier tensor inventory is missing or unordered."
            )
        for item in inventory:
            name = item.get("name")
            if item != resume_bridge._tensor_record(name, tensors[name]):
                raise Lfm2CpcfV5ResumeError(
                    f"V5 frontier tensor {name!r} differs from its seal."
                )
        interface = report.get("interface")
        if not isinstance(interface, dict) or set(interface) != _INTERFACE_FIELDS:
            raise Lfm2CpcfV5ResumeError("V5 frontier interface payload is invalid.")
        for field in (
            "target_weight_sha256",
            "producer_encoder_sha256",
            "fit_future_sha256",
            "validation_future_sha256",
        ):
            _digest(interface[field], f"frontier interface {field}")
        active = _self_sealed_report(
            interface["active_interface_report"],
            "active interface",
        )
        base = _self_sealed_report(
            interface["base_interface_report"],
            "base interface",
        )
        anchor = _self_sealed_report(
            interface["causal_anchor_report"],
            "causal anchor",
        )
        if (
            active != anchor
            or interface["target_weight_tensor"] not in tensors
            or resume_bridge._runtime_tensor_sha256(
                tensors[interface["target_weight_tensor"]]
            )
            != interface["target_weight_sha256"]
            or active.get("target_weight_sha256") != interface["target_weight_sha256"]
            or active.get("producer_encoder_sha256")
            != interface["producer_encoder_sha256"]
            or anchor.get("base_interface_report_sha256") != base.get("report_sha256")
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier interface tensors or lineage changed."
            )
        anchor_splits = interface.get("anchor_split_reports")
        if not isinstance(anchor_splits, dict) or set(anchor_splits) != set(_SPLITS):
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier anchor split reports are incomplete."
            )
        charts = report.get("completed_layer_charts")
        if not isinstance(charts, list) or len(charts) != next_index:
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier chart prefix differs from next_layer_index."
            )
        used_tensors: set[str] = {interface["target_weight_tensor"]}
        for expected_index, layer in enumerate(charts):
            if (
                not isinstance(layer, dict)
                or set(layer) != resume_bridge._LAYER_FIELDS
                or layer.get("layer_index") != expected_index
                or layer.get("operator_kind") != layer_kinds[expected_index]
            ):
                raise Lfm2CpcfV5ResumeError(
                    "V5 frontier chart layers are not contiguous."
                )
            for slot in ("post_operator", "post_layer"):
                chart = resume_bridge._validate_chart_payload(
                    layer[slot],
                    layer_index=expected_index,
                    slot=slot,
                    tensors=tensors,
                )
                used_tensors.update(
                    {
                        chart["encoder_tensor"],
                        chart["decoder_tensor"],
                        *(item["tensor"] for item in chart["feature_predictors"]),
                    }
                )
        weighting = report.get("phase_weighting")
        if (
            not isinstance(weighting, dict)
            or set(weighting) != _PHASE_WEIGHTING_FIELDS
            or not isinstance(weighting["fit_weight_summary"], list)
            or not isinstance(weighting["validation_weight_summary"], list)
        ):
            raise Lfm2CpcfV5ResumeError("V5 frontier phase weighting is invalid.")
        _digest(
            weighting["fit_weights_sha256"],
            "frontier fit weights SHA-256",
        )
        _digest(
            weighting["validation_weights_sha256"],
            "frontier validation weights SHA-256",
        )
        if used_tensors != set(tensors):
            raise Lfm2CpcfV5ResumeError("V5 frontier contains unreferenced tensors.")
        observations = report.get("source_observations")
        expected_observation_count = len(phase_sha) * (1 + 2 * next_index)
        if (
            not isinstance(observations, list)
            or len(observations) != expected_observation_count
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 frontier source-observation prefix is incomplete."
            )
        expected_observations = []
        for phase_index, phase_digest in enumerate(phase_sha):
            fields = ["output.final_norm_output"]
            for layer_index in range(next_index):
                fields.extend(f"layer.{layer_index:03d}.{name}" for name in _BOUNDARIES)
            expected_observations.extend(
                (phase_index, phase_digest, field) for field in fields
            )
        for record, expected in zip(
            observations,
            expected_observations,
            strict=True,
        ):
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "phase_index",
                    "phase_sha256",
                    "field",
                    "tensor_sha256",
                }
                or (
                    record["phase_index"],
                    record["phase_sha256"],
                    record["field"],
                )
                != expected
            ):
                raise Lfm2CpcfV5ResumeError(
                    "V5 source observations are missing or reordered."
                )
            _digest(
                record["tensor_sha256"],
                "source observation tensor SHA-256",
            )
        object.__setattr__(self, "report", _freeze(report))
        object.__setattr__(
            self,
            "tensors",
            MappingProxyType(tensors),
        )

    @classmethod
    def capture(
        cls,
        compiler: Any,
        *,
        context: Mapping[str, Any],
        next_layer_index: int,
        expected_layer_count: int,
    ) -> Lfm2CpcfV5BoundaryFrontier:
        report, tensors = _frontier_payload(
            compiler,
            context=context,
            next_layer_index=next_layer_index,
            expected_layer_count=expected_layer_count,
        )
        return cls(report=report, tensors=tensors)

    @property
    def frontier_sha256(self) -> str:
        return str(self.report["frontier_sha256"])

    @property
    def next_layer_index(self) -> int:
        return int(self.report["next_layer_index"])

    @property
    def expected_layer_count(self) -> int:
        return int(self.report["expected_layer_count"])

    @property
    def context(self) -> Mapping[str, Any]:
        return self.report["context"]

    def hydrate(self, compiler: Any) -> None:
        """Hydrate one new selective boundary compiler without fitting."""

        phases = tuple(getattr(compiler, "phases", ()))
        if (
            [phase.phase_sha256 for phase in phases]
            != list(self.report["ordered_phase_sha256"])
            or list(getattr(compiler, "layer_kinds", ()))
            != list(self.report["layer_operator_kinds"])
            or dict(getattr(compiler, "split_phase_sha256", {}))
            != dict(self.report["split_phase_sha256"])
            or dict(getattr(compiler, "split_row_sha256", {}))
            != dict(self.report["split_row_sha256"])
            or dict(getattr(compiler, "split_rows", {}))
            != dict(self.report["split_row_counts"])
            or int(getattr(compiler, "source_width", -1))
            != int(self.report["source_width"])
            or int(getattr(compiler, "target_width", -1))
            != int(self.report["target_width"])
            or getattr(compiler, "interface", None) is None
            or getattr(compiler, "base_interface", None) is None
            or getattr(compiler, "anchor_fold", None) is None
            or getattr(compiler, "anchor_split_reports", None) is None
            or getattr(compiler, "fit_future", None) is None
            or getattr(compiler, "validation_future", None) is None
            or getattr(compiler, "fit_weights", None) is None
            or getattr(compiler, "validation_weights", None) is None
            or getattr(compiler, "layer_results", None) != []
        ):
            raise Lfm2CpcfV5ResumeError(
                "Final-normalization-rehydrated boundary compiler differs "
                "from the authenticated frontier."
            )
        interface_payload = self.report["interface"]
        weighting = self.report["phase_weighting"]
        final_observations = {
            (record["phase_index"], record["field"]): record["tensor_sha256"]
            for record in self.report["source_observations"]
            if record["field"] == "output.final_norm_output"
        }
        if (
            _thaw(compiler.interface.report)
            != _thaw(interface_payload["active_interface_report"])
            or _thaw(compiler.base_interface.report)
            != _thaw(interface_payload["base_interface_report"])
            or _thaw(compiler.anchor_fold.report)
            != _thaw(interface_payload["causal_anchor_report"])
            or _thaw(compiler.anchor_split_reports)
            != _thaw(interface_payload["anchor_split_reports"])
            or resume_bridge._runtime_tensor_sha256(compiler.interface.target_weight)
            != interface_payload["target_weight_sha256"]
            or not model_fold._torch().equal(
                _cpu_tensor(
                    compiler.interface.target_weight,
                    "recomputed interface target weight",
                ),
                self.tensors[interface_payload["target_weight_tensor"]],
            )
            or resume_bridge._runtime_tensor_sha256(compiler.interface.producer_encoder)
            != interface_payload["producer_encoder_sha256"]
            or resume_bridge._runtime_tensor_sha256(compiler.fit_future)
            != interface_payload["fit_future_sha256"]
            or resume_bridge._runtime_tensor_sha256(compiler.validation_future)
            != interface_payload["validation_future_sha256"]
            or resume_bridge._runtime_tensor_sha256(compiler.fit_weights)
            != weighting["fit_weights_sha256"]
            or resume_bridge._runtime_tensor_sha256(compiler.validation_weights)
            != weighting["validation_weights_sha256"]
            or _thaw(compiler.fit_weight_summary)
            != _thaw(weighting["fit_weight_summary"])
            or _thaw(compiler.validation_weight_summary)
            != _thaw(weighting["validation_weight_summary"])
            or {
                key: value
                for key, value in compiler._observation_sha.items()
                if key[1] == "output.final_norm_output"
            }
            != final_observations
        ):
            raise Lfm2CpcfV5ResumeError(
                "Authenticated cached final-normalization evidence did not "
                "recompute the exact interface, teacher coordinates, "
                "weights, or lineage sealed by the frontier."
            )
        compiler.layer_results = []
        for layer_payload in self.report["completed_layer_charts"]:
            post_operator = _chart_from_payload(
                layer_payload["post_operator"],
                tensors=self.tensors,
            )
            post_layer = _chart_from_payload(
                layer_payload["post_layer"],
                tensors=self.tensors,
            )
            compiler.layer_results.append(
                Lfm2CpcfLayerBoundaryCharts(
                    layer_index=layer_payload["layer_index"],
                    operator_kind=layer_payload["operator_kind"],
                    post_operator=post_operator,
                    post_layer=post_layer,
                    layer_charts_sha256=layer_payload["layer_charts_sha256"],
                )
            )
        compiler._observation_sha.update(
            {
                (record["phase_index"], record["field"]): record["tensor_sha256"]
                for record in self.report["source_observations"]
                if record["field"] != "output.final_norm_output"
            }
        )
        round_trip = Lfm2CpcfV5BoundaryFrontier.capture(
            compiler,
            context=self.context,
            next_layer_index=self.next_layer_index,
            expected_layer_count=self.expected_layer_count,
        )
        if (
            _thaw(round_trip.report) != _thaw(self.report)
            or set(round_trip.tensors) != set(self.tensors)
            or any(
                resume_bridge._tensor_record(name, round_trip.tensors[name])
                != resume_bridge._tensor_record(name, self.tensors[name])
                for name in self.tensors
            )
        ):
            raise Lfm2CpcfV5ResumeError(
                "Hydrated v5 boundary compiler is not an exact round trip."
            )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5ProgressEvent:
    """Owned target-prefix/state/frontier snapshot for synchronous publication."""

    frontier: Lfm2CpcfV5BoundaryFrontier
    incoming_state_bank: Any
    completed_layer_receipts: tuple[Lfm2CpcfCompletedLayerReceipt, ...]
    target_prefix_cpu_tensors: Mapping[str, Any]
    target_prefix_snapshot_metadata: Mapping[str, Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.frontier, Lfm2CpcfV5BoundaryFrontier):
            raise TypeError("frontier must be Lfm2CpcfV5BoundaryFrontier.")
        incoming = self.incoming_state_bank
        verify = getattr(incoming, "verify", None)
        if not callable(verify):
            raise Lfm2CpcfV5ResumeError(
                "Progress incoming state bank must be an owned typed snapshot."
            )
        verify()
        receipts = tuple(self.completed_layer_receipts)
        if (
            len(receipts) != self.frontier.next_layer_index
            or tuple(receipt.layer_index for receipt in receipts)
            != tuple(range(len(receipts)))
            or incoming.layer_index != self.frontier.next_layer_index
            or (
                receipts and receipts[-1].next_bank_sha256 != incoming.state_bank_sha256
            )
        ):
            raise Lfm2CpcfV5ResumeError(
                "Progress receipts and incoming child state are not contiguous."
            )
        tensors = {
            str(name): _cpu_tensor(value, f"target prefix tensor {name!r}")
            for name, value in self.target_prefix_cpu_tensors.items()
        }
        metadata = _mapping(
            self.target_prefix_snapshot_metadata,
            "target prefix snapshot metadata",
        )
        snapshot_receipt = model_fold._cpu_tensor_snapshot_receipt_from_parts(
            tensors,
            metadata,
        )
        report = _mapping(self.report, "v5 progress event")
        body = dict(report)
        claimed = _digest(
            body.pop("report_sha256", None),
            "progress report SHA-256",
        )
        if (
            set(report) != _PROGRESS_FIELDS
            or report.get("schema") != LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA
            or report.get("resume_schema") != LFM2_CPCF_V5_RESUME_SCHEMA
            or canonical_sha256(body) != claimed
            or report.get("frontier_sha256") != self.frontier.frontier_sha256
            or report.get("context_sha256") != self.frontier.context["context_sha256"]
            or report.get("next_layer_index") != self.frontier.next_layer_index
            or report.get("incoming_state_bank_sha256") != incoming.state_bank_sha256
            or report.get("completed_layer_result_sha256")
            != [receipt.result_sha256 for receipt in receipts]
            or report.get("target_prefix_snapshot_receipt") != dict(snapshot_receipt)
            or report.get("pending_rss_admission") is not True
            or report.get("rss_admission_claimed") is not False
            or report.get("parent_or_cache_tensors_archived") != 0
            or report.get("operation_audit") != _ZERO_AUDIT
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 progress event schema, lineage, or seal is invalid."
            )
        object.__setattr__(self, "completed_layer_receipts", receipts)
        object.__setattr__(
            self,
            "target_prefix_cpu_tensors",
            MappingProxyType(tensors),
        )
        object.__setattr__(self, "target_prefix_snapshot_metadata", _freeze(metadata))
        object.__setattr__(self, "report", _freeze(report))

    @classmethod
    def capture(
        cls,
        *,
        frontier: Lfm2CpcfV5BoundaryFrontier,
        incoming_state_bank: Lfm2CpcfPhaseStateBank,
        completed_layer_receipts: Sequence[
            Lfm2CpcfCompletedLayerReceipt | Mapping[str, Any]
        ],
        target_model: Any,
    ) -> Lfm2CpcfV5ProgressEvent:
        receipts = tuple(
            value
            if isinstance(value, Lfm2CpcfCompletedLayerReceipt)
            else Lfm2CpcfCompletedLayerReceipt(value)
            for value in completed_layer_receipts
        )
        incoming = model_fold._snapshot_incoming_bank_evidence(incoming_state_bank)
        if incoming is None:  # pragma: no cover - typed input boundary
            raise Lfm2CpcfV5ResumeError(
                "Progress event cannot snapshot the incoming state bank."
            )
        snapshot = model_fold._snapshot_model(target_model)
        tensors = {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in snapshot.tensor_values.items()
        }
        metadata = model_fold._snapshot_metadata_plain(snapshot)
        receipt = model_fold._cpu_tensor_snapshot_receipt_from_parts(
            tensors,
            metadata,
        )
        body = {
            "schema": LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "frontier_sha256": frontier.frontier_sha256,
            "context_sha256": frontier.context["context_sha256"],
            "next_layer_index": frontier.next_layer_index,
            "incoming_state_bank_sha256": incoming.state_bank_sha256,
            "completed_layer_result_sha256": [
                receipt.result_sha256 for receipt in receipts
            ],
            "target_prefix_snapshot_receipt": dict(receipt),
            "pending_rss_admission": True,
            "rss_admission_claimed": False,
            "parent_or_cache_tensors_archived": 0,
            "operation_audit": dict(_ZERO_AUDIT),
        }
        return cls(
            frontier=frontier,
            incoming_state_bank=incoming,
            completed_layer_receipts=receipts,
            target_prefix_cpu_tensors=tensors,
            target_prefix_snapshot_metadata=metadata,
            report={**body, "report_sha256": canonical_sha256(body)},
        )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5PendingCheckpoint:
    frontier_reference: Lfm2CpcfArtifactReference
    checkpoint: VerifiedLfm2CpcfPrefixCheckpoint
    frontier_sha256: str
    event_report_sha256: str
    next_layer_index: int
    rss_checkpoint_expectation: Mapping[str, Any]
    pending_sha256: str
    coordinator_identity_sha256: str
    capability_token: str


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5ResumeState:
    coordinator: Lfm2CpcfV5ResumeCoordinator
    frontier: Lfm2CpcfV5BoundaryFrontier
    frontier_reference: Lfm2CpcfArtifactReference
    checkpoint_reference: Lfm2CpcfArtifactReference
    admission: Mapping[str, Any]

    @property
    def next_layer_index(self) -> int:
        return self.frontier.next_layer_index


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RestoredPrefix:
    frontier: Lfm2CpcfV5BoundaryFrontier
    state_bank: Lfm2CpcfPhaseStateBank
    completed_layer_receipts: tuple[Lfm2CpcfCompletedLayerReceipt, ...]
    prefix_install_report: Mapping[str, Any]
    admission: Mapping[str, Any]


class Lfm2CpcfV5ResumeCoordinator:
    """Publish, admit, reopen, and restore exact v5 prefix checkpoints."""

    def __init__(self, store: Lfm2CpcfResumeArtifactStore) -> None:
        if type(store) is not Lfm2CpcfResumeArtifactStore:
            raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
        self.store = store
        self.store._authenticate_root()
        self._root_identity = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                self.store.root,
                name="V5 resume store root",
            )
        )
        self._coordinator_capability = secrets.token_hex(32)
        self._coordinator_identity_sha256 = hashlib.sha256(
            self._coordinator_capability.encode("ascii")
        ).hexdigest()
        self._pending_capabilities: dict[str, str] = {}
        self._pending_rss_authorities: dict[
            str,
            tuple[object, object],
        ] = {}
        self._collection_identities: dict[str, tuple[int, int]] = {}
        for collection in sorted(_V5_DOCUMENT_COLLECTIONS):
            self._collection(collection)

    def _authenticate_root(self) -> None:
        self.store._authenticate_root()
        current = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                self.store.root,
                name="V5 resume store root",
            )
        )
        if current != self._root_identity:
            raise Lfm2CpcfV5ResumeError("V5 resume store root changed identity.")

    def _collection(self, collection: str) -> Path:
        if collection not in _V5_DOCUMENT_COLLECTIONS:
            raise Lfm2CpcfV5ResumeError("Unknown V5 resume document collection.")
        self._authenticate_root()
        path = self.store.root / collection
        existed = os.path.lexists(path)
        try:
            path.mkdir(exist_ok=True)
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(f"Could not create {collection}.") from exc
        stat_result = resume_artifact._real_directory(
            path,
            name=f"{collection} collection",
        )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(f"Could not resolve {collection}.") from exc
        identity = resume_artifact._stat_identity(stat_result)
        expected = self._collection_identities.get(collection)
        if resolved.parent != self.store.root or (
            expected is not None and identity != expected
        ):
            raise Lfm2CpcfV5ResumeError(f"{collection} escaped or changed identity.")
        if expected is None:
            self._collection_identities[collection] = identity
        if not existed:
            resume_artifact._fsync_directory(self.store.root)
        self._authenticate_root()
        return resolved

    def _document_path(self, collection: str, key: str) -> Path:
        digest = _digest(key, f"{collection} key")
        return self._collection(collection) / f"{digest}.json"

    @staticmethod
    def _safe_unlink_temporary(
        path: Path,
        *,
        parent: Path,
        identity: tuple[int, int],
    ) -> None:
        if not os.path.lexists(path):
            return
        try:
            current = path.lstat()
            resolved_parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(
                "Could not authenticate V5 temporary document cleanup."
            ) from exc
        if (
            resume_artifact._stat_identity(current) != identity
            or not stat.S_ISREG(current.st_mode)
            or int(current.st_nlink) != 1
            or resume_artifact._is_link_or_junction(path)
            or resolved_parent != parent
            or _V5_DOCUMENT_TEMP.fullmatch(path.name) is None
        ):
            raise Lfm2CpcfV5ResumeError(
                "Refusing to unlink an unauthenticated V5 temporary document."
            )
        try:
            os.unlink(path)
            resume_artifact._fsync_directory(parent)
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(
                "Could not clean V5 temporary document."
            ) from exc

    def _read_document_bytes(self, path: Path, *, name: str) -> bytes:
        collection = self._collection(path.parent.name)
        if path.parent.resolve(strict=True) != collection:
            raise Lfm2CpcfV5ResumeError(f"{name} parent escaped its collection.")
        parent_before = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                collection,
                name=f"{name} collection",
            )
        )
        raw = resume_artifact._read_regular_bytes(
            path,
            name=name,
            maximum_bytes=8 * 1024 * 1024,
        )
        parent_after = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                collection,
                name=f"{name} collection",
            )
        )
        if (
            parent_before != parent_after
            or parent_after != self._collection_identities[path.parent.name]
        ):
            raise Lfm2CpcfV5ResumeError(f"{name} collection changed while it was read.")
        return raw

    def _write_once(
        self,
        path: Path,
        document: Mapping[str, Any],
    ) -> None:
        """Crash-safe no-clobber publication from a sibling temporary."""

        raw = canonical_json_bytes(document)
        parent = self._collection(path.parent.name)
        if path.parent.resolve(strict=True) != parent:
            raise Lfm2CpcfV5ResumeError(
                "V5 document destination escaped its collection."
            )
        temporary = parent / (f".v5-document-{secrets.token_hex(32)}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        identity: tuple[int, int] | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
                raise Lfm2CpcfV5ResumeError(
                    "V5 temporary document is not one regular file."
                )
            identity = resume_artifact._stat_identity(opened)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("short V5 document write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            current = resume_artifact._regular_file(
                temporary,
                name="V5 temporary document",
            )
            if (
                resume_artifact._stat_identity(current) != identity
                or temporary.resolve(strict=True).parent != parent
                or self._read_document_bytes(
                    temporary,
                    name="V5 temporary document",
                )
                != raw
            ):
                raise Lfm2CpcfV5ResumeError(
                    "V5 temporary document changed before publication."
                )
            try:
                resume_artifact._atomic_rename_noreplace(
                    temporary,
                    path,
                )
            except FileExistsError:
                existing = self._read_document_bytes(
                    path,
                    name=path.name,
                )
                if existing != raw:
                    raise Lfm2CpcfV5ResumeError(
                        "A conflicting V5 document occupies its content key."
                    )
            except OSError as exc:
                if not os.path.lexists(path):
                    raise Lfm2CpcfV5ResumeError(
                        "Atomic V5 publication failed before installing "
                        "the destination."
                    ) from exc
                existing = self._read_document_bytes(
                    path,
                    name=path.name,
                )
                if existing != raw:
                    raise Lfm2CpcfV5ResumeError(
                        "A conflicting V5 document occupies its content key."
                    ) from exc
            resume_artifact._fsync_directory(parent)
            if self._read_document_bytes(path, name=path.name) != raw:
                raise Lfm2CpcfV5ResumeError(
                    "Published V5 document differs from its staged bytes."
                )
        except Lfm2CpcfV5ResumeError:
            raise
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(
                "Could not atomically publish V5 document."
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if identity is not None and os.path.lexists(temporary):
                self._safe_unlink_temporary(
                    temporary,
                    parent=parent,
                    identity=identity,
                )

    def _open_document(
        self,
        path: Path,
        *,
        kind: str,
        expected_key: str,
    ) -> dict[str, Any]:
        key = _digest(expected_key, f"{kind} requested key")
        if path.name != f"{key}.json":
            raise Lfm2CpcfV5ResumeError(
                f"V5 {kind} path stem differs from its requested digest."
            )
        raw = self._read_document_bytes(path, name=f"V5 {kind} document")
        try:
            value = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2CpcfV5ResumeError(
                f"Cannot open canonical V5 {kind} document."
            ) from exc
        if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
            raise Lfm2CpcfV5ResumeError(f"V5 {kind} document is not canonical.")
        schema, seal_field = {
            "progress": (
                LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA,
                "report_sha256",
            ),
            "pending": (
                LFM2_CPCF_V5_PENDING_SCHEMA,
                "pending_sha256",
            ),
            "admission": (
                LFM2_CPCF_V5_ADMISSION_SCHEMA,
                "admission_sha256",
            ),
            "rejection": (
                LFM2_CPCF_V5_REJECTION_SCHEMA,
                "rejection_sha256",
            ),
        }.get(kind, (None, None))
        if schema is None or seal_field is None:
            raise Lfm2CpcfV5ResumeError("Unknown V5 document kind.")
        body = dict(value)
        claimed = _digest(
            body.pop(seal_field, None),
            f"{kind} document seal",
        )
        if (
            value.get("schema") != schema
            or value.get("resume_schema") != LFM2_CPCF_V5_RESUME_SCHEMA
            or canonical_sha256(body) != claimed
            or claimed != key
        ):
            raise Lfm2CpcfV5ResumeError(
                f"V5 {kind} schema, seal, or requested content key is invalid."
            )
        return value

    def _boundary_bindings(
        self,
        frontier: Lfm2CpcfV5BoundaryFrontier,
    ) -> Lfm2CpcfBoundaryBindings:
        context = frontier.context
        return Lfm2CpcfBoundaryBindings(
            parent_sha256=context["source_model_state_sha256"],
            cache_sha256=context["cache_sha256"],
            evidence_sha256=context["selective_catalog_sha256"],
            phase_id=f"v5-prefix-{frontier.next_layer_index:08d}",
            trace_sha256=context["trace_sha256"],
            bridge_sha256=frontier.frontier_sha256,
            tokenizer_sha256=context["tokenizer_sha256"],
            source_table_sha256=context["source_tied_table_sha256"],
            blank_sha256=context["blank_target_state_sha256"],
            target_sha256=context["target_topology_sha256"],
            boundary_policy_sha256=context["runtime_policy_sha256"],
            solver_schema_sha256=context["context_sha256"],
        )

    @staticmethod
    def _global_installed_paths(
        topology: Mapping[str, Any],
    ) -> list[str]:
        result = []
        for name, record in topology["tensor_metadata"].items():
            if record["role"] != "parameter":
                continue
            if re.search(r"(?:^|\.)layers\.[0-9]+\.", name) is None:
                result.append(name)
        return sorted(result)

    def publish_pending(
        self,
        event: Lfm2CpcfV5ProgressEvent,
        *,
        rss_checkpoint_expectation: Mapping[str, Any],
    ) -> Lfm2CpcfV5PendingCheckpoint:
        """Persist a non-admitted prefix before its RSS checkpoint."""

        if type(event) is not Lfm2CpcfV5ProgressEvent:
            raise TypeError("event must be Lfm2CpcfV5ProgressEvent.")
        if type(rss_checkpoint_expectation) is not (
            Lfm2CpcfV5PhysicalRssCheckpointExpectation
        ):
            raise Lfm2CpcfV5ResumeError(
                "Pending publication requires the exact one-shot expectation "
                "issued by the live physical RSS guard."
            )
        frontier = event.frontier
        rss_expectation = _rss_checkpoint_expectation(
            rss_checkpoint_expectation,
            next_layer_index=frontier.next_layer_index,
        )
        if (
            _rss_expectation_policy_sha256(rss_expectation)
            != frontier.context["physical_rss_policy_sha256"]
        ):
            raise Lfm2CpcfV5ResumeError(
                "RSS checkpoint expectation policy differs from the "
                "authenticated frontier runtime policy."
            )
        boundary = self.store.publish_boundary(
            bindings=self._boundary_bindings(frontier),
            reports=_thaw(frontier.report),
            tensors=frontier.tensors,
        )
        boundary = self.store.final_reauthenticate_boundary(boundary)
        reopened = Lfm2CpcfV5BoundaryFrontier(
            report=boundary.reports,
            tensors=boundary.tensors,
        )
        if reopened.frontier_sha256 != frontier.frontier_sha256:
            raise Lfm2CpcfV5ResumeError(
                "Published v5 frontier differs from the progress event."
            )
        incoming, incoming_metadata = flatten_lfm2_cpcf_incoming_state_bank(
            event.incoming_state_bank
        )
        topology = build_lfm2_cpcf_target_topology_snapshot(
            prefix_model=event.target_prefix_cpu_tensors,
            layer_count=frontier.expected_layer_count,
            snapshot_metadata=_thaw(event.target_prefix_snapshot_metadata),
        )
        if topology["topology_sha256"] != frontier.context["target_topology_sha256"]:
            raise Lfm2CpcfV5ResumeError(
                "Progress target topology differs from the v5 context."
            )
        installed_paths = build_lfm2_cpcf_installed_parameter_paths(
            target_topology_snapshot=topology,
            next_layer_index=frontier.next_layer_index,
            global_parameter_paths=self._global_installed_paths(topology),
        )
        receipt_maps = [receipt.as_dict() for receipt in event.completed_layer_receipts]
        bindings = build_lfm2_cpcf_checkpoint_bindings(
            boundary=boundary.reference,
            completed_layer_receipts=receipt_maps,
            installed_parameter_paths=installed_paths,
            target_topology_snapshot=topology,
            incoming_state_bank_metadata=incoming_metadata,
            full_policy_sha256=frontier.context["runtime_policy_sha256"],
            compiler_fingerprint_sha256=frontier.context[
                "compiler_implementation_sha256"
            ],
        )
        completed = build_lfm2_cpcf_completed_layers_document(
            bindings=bindings,
            completed_layer_receipts=receipt_maps,
            installed_parameter_paths=installed_paths,
            target_topology_snapshot=topology,
            incoming_state_bank=incoming,
            incoming_state_bank_metadata=incoming_metadata,
        )
        checkpoint = self.store.publish_checkpoint(
            bindings=bindings,
            prefix_model=event.target_prefix_cpu_tensors,
            incoming_state_bank=incoming,
            completed_layers=completed,
        )
        checkpoint = self.store.final_reauthenticate_checkpoint(checkpoint)
        self.store.final_reauthenticate_boundary(boundary)
        event_document = _thaw(event.report)
        event_sha256 = str(event_document["report_sha256"])
        self._write_once(
            self._document_path("v5-progress", event_sha256),
            event_document,
        )
        reopened_event = self._open_document(
            self._document_path("v5-progress", event_sha256),
            kind="progress",
            expected_key=event_sha256,
        )
        if reopened_event != event_document:
            raise Lfm2CpcfV5ResumeError(
                "Persisted progress event differs from its exact live event."
            )
        capability_token = secrets.token_hex(32)
        capability_sha256 = hashlib.sha256(capability_token.encode("ascii")).hexdigest()
        pending_body = {
            "schema": LFM2_CPCF_V5_PENDING_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "coordinator_identity_sha256": (self._coordinator_identity_sha256),
            "capability_sha256": capability_sha256,
            "frontier_reference": boundary.reference.as_dict(),
            "checkpoint_reference": checkpoint.reference.as_dict(),
            "frontier_sha256": frontier.frontier_sha256,
            "event_report_sha256": event_sha256,
            "next_layer_index": frontier.next_layer_index,
            "rss_checkpoint_expectation": rss_expectation,
            "parent_or_cache_tensors_archived_in_frontier": 0,
            "progress_event_persisted_and_reauthenticated": True,
            "operation_audit": dict(_ZERO_AUDIT),
        }
        pending_sha256 = canonical_sha256(pending_body)
        pending_document = {
            **pending_body,
            "pending_sha256": pending_sha256,
        }
        self._write_once(
            self._document_path("v5-pending", pending_sha256),
            pending_document,
        )
        self._open_document(
            self._document_path("v5-pending", pending_sha256),
            kind="pending",
            expected_key=pending_sha256,
        )
        self._pending_capabilities[pending_sha256] = capability_token
        self._pending_rss_authorities[pending_sha256] = (
            rss_checkpoint_expectation._guard_authority,
            rss_checkpoint_expectation._sequence_capability,
        )
        return Lfm2CpcfV5PendingCheckpoint(
            frontier_reference=boundary.reference,
            checkpoint=checkpoint,
            frontier_sha256=frontier.frontier_sha256,
            event_report_sha256=event_sha256,
            next_layer_index=frontier.next_layer_index,
            rss_checkpoint_expectation=_freeze(rss_expectation),
            pending_sha256=pending_sha256,
            coordinator_identity_sha256=(self._coordinator_identity_sha256),
            capability_token=capability_token,
        )

    def _open_pending_document(self, pending_sha256: str) -> dict[str, Any]:
        digest = _digest(pending_sha256, "pending SHA-256")
        document = self._open_document(
            self._document_path("v5-pending", digest),
            kind="pending",
            expected_key=digest,
        )
        expected_fields = {
            "schema",
            "resume_schema",
            "coordinator_identity_sha256",
            "capability_sha256",
            "frontier_reference",
            "checkpoint_reference",
            "frontier_sha256",
            "event_report_sha256",
            "next_layer_index",
            "rss_checkpoint_expectation",
            "parent_or_cache_tensors_archived_in_frontier",
            "progress_event_persisted_and_reauthenticated",
            "operation_audit",
            "pending_sha256",
        }
        if (
            set(document) != expected_fields
            or document.get("operation_audit") != _ZERO_AUDIT
            or document.get("parent_or_cache_tensors_archived_in_frontier") != 0
            or document.get("progress_event_persisted_and_reauthenticated") is not True
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 pending document has unexpected authority fields."
            )
        _digest(
            document.get("coordinator_identity_sha256"),
            "pending coordinator identity",
        )
        _digest(document.get("capability_sha256"), "pending capability")
        _digest(document.get("frontier_sha256"), "pending frontier")
        _digest(document.get("event_report_sha256"), "pending event")
        _rss_checkpoint_expectation(
            document.get("rss_checkpoint_expectation"),
            next_layer_index=_nonnegative_int(
                document.get("next_layer_index"),
                "pending next_layer_index",
            ),
        )
        return document

    def _recompute_progress_event(
        self,
        *,
        frontier: Lfm2CpcfV5BoundaryFrontier,
        checkpoint: VerifiedLfm2CpcfPrefixCheckpoint,
    ) -> dict[str, Any]:
        completed = checkpoint.completed_layers
        topology = completed["target_topology_snapshot"]
        metadata = {
            "schema": "neural-forge-model-tensor-snapshot-metadata-v1",
            "all_named_parameters_and_buffers": True,
            "nonpersistent_buffers_included": True,
            "tensor_metadata": topology["tensor_metadata"],
            "object_alias_groups": topology["object_alias_groups"],
            "storage_alias_groups": topology["storage_alias_groups"],
        }
        snapshot_receipt = model_fold._cpu_tensor_snapshot_receipt_from_parts(
            checkpoint.prefix_model,
            _thaw(metadata),
        )
        body = {
            "schema": LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "frontier_sha256": frontier.frontier_sha256,
            "context_sha256": frontier.context["context_sha256"],
            "next_layer_index": frontier.next_layer_index,
            "incoming_state_bank_sha256": completed["incoming_state_bank_metadata"][
                "state_bank_sha256"
            ],
            "completed_layer_result_sha256": [
                receipt["result_sha256"]
                for receipt in completed["completed_layer_receipts"]
            ],
            "target_prefix_snapshot_receipt": dict(snapshot_receipt),
            "pending_rss_admission": True,
            "rss_admission_claimed": False,
            "parent_or_cache_tensors_archived": 0,
            "operation_audit": dict(_ZERO_AUDIT),
        }
        return {**body, "report_sha256": canonical_sha256(body)}

    def _reauthenticate_pending_artifacts(
        self,
        pending_document: Mapping[str, Any],
    ) -> tuple[
        Lfm2CpcfV5BoundaryFrontier,
        VerifiedLfm2CpcfPrefixCheckpoint,
        Mapping[str, Any],
    ]:
        frontier_reference = _reference(
            pending_document["frontier_reference"],
            expected_kind="boundary",
        )
        checkpoint_reference = _reference(
            pending_document["checkpoint_reference"],
            expected_kind="checkpoint",
        )
        verified_boundary = self.store.final_reauthenticate_boundary(
            self.store.open_boundary(frontier_reference)
        )
        frontier = Lfm2CpcfV5BoundaryFrontier(
            report=verified_boundary.reports,
            tensors=verified_boundary.tensors,
        )
        checkpoint = self.store.final_reauthenticate_checkpoint(
            self.store.open_checkpoint(checkpoint_reference)
        )
        checkpoint_boundary = _reference(
            checkpoint.manifest["bindings"]["boundary"],
            expected_kind="boundary",
        )
        event_sha256 = _digest(
            pending_document["event_report_sha256"],
            "pending event report SHA-256",
        )
        event = self._open_document(
            self._document_path("v5-progress", event_sha256),
            kind="progress",
            expected_key=event_sha256,
        )
        recomputed_event = self._recompute_progress_event(
            frontier=frontier,
            checkpoint=checkpoint,
        )
        rss_expectation = _rss_checkpoint_expectation(
            pending_document["rss_checkpoint_expectation"],
            next_layer_index=frontier.next_layer_index,
        )
        if (
            checkpoint_boundary != frontier_reference
            or pending_document["frontier_sha256"] != frontier.frontier_sha256
            or pending_document["next_layer_index"] != frontier.next_layer_index
            or checkpoint.completed_layers["next_layer_index"]
            != frontier.next_layer_index
            or event != recomputed_event
            or event["report_sha256"] != event_sha256
            or _rss_expectation_policy_sha256(rss_expectation)
            != frontier.context["physical_rss_policy_sha256"]
        ):
            raise Lfm2CpcfV5ResumeError(
                "Pending reference, persisted progress event, boundary, "
                "checkpoint, prefix tensors, or receipt lineage changed."
            )
        return frontier, checkpoint, MappingProxyType(event)

    def _authenticate_pending(
        self,
        pending: Lfm2CpcfV5PendingCheckpoint,
    ) -> tuple[
        Mapping[str, Any],
        Lfm2CpcfV5BoundaryFrontier,
        VerifiedLfm2CpcfPrefixCheckpoint,
    ]:
        if type(pending) is not Lfm2CpcfV5PendingCheckpoint:
            raise TypeError("pending must be the exact V5 pending checkpoint type.")
        token = self._pending_capabilities.get(pending.pending_sha256)
        if (
            pending.coordinator_identity_sha256 != self._coordinator_identity_sha256
            or token is None
            or not hmac.compare_digest(token, pending.capability_token)
        ):
            raise Lfm2CpcfV5ResumeError(
                "Pending checkpoint was not issued by this coordinator."
            )
        document = self._open_pending_document(pending.pending_sha256)
        expected = {
            "coordinator_identity_sha256": (pending.coordinator_identity_sha256),
            "capability_sha256": hashlib.sha256(
                pending.capability_token.encode("ascii")
            ).hexdigest(),
            "frontier_reference": pending.frontier_reference.as_dict(),
            "checkpoint_reference": pending.checkpoint.reference.as_dict(),
            "frontier_sha256": pending.frontier_sha256,
            "event_report_sha256": pending.event_report_sha256,
            "next_layer_index": pending.next_layer_index,
            "rss_checkpoint_expectation": _thaw(pending.rss_checkpoint_expectation),
        }
        if any(document.get(name) != value for name, value in expected.items()):
            raise Lfm2CpcfV5ResumeError(
                "Pending capability differs from its persisted exact identity."
            )
        frontier, checkpoint, _event = self._reauthenticate_pending_artifacts(document)
        return MappingProxyType(document), frontier, checkpoint

    def _authenticate_live_rss_receipt_authority(
        self,
        pending: Lfm2CpcfV5PendingCheckpoint,
        receipt: Lfm2CpcfV5PhysicalRssCheckpointReceipt,
    ) -> None:
        authority = self._pending_rss_authorities.get(pending.pending_sha256)
        if (
            type(receipt) is not Lfm2CpcfV5PhysicalRssCheckpointReceipt
            or authority is None
            or receipt._guard_authority is not authority[0]
            or receipt._sequence_capability is not authority[1]
        ):
            raise Lfm2CpcfV5ResumeError(
                "RSS receipt does not carry the exact live guard and "
                "one-shot checkpoint authority bound to this pending prefix."
            )

    @staticmethod
    def _write_terminal_stage_file(
        path: Path,
        document: Mapping[str, Any],
    ) -> None:
        raw = canonical_json_bytes(document)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
                    raise Lfm2CpcfV5ResumeError(
                        "Terminal staging file is not one regular file."
                    )
                offset = 0
                while offset < len(raw):
                    written = os.write(descriptor, raw[offset:])
                    if written <= 0:
                        raise OSError("short terminal staging write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Lfm2CpcfV5ResumeError:
            raise
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(
                "Could not write terminal staging document."
            ) from exc

    @staticmethod
    def _cleanup_terminal_stage(
        stage: Path,
        *,
        collection: Path,
        identity: tuple[int, int],
    ) -> None:
        if not os.path.lexists(stage):
            return
        current = resume_artifact._real_directory(
            stage,
            name="V5 terminal staging",
        )
        if (
            resume_artifact._stat_identity(current) != identity
            or stage.resolve(strict=True).parent != collection
            or _V5_TERMINAL_STAGE.fullmatch(stage.name) is None
        ):
            raise Lfm2CpcfV5ResumeError(
                "Refusing to clean unauthenticated terminal staging."
            )
        entries = list(os.scandir(stage))
        for entry in entries:
            if (
                entry.name != "terminal.json"
                and re.fullmatch(r"[0-9a-f]{64}\.json", entry.name) is None
            ):
                raise Lfm2CpcfV5ResumeError(
                    "Terminal staging contains a foreign entry."
                )
            resume_artifact._regular_file(
                stage / entry.name,
                name=f"terminal staging {entry.name}",
            )
        try:
            for entry in entries:
                os.unlink(stage / entry.name)
            os.rmdir(stage)
            resume_artifact._fsync_directory(collection)
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(
                "Could not clean authenticated terminal staging."
            ) from exc

    def _read_terminal_file(
        self,
        bundle: Path,
        filename: str,
    ) -> bytes:
        before = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                bundle,
                name="V5 terminal bundle",
            )
        )
        raw = resume_artifact._read_regular_bytes(
            bundle / filename,
            name=f"V5 terminal {filename}",
            maximum_bytes=8 * 1024 * 1024,
        )
        after = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                bundle,
                name="V5 terminal bundle",
            )
        )
        if before != after:
            raise Lfm2CpcfV5ResumeError("V5 terminal bundle changed during read.")
        return raw

    def _open_terminal(
        self,
        pending_sha256: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        pending_key = _digest(pending_sha256, "terminal pending SHA-256")
        collection = self._collection("v5-terminals")
        bundle = collection / pending_key
        bundle_stat = resume_artifact._real_directory(
            bundle,
            name="V5 terminal bundle",
        )
        if bundle.resolve(strict=True).parent != collection:
            raise Lfm2CpcfV5ResumeError("V5 terminal bundle escaped its collection.")
        entries = {entry.name for entry in os.scandir(bundle)}
        if "terminal.json" not in entries or len(entries) != 2:
            raise Lfm2CpcfV5ResumeError("V5 terminal bundle inventory is incomplete.")
        try:
            terminal = json.loads(
                self._read_terminal_file(
                    bundle,
                    "terminal.json",
                ).decode("ascii")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2CpcfV5ResumeError("V5 terminal document is not JSON.") from exc
        terminal_raw = canonical_json_bytes(terminal)
        if terminal_raw != self._read_terminal_file(
            bundle,
            "terminal.json",
        ):
            raise Lfm2CpcfV5ResumeError("V5 terminal document is not canonical.")
        terminal_body = dict(terminal)
        terminal_sha = _digest(
            terminal_body.pop("terminal_sha256", None),
            "terminal document SHA-256",
        )
        outcome_kind = terminal.get("outcome_kind")
        outcome_sha = _digest(
            terminal.get("outcome_sha256"),
            "terminal outcome SHA-256",
        )
        outcome_filename = terminal.get("outcome_filename")
        if (
            set(terminal) != _TERMINAL_FIELDS
            or terminal.get("schema") != LFM2_CPCF_V5_TERMINAL_SCHEMA
            or terminal.get("resume_schema") != LFM2_CPCF_V5_RESUME_SCHEMA
            or terminal.get("pending_sha256") != pending_key
            or canonical_sha256(terminal_body) != terminal_sha
            or outcome_kind not in {"admission", "rejection"}
            or outcome_filename != f"{outcome_sha}.json"
            or entries != {"terminal.json", outcome_filename}
            or terminal.get("exactly_one_terminal_outcome") is not True
            or terminal.get("operation_audit") != _ZERO_AUDIT
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 terminal schema, seal, pending key, or outcome is invalid."
            )
        try:
            outcome = json.loads(
                self._read_terminal_file(
                    bundle,
                    outcome_filename,
                ).decode("ascii")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2CpcfV5ResumeError("V5 terminal outcome is not JSON.") from exc
        raw_outcome = canonical_json_bytes(outcome)
        if raw_outcome != self._read_terminal_file(
            bundle,
            outcome_filename,
        ):
            raise Lfm2CpcfV5ResumeError("V5 terminal outcome is not canonical.")
        seal_field = f"{outcome_kind}_sha256"
        outcome_body = dict(outcome)
        claimed = _digest(
            outcome_body.pop(seal_field, None),
            f"terminal {outcome_kind} seal",
        )
        expected_schema = (
            LFM2_CPCF_V5_ADMISSION_SCHEMA
            if outcome_kind == "admission"
            else LFM2_CPCF_V5_REJECTION_SCHEMA
        )
        expected_fields = (
            _ADMISSION_FIELDS if outcome_kind == "admission" else _REJECTION_FIELDS
        )
        rss_payload = outcome.get(
            "rss_checkpoint_receipt"
            if outcome_kind == "admission"
            else "rss_violation_report"
        )
        rss_policy_binding = outcome.get("rss_checkpoint_policy_binding")
        if (
            set(outcome) != expected_fields
            or outcome.get("schema") != expected_schema
            or outcome.get("resume_schema") != LFM2_CPCF_V5_RESUME_SCHEMA
            or outcome.get("pending_sha256") != pending_key
            or claimed != outcome_sha
            or canonical_sha256(outcome_body) != claimed
            or outcome.get("operation_audit") != _ZERO_AUDIT
            or not isinstance(rss_payload, dict)
            or not isinstance(rss_policy_binding, dict)
            or rss_policy_binding != _rss_checkpoint_policy_binding(rss_payload)
            or (
                outcome_kind == "admission"
                and (
                    outcome.get("original_policy_admitted") is not True
                    or outcome.get("readmission_used") is not False
                    or outcome.get("promotion_claimed") is not False
                )
            )
            or (
                outcome_kind == "rejection"
                and (
                    outcome.get("original_policy_admitted") is not False
                    or outcome.get("promotable") is not False
                )
            )
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 terminal outcome schema, seal, or pending key is invalid."
            )
        after = resume_artifact._real_directory(
            bundle,
            name="V5 terminal bundle",
        )
        if resume_artifact._stat_identity(after) != (
            resume_artifact._stat_identity(bundle_stat)
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 terminal bundle changed while it was opened."
            )
        return MappingProxyType(terminal), MappingProxyType(outcome)

    def _publish_terminal(
        self,
        *,
        pending_sha256: str,
        outcome_kind: str,
        outcome: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        pending_key = _digest(pending_sha256, "terminal pending SHA-256")
        if outcome_kind not in {"admission", "rejection"}:
            raise Lfm2CpcfV5ResumeError("Invalid terminal outcome kind.")
        seal_field = f"{outcome_kind}_sha256"
        outcome_sha = _digest(
            outcome.get(seal_field),
            f"{outcome_kind} SHA-256",
        )
        collection = self._collection("v5-terminals")
        destination = collection / pending_key
        stage = collection / (f".v5-terminal-{secrets.token_hex(32)}.tmp")
        try:
            stage.mkdir(mode=0o700)
            identity = resume_artifact._stat_identity(
                resume_artifact._real_directory(
                    stage,
                    name="V5 terminal staging",
                )
            )
        except OSError as exc:
            raise Lfm2CpcfV5ResumeError(
                "Could not create exclusive terminal staging."
            ) from exc
        terminal_body = {
            "schema": LFM2_CPCF_V5_TERMINAL_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "pending_sha256": pending_key,
            "outcome_kind": outcome_kind,
            "outcome_sha256": outcome_sha,
            "outcome_filename": f"{outcome_sha}.json",
            "exactly_one_terminal_outcome": True,
            "operation_audit": dict(_ZERO_AUDIT),
        }
        terminal = {
            **terminal_body,
            "terminal_sha256": canonical_sha256(terminal_body),
        }
        try:
            self._write_terminal_stage_file(
                stage / "terminal.json",
                terminal,
            )
            self._write_terminal_stage_file(
                stage / f"{outcome_sha}.json",
                outcome,
            )
            resume_artifact._fsync_directory(stage)
            try:
                resume_artifact._atomic_rename_noreplace(
                    stage,
                    destination,
                )
            except FileExistsError:
                _existing_terminal, existing_outcome = self._open_terminal(pending_key)
                if dict(existing_outcome) != dict(outcome):
                    raise Lfm2CpcfV5ResumeError(
                        "Pending checkpoint already has the opposite or "
                        "different terminal outcome."
                    )
            except OSError as exc:
                if not os.path.lexists(destination):
                    raise Lfm2CpcfV5ResumeError(
                        "Atomic terminal publication failed before "
                        "installing an outcome."
                    ) from exc
                _existing_terminal, existing_outcome = self._open_terminal(pending_key)
                if dict(existing_outcome) != dict(outcome):
                    raise Lfm2CpcfV5ResumeError(
                        "Pending checkpoint already has the opposite or "
                        "different terminal outcome."
                    ) from exc
            resume_artifact._fsync_directory(collection)
        finally:
            if os.path.lexists(stage):
                self._cleanup_terminal_stage(
                    stage,
                    collection=collection,
                    identity=identity,
                )
        _terminal, published = self._open_terminal(pending_key)
        if dict(published) != dict(outcome):
            raise Lfm2CpcfV5ResumeError(
                "Published terminal outcome changed after atomic install."
            )
        return published

    def admit(
        self,
        pending: Lfm2CpcfV5PendingCheckpoint,
        *,
        rss_checkpoint_receipt: (Lfm2CpcfV5PhysicalRssCheckpointReceipt),
    ) -> Mapping[str, Any]:
        """Atomically authorize one exact pending prefix after RSS success."""

        pending_document, _frontier, _checkpoint = self._authenticate_pending(pending)
        expectation = _rss_checkpoint_expectation(
            pending_document["rss_checkpoint_expectation"],
            next_layer_index=pending.next_layer_index,
        )
        rss = _authenticate_rss_checkpoint(
            rss_checkpoint_receipt,
            expectation=expectation,
            passed=True,
        )
        self._authenticate_live_rss_receipt_authority(
            pending,
            rss_checkpoint_receipt,
        )
        body = {
            "schema": LFM2_CPCF_V5_ADMISSION_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "pending_sha256": pending.pending_sha256,
            "frontier_reference": pending.frontier_reference.as_dict(),
            "checkpoint_reference": pending.checkpoint.reference.as_dict(),
            "frontier_sha256": pending.frontier_sha256,
            "event_report_sha256": pending.event_report_sha256,
            "next_layer_index": pending.next_layer_index,
            "rss_checkpoint_expectation_sha256": expectation["expectation_sha256"],
            "rss_checkpoint_policy_binding": (
                _rss_checkpoint_policy_binding(expectation)
            ),
            "rss_checkpoint_receipt": rss,
            "rss_checkpoint_report_sha256": rss["report_sha256"],
            "pending_checkpoint_reauthenticated": True,
            "persisted_progress_event_recomputed": True,
            "original_policy_admitted": True,
            "readmission_used": False,
            "promotion_claimed": False,
            "operation_audit": dict(_ZERO_AUDIT),
        }
        document = {
            **body,
            "admission_sha256": canonical_sha256(body),
        }
        return self._publish_terminal(
            pending_sha256=pending.pending_sha256,
            outcome_kind="admission",
            outcome=document,
        )

    def record_rss_rejection(
        self,
        pending: Lfm2CpcfV5PendingCheckpoint,
        *,
        rss_violation_report: (Lfm2CpcfV5PhysicalRssCheckpointReceipt),
    ) -> Mapping[str, Any]:
        """Atomically retain one non-promotable RSS terminal rejection."""

        pending_document, _frontier, _checkpoint = self._authenticate_pending(pending)
        expectation = _rss_checkpoint_expectation(
            pending_document["rss_checkpoint_expectation"],
            next_layer_index=pending.next_layer_index,
        )
        report = _authenticate_rss_checkpoint(
            rss_violation_report,
            expectation=expectation,
            passed=False,
        )
        self._authenticate_live_rss_receipt_authority(
            pending,
            rss_violation_report,
        )
        body = {
            "schema": LFM2_CPCF_V5_REJECTION_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "pending_sha256": pending.pending_sha256,
            "frontier_reference": pending.frontier_reference.as_dict(),
            "checkpoint_reference": pending.checkpoint.reference.as_dict(),
            "frontier_sha256": pending.frontier_sha256,
            "event_report_sha256": pending.event_report_sha256,
            "next_layer_index": pending.next_layer_index,
            "rss_checkpoint_expectation_sha256": expectation["expectation_sha256"],
            "rss_checkpoint_policy_binding": (
                _rss_checkpoint_policy_binding(expectation)
            ),
            "rss_violation_report": report,
            "rss_violation_report_sha256": report["report_sha256"],
            "pending_checkpoint_reauthenticated": True,
            "persisted_progress_event_recomputed": True,
            "original_policy_admitted": False,
            "promotable": False,
            "operation_audit": dict(_ZERO_AUDIT),
        }
        document = {
            **body,
            "rejection_sha256": canonical_sha256(body),
        }
        return self._publish_terminal(
            pending_sha256=pending.pending_sha256,
            outcome_kind="rejection",
            outcome=document,
        )

    def _terminal_pending_keys(self) -> tuple[str, ...]:
        collection = self._collection("v5-terminals")
        before = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                collection,
                name="V5 terminal collection",
            )
        )
        keys = []
        for entry in os.scandir(collection):
            path = collection / entry.name
            if _V5_TERMINAL_STAGE.fullmatch(entry.name):
                resume_artifact._real_directory(
                    path,
                    name="stale V5 terminal staging",
                )
                continue
            if _SHA256.fullmatch(entry.name) is None:
                raise Lfm2CpcfV5ResumeError(
                    "V5 terminal collection contains a foreign entry."
                )
            resume_artifact._real_directory(
                path,
                name="V5 terminal bundle",
            )
            keys.append(entry.name)
        after = resume_artifact._stat_identity(
            resume_artifact._real_directory(
                collection,
                name="V5 terminal collection",
            )
        )
        if before != after:
            raise Lfm2CpcfV5ResumeError(
                "V5 terminal collection changed during enumeration."
            )
        return tuple(sorted(keys))

    def open_admitted(
        self,
        admission_sha256: str,
    ) -> Lfm2CpcfV5ResumeState:
        """Open only one atomically terminalized admitted prefix."""

        admission_key = _digest(
            admission_sha256,
            "admission SHA-256",
        )
        matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for pending_key in self._terminal_pending_keys():
            terminal, outcome = self._open_terminal(pending_key)
            if (
                terminal["outcome_kind"] == "admission"
                and outcome.get("admission_sha256") == admission_key
            ):
                matches.append((terminal, outcome))
        if len(matches) != 1:
            raise Lfm2CpcfV5ResumeError(
                "Requested admission digest is not exactly one admitted "
                "terminal V5 prefix; pending and rejected prefixes fail "
                "closed."
            )
        _terminal, admission = matches[0]
        pending_document = self._open_pending_document(admission["pending_sha256"])
        frontier, checkpoint, _event = self._reauthenticate_pending_artifacts(
            pending_document
        )
        expectation = _rss_checkpoint_expectation(
            pending_document["rss_checkpoint_expectation"],
            next_layer_index=frontier.next_layer_index,
        )
        rss = _authenticate_rss_checkpoint(
            admission["rss_checkpoint_receipt"],
            expectation=expectation,
            passed=True,
            require_typed=False,
        )
        expected = {
            "frontier_reference": pending_document["frontier_reference"],
            "checkpoint_reference": pending_document["checkpoint_reference"],
            "frontier_sha256": pending_document["frontier_sha256"],
            "event_report_sha256": pending_document["event_report_sha256"],
            "next_layer_index": pending_document["next_layer_index"],
            "rss_checkpoint_expectation_sha256": expectation["expectation_sha256"],
            "rss_checkpoint_policy_binding": (
                _rss_checkpoint_policy_binding(expectation)
            ),
            "rss_checkpoint_report_sha256": rss["report_sha256"],
        }
        if (
            any(admission.get(name) != value for name, value in expected.items())
            or admission.get("original_policy_admitted") is not True
            or admission.get("pending_checkpoint_reauthenticated") is not True
            or admission.get("persisted_progress_event_recomputed") is not True
            or admission.get("promotion_claimed") is not False
        ):
            raise Lfm2CpcfV5ResumeError(
                "V5 admission differs from its terminal pending/artifact/RSS authority."
            )
        return Lfm2CpcfV5ResumeState(
            coordinator=self,
            frontier=frontier,
            frontier_reference=_reference(
                pending_document["frontier_reference"],
                expected_kind="boundary",
            ),
            checkpoint_reference=checkpoint.reference,
            admission=_freeze(admission),
        )

    def restore(
        self,
        state: Lfm2CpcfV5ResumeState,
        *,
        target_model: Any,
        phases: Sequence[Lfm2CpcfTargetPhase],
        boundary_compiler: Any,
        expected_context: Mapping[str, Any],
        expected_blank_state_sha256: str,
    ) -> Lfm2CpcfV5RestoredPrefix:
        """Install and hydrate an admitted prefix into an exact fresh blank."""

        if (
            not isinstance(state, Lfm2CpcfV5ResumeState)
            or state.coordinator.store.root != self.store.root
        ):
            raise Lfm2CpcfV5ResumeError(
                "Resume state belongs to a different coordinator/store."
            )
        reopened = self.open_admitted(str(state.admission["admission_sha256"]))
        if _thaw(reopened.frontier.report) != _thaw(state.frontier.report) or _context(
            expected_context
        ) != _thaw(state.frontier.context):
            raise Lfm2CpcfV5ResumeError(
                "Live resume context differs from the admitted frontier."
            )
        install_report = install_lfm2_cpcf_prefix_checkpoint(
            self.store,
            reopened.checkpoint_reference,
            target_model,
            expected_blank_state_sha256=expected_blank_state_sha256,
        )
        incoming = reconstruct_lfm2_cpcf_phase_state_bank(
            self.store,
            reopened.checkpoint_reference,
            phases=phases,
        )
        receipts = tuple(
            Lfm2CpcfCompletedLayerReceipt(_thaw(value))
            for value in self.store.open_checkpoint(
                reopened.checkpoint_reference
            ).completed_layers["completed_layer_receipts"]
        )
        seed = seed_lfm2_cpcf_phase_states(target_model, phases)
        if (not receipts and incoming.state_bank_sha256 != seed.state_bank_sha256) or (
            receipts
            and (
                receipts[0].input_bank_sha256 != seed.state_bank_sha256
                or receipts[-1].next_bank_sha256 != incoming.state_bank_sha256
            )
        ):
            raise Lfm2CpcfV5ResumeError(
                "Restored receipt chain does not begin at the installed "
                "interface or end at the incoming state bank."
            )
        reopened.frontier.hydrate(boundary_compiler)
        return Lfm2CpcfV5RestoredPrefix(
            frontier=reopened.frontier,
            state_bank=incoming,
            completed_layer_receipts=receipts,
            prefix_install_report=install_report,
            admission=reopened.admission,
        )


def _reference(
    value: Any,
    *,
    expected_kind: str,
) -> Lfm2CpcfArtifactReference:
    mapping = _mapping(value, f"{expected_kind} reference")
    if (
        set(mapping)
        != {
            "artifact_kind",
            "artifact_key",
            "manifest_sha256",
        }
        or mapping.get("artifact_kind") != expected_kind
    ):
        raise Lfm2CpcfV5ResumeError(f"V5 {expected_kind} reference is invalid.")
    return Lfm2CpcfArtifactReference(
        artifact_kind=mapping["artifact_kind"],
        artifact_key=mapping["artifact_key"],
        manifest_sha256=mapping["manifest_sha256"],
    )


__all__ = [
    "LFM2_CPCF_V5_ADMISSION_SCHEMA",
    "LFM2_CPCF_V5_BOUNDARY_FRONTIER_SCHEMA",
    "LFM2_CPCF_V5_PROGRESS_EVENT_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_POLICY_SCHEMA",
    "LFM2_CPCF_V5_REJECTION_SCHEMA",
    "LFM2_CPCF_V5_RESUME_IMPLEMENTATION_SCHEMA",
    "LFM2_CPCF_V5_RESUME_SCHEMA",
    "Lfm2CpcfV5BoundaryFrontier",
    "Lfm2CpcfV5PendingCheckpoint",
    "Lfm2CpcfV5PhysicalRssCheckpointExpectation",
    "Lfm2CpcfV5PhysicalRssCheckpointReceipt",
    "Lfm2CpcfV5ProgressEvent",
    "Lfm2CpcfV5RestoredPrefix",
    "Lfm2CpcfV5ResumeCoordinator",
    "Lfm2CpcfV5ResumeError",
    "Lfm2CpcfV5ResumeState",
    "build_lfm2_cpcf_v5_physical_rss_policy_sha256",
    "build_lfm2_cpcf_v5_resume_context",
    "lfm2_cpcf_v5_resume_implementation",
]
