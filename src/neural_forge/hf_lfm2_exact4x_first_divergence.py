"""Cached-evidence-only first-divergence ledger for an exact-budget LFM2 child.

The diagnostic runs one complete folded child on donor-token-time embeddings
reconstructed by its authenticated C_in route.  Parent component endpoints and
logits come only from an immutable circuit-trace cache.  Numeric measurements
describe where degradation grows; they never accept or reject intelligence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_balanced_basis_phase_cache import (
    LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
    AuthenticatedLfm2BalancedBasisPhase,
)
from .hf_lfm2_circuit_trace_cache import (
    SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
    Lfm2ChildEmbeddingReplay,
    Lfm2CircuitTraceSelectedBatchPayload,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from .hf_lfm2_direct_local import (
    count_lfm2_direct_local_unique_parameter_elements,
)
from .hf_lfm2_semantic_step_token_fold import (
    LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
    Lfm2SemanticStepLiveTableSession,
    Lfm2SemanticStepRoutingRuntime,
)
from .hf_lfm2_teacher_forced_trajectory_audit import parent_winner_rank_margin
from .lfm2_carrier_bridge_publication import (
    PACKAGED_ROUTE_SCHEMA,
    PORTABLE_CARRIER_RUNTIME_SCHEMA,
    AuthenticatedPackagedLfm2CarrierRoutes,
    Lfm2PortableCarrierRuntimeReceipt,
)


LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA = (
    "neural-forge-lfm2-exact4x-first-divergence-implementation-v2"
)
LFM2_EXACT4X_FIRST_DIVERGENCE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-exact4x-first-divergence-request-v2"
)
LFM2_EXACT4X_FIRST_DIVERGENCE_BATCH_SCHEMA = (
    "neural-forge-lfm2-exact4x-first-divergence-batch-v2"
)
LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA = (
    "neural-forge-lfm2-exact4x-first-divergence-ledger-v2"
)
LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA = (
    "neural-forge-lfm2-exact-budget-child-observation-v2"
)

STATE_BOUNDARY_SUFFIXES = ("residual.post_operator", "residual.post_layer")
BRANCH_BOUNDARY_SUFFIXES = ("operator.output", "mlp.down")
RESIDUAL_ENDPOINTS = (*BRANCH_BOUNDARY_SUFFIXES, *STATE_BOUNDARY_SUFFIXES)
SPLITS = ("fit", "heldout")


class Lfm2Exact4xFirstDivergenceError(RuntimeError):
    """Cached evidence, exact-budget state, or diagnostic geometry is invalid."""


def _plain_json(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xFirstDivergenceError(
            "First-divergence evidence must be finite canonical JSON."
        ) from exc


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2Exact4xFirstDivergenceError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2Exact4xFirstDivergenceError(f"{name} must be a nonnegative integer.")
    return value


def _layer_indices(values: Sequence[int], layer_count: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise Lfm2Exact4xFirstDivergenceError(
            "Layer indices must be an integer sequence."
        )
    result = tuple(values)
    if (
        not result
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in result
        )
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
        or result[0] < 0
        or result[-1] >= layer_count
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Layer indices must be unique, sorted, and inside the complete child."
        )
    return result


def _verify_seal(
    value: Mapping[str, Any], seal_name: str, schema: str
) -> dict[str, Any]:
    row = _plain_json(dict(value))
    if not isinstance(row, dict) or row.get("schema") != schema:
        raise Lfm2Exact4xFirstDivergenceError(
            f"{schema} evidence has the wrong schema."
        )
    claimed = row.pop(seal_name, None)
    if claimed != canonical_sha256(row):
        raise Lfm2Exact4xFirstDivergenceError(
            f"{schema} evidence does not verify its {seal_name}."
        )
    return {**row, seal_name: claimed}


def phase_names(forced_decode_count: int) -> tuple[str, ...]:
    count = _nonnegative_int(forced_decode_count, "forced_decode_count")
    return ("prefill", *(f"decode.{index:03d}" for index in range(count)))


def parent_tensor_names_for_first_divergence(
    reference: Any,
    *,
    layer_count: int,
    focus_layer_indices: Sequence[int],
) -> tuple[str, ...]:
    """Return complete state boundaries plus focused branch diagnostics.

    Only layer zero's ``residual.pre`` is C_in.  Later ``residual.pre`` tensors
    duplicate the preceding layer's post-layer state and are intentionally not
    selected.  Focus layers narrow branch-local operator/MLP diagnostics only;
    every layer's state boundaries and the final logits remain observed.
    """

    record = getattr(reference, "record", None)
    if not isinstance(record, Mapping):
        raise Lfm2Exact4xFirstDivergenceError(
            "Trace reference has no authenticated record."
        )
    capture = record.get("capture")
    if not isinstance(capture, Mapping):
        raise Lfm2Exact4xFirstDivergenceError(
            "Trace reference has no capture contract."
        )
    decode_steps = capture.get("decode_steps")
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps < 0
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Trace reference decode horizon is invalid."
        )
    count = _positive_int(layer_count, "layer_count")
    try:
        proposed_focus = tuple(focus_layer_indices)
    except TypeError as exc:
        raise Lfm2Exact4xFirstDivergenceError(
            "Focus layer indices must be an integer sequence."
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in proposed_focus
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Focus layer indices must contain only integers."
        )
    focus_layers = _layer_indices(proposed_focus, count)
    complete_layers = tuple(range(count))
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    inventory = set(getattr(reference, "ordered_tensor_names", ()))
    if "input.position_ids" in inventory:
        names.add("input.position_ids")
    for phase in phase_names(decode_steps):
        names.add(f"{phase}.logits")
        names.add(f"{phase}.layer.000.residual.pre")
        for layer_index in complete_layers:
            for suffix in STATE_BOUNDARY_SUFFIXES:
                names.add(f"{phase}.layer.{layer_index:03d}.{suffix}")
        for layer_index in focus_layers:
            for suffix in BRANCH_BOUNDARY_SUFFIXES:
                names.add(f"{phase}.layer.{layer_index:03d}.{suffix}")
    ordered = tuple(sorted(names))
    missing = tuple(name for name in ordered if name not in inventory)
    if missing:
        raise Lfm2Exact4xFirstDivergenceError(
            f"Trace reference is missing first-divergence tensors: {missing!r}."
        )
    return ordered


def parent_selection_logical_bytes(
    reference: Any,
    tensor_names: Sequence[str],
) -> int:
    record = getattr(reference, "record", None)
    inventory = record.get("tensors") if isinstance(record, Mapping) else None
    if not isinstance(inventory, Mapping):
        raise Lfm2Exact4xFirstDivergenceError(
            "Trace reference has no tensor inventory."
        )
    total = 0
    for name in tensor_names:
        descriptor = inventory.get(name)
        size = descriptor.get("size_bytes") if isinstance(descriptor, Mapping) else None
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise Lfm2Exact4xFirstDivergenceError(
                f"Trace tensor {name!r} has an invalid byte descriptor."
            )
        total += size
    return total


def authoritative_split_by_batch(
    phase_report: Mapping[str, Any],
    references: Sequence[Any],
) -> Mapping[int, str]:
    """Bind FIT/heldout only to the authenticated basis-phase record split."""

    cache = phase_report.get("cache") if isinstance(phase_report, Mapping) else None
    rows = cache.get("record_horizons") if isinstance(cache, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(references):
        raise Lfm2Exact4xFirstDivergenceError(
            "Balanced-basis report has no complete record-horizon split."
        )
    by_index: dict[int, str] = {}
    for row in rows:
        index = row.get("batch_index") if isinstance(row, Mapping) else None
        split = row.get("split") if isinstance(row, Mapping) else None
        record_id = row.get("record_id") if isinstance(row, Mapping) else None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(references)
            or index in by_index
            or split not in SPLITS
            or record_id != getattr(references[index], "input_sha256", None)
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                "Balanced-basis FIT/heldout rows differ from trace references."
            )
        by_index[index] = str(split)
    if set(by_index) != set(range(len(references))) or set(by_index.values()) != set(
        SPLITS
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Balanced-basis split must cover every batch and both evidence splits."
        )
    return MappingProxyType(dict(sorted(by_index.items())))


def _authenticate_route_package(
    route_package: AuthenticatedPackagedLfm2CarrierRoutes,
    *,
    expected_package_sha256: str | None = None,
) -> tuple[dict[str, Any], tuple[Any, Any], int]:
    if not isinstance(route_package, AuthenticatedPackagedLfm2CarrierRoutes):
        raise Lfm2Exact4xFirstDivergenceError(
            "C_in/C_out routes must be an authenticated packaged route object."
        )
    receipt = _verify_seal(
        route_package.receipt,
        "package_sha256",
        PACKAGED_ROUTE_SCHEMA,
    )
    if receipt.get("status") != "complete":
        raise Lfm2Exact4xFirstDivergenceError(
            "C_in/C_out route package is not complete."
        )
    package_sha256 = str(receipt["package_sha256"])
    if expected_package_sha256 is not None and package_sha256 != _digest(
        expected_package_sha256,
        "expected route_package_sha256",
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "C_in/C_out routes belong to a different requested route package."
        )
    allocations = tuple(route_package.allocations)
    if len(allocations) != 2:
        raise Lfm2Exact4xFirstDivergenceError(
            "Route package must own exactly one C_in and one C_out allocation."
        )
    expected_state_ids = (
        "semantic-carrier-route.c_in",
        "semantic-carrier-route.c_out",
    )
    try:
        allocation_rows = tuple(value.as_dict() for value in allocations)
        route_elements = sum(int(value.parameter_elements) for value in allocations)
    except (AttributeError, TypeError, ValueError) as exc:
        raise Lfm2Exact4xFirstDivergenceError(
            "Carrier route allocations are invalid."
        ) from exc
    if (
        tuple(getattr(value, "state_id", None) for value in allocations)
        != expected_state_ids
        or any(getattr(value, "dtype", None) != "float32" for value in allocations)
        or receipt.get("input_route_allocation") != allocation_rows[0]
        or receipt.get("output_route_allocation") != allocation_rows[1]
        or receipt.get("continuous_semantic_coefficient_count") != route_elements
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Sealed C_in/C_out allocations differ from the packaged route state."
        )
    return receipt, (allocations[0], allocations[1]), route_elements


def _authenticate_portable_route_runtime(
    portable_runtime_receipt: Lfm2PortableCarrierRuntimeReceipt,
) -> dict[str, Any]:
    if not isinstance(
        portable_runtime_receipt,
        Lfm2PortableCarrierRuntimeReceipt,
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Carrier runtime must have a typed portable relocation receipt."
        )
    return _verify_seal(
        portable_runtime_receipt.as_dict(),
        "receipt_sha256",
        PORTABLE_CARRIER_RUNTIME_SCHEMA,
    )


def _authenticated_phase_parameter_evidence(
    basis_phase: AuthenticatedLfm2BalancedBasisPhase,
) -> tuple[int, int, dict[str, Any]]:
    if not isinstance(basis_phase, AuthenticatedLfm2BalancedBasisPhase):
        raise Lfm2Exact4xFirstDivergenceError(
            "Parameter authority requires an authenticated balanced-basis phase."
        )
    receipt = _verify_seal(
        basis_phase.receipt,
        "receipt_sha256",
        LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
    )
    bindings = receipt.get("bindings")
    inventory = bindings.get("inventory") if isinstance(bindings, Mapping) else None
    selected_target = (
        bindings.get("selected_target") if isinstance(bindings, Mapping) else None
    )
    if not isinstance(inventory, Mapping) or not isinstance(
        selected_target,
        Mapping,
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Balanced-basis phase lacks sealed inventory or selected-target evidence."
        )
    inventory_body = _plain_json(inventory)
    inventory_sha256 = inventory_body.pop("inventory_sha256", None)
    alias_groups = inventory_body.get("parameter_alias_groups")
    dependencies = basis_phase.phase_key.as_dict().get("dependency_sha256s")
    selected_target_sha256 = canonical_sha256(selected_target)
    if (
        inventory_sha256 != canonical_sha256(inventory_body)
        or not isinstance(alias_groups, list)
        or not alias_groups
        or not isinstance(dependencies, Mapping)
        or dependencies.get("parent_inventory") != inventory_sha256
        or dependencies.get("selected_target") != selected_target_sha256
        or selected_target.get("hidden_size") != basis_phase.target_width
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Balanced-basis parameter inventory or selected target does not verify."
        )
    canonical_paths: set[str] = set()
    parent_total = 0
    for row in alias_groups:
        canonical_path = row.get("canonical_path") if isinstance(row, Mapping) else None
        numel = row.get("numel") if isinstance(row, Mapping) else None
        if (
            not isinstance(canonical_path, str)
            or not canonical_path
            or canonical_path in canonical_paths
            or isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel <= 0
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                "Balanced-basis unique parameter alias inventory is invalid."
            )
        canonical_paths.add(canonical_path)
        parent_total += numel
    dense_target = _positive_int(
        selected_target.get("exact_target_unique_parameters"),
        "selected_target.exact_target_unique_parameters",
    )
    evidence = {
        "basis_phase_key_sha256": basis_phase.phase_key.phase_key_sha256,
        "basis_completion_sha256": basis_phase.completion_sha256,
        "basis_receipt_sha256": receipt["receipt_sha256"],
        "parent_inventory_sha256": inventory_sha256,
        "selected_target_sha256": selected_target_sha256,
        "selected_target_exact_dense_parameter_elements": dense_target,
        "parent_unique_alias_group_count": len(canonical_paths),
    }
    return parent_total, dense_target, evidence


def verify_complete_exact_budget_child(
    child_model: Any,
    *,
    basis_phase: AuthenticatedLfm2BalancedBasisPhase,
    route_package: AuthenticatedPackagedLfm2CarrierRoutes,
    portable_runtime_receipt: Lfm2PortableCarrierRuntimeReceipt,
) -> dict[str, Any]:
    """Authenticate exact-4x counts from sealed phase/package evidence."""

    parent_total, expected_dense, phase_evidence = (
        _authenticated_phase_parameter_evidence(basis_phase)
    )
    route_receipt, allocations, expected_routes = _authenticate_route_package(
        route_package
    )
    portable = _authenticate_portable_route_runtime(portable_runtime_receipt)
    observed_dense = count_lfm2_direct_local_unique_parameter_elements(child_model)
    expected_total = expected_dense + expected_routes
    allocation_rows = [value.as_dict() for value in allocations]
    if (
        observed_dense != expected_dense
        or portable.get("carrier_route_package_sha256")
        != route_receipt["package_sha256"]
        or portable.get("route_allocations") != allocation_rows
        or portable.get("canonical_dense_unique_parameter_elements") != expected_dense
        or portable.get("runtime_dense_unique_parameter_elements") != expected_dense
        or portable.get("counted_carrier_parameter_elements") != expected_routes
        or portable.get("counted_total_parameter_elements") != expected_total
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Observed complete child differs from sealed selected-target, route, "
            "or portable-runtime parameter evidence."
        )
    if parent_total < 4 * expected_total:
        raise Lfm2Exact4xFirstDivergenceError(
            "Exact-4x diagnostic requires at least fourfold parameter compaction."
        )
    body = {
        "schema": LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
        "parent_parameter_elements": parent_total,
        "physical_dense_parameter_elements": observed_dense,
        "continuous_route_parameter_elements": expected_routes,
        "counted_total_parameter_elements": expected_total,
        "target_total_parameter_elements": expected_total,
        "parameter_compaction_ratio": parent_total / expected_total,
        "parameter_count_authority": {
            **phase_evidence,
            "route_package_sha256": route_receipt["package_sha256"],
            "portable_runtime_receipt_sha256": portable["receipt_sha256"],
            "caller_parameter_count_assertion_used": False,
        },
        "at_least_fourfold_parameter_compaction": True,
        "one_complete_child_observed": True,
        "partial_component_variant": False,
        "quantization_credit": False,
        "pruning_credit": False,
        "parameter_sharing_credit": False,
    }
    return {**body, "budget_observation_sha256": canonical_sha256(body)}


def _authenticate_implementation_identity(
    value: Mapping[str, Any],
    *,
    runtime_device: str,
    runtime_dtype: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xFirstDivergenceError(
            "Implementation identity must be a sealed mapping."
        )
    identity = _verify_seal(
        value,
        "implementation_sha256",
        LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
    )
    required_files = {
        "balanced_basis_phase",
        "canonical_lfm2",
        "carrier_child_package",
        "carrier_relocation_package",
        "core",
        "direct_local_loader",
        "dynamic_blank_execution",
        "routing_runtime",
        "runner",
        "teacher_forced_readout",
        "trace_cache",
        "transformers_lfm2_runtime",
    }
    files = identity.get("files")
    packages = identity.get("packages")
    numeric = identity.get("numeric_contract")
    execution = identity.get("execution_contract")
    if (
        not isinstance(files, Mapping)
        or set(files) != required_files
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row.get("path")
            or _digest(row.get("sha256"), f"implementation file {name}")
            != row.get("sha256")
            for name, row in files.items()
        )
        or not isinstance(packages, Mapping)
        or set(packages) != {"python", "safetensors", "torch", "transformers"}
        or any(not isinstance(item, str) or not item for item in packages.values())
        or not isinstance(numeric, Mapping)
        or numeric.get("runtime_device") != runtime_device
        or numeric.get("runtime_dtype") != runtime_dtype
        or not isinstance(execution, Mapping)
        or execution
        != {
            "network_allowed": False,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        }
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Implementation identity does not bind the complete execution/runtime contract."
        )
    return identity


def build_lfm2_exact4x_first_divergence_request(
    *,
    child_package_sha256: str,
    child_state_sha256: str,
    route_package_sha256: str,
    route_runtime_sha256: str,
    portable_runtime_receipt_sha256: str,
    trace_cache_key: str,
    trace_selective_session_sha256: str,
    basis_phase_key_sha256: str,
    basis_completion_sha256: str,
    basis_map_sha256: str,
    budget_observation: Mapping[str, Any],
    layer_count: int,
    focus_layer_indices: Sequence[int],
    runtime_device: str,
    runtime_dtype: str,
    maximum_selected_parent_bytes: int,
    implementation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    count = _positive_int(layer_count, "layer_count")
    focus = _layer_indices(focus_layer_indices, count)
    maximum_bytes = _positive_int(
        maximum_selected_parent_bytes,
        "maximum_selected_parent_bytes",
    )
    for value, name in (
        (child_package_sha256, "child_package_sha256"),
        (child_state_sha256, "child_state_sha256"),
        (route_package_sha256, "route_package_sha256"),
        (route_runtime_sha256, "route_runtime_sha256"),
        (
            portable_runtime_receipt_sha256,
            "portable_runtime_receipt_sha256",
        ),
        (trace_cache_key, "trace_cache_key"),
        (trace_selective_session_sha256, "trace_selective_session_sha256"),
        (basis_phase_key_sha256, "basis_phase_key_sha256"),
        (basis_completion_sha256, "basis_completion_sha256"),
        (basis_map_sha256, "basis_map_sha256"),
    ):
        _digest(value, name)
    budget = _verify_seal(
        budget_observation,
        "budget_observation_sha256",
        LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
    )
    if budget.get("one_complete_child_observed") is not True:
        raise Lfm2Exact4xFirstDivergenceError(
            "Diagnostic request requires one authenticated complete child."
        )
    if not isinstance(runtime_device, str) or not runtime_device:
        raise Lfm2Exact4xFirstDivergenceError("runtime_device must be nonempty.")
    if runtime_dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise Lfm2Exact4xFirstDivergenceError(
            "runtime_dtype must be FP16, BF16, FP32, or FP64."
        )
    implementation = _authenticate_implementation_identity(
        implementation_identity,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_REQUEST_SCHEMA,
        "child_package_sha256": child_package_sha256,
        "child_state_sha256": child_state_sha256,
        "route_package_sha256": route_package_sha256,
        "route_runtime_sha256": route_runtime_sha256,
        "portable_runtime_receipt_sha256": portable_runtime_receipt_sha256,
        "trace_cache_key": trace_cache_key,
        "trace_selective_session_sha256": trace_selective_session_sha256,
        "basis_phase_key_sha256": basis_phase_key_sha256,
        "basis_completion_sha256": basis_completion_sha256,
        "basis_map_sha256": basis_map_sha256,
        "budget_observation": budget,
        "layer_count": count,
        "observed_layer_indices": list(range(count)),
        "focus_layer_indices": list(focus),
        "state_path_suffixes": list(STATE_BOUNDARY_SUFFIXES),
        "branch_suffixes": list(BRANCH_BOUNDARY_SUFFIXES),
        "runtime_device": runtime_device,
        "runtime_dtype": runtime_dtype,
        "metric_dtype": "float64",
        "maximum_selected_parent_bytes": maximum_bytes,
        "implementation_identity": implementation,
        "implementation_sha256": implementation["implementation_sha256"],
        "policy": {
            "sealed_cached_parent_evidence_only": True,
            "parent_forward_calls": 0,
            "one_complete_exact_budget_child_per_observation": True,
            "complete_child_layer_state_path_observed": True,
            "focus_layers_limit_branch_diagnostics_only": True,
            "noninitial_residual_pre_parent_reads": 0,
            "model_mutation_allowed": False,
            "training_allowed": False,
            "numeric_metrics_are_diagnostic_only": True,
            "manual_whole_response_intelligence_review_is_authoritative": True,
            "safety_observations_are_separate_not_failures": True,
            "generation_cap_saturation_is_censored_not_failure": True,
            "free_generation_performed": False,
        },
    }
    return {**body, "request_sha256": canonical_sha256(body)}


def authenticate_lfm2_exact4x_first_divergence_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = _verify_seal(
        value,
        "request_sha256",
        LFM2_EXACT4X_FIRST_DIVERGENCE_REQUEST_SCHEMA,
    )
    implementation = _authenticate_implementation_identity(
        result.get("implementation_identity"),
        runtime_device=result.get("runtime_device"),
        runtime_dtype=result.get("runtime_dtype"),
    )
    if result.get("implementation_sha256") != implementation["implementation_sha256"]:
        raise Lfm2Exact4xFirstDivergenceError(
            "Request implementation digest differs from its complete identity."
        )
    return result


def _last_parent_row(value: Any, width: int, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or int(value.shape[0]) != 1
        or int(value.shape[1]) <= 0
        or int(value.shape[2]) != width
        or not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            f"Parent endpoint {name!r} has invalid geometry."
        )
    return value[0, -1].detach().to(device="cpu", dtype=torch.float64)


def _child_trajectory_row(
    replay: Lfm2ChildEmbeddingReplay,
    name: str,
    phase_index: int,
    width: int,
) -> torch.Tensor:
    value = replay.tensors.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or int(value.shape[0]) != 1
        or phase_index < 0
        or phase_index >= int(value.shape[1])
        or int(value.shape[2]) != width
        or not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            f"Child endpoint {name!r} has invalid donor-time geometry."
        )
    return value[0, phase_index].detach().to(device="cpu", dtype=torch.float64)


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if tuple(candidate.shape) != tuple(reference.shape) or candidate.ndim != 1:
        raise Lfm2Exact4xFirstDivergenceError(
            "Candidate and projected parent rows must have equal vector geometry."
        )
    left = candidate.to(dtype=torch.float64)
    right = reference.to(dtype=torch.float64)
    delta = left - right
    squared = torch.dot(delta, delta)
    l2 = torch.sqrt(squared)
    rmse = torch.sqrt(squared / max(1, int(delta.numel())))
    right_norm = torch.linalg.vector_norm(right)
    left_norm = torch.linalg.vector_norm(left)
    denominator = right_norm.clamp_min(torch.finfo(torch.float64).tiny)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        cosine = 1.0 if bool(torch.equal(left, right)) else 0.0
    else:
        cosine = float(torch.dot(left, right).div(left_norm * right_norm).item())
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "cosine": cosine,
        "cosine_distance": 1.0 - cosine,
        "l2": float(l2.item()),
        "relative_l2": float((l2 / denominator).item()),
        "rmse": float(rmse.item()),
        "candidate_norm": float(left_norm.item()),
        "projected_parent_norm": float(right_norm.item()),
    }


def _project_parent(coordinate_map: Any, parent: torch.Tensor) -> torch.Tensor:
    callback = getattr(coordinate_map, "encode", None)
    source_width = getattr(coordinate_map, "source_width", None)
    target_width = getattr(coordinate_map, "target_width", None)
    if (
        not callable(callback)
        or source_width != int(parent.shape[-1])
        or isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or target_width <= 0
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Coordinate map does not expose the authenticated residual widths."
        )
    projected = callback(parent.reshape(1, int(parent.shape[-1])))
    if (
        not isinstance(projected, torch.Tensor)
        or tuple(projected.shape) != (1, target_width)
        or not bool(torch.isfinite(projected).all())
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Coordinate map returned invalid projected parent state."
        )
    return projected[0].detach().to(device="cpu", dtype=torch.float64)


def _trajectory_replay_geometry(
    parent_tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
    prompt_ids = parent_tensors.get("input.input_ids")
    forced_ids = parent_tensors.get("input.forced_decode_ids")
    attention = parent_tensors.get("input.attention_mask")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(prompt_ids, torch.Tensor)
        or not isinstance(forced_ids, torch.Tensor)
        or not isinstance(attention, torch.Tensor)
        or prompt_ids.ndim != 2
        or forced_ids.ndim != 2
        or attention.ndim != 2
        or prompt_ids.dtype not in integer_dtypes
        or forced_ids.dtype not in integer_dtypes
        or attention.dtype not in integer_dtypes
        or tuple(attention.shape) != tuple(prompt_ids.shape)
        or int(prompt_ids.shape[0]) != 1
        or int(prompt_ids.shape[1]) <= 0
        or int(forced_ids.shape[0]) != 1
        or bool((prompt_ids < 0).any())
        or bool((forced_ids < 0).any())
        or not bool((attention == 1).all())
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Parent donor trajectory IDs or mask are invalid."
        )
    prompt_count = int(prompt_ids.shape[1])
    forced_count = int(forced_ids.shape[1])
    trajectory_ids = torch.cat(
        (prompt_ids.to(torch.long), forced_ids.to(torch.long)),
        dim=1,
    )
    forced_mask = torch.ones(
        (1, forced_count),
        dtype=attention.dtype,
        device=attention.device,
    )
    trajectory_mask = torch.cat((attention, forced_mask), dim=1).to(torch.long)
    prompt_positions = parent_tensors.get("input.position_ids")
    if prompt_positions is None:
        trajectory_positions = None
    else:
        if (
            not isinstance(prompt_positions, torch.Tensor)
            or prompt_positions.ndim != 2
            or prompt_positions.dtype not in integer_dtypes
            or tuple(prompt_positions.shape) != tuple(prompt_ids.shape)
            or bool((prompt_positions < 0).any())
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                "Cached parent position IDs are invalid."
            )
        continuation = prompt_positions[:, -1:].to(torch.long) + torch.arange(
            1,
            forced_count + 1,
            dtype=torch.long,
            device=prompt_positions.device,
        ).unsqueeze(0)
        trajectory_positions = torch.cat(
            (prompt_positions.to(torch.long), continuation),
            dim=1,
        )
    return (
        trajectory_ids,
        trajectory_mask,
        trajectory_positions,
        prompt_count,
        forced_count,
    )


def _authenticate_route_execution_context(
    *,
    request: Mapping[str, Any],
    route_package: AuthenticatedPackagedLfm2CarrierRoutes,
    portable_runtime_receipt: Lfm2PortableCarrierRuntimeReceipt,
    routing: Lfm2SemanticStepRoutingRuntime,
    input_routing_session: Lfm2SemanticStepLiveTableSession,
    output_routing_session: Lfm2SemanticStepLiveTableSession,
) -> dict[str, Any]:
    package, allocations, route_elements = _authenticate_route_package(
        route_package,
        expected_package_sha256=str(request["route_package_sha256"]),
    )
    portable = _authenticate_portable_route_runtime(portable_runtime_receipt)
    if not isinstance(routing, Lfm2SemanticStepRoutingRuntime):
        raise Lfm2Exact4xFirstDivergenceError(
            "C_in/C_out execution requires a typed semantic routing runtime."
        )
    runtime = _verify_seal(
        routing.as_dict(),
        "runtime_sha256",
        LFM2_SEMANTIC_STEP_ROUTING_RUNTIME_SCHEMA,
    )
    if (
        not isinstance(input_routing_session, Lfm2SemanticStepLiveTableSession)
        or not isinstance(output_routing_session, Lfm2SemanticStepLiveTableSession)
        or input_routing_session is output_routing_session
        or input_routing_session._runtime is not routing
        or output_routing_session._runtime is not routing
        or input_routing_session._route_role != "input"
        or output_routing_session._route_role != "output"
        or input_routing_session._state != "open"
        or output_routing_session._state != "open"
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "C_in/C_out require distinct open typed live sessions from the "
            "requested routing runtime."
        )
    allocation_rows = [value.as_dict() for value in allocations]
    if (
        runtime.get("runtime_sha256") != request["route_runtime_sha256"]
        or portable.get("receipt_sha256") != request["portable_runtime_receipt_sha256"]
        or portable.get("carrier_route_package_sha256") != package["package_sha256"]
        or portable.get("carrier_runtime_sha256") != runtime["runtime_sha256"]
        or portable.get("route_allocations") != allocation_rows
        or portable.get("counted_carrier_parameter_elements") != route_elements
        or runtime.get("owned_fixed_continuous_state_count") != route_elements
        or runtime.get("input_continuous_semantic_code_coefficient_count")
        != allocations[0].parameter_elements
        or runtime.get("output_continuous_semantic_code_coefficient_count")
        != allocations[1].parameter_elements
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Typed C_in/C_out runtime or sessions differ from the requested route package."
        )
    return {
        "route_package_sha256": package["package_sha256"],
        "route_runtime_sha256": runtime["runtime_sha256"],
        "portable_runtime_receipt_sha256": portable["receipt_sha256"],
        "input_route_role": "input",
        "output_route_role": "output",
        "typed_live_sessions_authenticated": True,
    }


def _circuit_tensor_descriptor(value: Any, name: str) -> dict[str, Any]:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            f"Selected parent tensor {name!r} is not a materialized strided tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-circuit-tensor-v1\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(size) for size in tensor.shape],
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": digest.hexdigest(),
    }


def _authenticate_selected_parent_payload(
    *,
    request: Mapping[str, Any],
    batch_index: int,
    trace_input_sha256: str,
    trace_record_sha256: str,
    parent_tensors: Mapping[str, torch.Tensor],
    selected_parent_payload: Lfm2CircuitTraceSelectedBatchPayload,
    trace_selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
) -> dict[str, Any]:
    if (
        not isinstance(
            trace_selective_session,
            Lfm2CircuitTraceSelectiveAuthenticatedSession,
        )
        or trace_selective_session.closed
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Parent provenance requires one open typed selective trace session."
        )
    if (
        not isinstance(selected_parent_payload, Lfm2CircuitTraceSelectedBatchPayload)
        or selected_parent_payload.released
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Parent provenance requires one live typed selected payload."
        )
    try:
        trace_selective_session.verify()
        session_receipt = trace_selective_session.as_dict()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2Exact4xFirstDivergenceError(
            "Selective trace session authentication failed."
        ) from exc
    if (
        session_receipt.get("schema") != SELECTIVE_AUTHENTICATED_SESSION_SCHEMA
        or canonical_sha256(
            {
                key: value
                for key, value in session_receipt.items()
                if key != "session_sha256"
            }
        )
        != session_receipt.get("session_sha256")
        or session_receipt.get("session_sha256")
        != request["trace_selective_session_sha256"]
        or session_receipt.get("cache_key") != request["trace_cache_key"]
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selective trace session differs from the requested cache evidence."
        )
    references = trace_selective_session.batch_references
    if batch_index >= len(references):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent payload leaves the authenticated batch inventory."
        )
    reference = references[batch_index]
    layer_count = int(request["layer_count"])
    focus_layers = tuple(int(value) for value in request["focus_layer_indices"])
    expected_names = parent_tensor_names_for_first_divergence(
        reference,
        layer_count=layer_count,
        focus_layer_indices=focus_layers,
    )
    selected_names = tuple(selected_parent_payload.ordered_tensor_names)
    if (
        reference.index != batch_index
        or reference.input_sha256 != trace_input_sha256
        or reference.record_sha256 != trace_record_sha256
        or selected_parent_payload.batch_index != batch_index
        or selected_parent_payload.input_sha256 != trace_input_sha256
        or selected_parent_payload.session_sha256 != session_receipt["session_sha256"]
        or selected_names != expected_names
        or tuple(sorted(parent_tensors)) != expected_names
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent payload identity or tensor inventory is not canonical."
        )
    inventory = reference.record.get("tensors")
    if not isinstance(inventory, Mapping):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent reference has no authenticated tensor inventory."
        )
    selected_tensors = selected_parent_payload.tensors
    descriptors: list[dict[str, Any]] = []
    compact_descriptors: list[dict[str, Any]] = []
    for name in expected_names:
        selected = selected_tensors.get(name)
        descriptor = _circuit_tensor_descriptor(selected, name)
        if descriptor != inventory.get(name):
            raise Lfm2Exact4xFirstDivergenceError(
                f"Selected parent tensor {name!r} differs from its sealed descriptor."
            )
        retained = selected if name.startswith("input.") else selected[:, -1:, :]
        supplied = parent_tensors.get(name)
        if (
            not isinstance(supplied, torch.Tensor)
            or supplied.device.type != "cpu"
            or not supplied.is_contiguous()
            or supplied.dtype != retained.dtype
            or tuple(supplied.shape) != tuple(retained.shape)
            or not bool(torch.equal(supplied, retained))
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                f"Retained parent tensor {name!r} is not the selected cache payload."
            )
        descriptors.append(descriptor)
        compact_descriptors.append(_circuit_tensor_descriptor(supplied, name))
    logical_bytes = sum(int(row["size_bytes"]) for row in descriptors)
    selection_body = {
        "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
        "session_sha256": session_receipt["session_sha256"],
        "cache_key": session_receipt["cache_key"],
        "batch_index": batch_index,
        "input_sha256": trace_input_sha256,
        "record_sha256": trace_record_sha256,
        "ordered_tensor_names": list(expected_names),
        "ordered_tensor_descriptors": descriptors,
        "logical_bytes": logical_bytes,
    }
    selection_sha256 = canonical_sha256(selection_body)
    if (
        selected_parent_payload.logical_bytes != logical_bytes
        or selected_parent_payload.selection_sha256 != selection_sha256
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent payload receipt does not authenticate its tensors."
        )
    receipt = {
        **selection_body,
        "selection_sha256": selection_sha256,
        "retained_tensor_descriptors": compact_descriptors,
        "retained_tensor_set_sha256": canonical_sha256(
            {
                "ordered_tensor_names": list(expected_names),
                "ordered_tensor_descriptors": compact_descriptors,
            }
        ),
        "typed_selective_session_authenticated": True,
        "typed_selected_payload_authenticated": True,
    }
    return receipt


def audit_lfm2_exact4x_first_divergence_batch(
    *,
    request: Mapping[str, Any],
    batch_index: int,
    trace_input_sha256: str,
    trace_record_sha256: str,
    split: str,
    parent_tensors: Mapping[str, torch.Tensor],
    selected_parent_payload: Lfm2CircuitTraceSelectedBatchPayload,
    trace_selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    reconstructed_trajectory_embeddings: torch.Tensor,
    child_replay: Lfm2ChildEmbeddingReplay,
    coordinate_map: Any,
    route_package: AuthenticatedPackagedLfm2CarrierRoutes,
    portable_runtime_receipt: Lfm2PortableCarrierRuntimeReceipt,
    routing: Lfm2SemanticStepRoutingRuntime,
    input_routing_session: Lfm2SemanticStepLiveTableSession,
    output_routing_session: Lfm2SemanticStepLiveTableSession,
    maximum_donor_rows_per_score_chunk: int = 4096,
) -> dict[str, Any]:
    """Measure one batch without selecting, installing, or training anything."""

    authenticated_request = authenticate_lfm2_exact4x_first_divergence_request(request)
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or batch_index < 0
    ):
        raise Lfm2Exact4xFirstDivergenceError("batch_index must be nonnegative.")
    for value, name in (
        (trace_input_sha256, "trace_input_sha256"),
        (trace_record_sha256, "trace_record_sha256"),
    ):
        _digest(value, name)
    if split not in SPLITS:
        raise Lfm2Exact4xFirstDivergenceError("split must be FIT or heldout.")
    selected_parent_authentication = _authenticate_selected_parent_payload(
        request=authenticated_request,
        batch_index=batch_index,
        trace_input_sha256=trace_input_sha256,
        trace_record_sha256=trace_record_sha256,
        parent_tensors=parent_tensors,
        selected_parent_payload=selected_parent_payload,
        trace_selective_session=trace_selective_session,
    )
    route_context = _authenticate_route_execution_context(
        request=authenticated_request,
        route_package=route_package,
        portable_runtime_receipt=portable_runtime_receipt,
        routing=routing,
        input_routing_session=input_routing_session,
        output_routing_session=output_routing_session,
    )
    (
        trajectory_ids,
        trajectory_mask,
        trajectory_positions,
        prompt_count,
        forced_count,
    ) = _trajectory_replay_geometry(parent_tensors)
    if not isinstance(child_replay, Lfm2ChildEmbeddingReplay):
        raise Lfm2Exact4xFirstDivergenceError(
            "child_replay must be an authenticated embedding replay."
        )
    try:
        replay_record = child_replay.authenticate_for_inputs(
            reconstructed_trajectory_embeddings,
            attention_mask=trajectory_mask,
            position_ids=trajectory_positions,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2Exact4xFirstDivergenceError(
            "Child embedding replay authentication failed."
        ) from exc
    if child_replay.child_fingerprint != authenticated_request["child_state_sha256"]:
        raise Lfm2Exact4xFirstDivergenceError(
            "Child embedding replay belongs to a different complete child state."
        )
    layer_count = int(authenticated_request["layer_count"])
    expected_layers = tuple(range(layer_count))
    focus_layers = tuple(
        int(value) for value in authenticated_request["focus_layer_indices"]
    )
    if child_replay.selected_layer_indices != expected_layers:
        raise Lfm2Exact4xFirstDivergenceError(
            "First-divergence replay must observe every layer of one complete child."
        )
    source_width = int(getattr(coordinate_map, "source_width", 0))
    target_width = int(getattr(coordinate_map, "target_width", 0))
    if (
        target_width <= 0
        or source_width <= target_width
        or routing.hidden_size != target_width
        or getattr(coordinate_map, "map_sha256", None)
        != authenticated_request["basis_map_sha256"]
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "First-divergence coordinate map must be the requested compacting map."
        )
    phases = phase_names(forced_count)
    embeddings = reconstructed_trajectory_embeddings
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 3
        or tuple(embeddings.shape) != (1, prompt_count + forced_count, target_width)
        or not torch.is_floating_point(embeddings)
        or embeddings.is_complex()
        or not bool(torch.isfinite(embeddings).all())
        or int(replay_record["sample_last_tokens"]) != len(phases)
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Reconstructed C_in trajectory or retained child horizon is invalid."
        )
    expected_embeddings = routing.reconstruct_selected_donor_embeddings(
        input_routing_session,
        trajectory_ids.reshape(-1),
    ).unsqueeze(0)
    if (
        tuple(expected_embeddings.shape) != tuple(embeddings.shape)
        or expected_embeddings.dtype != embeddings.dtype
        or expected_embeddings.device != embeddings.device
        or not bool(torch.equal(expected_embeddings, embeddings))
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Reconstructed trajectory is not the authenticated C_in route output."
        )
    state_positions = tuple(range(prompt_count - 1, prompt_count + forced_count))
    phase_rows: list[dict[str, Any]] = []
    for phase_index, (phase, state_position) in enumerate(
        zip(phases, state_positions, strict=True)
    ):
        state_path: list[dict[str, Any]] = []
        branches: list[dict[str, Any]] = []
        c_in_parent = _last_parent_row(
            parent_tensors[f"{phase}.layer.000.residual.pre"],
            source_width,
            f"{phase}.layer.000.residual.pre",
        )
        c_in_child = (
            embeddings[0, state_position].detach().to(device="cpu", dtype=torch.float64)
        )
        c_in_metrics = _metrics(
            c_in_child, _project_parent(coordinate_map, c_in_parent)
        )
        previous_error = c_in_metrics["relative_l2"]
        state_path.append(
            {
                "component_id": "c_in",
                "layer_index": 0,
                "endpoint": "residual.pre",
                "metrics": c_in_metrics,
                "incremental_relative_l2_delta": None,
            }
        )
        for layer_index in focus_layers:
            for suffix in BRANCH_BOUNDARY_SUFFIXES:
                key = f"{phase}.layer.{layer_index:03d}.{suffix}"
                child_key = f"trajectory.layer.{layer_index:03d}.{suffix}"
                parent_row = _last_parent_row(parent_tensors[key], source_width, key)
                child_row = _child_trajectory_row(
                    child_replay,
                    child_key,
                    phase_index,
                    target_width,
                )
                branches.append(
                    {
                        "component_id": f"layer.{layer_index:03d}.{suffix}",
                        "layer_index": layer_index,
                        "endpoint": suffix,
                        "metrics": _metrics(
                            child_row,
                            _project_parent(coordinate_map, parent_row),
                        ),
                    }
                )
        for layer_index in expected_layers:
            for suffix in STATE_BOUNDARY_SUFFIXES:
                key = f"{phase}.layer.{layer_index:03d}.{suffix}"
                child_key = f"trajectory.layer.{layer_index:03d}.{suffix}"
                parent_row = _last_parent_row(parent_tensors[key], source_width, key)
                child_row = _child_trajectory_row(
                    child_replay,
                    child_key,
                    phase_index,
                    target_width,
                )
                metrics = _metrics(
                    child_row,
                    _project_parent(coordinate_map, parent_row),
                )
                delta = metrics["relative_l2"] - previous_error
                previous_error = metrics["relative_l2"]
                state_path.append(
                    {
                        "component_id": f"layer.{layer_index:03d}.{suffix}",
                        "layer_index": layer_index,
                        "endpoint": suffix,
                        "metrics": metrics,
                        "incremental_relative_l2_delta": delta,
                    }
                )
        largest_jump = max(
            state_path[1:],
            key=lambda row: float(row["incremental_relative_l2_delta"]),
        )
        parent_logits_value = parent_tensors.get(f"{phase}.logits")
        if (
            not isinstance(parent_logits_value, torch.Tensor)
            or parent_logits_value.ndim != 3
            or int(parent_logits_value.shape[0]) != 1
            or int(parent_logits_value.shape[1]) <= 0
            or int(parent_logits_value.shape[2]) <= 1
            or not bool(torch.isfinite(parent_logits_value).all())
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                f"Parent logits for {phase!r} are invalid."
            )
        parent_logits = (
            parent_logits_value[0, -1].detach().to(device="cpu", dtype=torch.float64)
        )
        parent_winner = int(parent_logits.argmax().item())
        child_hidden = _child_trajectory_row(
            child_replay,
            "trajectory.output.lm_head_input",
            phase_index,
            target_width,
        )
        target_scores = routing.score_target_tokens(
            output_routing_session,
            child_hidden,
        )
        donor_scores = routing.compute_all_donor_logits_from_target_scores(
            output_routing_session,
            target_scores,
            maximum_donor_rows_per_chunk=maximum_donor_rows_per_score_chunk,
        )
        if int(donor_scores.numel()) != int(parent_logits.numel()):
            raise Lfm2Exact4xFirstDivergenceError(
                "C_out donor scores differ from the cached parent vocabulary."
            )
        phase_rows.append(
            {
                "phase": phase,
                "phase_index": phase_index,
                "consumed_forced_token_count": phase_index,
                "state_path": state_path,
                "branch_components": branches,
                "complete_state_layer_indices": list(expected_layers),
                "focused_branch_layer_indices": list(focus_layers),
                "focus_layers_limit_branch_diagnostics_only": True,
                "descriptive_largest_state_error_jump": {
                    "component_id": largest_jump["component_id"],
                    "incremental_relative_l2_delta": largest_jump[
                        "incremental_relative_l2_delta"
                    ],
                    "not_an_acceptance_threshold": True,
                },
                "c_out": {
                    "cached_parent": parent_winner_rank_margin(
                        parent_logits, parent_winner
                    ),
                    "folded_child": parent_winner_rank_margin(
                        donor_scores, parent_winner
                    ),
                    "exact_parent_winner_mismatch_is_discrete_diagnostic": True,
                },
            }
        )
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_BATCH_SCHEMA,
        "request_sha256": authenticated_request["request_sha256"],
        "batch_index": batch_index,
        "trace_input_sha256": trace_input_sha256,
        "trace_record_sha256": trace_record_sha256,
        "selected_payload_sha256": selected_parent_authentication["selection_sha256"],
        "selected_parent_authentication": selected_parent_authentication,
        "child_embedding_replay_sha256": child_replay.replay_sha256,
        "route_execution_context": route_context,
        "replay_input_authentication": {
            "attention_mask_sha256": replay_record["inputs"]["attention_mask_sha256"],
            "position_ids_sha256": replay_record["inputs"]["position_ids_sha256"],
            "position_ids_present": (
                replay_record["inputs"]["position_ids_sha256"] is not None
            ),
            "attention_mask_authenticated": True,
            "position_ids_authenticated": True,
        },
        "split": split,
        "complete_child_observed_layer_indices": list(expected_layers),
        "focused_branch_layer_indices": list(focus_layers),
        "focus_layers_limit_branch_diagnostics_only": True,
        "noninitial_residual_pre_parent_reads": 0,
        "phase_count": len(phases),
        "phases": phase_rows,
        "numeric_decision": None,
        "numeric_metrics_are_diagnostic_only": True,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "safety_observations": [],
        "safety_observations_are_separate_not_failures": True,
        "generation_cap_events": [],
        "generation_cap_saturation_is_censored_not_failure": True,
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": 1,
            "child_forward_call_scope": "batch-producing-run",
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_mutations": 0,
            "free_generation_calls": 0,
        },
    }
    return {**body, "batch_audit_sha256": canonical_sha256(body)}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2Exact4xFirstDivergenceError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2Exact4xFirstDivergenceError(f"{name} must be finite.")
    return result


def _authenticate_metric_row(value: Any, name: str) -> dict[str, Any]:
    expected = {
        "candidate_norm",
        "cosine",
        "cosine_distance",
        "l2",
        "projected_parent_norm",
        "relative_l2",
        "rmse",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Lfm2Exact4xFirstDivergenceError(
            f"{name} has a noncanonical metric inventory."
        )
    result = dict(value)
    for key in expected:
        _finite_number(result[key], f"{name}.{key}")
    if (
        result["l2"] < 0
        or result["relative_l2"] < 0
        or result["rmse"] < 0
        or result["candidate_norm"] < 0
        or result["projected_parent_norm"] < 0
        or result["cosine"] < -1
        or result["cosine"] > 1
        or result["cosine_distance"] != 1.0 - result["cosine"]
    ):
        raise Lfm2Exact4xFirstDivergenceError(f"{name} metrics are inconsistent.")
    return result


def _authenticate_rank_margin(value: Any, name: str) -> dict[str, Any]:
    expected = {
        "best_competing_score",
        "best_competing_token_id",
        "parent_winner_margin",
        "parent_winner_rank",
        "parent_winner_score",
        "parent_winner_token_id",
        "top1_matches_parent",
        "top1_token_id",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Lfm2Exact4xFirstDivergenceError(
            f"{name} has a noncanonical rank-margin inventory."
        )
    result = dict(value)
    integer_fields = (
        "best_competing_token_id",
        "parent_winner_rank",
        "parent_winner_token_id",
        "top1_token_id",
    )
    if any(
        isinstance(result[field], bool)
        or not isinstance(result[field], int)
        or result[field] < (1 if field == "parent_winner_rank" else 0)
        for field in integer_fields
    ):
        raise Lfm2Exact4xFirstDivergenceError(f"{name} token/rank fields are invalid.")
    for field in (
        "best_competing_score",
        "parent_winner_margin",
        "parent_winner_score",
    ):
        _finite_number(result[field], f"{name}.{field}")
    if (
        not isinstance(result["top1_matches_parent"], bool)
        or result["top1_matches_parent"]
        != (result["top1_token_id"] == result["parent_winner_token_id"])
        or not math.isclose(
            result["parent_winner_margin"],
            result["parent_winner_score"] - result["best_competing_score"],
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
    ):
        raise Lfm2Exact4xFirstDivergenceError(f"{name} rank margin is inconsistent.")
    return result


def _authenticate_selected_parent_receipt(
    value: Any,
    *,
    result: Mapping[str, Any],
    request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required = {
        "batch_index",
        "cache_key",
        "input_sha256",
        "logical_bytes",
        "ordered_tensor_descriptors",
        "ordered_tensor_names",
        "record_sha256",
        "retained_tensor_descriptors",
        "retained_tensor_set_sha256",
        "schema",
        "selection_sha256",
        "session_sha256",
        "typed_selected_payload_authenticated",
        "typed_selective_session_authenticated",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent authentication receipt is noncanonical."
        )
    receipt = dict(value)
    names = receipt.get("ordered_tensor_names")
    full = receipt.get("ordered_tensor_descriptors")
    retained = receipt.get("retained_tensor_descriptors")
    if (
        receipt.get("schema") != SELECTIVE_BATCH_PAYLOAD_SCHEMA
        or receipt.get("typed_selected_payload_authenticated") is not True
        or receipt.get("typed_selective_session_authenticated") is not True
        or not isinstance(names, list)
        or not names
        or names != sorted(set(names))
        or not isinstance(full, list)
        or not isinstance(retained, list)
        or len(names) != len(full)
        or len(names) != len(retained)
        or receipt.get("batch_index") != result.get("batch_index")
        or receipt.get("input_sha256") != result.get("trace_input_sha256")
        or receipt.get("record_sha256") != result.get("trace_record_sha256")
        or receipt.get("selection_sha256") != result.get("selected_payload_sha256")
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent authentication receipt does not bind its batch."
        )
    descriptor_keys = {"dtype", "numel", "sha256", "shape", "size_bytes"}
    for label, rows in (("full", full), ("retained", retained)):
        for index, row in enumerate(rows):
            if (
                not isinstance(row, Mapping)
                or set(row) != descriptor_keys
                or not isinstance(row.get("dtype"), str)
                or not row.get("dtype")
                or not isinstance(row.get("shape"), list)
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in row.get("shape", [])
                )
                or isinstance(row.get("numel"), bool)
                or not isinstance(row.get("numel"), int)
                or row.get("numel") <= 0
                or math.prod(row["shape"]) != row["numel"]
                or isinstance(row.get("size_bytes"), bool)
                or not isinstance(row.get("size_bytes"), int)
                or row.get("size_bytes") <= 0
            ):
                raise Lfm2Exact4xFirstDivergenceError(
                    f"Selected parent {label} descriptor {index} is invalid."
                )
            _digest(row.get("sha256"), f"selected parent {label} descriptor")
    selection_body = {
        key: receipt[key]
        for key in (
            "schema",
            "session_sha256",
            "cache_key",
            "batch_index",
            "input_sha256",
            "record_sha256",
            "ordered_tensor_names",
            "ordered_tensor_descriptors",
            "logical_bytes",
        )
    }
    retained_body = {
        "ordered_tensor_names": names,
        "ordered_tensor_descriptors": retained,
    }
    if (
        _positive_int(receipt.get("logical_bytes"), "selected logical_bytes")
        != sum(int(row["size_bytes"]) for row in full)
        or canonical_sha256(selection_body) != receipt["selection_sha256"]
        or canonical_sha256(retained_body) != receipt["retained_tensor_set_sha256"]
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent authentication receipt hashes do not verify."
        )
    _digest(receipt["session_sha256"], "selected parent session_sha256")
    _digest(receipt["cache_key"], "selected parent cache_key")
    _digest(receipt["retained_tensor_set_sha256"], "retained tensor set SHA-256")
    if request is not None and (
        receipt["session_sha256"] != request["trace_selective_session_sha256"]
        or receipt["cache_key"] != request["trace_cache_key"]
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Selected parent receipt differs from the diagnostic request."
        )
    return receipt


def _authenticate_batch_invariants(
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None,
) -> None:
    expected_keys = {
        "batch_audit_sha256",
        "batch_index",
        "child_embedding_replay_sha256",
        "complete_child_observed_layer_indices",
        "focus_layers_limit_branch_diagnostics_only",
        "focused_branch_layer_indices",
        "generation_cap_events",
        "generation_cap_saturation_is_censored_not_failure",
        "manual_whole_response_intelligence_review_is_authoritative",
        "noninitial_residual_pre_parent_reads",
        "numeric_decision",
        "numeric_metrics_are_diagnostic_only",
        "operation_audit",
        "phase_count",
        "phases",
        "replay_input_authentication",
        "request_sha256",
        "route_execution_context",
        "safety_observations",
        "safety_observations_are_separate_not_failures",
        "schema",
        "selected_parent_authentication",
        "selected_payload_sha256",
        "split",
        "trace_input_sha256",
        "trace_record_sha256",
    }
    if set(result) != expected_keys:
        raise Lfm2Exact4xFirstDivergenceError(
            "Batch diagnostic has a noncanonical field inventory."
        )
    if (
        isinstance(result.get("batch_index"), bool)
        or not isinstance(result.get("batch_index"), int)
        or result["batch_index"] < 0
        or result.get("split") not in SPLITS
    ):
        raise Lfm2Exact4xFirstDivergenceError("Batch identity fields are invalid.")
    for field in (
        "batch_audit_sha256",
        "child_embedding_replay_sha256",
        "request_sha256",
        "selected_payload_sha256",
        "trace_input_sha256",
        "trace_record_sha256",
    ):
        _digest(result.get(field), f"batch {field}")
    _authenticate_selected_parent_receipt(
        result.get("selected_parent_authentication"),
        result=result,
        request=request,
    )
    if request is not None:
        expected_layers = list(range(int(request["layer_count"])))
        focus_layers = list(request["focus_layer_indices"])
        route = result.get("route_execution_context")
        if (
            result.get("request_sha256") != request["request_sha256"]
            or result.get("complete_child_observed_layer_indices") != expected_layers
            or result.get("focused_branch_layer_indices") != focus_layers
            or not isinstance(route, Mapping)
            or route
            != {
                "input_route_role": "input",
                "output_route_role": "output",
                "portable_runtime_receipt_sha256": request[
                    "portable_runtime_receipt_sha256"
                ],
                "route_package_sha256": request["route_package_sha256"],
                "route_runtime_sha256": request["route_runtime_sha256"],
                "typed_live_sessions_authenticated": True,
            }
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                "Batch execution topology differs from its request."
            )
    else:
        expected_layers = result.get("complete_child_observed_layer_indices")
        focus_layers = result.get("focused_branch_layer_indices")
    if (
        not isinstance(expected_layers, list)
        or expected_layers != list(range(len(expected_layers)))
        or not expected_layers
        or not isinstance(focus_layers, list)
        or not focus_layers
        or focus_layers != sorted(set(focus_layers))
        or any(value not in expected_layers for value in focus_layers)
        or result.get("focus_layers_limit_branch_diagnostics_only") is not True
        or result.get("noninitial_residual_pre_parent_reads") != 0
        or result.get("numeric_decision") is not None
        or result.get("numeric_metrics_are_diagnostic_only") is not True
        or result.get("manual_whole_response_intelligence_review_is_authoritative")
        is not True
        or result.get("safety_observations") != []
        or result.get("safety_observations_are_separate_not_failures") is not True
        or result.get("generation_cap_events") != []
        or result.get("generation_cap_saturation_is_censored_not_failure") is not True
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Batch policy or complete-child invariants do not verify."
        )
    operation = result.get("operation_audit")
    if operation != {
        "parent_forward_calls": 0,
        "child_forward_calls": 1,
        "child_forward_call_scope": "batch-producing-run",
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "model_mutations": 0,
        "free_generation_calls": 0,
    }:
        raise Lfm2Exact4xFirstDivergenceError(
            "Batch operation audit does not prove one inference-only child forward."
        )
    replay = result.get("replay_input_authentication")
    if (
        not isinstance(replay, Mapping)
        or set(replay)
        != {
            "attention_mask_authenticated",
            "attention_mask_sha256",
            "position_ids_authenticated",
            "position_ids_present",
            "position_ids_sha256",
        }
        or replay.get("attention_mask_authenticated") is not True
        or replay.get("position_ids_authenticated") is not True
        or not isinstance(replay.get("position_ids_present"), bool)
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Batch replay input authentication is noncanonical."
        )
    _digest(replay.get("attention_mask_sha256"), "batch attention mask SHA-256")
    if replay["position_ids_present"]:
        _digest(replay.get("position_ids_sha256"), "batch position IDs SHA-256")
    elif replay.get("position_ids_sha256") is not None:
        raise Lfm2Exact4xFirstDivergenceError(
            "Absent replay position IDs must carry a canonical null digest."
        )
    phases = result.get("phases")
    if (
        isinstance(result.get("phase_count"), bool)
        or not isinstance(result.get("phase_count"), int)
        or result["phase_count"] <= 0
        or not isinstance(phases, list)
        or len(phases) != result["phase_count"]
    ):
        raise Lfm2Exact4xFirstDivergenceError("Batch phase inventory is invalid.")
    expected_phase_names = phase_names(len(phases) - 1)
    state_specs = [("c_in", 0, "residual.pre")]
    state_specs.extend(
        (f"layer.{layer:03d}.{suffix}", layer, suffix)
        for layer in expected_layers
        for suffix in STATE_BOUNDARY_SUFFIXES
    )
    branch_specs = [
        (f"layer.{layer:03d}.{suffix}", layer, suffix)
        for layer in focus_layers
        for suffix in BRANCH_BOUNDARY_SUFFIXES
    ]
    for phase_index, phase in enumerate(phases):
        if (
            not isinstance(phase, Mapping)
            or set(phase)
            != {
                "branch_components",
                "c_out",
                "complete_state_layer_indices",
                "consumed_forced_token_count",
                "descriptive_largest_state_error_jump",
                "focus_layers_limit_branch_diagnostics_only",
                "focused_branch_layer_indices",
                "phase",
                "phase_index",
                "state_path",
            }
            or phase.get("phase") != expected_phase_names[phase_index]
            or phase.get("phase_index") != phase_index
            or phase.get("consumed_forced_token_count") != phase_index
            or phase.get("complete_state_layer_indices") != expected_layers
            or phase.get("focused_branch_layer_indices") != focus_layers
            or phase.get("focus_layers_limit_branch_diagnostics_only") is not True
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                f"Batch phase {phase_index} identity is invalid."
            )
        state_rows = phase.get("state_path")
        branch_rows = phase.get("branch_components")
        if (
            not isinstance(state_rows, list)
            or len(state_rows) != len(state_specs)
            or not isinstance(branch_rows, list)
            or len(branch_rows) != len(branch_specs)
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                f"Batch phase {phase_index} component inventory is invalid."
            )
        previous_relative = None
        for row_index, (row, spec) in enumerate(
            zip(state_rows, state_specs, strict=True)
        ):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "component_id",
                    "endpoint",
                    "incremental_relative_l2_delta",
                    "layer_index",
                    "metrics",
                }
                or (
                    row.get("component_id"),
                    row.get("layer_index"),
                    row.get("endpoint"),
                )
                != spec
            ):
                raise Lfm2Exact4xFirstDivergenceError(
                    f"Batch phase {phase_index} state row {row_index} is invalid."
                )
            metrics = _authenticate_metric_row(
                row.get("metrics"), f"phase {phase_index} state {row_index}"
            )
            incremental = row.get("incremental_relative_l2_delta")
            if row_index == 0:
                if incremental is not None:
                    raise Lfm2Exact4xFirstDivergenceError(
                        "C_in cannot claim an incremental error delta."
                    )
            elif (
                _finite_number(incremental, "incremental_relative_l2_delta")
                != metrics["relative_l2"] - previous_relative
            ):
                raise Lfm2Exact4xFirstDivergenceError(
                    "State-path incremental error delta is inconsistent."
                )
            previous_relative = metrics["relative_l2"]
        for row_index, (row, spec) in enumerate(
            zip(branch_rows, branch_specs, strict=True)
        ):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"component_id", "endpoint", "layer_index", "metrics"}
                or (
                    row.get("component_id"),
                    row.get("layer_index"),
                    row.get("endpoint"),
                )
                != spec
            ):
                raise Lfm2Exact4xFirstDivergenceError(
                    f"Batch phase {phase_index} branch row {row_index} is invalid."
                )
            _authenticate_metric_row(
                row.get("metrics"), f"phase {phase_index} branch {row_index}"
            )
        largest = max(
            state_rows[1:],
            key=lambda row: float(row["incremental_relative_l2_delta"]),
        )
        if phase.get("descriptive_largest_state_error_jump") != {
            "component_id": largest["component_id"],
            "incremental_relative_l2_delta": largest["incremental_relative_l2_delta"],
            "not_an_acceptance_threshold": True,
        }:
            raise Lfm2Exact4xFirstDivergenceError(
                "Descriptive largest state error jump is inconsistent."
            )
        c_out = phase.get("c_out")
        if (
            not isinstance(c_out, Mapping)
            or set(c_out)
            != {
                "cached_parent",
                "exact_parent_winner_mismatch_is_discrete_diagnostic",
                "folded_child",
            }
            or c_out.get("exact_parent_winner_mismatch_is_discrete_diagnostic")
            is not True
        ):
            raise Lfm2Exact4xFirstDivergenceError("C_out evidence is noncanonical.")
        parent = _authenticate_rank_margin(c_out.get("cached_parent"), "cached C_out")
        child = _authenticate_rank_margin(c_out.get("folded_child"), "folded C_out")
        if (
            parent["parent_winner_token_id"] != child["parent_winner_token_id"]
            or parent["top1_matches_parent"] is not True
            or parent["parent_winner_rank"] != 1
        ):
            raise Lfm2Exact4xFirstDivergenceError(
                "Cached and folded C_out evidence use different parent winners."
            )


def authenticate_lfm2_exact4x_first_divergence_batch(
    value: Mapping[str, Any],
    *,
    expected_request_sha256: str | None = None,
    expected_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = _verify_seal(
        value,
        "batch_audit_sha256",
        LFM2_EXACT4X_FIRST_DIVERGENCE_BATCH_SCHEMA,
    )
    request = (
        None
        if expected_request is None
        else authenticate_lfm2_exact4x_first_divergence_request(expected_request)
    )
    requested_sha256 = (
        request["request_sha256"]
        if request is not None
        else (
            None
            if expected_request_sha256 is None
            else _digest(expected_request_sha256, "expected_request_sha256")
        )
    )
    if (
        requested_sha256 is not None
        and result.get("request_sha256") != requested_sha256
    ):
        raise Lfm2Exact4xFirstDivergenceError(
            "Batch diagnostic belongs to a different request."
        )
    _authenticate_batch_invariants(result, request=request)
    return result


def _aggregate_component_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    accumulators: dict[str, dict[str, Any]] = {}
    for row in rows:
        component_id = str(row["component_id"])
        metrics = row["metrics"]
        target = accumulators.setdefault(
            component_id,
            {
                "component_id": component_id,
                "layer_index": row["layer_index"],
                "endpoint": row["endpoint"],
                "row_count": 0,
                "cosine_sum": 0.0,
                "relative_l2_sum": 0.0,
                "rmse_sum": 0.0,
                "maximum_relative_l2": 0.0,
                "incremental_sum": 0.0,
                "incremental_count": 0,
            },
        )
        target["row_count"] += 1
        target["cosine_sum"] += float(metrics["cosine"])
        target["relative_l2_sum"] += float(metrics["relative_l2"])
        target["rmse_sum"] += float(metrics["rmse"])
        target["maximum_relative_l2"] = max(
            target["maximum_relative_l2"], float(metrics["relative_l2"])
        )
        incremental = row.get("incremental_relative_l2_delta")
        if incremental is not None:
            target["incremental_sum"] += float(incremental)
            target["incremental_count"] += 1
    result = []
    for target in accumulators.values():
        count = int(target.pop("row_count"))
        incremental_count = int(target.pop("incremental_count"))
        cosine_sum = float(target.pop("cosine_sum"))
        relative_sum = float(target.pop("relative_l2_sum"))
        rmse_sum = float(target.pop("rmse_sum"))
        incremental_sum = float(target.pop("incremental_sum"))
        result.append(
            {
                **target,
                "row_count": count,
                "mean_cosine": cosine_sum / count,
                "mean_relative_l2": relative_sum / count,
                "mean_rmse": rmse_sum / count,
                "mean_incremental_relative_l2_delta": (
                    None
                    if incremental_count == 0
                    else incremental_sum / incremental_count
                ),
            }
        )
    return result


def aggregate_lfm2_exact4x_first_divergence(
    request: Mapping[str, Any],
    batch_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authenticated_request = authenticate_lfm2_exact4x_first_divergence_request(request)
    if not batch_results:
        raise Lfm2Exact4xFirstDivergenceError(
            "At least one completed batch diagnostic is required."
        )
    batches = [
        authenticate_lfm2_exact4x_first_divergence_batch(
            value,
            expected_request=authenticated_request,
        )
        for value in batch_results
    ]
    indices = [int(value["batch_index"]) for value in batches]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise Lfm2Exact4xFirstDivergenceError(
            "Batch diagnostics must be unique and canonically ordered."
        )
    split_summaries: dict[str, Any] = {}
    first_c_out_mismatch: dict[str, Any] | None = None
    for split in SPLITS:
        selected = [value for value in batches if value["split"] == split]
        if not selected:
            raise Lfm2Exact4xFirstDivergenceError(
                "Final ledger requires both FIT and heldout diagnostics."
            )
        state_rows = []
        branch_rows = []
        mismatch_count = 0
        c_out_count = 0
        for batch in selected:
            for phase in batch["phases"]:
                state_rows.extend(phase["state_path"])
                branch_rows.extend(phase["branch_components"])
                c_out_count += 1
                if not bool(phase["c_out"]["folded_child"]["top1_matches_parent"]):
                    mismatch_count += 1
                    candidate = {
                        "batch_index": batch["batch_index"],
                        "split": split,
                        "phase": phase["phase"],
                        "phase_index": phase["phase_index"],
                        "parent_winner_token_id": phase["c_out"]["folded_child"][
                            "parent_winner_token_id"
                        ],
                        "child_top1_token_id": phase["c_out"]["folded_child"][
                            "top1_token_id"
                        ],
                    }
                    if first_c_out_mismatch is None or (
                        candidate["batch_index"],
                        candidate["phase_index"],
                    ) < (
                        first_c_out_mismatch["batch_index"],
                        first_c_out_mismatch["phase_index"],
                    ):
                        first_c_out_mismatch = candidate
        state_components = _aggregate_component_rows(state_rows)
        branch_components = _aggregate_component_rows(branch_rows)
        candidates = [
            value
            for value in state_components
            if value["mean_incremental_relative_l2_delta"] is not None
        ]
        descriptive = max(
            candidates,
            key=lambda value: float(value["mean_incremental_relative_l2_delta"]),
        )
        split_summaries[split] = {
            "batch_count": len(selected),
            "state_components": state_components,
            "branch_components": branch_components,
            "descriptive_largest_mean_state_error_jump": {
                "component_id": descriptive["component_id"],
                "mean_incremental_relative_l2_delta": descriptive[
                    "mean_incremental_relative_l2_delta"
                ],
                "not_an_acceptance_threshold": True,
            },
            "c_out_state_count": c_out_count,
            "c_out_parent_winner_mismatch_count": mismatch_count,
            "c_out_parent_winner_match_fraction": 1.0 - mismatch_count / c_out_count,
            "numeric_decision": None,
        }
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA,
        "status": "complete-observation",
        "request": authenticated_request,
        "ordered_batch_audit_sha256s": [
            value["batch_audit_sha256"] for value in batches
        ],
        "batch_count": len(batches),
        "split_summaries": split_summaries,
        "first_discrete_c_out_parent_winner_mismatch": first_c_out_mismatch,
        "component_first_divergence_is_descriptive_not_thresholded": True,
        "numeric_metrics_are_diagnostic_only": True,
        "numeric_acceptance_decision": None,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "safety_observations_are_separate_not_failures": True,
        "generation_cap_saturation_is_censored_not_failure": True,
        "free_generation_performed": False,
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": sum(
                int(value["operation_audit"]["child_forward_calls"])
                for value in batches
            ),
            "child_forward_call_scope": (
                "persisted-batch-evidence-across-producing-runs"
            ),
            "current_run_child_forward_calls": None,
            "current_vs_historical_forward_counts_are_runner_scoped": True,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_mutations": 0,
        },
    }
    return {**body, "ledger_sha256": canonical_sha256(body)}


__all__ = [
    "BRANCH_BOUNDARY_SUFFIXES",
    "LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA",
    "LFM2_EXACT4X_FIRST_DIVERGENCE_BATCH_SCHEMA",
    "LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA",
    "LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA",
    "LFM2_EXACT4X_FIRST_DIVERGENCE_REQUEST_SCHEMA",
    "Lfm2Exact4xFirstDivergenceError",
    "RESIDUAL_ENDPOINTS",
    "SPLITS",
    "STATE_BOUNDARY_SUFFIXES",
    "aggregate_lfm2_exact4x_first_divergence",
    "audit_lfm2_exact4x_first_divergence_batch",
    "authenticate_lfm2_exact4x_first_divergence_batch",
    "authenticate_lfm2_exact4x_first_divergence_request",
    "authoritative_split_by_batch",
    "build_lfm2_exact4x_first_divergence_request",
    "parent_selection_logical_bytes",
    "parent_tensor_names_for_first_divergence",
    "phase_names",
    "verify_complete_exact_budget_child",
]
