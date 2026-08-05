#!/usr/bin/env python3
"""Thin, fail-closed orchestration for the exact-4x LFM2 moving-gauge fold.

This runner does not introduce another folding algorithm.  It fixes the order,
lineage, split, replay, and publication contracts around the authenticated
boundary/feature imports and the existing closed-form moving-gauge solvers.
Live execution is supplied through an explicit handler for every phase; a
missing handler is rejected before any phase runs.

The command-line entry point is deliberately preflight-only.  It authenticates
the request shape, local path policy, and required runtime API surface, then
writes the immutable phase plan.  It cannot silently fall back to a partial
fold when live model wiring is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_counted_child_state import (
    CountedChildStatePlan,
    build_counted_child_state_plan,
    counted_child_state_plan_from_dict,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    fingerprint_lfm2_moving_gauge_tensor,
)
from neural_forge.lfm2_carrier_child_package import (
    CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS,
    CURRENT_LFM2_CARRIER_HIDDEN_SIZE,
    CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
    CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS,
    CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
    AuthenticatedLfm2CarrierChildPackage,
)


RUN_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-run-v1"
REQUEST_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-request-v1"
PHASE_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-phase-v1"
PHASE_PLAN_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-phase-plan-v1"
PHASE_OUTCOME_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-outcome-v1"
CANDIDATE_PACK_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-candidate-pack-v1"
RUNTIME_SURFACE_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-runtime-surface-v1"
LOCAL_PREFLIGHT_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-local-preflight-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-failure-v1"

EXPECTED_BATCH_COUNT = 256
EXPECTED_ALIGNED_ROW_COUNT = 67_583
FIT_BATCH_COUNT = 192
HELDOUT_BATCH_COUNT = 64
SOURCE_WIDTH = 1_024
TARGET_WIDTH = CURRENT_LFM2_CARRIER_HIDDEN_SIZE
LAYER_COUNT = 14
OPERATOR_LAYER_INDEX = 0
TERMINAL_LAYER_INDEX = LAYER_COUNT - 1
BASE_ENCODER_TENSOR_PATH = "basis_result.basis"
FOCUSED_FEATURE_FIELDS = ("short_conv.context", "mlp.post_swiglu")
REPLACEMENT_TENSOR_NAMES = (
    "model.layers.0.conv.out_proj.weight",
    "model.layers.0.feed_forward.w2.weight",
)
CHART_MODES = ("E0", "E0R", "E0A")
TARGET_ARMS = (
    "joint",
    "sequential_no_recapture",
    "sequential_conditioned",
)
PUBLISHABLE_TARGET_ARMS = ("joint", "sequential_conditioned")

PHASE_ORDER = (
    "authenticate_lineage",
    "import_boundary_rows",
    "import_feature_rows",
    "freeze_fit_charts",
    "build_baseline_target_arms",
    "checkpoint_conditioned_child_recapture",
    "freeze_conditioned_targets_and_solutions",
    "compare_frozen_candidates_on_heldout",
    "stage_atomic_candidate_packs",
)

REQUIRED_LOCAL_PATH_KEYS = frozenset(
    {
        "published_child",
        "v3_evidence",
        "parent_trace_cache",
        "child_trace_cache",
        "basis_phase_cache",
        "phase_cache",
        "checkpoint_root",
        "output_root",
    }
)
IMMUTABLE_INPUT_PATH_KEYS = frozenset(
    {
        "published_child",
        "v3_evidence",
        "parent_trace_cache",
        "child_trace_cache",
        "basis_phase_cache",
    }
)
MUTABLE_OUTPUT_PATH_KEYS = frozenset({"phase_cache", "checkpoint_root", "output_root"})
REQUIRED_LINEAGE_KEYS = frozenset(
    {
        "r12_child_package",
        "r12_child_identity",
        "r12_state_dict",
        "v3_request",
        "v3_ledger",
        "parent_trace_cache_completion",
        "child_trace_cache_completion",
        "basis_phase_key",
        "basis_phase_completion",
        "basis_source_anchor",
        "tokenizer_bridge",
        "counted_child_state_plan",
    }
)

FORBIDDEN_OPERATION_AUDIT = MappingProxyType(
    {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_model_objects_constructed": 0,
        "parent_model_loads": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "parameter_sharing_used": False,
        "parent_runtime_sidecar_used": False,
        "network_access": False,
    }
)

RUNTIME_API_BINDINGS = MappingProxyType(
    {
        "carrier_package_authentication": (
            "neural_forge.lfm2_carrier_child_package",
            "load_authenticated_lfm2_carrier_child_package",
        ),
        "v3_request_authentication": (
            "neural_forge.hf_lfm2_exact4x_first_divergence",
            "authenticate_lfm2_exact4x_first_divergence_request",
        ),
        "v3_ledger_authentication": (
            "neural_forge.hf_lfm2_exact4x_causal_diagnostic",
            "authenticate_v3_first_divergence_ledger",
        ),
        "trace_cache_authentication": (
            "neural_forge.hf_lfm2_circuit_trace_cache",
            "open_lfm2_circuit_trace_cache",
        ),
        "basis_phase_authentication": (
            "neural_forge.hf_lfm2_balanced_basis_phase_cache",
            "load_existing_lfm2_circuit_balanced_basis_phase",
        ),
        "boundary_phase_import": (
            "neural_forge.hf_lfm2_moving_gauge_boundary_phase_cache",
            "load_or_extract_lfm2_moving_gauge_boundary_phase",
        ),
        "feature_phase_import": (
            "neural_forge.hf_lfm2_moving_gauge_feature_phase_cache",
            "load_or_extract_lfm2_moving_gauge_feature_phase",
        ),
        "orthogonal_chart_fit": (
            "neural_forge.hf_lfm2_moving_gauge_residual_reclosure",
            "fit_lfm2_cycle_coupled_moving_gauge_boundary",
        ),
        "generalized_chart_fit": (
            "neural_forge.hf_lfm2_moving_gauge_residual_reclosure",
            "fit_lfm2_generalized_moving_gauge_chart_diagnostic",
        ),
        "base_endpoint_bind": (
            "neural_forge.hf_lfm2_moving_gauge_residual_reclosure",
            "bind_lfm2_base_e0_causal_endpoint_chart",
        ),
        "orthogonal_endpoint_bind": (
            "neural_forge.hf_lfm2_moving_gauge_residual_reclosure",
            "bind_lfm2_orthogonal_causal_endpoint_chart",
        ),
        "generalized_endpoint_bind": (
            "neural_forge.hf_lfm2_moving_gauge_residual_reclosure",
            "bind_lfm2_generalized_e0a_causal_endpoint_chart",
        ),
        "joint_target_build": (
            "neural_forge.hf_lfm2_moving_gauge_causal_targets",
            "build_lfm2_joint_writer_target",
        ),
        "sequential_target_build": (
            "neural_forge.hf_lfm2_moving_gauge_causal_targets",
            "build_lfm2_sequential_writer_target",
        ),
        "writer_closed_form_solve": (
            "neural_forge.hf_lfm2_moving_gauge_residual_reclosure",
            "solve_lfm2_moving_gauge_anchored_writer_replacement",
        ),
        "child_only_replay": (
            "neural_forge.hf_lfm2_circuit_trace_cache",
            "capture_lfm2_child_replay",
        ),
        "child_state_fingerprint": (
            "neural_forge.hf_lfm2_circuit_trace_cache",
            "fingerprint_lfm2_parent",
        ),
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")


class Lfm2Exact4xMovingGaugeV5Error(RuntimeError):
    """The v5 orchestration request or phase evidence is invalid."""


class Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError(Lfm2Exact4xMovingGaugeV5Error):
    """A required existing runtime API is missing; partial execution is forbidden."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"{name} must be a nonempty bounded stable identifier."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2Exact4xMovingGaugeV5Error(f"{name} must be a positive integer.")
    return value


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xMovingGaugeV5Error(f"{name} must be a mapping.")
    try:
        restored = json.loads(canonical_json_bytes(dict(value)).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"{name} must contain finite canonical JSON."
        ) from exc
    if not isinstance(restored, dict):
        raise Lfm2Exact4xMovingGaugeV5Error(f"{name} must be a JSON object.")
    return restored


def _sealed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    row = _canonical_mapping(body, field_name)
    return {**row, field_name: canonical_sha256(row)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_local_path(value: Any, name: str) -> str:
    if not isinstance(value, (str, os.PathLike)) or "://" in str(value):
        raise Lfm2Exact4xMovingGaugeV5Error(f"{name} must be an ordinary local path.")
    path = Path(value).expanduser().absolute()
    if not path.is_absolute():
        raise Lfm2Exact4xMovingGaugeV5Error(f"{name} must be absolute.")
    return str(path)


def normalize_v5_local_paths(values: Mapping[str, Any]) -> Mapping[str, str]:
    """Normalize the exact local path inventory without creating anything."""

    supplied = dict(values)
    if set(supplied) != REQUIRED_LOCAL_PATH_KEYS:
        missing = sorted(REQUIRED_LOCAL_PATH_KEYS.difference(supplied))
        extra = sorted(set(supplied).difference(REQUIRED_LOCAL_PATH_KEYS))
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Local path inventory changed; missing={missing}, extra={extra}."
        )
    return MappingProxyType(
        {name: _absolute_local_path(supplied[name], name) for name in sorted(supplied)}
    )


def _validated_exact4x_plan(value: Any) -> CountedChildStatePlan:
    if not isinstance(value, CountedChildStatePlan):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "counted_child_state_plan must be CountedChildStatePlan."
        )
    try:
        restored = counted_child_state_plan_from_dict(value.as_dict())
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Counted child-state plan failed canonical reconstruction."
        ) from exc
    expected_allocation_ids = (
        "semantic-carrier-route.c_in",
        "semantic-carrier-route.c_out",
    )
    if (
        restored.parent_unique_parameters != CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS
        or restored.target_total_parameter_elements
        != CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
        or restored.dense_model_parameter_elements
        != CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
        or tuple(value.state_id for value in restored.continuous_allocations)
        != expected_allocation_ids
        or any(
            value.parameter_elements != CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
            for value in restored.continuous_allocations
        )
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "The runner requires the authenticated R12 exact-4x counted plan."
        )
    return restored


def reconstruct_exact4x_counted_plan_from_package(
    package: AuthenticatedLfm2CarrierChildPackage,
    *,
    expected_package_sha256: str,
) -> CountedChildStatePlan:
    """Derive the budget from authenticated packaged routes, never V3 counts."""

    if not isinstance(package, AuthenticatedLfm2CarrierChildPackage):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Count reconstruction requires an authenticated carrier child package."
        )
    expected = _digest(expected_package_sha256, "expected_package_sha256")
    if package.package_sha256 != expected:
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Authenticated carrier package differs from its external lineage pin."
        )
    plan = build_counted_child_state_plan(
        parent_unique_parameters=CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
        target_total_parameter_elements=(
            CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
        ),
        continuous_allocations=package.routes.allocations,
    )
    return _validated_exact4x_plan(plan)


@dataclass(frozen=True, slots=True)
class Lfm2Exact4xMovingGaugeV5Request:
    """Immutable preflight identity for one R12-based v5 experiment."""

    local_paths: Mapping[str, str]
    lineage_sha256s: Mapping[str, str]
    counted_child_state_plan: CountedChildStatePlan = field(
        repr=False,
        compare=False,
    )
    replay_devices: tuple[str, ...] = ("cuda:0",)
    recapture_checkpoint_batch_count: int = 16
    request_sha256: str = ""

    def __post_init__(self) -> None:
        paths = normalize_v5_local_paths(self.local_paths)
        lineage = dict(self.lineage_sha256s)
        if set(lineage) != REQUIRED_LINEAGE_KEYS:
            missing = sorted(REQUIRED_LINEAGE_KEYS.difference(lineage))
            extra = sorted(set(lineage).difference(REQUIRED_LINEAGE_KEYS))
            raise Lfm2Exact4xMovingGaugeV5Error(
                f"Lineage inventory changed; missing={missing}, extra={extra}."
            )
        lineage = {
            _identifier(name, "lineage key"): _digest(value, name)
            for name, value in sorted(lineage.items())
        }
        counted_plan = _validated_exact4x_plan(self.counted_child_state_plan)
        if lineage["counted_child_state_plan"] != counted_plan.fingerprint:
            raise Lfm2Exact4xMovingGaugeV5Error(
                "Lineage does not bind the reconstructed counted child-state plan."
            )
        devices = tuple(self.replay_devices)
        if (
            not devices
            or len(set(devices)) != len(devices)
            or any(
                not isinstance(value, str) or not value.strip() or len(value) > 128
                for value in devices
            )
        ):
            raise Lfm2Exact4xMovingGaugeV5Error(
                "replay_devices must be a nonempty unique device sequence."
            )
        checkpoint_batch_count = _positive_int(
            self.recapture_checkpoint_batch_count,
            "recapture_checkpoint_batch_count",
        )
        if not 8 <= checkpoint_batch_count <= 16:
            raise Lfm2Exact4xMovingGaugeV5Error(
                "Recapture checkpoints must cover between 8 and 16 batches."
            )
        object.__setattr__(self, "local_paths", paths)
        object.__setattr__(self, "lineage_sha256s", MappingProxyType(lineage))
        object.__setattr__(self, "counted_child_state_plan", counted_plan)
        object.__setattr__(self, "replay_devices", devices)
        object.__setattr__(
            self,
            "recapture_checkpoint_batch_count",
            checkpoint_batch_count,
        )
        expected = canonical_sha256(self._body())
        if self.request_sha256 not in {"", expected}:
            raise Lfm2Exact4xMovingGaugeV5Error("V5 request seal is invalid.")
        object.__setattr__(self, "request_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": REQUEST_SCHEMA,
            "local_paths": dict(self.local_paths),
            "lineage_sha256s": dict(self.lineage_sha256s),
            "counted_child_state_plan": self.counted_child_state_plan.as_dict(),
            "exact_budget": {
                "parent_unique_parameters": CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
                "target_total_parameter_elements": (
                    CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
                ),
                "dense_model_parameter_elements": (
                    CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
                ),
                "continuous_parameter_elements": (
                    2 * CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
                ),
                "realized_compaction_ratio": (
                    CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS
                    / CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
                ),
                "parameter_count_not_quantized_bytes": True,
            },
            "trace_geometry": {
                "expected_batch_count": EXPECTED_BATCH_COUNT,
                "expected_aligned_row_count": EXPECTED_ALIGNED_ROW_COUNT,
                "fit_batch_count": FIT_BATCH_COUNT,
                "heldout_batch_count": HELDOUT_BATCH_COUNT,
                "source_width": SOURCE_WIDTH,
                "target_width": TARGET_WIDTH,
            },
            "model_geometry": {
                "layer_count": LAYER_COUNT,
                "operator_layer_index": OPERATOR_LAYER_INDEX,
                "terminal_layer_index": TERMINAL_LAYER_INDEX,
                "operator_kind": "short_conv",
                "focused_feature_fields": list(FOCUSED_FEATURE_FIELDS),
                "replacement_tensor_names": list(REPLACEMENT_TENSOR_NAMES),
            },
            "chart_contract": {
                "base_encoder_tensor_path": BASE_ENCODER_TENSOR_PATH,
                "chart_modes": list(CHART_MODES),
                "locked_interfaces_use_E0": True,
                "E0R_kind": "right_orthogonal_procrustes",
                "E0A_kind": "fixed_source_subspace_full_child_coordinate",
                "E0A_encoder": "E0@A",
                "unrestricted_generalized_retained_as_diagnostic": True,
                "unrestricted_generalized_is_target_authority": False,
                "candidate_hashes_frozen_before_heldout_read": True,
            },
            "target_contract": {
                "arms": list(TARGET_ARMS),
                "publishable_arms": list(PUBLISHABLE_TARGET_ARMS),
                "sequential_no_recapture_is_negative_control": True,
                "conditioned_recapture_required_for_conditioned_arm": True,
            },
            "replay_contract": {
                "devices": list(self.replay_devices),
                "checkpoint_batch_count": (self.recapture_checkpoint_batch_count),
                "capture_api": "capture_lfm2_child_replay",
                "selected_layer_indices": [OPERATOR_LAYER_INDEX],
                "decode_trace_mode": "full_prefix_canonical",
                "parent_supplied": False,
                "parent_forward_count": 0,
                "temporary_operator_intervention_restored": True,
                "full_child_fingerprint_equal_before_after": True,
            },
            "lineage_authority": {
                "basis_and_counted_plan_are_hmac_bound_authorities": True,
                "v3_counts_are_consistency_only": True,
                "v3_numeric_counts_are_not_budget_authority": True,
                "tokenizer_bridge_is_authenticated_lineage": True,
            },
            "operation_contract": dict(FORBIDDEN_OPERATION_AUDIT),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "request_sha256": self.request_sha256}

    def verify_authentication(self) -> bool:
        try:
            rebuilt = Lfm2Exact4xMovingGaugeV5Request(
                local_paths=self.local_paths,
                lineage_sha256s=self.lineage_sha256s,
                counted_child_state_plan=self.counted_child_state_plan,
                replay_devices=self.replay_devices,
                recapture_checkpoint_batch_count=(
                    self.recapture_checkpoint_batch_count
                ),
            )
            return self.as_dict() == rebuilt.as_dict()
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> Lfm2Exact4xMovingGaugeV5Request:
        row = _canonical_mapping(value, "V5 request")
        try:
            request = cls(
                local_paths=row["local_paths"],
                lineage_sha256s=row["lineage_sha256s"],
                counted_child_state_plan=counted_child_state_plan_from_dict(
                    row["counted_child_state_plan"]
                ),
                replay_devices=tuple(row["replay_contract"]["devices"]),
                recapture_checkpoint_batch_count=row["replay_contract"][
                    "checkpoint_batch_count"
                ],
                request_sha256=row.get("request_sha256", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Lfm2Exact4xMovingGaugeV5Error(
                "V5 request fields are malformed."
            ) from exc
        if request.as_dict() != row:
            raise Lfm2Exact4xMovingGaugeV5Error(
                "V5 request contains unknown, stale, or changed derived fields."
            )
        return request


def authenticate_v5_local_preflight(
    request: Lfm2Exact4xMovingGaugeV5Request,
) -> Mapping[str, Any]:
    """Verify local-only paths and prove mutable roots cannot alter R12 inputs."""

    if not isinstance(request, Lfm2Exact4xMovingGaugeV5Request) or not (
        request.verify_authentication()
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Local preflight requires an authenticated v5 request."
        )
    resolved: dict[str, Path] = {}
    for name, raw in request.local_paths.items():
        supplied = Path(raw)
        if supplied.is_symlink():
            raise Lfm2Exact4xMovingGaugeV5Error(
                f"Local path {name!r} must not be a symlink."
            )
        if name in IMMUTABLE_INPUT_PATH_KEYS:
            try:
                path = supplied.resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                raise Lfm2Exact4xMovingGaugeV5Error(
                    f"Immutable local input {name!r} is missing."
                ) from exc
            if not path.is_dir():
                raise Lfm2Exact4xMovingGaugeV5Error(
                    f"Immutable local input {name!r} must be a directory."
                )
        else:
            if supplied.exists() and not supplied.is_dir():
                raise Lfm2Exact4xMovingGaugeV5Error(
                    f"Mutable local path {name!r} must be a directory."
                )
            path = supplied.resolve(strict=False)
        resolved[name] = path
    mutable_values = [resolved[name] for name in sorted(MUTABLE_OUTPUT_PATH_KEYS)]
    if len(set(mutable_values)) != len(mutable_values):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Phase, checkpoint, and output roots must be distinct."
        )
    for mutable_name in MUTABLE_OUTPUT_PATH_KEYS:
        mutable = resolved[mutable_name]
        for input_name in IMMUTABLE_INPUT_PATH_KEYS:
            immutable = resolved[input_name]
            if mutable == immutable or immutable in mutable.parents:
                raise Lfm2Exact4xMovingGaugeV5Error(
                    f"Mutable root {mutable_name!r} would alter immutable "
                    f"input {input_name!r}."
                )
    body = {
        "schema": LOCAL_PREFLIGHT_SCHEMA,
        "request_sha256": request.request_sha256,
        "resolved_paths": {name: str(path) for name, path in sorted(resolved.items())},
        "immutable_inputs_exist": True,
        "mutable_roots_are_outside_immutable_inputs": True,
        "network_paths_or_uris_used": False,
        "hugging_face_loader_used": False,
        "r12_mutation_allowed": False,
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def verify_v5_runtime_surface() -> Mapping[str, Any]:
    """Resolve every existing implementation API or fail before model access."""

    rows: dict[str, Any] = {}
    unavailable: list[str] = []
    for role, (module_name, attribute_name) in RUNTIME_API_BINDINGS.items():
        try:
            module = importlib.import_module(module_name)
            value = getattr(module, attribute_name)
            source = inspect.getsourcefile(value)
            if not callable(value) or source is None:
                raise TypeError("runtime binding is not a sourced callable")
            source_path = Path(source).resolve(strict=True)
            rows[role] = {
                "module": module_name,
                "attribute": attribute_name,
                "qualified_name": f"{value.__module__}.{value.__qualname__}",
                "implementation_file_sha256": _file_sha256(source_path),
            }
        except (AttributeError, ImportError, OSError, TypeError) as exc:
            unavailable.append(f"{role}:{module_name}.{attribute_name}:{exc}")
    if unavailable:
        raise Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError(
            "V5 runtime surface is incomplete; no phase executed: "
            + "; ".join(unavailable)
        )
    body = {
        "schema": RUNTIME_SURFACE_SCHEMA,
        "apis": rows,
        "api_count": len(rows),
        "all_required_apis_callable": True,
        "feature_phase_api_required": True,
        "fallback_folding_implementation_allowed": False,
        "preflight_only": True,
        "live_lineage_authenticated": False,
        "model_accessed": False,
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _checkpoint_ranges(batch_count: int, chunk_size: int) -> list[dict[str, int]]:
    return [
        {
            "ordinal": ordinal,
            "batch_start": start,
            "batch_stop": min(start + chunk_size, batch_count),
        }
        for ordinal, start in enumerate(range(0, batch_count, chunk_size))
    ]


def _phase(
    request: Lfm2Exact4xMovingGaugeV5Request,
    *,
    index: int,
    name: str,
    dependencies: Sequence[str],
    runtime_apis: Sequence[str],
    reads_splits: Sequence[str],
    child_forward_policy: str,
    temporary_child_mutation: bool,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": PHASE_SCHEMA,
        "request_sha256": request.request_sha256,
        "index": index,
        "name": name,
        "dependencies": list(dependencies),
        "runtime_apis": list(runtime_apis),
        "reads_splits": list(reads_splits),
        "child_forward_policy": child_forward_policy,
        "temporary_child_mutation_allowed": temporary_child_mutation,
        "child_state_must_be_restored_if_mutated": temporary_child_mutation,
        "parent_model_allowed": False,
        "parent_forward_count": 0,
        "training_allowed": False,
        "closed_form_only": True,
        "numeric_metrics_are_diagnostic_only": True,
        "details": _canonical_mapping(details, f"{name} details"),
        "operation_contract": dict(FORBIDDEN_OPERATION_AUDIT),
    }
    return _sealed(body, "phase_sha256")


def build_v5_phase_plan(
    request: Lfm2Exact4xMovingGaugeV5Request,
    *,
    _verify_reconstruction: bool = True,
) -> Mapping[str, Any]:
    """Build the deterministic, immutable v5 phase DAG."""

    if not isinstance(request, Lfm2Exact4xMovingGaugeV5Request) or not (
        request.verify_authentication()
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Phase planning requires an authenticated v5 request."
        )
    checkpoint_root = Path(request.local_paths["checkpoint_root"])
    checkpoint_base = (
        checkpoint_root
        / request.request_sha256
        / "conditioned-child-recapture"
        / "checkpoints"
    )
    ranges_by_chart: dict[str, list[dict[str, Any]]] = {}
    for chart_mode in CHART_MODES:
        chart_ranges = _checkpoint_ranges(
            EXPECTED_BATCH_COUNT,
            request.recapture_checkpoint_batch_count,
        )
        for row in chart_ranges:
            row["path"] = str(
                checkpoint_base / chart_mode / f"{row['ordinal']:04d}.json"
            )
        ranges_by_chart[chart_mode] = chart_ranges
    chart_candidates = {
        "E0": {
            "kind": "locked_balanced_basis",
            "encoder": BASE_ENCODER_TENSOR_PATH,
            "intermediate_fit": False,
        },
        "E0R": {
            "kind": "right_orthogonal_procrustes",
            "encoder": "E0@R",
            "intermediate_fit": True,
        },
        "E0A": {
            "kind": "fixed_source_subspace_full_child_coordinate",
            "encoder": "E0@A",
            "intermediate_fit": True,
            "unrestricted_generalized_retained_as_diagnostic": True,
            "unrestricted_generalized_is_target_authority": False,
        },
    }
    target_candidates = {
        "joint": {
            "formula": (
                "E_post_layer(parent_post_layer)-actual_child_pre-fixed_other_write"
            ),
            "fresh_recapture": False,
            "negative_control": False,
            "publication_eligible": True,
        },
        "sequential_no_recapture": {
            "formula": "sequential formulas evaluated on stale baseline child state",
            "fresh_recapture": False,
            "negative_control": True,
            "publication_eligible": False,
        },
        "sequential_conditioned": {
            "formula": (
                "operator closure then E_post_layer(parent_post_layer)-"
                "fresh_child_post_operator-fixed_mlp_write"
            ),
            "fresh_recapture": True,
            "negative_control": False,
            "publication_eligible": True,
        },
    }
    phase_rows = [
        _phase(
            request,
            index=0,
            name=PHASE_ORDER[0],
            dependencies=(),
            runtime_apis=(
                "carrier_package_authentication",
                "v3_request_authentication",
                "v3_ledger_authentication",
                "trace_cache_authentication",
                "basis_phase_authentication",
            ),
            reads_splits=(),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "lineage_sha256s": dict(request.lineage_sha256s),
                "counted_child_state_plan_fingerprint": (
                    request.counted_child_state_plan.fingerprint
                ),
                "count_authority": "hmac_basis_plus_packaged_route_allocations",
                "v3_role": "consistency_only",
                "r12_mutated": False,
            },
        ),
        _phase(
            request,
            index=1,
            name=PHASE_ORDER[1],
            dependencies=(PHASE_ORDER[0],),
            runtime_apis=("boundary_phase_import",),
            reads_splits=(),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "logical_layer_indices": [0, 1],
                "physical_parent_layer_indices": [0, TERMINAL_LAYER_INDEX],
                "physical_child_layer_indices": [0, TERMINAL_LAYER_INDEX],
                "focused_writer_layer_indices": [0],
                "expected_batches": EXPECTED_BATCH_COUNT,
                "expected_rows": EXPECTED_ALIGNED_ROW_COUNT,
                "cache_hit_before_source_provider": True,
                "tensor_bytes_authenticated_lazily_on_field_access": True,
                "cache_transport_contains_splits": ["fit", "heldout"],
                "solver_visible_splits": [],
                "heldout_solver_rows_read": False,
            },
        ),
        _phase(
            request,
            index=2,
            name=PHASE_ORDER[2],
            dependencies=(PHASE_ORDER[0], PHASE_ORDER[1]),
            runtime_apis=("feature_phase_import",),
            reads_splits=(),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "layer_index": 0,
                "field_ids": list(FOCUSED_FEATURE_FIELDS),
                "source_cache_field_patterns": [
                    "{phase}.layer.000.short_conv.context",
                    "{phase}.layer.000.mlp.post_swiglu",
                ],
                "prefill_parent_field_shapes": {
                    "short_conv.context": [1, 256, 1024],
                    "mlp.post_swiglu": [1, 256, 2560],
                },
                "prefill_child_field_shapes": {
                    "short_conv.context": [1, 256, 1024],
                    "mlp.post_swiglu": [1, 256, 321],
                },
                "source_storage_dtype_preserved_per_cache": True,
                "parent_child_dtype_equality_required": False,
                "boundary_alignment_reused": True,
                "parent_or_child_model_forward_required": False,
                "cache_transport_contains_splits": ["fit", "heldout"],
                "solver_visible_splits": [],
                "heldout_solver_rows_read": False,
            },
        ),
        _phase(
            request,
            index=3,
            name=PHASE_ORDER[3],
            dependencies=(PHASE_ORDER[1], PHASE_ORDER[2]),
            runtime_apis=(
                "orthogonal_chart_fit",
                "generalized_chart_fit",
                "base_endpoint_bind",
                "orthogonal_endpoint_bind",
                "generalized_endpoint_bind",
            ),
            reads_splits=("fit",),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "base_encoder": BASE_ENCODER_TENSOR_PATH,
                "chart_candidates": chart_candidates,
                "locked_boundaries": [
                    "layer.000.residual.pre",
                    "stack.terminal.residual",
                ],
                "fitted_boundaries": [
                    "layer.000.residual.post_operator",
                    "layer.000.residual.post_layer",
                ],
                "candidate_hash_freeze_barrier": True,
                "heldout_rows_read": False,
            },
        ),
        _phase(
            request,
            index=4,
            name=PHASE_ORDER[4],
            dependencies=(PHASE_ORDER[3],),
            runtime_apis=(
                "joint_target_build",
                "sequential_target_build",
                "writer_closed_form_solve",
            ),
            reads_splits=("fit",),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "target_candidates": target_candidates,
                "chart_modes": list(CHART_MODES),
                "no_recapture_arm_is_required_negative_control": True,
                "heldout_rows_read": False,
            },
        ),
        _phase(
            request,
            index=5,
            name=PHASE_ORDER[5],
            dependencies=(PHASE_ORDER[4],),
            runtime_apis=("child_only_replay", "child_state_fingerprint"),
            reads_splits=("fit",),
            child_forward_policy=(
                "required_child_only_or_authenticated_checkpoint_reuse"
            ),
            temporary_child_mutation=True,
            details={
                "capture_api": "capture_lfm2_child_replay",
                "selected_layer_indices": [0],
                "replay_devices": list(request.replay_devices),
                "checkpoint_ranges_by_chart": ranges_by_chart,
                "checkpoint_count_per_chart": len(ranges_by_chart[CHART_MODES[0]]),
                "checkpoint_count": sum(len(value) for value in ranges_by_chart.values()),
                "checkpoint_chain_required": True,
                "transport_captured_splits": ["fit", "heldout"],
                "solver_visible_splits": [],
                "heldout_transport_is_not_solver_access": True,
                "conditioned_field_ids": [
                    "short_conv.context",
                    "operator.output",
                    "residual.pre",
                    "residual.post_operator",
                    "mlp.post_swiglu",
                    "mlp.down",
                    "residual.post_layer",
                ],
                "temporary_install": "operator_writer_only",
                "child_state_restore_in_finally": True,
                "full_fingerprint_before_equals_after": True,
                "parent_model_supplied": False,
                "parent_forward_count": 0,
            },
        ),
        _phase(
            request,
            index=6,
            name=PHASE_ORDER[6],
            dependencies=(PHASE_ORDER[5],),
            runtime_apis=(
                "sequential_target_build",
                "writer_closed_form_solve",
            ),
            reads_splits=("fit",),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "conditioned_state_receipt_required": True,
                "operator_solution_sha256_required": True,
                "fresh_child_post_operator_required": True,
                "fresh_child_post_layer_required": True,
                "fresh_child_post_swiglu_required": True,
                "baseline_post_swiglu_for_conditioned_mlp_forbidden": True,
                "chart_modes": list(CHART_MODES),
                "heldout_rows_read": False,
            },
        ),
        _phase(
            request,
            index=7,
            name=PHASE_ORDER[7],
            dependencies=(PHASE_ORDER[3], PHASE_ORDER[6]),
            runtime_apis=(),
            reads_splits=("heldout",),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "heldout_batch_count": HELDOUT_BATCH_COUNT,
                "same_authenticated_heldout_inputs_for_every_candidate": True,
                "candidate_hashes_already_frozen": True,
                "numeric_metrics_are_diagnostic_only": True,
                "manual_semantic_review_performed_in_phase": False,
                "manual_semantic_review_required_downstream": True,
                "manual_semantic_review_status": "pending",
                "safety_refusal_is_not_failure": True,
                "max_output_token_truncation_is_harness_failure": True,
                "joint_arm_is_baseline_feature_diagnostic": True,
                "joint_arm_claims_full_simultaneous_recapture": False,
            },
        ),
        _phase(
            request,
            index=8,
            name=PHASE_ORDER[8],
            dependencies=(PHASE_ORDER[7],),
            runtime_apis=(),
            reads_splits=(),
            child_forward_policy="forbidden",
            temporary_child_mutation=False,
            details={
                "replacement_tensor_names": list(REPLACEMENT_TENSOR_NAMES),
                "chart_modes": list(CHART_MODES),
                "publishable_target_arms": list(PUBLISHABLE_TARGET_ARMS),
                "candidate_pack_count": (
                    len(CHART_MODES) * len(PUBLISHABLE_TARGET_ARMS)
                ),
                "sequential_no_recapture_packaged": False,
                "installed": False,
                "ready_for_atomic_apply": True,
                "r12_mutated": False,
            },
        ),
    ]
    body = {
        "schema": PHASE_PLAN_SCHEMA,
        "request_sha256": request.request_sha256,
        "phase_order": list(PHASE_ORDER),
        "phases": phase_rows,
        "phase_count": len(phase_rows),
        "base_encoder_tensor_path": BASE_ENCODER_TENSOR_PATH,
        "chart_modes": list(CHART_MODES),
        "target_arms": list(TARGET_ARMS),
        "publishable_candidate_count": (
            len(CHART_MODES) * len(PUBLISHABLE_TARGET_ARMS)
        ),
        "preflight_plan_only": True,
        "missing_runtime_handler_fails_before_first_phase": True,
        "operation_contract": dict(FORBIDDEN_OPERATION_AUDIT),
    }
    plan = _sealed(body, "plan_sha256")
    if _verify_reconstruction and not verify_v5_phase_plan(plan, request):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Constructed v5 phase plan failed exact reconstruction."
        )
    return MappingProxyType(plan)


def verify_v5_phase_plan(
    value: Mapping[str, Any],
    request: Lfm2Exact4xMovingGaugeV5Request,
) -> bool:
    try:
        original = _canonical_mapping(value, "V5 phase plan")
        row = dict(original)
        supplied = row.pop("plan_sha256")
        if supplied != canonical_sha256(row):
            return False
        phases = row["phases"]
        if (
            row["schema"] != PHASE_PLAN_SCHEMA
            or row["request_sha256"] != request.request_sha256
            or row["phase_order"] != list(PHASE_ORDER)
            or row["phase_count"] != len(PHASE_ORDER)
            or len(phases) != len(PHASE_ORDER)
        ):
            return False
        completed: set[str] = set()
        for index, phase in enumerate(phases):
            phase_row = dict(phase)
            phase_sha = phase_row.pop("phase_sha256")
            name = phase_row.get("name")
            if (
                phase_sha != canonical_sha256(phase_row)
                or phase_row.get("schema") != PHASE_SCHEMA
                or phase_row.get("request_sha256") != request.request_sha256
                or phase_row.get("index") != index
                or name != PHASE_ORDER[index]
                or not set(phase_row.get("dependencies", ())).issubset(completed)
                or phase_row.get("parent_forward_count") != 0
                or phase_row.get("training_allowed") is not False
                or phase_row.get("operation_contract")
                != dict(FORBIDDEN_OPERATION_AUDIT)
            ):
                return False
            completed.add(name)
        return original == dict(
            build_v5_phase_plan(request, _verify_reconstruction=False)
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class V5PhaseOutcome:
    """One handler result bound to its phase and exact upstream artifacts."""

    phase_name: str
    phase_sha256: str
    upstream_artifact_sha256s: Mapping[str, str]
    evidence: Mapping[str, Any]
    child_forward_count: int = 0
    parent_forward_count: int = 0
    heldout_rows_read: bool = False
    temporary_child_mutation_performed: bool = False
    child_state_restored: bool = False
    child_state_before_sha256: str | None = None
    child_state_after_sha256: str | None = None
    installed: bool = False
    ready_for_atomic_apply: bool = False
    authenticated_checkpoint_reuse: bool = False
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        name = _identifier(self.phase_name, "phase_name")
        phase_sha = _digest(self.phase_sha256, "phase_sha256")
        upstream = {
            _identifier(key, "upstream phase name"): _digest(
                value,
                f"upstream {key}",
            )
            for key, value in sorted(dict(self.upstream_artifact_sha256s).items())
        }
        evidence = _canonical_mapping(self.evidence, "phase outcome evidence")
        for count_name in ("child_forward_count", "parent_forward_count"):
            count = getattr(self, count_name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise Lfm2Exact4xMovingGaugeV5Error(
                    f"{count_name} must be a nonnegative integer."
                )
        for flag_name in (
            "heldout_rows_read",
            "temporary_child_mutation_performed",
            "child_state_restored",
            "installed",
            "ready_for_atomic_apply",
            "authenticated_checkpoint_reuse",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise Lfm2Exact4xMovingGaugeV5Error(f"{flag_name} must be boolean.")
        before = self.child_state_before_sha256
        after = self.child_state_after_sha256
        if (before is None) != (after is None):
            raise Lfm2Exact4xMovingGaugeV5Error(
                "Child state before/after fingerprints must be supplied together."
            )
        if before is not None:
            before = _digest(before, "child_state_before_sha256")
            after = _digest(after, "child_state_after_sha256")
        object.__setattr__(self, "phase_name", name)
        object.__setattr__(self, "phase_sha256", phase_sha)
        object.__setattr__(
            self,
            "upstream_artifact_sha256s",
            MappingProxyType(upstream),
        )
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        object.__setattr__(self, "child_state_before_sha256", before)
        object.__setattr__(self, "child_state_after_sha256", after)
        expected = canonical_sha256(self._body())
        if self.artifact_sha256 not in {"", expected}:
            raise Lfm2Exact4xMovingGaugeV5Error(
                "Phase outcome artifact seal is invalid."
            )
        object.__setattr__(self, "artifact_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": PHASE_OUTCOME_SCHEMA,
            "phase_name": self.phase_name,
            "phase_sha256": self.phase_sha256,
            "upstream_artifact_sha256s": dict(self.upstream_artifact_sha256s),
            "evidence": dict(self.evidence),
            "child_forward_count": self.child_forward_count,
            "parent_forward_count": self.parent_forward_count,
            "heldout_rows_read": self.heldout_rows_read,
            "temporary_child_mutation_performed": (
                self.temporary_child_mutation_performed
            ),
            "child_state_restored": self.child_state_restored,
            "child_state_before_sha256": self.child_state_before_sha256,
            "child_state_after_sha256": self.child_state_after_sha256,
            "installed": self.installed,
            "ready_for_atomic_apply": self.ready_for_atomic_apply,
            "authenticated_checkpoint_reuse": self.authenticated_checkpoint_reuse,
            "operation_audit": dict(FORBIDDEN_OPERATION_AUDIT),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "artifact_sha256": self.artifact_sha256}


V5PhaseHandler = Callable[
    [Mapping[str, Any], Mapping[str, V5PhaseOutcome]],
    V5PhaseOutcome,
]


def _validate_phase_outcome(
    phase: Mapping[str, Any],
    outcome: V5PhaseOutcome,
    completed: Mapping[str, V5PhaseOutcome],
) -> None:
    name = str(phase["name"])
    expected_upstream = {
        dependency: completed[dependency].artifact_sha256
        for dependency in phase["dependencies"]
    }
    if (
        outcome.phase_name != name
        or outcome.phase_sha256 != phase["phase_sha256"]
        or dict(outcome.upstream_artifact_sha256s) != expected_upstream
        or outcome.parent_forward_count != 0
        or outcome.installed
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} outcome violates its authenticated contract."
        )
    policy = phase["child_forward_policy"]
    checkpoint_reuse_policy = (
        policy == "required_child_only_or_authenticated_checkpoint_reuse"
    )
    if (
        (policy == "forbidden" and outcome.child_forward_count != 0)
        or (
            policy == "required_child_only"
            and outcome.child_forward_count <= 0
        )
        or (
            checkpoint_reuse_policy
            and (
                (outcome.authenticated_checkpoint_reuse and outcome.child_forward_count != 0)
                or (
                    not outcome.authenticated_checkpoint_reuse
                    and outcome.child_forward_count <= 0
                )
            )
        )
        or (
            not checkpoint_reuse_policy and outcome.authenticated_checkpoint_reuse
        )
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} child-forward count violates its policy."
        )
    heldout_expected = name == "compare_frozen_candidates_on_heldout"
    if outcome.heldout_rows_read is not heldout_expected:
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} crossed the FIT/heldout freeze barrier."
        )
    mutation_allowed = bool(phase["temporary_child_mutation_allowed"])
    if outcome.temporary_child_mutation_performed and not mutation_allowed:
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} temporary mutation evidence is inconsistent."
        )
    if outcome.temporary_child_mutation_performed:
        if (
            not outcome.child_state_restored
            or outcome.child_state_before_sha256 is None
            or outcome.child_state_before_sha256 != outcome.child_state_after_sha256
        ):
            raise Lfm2Exact4xMovingGaugeV5Error(
                "Conditioned recapture did not prove exact child-state restoration."
            )
    elif (
        outcome.child_state_restored
        or outcome.child_state_before_sha256 is not None
        or outcome.child_state_after_sha256 is not None
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} claims restoration without a temporary mutation."
        )
    if checkpoint_reuse_policy and (
        outcome.authenticated_checkpoint_reuse
        is outcome.temporary_child_mutation_performed
    ):
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} checkpoint reuse and cold recapture evidence conflict."
        )
    ready_expected = name == "stage_atomic_candidate_packs"
    if outcome.ready_for_atomic_apply is not ready_expected:
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Phase {name!r} candidate readiness claim is invalid."
        )


def _emit(event: str, **details: Any) -> None:
    row = {
        "schema": RUN_SCHEMA,
        "event": _identifier(event, "event"),
        "details": _canonical_mapping(details, "log details"),
    }
    sys.stdout.write(canonical_json_bytes(row).decode("ascii") + "\n")
    sys.stdout.flush()


def run_v5_phase_machine(
    request: Lfm2Exact4xMovingGaugeV5Request,
    handlers: Mapping[str, V5PhaseHandler],
    *,
    logger: Callable[..., None] = _emit,
) -> tuple[Mapping[str, V5PhaseOutcome], Mapping[str, Any]]:
    """Execute every thin orchestration phase exactly once in fixed order."""

    plan = build_v5_phase_plan(request)
    supplied_handlers = dict(handlers)
    if set(supplied_handlers) != set(PHASE_ORDER) or any(
        not callable(value) for value in supplied_handlers.values()
    ):
        missing = sorted(set(PHASE_ORDER).difference(supplied_handlers))
        extra = sorted(set(supplied_handlers).difference(PHASE_ORDER))
        raise Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError(
            "Every v5 runtime handler is required before execution; "
            f"missing={missing}, extra={extra}."
        )
    completed: dict[str, V5PhaseOutcome] = {}
    for phase in plan["phases"]:
        name = str(phase["name"])
        logger(
            "phase_begin",
            phase=name,
            phase_index=phase["index"],
            phase_sha256=phase["phase_sha256"],
            checkpoint_paths=phase["details"].get(
                "checkpoint_ranges_by_chart",
                phase["details"].get("checkpoint_ranges", []),
            ),
        )
        outcome = supplied_handlers[name](
            MappingProxyType(dict(phase)),
            MappingProxyType(dict(completed)),
        )
        if not isinstance(outcome, V5PhaseOutcome):
            raise Lfm2Exact4xMovingGaugeV5Error(
                f"Phase {name!r} did not return V5PhaseOutcome."
            )
        _validate_phase_outcome(phase, outcome, completed)
        completed[name] = outcome
        logger(
            "phase_complete",
            phase=name,
            artifact_sha256=outcome.artifact_sha256,
            child_forward_count=outcome.child_forward_count,
            parent_forward_count=outcome.parent_forward_count,
        )
    body = {
        "schema": RUN_SCHEMA,
        "request_sha256": request.request_sha256,
        "phase_plan_sha256": plan["plan_sha256"],
        "phase_order": list(PHASE_ORDER),
        "phase_artifact_sha256s": {
            name: completed[name].artifact_sha256 for name in PHASE_ORDER
        },
        "all_phases_executed_once": True,
        "source_r12_mutated": False,
        "candidate_packs_installed": False,
        "candidate_packs_ready_for_atomic_apply": True,
        "manual_semantic_review_pending": True,
        "operation_audit": dict(FORBIDDEN_OPERATION_AUDIT),
    }
    receipt = MappingProxyType(_sealed(body, "result_sha256"))
    logger("run_complete", result_sha256=receipt["result_sha256"])
    return MappingProxyType(completed), receipt


def stage_atomic_candidate_pack(
    request: Lfm2Exact4xMovingGaugeV5Request,
    *,
    chart_mode: str,
    target_arm: str,
    replacements: Mapping[str, Any],
    upstream_receipt_sha256s: Mapping[str, str],
    child_state_before_sha256: str,
    child_state_after_sha256: str,
) -> tuple[Path, Mapping[str, Any]]:
    """Atomically stage exactly the L0 operator and MLP replacement tensors."""

    if not request.verify_authentication():
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Candidate staging requires an authenticated v5 request."
        )
    if chart_mode not in CHART_MODES:
        raise Lfm2Exact4xMovingGaugeV5Error("Unknown v5 chart mode.")
    if target_arm not in PUBLISHABLE_TARGET_ARMS:
        raise Lfm2Exact4xMovingGaugeV5Error(
            "The stale no-recapture negative control cannot be packaged."
        )
    before = _digest(child_state_before_sha256, "child_state_before_sha256")
    after = _digest(child_state_after_sha256, "child_state_after_sha256")
    if before != after:
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Source child changed before candidate staging."
        )
    receipts = {
        _identifier(name, "upstream receipt name"): _digest(value, name)
        for name, value in sorted(dict(upstream_receipt_sha256s).items())
    }
    required_receipts = {
        "chart",
        "heldout_comparison",
        "lineage",
        "mlp_solution",
        "operator_solution",
        "target_plan",
    }
    if set(receipts) != required_receipts:
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Candidate upstream receipt inventory is incomplete."
        )
    supplied = dict(replacements)
    if set(supplied) != set(REPLACEMENT_TENSOR_NAMES):
        raise Lfm2Exact4xMovingGaugeV5Error(
            "Candidate pack must contain exactly two physical replacement tensors."
        )
    try:
        import safetensors.torch
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError(
            "Candidate staging requires torch and safetensors."
        ) from exc
    tensors: dict[str, Any] = {}
    descriptors: dict[str, Any] = {}
    for name in REPLACEMENT_TENSOR_NAMES:
        value = supplied[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.ndim != 2
            or value.dtype != torch.float32
            or value.is_complex()
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise Lfm2Exact4xMovingGaugeV5Error(
                f"Replacement tensor {name!r} is not a finite FP32 matrix."
            )
        tensor = value.detach().to(device="cpu").contiguous().clone()
        tensors[name] = tensor
        descriptors[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "numel": int(tensor.numel()),
            "tensor_sha256": fingerprint_lfm2_moving_gauge_tensor(tensor),
        }
    output_root = Path(request.local_paths["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise Lfm2Exact4xMovingGaugeV5Error("Candidate output root is unsafe.")
    temporary = output_root / f".candidate-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        tensor_path = temporary / "replacement-tensors.safetensors"
        safetensors.torch.save_file(
            tensors,
            str(tensor_path),
            metadata={
                "schema": CANDIDATE_PACK_SCHEMA,
                "request_sha256": request.request_sha256,
                "chart_mode": chart_mode,
                "target_arm": target_arm,
                "installed": "false",
            },
        )
        with tensor_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        tensor_file_sha256 = _file_sha256(tensor_path)
        body = {
            "schema": CANDIDATE_PACK_SCHEMA,
            "request_sha256": request.request_sha256,
            "counted_child_state_plan_fingerprint": (
                request.counted_child_state_plan.fingerprint
            ),
            "chart_mode": chart_mode,
            "target_arm": target_arm,
            "replacement_tensor_names": list(REPLACEMENT_TENSOR_NAMES),
            "replacement_tensor_count": 2,
            "tensor_descriptors": descriptors,
            "tensor_file": {
                "filename": tensor_path.name,
                "sha256": tensor_file_sha256,
                "size_bytes": tensor_path.stat().st_size,
            },
            "upstream_receipt_sha256s": receipts,
            "child_state_before_sha256": before,
            "child_state_after_sha256": after,
            "source_child_state_preserved": True,
            "installed": False,
            "ready_for_atomic_apply": True,
            "contains_complete_child_checkpoint": False,
            "contains_only_two_replacement_tensors_and_receipt": True,
            "parent_runtime_sidecar": False,
            "quantization_credit": False,
            "pruning_credit": False,
            "parameter_sharing_credit": False,
            "training_steps": 0,
            "numeric_metrics_are_diagnostic_only": True,
            "manual_semantic_review_status": "pending",
            "manual_semantic_review_required_downstream": True,
        }
        receipt = _sealed(body, "candidate_pack_sha256")
        receipt_path = temporary / "receipt.json"
        receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
        with receipt_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        destination = output_root / receipt["candidate_pack_sha256"]
        try:
            temporary.rename(destination)
        except FileExistsError:
            existing_receipt = destination / "receipt.json"
            existing_tensors = destination / tensor_path.name
            if (
                not existing_receipt.is_file()
                or json.loads(existing_receipt.read_text("ascii")) != receipt
                or not existing_tensors.is_file()
                or _file_sha256(existing_tensors) != tensor_file_sha256
            ):
                raise Lfm2Exact4xMovingGaugeV5Error(
                    "Existing candidate pack differs from this content address."
                )
            shutil.rmtree(temporary)
        return destination, MappingProxyType(receipt)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_canonical_request(path: Path) -> Lfm2Exact4xMovingGaugeV5Request:
    if not path.is_file() or path.is_symlink():
        raise Lfm2Exact4xMovingGaugeV5Error(
            "V5 request JSON must be an ordinary local file."
        )
    raw = path.read_bytes()
    try:
        row = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5Error(
            "V5 request JSON must be strict ASCII JSON."
        ) from exc
    if raw != canonical_json_bytes(row) + b"\n":
        raise Lfm2Exact4xMovingGaugeV5Error(
            "V5 request JSON must use canonical encoding."
        )
    return Lfm2Exact4xMovingGaugeV5Request.from_dict(row)


def _write_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2Exact4xMovingGaugeV5Error(
            f"Refusing to replace existing preflight artifact {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--phase-plan-json", type=Path, required=True)
    parser.add_argument(
        "--skip-runtime-surface-check",
        action="store_true",
        help="Only for offline request inspection; live execution remains unavailable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = _read_canonical_request(args.request_json)
        local_preflight = authenticate_v5_local_preflight(request)
        runtime_surface = (
            None if args.skip_runtime_surface_check else verify_v5_runtime_surface()
        )
        plan = build_v5_phase_plan(request)
        document = _sealed(
            {
                "schema": RUN_SCHEMA,
                "kind": "preflight_phase_plan",
                "request": request.as_dict(),
                "local_preflight": dict(local_preflight),
                "runtime_surface": (
                    None if runtime_surface is None else dict(runtime_surface)
                ),
                "phase_plan": dict(plan),
                "live_phases_executed": 0,
                "explicit_runtime_handlers_still_required": True,
                "source_r12_mutated": False,
                "operation_audit": dict(FORBIDDEN_OPERATION_AUDIT),
            },
            "result_sha256",
        )
        _write_json_noreplace(args.phase_plan_json, document)
        _emit(
            "preflight_complete",
            output_path=str(args.phase_plan_json.absolute()),
            result_sha256=document["result_sha256"],
            live_phases_executed=0,
        )
        return 0
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        failure = _sealed(
            {
                "schema": FAILURE_SCHEMA,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "live_phases_executed": 0,
            },
            "failure_sha256",
        )
        sys.stderr.write(canonical_json_bytes(failure).decode("ascii") + "\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
