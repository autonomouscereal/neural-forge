#!/usr/bin/env python3
"""Live, child-only handlers for the exact-4x moving-gauge v5 plan.

The thin v5 runner deliberately owns only the state machine.  This module owns
the live objects needed by that machine and keeps them in a closure: models,
trace handles, chart tensors, targets, and solutions never cross the JSON
outcome boundary.  Only authenticated digests and bounded numeric summaries do.

There is no parent-model entry point in this module.  Parent evidence is read
only from the immutable circuit cache through deferred selective sessions.
The one model loader is the direct local LFM2 child loader.
"""

from __future__ import annotations

import importlib
import hashlib
import math
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch

try:  # Works both as ``python experiments/...`` and as a namespace import.
    from experiments import run_lfm2_exact4x_moving_gauge_v5 as v5
except ImportError:  # pragma: no cover - direct script/import-path execution
    import run_lfm2_exact4x_moving_gauge_v5 as v5

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    load_existing_lfm2_circuit_balanced_basis_phase,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2ChildEvidenceProfile,
    Lfm2CircuitTraceCaptureConfig,
    capture_lfm2_child_replay,
    compile_lfm2_child_replay_numeric_policy,
    fingerprint_lfm2_parent,
    implementation_identity,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_model
from neural_forge.hf_lfm2_exact4x_causal_diagnostic import (
    authenticate_v3_first_divergence_ledger,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (
    authenticate_lfm2_exact4x_first_divergence_request,
)
from neural_forge.hf_lfm2_moving_gauge_boundary_phase_cache import (
    AuthenticatedLfm2MovingGaugeBoundaryPhase,
    Lfm2MovingGaugeBoundaryExtractionSources,
    Lfm2MovingGaugeBoundaryImportRequest,
    Lfm2MovingGaugeBoundaryLayerBinding,
    fingerprint_lfm2_moving_gauge_basis_split,
    lfm2_moving_gauge_boundary_implementation_sha256,
    load_or_extract_lfm2_moving_gauge_boundary_phase,
)
from neural_forge.hf_lfm2_moving_gauge_causal_targets import (
    Lfm2MovingGaugeCausalBoundaryPlan,
    build_lfm2_joint_writer_target,
    build_lfm2_moving_gauge_causal_state_receipt,
    build_lfm2_sequential_writer_target,
)
from neural_forge.hf_lfm2_moving_gauge_feature_phase_cache import (
    AuthenticatedLfm2MovingGaugeFeaturePhase,
    Lfm2MovingGaugeFeatureExtractionSources,
    Lfm2MovingGaugeFeatureImportRequest,
    derive_lfm2_moving_gauge_writer_feature_bindings,
    lfm2_moving_gauge_feature_implementation_sha256,
    load_or_extract_lfm2_moving_gauge_feature_phase,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBoundaryEvidence,
    bind_lfm2_base_e0_causal_endpoint_chart,
    bind_lfm2_generalized_e0a_causal_endpoint_chart,
    bind_lfm2_orthogonal_causal_endpoint_chart,
    fingerprint_lfm2_moving_gauge_tensor,
    fit_lfm2_cycle_coupled_moving_gauge_boundary,
    fit_lfm2_generalized_moving_gauge_chart_diagnostic,
    solve_lfm2_moving_gauge_anchored_writer_replacement,
)
from neural_forge.lfm2_carrier_bridge_publication import (
    fingerprint_lfm2_carrier_runtime_owned_tensor_state,
    relocate_and_bind_packaged_lfm2_carrier_routes,
)
from neural_forge.lfm2_carrier_child_package import (
    load_authenticated_lfm2_carrier_child_package,
)


LIVE_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-live-v1"
CONDITIONED_RECAPTURE_PROFILE_ROLE = "moving_gauge.conditioned_recapture"
CONDITIONED_RECLOSURE_FIELDS = (
    "mlp.down",
    "mlp.post_swiglu",
    "operator.output",
    "residual.post_layer",
    "residual.post_operator",
    "residual.pre",
    "short_conv.context",
)
REQUIRED_BOUNDARY_PROVENANCE_KEYS = frozenset(
    {
        "basis_completion",
        "basis_map",
        "basis_phase_key",
        "bridge_artifact",
        "bridge_profile",
        "bridge_trace_adapter",
        "child_completion",
        "child_identity",
        "child_manifest",
        "child_package",
        "child_selective_session",
        "child_state_dict",
        "child_topology",
        "mapped_calibration_file",
        "parent_completion",
        "parent_identity",
        "parent_manifest",
        "parent_selective_session",
        "parent_topology",
        "portable_runtime_receipt",
        "r12_run_receipt",
        "route_package",
        "route_runtime",
        "source_spans",
        "v3_ledger",
        "v3_request",
    }
)
_CONDITIONED_MODULE_CANDIDATES = (
    "neural_forge.hf_lfm2_moving_gauge_conditioned_recapture_phase_cache",
    "neural_forge.hf_lfm2_moving_gauge_conditioned_recapture_cache",
    "neural_forge.hf_lfm2_conditioned_recapture_phase_cache",
)
_CONDITIONED_FUNCTION_CANDIDATES = (
    "load_or_capture_lfm2_conditioned_recapture_phase",
    "load_or_capture_lfm2_moving_gauge_conditioned_recapture",
    "load_or_extract_lfm2_moving_gauge_conditioned_recapture",
)


class Lfm2Exact4xMovingGaugeV5LiveError(v5.Lfm2Exact4xMovingGaugeV5Error):
    """A live phase violated the authenticated child-only contract."""


class ConditionedRecaptureLoader(Protocol):
    """Narrow bridge implemented by the conditioned-recapture cache module.

    A complete cache hit must not call ``source_provider``.  A cold or partial
    miss calls it with a canonical ``batch_start``/``batch_stop`` range.  The
    returned artifact must expose authenticated FIT and heldout rows through
    the conditioned-recapture phase cache's lazy field accessor.
    """

    def __call__(
        self,
        cache: DynamicFoldPhaseCache,
        request: Any,
        *,
        authority: Any,
        checkpoint_root: str | Path,
        range_provider: Callable[[int, int], Any] | None = None,
        maximum_new_checkpoints: int | None = None,
        events: Any | None = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class V5LivePhaseExecution:
    """Private phase result normalized into the thin runner's outcome type."""

    artifact: Any = field(repr=False, compare=False)
    evidence: Mapping[str, Any]
    child_forward_count: int = 0
    heldout_rows_read: bool = False
    temporary_child_mutation_performed: bool = False
    child_state_restored: bool = False
    child_state_before_sha256: str | None = None
    child_state_after_sha256: str | None = None
    authenticated_checkpoint_reuse: bool = False
    ready_for_atomic_apply: bool = False


V5LivePhaseOverride = Callable[
    [
        "Lfm2Exact4xMovingGaugeV5LiveRuntime",
        Mapping[str, Any],
        Mapping[str, v5.V5PhaseOutcome],
    ],
    V5LivePhaseExecution,
]


@dataclass(frozen=True, slots=True)
class Lfm2Exact4xMovingGaugeV5LiveBindings:
    """Non-serializable authorities and already-pinned source identities.

    Secrets remain inside the authority objects.  The exact 26-key provenance
    mapping closes the boundary-import request without inventing lineage from
    directory names.  No field accepts a parent model, parent loader, or parent
    forward callback.
    """

    trace_adapter: Any = field(repr=False, compare=False)
    basis_phase_authority: Any = field(repr=False, compare=False)
    boundary_phase_authority: Any = field(repr=False, compare=False)
    feature_phase_authority: Any = field(repr=False, compare=False)
    conditioned_recapture_authority: Any = field(repr=False, compare=False)
    provenance_sha256s: Mapping[str, str]
    v3_first_divergence_request: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    v3_first_divergence_ledger: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    basis_split_by_batch: Mapping[int, str]
    parent_storage_dtype: str
    child_storage_dtype: str
    child_feature_storage_dtypes: Mapping[str, str]
    conditioned_recapture_loader: ConditionedRecaptureLoader | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    stage_candidate_pack: Callable[..., Any] = field(
        default=v5.stage_atomic_candidate_pack,
        repr=False,
        compare=False,
    )
    runtime_device: str | None = None
    runtime_dtype: str | None = None
    boundary_shard_batch_count: int = 32
    feature_shard_batch_count: int = 32

    def __post_init__(self) -> None:
        provenance = dict(self.provenance_sha256s)
        if set(provenance) != REQUIRED_BOUNDARY_PROVENANCE_KEYS:
            missing = sorted(REQUIRED_BOUNDARY_PROVENANCE_KEYS.difference(provenance))
            extra = sorted(set(provenance).difference(REQUIRED_BOUNDARY_PROVENANCE_KEYS))
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                f"Boundary provenance inventory changed; missing={missing}, "
                f"extra={extra}."
            )
        feature_dtypes = dict(self.child_feature_storage_dtypes)
        if set(feature_dtypes) != set(v5.FOCUSED_FEATURE_FIELDS):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Feature storage dtype mapping must cover exactly the two "
                "focused writer features."
            )
        splits = dict(self.basis_split_by_batch)
        split_values = tuple(splits[index] for index in sorted(splits))
        if (
            set(splits) != set(range(v5.EXPECTED_BATCH_COUNT))
            or set(split_values) != {"fit", "heldout"}
            or split_values.count("fit") != v5.FIT_BATCH_COUNT
            or split_values.count("heldout")
            != v5.EXPECTED_BATCH_COUNT - v5.FIT_BATCH_COUNT
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Basis split must be the exact 256-entry 192 FIT / 64 HELDOUT "
                "assignment."
            )
        if (
            self.parent_storage_dtype != "bfloat16"
            or self.child_storage_dtype != "float32"
            or set(feature_dtypes.values()) != {"float32"}
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Exact v5 storage bindings require BF16 parent evidence and FP32 "
                "child residual plus focused-feature evidence."
            )
        for name in ("boundary_shard_batch_count", "feature_shard_batch_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    f"{name} must be a positive integer."
                )
        if not callable(self.stage_candidate_pack):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "stage_candidate_pack must be callable."
            )
        object.__setattr__(self, "provenance_sha256s", MappingProxyType(provenance))
        object.__setattr__(
            self,
            "child_feature_storage_dtypes",
            MappingProxyType(feature_dtypes),
        )
        object.__setattr__(self, "basis_split_by_batch", MappingProxyType(splits))


def _digest_from(value: Any, *names: str) -> str:
    """Read one authenticated digest without treating ``repr`` as evidence."""

    for name in names:
        if isinstance(value, Mapping):
            candidate = value.get(name)
        else:
            candidate = getattr(value, name, None)
        if isinstance(candidate, str) and len(candidate) == 64:
            return candidate
    report = getattr(value, "report", None)
    if isinstance(report, Mapping):
        for name in names:
            candidate = report.get(name)
            if isinstance(candidate, str) and len(candidate) == 64:
                return candidate
    raise Lfm2Exact4xMovingGaugeV5LiveError(
        f"Authenticated object exposes none of the required digests {names!r}."
    )


def _verify_exact_request_lineage(
    request: v5.Lfm2Exact4xMovingGaugeV5Request,
    actual_lineage_sha256s: Mapping[str, str],
) -> Mapping[str, str]:
    actual = dict(actual_lineage_sha256s)
    if set(actual) != v5.REQUIRED_LINEAGE_KEYS:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Reconstructed live lineage does not have the exact v5 inventory."
        )
    if actual != dict(request.lineage_sha256s):
        differing = sorted(
            name
            for name in v5.REQUIRED_LINEAGE_KEYS
            if actual.get(name) != request.lineage_sha256s.get(name)
        )
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Self-sealed v5 request differs from authenticated live lineage at "
            f"{differing}."
        )
    return MappingProxyType(actual)


def _runtime_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Authenticated v3 runtime dtype must be float32 or bfloat16."
        ) from exc


def _cache_completion_sha256(cache: Any) -> str:
    completion = getattr(cache, "_completion", None)
    if not isinstance(completion, Mapping):
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Trace cache does not expose authenticated completion evidence."
        )
    return str(completion["completion_sha256"])


def _named_parameter(model: Any, name: str) -> torch.nn.Parameter:
    try:
        value = dict(model.named_parameters())[name]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            f"Authenticated child has no physical parameter {name!r}."
        ) from exc
    if not isinstance(value, torch.nn.Parameter):
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            f"Physical child value {name!r} is not a parameter."
        )
    return value


def _tensor_tree_requires_grad(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad or value.grad_fn is not None)
    if isinstance(value, Mapping):
        return any(_tensor_tree_requires_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_tensor_tree_requires_grad(item) for item in value)
    fields = getattr(type(value), "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return any(
            _tensor_tree_requires_grad(getattr(value, name)) for name in fields
        )
    return False


def _solution_sha256(solution: torch.Tensor, report: Mapping[str, Any]) -> str:
    claimed = report.get("replacement_weight_sha256")
    actual = fingerprint_lfm2_moving_gauge_tensor(solution)
    if claimed != actual:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Closed-form solution report does not bind its replacement tensor."
        )
    receipt = report.get("report_sha256")
    if not isinstance(receipt, str) or len(receipt) != 64:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Closed-form solution has no authenticated report digest."
        )
    return receipt


def _joint_split_solution_receipts(
    operator_solution: torch.Tensor,
    mlp_solution: torch.Tensor,
    combined_report: Mapping[str, Any],
) -> Mapping[str, str]:
    if (
        operator_solution.ndim != 2
        or mlp_solution.ndim != 2
        or int(operator_solution.shape[0]) != int(mlp_solution.shape[0])
        or operator_solution.dtype != mlp_solution.dtype
    ):
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Joint solution slices cannot reconstruct one combined matrix."
        )
    combined = torch.cat((operator_solution, mlp_solution), dim=1).contiguous()
    combined_report_sha256 = _solution_sha256(combined, combined_report)
    combined_tensor_sha256 = fingerprint_lfm2_moving_gauge_tensor(combined)
    operator_width = int(operator_solution.shape[1])
    combined_width = int(combined.shape[1])
    shared = {
        "schema": "neural-forge-lfm2-joint-solution-split-receipt-v1",
        "combined_report_sha256": combined_report_sha256,
        "combined_replacement_tensor_sha256": combined_tensor_sha256,
        "combined_shape": [int(size) for size in combined.shape],
        "combined_dtype": str(combined.dtype).removeprefix("torch."),
        "operator_width": operator_width,
    }
    parts = {
        "operator": (
            v5.REPLACEMENT_TENSOR_NAMES[0],
            operator_solution,
            [0, operator_width],
        ),
        "mlp": (
            v5.REPLACEMENT_TENSOR_NAMES[1],
            mlp_solution,
            [operator_width, combined_width],
        ),
    }
    return MappingProxyType(
        {
            part: canonical_sha256(
                {
                    **shared,
                    "part": part,
                    "replacement_tensor_name": parameter_name,
                    "column_slice": column_slice,
                    "split_tensor_sha256": (
                        fingerprint_lfm2_moving_gauge_tensor(tensor)
                    ),
                    "split_shape": [int(size) for size in tensor.shape],
                }
            )
            for part, (parameter_name, tensor, column_slice) in parts.items()
        }
    )


def _circuit_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = b"".join(
        (
            b"neural-forge-lfm2-circuit-tensor-v1\0",
            str(tensor.dtype).removeprefix("torch.").encode("ascii"),
            b"\0",
            canonical_json_bytes([int(size) for size in tensor.shape]),
            b"\0",
        )
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _fingerprint_with_parameter_replacement(
    model: Any,
    *,
    parameter_name: str,
    replacement: torch.Tensor,
) -> Mapping[str, Any]:
    """Derive the exact child fingerprint without mutating a cache-hit child."""

    base = fingerprint_lfm2_parent(model).as_dict()
    state = model.state_dict()
    if parameter_name not in state:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Conditioned fingerprint replacement parameter is absent."
        )
    expected = state[parameter_name]
    candidate = replacement.detach().to(
        device="cpu",
        dtype=expected.dtype,
    ).contiguous()
    if tuple(candidate.shape) != tuple(expected.shape):
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Conditioned fingerprint replacement geometry changed."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-parent-state-v1\0")
    for name in sorted(state):
        value = candidate if name == parameter_name else state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_circuit_tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    state_sha256 = digest.hexdigest()
    body = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": base["model_class"],
        "config_sha256": base["config_sha256"],
        "state_dict_sha256": state_sha256,
        "state_tensor_count": base["state_tensor_count"],
        "runtime_buffer_sha256": base["runtime_buffer_sha256"],
        "runtime_buffer_count": base["runtime_buffer_count"],
    }
    return MappingProxyType(
        {**body, "identity_sha256": canonical_sha256(body)}
    )


def _chart_bundle_sha256(charts: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            endpoint: _digest_from(chart, "report_sha256", "result_sha256")
            for endpoint, chart in sorted(charts.items())
        }
    )


def _relative_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    prediction64 = prediction.detach().to(device="cpu", dtype=torch.float64)
    target64 = target.detach().to(device="cpu", dtype=torch.float64)
    weights64 = weights.detach().to(device="cpu", dtype=torch.float64)
    if prediction64.shape != target64.shape or prediction64.ndim != 2:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Heldout prediction and target geometry changed."
        )
    if weights64.ndim != 1 or int(weights64.shape[0]) != int(target64.shape[0]):
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Heldout weights do not align to heldout rows."
        )
    residual = prediction64 - target64
    numerator = torch.sum(weights64[:, None] * residual.square())
    denominator = torch.sum(weights64[:, None] * target64.square())
    floor = torch.finfo(torch.float64).tiny
    return float(torch.sqrt(numerator / torch.clamp(denominator, min=floor)).item())


def _conditioned_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _conditioned_rows(
    value: Any,
    *,
    split: str,
    field_name: str,
    heldout_capability: Any | None = None,
) -> torch.Tensor:
    field_rows = getattr(value, "field_rows", None)
    if callable(field_rows):
        selected = field_rows(
            field_name,
            split=split,
            heldout_capability=heldout_capability,
        )
        rows = getattr(selected, "rows", None)
    else:
        method = getattr(value, "rows", None)
        if callable(method):
            if heldout_capability is not None:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Legacy conditioned rows cannot authenticate HELDOUT access."
                )
            try:
                rows = method(split=split, field_name=field_name)
            except TypeError:
                rows = method(field_name, split=split)
        else:
            if heldout_capability is not None:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Mapping conditioned rows cannot authenticate HELDOUT access."
                )
            root = _conditioned_value(value, "rows")
            try:
                rows = root[split][field_name]
            except (KeyError, TypeError) as exc:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Conditioned recapture artifact has no authenticated split/field "
                    f"rows for {split}.{field_name}."
                ) from exc
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.numel() == 0:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            f"Conditioned rows {split}.{field_name} must be a nonempty matrix."
        )
    result = rows.detach().to(device="cpu").contiguous().clone()
    if result.requires_grad or result.grad_fn is not None:
        raise Lfm2Exact4xMovingGaugeV5LiveError(
            "Conditioned recapture rows retain an autograd graph."
        )
    return result


def _resolve_conditioned_recapture_loader() -> tuple[Callable[..., Any], str]:
    failures: list[str] = []
    for module_name in _CONDITIONED_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        implementation = getattr(
            module,
            "lfm2_conditioned_recapture_implementation_sha256",
            getattr(
                module,
                "lfm2_moving_gauge_conditioned_recapture_implementation_sha256",
                None,
            ),
        )
        for function_name in _CONDITIONED_FUNCTION_CANDIDATES:
            function = getattr(module, function_name, None)
            if callable(function) and callable(implementation):
                return function, str(implementation())
        failures.append(f"{module_name}: required callable surface is absent")
    raise v5.Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError(
        "Conditioned-recapture cache API is unavailable; phase 5 cannot "
        "truthfully run or synthesize checkpoints. " + "; ".join(failures)
    )


class Lfm2Exact4xMovingGaugeV5LiveRuntime:
    """Closure-owned live state behind the nine thin-runner handlers."""

    def __init__(
        self,
        request: v5.Lfm2Exact4xMovingGaugeV5Request,
        bindings: Lfm2Exact4xMovingGaugeV5LiveBindings,
        *,
        phase_overrides: Mapping[str, V5LivePhaseOverride] | None = None,
    ) -> None:
        if not isinstance(request, v5.Lfm2Exact4xMovingGaugeV5Request) or not (
            request.verify_authentication()
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Live handlers require an authenticated v5 request."
            )
        if not isinstance(bindings, Lfm2Exact4xMovingGaugeV5LiveBindings):
            raise TypeError("bindings must be Lfm2Exact4xMovingGaugeV5LiveBindings.")
        overrides = dict(phase_overrides or {})
        if not set(overrides).issubset(v5.PHASE_ORDER) or any(
            not callable(value) for value in overrides.values()
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Phase overrides must name only canonical v5 phases."
            )
        self.request = request
        self.bindings = bindings
        self.phase_overrides = MappingProxyType(overrides)
        self.artifacts: dict[str, Any] = {}
        self.phase_call_order: list[str] = []
        self._mutation_observations: list[dict[str, str]] = []

    def handlers(self) -> Mapping[str, v5.V5PhaseHandler]:
        """Return all nine handlers, each closed over this exact runtime."""

        return MappingProxyType(
            {
                name: self._handler(name)
                for name in v5.PHASE_ORDER
            }
        )

    def _handler(self, name: str) -> v5.V5PhaseHandler:
        def handler(
            phase: Mapping[str, Any],
            completed: Mapping[str, v5.V5PhaseOutcome],
        ) -> v5.V5PhaseOutcome:
            return self._execute(name, phase, completed)

        handler.__name__ = f"handle_{name}"
        return handler

    def _execute(
        self,
        name: str,
        phase: Mapping[str, Any],
        completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> v5.V5PhaseOutcome:
        if phase.get("name") != name or name in self.artifacts:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                f"Live phase {name!r} was misrouted or repeated."
            )
        expected_prior = tuple(v5.PHASE_ORDER[: int(phase["index"])])
        if tuple(completed) != expected_prior:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                f"Live phase {name!r} observed a noncanonical completion order."
            )
        override = self.phase_overrides.get(name)
        execution = (
            override(self, phase, completed)
            if override is not None
            else getattr(self, f"_phase_{name}")(phase, completed)
        )
        if not isinstance(execution, V5LivePhaseExecution):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                f"Live phase {name!r} did not return V5LivePhaseExecution."
            )
        self.artifacts[name] = execution.artifact
        self.phase_call_order.append(name)
        upstream = {
            dependency: completed[dependency].artifact_sha256
            for dependency in phase["dependencies"]
        }
        return v5.V5PhaseOutcome(
            phase_name=name,
            phase_sha256=str(phase["phase_sha256"]),
            upstream_artifact_sha256s=upstream,
            evidence=execution.evidence,
            child_forward_count=execution.child_forward_count,
            parent_forward_count=0,
            heldout_rows_read=execution.heldout_rows_read,
            temporary_child_mutation_performed=(
                execution.temporary_child_mutation_performed
            ),
            child_state_restored=execution.child_state_restored,
            child_state_before_sha256=execution.child_state_before_sha256,
            child_state_after_sha256=execution.child_state_after_sha256,
            installed=False,
            ready_for_atomic_apply=execution.ready_for_atomic_apply,
            authenticated_checkpoint_reuse=(
                execution.authenticated_checkpoint_reuse
            ),
        )

    # ------------------------------------------------------------------
    # Phase 0: authenticate every local lineage authority and load only the
    # direct local child.  Trace caches are opened metadata-only (verify=False).
    # ------------------------------------------------------------------
    def _phase_authenticate_lineage(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        provenance = self.bindings.provenance_sha256s
        request_doc = authenticate_lfm2_exact4x_first_divergence_request(
            dict(self.bindings.v3_first_divergence_request)
        )
        ledger = authenticate_v3_first_divergence_ledger(
            dict(self.bindings.v3_first_divergence_ledger),
            expected_request=request_doc,
        )
        if (
            request_doc["request_sha256"] != provenance["v3_request"]
            or ledger["ledger_sha256"] != provenance["v3_ledger"]
            or request_doc["runtime_device"]
            != (self.bindings.runtime_device or request_doc["runtime_device"])
            or request_doc["runtime_dtype"]
            != (self.bindings.runtime_dtype or request_doc["runtime_dtype"])
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Injected v3 evidence or runtime differs from its lineage pins."
            )
        runtime_device = torch.device(request_doc["runtime_device"])
        runtime_dtype = _runtime_dtype(request_doc["runtime_dtype"])
        child_directory = Path(self.request.local_paths["published_child"])
        package = load_authenticated_lfm2_carrier_child_package(
            child_directory,
            expected_package_sha256=self.request.lineage_sha256s[
                "r12_child_package"
            ],
        )
        canonical_child = load_lfm2_direct_local_model(
            child_directory,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        package = load_authenticated_lfm2_carrier_child_package(
            child_directory,
            child_model=canonical_child,
            expected_package_sha256=package.package_sha256,
        )
        child, routing, relocation = relocate_and_bind_packaged_lfm2_carrier_routes(
            canonical_child,
            package.routes,
            provenance_resource_fingerprint=request_doc["implementation_identity"][
                "implementation_sha256"
            ],
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            expected_target_rows=int(package.receipt["target_rows"]),
            expected_hidden_size=int(package.receipt["hidden_size"]),
        )
        child.eval()
        fingerprint = fingerprint_lfm2_parent(child)
        route_state = fingerprint_lfm2_carrier_runtime_owned_tensor_state(routing)
        relocation_row = relocation.as_dict()
        route_row = routing.as_dict()
        if (
            package.package_sha256 != provenance["child_package"]
            or package.routes.package_sha256 != provenance["route_package"]
            or fingerprint.identity_sha256 != provenance["child_identity"]
            or fingerprint.state_dict_sha256 != provenance["child_state_dict"]
            or route_row["runtime_sha256"] != provenance["route_runtime"]
            or relocation_row["receipt_sha256"]
            != provenance["portable_runtime_receipt"]
            or request_doc["child_state_sha256"] != fingerprint.identity_sha256
            or request_doc["child_package_sha256"] != package.package_sha256
            or request_doc["route_package_sha256"]
            != package.routes.package_sha256
            or request_doc["route_runtime_sha256"] != route_row["runtime_sha256"]
            or request_doc["portable_runtime_receipt_sha256"]
            != relocation_row["receipt_sha256"]
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Direct local child/package/route differs from R12 or v3 lineage."
            )
        reconstructed = v5.reconstruct_exact4x_counted_plan_from_package(
            package,
            expected_package_sha256=package.package_sha256,
        )
        if reconstructed.as_dict() != self.request.counted_child_state_plan.as_dict():
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Packaged routes do not reconstruct the requested exact-4x count."
            )
        adapter = self.bindings.trace_adapter
        verify_adapter = getattr(adapter, "verify", None)
        if not callable(verify_adapter) or verify_adapter() is not adapter:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Injected tokenizer-bridge trace adapter did not authenticate."
            )
        if (
            getattr(adapter, "evidence_sha256", None)
            != provenance["bridge_trace_adapter"]
            or getattr(adapter, "bridge_artifact_sha256", None)
            != provenance["bridge_artifact"]
            or getattr(adapter, "transport_profile_fingerprint", None)
            != provenance["bridge_profile"]
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Tokenizer-bridge adapter differs from boundary provenance."
            )
        parent_cache = open_lfm2_circuit_trace_cache(
            self.request.local_paths["parent_trace_cache"],
            verify=False,
        )
        child_cache = open_lfm2_circuit_trace_cache(
            self.request.local_paths["child_trace_cache"],
            verify=False,
        )
        for side, cache in (("parent", parent_cache), ("child", child_cache)):
            manifest = cache.manifest
            if (
                manifest["manifest_sha256"] != provenance[f"{side}_manifest"]
                or _cache_completion_sha256(cache)
                != provenance[f"{side}_completion"]
                or manifest["parent"]["identity_sha256"]
                != provenance[f"{side}_identity"]
                or manifest["layer_topology"]["layer_topology_sha256"]
                != provenance[f"{side}_topology"]
                or manifest["corpus"]["batch_count"] != v5.EXPECTED_BATCH_COUNT
            ):
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    f"{side} trace cache metadata differs from provenance."
                )
        if parent_cache.cache_key != request_doc["trace_cache_key"]:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Parent trace cache differs from the v3 request."
            )
        basis_cache = DynamicFoldPhaseCache(
            self.request.local_paths["basis_phase_cache"]
        )
        basis_phase = load_existing_lfm2_circuit_balanced_basis_phase(
            basis_cache,
            provenance["basis_phase_key"],
            anchor_authority=self.bindings.basis_phase_authority,
            expected_parent_identity_sha256=provenance["parent_identity"],
            expected_trace_cache_key=parent_cache.cache_key,
            expected_trace_selective_session_sha256=provenance[
                "parent_selective_session"
            ],
            expected_source_width=v5.SOURCE_WIDTH,
            expected_target_width=v5.TARGET_WIDTH,
            expected_map_sha256=provenance["basis_map"],
        )
        actual_lineage = _verify_exact_request_lineage(
            self.request,
            {
                "r12_child_package": package.package_sha256,
                "r12_child_identity": fingerprint.identity_sha256,
                "r12_state_dict": fingerprint.state_dict_sha256,
                "v3_request": request_doc["request_sha256"],
                "v3_ledger": ledger["ledger_sha256"],
                "parent_trace_cache_completion": _cache_completion_sha256(
                    parent_cache
                ),
                "child_trace_cache_completion": _cache_completion_sha256(
                    child_cache
                ),
                "basis_phase_key": basis_phase.phase_key.phase_key_sha256,
                "basis_phase_completion": basis_phase.completion_sha256,
                "basis_source_anchor": basis_phase.source_anchor[
                    "hmac_sha256"
                ],
                "tokenizer_bridge": adapter.bridge_artifact_sha256,
                "counted_child_state_plan": reconstructed.fingerprint,
            },
        )
        self.child = child
        self.routing = routing
        self.package = package
        self.child_fingerprint = fingerprint
        self.route_state = route_state
        self.relocation = relocation
        self.parent_cache = parent_cache
        self.child_cache = child_cache
        self.basis_cache = basis_cache
        self.basis_phase = basis_phase
        self.v3_request = request_doc
        self.v3_ledger = ledger
        artifact = {
            "child": child,
            "routing": routing,
            "basis_phase": basis_phase,
            "parent_cache": parent_cache,
            "child_cache": child_cache,
        }
        evidence = {
            "schema": LIVE_SCHEMA,
            "kind": "authenticated_lineage",
            "child_package_sha256": package.package_sha256,
            "child_identity_sha256": fingerprint.identity_sha256,
            "child_state_dict_sha256": fingerprint.state_dict_sha256,
            "route_runtime_sha256": route_row["runtime_sha256"],
            "route_state_sha256": route_state["state_sha256"],
            "portable_runtime_receipt_sha256": relocation_row["receipt_sha256"],
            "v3_request_sha256": request_doc["request_sha256"],
            "v3_ledger_sha256": ledger["ledger_sha256"],
            "basis_phase_key_sha256": basis_phase.phase_key.phase_key_sha256,
            "basis_completion_sha256": basis_phase.completion_sha256,
            "counted_child_state_plan_sha256": reconstructed.fingerprint,
            "exact_request_lineage_sha256": canonical_sha256(
                dict(actual_lineage)
            ),
            "parent_cache_open_verify": False,
            "child_cache_open_verify": False,
            "parent_model_loaded": False,
            "parent_forward_count": 0,
            "child_loader": (
                "neural_forge.hf_lfm2_direct_local.load_lfm2_direct_local_model"
            ),
            "hugging_face_loader_used": False,
            "network_access": False,
        }
        return V5LivePhaseExecution(artifact=artifact, evidence=evidence)

    # ------------------------------------------------------------------
    # Phase 1/2: authenticated compact imports.  Source providers and their
    # selective sessions are constructed only after a phase-cache miss.
    # ------------------------------------------------------------------
    def _phase_import_boundary_rows(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        provenance = self.bindings.provenance_sha256s
        parent_topology = self.parent_cache.manifest["layer_topology"]
        child_topology = self.child_cache.manifest["layer_topology"]
        bindings: list[Lfm2MovingGaugeBoundaryLayerBinding] = []
        for logical, physical in ((0, 0), (1, v5.TERMINAL_LAYER_INDEX)):
            parent_kind = parent_topology["layers"][physical]["contract"]["kind"]
            child_kind = child_topology["layers"][physical]["contract"]["kind"]
            if parent_kind != child_kind:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Physical parent/child boundary operator kinds differ."
                )
            bindings.append(
                Lfm2MovingGaugeBoundaryLayerBinding(
                    layer_index=logical,
                    parent_layer_index=physical,
                    child_layer_index=physical,
                    operator_kind=parent_kind,
                )
            )
        basis_split_sha256 = fingerprint_lfm2_moving_gauge_basis_split(
            self.bindings.basis_split_by_batch,
            basis_phase_key_sha256=provenance["basis_phase_key"],
            basis_completion_sha256=provenance["basis_completion"],
            basis_map_sha256=provenance["basis_map"],
        )
        request = Lfm2MovingGaugeBoundaryImportRequest(
            parent_cache_key=self.parent_cache.cache_key,
            child_cache_key=self.child_cache.cache_key,
            expected_batch_count=v5.EXPECTED_BATCH_COUNT,
            expected_aligned_row_count=v5.EXPECTED_ALIGNED_ROW_COUNT,
            parent_width=v5.SOURCE_WIDTH,
            child_width=v5.TARGET_WIDTH,
            parent_parameter_count=(
                self.request.counted_child_state_plan.parent_unique_parameters
            ),
            child_parameter_count=(
                self.request.counted_child_state_plan.target_total_parameter_elements
            ),
            parent_storage_dtype=self.bindings.parent_storage_dtype,
            child_storage_dtype=self.bindings.child_storage_dtype,
            layer_bindings=tuple(bindings),
            focused_writer_layer_indices=(0,),
            basis_split_sha256=basis_split_sha256,
            implementation_sha256=(
                lfm2_moving_gauge_boundary_implementation_sha256()
            ),
            provenance_sha256s=provenance,
            checkpoint_hmac_key_id=self.bindings.boundary_phase_authority.key_id,
            shard_batch_count=self.bindings.boundary_shard_batch_count,
        )

        def source_provider() -> Lfm2MovingGaugeBoundaryExtractionSources:
            return Lfm2MovingGaugeBoundaryExtractionSources(
                parent_cache=self.parent_cache,
                child_cache=self.child_cache,
                trace_adapter=self.bindings.trace_adapter,
                basis_phase=self.basis_phase,
                basis_phase_authority=self.bindings.basis_phase_authority,
                basis_phase_cache=self.basis_cache,
                counted_child_state_plan=self.request.counted_child_state_plan,
                v3_first_divergence_request=self.v3_request,
                v3_first_divergence_ledger=self.v3_ledger,
                parent_session_factory=(
                    self.parent_cache.selective_authenticated_session
                ),
                child_session_factory=(
                    self.child_cache.selective_authenticated_session
                ),
            )

        boundary = load_or_extract_lfm2_moving_gauge_boundary_phase(
            DynamicFoldPhaseCache(self.request.local_paths["phase_cache"]),
            request,
            authority=self.bindings.boundary_phase_authority,
            checkpoint_root=(
                Path(self.request.local_paths["checkpoint_root"])
                / "boundary-import"
            ),
            source_provider=source_provider,
        )
        if not isinstance(boundary, AuthenticatedLfm2MovingGaugeBoundaryPhase):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Boundary import stopped before complete authenticated publication."
            )
        self.boundary_request = request
        self.boundary = boundary
        return V5LivePhaseExecution(
            artifact=boundary,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "authenticated_boundary_rows",
                "boundary_request_sha256": request.request_sha256,
                "boundary_phase_key_sha256": request.phase_key.phase_key_sha256,
                "boundary_completion_sha256": boundary.entry.completion_sha256,
                "boundary_receipt_sha256": boundary.receipt_sha256,
                "row_index_sha256": boundary.row_index["row_index_sha256"],
                "row_alignment_sha256": boundary.row_alignment_sha256,
                "split_identity_sha256": boundary.split_identity_sha256,
                "logical_layer_indices": [0, 1],
                "physical_layer_indices": [0, v5.TERMINAL_LAYER_INDEX],
                "cache_hit_before_source_provider": True,
                "selective_sessions_deferred_until_miss": True,
                "solver_visible_splits": [],
            },
        )

    def _phase_import_feature_rows(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        provenance = self.bindings.provenance_sha256s
        child_contract = self.child_cache.manifest["layer_topology"]["layers"][0][
            "contract"
        ]
        feature_bindings = derive_lfm2_moving_gauge_writer_feature_bindings(
            child_contract,
            source_dtype_by_field=self.bindings.child_feature_storage_dtypes,
        )
        request = Lfm2MovingGaugeFeatureImportRequest(
            child_cache_key=self.child_cache.cache_key,
            child_manifest_sha256=provenance["child_manifest"],
            child_completion_sha256=provenance["child_completion"],
            child_identity_sha256=provenance["child_identity"],
            child_state_dict_sha256=provenance["child_state_dict"],
            child_topology_sha256=provenance["child_topology"],
            child_selective_session_sha256=provenance[
                "child_selective_session"
            ],
            child_layer_contract_sha256=canonical_sha256(child_contract),
            boundary_phase_key_sha256=(
                self.boundary.entry.phase_key.phase_key_sha256
            ),
            boundary_completion_sha256=self.boundary.entry.completion_sha256,
            boundary_request_sha256=self.boundary.request.request_sha256,
            boundary_receipt_sha256=self.boundary.receipt_sha256,
            boundary_row_index_sha256=self.boundary.row_index[
                "row_index_sha256"
            ],
            row_alignment_sha256=self.boundary.row_alignment_sha256,
            split_identity_sha256=self.boundary.split_identity_sha256,
            expected_batch_count=v5.EXPECTED_BATCH_COUNT,
            expected_aligned_row_count=v5.EXPECTED_ALIGNED_ROW_COUNT,
            logical_layer_index=0,
            child_layer_index=0,
            feature_bindings=feature_bindings,
            implementation_sha256=lfm2_moving_gauge_feature_implementation_sha256(),
            checkpoint_hmac_key_id=self.bindings.feature_phase_authority.key_id,
            shard_batch_count=self.bindings.feature_shard_batch_count,
        )

        def source_provider() -> Lfm2MovingGaugeFeatureExtractionSources:
            return Lfm2MovingGaugeFeatureExtractionSources(
                boundary_phase=self.boundary,
                child_cache=self.child_cache,
                child_session_factory=(
                    self.child_cache.selective_authenticated_session
                ),
            )

        feature = load_or_extract_lfm2_moving_gauge_feature_phase(
            DynamicFoldPhaseCache(self.request.local_paths["phase_cache"]),
            request,
            authority=self.bindings.feature_phase_authority,
            checkpoint_root=(
                Path(self.request.local_paths["checkpoint_root"])
                / "feature-import"
            ),
            source_provider=source_provider,
        )
        if not isinstance(feature, AuthenticatedLfm2MovingGaugeFeaturePhase):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Feature import stopped before complete authenticated publication."
            )
        self.feature_request = request
        self.feature = feature
        widths = {
            name: int(feature.feature_rows(name, split="fit").feature_width)
            for name in v5.FOCUSED_FEATURE_FIELDS
        }
        if widths["short_conv.context"] != v5.SOURCE_WIDTH:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Short-conv context width differs from its physical writer."
            )
        self.baseline_feature_widths = widths
        return V5LivePhaseExecution(
            artifact=feature,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "authenticated_variable_writer_features",
                "feature_request_sha256": request.request_sha256,
                "feature_phase_key_sha256": request.phase_key.phase_key_sha256,
                "feature_completion_sha256": feature.entry.completion_sha256,
                "feature_receipt_sha256": feature.receipt_sha256,
                "feature_widths": widths,
                "feature_fields": list(v5.FOCUSED_FEATURE_FIELDS),
                "parent_child_dtype_equality_required": False,
                "cache_hit_before_source_provider": True,
                "selective_session_deferred_until_miss": True,
                "solver_visible_splits": [],
            },
        )

    # ------------------------------------------------------------------
    # Phase 3/4: FIT-only charts and exact causal targets/closed-form solves.
    # ------------------------------------------------------------------
    def _fit_evidence(
        self,
        *,
        logical_layer: int,
        endpoint: str,
        boundary_id: str,
        role: str,
        locked: bool,
    ) -> Lfm2MovingGaugeBoundaryEvidence:
        pair = self.boundary.boundary_pair(
            logical_layer,
            endpoint,
            split="fit",
        )
        evidence = Lfm2MovingGaugeBoundaryEvidence(
            boundary_id=boundary_id,
            interface_role=role,
            interface_locked=locked,
            base_encoder=self.basis_phase.basis_result.basis,
            fit_parent_rows=pair.parent_rows,
            fit_child_rows=pair.child_rows,
            fit_weights=pair.weights,
            fit_provenance=(
                f"{self.boundary.receipt_sha256}:fit:{logical_layer}:{endpoint}"
            ),
            row_alignment_sha256=pair.row_alignment_sha256,
            cached_parent_evidence_sha256=(
                self.boundary.cached_parent_evidence_sha256
            ),
            live_child_recapture_sha256=(
                self.boundary.live_child_recapture_sha256
            ),
        )
        if evidence.heldout_parent_rows is not None:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "FIT chart evidence accidentally contains heldout rows."
            )
        return evidence

    def _phase_freeze_fit_charts(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        evidence = {
            "pre": self._fit_evidence(
                logical_layer=0,
                endpoint="pre",
                boundary_id="layer.000.residual.pre",
                role="input",
                locked=True,
            ),
            "post_operator": self._fit_evidence(
                logical_layer=0,
                endpoint="post_operator",
                boundary_id="layer.000.residual.post_operator",
                role="intermediate",
                locked=False,
            ),
            "post_layer": self._fit_evidence(
                logical_layer=0,
                endpoint="post_layer",
                boundary_id="layer.000.residual.post_layer",
                role="intermediate",
                locked=False,
            ),
            "terminal": self._fit_evidence(
                logical_layer=1,
                endpoint="post_layer",
                boundary_id="stack.terminal.residual",
                role="terminal",
                locked=True,
            ),
        }
        orthogonal = {
            name: fit_lfm2_cycle_coupled_moving_gauge_boundary(
                item,
                evaluate_heldout=False,
            )
            for name, item in evidence.items()
        }
        generalized = {
            name: fit_lfm2_generalized_moving_gauge_chart_diagnostic(
                evidence[name],
                orthogonal_chart=orthogonal[name],
            )
            for name in ("post_operator", "post_layer")
        }
        locked_charts = {
            name: bind_lfm2_orthogonal_causal_endpoint_chart(evidence[name])
            for name in ("pre", "terminal")
        }
        charts = {
            "E0": {
                **locked_charts,
                "post_operator": bind_lfm2_base_e0_causal_endpoint_chart(
                    evidence["post_operator"]
                ),
                "post_layer": bind_lfm2_base_e0_causal_endpoint_chart(
                    evidence["post_layer"]
                ),
            },
            "E0R": {
                **locked_charts,
                "post_operator": bind_lfm2_orthogonal_causal_endpoint_chart(
                    evidence["post_operator"]
                ),
                "post_layer": bind_lfm2_orthogonal_causal_endpoint_chart(
                    evidence["post_layer"]
                ),
            },
            "E0A": {
                **locked_charts,
                "post_operator": bind_lfm2_generalized_e0a_causal_endpoint_chart(
                    generalized["post_operator"],
                    evidence["post_operator"],
                ),
                "post_layer": bind_lfm2_generalized_e0a_causal_endpoint_chart(
                    generalized["post_layer"],
                    evidence["post_layer"],
                ),
            },
        }
        bundle_sha256s = {
            mode: _chart_bundle_sha256(value) for mode, value in charts.items()
        }
        if len(set(bundle_sha256s.values())) != len(v5.CHART_MODES):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "E0, E0R, and E0A did not freeze as distinct chart bundles."
            )
        self.fit_evidence = evidence
        self.fit_charts = charts
        self.chart_bundle_sha256s = bundle_sha256s
        artifact = {
            "evidence": evidence,
            "orthogonal": orthogonal,
            "generalized": generalized,
            "charts": charts,
        }
        return V5LivePhaseExecution(
            artifact=artifact,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "fit_frozen_causal_endpoint_charts",
                "chart_bundle_sha256s": bundle_sha256s,
                "chart_modes": list(v5.CHART_MODES),
                "fit_only": True,
                "heldout_rows_read": False,
                "base_e0_is_literal_basis_chart": True,
                "e0r_is_right_orthogonal_chart": True,
                "e0a_is_promoted_generalized_chart": True,
            },
        )

    def _baseline_rows(self, split: str) -> dict[str, Any]:
        pre = self.boundary.boundary_pair(0, "pre", split=split)
        post_operator = self.boundary.boundary_pair(
            0,
            "post_operator",
            split=split,
        )
        post_layer = self.boundary.boundary_pair(0, "post_layer", split=split)
        writers = self.boundary.focused_writer_outputs(0, split=split)
        context = self.feature.feature_rows("short_conv.context", split=split)
        post_swiglu = self.feature.feature_rows("mlp.post_swiglu", split=split)
        expected_indices = pre.global_row_indices
        for value in (post_operator, post_layer):
            if value.global_row_indices != expected_indices:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Boundary rows lost their authenticated alignment."
                )
        if (
            writers.global_row_indices != expected_indices
            or context.global_row_indices != expected_indices
            or post_swiglu.global_row_indices != expected_indices
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Writer features and residual rows are not aligned."
            )
        return {
            "pre": pre,
            "post_operator": post_operator,
            "post_layer": post_layer,
            "writers": writers,
            "context": context.rows,
            "post_swiglu": post_swiglu.rows,
            "weights": pre.weights,
        }

    def _causal_plan(self, mode: str) -> Lfm2MovingGaugeCausalBoundaryPlan:
        charts = self.fit_charts[mode]
        topology = self.child_cache.manifest["layer_topology"]
        plan = Lfm2MovingGaugeCausalBoundaryPlan(
            layer_index=0,
            layer_count=v5.LAYER_COUNT,
            operator_kind=topology["layers"][0]["contract"]["kind"],
            pre_chart=charts["pre"],
            post_operator_chart=charts["post_operator"],
            post_layer_chart=charts["post_layer"],
            terminal_chart=charts["terminal"],
            pre_evidence=self.fit_evidence["pre"],
            post_operator_evidence=self.fit_evidence["post_operator"],
            post_layer_evidence=self.fit_evidence["post_layer"],
            terminal_evidence=self.fit_evidence["terminal"],
            parent_cache_sha256=self.boundary.cached_parent_evidence_sha256,
            child_state_sha256=self.child_fingerprint.identity_sha256,
            child_recapture_sha256=self.boundary.live_child_recapture_sha256,
            row_alignment_sha256=self.boundary.row_alignment_sha256,
            split_identity_sha256=self.boundary.split_identity_sha256,
            layer_topology=topology,
        )
        return plan

    def _phase_build_baseline_target_arms(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        rows = self._baseline_rows("fit")
        operator_seed = _named_parameter(
            self.child,
            v5.REPLACEMENT_TENSOR_NAMES[0],
        ).detach()
        mlp_seed = _named_parameter(
            self.child,
            v5.REPLACEMENT_TENSOR_NAMES[1],
        ).detach()
        by_chart: dict[str, Any] = {}
        for mode in v5.CHART_MODES:
            plan = self._causal_plan(mode)
            joint_sources = {
                "actual_child_post_layer": rows["post_layer"].child_rows,
                "actual_child_pre": rows["pre"].child_rows,
                "existing_combined_writer_output": (
                    rows["writers"].child_combined_writer_output
                ),
                "parent_post_layer": rows["post_layer"].parent_rows,
            }
            joint_receipt = build_lfm2_moving_gauge_causal_state_receipt(
                plan,
                kind="baseline_current_child",
                tensors=joint_sources,
            )
            joint_target = build_lfm2_joint_writer_target(
                plan,
                parent_post_layer=rows["post_layer"].parent_rows,
                actual_child_pre=rows["pre"].child_rows,
                actual_child_post_layer=rows["post_layer"].child_rows,
                existing_combined_writer_output=(
                    rows["writers"].child_combined_writer_output
                ),
                state_receipt=joint_receipt,
            )
            combined_design = torch.cat(
                (rows["context"], rows["post_swiglu"]),
                dim=1,
            )
            combined_seed = torch.cat((operator_seed, mlp_seed), dim=1)
            combined_solution, combined_report = (
                solve_lfm2_moving_gauge_anchored_writer_replacement(
                    combined_design,
                    joint_target.target_combined_writer_output,
                    combined_seed,
                    rows["weights"],
                )
            )
            operator_width = int(operator_seed.shape[1])
            joint_operator = combined_solution[:, :operator_width].contiguous()
            joint_mlp = combined_solution[:, operator_width:].contiguous()
            joint_split_receipts = _joint_split_solution_receipts(
                joint_operator,
                joint_mlp,
                combined_report,
            )
            operator_sources = {
                "actual_child_post_operator": rows["post_operator"].child_rows,
                "actual_child_pre": rows["pre"].child_rows,
                "existing_operator_writer_output": (
                    rows["writers"].child_operator_output
                ),
                "parent_post_operator": rows["post_operator"].parent_rows,
            }
            operator_receipt = build_lfm2_moving_gauge_causal_state_receipt(
                plan,
                kind="baseline_current_child",
                tensors=operator_sources,
            )
            stale_mlp_sources = {
                "existing_mlp_writer_output": rows["writers"].child_mlp_down,
                "fresh_child_post_layer": rows["post_layer"].child_rows,
                "fresh_child_post_operator": rows["post_operator"].child_rows,
                "parent_post_layer": rows["post_layer"].parent_rows,
            }
            stale_mlp_receipt = build_lfm2_moving_gauge_causal_state_receipt(
                plan,
                kind="baseline_current_child",
                tensors=stale_mlp_sources,
            )
            stale_target = build_lfm2_sequential_writer_target(
                plan,
                parent_post_operator=rows["post_operator"].parent_rows,
                actual_child_pre=rows["pre"].child_rows,
                actual_child_post_operator=rows["post_operator"].child_rows,
                existing_operator_writer_output=(
                    rows["writers"].child_operator_output
                ),
                parent_post_layer=rows["post_layer"].parent_rows,
                fresh_child_post_operator=rows["post_operator"].child_rows,
                fresh_child_post_layer=rows["post_layer"].child_rows,
                existing_mlp_writer_output=rows["writers"].child_mlp_down,
                operator_source_receipt=operator_receipt,
                mlp_source_receipt=stale_mlp_receipt,
            )
            operator_solution, operator_report = (
                solve_lfm2_moving_gauge_anchored_writer_replacement(
                    rows["context"],
                    stale_target.target_operator_writer_output,
                    operator_seed,
                    rows["weights"],
                )
            )
            stale_mlp_solution, stale_mlp_report = (
                solve_lfm2_moving_gauge_anchored_writer_replacement(
                    rows["post_swiglu"],
                    stale_target.target_mlp_writer_output,
                    mlp_seed,
                    rows["weights"],
                )
            )
            by_chart[mode] = {
                "plan": plan,
                "joint": {
                    "target": joint_target,
                    "operator_solution": joint_operator,
                    "mlp_solution": joint_mlp,
                    "solve_report": combined_report,
                    "split_receipt_sha256s": joint_split_receipts,
                },
                "sequential_no_recapture": {
                    "target": stale_target,
                    "operator_solution": operator_solution,
                    "operator_report": operator_report,
                    "mlp_solution": stale_mlp_solution,
                    "mlp_report": stale_mlp_report,
                    "operator_source_receipt": operator_receipt,
                },
            }
        self.baseline_fit_rows = rows
        self.baseline_targets = by_chart
        evidence = {
            mode: {
                "plan_sha256": value["plan"].report["result_sha256"],
                "joint_target_sha256": value["joint"]["target"].report[
                    "result_sha256"
                ],
                "joint_solution_sha256": value["joint"]["solve_report"][
                    "report_sha256"
                ],
                "joint_split_receipt_sha256s": dict(
                    value["joint"]["split_receipt_sha256s"]
                ),
                "operator_solution_sha256": _solution_sha256(
                    value["sequential_no_recapture"]["operator_solution"],
                    value["sequential_no_recapture"]["operator_report"],
                ),
                "stale_mlp_solution_sha256": _solution_sha256(
                    value["sequential_no_recapture"]["mlp_solution"],
                    value["sequential_no_recapture"]["mlp_report"],
                ),
            }
            for mode, value in by_chart.items()
        }
        return V5LivePhaseExecution(
            artifact=by_chart,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "fit_only_baseline_causal_targets_and_solutions",
                "charts": evidence,
                "joint_formula_api_used": True,
                "sequential_formula_api_used": True,
                "closed_form_solver_api_used": True,
                "stale_sequential_arm_is_negative_control": True,
                "training_steps": 0,
                "heldout_rows_read": False,
            },
        )

    # ------------------------------------------------------------------
    # Phase 5: deferred authenticated conditioned recapture.  The cache decides
    # hit/miss before invoking the temporary-install source provider.
    # ------------------------------------------------------------------
    def _capture_replay_range(
        self,
        *,
        session: Any,
        operator_solution: torch.Tensor,
        batch_start: int,
        batch_stop: int,
    ) -> Any:
        if not (
            isinstance(batch_start, int)
            and isinstance(batch_stop, int)
            and 0 <= batch_start < batch_stop <= v5.EXPECTED_BATCH_COUNT
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Conditioned cache requested a noncanonical replay range."
            )
        if (
            getattr(session, "evidence_sha256", None)
            != self.bindings.provenance_sha256s["child_selective_session"]
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Child selective session differs from provenance."
            )
        replay_batches: list[dict[str, torch.Tensor]] = []
        for ordinal in range(batch_start, batch_stop):
            reference = session.batch_references[ordinal]
            names = tuple(
                name
                for name in reference.ordered_tensor_names
                if name.startswith("input.")
            )
            payload = session.load_selected_tensors(ordinal, names)
            try:
                replay_batches.append(
                    {
                        name.removeprefix("input."): tensor.detach().clone()
                        for name, tensor in payload.tensors.items()
                    }
                )
            finally:
                session.release_selected_tensors(payload)

        def capture() -> Any:
            profile = Lfm2ChildEvidenceProfile(
                profile_role=CONDITIONED_RECAPTURE_PROFILE_ROLE,
                layer_suffixes=((0, tuple(sorted(CONDITIONED_RECLOSURE_FIELDS))),),
                retain_logits=False,
                retain_lm_head_input=False,
                retain_phase_forced_token_ids=False,
            )
            config = Lfm2CircuitTraceCaptureConfig(
                decode_trace_mode="full_prefix_canonical",
                capture_causal_circuit_evidence=False,
            )
            with torch.inference_mode():
                replay = capture_lfm2_child_replay(
                    self.child,
                    replay_batches,
                    selected_layer_indices=(0,),
                    config=config,
                    input_device=torch.device(self.v3_request["runtime_device"]),
                    replay_capture_mode="prefix_by_prefix_canonical",
                    child_evidence_profile=profile,
                    replay_devices=self.request.replay_devices,
                )
            if _tensor_tree_requires_grad(replay):
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Conditioned child replay retained autograd state."
                )
            return replay

        return self._with_temporary_operator(operator_solution, capture)

    def _with_temporary_operator(
        self,
        replacement: torch.Tensor,
        callback: Callable[[], Any],
    ) -> Any:
        """Install one operator writer and restore child plus route in ``finally``."""

        parameter = _named_parameter(self.child, v5.REPLACEMENT_TENSOR_NAMES[0])
        original = parameter.detach().to(device="cpu").contiguous().clone()
        child_before = fingerprint_lfm2_parent(self.child).state_dict_sha256
        route_before = fingerprint_lfm2_carrier_runtime_owned_tensor_state(
            self.routing
        )["state_sha256"]
        callback_error: BaseException | None = None
        result: Any = None
        try:
            with torch.no_grad():
                parameter.copy_(
                    replacement.detach().to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
            with torch.inference_mode():
                result = callback()
        except BaseException as exc:
            callback_error = exc
        finally:
            with torch.no_grad():
                parameter.copy_(
                    original.to(device=parameter.device, dtype=parameter.dtype)
                )
            child_after = fingerprint_lfm2_parent(self.child).state_dict_sha256
            route_after = fingerprint_lfm2_carrier_runtime_owned_tensor_state(
                self.routing
            )["state_sha256"]
        if child_after != child_before or route_after != route_before:
            restoration_error = Lfm2Exact4xMovingGaugeV5LiveError(
                "Temporary operator intervention failed exact child/route restoration."
            )
            if callback_error is not None:
                restoration_error.add_note(
                    f"Capture also failed: {type(callback_error).__name__}: "
                    f"{callback_error}"
                )
            raise restoration_error
        self._mutation_observations.append(
            {
                "child_before": child_before,
                "child_after": child_after,
                "route_before": route_before,
                "route_after": route_after,
            }
        )
        if callback_error is not None:
            raise callback_error
        return result

    def _conditioned_phase_request(
        self,
        *,
        module: Any,
        chart_mode: str,
        operator_solution: torch.Tensor,
        operator_report: Mapping[str, Any],
    ) -> tuple[Any, str]:
        config = Lfm2CircuitTraceCaptureConfig(
            decode_trace_mode="full_prefix_canonical",
            capture_causal_circuit_evidence=False,
        )
        replay_implementation = implementation_identity()
        numeric_policy = compile_lfm2_child_replay_numeric_policy(
            self.child,
            input_device=torch.device(self.v3_request["runtime_device"]),
        )
        execution = self.child_cache.manifest["execution"]
        source_inputs = tuple(
            self.child_cache.manifest["corpus"]["ordered_batch_input_sha256"]
        )
        record_ids_by_batch: dict[int, str] = {}
        for row in self.boundary.row_index["rows"]:
            ordinal = int(row["record_ordinal"])
            record_id = str(row["record_id"])
            previous = record_ids_by_batch.setdefault(ordinal, record_id)
            if previous != record_id:
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Boundary row index assigns multiple records to one batch."
                )
        if set(record_ids_by_batch) != set(range(v5.EXPECTED_BATCH_COUNT)):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Boundary row index does not identify all conditioned batches."
            )
        source_record_ids = tuple(
            record_ids_by_batch[index] for index in range(v5.EXPECTED_BATCH_COUNT)
        )
        split_identity = module.derive_lfm2_conditioned_recapture_split_identity(
            source_input_session_sha256=self.bindings.provenance_sha256s[
                "child_selective_session"
            ],
            source_input_sha256s=source_inputs,
            source_record_ids=source_record_ids,
            basis_split_by_batch=tuple(
                self.bindings.basis_split_by_batch[index]
                for index in range(v5.EXPECTED_BATCH_COUNT)
            ),
        )
        conditioned_fingerprint = _fingerprint_with_parameter_replacement(
            self.child,
            parameter_name=v5.REPLACEMENT_TENSOR_NAMES[0],
            replacement=operator_solution,
        )
        descriptor = {
            "dtype": str(operator_solution.dtype).removeprefix("torch."),
            "shape": [int(size) for size in operator_solution.shape],
            "numel": int(operator_solution.numel()),
            "size_bytes": int(
                operator_solution.numel() * operator_solution.element_size()
            ),
            "sha256": fingerprint_lfm2_moving_gauge_tensor(operator_solution),
        }
        endpoint_chain = self.baseline_targets[chart_mode]["plan"].report[
            "result_sha256"
        ]
        operator_solution_sha256 = _solution_sha256(
            operator_solution,
            operator_report,
        )
        frozen_state_receipt = (
            module.derive_lfm2_frozen_conditioned_child_state_receipt_sha256(
                base_child_identity_sha256=(
                    self.child_fingerprint.identity_sha256
                ),
                base_child_state_dict_sha256=(
                    self.child_fingerprint.state_dict_sha256
                ),
                endpoint_chain_receipt_sha256=endpoint_chain,
                operator_replacement_tensor_descriptor=descriptor,
                operator_solution_sha256=operator_solution_sha256,
                conditioned_child_identity_sha256=conditioned_fingerprint[
                    "identity_sha256"
                ],
                conditioned_child_state_dict_sha256=conditioned_fingerprint[
                    "state_dict_sha256"
                ],
            )
        )
        widths = {
            "short_conv.context": int(operator_solution.shape[1]),
            "operator.output": v5.TARGET_WIDTH,
            "residual.pre": v5.TARGET_WIDTH,
            "residual.post_operator": v5.TARGET_WIDTH,
            "residual.post_layer": v5.TARGET_WIDTH,
            "mlp.post_swiglu": int(
                _named_parameter(
                    self.child,
                    v5.REPLACEMENT_TENSOR_NAMES[1],
                ).shape[1]
            ),
            "mlp.down": v5.TARGET_WIDTH,
        }
        dtypes = {
            name: self.bindings.child_storage_dtype
            for name in module.LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS
        }
        dtypes.update(self.bindings.child_feature_storage_dtypes)
        field_contracts = tuple(
            module.Lfm2ConditionedRecaptureFieldContract(
                field_name=name,
                source_dtype=dtypes[name],
                feature_width=widths[name],
            )
            for name in module.LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS
        )
        floating_dtypes = tuple(
            sorted(
                numeric_policy["numeric_execution_policy"]["target_inventory"][
                    "floating_dtype_counts"
                ]
            )
        )
        if (
            self.v3_request["runtime_dtype"] != "float32"
            or floating_dtypes != ("float32",)
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Conditioned recapture requires the authenticated FP32 child "
                "runtime and numeric inventory."
            )
        request = module.Lfm2ConditionedRecapturePhaseRequest(
            v5_request_sha256=self.request.request_sha256,
            v5_lineage_sha256=canonical_sha256(
                dict(self.request.lineage_sha256s)
            ),
            base_child_identity_sha256=self.child_fingerprint.identity_sha256,
            base_child_state_dict_sha256=(
                self.child_fingerprint.state_dict_sha256
            ),
            endpoint_chain_receipt_sha256=endpoint_chain,
            operator_replacement_tensor_descriptor=descriptor,
            operator_solution_sha256=operator_solution_sha256,
            frozen_conditioned_child_state_receipt_sha256=frozen_state_receipt,
            conditioned_child_identity_sha256=conditioned_fingerprint[
                "identity_sha256"
            ],
            conditioned_child_state_dict_sha256=conditioned_fingerprint[
                "state_dict_sha256"
            ],
            child_layer_topology_sha256=self.bindings.provenance_sha256s[
                "child_topology"
            ],
            chart_mode=chart_mode,
            child_replay_config_sha256=config.as_dict()["selection_sha256"],
            child_replay_implementation_sha256=replay_implementation[
                "implementation_sha256"
            ],
            child_replay_execution_sha256=execution["execution_sha256"],
            child_replay_numeric_policy_sha256=numeric_policy[
                "replay_numeric_policy_sha256"
            ],
            child_replay_capture_mode="prefix_by_prefix_canonical",
            source_input_session_sha256=self.bindings.provenance_sha256s[
                "child_selective_session"
            ],
            source_input_sha256s=source_inputs,
            source_record_ids=source_record_ids,
            basis_split_by_batch=tuple(
                self.bindings.basis_split_by_batch[index]
                for index in range(v5.EXPECTED_BATCH_COUNT)
            ),
            split_identity_sha256=split_identity,
            field_contracts=field_contracts,
            runtime_device_type=torch.device(
                self.v3_request["runtime_device"]
            ).type,
            runtime_floating_dtypes=floating_dtypes,
            conditioned_recapture_spec_sha256=(
                module.lfm2_conditioned_recapture_spec_sha256()
            ),
            implementation_sha256=(
                module.lfm2_conditioned_recapture_implementation_sha256()
            ),
            checkpoint_hmac_key_id=(
                self.bindings.conditioned_recapture_authority.key_id
            ),
            checkpoint_batch_count=(
                self.request.recapture_checkpoint_batch_count
            ),
        )
        self.conditioned_replay_config = config
        self.conditioned_replay_numeric_policy = numeric_policy
        return request, frozen_state_receipt

    def _phase_checkpoint_conditioned_child_recapture(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        module = importlib.import_module(
            "neural_forge.hf_lfm2_moving_gauge_"
            "conditioned_recapture_phase_cache"
        )
        if (
            module.LFM2_CONDITIONED_RECAPTURE_PROFILE_ROLE
            != CONDITIONED_RECAPTURE_PROFILE_ROLE
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Conditioned replay profile role differs from the cache contract."
            )
        loader = self.bindings.conditioned_recapture_loader
        if loader is None:
            loader, implementation_sha256 = _resolve_conditioned_recapture_loader()
        else:
            implementation_sha256 = (
                module.lfm2_conditioned_recapture_implementation_sha256()
            )
        self._mutation_observations.clear()
        recaptures: dict[str, Any] = {}
        total_forwards = 0
        full_hits = 0
        requests: dict[str, Any] = {}
        child_session: Any | None = None
        selective_session_open_count = 0
        session_stack = ExitStack()

        def session_for_miss() -> Any:
            nonlocal child_session, selective_session_open_count
            if child_session is None:
                child_session = session_stack.enter_context(
                    self.child_cache.selective_authenticated_session()
                )
                selective_session_open_count += 1
                if (
                    child_session.evidence_sha256
                    != self.bindings.provenance_sha256s[
                        "child_selective_session"
                    ]
                ):
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Child selective session differs from provenance."
                    )
            return child_session

        try:
            for mode in v5.CHART_MODES:
                sequential = self.baseline_targets[mode][
                    "sequential_no_recapture"
                ]
                operator_solution = sequential["operator_solution"]
                conditioned_request, _frozen_state_receipt = (
                    self._conditioned_phase_request(
                        module=module,
                        chart_mode=mode,
                        operator_solution=operator_solution,
                        operator_report=sequential["operator_report"],
                    )
                )
                if not isinstance(
                    conditioned_request,
                    module.Lfm2ConditionedRecapturePhaseRequest,
                ):
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Conditioned request construction returned an untyped value."
                    )
                requests[mode] = conditioned_request
                chart_forward_count = 0
                provider_calls = 0

                def source_provider(
                    batch_start: int,
                    batch_stop: int,
                    solution: torch.Tensor = operator_solution,
                ) -> Any:
                    nonlocal chart_forward_count, provider_calls
                    replay = self._capture_replay_range(
                        session=session_for_miss(),
                        operator_solution=solution,
                        batch_start=batch_start,
                        batch_stop=batch_stop,
                    )
                    provider_calls += 1
                    chart_forward_count += int(
                        replay.metadata["forward_counts"]["child"]
                    )
                    return replay

                artifact = loader(
                    DynamicFoldPhaseCache(
                        self.request.local_paths["phase_cache"]
                    ),
                    conditioned_request,
                    authority=self.bindings.conditioned_recapture_authority,
                    checkpoint_root=Path(
                        self.request.local_paths["checkpoint_root"]
                    )
                    / self.request.request_sha256
                    / "conditioned-child-recapture"
                    / mode,
                    range_provider=source_provider,
                )
                if not isinstance(
                    artifact,
                    module.AuthenticatedLfm2ConditionedRecapturePhase,
                ):
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Conditioned recapture did not return a complete "
                        "authenticated phase artifact."
                    )
                if not artifact.verify_authentication():
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Conditioned recapture artifact authentication failed."
                    )
                receipt_sha256 = _digest_from(
                    artifact,
                    "receipt_sha256",
                    "result_sha256",
                    "recapture_sha256",
                )
                for field_name in (
                    "residual.post_operator",
                    "residual.post_layer",
                    "mlp.down",
                    "mlp.post_swiglu",
                ):
                    _conditioned_rows(
                        artifact,
                        split="fit",
                        field_name=field_name,
                    )
                forwards = chart_forward_count
                cache_hit = provider_calls == 0
                if cache_hit and forwards != 0:
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Authenticated recapture hit claims child forwards."
                    )
                if not cache_hit and forwards <= 0:
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Cold/partial conditioned recapture did not prove child "
                        "forwards."
                    )
                full_hits += int(cache_hit)
                total_forwards += forwards
                recaptures[mode] = artifact
                if receipt_sha256 == self.boundary.live_child_recapture_sha256:
                    raise Lfm2Exact4xMovingGaugeV5LiveError(
                        "Conditioned recapture reused the baseline child evidence "
                        "digest."
                    )
        finally:
            session_stack.close()
        any_mutation = bool(self._mutation_observations)
        all_hit = full_hits == len(v5.CHART_MODES)
        if selective_session_open_count != (0 if all_hit else 1):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Conditioned cache hit/miss opened an invalid number of child "
                "selective sessions."
            )
        if all_hit is any_mutation or (all_hit and total_forwards != 0):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Conditioned cache hit/miss and mutation evidence conflict."
            )
        if not all_hit and (not any_mutation or total_forwards <= 0):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Conditioned cache miss did not execute a restorable child replay."
            )
        before = (
            None
            if all_hit
            else self._mutation_observations[0]["child_before"]
        )
        after = (
            None
            if all_hit
            else self._mutation_observations[-1]["child_after"]
        )
        if not all_hit and any(
            row["child_before"] != before
            or row["child_after"] != before
            or row["route_before"] != row["route_after"]
            for row in self._mutation_observations
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "One conditioned shard changed child/route restoration identity."
            )
        self.conditioned_recaptures = recaptures
        self.conditioned_requests = requests
        self.conditioned_cache_implementation_sha256 = implementation_sha256
        return V5LivePhaseExecution(
            artifact=recaptures,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "operator_conditioned_child_recapture",
                "cache_implementation_sha256": implementation_sha256,
                "causal_endpoint_chain_sha256s": {
                    mode: self.baseline_targets[mode]["plan"].report[
                        "result_sha256"
                    ]
                    for mode in v5.CHART_MODES
                },
                "recapture_receipt_sha256s": {
                    mode: _digest_from(
                        artifact,
                        "receipt_sha256",
                        "result_sha256",
                        "recapture_sha256",
                    )
                    for mode, artifact in recaptures.items()
                },
                "operator_solution_sha256s": {
                    mode: _solution_sha256(
                        self.baseline_targets[mode][
                            "sequential_no_recapture"
                        ]["operator_solution"],
                        self.baseline_targets[mode][
                            "sequential_no_recapture"
                        ]["operator_report"],
                    )
                    for mode in v5.CHART_MODES
                },
                "all_256_batches_transport_captured": True,
                "transport_contains_fit_and_heldout": True,
                "conditioned_request_sha256s": {
                    mode: request.request_sha256
                    for mode, request in requests.items()
                },
                "frozen_conditioned_child_state_receipt_sha256s": {
                    mode: request.frozen_conditioned_child_state_receipt_sha256
                    for mode, request in requests.items()
                },
                "heldout_solver_rows_read": False,
                "fresh_post_swiglu_present": True,
                "torch_inference_mode": True,
                "requires_grad_outputs": False,
                "optimizer_activity": False,
                "temporary_mutation_count": len(self._mutation_observations),
                "child_selective_session_open_count": (
                    selective_session_open_count
                ),
                "route_state_restored": not all_hit,
            },
            child_forward_count=total_forwards,
            temporary_child_mutation_performed=not all_hit,
            child_state_restored=not all_hit,
            child_state_before_sha256=before,
            child_state_after_sha256=after,
            authenticated_checkpoint_reuse=all_hit,
        )

    # ------------------------------------------------------------------
    # Phase 6: consume only conditioned FIT rows, including fresh post-SwiGLU.
    # ------------------------------------------------------------------
    def _phase_freeze_conditioned_targets_and_solutions(
        self,
        _phase: Mapping[str, Any],
        _completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        rows = self.baseline_fit_rows
        mlp_seed = _named_parameter(
            self.child,
            v5.REPLACEMENT_TENSOR_NAMES[1],
        ).detach()
        conditioned: dict[str, Any] = {}
        for mode in v5.CHART_MODES:
            baseline = self.baseline_targets[mode]["sequential_no_recapture"]
            plan = self.baseline_targets[mode]["plan"]
            recapture = self.conditioned_recaptures[mode]
            fresh_post_operator = _conditioned_rows(
                recapture,
                split="fit",
                field_name="residual.post_operator",
            )
            fresh_post_layer = _conditioned_rows(
                recapture,
                split="fit",
                field_name="residual.post_layer",
            )
            fresh_mlp_down = _conditioned_rows(
                recapture,
                split="fit",
                field_name="mlp.down",
            )
            fresh_post_swiglu = _conditioned_rows(
                recapture,
                split="fit",
                field_name="mlp.post_swiglu",
            )
            recapture_sha256 = _digest_from(
                recapture,
                "receipt_sha256",
                "result_sha256",
                "recapture_sha256",
            )
            operator_solution_sha256 = _solution_sha256(
                baseline["operator_solution"],
                baseline["operator_report"],
            )
            mlp_sources = {
                "existing_mlp_writer_output": fresh_mlp_down,
                "fresh_child_post_layer": fresh_post_layer,
                "fresh_child_post_operator": fresh_post_operator,
                "parent_post_layer": rows["post_layer"].parent_rows,
            }
            mlp_receipt = build_lfm2_moving_gauge_causal_state_receipt(
                plan,
                kind="operator_conditioned_temporary_intervention",
                tensors=mlp_sources,
                child_recapture_sha256=recapture_sha256,
                operator_solution_sha256=operator_solution_sha256,
            )
            target = build_lfm2_sequential_writer_target(
                plan,
                parent_post_operator=rows["post_operator"].parent_rows,
                actual_child_pre=rows["pre"].child_rows,
                actual_child_post_operator=rows["post_operator"].child_rows,
                existing_operator_writer_output=(
                    rows["writers"].child_operator_output
                ),
                parent_post_layer=rows["post_layer"].parent_rows,
                fresh_child_post_operator=fresh_post_operator,
                fresh_child_post_layer=fresh_post_layer,
                existing_mlp_writer_output=fresh_mlp_down,
                operator_source_receipt=baseline["operator_source_receipt"],
                mlp_source_receipt=mlp_receipt,
            )
            mlp_solution, mlp_report = (
                solve_lfm2_moving_gauge_anchored_writer_replacement(
                    fresh_post_swiglu,
                    target.target_mlp_writer_output,
                    mlp_seed,
                    rows["weights"],
                )
            )
            conditioned[mode] = {
                "target": target,
                "operator_solution": baseline["operator_solution"],
                "operator_report": baseline["operator_report"],
                "mlp_solution": mlp_solution,
                "mlp_report": mlp_report,
                "mlp_source_receipt": mlp_receipt,
                "fresh_post_swiglu_sha256": (
                    fingerprint_lfm2_moving_gauge_tensor(fresh_post_swiglu)
                ),
                "recapture_sha256": recapture_sha256,
            }
        self.conditioned_targets = conditioned
        return V5LivePhaseExecution(
            artifact=conditioned,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "fit_frozen_operator_conditioned_targets_and_solutions",
                "charts": {
                    mode: {
                        "target_sha256": value["target"].report[
                            "result_sha256"
                        ],
                        "operator_solution_sha256": _solution_sha256(
                            value["operator_solution"],
                            value["operator_report"],
                        ),
                        "mlp_solution_sha256": _solution_sha256(
                            value["mlp_solution"],
                            value["mlp_report"],
                        ),
                        "fresh_post_swiglu_sha256": value[
                            "fresh_post_swiglu_sha256"
                        ],
                        "conditioned_recapture_sha256": value[
                            "recapture_sha256"
                        ],
                    }
                    for mode, value in conditioned.items()
                },
                "fit_only": True,
                "baseline_post_swiglu_used_for_conditioned_mlp": False,
                "fresh_post_swiglu_used_for_conditioned_mlp": True,
                "heldout_rows_read": False,
                "training_steps": 0,
            },
        )

    # ------------------------------------------------------------------
    # Phase 7/8: heldout diagnostics only after hashes freeze; then exactly six
    # FP32 two-tensor packs, still uninstalled and pending manual review.
    # ------------------------------------------------------------------
    def _phase_compare_frozen_candidates_on_heldout(
        self,
        _phase: Mapping[str, Any],
        completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        try:
            frozen_phase_artifact_sha256 = completed[
                "freeze_conditioned_targets_and_solutions"
            ].artifact_sha256
        except (AttributeError, KeyError) as exc:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "HELDOUT comparison requires the frozen-candidate phase barrier."
            ) from exc
        capabilities: dict[str, Any] = {}
        capability_sha256s: dict[str, str] = {}
        barrier_sha256s: dict[str, str] = {}
        for mode in v5.CHART_MODES:
            joint = self.baseline_targets[mode]["joint"]
            joint_split_receipts = _joint_split_solution_receipts(
                joint["operator_solution"],
                joint["mlp_solution"],
                joint["solve_report"],
            )
            frozen = self.conditioned_targets[mode]
            barrier_sha256 = canonical_sha256(
                {
                    "schema": (
                        "neural-forge-lfm2-v5-frozen-candidate-heldout-"
                        "barrier-v1"
                    ),
                    "v5_request_sha256": self.request.request_sha256,
                    "chart_mode": mode,
                    "freeze_phase_artifact_sha256": (
                        frozen_phase_artifact_sha256
                    ),
                    "joint": {
                        "operator_solution_tensor_sha256": (
                            fingerprint_lfm2_moving_gauge_tensor(
                                joint["operator_solution"]
                            )
                        ),
                        "mlp_solution_tensor_sha256": (
                            fingerprint_lfm2_moving_gauge_tensor(
                                joint["mlp_solution"]
                            )
                        ),
                        "combined_solve_report_sha256": _solution_sha256(
                            torch.cat(
                                (
                                    joint["operator_solution"],
                                    joint["mlp_solution"],
                                ),
                                dim=1,
                            ),
                            joint["solve_report"],
                        ),
                        "split_receipt_sha256s": dict(joint_split_receipts),
                    },
                    "sequential_conditioned": {
                        "operator_solution_tensor_sha256": (
                            fingerprint_lfm2_moving_gauge_tensor(
                                frozen["operator_solution"]
                            )
                        ),
                        "operator_solve_report_sha256": _solution_sha256(
                            frozen["operator_solution"],
                            frozen["operator_report"],
                        ),
                        "mlp_solution_tensor_sha256": (
                            fingerprint_lfm2_moving_gauge_tensor(
                                frozen["mlp_solution"]
                            )
                        ),
                        "mlp_solve_report_sha256": _solution_sha256(
                            frozen["mlp_solution"],
                            frozen["mlp_report"],
                        ),
                    },
                }
            )
            barrier_sha256s[mode] = barrier_sha256
            recapture = self.conditioned_recaptures[mode]
            issue_capability = getattr(
                recapture,
                "issue_heldout_access_capability",
                None,
            )
            if not callable(issue_capability):
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Conditioned recapture cannot authorize frozen HELDOUT access."
                )
            capability = issue_capability(
                frozen_candidate_barrier_sha256=barrier_sha256
            )
            capabilities[mode] = capability
            capability_sha256s[mode] = _digest_from(
                capability,
                "capability_hmac_sha256",
            )
        rows = self._baseline_rows("heldout")
        heldout64 = {
            "pre_child": rows["pre"].child_rows.to(torch.float64),
            "post_operator_parent": rows["post_operator"].parent_rows.to(
                torch.float64
            ),
            "post_operator_child": rows["post_operator"].child_rows.to(
                torch.float64
            ),
            "post_layer_parent": rows["post_layer"].parent_rows.to(
                torch.float64
            ),
            "post_layer_child": rows["post_layer"].child_rows.to(torch.float64),
            "combined_writer": rows[
                "writers"
            ].child_combined_writer_output.to(torch.float64),
            "operator_writer": rows["writers"].child_operator_output.to(
                torch.float64
            ),
            "mlp_writer": rows["writers"].child_mlp_down.to(torch.float64),
            "context": rows["context"].to(torch.float64),
            "post_swiglu": rows["post_swiglu"].to(torch.float64),
            "weights": rows["weights"].to(torch.float64),
        }
        dossier: dict[str, Any] = {}
        for mode in v5.CHART_MODES:
            charts = self.fit_charts[mode]
            endpoint_operator = charts["post_operator"].chart.encode_shaped(
                heldout64["post_operator_parent"]
            ).to(torch.float64)
            endpoint_layer = charts["post_layer"].chart.encode_shaped(
                heldout64["post_layer_parent"]
            ).to(torch.float64)
            fixed_joint = (
                heldout64["post_layer_child"]
                - heldout64["pre_child"]
                - heldout64["combined_writer"]
            )
            joint = self.baseline_targets[mode]["joint"]
            joint_prediction = (
                heldout64["pre_child"]
                + fixed_joint
                + heldout64["context"]
                @ joint["operator_solution"].to(torch.float64).T
                + heldout64["post_swiglu"]
                @ joint["mlp_solution"].to(torch.float64).T
            )
            fixed_operator = (
                heldout64["post_operator_child"]
                - heldout64["pre_child"]
                - heldout64["operator_writer"]
            )
            stale = self.baseline_targets[mode]["sequential_no_recapture"]
            stale_operator_prediction = (
                heldout64["pre_child"]
                + fixed_operator
                + heldout64["context"]
                @ stale["operator_solution"].to(torch.float64).T
            )
            fixed_stale_mlp = (
                heldout64["post_layer_child"]
                - heldout64["post_operator_child"]
                - heldout64["mlp_writer"]
            )
            stale_layer_prediction = (
                stale_operator_prediction
                + fixed_stale_mlp
                + heldout64["post_swiglu"]
                @ stale["mlp_solution"].to(torch.float64).T
            )
            recapture = self.conditioned_recaptures[mode]
            fresh_post_operator = _conditioned_rows(
                recapture,
                split="heldout",
                field_name="residual.post_operator",
                heldout_capability=capabilities[mode],
            ).to(torch.float64)
            fresh_post_layer = _conditioned_rows(
                recapture,
                split="heldout",
                field_name="residual.post_layer",
                heldout_capability=capabilities[mode],
            ).to(torch.float64)
            fresh_mlp_down = _conditioned_rows(
                recapture,
                split="heldout",
                field_name="mlp.down",
                heldout_capability=capabilities[mode],
            ).to(torch.float64)
            fresh_post_swiglu = _conditioned_rows(
                recapture,
                split="heldout",
                field_name="mlp.post_swiglu",
                heldout_capability=capabilities[mode],
            ).to(torch.float64)
            conditioned = self.conditioned_targets[mode]
            fixed_conditioned_mlp = (
                fresh_post_layer - fresh_post_operator - fresh_mlp_down
            )
            conditioned_prediction = (
                fresh_post_operator
                + fixed_conditioned_mlp
                + fresh_post_swiglu
                @ conditioned["mlp_solution"].to(torch.float64).T
            )
            dossier[mode] = {
                "joint_post_layer_relative_rmse": _relative_rmse(
                    joint_prediction,
                    endpoint_layer,
                    heldout64["weights"],
                ),
                "sequential_no_recapture_post_operator_relative_rmse": (
                    _relative_rmse(
                        stale_operator_prediction,
                        endpoint_operator,
                        heldout64["weights"],
                    )
                ),
                "sequential_no_recapture_post_layer_relative_rmse": (
                    _relative_rmse(
                        stale_layer_prediction,
                        endpoint_layer,
                        heldout64["weights"],
                    )
                ),
                "sequential_conditioned_post_layer_relative_rmse": (
                    _relative_rmse(
                        conditioned_prediction,
                        endpoint_layer,
                        heldout64["weights"],
                    )
                ),
                "chart_bundle_sha256": self.chart_bundle_sha256s[mode],
                "joint_solution_sha256": joint["solve_report"]["report_sha256"],
                "conditioned_mlp_solution_sha256": _solution_sha256(
                    conditioned["mlp_solution"],
                    conditioned["mlp_report"],
                ),
            }
            if any(
                not math.isfinite(value)
                for key, value in dossier[mode].items()
                if key.endswith("relative_rmse")
            ):
                raise Lfm2Exact4xMovingGaugeV5LiveError(
                    "Heldout numeric dossier contains a nonfinite metric."
                )
        self.heldout_dossier = dossier
        return V5LivePhaseExecution(
            artifact=dossier,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "frozen_candidate_heldout_numeric_dossier",
                "charts": dossier,
                "candidate_hashes_frozen_before_heldout": True,
                "freeze_phase_artifact_sha256": frozen_phase_artifact_sha256,
                "frozen_candidate_barrier_sha256s": barrier_sha256s,
                "heldout_capability_hmac_sha256s": capability_sha256s,
                "numeric_metrics_are_diagnostic_only": True,
                "numeric_quality_gate": False,
                "manual_semantic_review_status": "pending",
                "manual_semantic_review_performed": False,
                "manual_semantic_review_required_downstream": True,
                "child_forward_count": 0,
            },
            heldout_rows_read=True,
        )

    def _phase_stage_atomic_candidate_packs(
        self,
        _phase: Mapping[str, Any],
        completed: Mapping[str, v5.V5PhaseOutcome],
    ) -> V5LivePhaseExecution:
        current = fingerprint_lfm2_parent(self.child)
        current_route_state_sha256 = (
            fingerprint_lfm2_carrier_runtime_owned_tensor_state(self.routing)[
                "state_sha256"
            ]
        )
        if (
            current.identity_sha256 != self.child_fingerprint.identity_sha256
            or current.state_dict_sha256
            != self.child_fingerprint.state_dict_sha256
            or current_route_state_sha256 != self.route_state["state_sha256"]
        ):
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Source child or route state changed before candidate staging."
            )
        packs: dict[str, Any] = {}
        for mode in v5.CHART_MODES:
            candidates = {
                "joint": self.baseline_targets[mode]["joint"],
                "sequential_conditioned": self.conditioned_targets[mode],
            }
            for arm, candidate in candidates.items():
                operator = candidate["operator_solution"].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).contiguous()
                mlp = candidate["mlp_solution"].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).contiguous()
                if arm == "joint":
                    target_sha256 = candidate["target"].report["result_sha256"]
                    split_receipts = _joint_split_solution_receipts(
                        candidate["operator_solution"],
                        candidate["mlp_solution"],
                        candidate["solve_report"],
                    )
                    if dict(candidate["split_receipt_sha256s"]) != dict(
                        split_receipts
                    ):
                        raise Lfm2Exact4xMovingGaugeV5LiveError(
                            "Joint solution split receipts changed before staging."
                        )
                    operator_receipt = split_receipts["operator"]
                    mlp_receipt = split_receipts["mlp"]
                else:
                    target_sha256 = candidate["target"].report["result_sha256"]
                    operator_receipt = candidate["operator_report"]["report_sha256"]
                    mlp_receipt = candidate["mlp_report"]["report_sha256"]
                destination, receipt = self.bindings.stage_candidate_pack(
                    self.request,
                    chart_mode=mode,
                    target_arm=arm,
                    replacements={
                        v5.REPLACEMENT_TENSOR_NAMES[0]: operator,
                        v5.REPLACEMENT_TENSOR_NAMES[1]: mlp,
                    },
                    upstream_receipt_sha256s={
                        "chart": self.chart_bundle_sha256s[mode],
                        "heldout_comparison": completed[
                            "compare_frozen_candidates_on_heldout"
                        ].artifact_sha256,
                        "lineage": completed["authenticate_lineage"].artifact_sha256,
                        "mlp_solution": mlp_receipt,
                        "operator_solution": operator_receipt,
                        "target_plan": target_sha256,
                    },
                    child_state_before_sha256=current.state_dict_sha256,
                    child_state_after_sha256=current.state_dict_sha256,
                )
                key = f"{mode}:{arm}"
                packs[key] = {
                    "path": str(Path(destination).absolute()),
                    "candidate_pack_sha256": receipt["candidate_pack_sha256"],
                    "chart_mode": mode,
                    "target_arm": arm,
                }
        if set(packs) != {
            f"{mode}:{arm}"
            for mode in v5.CHART_MODES
            for arm in v5.PUBLISHABLE_TARGET_ARMS
        }:
            raise Lfm2Exact4xMovingGaugeV5LiveError(
                "Exactly six publishable candidate packs were not staged."
            )
        self.candidate_packs = packs
        return V5LivePhaseExecution(
            artifact=packs,
            evidence={
                "schema": LIVE_SCHEMA,
                "kind": "six_atomic_fp32_candidate_packs",
                "packs": packs,
                "candidate_pack_count": len(packs),
                "replacement_tensor_names": list(v5.REPLACEMENT_TENSOR_NAMES),
                "replacement_dtype": "float32",
                "installed": False,
                "ready_for_atomic_apply": True,
                "manual_semantic_review_status": "pending",
                "source_child_state_preserved": True,
                "source_child_identity_sha256": current.identity_sha256,
                "source_child_state_dict_sha256": current.state_dict_sha256,
                "source_route_state_sha256": current_route_state_sha256,
            },
            ready_for_atomic_apply=True,
        )


def build_lfm2_exact4x_moving_gauge_v5_live_runtime(
    request: v5.Lfm2Exact4xMovingGaugeV5Request,
    *,
    trace_adapter: Any,
    basis_phase_authority: Any,
    boundary_phase_authority: Any,
    feature_phase_authority: Any,
    conditioned_recapture_authority: Any,
    provenance_sha256s: Mapping[str, str],
    v3_first_divergence_request: Mapping[str, Any],
    v3_first_divergence_ledger: Mapping[str, Any],
    basis_split_by_batch: Mapping[int, str],
    parent_storage_dtype: str,
    child_storage_dtype: str,
    child_feature_storage_dtypes: Mapping[str, str],
    conditioned_recapture_loader: ConditionedRecaptureLoader | None = None,
    stage_candidate_pack: Callable[..., Any] = v5.stage_atomic_candidate_pack,
    runtime_device: str | None = None,
    runtime_dtype: str | None = None,
    boundary_shard_batch_count: int = 32,
    feature_shard_batch_count: int = 32,
    phase_overrides: Mapping[str, V5LivePhaseOverride] | None = None,
) -> Lfm2Exact4xMovingGaugeV5LiveRuntime:
    """Build one authenticated, non-shareable live execution context."""

    bindings = Lfm2Exact4xMovingGaugeV5LiveBindings(
        trace_adapter=trace_adapter,
        basis_phase_authority=basis_phase_authority,
        boundary_phase_authority=boundary_phase_authority,
        feature_phase_authority=feature_phase_authority,
        conditioned_recapture_authority=conditioned_recapture_authority,
        provenance_sha256s=provenance_sha256s,
        v3_first_divergence_request=v3_first_divergence_request,
        v3_first_divergence_ledger=v3_first_divergence_ledger,
        basis_split_by_batch=basis_split_by_batch,
        parent_storage_dtype=parent_storage_dtype,
        child_storage_dtype=child_storage_dtype,
        child_feature_storage_dtypes=child_feature_storage_dtypes,
        conditioned_recapture_loader=conditioned_recapture_loader,
        stage_candidate_pack=stage_candidate_pack,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        boundary_shard_batch_count=boundary_shard_batch_count,
        feature_shard_batch_count=feature_shard_batch_count,
    )
    return Lfm2Exact4xMovingGaugeV5LiveRuntime(
        request,
        bindings,
        phase_overrides=phase_overrides,
    )


def build_lfm2_exact4x_moving_gauge_v5_live_handlers(
    request: v5.Lfm2Exact4xMovingGaugeV5Request,
    **kwargs: Any,
) -> Mapping[str, v5.V5PhaseHandler]:
    """Factory requested by the thin runner: exactly nine closure-backed handlers."""

    return build_lfm2_exact4x_moving_gauge_v5_live_runtime(
        request,
        **kwargs,
    ).handlers()


# Short aliases are intentionally explicit for callers that already name the
# experiment in their import path.
build_v5_live_runtime = build_lfm2_exact4x_moving_gauge_v5_live_runtime
build_v5_live_handlers = build_lfm2_exact4x_moving_gauge_v5_live_handlers


__all__ = [
    "CONDITIONED_RECLOSURE_FIELDS",
    "Lfm2Exact4xMovingGaugeV5LiveBindings",
    "Lfm2Exact4xMovingGaugeV5LiveError",
    "Lfm2Exact4xMovingGaugeV5LiveRuntime",
    "V5LivePhaseExecution",
    "build_lfm2_exact4x_moving_gauge_v5_live_handlers",
    "build_lfm2_exact4x_moving_gauge_v5_live_runtime",
    "build_v5_live_handlers",
    "build_v5_live_runtime",
]
