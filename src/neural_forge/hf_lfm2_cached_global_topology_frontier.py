"""Cache-only global exact-topology frontier for production LFM2 folding.

The existing heterogeneous selector chooses MLP and operator capacity at one
residual width.  This module lifts that selector over every catalog-legal
residual width without changing the downstream blank-target or fold path.

Every feasible point is a complete, full-depth, dense child with the same
exact learned-parameter count.  Width is scored from a role-balanced cached
functional spectrum.  Per-layer geometry is scored from the cached nonlinear
capacity bank.  Parameter count is an equality constraint, never a quality
score.  No model is executed and no parameter is trained, pruned, quantized,
shared, or sparsified here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import textwrap
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch

from . import dynamic_parameter_frontier as _dynamic_parameter_frontier_module
from . import dynamic_fold_phase_cache as _dynamic_fold_phase_cache_module
from . import hf_lfm2_cached_functional_width as _cached_functional_width_module
from . import (
    hf_lfm2_cached_heterogeneous_capacity as _cached_heterogeneous_capacity_module,
)
from . import hf_lfm2_circuit_trace_cache as _circuit_trace_cache_module
from . import hf_lfm2_retained_trace_rows as _retained_trace_rows_module
from . import (
    hf_lfm2_heterogeneous_layout_selection as _heterogeneous_layout_selection_module,
)
from .dynamic_circuit_ir import (
    ModelCircuitGraph,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseEventLog,
    FoldPhaseKey,
    FoldPhaseKeyError,
    VerifiedFoldPhaseCacheEntry,
)
from .dynamic_parameter_frontier import Lfm2HeterogeneousLayoutSelectionInput
from .hf_lfm2_cached_functional_width import (
    LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA,
    LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA,
    LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
    LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
    LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA,
    Lfm2CachedFunctionalWidthEvidence,
    Lfm2FunctionalWidthScore,
    build_lfm2_cached_functional_width_evidence,
    enumerate_lfm2_global_hidden_sizes,
)
from .hf_lfm2_cached_heterogeneous_capacity import (
    ATTENTION_GEOMETRY_ENABLED_POLICY,
    ATTENTION_GEOMETRY_RISK_METHOD,
    AUTOMATIC_MLP_OPTION_POLICY,
    CAPACITY_OPTION_BANK_SCHEMA,
    EXPLICIT_MLP_OPTION_POLICY,
    MLP_CAPACITY_RISK_METHOD,
    SHORT_CONV_WRITER_ACTION_RISK_METHOD,
    Lfm2CachedAttentionGeometryRisk,
    Lfm2CachedHeterogeneousCapacityError,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    Lfm2CachedLayerCapacityEvidence,
    Lfm2CachedMlpCapacityRisk,
    Lfm2CachedShortConvWriterRisk,
    Lfm2MlpWidthOptionPolicy,
    build_lfm2_cached_heterogeneous_capacity_option_bank,
    derive_lfm2_automatic_mlp_width_policy,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from .hf_lfm2_global_topology_phase_compatibility import (
    GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_RECEIPT_SCHEMA,
    explicit_global_topology_phase_compatibility_receipt,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    LFM2_HETEROGENEOUS_LAYER_OPTION_SCHEMA,
    Lfm2AuthenticatedLayerPhysicalOption,
    Lfm2HeterogeneousLayoutSelection,
    Lfm2HeterogeneousLayoutUnattainable,
    build_lfm2_attention_layer_option,
    build_lfm2_heterogeneous_target_pins,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)
from .hf_lfm2_retained_trace_rows import (
    Lfm2RetainedTraceRowAdapter,
    Lfm2RetainedTraceRowsError,
)


GLOBAL_TOPOLOGY_RUNTIME_SCHEMA = "neural-forge-lfm2-global-runtime-pins-v1"
GLOBAL_TOPOLOGY_CANDIDATE_SCHEMA = "neural-forge-lfm2-global-topology-candidate-v1"
GLOBAL_TOPOLOGY_REJECTION_SCHEMA = "neural-forge-lfm2-global-topology-rejection-v1"
GLOBAL_TOPOLOGY_FRONTIER_SCHEMA = "neural-forge-lfm2-global-topology-frontier-v1"
GLOBAL_TOPOLOGY_SCORE_METHOD = (
    "role-balanced-functional-loss-plus-mean-cached-component-risk-v1"
)
RETARGETED_CAPACITY_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-retargeted-capacity-evidence-v2"
)
GLOBAL_TOPOLOGY_CAPACITY_PHASE = "lfm2-global-capacity-union-v1"
GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE = "lfm2-global-functional-width-v1"
GLOBAL_TOPOLOGY_SELECTION_PHASE = "lfm2-global-frontier-selection-v1"
GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-global-capacity-phase-payload-v1"
)
GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-global-functional-phase-payload-v1"
)
GLOBAL_TOPOLOGY_SELECTION_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-global-selection-phase-payload-v1"
)
GLOBAL_TOPOLOGY_PHASE_PAYLOAD_FILENAME = "evidence.json"
GLOBAL_TOPOLOGY_PHASE_CACHE_ABI = (
    "neural-forge-lfm2-global-topology-derived-phase-cache-v1"
)
GLOBAL_TOPOLOGY_TRACE_CACHE_INTERFACE_SCHEMA = (
    "neural-forge-lfm2-global-topology-trace-cache-interface-v1"
)

PhaseProgressCallback = Callable[[str, Mapping[str, Any]], None]


class Lfm2CachedGlobalTopologyFrontierError(RuntimeError):
    """The cache cannot authenticate a complete global exact frontier."""


class Lfm2CachedGlobalTopologyFrontierUnattainable(
    Lfm2CachedGlobalTopologyFrontierError
):
    """No legal residual width can materialize the requested exact child."""


class Lfm2CachedGlobalTopologyEvidenceCacheMiss(Lfm2CachedGlobalTopologyFrontierError):
    """Cold B1/B2 evidence was intentionally deferred until rows exist."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be a non-negative integer."
        )
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be finite and non-negative."
        )
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise Lfm2CachedGlobalTopologyFrontierError(f"{name} must be positive.")
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CachedGlobalTopologyFrontierError(f"{name} must be nonempty text.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Lfm2ExistingGlobalTopologyPhaseReuseRequest:
    """Exact immutable reference to one historical capacity or functional hit."""

    phase_name: str
    phase_key_sha256: str
    expected_completion_sha256: str
    expected_evidence_sha256: str
    expected_compatibility_receipt_sha256: str

    def __post_init__(self) -> None:
        phase_name = _text(self.phase_name, "phase_name")
        if phase_name not in {
            GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        }:
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Historical global-topology reuse is limited to capacity and "
                "functional evidence phases."
            )
        object.__setattr__(self, "phase_name", phase_name)
        for name in (
            "phase_key_sha256",
            "expected_completion_sha256",
            "expected_evidence_sha256",
            "expected_compatibility_receipt_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": (
                "neural-forge-lfm2-existing-global-topology-phase-reuse-request-v1"
            ),
            "phase_name": self.phase_name,
            "phase_key_sha256": self.phase_key_sha256,
            "expected_completion_sha256": self.expected_completion_sha256,
            "expected_evidence_sha256": self.expected_evidence_sha256,
            "expected_compatibility_receipt_sha256": (
                self.expected_compatibility_receipt_sha256
            ),
            "cache_miss_build_allowed": False,
        }
        return {**body, "request_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTopologyPhaseCacheContext:
    """Operational cache binding for resumable cache-only topology evidence."""

    cache: DynamicFoldPhaseCache
    evidence_split_sha256: str
    events: FoldPhaseEventLog | None = None
    progress: PhaseProgressCallback | None = None
    selective_session_factory: Callable[[], Any] | None = None
    existing_capacity_phase_reuse: (
        Lfm2ExistingGlobalTopologyPhaseReuseRequest | None
    ) = None
    existing_functional_phase_reuse: (
        Lfm2ExistingGlobalTopologyPhaseReuseRequest | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache, DynamicFoldPhaseCache):
            raise TypeError("cache must be DynamicFoldPhaseCache.")
        _digest(self.evidence_split_sha256, "evidence_split_sha256")
        if self.events is not None and not isinstance(self.events, FoldPhaseEventLog):
            raise TypeError("events must be FoldPhaseEventLog or None.")
        if self.progress is not None and not callable(self.progress):
            raise TypeError("progress must be callable or None.")
        if self.selective_session_factory is not None and not callable(
            self.selective_session_factory
        ):
            raise TypeError("selective_session_factory must be callable or None.")
        for field_name, expected_phase_name in (
            ("existing_capacity_phase_reuse", GLOBAL_TOPOLOGY_CAPACITY_PHASE),
            ("existing_functional_phase_reuse", GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE),
        ):
            request = getattr(self, field_name)
            if request is not None and not isinstance(
                request,
                Lfm2ExistingGlobalTopologyPhaseReuseRequest,
            ):
                raise TypeError(
                    f"{field_name} must be "
                    "Lfm2ExistingGlobalTopologyPhaseReuseRequest or None."
                )
            if request is not None and request.phase_name != expected_phase_name:
                raise Lfm2CachedGlobalTopologyFrontierError(
                    f"{field_name} must name {expected_phase_name}."
                )


def _phase_progress(
    context: Lfm2GlobalTopologyPhaseCacheContext | None,
    stage: str,
    **details: Any,
) -> None:
    if context is not None and context.progress is not None:
        context.progress(stage, MappingProxyType(dict(details)))


def _strict_mapping(
    value: Any,
    *,
    expected_fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise FoldPhaseCacheIntegrityError(
            f"{label} fields are missing, foreign, or noncanonical."
        )
    return dict(value)


def _sealed_phase_payload(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    payload_body = {"schema": schema, **dict(body)}
    return {
        **payload_body,
        "payload_sha256": canonical_sha256(payload_body),
    }


def _write_phase_payload(
    path: Path,
    *,
    schema: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _sealed_phase_payload(schema, body)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def _read_phase_payload(
    entry: VerifiedFoldPhaseCacheEntry,
    *,
    schema: str,
    expected_phase_key: FoldPhaseKey,
) -> dict[str, Any]:
    if entry.phase_key.as_dict() != expected_phase_key.as_dict():
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase entry differs from its requested phase key."
        )
    payload_path = entry.payload_path / GLOBAL_TOPOLOGY_PHASE_PAYLOAD_FILENAME
    if not payload_path.is_file() or payload_path.is_symlink():
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase payload is missing or unsafe."
        )
    try:
        raw = payload_path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase payload is unreadable."
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase payload is not canonical JSON."
        )
    if value.get("schema") != schema:
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase payload schema is unsupported."
        )
    body = dict(value)
    seal = body.pop("payload_sha256", None)
    if seal != canonical_sha256(body):
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase payload seal is invalid."
        )
    if body.get("phase_key_sha256") != expected_phase_key.phase_key_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 payload is not bound to its phase key."
        )
    if entry.result.get("payload_sha256") != seal:
        raise FoldPhaseCacheIntegrityError(
            "Derived LFM2 phase completion does not bind its payload receipt."
        )
    return value


def _raw_module_source_sha256(path_value: Any, *, label: str) -> str:
    if not isinstance(path_value, str) or not path_value:
        raise FoldPhaseCacheIntegrityError(
            f"Cannot locate current {label} source for historical phase reuse."
        )
    try:
        raw = Path(path_value).resolve(strict=True).read_bytes()
    except OSError as exc:
        raise FoldPhaseCacheIntegrityError(
            f"Cannot authenticate current {label} source for historical phase reuse."
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _existing_global_topology_phase_key_from_digest(
    cache: DynamicFoldPhaseCache,
    *,
    phase_name: str,
    phase_key_sha256: str,
) -> FoldPhaseKey:
    """Read one immutable historical completion key without creating a path."""

    digest = _digest(phase_key_sha256, "phase_key_sha256")
    entry = cache.root / phase_name / digest
    completion = entry / _dynamic_fold_phase_cache_module.COMPLETE_FILENAME
    _dynamic_fold_phase_cache_module._assert_no_reparse_ancestors(
        entry.parent,
        FoldPhaseCacheIntegrityError,
    )
    if not _dynamic_fold_phase_cache_module._lexists(entry):
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"Requested historical {phase_name} phase is absent; hit-only "
            "loading forbids building or repairing it."
        )
    _dynamic_fold_phase_cache_module._assert_ordinary_directory(
        entry,
        "historical global-topology phase entry",
        FoldPhaseCacheIntegrityError,
    )
    _dynamic_fold_phase_cache_module._assert_ordinary_file(
        completion,
        "historical global-topology completion marker",
        FoldPhaseCacheIntegrityError,
    )
    raw = _dynamic_fold_phase_cache_module._read_limited(
        completion,
        _dynamic_fold_phase_cache_module._MAX_JSON_BYTES,
        "historical global-topology completion marker",
        FoldPhaseCacheIntegrityError,
    )
    value = _dynamic_fold_phase_cache_module._strict_canonical_json_bytes(
        raw,
        "historical global-topology completion marker",
        FoldPhaseCacheIntegrityError,
    )
    if value.get("phase_key_sha256") != digest:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology completion is noncanonical or has a stale key."
        )
    try:
        key = FoldPhaseKey.from_dict(value.get("phase_key"))
    except (FoldPhaseKeyError, TypeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology completion contains an invalid phase key."
        ) from exc
    if key.phase_name != phase_name or key.phase_key_sha256 != digest:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology phase key differs from its immutable "
            "cache address."
        )
    return key


def _current_historical_phase_compatibility_receipt(
    *,
    historical_algorithm_sha256: str,
    current_algorithm_sha256: str,
) -> Mapping[str, Any]:
    live_algorithm_sha256 = _global_algorithm_fingerprint()
    if live_algorithm_sha256 != current_algorithm_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Current global-topology implementation changed while historical "
            "phase compatibility was checked."
        )
    global_source_sha256 = _raw_module_source_sha256(
        __file__,
        label="global-frontier",
    )
    functional_source_sha256 = _raw_module_source_sha256(
        _cached_functional_width_module.__file__,
        label="functional-width",
    )
    receipt = explicit_global_topology_phase_compatibility_receipt(
        historical_algorithm_sha256=historical_algorithm_sha256,
        current_global_frontier_source_sha256=global_source_sha256,
        current_functional_width_source_sha256=functional_source_sha256,
        current_algorithm_sha256=current_algorithm_sha256,
    )
    if receipt is None:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology phase has no exact directional "
            "compatibility receipt for the current implementation."
        )
    return _validate_global_topology_phase_compatibility_receipt(
        receipt,
        historical_algorithm_sha256=historical_algorithm_sha256,
        current_global_frontier_source_sha256=global_source_sha256,
        current_functional_width_source_sha256=functional_source_sha256,
        current_algorithm_sha256=current_algorithm_sha256,
    )


def _validate_global_topology_phase_compatibility_receipt(
    receipt: Any,
    *,
    historical_algorithm_sha256: str,
    current_global_frontier_source_sha256: str,
    current_functional_width_source_sha256: str,
    current_algorithm_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate every field of one exact directional policy receipt."""

    raw = _strict_mapping(
        receipt,
        expected_fields={
            "schema",
            "policy_sha256",
            "historical_algorithm_sha256",
            "current_global_frontier_source_sha256",
            "current_functional_width_source_sha256",
            "current_algorithm_sha256",
            "direction",
            "historical_artifact_rewritten",
            "compatibility_receipt_sha256",
        },
        label="global-topology phase compatibility receipt",
    )
    expected = {
        "historical_algorithm_sha256": historical_algorithm_sha256,
        "current_global_frontier_source_sha256": (
            current_global_frontier_source_sha256
        ),
        "current_functional_width_source_sha256": (
            current_functional_width_source_sha256
        ),
        "current_algorithm_sha256": current_algorithm_sha256,
    }
    try:
        _digest(raw["policy_sha256"], "compatibility policy_sha256")
        for field_name, expected_digest in expected.items():
            _digest(raw[field_name], f"compatibility {field_name}")
            _digest(expected_digest, f"expected compatibility {field_name}")
        claimed_receipt_sha256 = _digest(
            raw["compatibility_receipt_sha256"],
            "compatibility_receipt_sha256",
        )
    except Lfm2CachedGlobalTopologyFrontierError as exc:
        raise FoldPhaseCacheIntegrityError(str(exc)) from exc
    if (
        raw["schema"] != GLOBAL_TOPOLOGY_PHASE_COMPATIBILITY_RECEIPT_SCHEMA
        or raw["direction"] != "historical-to-current"
        or raw["historical_artifact_rewritten"] is not False
        or any(raw[field_name] != value for field_name, value in expected.items())
    ):
        raise FoldPhaseCacheIntegrityError(
            "Global-topology phase compatibility receipt contract is invalid."
        )
    body = dict(raw)
    body.pop("compatibility_receipt_sha256")
    if canonical_sha256(body) != claimed_receipt_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Global-topology phase compatibility receipt self-hash does not verify."
        )
    return MappingProxyType(raw)


def _lookup_existing_global_topology_phase_payload(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    current_key: FoldPhaseKey,
    request: Lfm2ExistingGlobalTopologyPhaseReuseRequest,
) -> tuple[
    VerifiedFoldPhaseCacheEntry,
    dict[str, Any],
    Mapping[str, Any],
]:
    """Load one exact historical hit after proving a single allowed delta."""

    if request.phase_name != current_key.phase_name:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology request names a different phase."
        )
    _phase_progress(
        context,
        "global_topology_existing_phase_reuse_lookup",
        phase_name=request.phase_name,
        historical_phase_key_sha256=request.phase_key_sha256,
        current_phase_key_sha256=current_key.phase_key_sha256,
        request_sha256=request.as_dict()["request_sha256"],
        cache_miss_build_allowed=False,
    )
    historical_key = _existing_global_topology_phase_key_from_digest(
        context.cache,
        phase_name=request.phase_name,
        phase_key_sha256=request.phase_key_sha256,
    )
    historical_document = historical_key.as_dict()
    current_document = current_key.as_dict()
    historical_dependencies = dict(historical_document["dependency_sha256s"])
    current_dependencies = dict(current_document["dependency_sha256s"])
    historical_algorithm_sha256 = historical_dependencies.get(
        "global_topology_algorithms"
    )
    current_algorithm_sha256 = current_dependencies.get("global_topology_algorithms")
    try:
        historical_algorithm_sha256 = _digest(
            historical_algorithm_sha256,
            "historical global_topology_algorithms",
        )
        current_algorithm_sha256 = _digest(
            current_algorithm_sha256,
            "current global_topology_algorithms",
        )
    except Lfm2CachedGlobalTopologyFrontierError as exc:
        raise FoldPhaseCacheIntegrityError(str(exc)) from exc
    historical_dependencies["global_topology_algorithms"] = current_algorithm_sha256
    normalized_historical_key = FoldPhaseKey(
        phase_name=historical_key.phase_name,
        dependency_sha256s=historical_dependencies,
        numeric_contract=historical_document["numeric_contract"],
        abi=historical_document["abi"],
    )
    if normalized_historical_key.as_dict() != current_document:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology phase differs from the current phase "
            "outside dependency_sha256s.global_topology_algorithms."
        )
    compatibility_receipt = _current_historical_phase_compatibility_receipt(
        historical_algorithm_sha256=historical_algorithm_sha256,
        current_algorithm_sha256=current_algorithm_sha256,
    )
    if (
        compatibility_receipt["compatibility_receipt_sha256"]
        != request.expected_compatibility_receipt_sha256
    ):
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology request pins a different exact "
            "compatibility receipt."
        )
    entry = context.cache.lookup(historical_key, events=context.events)
    if entry is None:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"Requested historical {request.phase_name} phase is absent; "
            "hit-only loading forbids building or repairing it."
        )
    if entry.completion_sha256 != request.expected_completion_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology completion differs from the exact "
            "requested completion digest."
        )
    schema = {
        GLOBAL_TOPOLOGY_CAPACITY_PHASE: GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE: GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
    }[request.phase_name]
    payload = _read_phase_payload(
        entry,
        schema=schema,
        expected_phase_key=historical_key,
    )
    return entry, payload, compatibility_receipt


def _complete_existing_global_topology_phase_reuse(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    current_key: FoldPhaseKey,
    request: Lfm2ExistingGlobalTopologyPhaseReuseRequest,
    entry: VerifiedFoldPhaseCacheEntry,
    payload: Mapping[str, Any],
    compatibility_receipt: Mapping[str, Any],
    *,
    evidence_sha256: str,
) -> Mapping[str, Any]:
    """Reauthenticate the immutable hit and emit a sealed no-rewrite receipt."""

    evidence = _digest(evidence_sha256, "historical evidence_sha256")
    if evidence != request.expected_evidence_sha256:
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology evidence differs from the exact "
            "requested evidence digest."
        )
    historical_algorithm_sha256 = compatibility_receipt["historical_algorithm_sha256"]
    current_algorithm_sha256 = compatibility_receipt["current_algorithm_sha256"]
    current_receipt = _current_historical_phase_compatibility_receipt(
        historical_algorithm_sha256=historical_algorithm_sha256,
        current_algorithm_sha256=current_algorithm_sha256,
    )
    if dict(current_receipt) != dict(compatibility_receipt):
        raise FoldPhaseCacheIntegrityError(
            "Global-topology compatibility policy changed while the historical "
            "phase was read."
        )
    reauthenticated = context.cache.lookup(entry.phase_key)
    if (
        reauthenticated is None
        or reauthenticated.completion_sha256 != entry.completion_sha256
        or reauthenticated.payload_inventory != entry.payload_inventory
        or reauthenticated.result != entry.result
    ):
        raise FoldPhaseCacheIntegrityError(
            "Historical global-topology phase changed while it was read."
        )
    body = {
        "schema": "neural-forge-lfm2-global-topology-phase-live-reuse-v1",
        "request_sha256": request.as_dict()["request_sha256"],
        "phase_name": request.phase_name,
        "historical_phase_key_sha256": entry.phase_key.phase_key_sha256,
        "current_phase_key_sha256": current_key.phase_key_sha256,
        "historical_completion_sha256": entry.completion_sha256,
        "historical_payload_sha256": payload["payload_sha256"],
        "historical_evidence_sha256": evidence,
        "historical_algorithm_sha256": historical_algorithm_sha256,
        "current_algorithm_sha256": current_algorithm_sha256,
        "compatibility_policy_sha256": compatibility_receipt["policy_sha256"],
        "compatibility_receipt_sha256": compatibility_receipt[
            "compatibility_receipt_sha256"
        ],
        "semantic_phase_contract_exact": True,
        "historical_artifact_rewritten": False,
        "cache_miss_build_allowed": False,
        "parent_model_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    receipt = MappingProxyType({**body, "reuse_receipt_sha256": canonical_sha256(body)})
    _phase_progress(
        context,
        "global_topology_existing_phase_reuse_hit",
        **dict(receipt),
    )
    return receipt


def _lookup_phase_payload(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    key: FoldPhaseKey,
) -> tuple[VerifiedFoldPhaseCacheEntry, dict[str, Any]] | None:
    _phase_progress(
        context,
        "global_topology_phase_cache_lookup",
        phase_name=key.phase_name,
        phase_key_sha256=key.phase_key_sha256,
    )
    entry = context.cache.lookup(key, events=context.events)
    if entry is None:
        _phase_progress(
            context,
            "global_topology_phase_cache_miss",
            phase_name=key.phase_name,
            phase_key_sha256=key.phase_key_sha256,
        )
        return None
    schema = {
        GLOBAL_TOPOLOGY_CAPACITY_PHASE: GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE: GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
        GLOBAL_TOPOLOGY_SELECTION_PHASE: GLOBAL_TOPOLOGY_SELECTION_PAYLOAD_SCHEMA,
    }[key.phase_name]
    payload = _read_phase_payload(entry, schema=schema, expected_phase_key=key)
    _phase_progress(
        context,
        "global_topology_phase_cache_hit",
        phase_name=key.phase_name,
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        payload_bytes=entry.payload_inventory["total_bytes"],
        resumed=True,
    )
    return entry, payload


def _publish_phase_payload(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    key: FoldPhaseKey,
    *,
    schema: str,
    body: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[VerifiedFoldPhaseCacheEntry, dict[str, Any], bool]:
    _phase_progress(
        context,
        "global_topology_phase_cache_publish_started",
        phase_name=key.phase_name,
        phase_key_sha256=key.phase_key_sha256,
    )
    try:
        transaction = context.cache.begin(key, events=context.events)
    except FoldPhaseCacheAlreadyCompleteError:
        raced = context.cache.lookup(key, events=context.events)
        if raced is None:  # pragma: no cover - defensive concurrent deletion
            raise FoldPhaseCacheIntegrityError(
                "Concurrent derived-phase publication disappeared."
            )
        payload = _read_phase_payload(raced, schema=schema, expected_phase_key=key)
        _phase_progress(
            context,
            "global_topology_phase_cache_publish_race_resumed",
            phase_name=key.phase_name,
            phase_key_sha256=key.phase_key_sha256,
            completion_sha256=raced.completion_sha256,
            payload_bytes=raced.payload_inventory["total_bytes"],
        )
        return raced, payload, False
    with transaction:
        payload = _write_phase_payload(
            transaction.payload_path / GLOBAL_TOPOLOGY_PHASE_PAYLOAD_FILENAME,
            schema=schema,
            body={"phase_key_sha256": key.phase_key_sha256, **dict(body)},
        )
        entry = transaction.complete(
            result={"payload_sha256": payload["payload_sha256"], **dict(result)}
        )
    _phase_progress(
        context,
        "global_topology_phase_cache_published",
        phase_name=key.phase_name,
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        payload_bytes=entry.payload_inventory["total_bytes"],
    )
    return entry, payload, True


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTargetRuntimePins:
    """Runtime and token-interface pins shared by every residual width."""

    vocab_size: int
    conv_kernel_size: int
    rope_theta: float
    norm_eps: float
    max_position_embeddings: int
    use_cache: bool
    attention_implementation: str
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    block_auto_adjust_ff_dim: bool
    target_vocab_evidence_sha256: str
    token_bridge_evidence_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.vocab_size, "vocab_size")
        _positive_int(self.conv_kernel_size, "conv_kernel_size")
        _positive_float(self.rope_theta, "rope_theta")
        _positive_float(self.norm_eps, "norm_eps")
        _positive_int(self.max_position_embeddings, "max_position_embeddings")
        if type(self.use_cache) is not bool:
            raise Lfm2CachedGlobalTopologyFrontierError("use_cache must be boolean.")
        _text(self.attention_implementation, "attention_implementation")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            _nonnegative_int(getattr(self, name), name)
        if type(self.block_auto_adjust_ff_dim) is not bool:
            raise Lfm2CachedGlobalTopologyFrontierError(
                "block_auto_adjust_ff_dim must be boolean."
            )
        _digest(
            self.target_vocab_evidence_sha256,
            "target_vocab_evidence_sha256",
        )
        _digest(
            self.token_bridge_evidence_sha256,
            "token_bridge_evidence_sha256",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_RUNTIME_SCHEMA,
            "vocab_size": self.vocab_size,
            "conv_kernel_size": self.conv_kernel_size,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "use_cache": self.use_cache,
            "attention_implementation": self.attention_implementation,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "block_auto_adjust_ff_dim": self.block_auto_adjust_ff_dim,
            "target_vocab_evidence_sha256": (self.target_vocab_evidence_sha256),
            "token_bridge_evidence_sha256": (self.token_bridge_evidence_sha256),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTopologyCandidateReport:
    """One feasible complete exact child on the global frontier."""

    hidden_size: int
    exact_unique_parameters: int
    target_layer_count: int
    role_balanced_functional_retention: float
    role_balanced_functional_loss: float
    mean_selected_component_risk: float
    combined_predicted_risk: float
    functional_score_sha256: str
    mlp_width_lattice_sha256: str
    capacity_option_bank_sha256: str
    selection_input_sha256: str
    descriptor_sha256: str
    layout_selection_receipt_sha256: str
    candidate_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.hidden_size, "hidden_size")
        _positive_int(self.exact_unique_parameters, "exact_unique_parameters")
        _positive_int(self.target_layer_count, "target_layer_count")
        retained = _finite_nonnegative(
            self.role_balanced_functional_retention,
            "role_balanced_functional_retention",
        )
        loss = _finite_nonnegative(
            self.role_balanced_functional_loss,
            "role_balanced_functional_loss",
        )
        if (
            retained > 1.0 + 1e-12
            or loss > 1.0 + 1e-12
            or not math.isclose(
                retained + loss,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Functional retention and loss must be complementary fractions."
            )
        component = _finite_nonnegative(
            self.mean_selected_component_risk,
            "mean_selected_component_risk",
        )
        combined = _finite_nonnegative(
            self.combined_predicted_risk,
            "combined_predicted_risk",
        )
        if not math.isclose(combined, loss + component, rel_tol=0.0, abs_tol=1e-10):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Combined frontier risk differs from its cached evidence terms."
            )
        for name in (
            "functional_score_sha256",
            "mlp_width_lattice_sha256",
            "capacity_option_bank_sha256",
            "selection_input_sha256",
            "descriptor_sha256",
            "layout_selection_receipt_sha256",
        ):
            _digest(getattr(self, name), name)
        if canonical_sha256(self._body()) != _digest(
            self.candidate_sha256,
            "candidate_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global topology candidate seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_CANDIDATE_SCHEMA,
            "hidden_size": self.hidden_size,
            "exact_unique_parameters": self.exact_unique_parameters,
            "target_layer_count": self.target_layer_count,
            "role_balanced_functional_retention": (
                self.role_balanced_functional_retention
            ),
            "role_balanced_functional_loss": (self.role_balanced_functional_loss),
            "mean_selected_component_risk": self.mean_selected_component_risk,
            "combined_predicted_risk": self.combined_predicted_risk,
            "score_method": GLOBAL_TOPOLOGY_SCORE_METHOD,
            "functional_score_sha256": self.functional_score_sha256,
            "mlp_width_lattice_sha256": self.mlp_width_lattice_sha256,
            "capacity_option_bank_sha256": self.capacity_option_bank_sha256,
            "selection_input_sha256": self.selection_input_sha256,
            "descriptor_sha256": self.descriptor_sha256,
            "layout_selection_receipt_sha256": (self.layout_selection_receipt_sha256),
            "parameter_count_used_as_quality_score": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "candidate_sha256": self.candidate_sha256}

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        exact_unique_parameters: int,
        target_layer_count: int,
        functional_retention: float,
        functional_loss: float,
        mean_component_risk: float,
        functional_score_sha256: str,
        mlp_width_lattice_sha256: str,
        capacity_option_bank_sha256: str,
        selection_input_sha256: str,
        descriptor_sha256: str,
        layout_selection_receipt_sha256: str,
    ) -> Lfm2GlobalTopologyCandidateReport:
        values = {
            "hidden_size": hidden_size,
            "exact_unique_parameters": exact_unique_parameters,
            "target_layer_count": target_layer_count,
            "role_balanced_functional_retention": functional_retention,
            "role_balanced_functional_loss": functional_loss,
            "mean_selected_component_risk": mean_component_risk,
            "combined_predicted_risk": functional_loss + mean_component_risk,
            "functional_score_sha256": functional_score_sha256,
            "mlp_width_lattice_sha256": mlp_width_lattice_sha256,
            "capacity_option_bank_sha256": capacity_option_bank_sha256,
            "selection_input_sha256": selection_input_sha256,
            "descriptor_sha256": descriptor_sha256,
            "layout_selection_receipt_sha256": (layout_selection_receipt_sha256),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "candidate_sha256", "0" * 64)
        return cls(**values, candidate_sha256=canonical_sha256(provisional._body()))


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTopologyRejectedWidth:
    hidden_size: int
    reason_class: str
    reason: str
    rejection_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.hidden_size, "hidden_size")
        _text(self.reason_class, "reason_class")
        _text(self.reason, "reason")
        if canonical_sha256(self._body()) != _digest(
            self.rejection_sha256,
            "rejection_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global topology rejection seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_REJECTION_SCHEMA,
            "hidden_size": self.hidden_size,
            "reason_class": self.reason_class,
            "reason": self.reason,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "rejection_sha256": self.rejection_sha256}

    @classmethod
    def build(
        cls,
        hidden_size: int,
        reason_class: str,
        reason: str,
    ) -> Lfm2GlobalTopologyRejectedWidth:
        body = {
            "schema": GLOBAL_TOPOLOGY_REJECTION_SCHEMA,
            "hidden_size": hidden_size,
            "reason_class": reason_class,
            "reason": reason,
        }
        return cls(
            hidden_size=hidden_size,
            reason_class=reason_class,
            reason=reason,
            rejection_sha256=canonical_sha256(body),
        )


def select_lfm2_global_topology_report(
    candidates: Sequence[Lfm2GlobalTopologyCandidateReport],
) -> Lfm2GlobalTopologyCandidateReport:
    """Choose by cached function and component risk, never width or count."""

    values = tuple(candidates)
    if not values or any(
        type(value) is not Lfm2GlobalTopologyCandidateReport for value in values
    ):
        raise Lfm2CachedGlobalTopologyFrontierUnattainable(
            "No authenticated exact global-topology candidate is available."
        )
    counts = {value.exact_unique_parameters for value in values}
    if len(counts) != 1:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "Global candidates must share one exact learned-parameter count."
        )
    return min(
        values,
        key=lambda value: (
            value.combined_predicted_risk,
            value.role_balanced_functional_loss,
            value.mean_selected_component_risk,
            value.candidate_sha256,
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CachedGlobalTopologyFrontier:
    """Selected production input plus the authenticated global-width audit."""

    source_graph_fingerprint: str
    parent_cache_key: str
    exact_target_unique_parameters: int
    source_layer_count: int
    required_target_hidden_size: int | None
    candidate_hidden_sizes: tuple[int, ...]
    base_union_capacity_option_bank_sha256: str
    functional_width_evidence: Lfm2CachedFunctionalWidthEvidence
    candidates: tuple[Lfm2GlobalTopologyCandidateReport, ...]
    rejected_widths: tuple[Lfm2GlobalTopologyRejectedWidth, ...]
    selected_candidate_sha256: str
    selected_input: Lfm2HeterogeneousLayoutSelectionInput
    selected_capacity_option_bank: Lfm2CachedHeterogeneousCapacityOptionBank
    selected_layout: Lfm2HeterogeneousLayoutSelection
    frontier_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_graph_fingerprint, "source_graph_fingerprint")
        _digest(self.parent_cache_key, "parent_cache_key")
        target = _positive_int(
            self.exact_target_unique_parameters,
            "exact_target_unique_parameters",
        )
        source_layer_count = _positive_int(
            self.source_layer_count,
            "source_layer_count",
        )
        required_hidden_size = (
            None
            if self.required_target_hidden_size is None
            else _positive_int(
                self.required_target_hidden_size,
                "required_target_hidden_size",
            )
        )
        widths = tuple(self.candidate_hidden_sizes)
        if not widths or widths != tuple(sorted(set(widths))):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global candidate widths must be nonempty and canonical."
            )
        if required_hidden_size is not None and widths != (required_hidden_size,):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "A required target hidden size must constrain the authenticated "
                "candidate inventory to that exact width."
            )
        _digest(
            self.base_union_capacity_option_bank_sha256,
            "base_union_capacity_option_bank_sha256",
        )
        candidates = tuple(self.candidates)
        rejected = tuple(self.rejected_widths)
        if (
            tuple(value.hidden_size for value in candidates)
            != tuple(sorted(value.hidden_size for value in candidates))
            or tuple(value.hidden_size for value in rejected)
            != tuple(sorted(value.hidden_size for value in rejected))
            or {value.hidden_size for value in candidates}
            & {value.hidden_size for value in rejected}
            or {value.hidden_size for value in candidates}
            | {value.hidden_size for value in rejected}
            != set(widths)
            or any(value.exact_unique_parameters != target for value in candidates)
            or any(
                value.target_layer_count != source_layer_count for value in candidates
            )
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global frontier coverage or exact accounting is inconsistent."
            )
        selected = select_lfm2_global_topology_report(candidates)
        if selected.candidate_sha256 != _digest(
            self.selected_candidate_sha256,
            "selected_candidate_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Selected global candidate is not the cached-risk optimum."
            )
        if (
            self.selected_input.fingerprint != selected.selection_input_sha256
            or self.selected_capacity_option_bank.evidence_sha256
            != selected.capacity_option_bank_sha256
            or self.selected_capacity_option_bank.attention_geometry_policy
            != ATTENTION_GEOMETRY_ENABLED_POLICY
            or self.selected_layout.descriptor.descriptor_sha256
            != selected.descriptor_sha256
            or self.selected_layout.receipt.receipt_sha256
            != selected.layout_selection_receipt_sha256
            or self.selected_layout.descriptor.accounting.unique_parameter_elements
            != target
            or self.selected_input.target_pins.hidden_size != selected.hidden_size
            or (
                required_hidden_size is not None
                and selected.hidden_size != required_hidden_size
            )
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Selected production artifacts differ from the frontier report."
            )
        if canonical_sha256(self._body()) != _digest(
            self.frontier_sha256,
            "frontier_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global topology frontier seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_FRONTIER_SCHEMA,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "parent_cache_key": self.parent_cache_key,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "source_layer_count": self.source_layer_count,
            "required_target_hidden_size": self.required_target_hidden_size,
            "candidate_hidden_sizes": list(self.candidate_hidden_sizes),
            "base_union_capacity_option_bank_sha256": (
                self.base_union_capacity_option_bank_sha256
            ),
            "functional_width_evidence_sha256": (
                self.functional_width_evidence.evidence_sha256
            ),
            "candidates": [value.as_dict() for value in self.candidates],
            "rejected_widths": [value.as_dict() for value in self.rejected_widths],
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "selected_input_sha256": self.selected_input.fingerprint,
            "selected_capacity_option_bank_sha256": (
                self.selected_capacity_option_bank.evidence_sha256
            ),
            "attention_geometry_policy": (
                self.selected_capacity_option_bank.attention_geometry_policy
            ),
            "selected_descriptor_sha256": (
                self.selected_layout.descriptor.descriptor_sha256
            ),
            "selected_layout_selection_receipt_sha256": (
                self.selected_layout.receipt.receipt_sha256
            ),
            "selection_contract": {
                "complete_full_depth_child_per_feasible_width": True,
                "same_exact_parameter_count_per_feasible_width": True,
                "parameter_count_used_as_quality_score": False,
                "cached_parent_function_only": True,
                "cached_attention_geometry_allocation": True,
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "sparse_parameters": 0,
                "shared_parameters": 0,
            },
        }

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **self._body(),
                "functional_width_evidence": dict(
                    self.functional_width_evidence.as_dict()
                ),
                "frontier_sha256": self.frontier_sha256,
            }
        )


def _parent_phase_dependencies(
    parent_cache: Lfm2CircuitTraceCache,
    *,
    evidence_split_sha256: str,
) -> dict[str, str]:
    manifest = parent_cache.manifest
    try:
        implementation = manifest["implementation"]
        execution = manifest["execution"]
        capture = manifest["capture"]
        dependencies = {
            "parent_cache_key": parent_cache.cache_key,
            "parent_manifest": manifest["manifest_sha256"],
            "parent_completion": parent_cache._completion["completion_sha256"],
            "parent_identity": parent_cache.parent_fingerprint,
            "parent_corpus": parent_cache.corpus_fingerprint,
            "parent_topology": parent_cache.layer_topology_fingerprint,
            "parent_implementation": parent_cache.implementation_fingerprint,
            "collector_source": implementation["collector_source_sha256"],
            "canonical_contract_source": implementation[
                "canonical_contract_source_sha256"
            ],
            "transformers_lfm2_runtime_source": implementation[
                "transformers_lfm2_source_sha256"
            ],
            "parent_execution": execution["execution_sha256"],
            "parent_capture_selection": capture["selection_sha256"],
            "evidence_split": evidence_split_sha256,
        }
    except (KeyError, TypeError) as exc:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "Parent cache lacks a complete derived-phase lineage."
        ) from exc
    for name, value in dependencies.items():
        _digest(value, name)
    return dependencies


def _callable_source_fingerprint(value: Callable[..., Any], *, role: str) -> str:
    """Bind a derived cache key to portable, exact Python source identity.

    ``code.co_code`` is an interpreter artifact rather than a source identity:
    Python 3.11 and 3.12 compile the same authenticated function text to
    different bytecode.  Including it made a phase built and reviewed on one
    interpreter impossible to authenticate on another even when every source
    module byte was identical.  The normalized callable source below and the
    complete implementation-module fingerprints in
    :func:`_global_algorithm_fingerprint` already bind the implementation.
    """

    target = getattr(value, "__func__", value)
    try:
        source = textwrap.dedent(inspect.getsource(target)).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"Cannot authenticate source for derived-cache algorithm {role}."
        ) from exc
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-global-callable-source-v2",
            "role": role,
            "module": str(getattr(target, "__module__", "")),
            "qualname": str(getattr(target, "__qualname__", "")),
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    )


def _module_file_fingerprint(path_value: Any, *, role: str) -> str:
    """Bind transitive helpers by hashing the complete implementation module."""

    if not isinstance(path_value, str) or not path_value:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"Cannot locate derived-cache implementation module {role}."
        )
    try:
        path = Path(path_value).resolve(strict=True)
        raw = path.read_bytes()
    except OSError as exc:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"Cannot authenticate derived-cache implementation module {role}."
        ) from exc
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-global-module-source-v1",
            "role": role,
            "module_file_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _property_getter(owner: type[Any], name: str) -> Callable[..., Any]:
    """Return one concrete property getter for interface source binding."""

    descriptor = inspect.getattr_static(owner, name, None)
    if not isinstance(descriptor, property) or descriptor.fget is None:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"Trace-cache topology interface property {owner.__name__}.{name} "
            "is unavailable."
        )
    return descriptor.fget


def _trace_cache_topology_interface_fingerprint() -> str:
    """Bind only the sealed parent-row reader used by topology derivation.

    Child replay capture and verification share the circuit-trace module for
    historical reasons, but they do not participate in the B1/B2 global
    topology phases.  Hashing that entire module invalidated expensive parent
    evidence whenever an unrelated child-replay diagnostic changed.  This
    contract instead names the exact cache/session ABI, public accessors, and
    transitive authentication helpers used to read topology evidence.
    """

    module = _circuit_trace_cache_module
    cache_type = module.Lfm2CircuitTraceCache
    session_type = module.Lfm2CircuitTraceSelectiveAuthenticatedSession
    reference_type = module.Lfm2CircuitTraceSelectiveBatchReference
    payload_type = module.Lfm2CircuitTraceSelectedBatchPayload
    identity_type = module._Lfm2CircuitTraceFileIdentity
    selective_state_type = module._Lfm2SelectiveBatchState

    callables: dict[str, Callable[..., Any]] = {
        "cache.manifest": _property_getter(cache_type, "manifest"),
        "cache.cache_key": _property_getter(cache_type, "cache_key"),
        "cache.parent_fingerprint": _property_getter(
            cache_type,
            "parent_fingerprint",
        ),
        "cache.corpus_fingerprint": _property_getter(
            cache_type,
            "corpus_fingerprint",
        ),
        "cache.implementation_fingerprint": _property_getter(
            cache_type,
            "implementation_fingerprint",
        ),
        "cache.layer_topology_fingerprint": _property_getter(
            cache_type,
            "layer_topology_fingerprint",
        ),
        "cache.batch_count": _property_getter(cache_type, "batch_count"),
        "cache.len": cache_type.__len__,
        "cache.selective_authenticated_session": (
            cache_type.selective_authenticated_session
        ),
        "session.init": session_type.__init__,
        "session.path": _property_getter(session_type, "path"),
        "session.manifest": _property_getter(session_type, "manifest"),
        "session.cache_key": _property_getter(session_type, "cache_key"),
        "session.batch_count": _property_getter(session_type, "batch_count"),
        "session.evidence_sha256": _property_getter(
            session_type,
            "evidence_sha256",
        ),
        "session.batch_references": _property_getter(
            session_type,
            "batch_references",
        ),
        "session.len": session_type.__len__,
        "session.open_batch_state": session_type._open_batch_state,
        "session.close_batch_state": session_type._close_batch_state,
        "session.verify_batch_identity": session_type._verify_batch_identity,
        "session.verify_immutable_metadata": (session_type._verify_immutable_metadata),
        "session.seal_evidence": session_type._seal_evidence,
        "session.state": session_type._state,
        "session.load_selected_tensors": session_type.load_selected_tensors,
        "session.release_selected_tensors": (session_type.release_selected_tensors),
        "session.verify": session_type.verify,
        "session.close": session_type.close,
        "session.enter": session_type.__enter__,
        "session.exit": session_type.__exit__,
        "reference.record": _property_getter(reference_type, "record"),
        "payload.init": payload_type.__init__,
        "payload.released": _property_getter(payload_type, "released"),
        "payload.tensors": _property_getter(payload_type, "tensors"),
        "payload.release": payload_type._release,
        "identity.from_stat": identity_type.from_stat,
        "helper.torch": module._torch,
        "helper.safetensors": module._safetensors,
        "helper.canonical_json": module._canonical_json,
        "helper.json_sha256": module._json_sha256,
        "helper.file_sha256": module._file_sha256,
        "helper.strict_pairs": module._strict_pairs,
        "helper.read_canonical_json": module._read_canonical_json,
        "helper.verify_seal": module._verify_seal,
        "helper.tensor_bytes": module._tensor_bytes,
        "helper.tensor_sha256": module._tensor_sha256,
        "helper.tensor_descriptor": module._tensor_descriptor,
        "helper.cpu_tensor": module._cpu_tensor,
        "helper.source_file_sha256": module._source_file_sha256,
        "helper.package_version": module._package_version,
        "helper.implementation_identity": module.implementation_identity,
        "helper.implementation_identity_consistency": (
            module._implementation_identity_is_internally_consistent
        ),
        "helper.compatible_implementation_identity": (
            module._compatible_implementation_identity
        ),
        "helper.explicit_implementation_compatibility": (
            module.permits_explicit_implementation_compatibility
        ),
        "helper.validate_manifest": module._validate_manifest,
        "helper.validate_completion": module._validate_completion,
        "helper.verify_cache_layout": module._verify_cache_layout,
        "helper.fresh_verified_cache_state": module._fresh_verified_cache_state,
        "helper.path_file_identity": module._path_file_identity,
        "helper.handle_file_identity": module._handle_file_identity,
    }
    abi = {
        "schema": GLOBAL_TOPOLOGY_TRACE_CACHE_INTERFACE_SCHEMA,
        "cache_schema": module.CACHE_SCHEMA,
        "batch_schema": module.BATCH_SCHEMA,
        "completion_schema": module.COMPLETION_SCHEMA,
        "selective_session_schema": module.SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
        "selected_batch_payload_schema": module.SELECTIVE_BATCH_PAYLOAD_SCHEMA,
        "manifest_filename": module.MANIFEST_FILENAME,
        "completion_filename": module.COMPLETION_FILENAME,
        "trace_filename": module.TRACE_FILENAME,
        "batch_record_filename": module.BATCH_RECORD_FILENAME,
        "sha256_hex_length": module._SHA256_LENGTH,
        "cache_fields": list(cache_type.__dataclass_fields__),
        "selective_session_slots": list(session_type.__slots__),
        "selective_batch_state_fields": list(selective_state_type.__dataclass_fields__),
        "selective_reference_fields": list(reference_type.__dataclass_fields__),
        "selected_payload_slots": list(payload_type.__slots__),
        "file_identity_fields": list(identity_type.__dataclass_fields__),
        "topology_consumers": [
            GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        ],
        "child_replay_interface_included": False,
    }
    return canonical_sha256(
        {
            **abi,
            "callable_source_sha256s": {
                role: _callable_source_fingerprint(value, role=role)
                for role, value in sorted(callables.items())
            },
        }
    )


def _torch_numeric_runtime_contract() -> dict[str, Any]:
    """Describe every ambient CPU/PyTorch setting that can alter cached math."""

    build_config = torch.__config__.show().replace("\r\n", "\n")
    return {
        "schema": "neural-forge-lfm2-global-torch-numeric-runtime-v1",
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "")),
        "torch_build_config_sha256": hashlib.sha256(
            build_config.encode("utf-8")
        ).hexdigest(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "mkldnn_deterministic": bool(
            getattr(torch.backends.mkldnn, "deterministic", False)
        ),
    }


def _global_algorithm_fingerprint() -> str:
    algorithms = {
        "enumerate_hidden_sizes": enumerate_lfm2_global_hidden_sizes,
        "derive_width_policy": derive_lfm2_automatic_mlp_width_policy,
        "build_capacity_bank": (build_lfm2_cached_heterogeneous_capacity_option_bank),
        "build_functional_width": build_lfm2_cached_functional_width_evidence,
        "retarget_capacity_bank": _retarget_capacity_option_bank,
        "target_pins_for_width": _target_pins_for_width,
        "build_target_pins": build_lfm2_heterogeneous_target_pins,
        "build_attention_option": build_lfm2_attention_layer_option,
        "build_short_conv_option": build_lfm2_short_conv_layer_option,
        "exact_selector": Lfm2HeterogeneousLayoutSelectionInput.select,
        "select_global_report": select_lfm2_global_topology_report,
        "restore_capacity_bank": _restore_capacity_option_bank,
        "restore_functional_width": _restore_functional_width_evidence,
        "restore_candidate": _restore_candidate,
        "restore_rejection": _restore_rejection,
    }
    implementation_modules = {
        "global_frontier": __file__,
        "cached_functional_width": _cached_functional_width_module.__file__,
        "cached_heterogeneous_capacity": (
            _cached_heterogeneous_capacity_module.__file__
        ),
        "heterogeneous_layout_selection": (
            _heterogeneous_layout_selection_module.__file__
        ),
        "dynamic_parameter_frontier": _dynamic_parameter_frontier_module.__file__,
        "dynamic_fold_phase_cache": _dynamic_fold_phase_cache_module.__file__,
        "retained_trace_rows": _retained_trace_rows_module.__file__,
    }
    return canonical_sha256(
        {
            "schema": GLOBAL_TOPOLOGY_PHASE_CACHE_ABI,
            "capacity_option_bank_schema": CAPACITY_OPTION_BANK_SCHEMA,
            "functional_width_schema": LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA,
            "functional_score_schema": LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA,
            "functional_sampling_policy": LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
            "mlp_capacity_risk_method": MLP_CAPACITY_RISK_METHOD,
            "attention_geometry_risk_method": ATTENTION_GEOMETRY_RISK_METHOD,
            "short_conv_writer_risk_method": (SHORT_CONV_WRITER_ACTION_RISK_METHOD),
            "global_score_method": GLOBAL_TOPOLOGY_SCORE_METHOD,
            "retargeted_capacity_schema": RETARGETED_CAPACITY_EVIDENCE_SCHEMA,
            "layer_option_schema": LFM2_HETEROGENEOUS_LAYER_OPTION_SCHEMA,
            "trace_cache_topology_interface_sha256": (
                _trace_cache_topology_interface_fingerprint()
            ),
            "callable_source_sha256s": {
                role: _callable_source_fingerprint(value, role=role)
                for role, value in sorted(algorithms.items())
            },
            "implementation_module_sha256s": {
                role: _module_file_fingerprint(path, role=role)
                for role, path in sorted(implementation_modules.items())
            },
        }
    )


def _capacity_phase_key(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    *,
    runtime: Lfm2GlobalTargetRuntimePins,
    exact_target_unique_parameters: int,
    candidate_hidden_sizes: tuple[int, ...],
    union_mlp_widths: tuple[int, ...],
    union_writer_ranks: tuple[int, ...],
    short_conv_internal_widths: tuple[int, ...],
    attention_head_options: tuple[tuple[int, int, int], ...],
    discover_exact_short_conv_writer_ranks: bool,
    required_target_hidden_size: int | None = None,
) -> FoldPhaseKey:
    numeric_runtime = _torch_numeric_runtime_contract()
    dependencies = _parent_phase_dependencies(
        parent_cache,
        evidence_split_sha256=context.evidence_split_sha256,
    )
    dependencies.update(
        {
            "source_graph": source_graph.fingerprint,
            "target_runtime": runtime.fingerprint,
            "global_topology_algorithms": _global_algorithm_fingerprint(),
            "candidate_hidden_sizes": canonical_sha256(list(candidate_hidden_sizes)),
            "union_mlp_widths": canonical_sha256(list(union_mlp_widths)),
            "union_writer_ranks": canonical_sha256(list(union_writer_ranks)),
            "torch_numeric_runtime": canonical_sha256(numeric_runtime),
        }
    )
    return FoldPhaseKey(
        phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        dependency_sha256s=dependencies,
        numeric_contract={
            "exact_target_unique_parameters": exact_target_unique_parameters,
            "required_target_hidden_size": required_target_hidden_size,
            "candidate_hidden_sizes": list(candidate_hidden_sizes),
            "union_mlp_widths": list(union_mlp_widths),
            "union_writer_ranks": list(union_writer_ranks),
            "short_conv_internal_widths": list(short_conv_internal_widths),
            "attention_head_options": [list(value) for value in attention_head_options],
            "target_hidden_size": max(candidate_hidden_sizes),
            "target_vocabulary_size": runtime.vocab_size,
            "conv_kernel_size": runtime.conv_kernel_size,
            "attention_geometry_risk_enabled": True,
            "attention_geometry_score_query_chunk_size": 32,
            "discover_exact_short_conv_writer_ranks": (
                discover_exact_short_conv_writer_ranks
            ),
            "emit_factor_writer_options": False,
            "accumulation_dtype": "torch.float64",
            "numeric_runtime": numeric_runtime,
        },
        abi={
            "schema": GLOBAL_TOPOLOGY_PHASE_CACHE_ABI,
            "payload_schema": GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
            "capacity_option_bank_schema": CAPACITY_OPTION_BANK_SCHEMA,
            "strict_typed_restore": True,
            "sealed_parent_cache_only": True,
            "parent_model_calls": 0,
            "training_steps": 0,
        },
    )


def _functional_phase_key(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    *,
    runtime: Lfm2GlobalTargetRuntimePins,
    candidate_hidden_sizes: tuple[int, ...],
    maximum_rows_per_role: int,
    required_target_hidden_size: int | None = None,
) -> FoldPhaseKey:
    numeric_runtime = _torch_numeric_runtime_contract()
    dependencies = _parent_phase_dependencies(
        parent_cache,
        evidence_split_sha256=context.evidence_split_sha256,
    )
    dependencies.update(
        {
            "source_graph": source_graph.fingerprint,
            "target_runtime": runtime.fingerprint,
            "global_topology_algorithms": _global_algorithm_fingerprint(),
            "candidate_hidden_sizes": canonical_sha256(list(candidate_hidden_sizes)),
            "torch_numeric_runtime": canonical_sha256(numeric_runtime),
        }
    )
    return FoldPhaseKey(
        phase_name=GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        dependency_sha256s=dependencies,
        numeric_contract={
            "required_target_hidden_size": required_target_hidden_size,
            "candidate_hidden_sizes": list(candidate_hidden_sizes),
            "maximum_rows_per_role": maximum_rows_per_role,
            "source_hidden_size": int(dict(source_graph.metadata)["hidden_size"]),
            "canonical_roles": [
                "nonlinear_mlp_write",
                "operator_write",
                "residual_context",
                "terminal_context",
            ],
            "row_l2_normalized_before_gram": True,
            "role_grams_normalized_to_unit_trace": True,
            "roles_weighted_equally": True,
            "accumulation_dtype": "torch.float64",
            "gram_accumulation_schema": (
                LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA
            ),
            "gram_row_batch_size": LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
            "gram_accumulation_threads": 1,
            "eigensolver_threads": 1,
            "numeric_runtime": numeric_runtime,
        },
        abi={
            "schema": GLOBAL_TOPOLOGY_PHASE_CACHE_ABI,
            "payload_schema": GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
            "functional_width_schema": LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA,
            "functional_score_schema": LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA,
            "sampling_policy": LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
            "strict_typed_restore": True,
            "sealed_parent_cache_only": True,
            "parent_model_calls": 0,
            "training_steps": 0,
        },
    )


def _selection_phase_key(
    context: Lfm2GlobalTopologyPhaseCacheContext,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    *,
    runtime: Lfm2GlobalTargetRuntimePins,
    exact_target_unique_parameters: int,
    candidate_hidden_sizes: tuple[int, ...],
    policies: Mapping[int, Lfm2MlpWidthOptionPolicy],
    base_bank: Lfm2CachedHeterogeneousCapacityOptionBank,
    functional: Lfm2CachedFunctionalWidthEvidence,
    max_states: int,
    required_target_hidden_size: int | None = None,
) -> FoldPhaseKey:
    numeric_runtime = _torch_numeric_runtime_contract()
    dependencies = _parent_phase_dependencies(
        parent_cache,
        evidence_split_sha256=context.evidence_split_sha256,
    )
    dependencies.update(
        {
            "source_graph": source_graph.fingerprint,
            "target_runtime": runtime.fingerprint,
            "global_topology_algorithms": _global_algorithm_fingerprint(),
            "base_capacity_option_bank": base_bank.evidence_sha256,
            "functional_width_evidence": functional.evidence_sha256,
            "candidate_hidden_sizes": canonical_sha256(list(candidate_hidden_sizes)),
            "ordered_policy_lattices": canonical_sha256(
                [
                    {
                        "hidden_size": hidden,
                        "policy": policy.policy_id,
                        "lattice_sha256": policy.exact_budget_lattice_sha256,
                    }
                    for hidden, policy in sorted(policies.items())
                ]
            ),
            "torch_numeric_runtime": canonical_sha256(numeric_runtime),
        }
    )
    return FoldPhaseKey(
        phase_name=GLOBAL_TOPOLOGY_SELECTION_PHASE,
        dependency_sha256s=dependencies,
        numeric_contract={
            "exact_target_unique_parameters": exact_target_unique_parameters,
            "required_target_hidden_size": required_target_hidden_size,
            "candidate_hidden_sizes": list(candidate_hidden_sizes),
            "max_states": max_states,
            "complete_full_depth_child_per_feasible_width": True,
            "same_exact_parameter_count_per_feasible_width": True,
            "numeric_runtime": numeric_runtime,
        },
        abi={
            "schema": GLOBAL_TOPOLOGY_PHASE_CACHE_ABI,
            "payload_schema": GLOBAL_TOPOLOGY_SELECTION_PAYLOAD_SCHEMA,
            "frontier_schema": GLOBAL_TOPOLOGY_FRONTIER_SCHEMA,
            "candidate_schema": GLOBAL_TOPOLOGY_CANDIDATE_SCHEMA,
            "rejection_schema": GLOBAL_TOPOLOGY_REJECTION_SCHEMA,
            "score_method": GLOBAL_TOPOLOGY_SCORE_METHOD,
            "selected_layout_reconstructed_and_reauthenticated_on_hit": True,
            "parameter_count_used_as_quality_score": False,
            "training_steps": 0,
        },
    )


def _restore_capacity_option_bank(
    value: Any,
) -> Lfm2CachedHeterogeneousCapacityOptionBank:
    try:
        raw = _strict_mapping(
            value,
            expected_fields={
                "schema",
                "source_graph_fingerprint",
                "parent_cache_key",
                "mlp_option_policy",
                "attention_geometry_policy",
                "exact_budget_lattice_sha256",
                "layer_evidence",
                "layer_options",
                "selection_contract",
                "evidence_sha256",
            },
            label="cached capacity option bank",
        )
        if raw["schema"] != CAPACITY_OPTION_BANK_SCHEMA:
            raise FoldPhaseCacheIntegrityError(
                "Cached capacity option-bank schema is unsupported."
            )
        layer_rows = raw["layer_evidence"]
        option_rows = raw["layer_options"]
        if not isinstance(layer_rows, list) or not isinstance(option_rows, list):
            raise FoldPhaseCacheIntegrityError(
                "Cached capacity bank nested inventories are invalid."
            )
        layers = []
        for row in layer_rows:
            layer = _strict_mapping(
                row,
                expected_fields={
                    "layer_index",
                    "layer_type",
                    "source_layer_sha256",
                    "operator_delta_energy_fraction",
                    "mlp_delta_energy_fraction",
                    "sampled_scalar_count",
                    "nonlinear_sampled_row_count",
                    "source_mlp_intermediate_size",
                    "mlp_capacity_risks",
                    "attention_geometry_risks",
                    "short_conv_writer_risks",
                    "selection_evidence_sha256",
                },
                label="cached layer capacity evidence",
            )
            mlp_rows = layer.pop("mlp_capacity_risks")
            attention_rows = layer.pop("attention_geometry_risks")
            writer_rows = layer.pop("short_conv_writer_risks")
            if not all(
                isinstance(rows, list)
                for rows in (mlp_rows, attention_rows, writer_rows)
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Cached layer risk inventories are invalid."
                )
            layers.append(
                Lfm2CachedLayerCapacityEvidence(
                    **layer,
                    mlp_capacity_risks=tuple(
                        Lfm2CachedMlpCapacityRisk(
                            **_strict_mapping(
                                item,
                                expected_fields={
                                    "intermediate_size",
                                    "source_intermediate_size",
                                    "fold_method_id",
                                    "post_swiglu_tail_fraction",
                                    "signed_regime_tail_fraction",
                                    "tangent_tail_fraction",
                                    "downstream_weighted_tail_fraction",
                                    "collision_fraction",
                                    "predicted_defect",
                                },
                                label="cached MLP capacity risk",
                            )
                        )
                        for item in mlp_rows
                    ),
                    attention_geometry_risks=tuple(
                        Lfm2CachedAttentionGeometryRisk(
                            **_strict_mapping(
                                item,
                                expected_fields={
                                    "query_heads",
                                    "key_value_heads",
                                    "head_dim",
                                    "source_query_heads",
                                    "source_key_value_heads",
                                    "source_head_dim",
                                    "fold_method_id",
                                    "coupled_group_tail_fraction",
                                    "query_offset_tail_fraction",
                                    "causal_score_tail_fraction",
                                    "context_tail_fraction",
                                    "key_value_memory_tail_fraction",
                                    "predicted_defect",
                                    "causal_query_row_count",
                                    "cached_key_value_row_count",
                                },
                                label="cached attention geometry risk",
                            )
                        )
                        for item in attention_rows
                    ),
                    short_conv_writer_risks=tuple(
                        Lfm2CachedShortConvWriterRisk(
                            **_strict_mapping(
                                item,
                                expected_fields={
                                    "writer_rank",
                                    "source_output_width",
                                    "fold_method_id",
                                    "output_action_tail_fraction",
                                    "sampling_uncertainty_fraction",
                                    "regularized_tail_fraction",
                                    "observed_effective_rank",
                                    "predicted_defect",
                                    "sampled_output_row_count",
                                },
                                label="cached ShortConv writer risk",
                            )
                        )
                        for item in writer_rows
                    ),
                )
            )
        restored = Lfm2CachedHeterogeneousCapacityOptionBank(
            source_graph_fingerprint=raw["source_graph_fingerprint"],
            parent_cache_key=raw["parent_cache_key"],
            mlp_option_policy=raw["mlp_option_policy"],
            attention_geometry_policy=raw["attention_geometry_policy"],
            exact_budget_lattice_sha256=raw["exact_budget_lattice_sha256"],
            layer_evidence=tuple(layers),
            layer_options=tuple(
                Lfm2AuthenticatedLayerPhysicalOption.restore(item)
                for item in option_rows
            ),
            evidence_sha256=raw["evidence_sha256"],
        )
        if dict(restored.as_dict()) != raw:
            raise FoldPhaseCacheIntegrityError(
                "Cached capacity option bank did not round-trip canonically."
            )
        return restored
    except FoldPhaseCacheIntegrityError:
        raise
    except Exception as exc:
        raise FoldPhaseCacheIntegrityError(
            "Cached capacity option bank failed typed authentication."
        ) from exc


def _restore_functional_width_evidence(
    value: Any,
) -> Lfm2CachedFunctionalWidthEvidence:
    try:
        raw = _strict_mapping(
            value,
            expected_fields={
                "schema",
                "source_graph_fingerprint",
                "parent_cache_key",
                "selective_session_sha256",
                "source_hidden_size",
                "sampling",
                "role_gram_sha256",
                "functional_eigenspectrum_sha256",
                "candidate_scores",
                "selection_contract",
                "evidence_sha256",
            },
            label="cached functional-width evidence",
        )
        if raw["schema"] != LFM2_CACHED_FUNCTIONAL_WIDTH_SCHEMA:
            raise FoldPhaseCacheIntegrityError(
                "Cached functional-width schema is unsupported."
            )
        sampling = _strict_mapping(
            raw["sampling"],
            expected_fields={
                "policy",
                "maximum_rows_per_role",
                "sampled_rows_by_role",
                "selection_manifest_sha256",
                "row_l2_normalized_before_gram",
                "role_grams_normalized_to_unit_trace",
                "roles_weighted_equally",
                "raw_residual_energy_used_as_width_score",
                "gram_accumulation",
            },
            label="cached functional sampling evidence",
        )
        rows = sampling["sampled_rows_by_role"]
        grams = raw["role_gram_sha256"]
        scores = raw["candidate_scores"]
        if (
            not isinstance(rows, Mapping)
            or not isinstance(grams, Mapping)
            or not isinstance(scores, list)
        ):
            raise FoldPhaseCacheIntegrityError(
                "Cached functional-width nested evidence is invalid."
            )
        restored_scores = []
        for item in scores:
            score = _strict_mapping(
                item,
                expected_fields={
                    "schema",
                    "hidden_size",
                    "role_balanced_functional_retention",
                    "role_balanced_functional_loss",
                    "functional_eigenspectrum_sha256",
                    "selection_evidence_sha256",
                    "score_sha256",
                },
                label="cached functional-width score",
            )
            if score.pop("schema") != LFM2_FUNCTIONAL_WIDTH_SCORE_SCHEMA:
                raise FoldPhaseCacheIntegrityError(
                    "Cached functional score schema is unsupported."
                )
            restored_scores.append(Lfm2FunctionalWidthScore(**score))
        roles = (
            "nonlinear_mlp_write",
            "operator_write",
            "residual_context",
            "terminal_context",
        )
        if tuple(rows) != roles or tuple(grams) != roles:
            raise FoldPhaseCacheIntegrityError(
                "Cached functional role ordering is noncanonical."
            )
        restored = Lfm2CachedFunctionalWidthEvidence(
            source_graph_fingerprint=raw["source_graph_fingerprint"],
            parent_cache_key=raw["parent_cache_key"],
            selective_session_sha256=raw["selective_session_sha256"],
            source_hidden_size=raw["source_hidden_size"],
            maximum_rows_per_role=sampling["maximum_rows_per_role"],
            sampled_rows_by_role=tuple((role, rows[role]) for role in roles),
            candidate_scores=tuple(restored_scores),
            role_gram_sha256=tuple((role, grams[role]) for role in roles),
            functional_eigenspectrum_sha256=raw["functional_eigenspectrum_sha256"],
            selection_manifest_sha256=sampling["selection_manifest_sha256"],
            evidence_sha256=raw["evidence_sha256"],
        )
        if dict(restored.as_dict()) != raw:
            raise FoldPhaseCacheIntegrityError(
                "Cached functional-width evidence did not round-trip canonically."
            )
        return restored
    except FoldPhaseCacheIntegrityError:
        raise
    except Exception as exc:
        raise FoldPhaseCacheIntegrityError(
            "Cached functional-width evidence failed typed authentication."
        ) from exc


def _validate_capacity_option_bank_context(
    bank: Lfm2CachedHeterogeneousCapacityOptionBank,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
) -> None:
    """Require restored B1 evidence to name the active graph and cache."""

    if (
        bank.source_graph_fingerprint != source_graph.fingerprint
        or bank.parent_cache_key != parent_cache.cache_key
    ):
        raise FoldPhaseCacheIntegrityError(
            "Cached capacity option bank is bound to a different source "
            "graph or parent cache."
        )


def _validate_functional_width_context(
    evidence: Lfm2CachedFunctionalWidthEvidence,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    candidate_hidden_sizes: Sequence[int],
) -> None:
    """Require restored B2 evidence to match every active width input."""

    metadata = dict(source_graph.metadata)
    source_hidden_size = metadata.get("hidden_size")
    score_widths = tuple(score.hidden_size for score in evidence.candidate_scores)
    expected_widths = tuple(candidate_hidden_sizes)
    if (
        evidence.source_graph_fingerprint != source_graph.fingerprint
        or evidence.parent_cache_key != parent_cache.cache_key
        or type(source_hidden_size) is not int
        or evidence.source_hidden_size != source_hidden_size
        or score_widths != expected_widths
    ):
        raise FoldPhaseCacheIntegrityError(
            "Cached functional-width evidence is bound to different source, "
            "parent-cache, hidden-width, or candidate-width inputs."
        )


def _restore_candidate(value: Any) -> Lfm2GlobalTopologyCandidateReport:
    raw = _strict_mapping(
        value,
        expected_fields={
            "schema",
            "hidden_size",
            "exact_unique_parameters",
            "target_layer_count",
            "role_balanced_functional_retention",
            "role_balanced_functional_loss",
            "mean_selected_component_risk",
            "combined_predicted_risk",
            "score_method",
            "functional_score_sha256",
            "mlp_width_lattice_sha256",
            "capacity_option_bank_sha256",
            "selection_input_sha256",
            "descriptor_sha256",
            "layout_selection_receipt_sha256",
            "parameter_count_used_as_quality_score",
            "candidate_sha256",
        },
        label="cached global topology candidate",
    )
    if (
        raw.pop("schema") != GLOBAL_TOPOLOGY_CANDIDATE_SCHEMA
        or raw.pop("score_method") != GLOBAL_TOPOLOGY_SCORE_METHOD
        or raw.pop("parameter_count_used_as_quality_score") is not False
    ):
        raise FoldPhaseCacheIntegrityError(
            "Cached global topology candidate contract is invalid."
        )
    try:
        return Lfm2GlobalTopologyCandidateReport(**raw)
    except Exception as exc:
        raise FoldPhaseCacheIntegrityError(
            "Cached global topology candidate failed authentication."
        ) from exc


def _restore_rejection(value: Any) -> Lfm2GlobalTopologyRejectedWidth:
    raw = _strict_mapping(
        value,
        expected_fields={
            "schema",
            "hidden_size",
            "reason_class",
            "reason",
            "rejection_sha256",
        },
        label="cached global topology rejection",
    )
    if raw.pop("schema") != GLOBAL_TOPOLOGY_REJECTION_SCHEMA:
        raise FoldPhaseCacheIntegrityError(
            "Cached global topology rejection schema is invalid."
        )
    try:
        return Lfm2GlobalTopologyRejectedWidth(**raw)
    except Exception as exc:
        raise FoldPhaseCacheIntegrityError(
            "Cached global topology rejection failed authentication."
        ) from exc


def _source_layer_contract(
    source_graph: ModelCircuitGraph,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    discovered = discover_lfm2_source_layers(source_graph)
    nodes = {value.node_id: value for value in source_graph.nodes}
    layer_types = tuple(value.layer_type for value in discovered)
    widths = []
    for layer in discovered:
        node = nodes[layer.mlp_node_id]
        attributes = dict(node.attributes)
        width = attributes.get("intermediate_size")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            producer_widths = {
                parameter.shape[0]
                for parameter in node.parameters
                if parameter.role in {"mlp.w1.weight", "mlp.w3.weight"}
                and len(parameter.shape) == 2
            }
            writer_widths = {
                parameter.shape[1]
                for parameter in node.parameters
                if parameter.role == "mlp.w2.weight" and len(parameter.shape) == 2
            }
            inferred = producer_widths & writer_widths
            if len(inferred) != 1:
                raise Lfm2CachedGlobalTopologyFrontierError(
                    "Cannot discover one source SwiGLU width for layer "
                    f"{layer.ordinal}."
                )
            width = inferred.pop()
        widths.append(width)
    if len(set(widths)) != 1:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The current LFM2 runtime requires one configured source MLP width; "
            "the inspected graph exposes heterogeneous source widths."
        )
    return layer_types, tuple(widths)


def _retarget_capacity_option_bank(
    base: Lfm2CachedHeterogeneousCapacityOptionBank,
    policy: Lfm2MlpWidthOptionPolicy,
    *,
    target_hidden_size: int,
) -> Lfm2CachedHeterogeneousCapacityOptionBank:
    """Filter one union cache scan into an authenticated per-H option bank."""

    hidden = _positive_int(target_hidden_size, "target_hidden_size")
    if base.mlp_option_policy != EXPLICIT_MLP_OPTION_POLICY:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The global frontier base bank must be an explicit union scan."
        )
    if base.attention_geometry_policy != ATTENTION_GEOMETRY_ENABLED_POLICY:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The residual-attention frontier requires an authenticated cached "
            "attention-geometry option bank."
        )
    if policy.policy_id != AUTOMATIC_MLP_OPTION_POLICY or len(
        policy.widths_by_layer
    ) != len(base.layer_evidence):
        raise Lfm2CachedGlobalTopologyFrontierError(
            "Automatic MLP policy differs from the union capacity bank layers."
        )
    filtered_rows = []
    for evidence, widths in zip(
        base.layer_evidence,
        policy.widths_by_layer,
        strict=True,
    ):
        risk_by_width = {
            value.intermediate_size: value for value in evidence.mlp_capacity_risks
        }
        missing = tuple(width for width in widths if width not in risk_by_width)
        if missing:
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Union nonlinear scan lacks automatic widths "
                f"{missing} for layer {evidence.layer_index}."
            )
        risks = tuple(risk_by_width[width] for width in widths)
        writer_risk_by_rank = {
            value.writer_rank: value for value in evidence.short_conv_writer_risks
        }
        writer_risks = tuple(
            writer_risk_by_rank[rank]
            for rank in policy.short_conv_writer_ranks
            if rank in writer_risk_by_rank
        )
        if evidence.layer_type == "conv" and len(writer_risks) != len(
            policy.short_conv_writer_ranks
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Union writer scan lacks a policy rank for ShortConv layer "
                f"{evidence.layer_index}."
            )
        filtered_rows.append((evidence, widths, risks, writer_risks))

    factor_source_sha256 = None
    factor_site_evidence_sha256 = None
    pressure_by_source: dict[str, float] = {}
    if policy.short_conv_writer_ranks:
        conv_evidence = tuple(
            evidence
            for evidence, _, _, _ in filtered_rows
            if evidence.layer_type == "conv"
        )
        if not conv_evidence:
            raise Lfm2CachedGlobalTopologyFrontierError(
                "A factor-writer policy requires at least one ShortConv layer."
            )
        maximum_pressure = max(
            value.operator_delta_energy_fraction for value in conv_evidence
        )
        pressure_by_source = {
            value.source_layer_sha256: (
                0.0
                if maximum_pressure <= 0.0
                else value.operator_delta_energy_fraction / maximum_pressure
            )
            for value in conv_evidence
        }
        site_scores = {}
        for evidence, _, _, writer_risks in filtered_rows:
            if evidence.layer_type != "conv":
                continue
            site_scores[evidence.source_layer_sha256] = (
                pressure_by_source[evidence.source_layer_sha256]
                * sum(value.predicted_defect for value in writer_risks)
                / len(writer_risks)
            )
        factor_source_sha256 = min(
            site_scores,
            key=lambda source: (site_scores[source], source),
        )
        factor_site_evidence_sha256 = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-retargeted-factor-writer-site-selection-v1"
                ),
                "base_capacity_option_bank_sha256": base.evidence_sha256,
                "exact_budget_lattice_sha256": policy.exact_budget_lattice_sha256,
                "target_hidden_size": hidden,
                "candidate_source_scores": site_scores,
                "selected_source_layer_sha256": factor_source_sha256,
                "writer_ranks": list(policy.short_conv_writer_ranks),
                "selection_method": (
                    "minimum-policy-rank-cached-output-action-defect-times-"
                    "operator-pressure"
                ),
                "hardcoded_layer_indices": False,
            }
        )

    layer_evidence = []
    allowed_by_source: dict[str, frozenset[int]] = {}
    evidence_by_source: dict[str, str] = {}
    writer_risk_by_source = {}
    for evidence, widths, risks, writer_risks in filtered_rows:
        evidence_body = {
            "schema": RETARGETED_CAPACITY_EVIDENCE_SCHEMA,
            "base_capacity_option_bank_sha256": base.evidence_sha256,
            "base_layer_evidence_sha256": evidence.selection_evidence_sha256,
            "automatic_mlp_option_policy": policy.policy_id,
            "attention_geometry_policy": base.attention_geometry_policy,
            "exact_budget_lattice_sha256": policy.exact_budget_lattice_sha256,
            "target_hidden_size": hidden,
            "layer_index": evidence.layer_index,
            "source_layer_sha256": evidence.source_layer_sha256,
            "mlp_capacity_risks": [value.as_dict() for value in risks],
            "attention_geometry_risks": [
                value.as_dict() for value in evidence.attention_geometry_risks
            ],
            "short_conv_writer_risks": [value.as_dict() for value in writer_risks],
            "factor_writer_candidate_site": (
                evidence.source_layer_sha256 == factor_source_sha256
            ),
            "factor_writer_site_evidence_sha256": (factor_site_evidence_sha256),
        }
        retargeted = Lfm2CachedLayerCapacityEvidence(
            layer_index=evidence.layer_index,
            layer_type=evidence.layer_type,
            source_layer_sha256=evidence.source_layer_sha256,
            operator_delta_energy_fraction=(evidence.operator_delta_energy_fraction),
            mlp_delta_energy_fraction=evidence.mlp_delta_energy_fraction,
            sampled_scalar_count=evidence.sampled_scalar_count,
            nonlinear_sampled_row_count=evidence.nonlinear_sampled_row_count,
            source_mlp_intermediate_size=evidence.source_mlp_intermediate_size,
            mlp_capacity_risks=risks,
            attention_geometry_risks=evidence.attention_geometry_risks,
            short_conv_writer_risks=writer_risks,
            selection_evidence_sha256=canonical_sha256(evidence_body),
        )
        layer_evidence.append(retargeted)
        allowed_by_source[evidence.source_layer_sha256] = frozenset(widths)
        evidence_by_source[evidence.source_layer_sha256] = (
            retargeted.selection_evidence_sha256
        )
        writer_risk_by_source[evidence.source_layer_sha256] = {
            value.writer_rank: value for value in writer_risks
        }

    options = []
    emitted_factor_ranks = set()
    for option in base.layer_options:
        if option.short_conv_writer_rank is not None:
            # The union scan may contain a provisional site for standalone
            # callers.  A global H candidate always reselects its own site.
            continue
        if (
            option.mlp_intermediate_size
            not in allowed_by_source[option.source_layer_sha256]
        ):
            continue
        option_evidence = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-retargeted-capacity-option-v2",
                "base_option_sha256": option.option_sha256,
                "base_option_evidence_sha256": (option.selection_evidence_sha256),
                "retargeted_layer_evidence_sha256": evidence_by_source[
                    option.source_layer_sha256
                ],
                "automatic_mlp_option_policy": policy.policy_id,
                "attention_geometry_policy": base.attention_geometry_policy,
                "exact_budget_lattice_sha256": (policy.exact_budget_lattice_sha256),
                "target_hidden_size": hidden,
                "factor_writer_site_evidence_sha256": (factor_site_evidence_sha256),
            }
        )
        if option.layer_type == "conv":
            assert option.short_conv_internal_width is not None
            options.append(
                build_lfm2_short_conv_layer_option(
                    source_layer_sha256=option.source_layer_sha256,
                    mlp_intermediate_size=option.mlp_intermediate_size,
                    internal_width=option.short_conv_internal_width,
                    writer_rank=option.short_conv_writer_rank,
                    selection_evidence_sha256=option_evidence,
                    structural_risk=option.structural_risk,
                )
            )
            if option.source_layer_sha256 == factor_source_sha256:
                internal = option.short_conv_internal_width
                for rank, writer_risk in writer_risk_by_source[
                    option.source_layer_sha256
                ].items():
                    if (
                        rank >= min(hidden, internal)
                        or rank * (hidden + internal) >= hidden * internal
                    ):
                        continue
                    factor_evidence = canonical_sha256(
                        {
                            "schema": (
                                "neural-forge-lfm2-retargeted-factor-writer-option-v1"
                            ),
                            "base_dense_option_sha256": option.option_sha256,
                            "retargeted_dense_option_evidence_sha256": (
                                option_evidence
                            ),
                            "retargeted_layer_evidence_sha256": evidence_by_source[
                                option.source_layer_sha256
                            ],
                            "factor_writer_site_evidence_sha256": (
                                factor_site_evidence_sha256
                            ),
                            "target_hidden_size": hidden,
                            "writer_risk": writer_risk.as_dict(),
                        }
                    )
                    options.append(
                        build_lfm2_short_conv_layer_option(
                            source_layer_sha256=option.source_layer_sha256,
                            mlp_intermediate_size=option.mlp_intermediate_size,
                            internal_width=internal,
                            writer_rank=rank,
                            selection_evidence_sha256=factor_evidence,
                            structural_risk=(
                                option.structural_risk
                                + pressure_by_source[option.source_layer_sha256]
                                * writer_risk.predicted_defect
                            ),
                        )
                    )
                    emitted_factor_ranks.add(rank)
        else:
            assert option.query_heads is not None
            assert option.key_value_heads is not None
            assert option.head_dim is not None
            options.append(
                build_lfm2_attention_layer_option(
                    source_layer_sha256=option.source_layer_sha256,
                    mlp_intermediate_size=option.mlp_intermediate_size,
                    query_heads=option.query_heads,
                    key_value_heads=option.key_value_heads,
                    head_dim=option.head_dim,
                    selection_evidence_sha256=option_evidence,
                    structural_risk=option.structural_risk,
                )
            )
    if emitted_factor_ranks != set(policy.short_conv_writer_ranks):
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The per-H factor site cannot represent every exact-lattice writer rank."
        )
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    values = {
        "source_graph_fingerprint": base.source_graph_fingerprint,
        "parent_cache_key": base.parent_cache_key,
        "mlp_option_policy": policy.policy_id,
        "attention_geometry_policy": base.attention_geometry_policy,
        "exact_budget_lattice_sha256": policy.exact_budget_lattice_sha256,
        "layer_evidence": tuple(layer_evidence),
        "layer_options": tuple(options),
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    return Lfm2CachedHeterogeneousCapacityOptionBank(
        **values,
        evidence_sha256=canonical_sha256(provisional._body()),
    )


def _target_pins_for_width(
    *,
    hidden_size: int,
    runtime: Lfm2GlobalTargetRuntimePins,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    base_bank: Lfm2CachedHeterogeneousCapacityOptionBank,
    selected_bank: Lfm2CachedHeterogeneousCapacityOptionBank,
    functional_evidence: Lfm2CachedFunctionalWidthEvidence,
    policy: Lfm2MlpWidthOptionPolicy,
    exact_target_unique_parameters: int,
) -> Any:
    runtime_body = {
        "schema": "neural-forge-lfm2-global-topology-runtime-evidence-v1",
        "runtime_pins_sha256": runtime.fingerprint,
        "source_graph_fingerprint": source_graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "base_union_capacity_option_bank_sha256": base_bank.evidence_sha256,
        "selected_capacity_option_bank_sha256": selected_bank.evidence_sha256,
        "functional_width_evidence_sha256": functional_evidence.evidence_sha256,
        "functional_width_score_sha256": (
            functional_evidence.score_for(hidden_size).score_sha256
        ),
        "mlp_width_lattice_sha256": policy.exact_budget_lattice_sha256,
        "attention_geometry_policy": selected_bank.attention_geometry_policy,
        "hidden_size": hidden_size,
        "exact_target_unique_parameters": exact_target_unique_parameters,
        "cached_parent_traces_only": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
    }
    return build_lfm2_heterogeneous_target_pins(
        hidden_size=hidden_size,
        vocab_size=runtime.vocab_size,
        conv_kernel_size=runtime.conv_kernel_size,
        rope_theta=runtime.rope_theta,
        norm_eps=runtime.norm_eps,
        max_position_embeddings=runtime.max_position_embeddings,
        use_cache=runtime.use_cache,
        attention_implementation=runtime.attention_implementation,
        bos_token_id=runtime.bos_token_id,
        eos_token_id=runtime.eos_token_id,
        pad_token_id=runtime.pad_token_id,
        block_auto_adjust_ff_dim=runtime.block_auto_adjust_ff_dim,
        target_vocab_evidence_sha256=runtime.target_vocab_evidence_sha256,
        runtime_evidence_sha256=canonical_sha256(runtime_body),
        token_bridge_evidence_sha256=runtime.token_bridge_evidence_sha256,
    )


def _policy_is_unattainable(error: Lfm2CachedHeterogeneousCapacityError) -> bool:
    message = str(error)
    return message.startswith(
        "The exact whole-child budget has no dense compact MLP/operator "
    ) or message.startswith(
        "The exact whole-child budget has no dense-or-single-factor compact "
        "MLP/operator "
    )


def build_lfm2_cached_global_topology_frontier(
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    *,
    runtime: Lfm2GlobalTargetRuntimePins,
    exact_target_unique_parameters: int,
    required_target_hidden_size: int | None = None,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
    retained_parent_rows: Any | None = None,
    defer_evidence_cache_miss: bool = False,
    minimum_hidden_size: int = 32,
    hidden_size_multiple: int = 8,
    short_conv_internal_widths: Sequence[int] = (704, 1024),
    discover_exact_short_conv_writer_ranks: bool = False,
    attention_head_options: Sequence[tuple[int, int, int]] = (
        (12, 6, 64),
        (16, 8, 64),
    ),
    maximum_functional_rows_per_role: int = 2048,
    max_states: int = 100_000,
    phase_cache_context: Lfm2GlobalTopologyPhaseCacheContext | None = None,
) -> Lfm2CachedGlobalTopologyFrontier:
    """Build and choose the complete exact child across ABI-legal widths."""

    if not isinstance(source_graph, ModelCircuitGraph) or source_graph.family != "lfm2":
        raise TypeError("source_graph must be an inspected LFM2 ModelCircuitGraph.")
    if not isinstance(parent_cache, Lfm2CircuitTraceCache):
        raise TypeError("parent_cache must be Lfm2CircuitTraceCache.")
    if type(runtime) is not Lfm2GlobalTargetRuntimePins:
        raise TypeError("runtime must be Lfm2GlobalTargetRuntimePins.")
    if selective_session is not None and not isinstance(
        selective_session,
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise TypeError(
            "selective_session must be an authenticated selective session or None."
        )
    if type(defer_evidence_cache_miss) is not bool:
        raise TypeError("defer_evidence_cache_miss must be bool.")
    if phase_cache_context is not None and not isinstance(
        phase_cache_context,
        Lfm2GlobalTopologyPhaseCacheContext,
    ):
        raise TypeError(
            "phase_cache_context must be Lfm2GlobalTopologyPhaseCacheContext or None."
        )
    target = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    required_hidden_size = (
        None
        if required_target_hidden_size is None
        else _positive_int(
            required_target_hidden_size,
            "required_target_hidden_size",
        )
    )
    state_ceiling = _positive_int(max_states, "max_states")
    functional_row_ceiling = _positive_int(
        maximum_functional_rows_per_role,
        "maximum_functional_rows_per_role",
    )
    conv_widths = tuple(
        sorted(
            {
                _positive_int(value, "short_conv_internal_width")
                for value in short_conv_internal_widths
            }
        )
    )
    attention_options = tuple(
        sorted(
            {
                (
                    _positive_int(query, "query_heads"),
                    _positive_int(key_value, "key_value_heads"),
                    _positive_int(head_dim, "head_dim"),
                )
                for query, key_value, head_dim in attention_head_options
            }
        )
    )
    legal_widths = enumerate_lfm2_global_hidden_sizes(
        source_graph,
        exact_target_unique_parameters=target,
        target_vocabulary_size=runtime.vocab_size,
        minimum_hidden_size=minimum_hidden_size,
        hidden_size_multiple=hidden_size_multiple,
    )
    if required_hidden_size is not None:
        if required_hidden_size not in legal_widths:
            raise Lfm2CachedGlobalTopologyFrontierUnattainable(
                "The required target hidden size is not legal for the exact "
                f"whole-child budget: H{required_hidden_size}."
            )
        widths = (required_hidden_size,)
    else:
        widths = legal_widths
    layer_types, source_widths = _source_layer_contract(source_graph)
    policies: dict[int, Lfm2MlpWidthOptionPolicy] = {}
    rejected = []
    for hidden_size in widths:
        try:
            policies[hidden_size] = derive_lfm2_automatic_mlp_width_policy(
                layer_types=layer_types,
                source_widths=source_widths,
                short_conv_internal_widths=conv_widths,
                attention_head_options=attention_options,
                target_hidden_size=hidden_size,
                target_vocab_size=runtime.vocab_size,
                exact_target_unique_parameters=target,
                conv_kernel_size=runtime.conv_kernel_size,
                discover_exact_short_conv_writer_ranks=(
                    discover_exact_short_conv_writer_ranks
                ),
            )
        except Lfm2CachedHeterogeneousCapacityError as error:
            if not _policy_is_unattainable(error):
                raise
            rejected.append(
                Lfm2GlobalTopologyRejectedWidth.build(
                    hidden_size,
                    "no_exact_integer_lattice",
                    str(error),
                )
            )
    if not policies:
        if required_hidden_size is not None:
            raise Lfm2CachedGlobalTopologyFrontierUnattainable(
                "The required target hidden size has no exact MLP/operator "
                f"integer lattice: H{required_hidden_size}."
            )
        raise Lfm2CachedGlobalTopologyFrontierUnattainable(
            "No legal residual width has an exact dense MLP/operator lattice."
        )
    union_widths = tuple(
        sorted(
            {
                width
                for policy in policies.values()
                for row in policy.widths_by_layer
                for width in row
            }
        )
    )
    union_writer_ranks = tuple(
        sorted(
            {
                rank
                for policy in policies.values()
                for rank in policy.short_conv_writer_ranks
            }
        )
    )
    capacity_key = None
    functional_key = None
    base_bank = None
    functional = None
    if phase_cache_context is not None:
        capacity_key = _capacity_phase_key(
            phase_cache_context,
            source_graph,
            parent_cache,
            runtime=runtime,
            exact_target_unique_parameters=target,
            required_target_hidden_size=required_hidden_size,
            candidate_hidden_sizes=widths,
            union_mlp_widths=union_widths,
            union_writer_ranks=union_writer_ranks,
            short_conv_internal_widths=conv_widths,
            attention_head_options=attention_options,
            discover_exact_short_conv_writer_ranks=(
                discover_exact_short_conv_writer_ranks
            ),
        )
        functional_key = _functional_phase_key(
            phase_cache_context,
            source_graph,
            parent_cache,
            runtime=runtime,
            required_target_hidden_size=required_hidden_size,
            candidate_hidden_sizes=widths,
            maximum_rows_per_role=functional_row_ceiling,
        )
        capacity_reuse = phase_cache_context.existing_capacity_phase_reuse
        capacity_compatibility_receipt = None
        if capacity_reuse is None:
            capacity_hit = _lookup_phase_payload(phase_cache_context, capacity_key)
        else:
            (
                capacity_entry,
                capacity_payload,
                capacity_compatibility_receipt,
            ) = _lookup_existing_global_topology_phase_payload(
                phase_cache_context,
                capacity_key,
                capacity_reuse,
            )
            capacity_hit = (capacity_entry, capacity_payload)
        if capacity_hit is not None:
            capacity_entry, capacity_payload = capacity_hit
            capacity_payload = _strict_mapping(
                capacity_payload,
                expected_fields={
                    "schema",
                    "phase_key_sha256",
                    "capacity_option_bank",
                    "payload_sha256",
                },
                label="global capacity phase payload",
            )
            base_bank = _restore_capacity_option_bank(
                capacity_payload["capacity_option_bank"]
            )
            _validate_capacity_option_bank_context(
                base_bank,
                source_graph,
                parent_cache,
            )
            capacity_result = _strict_mapping(
                capacity_entry.result,
                expected_fields={
                    "payload_sha256",
                    "evidence_sha256",
                    "layer_count",
                    "layer_option_count",
                },
                label="global capacity phase completion result",
            )
            if (
                capacity_result["payload_sha256"] != capacity_payload["payload_sha256"]
                or capacity_result["evidence_sha256"] != base_bank.evidence_sha256
                or type(capacity_result["layer_count"]) is not int
                or capacity_result["layer_count"] != len(base_bank.layer_evidence)
                or type(capacity_result["layer_option_count"]) is not int
                or capacity_result["layer_option_count"] != len(base_bank.layer_options)
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Capacity phase completion does not bind its typed evidence."
                )
            if capacity_reuse is not None:
                assert capacity_compatibility_receipt is not None
                _complete_existing_global_topology_phase_reuse(
                    phase_cache_context,
                    capacity_key,
                    capacity_reuse,
                    capacity_entry,
                    capacity_payload,
                    capacity_compatibility_receipt,
                    evidence_sha256=base_bank.evidence_sha256,
                )
        functional_reuse = phase_cache_context.existing_functional_phase_reuse
        functional_compatibility_receipt = None
        if functional_reuse is None:
            functional_hit = _lookup_phase_payload(
                phase_cache_context,
                functional_key,
            )
        else:
            (
                functional_entry,
                functional_payload,
                functional_compatibility_receipt,
            ) = _lookup_existing_global_topology_phase_payload(
                phase_cache_context,
                functional_key,
                functional_reuse,
            )
            functional_hit = (functional_entry, functional_payload)
        if functional_hit is not None:
            functional_entry, functional_payload = functional_hit
            functional_payload = _strict_mapping(
                functional_payload,
                expected_fields={
                    "schema",
                    "phase_key_sha256",
                    "functional_width_evidence",
                    "payload_sha256",
                },
                label="global functional phase payload",
            )
            functional = _restore_functional_width_evidence(
                functional_payload["functional_width_evidence"]
            )
            _validate_functional_width_context(
                functional,
                source_graph,
                parent_cache,
                widths,
            )
            functional_result = _strict_mapping(
                functional_entry.result,
                expected_fields={
                    "payload_sha256",
                    "evidence_sha256",
                    "candidate_score_count",
                },
                label="global functional phase completion result",
            )
            if (
                functional_result["payload_sha256"]
                != functional_payload["payload_sha256"]
                or functional_result["evidence_sha256"] != functional.evidence_sha256
                or type(functional_result["candidate_score_count"]) is not int
                or functional_result["candidate_score_count"]
                != len(functional.candidate_scores)
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Functional phase completion does not bind its typed evidence."
                )
            if functional_reuse is not None:
                assert functional_compatibility_receipt is not None
                _complete_existing_global_topology_phase_reuse(
                    phase_cache_context,
                    functional_key,
                    functional_reuse,
                    functional_entry,
                    functional_payload,
                    functional_compatibility_receipt,
                    evidence_sha256=functional.evidence_sha256,
                )

    if (
        (base_bank is None or functional is None)
        and defer_evidence_cache_miss
        and retained_parent_rows is None
    ):
        _phase_progress(
            phase_cache_context,
            "global_topology_evidence_compute_deferred",
            capacity_missing=base_bank is None,
            functional_missing=functional is None,
            required_target_hidden_size=required_hidden_size,
            retained_parent_rows_available=False,
            selective_session_opened=False,
        )
        raise Lfm2CachedGlobalTopologyEvidenceCacheMiss(
            "Global topology B1/B2 phase evidence is cold and retained parent "
            "rows are not materialized yet."
        )

    if base_bank is None or functional is None:
        if selective_session is not None:
            session_context = nullcontext(selective_session)
            session_source = "supplied"
        elif (
            phase_cache_context is not None
            and phase_cache_context.selective_session_factory is not None
        ):
            session_context = phase_cache_context.selective_session_factory()
            if not hasattr(session_context, "__enter__") or not hasattr(
                session_context,
                "__exit__",
            ):
                raise TypeError(
                    "selective_session_factory must return a context manager."
                )
            session_source = "lazy_external_factory"
        else:
            session_context = parent_cache.selective_authenticated_session()
            session_source = "owned_cache_session"
        _phase_progress(
            phase_cache_context,
            "global_topology_evidence_compute_started",
            capacity_missing=base_bank is None,
            functional_missing=functional is None,
            required_target_hidden_size=required_hidden_size,
            selective_session_source=session_source,
            selective_session_opened=(selective_session is None),
        )
        with session_context as session:
            if not isinstance(
                session,
                Lfm2CircuitTraceSelectiveAuthenticatedSession,
            ):
                raise TypeError(
                    "selective_session_factory must yield an authenticated "
                    "LFM2 circuit-trace session."
                )
            if (
                session.cache_key != parent_cache.cache_key
                or session.path != parent_cache.path
                or len(session) != len(parent_cache)
            ):
                raise Lfm2CachedGlobalTopologyFrontierError(
                    "selective_session_factory yielded a session for a "
                    "different parent cache."
                )
            session.verify()
            authenticated_retained_rows = retained_parent_rows
            if retained_parent_rows is not None:
                try:
                    authenticated_retained_rows = Lfm2RetainedTraceRowAdapter.build(
                        retained_parent_rows,
                        cache_key=parent_cache.cache_key,
                        corpus_fingerprint=parent_cache.corpus_fingerprint,
                        batch_references=session.batch_references,
                        phases_by_batch=tuple(
                            _cached_heterogeneous_capacity_module._ordered_parent_phases(
                                reference.record
                            )
                            for reference in session.batch_references
                        ),
                        expected_layer_kinds=tuple(
                            "short_conv" if layer_type == "conv" else "full_attention"
                            for layer_type in layer_types
                        ),
                    )
                except Lfm2RetainedTraceRowsError as exc:
                    raise Lfm2CachedGlobalTopologyFrontierError(
                        f"Retained global topology rows failed authentication: {exc}"
                    ) from exc
            if base_bank is None:
                capacity_batch_progress = None
                if phase_cache_context is not None:

                    def relay_capacity_batch_progress(
                        details: Mapping[str, Any],
                    ) -> None:
                        _phase_progress(
                            phase_cache_context,
                            "global_topology_capacity_batch_progress",
                            **dict(details),
                        )

                    capacity_batch_progress = relay_capacity_batch_progress
                base_bank = build_lfm2_cached_heterogeneous_capacity_option_bank(
                    source_graph,
                    parent_cache,
                    selective_session=session,
                    retained_parent_rows=authenticated_retained_rows,
                    batch_progress=capacity_batch_progress,
                    short_conv_internal_widths=conv_widths,
                    short_conv_writer_ranks=union_writer_ranks,
                    attention_head_options=attention_options,
                    mlp_intermediate_sizes=union_widths,
                    target_hidden_size=max(widths),
                    conv_kernel_size=runtime.conv_kernel_size,
                    attention_geometry_risk_enabled=True,
                    emit_factor_writer_options=False,
                )
                if phase_cache_context is not None:
                    assert capacity_key is not None
                    computed_capacity = dict(base_bank.as_dict())
                    (
                        capacity_entry,
                        capacity_payload,
                        capacity_published_locally,
                    ) = _publish_phase_payload(
                        phase_cache_context,
                        capacity_key,
                        schema=GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
                        body={
                            "capacity_option_bank": computed_capacity,
                        },
                        result={
                            "evidence_sha256": base_bank.evidence_sha256,
                            "layer_count": len(base_bank.layer_evidence),
                            "layer_option_count": len(base_bank.layer_options),
                        },
                    )
                    capacity_payload = _strict_mapping(
                        capacity_payload,
                        expected_fields={
                            "schema",
                            "phase_key_sha256",
                            "capacity_option_bank",
                            "payload_sha256",
                        },
                        label="published global capacity phase payload",
                    )
                    if capacity_payload["capacity_option_bank"] != (computed_capacity):
                        raise FoldPhaseCacheIntegrityError(
                            "Concurrent capacity publication differs from the "
                            "locally authenticated evidence."
                        )
                    published_capacity = (
                        base_bank
                        if capacity_published_locally
                        else _restore_capacity_option_bank(
                            capacity_payload["capacity_option_bank"]
                        )
                    )
                    _validate_capacity_option_bank_context(
                        published_capacity,
                        source_graph,
                        parent_cache,
                    )
                    capacity_result = _strict_mapping(
                        capacity_entry.result,
                        expected_fields={
                            "payload_sha256",
                            "evidence_sha256",
                            "layer_count",
                            "layer_option_count",
                        },
                        label="published global capacity completion result",
                    )
                    if (
                        capacity_result["payload_sha256"]
                        != capacity_payload["payload_sha256"]
                        or capacity_result["evidence_sha256"]
                        != published_capacity.evidence_sha256
                        or type(capacity_result["layer_count"]) is not int
                        or capacity_result["layer_count"]
                        != len(published_capacity.layer_evidence)
                        or type(capacity_result["layer_option_count"]) is not int
                        or capacity_result["layer_option_count"]
                        != len(published_capacity.layer_options)
                    ):
                        raise FoldPhaseCacheIntegrityError(
                            "Published capacity completion does not bind its "
                            "typed evidence."
                        )
                    base_bank = published_capacity
            if functional is None:
                functional_progress = None
                if phase_cache_context is not None:

                    def relay_functional_progress(
                        stage: str,
                        details: Mapping[str, Any],
                    ) -> None:
                        _phase_progress(
                            phase_cache_context,
                            stage,
                            **dict(details),
                        )

                    functional_progress = relay_functional_progress
                functional = build_lfm2_cached_functional_width_evidence(
                    source_graph,
                    parent_cache,
                    widths,
                    selective_session=session,
                    retained_parent_rows=authenticated_retained_rows,
                    maximum_rows_per_role=functional_row_ceiling,
                    progress=functional_progress,
                )
                if phase_cache_context is not None:
                    assert functional_key is not None
                    computed_functional = dict(functional.as_dict())
                    (
                        functional_entry,
                        functional_payload,
                        functional_published_locally,
                    ) = _publish_phase_payload(
                        phase_cache_context,
                        functional_key,
                        schema=GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
                        body={
                            "functional_width_evidence": dict(computed_functional),
                        },
                        result={
                            "evidence_sha256": functional.evidence_sha256,
                            "candidate_score_count": len(functional.candidate_scores),
                        },
                    )
                    functional_payload = _strict_mapping(
                        functional_payload,
                        expected_fields={
                            "schema",
                            "phase_key_sha256",
                            "functional_width_evidence",
                            "payload_sha256",
                        },
                        label="published global functional phase payload",
                    )
                    if functional_payload["functional_width_evidence"] != (
                        computed_functional
                    ):
                        raise FoldPhaseCacheIntegrityError(
                            "Concurrent functional-width publication differs "
                            "from the locally authenticated evidence."
                        )
                    published_functional = (
                        functional
                        if functional_published_locally
                        else _restore_functional_width_evidence(
                            functional_payload["functional_width_evidence"]
                        )
                    )
                    _validate_functional_width_context(
                        published_functional,
                        source_graph,
                        parent_cache,
                        widths,
                    )
                    functional_result = _strict_mapping(
                        functional_entry.result,
                        expected_fields={
                            "payload_sha256",
                            "evidence_sha256",
                            "candidate_score_count",
                        },
                        label="published global functional completion result",
                    )
                    if (
                        functional_result["payload_sha256"]
                        != functional_payload["payload_sha256"]
                        or functional_result["evidence_sha256"]
                        != published_functional.evidence_sha256
                        or type(functional_result["candidate_score_count"]) is not int
                        or functional_result["candidate_score_count"]
                        != len(published_functional.candidate_scores)
                    ):
                        raise FoldPhaseCacheIntegrityError(
                            "Published functional completion does not bind its "
                            "typed evidence."
                        )
                    functional = published_functional
            session.verify()
        _phase_progress(
            phase_cache_context,
            "global_topology_evidence_compute_completed",
            base_capacity_option_bank_sha256=base_bank.evidence_sha256,
            functional_width_evidence_sha256=functional.evidence_sha256,
            required_target_hidden_size=required_hidden_size,
        )

    assert base_bank is not None
    assert functional is not None
    _validate_capacity_option_bank_context(
        base_bank,
        source_graph,
        parent_cache,
    )
    _validate_functional_width_context(
        functional,
        source_graph,
        parent_cache,
        widths,
    )

    def materialize_width(
        hidden_size: int,
        policy: Lfm2MlpWidthOptionPolicy,
    ) -> tuple[
        Lfm2HeterogeneousLayoutSelectionInput,
        Lfm2CachedHeterogeneousCapacityOptionBank,
        Lfm2HeterogeneousLayoutSelection,
    ]:
        bank = _retarget_capacity_option_bank(
            base_bank,
            policy,
            target_hidden_size=hidden_size,
        )
        pins = _target_pins_for_width(
            hidden_size=hidden_size,
            runtime=runtime,
            source_graph=source_graph,
            parent_cache=parent_cache,
            base_bank=base_bank,
            selected_bank=bank,
            functional_evidence=functional,
            policy=policy,
            exact_target_unique_parameters=target,
        )
        selection_input = Lfm2HeterogeneousLayoutSelectionInput(
            target_pins=pins,
            layer_options=bank.layer_options,
            max_states=state_ceiling,
        )
        selection = selection_input.select(
            source_graph,
            exact_target_unique_parameters=target,
        )
        return selection_input, bank, selection

    selection_key = None
    if phase_cache_context is not None:
        selection_key = _selection_phase_key(
            phase_cache_context,
            source_graph,
            parent_cache,
            runtime=runtime,
            exact_target_unique_parameters=target,
            required_target_hidden_size=required_hidden_size,
            candidate_hidden_sizes=widths,
            policies=policies,
            base_bank=base_bank,
            functional=functional,
            max_states=state_ceiling,
        )
        selection_hit = _lookup_phase_payload(phase_cache_context, selection_key)
        if selection_hit is not None:
            selection_entry, selection_payload = selection_hit
            selection_payload = _strict_mapping(
                selection_payload,
                expected_fields={
                    "schema",
                    "phase_key_sha256",
                    "frontier",
                    "payload_sha256",
                },
                label="global frontier-selection phase payload",
            )
            raw_frontier = _strict_mapping(
                selection_payload["frontier"],
                expected_fields={
                    "schema",
                    "source_graph_fingerprint",
                    "parent_cache_key",
                    "exact_target_unique_parameters",
                    "source_layer_count",
                    "required_target_hidden_size",
                    "candidate_hidden_sizes",
                    "base_union_capacity_option_bank_sha256",
                    "functional_width_evidence_sha256",
                    "candidates",
                    "rejected_widths",
                    "selected_candidate_sha256",
                    "selected_input_sha256",
                    "selected_capacity_option_bank_sha256",
                    "attention_geometry_policy",
                    "selected_descriptor_sha256",
                    "selected_layout_selection_receipt_sha256",
                    "selection_contract",
                    "functional_width_evidence",
                    "frontier_sha256",
                },
                label="cached global topology frontier",
            )
            candidate_rows = raw_frontier["candidates"]
            rejection_rows = raw_frontier["rejected_widths"]
            if not isinstance(candidate_rows, list) or not isinstance(
                rejection_rows,
                list,
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Cached global frontier candidate inventories are invalid."
                )
            cached_reports = tuple(_restore_candidate(row) for row in candidate_rows)
            cached_rejected = tuple(_restore_rejection(row) for row in rejection_rows)
            selected_report = select_lfm2_global_topology_report(cached_reports)
            if (
                selected_report.candidate_sha256
                != raw_frontier["selected_candidate_sha256"]
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Cached global frontier selected the wrong risk optimum."
                )
            selected_policy = policies.get(selected_report.hidden_size)
            if selected_policy is None:
                raise FoldPhaseCacheIntegrityError(
                    "Cached selected width has no current exact lattice policy."
                )
            _phase_progress(
                phase_cache_context,
                "global_topology_selected_width_reconstruction_started",
                selected_hidden_size=selected_report.hidden_size,
                required_target_hidden_size=required_hidden_size,
                full_frontier_selector_widths_skipped=len(policies) - 1,
            )
            try:
                selected_input, selected_bank, selected_layout = materialize_width(
                    selected_report.hidden_size,
                    selected_policy,
                )
            except Exception as exc:
                raise FoldPhaseCacheIntegrityError(
                    "Cached selected frontier width no longer materializes."
                ) from exc
            restored = Lfm2CachedGlobalTopologyFrontier(
                source_graph_fingerprint=source_graph.fingerprint,
                parent_cache_key=parent_cache.cache_key,
                exact_target_unique_parameters=target,
                source_layer_count=len(layer_types),
                required_target_hidden_size=required_hidden_size,
                candidate_hidden_sizes=widths,
                base_union_capacity_option_bank_sha256=base_bank.evidence_sha256,
                functional_width_evidence=functional,
                candidates=cached_reports,
                rejected_widths=cached_rejected,
                selected_candidate_sha256=selected_report.candidate_sha256,
                selected_input=selected_input,
                selected_capacity_option_bank=selected_bank,
                selected_layout=selected_layout,
                frontier_sha256=raw_frontier["frontier_sha256"],
            )
            if dict(restored.as_dict()) != raw_frontier:
                raise FoldPhaseCacheIntegrityError(
                    "Cached global frontier did not round-trip canonically."
                )
            selection_result = _strict_mapping(
                selection_entry.result,
                expected_fields={
                    "payload_sha256",
                    "frontier_sha256",
                    "selected_candidate_sha256",
                    "selected_hidden_size",
                    "required_target_hidden_size",
                    "candidate_count",
                    "rejected_width_count",
                },
                label="global frontier-selection completion result",
            )
            if (
                selection_result["payload_sha256"]
                != selection_payload["payload_sha256"]
                or selection_result["frontier_sha256"] != restored.frontier_sha256
                or selection_result["selected_candidate_sha256"]
                != restored.selected_candidate_sha256
                or type(selection_result["selected_hidden_size"]) is not int
                or selection_result["selected_hidden_size"]
                != restored.selected_input.target_pins.hidden_size
                or type(selection_result["required_target_hidden_size"])
                not in {int, type(None)}
                or selection_result["required_target_hidden_size"]
                != restored.required_target_hidden_size
                or type(selection_result["candidate_count"]) is not int
                or selection_result["candidate_count"] != len(restored.candidates)
                or type(selection_result["rejected_width_count"]) is not int
                or selection_result["rejected_width_count"]
                != len(restored.rejected_widths)
            ):
                raise FoldPhaseCacheIntegrityError(
                    "Frontier phase completion does not bind its typed result."
                )
            _phase_progress(
                phase_cache_context,
                "global_topology_selected_width_reconstructed",
                selected_hidden_size=selected_report.hidden_size,
                required_target_hidden_size=required_hidden_size,
                frontier_sha256=restored.frontier_sha256,
                selector_width_count=1,
            )
            return restored

    _phase_progress(
        phase_cache_context,
        "global_topology_frontier_selection_started",
        policy_width_count=len(policies),
        required_target_hidden_size=required_hidden_size,
    )
    reports = []
    artifacts: dict[
        str,
        tuple[
            Lfm2HeterogeneousLayoutSelectionInput,
            Lfm2CachedHeterogeneousCapacityOptionBank,
            Lfm2HeterogeneousLayoutSelection,
        ],
    ] = {}
    for hidden_size, policy in sorted(policies.items()):
        try:
            selection_input, bank, selection = materialize_width(
                hidden_size,
                policy,
            )
        except Lfm2HeterogeneousLayoutUnattainable as error:
            rejected.append(
                Lfm2GlobalTopologyRejectedWidth.build(
                    hidden_size,
                    "exact_selector_unattainable",
                    str(error),
                )
            )
            continue
        score = functional.score_for(hidden_size)
        mean_component_risk = selection.budget_selection.total_predicted_risk / len(
            layer_types
        )
        report = Lfm2GlobalTopologyCandidateReport.build(
            hidden_size=hidden_size,
            exact_unique_parameters=target,
            target_layer_count=len(selection.descriptor.layers),
            functional_retention=score.role_balanced_functional_retention,
            functional_loss=score.role_balanced_functional_loss,
            mean_component_risk=mean_component_risk,
            functional_score_sha256=score.score_sha256,
            mlp_width_lattice_sha256=policy.exact_budget_lattice_sha256,
            capacity_option_bank_sha256=bank.evidence_sha256,
            selection_input_sha256=selection_input.fingerprint,
            descriptor_sha256=selection.descriptor.descriptor_sha256,
            layout_selection_receipt_sha256=selection.receipt.receipt_sha256,
        )
        reports.append(report)
        artifacts[report.candidate_sha256] = (selection_input, bank, selection)
    if not reports:
        if required_hidden_size is not None:
            raise Lfm2CachedGlobalTopologyFrontierUnattainable(
                "The required target hidden size did not produce a complete "
                f"exact physical child: H{required_hidden_size}."
            )
        raise Lfm2CachedGlobalTopologyFrontierUnattainable(
            "No legal residual width produced a complete exact physical child."
        )
    reports.sort(key=lambda value: value.hidden_size)
    rejected.sort(key=lambda value: value.hidden_size)
    selected = select_lfm2_global_topology_report(reports)
    selected_input, selected_bank, selected_layout = artifacts[
        selected.candidate_sha256
    ]
    values = {
        "source_graph_fingerprint": source_graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "exact_target_unique_parameters": target,
        "source_layer_count": len(layer_types),
        "required_target_hidden_size": required_hidden_size,
        "candidate_hidden_sizes": widths,
        "base_union_capacity_option_bank_sha256": base_bank.evidence_sha256,
        "functional_width_evidence": functional,
        "candidates": tuple(reports),
        "rejected_widths": tuple(rejected),
        "selected_candidate_sha256": selected.candidate_sha256,
        "selected_input": selected_input,
        "selected_capacity_option_bank": selected_bank,
        "selected_layout": selected_layout,
    }
    provisional = object.__new__(Lfm2CachedGlobalTopologyFrontier)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "frontier_sha256", "0" * 64)
    result = Lfm2CachedGlobalTopologyFrontier(
        **values,
        frontier_sha256=canonical_sha256(provisional._body()),
    )
    if phase_cache_context is not None:
        assert selection_key is not None
        selection_entry, selection_payload, _selection_published_locally = (
            _publish_phase_payload(
                phase_cache_context,
                selection_key,
                schema=GLOBAL_TOPOLOGY_SELECTION_PAYLOAD_SCHEMA,
                body={"frontier": dict(result.as_dict())},
                result={
                    "frontier_sha256": result.frontier_sha256,
                    "selected_candidate_sha256": result.selected_candidate_sha256,
                    "selected_hidden_size": result.selected_input.target_pins.hidden_size,
                    "required_target_hidden_size": (result.required_target_hidden_size),
                    "candidate_count": len(result.candidates),
                    "rejected_width_count": len(result.rejected_widths),
                },
            )
        )
        selection_payload = _strict_mapping(
            selection_payload,
            expected_fields={
                "schema",
                "phase_key_sha256",
                "frontier",
                "payload_sha256",
            },
            label="published global frontier-selection phase payload",
        )
        if selection_payload["frontier"] != dict(result.as_dict()):
            raise FoldPhaseCacheIntegrityError(
                "Concurrent frontier publication differs from the locally "
                "authenticated selection."
            )
        selection_result = _strict_mapping(
            selection_entry.result,
            expected_fields={
                "payload_sha256",
                "frontier_sha256",
                "selected_candidate_sha256",
                "selected_hidden_size",
                "required_target_hidden_size",
                "candidate_count",
                "rejected_width_count",
            },
            label="published global frontier-selection completion result",
        )
        if (
            selection_result["payload_sha256"] != selection_payload["payload_sha256"]
            or selection_result["frontier_sha256"] != result.frontier_sha256
            or selection_result["selected_candidate_sha256"]
            != result.selected_candidate_sha256
            or type(selection_result["selected_hidden_size"]) is not int
            or selection_result["selected_hidden_size"]
            != result.selected_input.target_pins.hidden_size
            or type(selection_result["required_target_hidden_size"])
            not in {int, type(None)}
            or selection_result["required_target_hidden_size"]
            != result.required_target_hidden_size
            or type(selection_result["candidate_count"]) is not int
            or selection_result["candidate_count"] != len(result.candidates)
            or type(selection_result["rejected_width_count"]) is not int
            or selection_result["rejected_width_count"] != len(result.rejected_widths)
        ):
            raise FoldPhaseCacheIntegrityError(
                "Published frontier completion does not bind its typed result."
            )
    _phase_progress(
        phase_cache_context,
        "global_topology_frontier_selection_completed",
        frontier_sha256=result.frontier_sha256,
        selected_hidden_size=result.selected_input.target_pins.hidden_size,
        required_target_hidden_size=required_hidden_size,
        selector_width_count=len(policies),
    )
    return result


__all__ = [
    "GLOBAL_TOPOLOGY_CAPACITY_PHASE",
    "GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE",
    "GLOBAL_TOPOLOGY_SELECTION_PHASE",
    "GLOBAL_TOPOLOGY_SCORE_METHOD",
    "Lfm2CachedGlobalTopologyFrontier",
    "Lfm2CachedGlobalTopologyEvidenceCacheMiss",
    "Lfm2CachedGlobalTopologyFrontierError",
    "Lfm2CachedGlobalTopologyFrontierUnattainable",
    "Lfm2ExistingGlobalTopologyPhaseReuseRequest",
    "Lfm2GlobalTopologyPhaseCacheContext",
    "Lfm2GlobalTargetRuntimePins",
    "Lfm2GlobalTopologyCandidateReport",
    "Lfm2GlobalTopologyRejectedWidth",
    "build_lfm2_cached_global_topology_frontier",
    "select_lfm2_global_topology_report",
]
