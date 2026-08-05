"""Authenticated reusable phase cache for the LFM2 balanced residual basis.

This is deliberately downstream of the sealed circuit-trace collector.  A
cache hit authenticates the parent, graph, trace-cache metadata/session,
target/token bindings, implementation, and numeric contract, then restores
the seven canonical CPU/float64 tensors without loading a trace tensor or
executing either model.

The cache stores only the residual-basis phase.  Future per-layer fold-unit
resume must use separate immutable phases replayed into a fresh blank child;
partially mutated child state is never a valid resume boundary here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import platform
import re
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
import torch

from . import balanced_krylov as _balanced_krylov_module
from . import dynamic_balanced_circuit_basis as _balanced_basis_module
from . import dynamic_circuit_ir as _circuit_ir_module
from . import dynamic_fold_phase_cache as _phase_cache_module
from . import dynamic_model_inspector as _model_inspector_module
from . import dynamic_weight_complete_basis as _weight_basis_module
from . import hf_lfm2_balanced_circuit_basis as _lfm2_balanced_module
from . import hf_lfm2_circuit_trace_cache as _trace_cache_module
from . import hf_lfm2_circuit_trace_row_selection as _row_selection_module
from . import hf_lfm2_streaming_vocabulary_margin as _streaming_margin_module
from . import (
    hf_lfm2_streaming_vocabulary_margin_adapter as _streaming_margin_adapter_module,
)
from . import hf_lfm2_weight_complete_basis as _lfm2_weight_basis_module
from .dynamic_balanced_circuit_basis import (
    BalancedCircuitBasisResult,
    _tensor_sha256 as _balanced_tensor_sha256,
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
    VerifiedFoldPhaseCacheEntry,
)
from .hf_lfm2_balanced_circuit_basis import (
    LFM2_CIRCUIT_BALANCED_BASIS_SCHEMA,
    Lfm2CircuitBalancedBasisResult,
    build_lfm2_circuit_balanced_basis,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    fingerprint_lfm2_parent,
)
from .hf_lfm2_weight_complete_basis import (
    Lfm2WeightCompleteInventory,
    inventory_lfm2_weight_complete_operators,
)


LFM2_BALANCED_BASIS_PHASE = "lfm2-circuit-balanced-residual-basis-v1"
LFM2_BALANCED_BASIS_PHASE_ABI = (
    "neural-forge-lfm2-circuit-balanced-residual-basis-phase-cache-v1"
)
LFM2_BALANCED_BASIS_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-residual-basis-phase-receipt-v1"
)
LFM2_BALANCED_BASIS_BINDINGS_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-residual-basis-live-bindings-v1"
)
LFM2_BALANCED_BASIS_NUMERIC_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-residual-basis-numeric-v1"
)
LFM2_BALANCED_BASIS_TENSOR_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-residual-basis-tensors-v1"
)
LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-residual-basis-source-anchor-v1"
)
_LFM2_BALANCED_BASIS_GRAPH_EXECUTION_SCHEMA = (
    "neural-forge-lfm2-balanced-basis-graph-execution-v1"
)

_TENSOR_FILENAME = "basis.safetensors"
_RECEIPT_FILENAME = "receipt.json"
_SOURCE_ANCHOR_FILENAME = "source-anchor.json"
_SOURCE_ANCHOR_ALGORITHM = "hmac-sha256"
_SOURCE_ANCHOR_DOMAIN = (
    b"neural-forge-lfm2-circuit-balanced-residual-basis-source-anchor-v1\0"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_MINIMUM_HMAC_KEY_BYTES = 32
_ROW_SELECTION_PROGRESS_SAMPLE_INTERVAL = 64
_PROGRESS_FORWARD_LOCK = threading.RLock()
_TENSOR_NAMES = (
    "balanced_decoder",
    "balanced_encoder",
    "basis",
    "controllability_gram",
    "full_weight_gram",
    "heldout_gram",
    "observability_gram",
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "phase_key",
        "phase_key_sha256",
        "bindings",
        "basis_report",
        "balanced_basis_report",
        "tensor_file",
        "tensor_descriptors",
        "execution",
        "receipt_sha256",
    }
)
_TENSOR_FILE_FIELDS = frozenset(
    {"filename", "format", "metadata", "size_bytes", "sha256"}
)
LFM2_BALANCED_BASIS_TENSOR_DESCRIPTOR_FIELDS = frozenset(
    {"name", "dtype", "shape", "stride", "numel", "size_bytes", "sha256"}
)
_SOURCE_ANCHOR_BODY_FIELDS = frozenset(
    {
        "schema",
        "algorithm",
        "key_id",
        "phase_name",
        "phase_key_sha256",
        "receipt_sha256",
        "tensor_file_sha256",
        "tensor_inventory_sha256",
        "basis_result_sha256",
        "balanced_run_sha256",
        "basis_map_sha256",
    }
)
_SOURCE_ANCHOR_FIELDS = _SOURCE_ANCHOR_BODY_FIELDS | {"hmac_sha256"}
_EXECUTION_RECEIPT = {
    "parent_forward_calls": 0,
    "child_forward_calls": 0,
    "trace_tensor_loads_on_hit": 0,
    "trace_regeneration_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "canonical_device": "cpu",
    "canonical_dtype": "torch.float64",
    "live_vocabulary_candidate_gate_serialized": False,
}
_TENSOR_METADATA = {
    "contract": canonical_json_bytes(
        {
            "schema": LFM2_BALANCED_BASIS_TENSOR_SCHEMA,
            "canonical_device": "cpu",
            "canonical_dtype": "torch.float64",
            "ordered_tensor_names": list(_TENSOR_NAMES),
        }
    ).decode("ascii")
}

PhaseProgressCallback = Callable[[str, Mapping[str, Any]], None]
PhaseBindingProvider = Callable[[], Mapping[str, Mapping[str, Any]]]
PhaseDependencyProvider = Callable[[], Mapping[str, str]]


class Lfm2BalancedBasisPhaseCacheError(RuntimeError):
    """The reusable balanced-basis phase contract is invalid."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2BalancedBasisPhaseCacheError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise Lfm2BalancedBasisPhaseCacheError(
            f"{name} must be a stable identifier without path separators."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be a positive integer.")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be finite.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be finite.")
    return result


def _unit_interval(
    value: Any,
    name: str,
    *,
    include_zero: bool,
) -> float:
    result = _finite_float(value, name)
    minimum_valid = result >= 0.0 if include_zero else result > 0.0
    if not minimum_valid or result > 1.0:
        boundary = "[0, 1]" if include_zero else "(0, 1]"
        raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be in {boundary}.")
    return result


def _canonical_value(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    normalized = _canonical_value(value, name)
    if not isinstance(normalized, dict):
        raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be a mapping.")
    return normalized


class PhaseAnchorAuthority:
    """Runtime-only HMAC authority for one source-authentic phase cache.

    The secret is deliberately absent from ``repr`` and from every serialized
    cache document.  Callers must obtain it from a runtime secret provider;
    placing the same secret under the mutable cache root would destroy the
    source-authentication boundary.
    """

    __slots__ = ("_key_id", "__secret", "__sealed")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        object.__setattr__(self, "_key_id", _identifier(key_id, "key_id"))
        if not isinstance(secret, bytes) or len(secret) < _MINIMUM_HMAC_KEY_BYTES:
            raise Lfm2BalancedBasisPhaseCacheError(
                "Phase-anchor secret must be at least 32 runtime-supplied bytes."
            )
        object.__setattr__(self, "_PhaseAnchorAuthority__secret", bytes(secret))
        object.__setattr__(self, "_PhaseAnchorAuthority__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_PhaseAnchorAuthority__sealed", False):
            raise AttributeError("PhaseAnchorAuthority is immutable.")
        object.__setattr__(self, name, value)

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("PhaseAnchorAuthority cannot be serialized.")

    @property
    def key_id(self) -> str:
        return self._key_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def _mac(self, body: Mapping[str, Any]) -> str:
        payload = _SOURCE_ANCHOR_DOMAIN + canonical_json_bytes(dict(body))
        return hmac.new(self.__secret, payload, hashlib.sha256).hexdigest()

    def sign(self, body: Mapping[str, Any]) -> str:
        """Return a domain-separated HMAC without exposing the secret."""

        return self._mac(body)

    def verify(self, body: Mapping[str, Any], claimed_hmac_sha256: Any) -> bool:
        """Verify a phase anchor in constant-time after strict digest parsing."""

        if (
            not isinstance(claimed_hmac_sha256, str)
            or _SHA256.fullmatch(claimed_hmac_sha256) is None
        ):
            return False
        return hmac.compare_digest(self._mac(body), claimed_hmac_sha256)


@dataclass(frozen=True, slots=True)
class Lfm2BalancedBasisPhaseBuildRequest:
    """One immutable request used for both phase identity and cold building."""

    adapter_source_sha256: str
    validation_fraction: float = 0.25
    split_seed: str = "lfm2-circuit-balanced-v1"
    max_rows_per_block: int = 512
    structural_controllability_weight: float = 0.35
    cached_controllability_weight: float = 0.65
    gram_regularization: float = 1e-10
    rank_tolerance: float = 1e-12
    max_condition_number: float = 1e12
    minimum_heldout_retained_energy_fraction: float = 0.0
    vocabulary_margin_observability: bool = False
    vocabulary_margin_top_k: int = 8
    vocabulary_margin_floor_fraction: float = 0.05
    streaming_vocabulary_margin_v2: bool = False
    streaming_vocabulary_margin_pair_chunk_size: int = 256
    target_tokenizer_vocabulary_margin_binding: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adapter_source_sha256",
            _digest(self.adapter_source_sha256, "adapter_source_sha256"),
        )
        object.__setattr__(
            self,
            "validation_fraction",
            _unit_interval(
                self.validation_fraction,
                "validation_fraction",
                include_zero=False,
            ),
        )
        if not isinstance(self.split_seed, str) or not self.split_seed:
            raise Lfm2BalancedBasisPhaseCacheError("split_seed must be nonempty text.")
        object.__setattr__(
            self,
            "max_rows_per_block",
            _positive_int(self.max_rows_per_block, "max_rows_per_block"),
        )
        for name in (
            "structural_controllability_weight",
            "cached_controllability_weight",
            "gram_regularization",
            "rank_tolerance",
        ):
            normalized = _finite_float(getattr(self, name), name)
            if normalized <= 0.0:
                raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be positive.")
            object.__setattr__(self, name, normalized)
        maximum_condition = _finite_float(
            self.max_condition_number,
            "max_condition_number",
        )
        if maximum_condition < 1.0:
            raise Lfm2BalancedBasisPhaseCacheError(
                "max_condition_number must be at least one."
            )
        object.__setattr__(self, "max_condition_number", maximum_condition)
        object.__setattr__(
            self,
            "minimum_heldout_retained_energy_fraction",
            _unit_interval(
                self.minimum_heldout_retained_energy_fraction,
                "minimum_heldout_retained_energy_fraction",
                include_zero=True,
            ),
        )
        for name in (
            "vocabulary_margin_observability",
            "streaming_vocabulary_margin_v2",
        ):
            if not isinstance(getattr(self, name), bool):
                raise Lfm2BalancedBasisPhaseCacheError(f"{name} must be boolean.")
        object.__setattr__(
            self,
            "vocabulary_margin_top_k",
            _positive_int(self.vocabulary_margin_top_k, "vocabulary_margin_top_k"),
        )
        object.__setattr__(
            self,
            "vocabulary_margin_floor_fraction",
            _unit_interval(
                self.vocabulary_margin_floor_fraction,
                "vocabulary_margin_floor_fraction",
                include_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "streaming_vocabulary_margin_pair_chunk_size",
            _positive_int(
                self.streaming_vocabulary_margin_pair_chunk_size,
                "streaming_vocabulary_margin_pair_chunk_size",
            ),
        )
        if (
            self.streaming_vocabulary_margin_v2
            and not self.vocabulary_margin_observability
        ):
            raise Lfm2BalancedBasisPhaseCacheError(
                "streaming_vocabulary_margin_v2 requires vocabulary-margin "
                "observability."
            )
        self._verify_target_binding()

    def _verify_target_binding(self) -> str | None:
        binding = self.target_tokenizer_vocabulary_margin_binding
        if binding is None:
            return None
        expected_type = (
            _streaming_margin_adapter_module.Lfm2TargetTokenizerVocabularyMarginBinding
        )
        if not isinstance(binding, expected_type):
            raise Lfm2BalancedBasisPhaseCacheError(
                "target_tokenizer_vocabulary_margin_binding has the wrong type."
            )
        try:
            binding.verify()
        except ValueError as exc:
            raise Lfm2BalancedBasisPhaseCacheError(
                "Target tokenizer vocabulary-margin binding failed verification."
            ) from exc
        return _digest(
            binding.binding_sha256,
            "target_tokenizer_vocabulary_margin_binding.binding_sha256",
        )

    def as_basis_config(self) -> dict[str, Any]:
        """Return the exact canonical values bound into the phase key."""

        return {
            "adapter_source_sha256": self.adapter_source_sha256,
            "validation_fraction": self.validation_fraction,
            "split_seed": self.split_seed,
            "max_rows_per_block": self.max_rows_per_block,
            "structural_controllability_weight": (
                self.structural_controllability_weight
            ),
            "cached_controllability_weight": self.cached_controllability_weight,
            "gram_regularization": self.gram_regularization,
            "rank_tolerance": self.rank_tolerance,
            "max_condition_number": self.max_condition_number,
            "minimum_heldout_retained_energy_fraction": (
                self.minimum_heldout_retained_energy_fraction
            ),
            "vocabulary_margin_observability": (self.vocabulary_margin_observability),
            "vocabulary_margin_top_k": self.vocabulary_margin_top_k,
            "vocabulary_margin_floor_fraction": (self.vocabulary_margin_floor_fraction),
            "streaming_vocabulary_margin_v2": self.streaming_vocabulary_margin_v2,
            "streaming_vocabulary_margin_pair_chunk_size": (
                self.streaming_vocabulary_margin_pair_chunk_size
            ),
            "target_tokenizer_vocabulary_margin_binding_sha256": (
                self._verify_target_binding()
            ),
        }

    def build(
        self,
        parent_model: Any,
        trace_cache: Lfm2CircuitTraceCache,
        target_width: int,
        *,
        circuit_graph: ModelCircuitGraph,
        selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None,
    ) -> Lfm2CircuitBalancedBasisResult:
        """Call the canonical builder with exactly the values in this request."""

        self._verify_target_binding()
        return build_lfm2_circuit_balanced_basis(
            parent_model,
            trace_cache,
            target_width,
            validation_fraction=self.validation_fraction,
            split_seed=self.split_seed,
            max_rows_per_block=self.max_rows_per_block,
            structural_controllability_weight=(self.structural_controllability_weight),
            cached_controllability_weight=self.cached_controllability_weight,
            gram_regularization=self.gram_regularization,
            rank_tolerance=self.rank_tolerance,
            max_condition_number=self.max_condition_number,
            minimum_heldout_retained_energy_fraction=(
                self.minimum_heldout_retained_energy_fraction
            ),
            circuit_graph=circuit_graph,
            vocabulary_margin_observability=(self.vocabulary_margin_observability),
            vocabulary_margin_top_k=self.vocabulary_margin_top_k,
            vocabulary_margin_floor_fraction=(self.vocabulary_margin_floor_fraction),
            selective_session=selective_session,
            streaming_vocabulary_margin_v2=self.streaming_vocabulary_margin_v2,
            streaming_vocabulary_margin_pair_chunk_size=(
                self.streaming_vocabulary_margin_pair_chunk_size
            ),
            target_tokenizer_vocabulary_margin_binding=(
                self.target_tokenizer_vocabulary_margin_binding
            ),
        )


@dataclass(frozen=True, slots=True)
class Lfm2ExistingBalancedBasisPhaseReuseRequest:
    """Explicit immutable request to consume one historical phase hit only."""

    phase_key_sha256: str
    expected_map_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_key_sha256",
            _digest(self.phase_key_sha256, "phase_key_sha256"),
        )
        if self.expected_map_sha256 is not None:
            object.__setattr__(
                self,
                "expected_map_sha256",
                _digest(self.expected_map_sha256, "expected_map_sha256"),
            )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": "neural-forge-lfm2-existing-balanced-basis-reuse-request-v1",
            "phase_key_sha256": self.phase_key_sha256,
            "expected_map_sha256": self.expected_map_sha256,
            "cache_miss_build_allowed": False,
        }
        return {**body, "request_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2BalancedBasisPhaseCacheContext:
    """Optional operator hook for a stable derived-phase cache root.

    ``dependency_provider`` must reauthenticate any optional upstream
    route/carrier/counting artifacts each time live phase bindings are checked.
    Operational paths and run IDs intentionally do not affect phase identity.
    """

    cache: DynamicFoldPhaseCache
    anchor_authority: PhaseAnchorAuthority
    events: FoldPhaseEventLog | None = None
    progress: PhaseProgressCallback | None = None
    dependency_provider: PhaseDependencyProvider | None = None
    existing_phase_reuse_request: (
        Lfm2ExistingBalancedBasisPhaseReuseRequest | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.cache, DynamicFoldPhaseCache):
            raise TypeError("cache must be DynamicFoldPhaseCache.")
        if not isinstance(self.anchor_authority, PhaseAnchorAuthority):
            raise TypeError("anchor_authority must be PhaseAnchorAuthority.")
        if self.events is not None and not isinstance(self.events, FoldPhaseEventLog):
            raise TypeError("events must be FoldPhaseEventLog or None.")
        if self.progress is not None and not callable(self.progress):
            raise TypeError("progress must be callable or None.")
        if self.dependency_provider is not None and not callable(
            self.dependency_provider
        ):
            raise TypeError("dependency_provider must be callable or None.")
        if self.existing_phase_reuse_request is not None and not isinstance(
            self.existing_phase_reuse_request,
            Lfm2ExistingBalancedBasisPhaseReuseRequest,
        ):
            raise TypeError(
                "existing_phase_reuse_request must be "
                "Lfm2ExistingBalancedBasisPhaseReuseRequest or None."
            )


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2BalancedBasisPhase:
    """A source-anchored, hit-only residual map for post-fold diagnostics."""

    phase_key: FoldPhaseKey
    completion_sha256: str
    receipt: Mapping[str, Any]
    source_anchor: Mapping[str, Any]
    basis_result: BalancedCircuitBasisResult
    report: Mapping[str, Any]

    @property
    def source_width(self) -> int:
        return self.basis_result.source_width

    @property
    def target_width(self) -> int:
        return self.basis_result.target_width

    @property
    def map_sha256(self) -> str:
        return self.basis_result.map_sha256

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.encode(rows)

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": "neural-forge-lfm2-authenticated-balanced-basis-phase-v1",
            "phase_key_sha256": self.phase_key.phase_key_sha256,
            "completion_sha256": self.completion_sha256,
            "receipt_sha256": self.receipt["receipt_sha256"],
            "source_anchor_hmac_sha256": self.source_anchor["hmac_sha256"],
            "basis_result_sha256": self.report["result_sha256"],
            "basis_map_sha256": self.map_sha256,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "cache_miss_build_allowed": False,
            "parent_forward_calls": 0,
            "trace_tensor_loads": 0,
            "training_steps": 0,
        }
        return {**body, "authenticated_phase_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class _LiveBindings:
    parent: Any
    inventory: Lfm2WeightCompleteInventory
    circuit_graph: ModelCircuitGraph
    document: Mapping[str, Any]


def _progress(
    context: Lfm2BalancedBasisPhaseCacheContext,
    stage: str,
    **details: Any,
) -> None:
    if context.progress is not None:
        context.progress(stage, MappingProxyType(dict(details)))


def _external_dependencies(
    context: Lfm2BalancedBasisPhaseCacheContext,
) -> dict[str, str]:
    provider = context.dependency_provider
    if provider is None:
        return {}
    try:
        values = provider()
    except Exception as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "External phase dependency provider failed."
        ) from exc
    if not isinstance(values, Mapping):
        raise Lfm2BalancedBasisPhaseCacheError(
            "External phase dependency provider must return a mapping."
        )
    dependencies: dict[str, str] = {}
    for raw_name, value in sorted(values.items()):
        name = _identifier(raw_name, "external dependency name")
        if name in dependencies:
            raise Lfm2BalancedBasisPhaseCacheError(
                "External phase dependency names must be unique."
            )
        dependencies[name] = _digest(value, f"external dependency {name}")
    return dependencies


class _RowSelectionProgressHandler(logging.Handler):
    """Forward sampled builder telemetry without changing semantic evidence."""

    def __init__(self, context: Lfm2BalancedBasisPhaseCacheContext) -> None:
        super().__init__(level=logging.INFO)
        self._context = context

    def emit(self, record: logging.LogRecord) -> None:
        value = getattr(record, "neural_forge_progress", None)
        if not isinstance(value, Mapping):
            return
        payload = _canonical_mapping(value, "balanced-basis progress")
        row_selection_schema = payload.pop("schema", None)
        if not isinstance(row_selection_schema, str) or not row_selection_schema:
            return
        completed = payload.get("completed_batch_count")
        total = payload.get("batch_count")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or completed <= 0
            or total <= 0
            or completed > total
        ):
            return
        if (
            completed != 1
            and completed != total
            and completed % _ROW_SELECTION_PROGRESS_SAMPLE_INTERVAL != 0
        ):
            return
        safe_payload = {
            key: item
            for key, item in payload.items()
            if key
            not in {
                "schema",
                "timestamp_utc",
                "configuration_sha256",
                "sequence",
                "stage",
            }
            and (
                item is None
                or isinstance(item, (bool, int, str))
                or (isinstance(item, float) and math.isfinite(item))
            )
        }
        _progress(
            self._context,
            "balanced_basis_row_selection_progress",
            row_selection_schema=row_selection_schema,
            **safe_payload,
        )


@contextmanager
def _forward_row_selection_progress(
    context: Lfm2BalancedBasisPhaseCacheContext,
) -> Iterator[None]:
    """Temporarily expose INFO progress and restore all logger state."""

    if context.progress is None:
        yield
        return
    logger = _lfm2_balanced_module._LOGGER
    handler = _RowSelectionProgressHandler(context)
    with _PROGRESS_FORWARD_LOCK:
        original_level = logger.level
        original_disabled = logger.disabled
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.disabled = False
        try:
            yield
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
            logger.disabled = original_disabled


def _binding_documents(provider: PhaseBindingProvider) -> dict[str, Any]:
    if not callable(provider):
        raise Lfm2BalancedBasisPhaseCacheError(
            "binding_provider must be callable when phase caching is enabled."
        )
    value = _canonical_mapping(provider(), "phase binding provider result")
    if set(value) != {"selected_target", "tokenizer"}:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Phase binding provider must return exactly selected_target and tokenizer."
        )
    for name in ("selected_target", "tokenizer"):
        if not isinstance(value[name], dict) or not value[name]:
            raise Lfm2BalancedBasisPhaseCacheError(
                f"Phase {name} binding must be a nonempty mapping."
            )
    return value


def _trace_cache_binding(
    trace_cache: Lfm2CircuitTraceCache,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
) -> dict[str, Any]:
    if not isinstance(trace_cache, Lfm2CircuitTraceCache):
        raise Lfm2BalancedBasisPhaseCacheError(
            "trace_cache must be Lfm2CircuitTraceCache."
        )
    if not isinstance(
        selective_session,
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "A selective authenticated trace session is required for reusable "
            "basis caching."
        )
    selective_session.verify()
    if (
        selective_session.path != trace_cache.path
        or selective_session.cache_key != trace_cache.cache_key
        or selective_session.batch_count != trace_cache.batch_count
        or selective_session.manifest != trace_cache.manifest
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Selective trace session differs from its phase-cache handle."
        )
    manifest = trace_cache.manifest
    completion = _canonical_mapping(
        dict(trace_cache._completion),
        "trace cache completion",
    )
    session = selective_session.as_dict()
    try:
        binding = {
            "cache_key": trace_cache.cache_key,
            "manifest_sha256": manifest["manifest_sha256"],
            "completion_sha256": completion["completion_sha256"],
            "parent_identity_sha256": trace_cache.parent_fingerprint,
            "corpus_sha256": trace_cache.corpus_fingerprint,
            "layer_topology_sha256": trace_cache.layer_topology_fingerprint,
            "implementation_sha256": trace_cache.implementation_fingerprint,
            "collector_source_sha256": manifest["implementation"][
                "collector_source_sha256"
            ],
            "canonical_contract_source_sha256": manifest["implementation"][
                "canonical_contract_source_sha256"
            ],
            "transformers_lfm2_source_sha256": manifest["implementation"][
                "transformers_lfm2_source_sha256"
            ],
            "execution_sha256": manifest["execution"]["execution_sha256"],
            "capture_selection_sha256": manifest["capture"]["selection_sha256"],
            "selective_session_sha256": selective_session.evidence_sha256,
            "batch_count": trace_cache.batch_count,
        }
    except (KeyError, TypeError) as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Trace cache lacks complete residual-basis phase lineage."
        ) from exc
    for name, value in binding.items():
        if name != "batch_count":
            _digest(value, f"trace_cache.{name}")
    if (
        session.get("cache_key") != binding["cache_key"]
        or session.get("manifest_sha256") != binding["manifest_sha256"]
        or session.get("completion_sha256") != binding["completion_sha256"]
        or session.get("session_sha256") != binding["selective_session_sha256"]
        or session.get("batch_count") != binding["batch_count"]
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Selective trace-session seal differs from cache metadata."
        )
    return binding


def _live_bindings(
    parent_model: Any,
    circuit_graph: ModelCircuitGraph,
    trace_cache: Lfm2CircuitTraceCache,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    target_width: int,
    binding_provider: PhaseBindingProvider,
    context: Lfm2BalancedBasisPhaseCacheContext,
    *,
    reuse_inventory_from: _LiveBindings | None = None,
) -> _LiveBindings:
    if (
        not isinstance(circuit_graph, ModelCircuitGraph)
        or circuit_graph.family != "lfm2"
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "circuit_graph must be the canonical LFM2 graph."
        )
    if (
        isinstance(target_width, bool)
        or not isinstance(target_width, int)
        or target_width <= 0
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "target_width must be a positive integer."
        )
    try:
        fresh_graph = _model_inspector_module.inspect_model_circuits(
            parent_model,
            model_name=circuit_graph.model_name,
            tokenizer_fingerprint=circuit_graph.tokenizer_fingerprint,
        )
    except Exception as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Could not freshly inspect the live parent circuit graph."
        ) from exc

    def trainability_binding(graph: ModelCircuitGraph) -> dict[str, Any]:
        rows = tuple(
            sorted(
                (parameter.path, parameter.trainable)
                for node in graph.nodes
                for parameter in node.parameters
            )
        )
        body = {
            "schema": _LFM2_BALANCED_BASIS_GRAPH_EXECUTION_SCHEMA,
            "parameter_spec_count": len(rows),
            "trainable_parameter_spec_count": sum(
                1 for _path, trainable in rows if trainable
            ),
            "all_parameters_frozen": bool(rows)
            and all(not trainable for _path, trainable in rows),
            "ordered_trainability_sha256": canonical_sha256(
                [{"path": path, "trainable": trainable} for path, trainable in rows]
            ),
        }
        return {**body, "execution_state_sha256": canonical_sha256(body)}

    supplied_structural = circuit_graph.as_structural_topology_dict()
    runtime_structural = fresh_graph.as_structural_topology_dict()
    supplied_structural_sha256 = circuit_graph.structural_topology_fingerprint
    runtime_structural_sha256 = fresh_graph.structural_topology_fingerprint
    supplied_trainability = trainability_binding(circuit_graph)
    runtime_trainability = trainability_binding(fresh_graph)
    full_graph_matches = canonical_json_bytes(fresh_graph.as_dict()) == (
        canonical_json_bytes(circuit_graph.as_dict())
    )
    structural_graph_matches = fresh_graph.family == "lfm2" and canonical_json_bytes(
        runtime_structural
    ) == canonical_json_bytes(supplied_structural)
    mismatch_details = {
        "supplied_graph_fingerprint": circuit_graph.fingerprint,
        "runtime_graph_fingerprint": fresh_graph.fingerprint,
        "supplied_structural_topology_fingerprint": (supplied_structural_sha256),
        "runtime_structural_topology_fingerprint": runtime_structural_sha256,
        "runtime_parameter_spec_count": runtime_trainability["parameter_spec_count"],
        "runtime_trainable_parameter_spec_count": runtime_trainability[
            "trainable_parameter_spec_count"
        ],
        "runtime_all_parameters_frozen": runtime_trainability["all_parameters_frozen"],
    }
    if not structural_graph_matches:
        _progress(
            context,
            "balanced_basis_live_graph_mismatch",
            mismatch_kind="structural_topology",
            **mismatch_details,
        )
        raise Lfm2BalancedBasisPhaseCacheError(
            "Supplied circuit graph differs structurally from a fresh live-parent "
            "inspection "
            f"(supplied={circuit_graph.fingerprint}, "
            f"runtime={fresh_graph.fingerprint}, "
            f"supplied_structural={supplied_structural_sha256}, "
            f"runtime_structural={runtime_structural_sha256})."
        )
    executor_freeze_normalized = not full_graph_matches
    if executor_freeze_normalized and not runtime_trainability["all_parameters_frozen"]:
        _progress(
            context,
            "balanced_basis_live_graph_mismatch",
            mismatch_kind="trainability_not_executor_freeze",
            **mismatch_details,
        )
        raise Lfm2BalancedBasisPhaseCacheError(
            "Fresh live-parent graph trainability differs outside the executor "
            "all-frozen contract "
            f"(supplied={circuit_graph.fingerprint}, "
            f"runtime={fresh_graph.fingerprint}, "
            "runtime_trainable_parameter_spec_count="
            f"{runtime_trainability['trainable_parameter_spec_count']}, "
            "runtime_parameter_spec_count="
            f"{runtime_trainability['parameter_spec_count']})."
        )
    supplied = _binding_documents(binding_provider)
    target = supplied["selected_target"]
    if target.get("source_graph_fingerprint") != circuit_graph.fingerprint:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Selected target differs from the supplied pre-freeze parent graph."
        )
    if target.get("hidden_size") != target_width:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Selected target differs from the requested residual width."
        )
    parent = fingerprint_lfm2_parent(parent_model)
    cache = _trace_cache_binding(trace_cache, selective_session)
    implementation_sha256 = _implementation_fingerprint()
    if cache["parent_identity_sha256"] != parent.identity_sha256:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Trace cache is not bound to the live parent."
        )
    if reuse_inventory_from is None:
        inventory = inventory_lfm2_weight_complete_operators(parent_model)
    else:
        if not isinstance(reuse_inventory_from, _LiveBindings):
            raise TypeError("reuse_inventory_from must be _LiveBindings or None.")
        if parent.as_dict() != reuse_inventory_from.parent.as_dict():
            raise Lfm2BalancedBasisPhaseCacheError(
                "Live parent changed while reauthenticating phase bindings."
            )
        if (
            implementation_sha256
            != reuse_inventory_from.document["basis_implementation_sha256"]
        ):
            raise Lfm2BalancedBasisPhaseCacheError(
                "Balanced-basis implementation changed during live reauthentication."
            )
        inventory = reuse_inventory_from.inventory
    if inventory.source_width <= target_width:
        if inventory.source_width < target_width:
            raise Lfm2BalancedBasisPhaseCacheError(
                "Target residual width exceeds the live parent width."
            )
    inventory_sha256 = _digest(
        inventory.manifest.get("inventory_sha256"),
        "inventory.inventory_sha256",
    )
    graph_binding = {
        "family": fresh_graph.family,
        # ``fingerprint`` remains the graph consumed by the basis builder and
        # therefore by its sealed circuit report.
        "fingerprint": fresh_graph.fingerprint,
        "supplied_fingerprint": circuit_graph.fingerprint,
        "runtime_fingerprint": fresh_graph.fingerprint,
        "structural_topology_fingerprint": supplied_structural_sha256,
        "supplied_trainability": supplied_trainability,
        "runtime_trainability": runtime_trainability,
        "executor_freeze_normalized": executor_freeze_normalized,
        "structural_topology_exactly_preserved": True,
        "structural_topology_excluded_parameter_fields": ["trainable"],
    }
    _progress(
        context,
        "balanced_basis_live_graph_authenticated",
        executor_freeze_normalized=executor_freeze_normalized,
        structural_topology_exactly_preserved=True,
        **mismatch_details,
    )
    document = {
        "schema": LFM2_BALANCED_BASIS_BINDINGS_SCHEMA,
        "parent": parent.as_dict(),
        "inventory": inventory.as_dict(),
        "inventory_sha256": inventory_sha256,
        "circuit_graph": graph_binding,
        "trace_cache": cache,
        "basis_implementation_sha256": implementation_sha256,
        "selected_target": target,
        "tokenizer": supplied["tokenizer"],
        "external_dependency_sha256s": _external_dependencies(context),
    }
    return _LiveBindings(
        parent=parent,
        inventory=inventory,
        circuit_graph=fresh_graph,
        document=MappingProxyType(_canonical_mapping(document, "live phase bindings")),
    )


def _numpy_runtime_contract() -> dict[str, Any]:
    try:
        build = np.__config__.show(mode="dicts")
    except TypeError:  # pragma: no cover - pre-2.x NumPy compatibility
        build = {
            "fallback_configuration": str(
                getattr(np.__config__, "CONFIG", "unavailable")
            )
        }
    return {
        "numpy_version": str(np.__version__),
        "numpy_build_config_sha256": canonical_sha256(build),
    }


def _safetensors_runtime_contract() -> dict[str, Any]:
    try:
        import safetensors
        import safetensors._safetensors_rust as safetensors_rust
        import safetensors.torch as safetensors_torch
    except ImportError as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "safetensors runtime is required for phase-cache authentication."
        ) from exc
    files = {
        "package": getattr(safetensors, "__file__", None),
        "torch_adapter": getattr(safetensors_torch, "__file__", None),
        "native_runtime": getattr(safetensors_rust, "__file__", None),
    }
    version = getattr(safetensors, "__version__", None)
    if not isinstance(version, str) or not version:
        raise Lfm2BalancedBasisPhaseCacheError(
            "safetensors runtime does not expose a stable package version."
        )
    return {
        "safetensors_version": version,
        "module_file_sha256s": {
            name: _module_file_sha256(path, f"safetensors.{name}")
            for name, path in sorted(files.items())
        },
    }


def _numeric_runtime_contract(basis_config: Mapping[str, Any]) -> dict[str, Any]:
    torch_build = torch.__config__.show().replace("\r\n", "\n")
    return {
        "schema": LFM2_BALANCED_BASIS_NUMERIC_SCHEMA,
        "basis_config": _canonical_mapping(basis_config, "basis_config"),
        "calculation_device": "cpu",
        "calculation_dtype": "torch.float64",
        "internal_torch_threads": 1,
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "")),
        "torch_build_config_sha256": hashlib.sha256(
            torch_build.encode("utf-8")
        ).hexdigest(),
        **_numpy_runtime_contract(),
        "safetensors_runtime": _safetensors_runtime_contract(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def _module_file_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Lfm2BalancedBasisPhaseCacheError(
            f"Cannot locate implementation module {name}."
        )
    try:
        return hashlib.sha256(Path(value).resolve(strict=True).read_bytes()).hexdigest()
    except OSError as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            f"Cannot authenticate implementation module {name}."
        ) from exc


def _implementation_fingerprint() -> str:
    modules = {
        "balanced_krylov": _balanced_krylov_module.__file__,
        "dynamic_balanced_circuit_basis": _balanced_basis_module.__file__,
        "dynamic_circuit_ir": _circuit_ir_module.__file__,
        "dynamic_fold_phase_cache": _phase_cache_module.__file__,
        "dynamic_model_inspector": _model_inspector_module.__file__,
        "dynamic_weight_complete_basis": _weight_basis_module.__file__,
        "hf_lfm2_balanced_circuit_basis": _lfm2_balanced_module.__file__,
        "hf_lfm2_circuit_trace_cache": _trace_cache_module.__file__,
        "hf_lfm2_circuit_trace_row_selection": _row_selection_module.__file__,
        "hf_lfm2_streaming_vocabulary_margin": _streaming_margin_module.__file__,
        "hf_lfm2_streaming_vocabulary_margin_adapter": (
            _streaming_margin_adapter_module.__file__
        ),
        "hf_lfm2_weight_complete_basis": _lfm2_weight_basis_module.__file__,
        "phase_adapter": __file__,
    }
    return canonical_sha256(
        {
            "schema": LFM2_BALANCED_BASIS_PHASE_ABI,
            "module_file_sha256s": {
                name: _module_file_sha256(value, name)
                for name, value in sorted(modules.items())
            },
            "safetensors_runtime": _safetensors_runtime_contract(),
        }
    )


def _phase_key(
    live: _LiveBindings,
    target_width: int,
    basis_config: Mapping[str, Any],
    anchor_authority: PhaseAnchorAuthority,
) -> FoldPhaseKey:
    bindings = dict(live.document)
    parent = bindings["parent"]
    cache = bindings["trace_cache"]
    runtime = _numeric_runtime_contract(basis_config)
    dependencies = {
        "parent_identity": parent["identity_sha256"],
        "parent_config": parent["config_sha256"],
        "parent_state": parent["state_dict_sha256"],
        "parent_runtime_buffers": parent["runtime_buffer_sha256"],
        "parent_inventory": bindings["inventory_sha256"],
        "source_graph": bindings["circuit_graph"]["supplied_fingerprint"],
        "runtime_source_graph": bindings["circuit_graph"]["runtime_fingerprint"],
        "source_structural_topology": bindings["circuit_graph"][
            "structural_topology_fingerprint"
        ],
        "source_graph_execution_state": canonical_sha256(bindings["circuit_graph"]),
        "trace_cache": cache["cache_key"],
        "trace_manifest": cache["manifest_sha256"],
        "trace_completion": cache["completion_sha256"],
        "trace_corpus": cache["corpus_sha256"],
        "trace_topology": cache["layer_topology_sha256"],
        "trace_implementation": cache["implementation_sha256"],
        "trace_selective_session": cache["selective_session_sha256"],
        "selected_target": canonical_sha256(bindings["selected_target"]),
        "tokenizer_binding": canonical_sha256(bindings["tokenizer"]),
        "basis_implementation": bindings["basis_implementation_sha256"],
        "numeric_runtime": canonical_sha256(runtime),
        "source_anchor_authority": canonical_sha256(
            {
                "schema": LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
                "algorithm": _SOURCE_ANCHOR_ALGORITHM,
                "key_id": anchor_authority.key_id,
            }
        ),
    }
    for name, value in bindings["external_dependency_sha256s"].items():
        key = f"external.{name}"
        if key in dependencies:
            raise Lfm2BalancedBasisPhaseCacheError(
                f"External phase dependency collides at {key!r}."
            )
        dependencies[key] = value
    return FoldPhaseKey(
        phase_name=LFM2_BALANCED_BASIS_PHASE,
        dependency_sha256s=dependencies,
        numeric_contract=runtime,
        abi={
            "schema": LFM2_BALANCED_BASIS_PHASE_ABI,
            "wrapper_schema": LFM2_CIRCUIT_BALANCED_BASIS_SCHEMA,
            "receipt_schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
            "tensor_schema": LFM2_BALANCED_BASIS_TENSOR_SCHEMA,
            "tensor_filename": _TENSOR_FILENAME,
            "receipt_filename": _RECEIPT_FILENAME,
            "source_anchor_filename": _SOURCE_ANCHOR_FILENAME,
            "source_anchor_schema": LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
            "source_anchor_algorithm": _SOURCE_ANCHOR_ALGORITHM,
            "source_anchor_key_id": anchor_authority.key_id,
            "ordered_tensor_names": list(_TENSOR_NAMES),
            "source_width": live.inventory.source_width,
            "target_width": target_width,
            "canonical_device": "cpu",
            "canonical_dtype": "torch.float64",
            "live_vocabulary_candidate_gate_serialized": False,
        },
    )


def _basis_tensors(result: Lfm2CircuitBalancedBasisResult) -> dict[str, torch.Tensor]:
    if not isinstance(result, Lfm2CircuitBalancedBasisResult):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Cold builder did not return Lfm2CircuitBalancedBasisResult."
        )
    tensors = {
        "basis": result.basis,
        "balanced_encoder": result.balanced_encoder,
        "balanced_decoder": result.balanced_decoder,
        "full_weight_gram": result.full_weight_gram,
        "controllability_gram": result.controllability_gram,
        "observability_gram": result.observability_gram,
        "heldout_gram": result.heldout_gram,
    }
    normalized: dict[str, torch.Tensor] = {}
    for name in _TENSOR_NAMES:
        value = tensors[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.ndim != 2
            or not bool(torch.isfinite(value).all().item())
        ):
            raise Lfm2BalancedBasisPhaseCacheError(
                f"Balanced basis tensor {name!r} is not canonical CPU float64."
            )
        normalized[name] = value.detach().contiguous().clone()
    return normalized


def _expected_shapes(source_width: int, target_width: int) -> dict[str, list[int]]:
    return {
        "basis": [source_width, target_width],
        "balanced_encoder": [target_width, source_width],
        "balanced_decoder": [source_width, target_width],
        "full_weight_gram": [source_width, source_width],
        "controllability_gram": [source_width, source_width],
        "observability_gram": [source_width, source_width],
        "heldout_gram": [source_width, source_width],
    }


def _tensor_descriptors(
    tensors: Mapping[str, torch.Tensor],
    *,
    source_width: int,
    target_width: int,
) -> list[dict[str, Any]]:
    if set(tensors) != set(_TENSOR_NAMES):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Balanced basis tensor inventory is missing or foreign."
        )
    shapes = _expected_shapes(source_width, target_width)
    descriptors: list[dict[str, Any]] = []
    for name in _TENSOR_NAMES:
        value = tensors[name]
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or value.layout != torch.strided
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
            or [int(size) for size in value.shape] != shapes[name]
        ):
            raise Lfm2BalancedBasisPhaseCacheError(
                f"Balanced basis tensor {name!r} violates its v1 descriptor."
            )
        descriptors.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": shapes[name],
                "stride": [int(size) for size in value.stride()],
                "numel": int(value.numel()),
                "size_bytes": int(value.numel() * value.element_size()),
                "sha256": _balanced_tensor_sha256(value),
            }
        )
    return descriptors


def _verify_self_seal(
    report: Mapping[str, Any],
    seal_name: str,
    label: str,
) -> dict[str, Any]:
    value = _canonical_mapping(report, label)
    claimed = value.pop(seal_name, None)
    if claimed != canonical_sha256(value):
        raise Lfm2BalancedBasisPhaseCacheError(f"{label} {seal_name} does not verify.")
    return {**value, seal_name: claimed}


def _validate_result(
    result: Lfm2CircuitBalancedBasisResult,
    live: _LiveBindings,
    target_width: int,
    tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    wrapper = _verify_self_seal(result.as_dict(), "result_sha256", "basis report")
    nested = _verify_self_seal(
        result.basis_result.as_dict(),
        "run_sha256",
        "balanced basis report",
    )
    descriptors = _tensor_descriptors(
        tensors,
        source_width=live.inventory.source_width,
        target_width=target_width,
    )
    by_name = {value["name"]: value for value in descriptors}
    hashes = nested.get("hashes")
    wrapper_hashes = wrapper.get("hashes")
    cache_report = wrapper.get("cache")
    graph_report = wrapper.get("circuit_ir")
    if (
        result.inventory.as_dict() != live.inventory.as_dict()
        or result.circuit_graph.fingerprint
        != live.document["circuit_graph"]["fingerprint"]
        or result.source_width != live.inventory.source_width
        or result.target_width != target_width
        or wrapper.get("schema") != LFM2_CIRCUIT_BALANCED_BASIS_SCHEMA
        or wrapper.get("parent") != live.document["parent"]
        or wrapper.get("inventory") != live.document["inventory"]
        or not isinstance(cache_report, dict)
        or cache_report.get("cache_key") != live.document["trace_cache"]["cache_key"]
        or cache_report.get("corpus_sha256")
        != live.document["trace_cache"]["corpus_sha256"]
        or cache_report.get("parent_identity_sha256")
        != live.document["parent"]["identity_sha256"]
        or not isinstance(graph_report, dict)
        or graph_report.get("fingerprint")
        != live.document["circuit_graph"]["fingerprint"]
        or wrapper.get("balanced_basis") != nested
        or not isinstance(wrapper_hashes, dict)
        or wrapper_hashes.get("basis_sha256") != by_name["basis"]["sha256"]
        or wrapper_hashes.get("inventory_sha256") != live.document["inventory_sha256"]
        or wrapper_hashes.get("circuit_graph_sha256")
        != live.document["circuit_graph"]["fingerprint"]
        or not isinstance(hashes, dict)
        or hashes.get("basis_sha256") != by_name["basis"]["sha256"]
        or hashes.get("balanced_encoder_sha256")
        != by_name["balanced_encoder"]["sha256"]
        or hashes.get("balanced_decoder_sha256")
        != by_name["balanced_decoder"]["sha256"]
        or hashes.get("full_weight_gram_sha256")
        != by_name["full_weight_gram"]["sha256"]
        or hashes.get("controllability_gram_sha256")
        != by_name["controllability_gram"]["sha256"]
        or hashes.get("observability_gram_sha256")
        != by_name["observability_gram"]["sha256"]
        or hashes.get("heldout_gram_sha256") != by_name["heldout_gram"]["sha256"]
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Balanced basis result differs from live parent/cache/graph/tensors."
        )
    for report, label in ((wrapper, "basis report"), (nested, "balanced report")):
        execution = report.get("execution")
        if label == "basis report":
            counters = {
                "model_forward_calls": report.get("model_forward_calls"),
                "parent_trace_regeneration_calls": report.get(
                    "parent_trace_regeneration_calls"
                ),
                "training_steps": report.get("training_steps"),
                "backward_calls": report.get("backward_calls"),
                "optimizer_steps": report.get("optimizer_steps"),
            }
        else:
            counters = {
                "model_forward_calls": execution.get("model_forward_calls")
                if isinstance(execution, dict)
                else None,
                "parent_forward_calls": execution.get("parent_forward_calls")
                if isinstance(execution, dict)
                else None,
                "trace_capture_calls": execution.get("trace_capture_calls")
                if isinstance(execution, dict)
                else None,
                "training_steps": execution.get("training_steps")
                if isinstance(execution, dict)
                else None,
                "backward_calls": execution.get("backward_calls")
                if isinstance(execution, dict)
                else None,
                "optimizer_steps": execution.get("optimizer_steps")
                if isinstance(execution, dict)
                else None,
            }
        if any(value != 0 for value in counters.values()):
            raise Lfm2BalancedBasisPhaseCacheError(
                f"{label} makes an execution or training claim."
            )
    return wrapper, nested, descriptors


def _safetensors_metadata(payload: bytes) -> dict[str, str]:
    if len(payload) < 8:
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors payload is truncated."
        )
    size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if size <= 0 or size > len(payload) - 8:
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors header size is invalid."
        )
    try:
        header = json.loads(payload[8 : 8 + size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors header is invalid."
        ) from exc
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors metadata is invalid."
        )
    return metadata


def _save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> bytes:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise Lfm2BalancedBasisPhaseCacheError(
            "safetensors is required for balanced-basis phase caching."
        ) from exc
    if path.exists() or path.is_symlink():
        raise Lfm2BalancedBasisPhaseCacheError(
            "Refusing to replace a staged balanced-basis tensor file."
        )
    try:
        save_file(
            {name: tensors[name] for name in _TENSOR_NAMES},
            str(path),
            metadata=_TENSOR_METADATA,
        )
        return path.read_bytes()
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Could not write the balanced-basis tensor payload."
        ) from exc


def _load_tensors(payload: bytes) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise FoldPhaseCacheIntegrityError(
            "safetensors is required to load a balanced-basis phase."
        ) from exc
    if _safetensors_metadata(payload) != _TENSOR_METADATA:
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors metadata differs from the v1 ABI."
        )
    try:
        loaded = dict(load(payload))
    except (RuntimeError, ValueError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors payload is unreadable."
        ) from exc
    if tuple(sorted(loaded)) != _TENSOR_NAMES:
        raise FoldPhaseCacheIntegrityError(
            "Balanced basis safetensors inventory is missing or foreign."
        )
    return {
        name: loaded[name].detach().to(device="cpu").contiguous().clone()
        for name in _TENSOR_NAMES
    }


def _write_receipt(
    path: Path,
    *,
    phase_key: FoldPhaseKey,
    live: _LiveBindings,
    wrapper: Mapping[str, Any],
    nested: Mapping[str, Any],
    descriptors: list[dict[str, Any]],
    tensor_payload: bytes,
) -> dict[str, Any]:
    tensor_file = {
        "filename": _TENSOR_FILENAME,
        "format": "safetensors",
        "metadata": _TENSOR_METADATA,
        "size_bytes": len(tensor_payload),
        "sha256": hashlib.sha256(tensor_payload).hexdigest(),
    }
    body = {
        "schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
        "phase_key": phase_key.as_dict(),
        "phase_key_sha256": phase_key.phase_key_sha256,
        "bindings": dict(live.document),
        "basis_report": dict(wrapper),
        "balanced_basis_report": dict(nested),
        "tensor_file": tensor_file,
        "tensor_descriptors": descriptors,
        "execution": dict(_EXECUTION_RECEIPT),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    try:
        path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    except OSError as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Could not write balanced-basis phase receipt."
        ) from exc
    return receipt


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase receipt is unreadable."
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != _RECEIPT_FIELDS
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase receipt is not strict canonical v1 JSON."
        )
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if claimed != canonical_sha256(body):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase receipt seal is invalid."
        )
    return value


def _completion_result(receipt: Mapping[str, Any]) -> dict[str, Any]:
    tensor_file = receipt["tensor_file"]
    descriptors = receipt["tensor_descriptors"]
    return {
        "schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
        "receipt_sha256": receipt["receipt_sha256"],
        "tensor_file_sha256": tensor_file["sha256"],
        "tensor_inventory_sha256": canonical_sha256(descriptors),
        "basis_result_sha256": receipt["basis_report"]["result_sha256"],
        "balanced_run_sha256": receipt["balanced_basis_report"]["run_sha256"],
        "basis_map_sha256": next(
            value["sha256"] for value in descriptors if value["name"] == "basis"
        ),
    }


def _source_anchor_body(
    phase_key: FoldPhaseKey,
    receipt: Mapping[str, Any],
    authority: PhaseAnchorAuthority,
) -> dict[str, Any]:
    result = _completion_result(receipt)
    return {
        "schema": LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
        "algorithm": _SOURCE_ANCHOR_ALGORITHM,
        "key_id": authority.key_id,
        "phase_name": LFM2_BALANCED_BASIS_PHASE,
        "phase_key_sha256": phase_key.phase_key_sha256,
        "receipt_sha256": result["receipt_sha256"],
        "tensor_file_sha256": result["tensor_file_sha256"],
        "tensor_inventory_sha256": result["tensor_inventory_sha256"],
        "basis_result_sha256": result["basis_result_sha256"],
        "balanced_run_sha256": result["balanced_run_sha256"],
        "basis_map_sha256": result["basis_map_sha256"],
    }


def _write_source_anchor(
    path: Path,
    *,
    phase_key: FoldPhaseKey,
    receipt: Mapping[str, Any],
    authority: PhaseAnchorAuthority,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise Lfm2BalancedBasisPhaseCacheError(
            "Refusing to replace a staged balanced-basis source anchor."
        )
    body = _source_anchor_body(phase_key, receipt, authority)
    anchor = {**body, "hmac_sha256": authority.sign(body)}
    try:
        path.write_bytes(canonical_json_bytes(anchor) + b"\n")
    except OSError as exc:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Could not write balanced-basis source anchor."
        ) from exc
    return anchor


def _read_source_anchor(
    path: Path,
    *,
    phase_key: FoldPhaseKey,
    receipt: Mapping[str, Any],
    authority: PhaseAnchorAuthority,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis source anchor is unreadable."
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != _SOURCE_ANCHOR_FIELDS
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis source anchor is not strict canonical v1 JSON."
        )
    body = dict(value)
    claimed = body.pop("hmac_sha256", None)
    if (
        set(body) != _SOURCE_ANCHOR_BODY_FIELDS
        or body.get("schema") != LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA
        or body.get("algorithm") != _SOURCE_ANCHOR_ALGORITHM
        or body.get("key_id") != authority.key_id
        or body.get("phase_name") != LFM2_BALANCED_BASIS_PHASE
        or not authority.verify(body, claimed)
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis source anchor HMAC does not verify."
        )
    for name in (
        "phase_key_sha256",
        "receipt_sha256",
        "tensor_file_sha256",
        "tensor_inventory_sha256",
        "basis_result_sha256",
        "balanced_run_sha256",
        "basis_map_sha256",
    ):
        try:
            _digest(body.get(name), f"source anchor {name}")
        except Lfm2BalancedBasisPhaseCacheError as exc:
            raise FoldPhaseCacheIntegrityError(str(exc)) from exc
    expected = _source_anchor_body(phase_key, receipt, authority)
    if body != expected:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis source anchor does not bind the requested phase receipt."
        )
    return value


def _load_entry(
    context: Lfm2BalancedBasisPhaseCacheContext,
    entry: VerifiedFoldPhaseCacheEntry,
    phase_key: FoldPhaseKey,
    live: _LiveBindings,
    target_width: int,
) -> Lfm2CircuitBalancedBasisResult:
    if entry.phase_key.as_dict() != phase_key.as_dict():
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase entry differs from the requested key."
        )
    inventory_files = entry.payload_inventory.get("files")
    if not isinstance(inventory_files, list) or {
        value.get("path") for value in inventory_files if isinstance(value, dict)
    } != {_TENSOR_FILENAME, _RECEIPT_FILENAME, _SOURCE_ANCHOR_FILENAME}:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase payload inventory is missing or foreign."
        )
    receipt = _read_receipt(entry.payload_path / _RECEIPT_FILENAME)
    tensor_file = receipt.get("tensor_file")
    if (
        receipt.get("schema") != LFM2_BALANCED_BASIS_RECEIPT_SCHEMA
        or receipt.get("phase_key") != phase_key.as_dict()
        or receipt.get("phase_key_sha256") != phase_key.phase_key_sha256
        or receipt.get("bindings") != dict(live.document)
        or receipt.get("execution") != _EXECUTION_RECEIPT
        or not isinstance(tensor_file, dict)
        or set(tensor_file) != _TENSOR_FILE_FIELDS
        or tensor_file.get("filename") != _TENSOR_FILENAME
        or tensor_file.get("format") != "safetensors"
        or tensor_file.get("metadata") != _TENSOR_METADATA
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase receipt differs from its live v1 contract."
        )
    _read_source_anchor(
        entry.payload_path / _SOURCE_ANCHOR_FILENAME,
        phase_key=phase_key,
        receipt=receipt,
        authority=context.anchor_authority,
    )
    try:
        tensor_payload = (entry.payload_path / _TENSOR_FILENAME).read_bytes()
    except OSError as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensor payload is unreadable."
        ) from exc
    if (
        tensor_file.get("size_bytes") != len(tensor_payload)
        or tensor_file.get("sha256") != hashlib.sha256(tensor_payload).hexdigest()
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensor file seal is invalid."
        )
    tensors = _load_tensors(tensor_payload)
    try:
        descriptors = _tensor_descriptors(
            tensors,
            source_width=live.inventory.source_width,
            target_width=target_width,
        )
    except Lfm2BalancedBasisPhaseCacheError as exc:
        raise FoldPhaseCacheIntegrityError(str(exc)) from exc
    if receipt.get("tensor_descriptors") != descriptors:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensor descriptors do not verify."
        )
    wrapper = receipt.get("basis_report")
    nested = receipt.get("balanced_basis_report")
    try:
        basis_result = BalancedCircuitBasisResult(
            basis=tensors["basis"],
            balanced_encoder=tensors["balanced_encoder"],
            balanced_decoder=tensors["balanced_decoder"],
            full_weight_gram=tensors["full_weight_gram"],
            controllability_gram=tensors["controllability_gram"],
            observability_gram=tensors["observability_gram"],
            heldout_gram=tensors["heldout_gram"],
            report=nested,
        )
        result = Lfm2CircuitBalancedBasisResult(
            inventory=live.inventory,
            circuit_graph=live.circuit_graph,
            basis_result=basis_result,
            report=wrapper,
            vocabulary_margin_gate=None,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensors or reports violate result invariants."
        ) from exc
    try:
        _validate_result(result, live, target_width, tensors)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase result differs from its authenticated receipt."
        ) from exc
    if entry.result != _completion_result(receipt):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis completion does not bind its exact receipt."
        )
    reauthenticated = context.cache.lookup(phase_key)
    if (
        reauthenticated is None
        or reauthenticated.completion_sha256 != entry.completion_sha256
        or reauthenticated.payload_inventory != entry.payload_inventory
        or reauthenticated.result != entry.result
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase changed while it was loaded."
        )
    return result


def _existing_phase_key_from_digest(
    cache: DynamicFoldPhaseCache,
    phase_key_sha256: str,
) -> FoldPhaseKey:
    """Read only the immutable completion key; never create a cache path."""

    digest = _digest(phase_key_sha256, "phase_key_sha256")
    entry = cache.root / LFM2_BALANCED_BASIS_PHASE / digest
    completion = entry / _phase_cache_module.COMPLETE_FILENAME
    if (
        not entry.is_dir()
        or entry.is_symlink()
        or not completion.is_file()
        or completion.is_symlink()
        or getattr(completion.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Requested balanced-basis phase is absent; hit-only loading forbids "
            "building or repairing it."
        )
    try:
        raw = completion.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase completion is unreadable."
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("phase_key_sha256") != digest
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase completion is not canonical or has a stale key."
        )
    try:
        key = FoldPhaseKey.from_dict(value.get("phase_key"))
    except (TypeError, ValueError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase completion contains an invalid phase key."
        ) from exc
    if key.phase_name != LFM2_BALANCED_BASIS_PHASE or key.phase_key_sha256 != digest:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase key differs from its immutable cache address."
        )
    return key


def load_existing_lfm2_circuit_balanced_basis_phase(
    cache: DynamicFoldPhaseCache,
    phase_key_sha256: str,
    *,
    anchor_authority: PhaseAnchorAuthority,
    expected_parent_identity_sha256: str,
    expected_trace_cache_key: str,
    expected_trace_selective_session_sha256: str,
    expected_source_width: int,
    expected_target_width: int,
    expected_map_sha256: str | None = None,
    events: FoldPhaseEventLog | None = None,
) -> AuthenticatedLfm2BalancedBasisPhase:
    """Load one HMAC-anchored phase hit without a builder or parent model.

    This post-fold surface exists specifically for diagnostics.  A missing
    entry is an error: it never calls a basis builder, opens a transaction,
    loads a trace tensor, or executes a parent forward.
    """

    if not isinstance(cache, DynamicFoldPhaseCache):
        raise TypeError("cache must be DynamicFoldPhaseCache.")
    if not isinstance(anchor_authority, PhaseAnchorAuthority):
        raise TypeError("anchor_authority must be PhaseAnchorAuthority.")
    parent_identity = _digest(
        expected_parent_identity_sha256,
        "expected_parent_identity_sha256",
    )
    trace_key = _digest(expected_trace_cache_key, "expected_trace_cache_key")
    session_sha = _digest(
        expected_trace_selective_session_sha256,
        "expected_trace_selective_session_sha256",
    )
    source_width = _positive_int(expected_source_width, "expected_source_width")
    target_width = _positive_int(expected_target_width, "expected_target_width")
    expected_map = (
        None
        if expected_map_sha256 is None
        else _digest(expected_map_sha256, "expected_map_sha256")
    )
    key = _existing_phase_key_from_digest(cache, phase_key_sha256)
    key_document = key.as_dict()
    dependencies = key_document.get("dependency_sha256s")
    abi = key_document.get("abi")
    if (
        not isinstance(dependencies, dict)
        or not isinstance(abi, dict)
        or dependencies.get("parent_identity") != parent_identity
        or dependencies.get("trace_cache") != trace_key
        or dependencies.get("trace_selective_session") != session_sha
        or abi.get("source_width") != source_width
        or abi.get("target_width") != target_width
        or abi.get("source_anchor_key_id") != anchor_authority.key_id
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase identity differs from the requested parent, "
            "trace session, widths, or HMAC authority."
        )
    entry = cache.lookup(key, events=events)
    if entry is None:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Requested balanced-basis phase is absent; hit-only loading forbids "
            "building or repairing it."
        )
    inventory_files = entry.payload_inventory.get("files")
    if not isinstance(inventory_files, list) or {
        value.get("path") for value in inventory_files if isinstance(value, dict)
    } != {_TENSOR_FILENAME, _RECEIPT_FILENAME, _SOURCE_ANCHOR_FILENAME}:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase payload inventory is missing or foreign."
        )
    receipt = _read_receipt(entry.payload_path / _RECEIPT_FILENAME)
    tensor_file = receipt.get("tensor_file")
    if (
        receipt.get("schema") != LFM2_BALANCED_BASIS_RECEIPT_SCHEMA
        or receipt.get("phase_key") != key_document
        or receipt.get("phase_key_sha256") != key.phase_key_sha256
        or receipt.get("execution") != _EXECUTION_RECEIPT
        or not isinstance(tensor_file, dict)
        or set(tensor_file) != _TENSOR_FILE_FIELDS
        or tensor_file.get("filename") != _TENSOR_FILENAME
        or tensor_file.get("format") != "safetensors"
        or tensor_file.get("metadata") != _TENSOR_METADATA
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase receipt differs from the read-only v1 contract."
        )
    anchor = _read_source_anchor(
        entry.payload_path / _SOURCE_ANCHOR_FILENAME,
        phase_key=key,
        receipt=receipt,
        authority=anchor_authority,
    )
    try:
        tensor_payload = (entry.payload_path / _TENSOR_FILENAME).read_bytes()
    except OSError as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensor payload is unreadable."
        ) from exc
    if (
        tensor_file.get("size_bytes") != len(tensor_payload)
        or tensor_file.get("sha256") != hashlib.sha256(tensor_payload).hexdigest()
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensor file seal is invalid."
        )
    tensors = _load_tensors(tensor_payload)
    descriptors = _tensor_descriptors(
        tensors,
        source_width=source_width,
        target_width=target_width,
    )
    if receipt.get("tensor_descriptors") != descriptors:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensor descriptors do not verify."
        )
    wrapper = _verify_self_seal(
        receipt.get("basis_report"),
        "result_sha256",
        "basis report",
    )
    nested = _verify_self_seal(
        receipt.get("balanced_basis_report"),
        "run_sha256",
        "balanced basis report",
    )
    bindings = receipt.get("bindings")
    trace_binding = bindings.get("trace_cache") if isinstance(bindings, dict) else None
    parent_binding = bindings.get("parent") if isinstance(bindings, dict) else None
    wrapper_cache = wrapper.get("cache")
    descriptor_by_name = {value["name"]: value for value in descriptors}
    wrapper_hashes = wrapper.get("hashes")
    nested_hashes = nested.get("hashes")
    if (
        not isinstance(trace_binding, dict)
        or not isinstance(parent_binding, dict)
        or trace_binding.get("cache_key") != trace_key
        or trace_binding.get("selective_session_sha256") != session_sha
        or parent_binding.get("identity_sha256") != parent_identity
        or not isinstance(wrapper_cache, dict)
        or wrapper_cache.get("cache_key") != trace_key
        or wrapper_cache.get("parent_identity_sha256") != parent_identity
        or wrapper.get("parent") != parent_binding
        or wrapper.get("balanced_basis") != nested
        or not isinstance(wrapper_hashes, dict)
        or not isinstance(nested_hashes, dict)
        or wrapper_hashes.get("basis_sha256") != descriptor_by_name["basis"]["sha256"]
        or nested_hashes.get("basis_sha256") != descriptor_by_name["basis"]["sha256"]
        or nested_hashes.get("balanced_encoder_sha256")
        != descriptor_by_name["balanced_encoder"]["sha256"]
        or nested_hashes.get("balanced_decoder_sha256")
        != descriptor_by_name["balanced_decoder"]["sha256"]
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis reports differ from their authenticated source and map."
        )
    try:
        basis_result = BalancedCircuitBasisResult(
            basis=tensors["basis"],
            balanced_encoder=tensors["balanced_encoder"],
            balanced_decoder=tensors["balanced_decoder"],
            full_weight_gram=tensors["full_weight_gram"],
            controllability_gram=tensors["controllability_gram"],
            observability_gram=tensors["observability_gram"],
            heldout_gram=tensors["heldout_gram"],
            report=nested,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase tensors violate residual-map invariants."
        ) from exc
    if (
        basis_result.source_width != source_width
        or basis_result.target_width != target_width
        or (expected_map is not None and basis_result.map_sha256 != expected_map)
        or entry.result != _completion_result(receipt)
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase map or completion differs from expectation."
        )
    reauthenticated = cache.lookup(key)
    if (
        reauthenticated is None
        or reauthenticated.completion_sha256 != entry.completion_sha256
        or reauthenticated.payload_inventory != entry.payload_inventory
        or reauthenticated.result != entry.result
    ):
        raise FoldPhaseCacheIntegrityError(
            "Balanced-basis phase changed while it was read."
        )
    return AuthenticatedLfm2BalancedBasisPhase(
        phase_key=key,
        completion_sha256=entry.completion_sha256,
        receipt=MappingProxyType(dict(receipt)),
        source_anchor=MappingProxyType(dict(anchor)),
        basis_result=basis_result,
        report=MappingProxyType(dict(wrapper)),
    )


def _load_existing_lfm2_circuit_balanced_basis_for_live_fold(
    context: Lfm2BalancedBasisPhaseCacheContext,
    request: Lfm2ExistingBalancedBasisPhaseReuseRequest,
    parent_model: Any,
    trace_cache: Lfm2CircuitTraceCache,
    target_width: int,
    *,
    circuit_graph: ModelCircuitGraph,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    build_request: Lfm2BalancedBasisPhaseBuildRequest,
    binding_provider: PhaseBindingProvider,
) -> Lfm2CircuitBalancedBasisResult:
    """Rebind one authenticated historical map to an exact live parent.

    The historical receipt, tensors, phase key, and source anchor remain
    immutable.  Only the historical producer implementation hashes and its
    runtime contract may differ from the current consumer.  Every semantic
    binding and every basis-math setting remains exact.
    """

    live = _live_bindings(
        parent_model,
        circuit_graph,
        trace_cache,
        selective_session,
        target_width,
        binding_provider,
        context,
    )
    cache_binding = live.document["trace_cache"]
    parent_binding = live.document["parent"]
    authenticated = load_existing_lfm2_circuit_balanced_basis_phase(
        context.cache,
        request.phase_key_sha256,
        anchor_authority=context.anchor_authority,
        expected_parent_identity_sha256=parent_binding["identity_sha256"],
        expected_trace_cache_key=cache_binding["cache_key"],
        expected_trace_selective_session_sha256=(
            cache_binding["selective_session_sha256"]
        ),
        expected_source_width=live.inventory.source_width,
        expected_target_width=target_width,
        expected_map_sha256=request.expected_map_sha256,
        events=context.events,
    )

    receipt_bindings = _canonical_mapping(
        authenticated.receipt.get("bindings"),
        "historical phase bindings",
    )
    current_bindings = _canonical_mapping(
        dict(live.document),
        "current live phase bindings",
    )
    historical_implementation = receipt_bindings.pop(
        "basis_implementation_sha256",
        None,
    )
    current_implementation = current_bindings.pop(
        "basis_implementation_sha256",
        None,
    )
    _digest(historical_implementation, "historical basis implementation")
    _digest(current_implementation, "current basis implementation")
    if receipt_bindings != current_bindings:
        raise FoldPhaseCacheIntegrityError(
            "Historical balanced-basis phase differs from the current parent, "
            "inventory, graph, trace/session, target, tokenizer, or external "
            "dependency bindings."
        )

    historical_key_document = authenticated.phase_key.as_dict()
    historical_runtime = _canonical_mapping(
        historical_key_document["numeric_contract"],
        "historical numeric runtime",
    )
    historical_basis_config = _canonical_mapping(
        historical_runtime.get("basis_config"),
        "historical basis config",
    )
    current_basis_config = _canonical_mapping(
        build_request.as_basis_config(),
        "current basis config",
    )
    historical_adapter_source = historical_basis_config.pop(
        "adapter_source_sha256",
        None,
    )
    current_adapter_source = current_basis_config.pop(
        "adapter_source_sha256",
        None,
    )
    _digest(historical_adapter_source, "historical adapter source")
    _digest(current_adapter_source, "current adapter source")
    if historical_basis_config != current_basis_config:
        raise FoldPhaseCacheIntegrityError(
            "Historical balanced-basis phase uses different basis-math settings."
        )

    current_key = _phase_key(
        live,
        target_width,
        _canonical_mapping(
            build_request.as_basis_config(),
            "current basis config",
        ),
        context.anchor_authority,
    )
    current_key_document = current_key.as_dict()
    historical_dependencies = dict(
        historical_key_document["dependency_sha256s"]
    )
    current_dependencies = dict(current_key_document["dependency_sha256s"])
    for name in ("basis_implementation", "numeric_runtime"):
        historical_dependencies.pop(name, None)
        current_dependencies.pop(name, None)
    if (
        historical_dependencies != current_dependencies
        or historical_key_document["abi"] != current_key_document["abi"]
    ):
        raise FoldPhaseCacheIntegrityError(
            "Historical balanced-basis phase has incompatible semantic "
            "dependencies or ABI."
        )

    provenance_body = {
        "schema": "neural-forge-lfm2-balanced-basis-live-reuse-v1",
        "request_sha256": request.as_dict()["request_sha256"],
        "historical_phase_key_sha256": (
            authenticated.phase_key.phase_key_sha256
        ),
        "historical_completion_sha256": authenticated.completion_sha256,
        "historical_map_sha256": authenticated.map_sha256,
        "historical_basis_implementation_sha256": historical_implementation,
        "current_basis_implementation_sha256": current_implementation,
        "historical_adapter_source_sha256": historical_adapter_source,
        "current_adapter_source_sha256": current_adapter_source,
        "historical_numeric_runtime_sha256": canonical_sha256(
            historical_runtime
        ),
        "current_numeric_runtime_sha256": canonical_sha256(
            current_key_document["numeric_contract"]
        ),
        "semantic_bindings_exact": True,
        "basis_math_settings_exact": True,
        "historical_artifact_rewritten": False,
        "cache_miss_build_allowed": False,
        "trace_tensor_loads": 0,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    provenance = {
        **provenance_body,
        "reuse_sha256": canonical_sha256(provenance_body),
    }
    try:
        result = Lfm2CircuitBalancedBasisResult(
            inventory=live.inventory,
            circuit_graph=live.circuit_graph,
            basis_result=authenticated.basis_result,
            report=authenticated.report,
            vocabulary_margin_gate=None,
            reuse_provenance=provenance,
        )
        _validate_result(
            result,
            live,
            target_width,
            _basis_tensors(result),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise FoldPhaseCacheIntegrityError(
            "Historical balanced basis cannot be rebound to the live fold."
        ) from exc

    final_live = _live_bindings(
        parent_model,
        circuit_graph,
        trace_cache,
        selective_session,
        target_width,
        binding_provider,
        context,
        reuse_inventory_from=live,
    )
    if dict(final_live.document) != dict(live.document):
        raise FoldPhaseCacheIntegrityError(
            "Live residual-basis dependencies changed during historical reuse."
        )
    _progress(
        context,
        "balanced_basis_existing_phase_reuse_hit",
        basis_reuse_provenance_sha256=provenance["reuse_sha256"],
        basis_reuse_historical_phase_key_sha256=provenance[
            "historical_phase_key_sha256"
        ],
        basis_reuse_historical_completion_sha256=provenance[
            "historical_completion_sha256"
        ],
        basis_reuse_historical_map_sha256=provenance[
            "historical_map_sha256"
        ],
        basis_reuse_semantic_bindings_exact=provenance[
            "semantic_bindings_exact"
        ],
        basis_reuse_math_settings_exact=provenance[
            "basis_math_settings_exact"
        ],
        basis_reuse_historical_artifact_rewritten=provenance[
            "historical_artifact_rewritten"
        ],
        basis_reuse_cache_miss_build_allowed=provenance[
            "cache_miss_build_allowed"
        ],
        basis_reuse_trace_tensor_loads=provenance["trace_tensor_loads"],
        basis_reuse_parent_forward_calls=provenance["parent_forward_calls"],
        basis_reuse_training_steps=provenance["training_steps"],
        basis_reuse_optimizer_steps=provenance["optimizer_steps"],
        basis_reuse_backward_calls=provenance["backward_calls"],
    )
    return result


def load_or_build_lfm2_circuit_balanced_basis(
    context: Lfm2BalancedBasisPhaseCacheContext,
    parent_model: Any,
    trace_cache: Lfm2CircuitTraceCache,
    target_width: int,
    *,
    circuit_graph: ModelCircuitGraph,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None,
    build_request: Lfm2BalancedBasisPhaseBuildRequest,
    binding_provider: PhaseBindingProvider,
    live_vocabulary_candidate_gate_required: bool,
) -> Lfm2CircuitBalancedBasisResult:
    """Return one cold-built or authenticated cached balanced residual basis."""

    if not isinstance(context, Lfm2BalancedBasisPhaseCacheContext):
        raise TypeError("context must be Lfm2BalancedBasisPhaseCacheContext.")
    if not isinstance(live_vocabulary_candidate_gate_required, bool):
        raise TypeError("live_vocabulary_candidate_gate_required must be boolean.")
    if not isinstance(build_request, Lfm2BalancedBasisPhaseBuildRequest):
        raise TypeError("build_request must be Lfm2BalancedBasisPhaseBuildRequest.")
    reuse_request = context.existing_phase_reuse_request
    if live_vocabulary_candidate_gate_required and reuse_request is not None:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Historical balanced-basis reuse cannot restore the live "
            "vocabulary candidate gate."
        )
    if live_vocabulary_candidate_gate_required:
        _progress(
            context,
            "balanced_basis_phase_cache_bypassed",
            reason="live_vocabulary_candidate_gate_required",
        )
        return build_request.build(
            parent_model,
            trace_cache,
            target_width,
            circuit_graph=circuit_graph,
            selective_session=selective_session,
        )
    if selective_session is None:
        raise Lfm2BalancedBasisPhaseCacheError(
            "Reusable balanced-basis caching requires a selective authenticated "
            "session so hits never load trace tensors."
        )
    if reuse_request is not None:
        return _load_existing_lfm2_circuit_balanced_basis_for_live_fold(
            context,
            reuse_request,
            parent_model,
            trace_cache,
            target_width,
            circuit_graph=circuit_graph,
            selective_session=selective_session,
            build_request=build_request,
            binding_provider=binding_provider,
        )
    numeric_config = _canonical_mapping(
        build_request.as_basis_config(),
        "basis_config",
    )
    live = _live_bindings(
        parent_model,
        circuit_graph,
        trace_cache,
        selective_session,
        target_width,
        binding_provider,
        context,
    )
    key = _phase_key(
        live,
        target_width,
        numeric_config,
        context.anchor_authority,
    )
    _progress(
        context,
        "balanced_basis_phase_cache_lookup",
        phase_key_sha256=key.phase_key_sha256,
    )
    entry = context.cache.lookup(key, events=context.events)
    if entry is not None:
        result = _load_entry(context, entry, key, live, target_width)
        final_live = _live_bindings(
            parent_model,
            circuit_graph,
            trace_cache,
            selective_session,
            target_width,
            binding_provider,
            context,
            reuse_inventory_from=live,
        )
        if dict(final_live.document) != dict(live.document):
            raise FoldPhaseCacheIntegrityError(
                "Live residual-basis dependencies changed during cache hit."
            )
        _progress(
            context,
            "balanced_basis_phase_cache_hit",
            phase_key_sha256=key.phase_key_sha256,
            completion_sha256=entry.completion_sha256,
            trace_tensor_loads=0,
            parent_forward_calls=0,
            training_steps=0,
        )
        return result

    _progress(
        context,
        "balanced_basis_phase_cache_miss",
        phase_key_sha256=key.phase_key_sha256,
    )
    with _forward_row_selection_progress(context):
        cold = build_request.build(
            parent_model,
            trace_cache,
            target_width,
            circuit_graph=live.circuit_graph,
            selective_session=selective_session,
        )
    cold_tensors = _basis_tensors(cold)
    cold_wrapper, cold_nested, cold_descriptors = _validate_result(
        cold,
        live,
        target_width,
        cold_tensors,
    )
    post_build_live = _live_bindings(
        parent_model,
        circuit_graph,
        trace_cache,
        selective_session,
        target_width,
        binding_provider,
        context,
        reuse_inventory_from=live,
    )
    if dict(post_build_live.document) != dict(live.document):
        raise Lfm2BalancedBasisPhaseCacheError(
            "Live residual-basis dependencies changed during cold build."
        )
    try:
        transaction = context.cache.begin(key, events=context.events)
    except FoldPhaseCacheAlreadyCompleteError:
        raced = context.cache.lookup(key, events=context.events)
        if raced is None:  # pragma: no cover - defensive concurrent deletion
            raise FoldPhaseCacheIntegrityError(
                "Concurrent balanced-basis publication disappeared."
            )
        raced_result = _load_entry(
            context,
            raced,
            key,
            post_build_live,
            target_width,
        )
        raced_final_live = _live_bindings(
            parent_model,
            circuit_graph,
            trace_cache,
            selective_session,
            target_width,
            binding_provider,
            context,
            reuse_inventory_from=live,
        )
        if dict(raced_final_live.document) != dict(live.document):
            raise FoldPhaseCacheIntegrityError(
                "Live residual-basis dependencies changed during raced load."
            )
        return raced_result
    with transaction:
        tensor_payload = _save_tensors(
            transaction.payload_path / _TENSOR_FILENAME,
            cold_tensors,
        )
        receipt = _write_receipt(
            transaction.payload_path / _RECEIPT_FILENAME,
            phase_key=key,
            live=post_build_live,
            wrapper=cold_wrapper,
            nested=cold_nested,
            descriptors=cold_descriptors,
            tensor_payload=tensor_payload,
        )
        _write_source_anchor(
            transaction.payload_path / _SOURCE_ANCHOR_FILENAME,
            phase_key=key,
            receipt=receipt,
            authority=context.anchor_authority,
        )
        entry = transaction.complete(result=_completion_result(receipt))
    loaded = _load_entry(context, entry, key, post_build_live, target_width)
    loaded_tensors = _basis_tensors(loaded)
    if (
        any(
            not torch.equal(cold_tensors[name], loaded_tensors[name])
            for name in _TENSOR_NAMES
        )
        or canonical_json_bytes(cold_wrapper) != canonical_json_bytes(loaded.as_dict())
        or canonical_json_bytes(cold_nested)
        != canonical_json_bytes(loaded.basis_result.as_dict())
    ):
        raise FoldPhaseCacheIntegrityError(
            "Cold and reloaded balanced-basis results are not exact."
        )
    final_live = _live_bindings(
        parent_model,
        circuit_graph,
        trace_cache,
        selective_session,
        target_width,
        binding_provider,
        context,
        reuse_inventory_from=live,
    )
    if dict(final_live.document) != dict(live.document):
        raise FoldPhaseCacheIntegrityError(
            "Live residual-basis dependencies changed during cold reload."
        )
    _progress(
        context,
        "balanced_basis_phase_cache_published",
        phase_key_sha256=key.phase_key_sha256,
        completion_sha256=entry.completion_sha256,
        cold_reload_exact=True,
    )
    return loaded


__all__ = [
    "AuthenticatedLfm2BalancedBasisPhase",
    "LFM2_BALANCED_BASIS_PHASE",
    "LFM2_BALANCED_BASIS_PHASE_ABI",
    "LFM2_BALANCED_BASIS_RECEIPT_SCHEMA",
    "LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA",
    "LFM2_BALANCED_BASIS_TENSOR_DESCRIPTOR_FIELDS",
    "Lfm2BalancedBasisPhaseBuildRequest",
    "Lfm2BalancedBasisPhaseCacheContext",
    "Lfm2BalancedBasisPhaseCacheError",
    "Lfm2ExistingBalancedBasisPhaseReuseRequest",
    "PhaseAnchorAuthority",
    "load_existing_lfm2_circuit_balanced_basis_phase",
    "load_or_build_lfm2_circuit_balanced_basis",
]
