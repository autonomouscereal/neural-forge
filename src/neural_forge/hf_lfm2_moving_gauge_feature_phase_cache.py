"""Authenticated child-only writer-feature import for LFM2 moving-gauge folds.

The moving-gauge boundary phase deliberately stores residual-width tensors.
The two inputs needed to refit the first physical writers are not necessarily
residual width: a heterogeneous ShortConv context uses its internal channel
width and a SwiGLU activation uses the FFN intermediate width.  This module
imports those two tensors from the already completed immutable child selective
cache without constructing or forwarding a model.

Every cold shard is admitted only after the selective source session verifies,
is sealed into a domain-separated HMAC checkpoint chain, and is atomically
published through :class:`DynamicFoldPhaseCache`.  The completed moving-gauge
boundary row index is reused byte-for-byte and authenticated before any source
tensor is selected.  On a phase-cache hit the deferred source provider is never
called.  Tensor bytes are authenticated lazily when a field is requested.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from .hf_lfm2_circuit_trace_cache import (
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
    _tensor_descriptor as _circuit_trace_tensor_descriptor,
)
from .hf_lfm2_exact4x_causal_diagnostic import equal_record_mass_weights
from .hf_lfm2_moving_gauge_boundary_phase_cache import (
    LFM2_MOVING_GAUGE_BOUNDARY_ROW_INDEX_SCHEMA,
    AuthenticatedLfm2MovingGaugeBoundaryPhase,
)
from .hf_lfm2_moving_gauge_residual_reclosure import (
    fingerprint_lfm2_moving_gauge_tensor,
)


LFM2_MOVING_GAUGE_FEATURE_PHASE = "lfm2-moving-gauge-writer-feature-import-v1"
LFM2_MOVING_GAUGE_FEATURE_REQUEST_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-writer-feature-request-v1"
)
LFM2_MOVING_GAUGE_FEATURE_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-writer-feature-checkpoint-v1"
)
LFM2_MOVING_GAUGE_FEATURE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-writer-feature-receipt-v1"
)
LFM2_MOVING_GAUGE_FEATURE_ANCHOR_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-writer-feature-anchor-v1"
)
LFM2_MOVING_GAUGE_FEATURE_SHARD_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-writer-feature-shard-v1"
)

_FEATURE_FIELDS = ("short_conv.context", "mlp.post_swiglu")
_FEATURE_WRITER_ROLES = {
    "short_conv.context": "short_conv.out_proj",
    "mlp.post_swiglu": "mlp.w2",
}
_CHECKPOINT_DOMAIN = b"neural-forge-lfm2-moving-gauge-writer-feature-checkpoint-v1\0"
_ANCHOR_DOMAIN = b"neural-forge-lfm2-moving-gauge-writer-feature-anchor-v1\0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_PHASE = re.compile(r"(?:prefill|decode\.(\d{3}))\Z")
_ZERO_SHA256 = "0" * 64
_BASIS_SPLITS = frozenset({"fit", "heldout"})
_ROW_FIELDS = frozenset(
    {
        "alignment_sha256",
        "basis_split",
        "bridge_split",
        "child_input_sha256",
        "global_row_index",
        "parent_input_sha256",
        "phase",
        "phase_row_index",
        "record_id",
        "record_ordinal",
    }
)
_TENSOR_DESCRIPTOR_FIELDS = frozenset(
    {"dtype", "numel", "sha256", "shape", "size_bytes"}
)
_SOURCE_RECORD_FIELDS = frozenset(
    {
        "alignment_sha256",
        "batch_index",
        "input_sha256",
        "record_sha256",
        "selected_payload_sha256",
    }
)
_SHARD_FIELDS = frozenset(
    {
        "file_sha256",
        "filename",
        "size_bytes",
        "tensor_descriptors",
        "tensor_inventory_sha256",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "batch_start",
        "batch_stop",
        "checkpoint_hmac_key_id",
        "checkpoint_hmac_sha256",
        "checkpoint_ordinal",
        "checkpoint_sha256",
        "execution",
        "phase_key_sha256",
        "previous_checkpoint_hmac_sha256",
        "request_sha256",
        "row_index_slice_sha256",
        "row_start",
        "row_stop",
        "rows",
        "schema",
        "shard",
        "source_cache_key",
        "source_records",
        "source_selective_session_sha256",
        "source_storage_dtypes_preserved",
        "unselected_tensor_payloads_loaded",
    }
)
_ZERO_EXECUTION = {
    "parent_model_objects_constructed": 0,
    "child_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "parent_trace_recaptures": 0,
    "child_trace_recaptures": 0,
    "model_mutations": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2MovingGaugeFeaturePhaseError(RuntimeError):
    """Base error for the child writer-feature phase."""


class Lfm2MovingGaugeFeaturePhaseAuthenticationError(Lfm2MovingGaugeFeaturePhaseError):
    """Raised when source or published feature evidence fails closed."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2MovingGaugeFeaturePhaseError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or _IDENTIFIER.fullmatch(value) is None:
        raise Lfm2MovingGaugeFeaturePhaseError(f"{name} is not a safe identifier.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2MovingGaugeFeaturePhaseError(f"{name} must be a positive integer.")
    return value


def _canonical(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"{name} is not canonical JSON."
        ) from exc


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Required feature document is absent or unsafe: {path}."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature document is unreadable: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature document must contain one mapping: {path}."
        )
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with stage.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": fingerprint_lfm2_moving_gauge_tensor(tensor),
    }


def _phase_sort_key(value: Any) -> tuple[int, int]:
    if not isinstance(value, str):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Source phase must be a string."
        )
    match = _PHASE.fullmatch(value)
    if match is None:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Unsupported source trace phase {value!r}."
        )
    decode = match.group(1)
    return (0, 0) if decode is None else (1, int(decode))


def lfm2_moving_gauge_feature_implementation_sha256() -> str:
    """Fingerprint the executing importer; requests cannot silently change code."""

    return _file_sha256(Path(__file__).resolve())


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeWriterFeatureBinding:
    """Variable-width activation and the physical writer that consumes it."""

    field_name: str
    writer_role: str
    source_dtype: str
    feature_width: int
    writer_topology: str
    writer_weight_shapes: tuple[tuple[int, int], ...]
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        if self.field_name not in _FEATURE_FIELDS:
            raise Lfm2MovingGaugeFeaturePhaseError(
                f"Unsupported writer feature {self.field_name!r}."
            )
        expected_role = _FEATURE_WRITER_ROLES[self.field_name]
        if self.writer_role != expected_role:
            raise Lfm2MovingGaugeFeaturePhaseError(
                f"{self.field_name} must bind writer role {expected_role}."
            )
        object.__setattr__(
            self,
            "source_dtype",
            _identifier(self.source_dtype, "source_dtype"),
        )
        object.__setattr__(
            self,
            "feature_width",
            _positive_int(self.feature_width, "feature_width"),
        )
        topology = _identifier(self.writer_topology, "writer_topology")
        if topology not in {"dense", "factorized"}:
            raise Lfm2MovingGaugeFeaturePhaseError(
                "writer_topology must be dense or factorized."
            )
        shapes = tuple(tuple(shape) for shape in self.writer_weight_shapes)
        if (
            not shapes
            or any(
                len(shape) != 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
                for shape in shapes
            )
            or shapes[0][1] != self.feature_width
            or any(left[0] != right[1] for left, right in zip(shapes, shapes[1:]))
            or (topology == "dense" and len(shapes) != 1)
            or (topology == "factorized" and len(shapes) < 2)
        ):
            raise Lfm2MovingGaugeFeaturePhaseError(
                "Writer weight shapes do not form a valid feature-to-output chain."
            )
        object.__setattr__(self, "writer_topology", topology)
        object.__setattr__(self, "writer_weight_shapes", shapes)
        expected = canonical_sha256(self._body())
        if self.binding_sha256 not in {"", expected}:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "Writer feature binding seal is invalid."
            )
        object.__setattr__(self, "binding_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "writer_role": self.writer_role,
            "source_dtype": self.source_dtype,
            "feature_width": self.feature_width,
            "writer_topology": self.writer_topology,
            "writer_weight_shapes": [
                list(shape) for shape in self.writer_weight_shapes
            ],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "binding_sha256": self.binding_sha256}


def derive_lfm2_moving_gauge_writer_feature_bindings(
    child_layer_contract: Mapping[str, Any],
    *,
    source_dtype_by_field: Mapping[str, str],
) -> tuple[Lfm2MovingGaugeWriterFeatureBinding, ...]:
    """Derive both feature widths from the sealed L0 writer topology."""

    contract = _canonical(dict(child_layer_contract), "child layer contract")
    if not isinstance(contract, dict):  # pragma: no cover - canonical guard
        raise Lfm2MovingGaugeFeaturePhaseError("Child layer contract is invalid.")
    if set(source_dtype_by_field) != set(_FEATURE_FIELDS):
        raise Lfm2MovingGaugeFeaturePhaseError(
            "source_dtype_by_field must cover both writer features exactly."
        )
    hidden = contract.get("hidden_size")
    intermediate = contract.get("ffn_intermediate_size")
    operator = contract.get("operator")
    if (
        contract.get("layer_index") != 0
        or contract.get("kind") != "short_conv"
        or isinstance(hidden, bool)
        or not isinstance(hidden, int)
        or hidden <= 0
        or isinstance(intermediate, bool)
        or not isinstance(intermediate, int)
        or intermediate <= 0
        or not isinstance(operator, dict)
        or operator.get("operator") != "short_conv"
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child L0 is not a canonical ShortConv plus gated-MLP contract."
        )
    if "output_projection_shapes" in operator:
        raw_shapes = operator.get("output_projection_shapes")
        topology = operator.get("writer_topology")
    else:
        raw_shape = operator.get("output_projection_shape")
        raw_shapes = [raw_shape] if isinstance(raw_shape, list) else None
        topology = "dense"
    if not isinstance(raw_shapes, list):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "ShortConv output-writer shapes are absent from the topology."
        )
    if any(
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        )
        for shape in raw_shapes
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "ShortConv output-writer shapes are malformed."
        )
    short_shapes = tuple(tuple(shape) for shape in raw_shapes)
    short = Lfm2MovingGaugeWriterFeatureBinding(
        field_name="short_conv.context",
        writer_role="short_conv.out_proj",
        source_dtype=source_dtype_by_field["short_conv.context"],
        feature_width=short_shapes[0][1]
        if short_shapes and len(short_shapes[0]) == 2
        else 0,
        writer_topology=str(topology),
        writer_weight_shapes=short_shapes,
    )
    mlp = Lfm2MovingGaugeWriterFeatureBinding(
        field_name="mlp.post_swiglu",
        writer_role="mlp.w2",
        source_dtype=source_dtype_by_field["mlp.post_swiglu"],
        feature_width=intermediate,
        writer_topology="dense",
        writer_weight_shapes=((hidden, intermediate),),
    )
    if short.writer_weight_shapes[-1][0] != hidden:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "ShortConv output writer does not terminate at residual width."
        )
    return (short, mlp)


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeFeatureImportRequest:
    """Immutable child-only import identity, including its boundary authority."""

    child_cache_key: str
    child_manifest_sha256: str
    child_completion_sha256: str
    child_identity_sha256: str
    child_state_dict_sha256: str
    child_topology_sha256: str
    child_selective_session_sha256: str
    child_layer_contract_sha256: str
    boundary_phase_key_sha256: str
    boundary_completion_sha256: str
    boundary_request_sha256: str
    boundary_receipt_sha256: str
    boundary_row_index_sha256: str
    row_alignment_sha256: str
    split_identity_sha256: str
    expected_batch_count: int
    expected_aligned_row_count: int
    logical_layer_index: int
    child_layer_index: int
    feature_bindings: tuple[Lfm2MovingGaugeWriterFeatureBinding, ...]
    implementation_sha256: str
    checkpoint_hmac_key_id: str
    shard_batch_count: int = 32
    request_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "child_cache_key",
            "child_manifest_sha256",
            "child_completion_sha256",
            "child_identity_sha256",
            "child_state_dict_sha256",
            "child_topology_sha256",
            "child_selective_session_sha256",
            "child_layer_contract_sha256",
            "boundary_phase_key_sha256",
            "boundary_completion_sha256",
            "boundary_request_sha256",
            "boundary_receipt_sha256",
            "boundary_row_index_sha256",
            "row_alignment_sha256",
            "split_identity_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in (
            "expected_batch_count",
            "expected_aligned_row_count",
            "shard_batch_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("logical_layer_index", "child_layer_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Lfm2MovingGaugeFeaturePhaseError(
                    f"{name} must be a nonnegative integer."
                )
        if self.logical_layer_index != 0:
            raise Lfm2MovingGaugeFeaturePhaseError(
                "The first writer-feature phase is intentionally restricted to L0."
            )
        bindings = tuple(self.feature_bindings)
        if tuple(value.field_name for value in bindings) != _FEATURE_FIELDS or any(
            not isinstance(value, Lfm2MovingGaugeWriterFeatureBinding)
            for value in bindings
        ):
            raise Lfm2MovingGaugeFeaturePhaseError(
                "feature_bindings must contain ShortConv context then post-SwiGLU."
            )
        object.__setattr__(self, "feature_bindings", bindings)
        implementation = _digest(self.implementation_sha256, "implementation_sha256")
        if implementation != lfm2_moving_gauge_feature_implementation_sha256():
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "Feature request implementation differs from executing source."
            )
        object.__setattr__(self, "implementation_sha256", implementation)
        object.__setattr__(
            self,
            "checkpoint_hmac_key_id",
            _identifier(self.checkpoint_hmac_key_id, "checkpoint_hmac_key_id"),
        )
        expected = canonical_sha256(self._body())
        if self.request_sha256 not in {"", expected}:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "Feature import request seal is invalid."
            )
        object.__setattr__(self, "request_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_MOVING_GAUGE_FEATURE_REQUEST_SCHEMA,
            "phase_name": LFM2_MOVING_GAUGE_FEATURE_PHASE,
            "child_cache_key": self.child_cache_key,
            "child_manifest_sha256": self.child_manifest_sha256,
            "child_completion_sha256": self.child_completion_sha256,
            "child_identity_sha256": self.child_identity_sha256,
            "child_state_dict_sha256": self.child_state_dict_sha256,
            "child_topology_sha256": self.child_topology_sha256,
            "child_selective_session_sha256": self.child_selective_session_sha256,
            "child_layer_contract_sha256": self.child_layer_contract_sha256,
            "boundary_phase_key_sha256": self.boundary_phase_key_sha256,
            "boundary_completion_sha256": self.boundary_completion_sha256,
            "boundary_request_sha256": self.boundary_request_sha256,
            "boundary_receipt_sha256": self.boundary_receipt_sha256,
            "boundary_row_index_sha256": self.boundary_row_index_sha256,
            "row_alignment_sha256": self.row_alignment_sha256,
            "split_identity_sha256": self.split_identity_sha256,
            "expected_batch_count": self.expected_batch_count,
            "expected_aligned_row_count": self.expected_aligned_row_count,
            "logical_layer_index": self.logical_layer_index,
            "child_layer_index": self.child_layer_index,
            "feature_bindings": [value.as_dict() for value in self.feature_bindings],
            "implementation_sha256": self.implementation_sha256,
            "checkpoint_hmac_key_id": self.checkpoint_hmac_key_id,
            "shard_batch_count": self.shard_batch_count,
            "variable_width_features": True,
            "feature_width_is_not_residual_width": True,
            "source_storage_dtypes_preserved": True,
            "boundary_row_index_reused_exactly": True,
            "tensor_bytes_authenticated_lazily_on_field_access": True,
            "operation_contract": dict(_ZERO_EXECUTION),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "request_sha256": self.request_sha256}

    def verify_authentication(self) -> bool:
        try:
            return bool(
                self.request_sha256 == canonical_sha256(self._body())
                and self.implementation_sha256
                == lfm2_moving_gauge_feature_implementation_sha256()
                and all(
                    value.binding_sha256 == canonical_sha256(value._body())
                    for value in self.feature_bindings
                )
            )
        except (OSError, TypeError, ValueError):
            return False

    @property
    def phase_key(self) -> FoldPhaseKey:
        dependencies = {
            "request": self.request_sha256,
            "child_cache_key": self.child_cache_key,
            "child_manifest": self.child_manifest_sha256,
            "child_completion": self.child_completion_sha256,
            "child_identity": self.child_identity_sha256,
            "child_state_dict": self.child_state_dict_sha256,
            "child_topology": self.child_topology_sha256,
            "child_selective_session": self.child_selective_session_sha256,
            "child_layer_contract": self.child_layer_contract_sha256,
            "boundary_phase_key": self.boundary_phase_key_sha256,
            "boundary_completion": self.boundary_completion_sha256,
            "boundary_request": self.boundary_request_sha256,
            "boundary_receipt": self.boundary_receipt_sha256,
            "boundary_row_index": self.boundary_row_index_sha256,
            "row_alignment": self.row_alignment_sha256,
            "split_identity": self.split_identity_sha256,
            **{
                f"feature.{value.field_name}": value.binding_sha256
                for value in self.feature_bindings
            },
        }
        numeric = {
            "expected_batch_count": self.expected_batch_count,
            "expected_aligned_row_count": self.expected_aligned_row_count,
            "logical_layer_index": self.logical_layer_index,
            "child_layer_index": self.child_layer_index,
            "shard_batch_count": self.shard_batch_count,
            "feature_widths": {
                value.field_name: value.feature_width for value in self.feature_bindings
            },
            "source_dtypes": {
                value.field_name: value.source_dtype for value in self.feature_bindings
            },
            "phase_order": "prefill-then-numeric-decode",
            "row_order": "completed-boundary-global-row-index",
        }
        abi = {
            "request_schema": LFM2_MOVING_GAUGE_FEATURE_REQUEST_SCHEMA,
            "checkpoint_schema": LFM2_MOVING_GAUGE_FEATURE_CHECKPOINT_SCHEMA,
            "receipt_schema": LFM2_MOVING_GAUGE_FEATURE_RECEIPT_SCHEMA,
            "anchor_schema": LFM2_MOVING_GAUGE_FEATURE_ANCHOR_SCHEMA,
            "shard_schema": LFM2_MOVING_GAUGE_FEATURE_SHARD_SCHEMA,
            "implementation_sha256": self.implementation_sha256,
        }
        return FoldPhaseKey(
            phase_name=LFM2_MOVING_GAUGE_FEATURE_PHASE,
            dependency_sha256s=dependencies,
            numeric_contract=numeric,
            abi=abi,
        )


class Lfm2MovingGaugeFeaturePhaseAuthority:
    """Runtime-only HMAC authority with feature-specific domains."""

    __slots__ = ("__secret", "key_id")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        self.key_id = _identifier(key_id, "feature phase HMAC key_id")
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise Lfm2MovingGaugeFeaturePhaseError(
                "Feature phase HMAC secret must contain at least 32 bytes."
            )
        self.__secret = bytes(secret)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("Feature phase HMAC authority cannot be serialized.")

    def _sign(self, domain: bytes, body: Mapping[str, Any]) -> str:
        return hmac.new(
            self.__secret,
            domain + canonical_json_bytes(dict(body)),
            hashlib.sha256,
        ).hexdigest()

    def sign_checkpoint(self, body: Mapping[str, Any]) -> str:
        return self._sign(_CHECKPOINT_DOMAIN, body)

    def verify_checkpoint(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return bool(
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_checkpoint(body), claimed)
        )

    def sign_anchor(self, body: Mapping[str, Any]) -> str:
        return self._sign(_ANCHOR_DOMAIN, body)

    def verify_anchor(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return bool(
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_anchor(body), claimed)
        )


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeFeatureExtractionSources:
    """Deferred cold sources; never instantiated on a completed phase hit."""

    boundary_phase: AuthenticatedLfm2MovingGaugeBoundaryPhase = field(
        repr=False, compare=False
    )
    child_cache: Any = field(repr=False, compare=False)
    child_session_factory: Callable[[], Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.boundary_phase, AuthenticatedLfm2MovingGaugeBoundaryPhase
        ):
            raise TypeError(
                "boundary_phase must be an authenticated moving-gauge boundary phase."
            )
        if not callable(self.child_session_factory):
            raise TypeError("child_session_factory must be callable.")


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeFeatureExtractionProgress:
    """Authenticated partial-prefix status for an interrupted feature import."""

    phase_key_sha256: str
    total_shards: int
    completed_shards: int
    checkpoint_chain_sha256: str
    complete: bool = False


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeWriterFeatureRows:
    """One variable-width writer feature over an authenticated boundary split."""

    field_name: str
    writer_role: str
    split: str
    rows: torch.Tensor = field(repr=False, compare=False)
    record_ids: tuple[str, ...]
    global_row_indices: tuple[int, ...]
    weights: torch.Tensor = field(repr=False, compare=False)
    source_dtype: str
    feature_width: int
    row_alignment_sha256: str
    split_identity_sha256: str


def _cache_completion_sha256(cache: Any) -> str:
    completion = getattr(cache, "_completion", None)
    if isinstance(completion, Mapping):
        return _digest(completion.get("completion_sha256"), "child cache completion")
    path = getattr(cache, "path", None)
    if path is None:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child source cache exposes no completion evidence."
        )
    return _digest(
        _read_json(Path(path) / "complete.json").get("completion_sha256"),
        "child cache completion",
    )


def _cache_manifest(cache: Any) -> dict[str, Any]:
    manifest = getattr(cache, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child source cache exposes no manifest."
        )
    result = _canonical(dict(manifest), "child source cache manifest")
    if not isinstance(result, dict):  # pragma: no cover - canonical guard
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child source cache manifest is invalid."
        )
    return result


def _writer_contract_body(
    layer_contract: Mapping[str, Any],
    bindings: Sequence[Lfm2MovingGaugeWriterFeatureBinding],
) -> dict[str, Any]:
    return {
        "schema": "neural-forge-lfm2-moving-gauge-writer-contract-v1",
        "layer_contract": _canonical(dict(layer_contract), "child layer contract"),
        "feature_bindings": [value.as_dict() for value in bindings],
    }


def _verify_child_cache_binding(
    request: Lfm2MovingGaugeFeatureImportRequest,
    cache: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _cache_manifest(cache)
    if (
        getattr(cache, "cache_key", None) != request.child_cache_key
        or manifest.get("manifest_sha256") != request.child_manifest_sha256
        or _cache_completion_sha256(cache) != request.child_completion_sha256
        or manifest.get("parent", {}).get("identity_sha256")
        != request.child_identity_sha256
        or manifest.get("parent", {}).get("state_dict_sha256")
        != request.child_state_dict_sha256
        or manifest.get("layer_topology", {}).get("layer_topology_sha256")
        != request.child_topology_sha256
        or manifest.get("corpus", {}).get("batch_count") != request.expected_batch_count
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child feature source cache differs from the immutable request."
        )
    topology = manifest.get("layer_topology", {}).get("layers")
    if (
        not isinstance(topology, list)
        or request.child_layer_index >= len(topology)
        or not isinstance(topology[request.child_layer_index], dict)
        or not isinstance(topology[request.child_layer_index].get("contract"), dict)
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child feature source topology is incomplete."
        )
    contract = topology[request.child_layer_index]["contract"]
    if canonical_sha256(contract) != request.child_layer_contract_sha256:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child L0 contract differs from the feature request."
        )
    derived = derive_lfm2_moving_gauge_writer_feature_bindings(
        contract,
        source_dtype_by_field={
            value.field_name: value.source_dtype for value in request.feature_bindings
        },
    )
    if tuple(value.as_dict() for value in derived) != tuple(
        value.as_dict() for value in request.feature_bindings
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature widths or writer weight topology differ from child L0."
        )
    return manifest, contract


def _boundary_row_index_from_rows(
    request: Lfm2MovingGaugeFeatureImportRequest,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema": LFM2_MOVING_GAUGE_BOUNDARY_ROW_INDEX_SCHEMA,
        "request_sha256": request.boundary_request_sha256,
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
        "basis_split_is_chart_authority": True,
        "bridge_split_is_provenance_only": True,
    }
    return {**body, "row_index_sha256": canonical_sha256(body)}


def _verify_boundary_binding(
    request: Lfm2MovingGaugeFeatureImportRequest,
    boundary: AuthenticatedLfm2MovingGaugeBoundaryPhase,
) -> tuple[dict[str, Any], ...]:
    try:
        authentic = boundary.verify_authentication()
        boundary_request = boundary.request
        rows = tuple(dict(row) for row in boundary.row_index["rows"])
        binding = boundary_request.layer_bindings[request.logical_layer_index]
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Completed moving-gauge boundary could not be reconstructed."
        ) from exc
    expected_row_index = _boundary_row_index_from_rows(request, rows)
    if (
        authentic is not True
        or boundary.entry.phase_key.phase_key_sha256
        != request.boundary_phase_key_sha256
        or boundary.entry.completion_sha256 != request.boundary_completion_sha256
        or boundary_request.request_sha256 != request.boundary_request_sha256
        or boundary.receipt_sha256 != request.boundary_receipt_sha256
        or boundary.row_index.get("row_index_sha256")
        != request.boundary_row_index_sha256
        or _canonical(
            _deep_thaw(boundary.row_index), "authenticated boundary row index"
        )
        != expected_row_index
        or boundary.row_alignment_sha256 != request.row_alignment_sha256
        or boundary.split_identity_sha256 != request.split_identity_sha256
        or boundary.receipt.get("tensor_bytes_authenticated_lazily_on_field_access")
        is not True
        or boundary_request.child_cache_key != request.child_cache_key
        or boundary_request.provenance_sha256s["child_manifest"]
        != request.child_manifest_sha256
        or boundary_request.provenance_sha256s["child_completion"]
        != request.child_completion_sha256
        or boundary_request.provenance_sha256s["child_identity"]
        != request.child_identity_sha256
        or boundary_request.provenance_sha256s["child_state_dict"]
        != request.child_state_dict_sha256
        or boundary_request.provenance_sha256s["child_topology"]
        != request.child_topology_sha256
        or boundary_request.provenance_sha256s["child_selective_session"]
        != request.child_selective_session_sha256
        or boundary_request.expected_batch_count != request.expected_batch_count
        or boundary_request.expected_aligned_row_count
        != request.expected_aligned_row_count
        or binding.child_layer_index != request.child_layer_index
        or binding.operator_kind != "short_conv"
        or len(rows) != request.expected_aligned_row_count
        or {row.get("basis_split") for row in rows} != _BASIS_SPLITS
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature request differs from its authenticated boundary row authority."
        )
    return rows


def _source_names(
    request: Lfm2MovingGaugeFeatureImportRequest,
    phases: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{phase}.layer.{request.child_layer_index:03d}.{binding.field_name}"
            for phase in phases
            for binding in request.feature_bindings
        )
    )


def _normalized_name(field_name: str) -> str:
    return f"layer.000.{field_name}"


def _selected_payload_body(
    session: Any,
    reference: Any,
    batch_index: int,
    tensor_names: Sequence[str],
    descriptors: Sequence[Mapping[str, Any]],
    logical_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
        "session_sha256": getattr(session, "evidence_sha256", None),
        "cache_key": getattr(session, "cache_key", None),
        "batch_index": batch_index,
        "input_sha256": getattr(reference, "input_sha256", None),
        "record_sha256": getattr(reference, "record_sha256", None),
        "ordered_tensor_names": list(tensor_names),
        "ordered_tensor_descriptors": [dict(value) for value in descriptors],
        "logical_bytes": logical_bytes,
    }


def _session_plan(
    request: Lfm2MovingGaugeFeatureImportRequest,
    session: Any,
    boundary_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    if (
        getattr(session, "cache_key", None) != request.child_cache_key
        or getattr(session, "evidence_sha256", None)
        != request.child_selective_session_sha256
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child selective session differs from the feature request."
        )
    references = tuple(getattr(session, "batch_references", ()))
    if len(references) != request.expected_batch_count:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child selective session has the wrong batch count."
        )
    binding_by_field = {value.field_name: value for value in request.feature_bindings}
    names_by_batch: list[tuple[str, ...]] = []
    phases_by_batch: list[tuple[str, ...]] = []
    for batch_index, reference in enumerate(references):
        if getattr(reference, "index", None) != batch_index:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Child reference {batch_index} has the wrong batch index."
            )
        rows = tuple(
            dict(row) for row in boundary_rows if row["record_ordinal"] == batch_index
        )
        if not rows:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Boundary has no rows for child batch {batch_index}."
            )
        child_inputs = {row["child_input_sha256"] for row in rows}
        if child_inputs != {getattr(reference, "input_sha256", None)}:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Child reference {batch_index} differs from boundary input evidence."
            )
        phases = tuple(dict.fromkeys(row["phase"] for row in rows))
        if tuple(sorted(phases, key=_phase_sort_key)) != phases:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Boundary phases for child batch {batch_index} are not canonical."
            )
        inventory = getattr(reference, "record", {}).get("tensors")
        if (
            not isinstance(inventory, Mapping)
            or tuple(getattr(reference, "ordered_tensor_names", ()))
            != tuple(sorted(inventory))
        ):
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Child reference {batch_index} has no sealed tensor inventory."
            )
        names = _source_names(request, phases)
        for phase in phases:
            expected_rows = sum(1 for row in rows if row["phase"] == phase)
            for field_name, binding in binding_by_field.items():
                name = f"{phase}.layer.{request.child_layer_index:03d}.{field_name}"
                descriptor = inventory.get(name)
                shape = (
                    descriptor.get("shape") if isinstance(descriptor, Mapping) else None
                )
                dtype = getattr(torch, binding.source_dtype, None)
                if not isinstance(dtype, torch.dtype):
                    raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                        f"Unsupported feature storage dtype {binding.source_dtype!r}."
                    )
                expected_size_bytes = (
                    expected_rows
                    * binding.feature_width
                    * torch.empty((), dtype=dtype).element_size()
                )
                if (
                    not isinstance(descriptor, Mapping)
                    or set(descriptor) != _TENSOR_DESCRIPTOR_FIELDS
                    or shape != [1, expected_rows, binding.feature_width]
                    or descriptor.get("dtype") != binding.source_dtype
                    or descriptor.get("numel") != expected_rows * binding.feature_width
                    or descriptor.get("size_bytes") != expected_size_bytes
                    or _SHA256.fullmatch(str(descriptor.get("sha256", ""))) is None
                ):
                    raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                        f"Child writer feature {name!r} changed width, dtype, or rows."
                    )
        names_by_batch.append(names)
        phases_by_batch.append(phases)
    return tuple(names_by_batch), tuple(phases_by_batch)


def _authenticate_selected_payload(
    session: Any,
    reference: Any,
    batch_index: int,
    tensor_names: Sequence[str],
    payload: Any,
) -> Mapping[str, torch.Tensor]:
    names = tuple(tensor_names)
    record = getattr(reference, "record", None)
    inventory = record.get("tensors") if isinstance(record, Mapping) else None
    tensors = getattr(payload, "tensors", None)
    if (
        not isinstance(inventory, Mapping)
        or not isinstance(tensors, Mapping)
        or set(tensors) != set(names)
        or getattr(payload, "batch_index", None) != batch_index
        or getattr(payload, "input_sha256", None)
        != getattr(reference, "input_sha256", None)
        or tuple(getattr(payload, "ordered_tensor_names", ())) != names
        or getattr(payload, "session_sha256", None)
        != getattr(session, "evidence_sha256", None)
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Selected feature payload {batch_index} is not session-bound."
        )
    descriptors: list[dict[str, Any]] = []
    logical_bytes = 0
    for name in names:
        descriptor = inventory.get(name)
        tensor = tensors.get(name)
        expected_descriptor = (
            dict(descriptor) if isinstance(descriptor, Mapping) else {}
        )
        actual_descriptor = (
            _circuit_trace_tensor_descriptor(torch, tensor)
            if isinstance(tensor, torch.Tensor)
            else None
        )
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != _TENSOR_DESCRIPTOR_FIELDS
            or not isinstance(tensor, torch.Tensor)
            or actual_descriptor != expected_descriptor
        ):
            mismatch_fields = sorted(
                key
                for key in set(expected_descriptor) | set(actual_descriptor or {})
                if expected_descriptor.get(key) != (actual_descriptor or {}).get(key)
            )
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Selected writer feature {name!r} differs from sealed bytes; "
                f"circuit-trace descriptor fields {mismatch_fields!r}; "
                f"expected={expected_descriptor!r}, "
                f"actual={actual_descriptor!r}."
            )
        descriptors.append(dict(descriptor))
        logical_bytes += int(descriptor["size_bytes"])
    selection_body = _selected_payload_body(
        session,
        reference,
        batch_index,
        names,
        descriptors,
        logical_bytes,
    )
    if getattr(payload, "logical_bytes", None) != logical_bytes or getattr(
        payload, "selection_sha256", None
    ) != canonical_sha256(selection_body):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Selected feature payload {batch_index} receipt is noncanonical."
        )
    return tensors


def _checkpoint_count(request: Lfm2MovingGaugeFeatureImportRequest) -> int:
    return math.ceil(request.expected_batch_count / request.shard_batch_count)


def _checkpoint_range(
    request: Lfm2MovingGaugeFeatureImportRequest,
    ordinal: int,
) -> tuple[int, int]:
    total = _checkpoint_count(request)
    if ordinal < 0 or ordinal >= total:
        raise Lfm2MovingGaugeFeaturePhaseError("Feature checkpoint is out of range.")
    start = ordinal * request.shard_batch_count
    return start, min(request.expected_batch_count, start + request.shard_batch_count)


def _checkpoint_path(root: Path, ordinal: int) -> Path:
    return root / "checkpoints" / f"checkpoint-{ordinal:04d}.json"


def _row_range_for_batches(
    rows: Sequence[Mapping[str, Any]],
    start: int,
    stop: int,
) -> tuple[dict[str, Any], ...]:
    selected = tuple(
        dict(row) for row in rows if start <= int(row["record_ordinal"]) < stop
    )
    if not selected:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature shard contains no boundary rows."
        )
    indices = tuple(int(row["global_row_index"]) for row in selected)
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature shard boundary rows are not contiguous."
        )
    return selected


def _extract_one_shard(
    request: Lfm2MovingGaugeFeatureImportRequest,
    session: Any,
    boundary_rows: Sequence[Mapping[str, Any]],
    names_by_batch: Sequence[Sequence[str]],
    phases_by_batch: Sequence[Sequence[str]],
    *,
    checkpoint_ordinal: int,
    batch_start: int,
    batch_stop: int,
    previous_hmac: str,
    work_root: Path,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
) -> dict[str, Any]:
    pieces: dict[str, list[torch.Tensor]] = {}
    source_records: list[dict[str, Any]] = []
    for batch_index in range(batch_start, batch_stop):
        reference = session.batch_references[batch_index]
        names = tuple(names_by_batch[batch_index])
        payload = session.load_selected_tensors(batch_index, names)
        try:
            tensors = _authenticate_selected_payload(
                session, reference, batch_index, names, payload
            )
            for phase in phases_by_batch[batch_index]:
                for binding in request.feature_bindings:
                    source_name = (
                        f"{phase}.layer.{request.child_layer_index:03d}."
                        f"{binding.field_name}"
                    )
                    tensor = tensors[source_name]
                    rows = (
                        tensor.detach()
                        .to(device="cpu")
                        .contiguous()
                        .reshape(int(tensor.shape[1]), int(tensor.shape[2]))
                    )
                    pieces.setdefault(_normalized_name(binding.field_name), []).append(
                        rows
                    )
            source_rows = tuple(
                row for row in boundary_rows if row["record_ordinal"] == batch_index
            )
            source_records.append(
                {
                    "batch_index": batch_index,
                    "input_sha256": str(reference.input_sha256),
                    "record_sha256": str(reference.record_sha256),
                    "alignment_sha256": str(source_rows[0]["alignment_sha256"]),
                    "selected_payload_sha256": str(payload.selection_sha256),
                }
            )
        finally:
            session.release_selected_tensors(payload)
    joined = {
        name: torch.cat(values, dim=0).contiguous()
        for name, values in sorted(pieces.items())
    }
    shard_rows = _row_range_for_batches(boundary_rows, batch_start, batch_stop)
    expected_rows = len(shard_rows)
    binding_by_name = {
        _normalized_name(value.field_name): value for value in request.feature_bindings
    }
    if set(joined) != set(binding_by_name) or any(
        tuple(value.shape) != (expected_rows, binding_by_name[name].feature_width)
        or str(value.dtype).removeprefix("torch.") != binding_by_name[name].source_dtype
        for name, value in joined.items()
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Writer-feature shard geometry or dtype changed during copy."
        )
    from safetensors.torch import save_file

    shard_root = work_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    stage = shard_root / f".stage-{uuid.uuid4().hex}.safetensors"
    save_file(
        joined,
        str(stage),
        metadata={
            "schema": LFM2_MOVING_GAUGE_FEATURE_SHARD_SCHEMA,
            "request_sha256": request.request_sha256,
            "checkpoint_ordinal": str(checkpoint_ordinal),
        },
    )
    file_sha = _file_sha256(stage)
    filename = f"child-feature-{checkpoint_ordinal:04d}-{file_sha[:16]}.safetensors"
    shard_path = shard_root / filename
    if shard_path.exists():
        if shard_path.is_symlink() or _file_sha256(shard_path) != file_sha:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "A content-named feature shard conflicts with staged bytes."
            )
        stage.unlink()
    else:
        os.replace(stage, shard_path)
    descriptors = {
        name: _tensor_descriptor(value) for name, value in sorted(joined.items())
    }
    body = {
        "schema": LFM2_MOVING_GAUGE_FEATURE_CHECKPOINT_SCHEMA,
        "request_sha256": request.request_sha256,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "checkpoint_hmac_key_id": authority.key_id,
        "checkpoint_ordinal": checkpoint_ordinal,
        "batch_start": batch_start,
        "batch_stop": batch_stop,
        "previous_checkpoint_hmac_sha256": previous_hmac,
        "source_cache_key": request.child_cache_key,
        "source_selective_session_sha256": request.child_selective_session_sha256,
        "source_records": source_records,
        "rows": [dict(row) for row in shard_rows],
        "row_start": int(shard_rows[0]["global_row_index"]),
        "row_stop": int(shard_rows[-1]["global_row_index"]) + 1,
        "row_index_slice_sha256": canonical_sha256(list(shard_rows)),
        "shard": {
            "filename": filename,
            "file_sha256": file_sha,
            "size_bytes": shard_path.stat().st_size,
            "tensor_descriptors": descriptors,
            "tensor_inventory_sha256": canonical_sha256(descriptors),
        },
        "source_storage_dtypes_preserved": True,
        "unselected_tensor_payloads_loaded": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    return {
        **sealed,
        "checkpoint_hmac_sha256": authority.sign_checkpoint(sealed),
    }


def _authenticate_checkpoint(
    value: Mapping[str, Any],
    *,
    request: Lfm2MovingGaugeFeatureImportRequest,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    expected_ordinal: int,
    expected_previous_hmac: str,
    shard_root: Path,
) -> dict[str, Any]:
    row = _canonical(dict(value), "feature checkpoint")
    if not isinstance(row, dict) or set(row) != _CHECKPOINT_FIELDS:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature checkpoint has a noncanonical field inventory."
        )
    claimed_hmac = row.pop("checkpoint_hmac_sha256", None)
    claimed_sha = row.pop("checkpoint_sha256", None)
    start, stop = _checkpoint_range(request, expected_ordinal)
    if (
        row.get("schema") != LFM2_MOVING_GAUGE_FEATURE_CHECKPOINT_SCHEMA
        or row.get("request_sha256") != request.request_sha256
        or row.get("phase_key_sha256") != request.phase_key.phase_key_sha256
        or row.get("checkpoint_hmac_key_id") != authority.key_id
        or row.get("checkpoint_ordinal") != expected_ordinal
        or row.get("batch_start") != start
        or row.get("batch_stop") != stop
        or row.get("previous_checkpoint_hmac_sha256") != expected_previous_hmac
        or row.get("source_cache_key") != request.child_cache_key
        or row.get("source_selective_session_sha256")
        != request.child_selective_session_sha256
        or row.get("source_storage_dtypes_preserved") is not True
        or row.get("unselected_tensor_payloads_loaded") is not False
        or row.get("execution") != _ZERO_EXECUTION
        or claimed_sha != canonical_sha256(row)
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} differs from its request."
        )
    sealed = {**row, "checkpoint_sha256": claimed_sha}
    if not authority.verify_checkpoint(sealed, claimed_hmac):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} HMAC does not verify."
        )
    source_records = row.get("source_records")
    if (
        not isinstance(source_records, list)
        or len(source_records) != stop - start
        or any(
            not isinstance(source, dict)
            or set(source) != _SOURCE_RECORD_FIELDS
            or source.get("batch_index") != batch_index
            or any(
                _SHA256.fullmatch(str(source.get(name, ""))) is None
                for name in (
                    "alignment_sha256",
                    "input_sha256",
                    "record_sha256",
                    "selected_payload_sha256",
                )
            )
            for batch_index, source in zip(
                range(start, stop), source_records, strict=True
            )
        )
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} source records are invalid."
        )
    rows = row.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} has no boundary rows."
        )
    for item in rows:
        if (
            not isinstance(item, dict)
            or set(item) != _ROW_FIELDS
            or item.get("record_ordinal") not in range(start, stop)
            or item.get("basis_split") not in _BASIS_SPLITS
            or not isinstance(item.get("bridge_split"), str)
            or not item["bridge_split"].strip()
            or not isinstance(item.get("record_id"), str)
            or not item["record_id"].strip()
            or any(
                _SHA256.fullmatch(str(item.get(name, ""))) is None
                for name in (
                    "alignment_sha256",
                    "child_input_sha256",
                    "parent_input_sha256",
                )
            )
            or isinstance(item.get("phase_row_index"), bool)
            or not isinstance(item.get("phase_row_index"), int)
            or item["phase_row_index"] < 0
        ):
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Feature checkpoint {expected_ordinal} boundary row is invalid."
            )
        _phase_sort_key(item.get("phase"))
    global_indices = tuple(item["global_row_index"] for item in rows)
    if (
        any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in global_indices
        )
        or global_indices
        != tuple(range(global_indices[0], global_indices[0] + len(global_indices)))
        or row.get("row_start") != global_indices[0]
        or row.get("row_stop") != global_indices[-1] + 1
        or row.get("row_index_slice_sha256") != canonical_sha256(rows)
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} row range is invalid."
        )
    for batch_index in range(start, stop):
        selected = [item for item in rows if item["record_ordinal"] == batch_index]
        if not selected:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Feature checkpoint {expected_ordinal} omits batch rows."
            )
        source = source_records[batch_index - start]
        if {item["alignment_sha256"] for item in selected} != {
            source["alignment_sha256"]
        } or {item["child_input_sha256"] for item in selected} != {
            source["input_sha256"]
        }:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Feature checkpoint {expected_ordinal} source record differs "
                "from its boundary row claims."
            )
        phases = tuple(dict.fromkeys(item["phase"] for item in selected))
        if tuple(sorted(phases, key=_phase_sort_key)) != phases:
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Feature checkpoint {expected_ordinal} phase order is invalid."
            )
        for phase in phases:
            local = tuple(
                item["phase_row_index"] for item in selected if item["phase"] == phase
            )
            if local != tuple(range(len(local))):
                raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                    f"Feature checkpoint {expected_ordinal} phase rows are invalid."
                )
    shard = row.get("shard")
    if not isinstance(shard, dict) or set(shard) != _SHARD_FIELDS:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature checkpoint shard has a noncanonical field inventory."
        )
    filename = shard.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature checkpoint shard filename is unsafe."
        )
    shard_path = shard_root / filename
    if (
        not shard_path.is_file()
        or shard_path.is_symlink()
        or _file_sha256(shard_path) != shard.get("file_sha256")
        or shard_path.stat().st_size != shard.get("size_bytes")
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} shard changed."
        )
    descriptors = shard.get("tensor_descriptors")
    expected_names = {
        _normalized_name(value.field_name) for value in request.feature_bindings
    }
    binding_by_name = {
        _normalized_name(value.field_name): value for value in request.feature_bindings
    }
    expected_rows = len(rows)
    if not isinstance(descriptors, dict) or set(descriptors) != expected_names:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} tensor inventory is incomplete."
        )
    if shard.get("tensor_inventory_sha256") != canonical_sha256(descriptors):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            f"Feature checkpoint {expected_ordinal} tensor inventory seal changed."
        )
    for name, descriptor in descriptors.items():
        binding = binding_by_name[name]
        dtype = getattr(torch, binding.source_dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Unsupported feature storage dtype {binding.source_dtype!r}."
            )
        element_size = torch.empty((), dtype=dtype).element_size()
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != _TENSOR_DESCRIPTOR_FIELDS
            or descriptor.get("dtype") != binding.source_dtype
            or descriptor.get("shape") != [expected_rows, binding.feature_width]
            or descriptor.get("numel") != expected_rows * binding.feature_width
            or descriptor.get("size_bytes")
            != expected_rows * binding.feature_width * element_size
            or _SHA256.fullmatch(str(descriptor.get("sha256", ""))) is None
        ):
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                f"Feature checkpoint {expected_ordinal} descriptor {name!r} is invalid."
            )
    return {**sealed, "checkpoint_hmac_sha256": claimed_hmac}


def _load_checkpoint_prefix(
    root: Path,
    *,
    request: Lfm2MovingGaugeFeatureImportRequest,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    shard_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    checkpoint_root = root / "checkpoints"
    if not checkpoint_root.exists():
        return ()
    paths = tuple(sorted(checkpoint_root.glob("checkpoint-*.json")))
    expected_names = tuple(
        f"checkpoint-{index:04d}.json" for index in range(len(paths))
    )
    if tuple(path.name for path in paths) != expected_names:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature checkpoints must form one canonical prefix."
        )
    if len(paths) > _checkpoint_count(request):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature checkpoint prefix exceeds the request."
        )
    previous = _ZERO_SHA256
    checkpoints: list[dict[str, Any]] = []
    selected_shards = root / "shards" if shard_root is None else shard_root
    for index, path in enumerate(paths):
        checkpoint = _authenticate_checkpoint(
            _read_json(path),
            request=request,
            authority=authority,
            expected_ordinal=index,
            expected_previous_hmac=previous,
            shard_root=selected_shards,
        )
        checkpoints.append(checkpoint)
        previous = checkpoint["checkpoint_hmac_sha256"]
    return tuple(checkpoints)


def _verify_source_session_complete(session: Any) -> None:
    verify = getattr(session, "verify", None)
    if not callable(verify) or verify() is not session:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Child selective source session did not verify before checkpoint admission."
        )


def _discard_unverified_session_drafts(
    work_root: Path,
    drafts: Sequence[Mapping[str, Any]],
) -> None:
    for draft in drafts:
        filename = draft.get("shard", {}).get("filename")
        if isinstance(filename, str) and Path(filename).name == filename:
            (work_root / "shards" / filename).unlink(missing_ok=True)


def _verify_resumed_checkpoint_sources(
    request: Lfm2MovingGaugeFeatureImportRequest,
    session: Any,
    boundary_rows: Sequence[Mapping[str, Any]],
    names_by_batch: Sequence[Sequence[str]],
    checkpoints: Sequence[Mapping[str, Any]],
) -> None:
    references = tuple(getattr(session, "batch_references", ()))
    for checkpoint in checkpoints:
        for source in checkpoint["source_records"]:
            batch_index = int(source["batch_index"])
            reference = references[batch_index]
            if (
                getattr(reference, "index", None) != batch_index
                or getattr(reference, "input_sha256", None)
                != source["input_sha256"]
                or _SHA256.fullmatch(
                    str(getattr(reference, "record_sha256", ""))
                )
                is None
            ):
                raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                    "Resumed feature checkpoint differs from its live selective "
                    "source reference."
                )
            record = getattr(reference, "record", None)
            inventory = record.get("tensors") if isinstance(record, Mapping) else None
            names = tuple(names_by_batch[batch_index])
            if not isinstance(inventory, Mapping) or any(
                name not in inventory for name in names
            ):
                raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                    "Resumed feature checkpoint has no live source descriptors."
                )
            descriptors = tuple(dict(inventory[name]) for name in names)
            logical_bytes = sum(int(value["size_bytes"]) for value in descriptors)
            selection_sha256 = canonical_sha256(
                _selected_payload_body(
                    session,
                    reference,
                    batch_index,
                    names,
                    descriptors,
                    logical_bytes,
                )
            )
            source_rows = tuple(
                row
                for row in boundary_rows
                if row["record_ordinal"] == batch_index
            )
            expected = {
                "batch_index": batch_index,
                "input_sha256": str(reference.input_sha256),
                "record_sha256": str(reference.record_sha256),
                "alignment_sha256": str(source_rows[0]["alignment_sha256"]),
                "selected_payload_sha256": selection_sha256,
            }
            if dict(source) != expected:
                raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                    "Resumed feature checkpoint differs from its live selective "
                    "source record or tensor descriptors."
                )


def _extract_missing_prefix(
    request: Lfm2MovingGaugeFeatureImportRequest,
    sources: Lfm2MovingGaugeFeatureExtractionSources,
    work_root: Path,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    checkpoints: tuple[dict[str, Any], ...],
    *,
    maximum_new_shards: int | None,
) -> tuple[dict[str, Any], ...]:
    boundary_rows = _verify_boundary_binding(request, sources.boundary_phase)
    _verify_child_cache_binding(request, sources.child_cache)
    completed = list(checkpoints)
    total = _checkpoint_count(request)
    if len(completed) >= total:
        return tuple(completed)
    drafts: list[dict[str, Any]] = []
    try:
        with sources.child_session_factory() as session:
            names_by_batch, phases_by_batch = _session_plan(
                request, session, boundary_rows
            )
            if completed:
                _verify_source_session_complete(session)
                _verify_resumed_checkpoint_sources(
                    request,
                    session,
                    boundary_rows,
                    names_by_batch,
                    completed,
                )
            while len(completed) + len(drafts) < total:
                if maximum_new_shards is not None and len(drafts) >= maximum_new_shards:
                    break
                ordinal = len(completed) + len(drafts)
                start, stop = _checkpoint_range(request, ordinal)
                previous = (
                    _ZERO_SHA256
                    if not completed and not drafts
                    else (
                        drafts[-1]["checkpoint_hmac_sha256"]
                        if drafts
                        else completed[-1]["checkpoint_hmac_sha256"]
                    )
                )
                drafts.append(
                    _extract_one_shard(
                        request,
                        session,
                        boundary_rows,
                        names_by_batch,
                        phases_by_batch,
                        checkpoint_ordinal=ordinal,
                        batch_start=start,
                        batch_stop=stop,
                        previous_hmac=previous,
                        work_root=work_root,
                        authority=authority,
                    )
                )
            _verify_source_session_complete(session)
    except BaseException:
        _discard_unverified_session_drafts(work_root, drafts)
        raise
    verified_drafts: list[dict[str, Any]] = []
    previous = (
        _ZERO_SHA256 if not completed else completed[-1]["checkpoint_hmac_sha256"]
    )
    try:
        for offset, checkpoint in enumerate(drafts):
            verified = _authenticate_checkpoint(
                checkpoint,
                request=request,
                authority=authority,
                expected_ordinal=len(completed) + offset,
                expected_previous_hmac=previous,
                shard_root=work_root / "shards",
            )
            verified_drafts.append(verified)
            previous = verified["checkpoint_hmac_sha256"]
    except BaseException:
        _discard_unverified_session_drafts(work_root, drafts)
        raise
    for checkpoint in verified_drafts:
        _write_json(
            _checkpoint_path(work_root, int(checkpoint["checkpoint_ordinal"])),
            checkpoint,
        )
        completed.append(checkpoint)
    return tuple(completed)


def _checkpoint_chain_sha256(checkpoints: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "checkpoint_ordinal": value["checkpoint_ordinal"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "checkpoint_hmac_sha256": value["checkpoint_hmac_sha256"],
            }
            for value in checkpoints
        ]
    )


def _complete_phase_documents(
    request: Lfm2MovingGaugeFeatureImportRequest,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(checkpoints) != _checkpoint_count(request):
        raise Lfm2MovingGaugeFeaturePhaseError(
            "Cannot complete an incomplete writer-feature checkpoint chain."
        )
    rows = [row for checkpoint in checkpoints for row in checkpoint["rows"]]
    if len(rows) != request.expected_aligned_row_count or tuple(
        row["global_row_index"] for row in rows
    ) != tuple(range(request.expected_aligned_row_count)):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Completed feature checkpoints do not reproduce the boundary row count."
        )
    row_index = _boundary_row_index_from_rows(request, rows)
    if row_index["row_index_sha256"] != request.boundary_row_index_sha256:
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Completed feature checkpoints do not reproduce the exact boundary row index."
        )
    summaries = [
        {
            "checkpoint_ordinal": value["checkpoint_ordinal"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "checkpoint_hmac_sha256": value["checkpoint_hmac_sha256"],
            "batch_start": value["batch_start"],
            "batch_stop": value["batch_stop"],
            "row_start": value["row_start"],
            "row_stop": value["row_stop"],
            "shard": value["shard"],
        }
        for value in checkpoints
    ]
    source_evidence = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-moving-gauge-writer-feature-source-v1",
            "child_cache_key": request.child_cache_key,
            "child_manifest_sha256": request.child_manifest_sha256,
            "child_completion_sha256": request.child_completion_sha256,
            "child_selective_session_sha256": request.child_selective_session_sha256,
            "ordered_checkpoint_hmac_sha256s": [
                value["checkpoint_hmac_sha256"] for value in checkpoints
            ],
            "ordered_shard_file_sha256s": [
                value["shard"]["file_sha256"] for value in checkpoints
            ],
            "execution": dict(_ZERO_EXECUTION),
        }
    )
    receipt_body = {
        "schema": LFM2_MOVING_GAUGE_FEATURE_RECEIPT_SCHEMA,
        "request": request.as_dict(),
        "phase_key": request.phase_key.as_dict(),
        "checkpoint_hmac_key_id": authority.key_id,
        "checkpoint_chain_sha256": _checkpoint_chain_sha256(checkpoints),
        "ordered_checkpoints": summaries,
        "boundary_phase_key_sha256": request.boundary_phase_key_sha256,
        "boundary_completion_sha256": request.boundary_completion_sha256,
        "boundary_receipt_sha256": request.boundary_receipt_sha256,
        "boundary_row_index_sha256": request.boundary_row_index_sha256,
        "row_alignment_sha256": request.row_alignment_sha256,
        "split_identity_sha256": request.split_identity_sha256,
        "child_source_evidence_sha256": source_evidence,
        "feature_bindings": [value.as_dict() for value in request.feature_bindings],
        "source_storage_dtypes_preserved": True,
        "variable_width_features_authenticated_from_writer_topology": True,
        "boundary_row_index_reused_exactly": True,
        "unselected_tensor_payloads_loaded": False,
        "tensor_bytes_authenticated_lazily_on_field_access": True,
        "execution": dict(_ZERO_EXECUTION),
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    anchor_body = {
        "schema": LFM2_MOVING_GAUGE_FEATURE_ANCHOR_SCHEMA,
        "algorithm": "hmac-sha256",
        "key_id": authority.key_id,
        "phase_name": LFM2_MOVING_GAUGE_FEATURE_PHASE,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "boundary_row_index_sha256": request.boundary_row_index_sha256,
        "checkpoint_chain_sha256": receipt["checkpoint_chain_sha256"],
        "ordered_checkpoint_hmac_sha256s": [
            value["checkpoint_hmac_sha256"] for value in checkpoints
        ],
        "ordered_shard_file_sha256s": [
            value["shard"]["file_sha256"] for value in checkpoints
        ],
        "child_source_evidence_sha256": source_evidence,
    }
    anchor = {**anchor_body, "hmac_sha256": authority.sign_anchor(anchor_body)}
    result = {
        "receipt_sha256": receipt["receipt_sha256"],
        "boundary_row_index_sha256": request.boundary_row_index_sha256,
        "row_alignment_sha256": request.row_alignment_sha256,
        "split_identity_sha256": request.split_identity_sha256,
        "checkpoint_chain_sha256": receipt["checkpoint_chain_sha256"],
        "source_anchor_hmac_sha256": anchor["hmac_sha256"],
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
    }
    return row_index, receipt, anchor, result


def _publish_complete_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2MovingGaugeFeatureImportRequest,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    work_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    events: FoldPhaseEventLog | None,
) -> VerifiedFoldPhaseCacheEntry:
    row_index, receipt, anchor, result = _complete_phase_documents(
        request, authority, checkpoints
    )
    try:
        transaction = cache.begin(request.phase_key, events=events)
    except FoldPhaseCacheAlreadyCompleteError:
        existing = cache.lookup(request.phase_key, events=events)
        if existing is None:  # pragma: no cover - concurrent deletion defense
            raise FoldPhaseCacheIntegrityError(
                "Concurrent feature phase publication disappeared."
            )
        return existing
    with transaction:
        payload = transaction.payload_path
        (payload / "checkpoints").mkdir()
        (payload / "shards").mkdir()
        _write_json(payload / "request.json", request.as_dict())
        _write_json(payload / "row-index.json", row_index)
        _write_json(payload / "receipt.json", receipt)
        _write_json(payload / "source-anchor.json", anchor)
        for checkpoint in checkpoints:
            ordinal = int(checkpoint["checkpoint_ordinal"])
            shutil.copyfile(
                _checkpoint_path(work_root, ordinal),
                payload / "checkpoints" / f"checkpoint-{ordinal:04d}.json",
            )
            filename = checkpoint["shard"]["filename"]
            shutil.copyfile(
                work_root / "shards" / filename,
                payload / "shards" / filename,
            )
        staged_checkpoints = _load_checkpoint_prefix(
            payload,
            request=request,
            authority=authority,
            shard_root=payload / "shards",
        )
        staged_row_index, staged_receipt, staged_anchor, staged_result = (
            _complete_phase_documents(request, authority, staged_checkpoints)
        )
        if (
            staged_checkpoints != tuple(checkpoints)
            or _read_json(payload / "request.json") != request.as_dict()
            or _read_json(payload / "row-index.json") != staged_row_index
            or _read_json(payload / "receipt.json") != staged_receipt
            or _read_json(payload / "source-anchor.json") != staged_anchor
            or staged_row_index != row_index
            or staged_receipt != receipt
            or staged_anchor != anchor
            or staged_result != result
        ):
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "Staged writer-feature phase differs from its authenticated "
                "checkpoint reconstruction."
            )
        return transaction.complete(result=staged_result)


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2MovingGaugeFeaturePhase:
    """Completed child writer-feature phase with lazy byte authentication."""

    entry: VerifiedFoldPhaseCacheEntry = field(repr=False, compare=False)
    request: Lfm2MovingGaugeFeatureImportRequest
    receipt: Mapping[str, Any]
    row_index: Mapping[str, Any]
    source_anchor: Mapping[str, Any]
    authority: Lfm2MovingGaugeFeaturePhaseAuthority = field(repr=False, compare=False)

    @property
    def receipt_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])

    @property
    def row_alignment_sha256(self) -> str:
        return str(self.receipt["row_alignment_sha256"])

    @property
    def split_identity_sha256(self) -> str:
        return str(self.receipt["split_identity_sha256"])

    def verify_authentication(self) -> bool:
        try:
            loaded = load_existing_lfm2_moving_gauge_feature_phase(
                DynamicFoldPhaseCache(self.entry.path.parents[1]),
                self.request,
                authority=self.authority,
            )
            return bool(
                loaded.entry.path == self.entry.path
                and loaded.entry.completion_sha256 == self.entry.completion_sha256
                and dict(loaded.receipt) == dict(self.receipt)
                and dict(loaded.row_index) == dict(self.row_index)
                and dict(loaded.source_anchor) == dict(self.source_anchor)
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    def _rows_for_split(self, split: str) -> tuple[dict[str, Any], ...]:
        if split not in {"fit", "heldout", "all"}:
            raise Lfm2MovingGaugeFeaturePhaseError(
                "Feature split must be fit, heldout, or all."
            )
        rows = tuple(dict(row) for row in self.row_index["rows"])
        return (
            rows
            if split == "all"
            else tuple(row for row in rows if row["basis_split"] == split)
        )

    def _load_field(
        self,
        binding: Lfm2MovingGaugeWriterFeatureBinding,
        rows: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        key = _normalized_name(binding.field_name)
        selected = {int(row["global_row_index"]) for row in rows}
        pieces: list[torch.Tensor] = []
        from safetensors import safe_open

        for summary in self.receipt["ordered_checkpoints"]:
            start = int(summary["row_start"])
            stop = int(summary["row_stop"])
            local = [index - start for index in range(start, stop) if index in selected]
            if not local:
                continue
            descriptor = summary["shard"]["tensor_descriptors"].get(key)
            if not isinstance(descriptor, Mapping):
                raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                    f"Writer feature {key!r} is absent from a selected shard."
                )
            path = self.entry.payload_path / "shards" / summary["shard"]["filename"]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if (handle.metadata() or {}) != {
                    "schema": LFM2_MOVING_GAUGE_FEATURE_SHARD_SCHEMA,
                    "request_sha256": self.request.request_sha256,
                    "checkpoint_ordinal": str(summary["checkpoint_ordinal"]),
                }:
                    raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                        "Writer-feature shard metadata changed."
                    )
                tensor = handle.get_tensor(key).detach().to(device="cpu").contiguous()
            if _tensor_descriptor(tensor) != _deep_thaw(descriptor):
                raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                    f"Writer-feature tensor {key!r} changed."
                )
            pieces.append(
                tensor.index_select(
                    0, torch.tensor(local, dtype=torch.long)
                ).contiguous()
            )
        if not pieces:
            raise Lfm2MovingGaugeFeaturePhaseError(
                f"Writer-feature selection for {binding.field_name!r} is empty."
            )
        result = torch.cat(pieces, dim=0).contiguous()
        if tuple(result.shape) != (len(rows), binding.feature_width):
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "Writer-feature selection changed its authenticated geometry."
            )
        return result

    def feature_rows(
        self,
        field_name: str,
        *,
        split: str,
    ) -> Lfm2MovingGaugeWriterFeatureRows:
        try:
            binding = next(
                value
                for value in self.request.feature_bindings
                if value.field_name == field_name
            )
        except StopIteration as exc:
            raise Lfm2MovingGaugeFeaturePhaseError(
                f"Writer feature {field_name!r} was not authenticated."
            ) from exc
        rows = self._rows_for_split(split)
        record_ids = tuple(str(row["record_id"]) for row in rows)
        return Lfm2MovingGaugeWriterFeatureRows(
            field_name=field_name,
            writer_role=binding.writer_role,
            split=split,
            rows=self._load_field(binding, rows),
            record_ids=record_ids,
            global_row_indices=tuple(int(row["global_row_index"]) for row in rows),
            weights=equal_record_mass_weights(record_ids),
            source_dtype=binding.source_dtype,
            feature_width=binding.feature_width,
            row_alignment_sha256=self.row_alignment_sha256,
            split_identity_sha256=self.split_identity_sha256,
        )


def _authenticate_complete_entry(
    entry: VerifiedFoldPhaseCacheEntry,
    request: Lfm2MovingGaugeFeatureImportRequest,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
) -> AuthenticatedLfm2MovingGaugeFeaturePhase:
    if not request.verify_authentication():
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature import request authentication failed."
        )
    payload = entry.payload_path
    if _read_json(payload / "request.json") != request.as_dict():
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Completed feature request differs from its phase key."
        )
    row_index = _read_json(payload / "row-index.json")
    receipt = _read_json(payload / "receipt.json")
    anchor = _read_json(payload / "source-anchor.json")
    checkpoints = _load_checkpoint_prefix(
        payload,
        request=request,
        authority=authority,
        shard_root=payload / "shards",
    )
    expected_row, expected_receipt, expected_anchor, expected_result = (
        _complete_phase_documents(request, authority, checkpoints)
    )
    if (
        row_index != expected_row
        or receipt != expected_receipt
        or anchor != expected_anchor
        or _canonical(dict(entry.result), "feature phase result") != expected_result
        or entry.phase_key.as_dict() != request.phase_key.as_dict()
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Completed writer-feature documents differ from exact reconstruction."
        )
    return AuthenticatedLfm2MovingGaugeFeaturePhase(
        entry=entry,
        request=request,
        receipt=_deep_freeze(receipt),
        row_index=_deep_freeze(row_index),
        source_anchor=_deep_freeze(anchor),
        authority=authority,
    )


def load_existing_lfm2_moving_gauge_feature_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2MovingGaugeFeatureImportRequest,
    *,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    events: FoldPhaseEventLog | None = None,
) -> AuthenticatedLfm2MovingGaugeFeaturePhase:
    """Load one complete HMAC-authenticated writer-feature phase."""

    if not isinstance(cache, DynamicFoldPhaseCache):
        raise TypeError("cache must be DynamicFoldPhaseCache.")
    if not isinstance(request, Lfm2MovingGaugeFeatureImportRequest):
        raise TypeError("request must be Lfm2MovingGaugeFeatureImportRequest.")
    if not request.verify_authentication():
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature import request authentication failed."
        )
    if (
        not isinstance(authority, Lfm2MovingGaugeFeaturePhaseAuthority)
        or authority.key_id != request.checkpoint_hmac_key_id
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature phase authority differs from the request."
        )
    entry = cache.lookup(request.phase_key, events=events)
    if entry is None:
        raise Lfm2MovingGaugeFeaturePhaseError(
            "Requested writer-feature phase is not complete."
        )
    return _authenticate_complete_entry(entry, request, authority)


def load_or_extract_lfm2_moving_gauge_feature_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2MovingGaugeFeatureImportRequest,
    *,
    authority: Lfm2MovingGaugeFeaturePhaseAuthority,
    checkpoint_root: str | os.PathLike[str],
    source_provider: Callable[[], Lfm2MovingGaugeFeatureExtractionSources]
    | None = None,
    maximum_new_shards: int | None = None,
    events: FoldPhaseEventLog | None = None,
) -> (
    AuthenticatedLfm2MovingGaugeFeaturePhase | Lfm2MovingGaugeFeatureExtractionProgress
):
    """Load a hit or resume a zero-forward child-only writer-feature import."""

    if not isinstance(cache, DynamicFoldPhaseCache):
        raise TypeError("cache must be DynamicFoldPhaseCache.")
    if not isinstance(request, Lfm2MovingGaugeFeatureImportRequest):
        raise TypeError("request must be Lfm2MovingGaugeFeatureImportRequest.")
    if not request.verify_authentication():
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature import request authentication failed."
        )
    if (
        not isinstance(authority, Lfm2MovingGaugeFeaturePhaseAuthority)
        or authority.key_id != request.checkpoint_hmac_key_id
    ):
        raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
            "Feature phase authority differs from the request."
        )
    if maximum_new_shards is not None:
        _positive_int(maximum_new_shards, "maximum_new_shards")
    hit = cache.lookup(request.phase_key, events=events)
    if hit is not None:
        return _authenticate_complete_entry(hit, request, authority)
    root = (
        Path(checkpoint_root).expanduser().absolute()
        / request.phase_key.phase_key_sha256
    )
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request.json"
    if request_path.exists():
        if _read_json(request_path) != request.as_dict():
            raise Lfm2MovingGaugeFeaturePhaseAuthenticationError(
                "Feature checkpoint root belongs to another request."
            )
    else:
        _write_json(request_path, request.as_dict())
    checkpoints = _load_checkpoint_prefix(root, request=request, authority=authority)
    total = _checkpoint_count(request)
    if len(checkpoints) < total:
        if source_provider is None:
            raise Lfm2MovingGaugeFeaturePhaseError(
                "Incomplete writer-feature import requires a deferred source provider."
            )
        sources = source_provider()
        if not isinstance(sources, Lfm2MovingGaugeFeatureExtractionSources):
            raise TypeError(
                "source_provider must return Lfm2MovingGaugeFeatureExtractionSources."
            )
        checkpoints = _extract_missing_prefix(
            request,
            sources,
            root,
            authority,
            checkpoints,
            maximum_new_shards=maximum_new_shards,
        )
    if len(checkpoints) < total:
        return Lfm2MovingGaugeFeatureExtractionProgress(
            phase_key_sha256=request.phase_key.phase_key_sha256,
            total_shards=total,
            completed_shards=len(checkpoints),
            checkpoint_chain_sha256=_checkpoint_chain_sha256(checkpoints),
        )
    entry = _publish_complete_phase(
        cache,
        request,
        authority,
        root,
        checkpoints,
        events=events,
    )
    return _authenticate_complete_entry(entry, request, authority)


__all__ = [
    "AuthenticatedLfm2MovingGaugeFeaturePhase",
    "LFM2_MOVING_GAUGE_FEATURE_ANCHOR_SCHEMA",
    "LFM2_MOVING_GAUGE_FEATURE_CHECKPOINT_SCHEMA",
    "LFM2_MOVING_GAUGE_FEATURE_PHASE",
    "LFM2_MOVING_GAUGE_FEATURE_RECEIPT_SCHEMA",
    "LFM2_MOVING_GAUGE_FEATURE_REQUEST_SCHEMA",
    "Lfm2MovingGaugeFeatureExtractionProgress",
    "Lfm2MovingGaugeFeatureExtractionSources",
    "Lfm2MovingGaugeFeatureImportRequest",
    "Lfm2MovingGaugeFeaturePhaseAuthenticationError",
    "Lfm2MovingGaugeFeaturePhaseAuthority",
    "Lfm2MovingGaugeFeaturePhaseError",
    "Lfm2MovingGaugeWriterFeatureBinding",
    "Lfm2MovingGaugeWriterFeatureRows",
    "derive_lfm2_moving_gauge_writer_feature_bindings",
    "lfm2_moving_gauge_feature_implementation_sha256",
    "load_existing_lfm2_moving_gauge_feature_phase",
    "load_or_extract_lfm2_moving_gauge_feature_phase",
]
