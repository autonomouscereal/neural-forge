"""Authenticated zero-forward boundary import for LFM2 moving-gauge folding.

The exact-4x child already has a complete immutable child-reload trace cache.
This module imports only the residual endpoints needed to fit moving-gauge
charts, plus explicitly requested physical writer outputs.  It never receives
a model object and never executes a model forward.  Parent and child cache
inputs are paired by an authenticated tokenizer-bridge adapter rather than by
numeric token-ID equality.

Cold imports are split into source-dtype safetensor shards.  Every shard has a
domain-separated HMAC checkpoint in one canonical chain, so a killed import can
resume without accepting orphan files.  A completed import is published through
``DynamicFoldPhaseCache`` and receives a second domain-separated source anchor.
On a phase-cache hit, no source cache, adapter, or selective session is opened.
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
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_counted_child_state import (
    CountedChildStatePlan,
    counted_child_state_plan_from_dict,
)
from .dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseCacheAlreadyCompleteError,
    FoldPhaseCacheIntegrityError,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from .hf_lfm2_exact4x_causal_diagnostic import equal_record_mass_weights
from .hf_lfm2_exact4x_causal_diagnostic import (
    authenticate_v3_first_divergence_ledger,
)
from .hf_lfm2_exact4x_first_divergence import (
    authenticate_lfm2_exact4x_first_divergence_request,
    authoritative_split_by_batch,
)
from .hf_lfm2_balanced_basis_phase_cache import (
    LFM2_BALANCED_BASIS_PHASE,
    LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
    LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
    AuthenticatedLfm2BalancedBasisPhase,
    PhaseAnchorAuthority,
)
from .hf_lfm2_circuit_trace_cache import SELECTIVE_BATCH_PAYLOAD_SCHEMA
from .hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBoundaryEvidence,
    fingerprint_lfm2_moving_gauge_tensor,
)


LFM2_MOVING_GAUGE_BOUNDARY_PHASE = "lfm2-moving-gauge-boundary-import-v1"
LFM2_MOVING_GAUGE_BOUNDARY_REQUEST_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-boundary-import-request-v1"
)
LFM2_MOVING_GAUGE_BOUNDARY_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-boundary-shard-checkpoint-v1"
)
LFM2_MOVING_GAUGE_BOUNDARY_ROW_INDEX_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-boundary-row-index-v1"
)
LFM2_MOVING_GAUGE_BOUNDARY_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-boundary-phase-receipt-v1"
)
LFM2_MOVING_GAUGE_BOUNDARY_ANCHOR_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-boundary-source-anchor-v1"
)
LFM2_MOVING_GAUGE_BOUNDARY_SHARD_SCHEMA = (
    "neural-forge-lfm2-moving-gauge-boundary-shard-v1"
)

_CHECKPOINT_DOMAIN = b"neural-forge-lfm2-moving-gauge-boundary-shard-checkpoint-v1\0"
_ANCHOR_DOMAIN = b"neural-forge-lfm2-moving-gauge-boundary-source-anchor-v1\0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_PHASE = re.compile(r"(?:prefill|decode\.(\d{3}))\Z")
_ZERO_SHA256 = "0" * 64
_SIDES = ("parent", "child")
_BASIS_SPLITS = frozenset({"fit", "heldout"})
_RESIDUAL_FIELDS = (
    "residual.post_layer",
    "residual.post_operator",
    "residual.pre",
)
_FOCUSED_WRITER_FIELDS = ("mlp.down", "operator.output")
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
        "side",
        "side_shard_ordinal",
        "source_cache_key",
        "source_records",
        "source_selective_session_sha256",
        "source_storage_dtype_preserved",
        "unselected_tensor_payloads_loaded",
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
_BALANCED_BASIS_RECEIPT_FIELDS = frozenset(
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
_BALANCED_BASIS_ANCHOR_BODY_FIELDS = frozenset(
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
_REQUIRED_PROVENANCE_KEYS = frozenset(
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


def fingerprint_lfm2_moving_gauge_basis_split(
    basis_split_by_batch: Mapping[int, str],
    *,
    basis_phase_key_sha256: str,
    basis_completion_sha256: str,
    basis_map_sha256: str,
) -> str:
    """Bind one authoritative balanced-basis batch split to its source phase."""

    splits = dict(basis_split_by_batch)
    if (
        not splits
        or set(splits) != set(range(len(splits)))
        or set(splits.values()) != _BASIS_SPLITS
    ):
        raise Lfm2MovingGaugeBoundaryPhaseError(
            "Basis split must cover every ordinal with fit and heldout."
        )
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-moving-gauge-basis-split-v1",
            "basis_phase_key_sha256": _digest(
                basis_phase_key_sha256,
                "basis_phase_key_sha256",
            ),
            "basis_completion_sha256": _digest(
                basis_completion_sha256,
                "basis_completion_sha256",
            ),
            "basis_map_sha256": _digest(
                basis_map_sha256,
                "basis_map_sha256",
            ),
            "ordered_splits": [splits[index] for index in range(len(splits))],
        }
    )


class Lfm2MovingGaugeBoundaryPhaseError(RuntimeError):
    """The boundary import request, source evidence, or phase is invalid."""


class Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
    Lfm2MovingGaugeBoundaryPhaseError
):
    """A digest, HMAC, tensor, or source binding failed authentication."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise Lfm2MovingGaugeBoundaryPhaseError(
            f"{name} must be a nonempty bounded stable identifier."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2MovingGaugeBoundaryPhaseError(f"{name} must be positive.")
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    return value


def _canonical(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(_deep_thaw(value)).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2MovingGaugeBoundaryPhaseError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Cannot hash boundary phase file {path}."
        ) from exc
    return digest.hexdigest()


def lfm2_moving_gauge_boundary_implementation_sha256() -> str:
    """Fingerprint the exact source implementation that imports the phase."""

    return _file_sha256(Path(__file__).resolve())


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary metadata cannot be a symlink: {path}."
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Cannot parse boundary metadata {path}."
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary metadata is not canonical: {path}."
        )
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    stage = path.parent / f".{path.name}.stage-{uuid.uuid4().hex}"
    try:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        with stage.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.replace(stage, path)
    except FileExistsError as exc:
        raise Lfm2MovingGaugeBoundaryPhaseError(
            f"Immutable boundary metadata already exists: {path}."
        ) from exc
    finally:
        stage.unlink(missing_ok=True)


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": fingerprint_lfm2_moving_gauge_tensor(tensor),
    }


def _phase_sort_key(value: str) -> tuple[int, int]:
    if value == "prefill":
        return (0, 0)
    match = _PHASE.fullmatch(value)
    if match is None or match.group(1) is None:
        raise Lfm2MovingGaugeBoundaryPhaseError(
            f"Unsupported source trace phase {value!r}."
        )
    return (1, int(match.group(1)))


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryLayerBinding:
    """One explicit logical child layer to physical parent/child layer pair."""

    layer_index: int
    parent_layer_index: int
    child_layer_index: int
    operator_kind: str

    def __post_init__(self) -> None:
        for name in ("layer_index", "parent_layer_index", "child_layer_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Lfm2MovingGaugeBoundaryPhaseError(
                    f"{name} must be a nonnegative integer."
                )
        object.__setattr__(
            self,
            "operator_kind",
            _identifier(self.operator_kind, "operator_kind"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "parent_layer_index": self.parent_layer_index,
            "child_layer_index": self.child_layer_index,
            "operator_kind": self.operator_kind,
        }


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryImportRequest:
    """Immutable identity of one compact parent/child boundary import."""

    parent_cache_key: str
    child_cache_key: str
    expected_batch_count: int
    expected_aligned_row_count: int
    parent_width: int
    child_width: int
    parent_parameter_count: int
    child_parameter_count: int
    parent_storage_dtype: str
    child_storage_dtype: str
    layer_bindings: tuple[Lfm2MovingGaugeBoundaryLayerBinding, ...]
    focused_writer_layer_indices: tuple[int, ...]
    basis_split_sha256: str
    implementation_sha256: str
    provenance_sha256s: Mapping[str, str]
    checkpoint_hmac_key_id: str
    shard_batch_count: int = 32
    request_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("parent_cache_key", "child_cache_key"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in (
            "expected_batch_count",
            "expected_aligned_row_count",
            "parent_width",
            "child_width",
            "parent_parameter_count",
            "child_parameter_count",
            "shard_batch_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("parent_storage_dtype", "child_storage_dtype"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        bindings = tuple(self.layer_bindings)
        if (
            not bindings
            or any(
                not isinstance(value, Lfm2MovingGaugeBoundaryLayerBinding)
                for value in bindings
            )
            or tuple(value.layer_index for value in bindings)
            != tuple(range(len(bindings)))
            or len({value.parent_layer_index for value in bindings}) != len(bindings)
            or len({value.child_layer_index for value in bindings}) != len(bindings)
        ):
            raise Lfm2MovingGaugeBoundaryPhaseError(
                "Layer bindings must cover logical layers contiguously with unique "
                "physical parent and child layers."
            )
        focused = tuple(self.focused_writer_layer_indices)
        if (
            not focused
            or focused != tuple(sorted(set(focused)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= len(bindings)
                for value in focused
            )
        ):
            raise Lfm2MovingGaugeBoundaryPhaseError(
                "focused_writer_layer_indices must be a nonempty ordered subset."
            )
        object.__setattr__(
            self,
            "basis_split_sha256",
            _digest(self.basis_split_sha256, "basis_split_sha256"),
        )
        implementation_sha256 = _digest(
            self.implementation_sha256,
            "implementation_sha256",
        )
        if implementation_sha256 != lfm2_moving_gauge_boundary_implementation_sha256():
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Boundary request implementation differs from the executing source."
            )
        object.__setattr__(self, "implementation_sha256", implementation_sha256)
        provenance = dict(self.provenance_sha256s)
        if set(provenance) != _REQUIRED_PROVENANCE_KEYS:
            missing = sorted(_REQUIRED_PROVENANCE_KEYS.difference(provenance))
            extra = sorted(set(provenance).difference(_REQUIRED_PROVENANCE_KEYS))
            raise Lfm2MovingGaugeBoundaryPhaseError(
                f"Provenance inventory differs; missing={missing}, extra={extra}."
            )
        provenance = {
            _identifier(name, "provenance name"): _digest(value, name)
            for name, value in sorted(provenance.items())
        }
        object.__setattr__(
            self,
            "checkpoint_hmac_key_id",
            _identifier(self.checkpoint_hmac_key_id, "checkpoint_hmac_key_id"),
        )
        object.__setattr__(self, "layer_bindings", bindings)
        object.__setattr__(self, "focused_writer_layer_indices", focused)
        object.__setattr__(
            self,
            "provenance_sha256s",
            MappingProxyType(provenance),
        )
        expected = canonical_sha256(self._body())
        if self.request_sha256 not in {"", expected}:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Boundary import request seal is invalid."
            )
        object.__setattr__(self, "request_sha256", expected)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_MOVING_GAUGE_BOUNDARY_REQUEST_SCHEMA,
            "phase_name": LFM2_MOVING_GAUGE_BOUNDARY_PHASE,
            "parent_cache_key": self.parent_cache_key,
            "child_cache_key": self.child_cache_key,
            "expected_batch_count": self.expected_batch_count,
            "expected_aligned_row_count": self.expected_aligned_row_count,
            "parent_width": self.parent_width,
            "child_width": self.child_width,
            "parent_parameter_count": self.parent_parameter_count,
            "child_parameter_count": self.child_parameter_count,
            "compression_ratio_exact_rational": [
                self.parent_parameter_count,
                self.child_parameter_count,
            ],
            "parent_storage_dtype": self.parent_storage_dtype,
            "child_storage_dtype": self.child_storage_dtype,
            "layer_bindings": [value.as_dict() for value in self.layer_bindings],
            "focused_writer_layer_indices": list(self.focused_writer_layer_indices),
            "basis_split_sha256": self.basis_split_sha256,
            "implementation_sha256": self.implementation_sha256,
            "selected_residual_fields": list(_RESIDUAL_FIELDS),
            "focused_physical_writer_fields": list(_FOCUSED_WRITER_FIELDS),
            "provenance_sha256s": dict(self.provenance_sha256s),
            "checkpoint_hmac_key_id": self.checkpoint_hmac_key_id,
            "shard_batch_count": self.shard_batch_count,
            "numeric_token_id_equality_required": False,
            "basis_split_is_chart_authority": True,
            "bridge_split_is_provenance_only": True,
            "source_storage_dtype_preserved": True,
            "moving_gauge_fit_dtype": "float64",
            "terminal_boundary": "final-layer-residual.post_layer",
            "lm_head_input_used_as_terminal": False,
            "operation_contract": dict(_ZERO_EXECUTION),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "request_sha256": self.request_sha256}

    def verify_authentication(self) -> bool:
        try:
            return bool(
                self.request_sha256 == canonical_sha256(self._body())
                and self.implementation_sha256
                == lfm2_moving_gauge_boundary_implementation_sha256()
            )
        except (OSError, TypeError, ValueError):
            return False

    @property
    def phase_key(self) -> FoldPhaseKey:
        dependencies = {
            "request": self.request_sha256,
            **dict(self.provenance_sha256s),
            "parent_cache_key": self.parent_cache_key,
            "child_cache_key": self.child_cache_key,
        }
        numeric = {
            "parent_storage_dtype": self.parent_storage_dtype,
            "child_storage_dtype": self.child_storage_dtype,
            "moving_gauge_fit_dtype": "float64",
            "phase_order": "prefill-then-numeric-decode",
            "row_order": "record-ordinal-phase-local-row",
            "record_weighting": "equal-record-mass",
            "basis_split_sha256": self.basis_split_sha256,
            "parent_parameter_count": self.parent_parameter_count,
            "child_parameter_count": self.child_parameter_count,
            "shard_batch_count": self.shard_batch_count,
            "numeric_token_id_equality_required": False,
        }
        abi = {
            "request_schema": LFM2_MOVING_GAUGE_BOUNDARY_REQUEST_SCHEMA,
            "checkpoint_schema": LFM2_MOVING_GAUGE_BOUNDARY_CHECKPOINT_SCHEMA,
            "row_index_schema": LFM2_MOVING_GAUGE_BOUNDARY_ROW_INDEX_SCHEMA,
            "receipt_schema": LFM2_MOVING_GAUGE_BOUNDARY_RECEIPT_SCHEMA,
            "anchor_schema": LFM2_MOVING_GAUGE_BOUNDARY_ANCHOR_SCHEMA,
            "shard_schema": LFM2_MOVING_GAUGE_BOUNDARY_SHARD_SCHEMA,
            "implementation_sha256": self.implementation_sha256,
        }
        return FoldPhaseKey(
            phase_name=LFM2_MOVING_GAUGE_BOUNDARY_PHASE,
            dependency_sha256s=dependencies,
            numeric_contract=numeric,
            abi=abi,
        )


class Lfm2MovingGaugeBoundaryPhaseAuthority:
    """Runtime-only HMAC authority with boundary-specific domains."""

    __slots__ = ("__secret", "key_id")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        self.key_id = _identifier(key_id, "boundary phase HMAC key_id")
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                "Boundary phase HMAC secret must contain at least 32 bytes."
            )
        self.__secret = bytes(secret)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("Boundary phase HMAC authority cannot be serialized.")

    def _sign(self, domain: bytes, body: Mapping[str, Any]) -> str:
        return hmac.new(
            self.__secret,
            domain + canonical_json_bytes(dict(body)),
            hashlib.sha256,
        ).hexdigest()

    def sign_checkpoint(self, body: Mapping[str, Any]) -> str:
        return self._sign(_CHECKPOINT_DOMAIN, body)

    def verify_checkpoint(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return (
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_checkpoint(body), claimed)
        )

    def sign_anchor(self, body: Mapping[str, Any]) -> str:
        return self._sign(_ANCHOR_DOMAIN, body)

    def verify_anchor(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return (
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_anchor(body), claimed)
        )


def _authenticated_self_seal(
    value: Any,
    *,
    seal_name: str,
    subject: str,
) -> dict[str, Any]:
    row = _canonical(value, subject)
    if not isinstance(row, dict):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{subject} must be a mapping."
        )
    body = dict(row)
    claimed = body.pop(seal_name, None)
    if _digest(claimed, f"{subject} {seal_name}") != canonical_sha256(body):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{subject} self-seal does not verify."
        )
    return row


def _authenticate_balanced_basis_budget_authority(
    phase: AuthenticatedLfm2BalancedBasisPhase,
    authority: PhaseAnchorAuthority,
    counted_plan: CountedChildStatePlan,
    phase_cache: DynamicFoldPhaseCache,
) -> dict[str, Any]:
    """Reauthenticate the HMAC basis source and derive, rather than trust, counts."""

    if not isinstance(phase, AuthenticatedLfm2BalancedBasisPhase):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis source has the wrong type."
        )
    if not isinstance(authority, PhaseAnchorAuthority):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis HMAC authority has the wrong type."
        )
    if not isinstance(counted_plan, CountedChildStatePlan):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Counted child-state authority has the wrong type."
        )
    if not isinstance(phase_cache, DynamicFoldPhaseCache):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis phase cache has the wrong type."
        )
    try:
        restored_plan = counted_child_state_plan_from_dict(counted_plan.as_dict())
        phase_key = phase.phase_key
        if not isinstance(phase_key, FoldPhaseKey):
            raise TypeError("balanced-basis phase key is not typed")
        phase_key_document = _canonical(
            phase_key.as_dict(),
            "balanced-basis phase key",
        )
        receipt = _canonical(phase.receipt, "balanced-basis receipt")
        anchor = _canonical(phase.source_anchor, "balanced-basis source anchor")
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis source documents could not be reconstructed."
        ) from exc
    if not isinstance(receipt, dict) or set(receipt) != _BALANCED_BASIS_RECEIPT_FIELDS:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis receipt has a noncanonical field inventory."
        )
    receipt_body = dict(receipt)
    receipt_sha256 = receipt_body.pop("receipt_sha256", None)
    if (
        receipt.get("schema") != LFM2_BALANCED_BASIS_RECEIPT_SCHEMA
        or _digest(receipt_sha256, "balanced-basis receipt_sha256")
        != canonical_sha256(receipt_body)
        or receipt.get("phase_key") != phase_key_document
        or receipt.get("phase_key_sha256") != phase_key.phase_key_sha256
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis receipt or phase-key seal does not verify."
        )
    if not isinstance(anchor, dict) or set(anchor) != (
        _BALANCED_BASIS_ANCHOR_BODY_FIELDS | {"hmac_sha256"}
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis source anchor has a noncanonical field inventory."
        )
    anchor_body = dict(anchor)
    claimed_hmac = anchor_body.pop("hmac_sha256", None)
    if (
        anchor_body.get("schema") != LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA
        or anchor_body.get("algorithm") != "hmac-sha256"
        or anchor_body.get("key_id") != authority.key_id
        or anchor_body.get("phase_name") != LFM2_BALANCED_BASIS_PHASE
        or not authority.verify(anchor_body, claimed_hmac)
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis source-anchor HMAC does not verify."
        )
    tensor_file = receipt.get("tensor_file")
    descriptors = receipt.get("tensor_descriptors")
    basis_report = _authenticated_self_seal(
        receipt.get("basis_report"),
        seal_name="result_sha256",
        subject="balanced-basis wrapper report",
    )
    nested_report = _authenticated_self_seal(
        receipt.get("balanced_basis_report"),
        seal_name="run_sha256",
        subject="balanced-basis nested report",
    )
    if (
        not isinstance(tensor_file, dict)
        or set(tensor_file)
        != {"filename", "format", "metadata", "size_bytes", "sha256"}
        or tensor_file.get("filename") != "basis.safetensors"
        or tensor_file.get("format") != "safetensors"
        or isinstance(tensor_file.get("size_bytes"), bool)
        or not isinstance(tensor_file.get("size_bytes"), int)
        or tensor_file["size_bytes"] <= 0
        or not isinstance(descriptors, list)
        or not descriptors
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis tensor receipt is malformed."
        )
    descriptor_by_name: dict[str, dict[str, Any]] = {}
    for raw in descriptors:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"name", "dtype", "shape", "numel", "size_bytes", "sha256"}
            or not isinstance(raw.get("name"), str)
            or not raw["name"]
            or raw["name"] in descriptor_by_name
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Balanced-basis tensor descriptor inventory is malformed."
            )
        _digest(raw.get("sha256"), f"balanced-basis tensor {raw['name']}")
        descriptor_by_name[raw["name"]] = dict(raw)
    basis_descriptor = descriptor_by_name.get("basis")
    expected_anchor = {
        "schema": LFM2_BALANCED_BASIS_SOURCE_ANCHOR_SCHEMA,
        "algorithm": "hmac-sha256",
        "key_id": authority.key_id,
        "phase_name": LFM2_BALANCED_BASIS_PHASE,
        "phase_key_sha256": phase_key.phase_key_sha256,
        "receipt_sha256": receipt_sha256,
        "tensor_file_sha256": _digest(
            tensor_file.get("sha256"),
            "balanced-basis tensor file",
        ),
        "tensor_inventory_sha256": canonical_sha256(descriptors),
        "basis_result_sha256": basis_report["result_sha256"],
        "balanced_run_sha256": nested_report["run_sha256"],
        "basis_map_sha256": (
            None if basis_descriptor is None else basis_descriptor["sha256"]
        ),
    }
    if basis_descriptor is None or anchor_body != expected_anchor:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis HMAC does not bind the exact receipt and map."
        )
    entry = phase_cache.lookup(phase_key)
    if entry is None:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis completion is absent from its immutable phase cache."
        )
    completion_result = {
        "schema": LFM2_BALANCED_BASIS_RECEIPT_SCHEMA,
        "receipt_sha256": receipt_sha256,
        "tensor_file_sha256": tensor_file["sha256"],
        "tensor_inventory_sha256": canonical_sha256(descriptors),
        "basis_result_sha256": basis_report["result_sha256"],
        "balanced_run_sha256": nested_report["run_sha256"],
        "basis_map_sha256": basis_descriptor["sha256"],
    }
    payload_files = entry.payload_inventory.get("files")
    file_by_path = (
        {
            value["path"]: value
            for value in payload_files
            if isinstance(value, dict) and isinstance(value.get("path"), str)
        }
        if isinstance(payload_files, list)
        else {}
    )
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    anchor_bytes = canonical_json_bytes(anchor) + b"\n"
    if (
        entry.phase_key.as_dict() != phase_key_document
        or entry.completion_sha256 != phase.completion_sha256
        or _canonical(dict(entry.result), "balanced-basis completion result")
        != completion_result
        or set(file_by_path)
        != {"basis.safetensors", "receipt.json", "source-anchor.json"}
        or file_by_path["basis.safetensors"].get("sha256") != tensor_file["sha256"]
        or file_by_path["basis.safetensors"].get("size_bytes")
        != tensor_file["size_bytes"]
        or file_by_path["receipt.json"].get("sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
        or file_by_path["receipt.json"].get("size_bytes") != len(receipt_bytes)
        or file_by_path["source-anchor.json"].get("sha256")
        != hashlib.sha256(anchor_bytes).hexdigest()
        or file_by_path["source-anchor.json"].get("size_bytes") != len(anchor_bytes)
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis completion does not reauthenticate against its "
            "immutable phase-cache entry."
        )
    if _canonical(phase.report, "balanced-basis phase report") != basis_report:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis phase report differs from its HMAC receipt."
        )
    basis_result = phase.basis_result
    if (
        getattr(basis_result, "map_sha256", None) != basis_descriptor["sha256"]
        or _canonical(
            getattr(basis_result, "report", None),
            "balanced-basis result report",
        )
        != nested_report
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis in-memory map differs from its HMAC receipt."
        )
    dependencies = phase_key_document.get("dependency_sha256s")
    abi = phase_key_document.get("abi")
    bindings = receipt.get("bindings")
    if (
        not isinstance(dependencies, dict)
        or not isinstance(abi, dict)
        or not isinstance(bindings, dict)
        or abi.get("source_anchor_key_id") != authority.key_id
        or abi.get("source_width") != getattr(basis_result, "source_width", None)
        or abi.get("target_width") != getattr(basis_result, "target_width", None)
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis key, widths, bindings, or HMAC authority changed."
        )
    external = bindings.get("external_dependency_sha256s")
    if (
        not isinstance(external, dict)
        or {
            key.removeprefix("external."): value
            for key, value in dependencies.items()
            if key.startswith("external.")
        }
        != external
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis external dependency bindings changed."
        )
    if (
        external.get("counted_child_state_plan") != restored_plan.fingerprint
        or dependencies.get("external.counted_child_state_plan")
        != restored_plan.fingerprint
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Counted child-state plan differs from the HMAC basis authority."
        )
    route_artifact = _digest(
        external.get("carrier_route_artifact"),
        "balanced-basis carrier route artifact",
    )
    if {
        allocation.artifact_sha256
        for allocation in restored_plan.continuous_allocations
    } != {route_artifact}:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Counted child-state allocations differ from the HMAC route artifact."
        )
    inventory = bindings.get("inventory")
    if not isinstance(inventory, dict):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis parent inventory is absent."
        )
    inventory_body = dict(inventory)
    inventory_sha256 = inventory_body.pop("inventory_sha256", None)
    alias_groups = inventory.get("parameter_alias_groups")
    if (
        _digest(inventory_sha256, "balanced-basis parent inventory")
        != canonical_sha256(inventory_body)
        or bindings.get("inventory_sha256") != inventory_sha256
        or dependencies.get("parent_inventory") != inventory_sha256
        or not isinstance(alias_groups, list)
        or not alias_groups
        or any(
            not isinstance(row, dict)
            or isinstance(row.get("numel"), bool)
            or not isinstance(row.get("numel"), int)
            or row["numel"] <= 0
            for row in alias_groups
        )
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis parent inventory seal or alias counts changed."
        )
    parent_parameter_count = sum(int(row["numel"]) for row in alias_groups)
    selected_target = bindings.get("selected_target")
    dense_parameter_count = (
        selected_target.get("exact_target_unique_parameters")
        if isinstance(selected_target, dict)
        else None
    )
    if (
        isinstance(dense_parameter_count, bool)
        or not isinstance(dense_parameter_count, int)
        or dense_parameter_count <= 0
        or restored_plan.parent_unique_parameters != parent_parameter_count
        or restored_plan.dense_model_parameter_elements != dense_parameter_count
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "HMAC basis inventory, selected dense target, and counted plan disagree."
        )
    return {
        "identity": phase.as_dict(),
        "basis_report": basis_report,
        "parent_parameter_count": parent_parameter_count,
        "dense_parameter_count": dense_parameter_count,
        "child_parameter_count": restored_plan.target_total_parameter_elements,
        "route_artifact_sha256": route_artifact,
        "counted_plan_fingerprint": restored_plan.fingerprint,
    }


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryExtractionSources:
    """Deferred cold-source factories; never constructed on a phase hit."""

    parent_cache: Any = field(repr=False, compare=False)
    child_cache: Any = field(repr=False, compare=False)
    trace_adapter: Any = field(repr=False, compare=False)
    basis_phase: AuthenticatedLfm2BalancedBasisPhase = field(
        repr=False,
        compare=False,
    )
    basis_phase_authority: PhaseAnchorAuthority = field(
        repr=False,
        compare=False,
    )
    basis_phase_cache: DynamicFoldPhaseCache = field(
        repr=False,
        compare=False,
    )
    counted_child_state_plan: CountedChildStatePlan = field(
        repr=False,
        compare=False,
    )
    v3_first_divergence_request: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    v3_first_divergence_ledger: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    parent_session_factory: Callable[[], Any] = field(repr=False, compare=False)
    child_session_factory: Callable[[], Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(self.parent_session_factory) or not callable(
            self.child_session_factory
        ):
            raise TypeError("Selective session factories must be callable.")
        if not isinstance(self.basis_phase, AuthenticatedLfm2BalancedBasisPhase):
            raise TypeError(
                "basis_phase must be an authenticated balanced-basis phase."
            )
        if not isinstance(self.basis_phase_authority, PhaseAnchorAuthority):
            raise TypeError("basis_phase_authority must be PhaseAnchorAuthority.")
        if not isinstance(self.basis_phase_cache, DynamicFoldPhaseCache):
            raise TypeError("basis_phase_cache must be DynamicFoldPhaseCache.")
        if not isinstance(self.counted_child_state_plan, CountedChildStatePlan):
            raise TypeError("counted_child_state_plan must be CountedChildStatePlan.")
        try:
            counted_child_state_plan_from_dict(self.counted_child_state_plan.as_dict())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Counted child-state plan did not reconstruct exactly."
            ) from exc
        try:
            v3_request = authenticate_lfm2_exact4x_first_divergence_request(
                _deep_thaw(self.v3_first_divergence_request)
            )
            v3_ledger = authenticate_v3_first_divergence_ledger(
                _deep_thaw(self.v3_first_divergence_ledger),
                expected_request=v3_request,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "V3 request or ledger did not authenticate."
            ) from exc
        object.__setattr__(
            self,
            "v3_first_divergence_request",
            _deep_freeze(v3_request),
        )
        object.__setattr__(
            self,
            "v3_first_divergence_ledger",
            _deep_freeze(v3_ledger),
        )


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryExtractionProgress:
    """Authenticated partial-prefix status for an interrupted cold import."""

    phase_key_sha256: str
    total_shards: int
    completed_shards: int
    checkpoint_chain_sha256: str
    complete: bool = False


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeBoundaryPair:
    """One aligned parent/child residual boundary subset."""

    layer_index: int
    endpoint: str
    split: str
    parent_rows: torch.Tensor = field(repr=False, compare=False)
    child_rows: torch.Tensor = field(repr=False, compare=False)
    record_ids: tuple[str, ...]
    global_row_indices: tuple[int, ...]
    weights: torch.Tensor = field(repr=False, compare=False)
    row_alignment_sha256: str
    split_identity_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2MovingGaugeFocusedWriterOutputs:
    """Physical operator and MLP writer outputs for one focused layer."""

    layer_index: int
    split: str
    parent_operator_output: torch.Tensor = field(repr=False, compare=False)
    parent_mlp_down: torch.Tensor = field(repr=False, compare=False)
    child_operator_output: torch.Tensor = field(repr=False, compare=False)
    child_mlp_down: torch.Tensor = field(repr=False, compare=False)
    record_ids: tuple[str, ...]
    global_row_indices: tuple[int, ...]

    @property
    def child_combined_writer_output(self) -> torch.Tensor:
        return (self.child_operator_output + self.child_mlp_down).contiguous()


def _cache_completion_sha256(cache: Any) -> str:
    completion = getattr(cache, "_completion", None)
    if isinstance(completion, Mapping):
        return _digest(completion.get("completion_sha256"), "cache completion")
    path = getattr(cache, "path", None)
    if path is None:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Source cache exposes no completion evidence."
        )
    return _digest(
        _read_json(Path(path) / "complete.json").get("completion_sha256"),
        "cache completion",
    )


def _cache_manifest(cache: Any) -> dict[str, Any]:
    manifest = getattr(cache, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Source cache exposes no manifest."
        )
    return _canonical(dict(manifest), "source cache manifest")


def _verify_cache_binding(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    cache: Any,
    *,
    side: str,
) -> dict[str, Any]:
    manifest = _cache_manifest(cache)
    provenance = request.provenance_sha256s
    expected_key = getattr(request, f"{side}_cache_key")
    expected_width = getattr(request, f"{side}_width")
    if (
        getattr(cache, "cache_key", None) != expected_key
        or manifest.get("manifest_sha256") != provenance[f"{side}_manifest"]
        or _cache_completion_sha256(cache) != provenance[f"{side}_completion"]
        or manifest.get("parent", {}).get("identity_sha256")
        != provenance[f"{side}_identity"]
        or manifest.get("layer_topology", {}).get("layer_topology_sha256")
        != provenance[f"{side}_topology"]
        or manifest.get("corpus", {}).get("batch_count") != request.expected_batch_count
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{side} cache differs from the immutable request."
        )
    if side == "child" and (
        manifest.get("parent", {}).get("state_dict_sha256")
        != provenance["child_state_dict"]
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Child cache state dictionary differs from the requested child."
        )
    topology = manifest.get("layer_topology", {}).get("layers")
    if not isinstance(topology, list):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{side} cache topology is incomplete."
        )
    for binding in request.layer_bindings:
        physical = getattr(binding, f"{side}_layer_index")
        if physical >= len(topology):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"{side} cache does not contain physical layer {physical}."
            )
        contract = topology[physical].get("contract", {})
        if (
            contract.get("kind") != binding.operator_kind
            or contract.get("hidden_size") != expected_width
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"{side} layer {physical} topology differs from its binding."
            )
    return manifest


def _verify_adapter_and_inputs(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    sources: Lfm2MovingGaugeBoundaryExtractionSources,
    parent_manifest: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
) -> tuple[tuple[Any, ...], Mapping[int, str]]:
    try:
        basis_authority = _authenticate_balanced_basis_budget_authority(
            sources.basis_phase,
            sources.basis_phase_authority,
            sources.counted_child_state_plan,
            sources.basis_phase_cache,
        )
        basis_identity = basis_authority["identity"]
        v3_request = authenticate_lfm2_exact4x_first_divergence_request(
            _deep_thaw(sources.v3_first_divergence_request)
        )
        v3_ledger = authenticate_v3_first_divergence_ledger(
            _deep_thaw(sources.v3_first_divergence_ledger),
            expected_request=v3_request,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis or v3 source authority did not authenticate."
        ) from exc
    budget = v3_request.get("budget_observation")
    if (
        basis_identity.get("phase_key_sha256")
        != request.provenance_sha256s["basis_phase_key"]
        or basis_identity.get("completion_sha256")
        != request.provenance_sha256s["basis_completion"]
        or basis_identity.get("basis_map_sha256")
        != request.provenance_sha256s["basis_map"]
        or basis_identity.get("source_width") != request.parent_width
        or basis_identity.get("target_width") != request.child_width
        or basis_authority["parent_parameter_count"] != request.parent_parameter_count
        or basis_authority["child_parameter_count"] != request.child_parameter_count
        or basis_authority["route_artifact_sha256"]
        != request.provenance_sha256s["route_package"]
        or v3_request.get("request_sha256") != request.provenance_sha256s["v3_request"]
        or v3_ledger.get("ledger_sha256") != request.provenance_sha256s["v3_ledger"]
        or v3_ledger.get("batch_count") != request.expected_batch_count
        or v3_request.get("child_package_sha256")
        != request.provenance_sha256s["child_package"]
        or v3_request.get("child_state_sha256")
        != request.provenance_sha256s["child_state_dict"]
        or v3_request.get("route_package_sha256")
        != request.provenance_sha256s["route_package"]
        or v3_request.get("route_runtime_sha256")
        != request.provenance_sha256s["route_runtime"]
        or v3_request.get("portable_runtime_receipt_sha256")
        != request.provenance_sha256s["portable_runtime_receipt"]
        or v3_request.get("trace_cache_key") != request.parent_cache_key
        or v3_request.get("trace_selective_session_sha256")
        != request.provenance_sha256s["parent_selective_session"]
        or v3_request.get("basis_phase_key_sha256")
        != request.provenance_sha256s["basis_phase_key"]
        or v3_request.get("basis_completion_sha256")
        != request.provenance_sha256s["basis_completion"]
        or v3_request.get("basis_map_sha256") != request.provenance_sha256s["basis_map"]
        or not isinstance(budget, Mapping)
        or budget.get("parent_parameter_elements") != request.parent_parameter_count
        or budget.get("physical_dense_parameter_elements")
        != basis_authority["dense_parameter_count"]
        or budget.get("continuous_route_parameter_elements")
        != sources.counted_child_state_plan.continuous_parameter_elements
        or budget.get("counted_total_parameter_elements")
        != request.child_parameter_count
        or budget.get("target_total_parameter_elements")
        != request.child_parameter_count
        or budget.get("one_complete_child_observed") is not True
        or budget.get("partial_component_variant") is not False
        or budget.get("pruning_credit") is not False
        or budget.get("quantization_credit") is not False
        or budget.get("parameter_sharing_credit") is not False
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary request differs from authenticated basis/v3 budget authority."
        )
    adapter = sources.trace_adapter
    verify = getattr(adapter, "verify", None)
    if not callable(verify) or verify() is not adapter:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Tokenizer-bridge trace adapter did not verify."
        )
    if (
        getattr(adapter, "evidence_sha256", None)
        != request.provenance_sha256s["bridge_trace_adapter"]
        or getattr(adapter, "bridge_artifact_sha256", None)
        != request.provenance_sha256s["bridge_artifact"]
        or getattr(adapter, "transport_profile_fingerprint", None)
        != request.provenance_sha256s["bridge_profile"]
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Tokenizer bridge differs from the requested artifact/profile."
        )
    records = tuple(getattr(adapter, "records", ()))
    parent_hashes = tuple(
        parent_manifest.get("corpus", {}).get("ordered_batch_input_sha256", ())
    )
    child_hashes = tuple(
        child_manifest.get("corpus", {}).get("ordered_batch_input_sha256", ())
    )
    if not (
        len(records)
        == len(parent_hashes)
        == len(child_hashes)
        == request.expected_batch_count
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Bridge, cache corpora, and basis split have different counts."
        )
    for ordinal, record in enumerate(records):
        if (
            getattr(record, "ordinal", None) != ordinal
            or getattr(record, "donor_cache_input_sha256", None)
            != parent_hashes[ordinal]
            or getattr(record, "target_cache_input_sha256", None)
            != child_hashes[ordinal]
            or not isinstance(getattr(record, "split", None), str)
            or not getattr(record, "split").strip()
            or not isinstance(getattr(record, "record_id", None), str)
            or not getattr(record, "record_id").strip()
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"Bridge ordinal {ordinal} does not authenticate both caches."
            )
        _digest(getattr(record, "alignment_sha256", None), "record alignment")
    try:
        basis_split_by_batch = authoritative_split_by_batch(
            basis_authority["basis_report"],
            tuple(SimpleNamespace(input_sha256=value) for value in parent_hashes),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Balanced-basis FIT/heldout authority did not match the parent cache."
        ) from exc
    actual_basis_split_sha256 = fingerprint_lfm2_moving_gauge_basis_split(
        basis_split_by_batch,
        basis_phase_key_sha256=request.provenance_sha256s["basis_phase_key"],
        basis_completion_sha256=request.provenance_sha256s["basis_completion"],
        basis_map_sha256=request.provenance_sha256s["basis_map"],
    )
    if actual_basis_split_sha256 != request.basis_split_sha256:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary split differs from the authenticated balanced-basis report."
        )
    return records, basis_split_by_batch


def _source_field_names(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    *,
    side: str,
    phases: Sequence[str],
) -> tuple[str, ...]:
    names: list[str] = []
    for phase in phases:
        for binding in request.layer_bindings:
            physical = getattr(binding, f"{side}_layer_index")
            prefix = f"{phase}.layer.{physical:03d}."
            names.extend(prefix + field for field in _RESIDUAL_FIELDS)
            if binding.layer_index in request.focused_writer_layer_indices:
                names.extend(prefix + field for field in _FOCUSED_WRITER_FIELDS)
    return tuple(sorted(names))


def _normalized_tensor_name(layer_index: int, field_name: str) -> str:
    return f"layer.{layer_index:03d}.{field_name}"


def _session_plan(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    session: Any,
    records: Sequence[Any],
    basis_split_by_batch: Mapping[int, str],
    *,
    side: str,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
]:
    if (
        getattr(session, "cache_key", None) != getattr(request, f"{side}_cache_key")
        or getattr(session, "evidence_sha256", None)
        != request.provenance_sha256s[f"{side}_selective_session"]
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{side} selective session differs from the request."
        )
    references = tuple(getattr(session, "batch_references", ()))
    if len(references) != request.expected_batch_count:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{side} selective session has the wrong batch count."
        )
    rows: list[dict[str, Any]] = []
    names_by_batch: list[tuple[str, ...]] = []
    phases_by_batch: list[tuple[str, ...]] = []
    global_row = 0
    for ordinal, (reference, record) in enumerate(
        zip(references, records, strict=True)
    ):
        expected_input = getattr(
            record,
            "donor_cache_input_sha256"
            if side == "parent"
            else "target_cache_input_sha256",
        )
        if (
            getattr(reference, "index", None) != ordinal
            or getattr(reference, "input_sha256", None) != expected_input
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"{side} reference {ordinal} differs from bridge evidence."
            )
        inventory = getattr(reference, "record", {}).get("tensors")
        if not isinstance(inventory, Mapping):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"{side} reference {ordinal} has no tensor inventory."
            )
        first = request.layer_bindings[0]
        first_physical = getattr(first, f"{side}_layer_index")
        suffix = f".layer.{first_physical:03d}.residual.pre"
        phases = tuple(
            sorted(
                {name[: -len(suffix)] for name in inventory if name.endswith(suffix)},
                key=_phase_sort_key,
            )
        )
        if not phases or phases[0] != "prefill":
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"{side} batch {ordinal} has no canonical prefill phase."
            )
        selected_names = _source_field_names(request, side=side, phases=phases)
        expected_width = getattr(request, f"{side}_width")
        expected_dtype = getattr(request, f"{side}_storage_dtype")
        phase_rows: dict[str, int] = {}
        for name in selected_names:
            descriptor = inventory.get(name)
            shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
            dtype = descriptor.get("dtype") if isinstance(descriptor, Mapping) else None
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[0] != 1
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
                or shape[2] != expected_width
                or dtype != expected_dtype
            ):
                raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                    f"{side} tensor {name!r} changed shape, width, or dtype."
                )
            phase = name.split(".layer.", 1)[0]
            previous = phase_rows.setdefault(phase, shape[1])
            if previous != shape[1]:
                raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                    f"{side} batch {ordinal} phase rows are not aligned."
                )
        batch_rows: list[dict[str, Any]] = []
        for phase in phases:
            for local_row in range(phase_rows[phase]):
                batch_rows.append(
                    {
                        "global_row_index": global_row,
                        "record_ordinal": ordinal,
                        "record_id": str(getattr(record, "record_id")),
                        "bridge_split": str(getattr(record, "split")),
                        "basis_split": str(basis_split_by_batch[ordinal]),
                        "alignment_sha256": str(getattr(record, "alignment_sha256")),
                        "parent_input_sha256": str(
                            getattr(record, "donor_cache_input_sha256")
                        ),
                        "child_input_sha256": str(
                            getattr(record, "target_cache_input_sha256")
                        ),
                        "phase": phase,
                        "phase_row_index": local_row,
                    }
                )
                global_row += 1
        rows.extend(batch_rows)
        names_by_batch.append(selected_names)
        phases_by_batch.append(phases)
    if global_row != request.expected_aligned_row_count:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{side} session produced {global_row} rows, expected "
            f"{request.expected_aligned_row_count}."
        )
    return tuple(rows), tuple(names_by_batch), tuple(phases_by_batch)


def _checkpoint_count(request: Lfm2MovingGaugeBoundaryImportRequest) -> int:
    return math.ceil(request.expected_batch_count / request.shard_batch_count)


def _checkpoint_side_and_range(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    ordinal: int,
) -> tuple[str, int, int, int]:
    per_side = _checkpoint_count(request)
    total = per_side * 2
    if ordinal < 0 or ordinal >= total:
        raise Lfm2MovingGaugeBoundaryPhaseError("Checkpoint ordinal is out of range.")
    side_index, side_ordinal = divmod(ordinal, per_side)
    side = _SIDES[side_index]
    start = side_ordinal * request.shard_batch_count
    stop = min(request.expected_batch_count, start + request.shard_batch_count)
    return side, side_ordinal, start, stop


def _checkpoint_path(root: Path, ordinal: int) -> Path:
    return root / "checkpoints" / f"checkpoint-{ordinal:04d}.json"


def _authenticate_checkpoint(
    value: Mapping[str, Any],
    *,
    request: Lfm2MovingGaugeBoundaryImportRequest,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
    expected_ordinal: int,
    expected_previous_hmac: str,
    shard_root: Path,
) -> dict[str, Any]:
    row = _canonical(dict(value), "boundary checkpoint")
    if not isinstance(row, dict):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary checkpoint must be a mapping."
        )
    if set(row) != _CHECKPOINT_FIELDS:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary checkpoint has a noncanonical field inventory."
        )
    claimed_hmac = row.pop("checkpoint_hmac_sha256", None)
    claimed_sha = row.pop("checkpoint_sha256", None)
    side, side_ordinal, start, stop = _checkpoint_side_and_range(
        request, expected_ordinal
    )
    if (
        row.get("schema") != LFM2_MOVING_GAUGE_BOUNDARY_CHECKPOINT_SCHEMA
        or row.get("request_sha256") != request.request_sha256
        or row.get("phase_key_sha256") != request.phase_key.phase_key_sha256
        or row.get("checkpoint_hmac_key_id") != authority.key_id
        or row.get("checkpoint_ordinal") != expected_ordinal
        or row.get("side") != side
        or row.get("side_shard_ordinal") != side_ordinal
        or row.get("batch_start") != start
        or row.get("batch_stop") != stop
        or row.get("previous_checkpoint_hmac_sha256") != expected_previous_hmac
        or row.get("source_cache_key") != getattr(request, f"{side}_cache_key")
        or row.get("source_selective_session_sha256")
        != request.provenance_sha256s[f"{side}_selective_session"]
        or row.get("source_storage_dtype_preserved") is not True
        or row.get("unselected_tensor_payloads_loaded") is not False
        or row.get("execution") != _ZERO_EXECUTION
        or claimed_sha != canonical_sha256(row)
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} differs from its request."
        )
    sealed = {**row, "checkpoint_sha256": claimed_sha}
    if not authority.verify_checkpoint(sealed, claimed_hmac):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} HMAC does not verify."
        )
    source_records = row.get("source_records")
    if (
        not isinstance(source_records, list)
        or len(source_records) != stop - start
        or any(
            not isinstance(value, dict)
            or set(value) != _SOURCE_RECORD_FIELDS
            or value.get("batch_index") != batch_index
            or any(
                _SHA256.fullmatch(str(value.get(name, ""))) is None
                for name in (
                    "alignment_sha256",
                    "input_sha256",
                    "record_sha256",
                    "selected_payload_sha256",
                )
            )
            for batch_index, value in zip(
                range(start, stop), source_records, strict=True
            )
        )
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} source records are invalid."
        )
    rows = row.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} row index is empty."
        )
    for value in rows:
        if (
            not isinstance(value, dict)
            or set(value) != _ROW_FIELDS
            or value.get("record_ordinal") not in range(start, stop)
            or value.get("basis_split") not in _BASIS_SPLITS
            or not isinstance(value.get("bridge_split"), str)
            or not value["bridge_split"].strip()
            or not isinstance(value.get("record_id"), str)
            or not value["record_id"].strip()
            or any(
                _SHA256.fullmatch(str(value.get(name, ""))) is None
                for name in (
                    "alignment_sha256",
                    "child_input_sha256",
                    "parent_input_sha256",
                )
            )
            or isinstance(value.get("phase_row_index"), bool)
            or not isinstance(value.get("phase_row_index"), int)
            or value["phase_row_index"] < 0
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"Boundary checkpoint {expected_ordinal} row fields are invalid."
            )
        _phase_sort_key(value.get("phase"))
    global_indices = tuple(value["global_row_index"] for value in rows)
    record_ordinals = tuple(value["record_ordinal"] for value in rows)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in global_indices
        )
        or global_indices
        != tuple(range(global_indices[0], global_indices[0] + len(rows)))
        or row.get("row_start") != global_indices[0]
        or row.get("row_stop") != global_indices[-1] + 1
        or set(record_ordinals) != set(range(start, stop))
        or row.get("row_index_slice_sha256") != canonical_sha256(rows)
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} row range is invalid."
        )
    for record_ordinal in range(start, stop):
        selected = [
            value for value in rows if value["record_ordinal"] == record_ordinal
        ]
        source_record = source_records[record_ordinal - start]
        input_field = f"{side}_input_sha256"
        if {value["alignment_sha256"] for value in selected} != {
            source_record["alignment_sha256"]
        } or {value[input_field] for value in selected} != {
            source_record["input_sha256"]
        }:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"Boundary checkpoint {expected_ordinal} source record differs "
                "from its aligned row claims."
            )
        phases = tuple(dict.fromkeys(value["phase"] for value in selected))
        if tuple(sorted(phases, key=_phase_sort_key)) != phases:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"Boundary checkpoint {expected_ordinal} phase order is invalid."
            )
        for phase in phases:
            local = tuple(
                value["phase_row_index"]
                for value in selected
                if value["phase"] == phase
            )
            if local != tuple(range(len(local))):
                raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                    f"Boundary checkpoint {expected_ordinal} phase rows are invalid."
                )
    shard = row.get("shard")
    if not isinstance(shard, dict) or set(shard) != _SHARD_FIELDS:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary checkpoint shard has a noncanonical field inventory."
        )
    filename = shard.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary checkpoint shard filename is unsafe."
        )
    shard_path = shard_root / filename
    if (
        not shard_path.is_file()
        or shard_path.is_symlink()
        or _file_sha256(shard_path) != shard.get("file_sha256")
        or shard_path.stat().st_size != shard.get("size_bytes")
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} shard changed."
        )
    descriptors = shard.get("tensor_descriptors")
    expected_names = {
        _normalized_tensor_name(binding.layer_index, field_name)
        for binding in request.layer_bindings
        for field_name in (
            (*_RESIDUAL_FIELDS, *_FOCUSED_WRITER_FIELDS)
            if binding.layer_index in request.focused_writer_layer_indices
            else _RESIDUAL_FIELDS
        )
    }
    expected_dtype = getattr(request, f"{side}_storage_dtype")
    torch_dtype = getattr(torch, expected_dtype, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Unsupported checkpoint storage dtype {expected_dtype!r}."
        )
    element_size = torch.empty((), dtype=torch_dtype).element_size()
    expected_width = getattr(request, f"{side}_width")
    expected_rows = len(rows)
    if (
        not isinstance(descriptors, dict)
        or set(descriptors) != expected_names
        or shard.get("tensor_inventory_sha256") != canonical_sha256(descriptors)
        or any(
            not isinstance(descriptor, dict)
            or set(descriptor) != _TENSOR_DESCRIPTOR_FIELDS
            or descriptor.get("dtype") != expected_dtype
            or descriptor.get("shape") != [expected_rows, expected_width]
            or descriptor.get("numel") != expected_rows * expected_width
            or descriptor.get("size_bytes")
            != expected_rows * expected_width * element_size
            or _SHA256.fullmatch(str(descriptor.get("sha256", ""))) is None
            for descriptor in descriptors.values()
        )
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Boundary checkpoint {expected_ordinal} tensor inventory is invalid."
        )
    return {**sealed, "checkpoint_hmac_sha256": claimed_hmac}


def _load_checkpoint_prefix(
    root: Path,
    *,
    request: Lfm2MovingGaugeBoundaryImportRequest,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
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
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary checkpoints must form one canonical prefix."
        )
    total = _checkpoint_count(request) * 2
    if len(paths) > total:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary checkpoint prefix exceeds the request."
        )
    previous = _ZERO_SHA256
    rows: list[dict[str, Any]] = []
    selected_shard_root = root / "shards" if shard_root is None else shard_root
    for index, path in enumerate(paths):
        checkpoint = _authenticate_checkpoint(
            _read_json(path),
            request=request,
            authority=authority,
            expected_ordinal=index,
            expected_previous_hmac=previous,
            shard_root=selected_shard_root,
        )
        rows.append(checkpoint)
        previous = checkpoint["checkpoint_hmac_sha256"]
    return tuple(rows)


def _row_range_for_batches(
    rows: Sequence[Mapping[str, Any]],
    start: int,
    stop: int,
) -> tuple[dict[str, Any], ...]:
    selected = tuple(
        dict(row) for row in rows if start <= int(row["record_ordinal"]) < stop
    )
    if not selected:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary shard contains no aligned rows."
        )
    indices = tuple(int(row["global_row_index"]) for row in selected)
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary shard rows are not one canonical contiguous interval."
        )
    return selected


def _authenticate_selected_payload(
    session: Any,
    reference: Any,
    batch_index: int,
    tensor_names: Sequence[str],
    payload: Any,
) -> Mapping[str, torch.Tensor]:
    """Bind selected tensor bytes to the sealed cache reference, not duck typing."""

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
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Selected payload {batch_index} is not bound to its session/reference."
        )
    expected_descriptors: list[dict[str, Any]] = []
    logical_bytes = 0
    for name in names:
        descriptor = inventory.get(name)
        tensor = tensors.get(name)
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != _TENSOR_DESCRIPTOR_FIELDS
            or not isinstance(tensor, torch.Tensor)
            or _tensor_descriptor(tensor) != dict(descriptor)
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                f"Selected tensor {name!r} differs from its sealed descriptor."
            )
        expected_descriptors.append(dict(descriptor))
        logical_bytes += int(descriptor["size_bytes"])
    selection_body = {
        "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
        "session_sha256": getattr(session, "evidence_sha256", None),
        "cache_key": getattr(session, "cache_key", None),
        "batch_index": batch_index,
        "input_sha256": getattr(reference, "input_sha256", None),
        "record_sha256": getattr(reference, "record_sha256", None),
        "ordered_tensor_names": list(names),
        "ordered_tensor_descriptors": expected_descriptors,
        "logical_bytes": logical_bytes,
    }
    if getattr(payload, "logical_bytes", None) != logical_bytes or getattr(
        payload, "selection_sha256", None
    ) != canonical_sha256(selection_body):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"Selected payload {batch_index} has a noncanonical selection receipt."
        )
    return tensors


def _extract_one_shard(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    session: Any,
    records: Sequence[Any],
    all_rows: Sequence[Mapping[str, Any]],
    names_by_batch: Sequence[Sequence[str]],
    phases_by_batch: Sequence[Sequence[str]],
    *,
    side: str,
    checkpoint_ordinal: int,
    side_shard_ordinal: int,
    batch_start: int,
    batch_stop: int,
    previous_hmac: str,
    work_root: Path,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
) -> dict[str, Any]:
    pieces: dict[str, list[torch.Tensor]] = {}
    source_records: list[dict[str, Any]] = []
    for batch_index in range(batch_start, batch_stop):
        reference = session.batch_references[batch_index]
        payload = session.load_selected_tensors(
            batch_index,
            tuple(names_by_batch[batch_index]),
        )
        try:
            tensors = _authenticate_selected_payload(
                session,
                reference,
                batch_index,
                tuple(names_by_batch[batch_index]),
                payload,
            )
            for phase in phases_by_batch[batch_index]:
                for binding in request.layer_bindings:
                    physical = getattr(binding, f"{side}_layer_index")
                    fields = list(_RESIDUAL_FIELDS)
                    if binding.layer_index in request.focused_writer_layer_indices:
                        fields.extend(_FOCUSED_WRITER_FIELDS)
                    for field_name in fields:
                        source_name = f"{phase}.layer.{physical:03d}.{field_name}"
                        tensor = tensors[source_name]
                        rows = (
                            tensor.detach()
                            .to(device="cpu")
                            .contiguous()
                            .reshape(int(tensor.shape[1]), int(tensor.shape[2]))
                        )
                        pieces.setdefault(
                            _normalized_tensor_name(binding.layer_index, field_name),
                            [],
                        ).append(rows)
            source_records.append(
                {
                    "batch_index": batch_index,
                    "input_sha256": str(reference.input_sha256),
                    "record_sha256": str(reference.record_sha256),
                    "alignment_sha256": str(
                        getattr(records[batch_index], "alignment_sha256")
                    ),
                    "selected_payload_sha256": str(payload.selection_sha256),
                }
            )
        finally:
            session.release_selected_tensors(payload)
    joined = {
        name: torch.cat(values, dim=0).contiguous()
        for name, values in sorted(pieces.items())
    }
    shard_rows = _row_range_for_batches(all_rows, batch_start, batch_stop)
    expected_rows = len(shard_rows)
    expected_dtype = getattr(request, f"{side}_storage_dtype")
    expected_width = getattr(request, f"{side}_width")
    if any(
        tuple(value.shape) != (expected_rows, expected_width)
        or str(value.dtype).removeprefix("torch.") != expected_dtype
        for value in joined.values()
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            f"{side} shard tensor geometry changed while it was copied."
        )
    from safetensors.torch import save_file

    shard_root = work_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    stage = shard_root / f".stage-{uuid.uuid4().hex}.safetensors"
    save_file(
        joined,
        str(stage),
        metadata={
            "schema": LFM2_MOVING_GAUGE_BOUNDARY_SHARD_SCHEMA,
            "request_sha256": request.request_sha256,
            "side": side,
            "side_shard_ordinal": str(side_shard_ordinal),
        },
    )
    file_sha = _file_sha256(stage)
    filename = f"{side}-shard-{side_shard_ordinal:04d}-{file_sha[:16]}.safetensors"
    shard_path = shard_root / filename
    if shard_path.exists():
        if shard_path.is_symlink() or _file_sha256(shard_path) != file_sha:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "A content-named boundary shard conflicts with staged bytes."
            )
        stage.unlink()
    else:
        os.replace(stage, shard_path)
    descriptors = {
        name: _tensor_descriptor(value) for name, value in sorted(joined.items())
    }
    body = {
        "schema": LFM2_MOVING_GAUGE_BOUNDARY_CHECKPOINT_SCHEMA,
        "request_sha256": request.request_sha256,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "checkpoint_hmac_key_id": authority.key_id,
        "checkpoint_ordinal": checkpoint_ordinal,
        "side": side,
        "side_shard_ordinal": side_shard_ordinal,
        "batch_start": batch_start,
        "batch_stop": batch_stop,
        "previous_checkpoint_hmac_sha256": previous_hmac,
        "source_cache_key": getattr(request, f"{side}_cache_key"),
        "source_selective_session_sha256": request.provenance_sha256s[
            f"{side}_selective_session"
        ],
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
        "source_storage_dtype_preserved": True,
        "unselected_tensor_payloads_loaded": False,
        "execution": dict(_ZERO_EXECUTION),
    }
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    checkpoint = {
        **sealed,
        "checkpoint_hmac_sha256": authority.sign_checkpoint(sealed),
    }
    return checkpoint


def _verify_source_session_complete(session: Any) -> None:
    verify = getattr(session, "verify", None)
    if not callable(verify) or verify() is not session:
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Selective source session did not verify before checkpoint admission."
        )


def _discard_unverified_session_drafts(
    work_root: Path,
    *,
    drafts: Sequence[Mapping[str, Any]],
) -> None:
    """Remove orphan shards produced by one unverified source session."""

    for draft in drafts:
        filename = draft.get("shard", {}).get("filename")
        if isinstance(filename, str) and Path(filename).name == filename:
            (work_root / "shards" / filename).unlink(missing_ok=True)


def _extract_missing_prefix(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    sources: Lfm2MovingGaugeBoundaryExtractionSources,
    work_root: Path,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
    checkpoints: tuple[dict[str, Any], ...],
    *,
    maximum_new_shards: int | None,
) -> tuple[dict[str, Any], ...]:
    parent_manifest = _verify_cache_binding(
        request, sources.parent_cache, side="parent"
    )
    child_manifest = _verify_cache_binding(request, sources.child_cache, side="child")
    records, basis_split_by_batch = _verify_adapter_and_inputs(
        request, sources, parent_manifest, child_manifest
    )
    total = _checkpoint_count(request) * 2
    completed = list(checkpoints)
    added = 0
    while len(completed) < total:
        if maximum_new_shards is not None and added >= maximum_new_shards:
            break
        next_side, _next_side_ordinal, _start, _stop = _checkpoint_side_and_range(
            request, len(completed)
        )
        factory = getattr(sources, f"{next_side}_session_factory")
        session_drafts: list[dict[str, Any]] = []
        try:
            with factory() as session:
                all_rows, names_by_batch, phases_by_batch = _session_plan(
                    request,
                    session,
                    records,
                    basis_split_by_batch,
                    side=next_side,
                )
                per_side = _checkpoint_count(request)
                side_limit = per_side if next_side == "parent" else total
                while len(completed) + len(session_drafts) < side_limit:
                    if (
                        maximum_new_shards is not None
                        and added + len(session_drafts) >= maximum_new_shards
                    ):
                        break
                    next_ordinal = len(completed) + len(session_drafts)
                    side, side_ordinal, start, stop = _checkpoint_side_and_range(
                        request, next_ordinal
                    )
                    if side != next_side:
                        break
                    if side == "child":
                        parent_checkpoint = completed[side_ordinal]
                        expected_rows = _row_range_for_batches(all_rows, start, stop)
                        if parent_checkpoint["rows"] != list(expected_rows):
                            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                                "Parent and child aligned row indices differ."
                            )
                    previous = (
                        _ZERO_SHA256
                        if not completed and not session_drafts
                        else (
                            session_drafts[-1]["checkpoint_hmac_sha256"]
                            if session_drafts
                            else completed[-1]["checkpoint_hmac_sha256"]
                        )
                    )
                    checkpoint = _extract_one_shard(
                        request,
                        session,
                        records,
                        all_rows,
                        names_by_batch,
                        phases_by_batch,
                        side=side,
                        checkpoint_ordinal=next_ordinal,
                        side_shard_ordinal=side_ordinal,
                        batch_start=start,
                        batch_stop=stop,
                        previous_hmac=previous,
                        work_root=work_root,
                        authority=authority,
                    )
                    session_drafts.append(checkpoint)
                _verify_source_session_complete(session)
        except BaseException:
            _discard_unverified_session_drafts(
                work_root,
                drafts=session_drafts,
            )
            raise
        for checkpoint in session_drafts:
            _write_json(
                _checkpoint_path(work_root, int(checkpoint["checkpoint_ordinal"])),
                checkpoint,
            )
            completed.append(checkpoint)
        added += len(session_drafts)
        if maximum_new_shards is not None and added >= maximum_new_shards:
            break
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


def _source_evidence_sha256(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    checkpoints: Sequence[Mapping[str, Any]],
    side: str,
) -> str:
    selected = [value for value in checkpoints if value["side"] == side]
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-moving-gauge-boundary-source-evidence-v1",
            "side": side,
            "cache_key": getattr(request, f"{side}_cache_key"),
            "manifest_sha256": request.provenance_sha256s[f"{side}_manifest"],
            "completion_sha256": request.provenance_sha256s[f"{side}_completion"],
            "identity_sha256": request.provenance_sha256s[f"{side}_identity"],
            "selective_session_sha256": request.provenance_sha256s[
                f"{side}_selective_session"
            ],
            "ordered_checkpoint_hmac_sha256s": [
                value["checkpoint_hmac_sha256"] for value in selected
            ],
            "ordered_shard_file_sha256s": [
                value["shard"]["file_sha256"] for value in selected
            ],
            "execution": dict(_ZERO_EXECUTION),
        }
    )


def _complete_phase_documents(
    request: Lfm2MovingGaugeBoundaryImportRequest,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
    checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reconstruct every derived completed-phase document exactly."""

    total = _checkpoint_count(request) * 2
    if len(checkpoints) != total:
        raise Lfm2MovingGaugeBoundaryPhaseError(
            "Cannot construct documents for an incomplete boundary checkpoint chain."
        )
    if any(
        value.get("source_storage_dtype_preserved") is not True
        or value.get("unselected_tensor_payloads_loaded") is not False
        or value.get("source_cache_key")
        != getattr(request, f"{value.get('side')}_cache_key", None)
        or value.get("source_selective_session_sha256")
        != request.provenance_sha256s.get(f"{value.get('side')}_selective_session")
        or value.get("execution") != _ZERO_EXECUTION
        for value in checkpoints
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Completed boundary claims differ from their checkpoint sources."
        )
    per_side = _checkpoint_count(request)
    parent_rows = [row for value in checkpoints[:per_side] for row in value["rows"]]
    child_rows = [row for value in checkpoints[per_side:] for row in value["rows"]]
    if (
        parent_rows != child_rows
        or len(parent_rows) != request.expected_aligned_row_count
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Complete parent and child row indices do not match."
        )
    row_body = {
        "schema": LFM2_MOVING_GAUGE_BOUNDARY_ROW_INDEX_SCHEMA,
        "request_sha256": request.request_sha256,
        "row_count": len(parent_rows),
        "rows": parent_rows,
        "basis_split_is_chart_authority": True,
        "bridge_split_is_provenance_only": True,
    }
    row_index = {**row_body, "row_index_sha256": canonical_sha256(row_body)}
    row_alignment_sha256 = canonical_sha256(
        {
            "request_sha256": request.request_sha256,
            "bridge_trace_adapter_sha256": request.provenance_sha256s[
                "bridge_trace_adapter"
            ],
            "parent_cache_key": request.parent_cache_key,
            "child_cache_key": request.child_cache_key,
            "ordered_rows": parent_rows,
        }
    )
    split_identity_sha256 = canonical_sha256(
        [
            {
                "global_row_index": row["global_row_index"],
                "record_ordinal": row["record_ordinal"],
                "bridge_split": row["bridge_split"],
                "basis_split": row["basis_split"],
            }
            for row in parent_rows
        ]
    )
    parent_evidence = _source_evidence_sha256(request, checkpoints, "parent")
    child_evidence = _source_evidence_sha256(request, checkpoints, "child")
    checkpoint_summaries = [
        {
            "checkpoint_ordinal": value["checkpoint_ordinal"],
            "side": value["side"],
            "side_shard_ordinal": value["side_shard_ordinal"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "checkpoint_hmac_sha256": value["checkpoint_hmac_sha256"],
            "shard": value["shard"],
            "row_start": value["row_start"],
            "row_stop": value["row_stop"],
        }
        for value in checkpoints
    ]
    receipt_body = {
        "schema": LFM2_MOVING_GAUGE_BOUNDARY_RECEIPT_SCHEMA,
        "request": request.as_dict(),
        "phase_key": request.phase_key.as_dict(),
        "checkpoint_hmac_key_id": authority.key_id,
        "checkpoint_chain_sha256": _checkpoint_chain_sha256(checkpoints),
        "ordered_checkpoints": checkpoint_summaries,
        "row_index_sha256": row_index["row_index_sha256"],
        "row_alignment_sha256": row_alignment_sha256,
        "split_identity_sha256": split_identity_sha256,
        "cached_parent_evidence_sha256": parent_evidence,
        "live_child_recapture_sha256": child_evidence,
        "source_storage_dtype_preserved": True,
        "physical_writer_output_layers": list(request.focused_writer_layer_indices),
        "base_child_post_operator_is_not_conditioned_recapture": True,
        "numeric_token_id_equality_required": False,
        "unselected_tensor_payloads_loaded": False,
        "tensor_bytes_authenticated_lazily_on_field_access": True,
        "execution": dict(_ZERO_EXECUTION),
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    anchor_body = {
        "schema": LFM2_MOVING_GAUGE_BOUNDARY_ANCHOR_SCHEMA,
        "algorithm": "hmac-sha256",
        "key_id": authority.key_id,
        "phase_name": LFM2_MOVING_GAUGE_BOUNDARY_PHASE,
        "phase_key_sha256": request.phase_key.phase_key_sha256,
        "request_sha256": request.request_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "row_index_sha256": row_index["row_index_sha256"],
        "checkpoint_chain_sha256": receipt["checkpoint_chain_sha256"],
        "ordered_checkpoint_hmac_sha256s": [
            value["checkpoint_hmac_sha256"] for value in checkpoints
        ],
        "ordered_shard_file_sha256s": [
            value["shard"]["file_sha256"] for value in checkpoints
        ],
        "cached_parent_evidence_sha256": parent_evidence,
        "live_child_recapture_sha256": child_evidence,
    }
    anchor = {
        **anchor_body,
        "hmac_sha256": authority.sign_anchor(anchor_body),
    }
    result = {
        "receipt_sha256": receipt["receipt_sha256"],
        "row_index_sha256": row_index["row_index_sha256"],
        "row_alignment_sha256": row_alignment_sha256,
        "split_identity_sha256": split_identity_sha256,
        "checkpoint_chain_sha256": receipt["checkpoint_chain_sha256"],
        "source_anchor_hmac_sha256": anchor["hmac_sha256"],
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
    }
    return row_index, receipt, anchor, result


def _publish_complete_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2MovingGaugeBoundaryImportRequest,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
    work_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    events: FoldPhaseEventLog | None,
) -> VerifiedFoldPhaseCacheEntry:
    row_index, receipt, anchor, result = _complete_phase_documents(
        request,
        authority,
        checkpoints,
    )
    try:
        transaction = cache.begin(request.phase_key, events=events)
    except FoldPhaseCacheAlreadyCompleteError:
        existing = cache.lookup(request.phase_key, events=events)
        if existing is None:  # pragma: no cover - concurrent deletion defense
            raise FoldPhaseCacheIntegrityError(
                "Concurrent boundary phase publication disappeared."
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
            destination = payload / "shards" / filename
            if not destination.exists():
                shutil.copyfile(work_root / "shards" / filename, destination)
        return transaction.complete(result=result)


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2MovingGaugeBoundaryPhase:
    """Authenticated compact boundary shards with lazy field access."""

    entry: VerifiedFoldPhaseCacheEntry = field(repr=False, compare=False)
    request: Lfm2MovingGaugeBoundaryImportRequest
    receipt: Mapping[str, Any]
    row_index: Mapping[str, Any]
    source_anchor: Mapping[str, Any]
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority = field(
        repr=False,
        compare=False,
    )

    @property
    def row_alignment_sha256(self) -> str:
        return str(self.receipt["row_alignment_sha256"])

    @property
    def split_identity_sha256(self) -> str:
        return str(self.receipt["split_identity_sha256"])

    @property
    def cached_parent_evidence_sha256(self) -> str:
        return str(self.receipt["cached_parent_evidence_sha256"])

    @property
    def live_child_recapture_sha256(self) -> str:
        return str(self.receipt["live_child_recapture_sha256"])

    @property
    def receipt_sha256(self) -> str:
        return str(self.receipt["receipt_sha256"])

    def verify_authentication(self) -> bool:
        try:
            if not self.request.verify_authentication():
                return False
            loaded = load_existing_lfm2_moving_gauge_boundary_phase(
                DynamicFoldPhaseCache(self.entry.path.parents[1]),
                self.request,
                authority=self.authority,
            )
            return bool(
                loaded.entry.path == self.entry.path
                and loaded.entry.payload_path == self.entry.payload_path
                and loaded.entry.completion_sha256 == self.entry.completion_sha256
                and loaded.entry.phase_key.as_dict() == self.entry.phase_key.as_dict()
                and _canonical(dict(loaded.entry.result), "loaded phase result")
                == _canonical(dict(self.entry.result), "current phase result")
                and _canonical(dict(loaded.receipt), "loaded receipt")
                == _canonical(dict(self.receipt), "current receipt")
                and _canonical(dict(loaded.row_index), "loaded row index")
                == _canonical(dict(self.row_index), "current row index")
                and _canonical(dict(loaded.source_anchor), "loaded source anchor")
                == _canonical(dict(self.source_anchor), "current source anchor")
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    def _rows_for_split(self, split: str) -> tuple[dict[str, Any], ...]:
        if split not in {"fit", "heldout", "all"}:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                "Boundary split must be fit, heldout, or all."
            )
        rows = tuple(dict(row) for row in self.row_index["rows"])
        return (
            rows
            if split == "all"
            else tuple(row for row in rows if row["basis_split"] == split)
        )

    def _load_field(
        self,
        side: str,
        layer_index: int,
        field_name: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if side not in _SIDES:
            raise Lfm2MovingGaugeBoundaryPhaseError("side must be parent or child.")
        key = _normalized_tensor_name(layer_index, field_name)
        selected_global = {int(row["global_row_index"]) for row in rows}
        pieces: list[torch.Tensor] = []
        from safetensors import safe_open

        for summary in self.receipt["ordered_checkpoints"]:
            if summary["side"] != side:
                continue
            start = int(summary["row_start"])
            stop = int(summary["row_stop"])
            local = [
                index - start
                for index in range(start, stop)
                if index in selected_global
            ]
            if not local:
                continue
            descriptor = summary["shard"]["tensor_descriptors"].get(key)
            if not isinstance(descriptor, Mapping):
                raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                    f"Boundary field {key!r} is absent from {side} shard."
                )
            path = self.entry.payload_path / "shards" / summary["shard"]["filename"]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata != {
                    "schema": LFM2_MOVING_GAUGE_BOUNDARY_SHARD_SCHEMA,
                    "request_sha256": self.request.request_sha256,
                    "side": side,
                    "side_shard_ordinal": str(summary["side_shard_ordinal"]),
                }:
                    raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                        "Boundary shard safetensors metadata changed."
                    )
                tensor = handle.get_tensor(key).detach().to(device="cpu").contiguous()
            if _canonical(
                _tensor_descriptor(tensor),
                "loaded boundary tensor descriptor",
            ) != _canonical(dict(descriptor), "authenticated tensor descriptor"):
                raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                    f"Boundary tensor {key!r} changed."
                )
            pieces.append(
                tensor.index_select(
                    0, torch.tensor(local, dtype=torch.long)
                ).contiguous()
            )
        if not pieces:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                f"Boundary selection for {side}.{key} is empty."
            )
        result = torch.cat(pieces, dim=0).contiguous()
        if int(result.shape[0]) != len(rows):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Boundary shard selection changed aligned row count."
            )
        return result

    def field_pair(
        self,
        layer_index: int,
        field_name: str,
        *,
        split: str,
    ) -> Lfm2MovingGaugeBoundaryPair:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index >= len(self.request.layer_bindings)
        ):
            raise Lfm2MovingGaugeBoundaryPhaseError("layer_index is out of range.")
        allowed = set(_RESIDUAL_FIELDS)
        if layer_index in self.request.focused_writer_layer_indices:
            allowed.update(_FOCUSED_WRITER_FIELDS)
        if field_name not in allowed:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                f"Field {field_name!r} was not authenticated for layer {layer_index}."
            )
        rows = self._rows_for_split(split)
        record_ids = tuple(str(row["record_id"]) for row in rows)
        indices = tuple(int(row["global_row_index"]) for row in rows)
        return Lfm2MovingGaugeBoundaryPair(
            layer_index=layer_index,
            endpoint=field_name,
            split=split,
            parent_rows=self._load_field("parent", layer_index, field_name, rows),
            child_rows=self._load_field("child", layer_index, field_name, rows),
            record_ids=record_ids,
            global_row_indices=indices,
            weights=equal_record_mass_weights(record_ids),
            row_alignment_sha256=self.row_alignment_sha256,
            split_identity_sha256=self.split_identity_sha256,
        )

    def boundary_pair(
        self,
        layer_index: int,
        endpoint: str,
        *,
        split: str,
    ) -> Lfm2MovingGaugeBoundaryPair:
        normalized = {
            "pre": "residual.pre",
            "post_operator": "residual.post_operator",
            "post_layer": "residual.post_layer",
            "residual.pre": "residual.pre",
            "residual.post_operator": "residual.post_operator",
            "residual.post_layer": "residual.post_layer",
        }.get(endpoint)
        if normalized is None:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                "endpoint must be pre, post_operator, or post_layer."
            )
        return self.field_pair(layer_index, normalized, split=split)

    def focused_writer_outputs(
        self,
        layer_index: int,
        *,
        split: str,
    ) -> Lfm2MovingGaugeFocusedWriterOutputs:
        if layer_index not in self.request.focused_writer_layer_indices:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                f"Layer {layer_index} has no focused physical writer extension."
            )
        operator = self.field_pair(layer_index, "operator.output", split=split)
        mlp = self.field_pair(layer_index, "mlp.down", split=split)
        if (
            operator.record_ids != mlp.record_ids
            or operator.global_row_indices != mlp.global_row_indices
        ):
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Focused operator and MLP writer rows are not aligned."
            )
        return Lfm2MovingGaugeFocusedWriterOutputs(
            layer_index=layer_index,
            split=split,
            parent_operator_output=operator.parent_rows,
            parent_mlp_down=mlp.parent_rows,
            child_operator_output=operator.child_rows,
            child_mlp_down=mlp.child_rows,
            record_ids=operator.record_ids,
            global_row_indices=operator.global_row_indices,
        )

    def build_boundary_evidence(
        self,
        *,
        layer_index: int,
        endpoint: str,
        boundary_id: str,
        interface_role: str,
        interface_locked: bool,
        base_encoder: Any,
    ) -> Lfm2MovingGaugeBoundaryEvidence:
        normalized_endpoint = {
            "pre": "residual.pre",
            "post_operator": "residual.post_operator",
            "post_layer": "residual.post_layer",
            "residual.pre": "residual.pre",
            "residual.post_operator": "residual.post_operator",
            "residual.post_layer": "residual.post_layer",
        }.get(endpoint)
        last_layer = len(self.request.layer_bindings) - 1
        if interface_role == "input":
            valid_role = (
                layer_index == 0
                and normalized_endpoint == "residual.pre"
                and interface_locked is True
            )
            expected_boundary_id = "layer.000.residual.pre"
        elif interface_role == "terminal":
            valid_role = (
                layer_index == last_layer
                and normalized_endpoint == "residual.post_layer"
                and interface_locked is True
            )
            expected_boundary_id = "stack.terminal.residual"
        elif interface_role == "intermediate":
            valid_role = bool(
                normalized_endpoint is not None
                and interface_locked is False
                and not (layer_index == 0 and normalized_endpoint == "residual.pre")
                and not (
                    layer_index == last_layer
                    and normalized_endpoint == "residual.post_layer"
                )
            )
            expected_boundary_id = (
                ""
                if normalized_endpoint is None
                else f"layer.{layer_index:03d}.{normalized_endpoint}"
            )
        else:
            valid_role = False
            expected_boundary_id = ""
        if not valid_role or boundary_id != expected_boundary_id:
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Boundary evidence role, layer, endpoint, lock, or identifier is invalid."
            )
        fit = self.boundary_pair(layer_index, endpoint, split="fit")
        heldout = self.boundary_pair(layer_index, endpoint, split="heldout")
        evidence = Lfm2MovingGaugeBoundaryEvidence(
            boundary_id=boundary_id,
            interface_role=interface_role,
            interface_locked=interface_locked,
            base_encoder=base_encoder,
            fit_parent_rows=fit.parent_rows,
            fit_child_rows=fit.child_rows,
            fit_weights=fit.weights,
            fit_provenance=f"{self.receipt_sha256}:basis-fit:{endpoint}:{layer_index}",
            row_alignment_sha256=self.row_alignment_sha256,
            cached_parent_evidence_sha256=self.cached_parent_evidence_sha256,
            live_child_recapture_sha256=self.live_child_recapture_sha256,
            heldout_parent_rows=heldout.parent_rows,
            heldout_child_rows=heldout.child_rows,
            heldout_weights=heldout.weights,
            heldout_provenance=(
                f"{self.receipt_sha256}:basis-heldout:{endpoint}:{layer_index}"
            ),
        )
        if not evidence.verify_authentication():
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Constructed boundary evidence failed self-authentication."
            )
        return evidence


def _authenticate_complete_entry(
    entry: VerifiedFoldPhaseCacheEntry,
    request: Lfm2MovingGaugeBoundaryImportRequest,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
) -> AuthenticatedLfm2MovingGaugeBoundaryPhase:
    if not request.verify_authentication():
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary import request authentication failed."
        )
    payload = entry.payload_path
    if _read_json(payload / "request.json") != request.as_dict():
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Completed boundary phase request differs from its lookup key."
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
    expected_row_index, expected_receipt, expected_anchor, expected_result = (
        _complete_phase_documents(request, authority, checkpoints)
    )
    if (
        len(checkpoints) != _checkpoint_count(request) * 2
        or row_index != expected_row_index
        or receipt != expected_receipt
        or anchor != expected_anchor
        or _canonical(dict(entry.result), "phase result") != expected_result
        or entry.phase_key.as_dict() != request.phase_key.as_dict()
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Completed boundary documents differ from exact checkpoint reconstruction."
        )
    return AuthenticatedLfm2MovingGaugeBoundaryPhase(
        entry=entry,
        request=request,
        receipt=_deep_freeze(receipt),
        row_index=_deep_freeze(row_index),
        source_anchor=_deep_freeze(anchor),
        authority=authority,
    )


def load_existing_lfm2_moving_gauge_boundary_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2MovingGaugeBoundaryImportRequest,
    *,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
    events: FoldPhaseEventLog | None = None,
) -> AuthenticatedLfm2MovingGaugeBoundaryPhase:
    """Load one complete HMAC-authenticated compact boundary phase."""

    if not isinstance(cache, DynamicFoldPhaseCache):
        raise TypeError("cache must be DynamicFoldPhaseCache.")
    if not isinstance(request, Lfm2MovingGaugeBoundaryImportRequest):
        raise TypeError("request must be Lfm2MovingGaugeBoundaryImportRequest.")
    if not request.verify_authentication():
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary import request authentication failed."
        )
    if (
        not isinstance(authority, Lfm2MovingGaugeBoundaryPhaseAuthority)
        or authority.key_id != request.checkpoint_hmac_key_id
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary phase authority differs from the request."
        )
    entry = cache.lookup(request.phase_key, events=events)
    if entry is None:
        raise Lfm2MovingGaugeBoundaryPhaseError(
            "Requested moving-gauge boundary phase is not complete."
        )
    return _authenticate_complete_entry(entry, request, authority)


def load_or_extract_lfm2_moving_gauge_boundary_phase(
    cache: DynamicFoldPhaseCache,
    request: Lfm2MovingGaugeBoundaryImportRequest,
    *,
    authority: Lfm2MovingGaugeBoundaryPhaseAuthority,
    checkpoint_root: str | os.PathLike[str],
    source_provider: (
        Callable[[], Lfm2MovingGaugeBoundaryExtractionSources] | None
    ) = None,
    maximum_new_shards: int | None = None,
    events: FoldPhaseEventLog | None = None,
) -> (
    AuthenticatedLfm2MovingGaugeBoundaryPhase
    | Lfm2MovingGaugeBoundaryExtractionProgress
):
    """Load a hit or resume a zero-forward source import.

    The final phase lookup occurs before ``source_provider`` is inspected.  A
    cache hit therefore proves zero source-cache access.  On a miss, valid HMAC
    checkpoints are reused and only the missing canonical suffix is extracted.
    """

    if not isinstance(cache, DynamicFoldPhaseCache):
        raise TypeError("cache must be DynamicFoldPhaseCache.")
    if not isinstance(request, Lfm2MovingGaugeBoundaryImportRequest):
        raise TypeError("request must be Lfm2MovingGaugeBoundaryImportRequest.")
    if not request.verify_authentication():
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary import request authentication failed."
        )
    if (
        not isinstance(authority, Lfm2MovingGaugeBoundaryPhaseAuthority)
        or authority.key_id != request.checkpoint_hmac_key_id
    ):
        raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
            "Boundary phase authority differs from the request."
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
            raise Lfm2MovingGaugeBoundaryPhaseAuthenticationError(
                "Boundary checkpoint root belongs to a different request."
            )
    else:
        _write_json(request_path, request.as_dict())
    checkpoints = _load_checkpoint_prefix(
        root,
        request=request,
        authority=authority,
    )
    total = _checkpoint_count(request) * 2
    if len(checkpoints) < total:
        if source_provider is None:
            raise Lfm2MovingGaugeBoundaryPhaseError(
                "Incomplete boundary import requires a deferred source provider."
            )
        sources = source_provider()
        if not isinstance(sources, Lfm2MovingGaugeBoundaryExtractionSources):
            raise TypeError(
                "source_provider must return Lfm2MovingGaugeBoundaryExtractionSources."
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
        return Lfm2MovingGaugeBoundaryExtractionProgress(
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
    "AuthenticatedLfm2MovingGaugeBoundaryPhase",
    "LFM2_MOVING_GAUGE_BOUNDARY_ANCHOR_SCHEMA",
    "LFM2_MOVING_GAUGE_BOUNDARY_CHECKPOINT_SCHEMA",
    "LFM2_MOVING_GAUGE_BOUNDARY_PHASE",
    "LFM2_MOVING_GAUGE_BOUNDARY_RECEIPT_SCHEMA",
    "LFM2_MOVING_GAUGE_BOUNDARY_REQUEST_SCHEMA",
    "LFM2_MOVING_GAUGE_BOUNDARY_ROW_INDEX_SCHEMA",
    "Lfm2MovingGaugeBoundaryExtractionProgress",
    "Lfm2MovingGaugeBoundaryExtractionSources",
    "Lfm2MovingGaugeBoundaryImportRequest",
    "Lfm2MovingGaugeBoundaryLayerBinding",
    "Lfm2MovingGaugeBoundaryPair",
    "Lfm2MovingGaugeBoundaryPhaseAuthenticationError",
    "Lfm2MovingGaugeBoundaryPhaseAuthority",
    "Lfm2MovingGaugeBoundaryPhaseError",
    "Lfm2MovingGaugeFocusedWriterOutputs",
    "fingerprint_lfm2_moving_gauge_basis_split",
    "lfm2_moving_gauge_boundary_implementation_sha256",
    "load_existing_lfm2_moving_gauge_boundary_phase",
    "load_or_extract_lfm2_moving_gauge_boundary_phase",
]
