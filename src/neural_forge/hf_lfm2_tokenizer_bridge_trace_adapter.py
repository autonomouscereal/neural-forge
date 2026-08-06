"""Strict cross-tokenizer LFM2 trace and logit-projection evidence.

This module is the sequence-level boundary between a sealed one-row tokenizer
bridge and the existing LFM2 parent-cache/child-replay machinery.  It performs
no model forward, optimization, pruning, quantization, or parameter mutation.

The important distinction is explicit:

* a child record contains target-tokenizer IDs;
* the corresponding parent-cache record contains the bridge-selected donor
  IDs; and
* the two ID tensors are authenticated by the row map, not by false numeric-ID
  equality.

Masks, positions, batch/sequence geometry, split labels, and forced-decode
boundaries remain unchanged.  Full donor-vocabulary cached logits can then be
projected to every physical target vocabulary column before target-space top-k
and margin evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    StrictTokenizerBridgeError,
    strict_tokenizer_bridge_target_package_fingerprint,
)
from neural_forge.hf_lfm2_sealed_evidence_set import (
    SEALED_EVIDENCE_SESSION_TYPES,
)


TRACE_ADAPTER_SCHEMA = "neural-forge-lfm2-strict-tokenizer-trace-adapter-v1"
ALIGNED_RECORD_SCHEMA = "neural-forge-lfm2-strict-tokenizer-aligned-record-v1"
PAIRED_INPUT_SCHEMA = "neural-forge-lfm2-strict-tokenizer-paired-input-v1"
INPUT_PAIR_LEDGER_SCHEMA = (
    "neural-forge-lfm2-strict-tokenizer-input-pair-ledger-v1"
)
LOGIT_PROJECTION_SCHEMA = (
    "neural-forge-lfm2-strict-tokenizer-logit-projection-v1"
)

_CACHE_INPUT_DOMAIN = b"neural-forge-lfm2-circuit-input-v1\0"
_CACHE_TENSOR_DOMAIN = b"neural-forge-lfm2-circuit-tensor-v1\0"
_FORCED_IDS = "forced_decode_ids"
_BATCH_KEYS = frozenset(
    {"input_ids", "attention_mask", "position_ids", _FORCED_IDS}
)
_PROFILE_RESOLUTIONS = frozenset(
    {
        "direct-tokenizer-identity",
        "bytelevel-bpe-token-string",
        "decoded-token-surface",
    }
)
_ROW_RESOLUTIONS = frozenset(
    {
        "direct-tokenizer-identity",
        "bytelevel-bpe-token-string",
        "special-token-role",
        "exact-token",
    }
)
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERIFIED_SESSION_CONSTRUCTION_TOKEN = object()


class Lfm2StrictTokenizerBridgeTraceError(ValueError):
    """The bridge trace or projection cannot be authenticated."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Strict LFM2 tokenizer-bridge traces require PyTorch."
        ) from exc
    return torch


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
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Bridge trace metadata must be finite canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} must be nonempty text."
        )
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} must be a positive integer."
        )
    return result


def _is_integer_tensor(torch: Any, value: Any) -> bool:
    return torch.is_tensor(value) and value.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _plain_integer_tensor(
    value: Any,
    name: str,
    *,
    rank: int = 2,
) -> Any:
    torch = _torch()
    if (
        not _is_integer_tensor(torch, value)
        or value.ndim != rank
        or value.numel() == 0
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} must be a nonempty plain rank-{rank} integer tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.long).contiguous().clone()
    if bool((result < 0).any().item()):
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} contains a negative value."
        )
    return result


def _attention_mask(value: Any, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not torch.is_tensor(value):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "attention_mask must be a rank-2 boolean or integer tensor."
        )
    if value.dtype == torch.bool:
        normalized = value.detach().to(
            device="cpu",
            dtype=torch.long,
        ).contiguous().clone()
    else:
        normalized = _plain_integer_tensor(value, "attention_mask")
    if tuple(normalized.shape) != shape:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "attention_mask must match input_ids shape."
        )
    if not bool(((normalized == 0) | (normalized == 1)).all().item()):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "attention_mask must contain only zero and one."
        )
    if bool((normalized.sum(dim=1) == 0).any().item()):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Every sequence boundary must contain at least one active token."
        )
    return normalized


def _position_ids(value: Any | None, shape: tuple[int, int]) -> Any | None:
    if value is None:
        return None
    result = _plain_integer_tensor(value, "position_ids")
    if tuple(result.shape) != shape:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "position_ids must match input_ids shape."
        )
    return result


def _tensor_bytes(value: Any) -> bytes:
    torch = _torch()
    return (
        value.detach()
        .to(device="cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def _tensor_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_CACHE_TENSOR_DOMAIN)
    digest.update(str(value.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in value.shape]))
    digest.update(b"\0")
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype).replace("torch.", ""),
        "shape": [int(size) for size in value.shape],
        "numel": int(value.numel()),
        "sha256": _tensor_sha256(value),
    }


def _cache_input_sha256(
    *,
    input_ids: Any,
    attention_mask: Any,
    forced_decode_ids: Any,
    position_ids: Any | None,
) -> str:
    named = {
        "attention_mask": attention_mask,
        _FORCED_IDS: forced_decode_ids,
        "input_ids": input_ids,
    }
    if position_ids is not None:
        named["position_ids"] = position_ids
    digest = hashlib.sha256()
    digest.update(_CACHE_INPUT_DOMAIN)
    for name in sorted(named):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(named[name]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _clone_batch(
    *,
    input_ids: Any,
    attention_mask: Any,
    forced_decode_ids: Any,
    position_ids: Any | None,
) -> dict[str, Any]:
    result = {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
        _FORCED_IDS: forced_decode_ids.clone(),
    }
    if position_ids is not None:
        result["position_ids"] = position_ids.clone()
    return result


@dataclass(frozen=True, slots=True, init=False)
class Lfm2TargetTokenizedRecord:
    """One detached target-tokenized prompt and forced-decode trajectory."""

    record_id: str
    split: str
    _input_ids: Any
    _attention_mask: Any
    _forced_decode_ids: Any
    _position_ids: Any | None

    def __init__(
        self,
        *,
        record_id: str,
        split: str,
        input_ids: Any,
        attention_mask: Any,
        forced_decode_ids: Any,
        position_ids: Any | None = None,
    ) -> None:
        normalized_input = _plain_integer_tensor(input_ids, "input_ids")
        shape = tuple(int(size) for size in normalized_input.shape)
        normalized_mask = _attention_mask(attention_mask, shape)
        normalized_positions = _position_ids(position_ids, shape)
        normalized_forced = _plain_integer_tensor(
            forced_decode_ids,
            _FORCED_IDS,
        )
        if int(normalized_forced.shape[0]) != shape[0]:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "forced_decode_ids must preserve the input batch boundary."
            )
        object.__setattr__(self, "record_id", _text(record_id, "record_id"))
        object.__setattr__(self, "split", _text(split, "split"))
        object.__setattr__(self, "_input_ids", normalized_input)
        object.__setattr__(self, "_attention_mask", normalized_mask)
        object.__setattr__(self, "_forced_decode_ids", normalized_forced)
        object.__setattr__(self, "_position_ids", normalized_positions)

    @property
    def input_ids(self) -> Any:
        return self._input_ids.clone()

    @property
    def attention_mask(self) -> Any:
        return self._attention_mask.clone()

    @property
    def forced_decode_ids(self) -> Any:
        return self._forced_decode_ids.clone()

    @property
    def position_ids(self) -> Any | None:
        return None if self._position_ids is None else self._position_ids.clone()

    @property
    def batch_size(self) -> int:
        return int(self._input_ids.shape[0])

    @property
    def prompt_width(self) -> int:
        return int(self._input_ids.shape[1])

    @property
    def forced_decode_steps(self) -> int:
        return int(self._forced_decode_ids.shape[1])

    def as_child_replay_batch(self) -> dict[str, Any]:
        return _clone_batch(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask,
            forced_decode_ids=self._forced_decode_ids,
            position_ids=self._position_ids,
        )


def _verified_bridge(
    bridge: Any,
    expected_artifact_sha256: str | None,
) -> StrictTokenizerBridgeArtifact:
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "bridge must be a verified StrictTokenizerBridgeArtifact."
        )
    try:
        bridge.verify_files()
    except StrictTokenizerBridgeError as exc:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Strict tokenizer bridge file authentication failed."
        ) from exc
    if (
        expected_artifact_sha256 is not None
        and bridge.artifact_sha256
        != _sha256(
            expected_artifact_sha256,
            "expected_artifact_sha256",
        )
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Strict tokenizer bridge differs from the expected artifact."
        )
    profile = bridge.transport_profile
    if profile.alignment_resolution not in _PROFILE_RESOLUTIONS:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Strict tokenizer bridge uses an unsupported or span-only "
            "alignment resolution."
        )
    for row in profile.row_mappings:
        if row.resolution not in _ROW_RESOLUTIONS:
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Target row {row.target_row_index} is not authenticated by "
                "a one-row donor resolution."
            )
    return bridge


def _ids_within_capacity(value: Any, capacity: int, name: str) -> None:
    if bool((value >= capacity).any().item()):
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} contains an ID outside target row capacity {capacity}."
        )


def _map_target_ids(value: Any, mapping: tuple[int, ...]) -> Any:
    torch = _torch()
    index = torch.tensor(mapping, dtype=torch.long)
    return index[value].contiguous().clone()


def _special_role_counts(
    target_input_ids: Any,
    target_forced_decode_ids: Any,
    bridge: StrictTokenizerBridgeArtifact,
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    all_ids = _torch().cat(
        (target_input_ids.reshape(-1), target_forced_decode_ids.reshape(-1))
    )
    for row in bridge.transport_profile.row_mappings:
        if not row.special_token_roles:
            continue
        occurrences = int((all_ids == row.target_row_index).sum().item())
        for role in row.special_token_roles:
            counts[role] = counts.get(role, 0) + occurrences
    return tuple(sorted(counts.items()))


def _aligned_record_body(
    *,
    ordinal: int,
    record_id: str,
    split: str,
    bridge: StrictTokenizerBridgeArtifact,
    target_input_ids: Any,
    donor_input_ids: Any,
    attention_mask: Any,
    target_forced_decode_ids: Any,
    donor_forced_decode_ids: Any,
    position_ids: Any | None,
    target_cache_input_sha256: str,
    donor_cache_input_sha256: str,
    numerically_different_token_count: int,
    physical_padding_token_count: int,
    special_token_role_counts: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    position_descriptor = (
        None if position_ids is None else _tensor_descriptor(position_ids)
    )
    return {
        "schema": ALIGNED_RECORD_SCHEMA,
        "ordinal": ordinal,
        "record_id": record_id,
        "split": split,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "transport_profile_fingerprint": bridge.profile_fingerprint,
        "mapping_sha256": _canonical_sha256(
            list(bridge.transport_profile.target_to_donor_row_indices)
        ),
        "target_row_capacity": bridge.transport_profile.target_row_capacity,
        "donor_row_capacity": bridge.transport_profile.donor_row_capacity,
        "target_input_ids": _tensor_descriptor(target_input_ids),
        "donor_input_ids": _tensor_descriptor(donor_input_ids),
        "attention_mask": _tensor_descriptor(attention_mask),
        "position_ids": position_descriptor,
        "target_forced_decode_ids": _tensor_descriptor(
            target_forced_decode_ids
        ),
        "donor_forced_decode_ids": _tensor_descriptor(donor_forced_decode_ids),
        "target_cache_input_sha256": target_cache_input_sha256,
        "donor_cache_input_sha256": donor_cache_input_sha256,
        "batch_size": int(target_input_ids.shape[0]),
        "prompt_width": int(target_input_ids.shape[1]),
        "forced_decode_steps": int(target_forced_decode_ids.shape[1]),
        "active_prompt_token_count": int(attention_mask.sum().item()),
        "numerically_different_token_count": numerically_different_token_count,
        "physical_padding_token_count": physical_padding_token_count,
        "special_token_role_counts": [
            [role, count] for role, count in special_token_role_counts
        ],
        "alignment_contract": {
            "target_to_donor_relation": "one-row-map",
            "numeric_token_id_equality_required": False,
            "batch_sequence_geometry_preserved": True,
            "attention_mask_preserved": True,
            "position_ids_preserved": True,
            "forced_decode_boundary_preserved": True,
            "split_preserved": True,
        },
        "model_operations": {
            "parent_model_calls": 0,
            "child_model_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
        },
    }


@dataclass(frozen=True, slots=True, init=False)
class Lfm2BridgeAlignedRecord:
    """Immutable paired target/donor execution record and its evidence seal."""

    ordinal: int
    record_id: str
    split: str
    bridge_artifact_sha256: str
    transport_profile_fingerprint: str
    target_cache_input_sha256: str
    donor_cache_input_sha256: str
    numerically_different_token_count: int
    physical_padding_token_count: int
    special_token_role_counts: tuple[tuple[str, int], ...]
    alignment_sha256: str
    _target_input_ids: Any
    _donor_input_ids: Any
    _attention_mask: Any
    _target_forced_decode_ids: Any
    _donor_forced_decode_ids: Any
    _position_ids: Any | None

    @property
    def target_input_ids(self) -> Any:
        return self._target_input_ids.clone()

    @property
    def donor_input_ids(self) -> Any:
        return self._donor_input_ids.clone()

    @property
    def attention_mask(self) -> Any:
        return self._attention_mask.clone()

    @property
    def target_forced_decode_ids(self) -> Any:
        return self._target_forced_decode_ids.clone()

    @property
    def donor_forced_decode_ids(self) -> Any:
        return self._donor_forced_decode_ids.clone()

    @property
    def position_ids(self) -> Any | None:
        return None if self._position_ids is None else self._position_ids.clone()

    @property
    def batch_size(self) -> int:
        return int(self._target_input_ids.shape[0])

    @property
    def prompt_width(self) -> int:
        return int(self._target_input_ids.shape[1])

    @property
    def forced_decode_steps(self) -> int:
        return int(self._target_forced_decode_ids.shape[1])

    def as_parent_cache_batch(self) -> dict[str, Any]:
        return _clone_batch(
            input_ids=self._donor_input_ids,
            attention_mask=self._attention_mask,
            forced_decode_ids=self._donor_forced_decode_ids,
            position_ids=self._position_ids,
        )

    def as_child_replay_batch(self) -> dict[str, Any]:
        return _clone_batch(
            input_ids=self._target_input_ids,
            attention_mask=self._attention_mask,
            forced_decode_ids=self._target_forced_decode_ids,
            position_ids=self._position_ids,
        )

    def to_evidence_dict(self) -> dict[str, Any]:
        return {
            "schema": ALIGNED_RECORD_SCHEMA,
            "ordinal": self.ordinal,
            "record_id": self.record_id,
            "split": self.split,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "transport_profile_fingerprint": self.transport_profile_fingerprint,
            "target_cache_input_sha256": self.target_cache_input_sha256,
            "donor_cache_input_sha256": self.donor_cache_input_sha256,
            "batch_size": self.batch_size,
            "prompt_width": self.prompt_width,
            "forced_decode_steps": self.forced_decode_steps,
            "numerically_different_token_count": (
                self.numerically_different_token_count
            ),
            "physical_padding_token_count": self.physical_padding_token_count,
            "special_token_role_counts": [
                [role, count] for role, count in self.special_token_role_counts
            ],
            "numeric_token_id_equality_required": False,
            "alignment_sha256": self.alignment_sha256,
        }

    def _verify(self, bridge: StrictTokenizerBridgeArtifact) -> None:
        mapping = bridge.transport_profile.target_to_donor_row_indices
        observed_donor_ids = _map_target_ids(self._target_input_ids, mapping)
        observed_donor_forced = _map_target_ids(
            self._target_forced_decode_ids,
            mapping,
        )
        torch = _torch()
        if not torch.equal(observed_donor_ids, self._donor_input_ids) or not (
            torch.equal(observed_donor_forced, self._donor_forced_decode_ids)
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {self.record_id!r} no longer follows the "
                "strict one-row mapping."
            )
        target_sha = _cache_input_sha256(
            input_ids=self._target_input_ids,
            attention_mask=self._attention_mask,
            forced_decode_ids=self._target_forced_decode_ids,
            position_ids=self._position_ids,
        )
        donor_sha = _cache_input_sha256(
            input_ids=self._donor_input_ids,
            attention_mask=self._attention_mask,
            forced_decode_ids=self._donor_forced_decode_ids,
            position_ids=self._position_ids,
        )
        if (
            target_sha != self.target_cache_input_sha256
            or donor_sha != self.donor_cache_input_sha256
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {self.record_id!r} input hashes changed."
            )
        body = _aligned_record_body(
            ordinal=self.ordinal,
            record_id=self.record_id,
            split=self.split,
            bridge=bridge,
            target_input_ids=self._target_input_ids,
            donor_input_ids=self._donor_input_ids,
            attention_mask=self._attention_mask,
            target_forced_decode_ids=self._target_forced_decode_ids,
            donor_forced_decode_ids=self._donor_forced_decode_ids,
            position_ids=self._position_ids,
            target_cache_input_sha256=target_sha,
            donor_cache_input_sha256=donor_sha,
            numerically_different_token_count=(
                self.numerically_different_token_count
            ),
            physical_padding_token_count=self.physical_padding_token_count,
            special_token_role_counts=self.special_token_role_counts,
        )
        if _canonical_sha256(body) != self.alignment_sha256:
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {self.record_id!r} evidence seal is invalid."
            )


def _build_aligned_record(
    ordinal: int,
    source: Lfm2TargetTokenizedRecord,
    bridge: StrictTokenizerBridgeArtifact,
) -> Lfm2BridgeAlignedRecord:
    profile = bridge.transport_profile
    _ids_within_capacity(
        source._input_ids,
        profile.target_row_capacity,
        "input_ids",
    )
    _ids_within_capacity(
        source._forced_decode_ids,
        profile.target_row_capacity,
        _FORCED_IDS,
    )
    mapping = profile.target_to_donor_row_indices
    donor_ids = _map_target_ids(source._input_ids, mapping)
    donor_forced = _map_target_ids(source._forced_decode_ids, mapping)
    target_sha = _cache_input_sha256(
        input_ids=source._input_ids,
        attention_mask=source._attention_mask,
        forced_decode_ids=source._forced_decode_ids,
        position_ids=source._position_ids,
    )
    donor_sha = _cache_input_sha256(
        input_ids=donor_ids,
        attention_mask=source._attention_mask,
        forced_decode_ids=donor_forced,
        position_ids=source._position_ids,
    )
    different = int((source._input_ids != donor_ids).sum().item()) + int(
        (source._forced_decode_ids != donor_forced).sum().item()
    )
    all_target_ids = _torch().cat(
        (source._input_ids.reshape(-1), source._forced_decode_ids.reshape(-1))
    )
    physical_padding = int(
        (all_target_ids >= profile.target_vocab_size).sum().item()
    )
    role_counts = _special_role_counts(
        source._input_ids,
        source._forced_decode_ids,
        bridge,
    )
    body = _aligned_record_body(
        ordinal=ordinal,
        record_id=source.record_id,
        split=source.split,
        bridge=bridge,
        target_input_ids=source._input_ids,
        donor_input_ids=donor_ids,
        attention_mask=source._attention_mask,
        target_forced_decode_ids=source._forced_decode_ids,
        donor_forced_decode_ids=donor_forced,
        position_ids=source._position_ids,
        target_cache_input_sha256=target_sha,
        donor_cache_input_sha256=donor_sha,
        numerically_different_token_count=different,
        physical_padding_token_count=physical_padding,
        special_token_role_counts=role_counts,
    )
    result = object.__new__(Lfm2BridgeAlignedRecord)
    values = {
        "ordinal": ordinal,
        "record_id": source.record_id,
        "split": source.split,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "transport_profile_fingerprint": bridge.profile_fingerprint,
        "target_cache_input_sha256": target_sha,
        "donor_cache_input_sha256": donor_sha,
        "numerically_different_token_count": different,
        "physical_padding_token_count": physical_padding,
        "special_token_role_counts": role_counts,
        "alignment_sha256": _canonical_sha256(body),
        "_target_input_ids": source._input_ids.clone(),
        "_donor_input_ids": donor_ids.clone(),
        "_attention_mask": source._attention_mask.clone(),
        "_target_forced_decode_ids": source._forced_decode_ids.clone(),
        "_donor_forced_decode_ids": donor_forced.clone(),
        "_position_ids": (
            None if source._position_ids is None else source._position_ids.clone()
        ),
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    result._verify(bridge)
    return result


def _adapter_body(
    bridge: StrictTokenizerBridgeArtifact,
    records: tuple[Lfm2BridgeAlignedRecord, ...],
) -> dict[str, Any]:
    return {
        "schema": TRACE_ADAPTER_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "transport_profile_fingerprint": bridge.profile_fingerprint,
        "donor_row_capacity": bridge.transport_profile.donor_row_capacity,
        "target_row_capacity": bridge.transport_profile.target_row_capacity,
        "mapping_sha256": _canonical_sha256(
            list(bridge.transport_profile.target_to_donor_row_indices)
        ),
        "record_count": len(records),
        "ordered_records": [
            {
                "ordinal": record.ordinal,
                "record_id": record.record_id,
                "split": record.split,
                "alignment_sha256": record.alignment_sha256,
                "target_cache_input_sha256": record.target_cache_input_sha256,
                "donor_cache_input_sha256": record.donor_cache_input_sha256,
            }
            for record in records
        ],
        "execution_contract": {
            "parent_cache_uses_mapped_donor_ids": True,
            "child_replay_uses_original_target_ids": True,
            "numeric_token_id_equality_required": False,
            "parent_reexecution_performed": False,
        },
        "model_operations": {
            "parent_model_calls": 0,
            "child_model_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
        },
    }


@dataclass(frozen=True, slots=True, init=False)
class Lfm2StrictTokenizerBridgeTraceAdapter:
    """Verified immutable corpus adapter for parent cache and child replay."""

    bridge_artifact_sha256: str
    transport_profile_fingerprint: str
    evidence_sha256: str
    _bridge: StrictTokenizerBridgeArtifact
    _records: tuple[Lfm2BridgeAlignedRecord, ...]

    @property
    def records(self) -> tuple[Lfm2BridgeAlignedRecord, ...]:
        return self._records

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def donor_row_capacity(self) -> int:
        return self._bridge.transport_profile.donor_row_capacity

    @property
    def target_row_capacity(self) -> int:
        return self._bridge.transport_profile.target_row_capacity

    @property
    def donor_tokenizer_semantics_sha256(self) -> str:
        return self._bridge.donor_tokenizer_semantics_sha256

    @property
    def target_tokenizer_semantics_sha256(self) -> str:
        return self._bridge.target_tokenizer_semantics_sha256

    @property
    def target_tokenizer_package_fingerprint(self) -> str:
        return strict_tokenizer_bridge_target_package_fingerprint(self._bridge)

    @property
    def mapping(self) -> tuple[int, ...]:
        return self._bridge.transport_profile.target_to_donor_row_indices

    def verify(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        bridge = _verified_bridge(
            self._bridge,
            self.bridge_artifact_sha256,
        )
        if bridge.profile_fingerprint != self.transport_profile_fingerprint:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Tokenizer bridge transport profile changed."
            )
        for ordinal, record in enumerate(self._records):
            if record.ordinal != ordinal:
                raise Lfm2StrictTokenizerBridgeTraceError(
                    "Aligned record ordering changed."
                )
            record._verify(bridge)
        if _canonical_sha256(_adapter_body(bridge, self._records)) != (
            self.evidence_sha256
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Tokenizer bridge trace adapter seal is invalid."
            )
        return self

    def get_record(self, record_id: str) -> Lfm2BridgeAlignedRecord:
        selected = _text(record_id, "record_id")
        matches = tuple(
            record for record in self._records if record.record_id == selected
        )
        if len(matches) != 1:
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {selected!r} does not exist uniquely."
            )
        return matches[0]

    def records_for_split(
        self,
        split: str,
    ) -> tuple[Lfm2BridgeAlignedRecord, ...]:
        selected = _text(split, "split")
        return tuple(record for record in self._records if record.split == selected)

    def parent_cache_batches(
        self,
        *,
        split: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        records = (
            self._records if split is None else self.records_for_split(split)
        )
        return tuple(record.as_parent_cache_batch() for record in records)

    def child_replay_batches(
        self,
        *,
        split: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        records = (
            self._records if split is None else self.records_for_split(split)
        )
        return tuple(record.as_child_replay_batch() for record in records)

    def to_evidence_dict(self) -> dict[str, Any]:
        body = _adapter_body(self._bridge, self._records)
        return {**body, "evidence_sha256": self.evidence_sha256}


class Lfm2PreverifiedTokenizerBridgeTraceSession:
    """Two full adapter verifications around sealed O(1) pair traversal.

    Opening authenticates the whole adapter once.  Pair evidence is the only
    supported state transition: each successful call reauthenticates one
    selected record and proves its actual parent/child runtime inputs.  Final
    completion performs one more whole-adapter verification so backing-tensor
    mutation after an earlier pair was accepted cannot evade the session seal.
    """

    __slots__ = (
        "_adapter",
        "_authenticated_pair_evidence",
        "_bridge",
        "_bridge_artifact_sha256",
        "_evidence_sha256",
        "_profile_fingerprint",
        "_records",
        "_records_by_id",
    )

    def __init__(self, adapter: Lfm2StrictTokenizerBridgeTraceAdapter) -> None:
        self._initialize(adapter.verify())

    @classmethod
    def _from_verified(
        cls,
        adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
        token: object,
    ) -> Lfm2PreverifiedTokenizerBridgeTraceSession:
        if token is not _VERIFIED_SESSION_CONSTRUCTION_TOKEN:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified trace sessions require a full adapter verification."
            )
        result = object.__new__(cls)
        result._initialize(adapter)
        return result

    def _initialize(
        self,
        adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    ) -> None:
        if not isinstance(adapter, Lfm2StrictTokenizerBridgeTraceAdapter):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified trace sessions require a strict trace adapter."
            )
        records = adapter.records
        records_by_id = {record.record_id: record for record in records}
        if len(records_by_id) != len(records):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified trace sessions require unique record IDs."
            )
        self._adapter = adapter
        self._bridge = adapter._bridge
        self._records = records
        self._records_by_id = MappingProxyType(records_by_id)
        self._bridge_artifact_sha256 = adapter.bridge_artifact_sha256
        self._profile_fingerprint = adapter.transport_profile_fingerprint
        self._evidence_sha256 = adapter.evidence_sha256
        self._authenticated_pair_evidence: dict[
            int, Lfm2PairedInputAlignmentEvidence
        ] = {}

    def _verify_binding(self) -> None:
        adapter = self._adapter
        if (
            adapter._bridge is not self._bridge
            or adapter._records is not self._records
            or adapter.bridge_artifact_sha256
            != self._bridge_artifact_sha256
            or adapter.transport_profile_fingerprint
            != self._profile_fingerprint
            or adapter.evidence_sha256 != self._evidence_sha256
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified tokenizer bridge adapter binding changed."
            )

    @property
    def adapter(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        self._verify_binding()
        return self._adapter

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def authenticated_record_count(self) -> int:
        return len(self._authenticated_pair_evidence)

    def _pending_record(self, record_id: str) -> Lfm2BridgeAlignedRecord:
        self._verify_binding()
        selected = _text(record_id, "record_id")
        record = self._records_by_id.get(selected)
        if record is None:
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {selected!r} does not exist uniquely."
            )
        if record.ordinal in self._authenticated_pair_evidence:
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {selected!r} was already authenticated."
            )
        record._verify(self._bridge)
        return record

    def authenticate_input_pair(
        self,
        *,
        record_id: str,
        parent_cache_batch: Mapping[str, Any],
        child_replay_batch: Mapping[str, Any],
    ) -> Lfm2PairedInputAlignmentEvidence:
        """Verify one actual runtime pair and atomically record its evidence."""

        record = self._pending_record(record_id)
        evidence = _authenticate_lfm2_bridge_input_pair(
            self._adapter,
            record,
            parent_cache_batch=parent_cache_batch,
            child_replay_batch=child_replay_batch,
        )
        self._verify_binding()
        if (
            record.record_id not in self._records_by_id
            or self._records_by_id[record.record_id] is not record
            or record.ordinal in self._authenticated_pair_evidence
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified tokenizer bridge record completion changed."
            )
        _verify_pair_evidence_binding(self._adapter, record, evidence)
        self._authenticated_pair_evidence[record.ordinal] = evidence
        return evidence

    def verify_complete(self) -> Lfm2PreverifiedTokenizerBridgeTraceSession:
        """Require every runtime pair once, then reverify all live tensors."""

        self._verify_binding()
        expected = set(range(len(self._records)))
        if set(self._authenticated_pair_evidence) != expected:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified tokenizer bridge session did not authenticate "
                "every record exactly once."
            )
        self._adapter.verify()
        self._verify_binding()
        for record in self._records:
            _verify_pair_evidence_binding(
                self._adapter,
                record,
                self._authenticated_pair_evidence[record.ordinal],
            )
        return self


def _build_lfm2_strict_tokenizer_bridge_trace_adapter(
    bridge: StrictTokenizerBridgeArtifact,
    records: Iterable[Lfm2TargetTokenizedRecord],
    *,
    expected_artifact_sha256: str | None = None,
) -> Lfm2StrictTokenizerBridgeTraceAdapter:
    verified = _verified_bridge(bridge, expected_artifact_sha256)
    try:
        sources = tuple(records)
    except TypeError as exc:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "records must be an iterable of Lfm2TargetTokenizedRecord values."
        ) from exc
    if not sources:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "At least one target-tokenized record is required."
        )
    if any(not isinstance(value, Lfm2TargetTokenizedRecord) for value in sources):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "records must contain only Lfm2TargetTokenizedRecord values."
        )
    record_ids = tuple(record.record_id for record in sources)
    if len(set(record_ids)) != len(record_ids):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "record_id values must be unique across every split."
        )
    aligned = tuple(
        _build_aligned_record(ordinal, source, verified)
        for ordinal, source in enumerate(sources)
    )
    result = object.__new__(Lfm2StrictTokenizerBridgeTraceAdapter)
    body = _adapter_body(verified, aligned)
    values = {
        "bridge_artifact_sha256": verified.artifact_sha256,
        "transport_profile_fingerprint": verified.profile_fingerprint,
        "evidence_sha256": _canonical_sha256(body),
        "_bridge": verified,
        "_records": aligned,
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def build_lfm2_strict_tokenizer_bridge_trace_adapter(
    bridge: StrictTokenizerBridgeArtifact,
    records: Iterable[Lfm2TargetTokenizedRecord],
    *,
    expected_artifact_sha256: str | None = None,
) -> Lfm2StrictTokenizerBridgeTraceAdapter:
    """Compile and fully verify paired donor-cache/target-child evidence."""

    return _build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=expected_artifact_sha256,
    ).verify()


def build_preverified_lfm2_strict_tokenizer_bridge_trace_session(
    bridge: StrictTokenizerBridgeArtifact,
    records: Iterable[Lfm2TargetTokenizedRecord],
    *,
    expected_artifact_sha256: str | None = None,
) -> Lfm2PreverifiedTokenizerBridgeTraceSession:
    """Build an adapter and carry its opening verification into a session."""

    adapter = _build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    adapter.verify()
    return Lfm2PreverifiedTokenizerBridgeTraceSession._from_verified(
        adapter,
        _VERIFIED_SESSION_CONSTRUCTION_TOKEN,
    )


def _normalized_runtime_batch(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} must be a batch mapping."
        )
    unexpected = set(value).difference(_BATCH_KEYS)
    required = {"input_ids", "attention_mask", _FORCED_IDS}
    missing = required.difference(value)
    if unexpected or missing:
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} fields are incomplete or unsupported."
        )
    input_ids = _plain_integer_tensor(value["input_ids"], f"{name}.input_ids")
    shape = tuple(int(size) for size in input_ids.shape)
    mask = _attention_mask(value["attention_mask"], shape)
    positions = _position_ids(value.get("position_ids"), shape)
    forced = _plain_integer_tensor(value[_FORCED_IDS], f"{name}.{_FORCED_IDS}")
    if int(forced.shape[0]) != shape[0]:
        raise Lfm2StrictTokenizerBridgeTraceError(
            f"{name} forced decode changed the batch boundary."
        )
    return _clone_batch(
        input_ids=input_ids,
        attention_mask=mask,
        forced_decode_ids=forced,
        position_ids=positions,
    )


def _optional_tensor_equal(left: Any | None, right: Any | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(_torch().equal(left, right))


@dataclass(frozen=True, slots=True)
class Lfm2PairedInputAlignmentEvidence:
    """Seal proving actual parent/child batches obey one bridge record."""

    record_id: str
    split: str
    bridge_artifact_sha256: str
    alignment_sha256: str
    donor_cache_input_sha256: str
    target_cache_input_sha256: str
    numerically_different_token_count: int
    numeric_token_id_equality_required: bool
    parent_child_geometry_equal: bool
    masks_equal: bool
    positions_equal: bool
    forced_decode_mapped: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.numeric_token_id_equality_required is not False:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Paired evidence must not require numeric token-ID equality."
            )
        for name in (
            "parent_child_geometry_equal",
            "masks_equal",
            "positions_equal",
            "forced_decode_mapped",
        ):
            if getattr(self, name) is not True:
                raise Lfm2StrictTokenizerBridgeTraceError(
                    f"Paired input evidence requires {name}=True."
                )
        body = self.to_dict()
        body.pop("evidence_sha256")
        if _canonical_sha256(body) != self.evidence_sha256:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Paired input alignment evidence seal is invalid."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIRED_INPUT_SCHEMA,
            "record_id": self.record_id,
            "split": self.split,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "alignment_sha256": self.alignment_sha256,
            "donor_cache_input_sha256": self.donor_cache_input_sha256,
            "target_cache_input_sha256": self.target_cache_input_sha256,
            "numerically_different_token_count": (
                self.numerically_different_token_count
            ),
            "numeric_token_id_equality_required": (
                self.numeric_token_id_equality_required
            ),
            "parent_child_geometry_equal": self.parent_child_geometry_equal,
            "masks_equal": self.masks_equal,
            "positions_equal": self.positions_equal,
            "forced_decode_mapped": self.forced_decode_mapped,
            "model_operations": {
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
            },
            "evidence_sha256": self.evidence_sha256,
        }


def _verify_pair_evidence_binding(
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    record: Lfm2BridgeAlignedRecord,
    evidence: Lfm2PairedInputAlignmentEvidence,
) -> None:
    if not isinstance(evidence, Lfm2PairedInputAlignmentEvidence):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Authenticated pair evidence changed type."
        )
    evidence.__post_init__()
    if (
        evidence.record_id != record.record_id
        or evidence.split != record.split
        or evidence.bridge_artifact_sha256 != adapter.bridge_artifact_sha256
        or evidence.alignment_sha256 != record.alignment_sha256
        or evidence.donor_cache_input_sha256
        != record.donor_cache_input_sha256
        or evidence.target_cache_input_sha256
        != record.target_cache_input_sha256
        or evidence.numerically_different_token_count
        != record.numerically_different_token_count
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Authenticated runtime-pair evidence changed its record binding."
        )


def _authenticate_lfm2_bridge_input_pair(
    concrete_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    record: Lfm2BridgeAlignedRecord,
    *,
    parent_cache_batch: Mapping[str, Any],
    child_replay_batch: Mapping[str, Any],
) -> Lfm2PairedInputAlignmentEvidence:
    parent = _normalized_runtime_batch(parent_cache_batch, "parent_cache_batch")
    child = _normalized_runtime_batch(child_replay_batch, "child_replay_batch")
    expected_parent = record.as_parent_cache_batch()
    expected_child = record.as_child_replay_batch()
    torch = _torch()
    for name, observed, expected in (
        ("parent input_ids", parent["input_ids"], expected_parent["input_ids"]),
        (
            "parent attention_mask",
            parent["attention_mask"],
            expected_parent["attention_mask"],
        ),
        (
            "parent forced_decode_ids",
            parent[_FORCED_IDS],
            expected_parent[_FORCED_IDS],
        ),
        ("child input_ids", child["input_ids"], expected_child["input_ids"]),
        (
            "child attention_mask",
            child["attention_mask"],
            expected_child["attention_mask"],
        ),
        (
            "child forced_decode_ids",
            child[_FORCED_IDS],
            expected_child[_FORCED_IDS],
        ),
    ):
        if not torch.equal(observed, expected):
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Actual {name} differs from the sealed aligned record."
            )
    if not _optional_tensor_equal(
        parent.get("position_ids"),
        expected_parent.get("position_ids"),
    ) or not _optional_tensor_equal(
        child.get("position_ids"),
        expected_child.get("position_ids"),
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Actual position_ids differ from the sealed aligned record."
        )
    geometry_equal = (
        tuple(parent["input_ids"].shape) == tuple(child["input_ids"].shape)
        and tuple(parent[_FORCED_IDS].shape)
        == tuple(child[_FORCED_IDS].shape)
    )
    masks_equal = bool(
        torch.equal(parent["attention_mask"], child["attention_mask"])
    )
    positions_equal = _optional_tensor_equal(
        parent.get("position_ids"),
        child.get("position_ids"),
    )
    mapping = concrete_adapter.mapping
    mapped_prompt = _map_target_ids(child["input_ids"], mapping)
    mapped_forced = _map_target_ids(child[_FORCED_IDS], mapping)
    prompt_mapped = bool(torch.equal(mapped_prompt, parent["input_ids"]))
    forced_mapped = bool(torch.equal(mapped_forced, parent[_FORCED_IDS]))
    if not (geometry_equal and masks_equal and positions_equal and prompt_mapped):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Actual parent/child inputs do not satisfy the strict row map."
        )
    donor_sha = _cache_input_sha256(
        input_ids=parent["input_ids"],
        attention_mask=parent["attention_mask"],
        forced_decode_ids=parent[_FORCED_IDS],
        position_ids=parent.get("position_ids"),
    )
    target_sha = _cache_input_sha256(
        input_ids=child["input_ids"],
        attention_mask=child["attention_mask"],
        forced_decode_ids=child[_FORCED_IDS],
        position_ids=child.get("position_ids"),
    )
    body = {
        "schema": PAIRED_INPUT_SCHEMA,
        "record_id": record.record_id,
        "split": record.split,
        "bridge_artifact_sha256": concrete_adapter.bridge_artifact_sha256,
        "alignment_sha256": record.alignment_sha256,
        "donor_cache_input_sha256": donor_sha,
        "target_cache_input_sha256": target_sha,
        "numerically_different_token_count": (
            record.numerically_different_token_count
        ),
        "numeric_token_id_equality_required": False,
        "parent_child_geometry_equal": geometry_equal,
        "masks_equal": masks_equal,
        "positions_equal": positions_equal,
        "forced_decode_mapped": forced_mapped,
        "model_operations": {
            "parent_model_calls": 0,
            "child_model_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
        },
    }
    result = Lfm2PairedInputAlignmentEvidence(
        record_id=record.record_id,
        split=record.split,
        bridge_artifact_sha256=concrete_adapter.bridge_artifact_sha256,
        alignment_sha256=record.alignment_sha256,
        donor_cache_input_sha256=donor_sha,
        target_cache_input_sha256=target_sha,
        numerically_different_token_count=(
            record.numerically_different_token_count
        ),
        numeric_token_id_equality_required=False,
        parent_child_geometry_equal=geometry_equal,
        masks_equal=masks_equal,
        positions_equal=positions_equal,
        forced_decode_mapped=forced_mapped,
        evidence_sha256=_canonical_sha256(body),
    )
    return result


def authenticate_lfm2_bridge_input_pair(
    adapter: (
        Lfm2StrictTokenizerBridgeTraceAdapter
        | Lfm2PreverifiedTokenizerBridgeTraceSession
    ),
    *,
    record_id: str,
    parent_cache_batch: Mapping[str, Any],
    child_replay_batch: Mapping[str, Any],
) -> Lfm2PairedInputAlignmentEvidence:
    """Authenticate actual parent/child inputs without comparing raw IDs."""

    if isinstance(adapter, Lfm2PreverifiedTokenizerBridgeTraceSession):
        return adapter.authenticate_input_pair(
            record_id=record_id,
            parent_cache_batch=parent_cache_batch,
            child_replay_batch=child_replay_batch,
        )
    if not isinstance(adapter, Lfm2StrictTokenizerBridgeTraceAdapter):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "adapter must be a strict trace adapter or preverified session."
        )
    concrete_adapter = adapter.verify()
    record = concrete_adapter.get_record(record_id)
    return _authenticate_lfm2_bridge_input_pair(
        concrete_adapter,
        record,
        parent_cache_batch=parent_cache_batch,
        child_replay_batch=child_replay_batch,
    )


def _ledger_record_tensor_snapshots(
    record: Lfm2BridgeAlignedRecord,
) -> tuple[tuple[str, Any | None, Any | None], ...]:
    snapshots = []
    for name in (
        "_target_input_ids",
        "_donor_input_ids",
        "_attention_mask",
        "_target_forced_decode_ids",
        "_donor_forced_decode_ids",
        "_position_ids",
    ):
        tensor = getattr(record, name)
        snapshots.append(
            (
                name,
                tensor,
                (
                    None
                    if tensor is None
                    else tensor.detach().to(device="cpu").contiguous().clone()
                ),
            )
        )
    return tuple(snapshots)


def _ledger_record_scalar_guard(
    record: Lfm2BridgeAlignedRecord,
) -> tuple[Any, ...]:
    return (
        record.ordinal,
        record.record_id,
        record.split,
        record.bridge_artifact_sha256,
        record.transport_profile_fingerprint,
        record.target_cache_input_sha256,
        record.donor_cache_input_sha256,
        record.numerically_different_token_count,
        record.physical_padding_token_count,
        record.special_token_role_counts,
        record.alignment_sha256,
    )


def _ledger_profile_guard(
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
) -> tuple[Any, ...]:
    bridge = adapter._bridge
    profile = bridge.transport_profile
    return (
        (
            profile.donor_tokenizer_class,
            profile.target_tokenizer_class,
            profile.donor_vocab_size,
            profile.target_vocab_size,
            profile.donor_row_capacity,
            profile.target_row_capacity,
            profile.alignment_fingerprint,
            profile.alignment_resolution,
        ),
        tuple(
            (
                row,
                row.target_row_index,
                row.donor_row_index,
                row.resolution,
                row.special_token_roles,
            )
            for row in profile.row_mappings
        ),
        tuple(
            (
                record,
                record.relative_path,
                record.size_bytes,
                record.sha256,
            )
            for record in bridge.target_files
        ),
        (
            bridge.root_dir,
            bridge.donor_tokenizer_semantics_sha256,
            bridge.target_tokenizer_semantics_sha256,
            bridge.artifact_sha256,
        ),
    )


def _ledger_exact_integer_tensor(
    observed: Any,
    expected: Any,
    *,
    allow_bool: bool,
) -> bool:
    torch = _torch()
    return bool(
        torch.is_tensor(observed)
        and (
            _is_integer_tensor(torch, observed)
            or (allow_bool and observed.dtype == torch.bool)
        )
        and observed.layout == torch.strided
        and observed.device.type == "cpu"
        and not observed.requires_grad
        and torch.equal(observed, expected)
    )


def _ledger_record_binding(
    record: Lfm2BridgeAlignedRecord,
    evidence: Lfm2PairedInputAlignmentEvidence,
) -> tuple[Any, ...]:
    return (
        record.ordinal,
        record.record_id,
        record.split,
        record.bridge_artifact_sha256,
        record.transport_profile_fingerprint,
        record.alignment_sha256,
        record.donor_cache_input_sha256,
        record.target_cache_input_sha256,
        record.numerically_different_token_count,
        evidence.evidence_sha256,
    )


def _ledger_body(
    *,
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    parent_cache_key: str,
    parent_cache_session_sha256: str | None,
    parent_identity_sha256: str,
    parent_corpus_fingerprint: str,
    child_corpus_fingerprint: str,
    ordered_bindings: tuple[tuple[Any, ...], ...],
) -> dict[str, Any]:
    return {
        "schema": INPUT_PAIR_LEDGER_SCHEMA,
        "bridge_artifact_sha256": adapter.bridge_artifact_sha256,
        "bridge_trace_evidence_sha256": adapter.evidence_sha256,
        "transport_profile_fingerprint": (
            adapter.transport_profile_fingerprint
        ),
        "parent_cache_key": parent_cache_key,
        "parent_cache_session_sha256": parent_cache_session_sha256,
        "parent_identity_sha256": parent_identity_sha256,
        "parent_corpus_fingerprint": parent_corpus_fingerprint,
        "child_corpus_fingerprint": child_corpus_fingerprint,
        "record_count": len(ordered_bindings),
        "ordered_records": [
            {
                "ordinal": binding[0],
                "record_id": binding[1],
                "split": binding[2],
                "alignment_sha256": binding[5],
                "donor_cache_input_sha256": binding[6],
                "target_cache_input_sha256": binding[7],
                "pair_evidence_sha256": binding[9],
            }
            for binding in ordered_bindings
        ],
        "reuse_contract": {
            "fold_lifetime_input_pair_authentication": True,
            "child_activation_and_weight_changes_permitted": True,
            "ordered_record_and_input_sha_binding_required_per_replay": True,
            "actual_input_value_binding_required_per_replay": True,
            "adapter_full_verification_per_replay": False,
            "runtime_row_remap_per_replay": False,
            "runtime_input_hash_per_replay": False,
            "runtime_input_clone_per_replay": False,
        },
        "model_operations": {
            "parent_model_calls": 0,
            "child_model_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
        },
    }


class Lfm2PreverifiedTokenizerBridgeInputPairLedger:
    """Fold-lifetime authentication of one immutable tokenizer input corpus.

    Construction performs the conservative adapter and runtime-pair traversal
    exactly once.  Replays then retain exact value and order checks without
    rebuilding the target-to-donor map, normalizing/cloning inputs, hashing
    their bytes, or invoking the full adapter verifier.  The child model and
    all non-input replay tensors may change between uses.
    """

    __slots__ = (
        "_adapter_guards",
        "_context",
        "_evidence_sha256",
        "_ordered_bindings",
        "_pair_evidence",
        "_profile_guard",
        "_record_guards",
        "_storage_canary",
    )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Preverified tokenizer bridge input-pair ledgers are immutable."
        )

    @property
    def adapter(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        self._verify_storage()
        return self._adapter_guards[0]

    @property
    def evidence_sha256(self) -> str:
        self._verify_storage()
        return self._evidence_sha256

    @property
    def record_count(self) -> int:
        self._verify_storage()
        return len(self._ordered_bindings)

    @property
    def authenticated_record_count(self) -> int:
        return self.record_count

    @property
    def parent_cache_key(self) -> str:
        self._verify_storage()
        return self._context[0]

    @property
    def parent_cache_session_sha256(self) -> str | None:
        self._verify_storage()
        return self._context[1]

    @property
    def parent_corpus_fingerprint(self) -> str:
        self._verify_storage()
        return self._context[3]

    @property
    def child_corpus_fingerprint(self) -> str:
        self._verify_storage()
        return self._context[4]

    def _verify_storage(self) -> None:
        canary = self._storage_canary
        if (
            id(self._adapter_guards) != canary[0]
            or id(self._context) != canary[1]
            or id(self._ordered_bindings) != canary[2]
            or id(self._pair_evidence) != canary[3]
            or id(self._profile_guard) != canary[4]
            or id(self._record_guards) != canary[5]
            or self._evidence_sha256 != canary[6]
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified tokenizer bridge input-pair ledger storage changed."
            )

    def _verify_adapter_binding(self) -> None:
        self._verify_storage()
        (
            adapter,
            bridge,
            records,
            profile,
            row_mappings,
            bridge_artifact_sha256,
            profile_fingerprint,
            adapter_evidence_sha256,
        ) = self._adapter_guards
        if (
            adapter._bridge is not bridge
            or adapter._records is not records
            or bridge.transport_profile is not profile
            or profile.row_mappings is not row_mappings
            or adapter.bridge_artifact_sha256 != bridge_artifact_sha256
            or adapter.transport_profile_fingerprint != profile_fingerprint
            or adapter.evidence_sha256 != adapter_evidence_sha256
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified input-pair ledger bridge binding changed."
            )
        (
            profile_scalars,
            guarded_rows,
            guarded_files,
            bridge_scalars,
        ) = self._profile_guard
        if (
            (
                profile.donor_tokenizer_class,
                profile.target_tokenizer_class,
                profile.donor_vocab_size,
                profile.target_vocab_size,
                profile.donor_row_capacity,
                profile.target_row_capacity,
                profile.alignment_fingerprint,
                profile.alignment_resolution,
            )
            != profile_scalars
            or len(row_mappings) != len(guarded_rows)
            or any(
                observed is not guarded[0]
                or observed.target_row_index != guarded[1]
                or observed.donor_row_index != guarded[2]
                or observed.resolution != guarded[3]
                or observed.special_token_roles != guarded[4]
                for observed, guarded in zip(
                    row_mappings,
                    guarded_rows,
                    strict=True,
                )
            )
            or len(bridge.target_files) != len(guarded_files)
            or any(
                observed is not guarded[0]
                or observed.relative_path != guarded[1]
                or observed.size_bytes != guarded[2]
                or observed.sha256 != guarded[3]
                for observed, guarded in zip(
                    bridge.target_files,
                    guarded_files,
                    strict=True,
                )
            )
            or (
                bridge.root_dir,
                bridge.donor_tokenizer_semantics_sha256,
                bridge.target_tokenizer_semantics_sha256,
                bridge.artifact_sha256,
            )
            != bridge_scalars
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified input-pair ledger nested bridge evidence changed."
            )

    def verify_replay_binding(
        self,
        adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
        *,
        parent_cache_key: str,
        parent_cache_session: object | None,
        parent_cache_session_sha256: str | None,
        parent_identity_sha256: str,
        parent_corpus_fingerprint: str,
        child_corpus_fingerprint: str,
        record_count: int,
    ) -> Lfm2PreverifiedTokenizerBridgeInputPairLedger:
        """Check shallow fold/cache/session/corpus identity bindings."""

        self._verify_adapter_binding()
        (
            bound_adapter,
            bound_bridge,
            bound_records,
            bound_profile,
            bound_row_mappings,
            bridge_artifact_sha256,
            profile_fingerprint,
            adapter_evidence_sha256,
        ) = self._adapter_guards
        context = self._context
        bound_session = context[5]
        if bound_session is not None and (
            not isinstance(bound_session, SEALED_EVIDENCE_SESSION_TYPES)
            or getattr(bound_session, "closed", False)
            or bound_session.cache_key != context[0]
            or bound_session.batch_count != len(self._ordered_bindings)
            or bound_session.evidence_sha256 != context[1]
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified input-pair ledger parent session changed or closed."
            )
        if (
            adapter is not bound_adapter
            or adapter._bridge is not bound_bridge
            or adapter._records is not bound_records
            or adapter._bridge.transport_profile is not bound_profile
            or bound_profile.row_mappings is not bound_row_mappings
            or adapter.bridge_artifact_sha256 != bridge_artifact_sha256
            or adapter.transport_profile_fingerprint != profile_fingerprint
            or adapter.evidence_sha256 != adapter_evidence_sha256
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified input-pair ledger bridge binding changed."
            )
        observed_session_sha = (
            None
            if parent_cache_session_sha256 is None
            else _sha256(
                parent_cache_session_sha256,
                "parent_cache_session_sha256",
            )
        )
        if (
            _sha256(parent_cache_key, "parent_cache_key") != context[0]
            or observed_session_sha != context[1]
            or parent_cache_session is not context[5]
            or _sha256(parent_identity_sha256, "parent_identity_sha256")
            != context[2]
            or _sha256(
                parent_corpus_fingerprint,
                "parent_corpus_fingerprint",
            )
            != context[3]
            or _sha256(
                child_corpus_fingerprint,
                "child_corpus_fingerprint",
            )
            != context[4]
            or isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count != len(self._ordered_bindings)
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified input-pair ledger cache, session, corpus, or "
                "record-count binding changed."
            )
        return self

    def replay_record_evidence(
        self,
        *,
        index: int,
        record_id: str,
        parent_cache_input_sha256: str,
        target_cache_input_sha256: str,
        parent_cache_batch: Mapping[str, Any],
        child_replay_batch: Mapping[str, Any],
    ) -> Lfm2PairedInputAlignmentEvidence:
        """Rebind one standalone pair after a complete bridge guard check."""

        self._verify_adapter_binding()
        return self._replay_record_evidence_prebound(
            index=index,
            record_id=record_id,
            parent_cache_input_sha256=parent_cache_input_sha256,
            target_cache_input_sha256=target_cache_input_sha256,
            parent_cache_batch=parent_cache_batch,
            child_replay_batch=child_replay_batch,
        )

    def _replay_record_evidence_prebound(
        self,
        *,
        index: int,
        record_id: str,
        parent_cache_input_sha256: str,
        target_cache_input_sha256: str,
        parent_cache_batch: Mapping[str, Any],
        child_replay_batch: Mapping[str, Any],
    ) -> Lfm2PairedInputAlignmentEvidence:
        """Rebind one replay pair without mapping, hashing, or cloning it."""

        self._verify_storage()
        ordinal = _nonnegative_int(index, "index")
        if ordinal >= len(self._ordered_bindings):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger index is outside its ordered corpus."
            )
        binding = self._ordered_bindings[ordinal]
        selected_id = _text(record_id, "record_id")
        parent_sha = _sha256(
            parent_cache_input_sha256,
            "parent_cache_input_sha256",
        )
        target_sha = _sha256(
            target_cache_input_sha256,
            "target_cache_input_sha256",
        )
        if (
            binding[0] != ordinal
            or binding[1] != selected_id
            or binding[6] != parent_sha
            or binding[7] != target_sha
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger replay order, record, or exact input "
                "binding changed."
            )

        record, scalar_guard, tensor_snapshots = self._record_guards[ordinal]
        if (
            self._adapter_guards[2][ordinal] is not record
            or _ledger_record_scalar_guard(record) != scalar_guard
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger aligned-record evidence changed."
            )
        for name, bound_tensor, expected_tensor in tensor_snapshots:
            observed_tensor = getattr(record, name)
            if expected_tensor is None:
                unchanged = observed_tensor is bound_tensor is None
            else:
                unchanged = bool(
                    observed_tensor is bound_tensor
                    and _ledger_exact_integer_tensor(
                        observed_tensor,
                        expected_tensor,
                        allow_bool=name == "_attention_mask",
                    )
                )
            if not unchanged:
                raise Lfm2StrictTokenizerBridgeTraceError(
                    "Input-pair ledger aligned-record tensors changed."
                )

        parent = parent_cache_batch
        child = child_replay_batch
        required = {"input_ids", "attention_mask", _FORCED_IDS}
        if (
            not isinstance(parent, Mapping)
            or not isinstance(child, Mapping)
            or set(parent) not in (required, required | {"position_ids"})
            or set(child) not in (required, required | {"position_ids"})
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger replay inputs are incomplete or unsupported."
            )
        (
            target_input_ids,
            donor_input_ids,
            attention_mask,
            target_forced_decode_ids,
            donor_forced_decode_ids,
            position_ids,
        ) = tuple(snapshot for _name, _bound, snapshot in tensor_snapshots)
        exact_pairs = (
            (parent["input_ids"], donor_input_ids, False),
            (parent["attention_mask"], attention_mask, True),
            (parent[_FORCED_IDS], donor_forced_decode_ids, False),
            (child["input_ids"], target_input_ids, False),
            (child["attention_mask"], attention_mask, True),
            (child[_FORCED_IDS], target_forced_decode_ids, False),
        )
        if any(
            not _ledger_exact_integer_tensor(
                observed,
                expected,
                allow_bool=allow_bool,
            )
            for observed, expected, allow_bool in exact_pairs
        ) or not (
            parent.get("position_ids") is position_ids is None
            or _ledger_exact_integer_tensor(
                parent.get("position_ids"),
                position_ids,
                allow_bool=False,
            )
        ) or not (
            child.get("position_ids") is position_ids is None
            or _ledger_exact_integer_tensor(
                child.get("position_ids"),
                position_ids,
                allow_bool=False,
            )
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger replay values changed from their "
                "preverified exact inputs."
            )

        evidence = self._pair_evidence[ordinal]
        if (
            evidence.record_id != binding[1]
            or evidence.split != binding[2]
            or evidence.bridge_artifact_sha256 != binding[3]
            or evidence.alignment_sha256 != binding[5]
            or evidence.donor_cache_input_sha256 != binding[6]
            or evidence.target_cache_input_sha256 != binding[7]
            or evidence.numerically_different_token_count != binding[8]
            or evidence.numeric_token_id_equality_required is not False
            or evidence.parent_child_geometry_equal is not True
            or evidence.masks_equal is not True
            or evidence.positions_equal is not True
            or evidence.forced_decode_mapped is not True
            or evidence.evidence_sha256 != binding[9]
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger runtime evidence changed."
            )
        return evidence

    def to_evidence_dict(self) -> dict[str, Any]:
        self._verify_adapter_binding()
        body = _ledger_body(
            adapter=self._adapter_guards[0],
            parent_cache_key=self._context[0],
            parent_cache_session_sha256=self._context[1],
            parent_identity_sha256=self._context[2],
            parent_corpus_fingerprint=self._context[3],
            child_corpus_fingerprint=self._context[4],
            ordered_bindings=self._ordered_bindings,
        )
        if _canonical_sha256(body) != self._evidence_sha256:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Input-pair ledger evidence seal changed."
            )
        return {**body, "evidence_sha256": self._evidence_sha256}


def _build_preverified_lfm2_tokenizer_bridge_input_pair_ledger(
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    *,
    parent_cache_key: str,
    parent_cache_session: object | None,
    parent_cache_session_sha256: str | None,
    parent_identity_sha256: str,
    parent_corpus_fingerprint: str,
    child_corpus_fingerprint: str,
    parent_cache_batches: Iterable[Mapping[str, Any]],
    child_replay_batches: Iterable[Mapping[str, Any]],
) -> Lfm2PreverifiedTokenizerBridgeInputPairLedger:
    """Authenticate and freeze every ordered runtime input pair once."""

    if not isinstance(adapter, Lfm2StrictTokenizerBridgeTraceAdapter):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Input-pair ledgers require a strict trace adapter."
        )
    cache_key = _sha256(parent_cache_key, "parent_cache_key")
    session_sha = (
        None
        if parent_cache_session_sha256 is None
        else _sha256(
            parent_cache_session_sha256,
            "parent_cache_session_sha256",
        )
    )
    if (parent_cache_session is None) != (session_sha is None):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Parent cache session object and digest must be supplied together."
        )
    if parent_cache_session is not None and (
        not isinstance(parent_cache_session, SEALED_EVIDENCE_SESSION_TYPES)
        or getattr(parent_cache_session, "closed", False)
        or parent_cache_session.cache_key != cache_key
        or parent_cache_session.batch_count != adapter.record_count
        or parent_cache_session.evidence_sha256 != session_sha
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Parent cache session is not a live authenticated session bound "
            "to this cache key, digest, and record count."
        )
    parent_identity = _sha256(
        parent_identity_sha256,
        "parent_identity_sha256",
    )
    parent_corpus = _sha256(
        parent_corpus_fingerprint,
        "parent_corpus_fingerprint",
    )
    child_corpus = _sha256(
        child_corpus_fingerprint,
        "child_corpus_fingerprint",
    )
    try:
        parent_batches = tuple(parent_cache_batches)
        child_batches = tuple(child_replay_batches)
    except TypeError as exc:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Input-pair ledger batches must be ordered iterables."
        ) from exc
    if (
        len(parent_batches) != adapter.record_count
        or len(child_batches) != adapter.record_count
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Input-pair ledger batch count differs from its adapter."
        )

    session = Lfm2PreverifiedTokenizerBridgeTraceSession(adapter)
    evidence = tuple(
        session.authenticate_input_pair(
            record_id=record.record_id,
            parent_cache_batch=parent_batches[index],
            child_replay_batch=child_batches[index],
        )
        for index, record in enumerate(adapter.records)
    )
    session.verify_complete()
    ordered_bindings = tuple(
        _ledger_record_binding(record, evidence[record.ordinal])
        for record in adapter.records
    )
    record_guards = tuple(
        (
            record,
            _ledger_record_scalar_guard(record),
            _ledger_record_tensor_snapshots(record),
        )
        for record in adapter.records
    )
    profile_guard = _ledger_profile_guard(adapter)
    profile = adapter._bridge.transport_profile
    adapter_guards = (
        adapter,
        adapter._bridge,
        adapter._records,
        profile,
        profile.row_mappings,
        adapter.bridge_artifact_sha256,
        adapter.transport_profile_fingerprint,
        adapter.evidence_sha256,
    )
    context = (
        cache_key,
        session_sha,
        parent_identity,
        parent_corpus,
        child_corpus,
        parent_cache_session,
    )
    body = _ledger_body(
        adapter=adapter,
        parent_cache_key=cache_key,
        parent_cache_session_sha256=session_sha,
        parent_identity_sha256=parent_identity,
        parent_corpus_fingerprint=parent_corpus,
        child_corpus_fingerprint=child_corpus,
        ordered_bindings=ordered_bindings,
    )
    result = object.__new__(Lfm2PreverifiedTokenizerBridgeInputPairLedger)
    object.__setattr__(result, "_adapter_guards", adapter_guards)
    object.__setattr__(result, "_context", context)
    object.__setattr__(result, "_ordered_bindings", ordered_bindings)
    object.__setattr__(result, "_pair_evidence", evidence)
    object.__setattr__(result, "_profile_guard", profile_guard)
    object.__setattr__(result, "_record_guards", record_guards)
    object.__setattr__(result, "_evidence_sha256", _canonical_sha256(body))
    object.__setattr__(
        result,
        "_storage_canary",
        (
            id(adapter_guards),
            id(context),
            id(ordered_bindings),
            id(evidence),
            id(profile_guard),
            id(record_guards),
            result._evidence_sha256,
        ),
    )
    result.to_evidence_dict()
    return result


def _float_logits(value: Any, donor_capacity: int, batch_size: int) -> Any:
    torch = _torch()
    if (
        not torch.is_tensor(value)
        or value.ndim != 3
        or value.numel() == 0
        or not value.is_floating_point()
        or value.dtype.is_complex
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "donor_logits must be a nonempty plain rank-3 real floating tensor."
        )
    if int(value.shape[0]) != batch_size:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "donor_logits changed the sealed batch boundary."
        )
    if int(value.shape[-1]) != donor_capacity:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "donor_logits width differs from sealed donor row capacity."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "donor_logits contains NaN or infinity."
        )
    return result


def _phase(
    value: Any,
    forced_decode_steps: int,
    prompt_width: int,
    logit_rows: int,
) -> str:
    selected = _text(value, "phase")
    if selected in {"prefill", "cached_prefill"}:
        if logit_rows <= 0 or logit_rows > prompt_width:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Prefill logits must contain a nonempty suffix within the "
                "sealed prompt boundary."
            )
        return selected
    match = _DECODE_PHASE.fullmatch(selected)
    if match is None:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "phase must be prefill, cached_prefill, or decode.NNN."
        )
    index = int(match.group(1))
    if index >= forced_decode_steps:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Decode phase exceeds the sealed forced-decode trajectory."
        )
    if logit_rows != 1:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Decode logits must contain exactly one sequence row."
        )
    return selected


@dataclass(frozen=True, slots=True, init=False)
class Lfm2ProjectedCachedParentLogits:
    """Detached target-column logits and deterministic target-space margins."""

    record_id: str
    split: str
    phase: str
    cache_key: str
    cache_batch_index: int
    bridge_artifact_sha256: str
    alignment_sha256: str
    parent_cache_input_sha256: str
    source_logits_sha256: str
    top_k: int
    projection_sha256: str
    _target_logits: Any
    _topk_target_ids: Any
    _topk_values: Any
    _winner_to_topk_margins: Any

    @property
    def target_logits(self) -> Any:
        return self._target_logits.clone()

    @property
    def topk_target_ids(self) -> Any:
        return self._topk_target_ids.clone()

    @property
    def topk_values(self) -> Any:
        return self._topk_values.clone()

    @property
    def winner_to_topk_margins(self) -> Any:
        return self._winner_to_topk_margins.clone()

    @property
    def winner_margin(self) -> Any:
        return self._winner_to_topk_margins[..., 0].clone()

    def to_evidence_dict(self) -> dict[str, Any]:
        return {
            "schema": LOGIT_PROJECTION_SCHEMA,
            "record_id": self.record_id,
            "split": self.split,
            "phase": self.phase,
            "cache_key": self.cache_key,
            "cache_batch_index": self.cache_batch_index,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "alignment_sha256": self.alignment_sha256,
            "parent_cache_input_sha256": self.parent_cache_input_sha256,
            "source_logits_sha256": self.source_logits_sha256,
            "target_logits": _tensor_descriptor(self._target_logits),
            "top_k": self.top_k,
            "topk_target_ids": _tensor_descriptor(self._topk_target_ids),
            "topk_values": _tensor_descriptor(self._topk_values),
            "winner_to_topk_margins": _tensor_descriptor(
                self._winner_to_topk_margins
            ),
            "projection_contract": {
                "source": "verified-full-donor-vocabulary-cached-logits",
                "target_columns": "strict-target-to-donor-row-index-select",
                "topk_space": "target-physical-vocabulary",
                "tie_break": "lower-target-row-first",
                "numeric_token_id_equality_required": False,
            },
            "model_operations": {
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "pruned_parameters": 0,
                "quantized_parameters": 0,
            },
            "projection_sha256": self.projection_sha256,
        }

    def verify(
        self,
        adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    ) -> Lfm2ProjectedCachedParentLogits:
        adapter.verify()
        record = adapter.get_record(self.record_id)
        return _verify_projected_cached_parent_logits(self, adapter, record)


def _verify_projected_cached_parent_logits(
    projection: Lfm2ProjectedCachedParentLogits,
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    record: Lfm2BridgeAlignedRecord,
) -> Lfm2ProjectedCachedParentLogits:
    """Verify one projection against an already-authenticated adapter record."""

    if not isinstance(projection, Lfm2ProjectedCachedParentLogits):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "projection must be Lfm2ProjectedCachedParentLogits."
        )
    if (
        projection.record_id != record.record_id
        or projection.bridge_artifact_sha256 != adapter.bridge_artifact_sha256
        or projection.alignment_sha256 != record.alignment_sha256
        or projection.parent_cache_input_sha256
        != record.donor_cache_input_sha256
        or projection.cache_batch_index != record.ordinal
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Projected logits are not bound to this aligned cache record."
        )
    body = projection.to_evidence_dict()
    body.pop("projection_sha256")
    if _canonical_sha256(body) != projection.projection_sha256:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Projected cached-parent logits seal is invalid."
        )
    return projection


def _build_projected_cached_parent_logits(
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    record: Lfm2BridgeAlignedRecord,
    mapping: Any,
    *,
    parent_cache_input_sha256: str,
    cache_key: str,
    cache_batch_index: int,
    phase: str,
    donor_logits: Any,
    top_k: int,
) -> Lfm2ProjectedCachedParentLogits:
    if (
        record.ordinal < 0
        or record.ordinal >= len(adapter.records)
        or adapter.records[record.ordinal] is not record
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Projection record is not owned by the authenticated adapter."
        )
    parent_input_sha = _sha256(
        parent_cache_input_sha256,
        "parent_cache_input_sha256",
    )
    if parent_input_sha != record.donor_cache_input_sha256:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Cached parent input is not the mapped donor record."
        )
    selected_cache_key = _sha256(cache_key, "cache_key")
    batch_index = _nonnegative_int(cache_batch_index, "cache_batch_index")
    if batch_index != record.ordinal:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "cache_batch_index differs from the sealed record ordering."
        )
    logits = _float_logits(
        donor_logits,
        adapter.donor_row_capacity,
        record.batch_size,
    )
    selected_phase = _phase(
        phase,
        record.forced_decode_steps,
        record.prompt_width,
        int(logits.shape[1]),
    )
    requested_top_k = _positive_int(top_k, "top_k")
    if adapter.target_row_capacity < 2:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Target vocabulary needs at least two rows for margin evidence."
        )
    effective_top_k = min(requested_top_k, adapter.target_row_capacity)
    if effective_top_k < 2:
        raise Lfm2StrictTokenizerBridgeTraceError(
            "top_k must select at least two target rows."
        )
    torch = _torch()
    if (
        not isinstance(mapping, torch.Tensor)
        or mapping.dtype != torch.long
        or mapping.device.type != "cpu"
        or tuple(mapping.shape) != (adapter.target_row_capacity,)
    ):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Projection mapping differs from the authenticated bridge profile."
        )
    projected = logits.index_select(-1, mapping).contiguous().clone()
    try:
        ordered_ids = torch.argsort(
            projected,
            dim=-1,
            descending=True,
            stable=True,
        )
    except TypeError as exc:  # pragma: no cover - supported torch floor
        raise Lfm2StrictTokenizerBridgeTraceError(
            "Installed PyTorch lacks deterministic stable target top-k."
        ) from exc
    topk_ids = ordered_ids[..., :effective_top_k].contiguous()
    topk_values = projected.gather(-1, topk_ids).contiguous()
    margins = (topk_values[..., :1] - topk_values[..., 1:]).contiguous()
    source_sha = _tensor_sha256(logits)
    result = object.__new__(Lfm2ProjectedCachedParentLogits)
    values = {
        "record_id": record.record_id,
        "split": record.split,
        "phase": selected_phase,
        "cache_key": selected_cache_key,
        "cache_batch_index": batch_index,
        "bridge_artifact_sha256": adapter.bridge_artifact_sha256,
        "alignment_sha256": record.alignment_sha256,
        "parent_cache_input_sha256": parent_input_sha,
        "source_logits_sha256": source_sha,
        "top_k": effective_top_k,
        "_target_logits": projected,
        "_topk_target_ids": topk_ids,
        "_topk_values": topk_values,
        "_winner_to_topk_margins": margins,
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "projection_sha256", "0" * 64)
    body = result.to_evidence_dict()
    body.pop("projection_sha256")
    object.__setattr__(result, "projection_sha256", _canonical_sha256(body))
    return result


class Lfm2PreverifiedCachedParentProjectionSession:
    """Two whole-adapter verifications around one complete projection sweep.

    The adapter and bridge files are authenticated when the session opens and
    closes.  Inside that boundary, record lookup is O(1), the target-row map is
    materialized once, and each record is locally reauthenticated once.  Every
    cached-prefill/decode phase must appear exactly once before completion.
    """

    __slots__ = (
        "_adapter",
        "_bridge",
        "_bridge_artifact_sha256",
        "_cache_key",
        "_complete",
        "_evidence_sha256",
        "_expected_projection_keys",
        "_mapping",
        "_profile_fingerprint",
        "_projections",
        "_records",
        "_records_by_id",
        "_top_k",
        "_verified_record_ordinals",
    )

    def __init__(self, adapter: Lfm2StrictTokenizerBridgeTraceAdapter) -> None:
        if not isinstance(adapter, Lfm2StrictTokenizerBridgeTraceAdapter):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection sessions require a strict trace adapter."
            )
        verified = adapter.verify()
        records = verified.records
        records_by_id = {record.record_id: record for record in records}
        if len(records_by_id) != len(records):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection sessions require unique record IDs."
            )
        self._adapter = verified
        self._bridge = verified._bridge
        self._records = records
        self._records_by_id = MappingProxyType(records_by_id)
        self._bridge_artifact_sha256 = verified.bridge_artifact_sha256
        self._profile_fingerprint = verified.transport_profile_fingerprint
        self._evidence_sha256 = verified.evidence_sha256
        self._mapping = _torch().tensor(verified.mapping, dtype=_torch().long)
        self._expected_projection_keys = frozenset(
            (record.ordinal, phase)
            for record in records
            for phase in (
                "cached_prefill",
                *(
                    f"decode.{step:03d}"
                    for step in range(record.forced_decode_steps)
                ),
            )
        )
        self._projections: dict[
            tuple[int, str], Lfm2ProjectedCachedParentLogits
        ] = {}
        self._verified_record_ordinals: set[int] = set()
        self._cache_key: str | None = None
        self._top_k: int | None = None
        self._complete = False

    def _verify_binding(self) -> None:
        adapter = self._adapter
        if (
            adapter._bridge is not self._bridge
            or adapter._records is not self._records
            or adapter.bridge_artifact_sha256 != self._bridge_artifact_sha256
            or adapter.transport_profile_fingerprint
            != self._profile_fingerprint
            or adapter.evidence_sha256 != self._evidence_sha256
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Preverified projection-session adapter binding changed."
            )

    def _record(self, record_id: str) -> Lfm2BridgeAlignedRecord:
        self._verify_binding()
        if self._complete:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Completed projection sessions cannot accept more phases."
            )
        selected = _text(record_id, "record_id")
        record = self._records_by_id.get(selected)
        if record is None:
            raise Lfm2StrictTokenizerBridgeTraceError(
                f"Aligned record {selected!r} does not exist uniquely."
            )
        if record.ordinal not in self._verified_record_ordinals:
            record._verify(self._bridge)
            self._verified_record_ordinals.add(record.ordinal)
        return record

    def _record_projection(
        self,
        record: Lfm2BridgeAlignedRecord,
        projection: Lfm2ProjectedCachedParentLogits,
    ) -> Lfm2ProjectedCachedParentLogits:
        key = (record.ordinal, projection.phase)
        if key not in self._expected_projection_keys:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection phase is outside the complete cached trajectory."
            )
        if key in self._projections:
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection phase was authenticated more than once."
            )
        if self._cache_key is None:
            self._cache_key = projection.cache_key
            self._top_k = projection.top_k
        elif (
            projection.cache_key != self._cache_key
            or projection.top_k != self._top_k
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection session mixed cache keys or top-k policies."
            )
        self._projections[key] = projection
        return projection

    @property
    def projection_count(self) -> int:
        return len(self._projections)

    @property
    def expected_projection_count(self) -> int:
        return len(self._expected_projection_keys)

    def project(
        self,
        *,
        record_id: str,
        parent_cache_input_sha256: str,
        cache_key: str,
        cache_batch_index: int,
        phase: str,
        donor_logits: Any,
        top_k: int = 5,
    ) -> Lfm2ProjectedCachedParentLogits:
        record = self._record(record_id)
        projection = _build_projected_cached_parent_logits(
            self._adapter,
            record,
            self._mapping,
            parent_cache_input_sha256=parent_cache_input_sha256,
            cache_key=cache_key,
            cache_batch_index=cache_batch_index,
            phase=phase,
            donor_logits=donor_logits,
            top_k=top_k,
        )
        return self._record_projection(record, projection)

    def authenticate_projection(
        self,
        projection: Lfm2ProjectedCachedParentLogits,
    ) -> Lfm2ProjectedCachedParentLogits:
        if not isinstance(projection, Lfm2ProjectedCachedParentLogits):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "projection must be Lfm2ProjectedCachedParentLogits."
            )
        record = self._record(projection.record_id)
        verified = _verify_projected_cached_parent_logits(
            projection,
            self._adapter,
            record,
        )
        return self._record_projection(record, verified)

    def verify_complete(
        self,
    ) -> Lfm2PreverifiedCachedParentProjectionSession:
        self._verify_binding()
        if set(self._projections) != set(self._expected_projection_keys):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection session did not authenticate every cached phase "
                "exactly once."
            )
        if tuple(int(value) for value in self._mapping.tolist()) != (
            self._adapter.mapping
        ):
            raise Lfm2StrictTokenizerBridgeTraceError(
                "Projection-session target-row mapping changed."
            )
        self._adapter.verify()
        self._verify_binding()
        for key in sorted(self._projections):
            record = self._records[key[0]]
            _verify_projected_cached_parent_logits(
                self._projections[key],
                self._adapter,
                record,
            )
        self._complete = True
        return self


def project_lfm2_cached_parent_logits(
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    *,
    record_id: str,
    parent_cache_input_sha256: str,
    cache_key: str,
    cache_batch_index: int,
    phase: str,
    donor_logits: Any,
    top_k: int = 5,
) -> Lfm2ProjectedCachedParentLogits:
    """Project one authenticated full donor-logit cache slice to target rows."""

    if not isinstance(adapter, Lfm2StrictTokenizerBridgeTraceAdapter):
        raise Lfm2StrictTokenizerBridgeTraceError(
            "adapter must be Lfm2StrictTokenizerBridgeTraceAdapter."
        )
    adapter.verify()
    record = adapter.get_record(record_id)
    torch = _torch()
    mapping = torch.tensor(adapter.mapping, dtype=torch.long)
    result = _build_projected_cached_parent_logits(
        adapter,
        record,
        mapping,
        parent_cache_input_sha256=parent_cache_input_sha256,
        cache_key=cache_key,
        cache_batch_index=cache_batch_index,
        phase=phase,
        donor_logits=donor_logits,
        top_k=top_k,
    )
    return _verify_projected_cached_parent_logits(result, adapter, record)


__all__ = [
    "ALIGNED_RECORD_SCHEMA",
    "INPUT_PAIR_LEDGER_SCHEMA",
    "LOGIT_PROJECTION_SCHEMA",
    "Lfm2BridgeAlignedRecord",
    "Lfm2PairedInputAlignmentEvidence",
    "Lfm2PreverifiedCachedParentProjectionSession",
    "Lfm2PreverifiedTokenizerBridgeInputPairLedger",
    "Lfm2PreverifiedTokenizerBridgeTraceSession",
    "Lfm2ProjectedCachedParentLogits",
    "Lfm2StrictTokenizerBridgeTraceAdapter",
    "Lfm2StrictTokenizerBridgeTraceError",
    "Lfm2TargetTokenizedRecord",
    "PAIRED_INPUT_SCHEMA",
    "TRACE_ADAPTER_SCHEMA",
    "authenticate_lfm2_bridge_input_pair",
    "build_preverified_lfm2_strict_tokenizer_bridge_trace_session",
    "build_lfm2_strict_tokenizer_bridge_trace_adapter",
    "project_lfm2_cached_parent_logits",
]
