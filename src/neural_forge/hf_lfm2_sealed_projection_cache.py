"""Durable, value-authenticated cache primitives for LFM2 bridge projections.

The tokenizer bridge projects cached parent logits into the child's physical
vocabulary.  Those projections are independent of the child instance and can
therefore be reused by the in-memory, staged-reload, and final-destination
evaluations.  This module provides the content-addressed boundary for that
reuse without depending on Python object identity.

It performs no model calls, optimization, pruning, quantization, or parameter
mutation.  Tensors are detached CPU values, and every materialized tensor is
hashed exactly once while an entry is sealed or authenticated.  Aggregate
evidence is then computed exclusively from the stored manifests.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


SEALED_PROJECTION_BINDING_SCHEMA = "neural-forge-lfm2-sealed-projection-binding-v1"
SEALED_PROJECTION_ENTRY_SCHEMA = "neural-forge-lfm2-sealed-projection-entry-v1"
SEALED_PROJECTION_CACHE_SCHEMA = "neural-forge-lfm2-sealed-projection-cache-v1"
SEALED_PROJECTION_EVIDENCE_SCHEMA = "neural-forge-lfm2-sealed-projection-evidence-v1"
LOGIT_PROJECTION_SCHEMA = "neural-forge-lfm2-strict-tokenizer-logit-projection-v1"

_CACHE_TENSOR_DOMAIN = b"neural-forge-lfm2-circuit-tensor-v1\0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")
NO_SELECTIVE_SESSION_EVIDENCE_SHA256 = hashlib.sha256(
    b"neural-forge-lfm2-no-selective-session-v1\0"
).hexdigest()
_PROJECTION_CONTRACT = {
    "source": "verified-full-donor-vocabulary-cached-logits",
    "target_columns": "strict-target-to-donor-row-index-select",
    "topk_space": "target-physical-vocabulary",
    "tie_break": "lower-target-row-first",
    "numeric_token_id_equality_required": False,
}
_ZERO_MODEL_OPERATIONS = {
    "parent_model_calls": 0,
    "child_model_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "pruned_parameters": 0,
    "quantized_parameters": 0,
}
_PROJECTION_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "record_id",
        "split",
        "phase",
        "cache_key",
        "cache_batch_index",
        "bridge_artifact_sha256",
        "alignment_sha256",
        "parent_cache_input_sha256",
        "source_logits_sha256",
        "target_logits",
        "top_k",
        "topk_target_ids",
        "topk_values",
        "winner_to_topk_margins",
        "projection_contract",
        "model_operations",
        "projection_sha256",
    }
)
_BINDING_KEYS = frozenset(
    {
        "schema",
        "cache_key",
        "parent_fingerprint_sha256",
        "corpus_fingerprint_sha256",
        "selective_session_evidence_sha256",
        "bridge_artifact_sha256",
        "bridge_profile_sha256",
        "adapter_evidence_sha256",
        "adapter_records_sha256",
        "mapping_sha256",
        "top_k",
        "expected_record_count",
        "expected_projection_count",
        "projection_inventory",
        "binding_sha256",
    }
)
_ENTRY_MANIFEST_KEYS = frozenset({"schema", "projection_evidence", "entry_sha256"})
_CACHE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "binding",
        "projection_entries",
        "projection_evidence_sha256",
        "model_operations",
        "cache_sha256",
    }
)
_TENSOR_PAYLOAD_KEYS = frozenset(
    {
        "target_logits",
        "topk_target_ids",
        "topk_values",
        "winner_to_topk_margins",
    }
)


class Lfm2SealedProjectionCacheError(ValueError):
    """A projected-logit cache payload cannot be authenticated."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Sealed LFM2 projection caches require PyTorch.") from exc
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
        raise Lfm2SealedProjectionCacheError(
            "Projection cache metadata must be finite canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except Lfm2SealedProjectionCacheError as exc:
        raise Lfm2SealedProjectionCacheError(
            f"{name} must contain canonical JSON values."
        ) from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SealedProjectionCacheError(f"{name} must be a mapping.")
    copied = _json_copy(dict(value), name)
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise Lfm2SealedProjectionCacheError(f"{name} must be a mapping.")
    return copied


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if frozenset(value) != expected:
        raise Lfm2SealedProjectionCacheError(
            f"{name} fields differ from the sealed schema."
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2SealedProjectionCacheError(f"{name} must be nonempty text.")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise Lfm2SealedProjectionCacheError(
            f"{name} must be canonical NFC text without surrounding whitespace."
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2SealedProjectionCacheError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SealedProjectionCacheError(f"{name} must be a nonnegative integer.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise Lfm2SealedProjectionCacheError(f"{name} must be a positive integer.")
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


def _tensor_sha256_from_bytes(
    *,
    dtype_name: str,
    shape: Sequence[int],
    raw: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(_CACHE_TENSOR_DOMAIN)
    digest.update(dtype_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in shape]))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _capture_tensor(value: Any) -> tuple[dict[str, Any], bytes]:
    dtype_name = str(value.dtype).replace("torch.", "")
    shape = [int(size) for size in value.shape]
    raw = _tensor_bytes(value)
    descriptor = {
        "dtype": dtype_name,
        "shape": shape,
        "numel": int(value.numel()),
        "sha256": _tensor_sha256_from_bytes(
            dtype_name=dtype_name,
            shape=shape,
            raw=raw,
        ),
    }
    return descriptor, raw


def _plain_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not torch.is_tensor(value)
        or value.numel() == 0
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
    ):
        raise Lfm2SealedProjectionCacheError(
            f"{name} must be a nonempty detached strided tensor."
        )
    try:
        return value.detach().to(device="cpu").contiguous()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2SealedProjectionCacheError(
            f"{name} cannot be canonicalized as a CPU tensor."
        ) from exc


def _materialize_tensor(raw: bytes, descriptor: Mapping[str, Any], name: str) -> Any:
    torch = _torch()
    selected = _mapping(descriptor, f"{name} descriptor")
    _exact_keys(
        selected,
        frozenset({"dtype", "shape", "numel", "sha256"}),
        f"{name} descriptor",
    )
    dtype_name = _text(selected["dtype"], f"{name} dtype")
    supported = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
    }
    dtype = supported.get(dtype_name)
    if dtype is None:
        raise Lfm2SealedProjectionCacheError(
            f"{name} uses unsupported sealed dtype {dtype_name!r}."
        )
    shape_value = selected["shape"]
    if not isinstance(shape_value, list) or not shape_value:
        raise Lfm2SealedProjectionCacheError(
            f"{name} descriptor shape must be a nonempty list."
        )
    shape = tuple(_positive_int(size, f"{name} shape") for size in shape_value)
    numel = _positive_int(selected["numel"], f"{name} numel")
    expected_numel = 1
    for size in shape:
        expected_numel *= size
    if numel != expected_numel:
        raise Lfm2SealedProjectionCacheError(
            f"{name} descriptor numel differs from its shape."
        )
    _sha256(selected["sha256"], f"{name} sha256")
    expected_bytes = numel * int(torch.empty((), dtype=dtype).element_size())
    if len(raw) != expected_bytes:
        raise Lfm2SealedProjectionCacheError(
            f"{name} immutable bytes differ from their descriptor size."
        )
    try:
        value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2SealedProjectionCacheError(
            f"{name} immutable bytes cannot be materialized."
        ) from exc
    if int(value.numel()) != numel:
        raise Lfm2SealedProjectionCacheError(
            f"{name} materialized tensor has the wrong size."
        )
    return value


def _projection_tensors(
    *,
    target_logits: Any,
    topk_target_ids: Any,
    topk_values: Any,
    winner_to_topk_margins: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    supported_float_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    logits = _plain_tensor(target_logits, "target_logits")
    ids = _plain_tensor(topk_target_ids, "topk_target_ids")
    values = _plain_tensor(topk_values, "topk_values")
    margins = _plain_tensor(
        winner_to_topk_margins,
        "winner_to_topk_margins",
    )
    if (
        logits.ndim != 3
        or logits.dtype not in supported_float_dtypes
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise Lfm2SealedProjectionCacheError(
            "target_logits must be a finite rank-3 supported floating tensor."
        )
    if ids.dtype != torch.long or ids.ndim != 3:
        raise Lfm2SealedProjectionCacheError(
            "topk_target_ids must be a rank-3 torch.long tensor."
        )
    if (
        values.ndim != 3
        or margins.ndim != 3
        or values.dtype != logits.dtype
        or margins.dtype != logits.dtype
        or not bool(torch.isfinite(values).all().item())
        or not bool(torch.isfinite(margins).all().item())
    ):
        raise Lfm2SealedProjectionCacheError(
            "Top-k values and margins must be finite rank-3 tensors matching "
            "target_logits dtype."
        )
    return logits, ids, values, margins


def _validate_projection_semantics(
    *,
    target_logits: Any,
    topk_target_ids: Any,
    topk_values: Any,
    winner_to_topk_margins: Any,
    top_k: int,
) -> None:
    torch = _torch()
    prefix = tuple(target_logits.shape[:-1])
    vocabulary = int(target_logits.shape[-1])
    if top_k < 2 or top_k > vocabulary:
        raise Lfm2SealedProjectionCacheError(
            "top_k must select between two rows and the full target vocabulary."
        )
    if (
        tuple(topk_target_ids.shape) != (*prefix, top_k)
        or tuple(topk_values.shape) != (*prefix, top_k)
        or tuple(winner_to_topk_margins.shape) != (*prefix, top_k - 1)
    ):
        raise Lfm2SealedProjectionCacheError(
            "Projection tensor shapes differ from top_k and target logits."
        )
    if bool((topk_target_ids < 0).any().item()) or bool(
        (topk_target_ids >= vocabulary).any().item()
    ):
        raise Lfm2SealedProjectionCacheError(
            "topk_target_ids contains a row outside the target vocabulary."
        )
    try:
        expected_ids = torch.argsort(
            target_logits,
            dim=-1,
            descending=True,
            stable=True,
        )[..., :top_k].contiguous()
    except TypeError as exc:  # pragma: no cover - supported torch floor
        raise Lfm2SealedProjectionCacheError(
            "Installed PyTorch lacks deterministic stable target top-k."
        ) from exc
    expected_values = target_logits.gather(-1, expected_ids).contiguous()
    expected_margins = (
        expected_values[..., :1] - expected_values[..., 1:]
    ).contiguous()
    if not torch.equal(topk_target_ids, expected_ids):
        raise Lfm2SealedProjectionCacheError(
            "topk_target_ids differs from deterministic target-space ordering."
        )
    if not torch.equal(topk_values, expected_values):
        raise Lfm2SealedProjectionCacheError(
            "topk_values differs from the selected target logits."
        )
    if not torch.equal(winner_to_topk_margins, expected_margins):
        raise Lfm2SealedProjectionCacheError(
            "winner_to_topk_margins differs from the sealed top-k values."
        )


def _projection_evidence_body(
    *,
    record_id: str,
    split: str,
    phase: str,
    cache_key: str,
    cache_batch_index: int,
    bridge_artifact_sha256: str,
    alignment_sha256: str,
    parent_cache_input_sha256: str,
    source_logits_sha256: str,
    top_k: int,
    tensor_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": LOGIT_PROJECTION_SCHEMA,
        "record_id": _text(record_id, "record_id"),
        "split": _text(split, "split"),
        "phase": _text(phase, "phase"),
        "cache_key": _sha256(cache_key, "cache_key"),
        "cache_batch_index": _nonnegative_int(
            cache_batch_index,
            "cache_batch_index",
        ),
        "bridge_artifact_sha256": _sha256(
            bridge_artifact_sha256,
            "bridge_artifact_sha256",
        ),
        "alignment_sha256": _sha256(alignment_sha256, "alignment_sha256"),
        "parent_cache_input_sha256": _sha256(
            parent_cache_input_sha256,
            "parent_cache_input_sha256",
        ),
        "source_logits_sha256": _sha256(
            source_logits_sha256,
            "source_logits_sha256",
        ),
        "target_logits": dict(tensor_descriptors["target_logits"]),
        "top_k": _positive_int(top_k, "top_k"),
        "topk_target_ids": dict(tensor_descriptors["topk_target_ids"]),
        "topk_values": dict(tensor_descriptors["topk_values"]),
        "winner_to_topk_margins": dict(tensor_descriptors["winner_to_topk_margins"]),
        "projection_contract": dict(_PROJECTION_CONTRACT),
        "model_operations": dict(_ZERO_MODEL_OPERATIONS),
    }


def _canonical_projection_inventory(
    keys: Sequence[tuple[int, str]],
    record_count: int,
) -> tuple[tuple[int, str], ...]:
    expected: list[tuple[int, str]] = []
    cursor = 0
    for record_index in range(record_count):
        if cursor >= len(keys) or keys[cursor] != (
            record_index,
            "cached_prefill",
        ):
            raise Lfm2SealedProjectionCacheError(
                "Each record must begin with one cached_prefill projection."
            )
        expected.append(keys[cursor])
        cursor += 1
        decode_index = 0
        while cursor < len(keys) and keys[cursor][0] == record_index:
            phase = keys[cursor][1]
            match = _DECODE_PHASE.fullmatch(phase)
            if match is None or int(match.group(1)) != decode_index:
                raise Lfm2SealedProjectionCacheError(
                    "Decode projections must be contiguous and canonically ordered."
                )
            if phase != f"decode.{decode_index:03d}":
                raise Lfm2SealedProjectionCacheError(
                    "Decode phase spelling is not canonical."
                )
            expected.append(keys[cursor])
            cursor += 1
            decode_index += 1
    if cursor != len(keys):
        raise Lfm2SealedProjectionCacheError(
            "Projection inventory is not grouped in canonical record order."
        )
    return tuple(expected)


@dataclass(frozen=True, slots=True, init=False)
class Lfm2SealedProjectionCacheBinding:
    """Content identity shared by one complete projection inventory."""

    cache_key: str
    parent_fingerprint_sha256: str
    corpus_fingerprint_sha256: str
    selective_session_evidence_sha256: str
    bridge_artifact_sha256: str
    bridge_profile_sha256: str
    adapter_evidence_sha256: str
    adapter_records_sha256: str
    mapping_sha256: str
    top_k: int
    expected_record_count: int
    expected_projection_count: int
    binding_sha256: str
    _projection_keys: tuple[tuple[int, str], ...]

    @property
    def projection_keys(self) -> tuple[tuple[int, str], ...]:
        return self._projection_keys

    def to_evidence_dict(self) -> dict[str, Any]:
        return {
            "schema": SEALED_PROJECTION_BINDING_SCHEMA,
            "cache_key": self.cache_key,
            "parent_fingerprint_sha256": self.parent_fingerprint_sha256,
            "corpus_fingerprint_sha256": self.corpus_fingerprint_sha256,
            "selective_session_evidence_sha256": (
                self.selective_session_evidence_sha256
            ),
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "bridge_profile_sha256": self.bridge_profile_sha256,
            "adapter_evidence_sha256": self.adapter_evidence_sha256,
            "adapter_records_sha256": self.adapter_records_sha256,
            "mapping_sha256": self.mapping_sha256,
            "top_k": self.top_k,
            "expected_record_count": self.expected_record_count,
            "expected_projection_count": self.expected_projection_count,
            "projection_inventory": [
                {"cache_batch_index": batch_index, "phase": phase}
                for batch_index, phase in self._projection_keys
            ],
            "binding_sha256": self.binding_sha256,
        }

    def verify(self) -> Lfm2SealedProjectionCacheBinding:
        return authenticate_lfm2_projection_cache_binding(self.to_evidence_dict())


def seal_lfm2_projection_cache_binding(
    *,
    cache_key: str,
    parent_fingerprint_sha256: str,
    corpus_fingerprint_sha256: str,
    selective_session_evidence_sha256: str | None,
    bridge_artifact_sha256: str,
    bridge_profile_sha256: str,
    adapter_evidence_sha256: str,
    adapter_records_sha256: str,
    mapping_sha256: str,
    top_k: int,
    expected_record_count: int,
    projection_keys: Iterable[tuple[int, str]],
) -> Lfm2SealedProjectionCacheBinding:
    record_count = _positive_int(expected_record_count, "expected_record_count")
    keys: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    covered_records: set[int] = set()
    for raw_key in projection_keys:
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise Lfm2SealedProjectionCacheError(
                "Every projection key must be (cache_batch_index, phase)."
            )
        batch_index = _nonnegative_int(raw_key[0], "cache_batch_index")
        phase = _text(raw_key[1], "phase")
        if batch_index >= record_count:
            raise Lfm2SealedProjectionCacheError(
                "Projection inventory exceeds expected_record_count."
            )
        key = (batch_index, phase)
        if key in seen:
            raise Lfm2SealedProjectionCacheError(
                "Projection inventory contains a duplicate phase key."
            )
        seen.add(key)
        covered_records.add(batch_index)
        keys.append(key)
    if not keys:
        raise Lfm2SealedProjectionCacheError(
            "Projection inventory must contain at least one phase."
        )
    if covered_records != set(range(record_count)):
        raise Lfm2SealedProjectionCacheError(
            "Projection inventory does not cover every cache record."
        )
    canonical_keys = _canonical_projection_inventory(keys, record_count)
    selective_sha = (
        NO_SELECTIVE_SESSION_EVIDENCE_SHA256
        if selective_session_evidence_sha256 is None
        else _sha256(
            selective_session_evidence_sha256,
            "selective_session_evidence_sha256",
        )
    )
    body = {
        "schema": SEALED_PROJECTION_BINDING_SCHEMA,
        "cache_key": _sha256(cache_key, "cache_key"),
        "parent_fingerprint_sha256": _sha256(
            parent_fingerprint_sha256,
            "parent_fingerprint_sha256",
        ),
        "corpus_fingerprint_sha256": _sha256(
            corpus_fingerprint_sha256,
            "corpus_fingerprint_sha256",
        ),
        "selective_session_evidence_sha256": selective_sha,
        "bridge_artifact_sha256": _sha256(
            bridge_artifact_sha256,
            "bridge_artifact_sha256",
        ),
        "bridge_profile_sha256": _sha256(
            bridge_profile_sha256,
            "bridge_profile_sha256",
        ),
        "adapter_evidence_sha256": _sha256(
            adapter_evidence_sha256,
            "adapter_evidence_sha256",
        ),
        "adapter_records_sha256": _sha256(
            adapter_records_sha256,
            "adapter_records_sha256",
        ),
        "mapping_sha256": _sha256(mapping_sha256, "mapping_sha256"),
        "top_k": _positive_int(top_k, "top_k"),
        "expected_record_count": record_count,
        "expected_projection_count": len(keys),
        "projection_inventory": [
            {"cache_batch_index": batch_index, "phase": phase}
            for batch_index, phase in canonical_keys
        ],
    }
    evidence = {**body, "binding_sha256": _canonical_sha256(body)}
    return authenticate_lfm2_projection_cache_binding(evidence)


def authenticate_lfm2_projection_cache_binding(
    evidence: Mapping[str, Any],
) -> Lfm2SealedProjectionCacheBinding:
    value = _mapping(evidence, "binding evidence")
    _exact_keys(value, _BINDING_KEYS, "binding evidence")
    if value["schema"] != SEALED_PROJECTION_BINDING_SCHEMA:
        raise Lfm2SealedProjectionCacheError(
            "Projection binding schema is not supported."
        )
    claimed_sha = _sha256(value["binding_sha256"], "binding_sha256")
    body = dict(value)
    body.pop("binding_sha256")
    if _canonical_sha256(body) != claimed_sha:
        raise Lfm2SealedProjectionCacheError("Projection binding seal is invalid.")
    inventory = value["projection_inventory"]
    if not isinstance(inventory, list):
        raise Lfm2SealedProjectionCacheError("projection_inventory must be a list.")
    keys: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in inventory:
        selected = _mapping(row, "projection inventory row")
        _exact_keys(
            selected,
            frozenset({"cache_batch_index", "phase"}),
            "projection inventory row",
        )
        key = (
            _nonnegative_int(
                selected["cache_batch_index"],
                "cache_batch_index",
            ),
            _text(selected["phase"], "phase"),
        )
        if key in seen:
            raise Lfm2SealedProjectionCacheError(
                "Projection binding contains a duplicate phase key."
            )
        seen.add(key)
        keys.append(key)
    record_count = _positive_int(
        value["expected_record_count"],
        "expected_record_count",
    )
    projection_count = _positive_int(
        value["expected_projection_count"],
        "expected_projection_count",
    )
    if projection_count != len(keys):
        raise Lfm2SealedProjectionCacheError(
            "Projection binding count differs from its inventory."
        )
    if any(index >= record_count for index, _phase_name in keys) or {
        index for index, _phase_name in keys
    } != set(range(record_count)):
        raise Lfm2SealedProjectionCacheError(
            "Projection binding inventory does not cover its record range."
        )
    canonical_keys = _canonical_projection_inventory(keys, record_count)
    result = object.__new__(Lfm2SealedProjectionCacheBinding)
    values = {
        "cache_key": _sha256(value["cache_key"], "cache_key"),
        "parent_fingerprint_sha256": _sha256(
            value["parent_fingerprint_sha256"],
            "parent_fingerprint_sha256",
        ),
        "corpus_fingerprint_sha256": _sha256(
            value["corpus_fingerprint_sha256"],
            "corpus_fingerprint_sha256",
        ),
        "selective_session_evidence_sha256": _sha256(
            value["selective_session_evidence_sha256"],
            "selective_session_evidence_sha256",
        ),
        "bridge_artifact_sha256": _sha256(
            value["bridge_artifact_sha256"],
            "bridge_artifact_sha256",
        ),
        "bridge_profile_sha256": _sha256(
            value["bridge_profile_sha256"],
            "bridge_profile_sha256",
        ),
        "adapter_evidence_sha256": _sha256(
            value["adapter_evidence_sha256"],
            "adapter_evidence_sha256",
        ),
        "adapter_records_sha256": _sha256(
            value["adapter_records_sha256"],
            "adapter_records_sha256",
        ),
        "mapping_sha256": _sha256(value["mapping_sha256"], "mapping_sha256"),
        "top_k": _positive_int(value["top_k"], "top_k"),
        "expected_record_count": record_count,
        "expected_projection_count": projection_count,
        "binding_sha256": claimed_sha,
        "_projection_keys": canonical_keys,
    }
    for name, selected in values.items():
        object.__setattr__(result, name, selected)
    return result


@dataclass(frozen=True, slots=True, init=False)
class Lfm2SealedProjectionEntry:
    """One projection plus an immutable JSON evidence manifest."""

    record_id: str
    split: str
    phase: str
    cache_key: str
    cache_batch_index: int
    bridge_artifact_sha256: str
    top_k: int
    projection_sha256: str
    entry_sha256: str
    _projection_evidence_json: bytes
    _target_logits_bytes: bytes
    _topk_target_ids_bytes: bytes
    _topk_values_bytes: bytes
    _winner_to_topk_margins_bytes: bytes

    @property
    def key(self) -> tuple[int, str]:
        return (self.cache_batch_index, self.phase)

    @property
    def target_logits(self) -> Any:
        evidence = self.projection_evidence_dict()
        return _materialize_tensor(
            self._target_logits_bytes,
            evidence["target_logits"],
            "target_logits",
        )

    @property
    def topk_target_ids(self) -> Any:
        evidence = self.projection_evidence_dict()
        return _materialize_tensor(
            self._topk_target_ids_bytes,
            evidence["topk_target_ids"],
            "topk_target_ids",
        )

    @property
    def topk_values(self) -> Any:
        evidence = self.projection_evidence_dict()
        return _materialize_tensor(
            self._topk_values_bytes,
            evidence["topk_values"],
            "topk_values",
        )

    @property
    def winner_to_topk_margins(self) -> Any:
        evidence = self.projection_evidence_dict()
        return _materialize_tensor(
            self._winner_to_topk_margins_bytes,
            evidence["winner_to_topk_margins"],
            "winner_to_topk_margins",
        )

    def projection_evidence_dict(self) -> dict[str, Any]:
        value = json.loads(self._projection_evidence_json)
        if not isinstance(value, dict):  # pragma: no cover - construction guard
            raise Lfm2SealedProjectionCacheError(
                "Stored projection evidence is not a mapping."
            )
        return value

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema": SEALED_PROJECTION_ENTRY_SCHEMA,
            "projection_evidence": self.projection_evidence_dict(),
            "entry_sha256": self.entry_sha256,
        }

    def to_tensor_payload_dict(self) -> dict[str, Any]:
        return {
            "target_logits": self.target_logits,
            "topk_target_ids": self.topk_target_ids,
            "topk_values": self.topk_values,
            "winner_to_topk_margins": self.winner_to_topk_margins,
        }


def seal_lfm2_materialized_projection_entry(
    *,
    record_id: str,
    split: str,
    phase: str,
    cache_key: str,
    cache_batch_index: int,
    bridge_artifact_sha256: str,
    alignment_sha256: str,
    parent_cache_input_sha256: str,
    source_logits_sha256: str,
    top_k: int,
    target_logits: Any,
    topk_target_ids: Any,
    topk_values: Any,
    winner_to_topk_margins: Any,
    expected_projection_sha256: str | None = None,
) -> Lfm2SealedProjectionEntry:
    logits, ids, values, margins = _projection_tensors(
        target_logits=target_logits,
        topk_target_ids=topk_target_ids,
        topk_values=topk_values,
        winner_to_topk_margins=winner_to_topk_margins,
    )
    requested_top_k = _positive_int(top_k, "top_k")
    _validate_projection_semantics(
        target_logits=logits,
        topk_target_ids=ids,
        topk_values=values,
        winner_to_topk_margins=margins,
        top_k=requested_top_k,
    )
    captured = {
        "target_logits": _capture_tensor(logits),
        "topk_target_ids": _capture_tensor(ids),
        "topk_values": _capture_tensor(values),
        "winner_to_topk_margins": _capture_tensor(margins),
    }
    descriptors = {name: value[0] for name, value in captured.items()}
    body = _projection_evidence_body(
        record_id=record_id,
        split=split,
        phase=phase,
        cache_key=cache_key,
        cache_batch_index=cache_batch_index,
        bridge_artifact_sha256=bridge_artifact_sha256,
        alignment_sha256=alignment_sha256,
        parent_cache_input_sha256=parent_cache_input_sha256,
        source_logits_sha256=source_logits_sha256,
        top_k=requested_top_k,
        tensor_descriptors=descriptors,
    )
    projection_sha = _canonical_sha256(body)
    if expected_projection_sha256 is not None and projection_sha != _sha256(
        expected_projection_sha256,
        "expected_projection_sha256",
    ):
        raise Lfm2SealedProjectionCacheError(
            "Materialized tensors differ from the expected projection seal."
        )
    evidence = {**body, "projection_sha256": projection_sha}
    entry_body = {
        "schema": SEALED_PROJECTION_ENTRY_SCHEMA,
        "projection_evidence": evidence,
    }
    return _new_projection_entry(
        evidence=evidence,
        entry_sha256=_canonical_sha256(entry_body),
        tensor_bytes={name: value[1] for name, value in captured.items()},
    )


def seal_lfm2_projection_entry(projection: Any) -> Lfm2SealedProjectionEntry:
    """Seal a public projected-logit object without accessing private tensors."""

    try:
        public = {
            "record_id": projection.record_id,
            "split": projection.split,
            "phase": projection.phase,
            "cache_key": projection.cache_key,
            "cache_batch_index": projection.cache_batch_index,
            "bridge_artifact_sha256": projection.bridge_artifact_sha256,
            "alignment_sha256": projection.alignment_sha256,
            "parent_cache_input_sha256": projection.parent_cache_input_sha256,
            "source_logits_sha256": projection.source_logits_sha256,
            "top_k": projection.top_k,
            "projection_sha256": projection.projection_sha256,
            "target_logits": projection.target_logits,
            "topk_target_ids": projection.topk_target_ids,
            "topk_values": projection.topk_values,
            "winner_to_topk_margins": projection.winner_to_topk_margins,
        }
    except AttributeError as exc:
        raise Lfm2SealedProjectionCacheError(
            "projection does not expose the authenticated public projection API."
        ) from exc
    return seal_lfm2_materialized_projection_entry(
        record_id=public["record_id"],
        split=public["split"],
        phase=public["phase"],
        cache_key=public["cache_key"],
        cache_batch_index=public["cache_batch_index"],
        bridge_artifact_sha256=public["bridge_artifact_sha256"],
        alignment_sha256=public["alignment_sha256"],
        parent_cache_input_sha256=public["parent_cache_input_sha256"],
        source_logits_sha256=public["source_logits_sha256"],
        top_k=public["top_k"],
        target_logits=public["target_logits"],
        topk_target_ids=public["topk_target_ids"],
        topk_values=public["topk_values"],
        winner_to_topk_margins=public["winner_to_topk_margins"],
        expected_projection_sha256=public["projection_sha256"],
    )


def _new_projection_entry(
    *,
    evidence: Mapping[str, Any],
    entry_sha256: str,
    tensor_bytes: Mapping[str, bytes],
) -> Lfm2SealedProjectionEntry:
    result = object.__new__(Lfm2SealedProjectionEntry)
    values = {
        "record_id": evidence["record_id"],
        "split": evidence["split"],
        "phase": evidence["phase"],
        "cache_key": evidence["cache_key"],
        "cache_batch_index": evidence["cache_batch_index"],
        "bridge_artifact_sha256": evidence["bridge_artifact_sha256"],
        "top_k": evidence["top_k"],
        "projection_sha256": evidence["projection_sha256"],
        "entry_sha256": entry_sha256,
        "_projection_evidence_json": _canonical_json(dict(evidence)),
        "_target_logits_bytes": bytes(tensor_bytes["target_logits"]),
        "_topk_target_ids_bytes": bytes(tensor_bytes["topk_target_ids"]),
        "_topk_values_bytes": bytes(tensor_bytes["topk_values"]),
        "_winner_to_topk_margins_bytes": bytes(tensor_bytes["winner_to_topk_margins"]),
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def authenticate_lfm2_materialized_projection_entry(
    *,
    projection_evidence: Mapping[str, Any],
    entry_sha256: str,
    target_logits: Any,
    topk_target_ids: Any,
    topk_values: Any,
    winner_to_topk_margins: Any,
) -> Lfm2SealedProjectionEntry:
    evidence = _mapping(projection_evidence, "projection evidence")
    _exact_keys(evidence, _PROJECTION_EVIDENCE_KEYS, "projection evidence")
    if evidence["schema"] != LOGIT_PROJECTION_SCHEMA:
        raise Lfm2SealedProjectionCacheError(
            "Projection evidence schema is not supported."
        )
    claimed_projection_sha = _sha256(
        evidence["projection_sha256"],
        "projection_sha256",
    )
    projection_body = dict(evidence)
    projection_body.pop("projection_sha256")
    if _canonical_sha256(projection_body) != claimed_projection_sha:
        raise Lfm2SealedProjectionCacheError("Projection evidence seal is invalid.")
    if evidence["projection_contract"] != _PROJECTION_CONTRACT:
        raise Lfm2SealedProjectionCacheError(
            "Projection contract differs from the strict target-space policy."
        )
    if evidence["model_operations"] != _ZERO_MODEL_OPERATIONS:
        raise Lfm2SealedProjectionCacheError(
            "Projection evidence reports a forbidden model operation."
        )
    logits, ids, values, margins = _projection_tensors(
        target_logits=target_logits,
        topk_target_ids=topk_target_ids,
        topk_values=topk_values,
        winner_to_topk_margins=winner_to_topk_margins,
    )
    top_k = _positive_int(evidence["top_k"], "top_k")
    _validate_projection_semantics(
        target_logits=logits,
        topk_target_ids=ids,
        topk_values=values,
        winner_to_topk_margins=margins,
        top_k=top_k,
    )
    captured = {
        "target_logits": _capture_tensor(logits),
        "topk_target_ids": _capture_tensor(ids),
        "topk_values": _capture_tensor(values),
        "winner_to_topk_margins": _capture_tensor(margins),
    }
    descriptors = {name: value[0] for name, value in captured.items()}
    for name, descriptor in descriptors.items():
        if evidence[name] != descriptor:
            raise Lfm2SealedProjectionCacheError(
                f"{name} differs from its sealed projection descriptor."
            )
    _text(evidence["record_id"], "record_id")
    _text(evidence["split"], "split")
    _text(evidence["phase"], "phase")
    _sha256(evidence["cache_key"], "cache_key")
    _nonnegative_int(evidence["cache_batch_index"], "cache_batch_index")
    for name in (
        "bridge_artifact_sha256",
        "alignment_sha256",
        "parent_cache_input_sha256",
        "source_logits_sha256",
    ):
        _sha256(evidence[name], name)
    claimed_entry_sha = _sha256(entry_sha256, "entry_sha256")
    entry_body = {
        "schema": SEALED_PROJECTION_ENTRY_SCHEMA,
        "projection_evidence": evidence,
    }
    if _canonical_sha256(entry_body) != claimed_entry_sha:
        raise Lfm2SealedProjectionCacheError("Projection entry seal is invalid.")
    return _new_projection_entry(
        evidence=evidence,
        entry_sha256=claimed_entry_sha,
        tensor_bytes={name: value[1] for name, value in captured.items()},
    )


@dataclass(frozen=True, slots=True, init=False)
class Lfm2SealedProjectionCache:
    """An ordered, content-addressed collection of projected parent logits."""

    binding: Lfm2SealedProjectionCacheBinding
    projection_evidence_sha256: str
    cache_sha256: str
    _entries: tuple[Lfm2SealedProjectionEntry, ...]
    _entry_ordinals: Mapping[tuple[int, str], int]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[Lfm2SealedProjectionEntry, ...]:
        return self._entries

    def entry_for_key(
        self,
        cache_batch_index: int,
        phase: str,
    ) -> Lfm2SealedProjectionEntry:
        key = (
            _nonnegative_int(cache_batch_index, "cache_batch_index"),
            _text(phase, "phase"),
        )
        ordinal = self._entry_ordinals.get(key)
        if ordinal is None:
            raise Lfm2SealedProjectionCacheError(
                "Projection cache does not contain the requested phase key."
            )
        return self._entries[ordinal]

    def load_target_logits(self, cache_batch_index: int, phase: str) -> Any:
        """Materialize only one requested target-logit tensor."""

        return self.entry_for_key(cache_batch_index, phase).target_logits

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema": SEALED_PROJECTION_CACHE_SCHEMA,
            "binding": self.binding.to_evidence_dict(),
            "projection_entries": [entry.to_manifest_dict() for entry in self._entries],
            "projection_evidence_sha256": self.projection_evidence_sha256,
            "model_operations": dict(_ZERO_MODEL_OPERATIONS),
            "cache_sha256": self.cache_sha256,
        }

    def to_materialized_payload(self) -> dict[str, Any]:
        return {
            "manifest": self.to_manifest_dict(),
            "tensors": [entry.to_tensor_payload_dict() for entry in self._entries],
        }


def _projection_evidence_sha256(
    binding_sha256: str,
    entries: Sequence[Lfm2SealedProjectionEntry],
) -> str:
    return _canonical_sha256(
        {
            "schema": SEALED_PROJECTION_EVIDENCE_SCHEMA,
            "binding_sha256": binding_sha256,
            "projections": [entry.projection_evidence_dict() for entry in entries],
        }
    )


def seal_lfm2_projection_cache(
    *,
    binding: Lfm2SealedProjectionCacheBinding,
    entries: Iterable[Lfm2SealedProjectionEntry],
) -> Lfm2SealedProjectionCache:
    if not isinstance(binding, Lfm2SealedProjectionCacheBinding):
        raise Lfm2SealedProjectionCacheError(
            "binding must be an authenticated projection cache binding."
        )
    binding = binding.verify()
    selected = tuple(entries)
    if not selected or any(
        not isinstance(entry, Lfm2SealedProjectionEntry) for entry in selected
    ):
        raise Lfm2SealedProjectionCacheError(
            "entries must contain authenticated sealed projections."
        )
    if tuple(entry.key for entry in selected) != binding.projection_keys:
        raise Lfm2SealedProjectionCacheError(
            "Projection entry order differs from the binding inventory."
        )
    if len(selected) != binding.expected_projection_count:
        raise Lfm2SealedProjectionCacheError(
            "Projection entry count differs from the binding."
        )
    if len({entry.key for entry in selected}) != len(selected):
        raise Lfm2SealedProjectionCacheError(
            "Projection cache contains duplicate phase keys."
        )
    for entry in selected:
        if (
            entry.cache_key != binding.cache_key
            or entry.bridge_artifact_sha256 != binding.bridge_artifact_sha256
            or entry.top_k != binding.top_k
        ):
            raise Lfm2SealedProjectionCacheError(
                "Projection entry differs from the cache binding."
            )
    evidence_sha = _projection_evidence_sha256(
        binding.binding_sha256,
        selected,
    )
    body = {
        "schema": SEALED_PROJECTION_CACHE_SCHEMA,
        "binding": binding.to_evidence_dict(),
        "projection_entries": [entry.to_manifest_dict() for entry in selected],
        "projection_evidence_sha256": evidence_sha,
        "model_operations": dict(_ZERO_MODEL_OPERATIONS),
    }
    result = object.__new__(Lfm2SealedProjectionCache)
    object.__setattr__(result, "binding", binding)
    object.__setattr__(result, "projection_evidence_sha256", evidence_sha)
    object.__setattr__(result, "cache_sha256", _canonical_sha256(body))
    object.__setattr__(result, "_entries", selected)
    object.__setattr__(
        result,
        "_entry_ordinals",
        MappingProxyType(
            {entry.key: ordinal for ordinal, entry in enumerate(selected)}
        ),
    )
    return result


def authenticate_lfm2_materialized_projection_cache(
    payload: Mapping[str, Any],
    *,
    expected_cache_sha256: str,
) -> Lfm2SealedProjectionCache:
    if not isinstance(payload, Mapping):
        raise Lfm2SealedProjectionCacheError(
            "Materialized projection cache payload must be a mapping."
        )
    if frozenset(payload) != frozenset({"manifest", "tensors"}):
        raise Lfm2SealedProjectionCacheError(
            "Materialized projection cache payload fields differ from schema."
        )
    manifest = _mapping(payload["manifest"], "projection cache manifest")
    _exact_keys(manifest, _CACHE_MANIFEST_KEYS, "projection cache manifest")
    if manifest["schema"] != SEALED_PROJECTION_CACHE_SCHEMA:
        raise Lfm2SealedProjectionCacheError(
            "Projection cache schema is not supported."
        )
    expected_sha = _sha256(expected_cache_sha256, "expected_cache_sha256")
    claimed_sha = _sha256(manifest["cache_sha256"], "cache_sha256")
    if claimed_sha != expected_sha:
        raise Lfm2SealedProjectionCacheError(
            "Projection cache does not match the expected external seal."
        )
    if manifest["model_operations"] != _ZERO_MODEL_OPERATIONS:
        raise Lfm2SealedProjectionCacheError(
            "Projection cache reports a forbidden model operation."
        )
    binding = authenticate_lfm2_projection_cache_binding(manifest["binding"])
    entry_manifests = manifest["projection_entries"]
    tensor_payloads = payload["tensors"]
    if not isinstance(entry_manifests, list) or not isinstance(
        tensor_payloads,
        Sequence,
    ):
        raise Lfm2SealedProjectionCacheError(
            "Projection entries and tensors must be ordered sequences."
        )
    if isinstance(tensor_payloads, (str, bytes, bytearray)):
        raise Lfm2SealedProjectionCacheError(
            "Projection tensor payloads must not be text or bytes."
        )
    if len(entry_manifests) != len(tensor_payloads):
        raise Lfm2SealedProjectionCacheError(
            "Projection manifest and tensor counts differ."
        )
    entries: list[Lfm2SealedProjectionEntry] = []
    for ordinal, (entry_manifest, tensor_payload) in enumerate(
        zip(entry_manifests, tensor_payloads)
    ):
        selected_manifest = _mapping(
            entry_manifest,
            f"projection entry manifest {ordinal}",
        )
        _exact_keys(
            selected_manifest,
            _ENTRY_MANIFEST_KEYS,
            f"projection entry manifest {ordinal}",
        )
        if selected_manifest["schema"] != SEALED_PROJECTION_ENTRY_SCHEMA:
            raise Lfm2SealedProjectionCacheError(
                "Projection entry schema is not supported."
            )
        if not isinstance(tensor_payload, Mapping):
            raise Lfm2SealedProjectionCacheError(
                "Projection tensor payload must be a mapping."
            )
        if frozenset(tensor_payload) != _TENSOR_PAYLOAD_KEYS:
            raise Lfm2SealedProjectionCacheError(
                "Projection tensor payload fields differ from schema."
            )
        entries.append(
            authenticate_lfm2_materialized_projection_entry(
                projection_evidence=selected_manifest["projection_evidence"],
                entry_sha256=selected_manifest["entry_sha256"],
                target_logits=tensor_payload["target_logits"],
                topk_target_ids=tensor_payload["topk_target_ids"],
                topk_values=tensor_payload["topk_values"],
                winner_to_topk_margins=tensor_payload["winner_to_topk_margins"],
            )
        )
    result = seal_lfm2_projection_cache(binding=binding, entries=entries)
    if result.projection_evidence_sha256 != _sha256(
        manifest["projection_evidence_sha256"],
        "projection_evidence_sha256",
    ):
        raise Lfm2SealedProjectionCacheError(
            "Projection aggregate evidence seal is invalid."
        )
    if result.cache_sha256 != claimed_sha:
        raise Lfm2SealedProjectionCacheError(
            "Projection cache manifest seal is invalid."
        )
    if _canonical_json(result.to_manifest_dict()) != _canonical_json(manifest):
        raise Lfm2SealedProjectionCacheError(
            "Projection cache manifest changed during authentication."
        )
    return result


__all__ = [
    "LOGIT_PROJECTION_SCHEMA",
    "NO_SELECTIVE_SESSION_EVIDENCE_SHA256",
    "SEALED_PROJECTION_BINDING_SCHEMA",
    "SEALED_PROJECTION_CACHE_SCHEMA",
    "SEALED_PROJECTION_ENTRY_SCHEMA",
    "Lfm2SealedProjectionCache",
    "Lfm2SealedProjectionCacheBinding",
    "Lfm2SealedProjectionCacheError",
    "Lfm2SealedProjectionEntry",
    "authenticate_lfm2_materialized_projection_cache",
    "authenticate_lfm2_materialized_projection_entry",
    "authenticate_lfm2_projection_cache_binding",
    "seal_lfm2_materialized_projection_entry",
    "seal_lfm2_projection_cache",
    "seal_lfm2_projection_cache_binding",
    "seal_lfm2_projection_entry",
]
