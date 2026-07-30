"""Cache-only causal span transport for a reduced LFM2 token interface.

A row projection cannot transport a donor token whose target representation is
an autoregressive token *sequence*.  This module compiles the smallest useful
counterpart: a root-specific trie whose edges contain exact donor probability
mass and whose nodes expose target-token conditionals after an emitted span
prefix.  The resulting state is owned by the child and can be serialized with
it; neither compilation nor inference calls the parent.

This is deliberately an isolated first slice.  Suffix rewrites and the next
donor conditional after a decomposition terminates are authenticated and
reported, never invented.  The compiled head is therefore exact on supported
append-only trie edges and refuses unresolved roots or ambiguous terminal
nodes unless the caller explicitly opts into those limitations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    StrictTokenizerBridgeError,
)


LFM2_CAUSAL_SPAN_ROOT_LOGITS_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-causal-span-root-logits-evidence-v1"
)
LFM2_FULL_DONOR_SPAN_TRANSITION_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-full-donor-span-transition-evidence-v1"
)
LFM2_CAUSAL_SPAN_TOKEN_BRIDGE_PLAN_SCHEMA = (
    "neural-forge-lfm2-causal-span-token-bridge-plan-v1"
)
LFM2_CAUSAL_SPAN_HEAD_RUNTIME_SCHEMA = (
    "neural-forge-lfm2-causal-span-head-runtime-v1"
)
LFM2_CAUSAL_SPAN_CONTEXTUAL_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-causal-span-contextual-progress-v1"
)

_APPEND_KINDS = frozenset(
    {
        "append",
        "canonical-contextual-append",
        "authenticated-special-control",
    }
)
_LIMITATION_KINDS = frozenset(
    {
        "suffix-rewrite",
        "zero-width",
        "unmapped-control",
        "nonappend-only",
        "roundtrip-mismatch",
        "unsupported",
    }
)


class Lfm2CausalSpanTokenBridgeError(RuntimeError):
    """Sealed span evidence, compilation, or runtime state is invalid."""


def _dtype_name(value: Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().to(device="cpu").contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_sha256(value: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-causal-span-tensor-v1\0")
    digest.update(_dtype_name(value).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        str(tuple(int(size) for size in value.shape)).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CausalSpanTokenBridgeError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CausalSpanTokenBridgeError(f"{name} must be nonnegative.")
    return int(value)


def _integer_tuple(
    value: Any,
    name: str,
    *,
    allow_empty: bool,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2CausalSpanTokenBridgeError(f"{name} must be an ID sequence.")
    result = tuple(value)
    if not allow_empty and not result:
        raise Lfm2CausalSpanTokenBridgeError(f"{name} cannot be empty.")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        or (upper_bound is not None and item >= upper_bound)
        for item in result
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            f"{name} contains an invalid token ID."
        )
    return tuple(int(item) for item in result)


def _validated_logits(value: Any) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
        or min(int(size) for size in value.shape) <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            "Donor root logits must be one detached finite dense float matrix."
        )
    return value


def _sealed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(body)
    value[field] = canonical_sha256(value)
    return value


def _verify_sealed_mapping(
    value: Mapping[str, Any],
    *,
    schema: str | None,
    preferred_fields: Sequence[str],
    name: str,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(value, Mapping):
        raise Lfm2CausalSpanTokenBridgeError(f"{name} must be a sealed mapping.")
    document = dict(value)
    present = tuple(field for field in preferred_fields if field in document)
    if len(present) != 1:
        raise Lfm2CausalSpanTokenBridgeError(
            f"{name} must contain exactly one recognized seal field."
        )
    field = present[0]
    supplied = _digest(document[field], f"{name} {field}")
    body = dict(document)
    body.pop(field)
    if supplied != canonical_sha256(body) or (
        schema is not None and body.get("schema") != schema
    ):
        raise Lfm2CausalSpanTokenBridgeError(f"{name} seal is invalid.")
    return document, field, supplied


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high = max(left, right)
    low = min(left, right)
    return high + math.log1p(math.exp(low - high))


def _logsumexp(values: Sequence[float]) -> float:
    result = -math.inf
    for value in values:
        result = _logaddexp(result, float(value))
    return result


def seal_lfm2_causal_span_root_logits_evidence(
    donor_logits: Tensor,
    *,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal already-cached root logits without executing any model.

    Production callers may instead supply the existing sealed terminal-cache
    receipt directly to ``build_lfm2_full_donor_span_transition_evidence``.
    This small wrapper is useful when a cache reader exposes only a tensor and
    immutable lineage identifiers.
    """

    logits = _validated_logits(donor_logits)
    lineage_value = {} if lineage is None else dict(lineage)
    # Prove the lineage itself is canonical JSON before placing it under seal.
    canonical_sha256(lineage_value)
    body = {
        "schema": LFM2_CAUSAL_SPAN_ROOT_LOGITS_EVIDENCE_SCHEMA,
        "status": "sealed-existing-root-logits-no-model-execution",
        "lineage": lineage_value,
        "root_row_count": int(logits.shape[0]),
        "donor_row_count": int(logits.shape[1]),
        "donor_logits_sha256": _tensor_sha256(logits),
        "source_dtype": _dtype_name(logits),
        "source_device_type": logits.device.type,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "network_access": False,
        },
    }
    return _sealed(body, "evidence_sha256")


def _normalize_transition(
    value: Any,
    *,
    expected_donor_id: int,
    target_row_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CausalSpanTokenBridgeError(
            "Each contextual donor transition must be a mapping."
        )
    donor_id = value.get("donor_id", expected_donor_id)
    if donor_id != expected_donor_id:
        raise Lfm2CausalSpanTokenBridgeError(
            "Contextual donor transitions must be in exact donor-row order."
        )
    raw_kind = value.get("kind", value.get("transition_kind"))
    reason = value.get("reason")
    if raw_kind in _APPEND_KINDS:
        kind = "append"
    elif raw_kind in _LIMITATION_KINDS:
        kind = str(raw_kind)
    elif reason == "target-suffix-rewrite":
        kind = "suffix-rewrite"
    elif reason in {
        "zero-width-target-transition",
        "unmapped-or-ambiguous-donor-control",
        "donor-text-nonappend-only",
        "target-roundtrip-mismatch",
    }:
        kind = {
            "zero-width-target-transition": "zero-width",
            "unmapped-or-ambiguous-donor-control": "unmapped-control",
            "donor-text-nonappend-only": "nonappend-only",
            "target-roundtrip-mismatch": "roundtrip-mismatch",
        }[str(reason)]
    else:
        raise Lfm2CausalSpanTokenBridgeError(
            f"Donor row {expected_donor_id} has no recognized transition kind."
        )

    if kind == "append":
        token_ids = _integer_tuple(
            value.get("target_append_token_ids", value.get("target_token_ids")),
            f"donor row {expected_donor_id} append IDs",
            allow_empty=False,
            upper_bound=target_row_count,
        )
        result = {
            "donor_id": expected_donor_id,
            "kind": "append",
            "target_append_token_ids": list(token_ids),
        }
        if raw_kind == "authenticated-special-control":
            result["append_classification"] = "authenticated-special-control"
        candidate_text_sha256 = value.get("candidate_text_sha256")
        if candidate_text_sha256 is not None:
            result["candidate_text_sha256"] = _digest(
                candidate_text_sha256,
                f"donor row {expected_donor_id} candidate_text_sha256",
            )
        return result

    result: dict[str, Any] = {
        "donor_id": expected_donor_id,
        "kind": kind,
        "reason": str(reason if reason is not None else kind),
    }
    candidate = value.get("candidate_target_token_ids")
    if candidate is not None:
        result["candidate_target_token_ids"] = list(
            _integer_tuple(
                candidate,
                f"donor row {expected_donor_id} candidate target IDs",
                allow_empty=True,
                upper_bound=target_row_count,
            )
        )
    candidate_text_sha256 = value.get("candidate_text_sha256")
    if candidate_text_sha256 is not None:
        result["candidate_text_sha256"] = _digest(
            candidate_text_sha256,
            f"donor row {expected_donor_id} candidate_text_sha256",
        )
    if kind == "suffix-rewrite":
        common = _nonnegative_int(
            value.get("target_common_prefix_tokens", 0),
            "target_common_prefix_tokens",
        )
        replaced = _nonnegative_int(
            value.get("target_replaced_root_suffix_tokens", 0),
            "target_replaced_root_suffix_tokens",
        )
        result.update(
            {
                "target_common_prefix_tokens": common,
                "target_replaced_root_suffix_tokens": replaced,
            }
        )
    return result


def build_lfm2_full_donor_span_transition_evidence(
    donor_logits: Tensor,
    *,
    parent_root_logits_evidence: Mapping[str, Any],
    root_terminal_row_indices: Sequence[int],
    target_row_count: int,
    root_target_prefix_token_ids: Sequence[Sequence[int]] | None = None,
    contextual_transitions: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    donor_target_decompositions: Sequence[Sequence[int]] | None = None,
    donor_defined_row_count: int | None = None,
) -> dict[str, Any]:
    """Authenticate complete donor-row span evidence for every causal root.

    Exactly one source is accepted:

    * root-specific ``contextual_transitions`` including explicit limitation
      rows, or
    * one complete donor decomposition inventory shared by all roots.

    The latter is useful for structural controls but is labelled context-free;
    it is never misreported as contextual tokenizer evidence.
    """

    logits = _validated_logits(donor_logits)
    parent, parent_seal_field, parent_sha256 = _verify_sealed_mapping(
        parent_root_logits_evidence,
        schema=None,
        preferred_fields=("evidence_sha256", "result_sha256"),
        name="parent root-logit evidence",
    )
    observed_logits_sha256 = _tensor_sha256(logits)
    if parent.get("donor_logits_sha256") not in {None, observed_logits_sha256}:
        raise Lfm2CausalSpanTokenBridgeError(
            "Parent evidence is sealed around different donor logits."
        )
    for field, observed in (
        ("root_row_count", int(logits.shape[0])),
        ("terminal_row_count", int(logits.shape[0])),
        ("donor_row_count", int(logits.shape[1])),
        ("donor_vocabulary_size", int(logits.shape[1])),
    ):
        if field in parent and parent[field] != observed:
            raise Lfm2CausalSpanTokenBridgeError(
                f"Parent evidence {field} differs from the supplied logits."
            )

    target_count = _positive_int(target_row_count, "target_row_count")
    donor_capacity = int(logits.shape[1])
    donor_defined = (
        donor_capacity
        if donor_defined_row_count is None
        else _positive_int(donor_defined_row_count, "donor_defined_row_count")
    )
    if donor_defined > donor_capacity:
        raise Lfm2CausalSpanTokenBridgeError(
            "Defined donor rows exceed the cached donor-logit width."
        )
    roots = tuple(root_terminal_row_indices)
    if not roots or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= int(logits.shape[0])
        for value in roots
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            "Root terminal row indices must address cached logit rows."
        )
    if root_target_prefix_token_ids is None:
        root_prefixes = ((),) * len(roots)
    else:
        if len(root_target_prefix_token_ids) != len(roots):
            raise Lfm2CausalSpanTokenBridgeError(
                "Target root prefixes must align with root terminal rows."
            )
        root_prefixes = tuple(
            _integer_tuple(
                value,
                f"target root prefix {index}",
                allow_empty=True,
                upper_bound=target_count,
            )
            for index, value in enumerate(root_target_prefix_token_ids)
        )

    if (contextual_transitions is None) == (donor_target_decompositions is None):
        raise Lfm2CausalSpanTokenBridgeError(
            "Supply exactly one of contextual_transitions or "
            "donor_target_decompositions."
        )

    root_records: list[dict[str, Any]] = []
    global_decompositions: list[list[int]] | None = None
    if contextual_transitions is not None:
        if len(contextual_transitions) != len(roots):
            raise Lfm2CausalSpanTokenBridgeError(
                "Contextual transition roots do not align with cached roots."
            )
        for root_index, (terminal_index, prefix, raw_rows) in enumerate(
            zip(roots, root_prefixes, contextual_transitions, strict=True)
        ):
            if len(raw_rows) != donor_defined:
                raise Lfm2CausalSpanTokenBridgeError(
                    "Every contextual root must classify every defined donor row."
                )
            transitions = [
                _normalize_transition(
                    raw,
                    expected_donor_id=donor_id,
                    target_row_count=target_count,
                )
                for donor_id, raw in enumerate(raw_rows)
            ]
            transition_sha256 = canonical_sha256(transitions)
            root_records.append(
                {
                    "root_index": root_index,
                    "terminal_row_index": int(terminal_index),
                    "target_root_token_ids": list(prefix),
                    "transition_count": len(transitions),
                    "transitions": transitions,
                    "transitions_sha256": transition_sha256,
                }
            )
        source_kind = "full-donor-root-contextual-transitions"
        contextual = True
    else:
        if len(donor_target_decompositions or ()) != donor_defined:
            raise Lfm2CausalSpanTokenBridgeError(
                "The decomposition inventory must cover every defined donor row."
            )
        global_decompositions = [
            list(
                _integer_tuple(
                    raw,
                    f"donor decomposition {donor_id}",
                    allow_empty=False,
                    upper_bound=target_count,
                )
            )
            for donor_id, raw in enumerate(donor_target_decompositions or ())
        ]
        for root_index, (terminal_index, prefix) in enumerate(
            zip(roots, root_prefixes, strict=True)
        ):
            root_records.append(
                {
                    "root_index": root_index,
                    "terminal_row_index": int(terminal_index),
                    "target_root_token_ids": list(prefix),
                }
            )
        source_kind = "full-donor-global-decomposition-inventory"
        contextual = False

    body: dict[str, Any] = {
        "schema": LFM2_FULL_DONOR_SPAN_TRANSITION_EVIDENCE_SCHEMA,
        "status": "complete-defined-donor-span-evidence-sealed",
        "source_kind": source_kind,
        "root_specific_contextual_transitions": contextual,
        "parent_root_logits_evidence_sha256": parent_sha256,
        "parent_root_logits_seal_field": parent_seal_field,
        "donor_logits_sha256": observed_logits_sha256,
        "source_dtype": _dtype_name(logits),
        "source_device_type": logits.device.type,
        "root_count": len(root_records),
        "donor_row_count": donor_capacity,
        "donor_defined_row_count": donor_defined,
        "undefined_donor_row_count": donor_capacity - donor_defined,
        "target_row_count": target_count,
        "roots": root_records,
        "every_defined_donor_row_classified_once_per_root": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "network_access": False,
        },
    }
    if global_decompositions is not None:
        body["donor_target_decompositions"] = global_decompositions
        body["donor_target_decompositions_sha256"] = canonical_sha256(
            global_decompositions
        )
    return _sealed(body, "evidence_sha256")


def _live_tokenizer_row_count(tokenizer: Any, name: str) -> int:
    try:
        value = len(tokenizer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CausalSpanTokenBridgeError(
            f"{name} tokenizer must expose a positive row count."
        ) from exc
    return _positive_int(value, f"{name} tokenizer row count")


def _normalized_causal_root_records(
    records: Sequence[Mapping[str, Any]],
    *,
    donor_row_count: int,
    target_row_count: int,
    terminal_row_count: int,
) -> tuple[dict[str, Any], ...]:
    """Canonicalize the cache collector's exact causal-root record format."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise Lfm2CausalSpanTokenBridgeError(
            "causal_root_records must be a nonempty sequence."
        )
    result: list[dict[str, Any]] = []
    seen_terminal_rows: set[int] = set()
    integer_fields = (
        "terminal_row_index",
        "batch_index",
        "phase_row_index",
        "donor_boundary_position",
        "target_boundary_position",
    )
    for root_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Lfm2CausalSpanTokenBridgeError(
                f"Causal root {root_index} must be a mapping."
            )
        normalized: dict[str, Any] = {}
        for field in integer_fields:
            normalized[field] = _nonnegative_int(
                record.get(field),
                f"causal root {root_index} {field}",
            )
        terminal_index = normalized["terminal_row_index"]
        if terminal_index >= terminal_row_count:
            raise Lfm2CausalSpanTokenBridgeError(
                f"Causal root {root_index} leaves the cached terminal rows."
            )
        if terminal_index in seen_terminal_rows:
            raise Lfm2CausalSpanTokenBridgeError(
                "Causal roots must not repeat one terminal logit row."
            )
        seen_terminal_rows.add(terminal_index)
        phase = record.get("phase")
        if not isinstance(phase, str) or not phase:
            raise Lfm2CausalSpanTokenBridgeError(
                f"Causal root {root_index} phase must be nonempty text."
            )
        normalized["phase"] = phase
        donor_prefix = _integer_tuple(
            record.get("donor_prefix_token_ids"),
            f"causal root {root_index} donor prefix",
            allow_empty=False,
            upper_bound=donor_row_count,
        )
        target_prefix = _integer_tuple(
            record.get("target_prefix_token_ids"),
            f"causal root {root_index} target prefix",
            allow_empty=False,
            upper_bound=target_row_count,
        )
        if (
            normalized["donor_boundary_position"] != len(donor_prefix) - 1
            or normalized["target_boundary_position"] != len(target_prefix) - 1
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                f"Causal root {root_index} prefixes are not terminal at their "
                "declared boundaries."
            )
        normalized["donor_prefix_token_ids"] = list(donor_prefix)
        normalized["target_prefix_token_ids"] = list(target_prefix)
        normalized["semantic_alignment_sha256"] = _digest(
            record.get("semantic_alignment_sha256"),
            f"causal root {root_index} semantic_alignment_sha256",
        )
        result.append(normalized)
    if not result:
        raise Lfm2CausalSpanTokenBridgeError(
            "At least one causal root record is required."
        )
    return tuple(result)


def _decode_exact(tokenizer: Any, token_ids: Sequence[int], name: str) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise Lfm2CausalSpanTokenBridgeError(
            f"{name} tokenizer must expose decode()."
        )
    try:
        value = decoder(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:  # pragma: no cover - legacy tokenizer boundary
        value = decoder(list(token_ids))
    if not isinstance(value, str):
        raise Lfm2CausalSpanTokenBridgeError(
            f"{name} tokenizer decode() must return text."
        )
    return value


def _batch_decode_exact(
    tokenizer: Any,
    rows: Sequence[Sequence[int]],
    *,
    name: str,
    audit: dict[str, Any],
) -> tuple[str, ...]:
    if not rows:
        return ()
    sequences = [list(value) for value in rows]
    batch_decoder = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decoder):
        try:
            values = batch_decoder(
                sequences,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:  # pragma: no cover - legacy tokenizer boundary
            try:
                values = batch_decoder(
                    sequences,
                    skip_special_tokens=False,
                )
            except (TypeError, NotImplementedError, AttributeError):
                values = None
        except (NotImplementedError, AttributeError):
            values = None
        if values is not None:
            if (
                isinstance(values, (str, bytes))
                or not isinstance(values, Sequence)
                or len(values) != len(sequences)
                or any(not isinstance(value, str) for value in values)
            ):
                raise Lfm2CausalSpanTokenBridgeError(
                    f"{name} tokenizer batch_decode() returned invalid text rows."
                )
            key = f"{name}_batch_decode_calls"
            audit[key] = int(audit.get(key, 0)) + 1
            audit.setdefault("tokenizer_methods", set()).add(
                f"{name}.batch_decode"
            )
            return tuple(values)
    key = f"{name}_single_decode_calls"
    audit[key] = int(audit.get(key, 0)) + len(sequences)
    audit.setdefault("tokenizer_methods", set()).add(f"{name}.decode")
    return tuple(_decode_exact(tokenizer, value, name) for value in sequences)


def _normalized_encoded_rows(
    value: Any,
    *,
    expected_count: int,
    target_row_count: int,
    source: str,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != expected_count
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            f"{source} returned invalid batch input_ids."
        )
    return tuple(
        _integer_tuple(
            row,
            f"{source} row {row_index}",
            allow_empty=True,
            upper_bound=target_row_count,
        )
        for row_index, row in enumerate(value)
    )


def _batch_encode_exact(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    target_row_count: int,
    audit: dict[str, Any],
) -> tuple[tuple[int, ...], ...]:
    if not texts:
        return ()
    values = list(texts)
    if any(not isinstance(value, str) for value in values):
        raise Lfm2CausalSpanTokenBridgeError(
            "Target tokenizer input rows must be text."
        )
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                values,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
        except (TypeError, NotImplementedError, AttributeError):
            encoded = None
        if encoded is not None:
            result = _normalized_encoded_rows(
                encoded,
                expected_count=len(values),
                target_row_count=target_row_count,
                source="target tokenizer batch call",
            )
            audit["target_batch_encode_calls"] = int(
                audit.get("target_batch_encode_calls", 0)
            ) + 1
            audit.setdefault("tokenizer_methods", set()).add("target.__call__")
            return result
    batch_encoder = getattr(tokenizer, "batch_encode_plus", None)
    if callable(batch_encoder):
        try:
            encoded = batch_encoder(
                values,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
        except (TypeError, NotImplementedError, AttributeError):
            encoded = None
        if encoded is not None:
            result = _normalized_encoded_rows(
                encoded,
                expected_count=len(values),
                target_row_count=target_row_count,
                source="target tokenizer batch_encode_plus",
            )
            audit["target_batch_encode_calls"] = int(
                audit.get("target_batch_encode_calls", 0)
            ) + 1
            audit.setdefault("tokenizer_methods", set()).add(
                "target.batch_encode_plus"
            )
            return result
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        raise Lfm2CausalSpanTokenBridgeError(
            "Target tokenizer must expose a batch encoder or encode()."
        )
    result = []
    for row_index, value in enumerate(values):
        try:
            encoded = encoder(value, add_special_tokens=False)
        except TypeError:  # pragma: no cover - legacy tokenizer boundary
            encoded = encoder(value)
        result.append(
            _integer_tuple(
                encoded,
                f"target tokenizer encode row {row_index}",
                allow_empty=True,
                upper_bound=target_row_count,
            )
        )
    audit["target_single_encode_calls"] = int(
        audit.get("target_single_encode_calls", 0)
    ) + len(values)
    audit.setdefault("tokenizer_methods", set()).add("target.encode")
    return tuple(result)


def _authenticated_special_transition_map(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[dict[int, int], frozenset[int], tuple[dict[str, Any], ...]]:
    profile = bridge.transport_profile
    donor_specials = {
        int(value) for value in getattr(donor_tokenizer, "all_special_ids", ())
    }
    target_specials = {
        int(value) for value in getattr(target_tokenizer, "all_special_ids", ())
    }
    candidates: dict[int, set[int]] = {}
    lineage: list[dict[str, Any]] = []
    mapping = tuple(int(value) for value in profile.target_to_donor_row_indices)
    for role in ("bos", "eos", "pad", "unk"):
        donor_id = getattr(donor_tokenizer, f"{role}_token_id", None)
        target_id = getattr(target_tokenizer, f"{role}_token_id", None)
        if isinstance(donor_id, int) and not isinstance(donor_id, bool):
            donor_specials.add(int(donor_id))
        if (
            isinstance(donor_id, int)
            and not isinstance(donor_id, bool)
            and isinstance(target_id, int)
            and not isinstance(target_id, bool)
            and 0 <= donor_id < profile.donor_row_capacity
            and 0 <= target_id < profile.target_row_capacity
            and mapping[target_id] == donor_id
        ):
            candidates.setdefault(int(donor_id), set()).add(int(target_id))
            lineage.append(
                {
                    "source": "special-role-and-strict-row-anchor",
                    "role": role,
                    "donor_id": int(donor_id),
                    "target_id": int(target_id),
                }
            )
    for target_id in sorted(target_specials):
        if not 0 <= target_id < len(mapping):
            continue
        donor_id = mapping[target_id]
        candidates.setdefault(donor_id, set()).add(target_id)
        lineage.append(
            {
                "source": "strict-bridge-special-anchor",
                "donor_id": donor_id,
                "target_id": target_id,
            }
        )
    exact = {
        donor_id: next(iter(target_ids))
        for donor_id, target_ids in candidates.items()
        if len(target_ids) == 1
    }
    return exact, frozenset(donor_specials), tuple(lineage)


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        count += 1
    return count


def compile_lfm2_full_donor_contextual_transition_evidence(
    donor_logits: Tensor,
    causal_root_records: Sequence[Mapping[str, Any]],
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    parent_root_logits_evidence: Mapping[str, Any],
    maximum_donor_rows_per_chunk: int = 1024,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile every defined donor row at every cached causal root.

    This is tokenizer work over already-sealed logits and causal prefixes.  It
    never executes or recaptures the donor model.  Ordinary rows are classified
    by exact raw-text continuation and target retokenization; authenticated
    controls are transported through the sealed strict bridge.  Chunking only
    bounds tokenizer work and does not alter row order or probability math.
    """

    logits = _validated_logits(donor_logits)
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    chunk_size = _positive_int(
        maximum_donor_rows_per_chunk,
        "maximum_donor_rows_per_chunk",
    )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    parent, parent_seal_field, parent_sha256 = _verify_sealed_mapping(
        parent_root_logits_evidence,
        schema=None,
        preferred_fields=("evidence_sha256", "result_sha256"),
        name="parent root-logit evidence",
    )
    observed_logits_sha256 = _tensor_sha256(logits)
    if parent.get("donor_logits_sha256") not in {None, observed_logits_sha256}:
        raise Lfm2CausalSpanTokenBridgeError(
            "Parent evidence is sealed around different donor logits."
        )
    profile = bridge.transport_profile
    if int(logits.shape[1]) != profile.donor_row_capacity:
        raise Lfm2CausalSpanTokenBridgeError(
            "Donor logit width differs from the sealed bridge capacity."
        )
    donor_defined = _live_tokenizer_row_count(donor_tokenizer, "donor")
    target_defined = _live_tokenizer_row_count(target_tokenizer, "target")
    if (
        donor_defined > profile.donor_row_capacity
        or target_defined > profile.target_row_capacity
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            "Live tokenizer rows exceed their sealed bridge capacities."
        )
    roots = _normalized_causal_root_records(
        causal_root_records,
        donor_row_count=donor_defined,
        target_row_count=target_defined,
        terminal_row_count=int(logits.shape[0]),
    )
    roots_sha256 = canonical_sha256(list(roots))
    for field in ("causal_root_rows_sha256", "causal_root_records_sha256"):
        if field in parent and parent[field] != roots_sha256:
            raise Lfm2CausalSpanTokenBridgeError(
                f"Parent evidence {field} differs from supplied causal roots."
            )
    for field, expected in (
        ("bridge_artifact_sha256", bridge.artifact_sha256),
        ("bridge_profile_sha256", bridge.profile_fingerprint),
        (
            "donor_tokenizer_semantics_sha256",
            bridge.donor_tokenizer_semantics_sha256,
        ),
        (
            "target_tokenizer_semantics_sha256",
            bridge.target_tokenizer_semantics_sha256,
        ),
    ):
        if field in parent and parent[field] != expected:
            raise Lfm2CausalSpanTokenBridgeError(
                f"Parent evidence {field} differs from the sealed tokenizer bridge."
            )

    tokenizer_audit: dict[str, Any] = {"tokenizer_methods": set()}
    progress_count = 0
    progress_previous_sha256: str | None = None
    progress_callback_invocations = 0
    processed_rows = 0
    total_rows = len(roots) * donor_defined
    classification_counts = {
        kind: 0
        for kind in sorted(
            _LIMITATION_KINDS | {"append", "authenticated-special-control"}
        )
    }

    def emit(stage: str, **details: Any) -> None:
        nonlocal progress_count, progress_previous_sha256
        nonlocal progress_callback_invocations
        body = {
            "schema": LFM2_CAUSAL_SPAN_CONTEXTUAL_PROGRESS_SCHEMA,
            "ordinal": progress_count,
            "previous_progress_sha256": progress_previous_sha256,
            "stage": stage,
            "donor_rows_processed": processed_rows,
            "total_donor_root_rows": total_rows,
            "root_count": len(roots),
            "donor_defined_row_count": donor_defined,
            **details,
        }
        seal = canonical_sha256(body)
        record = MappingProxyType({**body, "progress_sha256": seal})
        progress_count += 1
        progress_previous_sha256 = seal
        if progress_callback is not None:
            progress_callback(record)
            progress_callback_invocations += 1

    emit(
        "start",
        bridge_artifact_sha256=bridge.artifact_sha256,
        donor_logits_sha256=observed_logits_sha256,
        causal_root_rows_sha256=roots_sha256,
    )
    contextual_transitions: list[list[dict[str, Any]]] = []
    special_lineage: tuple[dict[str, Any], ...] = ()
    selective_session_sha256 = parent.get(
        "selective_session_sha256",
        parent_sha256,
    )
    try:
        session_context = bridge.authenticated_live_session(
            donor_tokenizer,
            target_tokenizer,
            selective_session_sha256=_digest(
                selective_session_sha256,
                "selective_session_sha256",
            ),
        )
        with session_context as tokenizer_session:
            special_map, donor_special_ids, special_lineage = (
                _authenticated_special_transition_map(
                    bridge,
                    donor_tokenizer,
                    target_tokenizer,
                )
            )
            for root_index, root in enumerate(roots):
                tokenizer_session.verify_bound(
                    donor_tokenizer,
                    target_tokenizer,
                    expected_donor_sha256=(
                        bridge.donor_tokenizer_semantics_sha256
                    ),
                    expected_target_sha256=(
                        bridge.target_tokenizer_semantics_sha256
                    ),
                )
                donor_prefix = tuple(root["donor_prefix_token_ids"])
                target_root = tuple(root["target_prefix_token_ids"])
                donor_root_text = _batch_decode_exact(
                    donor_tokenizer,
                    [donor_prefix],
                    name="donor",
                    audit=tokenizer_audit,
                )[0]
                canonical_target_root = _batch_encode_exact(
                    target_tokenizer,
                    [donor_root_text],
                    target_row_count=target_defined,
                    audit=tokenizer_audit,
                )[0]
                target_root_text = _batch_decode_exact(
                    target_tokenizer,
                    [target_root],
                    name="target",
                    audit=tokenizer_audit,
                )[0]
                if (
                    canonical_target_root != target_root
                    or target_root_text != donor_root_text
                ):
                    raise Lfm2CausalSpanTokenBridgeError(
                        f"Causal root {root_index} is not the exact canonical "
                        "target encoding of its donor prefix."
                    )
                emit(
                    "root-start",
                    root_index=root_index,
                    terminal_row_index=root["terminal_row_index"],
                    canonical_root_text_sha256=canonical_sha256(donor_root_text),
                )
                root_rows: list[dict[str, Any] | None] = [None] * donor_defined
                root_counts = dict.fromkeys(classification_counts, 0)
                for start in range(0, donor_defined, chunk_size):
                    stop = min(start + chunk_size, donor_defined)
                    chunk_counts = dict.fromkeys(classification_counts, 0)
                    ordinary_ids: list[int] = []
                    ordinary_sequences: list[tuple[int, ...]] = []
                    for donor_id in range(start, stop):
                        if donor_id not in donor_special_ids:
                            ordinary_ids.append(donor_id)
                            ordinary_sequences.append((*donor_prefix, donor_id))
                            continue
                        target_special_id = special_map.get(donor_id)
                        if target_special_id is None:
                            row = {
                                "donor_id": donor_id,
                                "reason": "unmapped-or-ambiguous-donor-control",
                            }
                            kind = "unmapped-control"
                        else:
                            row = {
                                "donor_id": donor_id,
                                "transition_kind": (
                                    "authenticated-special-control"
                                ),
                                "target_append_token_ids": [target_special_id],
                            }
                            kind = "authenticated-special-control"
                        root_rows[donor_id] = row
                        chunk_counts[kind] += 1
                        root_counts[kind] += 1
                        classification_counts[kind] += 1

                    candidate_texts = _batch_decode_exact(
                        donor_tokenizer,
                        ordinary_sequences,
                        name="donor",
                        audit=tokenizer_audit,
                    )
                    candidate_targets = _batch_encode_exact(
                        target_tokenizer,
                        candidate_texts,
                        target_row_count=target_defined,
                        audit=tokenizer_audit,
                    )
                    target_roundtrips = _batch_decode_exact(
                        target_tokenizer,
                        candidate_targets,
                        name="target",
                        audit=tokenizer_audit,
                    )
                    for donor_id, candidate_text, candidate_target, roundtrip in zip(
                        ordinary_ids,
                        candidate_texts,
                        candidate_targets,
                        target_roundtrips,
                        strict=True,
                    ):
                        text_sha256 = canonical_sha256(candidate_text)
                        common = _common_prefix_length(target_root, candidate_target)
                        if not candidate_text.startswith(donor_root_text):
                            row = {
                                "donor_id": donor_id,
                                "reason": "donor-text-nonappend-only",
                                "candidate_text_sha256": text_sha256,
                            }
                            kind = "nonappend-only"
                        elif roundtrip != candidate_text:
                            row = {
                                "donor_id": donor_id,
                                "reason": "target-roundtrip-mismatch",
                                "candidate_target_token_ids": list(candidate_target),
                                "target_common_prefix_tokens": common,
                                "target_replaced_root_suffix_tokens": (
                                    len(target_root) - common
                                ),
                                "candidate_text_sha256": text_sha256,
                            }
                            kind = "roundtrip-mismatch"
                        elif candidate_target[: len(target_root)] != target_root:
                            row = {
                                "donor_id": donor_id,
                                "reason": "target-suffix-rewrite",
                                "candidate_target_token_ids": list(candidate_target),
                                "target_common_prefix_tokens": common,
                                "target_replaced_root_suffix_tokens": (
                                    len(target_root) - common
                                ),
                                "candidate_text_sha256": text_sha256,
                            }
                            kind = "suffix-rewrite"
                        elif len(candidate_target) == len(target_root):
                            row = {
                                "donor_id": donor_id,
                                "reason": "zero-width-target-transition",
                                "candidate_target_token_ids": list(candidate_target),
                                "candidate_text_sha256": text_sha256,
                            }
                            kind = "zero-width"
                        else:
                            row = {
                                "donor_id": donor_id,
                                "transition_kind": "canonical-contextual-append",
                                "target_append_token_ids": list(
                                    candidate_target[len(target_root) :]
                                ),
                                "candidate_text_sha256": text_sha256,
                            }
                            kind = "append"
                        root_rows[donor_id] = row
                        chunk_counts[kind] += 1
                        root_counts[kind] += 1
                        classification_counts[kind] += 1
                    processed_rows += stop - start
                    emit(
                        "chunk-complete",
                        root_index=root_index,
                        donor_row_start=start,
                        donor_row_stop=stop,
                        chunk_classification_counts=chunk_counts,
                    )
                if any(row is None for row in root_rows):
                    raise Lfm2CausalSpanTokenBridgeError(
                        f"Causal root {root_index} did not classify every donor row."
                    )
                contextual_transitions.append(
                    [dict(row) for row in root_rows if row is not None]
                )
                emit(
                    "root-complete",
                    root_index=root_index,
                    root_classification_counts=root_counts,
                )
        tokenizer_session_receipt = session_context.as_dict()
    except StrictTokenizerBridgeError as exc:
        raise Lfm2CausalSpanTokenBridgeError(
            "Live tokenizers failed sealed bridge authentication."
        ) from exc

    emit(
        "classification-complete",
        classification_counts=classification_counts,
    )
    base = build_lfm2_full_donor_span_transition_evidence(
        logits,
        parent_root_logits_evidence=parent_root_logits_evidence,
        root_terminal_row_indices=[
            int(root["terminal_row_index"]) for root in roots
        ],
        root_target_prefix_token_ids=[
            list(root["target_prefix_token_ids"]) for root in roots
        ],
        target_row_count=profile.target_row_capacity,
        contextual_transitions=contextual_transitions,
        donor_defined_row_count=donor_defined,
    )
    body = dict(base)
    body.pop("evidence_sha256")
    tokenizer_methods = sorted(tokenizer_audit.pop("tokenizer_methods"))
    body.update(
        {
            "status": "complete-local-tokenizer-contextual-span-evidence-sealed",
            "source_kind": "full-donor-live-tokenizer-root-contextual-transitions",
            "parent_root_logits_evidence_sha256": parent_sha256,
            "parent_root_logits_seal_field": parent_seal_field,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                bridge.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                bridge.target_tokenizer_semantics_sha256
            ),
            "tokenizer_live_session_sha256": tokenizer_session_receipt[
                "receipt_sha256"
            ],
            "tokenizer_live_session_receipt": tokenizer_session_receipt,
            "causal_root_rows_sha256": roots_sha256,
            "donor_tokenizer_defined_row_count": donor_defined,
            "target_tokenizer_defined_row_count": target_defined,
            "maximum_donor_rows_per_chunk": chunk_size,
            "classification_counts": classification_counts,
            "special_transition_lineage": list(special_lineage),
            "special_transition_lineage_sha256": canonical_sha256(
                list(special_lineage)
            ),
            "progress_record_count": progress_count,
            "progress_final_sha256": progress_previous_sha256,
            "progress_callback_supplied": progress_callback is not None,
            "progress_callback_invocation_count": (
                progress_callback_invocations
            ),
            "tokenizer_methods": tokenizer_methods,
            "tokenizer_operation_audit": {
                **tokenizer_audit,
                "live_tokenizer_authentication_sessions": 1,
                "parent_forward_calls": 0,
                "parent_generate_calls": 0,
                "parent_trace_recaptures": 0,
                "network_access": False,
            },
            "every_defined_donor_row_classified_once_per_root": True,
            "classification_is_exhaustive_not_topk": True,
            "chunking_changes_classification_order_or_math": False,
            "parent_execution_or_recapture_used": False,
        }
    )
    return _sealed(body, "evidence_sha256")


def compile_lfm2_causal_span_token_bridge_plan_from_local_tokenizers(
    donor_logits: Tensor,
    causal_root_records: Sequence[Mapping[str, Any]],
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    parent_root_logits_evidence: Mapping[str, Any],
    maximum_donor_rows_per_chunk: int = 1024,
    maximum_plan_rows_per_chunk: int = 1024,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Lfm2CausalSpanTokenBridgePlan:
    """Compile local tokenizer evidence and feed it directly into the plan."""

    evidence = compile_lfm2_full_donor_contextual_transition_evidence(
        donor_logits,
        causal_root_records,
        bridge,
        donor_tokenizer,
        target_tokenizer,
        parent_root_logits_evidence=parent_root_logits_evidence,
        maximum_donor_rows_per_chunk=maximum_donor_rows_per_chunk,
        progress_callback=progress_callback,
    )
    return compile_lfm2_causal_span_token_bridge_plan(
        donor_logits,
        evidence,
        maximum_rows_per_chunk=maximum_plan_rows_per_chunk,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CausalSpanTokenBridgePlan:
    """Authenticated root-specific CSR tries and conditional log masses."""

    root_state_ids: Tensor
    state_root_indices: Tensor
    state_prefix_token_ids: tuple[tuple[int, ...], ...]
    state_edge_offsets: Tensor
    edge_target_ids: Tensor
    edge_next_state_ids: Tensor
    edge_log_mass: Tensor
    edge_log_probability: Tensor
    state_log_mass: Tensor
    state_terminal_log_mass: Tensor
    state_terminal_mask: Tensor
    state_unresolved_log_mass: Tensor
    state_unresolved_mask: Tensor
    root_target_log_mass: Tensor
    root_target_support_mask: Tensor
    root_full_donor_log_mass: Tensor
    root_unresolved_log_mass: Tensor
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        tensors = {
            name: getattr(self, name)
            for name in (
                "root_state_ids",
                "state_root_indices",
                "state_edge_offsets",
                "edge_target_ids",
                "edge_next_state_ids",
                "edge_log_mass",
                "edge_log_probability",
                "state_log_mass",
                "state_terminal_log_mass",
                "state_terminal_mask",
                "state_unresolved_log_mass",
                "state_unresolved_mask",
                "root_target_log_mass",
                "root_target_support_mask",
                "root_full_donor_log_mass",
                "root_unresolved_log_mass",
            )
        }
        if any(
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.requires_grad
            for value in tensors.values()
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Span-plan tensors must be detached dense tensors."
            )
        prefixes = tuple(tuple(value) for value in self.state_prefix_token_ids)
        state_count = len(prefixes)
        root_count = int(self.root_state_ids.numel())
        target_count = int(self.root_target_log_mass.shape[1])
        edge_count = int(self.edge_target_ids.numel())
        if (
            state_count <= 0
            or root_count <= 0
            or target_count <= 0
            or tuple(self.root_state_ids.shape) != (root_count,)
            or tuple(self.state_root_indices.shape) != (state_count,)
            or tuple(self.state_edge_offsets.shape) != (state_count + 1,)
            or tuple(self.edge_target_ids.shape) != (edge_count,)
            or tuple(self.edge_next_state_ids.shape) != (edge_count,)
            or tuple(self.edge_log_mass.shape) != (edge_count,)
            or tuple(self.edge_log_probability.shape) != (edge_count,)
            or tuple(self.state_log_mass.shape) != (state_count,)
            or tuple(self.state_terminal_log_mass.shape) != (state_count,)
            or tuple(self.state_terminal_mask.shape) != (state_count,)
            or tuple(self.state_unresolved_log_mass.shape) != (state_count,)
            or tuple(self.state_unresolved_mask.shape) != (state_count,)
            or tuple(self.root_target_log_mass.shape) != (root_count, target_count)
            or tuple(self.root_target_support_mask.shape)
            != (root_count, target_count)
            or tuple(self.root_full_donor_log_mass.shape) != (root_count,)
            or tuple(self.root_unresolved_log_mass.shape) != (root_count,)
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Span-plan tensor dimensions are inconsistent."
            )
        integer_names = (
            "root_state_ids",
            "state_root_indices",
            "state_edge_offsets",
            "edge_target_ids",
            "edge_next_state_ids",
        )
        bool_names = (
            "state_terminal_mask",
            "state_unresolved_mask",
            "root_target_support_mask",
        )
        if any(tensors[name].dtype != torch.int64 for name in integer_names) or any(
            tensors[name].dtype != torch.bool for name in bool_names
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Span-plan indices must be int64 and masks must be bool."
            )
        float_names = tuple(
            name for name in tensors if name not in integer_names + bool_names
        )
        if any(tensors[name].dtype != torch.float64 for name in float_names):
            raise Lfm2CausalSpanTokenBridgeError(
                "Authenticated span masses must use the FP64 compiler dtype."
            )
        if any(bool(torch.isnan(tensors[name]).any()) for name in float_names):
            raise Lfm2CausalSpanTokenBridgeError("Span-plan masses contain NaN.")
        if any(bool(torch.isposinf(tensors[name]).any()) for name in float_names):
            raise Lfm2CausalSpanTokenBridgeError(
                "Span-plan masses contain positive infinity."
            )
        offsets = self.state_edge_offsets.tolist()
        if (
            offsets[0] != 0
            or offsets[-1] != edge_count
            or any(right < left for left, right in zip(offsets, offsets[1:]))
            or (edge_count and int(self.edge_target_ids.min()) < 0)
            or (edge_count and int(self.edge_target_ids.max()) >= target_count)
            or (edge_count and int(self.edge_next_state_ids.min()) < 0)
            or (edge_count and int(self.edge_next_state_ids.max()) >= state_count)
            or int(self.root_state_ids.min()) < 0
            or int(self.root_state_ids.max()) >= state_count
        ):
            raise Lfm2CausalSpanTokenBridgeError("Span-plan CSR indices are invalid.")

        receipt, _field, _seal = _verify_sealed_mapping(
            self.receipt,
            schema=LFM2_CAUSAL_SPAN_TOKEN_BRIDGE_PLAN_SCHEMA,
            preferred_fields=("result_sha256",),
            name="causal span plan",
        )
        if (
            receipt.get("root_count") != root_count
            or receipt.get("state_count") != state_count
            or receipt.get("edge_count") != edge_count
            or receipt.get("target_row_count") != target_count
            or receipt.get("state_prefix_rows_sha256")
            != canonical_sha256([list(value) for value in prefixes])
            or any(
                receipt.get(f"{name}_sha256") != _tensor_sha256(value)
                for name, value in tensors.items()
            )
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Causal span plan receipt does not authenticate its tensors."
            )
        object.__setattr__(self, "state_prefix_token_ids", prefixes)
        for name, value in tensors.items():
            object.__setattr__(
                self,
                name,
                value.detach().to(device="cpu").contiguous().clone(),
            )
        object.__setattr__(self, "receipt", MappingProxyType(receipt))

    @property
    def root_count(self) -> int:
        return int(self.root_state_ids.numel())

    @property
    def state_count(self) -> int:
        return len(self.state_prefix_token_ids)

    @property
    def target_row_count(self) -> int:
        return int(self.root_target_log_mass.shape[1])

    def root_state_id(self, root_index: int) -> int:
        index = _nonnegative_int(root_index, "root_index")
        if index >= self.root_count:
            raise Lfm2CausalSpanTokenBridgeError("root_index is outside the plan.")
        return int(self.root_state_ids[index].item())

    def state_conditional_row(self, state_id: int) -> tuple[Tensor, Tensor]:
        """Materialize one diagnostic dense row from compact CSR edge evidence."""

        index = _nonnegative_int(state_id, "state_id")
        if index >= self.state_count:
            raise Lfm2CausalSpanTokenBridgeError("state_id is outside the plan.")
        start = int(self.state_edge_offsets[index].item())
        stop = int(self.state_edge_offsets[index + 1].item())
        values = torch.zeros(self.target_row_count, dtype=torch.float64)
        support = torch.zeros(self.target_row_count, dtype=torch.bool)
        if stop > start:
            token_ids = self.edge_target_ids[start:stop]
            values.index_copy_(0, token_ids, self.edge_log_probability[start:stop])
            support.index_fill_(0, token_ids, True)
        return values, support

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def _transitions_for_root(
    evidence: Mapping[str, Any],
    root: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if evidence["source_kind"] == "full-donor-global-decomposition-inventory":
        decompositions = evidence.get("donor_target_decompositions")
        if (
            not isinstance(decompositions, list)
            or evidence.get("donor_target_decompositions_sha256")
            != canonical_sha256(decompositions)
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Global donor decomposition inventory is unsealed."
            )
        return tuple(
            {
                "donor_id": donor_id,
                "kind": "append",
                "target_append_token_ids": list(sequence),
            }
            for donor_id, sequence in enumerate(decompositions)
        )
    transitions = root.get("transitions")
    if (
        not isinstance(transitions, list)
        or root.get("transitions_sha256") != canonical_sha256(transitions)
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            "Contextual root transition inventory is unsealed."
        )
    return tuple(dict(value) for value in transitions)


def compile_lfm2_causal_span_token_bridge_plan(
    donor_logits: Tensor,
    transition_evidence: Mapping[str, Any],
    *,
    maximum_rows_per_chunk: int = 1024,
) -> Lfm2CausalSpanTokenBridgePlan:
    """Compile exact supported span masses into deterministic CSR tries.

    The accumulation order is root order, donor-row order, then prefix order in
    an FP64 workspace.  ``maximum_rows_per_chunk`` bounds tensor staging only;
    it never changes donor order or introduces a parallel reduction tree.
    """

    logits_input = _validated_logits(donor_logits)
    evidence, _field, evidence_sha256 = _verify_sealed_mapping(
        transition_evidence,
        schema=LFM2_FULL_DONOR_SPAN_TRANSITION_EVIDENCE_SCHEMA,
        preferred_fields=("evidence_sha256",),
        name="full-donor span transition evidence",
    )
    chunk_size = _positive_int(maximum_rows_per_chunk, "maximum_rows_per_chunk")
    logits_sha256 = _tensor_sha256(logits_input)
    if evidence.get("donor_logits_sha256") != logits_sha256:
        raise Lfm2CausalSpanTokenBridgeError(
            "Transition evidence is bound to different donor logits."
        )
    root_count = _positive_int(evidence.get("root_count"), "root_count")
    donor_row_count = _positive_int(
        evidence.get("donor_row_count"), "donor_row_count"
    )
    donor_defined = _positive_int(
        evidence.get("donor_defined_row_count"), "donor_defined_row_count"
    )
    target_count = _positive_int(
        evidence.get("target_row_count"), "target_row_count"
    )
    roots = evidence.get("roots")
    if (
        tuple(logits_input.shape)[1] != donor_row_count
        or donor_defined > donor_row_count
        or not isinstance(roots, list)
        or len(roots) != root_count
    ):
        raise Lfm2CausalSpanTokenBridgeError(
            "Transition evidence dimensions differ from donor logits."
        )

    logits = logits_input.detach().to(device="cpu", dtype=torch.float64).contiguous()
    root_transitions: list[tuple[dict[str, Any], ...]] = []
    state_prefixes: list[tuple[int, ...]] = []
    state_roots: list[int] = []
    root_state_ids: list[int] = []
    root_prefix_to_state: list[dict[tuple[int, ...], int]] = []
    limitation_records: list[dict[str, Any]] = []

    for expected_root_index, root in enumerate(roots):
        if (
            not isinstance(root, Mapping)
            or root.get("root_index") != expected_root_index
            or isinstance(root.get("terminal_row_index"), bool)
            or not isinstance(root.get("terminal_row_index"), int)
            or not 0 <= int(root["terminal_row_index"]) < int(logits.shape[0])
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Root transition records are incomplete or out of order."
            )
        transitions = _transitions_for_root(evidence, root)
        if len(transitions) != donor_defined or any(
            transition.get("donor_id") != donor_id
            for donor_id, transition in enumerate(transitions)
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "A root does not classify every defined donor row exactly once."
            )
        prefixes: set[tuple[int, ...]] = {()}
        for transition in transitions:
            if transition["kind"] != "append":
                continue
            sequence = tuple(transition["target_append_token_ids"])
            for stop in range(1, len(sequence) + 1):
                prefixes.add(sequence[:stop])
        ordered_prefixes = tuple(sorted(prefixes, key=lambda item: (len(item), item)))
        mapping: dict[tuple[int, ...], int] = {}
        for prefix in ordered_prefixes:
            state_id = len(state_prefixes)
            mapping[prefix] = state_id
            state_prefixes.append(prefix)
            state_roots.append(expected_root_index)
        root_state_ids.append(mapping[()])
        root_prefix_to_state.append(mapping)
        root_transitions.append(transitions)

    state_count = len(state_prefixes)
    state_mass = [-math.inf] * state_count
    terminal_mass = [-math.inf] * state_count
    unresolved_mass = [-math.inf] * state_count
    edge_masses: list[dict[int, float]] = [dict() for _ in range(state_count)]
    limitation_kind_counts: dict[str, int] = {}

    with torch.no_grad():
        for root_index, (root, transitions, prefix_to_state) in enumerate(
            zip(roots, root_transitions, root_prefix_to_state, strict=True)
        ):
            terminal_row_index = int(root["terminal_row_index"])
            root_state = prefix_to_state[()]
            for start in range(0, donor_defined, chunk_size):
                stop = min(donor_defined, start + chunk_size)
                staged_values = logits[terminal_row_index, start:stop].tolist()
                for local_index, raw_logit in enumerate(staged_values):
                    donor_id = start + local_index
                    logit = float(raw_logit)
                    transition = transitions[donor_id]
                    if transition["kind"] != "append":
                        state_mass[root_state] = _logaddexp(
                            state_mass[root_state], logit
                        )
                        unresolved_mass[root_state] = _logaddexp(
                            unresolved_mass[root_state], logit
                        )
                        kind = str(transition["kind"])
                        limitation_kind_counts[kind] = (
                            limitation_kind_counts.get(kind, 0) + 1
                        )
                        limitation_records.append(
                            {
                                "root_index": root_index,
                                **dict(transition),
                            }
                        )
                        continue

                    sequence = tuple(transition["target_append_token_ids"])
                    for prefix_length in range(len(sequence) + 1):
                        state_id = prefix_to_state[sequence[:prefix_length]]
                        state_mass[state_id] = _logaddexp(
                            state_mass[state_id], logit
                        )
                        if prefix_length == len(sequence):
                            terminal_mass[state_id] = _logaddexp(
                                terminal_mass[state_id], logit
                            )
                        else:
                            target_id = sequence[prefix_length]
                            edge_masses[state_id][target_id] = _logaddexp(
                                edge_masses[state_id].get(target_id, -math.inf),
                                logit,
                            )

    state_offsets = [0]
    edge_target_ids: list[int] = []
    edge_next_states: list[int] = []
    edge_log_masses: list[float] = []
    edge_log_probabilities: list[float] = []
    root_target_mass = torch.zeros(
        (root_count, target_count), dtype=torch.float64
    )
    root_target_support = torch.zeros(
        (root_count, target_count), dtype=torch.bool
    )
    maximum_internal_conservation_error = 0.0
    ambiguous_terminal_count = 0
    terminal_state_count = 0

    for state_id, (root_index, prefix, outgoing) in enumerate(
        zip(state_roots, state_prefixes, edge_masses, strict=True)
    ):
        ordered_edges = tuple(sorted(outgoing.items()))
        for target_id, log_mass in ordered_edges:
            next_prefix = (*prefix, target_id)
            next_state = root_prefix_to_state[root_index].get(next_prefix)
            if next_state is None:
                raise Lfm2CausalSpanTokenBridgeError(
                    "Compiled trie edge has no deterministic destination state."
                )
            if abs(state_mass[next_state] - log_mass) > 1e-12:
                raise Lfm2CausalSpanTokenBridgeError(
                    "Trie edge mass differs from its child prefix mass."
                )
            edge_target_ids.append(target_id)
            edge_next_states.append(next_state)
            edge_log_masses.append(log_mass)
            edge_log_probabilities.append(log_mass - state_mass[state_id])
            if not prefix:
                root_target_mass[root_index, target_id] = log_mass
                root_target_support[root_index, target_id] = True
        state_offsets.append(len(edge_target_ids))
        has_terminal = terminal_mass[state_id] != -math.inf
        terminal_state_count += int(has_terminal)
        ambiguous_terminal_count += int(has_terminal and bool(ordered_edges))
        composed = _logsumexp([value for _target, value in ordered_edges])
        composed = _logaddexp(composed, terminal_mass[state_id])
        composed = _logaddexp(composed, unresolved_mass[state_id])
        error = abs(composed - state_mass[state_id])
        maximum_internal_conservation_error = max(
            maximum_internal_conservation_error,
            error,
        )
        if error > 1e-12:
            raise Lfm2CausalSpanTokenBridgeError(
                "Trie node child, terminal, and unresolved masses do not conserve."
            )

    root_state_tensor = torch.tensor(root_state_ids, dtype=torch.int64)
    root_full_mass_values: list[float] = []
    root_unresolved_values: list[float] = []
    maximum_root_conservation_error = 0.0
    for root_index, root in enumerate(roots):
        terminal_row_index = int(root["terminal_row_index"])
        full_mass = _logsumexp(
            logits[terminal_row_index, :donor_defined].tolist()
        )
        state_id = root_state_ids[root_index]
        root_full_mass_values.append(full_mass)
        root_unresolved_values.append(unresolved_mass[state_id])
        representable = _logsumexp(
            [
                float(root_target_mass[root_index, target_id].item())
                for target_id in range(target_count)
                if bool(root_target_support[root_index, target_id])
            ]
        )
        reconstructed = _logaddexp(representable, unresolved_mass[state_id])
        error = abs(reconstructed - full_mass)
        maximum_root_conservation_error = max(
            maximum_root_conservation_error,
            error,
        )
        if error > 1e-12:
            raise Lfm2CausalSpanTokenBridgeError(
                "Root target buckets plus unresolved mass do not conserve donor mass."
            )

    tensors: dict[str, Tensor] = {
        "root_state_ids": root_state_tensor,
        "state_root_indices": torch.tensor(state_roots, dtype=torch.int64),
        "state_edge_offsets": torch.tensor(state_offsets, dtype=torch.int64),
        "edge_target_ids": torch.tensor(edge_target_ids, dtype=torch.int64),
        "edge_next_state_ids": torch.tensor(edge_next_states, dtype=torch.int64),
        "edge_log_mass": torch.tensor(edge_log_masses, dtype=torch.float64),
        "edge_log_probability": torch.tensor(
            edge_log_probabilities, dtype=torch.float64
        ),
        "state_log_mass": torch.tensor(state_mass, dtype=torch.float64),
        "state_terminal_log_mass": torch.tensor(
            terminal_mass, dtype=torch.float64
        ),
        "state_terminal_mask": torch.tensor(
            [value != -math.inf for value in terminal_mass], dtype=torch.bool
        ),
        "state_unresolved_log_mass": torch.tensor(
            unresolved_mass, dtype=torch.float64
        ),
        "state_unresolved_mask": torch.tensor(
            [value != -math.inf for value in unresolved_mass], dtype=torch.bool
        ),
        "root_target_log_mass": root_target_mass,
        "root_target_support_mask": root_target_support,
        "root_full_donor_log_mass": torch.tensor(
            root_full_mass_values, dtype=torch.float64
        ),
        "root_unresolved_log_mass": torch.tensor(
            root_unresolved_values, dtype=torch.float64
        ),
    }
    body = {
        "schema": LFM2_CAUSAL_SPAN_TOKEN_BRIDGE_PLAN_SCHEMA,
        "status": "full-donor-causal-span-csr-plan-compiled",
        "transition_evidence_sha256": evidence_sha256,
        "parent_root_logits_evidence_sha256": evidence[
            "parent_root_logits_evidence_sha256"
        ],
        "donor_logits_sha256": logits_sha256,
        "source_kind": evidence["source_kind"],
        "root_specific_contextual_transitions": evidence[
            "root_specific_contextual_transitions"
        ],
        "root_count": root_count,
        "donor_row_count": donor_row_count,
        "donor_defined_row_count": donor_defined,
        "undefined_donor_row_count": donor_row_count - donor_defined,
        "target_row_count": target_count,
        "state_count": state_count,
        "internal_state_count": state_count - root_count,
        "edge_count": len(edge_target_ids),
        "terminal_state_count": terminal_state_count,
        "ambiguous_terminal_state_count": ambiguous_terminal_count,
        "limitation_transition_count": len(limitation_records),
        "limitation_kind_counts": dict(sorted(limitation_kind_counts.items())),
        "limitation_records": limitation_records,
        "suffix_rewrite_transitions_are_not_installed_as_append_edges": True,
        "terminal_mass_without_next_donor_conditionals_is_explicit": True,
        "exact_all_branch_autoregressive_equivalence_claimed": False,
        "root_target_mass_is_exact_for_every_append_only_defined_donor_row": True,
        "root_target_plus_unresolved_mass_covers_every_defined_donor_row": True,
        "internal_child_plus_terminal_plus_unresolved_mass_conserved": True,
        "maximum_root_log_mass_conservation_error": (
            maximum_root_conservation_error
        ),
        "maximum_internal_log_mass_conservation_error": (
            maximum_internal_conservation_error
        ),
        "state_prefix_rows_sha256": canonical_sha256(
            [list(value) for value in state_prefixes]
        ),
        "source_dtype": _dtype_name(logits_input),
        "source_device_type": logits_input.device.type,
        "accumulation_dtype": "float64",
        "published_mass_dtype": "float64",
        "maximum_rows_per_chunk": chunk_size,
        "deterministic_accumulation_order": (
            "root-then-donor-row-then-prefix-left-to-right"
        ),
        "chunking_changes_staging_not_reduction_order": True,
        **{
            f"{name}_sha256": _tensor_sha256(value)
            for name, value in tensors.items()
        },
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "network_access": False,
        },
    }
    receipt = _sealed(body, "result_sha256")
    return Lfm2CausalSpanTokenBridgePlan(
        state_prefix_token_ids=tuple(state_prefixes),
        receipt=receipt,
        **tensors,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CausalSpanRuntimeState:
    """One child-owned causal trie cursor for a single generated sequence."""

    root_index: int
    state_id: int
    consumed_target_token_ids: tuple[int, ...] = ()
    terminal_available: bool = False
    terminal_ambiguous: bool = False
    completed: bool = False


class Lfm2CausalSpanHead(nn.Module):
    """Parameter-free child runtime over one authenticated causal span plan.

    The first slice publishes exact cached conditionals rather than fitting a
    hidden-state residual.  ``hidden_states`` determines output batch shape,
    device, and dtype; explicit trie state determines which conditional row is
    active.  A later integration can add a folded child-feature residual while
    retaining these buffers and runtime semantics.
    """

    def __init__(
        self,
        plan: Lfm2CausalSpanTokenBridgePlan,
        *,
        allow_unresolved_roots: bool = False,
        allow_terminal_ambiguity: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(plan, Lfm2CausalSpanTokenBridgePlan):
            raise TypeError("plan must be Lfm2CausalSpanTokenBridgePlan.")
        if not isinstance(allow_unresolved_roots, bool) or not isinstance(
            allow_terminal_ambiguity, bool
        ):
            raise TypeError("Span-head limitation policies must be boolean.")
        if not allow_unresolved_roots and bool(plan.state_unresolved_mask.any()):
            raise Lfm2CausalSpanTokenBridgeError(
                "Plan contains unresolved suffix-rewrite or nonappend mass; "
                "explicitly opt in before installing this partial runtime."
            )
        self.allow_unresolved_roots = allow_unresolved_roots
        self.allow_terminal_ambiguity = allow_terminal_ambiguity
        self.plan_result_sha256 = plan.receipt["result_sha256"]
        self.target_row_count = plan.target_row_count
        for name in (
            "root_state_ids",
            "state_root_indices",
            "state_edge_offsets",
            "edge_target_ids",
            "edge_next_state_ids",
            "edge_log_probability",
            "state_terminal_mask",
            "state_unresolved_mask",
        ):
            self.register_buffer(name, getattr(plan, name).clone(), persistent=True)

    @property
    def runtime_receipt(self) -> dict[str, Any]:
        body = {
            "schema": LFM2_CAUSAL_SPAN_HEAD_RUNTIME_SCHEMA,
            "plan_result_sha256": self.plan_result_sha256,
            "target_row_count": self.target_row_count,
            "learned_parameter_count": sum(
                int(value.numel()) for value in self.parameters()
            ),
            "allow_unresolved_roots": self.allow_unresolved_roots,
            "allow_terminal_ambiguity": self.allow_terminal_ambiguity,
            "runtime_parent_or_sidecar_required": False,
            "state_is_child_owned": True,
            "terminal_commit_requires_explicit_next_root": True,
            "training_steps": 0,
        }
        return _sealed(body, "runtime_sha256")

    def start(self, root_index: int) -> Lfm2CausalSpanRuntimeState:
        index = _nonnegative_int(root_index, "root_index")
        if index >= int(self.root_state_ids.numel()):
            raise Lfm2CausalSpanTokenBridgeError("root_index is outside the plan.")
        state_id = int(self.root_state_ids[index].item())
        unresolved = bool(self.state_unresolved_mask[state_id].item())
        if unresolved and not self.allow_unresolved_roots:
            raise Lfm2CausalSpanTokenBridgeError(
                "Root has unresolved probability mass under the active policy."
            )
        return Lfm2CausalSpanRuntimeState(root_index=index, state_id=state_id)

    def advance(
        self,
        state: Lfm2CausalSpanRuntimeState,
        target_token_id: int,
    ) -> Lfm2CausalSpanRuntimeState:
        if not isinstance(state, Lfm2CausalSpanRuntimeState):
            raise TypeError("state must be Lfm2CausalSpanRuntimeState.")
        token_id = _nonnegative_int(target_token_id, "target_token_id")
        if token_id >= self.target_row_count:
            raise Lfm2CausalSpanTokenBridgeError(
                "target_token_id is outside the target vocabulary."
            )
        if state.completed:
            raise Lfm2CausalSpanTokenBridgeError(
                "Completed span state must be committed/reset before advancing."
            )
        start = int(self.state_edge_offsets[state.state_id].item())
        stop = int(self.state_edge_offsets[state.state_id + 1].item())
        matches = [
            edge
            for edge in range(start, stop)
            if int(self.edge_target_ids[edge].item()) == token_id
        ]
        if len(matches) != 1:
            raise Lfm2CausalSpanTokenBridgeError(
                "Target token is not one deterministic outgoing trie edge."
            )
        next_state = int(self.edge_next_state_ids[matches[0]].item())
        if int(self.state_root_indices[next_state].item()) != state.root_index:
            raise Lfm2CausalSpanTokenBridgeError(
                "Trie transition crossed its authenticated causal root."
            )
        next_start = int(self.state_edge_offsets[next_state].item())
        next_stop = int(self.state_edge_offsets[next_state + 1].item())
        terminal = bool(self.state_terminal_mask[next_state].item())
        ambiguous = terminal and next_stop > next_start
        return Lfm2CausalSpanRuntimeState(
            root_index=state.root_index,
            state_id=next_state,
            consumed_target_token_ids=(
                *state.consumed_target_token_ids,
                token_id,
            ),
            terminal_available=terminal,
            terminal_ambiguous=ambiguous,
            completed=terminal and not ambiguous,
        )

    def commit_terminal(
        self,
        state: Lfm2CausalSpanRuntimeState,
        *,
        next_root_index: int | None = None,
    ) -> Lfm2CausalSpanRuntimeState:
        if not isinstance(state, Lfm2CausalSpanRuntimeState) or not (
            state.terminal_available or state.completed
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Only a terminal span state can be committed."
            )
        if next_root_index is None:
            raise Lfm2CausalSpanTokenBridgeError(
                "The sealed span plan has no next-donor conditional; committing "
                "a terminal span requires an explicit next_root_index."
            )
        return self.start(next_root_index)

    def forward(self, hidden_states: Tensor, state_ids: Tensor) -> Tensor:
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.ndim < 1
            or not torch.is_floating_point(hidden_states)
            or torch.is_complex(hidden_states)
            or hidden_states.device.type == "meta"
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "hidden_states must be a non-scalar dense floating tensor."
            )
        leading_shape = tuple(int(size) for size in hidden_states.shape[:-1])
        if (
            not isinstance(state_ids, Tensor)
            or state_ids.dtype not in {torch.int32, torch.int64}
            or tuple(state_ids.shape) != leading_shape
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "state_ids must match the hidden-state leading dimensions."
            )
        flat_states = state_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if flat_states.numel() and (
            int(flat_states.min()) < 0
            or int(flat_states.max()) >= int(self.state_root_indices.numel())
        ):
            raise Lfm2CausalSpanTokenBridgeError("state_ids leave the span plan.")
        if not self.allow_unresolved_roots and bool(
            self.state_unresolved_mask.index_select(0, flat_states).any()
        ):
            raise Lfm2CausalSpanTokenBridgeError(
                "Selected span state contains unresolved probability mass."
            )
        offsets = self.state_edge_offsets
        terminal = self.state_terminal_mask.index_select(0, flat_states)
        edge_counts = offsets.index_select(0, flat_states + 1) - offsets.index_select(
            0, flat_states
        )
        ambiguous = terminal & edge_counts.gt(0)
        if not self.allow_terminal_ambiguity and bool(ambiguous.any()):
            raise Lfm2CausalSpanTokenBridgeError(
                "Selected state mixes terminal mass with continuation mass, but "
                "the cache has no next-donor conditional for the terminal branch."
            )
        if bool(edge_counts.eq(0).any()):
            raise Lfm2CausalSpanTokenBridgeError(
                "A terminal-only state has no next target-token distribution."
            )
        floor = torch.finfo(hidden_states.dtype).min
        result = torch.full(
            (int(flat_states.numel()), self.target_row_count),
            floor,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        for row_index, state_id in enumerate(flat_states.tolist()):
            start = int(self.state_edge_offsets[state_id].item())
            stop = int(self.state_edge_offsets[state_id + 1].item())
            token_ids = self.edge_target_ids[start:stop].to(
                device=hidden_states.device
            )
            probabilities = self.edge_log_probability[start:stop].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            result[row_index].index_copy_(0, token_ids, probabilities)
        return result.reshape((*leading_shape, self.target_row_count))

    def forward_state(
        self,
        hidden_states: Tensor,
        state: Lfm2CausalSpanRuntimeState,
    ) -> Tensor:
        if not isinstance(state, Lfm2CausalSpanRuntimeState):
            raise TypeError("state must be Lfm2CausalSpanRuntimeState.")
        if state.completed:
            raise Lfm2CausalSpanTokenBridgeError(
                "Completed span state has no next-token conditional."
            )
        leading = tuple(int(size) for size in hidden_states.shape[:-1])
        states = torch.full(
            leading,
            state.state_id,
            dtype=torch.int64,
            device=hidden_states.device,
        )
        return self.forward(hidden_states, states)


__all__ = [
    "LFM2_CAUSAL_SPAN_CONTEXTUAL_PROGRESS_SCHEMA",
    "LFM2_CAUSAL_SPAN_HEAD_RUNTIME_SCHEMA",
    "LFM2_CAUSAL_SPAN_ROOT_LOGITS_EVIDENCE_SCHEMA",
    "LFM2_CAUSAL_SPAN_TOKEN_BRIDGE_PLAN_SCHEMA",
    "LFM2_FULL_DONOR_SPAN_TRANSITION_EVIDENCE_SCHEMA",
    "Lfm2CausalSpanHead",
    "Lfm2CausalSpanRuntimeState",
    "Lfm2CausalSpanTokenBridgeError",
    "Lfm2CausalSpanTokenBridgePlan",
    "build_lfm2_full_donor_span_transition_evidence",
    "compile_lfm2_causal_span_token_bridge_plan_from_local_tokenizers",
    "compile_lfm2_causal_span_token_bridge_plan",
    "compile_lfm2_full_donor_contextual_transition_evidence",
    "seal_lfm2_causal_span_root_logits_evidence",
]
