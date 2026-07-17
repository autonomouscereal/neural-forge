"""Cached-causal mixed-precision allocation for standalone LFM2 checkpoints.

The existing packed-value control proves that a real LFM2 checkpoint can be
stored at an exact 50% full-directory byte ceiling, but it selects seven-bit
groups only from local weight reconstruction error.  This module keeps the
same audited payload and fresh-reload runtime while replacing that allocator
with a parent-bound, cached-causal policy:

* the tied embedding/output surface, norms, convolution state, and Q/K paths
  remain at the higher eight-bit width;
* lower precision is spent only on FFN producer/residual and attention V/O
  tensors;
* layers are ranked with the already-sealed activation-selected physical
  proposals and their downstream logit, EOS, repetition, task, residual, and
  attention damage; and
* the exporter performs no parent forward, generation, backward, optimizer,
  or training operation.

The physical representation is intentionally not a parallel format.  It is
the existing ``neural-forge-lfm2-packed-value-fold-control-v1`` format with a
sealed ``causal_value_allocation`` record in its manifest.  Consequently the
same strict raw-payload loader supplies the standalone/fresh-reload proof and
all files, selectors, scales, padding, configuration, tokenizer data, and
manifest bytes remain inside the exact checkpoint ceiling.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional HF dependency
    raise RuntimeError(
        "PyTorch is required for the LFM2 causal value allocator."
    ) from exc

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_lfm2_cached_causal_evidence import (
    verify_lfm2_cached_causal_frontier_document,
)
from .hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from . import hf_lfm2_value_fold_control as packed


ALLOCATION_SCHEMA = "neural-forge-lfm2-cached-causal-value-allocation-v1"
EVIDENCE_SCHEMA = "neural-forge-lfm2-cached-causal-value-evidence-v1"
POLICY_ID = "lfm2_cached_causal_role_aware_8b7b_v1"
TOKEN_SELECTION_SCHEMA = "neural-forge-lfm2-cached-causal-value-token-selection-v1"
TARGET_REDUCTION = 0.50
_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
_FFN_PATTERN = re.compile(r"^model\.layers\.(\d+)\.feed_forward\.(w1|w2|w3)\.weight$")
_ATTENTION_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj)\.weight$"
)
_ROLE_RISK_MULTIPLIERS = {
    "ffn_producer": 1.0,
    "ffn_residual_writer": 2.0,
    "attention_value": 4.0,
    "attention_residual_writer": 6.0,
}
_PROTECTED_ROLES = {
    "tied_embedding_output",
    "normalization",
    "attention_query_key",
    "convolution_state",
    "bias",
    "unclassified",
}
_CAUSAL_CHANNELS = (
    "final_logit_margin",
    "eos",
    "repetition",
    "task_anchor",
    "residual_path",
    "attention_path",
)
_DIGEST_CHARS = frozenset("0123456789abcdef")
_EXACT_FULL_TENSOR_ROLES = frozenset({"normalization"})


class Lfm2CausalValueAllocatorError(packed.Lfm2ValueFoldControlError):
    """The cached evidence, allocation policy, or parent binding is invalid."""


class Lfm2CausalValueAllocatorBudgetError(
    packed.Lfm2ValueFoldControlBudgetError,
    Lfm2CausalValueAllocatorError,
):
    """Eligible lower-precision groups cannot meet the exact byte target."""


class Lfm2CausalValueAllocatorFormatError(
    packed.Lfm2ValueFoldControlFormatError,
    Lfm2CausalValueAllocatorError,
):
    """A packed artifact does not contain a valid causal allocation record."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise Lfm2CausalValueAllocatorError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Lfm2CausalValueAllocatorError(f"{name} must be finite and nonnegative.")
    return float(value)


def _token_contents(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return (str(value["content"]),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            result.extend(_token_contents(item))
        return tuple(result)
    return ()


def _local_special_token_ids(tokenizer_directory: Path) -> tuple[int, ...]:
    """Resolve every local special/chat-control token without a hub loader."""

    tokenizer = packed._strict_json(tokenizer_directory / "tokenizer.json")
    tokenizer_config = (
        packed._strict_json(tokenizer_directory / "tokenizer_config.json")
        if (tokenizer_directory / "tokenizer_config.json").is_file()
        else {}
    )
    special_map = (
        packed._strict_json(tokenizer_directory / "special_tokens_map.json")
        if (tokenizer_directory / "special_tokens_map.json").is_file()
        else {}
    )
    content_to_id: dict[str, int] = {}
    selected: set[int] = set()
    added = tokenizer.get("added_tokens", ())
    if isinstance(added, Sequence) and not isinstance(added, (str, bytes)):
        for item in added:
            if (
                isinstance(item, Mapping)
                and isinstance(item.get("content"), str)
                and isinstance(item.get("id"), int)
                and not isinstance(item.get("id"), bool)
                and int(item["id"]) >= 0
            ):
                token_id = int(item["id"])
                content_to_id[str(item["content"])] = token_id
                if item.get("special") is True:
                    selected.add(token_id)
    model = tokenizer.get("model")
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    if isinstance(vocab, Mapping):
        for content, token_id in vocab.items():
            if (
                isinstance(content, str)
                and isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and token_id >= 0
            ):
                content_to_id.setdefault(content, token_id)
    elif isinstance(vocab, Sequence) and not isinstance(vocab, (str, bytes)):
        for token_id, item in enumerate(vocab):
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and item
                and isinstance(item[0], str)
            ):
                content_to_id.setdefault(str(item[0]), token_id)
    stems = (
        "eos",
        "bos",
        "pad",
        "unk",
        "sep",
        "cls",
        "mask",
    )
    for stem in stems:
        for source in (tokenizer_config, special_map):
            direct = source.get(f"{stem}_token_id")
            values = (
                direct
                if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes))
                else (direct,)
            )
            for token_id in values:
                if (
                    isinstance(token_id, int)
                    and not isinstance(token_id, bool)
                    and token_id >= 0
                ):
                    selected.add(int(token_id))
            for content in _token_contents(source.get(f"{stem}_token")):
                if content not in content_to_id:
                    raise Lfm2CausalValueAllocatorError(
                        f"Local special token {content!r} is absent from tokenizer.json."
                    )
                selected.add(content_to_id[content])
    for source in (tokenizer_config, special_map):
        for content in _token_contents(source.get("additional_special_tokens")):
            if content not in content_to_id:
                raise Lfm2CausalValueAllocatorError(
                    f"Local chat-control token {content!r} is absent from "
                    "tokenizer.json."
                )
            selected.add(content_to_id[content])
        direct = source.get("additional_special_tokens_ids")
        if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
            for token_id in direct:
                if (
                    isinstance(token_id, int)
                    and not isinstance(token_id, bool)
                    and token_id >= 0
                ):
                    selected.add(int(token_id))
    if not selected:
        raise Lfm2CausalValueAllocatorError(
            "No local BOS/EOS/PAD/chat-control token IDs could be resolved."
        )
    return tuple(sorted(selected))


def _calibration_selected_token_ids(
    document: Mapping[str, Any] | None,
    evidence: "Lfm2CausalValueEvidence",
) -> tuple[int, ...]:
    if document is None:
        return ()
    if not isinstance(document, Mapping):
        raise Lfm2CausalValueAllocatorError(
            "calibration_token_selection must be a mapping or None."
        )
    body = dict(document)
    seal = body.pop("selection_sha256", None)
    token_ids = body.get("token_ids")
    if (
        body.get("schema") != TOKEN_SELECTION_SCHEMA
        or _digest(seal, "selection_sha256") != canonical_sha256(body)
        or body.get("parent_trace_cache_key") != evidence.parent_trace_cache_key
        or body.get("parent_corpus_sha256") != evidence.parent_corpus_sha256
        or body.get("selection_rows_sha256") != evidence.selection_rows_sha256
        or body.get("split_role") != "selection"
        or body.get("selection_method")
        not in {"frequency", "margin", "frequency_margin"}
        or body.get("independent_holdout_used") is not False
        or not isinstance(token_ids, list)
        or not token_ids
    ):
        raise Lfm2CausalValueAllocatorError(
            "Calibration token selection is unsealed, cache-stale, or not "
            "selection-only."
        )
    values = tuple(sorted(set(token_ids)))
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise Lfm2CausalValueAllocatorError(
            "Calibration-selected token IDs must be nonnegative integers."
        )
    return values


@dataclass(frozen=True, slots=True)
class Lfm2CausalValueLayerSensitivity:
    """One layer's authenticated cached-causal sensitivity summary."""

    layer_index: int
    normalized_sensitivity: float
    maximum_damage_slope: float
    channel_maxima: Mapping[str, float]
    option_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2CausalValueAllocatorError(
                "layer_index must be a nonnegative integer."
            )
        normalized = _finite_nonnegative(
            self.normalized_sensitivity,
            "normalized_sensitivity",
        )
        if normalized > 1.0:
            raise Lfm2CausalValueAllocatorError(
                "normalized_sensitivity must be in [0, 1]."
            )
        object.__setattr__(self, "normalized_sensitivity", normalized)
        object.__setattr__(
            self,
            "maximum_damage_slope",
            _finite_nonnegative(
                self.maximum_damage_slope,
                "maximum_damage_slope",
            ),
        )
        maxima = dict(self.channel_maxima)
        if set(maxima) != set(_CAUSAL_CHANNELS):
            raise Lfm2CausalValueAllocatorError(
                "Every layer sensitivity needs all six causal channels."
            )
        object.__setattr__(
            self,
            "channel_maxima",
            {
                name: _finite_nonnegative(maxima[name], f"channel_maxima.{name}")
                for name in _CAUSAL_CHANNELS
            },
        )
        options = tuple(self.option_ids)
        if (
            not options
            or len(set(options)) != len(options)
            or any(not isinstance(value, str) or not value for value in options)
        ):
            raise Lfm2CausalValueAllocatorError(
                "option_ids must be nonempty and unique."
            )
        object.__setattr__(self, "option_ids", options)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "normalized_sensitivity": self.normalized_sensitivity,
            "maximum_damage_slope": self.maximum_damage_slope,
            "channel_maxima": dict(self.channel_maxima),
            "option_ids": list(self.option_ids),
        }


@dataclass(frozen=True, slots=True)
class Lfm2CausalValueEvidence:
    """Portable allocation evidence distilled from the sealed causal frontier."""

    schema: str
    frontier_sha256: str
    parent_identity_sha256: str
    parent_trace_cache_key: str
    parent_corpus_sha256: str
    selection_rows_sha256: str
    validation_case_ids_sha256: str
    damage_weights: Mapping[str, float]
    layers: tuple[Lfm2CausalValueLayerSensitivity, ...]
    evidence_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frontier_sha256": self.frontier_sha256,
            "parent_identity_sha256": self.parent_identity_sha256,
            "parent_trace_cache_key": self.parent_trace_cache_key,
            "parent_corpus_sha256": self.parent_corpus_sha256,
            "selection_rows_sha256": self.selection_rows_sha256,
            "validation_case_ids_sha256": self.validation_case_ids_sha256,
            "damage_weights": dict(self.damage_weights),
            "layers": [value.as_dict() for value in self.layers],
            "source_contract": {
                "activation_selected_physical_proposals": True,
                "cached_parent_trace_reused": True,
                "candidate_child_full_prefix_traces": True,
                "residual_path_damage_used": True,
                "attention_path_damage_used": True,
                "exact_token_match_objective_used": False,
                "parent_forward_calls": 0,
                "parent_generation_calls": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
            },
            "evidence_sha256": self.evidence_sha256,
        }


def _evidence_body(value: Lfm2CausalValueEvidence) -> dict[str, Any]:
    body = value.as_dict()
    body.pop("evidence_sha256", None)
    return body


def verify_lfm2_causal_value_evidence(
    evidence: Lfm2CausalValueEvidence,
) -> Lfm2CausalValueEvidence:
    if (
        not isinstance(evidence, Lfm2CausalValueEvidence)
        or evidence.schema != EVIDENCE_SCHEMA
        or canonical_sha256(_evidence_body(evidence)) != evidence.evidence_sha256
        or tuple(value.layer_index for value in evidence.layers)
        != tuple(range(len(evidence.layers)))
    ):
        raise Lfm2CausalValueAllocatorError(
            "Cached-causal value evidence seal or layer coverage is invalid."
        )
    weights = dict(evidence.damage_weights)
    if set(weights) != set(_CAUSAL_CHANNELS):
        raise Lfm2CausalValueAllocatorError(
            "Cached-causal value evidence lacks damage weights."
        )
    total = 0.0
    for name in _CAUSAL_CHANNELS:
        weight = _finite_nonnegative(weights[name], f"damage_weights.{name}")
        if weight <= 0.0:
            raise Lfm2CausalValueAllocatorError(
                "Every cached-causal damage channel must have positive weight."
            )
        total += weight
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Lfm2CausalValueAllocatorError(
            "Cached-causal damage weights must sum to one."
        )
    for name in (
        "frontier_sha256",
        "parent_identity_sha256",
        "parent_trace_cache_key",
        "parent_corpus_sha256",
        "selection_rows_sha256",
        "validation_case_ids_sha256",
        "evidence_sha256",
    ):
        _digest(getattr(evidence, name), name)
    return evidence


def derive_lfm2_causal_value_evidence(
    frontier_document: Mapping[str, Any],
) -> Lfm2CausalValueEvidence:
    """Authenticate a frontier and reduce its proposals to per-layer risk."""

    verified = verify_lfm2_cached_causal_frontier_document(frontier_document)
    binding = verified["parent_binding"]
    global_evidence = verified["global_evidence"]
    weights = {
        name: _finite_nonnegative(
            global_evidence["damage_weights"][name],
            f"damage_weights.{name}",
        )
        for name in _CAUSAL_CHANNELS
    }
    candidates_by_layer: dict[int, list[Mapping[str, Any]]] = {}
    validation_seals: set[str] = set()
    for candidate in verified["candidates"]:
        if candidate.get("site_kind") != "ffn_width":
            continue
        layer_index = candidate.get("layer_index")
        source_units = candidate.get("source_units")
        target_units = candidate.get("target_units")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or isinstance(source_units, bool)
            or not isinstance(source_units, int)
            or isinstance(target_units, bool)
            or not isinstance(target_units, int)
            or not 0 < target_units < source_units
        ):
            raise Lfm2CausalValueAllocatorError(
                "Cached-causal FFN candidate geometry is invalid."
            )
        candidates_by_layer.setdefault(layer_index, []).append(candidate)
        validation_seals.add(
            _digest(
                candidate.get("validation_case_ids_sha256"),
                "candidate.validation_case_ids_sha256",
            )
        )
    if (
        not candidates_by_layer
        or tuple(sorted(candidates_by_layer)) != tuple(range(len(candidates_by_layer)))
        or len(validation_seals) != 1
    ):
        raise Lfm2CausalValueAllocatorError(
            "Cached-causal candidates do not provide contiguous, commonly "
            "validated layer coverage."
        )

    raw_layers: list[tuple[float, dict[str, float], tuple[str, ...]]] = []
    for layer_index in range(len(candidates_by_layer)):
        layer_candidates = candidates_by_layer[layer_index]
        slopes = []
        channel_maxima = {name: 0.0 for name in _CAUSAL_CHANNELS}
        option_ids = []
        for candidate in layer_candidates:
            damage = candidate.get("damage")
            if not isinstance(damage, Mapping):
                raise Lfm2CausalValueAllocatorError(
                    "Cached-causal candidate damage is missing."
                )
            values = {
                name: _finite_nonnegative(
                    damage.get(name),
                    f"candidate.damage.{name}",
                )
                for name in _CAUSAL_CHANNELS
            }
            for name, value in values.items():
                channel_maxima[name] = max(channel_maxima[name], value)
            weighted_damage = sum(values[name] * weights[name] for name in values)
            removal_fraction = (
                int(candidate["source_units"]) - int(candidate["target_units"])
            ) / int(candidate["source_units"])
            slopes.append(weighted_damage / removal_fraction)
            option_id = candidate.get("option_id")
            if not isinstance(option_id, str) or not option_id:
                raise Lfm2CausalValueAllocatorError(
                    "Cached-causal candidate option ID is invalid."
                )
            option_ids.append(option_id)
        raw_layers.append(
            (
                max(slopes),
                channel_maxima,
                tuple(sorted(option_ids)),
            )
        )
    minimum = min(value[0] for value in raw_layers)
    maximum = max(value[0] for value in raw_layers)
    spread = maximum - minimum
    layers = tuple(
        Lfm2CausalValueLayerSensitivity(
            layer_index=index,
            normalized_sensitivity=(
                0.0 if spread <= 1e-30 else (value[0] - minimum) / spread
            ),
            maximum_damage_slope=value[0],
            channel_maxima=value[1],
            option_ids=value[2],
        )
        for index, value in enumerate(raw_layers)
    )
    evidence = Lfm2CausalValueEvidence(
        schema=EVIDENCE_SCHEMA,
        frontier_sha256=_digest(
            verified.get("frontier_sha256"),
            "frontier_sha256",
        ),
        parent_identity_sha256=_digest(
            binding.get("parent_identity_sha256"),
            "parent_identity_sha256",
        ),
        parent_trace_cache_key=_digest(
            binding.get("parent_trace_cache_key"),
            "parent_trace_cache_key",
        ),
        parent_corpus_sha256=_digest(
            global_evidence.get("parent_corpus_sha256"),
            "parent_corpus_sha256",
        ),
        selection_rows_sha256=_digest(
            verified.get("split", {}).get("selection_rows_sha256"),
            "selection_rows_sha256",
        ),
        validation_case_ids_sha256=next(iter(validation_seals)),
        damage_weights=weights,
        layers=layers,
        evidence_sha256="",
    )
    sealed = Lfm2CausalValueEvidence(
        schema=evidence.schema,
        frontier_sha256=evidence.frontier_sha256,
        parent_identity_sha256=evidence.parent_identity_sha256,
        parent_trace_cache_key=evidence.parent_trace_cache_key,
        parent_corpus_sha256=evidence.parent_corpus_sha256,
        selection_rows_sha256=evidence.selection_rows_sha256,
        validation_case_ids_sha256=evidence.validation_case_ids_sha256,
        damage_weights=evidence.damage_weights,
        layers=evidence.layers,
        evidence_sha256=canonical_sha256(_evidence_body(evidence)),
    )
    return verify_lfm2_causal_value_evidence(sealed)


@dataclass(frozen=True, slots=True)
class _TensorRole:
    role: str
    layer_index: int | None
    eligible_for_lower_precision: bool
    risk_multiplier: float | None


def _tensor_role(name: str) -> _TensorRole:
    if name == "model.embed_tokens.weight":
        return _TensorRole("tied_embedding_output", None, False, None)
    layer_match = _LAYER_PATTERN.match(name)
    layer_index = int(layer_match.group(1)) if layer_match else None
    if name.endswith(".bias"):
        return _TensorRole("bias", layer_index, False, None)
    if (
        "norm" in name
        or name == "model.embedding_norm.weight"
        or ".q_layernorm." in name
        or ".k_layernorm." in name
    ):
        return _TensorRole("normalization", layer_index, False, None)
    ffn_match = _FFN_PATTERN.match(name)
    if ffn_match:
        role = "ffn_residual_writer" if ffn_match.group(2) == "w2" else "ffn_producer"
        return _TensorRole(
            role,
            int(ffn_match.group(1)),
            True,
            _ROLE_RISK_MULTIPLIERS[role],
        )
    attention_match = _ATTENTION_PATTERN.match(name)
    if attention_match:
        projection = attention_match.group(2)
        if projection in {"q_proj", "k_proj"}:
            return _TensorRole(
                "attention_query_key",
                int(attention_match.group(1)),
                False,
                None,
            )
        role = (
            "attention_value" if projection == "v_proj" else "attention_residual_writer"
        )
        return _TensorRole(
            role,
            int(attention_match.group(1)),
            True,
            _ROLE_RISK_MULTIPLIERS[role],
        )
    if ".conv." in name:
        return _TensorRole("convolution_state", layer_index, False, None)
    return _TensorRole("unclassified", layer_index, False, None)


def _raw_tensor_segment(
    analysis: Any,
    *,
    flat_start: int,
    value_count: int,
    scope: str,
) -> dict[str, Any]:
    if (
        flat_start < 0
        or value_count <= 0
        or flat_start + value_count > analysis.source_numel
    ):
        raise Lfm2CausalValueAllocatorError(
            f"Exact overlay range for {analysis.canonical_name!r} is invalid."
        )
    values = (
        analysis.parameter.detach()
        .reshape(-1)[flat_start : flat_start + value_count]
        .to(device="cpu")
        .contiguous()
    )
    raw = values.view(torch.uint8).numpy().tobytes(order="C")
    if len(raw) != value_count * int(analysis.parameter.element_size()):
        raise AssertionError("Exact overlay source-byte count is inconsistent.")
    return {
        "scope": scope,
        "flat_start": flat_start,
        "value_count": value_count,
        "raw_size_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }


def _exact_overlay_payloads(
    analyses: Sequence[Any],
    roles: Mapping[str, _TensorRole],
    *,
    protected_token_ids: Sequence[int],
    hidden_size: int,
    group_size: int,
) -> dict[str, Any]:
    """Capture exact source-dtype hard anchors inside the counted manifest."""

    overlays: dict[str, Any] = {}
    protected_ids = tuple(sorted(set(int(value) for value in protected_token_ids)))
    for analysis in analyses:
        name = analysis.canonical_name
        role = roles[name]
        segments = []
        if role.role in _EXACT_FULL_TENSOR_ROLES:
            segments.append(
                _raw_tensor_segment(
                    analysis,
                    flat_start=0,
                    value_count=analysis.source_numel,
                    scope="full_normalization_tensor",
                )
            )
        elif re.fullmatch(r"model\.layers\.\d+\.conv\.conv\.weight", name):
            segments.append(
                _raw_tensor_segment(
                    analysis,
                    flat_start=0,
                    value_count=analysis.source_numel,
                    scope="full_depthwise_convolution_kernel",
                )
            )
        elif name == "model.embed_tokens.weight":
            if analysis.shape[1] != hidden_size:
                raise Lfm2CausalValueAllocatorError(
                    "Tied embedding width differs from the parent hidden size."
                )
            for token_id in protected_ids:
                if token_id >= analysis.shape[0]:
                    raise Lfm2CausalValueAllocatorError(
                        "A protected token ID is outside the tied vocabulary."
                    )
                segments.append(
                    _raw_tensor_segment(
                        analysis,
                        flat_start=token_id * hidden_size,
                        value_count=hidden_size,
                        scope=f"tied_input_output_token_row:{token_id}",
                    )
                )
        if not segments:
            continue
        touched_groups: set[int] = set()
        for segment in segments:
            first = int(segment["flat_start"]) // group_size
            last = (
                int(segment["flat_start"]) + int(segment["value_count"]) - 1
            ) // group_size
            touched_groups.update(range(first, last + 1))
        overlays[name] = {
            "shape": list(analysis.shape),
            "source_dtype": analysis.source_dtype,
            "source_numel": analysis.source_numel,
            "segments": segments,
            "exact_value_count": sum(int(value["value_count"]) for value in segments),
            "exact_raw_bytes": sum(int(value["raw_size_bytes"]) for value in segments),
            "exact_group_count": len(touched_groups),
        }
    return dict(sorted(overlays.items()))


def _allocation_order(
    analyses: Sequence[Any],
    evidence: Lfm2CausalValueEvidence,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], Mapping[str, _TensorRole]]:
    verify_lfm2_causal_value_evidence(evidence)
    sensitivity = {
        value.layer_index: value.normalized_sensitivity for value in evidence.layers
    }
    offsets = []
    roles: dict[str, _TensorRole] = {}
    eligible: list[tuple[float, str, int, int]] = []
    cursor = 0
    for analysis in analyses:
        start = cursor
        cursor += analysis.group_count
        offsets.append((start, cursor))
        role = _tensor_role(analysis.canonical_name)
        roles[analysis.canonical_name] = role
        if role.layer_index is not None and role.layer_index not in sensitivity:
            raise Lfm2CausalValueAllocatorError(
                f"Tensor {analysis.canonical_name!r} has no cached layer sensitivity."
            )
        if not role.eligible_for_lower_precision:
            continue
        layer_sensitivity = sensitivity[role.layer_index]
        causal_multiplier = 1.0 + 15.0 * layer_sensitivity
        role_multiplier = float(role.risk_multiplier)
        for group_index, distortion in enumerate(analysis.distortion_penalty):
            incremental = _finite_nonnegative(
                float(distortion),
                f"{analysis.canonical_name}.distortion_penalty",
            )
            # The tiny additive term makes a zero-error tie still prefer the
            # less causally sensitive layer.  The canonical name/group index
            # then provide a deterministic final tie-break.
            risk = (
                incremental + 1e-30
            ) * causal_multiplier * role_multiplier + layer_sensitivity * 1e-24
            eligible.append(
                (
                    risk,
                    analysis.canonical_name,
                    group_index,
                    start + group_index,
                )
            )
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    return (
        np.asarray([value[3] for value in eligible], dtype=np.int64),
        tuple(offsets),
        roles,
    )


def _allocation_record(
    *,
    evidence: Lfm2CausalValueEvidence,
    analyses: Sequence[Any],
    masks: Sequence[np.ndarray],
    roles: Mapping[str, _TensorRole],
    eos_token_ids: Sequence[int],
    protected_token_ids: Sequence[int],
    local_special_token_ids: Sequence[int],
    calibration_selected_token_ids: Sequence[int],
    exact_overlays: Mapping[str, Any],
    counterfactual_lower_group_count: int | None,
    bytes_saved_per_lower_group: int,
) -> dict[str, Any]:
    layer_sensitivity = {
        value.layer_index: value.normalized_sensitivity for value in evidence.layers
    }
    tensors = {}
    protected_groups = 0
    eligible_groups = 0
    lower_groups = 0
    by_role: dict[str, dict[str, int]] = {}
    for analysis, mask in zip(analyses, masks, strict=True):
        role = roles[analysis.canonical_name]
        low = int(np.count_nonzero(mask))
        if not role.eligible_for_lower_precision and low:
            raise AssertionError("A protected tensor received lower precision.")
        group_count = int(analysis.group_count)
        protected_groups += group_count if not role.eligible_for_lower_precision else 0
        eligible_groups += group_count if role.eligible_for_lower_precision else 0
        lower_groups += low
        role_counts = by_role.setdefault(
            role.role,
            {"tensor_count": 0, "group_count": 0, "lower_group_count": 0},
        )
        role_counts["tensor_count"] += 1
        role_counts["group_count"] += group_count
        role_counts["lower_group_count"] += low
        tensors[analysis.canonical_name] = {
            "role": role.role,
            "layer_index": role.layer_index,
            "eligible_for_lower_precision": (role.eligible_for_lower_precision),
            "risk_multiplier": role.risk_multiplier,
            "normalized_layer_sensitivity": (
                None
                if role.layer_index is None
                else layer_sensitivity[role.layer_index]
            ),
            "group_count": group_count,
            "base_8bit_group_count": group_count - low,
            "lower_7bit_group_count": low,
            "exact_source_dtype_overlay": (analysis.canonical_name in exact_overlays),
        }
    overlay_raw_bytes = sum(
        int(value["exact_raw_bytes"]) for value in exact_overlays.values()
    )
    overlay_base64_characters = sum(
        len(segment["raw_base64"])
        for value in exact_overlays.values()
        for segment in value["segments"]
    )
    counterfactual = (
        None
        if counterfactual_lower_group_count is None
        else int(counterfactual_lower_group_count)
    )
    additional_lower = None if counterfactual is None else lower_groups - counterfactual
    body = {
        "schema": ALLOCATION_SCHEMA,
        "policy_id": POLICY_ID,
        "evidence": evidence.as_dict(),
        "eos_token_ids": list(eos_token_ids),
        "protected_token_ids": list(protected_token_ids),
        "protected_token_sources": {
            "local_bos_eos_pad_and_chat_control_ids": list(local_special_token_ids),
            "cached_calibration_selection_ids": list(calibration_selected_token_ids),
            "sealed_independent_holdout_ids_used": [],
            "validation_or_holdout_blind": True,
        },
        "policy": {
            "base_bit_width": 8,
            "lower_bit_width": 7,
            "causal_sensitivity_multiplier": "1 + 15 * normalized_layer_sensitivity",
            "group_risk": (
                "incremental_normalized_reconstruction_sse * "
                "causal_sensitivity_multiplier * role_risk_multiplier"
            ),
            "role_risk_multipliers": dict(_ROLE_RISK_MULTIPLIERS),
            "hard_protected_roles": sorted(_PROTECTED_ROLES),
            "embedding_and_tied_lm_head_base_width_protected": True,
            "special_and_selected_rows_exact_source_dtype": True,
            "tied_rows_protect_input_identity_and_output_competitor_logits": True,
            "all_norms_exact_source_dtype": True,
            "all_qk_norms_exact_source_dtype": True,
            "all_depthwise_conv_kernels_exact_source_dtype": True,
            "all_attention_qk_base_width_protected": True,
            "all_convolution_state_base_width_protected": True,
            "lower_precision_spend_order_is_global": True,
            "exact_token_match_objective_used": False,
        },
        "counts": {
            "protected_group_count": protected_groups,
            "eligible_group_count": eligible_groups,
            "lower_group_count": lower_groups,
            "base_group_count": protected_groups + eligible_groups - lower_groups,
        },
        "by_role": dict(sorted(by_role.items())),
        "tensors": tensors,
        "exact_overlays": dict(exact_overlays),
        "exact_byte_exchange": {
            "exact_source_value_bytes": overlay_raw_bytes,
            "base64_payload_characters_inside_manifest": (overlay_base64_characters),
            "canonical_overlay_record_bytes": len(
                canonical_json_bytes(dict(exact_overlays))
            ),
            "lower_group_count_without_exact_overlays": counterfactual,
            "additional_lower_groups_paid_for_exact_overlays": additional_lower,
            "bytes_saved_per_additional_lower_group": (bytes_saved_per_lower_group),
            "compensating_payload_bytes_saved": (
                None
                if additional_lower is None
                else additional_lower * bytes_saved_per_lower_group
            ),
            "all_exchange_bytes_counted_in_checkpoint_ceiling": True,
        },
        "execution_audit": {
            "parent_cache_build_calls": 0,
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_used": False,
        },
    }
    return {**body, "allocation_sha256": canonical_sha256(body)}


def _causal_manifest_document(
    *,
    model: nn.Module,
    target: packed.Lfm2ValueFoldTarget,
    aliases: Mapping[str, str],
    entries: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    dense_tensor_bytes: int,
    shared_nonweight_bytes: int,
    payload_bytes: int,
    compressed_checkpoint_bytes: int,
    manifest_bytes: int,
    lower_group_count: int,
    total_group_count: int,
    tokenizer_fingerprint: str,
    allocation: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    baseline, _raw = packed._manifest_document(
        model=model,
        target=target,
        aliases=aliases,
        entries=entries,
        artifacts=artifacts,
        dense_tensor_bytes=dense_tensor_bytes,
        shared_nonweight_bytes=shared_nonweight_bytes,
        payload_bytes=payload_bytes,
        compressed_checkpoint_bytes=compressed_checkpoint_bytes,
        manifest_bytes=manifest_bytes,
        lower_group_count=lower_group_count,
        total_group_count=total_group_count,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    core = dict(baseline)
    core.pop("manifest_sha256", None)
    core["control_label"] = (
        "CACHED-CAUSAL MIXED-PRECISION VALUE FOLD; "
        "EXACT 50% SERIALIZED-BYTE TARGET; NOT STRUCTURAL PARAMETER COMPACTION"
    )
    core["method"] = {
        **core["method"],
        "parent_traces_used": True,
        "cached_parent_traces_used": True,
        "new_parent_traces_generated": False,
        "causal_allocation_used": True,
    }
    core["quantization"] = {
        **core["quantization"],
        "distortion_allocator": (
            "cached-causal role/layer risk times incremental normalized "
            "reconstruction SSE"
        ),
        "allocator_policy_id": POLICY_ID,
    }
    core["causal_value_allocation"] = dict(allocation)
    document = {**core, "manifest_sha256": canonical_sha256(core)}
    return document, canonical_json_bytes(document) + b"\n"


def _budget_layout(
    *,
    model: nn.Module,
    target: packed.Lfm2ValueFoldTarget,
    analyses: Sequence[Any],
    aliases: Mapping[str, str],
    masks: Sequence[np.ndarray],
    roles: Mapping[str, _TensorRole],
    evidence: Lfm2CausalValueEvidence,
    eos_token_ids: Sequence[int],
    protected_token_ids: Sequence[int],
    local_special_token_ids: Sequence[int],
    calibration_selected_token_ids: Sequence[int],
    exact_overlays: Mapping[str, Any],
    counterfactual_lower_group_count: int | None,
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], int]:
    entries, payload_bytes = packed._tensor_entry_layout(analyses, masks, target)
    allocation = _allocation_record(
        evidence=evidence,
        analyses=analyses,
        masks=masks,
        roles=roles,
        eos_token_ids=eos_token_ids,
        protected_token_ids=protected_token_ids,
        local_special_token_ids=local_special_token_ids,
        calibration_selected_token_ids=calibration_selected_token_ids,
        exact_overlays=exact_overlays,
        counterfactual_lower_group_count=counterfactual_lower_group_count,
        bytes_saved_per_lower_group=target.bytes_saved_per_lower_group,
    )
    for name, metadata in allocation["tensors"].items():
        entries[name] = {**entries[name], "causal_allocation": metadata}
    shared_nonweight_bytes = (
        int(config_artifact["size_bytes"])
        + int(generation_artifact["size_bytes"])
        + sum(int(value["size_bytes"]) for value in tokenizer_artifacts)
    )
    dense_bytes = sum(value.source_nbytes for value in analyses)
    lower_count = sum(int(np.count_nonzero(value)) for value in masks)
    total_groups = sum(value.group_count for value in analyses)
    artifacts = packed._placeholder_artifacts(
        config_artifact,
        generation_artifact,
        tokenizer_artifacts,
        payload_bytes,
    )
    other_bytes = shared_nonweight_bytes + payload_bytes

    def factory(total: int, manifest_bytes: int):
        return _causal_manifest_document(
            model=model,
            target=target,
            aliases=aliases,
            entries=entries,
            artifacts=artifacts,
            dense_tensor_bytes=dense_bytes,
            shared_nonweight_bytes=shared_nonweight_bytes,
            payload_bytes=payload_bytes,
            compressed_checkpoint_bytes=total,
            manifest_bytes=manifest_bytes,
            lower_group_count=lower_count,
            total_group_count=total_groups,
            tokenizer_fingerprint=tokenizer_fingerprint,
            allocation=allocation,
        )

    document, raw = packed._stable_manifest(
        other_bytes=other_bytes,
        factory=factory,
    )
    return document, raw, entries, payload_bytes


def _solve_budget(
    *,
    model: nn.Module,
    target: packed.Lfm2ValueFoldTarget,
    analyses: Sequence[Any],
    aliases: Mapping[str, str],
    order: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    roles: Mapping[str, _TensorRole],
    evidence: Lfm2CausalValueEvidence,
    eos_token_ids: Sequence[int],
    protected_token_ids: Sequence[int],
    local_special_token_ids: Sequence[int],
    calibration_selected_token_ids: Sequence[int],
    exact_overlays: Mapping[str, Any],
    counterfactual_lower_group_count: int | None,
    config_artifact: Mapping[str, Any],
    generation_artifact: Mapping[str, Any],
    tokenizer_artifacts: Sequence[Mapping[str, Any]],
    tokenizer_fingerprint: str,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    lower_count = 0
    while True:
        masks = packed._selection_masks(
            analyses,
            offsets,
            order,
            lower_count,
        )
        document, _raw, _entries, _payload_bytes = _budget_layout(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            masks=masks,
            roles=roles,
            evidence=evidence,
            eos_token_ids=eos_token_ids,
            protected_token_ids=protected_token_ids,
            local_special_token_ids=local_special_token_ids,
            calibration_selected_token_ids=calibration_selected_token_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=(counterfactual_lower_group_count),
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        total = int(document["storage"]["compressed_checkpoint_bytes"])
        ceiling = int(document["target"]["target_checkpoint_bytes"])
        if total <= ceiling:
            break
        deficit = total - ceiling
        lower_count += max(
            1,
            math.ceil(deficit / target.bytes_saved_per_lower_group),
        )
        if lower_count > len(order):
            maximum_savings = len(order) * target.bytes_saved_per_lower_group
            raise Lfm2CausalValueAllocatorBudgetError(
                "The exact 50% checkpoint needs more lower-bit groups than the "
                "cached-causal policy permits. Protected embedding/output, "
                "norm, convolution, and attention Q/K groups remain immutable; "
                f"eligible groups can save at most {maximum_savings} bytes."
            )
    if lower_count:
        previous_masks = packed._selection_masks(
            analyses,
            offsets,
            order,
            lower_count - 1,
        )
        previous, _raw, _entries, _payload_bytes = _budget_layout(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            masks=previous_masks,
            roles=roles,
            evidence=evidence,
            eos_token_ids=eos_token_ids,
            protected_token_ids=protected_token_ids,
            local_special_token_ids=local_special_token_ids,
            calibration_selected_token_ids=calibration_selected_token_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=(counterfactual_lower_group_count),
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if (
            previous["storage"]["compressed_checkpoint_bytes"]
            <= previous["target"]["target_checkpoint_bytes"]
        ):
            raise AssertionError(
                "The causal allocator did not choose the minimal lower-group count."
            )
    return masks, document


def export_lfm2_causal_value_fold(
    model: nn.Module,
    destination: str | os.PathLike[str],
    *,
    tokenizer_directory: str | os.PathLike[str],
    frontier_document: Mapping[str, Any],
    eos_token_ids: Sequence[int],
    calibration_token_selection: Mapping[str, Any] | None = None,
    group_size: int = 256,
    analysis_chunk_groups: int = 4096,
) -> dict[str, Any]:
    """Export one exact-50%, standalone, cached-causal packed LFM2."""

    target = packed.Lfm2ValueFoldTarget.from_reduction(
        TARGET_REDUCTION,
        group_size=group_size,
        analysis_chunk_groups=analysis_chunk_groups,
    )
    evidence = derive_lfm2_causal_value_evidence(frontier_document)
    parent_identity = fingerprint_lfm2_parent(model).identity_sha256
    if evidence.parent_identity_sha256 != parent_identity:
        raise Lfm2CausalValueAllocatorError(
            "Cached-causal evidence binds a different LFM2 parent."
        )
    if len(evidence.layers) != int(model.config.num_hidden_layers):
        raise Lfm2CausalValueAllocatorError(
            "Cached-causal evidence layer count differs from the parent."
        )
    eos_ids = tuple(sorted({int(value) for value in eos_token_ids}))
    if not eos_ids or any(
        value < 0 or value >= int(model.config.vocab_size) for value in eos_ids
    ):
        raise Lfm2CausalValueAllocatorError(
            "eos_token_ids must be nonempty and inside the parent vocabulary."
        )
    output = Path(destination).expanduser().absolute()
    if output.exists():
        raise Lfm2CausalValueAllocatorError(
            f"Refusing to replace existing destination {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_source = packed._safe_local_directory(
        tokenizer_directory,
        "tokenizer_directory",
    )
    local_special_ids = _local_special_token_ids(tokenizer_source)
    calibration_ids = _calibration_selected_token_ids(
        calibration_token_selection,
        evidence,
    )
    protected_token_ids = tuple(sorted(set(local_special_ids + calibration_ids)))
    if not set(eos_ids).issubset(protected_token_ids):
        raise Lfm2CausalValueAllocatorError(
            "Declared EOS IDs differ from the locally resolved special tokens."
        )
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        config, generation = packed._config_documents(model)
        packed._write_canonical_json(temporary / packed.CONFIG_FILENAME, config)
        packed._write_canonical_json(
            temporary / packed.GENERATION_CONFIG_FILENAME,
            generation,
        )
        tokenizer_artifacts = packed._copy_tokenizer_artifacts(
            tokenizer_source,
            temporary,
        )
        tokenizer_fingerprint = packed._tokenizer_fingerprint(tokenizer_artifacts)
        config_artifact = packed._artifact(temporary / packed.CONFIG_FILENAME)
        generation_artifact = packed._artifact(
            temporary / packed.GENERATION_CONFIG_FILENAME
        )
        analyses, aliases = packed._analyze_model(model, target)
        order, offsets, roles = _allocation_order(analyses, evidence)
        exact_overlays = _exact_overlay_payloads(
            analyses,
            roles,
            protected_token_ids=protected_token_ids,
            hidden_size=int(model.config.hidden_size),
            group_size=target.group_size,
        )
        counterfactual_masks, counterfactual = _solve_budget(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            order=order,
            offsets=offsets,
            roles=roles,
            evidence=evidence,
            eos_token_ids=eos_ids,
            protected_token_ids=protected_token_ids,
            local_special_token_ids=local_special_ids,
            calibration_selected_token_ids=calibration_ids,
            exact_overlays={},
            counterfactual_lower_group_count=None,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        counterfactual_lower_count = sum(
            int(np.count_nonzero(value)) for value in counterfactual_masks
        )
        masks, planned = _solve_budget(
            model=model,
            target=target,
            analyses=analyses,
            aliases=aliases,
            order=order,
            offsets=offsets,
            roles=roles,
            evidence=evidence,
            eos_token_ids=eos_ids,
            protected_token_ids=protected_token_ids,
            local_special_token_ids=local_special_ids,
            calibration_selected_token_ids=calibration_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=counterfactual_lower_count,
            config_artifact=config_artifact,
            generation_artifact=generation_artifact,
            tokenizer_artifacts=tokenizer_artifacts,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        segment_digests, _error = packed._write_quantized_payload(
            temporary / packed.PAYLOAD_FILENAME,
            analyses,
            masks,
            target,
            temporary,
        )
        entries, payload_bytes = packed._tensor_entry_layout(
            analyses,
            masks,
            target,
            segment_digests=segment_digests,
        )
        allocation = _allocation_record(
            evidence=evidence,
            analyses=analyses,
            masks=masks,
            roles=roles,
            eos_token_ids=eos_ids,
            protected_token_ids=protected_token_ids,
            local_special_token_ids=local_special_ids,
            calibration_selected_token_ids=calibration_ids,
            exact_overlays=exact_overlays,
            counterfactual_lower_group_count=counterfactual_lower_count,
            bytes_saved_per_lower_group=target.bytes_saved_per_lower_group,
        )
        for name, metadata in allocation["tensors"].items():
            entries[name] = {**entries[name], "causal_allocation": metadata}
        payload_artifact = packed._artifact(temporary / packed.PAYLOAD_FILENAME)
        if payload_artifact["size_bytes"] != payload_bytes:
            raise AssertionError("Physical payload size differs from the plan.")
        shared_nonweight_bytes = (
            config_artifact["size_bytes"]
            + generation_artifact["size_bytes"]
            + sum(value["size_bytes"] for value in tokenizer_artifacts)
        )
        dense_bytes = sum(value.source_nbytes for value in analyses)
        lower_count = sum(int(np.count_nonzero(value)) for value in masks)
        total_groups = sum(value.group_count for value in analyses)
        artifacts = {
            "config": config_artifact,
            "generation_config": generation_artifact,
            "payload": payload_artifact,
            "tokenizer": list(tokenizer_artifacts),
        }
        other_bytes = shared_nonweight_bytes + payload_bytes

        def factory(total: int, manifest_bytes: int):
            return _causal_manifest_document(
                model=model,
                target=target,
                aliases=aliases,
                entries=entries,
                artifacts=artifacts,
                dense_tensor_bytes=dense_bytes,
                shared_nonweight_bytes=shared_nonweight_bytes,
                payload_bytes=payload_bytes,
                compressed_checkpoint_bytes=total,
                manifest_bytes=manifest_bytes,
                lower_group_count=lower_count,
                total_group_count=total_groups,
                tokenizer_fingerprint=tokenizer_fingerprint,
                allocation=allocation,
            )

        manifest, manifest_raw = packed._stable_manifest(
            other_bytes=other_bytes,
            factory=factory,
        )
        if (
            manifest["storage"]["compressed_checkpoint_bytes"]
            > manifest["target"]["target_checkpoint_bytes"]
        ):
            raise Lfm2CausalValueAllocatorBudgetError(
                "Final hashes changed the manifest enough to exceed 50%."
            )
        if (
            planned["causal_value_allocation"]["counts"]["lower_group_count"]
            != allocation["counts"]["lower_group_count"]
        ):
            raise AssertionError("Final causal group allocation changed.")
        if (
            counterfactual["target"]["requested_reduction"] != TARGET_REDUCTION
            or allocation["exact_byte_exchange"][
                "additional_lower_groups_paid_for_exact_overlays"
            ]
            != lower_count - counterfactual_lower_count
            or lower_count < counterfactual_lower_count
        ):
            raise AssertionError("Exact protected-byte exchange changed.")
        packed._write_new(
            temporary / packed.MANIFEST_FILENAME,
            manifest_raw,
        )
        observed_total = sum(
            path.stat().st_size for path in temporary.iterdir() if path.is_file()
        )
        if observed_total != manifest["storage"]["compressed_checkpoint_bytes"]:
            raise AssertionError(
                "On-disk checkpoint byte count differs from the manifest."
            )
        temporary.replace(output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validated_causal_manifest(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    root, manifest, _paths = packed._manifest_and_artifacts(checkpoint_directory)
    allocation = manifest.get("causal_value_allocation")
    if not isinstance(allocation, Mapping):
        raise Lfm2CausalValueAllocatorFormatError(
            "Packed checkpoint has no cached-causal allocation record."
        )
    body = dict(allocation)
    allocation_sha = body.pop("allocation_sha256", None)
    evidence_raw = body.get("evidence")
    if (
        body.get("schema") != ALLOCATION_SCHEMA
        or body.get("policy_id") != POLICY_ID
        or _digest(allocation_sha, "allocation_sha256") != canonical_sha256(body)
        or not isinstance(evidence_raw, Mapping)
    ):
        raise Lfm2CausalValueAllocatorFormatError(
            "Cached-causal allocation seal or policy is invalid."
        )
    counts = body.get("counts")
    tensors = body.get("tensors")
    exact_overlays = body.get("exact_overlays")
    entries = manifest.get("tensors")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(tensors, Mapping)
        or not isinstance(exact_overlays, Mapping)
        or not isinstance(entries, Mapping)
        or set(tensors) != set(entries)
        or counts.get("lower_group_count")
        != manifest["quantization"]["lower_group_count"]
    ):
        raise Lfm2CausalValueAllocatorFormatError(
            "Cached-causal tensor allocation differs from the packed payload."
        )
    observed_lower = 0
    observed_protected = 0
    for name, metadata in tensors.items():
        entry = entries[name]
        if entry.get("causal_allocation") != metadata:
            raise Lfm2CausalValueAllocatorFormatError(
                f"Tensor {name!r} causal allocation metadata is inconsistent."
            )
        low = metadata.get("lower_7bit_group_count")
        eligible = metadata.get("eligible_for_lower_precision")
        if (
            isinstance(low, bool)
            or not isinstance(low, int)
            or low < 0
            or low != entry.get("lower_group_count")
            or eligible not in {True, False}
            or (eligible is False and low != 0)
        ):
            raise Lfm2CausalValueAllocatorFormatError(
                f"Tensor {name!r} violates the protected-role policy."
            )
        observed_lower += low
        if eligible is False:
            observed_protected += int(entry["group_count"])
    if (
        observed_lower != counts.get("lower_group_count")
        or observed_protected != counts.get("protected_group_count")
        or tensors["model.embed_tokens.weight"]["lower_7bit_group_count"] != 0
    ):
        raise Lfm2CausalValueAllocatorFormatError(
            "Cached-causal protected/lower group totals are inconsistent."
        )
    protected_ids = body.get("protected_token_ids")
    eos_ids = body.get("eos_token_ids")
    if (
        not isinstance(protected_ids, list)
        or not protected_ids
        or len(set(protected_ids)) != len(protected_ids)
        or protected_ids != sorted(protected_ids)
        or not isinstance(eos_ids, list)
        or not set(eos_ids).issubset(protected_ids)
    ):
        raise Lfm2CausalValueAllocatorFormatError(
            "Protected special/calibration token rows are invalid."
        )
    overlay_raw_bytes = 0
    overlay_base64_characters = 0
    for name, metadata in exact_overlays.items():
        if name not in entries or not isinstance(metadata, Mapping):
            raise Lfm2CausalValueAllocatorFormatError(
                "Exact source-dtype overlay names differ from packed tensors."
            )
        entry = entries[name]
        if (
            metadata.get("shape") != entry.get("shape")
            or metadata.get("source_dtype") != entry.get("source_dtype")
            or metadata.get("source_numel") != entry.get("source_numel")
            or not isinstance(metadata.get("segments"), list)
            or not metadata["segments"]
        ):
            raise Lfm2CausalValueAllocatorFormatError(
                f"Exact overlay {name!r} metadata is invalid."
            )
        previous_end = 0
        observed_values = 0
        observed_bytes = 0
        for segment in metadata["segments"]:
            if not isinstance(segment, Mapping):
                raise Lfm2CausalValueAllocatorFormatError(
                    f"Exact overlay {name!r} has a malformed segment."
                )
            start = segment.get("flat_start")
            count = segment.get("value_count")
            encoded = segment.get("raw_base64")
            raw_size = segment.get("raw_size_bytes")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or start < previous_end
                or count <= 0
                or start + count > int(entry["source_numel"])
                or not isinstance(encoded, str)
                or isinstance(raw_size, bool)
                or not isinstance(raw_size, int)
                or raw_size != count * 2
            ):
                raise Lfm2CausalValueAllocatorFormatError(
                    f"Exact overlay {name!r} segment range is invalid."
                )
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise Lfm2CausalValueAllocatorFormatError(
                    f"Exact overlay {name!r} base64 is invalid."
                ) from exc
            if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != segment.get(
                "raw_sha256"
            ):
                raise Lfm2CausalValueAllocatorFormatError(
                    f"Exact overlay {name!r} source bytes failed authentication."
                )
            previous_end = start + count
            observed_values += count
            observed_bytes += raw_size
            overlay_base64_characters += len(encoded)
        if observed_values != metadata.get(
            "exact_value_count"
        ) or observed_bytes != metadata.get("exact_raw_bytes"):
            raise Lfm2CausalValueAllocatorFormatError(
                f"Exact overlay {name!r} totals are inconsistent."
            )
        overlay_raw_bytes += observed_bytes
    exchange = body.get("exact_byte_exchange")
    if (
        not isinstance(exchange, Mapping)
        or exchange.get("exact_source_value_bytes") != overlay_raw_bytes
        or exchange.get("base64_payload_characters_inside_manifest")
        != overlay_base64_characters
        or exchange.get("all_exchange_bytes_counted_in_checkpoint_ceiling") is not True
    ):
        raise Lfm2CausalValueAllocatorFormatError(
            "Exact protected-byte exchange accounting is inconsistent."
        )
    return root, manifest


class ExactOverlayPackedTensor(nn.Module):
    """A packed tensor whose small hard-anchor ranges reload byte-exactly."""

    def __init__(
        self,
        base: packed.PackedQuantizedTensor,
        segments: Sequence[tuple[int, torch.Tensor]],
    ) -> None:
        super().__init__()
        if not isinstance(base, packed.PackedQuantizedTensor):
            raise Lfm2CausalValueAllocatorFormatError(
                "Exact overlay base must be a packed tensor."
            )
        object.__setattr__(self, "_base", base)
        self.shape = base.shape
        self.source_numel = base.source_numel
        self.source_dtype_name = base.source_dtype_name
        self._starts = []
        for index, (start, values) in enumerate(segments):
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or not isinstance(values, torch.Tensor)
                or values.ndim != 1
                or values.dtype != base.source_dtype
                or start + int(values.numel()) > base.source_numel
            ):
                raise Lfm2CausalValueAllocatorFormatError(
                    "Exact overlay runtime segment is invalid."
                )
            self._starts.append(start)
            self.register_buffer(
                f"exact_{index:04d}",
                values,
                persistent=False,
            )

    @property
    def source_dtype(self) -> torch.dtype:
        return self._base.source_dtype

    @property
    def exact_nbytes(self) -> int:
        return sum(
            int(getattr(self, f"exact_{index:04d}").numel())
            * int(getattr(self, f"exact_{index:04d}").element_size())
            for index in range(len(self._starts))
        )

    def dequantize(self) -> torch.Tensor:
        restored = self._base.dequantize()
        flat = restored.reshape(-1)
        for index, start in enumerate(self._starts):
            values = getattr(self, f"exact_{index:04d}")
            flat[start : start + int(values.numel())].copy_(values)
        return restored


def _install_exact_overlays(
    model: nn.Module,
    manifest: Mapping[str, Any],
) -> int:
    allocation = manifest["causal_value_allocation"]
    overlay_metadata = allocation["exact_overlays"]
    names = sorted(manifest["tensors"])
    registry = model.neural_forge_packed_registry
    if len(names) != len(registry):
        raise Lfm2CausalValueAllocatorFormatError(
            "Packed registry differs from the causal tensor inventory."
        )
    stores = dict(zip(names, registry, strict=True))
    overlays: dict[int, ExactOverlayPackedTensor] = {}
    overlay_modules = []
    target_device = registry[0].scales.device
    for name, metadata in overlay_metadata.items():
        base = stores[name]
        runtime_segments = []
        for segment in metadata["segments"]:
            raw = base64.b64decode(segment["raw_base64"], validate=True)
            values = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .clone()
                .view(base.source_dtype)
                .to(target_device)
            )
            if int(values.numel()) != int(segment["value_count"]):
                raise Lfm2CausalValueAllocatorFormatError(
                    "Exact overlay runtime value count changed after decoding."
                )
            runtime_segments.append((int(segment["flat_start"]), values))
        overlay = ExactOverlayPackedTensor(base, runtime_segments)
        overlays[id(base)] = overlay
        overlay_modules.append(overlay)
    for module in model.modules():
        for attribute in ("_packed_weight", "_packed_bias"):
            current = getattr(module, attribute, None)
            replacement = overlays.get(id(current))
            if replacement is not None:
                object.__setattr__(module, attribute, replacement)
    model.add_module(
        "neural_forge_exact_overlay_registry",
        nn.ModuleList(overlay_modules),
    )
    expected_tied = overlays.get(id(stores["model.embed_tokens.weight"]))
    if (
        expected_tied is None
        or model.model.embed_tokens._packed_weight is not expected_tied
        or model.lm_head._packed_weight is not expected_tied
    ):
        raise Lfm2CausalValueAllocatorFormatError(
            "Exact protected token rows did not remain tied on fresh reload."
        )
    observed = sum(value.exact_nbytes for value in overlay_modules)
    expected = allocation["exact_byte_exchange"]["exact_source_value_bytes"]
    if observed != expected:
        raise Lfm2CausalValueAllocatorFormatError(
            "Exact overlay runtime bytes differ from checkpoint accounting."
        )
    return observed


def load_lfm2_causal_value_fold(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    max_checkpoint_bytes: int | None = None,
    max_packed_runtime_bytes: int | None = None,
) -> nn.Module:
    """Freshly reload the standalone artifact through the strict base loader."""

    _root, manifest = _validated_causal_manifest(checkpoint_directory)
    model = packed.load_lfm2_value_fold_control(
        checkpoint_directory,
        device=device,
        max_checkpoint_bytes=max_checkpoint_bytes,
        max_packed_runtime_bytes=max_packed_runtime_bytes,
    )
    exact_runtime_bytes = _install_exact_overlays(model, manifest)
    total_runtime_bytes = (
        model.neural_forge_load_report["packed_runtime_bytes"] + exact_runtime_bytes
    )
    if (
        max_packed_runtime_bytes is not None
        and total_runtime_bytes > max_packed_runtime_bytes
    ):
        raise Lfm2CausalValueAllocatorBudgetError(
            "Packed plus exact-overlay runtime bytes exceed max_packed_runtime_bytes."
        )
    model.neural_forge_load_report = {
        **model.neural_forge_load_report,
        "causal_allocator_policy_id": POLICY_ID,
        "causal_allocation_sha256": manifest["causal_value_allocation"][
            "allocation_sha256"
        ],
        "causal_evidence_sha256": manifest["causal_value_allocation"]["evidence"][
            "evidence_sha256"
        ],
        "exact_source_dtype_overlay_bytes": exact_runtime_bytes,
        "total_packed_plus_exact_runtime_bytes": total_runtime_bytes,
    }
    return model


def load_lfm2_causal_value_tokenizer(
    checkpoint_directory: str | os.PathLike[str],
) -> Any:
    _validated_causal_manifest(checkpoint_directory)
    return packed.load_lfm2_value_fold_tokenizer(checkpoint_directory)


def inspect_lfm2_causal_value_fold(
    checkpoint_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    root, manifest = _validated_causal_manifest(checkpoint_directory)
    baseline = packed.inspect_lfm2_value_fold_control(root)
    allocation = manifest["causal_value_allocation"]
    exchange = allocation["exact_byte_exchange"]
    return {
        **baseline,
        "allocator_policy_id": POLICY_ID,
        "allocation_sha256": allocation["allocation_sha256"],
        "causal_evidence_sha256": allocation["evidence"]["evidence_sha256"],
        "frontier_sha256": allocation["evidence"]["frontier_sha256"],
        "protected_group_count": allocation["counts"]["protected_group_count"],
        "eligible_group_count": allocation["counts"]["eligible_group_count"],
        "lower_group_count": allocation["counts"]["lower_group_count"],
        "exact_source_dtype_overlay_bytes": exchange["exact_source_value_bytes"],
        "exact_overlay_manifest_base64_characters": exchange[
            "base64_payload_characters_inside_manifest"
        ],
        "additional_lower_groups_paid_for_exact_overlays": exchange[
            "additional_lower_groups_paid_for_exact_overlays"
        ],
        "compensating_payload_bytes_saved": exchange[
            "compensating_payload_bytes_saved"
        ],
        "embedding_lower_group_count": allocation["tensors"][
            "model.embed_tokens.weight"
        ]["lower_7bit_group_count"],
        "all_norm_qk_conv_embedding_groups_base_width": all(
            value["lower_7bit_group_count"] == 0
            for value in allocation["tensors"].values()
            if value["role"] in _PROTECTED_ROLES
        ),
        "parent_forward_calls": allocation["execution_audit"]["parent_forward_calls"],
        "optimizer_steps": allocation["execution_audit"]["optimizer_steps"],
        "backward_calls": allocation["execution_audit"]["backward_calls"],
    }


__all__ = [
    "ALLOCATION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "Lfm2CausalValueAllocatorBudgetError",
    "Lfm2CausalValueAllocatorError",
    "Lfm2CausalValueAllocatorFormatError",
    "Lfm2CausalValueEvidence",
    "Lfm2CausalValueLayerSensitivity",
    "POLICY_ID",
    "TARGET_REDUCTION",
    "TOKEN_SELECTION_SCHEMA",
    "derive_lfm2_causal_value_evidence",
    "export_lfm2_causal_value_fold",
    "inspect_lfm2_causal_value_fold",
    "load_lfm2_causal_value_fold",
    "load_lfm2_causal_value_tokenizer",
    "verify_lfm2_causal_value_evidence",
]
