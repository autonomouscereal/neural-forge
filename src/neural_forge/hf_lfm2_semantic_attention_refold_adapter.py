"""Closed-form semantic-span attention refolding for a saved dense LFM2 child.

The semantic lattice is a teacher, not a weight-matching target.  This module
turns its probability/context evidence and an authenticated actual-child
macrocell replay into endpoint cases for the existing matrix-free Q/K
compiler, then solves an analytic residual output writer while holding V
fixed.  FIT, FIT-CHECK, SELECTION, and VALIDATION remain disjoint.  Candidates
are frozen before VALIDATION and dense Q/K/V/O is copied into the existing
child topology only after every applicable held-out gate passes.

No model is executed here.  There are no gradients, optimizers, training
steps, pruning operations, quantization operations, or numerical parent/child
weight-equality objectives.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2CompiledEndpointQkResidualFold,
    Lfm2DenseQkvoSourceInventory,
    Lfm2EndpointQkMetrics,
    Lfm2EndpointQkResidualCase,
    Lfm2EndpointQkResidualEvidenceBank,
    Lfm2EndpointQkResidualFoldPolicy,
    Lfm2EndpointQkvoSeed,
    build_lfm2_endpoint_qkvo_seed,
    compile_lfm2_cached_endpoint_qk_residual_fold,
    evaluate_lfm2_endpoint_qkvo_bank,
    replay_lfm2_endpoint_qkvo,
    seal_lfm2_endpoint_qk_fit_bank,
    seal_lfm2_endpoint_qk_fit_check_bank,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    Lfm2SemanticSpanCpcfPhaseAlignment,
)
from neural_forge.hf_semantic_attention_lattice import (
    SemanticAttentionExpansionTransport,
    SemanticAttentionRephasedProjection,
)


LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA = "neural-forge-lfm2-semantic-attention-refold-v1"
LFM2_SEMANTIC_ATTENTION_REPLAY_ENDPOINT_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-replay-endpoint-v1"
)
_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-attention-installation-v1"
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
_INSTALLATION_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)
_ZERO_REFOLD_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "model_forward_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)


class Lfm2SemanticAttentionRefoldError(RuntimeError):
    """Semantic attention evidence, validation, or installation failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic attention refolding requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _finite_cpu_tensor(
    value: Any,
    name: str,
    *,
    ndim: int,
    dtype: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or (dtype is not None and value.dtype != dtype)
        or not bool(torch.isfinite(value).all().item())
    ):
        expected = "" if dtype is None else f" {dtype}"
        raise Lfm2SemanticAttentionRefoldError(
            f"{name} must be a detached finite rank-{ndim}{expected} tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _installation_receipt_body(
    *,
    layer_index: int,
    evidence_sha256: Sequence[str],
    model_state_before_sha256: str,
    model_state_after_sha256: str,
    changed_state_keys: Sequence[str],
    parameter_element_count: int,
) -> dict[str, Any]:
    evidence = tuple(
        _digest(value, "transport evidence SHA-256") for value in evidence_sha256
    )
    changed = tuple(changed_state_keys)
    before = _digest(model_state_before_sha256, "model state before")
    after = _digest(model_state_after_sha256, "model state after")
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or not evidence
        or not changed
        or any(not isinstance(value, str) or not value for value in changed)
        or isinstance(parameter_element_count, bool)
        or not isinstance(parameter_element_count, int)
        or parameter_element_count <= 0
        or before == after
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Attention installation receipt inputs are incomplete."
        )
    return {
        "schema": _INSTALLATION_SCHEMA,
        "layer_index": layer_index,
        "installed": True,
        "transport_evidence_sha256": list(evidence),
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "changed_state_keys": list(changed),
        "parameter_element_count_before": parameter_element_count,
        "parameter_element_count_after": parameter_element_count,
        "parameter_count_changed": False,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(_INSTALLATION_ZERO_OPERATIONS),
    }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionReplayEndpoint:
    """Cached analytic child replay and encoded parent endpoint for one case."""

    batch_index: int
    layer_index: int
    alignment_sha256: str
    cache_key: str
    bridge_artifact_sha256: str
    captured_child_model_state_sha256: str
    target_query_inputs: Any = field(repr=False, compare=False)
    target_key_value_inputs: Any = field(repr=False, compare=False)
    actual_child_pre_attention_residual: Any = field(repr=False, compare=False)
    parent_post_attention_residual_target_space: Any = field(
        repr=False,
        compare=False,
    )
    query_row_sqrt_weights: Any = field(repr=False, compare=False)
    replay_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
            or isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Replay batch/layer coordinates are invalid."
            )
        for value, name in (
            (self.alignment_sha256, "alignment SHA-256"),
            (self.cache_key, "cache key"),
            (self.bridge_artifact_sha256, "bridge artifact SHA-256"),
            (self.captured_child_model_state_sha256, "captured child state SHA-256"),
        ):
            _digest(value, name)
        query = _finite_cpu_tensor(
            self.target_query_inputs,
            "target_query_inputs",
            ndim=3,
            dtype=torch.bfloat16,
        )
        key_value = _finite_cpu_tensor(
            self.target_key_value_inputs,
            "target_key_value_inputs",
            ndim=3,
            dtype=torch.bfloat16,
        )
        actual_pre = _finite_cpu_tensor(
            self.actual_child_pre_attention_residual,
            "actual_child_pre_attention_residual",
            ndim=3,
        ).to(torch.float32)
        parent_post = _finite_cpu_tensor(
            self.parent_post_attention_residual_target_space,
            "parent_post_attention_residual_target_space",
            ndim=3,
        ).to(torch.float32)
        weights = _finite_cpu_tensor(
            self.query_row_sqrt_weights,
            "query_row_sqrt_weights",
            ndim=2,
        ).to(torch.float64)
        batch, queries, hidden = map(int, query.shape)
        if (
            tuple(key_value.shape[0::2]) != (batch, hidden)
            or tuple(actual_pre.shape) != (batch, queries, hidden)
            or tuple(parent_post.shape) != tuple(actual_pre.shape)
            or tuple(weights.shape) != (batch, queries)
            or bool((weights <= 0.0).any().item())
            or int(key_value.shape[1]) < queries
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Actual-child replay tensors do not share one suffix-causal geometry."
            )
        object.__setattr__(self, "target_query_inputs", query)
        object.__setattr__(self, "target_key_value_inputs", key_value)
        object.__setattr__(self, "actual_child_pre_attention_residual", actual_pre)
        object.__setattr__(
            self,
            "parent_post_attention_residual_target_space",
            parent_post,
        )
        object.__setattr__(self, "query_row_sqrt_weights", weights)
        object.__setattr__(self, "replay_sha256", canonical_sha256(self._body()))

    @property
    def parent_endpoint_target(self) -> Any:
        return (
            self.parent_post_attention_residual_target_space
            - self.actual_child_pre_attention_residual
        ).contiguous()

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_ATTENTION_REPLAY_ENDPOINT_SCHEMA,
            "batch_index": self.batch_index,
            "layer_index": self.layer_index,
            "alignment_sha256": self.alignment_sha256,
            "cache_key": self.cache_key,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "captured_child_model_state_sha256": (
                self.captured_child_model_state_sha256
            ),
            "tensors": {
                name: _tensor_sha256(getattr(self, name))
                for name in (
                    "target_query_inputs",
                    "target_key_value_inputs",
                    "actual_child_pre_attention_residual",
                    "parent_post_attention_residual_target_space",
                    "query_row_sqrt_weights",
                )
            },
            "endpoint_objective": (
                "encoded_parent_post_attention_minus_actual_child_pre_attention"
            ),
            "numeric_parent_child_weight_match_objective": False,
            "analytic_macrocell_replay_consumed": True,
            "refold_operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "replay_sha256": self.replay_sha256}


@dataclass(frozen=True, slots=True)
class _CaseSource:
    batch_index: int
    alignment: Lfm2SemanticSpanCpcfPhaseAlignment
    transport: SemanticAttentionExpansionTransport | SemanticAttentionRephasedProjection
    replay: Lfm2SemanticAttentionReplayEndpoint
    evidence_sha256: str
    compiler_lineage_sha256: str
    query_groups: tuple[tuple[int, ...], ...]
    query_group_weights: tuple[tuple[float, ...], ...]


def _attention_module(target_model: Any, layer_index: int) -> Any:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SemanticAttentionRefoldError(
            f"Target model has no layer {layer_index}."
        )
    attention = getattr(layers[layer_index], "self_attn", None)
    required = ("q_proj", "k_proj", "v_proj", "out_proj", "q_layernorm", "k_layernorm")
    if attention is None or any(
        getattr(attention, name, None) is None for name in required
    ):
        raise Lfm2SemanticAttentionRefoldError(
            f"Layer {layer_index} is not one ordinary dense LFM2 attention module."
        )
    return attention


def _source_contract(
    evidence: Any,
) -> tuple[
    Lfm2DenseQkvoSourceInventory,
    tuple[tuple[int, ...], ...],
    tuple[tuple[float, ...], ...],
    str,
]:
    inventory = getattr(evidence, "source_inventory", None)
    groups = tuple(
        tuple(int(member) for member in group)
        for group in getattr(evidence, "query_groups", ())
    )
    weights = tuple(
        tuple(float(weight) for weight in group)
        for group in getattr(evidence, "query_group_weights", ())
    )
    lineage = _digest(
        getattr(evidence, "compiler_lineage_sha256", None),
        "compiler lineage SHA-256",
    )
    if (
        not isinstance(inventory, Lfm2DenseQkvoSourceInventory)
        or inventory.verify() != inventory.inventory_sha256
        or not groups
        or len(groups) != len(weights)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Evidence lacks its typed parent Q/K/V/O inventory or original head groups."
        )
    return inventory, groups, weights, lineage


def _normalized_case_source(
    evidence: Any,
    *,
    layer_index: int,
    cache_key: str,
    bridge_artifact_sha256: str,
) -> _CaseSource:
    alignment = getattr(evidence, "alignment", None)
    transport = getattr(evidence, "transport", None)
    replay = getattr(evidence, "actual_child_replay", None)
    batch_index = getattr(evidence, "batch_index", None)
    evidence_sha = _digest(
        getattr(evidence, "evidence_sha256", None),
        "transport evidence SHA-256",
    )
    inventory, groups, weights, lineage = _source_contract(evidence)
    del inventory
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or batch_index < 0
        or getattr(evidence, "layer_index", None) != layer_index
        or not isinstance(alignment, Lfm2SemanticSpanCpcfPhaseAlignment)
        or not isinstance(
            transport,
            (SemanticAttentionExpansionTransport, SemanticAttentionRephasedProjection),
        )
        or not isinstance(replay, Lfm2SemanticAttentionReplayEndpoint)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Semantic attention evidence lacks typed alignment/transport/replay data."
        )
    if (
        alignment.alignment_sha256 != getattr(evidence, "alignment_sha256", None)
        or transport.semantic_alignment_sha256 != alignment.alignment_sha256
        or replay.batch_index != batch_index
        or replay.layer_index != layer_index
        or replay.alignment_sha256 != alignment.alignment_sha256
        or replay.cache_key != cache_key
        or replay.bridge_artifact_sha256 != bridge_artifact_sha256
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Alignment, transport, cache, bridge, and replay identities diverge."
        )
    capability = dict(getattr(evidence, "capability", {}))
    history = dict(getattr(evidence, "history", {}))
    if (
        capability.get("exact_history_supported") is not True
        or capability.get("blocking_reasons") != []
        or capability.get("missing_donor_prefix_positions") != []
        or capability.get("layer_index") != layer_index
        or history.get("token_count") != alignment.donor_full_prefix_tokens
        or history.get("source_capability_sha256")
        != capability.get("capability_sha256")
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Complete-causal Q/K/V history is absent or not bound to the layer."
        )
    alignment.require_probability_preserving_attention_transport()
    if isinstance(transport, SemanticAttentionExpansionTransport):
        expected_owners = tuple(
            int(value)
            for value in alignment.target_to_donor_span_owner_positions
            if value is not None
        )
        if (
            not transport.inserted_query_rows_are_semantically_exact
            or transport.target_to_donor_boundary_positions
            != alignment.target_to_donor_boundary_positions
            or transport.target_to_donor_span_owner_positions != expected_owners
            or transport.donor_span_multiplicities
            != alignment.donor_span_multiplicities
            or any(value <= 0 for value in transport.donor_span_multiplicities)
            or transport.maximum_probability_mass_error > 1e-12
            or transport.maximum_context_absolute_error > 1e-12
            or not transport.query_target_positions
            or transport.query_target_positions[-1]
            != len(transport.target_to_donor_span_owner_positions) - 1
            or transport.query_donor_positions[-1]
            != alignment.donor_full_prefix_tokens - 1
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Expansion transport is contracted, approximate, or has no exact "
                "terminal suffix query."
            )
        # Exact semantic endpoints are scattered under token splitting.  The
        # endpoint compiler requires its query rows to be a literal key suffix,
        # so each phase contributes only its exact terminal endpoint.  Inserted
        # target rows are deliberately not relabeled as parent-exact queries.
        query_length = 1
        key_length = len(transport.target_to_donor_span_owner_positions)
    else:
        query_length = 1
        key_length = int(transport.probabilities.shape[3])
        if int(transport.probabilities.shape[2]) != key_length:
            raise Lfm2SemanticAttentionRefoldError(
                "Rephased endpoint evidence must cover one complete causal lattice."
            )
    replay_query_length = int(replay.target_query_inputs.shape[1])
    allowed_query_lengths = {query_length, key_length}
    if isinstance(transport, SemanticAttentionExpansionTransport):
        allowed_query_lengths.add(len(transport.query_target_positions))
    if (
        replay_query_length not in allowed_query_lengths
        or int(replay.target_key_value_inputs.shape[1]) != key_length
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Actual-child replay does not cover the teacher query/key suffix."
        )
    returned = getattr(evidence, "to_evidence_dict", lambda: {})()
    if returned.get("evidence_sha256") != evidence_sha:
        raise Lfm2SemanticAttentionRefoldError(
            "Transport evidence does not reproduce its external seal."
        )
    return _CaseSource(
        batch_index=batch_index,
        alignment=alignment,
        transport=transport,
        replay=replay,
        evidence_sha256=evidence_sha,
        compiler_lineage_sha256=lineage,
        query_groups=groups,
        query_group_weights=weights,
    )


def _split_sources(
    sources: Sequence[_CaseSource],
) -> Mapping[str, tuple[_CaseSource, ...]]:
    grouped: dict[int, list[_CaseSource]] = {}
    for source in sources:
        grouped.setdefault(source.batch_index, []).append(source)
    batch_indices = tuple(sorted(grouped))
    count = len(batch_indices)
    if count < 4 or any(
        len({value.alignment.alignment_sha256 for value in values}) != len(values)
        for values in grouped.values()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "At least four unique batches with nonrepeated phases are required "
            "for four trajectory-isolated banks."
        )
    counts = [max(1, count // 2), max(1, (3 * count) // 16), max(1, count // 8), 1]
    while sum(counts) > count:
        for index in (0, 1, 2):
            if counts[index] > 1 and sum(counts) > count:
                counts[index] -= 1
    counts[3] += count - sum(counts)
    cursor = 0
    result: dict[str, tuple[_CaseSource, ...]] = {}
    for partition, size in zip(_PARTITIONS, counts, strict=True):
        selected_batches = batch_indices[cursor : cursor + size]
        result[partition] = tuple(
            source
            for batch_index in selected_batches
            for source in sorted(
                grouped[batch_index],
                key=lambda value: value.alignment.phase,
            )
        )
        cursor += size
    if cursor != count or any(not result[name] for name in _PARTITIONS):
        raise AssertionError("Internal four-way split construction failed.")
    return MappingProxyType(result)


def _case_from_source(
    source: _CaseSource,
    *,
    layer_index: int,
    cache_key: str,
    bridge_artifact_sha256: str,
    trajectory_row_weight_scale: float,
) -> Lfm2EndpointQkResidualCase:
    if (
        not math.isfinite(trajectory_row_weight_scale)
        or trajectory_row_weight_scale <= 0.0
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Trajectory row-weight scale must be finite and positive."
        )
    raw_evidence = source.transport.endpoint_residual_evidence()
    # One terminal query is a valid suffix case even when earlier exact donor
    # endpoints are scattered among inserted target subtokens.
    evidence = {
        **raw_evidence,
        "parent_probabilities": raw_evidence["parent_probabilities"][:, :, -1:, :],
        "parent_causal_mask": raw_evidence["parent_causal_mask"][:, :, -1:, :],
        "parent_context": raw_evidence["parent_context"][:, :, -1:, :],
        "query_rope_cos": raw_evidence["query_rope_cos"][:, -1:, :],
        "query_rope_sin": raw_evidence["query_rope_sin"][:, -1:, :],
    }
    trajectory = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-semantic-attention-trajectory-v1",
            "cache_key": cache_key,
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "layer_index": layer_index,
            "batch_index": source.batch_index,
            "alignment_sha256": source.alignment.alignment_sha256,
            "transport_evidence_sha256": source.evidence_sha256,
            "actual_child_replay_sha256": source.replay.replay_sha256,
            "compiler_lineage_sha256": source.compiler_lineage_sha256,
            "trajectory_row_weight_scale": trajectory_row_weight_scale,
        }
    )
    return Lfm2EndpointQkResidualCase(
        target_query_inputs=source.replay.target_query_inputs[:, -1:, :],
        target_key_value_inputs=source.replay.target_key_value_inputs,
        parent_probabilities=evidence["parent_probabilities"],
        parent_causal_mask=evidence["parent_causal_mask"],
        parent_endpoint_target=source.replay.parent_endpoint_target[:, -1:, :],
        query_rope_cos=evidence["query_rope_cos"],
        query_rope_sin=evidence["query_rope_sin"],
        key_rope_cos=evidence["key_rope_cos"],
        key_rope_sin=evidence["key_rope_sin"],
        query_row_sqrt_weights=(
            source.replay.query_row_sqrt_weights[:, -1:] * trajectory_row_weight_scale
        ),
        query_groups=source.query_groups,
        query_group_weights=source.query_group_weights,
        trajectory_sha256=trajectory,
        parent_probability_provenance=evidence["parent_probability_provenance"],
        parent_context_replay_relative_rmse=0.0,
        parent_context_replay_maximum_absolute_error=0.0,
        parent_context_replay_rtol=0.0,
        parent_context_replay_atol=0.0,
    )


def _balanced_endpoint_cases(
    sources: Sequence[_CaseSource],
    *,
    layer_index: int,
    cache_key: str,
    bridge_artifact_sha256: str,
) -> tuple[tuple[Lfm2EndpointQkResidualCase, ...], Mapping[str, Any]]:
    torch = _torch()
    selected = tuple(sources)
    if not selected:
        raise Lfm2SemanticAttentionRefoldError(
            "Trajectory balancing requires at least one semantic phase."
        )
    raw_mass: dict[int, float] = {}
    phase_counts: dict[int, int] = {}
    source_weight_sha256: dict[int, list[str]] = {}
    for source in selected:
        terminal = source.replay.query_row_sqrt_weights[:, -1:].to(torch.float64)
        mass = float(terminal.square().sum().item())
        if not math.isfinite(mass) or mass <= 0.0:
            raise Lfm2SemanticAttentionRefoldError(
                "Replay-provided terminal row weights have no positive mass."
            )
        raw_mass[source.batch_index] = raw_mass.get(source.batch_index, 0.0) + mass
        phase_counts[source.batch_index] = phase_counts.get(source.batch_index, 0) + 1
        source_weight_sha256.setdefault(source.batch_index, []).append(
            _tensor_sha256(terminal)
        )
    scales = {
        batch_index: 1.0 / math.sqrt(mass) for batch_index, mass in raw_mass.items()
    }
    cases = tuple(
        _case_from_source(
            source,
            layer_index=layer_index,
            cache_key=cache_key,
            bridge_artifact_sha256=bridge_artifact_sha256,
            trajectory_row_weight_scale=scales[source.batch_index],
        )
        for source in selected
    )
    normalized_mass = {batch_index: 0.0 for batch_index in raw_mass}
    for source, case in zip(selected, cases, strict=True):
        normalized_mass[source.batch_index] += float(
            case.query_row_sqrt_weights.square().sum().item()
        )
    if any(
        not math.isclose(value, 1.0, rel_tol=1e-10, abs_tol=1e-12)
        for value in normalized_mass.values()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Whole-trajectory row-weight balancing did not conserve unit mass."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-trajectory-row-balance-v1",
        "rule": (
            "preserve_replay_phase_ratios_and_normalize_each_trajectory_to_unit_mass"
        ),
        "groups": [
            {
                "batch_index": batch_index,
                "phase_count": phase_counts[batch_index],
                "raw_squared_row_weight_mass": raw_mass[batch_index],
                "sqrt_weight_scale": scales[batch_index],
                "normalized_squared_row_weight_mass": normalized_mass[batch_index],
                "source_terminal_row_weight_sha256": source_weight_sha256[batch_index],
            }
            for batch_index in sorted(raw_mass)
        ],
        "longer_trajectories_receive_more_total_weight": False,
        "relative_replay_phase_weights_preserved": True,
    }
    return cases, MappingProxyType({**body, "balance_sha256": canonical_sha256(body)})


def _seed(
    target_model: Any,
    *,
    layer_index: int,
    source_inventory: Lfm2DenseQkvoSourceInventory,
    source_artifact_sha256: str,
) -> Lfm2EndpointQkvoSeed:
    torch = _torch()
    attention = _attention_module(target_model, layer_index)
    tensors = {
        name: getattr(getattr(attention, module_name), "weight", None)
        for name, module_name in (
            ("query", "q_proj"),
            ("key", "k_proj"),
            ("value", "v_proj"),
            ("output", "out_proj"),
            ("query_norm", "q_layernorm"),
            ("key_norm", "k_layernorm"),
        )
    }
    if any(
        not isinstance(value, torch.nn.Parameter) or value.dtype != torch.bfloat16
        for value in tensors.values()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Target Q/K/V/O and Q/K norms must be frozen dense BF16 Parameters."
        )
    epsilon = getattr(getattr(target_model, "config", None), "norm_eps", None)
    if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)):
        raise Lfm2SemanticAttentionRefoldError("Target model has no finite norm_eps.")
    cpu = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in tensors.items()
    }
    return build_lfm2_endpoint_qkvo_seed(
        query_weight=cpu["query"],
        key_weight=cpu["key"],
        value_weight=cpu["value"],
        output_weight=cpu["output"],
        query_norm_weight=cpu["query_norm"],
        key_norm_weight=cpu["key_norm"],
        norm_epsilon=float(epsilon),
        source_inventory=source_inventory,
        source_artifact_sha256=source_artifact_sha256,
        target_query_weight_template=cpu["query"],
        target_key_weight_template=cpu["key"],
        target_value_weight_template=cpu["value"],
        target_output_weight_template=cpu["output"],
    )


def _execution_policy(case_count: int) -> Lfm2EndpointQkResidualFoldPolicy:
    torch = _torch()
    devices = (
        tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else ()
    )
    return Lfm2EndpointQkResidualFoldPolicy(
        solver_device=devices[0] if devices else "cpu",
        compute_devices=devices,
        maximum_fit_cases=case_count,
        maximum_fit_check_cases=case_count,
    )


def _evaluate_bank(
    bank: Lfm2EndpointQkResidualEvidenceBank,
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any,
    key_weight: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> Lfm2EndpointQkMetrics:
    return evaluate_lfm2_endpoint_qkvo_bank(
        bank,
        seed,
        query_weight=query_weight,
        key_weight=key_weight,
        compute_devices=policy.compute_devices,
    )


def _metrics_nonregressing(
    candidate: Lfm2EndpointQkMetrics,
    baseline: Lfm2EndpointQkMetrics,
    *,
    tolerance: float,
) -> bool:
    return (
        candidate.endpoint_relative_rmse <= baseline.endpoint_relative_rmse + tolerance
        and candidate.probability_relative_rmse
        <= baseline.probability_relative_rmse + tolerance
        and candidate.js_divergence_mean <= baseline.js_divergence_mean + tolerance
        and candidate.js_divergence_worst <= baseline.js_divergence_worst + tolerance
        and candidate.top1_route_agreement_fraction + tolerance
        >= baseline.top1_route_agreement_fraction
        and candidate.maximum_causal_probability_leakage == 0.0
    )


@dataclass(frozen=True, slots=True)
class _WriterCase:
    context: Any = field(repr=False, compare=False)
    endpoint_target: Any = field(repr=False, compare=False)
    row_weights: Any = field(repr=False, compare=False)
    trajectory_sha256: str
    case_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        context = _finite_cpu_tensor(
            self.context,
            "output-writer context",
            ndim=3,
            dtype=torch.float64,
        )
        endpoint = _finite_cpu_tensor(
            self.endpoint_target,
            "output-writer endpoint target",
            ndim=3,
            dtype=torch.float64,
        )
        weights = _finite_cpu_tensor(
            self.row_weights,
            "output-writer row weights",
            ndim=2,
            dtype=torch.float64,
        )
        if (
            tuple(context.shape[:2]) != tuple(endpoint.shape[:2])
            or tuple(weights.shape) != tuple(context.shape[:2])
            or bool((weights <= 0.0).any().item())
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Output-writer case tensors are not row aligned and positive."
            )
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "endpoint_target", endpoint)
        object.__setattr__(self, "row_weights", weights)
        object.__setattr__(
            self,
            "trajectory_sha256",
            _digest(self.trajectory_sha256, "writer trajectory SHA-256"),
        )
        object.__setattr__(
            self,
            "case_sha256",
            canonical_sha256(
                {
                    "schema": "neural-forge-lfm2-semantic-output-writer-case-v1",
                    "trajectory_sha256": self.trajectory_sha256,
                    "context_sha256": _tensor_sha256(context),
                    "endpoint_target_sha256": _tensor_sha256(endpoint),
                    "row_weights_sha256": _tensor_sha256(weights),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _WriterMetrics:
    case_count: int
    row_count: int
    endpoint_rmse: float
    endpoint_relative_rmse: float
    endpoint_maximum_absolute_error: float
    endpoint_cosine_similarity_mean: float
    endpoint_cosine_similarity_worst: float

    def __post_init__(self) -> None:
        values = (
            self.endpoint_rmse,
            self.endpoint_relative_rmse,
            self.endpoint_maximum_absolute_error,
            self.endpoint_cosine_similarity_mean,
            self.endpoint_cosine_similarity_worst,
        )
        if (
            self.case_count <= 0
            or self.row_count <= 0
            or any(not math.isfinite(value) for value in values)
            or any(value < 0.0 for value in values[:3])
            or any(value < -1.0 or value > 1.0 for value in values[3:])
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Output-writer metrics must be finite nonnegative measurements."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "row_count": self.row_count,
            "endpoint_rmse": self.endpoint_rmse,
            "endpoint_relative_rmse": self.endpoint_relative_rmse,
            "endpoint_maximum_absolute_error": (self.endpoint_maximum_absolute_error),
            "endpoint_cosine_similarity_mean": self.endpoint_cosine_similarity_mean,
            "endpoint_cosine_similarity_worst": (self.endpoint_cosine_similarity_worst),
        }


@dataclass(frozen=True, slots=True)
class _OutputWriterProposal:
    output_weight: Any = field(repr=False, compare=False)
    changed: bool
    fit_improved: bool
    selected_alpha: float | None
    baseline_fit_metrics: _WriterMetrics
    candidate_fit_metrics: _WriterMetrics
    ridge: float
    effective_ridge: float
    fit_context_sha256: str
    proposal_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        output = _finite_cpu_tensor(
            self.output_weight,
            "output writer proposal",
            ndim=2,
            dtype=torch.bfloat16,
        )
        if (
            not math.isfinite(self.ridge)
            or self.ridge <= 0.0
            or not math.isfinite(self.effective_ridge)
            or self.effective_ridge <= 0.0
            or (self.selected_alpha is None) != (not self.changed)
            or self.fit_improved != self.changed
            or (
                self.changed
                and (
                    self.selected_alpha not in (0.125, 0.25, 0.5, 0.75, 1.0)
                    or self.candidate_fit_metrics.endpoint_relative_rmse
                    >= self.baseline_fit_metrics.endpoint_relative_rmse
                )
            )
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Output-writer proposal metadata is inconsistent."
            )
        _digest(self.fit_context_sha256, "FIT context SHA-256")
        object.__setattr__(self, "output_weight", output)
        object.__setattr__(
            self,
            "proposal_sha256",
            canonical_sha256(self._body()),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-semantic-output-writer-proposal-v1",
            "output_weight_sha256": _tensor_sha256(self.output_weight),
            "changed": self.changed,
            "fit_improved": self.fit_improved,
            "selected_alpha": self.selected_alpha,
            "baseline_fit_metrics": self.baseline_fit_metrics.as_dict(),
            "candidate_fit_metrics": self.candidate_fit_metrics.as_dict(),
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "fit_context_sha256": self.fit_context_sha256,
            "solver": "closed_form_weighted_residual_ridge_plus_bfloat16_frontier",
            "value_weight_fixed": True,
            "fit_only": True,
            "numeric_parent_child_weight_match_objective": False,
            "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "proposal_sha256": self.proposal_sha256}


def _writer_cases(
    bank: Lfm2EndpointQkResidualEvidenceBank,
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any,
    key_weight: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[_WriterCase, ...]:
    torch = _torch()
    devices = policy.compute_devices or ("cpu",)
    result = []
    for case_index, case in enumerate(bank.cases):
        device = devices[case_index % len(devices)]
        probabilities, replayed_endpoint = replay_lfm2_endpoint_qkvo(
            case,
            seed,
            query_weight=query_weight,
            key_weight=key_weight,
            device=device,
        )
        resolved = torch.device(device)
        key_inputs = case.target_key_value_inputs.to(device=resolved)
        batch, key_length, _hidden = map(int, key_inputs.shape)
        query_length = int(case.target_query_inputs.shape[1])
        with torch.inference_mode():
            value_heads = (
                (key_inputs @ seed.value_weight.to(device=resolved).T)
                .reshape(
                    batch,
                    key_length,
                    seed.key_value_heads,
                    seed.head_dim,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            repeated_value = value_heads.repeat_interleave(
                seed.queries_per_key_value,
                dim=1,
            ).to(torch.float64)
            context_on_device = (
                probabilities.to(device=resolved, dtype=torch.float64)
                .matmul(repeated_value)
                .permute(0, 2, 1, 3)
                .reshape(
                    batch,
                    query_length,
                    seed.query_heads * seed.head_dim,
                )
                .contiguous()
            )
            reconstructed_endpoint = (
                context_on_device
                @ seed.output_weight.to(device=resolved, dtype=torch.float64).T
            ).cpu()
            context = context_on_device.cpu()
        if (
            tuple(context.shape[-1:]) != tuple(seed.output_weight.shape[1:])
            or tuple(case.parent_endpoint_target.shape[-1:])
            != tuple(seed.output_weight.shape[:1])
            or not torch.allclose(
                reconstructed_endpoint,
                replayed_endpoint.to(torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Inherited-V context reconstruction does not reproduce exact replay."
            )
        result.append(
            _WriterCase(
                context=context,
                endpoint_target=case.parent_endpoint_target.to(torch.float64),
                row_weights=case.query_row_sqrt_weights.to(torch.float64).square(),
                trajectory_sha256=case.trajectory_sha256,
            )
        )
    return tuple(result)


def _writer_metrics(
    cases: Sequence[_WriterCase],
    output_weight: Any,
) -> _WriterMetrics:
    torch = _torch()
    selected = tuple(cases)
    if not selected:
        raise Lfm2SemanticAttentionRefoldError(
            "Output-writer metrics require at least one case."
        )
    output = output_weight.detach().to(device="cpu", dtype=torch.float64)
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_target = torch.zeros((), dtype=torch.float64)
    weighted_elements = torch.zeros((), dtype=torch.float64)
    weighted_cosine = torch.zeros((), dtype=torch.float64)
    cosine_weight = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    worst_cosine = 1.0
    rows = 0
    for case in selected:
        prediction = case.context @ output.T
        difference = prediction - case.endpoint_target
        weights = case.row_weights.unsqueeze(-1)
        squared_error += (difference.square() * weights).sum()
        squared_target += (case.endpoint_target.square() * weights).sum()
        weighted_elements += weights.sum() * int(case.endpoint_target.shape[-1])
        maximum = max(maximum, float(difference.abs().max().item()))
        prediction_norm = torch.linalg.vector_norm(prediction, dim=-1)
        target_norm = torch.linalg.vector_norm(case.endpoint_target, dim=-1)
        cosine_denominator = prediction_norm * target_norm
        both_zero = (prediction_norm == 0.0) & (target_norm == 0.0)
        cosine = torch.where(
            both_zero,
            torch.ones_like(cosine_denominator),
            torch.where(
                cosine_denominator > torch.finfo(torch.float64).tiny,
                (prediction * case.endpoint_target).sum(dim=-1)
                / cosine_denominator.clamp_min(torch.finfo(torch.float64).tiny),
                torch.zeros_like(cosine_denominator),
            ),
        ).clamp(-1.0, 1.0)
        weighted_cosine += (cosine * case.row_weights).sum()
        cosine_weight += case.row_weights.sum()
        worst_cosine = min(worst_cosine, float(cosine.min().item()))
        rows += int(case.row_weights.numel())
    denominator = weighted_elements.clamp_min(torch.finfo(torch.float64).tiny)
    rmse = torch.sqrt(squared_error / denominator)
    target_scale = torch.sqrt(squared_target / denominator).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return _WriterMetrics(
        case_count=len(selected),
        row_count=rows,
        endpoint_rmse=float(rmse.item()),
        endpoint_relative_rmse=float((rmse / target_scale).item()),
        endpoint_maximum_absolute_error=maximum,
        endpoint_cosine_similarity_mean=float((weighted_cosine / cosine_weight).item()),
        endpoint_cosine_similarity_worst=worst_cosine,
    )


def _compile_output_writer(
    fit_cases: Sequence[_WriterCase],
    incumbent_output_weight: Any,
    *,
    solver_device: str,
    ridge: float = 1e-4,
) -> _OutputWriterProposal:
    torch = _torch()
    selected = tuple(fit_cases)
    if not selected:
        raise Lfm2SemanticAttentionRefoldError("Output-writer FIT bank is empty.")
    incumbent = incumbent_output_weight.detach().to(
        device="cpu",
        dtype=torch.bfloat16,
    )
    baseline = _writer_metrics(selected, incumbent)
    contexts = torch.cat(
        tuple(case.context.reshape(-1, case.context.shape[-1]) for case in selected),
        dim=0,
    )
    targets = torch.cat(
        tuple(
            case.endpoint_target.reshape(-1, case.endpoint_target.shape[-1])
            for case in selected
        ),
        dim=0,
    )
    weights = torch.cat(
        tuple(case.row_weights.reshape(-1) for case in selected),
        dim=0,
    )
    context_sha = canonical_sha256(
        {
            "case_sha256": [case.case_sha256 for case in selected],
        }
    )
    device = torch.device(solver_device)
    x = contexts.to(device=device, dtype=torch.float64)
    y = targets.to(device=device, dtype=torch.float64)
    root = torch.sqrt(weights.to(device=device, dtype=torch.float64)).unsqueeze(1)
    incumbent_transpose = incumbent.T.to(device=device, dtype=torch.float64)
    weighted_x = x * root
    residual = (y - x @ incumbent_transpose) * root
    gram = weighted_x.T @ weighted_x
    scale = float(gram.diagonal().mean().clamp_min(1.0).item())
    effective_ridge = float(ridge) * scale
    identity = torch.eye(int(gram.shape[0]), dtype=torch.float64, device=device)
    with torch.inference_mode():
        delta = torch.linalg.solve(
            gram + effective_ridge * identity,
            weighted_x.T @ residual,
        )
    if not bool(torch.isfinite(delta).all().item()):
        raise Lfm2SemanticAttentionRefoldError(
            "Closed-form output-writer ridge solve produced nonfinite values."
        )
    candidates = []
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        candidate = (
            (incumbent_transpose + alpha * delta)
            .T.to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        if torch.equal(candidate, incumbent):
            continue
        metrics = _writer_metrics(selected, candidate)
        candidates.append((metrics.endpoint_relative_rmse, alpha, candidate, metrics))
    if not candidates:
        return _OutputWriterProposal(
            output_weight=incumbent,
            changed=False,
            fit_improved=False,
            selected_alpha=None,
            baseline_fit_metrics=baseline,
            candidate_fit_metrics=baseline,
            ridge=float(ridge),
            effective_ridge=effective_ridge,
            fit_context_sha256=context_sha,
        )
    _score, alpha, candidate, candidate_metrics = min(
        candidates,
        key=lambda value: (value[0], value[1]),
    )
    improvement = (
        baseline.endpoint_relative_rmse - candidate_metrics.endpoint_relative_rmse
    )
    if improvement <= 1e-8:
        return _OutputWriterProposal(
            output_weight=incumbent,
            changed=False,
            fit_improved=False,
            selected_alpha=None,
            baseline_fit_metrics=baseline,
            candidate_fit_metrics=baseline,
            ridge=float(ridge),
            effective_ridge=effective_ridge,
            fit_context_sha256=context_sha,
        )
    return _OutputWriterProposal(
        output_weight=candidate,
        changed=True,
        fit_improved=True,
        selected_alpha=alpha,
        baseline_fit_metrics=baseline,
        candidate_fit_metrics=candidate_metrics,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
        fit_context_sha256=context_sha,
    )


def _writer_nonregressing(
    candidate: _WriterMetrics,
    baseline: _WriterMetrics,
    *,
    tolerance: float,
) -> bool:
    return (
        candidate.endpoint_relative_rmse <= baseline.endpoint_relative_rmse + tolerance
        and candidate.endpoint_maximum_absolute_error
        <= baseline.endpoint_maximum_absolute_error + tolerance
        and candidate.endpoint_cosine_similarity_mean + tolerance
        >= baseline.endpoint_cosine_similarity_mean
        and candidate.endpoint_cosine_similarity_worst + tolerance
        >= baseline.endpoint_cosine_similarity_worst
    )


def _seal_writer_gate(body: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(body)
    return MappingProxyType(
        {
            **normalized,
            "gate_sha256": canonical_sha256(normalized),
        }
    )


def _gate_output_writer(
    proposal: _OutputWriterProposal,
    *,
    fit_check_cases: Sequence[_WriterCase],
    selection_cases: Sequence[_WriterCase],
    validation_cases: Sequence[_WriterCase],
    incumbent_output_weight: Any,
    tolerance: float,
) -> tuple[Any, Mapping[str, Any]]:
    incumbent = incumbent_output_weight.detach().to(device="cpu").contiguous()
    frozen_candidate_sha256 = _tensor_sha256(proposal.output_weight)
    if not proposal.changed:
        body = {
            "status": "rejected-no-bfloat16-fit-improvement",
            "accepted": False,
            "proposal": proposal.as_dict(),
            "rejection_partition": "fit",
            "fit_check": None,
            "selection": None,
            "validation": None,
        }
        return incumbent, _seal_writer_gate(body)
    reports: dict[str, Any] = {}
    for name, cases in (
        ("fit_check", fit_check_cases),
        ("selection", selection_cases),
        ("validation", validation_cases),
    ):
        baseline = _writer_metrics(cases, incumbent)
        candidate = _writer_metrics(cases, proposal.output_weight)
        if _tensor_sha256(
            proposal.output_weight
        ) != frozen_candidate_sha256 or proposal.proposal_sha256 != canonical_sha256(
            proposal._body()
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Output-writer candidate changed while traversing held-out gates."
            )
        nonregressing = _writer_nonregressing(
            candidate,
            baseline,
            tolerance=tolerance,
        )
        reports[name] = {
            "case_sha256": [case.case_sha256 for case in cases],
            "baseline": baseline.as_dict(),
            "candidate": candidate.as_dict(),
            "nonregressing": nonregressing,
        }
        if not nonregressing:
            body = {
                "status": f"rejected-{name}-regression",
                "accepted": False,
                "proposal": proposal.as_dict(),
                "rejection_partition": name,
                "fit_check": reports.get("fit_check"),
                "selection": reports.get("selection"),
                "validation": reports.get("validation"),
            }
            return incumbent, _seal_writer_gate(body)
    body = {
        "status": "accepted-after-frozen-validation",
        "accepted": True,
        "proposal": proposal.as_dict(),
        "rejection_partition": None,
        "fit_check": reports["fit_check"],
        "selection": reports["selection"],
        "validation": reports["validation"],
        "candidate_frozen_across_heldout_gates": True,
    }
    return proposal.output_weight, _seal_writer_gate(body)


def _parameter_bundle(target_model: Any, layer_index: int) -> Mapping[str, Any]:
    torch = _torch()
    attention = _attention_module(target_model, layer_index)
    prefix = f"model.layers.{layer_index}.self_attn."
    values = {
        prefix + name + ".weight": getattr(getattr(attention, name), "weight", None)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    }
    named = dict(target_model.named_parameters())
    if (
        any(not isinstance(value, torch.nn.Parameter) for value in values.values())
        or any(named.get(name) is not value for name, value in values.items())
        or len({id(value) for value in values.values()}) != 4
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Target attention Q/K/V/O Parameters are missing, aliased, or renamed."
        )
    return MappingProxyType(values)


def _install_qkvo_transaction(
    target_model: Any,
    *,
    layer_index: int,
    state: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], int]:
    torch = _torch()
    parameters = _parameter_bundle(target_model, layer_index)
    suffix_state = {
        "q_proj.weight": state["q_proj.weight"],
        "k_proj.weight": state["k_proj.weight"],
        "v_proj.weight": state["v_proj.weight"],
        "out_proj.weight": state["out_proj.weight"],
    }
    before_model = fingerprint_lfm2_parent(target_model).identity_sha256
    count = _parameter_count(target_model)
    snapshots = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in parameters.items()
    }
    parameter_ids = {name: id(value) for name, value in parameters.items()}
    gradient_ids = {
        name: None if value.grad is None else id(value.grad)
        for name, value in parameters.items()
    }
    gradient_hashes = {
        name: None if value.grad is None else _tensor_sha256(value.grad)
        for name, value in parameters.items()
    }
    prefix = f"model.layers.{layer_index}.self_attn."
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                candidate = suffix_state[name.removeprefix(prefix)]
                if (
                    tuple(candidate.shape) != tuple(parameter.shape)
                    or candidate.dtype != parameter.dtype
                ):
                    raise Lfm2SemanticAttentionRefoldError(
                        f"Compiled state changed the topology of {name}."
                    )
                parameter.copy_(candidate.to(device=parameter.device))
        if (
            _parameter_count(target_model) != count
            or any(id(parameters[name]) != parameter_ids[name] for name in parameters)
            or any(
                (None if parameters[name].grad is None else id(parameters[name].grad))
                != gradient_ids[name]
                for name in parameters
            )
            or any(
                parameters[name].grad is not None
                and _tensor_sha256(parameters[name].grad) != gradient_hashes[name]
                for name in parameters
            )
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Attention installation changed topology, identities, or gradients."
            )
        changed = tuple(
            name
            for name, parameter in parameters.items()
            if not torch.equal(parameter.detach().cpu(), snapshots[name])
        )
        after_model = fingerprint_lfm2_parent(target_model).identity_sha256
        if not changed or after_model == before_model:
            raise Lfm2SemanticAttentionRefoldError(
                "Frozen semantic attention candidate installed no correction."
            )
        return before_model, after_model, changed, count
    except BaseException:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(
                    snapshots[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_model
            or _parameter_count(target_model) != count
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Attention installation failed and exact rollback was not proven."
            )
        raise


def compile_install_lfm2_semantic_attention_postfold(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Any],
    cache_key: str,
    bridge_artifact_sha256: str,
) -> Mapping[str, Any]:
    """Compile and atomically install one semantic long-horizon Q/K/V/O fold."""

    torch = _torch()
    cache = _digest(cache_key, "cache key")
    bridge = _digest(bridge_artifact_sha256, "bridge artifact SHA-256")
    raw = tuple(evidence)
    if not raw:
        raise Lfm2SemanticAttentionRefoldError(
            "Semantic attention refolding requires at least one evidence item."
        )
    normalized = tuple(
        _normalized_case_source(
            item,
            layer_index=layer_index,
            cache_key=cache,
            bridge_artifact_sha256=bridge,
        )
        for item in raw
    )
    current_child_state = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(
        item.replay.captured_child_model_state_sha256 != current_child_state
        for item in normalized
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Actual-child replay was captured from a different frozen child state."
        )
    inventory, groups, weights, lineage = _source_contract(raw[0])
    if any(
        _source_contract(item) != (inventory, groups, weights, lineage)
        for item in raw[1:]
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Layer evidence disagrees on parent inventory or original head lineage."
        )
    attention = _attention_module(target_model, layer_index)
    head_dim = int(getattr(attention, "head_dim", 0))
    target_query_heads = int(attention.q_proj.weight.shape[0]) // head_dim
    source_query_heads = int(normalized[0].transport.probabilities.shape[1])
    if (
        head_dim <= 0
        or len(groups) != target_query_heads
        or sorted(member for group in groups for member in group)
        != list(range(source_query_heads))
        or any(
            len(group) != len(group_weights)
            for group, group_weights in zip(groups, weights, strict=True)
        )
        or any(
            not math.isclose(math.fsum(group_weights), 1.0, rel_tol=1e-8, abs_tol=1e-10)
            or any(value <= 0.0 or not math.isfinite(value) for value in group_weights)
            for group_weights in weights
        )
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Original query-head barycentric lineage does not cover this topology."
        )
    partitions = _split_sources(normalized)
    balanced = {
        name: _balanced_endpoint_cases(
            partitions[name],
            layer_index=layer_index,
            cache_key=cache,
            bridge_artifact_sha256=bridge,
        )
        for name in _PARTITIONS
    }
    cases = {name: balanced[name][0] for name in _PARTITIONS}
    balance_receipts = {name: balanced[name][1] for name in _PARTITIONS}
    split_sha = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-semantic-attention-four-way-split-v1",
            "cache_key": cache,
            "bridge_artifact_sha256": bridge,
            "layer_index": layer_index,
            "partitions": {
                name: [item.batch_index for item in partitions[name]]
                for name in _PARTITIONS
            },
        }
    )
    source_artifact = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-semantic-attention-source-v1",
            "cache_key": cache,
            "bridge_artifact_sha256": bridge,
            "layer_index": layer_index,
            "source_inventory_sha256": inventory.inventory_sha256,
            "compiler_lineage_sha256": lineage,
        }
    )
    seed = _seed(
        target_model,
        layer_index=layer_index,
        source_inventory=inventory,
        source_artifact_sha256=source_artifact,
    )
    fit_bank = seal_lfm2_endpoint_qk_fit_bank(
        cases["fit"],
        source_split_sha256=split_sha,
        source_artifact_sha256=source_artifact,
    )
    fit_check_bank = seal_lfm2_endpoint_qk_fit_check_bank(
        cases["fit_check"],
        source_split_sha256=split_sha,
        source_artifact_sha256=source_artifact,
    )
    selection_bank = seal_lfm2_endpoint_qk_fit_bank(
        cases["selection"],
        source_split_sha256=split_sha,
        source_artifact_sha256=source_artifact,
    )
    validation_bank = seal_lfm2_endpoint_qk_fit_bank(
        cases["validation"],
        source_split_sha256=split_sha,
        source_artifact_sha256=source_artifact,
    )
    policy = _execution_policy(len(normalized))
    compiled = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        policy=policy,
    )
    if (
        not isinstance(compiled, Lfm2CompiledEndpointQkResidualFold)
        or compiled.verify() != compiled.report["report_sha256"]
        or not compiled.accepted
        or not compiled.changed
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "FIT/FIT-CHECK produced no authenticated nonempty correction."
        )
    baseline_selection = _evaluate_bank(
        selection_bank,
        seed,
        query_weight=seed.query_weight,
        key_weight=seed.key_weight,
        policy=policy,
    )
    candidate_selection = _evaluate_bank(
        selection_bank,
        seed,
        query_weight=compiled.query_weight,
        key_weight=compiled.key_weight,
        policy=policy,
    )
    tolerance = policy.metric_regression_tolerance
    if not _metrics_nonregressing(
        candidate_selection,
        baseline_selection,
        tolerance=tolerance,
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Frozen candidate regressed on isolated SELECTION evidence."
        )
    frozen_query = compiled.query_weight.detach().cpu().clone()
    frozen_key = compiled.key_weight.detach().cpu().clone()
    frozen_qk_sha = canonical_sha256(
        {
            "q": _tensor_sha256(frozen_query),
            "k": _tensor_sha256(frozen_key),
        }
    )
    baseline_validation = _evaluate_bank(
        validation_bank,
        seed,
        query_weight=seed.query_weight,
        key_weight=seed.key_weight,
        policy=policy,
    )
    candidate_validation = _evaluate_bank(
        validation_bank,
        seed,
        query_weight=frozen_query,
        key_weight=frozen_key,
        policy=policy,
    )
    if (
        not _metrics_nonregressing(
            candidate_validation,
            baseline_validation,
            tolerance=tolerance,
        )
        or _tensor_sha256(frozen_query) != _tensor_sha256(compiled.query_weight)
        or _tensor_sha256(frozen_key) != _tensor_sha256(compiled.key_weight)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Frozen candidate failed VALIDATION or changed after selection freeze."
        )
    writer_banks = {
        "fit": _writer_cases(
            fit_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
        "fit_check": _writer_cases(
            fit_check_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
        "selection": _writer_cases(
            selection_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
        "validation": _writer_cases(
            validation_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
    }
    writer_proposal = _compile_output_writer(
        writer_banks["fit"],
        seed.output_weight,
        solver_device=policy.solver_device,
    )
    published_output, writer_gate = _gate_output_writer(
        writer_proposal,
        fit_check_cases=writer_banks["fit_check"],
        selection_cases=writer_banks["selection"],
        validation_cases=writer_banks["validation"],
        incumbent_output_weight=seed.output_weight,
        tolerance=tolerance,
    )
    writer_accepted = bool(writer_gate["accepted"])
    if writer_accepted != (not torch.equal(published_output, seed.output_weight)):
        raise Lfm2SemanticAttentionRefoldError(
            "Output-writer gate acceptance disagrees with its published state."
        )
    frozen_qkvo_sha = canonical_sha256(
        {
            "q": _tensor_sha256(frozen_query),
            "k": _tensor_sha256(frozen_key),
            "v": _tensor_sha256(seed.value_weight),
            "o": _tensor_sha256(published_output),
        }
    )
    state = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": seed.value_weight,
        "out_proj.weight": published_output,
    }
    before, after, changed, parameter_count = _install_qkvo_transaction(
        target_model,
        layer_index=layer_index,
        state=state,
    )
    base = _installation_receipt_body(
        layer_index=layer_index,
        evidence_sha256=[item.evidence_sha256 for item in raw],
        model_state_before_sha256=before,
        model_state_after_sha256=after,
        changed_state_keys=changed,
        parameter_element_count=parameter_count,
    )
    body = {
        **base,
        "analytic_refold": {
            "schema": LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA,
            "cache_key": cache,
            "bridge_artifact_sha256": bridge,
            "source_inventory_sha256": inventory.inventory_sha256,
            "compiler_lineage_sha256": lineage,
            "split_sha256": split_sha,
            "partition_batch_indices": {
                name: sorted({item.batch_index for item in partitions[name]})
                for name in _PARTITIONS
            },
            "trajectory_row_weight_balance": {
                name: dict(balance_receipts[name]) for name in _PARTITIONS
            },
            "fit_evidence_sha256": fit_bank.evidence_sha256,
            "fit_check_evidence_sha256": fit_check_bank.evidence_sha256,
            "selection_evidence_sha256": selection_bank.evidence_sha256,
            "validation_evidence_sha256": validation_bank.evidence_sha256,
            "endpoint_compiler_report_sha256": compiled.report["report_sha256"],
            "selection_baseline": baseline_selection.as_dict(),
            "selection_candidate": candidate_selection.as_dict(),
            "validation_baseline": baseline_validation.as_dict(),
            "validation_candidate": candidate_validation.as_dict(),
            "frozen_qk_sha256": frozen_qk_sha,
            "frozen_qkvo_sha256": frozen_qkvo_sha,
            "candidate_frozen_before_validation": True,
            "qk_candidate_frozen_before_validation": True,
            "output_writer_candidate_frozen_before_validation": (
                writer_proposal.changed
            ),
            "validation_triggered_refit": False,
            "numeric_parent_child_weight_match_objective": False,
            "dense_existing_qkvo_topology_only": True,
            "analytically_refit_projection_roles": [
                "q_proj",
                "k_proj",
                *(["out_proj"] if writer_accepted else []),
            ],
            "fixed_inherited_projection_roles": [
                "v_proj",
                *([] if writer_accepted else ["out_proj"]),
            ],
            "fixed_value_source": (
                "authenticated_live_c61_dense_seed_already_folded_on_broad_cache"
            ),
            "output_writer_source": (
                "fit_only_closed_form_weighted_residual_ridge"
                if writer_accepted
                else "authenticated_live_c61_seed_after_explicit_analytic_rejection"
            ),
            "output_writer": dict(writer_gate),
            "output_writer_cached_qk_probability_replay_calls": sum(
                len(bank) for bank in writer_banks.values()
            ),
            "output_writer_model_forward_calls": 0,
            "output_writer_fit_partition_only_used_for_solve": True,
            "output_writer_heldout_partitions_never_used_for_solve": True,
            "fresh_joint_value_output_refit_performed": False,
            "fresh_value_refit_performed": False,
            "fresh_output_writer_refit_performed": writer_accepted,
            "fresh_four_matrix_qkvo_refit_claimed": False,
            "complete_qkvo_bundle_installed_transactionally": True,
            "complete_causal_cache_only": True,
            "contractions_supported": False,
            "query_supervision_mode": (
                "one_parent_exact_terminal_endpoint_per_semantic_phase"
            ),
            "inserted_target_query_rows_parent_exact": False,
            "inserted_target_query_rows_used_for_attention_fit": False,
            "inserted_target_span_completion_requires_lexical_terminal_bank": True,
            "refold_operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        },
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


__all__ = [
    "LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_REPLAY_ENDPOINT_SCHEMA",
    "Lfm2SemanticAttentionRefoldError",
    "Lfm2SemanticAttentionReplayEndpoint",
    "compile_install_lfm2_semantic_attention_postfold",
]
