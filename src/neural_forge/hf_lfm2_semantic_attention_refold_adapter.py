"""Closed-form semantic-span attention refolding for a saved dense LFM2 child.

The semantic lattice is a teacher, not a weight-matching target.  This module
turns its probability/context evidence and an authenticated actual-child
macrocell replay into endpoint cases for the existing matrix-free Q/K
compiler, then competes an analytic fixed-V output writer with a joint V/O
residual factorization.  The latter solves in endpoint-row space and projects
each GQA group's composite maps back through its exact shared-V,
head-specific-O rank.  FIT, FIT-CHECK, SELECTION, and VALIDATION remain
disjoint.  Candidates are frozen before VALIDATION and dense Q/K/V/O is copied
into the existing child topology only after every applicable held-out gate
passes.

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


def _topology_sha256(model: Any) -> str:
    aliases: dict[int, int] = {}
    records = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        alias = aliases.setdefault(id(parameter), len(aliases))
        records.append(
            {
                "name": name,
                "alias_group": alias,
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "layout": str(parameter.layout).removeprefix("torch."),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    if not records:
        raise Lfm2SemanticAttentionRefoldError("Target model has no Parameters.")
    return canonical_sha256(records)


def _installation_receipt_body(
    *,
    layer_index: int,
    evidence_sha256: Sequence[str],
    model_state_before_sha256: str,
    model_state_after_sha256: str,
    changed_state_keys: Sequence[str],
    parameter_element_count: int,
    target_topology_sha256_before: str,
    target_topology_sha256_after: str,
) -> dict[str, Any]:
    evidence = tuple(
        _digest(value, "transport evidence SHA-256") for value in evidence_sha256
    )
    changed = tuple(changed_state_keys)
    before = _digest(model_state_before_sha256, "model state before")
    after = _digest(model_state_after_sha256, "model state after")
    topology_before = _digest(
        target_topology_sha256_before,
        "target topology before",
    )
    topology_after = _digest(
        target_topology_sha256_after,
        "target topology after",
    )
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or not evidence
        or any(not isinstance(value, str) or not value for value in changed)
        or isinstance(parameter_element_count, bool)
        or not isinstance(parameter_element_count, int)
        or parameter_element_count <= 0
        or bool(changed) != (before != after)
        or topology_before != topology_after
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Attention installation receipt inputs are incomplete."
        )
    installed = bool(changed)
    return {
        "schema": _INSTALLATION_SCHEMA,
        "status": (
            "installed-frozen-semantic-attention-candidate"
            if installed
            else "heldout-selected-incumbent-noop"
        ),
        "layer_index": layer_index,
        "installed": installed,
        "install_count": int(installed),
        "transport_evidence_sha256": list(evidence),
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "changed_state_keys": list(changed),
        "parameter_element_count_before": parameter_element_count,
        "parameter_element_count_after": parameter_element_count,
        "parameter_count_changed": False,
        "target_topology_sha256_before": topology_before,
        "target_topology_sha256_after": topology_after,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "exact_noop_state_proven": not installed,
        "incumbent_selected_by_heldout_gates": not installed,
        "fit_check_nonregression_passed": True,
        "selection_nonregression_passed": True,
        "validation_nonregression_passed": True,
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
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


def _exact_zero(value: Any) -> bool:
    return int(_torch().count_nonzero(value.detach()).item()) == 0


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _cold_projection_preimage(
    runtime_rows: Any,
    anchor_rows: Any,
    *,
    norm_weight: Any,
    norm_epsilon: float,
    cosine: Any,
    sine: Any,
    role: str,
) -> Any:
    """Recover deterministic raw projection rows for one sealed route plant."""

    torch = _torch()
    runtime = runtime_rows.detach().to(device="cpu", dtype=torch.float64)
    anchors = anchor_rows.detach().to(device="cpu", dtype=torch.float64)
    gain = norm_weight.detach().to(device="cpu", dtype=torch.float64)
    cos = cosine.detach().to(device="cpu", dtype=torch.float64)
    sin = sine.detach().to(device="cpu", dtype=torch.float64)
    if (
        runtime.ndim != 4
        or tuple(runtime.shape) != tuple(anchors.shape)
        or tuple(gain.shape) != (int(runtime.shape[-1]),)
        or tuple(cos.shape)
        != (int(runtime.shape[0]), int(runtime.shape[2]), int(runtime.shape[3]))
        or tuple(sin.shape) != tuple(cos.shape)
        or bool((gain.abs() <= torch.finfo(torch.float64).eps).any().item())
    ):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} rows cannot pass the target RMSNorm/RoPE geometry."
        )
    radius = cos.square() + sin.square()
    if bool((radius <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} evidence contains a zero-radius RoPE row."
        )
    unrotated = (
        runtime * cos.unsqueeze(1) - _rotate_half(runtime) * sin.unsqueeze(1)
    ) / radius.unsqueeze(1)
    direction = unrotated / gain.reshape(1, 1, 1, -1)
    direction_rms = torch.sqrt(direction.square().mean(dim=-1, keepdim=True))
    if bool((direction_rms <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} route target has an empty normalized direction."
        )
    anchor_rms = torch.sqrt(anchors.square().mean(dim=-1, keepdim=True)).clamp_min(
        math.sqrt(norm_epsilon)
    )
    result = direction / direction_rms * anchor_rms
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} preimage produced nonfinite rows."
        )
    return result.contiguous()


def _cold_route_runtime_targets(
    case: Lfm2EndpointQkResidualCase,
    seed: Lfm2EndpointQkvoSeed,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Factor sealed route logits into target-native post-RoPE Q/K rows."""

    torch = _torch()
    probabilities = case.target_probability_barycenter.to(torch.float64)
    mask = case.target_causal_mask
    batch, query_heads, query_length, key_length = map(int, probabilities.shape)
    head_dim = seed.head_dim
    key_value_heads = seed.key_value_heads
    queries_per_key_value = seed.queries_per_key_value
    if (
        query_length != 1
        or query_heads != seed.query_heads
        or query_heads != key_value_heads * queries_per_key_value
        or queries_per_key_value > head_dim
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Cold-start Q/K planting requires one terminal query and an integral "
            "target GQA head geometry."
        )
    expanded_mask = mask
    safe = torch.where(
        expanded_mask,
        probabilities.clamp_min(torch.finfo(torch.float64).tiny),
        torch.ones_like(probabilities),
    )
    logits = torch.where(expanded_mask, torch.log(safe), torch.zeros_like(safe))
    counts = expanded_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = torch.where(
        expanded_mask,
        logits - (logits * expanded_mask).sum(dim=-1, keepdim=True) / counts,
        torch.zeros_like(logits),
    )
    query_runtime = torch.zeros(
        (batch, query_heads, 1, head_dim),
        dtype=torch.float64,
    )
    key_runtime = torch.zeros(
        (batch, key_value_heads, key_length, head_dim),
        dtype=torch.float64,
    )
    root_head_dim = math.sqrt(head_dim)
    clipped_vectors = 0
    uniform_vectors = 0
    for group_index in range(key_value_heads):
        first_head = group_index * queries_per_key_value
        heads = tuple(range(first_head, first_head + queries_per_key_value))
        for local_index, head_index in enumerate(heads):
            query_runtime[:, head_index, 0, local_index] = root_head_dim
        coordinates = (
            logits[:, heads, 0, :].permute(0, 2, 1).contiguous().clone()
        )
        energy = coordinates.square().sum(dim=-1, keepdim=True)
        budget = float(head_dim) * 0.75
        scale = torch.sqrt(
            torch.full_like(energy, budget)
            / energy.clamp_min(torch.finfo(torch.float64).tiny)
        ).clamp_max(1.0)
        clipped_vectors += int((scale < 1.0).sum().item())
        coordinates.mul_(scale)
        energy = coordinates.square().sum(dim=-1, keepdim=True)
        key_runtime[:, group_index, :, :queries_per_key_value] = coordinates
        if queries_per_key_value < head_dim:
            key_runtime[
                :, group_index, :, queries_per_key_value
            ] = torch.sqrt((float(head_dim) - energy[..., 0]).clamp_min(0.0))
        else:
            empty = energy[..., 0] <= torch.finfo(torch.float64).tiny
            uniform_vectors += int(empty.sum().item())
            if bool(empty.any().item()):
                key_runtime[:, group_index, :, -1] = torch.where(
                    empty,
                    torch.full_like(energy[..., 0], root_head_dim),
                    key_runtime[:, group_index, :, -1],
                )
    body = {
        "schema": "neural-forge-lfm2-semantic-qk-cold-runtime-target-v1",
        "case_evidence_sha256": case.evidence_sha256,
        "target_probability_sha256": _tensor_sha256(probabilities),
        "target_mask_sha256": _tensor_sha256(mask),
        "query_runtime_sha256": _tensor_sha256(query_runtime),
        "key_runtime_sha256": _tensor_sha256(key_runtime),
        "centered_log_probability_factorization": True,
        "post_rope_score_scale": "query_coordinate_equals_sqrt_head_dim",
        "gqa_shared_key_coordinates": True,
        "clipped_key_route_vectors": clipped_vectors,
        "uniform_key_route_vectors": uniform_vectors,
    }
    return query_runtime, key_runtime, MappingProxyType(
        {**body, "target_sha256": canonical_sha256(body)}
    )


def _cold_ridge_projection(
    inputs: Any,
    targets: Any,
    sqrt_weights: Any,
    *,
    role: str,
    ridge: float = 1e-4,
) -> tuple[Any, Mapping[str, Any]]:
    """Fit one deterministic dense BF16 plant without gradients or randomness."""

    torch = _torch()
    x = inputs.detach().to(device="cpu", dtype=torch.float64)
    y = targets.detach().to(device="cpu", dtype=torch.float64)
    root = sqrt_weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 1)
    if (
        x.ndim != 2
        or y.ndim != 2
        or int(x.shape[0]) != int(y.shape[0])
        or tuple(root.shape) != (int(x.shape[0]), 1)
        or bool((root <= 0.0).any().item())
    ):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} ridge rows are incompatible."
        )
    weighted_x = x * root
    weighted_y = y * root
    rows, features = map(int, weighted_x.shape)
    try:
        with torch.inference_mode():
            if rows <= features:
                gram = weighted_x @ weighted_x.T
                scale = float(gram.diagonal().mean().clamp_min(1.0).item())
                effective_ridge = float(ridge) * scale
                dual = torch.linalg.solve(
                    gram
                    + effective_ridge
                    * torch.eye(rows, dtype=torch.float64),
                    weighted_y,
                )
                weight = dual.T @ weighted_x
                solve_form = "deterministic_dual_ridge"
            else:
                gram = weighted_x.T @ weighted_x
                scale = float(gram.diagonal().mean().clamp_min(1.0).item())
                effective_ridge = float(ridge) * scale
                weight = torch.linalg.solve(
                    gram
                    + effective_ridge
                    * torch.eye(features, dtype=torch.float64),
                    weighted_x.T @ weighted_y,
                ).T
                solve_form = "deterministic_primal_ridge"
    except RuntimeError as exc:
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} ridge solve failed."
        ) from exc
    if not bool(torch.isfinite(weight).all().item()) or _exact_zero(weight):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} ridge solve produced no finite plant."
        )
    target_rms = 1.0 / math.sqrt(features)
    observed_rms = torch.sqrt(weight.square().mean())
    weight = weight * (target_rms / observed_rms)
    published = weight.to(torch.bfloat16).contiguous()
    if _exact_zero(published) or not bool(torch.isfinite(published).all().item()):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} plant vanished during BF16 publication."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-qk-cold-ridge-v1",
        "role": role,
        "solve_form": solve_form,
        "row_count": rows,
        "input_width": features,
        "output_width": int(weight.shape[0]),
        "ridge": float(ridge),
        "effective_ridge": effective_ridge,
        "input_sha256": _tensor_sha256(x),
        "target_sha256": _tensor_sha256(y),
        "sqrt_weight_sha256": _tensor_sha256(root),
        "published_weight_sha256": _tensor_sha256(published),
        "published_nonzero_elements": int(torch.count_nonzero(published).item()),
        "normalized_published_element_rms": target_rms,
    }
    return published, MappingProxyType(
        {**body, "ridge_sha256": canonical_sha256(body)}
    )


def _compile_qk_cold_start_plant(
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Compile a FIT-only route plant for an exact-zero Q/K incumbent."""

    torch = _torch()
    cases = tuple(fit_bank.cases)
    if fit_bank.partition != "fit" or not cases:
        raise Lfm2SemanticAttentionRefoldError(
            "Q/K cold-start planting requires one authenticated FIT bank."
        )
    query_inputs = []
    query_targets = []
    query_weights = []
    key_inputs = []
    key_targets = []
    key_weights = []
    target_receipts = []
    query_weight64 = seed.query_weight.to(torch.float64)
    key_weight64 = seed.key_weight.to(torch.float64)
    for case in cases:
        query_runtime, key_runtime, target_receipt = _cold_route_runtime_targets(
            case,
            seed,
        )
        batch, query_length, hidden = map(int, case.target_query_inputs.shape)
        key_length = int(case.target_key_value_inputs.shape[1])
        query_anchor = (
            case.target_query_inputs.to(torch.float64) @ query_weight64.T
        ).reshape(batch, query_length, seed.query_heads, seed.head_dim)
        query_anchor = query_anchor.permute(0, 2, 1, 3).contiguous()
        key_anchor = (
            case.target_key_value_inputs.to(torch.float64) @ key_weight64.T
        ).reshape(batch, key_length, seed.key_value_heads, seed.head_dim)
        key_anchor = key_anchor.permute(0, 2, 1, 3).contiguous()
        query_raw = _cold_projection_preimage(
            query_runtime,
            query_anchor,
            norm_weight=seed.query_norm_weight,
            norm_epsilon=seed.norm_epsilon,
            cosine=case.query_rope_cos,
            sine=case.query_rope_sin,
            role="query",
        )
        key_raw = _cold_projection_preimage(
            key_runtime,
            key_anchor,
            norm_weight=seed.key_norm_weight,
            norm_epsilon=seed.norm_epsilon,
            cosine=case.key_rope_cos,
            sine=case.key_rope_sin,
            role="key",
        )
        query_inputs.append(case.target_query_inputs.reshape(-1, hidden))
        query_targets.append(
            query_raw.permute(0, 2, 1, 3).reshape(-1, seed.query_heads * seed.head_dim)
        )
        query_weights.append(case.query_row_sqrt_weights.reshape(-1))
        key_inputs.append(case.target_key_value_inputs.reshape(-1, hidden))
        key_targets.append(
            key_raw.permute(0, 2, 1, 3).reshape(
                -1,
                seed.key_value_heads * seed.head_dim,
            )
        )
        case_scale = torch.sqrt(
            case.query_row_sqrt_weights.to(torch.float64).square().mean()
        )
        key_weights.append(
            torch.full((batch * key_length,), float(case_scale.item()))
        )
        target_receipts.append(dict(target_receipt))
    planted_query, query_receipt = _cold_ridge_projection(
        torch.cat(query_inputs, dim=0),
        torch.cat(query_targets, dim=0),
        torch.cat(query_weights, dim=0),
        role="query",
    )
    planted_key, key_receipt = _cold_ridge_projection(
        torch.cat(key_inputs, dim=0),
        torch.cat(key_targets, dim=0),
        torch.cat(key_weights, dim=0),
        role="key",
    )
    if (
        tuple(planted_query.shape) != tuple(seed.query_weight.shape)
        or tuple(planted_key.shape) != tuple(seed.key_weight.shape)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Q/K cold-start plant changed the dense target topology."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-qk-cold-start-plant-v1",
        "fit_evidence_sha256": fit_bank.evidence_sha256,
        "source_artifact_sha256": fit_bank.source_artifact_sha256,
        "seed_sha256": seed.seed_sha256,
        "case_evidence_sha256": [case.evidence_sha256 for case in cases],
        "runtime_targets": target_receipts,
        "query_ridge": dict(query_receipt),
        "key_ridge": dict(key_receipt),
        "planted_query_sha256": _tensor_sha256(planted_query),
        "planted_key_sha256": _tensor_sha256(planted_key),
        "fit_partition_only_used_for_plant": True,
        "heldout_partitions_used_for_plant": False,
        "randomness_used": False,
        "gradients_used": False,
        "training_steps": 0,
    }
    return planted_query, planted_key, MappingProxyType(
        {**body, "plant_sha256": canonical_sha256(body)}
    )


def _strict_metric_improvement(
    candidate: Lfm2EndpointQkMetrics,
    baseline: Lfm2EndpointQkMetrics,
) -> bool:
    tolerance = 1e-12
    return bool(
        candidate.endpoint_relative_rmse
        < baseline.endpoint_relative_rmse - tolerance
        or candidate.probability_relative_rmse
        < baseline.probability_relative_rmse - tolerance
        or candidate.js_divergence_mean < baseline.js_divergence_mean - tolerance
        or candidate.js_divergence_worst < baseline.js_divergence_worst - tolerance
        or candidate.top1_route_agreement_fraction
        > baseline.top1_route_agreement_fraction + tolerance
    )


def _cold_start_qk_seed(
    seed: Lfm2EndpointQkvoSeed,
    *,
    banks: Mapping[str, Lfm2EndpointQkResidualEvidenceBank],
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[Lfm2EndpointQkvoSeed, Mapping[str, Any] | None]:
    """Plant and four-way gate Q/K only when the incumbent is exactly zero."""

    query_was_zero = _exact_zero(seed.query_weight)
    key_was_zero = _exact_zero(seed.key_weight)
    if not query_was_zero and not key_was_zero:
        return seed, None
    planted_query, planted_key, plant = _compile_qk_cold_start_plant(
        seed,
        banks["fit"],
    )
    planted = Lfm2EndpointQkvoSeed(
        query_weight=planted_query,
        key_weight=planted_key,
        value_weight=seed.value_weight,
        output_weight=seed.output_weight,
        query_norm_weight=seed.query_norm_weight,
        key_norm_weight=seed.key_norm_weight,
        norm_epsilon=seed.norm_epsilon,
        source_inventory=seed.source_inventory,
        source_artifact_sha256=seed.source_artifact_sha256,
        target_template_spec_sha256=seed.target_template_spec_sha256,
    )
    metrics: dict[str, Any] = {}
    rejected = None
    for partition in _PARTITIONS:
        baseline = _evaluate_bank(
            banks[partition],
            seed,
            query_weight=seed.query_weight,
            key_weight=seed.key_weight,
            policy=policy,
        )
        candidate = _evaluate_bank(
            banks[partition],
            planted,
            query_weight=planted.query_weight,
            key_weight=planted.key_weight,
            policy=policy,
        )
        nonregressing = _metrics_nonregressing(
            candidate,
            baseline,
            tolerance=policy.metric_regression_tolerance,
        )
        metrics[partition] = {
            "baseline": baseline.as_dict(),
            "candidate": candidate.as_dict(),
            "nonregressing": nonregressing,
        }
        if not nonregressing and rejected is None:
            rejected = partition
    if rejected is not None or not _strict_metric_improvement(
        Lfm2EndpointQkMetrics(**metrics["fit"]["candidate"]),
        Lfm2EndpointQkMetrics(**metrics["fit"]["baseline"]),
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Exact-zero Q/K cold-start plant did not pass frozen four-way "
            f"nonregression with strict FIT improvement (rejected={rejected!r})."
        )
    if (query_was_zero and _exact_zero(planted.query_weight)) or (
        key_was_zero and _exact_zero(planted.key_weight)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Authenticated Q/K cold-start left an originally zero role at zero."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-qk-cold-start-gate-v1",
        "original_seed_sha256": seed.seed_sha256,
        "planted_seed_sha256": planted.seed_sha256,
        "original_query_exact_zero": query_was_zero,
        "original_key_exact_zero": key_was_zero,
        "plant": dict(plant),
        "partitions": metrics,
        "fit_strictly_improved": True,
        "candidate_frozen_before_fit_check_selection_validation": True,
        "heldout_refit_or_reselection": False,
        "zero_incumbent_noop_allowed": False,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return planted, MappingProxyType(
        {**body, "gate_sha256": canonical_sha256(body)}
    )


def _execution_policy(case_count: int) -> Lfm2EndpointQkResidualFoldPolicy:
    torch = _torch()
    devices = ()
    if torch.cuda.is_available():
        native = []
        for index in range(torch.cuda.device_count()):
            try:
                major, _minor = torch.cuda.get_device_capability(index)
            except (AssertionError, RuntimeError):
                continue
            # PyTorch's is_bf16_supported positional argument is an emulation
            # flag, not a device index.  Capability is queried explicitly so a
            # heterogeneous host cannot inherit cuda:0's answer for P40/Turing
            # cards.  Native CUDA BF16 tensor-core execution begins at SM80.
            if int(major) >= 8:
                native.append(f"cuda:{index}")
        devices = tuple(native)
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


def _bfloat16_endpoint_equivalence(actual: Any, expected: Any) -> Mapping[str, Any]:
    """Seal exact same-device replay identity for the inherited V/O seed."""

    torch = _torch()
    observed = _finite_cpu_tensor(
        actual,
        "observed seed endpoint",
        ndim=3,
    ).to(torch.float64)
    reference = _finite_cpu_tensor(
        expected,
        "authenticated seed endpoint",
        ndim=3,
    ).to(torch.float64)
    if tuple(observed.shape) != tuple(reference.shape):
        raise Lfm2SemanticAttentionRefoldError(
            "Seed replay endpoints do not have the same geometry."
        )
    difference = observed - reference
    rmse = float(torch.sqrt(difference.square().mean()).item())
    reference_rms = float(torch.sqrt(reference.square().mean()).item())
    maximum = float(difference.abs().max().item())
    reference_maximum = float(reference.abs().max().item())
    observed_rows = observed.reshape(-1, int(observed.shape[-1]))
    reference_rows = reference.reshape(-1, int(reference.shape[-1]))
    observed_norm = torch.linalg.vector_norm(observed_rows, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference_rows, dim=-1)
    denominator = observed_norm * reference_norm
    both_zero = (observed_norm == 0.0) & (reference_norm == 0.0)
    cosine = torch.where(
        both_zero,
        torch.ones_like(denominator),
        torch.where(
            denominator > torch.finfo(torch.float64).tiny,
            (observed_rows * reference_rows).sum(dim=-1)
            / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(denominator),
        ),
    ).clamp(-1.0, 1.0)
    mean_cosine = float(cosine.mean().item())
    worst_cosine = float(cosine.min().item())
    normalized_rmse = rmse / max(reference_rms, torch.finfo(torch.float64).tiny)
    normalized_maximum = maximum / max(
        reference_maximum,
        torch.finfo(torch.float64).tiny,
    )
    bitwise_equal = torch.equal(observed, reference)
    body = {
        "schema": "neural-forge-lfm2-semantic-same-device-endpoint-identity-v1",
        "actual_sha256": _tensor_sha256(observed),
        "expected_sha256": _tensor_sha256(reference),
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "maximum_absolute_error": maximum,
        "normalized_maximum_absolute_error": normalized_maximum,
        "mean_cosine_similarity": mean_cosine,
        "worst_cosine_similarity": worst_cosine,
        "maximum_normalized_rmse": 0.0,
        "maximum_normalized_absolute_error": 0.0,
        "minimum_mean_cosine_similarity": 1.0,
        "minimum_worst_cosine_similarity": 1.0,
        "equivalent": bitwise_equal,
        "same_execution_device": True,
        "same_operation_order": True,
        "bitwise_identity_required": True,
    }
    return MappingProxyType({**body, "equivalence_sha256": canonical_sha256(body)})


@dataclass(frozen=True, slots=True)
class _JointVoCase:
    """Probability-weighted pre-V features for parent-exact endpoint rows."""

    head_features: Any = field(repr=False, compare=False)
    probabilities: Any = field(repr=False, compare=False)
    key_value_inputs: Any = field(repr=False, compare=False)
    endpoint_target: Any = field(repr=False, compare=False)
    row_weights: Any = field(repr=False, compare=False)
    trajectory_sha256: str
    seed_replay_equivalence: Mapping[str, Any]
    case_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        features = _finite_cpu_tensor(
            self.head_features,
            "joint V/O probability-weighted input features",
            ndim=4,
            dtype=torch.float64,
        )
        probabilities = _finite_cpu_tensor(
            self.probabilities,
            "joint V/O frozen Q/K probabilities",
            ndim=4,
            dtype=torch.float64,
        )
        key_inputs = _finite_cpu_tensor(
            self.key_value_inputs,
            "joint V/O replay-fresh key/value inputs",
            ndim=3,
            dtype=torch.bfloat16,
        )
        endpoint = _finite_cpu_tensor(
            self.endpoint_target,
            "joint V/O endpoint target",
            ndim=3,
            dtype=torch.float64,
        )
        weights = _finite_cpu_tensor(
            self.row_weights,
            "joint V/O row weights",
            ndim=2,
            dtype=torch.float64,
        )
        if (
            tuple(features.shape[:2]) != tuple(endpoint.shape[:2])
            or tuple(probabilities.shape[:1]) != tuple(endpoint.shape[:1])
            or int(probabilities.shape[1]) != int(features.shape[2])
            or int(probabilities.shape[2]) != int(endpoint.shape[1])
            or tuple(probabilities.shape[0::3])
            != (int(key_inputs.shape[0]), int(key_inputs.shape[1]))
            or int(features.shape[3]) != int(key_inputs.shape[2])
            or tuple(weights.shape) != tuple(endpoint.shape[:2])
            or bool((weights <= 0.0).any().item())
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Joint V/O features, targets, and positive row weights are not aligned."
            )
        trajectory = _digest(
            self.trajectory_sha256,
            "joint V/O trajectory SHA-256",
        )
        equivalence = dict(self.seed_replay_equivalence)
        equivalence_sha = equivalence.pop("equivalence_sha256", None)
        if equivalence.get(
            "equivalent"
        ) is not True or equivalence_sha != canonical_sha256(equivalence):
            raise Lfm2SemanticAttentionRefoldError(
                "Joint V/O case lacks a sealed equivalent seed replay."
            )
        object.__setattr__(self, "head_features", features)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "key_value_inputs", key_inputs)
        object.__setattr__(self, "endpoint_target", endpoint)
        object.__setattr__(self, "row_weights", weights)
        object.__setattr__(self, "trajectory_sha256", trajectory)
        object.__setattr__(
            self,
            "seed_replay_equivalence",
            MappingProxyType({**equivalence, "equivalence_sha256": equivalence_sha}),
        )
        object.__setattr__(
            self,
            "case_sha256",
            canonical_sha256(
                {
                    "schema": "neural-forge-lfm2-semantic-joint-vo-case-v1",
                    "trajectory_sha256": trajectory,
                    "head_features_sha256": _tensor_sha256(features),
                    "probabilities_sha256": _tensor_sha256(probabilities),
                    "key_value_inputs_sha256": _tensor_sha256(key_inputs),
                    "endpoint_target_sha256": _tensor_sha256(endpoint),
                    "row_weights_sha256": _tensor_sha256(weights),
                    "seed_replay_equivalence_sha256": equivalence_sha,
                    "parent_exact_semantic_endpoint_rows_only": True,
                    "inserted_target_query_rows_parent_exact": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _JointVoProposal:
    value_weight: Any = field(repr=False, compare=False)
    output_weight: Any = field(repr=False, compare=False)
    changed: bool
    fit_improved: bool
    baseline_fit_metrics: _WriterMetrics
    candidate_fit_metrics: _WriterMetrics
    ridge: float
    effective_ridge: float
    dual_dimension: int
    primal_feature_dimension: int
    fit_evidence_sha256: str
    factorization_receipts: tuple[Mapping[str, Any], ...]
    proposal_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        value = _finite_cpu_tensor(
            self.value_weight,
            "joint V/O value proposal",
            ndim=2,
            dtype=torch.bfloat16,
        )
        output = _finite_cpu_tensor(
            self.output_weight,
            "joint V/O output proposal",
            ndim=2,
            dtype=torch.bfloat16,
        )
        receipts = tuple(
            MappingProxyType(dict(item)) for item in self.factorization_receipts
        )
        if (
            not math.isfinite(self.ridge)
            or self.ridge <= 0.0
            or not math.isfinite(self.effective_ridge)
            or self.effective_ridge <= 0.0
            or self.dual_dimension <= 0
            or self.primal_feature_dimension <= 0
            or not receipts
            or self.fit_improved != self.changed
            or (
                self.changed
                and self.candidate_fit_metrics.endpoint_relative_rmse
                >= self.baseline_fit_metrics.endpoint_relative_rmse
            )
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Joint V/O proposal metadata is inconsistent."
            )
        fit_sha = _digest(self.fit_evidence_sha256, "joint V/O FIT evidence SHA-256")
        object.__setattr__(self, "value_weight", value)
        object.__setattr__(self, "output_weight", output)
        object.__setattr__(self, "fit_evidence_sha256", fit_sha)
        object.__setattr__(self, "factorization_receipts", receipts)
        object.__setattr__(self, "proposal_sha256", canonical_sha256(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-semantic-joint-vo-proposal-v1",
            "value_weight_sha256": _tensor_sha256(self.value_weight),
            "output_weight_sha256": _tensor_sha256(self.output_weight),
            "changed": self.changed,
            "fit_improved": self.fit_improved,
            "baseline_fit_metrics": self.baseline_fit_metrics.as_dict(),
            "candidate_fit_metrics": self.candidate_fit_metrics.as_dict(),
            "ridge": self.ridge,
            "effective_ridge": self.effective_ridge,
            "dual_dimension": self.dual_dimension,
            "primal_feature_dimension": self.primal_feature_dimension,
            "giant_primal_system_formed": False,
            "giant_primal_system_inverted": False,
            "fit_evidence_sha256": self.fit_evidence_sha256,
            "factorization_receipts": [
                dict(item) for item in self.factorization_receipts
            ],
            "solver": (
                "closed_form_weighted_residual_dual_ridge_then_deterministic_"
                "shared_value_low_rank_factorization"
            ),
            "shared_value_per_key_value_group": True,
            "head_specific_output_blocks": True,
            "factorized_candidate_published_in_proposal": self.changed,
            "factorization_attempt_rejected_after_bfloat16_fit_replay": (
                not self.changed
            ),
            "fit_only": True,
            "numeric_parent_child_weight_match_objective": False,
            "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "proposal_sha256": self.proposal_sha256}


def _joint_vo_prediction(
    case: _JointVoCase,
    value_weight: Any,
    output_weight: Any,
    *,
    head_dim: int,
    device: str = "cpu",
) -> Any:
    """Replay materialized BF16 V followed by the frozen-probability O path."""

    torch = _torch()
    resolved = torch.device(device)
    value = value_weight.detach().to(device=resolved, dtype=torch.bfloat16)
    output = output_weight.detach().to(device=resolved, dtype=torch.float64)
    query_heads = int(case.probabilities.shape[1])
    if (
        head_dim <= 0
        or int(value.shape[0]) % head_dim
        or int(output.shape[1]) != query_heads * head_dim
        or int(value.shape[1]) != int(case.key_value_inputs.shape[2])
        or int(output.shape[0]) != int(case.endpoint_target.shape[2])
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O candidate does not match its exact dense GQA topology."
        )
    key_value_heads = int(value.shape[0]) // head_dim
    if query_heads % key_value_heads:
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O candidate has a nonintegral GQA repeat factor."
        )
    batch, key_length, _hidden = map(int, case.key_value_inputs.shape)
    query_length = int(case.endpoint_target.shape[1])
    with torch.inference_mode():
        key_inputs = case.key_value_inputs.to(device=resolved)
        value_heads = (
            (key_inputs @ value.T)
            .reshape(batch, key_length, key_value_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        repeated_value = value_heads.repeat_interleave(
            query_heads // key_value_heads,
            dim=1,
        ).to(torch.float64)
        context = (
            case.probabilities.to(device=resolved)
            .matmul(repeated_value)
            .permute(0, 2, 1, 3)
            .reshape(batch, query_length, query_heads * head_dim)
        )
        return (context @ output.T).detach().cpu().contiguous()


def _joint_vo_cases(
    bank: Lfm2EndpointQkResidualEvidenceBank,
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any,
    key_weight: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[_JointVoCase, ...]:
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
        key_inputs = case.target_key_value_inputs.detach().to(
            device="cpu",
            dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            head_features = torch.einsum(
                "bhqk,bkf->bqhf",
                probabilities.to(torch.float64),
                key_inputs.to(torch.float64),
            ).contiguous()
        common = {
            "head_features": head_features,
            "probabilities": probabilities.to(torch.float64),
            "key_value_inputs": key_inputs,
            "endpoint_target": case.parent_endpoint_target.to(torch.float64),
            "row_weights": case.query_row_sqrt_weights.to(torch.float64).square(),
            "trajectory_sha256": case.trajectory_sha256,
        }
        joint_case = _JointVoCase(
            **common,
            seed_replay_equivalence=_bfloat16_endpoint_equivalence(
                replayed_endpoint,
                replayed_endpoint,
            ),
        )
        reconstructed = _joint_vo_prediction(
            joint_case,
            seed.value_weight,
            seed.output_weight,
            head_dim=seed.head_dim,
            device=device,
        )
        equivalence = _bfloat16_endpoint_equivalence(
            reconstructed,
            replayed_endpoint,
        )
        if equivalence["equivalent"] is not True:
            raise Lfm2SemanticAttentionRefoldError(
                "Joint V/O seed replay exceeds the sealed BF16 numerical envelope."
            )
        result.append(
            _JointVoCase(
                **common,
                seed_replay_equivalence=equivalence,
            )
        )
    return tuple(result)


def _writer_cases_from_joint(
    joint_cases: Sequence[_JointVoCase],
    seed: Lfm2EndpointQkvoSeed,
    *,
    compute_devices: Sequence[str] = (),
) -> tuple[_WriterCase, ...]:
    torch = _torch()
    devices = tuple(compute_devices) or ("cpu",)
    result = []
    for case_index, case in enumerate(joint_cases):
        device = torch.device(devices[case_index % len(devices)])
        value = seed.value_weight.to(device=device, dtype=torch.bfloat16)
        key_inputs = case.key_value_inputs.to(device=device)
        batch, key_length, _hidden = map(int, case.key_value_inputs.shape)
        query_length = int(case.endpoint_target.shape[1])
        with torch.inference_mode():
            value_heads = (
                (key_inputs @ value.T)
                .reshape(
                    batch,
                    key_length,
                    seed.key_value_heads,
                    seed.head_dim,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            context = (
                case.probabilities.to(device=device)
                .matmul(
                    value_heads.repeat_interleave(
                        seed.queries_per_key_value,
                        dim=1,
                    ).to(torch.float64)
                )
                .permute(0, 2, 1, 3)
                .reshape(
                    batch,
                    query_length,
                    seed.query_heads * seed.head_dim,
                )
                .contiguous()
            )
        result.append(
            _WriterCase(
                context=context.detach().cpu(),
                endpoint_target=case.endpoint_target,
                row_weights=case.row_weights,
                trajectory_sha256=case.trajectory_sha256,
            )
        )
    return tuple(result)


def _writer_cases(
    bank: Lfm2EndpointQkResidualEvidenceBank,
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any,
    key_weight: Any,
    policy: Lfm2EndpointQkResidualFoldPolicy,
) -> tuple[_WriterCase, ...]:
    return _writer_cases_from_joint(
        _joint_vo_cases(
            bank,
            seed,
            query_weight=query_weight,
            key_weight=key_weight,
            policy=policy,
        ),
        seed,
    )


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


def _joint_vo_metrics(
    cases: Sequence[_JointVoCase],
    value_weight: Any,
    output_weight: Any,
    *,
    head_dim: int,
    compute_devices: Sequence[str] = (),
) -> _WriterMetrics:
    torch = _torch()
    selected = tuple(cases)
    if not selected:
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O metrics require at least one exact endpoint case."
        )
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_target = torch.zeros((), dtype=torch.float64)
    weighted_elements = torch.zeros((), dtype=torch.float64)
    weighted_cosine = torch.zeros((), dtype=torch.float64)
    cosine_weight = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    worst_cosine = 1.0
    rows = 0
    devices = tuple(compute_devices) or ("cpu",)
    for case_index, case in enumerate(selected):
        prediction = _joint_vo_prediction(
            case,
            value_weight,
            output_weight,
            head_dim=head_dim,
            device=devices[case_index % len(devices)],
        )
        difference = prediction - case.endpoint_target
        weights = case.row_weights.unsqueeze(-1)
        squared_error += (difference.square() * weights).sum()
        squared_target += (case.endpoint_target.square() * weights).sum()
        weighted_elements += weights.sum() * int(case.endpoint_target.shape[-1])
        maximum = max(maximum, float(difference.abs().max().item()))
        prediction_norm = torch.linalg.vector_norm(prediction, dim=-1)
        target_norm = torch.linalg.vector_norm(case.endpoint_target, dim=-1)
        denominator = prediction_norm * target_norm
        both_zero = (prediction_norm == 0.0) & (target_norm == 0.0)
        cosine = torch.where(
            both_zero,
            torch.ones_like(denominator),
            torch.where(
                denominator > torch.finfo(torch.float64).tiny,
                (prediction * case.endpoint_target).sum(dim=-1)
                / denominator.clamp_min(torch.finfo(torch.float64).tiny),
                torch.zeros_like(denominator),
            ),
        ).clamp(-1.0, 1.0)
        weighted_cosine += (cosine * case.row_weights).sum()
        cosine_weight += case.row_weights.sum()
        worst_cosine = min(worst_cosine, float(cosine.min().item()))
        rows += int(case.row_weights.numel())
    element_mass = weighted_elements.clamp_min(torch.finfo(torch.float64).tiny)
    rmse = torch.sqrt(squared_error / element_mass)
    target_scale = torch.sqrt(squared_target / element_mass).clamp_min(
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


def _factor_group_composites(
    composites: Sequence[Any],
    *,
    rowspace_generators: Any,
    target_rank: int,
    group_index: int,
) -> tuple[Any, tuple[Any, ...], Mapping[str, Any]]:
    """Best deterministic shared-rowspace rank factor in a reduced basis."""

    torch = _torch()
    selected = tuple(
        item.detach().to(device="cpu", dtype=torch.float64).contiguous()
        for item in composites
    )
    generators = rowspace_generators.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    if (
        not selected
        or target_rank <= 0
        or generators.ndim != 2
        or generators.numel() == 0
        or any(
            item.ndim != 2 or tuple(item.shape) != tuple(selected[0].shape)
            for item in selected
        )
        or int(generators.shape[1]) != int(selected[0].shape[1])
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O group composites do not define one factorable topology."
        )
    stacked = torch.cat(selected, dim=0)
    with torch.inference_mode():
        basis_columns, _triangular = torch.linalg.qr(
            generators.T,
            mode="reduced",
        )
        reduced = stacked @ basis_columns
        reconstructed = reduced @ basis_columns.T
        rowspace_error = float((reconstructed - stacked).abs().max().item())
        rowspace_scale = float(stacked.abs().max().clamp_min(1.0).item())
        if rowspace_error > 1e-9 * rowspace_scale:
            raise Lfm2SemanticAttentionRefoldError(
                "Reduced joint V/O row space does not contain its composite target."
            )
        left, singular_values, reduced_right = torch.linalg.svd(
            reduced,
            full_matrices=False,
        )
        right = reduced_right @ basis_columns.T
    available_rank = int(singular_values.numel())
    retained_rank = min(target_rank, available_rank)
    if retained_rank <= 0:
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O factorization found no available singular direction."
        )
    numerical_threshold = (
        torch.finfo(torch.float64).eps
        * max(stacked.shape)
        * singular_values[0].abs().clamp_min(torch.finfo(torch.float64).tiny)
    )
    numerical_rank = int((singular_values > numerical_threshold).sum().item())
    retained_effective_rank = min(retained_rank, numerical_rank)
    left_rank = left[:, :retained_rank].clone()
    right_rank = right[:retained_rank].clone()
    # SVD signs are arbitrary.  Canonicalize each pair by the largest-magnitude
    # coordinate of the right factor so receipts and BF16 materialization remain
    # deterministic for one numeric backend.
    for index in range(retained_rank):
        pivot = int(torch.argmax(right_rank[index].abs()).item())
        if float(right_rank[index, pivot].item()) < 0.0:
            right_rank[index].neg_()
            left_rank[:, index].neg_()
    root = torch.sqrt(singular_values[:retained_rank].clamp_min(0.0))
    output_stack = left_rank * root.unsqueeze(0)
    value = root.unsqueeze(1) * right_rank
    if retained_rank < target_rank:
        output_stack = torch.nn.functional.pad(
            output_stack,
            (0, target_rank - retained_rank),
        )
        value = torch.nn.functional.pad(
            value,
            (0, 0, 0, target_rank - retained_rank),
        )
    value_bfloat16 = value.to(torch.bfloat16).contiguous()
    output_bfloat16 = output_stack.to(torch.bfloat16).contiguous()
    output_size = int(selected[0].shape[0])
    output_blocks = tuple(
        output_bfloat16[index * output_size : (index + 1) * output_size].clone()
        for index in range(len(selected))
    )
    realized = output_bfloat16.to(torch.float64) @ value_bfloat16.to(torch.float64)
    factor_error = realized - stacked
    total_energy = float(singular_values.square().sum().item())
    discarded_energy = float(singular_values[retained_rank:].square().sum().item())
    body = {
        "schema": "neural-forge-lfm2-semantic-joint-vo-group-factor-v1",
        "key_value_group_index": group_index,
        "query_head_count": len(selected),
        "output_size": output_size,
        "input_size": int(selected[0].shape[1]),
        "exact_topology_rank": target_rank,
        "reduced_rowspace_dimension": int(basis_columns.shape[1]),
        "available_spectrum_rank": available_rank,
        "numerical_rank": numerical_rank,
        "retained_rank_capacity": retained_rank,
        "retained_effective_rank": retained_effective_rank,
        "rank_truncation_performed": numerical_rank > target_rank,
        "discarded_spectral_energy": discarded_energy,
        "discarded_spectral_energy_fraction": (
            0.0 if total_energy == 0.0 else discarded_energy / total_energy
        ),
        "reduced_rowspace_maximum_absolute_error": rowspace_error,
        "materialized_bfloat16_factor_rmse": float(
            torch.sqrt(factor_error.square().mean()).item()
        ),
        "materialized_bfloat16_factor_maximum_absolute_error": float(
            factor_error.abs().max().item()
        ),
        "singular_values_sha256": _tensor_sha256(singular_values),
        "value_weight_sha256": _tensor_sha256(value_bfloat16),
        "output_block_sha256": [_tensor_sha256(item) for item in output_blocks],
        "deterministic_sign_canonicalization": "right_max_abs_coordinate_nonnegative",
        "balanced_square_root_singular_value_factors": True,
    }
    return (
        value_bfloat16,
        output_blocks,
        MappingProxyType({**body, "factor_sha256": canonical_sha256(body)}),
    )


def _compile_joint_vo(
    fit_cases: Sequence[_JointVoCase],
    incumbent_value_weight: Any,
    incumbent_output_weight: Any,
    *,
    head_dim: int,
    solver_device: str,
    ridge: float = 1e-4,
) -> _JointVoProposal:
    """Solve composite V/O residuals in row-space and refactor exact GQA ranks."""

    torch = _torch()
    selected = tuple(fit_cases)
    if not selected:
        raise Lfm2SemanticAttentionRefoldError("Joint V/O FIT bank is empty.")
    incumbent_value = incumbent_value_weight.detach().to(
        device="cpu",
        dtype=torch.bfloat16,
    )
    incumbent_output = incumbent_output_weight.detach().to(
        device="cpu",
        dtype=torch.bfloat16,
    )
    query_heads = int(selected[0].head_features.shape[2])
    hidden = int(selected[0].head_features.shape[3])
    output_size = int(selected[0].endpoint_target.shape[2])
    if (
        head_dim <= 0
        or any(
            tuple(case.head_features.shape[2:]) != (query_heads, hidden)
            or int(case.endpoint_target.shape[2]) != output_size
            for case in selected
        )
        or int(incumbent_value.shape[0]) % head_dim
        or int(incumbent_value.shape[1]) != hidden
        or tuple(incumbent_output.shape) != (output_size, query_heads * head_dim)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O FIT cases disagree with the incumbent dense GQA topology."
        )
    key_value_heads = int(incumbent_value.shape[0]) // head_dim
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise Lfm2SemanticAttentionRefoldError(
            "Joint V/O FIT topology has a nonintegral query-to-value repeat."
        )
    queries_per_key_value = query_heads // key_value_heads
    baseline = _joint_vo_metrics(
        selected,
        incumbent_value,
        incumbent_output,
        head_dim=head_dim,
        compute_devices=(solver_device,),
    )
    features = torch.cat(
        tuple(case.head_features.reshape(-1, query_heads, hidden) for case in selected),
        dim=0,
    )
    targets = torch.cat(
        tuple(case.endpoint_target.reshape(-1, output_size) for case in selected),
        dim=0,
    )
    weights = torch.cat(
        tuple(case.row_weights.reshape(-1) for case in selected),
        dim=0,
    )
    baseline_predictions = torch.cat(
        tuple(
            _joint_vo_prediction(
                case,
                incumbent_value,
                incumbent_output,
                head_dim=head_dim,
                device=solver_device,
            ).reshape(-1, output_size)
            for case in selected
        ),
        dim=0,
    )
    device = torch.device(solver_device)
    root = torch.sqrt(weights.to(device=device, dtype=torch.float64))
    weighted_features = (
        features.to(device=device, dtype=torch.float64) * root[:, None, None]
    )
    weighted_residual = (targets - baseline_predictions).to(
        device=device, dtype=torch.float64
    ) * root[:, None]
    dual_gram = torch.zeros(
        (int(features.shape[0]), int(features.shape[0])),
        dtype=torch.float64,
        device=device,
    )
    with torch.inference_mode():
        for head_index in range(query_heads):
            head_rows = weighted_features[:, head_index, :]
            dual_gram.add_(head_rows @ head_rows.T)
        scale = float(dual_gram.diagonal().mean().clamp_min(1.0).item())
        effective_ridge = float(ridge) * scale
        dual = torch.linalg.solve(
            dual_gram
            + effective_ridge
            * torch.eye(
                int(dual_gram.shape[0]),
                dtype=torch.float64,
                device=device,
            ),
            weighted_residual,
        )
    if not bool(torch.isfinite(dual).all().item()):
        raise Lfm2SemanticAttentionRefoldError(
            "Closed-form joint V/O dual ridge solve produced nonfinite values."
        )
    incumbent_value64 = incumbent_value.to(torch.float64)
    incumbent_output64 = incumbent_output.to(torch.float64)
    candidate_value = incumbent_value.clone()
    candidate_output = incumbent_output.clone()
    factor_receipts = []
    for group_index in range(key_value_heads):
        first_head = group_index * queries_per_key_value
        heads = tuple(range(first_head, first_head + queries_per_key_value))
        value_slice = incumbent_value64[
            group_index * head_dim : (group_index + 1) * head_dim
        ]
        composites = []
        for head_index in heads:
            output_slice = incumbent_output64[
                :,
                head_index * head_dim : (head_index + 1) * head_dim,
            ]
            baseline_composite = output_slice @ value_slice
            delta_composite = (
                (dual.T @ weighted_features[:, head_index, :]).detach().cpu()
            )
            composites.append((baseline_composite + delta_composite).contiguous())
        generators = torch.cat(
            (
                value_slice,
                *tuple(
                    weighted_features[:, head_index, :].detach().cpu()
                    for head_index in heads
                ),
            ),
            dim=0,
        )
        group_value, group_outputs, factor_receipt = _factor_group_composites(
            composites,
            rowspace_generators=generators,
            target_rank=head_dim,
            group_index=group_index,
        )
        candidate_value[group_index * head_dim : (group_index + 1) * head_dim] = (
            group_value
        )
        for head_index, output_block in zip(heads, group_outputs, strict=True):
            candidate_output[
                :,
                head_index * head_dim : (head_index + 1) * head_dim,
            ] = output_block
        factor_receipts.append(factor_receipt)
    candidate_metrics = _joint_vo_metrics(
        selected,
        candidate_value,
        candidate_output,
        head_dim=head_dim,
        compute_devices=(solver_device,),
    )
    changed = not (
        torch.equal(candidate_value, incumbent_value)
        and torch.equal(candidate_output, incumbent_output)
    )
    improvement = (
        baseline.endpoint_relative_rmse - candidate_metrics.endpoint_relative_rmse
    )
    accepted_fit = changed and improvement > 1e-8
    if not accepted_fit:
        candidate_value = incumbent_value
        candidate_output = incumbent_output
        candidate_metrics = baseline
    fit_sha = canonical_sha256(
        {
            "case_sha256": [case.case_sha256 for case in selected],
            "probability_weighted_input_features_used": True,
        }
    )
    return _JointVoProposal(
        value_weight=candidate_value,
        output_weight=candidate_output,
        changed=accepted_fit,
        fit_improved=accepted_fit,
        baseline_fit_metrics=baseline,
        candidate_fit_metrics=candidate_metrics,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
        dual_dimension=int(features.shape[0]),
        primal_feature_dimension=query_heads * hidden,
        fit_evidence_sha256=fit_sha,
        factorization_receipts=tuple(factor_receipts),
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


def _select_and_validate_vo_candidate(
    *,
    banks: Mapping[str, Sequence[_JointVoCase]],
    seed: Lfm2EndpointQkvoSeed,
    output_proposal: _OutputWriterProposal,
    joint_proposal: _JointVoProposal,
    tolerance: float,
    validation_route_baseline: Lfm2EndpointQkMetrics,
    validation_route_candidate: Lfm2EndpointQkMetrics,
    compute_devices: Sequence[str] = (),
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Select without VALIDATION leakage, freeze once, then gate once."""

    candidates = [
        {
            "name": "incumbent",
            "complexity": 0,
            "value": seed.value_weight,
            "output": seed.output_weight,
            "proposal_changed": False,
            "proposal_sha256": seed.seed_sha256,
        },
        {
            "name": "fixed_v_output_only",
            "complexity": 1,
            "value": seed.value_weight,
            "output": output_proposal.output_weight,
            "proposal_changed": output_proposal.changed,
            "proposal_sha256": output_proposal.proposal_sha256,
        },
        {
            "name": "joint_value_output",
            "complexity": 2,
            "value": joint_proposal.value_weight,
            "output": joint_proposal.output_weight,
            "proposal_changed": joint_proposal.changed,
            "proposal_sha256": joint_proposal.proposal_sha256,
        },
    ]
    baseline_metrics = {
        partition: _joint_vo_metrics(
            banks[partition],
            seed.value_weight,
            seed.output_weight,
            head_dim=seed.head_dim,
            compute_devices=compute_devices,
        )
        for partition in ("fit", "fit_check", "selection")
    }
    reports: dict[str, dict[str, Any]] = {}
    eligible = []
    for candidate in candidates:
        partition_reports = {}
        rejection_partition = None
        for partition in ("fit", "fit_check", "selection"):
            metrics = _joint_vo_metrics(
                banks[partition],
                candidate["value"],
                candidate["output"],
                head_dim=seed.head_dim,
                compute_devices=compute_devices,
            )
            nonregressing = _writer_nonregressing(
                metrics,
                baseline_metrics[partition],
                tolerance=tolerance,
            )
            if candidate["name"] != "incumbent" and partition == "fit":
                nonregressing = (
                    nonregressing
                    and metrics.endpoint_relative_rmse
                    < baseline_metrics[partition].endpoint_relative_rmse
                )
            partition_reports[partition] = {
                "case_sha256": [case.case_sha256 for case in banks[partition]],
                "baseline": baseline_metrics[partition].as_dict(),
                "candidate": metrics.as_dict(),
                "nonregressing": nonregressing,
            }
            if rejection_partition is None and not nonregressing:
                rejection_partition = partition
        is_eligible = candidate["name"] == "incumbent" or (
            candidate["proposal_changed"] and rejection_partition is None
        )
        reports[candidate["name"]] = {
            "complexity": candidate["complexity"],
            "proposal_changed": candidate["proposal_changed"],
            "proposal_sha256": candidate["proposal_sha256"],
            "eligible_after_fit_fit_check_selection": is_eligible,
            "rejection_partition": (
                rejection_partition
                if candidate["proposal_changed"]
                else (None if candidate["name"] == "incumbent" else "fit")
            ),
            "partitions": partition_reports,
        }
        if is_eligible:
            eligible.append(candidate)
    strongest = min(
        eligible,
        key=lambda candidate: (
            reports[candidate["name"]]["partitions"]["selection"]["candidate"][
                "endpoint_relative_rmse"
            ],
            reports[candidate["name"]]["partitions"]["selection"]["candidate"][
                "endpoint_maximum_absolute_error"
            ],
            -reports[candidate["name"]]["partitions"]["selection"]["candidate"][
                "endpoint_cosine_similarity_mean"
            ],
            -reports[candidate["name"]]["partitions"]["selection"]["candidate"][
                "endpoint_cosine_similarity_worst"
            ],
            candidate["complexity"],
        ),
    )
    strongest_metrics = _joint_vo_metrics(
        banks["selection"],
        strongest["value"],
        strongest["output"],
        head_dim=seed.head_dim,
        compute_devices=compute_devices,
    )
    selected = next(
        candidate
        for candidate in sorted(eligible, key=lambda item: item["complexity"])
        if _writer_nonregressing(
            _joint_vo_metrics(
                banks["selection"],
                candidate["value"],
                candidate["output"],
                head_dim=seed.head_dim,
                compute_devices=compute_devices,
            ),
            strongest_metrics,
            tolerance=tolerance,
        )
    )
    frozen = {
        "value": selected["value"].detach().to(device="cpu").contiguous().clone(),
        "output": selected["output"].detach().to(device="cpu").contiguous().clone(),
    }
    frozen_sha = canonical_sha256(
        {
            "candidate_name": selected["name"],
            "value_weight_sha256": _tensor_sha256(frozen["value"]),
            "output_weight_sha256": _tensor_sha256(frozen["output"]),
            "proposal_sha256": selected["proposal_sha256"],
        }
    )
    route_nonregressing = _metrics_nonregressing(
        validation_route_candidate,
        validation_route_baseline,
        tolerance=tolerance,
    )
    if not route_nonregressing:
        raise Lfm2SemanticAttentionRefoldError(
            "Frozen V/O selection reached VALIDATION with regressing Q/K routes."
        )
    validation_baseline = _joint_vo_metrics(
        banks["validation"],
        seed.value_weight,
        seed.output_weight,
        head_dim=seed.head_dim,
        compute_devices=compute_devices,
    )
    validation_selected = _joint_vo_metrics(
        banks["validation"],
        frozen["value"],
        frozen["output"],
        head_dim=seed.head_dim,
        compute_devices=compute_devices,
    )
    endpoint_nonregressing = _writer_nonregressing(
        validation_selected,
        validation_baseline,
        tolerance=tolerance,
    )
    if _tensor_sha256(frozen["value"]) != _tensor_sha256(
        selected["value"]
    ) or _tensor_sha256(frozen["output"]) != _tensor_sha256(selected["output"]):
        raise Lfm2SemanticAttentionRefoldError(
            "Selected V/O candidate changed after the pre-VALIDATION freeze."
        )
    published_name = selected["name"] if endpoint_nonregressing else "incumbent"
    published_value = frozen["value"] if endpoint_nonregressing else seed.value_weight
    published_output = (
        frozen["output"] if endpoint_nonregressing else seed.output_weight
    )
    body = {
        "schema": "neural-forge-lfm2-semantic-vo-candidate-gate-v1",
        "candidate_reports": reports,
        "strongest_selection_candidate": strongest["name"],
        "selected_candidate": selected["name"],
        "published_candidate": published_name,
        "selection_rule": (
            "choose_lowest_complexity_candidate_only_when_it_nonregresses_the_"
            "strongest_candidate_on_isolated_selection"
        ),
        "complexity_order": [
            "incumbent",
            "fixed_v_output_only",
            "joint_value_output",
        ],
        "selected_candidate_frozen_before_validation": True,
        "frozen_candidate_sha256": frozen_sha,
        "validation_triggered_refit_or_reselection": False,
        "validation": {
            "case_sha256": [case.case_sha256 for case in banks["validation"]],
            "endpoint_baseline": validation_baseline.as_dict(),
            "endpoint_candidate": validation_selected.as_dict(),
            "endpoint_nonregressing": endpoint_nonregressing,
            "route_baseline": validation_route_baseline.as_dict(),
            "route_candidate": validation_route_candidate.as_dict(),
            "route_nonregressing": route_nonregressing,
            "gated_metrics": [
                "endpoint_relative_rmse",
                "endpoint_maximum_absolute_error",
                "endpoint_cosine_similarity_mean",
                "endpoint_cosine_similarity_worst",
                "qk_route_metrics",
            ],
        },
        "selected_correction_rejected_on_validation": (
            selected["name"] != "incumbent" and not endpoint_nonregressing
        ),
        "alternate_candidate_selected_after_validation": False,
        "published_value_weight_sha256": _tensor_sha256(published_value),
        "published_output_weight_sha256": _tensor_sha256(published_output),
        "fit_only_analytic_solves": True,
        "inserted_target_query_rows_parent_exact": False,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return (
        published_value,
        published_output,
        MappingProxyType({**body, "gate_sha256": canonical_sha256(body)}),
    )


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


def _copy_candidate(parameter: Any, candidate: Any) -> None:
    """Single observable mutation point; authenticated total no-ops bypass it."""

    parameter.copy_(candidate.to(device=parameter.device))


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
    topology = _topology_sha256(target_model)
    parameter_ids = {name: id(value) for name, value in parameters.items()}
    parameter_pointers = {
        name: int(value.data_ptr()) for name, value in parameters.items()
    }
    parameter_shapes = {
        name: tuple(value.shape) for name, value in parameters.items()
    }
    parameter_dtypes = {name: value.dtype for name, value in parameters.items()}
    parameter_requires_grad = {
        name: bool(value.requires_grad) for name, value in parameters.items()
    }
    gradient_ids = {
        name: None if value.grad is None else id(value.grad)
        for name, value in parameters.items()
    }
    gradient_hashes = {
        name: None if value.grad is None else _tensor_sha256(value.grad)
        for name, value in parameters.items()
    }
    prefix = f"model.layers.{layer_index}.self_attn."

    def integrity_preserved() -> bool:
        return bool(
            _parameter_count(target_model) == count
            and _topology_sha256(target_model) == topology
            and all(id(parameters[name]) == parameter_ids[name] for name in parameters)
            and all(
                int(parameters[name].data_ptr()) == parameter_pointers[name]
                for name in parameters
            )
            and all(
                tuple(parameters[name].shape) == parameter_shapes[name]
                for name in parameters
            )
            and all(
                parameters[name].dtype == parameter_dtypes[name] for name in parameters
            )
            and all(
                bool(parameters[name].requires_grad)
                == parameter_requires_grad[name]
                for name in parameters
            )
            and all(
                (None if parameters[name].grad is None else id(parameters[name].grad))
                == gradient_ids[name]
                for name in parameters
            )
            and all(
                parameters[name].grad is None
                or _tensor_sha256(parameters[name].grad) == gradient_hashes[name]
                for name in parameters
            )
        )

    candidates = {}
    for name, parameter in parameters.items():
        candidate = suffix_state[name.removeprefix(prefix)]
        if (
            not isinstance(candidate, torch.Tensor)
            or tuple(candidate.shape) != tuple(parameter.shape)
            or candidate.dtype != parameter.dtype
            or candidate.requires_grad
            or candidate.grad_fn is not None
        ):
            raise Lfm2SemanticAttentionRefoldError(
                f"Compiled state changed the topology or gradient state of {name}."
            )
        candidates[name] = candidate.detach().to(device="cpu").contiguous()
    expected_changed = tuple(
        name
        for name in parameters
        if not torch.equal(candidates[name], snapshots[name])
    )
    if not expected_changed:
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_model
            or not integrity_preserved()
            or any(
                not torch.equal(parameters[name].detach().cpu(), snapshots[name])
                for name in parameters
            )
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Authenticated attention incumbent no-op did not preserve exact state."
            )
        return before_model, before_model, (), count
    try:
        with torch.no_grad():
            for name in expected_changed:
                _copy_candidate(parameters[name], candidates[name])
        if not integrity_preserved():
            raise Lfm2SemanticAttentionRefoldError(
                "Attention installation changed topology, identities, or gradients."
            )
        changed = tuple(
            name
            for name, parameter in parameters.items()
            if not torch.equal(parameter.detach().cpu(), snapshots[name])
        )
        after_model = fingerprint_lfm2_parent(target_model).identity_sha256
        if changed != expected_changed or after_model == before_model:
            raise Lfm2SemanticAttentionRefoldError(
                "Frozen semantic attention candidate did not install exactly its correction."
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
            or not integrity_preserved()
            or any(
                not torch.equal(parameters[name].detach().cpu(), snapshots[name])
                for name in parameters
            )
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
    incumbent_seed = _seed(
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
    banks = MappingProxyType(
        {
            "fit": fit_bank,
            "fit_check": fit_check_bank,
            "selection": selection_bank,
            "validation": validation_bank,
        }
    )
    seed, cold_start = _cold_start_qk_seed(
        incumbent_seed,
        banks=banks,
        policy=policy,
    )
    compiled = compile_lfm2_cached_endpoint_qk_residual_fold(
        seed=seed,
        fit_bank=fit_bank,
        fit_check_bank=fit_check_bank,
        policy=policy,
    )
    if (
        not isinstance(compiled, Lfm2CompiledEndpointQkResidualFold)
        or compiled.verify() != compiled.report["report_sha256"]
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "FIT/FIT-CHECK produced no authenticated Q/K result."
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
    compiler_query_changed = not torch.equal(frozen_query, seed.query_weight)
    compiler_key_changed = not torch.equal(frozen_key, seed.key_weight)
    if compiled.changed != (
        compiler_query_changed or compiler_key_changed
    ) or compiled.accepted != (
        compiler_query_changed or compiler_key_changed
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Authenticated Q/K acceptance disagrees with its published tensors."
        )
    query_changed = not torch.equal(frozen_query, incumbent_seed.query_weight)
    key_changed = not torch.equal(frozen_key, incumbent_seed.key_weight)
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
    if cold_start is not None and (
        (
            cold_start["original_query_exact_zero"] is True
            and _exact_zero(frozen_query)
        )
        or (
            cold_start["original_key_exact_zero"] is True
            and _exact_zero(frozen_key)
        )
        or not (query_changed or key_changed)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Exact-zero Q/K cold-start cannot publish a zero or total-incumbent no-op."
        )
    joint_banks = {
        "fit": _joint_vo_cases(
            fit_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
        "fit_check": _joint_vo_cases(
            fit_check_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
        "selection": _joint_vo_cases(
            selection_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
        "validation": _joint_vo_cases(
            validation_bank,
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        ),
    }
    writer_banks = {
        name: _writer_cases_from_joint(
            joint_banks[name],
            seed,
            compute_devices=policy.compute_devices,
        )
        for name in _PARTITIONS
    }
    writer_proposal = _compile_output_writer(
        writer_banks["fit"],
        seed.output_weight,
        solver_device=policy.solver_device,
    )
    joint_proposal = _compile_joint_vo(
        joint_banks["fit"],
        seed.value_weight,
        seed.output_weight,
        head_dim=seed.head_dim,
        solver_device=policy.solver_device,
    )
    published_value, published_output, vo_gate = _select_and_validate_vo_candidate(
        banks=joint_banks,
        seed=seed,
        output_proposal=writer_proposal,
        joint_proposal=joint_proposal,
        tolerance=tolerance,
        validation_route_baseline=baseline_validation,
        validation_route_candidate=candidate_validation,
        compute_devices=policy.compute_devices,
    )
    published_vo_kind = str(vo_gate["published_candidate"])
    if published_vo_kind not in {
        "incumbent",
        "fixed_v_output_only",
        "joint_value_output",
    }:
        raise Lfm2SemanticAttentionRefoldError(
            "V/O candidate gate published an unknown candidate role."
        )
    writer_accepted = published_vo_kind == "fixed_v_output_only"
    joint_accepted = published_vo_kind == "joint_value_output"
    value_changed = not torch.equal(published_value, incumbent_seed.value_weight)
    output_changed = not torch.equal(published_output, incumbent_seed.output_weight)
    if (
        (writer_accepted and (value_changed or not output_changed))
        or (joint_accepted and not (value_changed or output_changed))
        or (
            not writer_accepted
            and not joint_accepted
            and (value_changed or output_changed)
        )
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "V/O candidate gate acceptance disagrees with its published state."
        )
    total_incumbent_noop = not (
        query_changed or key_changed or value_changed or output_changed
    )
    if total_incumbent_noop:
        incumbent_report = vo_gate.get("candidate_reports", {}).get("incumbent", {})
        incumbent_partitions = incumbent_report.get("partitions", {})
        validation_report = vo_gate.get("validation", {})
        if (
            compiled.accepted
            or compiled.changed
            or published_vo_kind != "incumbent"
            or incumbent_report.get(
                "eligible_after_fit_fit_check_selection"
            )
            is not True
            or any(
                incumbent_partitions.get(name, {}).get("nonregressing") is not True
                for name in ("fit_check", "selection")
            )
            or vo_gate.get("selected_candidate_frozen_before_validation") is not True
            or vo_gate.get("validation_triggered_refit_or_reselection") is not False
            or validation_report.get("endpoint_nonregressing") is not True
            or validation_report.get("route_nonregressing") is not True
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Total attention no-op lacks authenticated held-out incumbent gates."
            )
    output_report = vo_gate["candidate_reports"]["fixed_v_output_only"]
    writer_gate = _seal_writer_gate(
        {
            "status": (
                "accepted-after-frozen-validation"
                if writer_accepted
                else "rejected-by-joint-vo-candidate-competition"
            ),
            "accepted": writer_accepted,
            "proposal": writer_proposal.as_dict(),
            "rejection_partition": (
                None
                if writer_accepted
                else (output_report["rejection_partition"] or "selection")
            ),
            "fit_check": output_report["partitions"]["fit_check"],
            "selection": output_report["partitions"]["selection"],
            "validation": vo_gate["validation"] if writer_accepted else None,
            "candidate_frozen_across_heldout_gates": writer_accepted,
            "competed_against_incumbent_and_joint_value_output": True,
        }
    )
    frozen_qkvo_sha = canonical_sha256(
        {
            "q": _tensor_sha256(frozen_query),
            "k": _tensor_sha256(frozen_key),
            "v": _tensor_sha256(published_value),
            "o": _tensor_sha256(published_output),
        }
    )
    state = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": published_value,
        "out_proj.weight": published_output,
    }
    topology_before = _topology_sha256(target_model)
    before, after, changed, parameter_count = _install_qkvo_transaction(
        target_model,
        layer_index=layer_index,
        state=state,
    )
    topology_after = _topology_sha256(target_model)
    base = _installation_receipt_body(
        layer_index=layer_index,
        evidence_sha256=[item.evidence_sha256 for item in raw],
        model_state_before_sha256=before,
        model_state_after_sha256=after,
        changed_state_keys=changed,
        parameter_element_count=parameter_count,
        target_topology_sha256_before=topology_before,
        target_topology_sha256_after=topology_after,
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
            **(
                {}
                if cold_start is None
                else {"qk_cold_start": dict(cold_start)}
            ),
            "endpoint_compiler_report_sha256": compiled.report["report_sha256"],
            "analytic_execution_policy": {
                "solver_device": policy.solver_device,
                "compute_devices": list(policy.compute_devices),
                "native_bfloat16_sm80_or_newer_only": True,
                "non_native_bfloat16_devices_used": False,
                "candidate_endpoint_replay_follows_compute_devices": True,
            },
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
            "joint_value_output_candidate_frozen_before_validation": (
                joint_proposal.changed
            ),
            "validation_triggered_refit": False,
            "numeric_parent_child_weight_match_objective": False,
            "dense_existing_qkvo_topology_only": True,
            "analytically_refit_projection_roles": [
                *(["q_proj"] if query_changed else []),
                *(["k_proj"] if key_changed else []),
                *(["v_proj"] if value_changed else []),
                *(["out_proj"] if output_changed else []),
            ],
            "fixed_inherited_projection_roles": [
                *([] if query_changed else ["q_proj"]),
                *([] if key_changed else ["k_proj"]),
                *([] if value_changed else ["v_proj"]),
                *([] if output_changed else ["out_proj"]),
            ],
            "qk_compiler_accepted": compiled.accepted,
            "qk_compiler_changed": compiled.changed,
            "authenticated_qk_seed_fallback_entered_vo_competition": (
                not compiled.changed
            ),
            "fixed_value_source": (
                "fit_only_joint_residual_composite_low_rank_factorization"
                if value_changed
                else "authenticated_live_c61_dense_seed_already_folded_on_broad_cache"
            ),
            "output_writer_source": (
                "fit_only_joint_residual_composite_low_rank_factorization"
                if joint_accepted and output_changed
                else (
                    "fit_only_closed_form_weighted_residual_ridge"
                    if writer_accepted
                    else "authenticated_live_c61_seed_after_explicit_analytic_rejection"
                )
            ),
            "output_writer": dict(writer_gate),
            "joint_value_output": joint_proposal.as_dict(),
            "value_output_candidate_gate": dict(vo_gate),
            "output_writer_cached_qk_probability_replay_calls": sum(
                len(bank) for bank in joint_banks.values()
            ),
            "output_writer_model_forward_calls": 0,
            "output_writer_fit_partition_only_used_for_solve": True,
            "output_writer_heldout_partitions_never_used_for_solve": True,
            "joint_value_output_candidate_published": joint_accepted,
            "fresh_joint_value_output_refit_performed": (
                joint_accepted and value_changed and output_changed
            ),
            "fresh_value_refit_performed": value_changed,
            "fresh_output_writer_refit_performed": output_changed,
            "fresh_query_refit_performed": query_changed,
            "fresh_key_refit_performed": key_changed,
            "fresh_four_matrix_qkvo_refit_claimed": (
                query_changed
                and key_changed
                and joint_accepted
                and value_changed
                and output_changed
            ),
            "complete_qkvo_bundle_installed_transactionally": bool(changed),
            "complete_qkvo_bundle_transaction_verified": True,
            "authenticated_incumbent_qkvo_retained_without_installation": (
                not changed
            ),
            "projection_parameter_copy_calls": len(changed),
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
