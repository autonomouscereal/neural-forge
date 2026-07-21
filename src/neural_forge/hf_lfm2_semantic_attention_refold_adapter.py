"""Closed-form semantic-span attention refolding for a saved dense LFM2 child.

The semantic lattice is a teacher, not a weight-matching target.  This module
turns its probability/context evidence and an authenticated actual-child
macrocell replay into endpoint cases for the existing matrix-free Q/K
compiler.  FIT, FIT-CHECK, SELECTION, and VALIDATION remain disjoint.  The
candidate is frozen before VALIDATION and dense Q/K/V/O is copied into the
existing child topology only after every held-out gate passes.

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


LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-refold-v1"
)
LFM2_SEMANTIC_ATTENTION_REPLAY_ENDPOINT_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-replay-endpoint-v1"
)
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
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
    if attention is None or any(getattr(attention, name, None) is None for name in required):
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
    if (
        int(replay.target_query_inputs.shape[1]) != query_length
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
) -> Lfm2EndpointQkResidualCase:
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
        }
    )
    return Lfm2EndpointQkResidualCase(
        target_query_inputs=source.replay.target_query_inputs,
        target_key_value_inputs=source.replay.target_key_value_inputs,
        parent_probabilities=evidence["parent_probabilities"],
        parent_causal_mask=evidence["parent_causal_mask"],
        parent_endpoint_target=source.replay.parent_endpoint_target,
        query_rope_cos=evidence["query_rope_cos"],
        query_rope_sin=evidence["query_rope_sin"],
        key_rope_cos=evidence["key_rope_cos"],
        key_rope_sin=evidence["key_rope_sin"],
        query_row_sqrt_weights=source.replay.query_row_sqrt_weights,
        query_groups=source.query_groups,
        query_group_weights=source.query_group_weights,
        trajectory_sha256=trajectory,
        parent_probability_provenance=evidence[
            "parent_probability_provenance"
        ],
        parent_context_replay_relative_rmse=0.0,
        parent_context_replay_maximum_absolute_error=0.0,
        parent_context_replay_rtol=0.0,
        parent_context_replay_atol=0.0,
    )


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
        not isinstance(value, torch.nn.Parameter)
        or value.dtype != torch.bfloat16
        for value in tensors.values()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Target Q/K/V/O and Q/K norms must be frozen dense BF16 Parameters."
        )
    epsilon = getattr(getattr(target_model, "config", None), "norm_eps", None)
    if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)):
        raise Lfm2SemanticAttentionRefoldError(
            "Target model has no finite norm_eps."
        )
    cpu = {name: value.detach().to(device="cpu").contiguous().clone() for name, value in tensors.items()}
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
        candidate.endpoint_relative_rmse
        <= baseline.endpoint_relative_rmse + tolerance
        and candidate.probability_relative_rmse
        <= baseline.probability_relative_rmse + tolerance
        and candidate.js_divergence_mean <= baseline.js_divergence_mean + tolerance
        and candidate.js_divergence_worst <= baseline.js_divergence_worst + tolerance
        and candidate.top1_route_agreement_fraction + tolerance
        >= baseline.top1_route_agreement_fraction
        and candidate.maximum_causal_probability_leakage == 0.0
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
                if tuple(candidate.shape) != tuple(parameter.shape) or candidate.dtype != parameter.dtype:
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
        _source_contract(item)
        != (inventory, groups, weights, lineage)
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
        or any(len(group) != len(group_weights) for group, group_weights in zip(groups, weights, strict=True))
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
    cases = {
        name: tuple(
            _case_from_source(
                item,
                layer_index=layer_index,
                cache_key=cache,
                bridge_artifact_sha256=bridge,
            )
            for item in partitions[name]
        )
        for name in _PARTITIONS
    }
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
    frozen_sha = canonical_sha256(
        {
            "q": _tensor_sha256(frozen_query),
            "k": _tensor_sha256(frozen_key),
            "v": _tensor_sha256(seed.value_weight),
            "o": _tensor_sha256(seed.output_weight),
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
    state = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": seed.value_weight,
        "out_proj.weight": seed.output_weight,
    }
    before, after, changed, parameter_count = _install_qkvo_transaction(
        target_model,
        layer_index=layer_index,
        state=state,
    )
    from neural_forge.hf_lfm2_semantic_postfold import (
        seal_lfm2_semantic_attention_installation_receipt,
    )

    base = dict(
        seal_lfm2_semantic_attention_installation_receipt(
            layer_index=layer_index,
            transport_evidence_sha256=[item.evidence_sha256 for item in raw],
            model_state_before_sha256=before,
            model_state_after_sha256=after,
            changed_state_keys=changed,
            parameter_element_count=parameter_count,
        )
    )
    base.pop("result_sha256")
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
            "fit_evidence_sha256": fit_bank.evidence_sha256,
            "fit_check_evidence_sha256": fit_check_bank.evidence_sha256,
            "selection_evidence_sha256": selection_bank.evidence_sha256,
            "validation_evidence_sha256": validation_bank.evidence_sha256,
            "endpoint_compiler_report_sha256": compiled.report["report_sha256"],
            "selection_baseline": baseline_selection.as_dict(),
            "selection_candidate": candidate_selection.as_dict(),
            "validation_baseline": baseline_validation.as_dict(),
            "validation_candidate": candidate_validation.as_dict(),
            "frozen_qkvo_sha256": frozen_sha,
            "candidate_frozen_before_validation": True,
            "validation_triggered_refit": False,
            "numeric_parent_child_weight_match_objective": False,
            "dense_existing_qkvo_topology_only": True,
            "analytically_refit_projection_roles": ["q_proj", "k_proj"],
            "fixed_inherited_projection_roles": ["v_proj", "out_proj"],
            "fixed_value_output_source": (
                "authenticated_live_c61_dense_seed_already_folded_on_broad_cache"
            ),
            "fresh_value_output_refit_performed": False,
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
