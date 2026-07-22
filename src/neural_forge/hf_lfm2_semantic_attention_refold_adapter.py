"""Closed-form semantic-span attention refolding for a saved dense LFM2 child.

The semantic lattice is a teacher, not a weight-matching target.  This module
turns its probability/context evidence and an authenticated actual-child
macrocell replay into endpoint cases for the existing matrix-free Q/K
compiler, then competes an analytic fixed-V output writer with a joint V/O
residual factorization.  The latter solves in endpoint-row space and projects
each GQA group's composite maps back through its exact shared-V,
head-specific-O rank.  FIT, FIT-CHECK, SELECTION, and VALIDATION remain
disjoint.  Candidates are frozen before VALIDATION and dense Q/K/V/O is copied
into the existing child topology after FIT-only structural selection.  Every
held-out partition is evaluated after the selected tensors are frozen and is
diagnostic only; it can neither reject the candidate nor restore an incumbent.

No model is executed here.  There are no gradients, optimizers, training
steps, pruning operations, quantization operations, or numerical parent/child
weight-equality objectives.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from neural_forge.hf_lfm2_semantic_body_recipe import (
    Lfm2SemanticBodyRecipe,
    lfm2_semantic_body_recipe_ids_for_component_lane,
    resolve_lfm2_semantic_body_recipe,
)
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
LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS = (
    "structurally-identical-or-invalid-incumbent-noop"
)
_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-attention-installation-v1"
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
_VO_CANDIDATE_ELIGIBILITY_BASIS = (
    "finite_dense_bfloat16_exact_shape_and_nonincumbent_effect"
)
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
    structurally_eligible_candidate_count: int,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> dict[str, Any]:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
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
        or isinstance(structurally_eligible_candidate_count, bool)
        or not isinstance(structurally_eligible_candidate_count, int)
        or structurally_eligible_candidate_count < 0
        or bool(changed) != (before != after)
        or bool(changed) != (structurally_eligible_candidate_count > 0)
        or topology_before != topology_after
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Attention installation receipt inputs are incomplete."
        )
    installed = bool(changed)
    return {
        "schema": _INSTALLATION_SCHEMA,
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["qk"],
        "status": (
            "installed-frozen-semantic-attention-candidate"
            if installed
            else LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS
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
        "selected_candidate": (
            "frozen-semantic-attention-candidate" if installed else "incumbent-noop"
        ),
        "selected_candidate_state_identical_to_incumbent": not installed,
        "incumbent_noop_fallback_used": not installed,
        "incumbent_noop_fallback_reason": (
            None
            if installed
            else "all-analytic-proposals-structurally-invalid-or-identical"
        ),
        "structurally_eligible_candidate_count": (
            structurally_eligible_candidate_count
        ),
        "candidate_eligibility_basis": _VO_CANDIDATE_ELIGIBILITY_BASIS,
        "structural_safety_is_only_candidate_veto": True,
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates
        ),
        "fit_metrics_used_for_structural_eligibility": False,
        "incumbent_selected_by_heldout_gates": False,
        "fit_check_nonregression_passed": None,
        "selection_nonregression_passed": None,
        "validation_nonregression_passed": None,
        "heldout_metrics_are_diagnostic_only": True,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "validation_rejected_candidate": None,
        "numeric_metric_used_for_publication_or_rollback": False,
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
    key_value_groups: tuple[tuple[int, ...], ...]
    key_value_group_weights: tuple[tuple[float, ...], ...]
    folded_parent_query_pre_rope: Any | None
    folded_parent_key_pre_rope: Any | None
    parent_query_weight_prior: Any | None
    parent_key_weight_prior: Any | None
    parent_value_weight_prior: Any | None
    parent_output_weight_prior: Any | None
    parent_query_norm_weight: Any | None
    parent_key_norm_weight: Any | None
    parent_weight_prior_receipt: Mapping[str, Any] | None


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
    key_value_groups = tuple(
        tuple(int(member) for member in group)
        for group in getattr(evidence, "key_value_groups", ())
    )
    key_value_weights = tuple(
        tuple(float(weight) for weight in group)
        for group in getattr(evidence, "key_value_group_weights", ())
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
        or not key_value_groups
        or len(key_value_groups) != len(key_value_weights)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Evidence lacks its typed parent Q/K/V/O inventory or original head groups."
        )
    return (
        inventory,
        groups,
        weights,
        key_value_groups,
        key_value_weights,
        lineage,
    )


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
    (
        inventory,
        groups,
        weights,
        key_value_groups,
        key_value_weights,
        lineage,
    ) = _source_contract(evidence)
    del inventory
    folded_parent_query_pre_rope = getattr(
        evidence,
        "folded_parent_query_pre_rope",
        None,
    )
    folded_parent_key_pre_rope = getattr(
        evidence,
        "folded_parent_key_pre_rope",
        None,
    )
    parent_prior_receipt = getattr(
        evidence,
        "parent_pre_rope_prior_receipt",
        None,
    )
    parent_query_weight_prior = getattr(evidence, "parent_query_weight_prior", None)
    parent_key_weight_prior = getattr(evidence, "parent_key_weight_prior", None)
    parent_value_weight_prior = getattr(evidence, "parent_value_weight_prior", None)
    parent_output_weight_prior = getattr(evidence, "parent_output_weight_prior", None)
    parent_query_norm_weight = getattr(evidence, "parent_query_norm_weight", None)
    parent_key_norm_weight = getattr(evidence, "parent_key_norm_weight", None)
    parent_weight_prior_receipt = getattr(
        evidence,
        "parent_weight_prior_receipt",
        None,
    )
    if any(
        value is not None
        for value in (
            folded_parent_query_pre_rope,
            folded_parent_key_pre_rope,
            parent_prior_receipt,
        )
    ):
        if (
            folded_parent_query_pre_rope is None
            or folded_parent_key_pre_rope is None
            or not isinstance(parent_prior_receipt, Mapping)
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Parent pre-RoPE activation evidence is incomplete."
            )
        prior_body = dict(parent_prior_receipt)
        supplied_prior_sha = prior_body.pop("prior_sha256", None)
        if (
            supplied_prior_sha != canonical_sha256(prior_body)
            or parent_prior_receipt.get("query_prior_sha256")
            != _tensor_sha256(folded_parent_query_pre_rope)
            or parent_prior_receipt.get("key_prior_sha256")
            != _tensor_sha256(folded_parent_key_pre_rope)
            or parent_prior_receipt.get("parent_activation_evidence_only") is not True
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Parent pre-RoPE activation evidence changed after sealing."
            )
    weight_prior_values = (
        parent_query_weight_prior,
        parent_key_weight_prior,
        parent_value_weight_prior,
        parent_output_weight_prior,
        parent_query_norm_weight,
        parent_key_norm_weight,
        parent_weight_prior_receipt,
    )
    if any(value is not None for value in weight_prior_values):
        if not all(
            value is not None for value in weight_prior_values
        ) or not isinstance(
            parent_weight_prior_receipt,
            Mapping,
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Parent Q/K weight and norm prior evidence is incomplete."
            )
        weight_body = dict(parent_weight_prior_receipt)
        supplied_weight_sha = weight_body.pop("prior_sha256", None)
        if (
            supplied_weight_sha != canonical_sha256(weight_body)
            or parent_weight_prior_receipt.get("query_prior_sha256")
            != _tensor_sha256(parent_query_weight_prior)
            or parent_weight_prior_receipt.get("key_prior_sha256")
            != _tensor_sha256(parent_key_weight_prior)
            or parent_weight_prior_receipt.get("value_prior_sha256")
            != _tensor_sha256(parent_value_weight_prior)
            or parent_weight_prior_receipt.get("output_prior_sha256")
            != _tensor_sha256(parent_output_weight_prior)
            or parent_weight_prior_receipt.get("query_norm_weight_sha256")
            != _tensor_sha256(parent_query_norm_weight)
            or parent_weight_prior_receipt.get("key_norm_weight_sha256")
            != _tensor_sha256(parent_key_norm_weight)
            or parent_weight_prior_receipt.get("direct_parent_weights_used") is not True
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Parent Q/K weight and norm prior changed after sealing."
            )
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
                "Expansion transport is contracted, approximate, or does not "
                "cover the complete exact semantic suffix endpoint set."
            )
        query_length = len(transport.query_target_positions)
        key_length = len(transport.target_to_donor_span_owner_positions)
    else:
        query_length = int(transport.probabilities.shape[2])
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
        key_value_groups=key_value_groups,
        key_value_group_weights=key_value_weights,
        folded_parent_query_pre_rope=folded_parent_query_pre_rope,
        folded_parent_key_pre_rope=folded_parent_key_pre_rope,
        parent_query_weight_prior=parent_query_weight_prior,
        parent_key_weight_prior=parent_key_weight_prior,
        parent_value_weight_prior=parent_value_weight_prior,
        parent_output_weight_prior=parent_output_weight_prior,
        parent_query_norm_weight=parent_query_norm_weight,
        parent_key_norm_weight=parent_key_norm_weight,
        parent_weight_prior_receipt=parent_weight_prior_receipt,
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
    evidence = source.transport.endpoint_residual_evidence()
    query_length = int(evidence["parent_probabilities"].shape[2])
    key_length = int(evidence["parent_probabilities"].shape[3])
    replay_query = source.replay.target_query_inputs
    replay_endpoint = source.replay.parent_endpoint_target
    replay_weights = source.replay.query_row_sqrt_weights
    if int(replay_query.shape[1]) == key_length and query_length != key_length:
        if not isinstance(source.transport, SemanticAttentionExpansionTransport):
            raise Lfm2SemanticAttentionRefoldError(
                "Full-history replay query selection requires expansion positions."
            )
        endpoint_index = _torch().tensor(
            source.transport.query_target_positions,
            dtype=_torch().long,
        )
        replay_query = replay_query.index_select(1, endpoint_index)
        replay_endpoint = replay_endpoint.index_select(1, endpoint_index)
        replay_weights = replay_weights.index_select(1, endpoint_index)
    if int(replay_query.shape[1]) != query_length:
        raise Lfm2SemanticAttentionRefoldError(
            "Actual-child replay does not align every exact semantic endpoint."
        )
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
        target_query_inputs=replay_query,
        target_key_value_inputs=source.replay.target_key_value_inputs,
        parent_probabilities=evidence["parent_probabilities"],
        parent_causal_mask=evidence["parent_causal_mask"],
        parent_endpoint_target=replay_endpoint,
        query_rope_cos=evidence["query_rope_cos"],
        query_rope_sin=evidence["query_rope_sin"],
        key_rope_cos=evidence["key_rope_cos"],
        key_rope_sin=evidence["key_rope_sin"],
        query_row_sqrt_weights=(replay_weights * trajectory_row_weight_scale),
        query_groups=source.query_groups,
        query_group_weights=source.query_group_weights,
        key_value_groups=source.key_value_groups,
        key_value_group_weights=source.key_value_group_weights,
        trajectory_sha256=trajectory,
        parent_probability_provenance=evidence["parent_probability_provenance"],
        parent_context_replay_relative_rmse=0.0,
        parent_context_replay_maximum_absolute_error=0.0,
        parent_context_replay_rtol=0.0,
        parent_context_replay_atol=0.0,
        folded_parent_query_pre_rope=(
            None
            if source.folded_parent_query_pre_rope is None
            else source.folded_parent_query_pre_rope
        ),
        folded_parent_key_pre_rope=source.folded_parent_key_pre_rope,
        parent_query_weight_prior=source.parent_query_weight_prior,
        parent_key_weight_prior=source.parent_key_weight_prior,
        parent_value_weight_prior=source.parent_value_weight_prior,
        parent_output_weight_prior=source.parent_output_weight_prior,
        parent_query_norm_weight=source.parent_query_norm_weight,
        parent_key_norm_weight=source.parent_key_norm_weight,
        parent_weight_prior_sha256=(
            None
            if source.parent_weight_prior_receipt is None
            else source.parent_weight_prior_receipt["prior_sha256"]
        ),
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
        endpoint_weights = source.replay.query_row_sqrt_weights.to(torch.float64)
        mass = float(endpoint_weights.square().sum().item())
        if not math.isfinite(mass) or mass <= 0.0:
            raise Lfm2SemanticAttentionRefoldError(
                "Replay-provided semantic endpoint row weights have no positive mass."
            )
        raw_mass[source.batch_index] = raw_mass.get(source.batch_index, 0.0) + mass
        phase_counts[source.batch_index] = phase_counts.get(source.batch_index, 0) + 1
        source_weight_sha256.setdefault(source.batch_index, []).append(
            _tensor_sha256(endpoint_weights)
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
        "schema": "neural-forge-lfm2-semantic-trajectory-row-balance-v2",
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
                "source_all_endpoint_row_weight_sha256": (
                    source_weight_sha256[batch_index]
                ),
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


def _dense_axis_coverage(value: Any, *, role: str) -> Mapping[str, Any]:
    """Measure the same per-axis structural coverage used by the layer stack."""

    torch = _torch()
    tensor = _finite_cpu_tensor(
        value,
        f"{role} structural candidate",
        ndim=2,
        dtype=torch.bfloat16,
    )
    nonzero = tensor.ne(0)
    output_units, input_units = map(int, tensor.shape)
    covered_outputs = int(nonzero.any(dim=1).sum().item())
    covered_inputs = int(nonzero.any(dim=0).sum().item())
    nonzero_elements = int(nonzero.sum().item())
    body = {
        "schema": "neural-forge-dense-weight-structural-coverage-v1",
        "parameter_name": role,
        "ordinary_parameter_required": False,
        "ordinary_parameter": False,
        "shape": [output_units, input_units],
        "dtype": "bfloat16",
        "tensor_sha256": _tensor_sha256(tensor),
        "element_count": int(tensor.numel()),
        "nonzero_element_count": nonzero_elements,
        "nonzero_element_fraction": nonzero_elements / int(tensor.numel()),
        "finite": True,
        "output_structural_unit_count": output_units,
        "covered_output_structural_unit_count": covered_outputs,
        "input_structural_unit_count": input_units,
        "covered_input_structural_unit_count": covered_inputs,
        "all_output_structural_units_covered": covered_outputs == output_units,
        "all_input_structural_units_covered": covered_inputs == input_units,
        "structurally_nonzero": bool(
            nonzero_elements > 0
            and covered_outputs == output_units
            and covered_inputs == input_units
        ),
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _repair_dense_axis_coverage(
    value: Any,
    *,
    role: str,
    fit_inputs: Any | None = None,
    fit_targets: Any | None = None,
    fit_sqrt_weights: Any | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Complete uncovered axes from FIT ridge and its exact design nullspace."""

    torch = _torch()
    candidate = _finite_cpu_tensor(
        value,
        f"{role} structural candidate",
        ndim=2,
        dtype=torch.bfloat16,
    )
    before = _dense_axis_coverage(candidate, role=role)
    if before["structurally_nonzero"] is True:
        body = {
            "schema": "neural-forge-lfm2-attention-fit-axis-repair-v1",
            "role": role,
            "coverage_before": dict(before),
            "coverage_after_fit_ridge": dict(before),
            "coverage_after": dict(before),
            "repaired_element_count": 0,
            "repair_mask_sha256": canonical_sha256([]),
            "fit_ridge": None,
            "fit_nullspace_completion": None,
            "existing_nonzero_entries_overwritten": 0,
            "fit_evidence_only": True,
            "heldout_evidence_used": False,
            "randomness_used": False,
            "synthetic_epsilon_scaffolding_used": False,
            "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        }
        return candidate, MappingProxyType(
            {**body, "repair_sha256": canonical_sha256(body)}
        )
    if fit_inputs is None or fit_targets is None or fit_sqrt_weights is None:
        raise Lfm2SemanticAttentionRefoldError(
            f"{role} has uncovered axes and no identifiable FIT linear system "
            f"(outputs {before['covered_output_structural_unit_count']}/"
            f"{before['output_structural_unit_count']}, inputs "
            f"{before['covered_input_structural_unit_count']}/"
            f"{before['input_structural_unit_count']})."
        )
    alternate, ridge_receipt = _cold_ridge_projection(
        fit_inputs,
        fit_targets,
        fit_sqrt_weights,
        role=f"{role}-axis-completion",
        normalize_element_rms=False,
    )
    if tuple(alternate.shape) != tuple(candidate.shape):
        raise Lfm2SemanticAttentionRefoldError(
            f"{role} FIT axis-completion ridge changed the target topology."
        )
    repaired = candidate.clone()
    original_nonzero = repaired.ne(0)
    repair_mask = torch.zeros_like(original_nonzero)
    uncovered_outputs = ~original_nonzero.any(dim=1)
    if bool(uncovered_outputs.any().item()):
        output_mask = uncovered_outputs.unsqueeze(1).expand_as(repair_mask)
        repair_mask |= output_mask & alternate.ne(0)
    staged_nonzero = original_nonzero | repair_mask
    uncovered_inputs = ~staged_nonzero.any(dim=0)
    if bool(uncovered_inputs.any().item()):
        input_mask = uncovered_inputs.unsqueeze(0).expand_as(repair_mask)
        repair_mask |= input_mask & alternate.ne(0)
    repaired[repair_mask] = alternate[repair_mask]
    after_ridge = _dense_axis_coverage(repaired, role=role)
    if after_ridge["all_output_structural_units_covered"] is not True:
        raise Lfm2SemanticAttentionRefoldError(
            f"{role} uncovered output axes are not identifiable from the FIT ridge "
            f"system (before outputs "
            f"{before['covered_output_structural_unit_count']}/"
            f"{before['output_structural_unit_count']}; alternate outputs "
            f"{after_ridge['covered_output_structural_unit_count']}/"
            f"{after_ridge['output_structural_unit_count']}; before inputs "
            f"{before['covered_input_structural_unit_count']}/"
            f"{before['input_structural_unit_count']}; alternate inputs "
            f"{after_ridge['covered_input_structural_unit_count']}/"
            f"{after_ridge['input_structural_unit_count']})."
        )
    nullspace_report = None
    if after_ridge["all_input_structural_units_covered"] is not True:
        inputs64 = fit_inputs.detach().to(device="cpu", dtype=torch.float64)
        roots64 = (
            fit_sqrt_weights.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1, 1)
        )
        if (
            inputs64.ndim != 2
            or tuple(roots64.shape) != (int(inputs64.shape[0]), 1)
            or int(inputs64.shape[1]) != int(repaired.shape[1])
            or bool((roots64 <= 0.0).any().item())
        ):
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} FIT nullspace completion has incompatible design rows."
            )
        weighted_design = inputs64 * roots64
        try:
            _left, singular, right_h = torch.linalg.svd(
                weighted_design,
                full_matrices=True,
            )
        except RuntimeError as exc:
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} FIT nullspace SVD failed."
            ) from exc
        maximum_singular = (
            0.0 if int(singular.numel()) == 0 else float(singular.max().item())
        )
        rank_threshold = (
            maximum_singular
            * max(map(int, weighted_design.shape))
            * torch.finfo(torch.float64).eps
        )
        rank = int((singular > rank_threshold).sum().item())
        null_basis = right_h[rank:]
        nullity = int(null_basis.shape[0])
        missing_inputs = torch.nonzero(
            ~repaired.ne(0).any(dim=0),
            as_tuple=False,
        ).reshape(-1)
        if nullity == 0 or int(missing_inputs.numel()) == 0:
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} still has uncovered input axes but its FIT design has "
                f"rank {rank}/{int(weighted_design.shape[1])} and nullity {nullity}."
            )
        null_at_missing = null_basis.index_select(1, missing_inputs)
        missing_gram = null_at_missing.T @ null_at_missing
        missing_eigenvalues = torch.linalg.eigvalsh(missing_gram)
        maximum_missing_eigenvalue = float(missing_eigenvalues.max().item())
        minimum_missing_eigenvalue = float(missing_eigenvalues.min().item())
        invertibility_threshold = (
            maximum_missing_eigenvalue
            * max(1, int(missing_inputs.numel()))
            * torch.finfo(torch.float64).eps
        )
        if minimum_missing_eigenvalue <= invertibility_threshold:
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} FIT nullspace cannot independently identify all "
                f"{int(missing_inputs.numel())} uncovered input axes (design rank "
                f"{rank}/{int(weighted_design.shape[1])}, nullity {nullity})."
            )
        missing_to_all = null_at_missing.T @ null_basis
        repaired64 = repaired.to(torch.float64)
        row_rms = torch.sqrt(repaired64.square().mean(dim=1))
        largest_indices = repaired64.abs().argmax(dim=1, keepdim=True)
        row_sign = torch.sign(repaired64.gather(1, largest_indices)[:, 0])
        if bool((row_rms <= 0.0).any().item()) or bool((row_sign == 0.0).any().item()):
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} FIT ridge has no evidence-derived output scale for "
                "nullspace completion."
            )
        leverage = torch.sqrt(torch.diagonal(missing_gram))
        desired_missing = (
            row_sign.reshape(-1, 1) * row_rms.reshape(-1, 1) * leverage.reshape(1, -1)
        )
        try:
            completion_map = torch.linalg.solve(
                missing_gram,
                missing_to_all,
            )
        except RuntimeError as exc:
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} FIT nullspace constrained solve failed."
            ) from exc
        delta = desired_missing @ completion_map
        completed64 = repaired64 + delta
        fp64_prediction_change = weighted_design @ delta.T
        constraint_error = float(
            (delta.index_select(1, missing_inputs) - desired_missing).abs().max().item()
        )
        fp64_prediction_change_maximum = float(
            fp64_prediction_change.abs().max().item()
        )
        nullspace_tolerance = (
            torch.finfo(torch.float64).eps
            * max(
                int(weighted_design.shape[0]),
                int(weighted_design.shape[1]),
                int(delta.shape[0]),
            )
            * max(1.0, float(torch.linalg.matrix_norm(weighted_design).item()))
            * max(1.0, float(torch.linalg.matrix_norm(delta).item()))
            * 64.0
        )
        constraint_tolerance = (
            torch.finfo(torch.float64).eps
            * max(1, int(missing_inputs.numel()))
            * max(1.0, float(desired_missing.abs().max().item()))
            * 64.0
        )
        if (
            fp64_prediction_change_maximum > nullspace_tolerance
            or constraint_error > constraint_tolerance
        ):
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} constrained completion did not remain in the numerical "
                "FIT nullspace."
            )
        completed = completed64.to(torch.bfloat16).contiguous()
        bf16_prediction_change = (
            weighted_design @ (completed.to(torch.float64) - repaired64).T
        )
        if not bool(torch.isfinite(completed).all().item()):
            raise Lfm2SemanticAttentionRefoldError(
                f"{role} FIT nullspace completion became nonfinite in BF16."
            )
        repaired = completed
        null_body = {
            "schema": "neural-forge-lfm2-attention-fit-nullspace-axis-completion-v1",
            "weighted_design_sha256": _tensor_sha256(weighted_design),
            "singular_values_sha256": _tensor_sha256(singular),
            "design_row_count": int(weighted_design.shape[0]),
            "design_width": int(weighted_design.shape[1]),
            "numerical_rank": rank,
            "nullity": nullity,
            "rank_threshold": rank_threshold,
            "missing_input_axis_count": int(missing_inputs.numel()),
            "missing_input_axis_indices": [
                int(value) for value in missing_inputs.tolist()
            ],
            "missing_input_axis_indices_sha256": canonical_sha256(
                [int(value) for value in missing_inputs.tolist()]
            ),
            "missing_null_gram_sha256": _tensor_sha256(missing_gram),
            "minimum_missing_null_eigenvalue": minimum_missing_eigenvalue,
            "maximum_missing_null_eigenvalue": maximum_missing_eigenvalue,
            "missing_null_invertibility_threshold": invertibility_threshold,
            "missing_null_condition_number": (
                maximum_missing_eigenvalue / minimum_missing_eigenvalue
            ),
            "source_scale_rule": (
                "fit_ridge_output_row_rms_times_missing_axis_null_leverage_with_"
                "largest_coefficient_sign"
            ),
            "desired_missing_values_sha256": _tensor_sha256(desired_missing),
            "minimum_norm_completion_map_sha256": _tensor_sha256(completion_map),
            "delta_sha256": _tensor_sha256(delta),
            "missing_axis_constraint_maximum_absolute_error": constraint_error,
            "fp64_weighted_fit_prediction_maximum_absolute_change": (
                fp64_prediction_change_maximum
            ),
            "fp64_nullspace_tolerance": nullspace_tolerance,
            "missing_axis_constraint_tolerance": constraint_tolerance,
            "bf16_weighted_fit_prediction_maximum_absolute_change": float(
                bf16_prediction_change.abs().max().item()
            ),
            "completion_is_minimum_frobenius_norm_under_axis_constraints": True,
            "completion_lies_in_numerical_fit_design_nullspace": True,
            "unidentified_function_claimed": False,
            "fit_evidence_only": True,
            "heldout_evidence_used": False,
            "randomness_used": False,
            "synthetic_epsilon_scaffolding_used": False,
            "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        }
        nullspace_report = MappingProxyType(
            {**null_body, "completion_sha256": canonical_sha256(null_body)}
        )
    after = _dense_axis_coverage(repaired, role=role)
    if after["structurally_nonzero"] is not True:
        raise Lfm2SemanticAttentionRefoldError(
            f"{role} FIT completion remains structurally incomplete (before "
            f"outputs {before['covered_output_structural_unit_count']}/"
            f"{before['output_structural_unit_count']}, inputs "
            f"{before['covered_input_structural_unit_count']}/"
            f"{before['input_structural_unit_count']}; ridge outputs "
            f"{after_ridge['covered_output_structural_unit_count']}/"
            f"{after_ridge['output_structural_unit_count']}, inputs "
            f"{after_ridge['covered_input_structural_unit_count']}/"
            f"{after_ridge['input_structural_unit_count']}; final outputs "
            f"{after['covered_output_structural_unit_count']}/"
            f"{after['output_structural_unit_count']}, inputs "
            f"{after['covered_input_structural_unit_count']}/"
            f"{after['input_structural_unit_count']})."
        )
    final_repair_mask = repaired.ne(candidate)
    body = {
        "schema": "neural-forge-lfm2-attention-fit-axis-repair-v1",
        "role": role,
        "coverage_before": dict(before),
        "coverage_after_fit_ridge": dict(after_ridge),
        "coverage_after": dict(after),
        "repaired_element_count": int(final_repair_mask.sum().item()),
        "repair_mask_sha256": _tensor_sha256(final_repair_mask),
        "fit_ridge": dict(ridge_receipt),
        "fit_nullspace_completion": (
            None if nullspace_report is None else dict(nullspace_report)
        ),
        "existing_nonzero_entries_overwritten": int(
            (final_repair_mask & original_nonzero).sum().item()
        ),
        "fit_evidence_only": True,
        "heldout_evidence_used": False,
        "randomness_used": False,
        "synthetic_epsilon_scaffolding_used": False,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return repaired, MappingProxyType({**body, "repair_sha256": canonical_sha256(body)})


def _copy_seed(
    seed: Lfm2EndpointQkvoSeed,
    *,
    query_weight: Any | None = None,
    key_weight: Any | None = None,
    value_weight: Any | None = None,
    output_weight: Any | None = None,
    query_norm_weight: Any | None = None,
    key_norm_weight: Any | None = None,
) -> Lfm2EndpointQkvoSeed:
    return Lfm2EndpointQkvoSeed(
        query_weight=seed.query_weight if query_weight is None else query_weight,
        key_weight=seed.key_weight if key_weight is None else key_weight,
        value_weight=seed.value_weight if value_weight is None else value_weight,
        output_weight=seed.output_weight if output_weight is None else output_weight,
        query_norm_weight=(
            seed.query_norm_weight if query_norm_weight is None else query_norm_weight
        ),
        key_norm_weight=(
            seed.key_norm_weight if key_norm_weight is None else key_norm_weight
        ),
        norm_epsilon=seed.norm_epsilon,
        source_inventory=seed.source_inventory,
        source_artifact_sha256=seed.source_artifact_sha256,
        target_template_spec_sha256=seed.target_template_spec_sha256,
    )


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
    """Exactly invert finite RMSNorm/RoPE rows; the RMS=1 boundary is invalid."""

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
    normalized = unrotated / gain.reshape(1, 1, 1, -1)
    normalized_energy = normalized.square().mean(dim=-1, keepdim=True)
    if bool((normalized_energy >= 1.0).any().item()):
        maximum = float(normalized_energy.max().item())
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} normalized target has RMS^2={maximum:.9g}; "
            "no finite RMSNorm preimage exists at or beyond one."
        )
    result = normalized * torch.sqrt(
        torch.tensor(float(norm_epsilon), dtype=torch.float64)
        / (1.0 - normalized_energy)
    )
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} preimage produced nonfinite rows."
        )
    replay_normalized = result * torch.rsqrt(
        result.square().mean(dim=-1, keepdim=True) + float(norm_epsilon)
    )
    replay_runtime = _rope_forward_rows(
        replay_normalized * gain.reshape(1, 1, 1, -1),
        cos.unsqueeze(1),
        sin.unsqueeze(1),
    )
    tolerance = torch.finfo(torch.float64).eps * max(32, int(runtime.shape[-1]))
    if not bool(
        torch.allclose(replay_runtime, runtime, rtol=tolerance, atol=tolerance)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} exact RMSNorm/RoPE replay failed."
        )
    # Anchors are not a radius source.  They are accepted only as a topology
    # authentication input so zero anchors can never create the old 1/sqrt(2)
    # amplitude bug.
    del anchors
    return result.contiguous()


def _fit_pre_rope_dense_gauge(
    seed: Lfm2EndpointQkvoSeed,
    cases: Sequence[Lfm2EndpointQkResidualCase],
) -> tuple[Any, Mapping[str, Any]]:
    """Seal the native RoPE basis; arbitrary dense gauges do not commute with RoPE."""

    torch = _torch()
    head_dim = seed.head_dim
    gauge = torch.eye(head_dim, dtype=torch.float64)
    body = {
        "schema": "neural-forge-lfm2-attention-native-pre-rope-basis-v1",
        "fit_case_evidence_sha256": [case.evidence_sha256 for case in cases],
        "head_dim": head_dim,
        "gauge_sha256": _tensor_sha256(gauge),
        "basis": "native_identity",
        "arbitrary_dense_householder_used": False,
        "reason": "dense_pre_rope_maps_do_not_commute_with_rotary_coordinates",
        "fit_evidence_only": True,
        "heldout_evidence_used": False,
        "randomness_used": False,
        "synthetic_epsilon_scaffolding_used": False,
        "gradients_used": False,
        "training_steps": 0,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return gauge, MappingProxyType(
        {**body, "gauge_receipt_sha256": canonical_sha256(body)}
    )


def _rope_forward_rows(value: Any, cosine: Any, sine: Any) -> Any:
    return value * cosine + _rotate_half(value) * sine


def _rope_transpose_rows(value: Any, cosine: Any, sine: Any) -> Any:
    return value * cosine - _rotate_half(value) * sine


def _cold_route_runtime_targets(
    case: Lfm2EndpointQkResidualCase,
    seed: Lfm2EndpointQkvoSeed,
    *,
    pre_rope_gauge: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Build realizable raw-first Q/K targets and fit masked score residuals."""

    torch = _torch()
    probabilities = case.target_probability_barycenter.to(torch.float64)
    mask = case.target_causal_mask
    batch, query_heads, query_length, key_length = map(int, probabilities.shape)
    head_dim = seed.head_dim
    key_value_heads = seed.key_value_heads
    queries_per_key_value = seed.queries_per_key_value
    gauge = pre_rope_gauge.detach().to(device="cpu", dtype=torch.float64)
    if (
        query_heads != seed.query_heads
        or query_heads != key_value_heads * queries_per_key_value
        or tuple(gauge.shape) != (head_dim, head_dim)
        or not torch.equal(gauge, torch.eye(head_dim, dtype=torch.float64))
        or query_length <= 0
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Cold-start Q/K planting requires all explicit endpoint rows in the "
            "native target GQA/RoPE basis."
        )
    query_gain = seed.query_norm_weight.to(torch.float64)
    key_gain = seed.key_norm_weight.to(torch.float64)
    if bool((query_gain.abs() <= torch.finfo(torch.float64).eps).any().item()) or bool(
        (key_gain.abs() <= torch.finfo(torch.float64).eps).any().item()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Parent-derived Q/K RMSNorm gains contain a zero coordinate."
        )
    query_weight_prior = (
        seed.query_weight.to(torch.float64)
        if case.parent_query_weight_prior is None
        else case.parent_query_weight_prior
    )
    key_weight_prior = (
        seed.key_weight.to(torch.float64)
        if case.parent_key_weight_prior is None
        else case.parent_key_weight_prior
    )
    folded_query_pre_rope = case.folded_parent_query_pre_rope
    folded_key_pre_rope = case.folded_parent_key_pre_rope
    if (
        folded_query_pre_rope is None
        or folded_key_pre_rope is None
        or tuple(folded_query_pre_rope.shape)
        != (batch, query_heads, query_length, head_dim)
        or tuple(folded_key_pre_rope.shape)
        != (batch, key_value_heads, key_length, head_dim)
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Cold-start Q/K planting requires authenticated query and KV "
            "parent activation priors on every semantic endpoint."
        )
    folded_query_pre_rope = folded_query_pre_rope.to(torch.float64)
    folded_key_pre_rope = folded_key_pre_rope.to(torch.float64)
    query_weight_raw_anchor = (
        (case.target_query_inputs.to(torch.float64) @ query_weight_prior.T)
        .reshape(batch, query_length, query_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    key_weight_raw_anchor = (
        (case.target_key_value_inputs.to(torch.float64) @ key_weight_prior.T)
        .reshape(batch, key_length, key_value_heads, head_dim)
        .permute(0, 2, 1, 3)
    )

    def forward_raw(raw: Any, gain: Any, cosine: Any, sine: Any) -> tuple[Any, Any]:
        normalized = raw * torch.rsqrt(
            raw.square().mean(dim=-1, keepdim=True) + float(seed.norm_epsilon)
        )
        pre_rope = normalized * gain.reshape(1, 1, 1, -1)
        runtime = _rope_forward_rows(
            pre_rope,
            cosine.to(torch.float64).unsqueeze(1),
            sine.to(torch.float64).unsqueeze(1),
        )
        return runtime, normalized

    query_weight_runtime_anchor, query_weight_normalized_anchor = forward_raw(
        query_weight_raw_anchor,
        query_gain,
        case.query_rope_cos,
        case.query_rope_sin,
    )
    key_weight_runtime_anchor, key_weight_normalized_anchor = forward_raw(
        key_weight_raw_anchor,
        key_gain,
        case.key_rope_cos,
        case.key_rope_sin,
    )
    feasibility_energy_limit = 1.0 - 1e-8

    def finite_activation_residual(
        prior_normalized: Any,
        desired_normalized: Any,
    ) -> tuple[Any, float, bool]:
        desired_energy = float(desired_normalized.square().mean().item())
        if desired_energy < 1.0:
            return desired_normalized, 1.0, False
        prior_energy = float(prior_normalized.square().mean().item())
        local_energy_limit = max(
            feasibility_energy_limit,
            0.5 * (1.0 + prior_energy),
        )
        if not prior_energy < local_energy_limit < 1.0:
            return prior_normalized, 0.0, True
        direction = desired_normalized - prior_normalized
        a = float(direction.square().mean().item())
        b = float((2.0 * prior_normalized * direction).mean().item())
        c = float(prior_normalized.square().mean().item() - local_energy_limit)
        discriminant = max(0.0, b * b - 4.0 * a * c)
        fraction = (
            0.0
            if a <= torch.finfo(torch.float64).tiny
            else min(1.0, max(0.0, (-b + math.sqrt(discriminant)) / (2.0 * a)))
        )
        return prior_normalized + fraction * direction, fraction, True

    def activation_target_raw(
        desired_pre_rope: Any,
        prior_normalized: Any,
        gain: Any,
    ) -> tuple[Any, Any, int, float]:
        realized = torch.empty_like(desired_pre_rope)
        infeasible = 0
        minimum_fraction = 1.0
        for index in itertools.product(
            *(range(int(size)) for size in realized.shape[:-1])
        ):
            desired_normalized = desired_pre_rope[index] / gain
            row, fraction, was_infeasible = finite_activation_residual(
                prior_normalized[index],
                desired_normalized,
            )
            realized[index] = row
            infeasible += int(was_infeasible)
            minimum_fraction = min(minimum_fraction, fraction)
        energy = realized.square().mean(dim=-1, keepdim=True)
        raw = realized * torch.sqrt(
            torch.tensor(float(seed.norm_epsilon), dtype=torch.float64) / (1.0 - energy)
        )
        return raw, realized, infeasible, minimum_fraction

    (
        query_raw,
        query_activation_normalized,
        query_activation_infeasible_count,
        query_activation_minimum_fraction,
    ) = activation_target_raw(
        folded_query_pre_rope,
        query_weight_normalized_anchor,
        query_gain,
    )
    (
        key_raw,
        key_activation_normalized,
        key_activation_infeasible_count,
        key_activation_minimum_fraction,
    ) = activation_target_raw(
        folded_key_pre_rope,
        key_weight_normalized_anchor,
        key_gain,
    )
    query_runtime, _ = forward_raw(
        query_raw,
        query_gain,
        case.query_rope_cos,
        case.query_rope_sin,
    )
    key_runtime_anchor, _ = forward_raw(
        key_raw,
        key_gain,
        case.key_rope_cos,
        case.key_rope_sin,
    )
    key_runtime = key_runtime_anchor.clone()
    positive_mask = mask & (probabilities > 0.0)
    nonpositive_visible_probability_count = int((mask & ~positive_mask).sum().item())
    safe = torch.where(
        positive_mask,
        probabilities,
        torch.ones_like(probabilities),
    )
    logits = torch.where(positive_mask, torch.log(safe), torch.zeros_like(safe))
    counts = positive_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    logits = torch.where(
        positive_mask,
        logits - (logits * positive_mask).sum(dim=-1, keepdim=True) / counts,
        torch.zeros_like(logits),
    )
    root_head_dim = math.sqrt(head_dim)
    infeasible_normalized_constraint_count = 0
    unconstrained_key_count = 0
    maximum_constraint_error = 0.0
    minimum_feasible_residual_fraction = 1.0
    for batch_index in range(batch):
        for group_index in range(key_value_heads):
            first_head = group_index * queries_per_key_value
            heads = tuple(range(first_head, first_head + queries_per_key_value))
            for key_index in range(key_length):
                design_rows = []
                target_rows = []
                row_weights = []
                for head_index in heads:
                    for query_index in range(query_length):
                        if bool(
                            positive_mask[
                                batch_index, head_index, query_index, key_index
                            ]
                        ):
                            design_rows.append(
                                query_runtime[batch_index, head_index, query_index]
                            )
                            target_rows.append(
                                logits[batch_index, head_index, query_index, key_index]
                                * root_head_dim
                            )
                            row_weights.append(
                                case.query_row_sqrt_weights[
                                    batch_index,
                                    query_index,
                                ]
                            )
                if not design_rows:
                    unconstrained_key_count += 1
                    continue
                design = torch.stack(design_rows, dim=0)
                target = torch.stack(target_rows).to(torch.float64)
                roots = torch.stack(row_weights).to(torch.float64)
                prior_post = key_runtime_anchor[batch_index, group_index, key_index]
                weighted_design = design * roots.reshape(-1, 1)
                weighted_residual = (target - design @ prior_post) * roots
                correction = torch.linalg.lstsq(
                    weighted_design,
                    weighted_residual,
                    driver="gelsd",
                ).solution
                desired_post = prior_post + correction
                key_cos = case.key_rope_cos[batch_index, key_index].to(torch.float64)
                key_sin = case.key_rope_sin[batch_index, key_index].to(torch.float64)
                radius = key_cos.square() + key_sin.square()
                desired_pre = (
                    _rope_transpose_rows(
                        desired_post,
                        key_cos,
                        key_sin,
                    )
                    / radius
                )
                desired_normalized = desired_pre / key_gain
                realized_normalized = desired_normalized
                residual_fraction = 1.0
                if float(desired_normalized.square().mean().item()) >= 1.0:
                    infeasible_normalized_constraint_count += 1
                    prior_normalized = key_activation_normalized[
                        batch_index,
                        group_index,
                        key_index,
                    ]
                    (
                        realized_normalized,
                        residual_fraction,
                        _was_infeasible,
                    ) = finite_activation_residual(
                        prior_normalized,
                        desired_normalized,
                    )
                    minimum_feasible_residual_fraction = min(
                        minimum_feasible_residual_fraction,
                        residual_fraction,
                    )
                realized_energy = realized_normalized.square().mean()
                realized_raw = realized_normalized * torch.sqrt(
                    torch.tensor(float(seed.norm_epsilon), dtype=torch.float64)
                    / (1.0 - realized_energy)
                )
                realized_pre = realized_normalized * key_gain
                realized_post = _rope_forward_rows(
                    realized_pre,
                    key_cos,
                    key_sin,
                )
                key_raw[batch_index, group_index, key_index] = realized_raw
                key_runtime[batch_index, group_index, key_index] = realized_post
                maximum_constraint_error = max(
                    maximum_constraint_error,
                    float((design @ realized_post - target).abs().max().item()),
                )
    replayed_query_runtime, _ = forward_raw(
        query_raw,
        query_gain,
        case.query_rope_cos,
        case.query_rope_sin,
    )
    replayed_key_runtime, _ = forward_raw(
        key_raw,
        key_gain,
        case.key_rope_cos,
        case.key_rope_sin,
    )
    replay_tolerance = torch.finfo(torch.float64).eps * max(64, head_dim * 4)
    query_runtime_replay_error = float(
        (replayed_query_runtime - query_runtime).abs().max().item()
    )
    key_runtime_replay_error = float(
        (replayed_key_runtime - key_runtime).abs().max().item()
    )
    if (
        query_runtime_replay_error > replay_tolerance
        or key_runtime_replay_error > replay_tolerance
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Cold-start raw Q/K targets failed exact target RMSNorm/RoPE replay."
        )
    repeated_key = key_runtime.repeat_interleave(
        queries_per_key_value,
        dim=1,
    )
    realized_scores = (
        torch.matmul(
            query_runtime,
            repeated_key.transpose(2, 3),
        )
        / root_head_dim
    )
    realized_probabilities = torch.softmax(
        realized_scores.masked_fill(~mask, -torch.inf),
        dim=-1,
        dtype=torch.float64,
    )
    realized_probabilities = torch.where(
        mask,
        realized_probabilities,
        torch.zeros_like(realized_probabilities),
    )
    probability_error = float(
        (realized_probabilities - probabilities).abs().max().item()
    )
    body = {
        "schema": "neural-forge-lfm2-semantic-qk-cold-runtime-target-v3",
        "case_evidence_sha256": case.evidence_sha256,
        "target_probability_sha256": _tensor_sha256(probabilities),
        "target_mask_sha256": _tensor_sha256(mask),
        "query_runtime_sha256": _tensor_sha256(query_runtime),
        "key_runtime_sha256": _tensor_sha256(key_runtime),
        "all_semantically_exact_endpoint_rows_consumed": True,
        "query_length": query_length,
        "explicit_causal_mask_consumed": True,
        "centered_log_probability_factorization": True,
        "factorization": (
            "cached_parent_pre_rope_qk_prior_plus_relative_rope_score_residual"
        ),
        "pre_rope_gauge_sha256": _tensor_sha256(gauge),
        "arbitrary_dense_pre_rope_gauge_used": False,
        "parent_query_weight_prior_used": case.parent_query_weight_prior is not None,
        "parent_key_weight_prior_used": case.parent_key_weight_prior is not None,
        "folded_parent_query_activation_prior_consumed": True,
        "folded_parent_key_value_activation_prior_consumed": True,
        "query_weight_runtime_anchor_sha256": _tensor_sha256(
            query_weight_runtime_anchor
        ),
        "key_weight_runtime_anchor_sha256": _tensor_sha256(key_weight_runtime_anchor),
        "query_activation_residual_from_weight_prior_sha256": _tensor_sha256(
            folded_query_pre_rope
            - query_weight_normalized_anchor * query_gain.reshape(1, 1, 1, -1)
        ),
        "key_activation_residual_from_weight_prior_sha256": _tensor_sha256(
            folded_key_pre_rope
            - key_weight_normalized_anchor * key_gain.reshape(1, 1, 1, -1)
        ),
        "query_activation_prior_infeasible_row_count": (
            query_activation_infeasible_count
        ),
        "key_activation_prior_infeasible_row_count": (key_activation_infeasible_count),
        "query_activation_prior_minimum_residual_fraction": (
            query_activation_minimum_fraction
        ),
        "key_activation_prior_minimum_residual_fraction": (
            key_activation_minimum_fraction
        ),
        "causal_logit_residual_applied_after_activation_prior": True,
        "query_raw_target_sha256": _tensor_sha256(query_raw),
        "key_raw_target_sha256": _tensor_sha256(key_raw),
        "raw_targets_forwarded_through_actual_rmsnorm_and_rope": True,
        "exact_rmsnorm_rope_replay_claimed": True,
        "query_raw_to_runtime_maximum_absolute_replay_error": (
            query_runtime_replay_error
        ),
        "key_raw_to_runtime_maximum_absolute_replay_error": (key_runtime_replay_error),
        "raw_to_runtime_replay_tolerance": replay_tolerance,
        "score_constraints_exactly_feasible": (
            infeasible_normalized_constraint_count == 0
            and nonpositive_visible_probability_count == 0
        ),
        "infeasible_normalized_constraint_count": (
            infeasible_normalized_constraint_count
        ),
        "nonpositive_visible_probability_count": (
            nonpositive_visible_probability_count
        ),
        "unconstrained_key_count": unconstrained_key_count,
        "infeasible_constraint_resolution": (
            "closest_parent_anchored_residual_on_finite_rmsnorm_interior"
        ),
        "minimum_feasible_residual_fraction": minimum_feasible_residual_fraction,
        "normalized_energy_limit": feasibility_energy_limit,
        "maximum_score_constraint_absolute_error": maximum_constraint_error,
        "maximum_target_probability_absolute_error": probability_error,
        "gqa_shared_key_residual_solve": True,
        "silent_logit_clipping_used": False,
        "probability_preservation_claimed": probability_error <= 1e-12,
        "fit_evidence_only": True,
        "heldout_evidence_used": False,
        "randomness_used": False,
        "gradients_used": False,
        "training_steps": 0,
    }
    return (
        query_runtime,
        key_runtime,
        MappingProxyType({**body, "target_sha256": canonical_sha256(body)}),
    )


def _cold_ridge_projection(
    inputs: Any,
    targets: Any,
    sqrt_weights: Any,
    *,
    role: str,
    ridge: float = 1e-4,
    normalize_element_rms: bool = True,
    prior: Any | None = None,
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
    prior64 = None
    residual_target = y
    if prior is not None:
        prior64 = prior.detach().to(device="cpu", dtype=torch.float64)
        if tuple(prior64.shape) != (int(y.shape[1]), int(x.shape[1])):
            raise Lfm2SemanticAttentionRefoldError(
                f"Cold-start {role} prior changed target topology."
            )
        if normalize_element_rms:
            raise Lfm2SemanticAttentionRefoldError(
                f"Cold-start {role} cannot globally rescale a parent-derived prior."
            )
        residual_target = y - x @ prior64.T
    weighted_x = x * root
    weighted_y = residual_target * root
    rows, features = map(int, weighted_x.shape)
    try:
        with torch.inference_mode():
            if rows <= features:
                gram = weighted_x @ weighted_x.T
                scale = float(gram.diagonal().mean().clamp_min(1.0).item())
                effective_ridge = float(ridge) * scale
                dual = torch.linalg.solve(
                    gram + effective_ridge * torch.eye(rows, dtype=torch.float64),
                    weighted_y,
                )
                correction = dual.T @ weighted_x
                solve_form = "deterministic_dual_ridge"
            else:
                gram = weighted_x.T @ weighted_x
                scale = float(gram.diagonal().mean().clamp_min(1.0).item())
                effective_ridge = float(ridge) * scale
                correction = torch.linalg.solve(
                    gram + effective_ridge * torch.eye(features, dtype=torch.float64),
                    weighted_x.T @ weighted_y,
                ).T
                solve_form = "deterministic_primal_ridge"
    except RuntimeError as exc:
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} ridge solve failed."
        ) from exc
    weight = correction if prior64 is None else prior64 + correction
    if not bool(torch.isfinite(weight).all().item()) or _exact_zero(weight):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} ridge solve produced no finite plant."
        )
    target_rms = 1.0 / math.sqrt(features)
    if normalize_element_rms:
        observed_rms = torch.sqrt(weight.square().mean())
        weight = weight * (target_rms / observed_rms)
    published = weight.to(torch.bfloat16).contiguous()
    if _exact_zero(published) or not bool(torch.isfinite(published).all().item()):
        raise Lfm2SemanticAttentionRefoldError(
            f"Cold-start {role} plant vanished during BF16 publication."
        )
    _left, singular, right_h = torch.linalg.svd(weighted_x, full_matrices=True)
    maximum_singular = (
        0.0 if int(singular.numel()) == 0 else float(singular.max().item())
    )
    rank_threshold = (
        maximum_singular
        * max(map(int, weighted_x.shape))
        * torch.finfo(torch.float64).eps
    )
    design_rank = int((singular > rank_threshold).sum().item())
    null_basis = right_h[design_rank:]
    nullspace_delta = (
        torch.zeros((), dtype=torch.float64)
        if int(null_basis.shape[0]) == 0
        else (correction @ null_basis.T).abs().max()
    )
    published_prior = (
        torch.zeros_like(published, dtype=torch.bfloat16)
        if prior64 is None
        else prior64.to(torch.bfloat16)
    )
    published_correction = published.to(torch.float64) - published_prior.to(
        torch.float64
    )
    published_nullspace_delta = (
        torch.zeros((), dtype=torch.float64)
        if int(null_basis.shape[0]) == 0
        else (published_correction @ null_basis.T).abs().max()
    )
    published_from_fp64_prior_nullspace_delta = (
        None
        if prior64 is None
        else (
            torch.zeros((), dtype=torch.float64)
            if int(null_basis.shape[0]) == 0
            else ((published.to(torch.float64) - prior64) @ null_basis.T).abs().max()
        )
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
        "residual_target_sha256": _tensor_sha256(residual_target),
        "sqrt_weight_sha256": _tensor_sha256(root),
        "published_weight_sha256": _tensor_sha256(published),
        "published_nonzero_elements": int(torch.count_nonzero(published).item()),
        "parent_derived_prior_used": prior64 is not None,
        "prior_sha256": None if prior64 is None else _tensor_sha256(prior64),
        "correction_sha256": _tensor_sha256(correction),
        "fit_design_rank": design_rank,
        "fit_design_nullity": int(null_basis.shape[0]),
        "fit_design_rank_threshold": rank_threshold,
        "correction_action_on_fit_design_nullspace_maximum_absolute": float(
            nullspace_delta.item()
        ),
        "published_bfloat16_correction_action_on_fit_design_nullspace_maximum_absolute": float(
            published_nullspace_delta.item()
        ),
        "published_bfloat16_weight_minus_fp64_prior_action_on_fit_design_nullspace_maximum_absolute": (
            None
            if published_from_fp64_prior_nullspace_delta is None
            else float(published_from_fp64_prior_nullspace_delta.item())
        ),
        "parent_prior_preserved_in_fit_input_nullspace": prior64 is not None,
        "parent_prior_preserved_in_published_bfloat16_fit_input_nullspace": (
            prior64 is not None and float(published_nullspace_delta.item()) == 0.0
        ),
        "target_rms_normalization_applied": normalize_element_rms,
        "target_normalized_element_rms": (
            target_rms if normalize_element_rms else None
        ),
        "normalized_published_element_rms": (
            target_rms if normalize_element_rms else None
        ),
        "published_element_rms": float(
            torch.sqrt(published.to(torch.float64).square().mean()).item()
        ),
    }
    return published, MappingProxyType({**body, "ridge_sha256": canonical_sha256(body)})


def _fit_qk_target_geometry_receipt(
    query_targets: Any,
    key_targets: Any,
    *,
    seed: Lfm2EndpointQkvoSeed,
    parent_prior_case_count: int,
    fit_case_count: int,
) -> Mapping[str, Any]:
    """Seal structural coverage and contextual direction rank of FIT targets."""

    torch = _torch()

    def role_report(value: Any, heads: int) -> Mapping[str, Any]:
        rows = (
            value.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(
                int(value.shape[0]),
                heads,
                seed.head_dim,
            )
        )
        output_coverage = rows.ne(0.0).any(dim=0)
        ranks = []
        singular_hashes = []
        thresholds = []
        for head_index in range(heads):
            directions = rows[:, head_index]
            norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
            nonempty = norms[:, 0] > torch.finfo(torch.float64).tiny
            directions = directions[nonempty] / norms[nonempty]
            if int(directions.shape[0]) == 0:
                singular = torch.zeros(0, dtype=torch.float64)
                threshold = 0.0
                rank = 0
            else:
                singular = torch.linalg.svdvals(directions)
                maximum = float(singular.max().item())
                threshold = (
                    maximum
                    * max(int(directions.shape[0]), seed.head_dim)
                    * torch.finfo(torch.float64).eps
                )
                rank = int((singular > threshold).sum().item())
            ranks.append(rank)
            singular_hashes.append(_tensor_sha256(singular))
            thresholds.append(threshold)
        maximum_possible_rank = min(int(rows.shape[0]), seed.head_dim)
        return MappingProxyType(
            {
                "row_count": int(rows.shape[0]),
                "head_count": heads,
                "head_dim": seed.head_dim,
                "covered_output_axis_count": int(output_coverage.sum().item()),
                "output_axis_count": int(output_coverage.numel()),
                "all_output_axes_covered": bool(output_coverage.all().item()),
                "output_axis_coverage_sha256": _tensor_sha256(output_coverage),
                "per_head_contextual_direction_rank": ranks,
                "minimum_contextual_direction_rank": min(ranks),
                "maximum_contextual_direction_rank": max(ranks),
                "maximum_possible_contextual_direction_rank": maximum_possible_rank,
                "all_heads_span_available_fit_context_rank": all(
                    rank == maximum_possible_rank for rank in ranks
                ),
                "per_head_singular_values_sha256": singular_hashes,
                "per_head_rank_threshold": thresholds,
                "target_sha256": _tensor_sha256(rows),
            }
        )

    query = role_report(query_targets, seed.query_heads)
    key = role_report(key_targets, seed.key_value_heads)
    body = {
        "schema": "neural-forge-lfm2-attention-fit-qk-target-geometry-v1",
        "fit_case_count": fit_case_count,
        "parent_activation_prior_case_count": parent_prior_case_count,
        "all_fit_cases_parent_activation_bound": (
            parent_prior_case_count == fit_case_count
        ),
        "query": dict(query),
        "key": dict(key),
        "structural_coverage_is_not_contextual_function_proof": True,
        "unseen_attention_function_retention_claimed": False,
        "rank_is_diagnostic_not_a_publication_selection_signal": True,
        "fit_evidence_only": True,
        "heldout_evidence_used": False,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return MappingProxyType({**body, "geometry_sha256": canonical_sha256(body)})


def _qk_cold_fit_systems(
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
) -> Mapping[str, Any]:
    """Assemble the authenticated FIT linear systems behind the Q/K plant."""

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
    first = cases[0]
    if any(
        case.parent_query_weight_prior is None
        or case.parent_key_weight_prior is None
        or case.parent_value_weight_prior is None
        or case.parent_output_weight_prior is None
        or case.parent_query_norm_weight is None
        or case.parent_key_norm_weight is None
        or case.parent_weight_prior_sha256 != first.parent_weight_prior_sha256
        or not torch.equal(
            case.parent_query_weight_prior,
            first.parent_query_weight_prior,
        )
        or not torch.equal(case.parent_key_weight_prior, first.parent_key_weight_prior)
        or not torch.equal(
            case.parent_value_weight_prior,
            first.parent_value_weight_prior,
        )
        or not torch.equal(
            case.parent_output_weight_prior,
            first.parent_output_weight_prior,
        )
        or not torch.equal(
            case.parent_query_norm_weight,
            first.parent_query_norm_weight,
        )
        or not torch.equal(case.parent_key_norm_weight, first.parent_key_norm_weight)
        for case in cases
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "FIT cases do not share one authenticated parent Q/K weight/norm prior."
        )
    prior_seed = _copy_seed(
        seed,
        query_weight=first.parent_query_weight_prior.to(torch.bfloat16),
        key_weight=first.parent_key_weight_prior.to(torch.bfloat16),
        value_weight=first.parent_value_weight_prior.to(torch.bfloat16),
        output_weight=first.parent_output_weight_prior.to(torch.bfloat16),
        query_norm_weight=first.parent_query_norm_weight.to(torch.bfloat16),
        key_norm_weight=first.parent_key_norm_weight.to(torch.bfloat16),
    )
    pre_rope_gauge, gauge_receipt = _fit_pre_rope_dense_gauge(prior_seed, cases)
    query_weight64 = first.parent_query_weight_prior.to(torch.float64)
    key_weight64 = first.parent_key_weight_prior.to(torch.float64)
    for case in cases:
        query_runtime, key_runtime, target_receipt = _cold_route_runtime_targets(
            case,
            prior_seed,
            pre_rope_gauge=pre_rope_gauge,
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
            norm_weight=prior_seed.query_norm_weight,
            norm_epsilon=prior_seed.norm_epsilon,
            cosine=case.query_rope_cos,
            sine=case.query_rope_sin,
            role="query",
        )
        key_raw = _cold_projection_preimage(
            key_runtime,
            key_anchor,
            norm_weight=prior_seed.key_norm_weight,
            norm_epsilon=prior_seed.norm_epsilon,
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
        key_weights.append(torch.full((batch * key_length,), float(case_scale.item())))
        target_receipts.append(dict(target_receipt))
    concatenated_query_targets = torch.cat(query_targets, dim=0)
    concatenated_key_targets = torch.cat(key_targets, dim=0)
    target_geometry = _fit_qk_target_geometry_receipt(
        concatenated_query_targets,
        concatenated_key_targets,
        seed=seed,
        parent_prior_case_count=sum(
            case.folded_parent_query_pre_rope is not None for case in cases
        ),
        fit_case_count=len(cases),
    )
    return MappingProxyType(
        {
            "query_inputs": torch.cat(query_inputs, dim=0),
            "query_targets": concatenated_query_targets,
            "query_weights": torch.cat(query_weights, dim=0),
            "key_inputs": torch.cat(key_inputs, dim=0),
            "key_targets": concatenated_key_targets,
            "key_weights": torch.cat(key_weights, dim=0),
            "target_receipts": tuple(target_receipts),
            "pre_rope_gauge": pre_rope_gauge,
            "pre_rope_gauge_receipt": gauge_receipt,
            "target_geometry": target_geometry,
            "query_weight_prior": first.parent_query_weight_prior,
            "key_weight_prior": first.parent_key_weight_prior,
            "value_weight_prior": first.parent_value_weight_prior,
            "output_weight_prior": first.parent_output_weight_prior,
            "query_norm_weight": first.parent_query_norm_weight,
            "key_norm_weight": first.parent_key_norm_weight,
            "parent_weight_prior_sha256": first.parent_weight_prior_sha256,
        }
    )


def _compile_qk_cold_start_plant(
    seed: Lfm2EndpointQkvoSeed,
    fit_bank: Lfm2EndpointQkResidualEvidenceBank,
    *,
    fit_systems: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Compile a FIT-only route plant for an exact-zero Q/K incumbent."""

    systems = (
        _qk_cold_fit_systems(seed, fit_bank) if fit_systems is None else fit_systems
    )
    planted_query, query_receipt = _cold_ridge_projection(
        systems["query_inputs"],
        systems["query_targets"],
        systems["query_weights"],
        role="query",
        normalize_element_rms=False,
        prior=systems["query_weight_prior"],
    )
    planted_key, key_receipt = _cold_ridge_projection(
        systems["key_inputs"],
        systems["key_targets"],
        systems["key_weights"],
        role="key",
        normalize_element_rms=False,
        prior=systems["key_weight_prior"],
    )
    if tuple(planted_query.shape) != tuple(seed.query_weight.shape) or tuple(
        planted_key.shape
    ) != tuple(seed.key_weight.shape):
        raise Lfm2SemanticAttentionRefoldError(
            "Q/K cold-start plant changed the dense target topology."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-qk-cold-start-plant-v1",
        "fit_evidence_sha256": fit_bank.evidence_sha256,
        "source_artifact_sha256": fit_bank.source_artifact_sha256,
        "seed_sha256": seed.seed_sha256,
        "case_evidence_sha256": [case.evidence_sha256 for case in fit_bank.cases],
        "runtime_targets": list(systems["target_receipts"]),
        "pre_rope_dense_gauge": dict(systems["pre_rope_gauge_receipt"]),
        "fit_target_geometry": dict(systems["target_geometry"]),
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
    return (
        planted_query,
        planted_key,
        MappingProxyType({**body, "plant_sha256": canonical_sha256(body)}),
    )


def _strict_metric_improvement(
    candidate: Lfm2EndpointQkMetrics,
    baseline: Lfm2EndpointQkMetrics,
) -> bool:
    tolerance = 1e-12
    return bool(
        candidate.endpoint_relative_rmse < baseline.endpoint_relative_rmse - tolerance
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
    force_structural_coverage: bool = False,
    defer_heldout_diagnostics: bool = False,
) -> tuple[Lfm2EndpointQkvoSeed, Mapping[str, Any] | None]:
    """Plant zero Q/K from FIT and treat every held-out metric as diagnostic."""

    torch = _torch()
    query_was_zero = _exact_zero(seed.query_weight)
    key_was_zero = _exact_zero(seed.key_weight)
    if not query_was_zero and not key_was_zero and not force_structural_coverage:
        return seed, None
    fit_systems = _qk_cold_fit_systems(seed, banks["fit"])
    if query_was_zero or key_was_zero:
        raw_query, raw_key, plant = _compile_qk_cold_start_plant(
            seed,
            banks["fit"],
            fit_systems=fit_systems,
        )
    else:
        raw_query = seed.query_weight
        raw_key = seed.key_weight
        plant_body = {
            "schema": "neural-forge-lfm2-semantic-qk-cold-start-plant-v1",
            "fit_evidence_sha256": banks["fit"].evidence_sha256,
            "source_artifact_sha256": banks["fit"].source_artifact_sha256,
            "seed_sha256": seed.seed_sha256,
            "case_evidence_sha256": [
                case.evidence_sha256 for case in banks["fit"].cases
            ],
            "runtime_targets": [],
            "pre_rope_dense_gauge": dict(fit_systems["pre_rope_gauge_receipt"]),
            "fit_target_geometry": dict(fit_systems["target_geometry"]),
            "query_ridge": None,
            "key_ridge": None,
            "planted_query_sha256": _tensor_sha256(raw_query),
            "planted_key_sha256": _tensor_sha256(raw_key),
            "fit_partition_only_used_for_plant": True,
            "heldout_partitions_used_for_plant": False,
            "randomness_used": False,
            "gradients_used": False,
            "training_steps": 0,
            "established_qk_preserved_before_axis_coverage": True,
        }
        plant = MappingProxyType(
            {**plant_body, "plant_sha256": canonical_sha256(plant_body)}
        )
    candidate_query = raw_query if query_was_zero else seed.query_weight
    candidate_key = raw_key if key_was_zero else seed.key_weight

    def prior_preserving_coverage(value: Any, role: str) -> Mapping[str, Any]:
        coverage = dict(_dense_axis_coverage(value, role=role))
        body = {
            "schema": "neural-forge-lfm2-qk-prior-preserving-axis-diagnostic-v1",
            "role": role,
            "coverage": coverage,
            "structural_axis_mutation_performed": False,
            "row_sign_nullspace_fill_used": False,
            "parent_weight_prior_preserved": True,
            "fit_input_nullspace_preserved": True,
            "numeric_diagnostic_only": True,
        }
        return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})

    query_coverage = prior_preserving_coverage(
        candidate_query,
        "q_proj.weight",
    )
    key_coverage = prior_preserving_coverage(
        candidate_key,
        "k_proj.weight",
    )
    planted = _copy_seed(
        seed,
        query_weight=candidate_query,
        key_weight=candidate_key,
        value_weight=fit_systems["value_weight_prior"].to(torch.bfloat16),
        output_weight=fit_systems["output_weight_prior"].to(torch.bfloat16),
        query_norm_weight=fit_systems["query_norm_weight"].to(torch.bfloat16),
        key_norm_weight=fit_systems["key_norm_weight"].to(torch.bfloat16),
    )
    frozen_sha = canonical_sha256(
        {
            "query_weight_sha256": _tensor_sha256(planted.query_weight),
            "key_weight_sha256": _tensor_sha256(planted.key_weight),
        }
    )
    metrics: dict[str, Any] = {}
    evaluated_partitions = ("fit",) if defer_heldout_diagnostics else _PARTITIONS
    for partition in evaluated_partitions:
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
            "diagnostic_only": partition != "fit",
            "used_for_plant_or_selection": partition == "fit",
        }
        if (
            canonical_sha256(
                {
                    "query_weight_sha256": _tensor_sha256(planted.query_weight),
                    "key_weight_sha256": _tensor_sha256(planted.key_weight),
                }
            )
            != frozen_sha
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Q/K cold-start candidate changed during diagnostic replay."
            )
    fit_strictly_improved = _strict_metric_improvement(
        Lfm2EndpointQkMetrics(**metrics["fit"]["candidate"]),
        Lfm2EndpointQkMetrics(**metrics["fit"]["baseline"]),
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
        "query_axis_coverage": dict(query_coverage),
        "key_axis_coverage": dict(key_coverage),
        "partitions": metrics,
        "fit_strictly_improved": fit_strictly_improved,
        "heldout_nonregression_passed": (
            None
            if defer_heldout_diagnostics
            else all(
                metrics[partition]["nonregressing"]
                for partition in ("fit_check", "selection", "validation")
            )
        ),
        "heldout_metrics_are_diagnostics_not_publication_gates": True,
        "heldout_metrics_deferred_until_complete_qkvo_freeze": (
            defer_heldout_diagnostics
        ),
        "fit_check_used_for_plant": False,
        "selection_used_for_plant": False,
        "validation_used_for_plant": False,
        "candidate_frozen_before_fit_check_selection_validation": True,
        "heldout_refit_or_reselection": False,
        "zero_incumbent_noop_allowed": False,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return planted, MappingProxyType({**body, "gate_sha256": canonical_sha256(body)})


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


def _qk_policy_for_body_recipe(
    policy: Lfm2EndpointQkResidualFoldPolicy,
    recipe: Lfm2SemanticBodyRecipe,
) -> Lfm2EndpointQkResidualFoldPolicy:
    """Select the complete-alpha Q/K lane without hiding structural choices."""

    if recipe.fit_metrics_order_candidates:
        return policy
    # Keeping alpha zero alongside alpha one authenticates the incumbent
    # control while making the only effectful proposal the full residual step
    # under the compiler's largest authenticated trust radius. FIT metrics are
    # still emitted, but cannot choose a weaker alpha in any non-FIT recipe.
    # This is deliberately named as maximum-trust, not uncapped.
    return replace(
        policy,
        trust_radius=2.0,
        alpha_frontier=(0.0, 1.0),
        authentication_sha256="",
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
                    "all_semantically_exact_endpoint_rows": True,
                    "terminal_endpoint_only": False,
                    "context_values_quantized_by_bfloat16_runtime_before_fp64_aggregation": True,
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
    prior_preservation_receipt: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
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
            or (self.changed and self.selected_alpha != 1.0)
            or not isinstance(self.fit_improved, bool)
            or not isinstance(self.prior_preservation_receipt, Mapping)
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Output-writer proposal metadata is inconsistent."
            )
        _digest(self.fit_context_sha256, "FIT context SHA-256")
        object.__setattr__(self, "output_weight", output)
        object.__setattr__(
            self,
            "prior_preservation_receipt",
            MappingProxyType(dict(self.prior_preservation_receipt)),
        )
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
            "prior_preservation": dict(self.prior_preservation_receipt),
            "solver": "closed_form_weighted_residual_ridge_full_step_bfloat16_publish",
            "value_weight_fixed": True,
            "deterministic_full_fit_correction_published": self.changed,
            "fit_improved_is_diagnostic_only": True,
            "numeric_metric_used_for_step_size_selection": False,
            "numeric_metric_used_for_publication": False,
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
                    "all_semantically_exact_endpoint_rows": True,
                    "terminal_endpoint_only": False,
                    "probabilities_quantized_by_bfloat16_runtime_before_fp64_aggregation": True,
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
            or not isinstance(self.fit_improved, bool)
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
            "factorized_candidate_published_in_proposal": True,
            "factorization_attempt_rejected_after_bfloat16_fit_replay": False,
            "diagnostic_only": False,
            "publication_allowed": True,
            "publication_lane": "structural_fit_ordered_value_output_candidates",
            "fit_improved_is_diagnostic_only": True,
            "numeric_metric_used_for_publication": False,
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
    output = output_weight.detach().to(device=resolved, dtype=torch.bfloat16)
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
        key_inputs = case.key_value_inputs.to(
            device=resolved,
            dtype=torch.bfloat16,
        )
        value_heads = (
            torch.nn.functional.linear(key_inputs, value)
            .reshape(batch, key_length, key_value_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        repeated_value = value_heads.repeat_interleave(
            query_heads // key_value_heads,
            dim=1,
        )
        context = (
            case.probabilities.to(device=resolved, dtype=torch.bfloat16)
            .matmul(repeated_value)
            .permute(0, 2, 1, 3)
            .reshape(batch, query_length, query_heads * head_dim)
        )
        return torch.nn.functional.linear(context, output).detach().cpu().contiguous()


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
                torch.nn.functional.linear(key_inputs, value)
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
                case.probabilities.to(device=device, dtype=torch.bfloat16)
                .matmul(
                    value_heads.repeat_interleave(
                        seed.queries_per_key_value,
                        dim=1,
                    )
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
                # Seal the live BF16 context first, then widen only for the
                # closed-form FP64 writer solve and diagnostic aggregation.
                context=context.detach().to(device="cpu", dtype=torch.float64),
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
    output = output_weight.detach().to(device="cpu", dtype=torch.bfloat16)
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_target = torch.zeros((), dtype=torch.float64)
    weighted_elements = torch.zeros((), dtype=torch.float64)
    weighted_cosine = torch.zeros((), dtype=torch.float64)
    cosine_weight = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    worst_cosine = 1.0
    rows = 0
    for case in selected:
        prediction = torch.nn.functional.linear(
            case.context.to(torch.bfloat16),
            output,
        ).to(torch.float64)
        difference = prediction - case.endpoint_target.to(torch.float64)
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
    candidate = (
        (incumbent_transpose + delta)
        .T.to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
    )
    changed = not torch.equal(candidate, incumbent)
    candidate_metrics = _writer_metrics(selected, candidate) if changed else baseline
    _left, singular, right_h = torch.linalg.svd(
        weighted_x,
        full_matrices=True,
    )
    largest = 0.0 if int(singular.numel()) == 0 else float(singular.max().item())
    rank_threshold = (
        largest * max(map(int, weighted_x.shape)) * torch.finfo(torch.float64).eps
    )
    rank = int((singular > rank_threshold).sum().item())
    null_basis = right_h[rank:]
    fp64_nullspace_action = (
        torch.zeros((), dtype=torch.float64, device=device)
        if int(null_basis.shape[0]) == 0
        else (null_basis @ delta).abs().max()
    )
    published_delta = candidate.to(torch.float64) - incumbent.to(torch.float64)
    published_nullspace_action = (
        torch.zeros((), dtype=torch.float64)
        if int(null_basis.shape[0]) == 0
        else (published_delta @ null_basis.detach().cpu().T).abs().max()
    )
    preservation_body = {
        "schema": "neural-forge-lfm2-fixed-v-output-prior-preservation-v1",
        "fit_context_design_sha256": _tensor_sha256(weighted_x.detach().cpu()),
        "fit_context_rank": rank,
        "fit_context_width": int(weighted_x.shape[1]),
        "fit_context_nullity": int(null_basis.shape[0]),
        "rank_threshold": rank_threshold,
        "analytic_fp64_delta_action_on_fit_context_nullspace_maximum_absolute": float(
            fp64_nullspace_action.item()
        ),
        "published_bfloat16_delta_action_on_fit_context_nullspace_maximum_absolute": float(
            published_nullspace_action.item()
        ),
        "analytic_delta_lies_in_fit_context_rowspace": True,
        "value_weight_fixed_exactly": True,
        "incumbent_output_is_broad_prior": True,
        "bfloat16_publication_error_reported_not_hidden": True,
        "joint_svd_refactor_used_for_publication": False,
    }
    preservation = MappingProxyType(
        {**preservation_body, "result_sha256": canonical_sha256(preservation_body)}
    )
    return _OutputWriterProposal(
        output_weight=candidate,
        changed=changed,
        fit_improved=(
            candidate_metrics.endpoint_relative_rmse < baseline.endpoint_relative_rmse
        ),
        selected_alpha=1.0 if changed else None,
        baseline_fit_metrics=baseline,
        candidate_fit_metrics=candidate_metrics,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
        fit_context_sha256=context_sha,
        prior_preservation_receipt=preservation,
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
    fit_sha = canonical_sha256(
        {
            "case_sha256": [case.case_sha256 for case in selected],
            "probability_weighted_input_features_used": True,
        }
    )
    return _JointVoProposal(
        value_weight=candidate_value,
        output_weight=candidate_output,
        changed=changed,
        fit_improved=improvement > 0.0,
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
            "diagnostic_only": True,
            "used_for_publication": False,
        }
    body = {
        "status": (
            "published-deterministic-fit-only-correction"
            if proposal.changed
            else "deterministic-fit-correction-rounded-to-incumbent"
        ),
        "accepted": proposal.changed,
        "proposal": proposal.as_dict(),
        "rejection_partition": None,
        "fit_check": reports["fit_check"],
        "selection": reports["selection"],
        "validation": reports["validation"],
        "candidate_frozen_across_heldout_gates": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metric_used_for_publication": False,
        "incumbent_rollback_allowed": False,
    }
    published = proposal.output_weight if proposal.changed else incumbent
    return published, _seal_writer_gate(body)


def _vo_fit_ordering_key(
    metrics: _WriterMetrics,
    *,
    proposal_sha256: str,
) -> tuple[float, float, float, float, float, str]:
    """Order structurally eligible proposals with FIT values and stable ties."""

    return (
        metrics.endpoint_relative_rmse,
        metrics.endpoint_rmse,
        metrics.endpoint_maximum_absolute_error,
        -metrics.endpoint_cosine_similarity_mean,
        -metrics.endpoint_cosine_similarity_worst,
        proposal_sha256,
    )


def _vo_structural_eligibility(
    *,
    value_weight: Any,
    output_weight: Any,
    incumbent_value_weight: Any,
    incumbent_output_weight: Any,
) -> Mapping[str, Any]:
    """Classify a V/O proposal without consulting any numeric quality metric."""

    torch = _torch()
    tensors = (value_weight, output_weight)
    tensor_objects = all(isinstance(value, torch.Tensor) for value in tensors)
    dense = bool(
        tensor_objects and all(value.layout == torch.strided for value in tensors)
    )
    bfloat16 = bool(
        tensor_objects and all(value.dtype == torch.bfloat16 for value in tensors)
    )
    exact_shape = bool(
        tensor_objects
        and tuple(value_weight.shape) == tuple(incumbent_value_weight.shape)
        and tuple(output_weight.shape) == tuple(incumbent_output_weight.shape)
    )
    finite = bool(
        tensor_objects
        and dense
        and all(bool(torch.isfinite(value).all().item()) for value in tensors)
    )
    effectful = bool(
        tensor_objects
        and dense
        and not (
            torch.equal(value_weight, incumbent_value_weight)
            and torch.equal(output_weight, incumbent_output_weight)
        )
    )
    eligible = bool(finite and dense and bfloat16 and exact_shape and effectful)
    return MappingProxyType(
        {
            "finite": finite,
            "dense_strided": dense,
            "bfloat16": bfloat16,
            "exact_shape": exact_shape,
            "effectful": effectful,
            "eligible": eligible,
            "eligibility_basis": _VO_CANDIDATE_ELIGIBILITY_BASIS,
            "numeric_metric_used_for_eligibility": False,
        }
    )


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
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Freeze a FIT-ordered structural proposal, then diagnose held-out rows."""

    torch = _torch()
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    candidate_rows = [
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
    baseline_fit = _joint_vo_metrics(
        banks["fit"],
        seed.value_weight,
        seed.output_weight,
        head_dim=seed.head_dim,
        compute_devices=compute_devices,
    )
    reports: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    for candidate in candidate_rows:
        structural = _vo_structural_eligibility(
            value_weight=candidate["value"],
            output_weight=candidate["output"],
            incumbent_value_weight=seed.value_weight,
            incumbent_output_weight=seed.output_weight,
        )
        if candidate["name"] == "incumbent":
            structural = MappingProxyType(
                {
                    **dict(structural),
                    "effectful": False,
                    "eligible": False,
                    "incumbent_fallback_only": True,
                }
            )
        elif structural["dense_strided"] and structural["effectful"] is not bool(
            candidate["proposal_changed"]
        ):
            raise Lfm2SemanticAttentionRefoldError(
                f"{candidate['name']} proposal metadata disagrees with its tensor effect."
            )
        fit_metrics = (
            _joint_vo_metrics(
                banks["fit"],
                candidate["value"],
                candidate["output"],
                head_dim=seed.head_dim,
                compute_devices=compute_devices,
            )
            if structural["finite"]
            and structural["dense_strided"]
            and structural["bfloat16"]
            and structural["exact_shape"]
            else None
        )
        reports[candidate["name"]] = {
            "complexity": candidate["complexity"],
            "proposal_changed": candidate["proposal_changed"],
            "proposal_sha256": candidate["proposal_sha256"],
            **dict(structural),
            "eligible_after_fit_structural_check": structural["eligible"],
            "eligible_after_fit_fit_check_selection": structural["eligible"],
            "structurally_publishable": structural["eligible"],
            "rejection_partition": None,
            "partitions": {
                "fit": {
                    "case_sha256": [case.case_sha256 for case in banks["fit"]],
                    "baseline": baseline_fit.as_dict(),
                    "candidate": None if fit_metrics is None else fit_metrics.as_dict(),
                    "nonregressing": (
                        None
                        if fit_metrics is None
                        else _writer_nonregressing(
                            fit_metrics,
                            baseline_fit,
                            tolerance=tolerance,
                        )
                    ),
                    "diagnostic_only": not recipe.fit_metrics_order_candidates,
                    "used_for_candidate_ordering": (
                        structural["eligible"] and recipe.fit_metrics_order_candidates
                    ),
                    "used_for_structural_eligibility": False,
                    "used_for_publication_threshold": False,
                    "used_for_publication": False,
                }
            },
        }
        if structural["eligible"]:
            assert fit_metrics is not None
            eligible.append(
                (
                    _vo_fit_ordering_key(
                        fit_metrics,
                        proposal_sha256=str(candidate["proposal_sha256"]),
                    ),
                    candidate,
                )
            )
    if not eligible:
        selected = candidate_rows[0]
    elif recipe.fit_metrics_order_candidates:
        selected = min(eligible, key=lambda value: value[0])[1]
    else:
        structural_by_name = {str(value[1]["name"]): value[1] for value in eligible}
        if (
            recipe.component_lanes["value_output"]
            == "fixed_writer_then_joint_residual_priority"
        ):
            selected = structural_by_name.get(
                "fixed_v_output_only",
                structural_by_name.get("joint_value_output"),
            )
        else:
            selected = structural_by_name.get(
                "joint_value_output",
                structural_by_name.get("fixed_v_output_only"),
            )
        if selected is None:  # pragma: no cover - eligible names are closed above
            raise AssertionError("Structural V/O frontier has no known proposal.")
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

    # FIT-CHECK, SELECTION, and VALIDATION are opened only after the selected
    # tensors above are frozen.  They populate diagnostics for every analytic
    # proposal but never refit, reselect, reject, or restore the incumbent.
    baseline_metrics: dict[str, _WriterMetrics] = {"fit": baseline_fit}
    for partition in ("fit_check", "selection", "validation"):
        baseline = _joint_vo_metrics(
            banks[partition],
            seed.value_weight,
            seed.output_weight,
            head_dim=seed.head_dim,
            compute_devices=compute_devices,
        )
        baseline_metrics[partition] = baseline
        for candidate in candidate_rows:
            report = reports[candidate["name"]]
            structural_valid = bool(
                report["finite"]
                and report["dense_strided"]
                and report["bfloat16"]
                and report["exact_shape"]
            )
            metrics = (
                _joint_vo_metrics(
                    banks[partition],
                    candidate["value"],
                    candidate["output"],
                    head_dim=seed.head_dim,
                    compute_devices=compute_devices,
                )
                if structural_valid
                else None
            )
            report["partitions"][partition] = {
                "case_sha256": [case.case_sha256 for case in banks[partition]],
                "baseline": baseline.as_dict(),
                "candidate": None if metrics is None else metrics.as_dict(),
                "nonregressing": (
                    None
                    if metrics is None
                    else _writer_nonregressing(metrics, baseline, tolerance=tolerance)
                ),
                "diagnostic_only": True,
                "used_for_candidate_ordering": False,
                "used_for_structural_eligibility": False,
                "used_for_publication_threshold": False,
                "used_for_publication": False,
            }
        if (
            canonical_sha256(
                {
                    "candidate_name": selected["name"],
                    "value_weight_sha256": _tensor_sha256(frozen["value"]),
                    "output_weight_sha256": _tensor_sha256(frozen["output"]),
                    "proposal_sha256": selected["proposal_sha256"],
                }
            )
            != frozen_sha
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Selected V/O candidate changed during held-out diagnostics."
            )

    route_nonregressing = _metrics_nonregressing(
        validation_route_candidate,
        validation_route_baseline,
        tolerance=tolerance,
    )
    validation_baseline = baseline_metrics["validation"]
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
            "Selected V/O candidate changed after the pre-heldout freeze."
        )
    published_name = str(selected["name"])
    body = {
        "schema": "neural-forge-lfm2-semantic-vo-candidate-gate-v1",
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["value_output"],
        "candidate_reports": reports,
        "strongest_selection_candidate": (
            None if published_name == "incumbent" else published_name
        ),
        "selected_candidate": published_name,
        "published_candidate": published_name,
        "selection_rule": (
            "exclude-incumbent-when-any-finite-dense-bfloat16-exact-shape-"
            "effectful-proposal-exists-then-order-eligible-proposals-by-fit"
            if recipe.fit_metrics_order_candidates
            else (
                "exclude-incumbent-then-fixed-output-writer-before-joint-value-"
                "output-by-authenticated-residual-writer-priority"
                if recipe.component_lanes["value_output"]
                == "fixed_writer_then_joint_residual_priority"
                else "exclude-incumbent-then-joint-value-output-before-fixed-"
                "writer-by-authenticated-structural-priority"
            )
        ),
        "component_selection_policy_equivalent_recipe_ids": list(
            lfm2_semantic_body_recipe_ids_for_component_lane(
                "value_output",
                recipe.component_lanes["value_output"],
            )
        ),
        "component_selection_policy_equivalent_to_other_recipe": (
            len(
                lfm2_semantic_body_recipe_ids_for_component_lane(
                    "value_output",
                    recipe.component_lanes["value_output"],
                )
            )
            > 1
        ),
        "fit_ordering_metrics": [
            "endpoint_relative_rmse",
            "endpoint_rmse",
            "endpoint_maximum_absolute_error",
            "negative_endpoint_cosine_similarity_mean",
            "negative_endpoint_cosine_similarity_worst",
        ],
        "exact_fit_tie_breaker": "proposal_sha256",
        "structurally_eligible_candidate_count": len(eligible),
        "candidate_eligibility_basis": _VO_CANDIDATE_ELIGIBILITY_BASIS,
        "structural_safety_is_only_candidate_veto": True,
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates
        ),
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_metric_threshold_used_for_candidate_selection": False,
        "incumbent_noop_fallback_used": not eligible,
        "incumbent_noop_fallback_reason": (
            "all-analytic-proposals-structurally-invalid-or-identical"
            if not eligible
            else None
        ),
        "selected_candidate_frozen_before_validation": True,
        "selected_candidate_frozen_before_heldout_diagnostics": True,
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
            "diagnostic_metrics": [
                "endpoint_relative_rmse",
                "endpoint_maximum_absolute_error",
                "endpoint_cosine_similarity_mean",
                "endpoint_cosine_similarity_worst",
                "qk_route_metrics",
            ],
            "diagnostic_only": True,
            "used_for_publication": False,
        },
        "selected_correction_rejected_on_validation": False,
        "alternate_candidate_selected_after_validation": False,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "published_value_weight_sha256": _tensor_sha256(frozen["value"]),
        "published_output_weight_sha256": _tensor_sha256(frozen["output"]),
        "fit_only_analytic_solves": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metric_used_for_candidate_selection": bool(
            recipe.fit_metrics_order_candidates and len(eligible) > 1
        ),
        "numeric_metric_used_for_publication_or_rollback": False,
        "joint_value_output_publication_allowed": True,
        "fixed_value_weight_preserved": torch.equal(frozen["value"], seed.value_weight),
        "all_semantically_exact_endpoint_rows_used": True,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return (
        frozen["value"],
        frozen["output"],
        MappingProxyType({**body, "gate_sha256": canonical_sha256(body)}),
    )


def _compile_complete_attention_cold_start(
    incumbent_seed: Lfm2EndpointQkvoSeed,
    *,
    banks: Mapping[str, Lfm2EndpointQkResidualEvidenceBank],
    policy: Lfm2EndpointQkResidualFoldPolicy,
    exact_zero_roles: Sequence[str],
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Mapping[str, Any]:
    """Compile one FIT-only, coverage-complete Q/K/V/O blank-target plant."""

    torch = _torch()
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    zero_roles = tuple(exact_zero_roles)
    expected_roles = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    )
    if not zero_roles or any(role not in expected_roles for role in zero_roles):
        raise Lfm2SemanticAttentionRefoldError(
            "Complete attention cold-start requires authenticated exact-zero roles."
        )
    entry_weights = {
        "q_proj.weight": incumbent_seed.query_weight,
        "k_proj.weight": incumbent_seed.key_weight,
        "v_proj.weight": incumbent_seed.value_weight,
        "out_proj.weight": incumbent_seed.output_weight,
    }
    observed_zero_roles = tuple(
        role for role in expected_roles if _exact_zero(entry_weights[role])
    )
    if zero_roles != observed_zero_roles:
        raise Lfm2SemanticAttentionRefoldError(
            "Supplied exact-zero attention roles disagree with incumbent tensors."
        )
    entry_coverage = {
        role: dict(_dense_axis_coverage(weight, role=role))
        for role, weight in entry_weights.items()
    }
    seed, qk_cold_start = _cold_start_qk_seed(
        incumbent_seed,
        banks=banks,
        policy=policy,
        force_structural_coverage=True,
        defer_heldout_diagnostics=True,
    )
    if qk_cold_start is None:
        raise AssertionError("Forced Q/K structural cold-start produced no receipt.")
    frozen_query = seed.query_weight.detach().cpu().clone()
    frozen_key = seed.key_weight.detach().cpu().clone()
    joint_banks = {
        "fit": _joint_vo_cases(
            banks["fit"],
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        )
    }
    joint_proposal = _compile_joint_vo(
        joint_banks["fit"],
        seed.value_weight,
        seed.output_weight,
        head_dim=seed.head_dim,
        solver_device=policy.solver_device,
    )
    candidate_value = seed.value_weight
    candidate_output = seed.output_weight
    value_coverage_body = dict(
        _dense_axis_coverage(candidate_value, role="v_proj.weight")
    )
    value_coverage = MappingProxyType(
        {
            **value_coverage_body,
            "repaired_element_count": 0,
            "source_prior_preserved_without_axis_mutation": True,
        }
    )
    if value_coverage["structurally_nonzero"] is not True:
        raise Lfm2SemanticAttentionRefoldError(
            "Parent-derived fixed V prior lacks complete structural coverage."
        )
    preliminary_output_coverage = _dense_axis_coverage(
        candidate_output,
        role="out_proj.weight",
    )
    writer_banks = {
        "fit": _writer_cases_from_joint(
            joint_banks["fit"],
            seed,
            compute_devices=policy.compute_devices,
        )
    }
    writer_proposal = _compile_output_writer(
        writer_banks["fit"],
        candidate_output,
        solver_device=policy.solver_device,
    )
    writer_component_published = bool(writer_proposal.changed)
    if writer_component_published:
        candidate_output = writer_proposal.output_weight
    fit_writer_cases = tuple(writer_banks["fit"])
    writer_fit_inputs = torch.cat(
        tuple(
            case.context.reshape(-1, int(case.context.shape[-1]))
            for case in fit_writer_cases
        ),
        dim=0,
    )
    writer_fit_targets = torch.cat(
        tuple(
            case.endpoint_target.reshape(-1, int(case.endpoint_target.shape[-1]))
            for case in fit_writer_cases
        ),
        dim=0,
    )
    writer_fit_sqrt_weights = torch.sqrt(
        torch.cat(
            tuple(case.row_weights.reshape(-1) for case in fit_writer_cases),
            dim=0,
        )
    )
    del writer_fit_inputs, writer_fit_targets, writer_fit_sqrt_weights
    output_coverage_body = dict(
        _dense_axis_coverage(candidate_output, role="out_proj.weight")
    )
    output_coverage = MappingProxyType(
        {
            **output_coverage_body,
            "repaired_element_count": 0,
            "source_prior_plus_fit_rowspace_correction_preserved": True,
        }
    )
    if output_coverage["structurally_nonzero"] is not True:
        raise Lfm2SemanticAttentionRefoldError(
            "Fixed-V output prior/correction lacks complete structural coverage."
        )
    published_weights = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": candidate_value,
        "out_proj.weight": candidate_output,
    }
    published_coverage = {
        role: dict(_dense_axis_coverage(weight, role=role))
        for role, weight in published_weights.items()
    }
    if not all(
        receipt["structurally_nonzero"] is True
        for receipt in published_coverage.values()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "FIT-only attention cold-start did not cover every Q/K/V/O axis."
        )
    for role in zero_roles:
        if _exact_zero(published_weights[role]) or torch.equal(
            published_weights[role], entry_weights[role]
        ):
            raise Lfm2SemanticAttentionRefoldError(
                f"Exact-zero attention role {role} remained an incumbent no-op."
            )
    frozen_sha = canonical_sha256(
        {role: _tensor_sha256(weight) for role, weight in published_weights.items()}
    )
    qk_body = dict(qk_cold_start)
    qk_body.pop("gate_sha256")
    qk_metrics = dict(qk_body["partitions"])
    for partition in ("fit_check", "selection", "validation"):
        baseline = _evaluate_bank(
            banks[partition],
            incumbent_seed,
            query_weight=incumbent_seed.query_weight,
            key_weight=incumbent_seed.key_weight,
            policy=policy,
        )
        candidate = _evaluate_bank(
            banks[partition],
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        )
        qk_metrics[partition] = {
            "baseline": baseline.as_dict(),
            "candidate": candidate.as_dict(),
            "nonregressing": _metrics_nonregressing(
                candidate,
                baseline,
                tolerance=policy.metric_regression_tolerance,
            ),
            "diagnostic_only": True,
            "used_for_plant_or_selection": False,
        }
        if (
            canonical_sha256(
                {
                    role: _tensor_sha256(weight)
                    for role, weight in published_weights.items()
                }
            )
            != frozen_sha
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Complete Q/K/V/O candidate changed during Q/K diagnostics."
            )
    qk_body.update(
        {
            "partitions": qk_metrics,
            "heldout_nonregression_passed": all(
                qk_metrics[partition]["nonregressing"]
                for partition in ("fit_check", "selection", "validation")
            ),
            "heldout_metrics_deferred_until_complete_qkvo_freeze": True,
            "complete_qkvo_frozen_before_heldout_diagnostics": True,
            "heldout_diagnostics_completed_after_complete_qkvo_freeze": True,
        }
    )
    qk_cold_start = MappingProxyType(
        {**qk_body, "gate_sha256": canonical_sha256(qk_body)}
    )
    qk_partitions = qk_cold_start["partitions"]
    baseline_selection = Lfm2EndpointQkMetrics(**qk_partitions["selection"]["baseline"])
    candidate_selection = Lfm2EndpointQkMetrics(
        **qk_partitions["selection"]["candidate"]
    )
    baseline_validation = Lfm2EndpointQkMetrics(
        **qk_partitions["validation"]["baseline"]
    )
    candidate_validation = Lfm2EndpointQkMetrics(
        **qk_partitions["validation"]["candidate"]
    )
    for partition in ("fit_check", "selection", "validation"):
        joint_banks[partition] = _joint_vo_cases(
            banks[partition],
            seed,
            query_weight=frozen_query,
            key_weight=frozen_key,
            policy=policy,
        )
        if (
            canonical_sha256(
                {
                    role: _tensor_sha256(weight)
                    for role, weight in published_weights.items()
                }
            )
            != frozen_sha
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Complete Q/K/V/O candidate changed while opening heldout diagnostics."
            )
    writer_banks.update(
        {
            partition: _writer_cases_from_joint(
                joint_banks[partition],
                seed,
                compute_devices=policy.compute_devices,
            )
            for partition in ("fit_check", "selection", "validation")
        }
    )
    tolerance = policy.metric_regression_tolerance
    candidate_value, candidate_output, vo_gate = _select_and_validate_vo_candidate(
        banks=joint_banks,
        seed=seed,
        output_proposal=writer_proposal,
        joint_proposal=joint_proposal,
        tolerance=tolerance,
        validation_route_baseline=baseline_validation,
        validation_route_candidate=candidate_validation,
        compute_devices=policy.compute_devices,
        body_recipe=recipe,
    )
    published_vo_kind = str(vo_gate["published_candidate"])
    writer_component_published = published_vo_kind == "fixed_v_output_only"
    joint_component_published = published_vo_kind == "joint_value_output"
    expected_value = (
        joint_proposal.value_weight if joint_component_published else seed.value_weight
    )
    expected_output = (
        joint_proposal.output_weight
        if joint_component_published
        else (
            writer_proposal.output_weight
            if writer_component_published
            else seed.output_weight
        )
    )
    if not torch.equal(candidate_value, expected_value) or not torch.equal(
        candidate_output,
        expected_output,
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Cold-start V/O selection disagrees with its frozen analytic proposal."
        )
    value_coverage_body = dict(
        _dense_axis_coverage(candidate_value, role="v_proj.weight")
    )
    value_coverage = MappingProxyType(
        {
            **value_coverage_body,
            "repaired_element_count": 0,
            "source_prior_preserved_without_axis_mutation": (
                not joint_component_published
            ),
        }
    )
    output_coverage_body = dict(
        _dense_axis_coverage(candidate_output, role="out_proj.weight")
    )
    output_coverage = MappingProxyType(
        {
            **output_coverage_body,
            "repaired_element_count": 0,
            "source_prior_plus_fit_rowspace_correction_preserved": True,
        }
    )
    published_weights = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": candidate_value,
        "out_proj.weight": candidate_output,
    }
    published_coverage = {
        role: dict(_dense_axis_coverage(weight, role=role))
        for role, weight in published_weights.items()
    }
    if not all(
        receipt["structurally_nonzero"] is True
        for receipt in published_coverage.values()
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "FIT-selected attention cold-start lost complete Q/K/V/O axis coverage."
        )
    for role in zero_roles:
        if _exact_zero(published_weights[role]) or torch.equal(
            published_weights[role],
            entry_weights[role],
        ):
            raise Lfm2SemanticAttentionRefoldError(
                f"Exact-zero attention role {role} remained an incumbent no-op."
            )
    frozen_sha = canonical_sha256(
        {role: _tensor_sha256(weight) for role, weight in published_weights.items()}
    )
    if (
        canonical_sha256(
            {role: _tensor_sha256(weight) for role, weight in published_weights.items()}
        )
        != frozen_sha
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Final cold-start candidate changed after held-out diagnostics."
        )
    selected_report = vo_gate["candidate_reports"][published_vo_kind]
    partition_diagnostics = {
        partition: {
            **dict(selected_report["partitions"][partition]),
            "used_for_plant_or_selection": partition == "fit",
            "can_restore_exact_zero_incumbent": False,
        }
        for partition in _PARTITIONS
    }
    output_report = vo_gate["candidate_reports"]["fixed_v_output_only"]
    writer_gate = _seal_writer_gate(
        {
            "status": (
                "published-as-fit-only-cold-start-component"
                if writer_component_published
                else (
                    "fit-eligible-not-selected"
                    if writer_proposal.changed
                    else "structurally-identical-to-incumbent"
                )
            ),
            "accepted": writer_component_published,
            "proposal": writer_proposal.as_dict(),
            "rejection_partition": None,
            "fit_check": output_report["partitions"]["fit_check"],
            "selection": output_report["partitions"]["selection"],
            "validation": vo_gate["validation"],
            "candidate_frozen_across_heldout_gates": True,
            "competed_against_incumbent_and_joint_value_output": True,
            "heldout_metrics_are_diagnostics_not_publication_gates": True,
            "heldout_metrics_are_diagnostic_only": True,
            "numeric_metric_used_for_publication": False,
        }
    )
    body = {
        "schema": "neural-forge-lfm2-semantic-complete-attention-cold-start-v1",
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "original_seed_sha256": incumbent_seed.seed_sha256,
        "route_seed_sha256": seed.seed_sha256,
        "exact_zero_roles_at_entry": list(zero_roles),
        "entry_role_coverage": entry_coverage,
        "qk_cold_start": dict(qk_cold_start),
        "joint_value_output_proposal": joint_proposal.as_dict(),
        "output_writer_proposal": writer_proposal.as_dict(),
        "value_axis_coverage": dict(value_coverage),
        "preliminary_output_axis_coverage": dict(preliminary_output_coverage),
        "output_axis_coverage": dict(output_coverage),
        "published_role_coverage": published_coverage,
        "partition_diagnostics": partition_diagnostics,
        "frozen_candidate_sha256": frozen_sha,
        "candidate_frozen_before_fit_check_selection_validation": True,
        "fit_partition_only_used_for_plant": True,
        "fit_check_used_for_plant_or_selection": False,
        "selection_used_for_plant_or_selection": False,
        "validation_used_for_plant_or_selection": False,
        "heldout_metrics_are_diagnostics_not_publication_gates": True,
        "heldout_regression_can_restore_blank_incumbent": False,
        "zero_role_incumbent_noop_allowed": False,
        "all_qkvo_roles_finite_and_structurally_covered": True,
        "dense_existing_qkvo_topology_only": True,
        "randomness_used": False,
        "autograd_used": False,
        "backward_used": False,
        "optimizer_used": False,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
    }
    return MappingProxyType(
        {
            "query_weight": frozen_query,
            "key_weight": frozen_key,
            "value_weight": candidate_value,
            "output_weight": candidate_output,
            "seed": seed,
            "baseline_selection": baseline_selection,
            "candidate_selection": candidate_selection,
            "baseline_validation": baseline_validation,
            "candidate_validation": candidate_validation,
            "joint_banks": joint_banks,
            "writer_banks": writer_banks,
            "writer_proposal": writer_proposal,
            "joint_proposal": joint_proposal,
            "vo_gate": vo_gate,
            "writer_gate": writer_gate,
            "published_vo_kind": published_vo_kind,
            "writer_component_published": writer_component_published,
            "joint_component_published": joint_component_published,
            "cold_start_report": MappingProxyType(
                {**body, "cold_start_sha256": canonical_sha256(body)}
            ),
        }
    )


def _parameter_bundle(target_model: Any, layer_index: int) -> Mapping[str, Any]:
    torch = _torch()
    attention = _attention_module(target_model, layer_index)
    prefix = f"model.layers.{layer_index}.self_attn."
    values = {
        prefix + name + ".weight": getattr(getattr(attention, name), "weight", None)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    }
    values.update(
        {
            prefix + name + ".weight": getattr(
                getattr(attention, name),
                "weight",
                None,
            )
            for name in ("q_layernorm", "k_layernorm")
        }
    )
    named = dict(target_model.named_parameters())
    if (
        any(not isinstance(value, torch.nn.Parameter) for value in values.values())
        or any(named.get(name) is not value for name, value in values.items())
        or len({id(value) for value in values.values()}) != 6
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "Target attention Q/K/V/O/norm Parameters are missing, aliased, or renamed."
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
        "q_layernorm.weight": state.get(
            "q_layernorm.weight",
            parameters[f"model.layers.{layer_index}.self_attn.q_layernorm.weight"]
            .detach()
            .cpu(),
        ),
        "k_layernorm.weight": state.get(
            "k_layernorm.weight",
            parameters[f"model.layers.{layer_index}.self_attn.k_layernorm.weight"]
            .detach()
            .cpu(),
        ),
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
    parameter_shapes = {name: tuple(value.shape) for name, value in parameters.items()}
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
                bool(parameters[name].requires_grad) == parameter_requires_grad[name]
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
    # Installation receipts are consumed alongside state-dict diffs, whose
    # keys are canonicalized lexically by the layer-stack transaction.  Keep
    # this local mutation inventory in the same deterministic order so a
    # multi-projection Q/K/V/O install cannot be rejected solely because the
    # module registration order differs from the state-dict audit order.
    expected_changed = tuple(
        sorted(
            name
            for name in parameters
            if not torch.equal(candidates[name], snapshots[name])
        )
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
        if any(
            _tensor_sha256(parameters[name]) != _tensor_sha256(candidates[name])
            for name in parameters
        ):
            raise Lfm2SemanticAttentionRefoldError(
                "Attention installation did not publish the exact frozen candidate."
            )
        changed = tuple(
            sorted(
                name
                for name, parameter in parameters.items()
                if not torch.equal(parameter.detach().cpu(), snapshots[name])
            )
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


def _publish_complete_attention_cold_start_unprotected(
    *,
    target_model: Any,
    layer_index: int,
    raw_evidence: Sequence[Any],
    cache_key: str,
    bridge_artifact_sha256: str,
    inventory: Lfm2DenseQkvoSourceInventory,
    compiler_lineage_sha256: str,
    split_sha256: str,
    partitions: Mapping[str, Sequence[_CaseSource]],
    balance_receipts: Mapping[str, Mapping[str, Any]],
    banks: Mapping[str, Lfm2EndpointQkResidualEvidenceBank],
    policy: Lfm2EndpointQkResidualFoldPolicy,
    incumbent_seed: Lfm2EndpointQkvoSeed,
    exact_zero_roles: Sequence[str],
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Mapping[str, Any]:
    """Install the compulsory blank-entry plant without any held-out rollback."""

    torch = _torch()
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    cold = _compile_complete_attention_cold_start(
        incumbent_seed,
        banks=banks,
        policy=policy,
        exact_zero_roles=exact_zero_roles,
        body_recipe=recipe,
    )
    frozen_query = cold["query_weight"]
    frozen_key = cold["key_weight"]
    published_value = cold["value_weight"]
    published_output = cold["output_weight"]
    query_changed = not torch.equal(frozen_query, incumbent_seed.query_weight)
    key_changed = not torch.equal(frozen_key, incumbent_seed.key_weight)
    value_changed = not torch.equal(published_value, incumbent_seed.value_weight)
    output_changed = not torch.equal(published_output, incumbent_seed.output_weight)
    changed_roles = {
        role
        for role, changed in (
            ("q_proj.weight", query_changed),
            ("k_proj.weight", key_changed),
            ("v_proj.weight", value_changed),
            ("out_proj.weight", output_changed),
        )
        if changed
    }
    if any(role not in changed_roles for role in exact_zero_roles):
        raise Lfm2SemanticAttentionRefoldError(
            "A zero attention incumbent survived the compulsory cold-start."
        )
    state = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": published_value,
        "out_proj.weight": published_output,
        "q_layernorm.weight": cold["seed"].query_norm_weight,
        "k_layernorm.weight": cold["seed"].key_norm_weight,
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
        evidence_sha256=[item.evidence_sha256 for item in raw_evidence],
        model_state_before_sha256=before,
        model_state_after_sha256=after,
        changed_state_keys=changed,
        parameter_element_count=parameter_count,
        target_topology_sha256_before=topology_before,
        target_topology_sha256_after=topology_after,
        structurally_eligible_candidate_count=(
            1 + int(cold["vo_gate"]["structurally_eligible_candidate_count"])
        ),
        body_recipe=recipe,
    )
    qk_cold = cold["cold_start_report"]["qk_cold_start"]
    joint_proposal = cold["joint_proposal"]
    frozen_qk_sha = canonical_sha256(
        {
            "q": _tensor_sha256(frozen_query),
            "k": _tensor_sha256(frozen_key),
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
    frozen_attention_parameter_bundle = {
        "q_proj.weight": _tensor_sha256(frozen_query),
        "k_proj.weight": _tensor_sha256(frozen_key),
        "v_proj.weight": _tensor_sha256(published_value),
        "out_proj.weight": _tensor_sha256(published_output),
        "q_layernorm.weight": _tensor_sha256(cold["seed"].query_norm_weight),
        "k_layernorm.weight": _tensor_sha256(cold["seed"].key_norm_weight),
    }
    frozen_attention_parameter_bundle_sha = canonical_sha256(
        frozen_attention_parameter_bundle
    )
    heldout_nonregression = {
        partition: bool(
            qk_cold["partitions"][partition]["nonregressing"]
            and cold["cold_start_report"]["partition_diagnostics"][partition][
                "nonregressing"
            ]
        )
        for partition in ("fit_check", "selection", "validation")
    }
    joint_component_contributed = bool(cold["joint_component_published"])
    writer_component_contributed = bool(cold["writer_component_published"])
    output_axis_completion_performed = bool(
        cold["cold_start_report"]["output_axis_coverage"]["repaired_element_count"]
    )
    joint_candidate_published_exactly = bool(
        joint_component_contributed
        and _tensor_sha256(joint_proposal.value_weight)
        == _tensor_sha256(published_value)
        and _tensor_sha256(joint_proposal.output_weight)
        == _tensor_sha256(published_output)
    )
    if joint_component_contributed:
        output_source = "fit_only_joint_value_output_residual_factorization"
    elif writer_component_contributed:
        output_source = "fit_only_fixed_v_prior_preserving_output_residual_ridge"
    else:
        output_source = "authenticated_parent_derived_output_prior"
    body = {
        **base,
        "fit_check_nonregression_passed": heldout_nonregression["fit_check"],
        "selection_nonregression_passed": heldout_nonregression["selection"],
        "validation_nonregression_passed": heldout_nonregression["validation"],
        "heldout_nonregression_is_diagnostic_only": True,
        "analytic_refold": {
            "schema": LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA,
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_qk_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
            "qk_component_selection_policy_equivalent_recipe_ids": list(
                lfm2_semantic_body_recipe_ids_for_component_lane(
                    "qk",
                    recipe.component_lanes["qk"],
                )
            ),
            "qk_component_selection_policy_equivalent_to_other_recipe": (
                len(
                    lfm2_semantic_body_recipe_ids_for_component_lane(
                        "qk",
                        recipe.component_lanes["qk"],
                    )
                )
                > 1
            ),
            "qk_component_equivalence_does_not_claim_complete_body_equivalence": (True),
            "cache_key": cache_key,
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "source_inventory_sha256": inventory.inventory_sha256,
            "compiler_lineage_sha256": compiler_lineage_sha256,
            "split_sha256": split_sha256,
            "partition_batch_indices": {
                name: sorted({item.batch_index for item in partitions[name]})
                for name in _PARTITIONS
            },
            "trajectory_row_weight_balance": {
                name: dict(balance_receipts[name]) for name in _PARTITIONS
            },
            "fit_evidence_sha256": banks["fit"].evidence_sha256,
            "fit_check_evidence_sha256": banks["fit_check"].evidence_sha256,
            "selection_evidence_sha256": banks["selection"].evidence_sha256,
            "validation_evidence_sha256": banks["validation"].evidence_sha256,
            "qk_cold_start": dict(qk_cold),
            "complete_attention_cold_start": dict(cold["cold_start_report"]),
            "endpoint_compiler_report_sha256": qk_cold["plant"]["plant_sha256"],
            "endpoint_compiler_bypassed_for_fit_only_cold_start": True,
            "analytic_execution_policy": {
                "solver_device": policy.solver_device,
                "compute_devices": list(policy.compute_devices),
                "native_bfloat16_sm80_or_newer_only": True,
                "non_native_bfloat16_devices_used": False,
                "candidate_endpoint_replay_follows_compute_devices": True,
            },
            "selection_baseline": cold["baseline_selection"].as_dict(),
            "selection_candidate": cold["candidate_selection"].as_dict(),
            "validation_baseline": cold["baseline_validation"].as_dict(),
            "validation_candidate": cold["candidate_validation"].as_dict(),
            "frozen_qk_sha256": frozen_qk_sha,
            "frozen_qkvo_sha256": frozen_qkvo_sha,
            "frozen_attention_six_tensor_bundle_sha256": (
                frozen_attention_parameter_bundle_sha
            ),
            "frozen_attention_six_tensor_bundle": dict(
                frozen_attention_parameter_bundle
            ),
            "frozen_attention_six_tensor_bundle_roles": list(
                frozen_attention_parameter_bundle
            ),
            "candidate_frozen_before_validation": True,
            "qk_candidate_frozen_before_validation": True,
            "output_writer_candidate_frozen_before_validation": True,
            "joint_value_output_candidate_frozen_before_validation": True,
            "validation_triggered_refit": False,
            "heldout_metrics_are_diagnostics_not_publication_gates": True,
            "heldout_metrics_are_diagnostic_only": True,
            "numeric_metric_used_for_publication_or_rollback": False,
            "joint_value_output_candidate_diagnostic_only": False,
            "joint_value_output_publication_allowed": True,
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
            "qk_compiler_accepted": False,
            "qk_compiler_changed": False,
            "authenticated_qk_seed_fallback_entered_vo_competition": False,
            "fixed_value_source": (
                "fit_only_joint_value_output_residual_factorization"
                if joint_component_contributed
                else "authenticated_parent_derived_fixed_value_prior"
            ),
            "output_writer_source": output_source,
            "output_writer": dict(cold["writer_gate"]),
            "joint_value_output": joint_proposal.as_dict(),
            "value_output_candidate_gate": dict(cold["vo_gate"]),
            "output_writer_cached_qk_probability_replay_calls": sum(
                len(bank) for bank in cold["joint_banks"].values()
            ),
            "output_writer_model_forward_calls": 0,
            "output_writer_fit_partition_only_used_for_solve": True,
            "output_writer_heldout_partitions_never_used_for_solve": True,
            "joint_value_output_candidate_published": (
                joint_candidate_published_exactly
            ),
            "joint_value_output_component_contributed": (joint_component_contributed),
            "fresh_joint_value_output_refit_performed": (joint_component_contributed),
            "fresh_value_refit_performed": joint_component_contributed,
            "source_derived_value_prior_installed": bool(
                value_changed and not joint_component_contributed
            ),
            "fresh_output_writer_refit_performed": writer_component_contributed,
            "fresh_output_projection_refit_performed": output_changed,
            "output_axis_completion_performed": output_axis_completion_performed,
            "fresh_query_refit_performed": query_changed,
            "fresh_key_refit_performed": key_changed,
            "fresh_four_matrix_qkvo_refit_claimed": bool(False),
            "complete_qkvo_bundle_installed_transactionally": True,
            "complete_qkvo_bundle_transaction_verified": True,
            "authenticated_incumbent_qkvo_retained_without_installation": False,
            "projection_parameter_copy_calls": len(changed),
            "complete_causal_cache_only": True,
            "contractions_supported": False,
            "query_supervision_mode": (
                "all_semantically_exact_query_endpoints_per_semantic_phase"
            ),
            "all_semantically_exact_endpoint_rows_used_for_attention_fit": True,
            "terminal_endpoint_only_supervision": False,
            "analytic_fp64_target_construction_distinct_from_bfloat16_runtime_replay": True,
            "runtime_replay_numeric_contract": (
                "bfloat16-linears-fp32-qk-softmax-bfloat16-probability-context"
            ),
            "cross_head_dimension_bridge_deferred": True,
            "refold_operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        },
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _publish_complete_attention_cold_start(
    *,
    target_model: Any,
    layer_index: int,
    raw_evidence: Sequence[Any],
    cache_key: str,
    bridge_artifact_sha256: str,
    inventory: Lfm2DenseQkvoSourceInventory,
    compiler_lineage_sha256: str,
    split_sha256: str,
    partitions: Mapping[str, Sequence[_CaseSource]],
    balance_receipts: Mapping[str, Mapping[str, Any]],
    banks: Mapping[str, Lfm2EndpointQkResidualEvidenceBank],
    policy: Lfm2EndpointQkResidualFoldPolicy,
    incumbent_seed: Lfm2EndpointQkvoSeed,
    exact_zero_roles: Sequence[str],
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Mapping[str, Any]:
    """Keep the blank-entry transaction atomic through receipt sealing."""

    torch = _torch()
    parameters = _parameter_bundle(target_model, layer_index)
    snapshots = {
        name: parameter.detach().to(device="cpu").contiguous().clone()
        for name, parameter in parameters.items()
    }
    parameter_ids = {name: id(parameter) for name, parameter in parameters.items()}
    parameter_pointers = {
        name: int(parameter.data_ptr()) for name, parameter in parameters.items()
    }
    parameter_shapes = {
        name: tuple(parameter.shape) for name, parameter in parameters.items()
    }
    parameter_dtypes = {name: parameter.dtype for name, parameter in parameters.items()}
    parameter_requires_grad = {
        name: bool(parameter.requires_grad) for name, parameter in parameters.items()
    }
    gradient_ids = {
        name: None if parameter.grad is None else id(parameter.grad)
        for name, parameter in parameters.items()
    }
    gradient_hashes = {
        name: None if parameter.grad is None else _tensor_sha256(parameter.grad)
        for name, parameter in parameters.items()
    }
    model_state_before = fingerprint_lfm2_parent(target_model).identity_sha256
    topology_before = _topology_sha256(target_model)
    parameter_count_before = _parameter_count(target_model)

    def exact_outer_state_restored() -> bool:
        live = _parameter_bundle(target_model, layer_index)
        return bool(
            fingerprint_lfm2_parent(target_model).identity_sha256 == model_state_before
            and _topology_sha256(target_model) == topology_before
            and _parameter_count(target_model) == parameter_count_before
            and all(live[name] is parameters[name] for name in parameters)
            and all(id(live[name]) == parameter_ids[name] for name in parameters)
            and all(
                int(live[name].data_ptr()) == parameter_pointers[name]
                for name in parameters
            )
            and all(
                tuple(live[name].shape) == parameter_shapes[name] for name in parameters
            )
            and all(live[name].dtype == parameter_dtypes[name] for name in parameters)
            and all(
                bool(live[name].requires_grad) == parameter_requires_grad[name]
                for name in parameters
            )
            and all(
                (None if live[name].grad is None else id(live[name].grad))
                == gradient_ids[name]
                for name in parameters
            )
            and all(
                live[name].grad is None
                or _tensor_sha256(live[name].grad) == gradient_hashes[name]
                for name in parameters
            )
            and all(
                _tensor_sha256(live[name]) == _tensor_sha256(snapshots[name])
                for name in parameters
            )
        )

    try:
        result = _publish_complete_attention_cold_start_unprotected(
            target_model=target_model,
            layer_index=layer_index,
            raw_evidence=raw_evidence,
            cache_key=cache_key,
            bridge_artifact_sha256=bridge_artifact_sha256,
            inventory=inventory,
            compiler_lineage_sha256=compiler_lineage_sha256,
            split_sha256=split_sha256,
            partitions=partitions,
            balance_receipts=balance_receipts,
            banks=banks,
            policy=policy,
            incumbent_seed=incumbent_seed,
            exact_zero_roles=exact_zero_roles,
            body_recipe=body_recipe,
        )
        if not isinstance(result, Mapping):
            raise Lfm2SemanticAttentionRefoldError(
                "Complete attention cold-start returned no sealed receipt."
            )
        sealed = dict(result)
        result_sha256 = sealed.pop("result_sha256", None)
        model_state_after = fingerprint_lfm2_parent(target_model).identity_sha256
        topology_after = _topology_sha256(target_model)
        actual_changed = tuple(
            sorted(
                name
                for name, parameter in parameters.items()
                if _tensor_sha256(parameter) != _tensor_sha256(snapshots[name])
            )
        )
        if (
            result.get("schema") != _INSTALLATION_SCHEMA
            or result_sha256 != canonical_sha256(sealed)
            or result.get("model_state_before_sha256") != model_state_before
            or result.get("model_state_after_sha256") != model_state_after
            or result.get("changed_state_keys") != list(actual_changed)
            or not actual_changed
            or result.get("installed") is not True
            or result.get("install_count") != 1
            or result.get("parameter_element_count_before") != parameter_count_before
            or result.get("parameter_element_count_after") != parameter_count_before
            or result.get("parameter_count_changed") is not False
            or result.get("target_topology_sha256_before") != topology_before
            or result.get("target_topology_sha256_after") != topology_after
            or topology_after != topology_before
            or result.get("topology_changed") is not False
            or _parameter_count(target_model) != parameter_count_before
            or any(id(parameters[name]) != parameter_ids[name] for name in parameters)
            or any(
                int(parameters[name].data_ptr()) != parameter_pointers[name]
                for name in parameters
            )
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
                "Complete attention cold-start receipt is stale or unsealed."
            )
        return result
    except BaseException as exc:
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(
                        snapshots[name].to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )
            if not exact_outer_state_restored():
                raise Lfm2SemanticAttentionRefoldError(
                    "Complete attention cold-start outer rollback was not exact."
                )
        except BaseException as rollback_exc:
            raise Lfm2SemanticAttentionRefoldError(
                "Complete attention cold-start failed and exact outer rollback "
                "was not proven."
            ) from rollback_exc
        raise exc


def compile_install_lfm2_semantic_attention_postfold(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Any],
    cache_key: str,
    bridge_artifact_sha256: str,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Mapping[str, Any]:
    """Compile and atomically install one semantic long-horizon Q/K/V/O fold."""

    torch = _torch()
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
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
    source_contract = _source_contract(raw[0])
    (
        inventory,
        groups,
        weights,
        _key_value_groups,
        _key_value_weights,
        lineage,
    ) = source_contract
    if any(_source_contract(item) != source_contract for item in raw[1:]):
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
    policy = _qk_policy_for_body_recipe(
        _execution_policy(len(normalized)),
        recipe,
    )
    banks = MappingProxyType(
        {
            "fit": fit_bank,
            "fit_check": fit_check_bank,
            "selection": selection_bank,
            "validation": validation_bank,
        }
    )
    exact_zero_roles = tuple(
        role
        for role, weight in (
            ("q_proj.weight", incumbent_seed.query_weight),
            ("k_proj.weight", incumbent_seed.key_weight),
            ("v_proj.weight", incumbent_seed.value_weight),
            ("out_proj.weight", incumbent_seed.output_weight),
        )
        if _exact_zero(weight)
    )
    if exact_zero_roles:
        return _publish_complete_attention_cold_start(
            target_model=target_model,
            layer_index=layer_index,
            raw_evidence=raw,
            cache_key=cache,
            bridge_artifact_sha256=bridge,
            inventory=inventory,
            compiler_lineage_sha256=lineage,
            split_sha256=split_sha,
            partitions=partitions,
            balance_receipts=balance_receipts,
            banks=banks,
            policy=policy,
            incumbent_seed=incumbent_seed,
            exact_zero_roles=exact_zero_roles,
            body_recipe=recipe,
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
    selection_nonregression_diagnostic = _metrics_nonregressing(
        candidate_selection,
        baseline_selection,
        tolerance=tolerance,
    )
    frozen_query = compiled.query_weight.detach().cpu().clone()
    frozen_key = compiled.key_weight.detach().cpu().clone()
    compiler_query_changed = not torch.equal(frozen_query, seed.query_weight)
    compiler_key_changed = not torch.equal(frozen_key, seed.key_weight)
    if compiled.changed != (
        compiler_query_changed or compiler_key_changed
    ) or compiled.accepted != (compiler_query_changed or compiler_key_changed):
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
    validation_nonregression_diagnostic = _metrics_nonregressing(
        candidate_validation,
        baseline_validation,
        tolerance=tolerance,
    )
    if _tensor_sha256(frozen_query) != _tensor_sha256(
        compiled.query_weight
    ) or _tensor_sha256(frozen_key) != _tensor_sha256(compiled.key_weight):
        raise Lfm2SemanticAttentionRefoldError(
            "Frozen Q/K candidate changed after its deterministic FIT compilation."
        )
    if cold_start is not None and (
        (cold_start["original_query_exact_zero"] is True and _exact_zero(frozen_query))
        or (cold_start["original_key_exact_zero"] is True and _exact_zero(frozen_key))
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
        body_recipe=recipe,
    )
    published_vo_kind = str(vo_gate["published_candidate"])
    if published_vo_kind not in {
        "incumbent",
        "fixed_v_output_only",
        "joint_value_output",
    }:
        raise Lfm2SemanticAttentionRefoldError(
            "V/O publication selected an unknown structural candidate."
        )
    writer_accepted = published_vo_kind == "fixed_v_output_only"
    joint_accepted = published_vo_kind == "joint_value_output"
    value_changed = not torch.equal(published_value, incumbent_seed.value_weight)
    output_changed = not torch.equal(published_output, incumbent_seed.output_weight)
    expected_value = (
        joint_proposal.value_weight if joint_accepted else seed.value_weight
    )
    expected_output = (
        joint_proposal.output_weight
        if joint_accepted
        else (writer_proposal.output_weight if writer_accepted else seed.output_weight)
    )
    if not torch.equal(published_value, expected_value) or not torch.equal(
        published_output,
        expected_output,
    ):
        raise Lfm2SemanticAttentionRefoldError(
            "V/O candidate gate acceptance disagrees with its published state."
        )
    output_report = vo_gate["candidate_reports"]["fixed_v_output_only"]
    writer_gate = _seal_writer_gate(
        {
            "status": (
                "published-deterministic-fit-only-correction"
                if writer_accepted
                else (
                    "fit-eligible-not-selected"
                    if writer_proposal.changed
                    else "structurally-identical-to-incumbent"
                )
            ),
            "accepted": writer_accepted,
            "proposal": writer_proposal.as_dict(),
            "rejection_partition": None,
            "fit_check": output_report["partitions"]["fit_check"],
            "selection": output_report["partitions"]["selection"],
            "validation": vo_gate["validation"],
            "candidate_frozen_across_heldout_gates": True,
            "competed_against_incumbent_and_joint_value_output": True,
            "heldout_metrics_are_diagnostic_only": True,
            "numeric_metric_used_for_publication": False,
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
    frozen_attention_parameter_bundle = {
        "q_proj.weight": _tensor_sha256(frozen_query),
        "k_proj.weight": _tensor_sha256(frozen_key),
        "v_proj.weight": _tensor_sha256(published_value),
        "out_proj.weight": _tensor_sha256(published_output),
        "q_layernorm.weight": _tensor_sha256(seed.query_norm_weight),
        "k_layernorm.weight": _tensor_sha256(seed.key_norm_weight),
    }
    frozen_attention_parameter_bundle_sha = canonical_sha256(
        frozen_attention_parameter_bundle
    )
    state = {
        "q_proj.weight": frozen_query,
        "k_proj.weight": frozen_key,
        "v_proj.weight": published_value,
        "out_proj.weight": published_output,
        "q_layernorm.weight": seed.query_norm_weight,
        "k_layernorm.weight": seed.key_norm_weight,
    }
    topology_before = _topology_sha256(target_model)
    before, after, changed, parameter_count = _install_qkvo_transaction(
        target_model,
        layer_index=layer_index,
        state=state,
    )
    topology_after = _topology_sha256(target_model)
    structurally_eligible_candidate_count = int(query_changed or key_changed) + int(
        vo_gate["structurally_eligible_candidate_count"]
    )
    base = _installation_receipt_body(
        layer_index=layer_index,
        evidence_sha256=[item.evidence_sha256 for item in raw],
        model_state_before_sha256=before,
        model_state_after_sha256=after,
        changed_state_keys=changed,
        parameter_element_count=parameter_count,
        target_topology_sha256_before=topology_before,
        target_topology_sha256_after=topology_after,
        structurally_eligible_candidate_count=(structurally_eligible_candidate_count),
        body_recipe=recipe,
    )
    body = {
        **base,
        "analytic_refold": {
            "schema": LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA,
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_qk_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
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
            **({} if cold_start is None else {"qk_cold_start": dict(cold_start)}),
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
            "selection_nonregression_diagnostic": (selection_nonregression_diagnostic),
            "validation_nonregression_diagnostic": (
                validation_nonregression_diagnostic
            ),
            "heldout_metrics_are_diagnostic_only": True,
            "numeric_metric_used_for_publication_or_rollback": False,
            "frozen_qk_sha256": frozen_qk_sha,
            "frozen_qkvo_sha256": frozen_qkvo_sha,
            "frozen_attention_six_tensor_bundle_sha256": (
                frozen_attention_parameter_bundle_sha
            ),
            "frozen_attention_six_tensor_bundle": dict(
                frozen_attention_parameter_bundle
            ),
            "frozen_attention_six_tensor_bundle_roles": list(
                frozen_attention_parameter_bundle
            ),
            "candidate_frozen_before_validation": True,
            "qk_candidate_frozen_before_validation": True,
            "output_writer_candidate_frozen_before_validation": (
                writer_proposal.changed
            ),
            "joint_value_output_candidate_frozen_before_validation": (
                joint_proposal.changed
            ),
            "joint_value_output_candidate_diagnostic_only": False,
            "joint_value_output_publication_allowed": True,
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
                "fit_only_joint_value_output_residual_factorization"
                if joint_accepted
                else "authenticated_live_c61_dense_seed_already_folded_on_broad_cache"
            ),
            "output_writer_source": (
                "fit_only_joint_value_output_residual_factorization"
                if joint_accepted
                else (
                    "fit_only_fixed_v_prior_preserving_output_residual_ridge"
                    if writer_accepted
                    else "authenticated_live_c61_seed_after_structural_noop"
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
            "fresh_joint_value_output_refit_performed": joint_accepted,
            "fresh_value_refit_performed": value_changed,
            "fresh_output_writer_refit_performed": writer_accepted,
            "fresh_output_projection_refit_performed": output_changed,
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
            "authenticated_incumbent_qkvo_retained_without_installation": (not changed),
            "projection_parameter_copy_calls": len(changed),
            "complete_causal_cache_only": True,
            "contractions_supported": False,
            "query_supervision_mode": (
                "all_semantically_exact_query_endpoints_per_semantic_phase"
            ),
            "all_semantically_exact_endpoint_rows_used_for_attention_fit": True,
            "terminal_endpoint_only_supervision": False,
            "analytic_fp64_target_construction_distinct_from_bfloat16_runtime_replay": True,
            "runtime_replay_numeric_contract": (
                "bfloat16-linears-fp32-qk-softmax-bfloat16-probability-context"
            ),
            "cross_head_dimension_bridge_deferred": True,
            "refold_operation_audit": dict(_ZERO_REFOLD_OPERATIONS),
        },
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


__all__ = [
    "LFM2_SEMANTIC_ATTENTION_REFOLD_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_REPLAY_ENDPOINT_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS",
    "Lfm2SemanticAttentionRefoldError",
    "Lfm2SemanticAttentionReplayEndpoint",
    "compile_install_lfm2_semantic_attention_postfold",
]
