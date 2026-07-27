"""Causal target-lattice Q/K closure for an already folded LFM2 body.

This stage is deliberately factor isolated.  It consumes the immutable
semantic cache, selects the exact source query heads previously planted by the
intact-group Q-DEIM fold, and analytically adjusts only the saved child's Q/K
weights.  Child V/O and both Q/K norms define the endpoint teacher and remain
byte-identical.  Layers are replayed in causal order, so every later layer is
compiled from residuals produced by the already-treated prefix.

No parent model or parent weight tensor is opened, no model ``forward`` method
is called, and no gradient, optimizer, training, pruning, or quantization path
exists here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2DenseQkvoSourceInventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
    Lfm2SemanticAttentionReplayEndpoint,
    compile_install_lfm2_semantic_target_lattice_qk_only,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    _WholeModelSnapshot,
    _initial_residual_bank,
    _next_residual_bank,
    _replay_current_layer,
    _validated_operator_topology,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    Lfm2SemanticAttentionTransportEvidence,
    Lfm2SemanticCompilerLayerLineage,
    _attention_tensor_names,
    _changed_state_keys,
    _parameter_count,
    _semantic_alignments_from_session,
    _state_tensor_hashes,
    _target_rope,
    _tensor_sha256,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    audit_lfm2_semantic_span_cpcf_refinement,
)
from neural_forge.hf_semantic_attention_lattice import (
    Lfm2SemanticAttentionCacheCapability,
    SemanticAttentionPreRopeHistory,
    assemble_lfm2_explicit_pre_rope_history,
    inspect_lfm2_semantic_attention_cache_capability,
    select_semantic_attention_expansion_query_heads,
    transport_lfm2_phase_probability_preserving_attention,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA = (
    "neural-forge-lfm2-saved-body-target-lattice-qk-closure-v1"
)
LFM2_QDEIM_HEAD_SELECTION_SCHEMA = (
    "neural-forge-lfm2-saved-body-qdeim-head-selection-v1"
)
_ZERO_FORBIDDEN_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_weight_tensor_loads": 0,
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


class Lfm2SavedBodyTargetLatticeQkError(RuntimeError):
    """Saved-body Q/K closure evidence or state transition failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Saved-body target-lattice Q/K closure requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _verify_sealed(
    value: Any,
    *,
    seal_name: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SavedBodyTargetLatticeQkError(f"{label} is not a mapping.")
    body = dict(value)
    supplied = body.pop(seal_name, None)
    if supplied != canonical_sha256(body):
        raise Lfm2SavedBodyTargetLatticeQkError(f"{label} seal is invalid.")
    return {**body, seal_name: supplied}


@dataclass(frozen=True, slots=True)
class Lfm2QdeimHeadSelection:
    """Authenticated whole-GQA-group ordering planted into one target layer."""

    layer_index: int
    source_query_head_count: int
    source_key_value_head_count: int
    target_query_head_count: int
    target_key_value_head_count: int
    selected_source_key_value_groups: tuple[int, ...]
    selected_source_query_heads: tuple[int, ...]
    source_receipt_sha256: str
    selection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        integers = (
            self.layer_index,
            self.source_query_head_count,
            self.source_key_value_head_count,
            self.target_query_head_count,
            self.target_key_value_head_count,
            *self.selected_source_key_value_groups,
            *self.selected_source_query_heads,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Q-DEIM head-selection coordinates must be integers."
            )
        if (
            self.layer_index < 0
            or self.source_query_head_count <= 0
            or self.source_key_value_head_count <= 0
            or self.target_query_head_count <= 0
            or self.target_key_value_head_count <= 0
            or self.source_query_head_count % self.source_key_value_head_count
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Q-DEIM head-selection topology is invalid."
            )
        selected_groups = tuple(self.selected_source_key_value_groups)
        selected_queries = tuple(self.selected_source_query_heads)
        queries_per_group = (
            self.source_query_head_count // self.source_key_value_head_count
        )
        expected_queries = tuple(
            group * queries_per_group + offset
            for group in selected_groups
            for offset in range(queries_per_group)
        )
        if (
            len(selected_groups) != self.target_key_value_head_count
            or len(set(selected_groups)) != len(selected_groups)
            or any(
                value < 0 or value >= self.source_key_value_head_count
                for value in selected_groups
            )
            or selected_queries != expected_queries
            or len(selected_queries) != self.target_query_head_count
            or len(set(selected_queries)) != len(selected_queries)
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Q-DEIM selection is not one ordered set of intact source GQA groups."
            )
        _digest(self.source_receipt_sha256, "Q-DEIM source receipt SHA-256")
        object.__setattr__(
            self,
            "selection_sha256",
            canonical_sha256(self.authentication_body()),
        )

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_QDEIM_HEAD_SELECTION_SCHEMA,
            "layer_index": self.layer_index,
            "source_query_head_count": self.source_query_head_count,
            "source_key_value_head_count": self.source_key_value_head_count,
            "target_query_head_count": self.target_query_head_count,
            "target_key_value_head_count": self.target_key_value_head_count,
            "selected_source_key_value_groups": list(
                self.selected_source_key_value_groups
            ),
            "selected_source_query_heads": list(self.selected_source_query_heads),
            "source_receipt_sha256": self.source_receipt_sha256,
            "whole_gqa_groups_only": True,
            "head_averaging_performed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.authentication_body(), "selection_sha256": self.selection_sha256}


def extract_lfm2_qdeim_head_selections(
    semantic_layer_stack_receipt: Mapping[str, Any],
) -> Mapping[int, Lfm2QdeimHeadSelection]:
    """Recover and authenticate planted source-head order from a stack receipt."""

    stack = _verify_sealed(
        semantic_layer_stack_receipt,
        seal_name="result_sha256",
        label="semantic layer-stack receipt",
    )
    raw_layers = stack.get("layer_installations")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Semantic layer-stack receipt has no layer installations."
        )
    selections: dict[int, Lfm2QdeimHeadSelection] = {}
    for raw_layer in raw_layers:
        layer = _verify_sealed(
            raw_layer,
            seal_name="result_sha256",
            label="semantic layer transaction",
        )
        if layer.get("operator_kind") != "full_attention":
            continue
        layer_index = layer.get("layer_index")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index in selections
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Semantic attention layer coordinates are invalid or duplicated."
            )
        operator = _verify_sealed(
            layer.get("attention_operator_installation_receipt"),
            seal_name="result_sha256",
            label=f"layer {layer_index} attention installation",
        )
        # Source-local output folding wraps the original structural Q-DEIM
        # installation so it can replace O without discarding the planted
        # Q/K/V provenance.  The outer composite is still the installed
        # operator, while head topology and selection remain authenticated by
        # its sealed structural child.
        structural = operator
        if "structural_attention_operator_receipt" in operator:
            structural = _verify_sealed(
                operator.get("structural_attention_operator_receipt"),
                seal_name="result_sha256",
                label=f"layer {layer_index} structural Q-DEIM installation",
            )
        adapter = _verify_sealed(
            structural.get("coupled_attention_candidate_receipt"),
            seal_name="result_sha256",
            label=f"layer {layer_index} Q-DEIM candidate",
        )
        compiler = _verify_sealed(
            structural.get("coupled_attention_compiler_receipt"),
            seal_name="result_sha256",
            label=f"layer {layer_index} Q-DEIM compiler",
        )
        topology = compiler.get("topology")
        selection = compiler.get("selection")
        if not isinstance(topology, Mapping) or not isinstance(selection, Mapping):
            raise Lfm2SavedBodyTargetLatticeQkError(
                f"Layer {layer_index} Q-DEIM receipt lacks topology or selection."
            )
        if (
            structural.get("coupled_attention_candidate_receipt") != adapter
            or structural.get("coupled_attention_compiler_receipt") != compiler
            or adapter.get("standalone_compiler_result_sha256")
            != compiler["result_sha256"]
            or adapter.get("selected_source_key_value_groups")
            != selection.get("selected_source_key_value_groups")
            or adapter.get("selected_source_query_heads")
            != selection.get("selected_source_query_heads")
            or selection.get("whole_gqa_groups_only") is not True
            or selection.get("qkv_producer_averaging_performed") is not False
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                f"Layer {layer_index} Q-DEIM head ordering changed after planting."
            )
        try:
            selections[layer_index] = Lfm2QdeimHeadSelection(
                layer_index=layer_index,
                source_query_head_count=topology["source_query_head_count"],
                source_key_value_head_count=(
                    topology["source_key_value_head_count"]
                ),
                target_query_head_count=topology["target_query_head_count"],
                target_key_value_head_count=(
                    topology["target_key_value_head_count"]
                ),
                selected_source_key_value_groups=tuple(
                    adapter["selected_source_key_value_groups"]
                ),
                selected_source_query_heads=tuple(
                    adapter["selected_source_query_heads"]
                ),
                source_receipt_sha256=operator["result_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Lfm2SavedBodyTargetLatticeQkError(
                f"Layer {layer_index} Q-DEIM head-selection receipt is malformed."
            ) from exc
    if not selections:
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Semantic layer-stack receipt contains no coupled attention selection."
        )
    return MappingProxyType(selections)


def _inventory_from_dict(value: Any) -> Lfm2DenseQkvoSourceInventory:
    if not isinstance(value, Mapping):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved attention lineage has no source Q/K/V/O inventory."
        )
    try:
        inventory = Lfm2DenseQkvoSourceInventory(
            tensor_receipts=value.get("tensor_receipts", {}),
            source_state_sha256=value.get("source_state_sha256"),
            structural_source_state_sha256=value.get("structural_source_state_sha256"),
            logical_parameters=value.get("logical_parameters"),
            inventory_sha256=value.get("inventory_sha256", ""),
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved attention lineage source inventory is invalid."
        ) from exc
    if inventory.as_dict() != dict(value):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved attention lineage source inventory is not canonical."
        )
    return inventory


def authenticate_lfm2_saved_attention_lineage(
    value: Mapping[str, Any],
) -> Mapping[int, Lfm2SemanticCompilerLayerLineage]:
    """Rehydrate the exact raw/typed lineage file without parent checkpoint I/O."""

    if not isinstance(value, Mapping) or set(value) != {"raw", "typed"}:
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved attention-lineage document must contain raw and typed maps."
        )
    raw = value["raw"]
    typed = value["typed"]
    if (
        not isinstance(raw, Mapping)
        or not isinstance(typed, Mapping)
        or not raw
        or set(raw) != set(typed)
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved raw and typed attention lineages do not cover the same layers."
        )
    result: dict[int, Lfm2SemanticCompilerLayerLineage] = {}
    try:
        ordered_keys = sorted(raw, key=lambda item: int(item))
    except (TypeError, ValueError) as exc:
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved attention-lineage layer keys must be canonical integers."
        ) from exc
    for key in ordered_keys:
        raw_report = _verify_sealed(
            raw[key],
            seal_name="lineage_sha256",
            label=f"raw attention lineage {key}",
        )
        typed_report = _verify_sealed(
            typed[key],
            seal_name="lineage_sha256",
            label=f"typed attention lineage {key}",
        )
        try:
            layer_index = int(key)
            inventory = _inventory_from_dict(typed_report["source_inventory"])
            lineage = Lfm2SemanticCompilerLayerLineage(
                layer_index=layer_index,
                source_inventory=inventory,
                query_groups=tuple(
                    tuple(group) for group in typed_report["query_groups"]
                ),
                query_group_weights=tuple(
                    tuple(group) for group in typed_report["query_group_weights"]
                ),
                key_value_groups=tuple(
                    tuple(group) for group in typed_report["key_value_groups"]
                ),
                key_value_group_weights=tuple(
                    tuple(group)
                    for group in typed_report["key_value_group_weights"]
                ),
                compiler_lineage_sha256=typed_report["lineage_sha256"],
                report=typed_report,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Lfm2SavedBodyTargetLatticeQkError(
                f"Saved typed attention lineage {key} is malformed."
            ) from exc
        if (
            str(layer_index) != str(key)
            or raw_report.get("layer_index") != layer_index
            or typed_report.get("layer_index") != layer_index
            or typed_report.get("foundation_lineage_sha256")
            != raw_report["lineage_sha256"]
            or raw_report.get("source_inventory") != inventory.as_dict()
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                f"Saved raw and typed attention lineage {key} diverged."
            )
        result[layer_index] = lineage
    return MappingProxyType(result)


def _fixed_value_output_endpoint(
    *,
    target_model: Any,
    layer_index: int,
    target_key_value_inputs: Any,
    probabilities: Any,
) -> Any:
    """Replay fixed child V/O under externally supplied target-lattice routes."""

    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SavedBodyTargetLatticeQkError(
            f"Saved child has no layer {layer_index}."
        )
    attention = getattr(layers[layer_index], "self_attn", None)
    if attention is None:
        raise Lfm2SavedBodyTargetLatticeQkError(
            f"Saved child layer {layer_index} has no dense attention module."
        )
    value_weight = getattr(getattr(attention, "v_proj", None), "weight", None)
    output_weight = getattr(getattr(attention, "out_proj", None), "weight", None)
    head_dim = int(getattr(attention, "head_dim", 0))
    query_heads = (
        0
        if not isinstance(output_weight, torch.nn.Parameter) or head_dim <= 0
        else int(output_weight.shape[1]) // head_dim
    )
    key_value_heads = (
        0
        if not isinstance(value_weight, torch.nn.Parameter) or head_dim <= 0
        else int(value_weight.shape[0]) // head_dim
    )
    supported_dtypes = {torch.bfloat16, torch.float32}
    if (
        not isinstance(value_weight, torch.nn.Parameter)
        or not isinstance(output_weight, torch.nn.Parameter)
        or value_weight.dtype not in supported_dtypes
        or output_weight.dtype != value_weight.dtype
        or head_dim <= 0
        or query_heads <= 0
        or key_value_heads <= 0
        or query_heads % key_value_heads
        or tuple(value_weight.shape)
        != (key_value_heads * head_dim, int(target_key_value_inputs.shape[-1]))
        or tuple(output_weight.shape)
        != (int(target_key_value_inputs.shape[-1]), query_heads * head_dim)
        or tuple(probabilities.shape[0:2])
        != (int(target_key_value_inputs.shape[0]), query_heads)
        or int(probabilities.shape[-1]) != int(target_key_value_inputs.shape[1])
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Fixed child V/O dtype or target-lattice probability geometry diverged."
        )
    device = value_weight.device
    inputs = target_key_value_inputs.to(
        device=device,
        dtype=value_weight.dtype,
        non_blocking=False,
    )
    runtime_probabilities = probabilities.to(
        device=device,
        dtype=value_weight.dtype,
        non_blocking=False,
    )
    batch, keys = int(inputs.shape[0]), int(inputs.shape[1])
    queries = int(runtime_probabilities.shape[2])
    with torch.inference_mode():
        value = torch.nn.functional.linear(inputs, value_weight).view(
            batch,
            keys,
            key_value_heads,
            head_dim,
        )
        repeated_value = value.transpose(1, 2).repeat_interleave(
            query_heads // key_value_heads,
            dim=1,
        )
        context = torch.matmul(runtime_probabilities, repeated_value)
        concatenated = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(batch, queries, query_heads * head_dim)
        )
        output = torch.nn.functional.linear(concatenated, output_weight)
    if not bool(torch.isfinite(output).all().item()):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Fixed child V/O target produced a non-finite endpoint."
        )
    return output.detach().to(device="cpu").contiguous().clone()


def _build_target_lattice_evidence(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    bridge: StrictTokenizerBridgeArtifact,
    alignments: Sequence[Any],
    child_bank: Any,
    target_model: Any,
    layer_index: int,
    lineage: Lfm2SemanticCompilerLayerLineage,
    selection: Lfm2QdeimHeadSelection,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float,
) -> tuple[Lfm2SemanticAttentionTransportEvidence, ...]:
    """Build selected-head target-lattice evidence from one cache session."""

    torch = _torch()
    if (
        selection.layer_index != layer_index
        or lineage.layer_index != layer_index
        or selection.source_query_head_count
        != sum(len(group) for group in lineage.query_groups)
        or selection.source_key_value_head_count
        != sum(len(group) for group in lineage.key_value_groups)
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Q-DEIM selection, source lineage, and target layer diverged."
        )
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    attention = (
        None
        if layer_index < 0 or layer_index >= len(layers)
        else getattr(layers[layer_index], "self_attn", None)
    )
    query_weight = getattr(getattr(attention, "q_proj", None), "weight", None)
    key_weight = getattr(getattr(attention, "k_proj", None), "weight", None)
    head_dim = int(getattr(attention, "head_dim", 0))
    if (
        not isinstance(query_weight, torch.nn.Parameter)
        or not isinstance(key_weight, torch.nn.Parameter)
        or head_dim <= 0
        or int(query_weight.shape[0]) % head_dim
        or int(key_weight.shape[0]) % head_dim
        or selection.target_query_head_count
        != int(query_weight.shape[0]) // head_dim
        or selection.target_key_value_head_count
        != int(key_weight.shape[0]) // head_dim
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Q-DEIM selection does not match the live target GQA topology."
        )
    reference_by_index = {
        int(reference.index): reference for reference in session.batch_references
    }
    local_query_groups = tuple(
        (index,) for index in range(selection.target_query_head_count)
    )
    local_query_weights = tuple((1.0,) for _ in local_query_groups)
    selected_key_value_groups = tuple(
        (index,) for index in selection.selected_source_key_value_groups
    )
    selected_key_value_weights = tuple((1.0,) for _ in selected_key_value_groups)
    result = []
    for alignment in sorted(
        alignments,
        key=lambda value: value.source_batch_index,
    ):
        batch_index = alignment.source_batch_index
        reference = reference_by_index.get(batch_index)
        if reference is None:
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Semantic alignment has no authenticated cache batch."
            )
        phases = alignment.phase_order
        payload = session.load_selected_tensors(
            batch_index,
            _attention_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=phases,
            ),
        )
        try:
            tensors = payload.tensors
            prompt = alignment.phases[0].donor_full_prefix_tokens
            capability = inspect_lfm2_semantic_attention_cache_capability(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            capability.require_exact_history()
            history = assemble_lfm2_explicit_pre_rope_history(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            parent_query = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.q"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            parent_key = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.k"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            replay = child_bank.replay_by_batch[batch_index]
            pre_residual = child_bank.pre_residual_by_batch[batch_index]
            for phase_index, phase_alignment in enumerate(alignment.phases):
                donor_tokens = phase_alignment.donor_full_prefix_tokens
                target_tokens = phase_alignment.target_full_prefix_tokens
                phase_order = phases[: phase_index + 1]
                phase_capability = Lfm2SemanticAttentionCacheCapability(
                    layer_index=layer_index,
                    phase_order=phase_order,
                    donor_prompt_token_count=prompt,
                    selected_route="explicit_pre_rope",
                    prefill_captured_token_count=prompt,
                    assembled_current_row_count=donor_tokens,
                    missing_enhanced_tensor_names=(),
                    missing_legacy_tensor_names=(),
                    missing_donor_prefix_positions=(),
                    blocking_reasons=(),
                    exact_history_supported=True,
                )
                phase_history = SemanticAttentionPreRopeHistory(
                    query_normalized_pre_rope=(
                        history.query_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    key_normalized_pre_rope=(
                        history.key_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    value=history.value[:, :, :donor_tokens],
                    source_capability_sha256=phase_capability.capability_sha256,
                    source_tensor_names=tuple(
                        name
                        for name in history.source_tensor_names
                        if any(name.startswith(item + ".") for item in phase_order)
                    ),
                )
                full_transport = transport_lfm2_phase_probability_preserving_attention(
                    phase_alignment,
                    parent_query_post_rope=parent_query[:, :, :donor_tokens],
                    parent_key_post_rope=parent_key[:, :, :donor_tokens],
                    parent_value=history.value[:, :, :donor_tokens],
                    target_rope_inv_freq=target_rope_inv_freq,
                    target_rope_attention_scaling=(
                        target_rope_attention_scaling
                    ),
                    query_mode="semantic_endpoints",
                )
                transport = select_semantic_attention_expansion_query_heads(
                    full_transport,
                    selection.selected_source_query_heads,
                )
                query_positions = torch.tensor(
                    transport.query_target_positions,
                    dtype=torch.long,
                )
                child_key_value = replay.operator_input[:, :target_tokens]
                child_query = child_key_value.index_select(1, query_positions)
                actual_pre = pre_residual[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                fixed_output = _fixed_value_output_endpoint(
                    target_model=target_model,
                    layer_index=layer_index,
                    target_key_value_inputs=child_key_value,
                    probabilities=transport.probabilities,
                )
                parent_post_target = actual_pre.to(torch.float32) + fixed_output.to(
                    torch.float32
                )
                weights = torch.full(
                    (1, len(transport.query_target_positions)),
                    1.0 / math.sqrt(len(alignment.phases)),
                    dtype=torch.float64,
                )
                endpoint = Lfm2SemanticAttentionReplayEndpoint(
                    batch_index=batch_index,
                    layer_index=layer_index,
                    alignment_sha256=phase_alignment.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    captured_child_model_state_sha256=(
                        child_bank.captured_child_model_state_sha256
                    ),
                    target_query_inputs=child_query,
                    target_key_value_inputs=child_key_value,
                    actual_child_pre_attention_residual=actual_pre,
                    parent_post_attention_residual_target_space=(parent_post_target),
                    query_row_sqrt_weights=weights,
                )
                closure_lineage = canonical_sha256(
                    {
                        "schema": "neural-forge-lfm2-target-lattice-evidence-lineage-v1",
                        "base_compiler_lineage_sha256": (
                            lineage.compiler_lineage_sha256
                        ),
                        "qdeim_selection_sha256": selection.selection_sha256,
                        "cache_key": cache.cache_key,
                        "bridge_artifact_sha256": bridge.artifact_sha256,
                        "layer_index": layer_index,
                        "fixed_value_weight_sha256": _tensor_sha256(
                            target_model.model.layers[
                                layer_index
                            ].self_attn.v_proj.weight
                        ),
                        "fixed_output_weight_sha256": _tensor_sha256(
                            target_model.model.layers[
                                layer_index
                            ].self_attn.out_proj.weight
                        ),
                    }
                )
                result.append(
                    Lfm2SemanticAttentionTransportEvidence(
                        batch_index=batch_index,
                        layer_index=layer_index,
                        alignment_sha256=phase_alignment.alignment_sha256,
                        alignment=phase_alignment,
                        capability=phase_capability.to_evidence_dict(),
                        history=phase_history.to_evidence_dict(),
                        actual_child_replay=endpoint,
                        source_inventory=lineage.source_inventory,
                        query_groups=local_query_groups,
                        query_group_weights=local_query_weights,
                        key_value_groups=selected_key_value_groups,
                        key_value_group_weights=selected_key_value_weights,
                        compiler_lineage_sha256=closure_lineage,
                        transport=transport,
                    )
                )
        finally:
            session.release_selected_tensors(payload)
    return tuple(result)


def compile_install_lfm2_saved_body_target_lattice_qk_closure(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    qdeim_head_selections: Mapping[int, Lfm2QdeimHeadSelection],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    authenticated_cache_session: Any | None = None,
) -> Mapping[str, Any]:
    """Install one causal-order, whole-body Q/K-only closure transaction."""

    from contextlib import nullcontext

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    if not isinstance(compiler_lineage, Mapping) or not isinstance(
        qdeim_head_selections,
        Mapping,
    ):
        raise TypeError("compiler_lineage and qdeim_head_selections must be mappings.")
    kinds = _validated_operator_topology(cache, target_model, compiler_lineage)
    attention_layers = tuple(
        index for index, kind in enumerate(kinds) if kind == "full_attention"
    )
    if set(qdeim_head_selections) != set(attention_layers) or any(
        not isinstance(qdeim_head_selections[index], Lfm2QdeimHeadSelection)
        or qdeim_head_selections[index].layer_index != index
        for index in attention_layers
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Q-DEIM head selections do not cover the exact attention topology."
        )
    torch = _torch()
    parameter_dtypes = {parameter.dtype for parameter in target_model.parameters()}
    if len(parameter_dtypes) != 1 or not parameter_dtypes.issubset(
        {torch.bfloat16, torch.float32}
    ):
        raise Lfm2SavedBodyTargetLatticeQkError(
            "Saved child Parameters must share one BF16 or FP32 checkpoint dtype."
        )
    parameter_dtype = next(iter(parameter_dtypes))
    parameter_dtype_name = str(parameter_dtype).removeprefix("torch.")
    parameter_count = _parameter_count(target_model)
    entry = _WholeModelSnapshot.capture(target_model)
    entry_hashes = _state_tensor_hashes(target_model)
    entry_identity = fingerprint_lfm2_parent(target_model).identity_sha256
    target_rope_inv_freq, target_rope_attention_scaling = _target_rope(target_model)
    cache_context = (
        cache.selective_authenticated_session()
        if authenticated_cache_session is None
        else nullcontext(authenticated_cache_session)
    )
    try:
        with cache_context as session:
            alignments, tokenizer_session_receipt = _semantic_alignments_from_session(
                cache=cache,
                session=session,
                bridge=bridge,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
            )
            refinement = dict(
                audit_lfm2_semantic_span_cpcf_refinement(
                    alignments,
                    expected_batch_count=cache.batch_count,
                )
            )
            if not refinement.get(
                "probability_preserving_attention_transport_go",
                False,
            ):
                raise Lfm2SavedBodyTargetLatticeQkError(
                    "Semantic target lattice is not exact: "
                    + ", ".join(refinement.get("blocking_reasons", ()))
                )
            residual_bank = _initial_residual_bank(target_model, alignments)
            layer_receipts = []
            attention_installations = []
            previous_state = entry_identity
            replay_count = 0
            for layer_index, kind in enumerate(kinds):
                before = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                replay_count += 1
                if before.captured_child_model_state_sha256 != previous_state:
                    raise Lfm2SavedBodyTargetLatticeQkError(
                        f"Layer {layer_index} replay is not causally fresh."
                    )
                if kind == "full_attention":
                    evidence = _build_target_lattice_evidence(
                        cache=cache,
                        session=session,
                        bridge=bridge,
                        alignments=alignments,
                        child_bank=before,
                        target_model=target_model,
                        layer_index=layer_index,
                        lineage=compiler_lineage[layer_index],
                        selection=qdeim_head_selections[layer_index],
                        target_rope_inv_freq=target_rope_inv_freq,
                        target_rope_attention_scaling=(
                            target_rope_attention_scaling
                        ),
                    )
                    installation = dict(
                        compile_install_lfm2_semantic_target_lattice_qk_only(
                            target_model=target_model,
                            layer_index=layer_index,
                            evidence=evidence,
                            cache_key=cache.cache_key,
                            bridge_artifact_sha256=bridge.artifact_sha256,
                        )
                    )
                    prefix = f"model.layers.{layer_index}.self_attn."
                    if (
                        installation.get("schema")
                        != LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA
                        or installation.get("model_state_before_sha256")
                        != previous_state
                        or installation.get("installed") is not True
                        or not installation.get("changed_state_keys")
                        or any(
                            name
                            not in {
                                prefix + "q_proj.weight",
                                prefix + "k_proj.weight",
                            }
                            for name in installation["changed_state_keys"]
                        )
                        or installation.get("actual_changed_state_keys_are_qk_only")
                        is not True
                        or installation.get("parameter_dtype")
                        != parameter_dtype_name
                        or installation.get("checkpoint_dtype_preserved") is not True
                        or installation.get("numeric_metric_used_for_publication_or_rollback")
                        is not False
                    ):
                        raise Lfm2SavedBodyTargetLatticeQkError(
                            f"Layer {layer_index} did not install one Q/K-only closure."
                        )
                    previous_state = installation["model_state_after_sha256"]
                    after = _replay_current_layer(
                        target_model,
                        alignments=alignments,
                        layer_index=layer_index,
                        pre_residual_by_batch=residual_bank,
                    )
                    replay_count += 1
                    if after.captured_child_model_state_sha256 != previous_state:
                        raise Lfm2SavedBodyTargetLatticeQkError(
                            f"Layer {layer_index} treated replay is stale."
                        )
                    propagation = after
                    attention_installations.append(installation)
                    evidence_sha256 = [item.evidence_sha256 for item in evidence]
                else:
                    propagation = before
                    evidence_sha256 = []
                residual_bank = _next_residual_bank(propagation)
                layer_body = {
                    "schema": "neural-forge-lfm2-saved-body-qk-layer-step-v1",
                    "layer_index": layer_index,
                    "operator_kind": kind,
                    "entry_replay_bank_sha256": before.bank_sha256,
                    "entry_model_state_sha256": (
                        before.captured_child_model_state_sha256
                    ),
                    "qk_installation_sha256": (
                        None
                        if kind != "full_attention"
                        else attention_installations[-1]["result_sha256"]
                    ),
                    "transport_evidence_sha256": evidence_sha256,
                    "propagation_replay_bank_sha256": propagation.bank_sha256,
                    "propagation_model_state_sha256": (
                        propagation.captured_child_model_state_sha256
                    ),
                    "qk_modified": kind == "full_attention",
                    "non_qk_tensor_modified": False,
                }
                layer_receipts.append(
                    {**layer_body, "result_sha256": canonical_sha256(layer_body)}
                )
        after_hashes = _state_tensor_hashes(target_model)
        changed = _changed_state_keys(entry_hashes, after_hashes)
        allowed = tuple(
            sorted(
                f"model.layers.{layer_index}.self_attn.{role}.weight"
                for layer_index in attention_layers
                for role in ("q_proj", "k_proj")
            )
        )
        final_identity = fingerprint_lfm2_parent(target_model).identity_sha256
        receipt_changed = tuple(
            sorted(
                name
                for receipt in attention_installations
                for name in receipt["changed_state_keys"]
            )
        )
        if (
            not changed
            or tuple(changed) != receipt_changed
            or not set(changed).issubset(allowed)
            or previous_state != final_identity
            or _parameter_count(target_model) != parameter_count
            or any(
                parameter.dtype != parameter_dtype
                for parameter in target_model.parameters()
            )
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Whole-body target-lattice transaction changed state outside Q/K."
            )
        body = {
            "schema": LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
            "status": "installed-causal-order-target-lattice-qk-closure",
            "cache_key": cache.cache_key,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "source_parent_identity_sha256": cache.parent_fingerprint,
            "model_state_before_sha256": entry_identity,
            "model_state_after_sha256": final_identity,
            "parameter_dtype": parameter_dtype_name,
            "checkpoint_dtype_preserved": all(
                parameter.dtype == parameter_dtype
                for parameter in target_model.parameters()
            ),
            "changed_state_keys": list(changed),
            "allowed_qk_state_keys": list(allowed),
            "actual_changed_state_keys_are_qk_only": True,
            "attention_layer_indices": list(attention_layers),
            "attention_layer_count": len(attention_layers),
            "attention_install_count": len(attention_installations),
            "short_conv_layer_indices": [
                index for index, kind in enumerate(kinds) if kind == "short_conv"
            ],
            "layer_steps": layer_receipts,
            "qdeim_head_selections": {
                str(index): qdeim_head_selections[index].as_dict()
                for index in attention_layers
            },
            "attention_installations": attention_installations,
            "tokenizer_session_sha256": _digest(
                tokenizer_session_receipt.get("receipt_sha256"),
                "tokenizer session SHA-256",
            ),
            "semantic_refinement_sha256": _digest(
                refinement.get("report_sha256"),
                "semantic refinement SHA-256",
            ),
            "child_macrocell_replay_count": replay_count,
            "causal_layer_order_proven": True,
            "later_layers_replayed_from_treated_prefix": True,
            "fixed_child_value_output_used_as_endpoint_teacher": True,
            "parent_probability_routes_transported_to_target_token_lattice": True,
            "qdeim_selected_source_heads_preserved_without_averaging": True,
            "all_non_qk_tensors_byte_identical": True,
            "parameter_element_count_before": parameter_count,
            "parameter_element_count_after": parameter_count,
            "parameter_count_changed": False,
            "numeric_metrics_are_navigation_evidence_only": True,
            "numeric_metrics_have_publication_or_rollback_authority": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
            "operation_audit": dict(_ZERO_FORBIDDEN_OPERATIONS),
        }
        return MappingProxyType(
            {**body, "result_sha256": canonical_sha256(body)}
        )
    except BaseException as exc:
        entry.restore(target_model)
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != entry_identity
            or _state_tensor_hashes(target_model) != entry_hashes
            or _parameter_count(target_model) != parameter_count
        ):
            raise Lfm2SavedBodyTargetLatticeQkError(
                "Target-lattice Q/K closure failed and whole-model rollback was not exact."
            ) from exc
        raise


__all__ = [
    "LFM2_QDEIM_HEAD_SELECTION_SCHEMA",
    "LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA",
    "Lfm2QdeimHeadSelection",
    "Lfm2SavedBodyTargetLatticeQkError",
    "authenticate_lfm2_saved_attention_lineage",
    "compile_install_lfm2_saved_body_target_lattice_qk_closure",
    "extract_lfm2_qdeim_head_selections",
]
