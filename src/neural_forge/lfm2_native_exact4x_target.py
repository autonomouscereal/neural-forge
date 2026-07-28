"""Authenticated native LFM2 target class for the semantic exact-4x lane.

The legacy standalone proof owns one immutable H776/I700 target.  The dynamic
facade, however, may select another dense native topology under the same
quarter-size ceiling.  This module defines the deliberately narrow class that
the existing semantic kernel can consume: the parent-native 14-layer operator
schedule, a 4096-row bridged and untied token interface, 64-wide full-coverage
GQA groups, and an authenticated whole-graph component-allocation receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
)
from .dynamic_parameter_frontier import DenseComponentAllocationReceipt
from .dynamic_target_synthesis import DynamicTargetSynthesisResult


SCHEMA = "neural-forge-lfm2-native-exact4x-target-v1"
LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID = (
    "lfm2.native.same-depth-4096-untied-gqa64.at-or-below-quarter.v1"
)
LFM2_NATIVE_EXACT4X_LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)
LFM2_NATIVE_EXACT4X_VOCABULARY_SIZE = 4_096
LFM2_NATIVE_EXACT4X_HEAD_DIM = 64
LFM2_NATIVE_EXACT4X_FIR_LENGTH = 3


class Lfm2NativeExact4xTargetError(ValueError):
    """A synthesized target is outside the registered semantic kernel class."""


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2NativeExact4xTargetError(f"{name} must be a positive integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2NativeExact4xTargetError(f"{name} must be a lowercase SHA-256.")
    return value


def _model_topology(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise Lfm2NativeExact4xTargetError(
            "Native exact-4x target has no materialized LFM2 configuration."
        )
    try:
        return {
            "vocab_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "tie_word_embeddings": bool(config.tie_word_embeddings),
            "layer_types": tuple(str(value) for value in config.layer_types),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise Lfm2NativeExact4xTargetError(
            "Native exact-4x target configuration is incomplete."
        ) from exc


@dataclass(frozen=True, slots=True)
class Lfm2NativeExact4xTargetContract:
    """Sealed geometry/count binding for one facade-selected complete child."""

    target_class_id: str
    parent_unique_parameters: int
    quarter_parameter_ceiling: int
    target_unique_parameters: int
    topology: tuple[tuple[str, Any], ...]
    target_topology_fingerprint: str
    target_graph_fingerprint: str
    target_request_fingerprint: str
    source_graph_fingerprint: str
    component_allocation_receipt_sha256: str
    component_allocation_topology_sha256: str
    tokenizer_bridge_artifact_sha256: str

    @property
    def topology_dict(self) -> dict[str, Any]:
        result = dict(self.topology)
        result["layer_types"] = list(result["layer_types"])
        return result

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "kind": "authenticated_native_exact4x_target_contract",
            "target_class_id": self.target_class_id,
            "parent_unique_parameters": self.parent_unique_parameters,
            "quarter_parameter_ceiling": self.quarter_parameter_ceiling,
            "target_unique_parameters": self.target_unique_parameters,
            "realized_parameter_compaction_ratio": (
                self.parent_unique_parameters / self.target_unique_parameters
            ),
            "target_topology": self.topology_dict,
            "target_topology_fingerprint": self.target_topology_fingerprint,
            "target_graph_fingerprint": self.target_graph_fingerprint,
            "target_request_fingerprint": self.target_request_fingerprint,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "component_allocation_receipt_sha256": (
                self.component_allocation_receipt_sha256
            ),
            "component_allocation_topology_sha256": (
                self.component_allocation_topology_sha256
            ),
            "tokenizer_bridge_artifact_sha256": (
                self.tokenizer_bridge_artifact_sha256
            ),
            "same_depth_parent_native_schedule": True,
            "token_interface_mode": "untied",
            "full_coverage_gqa": True,
            "at_or_below_quarter_parameter_ceiling": True,
            "dense_parameters": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "parent_sidecar_used": False,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def derive_lfm2_native_exact4x_target_contract(
    target: DynamicTargetSynthesisResult,
    *,
    expected_parent_unique_parameters: int,
    expected_bridge_artifact_sha256: str,
) -> Lfm2NativeExact4xTargetContract:
    """Bind an already compiler-authenticated target to the supported class.

    Callers must first run ``verify_compiler_authenticated_target`` and
    ``validate_lfm2_dense_budgeted_child``.  This routine adds semantic-kernel
    class checks and independently binds the typed allocation receipt, model
    configuration, target graph/counts, and source/bridge identities.
    """

    if not isinstance(target, DynamicTargetSynthesisResult):
        raise Lfm2NativeExact4xTargetError(
            "Native exact-4x binding requires DynamicTargetSynthesisResult."
        )
    selected = target.adapter_payload
    if not isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
        raise Lfm2NativeExact4xTargetError(
            "Native exact-4x binding requires a dense LFM2 selected target."
        )
    parent_count = _positive(
        expected_parent_unique_parameters,
        "expected_parent_unique_parameters",
    )
    bridge_sha256 = _digest(
        expected_bridge_artifact_sha256,
        "expected_bridge_artifact_sha256",
    )
    receipt = getattr(target.request, "component_allocation_receipt", None)
    if not isinstance(receipt, DenseComponentAllocationReceipt):
        raise Lfm2NativeExact4xTargetError(
            "Native exact-4x target lacks its typed component-allocation receipt."
        )
    request_fingerprint = _digest(
        getattr(target.request, "fingerprint", None),
        "target request fingerprint",
    )
    topology_fingerprint = _digest(
        target.topology_fingerprint,
        "target topology fingerprint",
    )
    graph_fingerprint = _digest(
        target.graph.fingerprint,
        "target graph fingerprint",
    )
    source_fingerprint = _digest(
        selected.source_graph_fingerprint,
        "source graph fingerprint",
    )
    target_count = _positive(
        selected.exact_target_unique_parameters,
        "target_unique_parameters",
    )
    quarter_ceiling = parent_count // 4
    model_topology = _model_topology(target.model)
    selected_topology = {
        "vocab_size": selected.vocabulary_size,
        "hidden_size": selected.hidden_size,
        "intermediate_size": selected.intermediate_size,
        "num_hidden_layers": selected.target_depth,
        "num_attention_heads": selected.query_heads,
        "num_key_value_heads": selected.key_value_heads,
        "head_dim": selected.head_dim,
        "tie_word_embeddings": selected.tie_word_embeddings,
        "layer_types": selected.layer_types,
    }
    parent_graphs = dict(target.parent_graph_fingerprints)
    full_coverage_gqa = (
        selected.query_heads >= selected.key_value_heads
        and selected.query_heads % selected.key_value_heads == 0
        and selected.query_heads * selected.head_dim <= selected.hidden_size
        and selected.key_value_heads * selected.head_dim <= selected.hidden_size
    )
    counts = (
        getattr(target.request, "exact_unique_parameters", None),
        getattr(target.resource_profile, "unique_parameters", None),
        getattr(target.graph, "unique_parameter_count", None),
        receipt.target_unique_parameters,
    )
    if (
        target.adapter_id != Lfm2TargetSynthesisAdapter.adapter_id
        or target.adapter_version != Lfm2TargetSynthesisAdapter.version
        or target_count > quarter_ceiling
        or any(value != target_count for value in counts)
        or receipt.source_unique_parameters != parent_count
        or receipt.source_graph_fingerprint != source_fingerprint
        or receipt.fingerprint != selected.component_allocation_receipt_sha256
        or receipt.topology_configuration_sha256
        != selected.component_allocation_topology_sha256
        or receipt.target_logical_parameters
        != selected.exact_target_logical_parameters
        or selected_topology != model_topology
        or selected.layer_types != LFM2_NATIVE_EXACT4X_LAYER_TYPES
        or selected.ordered_source_spans
        != tuple((index,) for index in range(len(LFM2_NATIVE_EXACT4X_LAYER_TYPES)))
        or selected.vocabulary_size != LFM2_NATIVE_EXACT4X_VOCABULARY_SIZE
        or selected.head_dim != LFM2_NATIVE_EXACT4X_HEAD_DIM
        or selected.fir_length != LFM2_NATIVE_EXACT4X_FIR_LENGTH
        or selected.tie_word_embeddings
        or selected.target_token_interface_mode != "untied"
        or not selected.tokenizer_bridge_required
        or selected.tokenizer_bridge_artifact_sha256 != bridge_sha256
        or parent_graphs.get(selected.source_parent) != source_fingerprint
        or getattr(target.resource_profile, "family", None) != "lfm2"
        or tuple(getattr(target.resource_profile, "modalities", ())) != ("text",)
        or getattr(target.resource_profile, "layer_count", None)
        != len(LFM2_NATIVE_EXACT4X_LAYER_TYPES)
        or not full_coverage_gqa
    ):
        raise Lfm2NativeExact4xTargetError(
            "Synthesized target is outside the registered native exact-4x "
            "semantic-kernel class or differs from its authenticated receipt."
        )
    topology = tuple(
        (key, tuple(value) if key == "layer_types" else value)
        for key, value in selected_topology.items()
    )
    return Lfm2NativeExact4xTargetContract(
        target_class_id=LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID,
        parent_unique_parameters=parent_count,
        quarter_parameter_ceiling=quarter_ceiling,
        target_unique_parameters=target_count,
        topology=topology,
        target_topology_fingerprint=topology_fingerprint,
        target_graph_fingerprint=graph_fingerprint,
        target_request_fingerprint=request_fingerprint,
        source_graph_fingerprint=source_fingerprint,
        component_allocation_receipt_sha256=receipt.fingerprint,
        component_allocation_topology_sha256=(
            receipt.topology_configuration_sha256
        ),
        tokenizer_bridge_artifact_sha256=bridge_sha256,
    )


__all__ = [
    "LFM2_NATIVE_EXACT4X_HEAD_DIM",
    "LFM2_NATIVE_EXACT4X_LAYER_TYPES",
    "LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID",
    "LFM2_NATIVE_EXACT4X_VOCABULARY_SIZE",
    "Lfm2NativeExact4xTargetContract",
    "Lfm2NativeExact4xTargetError",
    "derive_lfm2_native_exact4x_target_contract",
]
