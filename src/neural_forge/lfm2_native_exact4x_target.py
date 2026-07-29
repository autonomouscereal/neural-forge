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
from .dynamic_parameter_frontier import (
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_KIND,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    DenseComponentAllocationReceipt,
)
from .dynamic_target_synthesis import DynamicTargetSynthesisResult
from .hf_attention_fold import (
    AttentionFoldError,
    validate_gqa_attention_fold_geometry,
)
from .hf_lfm2_dynamic_blank_materializer import (
    LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA,
    Lfm2DynamicBlankMaterializationError,
    extract_lfm2_dynamic_physical_layout_from_config,
    materialize_lfm2_dynamic_blank_child,
)
from .hf_lfm2_dynamic_physical_layout import (
    LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA,
    Lfm2DynamicPhysicalLayoutError,
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    lfm2_expected_heterogeneous_physical_state_shapes,
    restore_lfm2_heterogeneous_physical_layout,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    Lfm2HeterogeneousLayoutSelectionError,
    Lfm2HeterogeneousLayoutSelectionReceipt,
)


SCHEMA = "neural-forge-lfm2-native-exact4x-target-v1"
HETEROGENEOUS_SCHEMA = "neural-forge-lfm2-native-heterogeneous-exact4x-target-v2"
LFM2_IDENTITY_TOKENIZER_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-identity-tokenizer-evidence-v1"
)
LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID = (
    "lfm2.native.same-depth-4096-untied-gqa64.at-or-below-quarter.v1"
)
LFM2_HETEROGENEOUS_NATIVE_EXACT4X_TARGET_CLASS_ID = (
    "lfm2.native.same-depth-untied-heterogeneous-dense-fp32.at-or-below-quarter.v2"
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
LFM2_NATIVE_EXACT4X_SOURCE_QUERY_HEADS = 16
LFM2_NATIVE_EXACT4X_SOURCE_KEY_VALUE_HEADS = 8
LFM2_NATIVE_EXACT4X_SOURCE_HEAD_DIM = 64
_LFM2_DYNAMIC_RUNTIME_SHA_FIELD = "_neural_forge_dynamic_blank_runtime_sha256"


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


def lfm2_identity_tokenizer_evidence_sha256(
    *,
    tokenizer_semantics_sha256: str,
    vocabulary_size: int,
) -> str:
    """Seal an explicitly identical parent/child token-ID interpretation.

    Equal vocabulary cardinality is insufficient: callers must authenticate
    the same tokenizer semantics on both sides.  This deterministic seal is
    the zero-parameter identity transport evidence stored in the physical
    descriptor and its allocation selection.
    """

    semantics_sha256 = _digest(
        tokenizer_semantics_sha256,
        "tokenizer_semantics_sha256",
    )
    vocab_size = _positive(vocabulary_size, "vocabulary_size")
    return canonical_sha256(
        {
            "schema": LFM2_IDENTITY_TOKENIZER_EVIDENCE_SCHEMA,
            "kind": "authenticated_identity_tokenizer_transport",
            "parent_tokenizer_semantics_sha256": semantics_sha256,
            "child_tokenizer_semantics_sha256": semantics_sha256,
            "vocabulary_size": vocab_size,
            "token_id_mapping": "identity",
            "token_row_mapping": "identity",
            "bridge_artifact_required": False,
            "runtime_parameter_elements": 0,
        }
    )


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


def _heterogeneous_scaffold_topology(
    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor,
) -> dict[str, Any]:
    """Describe only the native-construction scaffold around a ragged graph."""

    first_attention = next(
        (layer for layer in descriptor.layers if layer.layer_type == "full_attention"),
        None,
    )
    if first_attention is None:
        query_heads, key_value_heads, head_dim = 1, 1, 2
    else:
        assert first_attention.query_heads is not None
        assert first_attention.key_value_heads is not None
        assert first_attention.head_dim is not None
        query_heads = first_attention.query_heads
        key_value_heads = first_attention.key_value_heads
        head_dim = first_attention.head_dim
    return {
        "vocab_size": descriptor.vocab_size,
        "hidden_size": descriptor.hidden_size,
        "intermediate_size": descriptor.layers[0].mlp_intermediate_size,
        "num_hidden_layers": descriptor.layer_count,
        "num_attention_heads": query_heads,
        "num_key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "tie_word_embeddings": False,
        "layer_types": tuple(layer.layer_type for layer in descriptor.layers),
    }


def _restore_heterogeneous_target_artifacts(
    target: DynamicTargetSynthesisResult,
    selected: Lfm2DenseBudgetedSelectedTarget,
) -> (
    tuple[
        Lfm2HeterogeneousPhysicalLayoutDescriptor,
        Lfm2HeterogeneousLayoutSelectionReceipt,
    ]
    | None
):
    """Recover the canonical descriptor/selection pair from a selected target."""

    descriptor_value = getattr(
        selected,
        "heterogeneous_layout_descriptor",
        None,
    )
    selection_value = getattr(
        selected,
        "heterogeneous_layout_selection_receipt",
        None,
    )
    configuration = dict(selected.configuration)
    marker_keys = {
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    }
    present = marker_keys.intersection(configuration)
    if descriptor_value is None and selection_value is None and not present:
        return None
    if descriptor_value is None or selection_value is None or present != marker_keys:
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target descriptor/selection markers are incomplete."
        )
    descriptor_json = configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY]
    descriptor_sha256 = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY
    ]
    selection_json = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY
    ]
    selection_sha256 = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY
    ]
    if not all(
        isinstance(value, str)
        for value in (
            descriptor_json,
            descriptor_sha256,
            selection_json,
            selection_sha256,
        )
    ):
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target markers must be canonical ASCII JSON "
            "and SHA-256 strings."
        )
    try:
        descriptor = restore_lfm2_heterogeneous_physical_layout(
            descriptor_json,
            expected_descriptor_sha256=_digest(
                descriptor_sha256,
                "heterogeneous descriptor SHA-256",
            ),
        )
        supplied_descriptor = restore_lfm2_heterogeneous_physical_layout(
            descriptor_value,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )
        selection = Lfm2HeterogeneousLayoutSelectionReceipt.restore(selection_json)
        supplied_selection = Lfm2HeterogeneousLayoutSelectionReceipt.restore(
            selection_value
        )
    except (
        Lfm2DynamicPhysicalLayoutError,
        Lfm2HeterogeneousLayoutSelectionError,
    ) as exc:
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target artifacts failed canonical authentication."
        ) from exc
    if (
        supplied_descriptor != descriptor
        or supplied_selection != selection
        or selection.receipt_sha256
        != _digest(selection_sha256, "heterogeneous selection receipt SHA-256")
        or selection.descriptor_sha256 != descriptor.descriptor_sha256
    ):
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target artifacts differ from their selected "
            "configuration seals."
        )
    request_configuration = dict(getattr(target.request, "configuration", ()))
    if any(request_configuration.get(key) != configuration[key] for key in marker_keys):
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target request differs from its selected "
            "descriptor/selection artifacts."
        )
    try:
        configured_descriptor = extract_lfm2_dynamic_physical_layout_from_config(
            target.model.config
        )
    except (AttributeError, Lfm2DynamicPhysicalLayoutError) as exc:
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target model config does not authenticate "
            "its physical layout."
        ) from exc
    if (
        type(configured_descriptor) is not Lfm2HeterogeneousPhysicalLayoutDescriptor
        or configured_descriptor != descriptor
    ):
        raise Lfm2NativeExact4xTargetError(
            "Heterogeneous native target model config is bound to another "
            "physical layout."
        )
    return descriptor, selection


def _heterogeneous_graph_is_exact(
    target: DynamicTargetSynthesisResult,
    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor,
) -> bool:
    """Recompute and bind the graph's physical-materialization proof.

    The runtime marker precondition is intentional: this verifier may audit an
    already materialized child, but it must never turn a scaffold into one as a
    side effect of contract derivation.
    """

    try:
        metadata = dict(target.graph.metadata)
        metadata_audit_sha256 = _digest(
            metadata.get("physical_materialization_audit_sha256"),
            "physical materialization audit SHA-256",
        )
        metadata_materialization_sha256 = _digest(
            metadata.get("blank_materialization_receipt_sha256"),
            "blank materialization receipt SHA-256",
        )
        expected_state_key_count = len(
            lfm2_expected_heterogeneous_physical_state_shapes(descriptor)
        )
        model = target.model
        if (
            getattr(model, _LFM2_DYNAMIC_RUNTIME_SHA_FIELD, None)
            != descriptor.descriptor_sha256
        ):
            return False
        materialization = materialize_lfm2_dynamic_blank_child(
            model,
            model.config,
            descriptor,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )
        audit = dict(materialization["physical_materialization_audit"])
        audit_sha256 = _digest(
            audit.get("result_sha256"),
            "recomputed physical materialization audit SHA-256",
        )
        materialization_sha256 = _digest(
            materialization.get("receipt_sha256"),
            "recomputed blank materialization receipt SHA-256",
        )
        audit_body = dict(audit)
        del audit_body["result_sha256"]
        materialization_body = dict(materialization)
        del materialization_body["receipt_sha256"]
        recomputed_audit_sha256 = canonical_sha256(audit_body)
        recomputed_materialization_sha256 = canonical_sha256(materialization_body)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        Lfm2DynamicBlankMaterializationError,
        Lfm2DynamicPhysicalLayoutError,
        Lfm2NativeExact4xTargetError,
    ):
        return False
    expected = {
        "physical_layout": "heterogeneous_dense_fp32",
        "physical_layout_descriptor_sha256": descriptor.descriptor_sha256,
        "physical_layout_evidence_binding_sha256": (descriptor.evidence_binding_sha256),
        "physical_parameter_state_key_count": expected_state_key_count,
        "runtime_parameter_alias_group_count": 0,
        "cross_layer_parameter_sharing": False,
        "runtime_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
        "parameters_quantized": 0,
        "sparse_parameter_storage": False,
        "pruning_masks_present": False,
    }
    expected_audit = {
        "schema": LFM2_DYNAMIC_PHYSICAL_AUDIT_SCHEMA,
        "status": "dense-fp32-physical-layout-authenticated",
        "descriptor_sha256": descriptor.descriptor_sha256,
        "parameter_state_key_count": expected_state_key_count,
        "all_registered_modules_eval": True,
        "gradient_checkpointing_enabled": False,
        "all_parameters_frozen": True,
        "unique_parameter_object_count": expected_state_key_count,
        "unique_parameter_storage_count": expected_state_key_count,
        "parameter_elements": descriptor.accounting.unique_parameter_elements,
        "parameter_bytes_fp32": descriptor.accounting.parameter_bytes_fp32,
        "runtime_parameter_alias_group_count": 0,
        "cross_layer_parameter_sharing": False,
        "parameters_quantized": 0,
        "sparse_parameter_storage": False,
        "pruning_masks_present": False,
        "runtime_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
    }
    expected_materialization = {
        "schema": LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA,
        "status": "authenticated-dense-fp32-blank-child-materialization",
        "descriptor_sha256": descriptor.descriptor_sha256,
        "parameter_state_key_count": expected_state_key_count,
        "parameter_elements": descriptor.accounting.unique_parameter_elements,
    }
    return (
        metadata_audit_sha256 == audit_sha256
        and metadata_materialization_sha256 == materialization_sha256
        and audit_sha256 == recomputed_audit_sha256
        and materialization_sha256 == recomputed_materialization_sha256
        and all(metadata.get(key) == value for key, value in expected.items())
        and all(audit.get(key) == value for key, value in expected_audit.items())
        and all(
            materialization.get(key) == value
            for key, value in expected_materialization.items()
        )
        and getattr(target.graph, "logical_parameter_count", None)
        == descriptor.accounting.logical_parameter_elements
        and getattr(target.graph, "unique_parameter_count", None)
        == descriptor.accounting.unique_parameter_elements
        and getattr(target.graph, "unique_parameter_bytes", None)
        == descriptor.accounting.parameter_bytes_fp32
    )


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
    tokenizer_bridge_artifact_sha256: str | None
    source_query_heads: int
    source_key_value_heads: int
    source_head_dim: int

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
            "tokenizer_bridge_artifact_sha256": (self.tokenizer_bridge_artifact_sha256),
            "source_attention_geometry": {
                "query_heads": self.source_query_heads,
                "key_value_heads": self.source_key_value_heads,
                "head_dim": self.source_head_dim,
            },
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


@dataclass(frozen=True, slots=True)
class Lfm2HeterogeneousNativeExact4xTargetContract(Lfm2NativeExact4xTargetContract):
    """Sealed per-layer physical target accepted by the heterogeneous lane."""

    physical_layout_descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor
    layout_selection_receipt: Lfm2HeterogeneousLayoutSelectionReceipt

    def __post_init__(self) -> None:
        descriptor = self.physical_layout_descriptor
        selection = self.layout_selection_receipt
        if (
            type(descriptor) is not Lfm2HeterogeneousPhysicalLayoutDescriptor
            or type(selection) is not Lfm2HeterogeneousLayoutSelectionReceipt
            or selection.descriptor_sha256 != descriptor.descriptor_sha256
            or selection.allocation_decision.allocation_decision_sha256
            != descriptor.allocation_evidence_sha256
            or selection.allocation_decision.source_graph_fingerprint
            != descriptor.source_topology_sha256
            or selection.allocation_decision.token_bridge_evidence_sha256
            != descriptor.token_bridge_evidence_sha256
            or selection.allocation_decision.exact_target_unique_parameters
            != descriptor.accounting.unique_parameter_elements
            or selection.allocation_decision.selected_layer_option_sha256
            != tuple(layer.selection_evidence_sha256 for layer in descriptor.layers)
        ):
            raise Lfm2NativeExact4xTargetError(
                "Heterogeneous native target contract artifacts are not one "
                "authenticated physical selection."
            )

    @property
    def topology_dict(self) -> dict[str, Any]:
        result = dict(self.topology)
        result["layer_types"] = list(result["layer_types"])
        descriptor = self.physical_layout_descriptor
        layer_rows: list[dict[str, Any]] = []
        for physical, accounting in zip(
            descriptor.layers,
            descriptor.accounting.layers,
            strict=True,
        ):
            row = physical.as_dict()
            if physical.layer_type == "conv":
                assert physical.short_conv_internal_width is not None
                row["short_conv_writer"] = {
                    "kind": "dense",
                    "shape": [
                        descriptor.hidden_size,
                        physical.short_conv_internal_width,
                    ],
                }
            else:
                assert physical.query_heads is not None
                assert physical.key_value_heads is not None
                assert physical.head_dim is not None
                row["query_projection_width"] = physical.query_heads * physical.head_dim
                row["key_value_projection_width"] = (
                    physical.key_value_heads * physical.head_dim
                )
            row["parameter_accounting"] = accounting.as_dict()
            layer_rows.append(row)
        result.update(
            {
                "physical_layout_kind": LFM2_HETEROGENEOUS_LAYOUT_KIND,
                "physical_layout_descriptor_sha256": (descriptor.descriptor_sha256),
                "conv_L_cache": descriptor.conv_kernel_size,
                "layers": layer_rows,
            }
        )
        return result

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = Lfm2NativeExact4xTargetContract.as_dict(
            self,
            include_fingerprint=False,
        )
        descriptor = self.physical_layout_descriptor
        selection = self.layout_selection_receipt
        value.update(
            {
                "schema": HETEROGENEOUS_SCHEMA,
                "kind": ("authenticated_heterogeneous_native_exact4x_target_contract"),
                "target_topology": self.topology_dict,
                "physical_layout_kind": LFM2_HETEROGENEOUS_LAYOUT_KIND,
                "physical_layout_descriptor": descriptor.as_dict(),
                "physical_layout_descriptor_sha256": (descriptor.descriptor_sha256),
                "physical_layout_evidence_binding_sha256": (
                    descriptor.evidence_binding_sha256
                ),
                "physical_layout_allocation_evidence_sha256": (
                    descriptor.allocation_evidence_sha256
                ),
                "physical_layout_selection_receipt": selection.as_dict(),
                "physical_layout_selection_receipt_sha256": (selection.receipt_sha256),
                "heterogeneous_per_layer_dimensions": True,
                "shared_intermediate_or_operator_widths_assumed": False,
                "short_conv_writer_mode": "independent_dense",
                "physical_storage_dtype": "float32",
                "runtime_parameter_alias_group_count": 0,
                "cross_layer_parameter_sharing": False,
                "runtime_bridge_parameter_elements": 0,
                "runtime_sidecar_parameter_elements": 0,
            }
        )
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _heterogeneous_token_transport_is_authenticated(
    selected: Lfm2DenseBudgetedSelectedTarget,
    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor,
    selection: Lfm2HeterogeneousLayoutSelectionReceipt,
    bridge_sha256: str | None,
) -> bool:
    """Accept a sealed bridge, or an explicitly sealed identity transport."""

    decision_evidence = selection.allocation_decision.token_bridge_evidence_sha256
    descriptor_evidence = descriptor.token_bridge_evidence_sha256
    try:
        parent_semantics = _digest(
            selected.parent_tokenizer_semantics_sha256,
            "parent tokenizer semantics SHA-256",
        )
        child_semantics = _digest(
            selected.child_tokenizer_semantics_sha256,
            "child tokenizer semantics SHA-256",
        )
        if bridge_sha256 is not None:
            _digest(
                selected.tokenizer_transport_profile_sha256,
                "tokenizer transport profile SHA-256",
            )
            return (
                parent_semantics != child_semantics
                and selected.tokenizer_bridge_required
                and selected.tokenizer_bridge_artifact_sha256 == bridge_sha256
                and descriptor_evidence == bridge_sha256
                and decision_evidence == bridge_sha256
            )
        identity_evidence = lfm2_identity_tokenizer_evidence_sha256(
            tokenizer_semantics_sha256=parent_semantics,
            vocabulary_size=descriptor.vocab_size,
        )
    except Lfm2NativeExact4xTargetError:
        return False
    return (
        parent_semantics == child_semantics
        and not selected.tokenizer_bridge_required
        and selected.tokenizer_bridge_artifact_sha256 is None
        and selected.tokenizer_transport_profile_sha256 is None
        and selected.vocabulary_size == descriptor.vocab_size
        and descriptor_evidence == identity_evidence
        and decision_evidence == identity_evidence
    )


def _derive_heterogeneous_native_target_contract(
    target: DynamicTargetSynthesisResult,
    *,
    selected: Lfm2DenseBudgetedSelectedTarget,
    descriptor: Lfm2HeterogeneousPhysicalLayoutDescriptor,
    selection: Lfm2HeterogeneousLayoutSelectionReceipt,
    receipt: DenseComponentAllocationReceipt,
    parent_count: int,
    bridge_sha256: str | None,
    request_fingerprint: str,
    topology_fingerprint: str,
    graph_fingerprint: str,
    source_fingerprint: str,
    target_count: int,
    quarter_ceiling: int,
) -> Lfm2HeterogeneousNativeExact4xTargetContract:
    """Bind a selected ragged layout without treating scaffold widths as real."""

    if any(
        layer.short_conv_writer_rank is not None
        for layer in descriptor.layers
        if layer.layer_type == "conv"
    ):
        raise Lfm2NativeExact4xTargetError(
            "The first heterogeneous native execution lane accepts only "
            "independent dense ShortConv writers."
        )
    scaffold_topology = _heterogeneous_scaffold_topology(descriptor)
    model_topology = _model_topology(target.model)
    selected_scaffold = {
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
    for layer in descriptor.layers:
        if layer.layer_type != "full_attention":
            continue
        assert layer.query_heads is not None
        assert layer.key_value_heads is not None
        assert layer.head_dim is not None
        if (
            layer.query_heads < layer.key_value_heads
            or layer.query_heads % layer.key_value_heads != 0
        ):
            raise Lfm2NativeExact4xTargetError(
                "A heterogeneous target attention layer has invalid grouped-"
                "query geometry."
            )
        try:
            validate_gqa_attention_fold_geometry(
                source_query_heads=LFM2_NATIVE_EXACT4X_SOURCE_QUERY_HEADS,
                source_key_value_heads=(LFM2_NATIVE_EXACT4X_SOURCE_KEY_VALUE_HEADS),
                source_head_dim=LFM2_NATIVE_EXACT4X_SOURCE_HEAD_DIM,
                target_query_heads=layer.query_heads,
                target_key_value_heads=layer.key_value_heads,
                target_head_dim=layer.head_dim,
            )
        except AttentionFoldError as exc:
            raise Lfm2NativeExact4xTargetError(
                "A heterogeneous target GQA layer cannot populate every "
                "local-query channel from the registered source geometry."
            ) from exc
    decision = selection.allocation_decision
    allocation = receipt.allocation_selection
    allocation_topology = allocation.topology_selection
    parent_graphs = dict(target.parent_graph_fingerprints)
    counts = (
        getattr(target.request, "exact_unique_parameters", None),
        getattr(target.resource_profile, "unique_parameters", None),
        getattr(target.graph, "unique_parameter_count", None),
        getattr(target.graph, "logical_parameter_count", None),
        receipt.target_unique_parameters,
        receipt.target_logical_parameters,
        selected.exact_target_logical_parameters,
        descriptor.accounting.unique_parameter_elements,
        descriptor.accounting.logical_parameter_elements,
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
        or receipt.physical_layout_kind != LFM2_HETEROGENEOUS_LAYOUT_KIND
        or receipt.physical_layout_descriptor_sha256 != descriptor.descriptor_sha256
        or receipt.physical_layout_selection_receipt_sha256 != selection.receipt_sha256
        or receipt.physical_layout_source_topology_sha256
        != descriptor.source_topology_sha256
        or descriptor.source_topology_sha256 != source_fingerprint
        or descriptor.allocation_evidence_sha256 != decision.allocation_decision_sha256
        or decision.source_graph_fingerprint != source_fingerprint
        or decision.exact_target_unique_parameters != target_count
        or allocation.problem_fingerprint != decision.budget_problem_fingerprint
        or allocation.fingerprint != decision.budget_selection_fingerprint
        or allocation_topology.problem_fingerprint
        != decision.exact_topology_problem_fingerprint
        or allocation_topology.fingerprint
        != decision.exact_topology_selection_fingerprint
        or allocation.selected_option_ids != decision.selected_layer_option_sha256
        or decision.selected_layer_option_sha256
        != tuple(layer.selection_evidence_sha256 for layer in descriptor.layers)
        or selected_scaffold != scaffold_topology
        or model_topology != scaffold_topology
        or selected.fir_length != descriptor.conv_kernel_size
        or selected.ordered_source_spans
        != tuple((index,) for index in range(descriptor.layer_count))
        or selected.tie_word_embeddings
        or selected.target_token_interface_mode != "untied"
        or not _heterogeneous_token_transport_is_authenticated(
            selected,
            descriptor,
            selection,
            bridge_sha256,
        )
        or parent_graphs.get(selected.source_parent) != source_fingerprint
        or getattr(target.resource_profile, "family", None) != "lfm2"
        or tuple(getattr(target.resource_profile, "modalities", ())) != ("text",)
        or getattr(target.resource_profile, "layer_count", None)
        != descriptor.layer_count
        or not _heterogeneous_graph_is_exact(target, descriptor)
    ):
        raise Lfm2NativeExact4xTargetError(
            "Synthesized heterogeneous target differs from its authenticated "
            "descriptor, selection, graph, exact counts, or allocation receipt."
        )
    topology = tuple(
        (key, tuple(value) if key == "layer_types" else value)
        for key, value in scaffold_topology.items()
    )
    return Lfm2HeterogeneousNativeExact4xTargetContract(
        target_class_id=LFM2_HETEROGENEOUS_NATIVE_EXACT4X_TARGET_CLASS_ID,
        parent_unique_parameters=parent_count,
        quarter_parameter_ceiling=quarter_ceiling,
        target_unique_parameters=target_count,
        topology=topology,
        target_topology_fingerprint=topology_fingerprint,
        target_graph_fingerprint=graph_fingerprint,
        target_request_fingerprint=request_fingerprint,
        source_graph_fingerprint=source_fingerprint,
        component_allocation_receipt_sha256=receipt.fingerprint,
        component_allocation_topology_sha256=(receipt.topology_configuration_sha256),
        tokenizer_bridge_artifact_sha256=bridge_sha256,
        source_query_heads=LFM2_NATIVE_EXACT4X_SOURCE_QUERY_HEADS,
        source_key_value_heads=LFM2_NATIVE_EXACT4X_SOURCE_KEY_VALUE_HEADS,
        source_head_dim=LFM2_NATIVE_EXACT4X_SOURCE_HEAD_DIM,
        physical_layout_descriptor=descriptor,
        layout_selection_receipt=selection,
    )


def derive_lfm2_native_exact4x_target_contract(
    target: DynamicTargetSynthesisResult,
    *,
    expected_parent_unique_parameters: int,
    expected_bridge_artifact_sha256: str | None,
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
    heterogeneous = _restore_heterogeneous_target_artifacts(target, selected)
    bridge_sha256 = (
        None
        if heterogeneous is not None and expected_bridge_artifact_sha256 is None
        else _digest(
            expected_bridge_artifact_sha256,
            "expected_bridge_artifact_sha256",
        )
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
    if heterogeneous is not None:
        descriptor, selection = heterogeneous
        return _derive_heterogeneous_native_target_contract(
            target,
            selected=selected,
            descriptor=descriptor,
            selection=selection,
            receipt=receipt,
            parent_count=parent_count,
            bridge_sha256=bridge_sha256,
            request_fingerprint=request_fingerprint,
            topology_fingerprint=topology_fingerprint,
            graph_fingerprint=graph_fingerprint,
            source_fingerprint=source_fingerprint,
            target_count=target_count,
            quarter_ceiling=quarter_ceiling,
        )
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
    target_attention_instantiable = (
        selected.query_heads >= selected.key_value_heads
        and selected.query_heads % selected.key_value_heads == 0
        and selected.query_heads * selected.head_dim <= selected.hidden_size
        and selected.key_value_heads * selected.head_dim <= selected.hidden_size
    )
    try:
        validate_gqa_attention_fold_geometry(
            source_query_heads=LFM2_NATIVE_EXACT4X_SOURCE_QUERY_HEADS,
            source_key_value_heads=(LFM2_NATIVE_EXACT4X_SOURCE_KEY_VALUE_HEADS),
            source_head_dim=LFM2_NATIVE_EXACT4X_SOURCE_HEAD_DIM,
            target_query_heads=selected.query_heads,
            target_key_value_heads=selected.key_value_heads,
            target_head_dim=selected.head_dim,
        )
    except AttentionFoldError as exc:
        raise Lfm2NativeExact4xTargetError(
            "Native exact-4x target GQA geometry cannot populate every target "
            "local-query channel from the registered source geometry."
        ) from exc
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
        or receipt.target_logical_parameters != selected.exact_target_logical_parameters
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
        or not target_attention_instantiable
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
        component_allocation_topology_sha256=(receipt.topology_configuration_sha256),
        tokenizer_bridge_artifact_sha256=bridge_sha256,
        source_query_heads=LFM2_NATIVE_EXACT4X_SOURCE_QUERY_HEADS,
        source_key_value_heads=LFM2_NATIVE_EXACT4X_SOURCE_KEY_VALUE_HEADS,
        source_head_dim=LFM2_NATIVE_EXACT4X_SOURCE_HEAD_DIM,
    )


__all__ = [
    "HETEROGENEOUS_SCHEMA",
    "LFM2_IDENTITY_TOKENIZER_EVIDENCE_SCHEMA",
    "LFM2_HETEROGENEOUS_NATIVE_EXACT4X_TARGET_CLASS_ID",
    "LFM2_NATIVE_EXACT4X_HEAD_DIM",
    "LFM2_NATIVE_EXACT4X_LAYER_TYPES",
    "LFM2_NATIVE_EXACT4X_SOURCE_HEAD_DIM",
    "LFM2_NATIVE_EXACT4X_SOURCE_KEY_VALUE_HEADS",
    "LFM2_NATIVE_EXACT4X_SOURCE_QUERY_HEADS",
    "LFM2_NATIVE_EXACT4X_TARGET_CLASS_ID",
    "LFM2_NATIVE_EXACT4X_VOCABULARY_SIZE",
    "Lfm2HeterogeneousNativeExact4xTargetContract",
    "Lfm2NativeExact4xTargetContract",
    "Lfm2NativeExact4xTargetError",
    "derive_lfm2_native_exact4x_target_contract",
    "lfm2_identity_tokenizer_evidence_sha256",
]
