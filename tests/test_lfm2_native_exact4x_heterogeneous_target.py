from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_budget import (
    ArchitectureDiscoveredBudgetSelection,
    ExactTopologyBudgetSelection,
)
from neural_forge.dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_parameter_frontier import (
    COMPONENT_ALLOCATION_POLICY,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_KIND,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    DenseComponentAllocationReceipt,
)
from neural_forge.dynamic_target_synthesis import DynamicTargetSynthesisResult
from neural_forge.hf_lfm2_dynamic_blank_materializer import (
    LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
    lfm2_expected_heterogeneous_physical_state_shapes,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (
    LFM2_HETEROGENEOUS_ALLOCATION_DECISION_SCHEMA,
    LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA,
    Lfm2HeterogeneousAllocationDecisionCore,
    Lfm2HeterogeneousLayoutSelectionReceipt,
)
from neural_forge.lfm2_native_exact4x_target import (
    HETEROGENEOUS_SCHEMA,
    LFM2_HETEROGENEOUS_NATIVE_EXACT4X_TARGET_CLASS_ID,
    Lfm2HeterogeneousNativeExact4xTargetContract,
    Lfm2NativeExact4xTargetError,
    derive_lfm2_native_exact4x_target_contract,
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _decision_body(
    *,
    source_graph_fingerprint: str,
    bridge_sha256: str,
    target_count: int,
    selected_options: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema": LFM2_HETEROGENEOUS_ALLOCATION_DECISION_SCHEMA,
        "source_graph_fingerprint": source_graph_fingerprint,
        "source_structural_topology_fingerprint": _sha("source-structural"),
        "source_layer_order_sha256": _sha("source-layer-order"),
        "target_pins_sha256": _sha("target-pins"),
        "target_vocab_evidence_sha256": _sha("target-vocab"),
        "runtime_evidence_sha256": _sha("runtime"),
        "token_bridge_evidence_sha256": bridge_sha256,
        "exact_target_unique_parameters": target_count,
        "budget_problem_fingerprint": _sha("budget-problem"),
        "budget_selection_fingerprint": _sha("budget-selection"),
        "exact_topology_problem_fingerprint": _sha("topology-problem"),
        "exact_topology_selection_fingerprint": _sha("topology-selection"),
        "selected_layer_option_sha256": list(selected_options),
        "selected_layer_evidence_sha256": [
            _sha(f"selection-evidence-{index}")
            for index in range(len(selected_options))
        ],
        "total_predicted_risk": 0.0,
        "selection_policy": {
            "one_option_per_discovered_source_layer": True,
            "source_order_derived_from_inspected_graph": True,
            "residual_attention_component_protection": True,
            "risk_is_behavioral_hard_gate": False,
            "exact_parameter_count_required": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
        },
    }


def _selection_artifacts(*, factor_writer: bool = False):
    source_graph_fingerprint = _sha("source-graph")
    bridge_sha256 = _sha("token-bridge")
    option_sha256 = tuple(_sha(f"option-{index}") for index in range(4))
    provisional_layers = (
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=0,
            mlp_intermediate_size=13,
            internal_width=24,
            writer_rank=5 if factor_writer else None,
            selection_evidence_sha256=option_sha256[0],
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=1,
            mlp_intermediate_size=15,
            query_heads=8,
            key_value_heads=4,
            head_dim=8,
            selection_evidence_sha256=option_sha256[1],
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=2,
            mlp_intermediate_size=14,
            internal_width=20,
            selection_evidence_sha256=option_sha256[2],
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=3,
            mlp_intermediate_size=16,
            query_heads=4,
            key_value_heads=2,
            head_dim=8,
            selection_evidence_sha256=option_sha256[3],
        ),
    )
    provisional = build_lfm2_heterogeneous_physical_layout(
        hidden_size=32,
        vocab_size=64,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=2048,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        layers=provisional_layers,
        source_topology_sha256=source_graph_fingerprint,
        allocation_evidence_sha256="0" * 64,
        token_bridge_evidence_sha256=bridge_sha256,
    )
    target_count = provisional.accounting.unique_parameter_elements
    decision_body = _decision_body(
        source_graph_fingerprint=source_graph_fingerprint,
        bridge_sha256=bridge_sha256,
        target_count=target_count,
        selected_options=option_sha256,
    )
    decision = Lfm2HeterogeneousAllocationDecisionCore(
        source_graph_fingerprint=source_graph_fingerprint,
        source_structural_topology_fingerprint=_sha("source-structural"),
        source_layer_order_sha256=_sha("source-layer-order"),
        target_pins_sha256=_sha("target-pins"),
        target_vocab_evidence_sha256=_sha("target-vocab"),
        runtime_evidence_sha256=_sha("runtime"),
        token_bridge_evidence_sha256=bridge_sha256,
        exact_target_unique_parameters=target_count,
        budget_problem_fingerprint=_sha("budget-problem"),
        budget_selection_fingerprint=_sha("budget-selection"),
        exact_topology_problem_fingerprint=_sha("topology-problem"),
        exact_topology_selection_fingerprint=_sha("topology-selection"),
        selected_layer_option_sha256=option_sha256,
        selected_layer_evidence_sha256=tuple(
            _sha(f"selection-evidence-{index}") for index in range(4)
        ),
        total_predicted_risk=0.0,
        allocation_decision_sha256=canonical_sha256(decision_body),
    )
    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=32,
        vocab_size=64,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=2048,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        layers=provisional_layers,
        source_topology_sha256=source_graph_fingerprint,
        allocation_evidence_sha256=decision.allocation_decision_sha256,
        token_bridge_evidence_sha256=bridge_sha256,
        expected_unique_parameter_count=target_count,
    )
    selection_body = {
        "schema": LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA,
        "allocation_decision": decision.as_dict(),
        "descriptor_sha256": descriptor.descriptor_sha256,
        "exact_unique_parameter_count": target_count,
        "allocation_precedes_descriptor": True,
    }
    selection = Lfm2HeterogeneousLayoutSelectionReceipt(
        allocation_decision=decision,
        descriptor_sha256=descriptor.descriptor_sha256,
        receipt_sha256=canonical_sha256(selection_body),
    )
    return descriptor, selection, source_graph_fingerprint, bridge_sha256


def _target_stub(*, factor_writer: bool = False):
    descriptor, selection, source_graph, bridge_sha256 = _selection_artifacts(
        factor_writer=factor_writer
    )
    target_count = descriptor.accounting.unique_parameter_elements
    parent_count = 4 * target_count + 3
    option_id = "heterogeneous-physical-selection"
    topology_selection = ExactTopologyBudgetSelection(
        problem_fingerprint=_sha("receipt-topology-problem"),
        selected_option_ids=tuple(
            layer.selection_evidence_sha256 for layer in descriptor.layers
        ),
        shared_resource_ids=(),
        exact_unique_parameters=target_count,
        serialized_bytes=4 * target_count,
        total_predicted_risk=0.0,
        states_considered=1,
        peak_retained_states=1,
    )
    allocation = ArchitectureDiscoveredBudgetSelection(
        problem_fingerprint=_sha("receipt-allocation-problem"),
        source_graph_fingerprint=source_graph,
        topology_selection=topology_selection,
        rule_retention=(),
        semantic_retention=(),
        discovered_group_kinds=(),
    )
    topology_sha256 = _sha("target-topology-configuration")
    receipt = DenseComponentAllocationReceipt(
        policy=COMPONENT_ALLOCATION_POLICY,
        topology_option_id=option_id,
        topology_configuration_sha256=topology_sha256,
        source_graph_fingerprint=source_graph,
        source_unique_parameters=parent_count,
        source_logical_parameters=parent_count,
        target_unique_parameters=target_count,
        target_logical_parameters=target_count,
        allocation_selection=allocation,
        physical_layout_kind=LFM2_HETEROGENEOUS_LAYOUT_KIND,
        physical_layout_descriptor_sha256=descriptor.descriptor_sha256,
        physical_layout_selection_receipt_sha256=selection.receipt_sha256,
        physical_layout_source_topology_sha256=source_graph,
    )
    configuration = {
        LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY: (
            LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        ),
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY: (
            descriptor.serialize().decode("ascii")
        ),
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY: (
            descriptor.descriptor_sha256
        ),
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY: (
            selection.serialize().decode("ascii")
        ),
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY: (
            selection.receipt_sha256
        ),
    }
    first_attention = next(
        layer for layer in descriptor.layers if layer.layer_type == "full_attention"
    )
    selected = object.__new__(Lfm2DenseBudgetedSelectedTarget)
    for name, value in {
        "source_parent": "parent",
        "source_graph_fingerprint": source_graph,
        "source_state_fingerprint": _sha("parent-state"),
        "parent_tokenizer_semantics_sha256": _sha("parent-tokenizer"),
        "child_tokenizer_semantics_sha256": _sha("child-tokenizer"),
        "tokenizer_bridge_artifact_sha256": bridge_sha256,
        "tokenizer_transport_profile_sha256": _sha("bridge-profile"),
        "catalog_id": "heterogeneous-test-catalog",
        "component_allocation_policy": COMPONENT_ALLOCATION_POLICY,
        "component_allocation_receipt_sha256": receipt.fingerprint,
        "component_allocation_topology_sha256": topology_sha256,
        "hidden_size": descriptor.hidden_size,
        "intermediate_size": descriptor.layers[0].mlp_intermediate_size,
        "query_heads": first_attention.query_heads,
        "key_value_heads": first_attention.key_value_heads,
        "head_dim": first_attention.head_dim,
        "vocabulary_size": descriptor.vocab_size,
        "fir_length": descriptor.conv_kernel_size,
        "layer_types": tuple(layer.layer_type for layer in descriptor.layers),
        "ordered_source_spans": tuple(
            (index,) for index in range(descriptor.layer_count)
        ),
        "tie_word_embeddings": False,
        "exact_target_unique_parameters": target_count,
        "exact_target_logical_parameters": target_count,
        "configuration": tuple(sorted(configuration.items())),
        "heterogeneous_layout_descriptor": descriptor,
        "heterogeneous_layout_selection_receipt": selection,
    }.items():
        object.__setattr__(selected, name, value)
    config = SimpleNamespace(
        vocab_size=descriptor.vocab_size,
        hidden_size=descriptor.hidden_size,
        intermediate_size=descriptor.layers[0].mlp_intermediate_size,
        num_hidden_layers=descriptor.layer_count,
        num_attention_heads=first_attention.query_heads,
        num_key_value_heads=first_attention.key_value_heads,
        head_dim=first_attention.head_dim,
        tie_word_embeddings=False,
        layer_types=[layer.layer_type for layer in descriptor.layers],
        neural_forge_dynamic_blank_config_schema=(LFM2_DYNAMIC_BLANK_CONFIG_SCHEMA),
        neural_forge_dynamic_physical_layout=descriptor.as_dict(),
        neural_forge_dynamic_physical_layout_sha256=(descriptor.descriptor_sha256),
    )
    state_key_count = len(lfm2_expected_heterogeneous_physical_state_shapes(descriptor))
    graph = SimpleNamespace(
        fingerprint=_sha("target-graph"),
        logical_parameter_count=target_count,
        unique_parameter_count=target_count,
        unique_parameter_bytes=4 * target_count,
        metadata={
            "physical_layout": "heterogeneous_dense_fp32",
            "physical_layout_descriptor_sha256": descriptor.descriptor_sha256,
            "physical_layout_evidence_binding_sha256": (
                descriptor.evidence_binding_sha256
            ),
            "physical_materialization_audit_sha256": _sha("physical-audit"),
            "blank_materialization_receipt_sha256": _sha("materialization"),
            "physical_parameter_state_key_count": state_key_count,
            "runtime_parameter_alias_group_count": 0,
            "cross_layer_parameter_sharing": False,
            "runtime_bridge_parameter_elements": 0,
            "runtime_sidecar_parameter_elements": 0,
            "parameters_quantized": 0,
            "sparse_parameter_storage": False,
            "pruning_masks_present": False,
        },
    )
    request = SimpleNamespace(
        component_allocation_receipt=receipt,
        fingerprint=_sha("target-request"),
        exact_unique_parameters=target_count,
        configuration=tuple(sorted(configuration.items())),
    )
    target = object.__new__(DynamicTargetSynthesisResult)
    for name, value in {
        "request": request,
        "adapter_id": Lfm2TargetSynthesisAdapter.adapter_id,
        "adapter_version": Lfm2TargetSynthesisAdapter.version,
        "model": SimpleNamespace(config=config),
        "graph": graph,
        "resource_profile": SimpleNamespace(
            unique_parameters=target_count,
            family="lfm2",
            modalities=("text",),
            layer_count=descriptor.layer_count,
        ),
        "topology_fingerprint": _sha("compiler-topology"),
        "parent_graph_fingerprints": (("parent", source_graph),),
        "adapter_payload": selected,
    }.items():
        object.__setattr__(target, name, value)
    return target, descriptor, selection, receipt, parent_count, bridge_sha256


def _derive(target, parent_count: int, bridge_sha256: str):
    return derive_lfm2_native_exact4x_target_contract(
        target,
        expected_parent_unique_parameters=parent_count,
        expected_bridge_artifact_sha256=bridge_sha256,
    )


def test_heterogeneous_contract_describes_descriptor_owned_per_layer_geometry() -> None:
    target, descriptor, selection, receipt, parent_count, bridge = _target_stub()

    contract = _derive(target, parent_count, bridge)

    assert type(contract) is Lfm2HeterogeneousNativeExact4xTargetContract
    assert contract.target_class_id == (
        LFM2_HETEROGENEOUS_NATIVE_EXACT4X_TARGET_CLASS_ID
    )
    assert contract.target_unique_parameters == (
        descriptor.accounting.unique_parameter_elements
    )
    assert contract.component_allocation_receipt_sha256 == receipt.fingerprint
    payload = contract.as_dict()
    assert payload["schema"] == HETEROGENEOUS_SCHEMA
    assert payload["physical_layout_descriptor"] == descriptor.as_dict()
    assert payload["physical_layout_selection_receipt"] == selection.as_dict()
    assert payload["heterogeneous_per_layer_dimensions"] is True
    assert payload["shared_intermediate_or_operator_widths_assumed"] is False
    layers = payload["target_topology"]["layers"]
    assert [row["mlp_intermediate_size"] for row in layers] == [13, 15, 14, 16]
    assert layers[0]["short_conv_internal_width"] == 24
    assert layers[0]["short_conv_writer"] == {
        "kind": "dense",
        "shape": [32, 24],
    }
    assert (layers[1]["query_heads"], layers[1]["key_value_heads"]) == (8, 4)
    assert layers[1]["head_dim"] == 8
    assert layers[1]["query_projection_width"] == 64
    assert layers[1]["query_projection_width"] > descriptor.hidden_size
    assert layers[3]["query_projection_width"] == 32
    assert (
        sum(row["parameter_accounting"]["total_parameter_elements"] for row in layers)
        + descriptor.accounting.untied_vocab_interface_parameter_elements
        + (descriptor.accounting.final_norm_parameter_elements)
        == descriptor.accounting.unique_parameter_elements
    )
    assert payload["training_used"] is False
    assert payload["pruning_used"] is False
    assert payload["quantization_used"] is False
    assert payload["parent_sidecar_used"] is False


@pytest.mark.parametrize(
    "tamper",
    (
        "graph_count",
        "graph_descriptor",
        "graph_sidecar",
        "request_descriptor_json",
        "model_descriptor",
        "component_receipt",
    ),
)
def test_heterogeneous_contract_fails_closed_on_graph_receipt_and_config_tamper(
    tamper: str,
) -> None:
    target, descriptor, _selection, receipt, parent_count, bridge = _target_stub()
    if tamper == "graph_count":
        target.graph.unique_parameter_count += 1
    elif tamper == "graph_descriptor":
        target.graph.metadata["physical_layout_descriptor_sha256"] = _sha(
            "other-descriptor"
        )
    elif tamper == "graph_sidecar":
        target.graph.metadata["runtime_sidecar_parameter_elements"] = 1
    elif tamper == "request_descriptor_json":
        configuration = dict(target.request.configuration)
        raw = copy.deepcopy(descriptor.as_dict())
        raw["layers"][0]["mlp_intermediate_size"] += 1
        configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY] = str(raw)
        target.request.configuration = tuple(sorted(configuration.items()))
    elif tamper == "model_descriptor":
        raw = copy.deepcopy(descriptor.as_dict())
        raw["layers"][0]["short_conv_internal_width"] += 1
        target.model.config.neural_forge_dynamic_physical_layout = raw
    else:
        target.request.component_allocation_receipt = replace(
            receipt,
            physical_layout_descriptor_sha256=_sha("other-layout"),
        )

    with pytest.raises(Lfm2NativeExact4xTargetError):
        _derive(target, parent_count, bridge)


def test_heterogeneous_contract_rejects_factor_writer_before_execution() -> None:
    target, _descriptor, _selection, _receipt, parent_count, bridge = _target_stub(
        factor_writer=True
    )

    with pytest.raises(
        Lfm2NativeExact4xTargetError,
        match="only independent dense ShortConv writers",
    ):
        _derive(target, parent_count, bridge)
