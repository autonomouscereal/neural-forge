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
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
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
    lfm2_identity_tokenizer_evidence_sha256,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _decision_body(
    *,
    source_graph_fingerprint: str,
    bridge_sha256: str,
    target_count: int,
    selected_options: tuple[str, ...],
    allocation_selection: ArchitectureDiscoveredBudgetSelection,
) -> dict[str, object]:
    topology_selection = allocation_selection.topology_selection
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
        "budget_problem_fingerprint": allocation_selection.problem_fingerprint,
        "budget_selection_fingerprint": allocation_selection.fingerprint,
        "exact_topology_problem_fingerprint": (topology_selection.problem_fingerprint),
        "exact_topology_selection_fingerprint": topology_selection.fingerprint,
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


def _selection_artifacts(
    *,
    factor_writer: bool = False,
    identity_tokenizer: bool = False,
    invalid_identity_evidence: bool = False,
):
    source_graph_fingerprint = _sha("source-graph")
    if identity_tokenizer:
        parent_tokenizer_semantics = _sha("shared-tokenizer-semantics")
        child_tokenizer_semantics = parent_tokenizer_semantics
        bridge_sha256 = None
        token_evidence_sha256 = (
            _sha("untrusted-identity-claim")
            if invalid_identity_evidence
            else lfm2_identity_tokenizer_evidence_sha256(
                tokenizer_semantics_sha256=parent_tokenizer_semantics,
                vocabulary_size=64,
            )
        )
    else:
        parent_tokenizer_semantics = _sha("parent-tokenizer")
        child_tokenizer_semantics = _sha("child-tokenizer")
        bridge_sha256 = _sha("token-bridge")
        token_evidence_sha256 = bridge_sha256
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
        token_bridge_evidence_sha256=token_evidence_sha256,
    )
    target_count = provisional.accounting.unique_parameter_elements
    topology_selection = ExactTopologyBudgetSelection(
        problem_fingerprint=_sha("topology-problem"),
        selected_option_ids=option_sha256,
        shared_resource_ids=(),
        exact_unique_parameters=target_count,
        serialized_bytes=4 * target_count,
        total_predicted_risk=0.0,
        states_considered=1,
        peak_retained_states=1,
    )
    allocation_selection = ArchitectureDiscoveredBudgetSelection(
        problem_fingerprint=_sha("budget-problem"),
        source_graph_fingerprint=source_graph_fingerprint,
        topology_selection=topology_selection,
        rule_retention=(),
        semantic_retention=(),
        discovered_group_kinds=(),
    )
    decision_body = _decision_body(
        source_graph_fingerprint=source_graph_fingerprint,
        bridge_sha256=token_evidence_sha256,
        target_count=target_count,
        selected_options=option_sha256,
        allocation_selection=allocation_selection,
    )
    decision = Lfm2HeterogeneousAllocationDecisionCore(
        source_graph_fingerprint=source_graph_fingerprint,
        source_structural_topology_fingerprint=_sha("source-structural"),
        source_layer_order_sha256=_sha("source-layer-order"),
        target_pins_sha256=_sha("target-pins"),
        target_vocab_evidence_sha256=_sha("target-vocab"),
        runtime_evidence_sha256=_sha("runtime"),
        token_bridge_evidence_sha256=token_evidence_sha256,
        exact_target_unique_parameters=target_count,
        budget_problem_fingerprint=allocation_selection.problem_fingerprint,
        budget_selection_fingerprint=allocation_selection.fingerprint,
        exact_topology_problem_fingerprint=(topology_selection.problem_fingerprint),
        exact_topology_selection_fingerprint=topology_selection.fingerprint,
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
        token_bridge_evidence_sha256=token_evidence_sha256,
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
    return (
        descriptor,
        selection,
        source_graph_fingerprint,
        bridge_sha256,
        allocation_selection,
        parent_tokenizer_semantics,
        child_tokenizer_semantics,
    )


def _target_stub(
    *,
    factor_writer: bool = False,
    identity_tokenizer: bool = False,
    invalid_identity_evidence: bool = False,
):
    (
        descriptor,
        selection,
        source_graph,
        bridge_sha256,
        allocation,
        parent_tokenizer_semantics,
        child_tokenizer_semantics,
    ) = _selection_artifacts(
        factor_writer=factor_writer,
        identity_tokenizer=identity_tokenizer,
        invalid_identity_evidence=invalid_identity_evidence,
    )
    target_count = descriptor.accounting.unique_parameter_elements
    parent_count = 4 * target_count + 3
    option_id = "heterogeneous-physical-selection"
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
        "parent_tokenizer_semantics_sha256": parent_tokenizer_semantics,
        "child_tokenizer_semantics_sha256": child_tokenizer_semantics,
        "tokenizer_bridge_artifact_sha256": bridge_sha256,
        "tokenizer_transport_profile_sha256": (
            None if bridge_sha256 is None else _sha("bridge-profile")
        ),
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
    config = transformers.Lfm2Config(
        vocab_size=descriptor.vocab_size,
        hidden_size=descriptor.hidden_size,
        intermediate_size=descriptor.layers[0].mlp_intermediate_size,
        num_hidden_layers=descriptor.layer_count,
        num_attention_heads=first_attention.query_heads,
        num_key_value_heads=first_attention.key_value_heads,
        head_dim=first_attention.head_dim,
        conv_L_cache=descriptor.conv_kernel_size,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        layer_types=[layer.layer_type for layer in descriptor.layers],
        norm_eps=descriptor.norm_eps,
        max_position_embeddings=descriptor.max_position_embeddings,
        use_cache=descriptor.use_cache,
        pad_token_id=descriptor.pad_token_id,
        bos_token_id=descriptor.bos_token_id,
        eos_token_id=descriptor.eos_token_id,
    )
    model = transformers.Lfm2ForCausalLM(config)
    materialization = materialize_lfm2_dynamic_blank_child(
        model,
        config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    audit = materialization["physical_materialization_audit"]
    state_key_count = materialization["parameter_state_key_count"]
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
            "physical_materialization_audit_sha256": audit["result_sha256"],
            "blank_materialization_receipt_sha256": materialization["receipt_sha256"],
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
        "model": model,
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


def _derive(target, parent_count: int, bridge_sha256: str | None):
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


def test_heterogeneous_contract_accepts_authenticated_identity_tokenizer_transport() -> (
    None
):
    target, descriptor, _selection, _receipt, parent_count, bridge = _target_stub(
        identity_tokenizer=True
    )

    contract = _derive(target, parent_count, bridge)

    assert bridge is None
    assert contract.tokenizer_bridge_artifact_sha256 is None
    assert descriptor.token_bridge_evidence_sha256 == (
        lfm2_identity_tokenizer_evidence_sha256(
            tokenizer_semantics_sha256=(
                target.adapter_payload.parent_tokenizer_semantics_sha256
            ),
            vocabulary_size=descriptor.vocab_size,
        )
    )


@pytest.mark.parametrize(
    "tamper",
    ("arbitrary_identity_evidence", "different_tokenizer_semantics"),
)
def test_heterogeneous_contract_rejects_unauthenticated_no_bridge_transport(
    tamper: str,
) -> None:
    target, _descriptor, _selection, _receipt, parent_count, bridge = _target_stub(
        identity_tokenizer=True,
        invalid_identity_evidence=tamper == "arbitrary_identity_evidence",
    )
    if tamper == "different_tokenizer_semantics":
        object.__setattr__(
            target.adapter_payload,
            "child_tokenizer_semantics_sha256",
            _sha("different-child-tokenizer-semantics"),
        )

    with pytest.raises(Lfm2NativeExact4xTargetError):
        _derive(target, parent_count, bridge)


def test_heterogeneous_contract_does_not_treat_a_missing_bridge_as_identity() -> None:
    target, _descriptor, _selection, _receipt, parent_count, _bridge = _target_stub()

    with pytest.raises(Lfm2NativeExact4xTargetError):
        _derive(target, parent_count, None)


@pytest.mark.parametrize(
    "tamper",
    (
        "graph_count",
        "graph_descriptor",
        "graph_audit_digest",
        "graph_materialization_digest",
        "graph_sidecar",
        "request_descriptor_json",
        "model_descriptor",
        "runtime_marker",
        "runtime_parameter_alias",
        "component_receipt",
        "allocation_selection",
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
    elif tamper == "graph_audit_digest":
        target.graph.metadata["physical_materialization_audit_sha256"] = _sha(
            "invented-valid-audit-digest"
        )
    elif tamper == "graph_materialization_digest":
        target.graph.metadata["blank_materialization_receipt_sha256"] = _sha(
            "invented-valid-materialization-digest"
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
    elif tamper == "runtime_marker":
        target.model._neural_forge_dynamic_blank_runtime_sha256 = _sha(
            "another-runtime-layout"
        )
    elif tamper == "runtime_parameter_alias":
        attention = target.model.model.layers[1].self_attn
        attention.k_proj.weight = attention.v_proj.weight
    elif tamper == "component_receipt":
        target.request.component_allocation_receipt = replace(
            receipt,
            physical_layout_descriptor_sha256=_sha("other-layout"),
        )
    else:
        altered_allocation = replace(
            receipt.allocation_selection,
            problem_fingerprint=_sha("other-allocation-problem"),
        )
        altered_receipt = replace(
            receipt,
            allocation_selection=altered_allocation,
        )
        target.request.component_allocation_receipt = altered_receipt
        object.__setattr__(
            target.adapter_payload,
            "component_allocation_receipt_sha256",
            altered_receipt.fingerprint,
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
