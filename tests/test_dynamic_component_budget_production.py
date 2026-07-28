from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_budget import (
    ComponentProtectionRule,
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_MLP_PRODUCER,
    SEMANTIC_STATE_MIXER_INTERNAL,
)
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_FFN_WIDTH_PROJECTION,
    CAP_HYBRID_OPERATOR_COMPOSITION,
    CAP_ORDERED_DEPTH_COMPOSITION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
    COMPONENT_ALLOCATION_POLICY,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
    component_allocation_core_configuration,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_TOKENIZER_FINGERPRINT = canonical_sha256(
    {"kind": "component-budget-production-test-tokenizer", "rows": 64}
)
_ALL_LEGAL_LFM2_TOPOLOGY_CAPABILITIES = (
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_FFN_WIDTH_PROJECTION,
    CAP_HYBRID_OPERATOR_COMPOSITION,
    CAP_ORDERED_DEPTH_COMPOSITION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
)


def _parent(seed: int = 901):
    torch.manual_seed(seed)
    configuration = transformers.Lfm2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=[
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(configuration)
    model.eval()
    return model


def _graph(parent):
    return inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=_TOKENIZER_FINGERPRINT,
    )


def _focused_catalog(semantic: str) -> Lfm2DenseTargetTopologyCatalog:
    return Lfm2DenseTargetTopologyCatalog(
        compiler_capabilities=_ALL_LEGAL_LFM2_TOPOLOGY_CAPABILITIES,
        component_protection_rules=(
            ComponentProtectionRule(
                "test.focused-role",
                parameter_semantics=(semantic,),
                loss_weight=1_000.0,
            ),
        ),
    )


def _plan(graph, semantic: str):
    return plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            maximum_candidates=12,
        ),
        child_name=f"role-aware-{semantic}",
        tokenizer_fingerprint=_TOKENIZER_FINGERPRINT,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (_focused_catalog(semantic),)
        ),
    )


def _selected(plan):
    return plan.frontier.candidates[
        plan.target_certificate.selected_candidate_rank
    ]


def _rename_module_and_layer_paths(graph):
    node_ids = {
        node.node_id: f"renamed.component.{node.order:03d}"
        for node in graph.nodes
    }
    storage_ids = {
        parameter.storage_group: f"renamed.storage.{index:03d}"
        for index, parameter in enumerate(
            {
                parameter.storage_group: parameter
                for node in graph.nodes
                for parameter in node.parameters
            }.values()
        )
    }
    tensor_index = 0
    nodes = []
    for node in graph.nodes:
        parameters = []
        for parameter in node.parameters:
            parameters.append(
                replace(
                    parameter,
                    path=f"renamed.tensor.{tensor_index:03d}",
                    storage_group=storage_ids[parameter.storage_group],
                )
            )
            tensor_index += 1
        nodes.append(
            replace(
                node,
                node_id=node_ids[node.node_id],
                path=f"renamed.layer_path.{node.order:03d}",
                parameters=tuple(parameters),
            )
        )
    return replace(
        graph,
        model_name="renamed-parent",
        nodes=tuple(nodes),
        edges=tuple(
            replace(
                edge,
                source_node=node_ids[edge.source_node],
                target_node=node_ids[edge.target_node],
            )
            for edge in graph.edges
        ),
        interfaces=tuple(
            replace(interface, node_id=node_ids[interface.node_id])
            for interface in graph.interfaces
        ),
        config_fingerprint=canonical_sha256(
            {"renamed_module_and_layer_paths": True}
        ),
    )


def test_budgeted_plan_changes_global_topology_from_discovered_component_roles(
) -> None:
    graph = _graph(_parent())
    state_plan = _plan(graph, SEMANTIC_STATE_MIXER_INTERNAL)
    query_key_plan = _plan(graph, SEMANTIC_ATTENTION_QUERY_KEY)
    mlp_plan = _plan(graph, SEMANTIC_MLP_PRODUCER)

    state = _selected(state_plan)
    query_key = _selected(query_key_plan)
    mlp = _selected(mlp_plan)
    state_configuration = dict(state.configuration)
    query_key_configuration = dict(query_key.configuration)
    mlp_configuration = dict(mlp.configuration)

    assert state_configuration["num_hidden_layers"] == 5
    assert query_key_configuration["hidden_size"] == 48
    assert mlp_configuration["intermediate_size"] > (
        state_configuration["intermediate_size"]
    )
    # The finite zero-role log barrier can legitimately make two evidence
    # focuses share one balanced topology when every higher-retention
    # alternative for one focus deletes another discovered internal role.
    assert len({
        tuple(component_allocation_core_configuration(candidate.configuration))
        for candidate in (state, query_key, mlp)
    }) >= 2
    assert state.component_allocation_receipt.semantic_retention[
        SEMANTIC_STATE_MIXER_INTERNAL
    ] > query_key.component_allocation_receipt.semantic_retention[
        SEMANTIC_STATE_MIXER_INTERNAL
    ]
    assert query_key.component_allocation_receipt.semantic_retention[
        SEMANTIC_ATTENTION_QUERY_KEY
    ] > state.component_allocation_receipt.semantic_retention[
        SEMANTIC_ATTENTION_QUERY_KEY
    ]

    exact_plan = plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=state.unique_parameters,
            match_policy="exact",
            maximum_candidates=12,
        ),
        child_name="role-aware-exact-count",
        tokenizer_fingerprint=_TOKENIZER_FINGERPRINT,
        catalogs=DenseTargetTopologyCatalogRegistry(
            (_focused_catalog(SEMANTIC_STATE_MIXER_INTERNAL),)
        ),
    )
    exact_candidate = _selected(exact_plan)
    assert exact_candidate.unique_parameters == state.unique_parameters
    assert exact_plan.frontier.requested_unique_parameters == state.unique_parameters
    assert exact_candidate.exact_budget_match is True

    for plan, candidate in (
        (state_plan, state),
        (query_key_plan, query_key),
        (mlp_plan, mlp),
    ):
        receipt = candidate.component_allocation_receipt
        assert receipt is not None
        assert receipt.policy == COMPONENT_ALLOCATION_POLICY
        assert candidate.unique_parameters == plan.target_request.exact_unique_parameters
        assert receipt.target_unique_parameters == candidate.unique_parameters
        assert receipt.allocation_selection.exact_unique_parameters == (
            candidate.unique_parameters
        )
        assert receipt.as_dict()["heterogeneous_per_layer_widths_claimed"] is False
        assert plan.target_certificate.component_allocation_receipt == receipt
        assert plan.target_certificate.selection_policy == (
            "architecture_discovered_component_risk_executable_v1"
        )


def test_component_role_selection_is_invariant_to_module_and_layer_path_names(
) -> None:
    original = _graph(_parent(902))
    renamed = _rename_module_and_layer_paths(original)

    original_plan = _plan(original, SEMANTIC_ATTENTION_QUERY_KEY)
    renamed_plan = _plan(renamed, SEMANTIC_ATTENTION_QUERY_KEY)
    original_candidate = _selected(original_plan)
    renamed_candidate = _selected(renamed_plan)
    original_receipt = original_candidate.component_allocation_receipt
    renamed_receipt = renamed_candidate.component_allocation_receipt

    assert original.fingerprint != renamed.fingerprint
    assert component_allocation_core_configuration(
        original_candidate.configuration
    ) == component_allocation_core_configuration(renamed_candidate.configuration)
    assert original_candidate.unique_parameters == renamed_candidate.unique_parameters
    assert original_receipt.semantic_retention == renamed_receipt.semantic_retention
    assert original_receipt.total_predicted_risk == pytest.approx(
        renamed_receipt.total_predicted_risk
    )
    assert original_receipt.source_graph_fingerprint == original.fingerprint
    assert renamed_receipt.source_graph_fingerprint == renamed.fingerprint


def test_certified_component_allocation_materializes_the_same_exact_topology(
) -> None:
    parent = _parent(903)
    graph = _graph(parent)
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=canonical_sha256({"kind": "sealed-test-cache"}),
        device="cpu",
        dtype=torch.float32,
    )
    adapter_registry = DynamicTargetAdapterRegistry((adapter,))
    plan = plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            maximum_candidates=8,
        ),
        child_name="component-certified-blank",
        tokenizer_fingerprint=_TOKENIZER_FINGERPRINT,
        adapter_id=adapter.adapter_id,
    )
    candidate = _selected(plan)
    receipt = plan.target_certificate.component_allocation_receipt
    assert plan.target_request.component_allocation_receipt == receipt

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=adapter_registry,
    ).synthesize_planned(
        plan=plan,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={
            "parent": _TOKENIZER_FINGERPRINT,
        },
    )

    assert receipt is not None
    assert result.graph.unique_parameter_count == candidate.unique_parameters
    assert result.graph.logical_parameter_count == candidate.logical_parameters
    assert dict(result.adapter_report)[
        "component_allocation_receipt_sha256"
    ] == receipt.fingerprint
    assert dict(result.adapter_report)[
        "component_allocation_verified_by_target_adapter"
    ] is True
    assert getattr(
        result.model.config,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    ) == receipt.fingerprint
    assert int(result.model.config.hidden_size) == dict(candidate.configuration)[
        "hidden_size"
    ]
    assert int(result.model.config.intermediate_size) == dict(
        candidate.configuration
    )["intermediate_size"]


def test_forged_component_receipt_digest_without_typed_binding_fails_closed(
) -> None:
    graph = _graph(_parent(904))
    plan = plan_budgeted_dynamic_target(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=4,
            maximum_candidates=8,
        ),
        child_name="forged-component-receipt",
        tokenizer_fingerprint=_TOKENIZER_FINGERPRINT,
    )
    configuration = dict(plan.target_request.configuration)
    configuration[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY] = "0" * 64

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="require their authenticated",
    ):
        replace(
            plan.target_request,
            configuration=tuple(configuration.items()),
            component_allocation_receipt=None,
        )

    with pytest.raises(
        DynamicTargetSynthesisError,
        match="Authenticated component-allocation receipt differs",
    ):
        replace(
            plan.target_request,
            configuration=tuple(configuration.items()),
        )
