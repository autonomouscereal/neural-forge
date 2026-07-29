from __future__ import annotations

import math

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_sha256,
)
from neural_forge.dynamic_fold_budget import (
    ArchitectureDiscoveredBudgetProblem,
    ComponentProtectionRule,
    DiscoveredComponentChoiceGroup,
    DiscoveredComponentTopologyOption,
    DynamicFoldBudgetError,
    DynamicFoldBudgetUnattainable,
    RetainedParameterAllocation,
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_ATTENTION_WRITER,
    SEMANTIC_MLP_PRODUCER,
    SEMANTIC_MLP_WRITER,
    SEMANTIC_STATE_MIXER_INTERNAL,
    solve_architecture_discovered_budget,
)


def _parameter(path: str, role: str, count: int) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        role=role,
        shape=(count,),
        dtype="torch.float32",
        storage_group=path,
        trainable=False,
    )


def _synthetic_graph(*, renamed: bool = False) -> ModelCircuitGraph:
    prefix = "alternate" if renamed else "model"
    layer_offset = 70 if renamed else 0
    specifications = (
        (
            f"{prefix}.interface",
            CircuitKind.TOKEN_EMBEDDING,
            None,
            (("embedding.weight", 20),),
        ),
        (
            f"{prefix}.state_mixer",
            CircuitKind.SHORT_CONV,
            layer_offset + 3,
            (("operator.conv_kernel", 100),),
        ),
        (
            f"{prefix}.attention",
            CircuitKind.ATTENTION,
            layer_offset + 9,
            (
                ("operator.q_proj", 25),
                ("operator.k_proj", 25),
                ("operator.v_proj", 25),
                ("operator.o_proj", 25),
            ),
        ),
        (
            f"{prefix}.producer",
            CircuitKind.GATED_MLP,
            layer_offset + 11,
            (
                ("mlp.gate_proj", 40),
                ("mlp.up_proj", 40),
                ("mlp.down_proj", 20),
            ),
        ),
    )
    nodes = []
    logical = 0
    for order, (node_id, kind, layer_index, roles) in enumerate(specifications):
        parameters = tuple(
            _parameter(
                f"{node_id}.{role.replace('.', '_')}",
                role,
                count,
            )
            for role, count in roles
        )
        logical += sum(value.numel for value in parameters)
        nodes.append(
            CircuitNode(
                node_id=node_id,
                kind=kind,
                order=order,
                path=node_id,
                layer_index=layer_index,
                inputs=(),
                outputs=(),
                parameters=parameters,
            )
        )
    return ModelCircuitGraph(
        model_name=f"synthetic.{prefix}",
        family="synthetic",
        model_class="tests.Synthetic",
        nodes=tuple(reversed(nodes)),
        edges=(),
        interfaces=(),
        logical_parameter_count=logical,
        unique_parameter_count=logical,
        unique_parameter_bytes=logical * 4,
        config_fingerprint=canonical_sha256({"renamed": renamed}),
    )


def _allocation(
    node: CircuitNode,
    retained_by_role: dict[str, int],
) -> tuple[RetainedParameterAllocation, ...]:
    return tuple(
        RetainedParameterAllocation(node.node_id, role, count)
        for role, count in retained_by_role.items()
    )


def _groups_from_discovered_graph(
    graph: ModelCircuitGraph,
) -> tuple[DiscoveredComponentChoiceGroup, ...]:
    """Synthetic catalog: dispatch on IR kinds, never IDs or layer numbers."""

    groups = []
    for node in graph.nodes:
        source = node.logical_parameter_count
        if node.kind == CircuitKind.TOKEN_EMBEDDING.value:
            choices = (
                DiscoveredComponentTopologyOption(
                    "interface.fixed",
                    "interface",
                    (node.node_id,),
                    source,
                    source * 4,
                ),
            )
            group_id = "interface"
        elif node.kind == CircuitKind.SHORT_CONV.value:
            choices = (
                DiscoveredComponentTopologyOption(
                    "mixer.compact",
                    "mixer",
                    (node.node_id,),
                    50,
                    200,
                ),
                DiscoveredComponentTopologyOption(
                    "mixer.full",
                    "mixer",
                    (node.node_id,),
                    source,
                    source * 4,
                ),
            )
            group_id = "mixer"
        elif node.kind == CircuitKind.ATTENTION.value:
            choices = (
                DiscoveredComponentTopologyOption(
                    "attention.compact_qk",
                    "attention",
                    (node.node_id,),
                    50,
                    200,
                    retained_parameters=_allocation(
                        node,
                        {
                            "operator.q_proj": 0,
                            "operator.k_proj": 0,
                            "operator.v_proj": 25,
                            "operator.o_proj": 25,
                        },
                    ),
                ),
                DiscoveredComponentTopologyOption(
                    "attention.compact_writer",
                    "attention",
                    (node.node_id,),
                    50,
                    200,
                    retained_parameters=_allocation(
                        node,
                        {
                            "operator.q_proj": 25,
                            "operator.k_proj": 25,
                            "operator.v_proj": 0,
                            "operator.o_proj": 0,
                        },
                    ),
                ),
                DiscoveredComponentTopologyOption(
                    "attention.full",
                    "attention",
                    (node.node_id,),
                    source,
                    source * 4,
                ),
            )
            group_id = "attention"
        elif node.kind == CircuitKind.GATED_MLP.value:
            choices = (
                DiscoveredComponentTopologyOption(
                    "mlp.compact",
                    "mlp",
                    (node.node_id,),
                    50,
                    200,
                ),
                DiscoveredComponentTopologyOption(
                    "mlp.full",
                    "mlp",
                    (node.node_id,),
                    source,
                    source * 4,
                ),
            )
            group_id = "mlp"
        else:  # pragma: no cover - fixture guards its own architecture
            raise AssertionError(node.kind)
        groups.append(DiscoveredComponentChoiceGroup(group_id, choices))
    return tuple(groups)


def _two_role_graph(
    *,
    query_key_parameters: int,
    mlp_parameters: int,
) -> ModelCircuitGraph:
    nodes = (
        CircuitNode(
            node_id="model.attention",
            kind=CircuitKind.ATTENTION,
            order=0,
            path="model.attention",
            layer_index=0,
            inputs=(),
            outputs=(),
            parameters=(
                _parameter(
                    "model.attention.q_proj",
                    "operator.q_proj",
                    query_key_parameters,
                ),
            ),
        ),
        CircuitNode(
            node_id="model.mlp",
            kind=CircuitKind.GATED_MLP,
            order=1,
            path="model.mlp",
            layer_index=0,
            inputs=(),
            outputs=(),
            parameters=(
                _parameter(
                    "model.mlp.up_proj",
                    "mlp.up_proj",
                    mlp_parameters,
                ),
            ),
        ),
    )
    total = query_key_parameters + mlp_parameters
    return ModelCircuitGraph(
        model_name="two-role",
        family="synthetic",
        model_class="tests.TwoRole",
        nodes=nodes,
        edges=(),
        interfaces=(),
        logical_parameter_count=total,
        unique_parameter_count=total,
        unique_parameter_bytes=total * 4,
        config_fingerprint=canonical_sha256(
            {
                "query_key_parameters": query_key_parameters,
                "mlp_parameters": mlp_parameters,
            }
        ),
    )


def _whole_graph_problem(
    graph: ModelCircuitGraph,
    *,
    options: tuple[
        tuple[str, int, int],
        ...,
    ],
    protection_rules: tuple[ComponentProtectionRule, ...],
    bottleneck_risk_weight: float = 8.0,
) -> ArchitectureDiscoveredBudgetProblem:
    topology_options = tuple(
        DiscoveredComponentTopologyOption(
            option_id=option_id,
            group_id="whole-graph",
            source_node_ids=("model.attention", "model.mlp"),
            local_unique_parameters=query_key_retained + mlp_retained,
            local_serialized_bytes=(
                query_key_retained + mlp_retained
            )
            * 4,
            retained_parameters=(
                RetainedParameterAllocation(
                    "model.attention",
                    "operator.q_proj",
                    query_key_retained,
                ),
                RetainedParameterAllocation(
                    "model.mlp",
                    "mlp.up_proj",
                    mlp_retained,
                ),
            ),
        )
        for option_id, query_key_retained, mlp_retained in options
    )
    target_counts = {
        option.local_unique_parameters for option in topology_options
    }
    assert len(target_counts) == 1
    return ArchitectureDiscoveredBudgetProblem(
        source_graph=graph,
        exact_target_unique_parameters=target_counts.pop(),
        groups=(
            DiscoveredComponentChoiceGroup(
                "whole-graph",
                topology_options,
            ),
        ),
        protection_rules=protection_rules,
        bottleneck_risk_weight=bottleneck_risk_weight,
    )


def _heterogeneous_layer_lattice(
    *,
    reverse_groups: bool = False,
    reverse_options: bool = False,
    target_offset: int = 0,
) -> ArchitectureDiscoveredBudgetProblem:
    nodes = []
    groups = []
    target = 0
    source_total = 0
    for layer_index in range(14):
        node_id = f"model.layers.{layer_index:02d}.mlp"
        source_parameters = 1_000 + 20 * layer_index
        source_total += source_parameters
        nodes.append(
            CircuitNode(
                node_id=node_id,
                kind=CircuitKind.GATED_MLP,
                order=layer_index,
                path=node_id,
                layer_index=layer_index,
                inputs=(),
                outputs=(),
                parameters=(
                    _parameter(
                        f"{node_id}.up_proj",
                        "mlp.up_proj",
                        source_parameters,
                    ),
                ),
            )
        )
        group_id = f"layer.{layer_index:02d}.mlp"
        base_parameters = 120 + 4 * layer_index
        retention_step = 4 + 2 * (layer_index % 3)
        choices = tuple(
            DiscoveredComponentTopologyOption(
                option_id=f"{group_id}.shape.{option_index}",
                group_id=group_id,
                source_node_ids=(node_id,),
                local_unique_parameters=(
                    base_parameters + option_index * retention_step
                ),
                local_serialized_bytes=(
                    base_parameters + option_index * retention_step
                )
                * 4,
                structural_risk=float(3 - option_index) / 100.0,
            )
            for option_index in range(4)
        )
        target += base_parameters + 2 * retention_step
        groups.append(
            DiscoveredComponentChoiceGroup(
                group_id,
                tuple(reversed(choices)) if reverse_options else choices,
            )
        )
    graph = ModelCircuitGraph(
        model_name="heterogeneous-fourteen-layer-lattice",
        family="synthetic",
        model_class="tests.HeterogeneousLayerLattice",
        nodes=tuple(nodes),
        edges=(),
        interfaces=(),
        logical_parameter_count=source_total,
        unique_parameter_count=source_total,
        unique_parameter_bytes=source_total * 4,
        config_fingerprint=canonical_sha256(
            {"kind": "heterogeneous-fourteen-layer-lattice"}
        ),
    )
    return ArchitectureDiscoveredBudgetProblem(
        source_graph=graph,
        exact_target_unique_parameters=target + target_offset,
        groups=tuple(reversed(groups)) if reverse_groups else tuple(groups),
        max_states=256,
    )


@pytest.mark.parametrize("renamed", (False, True))
def test_discovered_allocator_preserves_brittle_components_at_exact_budget(
    renamed: bool,
) -> None:
    graph = _synthetic_graph(renamed=renamed)
    problem = ArchitectureDiscoveredBudgetProblem(
        source_graph=graph,
        exact_target_unique_parameters=220,
        groups=_groups_from_discovered_graph(graph),
    )

    selected = solve_architecture_discovered_budget(problem)

    assert selected.exact_unique_parameters == 220
    assert selected.serialized_bytes == 880
    assert set(selected.selected_option_ids) == {
        "attention.compact_writer",
        "interface.fixed",
        "mixer.full",
        "mlp.compact",
    }
    semantic = {
        audit.scope_id: audit for audit in selected.semantic_retention
    }
    assert semantic[SEMANTIC_STATE_MIXER_INTERNAL].retention_ppm == 1_000_000
    assert semantic[SEMANTIC_ATTENTION_QUERY_KEY].retention_ppm == 1_000_000
    assert semantic[SEMANTIC_ATTENTION_WRITER].retention_ppm == 0
    assert semantic[SEMANTIC_MLP_PRODUCER].retention_ppm == 500_000
    assert semantic[SEMANTIC_MLP_WRITER].retention_ppm == 500_000
    assert selected.as_dict()["complete_parameterized_node_coverage"] is True
    assert selected.as_dict()["exact_global_parameter_budget_met"] is True


def test_discovered_allocator_rejects_incomplete_model_coverage() -> None:
    graph = _synthetic_graph()
    groups = tuple(
        group
        for group in _groups_from_discovered_graph(graph)
        if group.group_id != "mixer"
    )

    with pytest.raises(DynamicFoldBudgetError, match="cover every"):
        ArchitectureDiscoveredBudgetProblem(
            source_graph=graph,
            exact_target_unique_parameters=220,
            groups=groups,
        )


def test_component_risk_is_role_scale_and_rule_expression_invariant() -> None:
    combined_rule = ComponentProtectionRule(
        "combined",
        parameter_semantics=(
            SEMANTIC_ATTENTION_QUERY_KEY,
            SEMANTIC_MLP_PRODUCER,
        ),
        loss_weight=4.0,
    )

    def risk(
        graph: ModelCircuitGraph,
        *,
        query_key_retained: int,
        mlp_retained: int,
        rules: tuple[ComponentProtectionRule, ...],
    ) -> float:
        selected = solve_architecture_discovered_budget(
            _whole_graph_problem(
                graph,
                options=(("candidate", query_key_retained, mlp_retained),),
                protection_rules=rules,
            )
        )
        return selected.total_predicted_risk

    baseline = risk(
        _two_role_graph(query_key_parameters=100, mlp_parameters=100),
        query_key_retained=50,
        mlp_retained=25,
        rules=(combined_rule,),
    )
    independently_scaled = risk(
        _two_role_graph(
            query_key_parameters=10_000,
            mlp_parameters=1_000,
        ),
        query_key_retained=5_000,
        mlp_retained=250,
        rules=(combined_rule,),
    )
    split_rules = (
        ComponentProtectionRule(
            "split.query-key",
            parameter_semantics=(SEMANTIC_ATTENTION_QUERY_KEY,),
            loss_weight=4.0,
        ),
        ComponentProtectionRule(
            "split.mlp",
            parameter_semantics=(SEMANTIC_MLP_PRODUCER,),
            loss_weight=4.0,
        ),
    )
    split = risk(
        _two_role_graph(query_key_parameters=100, mlp_parameters=100),
        query_key_retained=50,
        mlp_retained=25,
        rules=split_rules,
    )
    duplicate = risk(
        _two_role_graph(query_key_parameters=100, mlp_parameters=100),
        query_key_retained=50,
        mlp_retained=25,
        rules=(
            combined_rule,
            ComponentProtectionRule(
                "combined.duplicate-id",
                parameter_semantics=(
                    SEMANTIC_ATTENTION_QUERY_KEY,
                    SEMANTIC_MLP_PRODUCER,
                ),
                loss_weight=4.0,
            ),
        ),
    )

    assert independently_scaled == pytest.approx(baseline)
    assert split == pytest.approx(baseline)
    assert duplicate == pytest.approx(baseline)


def test_log_balance_penalizes_role_collapse_without_forbidding_it() -> None:
    graph = _two_role_graph(
        query_key_parameters=100,
        mlp_parameters=100,
    )
    neutral_rule = ComponentProtectionRule(
        "neutral.explicit-policy",
        parameter_semantics=(
            SEMANTIC_ATTENTION_QUERY_KEY,
            SEMANTIC_MLP_PRODUCER,
        ),
        loss_weight=0.0,
    )
    selected = solve_architecture_discovered_budget(
        _whole_graph_problem(
            graph,
            options=(
                ("balanced", 50, 50),
                ("collapsed", 0, 100),
            ),
            protection_rules=(neutral_rule,),
        )
    )
    unavoidable = solve_architecture_discovered_budget(
        _whole_graph_problem(
            graph,
            options=(("collapsed", 0, 100),),
            protection_rules=(neutral_rule,),
        )
    )

    assert selected.selected_option_ids == ("balanced",)
    assert unavoidable.selected_option_ids == ("collapsed",)
    assert math.isfinite(unavoidable.total_predicted_risk)
    assert (
        unavoidable.total_predicted_risk
        > selected.total_predicted_risk
    )


def test_fourteen_layer_lattice_is_exact_and_permutation_deterministic() -> None:
    canonical_problem = _heterogeneous_layer_lattice()
    permuted_problem = _heterogeneous_layer_lattice(
        reverse_groups=True,
        reverse_options=True,
    )

    canonical = solve_architecture_discovered_budget(canonical_problem)
    permuted = solve_architecture_discovered_budget(permuted_problem)

    assert canonical.exact_unique_parameters == (
        canonical_problem.exact_target_unique_parameters
    )
    assert len(canonical.selected_option_ids) == 14
    assert canonical.selected_option_ids == permuted.selected_option_ids
    assert canonical.fingerprint == permuted.fingerprint
    assert canonical.topology_selection.peak_retained_states <= 256


def test_fourteen_layer_lattice_unattainable_budget_fails_closed() -> None:
    problem = _heterogeneous_layer_lattice(target_offset=1)

    with pytest.raises(DynamicFoldBudgetUnattainable, match="exactly reaches"):
        solve_architecture_discovered_budget(problem)
