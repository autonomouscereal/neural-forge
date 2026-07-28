from __future__ import annotations

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
    DiscoveredComponentChoiceGroup,
    DiscoveredComponentTopologyOption,
    DynamicFoldBudgetError,
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
