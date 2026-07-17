from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitEdge,
    CircuitKind,
    CircuitNode,
    CircuitPort,
    DynamicCircuitIRError,
    EdgeKind,
    ModalityInterface,
    ModelCircuitGraph,
    ParameterSpec,
    PortKind,
    canonical_sha256,
)


def _parameter(path: str, storage: str) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        role="weight",
        shape=(4, 4),
        dtype="torch.float32",
        storage_group=storage,
        trainable=False,
    )


def _graph() -> ModelCircuitGraph:
    embedding = CircuitNode(
        node_id="input.embedding",
        kind=CircuitKind.TOKEN_EMBEDDING,
        order=0,
        path="embed",
        layer_index=None,
        inputs=(CircuitPort("tokens", PortKind.TOKEN_IDS, ("batch", "sequence")),),
        outputs=(
            CircuitPort(
                "residual",
                PortKind.RESIDUAL,
                ("batch", "sequence", 4),
                modality="text",
            ),
        ),
        parameters=(_parameter("embed.weight", "storage-0"),),
    )
    attention = CircuitNode(
        node_id="layer.000.attention",
        kind=CircuitKind.ATTENTION,
        order=1,
        path="layers.0.attention",
        layer_index=0,
        inputs=(
            CircuitPort(
                "residual",
                PortKind.RESIDUAL,
                ("batch", "sequence", 4),
            ),
            CircuitPort("cache_in", PortKind.CACHE, ("layers",), stateful=True),
        ),
        outputs=(
            CircuitPort(
                "delta",
                PortKind.BRANCH_DELTA,
                ("batch", "sequence", 4),
            ),
            CircuitPort("cache_out", PortKind.CACHE, ("layers",), stateful=True),
        ),
        parameters=(_parameter("layers.0.attention.weight", "storage-1"),),
        attributes=(("query_heads", 2), ("key_value_heads", 1)),
    )
    head = CircuitNode(
        node_id="output.head",
        kind=CircuitKind.OUTPUT_HEAD,
        order=2,
        path="head",
        layer_index=None,
        inputs=(
            CircuitPort(
                "residual",
                PortKind.BRANCH_DELTA,
                ("batch", "sequence", 4),
            ),
        ),
        outputs=(
            CircuitPort(
                "logits",
                PortKind.LOGITS,
                ("batch", "sequence", 4),
                modality="text",
            ),
        ),
        parameters=(_parameter("head.weight", "storage-0"),),
    )
    return ModelCircuitGraph(
        model_name="tiny",
        family="test",
        model_class="tests.Tiny",
        nodes=(head, attention, embedding),
        edges=(
            CircuitEdge(
                "input.embedding",
                "residual",
                "layer.000.attention",
                "residual",
                EdgeKind.DATA,
            ),
            CircuitEdge(
                "layer.000.attention",
                "cache_out",
                "layer.000.attention",
                "cache_in",
                EdgeKind.CACHE,
                recurrent=True,
            ),
            CircuitEdge(
                "layer.000.attention",
                "delta",
                "output.head",
                "residual",
                EdgeKind.RESIDUAL,
            ),
        ),
        interfaces=(
            ModalityInterface(
                "text.input",
                "text",
                "input",
                "input.embedding",
                "tokens",
                "token_ids",
            ),
            ModalityInterface(
                "text.output",
                "text",
                "output",
                "output.head",
                "logits",
                "token_logits",
            ),
        ),
        logical_parameter_count=48,
        unique_parameter_count=32,
        unique_parameter_bytes=128,
        config_fingerprint=canonical_sha256({"kind": "tiny"}),
    )


def test_graph_is_canonical_fingerprinted_and_explicitly_stateful() -> None:
    graph = _graph()

    assert [node.node_id for node in graph.nodes] == [
        "input.embedding",
        "layer.000.attention",
        "output.head",
    ]
    assert graph.layer_indices == (0,)
    assert graph.unique_parameter_count == 32
    assert graph.logical_parameter_count == 48
    assert graph.as_dict()["explicit_state_or_cache_edges"] is True
    assert graph.as_dict()["complete_parameter_coverage"] is True
    assert len(graph.fingerprint) == 64
    assert graph.fingerprint == _graph().fingerprint


def test_graph_rejects_backward_nonrecurrent_edges() -> None:
    graph = _graph()
    backward = CircuitEdge(
        "output.head",
        "logits",
        "input.embedding",
        "tokens",
        EdgeKind.DATA,
    )

    with pytest.raises(DynamicCircuitIRError, match="backward"):
        replace(graph, edges=graph.edges + (backward,))


def test_graph_rejects_missing_parameter_coverage() -> None:
    graph = _graph()

    with pytest.raises(DynamicCircuitIRError, match="exactly cover"):
        replace(graph, logical_parameter_count=49)


def test_graph_rejects_incompatible_port_contracts() -> None:
    graph = _graph()
    bad = replace(
        graph.edges[0],
        target_port="cache_in",
    )

    with pytest.raises(DynamicCircuitIRError, match="port kinds"):
        replace(graph, edges=(bad,) + graph.edges[1:])


def test_graph_rejects_nonrecurrent_causal_cycles() -> None:
    graph = _graph()
    cyclic_head = replace(
        graph.node("output.head"),
        order=1,
        outputs=(
            CircuitPort(
                "logits",
                PortKind.RESIDUAL,
                ("batch", "sequence", 4),
                modality="text",
            ),
        ),
    )
    cyclic_attention = replace(graph.node("layer.000.attention"), order=1)
    back = CircuitEdge(
        "output.head",
        "logits",
        "layer.000.attention",
        "residual",
        EdgeKind.DATA,
    )

    with pytest.raises(DynamicCircuitIRError, match="cycle"):
        replace(
            graph,
            nodes=(
                graph.node("input.embedding"),
                cyclic_attention,
                cyclic_head,
            ),
            edges=graph.edges + (back,),
        )
