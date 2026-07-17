from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import CircuitKind, EdgeKind
from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    InspectorRegistry,
    inspect_model_circuits,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_llama(*, hidden_size: int = 16):
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )
    return transformers.LlamaForCausalLM(config)


def test_llama_inspection_covers_every_parameter_and_causal_coupling() -> None:
    model = _tiny_llama()
    graph = inspect_model_circuits(model, model_name="tiny-llama")

    assert graph.family == "llama"
    assert graph.layer_indices == (0, 1)
    assert set(graph.parameter_paths) == {
        name
        for name, _value in model.named_parameters(recurse=True, remove_duplicate=False)
    }
    assert graph.nodes_of_kind(CircuitKind.ATTENTION, layer_index=0)[0].kind == (
        "attention"
    )
    assert graph.nodes_of_kind(CircuitKind.GATED_MLP, layer_index=1)[0].kind == (
        "gated_mlp"
    )
    assert any(edge.kind == EdgeKind.RESIDUAL.value for edge in graph.edges)
    assert any(
        edge.kind == EdgeKind.CACHE.value and edge.recurrent for edge in graph.edges
    )
    assert {value.modality for value in graph.interfaces} == {"text"}
    assert graph.logical_parameter_count == sum(
        int(value.numel())
        for _name, value in model.named_parameters(recurse=True, remove_duplicate=False)
    )
    assert graph.unique_parameter_count == sum(
        int(value.numel()) for value in model.parameters()
    )


def test_inspector_fails_closed_for_unknown_model_family() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="unknown"))

    with pytest.raises(DynamicModelInspectionError, match="No registered inspector"):
        inspect_model_circuits(model, model_name="unknown")


def test_registry_rejects_ambiguous_highest_priority() -> None:
    class Inspector:
        priority = 10

        def __init__(self, name: str) -> None:
            self.name = name

        def supports(self, _model) -> bool:
            return True

        def inspect(self, *_args, **_kwargs):
            raise AssertionError

    registry = InspectorRegistry((Inspector("a"), Inspector("b")))

    with pytest.raises(DynamicModelInspectionError, match="equal highest priority"):
        registry.select(object())
