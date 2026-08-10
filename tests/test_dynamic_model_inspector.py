from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import CircuitKind, EdgeKind
from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    InspectorRegistry,
    _parameter_counts,
    _storage_groups,
    inspect_model_circuits,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
numpy = pytest.importorskip("numpy")


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


def test_storage_groups_accept_exact_ties_and_disjoint_views() -> None:
    class Views(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            tied = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
            self.tied_left = tied
            self.tied_right = tied
            backing = torch.arange(12, dtype=torch.float32)
            self.disjoint_left = torch.nn.Parameter(backing[:6])
            self.disjoint_right = torch.nn.Parameter(backing[6:])

    groups = _storage_groups(Views())
    assert groups["tied_left"] == groups["tied_right"]
    assert groups["disjoint_left"] != groups["disjoint_right"]


def test_storage_groups_fail_closed_for_partial_overlap() -> None:
    class PartialOverlap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backing = torch.arange(12, dtype=torch.float32)
            self.left = torch.nn.Parameter(backing[:8])
            self.right = torch.nn.Parameter(backing[4:])

    with pytest.raises(DynamicModelInspectionError, match="Partially overlapping"):
        _storage_groups(PartialOverlap())


def test_storage_groups_fail_closed_for_internal_overlap() -> None:
    class InternalOverlap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backing = torch.arange(1, dtype=torch.float32)
            self.expanded = torch.nn.Parameter(backing.expand(4))

    with pytest.raises(DynamicModelInspectionError, match="Internally overlapping"):
        _storage_groups(InternalOverlap())


def test_storage_groups_reject_zero_sized_parameters_explicitly() -> None:
    class Empty(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.empty = torch.nn.Parameter(torch.empty(0))

    with pytest.raises(DynamicModelInspectionError, match="Zero-sized parameter"):
        _storage_groups(Empty())


@pytest.mark.parametrize(
    "storage_wrapper",
    (SimpleNamespace(), SimpleNamespace(_cdata=0)),
    ids=("missing-identity", "zero-identity"),
)
def test_storage_groups_ignore_invalid_subclass_storage_identity_override(
    storage_wrapper,
) -> None:
    class WrappedTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        def untyped_storage(self):
            return storage_wrapper

    class IndependentWrappedParameters(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Parameter(
                WrappedTensor(torch.arange(4, dtype=torch.float32)),
                requires_grad=False,
            )
            self.right = torch.nn.Parameter(
                WrappedTensor(torch.arange(4, dtype=torch.float32)),
                requires_grad=False,
            )

    groups = _storage_groups(IndependentWrappedParameters())
    assert groups["left"] != groups["right"]


def test_storage_groups_ignore_positive_subclass_storage_spoof() -> None:
    class WrappedTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        def untyped_storage(self):
            return SimpleNamespace(_cdata=8675309)

        def data_ptr(self):
            return 8675309

        def numel(self):
            return 999

        def element_size(self):
            return 999

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func in {torch.Tensor.numel, torch.Tensor.element_size}:
                return 999
            return super().__torch_function__(func, types, args, kwargs or {})

    class IndependentWrappedParameters(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Parameter(
                WrappedTensor(torch.arange(4, dtype=torch.float32)),
                requires_grad=False,
            )
            self.right = torch.nn.Parameter(
                WrappedTensor(torch.arange(4, dtype=torch.float32) + 10),
                requires_grad=False,
            )

    model = IndependentWrappedParameters()
    groups = _storage_groups(model)
    assert groups["left"] != groups["right"]
    assert _parameter_counts(model) == (8, 8, 32)


def test_storage_groups_detect_cross_storageimpl_numpy_aliases() -> None:
    class ExactExternalAlias(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backing = numpy.arange(8, dtype=numpy.float32)
            self.left = torch.nn.Parameter(torch.from_numpy(backing))
            self.right = torch.nn.Parameter(torch.from_numpy(backing))

    model = ExactExternalAlias()
    groups = _storage_groups(model)
    assert groups["left"] == groups["right"]
    assert _parameter_counts(model) == (16, 8, 32)


def test_storage_groups_reject_cross_storageimpl_numpy_partial_overlap() -> None:
    class PartialExternalAlias(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backing = numpy.arange(12, dtype=numpy.float32)
            self.left = torch.nn.Parameter(torch.from_numpy(backing[:8]))
            self.right = torch.nn.Parameter(torch.from_numpy(backing[4:]))

    with pytest.raises(DynamicModelInspectionError, match="Partially overlapping"):
        _storage_groups(PartialExternalAlias())


def test_storage_group_interval_sweep_accepts_many_disjoint_views() -> None:
    model = torch.nn.Module()
    backing = torch.arange(10_000, dtype=torch.float32)
    for index in reversed(range(10_000)):
        model.register_parameter(
            f"parameter_{index:04d}",
            torch.nn.Parameter(backing[index : index + 1]),
        )

    started = time.perf_counter()
    groups = _storage_groups(model)
    elapsed = time.perf_counter() - started

    assert tuple(groups) == tuple(sorted(groups))
    assert len(set(groups.values())) == 10_000
    assert elapsed < 5.0
