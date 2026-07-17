from __future__ import annotations

import copy

import pytest

from neural_forge.dynamic_circuit_ir import CircuitKind, EdgeKind
from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.hf_qwen_codebook_model import (
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tuple(tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: token_id for token_id, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _tiny_qwen3_codebook() -> torch.nn.Module:
    torch.manual_seed(2601)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    parent = _Tokenizer([f"token-{index}" for index in range(12)])
    target = _Tokenizer(
        [
            "token-0",
            "token-1",
            "token-2",
            "token-5",
            "token-7",
            "token-9",
            "token-11",
        ]
    )
    return prepare_qwen3_codebook_model(
        model,
        parent,
        target,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=2,
    )


@pytest.fixture(scope="module")
def runtime() -> torch.nn.Module:
    return _tiny_qwen3_codebook()


def test_qwen3_codebook_inspector_owns_shared_storage_and_full_depth(
    runtime: torch.nn.Module,
) -> None:
    registry = default_inspector_registry()
    assert registry.select(runtime).name == "qwen3-codebook"

    graph = inspect_model_circuits(
        runtime,
        model_name="tiny-qwen3-codebook",
        tokenizer_fingerprint="7" * 64,
    )
    report = qwen3_codebook_parameter_report(runtime)
    metadata = dict(graph.metadata)

    assert graph.family == "qwen3_codebook"
    assert graph.layer_indices == (0, 1)
    assert metadata["inspector"] == "qwen3-codebook"
    assert metadata["full_logical_depth"] is True
    assert metadata["dense_logical_parameter_count"] == (report.logical_parameter_count)
    assert metadata["registered_logical_parameter_count"] == (
        report.registered_logical_parameter_count
    )
    assert metadata["index_buffer_bytes"] == report.index_buffer_bytes
    assert graph.logical_parameter_count == (report.registered_logical_parameter_count)
    assert graph.unique_parameter_count == report.unique_parameter_count
    assert graph.unique_parameter_bytes == sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in runtime.parameters()
    )
    assert set(graph.parameter_paths) == {
        name
        for name, _parameter in runtime.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }

    attention_owner = graph.node("storage.attention_codebook")
    neuron_owner = graph.node("storage.neuron_codebook")
    assert attention_owner.kind == "attention_codebook_registry"
    assert neuron_owner.kind == "neuron_codebook_registry"
    assert set(attention_owner.parameter_paths) == {
        f"attention_codebook.registry.{name}"
        for name, _parameter in (
            runtime.attention_codebook.registry.named_parameters(remove_duplicate=False)
        )
    }
    assert set(neuron_owner.parameter_paths) == {
        f"neuron_codebook.registry.{name}"
        for name, _parameter in (
            runtime.neuron_codebook.registry.named_parameters(remove_duplicate=False)
        )
    }
    assert dict(attention_owner.attributes)["coupled_roles"] == "q_k_v_o"
    assert dict(neuron_owner.attributes)["coupled_roles"] == (
        "gate_up_post_swiglu_down"
    )

    attention_nodes = graph.nodes_of_kind(CircuitKind.ATTENTION)
    mlp_nodes = graph.nodes_of_kind(CircuitKind.GATED_MLP)
    assert len(attention_nodes) == len(mlp_nodes) == 2
    assert all(
        dict(node.attributes)["projection_storage_owner"] == attention_owner.node_id
        for node in attention_nodes
    )
    assert all(
        {parameter.path.rsplit(".", 2)[-2] for parameter in node.parameters}
        == {"q_norm", "k_norm"}
        for node in attention_nodes
    )
    assert all(not node.parameters for node in mlp_nodes)
    assert all(
        dict(node.attributes)["projection_storage_owner"] == neuron_owner.node_id
        for node in mlp_nodes
    )

    recurrent_cache_edges = [
        edge
        for edge in graph.edges
        if edge.kind == EdgeKind.CACHE.value and edge.recurrent
    ]
    assert len(recurrent_cache_edges) == 2
    assert all(
        edge.source_node == edge.target_node
        and edge.source_port == "cache_out"
        and edge.target_port == "cache_in"
        for edge in recurrent_cache_edges
    )
    assert (
        len(
            [
                edge
                for edge in graph.edges
                if edge.source_node == attention_owner.node_id
                and edge.kind == EdgeKind.CONTROL.value
            ]
        )
        == 2
    )
    assert (
        len(
            [
                edge
                for edge in graph.edges
                if edge.source_node == neuron_owner.node_id
                and edge.kind == EdgeKind.CONTROL.value
            ]
        )
        == 2
    )
    assert (
        len([edge for edge in graph.edges if edge.kind == EdgeKind.RESIDUAL.value]) == 8
    )

    embedding_parameter = graph.node("input.token_embedding").parameters[0]
    head_parameter = graph.node("output.lm_head").parameters[0]
    assert embedding_parameter.path == "model.embed_tokens.weight"
    assert head_parameter.path == "lm_head.weight"
    assert embedding_parameter.storage_group == head_parameter.storage_group
    assert {
        (
            interface.interface_id,
            interface.direction,
            interface.representation,
        )
        for interface in graph.interfaces
    } == {
        ("text.tokens.input", "input", "token_ids"),
        ("text.logits.output", "output", "token_logits"),
    }


def test_qwen3_codebook_inspector_fails_closed_on_partial_projection_install(
    runtime: torch.nn.Module,
) -> None:
    malformed = copy.deepcopy(runtime)
    malformed.model.layers[0].self_attn.q_proj = torch.nn.Linear(
        8,
        8,
        bias=False,
    )

    with pytest.raises(
        DynamicModelInspectionError,
        match="parameterless codebook view",
    ):
        inspect_model_circuits(
            malformed,
            model_name="malformed-qwen3-codebook",
        )
