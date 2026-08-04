from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_forge.dynamic_circuit_ir import CircuitKind, EdgeKind
from neural_forge.dynamic_model_inspector import (
    SMOLVLM_FAMILY,
    SMOLVLM_MODEL_CLASS,
    DynamicModelInspectionError,
    SmolVlmCircuitInspector,
    default_inspector_registry,
    inspect_model_circuits,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

VISION_CONFIG = {
    "model_type": "smolvlm_vision",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "patch_size": 8,
    "image_size": 16,
    "num_channels": 3,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
}
TEXT_CONFIG = {
    "model_type": "llama",
    "hidden_size": 24,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 3,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "vocab_size": 40,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
}


def _tiny_smolvlm(**overrides):
    vision = {**VISION_CONFIG, **overrides.pop("vision_config", {})}
    text = {**TEXT_CONFIG, **overrides.pop("text_config", {})}
    config = transformers.SmolVLMConfig(
        vision_config=vision,
        text_config=text,
        scale_factor=2,
        image_token_id=39,
        vocab_size=text["vocab_size"],
        tie_word_embeddings=False,
        **overrides,
    )
    model = transformers.SmolVLMForConditionalGeneration(config)
    model.eval()
    return model


def _named_parameter_paths(model) -> set[str]:
    return {
        name
        for name, _value in model.named_parameters(recurse=True, remove_duplicate=False)
    }


def test_registry_selects_the_smolvlm_inspector_for_the_exact_class() -> None:
    model = _tiny_smolvlm()

    assert f"{type(model).__module__}.{type(model).__qualname__}" == SMOLVLM_MODEL_CLASS
    assert default_inspector_registry().select(model).name == "smolvlm-native-vision"


def test_smolvlm_inspection_owns_every_parameter_exactly_once() -> None:
    model = _tiny_smolvlm()
    graph = inspect_model_circuits(model, model_name="tiny-smolvlm")

    assert graph.family == SMOLVLM_FAMILY
    assert set(graph.parameter_paths) == _named_parameter_paths(model)
    assert len(graph.parameter_paths) == len(set(graph.parameter_paths))
    assert graph.logical_parameter_count == sum(
        int(value.numel())
        for _name, value in model.named_parameters(recurse=True, remove_duplicate=False)
    )
    assert graph.layer_indices == (0, 1)


def test_vision_tower_and_connector_are_modality_input_side_operators() -> None:
    model = _tiny_smolvlm()
    graph = inspect_model_circuits(model, model_name="tiny-smolvlm")

    vision_paths = {
        name
        for name in _named_parameter_paths(model)
        if name.startswith("model.vision_model.") or name.startswith("model.connector.")
    }
    owners = {
        path: node
        for node in graph.nodes
        for path in node.parameter_paths
        if path in vision_paths
    }
    assert set(owners) == vision_paths
    assert all(node.node_id.startswith("input.image") for node in owners.values())
    assert all(node.layer_index is None for node in owners.values())

    patch = graph.node("input.image.patch_embedding")
    assert patch.kind == CircuitKind.MODALITY_INPUT.value
    assert {port.modality for port in patch.inputs} == {"image"}
    projection = graph.node("input.image.modality_projection")
    assert projection.kind == CircuitKind.INTERFACE_PROJECTION.value
    assert projection.output("image_embeddings").dimensions[-1] == (
        int(model.config.text_config.hidden_size)
    )
    assert set(projection.parameter_paths) == {
        "model.connector.modality_projection.proj.weight"
    }


def test_image_and_text_interfaces_are_declared_and_video_is_recorded() -> None:
    model = _tiny_smolvlm()
    graph = inspect_model_circuits(model, model_name="tiny-smolvlm")

    interfaces = {value.interface_id: value for value in graph.interfaces}
    assert set(interfaces) == {
        "image.pixels.input",
        "image.patch_mask.input",
        "text.tokens.input",
        "text.logits.output",
    }
    assert interfaces["image.pixels.input"].modality == "image"
    assert interfaces["text.logits.output"].direction == "output"
    assert {value.modality for value in graph.interfaces} == {"image", "text"}

    metadata = dict(graph.metadata)
    assert metadata["inspector"] == "smolvlm-native-vision"
    assert metadata["decoder_inspector"] == "hf-causal-decoder"
    assert metadata["decoder_family"] == "llama"
    assert metadata["position_policy"] == "rope"
    assert metadata["image_token_replacement_fusion"] is True
    # Video is not a separate lane in this checkpoint; the honest emission says
    # so rather than inventing a video interface the weights do not have.
    assert "image frames" in metadata["video_lane"]


def test_fusion_joins_the_connector_and_the_token_embedding() -> None:
    model = _tiny_smolvlm()
    graph = inspect_model_circuits(model, model_name="tiny-smolvlm")

    fusion = graph.node("input.image_text_embedding_fusion")
    assert fusion.kind == CircuitKind.INTERFACE_PROJECTION.value
    assert fusion.parameter_paths == ()
    assert dict(fusion.attributes)["image_token_id"] == int(model.config.image_token_id)
    incoming = {
        (edge.source_node, edge.kind)
        for edge in graph.edges
        if edge.target_node == fusion.node_id
    }
    assert (
        "input.image.modality_projection",
        EdgeKind.MODALITY.value,
    ) in incoming
    assert (
        "decoder.input.token_embedding",
        EdgeKind.TOKEN.value,
    ) in incoming


def test_decoder_subgraph_uses_the_shared_causal_decoder_concepts() -> None:
    model = _tiny_smolvlm()
    graph = inspect_model_circuits(model, model_name="tiny-smolvlm")

    decoder_paths = {
        name
        for name in _named_parameter_paths(model)
        if name.startswith("model.text_model.") or name.startswith("lm_head.")
    }
    owners = {
        path: node
        for node in graph.nodes
        for path in node.parameter_paths
        if path in decoder_paths
    }
    assert set(owners) == decoder_paths
    assert all(node.node_id.startswith("decoder.") for node in owners.values())

    attention = graph.nodes_of_kind(CircuitKind.ATTENTION, layer_index=0)
    assert [node.node_id for node in attention] == ["decoder.layer.000.attention"]
    assert graph.nodes_of_kind(CircuitKind.GATED_MLP, layer_index=1)[0].node_id == (
        "decoder.layer.001.mlp"
    )
    assert any(
        edge.kind == EdgeKind.CACHE.value and edge.recurrent for edge in graph.edges
    )
    assert graph.node("decoder.output.lm_head").kind == CircuitKind.OUTPUT_HEAD.value


def test_supports_rejects_a_look_alike_config_without_the_exact_class() -> None:
    inspector = SmolVlmCircuitInspector()
    impostor = SimpleNamespace(config=SimpleNamespace(model_type="smolvlm"))

    assert inspector.supports(impostor) is False
    with pytest.raises(
        DynamicModelInspectionError, match="SmolVLM inspection requires"
    ):
        inspector.inspect(impostor, model_name="impostor")


def test_inspection_fails_closed_on_a_learned_vision_parameter_it_does_not_own() -> (
    None
):
    model = _tiny_smolvlm()
    model.model.vision_model.register_parameter(
        "unmodelled_gain", torch.nn.Parameter(torch.ones(4))
    )

    with pytest.raises(
        DynamicModelInspectionError, match="outside their proven front-end operators"
    ):
        inspect_model_circuits(model, model_name="leaky-vision")


def test_inspection_fails_closed_on_a_biased_modality_projection() -> None:
    model = _tiny_smolvlm()
    projection = model.model.connector.modality_projection.proj
    projection.bias = torch.nn.Parameter(torch.zeros(projection.out_features))

    with pytest.raises(DynamicModelInspectionError, match="must be a bias-free"):
        inspect_model_circuits(model, model_name="biased-connector")


def test_inspection_fails_closed_on_a_repatched_vision_stem() -> None:
    model = _tiny_smolvlm()
    stem = model.model.vision_model.embeddings.patch_embedding
    model.model.vision_model.embeddings.patch_embedding = torch.nn.Conv2d(
        stem.in_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=(1, 1),
    )

    with pytest.raises(
        DynamicModelInspectionError, match="declared raw-patch topology"
    ):
        inspect_model_circuits(model, model_name="restrided-vision")


def test_inspection_fails_closed_when_the_vision_depth_disagrees() -> None:
    model = _tiny_smolvlm()
    model.model.vision_model.encoder.layers = torch.nn.ModuleList(
        list(model.model.vision_model.encoder.layers)[:1]
    )

    with pytest.raises(
        DynamicModelInspectionError, match="vision depth differs from vision_config"
    ):
        inspect_model_circuits(model, model_name="short-vision")


def test_inspection_fails_closed_on_an_unsupported_text_decoder() -> None:
    model = _tiny_smolvlm()
    model.config.text_config.model_type = "mamba"

    with pytest.raises(
        DynamicModelInspectionError, match="requires a llama text decoder"
    ):
        inspect_model_circuits(model, model_name="wrong-decoder")


def test_inspection_fails_closed_when_the_decoder_layout_is_broken() -> None:
    model = _tiny_smolvlm()
    model.model.text_model.layers[0].mlp.gate_proj = torch.nn.Linear(
        int(model.config.text_config.hidden_size), 7, bias=False
    )

    with pytest.raises(
        DynamicModelInspectionError,
        match="failed the shared causal-decoder contract",
    ):
        inspect_model_circuits(model, model_name="broken-decoder")
