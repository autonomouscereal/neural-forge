"""Inspector tests for the chimera-omni encoder-free omnimodal trunk.

These live in their own file rather than in ``test_dynamic_model_inspector.py``
so concurrent work on the shared inspector suite cannot collide with them, the
same separation ``NOTES-W9B.md`` used for the bare tied-body adapter.

Every model here is a **real** ``chimera_omni.modeling.ChimeraOmniModel`` built
through chimera-omni's own configuration API, imported from the target
repository's checkout.  Nothing about that repository is vendored, copied, or
re-implemented; when the checkout is unavailable the suite skips and says so.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    EdgeKind,
    ModelCircuitGraph,
    emitted_media_interfaces,
)
from neural_forge.dynamic_model_inspector import (
    CHIMERA_OMNI_BLOCK_KINDS,
    CHIMERA_OMNI_FAMILY,
    ChimeraOmniCircuitInspector,
    ChimeraOmniInspectionError,
    default_inspector_registry,
    inspect_model_circuits,
)


torch = pytest.importorskip("torch")

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_chimera_omni_toy_fold.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

try:
    _REPO = runner.resolve_chimera_omni_repo()
    _CHIMERA = runner.import_chimera_omni(_REPO)
except runner.ChimeraOmniFoldError as exc:  # pragma: no cover - environment gate
    pytest.skip(f"chimera-omni checkout unavailable: {exc}", allow_module_level=True)

chimera_config = _CHIMERA["config"]
chimera_modeling = _CHIMERA["modeling"]


def _coverable(hidden_size: int = 128):
    return chimera_modeling.ChimeraOmniModel(
        runner.coverable_config(chimera_config, hidden_size=hidden_size)
    )


def _owns_every_tensor_once(graph: ModelCircuitGraph, model) -> None:
    live = {
        name
        for name, _value in model.named_parameters(
            recurse=True, remove_duplicate=False
        )
    }
    paths = list(graph.parameter_paths)
    assert set(paths) == live
    assert len(paths) == len(set(paths))
    assert graph.logical_parameter_count == sum(
        int(value.numel())
        for _name, value in model.named_parameters(
            recurse=True, remove_duplicate=False
        )
    )
    assert graph.unique_parameter_count == sum(
        int(value.numel()) for value in model.parameters()
    )


def test_registry_routes_a_chimera_model_to_the_chimera_inspector() -> None:
    model = _coverable()
    selected = default_inspector_registry().select(model)

    assert selected.name == ChimeraOmniCircuitInspector.name


def test_coverable_model_owns_every_tensor_exactly_once() -> None:
    model = _coverable()
    graph = inspect_model_circuits(model, model_name="chimera-coverable")

    assert graph.family == CHIMERA_OMNI_FAMILY
    assert graph.model_class == "chimera_omni.modeling.chimera.ChimeraOmniModel"
    _owns_every_tensor_once(graph, model)


def test_every_declared_block_family_compiles() -> None:
    """kda/local/csa/hca and gated_mla + Stable LatentMoE all decompose."""

    for factory in (
        chimera_config.small_test_config,
        chimera_config.k3_aligned_test_config,
    ):
        model = chimera_modeling.ChimeraOmniModel(factory())
        graph = inspect_model_circuits(model, model_name="chimera-families")
        _owns_every_tensor_once(graph, model)

    families = set()
    for factory in (
        chimera_config.small_test_config,
        chimera_config.k3_aligned_test_config,
    ):
        families.update(
            dict(
                inspect_model_circuits(
                    chimera_modeling.ChimeraOmniModel(factory()),
                    model_name="chimera-families",
                ).metadata
            )["block_kinds"].split(",")
        )
    assert families == set(CHIMERA_OMNI_BLOCK_KINDS)


def test_kda_is_a_recurrent_mixer_with_an_explicit_state_edge() -> None:
    model = chimera_modeling.ChimeraOmniModel(chimera_config.small_test_config())
    graph = inspect_model_circuits(model, model_name="chimera-kda")

    mixers = graph.nodes_of_kind(CircuitKind.RECURRENT_MIXER)
    assert [value.node_id for value in mixers] == ["layer.000.operator"]
    attributes = dict(mixers[0].attributes)
    assert attributes["operator_type"] == "kimi_delta_attention"
    assert attributes["softmax_free"] is True
    assert attributes["internal_short_conv_operators"] == 3
    assert attributes["fold_kernel_family_required"] == (
        "bounded_gated_delta_recurrence"
    )
    assert any(
        edge.kind == EdgeKind.STATE.value
        and edge.recurrent
        and edge.source_node == mixers[0].node_id
        and edge.target_node == mixers[0].node_id
        for edge in graph.edges
    )
    # Every KDA tensor, including the three internal depthwise short convs and
    # the headwise output norm, is owned by that one node.
    assert set(mixers[0].parameter_paths) == {
        f"blocks.0.mixer.{local}"
        for local, _value in model.blocks[0].mixer.named_parameters(
            recurse=True, remove_duplicate=False
        )
    }
    assert "blocks.0.mixer.q_conv.conv.weight" in mixers[0].parameter_paths
    assert "blocks.0.mixer.output_norm.weight" in mixers[0].parameter_paths


def test_compressed_summary_blocks_split_into_two_attention_circuits() -> None:
    model = chimera_modeling.ChimeraOmniModel(chimera_config.small_test_config())
    graph = inspect_model_circuits(model, model_name="chimera-csa")

    window = graph.node("layer.002.operator.local_window")
    summary = graph.node("layer.002.operator.compressed_summary")
    combine = graph.node("layer.002.operator.combine")
    assert window.kind == CircuitKind.ATTENTION.value
    assert summary.kind == CircuitKind.ATTENTION.value
    assert combine.kind == CircuitKind.RESIDUAL_ADD.value
    assert set(window.parameter_paths) == {
        f"blocks.2.mixer.local.{name}.weight"
        for name in ("q_proj", "k_proj", "v_proj", "out")
    }
    assert set(summary.parameter_paths) == {
        f"blocks.2.mixer.{name}.weight"
        for name in ("summary_q", "summary_k", "summary_v", "out")
    }
    assert dict(summary.attributes)["compress_stride"] == 64
    assert dict(graph.node("layer.003.operator.compressed_summary").attributes)[
        "compress_stride"
    ] == 1024
    assert dict(summary.attributes)["compound_parent"] == "csa"
    assert dict(
        graph.node("layer.003.operator.compressed_summary").attributes
    )["compound_parent"] == "hca"


def test_boundary_projection_is_one_shared_multi_modality_input() -> None:
    model = _coverable()
    graph = inspect_model_circuits(model, model_name="chimera-boundary")

    boundary = graph.node("input.native_media.boundary_projection")
    assert boundary.kind == CircuitKind.MODALITY_INPUT.value
    assert set(boundary.parameter_paths) == {
        "native_media_bridge.feature_proj.weight"
    }
    assert [value.name for value in boundary.inputs] == [
        "image_features",
        "video_features",
        "audio_features",
        "music_features",
        "tts_features",
        "ocr_features",
    ]
    attributes = dict(boundary.attributes)
    assert attributes["modality_tower_count"] == 0
    assert attributes["shared_projection_for_every_modality"] is True
    assert attributes["external_frozen_encoder"] is False
    inputs = [
        value for value in graph.interfaces if value.direction == "input"
    ]
    assert {value.interface_id for value in inputs} == {
        "text.tokens.input",
        "native_media.image.input",
        "native_media.video.input",
        "native_media.audio.input",
        "native_media.music.input",
        "native_media.tts.input",
        "native_media.ocr.input",
    }
    # music and tts are waveform lanes, so they enter as canonical audio.
    lanes = {value.interface_id: value.modality for value in inputs}
    assert lanes["native_media.music.input"] == "audio"
    assert lanes["native_media.tts.input"] == "audio"
    assert lanes["native_media.ocr.input"] == "ocr"


def test_emission_head_is_a_modality_output_sibling_of_the_text_head() -> None:
    model = _coverable()
    graph = inspect_model_circuits(model, model_name="chimera-emission")

    emitters = graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT)
    assert [value.node_id for value in emitters] == [
        "output.native_media.emission_head"
    ]
    assert set(emitters[0].parameter_paths) == {"native_media_head.out.weight"}
    assert dict(emitters[0].attributes)[
        "shared_emission_head_for_every_modality"
    ] is True
    emitted = emitted_media_interfaces(graph)
    assert {value.interface_id for value in emitted} == {
        f"native_media.{lane}.output"
        for lane in ("image", "video", "audio", "music", "tts", "ocr")
    }
    assert all(value.direction == "output" for value in emitted)
    # The text head and the media head both read the trunk's final norm.
    readers = {
        edge.target_node
        for edge in graph.edges
        if edge.source_node == "output.final_norm"
    }
    assert {"output.lm_head", "output.native_media.gather"} <= readers


def test_block_attnres_depth_sources_are_explicit_residual_edges() -> None:
    model = _coverable()
    graph = inspect_model_circuits(model, model_name="chimera-depth")
    metadata = dict(graph.metadata)

    assert metadata["depth_axis_block_attnres"] is True
    assert metadata["depth_checkpoint_layers"] == "0,2"
    selector = graph.node("layer.003.attention_depth_select")
    assert selector.kind == CircuitKind.RESIDUAL_ADD.value
    assert set(selector.parameter_paths) == {
        "blocks.3.attention_residual.query",
        "blocks.3.attention_residual.norm.weight",
    }
    attributes = dict(selector.attributes)
    assert attributes["depth_source_count"] == 2
    assert attributes["mixed_source_count"] == 3
    assert attributes["invoked_at_this_depth"] is True
    sources = {
        edge.source_node
        for edge in graph.edges
        if edge.target_node == selector.node_id
        and edge.target_port.startswith("depth_source_")
    }
    assert sources == {
        "input.native_media_text_fusion",
        "layer.001.feed_forward_residual",
    }
    assert all(
        edge.kind == EdgeKind.RESIDUAL.value
        for edge in graph.edges
        if edge.target_node == selector.node_id
    )
    # The first block's attention selector is never invoked: no block has
    # completed at that depth.
    first = dict(graph.node("layer.000.attention_depth_select").attributes)
    assert first["depth_source_count"] == 0
    assert first["invoked_at_this_depth"] is False
    assert first["identity_passthrough"] is True


def test_checkpoint_layers_restart_the_residual_accumulator() -> None:
    model = _coverable()
    graph = inspect_model_circuits(model, model_name="chimera-reset")

    reset = graph.node("layer.002.operator_residual")
    assert dict(reset.attributes)["residual_accumulator_reset"] is True
    assert [value.name for value in reset.inputs] == ["delta"]
    ordinary = graph.node("layer.001.operator_residual")
    assert dict(ordinary.attributes)["residual_accumulator_reset"] is False
    assert [value.name for value in ordinary.inputs] == ["base", "delta"]


def test_residual_mode_none_has_no_depth_selection() -> None:
    config = chimera_config.ChimeraConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_sequence_length=32,
        block_pattern=("local",),
        residual_mode="none",
        native_feature_dim=32,
        mtp_heads=0,
        aux_heads=(),
        local_window=8,
    )
    model = chimera_modeling.ChimeraOmniModel(config)
    graph = inspect_model_circuits(model, model_name="chimera-plain")

    _owns_every_tensor_once(graph, model)
    assert dict(graph.metadata)["depth_axis_block_attnres"] is False
    assert not [
        node
        for node in graph.nodes_of_kind(CircuitKind.RESIDUAL_ADD)
        if dict(node.attributes).get("residual_mode") == "block_attnres"
    ]
    assert dict(graph.metadata)["depth_checkpoint_layers"] == ""
    # Both ordinary residual writes survive: pre-norm attention then pre-norm
    # feed forward, each adding into the running stream.
    writes = [
        node.node_id
        for node in graph.nodes_of_kind(CircuitKind.RESIDUAL_ADD)
        if node.layer_index == 0
    ]
    assert writes == ["layer.000.operator_residual", "layer.000.feed_forward_residual"]
    for node_id in writes:
        assert [value.name for value in graph.node(node_id).inputs] == [
            "base",
            "delta",
        ]


def test_graph_names_the_kernel_gap_families() -> None:
    coverable = inspect_model_circuits(_coverable(), model_name="chimera-coverable")
    mixed = inspect_model_circuits(
        chimera_modeling.ChimeraOmniModel(chimera_config.small_test_config()),
        model_name="chimera-mixed",
    )

    assert dict(coverable.metadata)["kernel_gap_block_kinds"] == ""
    assert dict(coverable.metadata)["structurally_coverable_block_kinds"] == "local"
    assert dict(mixed.metadata)["kernel_gap_block_kinds"] == "csa,hca,kda"


def test_an_unknown_sequence_mixer_fails_closed_naming_the_family() -> None:
    model = _coverable()
    model.blocks[0].kind = "mamba"

    with pytest.raises(ChimeraOmniInspectionError) as error:
        inspect_model_circuits(model, model_name="chimera-unknown")
    message = str(error.value)
    assert "'mamba'" in message
    assert "no inspected decomposition" in message
    for family in CHIMERA_OMNI_BLOCK_KINDS:
        assert family in message


def test_a_block_family_that_owns_unpredicted_tensors_fails_closed() -> None:
    model = _coverable()
    model.blocks[1].mixer.register_parameter(
        "smuggled", torch.nn.Parameter(torch.zeros(4))
    )

    with pytest.raises(ChimeraOmniInspectionError) as error:
        inspect_model_circuits(model, model_name="chimera-smuggled")
    message = str(error.value)
    assert "'local'" in message
    assert "smuggled" in message


def test_a_relabelled_block_disagreeing_with_the_pattern_fails_closed() -> None:
    model = _coverable()
    model.blocks[2].kind = "kda"

    with pytest.raises(ChimeraOmniInspectionError) as error:
        inspect_model_circuits(model, model_name="chimera-relabelled")
    assert "runs family 'kda'" in str(error.value)
    assert "declares 'local'" in str(error.value)


def test_a_type_table_too_small_for_the_media_lanes_fails_closed() -> None:
    config = chimera_config.ChimeraConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_sequence_length=16,
        block_pattern=("local",),
        native_feature_dim=16,
        native_type_count=4,
        mtp_heads=0,
        aux_heads=(),
    )
    model = chimera_modeling.ChimeraOmniModel(config)

    with pytest.raises(ChimeraOmniInspectionError) as error:
        inspect_model_circuits(model, model_name="chimera-small-type-table")
    assert "native_type_count 4" in str(error.value)


def test_a_model_without_a_chimera_manifest_is_not_supported() -> None:
    inspector = ChimeraOmniCircuitInspector()

    assert inspector.supports(object()) is False


def test_the_coverable_topology_is_exactly_four_times_by_construction() -> None:
    parent = _coverable(runner.PARENT_HIDDEN_SIZE)
    child = _coverable(runner.CHILD_HIDDEN_SIZE)

    parent_graph = inspect_model_circuits(parent, model_name="chimera-parent")
    child_graph = inspect_model_circuits(child, model_name="chimera-child")

    assert parent_graph.unique_parameter_count == 1_104_128
    assert child_graph.unique_parameter_count == 276_032
    assert parent_graph.unique_parameter_count == (
        4 * child_graph.unique_parameter_count
    )
    rows = runner.parameter_map(parent, child)
    assert rows and all(row["exactly_four_times"] for row in rows)
