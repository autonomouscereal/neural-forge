"""End-to-end proof that the native points3d lane is executable at toy scale.

The architecture below is the points3d mirror of ``_TinyNativeVideoChatModel``
in ``tests/test_native_video_executable_lane.py``: a *learned* per-point linear
plus a symmetric max pool feeds point context tokens into a tiny causal decoder
alongside text tokens.  Nothing here loads or freezes an external encoder; every
geometry tensor is a parameter this repository owns, which is what the project's
native-modality rule requires.

**Honest limit.** This proves the *compiler contract*, not geometric
understanding.  The toy decoder cannot read a point cloud: the closed-set answer
in ``_behavior_runner`` is gated on a parent-derived marker weight surviving the
fold exactly, precisely as the validated NEO vision, native audio, and native
video proofs do.  What is actually proved end to end is that a real decoded
quantized point set drives a real forward pass through the whole model, that
inspection, target synthesis, planning, evidence sealing, folding,
parent-relative gating, publication, and a standalone fresh reload all agree on
one authenticated artifact, and that the published child differs from its parent
by exactly the intended parameters.

The emitted-output half of this lane -- the first ``MODALITY_OUTPUT`` node in
the repository -- is proved separately in
``tests/test_native_points3d_output_lane.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    EdgeKind,
    canonical_sha256,
    emitted_media_interfaces,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_fold_registry import MethodStatus, default_method_registry
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import (
    NATIVE_POINTS3D_LANE_MARKER,
    DynamicModelInspectionError,
    NativePoints3dCircuitInspector,
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_points3d_adapter import (
    DynamicPoints3dAdapterError,
    NativePoints3dTargetSynthesisAdapter,
)
from neural_forge.dynamic_points3d_fold import (
    FOLD_CACHE_SCHEMA,
    METHOD_ID,
    DynamicPoints3dFoldError,
    NativePoints3dBehaviorGate,
    NativePoints3dFoldCache,
    NativePoints3dFoldStackAdapter,
    NativePoints3dStackResources,
    Points3dTraceStatisticsCache,
    build_native_points3d_fold_trace_cache,
    derive_native_points3d_fold_cache,
    load_points3d_parent_behavior_cache,
    points3d_lane_status,
    write_native_points3d_dynamic_fold_receipt,
    write_points3d_parent_output_cache,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesizer,
)
from neural_forge.native_points3d_probe import (
    COMPONENTS,
    POINT_COUNT,
    build_native_points3d_probe,
    read_points3d_cloud,
)


torch = pytest.importorskip("torch")


EXAMPLES_PER_CATEGORY = 4
HIDDEN = 8
INTERMEDIATE = 6
LAYERS = 2
VOCAB = 12
QUERY_HEADS = 4
KEY_VALUE_HEADS = 2
HEAD_DIM = 2
MAX_POSITIONS = 32
POINTS_HIDDEN = 4
POINT_TOKENS = 8
PROMPT_TOKENS = 6
MARKER_VALUE = 0.375


# --------------------------------------------------------------------------
# A tiny native points3d-text architecture
# --------------------------------------------------------------------------


class _PointFrontEndConfig:
    def __init__(
        self,
        *,
        num_points: int = POINT_COUNT,
        num_coordinates: int = COMPONENTS,
        num_tokens: int = POINT_TOKENS,
        hidden_size: int = POINTS_HIDDEN,
        pooling: str = "max_over_points",
        emit_centroid: bool = False,
    ) -> None:
        self.num_points = num_points
        self.num_coordinates = num_coordinates
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.emit_centroid = emit_centroid
        self.point_encoding = "int16_quantized_xyz"

    def to_dict(self) -> dict[str, object]:
        return {
            "num_points": self.num_points,
            "num_coordinates": self.num_coordinates,
            "num_tokens": self.num_tokens,
            "hidden_size": self.hidden_size,
            "pooling": self.pooling,
            "emit_centroid": self.emit_centroid,
            "point_encoding": self.point_encoding,
        }


class _Points3dDecoderConfig:
    model_type = "native_points3d_text_decoder"

    def __init__(self, *, intermediate_size: int = INTERMEDIATE) -> None:
        self.vocab_size = VOCAB
        self.hidden_size = HIDDEN
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = LAYERS
        self.num_attention_heads = QUERY_HEADS
        self.num_key_value_heads = KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM
        self.max_position_embeddings = MAX_POSITIONS
        self.hidden_act = "silu"
        self.attention_bias = False
        self.rms_norm_eps = 1.0e-6
        self.tie_word_embeddings = False

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "attention_bias": self.attention_bias,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class _NativePoints3dConfig:
    model_type = "native_points3d_chat"
    neural_forge_native_points3d_lane = NATIVE_POINTS3D_LANE_MARKER

    def __init__(
        self,
        points_config: _PointFrontEndConfig,
        decoder_config: _Points3dDecoderConfig,
    ) -> None:
        self.points_config = points_config
        self.decoder_config = decoder_config
        self.hidden_size = decoder_config.hidden_size
        self.tie_word_embeddings = decoder_config.tie_word_embeddings

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "neural_forge_native_points3d_lane": (
                self.neural_forge_native_points3d_lane
            ),
            "points_config": self.points_config.to_dict(),
            "decoder_config": self.decoder_config.to_dict(),
            "hidden_size": self.hidden_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class _RMSNorm(torch.nn.Module):
    def __init__(self, width: int, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, hidden_states: Any) -> Any:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.epsilon)


class _PointFrontEnd(torch.nn.Module):
    """Learned per-point linear plus a symmetric max pool into K tokens.

    Permutation invariance is structural, not asserted: the only reduction over
    the point axis is ``amax``, so shuffling the stored rows of a cloud cannot
    change a single output value.
    """

    def __init__(self, config: _NativePoints3dConfig) -> None:
        super().__init__()
        points = config.points_config
        self.num_tokens = points.num_tokens
        self.hidden_size = points.hidden_size
        self.point_linear = torch.nn.Linear(
            points.num_coordinates,
            points.num_tokens * points.hidden_size,
        )
        self.projection = torch.nn.Linear(
            points.hidden_size,
            config.decoder_config.hidden_size,
            bias=False,
        )
        self.activation = torch.nn.GELU()

    def forward(self, cloud: Any) -> Any:
        batch, points, _coordinates = cloud.shape
        features = self.activation(self.point_linear(cloud))
        grouped = features.view(batch, points, self.num_tokens, self.hidden_size)
        pooled = grouped.amax(dim=1)
        return self.projection(pooled)


class _Attention(torch.nn.Module):
    def __init__(self, config: _Points3dDecoderConfig) -> None:
        super().__init__()
        query_width = config.num_attention_heads * config.head_dim
        key_value_width = config.num_key_value_heads * config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = torch.nn.Linear(config.hidden_size, query_width, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, key_value_width, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, key_value_width, bias=False)
        self.o_proj = torch.nn.Linear(query_width, config.hidden_size, bias=False)

    def forward(self, hidden_states: Any) -> Any:
        batch, sequence, _width = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch, sequence, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch, sequence, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch, sequence, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        groups = self.num_attention_heads // self.num_key_value_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        context = context.transpose(1, 2).reshape(
            batch,
            sequence,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(context)


class _MLP(torch.nn.Module):
    def __init__(self, config: _Points3dDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = torch.nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = torch.nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Any) -> Any:
        gated = torch.nn.functional.silu(
            self.gate_proj(hidden_states)
        ) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class _DecoderLayer(torch.nn.Module):
    def __init__(self, config: _Points3dDecoderConfig) -> None:
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = _MLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(self, hidden_states: Any) -> Any:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states)
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class _DecoderCore(torch.nn.Module):
    def __init__(self, config: _Points3dDecoderConfig) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = torch.nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.layers = torch.nn.ModuleList(
            _DecoderLayer(config) for _index in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)


class _Points3dTextLanguageModel(torch.nn.Module):
    def __init__(self, config: _Points3dDecoderConfig) -> None:
        super().__init__()
        self.model = _DecoderCore(config)
        self.lm_head = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def normalized(self, hidden_states: Any) -> Any:
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return self.model.norm(hidden_states)

    def forward(self, hidden_states: Any) -> Any:
        return self.lm_head(self.normalized(hidden_states))


class _TinyNativePoints3dChatModel(torch.nn.Module):
    constructor_calls = 0

    def __init__(self, config: _NativePoints3dConfig) -> None:
        super().__init__()
        type(self).constructor_calls += 1
        self.config = config
        self.point_encoder = _PointFrontEnd(config)
        self.language_model = _Points3dTextLanguageModel(config.decoder_config)
        if config.points_config.emit_centroid:
            self.centroid_head = torch.nn.Linear(
                config.decoder_config.hidden_size,
                config.points_config.num_coordinates,
                bias=False,
            )

    def _residual(self, cloud: Any, token_ids: Any) -> Any:
        point_tokens = self.point_encoder(cloud)
        text_tokens = self.language_model.model.embed_tokens(token_ids)
        hidden_states = torch.cat((point_tokens, text_tokens), dim=1)
        positions = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        )
        return hidden_states + self.language_model.model.position_embedding(positions)

    def forward(self, cloud: Any, token_ids: Any) -> Any:
        return self.language_model(self._residual(cloud, token_ids))

    def emit_centroid(self, cloud: Any, token_ids: Any) -> Any:
        """Run the whole model and read the emitted-media head, not the logits."""

        head = getattr(self, "centroid_head", None)
        if head is None:
            raise AttributeError("This points3d model declares no centroid head.")
        normalized = self.language_model.normalized(self._residual(cloud, token_ids))
        return head(normalized[:, -1, :])


def _config(
    *,
    intermediate_size: int = INTERMEDIATE,
    emit_centroid: bool = False,
) -> _NativePoints3dConfig:
    return _NativePoints3dConfig(
        _PointFrontEndConfig(emit_centroid=emit_centroid),
        _Points3dDecoderConfig(intermediate_size=intermediate_size),
    )


def _model(
    seed: int,
    *,
    intermediate_size: int = INTERMEDIATE,
    emit_centroid: bool = False,
) -> _TinyNativePoints3dChatModel:
    torch.manual_seed(seed)
    return _TinyNativePoints3dChatModel(
        _config(intermediate_size=intermediate_size, emit_centroid=emit_centroid)
    ).eval()


def _tokenizer_fingerprint() -> str:
    return canonical_sha256({"tokenizer": "tiny-native-points3d"})


def _cloud(points_path: str) -> Any:
    points, quantization = read_points3d_cloud(points_path)
    metres = quantization.dequantize(points).astype(np.float32)
    # (points, 3) -> (batch, points, 3)
    return torch.from_numpy(np.ascontiguousarray(metres[None, ...]))


def _prompt_tokens(category: str) -> Any:
    padded = (category + "______")[:PROMPT_TOKENS]
    return torch.tensor(
        [[1 + (ord(character) % (VOCAB - 2)) for character in padded]],
        dtype=torch.long,
    )


# --------------------------------------------------------------------------
# Inspector
# --------------------------------------------------------------------------


def test_native_points3d_inspector_owns_point_encoder_and_decoder_exactly() -> None:
    model = _model(1)
    tokenizer = _tokenizer_fingerprint()

    graph = inspect_model_circuits(
        model,
        model_name="tiny-native-points3d",
        tokenizer_fingerprint=tokenizer,
    )

    assert default_inspector_registry().select(model).name == "native-points3d"
    assert graph.family == "native_points3d_chat"
    assert graph.layer_indices == tuple(range(LAYERS))
    assert set(graph.parameter_paths) == {
        name
        for name, _parameter in model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }
    assert graph.logical_parameter_count == sum(
        int(parameter.numel())
        for _name, parameter in model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    assert graph.unique_parameter_count == sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    assert graph.unique_parameter_bytes == graph.unique_parameter_count * 4
    assert {interface.modality for interface in graph.interfaces} == {
        "points3d",
        "text",
    }
    assert {interface.interface_id for interface in graph.interfaces} == {
        "points3d.quantized_xyz.input",
        "text.logits.output",
        "text.tokens.input",
    }
    points_interface = next(
        interface
        for interface in graph.interfaces
        if interface.modality == "points3d"
    )
    assert points_interface.direction == "input"
    assert points_interface.representation == "quantized_xyz"
    # Without the optional head this graph emits no media at all.
    assert emitted_media_interfaces(graph) == ()
    assert graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT) == ()

    modality_inputs = graph.nodes_of_kind(CircuitKind.MODALITY_INPUT)
    assert len(modality_inputs) == 1
    assert modality_inputs[0].node_id == "input.points3d.quantized_point_encoder"
    assert points_interface.node_id == modality_inputs[0].node_id
    assert modality_inputs[0].input("cloud").modality == "points3d"
    assert set(modality_inputs[0].parameter_paths) == {
        "point_encoder.point_linear.weight",
        "point_encoder.point_linear.bias",
    }
    assert len(graph.nodes_of_kind(CircuitKind.GATED_MLP)) == LAYERS
    assert any(edge.kind == EdgeKind.MODALITY.value for edge in graph.edges)

    projection = graph.node("input.points3d.dense_projection")
    assert projection.parameter_paths == ("point_encoder.projection.weight",)

    metadata = dict(graph.metadata)
    expected_points_parameters = (
        COMPONENTS * POINT_TOKENS * POINTS_HIDDEN
        + POINT_TOKENS * POINTS_HIDDEN
        + HIDDEN * POINTS_HIDDEN
    )
    assert metadata["points_parameter_count"] == expected_points_parameters
    assert metadata["decoder_intermediate_size"] == INTERMEDIATE
    assert metadata["native_quantized_point_set_encoder"] is True
    assert metadata["external_frozen_points3d_encoder"] is False
    assert metadata["points3d_token_count"] == POINT_TOKENS
    assert metadata["points3d_point_count"] == POINT_COUNT
    assert metadata["points3d_pooling"] == "max_over_points"
    assert metadata["emits_quantized_centroid"] is False


def test_native_points3d_front_end_is_permutation_invariant() -> None:
    """The declared symmetric pooling is real, not a documented intention."""

    model = _model(31)
    cloud = torch.randn(1, POINT_COUNT, COMPONENTS)
    shuffled = cloud[:, torch.randperm(POINT_COUNT), :]
    with torch.no_grad():
        assert torch.allclose(
            model.point_encoder(cloud),
            model.point_encoder(shuffled),
            atol=0.0,
            rtol=0.0,
        )


def test_native_points3d_inspector_never_claims_an_unmarked_model() -> None:
    model = _model(2)
    inspector = NativePoints3dCircuitInspector()
    assert inspector.supports(model)

    model.config.neural_forge_native_points3d_lane = "some-other-lane"
    assert not inspector.supports(model)
    with pytest.raises(
        DynamicModelInspectionError,
        match="marked native_points3d_chat",
    ):
        inspector.inspect(model, model_name="unmarked")


def test_native_points3d_inspector_fails_on_unowned_learned_operator() -> None:
    model = _model(3)
    model.point_encoder.extra_projection = torch.nn.Linear(
        POINTS_HIDDEN,
        POINTS_HIDDEN,
    )

    with pytest.raises(
        DynamicModelInspectionError,
        match="outside its per-point encoder",
    ):
        NativePoints3dCircuitInspector().inspect(
            model,
            model_name="malformed-points3d",
        )


def test_native_points3d_inspector_rejects_a_precomputed_feature_front_end() -> None:
    model = _model(4)
    model.config.points_config.point_encoding = "pointnet2_features"

    with pytest.raises(
        DynamicModelInspectionError,
        match="dequantized int16 XYZ",
    ):
        NativePoints3dCircuitInspector().inspect(
            model,
            model_name="feature-front-end",
        )


def test_native_points3d_inspector_rejects_an_order_dependent_reduction() -> None:
    model = _model(5)
    model.config.points_config.pooling = "first_point"

    with pytest.raises(
        DynamicModelInspectionError,
        match="symmetric pooling",
    ):
        NativePoints3dCircuitInspector().inspect(model, model_name="ordered-pool")


def test_native_points3d_inspector_requires_the_token_factorization_to_divide(
) -> None:
    model = _model(6)
    model.config.points_config.num_tokens = POINT_TOKENS + 1

    with pytest.raises(
        DynamicModelInspectionError,
        match="num_tokens \\* hidden_size",
    ):
        NativePoints3dCircuitInspector().inspect(model, model_name="ragged-tokens")


def test_native_points3d_inspector_rejects_an_undeclared_emission_head() -> None:
    model = _model(7)
    model.centroid_head = torch.nn.Linear(HIDDEN, COMPONENTS, bias=False)

    with pytest.raises(
        DynamicModelInspectionError,
        match="without declaring",
    ):
        NativePoints3dCircuitInspector().inspect(model, model_name="undeclared-head")


# --------------------------------------------------------------------------
# Target synthesis
# --------------------------------------------------------------------------


def test_native_points3d_target_adapter_creates_exact_seeded_local_blank() -> None:
    parent = _model(8)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    expected_delta = LAYERS * (INTERMEDIATE - (INTERMEDIATE - 1)) * HIDDEN * 3
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="native-points3d-child",
        target_family="native_points3d_chat",
        target_modalities=("points3d", "text"),
        exact_unique_parameters=expected_count,
        maximum_parameter_bytes=expected_count * 4,
        tokenizer_fingerprint=tokenizer,
        blank_seed=123,
        configuration=(("decoder_intermediate_size", INTERMEDIATE - 1),),
    )
    adapter = NativePoints3dTargetSynthesisAdapter()
    constructor_calls = _TinyNativePoints3dChatModel.constructor_calls

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert _TinyNativePoints3dChatModel.constructor_calls == constructor_calls + 1
    assert type(result.model) is type(parent)
    assert parent.config.decoder_config.intermediate_size == INTERMEDIATE
    assert result.model.config.decoder_config.intermediate_size == INTERMEDIATE - 1
    assert result.graph.unique_parameter_count == expected_count
    assert result.resource_profile.modalities == ("points3d", "text")
    assert result.adapter_report["exact_parameter_delta"] == expected_delta
    assert result.adapter_report["points3d_front_end_copied_exactly"] is True
    assert result.adapter_report["local_parent_class_instantiation"] is True
    assert result.adapter_report["hub_loader_used"] is False
    assert result.adapter_report["optimizer_steps"] == 0
    assert result.adapter_report["backward_calls"] == 0
    assert result.as_dict()["independently_reinspected"] is True
    assert result.structural_buffer_audit is not None
    assert result.structural_buffer_audit.nonzero_buffer_count == 0
    assert result.blank_provenance is not None
    assert result.blank_provenance.nonzero_structural_buffer_count == 0

    repeated = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )
    for name, parameter in result.model.state_dict().items():
        assert torch.equal(parameter, repeated.model.state_dict()[name])


def test_native_points3d_target_adapter_fails_closed_on_wrong_budget() -> None:
    parent = _model(9)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    expected_count = parent_graph.unique_parameter_count - (LAYERS * HIDDEN * 3)
    request = DynamicTargetRequest(
        child_name="wrong-budget",
        target_family="native_points3d_chat",
        target_modalities=("points3d", "text"),
        exact_unique_parameters=expected_count - 1,
        tokenizer_fingerprint=tokenizer,
        configuration=(("decoder_intermediate_size", INTERMEDIATE - 1),),
    )

    with pytest.raises(
        DynamicPoints3dAdapterError,
        match="exact requested parameter count",
    ):
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry(
                (NativePoints3dTargetSynthesisAdapter(),)
            ),
        ).synthesize(
            request=request,
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


def test_native_points3d_target_adapter_rejects_extra_surgery() -> None:
    parent = _model(10)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (NativePoints3dTargetSynthesisAdapter(),)
        ),
    )

    with pytest.raises(DynamicPoints3dAdapterError, match="smaller than the parent"):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="not-smaller",
                target_family="native_points3d_chat",
                target_modalities=("points3d", "text"),
                exact_unique_parameters=parent_graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
                configuration=(("decoder_intermediate_size", INTERMEDIATE),),
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    with pytest.raises(
        DynamicPoints3dAdapterError,
        match="Unsupported native points3d target configuration",
    ):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="extra-surgery",
                target_family="native_points3d_chat",
                target_modalities=("points3d", "text"),
                exact_unique_parameters=parent_graph.unique_parameter_count
                - LAYERS * HIDDEN * 3,
                tokenizer_fingerprint=tokenizer,
                configuration=(
                    ("decoder_intermediate_size", INTERMEDIATE - 1),
                    ("points_hidden_size", 3),
                ),
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


# --------------------------------------------------------------------------
# Planner and registry
# --------------------------------------------------------------------------


def test_native_points3d_method_is_registered_experimental() -> None:
    method = default_method_registry().get(METHOD_ID)

    assert method.status == MethodStatus.EXPERIMENTAL.value
    assert method.selectable
    assert method.zero_gradient
    assert method.family_id == "native_points3d_coupled_swiglu_model"
    assert method.source_kinds == ("native_points3d_model",)
    assert method.target_kinds == ("native_points3d_model",)
    assert method.evidence_channels == (
        "cached_parent_emitted_media",
        "cached_parent_outputs",
        "child_state",
        "points3d_conditioned_post_swiglu",
        "residual_endpoints",
    )
    assert set(method.coupled_roles) == {
        "q_k_v_o_cache",
        "gate_up_post_swiglu_down",
        "all_layers_ordered_residual",
    }


def test_planner_emits_one_native_points3d_action_with_bound_evidence() -> None:
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        _model(11),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child_graph = inspect_model_circuits(
        _model(12, intermediate_size=INTERMEDIATE - 1),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    method = default_method_registry().get(METHOD_ID)

    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "points3d"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert plan.gaps == ()
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.method_id == METHOD_ID
    assert action.evidence_requirements == method.evidence_channels
    configuration = dict(action.configuration)
    assert configuration["promotion_gate"] == (
        "cached_parent_points3d_behavior_then_fresh_reload"
    )
    assert configuration["fit_objective"].startswith("points3d_conditioned_single")
    assert configuration["layer_count"] == LAYERS
    assert configuration["emits_modality_output"] is False
    assert set(action.target_parameter_paths) == set(child_graph.parameter_paths)


def test_planner_refuses_a_native_points3d_tokenizer_change() -> None:
    parent_graph = inspect_model_circuits(
        _model(13),
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    child_graph = inspect_model_circuits(
        _model(14, intermediate_size=INTERMEDIATE - 1),
        model_name="child",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "other"}),
    )

    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "points3d"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert all(action.method_id != METHOD_ID for action in plan.actions)


def test_planner_refuses_to_fold_an_emitting_parent_into_a_silent_child() -> None:
    """Folding must never change whether -- or what -- a child emits."""

    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        _model(15, emit_centroid=True),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child_graph = inspect_model_circuits(
        _model(16, intermediate_size=INTERMEDIATE - 1),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "points3d"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert all(action.method_id != METHOD_ID for action in plan.actions)


# --------------------------------------------------------------------------
# Executable end-to-end fold
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def points3d_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("native-points3d-e2e")
    payload = build_native_points3d_probe(
        root / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=17,
    )
    outputs = write_points3d_parent_output_cache(
        payload["manifest"],
        root / "parent-outputs.jsonl",
        parent_runner=lambda probe: probe.answer,
    )
    return {"manifest": payload["manifest"], "outputs": str(outputs)}


def _trace_runner(calls: list[str]):
    def run(model: Any, record: Any) -> Any:
        calls.append(record.probe_id)
        return model(
            _cloud(record.points_path),
            _prompt_tokens(record.category),
        )

    return run


def _behavior_runner(model: Any, record: Any) -> str:
    """Run the whole child, then answer only if the front end survived exactly.

    The toy decoder cannot really read a point cloud, so -- exactly as the NEO
    vision, native audio, and native video proofs do -- the closed-set answer is
    gated on the parent-derived marker rather than on emergent competence.  The
    forward call above is not decorative: the gate's execution contract rejects
    any runner that answers without executing the model root, and the cloud is
    the real decoded quantized payload of the sealed probe.
    """

    model(_cloud(record.points_path), _prompt_tokens(record.category))
    marker = float(model.point_encoder.point_linear.weight.detach().flatten()[0])
    if marker == pytest.approx(MARKER_VALUE):
        return record.expected
    return "not-a-direct-answer"


def _tiny_model_saver(model: Any, destination: Path) -> None:
    payload = {
        "intermediate_size": model.config.decoder_config.intermediate_size,
        "emit_centroid": bool(model.config.points_config.emit_centroid),
        "state_dict": {
            name: value.detach().to(device="cpu")
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, destination / "tiny_native_points3d.pt")


def _tiny_model_loader(destination: Path):
    payload = torch.load(
        destination / "tiny_native_points3d.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = _model(
        99,
        intermediate_size=int(payload["intermediate_size"]),
        emit_centroid=bool(payload["emit_centroid"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    # The published standalone loader is the artifact's inference contract: it
    # must hand back frozen parameters, not merely an eval-mode module.
    return model.requires_grad_(False).eval()


def test_native_points3d_fold_cache_stack_behavior_artifact_and_receipt(
    points3d_evidence: dict[str, Any],
    tmp_path: Path,
) -> None:
    behavior_cache = load_points3d_parent_behavior_cache(
        points3d_evidence["manifest"],
        points3d_evidence["outputs"],
    )
    assert len(behavior_cache.calibration) == 12
    assert len(behavior_cache.heldout) == 4

    parent = _model(21)
    with torch.no_grad():
        parent.point_encoder.point_linear.weight.fill_(MARKER_VALUE)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    assert {interface.modality for interface in parent_graph.interfaces} == {
        "points3d",
        "text",
    }

    trace_calls: list[str] = []
    statistics_path = tmp_path / "trace-cache"
    statistics = build_native_points3d_fold_trace_cache(
        parent,
        parent_graph,
        behavior_cache,
        trace_runner=_trace_runner(trace_calls),
        destination=statistics_path,
        sketch_dim=4,
    )
    assert trace_calls == [record.probe_id for record in behavior_cache.calibration]
    assert set(trace_calls).isdisjoint(
        record.probe_id for record in behavior_cache.heldout
    )
    assert statistics.sample_runner_calls == len(behavior_cache.calibration)
    assert statistics.trace_rows == (
        len(behavior_cache.calibration) * (POINT_TOKENS + PROMPT_TOKENS),
    ) * LAYERS

    fold_cache_path = tmp_path / "fold-cache"
    cache = derive_native_points3d_fold_cache(
        statistics,
        destination=fold_cache_path,
        candidate_count=3,
        anchor_count=2,
    )
    assert trace_calls == [record.probe_id for record in behavior_cache.calibration]
    assert cache.source_cache_key == statistics.cache_key
    assert cache.anchor_count == 2
    assert cache.sample_runner_calls == statistics.sample_runner_calls
    assert all(
        selection.source_neurons == INTERMEDIATE for selection in cache.selections
    )
    assert all(
        selection.target_neurons == INTERMEDIATE - 1 for selection in cache.selections
    )
    assert all(len(selection.compensation_terms) == 2 for selection in cache.selections)
    manifest = json.loads(
        (fold_cache_path / "manifest.json").read_text(encoding="ascii")
    )
    assert manifest["schema"] == FOLD_CACHE_SCHEMA
    assert manifest["cache_key"] == cache.cache_key
    assert manifest["selection_excluded_splits"] == ["heldout"]
    assert manifest["statistics_reused_without_parent_forward"] is True

    # The statistics cache is untouched by selection and still seals as v1
    # evidence with no fold policy inside it.
    statistics_manifest = json.loads(
        (statistics_path / "manifest.json").read_text(encoding="ascii")
    )
    assert statistics_manifest["fold_selection_included"] is False
    reopened_statistics = Points3dTraceStatisticsCache.open(
        statistics_path,
        behavior_cache=behavior_cache,
    )
    assert reopened_statistics.cache_key == statistics.cache_key

    tampered = tmp_path / "tampered-fold-cache"
    shutil.copytree(fold_cache_path, tampered)
    tampered_manifest = tampered / "manifest.json"
    tampered_payload = json.loads(tampered_manifest.read_text(encoding="ascii"))
    tampered_payload["candidate_count"] = 2
    tampered_manifest.write_text(
        json.dumps(tampered_payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(DynamicPoints3dFoldError, match="manifest seal"):
        NativePoints3dFoldCache.open(tampered, behavior_cache=behavior_cache)
    assert (
        NativePoints3dFoldCache.open(
            fold_cache_path,
            behavior_cache=behavior_cache,
        ).cache_key
        == cache.cache_key
    )

    behavior_gate = NativePoints3dBehaviorGate(
        behavior_cache,
        child_runner=_behavior_runner,
    )
    final_binding = resolve_behavior_gate_binding(behavior_gate.evaluate)
    fresh_binding = resolve_behavior_gate_binding(behavior_gate.evaluate_model)
    assert final_binding.manifest_fingerprint == fresh_binding.manifest_fingerprint
    assert final_binding.manifest["inference_execution"]["entrypoint"] == "__call__"
    unmarked = _model(22, intermediate_size=INTERMEDIATE - 1)
    assert not behavior_gate.evaluate_model(unmarked).accepted

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    resources = NativePoints3dStackResources(
        cache=cache,
        behavior_gate=behavior_gate,
        model_saver=_tiny_model_saver,
        model_loader=_tiny_model_loader,
        saver_id="tiny-native-points3d-direct-state-v1",
        loader_id="tiny-native-points3d-direct-state-v1",
        snapshot_config=DynamicFoldSnapshotConfig.disk(snapshots, chunk_bytes=1024),
    )
    expected_delta = LAYERS * 3 * HIDDEN
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="tiny-native-points3d-folded",
        target_family="native_points3d_chat",
        target_modalities=("points3d", "text"),
        exact_unique_parameters=expected_count,
        maximum_parameter_bytes=expected_count * 4,
        tokenizer_fingerprint=tokenizer,
        blank_seed=123,
        configuration=(
            ("decoder_intermediate_size", INTERMEDIATE - 1),
            ("fold_cache_key", cache.cache_key),
        ),
    )
    artifact_path = tmp_path / "published"
    result = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry(
            (NativePoints3dTargetSynthesisAdapter(),)
        ),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (NativePoints3dFoldStackAdapter(resources),)
        ),
        snapshot_config=resources.snapshot_config,
    ).run(
        parent_models={"parent": parent},
        target_request=request,
        evidence_fingerprint=cache.cache_key,
        destination=artifact_path,
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert result.complete
    assert len(result.published.compilation.plan.actions) == 1
    assert result.published.compilation.plan.actions[0].method_id == METHOD_ID
    assert result.target.graph.unique_parameter_count == expected_count
    assert result.published.artifact.exact_unique_parameters == expected_count
    assert result.published.artifact.fresh_reload_decision.accepted
    assert result.published.artifact.fresh_reload_parameters_frozen is True
    assert behavior_gate.evaluation_count == 4
    assert behavior_gate.last_report is not None
    assert behavior_gate.last_report["parent_forward_count"] == 0
    assert behavior_gate.last_report["failures"] == []
    assert behavior_gate.last_report["by_split"]["heldout"][
        "parent_correct_retained"
    ] == behavior_gate.last_report["by_split"]["heldout"]["parent_correct_total"]
    assert not any(snapshots.iterdir())

    reloaded = _tiny_model_loader(artifact_path / "child")
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    parent_state = parent.state_dict()
    child_state = reloaded.state_dict()
    assert set(parent_state) == set(child_state)
    for name, value in child_state.items():
        if ".mlp." not in name:
            assert torch.equal(value, parent_state[name]), name
    for selection in cache.selections:
        prefix = f"language_model.model.layers.{selection.layer_index}.mlp"
        survivors = torch.tensor(selection.survivors, dtype=torch.long)
        assert torch.equal(
            child_state[f"{prefix}.gate_proj.weight"],
            parent_state[f"{prefix}.gate_proj.weight"].index_select(0, survivors),
        )
        assert torch.equal(
            child_state[f"{prefix}.up_proj.weight"],
            parent_state[f"{prefix}.up_proj.weight"].index_select(0, survivors),
        )
        expected_down = (
            parent_state[f"{prefix}.down_proj.weight"]
            .index_select(1, survivors)
            .clone()
        )
        for (
            _anchor_neuron,
            anchor_target_index,
            coefficient,
        ) in selection.compensation_terms:
            expected_down[:, anchor_target_index].add_(
                parent_state[f"{prefix}.down_proj.weight"][:, selection.removed_neuron],
                alpha=coefficient,
            )
        assert torch.equal(
            child_state[f"{prefix}.down_proj.weight"],
            expected_down,
        )
    changed = sum(
        int(parent_state[name].numel()) - int(child_state[name].numel())
        for name in parent_state
    )
    assert changed == expected_delta

    # The reloaded child still runs the real geometry path end to end.
    with torch.no_grad():
        logits = reloaded(
            _cloud(behavior_cache.records[0].points_path),
            _prompt_tokens(behavior_cache.records[0].category),
        )
    assert tuple(logits.shape) == (1, POINT_TOKENS + PROMPT_TOKENS, VOCAB)
    assert bool(torch.isfinite(logits).all())

    receipt = write_native_points3d_dynamic_fold_receipt(
        result,
        cache,
        behavior_gate,
        destination=tmp_path / "receipt.json",
    )
    receipt_body = dict(receipt)
    receipt_sha = receipt_body.pop("receipt_sha256")
    assert canonical_sha256(receipt_body) == receipt_sha
    assert receipt["method_id"] == METHOD_ID
    assert receipt["modality"] == "points3d"
    assert receipt["exact_unique_parameters"] == expected_count
    assert receipt["heldout_samples_used_for_selection"] == 0
    assert receipt["parent_forward_count_during_fold_and_gates"] == 0
    assert receipt["fresh_reload_parameters_frozen"] is True
    assert receipt["trace_capture_calls_this_run"] == len(behavior_cache.calibration)
    assert receipt["optimizer_steps"] == 0
    assert receipt["backward_calls"] == 0
    assert receipt["source_cache_key"] == statistics.cache_key


def test_points3d_lane_status_reports_the_executable_lane() -> None:
    status = points3d_lane_status()

    assert status["modality"] == "points3d"
    assert status["method_id"] == METHOD_ID
    assert status["parent_forwards_during_gating"] == 0
    assert any("inspector" in item for item in status["provided"])
    assert any("MODALITY_OUTPUT" in item for item in status["provided"])
