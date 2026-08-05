"""End-to-end proof that the native audio lane is executable at toy scale.

The architecture below is the audio mirror of ``_TinyNeoChatModel`` in
``tests/test_dynamic_neo_native_vision.py``: a *learned* strided PCM16 frame
encoder feeds audio context tokens into a tiny causal decoder alongside text
tokens.  Nothing here loads or freezes an external encoder; every audio tensor
is a parameter this repository owns, which is what the project's
native-modality rule requires.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_forge.dynamic_audio_adapter import (
    DynamicAudioAdapterError,
    NativeAudioTargetSynthesisAdapter,
)
from neural_forge.dynamic_audio_fold import (
    FOLD_CACHE_SCHEMA,
    METHOD_ID,
    AudioTraceStatisticsCache,
    DynamicAudioFoldError,
    NativeAudioBehaviorGate,
    NativeAudioFoldCache,
    NativeAudioFoldStackAdapter,
    NativeAudioStackResources,
    audio_lane_status,
    build_native_audio_fold_trace_cache,
    derive_native_audio_fold_cache,
    load_audio_parent_behavior_cache,
    write_audio_parent_output_cache,
    write_native_audio_dynamic_fold_receipt,
)
from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    EdgeKind,
    canonical_sha256,
)
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_fold_registry import MethodStatus, default_method_registry
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import (
    NATIVE_AUDIO_LANE_MARKER,
    DynamicModelInspectionError,
    NativeAudioCircuitInspector,
    default_inspector_registry,
    inspect_model_circuits,
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
from neural_forge.native_audio_probe import (
    PROBE_FRAMES,
    build_native_audio_probe,
    read_pcm16_wav,
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
AUDIO_HIDDEN = 4
FRAME_SIZE = 500
AUDIO_FRAMES = PROBE_FRAMES // FRAME_SIZE
PROMPT_TOKENS = 6
MARKER_VALUE = 0.375


# --------------------------------------------------------------------------
# A tiny native audio-text architecture
# --------------------------------------------------------------------------


class _AudioFrontEndConfig:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_size: int = FRAME_SIZE,
        num_channels: int = 1,
        hidden_size: int = AUDIO_HIDDEN,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.sample_encoding = "int16_normalized_pcm"

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "num_channels": self.num_channels,
            "hidden_size": self.hidden_size,
            "sample_encoding": self.sample_encoding,
        }


class _AudioDecoderConfig:
    model_type = "native_audio_text_decoder"

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


class _NativeAudioConfig:
    model_type = "native_audio_chat"
    neural_forge_native_audio_lane = NATIVE_AUDIO_LANE_MARKER

    def __init__(
        self,
        audio_config: _AudioFrontEndConfig,
        decoder_config: _AudioDecoderConfig,
    ) -> None:
        self.audio_config = audio_config
        self.decoder_config = decoder_config
        self.hidden_size = decoder_config.hidden_size
        self.tie_word_embeddings = decoder_config.tie_word_embeddings

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "neural_forge_native_audio_lane": self.neural_forge_native_audio_lane,
            "audio_config": self.audio_config.to_dict(),
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


class _AudioFrontEnd(torch.nn.Module):
    """Learned strided 1D convolution over int16-normalized PCM samples."""

    def __init__(self, config: _NativeAudioConfig) -> None:
        super().__init__()
        audio = config.audio_config
        self.frame_conv = torch.nn.Conv1d(
            audio.num_channels,
            audio.hidden_size,
            kernel_size=audio.frame_size,
            stride=audio.frame_size,
        )
        self.projection = torch.nn.Linear(
            audio.hidden_size,
            config.decoder_config.hidden_size,
            bias=False,
        )
        self.activation = torch.nn.GELU()

    def forward(self, waveform: Any) -> Any:
        frames = self.activation(self.frame_conv(waveform))
        return self.projection(frames.transpose(1, 2))


class _Attention(torch.nn.Module):
    def __init__(self, config: _AudioDecoderConfig) -> None:
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
    def __init__(self, config: _AudioDecoderConfig) -> None:
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
    def __init__(self, config: _AudioDecoderConfig) -> None:
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
    def __init__(self, config: _AudioDecoderConfig) -> None:
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


class _AudioTextLanguageModel(torch.nn.Module):
    def __init__(self, config: _AudioDecoderConfig) -> None:
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

    def forward(self, hidden_states: Any) -> Any:
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.model.norm(hidden_states))


class _TinyNativeAudioChatModel(torch.nn.Module):
    constructor_calls = 0

    def __init__(self, config: _NativeAudioConfig) -> None:
        super().__init__()
        type(self).constructor_calls += 1
        self.config = config
        self.audio_encoder = _AudioFrontEnd(config)
        self.language_model = _AudioTextLanguageModel(config.decoder_config)

    def forward(self, waveform: Any, token_ids: Any) -> Any:
        audio_tokens = self.audio_encoder(waveform)
        text_tokens = self.language_model.model.embed_tokens(token_ids)
        hidden_states = torch.cat((audio_tokens, text_tokens), dim=1)
        positions = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        )
        hidden_states = hidden_states + self.language_model.model.position_embedding(
            positions
        )
        return self.language_model(hidden_states)


def _config(*, intermediate_size: int = INTERMEDIATE) -> _NativeAudioConfig:
    return _NativeAudioConfig(
        _AudioFrontEndConfig(),
        _AudioDecoderConfig(intermediate_size=intermediate_size),
    )


def _model(
    seed: int,
    *,
    intermediate_size: int = INTERMEDIATE,
) -> _TinyNativeAudioChatModel:
    torch.manual_seed(seed)
    return _TinyNativeAudioChatModel(
        _config(intermediate_size=intermediate_size)
    ).eval()


def _tokenizer_fingerprint() -> str:
    return canonical_sha256({"tokenizer": "tiny-native-audio"})


def _waveform(audio_path: str) -> Any:
    samples, _rate, _channels = read_pcm16_wav(audio_path)
    values = np.asarray(samples, dtype=np.float32) / 32_768.0
    return torch.from_numpy(values.reshape(1, 1, -1).copy())


def _prompt_tokens(category: str) -> Any:
    padded = (category + "______")[:PROMPT_TOKENS]
    return torch.tensor(
        [[1 + (ord(character) % (VOCAB - 2)) for character in padded]],
        dtype=torch.long,
    )


# --------------------------------------------------------------------------
# Inspector
# --------------------------------------------------------------------------


def test_native_audio_inspector_owns_frame_encoder_and_decoder_exactly() -> None:
    model = _model(1)
    tokenizer = _tokenizer_fingerprint()

    graph = inspect_model_circuits(
        model,
        model_name="tiny-native-audio",
        tokenizer_fingerprint=tokenizer,
    )

    assert default_inspector_registry().select(model).name == "native-audio"
    assert graph.family == "native_audio_chat"
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
    assert {interface.modality for interface in graph.interfaces} == {"audio", "text"}
    assert {interface.interface_id for interface in graph.interfaces} == {
        "audio.pcm16_frames.input",
        "text.logits.output",
        "text.tokens.input",
    }
    audio_interface = next(
        interface
        for interface in graph.interfaces
        if interface.modality == "audio"
    )
    assert audio_interface.direction == "input"
    assert audio_interface.representation == "pcm16_frames"

    modality_inputs = graph.nodes_of_kind(CircuitKind.MODALITY_INPUT)
    assert len(modality_inputs) == 1
    assert modality_inputs[0].node_id == "input.audio.pcm16_frame_encoder"
    assert audio_interface.node_id == modality_inputs[0].node_id
    assert modality_inputs[0].input("waveform").modality == "audio"
    assert set(modality_inputs[0].parameter_paths) == {
        "audio_encoder.frame_conv.weight",
        "audio_encoder.frame_conv.bias",
    }
    assert len(graph.nodes_of_kind(CircuitKind.GATED_MLP)) == LAYERS
    assert any(edge.kind == EdgeKind.MODALITY.value for edge in graph.edges)

    projection = graph.node("input.audio.dense_projection")
    assert projection.parameter_paths == ("audio_encoder.projection.weight",)

    metadata = dict(graph.metadata)
    expected_audio_parameters = (
        AUDIO_HIDDEN * 1 * FRAME_SIZE + AUDIO_HIDDEN + HIDDEN * AUDIO_HIDDEN
    )
    assert metadata["audio_parameter_count"] == expected_audio_parameters
    assert metadata["decoder_intermediate_size"] == INTERMEDIATE
    assert metadata["native_pcm16_frame_encoder"] is True
    assert metadata["external_frozen_audio_encoder"] is False
    assert metadata["audio_frame_size"] == FRAME_SIZE


def test_native_audio_inspector_never_claims_an_unmarked_model() -> None:
    model = _model(2)
    inspector = NativeAudioCircuitInspector()
    assert inspector.supports(model)

    model.config.neural_forge_native_audio_lane = "some-other-lane"
    assert not inspector.supports(model)
    with pytest.raises(DynamicModelInspectionError, match="marked native_audio_chat"):
        inspector.inspect(model, model_name="unmarked")


def test_native_audio_inspector_fails_on_unowned_learned_audio_operator() -> None:
    model = _model(3)
    model.audio_encoder.extra_projection = torch.nn.Linear(AUDIO_HIDDEN, AUDIO_HIDDEN)

    with pytest.raises(
        DynamicModelInspectionError,
        match="outside its frame encoder",
    ):
        NativeAudioCircuitInspector().inspect(model, model_name="malformed-audio")


def test_native_audio_inspector_rejects_a_precomputed_feature_front_end() -> None:
    model = _model(4)
    model.config.audio_config.sample_encoding = "log_mel"

    with pytest.raises(
        DynamicModelInspectionError,
        match="int16-normalized PCM",
    ):
        NativeAudioCircuitInspector().inspect(model, model_name="feature-front-end")


# --------------------------------------------------------------------------
# Target synthesis
# --------------------------------------------------------------------------


def test_native_audio_target_adapter_creates_exact_seeded_local_blank() -> None:
    parent = _model(5)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    expected_delta = LAYERS * (INTERMEDIATE - (INTERMEDIATE - 1)) * HIDDEN * 3
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="native-audio-child",
        target_family="native_audio_chat",
        target_modalities=("text", "audio"),
        exact_unique_parameters=expected_count,
        maximum_parameter_bytes=expected_count * 4,
        tokenizer_fingerprint=tokenizer,
        blank_seed=123,
        configuration=(("decoder_intermediate_size", INTERMEDIATE - 1),),
    )
    adapter = NativeAudioTargetSynthesisAdapter()
    constructor_calls = _TinyNativeAudioChatModel.constructor_calls

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert _TinyNativeAudioChatModel.constructor_calls == constructor_calls + 1
    assert type(result.model) is type(parent)
    assert parent.config.decoder_config.intermediate_size == INTERMEDIATE
    assert result.model.config.decoder_config.intermediate_size == INTERMEDIATE - 1
    assert result.graph.unique_parameter_count == expected_count
    assert result.resource_profile.modalities == ("audio", "text")
    assert result.adapter_report["exact_parameter_delta"] == expected_delta
    assert result.adapter_report["audio_front_end_copied_exactly"] is True
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


def test_native_audio_target_adapter_fails_closed_on_wrong_budget() -> None:
    parent = _model(6)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    expected_count = parent_graph.unique_parameter_count - (LAYERS * HIDDEN * 3)
    request = DynamicTargetRequest(
        child_name="wrong-budget",
        target_family="native_audio_chat",
        target_modalities=("audio", "text"),
        exact_unique_parameters=expected_count - 1,
        tokenizer_fingerprint=tokenizer,
        configuration=(("decoder_intermediate_size", INTERMEDIATE - 1),),
    )

    with pytest.raises(
        DynamicAudioAdapterError,
        match="exact requested parameter count",
    ):
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry(
                (NativeAudioTargetSynthesisAdapter(),)
            ),
        ).synthesize(
            request=request,
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


def test_native_audio_target_adapter_rejects_extra_surgery() -> None:
    parent = _model(7)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((NativeAudioTargetSynthesisAdapter(),)),
    )

    with pytest.raises(DynamicAudioAdapterError, match="smaller than the parent"):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="not-smaller",
                target_family="native_audio_chat",
                target_modalities=("audio", "text"),
                exact_unique_parameters=parent_graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
                configuration=(("decoder_intermediate_size", INTERMEDIATE),),
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    with pytest.raises(
        DynamicAudioAdapterError,
        match="Unsupported native audio target configuration",
    ):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="extra-surgery",
                target_family="native_audio_chat",
                target_modalities=("audio", "text"),
                exact_unique_parameters=parent_graph.unique_parameter_count
                - LAYERS * HIDDEN * 3,
                tokenizer_fingerprint=tokenizer,
                configuration=(
                    ("decoder_intermediate_size", INTERMEDIATE - 1),
                    ("audio_hidden_size", 3),
                ),
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


# --------------------------------------------------------------------------
# Planner and registry
# --------------------------------------------------------------------------


def test_native_audio_method_is_registered_experimental_with_audio_channels() -> None:
    method = default_method_registry().get(METHOD_ID)

    assert method.status == MethodStatus.EXPERIMENTAL.value
    assert method.selectable
    assert method.zero_gradient
    assert method.family_id == "native_audio_coupled_swiglu_model"
    assert method.source_kinds == ("native_audio_model",)
    assert method.target_kinds == ("native_audio_model",)
    assert method.evidence_channels == (
        "audio_conditioned_post_swiglu",
        "cached_parent_outputs",
        "child_state",
        "residual_endpoints",
    )
    assert set(method.coupled_roles) == {
        "q_k_v_o_cache",
        "gate_up_post_swiglu_down",
        "all_layers_ordered_residual",
    }


def test_planner_emits_one_native_audio_action_with_bound_evidence() -> None:
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        _model(8),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child_graph = inspect_model_circuits(
        _model(9, intermediate_size=INTERMEDIATE - 1),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    method = default_method_registry().get(METHOD_ID)

    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "audio"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert plan.gaps == ()
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.method_id == METHOD_ID
    assert action.evidence_requirements == method.evidence_channels
    configuration = dict(action.configuration)
    assert configuration["promotion_gate"] == (
        "cached_parent_audio_behavior_then_fresh_reload"
    )
    assert configuration["fit_objective"].startswith("audio_conditioned_single_swiglu")
    assert configuration["layer_count"] == LAYERS
    assert set(action.target_parameter_paths) == set(child_graph.parameter_paths)


def test_planner_refuses_a_native_audio_tokenizer_change() -> None:
    parent_graph = inspect_model_circuits(
        _model(10),
        model_name="parent",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    child_graph = inspect_model_circuits(
        _model(11, intermediate_size=INTERMEDIATE - 1),
        model_name="child",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "other"}),
    )

    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "audio"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert all(action.method_id != METHOD_ID for action in plan.actions)


# --------------------------------------------------------------------------
# Executable end-to-end fold
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audio_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("native-audio-e2e")
    payload = build_native_audio_probe(
        root / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=17,
    )
    outputs = write_audio_parent_output_cache(
        payload["manifest"],
        root / "parent-outputs.jsonl",
        parent_runner=lambda probe: probe.answer,
    )
    return {"manifest": payload["manifest"], "outputs": str(outputs)}


def _trace_runner(calls: list[str]):
    def run(model: Any, record: Any) -> Any:
        calls.append(record.probe_id)
        return model(
            _waveform(record.audio_path),
            _prompt_tokens(record.category),
        )

    return run


def _behavior_runner(model: Any, record: Any) -> str:
    """Run the whole child, then answer only if the front end survived exactly.

    The toy decoder cannot really transcribe audio, so -- exactly as the NEO
    vision proof does -- the closed-set answer is gated on the parent-derived
    marker rather than on emergent competence.  The forward call above is not
    decorative: the gate's execution contract rejects any runner that answers
    without executing the model root, and the waveform is the real decoded
    PCM16 of the sealed probe.
    """

    model(_waveform(record.audio_path), _prompt_tokens(record.category))
    marker = float(model.audio_encoder.frame_conv.weight.detach().flatten()[0])
    if marker == pytest.approx(MARKER_VALUE):
        return record.expected
    return "not-a-direct-answer"


def _tiny_model_saver(model: Any, destination: Path) -> None:
    payload = {
        "intermediate_size": model.config.decoder_config.intermediate_size,
        "state_dict": {
            name: value.detach().to(device="cpu")
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, destination / "tiny_native_audio.pt")


def _tiny_model_loader(destination: Path):
    payload = torch.load(
        destination / "tiny_native_audio.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = _model(99, intermediate_size=int(payload["intermediate_size"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    # The published standalone loader is the artifact's inference contract: it
    # must hand back frozen parameters, not merely an eval-mode module.
    return model.requires_grad_(False).eval()


@dataclass(frozen=True)
class _GatedChild:
    """One model a behavior gate actually decided on, recorded by identity."""

    model: Any
    state_fingerprint: str
    accepted: bool
    parameter_storages: frozenset[int]


class _RecordingNativeAudioBehaviorGate(NativeAudioBehaviorGate):
    """An audio gate that also records *which* children it decided on.

    ``evaluation_count`` is a bare integer.  It cannot separate "the gate
    decided on the folded child in memory and again on a real reload of the
    published bytes" from "the gate decided on the same object twice", and it
    silently absorbs a publication-side cache that skips a required decision.
    Recording every gated child lets this proof assert on the evidence set the
    count is only a summary of.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gated_children: list[_GatedChild] = []

    def evaluate_model(self, model: Any) -> Any:
        decision = super().evaluate_model(model)
        # Retaining the model keeps its storages alive for the whole test, so
        # ``data_ptr`` values cannot be recycled into a false "disjoint" or a
        # false "same child" reading later.
        self.gated_children.append(
            _GatedChild(
                model=model,
                state_fingerprint=fingerprint_model_state(model),
                accepted=bool(decision.accepted),
                parameter_storages=frozenset(
                    parameter.data_ptr() for parameter in model.parameters()
                ),
            )
        )
        return decision


def test_native_audio_fold_cache_stack_behavior_artifact_and_receipt(
    audio_evidence: dict[str, Any],
    tmp_path: Path,
) -> None:
    behavior_cache = load_audio_parent_behavior_cache(
        audio_evidence["manifest"],
        audio_evidence["outputs"],
    )
    assert len(behavior_cache.calibration) == 9
    assert len(behavior_cache.heldout) == 3

    parent = _model(21)
    with torch.no_grad():
        parent.audio_encoder.frame_conv.weight.fill_(MARKER_VALUE)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    assert {interface.modality for interface in parent_graph.interfaces} == {
        "audio",
        "text",
    }

    trace_calls: list[str] = []
    statistics_path = tmp_path / "trace-cache"
    statistics = build_native_audio_fold_trace_cache(
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
        len(behavior_cache.calibration) * (AUDIO_FRAMES + PROMPT_TOKENS),
    ) * LAYERS

    fold_cache_path = tmp_path / "fold-cache"
    cache = derive_native_audio_fold_cache(
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
    reopened_statistics = AudioTraceStatisticsCache.open(
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
    with pytest.raises(DynamicAudioFoldError, match="manifest seal"):
        NativeAudioFoldCache.open(tampered, behavior_cache=behavior_cache)
    assert (
        NativeAudioFoldCache.open(
            fold_cache_path,
            behavior_cache=behavior_cache,
        ).cache_key
        == cache.cache_key
    )

    behavior_gate = _RecordingNativeAudioBehaviorGate(
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
    resources = NativeAudioStackResources(
        cache=cache,
        behavior_gate=behavior_gate,
        model_saver=_tiny_model_saver,
        model_loader=_tiny_model_loader,
        saver_id="tiny-native-audio-direct-state-v1",
        loader_id="tiny-native-audio-direct-state-v1",
        snapshot_config=DynamicFoldSnapshotConfig.disk(snapshots, chunk_bytes=1024),
    )
    expected_delta = LAYERS * 3 * HIDDEN
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="tiny-native-audio-folded",
        target_family="native_audio_chat",
        target_modalities=("audio", "text"),
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
            (NativeAudioTargetSynthesisAdapter(),)
        ),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (NativeAudioFoldStackAdapter(resources),)
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
    # Three decisions, and -- what the count is only shorthand for -- three
    # decisions on the right subjects: the unmarked control this test rejected
    # by hand, the folded child in memory, and the folded child freshly loaded
    # back off disk.  Publication reuses its own staged fresh-reload verdict
    # for the post-rename destination only after proving the artifact bytes,
    # the reloaded child, the loader, the evaluator, and the gate manifest are
    # all identical, so that fourth call would re-decide identical evidence.
    assert behavior_gate.evaluation_count == 3
    assert len(behavior_gate.gated_children) == 3
    control, in_memory_child, reloaded_child = behavior_gate.gated_children
    assert [record.accepted for record in behavior_gate.gated_children] == [
        False,
        True,
        True,
    ]
    assert control.model is unmarked
    assert control.state_fingerprint != in_memory_child.state_fingerprint
    # The accepted pair must be two *different* children carrying the same
    # folded state: a gate that only ever saw the in-memory child, or a reload
    # step handed the in-memory object back instead of real bytes, fails here.
    assert in_memory_child.state_fingerprint == reloaded_child.state_fingerprint
    assert reloaded_child.model is not in_memory_child.model
    assert in_memory_child.parameter_storages.isdisjoint(
        reloaded_child.parameter_storages
    )
    assert behavior_gate.last_report is not None
    assert behavior_gate.last_report["parent_forward_count"] == 0
    assert behavior_gate.last_report["failures"] == []
    assert behavior_gate.last_report["by_split"]["heldout"][
        "parent_correct_retained"
    ] == behavior_gate.last_report["by_split"]["heldout"]["parent_correct_total"]
    assert not any(snapshots.iterdir())

    reloaded = _tiny_model_loader(artifact_path / "child")
    # The gated reload really was this artifact, not some other child.
    assert fingerprint_model_state(reloaded) == reloaded_child.state_fingerprint
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

    # The reloaded child still runs the real audio path end to end.
    with torch.no_grad():
        logits = reloaded(
            _waveform(behavior_cache.records[0].audio_path),
            _prompt_tokens(behavior_cache.records[0].category),
        )
    assert tuple(logits.shape) == (1, AUDIO_FRAMES + PROMPT_TOKENS, VOCAB)
    assert bool(torch.isfinite(logits).all())

    receipt = write_native_audio_dynamic_fold_receipt(
        result,
        cache,
        behavior_gate,
        destination=tmp_path / "receipt.json",
    )
    receipt_body = dict(receipt)
    receipt_sha = receipt_body.pop("receipt_sha256")
    assert canonical_sha256(receipt_body) == receipt_sha
    assert receipt["method_id"] == METHOD_ID
    assert receipt["modality"] == "audio"
    assert receipt["exact_unique_parameters"] == expected_count
    assert receipt["heldout_samples_used_for_selection"] == 0
    assert receipt["parent_forward_count_during_fold_and_gates"] == 0
    assert receipt["fresh_reload_parameters_frozen"] is True
    assert receipt["trace_capture_calls_this_run"] == len(behavior_cache.calibration)
    assert receipt["optimizer_steps"] == 0
    assert receipt["backward_calls"] == 0
    assert receipt["source_cache_key"] == statistics.cache_key


def test_audio_lane_status_reports_the_executable_lane() -> None:
    status = audio_lane_status()

    assert status["modality"] == "audio"
    assert status["method_id"] == METHOD_ID
    assert status["parent_forwards_during_gating"] == 0
    assert any("inspector" in item for item in status["provided"])
    assert any("MODALITY_OUTPUT" in item for item in status["not_provided"])
