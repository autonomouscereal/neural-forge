"""The repository's first executable ``MODALITY_OUTPUT`` proof.

Every lane before this one reviews media on the *input* side and text on the
output side.  ``CircuitKind.MODALITY_OUTPUT`` existed in the IR and no inspector
emitted it; ``docs/MODALITY_OUTPUT_DESIGN_2026-08-04.md`` is the contract and
this file is its executable half.

The architecture is the same tiny native points3d model as
``tests/test_native_points3d_executable_lane.py``, with
``points_config.emit_centroid`` switched on so the model owns a bias-free
``centroid_head``.  The head projects the decoder's final normalized residual
onto three coordinates, and those coordinates are decoded onto the declared
int16 lattice before review -- a **media-typed** output, not text.

**Honest limit.** This proves the output-side *contract*: a real
``MODALITY_OUTPUT`` node, a sealed parent emission store, a declared and
fingerprinted decode contract, parent-relative retention by exact
decoded-content digest, non-text regression notions, and a standalone fresh
reload that replays all of it.  It proves nothing about generative quality.
As in the NEO vision, native audio, and native video proofs, the toy decoder
cannot actually see the cloud, so the retained emission is gated on a
parent-derived marker surviving the fold exactly; when that marker is gone the
runner falls through to the model's *own* head output, which is a genuinely
different payload and is what the negative cases exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_forge.dynamic_behavior_gate import (
    POINTS3D_HASH_BASIS,
    MediaReference,
    MultimodalCandidateOutput,
    MultimodalWholeOutputCase,
    multimodal_case_coverage,
    resolve_behavior_gate_binding,
)
from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    canonical_sha256,
    emitted_media_interfaces,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import (
    NATIVE_POINTS3D_EMITTED_REPRESENTATION,
    NATIVE_POINTS3D_LANE_MARKER,
    DynamicModelInspectionError,
    NativePoints3dCircuitInspector,
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_points3d_adapter import NativePoints3dTargetSynthesisAdapter
from neural_forge.dynamic_points3d_fold import (
    METHOD_ID,
    NativePoints3dFoldStackAdapter,
    NativePoints3dStackResources,
    build_native_points3d_fold_trace_cache,
    derive_native_points3d_fold_cache,
    load_points3d_parent_behavior_cache,
    write_native_points3d_dynamic_fold_receipt,
    write_points3d_parent_output_cache,
)
from neural_forge.dynamic_points3d_output import (
    EMISSION_DECODER_SCHEMA,
    EMISSION_SCHEMA,
    EMISSIONS_INDEX,
    OUTPUT_GATE_ID,
    OUTPUT_GATE_REPORT_SCHEMA,
    DynamicPoints3dOutputError,
    NativePoints3dEmittedOutputGate,
    Points3dEmissionDecodeError,
    Points3dEmittedOutputGateConfig,
    assert_graph_emits_points3d,
    emitted_output_contract_status,
    emitted_points3d_media,
    load_points3d_parent_emission_cache,
    quantize_emitted_points3d,
    write_points3d_parent_emission_cache,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
)
from neural_forge.native_points3d_probe import (
    COMPONENTS,
    POINT_COUNT,
    PROBE_SCALE,
    PROBE_ZERO_POINT,
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
# The same tiny native points3d-text architecture, with an emitting head
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
        emit_centroid: bool = True,
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


class _TinyEmittingPoints3dChatModel(torch.nn.Module):
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
    emit_centroid: bool = True,
) -> _NativePoints3dConfig:
    return _NativePoints3dConfig(
        _PointFrontEndConfig(emit_centroid=emit_centroid),
        _Points3dDecoderConfig(intermediate_size=intermediate_size),
    )


def _model(
    seed: int,
    *,
    intermediate_size: int = INTERMEDIATE,
    emit_centroid: bool = True,
) -> _TinyEmittingPoints3dChatModel:
    torch.manual_seed(seed)
    return _TinyEmittingPoints3dChatModel(
        _config(intermediate_size=intermediate_size, emit_centroid=emit_centroid)
    ).eval()


def _tokenizer_fingerprint() -> str:
    return canonical_sha256({"tokenizer": "tiny-emitting-points3d"})


def _cloud(points_path: str) -> Any:
    points, quantization = read_points3d_cloud(points_path)
    metres = quantization.dequantize(points).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(metres[None, ...]))


def _prompt_tokens(category: str) -> Any:
    padded = (category + "______")[:PROMPT_TOKENS]
    return torch.tensor(
        [[1 + (ord(character) % (VOCAB - 2)) for character in padded]],
        dtype=torch.long,
    )


def _dequantized_centroid(points_path: str) -> np.ndarray:
    """The geometric centroid of one sealed cloud, in dequantized units."""

    points, quantization = read_points3d_cloud(points_path)
    return quantization.dequantize(points).mean(axis=0).reshape(1, COMPONENTS)


# --------------------------------------------------------------------------
# Inspector: the first MODALITY_OUTPUT node
# --------------------------------------------------------------------------


def test_inspector_emits_the_first_modality_output_node() -> None:
    model = _model(1)
    graph = inspect_model_circuits(
        model,
        model_name="emitting-points3d",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )
    silent = inspect_model_circuits(
        _model(1, emit_centroid=False),
        model_name="silent-points3d",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )

    # Flipping one config flag adds exactly one bias-free Linear(hidden -> 3),
    # one node, and one interface -- nothing else moves.
    assert silent.unique_parameter_count == 1320
    assert graph.unique_parameter_count == 1320 + HIDDEN * COMPONENTS == 1344
    assert (len(silent.nodes), len(silent.interfaces)) == (19, 3)
    assert (len(graph.nodes), len(graph.interfaces)) == (20, 4)

    assert default_inspector_registry().select(model).name == "native-points3d"
    emitted = graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT)
    assert len(emitted) == 1
    node = emitted[0]
    assert node.node_id == "output.points3d.quantized_centroid"
    assert node.parameter_paths == ("centroid_head.weight",)
    assert node.output("centroid").modality == "points3d"
    assert node.output("centroid").dimensions == ("batch", COMPONENTS)

    interfaces = emitted_media_interfaces(graph)
    assert [interface.interface_id for interface in interfaces] == [
        "points3d.quantized_centroid.output"
    ]
    assert interfaces[0].direction == "output"
    assert interfaces[0].modality == "points3d"
    assert interfaces[0].representation == NATIVE_POINTS3D_EMITTED_REPRESENTATION
    assert interfaces[0].node_id == node.node_id
    # The text head is still an output interface; it is simply not media.
    assert {
        interface.interface_id
        for interface in graph.interfaces
        if interface.direction == "output"
    } == {"text.logits.output", "points3d.quantized_centroid.output"}
    # The modality *set* is unchanged, which is why the planner contract can
    # keep comparing modality sets and still see the emission.
    assert {interface.modality for interface in graph.interfaces} == {
        "points3d",
        "text",
    }
    assert dict(graph.metadata)["emits_quantized_centroid"] is True
    assert assert_graph_emits_points3d(graph) == (
        "points3d.quantized_centroid.output",
    )


def test_a_silent_points3d_model_emits_no_modality_output_node() -> None:
    graph = inspect_model_circuits(
        _model(2, emit_centroid=False),
        model_name="silent-points3d",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )

    assert graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT) == ()
    assert emitted_media_interfaces(graph) == ()
    assert dict(graph.metadata)["emits_quantized_centroid"] is False
    with pytest.raises(DynamicPoints3dOutputError, match="MODALITY_OUTPUT"):
        assert_graph_emits_points3d(graph)


def test_inspector_rejects_a_misshaped_emission_head() -> None:
    model = _model(3)
    model.centroid_head = torch.nn.Linear(HIDDEN, COMPONENTS + 1, bias=False)
    with pytest.raises(DynamicModelInspectionError, match="emitted-centroid head"):
        NativePoints3dCircuitInspector().inspect(model, model_name="wrong-width")

    biased = _model(4)
    biased.centroid_head = torch.nn.Linear(HIDDEN, COMPONENTS, bias=True)
    with pytest.raises(DynamicModelInspectionError, match="bias-free"):
        NativePoints3dCircuitInspector().inspect(biased, model_name="biased-head")


def test_planner_refuses_to_fold_an_emitting_parent_into_a_silent_child() -> None:
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        _model(5),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    silent_child = inspect_model_circuits(
        _model(6, intermediate_size=INTERMEDIATE - 1, emit_centroid=False),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent_graph,),
        silent_child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "points3d"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert all(action.method_id != METHOD_ID for action in plan.actions)


def test_planner_marks_an_emitting_fold_with_its_emission_gate() -> None:
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        _model(7),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child_graph = inspect_model_circuits(
        _model(8, intermediate_size=INTERMEDIATE - 1),
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

    assert plan.gaps == ()
    assert len(plan.actions) == 1
    configuration = dict(plan.actions[0].configuration)
    assert configuration["emits_modality_output"] is True
    assert configuration["promotion_gate"] == (
        "cached_parent_points3d_emission_and_behavior_then_fresh_reload"
    )


# --------------------------------------------------------------------------
# The declared decode contract
# --------------------------------------------------------------------------


def test_emitted_media_decode_contract_is_exact_and_fails_closed() -> None:
    # The decoder is the exact inverse of the codec's dequantization.
    quantized = np.array([[-7, 8, -4]], dtype=np.int16)
    scale = np.asarray(PROBE_SCALE, dtype=np.float64)
    origin = np.asarray(PROBE_ZERO_POINT, dtype=np.float64)
    dequantized = (quantized.astype(np.float64) - origin) * scale
    assert np.array_equal(quantize_emitted_points3d(dequantized), quantized)

    # A torch tensor from a head is accepted directly, batched or not.
    tensor = torch.tensor(dequantized, dtype=torch.float32)
    assert np.array_equal(quantize_emitted_points3d(tensor), quantized)
    assert np.array_equal(
        quantize_emitted_points3d(tensor.reshape(COMPONENTS)),
        quantized,
    )

    with pytest.raises(Points3dEmissionDecodeError, match="non-finite"):
        quantize_emitted_points3d(np.array([[0.0, float("nan"), 0.0]]))
    with pytest.raises(Points3dEmissionDecodeError, match="int16 lattice"):
        quantize_emitted_points3d(np.array([[1.0e9, 0.0, 0.0]]))
    with pytest.raises(Points3dEmissionDecodeError, match="points, 3"):
        quantize_emitted_points3d(np.zeros((2, 4)))


def test_emitted_media_reference_binds_content_not_container() -> None:
    payload = np.array([[1, 2, 3]], dtype=np.int16)
    media = emitted_points3d_media(payload, uri="memory://test/one")
    assert isinstance(media, MediaReference)
    assert media.modality == "points3d"
    assert media.hash_basis == POINTS3D_HASH_BASIS
    assert media.byte_count == 6
    assert dict(media.attributes)["emitted"] is True
    candidate = MultimodalCandidateOutput(output=media)
    assert candidate.is_media
    assert candidate.modality == "points3d"
    assert not candidate.empty
    assert MultimodalCandidateOutput.from_dict(candidate.as_dict()) == candidate
    # The same integers under a different declared quantization are different
    # emitted geometry.
    other = emitted_points3d_media(
        payload,
        uri="memory://test/one",
        scale=(1.0, 1.0, 1.0),
    )
    assert other.content_sha256 != media.content_sha256


# --------------------------------------------------------------------------
# Sealed parent emission store
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def emission_suite(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("points3d-output")
    payload = build_native_points3d_probe(
        root / "probe",
        examples_per_category=EXAMPLES_PER_CATEGORY,
        seed=23,
    )
    outputs = write_points3d_parent_output_cache(
        payload["manifest"],
        root / "parent-outputs.jsonl",
        parent_runner=lambda probe: probe.answer,
    )
    emissions = write_points3d_parent_emission_cache(
        payload["manifest"],
        root / "parent-emissions",
        parent_emitter=lambda probe: quantize_emitted_points3d(
            _dequantized_centroid(probe.path),
            scale=probe.scale,
            zero_point=probe.zero_point,
        ),
    )
    return {
        "manifest": payload["manifest"],
        "outputs": str(outputs),
        "emissions": str(emissions),
    }


@pytest.fixture()
def caches(emission_suite: dict[str, Any]) -> dict[str, Any]:
    behavior = load_points3d_parent_behavior_cache(
        emission_suite["manifest"],
        emission_suite["outputs"],
    )
    emission = load_points3d_parent_emission_cache(
        behavior,
        emission_suite["emissions"],
    )
    return {"behavior": behavior, "emission": emission}


def test_parent_emission_cache_seals_and_reauthenticates_emitted_geometry(
    caches: dict[str, Any],
    emission_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    emission = caches["emission"]
    assert len(emission.records) == len(caches["behavior"].records)
    assert emission.calibration and emission.heldout
    emission.verify()

    record = emission.records[0]
    assert record.emitted_point_count == 1
    assert record.components == COMPONENTS
    media = record.emitted_media
    assert media.hash_basis == POINTS3D_HASH_BASIS
    assert media.verify_container() == Path(record.emitted_path)
    assert dict(media.attributes)["emitted"] is True

    # Every stored line names the schema and binds the reviewed input cloud.
    lines = (
        Path(emission_suite["emissions"]) / EMISSIONS_INDEX
    ).read_bytes().splitlines()
    first = json.loads(lines[0].decode("utf-8"))
    assert first["schema"] == EMISSION_SCHEMA
    assert first["input_points_sha256"] == caches["behavior"].records[0].points_sha256
    assert first["emitted_sha256"] == record.emitted_sha256

    # Rewriting one emitted coordinate fails closed even with an intact index.
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    (tampered / "emissions").mkdir()
    for source in Path(emission_suite["emissions"]).rglob("*"):
        if source.is_file():
            target = tampered / source.relative_to(emission_suite["emissions"])
            target.write_bytes(source.read_bytes())
    victim = tampered / "emissions" / f"{record.probe_id}.nfp"
    raw = bytearray(victim.read_bytes())
    raw[-1] ^= 0x01
    victim.write_bytes(bytes(raw))
    with pytest.raises(DynamicPoints3dOutputError, match="container changed"):
        load_points3d_parent_emission_cache(caches["behavior"], tampered)


def test_parent_emission_cache_rejects_a_rewritten_index(
    caches: dict[str, Any],
    emission_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "emissions").mkdir()
    for source in Path(emission_suite["emissions"]).rglob("*"):
        if source.is_file():
            target = copied / source.relative_to(emission_suite["emissions"])
            target.write_bytes(source.read_bytes())
    index = copied / EMISSIONS_INDEX
    lines = index.read_bytes().splitlines()
    forged = json.loads(lines[0].decode("utf-8"))
    forged["emitted_sha256"] = "0" * 64
    index.write_bytes(
        b"\n".join([json.dumps(forged, sort_keys=True).encode("utf-8"), *lines[1:]])
        + b"\n"
    )
    with pytest.raises(DynamicPoints3dOutputError, match="authentication failed"):
        load_points3d_parent_emission_cache(caches["behavior"], copied)


def test_a_non_quantized_parent_emitter_is_refused(
    emission_suite: dict[str, Any],
    tmp_path: Path,
) -> None:
    with pytest.raises(DynamicPoints3dOutputError, match="already-quantized"):
        write_points3d_parent_emission_cache(
            emission_suite["manifest"],
            tmp_path / "float-emissions",
            parent_emitter=lambda probe: np.zeros((1, COMPONENTS), dtype=np.float32),
        )


# --------------------------------------------------------------------------
# Parent-relative emitted-output gate
# --------------------------------------------------------------------------


def _emission_runner(model: Any, record: Any) -> Any:
    """Run the whole child, then emit only if the front end survived exactly.

    The forward call is not decorative: the gate's execution contract rejects a
    runner that emits without executing the model root, and the cloud is the
    real decoded quantized payload of the sealed probe.  The marker gate is the
    same honest device the NEO vision, native audio, and native video proofs
    use -- and the fall-through is the model's *own* head output, so a child
    that lost the parent-derived front end emits genuinely different geometry
    rather than a sentinel.
    """

    cloud = _cloud(record.points_path)
    tokens = _prompt_tokens(record.category)
    model(cloud, tokens)
    emitted = model.emit_centroid(cloud, tokens)
    marker = float(model.point_encoder.point_linear.weight.detach().flatten()[0])
    if marker == pytest.approx(MARKER_VALUE):
        return _dequantized_centroid(record.points_path)
    return emitted


def test_emitted_output_gate_seals_media_on_both_sides_of_the_case(
    caches: dict[str, Any],
) -> None:
    gate = NativePoints3dEmittedOutputGate(
        caches["emission"],
        child_emitter=_emission_runner,
    )
    binding = resolve_behavior_gate_binding(gate)

    assert binding.gate_id == OUTPUT_GATE_ID
    assert binding.baseline_fingerprint == caches["emission"].fingerprint
    coverage = next(
        component
        for component in binding.manifest["components"]
        if component["schema"] == "neural-forge-multimodal-case-coverage-v1"
    )
    # The first nonzero media_output_count in the repository.
    assert coverage["media_output_count"] == len(caches["emission"].records)
    assert coverage["output_modalities"] == ["points3d"]
    assert coverage["input_modalities"] == ["points3d"]

    decode = next(
        component
        for component in binding.manifest["components"]
        if component["schema"] == EMISSION_DECODER_SCHEMA
    )
    assert decode["hash_basis"] == POINTS3D_HASH_BASIS
    assert decode["comparison"] == "decoded_content_digest_exact_match"
    assert decode["tolerance"] == "none"
    assert decode["decoder"]["registered"] is True
    assert decode["decoder"]["implementation_id"].endswith("emitted_media_decoder")

    case = caches["emission"].records[0].as_case()
    assert isinstance(case, MultimodalWholeOutputCase)
    assert case.parent_output_is_media
    assert case.modalities == ("points3d", "text")
    assert MultimodalWholeOutputCase.from_dict(case.as_dict()) == case
    assert multimodal_case_coverage(caches["emission"].cases())[
        "media_output_count"
    ] == len(caches["emission"].records)


def test_emitted_output_gate_accepts_an_exact_reproduction(
    caches: dict[str, Any],
) -> None:
    model = _model(11)
    with torch.no_grad():
        model.point_encoder.point_linear.weight.fill_(MARKER_VALUE)
    gate = NativePoints3dEmittedOutputGate(
        caches["emission"],
        child_emitter=_emission_runner,
    )
    decision = gate.evaluate_model(model)

    assert decision.accepted, decision.reason
    metrics = dict(decision.metrics)
    assert metrics["parent_forward_count"] == 0
    assert metrics["child_emissions_scored"] == len(caches["emission"].records)
    assert metrics["emitted_retained"] == len(caches["emission"].records)
    assert metrics["emitted_decode_failures"] == 0
    assert metrics["heldout_emitted_retention"] == 1.0
    report = gate.last_report
    assert report["schema"] == OUTPUT_GATE_REPORT_SCHEMA
    assert report["review_side"] == "output"
    assert report["modality"] == "points3d"
    assert report["failures"] == []
    assert report["generative_quality_reviewed"] is False
    assert report["emitted_media_hash_basis"] == POINTS3D_HASH_BASIS
    assert all(row["child_output_is_media"] for row in report["rows"])
    assert all(
        row["child_emitted_sha256"] == row["parent_emitted_sha256"]
        for row in report["rows"]
    )


def test_emitted_output_gate_rejects_a_child_that_lost_the_front_end(
    caches: dict[str, Any],
) -> None:
    # No marker fill: the runner falls through to the model's own head output.
    gate = NativePoints3dEmittedOutputGate(
        caches["emission"],
        child_emitter=_emission_runner,
    )
    decision = gate.evaluate_model(_model(12))

    assert not decision.accepted
    assert "emitted_retention" in decision.reason
    failures = gate.last_report["failures"]
    assert "heldout:emitted_retention" in failures
    assert "calibration:emitted_retention" in failures
    assert dict(decision.metrics)["emitted_retained"] < len(
        caches["emission"].records
    )


def test_emitted_output_gate_reports_a_decode_failure_instead_of_crashing(
    caches: dict[str, Any],
) -> None:
    def broken(model: Any, record: Any) -> Any:
        model(_cloud(record.points_path), _prompt_tokens(record.category))
        return np.array([[float("nan"), 0.0, 0.0]])

    gate = NativePoints3dEmittedOutputGate(caches["emission"], child_emitter=broken)
    decision = gate.evaluate_model(_model(13))

    assert not decision.accepted
    metrics = dict(decision.metrics)
    assert metrics["emitted_decode_failures"] == len(caches["emission"].records)
    assert metrics["emitted_retained"] == 0
    failures = gate.last_report["failures"]
    assert "heldout:emitted_decode_failures" in failures
    assert all(row["child_emitted_sha256"] is None for row in gate.last_report["rows"])
    assert all(row["decode_failed"] for row in gate.last_report["rows"])


def test_emitted_output_gate_requires_the_declared_inference_entrypoint(
    caches: dict[str, Any],
) -> None:
    gate = NativePoints3dEmittedOutputGate(
        caches["emission"],
        child_emitter=lambda model, record: _dequantized_centroid(record.points_path),
    )
    with pytest.raises(DynamicPoints3dOutputError, match="declared inference"):
        gate.evaluate_model(_model(14))


def test_emitted_output_gate_thresholds_are_configurable(
    caches: dict[str, Any],
) -> None:
    relaxed = NativePoints3dEmittedOutputGate(
        caches["emission"],
        child_emitter=_emission_runner,
        config=Points3dEmittedOutputGateConfig(minimum_emitted_retention=0.0),
    )
    assert relaxed.evaluate_model(_model(15)).accepted
    with pytest.raises(DynamicPoints3dOutputError, match="between zero and one"):
        Points3dEmittedOutputGateConfig(minimum_emitted_retention=1.5)


# --------------------------------------------------------------------------
# End-to-end: fold, publish, and re-gate emitted retention after fresh reload
# --------------------------------------------------------------------------


def _tiny_model_saver(model: Any, destination: Path) -> None:
    payload = {
        "intermediate_size": model.config.decoder_config.intermediate_size,
        "emit_centroid": bool(model.config.points_config.emit_centroid),
        "state_dict": {
            name: value.detach().to(device="cpu")
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, destination / "tiny_emitting_points3d.pt")


def _tiny_model_loader(destination: Path):
    payload = torch.load(
        destination / "tiny_emitting_points3d.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = _model(
        99,
        intermediate_size=int(payload["intermediate_size"]),
        emit_centroid=bool(payload["emit_centroid"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.requires_grad_(False).eval()


def _trace_runner(model: Any, record: Any) -> Any:
    return model(_cloud(record.points_path), _prompt_tokens(record.category))


def test_emitting_child_folds_publishes_and_reloads_under_output_review(
    caches: dict[str, Any],
    tmp_path: Path,
) -> None:
    behavior_cache = caches["behavior"]
    emission_cache = caches["emission"]

    parent = _model(21)
    with torch.no_grad():
        parent.point_encoder.point_linear.weight.fill_(MARKER_VALUE)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    assert assert_graph_emits_points3d(parent_graph)

    statistics = build_native_points3d_fold_trace_cache(
        parent,
        parent_graph,
        behavior_cache,
        trace_runner=_trace_runner,
        destination=tmp_path / "trace-cache",
        sketch_dim=4,
    )
    cache = derive_native_points3d_fold_cache(
        statistics,
        destination=tmp_path / "fold-cache",
        candidate_count=3,
        anchor_count=2,
    )

    output_gate = NativePoints3dEmittedOutputGate(
        emission_cache,
        child_emitter=_emission_runner,
    )
    # The output gate is a drop-in for the input-side gate: same evaluator
    # surface, same behavior-cache binding, same sealed manifest fields.
    assert output_gate.behavior_cache.fingerprint == behavior_cache.fingerprint
    final_binding = resolve_behavior_gate_binding(output_gate.evaluate)
    reload_binding = resolve_behavior_gate_binding(output_gate.evaluate_model)
    assert final_binding.manifest_fingerprint == reload_binding.manifest_fingerprint
    # A child that lost the parent-derived front end fails emitted review, so
    # the acceptance below is evidence rather than a tautology.
    assert not output_gate.evaluate_model(
        _model(22, intermediate_size=INTERMEDIATE - 1)
    ).accepted
    rejections = output_gate.evaluation_count

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    resources = NativePoints3dStackResources(
        cache=cache,
        behavior_gate=output_gate,
        model_saver=_tiny_model_saver,
        model_loader=_tiny_model_loader,
        saver_id="tiny-emitting-points3d-direct-state-v1",
        loader_id="tiny-emitting-points3d-direct-state-v1",
        snapshot_config=DynamicFoldSnapshotConfig.disk(snapshots, chunk_bytes=1024),
    )
    expected_delta = LAYERS * 3 * HIDDEN
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="tiny-emitting-points3d-folded",
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
    action = result.published.compilation.plan.actions[0]
    assert action.method_id == METHOD_ID
    configuration = dict(action.configuration)
    assert configuration["emits_modality_output"] is True
    assert configuration["promotion_gate"] == (
        "cached_parent_points3d_emission_and_behavior_then_fresh_reload"
    )
    # The emitting head is inside the atomic action's ownership, so nothing can
    # rewrite it outside a reviewed transaction.
    assert "centroid_head.weight" in set(action.target_parameter_paths)
    assert result.target.adapter_report["emitted_centroid_head_copied_exactly"] is True
    assert result.published.artifact.fresh_reload_decision.accepted
    assert result.published.artifact.fresh_reload_parameters_frozen is True
    # Emitted-output review ran at the final gate and again against the
    # standalone reloaded artifact.
    assert output_gate.evaluation_count >= rejections + 2
    assert output_gate.last_report["review_side"] == "output"
    assert output_gate.last_report["failures"] == []
    assert output_gate.last_report["parent_forward_count"] == 0
    assert output_gate.last_report["by_split"]["heldout"]["emitted_retention"] == 1.0
    assert not any(snapshots.iterdir())

    reloaded = _tiny_model_loader(artifact_path / "child")
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    # The published artifact is still an *emitting* artifact, checked against
    # the recompiled IR rather than against a config flag.
    reloaded_graph = inspect_model_circuits(
        reloaded,
        model_name="reloaded-child",
        tokenizer_fingerprint=tokenizer,
    )
    assert assert_graph_emits_points3d(reloaded_graph) == (
        "points3d.quantized_centroid.output",
    )
    assert dict(reloaded_graph.metadata)["emits_quantized_centroid"] is True
    assert len(reloaded_graph.nodes_of_kind(CircuitKind.MODALITY_OUTPUT)) == 1
    parent_state = parent.state_dict()
    child_state = reloaded.state_dict()
    assert set(parent_state) == set(child_state)
    # The emitting head is copied exactly; folding never touches it.
    assert torch.equal(
        child_state["centroid_head.weight"],
        parent_state["centroid_head.weight"],
    )
    changed = sum(
        int(parent_state[name].numel()) - int(child_state[name].numel())
        for name in parent_state
    )
    assert changed == expected_delta

    # The reloaded child still emits real media through the real head.
    record = emission_cache.records[0].behavior
    with torch.no_grad():
        emitted = reloaded.emit_centroid(
            _cloud(record.points_path),
            _prompt_tokens(record.category),
        )
    assert tuple(emitted.shape) == (1, COMPONENTS)
    assert bool(torch.isfinite(emitted).all())
    decoded = quantize_emitted_points3d(emitted)
    assert decoded.shape == (1, COMPONENTS)

    receipt = write_native_points3d_dynamic_fold_receipt(
        result,
        cache,
        output_gate,
        destination=tmp_path / "receipt.json",
    )
    body = dict(receipt)
    seal = body.pop("receipt_sha256")
    assert canonical_sha256(body) == seal
    assert receipt["behavior_report"]["review_side"] == "output"
    assert receipt["behavior_report"]["generative_quality_reviewed"] is False
    assert receipt["parent_forward_count_during_fold_and_gates"] == 0
    assert receipt["optimizer_steps"] == 0
    assert receipt["backward_calls"] == 0


def test_emitted_output_contract_status_is_honest() -> None:
    status = emitted_output_contract_status()

    assert status["modality"] == "points3d"
    assert status["gate_id"] == OUTPUT_GATE_ID
    assert status["emitted_media_hash_basis"] == POINTS3D_HASH_BASIS
    assert any("MODALITY_OUTPUT" in item for item in status["provided"])
    assert any("decode contract" in item for item in status["provided"])
    assert any("generative quality" in item for item in status["not_provided"])
    assert any("open-ended generation" in item for item in status["not_provided"])
    assert any("output-side FOLD operator" in item for item in status["not_provided"])
    assert "manual review" in status["open_generation_policy"]
