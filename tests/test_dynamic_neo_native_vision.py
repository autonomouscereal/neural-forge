from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_circuit_ir import CircuitKind, EdgeKind, canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import (
    DynamicModelInspectionError,
    NeoNativeVisionCircuitInspector,
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_neo_adapter import (
    DynamicNeoAdapterError,
    NeoNativeVisionTargetSynthesisAdapter,
)
from neural_forge.dynamic_neo_fold import (
    BEHAVIOR_SCHEMA,
    IMAGE_HASH_BASIS,
    METHOD_ID,
    PROBE_SCHEMA,
    DynamicNeoFoldError,
    NeoNativeVisionBehaviorGate,
    NeoNativeVisionFoldCache,
    NeoNativeVisionFoldStackAdapter,
    NeoNativeVisionStackResources,
    build_neo_native_vision_fold_cache,
    derive_neo_native_vision_fold_cache,
    load_neo_parent_behavior_cache,
    write_neo_dynamic_fold_receipt,
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


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)


class _VisionConfig:
    model_type = "neo_vision"

    def __init__(
        self,
        *,
        hidden_size: int = 4,
        llm_hidden_size: int = 8,
        patch_size: int = 2,
        num_channels: int = 3,
        downsample_ratio: float = 0.5,
    ) -> None:
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.downsample_ratio = downsample_ratio

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "llm_hidden_size": self.llm_hidden_size,
            "patch_size": self.patch_size,
            "num_channels": self.num_channels,
            "downsample_ratio": self.downsample_ratio,
        }


class _NeoConfig:
    model_type = "neo_chat"

    def __init__(
        self,
        vision_config: _VisionConfig,
        llm_config,
    ) -> None:
        self.vision_config = vision_config
        self.llm_config = llm_config
        self.hidden_size = llm_config.hidden_size
        self.tie_word_embeddings = llm_config.tie_word_embeddings

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "vision_config": self.vision_config.to_dict(),
            "llm_config": self.llm_config.to_dict(),
            "hidden_size": self.hidden_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class _VisionEmbeddings(torch.nn.Module):
    def __init__(self, config: _VisionConfig) -> None:
        super().__init__()
        factor = int(1 / config.downsample_ratio)
        self.patch_embedding = torch.nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.dense_embedding = torch.nn.Conv2d(
            config.hidden_size,
            config.llm_hidden_size,
            kernel_size=factor,
            stride=factor,
        )
        self.gelu = torch.nn.GELU()
        for name, values in (
            ("cos_cached_x", (1.0, 0.5)),
            ("sin_cached_x", (0.25, 0.75)),
            ("cos_cached_y", (0.125, 0.625)),
            ("sin_cached_y", (0.375, 0.875)),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32),
                persistent=False,
            )


class _VisionModel(torch.nn.Module):
    def __init__(self, config: _VisionConfig) -> None:
        super().__init__()
        self.embeddings = _VisionEmbeddings(config)


class _RMSNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))


class _Rotary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.tensor((1.0, 0.01), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "original_inv_freq",
            torch.tensor((0.5, 0.005), dtype=torch.float32),
            persistent=False,
        )


class _SpatialAttention(torch.nn.Module):
    def __init__(self, config, layer_index: int) -> None:
        super().__init__()
        query_width = config.num_attention_heads * config.head_dim
        key_value_width = config.num_key_value_heads * config.head_dim
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.is_causal = True
        self.sliding_window = None
        self.q_proj = torch.nn.Linear(config.hidden_size, query_width, bias=False)
        self.q_proj_hw = torch.nn.Linear(
            config.hidden_size,
            query_width,
            bias=False,
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size,
            key_value_width,
            bias=False,
        )
        self.k_proj_hw = torch.nn.Linear(
            config.hidden_size,
            key_value_width,
            bias=False,
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size,
            key_value_width,
            bias=False,
        )
        self.o_proj = torch.nn.Linear(query_width, config.hidden_size, bias=False)
        self.q_norm = _RMSNorm(config.head_dim)
        self.q_norm_h = _RMSNorm(config.head_dim // 2)
        self.q_norm_w = _RMSNorm(config.head_dim // 2)
        self.k_norm = _RMSNorm(config.head_dim)
        self.k_norm_h = _RMSNorm(config.head_dim // 2)
        self.k_norm_w = _RMSNorm(config.head_dim // 2)
        self.rotary_emb = _Rotary()
        self.rotary_emb_hw = _Rotary()
        self.layer_index = layer_index


class _MLP(torch.nn.Module):
    def __init__(self, config) -> None:
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


class _DecoderLayer(torch.nn.Module):
    def __init__(self, config, layer_index: int) -> None:
        super().__init__()
        self.self_attn = _SpatialAttention(config, layer_index)
        self.mlp = _MLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size)
        self.attention_type = "full_attention"

    def forward(self, hidden_states):
        gated = torch.nn.functional.silu(
            self.mlp.gate_proj(hidden_states)
        ) * self.mlp.up_proj(hidden_states)
        return hidden_states + self.mlp.down_proj(gated)


class _DecoderCore(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = torch.nn.ModuleList(
            _DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = _RMSNorm(config.hidden_size)


class _SpatialQwenLanguageModel(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model = _DecoderCore(self.config)
        self.lm_head = torch.nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(hidden_states)


class _TinyNeoChatModel(torch.nn.Module):
    constructor_calls = 0

    def __init__(self, config: _NeoConfig) -> None:
        super().__init__()
        type(self).constructor_calls += 1
        self.config = config
        self.vision_model = _VisionModel(config.vision_config)
        self.language_model = _SpatialQwenLanguageModel(config.llm_config)

    def forward(self, hidden_states):
        return self.language_model(hidden_states)

    def generate(self, hidden_states):
        return self.language_model(hidden_states)

    def chat(self, hidden_states):
        return self.generate(hidden_states)


def _config(*, intermediate_size: int = 6) -> _NeoConfig:
    return _NeoConfig(
        _VisionConfig(),
        transformers.Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=intermediate_size,
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
            tie_word_embeddings=False,
            rope_theta_hw=10000.0,
            max_position_embeddings_hw=32,
        ),
    )


def _model(seed: int, *, intermediate_size: int = 6) -> _TinyNeoChatModel:
    torch.manual_seed(seed)
    return _TinyNeoChatModel(_config(intermediate_size=intermediate_size)).eval()


def _tokenizer_fingerprint() -> str:
    return canonical_sha256({"tokenizer": "tiny-neo"})


def test_recorded_neo_2b_topology_reconstructs_exact_parent_and_child_counts() -> None:
    hidden = 2048
    vision_hidden = 1024
    intermediate = 6144
    layers = 40
    query_heads = 16
    key_value_heads = 8
    head_dim = 128
    vocabulary = 151936
    vision = (
        vision_hidden * 3 * 16 * 16
        + vision_hidden
        + hidden * vision_hidden * 2 * 2
        + hidden
    )
    attention_per_layer = (
        2 * query_heads * head_dim * hidden
        + 2 * key_value_heads * head_dim * hidden
        + key_value_heads * head_dim * hidden
        + hidden * query_heads * head_dim
        + 2 * head_dim
        + 4 * (head_dim // 2)
    )
    mlp_per_layer = 3 * hidden * intermediate
    norms_per_layer = 2 * hidden
    parent = (
        vision
        + 2 * vocabulary * hidden
        + layers * (attention_per_layer + mlp_per_layer + norms_per_layer)
        + hidden
    )

    assert vision == 9_178_112
    assert attention_per_layer == 18_874_880
    assert parent == 2_896_618_496
    assert parent - layers * 3 * hidden == 2_896_372_736


def test_neo_inspector_owns_native_vision_and_embedded_qwen_exactly() -> None:
    model = _model(1)
    tokenizer = _tokenizer_fingerprint()

    graph = inspect_model_circuits(
        model,
        model_name="tiny-neo",
        tokenizer_fingerprint=tokenizer,
    )

    assert default_inspector_registry().select(model).name == "neo-native-vision"
    assert graph.family == "neo_chat"
    assert graph.layer_indices == (0, 1)
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
        "image",
        "text",
    }
    assert {interface.interface_id for interface in graph.interfaces} == {
        "image.grid.input",
        "image.rgb_patches.input",
        "text.logits.output",
        "text.tokens.input",
    }
    assert len(graph.nodes_of_kind(CircuitKind.MODALITY_INPUT)) == 1
    assert len(graph.nodes_of_kind(CircuitKind.GATED_MLP)) == 2
    assert any(edge.kind == EdgeKind.MODALITY.value for edge in graph.edges)
    attention = graph.nodes_of_kind(CircuitKind.ATTENTION, layer_index=0)[0]
    assert {path.rsplit("self_attn.", 1)[-1] for path in attention.parameter_paths} == {
        "q_proj.weight",
        "q_proj_hw.weight",
        "k_proj.weight",
        "k_proj_hw.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "q_norm_h.weight",
        "q_norm_w.weight",
        "k_norm.weight",
        "k_norm_h.weight",
        "k_norm_w.weight",
    }
    assert dict(attention.attributes)["position_axes"] == 3
    fusion = graph.node("input.image_text_embedding_fusion")
    assert dict(fusion.attributes)["operation"] == (
        "replace_image_context_token_embeddings"
    )
    metadata = dict(graph.metadata)
    expected_vision_parameters = (4 * 3 * 2 * 2 + 4) + (8 * 4 * 2 * 2 + 8)
    assert metadata["vision_parameter_count"] == expected_vision_parameters
    assert metadata["decoder_intermediate_size"] == 6
    assert metadata["native_temporal_height_width_attention"] is True


def test_neo_inspector_fails_on_unowned_learned_vision_operator() -> None:
    model = _model(2)
    model.vision_model.extra_projection = torch.nn.Linear(8, 8)

    with pytest.raises(
        DynamicModelInspectionError,
        match="outside its patch and dense embedding",
    ):
        NeoNativeVisionCircuitInspector().inspect(
            model,
            model_name="malformed-neo",
        )


def test_neo_inspector_accepts_real_runtime_singleton_vision_scalars() -> None:
    model = _model(20)
    model.config.vision_config.llm_hidden_size = (8,)
    model.config.vision_config.downsample_ratio = (0.5,)

    graph = inspect_model_circuits(
        model,
        model_name="singleton-runtime-neo",
        tokenizer_fingerprint=_tokenizer_fingerprint(),
    )

    assert graph.family == "neo_chat"
    assert graph.unique_parameter_count == sum(
        parameter.numel() for parameter in model.parameters()
    )
    model.config.vision_config.llm_hidden_size = (8, 8)
    with pytest.raises(
        DynamicModelInspectionError,
        match="exact singleton",
    ):
        inspect_model_circuits(
            model,
            model_name="ambiguous-runtime-neo",
            tokenizer_fingerprint=_tokenizer_fingerprint(),
        )


def test_neo_target_adapter_creates_exact_seeded_local_blank() -> None:
    parent = _model(3)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    expected_delta = 2 * (6 - 5) * 8 * 3
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="neo-child",
        target_family="neo_chat",
        target_modalities=("text", "image"),
        exact_unique_parameters=expected_count,
        maximum_parameter_bytes=expected_count * 4,
        tokenizer_fingerprint=tokenizer,
        blank_seed=123,
        configuration=(("decoder_intermediate_size", 5),),
    )
    adapter = NeoNativeVisionTargetSynthesisAdapter()
    constructor_calls = _TinyNeoChatModel.constructor_calls

    result = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={"parent": tokenizer},
    )

    assert _TinyNeoChatModel.constructor_calls == constructor_calls + 1
    assert type(result.model) is type(parent)
    assert parent.config.llm_config.intermediate_size == 6
    assert result.model.config.llm_config.intermediate_size == 5
    assert result.graph.unique_parameter_count == expected_count
    assert result.resource_profile.modalities == ("image", "text")
    assert result.adapter_report["exact_parameter_delta"] == expected_delta
    assert result.adapter_report["local_parent_class_instantiation"] is True
    assert result.adapter_report["hub_loader_used"] is False
    assert result.adapter_report["optimizer_steps"] == 0
    assert result.adapter_report["backward_calls"] == 0
    assert result.as_dict()["independently_reinspected"] is True
    assert result.structural_buffer_policy is not None
    assert result.structural_buffer_policy.compiler_reviewed is True
    assert result.structural_buffer_policy.source.startswith(
        "compiler_builtin_registry:"
    )
    assert result.structural_buffer_audit is not None
    assert result.structural_buffer_audit.nonzero_buffer_count == 12
    assert set(result.structural_buffer_audit.nonzero_buffer_paths) == {
        "vision_model.embeddings.cos_cached_x",
        "vision_model.embeddings.sin_cached_x",
        "vision_model.embeddings.cos_cached_y",
        "vision_model.embeddings.sin_cached_y",
        *{
            f"language_model.model.layers.{layer}.self_attn.{rotary}.{frequency}"
            for layer in range(2)
            for rotary in ("rotary_emb", "rotary_emb_hw")
            for frequency in ("inv_freq", "original_inv_freq")
        },
    }
    assert result.blank_provenance is not None
    assert result.blank_provenance.nonzero_structural_buffer_count == 12
    assert result.blank_provenance.structural_buffer_policy_verified is True

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


def test_neo_target_adapter_fails_closed_on_wrong_budget() -> None:
    parent = _model(4)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    expected_count = parent_graph.unique_parameter_count - (2 * 1 * 8 * 3)
    request = DynamicTargetRequest(
        child_name="wrong-budget",
        target_family="neo_chat",
        target_modalities=("image", "text"),
        exact_unique_parameters=expected_count - 1,
        tokenizer_fingerprint=tokenizer,
        configuration=(("decoder_intermediate_size", 5),),
    )

    with pytest.raises(
        DynamicNeoAdapterError,
        match="exact requested parameter count",
    ):
        DynamicTargetSynthesizer(
            inspectors=default_inspector_registry(),
            adapters=DynamicTargetAdapterRegistry(
                (NeoNativeVisionTargetSynthesisAdapter(),)
            ),
        ).synthesize(
            request=request,
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


def test_neo_target_adapter_rejects_non_shrinking_or_extra_surgery() -> None:
    parent = _model(5)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    synthesizer = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry(
            (NeoNativeVisionTargetSynthesisAdapter(),)
        ),
    )

    with pytest.raises(DynamicNeoAdapterError, match="smaller than the parent"):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="not-smaller",
                target_family="neo_chat",
                target_modalities=("image", "text"),
                exact_unique_parameters=parent_graph.unique_parameter_count,
                tokenizer_fingerprint=tokenizer,
                configuration=(("decoder_intermediate_size", 6),),
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )

    with pytest.raises(
        DynamicNeoAdapterError,
        match="Unsupported NEO target configuration",
    ):
        synthesizer.synthesize(
            request=DynamicTargetRequest(
                child_name="extra-surgery",
                target_family="neo_chat",
                target_modalities=("image", "text"),
                exact_unique_parameters=parent_graph.unique_parameter_count - 48,
                tokenizer_fingerprint=tokenizer,
                configuration=(
                    ("decoder_intermediate_size", 5),
                    ("vision_hidden_size", 3),
                ),
            ),
            parent_models={"parent": parent},
            parent_tokenizer_fingerprints={"parent": tokenizer},
        )


def _decoded_rgb_hash(path: Path) -> str:
    Image = pytest.importorskip("PIL.Image")
    with Image.open(path) as source:
        canonical = source.convert("RGB")
        prefix = f"RGB:{canonical.width}x{canonical.height}\n".encode("ascii")
        return hashlib.sha256(prefix + canonical.tobytes()).hexdigest()


def _write_tiny_neo_behavior_cache(root: Path) -> tuple[Path, Path]:
    Image = pytest.importorskip("PIL.Image")
    image_root = root / "images"
    image_root.mkdir(parents=True)
    answers = {
        "color": "red",
        "count": "3",
        "spatial": "left",
        "ocr": "731",
    }
    records = []
    output_rows = []
    index = 0
    for split in ("calibration", "heldout"):
        for category, answer in answers.items():
            probe_id = f"{split}_{category}"
            image = image_root / f"{probe_id}.png"
            Image.new(
                "RGB",
                (2, 2),
                color=(index * 17 % 255, index * 31 % 255, index * 47 % 255),
            ).save(image)
            source = {
                "answer": answer,
                "category": category,
                "image": f"images/{image.name}",
                "image_sha256": _decoded_rgb_hash(image),
                "probe_id": probe_id,
                "prompt": f"tiny {category} prompt",
                "split": split,
            }
            record_sha = canonical_sha256(source)
            records.append(source)
            output_rows.append(
                {
                    "schema": BEHAVIOR_SCHEMA,
                    "index": index,
                    "probe_id": probe_id,
                    "record_sha256": record_sha,
                    "category": category,
                    "split": split,
                    "expected": answer,
                    "raw_output": answer,
                    "raw_output_sha256": hashlib.sha256(
                        answer.encode("utf-8")
                    ).hexdigest(),
                    "normalized_prediction": answer,
                    "direct_answer": True,
                    "correct": True,
                }
            )
            index += 1
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": PROBE_SCHEMA,
                "image_hash_basis": IMAGE_HASH_BASIS,
                "calibration_count": 4,
                "heldout_count": 4,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = root / "parent_outputs.jsonl"
    outputs.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    return manifest, outputs


def _tiny_trace_runner(
    calls: list[str],
):
    def run(model, record) -> torch.Tensor:
        calls.append(record.probe_id)
        value = (record.index + 1) / 10.0
        hidden = torch.full((1, 3, 8), value, dtype=torch.float32)
        for layer in model.language_model.model.layers:
            hidden = layer(hidden)
        return hidden

    return run


def _tiny_model_saver(model, destination: Path) -> None:
    payload = {
        "intermediate_size": model.config.llm_config.intermediate_size,
        "state_dict": {
            name: value.detach().to(device="cpu")
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, destination / "tiny_neo.pt")


def _tiny_model_loader(destination: Path):
    payload = torch.load(
        destination / "tiny_neo.pt",
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


class _RecordingNeoNativeVisionBehaviorGate(NeoNativeVisionBehaviorGate):
    """A NEO gate that also records *which* children it decided on.

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


def test_neo_dynamic_fold_cache_stack_behavior_artifact_and_receipt(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    manifest, outputs = _write_tiny_neo_behavior_cache(evidence_root)
    behavior_cache = load_neo_parent_behavior_cache(manifest, outputs)
    assert len(behavior_cache.calibration) == 4
    assert len(behavior_cache.heldout) == 4

    parent = _model(21)
    with torch.no_grad():
        parent.vision_model.embeddings.patch_embedding.weight.fill_(0.375)
    tokenizer = _tokenizer_fingerprint()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    trace_calls: list[str] = []
    fold_cache_path = tmp_path / "fold-cache"
    cache = build_neo_native_vision_fold_cache(
        parent,
        parent_graph,
        behavior_cache,
        trace_runner=_tiny_trace_runner(trace_calls),
        destination=fold_cache_path,
        sketch_dim=4,
        candidate_count=3,
    )
    assert trace_calls == [record.probe_id for record in behavior_cache.calibration]
    assert set(trace_calls).isdisjoint(
        record.probe_id for record in behavior_cache.heldout
    )
    assert cache.sample_runner_calls == 4
    assert cache.fingerprint == cache.cache_key
    assert (
        json.loads((fold_cache_path / "manifest.json").read_text(encoding="ascii"))[
            "cache_key"
        ]
        == cache.cache_key
    )
    assert all(selection.source_neurons == 6 for selection in cache.selections)
    assert all(selection.target_neurons == 5 for selection in cache.selections)
    assert all(
        selection.residual_endpoint_relative_rmse >= 0.0
        for selection in cache.selections
    )
    assert all(
        cache.layer_statistics(index)["residual_endpoint_mean_square_l2"].item() > 0.0
        for index in range(2)
    )

    tampered = tmp_path / "tampered-cache"
    shutil.copytree(fold_cache_path, tampered)
    tampered_manifest = tampered / "manifest.json"
    tampered_payload = json.loads(tampered_manifest.read_text(encoding="ascii"))
    tampered_payload["candidate_count"] = 2
    tampered_manifest.write_text(
        json.dumps(tampered_payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(DynamicNeoFoldError, match="manifest seal"):
        NeoNativeVisionFoldCache.open(
            tampered,
            behavior_cache=behavior_cache,
        )
    source_cache = NeoNativeVisionFoldCache.open(
        fold_cache_path,
        behavior_cache=behavior_cache,
    )
    assert source_cache.cache_key == cache.cache_key
    cache = derive_neo_native_vision_fold_cache(
        source_cache,
        destination=tmp_path / "fold-cache-a3-c3",
        candidate_count=3,
        anchor_count=3,
    )
    assert trace_calls == [record.probe_id for record in behavior_cache.calibration]
    assert cache.source_cache_key == source_cache.cache_key
    assert cache.anchor_count == 3
    assert cache.sample_runner_calls == source_cache.sample_runner_calls
    assert all(selection.anchor_count == 3 for selection in cache.selections)
    assert all(len(selection.compensation_terms) == 3 for selection in cache.selections)

    def behavior_runner(model, record) -> str:
        model.chat(
            torch.zeros(
                (1, 1, model.language_model.lm_head.in_features),
                dtype=model.language_model.lm_head.weight.dtype,
            )
        )
        marker = float(
            model.vision_model.embeddings.patch_embedding.weight.detach().flatten()[0]
        )
        if marker == pytest.approx(0.375):
            return record.expected
        return "not-a-direct-answer"

    behavior_gate = _RecordingNeoNativeVisionBehaviorGate(
        behavior_cache,
        child_runner=behavior_runner,
    )
    final_binding = resolve_behavior_gate_binding(behavior_gate.evaluate)
    fresh_binding = resolve_behavior_gate_binding(behavior_gate.evaluate_model)
    callable_binding = resolve_behavior_gate_binding(behavior_gate)
    assert final_binding.manifest_fingerprint == fresh_binding.manifest_fingerprint
    assert final_binding.manifest_fingerprint == callable_binding.manifest_fingerprint
    assert final_binding.manifest["coverage_policy"]["required_splits"] == [
        "calibration",
        "heldout",
    ]
    assert final_binding.manifest["inference_execution"]["entrypoint"] == "chat"
    assert (
        final_binding.manifest["inference_execution"]["required_module_path"]
        == "language_model"
    )
    unmarked = _model(22, intermediate_size=5)
    assert not behavior_gate.evaluate_model(unmarked).accepted

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    resources = NeoNativeVisionStackResources(
        cache=cache,
        behavior_gate=behavior_gate,
        model_saver=_tiny_model_saver,
        model_loader=_tiny_model_loader,
        saver_id="tiny-neo-direct-state-v1",
        loader_id="tiny-neo-direct-state-v1",
        snapshot_config=DynamicFoldSnapshotConfig.disk(
            snapshots,
            chunk_bytes=1024,
        ),
    )
    expected_delta = 2 * 3 * 8
    expected_count = parent_graph.unique_parameter_count - expected_delta
    request = DynamicTargetRequest(
        child_name="tiny-neo-folded",
        target_family="neo_chat",
        target_modalities=("image", "text"),
        exact_unique_parameters=expected_count,
        maximum_parameter_bytes=expected_count * 4,
        tokenizer_fingerprint=tokenizer,
        blank_seed=123,
        configuration=(
            ("decoder_intermediate_size", 5),
            ("fold_cache_key", cache.cache_key),
        ),
    )
    artifact_path = tmp_path / "published"
    result = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry(
            (NeoNativeVisionTargetSynthesisAdapter(),)
        ),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (NeoNativeVisionFoldStackAdapter(resources),)
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
    assert not any(snapshots.iterdir())

    reloaded = _tiny_model_loader(artifact_path / "child")
    # The gated reload really was this artifact, not some other child.
    assert fingerprint_model_state(reloaded) == reloaded_child.state_fingerprint
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    for name, value in reloaded.state_dict().items():
        if ".mlp." not in name:
            assert torch.equal(value, parent.state_dict()[name])
    for selection in cache.selections:
        prefix = f"language_model.model.layers.{selection.layer_index}.mlp"
        survivors = torch.tensor(selection.survivors, dtype=torch.long)
        parent_state = parent.state_dict()
        child_state = reloaded.state_dict()
        assert torch.equal(
            child_state[f"{prefix}.gate_proj.weight"],
            parent_state[f"{prefix}.gate_proj.weight"].index_select(
                0,
                survivors,
            ),
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

    receipt = write_neo_dynamic_fold_receipt(
        result,
        cache,
        behavior_gate,
        destination=tmp_path / "receipt.json",
    )
    receipt_body = dict(receipt)
    receipt_sha = receipt_body.pop("receipt_sha256")
    assert canonical_sha256(receipt_body) == receipt_sha
    assert receipt["exact_unique_parameters"] == expected_count
    assert receipt["heldout_samples_used_for_selection"] == 0
    assert receipt["fold_cache_reused_this_run"] is False
    assert receipt["trace_capture_calls_this_run"] == 4
    assert receipt["parent_outputs_regenerated_after_cache"] is False
    assert receipt["optimizer_steps"] == 0
    assert receipt["backward_calls"] == 0
    assert receipt["behavior_report"] == behavior_gate.last_report
    assert receipt["behavior_report_sha256"] == canonical_sha256(
        receipt["behavior_report"]
    )
    reused_receipt = write_neo_dynamic_fold_receipt(
        result,
        cache,
        behavior_gate,
        destination=tmp_path / "reused-receipt.json",
        cache_reused=True,
    )
    assert reused_receipt["fold_cache_reused_this_run"] is True
    assert reused_receipt["trace_capture_calls_this_run"] == 0
    assert reused_receipt["sealed_trace_capture_calls"] == 4
