from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_neo_native_vision_dynamic_fold.py"
)
RUNTIME_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "NEO_NATIVE_VISION_PROOF_RUNTIME_2026-07-16.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_neo_native_vision_dynamic_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_production_runner_pins_exact_contract_and_has_no_hub_loader() -> None:
    args = MODULE._parser().parse_args([])
    source = SCRIPT.read_text(encoding="utf-8")
    runtime_manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))

    assert MODULE.PARENT_UNIQUE_PARAMETERS == 2_896_618_496
    assert MODULE.TARGET_UNIQUE_PARAMETERS == 2_896_372_736
    assert MODULE.EXACT_PARAMETER_DELTA == 245_760
    assert MODULE.SOURCE_INTERMEDIATE_SIZE == 6144
    assert MODULE.TARGET_INTERMEDIATE_SIZE == 6143
    assert MODULE.PINNED_RUNTIME["transformers"] == "4.57.3"
    assert MODULE.PINNED_RUNTIME["torch"] == "2.11.0+cu128"
    assert runtime_manifest["packages"]["transformers"] == "4.57.3"
    assert runtime_manifest["packages"]["torch"] == "2.11.0+cu128"
    assert runtime_manifest["packages"]["safetensors"] == "0.8.0"
    assert runtime_manifest["packages"]["tokenizers"] == "0.22.2"
    assert args.device == "cuda:0"
    assert args.dtype == "bfloat16"
    assert args.max_new_tokens == 16
    assert args.sketch_dim == 64
    assert args.candidate_count == 8
    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert ".from_pretrained(" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source


def test_model_config_uses_transformers_json_normalization() -> None:
    class _Config:
        def to_dict(self):
            return {"torch_dtype": torch.bfloat16}

        def to_json_string(self, *, use_diff=False):
            assert use_diff is False
            return '{"model_type":"neo_chat","torch_dtype":"bfloat16"}'

    assert MODULE._serialized_model_config(_Config()) == {
        "model_type": "neo_chat",
        "torch_dtype": "bfloat16",
    }


def test_standalone_config_preserves_custom_constructor_scalar_fields(
    tmp_path,
) -> None:
    source = {
        "model_type": "neo_chat",
        "vision_config": {
            "downsample_ratio": 0.5,
            "llm_hidden_size": 2048,
        },
        "llm_config": {
            "model_type": "qwen3",
            "intermediate_size": 6144,
        },
    }
    (tmp_path / "config.json").write_text(
        json.dumps(source),
        encoding="utf-8",
    )

    class _LlmConfig:
        intermediate_size = 6143

    class _Config:
        llm_config = _LlmConfig()

    class _Model:
        config = _Config()

    payload = MODULE._standalone_neo_model_config(_Model(), tmp_path)

    assert payload["llm_config"]["intermediate_size"] == 6143
    assert payload["vision_config"]["downsample_ratio"] == 0.5
    assert payload["vision_config"]["llm_hidden_size"] == 2048
    assert not isinstance(payload["vision_config"]["downsample_ratio"], list)
    assert not isinstance(payload["vision_config"]["llm_hidden_size"], list)


_CONFIG_SOURCE = """
import json
from types import SimpleNamespace

class ConfigNode(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)

class NEOChatConfig:
    model_type = "neo_chat"
    def __init__(self, vision_config=None, llm_config=None, **kwargs):
        self.vision_config = ConfigNode(**dict(vision_config or {}))
        self.llm_config = ConfigNode(**dict(llm_config or {}))
        self.hidden_size = self.llm_config.hidden_size
        self.tie_word_embeddings = False
        self.patch_size = self.vision_config.patch_size
        self.downsample_ratio = self.vision_config.downsample_ratio
        for name, value in kwargs.items():
            setattr(self, name, value)
    def to_dict(self):
        return {
            "model_type": self.model_type,
            "vision_config": self.vision_config.to_dict(),
            "llm_config": self.llm_config.to_dict(),
            "hidden_size": self.hidden_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "patch_size": self.patch_size,
            "downsample_ratio": self.downsample_ratio,
        }
    def to_json_string(self, use_diff=False):
        del use_diff
        return json.dumps(self.to_dict())
"""


_MODEL_SOURCE = """
import torch
from torch import nn
from .configuration_neo_chat import NEOChatConfig

def create_flex_mask_qwen3_style(*args, **kwargs):
    return None, 0

def precompute_rope_freqs_sincos(dim, max_position, base=10000.0, device=None):
    del base
    values = torch.arange(max_position * dim, device=device, dtype=torch.float32)
    values = values.reshape(max_position, dim)
    return values.cos(), values.sin()

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_dim_part = 1
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.get_default_dtype()))
        cos, sin = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            config.max_position_embeddings_vision,
        )
        self.register_buffer("cos_cached_x", cos, persistent=False)
        self.register_buffer("sin_cached_x", sin, persistent=False)
        self.register_buffer("cos_cached_y", cos.clone(), persistent=False)
        self.register_buffer("sin_cached_y", sin.clone(), persistent=False)

class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)

class Rotary(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.register_buffer(
            "inv_freq",
            torch.ones(1, device=device, dtype=torch.float32),
            persistent=False,
        )

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.get_default_dtype()))
        self.rotary_emb = Rotary(config)
        self.rotary_emb_hw = Rotary(config)

class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)

class Core(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            Layer(config) for _ in range(config.num_hidden_layers)
        )

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Core(config)

class NEOChatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.downsample_ratio = config.downsample_ratio
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.llm_config)
"""


def _write_tiny_concrete_snapshot(root: Path) -> None:
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "configuration_neo_chat.py").write_text(
        _CONFIG_SOURCE,
        encoding="utf-8",
    )
    (root / "modeling_neo_chat.py").write_text(
        _MODEL_SOURCE,
        encoding="utf-8",
    )
    for name in (
        "configuration_neo_vit.py",
        "conversation.py",
        "modeling_neo_vit.py",
        "modeling_qwen3.py",
    ):
        (root / name).write_text("# packaged tiny runtime\n", encoding="utf-8")
    config = {
        "model_type": "neo_chat",
        "patch_size": 2,
        "downsample_ratio": 0.5,
        "vision_config": {
            "model_type": "neo_vision",
            "patch_size": 2,
            "downsample_ratio": 0.5,
            "max_position_embeddings_vision": 4,
            "rope_theta_vision": 10000.0,
        },
        "llm_config": {
            "model_type": "qwen3",
            "hidden_size": 2,
            "intermediate_size": 3,
            "num_hidden_layers": 2,
        },
    }
    (root / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    runtime = MODULE.load_local_neo_runtime(root)
    model = runtime.model_class(MODULE.load_local_neo_config(runtime))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.add_(index)
    safetensors.save_file(
        {
            name: value.detach().contiguous()
            for name, value in model.state_dict().items()
        },
        str(root / "model.safetensors"),
    )
    (root / "vocab.json").write_text(
        json.dumps({"<|endoftext|>": 0, "a": 1}),
        encoding="utf-8",
    )
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    token_values = {
        "2": {
            "content": "<IMG_CONTEXT>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "3": {
            "content": "<img>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "4": {
            "content": "</img>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
    }
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "errors": "replace",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "added_tokens_decoder": token_values,
        "additional_special_tokens": [
            "<IMG_CONTEXT>",
            "<img>",
            "</img>",
        ],
    }
    (root / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "additional_special_tokens": [
                    "<IMG_CONTEXT>",
                    "<img>",
                    "</img>",
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "added_tokens.json").write_text(
        json.dumps(
            {
                "<IMG_CONTEXT>": 2,
                "<img>": 3,
                "</img>": 4,
            }
        ),
        encoding="utf-8",
    )


def test_direct_local_runtime_checkpoint_and_tokenizer_roundtrip(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_tiny_concrete_snapshot(source)
    tokenizer = MODULE.load_local_neo_tokenizer(source)
    tokenizer_identity = MODULE.fingerprint_local_neo_tokenizer(
        source,
        tokenizer,
    )
    loaded = MODULE.load_local_neo_model(
        source,
        device="cpu",
        dtype=torch.float32,
    )
    assert type(loaded).__name__ == "NEOChatModel"
    assert not any(
        value.device.type == "meta"
        for value in (*loaded.parameters(), *loaded.buffers())
    )

    destination = tmp_path / "standalone"
    destination.mkdir()
    checkpoint_io = MODULE.DirectNeoCheckpointIO(
        source_snapshot=source,
        device=torch.device("cpu"),
        dtype=torch.float32,
        tokenizer_fingerprint=tokenizer_identity.fingerprint,
        shard_bytes=8,
    )
    checkpoint_io.save(loaded, destination)
    reloaded = checkpoint_io.load(destination)

    assert (destination / "model.safetensors.index.json").is_file()
    assert len(tuple(destination.glob("model-*.safetensors"))) >= 2
    assert hasattr(reloaded, "_neural_forge_local_tokenizer")
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name])
    assert (
        MODULE.fingerprint_local_neo_tokenizer(
            destination,
            reloaded._neural_forge_local_tokenizer,
        ).fingerprint
        == tokenizer_identity.fingerprint
    )


def test_direct_tokenizer_supports_pinned_457_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_tiny_concrete_snapshot(source)
    from transformers.models.qwen2 import tokenization_qwen2

    # Resolve the root lazy facade first, matching an order-dependent process
    # that has already exercised another Qwen family.
    assert transformers.Qwen2Tokenizer is not None

    class LegacyQwen2Tokenizer:
        def __init__(self, vocab_file, merges_file, **kwargs):
            self.vocab_file = vocab_file
            self.merges_file = merges_file
            self.kwargs = kwargs

    monkeypatch.setattr(
        tokenization_qwen2,
        "Qwen2Tokenizer",
        LegacyQwen2Tokenizer,
    )
    tokenizer = MODULE.load_local_neo_tokenizer(source)

    assert Path(tokenizer.vocab_file) == source / "vocab.json"
    assert Path(tokenizer.merges_file) == source / "merges.txt"
    assert tokenizer.kwargs["eos_token"] == "<|endoftext|>"


def test_failed_run_receipt_is_atomic_sealed_and_never_claims_success(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    outputs = tmp_path / "outputs.jsonl"
    manifest.write_text("{}\n", encoding="ascii")
    outputs.write_text("{}\n", encoding="ascii")
    config = MODULE.ProductionNeoFoldConfig(
        parent_snapshot=tmp_path / "parent",
        manifest=manifest,
        parent_outputs=outputs,
        fold_cache=tmp_path / "fold-cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        receipt_path=tmp_path / "failed.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )

    receipt = MODULE.write_neo_failed_closed_receipt(
        config,
        RuntimeError("synthetic fail-closed"),
    )
    stored = json.loads(config.receipt_path.read_text(encoding="ascii"))
    body = dict(stored)
    seal = body.pop("receipt_sha256")

    assert dict(receipt) == stored
    assert MODULE.canonical_sha256(body) == seal
    assert stored["status"] == "failed_closed"
    assert stored["accepted_child_published"] is False
    assert stored["publication_present_after_failure"] is False
    assert stored["parent_outputs_regenerated"] is False
    assert stored["runner_constructs_optimizer"] is False
    with pytest.raises(MODULE.ProductionNeoFoldError, match="overwrite"):
        MODULE.write_neo_failed_closed_receipt(
            config,
            RuntimeError("must not replace prior evidence"),
        )
