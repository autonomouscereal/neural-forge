from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_causal_residual_entropy_value_fold import (
    CrevfAuxiliarySpec,
    build_crevf_exact_spec,
    build_crevf_explicit_quantized_spec,
    encode_crevf_artifact,
    encode_crevf_auxiliary,
    encode_crevf_tensor,
)
from neural_forge.hf_lfm2_crevf_checkpoint import (
    ARTIFACT_FILENAME,
    CONFIG_FILENAME,
    GENERATION_CONFIG_FILENAME,
    MANIFEST_FILENAME,
    Lfm2CrevfCheckpointBudgetError,
    Lfm2CrevfCheckpointError,
    Lfm2CrevfCheckpointFormatError,
    export_lfm2_crevf_checkpoint,
    inspect_lfm2_crevf_checkpoint,
    load_lfm2_crevf_checkpoint,
    load_lfm2_crevf_tokenizer,
)
from neural_forge import hf_lfm2_crevf_checkpoint as checkpoint_module


class _FakeLfm2Config:
    def __init__(self, values):
        self._values = dict(values)
        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    def from_dict(cls, values):
        return cls(values)


class _FakeGenerationConfig:
    def __init__(self, values):
        self.values = dict(values)

    @classmethod
    def from_dict(cls, values):
        return cls(values)


class _FakeRotary(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.arange(
                max(1, config.hidden_size // 2),
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )


class Lfm2RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-5

    def forward(self, inputs):
        normalized = inputs.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(-1, keepdim=True)
            + self.variance_epsilon
        )
        return self.weight * normalized.to(inputs.dtype)


class _FakeLfm2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
            dtype=torch.bfloat16,
        )
        self.model.norm = Lfm2RMSNorm(config.hidden_size)
        self.model.conv = nn.Conv1d(
            1,
            1,
            3,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.model.rotary_emb = _FakeRotary(config)
        self.proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            dtype=torch.bfloat16,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, token_ids):
        hidden = self.model.embed_tokens(token_ids)
        hidden = self.model.norm(hidden)
        hidden = self.proj(hidden)
        return self.lm_head(hidden)


class _FakePreTrainedTokenizerFast:
    def __init__(self, *, tokenizer_object, **kwargs):
        self.backend = tokenizer_object
        self.kwargs = kwargs
        self.chat_template = None


@pytest.fixture
def fake_hf(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.Lfm2Config = _FakeLfm2Config
    transformers.Lfm2ForCausalLM = _FakeLfm2ForCausalLM
    transformers.GenerationConfig = _FakeGenerationConfig
    transformers.PreTrainedTokenizerFast = _FakePreTrainedTokenizerFast

    tokenizers = types.ModuleType("tokenizers")

    class Tokenizer:
        @classmethod
        def from_str(cls, value):
            json.loads(value)
            return {"authenticated_tokenizer_json": value}

    tokenizers.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)


def _write_package(root: Path) -> dict:
    root.mkdir()
    config = {
        "_name_or_path": "",
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 4096,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": ["conv"],
        "full_attn_idxs": [],
        "block_auto_adjust_ff_dim": False,
        "conv_bias": True,
        "conv_L_cache": 3,
        "tie_word_embeddings": True,
    }
    (root / CONFIG_FILENAME).write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    (root / GENERATION_CONFIG_FILENAME).write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": 2}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"type": "WordLevel"}}),
        encoding="utf-8",
    )
    common = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    (root / "tokenizer_config.json").write_text(
        json.dumps(common),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps(common),
        encoding="utf-8",
    )
    (root / "chat_template.jinja").write_text(
        "{{ messages | map(attribute='content') | join(' ') }}",
        encoding="utf-8",
    )
    return config


def _logical_and_aliases(model: nn.Module):
    logical = dict(model.named_parameters(remove_duplicate=False))
    groups = {}
    for name, parameter in logical.items():
        groups.setdefault(id(parameter), []).append(name)
    aliases = {}
    for names in groups.values():
        canonical = (
            "model.embed_tokens.weight"
            if "model.embed_tokens.weight" in names
            else sorted(names)[0]
        )
        aliases.update((name, canonical) for name in names)
    return logical, dict(sorted(aliases.items()))


def _zero_quantized_blob(name: str, shape: tuple[int, ...]) -> bytes:
    source_numel = int(np.prod(shape))
    group_size = 256
    group_count = (source_numel + group_size - 1) // group_size
    codes = np.full((group_count, group_size), 7, dtype=np.uint8)
    scales = np.ones(group_count, dtype=np.float16)
    return encode_crevf_tensor(
        build_crevf_explicit_quantized_spec(
            name=name,
            role="control",
            shape=shape,
            group_size=group_size,
            group_bit_widths=(4,) * group_count,
            group_codes=codes,
            scales=scales,
        )
    )


def _fixture_artifact(
    model: nn.Module,
    *,
    auxiliary_kind: str = "metadata.composite",
) -> tuple[bytes, dict[str, str]]:
    logical, aliases = _logical_and_aliases(model)
    tensor_blobs = []
    for canonical in sorted(set(aliases.values())):
        parameter = logical[canonical]
        if canonical == "model.norm.weight":
            tensor_blobs.append(
                encode_crevf_tensor(
                    build_crevf_exact_spec(
                        name=canonical,
                        role="normalization",
                        values=torch.ones(
                            tuple(parameter.shape),
                            dtype=torch.bfloat16,
                        ),
                    )
                )
            )
        else:
            tensor_blobs.append(
                _zero_quantized_blob(canonical, tuple(parameter.shape))
            )
    auxiliary = encode_crevf_auxiliary(
        CrevfAuxiliarySpec(
            name="checkpoint.bundle.metadata",
            kind=auxiliary_kind,
            payload=b'{"bundle":"tiny-lfm2","execution":false}',
        )
    )
    return (
        encode_crevf_artifact(
            tensor_blobs,
            auxiliary_blobs=(auxiliary,),
        ),
        aliases,
    )


def _dense_reference(
    model: nn.Module,
    package: Path,
) -> int:
    unique = {}
    for _name, parameter in model.named_parameters(remove_duplicate=False):
        unique[id(parameter)] = parameter
    tensor_bytes = sum(parameter.numel() * 2 for parameter in unique.values())
    package_bytes = sum(path.stat().st_size for path in package.iterdir())
    return tensor_bytes + package_bytes


def _checkpoint_inputs(tmp_path: Path, *, auxiliary_kind="metadata.composite"):
    package = tmp_path / "package"
    config_data = _write_package(package)
    config = _FakeLfm2Config.from_dict(config_data)
    model = _FakeLfm2ForCausalLM(config)
    artifact, aliases = _fixture_artifact(
        model,
        auxiliary_kind=auxiliary_kind,
    )
    reference = _dense_reference(model, package)
    return package, artifact, aliases, reference


def _export(
    tmp_path: Path,
    *,
    target: float = 0.65,
    name: str = "checkpoint",
    auxiliary_kind: str = "metadata.composite",
):
    package, artifact, aliases, reference = _checkpoint_inputs(
        tmp_path,
        auxiliary_kind=auxiliary_kind,
    )
    destination = tmp_path / name
    manifest = export_lfm2_crevf_checkpoint(
        destination,
        artifact=artifact,
        local_package_directory=package,
        logical_to_canonical=aliases,
        target_reduction=target,
        dense_reference_bytes=reference,
    )
    return destination, manifest, package, artifact, aliases, reference


@pytest.mark.parametrize("target", [0.50, 0.55, 0.60, 0.65])
def test_exact_physical_and_persistent_caps_for_every_supported_target(
    tmp_path,
    target,
) -> None:
    destination, manifest, _package, _artifact, _aliases, reference = _export(
        tmp_path,
        target=target,
    )
    storage = manifest["storage"]
    target_meta = manifest["target"]
    observed = sum(path.stat().st_size for path in destination.iterdir())
    ceiling = int(reference * (1.0 - target))
    assert target_meta["target_checkpoint_ceiling_bytes"] == ceiling
    assert target_meta["physical_cap_bytes"] == ceiling
    assert target_meta["persistent_cap_bytes"] <= ceiling
    assert observed == storage["checkpoint_physical_bytes"]
    assert observed <= ceiling
    assert storage["persistent_encoded_bytes"] <= target_meta[
        "persistent_cap_bytes"
    ]
    assert storage["manifest_bytes"] == (
        destination / MANIFEST_FILENAME
    ).stat().st_size
    assert storage["physical_headroom_bytes"] == ceiling - observed
    assert storage["persistent_headroom_bytes"] == (
        target_meta["persistent_cap_bytes"]
        - storage["persistent_encoded_bytes"]
    )
    assert set(path.name for path in destination.iterdir()) == {
        MANIFEST_FILENAME,
        ARTIFACT_FILENAME,
        CONFIG_FILENAME,
        GENERATION_CONFIG_FILENAME,
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    summary = inspect_lfm2_crevf_checkpoint(destination)
    assert summary["checkpoint_physical_bytes"] == observed
    assert summary["persistent_encoded_bytes"] == storage[
        "persistent_encoded_bytes"
    ]
    assert summary["auxiliary_count"] == 1


def test_fresh_meta_reload_restores_local_runtime_rotary_and_tokenizer(
    tmp_path,
    fake_hf,
) -> None:
    destination, manifest, *_rest = _export(tmp_path)
    first = load_lfm2_crevf_checkpoint(destination)
    second = load_lfm2_crevf_checkpoint(destination)
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    assert torch.equal(first(token_ids), second(token_ids))
    assert first.neural_forge_checkpoint_manifest == manifest
    assert first.neural_forge_load_report["constructed_on_meta"] is True
    assert first.neural_forge_load_report["hidden_dense_state_bytes"] == 0
    assert first.neural_forge_load_report["tokenizer_restored_locally"] is True
    assert first.neural_forge_load_report["generation_config_restored_locally"]
    assert all(parameter.numel() == 0 for parameter in first.parameters())
    assert tuple(first.state_dict()) == ("neural_forge_device_sentinel",)
    assert first.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert first.model.rotary_emb.inv_freq.device.type == "cpu"
    assert (
        first.model.embed_tokens._packed_weight
        is first.lm_head._packed_weight
        is first.neural_forge_crevf_runtime.tensor(
            "model.embed_tokens.weight"
        )
    )
    assert isinstance(first.generation_config, _FakeGenerationConfig)
    assert isinstance(first.neural_forge_tokenizer, _FakePreTrainedTokenizerFast)
    standalone_tokenizer = load_lfm2_crevf_tokenizer(destination)
    assert standalone_tokenizer.neural_forge_tokenizer_fingerprint == manifest[
        "tokenizer"
    ]["fingerprint"]


def test_repeat_exports_are_byte_identical(tmp_path) -> None:
    package, artifact, aliases, reference = _checkpoint_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = export_lfm2_crevf_checkpoint(
        first,
        artifact=artifact,
        local_package_directory=package,
        logical_to_canonical=aliases,
        target_reduction=0.65,
        dense_reference_bytes=reference,
    )
    second_manifest = export_lfm2_crevf_checkpoint(
        second,
        artifact=artifact,
        local_package_directory=package,
        logical_to_canonical=aliases,
        target_reduction=0.65,
        dense_reference_bytes=reference,
    )
    assert first_manifest == second_manifest
    for name in sorted(path.name for path in first.iterdir()):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_corruption_extra_file_and_symlink_fail_closed(tmp_path) -> None:
    destination, _manifest, *_rest = _export(tmp_path)

    corrupted = tmp_path / "corrupted"
    shutil.copytree(destination, corrupted)
    artifact = bytearray((corrupted / ARTIFACT_FILENAME).read_bytes())
    artifact[-1] ^= 1
    (corrupted / ARTIFACT_FILENAME).write_bytes(artifact)
    with pytest.raises(Lfm2CrevfCheckpointFormatError, match="SHA-256"):
        inspect_lfm2_crevf_checkpoint(corrupted)

    extra = tmp_path / "extra"
    shutil.copytree(destination, extra)
    (extra / "unclaimed.bin").write_bytes(b"x")
    with pytest.raises(
        Lfm2CrevfCheckpointFormatError,
        match="missing or unclaimed",
    ):
        inspect_lfm2_crevf_checkpoint(extra)

    linked = tmp_path / "linked"
    shutil.copytree(destination, linked)
    source = linked / "tokenizer_config.json"
    external = tmp_path / "external-tokenizer-config.json"
    external.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(external)
    except OSError:
        pytest.skip("This Windows account cannot create test symlinks.")
    with pytest.raises(Lfm2CrevfCheckpointFormatError, match="symlink"):
        inspect_lfm2_crevf_checkpoint(linked)


def _rewrite_manifest(path: Path, mutator, *, converge: bool = True) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("manifest_sha256")
    mutator(document)
    if not converge:
        document["manifest_sha256"] = canonical_sha256(document)
        path.write_bytes(canonical_json_bytes(document) + b"\n")
        return
    other_bytes = document["storage"]["checkpoint_physical_bytes"] - document[
        "storage"
    ]["manifest_bytes"]
    for _ in range(32):
        document["manifest_sha256"] = canonical_sha256(document)
        raw = canonical_json_bytes(document) + b"\n"
        manifest_bytes = len(raw)
        checkpoint_bytes = other_bytes + manifest_bytes
        core = dict(document)
        core.pop("manifest_sha256")
        changed = (
            core["storage"]["manifest_bytes"] != manifest_bytes
            or core["storage"]["checkpoint_physical_bytes"]
            != checkpoint_bytes
            or core["storage"]["physical_headroom_bytes"]
            != core["target"]["target_checkpoint_ceiling_bytes"]
            - checkpoint_bytes
        )
        core["storage"]["manifest_bytes"] = manifest_bytes
        core["storage"]["checkpoint_physical_bytes"] = checkpoint_bytes
        core["storage"]["physical_headroom_bytes"] = (
            core["target"]["target_checkpoint_ceiling_bytes"]
            - checkpoint_bytes
        )
        document = core
        if not changed:
            document["manifest_sha256"] = canonical_sha256(document)
            path.write_bytes(canonical_json_bytes(document) + b"\n")
            return
    raise AssertionError("Mutated test manifest did not converge.")


def test_manifest_size_and_persistent_cap_mutations_fail_closed(tmp_path) -> None:
    destination, _manifest, *_rest = _export(tmp_path)

    wrong_size = tmp_path / "wrong-size"
    shutil.copytree(destination, wrong_size)
    _rewrite_manifest(
        wrong_size / MANIFEST_FILENAME,
        lambda value: value["storage"].__setitem__(
            "manifest_bytes",
            value["storage"]["manifest_bytes"] + 1,
        ),
        converge=False,
    )
    with pytest.raises(
        Lfm2CrevfCheckpointFormatError,
        match="byte accounting",
    ):
        inspect_lfm2_crevf_checkpoint(wrong_size)

    bad_cap = tmp_path / "bad-cap"
    shutil.copytree(destination, bad_cap)

    def lower_cap(value):
        persistent = value["storage"]["persistent_encoded_bytes"]
        value["target"]["persistent_cap_bytes"] = persistent - 1
        value["storage"]["persistent_headroom_bytes"] = -1

    _rewrite_manifest(bad_cap / MANIFEST_FILENAME, lower_cap)
    with pytest.raises(Lfm2CrevfCheckpointBudgetError, match="cap"):
        inspect_lfm2_crevf_checkpoint(bad_cap)


def test_tie_and_dense_reference_are_proven_not_claimed(tmp_path) -> None:
    package, artifact, aliases, reference = _checkpoint_inputs(tmp_path)
    untied = dict(aliases)
    untied["lm_head.weight"] = "lm_head.weight"
    with pytest.raises(Lfm2CrevfCheckpointError, match="tie"):
        export_lfm2_crevf_checkpoint(
            tmp_path / "untied",
            artifact=artifact,
            local_package_directory=package,
            logical_to_canonical=untied,
            target_reduction=0.65,
            dense_reference_bytes=reference,
        )
    with pytest.raises(
        Lfm2CrevfCheckpointError,
        match="dense_reference_bytes",
    ):
        export_lfm2_crevf_checkpoint(
            tmp_path / "false-reference",
            artifact=artifact,
            local_package_directory=package,
            logical_to_canonical=aliases,
            target_reduction=0.65,
            dense_reference_bytes=reference + 1,
        )


def test_composite_auxiliary_is_opaque_and_executable_kind_is_rejected(
    tmp_path,
) -> None:
    destination, manifest, *_rest = _export(tmp_path)
    auxiliary = manifest["runtime"]["inventory"]["auxiliaries"][
        "checkpoint.bundle.metadata"
    ]
    assert auxiliary["kind"] == "metadata.composite"
    assert auxiliary["execution"] == "opaque_nonexecuting"

    other = tmp_path / "unsupported"
    other.mkdir()
    package, artifact, aliases, reference = _checkpoint_inputs(
        other,
        auxiliary_kind="serq.writer_indices",
    )
    with pytest.raises(
        Lfm2CrevfCheckpointFormatError,
        match="unsupported executable",
    ):
        export_lfm2_crevf_checkpoint(
            other / "checkpoint",
            artifact=artifact,
            local_package_directory=package,
            logical_to_canonical=aliases,
            target_reduction=0.65,
            dense_reference_bytes=reference,
        )


def test_export_is_transactional_on_atomic_rename_failure(
    tmp_path,
    monkeypatch,
) -> None:
    package, artifact, aliases, reference = _checkpoint_inputs(tmp_path)
    destination = tmp_path / "atomic-failure"

    def fail_replace(_source, _destination):
        raise OSError("injected atomic rename failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        export_lfm2_crevf_checkpoint(
            destination,
            artifact=artifact,
            local_package_directory=package,
            logical_to_canonical=aliases,
            target_reduction=0.65,
            dense_reference_bytes=reference,
        )
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".atomic-failure.*.tmp"))


def test_loader_enforces_caller_caps_before_materialized_model(
    tmp_path,
    fake_hf,
    monkeypatch,
) -> None:
    destination, manifest, *_rest = _export(tmp_path)
    called = False

    class ShouldNotConstruct(_FakeLfm2ForCausalLM):
        def __init__(self, config):
            nonlocal called
            called = True
            super().__init__(config)

    sys.modules["transformers"].Lfm2ForCausalLM = ShouldNotConstruct
    with pytest.raises(Lfm2CrevfCheckpointBudgetError, match="physical"):
        load_lfm2_crevf_checkpoint(
            destination,
            max_checkpoint_bytes=manifest["storage"][
                "checkpoint_physical_bytes"
            ]
            - 1,
        )
    assert called is False


def test_checkpoint_module_has_no_hub_parent_training_or_generation_calls() -> None:
    source = Path(checkpoint_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {
        "from_pretrained",
        "generate",
        "backward",
        "step",
        "train",
    }.intersection(attributes | names)
