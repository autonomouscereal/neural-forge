from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from neural_forge.hf_lfm2_exact50_all8 import (
    MANIFEST_FILENAME,
    PACKAGE_FILENAME,
    PAYLOAD_FILENAME,
    All8PackedQuantizedTensor,
    Lfm2Exact50All8BudgetError,
    Lfm2Exact50All8Error,
    Lfm2Exact50All8FormatError,
    export_lfm2_exact50_all8,
    inspect_lfm2_exact50_all8,
    load_lfm2_exact50_all8,
    load_lfm2_exact50_all8_tokenizer,
)
from neural_forge.hf_lfm2_value_fold_control import (
    _parameter_chunks,
    _quantize_groups,
)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _write_tokenizer(directory: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
                "world": 5,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    common = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    (directory / "tokenizer_config.json").write_text(
        json.dumps(common), encoding="utf-8"
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps(common), encoding="utf-8"
    )
    (directory / "chat_template.jinja").write_text(
        "{{ messages | map(attribute='content') | join(' ') }}",
        encoding="utf-8",
    )
    # The production tokenizer contributes several MiB to the exact reference.
    # Give the tiny fixture the same accounting shape without changing its
    # backend vocabulary.
    (directory / "added_tokens.json").write_text(
        json.dumps({f"<fixture-unused-{index}>": index + 6 for index in range(4096)}),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def all8_checkpoint(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("lfm2-exact50-all8")
    tokenizer_directory = root / "tokenizer"
    tokenizer_directory.mkdir()
    _write_tokenizer(tokenizer_directory)
    torch.manual_seed(29)
    config = transformers.Lfm2Config(
        vocab_size=16384,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        full_attn_idxs=[1],
        block_auto_adjust_ff_dim=False,
        conv_bias=True,
        conv_L_cache=3,
        tie_word_embeddings=True,
    )
    parent = transformers.Lfm2ForCausalLM(config).to(dtype=torch.bfloat16)
    parent.eval()
    before = _state_sha256(parent)
    destination = root / "checkpoint"
    manifest = export_lfm2_exact50_all8(
        parent,
        destination,
        tokenizer_directory=tokenizer_directory,
        group_size=256,
        analysis_chunk_groups=1024,
    )
    assert _state_sha256(parent) == before
    assert all(parameter.grad is None for parameter in parent.parameters())
    return tokenizer_directory, parent, destination, manifest


def test_all8_exact_disk_and_complete_runtime_accounting(all8_checkpoint) -> None:
    _tokenizer, _parent, destination, manifest = all8_checkpoint
    observed = sum(path.stat().st_size for path in destination.iterdir())
    storage = manifest["storage"]
    target = manifest["target"]
    assert {path.name for path in destination.iterdir()} == {
        MANIFEST_FILENAME,
        PACKAGE_FILENAME,
        PAYLOAD_FILENAME,
    }
    assert observed == storage["compressed_checkpoint_bytes"]
    assert observed <= target["target_checkpoint_bytes"]
    assert target["headroom_bytes"] == target["target_checkpoint_bytes"] - observed
    assert target["target_checkpoint_bytes"] == (
        storage["accounted_dense_reference_bytes"] // 2
    )
    assert storage["raw_8bit_value_bytes"] == sum(
        value["padded_value_count"] for value in manifest["tensors"].values()
    )
    assert storage["raw_float16_scale_bytes"] == (
        manifest["quantization"]["total_group_count"] * 2
    )
    assert storage["packed_weight_runtime_bytes"] == (
        storage["raw_8bit_value_bytes"] + storage["raw_float16_scale_bytes"]
    )
    assert storage["materialized_runtime_tensor_bytes"] == (
        storage["packed_weight_runtime_bytes"]
        + storage["auxiliary_runtime_buffer_bytes"]
    )
    assert storage["materialized_runtime_tensor_bytes"] <= target[
        "target_checkpoint_bytes"
    ]
    assert manifest["quantization"]["minimum_code_bit_width"] == 8
    assert manifest["quantization"]["maximum_code_bit_width"] == 8
    assert manifest["quantization"]["seven_bit_group_count"] == 0
    assert manifest["method"]["seven_bit_weight_codes"] == 0
    assert manifest["storage_allocator"]["mask_persistent_at_runtime"] is False
    assert storage["hidden_dense_checkpoint_copy"] is False
    assert storage["external_weight_sidecars"] == []
    summary = inspect_lfm2_exact50_all8(destination)
    assert summary["compressed_checkpoint_bytes"] == observed
    assert summary["minimum_code_bit_width"] == 8
    assert summary["seven_bit_group_count"] == 0


def test_every_runtime_code_exactly_matches_uniform_int8_quantizer(
    all8_checkpoint,
) -> None:
    _tokenizer, parent, destination, manifest = all8_checkpoint
    loaded = load_lfm2_exact50_all8(destination)
    parent_parameters = dict(
        parent.named_parameters(recurse=True, remove_duplicate=False)
    )
    registry = loaded.neural_forge_packed_registry
    canonical_names = sorted(manifest["tensors"])
    assert len(registry) == len(canonical_names)
    for name, store in zip(canonical_names, registry, strict=True):
        assert isinstance(store, All8PackedQuantizedTensor)
        expected_parts = []
        for _start, _end, _vs, _ve, values in _parameter_chunks(
            parent_parameters[name],
            group_size=manifest["quantization"]["group_size"],
            chunk_groups=1024,
        ):
            codes, _scales, _reconstructed = _quantize_groups(values, 8)
            expected_parts.append(codes.reshape(-1))
        expected = torch.cat(expected_parts)
        assert torch.equal(store.codes.cpu(), expected)
        assert store.codes.numel() == manifest["tensors"][name][
            "padded_value_count"
        ]


def test_two_fresh_reloads_are_tied_dense_free_and_logit_identical(
    all8_checkpoint,
) -> None:
    _tokenizer, parent, destination, manifest = all8_checkpoint
    first = load_lfm2_exact50_all8(destination)
    second = load_lfm2_exact50_all8(destination)
    assert tuple(name for name, _value in first.named_parameters()) == (
        "neural_forge_device_sentinel",
    )
    assert sum(parameter.numel() for parameter in first.parameters()) == 0
    assert first.model.embed_tokens._packed_weight is first.lm_head._packed_weight
    assert first.neural_forge_load_report[
        "all_logical_weight_codes_are_8bit"
    ] is True
    assert first.neural_forge_load_report["seven_bit_weight_codes"] == 0
    assert first.neural_forge_load_report[
        "materialized_runtime_tensor_bytes"
    ] == manifest["storage"]["materialized_runtime_tensor_bytes"]
    assert first.neural_forge_load_report["storage_class_mask_retained"] is False
    source_buffers = dict(parent.named_buffers(remove_duplicate=False))
    loaded_buffers = dict(first.named_buffers(remove_duplicate=False))
    for name in (
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    ):
        assert loaded_buffers[name].dtype == source_buffers[name].dtype
        assert torch.equal(loaded_buffers[name], source_buffers[name])
    input_ids = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    with torch.inference_mode():
        first_logits = first(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits
        second_logits = second(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits
    assert torch.equal(first_logits, second_logits)
    assert bool(torch.isfinite(first_logits).all())
    assert first_logits.shape == (1, 1, parent.config.vocab_size)


def test_compressed_tokenizer_loads_without_extracted_sidecars(
    all8_checkpoint,
) -> None:
    _tokenizer, _parent, destination, manifest = all8_checkpoint
    tokenizer = load_lfm2_exact50_all8_tokenizer(destination)
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]
    assert tokenizer.neural_forge_tokenizer_fingerprint == manifest["tokenizer"][
        "fingerprint"
    ]
    assert not (destination / "tokenizer.json").exists()


def test_loader_fails_closed_on_corruption_extra_file_and_runtime_limit(
    all8_checkpoint,
    tmp_path: Path,
) -> None:
    _tokenizer, _parent, source, manifest = all8_checkpoint
    corrupt = tmp_path / "corrupt"
    shutil.copytree(source, corrupt)
    payload_path = corrupt / PAYLOAD_FILENAME
    payload = bytearray(payload_path.read_bytes())
    payload[-1] ^= 1
    payload_path.write_bytes(payload)
    with pytest.raises(
        Lfm2Exact50All8FormatError,
        match="payload artifact failed path, size, or SHA-256",
    ):
        load_lfm2_exact50_all8(corrupt)

    extra = tmp_path / "extra"
    shutil.copytree(source, extra)
    (extra / "hidden_dense.safetensors").write_bytes(b"forbidden")
    with pytest.raises(Lfm2Exact50All8FormatError, match="unclaimed"):
        load_lfm2_exact50_all8(extra)

    with pytest.raises(
        Lfm2Exact50All8BudgetError,
        match="max_packed_runtime_bytes",
    ):
        load_lfm2_exact50_all8(
            source,
            max_packed_runtime_bytes=(
                manifest["storage"]["materialized_runtime_tensor_bytes"] - 1
            ),
        )


def test_export_refuses_existing_destination(all8_checkpoint, tmp_path: Path) -> None:
    tokenizer, parent, _destination, _manifest = all8_checkpoint
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(Lfm2Exact50All8Error, match="Refusing to replace"):
        export_lfm2_exact50_all8(
            parent,
            existing,
            tokenizer_directory=tokenizer,
        )
