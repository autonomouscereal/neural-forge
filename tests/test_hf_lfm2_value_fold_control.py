from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from neural_forge.hf_lfm2_value_fold_control import (
    CONFIG_FILENAME,
    GENERATION_CONFIG_FILENAME,
    MANIFEST_FILENAME,
    PAYLOAD_FILENAME,
    PAYLOAD_MAGIC,
    Lfm2ValueFoldControlBudgetError,
    Lfm2ValueFoldControlError,
    Lfm2ValueFoldControlFormatError,
    Lfm2ValueFoldTarget,
    PackedConv1d,
    PackedEmbedding,
    PackedLfm2RMSNorm,
    PackedLinear,
    export_lfm2_value_fold_control,
    inspect_lfm2_value_fold_control,
    load_lfm2_value_fold_control,
    load_lfm2_value_fold_tokenizer,
    pack_unsigned_codes,
    unpack_unsigned_codes,
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
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        ),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        ),
        encoding="utf-8",
    )
    (directory / "chat_template.jinja").write_text(
        "{{ messages | map(attribute='content') | join(' ') }}",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def packed_controls(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("lfm2-packed-value-control")
    tokenizer_directory = root / "tokenizer"
    tokenizer_directory.mkdir()
    _write_tokenizer(tokenizer_directory)
    torch.manual_seed(19)
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
    outputs = {}
    for reduction in (0.50, 0.75):
        destination = root / f"target-{int(reduction * 100)}"
        manifest = export_lfm2_value_fold_control(
            parent,
            destination,
            tokenizer_directory=tokenizer_directory,
            target_reduction=reduction,
            group_size=256,
            analysis_chunk_groups=1024,
        )
        outputs[reduction] = (destination, manifest)
    assert _state_sha256(parent) == before
    assert all(parameter.grad is None for parameter in parent.parameters())
    return root, tokenizer_directory, parent, outputs


@pytest.mark.parametrize("bit_width", [3, 4, 7, 8])
def test_unsigned_bit_pack_round_trips_lsb_first(bit_width: int) -> None:
    maximum = (1 << bit_width) - 1
    codes = (torch.arange(513, dtype=torch.int16) % (maximum + 1)).to(torch.uint8)
    packed = pack_unsigned_codes(codes, bit_width)
    assert len(packed) == (codes.numel() * bit_width + 7) // 8
    restored = unpack_unsigned_codes(
        torch.frombuffer(bytearray(packed), dtype=torch.uint8),
        value_count=codes.numel(),
        bit_width=bit_width,
    )
    assert torch.equal(restored, codes)


def test_value_fold_targets_are_the_two_large_controls() -> None:
    half = Lfm2ValueFoldTarget.from_reduction(0.50)
    three_quarters = Lfm2ValueFoldTarget.from_reduction(0.75)
    assert (half.base_bit_width, half.lower_bit_width) == (8, 7)
    assert (three_quarters.base_bit_width, three_quarters.lower_bit_width) == (
        4,
        3,
    )
    assert half.bytes_saved_per_lower_group == 32
    with pytest.raises(Lfm2ValueFoldControlError, match="0.50 or 0.75"):
        Lfm2ValueFoldTarget.from_reduction(0.60)
    with pytest.raises(Lfm2ValueFoldControlError, match="multiple of eight"):
        Lfm2ValueFoldTarget(500_000, group_size=255)


@pytest.mark.parametrize("reduction", [0.50, 0.75])
def test_exact_physical_accounting_includes_all_metadata(
    packed_controls,
    reduction: float,
) -> None:
    _root, _tokenizer, _parent, outputs = packed_controls
    destination, manifest = outputs[reduction]
    observed = sum(path.stat().st_size for path in destination.iterdir())
    storage = manifest["storage"]
    target = manifest["target"]
    assert observed == storage["compressed_checkpoint_bytes"]
    assert observed <= target["target_checkpoint_bytes"]
    assert target["headroom_bytes"] == target["target_checkpoint_bytes"] - observed
    assert 0 <= target["headroom_bytes"] <= 32
    assert storage["accounted_dense_reference_bytes"] == (
        storage["dense_source_tensor_bytes"]
        + storage["shared_nonweight_artifact_bytes"]
    )
    assert storage["payload_bytes"] == (
        storage["payload_header_bytes"]
        + storage["packed_value_bytes"]
        + storage["group_selector_index_bytes"]
        + storage["scale_bytes"]
    )
    manifest_bytes = (destination / MANIFEST_FILENAME).stat().st_size
    assert observed == (
        storage["payload_bytes"]
        + storage["shared_nonweight_artifact_bytes"]
        + manifest_bytes
    )
    assert target["target_checkpoint_bytes"] == (
        storage["accounted_dense_reference_bytes"]
        * (1_000_000 - target["requested_reduction_ppm"])
        // 1_000_000
    )
    assert manifest["method"]["structural_parameter_count_compaction"] is False
    assert manifest["method"]["logical_neurons_removed"] == 0
    assert manifest["method"]["logical_heads_removed"] == 0
    assert manifest["method"]["logical_layers_removed"] == 0
    assert manifest["state"]["tied_vocabulary_stored_once"] is True
    assert storage["hidden_dense_checkpoint_copy"] is False
    assert storage["external_weight_sidecars"] == []
    names = {path.name for path in destination.iterdir()}
    assert names == {
        CONFIG_FILENAME,
        GENERATION_CONFIG_FILENAME,
        MANIFEST_FILENAME,
        PAYLOAD_FILENAME,
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    assert not any(
        path.suffix in {".bin", ".pt", ".pth", ".safetensors", ".nfraw"}
        for path in destination.iterdir()
    )
    payload = (destination / PAYLOAD_FILENAME).read_bytes()
    assert payload.startswith(PAYLOAD_MAGIC)
    summary = inspect_lfm2_value_fold_control(destination)
    assert summary["compressed_checkpoint_bytes"] == observed
    assert summary["manifest_bytes"] == manifest_bytes
    assert summary["structural_parameter_count_compaction"] is False


@pytest.mark.parametrize("reduction", [0.50, 0.75])
def test_fresh_reload_is_compressed_tied_and_logit_identical(
    packed_controls,
    reduction: float,
) -> None:
    _root, _tokenizer, parent, outputs = packed_controls
    destination, manifest = outputs[reduction]
    first = load_lfm2_value_fold_control(destination)
    second = load_lfm2_value_fold_control(destination)
    assert sum(parameter.numel() for parameter in first.parameters()) == 0
    assert tuple(name for name, _ in first.named_parameters()) == (
        "neural_forge_device_sentinel",
    )
    assert first.device.type == "cpu"
    assert first.dtype == torch.bfloat16
    assert first.model.embed_tokens._packed_weight is first.lm_head._packed_weight
    assert first.neural_forge_load_report["tied_vocabulary_stored_once"] is True
    assert first.neural_forge_load_report["packed_runtime_bytes"] == manifest[
        "storage"
    ]["payload_bytes"] - len(PAYLOAD_MAGIC)
    assert isinstance(first.model.embed_tokens, PackedEmbedding)
    assert isinstance(first.lm_head, PackedLinear)
    assert isinstance(first.model.layers[0].conv.conv, PackedConv1d)
    assert isinstance(first.model.layers[0].operator_norm, PackedLfm2RMSNorm)
    assert len(first.model.layers) == len(parent.model.layers)
    assert first.config.num_attention_heads == parent.config.num_attention_heads
    assert first.config.num_key_value_heads == parent.config.num_key_value_heads
    assert first.config.intermediate_size == parent.config.intermediate_size
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


def test_verbatim_tokenizer_package_reloads_without_a_hub_loader(
    packed_controls,
) -> None:
    _root, _tokenizer, _parent, outputs = packed_controls
    destination, manifest = outputs[0.50]
    tokenizer = load_lfm2_value_fold_tokenizer(destination)
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]
    assert (
        tokenizer.neural_forge_tokenizer_fingerprint
        == manifest["tokenizer"]["fingerprint"]
    )


def test_loader_fails_closed_on_payload_corruption_and_extra_sidecar(
    packed_controls,
    tmp_path: Path,
) -> None:
    _root, _tokenizer, _parent, outputs = packed_controls
    source, manifest = outputs[0.75]
    corrupt = tmp_path / "corrupt"
    shutil.copytree(source, corrupt)
    payload_path = corrupt / PAYLOAD_FILENAME
    payload = bytearray(payload_path.read_bytes())
    payload[len(PAYLOAD_MAGIC) + 7] ^= 1
    payload_path.write_bytes(payload)
    with pytest.raises(
        Lfm2ValueFoldControlFormatError,
        match="payload artifact failed path, size, or SHA-256",
    ):
        load_lfm2_value_fold_control(corrupt)

    extra = tmp_path / "extra"
    shutil.copytree(source, extra)
    (extra / "hidden_dense.safetensors").write_bytes(b"forbidden")
    with pytest.raises(
        Lfm2ValueFoldControlFormatError,
        match="unclaimed",
    ):
        load_lfm2_value_fold_control(extra)

    with pytest.raises(
        Lfm2ValueFoldControlBudgetError,
        match="max_checkpoint_bytes",
    ):
        load_lfm2_value_fold_control(
            source,
            max_checkpoint_bytes=(
                manifest["storage"]["compressed_checkpoint_bytes"] - 1
            ),
        )


def test_export_refuses_float32_and_existing_destinations(
    packed_controls,
    tmp_path: Path,
) -> None:
    _root, tokenizer, parent, _outputs = packed_controls
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(Lfm2ValueFoldControlError, match="Refusing to replace"):
        export_lfm2_value_fold_control(
            parent,
            existing,
            tokenizer_directory=tokenizer,
            target_reduction=0.50,
        )
    float_parent = copy.deepcopy(parent).to(dtype=torch.float32)
    with pytest.raises(
        Lfm2ValueFoldControlError,
        match="float16 or bfloat16",
    ):
        export_lfm2_value_fold_control(
            float_parent,
            tmp_path / "float32",
            tokenizer_directory=tokenizer,
            target_reduction=0.50,
        )
