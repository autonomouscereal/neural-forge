from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neural_forge import hf_lfm2_causal_value_allocator as causal
from neural_forge import hf_lfm2_protected_weight_error_allocator as ablation
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent


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


@pytest.fixture(scope="module")
def ablation_checkpoint(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("lfm2-protected-weight-error")
    tokenizer_directory = root / "tokenizer"
    tokenizer_directory.mkdir()
    _write_tokenizer(tokenizer_directory)
    torch.manual_seed(31)
    config = transformers.Lfm2Config(
        vocab_size=4096,
        hidden_size=64,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        layer_types=[
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
            "conv",
            "full_attention",
        ],
        full_attn_idxs=[1, 3, 5, 7],
        block_auto_adjust_ff_dim=False,
        conv_bias=True,
        conv_L_cache=3,
        tie_word_embeddings=True,
    )
    parent = transformers.Lfm2ForCausalLM(config).to(dtype=torch.bfloat16)
    parent.eval()
    before = _state_sha256(parent)
    destination = root / "checkpoint"
    manifest = ablation.export_lfm2_protected_weight_error_fold(
        parent,
        destination,
        tokenizer_directory=tokenizer_directory,
        eos_token_ids=(2,),
        group_size=256,
        analysis_chunk_groups=256,
    )
    assert _state_sha256(parent) == before
    assert all(parameter.grad is None for parameter in parent.parameters())
    return root, parent, destination, manifest


def test_order_is_weight_error_times_role_with_causal_multiplier_one() -> None:
    analyses = [
        SimpleNamespace(
            canonical_name="model.layers.0.self_attn.q_proj.weight",
            group_count=1,
            distortion_penalty=np.asarray([0.0]),
        ),
        SimpleNamespace(
            canonical_name="model.layers.7.feed_forward.w1.weight",
            group_count=1,
            distortion_penalty=np.asarray([0.10]),
        ),
        SimpleNamespace(
            canonical_name="model.layers.0.feed_forward.w2.weight",
            group_count=1,
            distortion_penalty=np.asarray([0.06]),
        ),
        SimpleNamespace(
            canonical_name="model.layers.3.self_attn.v_proj.weight",
            group_count=1,
            distortion_penalty=np.asarray([0.02]),
        ),
    ]
    order, offsets, roles = ablation._allocation_order(analyses)
    assert offsets == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert order.tolist() == [3, 1, 2]
    assert roles[analyses[0].canonical_name].eligible_for_lower_precision is False


def test_exact_50_manifest_is_distinct_noncausal_and_protected(
    ablation_checkpoint,
) -> None:
    _root, parent, destination, manifest = ablation_checkpoint
    observed = sum(path.stat().st_size for path in destination.iterdir())
    allocation = manifest[ablation.ALLOCATION_KEY]
    assert "causal_value_allocation" not in manifest
    assert observed == manifest["storage"]["compressed_checkpoint_bytes"]
    assert observed <= manifest["target"]["target_checkpoint_bytes"]
    assert 0 <= manifest["target"]["headroom_bytes"] <= 32
    assert manifest["target"]["requested_reduction"] == 0.50
    assert manifest["method"]["parent_traces_used"] is False
    assert manifest["method"]["causal_allocation_used"] is False
    assert manifest["method"]["protected_weight_error_ablation_used"] is True
    assert allocation["policy_id"] == ablation.POLICY_ID
    assert allocation["policy"]["fixed_causal_multiplier"] == 1.0
    assert allocation["policy"]["causal_evidence_used"] is False
    assert allocation["source_binding"]["parent_identity_sha256"] == (
        fingerprint_lfm2_parent(parent).identity_sha256
    )
    assert allocation["source_binding"]["cached_causal_frontier_used"] is False
    assert allocation["counts"]["lower_group_count"] > 0
    assert (
        allocation["tensors"]["model.embed_tokens.weight"]["lower_7bit_group_count"]
        == 0
    )
    for value in allocation["tensors"].values():
        if value["eligible_for_lower_precision"]:
            assert value["fixed_causal_multiplier"] == 1.0
            assert value["normalized_layer_sensitivity"] is None
        else:
            assert value["lower_7bit_group_count"] == 0
    lower_roles = {
        value["role"]
        for value in allocation["tensors"].values()
        if value["lower_7bit_group_count"]
    }
    assert lower_roles <= {
        "ffn_producer",
        "ffn_residual_writer",
        "attention_value",
        "attention_residual_writer",
    }
    audit = allocation["execution_audit"]
    assert audit["frontier_reads"] == 0
    assert audit["parent_forward_calls"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["backward_calls"] == 0
    exchange = allocation["exact_byte_exchange"]
    assert exchange["exact_source_value_bytes"] > 0
    assert exchange["additional_lower_groups_paid_for_exact_overlays"] > 0
    assert exchange["compensating_payload_bytes_saved"] == (
        exchange["additional_lower_groups_paid_for_exact_overlays"] * 32
    )


def test_two_fresh_reloads_preserve_exact_anchors_and_tied_identity(
    ablation_checkpoint,
) -> None:
    _root, parent, destination, manifest = ablation_checkpoint
    first = ablation.load_lfm2_protected_weight_error_fold(destination)
    second = ablation.load_lfm2_protected_weight_error_fold(destination)
    assert tuple(name for name, _ in first.named_parameters()) == (
        "neural_forge_device_sentinel",
    )
    assert sum(parameter.numel() for parameter in first.parameters()) == 0
    assert first.model.embed_tokens._packed_weight is first.lm_head._packed_weight
    assert (
        first.neural_forge_load_report["protected_weight_error_allocation_sha256"]
        == manifest[ablation.ALLOCATION_KEY]["allocation_sha256"]
    )
    assert first.neural_forge_load_report["causal_evidence_used"] is False
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
    torch.testing.assert_close(
        first.model.embed_tokens.weight[[0, 1, 2, 3]],
        parent.model.embed_tokens.weight[[0, 1, 2, 3]],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.model.embedding_norm.weight,
        parent.model.embedding_norm.weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.model.layers[0].conv.conv.weight,
        parent.model.layers[0].conv.conv.weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.model.layers[1].self_attn.q_layernorm.weight,
        parent.model.layers[1].self_attn.q_layernorm.weight,
        rtol=0.0,
        atol=0.0,
    )
    summary = ablation.inspect_lfm2_protected_weight_error_fold(destination)
    assert summary["requested_reduction"] == 0.50
    assert summary["fixed_causal_multiplier"] == 1.0
    assert summary["causal_frontier_used"] is False
    assert summary["all_norm_qk_conv_embedding_groups_base_width"] is True


def test_causal_loader_rejects_ablation_and_export_has_no_frontier_argument(
    ablation_checkpoint,
) -> None:
    _root, _parent, destination, _manifest = ablation_checkpoint
    with pytest.raises(
        causal.Lfm2CausalValueAllocatorFormatError,
        match="no cached-causal allocation",
    ):
        causal.load_lfm2_causal_value_fold(destination)
    signature = inspect.signature(ablation.export_lfm2_protected_weight_error_fold)
    assert "frontier_document" not in signature.parameters
    assert "calibration_token_selection" not in signature.parameters
