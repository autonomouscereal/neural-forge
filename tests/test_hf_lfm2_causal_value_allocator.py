from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge import hf_lfm2_causal_value_allocator as causal


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


def _frontier(
    parent_identity: str,
    layer_count: int,
    source_units: int,
) -> dict:
    weights = {
        "final_logit_margin": 0.25,
        "eos": 0.15,
        "repetition": 0.15,
        "task_anchor": 0.25,
        "residual_path": 0.10,
        "attention_path": 0.10,
    }
    candidates = []
    for layer_index in range(layer_count):
        scale = (layer_index + 1) / layer_count
        for target in (source_units * 3 // 4, source_units // 2):
            damage = {
                "final_logit_margin": 0.08 * scale,
                "eos": 0.06 * scale,
                "repetition": 0.04 * scale,
                "task_anchor": 0.10 * scale,
                "residual_path": 0.12 * scale,
                "attention_path": 0.14 * scale,
            }
            candidates.append(
                {
                    "option_id": f"ffn-l{layer_index:02d}-w{target:04d}",
                    "site_kind": "ffn_width",
                    "layer_index": layer_index,
                    "source_units": source_units,
                    "target_units": target,
                    "damage": damage,
                    "validation_case_ids_sha256": "5" * 64,
                }
            )
    return {
        "frontier_sha256": "1" * 64,
        "parent_binding": {
            "parent_identity_sha256": parent_identity,
            "parent_trace_cache_key": "2" * 64,
        },
        "global_evidence": {
            "parent_corpus_sha256": "3" * 64,
            "damage_weights": weights,
        },
        "split": {
            "selection_rows_sha256": "4" * 64,
        },
        "candidates": candidates,
    }


@pytest.fixture(scope="module")
def causal_checkpoint(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("lfm2-causal-value")
    tokenizer_directory = root / "tokenizer"
    tokenizer_directory.mkdir()
    _write_tokenizer(tokenizer_directory)
    torch.manual_seed(29)
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
    # Equal FFN tensors make the cached-causal layer multiplier, rather than a
    # coincidental layerwise weight distribution, decide otherwise tied groups.
    with torch.no_grad():
        reference = parent.model.layers[0].feed_forward
        for layer in parent.model.layers[1:]:
            layer.feed_forward.w1.weight.copy_(reference.w1.weight)
            layer.feed_forward.w2.weight.copy_(reference.w2.weight)
            layer.feed_forward.w3.weight.copy_(reference.w3.weight)
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
    frontier = _frontier(
        parent_identity,
        len(parent.model.layers),
        config.intermediate_size,
    )
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        causal,
        "verify_lfm2_cached_causal_frontier_document",
        lambda value: value,
    )
    before = _state_sha256(parent)
    destination = root / "checkpoint"
    manifest = causal.export_lfm2_causal_value_fold(
        parent,
        destination,
        tokenizer_directory=tokenizer_directory,
        frontier_document=frontier,
        eos_token_ids=(2,),
        group_size=256,
        analysis_chunk_groups=256,
    )
    assert _state_sha256(parent) == before
    assert all(parameter.grad is None for parameter in parent.parameters())
    try:
        yield root, parent, frontier, destination, manifest
    finally:
        patcher.undo()


def test_frontier_is_reduced_to_per_layer_cached_causal_risk(
    causal_checkpoint,
) -> None:
    _root, _parent, frontier, _destination, _manifest = causal_checkpoint
    evidence = causal.derive_lfm2_causal_value_evidence(frontier)
    assert evidence.frontier_sha256 == "1" * 64
    assert evidence.layers[0].normalized_sensitivity == pytest.approx(0.0)
    assert evidence.layers[-1].normalized_sensitivity == pytest.approx(1.0)
    assert all(value.maximum_damage_slope > 0.0 for value in evidence.layers)
    assert causal.verify_lfm2_causal_value_evidence(evidence) is evidence
    assert evidence.as_dict()["source_contract"] == {
        "activation_selected_physical_proposals": True,
        "cached_parent_trace_reused": True,
        "candidate_child_full_prefix_traces": True,
        "residual_path_damage_used": True,
        "attention_path_damage_used": True,
        "exact_token_match_objective_used": False,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def test_optional_token_rows_must_come_from_the_bound_selection_cache(
    causal_checkpoint,
) -> None:
    _root, _parent, frontier, _destination, _manifest = causal_checkpoint
    evidence = causal.derive_lfm2_causal_value_evidence(frontier)
    body = {
        "schema": causal.TOKEN_SELECTION_SCHEMA,
        "parent_trace_cache_key": evidence.parent_trace_cache_key,
        "parent_corpus_sha256": evidence.parent_corpus_sha256,
        "selection_rows_sha256": evidence.selection_rows_sha256,
        "split_role": "selection",
        "selection_method": "frequency_margin",
        "token_ids": [11, 7, 11],
        "independent_holdout_used": False,
    }
    document = {**body, "selection_sha256": canonical_sha256(body)}
    assert causal._calibration_selected_token_ids(document, evidence) == (7, 11)
    contaminated_body = {
        **body,
        "independent_holdout_used": True,
    }
    contaminated = {
        **contaminated_body,
        "selection_sha256": canonical_sha256(contaminated_body),
    }
    with pytest.raises(
        causal.Lfm2CausalValueAllocatorError,
        match="selection-only",
    ):
        causal._calibration_selected_token_ids(contaminated, evidence)


def test_exact_50_checkpoint_spends_lower_bits_only_on_authorized_roles(
    causal_checkpoint,
) -> None:
    _root, _parent, _frontier, destination, manifest = causal_checkpoint
    observed = sum(path.stat().st_size for path in destination.iterdir())
    allocation = manifest["causal_value_allocation"]
    assert observed == manifest["storage"]["compressed_checkpoint_bytes"]
    assert observed <= manifest["target"]["target_checkpoint_bytes"]
    assert 0 <= manifest["target"]["headroom_bytes"] <= 32
    assert manifest["target"]["requested_reduction"] == 0.50
    assert allocation["counts"]["lower_group_count"] > 0
    assert (
        allocation["tensors"]["model.embed_tokens.weight"]["lower_7bit_group_count"]
        == 0
    )
    assert allocation["eos_token_ids"] == [2]
    for value in allocation["tensors"].values():
        if value["role"] in {
            "tied_embedding_output",
            "normalization",
            "attention_query_key",
            "convolution_state",
            "bias",
            "unclassified",
        }:
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
    assert manifest["method"]["training_used"] is False
    assert manifest["method"]["cached_parent_traces_used"] is True
    assert manifest["method"]["new_parent_traces_generated"] is False
    assert allocation["execution_audit"]["parent_forward_calls"] == 0
    assert allocation["execution_audit"]["optimizer_steps"] == 0
    assert allocation["execution_audit"]["backward_calls"] == 0
    exchange = allocation["exact_byte_exchange"]
    assert exchange["exact_source_value_bytes"] > 0
    assert exchange["additional_lower_groups_paid_for_exact_overlays"] > 0
    assert exchange["compensating_payload_bytes_saved"] == (
        exchange["additional_lower_groups_paid_for_exact_overlays"] * 32
    )


def test_causal_risk_prefers_the_less_sensitive_equal_weight_layer(
    causal_checkpoint,
) -> None:
    _root, _parent, _frontier, _destination, manifest = causal_checkpoint
    tensors = manifest["causal_value_allocation"]["tensors"]

    def lower(layer: int, role: str) -> int:
        return sum(
            value["lower_7bit_group_count"]
            for value in tensors.values()
            if value["layer_index"] == layer and value["role"] == role
        )

    assert lower(0, "ffn_producer") >= lower(3, "ffn_producer")


def test_two_fresh_reloads_are_standalone_tied_and_logit_identical(
    causal_checkpoint,
) -> None:
    _root, parent, _frontier, destination, manifest = causal_checkpoint
    first = causal.load_lfm2_causal_value_fold(destination)
    second = causal.load_lfm2_causal_value_fold(destination)
    assert tuple(name for name, _ in first.named_parameters()) == (
        "neural_forge_device_sentinel",
    )
    assert sum(parameter.numel() for parameter in first.parameters()) == 0
    assert first.model.embed_tokens._packed_weight is first.lm_head._packed_weight
    assert (
        first.neural_forge_load_report["causal_allocation_sha256"]
        == manifest["causal_value_allocation"]["allocation_sha256"]
    )
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
    assert first_logits.shape == (1, 1, parent.config.vocab_size)
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
    summary = causal.inspect_lfm2_causal_value_fold(destination)
    assert summary["requested_reduction"] == 0.50
    assert summary["all_norm_qk_conv_embedding_groups_base_width"] is True
    assert summary["embedding_lower_group_count"] == 0


def test_wrapper_rejects_an_ordinary_packed_control(
    causal_checkpoint,
    tmp_path: Path,
) -> None:
    _root, parent, _frontier, _destination, _manifest = causal_checkpoint
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    _write_tokenizer(tokenizer)
    ordinary = tmp_path / "ordinary"
    from neural_forge.hf_lfm2_value_fold_control import (
        export_lfm2_value_fold_control,
    )

    export_lfm2_value_fold_control(
        parent,
        ordinary,
        tokenizer_directory=tokenizer,
        target_reduction=0.50,
        group_size=256,
        analysis_chunk_groups=256,
    )
    with pytest.raises(
        causal.Lfm2CausalValueAllocatorFormatError,
        match="no cached-causal allocation",
    ):
        causal.load_lfm2_causal_value_fold(ordinary)
