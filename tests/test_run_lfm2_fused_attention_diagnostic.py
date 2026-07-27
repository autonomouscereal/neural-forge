from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments import prepare_lfm2_fused_attention_diagnostic as producer
from experiments import run_lfm2_fused_attention_diagnostic as runner
from experiments import run_lfm2_residual_attention_fold as residual_runner
from experiments.review_lfm2_semantic_frontier import PARENT_ENVELOPE_SCHEMA
from experiments.run_lfm2_dynamic_catalog import fingerprint_local_tokenizer
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_direct_local import (
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (
    _checkpoint_state,
    _module_state_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)
from tests import test_hf_lfm2_fused_attention_diagnostic as diagnostic_fixture
from tests.test_prepare_lfm2_progressive_component_ablation import _tokenizer


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _float_source():
    source = diagnostic_fixture._source()
    replacements = {
        name: tensor.float().clone()
        for name, tensor in source.named_tensors().items()
        if name.startswith("parent.") or name.startswith("compact.")
    }
    return replace(
        source,
        layer_index=0,
        parent_query_weight=replacements["parent.q_proj.weight"],
        parent_key_weight=replacements["parent.k_proj.weight"],
        parent_value_weight=replacements["parent.v_proj.weight"],
        parent_out_weight=replacements["parent.out_proj.weight"],
        parent_query_norm_weight=replacements["parent.q_layernorm.weight"],
        parent_key_norm_weight=replacements["parent.k_layernorm.weight"],
        compact_pristine_query_weight=replacements["compact.q_proj.pristine.weight"],
        compact_pristine_key_weight=replacements["compact.k_proj.pristine.weight"],
        compact_pristine_value_weight=replacements["compact.v_proj.pristine.weight"],
        compact_pristine_out_weight=replacements["compact.out_proj.pristine.weight"],
        compact_final_query_weight=replacements["compact.q_proj.final.weight"],
        compact_final_key_weight=replacements["compact.k_proj.final.weight"],
        compact_final_value_weight=replacements["compact.v_proj.final.weight"],
        compact_final_out_weight=replacements["compact.out_proj.final.weight"],
        compact_query_norm_weight=replacements["compact.q_layernorm.weight"],
        compact_key_norm_weight=replacements["compact.k_layernorm.weight"],
    )


def _publication(tmp_path: Path) -> dict:
    from safetensors.torch import save_file
    from transformers import Lfm2Config, Lfm2ForCausalLM

    source = _float_source()
    config = Lfm2Config(
        vocab_size=16,
        hidden_size=5,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=2,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        norm_eps=source.norm_eps,
    )
    config._attn_implementation = "eager"
    model = Lfm2ForCausalLM(config).float().eval()
    model.model.layers[0].self_attn.load_state_dict(
        {
            "q_proj.weight": source.parent_query_weight,
            "k_proj.weight": source.parent_key_weight,
            "v_proj.weight": source.parent_value_weight,
            "out_proj.weight": source.parent_out_weight,
            "q_layernorm.weight": source.parent_query_norm_weight,
            "k_layernorm.weight": source.parent_key_norm_weight,
        },
        strict=True,
    )
    checkpoint = tmp_path / "parent"
    checkpoint.mkdir()
    _write_json(checkpoint / "config.json", config.to_dict())
    save_file(
        _checkpoint_state(model),
        str(checkpoint / "model.safetensors"),
        metadata={"format": "pt"},
    )
    _tokenizer(checkpoint, vocabulary_size=16)
    loaded = load_lfm2_direct_local_model(
        checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    parent_state_sha = residual_runner.model_state_sha256(loaded)
    parent_identity_sha = fingerprint_lfm2_parent(loaded).identity_sha256
    parent_unique_parameters = runner._unique_parameter_elements(loaded)
    parent_tokenizer_sha = fingerprint_local_tokenizer(checkpoint)
    loaded_bfloat16 = load_lfm2_direct_local_model(
        checkpoint,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    planned_bfloat16_identity_sha = fingerprint_lfm2_parent(
        loaded_bfloat16
    ).identity_sha256
    cached_bfloat16 = residual_runner.load_local_lfm2_model(
        checkpoint,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    bfloat16_binding = {
        "parent_state_sha256": residual_runner.model_state_sha256(cached_bfloat16),
        "parent_identity_sha256": fingerprint_lfm2_parent(
            cached_bfloat16
        ).identity_sha256,
        "parent_unique_parameters": runner._unique_parameter_elements(cached_bfloat16),
        "tokenizer_sha256": parent_tokenizer_sha,
    }
    assert bfloat16_binding["parent_identity_sha256"] == planned_bfloat16_identity_sha
    source = replace(
        source,
        attention_implementation=str(loaded.config._attn_implementation),
    )
    authentication = (
        diagnostic_fixture.fingerprint_lfm2_fused_attention_diagnostic_input(source)
    )
    bank = diagnostic_fixture.compile_lfm2_fused_attention_diagnostic_bank(
        source,
        expected_input_bundle_sha256=authentication["input_bundle_sha256"],
        expected_tensor_sha256={
            name: row["sha256"]
            for name, row in authentication["tensor_records"].items()
        },
    )
    publication = tmp_path / "publication"
    publication.mkdir()
    selected_variants = ("native-selected", "fused-qkvo-final-s2")
    bank_tensors = {}
    candidates = {}
    for variant in selected_variants:
        candidate = bank.candidates[variant]
        state_sha = candidate.receipt["candidate_state_sha256"]
        names = {}
        for state_name, tensor in candidate.state_dict().items():
            name = f"layer.000.state.{state_sha}.{state_name}"
            bank_tensors[name] = tensor
            names[state_name] = name
        candidates[variant] = {
            "variant_id": variant,
            "candidate_class": "base-role-localized",
            "candidate_state_sha256": state_sha,
            "candidate_result_sha256": candidate.receipt["result_sha256"],
            "reconstruction_descriptor_sha256": candidate.reconstruction_descriptor[
                "descriptor_sha256"
            ],
            "canonical_stored_variant_id": variant,
            "byte_identical_candidate_deduplicated": False,
            "tensor_names": names,
            "receipt": dict(candidate.receipt),
            "reconstruction_descriptor": dict(candidate.reconstruction_descriptor),
        }
    bank_path = publication / producer.BANK_FILENAME
    save_file(bank_tensors, str(bank_path), metadata={"format": "pt"})
    layer = {
        "layer_index": 0,
        "parent_module_path": "model.layers.0.self_attn",
        "parent_attention_module_sha256": _module_state_sha256(
            loaded.model.layers[0].self_attn
        ),
        "final_attention_module_sha256": "f" * 64,
        "input_authentication": dict(authentication),
        "bank_receipt": dict(bank.receipt),
        "base_variant_ids": list(selected_variants),
        "explicit_group_variant_ids": [],
        "variant_ids": list(selected_variants),
        "candidates": candidates,
        "unique_stored_candidate_state_count": 2,
        "candidate_variant_count": 2,
    }
    schedule = producer._schedule_document([layer])
    records = {
        name: {
            "shape": [int(size) for size in tensor.shape],
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": lfm2_authenticated_tensor_sha256(tensor),
        }
        for name, tensor in sorted(bank_tensors.items())
    }
    inventory_sha = canonical_sha256(runner._checkpoint_inventory(checkpoint))
    plan_body = {
        "schema": producer.PLAN_SCHEMA,
        "status": producer.PLAN_STATUS,
        "producer_schema": producer.PRODUCER_SCHEMA,
        "source_authentication": {
            "synthetic_test": True,
            "parent_lfm2_identity_sha256": planned_bfloat16_identity_sha,
            "parent_tokenizer_sha256": parent_tokenizer_sha,
        },
        "parent_checkpoint_inventory_sha256": inventory_sha,
        "final_checkpoint_inventory_sha256": "e" * 64,
        "parent_tokenizer_sha256": parent_tokenizer_sha,
        "parent_model_state_sha256": fingerprint_lfm2_direct_local_state(loaded),
        "final_model_state_sha256": "d" * 64,
        "materialization_dtype": "float32",
        "bank": {
            "filename": producer.BANK_FILENAME,
            "file_sha256": _file_sha256(bank_path),
            "tensor_records": records,
            "tensor_count": len(records),
            "stored_tensor_elements": sum(
                value.numel() for value in bank_tensors.values()
            ),
        },
        "attention_layers": [layer],
        "attention_layer_indices": [0],
        "attention_layer_count": 1,
        "schedule": dict(schedule),
        "explicit_group_selection_document": None,
        "group_substitutions_included_only_when_explicitly_selected": True,
        "group_substitution_cartesian_expansion_performed": False,
        "byte_identical_candidate_states_stored_once": True,
        "diagnostic_parent_shell_only": True,
        "standalone_child_compression_claimed": False,
        "numeric_acceptance_gate_defined": False,
        "operation_audit": {"training_steps": 0},
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    plan_path = publication / producer.PLAN_FILENAME
    _write_json(plan_path, plan)
    receipt_body = {
        "schema": producer.PRODUCER_SCHEMA,
        "status": producer.PRODUCER_STATUS,
        "plan_filename": producer.PLAN_FILENAME,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": _file_sha256(plan_path),
        "bank_filename": producer.BANK_FILENAME,
        "bank_file_sha256": _file_sha256(bank_path),
        "component_producer_receipt_sha256": "c" * 64,
        "source_run_result_sha256": "b" * 64,
        "parent_checkpoint_inventory_sha256": inventory_sha,
        "final_checkpoint_inventory_sha256": "e" * 64,
        "parent_tokenizer_sha256": parent_tokenizer_sha,
        "attention_layer_count": 1,
        "candidate_schedule_count": schedule["candidate_count"],
        "unique_stored_candidate_state_count": 2,
        "fresh_source_receipt_authentication_performed": True,
        "fresh_archive_authentication_performed": True,
        "direct_local_checkpoint_loading_used": True,
        "human_authored_module_or_tensor_pins_accepted": False,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    receipt_path = publication / producer.PRODUCER_RECEIPT_FILENAME
    _write_json(receipt_path, receipt)
    return {
        "checkpoint": checkpoint,
        "publication": publication,
        "plan": plan,
        "receipt": receipt,
        "planned_bfloat16_identity_sha256": planned_bfloat16_identity_sha,
        "bfloat16_binding": bfloat16_binding,
        "parent_binding": {
            "parent_state_sha256": parent_state_sha,
            "parent_identity_sha256": parent_identity_sha,
            "parent_unique_parameters": parent_unique_parameters,
            "tokenizer_sha256": parent_tokenizer_sha,
        },
    }


def test_balanced_shards_are_disjoint_complete_and_tamper_evident() -> None:
    body = {
        "schema": producer.SCHEDULE_SCHEMA,
        "status": "NON-FINAL-FUSED-ATTENTION-DIAGNOSTIC-SCHEDULE",
        "candidate_ids": [f"candidate-{index}" for index in range(11)],
        "candidates": [{"candidate_id": f"candidate-{index}"} for index in range(11)],
        "candidate_count": 11,
        "mode_counts": {},
        "deduplicated_candidate_aliases": [],
        "deduplication_key": "test",
        "isolated_layer_variants_scheduled_first": True,
        "group_substitutions_default_cartesian_expansion_performed": False,
    }
    schedule = {**body, "schedule_sha256": canonical_sha256(body)}
    shards = [
        runner._schedule_shard(schedule, shard_count=3, shard_index=index)
        for index in range(3)
    ]
    selected = [
        candidate["candidate_id"]
        for shard in shards
        for candidate in runner._authenticate_schedule_shard(schedule, shard)
    ]
    assert selected == schedule["candidate_ids"]
    assert len(selected) == len(set(selected))
    assert [value["selected_candidate_count"] for value in shards] == [4, 4, 3]
    assert all(
        value["complete_full_schedule_sweep_claimed"] is False for value in shards
    )

    tampered = copy.deepcopy(shards[0])
    tampered["selected_candidate_ids"][0] = "wrong"
    with pytest.raises(runner.Lfm2FusedAttentionDiagnosticRunError, match="differs"):
        runner._authenticate_schedule_shard(schedule, tampered)


def test_parent_envelope_binding_rejects_missing_or_wrong_identity_tokenizer_and_short_cache(
    tmp_path: Path,
) -> None:
    fixture = _publication(tmp_path)
    binding = fixture["bfloat16_binding"]
    envelope = {
        "envelope_sha256": "a" * 64,
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "inference_dtype": "bfloat16",
        **binding,
        "responses": [{"id": "one"}],
        "execution_audit": {
            "parent_model_loads": 1,
            "parent_generate_invocations": 1,
            "response_cache_publications": 1,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "hub_loader_used": False,
            "network_access": False,
            "parent_state_unchanged": True,
            "parent_parameter_count_unchanged": True,
        },
    }

    authenticated = runner._authenticate_parent_envelope_binding(
        parent_cache=envelope,
        parent_checkpoint=fixture["checkpoint"],
        plan=fixture["plan"],
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
    )
    assert authenticated["parent_identity_sha256"] == binding["parent_identity_sha256"]
    assert authenticated["local_parent_reauthentication_generate_calls"] == 0
    assert authenticated["planned_bfloat16_identity_comparison_performed"] is True
    assert authenticated["planned_bfloat16_identity_match"] is True

    missing_identity = copy.deepcopy(envelope)
    missing_identity.pop("parent_identity_sha256")
    with pytest.raises(runner.Lfm2FusedAttentionDiagnosticRunError, match="identity"):
        runner._authenticate_parent_envelope_binding(
            parent_cache=missing_identity,
            parent_checkpoint=fixture["checkpoint"],
            plan=fixture["plan"],
            maximum_prompt_tokens=1024,
            maximum_new_tokens=256,
        )

    wrong_identity = copy.deepcopy(envelope)
    wrong_identity["parent_identity_sha256"] = "0" * 64
    with pytest.raises(runner.Lfm2FusedAttentionDiagnosticRunError, match="differs"):
        runner._authenticate_parent_envelope_binding(
            parent_cache=wrong_identity,
            parent_checkpoint=fixture["checkpoint"],
            plan=fixture["plan"],
            maximum_prompt_tokens=1024,
            maximum_new_tokens=256,
        )

    missing_tokenizer = copy.deepcopy(envelope)
    missing_tokenizer.pop("tokenizer_sha256")
    with pytest.raises(runner.Lfm2FusedAttentionDiagnosticRunError, match="tokenizer"):
        runner._authenticate_parent_envelope_binding(
            parent_cache=missing_tokenizer,
            parent_checkpoint=fixture["checkpoint"],
            plan=fixture["plan"],
            maximum_prompt_tokens=1024,
            maximum_new_tokens=256,
        )

    wrong_tokenizer = copy.deepcopy(envelope)
    wrong_tokenizer["tokenizer_sha256"] = "0" * 64
    with pytest.raises(runner.Lfm2FusedAttentionDiagnosticRunError, match="diverged"):
        runner._authenticate_parent_envelope_binding(
            parent_cache=wrong_tokenizer,
            parent_checkpoint=fixture["checkpoint"],
            plan=fixture["plan"],
            maximum_prompt_tokens=1024,
            maximum_new_tokens=256,
        )

    short_cache = copy.deepcopy(envelope)
    short_cache["maximum_new_tokens"] = 64
    with pytest.raises(runner.Lfm2FusedAttentionDiagnosticRunError, match="limits"):
        runner._authenticate_parent_envelope_binding(
            parent_cache=short_cache,
            parent_checkpoint=fixture["checkpoint"],
            plan=fixture["plan"],
            maximum_prompt_tokens=1024,
            maximum_new_tokens=256,
        )


def test_runner_saves_fresh_reloads_and_emits_manual_review_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _publication(tmp_path)
    prompts = {
        "prompts": [
            {"id": "one", "prompt": "hello", "category": "smoke"},
            {"id": "two", "prompt": "hello hello", "category": "smoke"},
        ]
    }
    prompts_path = tmp_path / "prompts.json"
    _write_json(prompts_path, prompts)
    rows = [dict(value) for value in runner.load_review_prompts(prompts_path)]
    responses = [
        {
            "id": row["id"],
            "response": f"parent-{row['id']}",
            "response_token_ids": [4, 2],
            "diagnostics": {"cached": True},
        }
        for row in rows
    ]
    envelope_body = {
        "schema": PARENT_ENVELOPE_SCHEMA,
        "status": "complete-one-time-parent-response-cache",
        "prompt_set_sha256": canonical_sha256(rows),
        "prompt_count": len(rows),
        "maximum_prompt_tokens": 8192,
        "maximum_new_tokens": 256,
        "inference_dtype": "float32",
        **fixture["parent_binding"],
        "responses": responses,
        "response_bank_sha256": canonical_sha256(responses),
        "execution_audit": {
            "parent_model_loads": 1,
            "parent_generate_invocations": len(rows),
            "response_cache_publications": 1,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "hub_loader_used": False,
            "network_access": False,
            "parent_state_unchanged": True,
            "parent_parameter_count_unchanged": True,
        },
    }
    envelope = {
        **envelope_body,
        "envelope_sha256": canonical_sha256(envelope_body),
    }
    envelope_path = tmp_path / "parent-envelope.json"
    _write_json(envelope_path, envelope)

    calls = []

    def fake_generate(_model, _tokenizer, prompt_rows, **kwargs):
        assert [value["id"] for value in prompt_rows] == ["one", "two"]
        assert kwargs["maximum_prompt_tokens"] == 8192
        assert kwargs["maximum_new_tokens"] == 256
        calls.append(True)
        return [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "response": f"child-{row['id']}",
                "response_token_ids": [4, 2],
                "diagnostics": {"manual": True},
            }
            for row in prompt_rows
        ]

    monkeypatch.setattr(runner, "_generate_rows", fake_generate)
    publication = fixture["publication"]
    result = runner.run(
        run_directory=tmp_path / "run",
        parent_checkpoint=fixture["checkpoint"],
        producer_receipt=publication / producer.PRODUCER_RECEIPT_FILENAME,
        expected_producer_receipt_sha256=fixture["receipt"]["receipt_sha256"],
        expected_producer_receipt_file_sha256=_file_sha256(
            publication / producer.PRODUCER_RECEIPT_FILENAME
        ),
        fused_plan=publication / producer.PLAN_FILENAME,
        expected_fused_plan_sha256=fixture["plan"]["plan_sha256"],
        expected_fused_plan_file_sha256=_file_sha256(
            publication / producer.PLAN_FILENAME
        ),
        fused_bank=publication / producer.BANK_FILENAME,
        expected_fused_bank_file_sha256=_file_sha256(
            publication / producer.BANK_FILENAME
        ),
        prompts_json=prompts_path,
        expected_prompts_file_sha256=_file_sha256(prompts_path),
        parent_envelope=envelope_path,
        expected_parent_envelope_sha256=envelope["envelope_sha256"],
        expected_parent_envelope_file_sha256=_file_sha256(envelope_path),
        device="cpu",
        dtype="float32",
        maximum_prompt_tokens=8192,
        maximum_new_tokens=256,
        checkpoint_retention="receipts-only",
    )
    assert len(calls) == result["reviewed_candidate_count"] == 2
    assert result["complete_full_schedule_sweep_claimed"] is True
    assert result["retained_checkpoint_paths"] == []
    assert (
        result["parent_envelope_binding"]["parent_identity_sha256"]
        == fixture["parent_binding"]["parent_identity_sha256"]
    )
    assert (
        result["parent_envelope_binding"][
            "local_parent_reauthentication_generate_calls"
        ]
        == 0
    )
    assert not any((tmp_path / "run" / "mixed-checkpoints").iterdir())
    assert all(
        value["responses"][0]["parent_response"] == "parent-one"
        for value in result["candidate_reviews"]
    )
    assert (
        result["evaluation_policy"]["numeric_metrics_have_decision_authority"] is False
    )


def test_runner_rejects_run_nested_in_parent_before_mutating_source(
    tmp_path: Path,
) -> None:
    fixture = _publication(tmp_path)
    publication = fixture["publication"]
    prompts_path = tmp_path / "prompts.json"
    envelope_path = tmp_path / "parent-envelope.json"
    _write_json(prompts_path, {"prompts": []})
    _write_json(envelope_path, {"not": "loaded"})
    parent_inventory_before = runner._checkpoint_inventory(fixture["checkpoint"])
    forbidden = fixture["checkpoint"] / "nested-diagnostic-run"

    with pytest.raises(
        runner.Lfm2FusedAttentionDiagnosticRunError,
        match="outside every authenticated input directory",
    ):
        runner.run(
            run_directory=forbidden,
            parent_checkpoint=fixture["checkpoint"],
            producer_receipt=publication / producer.PRODUCER_RECEIPT_FILENAME,
            expected_producer_receipt_sha256=fixture["receipt"]["receipt_sha256"],
            expected_producer_receipt_file_sha256=_file_sha256(
                publication / producer.PRODUCER_RECEIPT_FILENAME
            ),
            fused_plan=publication / producer.PLAN_FILENAME,
            expected_fused_plan_sha256=fixture["plan"]["plan_sha256"],
            expected_fused_plan_file_sha256=_file_sha256(
                publication / producer.PLAN_FILENAME
            ),
            fused_bank=publication / producer.BANK_FILENAME,
            expected_fused_bank_file_sha256=_file_sha256(
                publication / producer.BANK_FILENAME
            ),
            prompts_json=prompts_path,
            expected_prompts_file_sha256="0" * 64,
            parent_envelope=envelope_path,
            expected_parent_envelope_sha256="0" * 64,
            expected_parent_envelope_file_sha256="0" * 64,
            device="cpu",
            dtype="float32",
            maximum_prompt_tokens=256,
            maximum_new_tokens=256,
            checkpoint_retention="receipts-only",
        )

    assert not forbidden.exists()
    assert (
        runner._checkpoint_inventory(fixture["checkpoint"]) == parent_inventory_before
    )
