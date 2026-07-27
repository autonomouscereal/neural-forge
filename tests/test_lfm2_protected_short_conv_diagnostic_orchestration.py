from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments import prepare_lfm2_protected_short_conv_diagnostic as producer
from experiments import run_lfm2_progressive_component_ablation as b038
from experiments import run_lfm2_protected_short_conv_diagnostic as runner
from experiments.review_lfm2_semantic_frontier import PARENT_ENVELOPE_SCHEMA
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_direct_local import (
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (
    _module_state_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)
from tests.test_prepare_lfm2_progressive_component_ablation import _checkpoint


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _config(*, parent: bool):
    transformers = pytest.importorskip("transformers")
    return transformers.Lfm2Config(
        vocab_size=19,
        hidden_size=8 if parent else 6,
        intermediate_size=12 if parent else 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        block_auto_adjust_ff_dim=False,
        layer_types=["conv", "conv"],
        tie_word_embeddings=True,
        conv_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _publication_fixture(tmp_path: Path) -> dict:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    torch.manual_seed(661)
    parent_live = transformers.Lfm2ForCausalLM(_config(parent=True)).eval()
    final_live = transformers.Lfm2ForCausalLM(_config(parent=False)).eval()
    parent_checkpoint = tmp_path / "parent-checkpoint"
    final_checkpoint = tmp_path / "final-checkpoint"
    _checkpoint(parent_checkpoint, parent_live, vocabulary_size=19)
    _checkpoint(final_checkpoint, final_live, vocabulary_size=19)
    parent = load_lfm2_direct_local_model(
        parent_checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    final = load_lfm2_direct_local_model(
        final_checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    parent_bfloat16 = load_lfm2_direct_local_model(
        parent_checkpoint,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    parent_identity_sha = fingerprint_lfm2_parent(parent_bfloat16).identity_sha256
    encoder = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    decoder = torch.linalg.pinv(encoder, rcond=1e-12)
    publication = tmp_path / "b038-publication"
    publication.mkdir()
    bridge_path = publication / b038.BRIDGE_FILENAME
    safetensors.save_file(
        {
            "foundation.producer_encoder": encoder,
            "foundation.analytic_output_decoder": decoder,
        },
        str(bridge_path),
        metadata={"format": "pt"},
    )
    bridge_sha = _file_sha256(bridge_path)
    parent_inventory = b038._checkpoint_inventory(parent_checkpoint)
    final_inventory = b038._checkpoint_inventory(final_checkpoint)
    parent_inventory_sha = canonical_sha256(parent_inventory)
    final_inventory_sha = canonical_sha256(final_inventory)
    source_run_sha = _sha("synthetic-a0019-source-run")
    source_run_file_sha = _sha("synthetic-a0019-source-run-file")
    components = []
    fold_receipts = {}
    for layer_index in (0, 1):
        component_id = f"layer-{layer_index:03d}-short_conv"
        path = f"model.layers.{layer_index}.conv"
        fold_sha = _sha(f"{component_id}-fold")
        fold_receipts[component_id] = {
            "receipt_sha256": fold_sha,
            "receipt_schema": "synthetic-fold-receipt-v1",
            "source_run_locator": f"synthetic.{component_id}",
            "receipt_seal_freshly_verified": True,
        }
        components.append(
            {
                "component_id": component_id,
                "kind": "short_conv",
                "granularity": "whole-subsystem",
                "parent_module_path": path,
                "compact_module_path": path,
                "parent_module_sha256": _module_state_sha256(
                    parent.get_submodule(path)
                ),
                "compact_module_sha256": _module_state_sha256(
                    final.get_submodule(path)
                ),
                "input_encoder_tensor": "foundation.producer_encoder",
                "output_decoder_tensor": "foundation.analytic_output_decoder",
                "output_domain_decoder_tensor": None,
                "input_encoder_sha256": lfm2_authenticated_tensor_sha256(encoder),
                "output_decoder_sha256": lfm2_authenticated_tensor_sha256(decoder),
                "output_domain_decoder_sha256": None,
                "source_fold_receipt_sha256": fold_sha,
                "source_compact_checkpoint_sha256": final_inventory_sha,
                "physical_parameter_accounting": {"synthetic_source_row": True},
                "require_parameter_reduction": True,
            }
        )
    plan_body = {
        "schema": b038.PLAN_SCHEMA,
        "status": b038.AUTOMATIC_PLAN_STATUS,
        "producer_schema": b038.PLAN_PRODUCER_SCHEMA,
        "source_authentication": {
            "source_run_result_sha256": source_run_sha,
            "source_run_receipt_file_sha256": source_run_file_sha,
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "parent_lfm2_identity_sha256": parent_identity_sha,
            "compact_checkpoint_inventory_sha256": final_inventory_sha,
            "compact_complete_model_state_sha256": (
                fingerprint_lfm2_direct_local_state(final)
            ),
            "structural_body_checkpoint_inventory_sha256": _sha(
                "structural-body-inventory"
            ),
            "seed_archive_manifest_sha256": _sha("seed-manifest"),
            "candidate_archive_manifest_sha256": _sha("candidate-manifest"),
            "seed_archive_fresh_reload_verified": True,
            "candidate_archive_fresh_reload_verified": True,
            "component_fold_receipts": fold_receipts,
        },
        "parent_model_state_sha256": fingerprint_lfm2_direct_local_state(parent),
        "compact_model_state_sha256": fingerprint_lfm2_direct_local_state(final),
        "parent_checkpoint_inventory_sha256": parent_inventory_sha,
        "compact_checkpoint_inventory_sha256": final_inventory_sha,
        "materialization_dtype": "float32",
        "realized_source_parameter_ratio": (
            sum(value.numel() for value in parent.parameters())
            / sum(value.numel() for value in final.parameters())
        ),
        "bridge_archive": {
            "schema": b038.BRIDGE_ARCHIVE_SCHEMA,
            "filename": b038.BRIDGE_FILENAME,
            "file_sha256": bridge_sha,
            "tensor_records": {
                "foundation.producer_encoder": {
                    "shape": list(encoder.shape),
                    "dtype": "float64",
                    "sha256": lfm2_authenticated_tensor_sha256(encoder),
                },
                "foundation.analytic_output_decoder": {
                    "shape": list(decoder.shape),
                    "dtype": "float64",
                    "sha256": lfm2_authenticated_tensor_sha256(decoder),
                },
            },
        },
        "component_granularity": "whole-subsystem",
        "q_k_v_o_or_individual_head_isolation_claimed": False,
        "components": components,
        "excluded_components": [],
        "all_included_components_require_literal_parameter_reduction_true": True,
        "diagnostic_hybrid_only": True,
        "standalone_compression_claimed": False,
        "operation_audit": {"training_steps": 0, "model_forward_calls": 0},
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    plan_path = publication / b038.PLAN_FILENAME
    _write_json(plan_path, plan)
    receipt_body = {
        "schema": b038.PLAN_PRODUCER_SCHEMA,
        "status": b038.PRODUCER_RECEIPT_STATUS,
        "plan_filename": b038.PLAN_FILENAME,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": _file_sha256(plan_path),
        "bridge_filename": b038.BRIDGE_FILENAME,
        "bridge_file_sha256": bridge_sha,
        "source_run_result_sha256": source_run_sha,
        "parent_checkpoint_inventory_sha256": parent_inventory_sha,
        "compact_checkpoint_inventory_sha256": final_inventory_sha,
        "component_count": len(components),
        "excluded_component_count": 0,
        "component_granularity": "whole-subsystem",
        "automatic_topology_and_component_derivation_performed": True,
        "fresh_source_receipt_authentication_performed": True,
        "fresh_archive_authentication_performed": True,
        "human_authored_module_or_receipt_pins_accepted": False,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    receipt_path = publication / b038.PRODUCER_RECEIPT_FILENAME
    _write_json(receipt_path, receipt)
    return {
        "parent_checkpoint": parent_checkpoint,
        "final_checkpoint": final_checkpoint,
        "parent_inventory_sha": parent_inventory_sha,
        "final_inventory_sha": final_inventory_sha,
        "publication": publication,
        "component_plan": plan,
        "component_plan_file_sha": _file_sha256(plan_path),
        "component_receipt": receipt,
        "component_receipt_file_sha": _file_sha256(receipt_path),
        "component_bridge_file_sha": bridge_sha,
        "source_run_sha": source_run_sha,
        "source_run_file_sha": source_run_file_sha,
    }


def _producer_arguments(fixture: dict, *, output: Path) -> dict:
    return {
        "output_directory": output,
        "component_publication_directory": fixture["publication"],
        "expected_component_producer_receipt_sha256": fixture["component_receipt"][
            "receipt_sha256"
        ],
        "expected_component_producer_receipt_file_sha256": fixture[
            "component_receipt_file_sha"
        ],
        "expected_component_plan_sha256": fixture["component_plan"]["plan_sha256"],
        "expected_component_plan_file_sha256": fixture["component_plan_file_sha"],
        "expected_component_bridges_file_sha256": fixture["component_bridge_file_sha"],
        "expected_a0019_source_run_result_sha256": fixture["source_run_sha"],
        "expected_a0019_source_run_file_sha256": fixture["source_run_file_sha"],
        "parent_checkpoint": fixture["parent_checkpoint"],
        "expected_parent_checkpoint_inventory_sha256": fixture["parent_inventory_sha"],
        "final_checkpoint": fixture["final_checkpoint"],
        "expected_final_checkpoint_inventory_sha256": fixture["final_inventory_sha"],
    }


def _prepare(tmp_path: Path) -> dict:
    fixture = _publication_fixture(tmp_path)
    output = tmp_path / "protected-publication"
    receipt = producer.prepare(**_producer_arguments(fixture, output=output))
    plan_path = output / producer.PLAN_FILENAME
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    return {
        **fixture,
        "output": output,
        "protected_receipt": receipt,
        "protected_receipt_file_sha": _file_sha256(
            output / producer.PRODUCER_RECEIPT_FILENAME
        ),
        "protected_plan": plan,
        "protected_plan_file_sha": _file_sha256(plan_path),
        "protected_archive_file_sha": _file_sha256(output / producer.ARCHIVE_FILENAME),
    }


def _review_inputs(tmp_path: Path, fixture: dict) -> dict:
    root = tmp_path / "runtime-evidence"
    root.mkdir()
    prompts_path = root / "prompts.json"
    prompt_rows = [
        {
            "id": f"prompt-{index:02d}",
            "prompt": f"Question number {index}?",
            "category": "synthetic",
        }
        for index in range(16)
    ]
    _write_json(prompts_path, {"prompts": prompt_rows})
    normalized = [
        {
            **row,
            "context": None,
            "parent_response": None,
        }
        for row in prompt_rows
    ]
    responses = [
        {
            "id": row["id"],
            "category": row["category"],
            "response": f"cached-parent-{row['id']}",
            "response_token_ids": [4, 2],
            "diagnostics": {"tokens": 2, "interpretation": "diagnostic-only"},
        }
        for row in normalized
    ]
    binding = fixture["protected_plan"]["parent_cache_binding"]
    envelope_body = {
        "schema": PARENT_ENVELOPE_SCHEMA,
        "status": "complete-one-time-parent-response-cache",
        "parent_model_directory": str(fixture["parent_checkpoint"].resolve()),
        "tokenizer_directory": str(fixture["parent_checkpoint"].resolve()),
        "prompt_source": str(prompts_path.resolve()),
        "prompt_set_sha256": canonical_sha256(normalized),
        "prompt_count": 16,
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "inference_device": "cpu",
        "inference_dtype": "bfloat16",
        "parent_state_sha256": binding["parent_state_sha256"],
        "parent_identity_sha256": binding["parent_identity_sha256"],
        "parent_unique_parameters": binding["parent_unique_parameters"],
        "tokenizer_sha256": binding["tokenizer_sha256"],
        "responses": responses,
        "response_bank_sha256": canonical_sha256(responses),
        "execution_audit": {
            "parent_model_loads": 1,
            "parent_generate_invocations": 16,
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
    envelope_path = root / "parent-envelope.json"
    _write_json(envelope_path, envelope)
    return {
        "root": root,
        "prompts": prompts_path,
        "prompts_file_sha": _file_sha256(prompts_path),
        "envelope": envelope_path,
        "envelope_sha": envelope["envelope_sha256"],
        "envelope_file_sha": _file_sha256(envelope_path),
    }


def _runner_arguments(
    fixture: dict,
    review: dict,
    *,
    run_directory: Path,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict:
    return {
        "run_directory": run_directory,
        "parent_checkpoint": fixture["parent_checkpoint"],
        "expected_parent_checkpoint_inventory_sha256": fixture["parent_inventory_sha"],
        "final_checkpoint": fixture["final_checkpoint"],
        "expected_final_checkpoint_inventory_sha256": fixture["final_inventory_sha"],
        "component_publication_directory": fixture["publication"],
        "expected_component_producer_receipt_sha256": fixture["component_receipt"][
            "receipt_sha256"
        ],
        "expected_component_producer_receipt_file_sha256": fixture[
            "component_receipt_file_sha"
        ],
        "expected_component_plan_sha256": fixture["component_plan"]["plan_sha256"],
        "expected_component_plan_file_sha256": fixture["component_plan_file_sha"],
        "expected_component_bridges_file_sha256": fixture["component_bridge_file_sha"],
        "expected_a0019_source_run_result_sha256": fixture["source_run_sha"],
        "expected_a0019_source_run_file_sha256": fixture["source_run_file_sha"],
        "producer_receipt": fixture["output"] / producer.PRODUCER_RECEIPT_FILENAME,
        "expected_producer_receipt_sha256": fixture["protected_receipt"][
            "receipt_sha256"
        ],
        "expected_producer_receipt_file_sha256": fixture["protected_receipt_file_sha"],
        "protected_plan": fixture["output"] / producer.PLAN_FILENAME,
        "expected_protected_plan_sha256": fixture["protected_plan"]["plan_sha256"],
        "expected_protected_plan_file_sha256": fixture["protected_plan_file_sha"],
        "protected_archive": fixture["output"] / producer.ARCHIVE_FILENAME,
        "expected_protected_archive_file_sha256": fixture["protected_archive_file_sha"],
        "prompts_json": review["prompts"],
        "expected_prompts_file_sha256": review["prompts_file_sha"],
        "parent_envelope": review["envelope"],
        "expected_parent_envelope_sha256": review["envelope_sha"],
        "expected_parent_envelope_file_sha256": review["envelope_file_sha"],
        "device": "cpu",
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "checkpoint_retention": "receipts-only",
        "shard_count": shard_count,
        "shard_index": shard_index,
    }


def test_producer_publishes_fixed_algebra_diagnostics_and_counted_adapters(
    tmp_path: Path,
) -> None:
    fixture = _prepare(tmp_path)
    plan = fixture["protected_plan"]
    assert plan["schedule"]["candidate_ids"] == [
        "cached-parent-reference",
        "layer-000-parent-input-roundtrip",
        "layer-000-parent-output-roundtrip",
        "layer-000-parent-both-roundtrip",
        "layer-000-dense-compact-negative-control",
        "layer-000-standalone-child-stem-adapter-control",
        "layer-001-parent-both-roundtrip",
    ]
    assert plan["candidate_count"] == 6
    assert plan["schedule"]["protected_parent_shell_count"] == 4
    assert plan["schedule"]["dense_compact_negative_control_count"] == 1
    assert plan["schedule"]["protected_standalone_adapter_control_count"] == 1
    rows = {row["candidate_id"]: row for row in plan["candidates"]}
    standalone = rows["layer-000-standalone-child-stem-adapter-control"]
    accounting = standalone["parent_shell_wrapper_physical_parameter_accounting"]
    assert accounting["protected_standalone_core_parameter_elements"] == 216
    assert accounting["explicitly_counted_bridge_parameter_elements"] == 96
    assert accounting["installed_parent_shell_parameter_elements"] == 312
    assert accounting["original_parent_component_parameter_elements"] == 280
    assert accounting["physical_parameter_reduction_from_parent"] == -32
    assert standalone["explicit_interface_adapter_parameter_elements"] == 96
    assert standalone["hidden_sidecar_parameter_elements"] == 0
    assert set(standalone["wrapper_adapter_tensor_names"]) == {
        "input_bridge.weight",
        "output_bridge.weight",
    }
    assert plan["operation_audit"]["training_steps"] == 0
    assert plan["operation_audit"]["model_generate_calls"] == 0
    assert plan["operation_audit"]["network_access"] is False


def test_schedule_shards_are_disjoint_complete_and_tamper_evident(
    tmp_path: Path,
) -> None:
    schedule = _prepare(tmp_path)["protected_plan"]["schedule"]
    shards = [
        runner._schedule_shard(schedule, shard_count=3, shard_index=index)
        for index in range(3)
    ]
    selected = [set(row["selected_candidate_ids"]) for row in shards]
    assert not (selected[0] & selected[1])
    assert not (selected[0] & selected[2])
    assert not (selected[1] & selected[2])
    assert set().union(*selected) == set(schedule["candidate_ids"])
    tampered = copy.deepcopy(shards[0])
    tampered["selected_candidate_ids"].append("forged")
    with pytest.raises(runner.Lfm2ProtectedShortConvDiagnosticRunError):
        runner._authenticate_schedule_shard(schedule, tampered)


def test_runner_fresh_reloads_all_candidates_and_reuses_only_cached_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepare(tmp_path)
    review = _review_inputs(tmp_path, fixture)
    calls = []

    def fake_generate(
        model,
        tokenizer,
        prompts,
        *,
        maximum_prompt_tokens: int,
        maximum_new_tokens: int,
    ):
        calls.append((len(prompts), maximum_prompt_tokens, maximum_new_tokens))
        return [
            {
                **dict(prompt),
                "response": f"candidate-{prompt['id']}",
                "response_token_ids": [4, 2],
                "diagnostics": {"tokens": 2, "interpretation": "diagnostic-only"},
            }
            for prompt in prompts
        ]

    monkeypatch.setattr(runner, "_generate_rows", fake_generate)
    run_root = tmp_path / "run"
    result = runner.run(**_runner_arguments(fixture, review, run_directory=run_root))
    assert (
        result["status"]
        == "NON-FINAL-PROTECTED-SHORT-CONV-FULL-REVIEW-EVIDENCE-COMPLETE"
    )
    assert calls == [(16, 1024, 256)] * 6
    assert result["prompt_count"] == 16
    assert result["operation_audit"]["parent_model_generate_calls"] == 0
    assert result["operation_audit"]["hybrid_model_generate_passes"] == 6
    assert result["operation_audit"]["cached_parent_references_reused"] == 1
    reference = result["candidate_reviews"][0]
    assert reference["candidate_runtime_executed"] is False
    assert reference["response_origin"] == "authenticated-bfloat16-parent-cache"
    standalone = next(
        row
        for row in result["candidate_reviews"]
        if row["candidate_type"] == "protected-standalone-adapter-control"
    )
    installation = standalone["installation_receipt"]
    assert installation["realized_parameter_reduction"] == -32
    assert installation["explicit_interface_adapter_parameter_elements"] == 96
    assert installation["hidden_sidecar_parameter_elements"] == 0
    assert standalone["fresh_cache_per_candidate_and_prompt"] is True
    assert list((run_root / "differential-checkpoints").iterdir()) == []
    assert len(list((run_root / "manual-review").glob("*.json"))) == 8


def test_external_pins_horizon_and_output_containment_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _publication_fixture(tmp_path)
    arguments = _producer_arguments(
        fixture,
        output=tmp_path / "wrong-source-pin-output",
    )
    arguments["expected_a0019_source_run_result_sha256"] = "0" * 64
    with pytest.raises(Exception, match="A0019"):
        producer.prepare(**arguments)
    assert not (tmp_path / "wrong-source-pin-output").exists()

    nested = fixture["parent_checkpoint"] / "forbidden-output"
    with pytest.raises(
        producer.Lfm2ProtectedShortConvPlanProducerError,
        match="lexically separate",
    ):
        producer.prepare(**_producer_arguments(fixture, output=nested))
    assert not nested.exists()

    second = tmp_path / "second"
    second.mkdir()
    prepared = _prepare(second)
    review = _review_inputs(second, prepared)
    run_arguments = _runner_arguments(
        prepared,
        review,
        run_directory=tmp_path / "bad-horizon-run",
    )
    run_arguments["maximum_new_tokens"] = 64
    with pytest.raises(
        runner.Lfm2ProtectedShortConvDiagnosticRunError,
        match="1024-input/256-output",
    ):
        runner.run(**run_arguments)
    assert not (tmp_path / "bad-horizon-run").exists()
