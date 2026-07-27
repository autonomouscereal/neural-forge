from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments import prepare_lfm2_fused_body_diagnostic as producer
from experiments import run_lfm2_fused_body_diagnostic as runner
from experiments import run_lfm2_progressive_component_ablation as b038
from experiments.review_lfm2_semantic_frontier import (
    PARENT_ENVELOPE_SCHEMA,
    _load_parent_envelope,
)
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
        layer_types=["conv", "full_attention"],
        tie_word_embeddings=True,
        conv_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _publication_fixture(tmp_path: Path) -> dict:
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    torch.manual_seed(503)
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
    source_run_sha = _sha("synthetic-b038-source-run")
    components = []
    fold_receipts = {}
    specifications = (
        ("layer-000-short_conv", "short_conv", "model.layers.0.conv"),
        ("layer-000-mlp", "mlp", "model.layers.0.feed_forward"),
        ("layer-001-mlp", "mlp", "model.layers.1.feed_forward"),
    )
    for component_id, kind, path in specifications:
        fold_sha = _sha(f"{component_id}-fold")
        fold_receipts[component_id] = {
            "receipt_sha256": fold_sha,
            "receipt_schema": "synthetic-fold-receipt-v1",
            "source_run_locator": f"synthetic.{component_id}",
            "receipt_seal_freshly_verified": True,
        }
        parent_module = parent.get_submodule(path)
        final_module = final.get_submodule(path)
        components.append(
            {
                "component_id": component_id,
                "kind": kind,
                "granularity": "whole-subsystem",
                "parent_module_path": path,
                "compact_module_path": path,
                "parent_module_sha256": _module_state_sha256(parent_module),
                "compact_module_sha256": _module_state_sha256(final_module),
                "input_encoder_tensor": "foundation.producer_encoder",
                "output_decoder_tensor": "foundation.analytic_output_decoder",
                "output_domain_decoder_tensor": None,
                "input_encoder_sha256": lfm2_authenticated_tensor_sha256(encoder),
                "output_decoder_sha256": lfm2_authenticated_tensor_sha256(decoder),
                "output_domain_decoder_sha256": None,
                "source_fold_receipt_sha256": fold_sha,
                "source_compact_checkpoint_sha256": final_inventory_sha,
                "physical_parameter_accounting": {
                    "diagnostic_source_row": True,
                },
                "require_parameter_reduction": True,
            }
        )
    plan_body = {
        "schema": b038.PLAN_SCHEMA,
        "status": b038.AUTOMATIC_PLAN_STATUS,
        "producer_schema": b038.PLAN_PRODUCER_SCHEMA,
        "source_authentication": {
            "source_run_result_sha256": source_run_sha,
            "source_run_receipt_file_sha256": _sha("source-run-file"),
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
        "operation_audit": {
            "training_steps": 0,
            "model_forward_calls": 0,
        },
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
    }


def _prepare(tmp_path: Path) -> dict:
    fixture = _publication_fixture(tmp_path)
    output = tmp_path / "fused-body-publication"
    receipt = producer.prepare(**_producer_arguments(fixture, output=output))
    plan_path = output / producer.PLAN_FILENAME
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    return {
        **fixture,
        "output": output,
        "fused_receipt": receipt,
        "fused_receipt_file_sha": _file_sha256(
            output / producer.PRODUCER_RECEIPT_FILENAME
        ),
        "fused_plan": plan,
        "fused_plan_file_sha": _file_sha256(plan_path),
        "fused_archive_file_sha": _file_sha256(output / producer.ARCHIVE_FILENAME),
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
        "parent_checkpoint": fixture["parent_checkpoint"],
        "expected_parent_checkpoint_inventory_sha256": fixture["parent_inventory_sha"],
        "final_checkpoint": fixture["final_checkpoint"],
        "expected_final_checkpoint_inventory_sha256": fixture["final_inventory_sha"],
    }


def _review_inputs(tmp_path: Path, fixture: dict) -> dict:
    evidence_root = tmp_path / "runtime-evidence"
    evidence_root.mkdir()
    prompts_path = evidence_root / "prompts.json"
    _write_json(
        prompts_path,
        {
            "prompts": [
                {"id": "one", "prompt": "hello", "category": "basic"},
                {"id": "two", "prompt": "hello hello", "category": "basic"},
            ]
        },
    )
    normalized = [
        {
            "id": "one",
            "prompt": "hello",
            "category": "basic",
            "context": None,
            "parent_response": None,
        },
        {
            "id": "two",
            "prompt": "hello hello",
            "category": "basic",
            "context": None,
            "parent_response": None,
        },
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
    envelope_body = {
        "schema": PARENT_ENVELOPE_SCHEMA,
        "status": "complete-one-time-parent-response-cache",
        "parent_model_directory": str(fixture["parent_checkpoint"].resolve()),
        "tokenizer_directory": str(fixture["parent_checkpoint"].resolve()),
        "prompt_source": str(prompts_path.resolve()),
        "prompt_set_sha256": canonical_sha256(normalized),
        "prompt_count": len(normalized),
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "inference_device": "cpu",
        "inference_dtype": "bfloat16",
        "parent_state_sha256": fixture["fused_plan"]["parent_cache_binding"][
            "parent_state_sha256"
        ],
        "parent_identity_sha256": fixture["fused_plan"]["parent_cache_binding"][
            "parent_identity_sha256"
        ],
        "parent_unique_parameters": fixture["fused_plan"]["parent_cache_binding"][
            "parent_unique_parameters"
        ],
        "tokenizer_sha256": fixture["fused_plan"]["parent_cache_binding"][
            "tokenizer_sha256"
        ],
        "responses": responses,
        "response_bank_sha256": canonical_sha256(responses),
        "execution_audit": {
            "parent_model_loads": 1,
            "parent_generate_invocations": len(responses),
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
    envelope_path = evidence_root / "parent-envelope.json"
    _write_json(envelope_path, envelope)
    return {
        "prompts": prompts_path,
        "prompts_file_sha": _file_sha256(prompts_path),
        "envelope": envelope_path,
        "envelope_sha": envelope["envelope_sha256"],
        "envelope_file_sha": _file_sha256(envelope_path),
        "prompt_set_sha": canonical_sha256(normalized),
        "prompt_ids": [row["id"] for row in normalized],
        "root": evidence_root,
    }


def _runner_arguments(
    fixture: dict,
    review: dict,
    *,
    run_directory: Path,
    shard_count: int,
    shard_index: int,
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
        "producer_receipt": fixture["output"] / producer.PRODUCER_RECEIPT_FILENAME,
        "expected_producer_receipt_sha256": fixture["fused_receipt"]["receipt_sha256"],
        "expected_producer_receipt_file_sha256": fixture["fused_receipt_file_sha"],
        "fused_plan": fixture["output"] / producer.PLAN_FILENAME,
        "expected_fused_plan_sha256": fixture["fused_plan"]["plan_sha256"],
        "expected_fused_plan_file_sha256": fixture["fused_plan_file_sha"],
        "fused_archive": fixture["output"] / producer.ARCHIVE_FILENAME,
        "expected_fused_archive_file_sha256": fixture["fused_archive_file_sha"],
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


def test_producer_publishes_exact_candidates_and_economical_schedule(
    tmp_path: Path,
) -> None:
    fixture = _prepare(tmp_path)
    plan = fixture["fused_plan"]

    assert plan["component_count"] == 3
    assert plan["mlp_component_count"] == 2
    assert plan["short_conv_component_count"] == 1
    assert [row["kind"] for row in plan["components"]] == [
        "short_conv",
        "mlp",
        "mlp",
    ]
    assert plan["schedule"]["candidate_ids"] == [
        "cached-parent-reference",
        "isolated--layer-000-short_conv",
        "isolated--layer-000-mlp",
        "isolated--layer-001-mlp",
        "all-mlp",
        "paired-layer-000-conv-mlp",
        "all-fused-body",
    ]
    assert plan["schedule"]["candidate_count"] == 7
    assert plan["schedule"]["deduplicated_candidate_aliases"]
    assert all(
        row["physical_parameter_accounting"]["physical_parameter_reduction_from_parent"]
        > 0
        for row in plan["components"]
    )
    assert plan["operation_audit"]["parent_model_forward_calls"] == 0
    assert plan["operation_audit"]["training_steps"] == 0
    assert plan["bridge_sidecar_parameters_materialized"] is False
    cached_reference = plan["fold_stage_controls"]["cached_parent_reference"]
    assert cached_reference["response_runtime_dtype"] == "bfloat16"
    assert cached_reference["candidate_runtime_executed"] is False
    assert cached_reference["response_attributed_to_fp32_structural_shell"] is False
    fp32_shell = plan["fold_stage_controls"]["fp32_unmodified_parent_structural_shell"]
    assert fp32_shell["runtime_dtype"] == "float32"
    assert fp32_shell["candidate_runtime_executed"] is False
    assert fp32_shell["decoded_responses_claimed"] is False


def test_modulo_shards_are_disjoint_complete_and_tamper_evident(tmp_path: Path) -> None:
    schedule = _prepare(tmp_path)["fused_plan"]["schedule"]
    shards = [
        runner._schedule_shard(schedule, shard_count=3, shard_index=index)
        for index in range(3)
    ]
    selected = [set(row["selected_candidate_ids"]) for row in shards]
    assert not (selected[0] & selected[1])
    assert not (selected[0] & selected[2])
    assert not (selected[1] & selected[2])
    assert set().union(*selected) == set(schedule["candidate_ids"])
    for shard in shards:
        rows = runner._authenticate_schedule_shard(schedule, shard)
        assert [row["candidate_id"] for row in rows] == shard["selected_candidate_ids"]

    tampered_shard = copy.deepcopy(shards[0])
    tampered_shard["selected_candidate_ids"].append("forged")
    with pytest.raises(runner.Lfm2FusedBodyDiagnosticRunError, match="differs"):
        runner._authenticate_schedule_shard(schedule, tampered_shard)
    tampered_schedule = copy.deepcopy(schedule)
    tampered_schedule["candidates"][0]["candidate_id"] = "forged"
    with pytest.raises(Exception, match="schedule|Schedule"):
        runner._schedule_shard(tampered_schedule, shard_count=1, shard_index=0)


def test_runner_reconstructs_differentials_and_uses_cached_parent_reference(
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
                "response": f"hybrid-{prompt['id']}",
                "response_token_ids": [4, 2],
                "diagnostics": {"tokens": 2, "interpretation": "diagnostic-only"},
            }
            for prompt in prompts
        ]

    monkeypatch.setattr(runner, "_generate_rows", fake_generate)
    run_root = tmp_path / "run"
    result = runner.run(
        **_runner_arguments(
            fixture,
            review,
            run_directory=run_root,
            shard_count=4,
            shard_index=0,
        )
    )

    assert (
        result["status"]
        == "NON-FINAL-FUSED-BODY-PARTIAL-SHARD-REVIEW-EVIDENCE-COMPLETE"
    )
    assert result["candidate_ids"] == ["cached-parent-reference", "all-mlp"]
    assert calls == [(2, 1024, 256)]
    assert result["operation_audit"]["parent_model_generate_calls"] == 0
    assert result["operation_audit"]["hybrid_model_generate_passes"] == 1
    assert result["operation_audit"]["cached_parent_references_reused"] == 1
    assert result["candidate_reviews"][0]["cached_parent_reference_reused"] is True
    assert result["candidate_reviews"][0]["candidate_runtime_executed"] is False
    assert result["candidate_reviews"][0]["runtime_decode_performed"] is False
    assert (
        result["candidate_reviews"][0]["response_origin"]
        == "authenticated-bfloat16-parent-cache"
    )
    assert result["candidate_reviews"][0]["response_runtime_dtype"] == "bfloat16"
    assert (
        result["candidate_reviews"][0][
            "reference_and_structural_manifest_precision_differ"
        ]
        is True
    )
    assert (
        result["interpretation"]["final_s2_fused_components_installed_in_this_shard"]
        is True
    )
    all_mlp = result["candidate_reviews"][1]["installation_receipt"]
    assert all_mlp["realized_parameter_reduction"] > 0
    assert all_mlp["bridge_sidecar_parameter_elements"] == 0
    assert all_mlp["parameter_alias_topology_unchanged"] is True
    assert list((run_root / "differential-checkpoints").iterdir()) == []
    assert len(list((run_root / "manual-review").glob("*.json"))) == 3


def test_cached_reference_only_shard_does_not_claim_fp32_execution_or_s2_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepare(tmp_path)
    review = _review_inputs(tmp_path, fixture)

    def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("Cached BF16 reference must not execute the FP32 shell.")

    monkeypatch.setattr(runner, "_generate_rows", forbidden_generate)
    result = runner.run(
        **_runner_arguments(
            fixture,
            review,
            run_directory=tmp_path / "cached-reference-only-run",
            shard_count=7,
            shard_index=0,
        )
    )

    assert result["candidate_ids"] == ["cached-parent-reference"]
    assert result["reviewed_candidate_count"] == 1
    assert result["candidate_reviews"][0]["candidate_runtime_executed"] is False
    assert result["candidate_reviews"][0]["runtime_decode_performed"] is False
    assert result["operation_audit"]["hybrid_model_generate_passes"] == 0
    assert result["operation_audit"]["cached_parent_references_reused"] == 1
    assert (
        result["interpretation"]["final_s2_fused_components_installed_in_this_shard"]
        is False
    )
    assert (
        result["interpretation"]["cached_parent_reference_is_fp32_shell_execution"]
        is False
    )
    response = result["candidate_reviews"][0]["responses"][0]
    assert response["response_runtime_dtype"] == "bfloat16"
    assert response["fp32_structural_shell_runtime_dtype"] == "float32"
    assert response["reference_and_structural_shell_precision_differ"] is True


def test_runner_rejects_nested_run_root_before_mutating_authenticated_sources(
    tmp_path: Path,
) -> None:
    fixture = _prepare(tmp_path)
    review = _review_inputs(tmp_path, fixture)
    protected = (
        fixture["parent_checkpoint"],
        fixture["final_checkpoint"],
        fixture["publication"],
        fixture["output"],
        review["root"],
    )
    baseline = {str(root): b038._checkpoint_inventory(root) for root in protected}
    for index, root in enumerate(protected):
        nested = root / f"forbidden-run-{index}"
        arguments = _runner_arguments(
            fixture,
            review,
            run_directory=nested,
            shard_count=7,
            shard_index=0,
        )
        with pytest.raises(
            runner.Lfm2FusedBodyDiagnosticRunError,
            match="lexically separate",
        ):
            runner.run(**arguments)
        assert not nested.exists()
        assert b038._checkpoint_inventory(root) == baseline[str(root)]


def test_producer_fails_closed_on_external_b038_pin(tmp_path: Path) -> None:
    fixture = _publication_fixture(tmp_path)
    with pytest.raises(Exception, match="pin|differs"):
        producer.prepare(
            output_directory=tmp_path / "not-created",
            component_publication_directory=fixture["publication"],
            expected_component_producer_receipt_sha256=fixture["component_receipt"][
                "receipt_sha256"
            ],
            expected_component_producer_receipt_file_sha256="0" * 64,
            expected_component_plan_sha256=fixture["component_plan"]["plan_sha256"],
            expected_component_plan_file_sha256=fixture["component_plan_file_sha"],
            expected_component_bridges_file_sha256=fixture["component_bridge_file_sha"],
            parent_checkpoint=fixture["parent_checkpoint"],
            expected_parent_checkpoint_inventory_sha256=fixture["parent_inventory_sha"],
            final_checkpoint=fixture["final_checkpoint"],
            expected_final_checkpoint_inventory_sha256=fixture["final_inventory_sha"],
        )
    assert not (tmp_path / "not-created").exists()


def test_producer_rejects_nested_output_without_mutating_authenticated_sources(
    tmp_path: Path,
) -> None:
    fixture = _publication_fixture(tmp_path)
    protected = (
        fixture["parent_checkpoint"],
        fixture["final_checkpoint"],
        fixture["publication"],
    )
    baseline = {str(root): b038._checkpoint_inventory(root) for root in protected}
    for index, root in enumerate(protected):
        nested = root / f"forbidden-output-{index}"
        with pytest.raises(
            producer.Lfm2FusedBodyPlanProducerError,
            match="lexically separate",
        ):
            producer.prepare(**_producer_arguments(fixture, output=nested))
        assert not nested.exists()
        assert b038._checkpoint_inventory(root) == baseline[str(root)]


def test_parent_cache_binding_rejects_identity_tokenizer_and_short_horizon(
    tmp_path: Path,
) -> None:
    fixture = _prepare(tmp_path)
    review = _review_inputs(tmp_path, fixture)
    envelope = _load_parent_envelope(
        review["envelope"],
        expected_envelope_sha256=review["envelope_sha"],
        expected_file_sha256=review["envelope_file_sha"],
        expected_prompt_set_sha256=review["prompt_set_sha"],
        expected_prompt_ids=review["prompt_ids"],
    )
    authenticated = runner._authenticate_parent_cache_binding(
        envelope,
        plan=fixture["fused_plan"],
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
    )
    assert authenticated["direct_local_source_verified"] is True
    assert authenticated["zero_training_provenance_verified"] is True

    mutations = []
    missing_identity = copy.deepcopy(envelope)
    missing_identity.pop("parent_identity_sha256")
    mutations.append(missing_identity)
    wrong_identity = copy.deepcopy(envelope)
    wrong_identity["parent_identity_sha256"] = "0" * 64
    mutations.append(wrong_identity)
    missing_tokenizer = copy.deepcopy(envelope)
    missing_tokenizer.pop("tokenizer_sha256")
    mutations.append(missing_tokenizer)
    wrong_tokenizer = copy.deepcopy(envelope)
    wrong_tokenizer["tokenizer_sha256"] = "0" * 64
    mutations.append(wrong_tokenizer)
    wrong_state = copy.deepcopy(envelope)
    wrong_state["parent_state_sha256"] = "0" * 64
    mutations.append(wrong_state)
    short_new_horizon = copy.deepcopy(envelope)
    short_new_horizon["maximum_new_tokens"] = 64
    mutations.append(short_new_horizon)
    mismatched_new_horizon = copy.deepcopy(envelope)
    mismatched_new_horizon["maximum_new_tokens"] = 512
    mutations.append(mismatched_new_horizon)
    short_prompt_horizon = copy.deepcopy(envelope)
    short_prompt_horizon["maximum_prompt_tokens"] = 64
    mutations.append(short_prompt_horizon)
    mismatched_prompt_horizon = copy.deepcopy(envelope)
    mismatched_prompt_horizon["maximum_prompt_tokens"] = 512
    mutations.append(mismatched_prompt_horizon)
    training_provenance = copy.deepcopy(envelope)
    training_provenance["execution_audit"]["training_steps"] = 1
    mutations.append(training_provenance)
    hub_provenance = copy.deepcopy(envelope)
    hub_provenance["execution_audit"]["hub_loader_used"] = True
    mutations.append(hub_provenance)
    for mutated in mutations:
        with pytest.raises(
            runner.Lfm2FusedBodyDiagnosticRunError,
            match="not bound",
        ):
            runner._authenticate_parent_cache_binding(
                mutated,
                plan=fixture["fused_plan"],
                maximum_prompt_tokens=1024,
                maximum_new_tokens=256,
            )

    review["envelope"].write_text(
        review["envelope"].read_text(encoding="ascii") + " ",
        encoding="ascii",
    )
    with pytest.raises(Exception, match="bytes changed"):
        _load_parent_envelope(
            review["envelope"],
            expected_envelope_sha256=review["envelope_sha"],
            expected_file_sha256=review["envelope_file_sha"],
            expected_prompt_set_sha256=review["prompt_set_sha"],
            expected_prompt_ids=review["prompt_ids"],
        )
