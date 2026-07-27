from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from experiments import prepare_lfm2_progressive_component_ablation as producer
from experiments import run_lfm2_progressive_component_ablation as runner
from experiments.review_lfm2_semantic_frontier import (
    PARENT_ENVELOPE_SCHEMA,
    load_review_prompts,
)
from neural_forge import hf_lfm2_postfold_reclosure_seed_archive as seed_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_attention_group_qdeim import (
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_candidate_state_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_direct_local import (
    _checkpoint_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (
    LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
    write_lfm2_postfold_reclosure_seed_archive,
)
from neural_forge.hf_lfm2_semantic_foundation import (
    LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (
    write_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


def _sealed(body: dict, field: str = "result_sha256") -> dict:
    return {**body, field: canonical_sha256(body)}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _config(*, parent: bool):
    transformers = pytest.importorskip("transformers")
    return transformers.Lfm2Config(
        vocab_size=19 if parent else 9,
        hidden_size=8 if parent else 4,
        intermediate_size=12 if parent else 3,
        num_hidden_layers=2,
        num_attention_heads=2 if parent else 1,
        num_key_value_heads=1,
        max_position_embeddings=64,
        block_auto_adjust_ff_dim=False,
        layer_types=["conv", "full_attention"],
        tie_word_embeddings=parent,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _tokenizer(path: Path, *, vocabulary_size: int) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "hello": 4,
    }
    for index in range(5, vocabulary_size):
        vocab[f"token-{index}"] = index
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    _write_json(
        path / "tokenizer_config.json",
        {
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "model_max_length": 64,
        },
    )


def _checkpoint(path: Path, model, *, vocabulary_size: int) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    path.mkdir()
    _write_json(path / "config.json", model.config.to_dict())
    safetensors.save_file(
        _checkpoint_state(model),
        str(path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    _tokenizer(path, vocabulary_size=vocabulary_size)


def _candidate(
    child,
    encoder: torch.Tensor,
    *,
    layer_index: int,
) -> Lfm2CompiledAttentionGroupQdeim:
    attention = child.model.layers[layer_index].self_attn
    query = attention.q_proj.weight.detach().clone()
    key = attention.k_proj.weight.detach().clone()
    value = attention.v_proj.weight.detach().clone()
    out = attention.out_proj.weight.detach().clone()
    decoder = torch.linalg.pinv(encoder.to(torch.float64), rcond=1e-12)
    chart = encoder.to(torch.float64).clone()
    basis = torch.ones((1, 1), dtype=torch.float64)
    interpolation = torch.ones((1, 1), dtype=torch.float64)
    reconstruction = torch.eye(4, dtype=torch.float64)
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        "selection": {
            "selected_source_key_value_groups": [0],
            "selected_source_query_heads": [0],
        },
        "interpolation": {
            "group_pod_basis_sha256": lfm2_authenticated_tensor_sha256(basis),
            "group_interpolation_sha256": lfm2_authenticated_tensor_sha256(
                interpolation
            ),
            "context_reconstruction_sha256": lfm2_authenticated_tensor_sha256(
                reconstruction
            ),
        },
        "authentication": {
            "normalized_input_decoder_sha256": lfm2_authenticated_tensor_sha256(
                decoder
            ),
            "residual_output_chart_sha256": lfm2_authenticated_tensor_sha256(chart),
        },
        "materialization": {
            "candidate_state_sha256": (
                lfm2_attention_group_qdeim_candidate_state_sha256(
                    query,
                    key,
                    value,
                    out,
                )
            )
        },
    }
    return Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=(0,),
        selected_source_query_heads=(0,),
        query_weight=query,
        key_weight=key,
        value_weight=value,
        out_weight=out,
        group_pod_basis=basis,
        group_interpolation=interpolation,
        context_reconstruction=reconstruction,
        normalized_input_decoder=decoder,
        residual_output_chart=chart,
        report=_sealed(body),
    )


def _source_fixture(tmp_path: Path) -> dict:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(41)
    parent_live = transformers.Lfm2ForCausalLM(_config(parent=True)).eval()
    child_live = transformers.Lfm2ForCausalLM(_config(parent=False)).eval()
    parent_checkpoint = tmp_path / "parent-checkpoint"
    source_root = tmp_path / "source-run"
    source_root.mkdir()
    compact_checkpoint = source_root / "final-checkpoint"
    structural_checkpoint = source_root / "structural-body-checkpoint"
    _checkpoint(parent_checkpoint, parent_live, vocabulary_size=19)
    _checkpoint(compact_checkpoint, child_live, vocabulary_size=9)
    shutil.copytree(compact_checkpoint, structural_checkpoint)

    parent = load_lfm2_direct_local_model(
        parent_checkpoint,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    child = load_lfm2_direct_local_model(
        compact_checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert sum(value.numel() for value in parent.parameters()) == 1248
    assert sum(value.numel() for value in child.parameters()) == 312
    encoder = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float64,
    )
    mapped = torch.randn(9, 8)
    final_w2_seed = child.model.layers[1].feed_forward.w2.weight.detach().clone()
    active = _sealed(
        {
            "schema": "synthetic-active-interface-v1",
            "producer_encoder_sha256": "1" * 64,
        },
        "report_sha256",
    )
    foundation = _sealed(
        {
            "schema": LFM2_SEMANTIC_FOUNDATION_INTERFACE_SCHEMA,
            "active_interface_report": active,
            "active_interface_report_sha256": active["report_sha256"],
            "source_tied_table_sha256": seed_module._foundation_tensor_sha256(mapped),
            "producer_encoder_sha256": seed_module._foundation_tensor_sha256(
                encoder
            ),
        }
    )
    candidate = _candidate(child, encoder, layer_index=1)
    attention_operator_receipt = _sealed(
        {
            "schema": "synthetic-attention-fold-install-v1",
            "coupled_attention_compiler_receipt": dict(candidate.report),
            "installed": True,
        }
    )
    conv_operator_receipt = _sealed(
        {
            "schema": "synthetic-short-conv-fold-install-v1",
            "installed": True,
        }
    )
    layer_transactions = []
    for layer_index, (operator_kind, operator_receipt) in enumerate(
        (
            ("short_conv", conv_operator_receipt),
            ("full_attention", attention_operator_receipt),
        )
    ):
        mlp_receipt = _sealed(
            {
                "schema": "synthetic-mlp-fold-install-v1",
                "layer_index": layer_index,
                "installed": True,
            }
        )
        transaction_body = {
            "schema": "neural-forge-lfm2-semantic-layer-transaction-v1",
            "layer_index": layer_index,
            "operator_kind": operator_kind,
            "operator_installation_sha256": operator_receipt["result_sha256"],
            "operator_complete_candidate_frontier_receipt": operator_receipt,
            "mlp_installation_sha256": mlp_receipt["result_sha256"],
            "mlp_complete_candidate_frontier_receipt": mlp_receipt,
        }
        if operator_kind == "full_attention":
            transaction_body["attention_operator_installation_receipt"] = (
                operator_receipt
            )
        layer_transactions.append(_sealed(transaction_body))
    live_seed_sha = lfm2_cpcf_terminal_action_tensor_sha256(final_w2_seed)
    installed_prior_sha = "2" * 64
    prior_auth_sha = "3" * 64
    stack_state_sha = fingerprint_model_state(child)
    stack = _sealed(
        {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "tied_interface_lineage": {
                "interface_report_sha256": active["report_sha256"],
                "producer_encoder_sha256": active["producer_encoder_sha256"],
            },
            "deferred_final_w2_sha256": lfm2_authenticated_tensor_sha256(
                final_w2_seed
            ),
            "final_layer_mlp_down_weight_prior_live_w2_sha256": live_seed_sha,
            "final_layer_mlp_down_weight_prior_tensor_sha256": installed_prior_sha,
            "final_layer_mlp_down_weight_prior_authentication_sha256": prior_auth_sha,
            "model_state_after_sha256": stack_state_sha,
            "layer_installations": layer_transactions,
            "operator_layer_kinds": ["short_conv", "full_attention"],
        }
    )
    deferred = _sealed(
        {
            "schema": LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "prior_authentication_sha256": prior_auth_sha,
            "prior_tensor_sha256": installed_prior_sha,
            "live_w2_sha256_before": live_seed_sha,
            "live_w2_sha256_after": installed_prior_sha,
            "model_state_before_sha256": stack_state_sha,
            "model_state_after_sha256": stack_state_sha,
        }
    )
    candidate_auth = write_lfm2_structural_attention_candidate_archive(
        source_root / "structural-attention-candidates",
        {1: candidate},
        semantic_layer_stack_receipt=stack,
    )
    seed_auth = write_lfm2_postfold_reclosure_seed_archive(
        source_root / "postfold-reclosure-seeds",
        mapped_parent_tied_table=mapped,
        producer_encoder=encoder,
        deferred_final_w2_seed=final_w2_seed,
        foundation_interface_solve_receipt=foundation,
        semantic_layer_stack_receipt=stack,
        deferred_final_w2_installation_receipt=deferred,
    )
    compact_inventory = producer._checkpoint_inventory(compact_checkpoint)
    structural_inventory = producer._checkpoint_inventory(structural_checkpoint)
    structural_body = _sealed(
        {
            "schema": producer.SOURCE_BODY_SCHEMA,
            "status": "standalone-structural-body-saved-and-freshly-reloaded",
            "checkpoint_inventory": structural_inventory,
            "checkpoint_inventory_sha256": canonical_sha256(structural_inventory),
            "semantic_layer_stack_sha256": stack["result_sha256"],
            "deferred_final_w2_install_sha256": deferred["result_sha256"],
            "body_model_state_sha256": fingerprint_model_state(child),
            "structural_attention_candidate_archive_manifest_sha256": (
                candidate_auth["manifest_sha256"]
            ),
            "postfold_reclosure_seed_archive_manifest_sha256": seed_auth[
                "manifest_sha256"
            ],
        },
        "receipt_sha256",
    )
    causal_output = _sealed(
        {
            "schema": "synthetic-causal-output-install-v1",
            "installed": True,
        }
    )
    _write_json(source_root / "semantic-layer-stack.json", stack)
    _write_json(source_root / "deferred-final-w2-install.json", deferred)
    _write_json(source_root / "structural-body-receipt.json", structural_body)
    run_body = {
        "schema": producer.SOURCE_RUN_SCHEMA,
        "status": "standalone-exact4x-composite-child-decoded",
        "parent_identity_sha256": fingerprint_lfm2_parent(parent).identity_sha256,
        "parent_unique_parameters": 1248,
        "target_unique_parameters": 312,
        "realized_parameter_compaction_ratio": 4.0,
        "child_parameter_dtype": "float32",
        "foundation_interface_solve": foundation,
        "semantic_layer_stack": stack,
        "deferred_final_w2_install": deferred,
        "structural_body_checkpoint_receipt": structural_body,
        "structural_attention_candidate_archive": dict(candidate_auth),
        "postfold_reclosure_seed_archive_manifest_sha256": seed_auth[
            "manifest_sha256"
        ],
        "postfold_reclosure_seed_archive": dict(seed_auth),
        "causal_output_install": causal_output,
        "candidate_checkpoint_inventory": compact_inventory,
        "candidate_checkpoint_inventory_sha256": canonical_sha256(compact_inventory),
        "complete_child_model_state_sha256": fingerprint_model_state(child),
        "structural_body_model_state_sha256": fingerprint_model_state(child),
    }
    run_receipt = {**run_body, "result_sha256": canonical_sha256(run_body)}
    _write_json(source_root / "run-receipt.json", run_receipt)
    return {
        "source_root": source_root,
        "parent_checkpoint": parent_checkpoint,
        "compact_checkpoint": compact_checkpoint,
        "run": run_receipt,
        "candidate_auth": candidate_auth,
        "seed_auth": seed_auth,
    }


def test_automatic_producer_and_full_runner_orchestration_smoke(
    tmp_path: Path,
) -> None:
    fixture = _source_fixture(tmp_path)
    output = tmp_path / "prepared-ablation"
    receipt = producer.prepare(
        output_directory=output,
        source_run_directory=fixture["source_root"],
        expected_source_run_result_sha256=fixture["run"]["result_sha256"],
        expected_source_run_file_sha256=_file_sha256(
            fixture["source_root"] / "run-receipt.json"
        ),
        parent_checkpoint=fixture["parent_checkpoint"],
        expected_parent_checkpoint_inventory_sha256=canonical_sha256(
            producer._checkpoint_inventory(fixture["parent_checkpoint"])
        ),
        compact_checkpoint=fixture["compact_checkpoint"],
        expected_compact_checkpoint_inventory_sha256=canonical_sha256(
            producer._checkpoint_inventory(fixture["compact_checkpoint"])
        ),
        expected_seed_archive_manifest_sha256=fixture["seed_auth"][
            "manifest_sha256"
        ],
        expected_candidate_archive_manifest_sha256=fixture["candidate_auth"][
            "manifest_sha256"
        ],
    )
    plan_path = output / producer.PLAN_FILENAME
    bridge_path = output / producer.BRIDGE_FILENAME
    producer_receipt_path = output / producer.PRODUCER_RECEIPT_FILENAME
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    assert receipt["component_count"] == 4
    assert [value["kind"] for value in plan["components"]] == [
        "short_conv",
        "mlp",
        "attention",
        "mlp",
    ]
    assert plan["component_granularity"] == "whole-subsystem"
    assert plan["q_k_v_o_or_individual_head_isolation_claimed"] is False
    assert all(
        value["require_parameter_reduction"] is True
        for value in plan["components"]
    )
    assert plan["excluded_components"][0]["component_id"] == "terminal-readout"
    assert plan["excluded_components"][0]["reason"] == (
        "replacement-would-break-native-parent-input-output-alias"
    )

    prompts_path = tmp_path / "prompts.json"
    _write_json(
        prompts_path,
        {"prompts": [{"id": "hello", "prompt": "hello", "category": "smoke"}]},
    )
    prompt_rows = [dict(value) for value in load_review_prompts(prompts_path)]
    responses = [
        {
            "id": "hello",
            "response": "cached parent response",
            "response_token_ids": [4, 2],
            "diagnostics": {"interpretation": "cached-test-only"},
        }
    ]
    envelope_body = {
        "schema": PARENT_ENVELOPE_SCHEMA,
        "status": "complete-one-time-parent-response-cache",
        "prompt_set_sha256": canonical_sha256(prompt_rows),
        "prompt_count": 1,
        "responses": responses,
        "response_bank_sha256": canonical_sha256(responses),
    }
    envelope = {
        **envelope_body,
        "envelope_sha256": canonical_sha256(envelope_body),
    }
    envelope_path = tmp_path / "parent-envelope.json"
    _write_json(envelope_path, envelope)
    run_result = runner.run(
        run_directory=tmp_path / "ablation-run",
        parent_checkpoint=fixture["parent_checkpoint"],
        compact_checkpoint=fixture["compact_checkpoint"],
        producer_receipt=producer_receipt_path,
        expected_producer_receipt_sha256=receipt["receipt_sha256"],
        expected_producer_receipt_file_sha256=_file_sha256(
            producer_receipt_path
        ),
        component_plan=plan_path,
        expected_component_plan_sha256=plan["plan_sha256"],
        expected_component_plan_file_sha256=_file_sha256(plan_path),
        bridge_tensors=bridge_path,
        expected_bridge_tensors_file_sha256=_file_sha256(bridge_path),
        prompts_json=prompts_path,
        expected_prompts_file_sha256=_file_sha256(prompts_path),
        parent_envelope=envelope_path,
        expected_parent_envelope_sha256=envelope["envelope_sha256"],
        expected_parent_envelope_file_sha256=_file_sha256(envelope_path),
        device="cpu",
        dtype="float32",
        maximum_prompt_tokens=32,
        maximum_new_tokens=16,
        checkpoint_retention="receipts-only",
    )
    assert run_result["reviewed_candidate_count"] == 9
    assert [
        value["ablation_mode"] for value in run_result["candidate_reviews"]
    ] == [
        "isolated",
        "isolated",
        "isolated",
        "isolated",
        "subsystem_group",
        "subsystem_cumulative",
        "cumulative",
        "cumulative",
        "cumulative",
    ]
    all_mlp = next(
        value
        for value in run_result["schedule"]["candidates"]
        if value["candidate_id"] == "subsystem--all-mlp"
    )
    assert all_mlp["component_ids"] == ["layer-000-mlp", "layer-001-mlp"]
    assert run_result["schedule"]["candidate_mode_counts"] == {
        "isolated": 4,
        "subsystem_group": 1,
        "subsystem_cumulative": 1,
        "cumulative": 3,
    }
    component_sets = [
        frozenset(value["component_ids"])
        for value in run_result["schedule"]["candidates"]
    ]
    assert len(component_sets) == len(set(component_sets))
    assert all(
        value["responses"][0]["parent_response"] == "cached parent response"
        for value in run_result["candidate_reviews"]
    )
    assert run_result["operation_audit"]["training_steps"] == 0
    assert run_result["producer_receipt_sha256"] == receipt["receipt_sha256"]
    assert run_result["interpretation"]["standalone_child_compression_claimed"] is False


def test_analytic_decoder_requires_full_child_column_rank() -> None:
    with pytest.raises(
        producer.Lfm2ComponentAblationPlanProducerError,
        match="full child-column rank",
    ):
        producer._analytic_decoder(torch.ones(4, 2, dtype=torch.float64))


def test_resealed_failed_source_run_status_is_rejected(tmp_path: Path) -> None:
    fixture = _source_fixture(tmp_path)
    failed = dict(fixture["run"])
    failed.pop("result_sha256")
    failed["status"] = "failed-without-behavior-promotion"
    failed["result_sha256"] = canonical_sha256(failed)
    receipt_path = fixture["source_root"] / "run-receipt.json"
    _write_json(receipt_path, failed)
    with pytest.raises(
        producer.Lfm2ComponentAblationPlanProducerError,
        match="successful decoded exact4x status",
    ):
        producer._load_source_run(
            fixture["source_root"],
            expected_result_sha256=failed["result_sha256"],
            expected_file_sha256=_file_sha256(receipt_path),
        )
