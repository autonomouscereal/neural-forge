from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments import run_lfm2_exact4x_causal_reclosure as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256


RUNNER = Path(runner.__file__)


def _sealed(body: dict[str, object], field: str = "result_sha256") -> dict[str, object]:
    return {**body, field: canonical_sha256(body)}


def test_pipeline_stage_order_is_the_single_causal_reclosure_order() -> None:
    assert runner.PIPELINE_STAGE_ORDER == (
        "authenticate-load-pinned-composite-source",
        "fresh-load-postfold-reclosure-seeds",
        "rebase-installed-terminal-w2-to-archived-seed",
        "save-fresh-reload-rebased-source",
        "compile-causal-target-lattice-qk",
        "save-fresh-reload-seal-qk-v2",
        "fresh-load-structural-attention-candidates",
        "compile-causal-post-qk-output-projections",
        "compile-post-o-source-local-global-reclosure",
        "install-corrected-terminal-w2",
        "save-fresh-reload-standalone-child",
        "decode-retain-whole-responses-and-manual-review",
    )


def test_lightweight_stage_machine_executes_each_stage_once_in_order() -> None:
    calls: list[str] = []
    handlers = {}
    for sequence_number, stage in enumerate(runner.PIPELINE_STAGE_ORDER):

        def handler(state, *, stage=stage, sequence_number=sequence_number):
            assert tuple(state) == runner.PIPELINE_STAGE_ORDER[:sequence_number]
            calls.append(stage)
            return runner._StageOutcome(
                value={"stage": stage},
                evidence={"stage": stage, "sequence_number": sequence_number},
            )

        handlers[stage] = handler

    state, receipt = runner._run_ordered_stage_machine(handlers)

    assert tuple(calls) == runner.PIPELINE_STAGE_ORDER
    assert tuple(state) == runner.PIPELINE_STAGE_ORDER
    assert receipt["schema"] == runner.PIPELINE_RECEIPT_SCHEMA
    assert receipt["stage_count"] == len(runner.PIPELINE_STAGE_ORDER)
    assert receipt["every_required_stage_executed_once"] is True
    assert receipt["terminal_w2_published_only_after_qk_o_and_global_reclosure"] is True
    assert receipt["behavior_review_occurs_only_after_final_fresh_reload"] is True
    assert receipt["result_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "result_sha256"}
    )


def test_stage_machine_rejects_a_missing_or_extra_stage_before_execution() -> None:
    handlers = {
        stage: lambda _state: runner._StageOutcome(value=None, evidence={})
        for stage in runner.PIPELINE_STAGE_ORDER[:-1]
    }
    with pytest.raises(
        runner.Lfm2Exact4xCausalReclosureError,
        match="handler inventory changed",
    ):
        runner._run_ordered_stage_machine(handlers)


def test_production_calls_occur_in_the_required_source_order() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    names = (
        "load_lfm2_postfold_reclosure_seed_archive",
        "rebase_lfm2_installed_terminal_w2_for_output_reclosure",
        "compile_install_lfm2_saved_body_target_lattice_qk_closure",
        "load_lfm2_structural_attention_candidate_archive",
        "compile_install_lfm2_causal_post_qk_output_fold",
        "compile_install_lfm2_post_causal_o_source_local_global_reclosure",
        "install_lfm2_post_reclosure_terminal_w2",
        "review_frontier",
    )
    calls = {
        name: [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        for name in names
    }
    assert all(len(values) == 1 for values in calls.values())
    assert [calls[name][0].lineno for name in names] == sorted(
        calls[name][0].lineno for name in names
    )

    reload_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_save_and_fresh_reload"
    ]
    assert len(reload_calls) == 3
    assert (
        calls["rebase_lfm2_installed_terminal_w2_for_output_reclosure"][0].lineno
        < reload_calls[0].lineno
        < calls["compile_install_lfm2_saved_body_target_lattice_qk_closure"][0].lineno
        < reload_calls[1].lineno
        < calls["load_lfm2_structural_attention_candidate_archive"][0].lineno
        < calls["compile_install_lfm2_causal_post_qk_output_fold"][0].lineno
        < calls[
            "compile_install_lfm2_post_causal_o_source_local_global_reclosure"
        ][0].lineno
        < calls["install_lfm2_post_reclosure_terminal_w2"][0].lineno
        < reload_calls[2].lineno
        < calls["review_frontier"][0].lineno
    )


def test_runner_uses_loaded_archive_manifest_as_global_coordinate_authority() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "compile_install_lfm2_post_causal_o_source_local_global_reclosure"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    for name in (
        "mapped_parent_tied_table_receipt",
        "producer_encoder_receipt",
    ):
        assert isinstance(keywords[name], ast.Name)
        assert keywords[name].id == "manifest"
    for name in (
        "expected_mapped_parent_tied_table_sha256",
        "expected_producer_encoder_sha256",
    ):
        assert isinstance(keywords[name], ast.Subscript)
        assert "records" in ast.dump(keywords[name])
    for name in (
        "expected_mapped_parent_tied_table_receipt_sha256",
        "expected_producer_encoder_receipt_sha256",
    ):
        assert isinstance(keywords[name], ast.Subscript)
        assert "manifest_sha256" in ast.dump(keywords[name])
    assert isinstance(keywords["authenticated_cache_session"], ast.Subscript)


def test_runner_has_no_parent_model_loader_network_or_compression_substitute() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "load_lfm2_direct_local_model" not in imported
    assert "from_pretrained" not in called_attributes
    assert not {
        "train",
        "backward",
        "grad",
        "prune",
        "quantize",
        "sparsify",
        "share_parameters",
    }.intersection(called_names | called_attributes)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            alias.name.split(".")[0]
            in {"requests", "httpx", "urllib", "huggingface_hub"}
            for alias in node.names
        )
        for node in tree.body
    )


def test_forbidden_operation_audit_is_zero_and_covers_requested_exclusions() -> None:
    assert all(
        value in (0, False)
        for value in runner.FORBIDDEN_OPERATION_AUDIT.values()
    )
    assert {
        "training_steps",
        "optimizer_constructions",
        "optimizer_steps",
        "autograd_grad_calls",
        "backward_calls",
        "parent_model_objects_constructed",
        "parent_model_loads",
        "parent_model_forward_calls",
        "parent_model_generate_calls",
        "parent_trace_recaptures",
        "parameters_pruned",
        "parameters_quantized",
        "parameters_sparsified",
        "parameter_sharing_used",
        "parent_runtime_sidecar_used",
        "network_access",
    } == set(runner.FORBIDDEN_OPERATION_AUDIT)


def _production_shaped_coordinate_completion_proof() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    candidate = _sealed(
        {
            "structural_rank": 3,
            "materialized_rank": 3,
            "full_rank_materialized": True,
        }
    )
    wrapper = _sealed(
        {
            "required_full_rank": 3,
            "structural_rank": 3,
            "materialized_rank": 3,
            "full_rank_materialized": True,
            "selected_candidate_result_sha256": candidate["result_sha256"],
        }
    )
    stack = {
        "body_global_coordinate_completion_enabled": True,
        "body_global_coordinate_completion_compile_count": 1,
        "body_global_completion_shared_across_every_layer": True,
        "body_global_completion_shared_across_attention_mlp_and_short_conv": True,
        "body_global_coordinate_completion_receipt": wrapper,
        "body_global_coordinate_completion_candidate_receipt": candidate,
    }
    realization = _sealed(
        {
            "child_materialization_dtype": "float32",
            "selected_representability_level": "requested",
            "requested_unit_ray_preserved_without_ulp_escalation": True,
            "materialized_ray_multiplier": 1.0,
            "target_dtype_ulp_escalation_applied": False,
        },
        "receipt_sha256",
    )
    return stack, wrapper, realization


def test_full_rank_authentication_accepts_production_shaped_receipts() -> None:
    stack, _, realization = _production_shaped_coordinate_completion_proof()
    candidate = stack["body_global_coordinate_completion_candidate_receipt"]
    assert isinstance(candidate, dict)
    assert "required_full_rank" not in candidate

    proof = runner._authenticate_coordinate_completion_and_direction(
        semantic_stack=stack,
        body_realization=realization,
    )
    assert proof["required_full_rank"] == 3
    assert proof["full_rank_materialized"] is True
    assert proof["requested_unit_ray_preserved"] is True


def test_full_rank_authentication_rejects_tampered_wrapper() -> None:
    stack, wrapper, realization = _production_shaped_coordinate_completion_proof()
    tampered_wrapper = dict(wrapper)
    tampered_wrapper["required_full_rank"] = 2
    broken_stack = dict(stack)
    broken_stack["body_global_coordinate_completion_receipt"] = tampered_wrapper

    with pytest.raises(
        runner.Lfm2Exact4xCausalReclosureError,
        match="wrapper is stale or not canonically sealed",
    ):
        runner._authenticate_coordinate_completion_and_direction(
            semantic_stack=broken_stack,
            body_realization=realization,
        )


def test_full_rank_authentication_rejects_resealed_wrapper_rank_mismatch() -> None:
    stack, wrapper, realization = _production_shaped_coordinate_completion_proof()
    wrapper_body = {
        key: value for key, value in wrapper.items() if key != "result_sha256"
    }
    wrapper_body["structural_rank"] = 2
    broken_stack = dict(stack)
    broken_stack["body_global_coordinate_completion_receipt"] = _sealed(wrapper_body)

    with pytest.raises(
        runner.Lfm2Exact4xCausalReclosureError,
        match="full-rank completion",
    ):
        runner._authenticate_coordinate_completion_and_direction(
            semantic_stack=broken_stack,
            body_realization=realization,
        )


def test_full_rank_authentication_rejects_selected_candidate_rank_mismatch() -> None:
    stack, wrapper, realization = _production_shaped_coordinate_completion_proof()
    candidate = stack["body_global_coordinate_completion_candidate_receipt"]
    assert isinstance(candidate, dict)
    mismatched_body = {
        key: value for key, value in candidate.items() if key != "result_sha256"
    }
    mismatched_body["materialized_rank"] = 2
    mismatched_candidate = _sealed(mismatched_body)
    wrapper_body = {
        key: value for key, value in wrapper.items() if key != "result_sha256"
    }
    wrapper_body["selected_candidate_result_sha256"] = mismatched_candidate[
        "result_sha256"
    ]
    broken_stack = dict(stack)
    broken_stack["body_global_coordinate_completion_receipt"] = _sealed(wrapper_body)
    broken_stack["body_global_coordinate_completion_candidate_receipt"] = (
        mismatched_candidate
    )

    with pytest.raises(
        runner.Lfm2Exact4xCausalReclosureError,
        match="full-rank completion",
    ):
        runner._authenticate_coordinate_completion_and_direction(
            semantic_stack=broken_stack,
            body_realization=realization,
        )


def test_manual_review_artifact_retains_raw_outputs_without_a_gate() -> None:
    dossier = {
        "dossier_sha256": "a" * 64,
        "candidate_reviews": [
            {
                "candidate_id": runner.FINAL_CANDIDATE_ID,
                "responses": [
                    {"prompt_id": "p0", "response_text": "hello"},
                    {"prompt_id": "p1", "response_text": "world"},
                ],
            }
        ],
    }

    review = runner._manual_review_artifact(
        dossier=dossier,
        candidate_id=runner.FINAL_CANDIDATE_ID,
    )

    assert review["response_count"] == 2
    assert [
        item["raw_response"]["response_text"] for item in review["review_items"]
    ] == ["hello", "world"]
    assert all(item["coherent"] is None for item in review["review_items"])
    assert review["numeric_metric_threshold_used_as_gate"] is False
    assert review["manual_score_used_as_compilation_gate"] is False
    assert review["automatic_acceptance_or_rejection_performed"] is False


def test_parser_requires_pinned_source_but_exposes_no_parent_model() -> None:
    actions = {action.dest: action for action in runner._parser()._actions}

    assert actions["source_run_directory"].required is True
    assert actions["expected_source_run_result_sha256"].required is True
    assert actions["expected_source_run_file_sha256"].required is True
    assert actions["expected_attention_lineage_file_sha256"].required is True
    assert "parent_model_directory" not in actions
    assert "compiler_cache_directory" not in actions
    assert actions["maximum_prompt_tokens"].default == 1024
    assert actions["maximum_new_tokens"].default == 256
