from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_lfm2_semantic_fold_to_review as runner


def _sources(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in (
        "input",
        "cache",
        "bridge",
        "donor",
        "target",
        "parent",
        "statistics",
    ):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    for name in ("compiler", "companion", "prompts", "parent-envelope"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        paths[name] = path
    return paths


def _arguments(tmp_path: Path, paths: dict[str, Path]) -> dict[str, object]:
    recipe = runner.resolve_lfm2_semantic_body_recipe(
        "structural_sequence_residual_mlp_writer"
    )
    return {
        "run_directory": tmp_path / "run",
        "input_checkpoint": paths["input"],
        "expected_input_artifact_sha256": "1" * 64,
        "trace_cache": paths["cache"],
        "expected_cache_key": "2" * 64,
        "bridge_directory": paths["bridge"],
        "expected_bridge_artifact_sha256": "3" * 64,
        "donor_tokenizer_directory": paths["donor"],
        "target_tokenizer_directory": paths["target"],
        "parent_model_directory": paths["parent"],
        "compiler_diagnostic_json": paths["compiler"],
        "expected_compiler_diagnostic_sha256": "4" * 64,
        "interface_statistics_directory": paths["statistics"],
        "expected_interface_statistics_sha256": "5" * 64,
        "interface_frontier_companion_json": paths["companion"],
        "expected_interface_frontier_companion_file_sha256": "6" * 64,
        "expected_interface_frontier_companion_sha256": "7" * 64,
        "anchor_candidate_index": 7,
        "body_recipe": recipe.recipe_id,
        "expected_body_recipe_sha256": recipe.recipe_sha256,
        "tied_interface_mode": "strong-congruence-plus-anchor-residual",
        "device": "cpu",
        "fold_compute_device": "cpu",
        "dtype": "bfloat16",
        "prompts_json": paths["prompts"],
        "expected_prompts_file_sha256": runner._file_sha256(paths["prompts"]),
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "parent_envelope": paths["parent-envelope"],
        "expected_parent_envelope_sha256": "8" * 64,
        "expected_parent_envelope_file_sha256": "9" * 64,
    }


def _sealed_direct_parent_session(
    *,
    parent_identity_sha256: str,
    parameter_count: int,
) -> dict[str, object]:
    records = [
        {
            "name": "model.unique_weight",
            "shape": [parameter_count],
        }
    ]
    identity_body = {
        "schema": "neural-forge-direct-parent-exact-identity-reconstruction-v1",
        "cache_parent_identity_sha256": parent_identity_sha256,
        "checkpoint_tensor_inventory_sha256": runner.canonical_sha256(records),
        "checkpoint_tensor_count": len(records),
        "checkpoint_tensor_records": records,
        "reconstructed_parent_identity_sha256": parent_identity_sha256,
        "exact_parent_identity_authenticated": True,
        "parent_model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "network_access_performed": False,
    }
    identity = {
        **identity_body,
        "result_sha256": runner.canonical_sha256(identity_body),
    }
    session_body = {
        "schema": "neural-forge-direct-parent-session-v2",
        "parent_model_identity_sha256": parent_identity_sha256,
        "parent_identity_reconstruction_sha256": identity["result_sha256"],
        "parent_identity_reconstruction": identity,
        "exact_parent_identity_authenticated": True,
        "checkpoint_files_rehashed_after_materialization": True,
        "shards_hashed_once": True,
        "shards_opened_once": True,
        "parent_model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "network_access_performed": False,
    }
    return {
        **session_body,
        "session_sha256": runner.canonical_sha256(session_body),
    }


def test_correction_scale_frontier_folds_decodes_and_retains_every_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    scales_seen: list[float] = []
    basis_ranks_seen: list[int | None] = []
    compaction = {
        "parent_unique_parameters": 1_000_000,
        "child_parameter_element_count": 250_000,
        "parameter_reduction": 750_000,
        "parent_to_child_parameter_ratio": 4.0,
        "child_parameter_fraction_of_parent": 0.25,
        "parameter_reduction_fraction": 0.75,
    }

    def run_member(**kwargs):
        scale = float(kwargs["correction_scale"])
        scales_seen.append(scale)
        basis_ranks_seen.append(kwargs["global_residual_basis_rank"])
        member = Path(kwargs["run_directory"])
        member.mkdir()
        body = {
            "status": "decoded-behavior-evidence-complete-continuation-unblocked",
            "candidate_count": 2,
            "all_materialized_candidates_received_whole_response_review": True,
            "all_materialized_candidate_checkpoints_retained": True,
            "dense_parameter_compaction": compaction,
        }
        result = {**body, "pipeline_sha256": runner.canonical_sha256(body)}
        runner._atomic_json(member / "pipeline-receipt.json", result)
        return result

    monkeypatch.setattr(runner, "_run_fold_to_review_scale", run_member)
    result = runner.run_fold_to_review(
        **arguments,
        global_residual_closure_version="producer_aware_v2a",
        correction_scale_frontier=(0.25, 0.5, 1.0),
        correction_scale=0.5,
        global_residual_backend="cpu",
        global_residual_basis_rank=776,
    )

    assert scales_seen == [0.25, 0.5, 1.0]
    assert basis_ranks_seen == [776, 776, 776]
    assert result["scale_run_count"] == 3
    assert result["every_requested_scale_separately_folded"] is True
    assert result["every_requested_scale_separately_decoded"] is True
    assert result["every_requested_scale_retained"] is True
    assert result["automatic_scale_winner_selected"] is False
    assert result["numeric_or_manual_quality_gate"] is False
    assert result["dense_parameter_compaction"] == compaction
    assert all(run["selected_as_winner"] is False for run in result["scale_runs"])


@pytest.mark.parametrize("invalid_rank", (0, -1, True))
def test_global_residual_basis_rank_must_be_positive_or_omitted(
    tmp_path: Path,
    invalid_rank: object,
) -> None:
    paths = _sources(tmp_path)
    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="global_residual_basis_rank",
    ):
        runner.run_fold_to_review(
            **_arguments(tmp_path, paths),
            global_residual_basis_rank=invalid_rank,
        )


@pytest.mark.parametrize(
    "malformation",
    ("no-candidates", "review-incomplete", "checkpoints-not-retained"),
)
def test_correction_scale_frontier_refuses_to_overclaim_member_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)

    def run_member(**_kwargs):
        return {
            "candidate_count": 0 if malformation == "no-candidates" else 1,
            "all_materialized_candidates_received_whole_response_review": (
                malformation != "review-incomplete"
            ),
            "all_materialized_candidate_checkpoints_retained": (
                malformation != "checkpoints-not-retained"
            ),
            "dense_parameter_compaction": {
                "parent_unique_parameters": 4,
                "child_parameter_element_count": 1,
            },
        }

    monkeypatch.setattr(runner, "_run_fold_to_review_scale", run_member)
    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="complete decoded review with every checkpoint retained",
    ):
        runner.run_fold_to_review(
            **arguments,
            global_residual_closure_version="producer_aware_v2a",
            correction_scale_frontier=(0.5, 1.0),
            correction_scale=1.0,
            global_residual_backend="cpu",
        )


def _seal_review_inputs(
    paths: dict[str, Path],
    arguments: dict[str, object],
    *,
    parent_identity: str = "a" * 64,
    tokenizer_identity: str = "b" * 64,
    maximum_prompt_tokens: int = 1024,
    maximum_new_tokens: int = 256,
) -> dict[str, object]:
    paths["prompts"].write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "id": "review-1",
                        "prompt": "Explain why the sky looks blue.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prompts = runner.load_review_prompts(paths["prompts"])
    prompt_set_sha256 = runner.canonical_sha256([dict(row) for row in prompts])
    responses = [
        {
            "id": "review-1",
            "category": "general",
            "response": "Because blue light scatters strongly.",
            "response_token_ids": [1, 2, 3],
            "diagnostics": {},
        }
    ]
    body = {
        "schema": "neural-forge-lfm2-whole-response-parent-envelope-v1",
        "status": "complete-one-time-parent-response-cache",
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": 1,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "parent_state_sha256": "d" * 64,
        "parent_identity_sha256": parent_identity,
        "parent_unique_parameters": 1_000_000,
        "tokenizer_sha256": tokenizer_identity,
        "responses": responses,
        "response_bank_sha256": runner.canonical_sha256(responses),
    }
    envelope = {**body, "envelope_sha256": runner.canonical_sha256(body)}
    paths["parent-envelope"].write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    arguments["expected_prompts_file_sha256"] = runner._file_sha256(paths["prompts"])
    arguments["expected_parent_envelope_sha256"] = envelope["envelope_sha256"]
    arguments["expected_parent_envelope_file_sha256"] = runner._file_sha256(
        paths["parent-envelope"]
    )
    arguments["maximum_prompt_tokens"] = maximum_prompt_tokens
    arguments["maximum_new_tokens"] = maximum_new_tokens
    return envelope


def _install_manifest_preflight(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    *,
    parent_identity: str,
) -> None:
    monkeypatch.setattr(
        runner,
        "open_lfm2_circuit_trace_cache",
        lambda _path, **_kwargs: SimpleNamespace(
            cache_key=arguments["expected_cache_key"],
            parent_fingerprint=parent_identity,
            manifest={
                "cache_key": arguments["expected_cache_key"],
                "manifest_sha256": "f" * 64,
                "parent": {"identity_sha256": parent_identity},
            },
        ),
    )


@pytest.mark.parametrize(
    ("recipe_id", "nonblocking"),
    (
        ("structural_sequence_residual_mlp_writer", False),
        ("nonlinear_qdeim_coupled_attention_structural", True),
    ),
)
def test_one_command_driver_materializes_and_decodes_the_complete_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe_id: str,
    nonblocking: bool,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    recipe = runner.resolve_lfm2_semantic_body_recipe(recipe_id)
    arguments["body_recipe"] = recipe.recipe_id
    arguments["expected_body_recipe_sha256"] = recipe.recipe_sha256
    arguments["global_residual_basis_rank"] = 776 if nonblocking else None
    candidate_ids = ("writer-a", "writer-b")
    events: list[str] = []
    references: dict[str, object] = {}
    parent_state = "c" * 64
    parent_identity = "a" * 64
    parent_unique_parameters = 1_000_000
    tokenizer_identity = "b" * 64
    prompt_rows = (
        {
            "id": "review-1",
            "prompt": "First prompt",
            "category": "general",
            "context": None,
            "parent_response": None,
        },
        {
            "id": "review-2",
            "prompt": "Second prompt",
            "category": "general",
            "context": None,
            "parent_response": None,
        },
    )
    prompt_set_sha256 = runner.canonical_sha256([dict(row) for row in prompt_rows])
    parent_responses = [
        {
            "id": row["id"],
            "response": f"Parent response for {row['id']}",
            "response_token_ids": [1, 2, 3],
            "diagnostics": {},
        }
        for row in prompt_rows
    ]
    response_bank_sha256 = runner.canonical_sha256(parent_responses)

    monkeypatch.setattr(
        runner,
        "load_local_tokenizer",
        lambda path: SimpleNamespace(name_or_path=str(path)),
    )
    monkeypatch.setattr(runner, "load_review_prompts", lambda _path: prompt_rows)
    monkeypatch.setattr(
        runner,
        "_load_parent_envelope",
        lambda *_args, **_kwargs: {
            "envelope_sha256": str(arguments["expected_parent_envelope_sha256"]),
            "parent_state_sha256": parent_state,
            "parent_identity_sha256": parent_identity,
            "parent_unique_parameters": parent_unique_parameters,
            "tokenizer_sha256": tokenizer_identity,
            "prompt_count": len(prompt_rows),
            "maximum_prompt_tokens": arguments["maximum_prompt_tokens"],
            "maximum_new_tokens": arguments["maximum_new_tokens"],
            "responses": parent_responses,
            "response_bank_sha256": response_bank_sha256,
        },
    )
    monkeypatch.setattr(
        runner,
        "fingerprint_local_tokenizer",
        lambda _path: tokenizer_identity,
    )
    monkeypatch.setattr(
        runner,
        "open_lfm2_circuit_trace_cache",
        lambda _path, **_kwargs: SimpleNamespace(
            cache_key=arguments["expected_cache_key"],
            parent_fingerprint=parent_identity,
            manifest={
                "cache_key": arguments["expected_cache_key"],
                "manifest_sha256": "f" * 64,
                "parent": {"identity_sha256": parent_identity},
            },
        ),
    )
    monkeypatch.setattr(runner, "_release_cuda", lambda: None)

    def publish_bank(
        _result,
        _context,
        *,
        preterminal_capsule_reference,
        source_roots,
        output_directory,
    ):
        events.append("bank-export")
        assert isinstance(
            preterminal_capsule_reference,
            runner.Lfm2SemanticPreterminalCapsuleReference,
        )
        assert source_roots.preterminal_capsule_directory == (
            preterminal_capsule_reference.path
        )
        Path(output_directory).mkdir()
        reference = runner.Lfm2SemanticTerminalFrontierBankReference(
            path=Path(output_directory),
            receipt_sha256="b" * 64,
            receipt_file_sha256="c" * 64,
            manifest_sha256="d" * 64,
            preterminal_capsule_receipt_sha256=(
                preterminal_capsule_reference.receipt_sha256
            ),
            terminal_result_sha256="e" * 64,
            source_roots_sha256=source_roots.source_roots_sha256,
            candidate_ids=candidate_ids,
        )
        references["bank"] = reference
        references["source_roots"] = source_roots
        return reference

    monkeypatch.setattr(
        runner,
        "publish_lfm2_semantic_terminal_frontier_bank",
        publish_bank,
    )

    def run_postfold(**kwargs):
        events.append("postfold")
        assert kwargs["terminal_frontier_exporter_factory"]
        assert kwargs["body_recipe"].recipe_id == recipe_id
        capsule_path = Path(kwargs["preterminal_capsule_directory"])
        capsule_path.mkdir()
        capsule = runner.Lfm2SemanticPreterminalCapsuleReference(
            path=capsule_path,
            receipt_sha256="a" * 64,
            receipt_file_sha256="1" * 64,
            manifest_sha256="2" * 64,
            bindings_sha256="3" * 64,
            context_binding_sha256="4" * 64,
        )
        exporter = kwargs["terminal_frontier_exporter_factory"](capsule)
        bank = exporter(SimpleNamespace(), {"context_sha256": "5" * 64})
        context = {
            "preterminal_resume_mode": False,
            "same_live_cache_session_as_layer_stack": True,
            "frontier_export_occurs_before_final_w2_installation": True,
        }
        fused = {
            "result_sha256": "6" * 64,
            "terminal_frontier_export": bank.as_dict(),
            "terminal_frontier_export_context": context,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "final_selection_performed": False,
            "operation_audit": dict(runner._ZERO_OPERATION_AUDIT),
        }
        parent_session = _sealed_direct_parent_session(
            parent_identity_sha256=parent_identity,
            parameter_count=parent_unique_parameters,
        )
        stack_body = {
            "parameter_element_count": 250_000,
            "parent_checkpoint_session_sha256": parent_session["session_sha256"],
            "parent_checkpoint_session": parent_session,
        }
        semantic_layer_stack = {
            **stack_body,
            "result_sha256": runner.canonical_sha256(stack_body),
        }
        return {
            "status": "terminal-frontier-exported-review-pending",
            "result_sha256": "7" * 64,
            "final_selection_status": ("deferred-to-manual-whole-response-review"),
            "final_selected_candidate_id": None,
            "final_selection_performed": False,
            "automatic_candidate_selection_performed": False,
            "automatic_candidate_acceptance_performed": False,
            "candidate_published_before_manual_review": False,
            "checkpoint_published": False,
            "semantic_evidence_cache_key": arguments["expected_cache_key"],
            "global_residual_closure_version": kwargs[
                "global_residual_closure_version"
            ],
            "global_residual_backend": kwargs["global_residual_backend"],
            "global_residual_requested_basis_rank": kwargs[
                "global_residual_basis_rank"
            ],
            "global_residual_correction_scale_frontier": list(
                kwargs["correction_scale_frontier"]
            ),
            "global_residual_correction_scale": kwargs["correction_scale"],
            "global_residual_frontier_member_selected_as_winner": False,
            "semantic_layer_stack": semantic_layer_stack,
            "cache_parent_fingerprint": parent_identity,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "preterminal_capsule": capsule.as_dict(),
            "terminal_frontier_export": bank.as_dict(),
            "fused_semantic_terminal": fused,
            "pipeline_cache_session_reuse": {
                "same_pipeline_session_reused_by_layer_stack_and_terminal": True
            },
            "operation_audit": dict(runner._ZERO_OPERATION_AUDIT),
        }

    monkeypatch.setattr(runner, "run_lfm2_semantic_postfold", run_postfold)

    def load_bank(*_args, **_kwargs):
        events.append("bank-load")
        bank = references["bank"]
        return SimpleNamespace(
            reference=bank,
            manifest={
                # Production loader deep-freezes JSON arrays to tuples.  Keep
                # the integration fixture faithful so list/tuple comparisons
                # cannot strand an already-exported frontier after a long fold.
                "structurally_eligible_candidate_ids": tuple(candidate_ids),
                "structurally_eligible_candidate_count": len(candidate_ids),
                "all_structurally_eligible_candidates_exported": True,
                "structurally_ineligible_candidates_exported": False,
                "structural_safety_is_only_candidate_veto": True,
                "candidate_metric_threshold_used_for_export": False,
            },
        )

    monkeypatch.setattr(
        runner,
        "load_lfm2_semantic_terminal_frontier_bank",
        load_bank,
    )

    def materialize(**kwargs):
        candidate_id = kwargs["candidate_id"]
        events.append(f"materialize:{candidate_id}")
        assert kwargs["expected_source_roots"] is references["source_roots"]
        destination = Path(kwargs["output_directory"])
        checkpoint = destination / "checkpoint"
        checkpoint.mkdir(parents=True)
        return SimpleNamespace(
            as_dict=lambda: {
                "candidate_id": candidate_id,
                "candidate_state_sha256": "8" * 64,
                "model_state_sha256": "9" * 64,
                "path": str(destination),
                "checkpoint_path": str(checkpoint),
            }
        )

    monkeypatch.setattr(
        runner,
        "materialize_lfm2_semantic_terminal_frontier_candidate",
        materialize,
    )

    def review(**kwargs):
        events.append("whole-response-review")
        frontier = json.loads(
            Path(kwargs["frontier_run_receipt"]).read_text(encoding="utf-8")
        )
        assert kwargs["candidate_ids"] == ()
        assert (
            kwargs["expected_parent_envelope_sha256"]
            == (arguments["expected_parent_envelope_sha256"])
        )
        assert kwargs["expected_parent_envelope_file_sha256"] == runner._file_sha256(
            paths["parent-envelope"]
        )
        assert frontier["candidate_ids"] == list(candidate_ids)
        body = {
            "reviewed_candidate_count": len(candidate_ids),
            "requested_candidate_ids": list(candidate_ids),
            "prompt_count": 2,
            "prompt_set_sha256": prompt_set_sha256,
            "maximum_prompt_tokens": arguments["maximum_prompt_tokens"],
            "maximum_new_tokens": arguments["maximum_new_tokens"],
            "parent_response_envelope": {
                "envelope_sha256": arguments["expected_parent_envelope_sha256"],
                "parent_state_sha256": parent_state,
                "parent_identity_sha256": parent_identity,
                "parent_unique_parameters": parent_unique_parameters,
                "tokenizer_sha256": tokenizer_identity,
                "maximum_new_tokens": arguments["maximum_new_tokens"],
                "response_bank_sha256": response_bank_sha256,
            },
            "evaluation_policy": {
                "decoded_whole_responses_are_authoritative": True,
                "diagnostic_numbers_have_decision_authority": False,
            },
        }
        dossier = {**body, "dossier_sha256": runner.canonical_sha256(body)}
        runner._atomic_json(Path(kwargs["output_json"]), dossier)
        return dossier

    monkeypatch.setattr(runner, "review_frontier", review)

    def prepare(**kwargs):
        events.append("qualitative-preparation")
        assert nonblocking is False
        assert kwargs["lane_labels"] == (recipe_id,)
        assert len(kwargs["dossier_paths"]) == 1
        Path(kwargs["merged_json"]).write_text("{}\n", encoding="utf-8")
        Path(kwargs["markdown_path"]).write_text("# Review\n", encoding="utf-8")
        Path(kwargs["verdict_template_json"]).write_text("{}\n", encoding="utf-8")
        return {
            "status": "qualitative-review-artifacts-prepared",
            "candidate_count": len(candidate_ids),
            "automatic_decision_performed": False,
        }

    monkeypatch.setattr(runner, "prepare_qualitative_review", prepare)

    result = runner.run_fold_to_review(**arguments)

    expected_events = [
        "postfold",
        "bank-export",
        "bank-load",
        "materialize:writer-a",
        "materialize:writer-b",
        "whole-response-review",
    ]
    if not nonblocking:
        expected_events.append("qualitative-preparation")
    assert events == expected_events
    assert result["candidate_count"] == len(candidate_ids)
    assert result["automatic_candidate_selection_performed"] is False
    assert result["automatic_candidate_acceptance_performed"] is False
    assert result["global_residual_closure_version"] == "producer_aware_v2a"
    assert result["global_residual_requested_basis_rank"] == (
        776 if nonblocking else None
    )
    assert result["global_residual_correction_scale_frontier"] == [1.0]
    assert result["global_residual_correction_scale"] == 1.0
    assert result["global_residual_frontier_member_selected_as_winner"] is False
    compaction = result["dense_parameter_compaction"]
    assert compaction["parent_unique_parameters"] == 1_000_000
    assert compaction["child_parameter_element_count"] == 250_000
    assert compaction["parameter_reduction"] == 750_000
    assert compaction["parent_to_child_parameter_ratio"] == 4.0
    assert compaction["child_parameter_fraction_of_parent"] == 0.25
    assert compaction["parameter_reduction_fraction"] == 0.75
    assert compaction["parent_count_source"] == (
        "authenticated_direct_local_safetensor_tensor_inventory"
    )
    assert compaction["envelope_parent_count_cross_checked"] is True
    assert (
        compaction["local_parent_parameter_inventory"][
            "unique_stored_parameter_elements"
        ]
        == parent_unique_parameters
    )
    assert compaction["counts_recomputed_from_authenticated_values"] is True
    assert compaction["dense_parameter_compaction_is_quality_gate"] is False
    assert result["blank_manual_verdict_prepared"] is (not nonblocking)
    if nonblocking:
        assert "candidate_published_before_manual_review" not in result
        assert result["schema"] == runner.NONBLOCKING_PIPELINE_SCHEMA
        assert (
            result["status"]
            == "decoded-behavior-evidence-complete-continuation-unblocked"
        )
        assert result["continuation_unblocked"] is True
        assert result["decoded_behavior_evidence_complete"] is True
        assert result["manual_verdict_required_for_continuation"] is False
        assert result["continuation_requires_manual_review_or_verdict"] is False
        assert result["manual_verdict_artifact_created"] is False
        assert result["all_materialized_candidate_checkpoints_retained"] is True
        assert result["retained_candidate_checkpoint_ids"] == list(candidate_ids)
        assert result["retained_candidate_checkpoint_count"] == len(candidate_ids)
        assert (
            result["candidate_checkpoint_retention_depends_on_manual_verdict"] is False
        )
        assert result["quality_evidence_can_stop_iteration"] is False
        assert result["numeric_accept_veto_publish_delete_stop_authority"] is False
        assert (
            result["manual_or_binary_accept_veto_publish_delete_stop_authority"]
            is False
        )
        assert result["behavior_flow"]["body_recipe_sha256"] == (recipe.recipe_sha256)
        assert (
            result["behavior_flow_policy_sha256"]
            == (result["behavior_flow"]["policy_sha256"])
        )
        assert (
            result["legacy_v1_pending_terminal_receipt_interpretation"][
                "interpreted_as_frontier_handoff_only"
            ]
            is True
        )
        assert (
            result["legacy_v1_artifact_interpretation"][
                "nested_manual_review_required_or_authoritative_labels_are_historical_only"
            ]
            is True
        )
        assert (
            result["legacy_v1_artifact_interpretation"][
                "unchanged_v1_body_recipe_manual_authority_label_is_historical_only"
            ]
            is True
        )
        assert (
            result["legacy_v1_artifact_interpretation"][
                "nested_v1_quality_labels_have_v2_continuation_authority"
            ]
            is False
        )
        observations = result["optional_behavior_observations"]
        assert observations["observations"] == []
        assert observations["manual_input_required"] is False
        assert observations["binary_verdict_requested"] is False
        assert observations["accept_veto_publish_delete_stop_authority"] is False
        assert not (
            Path(arguments["run_directory"]) / "manual-verdict-template.json"
        ).exists()
    else:
        assert result["candidate_published_before_manual_review"] is False
        assert result["schema"] == runner.PIPELINE_SCHEMA
        assert result["status"] == (
            "ready-for-manual-whole-response-intelligence-review"
        )
    assert not (Path(arguments["run_directory"]) / "unselected-checkpoint").exists()
    persisted = json.loads(
        (Path(arguments["run_directory"]) / "pipeline-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == result


def test_prompt_pin_rejected_before_run_directory_or_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    arguments["expected_prompts_file_sha256"] = "f" * 64
    monkeypatch.setattr(
        runner,
        "load_local_tokenizer",
        lambda _path: pytest.fail("wrong prompt pin loaded a tokenizer"),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="prompt bytes changed",
    ):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


def test_nonblocking_checkpoint_retention_proof_fails_closed(
    tmp_path: Path,
) -> None:
    materializations = tmp_path / "materializations"
    materializations.mkdir()
    missing_checkpoint = materializations / "writer-a" / "checkpoint"
    frontier = {
        "materializations_directory": str(materializations),
        "candidate_ids": ["writer-a"],
        "candidate_count": 1,
        "materialized_count": 1,
        "all_retained_candidates_materialized": True,
        "materializations": [
            {
                "candidate_id": "writer-a",
                "checkpoint_path": str(missing_checkpoint),
            }
        ],
    }

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="unretained candidate checkpoint",
    ):
        runner._retained_candidate_checkpoint_ids(frontier)


def test_wrong_model_sealed_parent_envelope_fails_before_run_or_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    envelope = _seal_review_inputs(paths, arguments, parent_identity="a" * 64)
    monkeypatch.setattr(
        runner,
        "fingerprint_local_tokenizer",
        lambda _path: envelope["tokenizer_sha256"],
    )
    _install_manifest_preflight(
        monkeypatch,
        arguments,
        parent_identity="c" * 64,
    )
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **_kwargs: pytest.fail("wrong-model envelope reached folding"),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="do not share one parent identity",
    ):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


def test_wrong_tokenizer_envelope_fails_before_run_or_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    _seal_review_inputs(paths, arguments, tokenizer_identity="b" * 64)
    monkeypatch.setattr(
        runner,
        "fingerprint_local_tokenizer",
        lambda _path: "c" * 64,
    )
    monkeypatch.setattr(
        runner,
        "open_lfm2_circuit_trace_cache",
        lambda *_args, **_kwargs: pytest.fail(
            "wrong tokenizer reached cache authentication"
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **_kwargs: pytest.fail("wrong-tokenizer envelope reached folding"),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="different donor tokenizer",
    ):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


@pytest.mark.parametrize("stale_pin", ("file", "envelope", "seal"))
def test_stale_parent_envelope_pins_fail_before_run_or_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_pin: str,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    envelope = _seal_review_inputs(paths, arguments)
    if stale_pin == "file":
        arguments["expected_parent_envelope_file_sha256"] = "0" * 64
    elif stale_pin == "envelope":
        arguments["expected_parent_envelope_sha256"] = "0" * 64
    else:
        envelope["status"] = "tampered-after-seal"
        paths["parent-envelope"].write_text(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        arguments["expected_parent_envelope_file_sha256"] = runner._file_sha256(
            paths["parent-envelope"]
        )
    monkeypatch.setattr(
        runner,
        "fingerprint_local_tokenizer",
        lambda _path: pytest.fail("stale envelope reached tokenizer fingerprint"),
    )
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **_kwargs: pytest.fail("stale envelope reached folding"),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="Parent response envelope failed preflight authentication",
    ):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


@pytest.mark.parametrize(
    "malformation",
    (
        "missing-parent-state",
        "missing-parent-parameter-count",
        "boolean-parent-parameter-count",
        "nonpositive-parent-parameter-count",
        "boolean-prompt-count",
        "missing-response-text",
        "boolean-response-token-id",
        "negative-response-token-id",
        "missing-response-diagnostics",
        "wrong-response-bank",
    ),
)
def test_canonically_sealed_malformed_parent_envelope_fails_before_run_or_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    envelope = _seal_review_inputs(paths, arguments)
    body = dict(envelope)
    body.pop("envelope_sha256")
    if malformation == "missing-parent-state":
        body.pop("parent_state_sha256")
    elif malformation == "missing-parent-parameter-count":
        body.pop("parent_unique_parameters")
    elif malformation == "boolean-parent-parameter-count":
        body["parent_unique_parameters"] = True
    elif malformation == "nonpositive-parent-parameter-count":
        body["parent_unique_parameters"] = 0
    elif malformation == "boolean-prompt-count":
        body["prompt_count"] = True
    elif malformation == "missing-response-text":
        body["responses"][0].pop("response")
        body["response_bank_sha256"] = runner.canonical_sha256(body["responses"])
    elif malformation == "boolean-response-token-id":
        body["responses"][0]["response_token_ids"] = [True]
        body["response_bank_sha256"] = runner.canonical_sha256(body["responses"])
    elif malformation == "negative-response-token-id":
        body["responses"][0]["response_token_ids"] = [-1]
        body["response_bank_sha256"] = runner.canonical_sha256(body["responses"])
    elif malformation == "missing-response-diagnostics":
        body["responses"][0].pop("diagnostics")
        body["response_bank_sha256"] = runner.canonical_sha256(body["responses"])
    else:
        body["response_bank_sha256"] = "0" * 64
    malformed = {**body, "envelope_sha256": runner.canonical_sha256(body)}
    paths["parent-envelope"].write_text(
        json.dumps(malformed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    arguments["expected_parent_envelope_sha256"] = malformed["envelope_sha256"]
    arguments["expected_parent_envelope_file_sha256"] = runner._file_sha256(
        paths["parent-envelope"]
    )
    monkeypatch.setattr(
        runner,
        "fingerprint_local_tokenizer",
        lambda _path: pytest.fail("malformed envelope reached tokenizer fingerprint"),
    )
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **_kwargs: pytest.fail("malformed envelope reached folding"),
    )

    with pytest.raises(runner.Lfm2SemanticFoldToReviewError):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_prompt_tokens", 0),
        ("maximum_prompt_tokens", 8193),
        ("maximum_new_tokens", 15),
        ("maximum_new_tokens", 2049),
    ),
)
def test_invalid_review_horizon_fails_before_any_input_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    arguments[field] = value
    monkeypatch.setattr(
        runner,
        "load_review_prompts",
        lambda _path: pytest.fail("invalid horizon reached prompt loading"),
    )
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **_kwargs: pytest.fail("invalid horizon reached folding"),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match=f"{field} must be between",
    ):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


def test_parent_envelope_horizon_mismatch_fails_before_run_or_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    _seal_review_inputs(paths, arguments, maximum_new_tokens=512)
    arguments["maximum_new_tokens"] = 256
    monkeypatch.setattr(
        runner,
        "fingerprint_local_tokenizer",
        lambda _path: pytest.fail("horizon mismatch reached tokenizer fingerprint"),
    )
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **_kwargs: pytest.fail("horizon mismatch reached folding"),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="different review horizon",
    ):
        runner.run_fold_to_review(**arguments)

    assert not Path(arguments["run_directory"]).exists()


def test_postfold_parent_mismatch_fails_before_frontier_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    parent_identity = "a" * 64
    preflight = runner._AuthenticatedReviewPreflight(
        prompt_set_sha256="1" * 64,
        prompt_ids=("review-1",),
        envelope_sha256=str(arguments["expected_parent_envelope_sha256"]),
        envelope_file_sha256=runner._file_sha256(paths["parent-envelope"]),
        parent_state_sha256="9" * 64,
        parent_identity_sha256=parent_identity,
        parent_unique_parameters=1_000_000,
        tokenizer_sha256="b" * 64,
        response_bank_sha256="8" * 64,
        cache_key=str(arguments["expected_cache_key"]),
        cache_manifest_sha256="c" * 64,
        maximum_prompt_tokens=int(arguments["maximum_prompt_tokens"]),
        maximum_new_tokens=int(arguments["maximum_new_tokens"]),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_review_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(
        runner,
        "load_local_tokenizer",
        lambda path: SimpleNamespace(name_or_path=str(path)),
    )

    bank_references = []

    def publish_bank(
        _result,
        _context,
        *,
        preterminal_capsule_reference,
        source_roots,
        output_directory,
    ):
        Path(output_directory).mkdir()
        reference = runner.Lfm2SemanticTerminalFrontierBankReference(
            path=Path(output_directory),
            receipt_sha256="d" * 64,
            receipt_file_sha256="e" * 64,
            manifest_sha256="f" * 64,
            preterminal_capsule_receipt_sha256=(
                preterminal_capsule_reference.receipt_sha256
            ),
            terminal_result_sha256="3" * 64,
            source_roots_sha256=source_roots.source_roots_sha256,
            candidate_ids=("writer-a",),
        )
        bank_references.append(reference)
        return reference

    monkeypatch.setattr(
        runner,
        "publish_lfm2_semantic_terminal_frontier_bank",
        publish_bank,
    )

    def run_postfold(**kwargs):
        capsule_path = Path(kwargs["preterminal_capsule_directory"])
        capsule_path.mkdir()
        capsule = runner.Lfm2SemanticPreterminalCapsuleReference(
            path=capsule_path,
            receipt_sha256="4" * 64,
            receipt_file_sha256="5" * 64,
            manifest_sha256="6" * 64,
            bindings_sha256="7" * 64,
            context_binding_sha256="8" * 64,
        )
        exporter = kwargs["terminal_frontier_exporter_factory"](capsule)
        bank = exporter(SimpleNamespace(), {"context_sha256": "9" * 64})
        context = {
            "preterminal_resume_mode": False,
            "same_live_cache_session_as_layer_stack": True,
            "frontier_export_occurs_before_final_w2_installation": True,
        }
        fused = {
            "result_sha256": "0" * 64,
            "terminal_frontier_export": bank.as_dict(),
            "terminal_frontier_export_context": context,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "final_selection_performed": False,
            "operation_audit": dict(runner._ZERO_OPERATION_AUDIT),
        }
        return {
            "status": "terminal-frontier-exported-review-pending",
            "result_sha256": "1" * 64,
            "final_selection_status": ("deferred-to-manual-whole-response-review"),
            "final_selected_candidate_id": None,
            "final_selection_performed": False,
            "automatic_candidate_selection_performed": False,
            "automatic_candidate_acceptance_performed": False,
            "candidate_published_before_manual_review": False,
            "checkpoint_published": False,
            "semantic_evidence_cache_key": preflight.cache_key,
            "cache_parent_fingerprint": "2" * 64,
            "terminal_writer_install_count": 0,
            "terminal_writer_mutation_count": 0,
            "preterminal_capsule": capsule.as_dict(),
            "terminal_frontier_export": bank.as_dict(),
            "fused_semantic_terminal": fused,
            "pipeline_cache_session_reuse": {
                "same_pipeline_session_reused_by_layer_stack_and_terminal": True
            },
            "operation_audit": dict(runner._ZERO_OPERATION_AUDIT),
        }

    monkeypatch.setattr(runner, "run_lfm2_semantic_postfold", run_postfold)
    monkeypatch.setattr(
        runner,
        "load_lfm2_semantic_terminal_frontier_bank",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched postfold parent reached materialization"
        ),
    )

    with pytest.raises(
        runner.Lfm2SemanticFoldToReviewError,
        match="Continuous postfold did not stop",
    ):
        runner.run_fold_to_review(**arguments)

    assert len(bank_references) == 1
    assert not (Path(arguments["run_directory"]) / "materializations").exists()


def test_cli_forwards_pinned_recipe_and_review_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "run_fold_to_review",
        lambda **kwargs: calls.append(kwargs) or {"status": "ready"},
    )
    cli = []
    flag_names = {
        "run_directory": "run-directory",
        "input_checkpoint": "input-checkpoint",
        "trace_cache": "trace-cache",
        "bridge_directory": "bridge-directory",
        "donor_tokenizer_directory": "donor-tokenizer-directory",
        "target_tokenizer_directory": "target-tokenizer-directory",
        "parent_model_directory": "parent-model-directory",
        "compiler_diagnostic_json": "compiler-diagnostic-json",
        "interface_statistics_directory": "interface-statistics-directory",
        "interface_frontier_companion_json": "interface-frontier-companion-json",
        "prompts_json": "prompts-json",
        "parent_envelope": "parent-envelope",
    }
    for name, value in arguments.items():
        flag = flag_names.get(name, name.replace("_", "-"))
        cli.extend([f"--{flag}", str(value)])

    assert runner.main(cli) == 0
    assert len(calls) == 1
    assert calls[0]["body_recipe"] == "structural_sequence_residual_mlp_writer"
    assert calls[0]["maximum_new_tokens"] == 256
    assert calls[0]["anchor_candidate_index"] == 7
