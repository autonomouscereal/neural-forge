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
    arguments["expected_prompts_file_sha256"] = runner._file_sha256(
        paths["prompts"]
    )
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


def test_one_command_driver_exports_once_materializes_all_and_stops_for_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sources(tmp_path)
    arguments = _arguments(tmp_path, paths)
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
        assert kwargs["body_recipe"].recipe_id == (
            "structural_sequence_residual_mlp_writer"
        )
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
        return {
            "status": "terminal-frontier-exported-review-pending",
            "result_sha256": "7" * 64,
            "final_selection_status": (
                "deferred-to-manual-whole-response-review"
            ),
            "final_selected_candidate_id": None,
            "final_selection_performed": False,
            "automatic_candidate_selection_performed": False,
            "automatic_candidate_acceptance_performed": False,
            "candidate_published_before_manual_review": False,
            "checkpoint_published": False,
            "semantic_evidence_cache_key": arguments["expected_cache_key"],
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
        assert kwargs["expected_parent_envelope_sha256"] == (
            arguments["expected_parent_envelope_sha256"]
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
        assert kwargs["lane_labels"] == (
            "structural_sequence_residual_mlp_writer",
        )
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

    assert events == [
        "postfold",
        "bank-export",
        "bank-load",
        "materialize:writer-a",
        "materialize:writer-b",
        "whole-response-review",
        "qualitative-preparation",
    ]
    assert result["candidate_count"] == len(candidate_ids)
    assert result["automatic_candidate_selection_performed"] is False
    assert result["automatic_candidate_acceptance_performed"] is False
    assert result["candidate_published_before_manual_review"] is False
    assert result["blank_manual_verdict_prepared"] is True
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
            "final_selection_status": (
                "deferred-to-manual-whole-response-review"
            ),
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
