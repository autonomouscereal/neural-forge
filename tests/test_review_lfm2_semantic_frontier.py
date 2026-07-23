from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import review_lfm2_semantic_frontier as review


def _write_run(path: Path, candidate_ids: tuple[str, ...]) -> dict[str, object]:
    materializations = [
        {
            "candidate_id": candidate_id,
            "candidate_state_sha256": f"{index + 1:x}" * 64,
            "model_state_sha256": f"{index + 4:x}" * 64,
            "checkpoint_path": str(path.parent / f"child-{index}" / "checkpoint"),
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]
    for row in materializations:
        Path(str(row["checkpoint_path"])).mkdir(parents=True)
    body = {
        "schema": "neural-forge-lfm2-semantic-terminal-frontier-run-v1",
        "status": "complete",
        "candidate_ids": list(candidate_ids),
        "candidate_count": len(candidate_ids),
        "materialized_count": len(candidate_ids),
        "materializations": materializations,
        "all_retained_candidates_materialized": True,
        "candidate_metric_threshold_used_for_materialization": False,
    }
    document = {**body, "run_sha256": review.canonical_sha256(body)}
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def _write_prompts(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "id": "reasoning",
                        "category": "reasoning",
                        "prompt": "A box has 3 red and 2 blue balls. Explain the total.",
                        "parent_response": "There are five balls in total.",
                    },
                    {
                        "id": "chat",
                        "category": "conversation",
                        "messages": [{"role": "user", "content": "Say hello."}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_review_frontier_preserves_all_outputs_for_multidimensional_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_ids = ("candidate-a", "candidate-b")
    run_path = tmp_path / "run.json"
    run = _write_run(run_path, candidate_ids)
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path)
    output = tmp_path / "dossier.json"
    calls = []

    def review_child(
        materialization: dict[str, object],
        prompts: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        candidate_id = str(materialization["candidate_id"])
        calls.append(candidate_id)
        return {
            "candidate_id": candidate_id,
            "candidate_state_sha256": materialization["candidate_state_sha256"],
            "model_state_sha256": materialization["model_state_sha256"],
            "checkpoint_path": materialization["checkpoint_path"],
            "unique_parameters": 57_000_000,
            "responses": [
                {
                    "id": "reasoning",
                    "response": f"whole response from {candidate_id}",
                    "diagnostics": {
                        "interpretation": "diagnostic-only; decoded response controls review"
                    },
                }
                for _prompt in prompts
            ],
        }

    monkeypatch.setattr(review, "_review_one_child", review_child)
    document = review.review_frontier(
        frontier_run_receipt=run_path,
        expected_frontier_run_sha256=str(run["run_sha256"]),
        expected_frontier_run_file_sha256=None,
        prompts_json=prompts_path,
        output_json=output,
        candidate_ids=(),
        device="cpu",
        dtype="bfloat16",
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
    )

    assert calls == list(candidate_ids)
    assert document["status"] == "decoded-whole-response-evidence-complete"
    assert document["requested_candidate_ids"] == list(candidate_ids)
    assert document["reviewed_candidate_count"] == len(candidate_ids)
    assert document["maximum_new_tokens"] == 256
    assert document["evaluation_policy"] == {
        "decoded_whole_responses_are_primary_evidence": True,
        "complete_decoded_responses_preserved": True,
        "diagnostic_numbers_may_guide_navigation": True,
        "diagnostic_numbers_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
        "exact_parent_token_match_required": False,
        "multidimensional_behavior_observations_encouraged": True,
        "manual_observations_required_for_continuation": False,
        "single_binary_manual_verdict_has_decision_authority": False,
        "retention_target_is_objective_not_cutoff": True,
        "parent_comparison_uses_cached_text_only": False,
    }
    assert document["operation_audit"]["training_steps"] == 0
    assert document["operation_audit"]["parent_forward_passes"] == 0
    assert json.loads(output.read_text(encoding="utf-8")) == document
    progress = json.loads(
        (tmp_path / "dossier.progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
    assert progress["completed_candidate_ids"] == list(candidate_ids)
    assert progress["complete_decoded_responses_preserved"] is True
    assert progress["multidimensional_observations_optional"] is True
    assert progress["binary_manual_verdict_required"] is False


def test_review_frontier_can_review_named_subset_without_numeric_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_ids = ("candidate-a", "candidate-b")
    run_path = tmp_path / "run.json"
    run = _write_run(run_path, candidate_ids)
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path)
    monkeypatch.setattr(
        review,
        "_review_one_child",
        lambda materialization, _prompts, **_kwargs: {
            "candidate_id": materialization["candidate_id"],
            "responses": [],
        },
    )
    document = review.review_frontier(
        frontier_run_receipt=run_path,
        expected_frontier_run_sha256=str(run["run_sha256"]),
        expected_frontier_run_file_sha256=None,
        prompts_json=prompts_path,
        output_json=tmp_path / "subset.json",
        candidate_ids=("candidate-b",),
        device="cpu",
        dtype="bfloat16",
        maximum_prompt_tokens=1024,
        maximum_new_tokens=512,
    )
    assert document["requested_candidate_ids"] == ["candidate-b"]
    assert document["maximum_new_tokens"] == 512
    assert (
        document["evaluation_policy"]["diagnostic_numbers_have_decision_authority"]
        is False
    )


def test_response_diagnostics_never_returns_a_verdict() -> None:
    diagnostics = review._response_diagnostics(
        [7, 8, 7, 8],
        "a coherent answer",
        maximum_new_tokens=4,
        eos_token_ids=frozenset(),
    )
    assert diagnostics["generation_limit_reached"] is True
    assert diagnostics["maximum_repeated_bigram_share"] > 0.0
    assert "accepted" not in diagnostics
    assert "rejected" not in diagnostics
    assert diagnostics["interpretation"].startswith("diagnostic-only")


def test_review_one_child_uses_materialization_state_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    expected = "a" * 64
    model = object()

    monkeypatch.setattr(review.residual, "_torch_dtype", lambda _dtype: object())
    monkeypatch.setattr(
        review.residual,
        "load_local_tokenizer",
        lambda _checkpoint: object(),
    )
    monkeypatch.setattr(
        review.residual,
        "load_local_lfm2_model",
        lambda _checkpoint, **_kwargs: model,
    )
    monkeypatch.setattr(
        review.residual,
        "model_state_sha256",
        lambda _model: (_ for _ in ()).throw(
            AssertionError("reviewer used the residual-runner hash domain")
        ),
    )
    monkeypatch.setattr(review, "fingerprint_model_state", lambda _model: expected)
    monkeypatch.setattr(review, "_generate_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(review.residual, "unique_parameter_count", lambda _model: 57)

    result = review._review_one_child(
        {
            "candidate_id": "candidate-a",
            "candidate_state_sha256": "b" * 64,
            "model_state_sha256": expected,
            "checkpoint_path": str(checkpoint),
        },
        (),
        device="cpu",
        dtype="bfloat16",
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
    )

    assert result["model_state_sha256"] == expected
    assert result["unique_parameters"] == 57


def test_cached_parent_text_is_attached_without_running_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate_ids = ("candidate-a",)
    run_path = tmp_path / "run.json"
    run = _write_run(run_path, candidate_ids)
    prompts_path = tmp_path / "prompts.json"
    _write_prompts(prompts_path)
    prompts = review.load_review_prompts(prompts_path)
    prompt_sha256 = review.canonical_sha256([dict(row) for row in prompts])
    parent_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "response": f"cached parent response {row['id']}",
            "response_token_ids": [9, 10],
            "diagnostics": {"interpretation": "diagnostic-only"},
        }
        for row in prompts
    ]
    parent_body = {
        "schema": review.PARENT_ENVELOPE_SCHEMA,
        "status": "complete-one-time-parent-response-cache",
        "prompt_set_sha256": prompt_sha256,
        "prompt_count": len(prompts),
        "maximum_new_tokens": 256,
        "parent_state_sha256": "a" * 64,
        "parent_identity_sha256": "b" * 64,
        "parent_unique_parameters": 229_000_000,
        "tokenizer_sha256": "c" * 64,
        "responses": parent_rows,
        "response_bank_sha256": review.canonical_sha256(parent_rows),
    }
    parent_document = {
        **parent_body,
        "envelope_sha256": review.canonical_sha256(parent_body),
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent_document), encoding="utf-8")

    monkeypatch.setattr(
        review,
        "_review_one_child",
        lambda materialization, prompt_rows, **_kwargs: {
            "candidate_id": materialization["candidate_id"],
            "responses": [
                {"id": row["id"], "response": f"child response {row['id']}"}
                for row in prompt_rows
            ],
        },
    )
    document = review.review_frontier(
        frontier_run_receipt=run_path,
        expected_frontier_run_sha256=str(run["run_sha256"]),
        expected_frontier_run_file_sha256=None,
        prompts_json=prompts_path,
        output_json=tmp_path / "compared.json",
        candidate_ids=(),
        device="cpu",
        dtype="bfloat16",
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
        parent_envelope=parent_path,
        expected_parent_envelope_sha256=parent_document["envelope_sha256"],
    )
    responses = document["candidate_reviews"][0]["responses"]
    assert responses[0]["parent_response"].startswith("cached parent response")
    assert responses[0]["parent_response_token_ids"] == [9, 10]
    assert (
        document["parent_response_envelope"]["envelope_sha256"]
        == (parent_document["envelope_sha256"])
    )
    assert (
        document["evaluation_policy"]["parent_comparison_uses_cached_text_only"] is True
    )
    assert document["operation_audit"]["parent_forward_passes"] == 0
