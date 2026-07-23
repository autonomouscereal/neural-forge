from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import prepare_lfm2_semantic_qualitative_review as review


def _diagnostics(tokens: int) -> dict[str, object]:
    return {
        "characters": 24,
        "tokens": tokens,
        "empty": False,
        "fragment_like": False,
        "unique_token_ratio": 0.75,
        "maximum_repeated_bigram_share": 0.1,
        "maximum_repeated_trigram_share": 0.05,
        "ended_with_eos": True,
        "generation_limit_reached": False,
        "interpretation": "diagnostic-only; decoded response controls review",
    }


def _write_dossier(
    path: Path,
    *,
    candidate_id: str,
    child_response: str,
    parent_response: str = "The cached parent explains the answer.",
) -> tuple[str, str]:
    prompt_set_sha256 = "1" * 64
    parent_envelope = {
        "path": "/sealed/parent-envelope.json",
        "envelope_sha256": "2" * 64,
        "parent_state_sha256": "3" * 64,
        "parent_identity_sha256": "4" * 64,
        "parent_unique_parameters": 229_693_184,
        "tokenizer_sha256": "5" * 64,
        "maximum_new_tokens": 256,
        "inference_device": "cuda:0",
        "inference_dtype": "bfloat16",
        "response_bank_sha256": "6" * 64,
    }
    responses = [
        {
            "id": "reasoning-one",
            "category": "reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain why two plus two equals four.",
                }
            ],
            "context": "Look for a coherent explanation, not matching wording.",
            "parent_response": parent_response,
            "parent_response_token_ids": [7, 8, 9],
            "parent_diagnostics": _diagnostics(8),
            "response": child_response,
            "response_token_ids": [10, 11, 12],
            "diagnostics": _diagnostics(7),
        }
    ]
    candidate = {
        "candidate_id": candidate_id,
        "candidate_state_sha256": "7" * 64,
        "model_state_sha256": "8" * 64,
        "checkpoint_path": f"/folded/{candidate_id}/checkpoint",
        "unique_parameters": 57_405_368,
        "responses": responses,
    }
    body = {
        "schema": review.DOSSIER_SCHEMA,
        "status": "decoded-whole-response-evidence-complete",
        "frontier_run_receipt": "/sealed/frontier-run.json",
        "frontier_run_sha256": "9" * 64,
        "prompt_source": "/sealed/prompts.json",
        "prompt_set_schema": "neural-forge-whole-response-review-prompts-v1",
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": 1,
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "parent_response_envelope": parent_envelope,
        "requested_candidate_ids": [candidate_id],
        "reviewed_candidate_count": 1,
        "candidate_reviews": [candidate],
        "evaluation_policy": {
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
            "parent_comparison_uses_cached_text_only": True,
        },
        "operation_audit": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_passes": 0,
        },
        "progress_json": "/sealed/dossier.progress.json",
    }
    document = {**body, "dossier_sha256": review._canonical_sha256(body)}
    path.write_text(json.dumps(document), encoding="utf-8")
    return review._file_sha256(path), document["dossier_sha256"]


def _prepare_inputs(
    tmp_path: Path,
    *,
    count: int = 3,
) -> tuple[list[Path], list[str], list[str]]:
    paths = []
    file_hashes = []
    dossier_hashes = []
    for index in range(count):
        path = tmp_path / f"dossier-{index}.json"
        file_sha, dossier_sha = _write_dossier(
            path,
            candidate_id="same-terminal-candidate",
            child_response=f"Child lane {index} gives a coherent explanation.",
        )
        paths.append(path)
        file_hashes.append(file_sha)
        dossier_hashes.append(dossier_sha)
    return paths, file_hashes, dossier_hashes


def test_prepare_authenticates_merges_renders_and_seals_template(
    tmp_path: Path,
) -> None:
    paths, file_hashes, dossier_hashes = _prepare_inputs(tmp_path, count=5)
    merged_path = tmp_path / "merged.json"
    markdown_path = tmp_path / "review.md"
    template_path = tmp_path / "verdict-template.json"

    result = review.prepare_qualitative_review(
        dossier_paths=paths,
        expected_dossier_file_sha256s=file_hashes,
        expected_dossier_sha256s=dossier_hashes,
        lane_labels=(
            "fit_guided",
            "full_strength_structural",
            "residual_writer",
            "source_producer_residual_writer",
            "structural_sequence_residual_mlp_writer",
        ),
        merged_json=merged_path,
        markdown_path=markdown_path,
        verdict_template_json=template_path,
    )

    merged = json.loads(merged_path.read_text(encoding="ascii"))
    merged_body = dict(merged)
    assert merged_body.pop("merged_sha256") == review._canonical_sha256(merged_body)
    assert merged["source_dossier_count"] == 5
    assert merged["candidate_count"] == 5
    assert [row["file_sha256"] for row in merged["source_dossiers"]] == file_hashes
    assert [row["dossier_sha256"] for row in merged["source_dossiers"]] == (
        dossier_hashes
    )
    assert len({row["candidate_key"] for row in merged["candidates"]}) == 5
    assert (
        merged["evaluation_policy"]["numeric_diagnostics_have_decision_authority"]
        is False
    )
    assert (
        merged["evaluation_policy"][
            "automatic_acceptance_rejection_promotion_or_retention_computed"
        ]
        is False
    )
    assert (
        merged["evaluation_policy"][
            "single_binary_manual_verdict_has_decision_authority"
        ]
        is False
    )
    assert merged["evaluation_policy"]["candidate_winner_selection_requested"] is False

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "<table>" in markdown
    assert "Cached parent response" in markdown
    assert "Folded child response" in markdown
    assert "Child lane 2 gives a coherent explanation." in markdown
    assert "Navigation diagnostics only" in markdown
    assert "never accept, reject, promote, rank, or estimate" in markdown
    assert "Record observations across dimensions" in markdown

    template = json.loads(template_path.read_text(encoding="ascii"))
    template_body = dict(template)
    assert template_body.pop("template_sha256") == review._canonical_sha256(
        template_body
    )
    assert template["source_merged_evidence_sha256"] == merged["merged_sha256"]
    assert template["observation_dimensions"] == list(review.OBSERVATION_DIMENSIONS)
    assert template["failure_mode_categories"] == list(review.FAILURE_MODE_CATEGORIES)
    assert len(template["candidate_reviews"]) == 5
    for candidate in template["candidate_reviews"]:
        assert candidate["observed_strengths"] == []
        assert candidate["observed_limitations_and_regressions"] == []
        assert candidate["next_folding_hypotheses"] == []
        assert candidate["prompt_reviews"] == [
            {
                "prompt_id": "reasoning-one",
                "full_parent_and_child_responses_reviewed": False,
                "dimension_observations": {
                    dimension: "" for dimension in review.OBSERVATION_DIMENSIONS
                },
                "failure_modes": [],
                "additional_observations": [],
            }
        ]
    assert template["cross_candidate_observations"] == {
        "comparative_strengths": [],
        "comparative_regressions": [],
        "next_experiment_hypotheses": [],
    }
    assert (
        template["instructions"]["numeric_diagnostics_have_decision_authority"] is False
    )
    assert template["instructions"]["single_binary_manual_verdict_requested"] is False
    assert (
        template["completion_requirements"][
            "worksheet_completion_required_for_continuation"
        ]
        is False
    )
    assert result["automatic_decision_performed"] is False
    assert result["verdict_template_sha256"] == template["template_sha256"]
    assert result["observation_template_sha256"] == template["template_sha256"]


def test_prepare_rejects_changed_dossier_bytes(tmp_path: Path) -> None:
    paths, file_hashes, dossier_hashes = _prepare_inputs(tmp_path)
    paths[1].write_text(paths[1].read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(
        review.Lfm2SemanticQualitativeReviewError,
        match="bytes changed",
    ):
        review.prepare_qualitative_review(
            dossier_paths=paths,
            expected_dossier_file_sha256s=file_hashes,
            expected_dossier_sha256s=dossier_hashes,
            lane_labels=("lane-a", "lane-b", "lane-c"),
            merged_json=tmp_path / "merged.json",
            markdown_path=tmp_path / "review.md",
            verdict_template_json=tmp_path / "template.json",
        )


def test_prepare_rejects_resealed_but_inconsistent_parent_evidence(
    tmp_path: Path,
) -> None:
    paths, file_hashes, dossier_hashes = _prepare_inputs(tmp_path)
    file_hashes[2], dossier_hashes[2] = _write_dossier(
        paths[2],
        candidate_id="same-terminal-candidate",
        child_response="A coherent child response.",
        parent_response="A different cached parent response.",
    )

    with pytest.raises(
        review.Lfm2SemanticQualitativeReviewError,
        match="exact prompt set, cached parent evidence",
    ):
        review.prepare_qualitative_review(
            dossier_paths=paths,
            expected_dossier_file_sha256s=file_hashes,
            expected_dossier_sha256s=dossier_hashes,
            lane_labels=("lane-a", "lane-b", "lane-c"),
            merged_json=tmp_path / "merged.json",
            markdown_path=tmp_path / "review.md",
            verdict_template_json=tmp_path / "template.json",
        )


def test_prepare_requires_unique_lane_bindings(tmp_path: Path) -> None:
    paths, file_hashes, dossier_hashes = _prepare_inputs(tmp_path)
    with pytest.raises(
        review.Lfm2SemanticQualitativeReviewError,
        match="Lane labels must be unique",
    ):
        review.prepare_qualitative_review(
            dossier_paths=paths,
            expected_dossier_file_sha256s=file_hashes,
            expected_dossier_sha256s=dossier_hashes,
            lane_labels=("duplicate", "duplicate", "third"),
            merged_json=tmp_path / "merged.json",
            markdown_path=tmp_path / "review.md",
            verdict_template_json=tmp_path / "template.json",
        )


def test_cli_writes_all_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths, file_hashes, dossier_hashes = _prepare_inputs(tmp_path, count=5)
    merged = tmp_path / "cli-merged.json"
    markdown = tmp_path / "cli-review.md"
    template = tmp_path / "cli-template.json"
    argv = []
    for path in paths:
        argv.extend(("--dossier", str(path)))
    for value in file_hashes:
        argv.extend(("--expected-dossier-file-sha256", value))
    for value in dossier_hashes:
        argv.extend(("--expected-dossier-sha256", value))
    for value in (
        "fit",
        "structural",
        "residual",
        "producer-writer",
        "structural-sequence-writer",
    ):
        argv.extend(("--lane-label", value))
    argv.extend(
        (
            "--merged-json",
            str(merged),
            "--markdown",
            str(markdown),
            "--verdict-template-json",
            str(template),
        )
    )

    assert review.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["candidate_count"] == 5
    assert summary["automatic_decision_performed"] is False
    assert merged.is_file() and markdown.is_file() and template.is_file()
