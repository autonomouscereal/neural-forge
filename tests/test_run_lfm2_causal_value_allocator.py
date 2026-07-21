from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import finalize_lfm2_causal_value_allocator as finalizer
from experiments import run_lfm2_causal_value_allocator as runner


def _intelligence(*, failed: str | None = None) -> dict:
    required = [
        "task_correctness",
        "parent_capability",
        "nonempty_retention",
    ]
    return {
        "policy": {"required_components": required},
        "core_intelligence_gate": {
            "manual_review": {
                "status": "rejected",
                "passed": False,
                "case_pass_rate": 0.0,
            },
            "components": [
                {
                    "name": name,
                    "retention": 0.95,
                    "passed": name != failed,
                }
                for name in required
            ],
        },
        "parent_relative_task_retention": {
            "absolute_accuracy_retention": 0.93,
        },
        "inputs": {
            "manifest_sha256": "1" * 64,
            "manifest_document_sha256": "2" * 64,
            "case_count": 64,
        },
    }


def _sealed_intelligence() -> dict:
    body = {
        "schema": runner.holdout_eval.RUNNER_SCHEMA,
        **_intelligence(),
    }
    return {**body, "report_sha256": runner.canonical_sha256(body)}


def _packet(*, case_count: int = 64, exact_scope: bool = True) -> dict:
    cases = [
        {
            "case_id": f"case_{index:02d}",
            "prompt": f"prompt {index}",
            "parent_response": f"parent {index}",
            "candidate_response": f"child {index}",
            "generation_limit": 128,
            "parent_completion": {
                "generated_token_count": 8 + index,
                "eos_observed": True,
                "unfinished_at_generation_limit": False,
            },
            "candidate_completion": {
                "generated_token_count": 7 + index,
                "eos_observed": True,
                "unfinished_at_generation_limit": False,
            },
        }
        for index in range(case_count)
    ]
    case_ids = [value["case_id"] for value in cases]
    limit_rows = [
        {
            "case_id": value["case_id"],
            "generation_limit": value["generation_limit"],
        }
        for value in cases
    ]
    completion_rows = [
        {
            "case_id": value["case_id"],
            "parent_completion": value["parent_completion"],
            "candidate_completion": value["candidate_completion"],
        }
        for value in cases
    ]
    body = {
        "schema": runner.PARENT_RELATIVE_PACKET_SCHEMA,
        "review_scope": (
            "exact_authenticated_manifest_all_cases"
            if exact_scope
            else "legacy_generic_manual_subset"
        ),
        "all_case_manual_review_mode": exact_scope,
        "manifest_sha256": "1" * 64,
        "manifest_document_sha256": "2" * 64,
        "case_count": case_count,
        "case_ids_sha256": runner.canonical_sha256(case_ids),
        "generation_limit_rows_sha256": runner.canonical_sha256(limit_rows),
        "completion_metadata_rows_sha256": runner.canonical_sha256(completion_rows),
        "legacy_generic_manual_subset_is_diagnostic_only": exact_scope,
        "rubric": {
            "review_every_manifest_case": exact_scope,
            "full_parent_and_candidate_response_review_required": exact_scope,
            "per_case_generation_limit_bound": exact_scope,
            "qualitative_criteria": list(runner.QUALITATIVE_CRITERIA),
            "parent_and_candidate_completion_metadata_bound": exact_scope,
            "completion_metadata_is_diagnostic_context_only": exact_scope,
            "completion_metadata_automatic_accept_reject_permitted": False,
            "explicit_truncation_consideration_required": exact_scope,
            "criterion_specific_observations_required": exact_scope,
            "duplicate_boilerplate_observations_permitted": False,
            "qualitative_case_verdicts_are_authoritative": exact_scope,
            "holistic_suite_conclusion_is_authoritative": exact_scope,
            "automatic_accept_reject_permitted": False,
        },
        "cases": cases,
    }
    return {**body, "packet_sha256": runner.canonical_sha256(body)}


def _manual(conclusion: str, packet: dict) -> dict:
    case_ids = [value["case_id"] for value in packet["cases"]]
    return {
        "packet_sha256": packet["packet_sha256"],
        "retention": 0.0,
        "case_reports": [{"case_id": value} for value in case_ids],
        "holistic_review": {
            "status": "complete",
            "qualitative_judgment_is_authoritative": True,
            "numeric_scores_and_thresholds_are_diagnostic_only": True,
            "automatic_scalar_accept_reject_permitted": False,
            "qualitative_case_reviews": [
                {
                    "case_id": value,
                    "verdict": "different_but_equivalent",
                    "failure_modes": [],
                    "reason": (
                        f"The complete response pair for {value} received a direct "
                        "semantic review."
                    ),
                    "full_parent_and_candidate_responses_reviewed": True,
                    "criterion_observations": {
                        criterion: (
                            f"The {criterion} criterion for {value} received a "
                            "case-specific whole-response observation."
                        )
                        for criterion in runner.QUALITATIVE_CRITERIA
                    },
                    "truncation_consideration": {
                        "parent_unfinished_at_generation_limit": False,
                        "candidate_unfinished_at_generation_limit": False,
                        "assessment": "none",
                        "observation": (
                            f"The completion metadata for {value} was considered "
                            "before the qualitative judgment."
                        ),
                    },
                }
                for value in case_ids
            ],
            "suite_conclusion": {
                "conclusion": conclusion,
            },
        },
    }


def test_decision_uses_parent_relative_manual_not_absolute_manual_failure() -> None:
    packet = _packet()
    manual = _manual("credible_intelligent_compaction", packet)
    result = runner._parent_relative_decision(_intelligence(), manual, packet)
    assert result["promoted"] is True
    assert result["status"] == "promoted"
    assert result["certified_parent_relative_lower_bound"] is None
    assert result["observed_parent_relative_diagnostic_floor"] == pytest.approx(0.0)
    assert result["absolute_manual_scores_are_diagnostics_only"] is True
    assert result["abilities_absent_in_parent_are_neutralized"] is True
    assert result["qualitative_judgment_is_authoritative"] is True


def test_decision_withholds_pending_and_uses_only_holistic_conclusion() -> None:
    packet = _packet()
    pending = runner._parent_relative_decision(_intelligence(), None, packet)
    assert pending["promoted"] is False
    assert pending["status"] == "pending_parent_relative_manual"
    not_promoted = runner._parent_relative_decision(
        _intelligence(),
        _manual("not_yet_coherent_or_intelligent", packet),
        packet,
    )
    assert not_promoted["promoted"] is False
    assert not_promoted["status"] == "holistic_review_complete_not_promoted"
    assert (
        "authoritative_holistic_review:not_yet_coherent_or_intelligent"
        in (not_promoted["reasons"])
    )
    automatic = runner._parent_relative_decision(
        _intelligence(failed="parent_capability"),
        _manual("credible_intelligent_compaction", packet),
        packet,
    )
    assert automatic["promoted"] is True
    assert automatic["automatic_components_passed"] is False
    assert automatic["automatic_component_pass_flags_are_diagnostic_only"] is True


def test_legacy_six_case_review_cannot_authorize_promotion() -> None:
    packet = _packet(case_count=6, exact_scope=False)
    result = runner._parent_relative_decision(
        _intelligence(),
        _manual("credible_intelligent_compaction", packet),
        packet,
    )
    assert result["promoted"] is False
    assert result["status"] == "promotion_ineligible_incomplete_review_scope"
    assert result["exact_64_case_manual_review_contract_authenticated"] is False
    assert (
        "authoritative_holistic_review:exact_64_case_packet_required"
        in result["reasons"]
    )


def test_parent_relative_packet_builder_covers_exact_manifest_all_64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = tuple(
        SimpleNamespace(
            case_id=f"case_{index:02d}",
            prompt=f"prompt {index}",
            generation_limit=96 + index,
        )
        for index in range(64)
    )
    parents = tuple(
        SimpleNamespace(
            case_id=spec.case_id,
            prompt=spec.prompt,
            parent_output=f"parent {index}",
            generation=SimpleNamespace(
                generated_token_count=12,
                eos_observed=True,
                unfinished_at_generation_limit=False,
            ),
        )
        for index, spec in enumerate(specs)
    )
    envelope = SimpleNamespace(
        eos_token_ids=(0,),
        manifest=SimpleNamespace(
            cases=specs,
            manifest_sha256="1" * 64,
            document_sha256="2" * 64,
        ),
    )
    candidates = tuple(
        {
            "case_id": spec.case_id,
            "output": f"child {index}",
            "output_token_ids": [index + 1, 0],
            "generation_limit": spec.generation_limit,
        }
        for index, spec in enumerate(specs)
    )
    captured = {}
    monkeypatch.setattr(
        runner,
        "baseline_from_lfm2_intelligence_holdout_envelope",
        lambda _envelope: SimpleNamespace(cases=parents),
    )

    def _capture(_standard_packet, rows, **kwargs):
        captured["rows"] = list(rows)
        captured["kwargs"] = dict(kwargs)
        return {"case_count": len(rows)}

    monkeypatch.setattr(
        runner,
        "build_all_case_parent_relative_manual_packet",
        _capture,
    )
    result = runner._build_exact_all_case_parent_relative_packet(
        {"legacy": "packet"},
        envelope,
        candidates,
    )
    assert result["case_count"] == 64
    assert [value["case_id"] for value in captured["rows"]] == [
        spec.case_id for spec in specs
    ]
    assert [value["generation_limit"] for value in captured["rows"]] == [
        spec.generation_limit for spec in specs
    ]
    assert all(
        value["parent_completion"]["eos_observed"] is True
        and value["candidate_completion"]["eos_observed"] is True
        and value["candidate_completion"]["unfinished_at_generation_limit"] is False
        for value in captured["rows"]
    )
    assert captured["kwargs"] == {
        "manifest_sha256": "1" * 64,
        "manifest_document_sha256": "2" * 64,
    }


def test_exact_review_packet_is_required_and_report_schemas_are_versioned() -> None:
    parameter = inspect.signature(runner._parent_relative_decision).parameters[
        "manual_packet"
    ]
    assert parameter.default is inspect.Parameter.empty
    assert runner.LEGACY_TARGET_REPORT_SCHEMA.endswith("-v1")
    assert runner.TARGET_REPORT_SCHEMA.endswith("-v2")
    assert finalizer.LEGACY_FINAL_SCHEMA.endswith("-v1")
    assert finalizer.FINAL_SCHEMA.endswith("-v2")


def test_cached_intelligence_report_content_seal_is_recomputed() -> None:
    intelligence = _sealed_intelligence()
    assert (
        runner._authenticate_intelligence_report(intelligence)
        == (intelligence["report_sha256"])
    )
    tampered = {
        **intelligence,
        "inputs": {
            **intelligence["inputs"],
            "manifest_sha256": "f" * 64,
        },
    }
    with pytest.raises(
        runner.Lfm2CausalValueRunnerError,
        match="schema or content seal is invalid",
    ):
        runner._authenticate_intelligence_report(tampered)
    assert "_authenticate_intelligence_report" in inspect.getsource(
        finalizer.finalize_lfm2_causal_value_allocator
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "experiments/run_lfm2_causal_value_allocator.py",
        "experiments/finalize_lfm2_causal_value_allocator.py",
        "experiments/run_lfm2_protected_weight_error_allocator.py",
        "experiments/finalize_lfm2_protected_weight_error_allocator.py",
        "experiments/run_lfm2_exact50_all8.py",
        "experiments/finalize_lfm2_exact50_all8.py",
        "experiments/reevaluate_lfm2_packed_checkpoint.py",
    ],
)
def test_every_shared_decision_caller_passes_review_packet(
    relative_path: str,
) -> None:
    source = (Path(__file__).parents[1] / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "_parent_relative_decision"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "_parent_relative_decision"
        )
    ]
    assert calls
    assert all(
        len(node.args) >= 3
        or any(keyword.arg == "manual_packet" for keyword in node.keywords)
        for node in calls
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "experiments/run_lfm2_causal_value_allocator.py",
        "experiments/run_lfm2_protected_weight_error_allocator.py",
        "experiments/run_lfm2_exact50_all8.py",
        "experiments/reevaluate_lfm2_packed_checkpoint.py",
    ],
)
def test_all_generating_callers_build_exact_all_case_review_packet(
    relative_path: str,
) -> None:
    source = (Path(__file__).parents[1] / relative_path).read_text(encoding="utf-8")
    assert "_build_exact_all_case_parent_relative_packet(" in source
    assert "build_parent_relative_manual_packet(" not in source


def test_config_requires_sealed_local_frontier_and_paired_optional_inputs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    frontier = tmp_path / "frontier.json"
    envelope = tmp_path / "envelope.json"
    frontier.write_text("{}", encoding="utf-8")
    envelope.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    with pytest.raises(
        runner.Lfm2CausalValueRunnerError,
        match="must be supplied together",
    ):
        runner.CausalValueRunConfig(
            parent_model_directory=parent,
            tokenizer_directory=tokenizer,
            output_root=tmp_path / "output",
            cached_causal_frontier=frontier,
            expected_cached_causal_frontier_file_sha256=digest,
            parent_envelope=envelope,
            expected_parent_envelope_file_sha256=digest,
            expected_parent_envelope_sha256=digest,
            expected_manifest_sha256=digest,
            expected_parent_state_fingerprint=digest,
            expected_parent_model_fingerprint=digest,
            expected_tokenizer_fingerprint=digest,
            calibration_token_selection=frontier,
        )


@pytest.mark.parametrize("module", [runner, finalizer])
def test_runners_have_no_hub_loader_optimizer_backward_or_training_call(module) -> None:
    tree = ast.parse(inspect.getsource(module))
    forbidden = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"from_pretrained", "backward", "step"}
    }
    assert forbidden == set()
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("optim" in value.lower() for value in imports)
