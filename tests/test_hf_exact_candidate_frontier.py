from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from neural_forge import hf_exact_candidate_frontier as frontier


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "retain_qwen_exact_target_candidates.py"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frontier.file_sha256(path)


def _metrics(
    *,
    exact: int = 0,
    lcp4: int = 0,
    lcp: int = 1,
    tokens: int = 2,
    actions: int = 3,
    repeat: float = 0.25,
    adjacent: float = 0.25,
) -> dict:
    lcp = max(lcp, exact * 8, lcp4 * 4)
    tokens = max(tokens, lcp, exact * 8)
    actions = max(actions, exact * 8)
    rollout_count = 24
    rollout_tokens = 192
    action_count = 192
    return {
        "split": "calibration-selection",
        "action_count": action_count,
        "action_agreement_count": actions,
        "action_accuracy": actions / action_count,
        "parent_action_mean_logprob": -4.0,
        "parent_action_topk_fraction": 0.25,
        "parent_action_mean_rank": 10.0,
        "parent_action_worst_rank": 20,
        "rollout_count": rollout_count,
        "exact_rollout_count": exact,
        "exact_rollout_fraction": exact / rollout_count,
        "rollout_token_count": rollout_tokens,
        "token_match_count": tokens,
        "token_match_fraction": tokens / rollout_tokens,
        "lcp_token_count": lcp,
        "lcp_fraction": lcp / rollout_tokens,
        "lcp_at_least_4_count": lcp4,
        "repeat_fraction_excess": repeat,
        "adjacent_repeat_fraction_excess": adjacent,
        "longest_run_excess": 1,
        "report_sha256": _sha(
            f"metrics:{exact}:{lcp4}:{lcp}:{tokens}:{actions}:{repeat}:{adjacent}"
        ),
    }


def _artifact_auth(name: str) -> dict:
    return {
        "path": f"/not-used/{name}",
        "required_files": [
            {
                "path": "model.safetensors",
                "sha256": _sha(f"{name}:model"),
                "size_bytes": 7,
            }
        ],
        "tree": None,
        "authenticated_before_model_access": True,
    }


@dataclass(frozen=True)
class _RunCase:
    run: Path
    run_report_sha256: str
    preregistration_sha256: str
    factor_authenticator: frontier.FactorAuthenticator
    preregistration_authenticator: frontier.PreregistrationAuthenticator
    accepted_child_authenticator: frontier.AcceptedChildAuthenticator

    def candidate(
        self,
        *,
        expected_run_report_sha256: str | None = None,
        expected_preregistration_sha256: str | None = None,
        include_factor_authenticator: bool = True,
        include_child_authenticator: bool = True,
    ) -> frontier.TerminalCandidate:
        return frontier.authenticate_terminal_candidate(
            self.run,
            expected_run_report_sha256=(
                self.run_report_sha256
                if expected_run_report_sha256 is None
                else expected_run_report_sha256
            ),
            expected_preregistration_sha256=(
                self.preregistration_sha256
                if expected_preregistration_sha256 is None
                else expected_preregistration_sha256
            ),
            preregistration_authenticator=self.preregistration_authenticator,
            factor_authenticator=(
                self.factor_authenticator if include_factor_authenticator else None
            ),
            accepted_child_authenticator=(
                self.accepted_child_authenticator
                if include_child_authenticator
                else None
            ),
        )


def _make_run(
    tmp_path: Path,
    name: str,
    *,
    metrics: dict | None = None,
    step0_metrics: dict | None = None,
    status: str = "rejected",
    comparison: str | None = None,
    accepted_factor_fallback: bool = False,
) -> _RunCase:
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    blank = tmp_path / "canonical_blank"
    cache = tmp_path / "compiler_cache"
    blank.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    blank_file = blank / "model.safetensors"
    cache_file = cache / "receipt.json"
    if not blank_file.exists():
        blank_file.write_bytes(b"one immutable blank")
    if not cache_file.exists():
        cache_file.write_bytes(b"one immutable compiler cache")
    comparison_sha = _sha("comparison") if comparison is None else comparison
    runner_sha = _sha("canonical runner source")
    trainer_sha = _sha("canonical trainer source")
    preregistration = {
        "schema": frontier.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "parent": {
            "path": str(tmp_path / "parent"),
            "exact_unique_parameters": 4_000,
        },
        "blank_target": {
            "path": str(blank),
            "exact_unique_parameters": 1_400,
        },
        "implementation": {
            "runner_sha256": runner_sha,
            "trainer_module": "neural_forge.test_output_trainer",
            "trainer_sha256": trainer_sha,
        },
        "training": {"evaluation_horizon": 8},
    }
    prereg_path = run / "preregistration.json"
    prereg_sha = _write_json(prereg_path, preregistration)
    selected_metrics = (
        _metrics(lcp=2, tokens=3, actions=4) if metrics is None else metrics
    )
    baseline_metrics = (
        _metrics(lcp=0, tokens=0, actions=0) if step0_metrics is None else step0_metrics
    )
    step0_sha = _sha(f"{name}:step0")
    selected_sha = _sha(f"{name}:selected")
    evidence = {
        "selected_step": 8,
        "checkpoints": [
            {
                "step": 0,
                "factor_state_sha256": step0_sha,
                "metrics": baseline_metrics,
            },
            {
                "step": 8,
                "factor_state_sha256": selected_sha,
                "metrics": selected_metrics,
            },
        ],
        "candidate_metrics": selected_metrics,
        "factor_pack_manifest": {
            "schema": "test-factor-schema",
            "factor_state_sha256": selected_sha,
        },
        "selection_policy": "rollout-first-acquisition",
        "internal_numeric_matching_used": False,
        "selection_uses_internal_numeric_targets": False,
        "candidate_gate_passed": status == "accepted",
        "sealed_evaluation_unlocked": status == "accepted",
    }
    decisions = {
        "fit": {"passed": True, "metrics": selected_metrics},
        "candidate": {
            "passed": status == "accepted",
            "metrics": selected_metrics,
        },
        "sealed": (
            {"passed": True, "metrics": selected_metrics}
            if status == "accepted"
            else None
        ),
    }
    report = {
        "schema": frontier.RUN_SCHEMA,
        "status": status,
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": comparison_sha,
        "artifact_authentication": {
            "parent": _artifact_auth("parent"),
            "blank_target": _artifact_auth("blank"),
        },
        "implementation_authentication": {
            "runner": {
                "path": str(tmp_path / "runner.py"),
                "sha256": runner_sha,
            },
            "trainer": {
                "module": "neural_forge.test_output_trainer",
                "path": str(tmp_path / "trainer.py"),
                "sha256": trainer_sha,
            },
            "authenticated_before_model_access": True,
        },
        "splits": {
            role: {
                "path": str(run / f"{role}.json"),
                "sha256": _sha(f"{name}:{role}"),
                "row_count": count,
            }
            for role, count in (("fit", 120), ("candidate", 24), ("sealed", 32))
        },
        "compiler_split": {
            "path": str(run / "compiler.json"),
            "sha256": _sha(f"{name}:compiler"),
            "row_count": 120,
            "reuses_behavior_fit": False,
        },
        "thresholds": {
            role: {"minimum_action_accuracy": 0.5}
            for role in ("fit", "candidate", "sealed")
        },
        "compiler_cache": {"root": str(cache)},
        "trainer_evidence": evidence,
        "trainer_evidence_sha256": frontier.canonical_sha256(evidence),
        "factor_acquisition": {
            "selected_step": 8,
            "step0_factor_state_sha256": step0_sha,
            "selected_factor_state_sha256": selected_sha,
            "target_native_factors_acquired_behavior": True,
        },
        "behavior_decisions": decisions,
        "pre_bake_behavior_decisions": decisions,
    }
    if status == "rejected":
        factor_path = run / "rejected_factor_pack.pt"
        factor_path.write_bytes(f"factor:{selected_sha}".encode())
        report.update(
            {
                "reason": "candidate-gate-rejected",
                "standalone_compaction_passed": False,
                "model_checkpoint_written": False,
                "rollback_exact": True,
                "factor_pack": {
                    "path": str(factor_path),
                    "sha256": frontier.file_sha256(factor_path),
                },
            }
        )
    else:
        child = run / "accepted_child"
        child.mkdir()
        (child / "model.safetensors").write_bytes(b"tiny accepted child")
        (child / "config.json").write_text("{}\n", encoding="utf-8")
        tree = frontier.artifact_tree(child)
        accepted_decisions = {
            role: {"passed": True, "metrics": selected_metrics}
            for role in ("fit", "candidate", "sealed")
        }
        report.update(
            {
                "standalone_compaction_passed": True,
                "model_checkpoint_written": True,
                "rollback_exact": None,
                "pre_bake_behavior_decisions": accepted_decisions,
                "behavior_decisions": accepted_decisions,
                "post_bake_behavior_decisions": accepted_decisions,
                "fresh_reload_certificate": {
                    "fresh_reload_performed": True,
                    "exact_unique_parameters": 1_400,
                    "exact_budget_match": True,
                    "state_keys_match_pre_save": True,
                    "state_bytes_match_pre_save": True,
                    "state_dict_sha256": _sha(f"{name}:child-state"),
                    "published_adapter_parameter_count": 0,
                    "published_parent_runtime_reference_count": 0,
                    "published_source_sized_sidecar_parameter_count": 0,
                    "artifact_tree": {
                        key: tree[key]
                        for key in ("file_count", "total_bytes", "tree_sha256")
                    },
                    "path": str(child),
                },
            }
        )
        if accepted_factor_fallback:
            factor_path = run / "selected_factor_pack.pt"
            factor_path.write_bytes(f"factor:{selected_sha}".encode())
            report["selected_factor_pack"] = {
                "path": str(factor_path),
                "sha256": frontier.file_sha256(factor_path),
            }
    _write_json(run / "run_report.json", report)
    report_sha = frontier.file_sha256(run / "run_report.json")

    def authenticate(path: Path, lexical_rank: int | None) -> dict:
        expected_path = run / (
            "selected_factor_pack.pt"
            if status == "accepted" and accepted_factor_fallback
            else "rejected_factor_pack.pt"
        )
        assert path == expected_path
        assert lexical_rank is None
        return {
            "factor_state_sha256": selected_sha,
            "envelope_provenance": {
                "reason": (
                    "accepted-selected-factor"
                    if status == "accepted"
                    else "candidate-gate-rejected"
                ),
                "preregistration_sha256": prereg_sha,
                "evaluation_contract_sha256": comparison_sha,
            },
        }

    preflight = {
        "status": "preflight-passed",
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": comparison_sha,
        "artifact_authentication": report["artifact_authentication"],
        "implementation_authentication": report["implementation_authentication"],
        "splits": report["splits"],
        "compiler_split": report["compiler_split"],
        "thresholds": report["thresholds"],
    }

    def authenticate_preregistration(path: Path, digest: str) -> dict:
        assert path == prereg_path
        assert digest == prereg_sha
        return preflight

    def authenticate_child(path: Path, expected_parameters: int) -> dict:
        assert path == run / "accepted_child"
        assert expected_parameters == 1_400
        tree = frontier.artifact_tree(path)
        certificate = report["fresh_reload_certificate"]
        return {
            "fresh_reload_performed": True,
            "loaded_from_candidate_path": True,
            "exact_unique_parameters": 1_400,
            "state_dict_sha256": certificate["state_dict_sha256"],
            "artifact_tree": {
                key: tree[key] for key in ("file_count", "total_bytes", "tree_sha256")
            },
            "published_adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
        }

    return _RunCase(
        run=run,
        run_report_sha256=report_sha,
        preregistration_sha256=prereg_sha,
        factor_authenticator=authenticate,
        preregistration_authenticator=authenticate_preregistration,
        accepted_child_authenticator=authenticate_child,
    )


def test_authentication_truthfully_distinguishes_factor_and_accepted_child(
    tmp_path: Path,
) -> None:
    rejected = _make_run(tmp_path, "rejected")
    research = rejected.candidate()
    assert research.truth_label == "research-factor-only-candidate"
    assert research.fresh_reload_verified is False
    assert research.standalone_compaction_passed is False
    assert research.deployment_accepted is False

    accepted = _make_run(
        tmp_path,
        "accepted",
        status="accepted",
        metrics=_metrics(exact=1, lcp4=1, lcp=8, tokens=8, actions=8),
    )
    standalone = accepted.candidate()
    assert standalone.truth_label == "accepted-standalone"
    assert standalone.fresh_reload_verified is True
    assert standalone.standalone_compaction_passed is True
    assert standalone.deployment_accepted is True
    assert standalone.source_artifact_tree is not None


def test_factor_and_child_tampering_fail_before_frontier_mutation(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "factor-tamper")
    (case.run / "rejected_factor_pack.pt").write_bytes(b"changed")
    with pytest.raises(frontier.ExactCandidateFrontierError, match="file hash"):
        case.candidate()

    child_case = _make_run(tmp_path, "child-tamper", status="accepted")
    (child_case.run / "accepted_child" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="failed independent authentication",
    ):
        child_case.candidate()


def test_constructor_and_inspection_are_zero_write(tmp_path: Path) -> None:
    root = tmp_path / "not-created-by-read"
    manager = frontier.ExactCandidateFrontier(root)
    assert not root.exists()
    assert manager.inspect()["retained_record_count"] == 0
    assert not root.exists()


def test_preexisting_unowned_deterministic_dirs_and_extras_are_never_adopted(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "orphan")
    candidate = case.candidate()
    root = tmp_path / "orphan-frontier"
    artifact = (
        root / "artifacts" / f"factor-file-{candidate.source_artifact_identity_sha256}"
    )
    artifact.mkdir(parents=True)
    (root / "evidence").mkdir()
    (artifact / "factor_pack.pt").write_bytes(candidate.source_artifact.read_bytes())
    manager = frontier.ExactCandidateFrontier(root)
    with pytest.raises(frontier.ExactCandidateFrontierError, match="manifestless"):
        manager.register(candidate)
    assert not (root / "frontier_manifest.json").exists()
    assert candidate.source_artifact.is_file()

    extra_root = tmp_path / "extra-frontier"
    extra_root.mkdir()
    (extra_root / "unexpected.txt").write_text("unowned", encoding="utf-8")
    with pytest.raises(frontier.ExactCandidateFrontierError, match="unowned"):
        frontier.ExactCandidateFrontier(extra_root).inspect()


def test_source_run_and_artifacts_are_protected_before_frontier_creation(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "source-overlap")
    nested_frontier = case.run / "candidate-frontier"
    manager = frontier.ExactCandidateFrontier(nested_frontier)
    assert not nested_frontier.exists()
    with pytest.raises(frontier.ExactCandidateFrontierError, match="overlaps"):
        manager.register(case.candidate())
    assert not nested_frontier.exists()
    assert case.run.is_dir()
    assert (case.run / "rejected_factor_pack.pt").is_file()


def test_external_hash_roots_and_canonical_source_pins_fail_before_any_write(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "external-roots")
    frontier_root = tmp_path / "frontier"
    frontier.ExactCandidateFrontier(frontier_root)
    with pytest.raises(frontier.ExactCandidateFrontierError, match="external root"):
        case.candidate(expected_run_report_sha256="0" * 64)
    assert not frontier_root.exists()
    with pytest.raises(
        frontier.ExactCandidateFrontierError, match="external and report"
    ):
        case.candidate(expected_preregistration_sha256="1" * 64)
    assert not frontier_root.exists()

    def forged_preregistration(_path: Path, _digest: str) -> dict:
        value = case.preregistration_authenticator(
            case.run / "preregistration.json", case.preregistration_sha256
        )
        forged = json.loads(json.dumps(value))
        forged["implementation_authentication"]["runner"]["sha256"] = "2" * 64
        return forged

    with pytest.raises(
        frontier.ExactCandidateFrontierError, match="preflight disagrees"
    ):
        frontier.authenticate_terminal_candidate(
            case.run,
            expected_run_report_sha256=case.run_report_sha256,
            expected_preregistration_sha256=case.preregistration_sha256,
            preregistration_authenticator=forged_preregistration,
            factor_authenticator=case.factor_authenticator,
        )
    assert not frontier_root.exists()


def test_candidate_must_strictly_improve_authenticated_step_zero(
    tmp_path: Path,
) -> None:
    tied = _metrics(lcp=2, tokens=3, actions=4)
    case = _make_run(
        tmp_path,
        "step-zero-tie",
        metrics=tied,
        step0_metrics=tied,
    )
    with pytest.raises(frontier.ExactCandidateFrontierError, match="strictly improve"):
        case.candidate()


def test_accepted_child_needs_independent_reload_or_factor_only_fallback(
    tmp_path: Path,
) -> None:
    no_fallback = _make_run(tmp_path, "accepted-no-fallback", status="accepted")
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="no authenticated factor-only fallback",
    ):
        no_fallback.candidate(include_child_authenticator=False)

    fallback = _make_run(
        tmp_path,
        "accepted-factor-fallback",
        status="accepted",
        accepted_factor_fallback=True,
    )
    retained = fallback.candidate(include_child_authenticator=False)
    assert retained.truth_label == "research-factor-only-candidate"
    assert retained.terminal_status == "accepted"
    assert retained.fresh_reload_verified is False

    def forged_child(_path: Path, _count: int) -> dict:
        return {"fresh_reload_performed": True, "loaded_from_candidate_path": True}

    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="no authenticated factor-only fallback",
    ):
        frontier.authenticate_terminal_candidate(
            no_fallback.run,
            expected_run_report_sha256=no_fallback.run_report_sha256,
            expected_preregistration_sha256=no_fallback.preregistration_sha256,
            preregistration_authenticator=no_fallback.preregistration_authenticator,
            factor_authenticator=no_fallback.factor_authenticator,
            accepted_child_authenticator=forged_child,
        )


def test_rollout_first_ranking_rejects_internal_numeric_metrics() -> None:
    better_rollout = _metrics(exact=1, lcp4=1, lcp=8, tokens=8, actions=8)
    better_actions = _metrics(exact=0, lcp=7, tokens=7, actions=8)
    assert frontier.rollout_first_rank(better_rollout) < frontier.rollout_first_rank(
        better_actions
    )
    polluted = {**_metrics(), "weight_similarity": 1.0}
    with pytest.raises(frontier.ExactCandidateFrontierError, match="internal"):
        frontier.rollout_first_rank(polluted)
    inconsistent = {**_metrics(), "lcp_at_least_4_count": 5}
    with pytest.raises(frontier.ExactCandidateFrontierError, match="exceeds"):
        frontier.rollout_first_rank(inconsistent)

    impossible_exact = _metrics(exact=1, lcp4=1)
    impossible_exact.update(
        {
            "lcp_token_count": 0,
            "lcp_fraction": 0.0,
            "token_match_count": 0,
            "token_match_fraction": 0.0,
            "action_agreement_count": 0,
            "action_accuracy": 0.0,
        }
    )
    with pytest.raises(frontier.ExactCandidateFrontierError):
        frontier.rollout_first_rank(impossible_exact)


def test_candidate_counts_are_bound_to_authenticated_split_and_horizon(
    tmp_path: Path,
) -> None:
    malformed = _metrics(lcp=2, tokens=3, actions=4)
    malformed.update(
        {
            "rollout_count": 4,
            "rollout_token_count": 8,
            "action_count": 8,
            "exact_rollout_fraction": 0.0,
            "token_match_fraction": 3 / 8,
            "lcp_fraction": 2 / 8,
            "action_accuracy": 4 / 8,
            "report_sha256": _sha("malformed-denominators"),
        }
    )
    case = _make_run(tmp_path, "bad-denominators", metrics=malformed)
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="authenticated split and evaluation horizon",
    ):
        case.candidate()


def test_artifact_tree_rejects_nested_mount_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "child"
    nested = root / "mounted"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"outside mounted content")
    real_ismount = frontier.os.path.ismount
    monkeypatch.setattr(
        frontier.os.path,
        "ismount",
        lambda value: Path(value) == nested or real_ismount(value),
    )
    with pytest.raises(frontier.ExactCandidateFrontierError, match="nested mount"):
        frontier.artifact_tree(root)


def test_max_three_dry_run_then_explicit_prune_never_touches_sources_or_protected(
    tmp_path: Path,
) -> None:
    frontier_root = tmp_path / "candidate_frontier"
    runs: list[Path] = []
    blank = tmp_path / "canonical_blank" / "model.safetensors"
    cache = tmp_path / "compiler_cache" / "receipt.json"
    for index, lcp in enumerate((1, 2, 3, 4)):
        case = _make_run(
            tmp_path,
            f"rank-{index}",
            metrics=_metrics(lcp=lcp, tokens=max(lcp, 2), actions=3 + index),
        )
        runs.append(case.run)
        candidate = case.candidate()
        manager = frontier.ExactCandidateFrontier(frontier_root)
        manager.register(candidate)
    manager = frontier.ExactCandidateFrontier(frontier_root)
    inspection = manager.inspect()
    assert inspection["retained_record_count"] == 3
    assert inspection["pending_prune_count"] == 1
    assert all(run.is_dir() for run in runs)
    assert len(list((frontier_root / "artifacts").iterdir())) == 4
    blank_before = blank.read_bytes()
    cache_before = cache.read_bytes()

    with pytest.raises(frontier.ExactCandidateFrontierError, match="unsupported"):
        manager.apply_pending_prune()
    assert len(list((frontier_root / "artifacts").iterdir())) == 4
    assert all(run.is_dir() for run in runs)
    assert blank.read_bytes() == blank_before
    assert cache.read_bytes() == cache_before


def test_dominated_candidate_is_not_copied_and_contracts_are_bounded_separately(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root)
    best = _make_run(
        tmp_path,
        "best",
        metrics=_metrics(exact=2, lcp4=2, lcp=16, tokens=16, actions=16),
    )
    best_result = manager.register(best.candidate())
    assert best_result["candidate_retained"] is True
    artifacts_before = {path.name for path in (root / "artifacts").iterdir()}
    for index in range(2):
        case = _make_run(
            tmp_path,
            f"other-{index}",
            metrics=_metrics(lcp=2 + index),
        )
        manager.register(case.candidate())
    dominated = _make_run(
        tmp_path,
        "dominated",
        metrics=_metrics(lcp=0, tokens=1, actions=1, repeat=1.0, adjacent=1.0),
    )
    result = manager.register(dominated.candidate())
    assert result["candidate_retained"] is False
    assert len(list((root / "artifacts").iterdir())) == len(artifacts_before) + 2

    different = _make_run(
        tmp_path,
        "different-contract",
        comparison=_sha("different comparison"),
    )
    manager.register(different.candidate())
    inspection = manager.inspect()
    assert sorted(inspection["retained_per_contract"].values()) == [1, 3]


def test_best_accepted_child_keeps_one_slot_even_when_three_research_keys_win(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root)
    accepted = _make_run(
        tmp_path,
        "accepted-fallback",
        status="accepted",
        metrics=_metrics(lcp=1, tokens=1, actions=1),
    )
    manager.register(accepted.candidate())
    for index in range(3):
        case = _make_run(
            tmp_path,
            f"better-research-{index}",
            metrics=_metrics(
                exact=1,
                lcp4=1,
                lcp=5 + index,
                tokens=6 + index,
                actions=5 + index,
            ),
        )
        manager.register(case.candidate())
    manifest = frontier.strict_json(root / "frontier_manifest.json")
    labels = [record["truth_label"] for record in manifest["records"]]
    assert len(labels) == 3
    assert labels.count("accepted-standalone") == 1


def test_manifest_publication_failure_deletes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root, max_per_contract=1)
    first = _make_run(tmp_path, "first", metrics=_metrics(lcp=1))
    manager.register(first.candidate())
    prior_artifacts = {path.name for path in (root / "artifacts").iterdir()}
    better = _make_run(tmp_path, "better", metrics=_metrics(lcp=4))
    candidate = better.candidate()

    def fail_manifest(_path: Path, _payload: dict) -> None:
        raise OSError("injected manifest failure")

    monkeypatch.setattr(frontier, "_atomic_replace_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        manager.register(candidate)
    assert prior_artifacts.issubset(
        {path.name for path in (root / "artifacts").iterdir()}
    )
    assert first.run.is_dir()
    assert better.run.is_dir()


def test_pending_prune_is_durable_and_actual_deletion_is_unsupported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root, max_per_contract=1)
    first = _make_run(tmp_path, "prune-old", metrics=_metrics(lcp=1))
    second = _make_run(tmp_path, "prune-new", metrics=_metrics(lcp=4))
    manager.register(first.candidate())
    manager.register(second.candidate())
    before = {path.name for path in (root / "artifacts").iterdir()}
    manifest_before = (root / "frontier_manifest.json").read_bytes()
    with pytest.raises(frontier.ExactCandidateFrontierError, match="unsupported"):
        manager.apply_pending_prune()
    assert {path.name for path in (root / "artifacts").iterdir()} == before
    assert (root / "frontier_manifest.json").read_bytes() == manifest_before
    assert first.run.is_dir() and second.run.is_dir()


def test_changed_pending_tree_and_path_escape_refuse_all_deletion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root, max_per_contract=1)
    first = _make_run(tmp_path, "old", metrics=_metrics(lcp=1))
    second = _make_run(tmp_path, "new", metrics=_metrics(lcp=4))
    manager.register(first.candidate())
    manager.register(second.candidate())
    manifest = frontier.strict_json(root / "frontier_manifest.json")
    pending = manifest["pending_prune"][0]
    pending_bundle = root / pending["artifact_path"]
    (pending_bundle / "factor_pack.pt").write_bytes(b"tampered pending bytes")
    retained_paths = [root / record["artifact_path"] for record in manifest["records"]]
    with pytest.raises(frontier.ExactCandidateFrontierError, match="changed"):
        manager.inspect()
    assert all(path.is_dir() for path in retained_paths)
    assert first.run.is_dir() and second.run.is_dir()

    # Restore a clean frontier, then forge every hash while retaining an escape.
    clean_root = tmp_path / "clean-frontier"
    clean_manager = frontier.ExactCandidateFrontier(clean_root)
    case = _make_run(tmp_path, "escape", metrics=_metrics(lcp=3))
    clean_manager.register(case.candidate())
    forged = frontier.strict_json(clean_root / "frontier_manifest.json")
    record = forged["records"][0]
    record["artifact_path"] = "../canonical_blank"
    record["record_sha256"] = frontier._record_hash(record, "record_sha256")
    forged["manifest_sha256"] = frontier._record_hash(forged, "manifest_sha256")
    _write_json(clean_root / "frontier_manifest.json", forged)
    with pytest.raises(frontier.ExactCandidateFrontierError, match="escapes"):
        clean_manager.inspect()


def test_frontier_cannot_overlap_blank_or_cache_and_rejects_symlink_artifacts(
    tmp_path: Path,
) -> None:
    blank = tmp_path / "blank"
    blank.mkdir()
    with pytest.raises(frontier.ExactCandidateFrontierError, match="overlaps"):
        frontier.ExactCandidateFrontier(blank / "frontier", protected_roots=[blank])

    case = _make_run(tmp_path, "symlink")
    factor = case.run / "rejected_factor_pack.pt"
    real = case.run / "real-factor.pt"
    real.write_bytes(factor.read_bytes())
    factor.unlink()
    try:
        os.symlink(real, factor)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="canonical direct run artifact|ordinary file",
    ):
        case.candidate()


def test_cli_disables_apply_prune_before_creating_frontier(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("retention_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    root = tmp_path / "cli-frontier"
    args = parser.parse_args(["--frontier-root", str(root), "--inspect"])
    assert args.apply_prune is False
    module.main(["--frontier-root", str(root), "--inspect"])
    assert not root.exists()
    with pytest.raises(SystemExit, match="unsupported"):
        module.main(
            [
                "--frontier-root",
                str(root),
                "--prune-only",
                "--apply-prune",
            ]
        )
    assert not root.exists()


def test_conservative_v1_contains_no_recursive_delete_implementation() -> None:
    source = Path(frontier.__file__).read_text(encoding="utf-8")
    cli_source = SCRIPT.read_text(encoding="utf-8")
    assert "rmtree" not in source
    assert "Remove-Item" not in source
    assert "rmtree" not in cli_source
