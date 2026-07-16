from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import rescore_qwen_exact_target_native_guided_gud_v4_frontier as rescore  # noqa: E402


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _metrics(
    label: str,
    *,
    action_agreement_count: int = 55,
    parent_action_mean_rank: float = 400.0,
    exact_rollout_count: int = 0,
    token_match_count: int = 55,
    lcp_token_count: int = 30,
    lcp_at_least_4_count: int = 0,
    repeat_fraction_excess: float = 0.375,
    adjacent_repeat_fraction_excess: float = 0.525,
    longest_run_excess: int = 3,
) -> dict[str, Any]:
    action_count = 192
    rollout_count = 24
    rollout_token_count = 192
    return {
        "split": "calibration-selection",
        "action_count": action_count,
        "action_agreement_count": action_agreement_count,
        "action_accuracy": action_agreement_count / action_count,
        "parent_action_mean_logprob": -3.0,
        "parent_action_topk_fraction": 0.6,
        "parent_action_mean_rank": parent_action_mean_rank,
        "parent_action_worst_rank": 1_000,
        "rollout_count": rollout_count,
        "exact_rollout_count": exact_rollout_count,
        "exact_rollout_fraction": exact_rollout_count / rollout_count,
        "rollout_token_count": rollout_token_count,
        "token_match_count": token_match_count,
        "token_match_fraction": token_match_count / rollout_token_count,
        "lcp_token_count": lcp_token_count,
        "lcp_fraction": lcp_token_count / rollout_token_count,
        "lcp_at_least_4_count": lcp_at_least_4_count,
        "repeat_fraction_excess": repeat_fraction_excess,
        "adjacent_repeat_fraction_excess": adjacent_repeat_fraction_excess,
        "longest_run_excess": longest_run_excess,
        "report_sha256": _digest(f"metrics-{label}"),
    }


def _pareto_safe(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    left = rescore.publication._frontier_metrics(candidate, "candidate")
    right = rescore.publication._frontier_metrics(baseline, "baseline")
    pairs = tuple(zip(left.pareto_objectives, right.pareto_objectives, strict=True))
    return all(a == b for a, b in pairs) or (
        all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)
    )


def _frontier(
    *,
    overrides: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    observations: list[dict[str, Any]] = []
    for arm_index, arm_id in enumerate(rescore.tournament.ARM_IDS):
        baseline = _metrics(f"arm-{arm_index}-step-0")
        for local_index, step in enumerate(rescore.tournament.CHECKPOINT_STEPS):
            ordinal = arm_index * len(rescore.tournament.CHECKPOINT_STEPS) + local_index
            metrics = copy.deepcopy(
                overrides.get((arm_index, step), baseline)
            )
            # report_sha256 is provenance only, not a behavior objective or rank.
            if (arm_index, step) not in overrides:
                metrics["report_sha256"] = _digest(
                    f"arm-{arm_index}-step-{step}-metrics"
                )
            safe = _pareto_safe(metrics, baseline)
            rank = rescore.tournament.v3_tournament.tournament_behavior_rank(metrics)
            observations.append(
                {
                    "ordinal": ordinal,
                    "arm_id": arm_id,
                    "arm_index": arm_index,
                    "step": step,
                    "checkpoint_sha256": _digest(f"checkpoint-{ordinal}"),
                    "runtime_factor_state_sha256": _digest(f"runtime-{ordinal}"),
                    "portable_factor_state_sha256": _digest(f"portable-{ordinal}"),
                    "pareto_safe_vs_step0": safe,
                    "metrics": metrics,
                    "metrics_sha256": rescore.tournament.canonical_sha256(metrics),
                    "rank": list(rank),
                    "eligible_for_retention": step == 0 or safe,
                    "disposition": "not-retained-rank-cutoff",
                    "retained_after_callback": False,
                    "factor_pack_created": False,
                    "evicted_portable_factor_state_sha256s": [],
                    "terminal_verified": True,
                    "terminal_pareto_safe_vs_step0": safe,
                }
            )
    retained = []
    for row in observations[-3:]:
        retained.append(
            {
                "portable_factor_state_sha256": row[
                    "portable_factor_state_sha256"
                ]
            }
        )
    return {
        "schema": rescore.tournament.FRONTIER_SCHEMA,
        "evaluation_contract_sha256": _digest("evaluation"),
        "completed_arms": list(rescore.tournament.ARM_IDS),
        "retained_count": 3,
        "retained": retained,
        "observed_callback_count": 96,
        "pending_callback_count": 0,
        "observed_callbacks": observations,
        "distinct_full_v4_states_only": True,
        "portable_content_identity_used": True,
        "step_zero_ranked": True,
        "later_states_require_pareto_safety_vs_step_zero": True,
        "sealed_metrics_used": False,
        "internal_numeric_matching_used": False,
        "callback_metrics_are_durable_without_factor_pack_retention": True,
        "maximum_retained_factor_packs": 3,
    }


def _decision(payload: dict[str, Any], arm_index: int, step: int) -> dict[str, Any]:
    return next(
        row
        for row in payload["callback_decisions"]
        if row["arm_index"] == arm_index and row["step"] == step
    )


def test_frontier_tamper_and_incompleteness_fail_closed() -> None:
    frontier = _frontier()
    tampered = copy.deepcopy(frontier)
    tampered["observed_callbacks"][8]["metrics"]["token_match_count"] += 1
    with pytest.raises(rescore.V4RolloutFrontierRescoreError, match="tampered"):
        rescore.rescore_authenticated_frontier(
            tampered,
            evaluation_contract_sha256=_digest("evaluation"),
        )

    incomplete = copy.deepcopy(frontier)
    incomplete["observed_callbacks"].pop()
    with pytest.raises(
        rescore.V4RolloutFrontierRescoreError,
        match="complete sealed-free 8x12",
    ):
        rescore.rescore_authenticated_frontier(
            incomplete,
            evaluation_contract_sha256=_digest("evaluation"),
        )


def test_step8_like_rollout_gain_is_rejected_for_repeat_regression() -> None:
    candidate = _metrics(
        "step8-repeat-regression",
        action_agreement_count=56,
        token_match_count=56,
        lcp_token_count=32,
        lcp_at_least_4_count=1,
        repeat_fraction_excess=0.52,
        adjacent_repeat_fraction_excess=0.50,
    )
    payload = rescore.rescore_authenticated_frontier(
        _frontier(overrides={(0, 8): candidate}),
        evaluation_contract_sha256=_digest("evaluation"),
    )
    decision = _decision(payload, 0, 8)
    assert decision["gates"]["rollout_first_improves_vs_step0"] is True
    assert decision["gates"]["lcp_at_least_4_passed"] is True
    assert decision["gates"]["repeat_fraction_nonregression"] is False
    assert decision["eligible"] is False
    assert "repeat-fraction-excess-regressed-vs-arm-step0" in decision[
        "ineligibility_reasons"
    ]


def test_step12_like_behavior_is_rejected_for_no_acquisition() -> None:
    candidate = _metrics(
        "step12-no-acquisition",
        action_agreement_count=52,
        token_match_count=52,
        lcp_token_count=29,
        lcp_at_least_4_count=0,
        repeat_fraction_excess=0.31,
        adjacent_repeat_fraction_excess=0.40,
    )
    payload = rescore.rescore_authenticated_frontier(
        _frontier(overrides={(0, 12): candidate}),
        evaluation_contract_sha256=_digest("evaluation"),
    )
    decision = _decision(payload, 0, 12)
    assert decision["gates"]["lcp_at_least_4_passed"] is False
    assert decision["gates"]["rollout_first_improves_vs_step0"] is False
    assert decision["eligible"] is False


def test_rollout_acquisition_is_accepted_despite_action_and_rank_regression() -> None:
    candidate = _metrics(
        "rollout-gain-action-regression",
        action_agreement_count=50,
        parent_action_mean_rank=500.0,
        token_match_count=56,
        lcp_token_count=32,
        lcp_at_least_4_count=1,
        repeat_fraction_excess=0.35,
        adjacent_repeat_fraction_excess=0.50,
    )
    payload = rescore.rescore_authenticated_frontier(
        _frontier(overrides={(0, 8): candidate}),
        evaluation_contract_sha256=_digest("evaluation"),
    )
    decision = _decision(payload, 0, 8)
    assert candidate["action_agreement_count"] < 55
    assert candidate["parent_action_mean_rank"] > 400
    assert decision["eligible"] is True
    assert decision["pareto_diagnostic"]["recomputed_pareto_safe_vs_step0"] is False
    assert decision["pareto_diagnostic"]["role"] == (
        "diagnostic-only-not-an-eligibility-gate"
    )
    assert payload["arms"][0]["best_eligible_callback"]["step"] == 8
    assert payload["global_eligible_callbacks"][0]["step"] == 8
    assert payload["next_action"]["action"] == (
        "replay-one-preregistered-arm-to-recover-discarded-pack"
    )
    assert payload["next_action"]["maximum_discarded_pack_replays_authorized"] == 1


def test_no_qualifier_routes_to_balanced_all_site_dense_steps_9_through_11() -> None:
    payload = rescore.rescore_authenticated_frontier(
        _frontier(),
        evaluation_contract_sha256=_digest("evaluation"),
    )
    assert payload["authenticated_callback_count"] == 96
    assert payload["authenticated_arm_count"] == 8
    assert payload["global_eligible_callbacks"] == []
    assert payload["next_action"]["action"] == (
        "run-balanced-all-site-dense-checkpoints-9-through-11"
    )


def test_completed_authentication_binds_explicit_frontier_and_existing_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = tmp_path / "preregistration.json"
    run_report = tmp_path / "run_report.json"
    winner_selection = tmp_path / "winner_selection.json"
    frontier = tmp_path / "candidate_frontier.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    frontier_payload = _frontier()
    _write_json(frontier, frontier_payload)
    frontier_sha = rescore.stage._file_sha256(frontier)
    binding = {
        "schema": rescore.tournament.FRONTIER_SCHEMA,
        "path": str(frontier),
        "sha256": frontier_sha,
        "observed_callback_count": 96,
        "pending_callback_count": 0,
        "completed_arms": list(rescore.tournament.ARM_IDS),
    }
    run_payload = {
        "candidate_frontier": binding,
        "sealed_forward_count": 0,
        "sealed_evaluation_accessed": False,
    }
    winner_payload = {
        "candidate_frontier": binding,
        "sealed_evaluation_accessed": False,
    }
    _write_json(run_report, run_payload)
    _write_json(winner_selection, winner_payload)
    contract = SimpleNamespace(
        path=preregistration.resolve(),
        file_sha256=rescore.stage._file_sha256(preregistration),
        evaluation_contract_sha256=_digest("evaluation"),
    )
    receipt = SimpleNamespace(as_dict=lambda: {"mode": "test"})
    calls: list[str] = []

    def authenticate_history(path: Path, digest: str) -> Any:
        calls.append("preregistration")
        assert Path(path) == preregistration
        assert digest == contract.file_sha256
        return SimpleNamespace(contract=contract, receipt=receipt)

    def authenticate_completed(
        received_contract: Any,
        received_run: Any,
        received_winner: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append("run-frontier")
        assert received_contract is contract
        assert received_run == run_payload
        assert received_winner == winner_payload
        return (
            {
                "candidate_frontier": binding,
                "portable_factor_state_sha256": _digest("legacy-winner"),
            },
            {},
        )

    monkeypatch.setattr(
        rescore.history,
        "authenticate_guided_gud_candidate_preregistration_with_history",
        authenticate_history,
    )
    monkeypatch.setattr(rescore.publication, "_winner_payload", authenticate_completed)
    authenticated = rescore.authenticate_completed_v4(
        preregistration=preregistration,
        preregistration_sha256=contract.file_sha256,
        run_report=run_report,
        run_report_sha256=rescore.stage._file_sha256(run_report),
        winner_selection=winner_selection,
        winner_selection_sha256=rescore.stage._file_sha256(winner_selection),
        frontier=frontier,
        frontier_sha256=frontier_sha,
    )
    assert calls == ["preregistration", "run-frontier"]
    assert authenticated.frontier_sha256 == frontier_sha

    with pytest.raises(rescore.V4RolloutFrontierRescoreError, match="exact binding"):
        rescore.authenticate_completed_v4(
            preregistration=preregistration,
            preregistration_sha256=contract.file_sha256,
            run_report=run_report,
            run_report_sha256=_digest("tampered-run-binding"),
            winner_selection=winner_selection,
            winner_selection_sha256=rescore.stage._file_sha256(winner_selection),
            frontier=frontier,
            frontier_sha256=frontier_sha,
        )


def test_evidence_is_atomic_self_hashed_and_refuses_overwrite(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema": rescore.EVIDENCE_SCHEMA,
        "experimental_unaccepted": True,
    }
    payload["evidence_sha256"] = rescore.tournament.canonical_sha256(payload)
    output = tmp_path / "rescore.json"
    assert rescore.write_evidence(output, payload) == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(rescore.V4RolloutFrontierRescoreError, match="absent"):
        rescore.write_evidence(output, payload)
