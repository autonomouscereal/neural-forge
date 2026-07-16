#!/usr/bin/env python3
"""Model-free rollout-first rescore of one completed guided-GUD v4 frontier.

This does not select or publish a model.  It authenticates the completed
eight-arm tournament with the existing transitive authenticators, then uses
only its durable calibration-output callback metrics to ask a narrower
research question: did any scheduled post-step-zero callback acquire rollout
behavior without worsening any of the three repetition diagnostics relative
to *that arm's* step-zero state?

The original eleven-objective Pareto decision is recomputed for audit, but is
deliberately diagnostic-only.  A discarded qualifying factor pack must be
replayed from its preregistered arm and checkpoint; this evidence file never
reconstructs, accepts, or publishes a model.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import authenticate_qwen_exact_target_native_guided_gud_candidate_tournament_history as history
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament
import run_qwen_exact_target_native_guided_gud_winner_publication as publication
import run_qwen_exact_target_native_stage as stage
import neural_forge.hf_exact_candidate_frontier as frontier_tools
import neural_forge.hf_target_native_behavior_distillation as behavior


EVIDENCE_SCHEMA = (
    "neural-forge-qwen-exact-target-guided-gud-v4-rollout-frontier-rescore-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OBJECTIVE_NAMES = (
    "action_accuracy",
    "parent_action_mean_logprob",
    "parent_action_topk_fraction",
    "negative_parent_action_mean_rank",
    "negative_parent_action_worst_rank",
    "token_match_fraction",
    "lcp_fraction",
    "exact_rollout_fraction",
    "negative_repeat_fraction_excess",
    "negative_adjacent_repeat_fraction_excess",
    "negative_longest_run_excess",
)
_FRONTIER_FIELDS = {
    "schema",
    "evaluation_contract_sha256",
    "completed_arms",
    "retained_count",
    "retained",
    "observed_callback_count",
    "pending_callback_count",
    "observed_callbacks",
    "distinct_full_v4_states_only",
    "portable_content_identity_used",
    "step_zero_ranked",
    "later_states_require_pareto_safety_vs_step_zero",
    "sealed_metrics_used",
    "internal_numeric_matching_used",
    "callback_metrics_are_durable_without_factor_pack_retention",
    "maximum_retained_factor_packs",
}
_OBSERVATION_FIELDS = {
    "ordinal",
    "arm_id",
    "arm_index",
    "step",
    "checkpoint_sha256",
    "runtime_factor_state_sha256",
    "portable_factor_state_sha256",
    "pareto_safe_vs_step0",
    "metrics",
    "metrics_sha256",
    "rank",
    "eligible_for_retention",
    "disposition",
    "retained_after_callback",
    "factor_pack_created",
    "evicted_portable_factor_state_sha256s",
    "terminal_verified",
    "terminal_pareto_safe_vs_step0",
}


class V4RolloutFrontierRescoreError(RuntimeError):
    """Completed-v4 authentication or model-free rescoring failed closed."""


@dataclass(frozen=True, slots=True)
class AuthenticatedCompletedV4:
    """Immutable bindings returned by the existing production authenticators."""

    contract: tournament.AuthenticatedGuidedGUDCandidateTournament
    preregistration_receipt: history.V4PreregistrationAuthenticationReceipt
    run_report_path: Path
    run_report_sha256: str
    winner_selection_path: Path
    winner_selection_sha256: str
    frontier_path: Path
    frontier_sha256: str
    run_report: Mapping[str, Any]
    winner_selection: Mapping[str, Any]
    frontier: Mapping[str, Any]
    legacy_winner_authentication: Mapping[str, Any]


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V4RolloutFrontierRescoreError(
            f"{name} must be a lowercase SHA-256."
        )
    return value


def _bound_json(path: Path | str, digest: str, name: str) -> tuple[Path, Mapping[str, Any]]:
    expected = _sha256(digest, f"{name}.sha256")
    try:
        resolved = publication._bound_file(Path(path), expected, name)
        payload = publication._bound_json(resolved, expected, name)
    except publication.GuidedGUDWinnerPublicationError as error:
        raise V4RolloutFrontierRescoreError(f"{name} failed its exact binding.") from error
    if not isinstance(payload, Mapping):
        raise V4RolloutFrontierRescoreError(f"{name} must be a JSON object.")
    return resolved, payload


def authenticate_completed_v4(
    *,
    preregistration: Path | str,
    preregistration_sha256: str,
    run_report: Path | str,
    run_report_sha256: str,
    winner_selection: Path | str,
    winner_selection_sha256: str,
    frontier: Path | str,
    frontier_sha256: str,
) -> AuthenticatedCompletedV4:
    """Authenticate all completed-v4 identities without loading either model."""

    prereg_sha = _sha256(preregistration_sha256, "preregistration_sha256")
    try:
        prereg = (
            history.authenticate_guided_gud_candidate_preregistration_with_history(
                preregistration,
                prereg_sha,
            )
        )
    except (
        history.HistoricalV4PreregistrationAuthenticationError,
        tournament.GuidedGUDCandidateTournamentError,
        OSError,
    ) as error:
        raise V4RolloutFrontierRescoreError(
            "Completed v4 preregistration/source authentication failed."
        ) from error

    run_sha = _sha256(run_report_sha256, "run_report_sha256")
    winner_sha = _sha256(winner_selection_sha256, "winner_selection_sha256")
    run_path, run_payload = _bound_json(run_report, run_sha, "v4 run report")
    winner_path, winner_payload = _bound_json(
        winner_selection,
        winner_sha,
        "v4 winner selection",
    )
    try:
        # This existing gate authenticates the completed run receipts, all 96
        # callback rows, final retention simulation, and retained v4 packs.  It
        # loads only a factor pack on CPU; no parent/child model is constructed.
        legacy_winner, _factor_payload = publication._winner_payload(
            prereg.contract,
            run_payload,
            winner_payload,
        )
    except publication.GuidedGUDWinnerPublicationError as error:
        raise V4RolloutFrontierRescoreError(
            "Completed v4 run/frontier authentication failed."
        ) from error

    frontier_sha = _sha256(frontier_sha256, "frontier_sha256")
    frontier_path, frontier_payload = _bound_json(
        frontier,
        frontier_sha,
        "final v4 candidate frontier",
    )
    authenticated_binding = legacy_winner.get("candidate_frontier")
    if (
        not isinstance(authenticated_binding, Mapping)
        or authenticated_binding.get("sha256") != frontier_sha
        or Path(str(authenticated_binding.get("path"))).resolve() != frontier_path
        or run_payload.get("candidate_frontier") != authenticated_binding
        or winner_payload.get("candidate_frontier") != authenticated_binding
    ):
        raise V4RolloutFrontierRescoreError(
            "Explicit frontier is not the final frontier bound by run and selection."
        )
    if (
        run_payload.get("sealed_forward_count") != 0
        or run_payload.get("sealed_evaluation_accessed") is not False
        or winner_payload.get("sealed_evaluation_accessed") is not False
        or frontier_payload.get("sealed_metrics_used") is not False
    ):
        raise V4RolloutFrontierRescoreError(
            "Completed v4 evidence indicates sealed behavior access."
        )
    return AuthenticatedCompletedV4(
        contract=prereg.contract,
        preregistration_receipt=prereg.receipt,
        run_report_path=run_path,
        run_report_sha256=run_sha,
        winner_selection_path=winner_path,
        winner_selection_sha256=winner_sha,
        frontier_path=frontier_path,
        frontier_sha256=frontier_sha,
        run_report=run_payload,
        winner_selection=winner_payload,
        frontier=frontier_payload,
        legacy_winner_authentication=dict(legacy_winner),
    )


def _metrics(value: Any, name: str) -> Any:
    try:
        metric = publication._frontier_metrics(value, name)
        # rollout_first_rank performs additional count/fraction and banned-key
        # checks that the historical frontier parser intentionally shares.
        tournament.v3_tournament.tournament_behavior_rank(value)
    except (
        publication.GuidedGUDWinnerPublicationError,
        frontier_tools.ExactCandidateFrontierError,
    ) as error:
        raise V4RolloutFrontierRescoreError(f"{name} failed authentication.") from error
    return metric


def _pareto_diagnostic(candidate: Any, baseline: Any) -> dict[str, Any]:
    candidate_objectives = tuple(candidate.pareto_objectives)
    baseline_objectives = tuple(baseline.pareto_objectives)
    pairs = tuple(zip(candidate_objectives, baseline_objectives, strict=True))
    equal = all(left == right for left, right in pairs)
    dominates = all(left >= right for left, right in pairs) and any(
        left > right for left, right in pairs
    )
    return {
        "objective_names": list(_OBJECTIVE_NAMES),
        "candidate_objectives": list(candidate_objectives),
        "step0_objectives": list(baseline_objectives),
        "objectives_equal_to_step0": equal,
        "dominates_step0": dominates,
        "recomputed_pareto_safe_vs_step0": equal or dominates,
        "role": "diagnostic-only-not-an-eligibility-gate",
    }


def _candidate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": row["ordinal"],
        "arm_id": row["arm_id"],
        "arm_index": row["arm_index"],
        "step": row["step"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "runtime_factor_state_sha256": row["runtime_factor_state_sha256"],
        "portable_factor_state_sha256": row["portable_factor_state_sha256"],
        "metrics_sha256": row["metrics_sha256"],
        "rollout_first_rank": list(row["rollout_first_rank"]),
        "final_retained_pack_available": row["final_retained_pack_available"],
        "discarded_pack_replay_required": row["discarded_pack_replay_required"],
    }


def _validate_frontier_envelope(
    frontier: Mapping[str, Any],
    *,
    evaluation_contract_sha256: str,
) -> tuple[list[Mapping[str, Any]], frozenset[str]]:
    observations = frontier.get("observed_callbacks")
    retained = frontier.get("retained")
    if (
        set(frontier) != _FRONTIER_FIELDS
        or frontier.get("schema") != tournament.FRONTIER_SCHEMA
        or frontier.get("evaluation_contract_sha256") != evaluation_contract_sha256
        or frontier.get("completed_arms") != list(tournament.ARM_IDS)
        or frontier.get("observed_callback_count")
        != tournament.EXPECTED_CALLBACK_COUNT
        or frontier.get("pending_callback_count") != 0
        or not isinstance(observations, list)
        or len(observations) != tournament.EXPECTED_CALLBACK_COUNT
        or not isinstance(retained, list)
        or frontier.get("retained_count") != tournament.MAX_RETAINED_STATES
        or len(retained) != tournament.MAX_RETAINED_STATES
        or frontier.get("maximum_retained_factor_packs")
        != tournament.MAX_RETAINED_STATES
        or frontier.get("sealed_metrics_used") is not False
        or frontier.get("internal_numeric_matching_used") is not False
        or frontier.get("callback_metrics_are_durable_without_factor_pack_retention")
        is not True
    ):
        raise V4RolloutFrontierRescoreError(
            "Final frontier is not the complete sealed-free 8x12 envelope."
        )
    retained_hashes: set[str] = set()
    for index, row in enumerate(retained):
        if not isinstance(row, Mapping):
            raise V4RolloutFrontierRescoreError(
                f"retained[{index}] is not an authenticated mapping."
            )
        retained_hashes.add(
            _sha256(
                row.get("portable_factor_state_sha256"),
                f"retained[{index}].portable_factor_state_sha256",
            )
        )
    if len(retained_hashes) != tournament.MAX_RETAINED_STATES:
        raise V4RolloutFrontierRescoreError(
            "Final frontier retained identities are not distinct."
        )
    return observations, frozenset(retained_hashes)


def rescore_authenticated_frontier(
    frontier: Mapping[str, Any],
    *,
    evaluation_contract_sha256: str,
) -> dict[str, Any]:
    """Recompute rollout acquisition from one already authenticated frontier."""

    evaluation_sha = _sha256(
        evaluation_contract_sha256,
        "evaluation_contract_sha256",
    )
    observations, retained_hashes = _validate_frontier_envelope(
        frontier,
        evaluation_contract_sha256=evaluation_sha,
    )
    by_arm: list[list[dict[str, Any]]] = [list() for _ in tournament.ARM_IDS]
    all_eligible: list[dict[str, Any]] = []

    for ordinal, raw in enumerate(observations):
        expected_arm_index = ordinal // len(tournament.CHECKPOINT_STEPS)
        expected_step = tournament.CHECKPOINT_STEPS[
            ordinal % len(tournament.CHECKPOINT_STEPS)
        ]
        expected_arm = tournament.ARM_IDS[expected_arm_index]
        if (
            not isinstance(raw, Mapping)
            or set(raw) != _OBSERVATION_FIELDS
            or raw.get("ordinal") != ordinal
            or raw.get("arm_index") != expected_arm_index
            or raw.get("arm_id") != expected_arm
            or raw.get("step") != expected_step
            or raw.get("terminal_verified") is not True
            or raw.get("terminal_pareto_safe_vs_step0")
            != raw.get("pareto_safe_vs_step0")
            or raw.get("disposition")
            not in tournament.v3_tournament.FINAL_CALLBACK_DISPOSITIONS
            or raw.get("disposition") == "error-inconsistent-duplicate"
            or not isinstance(raw.get("metrics"), Mapping)
            or raw.get("metrics_sha256")
            != tournament.canonical_sha256(raw["metrics"])
        ):
            raise V4RolloutFrontierRescoreError(
                f"Callback {ordinal} is incomplete, reordered, or tampered."
            )
        for field in (
            "checkpoint_sha256",
            "runtime_factor_state_sha256",
            "portable_factor_state_sha256",
            "metrics_sha256",
        ):
            _sha256(raw.get(field), f"callback[{ordinal}].{field}")
        metric = _metrics(raw["metrics"], f"callback[{ordinal}].metrics")
        rank = tuple(tournament.v3_tournament.tournament_behavior_rank(raw["metrics"]))
        if raw.get("rank") != list(rank):
            raise V4RolloutFrontierRescoreError(
                f"Callback {ordinal} stored rollout-first rank was forged."
            )
        by_arm[expected_arm_index].append(
            {
                "raw": raw,
                "metrics": metric,
                "rank": rank,
            }
        )

    decisions: list[dict[str, Any]] = []
    arm_reports: list[dict[str, Any]] = []
    for arm_index, arm_id in enumerate(tournament.ARM_IDS):
        rows = by_arm[arm_index]
        if (
            len(rows) != len(tournament.CHECKPOINT_STEPS)
            or [row["raw"]["step"] for row in rows]
            != list(tournament.CHECKPOINT_STEPS)
        ):
            raise V4RolloutFrontierRescoreError(
                f"{arm_id} does not contain exactly the scheduled 12 callbacks."
            )
        baseline = rows[0]
        baseline_metric = baseline["metrics"]
        baseline_rank = baseline["rank"]
        arm_eligible: list[dict[str, Any]] = []
        for row in rows:
            raw = row["raw"]
            metric = row["metrics"]
            rank = row["rank"]
            pareto = _pareto_diagnostic(metric, baseline_metric)
            if raw["pareto_safe_vs_step0"] is not pareto[
                "recomputed_pareto_safe_vs_step0"
            ]:
                raise V4RolloutFrontierRescoreError(
                    f"Callback {raw['ordinal']} Pareto diagnostic was forged."
                )
            promoted = raw["step"] > 0
            lcp_gate = metric.lcp_at_least_4_count >= 1
            rank_gate = rank < baseline_rank
            repeat_gate = (
                metric.repeat_fraction_excess
                <= baseline_metric.repeat_fraction_excess
            )
            adjacent_gate = (
                metric.adjacent_repeat_fraction_excess
                <= baseline_metric.adjacent_repeat_fraction_excess
            )
            longest_gate = (
                metric.longest_run_excess <= baseline_metric.longest_run_excess
            )
            eligible = all(
                (
                    promoted,
                    lcp_gate,
                    rank_gate,
                    repeat_gate,
                    adjacent_gate,
                    longest_gate,
                )
            )
            reasons = []
            for passed, reason in (
                (promoted, "step-zero-baseline"),
                (lcp_gate, "no-rollout-reached-four-lcp-tokens"),
                (rank_gate, "rollout-first-key-did-not-improve-vs-arm-step0"),
                (repeat_gate, "repeat-fraction-excess-regressed-vs-arm-step0"),
                (
                    adjacent_gate,
                    "adjacent-repeat-fraction-excess-regressed-vs-arm-step0",
                ),
                (longest_gate, "longest-run-excess-regressed-vs-arm-step0"),
            ):
                if not passed:
                    reasons.append(reason)
            retained = raw["portable_factor_state_sha256"] in retained_hashes
            decision = {
                "ordinal": raw["ordinal"],
                "arm_id": arm_id,
                "arm_index": arm_index,
                "step": raw["step"],
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "runtime_factor_state_sha256": raw[
                    "runtime_factor_state_sha256"
                ],
                "portable_factor_state_sha256": raw[
                    "portable_factor_state_sha256"
                ],
                "metrics_sha256": raw["metrics_sha256"],
                "rollout_first_rank": rank,
                "step0_rollout_first_rank": baseline_rank,
                "gates": {
                    "post_step_zero": promoted,
                    "minimum_lcp_at_least_4_count": 1,
                    "observed_lcp_at_least_4_count": metric.lcp_at_least_4_count,
                    "lcp_at_least_4_passed": lcp_gate,
                    "rollout_first_improves_vs_step0": rank_gate,
                    "step0_repeat_fraction_excess": (
                        baseline_metric.repeat_fraction_excess
                    ),
                    "candidate_repeat_fraction_excess": (
                        metric.repeat_fraction_excess
                    ),
                    "repeat_fraction_nonregression": repeat_gate,
                    "step0_adjacent_repeat_fraction_excess": (
                        baseline_metric.adjacent_repeat_fraction_excess
                    ),
                    "candidate_adjacent_repeat_fraction_excess": (
                        metric.adjacent_repeat_fraction_excess
                    ),
                    "adjacent_repeat_fraction_nonregression": adjacent_gate,
                    "step0_longest_run_excess": baseline_metric.longest_run_excess,
                    "candidate_longest_run_excess": metric.longest_run_excess,
                    "longest_run_nonregression": longest_gate,
                },
                "pareto_diagnostic": pareto,
                "declared_pareto_safe_vs_step0": raw["pareto_safe_vs_step0"],
                "eligible": eligible,
                "ineligibility_reasons": reasons,
                "final_retained_pack_available": retained,
                "discarded_pack_replay_required": eligible and not retained,
            }
            decisions.append(decision)
            if eligible:
                arm_eligible.append(decision)
                all_eligible.append(decision)
        ordered_arm = sorted(
            arm_eligible,
            key=lambda value: (
                value["rollout_first_rank"],
                value["step"],
                value["ordinal"],
            ),
        )
        arm_reports.append(
            {
                "arm_id": arm_id,
                "arm_index": arm_index,
                "step0_callback_ordinal": baseline["raw"]["ordinal"],
                "step0_metrics_sha256": baseline["raw"]["metrics_sha256"],
                "step0_rollout_first_rank": list(baseline_rank),
                "scheduled_callback_count": len(rows),
                "eligible_callbacks": [
                    _candidate_summary(value) for value in ordered_arm
                ],
                "best_eligible_callback": (
                    _candidate_summary(ordered_arm[0]) if ordered_arm else None
                ),
            }
        )

    ordered_global = sorted(
        all_eligible,
        key=lambda value: (
            value["rollout_first_rank"],
            value["arm_index"],
            value["step"],
            value["ordinal"],
        ),
    )
    replay_required = [
        value for value in ordered_global if value["discarded_pack_replay_required"]
    ]
    best = ordered_global[0] if ordered_global else None
    if best is None:
        next_action = {
            "action": "run-balanced-all-site-dense-checkpoints-9-through-11",
            "reason": "no-completed-v4-callback-passed-rollout-acquisition-gates",
            "replay_callback": None,
            "maximum_discarded_pack_replays_authorized": 0,
        }
    elif best["discarded_pack_replay_required"]:
        next_action = {
            "action": "replay-one-preregistered-arm-to-recover-discarded-pack",
            "reason": "best-authenticated-qualifying-callback-pack-was-discarded",
            "replay_callback": _candidate_summary(best),
            "maximum_discarded_pack_replays_authorized": 1,
        }
    else:
        next_action = {
            "action": "use-existing-authenticated-retained-factor-pack-for-follow-up",
            "reason": "best-authenticated-qualifying-callback-pack-is-retained",
            "replay_callback": None,
            "maximum_discarded_pack_replays_authorized": 0,
        }
    return {
        "policy": {
            "selection_policy": "rollout-first-acquisition",
            "comparison_baseline": "each-arm-own-step-zero",
            "minimum_lcp_at_least_4_count": 1,
            "repeat_fraction_threshold": "less-than-or-equal-to-arm-step-zero",
            "adjacent_repeat_fraction_threshold": (
                "less-than-or-equal-to-arm-step-zero"
            ),
            "longest_run_threshold": "less-than-or-equal-to-arm-step-zero",
            "full_11_objective_pareto_role": "diagnostic-only",
            "global_order": (
                "existing-rollout-first-key-then-arm-index-step-ordinal"
            ),
            "fit_metrics_used": False,
            "sealed_metrics_used": False,
            "internal_numeric_matching_used": False,
        },
        "authenticated_callback_count": len(decisions),
        "authenticated_arm_count": len(arm_reports),
        "arms": arm_reports,
        "callback_decisions": decisions,
        "global_eligible_callbacks": [
            _candidate_summary(value) for value in ordered_global
        ],
        "global_replay_required_candidates": [
            _candidate_summary(value) for value in replay_required
        ],
        "next_action": next_action,
    }


def build_evidence(authenticated: AuthenticatedCompletedV4) -> dict[str, Any]:
    # Close the public authenticate/use gap: all four supplied JSON roots must
    # still have their exact bytes immediately before the rescore is derived.
    rebound: dict[str, Mapping[str, Any]] = {}
    for name, path, digest in (
        (
            "preregistration",
            authenticated.contract.path,
            authenticated.contract.file_sha256,
        ),
        ("run report", authenticated.run_report_path, authenticated.run_report_sha256),
        (
            "winner selection",
            authenticated.winner_selection_path,
            authenticated.winner_selection_sha256,
        ),
        ("frontier", authenticated.frontier_path, authenticated.frontier_sha256),
    ):
        _path, payload = _bound_json(path, digest, f"rebound {name}")
        rebound[name] = payload
    if (
        rebound["run report"] != authenticated.run_report
        or rebound["winner selection"] != authenticated.winner_selection
        or rebound["frontier"] != authenticated.frontier
    ):
        raise V4RolloutFrontierRescoreError(
            "Authenticated completed-v4 JSON changed before rescoring."
        )
    rescore = rescore_authenticated_frontier(
        rebound["frontier"],
        evaluation_contract_sha256=(
            authenticated.contract.evaluation_contract_sha256
        ),
    )
    legacy = authenticated.legacy_winner_authentication
    payload: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "authenticated-experimental-rescore-only",
        "authentication": {
            "preregistration": str(authenticated.contract.path),
            "preregistration_sha256": authenticated.contract.file_sha256,
            "preregistration_receipt": (
                authenticated.preregistration_receipt.as_dict()
            ),
            "evaluation_contract_sha256": (
                authenticated.contract.evaluation_contract_sha256
            ),
            "run_report": str(authenticated.run_report_path),
            "run_report_sha256": authenticated.run_report_sha256,
            "winner_selection": str(authenticated.winner_selection_path),
            "winner_selection_sha256": authenticated.winner_selection_sha256,
            "frontier": str(authenticated.frontier_path),
            "frontier_sha256": authenticated.frontier_sha256,
            "legacy_tournament_winner_portable_factor_state_sha256": (
                legacy["portable_factor_state_sha256"]
            ),
            "transitive_preregistration_source_authenticator": (
                "authenticate_qwen_exact_target_native_guided_gud_candidate_"
                "tournament_history.authenticate_guided_gud_candidate_"
                "preregistration_with_history"
            ),
            "completed_run_frontier_authenticator": (
                "run_qwen_exact_target_native_guided_gud_winner_publication."
                "_winner_payload"
            ),
            "rescore_implementation": {
                "rescorer_sha256": stage._file_sha256(Path(__file__).resolve()),
                "historical_preregistration_authenticator_sha256": (
                    stage._file_sha256(Path(history.__file__).resolve())
                ),
                "completed_run_frontier_authenticator_sha256": (
                    stage._file_sha256(Path(publication.__file__).resolve())
                ),
                "rollout_first_rank_sha256": stage._file_sha256(
                    Path(frontier_tools.__file__).resolve()
                ),
                "causal_behavior_metrics_sha256": stage._file_sha256(
                    Path(behavior.__file__).resolve()
                ),
            },
            "all_bound_producer_and_source_identities_authenticated": True,
            "complete_eight_arm_96_callback_frontier_authenticated": True,
        },
        "rescore": rescore,
        "experimental_unaccepted": True,
        "candidate_model_selected": False,
        "winner_declared": False,
        "accepted_model_written": False,
        "model_checkpoint_written": False,
        "publication_attempted": False,
        "parent_model_loaded": False,
        "child_model_loaded": False,
        "gpu_work_launched": False,
        "sealed_forward_count": 0,
        "sealed_metrics_accessed": False,
        "rescore_uses_durable_calibration_output_metrics_only": True,
    }
    payload["evidence_sha256"] = tournament.canonical_sha256(payload)
    return payload


def write_evidence(path: Path | str, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    if output.is_symlink() or output.exists():
        raise V4RolloutFrontierRescoreError(
            "Output evidence path must be absent and cannot be a symlink."
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise V4RolloutFrontierRescoreError(
            "Output evidence parent must be an existing real directory."
        )
    if payload.get("evidence_sha256") != tournament.canonical_sha256(
        {name: value for name, value in payload.items() if name != "evidence_sha256"}
    ):
        raise V4RolloutFrontierRescoreError("Evidence self-hash changed before write.")
    try:
        stage._atomic_json(output, payload)
    except (stage.ExactTargetNativeStageError, OSError, TypeError, ValueError) as error:
        raise V4RolloutFrontierRescoreError(
            "Atomic model-free evidence publication failed."
        ) from error
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--run-report-sha256", required=True)
    parser.add_argument("--winner-selection", type=Path, required=True)
    parser.add_argument("--winner-selection-sha256", required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--frontier-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    authenticated = authenticate_completed_v4(
        preregistration=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        run_report=args.run_report,
        run_report_sha256=args.run_report_sha256,
        winner_selection=args.winner_selection,
        winner_selection_sha256=args.winner_selection_sha256,
        frontier=args.frontier,
        frontier_sha256=args.frontier_sha256,
    )
    write_evidence(args.output, build_evidence(authenticated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
