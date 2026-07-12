from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from neural_forge.hf_trajectory_gate import (
    EVALUATION_SCHEMA,
    TRAJECTORY_SCHEMA,
    TrajectoryGateConfig,
    TrajectoryGateIntegrityError,
    evaluate_trajectory_gate,
    seal_evaluation,
    seal_trajectory,
    verify_gate_report,
)


BENCHMARK_SHA256 = "b" * 64
PARENT_A_SHA256 = "a" * 64
PARENT_B_SHA256 = "c" * 64
CANDIDATE_SHA256 = "d" * 64


def _evaluation(
    *,
    role: str,
    model_sha256: str,
    behavior: int = 800,
    generation: int = 500,
    choice: int = 200,
    manual: int = 56,
    ppl: float = 10.0,
    endpoint_cosine: float = 1.0,
    logit_cosine: float = 1.0,
    omitted_metrics: tuple[str, ...] = (),
) -> Mapping[str, Any]:
    metrics: dict[str, Any] = {
        "behavior": {"correct": behavior, "total": 1000},
        "exact_generation": {"correct": generation, "total": 1000},
        "choice": {"correct": choice, "total": 250},
        "manual_coherence": {"coherent": manual, "total": 64},
        "ppl": ppl,
        "endpoint_cosine": endpoint_cosine,
        "logit_cosine": logit_cosine,
    }
    for name in omitted_metrics:
        metrics.pop(name)
    document = {
        "schema": EVALUATION_SCHEMA,
        "role": role,
        "baseline_kind": "trained-parent" if role == "parent" else "folded-candidate",
        "model_sha256": model_sha256,
        "benchmark_sha256": BENCHMARK_SHA256,
        "metrics": metrics,
    }
    return seal_evaluation(document)


def _trajectory(
    mode: str,
    *,
    parent_ids: tuple[str, ...] = (PARENT_A_SHA256,),
    cosine: float = 0.96,
    drift: float = 0.02,
    evidence_class: str = "finite-trace",
) -> Mapping[str, Any]:
    comparisons = []
    for parent_id in parent_ids:
        comparisons.append(
            {
                "parent_model_sha256": parent_id,
                "evidence_class": evidence_class,
                "points": [
                    {
                        "depth": depth,
                        "step": step,
                        "cosine_similarity": cosine,
                        "normalized_l2_drift": drift,
                    }
                    for depth in (0, 1)
                    for step in range(4)
                ],
            }
        )
    return seal_trajectory(
        {
            "schema": TRAJECTORY_SCHEMA,
            "mode": mode,
            "candidate_model_sha256": CANDIDATE_SHA256,
            "benchmark_sha256": BENCHMARK_SHA256,
            "comparisons": comparisons,
        }
    )


def _gate(
    parents: tuple[Mapping[str, Any], ...],
    candidate: Mapping[str, Any],
    *,
    parent_weights: tuple[float, ...] | None = None,
    teacher: Mapping[str, Any] | None = None,
    rollout: Mapping[str, Any] | None = None,
    config: TrajectoryGateConfig | None = None,
) -> dict[str, Any]:
    parent_ids = tuple(str(parent["model_sha256"]) for parent in parents)
    return evaluate_trajectory_gate(
        parents,
        candidate,
        parent_weights=parent_weights,
        config=config,
        teacher_forced_trajectory=teacher or _trajectory(
            "teacher-forced", parent_ids=parent_ids
        ),
        free_rollout_trajectory=rollout or _trajectory(
            "free-rollout", parent_ids=parent_ids
        ),
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_rejects_high_endpoint_fit_with_total_autoregressive_collapse() -> None:
    parent = _evaluation(role="parent", model_sha256=PARENT_A_SHA256)
    candidate = _evaluation(
        role="candidate",
        model_sha256=CANDIDATE_SHA256,
        behavior=0,
        generation=0,
        choice=160,
        manual=0,
        ppl=8.0,
        endpoint_cosine=0.99,
        logit_cosine=0.99,
    )
    collapsed_rollout = _trajectory(
        "free-rollout", cosine=0.08, drift=2.5
    )

    report = _gate((parent,), candidate, rollout=collapsed_rollout)

    assert report["accepted"] is False
    assert report["decision"] == "reject"
    assert report["metric_gates"]["ppl"]["passed"] is True
    assert report["metric_gates"]["behavior"]["retention"] == 0.0
    assert report["metric_gates"]["exact_generation"]["retention"] == 0.0
    assert "behavior" in report["failed_gates"]
    assert "exact_generation" in report["failed_gates"]
    assert "trajectory.free_rollout" in report["failed_gates"]
    diagnostics = report["non_promotional_diagnostics"]
    assert diagnostics["endpoint_cosine"]["candidate"] == 0.99
    assert diagnostics["endpoint_cosine"]["used_for_promotion"] is False
    assert report["policy"]["endpoint_or_likelihood_only_can_promote"] is False


def test_accepts_coherent_candidate_with_complete_parent_relative_evidence() -> None:
    parent = _evaluation(role="parent", model_sha256=PARENT_A_SHA256)
    candidate = _evaluation(
        role="candidate",
        model_sha256=CANDIDATE_SHA256,
        behavior=720,
        generation=400,
        choice=180,
        manual=52,
        ppl=12.0,
        endpoint_cosine=0.92,
        logit_cosine=0.90,
    )

    report = _gate((parent,), candidate)

    assert report["accepted"] is True
    assert report["decision"] == "promote"
    assert report["failed_gates"] == []
    assert all(gate["passed"] for gate in report["metric_gates"].values())
    assert all(gate["passed"] for gate in report["trajectory_gates"].values())
    assert report["zero_gradient_contract"] == {
        "decision_only": True,
        "training_steps": 0,
        "gradients_used": False,
        "model_forwards_run": 0,
        "checkpoint_mutations": 0,
    }
    assert report["evidence_classification"]["overall"] == "finite-trace"
    assert report["evidence_classification"]["algebraic_equivalence_proven"] is False
    assert verify_gate_report(report)


def test_rejects_evaluation_cache_and_trajectory_fingerprint_tampering() -> None:
    parent = _evaluation(role="parent", model_sha256=PARENT_A_SHA256)
    candidate = _evaluation(role="candidate", model_sha256=CANDIDATE_SHA256)
    tampered_candidate = _thaw(candidate)
    tampered_candidate["metrics"]["behavior"]["correct"] = 0

    with pytest.raises(TrajectoryGateIntegrityError, match="evaluation_sha256"):
        _gate((parent,), tampered_candidate)

    tampered_trajectory = _thaw(_trajectory("free-rollout"))
    tampered_trajectory["trajectory_sha256"] = "0" * 64
    with pytest.raises(TrajectoryGateIntegrityError, match="trajectory_sha256"):
        _gate((parent,), candidate, rollout=tampered_trajectory)


def test_missing_required_metrics_and_trajectory_evidence_reject() -> None:
    parent = _evaluation(role="parent", model_sha256=PARENT_A_SHA256)
    candidate = _evaluation(
        role="candidate",
        model_sha256=CANDIDATE_SHA256,
        omitted_metrics=("manual_coherence",),
    )

    report = evaluate_trajectory_gate(
        (parent,),
        candidate,
        teacher_forced_trajectory=_trajectory("teacher-forced"),
        free_rollout_trajectory=None,
    )

    assert report["accepted"] is False
    assert report["metric_gates"]["manual_coherence"]["available"] is False
    assert report["metric_gates"]["manual_coherence"]["passed"] is False
    assert report["trajectory_gates"]["free_rollout"]["available"] is False
    assert report["trajectory_gates"]["free_rollout"]["passed"] is False
    assert "manual_coherence" in report["failed_gates"]
    assert "trajectory.free_rollout" in report["failed_gates"]


def test_multi_parent_weighted_baseline_controls_retention() -> None:
    parent_a = _evaluation(
        role="parent", model_sha256=PARENT_A_SHA256, behavior=900
    )
    parent_b = _evaluation(
        role="parent", model_sha256=PARENT_B_SHA256, behavior=500
    )
    candidate = _evaluation(
        role="candidate",
        model_sha256=CANDIDATE_SHA256,
        behavior=640,
        generation=450,
        choice=180,
        manual=52,
        ppl=11.0,
    )
    config = TrajectoryGateConfig(min_behavior_retention=0.79)

    report = _gate(
        (parent_a, parent_b),
        candidate,
        parent_weights=(3.0, 1.0),
        config=config,
    )

    behavior = report["metric_gates"]["behavior"]
    assert behavior["weighted_parent_baseline"] == pytest.approx(0.8)
    assert behavior["retention"] == pytest.approx(0.8)
    assert behavior["passed"] is True
    assert report["inputs"]["normalized_parent_weights"] == [0.75, 0.25]
    assert report["accepted"] is True


def test_random_or_blank_parent_baseline_is_forbidden() -> None:
    parent = _thaw(_evaluation(role="parent", model_sha256=PARENT_A_SHA256))
    parent["baseline_kind"] = "random-noise"
    parent = seal_evaluation(parent)
    candidate = _evaluation(role="candidate", model_sha256=CANDIDATE_SHA256)

    with pytest.raises(TrajectoryGateIntegrityError, match="random, blank, noise"):
        _gate((parent,), candidate)


def test_approximate_trajectory_is_never_mislabeled_as_exact() -> None:
    parent = _evaluation(role="parent", model_sha256=PARENT_A_SHA256)
    candidate = _evaluation(role="candidate", model_sha256=CANDIDATE_SHA256)
    approximate_teacher = _trajectory(
        "teacher-forced", evidence_class="approximate"
    )

    report = _gate((parent,), candidate, teacher=approximate_teacher)

    assert report["accepted"] is True
    assert report["trajectory_gates"]["teacher_forced"]["evidence_class"] == (
        "approximate"
    )
    assert report["evidence_classification"]["overall"] == "approximate"
    assert report["evidence_classification"]["algebraic_equivalence_proven"] is False


def test_report_config_and_input_fingerprints_detect_tampering() -> None:
    parent = _evaluation(role="parent", model_sha256=PARENT_A_SHA256)
    candidate = _evaluation(role="candidate", model_sha256=CANDIDATE_SHA256)
    report = _gate((parent,), candidate)
    tampered = _thaw(report)
    tampered["config"]["min_behavior_retention"] = 0.0

    with pytest.raises(TrajectoryGateIntegrityError, match="config_sha256"):
        verify_gate_report(tampered)
