"""Fail-closed parent-relative promotion gate for folded HF checkpoints.

This module is deliberately a decision-only surface.  It consumes sealed
evaluation documents and sealed, already-computed trajectory comparisons; it
does not import a model framework, run a forward pass, take a gradient, or
modify a checkpoint.  In particular, endpoint/logit agreement is diagnostic
only and can never substitute for autoregressive behavior and generation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal


EVALUATION_SCHEMA = "neural-forge-parent-relative-evaluation-v1"
TRAJECTORY_SCHEMA = "neural-forge-parent-relative-trajectory-v1"
GATE_REPORT_SCHEMA = "neural-forge-trajectory-promotion-gate-v1"

EvidenceClass = Literal["algebraic", "finite-trace", "approximate"]
_EVIDENCE_CLASSES = frozenset(("algebraic", "finite-trace", "approximate"))
_SHA256_LENGTH = 64
_PARENT_BASELINE_KINDS = frozenset(
    ("trained-parent", "donor", "teacher", "reference-parent")
)


class TrajectoryGateError(ValueError):
    """Base class for invalid gate configuration or evidence."""


class TrajectoryGateIntegrityError(TrajectoryGateError):
    """Raised when sealed evidence or a gate report fails verification."""


@dataclass(frozen=True, slots=True)
class TrajectoryGateConfig:
    """Independent retention floors and minimum evidence coverage.

    Retention values are in ``[0, 1]``.  Behavior-like retention is the
    candidate score divided by the weighted parent score.  PPL/NLL retention
    reverses that ratio because lower is better.  All ratios are capped at one
    so improvements are represented as full retention.
    """

    min_behavior_retention: float = 0.80
    min_exact_generation_retention: float = 0.70
    min_choice_retention: float = 0.80
    min_ppl_retention: float | None = 0.50
    min_nll_retention: float | None = None
    min_manual_coherence_retention: float = 0.80
    min_teacher_forced_trajectory_retention: float = 0.75
    min_free_rollout_trajectory_retention: float = 0.60
    min_behavior_samples: int = 1000
    min_exact_generation_samples: int = 1000
    min_choice_samples: int = 250
    min_manual_coherence_samples: int = 32
    min_trajectory_depths: int = 2
    min_teacher_forced_steps: int = 2
    min_free_rollout_steps: int = 4
    trajectory_lower_quantile: float = 0.10
    require_teacher_forced_trajectory: bool = True
    require_free_rollout_trajectory: bool = True

    def __post_init__(self) -> None:
        required_floors = (
            "min_behavior_retention",
            "min_exact_generation_retention",
            "min_choice_retention",
            "min_manual_coherence_retention",
            "min_teacher_forced_trajectory_retention",
            "min_free_rollout_trajectory_retention",
        )
        for name in required_floors:
            _retention_floor(getattr(self, name), name)
        for name in ("min_ppl_retention", "min_nll_retention"):
            value = getattr(self, name)
            if value is not None:
                _retention_floor(value, name)
        if self.min_ppl_retention is None and self.min_nll_retention is None:
            raise TrajectoryGateError(
                "at least one of min_ppl_retention or min_nll_retention is required"
            )
        for name in (
            "min_behavior_samples",
            "min_exact_generation_samples",
            "min_choice_samples",
            "min_manual_coherence_samples",
            "min_trajectory_depths",
            "min_teacher_forced_steps",
            "min_free_rollout_steps",
        ):
            _positive_integer(getattr(self, name), name)
        _retention_floor(
            self.trajectory_lower_quantile, "trajectory_lower_quantile"
        )
        if not isinstance(self.require_teacher_forced_trajectory, bool):
            raise TrajectoryGateError(
                "require_teacher_forced_trajectory must be a bool"
            )
        if not isinstance(self.require_free_rollout_trajectory, bool):
            raise TrajectoryGateError(
                "require_free_rollout_trajectory must be a bool"
            )

    def report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RateMetric:
    value: float
    sample_count: int | None


def canonical_json_sha256(value: Any) -> str:
    """Hash canonical, finite JSON data."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def seal_evaluation(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a deeply immutable evaluation with a content fingerprint."""

    body = _canonical_snapshot(document)
    body.pop("evaluation_sha256", None)
    if body.get("schema") != EVALUATION_SCHEMA:
        raise TrajectoryGateError(f"evaluation schema must be {EVALUATION_SCHEMA!r}")
    body["evaluation_sha256"] = canonical_json_sha256(body)
    return _deep_freeze(body)


def seal_trajectory(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return deeply immutable per-depth trajectory evidence."""

    body = _canonical_snapshot(document)
    body.pop("trajectory_sha256", None)
    if body.get("schema") != TRAJECTORY_SCHEMA:
        raise TrajectoryGateError(f"trajectory schema must be {TRAJECTORY_SCHEMA!r}")
    body["trajectory_sha256"] = canonical_json_sha256(body)
    return _deep_freeze(body)


def verify_gate_report(report: Mapping[str, Any]) -> bool:
    """Verify a previously emitted gate report, raising on any mutation."""

    snapshot = _canonical_snapshot(report)
    supplied = snapshot.pop("report_sha256", None)
    _validate_sha256(supplied, "report_sha256")
    if snapshot.get("schema") != GATE_REPORT_SCHEMA:
        raise TrajectoryGateIntegrityError("gate report schema is invalid")
    config_sha256 = snapshot.get("config_sha256")
    _validate_sha256(config_sha256, "config_sha256")
    if canonical_json_sha256(snapshot.get("config")) != config_sha256:
        raise TrajectoryGateIntegrityError(
            "config_sha256 does not match gate configuration"
        )
    input_sha256 = snapshot.get("input_sha256")
    _validate_sha256(input_sha256, "input_sha256")
    if canonical_json_sha256(snapshot.get("inputs")) != input_sha256:
        raise TrajectoryGateIntegrityError(
            "input_sha256 does not match gate inputs"
        )
    if canonical_json_sha256(snapshot) != supplied:
        raise TrajectoryGateIntegrityError("report_sha256 does not match report")
    return True


def evaluate_trajectory_gate(
    parent_evaluations: Sequence[Mapping[str, Any]],
    candidate_evaluation: Mapping[str, Any],
    *,
    config: TrajectoryGateConfig | None = None,
    parent_weights: Sequence[float] | None = None,
    teacher_forced_trajectory: Mapping[str, Any] | None = None,
    free_rollout_trajectory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate against trained parent baselines only.

    Structural or fingerprint corruption raises
    :class:`TrajectoryGateIntegrityError`.  Missing or insufficient benchmark
    evidence returns a normal rejection report, keeping absence distinct from
    tampering while still failing closed.
    """

    effective_config = config or TrajectoryGateConfig()
    if not isinstance(effective_config, TrajectoryGateConfig):
        raise TrajectoryGateError("config must be a TrajectoryGateConfig")

    parents = tuple(
        _verify_evaluation(value, expected_role="parent")
        for value in parent_evaluations
    )
    if not parents:
        raise TrajectoryGateIntegrityError(
            "at least one sealed trained parent evaluation is required"
        )
    candidate = _verify_evaluation(candidate_evaluation, expected_role="candidate")
    weights = _normalize_weights(parent_weights, len(parents))
    _validate_evaluation_relationships(parents, candidate)

    parent_ids = tuple(str(parent["model_sha256"]) for parent in parents)
    benchmark_sha256 = str(candidate["benchmark_sha256"])
    metric_gates = {
        "behavior": _rate_gate(
            "behavior",
            parents,
            candidate,
            weights,
            effective_config.min_behavior_retention,
            effective_config.min_behavior_samples,
        ),
        "exact_generation": _rate_gate(
            "exact_generation",
            parents,
            candidate,
            weights,
            effective_config.min_exact_generation_retention,
            effective_config.min_exact_generation_samples,
        ),
        "choice": _rate_gate(
            "choice",
            parents,
            candidate,
            weights,
            effective_config.min_choice_retention,
            effective_config.min_choice_samples,
        ),
        "manual_coherence": _rate_gate(
            "manual_coherence",
            parents,
            candidate,
            weights,
            effective_config.min_manual_coherence_retention,
            effective_config.min_manual_coherence_samples,
        ),
    }
    if effective_config.min_ppl_retention is not None:
        metric_gates["ppl"] = _loss_gate(
            "ppl",
            parents,
            candidate,
            weights,
            effective_config.min_ppl_retention,
            geometric_parent_baseline=True,
        )
    if effective_config.min_nll_retention is not None:
        metric_gates["nll"] = _loss_gate(
            "nll",
            parents,
            candidate,
            weights,
            effective_config.min_nll_retention,
            geometric_parent_baseline=False,
        )

    trajectory_gates = {
        "teacher_forced": _trajectory_gate(
            teacher_forced_trajectory,
            mode="teacher-forced",
            required=effective_config.require_teacher_forced_trajectory,
            floor=effective_config.min_teacher_forced_trajectory_retention,
            min_depths=effective_config.min_trajectory_depths,
            min_steps=effective_config.min_teacher_forced_steps,
            lower_quantile=effective_config.trajectory_lower_quantile,
            parent_ids=parent_ids,
            weights=weights,
            candidate_id=str(candidate["model_sha256"]),
            benchmark_sha256=benchmark_sha256,
        ),
        "free_rollout": _trajectory_gate(
            free_rollout_trajectory,
            mode="free-rollout",
            required=effective_config.require_free_rollout_trajectory,
            floor=effective_config.min_free_rollout_trajectory_retention,
            min_depths=effective_config.min_trajectory_depths,
            min_steps=effective_config.min_free_rollout_steps,
            lower_quantile=effective_config.trajectory_lower_quantile,
            parent_ids=parent_ids,
            weights=weights,
            candidate_id=str(candidate["model_sha256"]),
            benchmark_sha256=benchmark_sha256,
        ),
    }

    failed_gates = [
        name for name, gate in metric_gates.items() if not bool(gate["passed"])
    ]
    failed_gates.extend(
        f"trajectory.{name}"
        for name, gate in trajectory_gates.items()
        if not bool(gate["passed"])
    )
    accepted = not failed_gates
    config_report = effective_config.report()
    config_sha256 = canonical_json_sha256(config_report)
    teacher_hash = _sealed_hash_or_none(
        teacher_forced_trajectory, "trajectory_sha256"
    )
    rollout_hash = _sealed_hash_or_none(free_rollout_trajectory, "trajectory_sha256")
    input_fingerprints = {
        "parent_evaluation_sha256": [
            str(parent["evaluation_sha256"]) for parent in parents
        ],
        "candidate_evaluation_sha256": str(candidate["evaluation_sha256"]),
        "teacher_forced_trajectory_sha256": teacher_hash,
        "free_rollout_trajectory_sha256": rollout_hash,
        "normalized_parent_weights": list(weights),
        "benchmark_sha256": benchmark_sha256,
    }
    input_sha256 = canonical_json_sha256(input_fingerprints)
    trajectory_classes = [
        str(gate["evidence_class"])
        for gate in trajectory_gates.values()
        if gate.get("evidence_class") is not None
    ]
    overall_class: EvidenceClass = (
        "approximate" if "approximate" in trajectory_classes else "finite-trace"
    )
    diagnostics = _non_promotional_diagnostics(parents, candidate, weights)

    report: dict[str, Any] = {
        "schema": GATE_REPORT_SCHEMA,
        "accepted": accepted,
        "decision": "promote" if accepted else "reject",
        "failed_gates": failed_gates,
        "metric_gates": metric_gates,
        "trajectory_gates": trajectory_gates,
        "non_promotional_diagnostics": diagnostics,
        "policy": {
            "comparison_baseline": "weighted-trained-parents-only",
            "random_blank_or_noise_baselines_allowed": False,
            "all_required_gates_must_pass": True,
            "endpoint_or_likelihood_only_can_promote": False,
            "endpoint_and_logit_metrics_are_diagnostic_only": True,
            "parent_weight_aggregation": {
                "rates_and_nll": "normalized-weighted-arithmetic-mean",
                "ppl": "normalized-weighted-geometric-mean",
            },
        },
        "zero_gradient_contract": {
            "decision_only": True,
            "training_steps": 0,
            "gradients_used": False,
            "model_forwards_run": 0,
            "checkpoint_mutations": 0,
        },
        "evidence_classification": {
            "overall": overall_class,
            "algebraic_equivalence_proven": False,
            "classification_ceiling": "finite-trace",
            "scope": (
                "fingerprinted finite benchmark items and supplied per-depth "
                "trajectory comparisons only"
            ),
            "behavior_generation_choice_likelihood_manual": "finite-trace",
            "trajectory_inputs": trajectory_classes,
            "approximate_inputs_force_approximate_overall": True,
        },
        "config": config_report,
        "config_sha256": config_sha256,
        "inputs": input_fingerprints,
        "input_sha256": input_sha256,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def _rate_gate(
    name: str,
    parents: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    weights: Sequence[float],
    floor: float,
    minimum_samples: int,
) -> dict[str, Any]:
    parent_metrics = tuple(_extract_rate_metric(parent, name) for parent in parents)
    candidate_metric = _extract_rate_metric(candidate, name)
    missing = [
        index for index, metric in enumerate(parent_metrics) if metric is None
    ]
    if candidate_metric is None or missing:
        reasons = []
        if candidate_metric is None:
            reasons.append("candidate_metric_missing")
        if missing:
            reasons.append("parent_metric_missing")
        return _unavailable_gate(name, floor, reasons)

    assert candidate_metric is not None
    complete_parent_metrics = tuple(
        metric for metric in parent_metrics if metric is not None
    )
    counts = tuple(metric.sample_count for metric in complete_parent_metrics)
    candidate_count = candidate_metric.sample_count
    reasons = []
    if candidate_count is None or candidate_count < minimum_samples:
        reasons.append("candidate_sample_count_below_minimum")
    if any(count is None or count < minimum_samples for count in counts):
        reasons.append("parent_sample_count_below_minimum")
    if candidate_count is not None and any(
        count is not None and count != candidate_count for count in counts
    ):
        reasons.append("candidate_parent_sample_count_mismatch")
    if len(set(counts)) > 1:
        reasons.append("parent_sample_count_mismatch")

    baseline = sum(
        weight * metric.value
        for weight, metric in zip(weights, complete_parent_metrics, strict=True)
    )
    retention = _higher_is_better_retention(candidate_metric.value, baseline)
    if baseline <= 0.0:
        reasons.append("weighted_parent_baseline_has_no_positive_signal")
    if retention is None or retention < floor:
        reasons.append("parent_relative_retention_below_floor")
    passed = not reasons
    return {
        "metric": name,
        "available": True,
        "passed": passed,
        "reasons": reasons,
        "candidate_value": candidate_metric.value,
        "weighted_parent_baseline": baseline,
        "parent_values": [metric.value for metric in complete_parent_metrics],
        "retention": retention,
        "retention_floor": floor,
        "candidate_sample_count": candidate_count,
        "parent_sample_counts": list(counts),
        "minimum_sample_count": minimum_samples,
        "direction": "higher-is-better",
        "evidence_class": "finite-trace",
    }


def _loss_gate(
    name: str,
    parents: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    weights: Sequence[float],
    floor: float,
    *,
    geometric_parent_baseline: bool,
) -> dict[str, Any]:
    parent_values = tuple(_extract_loss_metric(parent, name) for parent in parents)
    candidate_value = _extract_loss_metric(candidate, name)
    missing = [index for index, value in enumerate(parent_values) if value is None]
    if candidate_value is None or missing:
        reasons = []
        if candidate_value is None:
            reasons.append("candidate_metric_missing")
        if missing:
            reasons.append("parent_metric_missing")
        return _unavailable_gate(name, floor, reasons)
    complete_values = tuple(value for value in parent_values if value is not None)
    if geometric_parent_baseline:
        baseline = math.exp(
            sum(
                weight * math.log(value)
                for weight, value in zip(weights, complete_values, strict=True)
            )
        )
        aggregation = "weighted-geometric-mean"
    else:
        baseline = sum(
            weight * value
            for weight, value in zip(weights, complete_values, strict=True)
        )
        aggregation = "weighted-arithmetic-mean"
    retention = _lower_is_better_retention(candidate_value, baseline)
    passed = retention >= floor
    return {
        "metric": name,
        "available": True,
        "passed": passed,
        "reasons": [] if passed else ["parent_relative_retention_below_floor"],
        "candidate_value": candidate_value,
        "weighted_parent_baseline": baseline,
        "parent_values": list(complete_values),
        "retention": retention,
        "retention_floor": floor,
        "direction": "lower-is-better",
        "parent_aggregation": aggregation,
        "evidence_class": "finite-trace",
    }


def _trajectory_gate(
    trajectory: Mapping[str, Any] | None,
    *,
    mode: str,
    required: bool,
    floor: float,
    min_depths: int,
    min_steps: int,
    lower_quantile: float,
    parent_ids: Sequence[str],
    weights: Sequence[float],
    candidate_id: str,
    benchmark_sha256: str,
) -> dict[str, Any]:
    if trajectory is None:
        return {
            "mode": mode,
            "required": required,
            "available": False,
            "passed": not required,
            "reasons": ["required_trajectory_missing"] if required else [],
            "retention": None,
            "retention_floor": floor,
            "evidence_class": None,
        }
    evidence = _verify_trajectory(trajectory)
    if evidence.get("mode") != mode:
        raise TrajectoryGateIntegrityError(
            f"trajectory mode must be {mode!r}"
        )
    if evidence.get("candidate_model_sha256") != candidate_id:
        raise TrajectoryGateIntegrityError(
            f"{mode} trajectory candidate fingerprint does not match evaluation"
        )
    if evidence.get("benchmark_sha256") != benchmark_sha256:
        raise TrajectoryGateIntegrityError(
            f"{mode} trajectory benchmark fingerprint does not match evaluation"
        )
    comparisons = evidence.get("comparisons")
    if not _is_sequence(comparisons):
        raise TrajectoryGateIntegrityError(
            f"{mode} trajectory comparisons must be a sequence"
        )
    by_parent: dict[str, Mapping[str, Any]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise TrajectoryGateIntegrityError(
                f"{mode} trajectory comparison must be a mapping"
            )
        parent_id = comparison.get("parent_model_sha256")
        _validate_sha256(parent_id, f"{mode} trajectory parent_model_sha256")
        if parent_id in by_parent:
            raise TrajectoryGateIntegrityError(
                f"{mode} trajectory has duplicate parent comparison"
            )
        by_parent[str(parent_id)] = comparison
    if set(by_parent) != set(parent_ids):
        raise TrajectoryGateIntegrityError(
            f"{mode} trajectory parent fingerprints do not match evaluations"
        )

    parent_reports = []
    coverage: set[tuple[int, int]] | None = None
    evidence_classes: list[EvidenceClass] = []
    for parent_id in parent_ids:
        comparison = by_parent[parent_id]
        evidence_class = comparison.get("evidence_class")
        if evidence_class not in _EVIDENCE_CLASSES:
            raise TrajectoryGateIntegrityError(
                f"{mode} trajectory evidence_class is invalid"
            )
        evidence_classes.append(evidence_class)
        points = comparison.get("points")
        if not _is_sequence(points) or not points:
            raise TrajectoryGateIntegrityError(
                f"{mode} trajectory points must be a non-empty sequence"
            )
        point_reports = tuple(_trajectory_point(point, mode) for point in points)
        point_keys = {(point["depth"], point["step"]) for point in point_reports}
        if len(point_keys) != len(point_reports):
            raise TrajectoryGateIntegrityError(
                f"{mode} trajectory contains duplicate depth/step points"
            )
        if coverage is None:
            coverage = point_keys
        elif point_keys != coverage:
            raise TrajectoryGateIntegrityError(
                f"{mode} trajectory parent comparisons have different coverage"
            )
        retentions = sorted(float(point["retention"]) for point in point_reports)
        aggregate = _linear_quantile(retentions, lower_quantile)
        parent_reports.append(
            {
                "parent_model_sha256": parent_id,
                "evidence_class": evidence_class,
                "point_count": len(point_reports),
                "depth_count": len({point["depth"] for point in point_reports}),
                "step_count": len({point["step"] for point in point_reports}),
                "minimum_cosine_similarity": min(
                    point["cosine_similarity"] for point in point_reports
                ),
                "maximum_normalized_l2_drift": max(
                    point["normalized_l2_drift"] for point in point_reports
                ),
                "lower_quantile_retention": aggregate,
            }
        )

    assert coverage is not None
    depth_count = len({depth for depth, _ in coverage})
    step_count = len({step for _, step in coverage})
    reasons = []
    if depth_count < min_depths:
        reasons.append("trajectory_depth_coverage_below_minimum")
    if step_count < min_steps:
        reasons.append("trajectory_step_coverage_below_minimum")
    weighted_retention = sum(
        weight * report["lower_quantile_retention"]
        for weight, report in zip(weights, parent_reports, strict=True)
    )
    if weighted_retention < floor:
        reasons.append("parent_relative_trajectory_retention_below_floor")
    overall_class: EvidenceClass = (
        "approximate"
        if "approximate" in evidence_classes
        else "finite-trace"
    )
    return {
        "mode": mode,
        "required": required,
        "available": True,
        "passed": not reasons,
        "reasons": reasons,
        "retention": weighted_retention,
        "retention_floor": floor,
        "lower_quantile": lower_quantile,
        "depth_count": depth_count,
        "minimum_depth_count": min_depths,
        "step_count": step_count,
        "minimum_step_count": min_steps,
        "parent_comparisons": parent_reports,
        "evidence_class": overall_class,
        "algebraically_exact": False,
        "exactness_scope": (
            "fingerprinted supplied candidate-to-parent trajectory points only"
        ),
        "trajectory_sha256": str(evidence["trajectory_sha256"]),
    }


def _trajectory_point(point: Any, mode: str) -> dict[str, Any]:
    if not isinstance(point, Mapping):
        raise TrajectoryGateIntegrityError(
            f"{mode} trajectory point must be a mapping"
        )
    depth = _nonnegative_integer(point.get("depth"), f"{mode} trajectory depth")
    step = _nonnegative_integer(point.get("step"), f"{mode} trajectory step")
    cosine = _finite_number(
        point.get("cosine_similarity"), f"{mode} cosine_similarity"
    )
    drift = _finite_number(
        point.get("normalized_l2_drift"), f"{mode} normalized_l2_drift"
    )
    if not -1.0 <= cosine <= 1.0:
        raise TrajectoryGateIntegrityError(
            f"{mode} cosine_similarity must be in [-1, 1]"
        )
    if drift < 0.0:
        raise TrajectoryGateIntegrityError(
            f"{mode} normalized_l2_drift must be non-negative"
        )
    retention = max(0.0, cosine) * math.exp(-drift)
    return {
        "depth": depth,
        "step": step,
        "cosine_similarity": cosine,
        "normalized_l2_drift": drift,
        "retention": retention,
    }


def _verify_evaluation(
    document: Mapping[str, Any], *, expected_role: str
) -> dict[str, Any]:
    value = _verify_sealed(
        document,
        hash_key="evaluation_sha256",
        schema=EVALUATION_SCHEMA,
        label="evaluation",
    )
    if value.get("role") != expected_role:
        raise TrajectoryGateIntegrityError(
            f"evaluation role must be {expected_role!r}"
        )
    _validate_sha256(value.get("model_sha256"), "model_sha256")
    _validate_sha256(value.get("benchmark_sha256"), "benchmark_sha256")
    if not isinstance(value.get("metrics"), Mapping):
        raise TrajectoryGateIntegrityError("evaluation metrics must be a mapping")
    if expected_role == "parent":
        baseline_kind = value.get("baseline_kind")
        if baseline_kind not in _PARENT_BASELINE_KINDS:
            raise TrajectoryGateIntegrityError(
                "parent baseline_kind must identify a trained parent/donor/teacher; "
                "random, blank, noise, and untrained baselines are forbidden"
            )
    return value


def _verify_trajectory(document: Mapping[str, Any]) -> dict[str, Any]:
    value = _verify_sealed(
        document,
        hash_key="trajectory_sha256",
        schema=TRAJECTORY_SCHEMA,
        label="trajectory",
    )
    _validate_sha256(value.get("candidate_model_sha256"), "candidate_model_sha256")
    _validate_sha256(value.get("benchmark_sha256"), "benchmark_sha256")
    return value


def _verify_sealed(
    document: Mapping[str, Any],
    *,
    hash_key: str,
    schema: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise TrajectoryGateIntegrityError(f"{label} must be a mapping")
    snapshot = _canonical_snapshot(document, integrity=True)
    supplied = snapshot.pop(hash_key, None)
    _validate_sha256(supplied, hash_key)
    if snapshot.get("schema") != schema:
        raise TrajectoryGateIntegrityError(f"{label} schema is invalid")
    if canonical_json_sha256(snapshot) != supplied:
        raise TrajectoryGateIntegrityError(f"{hash_key} does not match {label}")
    snapshot[hash_key] = supplied
    return snapshot


def _validate_evaluation_relationships(
    parents: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> None:
    benchmark = candidate["benchmark_sha256"]
    parent_ids = [str(parent["model_sha256"]) for parent in parents]
    if len(set(parent_ids)) != len(parent_ids):
        raise TrajectoryGateIntegrityError("duplicate parent model fingerprints")
    for parent in parents:
        if parent["benchmark_sha256"] != benchmark:
            raise TrajectoryGateIntegrityError(
                "candidate and every parent must use the same benchmark fingerprint"
            )


def _extract_rate_metric(
    evaluation: Mapping[str, Any], name: str
) -> _RateMetric | None:
    metrics = evaluation["metrics"]
    value = metrics.get(name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        score = _optional_finite_number(value)
        if score is None or not 0.0 <= score <= 1.0:
            return None
        return _RateMetric(score, None)

    numerator = _first_present(
        value,
        ("correct", "coherent", "passed", "pass_count", "successes"),
    )
    denominator = _first_present(value, ("total", "count", "sample_count"))
    sample_count: int | None = None
    score: float | None = None
    if denominator is not None:
        try:
            sample_count = _positive_integer(denominator, f"{name} sample count")
        except TrajectoryGateError:
            return None
    if numerator is not None and sample_count is not None:
        numeric_numerator = _optional_finite_number(numerator)
        if numeric_numerator is None or not 0.0 <= numeric_numerator <= sample_count:
            return None
        score = numeric_numerator / sample_count
    else:
        score = _optional_finite_number(
            _first_present(value, ("rate", "score", "accuracy", "mean"))
        )
    if score is None or not 0.0 <= score <= 1.0:
        return None
    return _RateMetric(score, sample_count)


def _extract_loss_metric(evaluation: Mapping[str, Any], name: str) -> float | None:
    value = evaluation["metrics"].get(name)
    if isinstance(value, Mapping):
        value = _first_present(value, ("value", "mean", "score"))
    result = _optional_finite_number(value)
    if result is None:
        return None
    if name == "ppl" and result <= 0.0:
        return None
    if name == "nll" and result < 0.0:
        return None
    return result


def _non_promotional_diagnostics(
    parents: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    weights: Sequence[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "promotional": False,
        "reason": (
            "endpoint/logit fit is finite anchor evidence and cannot establish "
            "autoregressive behavior"
        ),
    }
    for name in ("endpoint_cosine", "logit_cosine"):
        candidate_value = _extract_optional_diagnostic(candidate, name)
        parent_values = tuple(
            _extract_optional_diagnostic(parent, name) for parent in parents
        )
        available_parent_values = all(value is not None for value in parent_values)
        result[name] = {
            "candidate": candidate_value,
            "weighted_parent_baseline": (
                sum(
                    weight * float(value)
                    for weight, value in zip(weights, parent_values, strict=True)
                )
                if available_parent_values
                else None
            ),
            "used_for_promotion": False,
        }
    return result


def _extract_optional_diagnostic(
    evaluation: Mapping[str, Any], name: str
) -> float | None:
    value = evaluation["metrics"].get(name)
    if isinstance(value, Mapping):
        value = _first_present(value, ("value", "mean", "score"))
    return _optional_finite_number(value)


def _normalize_weights(
    weights: Sequence[float] | None, parent_count: int
) -> tuple[float, ...]:
    if weights is None:
        return tuple(1.0 / parent_count for _ in range(parent_count))
    values = tuple(_finite_number(value, "parent weight") for value in weights)
    if len(values) != parent_count:
        raise TrajectoryGateIntegrityError(
            "parent_weights length must match parent_evaluations"
        )
    if any(value <= 0.0 for value in values):
        raise TrajectoryGateIntegrityError("every parent weight must be positive")
    total = sum(values)
    return tuple(value / total for value in values)


def _unavailable_gate(
    name: str, floor: float, reasons: Sequence[str]
) -> dict[str, Any]:
    return {
        "metric": name,
        "available": False,
        "passed": False,
        "reasons": list(reasons) or ["required_metric_missing"],
        "candidate_value": None,
        "weighted_parent_baseline": None,
        "retention": None,
        "retention_floor": floor,
        "evidence_class": "finite-trace",
    }


def _higher_is_better_retention(candidate: float, baseline: float) -> float | None:
    if baseline <= 0.0:
        return None
    return min(1.0, candidate / baseline)


def _lower_is_better_retention(candidate: float, baseline: float) -> float:
    if candidate == 0.0:
        return 1.0
    return min(1.0, baseline / candidate)


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _sealed_hash_or_none(
    document: Mapping[str, Any] | None, hash_key: str
) -> str | None:
    if document is None:
        return None
    value = document.get(hash_key)
    _validate_sha256(value, hash_key)
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TrajectoryGateError("value is not finite canonical JSON data") from exc


def _canonical_snapshot(
    value: Any, *, integrity: bool = False
) -> dict[str, Any]:
    try:
        result = json.loads(_canonical_json_bytes(value))
    except TrajectoryGateError as exc:
        if integrity:
            raise TrajectoryGateIntegrityError(str(exc)) from exc
        raise
    if not isinstance(result, dict):
        error = "sealed document must be a JSON object"
        if integrity:
            raise TrajectoryGateIntegrityError(error)
        raise TrajectoryGateError(error)
    return result


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrajectoryGateError("canonical JSON mapping keys must be strings")
            result[key] = _thaw_json(item)
        return result
    if _is_sequence(value):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TrajectoryGateError("canonical JSON numbers must be finite")
        return value
    raise TrajectoryGateError(
        f"unsupported canonical JSON value type: {type(value).__name__}"
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if _is_sequence(value):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrajectoryGateIntegrityError(f"{name} must be a lowercase SHA-256")
    return value


def _retention_floor(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise TrajectoryGateError(f"{name} must be in [0, 1]")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrajectoryGateError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrajectoryGateIntegrityError(
            f"{name} must be a non-negative integer"
        )
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryGateError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TrajectoryGateError(f"{name} must be a finite number")
    return result


def _optional_finite_number(value: Any) -> float | None:
    try:
        return _finite_number(value, "metric")
    except TrajectoryGateError:
        return None


__all__ = [
    "EVALUATION_SCHEMA",
    "GATE_REPORT_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "TrajectoryGateConfig",
    "TrajectoryGateError",
    "TrajectoryGateIntegrityError",
    "canonical_json_sha256",
    "evaluate_trajectory_gate",
    "seal_evaluation",
    "seal_trajectory",
    "verify_gate_report",
]
