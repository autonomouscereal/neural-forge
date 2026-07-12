"""Deterministic contiguous donor-layer span planning.

The planner in this module has no model-framework dependency.  It solves the
ordered partition problem directly: every donor layer is assigned exactly
once, in order, to exactly ``target_layer_count`` nonempty contiguous spans.
Caller-provided costs remain authoritative; the only optional objective term
is the explicit penalty for putting multiple critical layers in one span.

All public and reported ranges use Python's half-open ``[start, end)``
convention.  Cost matrices can use either half-open end columns or conventional
inclusive-end columns, selected explicitly or inferred from their dimensions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


REPORT_SCHEMA_VERSION = 1

__all__ = [
    "InfeasibleSpanPlanError",
    "SpanPlanInfeasibleError",
    "SpanPlanningError",
    "plan_contiguous_spans",
    "plan_layer_spans",
]


class SpanPlanningError(ValueError):
    """Base class for invalid span-planning requests."""


class SpanPlanInfeasibleError(SpanPlanningError):
    """Raised when valid constraints and costs admit no complete partition."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


# Readable compatibility alias for callers that lead with the failure mode.
InfeasibleSpanPlanError = SpanPlanInfeasibleError


@dataclass(frozen=True)
class _State:
    total_cost: float
    base_cost: float
    penalty_cost: float
    boundaries: tuple[int, ...]


@dataclass(frozen=True)
class _CostTable:
    values: dict[tuple[int, int], float]
    provenance: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise SpanPlanningError(f"{name} must be an integer, not bool.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpanPlanningError(f"{name} must be an integer.") from exc
    if result != value:
        raise SpanPlanningError(f"{name} must be an integer.")
    if positive and result <= 0:
        raise SpanPlanningError(f"{name} must be positive.")
    return result


def _optional_cost(value: Any, location: str) -> float | None:
    """Return a finite cost, or ``None`` for an unavailable span."""

    if value is None:
        return None
    if isinstance(value, (bool, str, bytes)):
        raise SpanPlanningError(f"Span cost at {location} must be numeric or None.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpanPlanningError(
            f"Span cost at {location} must be numeric or None."
        ) from exc
    if math.isnan(result) or result == float("-inf"):
        raise SpanPlanningError(
            f"Span cost at {location} must be finite, None, or positive infinity."
        )
    if result == float("inf"):
        return None
    return 0.0 if result == 0.0 else result


def _penalty(value: Any) -> float | None:
    if value is None:
        return None
    result = _optional_cost(value, "critical_layer_sharing_penalty")
    if result is None or result < 0.0:
        raise SpanPlanningError(
            "critical_layer_sharing_penalty must be finite and non-negative "
            "when provided."
        )
    return result


def _critical_layers(
    values: Iterable[int] | None, donor_layer_count: int
) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise SpanPlanningError("critical_layers must be an iterable of indices.")
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise SpanPlanningError(
            "critical_layers must be an iterable of indices."
        ) from exc

    normalized = []
    for position, value in enumerate(raw_values):
        index = _exact_int(value, f"critical_layers[{position}]")
        if not 0 <= index < donor_layer_count:
            raise SpanPlanningError(
                f"critical layer index {index} is outside [0, {donor_layer_count})."
            )
        normalized.append(index)
    return tuple(sorted(set(normalized)))


def _layout_name(value: str) -> str:
    aliases = {
        "auto": "auto",
        "half-open": "half-open",
        "end-exclusive": "half-open",
        "inclusive-end": "inclusive-end",
        "end-inclusive": "inclusive-end",
    }
    try:
        return aliases[str(value).strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(aliases))
        raise SpanPlanningError(f"span_cost_layout must be one of: {choices}.") from exc


def _tie_break_name(value: str) -> str:
    aliases = {
        "earliest": "earliest-boundaries",
        "earliest-boundary": "earliest-boundaries",
        "earliest-boundaries": "earliest-boundaries",
        "lexicographically-earliest": "earliest-boundaries",
        "latest": "latest-boundaries",
        "latest-boundary": "latest-boundaries",
        "latest-boundaries": "latest-boundaries",
        "lexicographically-latest": "latest-boundaries",
    }
    try:
        return aliases[str(value).strip().lower()]
    except KeyError as exc:
        raise SpanPlanningError(
            "tie_break must be 'earliest-boundaries' or 'latest-boundaries'."
        ) from exc


def _mapping_entries(
    span_costs: Mapping[Any, Any],
) -> list[tuple[int, int, float | None]]:
    entries = []
    for raw_key, raw_value in span_costs.items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
            or isinstance(raw_key[0], bool)
            or isinstance(raw_key[1], bool)
        ):
            raise SpanPlanningError(
                "Span-cost mapping keys must be (start, end) integer tuples."
            )
        start = _exact_int(raw_key[0], f"span cost key {raw_key!r} start")
        end = _exact_int(raw_key[1], f"span cost key {raw_key!r} end")
        entries.append(
            (start, end, _optional_cost(raw_value, f"mapping key {raw_key!r}"))
        )
    return entries


def _infer_mapping_layout(
    entries: Iterable[tuple[int, int, float | None]], donor_layer_count: int
) -> str:
    available = [(start, end) for start, end, cost in entries if cost is not None]
    diagonal_evidence = any(
        start == end and 0 <= start < donor_layer_count for start, end in available
    )
    exclusive_evidence = any(end == donor_layer_count for _start, end in available)
    if diagonal_evidence and exclusive_evidence:
        raise SpanPlanningError(
            "Cannot infer span-cost mapping layout: available entries contain both "
            "inclusive-end diagonal keys and a half-open end equal to the donor count. "
            "Set span_cost_layout explicitly."
        )
    return "inclusive-end" if diagonal_evidence else "half-open"


def _costs_from_mapping(
    span_costs: Mapping[Any, Any],
    donor_layer_count: int,
    requested_layout: str,
) -> tuple[dict[tuple[int, int], float], str, dict[str, Any]]:
    entries = _mapping_entries(span_costs)
    resolved_layout = (
        _infer_mapping_layout(entries, donor_layer_count)
        if requested_layout == "auto"
        else requested_layout
    )
    values: dict[tuple[int, int], float] = {}
    unavailable_count = 0
    for raw_start, raw_end, cost in entries:
        if cost is None:
            unavailable_count += 1
            continue
        start = raw_start
        end = raw_end if resolved_layout == "half-open" else raw_end + 1
        if not 0 <= start < end <= donor_layer_count:
            convention = (
                "[start, end)" if resolved_layout == "half-open" else "[start, end]"
            )
            raise SpanPlanningError(
                f"Span-cost key ({raw_start}, {raw_end}) is outside the valid "
                f"{convention} donor-layer range."
            )
        key = (start, end)
        if key in values:
            raise SpanPlanningError(
                f"Multiple span-cost entries normalize to half-open span {key}."
            )
        values[key] = cost
    metadata = {
        "source_type": "mapping",
        "source_entry_count": len(entries),
        "source_available_entry_count": len(values),
        "source_unavailable_entry_count": unavailable_count,
    }
    return values, resolved_layout, metadata


def _matrix_rows(span_costs: Any, donor_layer_count: int) -> list[Any]:
    if isinstance(span_costs, (str, bytes)):
        raise SpanPlanningError(
            "span_costs must be a mapping or two-dimensional matrix."
        )
    try:
        rows = list(span_costs)
    except TypeError as exc:
        raise SpanPlanningError(
            "span_costs must be a mapping or two-dimensional matrix."
        ) from exc
    if len(rows) != donor_layer_count:
        raise SpanPlanningError(
            "A span-cost matrix must have one row per donor layer; "
            f"expected {donor_layer_count}, got {len(rows)}."
        )
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__len__"):
            raise SpanPlanningError(f"Span-cost matrix row {index} is not indexable.")
    return rows


def _resolve_matrix_layout(
    rows: list[Any], donor_layer_count: int, requested_layout: str
) -> str:
    if requested_layout != "auto":
        return requested_layout
    row_lengths = [len(row) for row in rows]
    if all(length >= donor_layer_count + 1 for length in row_lengths):
        return "half-open"
    if all(length >= donor_layer_count for length in row_lengths):
        return "inclusive-end"
    raise SpanPlanningError(
        "Cannot infer span-cost matrix layout. Use at least donor_layer_count "
        "columns for inclusive-end costs or donor_layer_count + 1 columns for "
        "half-open costs, or set span_cost_layout explicitly."
    )


def _costs_from_matrix(
    span_costs: Any,
    donor_layer_count: int,
    requested_layout: str,
) -> tuple[dict[tuple[int, int], float], str, dict[str, Any]]:
    rows = _matrix_rows(span_costs, donor_layer_count)
    resolved_layout = _resolve_matrix_layout(rows, donor_layer_count, requested_layout)
    values: dict[tuple[int, int], float] = {}
    unavailable_count = 0
    for start in range(donor_layer_count):
        for end in range(start + 1, donor_layer_count + 1):
            column = end if resolved_layout == "half-open" else end - 1
            row = rows[start]
            if column >= len(row):
                unavailable_count += 1
                continue
            cost = _optional_cost(row[column], f"matrix[{start}][{column}]")
            if cost is None:
                unavailable_count += 1
            else:
                values[(start, end)] = cost
    metadata = {
        "source_type": "matrix",
        "matrix_row_count": len(rows),
        "matrix_row_lengths": [len(row) for row in rows],
        "source_available_entry_count": len(values),
        "source_unavailable_entry_count": unavailable_count,
    }
    return values, resolved_layout, metadata


def _cost_table(
    span_costs: Any,
    donor_layer_count: int,
    minimum: int,
    maximum: int,
    requested_layout: str,
) -> _CostTable:
    if isinstance(span_costs, Mapping):
        values, resolved_layout, metadata = _costs_from_mapping(
            span_costs, donor_layer_count, requested_layout
        )
    else:
        values, resolved_layout, metadata = _costs_from_matrix(
            span_costs, donor_layer_count, requested_layout
        )

    candidate_records = []
    candidate_values: dict[tuple[int, int], float] = {}
    for start in range(donor_layer_count):
        last_end = min(donor_layer_count, start + maximum)
        for end in range(start + minimum, last_end + 1):
            cost = values.get((start, end))
            candidate_records.append([start, end, cost])
            if cost is not None:
                candidate_values[(start, end)] = cost

    metadata.update(
        {
            "requested_layout": requested_layout,
            "resolved_layout": resolved_layout,
            "canonical_interval_convention": "half-open [start, end)",
            "candidate_span_count": len(candidate_records),
            "available_candidate_span_count": len(candidate_values),
            "missing_candidate_span_count": len(candidate_records)
            - len(candidate_values),
            "candidate_costs_sha256": _json_sha256(candidate_records),
            "positive_infinity_and_none_mean_unavailable": True,
        }
    )
    return _CostTable(values=candidate_values, provenance=metadata)


def _base_report(
    donor_layer_count: int,
    target_layer_count: int,
    minimum: int,
    maximum: int,
    critical_layers: tuple[int, ...],
    sharing_penalty: float | None,
    tie_break: str,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation": "deterministic-contiguous-layer-span-planning",
        "inputs": {
            "donor_layer_count": donor_layer_count,
            "target_layer_count": target_layer_count,
        },
        "constraints": {
            "exact_target_span_count": target_layer_count,
            "nonempty_spans": True,
            "contiguous_spans": True,
            "ordered_spans": True,
            "all_donor_layers_required": True,
            "minimum_span_length": minimum,
            "maximum_span_length": maximum,
            "critical_layers": list(critical_layers),
            "critical_layers_may_share": sharing_penalty is not None,
            "critical_layer_sharing_mode": (
                "hard-constraint" if sharing_penalty is None else "penalized"
            ),
            "critical_layer_sharing_penalty_per_pair": sharing_penalty,
        },
        "determinism": {
            "tie_break": tie_break,
            "tie_definition": "exact equality of finite objective costs",
            "boundary_order": (
                "lexicographically-smallest internal end boundaries"
                if tie_break == "earliest-boundaries"
                else "lexicographically-largest internal end boundaries"
            ),
        },
    }


def _raise_infeasible(
    message: str,
    reason: str,
    report: dict[str, Any],
    **details: Any,
) -> None:
    failure_report = {
        **report,
        "status": "infeasible",
        "infeasibility": {"reason": reason, **details},
    }
    raise SpanPlanInfeasibleError(message, failure_report)


def _candidate_is_better(
    candidate: _State, incumbent: _State | None, tie_break: str
) -> tuple[bool, bool]:
    if incumbent is None:
        return True, False
    if candidate.total_cost < incumbent.total_cost:
        return True, False
    if candidate.total_cost > incumbent.total_cost:
        return False, False
    if tie_break == "earliest-boundaries":
        return candidate.boundaries < incumbent.boundaries, True
    return candidate.boundaries > incumbent.boundaries, True


def plan_contiguous_spans(
    donor_layer_count: int,
    target_layer_count: int,
    span_costs: Mapping[tuple[int, int], Any] | Any,
    *,
    min_span_length: int = 1,
    max_span_length: int | None = None,
    critical_layers: Iterable[int] | None = None,
    critical_layer_sharing_penalty: float | None = None,
    span_cost_layout: str = "auto",
    tie_break: str = "earliest-boundaries",
) -> dict[str, Any]:
    """Find the minimum-cost exact contiguous partition of donor layers.

    Args:
        donor_layer_count: Number of ordered donor layers, indexed from zero.
        target_layer_count: Exact number of target spans to produce.
        span_costs: Sparse mapping or two-dimensional matrix of span costs.
            Mapping keys are endpoint pairs in ``span_cost_layout``.  A
            half-open matrix stores ``cost[start][end]`` and therefore has at
            least ``donor_layer_count + 1`` columns.  An inclusive-end matrix
            stores ``cost[start][end_inclusive]`` and has at least
            ``donor_layer_count`` columns.  Missing mapping entries, ``None``,
            and positive infinity make a span unavailable.
        min_span_length: Inclusive minimum donor layers per target span.
        max_span_length: Inclusive maximum, or the donor count when omitted.
        critical_layers: Donor indices that should occupy distinct spans.
        critical_layer_sharing_penalty: If omitted, sharing a span is forbidden.
            If provided, sharing is allowed and this cost is charged for every
            unordered pair of critical layers in the same span.
        span_cost_layout: ``"half-open"``, ``"inclusive-end"``, or ``"auto"``.
        tie_break: Boundary ordering used only when total costs are exactly equal.

    Returns:
        A dictionary containing selected spans, objective decomposition, and a
        complete deterministic provenance report.

    Raises:
        SpanPlanningError: An input or cost source is malformed.
        SpanPlanInfeasibleError: No partition satisfies all hard constraints.
    """

    donor_count = _exact_int(donor_layer_count, "donor_layer_count", positive=True)
    target_count = _exact_int(target_layer_count, "target_layer_count", positive=True)
    minimum = _exact_int(min_span_length, "min_span_length", positive=True)
    maximum = (
        donor_count
        if max_span_length is None
        else _exact_int(max_span_length, "max_span_length", positive=True)
    )
    if minimum > maximum:
        raise SpanPlanningError(
            "min_span_length cannot be greater than max_span_length."
        )

    critical = _critical_layers(critical_layers, donor_count)
    sharing_penalty = _penalty(critical_layer_sharing_penalty)
    requested_layout = _layout_name(span_cost_layout)
    resolved_tie_break = _tie_break_name(tie_break)
    report = _base_report(
        donor_count,
        target_count,
        minimum,
        maximum,
        critical,
        sharing_penalty,
        resolved_tie_break,
    )

    if target_count > donor_count:
        _raise_infeasible(
            "Cannot create more nonempty target spans than donor layers.",
            "target_span_count_exceeds_donor_layer_count",
            report,
            donor_layer_count=donor_count,
            target_layer_count=target_count,
        )
    minimum_covered = target_count * minimum
    maximum_covered = target_count * maximum
    if not minimum_covered <= donor_count <= maximum_covered:
        _raise_infeasible(
            "Span-length bounds cannot cover every donor layer with the exact "
            "target span count.",
            "span_length_capacity",
            report,
            minimum_coverable_layers=minimum_covered,
            maximum_coverable_layers=maximum_covered,
            required_layers=donor_count,
        )
    if sharing_penalty is None and len(critical) > target_count:
        _raise_infeasible(
            "More critical donor layers require distinct spans than target spans exist.",
            "critical_layer_pigeonhole",
            report,
            critical_layer_count=len(critical),
            target_layer_count=target_count,
        )

    costs = _cost_table(
        span_costs,
        donor_count,
        minimum,
        maximum,
        requested_layout,
    )
    report["cost_provenance"] = costs.provenance

    critical_prefix = [0] * (donor_count + 1)
    critical_set = set(critical)
    for index in range(donor_count):
        critical_prefix[index + 1] = critical_prefix[index] + int(index in critical_set)

    span_constraints: dict[tuple[int, int], tuple[int, int, float | None]] = {}
    hard_forbidden_candidate_count = 0
    for start in range(donor_count):
        for end in range(start + minimum, min(donor_count, start + maximum) + 1):
            critical_count = critical_prefix[end] - critical_prefix[start]
            pair_count = critical_count * (critical_count - 1) // 2
            if pair_count and sharing_penalty is None:
                penalty_cost = None
                hard_forbidden_candidate_count += 1
            else:
                penalty_cost = float(pair_count) * float(sharing_penalty or 0.0)
            span_constraints[(start, end)] = (
                critical_count,
                pair_count,
                penalty_cost,
            )

    stats = {
        "algorithm": "dynamic-programming-exact-contiguous-partition",
        "state_dimensions": [target_count + 1, donor_count + 1],
        "reachable_state_count": 1,
        "transition_attempt_count": 0,
        "transition_missing_cost_count": 0,
        "transition_hard_critical_rejection_count": 0,
        "feasible_transition_count": 0,
        "state_update_count": 0,
        "equal_cost_comparison_count": 0,
        "equal_cost_replacement_count": 0,
        "hard_forbidden_candidate_span_count": hard_forbidden_candidate_count,
    }

    previous: dict[int, _State] = {
        0: _State(0.0, 0.0, 0.0, ()),
    }
    for span_number in range(1, target_count + 1):
        current: dict[int, _State] = {}
        remaining_spans = target_count - span_number
        smallest_end = span_number * minimum
        largest_end = min(donor_count, span_number * maximum)
        for end in range(smallest_end, largest_end + 1):
            remaining_layers = donor_count - end
            if not (
                remaining_spans * minimum
                <= remaining_layers
                <= remaining_spans * maximum
            ):
                continue
            first_start = max((span_number - 1) * minimum, end - maximum)
            last_start = end - minimum
            for start in range(first_start, last_start + 1):
                prior = previous.get(start)
                if prior is None:
                    continue
                stats["transition_attempt_count"] += 1
                base_cost = costs.values.get((start, end))
                if base_cost is None:
                    stats["transition_missing_cost_count"] += 1
                    continue
                _critical_count, _pair_count, penalty_cost = span_constraints[
                    (start, end)
                ]
                if penalty_cost is None:
                    stats["transition_hard_critical_rejection_count"] += 1
                    continue
                stats["feasible_transition_count"] += 1
                candidate_base = prior.base_cost + base_cost
                candidate_penalty = prior.penalty_cost + penalty_cost
                candidate = _State(
                    total_cost=candidate_base + candidate_penalty,
                    base_cost=candidate_base,
                    penalty_cost=candidate_penalty,
                    boundaries=prior.boundaries + (end,),
                )
                incumbent = current.get(end)
                better, equal_cost = _candidate_is_better(
                    candidate, incumbent, resolved_tie_break
                )
                if equal_cost:
                    stats["equal_cost_comparison_count"] += 1
                if better:
                    current[end] = candidate
                    stats["state_update_count"] += 1
                    if equal_cost:
                        stats["equal_cost_replacement_count"] += 1
        previous = current
        stats["reachable_state_count"] += len(current)

    selected = previous.get(donor_count)
    report["dynamic_programming"] = stats
    if selected is None:
        report["hashes"] = {
            "input_sha256": _json_sha256(
                {
                    "inputs": report["inputs"],
                    "constraints": report["constraints"],
                    "determinism": report["determinism"],
                    "candidate_costs_sha256": costs.provenance[
                        "candidate_costs_sha256"
                    ],
                }
            )
        }
        _raise_infeasible(
            "No complete partition satisfies the span costs and hard constraints.",
            "no_feasible_partition",
            report,
            available_candidate_span_count=costs.provenance[
                "available_candidate_span_count"
            ],
            missing_candidate_span_count=costs.provenance[
                "missing_candidate_span_count"
            ],
        )

    boundaries = (0, *selected.boundaries)
    spans = []
    total_sharing_pairs = 0
    for target_layer, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        critical_in_span = [index for index in critical if start <= index < end]
        pair_count = len(critical_in_span) * (len(critical_in_span) - 1) // 2
        penalty_cost = float(pair_count) * float(sharing_penalty or 0.0)
        total_sharing_pairs += pair_count
        base_cost = costs.values[(start, end)]
        spans.append(
            {
                "target_layer": target_layer,
                "start": start,
                "end": end,
                "donor_start": start,
                "donor_end_exclusive": end,
                "donor_end_inclusive": end - 1,
                "length": end - start,
                "donor_layers": list(range(start, end)),
                "critical_layers": critical_in_span,
                "critical_layer_count": len(critical_in_span),
                "critical_sharing_pair_count": pair_count,
                "base_cost": base_cost,
                "critical_sharing_penalty_cost": penalty_cost,
                "effective_cost": base_cost + penalty_cost,
            }
        )

    internal_boundaries = list(selected.boundaries[:-1])
    coverage = {
        "complete": boundaries[0] == 0 and boundaries[-1] == donor_count,
        "ordered": all(left < right for left, right in zip(boundaries, boundaries[1:])),
        "contiguous": all(
            spans[index]["end"] == spans[index + 1]["start"]
            for index in range(len(spans) - 1)
        ),
        "non_overlapping": True,
        "covered_donor_layer_count": sum(item["length"] for item in spans),
        "target_span_count": len(spans),
        "exact_target_span_count_satisfied": len(spans) == target_count,
    }
    critical_report = {
        "required_layers": list(critical),
        "layer_to_target_span": {
            str(layer): next(
                item["target_layer"]
                for item in spans
                if item["start"] <= layer < item["end"]
            )
            for layer in critical
        },
        "sharing_pair_count": total_sharing_pairs,
        "distinct_span_constraint_satisfied": total_sharing_pairs == 0,
        "hard_constraint": sharing_penalty is None,
        "penalty_per_pair": sharing_penalty,
        "penalty_cost": selected.penalty_cost,
    }
    objective = {
        "sense": "minimize",
        "base_span_cost": selected.base_cost,
        "critical_sharing_penalty_cost": selected.penalty_cost,
        "total_cost": selected.total_cost,
        "penalty_unit": "unordered critical-layer pair sharing one span",
    }
    partition = {
        "boundaries": list(boundaries),
        "internal_end_boundaries": internal_boundaries,
        "span_ranges": [[item["start"], item["end"]] for item in spans],
        "spans": spans,
    }
    input_hash = _json_sha256(
        {
            "inputs": report["inputs"],
            "constraints": report["constraints"],
            "determinism": report["determinism"],
            "candidate_costs_sha256": costs.provenance["candidate_costs_sha256"],
        }
    )
    plan_hash = _json_sha256(
        {
            "boundaries": list(boundaries),
            "base_costs": [item["base_cost"] for item in spans],
            "critical_sharing_pairs": [
                item["critical_sharing_pair_count"] for item in spans
            ],
            "objective": objective,
        }
    )
    report.update(
        {
            "status": "feasible",
            "method": {
                "algorithm": "dynamic-programming",
                "optimality": "global over all available valid contiguous spans",
                "objective_additive_by_span": True,
                "torch_dependency": False,
            },
            "objective": objective,
            "partition": partition,
            "coverage": coverage,
            "critical_layer_provenance": critical_report,
            "hashes": {
                "algorithm": "sha256",
                "input_sha256": input_hash,
                "candidate_costs_sha256": costs.provenance["candidate_costs_sha256"],
                "plan_sha256": plan_hash,
            },
        }
    )

    return {
        "spans": spans,
        "boundaries": list(boundaries),
        "span_ranges": partition["span_ranges"],
        "donor_layer_spans": [item["donor_layers"] for item in spans],
        "base_cost": selected.base_cost,
        "critical_sharing_penalty_cost": selected.penalty_cost,
        "total_cost": selected.total_cost,
        "report": report,
    }


def plan_layer_spans(
    donor_layer_count: int,
    target_layer_count: int,
    span_costs: Mapping[tuple[int, int], Any] | Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias for :func:`plan_contiguous_spans` with the same contract."""

    return plan_contiguous_spans(
        donor_layer_count,
        target_layer_count,
        span_costs,
        **kwargs,
    )
