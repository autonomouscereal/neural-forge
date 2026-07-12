from __future__ import annotations

import ast
import itertools
import json
from copy import deepcopy
from pathlib import Path

import pytest

from neural_forge.hf_span_planner import (
    SpanPlanInfeasibleError,
    SpanPlanningError,
    plan_contiguous_spans,
    plan_layer_spans,
)


def _all_half_open_costs(layer_count: int, value: float = 0.0) -> dict:
    return {
        (start, end): value
        for start in range(layer_count)
        for end in range(start + 1, layer_count + 1)
    }


def test_dynamic_programming_finds_global_optimum_not_greedy_first_span() -> None:
    costs = {
        (0, 1): 0.0,
        (1, 5): 100.0,
        (0, 2): 5.0,
        (2, 5): 0.0,
    }

    result = plan_contiguous_spans(5, 2, costs)

    assert result["boundaries"] == [0, 2, 5]
    assert result["donor_layer_spans"] == [[0, 1], [2, 3, 4]]
    assert result["total_cost"] == pytest.approx(5.0)


def test_planner_matches_exhaustive_partition_search() -> None:
    layer_count = 7
    target_count = 3
    costs = {
        (start, end): float((end - start - 2) ** 2) + start / 100.0
        for start in range(layer_count)
        for end in range(start + 1, layer_count + 1)
    }
    candidates = []
    for internal in itertools.combinations(range(1, layer_count), target_count - 1):
        boundaries = (0, *internal, layer_count)
        total = sum(
            costs[(start, end)]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        )
        candidates.append((total, internal, boundaries))
    expected = min(candidates, key=lambda item: (item[0], item[1]))

    result = plan_contiguous_spans(layer_count, target_count, costs)

    assert result["boundaries"] == list(expected[2])
    assert result["total_cost"] == pytest.approx(expected[0])


def test_exact_count_coverage_and_span_length_bounds() -> None:
    result = plan_contiguous_spans(
        8,
        3,
        _all_half_open_costs(8),
        min_span_length=2,
        max_span_length=3,
    )

    assert result["boundaries"] == [0, 2, 5, 8]
    assert [span["length"] for span in result["spans"]] == [2, 3, 3]
    assert [
        layer for span in result["spans"] for layer in span["donor_layers"]
    ] == list(range(8))
    coverage = result["report"]["coverage"]
    assert coverage["complete"] is True
    assert coverage["contiguous"] is True
    assert coverage["exact_target_span_count_satisfied"] is True


def test_equal_cost_ties_use_configured_boundary_order() -> None:
    costs = _all_half_open_costs(5)

    earliest = plan_contiguous_spans(5, 3, costs)
    latest = plan_contiguous_spans(5, 3, costs, tie_break="latest-boundaries")

    assert earliest["boundaries"] == [0, 1, 2, 5]
    assert latest["boundaries"] == [0, 3, 4, 5]
    assert earliest["report"]["determinism"]["tie_break"] == "earliest-boundaries"
    assert latest["report"]["determinism"]["tie_break"] == "latest-boundaries"


def test_mapping_insertion_order_does_not_change_plan_or_hashes() -> None:
    costs = _all_half_open_costs(6)
    reversed_costs = dict(reversed(list(costs.items())))

    first = plan_contiguous_spans(6, 3, costs)
    second = plan_contiguous_spans(6, 3, reversed_costs)

    assert first["boundaries"] == second["boundaries"]
    assert first["report"]["hashes"] == second["report"]["hashes"]


def test_half_open_cost_matrix_is_supported_and_not_mutated() -> None:
    matrix = [[None] * 5 for _ in range(4)]
    matrix[0][2] = 1.25
    matrix[2][4] = -0.5
    original = deepcopy(matrix)

    result = plan_contiguous_spans(4, 2, matrix)

    assert result["boundaries"] == [0, 2, 4]
    assert result["total_cost"] == pytest.approx(0.75)
    assert result["report"]["cost_provenance"]["resolved_layout"] == "half-open"
    assert matrix == original


def test_inclusive_end_cost_matrix_is_inferred_from_dimensions() -> None:
    matrix = [[None] * 4 for _ in range(4)]
    matrix[0][1] = 2.0  # Inclusive [0, 1], canonical half-open [0, 2).
    matrix[2][3] = 3.0  # Inclusive [2, 3], canonical half-open [2, 4).

    result = plan_contiguous_spans(4, 2, matrix)

    assert result["span_ranges"] == [[0, 2], [2, 4]]
    assert result["report"]["cost_provenance"]["resolved_layout"] == "inclusive-end"


def test_inclusive_end_mapping_is_inferred_from_diagonal_key() -> None:
    costs = {
        (0, 0): 1.0,
        (1, 2): 2.0,
    }

    result = plan_contiguous_spans(3, 2, costs)

    assert result["boundaries"] == [0, 1, 3]
    assert result["report"]["cost_provenance"]["resolved_layout"] == "inclusive-end"


def test_sparse_cost_map_can_omit_unavailable_spans() -> None:
    costs = {
        (0, 2): 1.0,
        (2, 3): 2.0,
        (3, 6): 3.0,
        (0, 1): None,
        (1, 3): float("inf"),
    }

    result = plan_contiguous_spans(6, 3, costs)

    assert result["boundaries"] == [0, 2, 3, 6]
    provenance = result["report"]["cost_provenance"]
    assert provenance["source_type"] == "mapping"
    assert provenance["missing_candidate_span_count"] > 0


def test_critical_layers_are_forced_into_distinct_spans_by_default() -> None:
    costs = _all_half_open_costs(6)
    costs[(0, 3)] = -100.0

    result = plan_contiguous_spans(
        6,
        3,
        costs,
        critical_layers=[0, 1, 2],
    )

    assert all(span["critical_layer_count"] <= 1 for span in result["spans"])
    assert (
        result["report"]["critical_layer_provenance"][
            "distinct_span_constraint_satisfied"
        ]
        is True
    )


def test_explicit_penalty_allows_critical_layers_to_share() -> None:
    costs = {
        (0, 1): 3.0,
        (1, 4): 3.0,
        (0, 2): 0.0,
        (2, 4): 0.0,
    }

    result = plan_contiguous_spans(
        4,
        2,
        costs,
        critical_layers=[0, 1],
        critical_layer_sharing_penalty=5.0,
    )

    assert result["boundaries"] == [0, 2, 4]
    assert result["base_cost"] == pytest.approx(0.0)
    assert result["critical_sharing_penalty_cost"] == pytest.approx(5.0)
    assert result["total_cost"] == pytest.approx(5.0)
    assert result["report"]["critical_layer_provenance"]["sharing_pair_count"] == 1


def test_critical_penalty_is_charged_per_unordered_pair() -> None:
    result = plan_contiguous_spans(
        4,
        2,
        {(0, 3): 0.0, (3, 4): 0.0},
        critical_layers=[0, 1, 2],
        critical_layer_sharing_penalty=2.0,
    )

    assert result["spans"][0]["critical_sharing_pair_count"] == 3
    assert result["critical_sharing_penalty_cost"] == pytest.approx(6.0)


def test_larger_critical_penalty_changes_the_optimal_partition() -> None:
    costs = {
        (0, 1): 3.0,
        (1, 4): 3.0,
        (0, 2): 0.0,
        (2, 4): 0.0,
    }

    result = plan_contiguous_spans(
        4,
        2,
        costs,
        critical_layers=[0, 1],
        critical_layer_sharing_penalty=7.0,
    )

    assert result["boundaries"] == [0, 1, 4]
    assert result["critical_sharing_penalty_cost"] == pytest.approx(0.0)
    assert result["total_cost"] == pytest.approx(6.0)


def test_too_many_hard_critical_layers_has_structured_infeasibility() -> None:
    with pytest.raises(SpanPlanInfeasibleError) as raised:
        plan_contiguous_spans(4, 2, {}, critical_layers=[0, 1, 2])

    assert raised.value.report["status"] == "infeasible"
    assert raised.value.report["infeasibility"]["reason"] == "critical_layer_pigeonhole"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"min_span_length": 3}, "span_length_capacity"),
        ({"max_span_length": 1}, "span_length_capacity"),
    ],
)
def test_impossible_length_capacity_has_structured_infeasibility(
    kwargs: dict, reason: str
) -> None:
    with pytest.raises(SpanPlanInfeasibleError) as raised:
        plan_contiguous_spans(7, 3, {}, **kwargs)

    assert raised.value.report["infeasibility"]["reason"] == reason


def test_missing_cost_path_has_dynamic_programming_infeasibility_report() -> None:
    with pytest.raises(SpanPlanInfeasibleError) as raised:
        plan_contiguous_spans(5, 2, {(0, 2): 1.0})

    report = raised.value.report
    assert report["infeasibility"]["reason"] == "no_feasible_partition"
    assert report["cost_provenance"]["available_candidate_span_count"] == 1
    assert report["dynamic_programming"]["transition_missing_cost_count"] > 0
    assert len(report["hashes"]["input_sha256"]) == 64


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((0, 1, {}), {}, "donor_layer_count"),
        ((4, 2, {}), {"min_span_length": 3, "max_span_length": 2}, "greater"),
        ((4, 2, {(0, 2): float("nan")}), {}, "finite"),
        ((4, 2, {(0, 2): True}), {}, "numeric"),
        ((4, 2, {}), {"critical_layers": [4]}, "outside"),
        (
            (4, 2, {}),
            {"critical_layer_sharing_penalty": -1.0},
            "non-negative",
        ),
        ((4, 2, {}), {"tie_break": "random"}, "tie_break"),
    ],
)
def test_malformed_inputs_are_rejected(args: tuple, kwargs: dict, message: str) -> None:
    with pytest.raises(SpanPlanningError, match=message):
        plan_contiguous_spans(*args, **kwargs)


def test_report_is_json_serializable_and_contains_reproducible_provenance() -> None:
    result = plan_contiguous_spans(
        5,
        2,
        _all_half_open_costs(5, 1.0),
        critical_layers=[1, 4],
    )
    report = result["report"]

    serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    assert serialized
    assert report["status"] == "feasible"
    assert report["method"]["algorithm"] == "dynamic-programming"
    assert report["method"]["torch_dependency"] is False
    assert report["objective"]["total_cost"] == result["total_cost"]
    assert len(report["hashes"]["candidate_costs_sha256"]) == 64
    assert len(report["hashes"]["plan_sha256"]) == 64


def test_plan_layer_spans_alias_uses_the_same_contract() -> None:
    direct = plan_contiguous_spans(3, 2, {(0, 1): 1.0, (1, 3): 2.0})
    aliased = plan_layer_spans(3, 2, {(0, 1): 1.0, (1, 3): 2.0})

    assert aliased == direct


def test_module_has_no_torch_import() -> None:
    source_path = (
        Path(__file__).parents[1] / "src" / "neural_forge" / "hf_span_planner.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert "torch" not in imported_roots
