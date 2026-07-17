from __future__ import annotations

import pytest

from neural_forge.dynamic_fold_budget import (
    DynamicFoldBudgetError,
    DynamicFoldBudgetUnattainable,
    ExactTopologyBudgetProblem,
    SharedTopologyResource,
    TopologyChoiceGroup,
    TopologyOption,
    solve_exact_topology_budget,
)


def _option(
    option_id: str,
    group_id: str,
    parameters: int,
    *,
    resources: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
    risk: float = 0.0,
) -> TopologyOption:
    return TopologyOption(
        option_id=option_id,
        group_id=group_id,
        local_unique_parameters=parameters,
        local_serialized_bytes=parameters * 2,
        shared_resource_ids=resources,
        dependency_option_ids=dependencies,
        conflict_keys=conflicts,
        predicted_risk=risk,
    )


def test_exact_budget_counts_shared_resources_once_and_minimizes_risk() -> None:
    problem = ExactTopologyBudgetProblem(
        base_unique_parameters=10,
        base_serialized_bytes=20,
        exact_target_unique_parameters=32,
        shared_resources=(SharedTopologyResource("shared.scale", 4, 8),),
        groups=(
            TopologyChoiceGroup(
                "attention",
                (
                    _option(
                        "attention.compact",
                        "attention",
                        8,
                        resources=("shared.scale",),
                        risk=0.2,
                    ),
                    _option("attention.dense", "attention", 12, risk=0.0),
                ),
            ),
            TopologyChoiceGroup(
                "mlp",
                (
                    _option(
                        "mlp.compact",
                        "mlp",
                        10,
                        resources=("shared.scale",),
                        risk=0.1,
                    ),
                    _option("mlp.dense", "mlp", 14, risk=0.0),
                ),
            ),
        ),
    )

    selection = solve_exact_topology_budget(problem)

    assert selection.selected_option_ids == (
        "attention.compact",
        "mlp.compact",
    )
    assert selection.shared_resource_ids == ("shared.scale",)
    assert selection.exact_unique_parameters == 32
    assert selection.total_predicted_risk == pytest.approx(0.3)


def test_dependencies_and_conflicts_are_hard_constraints() -> None:
    problem = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=12,
        groups=(
            TopologyChoiceGroup(
                "codec",
                (
                    _option(
                        "codec.byte",
                        "codec",
                        4,
                        conflicts=("native-only",),
                    ),
                    _option("codec.native", "codec", 6),
                ),
            ),
            TopologyChoiceGroup(
                "head",
                (
                    _option(
                        "head.byte",
                        "head",
                        8,
                        dependencies=("codec.byte",),
                    ),
                    _option(
                        "head.native",
                        "head",
                        6,
                        dependencies=("codec.native",),
                    ),
                    _option(
                        "head.conflict",
                        "head",
                        8,
                        conflicts=("native-only",),
                    ),
                ),
            ),
        ),
    )

    selection = solve_exact_topology_budget(problem)

    assert selection.selected_option_ids == ("codec.byte", "head.byte")


def test_unknown_dependencies_fail_before_search() -> None:
    with pytest.raises(DynamicFoldBudgetError, match="unknown dependencies"):
        ExactTopologyBudgetProblem(
            base_unique_parameters=0,
            base_serialized_bytes=0,
            exact_target_unique_parameters=1,
            groups=(
                TopologyChoiceGroup(
                    "only",
                    (
                        _option(
                            "only.option",
                            "only",
                            1,
                            dependencies=("missing",),
                        ),
                    ),
                ),
            ),
        )


def test_unattainable_exact_budget_and_state_ceiling_fail_closed() -> None:
    impossible = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=3,
        groups=(
            TopologyChoiceGroup(
                "a",
                (_option("a.2", "a", 2), _option("a.4", "a", 4)),
            ),
        ),
    )
    with pytest.raises(DynamicFoldBudgetUnattainable, match="exactly reaches"):
        solve_exact_topology_budget(impossible)

    bounded = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=4,
        max_states=1,
        groups=(
            TopologyChoiceGroup(
                "a",
                (_option("a.1", "a", 1), _option("a.2", "a", 2)),
            ),
            TopologyChoiceGroup(
                "b",
                (_option("b.2", "b", 2), _option("b.3", "b", 3)),
            ),
        ),
    )
    with pytest.raises(DynamicFoldBudgetUnattainable, match="state"):
        solve_exact_topology_budget(bounded)
