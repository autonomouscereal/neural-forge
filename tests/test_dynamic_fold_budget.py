from __future__ import annotations

import random
from itertools import product

import pytest

import neural_forge.dynamic_fold_budget as budget_module
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


def test_independent_suffix_solver_matches_exhaustive_objective() -> None:
    groups = (
        TopologyChoiceGroup(
            "a",
            (
                _option("a.1", "a", 1, risk=0.7),
                _option("a.2.expensive", "a", 2, risk=0.5),
                _option("a.2.preferred", "a", 2, risk=0.1),
                _option("a.4", "a", 4, risk=0.0),
            ),
        ),
        TopologyChoiceGroup(
            "b",
            (
                _option("b.1", "b", 1, risk=0.8),
                _option("b.3", "b", 3, risk=0.2),
                _option("b.5", "b", 5, risk=0.0),
            ),
        ),
        TopologyChoiceGroup(
            "c",
            (
                _option("c.1", "c", 1, risk=0.9),
                _option("c.2", "c", 2, risk=0.3),
                _option("c.4", "c", 4, risk=0.0),
            ),
        ),
    )
    problem = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=7,
        groups=groups,
    )
    exhaustive = min(
        (
            sum(option.predicted_risk for option in choices),
            sum(option.local_serialized_bytes for option in choices),
            tuple(option.option_id for option in choices),
        )
        for choices in product(*(group.options for group in problem.groups))
        if sum(option.local_unique_parameters for option in choices) == 7
    )

    selection = solve_exact_topology_budget(problem)

    assert (
        selection.total_predicted_risk,
        selection.serialized_bytes,
        selection.selected_option_ids,
    ) == exhaustive
    assert "a.2.expensive" not in selection.selected_option_ids


def test_independent_suffix_solver_matches_generic_solver(
    monkeypatch,
) -> None:
    randomizer = random.Random(20_260_729)
    problems = []
    for case_index in range(8):
        groups = []
        witness_parameters = 0
        for group_index in range(4):
            options = tuple(
                TopologyOption(
                    option_id=f"case{case_index}.g{group_index}.o{option_index}",
                    group_id=f"g{group_index}",
                    local_unique_parameters=randomizer.randint(1, 12),
                    local_serialized_bytes=randomizer.randint(4, 48),
                    predicted_risk=randomizer.randint(0, 40) / 10,
                )
                for option_index in range(6)
            )
            group = TopologyChoiceGroup(f"g{group_index}", options)
            groups.append(group)
            witness_parameters += randomizer.choice(group.options).local_unique_parameters
        problems.append(
            ExactTopologyBudgetProblem(
                base_unique_parameters=3,
                base_serialized_bytes=5,
                exact_target_unique_parameters=3 + witness_parameters,
                groups=tuple(groups),
            )
        )

    optimized = tuple(solve_exact_topology_budget(problem) for problem in problems)
    stateful = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=2,
        groups=(
            TopologyChoiceGroup(
                "a",
                (
                    _option(
                        "a.requires-b",
                        "a",
                        1,
                        dependencies=("b.option",),
                    ),
                ),
            ),
            TopologyChoiceGroup(
                "b",
                (_option("b.option", "b", 1),),
            ),
        ),
    )
    assert budget_module._solve_independent_exact_topology_budget(stateful) is None

    monkeypatch.setattr(
        budget_module,
        "_solve_independent_exact_topology_budget",
        lambda _problem: None,
    )
    generic = tuple(solve_exact_topology_budget(problem) for problem in problems)

    for fast, reference in zip(optimized, generic, strict=True):
        assert fast.selected_option_ids == reference.selected_option_ids
        assert fast.exact_unique_parameters == reference.exact_unique_parameters
        assert fast.serialized_bytes == reference.serialized_bytes
        assert fast.total_predicted_risk == pytest.approx(
            reference.total_predicted_risk
        )
