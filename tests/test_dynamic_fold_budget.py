from __future__ import annotations

import math
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


def _legacy_independent_exact_topology_budget(
    problem: ExactTopologyBudgetProblem,
):
    """Frozen pre-optimization reference for exact output-parity tests."""

    groups = budget_module._independent_exact_group_options(problem)
    if groups is None:
        return None
    minimum_counts = tuple(
        min(option.local_unique_parameters for option in group) for group in groups
    )
    delta_groups = tuple(
        tuple((option.local_unique_parameters - minimum, option) for option in group)
        for group, minimum in zip(groups, minimum_counts, strict=True)
    )
    required_delta = (
        problem.exact_target_unique_parameters
        - problem.base_unique_parameters
        - sum(minimum_counts)
    )
    maximum_deltas = tuple(
        max(delta for delta, _option_value in group) for group in delta_groups
    )
    maximum_delta = sum(maximum_deltas)
    if required_delta < 0 or required_delta > maximum_delta:
        raise DynamicFoldBudgetUnattainable(
            "No legal topology exactly reaches "
            f"{problem.exact_target_unique_parameters} unique parameters."
        )

    lattice_quantum = 0
    for group in delta_groups:
        for delta, _option_value in group:
            lattice_quantum = math.gcd(lattice_quantum, delta)
    if lattice_quantum == 0:
        lattice_quantum = 1
    if required_delta % lattice_quantum:
        raise DynamicFoldBudgetUnattainable(
            "No legal topology exactly reaches "
            f"{problem.exact_target_unique_parameters} unique parameters."
        )
    normalized_groups = tuple(
        tuple((delta // lattice_quantum, option) for delta, option in group)
        for group in delta_groups
    )
    normalized_required = required_delta // lattice_quantum
    normalized_maxima = tuple(maximum // lattice_quantum for maximum in maximum_deltas)
    suffix_total_bit_bound = len(groups) + 1
    suffix_span = 0
    for maximum in reversed(normalized_maxima):
        suffix_span += maximum
        suffix_total_bit_bound += suffix_span
    if suffix_total_bit_bound > (
        budget_module._INDEPENDENT_SUFFIX_REACHABILITY_TOTAL_BIT_LIMIT
    ):
        return None

    suffix_integers = [0] * (len(groups) + 1)
    suffix_integers[-1] = 1
    for group_index in range(len(groups) - 1, -1, -1):
        reachable = 0
        following = suffix_integers[group_index + 1]
        for delta, _option_value in normalized_groups[group_index]:
            reachable |= following << delta
        suffix_integers[group_index] = reachable
    suffix_bitsets = tuple(
        value.to_bytes((value.bit_length() + 7) // 8, "little")
        for value in suffix_integers
    )
    if not budget_module._bitset_contains(
        suffix_bitsets[0],
        normalized_required,
    ):
        raise DynamicFoldBudgetUnattainable(
            "No legal topology exactly reaches "
            f"{problem.exact_target_unique_parameters} unique parameters."
        )

    initial = budget_module._State(
        (),
        frozenset(),
        frozenset(),
        problem.base_unique_parameters,
        problem.base_serialized_bytes,
        0.0,
    )
    states = {0: initial}
    considered = 1
    peak = 1
    for group_index, group in enumerate(normalized_groups):
        next_by_delta = {}
        suffix = suffix_bitsets[group_index + 1]
        for prefix_delta, state in states.items():
            for option_delta, option in group:
                considered += 1
                candidate_delta = prefix_delta + option_delta
                if not budget_module._bitset_contains(
                    suffix,
                    normalized_required - candidate_delta,
                ):
                    continue
                candidate = budget_module._State(
                    state.selected_option_ids + (option.option_id,),
                    frozenset(),
                    frozenset(),
                    state.unique_parameters + option.local_unique_parameters,
                    state.serialized_bytes + option.local_serialized_bytes,
                    state.risk + option.predicted_risk,
                )
                retained = next_by_delta.get(candidate_delta)
                if retained is None or budget_module._state_objective(
                    candidate
                ) < budget_module._state_objective(retained):
                    next_by_delta[candidate_delta] = candidate
                if len(next_by_delta) > problem.max_states:
                    raise DynamicFoldBudgetUnattainable(
                        "Exact topology search exceeded its immutable retained-state "
                        "ceiling after exact suffix pruning; refusing an approximate "
                        "budget claim."
                    )
        if not next_by_delta:
            raise DynamicFoldBudgetUnattainable(
                "No legal topology exactly reaches "
                f"{problem.exact_target_unique_parameters} unique parameters."
            )
        states = next_by_delta
        peak = max(peak, len(states))

    selected = states.get(normalized_required)
    if selected is None or (
        selected.unique_parameters != problem.exact_target_unique_parameters
    ):
        raise DynamicFoldBudgetUnattainable(
            "No legal topology exactly reaches "
            f"{problem.exact_target_unique_parameters} unique parameters."
        )
    return budget_module.ExactTopologyBudgetSelection(
        problem_fingerprint=problem.fingerprint,
        selected_option_ids=selected.selected_option_ids,
        shared_resource_ids=(),
        exact_unique_parameters=selected.unique_parameters,
        serialized_bytes=selected.serialized_bytes,
        total_predicted_risk=selected.risk,
        states_considered=considered,
        peak_retained_states=peak,
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
        _legacy_independent_exact_topology_budget(bounded)
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
    legacy = _legacy_independent_exact_topology_budget(problem)

    assert (
        selection.total_predicted_risk,
        selection.serialized_bytes,
        selection.selected_option_ids,
    ) == exhaustive
    assert "a.2.expensive" not in selection.selected_option_ids
    assert legacy is not None
    assert selection.as_dict() == legacy.as_dict()
    assert selection.fingerprint == legacy.fingerprint


def test_independent_suffix_solver_preserves_bytes_and_id_tie_breaks() -> None:
    byte_problem = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=3,
        groups=(
            TopologyChoiceGroup(
                "a",
                (
                    TopologyOption("a.one", "a", 1, 100, predicted_risk=0.0),
                    TopologyOption("a.two", "a", 2, 1, predicted_risk=1.0),
                ),
            ),
            TopologyChoiceGroup(
                "b",
                (
                    TopologyOption("b.one", "b", 1, 1, predicted_risk=0.0),
                    TopologyOption("b.two", "b", 2, 1, predicted_risk=1.0),
                ),
            ),
        ),
    )
    byte_selection = solve_exact_topology_budget(byte_problem)
    byte_legacy = _legacy_independent_exact_topology_budget(byte_problem)
    assert byte_legacy is not None
    assert byte_selection.as_dict() == byte_legacy.as_dict()
    assert byte_selection.fingerprint == byte_legacy.fingerprint
    assert byte_selection.selected_option_ids == ("a.two", "b.one")

    id_problem = ExactTopologyBudgetProblem(
        base_unique_parameters=0,
        base_serialized_bytes=0,
        exact_target_unique_parameters=3,
        groups=(
            TopologyChoiceGroup(
                "a",
                (
                    _option("a.alpha", "a", 1, risk=0.0),
                    _option("a.beta", "a", 2, risk=0.0),
                ),
            ),
            TopologyChoiceGroup(
                "b",
                (
                    _option("b.alpha", "b", 1, risk=0.0),
                    _option("b.beta", "b", 2, risk=0.0),
                ),
            ),
        ),
    )
    id_selection = solve_exact_topology_budget(id_problem)
    id_legacy = _legacy_independent_exact_topology_budget(id_problem)
    assert id_legacy is not None
    assert id_selection.as_dict() == id_legacy.as_dict()
    assert id_selection.fingerprint == id_legacy.fingerprint
    assert id_selection.selected_option_ids == ("a.alpha", "b.beta")


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
    legacy = tuple(
        _legacy_independent_exact_topology_budget(problem) for problem in problems
    )
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

    for fast, frozen, reference in zip(optimized, legacy, generic, strict=True):
        assert frozen is not None
        assert fast.as_dict() == frozen.as_dict()
        assert fast.fingerprint == frozen.fingerprint
        assert fast.selected_option_ids == reference.selected_option_ids
        assert fast.exact_unique_parameters == reference.exact_unique_parameters
        assert fast.serialized_bytes == reference.serialized_bytes
        assert fast.total_predicted_risk == pytest.approx(
            reference.total_predicted_risk
        )


def test_independent_suffix_solver_preserves_exact_h816_policy_lattice() -> None:
    from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (
        derive_lfm2_automatic_mlp_width_policy,
    )

    layer_types = (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )
    hidden = 816
    vocabulary = 4_096
    exact_target = 53_552_368
    internal_widths = (704, 1_024)
    attention_options = ((12, 6, 64), (16, 8, 64))
    policy = derive_lfm2_automatic_mlp_width_policy(
        layer_types=layer_types,
        source_widths=(2_560,) * len(layer_types),
        short_conv_internal_widths=internal_widths,
        attention_head_options=attention_options,
        target_hidden_size=hidden,
        target_vocab_size=vocabulary,
        exact_target_unique_parameters=exact_target,
        conv_kernel_size=3,
        discover_exact_short_conv_writer_ranks=True,
    )
    assert policy.exact_budget_lattice_sha256 == (
        "f06ce3eed6247bfe1da3e2868bd0cc9b012a43ad00237fe32ba55dbf0edbeeb2"
    )
    assert policy.short_conv_writer_ranks == (40,)

    groups = []
    factor_layer_index = layer_types.index("conv")
    for layer_index, (layer_type, widths) in enumerate(
        zip(layer_types, policy.widths_by_layer, strict=True)
    ):
        group_id = f"{layer_index:02d}.{layer_type}"
        options = []
        for width in widths:
            mlp_parameters = 3 * hidden * width
            mlp_risk = (2_560 - width) / 2_560
            if layer_type == "conv":
                for internal in internal_widths:
                    fixed_operator = 3 * internal * hidden + internal * 3
                    dense_parameters = (
                        2 * hidden + mlp_parameters + fixed_operator + hidden * internal
                    )
                    options.append(
                        TopologyOption(
                            option_id=(f"{group_id}.m{width}.c{internal}.dense"),
                            group_id=group_id,
                            local_unique_parameters=dense_parameters,
                            local_serialized_bytes=4 * dense_parameters,
                            predicted_risk=(mlp_risk + (1.0 - internal / 1_024) ** 2),
                        )
                    )
                    if layer_index != factor_layer_index:
                        continue
                    for rank in policy.short_conv_writer_ranks:
                        if (
                            rank >= min(hidden, internal)
                            or rank * (hidden + internal) >= hidden * internal
                        ):
                            continue
                        factor_parameters = (
                            2 * hidden
                            + mlp_parameters
                            + fixed_operator
                            + rank * (hidden + internal)
                        )
                        options.append(
                            TopologyOption(
                                option_id=(f"{group_id}.m{width}.c{internal}.r{rank}"),
                                group_id=group_id,
                                local_unique_parameters=factor_parameters,
                                local_serialized_bytes=4 * factor_parameters,
                                predicted_risk=(
                                    mlp_risk
                                    + (1.0 - internal / 1_024) ** 2
                                    + (1.0 - rank / min(hidden, internal))
                                ),
                            )
                        )
            else:
                for query_heads, key_value_heads, head_dim in attention_options:
                    operator_parameters = (
                        2 * hidden * head_dim * (query_heads + key_value_heads)
                        + 2 * head_dim
                    )
                    local_parameters = 2 * hidden + mlp_parameters + operator_parameters
                    options.append(
                        TopologyOption(
                            option_id=(
                                f"{group_id}.m{width}.q{query_heads}."
                                f"k{key_value_heads}.d{head_dim}"
                            ),
                            group_id=group_id,
                            local_unique_parameters=local_parameters,
                            local_serialized_bytes=4 * local_parameters,
                            predicted_risk=(
                                mlp_risk + (0.05 if query_heads == 12 else 0.0)
                            ),
                        )
                    )
        groups.append(TopologyChoiceGroup(group_id, tuple(options)))

    boundary_parameters = 2 * vocabulary * hidden + hidden
    problem = ExactTopologyBudgetProblem(
        base_unique_parameters=boundary_parameters,
        base_serialized_bytes=4 * boundary_parameters,
        exact_target_unique_parameters=exact_target,
        groups=tuple(groups),
        max_states=100_000,
    )
    optimized = solve_exact_topology_budget(problem)
    legacy = _legacy_independent_exact_topology_budget(problem)

    assert legacy is not None
    assert optimized.as_dict() == legacy.as_dict()
    assert optimized.fingerprint == legacy.fingerprint
    assert optimized.exact_unique_parameters == exact_target
    assert sum(".r40" in option_id for option_id in optimized.selected_option_ids) == 1
