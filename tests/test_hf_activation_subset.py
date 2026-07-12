from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_activation_subset import (  # noqa: E402
    ActivationSubsetFiniteError,
    ActivationSubsetShapeError,
    build_gated_mlp_activation_subset_plan,
    build_gqa_activation_subset_plan,
)


def _ridge_validation_error(
    calibration_design: torch.Tensor,
    calibration_target: torch.Tensor,
    validation_design: torch.Tensor,
    validation_target: torch.Tensor,
    selected_columns: tuple[int, ...],
    *,
    ridge: float = 1e-6,
) -> float:
    design = calibration_design[:, selected_columns]
    validation = validation_design[:, selected_columns]
    gram = design.T @ design
    effective_ridge = ridge * max(
        float(torch.diagonal(gram).mean().item()),
        torch.finfo(torch.float64).eps,
    )
    coefficients = torch.linalg.solve(
        gram
        + effective_ridge
        * torch.eye(len(selected_columns), dtype=torch.float64),
        design.T @ calibration_target,
    )
    residual = validation_target - validation @ coefficients
    return float(
        torch.linalg.vector_norm(residual)
        / torch.linalg.vector_norm(validation_target)
    )


def _evenly_spaced(item_count: int, selected_count: int) -> tuple[int, ...]:
    if selected_count == 1:
        return (0,)
    return tuple(
        round(index * (item_count - 1) / (selected_count - 1))
        for index in range(selected_count)
    )


def _low_rank_mlp_fixture() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(11)
    calibration_latent = torch.randn(96, 3, dtype=torch.float64)
    validation_latent = torch.randn(80, 3, dtype=torch.float64)
    latent_kind = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0)
    scales = (1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.0, 1.1, 1.05, 1.0, 0.95)
    post = torch.stack(
        [
            calibration_latent[:, kind] * scales[index]
            for index, kind in enumerate(latent_kind)
        ],
        dim=1,
    )
    validation_post = torch.stack(
        [
            validation_latent[:, kind] * scales[index]
            for index, kind in enumerate(latent_kind)
        ],
        dim=1,
    )
    gate = torch.ones_like(post)
    up = post / torch.nn.functional.silu(gate)
    directions = torch.zeros(12, 3, dtype=torch.float64)
    for neuron, kind in enumerate(latent_kind):
        directions[neuron, kind] = 2.0 if kind == 0 else 1.0
    torch.testing.assert_close(torch.nn.functional.silu(gate) * up, post)
    return gate, up, post, directions, validation_post


def test_mlp_low_rank_subset_beats_energy_and_even_spacing_on_heldout_rows() -> (
    None
):
    gate, up, post, directions, validation_post = _low_rank_mlp_fixture()
    plan = build_gated_mlp_activation_subset_plan(
        gate,
        up,
        post,
        directions,
        target_neurons=3,
        validation_post_swiglu_activations=validation_post,
    )
    repeated = build_gated_mlp_activation_subset_plan(
        gate,
        up,
        post,
        directions,
        target_neurons=3,
        validation_post_swiglu_activations=validation_post,
    )
    assert plan == repeated
    assert plan.selected_source_neurons == (0, 7, 10)
    assert plan.report.validation is not None
    subset_error = plan.report.validation.relative_l2_error

    target = post @ directions
    validation_target = validation_post @ directions
    energy = torch.sqrt(
        post.square().sum(dim=0) * directions.square().sum(dim=1)
    )
    energy_only = tuple(
        sorted(
            range(post.shape[1]),
            key=lambda index: (-float(energy[index].item()), index),
        )[:3]
    )
    evenly_spaced = _evenly_spaced(int(post.shape[1]), 3)
    energy_error = _ridge_validation_error(
        post,
        target,
        validation_post,
        validation_target,
        energy_only,
    )
    evenly_spaced_error = _ridge_validation_error(
        post,
        target,
        validation_post,
        validation_target,
        evenly_spaced,
    )
    assert energy_only == (0, 1, 2)
    assert evenly_spaced == (0, 6, 11)
    assert subset_error < energy_error * 1e-3
    assert subset_error < evenly_spaced_error * 1e-3
    assert plan.report.calibration.explained_energy > 0.999999
    assert plan.report.selection.estimated_feature_rank >= 3


def test_mlp_plan_preserves_coupled_real_producers_and_joint_consumer() -> None:
    gate, up, post, directions, validation_post = _low_rank_mlp_fixture()
    plan = build_gated_mlp_activation_subset_plan(
        gate,
        up,
        post,
        directions,
        target_neurons=3,
        validation_post_swiglu_activations=validation_post,
    )
    gate_weight = torch.arange(12 * 5, dtype=torch.float64).reshape(12, 5)
    up_weight = gate_weight + 1_000.0
    selected_gate = plan.select_gate_producer(gate_weight)
    selected_up = plan.select_up_producer(up_weight)
    index = torch.tensor(plan.selected_source_neurons)
    assert torch.equal(selected_gate, gate_weight[index])
    assert torch.equal(selected_up, up_weight[index])
    assert tuple(plan.fitted_down_consumer().shape) == (3, 3)
    diagnostics = plan.evaluate_reconstruction(
        validation_post,
        residual_contribution_directions=directions,
    )
    assert diagnostics == plan.report.validation
    diagnostics_from_delta = plan.evaluate_reconstruction(
        validation_post,
        total_residual_delta=validation_post @ directions,
    )
    assert diagnostics_from_delta == diagnostics
    assert plan.report.classification == "approximate"
    assert plan.report.zero_gradient is True
    assert plan.report.coupled_gate_up_down_identities is True
    assert plan.report["producer_aggregation"] == (
        "none-real-source-neurons-only"
    )
    assert len(plan.report.input_sha256) == 64
    assert len(plan.report.fitted_consumer_sha256) == 64
    assert len(plan.report.plan_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        plan.report.target_neurons = 4  # type: ignore[misc]


def test_mlp_requires_enough_rows_unless_explicitly_overridden() -> None:
    torch.manual_seed(3)
    gate = torch.randn(2, 5, dtype=torch.float64)
    up = torch.randn(2, 5, dtype=torch.float64)
    post = torch.nn.functional.silu(gate) * up
    directions = torch.randn(5, 4, dtype=torch.float64)
    with pytest.raises(ActivationSubsetShapeError, match="observation_rows"):
        build_gated_mlp_activation_subset_plan(
            gate,
            up,
            post,
            directions,
            target_neurons=3,
        )
    plan = build_gated_mlp_activation_subset_plan(
        gate,
        up,
        post,
        directions,
        target_neurons=3,
        allow_insufficient_rows=True,
    )
    assert len(plan.selected_source_neurons) == 3
    assert plan.report.calibration.observation_rows == 2


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("direction_width", ActivationSubsetShapeError),
        ("activation_shape", ActivationSubsetShapeError),
        ("nonfinite", ActivationSubsetFiniteError),
    ),
)
def test_mlp_rejects_malformed_or_nonfinite_traces(
    mutation: str,
    error: type[Exception],
) -> None:
    gate, up, post, directions, _ = _low_rank_mlp_fixture()
    if mutation == "direction_width":
        directions = directions[:-1]
    elif mutation == "activation_shape":
        up = up[:, :-1]
    else:
        post = post.clone()
        post[0, 0] = float("nan")
    with pytest.raises(error):
        build_gated_mlp_activation_subset_plan(
            gate,
            up,
            post,
            directions,
            target_neurons=3,
        )


def test_mlp_fingerprint_changes_when_source_trace_changes() -> None:
    gate, up, post, directions, _ = _low_rank_mlp_fixture()
    original = build_gated_mlp_activation_subset_plan(
        gate,
        up,
        post,
        directions,
        target_neurons=3,
    )
    changed_post = post.clone()
    changed_post[0, 0] += 1e-9
    changed = build_gated_mlp_activation_subset_plan(
        gate,
        up,
        changed_post,
        directions,
        target_neurons=3,
    )
    assert original.report.input_sha256 != changed.report.input_sha256
    assert original.report.plan_sha256 != changed.report.plan_sha256


def _low_rank_gqa_fixture() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(21)
    calibration_latent = torch.randn(96, 2, dtype=torch.float64)
    validation_latent = torch.randn(80, 2, dtype=torch.float64)
    latent_kind = (0, 0, 1, 0)
    scales = (1.8, 1.7, 1.0, 1.6)
    context = torch.zeros(96, 4, 2, dtype=torch.float64)
    validation_context = torch.zeros(80, 4, 2, dtype=torch.float64)
    contribution = torch.zeros(96, 4, 2, dtype=torch.float64)
    validation_contribution = torch.zeros(80, 4, 2, dtype=torch.float64)
    for group, (kind, scale) in enumerate(zip(latent_kind, scales, strict=True)):
        context[:, group, 0] = calibration_latent[:, kind] * scale
        context[:, group, 1] = calibration_latent[:, kind] * scale * 0.5
        validation_context[:, group, 0] = validation_latent[:, kind] * scale
        validation_context[:, group, 1] = (
            validation_latent[:, kind] * scale * 0.5
        )
        contribution[:, group, kind] = (
            calibration_latent[:, kind] * scale * (2.0 if kind == 0 else 1.0)
        )
        validation_contribution[:, group, kind] = (
            validation_latent[:, kind] * scale * (2.0 if kind == 0 else 1.0)
        )
    return (
        context * 0.9,
        context * 0.8,
        context * 1.1,
        context,
        contribution,
        validation_context,
        validation_contribution,
    )


def test_gqa_low_rank_subset_beats_energy_and_even_spacing_on_heldout_rows() -> (
    None
):
    query, key, value, context, contribution, val_context, val_contribution = (
        _low_rank_gqa_fixture()
    )
    plan = build_gqa_activation_subset_plan(
        query,
        key,
        value,
        context,
        contribution,
        source_query_heads=4,
        source_key_value_heads=4,
        target_query_heads=2,
        target_key_value_heads=2,
        head_dim=2,
        validation_context_trajectories=val_context,
        validation_contribution_trajectories=val_contribution,
    )
    repeated = build_gqa_activation_subset_plan(
        query,
        key,
        value,
        context,
        contribution,
        source_query_heads=4,
        source_key_value_heads=4,
        target_query_heads=2,
        target_key_value_heads=2,
        head_dim=2,
        validation_context_trajectories=val_context,
        validation_contribution_trajectories=val_contribution,
    )
    assert plan == repeated
    assert plan.selected_source_groups == (0, 2)
    assert plan.selected_source_query_heads == (0, 2)
    assert plan.report.validation is not None
    subset_error = plan.report.validation.relative_l2_error

    target = contribution.sum(dim=1)
    validation_target = val_contribution.sum(dim=1)
    flat_context = context.reshape(context.shape[0], -1)
    flat_validation = val_context.reshape(val_context.shape[0], -1)
    energy = torch.sqrt(contribution.square().sum(dim=(0, 2)))
    energy_groups = tuple(
        sorted(
            range(4),
            key=lambda index: (-float(energy[index].item()), index),
        )[:2]
    )
    evenly_spaced_groups = _evenly_spaced(4, 2)

    def coordinates(groups: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            group * 2 + coordinate for group in groups for coordinate in range(2)
        )

    energy_error = _ridge_validation_error(
        flat_context,
        target,
        flat_validation,
        validation_target,
        coordinates(energy_groups),
    )
    evenly_error = _ridge_validation_error(
        flat_context,
        target,
        flat_validation,
        validation_target,
        coordinates(evenly_spaced_groups),
    )
    assert energy_groups == (0, 1)
    assert evenly_spaced_groups == (0, 3)
    assert subset_error < energy_error * 1e-3
    assert subset_error < evenly_error * 1e-3
    assert plan.report.group_selection.estimated_feature_rank == 2


def test_gqa_selects_complete_real_heads_and_preserves_group_topology() -> None:
    torch.manual_seed(9)
    rows = 32
    query = torch.randn(rows, 8, 2, dtype=torch.float64)
    key = torch.randn(rows, 4, 2, dtype=torch.float64)
    value = torch.randn(rows, 4, 2, dtype=torch.float64)
    context = torch.randn(rows, 8, 2, dtype=torch.float64)
    contribution = torch.randn(rows, 8, 5, dtype=torch.float64)
    plan = build_gqa_activation_subset_plan(
        query,
        key,
        value,
        context,
        contribution,
        source_query_heads=8,
        source_key_value_heads=4,
        target_query_heads=2,
        target_key_value_heads=2,
        source_head_dim=2,
        target_head_dim=2,
    )
    expected_heads = tuple(
        group * 2 + offset
        for group in plan.selected_source_groups
        for offset in plan.selected_query_offsets
    )
    assert plan.selected_source_query_heads == expected_heads
    query_weight = torch.arange(16 * 3, dtype=torch.float64).reshape(16, 3)
    key_weight = torch.arange(8 * 3, dtype=torch.float64).reshape(8, 3)
    value_weight = key_weight + 500.0
    selected_query = plan.select_query_producer(query_weight)
    selected_key = plan.select_key_producer(key_weight)
    selected_value = plan.select_value_producer(value_weight)
    query_coordinates = torch.tensor(
        [
            head * 2 + coordinate
            for head in plan.selected_source_query_heads
            for coordinate in range(2)
        ]
    )
    kv_coordinates = torch.tensor(
        [
            group * 2 + coordinate
            for group in plan.selected_source_groups
            for coordinate in range(2)
        ]
    )
    assert torch.equal(selected_query, query_weight[query_coordinates])
    assert torch.equal(selected_key, key_weight[kv_coordinates])
    assert torch.equal(selected_value, value_weight[kv_coordinates])
    assert tuple(plan.output_consumer().shape) == (5, 4)
    assert plan.report.classification == "approximate"
    assert plan.report.complete_real_heads is True
    assert plan.report["gqa_topology_preserved"] is True
    assert plan.report["head_dim_preserved"] is True
    assert len(plan.report.plan_sha256) == 64


def test_gqa_requires_enough_rows_unless_explicitly_overridden() -> None:
    torch.manual_seed(5)
    query = torch.randn(1, 4, 2, dtype=torch.float64)
    key = torch.randn(1, 4, 2, dtype=torch.float64)
    value = torch.randn(1, 4, 2, dtype=torch.float64)
    context = torch.randn(1, 4, 2, dtype=torch.float64)
    contribution = torch.randn(1, 4, 3, dtype=torch.float64)
    kwargs = {
        "source_query_heads": 4,
        "source_key_value_heads": 4,
        "target_query_heads": 2,
        "target_key_value_heads": 2,
        "head_dim": 2,
    }
    with pytest.raises(ActivationSubsetShapeError, match="observation_rows"):
        build_gqa_activation_subset_plan(
            query,
            key,
            value,
            context,
            contribution,
            **kwargs,
        )
    plan = build_gqa_activation_subset_plan(
        query,
        key,
        value,
        context,
        contribution,
        allow_insufficient_rows=True,
        **kwargs,
    )
    assert len(plan.selected_source_query_heads) == 2


@pytest.mark.parametrize(
    "overrides",
    (
        {"source_query_heads": 6, "source_key_value_heads": 4},
        {"target_query_heads": 3, "target_key_value_heads": 2},
        {"target_head_dim": 4},
    ),
)
def test_gqa_rejects_malformed_topology(overrides: dict[str, int]) -> None:
    query, key, value, context, contribution, _, _ = _low_rank_gqa_fixture()
    kwargs = {
        "source_query_heads": 4,
        "source_key_value_heads": 4,
        "target_query_heads": 2,
        "target_key_value_heads": 2,
        "head_dim": 2,
    }
    kwargs.update(overrides)
    with pytest.raises(ActivationSubsetShapeError):
        build_gqa_activation_subset_plan(
            query,
            key,
            value,
            context,
            contribution,
            **kwargs,
        )


def test_gqa_rejects_partial_or_ambiguous_validation_evidence() -> None:
    query, key, value, context, contribution, validation_context, _ = (
        _low_rank_gqa_fixture()
    )
    with pytest.raises(ActivationSubsetShapeError, match="exactly one"):
        build_gqa_activation_subset_plan(
            query,
            key,
            value,
            context,
            contribution,
            source_query_heads=4,
            source_key_value_heads=4,
            target_query_heads=2,
            target_key_value_heads=2,
            head_dim=2,
            validation_context_trajectories=validation_context,
        )

