from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import time

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_activation_cluster as activation_cluster  # noqa: E402
from neural_forge.hf_activation_cluster import (  # noqa: E402
    ActivationClusterFiniteError,
    ActivationClusterShapeError,
    build_gated_mlp_activation_cluster_plan,
    build_gqa_activation_cluster_plan,
)


def test_gqa_equal_offsets_skip_feature_materialization_without_semantic_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(20260719)
    query = torch.randn(8, 9, 4, generator=generator, dtype=torch.float64)
    key = torch.randn(4, 9, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(4, 9, 4, generator=generator, dtype=torch.float64)
    context = torch.randn(8, 9, 4, generator=generator, dtype=torch.float64)
    contribution = torch.randn(8, 9, 5, generator=generator, dtype=torch.float64)
    combine_calls = []
    original_combine = activation_cluster._combine_modalities

    def counted_combine(torch_module, matrices):
        materialized = tuple(matrices)
        combine_calls.append(tuple(tuple(matrix.shape) for matrix in materialized))
        return original_combine(torch_module, materialized)

    monkeypatch.setattr(
        activation_cluster,
        "_combine_modalities",
        counted_combine,
    )
    plan = activation_cluster.build_gqa_activation_cluster_plan(
        query,
        key,
        value,
        context,
        contribution,
        source_query_heads=8,
        source_key_value_heads=4,
        target_query_heads=4,
        target_key_value_heads=2,
        head_dim=4,
    )

    # Group features are still consumed once.  Equal source/target Q-per-KV
    # offsets have the exact pre-existing identity result without building the
    # three transposed, normalized trajectory matrices.
    assert len(combine_calls) == 1
    assert len(combine_calls[0]) == 5
    assert plan.query_offset_medoids == (0, 1)
    assert plan.query_offset_assignment == (0, 1)
    assert plan.report.query_offset_cluster_strategy == (
        "deterministic-identity-real-medoid"
    )
    assert plan.report.query_offset_group_sizes == (1, 1)
    assert plan.report.invariants.all_hold is True

    combine_calls.clear()
    reduced_offset_plan = activation_cluster.build_gqa_activation_cluster_plan(
        query,
        key,
        value,
        context,
        contribution,
        source_query_heads=8,
        source_key_value_heads=4,
        target_query_heads=2,
        target_key_value_heads=2,
        head_dim=4,
    )
    assert len(combine_calls) == 2
    assert len(combine_calls[1]) == 3
    assert reduced_offset_plan.report.query_offset_cluster_strategy != (
        "deterministic-identity-real-medoid"
    )
    assert reduced_offset_plan.report.invariants.all_hold is True


def _head_signatures(rows: torch.Tensor, head_axis: int) -> torch.Tensor:
    return rows.movedim(head_axis, 0).reshape(rows.shape[head_axis], -1)


def _combined_modalities(*matrices: torch.Tensor) -> torch.Tensor:
    parts = []
    for matrix in matrices:
        width = int(matrix.shape[1])
        scale = float(torch.sqrt(matrix.square().mean()).item()) or 1.0
        parts.append(matrix / (scale * math.sqrt(width)))
    return torch.cat(parts, dim=1)


def _gqa_context(
    inputs: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_width = int(inputs.shape[-1])
    query = torch.einsum(
        "bsh,qdh->bsqd",
        inputs,
        query_weight.reshape(query_heads, head_dim, hidden_width),
    )
    key = torch.einsum(
        "bsh,kdh->bskd",
        inputs,
        key_weight.reshape(key_value_heads, head_dim, hidden_width),
    )
    value = torch.einsum(
        "bsh,kdh->bskd",
        inputs,
        value_weight.reshape(key_value_heads, head_dim, hidden_width),
    )
    query_heads_per_group = query_heads // key_value_heads
    context_heads = []
    for query_head in range(query_heads):
        group = query_head // query_heads_per_group
        scores = query[:, :, query_head] @ key[:, :, group].transpose(-1, -2)
        attention = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
        context_heads.append(attention @ value[:, :, group])
    context = torch.stack(context_heads, dim=2)
    return query, key, value, context


def test_gqa_duplicate_circuits_reconstruct_exactly_without_producer_averaging() -> (
    None
):
    dtype = torch.float64
    source_groups = 4
    source_offsets = 4
    source_query_heads = source_groups * source_offsets
    target_groups = 2
    target_offsets = 2
    target_query_heads = target_groups * target_offsets
    head_dim = 2
    hidden_width = 3
    output_width = 3
    group_kind = (0, 0, 1, 1)
    offset_kind = (0, 0, 1, 1)

    query_prototypes = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
                [[0.5, -0.5, 1.0], [1.0, 0.5, 0.0]],
            ],
            [
                [[-0.5, 1.0, 0.0], [0.5, 0.0, 1.0]],
                [[1.0, 1.0, -0.5], [-0.5, 0.5, 1.0]],
            ],
        ],
        dtype=dtype,
    )
    key_prototypes = torch.tensor(
        [
            [[1.0, 0.0, 0.25], [0.0, 1.0, -0.25]],
            [[-0.5, 0.5, 1.0], [1.0, 0.25, 0.0]],
        ],
        dtype=dtype,
    )
    value_prototypes = torch.tensor(
        [
            [[0.5, 1.0, 0.0], [1.0, -0.5, 0.5]],
            [[1.0, 0.0, -1.0], [0.25, 1.0, 0.5]],
        ],
        dtype=dtype,
    )
    output_prototypes = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.5, 1.0], [0.0, -0.5]],
                [[0.0, 1.0], [1.0, -0.5], [0.5, 0.5]],
            ],
            [
                [[-0.5, 1.0], [0.0, 0.5], [1.0, 0.0]],
                [[1.0, 0.5], [-0.5, 1.0], [0.0, 1.0]],
            ],
        ],
        dtype=dtype,
    )

    query_blocks = []
    output_blocks = []
    for group in range(source_groups):
        for offset in range(source_offsets):
            query_blocks.append(
                query_prototypes[group_kind[group], offset_kind[offset]]
            )
            output_blocks.append(
                output_prototypes[group_kind[group], offset_kind[offset]]
            )
    query_weight = torch.stack(query_blocks).reshape(
        source_query_heads * head_dim, hidden_width
    )
    key_weight = torch.stack(
        [key_prototypes[group_kind[group]] for group in range(source_groups)]
    ).reshape(source_groups * head_dim, hidden_width)
    value_weight = torch.stack(
        [value_prototypes[group_kind[group]] for group in range(source_groups)]
    ).reshape(source_groups * head_dim, hidden_width)
    output_weight = (
        torch.stack(output_blocks)
        .permute(1, 0, 2)
        .reshape(output_width, source_query_heads * head_dim)
    )
    inputs = torch.tensor(
        [
            [[1.0, -0.5, 0.25], [0.0, 1.0, -1.0], [0.5, 0.5, 1.0]],
            [[-1.0, 0.5, 0.75], [1.0, 0.0, 0.5], [0.25, -0.5, 1.0]],
        ],
        dtype=dtype,
    )
    query, key, value, context = _gqa_context(
        inputs,
        query_weight,
        key_weight,
        value_weight,
        query_heads=source_query_heads,
        key_value_heads=source_groups,
        head_dim=head_dim,
    )
    contribution = torch.stack(
        [
            context[:, :, head] @ output_blocks[head].T
            for head in range(source_query_heads)
        ],
        dim=2,
    )
    signatures = {
        "query_signatures": _head_signatures(query, 2),
        "key_signatures": _head_signatures(key, 2),
        "value_signatures": _head_signatures(value, 2),
        "context_signatures": _head_signatures(context, 2),
        "contribution_signatures": _head_signatures(contribution, 2),
    }

    plan = build_gqa_activation_cluster_plan(
        **signatures,
        source_query_heads=source_query_heads,
        source_key_value_heads=source_groups,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_groups,
        source_head_dim=head_dim,
        target_head_dim=head_dim,
    )
    repeated = build_gqa_activation_cluster_plan(
        **signatures,
        source_query_heads=source_query_heads,
        source_key_value_heads=source_groups,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_groups,
        head_dim=head_dim,
    )
    assert plan == repeated
    assert plan.report.invariants.all_hold is True
    assert plan.report.full_source_coverage is True
    assert plan.report["producer_aggregation"] == "none"
    assert plan.report.group_cluster_strategy == (
        "deterministic-farthest-first-global-assignment-real-medoid-refinement"
    )
    assert plan.report.query_offset_cluster_strategy == (
        "deterministic-farthest-first-global-assignment-real-medoid-refinement"
    )
    assert plan.report["invariants"]["gqa_group_offsets_preserved"] is True
    assert len(plan.group_assignment) == source_groups
    assert len(plan.query_head_assignment) == source_query_heads
    assert sum(plan.report.group_sizes) == source_groups
    assert sum(plan.report.query_head_group_sizes) == source_query_heads
    assert all(0 <= source < source_groups for source in plan.group_medoids)
    assert all(0 <= source < source_query_heads for source in plan.query_head_medoids)
    assert plan.context_consumer_coefficients == (1.0,) * source_query_heads

    child_query_weight = plan.select_query_producer(query_weight)
    child_key_weight = plan.select_key_producer(key_weight)
    child_value_weight = plan.select_value_producer(value_weight)
    child_output_weight = plan.aggregate_output_consumer(output_weight, axis=1)
    for target, source in enumerate(plan.query_head_medoids):
        source_block = query_weight.reshape(source_query_heads, head_dim, hidden_width)[
            source
        ]
        child_block = child_query_weight.reshape(
            target_query_heads, head_dim, hidden_width
        )[target]
        assert torch.equal(child_block, source_block)

    _, _, _, child_context = _gqa_context(
        inputs,
        child_query_weight,
        child_key_weight,
        child_value_weight,
        query_heads=target_query_heads,
        key_value_heads=target_groups,
        head_dim=head_dim,
    )
    source_output = (
        context.reshape(*context.shape[:2], source_query_heads * head_dim)
        @ output_weight.T
    )
    child_output = (
        child_context.reshape(*child_context.shape[:2], target_query_heads * head_dim)
        @ child_output_weight.T
    )
    torch.testing.assert_close(child_output, source_output, atol=1e-12, rtol=1e-12)


def test_gated_mlp_duplicate_circuits_reconstruct_with_near_zero_gates() -> None:
    dtype = torch.float64
    inputs = torch.tensor(
        [
            [1.0, 0.0, -0.5],
            [0.0, 1.0, 0.5],
            [0.5, -1.0, 1.0],
            [-1.0, 0.25, 0.0],
            [0.25, 0.75, -1.0],
        ],
        dtype=dtype,
    )
    neuron_kind = (0, 0, 0, 1, 1, 1)
    gate_prototypes = torch.tensor([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]], dtype=dtype)
    up_prototypes = torch.tensor([[0.5, 1.0, -1.0], [-0.25, 0.75, 1.0]], dtype=dtype)
    down_prototypes = torch.tensor([[1.0, -0.5], [0.25, 1.0]], dtype=dtype)
    gate_weight = torch.stack([gate_prototypes[kind] for kind in neuron_kind])
    up_weight = torch.stack([up_prototypes[kind] for kind in neuron_kind])
    down_weight = torch.stack([down_prototypes[kind] for kind in neuron_kind], dim=1)
    gate_rows = inputs @ gate_weight.T
    up_rows = inputs @ up_weight.T
    post_rows = torch.nn.functional.silu(gate_rows) * up_rows
    contribution_rows = post_rows.unsqueeze(2) * down_weight.T.unsqueeze(0)

    plan = build_gated_mlp_activation_cluster_plan(
        gate_rows,
        up_rows,
        post_rows,
        contribution_rows,
        target_neurons=2,
        source_neurons=6,
    )
    repeated = build_gated_mlp_activation_cluster_plan(
        gate_rows,
        up_rows,
        post_rows,
        contribution_rows,
        target_neurons=2,
    )
    assert plan == repeated
    assert plan.report.invariants.all_hold is True
    assert plan.report["consumer_coefficient_gate_division"] is False
    assert plan.report["producer_aggregation"] == "none"
    assert plan.report.cluster_strategy == (
        "deterministic-farthest-first-global-assignment-real-medoid-refinement"
    )
    assert plan.report.near_zero_post_swiglu_medoid_count == 1
    assert plan.consumer_coefficients == (1.0,) * 6
    assert sum(plan.report.group_sizes) == 6
    assert len(plan.source_to_target) == 6

    child_gate = plan.select_gate_producer(gate_weight)
    child_up = plan.select_up_producer(up_weight)
    child_down = plan.aggregate_down_consumer(down_weight, axis=1)
    for target, source in enumerate(plan.medoid_source_neurons):
        assert torch.equal(child_gate[target], gate_weight[source])
        assert torch.equal(child_up[target], up_weight[source])
    source_output = post_rows @ down_weight.T
    child_post = torch.nn.functional.silu(inputs @ child_gate.T) * (inputs @ child_up.T)
    child_output = child_post @ child_down.T
    torch.testing.assert_close(child_output, source_output, atol=1e-12, rtol=1e-12)

    with pytest.raises(FrozenInstanceError):
        plan.report.target_neurons = 3  # type: ignore[misc]


def test_gated_mlp_scalable_real_medoids_cover_thousands_deterministically() -> None:
    dtype = torch.float64
    source_neurons = 4_099
    target_neurons = 1_024
    observation_rows = 6
    neuron = torch.arange(source_neurons, dtype=dtype).unsqueeze(0)
    row = torch.arange(observation_rows, dtype=dtype).unsqueeze(1)
    gate = torch.sin(neuron * 0.013 + row * 0.37) + (neuron.remainder(17) * 0.01)
    up = torch.cos(neuron * 0.009 - row * 0.23) + (row + 1.0) * 0.025
    post = torch.nn.functional.silu(gate) * up
    contribution = torch.stack(
        (
            post * (1.0 + neuron.remainder(7) * 0.03),
            post * (0.5 + row * 0.04) + gate * 0.07,
        ),
        dim=2,
    )

    started = time.perf_counter()
    plan = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=target_neurons,
    )
    elapsed = time.perf_counter() - started
    repeated = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=target_neurons,
    )

    assert elapsed < 15.0
    assert plan == repeated
    assert plan.report.cluster_strategy == (
        "deterministic-normalized-projection-morton-balanced-exact-local-medoid"
    )
    assert plan.report.cluster_balanced_partition is True
    assert plan.report.cluster_locality_dimensions == 5
    assert plan.report.cluster_iterations == 1
    assert plan.report.cluster_converged is True
    assert plan.report.invariants.all_hold is True
    assert len(plan.source_to_target) == source_neurons
    assert len(set(plan.medoid_source_neurons)) == target_neurons
    assert set(plan.report.group_sizes) == {4, 5}
    assert max(plan.report.group_sizes) - min(plan.report.group_sizes) == 1
    assert sum(plan.report.group_sizes) == source_neurons

    assignment = torch.tensor(plan.source_to_target, dtype=torch.long)
    medoids = torch.tensor(plan.medoid_source_neurons, dtype=torch.long)
    features = _combined_modalities(
        gate.T.contiguous(),
        up.T.contiguous(),
        post.T.contiguous(),
        contribution.movedim(1, 0).reshape(source_neurons, -1).contiguous(),
    )
    for target, medoid in enumerate(medoids):
        members = torch.nonzero(assignment == target, as_tuple=False).flatten()
        assert bool((members == medoid).any().item())
        member_rows = features.index_select(0, members)
        pairwise = (member_rows[:, None, :] - member_rows[None, :, :]).square()
        objectives = pairwise.sum(dim=(1, 2))
        selected = int(torch.nonzero(members == medoid).item())
        torch.testing.assert_close(
            objectives[selected], objectives.min(), atol=1e-11, rtol=1e-11
        )

    post_signatures = post.T.contiguous()
    selected_post = post_signatures.index_select(0, medoids)
    basis = selected_post.index_select(0, assignment)
    selected_energy = selected_post.square().sum(dim=1)
    effective_ridge = 1e-8 * float(selected_energy.mean().item())
    expected_coefficients = (post_signatures * basis).sum(dim=1) / (
        selected_energy.index_select(0, assignment) + effective_ridge
    )
    expected_coefficients[medoids] = 1.0
    torch.testing.assert_close(
        torch.tensor(plan.consumer_coefficients, dtype=dtype),
        expected_coefficients,
        atol=1e-12,
        rtol=1e-12,
    )


def test_gqa_restricted_group_medoids_cover_all_sources() -> None:
    generator = torch.Generator().manual_seed(981)
    query = torch.randn(8, 12, 4, generator=generator, dtype=torch.float64)
    key = torch.randn(4, 12, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(4, 12, 4, generator=generator, dtype=torch.float64)
    context = torch.randn(8, 12, 4, generator=generator, dtype=torch.float64)
    contribution = torch.randn(8, 12, 3, generator=generator, dtype=torch.float64)

    plan = build_gqa_activation_cluster_plan(
        query,
        key,
        value,
        context,
        contribution,
        source_query_heads=8,
        source_key_value_heads=4,
        target_query_heads=4,
        target_key_value_heads=2,
        source_head_dim=4,
        target_head_dim=4,
        group_medoid_candidates=(2, 3),
    )

    assert set(plan.group_medoids) <= {2, 3}
    assert plan.report.group_medoid_candidate_indices == (2, 3)
    assert len(plan.group_assignment) == 4
    assert len(plan.query_head_assignment) == 8
    assert plan.report.full_source_coverage is True
    assert plan.report.invariants.all_hold is True


def test_gqa_plan_fails_closed_on_topology_shapes_and_nonfinite_values() -> None:
    query = torch.ones((4, 6), dtype=torch.float64)
    key = torch.ones((2, 6), dtype=torch.float64)
    value = torch.ones((2, 6), dtype=torch.float64)
    context = torch.ones((4, 6), dtype=torch.float64)
    contribution = torch.ones((4, 3), dtype=torch.float64)

    with pytest.raises(ActivationClusterShapeError, match="divide exactly"):
        build_gqa_activation_cluster_plan(
            query,
            key,
            value,
            context,
            contribution,
            target_query_heads=3,
            target_key_value_heads=2,
            head_dim=2,
        )
    with pytest.raises(ActivationClusterShapeError, match="identical"):
        build_gqa_activation_cluster_plan(
            query,
            key,
            value,
            context,
            contribution,
            target_query_heads=2,
            target_key_value_heads=1,
            source_head_dim=2,
            target_head_dim=4,
        )
    with pytest.raises(ActivationClusterShapeError, match="even head dimension"):
        build_gqa_activation_cluster_plan(
            query,
            key,
            value,
            context,
            contribution,
            target_query_heads=2,
            target_key_value_heads=1,
            head_dim=3,
        )
    with pytest.raises(ActivationClusterShapeError, match="first dimension 2"):
        build_gqa_activation_cluster_plan(
            query,
            key,
            torch.ones((3, 6), dtype=torch.float64),
            context,
            contribution,
            target_query_heads=2,
            target_key_value_heads=1,
            head_dim=2,
        )
    nonfinite = contribution.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(ActivationClusterFiniteError, match="NaN or infinity"):
        build_gqa_activation_cluster_plan(
            query,
            key,
            value,
            context,
            nonfinite,
            target_query_heads=2,
            target_key_value_heads=1,
            head_dim=2,
        )


def test_gated_mlp_plan_fails_closed_on_misaligned_rows() -> None:
    gate = torch.ones((5, 4), dtype=torch.float64)
    up = torch.ones((5, 4), dtype=torch.float64)
    post = torch.ones((5, 4), dtype=torch.float64)
    contribution = torch.ones((5, 4, 2), dtype=torch.float64)

    with pytest.raises(ActivationClusterShapeError, match="identical shapes"):
        build_gated_mlp_activation_cluster_plan(
            gate,
            up[:, :3],
            post,
            contribution,
            target_neurons=2,
        )
    with pytest.raises(ActivationClusterShapeError, match="must begin"):
        build_gated_mlp_activation_cluster_plan(
            gate,
            up,
            post,
            contribution[:, :3],
            target_neurons=2,
        )
    with pytest.raises(ActivationClusterShapeError, match="cannot exceed"):
        build_gated_mlp_activation_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=5,
        )
    nonfinite = post.clone()
    nonfinite[1, 1] = torch.inf
    with pytest.raises(ActivationClusterFiniteError, match="NaN or infinity"):
        build_gated_mlp_activation_cluster_plan(
            gate,
            up,
            nonfinite,
            contribution,
            target_neurons=2,
        )
