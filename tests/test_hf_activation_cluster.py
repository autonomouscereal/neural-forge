from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_activation_cluster import (  # noqa: E402
    ActivationClusterFiniteError,
    ActivationClusterShapeError,
    build_gated_mlp_activation_cluster_plan,
    build_gqa_activation_cluster_plan,
)


def _head_signatures(rows: torch.Tensor, head_axis: int) -> torch.Tensor:
    return rows.movedim(head_axis, 0).reshape(rows.shape[head_axis], -1)


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
