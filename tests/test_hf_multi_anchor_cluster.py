from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_multi_anchor_cluster import (  # noqa: E402
    MultiAnchorClusterFiniteError,
    MultiAnchorClusterShapeError,
    build_gated_mlp_multi_anchor_cluster_plan,
)
from neural_forge.hf_gated_mlp_weight_signature import (  # noqa: E402
    gated_mlp_weight_signature_tensor_sha256,
)


def _composed_rows() -> tuple[torch.Tensor, ...]:
    dtype = torch.float64
    x = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=dtype)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=dtype)
    post = torch.column_stack((x, x, y, y, x + y, 0.5 * x - 0.75 * y))
    gate = post.clone()
    up = torch.ones_like(post)
    consumer = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=dtype,
    )
    contribution = post.unsqueeze(2) * consumer.unsqueeze(0)
    heldout_x = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=dtype)
    heldout_y = torch.tensor([-0.5, 1.5, 2.0, -1.0], dtype=dtype)
    heldout = torch.column_stack(
        (
            heldout_x,
            heldout_x,
            heldout_y,
            heldout_y,
            heldout_x + heldout_y,
            0.5 * heldout_x - 0.75 * heldout_y,
        )
    )
    return gate, up, post, contribution, consumer, heldout


def test_multi_anchor_reconstructs_composed_paths_better_than_one_anchor() -> None:
    gate, up, post, contribution, consumer, heldout = _composed_rows()
    one = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=2,
        consumer_directions=consumer,
        consumer_axis=0,
        anchors_per_source=1,
    )
    multi = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=2,
        consumer_directions=consumer,
        consumer_axis=0,
        anchors_per_source=2,
        validation_post_swiglu_rows=heldout,
        calibration_provenance="calibration-a",
        heldout_provenance="heldout-b",
    )

    assert multi.medoid_source_neurons == one.medoid_source_neurons
    assert multi.report.post_swiglu_fit_relative_error < (
        one.report.post_swiglu_fit_relative_error * 1e-6
    )
    assert multi.report.consumer_weighted_fit_relative_error is not None
    assert multi.report.consumer_weighted_fit_relative_error < 1e-7
    assert multi.report.heldout_post_swiglu_fit_relative_error is not None
    assert multi.report.heldout_post_swiglu_fit_relative_error < 1e-7
    assert multi.report.heldout_ready is True
    assert multi.report.full_source_coverage is True
    assert multi.report.invariants.all_hold is True
    assert multi.report.materialized_full_source_target_matrix is False
    assert max(multi.report.anchor_counts) <= 2
    assert len(multi.report.coefficient_l1_per_source) == post.shape[1]
    assert multi.report.maximum_coefficient_l1 <= 64.0
    for source in range(multi.source_neurons):
        assert (
            multi.anchor_targets_for_source(source)[0] == multi.source_to_target[source]
        )
        assert (
            multi.candidate_targets_for_source(source)[0]
            == multi.source_to_target[source]
        )
    for target, source in enumerate(multi.medoid_source_neurons):
        assert multi.anchor_targets_for_source(source) == (target,)
        assert multi.coefficients_for_source(source) == (1.0,)


def test_sparse_index_add_matches_explicit_tiny_dense_transport() -> None:
    gate, up, post, contribution, consumer, _ = _composed_rows()
    plan = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=2,
        anchors_per_source=2,
    )
    generator = torch.Generator().manual_seed(904)
    down = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    dense = torch.zeros(6, 2, dtype=torch.float64)
    for source in range(6):
        for target, coefficient in zip(
            plan.anchor_targets_for_source(source),
            plan.coefficients_for_source(source),
            strict=True,
        ):
            dense[source, target] = coefficient

    actual = plan.aggregate_down_consumer(down, axis=1)
    torch.testing.assert_close(actual, down @ dense, atol=1e-12, rtol=1e-12)
    diagnostics = plan.evaluate_post_swiglu_reconstruction(
        post,
        consumer_directions=consumer,
        consumer_axis=0,
    )
    assert diagnostics.post_swiglu_relative_l2 < 1e-7
    assert diagnostics.consumer_output_relative_l2 is not None
    assert diagnostics.consumer_output_relative_l2 < 1e-7


def test_real_producers_are_copied_deterministically_without_gradients() -> None:
    gate, up, post, contribution, _, _ = _composed_rows()
    originals = tuple(value.clone() for value in (gate, up, post, contribution))
    first = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=2,
        anchors_per_source=2,
        similarity_chunk_size=1,
    )
    second = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=2,
        anchors_per_source=2,
        similarity_chunk_size=1,
    )
    assert first == second
    for value, original in zip((gate, up, post, contribution), originals, strict=True):
        assert torch.equal(value, original)

    gate_weight = torch.arange(24, dtype=torch.float64).reshape(6, 4)
    up_weight = -gate_weight.clone()
    gate_weight.requires_grad_(True)
    up_weight.requires_grad_(True)
    child_gate = first.select_gate_producer(gate_weight)
    child_up = first.select_up_producer(up_weight)
    assert child_gate.requires_grad is False
    assert child_up.requires_grad is False
    for target, source in enumerate(first.medoid_source_neurons):
        assert torch.equal(child_gate[target], gate_weight.detach()[source])
        assert torch.equal(child_up[target], up_weight.detach()[source])
    with pytest.raises(FrozenInstanceError):
        first.report.target_neurons = 3  # type: ignore[misc]


def test_consumer_directions_can_supply_bounded_contribution_signatures() -> None:
    gate, up, post, _, consumer, _ = _composed_rows()
    plan = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        target_neurons=2,
        consumer_directions=consumer.T,
        consumer_axis=1,
        anchors_per_source=2,
    )

    assert plan.report.consumer_weighted_fit_relative_error is not None
    assert plan.report.full_source_coverage is True
    assert plan.report.similarity_products <= plan.report.max_similarity_products
    assert plan.report.projection_fit_count <= plan.report.max_projection_fits
    assert plan.report.total_sparse_entries <= plan.report.max_sparse_entries
    assert plan.report.peak_workspace_elements <= plan.report.max_workspace_elements


def test_authenticated_structural_signature_replaces_trace_limited_planning() -> None:
    gate, up, post, contribution, consumer, _ = _composed_rows()
    structural = torch.tensor(
        [
            [1.0, 0.0, 0.1, 0.2],
            [0.9, 0.1, 0.0, 0.3],
            [0.0, 1.0, 0.2, 0.1],
            [0.1, 0.9, 0.3, 0.0],
            [0.6, 0.7, 0.4, 0.5],
            [-0.4, 0.8, 0.6, -0.2],
        ],
        dtype=torch.float64,
    )
    structural_sha256 = gated_mlp_weight_signature_tensor_sha256(structural)
    common = {
        "target_neurons": 2,
        "consumer_directions": consumer,
        "consumer_axis": 0,
        "anchors_per_source": 2,
        "structural_selection_signatures": structural,
        "structural_selection_signature_sha256": structural_sha256,
        "structural_selection_provenance": "full-weight-signature-report-sha256",
        "calibration_provenance": "cached-fit-diagnostics-only",
    }
    first = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        **common,
    )
    changed_gate = torch.flip(gate, dims=(0,)).contiguous() * 17.0
    changed_up = torch.flip(up, dims=(0,)).contiguous() * -9.0
    changed_post = torch.nn.functional.silu(changed_gate) * changed_up
    changed_contribution = changed_post.unsqueeze(2) * consumer.unsqueeze(0)
    second = build_gated_mlp_multi_anchor_cluster_plan(
        changed_gate,
        changed_up,
        changed_post,
        changed_contribution,
        **common,
    )

    assert second.medoid_source_neurons == first.medoid_source_neurons
    assert second.source_to_target == first.source_to_target
    assert second.source_anchor_offsets == first.source_anchor_offsets
    assert second.anchor_target_indices == first.anchor_target_indices
    assert second.anchor_coefficients == first.anchor_coefficients
    report = first.report.as_dict()
    assert report["planning_signature_source"] == (
        "full-weight-complete-structural-signature"
    )
    assert report["planning_signature_width"] == 4
    assert report["planning_signature_sha256"] == structural_sha256
    assert report["planning_signature_provenance"] == (
        "full-weight-signature-report-sha256"
    )
    assert report["cached_activation_rows_used_for_medoid_selection"] is False
    assert report["cached_activation_rows_used_for_sparse_anchor_fit"] is False
    assert report["calibration_provenance"] == "cached-fit-diagnostics-only"
    assert (
        report["medoid_cluster_report"]["medoid_selection_signature_source"]
        == "caller-supplied-source-major-structural-signature"
    )
    assert (
        report["calibration_diagnostics"]
        != second.report.as_dict()["calibration_diagnostics"]
    )

    with pytest.raises(MultiAnchorClusterShapeError, match="failed authentication"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            contribution,
            **{
                **common,
                "structural_selection_signature_sha256": "0" * 64,
            },
        )


def test_candidate_search_is_bounded_and_always_includes_primary_target() -> None:
    source_neurons = 96
    target_neurons = 48
    rows = 6
    generator = torch.Generator().manual_seed(447)
    gate = torch.randn(rows, source_neurons, generator=generator, dtype=torch.float64)
    up = torch.randn(rows, source_neurons, generator=generator, dtype=torch.float64)
    post = torch.nn.functional.silu(gate) * up
    contribution = post * torch.linspace(
        0.5, 1.5, source_neurons, dtype=torch.float64
    ).unsqueeze(0)
    plan = build_gated_mlp_multi_anchor_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=target_neurons,
        anchors_per_source=3,
        max_candidate_targets=5,
        similarity_chunk_size=2,
    )

    bounded_worst_case = (source_neurons - target_neurons) * 2 * 5
    exhaustive_worst_case = (source_neurons - target_neurons) * 2 * target_neurons
    assert plan.report.candidate_pool_size == 5
    assert plan.report.max_candidate_targets == 5
    assert plan.report.worst_case_similarity_products == bounded_worst_case
    assert plan.report.similarity_products <= bounded_worst_case
    assert bounded_worst_case < exhaustive_worst_case
    assert plan.report.peak_similarity_elements <= 2
    for source in range(source_neurons):
        candidates = plan.candidate_targets_for_source(source)
        if source in plan.medoid_source_neurons:
            assert len(candidates) == 1
        else:
            assert len(candidates) == 5
        assert candidates[0] == plan.source_to_target[source]
        assert plan.anchor_targets_for_source(source)[0] == candidates[0]
        assert set(plan.anchor_targets_for_source(source)) <= set(candidates)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("max_similarity_products", 1, "similarity work"),
        ("max_projection_fits", 1, "projection work"),
        ("max_sparse_entries", 2, "Sparse plan size"),
        ("max_workspace_elements", 1, "fitting memory"),
    ],
)
def test_hard_work_and_memory_bounds_fail_before_fitting(
    keyword: str, value: int, message: str
) -> None:
    gate, up, post, contribution, _, _ = _composed_rows()
    arguments = {keyword: value}
    with pytest.raises(MultiAnchorClusterShapeError, match=message):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=2,
            anchors_per_source=2,
            **arguments,
        )


def test_invalid_shapes_and_nonfinite_values_fail_closed() -> None:
    gate, up, post, contribution, consumer, heldout = _composed_rows()
    with pytest.raises(MultiAnchorClusterShapeError, match="identical shapes"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up[:, :-1],
            post,
            contribution,
            target_neurons=2,
        )
    with pytest.raises(MultiAnchorClusterShapeError, match="cannot exceed"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=7,
        )
    nonfinite = post.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(MultiAnchorClusterFiniteError, match="NaN or infinity"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            nonfinite,
            contribution,
            target_neurons=2,
        )
    with pytest.raises(MultiAnchorClusterShapeError, match="finite positive"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=2,
            ridge=0.0,
        )
    with pytest.raises(MultiAnchorClusterShapeError, match="hard cap"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=2,
            max_candidate_targets=65,
        )
    with pytest.raises(MultiAnchorClusterShapeError, match="source axis"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            target_neurons=2,
            consumer_directions=consumer[:-1],
            consumer_axis=0,
        )
    with pytest.raises(MultiAnchorClusterShapeError, match="shape"):
        build_gated_mlp_multi_anchor_cluster_plan(
            gate,
            up,
            post,
            contribution,
            target_neurons=2,
            heldout_post_swiglu_rows=heldout[:, :-1],
        )
