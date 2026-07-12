from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_parent_consensus import (  # noqa: E402
    ParentConsensusFiniteError,
    ParentConsensusShapeError,
    ParentConsensusWeightError,
    parent_activation_consensus,
    robust_parent_consensus,
    weighted_mean_parent_consensus,
)


def test_weighted_mean_is_exact_and_reports_explicit_contributions() -> None:
    first = torch.tensor([[0.0, 2.0], [4.0, 6.0]], dtype=torch.float64)
    second = torch.tensor([[8.0, 10.0], [12.0, 14.0]], dtype=torch.float64)

    result = weighted_mean_parent_consensus(
        [first, second],
        [1.0, 3.0],
        parent_names=["text_parent", "vision_parent"],
    )

    torch.testing.assert_close(result.consensus, first * 0.25 + second * 0.75)
    torch.testing.assert_close(
        result.effective_weights,
        torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.float64),
    )
    assert result.method == "weighted_mean"
    assert result.convex_hull_bounded is True
    assert result.bounded_influence is False
    assert result.parent_reports[0].parent_name == "text_parent"
    assert result.parent_reports[1].mean_contribution_fraction == pytest.approx(0.75)
    assert sum(
        report.mean_contribution_fraction for report in result.parent_reports
    ) == pytest.approx(1.0)
    assert result.as_dict()["parents"][1]["requested_weight"] == 3.0


def test_robust_consensus_bounds_outlier_influence_and_stays_in_hull() -> None:
    first = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float64)
    second = torch.tensor([[0.2, -0.2], [2.2, 1.8]], dtype=torch.float64)
    outlier = torch.tensor([[100.0, -100.0], [80.0, -90.0]], dtype=torch.float64)
    mean = weighted_mean_parent_consensus([first, second, outlier], [1.0, 1.0, 1.0])

    robust = robust_parent_consensus(
        [first, second, outlier],
        [1.0, 1.0, 1.0],
        parent_names=["a", "b", "outlier"],
        robust_cutoff=1.5,
    )

    inlier_center = (first + second) * 0.5
    assert torch.linalg.vector_norm(robust.consensus - inlier_center) < 1.0
    assert torch.linalg.vector_norm(robust.consensus - inlier_center) < torch.linalg.vector_norm(
        mean.consensus - inlier_center
    )
    stacked = torch.stack([first, second, outlier])
    assert bool((robust.consensus >= stacked.amin(dim=0)).all())
    assert bool((robust.consensus <= stacked.amax(dim=0)).all())
    report = robust.parent_reports[2]
    assert report.robust_multiplier_mean < 0.01
    assert report.mean_contribution_fraction < 0.01
    assert report.suppressed_row_fraction == 1.0
    assert report.disagreement_rms > robust.parent_reports[0].disagreement_rms
    assert robust.bounded_influence is True
    assert robust.converged is True


def test_robust_weights_are_row_specific_not_global_parent_rejection() -> None:
    first = torch.tensor([[0.0, 0.0], [5.0, 5.0]], dtype=torch.float64)
    second = torch.tensor([[0.1, -0.1], [5.0, 5.0]], dtype=torch.float64)
    sometimes_bad = torch.tensor([[50.0, -50.0], [5.0, 5.0]], dtype=torch.float64)

    result = robust_parent_consensus(
        [first, second, sometimes_bad],
        [1.0, 1.0, 1.0],
    )

    assert result.effective_weights[2, 0] < 0.01
    assert result.effective_weights[2, 1] == pytest.approx(1.0 / 3.0)
    torch.testing.assert_close(
        result.consensus[1],
        torch.tensor([5.0, 5.0], dtype=torch.float64),
    )
    assert result.parent_reports[2].minimum_contribution_fraction < 0.01
    assert result.parent_reports[2].maximum_contribution_fraction == pytest.approx(
        1.0 / 3.0
    )


def test_equal_two_parent_robust_consensus_is_order_invariant() -> None:
    low = torch.tensor([[0.0, -2.0]], dtype=torch.float64)
    high = torch.tensor([[10.0, 4.0]], dtype=torch.float64)

    forward = robust_parent_consensus([low, high], [1.0, 1.0])
    reverse = robust_parent_consensus([high, low], [1.0, 1.0])

    torch.testing.assert_close(
        forward.consensus,
        torch.tensor([[5.0, 1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(forward.consensus, reverse.consensus)


def test_consensus_is_deterministic_detached_and_does_not_mutate_inputs() -> None:
    generator = torch.Generator().manual_seed(71)
    parents = [
        torch.randn(16, 7, generator=generator, dtype=torch.float32).requires_grad_()
        for _ in range(4)
    ]
    snapshots = [value.detach().clone() for value in parents]

    first = parent_activation_consensus(parents, [4.0, 3.0, 2.0, 1.0])
    second = parent_activation_consensus(parents, torch.tensor([4.0, 3.0, 2.0, 1.0]))

    assert torch.equal(first.consensus, second.consensus)
    assert torch.equal(first.effective_weights, second.effective_weights)
    assert first.consensus.requires_grad is False
    assert first.consensus.grad_fn is None
    assert first.deterministic is True and first.no_training is True
    assert all(value.grad is None for value in parents)
    for value, snapshot in zip(parents, snapshots, strict=True):
        torch.testing.assert_close(value.detach(), snapshot)


@pytest.mark.parametrize(
    ("parents", "message"),
    [
        ([torch.ones(2, 3)], "At least two"),
        ([torch.ones(2, 3), torch.ones(2, 4)], "same aligned shape"),
        ([torch.ones(2, 3), torch.ones(2, 3, 1)], r"\[rows, features\]"),
        ([torch.ones(0, 3), torch.ones(0, 3)], "empty dimension"),
        ([torch.ones(2, 3), torch.ones(2, 3, dtype=torch.int64)], "floating-point"),
        ([torch.ones(2, 3), "bad"], "torch.Tensor"),
    ],
)
def test_parent_shape_validation_fails_closed(parents, message: str) -> None:
    with pytest.raises(ParentConsensusShapeError, match=message):
        parent_activation_consensus(parents, [1.0, 1.0])


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_parent_rows_fail_closed(nonfinite: float) -> None:
    bad = torch.ones(2, 3)
    bad[0, 1] = nonfinite
    with pytest.raises(ParentConsensusFiniteError, match="NaN or infinity"):
        parent_activation_consensus([torch.ones(2, 3), bad], [1.0, 1.0])


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, float("nan")],
        [True, 1.0],
        torch.ones(2, 1),
        torch.ones(2, dtype=torch.int64),
    ],
)
def test_parent_weight_validation_fails_closed(weights) -> None:
    parents = [torch.ones(2, 3), torch.zeros(2, 3)]
    with pytest.raises(ParentConsensusWeightError):
        parent_activation_consensus(parents, weights)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "median"}, "method"),
        ({"robust_cutoff": 0.0}, "robust_cutoff"),
        ({"scale_floor": float("nan")}, "scale_floor"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"parent_names": ["only_one"]}, "exactly 2"),
        ({"parent_names": ["same", "same"]}, "unique"),
    ],
)
def test_algorithm_and_name_validation_fails_closed(kwargs, message: str) -> None:
    parents = [torch.ones(2, 3), torch.zeros(2, 3)]
    with pytest.raises((ParentConsensusShapeError, ParentConsensusWeightError), match=message):
        parent_activation_consensus(parents, [1.0, 1.0], **kwargs)


def test_mixed_floating_dtypes_promote_and_remain_finite() -> None:
    first = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    second = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    result = robust_parent_consensus([first, second], [1.0, 1.0])

    assert result.consensus.dtype == torch.float32
    assert bool(torch.isfinite(result.consensus).all())
    torch.testing.assert_close(result.consensus, torch.tensor([[2.0, 3.0]]))
