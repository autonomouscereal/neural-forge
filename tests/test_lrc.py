from __future__ import annotations

import pytest

from neural_forge.lrc import fit_linear_lrc_delta
from neural_forge.lrc import fit_materialized_linear_lrc_delta
from neural_forge.lrc import mean_row_cosine
from neural_forge.lrc import relative_reconstruction_error


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


def test_fit_materialized_linear_lrc_delta_accepts_low_rank_improvement(torch_module) -> None:
    torch = torch_module
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(96, 4, generator=generator)
    base_weight = torch.randn(3, 4, generator=generator) * 0.05
    left = torch.randn(3, 2, generator=generator)
    right = torch.randn(2, 4, generator=generator)
    true_delta = left.matmul(right) * 0.2
    targets = inputs.matmul((base_weight + true_delta).t())

    updated, report = fit_materialized_linear_lrc_delta(
        base_weight,
        inputs,
        targets,
        rank=2,
        factor_steps=40,
        factor_lr=0.05,
        validation_fraction=0.25,
        min_validation_rows=16,
        min_relative_improvement=0.90,
        max_cosine_drop=0.0,
    )

    before = inputs.matmul(base_weight.t())
    after = inputs.matmul(updated.t())
    assert report["accepted"] is True
    assert report["fit_gate"]["mode"] == "validation"
    assert report["rank"]["effective_rank"] == 2
    assert relative_reconstruction_error(after, targets) < relative_reconstruction_error(before, targets) * 0.1
    assert int(torch.linalg.matrix_rank((updated - base_weight).to(torch.float32))) <= 2


def test_fit_linear_lrc_delta_rejects_validation_regression_and_keeps_weight(torch_module) -> None:
    torch = torch_module
    linear = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
            [0.5, -0.5],
            [2.0, 2.0],
            [-2.0, 1.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [-3.0, 1.0],
            [1.0, -3.0],
        ]
    )
    train_targets = inputs[:8].matmul(torch.tensor([[2.0, 0.0], [0.0, -1.0]]).t())
    validation_targets = inputs[8:].matmul(linear.weight.detach().t())
    targets = torch.cat([train_targets, validation_targets], dim=0)
    before_weight = linear.weight.detach().clone()

    report = fit_linear_lrc_delta(
        linear,
        inputs,
        targets,
        rank=1,
        factor_steps=0,
        validation_fraction=1.0 / 3.0,
        min_validation_rows=4,
        min_relative_improvement=0.01,
        max_cosine_drop=0.0,
    )

    assert report["accepted"] is False
    assert report["mutation_skipped"] is True
    assert report["fit_gate"]["mode"] == "validation"
    assert report["reason"] == "relative_error_not_improved"
    torch.testing.assert_close(linear.weight, before_weight)


def test_fit_materialized_linear_lrc_delta_requires_validation_by_default(torch_module) -> None:
    torch = torch_module
    weight = torch.eye(2)
    inputs = torch.eye(2)
    targets = inputs.matmul((2.0 * weight).t())

    updated, report = fit_materialized_linear_lrc_delta(
        weight,
        inputs,
        targets,
        rank=1,
        validation_fraction=0.5,
        min_validation_rows=4,
    )

    assert report["accepted"] is False
    assert report["reason"] == "insufficient_validation_rows"
    torch.testing.assert_close(updated, weight)


def test_lrc_metrics_validate_shapes(torch_module) -> None:
    torch = torch_module
    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    assert relative_reconstruction_error(prediction, target) > 0.0
    assert mean_row_cosine(prediction, target) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same shape"):
        relative_reconstruction_error(prediction, target[:1])
