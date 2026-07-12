from __future__ import annotations

import numpy as np

from neural_forge.checkpoint import load_model, param_count_for_widths, random_toy_model, save_model
from neural_forge.surgery import (
    expand_depth_identity,
    expand_hidden_layer,
    merge_models,
    plan_widths_for_target_params,
    resize_to_widths,
    shrink_hidden_layer,
)


def test_net2wider_expansion_preserves_outputs_without_noise() -> None:
    model = random_toy_model([3, 4, 2], seed=7)
    expanded = expand_hidden_layer(model, 0, 9, noise_std=0.0)
    x = np.random.default_rng(11).normal(size=(16, 3))
    np.testing.assert_allclose(expanded.forward(x), model.forward(x), atol=1e-10)
    assert expanded.widths == [3, 9, 2]


def test_relu_identity_depth_expansion_preserves_outputs() -> None:
    model = random_toy_model([3, 4, 5, 2], seed=9)
    expanded = expand_depth_identity(model, insertions=2)
    x = np.random.default_rng(12).normal(size=(16, 3))
    np.testing.assert_allclose(expanded.forward(x), model.forward(x), atol=1e-10)
    assert expanded.hidden_depth == model.hidden_depth + 2


def test_shrink_hidden_layer_reduces_width_and_params() -> None:
    model = random_toy_model([5, 8, 4], seed=5)
    shrunk = shrink_hidden_layer(model, 0, 3)
    assert shrunk.widths == [5, 3, 4]
    assert shrunk.param_count < model.param_count
    assert shrunk.forward(np.zeros((2, 5))).shape == (2, 4)


def test_plan_widths_for_target_params_reaches_under_when_possible() -> None:
    model = random_toy_model([4, 10, 6, 3], seed=4)
    target = model.param_count - 35
    plan = plan_widths_for_target_params(model, target)
    assert abs(plan.achieved_params - target) <= 10
    assert param_count_for_widths(plan.widths) == plan.achieved_params


def test_plan_widths_for_target_params_prefers_closest_discrete_shape() -> None:
    model = random_toy_model([8, 13, 13, 4], seed=4)
    plan = plan_widths_for_target_params(model, 420)
    assert plan.widths == [8, 15, 14, 4]
    assert plan.achieved_params == 419


def test_resize_to_widths_can_shrink_and_expand_multiple_layers() -> None:
    model = random_toy_model([4, 5, 7, 2], seed=3)
    resized = resize_to_widths(model, [8, 3], seed=13, noise_std=0.0)
    assert resized.widths == [4, 8, 3, 2]
    assert resized.forward(np.zeros((2, 4))).shape == (2, 2)


def test_merge_different_width_models_into_target_shape() -> None:
    a = random_toy_model([3, 4, 2], seed=1)
    b = random_toy_model([3, 7, 2], seed=2)
    merged = merge_models([a, b], target_hidden_widths=[5], method="average", align=True)
    assert merged.widths == [3, 5, 2]
    assert merged.forward(np.zeros((6, 3))).shape == (6, 2)


def test_save_load_roundtrip(tmp_path) -> None:
    model = random_toy_model([2, 3, 1], seed=8)
    path = tmp_path / "model.npz"
    save_model(model, path)
    loaded = load_model(path)
    np.testing.assert_allclose(loaded.forward(np.ones((3, 2))), model.forward(np.ones((3, 2))))
