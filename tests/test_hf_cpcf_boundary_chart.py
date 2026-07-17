from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_cpcf_boundary_chart import (  # noqa: E402
    fit_cpcf_boundary_chart,
)


def test_independent_chart_absorbs_coordinate_change_into_residual_write():
    generator = torch.Generator().manual_seed(811)
    source_width = 8
    target_width = 3
    future_map = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    fit = torch.randn(
        48,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    validation = torch.randn(
        21,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    chart = fit_cpcf_boundary_chart(
        "layer.000.post_operator",
        fit,
        fit @ future_map,
        target_width=target_width,
        validation_parent_states=validation,
        validation_future_coordinates=validation @ future_map,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.99,
        minimum_validation_block_retention=0.99,
    )

    parent_endpoint = validation[:6].reshape(2, 3, source_width)
    actual_child_pre = torch.randn(
        2,
        3,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    endpoint, write = chart.residual_write_target(
        actual_child_pre_state=actual_child_pre,
        parent_endpoint=parent_endpoint,
    )

    torch.testing.assert_close(actual_child_pre + write, endpoint)
    torch.testing.assert_close(
        endpoint,
        chart.encode_shaped(parent_endpoint),
    )
    assert chart.report["identity_skip_path_accounted_for"] is True
    assert chart.report["independent_boundary_coordinates"] is True
    assert (
        chart.report["chart_transition_residual_formula"]
        == "encoded_parent_endpoint_minus_actual_child_pre_state"
    )
    assert chart.report["training_steps"] == 0


def test_chart_uses_disjoint_heldout_future_coordinates():
    generator = torch.Generator().manual_seed(823)
    fit = torch.randn(30, 7, generator=generator)
    validation = torch.randn(12, 7, generator=generator)
    projection = torch.randn(7, 2, generator=generator)

    chart = fit_cpcf_boundary_chart(
        "layer.001.post_layer",
        fit,
        fit @ projection,
        target_width=2,
        validation_parent_states=validation,
        validation_future_coordinates=validation @ projection,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.95,
        minimum_validation_block_retention=0.95,
    )

    assert chart.source_width == 7
    assert chart.target_width == 2
    assert chart.report["fit_rows"] == 30
    assert chart.report["validation_rows"] == 12
    assert chart.report["predictive_mapping"][
        "minimum_observed_validation_block_retention"
    ] > 0.99
