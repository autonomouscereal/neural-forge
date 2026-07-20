from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_cpcf_boundary_chart import (  # noqa: E402
    CpcfBoundaryChartError,
    fit_cpcf_boundary_chart,
    fit_cpcf_orthogonal_chart_transport,
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


def test_weighted_orthogonal_transport_aligns_actual_child_chart_invariantly():
    generator = torch.Generator().manual_seed(829)
    source_width = 7
    target_width = 3
    fit_parent = torch.randn(
        64,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    validation_parent = torch.randn(
        24,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    future_map = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    chart = fit_cpcf_boundary_chart(
        "layer.002.post_layer",
        fit_parent,
        fit_parent @ future_map,
        target_width=target_width,
        validation_parent_states=validation_parent,
        validation_future_coordinates=validation_parent @ future_map,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.99,
        minimum_validation_block_retention=0.99,
    )
    raw_rotation = torch.randn(
        target_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    expected_rotation, _ = torch.linalg.qr(raw_rotation)
    fit_actual = chart.encode(fit_parent) @ expected_rotation
    heldout_parent = torch.randn(
        17,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    heldout_actual = chart.encode(heldout_parent) @ expected_rotation
    fit_weights = torch.linspace(0.25, 2.0, fit_parent.shape[0])
    heldout_weights = torch.linspace(0.5, 1.5, heldout_parent.shape[0])

    transported = fit_cpcf_orthogonal_chart_transport(
        chart,
        fit_actual_child_rows=fit_actual,
        fit_provenance="fit-cache-shard-829",
        fit_parent_states=fit_parent,
        fit_weights=fit_weights,
        heldout_actual_child_rows=heldout_actual,
        heldout_provenance="heldout-cache-shard-829",
        heldout_parent_states=heldout_parent,
        heldout_weights=heldout_weights,
    )
    fit_only = fit_cpcf_orthogonal_chart_transport(
        chart,
        fit_actual_child_rows=fit_actual.clone(),
        fit_provenance="fit-cache-shard-829",
        fit_parent_states=fit_parent.clone(),
        fit_weights=fit_weights.clone(),
    )

    assert transported.chart is not None
    assert transported.chart.mapping is transported.mapping
    assert transported.report["fit_pre_alignment_rmse"] > 0.1
    assert transported.report["fit_post_alignment_rmse"] < 1e-12
    assert transported.report["required_write_rms_reduction"] > 0.1
    assert (
        transported.report["heldout_evaluation"]["post_alignment_rmse"]
        < 1e-12
    )
    assert transported.report["heldout_evaluation"]["used_by_solver"] is False
    assert transported.report["orthogonality_max_abs_error"] < 1e-12
    assert transported.report["decode_invariance_max_abs_error"] < 1e-12
    assert (
        transported.report["feature_predictor_invariance_max_abs_error"]
        < 1e-12
    )
    assert transported.report["training_steps"] == 0
    assert transported.report["optimizer_steps"] == 0
    assert transported.report["backward_calls"] == 0
    assert transported.report["parameters_pruned"] == 0
    assert transported.report["parameters_quantized"] == 0
    assert len(transported.report["report_sha256"]) == 64
    torch.testing.assert_close(
        transported.rotation,
        fit_only.rotation,
        atol=0.0,
        rtol=0.0,
    )
    assert (
        transported.report["tensor_sha256"]["rotation"]
        == fit_only.report["tensor_sha256"]["rotation"]
    )

    probe = torch.randn(
        13,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        transported.mapping.decode(transported.mapping.encode(probe)),
        chart.mapping.decode(chart.mapping.encode(probe)),
        atol=1e-12,
        rtol=1e-12,
    )
    assert set(transported.mapping.feature_predictors) == set(
        chart.mapping.feature_predictors
    )
    for name in chart.mapping.feature_predictors:
        torch.testing.assert_close(
            transported.mapping.predict_feature(name, probe),
            chart.mapping.predict_feature(name, probe),
            atol=1e-12,
            rtol=1e-12,
        )


def test_encoded_row_transport_is_deterministic_when_cross_moment_is_rank_deficient():
    generator = torch.Generator().manual_seed(839)
    fit_parent = torch.randn(
        40,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    validation_parent = torch.randn(
        18,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    future_map = torch.randn(
        6,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    chart = fit_cpcf_boundary_chart(
        "layer.003.post_operator",
        fit_parent,
        fit_parent @ future_map,
        target_width=3,
        validation_parent_states=validation_parent,
        validation_future_coordinates=validation_parent @ future_map,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.99,
        minimum_validation_block_retention=0.99,
    )
    encoded = chart.encode(fit_parent)
    encoded[:, 2] = 0.0
    actual = encoded.clone()

    first = fit_cpcf_orthogonal_chart_transport(
        chart.mapping,
        fit_actual_child_rows=actual,
        fit_provenance="rank-deficient-fit-839",
        fit_encoded_parent_rows=encoded,
    )
    second = fit_cpcf_orthogonal_chart_transport(
        chart.mapping,
        fit_actual_child_rows=actual.clone(),
        fit_provenance="rank-deficient-fit-839",
        fit_encoded_parent_rows=encoded.clone(),
    )

    assert first.chart is None
    assert first.report["effective_cross_covariance_rank"] == 2
    assert (
        first.report["tensor_sha256"]["rotation"]
        == second.report["tensor_sha256"]["rotation"]
    )
    assert first.report["report_sha256"] == second.report["report_sha256"]
    torch.testing.assert_close(
        first.rotation,
        second.rotation,
        atol=0.0,
        rtol=0.0,
    )
    assert first.report["fit_post_alignment_rmse"] <= (
        first.report["fit_pre_alignment_rmse"] + 1e-12
    )


def test_transport_rejects_leaky_heldout_and_invalid_fit_contracts():
    generator = torch.Generator().manual_seed(853)
    fit_parent = torch.randn(
        30,
        5,
        generator=generator,
        dtype=torch.float64,
    )
    validation_parent = torch.randn(
        12,
        5,
        generator=generator,
        dtype=torch.float64,
    )
    future_map = torch.randn(
        5,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    chart = fit_cpcf_boundary_chart(
        "layer.004.post_layer",
        fit_parent,
        fit_parent @ future_map,
        target_width=2,
        validation_parent_states=validation_parent,
        validation_future_coordinates=validation_parent @ future_map,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.99,
        minimum_validation_block_retention=0.99,
    )
    actual = chart.encode(fit_parent)

    with pytest.raises(CpcfBoundaryChartError, match="exactly one"):
        fit_cpcf_orthogonal_chart_transport(
            chart,
            fit_actual_child_rows=actual,
            fit_provenance="fit-853",
            fit_parent_states=fit_parent,
            fit_encoded_parent_rows=actual,
        )
    with pytest.raises(CpcfBoundaryChartError, match="positive"):
        fit_cpcf_orthogonal_chart_transport(
            chart,
            fit_actual_child_rows=actual,
            fit_provenance="fit-853",
            fit_parent_states=fit_parent,
            fit_weights=torch.zeros(fit_parent.shape[0]),
        )
    with pytest.raises(CpcfBoundaryChartError, match="identifiers must differ"):
        fit_cpcf_orthogonal_chart_transport(
            chart,
            fit_actual_child_rows=actual,
            fit_provenance="same-split",
            fit_parent_states=fit_parent,
            heldout_actual_child_rows=actual.clone(),
            heldout_provenance="same-split",
            heldout_parent_states=fit_parent.clone(),
        )
    with pytest.raises(CpcfBoundaryChartError, match="share tensor storage"):
        fit_cpcf_orthogonal_chart_transport(
            chart,
            fit_actual_child_rows=actual,
            fit_provenance="fit-853",
            fit_parent_states=fit_parent,
            heldout_actual_child_rows=actual.clone(),
            heldout_provenance="heldout-853",
            heldout_parent_states=fit_parent,
        )
