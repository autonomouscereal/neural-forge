from __future__ import annotations

import math
from dataclasses import replace

import pytest

from neural_forge import (
    hf_lfm2_moving_gauge_residual_reclosure as moving_gauge_reclosure,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBaseCausalEndpointChart,
    Lfm2MovingGaugeBoundaryEvidence,
    Lfm2MovingGaugeGeneralizedCausalEndpointChart,
    Lfm2MovingGaugeOrthogonalCausalEndpointChart,
    Lfm2MovingGaugeReclosurePolicy,
    Lfm2MovingGaugeResidualAuthenticationError,
    Lfm2MovingGaugeResidualReclosureError,
    Lfm2MovingGaugeWriterEvidence,
    bind_lfm2_base_e0_causal_endpoint_chart,
    bind_lfm2_generalized_e0a_causal_endpoint_chart,
    bind_lfm2_orthogonal_causal_endpoint_chart,
    compile_lfm2_moving_gauge_residual_reclosure,
    fit_lfm2_cycle_coupled_moving_gauge_boundary,
    fit_lfm2_generalized_moving_gauge_chart_diagnostic,
    fit_lfm2_moving_gauge_residual_chart_adapter,
    lfm2_moving_gauge_sequential_fit_provenance,
    solve_lfm2_trace_scaled_weighted_anchored_ridge,
    transport_lfm2_cycle_coupled_boundary_from_upstream,
    verify_lfm2_moving_gauge_report,
)


torch = pytest.importorskip("torch")


ALIGNMENT_SHA = "a" * 64
PARENT_SHA = "b" * 64
LIVE_CHILD_SHA = "c" * 64


def _direct_trace_scaled_anchored_solution(
    features: torch.Tensor,
    targets: torch.Tensor,
    anchor: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    features64 = features.to(torch.float64)
    targets64 = targets.to(torch.float64)
    anchor64 = anchor.to(torch.float64)
    weights64 = weights.to(torch.float64)
    root = torch.sqrt(weights64).unsqueeze(1)
    weighted_features = features64 * root
    weighted_targets = targets64 * root
    gram = weighted_features.T @ weighted_features
    right = weighted_features.T @ weighted_targets
    ridge = max(
        1e-12,
        1e-8 * float(torch.trace(gram).item()) / int(features.shape[1]),
    )
    system = gram + ridge * torch.eye(
        int(features.shape[1]),
        dtype=torch.float64,
    )
    factor = torch.linalg.cholesky(system)
    return torch.cholesky_solve(right + ridge * anchor64, factor)


def test_trace_scaled_weighted_anchor_matches_hand_computed_solution() -> None:
    features = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    targets = torch.tensor([[3.0], [5.0]], dtype=torch.float32)
    anchor = torch.tensor([[1.0]], dtype=torch.float32)
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    solution, report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights,
        output_dtype=torch.float32,
    )

    ridge = 9e-8
    expected = (23.0 + ridge) / (9.0 + ridge)
    assert float(solution.item()) == pytest.approx(expected, rel=1e-7)
    assert report["effective_ridge"] == pytest.approx(ridge, rel=0.0, abs=1e-20)
    assert report["weighted_gram_trace"] == pytest.approx(9.0)
    assert report["equation"] == ("V=(F^T D F+lambda I)^-1(F^T D Y+lambda V0)")
    assert report["rows"] == 2
    assert report["feature_width"] == 1
    assert report["output_width"] == 1
    assert report["accumulation_dtype"] == "float64"
    assert report["solve_dtype"] == "float64"
    assert report["output_dtype"] == "float32"
    assert report["output_dtype_cast_count"] == 1
    assert report["finite_checks"]["fp64_solution"] is True
    assert report["regularized_system_full_rank"] is True
    assert all(
        len(report[key]) == 64
        for key in (
            "feature_rows_source_sha256",
            "target_rows_source_sha256",
            "anchor_source_sha256",
            "row_weights_fp64_sha256",
            "fp64_solution_sha256",
            "output_solution_sha256",
        )
    )
    assert verify_lfm2_moving_gauge_report(report)


def test_trace_scaled_row_weights_can_equalize_records_with_unequal_rows() -> None:
    expanded_features = torch.ones(4, 1, dtype=torch.float64)
    expanded_targets = torch.tensor(
        [[0.0], [4.0], [4.0], [4.0]],
        dtype=torch.float64,
    )
    expanded_weights = torch.tensor(
        [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        dtype=torch.float64,
    )
    collapsed_features = torch.ones(2, 1, dtype=torch.float64)
    collapsed_targets = torch.tensor([[0.0], [4.0]], dtype=torch.float64)
    collapsed_weights = torch.ones(2, dtype=torch.float64)
    anchor = torch.zeros(1, 1, dtype=torch.float64)

    expanded, expanded_report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        expanded_features,
        expanded_targets,
        anchor,
        expanded_weights,
        output_dtype=torch.float32,
    )
    collapsed, _ = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        collapsed_features,
        collapsed_targets,
        anchor,
        collapsed_weights,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(expanded, collapsed, rtol=0.0, atol=0.0)
    assert expanded_report["row_weight_sum"] == pytest.approx(2.0)
    assert expanded_report["positive_weight_row_count"] == 4
    assert expanded_report["explicit_row_weights"] is True
    assert expanded_report["row_weights_frozen"] is True


def test_trace_scaled_row_weights_accept_zero_rows_but_reject_invalid_mass() -> None:
    features = torch.tensor([[1.0], [100.0]], dtype=torch.float64)
    targets = torch.tensor([[2.0], [999.0]], dtype=torch.float64)
    anchor = torch.zeros(1, 1, dtype=torch.float64)

    solution, report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        output_dtype=torch.float32,
    )
    assert float(solution.item()) == pytest.approx(2.0, rel=1e-7)
    assert report["zero_weight_row_count"] == 1
    assert report["positive_weight_row_count"] == 1

    with pytest.raises(
        Lfm2MovingGaugeResidualReclosureError,
        match="positive total mass",
    ):
        solve_lfm2_trace_scaled_weighted_anchored_ridge(
            features,
            targets,
            anchor,
            torch.zeros(2, dtype=torch.float64),
            output_dtype=torch.float32,
        )
    for invalid in (
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        torch.tensor([1.0, float("nan")], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64, requires_grad=True),
    ):
        with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
            solve_lfm2_trace_scaled_weighted_anchored_ridge(
                features,
                targets,
                anchor,
                invalid,
                output_dtype=torch.float32,
            )
    with pytest.raises(
        Lfm2MovingGaugeResidualReclosureError,
        match="total mass must be finite",
    ):
        solve_lfm2_trace_scaled_weighted_anchored_ridge(
            features,
            targets,
            anchor,
            torch.full(
                (2,),
                torch.finfo(torch.float64).max,
                dtype=torch.float64,
            ),
            output_dtype=torch.float32,
        )


def test_trace_scaled_solver_stays_primal_when_features_outnumber_rows() -> None:
    features = torch.tensor(
        [[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, -1.0, 2.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[3.0, -2.0], [1.0, 4.0]], dtype=torch.float64)
    anchor = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.5, 0.6],
            [0.7, -0.8],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([0.75, 1.25], dtype=torch.float64)

    solution, report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights,
        output_dtype=torch.float32,
    )
    expected64 = _direct_trace_scaled_anchored_solution(
        features,
        targets,
        anchor,
        weights,
    )

    torch.testing.assert_close(
        solution,
        expected64.to(torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert report["solver_path"] == ("primal_weighted_normal_equation_cholesky")
    assert report["primal_system"] is True
    assert report["dual_system"] is False
    assert report["pseudoinverse_used"] is False
    assert report["fallback_solve_used"] is False
    assert report["gram_rank"] is None
    assert report["gram_rank_computed"] is False
    assert report["regularized_system_rank"] == 4
    assert report["regularized_system_full_rank"] is True
    assert report["cholesky_info"] == 0


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_trace_scaled_solver_performs_one_requested_output_cast(
    output_dtype: torch.dtype,
) -> None:
    features = torch.tensor(
        [[0.25, 1.5], [2.0, -0.75], [1.25, 0.5]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]],
        dtype=torch.float32,
    )
    anchor = torch.tensor(
        [[0.125, -0.25], [0.5, 0.75]],
        dtype=torch.float32,
    )
    weights = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)

    solution, report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights,
        output_dtype=output_dtype,
    )
    expected64 = _direct_trace_scaled_anchored_solution(
        features,
        targets,
        anchor,
        weights,
    )

    assert solution.dtype == output_dtype
    torch.testing.assert_close(
        solution,
        expected64.to(output_dtype),
        rtol=0.0,
        atol=0.0,
    )
    assert report["output_dtype"] == str(output_dtype).removeprefix("torch.")
    assert report["output_dtype_cast_count"] == 1
    assert report["fp64_solution_sha256"] != report["output_solution_sha256"]


def test_trace_scaled_solution_is_invariant_to_global_weight_scale() -> None:
    features = torch.tensor(
        [[1.0, 0.5], [0.25, -2.0], [3.0, 1.5]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[2.0], [-1.0], [4.0]], dtype=torch.float64)
    anchor = torch.tensor([[0.75], [-0.25]], dtype=torch.float64)
    weights = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float64)

    baseline, baseline_report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights,
        output_dtype=torch.float32,
    )
    scaled, scaled_report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights * 8.0,
        output_dtype=torch.float32,
    )

    assert baseline_report["effective_ridge"] > 1e-12
    assert scaled_report["effective_ridge"] == pytest.approx(
        8.0 * baseline_report["effective_ridge"],
        rel=1e-15,
    )
    torch.testing.assert_close(baseline, scaled, rtol=0.0, atol=0.0)


def test_zero_feature_rows_use_absolute_floor_and_return_anchor() -> None:
    features = torch.zeros(3, 2, dtype=torch.float32)
    targets = torch.tensor([[1.0], [-4.0], [7.0]], dtype=torch.float32)
    anchor = torch.tensor([[0.375], [-1.25]], dtype=torch.float32)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    solution, report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(solution, anchor, rtol=0.0, atol=0.0)
    assert report["weighted_gram_trace"] == 0.0
    assert report["effective_ridge"] == 1e-12
    assert report["gram_rank"] is None
    assert report["gram_rank_computed"] is False
    assert report["regularized_system_rank"] == 2
    assert report["regularized_system_full_rank_proven_by_cholesky"] is True


@pytest.mark.parametrize(
    ("targets", "anchor", "weights"),
    [
        (
            torch.zeros(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, 1, dtype=torch.float32),
            torch.zeros(1, 1, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
        ),
    ],
)
def test_trace_scaled_solver_rejects_malformed_geometry(
    targets: torch.Tensor,
    anchor: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        solve_lfm2_trace_scaled_weighted_anchored_ridge(
            torch.zeros(3, 2, dtype=torch.float32),
            targets,
            anchor,
            weights,
            output_dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "output_dtype",
    [torch.float16, torch.float64, torch.int64, "float32"],
)
def test_trace_scaled_solver_rejects_unsupported_output_dtype(
    output_dtype: object,
) -> None:
    with pytest.raises(
        Lfm2MovingGaugeResidualReclosureError,
        match="output_dtype",
    ):
        solve_lfm2_trace_scaled_weighted_anchored_ridge(
            torch.ones(2, 1, dtype=torch.float32),
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(1, 1, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            output_dtype=output_dtype,
        )


@pytest.mark.parametrize("matrix_name", ["features", "targets", "anchor"])
def test_trace_scaled_solver_rejects_nonfinite_matrices(
    matrix_name: str,
) -> None:
    features = torch.ones(2, 1, dtype=torch.float64)
    targets = torch.ones(2, 1, dtype=torch.float64)
    anchor = torch.zeros(1, 1, dtype=torch.float64)
    values = {
        "features": features,
        "targets": targets,
        "anchor": anchor,
    }
    values[matrix_name][0, 0] = float("nan")

    with pytest.raises(
        Lfm2MovingGaugeResidualReclosureError,
        match="NaN or infinity",
    ):
        solve_lfm2_trace_scaled_weighted_anchored_ridge(
            values["features"],
            values["targets"],
            values["anchor"],
            torch.ones(2, dtype=torch.float64),
            output_dtype=torch.float32,
        )


def test_trace_scaled_receipt_truthfully_omits_cubic_rank_estimates() -> None:
    _, report = solve_lfm2_trace_scaled_weighted_anchored_ridge(
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0], [3.0]], dtype=torch.float64),
        torch.zeros(2, 1, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        output_dtype=torch.float32,
    )

    assert report["gram_rank"] is None
    assert report["gram_rank_computed"] is False
    assert report["gram_rank_reason"] == (
        "not_computed_to_avoid_full_o_d_cubed_decomposition"
    )
    assert report["rank_diagnostic_svd_calls"] == 0
    assert report["regularized_system_rank"] == 2
    assert report["regularized_system_rank_numerically_estimated"] is False
    assert report["regularized_system_rank_evidence"] == (
        "successful_fp64_cholesky_factorization"
    )
    assert report["regularized_system_full_rank"] is True
    assert report["regularized_system_full_rank_proven_by_cholesky"] is True
    assert verify_lfm2_moving_gauge_report(report)


def _rotation(angle: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=torch.float64,
    )


def _boundary(
    index: int,
    parent: torch.Tensor,
    child: torch.Tensor,
    *,
    role: str,
    locked: bool,
) -> Lfm2MovingGaugeBoundaryEvidence:
    return Lfm2MovingGaugeBoundaryEvidence(
        boundary_id=f"boundary:{index}",
        interface_role=role,
        interface_locked=locked,
        base_encoder=torch.eye(
            int(parent.shape[1]),
            int(child.shape[1]),
            dtype=torch.float64,
        ),
        fit_parent_rows=parent,
        fit_child_rows=child,
        fit_weights=torch.ones(int(parent.shape[0]), dtype=torch.float64),
        fit_provenance=f"cached-fit-boundary-{index}",
        row_alignment_sha256=ALIGNMENT_SHA,
        cached_parent_evidence_sha256=PARENT_SHA,
        live_child_recapture_sha256=LIVE_CHILD_SHA,
    )


def _writer(
    index: int,
    features: torch.Tensor,
    output_width: int,
) -> Lfm2MovingGaugeWriterEvidence:
    return Lfm2MovingGaugeWriterEvidence(
        parameter_id=f"model.layers.{index}.writer.weight",
        transition_index=index,
        seed_weight=torch.zeros(
            output_width,
            int(features.shape[1]),
            dtype=torch.float32,
        ),
        fit_input_rows=features,
        row_alignment_sha256=ALIGNMENT_SHA,
        live_child_recapture_sha256=LIVE_CHILD_SHA,
    )


def test_cycle_coupled_adapter_exposes_separate_maps_and_holds_interfaces() -> None:
    generator = torch.Generator().manual_seed(41)
    parent = torch.randn(18, 2, generator=generator, dtype=torch.float64)
    child = parent @ _rotation(0.7)
    chart = fit_lfm2_moving_gauge_residual_chart_adapter(
        boundary_id="layer:1:pre",
        interface_role="intermediate",
        interface_locked=False,
        base_producer_encoder=torch.eye(2, dtype=torch.float64),
        fit_parent_rows=parent,
        fit_live_child_rows=child,
        fit_weights=torch.ones(18, dtype=torch.float64),
        fit_provenance="cached-fit",
        row_alignment_sha256=ALIGNMENT_SHA,
        cached_parent_evidence_sha256=PARENT_SHA,
        live_child_recapture_sha256=LIVE_CHILD_SHA,
    )

    assert chart.verify_authentication()
    assert tuple(chart.encoder.shape) == (2, 2)
    assert tuple(chart.decoder.shape) == (2, 2)
    assert chart.report["separate_input_output_maps"] is True
    assert chart.report["cycle_consistency"]["method"] == (
        "closed_form_regularized_reciprocal_anchor"
    )
    assert chart.report["execution"]["training_steps"] == 0
    assert torch.max(torch.abs(parent @ chart.encoder - child)).item() < 1e-10

    locked = fit_lfm2_moving_gauge_residual_chart_adapter(
        boundary_id="input",
        interface_role="input",
        interface_locked=True,
        base_producer_encoder=torch.eye(2, dtype=torch.float64),
        fit_parent_rows=parent,
        fit_live_child_rows=child,
        fit_weights=torch.ones(18, dtype=torch.float64),
        fit_provenance="cached-fit",
        row_alignment_sha256=ALIGNMENT_SHA,
        cached_parent_evidence_sha256=PARENT_SHA,
        live_child_recapture_sha256=LIVE_CHILD_SHA,
    )
    torch.testing.assert_close(
        locked.encoder,
        torch.eye(2, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert locked.report["encoder_change_from_base"] == "none_interface_locked"


def _generalized_chart_evidence(
    *,
    parent: torch.Tensor,
    child: torch.Tensor,
    base_encoder: torch.Tensor,
    heldout_parent: torch.Tensor | None = None,
    heldout_child: torch.Tensor | None = None,
    heldout_provenance: str | None = None,
    interface_role: str = "intermediate",
    interface_locked: bool = False,
) -> Lfm2MovingGaugeBoundaryEvidence:
    return Lfm2MovingGaugeBoundaryEvidence(
        boundary_id=f"diagnostic-{interface_role}",
        interface_role=interface_role,
        interface_locked=interface_locked,
        base_encoder=base_encoder,
        fit_parent_rows=parent,
        fit_child_rows=child,
        fit_weights=torch.ones(int(parent.shape[0]), dtype=torch.float64),
        fit_provenance="generalized-fit",
        row_alignment_sha256=ALIGNMENT_SHA,
        cached_parent_evidence_sha256=PARENT_SHA,
        live_child_recapture_sha256=LIVE_CHILD_SHA,
        heldout_parent_rows=heldout_parent,
        heldout_child_rows=heldout_child,
        heldout_weights=(
            None
            if heldout_parent is None
            else torch.ones(int(heldout_parent.shape[0]), dtype=torch.float64)
        ),
        heldout_provenance=heldout_provenance,
    )


def _generalized_base_encoder() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )


def test_generalized_chart_diagnostic_keeps_orthogonal_rotation_control() -> None:
    generator = torch.Generator().manual_seed(701)
    parent = torch.randn(64, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        31,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    base = _generalized_base_encoder()
    rotation = _rotation(0.63)
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base @ rotation,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base @ rotation,
        heldout_provenance="generalized-heldout-rotation",
    )

    result = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)

    fit = result.report["diagnostics"]["fit"]
    assert result.verify_authentication()
    assert fit["orthogonal"]["relative_rmse"] < 1e-12
    assert fit["orthogonal"]["relative_rmse"] <= (
        fit["scale_shear"]["relative_rmse"] + 1e-12
    )
    assert fit["orthogonal"]["relative_rmse"] <= (
        fit["generalized"]["relative_rmse"] + 1e-12
    )
    assert (
        result.report["correction_decomposition"]["scale_shear"][
            "correction_frobenius_norm"
        ]
        < 1e-10
    )
    assert (
        result.report["correction_decomposition"]["generalized"][
            "correction_frobenius_norm"
        ]
        < 1e-10
    )
    assert result.report["orthogonal_default_replaced"] is False
    assert result.report["candidate_publication_authority"] is False
    assert result.report["execution"]["training_steps"] == 0
    assert "pre_gain_rms_direction_relative_rmse" in fit["orthogonal"]
    assert result.orthogonal_chart.report["heldout_evaluation_requested"] is False


def test_generalized_chart_scale_shear_candidate_fits_e0a_data() -> None:
    generator = torch.Generator().manual_seed(702)
    parent = torch.randn(80, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        37,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    base = _generalized_base_encoder()
    scale_shear = torch.tensor(
        [[1.45, 0.35], [-0.2, 0.68]],
        dtype=torch.float64,
    )
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base @ scale_shear,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base @ scale_shear,
        heldout_provenance="generalized-heldout-scale-shear",
    )

    result = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)

    fit = result.report["diagnostics"]["fit"]
    heldout = result.report["diagnostics"]["heldout"]
    assert fit["scale_shear"]["relative_rmse"] < (
        1e-5 * fit["orthogonal"]["relative_rmse"]
    )
    assert heldout["scale_shear"]["relative_rmse"] < (
        1e-5 * heldout["orthogonal"]["relative_rmse"]
    )
    assert (
        result.report["correction_decomposition"]["scale_shear"][
            "outside_base_source_subspace_frobenius_norm"
        ]
        < 1e-10
    )
    assert (
        result.report["gram_invariant_certificate"][
            "weighted_row_gram_difference_frobenius"
        ]
        > 0.0
    )


def test_generalized_e0a_candidate_binds_as_distinct_causal_endpoint() -> None:
    generator = torch.Generator().manual_seed(709)
    parent = torch.randn(56, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        27,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    base = _generalized_base_encoder()
    scale_shear = torch.tensor(
        [[1.3, 0.25], [-0.35, 0.72]],
        dtype=torch.float64,
    )
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base @ scale_shear,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base @ scale_shear,
        heldout_provenance="generalized-heldout-causal-endpoint",
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)

    endpoint = bind_lfm2_generalized_e0a_causal_endpoint_chart(
        diagnostic,
        evidence,
    )

    assert isinstance(endpoint, Lfm2MovingGaugeGeneralizedCausalEndpointChart)
    assert endpoint.verify_authentication()
    assert endpoint.report["selected_candidate_mode"] == "scale_shear_e0a"
    assert endpoint.report["orthogonal_chart_claimed"] is False
    assert endpoint.report["locked_chart_claimed"] is False
    assert (
        endpoint.report["generalized_diagnostic_sha256"]
        == (diagnostic.report["report_sha256"])
    )
    assert (
        endpoint.report["selected_candidate_encoder_sha256"]
        == (
            diagnostic.report["candidate_hashes_frozen_before_heldout_read"][
                "scale_shear"
            ]
        )
    )
    assert endpoint.report["heldout_read_after_candidate_freeze"] is True
    torch.testing.assert_close(
        endpoint.encoder,
        diagnostic.scale_shear_encoder,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.max(torch.abs(parent @ endpoint.encoder - evidence.fit_child_rows)) < (
        1e-5
    )


def test_generalized_e0a_endpoint_rejects_resealed_freeze_inventory() -> None:
    generator = torch.Generator().manual_seed(710)
    parent = torch.randn(44, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    body = dict(diagnostic.report)
    body.pop("report_sha256")
    frozen = dict(body["candidate_hashes_frozen_before_heldout_read"])
    frozen["scale_shear"] = "f" * 64
    body["candidate_hashes_frozen_before_heldout_read"] = frozen
    forged_report = moving_gauge_reclosure._seal(body)

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="recomputed FIT authority",
    ):
        moving_gauge_reclosure.Lfm2MovingGaugeGeneralizedChartDiagnostic(
            evidence_sha256=diagnostic.evidence_sha256,
            evidence=diagnostic.evidence,
            policy=diagnostic.policy,
            orthogonal_chart=diagnostic.orthogonal_chart,
            scale_shear_transform=diagnostic.scale_shear_transform,
            scale_shear_encoder=diagnostic.scale_shear_encoder,
            generalized_encoder=diagnostic.generalized_encoder,
            report=forged_report,
        )


def test_generalized_diagnostic_rejects_resealed_semantic_overclaims() -> None:
    generator = torch.Generator().manual_seed(719)
    parent = torch.randn(36, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    body = dict(diagnostic.report)
    body.pop("report_sha256")
    body.update(
        {
            "kind": "forged_generalized_authority",
            "candidate_publication_authority": True,
            "physical_jacobian_claimed": True,
            "runtime_sidecar_required": True,
            "heldout_read_after_candidate_freeze": True,
            "diagnostics": {
                "fit": {"attacker": "chosen"},
                "heldout": {"attacker": "chosen"},
                "all_metrics_diagnostic_only": False,
                "manual_semantic_review_remains_authoritative": False,
            },
        }
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="complete recomputed FIT authority",
    ):
        replace(diagnostic, report=moving_gauge_reclosure._seal(body))


def _reseal_generalized_e0a_candidate(
    diagnostic: object,
    transform: torch.Tensor,
    encoder: torch.Tensor,
) -> object:
    body = dict(diagnostic.report)  # type: ignore[attr-defined]
    body.pop("report_sha256")
    transform_sha = moving_gauge_reclosure.fingerprint_lfm2_moving_gauge_tensor(
        transform
    )
    encoder_sha = moving_gauge_reclosure.fingerprint_lfm2_moving_gauge_tensor(encoder)
    body["scale_shear_transform_sha256"] = transform_sha
    body["scale_shear_encoder_sha256"] = encoder_sha
    frozen = dict(body["candidate_hashes_frozen_before_heldout_read"])
    frozen["scale_shear_transform"] = transform_sha
    frozen["scale_shear"] = encoder_sha
    body["candidate_hashes_frozen_before_heldout_read"] = frozen
    return moving_gauge_reclosure._seal(body)


def test_generalized_diagnostic_rejects_self_consistent_arbitrary_e0a() -> None:
    generator = torch.Generator().manual_seed(711)
    parent = torch.randn(52, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    forged_transform = diagnostic.scale_shear_transform + 0.5 * torch.eye(
        2,
        dtype=torch.float64,
    )
    forged_encoder = base @ forged_transform
    forged_report = _reseal_generalized_e0a_candidate(
        diagnostic,
        forged_transform,
        forged_encoder,
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="do not reproduce from retained FIT evidence and policy",
    ):
        replace(
            diagnostic,
            scale_shear_transform=forged_transform,
            scale_shear_encoder=forged_encoder,
            report=forged_report,
        )


def test_generalized_diagnostic_rejects_heldout_derived_proper_e0a() -> None:
    generator = torch.Generator().manual_seed(712)
    parent = torch.randn(60, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(33, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    heldout_transform = torch.tensor(
        [[1.75, -0.4], [0.3, 0.55]],
        dtype=torch.float64,
    )
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base @ heldout_transform,
        heldout_provenance="heldout-derived-e0a-attack",
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    heldout_design = heldout_parent @ base
    derived_transform = torch.linalg.lstsq(
        heldout_design,
        evidence.heldout_child_rows,
    ).solution.contiguous()
    derived_encoder = (base @ derived_transform).contiguous()
    forged_report = _reseal_generalized_e0a_candidate(
        diagnostic,
        derived_transform,
        derived_encoder,
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="do not reproduce from retained FIT evidence and policy",
    ):
        replace(
            diagnostic,
            scale_shear_transform=derived_transform,
            scale_shear_encoder=derived_encoder,
            report=forged_report,
        )


@pytest.mark.parametrize("nested_kind", ["mapping", "chart"])
def test_generalized_endpoint_rejects_nested_cpcf_report_relabel(
    nested_kind: str,
) -> None:
    generator = torch.Generator().manual_seed(713)
    parent = torch.randn(46, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    endpoint = bind_lfm2_generalized_e0a_causal_endpoint_chart(
        diagnostic,
        evidence,
    )
    chart = endpoint.chart
    mapping = chart.mapping
    if nested_kind == "mapping":
        mapping_body = dict(mapping.report)
        mapping_body.pop("report_sha256")
        mapping_body["component"] = "forged_generalized_mapping"
        mapping_report = {
            **mapping_body,
            "report_sha256": moving_gauge_reclosure._sha256(mapping_body),
        }
        mapping = replace(mapping, report=mapping_report)
    chart_body = dict(chart.report)
    chart_body.pop("report_sha256")
    if nested_kind == "mapping":
        chart_body["predictive_mapping"] = mapping.as_dict()
    else:
        chart_body["component"] = "forged_generalized_chart"
    chart_report = {
        **chart_body,
        "report_sha256": moving_gauge_reclosure._sha256(chart_body),
    }
    forged_chart = replace(chart, mapping=mapping, report=chart_report)
    forged_endpoint_report = moving_gauge_reclosure._seal(
        moving_gauge_reclosure._generalized_e0a_causal_endpoint_body(
            diagnostic,
            forged_chart,
        )
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="nested CPCF chart does not reconstruct",
    ):
        replace(
            endpoint,
            chart=forged_chart,
            report=forged_endpoint_report,
        )


def test_generalized_authority_rejects_policy_and_evidence_substitution() -> None:
    generator = torch.Generator().manual_seed(714)
    parent = torch.randn(42, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
    )
    diagnostic = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    substituted_policy = replace(
        diagnostic.policy,
        chart_ridge_ratio=diagnostic.policy.chart_ridge_ratio * 100.0,
        authentication_sha256="",
    )
    policy_body = dict(diagnostic.report)
    policy_body.pop("report_sha256")
    policy_body["policy_sha256"] = substituted_policy.authentication_sha256
    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="do not reproduce|recomputed FIT authority",
    ):
        replace(
            diagnostic,
            policy=substituted_policy,
            report=moving_gauge_reclosure._seal(policy_body),
        )

    evidence_clone = replace(evidence)
    assert evidence_clone is not evidence
    assert evidence_clone.report == evidence.report
    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="evidence is not the authenticated",
    ):
        bind_lfm2_generalized_e0a_causal_endpoint_chart(
            diagnostic,
            evidence_clone,
        )


def _base_e0_endpoint_fixture() -> tuple[
    Lfm2MovingGaugeBoundaryEvidence,
    Lfm2MovingGaugeBaseCausalEndpointChart,
]:
    generator = torch.Generator().manual_seed(715)
    parent = torch.randn(54, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(25, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    child_transform = torch.tensor(
        [[1.4, 0.2], [-0.15, 0.7]],
        dtype=torch.float64,
    )
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base @ child_transform,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base @ child_transform,
        heldout_provenance="base-e0-heldout",
    )
    return evidence, bind_lfm2_base_e0_causal_endpoint_chart(evidence)


def test_base_e0_endpoint_is_bit_exact_and_distinct() -> None:
    evidence, endpoint = _base_e0_endpoint_fixture()

    assert isinstance(endpoint, Lfm2MovingGaugeBaseCausalEndpointChart)
    assert endpoint.verify_authentication()
    torch.testing.assert_close(
        endpoint.encoder,
        evidence.base_encoder,
        rtol=0.0,
        atol=0.0,
    )
    assert endpoint.report["selected_candidate_mode"] == "base_e0"
    assert endpoint.report["encoder_fit_performed"] is False
    assert endpoint.report["encoder_transform_applied"] is False
    assert endpoint.report["locked_chart_claimed"] is False
    assert endpoint.report["right_orthogonal_chart_claimed"] is False
    assert endpoint.report["generalized_chart_claimed"] is False
    assert endpoint.report["causal_endpoint_encoder_authority"] is True
    assert endpoint.report["physical_writer_publication_authority"] is False


def test_base_e0_endpoint_rejects_internally_resealed_encoder_substitution() -> None:
    evidence, endpoint = _base_e0_endpoint_fixture()
    forged_encoder = evidence.base_encoder.clone()
    forged_encoder[0, 0].add_(0.5)
    forged_sha = moving_gauge_reclosure.fingerprint_lfm2_moving_gauge_tensor(
        forged_encoder
    )
    lineage = moving_gauge_reclosure._base_e0_endpoint_lineage(
        evidence,
        endpoint.policy,
        endpoint.orthogonal_anchor,
    )
    lineage["selected_candidate_encoder_sha256"] = forged_sha
    forged_mapping = moving_gauge_reclosure._predictive_mapping(
        forged_encoder,
        endpoint.orthogonal_anchor.decoder,
        kind="base_e0_causal_endpoint_mapping",
        lineage=lineage,
    )
    forged_chart = moving_gauge_reclosure._cpcf_chart(
        evidence.boundary_id,
        forged_mapping,
        kind="base_e0_causal_endpoint_chart",
        lineage=lineage,
    )
    body = moving_gauge_reclosure._base_e0_causal_endpoint_body(
        evidence,
        endpoint.policy,
        endpoint.orthogonal_anchor,
        forged_chart,
    )
    body["selected_candidate_encoder_sha256"] = forged_sha
    body["base_encoder_sha256"] = forged_sha

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="nested CPCF chart does not reconstruct",
    ):
        replace(
            endpoint,
            chart=forged_chart,
            report=moving_gauge_reclosure._seal(body),
        )


@pytest.mark.parametrize("nested_kind", ["mapping", "chart"])
def test_base_e0_endpoint_rejects_nested_report_relabel(nested_kind: str) -> None:
    evidence, endpoint = _base_e0_endpoint_fixture()
    chart = endpoint.chart
    mapping = chart.mapping
    if nested_kind == "mapping":
        mapping_body = dict(mapping.report)
        mapping_body.pop("report_sha256")
        mapping_body["component"] = "forged_base_e0_mapping"
        mapping = replace(
            mapping,
            report={
                **mapping_body,
                "report_sha256": moving_gauge_reclosure._sha256(mapping_body),
            },
        )
    chart_body = dict(chart.report)
    chart_body.pop("report_sha256")
    if nested_kind == "mapping":
        chart_body["predictive_mapping"] = mapping.as_dict()
    else:
        chart_body["component"] = "forged_base_e0_chart"
    forged_chart = replace(
        chart,
        mapping=mapping,
        report={
            **chart_body,
            "report_sha256": moving_gauge_reclosure._sha256(chart_body),
        },
    )
    forged_report = moving_gauge_reclosure._seal(
        moving_gauge_reclosure._base_e0_causal_endpoint_body(
            evidence,
            endpoint.policy,
            endpoint.orthogonal_anchor,
            forged_chart,
        )
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="nested CPCF chart does not reconstruct",
    ):
        replace(endpoint, chart=forged_chart, report=forged_report)


def test_base_e0_endpoint_rejects_policy_substitution_and_locked_roles() -> None:
    evidence, endpoint = _base_e0_endpoint_fixture()
    policy = replace(
        endpoint.policy,
        chart_ridge_ratio=endpoint.policy.chart_ridge_ratio * 100.0,
        authentication_sha256="",
    )
    body = dict(endpoint.report)
    body.pop("report_sha256")
    body["policy_sha256"] = policy.authentication_sha256
    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="decoder anchor does not reproduce|report is stale",
    ):
        replace(
            endpoint,
            policy=policy,
            report=moving_gauge_reclosure._seal(body),
        )

    for role in ("input", "terminal"):
        locked = replace(
            evidence,
            boundary_id=f"base-e0-{role}",
            interface_role=role,
            interface_locked=True,
            report={},
        )
        with pytest.raises(
            Lfm2MovingGaugeResidualAuthenticationError,
            match="unlocked intermediate",
        ):
            bind_lfm2_base_e0_causal_endpoint_chart(locked)


def test_orthogonal_causal_endpoint_recursively_rejects_arbitrary_legacy_chart() -> None:
    evidence, _ = _base_e0_endpoint_fixture()
    endpoint = bind_lfm2_orthogonal_causal_endpoint_chart(evidence)
    assert isinstance(endpoint, Lfm2MovingGaugeOrthogonalCausalEndpointChart)
    legacy = endpoint.boundary_chart
    arbitrary_encoder = torch.tensor(
        [[9.0, -7.0], [5.0, 3.0], [-4.0, 11.0], [2.0, -6.0]],
        dtype=torch.float64,
    )
    forged_mapping = replace(
        legacy.chart.mapping,
        encoder=arbitrary_encoder,
        report={"attacker": "arbitrary_mapping"},
    )
    forged_chart = replace(
        legacy.chart,
        mapping=forged_mapping,
        report={"attacker": "arbitrary_chart"},
    )
    legacy_body = dict(legacy.report)
    legacy_body.pop("report_sha256")
    legacy_body["encoder_sha256"] = (
        moving_gauge_reclosure.fingerprint_lfm2_moving_gauge_tensor(
            arbitrary_encoder
        )
    )
    forged_legacy = replace(
        legacy,
        chart=forged_chart,
        report=moving_gauge_reclosure._seal(legacy_body),
    )
    # The historical container authenticates only its outer tensor hashes.
    assert forged_legacy.verify_authentication()

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="does not recursively reproduce",
    ):
        replace(endpoint, boundary_chart=forged_legacy)


def test_orthogonal_causal_endpoint_rejects_resealed_locked_role_relabel() -> None:
    evidence, _ = _base_e0_endpoint_fixture()
    endpoint = bind_lfm2_orthogonal_causal_endpoint_chart(evidence)
    relabeled_evidence = replace(
        evidence,
        boundary_id="layer.000.residual.pre",
        interface_role="input",
        interface_locked=True,
        fit_provenance="fit:forged-locked-input",
        report={},
    )
    legacy_body = dict(endpoint.boundary_chart.report)
    legacy_body.pop("report_sha256")
    legacy_body.update(
        {
            "boundary_id": relabeled_evidence.boundary_id,
            "interface_role": relabeled_evidence.interface_role,
            "interface_locked": relabeled_evidence.interface_locked,
            "evidence_sha256": relabeled_evidence.report["report_sha256"],
            "encoder_change_from_base": "none_interface_locked",
        }
    )
    relabeled_legacy = replace(
        endpoint.boundary_chart,
        evidence_sha256=relabeled_evidence.report["report_sha256"],
        report=moving_gauge_reclosure._seal(legacy_body),
    )
    assert relabeled_legacy.verify_authentication()
    forged_endpoint_report = moving_gauge_reclosure._seal(
        moving_gauge_reclosure._orthogonal_causal_endpoint_body(
            relabeled_evidence,
            endpoint.policy,
            relabeled_legacy,
        )
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="does not recursively reproduce",
    ):
        Lfm2MovingGaugeOrthogonalCausalEndpointChart(
            evidence_sha256=relabeled_evidence.report["report_sha256"],
            evidence=relabeled_evidence,
            policy=endpoint.policy,
            boundary_chart=relabeled_legacy,
            report=forged_endpoint_report,
        )


def test_generalized_chart_unrestricted_candidate_fits_source_subspace_drift() -> None:
    generator = torch.Generator().manual_seed(703)
    parent = torch.randn(96, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        41,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    base = _generalized_base_encoder()
    true_encoder = base.clone()
    true_encoder[2] = torch.tensor([0.65, -0.4], dtype=torch.float64)
    true_encoder[3] = torch.tensor([-0.25, 0.3], dtype=torch.float64)
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ true_encoder,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ true_encoder,
        heldout_provenance="generalized-heldout-source-drift",
    )

    result = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)

    fit = result.report["diagnostics"]["fit"]
    heldout = result.report["diagnostics"]["heldout"]
    assert fit["generalized"]["relative_rmse"] < (
        1e-5 * fit["scale_shear"]["relative_rmse"]
    )
    assert heldout["generalized"]["relative_rmse"] < (
        1e-5 * heldout["scale_shear"]["relative_rmse"]
    )
    decomposition = result.report["correction_decomposition"]["generalized"]
    assert decomposition["outside_base_source_subspace_frobenius_norm"] > 0.1
    assert (
        result.report["solver_reports"]["generalized"][
            "null_directions_preserve_anchor_analytically"
        ]
        is True
    )


def test_generalized_chart_solver_does_not_read_heldout_rows() -> None:
    generator = torch.Generator().manual_seed(704)
    parent = torch.randn(72, 4, generator=generator, dtype=torch.float64)
    child = torch.randn(72, 2, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        29,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    heldout_a = torch.randn(29, 2, generator=generator, dtype=torch.float64)
    heldout_b = 100.0 * torch.randn(
        29,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    common = {
        "parent": parent,
        "child": child,
        "base_encoder": _generalized_base_encoder(),
        "heldout_parent": heldout_parent,
    }
    first = fit_lfm2_generalized_moving_gauge_chart_diagnostic(
        _generalized_chart_evidence(
            **common,
            heldout_child=heldout_a,
            heldout_provenance="generalized-heldout-a",
        )
    )
    second = fit_lfm2_generalized_moving_gauge_chart_diagnostic(
        _generalized_chart_evidence(
            **common,
            heldout_child=heldout_b,
            heldout_provenance="generalized-heldout-b",
        )
    )

    torch.testing.assert_close(
        first.scale_shear_transform,
        second.scale_shear_transform,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.generalized_encoder,
        second.generalized_encoder,
        rtol=0.0,
        atol=0.0,
    )
    assert first.report["solver_reports"] == second.report["solver_reports"]
    assert first.report["heldout_used_by_solver"] is False
    assert second.report["heldout_used_by_solver"] is False
    assert (
        first.report["candidate_hashes_frozen_before_heldout_read"]
        == (second.report["candidate_hashes_frozen_before_heldout_read"])
    )


def test_generalized_chart_full_heldout_authentication_is_after_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(707)
    parent = torch.randn(48, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        23,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base,
        heldout_provenance="generalized-heldout-authentication-order",
    )
    state = {"candidate_hashes_frozen": False, "full_authentication_calls": 0}
    original_metrics = moving_gauge_reclosure._chart_candidate_metrics
    original_full_authentication = Lfm2MovingGaugeBoundaryEvidence.verify_authentication

    def mark_candidate_freeze(*args: object, **kwargs: object) -> object:
        state["candidate_hashes_frozen"] = True
        return original_metrics(*args, **kwargs)

    def guarded_full_authentication(
        self: Lfm2MovingGaugeBoundaryEvidence,
    ) -> bool:
        assert state["candidate_hashes_frozen"] is True
        state["full_authentication_calls"] += 1
        return original_full_authentication(self)

    monkeypatch.setattr(
        moving_gauge_reclosure,
        "_chart_candidate_metrics",
        mark_candidate_freeze,
    )
    monkeypatch.setattr(
        Lfm2MovingGaugeBoundaryEvidence,
        "verify_authentication",
        guarded_full_authentication,
    )

    result = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)

    assert result.report["heldout_read_after_candidate_freeze"] is True
    assert state["full_authentication_calls"] == 2


@pytest.mark.parametrize(
    ("rows", "expected_n_by_n"),
    ((3, True), (9, False)),
)
def test_generalized_chart_gram_receipt_reports_reduced_qr_shape(
    rows: int,
    expected_n_by_n: bool,
) -> None:
    generator = torch.Generator().manual_seed(708 + rows)
    left = torch.randn(rows, 2, generator=generator, dtype=torch.float64)
    right = torch.randn(rows, 2, generator=generator, dtype=torch.float64)
    weights = torch.linspace(0.5, 1.5, rows, dtype=torch.float64)

    receipt = moving_gauge_reclosure._weighted_row_gram_difference(
        left,
        right,
        weights,
    )

    root = torch.sqrt(weights).unsqueeze(1)
    weighted_left = left * root
    weighted_right = right * root
    explicit_difference = torch.linalg.matrix_norm(
        weighted_left @ weighted_left.T - weighted_right @ weighted_right.T
    ).item()
    assert math.isclose(
        receipt["weighted_row_gram_difference_frobenius"],
        explicit_difference,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert receipt["reduced_qr_dimension"] == min(rows, 4)
    assert receipt["materialized_square_dimension"] == min(rows, 4)
    assert receipt["n_by_n_row_gram_materialized"] is expected_n_by_n
    assert receipt["n_by_n_matrix_avoided"] is (not expected_n_by_n)


def test_generalized_chart_reauthenticates_heldout_and_nested_chart_tensors() -> None:
    generator = torch.Generator().manual_seed(706)
    parent = torch.randn(40, 4, generator=generator, dtype=torch.float64)
    heldout_parent = torch.randn(
        19,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base,
        heldout_provenance="generalized-heldout-authentication",
    )
    result = fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)
    assert result.verify_authentication()

    result.orthogonal_chart.encoder[0, 0].add_(0.25)
    assert result.verify_authentication() is False

    fresh_evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
        heldout_parent=heldout_parent,
        heldout_child=heldout_parent @ base,
        heldout_provenance="generalized-heldout-authentication-fresh",
    )
    fresh_evidence.heldout_child_rows[0, 0].add_(1.0)
    assert fresh_evidence.verify_fit_authentication() is True
    assert fresh_evidence.verify_authentication() is False
    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="Heldout boundary tensors changed after evidence authentication",
    ):
        fit_lfm2_generalized_moving_gauge_chart_diagnostic(fresh_evidence)


@pytest.mark.parametrize("role", ["input", "terminal"])
def test_generalized_chart_rejects_locked_interfaces(role: str) -> None:
    generator = torch.Generator().manual_seed(705)
    parent = torch.randn(24, 4, generator=generator, dtype=torch.float64)
    base = _generalized_base_encoder()
    evidence = _generalized_chart_evidence(
        parent=parent,
        child=parent @ base,
        base_encoder=base,
        interface_role=role,
        interface_locked=True,
    )

    with pytest.raises(
        Lfm2MovingGaugeResidualReclosureError,
        match="input and terminal interfaces remain locked",
    ):
        fit_lfm2_generalized_moving_gauge_chart_diagnostic(evidence)


def test_moving_gauges_reduce_endpoint_error_against_forced_identity() -> None:
    generator = torch.Generator().manual_seed(77)
    parents = tuple(
        torch.randn(24, 2, generator=generator, dtype=torch.float64) for _ in range(4)
    )
    rotations = (
        torch.eye(2, dtype=torch.float64),
        _rotation(0.9),
        _rotation(-0.65),
        torch.eye(2, dtype=torch.float64),
    )
    children = tuple(
        parent @ rotation for parent, rotation in zip(parents, rotations, strict=True)
    )
    boundaries = tuple(
        _boundary(
            index,
            parent,
            child,
            role=(
                "input"
                if index == 0
                else "terminal"
                if index == len(parents) - 1
                else "intermediate"
            ),
            locked=index in {0, len(parents) - 1},
        )
        for index, (parent, child) in enumerate(zip(parents, children, strict=True))
    )
    writers = tuple(
        _writer(
            index,
            torch.stack(
                (
                    torch.ones(24, dtype=torch.float64),
                    parents[index][:, 0],
                ),
                dim=1,
            ),
            2,
        )
        for index in range(3)
    )

    moving = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
        policy=Lfm2MovingGaugeReclosurePolicy(
            transition_mode="moving_gauge",
        ),
    )
    identity = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
        policy=Lfm2MovingGaugeReclosurePolicy(
            transition_mode="forced_identity_diagnostic",
        ),
    )

    moving_error = moving.report["diagnostics"]["reconciled_terminal_relative_rmse"]
    identity_error = identity.report["diagnostics"]["reconciled_terminal_relative_rmse"]
    assert moving_error < 0.5 * identity_error, (moving_error, identity_error)
    assert identity_error > 1e-3
    assert (
        moving.boundaries[1].report["sequential_transport_reused_mature_cpcf_primitive"]
        is True
    )
    assert all(transition.verify_authentication() for transition in moving.transitions)
    identity_transport_sha256 = moving.report["diagnostics"]["reverse_steps"][-1][
        "downstream_transport_sha256"
    ]
    assert any(
        not torch.allclose(
            transition.orthogonal_factor,
            torch.eye(int(transition.orthogonal_factor.shape[0]), dtype=torch.float64),
        )
        for transition in moving.transitions
    )
    assert all(
        step["downstream_transport_sha256"] == identity_transport_sha256
        and step["downstream_transport_kind"]
        == "exact_physical_identity_skip_under_frozen_writer_inputs"
        and step["chart_transition_used_as_physical_jacobian"] is False
        for step in moving.report["diagnostics"]["reverse_steps"]
    )
    assert (
        moving.report["diagnostics"]["chart_transition_used_as_physical_jacobian"]
        is False
    )
    for transition in moving.transitions:
        torch.testing.assert_close(
            transition.orthogonal_factor @ transition.positive_input_factor,
            transition.full_transition,
            rtol=1e-10,
            atol=1e-10,
        )
        assert transition.report["rmsnorm_crossing_factor"] == (
            "no_unconditional_factor_crossing_claim"
        )
        assert (
            transition.report[
                "rmsnorm_gain_commutation_must_be_authenticated_by_integrating_compiler"
            ]
            is True
        )
        assert (
            transition.report[
                "noncommuting_rmsnorm_gain_requires_adjacent_input_producer_absorption"
            ]
            is True
        )
        assert transition.report["square_rmsnorm_orthogonal_conjugacy_claim"] is False
        assert (
            transition.report["arbitrary_map_conjugacy_through_rmsnorm_claimed"]
            is False
        )


def test_incremental_and_whole_stack_charts_share_canonical_lineage() -> None:
    generator = torch.Generator().manual_seed(83)
    parents = tuple(
        torch.randn(16, 2, generator=generator, dtype=torch.float64) for _ in range(4)
    )
    rotations = (
        torch.eye(2, dtype=torch.float64),
        _rotation(0.35),
        _rotation(-0.55),
        torch.eye(2, dtype=torch.float64),
    )
    boundaries = tuple(
        _boundary(
            index,
            parent,
            parent @ rotations[index],
            role=(
                "input"
                if index == 0
                else "terminal"
                if index == len(parents) - 1
                else "intermediate"
            ),
            locked=index in {0, len(parents) - 1},
        )
        for index, parent in enumerate(parents)
    )
    initial = tuple(
        fit_lfm2_cycle_coupled_moving_gauge_boundary(value) for value in boundaries
    )
    incremental = [initial[0]]
    legacy_incremental = [initial[0]]
    for index in range(len(boundaries) - 1):
        kwargs = {
            "downstream_evidence": boundaries[index + 1],
            "fit_actual_child_upstream_rows": boundaries[index].fit_child_rows,
            "fit_weights": boundaries[index].fit_weights,
        }
        incremental.append(
            transport_lfm2_cycle_coupled_boundary_from_upstream(
                incremental[index],
                initial[index + 1],
                fit_provenance=lfm2_moving_gauge_sequential_fit_provenance(
                    boundaries[index],
                    boundaries[index + 1],
                ),
                **kwargs,
            )
        )
        legacy_incremental.append(
            transport_lfm2_cycle_coupled_boundary_from_upstream(
                legacy_incremental[index],
                initial[index + 1],
                fit_provenance=f"semantic-sequential:layer:{index + 1}:legacy",
                **kwargs,
            )
        )
    writers = tuple(
        _writer(index, torch.eye(16, dtype=torch.float64), 2) for index in range(3)
    )
    whole_stack = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
    )

    assert all(value.verify_authentication() for value in incremental)
    assert tuple(value.report["report_sha256"] for value in incremental) == tuple(
        value.report["report_sha256"] for value in whole_stack.boundaries
    )
    assert any(
        legacy.report["report_sha256"] != canonical.report["report_sha256"]
        for legacy, canonical in zip(
            legacy_incremental[1:-1],
            whole_stack.boundaries[1:-1],
            strict=True,
        )
    )
    for legacy, canonical in zip(
        legacy_incremental,
        whole_stack.boundaries,
        strict=True,
    ):
        torch.testing.assert_close(
            legacy.encoder, canonical.encoder, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            legacy.decoder, canonical.decoder, rtol=0.0, atol=0.0
        )


def test_forward_reverse_reclosure_emits_effectful_fp32_replacements() -> None:
    generator = torch.Generator().manual_seed(101)
    parent0 = torch.randn(10, 2, generator=generator, dtype=torch.float64)
    parent1 = torch.randn(10, 2, generator=generator, dtype=torch.float64)
    parent2 = torch.randn(10, 2, generator=generator, dtype=torch.float64)
    child0 = parent0.clone()
    child1 = parent1 @ _rotation(0.45)
    child2 = parent2 + torch.stack(
        (
            torch.linspace(-0.2, 0.2, 10, dtype=torch.float64),
            torch.linspace(0.15, -0.1, 10, dtype=torch.float64),
        ),
        dim=1,
    )
    boundaries = (
        _boundary(0, parent0, child0, role="input", locked=True),
        _boundary(1, parent1, child1, role="intermediate", locked=False),
        _boundary(2, parent2, child2, role="terminal", locked=True),
    )
    full_row_features = torch.eye(10, dtype=torch.float64)
    writers = (
        _writer(0, full_row_features, 2),
        _writer(1, full_row_features, 2),
    )

    result = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
    )

    assert result.verify_authentication()
    assert result.report["direct_target_dtype_dense_replacement"] is True
    assert result.report["analytic_small_additive_ray_materialized"] is False
    assert result.report["global_ray_multiplier"] is None
    assert result.report["changed_coordinate_count"] > 0
    assert result.report["changed_parameter_count"] > 0
    assert all(value.dtype == torch.float32 for value in result.replacement_weights)
    assert result.report["fold_only_guard"]["clean"] is True
    assert result.report["execution"]["parent_model_forward_calls"] == 0
    assert result.report["execution"]["backward_calls"] == 0
    assert result.report["execution"]["optimizer_steps"] == 0
    assert result.report["execution"]["parameters_pruned"] == 0
    assert result.report["execution"]["parameters_quantized"] == 0
    assert (
        result.report["diagnostics"]["reconciled_terminal_relative_rmse"]
        < result.report["diagnostics"]["forward_terminal_relative_rmse"]
    )


def test_mutated_evidence_fails_closed_before_reclosure() -> None:
    generator = torch.Generator().manual_seed(3)
    parent = torch.randn(8, 2, generator=generator, dtype=torch.float64)
    boundary0 = _boundary(0, parent, parent, role="input", locked=True)
    boundary1 = _boundary(1, parent, parent, role="terminal", locked=True)
    writer = _writer(0, torch.eye(8, dtype=torch.float64), 2)
    boundary1.fit_child_rows[0, 0] += 1.0

    assert boundary1.verify_authentication() is False
    with pytest.raises(Lfm2MovingGaugeResidualAuthenticationError):
        compile_lfm2_moving_gauge_residual_reclosure(
            (boundary0, boundary1),
            (writer,),
        )
