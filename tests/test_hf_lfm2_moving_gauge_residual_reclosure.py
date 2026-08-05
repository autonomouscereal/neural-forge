from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from neural_forge import (
    hf_lfm2_moving_gauge_residual_reclosure as moving_gauge_reclosure,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBaseCausalEndpointChart,
    Lfm2MovingGaugeBoundaryEvidence,
    Lfm2MovingGaugeGeneralizedCausalEndpointChart,
    Lfm2MovingGaugeOrthogonalCausalEndpointChart,
    Lfm2MovingGaugeRecapturedChildState,
    Lfm2MovingGaugeReclosurePolicy,
    Lfm2MovingGaugeResidualAuthenticationError,
    Lfm2MovingGaugeResidualReclosureError,
    Lfm2MovingGaugeWriterEvidence,
    PreparedAnchoredWriterProblem,
    bind_lfm2_base_e0_causal_endpoint_chart,
    bind_lfm2_generalized_e0a_causal_endpoint_chart,
    bind_lfm2_orthogonal_causal_endpoint_chart,
    compile_lfm2_moving_gauge_residual_reclosure,
    fit_lfm2_cycle_coupled_moving_gauge_boundary,
    fit_lfm2_generalized_moving_gauge_chart_diagnostic,
    fit_lfm2_moving_gauge_residual_chart_adapter,
    lfm2_moving_gauge_sequential_fit_provenance,
    prepare_lfm2_moving_gauge_anchored_writer_problem,
    solve_lfm2_moving_gauge_anchored_writer_replacement,
    solve_lfm2_prepared_anchored_writer_replacement,
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


@pytest.mark.parametrize("seed_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "input_width", "expected_solve_form"),
    [(9, 4, "primal"), (3, 5, "dual")],
)
def test_prepared_writer_solve_is_byte_exact_for_every_supported_path(
    seed_dtype: torch.dtype,
    rows: int,
    input_width: int,
    expected_solve_form: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(871 + rows + input_width)
    output_width = 3
    design = torch.randn(
        rows,
        input_width,
        generator=generator,
        dtype=torch.float32,
    )
    seed = torch.randn(
        output_width,
        input_width,
        generator=generator,
        dtype=torch.float32,
    ).to(seed_dtype)
    weights = torch.rand(rows, generator=generator, dtype=torch.float32) + 0.25
    targets = tuple(
        torch.randn(
            rows,
            output_width,
            generator=generator,
            dtype=torch.float32,
        )
        for _ in range(3)
    )
    baselines = tuple(
        solve_lfm2_moving_gauge_anchored_writer_replacement(
            design,
            target,
            seed,
            weights,
        )
        for target in targets
    )

    ridge_scale_calls = 0
    solve_calls = 0
    original_ridge_scale = moving_gauge_reclosure._ridge_scale
    original_solve = torch.linalg.solve

    def counted_ridge_scale(*args: object, **kwargs: object) -> float:
        nonlocal ridge_scale_calls
        ridge_scale_calls += 1
        return original_ridge_scale(*args, **kwargs)

    def counted_solve(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal solve_calls
        solve_calls += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(
        moving_gauge_reclosure,
        "_ridge_scale",
        counted_ridge_scale,
    )
    monkeypatch.setattr(torch.linalg, "solve", counted_solve)

    prepared = prepare_lfm2_moving_gauge_anchored_writer_problem(
        design,
        seed,
        weights,
    )
    actual = tuple(
        solve_lfm2_prepared_anchored_writer_replacement(prepared, target)
        for target in targets
    )

    assert isinstance(prepared, PreparedAnchoredWriterProblem)
    assert prepared.verify_authentication()
    assert prepared.solve_form == expected_solve_form
    assert prepared.report["linear_factorization_precomputed"] is False
    assert prepared.report["target_rows_consumed"] is False
    assert ridge_scale_calls == 1
    # Each target still executes the legacy unfactored solve.
    assert solve_calls == len(targets)
    for (expected_tensor, expected_report), (tensor, report) in zip(
        baselines,
        actual,
        strict=True,
    ):
        assert torch.equal(tensor, expected_tensor)
        assert dict(report) == dict(expected_report)


def test_prepared_writer_owns_sources_and_is_structurally_frozen() -> None:
    generator = torch.Generator().manual_seed(993)
    design = torch.randn(7, 3, generator=generator, dtype=torch.float32)
    target = torch.randn(7, 2, generator=generator, dtype=torch.float32)
    seed = torch.randn(2, 3, generator=generator, dtype=torch.float32)
    weights = torch.rand(7, generator=generator, dtype=torch.float32) + 0.5
    expected, expected_report = solve_lfm2_moving_gauge_anchored_writer_replacement(
        design.clone(),
        target,
        seed.clone(),
        weights.clone(),
    )
    prepared = prepare_lfm2_moving_gauge_anchored_writer_problem(
        design,
        seed,
        weights,
    )

    design.add_(100.0)
    seed.zero_()
    weights.mul_(4.0)
    actual, report = solve_lfm2_prepared_anchored_writer_replacement(
        prepared,
        target,
    )

    assert torch.equal(actual, expected)
    assert dict(report) == dict(expected_report)
    with pytest.raises(FrozenInstanceError):
        prepared.effective_ridge = 7.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        prepared.report["effective_ridge"] = 7.0  # type: ignore[index]


@pytest.mark.parametrize(
    "tensor_field",
    [
        "_design_fp64",
        "_seed_weight",
        "_anchor_columns_fp64",
        "_seed_prediction_fp64",
        "_fit_weights_fp64",
        "_root_weights_fp64",
        "_weighted_design_fp64",
        "_regularized_system_fp64",
    ],
)
def test_prepared_writer_rejects_exact_tensor_tampering(
    tensor_field: str,
) -> None:
    design = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    target = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 7
    seed = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 5
    weights = torch.arange(1, 9, dtype=torch.float32)
    prepared = prepare_lfm2_moving_gauge_anchored_writer_problem(
        design,
        seed,
        weights,
    )

    retained = getattr(prepared, tensor_field)
    retained.reshape(-1)[0].add_(1.0)

    assert prepared.verify_authentication() is False
    with pytest.raises(
        Lfm2MovingGaugeResidualAuthenticationError,
        match="failed exact authentication",
    ):
        solve_lfm2_prepared_anchored_writer_replacement(prepared, target)


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


TOY_WIDTH = 3
TOY_LAYERS = 3
TOY_RECAPTURE_ALIGNMENT_SHA = "d" * 64
TOY_CHILD_STATE_SHA = "e" * 64
TOY_CAPTURE_SHA = "f" * 64
# Pinned at the commit that introduced the recaptured reverse sweep: every
# reclosure sealed before it keeps this exact forward-only policy digest.
FORWARD_ONLY_POLICY_SHA256 = (
    "391318199830753ee60399a0725b7d774be5027760d62741b0ed82347f14eb55"
)


def _toy_phi(value: torch.Tensor) -> torch.Tensor:
    return torch.tanh(value)


class _ToyChild:
    """Residual stack whose writer inputs and terminal both move with a writer.

    Each layer writes ``phi(state @ reader) @ writer.T``, so publishing a new
    writer changes every later layer's input rows (defeating the frozen-writer
    -input approximation) and reaches the terminal through a nonlinearity
    (defeating the identity-transport approximation).
    """

    def __init__(self, readers: tuple[torch.Tensor, ...], writers) -> None:
        self.readers = readers
        self.writers = list(writers)

    @classmethod
    def seeded(cls, generator: torch.Generator) -> "_ToyChild":
        readers = tuple(
            torch.randn(
                TOY_WIDTH,
                TOY_WIDTH,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.8
            for _ in range(TOY_LAYERS)
        )
        writers = [
            torch.zeros(TOY_WIDTH, TOY_WIDTH, dtype=torch.float32)
            for _ in range(TOY_LAYERS)
        ]
        return cls(readers, writers)

    def run(self, entry: torch.Tensor):
        state = entry.clone()
        boundary_rows = [state.clone()]
        writer_inputs = []
        for index in range(TOY_LAYERS):
            features = _toy_phi(state @ self.readers[index])
            writer_inputs.append(features.clone())
            state = state + features @ self.writers[index].to(dtype=torch.float64).T
            boundary_rows.append(state.clone())
        return tuple(boundary_rows), tuple(writer_inputs)


class _ToyParent:
    """One fixed nonlinear parent stack; FIT and heldout share its weights."""

    def __init__(self, generator: torch.Generator) -> None:
        self.operators = tuple(
            torch.randn(
                TOY_WIDTH,
                TOY_WIDTH,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.6
            for _ in range(TOY_LAYERS)
        )

    def run(self, entry: torch.Tensor):
        state = entry.clone()
        series = [state.clone()]
        for operator in self.operators:
            state = state + _toy_phi(state @ operator) * 0.9
            series.append(state.clone())
        return tuple(series)


def _toy_parameter_id(index: int) -> str:
    return f"toy.layers.{index}.writer.weight"


def _toy_boundaries(parents, children):
    rows = int(parents[0].shape[0])
    return tuple(
        Lfm2MovingGaugeBoundaryEvidence(
            boundary_id=f"toy:boundary:{index}",
            interface_role=(
                "input"
                if index == 0
                else "terminal"
                if index == TOY_LAYERS
                else "intermediate"
            ),
            interface_locked=index in {0, TOY_LAYERS},
            base_encoder=torch.eye(TOY_WIDTH, TOY_WIDTH, dtype=torch.float64),
            fit_parent_rows=parent,
            fit_child_rows=child,
            fit_weights=torch.ones(rows, dtype=torch.float64),
            fit_provenance=f"toy-fit-{index}",
            row_alignment_sha256=TOY_RECAPTURE_ALIGNMENT_SHA,
            cached_parent_evidence_sha256=PARENT_SHA,
            live_child_recapture_sha256=LIVE_CHILD_SHA,
        )
        for index, (parent, child) in enumerate(zip(parents, children, strict=True))
    )


def _toy_writers(child: _ToyChild, writer_inputs):
    return tuple(
        Lfm2MovingGaugeWriterEvidence(
            parameter_id=_toy_parameter_id(index),
            transition_index=index,
            seed_weight=child.writers[index],
            fit_input_rows=writer_inputs[index],
            row_alignment_sha256=TOY_RECAPTURE_ALIGNMENT_SHA,
            live_child_recapture_sha256=LIVE_CHILD_SHA,
        )
        for index in range(TOY_LAYERS)
    )


def _toy_terminal_relative_rmse(
    state: torch.Tensor,
    target: torch.Tensor,
) -> float:
    difference = torch.sqrt(((state - target) ** 2).mean())
    denominator = torch.sqrt((target**2).mean()).clamp_min(1e-30)
    return float((difference / denominator).item())


def _toy_fold(seed: int):
    """Build one small nonlinear LFM2-shaped fold plus a recapture provider."""

    generator = torch.Generator().manual_seed(seed)
    child = _ToyChild.seeded(generator)
    parent = _ToyParent(generator)
    fit_entry = torch.randn(48, TOY_WIDTH, generator=generator, dtype=torch.float64)
    heldout_entry = torch.randn(
        32,
        TOY_WIDTH,
        generator=generator,
        dtype=torch.float64,
    )
    fit_parents = parent.run(fit_entry)
    heldout_parents = parent.run(heldout_entry)
    fit_children, fit_writer_inputs = child.run(fit_entry)
    boundaries = _toy_boundaries(fit_parents, fit_children)
    writers = _toy_writers(child, fit_writer_inputs)
    calls: list[str] = []

    def provider(request):
        calls.append(request["request_sha256"])
        replacements = request["forward_replacement_weights"]
        preview = _ToyChild(
            child.readers,
            [
                replacements[_toy_parameter_id(index)].clone()
                for index in range(TOY_LAYERS)
            ],
        )
        rows, inputs = preview.run(fit_entry)
        return Lfm2MovingGaugeRecapturedChildState(
            request_sha256=request["request_sha256"],
            boundary_child_rows=rows,
            writer_input_rows={
                _toy_parameter_id(index): inputs[index]
                for index in range(TOY_LAYERS)
            },
            row_alignment_sha256=TOY_RECAPTURE_ALIGNMENT_SHA,
            recapture_provenance="toy-mid-sweep-recapture",
            child_state_sha256=TOY_CHILD_STATE_SHA,
            capture_sha256=TOY_CAPTURE_SHA,
            child_replay_capture_calls=1,
            child_model_forward_calls=TOY_LAYERS,
        )

    return {
        "child": child,
        "boundaries": boundaries,
        "writers": writers,
        "provider": provider,
        "provider_calls": calls,
        "fit_entry": fit_entry,
        "heldout_entry": heldout_entry,
        "fit_parents": fit_parents,
        "heldout_parents": heldout_parents,
    }


def _toy_install(child: _ToyChild, state, entry: torch.Tensor):
    installed = _ToyChild(
        child.readers,
        [
            state.replacements_by_parameter_id[_toy_parameter_id(index)]
            for index in range(TOY_LAYERS)
        ],
    )
    rows, _inputs = installed.run(entry)
    return rows[-1]


def test_forward_only_reverse_sweep_is_the_sealed_default() -> None:
    policy = Lfm2MovingGaugeReclosurePolicy()
    recaptured = Lfm2MovingGaugeReclosurePolicy(
        reverse_sweep_mode="recaptured_terminal_anchored",
    )

    assert policy.reverse_sweep_mode == "forward_only_identity_transport"
    assert policy.authentication_sha256 == FORWARD_ONLY_POLICY_SHA256
    assert "reverse_sweep_mode" not in policy.authentication_body()
    assert "policy_schema" not in policy.authentication_body()
    assert recaptured.authentication_sha256 != policy.authentication_sha256
    assert recaptured.as_dict()["policy_schema"] == (
        moving_gauge_reclosure.LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA
    )
    assert recaptured.as_dict()["mid_sweep_child_recapture_count"] == 1
    assert recaptured.verify_authentication()


def test_recaptured_reverse_sweep_requires_exactly_one_provider() -> None:
    fold = _toy_fold(2026)

    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        compile_lfm2_moving_gauge_residual_reclosure(
            fold["boundaries"],
            fold["writers"],
            policy=Lfm2MovingGaugeReclosurePolicy(
                reverse_sweep_mode="recaptured_terminal_anchored",
            ),
        )
    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        compile_lfm2_moving_gauge_residual_reclosure(
            fold["boundaries"],
            fold["writers"],
            mid_sweep_recapture_provider=fold["provider"],
        )
    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        Lfm2MovingGaugeReclosurePolicy(reverse_sweep_mode="whatever_mode")
    assert fold["provider_calls"] == []


def test_recaptured_reverse_sweep_beats_forward_only_on_heldout_terminal() -> None:
    fold = _toy_fold(2026)
    forward_only = compile_lfm2_moving_gauge_residual_reclosure(
        fold["boundaries"],
        fold["writers"],
    )
    recaptured = compile_lfm2_moving_gauge_residual_reclosure(
        fold["boundaries"],
        fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            reverse_sweep_mode="recaptured_terminal_anchored",
        ),
        mid_sweep_recapture_provider=fold["provider"],
    )

    assert len(fold["provider_calls"]) == 1
    terminal_chart = forward_only.boundaries[-1].chart
    heldout_target = terminal_chart.encode(fold["heldout_parents"][-1])
    seeded = _ToyChild(
        fold["child"].readers,
        [
            torch.zeros(TOY_WIDTH, TOY_WIDTH, dtype=torch.float32)
            for _ in range(TOY_LAYERS)
        ],
    )
    unfolded_rows, _inputs = seeded.run(fold["heldout_entry"])
    unfolded = _toy_terminal_relative_rmse(unfolded_rows[-1], heldout_target)
    forward_error = _toy_terminal_relative_rmse(
        _toy_install(fold["child"], forward_only, fold["heldout_entry"]),
        heldout_target,
    )
    recaptured_error = _toy_terminal_relative_rmse(
        _toy_install(fold["child"], recaptured, fold["heldout_entry"]),
        heldout_target,
    )

    assert forward_error < unfolded
    assert recaptured_error < 0.8 * forward_error, (
        recaptured_error,
        forward_error,
        unfolded,
    )
    assert recaptured.verify_authentication()
    assert recaptured.report["replacement_weight_sha256"] != (
        forward_only.report["replacement_weight_sha256"]
    )


def test_recaptured_receipt_records_measured_transport_and_recapture() -> None:
    fold = _toy_fold(2029)
    seed_digests = [
        moving_gauge_reclosure.fingerprint_lfm2_moving_gauge_tensor(value)
        for value in fold["child"].writers
    ]
    recaptured = compile_lfm2_moving_gauge_residual_reclosure(
        fold["boundaries"],
        fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            reverse_sweep_mode="recaptured_terminal_anchored",
        ),
        mid_sweep_recapture_provider=fold["provider"],
    )
    diagnostics = recaptured.report["diagnostics"]
    steps = diagnostics["reverse_steps"]

    assert diagnostics["reverse_downstream_transport_kind"] == (
        "measured_recaptured_child_downstream_response"
    )
    assert diagnostics["frozen_writer_input_identity_transport"] is False
    assert diagnostics["chart_transition_used_as_physical_jacobian"] is False
    assert diagnostics["reverse_transport_ridge_ratio_used"] is True
    assert diagnostics["mid_sweep_child_recapture_count"] == 1
    assert diagnostics["mid_sweep_recapture_request_sha256"] == (
        fold["provider_calls"][0]
    )
    assert steps[0]["downstream_transport_kind"] == (
        "exact_physical_identity_at_terminal_boundary"
    )
    assert all(
        step["downstream_transport_kind"]
        == "measured_recaptured_child_downstream_response"
        for step in steps[1:]
    )
    for step in steps:
        assert step["downstream_transport_inverted"] is False
        assert step["transport_reciprocal_report_sha256"] is None
        assert step["writer_inputs_measured_after_forward_publication"] is True
        assert step["terminal_relative_rmse_after"] <= (
            step["terminal_relative_rmse_before"]
        )
        assert step["terminal_amplification_ratio"] is not None
    assert [step["terminal_relative_rmse_before"] for step in steps[1:]] == [
        step["terminal_relative_rmse_after"] for step in steps[:-1]
    ]
    assert recaptured.report["execution"]["parent_model_forward_calls"] == 0
    assert recaptured.report["execution"]["backward_calls"] == 0
    assert recaptured.report["execution"]["optimizer_steps"] == 0
    assert recaptured.report["execution"]["child_replay_capture_calls"] == 1
    assert recaptured.report["execution"]["target_model_forward_calls"] == TOY_LAYERS
    assert recaptured.report["mid_sweep_recapture"]["parent_model_forward_calls"] == 0
    assert verify_lfm2_moving_gauge_report(recaptured.report["mid_sweep_recapture"])
    # Rollback is exact because no seed tensor is ever mutated in place.
    assert seed_digests == [
        moving_gauge_reclosure.fingerprint_lfm2_moving_gauge_tensor(value)
        for value in fold["child"].writers
    ]
    assert recaptured.report["physical_seed_tensors_mutated"] is False


def test_stale_or_foreign_recapture_fails_closed() -> None:
    fold = _toy_fold(2031)
    policy = Lfm2MovingGaugeReclosurePolicy(
        reverse_sweep_mode="recaptured_terminal_anchored",
    )

    def stale_request(request):
        recaptured = fold["provider"](request)
        return replace(
            recaptured,
            request_sha256="9" * 64,
            report={},
        )

    def mutated_rows(request):
        recaptured = fold["provider"](request)
        recaptured.boundary_child_rows[-1][0, 0] += 1.0
        return recaptured

    def wrong_geometry(request):
        recaptured = fold["provider"](request)
        return replace(
            recaptured,
            boundary_child_rows=recaptured.boundary_child_rows[:-1],
            report={},
        )

    def not_a_recapture(request):
        return dict(request)

    for provider in (stale_request, mutated_rows, wrong_geometry, not_a_recapture):
        with pytest.raises(
            (
                Lfm2MovingGaugeResidualReclosureError,
                Lfm2MovingGaugeResidualAuthenticationError,
            )
        ):
            compile_lfm2_moving_gauge_residual_reclosure(
                fold["boundaries"],
                fold["writers"],
                policy=policy,
                mid_sweep_recapture_provider=provider,
            )


def test_identity_downstream_response_reproduces_the_anchored_writer_ridge() -> None:
    generator = torch.Generator().manual_seed(613)
    design = torch.randn(24, 5, generator=generator, dtype=torch.float64)
    residual = torch.randn(24, 3, generator=generator, dtype=torch.float64)
    anchor = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    weights = torch.rand(24, generator=generator, dtype=torch.float64) + 0.5

    composed, composed_report = (
        moving_gauge_reclosure._terminal_anchored_composed_ridge(
            design,
            torch.eye(3, dtype=torch.float64),
            residual,
            anchor,
            weights,
            ridge_ratio=1e-8,
            absolute_ridge_floor=1e-12,
        )
    )
    direct, _direct_report = moving_gauge_reclosure._anchored_ridge_replacement(
        design,
        design @ anchor + residual,
        anchor,
        weights,
        ridge_ratio=1e-8,
        absolute_ridge_floor=1e-12,
    )

    torch.testing.assert_close(composed, direct, rtol=1e-9, atol=1e-9)
    assert composed_report["downstream_transport_inverted"] is False
    assert composed_report["pseudoinverse_used"] is False


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

# ---------------------------------------------------------------------------
# NOTES-W13B section 8 item 3: the body reclosure must prove it improved every
# boundary it overwrites.  At r12's layer 0 the reclosure left ``in_proj``
# byte-identical, multiplied the writer's Frobenius norm by 4.73x, and took the
# composed post-operator residual from 0.494 to 0.836.  These tests are that
# arithmetic in miniature.
# ---------------------------------------------------------------------------

REQUIRE_IMPROVEMENT = (
    moving_gauge_reclosure.LFM2_MOVING_GAUGE_WRITER_RECLOSURE_REQUIRE_IMPROVEMENT
)
ACCEPTANCE_FAIL_CLOSED = (
    moving_gauge_reclosure.LFM2_MOVING_GAUGE_WRITER_RECLOSURE_FAIL_CLOSED
)
UNGUARDED = moving_gauge_reclosure.LFM2_MOVING_GAUGE_WRITER_RECLOSURE_UNGUARDED
# Large enough that no boundary can ever regress past it, so the guarded lane
# evaluates every overwrite and accepts every one of them.
UNBOUNDED_TOLERANCE = 1e9


def _converging_single_transition_fold(seed: int = 11):
    """One transition whose reverse sweep re-solves the forward sweep's target.

    With a single transition the terminal residual *is* the boundary's own
    remaining write error, so the reverse anchored ridge re-solves the identical
    weighted problem anchored at the forward answer and can only improve it.
    The design is deliberately over-determined (64 rows against 4 features) so
    the residual sits far above the FP32 publication-cast noise floor.
    """

    generator = torch.Generator().manual_seed(seed)
    rows, width, features_width = 64, 3, 4
    parent0 = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    parent1 = torch.tanh(
        parent0
        @ torch.randn(width, width, generator=generator, dtype=torch.float64)
    )
    child0 = parent0.clone()
    child1 = parent1 + 0.3 * torch.randn(
        rows,
        width,
        generator=generator,
        dtype=torch.float64,
    )
    features = torch.randn(
        rows,
        features_width,
        generator=generator,
        dtype=torch.float64,
    )
    seed_weight = (
        0.4
        * torch.randn(
            width,
            features_width,
            generator=generator,
            dtype=torch.float64,
        )
    ).to(dtype=torch.float32)
    writer = Lfm2MovingGaugeWriterEvidence(
        parameter_id="model.layers.0.conv.out_proj.weight",
        transition_index=0,
        seed_weight=seed_weight,
        fit_input_rows=features,
        row_alignment_sha256=ALIGNMENT_SHA,
        live_child_recapture_sha256=LIVE_CHILD_SHA,
    )
    boundaries = (
        _boundary(0, parent0, child0, role="input", locked=True),
        _boundary(1, parent1, child1, role="terminal", locked=True),
    )
    return boundaries, (writer,)


def _acceptance_ledger(state) -> dict:
    return state.report["diagnostics"]["boundary_acceptance"]


def _decision(ledger: dict, sweep: str, transition_index: int) -> dict:
    matches = [
        value
        for value in ledger["decisions"]
        if value["sweep"] == sweep
        and value["transition_index"] == transition_index
    ]
    assert len(matches) == 1, (sweep, transition_index, matches)
    return matches[0]


def test_unguarded_boundary_acceptance_is_the_sealed_default() -> None:
    policy = Lfm2MovingGaugeReclosurePolicy()
    guarded = Lfm2MovingGaugeReclosurePolicy(
        writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
    )
    fail_closed = Lfm2MovingGaugeReclosurePolicy(
        writer_reclosure_acceptance=ACCEPTANCE_FAIL_CLOSED,
    )
    tolerant = Lfm2MovingGaugeReclosurePolicy(
        writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
        boundary_regression_tolerance=0.25,
    )
    recaptured_guarded = Lfm2MovingGaugeReclosurePolicy(
        reverse_sweep_mode="recaptured_terminal_anchored",
        writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
    )

    # The unguarded body is pinned byte-for-byte against the digest every
    # reclosure sealed before this gate existed.
    assert policy.writer_reclosure_acceptance == UNGUARDED
    assert policy.boundary_regression_tolerance == 0.0
    assert policy.authentication_sha256 == FORWARD_ONLY_POLICY_SHA256
    assert "writer_reclosure_acceptance" not in policy.authentication_body()
    assert "acceptance_schema" not in policy.authentication_body()
    assert "boundary_regression_tolerance" not in policy.authentication_body()

    digests = {
        value.authentication_sha256
        for value in (
            policy,
            guarded,
            fail_closed,
            tolerant,
            recaptured_guarded,
            Lfm2MovingGaugeReclosurePolicy(
                reverse_sweep_mode="recaptured_terminal_anchored",
            ),
        )
    }
    assert len(digests) == 6
    assert guarded.as_dict()["acceptance_schema"] == (
        moving_gauge_reclosure.LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA
    )
    assert guarded.as_dict()[
        "boundary_acceptance_refusal_keeps_incumbent_writer"
    ] is True
    assert guarded.as_dict()[
        "boundary_acceptance_refusal_aborts_reclosure"
    ] is False
    assert fail_closed.as_dict()[
        "boundary_acceptance_refusal_aborts_reclosure"
    ] is True
    # Both extensions compose additively in one policy body.
    assert recaptured_guarded.as_dict()["policy_schema"] == (
        moving_gauge_reclosure.LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA
    )
    assert recaptured_guarded.as_dict()["acceptance_schema"] == (
        moving_gauge_reclosure.LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA
    )
    assert all(
        value.verify_authentication()
        for value in (policy, guarded, fail_closed, tolerant, recaptured_guarded)
    )

    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        Lfm2MovingGaugeReclosurePolicy(writer_reclosure_acceptance="whatever")
    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        # A tolerance without a gate would silently claim a guard that is off.
        Lfm2MovingGaugeReclosurePolicy(boundary_regression_tolerance=0.25)
    with pytest.raises(Lfm2MovingGaugeResidualReclosureError):
        Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
            boundary_regression_tolerance=-1.0,
        )


def test_boundary_gate_accepts_a_reclosure_that_improves_every_boundary() -> None:
    boundaries, writers = _converging_single_transition_fold()
    unguarded = compile_lfm2_moving_gauge_residual_reclosure(boundaries, writers)
    guarded = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
        ),
    )
    ledger = _acceptance_ledger(guarded)

    # Every overwrite is measured and every one of them is kept, so the guarded
    # publication is the unguarded publication, byte for byte.
    assert ledger["evaluated_writer_overwrites"] == 2
    assert ledger["refused_writer_overwrites"] == 0
    assert ledger["refused_transition_indices"] == []
    assert guarded.report["replacement_weight_sha256"] == (
        unguarded.report["replacement_weight_sha256"]
    )
    assert guarded.verify_authentication()
    forward = _decision(ledger, "forward", 0)
    reverse = _decision(ledger, "reverse", 0)
    assert forward["incumbent_stage"] == "pre_reclosure_seed"
    assert reverse["incumbent_stage"] == "post_forward_sweep"
    assert (
        forward["proposed_relative_write_error"]
        < forward["incumbent_relative_write_error"]
    )
    assert (
        reverse["proposed_relative_write_error"]
        <= reverse["incumbent_relative_write_error"]
    )
    assert reverse["incumbent_relative_write_error"] == pytest.approx(
        forward["proposed_relative_write_error"]
    )
    assert ledger["every_published_boundary_no_worse_than_seed"] is True
    assert ledger["per_boundary"][0][
        "published_relative_write_error"
    ] == pytest.approx(reverse["accepted_relative_write_error"])

    # A small tolerance changes nothing when nothing regresses.
    tolerant = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
            boundary_regression_tolerance=1e-6,
        ),
    )
    assert tolerant.report["replacement_weight_sha256"] == (
        unguarded.report["replacement_weight_sha256"]
    )


@pytest.mark.parametrize("recaptured_reverse", [False, True])
def test_boundary_gate_is_a_pure_filter_at_unbounded_tolerance(
    recaptured_reverse: bool,
) -> None:
    """Accepting every proposal must reproduce the unguarded run exactly.

    This is what makes a refusal attributable: the gate adds no solve, no
    reordering and no numerical perturbation of its own, so any difference
    between the guarded and unguarded publications is a refusal and nothing
    else.
    """

    unguarded_fold = _toy_fold(7)
    guarded_fold = _toy_fold(7)
    extra = (
        {"reverse_sweep_mode": "recaptured_terminal_anchored"}
        if recaptured_reverse
        else {}
    )
    unguarded = compile_lfm2_moving_gauge_residual_reclosure(
        unguarded_fold["boundaries"],
        unguarded_fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(**extra),
        mid_sweep_recapture_provider=(
            unguarded_fold["provider"] if recaptured_reverse else None
        ),
    )
    permissive = compile_lfm2_moving_gauge_residual_reclosure(
        guarded_fold["boundaries"],
        guarded_fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
            boundary_regression_tolerance=UNBOUNDED_TOLERANCE,
            **extra,
        ),
        mid_sweep_recapture_provider=(
            guarded_fold["provider"] if recaptured_reverse else None
        ),
    )
    ledger = _acceptance_ledger(permissive)

    assert permissive.report["replacement_weight_sha256"] == (
        unguarded.report["replacement_weight_sha256"]
    )
    assert ledger["evaluated_writer_overwrites"] == 2 * TOY_LAYERS
    assert ledger["refused_writer_overwrites"] == 0
    assert len(guarded_fold["provider_calls"]) == (1 if recaptured_reverse else 0)
    assert unguarded.report.get("boundary_acceptance_sha256") is None
    assert "boundary_acceptance" not in unguarded.report["diagnostics"]


def test_boundary_gate_refuses_the_regression_the_layer0_study_convicted() -> None:
    """The 0.836 -> 0.494 arithmetic of NOTES-W13B section 6, in miniature.

    The depth toy's reverse sweep optimizes a terminal objective that conflicts
    with the first two boundaries' own writes.  Unguarded, it publishes writers
    that regress those boundaries.  Guarded, it keeps the incumbents and every
    published boundary is strictly better.
    """

    unguarded_fold = _toy_fold(7)
    permissive_fold = _toy_fold(7)
    guarded_fold = _toy_fold(7)
    unguarded = compile_lfm2_moving_gauge_residual_reclosure(
        unguarded_fold["boundaries"],
        unguarded_fold["writers"],
    )
    # The unguarded lane reproduced exactly, but with the ledger attached so its
    # per-boundary write errors are readable without a second measurement path.
    permissive = compile_lfm2_moving_gauge_residual_reclosure(
        permissive_fold["boundaries"],
        permissive_fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
            boundary_regression_tolerance=UNBOUNDED_TOLERANCE,
        ),
    )
    guarded = compile_lfm2_moving_gauge_residual_reclosure(
        guarded_fold["boundaries"],
        guarded_fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
        ),
    )
    assert permissive.report["replacement_weight_sha256"] == (
        unguarded.report["replacement_weight_sha256"]
    )
    unguarded_ledger = _acceptance_ledger(permissive)
    guarded_ledger = _acceptance_ledger(guarded)

    assert guarded_ledger["refused_transition_indices"] == [0, 1]
    assert guarded_ledger["refused_by_sweep"] == {"forward": 0, "reverse": 2}
    assert guarded.report["replacement_weight_sha256"] != (
        unguarded.report["replacement_weight_sha256"]
    )

    unguarded_published = {
        row["transition_index"]: row["published_relative_write_error"]
        for row in unguarded_ledger["per_boundary"]
    }
    guarded_published = {
        row["transition_index"]: row["published_relative_write_error"]
        for row in guarded_ledger["per_boundary"]
    }
    # Refused boundaries are strictly better; the accepted one is untouched.
    for index in (0, 1):
        assert guarded_published[index] < unguarded_published[index], (
            index,
            guarded_published,
            unguarded_published,
        )
    assert guarded_published[TOY_LAYERS - 1] == pytest.approx(
        unguarded_published[TOY_LAYERS - 1]
    )
    # The r12 signature: the refused proposal both regressed the boundary and
    # inflated the write it publishes.
    for index in (0, 1):
        refusal = _decision(guarded_ledger, "reverse", index)
        assert refusal["accepted"] is False
        assert (
            refusal["proposed_relative_write_error"]
            > refusal["incumbent_relative_write_error"]
        )
        assert refusal["relative_write_error_ratio"] > 1.0
        assert refusal["write_norm_ratio"] > 1.0
        assert refusal["accepted_relative_write_error"] == pytest.approx(
            refusal["incumbent_relative_write_error"]
        )
        assert refusal["accepted_write_l2_norm"] == pytest.approx(
            refusal["incumbent_write_l2_norm"]
        )
        assert refusal["accepted_weight_frobenius_norm_by_parameter"] == (
            refusal["incumbent_weight_frobenius_norm_by_parameter"]
        )
        # A refusal publishes nothing, so the reverse step realizes no change.
        step = next(
            value
            for value in guarded.report["diagnostics"]["reverse_steps"]
            if value["transition_index"] == index
        )
        assert step["realized_terminal_l2_norm"] == 0.0
    assert guarded_ledger["every_published_boundary_no_worse_than_seed"] is True
    assert guarded_ledger["metric"] == "weighted_relative_l2_branch_write_error"
    assert guarded_ledger["reverse_row_frame"] == "aligned_fit_writer_input_rows"
    assert guarded.report["boundary_acceptance_sha256"] == (
        guarded_ledger["report_sha256"]
    )
    assert verify_lfm2_moving_gauge_report(guarded_ledger)
    assert (
        guarded.report["boundary_acceptance_summary"]["refused_writer_overwrites"]
        == 2
    )
    assert guarded.report["writer_reclosure_acceptance"] == REQUIRE_IMPROVEMENT
    assert guarded.report["boundary_regression_tolerance"] == 0.0
    assert guarded.report["per_boundary_monotone_writer_reclosure"] is True
    assert guarded.verify_authentication()


def test_boundary_gate_tolerance_admits_a_bounded_regression() -> None:
    strict_fold = _toy_fold(2031)
    loose_fold = _toy_fold(2031)
    strict = compile_lfm2_moving_gauge_residual_reclosure(
        strict_fold["boundaries"],
        strict_fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
        ),
    )
    strict_ledger = _acceptance_ledger(strict)
    refusal = _decision(strict_ledger, "reverse", 1)
    # Seed 2031's transition 1 regresses by a bounded ratio; a tolerance just
    # above it must admit the proposal, and the strict lane must still refuse.
    ratio = refusal["relative_write_error_ratio"]
    loose = compile_lfm2_moving_gauge_residual_reclosure(
        loose_fold["boundaries"],
        loose_fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
            boundary_regression_tolerance=ratio - 1.0 + 1e-9,
        ),
    )
    loose_ledger = _acceptance_ledger(loose)

    assert 1.0 < ratio < 2.0
    assert refusal["accepted"] is False
    assert _decision(loose_ledger, "reverse", 1)["accepted"] is True
    assert loose_ledger["refused_writer_overwrites"] < (
        strict_ledger["refused_writer_overwrites"]
    )


def test_boundary_gate_fail_closed_aborts_instead_of_skipping() -> None:
    fold = _toy_fold(7)
    skipping = compile_lfm2_moving_gauge_residual_reclosure(
        _toy_fold(7)["boundaries"],
        _toy_fold(7)["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
        ),
    )

    assert _acceptance_ledger(skipping)["refused_writer_overwrites"] == 2
    with pytest.raises(Lfm2MovingGaugeResidualReclosureError) as excinfo:
        compile_lfm2_moving_gauge_residual_reclosure(
            fold["boundaries"],
            fold["writers"],
            policy=Lfm2MovingGaugeReclosurePolicy(
                writer_reclosure_acceptance=ACCEPTANCE_FAIL_CLOSED,
            ),
        )
    message = str(excinfo.value)
    assert "reverse sweep regressed boundary" in message
    assert "toy:boundary:" in message
    # The fail-closed mode is not simply unusable: a fold whose reclosure does
    # improve every boundary compiles under it.
    boundaries, writers = _converging_single_transition_fold()
    converging = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
        policy=Lfm2MovingGaugeReclosurePolicy(
            writer_reclosure_acceptance=ACCEPTANCE_FAIL_CLOSED,
        ),
    )
    assert _acceptance_ledger(converging)["refused_writer_overwrites"] == 0


def test_boundary_gate_measures_the_recaptured_states_the_sweep_solves_on() -> None:
    """Composition with ``recaptured_terminal_anchored``.

    When both are on, the acceptance measurement must be taken on the same
    mid-sweep recapture the reverse sweep solves against, not on the stale FIT
    rows.  The check is an independent recomputation of one decision straight
    from the recapture the provider handed back.
    """

    fold = _toy_fold(7)
    captured: dict[str, object] = {}

    def provider(request):
        recaptured = fold["provider"](request)
        captured["request"] = request
        captured["recaptured"] = recaptured
        return recaptured

    guarded = compile_lfm2_moving_gauge_residual_reclosure(
        fold["boundaries"],
        fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            reverse_sweep_mode="recaptured_terminal_anchored",
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
            boundary_regression_tolerance=UNBOUNDED_TOLERANCE,
        ),
        mid_sweep_recapture_provider=provider,
    )
    forward_only = _acceptance_ledger(
        compile_lfm2_moving_gauge_residual_reclosure(
            _toy_fold(7)["boundaries"],
            _toy_fold(7)["writers"],
            policy=Lfm2MovingGaugeReclosurePolicy(
                writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
                boundary_regression_tolerance=UNBOUNDED_TOLERANCE,
            ),
        )
    )
    ledger = _acceptance_ledger(guarded)
    index = TOY_LAYERS - 1
    decision = _decision(ledger, "reverse", index)
    parameter_id = _toy_parameter_id(index)

    assert ledger["reverse_sweep_mode"] == "recaptured_terminal_anchored"
    assert ledger["reverse_row_frame"] == "recaptured_writer_input_rows"
    assert decision["row_frame"] == "recaptured_writer_input_rows"
    assert _decision(ledger, "forward", index)["row_frame"] == (
        "aligned_fit_writer_input_rows"
    )
    # The forward decisions are identical because the forward sweep is; only
    # the reverse frame moved, which is the whole point.
    assert _decision(ledger, "forward", index) == (
        _decision(forward_only, "forward", index)
    )
    assert decision["incumbent_relative_write_error"] != pytest.approx(
        _decision(forward_only, "reverse", index)[
            "incumbent_relative_write_error"
        ]
    )

    # Independent recomputation from the recapture the provider returned.
    recaptured = captured["recaptured"]
    incumbent = captured["request"]["forward_replacement_weights"][parameter_id]
    features = recaptured.writer_input_rows[parameter_id]
    pre_state = recaptured.boundary_child_rows[index]
    post_state = recaptured.boundary_child_rows[index + 1]
    writer_output = features @ incumbent.to(dtype=torch.float64).T
    _endpoint, desired_write = guarded.boundaries[
        index + 1
    ].chart.residual_write_target(
        actual_child_pre_state=pre_state,
        parent_endpoint=fold["boundaries"][index + 1].fit_parent_rows,
    )
    fixed_other_write = (post_state - pre_state) - writer_output
    expected = moving_gauge_reclosure._relative_rmse(
        writer_output,
        desired_write - fixed_other_write,
        fold["boundaries"][index].fit_weights,
    )

    assert decision["rows"] == int(features.shape[0])
    assert decision["incumbent_relative_write_error"] == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )
    assert guarded.report["policy_schema"] == (
        moving_gauge_reclosure.LFM2_MOVING_GAUGE_RECAPTURED_REVERSE_SCHEMA
    )
    assert guarded.report["acceptance_schema"] == (
        moving_gauge_reclosure.LFM2_MOVING_GAUGE_BOUNDARY_ACCEPTANCE_SCHEMA
    )
    assert guarded.report["mid_sweep_recapture_sha256"] == (
        recaptured.report["report_sha256"]
    )
    assert guarded.report["execution"]["backward_calls"] == 0
    assert guarded.report["execution"]["optimizer_steps"] == 0
    assert guarded.report["execution"]["parent_model_forward_calls"] == 0
    assert guarded.verify_authentication()


def test_recaptured_boundary_gate_refuses_and_keeps_the_incumbent() -> None:
    fold = _toy_fold(7)
    guarded = compile_lfm2_moving_gauge_residual_reclosure(
        fold["boundaries"],
        fold["writers"],
        policy=Lfm2MovingGaugeReclosurePolicy(
            reverse_sweep_mode="recaptured_terminal_anchored",
            writer_reclosure_acceptance=REQUIRE_IMPROVEMENT,
        ),
        mid_sweep_recapture_provider=fold["provider"],
    )
    ledger = _acceptance_ledger(guarded)
    refusals = [value for value in ledger["decisions"] if not value["accepted"]]

    assert len(fold["provider_calls"]) == 1
    assert refusals, ledger["decisions"]
    for refusal in refusals:
        assert refusal["sweep"] == "reverse"
        assert refusal["row_frame"] == "recaptured_writer_input_rows"
        assert (
            refusal["proposed_relative_write_error"]
            > refusal["incumbent_relative_write_error"]
        )
        step = next(
            value
            for value in guarded.report["diagnostics"]["reverse_steps"]
            if value["transition_index"] == refusal["transition_index"]
        )
        assert step["realized_local_l2_norm"] == 0.0
        assert step["realized_terminal_l2_norm"] == 0.0
    assert guarded.verify_authentication()
