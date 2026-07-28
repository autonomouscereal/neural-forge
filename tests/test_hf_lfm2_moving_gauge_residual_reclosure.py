from __future__ import annotations

import math

import pytest

from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBoundaryEvidence,
    Lfm2MovingGaugeReclosurePolicy,
    Lfm2MovingGaugeResidualAuthenticationError,
    Lfm2MovingGaugeWriterEvidence,
    compile_lfm2_moving_gauge_residual_reclosure,
    fit_lfm2_moving_gauge_residual_chart_adapter,
)


torch = pytest.importorskip("torch")


ALIGNMENT_SHA = "a" * 64
PARENT_SHA = "b" * 64
LIVE_CHILD_SHA = "c" * 64


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


def test_moving_gauges_reduce_endpoint_error_against_forced_identity() -> None:
    generator = torch.Generator().manual_seed(77)
    parents = tuple(
        torch.randn(24, 2, generator=generator, dtype=torch.float64)
        for _ in range(4)
    )
    rotations = (
        torch.eye(2, dtype=torch.float64),
        _rotation(0.9),
        _rotation(-0.65),
        torch.eye(2, dtype=torch.float64),
    )
    children = tuple(
        parent @ rotation
        for parent, rotation in zip(parents, rotations, strict=True)
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
        for index, (parent, child) in enumerate(
            zip(parents, children, strict=True)
        )
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

    moving_error = moving.report["diagnostics"][
        "reconciled_terminal_relative_rmse"
    ]
    identity_error = identity.report["diagnostics"][
        "reconciled_terminal_relative_rmse"
    ]
    assert moving_error < 0.5 * identity_error, (moving_error, identity_error)
    assert identity_error > 1e-3
    assert moving.boundaries[1].report[
        "sequential_transport_reused_mature_cpcf_primitive"
    ] is True
    assert all(transition.verify_authentication() for transition in moving.transitions)
    for transition in moving.transitions:
        torch.testing.assert_close(
            transition.orthogonal_factor @ transition.positive_input_factor,
            transition.full_transition,
            rtol=1e-10,
            atol=1e-10,
        )
        assert transition.report["rmsnorm_crossing_factor"] == (
            "orthogonal_factor_only"
        )
        assert transition.report[
            "arbitrary_map_conjugacy_through_rmsnorm_claimed"
        ] is False


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
    assert result.report["diagnostics"][
        "reconciled_terminal_relative_rmse"
    ] < result.report["diagnostics"]["forward_terminal_relative_rmse"]


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
