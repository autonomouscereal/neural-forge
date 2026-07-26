from __future__ import annotations

import ast
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.analytic_coordinate_completion import (  # noqa: E402
    AnalyticCoordinateCompletionError,
    AnalyticCoordinateCompletionRankError,
    analytic_coordinate_completion_evidence_sha256,
    compile_analytic_coordinate_completion_frontier,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    ResidualFacingOperator,
)


def _orthonormal_columns(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(value, mode="reduced").Q


def _case(
    *,
    target_width: int = 7,
    source_width: int = 11,
    trace_rank: int = 3,
    seed: int = 810,
    bridge: bool = True,
):
    generator = torch.Generator().manual_seed(seed)
    identified_basis = _orthonormal_columns(target_width, trace_rank, seed + 1)
    trace_coefficients = torch.randn(
        trace_rank + 4,
        trace_rank,
        generator=generator,
        dtype=torch.float64,
    )
    trace_design = trace_coefficients @ identified_basis.T
    identified_action = torch.randn(
        trace_rank,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    fitted_map = identified_basis @ identified_action
    target_projector = identified_basis @ identified_basis.T
    fitted_map = fitted_map + (
        torch.eye(target_width, dtype=torch.float64) - target_projector
    ) @ torch.randn(
        target_width,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    operators = (
        ResidualFacingOperator(
            role="attention_consumers",
            name="qkv_joint",
            tensor=torch.randn(
                source_width + 3,
                source_width,
                generator=generator,
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
        ),
        ResidualFacingOperator(
            role="downstream_writers",
            name="attention_o_and_mlp_down",
            tensor=torch.randn(
                source_width,
                source_width + 5,
                generator=generator,
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    bridge_prior = (
        torch.randn(
            target_width,
            source_width,
            generator=generator,
            dtype=torch.float64,
        )
        if bridge
        else None
    )
    evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted_map,
        operators,
        bridge_prior=bridge_prior,
    )
    return {
        "trace_design": trace_design,
        "fitted_map": fitted_map,
        "parent_operators": operators,
        "bridge_prior": bridge_prior,
        "expected_evidence_sha256": evidence,
        "identified_basis": identified_basis,
    }


def test_full_rank_frontier_preserves_only_identified_trace_action() -> None:
    case = _case()
    result = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=case["expected_evidence_sha256"],
        bridge_prior=case["bridge_prior"],
    )

    assert len(result.candidates) >= 3
    assert len({value.map_sha256 for value in result.candidates}) == len(
        result.candidates
    )
    assert "operator-sensitive-isometric" in {
        value.recipe for value in result.candidates
    }
    assert "operator-sensitive-reflected" in {
        value.recipe for value in result.candidates
    }
    assert "bridge-procrustes-isometric" in {
        value.recipe for value in result.candidates
    }
    target_width, source_width = case["fitted_map"].shape
    expected_rank = min(target_width, source_width)
    trace_reference = case["trace_design"] @ case["fitted_map"]
    identified_reference = case["identified_basis"].T @ case["fitted_map"]
    identified_projector = (
        result.identified_target_basis @ result.identified_target_basis.T
    )
    for candidate in result.candidates:
        assert torch.linalg.matrix_rank(candidate.coordinate_map).item() == expected_rank
        assert torch.allclose(
            case["trace_design"] @ candidate.coordinate_map,
            trace_reference,
            rtol=2e-12,
            atol=2e-12,
        )
        assert torch.allclose(
            case["identified_basis"].T @ candidate.coordinate_map,
            identified_reference,
            rtol=2e-12,
            atol=2e-12,
        )
        assert torch.allclose(
            identified_projector @ candidate.nullspace_update,
            torch.zeros_like(candidate.nullspace_update),
            rtol=0.0,
            atol=2e-12,
        )
        assert candidate.report["quality_score_gate_applied"] is False
        assert candidate.report["semantic_acceptance_decision"] is None
        assert candidate.report["training_steps"] == 0
        assert candidate.report["random_completion_used"] is False
    assert result.report["candidate_order_is_quality_ranking"] is False
    assert result.report["semantic_acceptance_requires_decoded_behavior_review"]
    assert result.report["unidentifiable_complement_is_only_modified_subspace"]


def test_frontier_is_byte_deterministic_and_evidence_sealed() -> None:
    case = _case(seed=911)
    first = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=case["expected_evidence_sha256"],
        bridge_prior=case["bridge_prior"],
    )
    second = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        tuple(reversed(case["parent_operators"])),
        expected_evidence_sha256=case["expected_evidence_sha256"],
        bridge_prior=case["bridge_prior"],
    )

    assert first.report["result_sha256"] == second.report["result_sha256"]
    assert [value.report["result_sha256"] for value in first.candidates] == [
        value.report["result_sha256"] for value in second.candidates
    ]
    assert all(
        torch.equal(left.coordinate_map, right.coordinate_map)
        for left, right in zip(first.candidates, second.candidates, strict=True)
    )

    tampered = case["fitted_map"].clone()
    tampered[0, 0] += 1e-6
    with pytest.raises(AnalyticCoordinateCompletionError, match="authentication"):
        compile_analytic_coordinate_completion_frontier(
            case["trace_design"],
            tampered,
            case["parent_operators"],
            expected_evidence_sha256=case["expected_evidence_sha256"],
            bridge_prior=case["bridge_prior"],
        )


def test_mutated_operator_tensor_cannot_reuse_its_authenticated_receipt() -> None:
    case = _case(seed=930)
    operator = case["parent_operators"][0]
    operator.tensor[0, 0] += 0.25

    with pytest.raises(
        AnalyticCoordinateCompletionError,
        match="changed after its authenticated receipt",
    ):
        analytic_coordinate_completion_evidence_sha256(
            case["trace_design"],
            case["fitted_map"],
            case["parent_operators"],
            bridge_prior=case["bridge_prior"],
        )
    with pytest.raises(
        AnalyticCoordinateCompletionError,
        match="changed after its authenticated receipt",
    ):
        compile_analytic_coordinate_completion_frontier(
            case["trace_design"],
            case["fitted_map"],
            case["parent_operators"],
            expected_evidence_sha256=case["expected_evidence_sha256"],
            bridge_prior=case["bridge_prior"],
        )


def test_large_trace_coordinates_do_not_turn_roundoff_into_a_quality_gate() -> None:
    case = _case(seed=950)
    scaled_design = case["trace_design"] * 1e12
    evidence = analytic_coordinate_completion_evidence_sha256(
        scaled_design,
        case["fitted_map"],
        case["parent_operators"],
        bridge_prior=case["bridge_prior"],
    )

    result = compile_analytic_coordinate_completion_frontier(
        scaled_design,
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=evidence,
        bridge_prior=case["bridge_prior"],
    )

    assert len(result.candidates) >= 2
    for candidate in result.candidates:
        assert candidate.report["quality_score_gate_applied"] is False
        assert candidate.report["maximum_trace_prediction_error"] <= (
            candidate.report["trace_prediction_fp64_roundoff_bound"]
        )


@pytest.mark.parametrize(
    ("target_width", "source_width", "trace_rank"),
    (
        (5, 9, 2),
        (9, 5, 3),
        (6, 6, 2),
        (3, 12, 1),
        (12, 4, 2),
    ),
    ids=(
        "narrow-child-to-wide-parent",
        "wide-child-to-narrow-parent",
        "equal-width-cross-family",
        "aggressive-width-compression",
        "modality-expansion-chart",
    ),
)
def test_cross_architecture_widths_materialize_maximum_rank(
    target_width: int,
    source_width: int,
    trace_rank: int,
) -> None:
    case = _case(
        target_width=target_width,
        source_width=source_width,
        trace_rank=trace_rank,
        seed=1000 + target_width * 20 + source_width,
        bridge=False,
    )
    result = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=case["expected_evidence_sha256"],
    )

    assert result.report["target_width"] == target_width
    assert result.report["source_width"] == source_width
    assert result.report["required_full_rank"] == min(target_width, source_width)
    assert len(result.candidates) >= 2
    for value in result.candidates:
        assert value.coordinate_map.shape == (target_width, source_width)
        assert torch.linalg.matrix_rank(value.coordinate_map).item() == min(
            target_width,
            source_width,
        )
        assert torch.allclose(
            case["trace_design"] @ value.coordinate_map,
            case["trace_design"] @ case["fitted_map"],
            rtol=2e-12,
            atol=2e-12,
        )


def test_parent_operator_geometry_changes_only_unidentified_completion() -> None:
    target_width = 4
    source_width = 6
    trace_design = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    fitted = torch.zeros(target_width, source_width, dtype=torch.float64)
    fitted[0, 0] = 2.0
    operator_a = ResidualFacingOperator(
        role="writer",
        name="anisotropic-a",
        tensor=torch.diag(
            torch.tensor([1.0, 9.0, 8.0, 7.0, 0.5, 0.25], dtype=torch.float64)
        ),
        orientation=FeatureOrientation.INPUT,
    )
    operator_b = ResidualFacingOperator(
        role="writer",
        name="anisotropic-b",
        tensor=torch.diag(
            torch.tensor([1.0, 0.25, 0.5, 7.0, 8.0, 9.0], dtype=torch.float64)
        ),
        orientation=FeatureOrientation.INPUT,
    )

    def compile_with(operator: ResidualFacingOperator):
        evidence = analytic_coordinate_completion_evidence_sha256(
            trace_design,
            fitted,
            (operator,),
        )
        return compile_analytic_coordinate_completion_frontier(
            trace_design,
            fitted,
            (operator,),
            expected_evidence_sha256=evidence,
        )

    first = compile_with(operator_a)
    second = compile_with(operator_b)
    first_map = first.candidate("operator-sensitive-isometric").coordinate_map
    second_map = second.candidate("operator-sensitive-isometric").coordinate_map

    assert torch.equal(first.identified_map, second.identified_map)
    assert not torch.equal(first.completion_source_basis, second.completion_source_basis)
    assert not torch.equal(first_map, second_map)
    assert torch.allclose(trace_design @ first_map, trace_design @ fitted)
    assert torch.allclose(trace_design @ second_map, trace_design @ fitted)


def test_role_balancing_and_fisher_blend_control_only_source_completion() -> None:
    trace_design = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    fitted = torch.zeros(2, 6, dtype=torch.float64)
    fitted[0, 0] = 1.0
    operators = (
        ResidualFacingOperator(
            role="large_raw_role",
            name="large",
            tensor=torch.tensor(
                [0.0, 1000.0, 900.0, 0.0, 0.0, 0.0],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
            diagonal=True,
        ),
        ResidualFacingOperator(
            role="small_raw_role",
            name="small",
            tensor=torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
            diagonal=True,
        ),
    )
    structural_evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        operators,
    )
    structural = compile_analytic_coordinate_completion_frontier(
        trace_design,
        fitted,
        operators,
        expected_evidence_sha256=structural_evidence,
    )

    # Equal role normalization makes the small raw role win axis five; raw
    # operator-energy aggregation would instead be dominated by axis one.
    assert torch.allclose(
        structural.completion_source_basis.abs(),
        torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], dtype=torch.float64),
    )
    role_reports = structural.report["parent_geometry"]["roles"]
    assert role_reports["large_raw_role"]["balanced_trace_contribution"] == 0.5
    assert role_reports["small_raw_role"]["balanced_trace_contribution"] == 0.5

    fisher = torch.diag(
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    )
    fisher_receipt = "b" * 64
    fisher_evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        operators,
        cached_source_fisher_gram=fisher,
        cached_source_fisher_receipt_sha256=fisher_receipt,
    )
    supplemented = compile_analytic_coordinate_completion_frontier(
        trace_design,
        fitted,
        operators,
        expected_evidence_sha256=fisher_evidence,
        cached_source_fisher_gram=fisher,
        cached_source_fisher_receipt_sha256=fisher_receipt,
        cached_source_fisher_weight=0.75,
    )
    assert torch.allclose(
        supplemented.completion_source_basis.abs(),
        torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]], dtype=torch.float64),
    )
    for candidate in supplemented.candidates:
        assert torch.equal(
            trace_design @ candidate.coordinate_map,
            trace_design @ fitted,
        )


def test_cached_fisher_and_bridge_are_authenticated_supplements() -> None:
    case = _case(seed=1200)
    source_width = int(case["fitted_map"].shape[1])
    fisher = torch.diag(torch.linspace(1.0, 3.0, source_width, dtype=torch.float64))
    receipt = "a" * 64
    evidence = analytic_coordinate_completion_evidence_sha256(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        bridge_prior=case["bridge_prior"],
        cached_source_fisher_gram=fisher,
        cached_source_fisher_receipt_sha256=receipt,
    )
    result = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=evidence,
        bridge_prior=case["bridge_prior"],
        cached_source_fisher_gram=fisher,
        cached_source_fisher_receipt_sha256=receipt,
        cached_source_fisher_weight=0.4,
    )

    supplement = result.report["parent_geometry"]["cached_fisher"]
    assert supplement["supplied"] is True
    assert supplement["weight"] == pytest.approx(0.4)
    assert supplement["structural_weight"] == pytest.approx(0.6)
    assert supplement["structural_geometry_cannot_be_replaced"] is True
    assert "bridge-procrustes-isometric" in {
        value.recipe for value in result.candidates
    }


def test_rectangular_bridge_procrustes_reaches_the_isometric_optimum() -> None:
    case = _case(
        target_width=9,
        source_width=5,
        trace_rank=2,
        seed=1250,
        bridge=True,
    )
    result = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=case["expected_evidence_sha256"],
        bridge_prior=case["bridge_prior"],
    )
    candidate = result.candidate("bridge-procrustes-isometric")
    target_null = result.target_null_basis
    source_completion = result.completion_source_basis
    scales = torch.tensor(
        candidate.report["completion_scales"],
        dtype=torch.float64,
    )
    orientation = (
        target_null.T
        @ candidate.nullspace_update
        @ source_completion
        @ torch.diag(scales.reciprocal())
    )
    bridge_coordinates = (
        target_null.T @ case["bridge_prior"] @ source_completion
    )

    assert orientation.shape[0] >= orientation.shape[1]
    assert torch.allclose(
        orientation.T @ orientation,
        torch.eye(orientation.shape[1], dtype=torch.float64),
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.trace(orientation.T @ bridge_coordinates).item() == pytest.approx(
        torch.linalg.svdvals(bridge_coordinates).sum().item(),
        rel=2e-12,
        abs=2e-12,
    )


def test_impossible_full_rank_fails_without_mutating_identified_fit() -> None:
    trace_design = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    fitted = torch.zeros(3, 5, dtype=torch.float64)
    fitted[0, 0] = 1.0
    fitted[1, 0] = 2.0
    operator = ResidualFacingOperator(
        role="writer",
        name="identity",
        tensor=torch.eye(5, dtype=torch.float64),
        orientation=FeatureOrientation.INPUT,
    )
    evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        (operator,),
    )

    with pytest.raises(AnalyticCoordinateCompletionRankError) as captured:
        compile_analytic_coordinate_completion_frontier(
            trace_design,
            fitted,
            (operator,),
            expected_evidence_sha256=evidence,
        )
    report = captured.value.report
    assert report["identified_map_rank"] == 1
    assert report["trace_identified_rank"] == 2
    assert report["maximum_achievable_rank"] == 2
    assert report["required_full_rank"] == 3
    assert report["fitted_subspace_mutation_permitted"] is False
    assert report["quality_score_gate_applied"] is False


def test_projection_roundoff_is_not_promoted_to_identified_rank() -> None:
    generator = torch.Generator().manual_seed(1601)
    full_basis = torch.linalg.qr(
        torch.randn(2, 2, generator=generator, dtype=torch.float64)
    ).Q
    trace_design = full_basis[:, :1].T.contiguous()
    identified_projector = full_basis[:, :1] @ full_basis[:, :1].T
    fitted = (
        (torch.eye(2, dtype=torch.float64) - identified_projector)
        @ torch.randn(2, 2, generator=generator, dtype=torch.float64)
    )
    # The construction has zero identified action analytically, but its FP64
    # projection leaves nonzero dust.  A purely relative rank threshold would
    # call that dust rank one and report a misleading full-rank completion.
    raw_projection = identified_projector @ fitted
    assert torch.count_nonzero(raw_projection).item() > 0
    operator = ResidualFacingOperator(
        role="writer",
        name="identity",
        tensor=torch.eye(2, dtype=torch.float64),
        orientation=FeatureOrientation.INPUT,
    )
    evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        (operator,),
    )

    with pytest.raises(AnalyticCoordinateCompletionRankError) as captured:
        compile_analytic_coordinate_completion_frontier(
            trace_design,
            fitted,
            (operator,),
            expected_evidence_sha256=evidence,
        )
    report = captured.value.report
    assert report["identified_map_rank"] == 0
    assert report["identified_map_projection_roundoff_floor"] > float(
        torch.linalg.svdvals(raw_projection).max().item()
    )
    assert report["maximum_achievable_rank"] == 1
    assert report["required_full_rank"] == 2


def test_identified_rank_is_scale_equivariant_for_genuinely_tiny_maps() -> None:
    trace_design = torch.eye(2, dtype=torch.float64)
    fitted = torch.diag(
        torch.tensor([1e-200, 2e-200], dtype=torch.float64)
    )
    operator = ResidualFacingOperator(
        role="writer",
        name="identity",
        tensor=torch.eye(2, dtype=torch.float64),
        orientation=FeatureOrientation.INPUT,
    )
    evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        (operator,),
    )

    result = compile_analytic_coordinate_completion_frontier(
        trace_design,
        fitted,
        (operator,),
        expected_evidence_sha256=evidence,
    )

    assert result.report["identified_map_rank"] == 2
    assert result.report["completion_rank"] == 0
    assert torch.equal(result.candidates[0].coordinate_map, fitted)


def test_ill_conditioned_optional_scaling_does_not_abort_valid_frontier() -> None:
    trace_design = torch.zeros((3, 5), dtype=torch.float64)
    trace_design[:3, :3] = torch.eye(3, dtype=torch.float64)
    fitted = torch.zeros((5, 5), dtype=torch.float64)
    fitted[0, 0] = 1.0
    fitted[1, 1] = 1.0
    fitted[2, 2] = 2e-12
    operator = ResidualFacingOperator(
        role="writer",
        name="rank-one-complement-geometry",
        tensor=torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float64,
        ),
        orientation=FeatureOrientation.INPUT,
        diagonal=True,
    )
    evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        (operator,),
    )

    result = compile_analytic_coordinate_completion_frontier(
        trace_design,
        fitted,
        (operator,),
        expected_evidence_sha256=evidence,
    )

    assert {value.recipe for value in result.candidates} == {
        "operator-sensitive-isometric",
        "operator-sensitive-reflected",
    }
    assert all(
        torch.linalg.matrix_rank(
            value.coordinate_map,
            rtol=1e-12,
        ).item()
        == 5
        for value in result.candidates
    )
    rejected = result.report["rejected_optional_recipe_materializations"]
    assert len(rejected) == 1
    assert rejected[0]["recipe"] == "operator-fisher-balanced"
    assert rejected[0]["reason"] == (
        "optional-scale-or-orientation-lost-numerical-full-rank"
    )
    assert len(rejected[0]["coordinate_map_sha256"]) == 64
    assert rejected[0]["materialized_rank"] == 4
    assert rejected[0]["required_full_rank"] == 5
    assert rejected[0]["rank_threshold"] > 0.0
    assert rejected[0]["quality_rejection"] is False
    assert result.report["quality_score_gate_applied"] is False


def test_already_full_rank_fit_is_returned_without_inventing_a_completion() -> None:
    trace_design = torch.eye(3, dtype=torch.float64)
    fitted = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [0.0, 0.0, 1.0, 4.0]],
        dtype=torch.float64,
    )
    operator = ResidualFacingOperator(
        role="writer",
        name="identity",
        tensor=torch.eye(4, dtype=torch.float64),
        orientation=FeatureOrientation.INPUT,
    )
    evidence = analytic_coordinate_completion_evidence_sha256(
        trace_design,
        fitted,
        (operator,),
    )

    result = compile_analytic_coordinate_completion_frontier(
        trace_design,
        fitted,
        (operator,),
        expected_evidence_sha256=evidence,
    )

    assert result.report["completion_rank"] == 0
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.recipe == "identified-map-already-full-rank"
    assert torch.equal(candidate.coordinate_map, fitted)
    assert torch.count_nonzero(candidate.nullspace_update).item() == 0


def test_module_has_zero_forbidden_execution_or_mutation_operations() -> None:
    source = Path("src/neural_forge/analytic_coordinate_completion.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_names.intersection(
        {"forward", "generate", "backward", "step", "save", "open"}
    )
    assert not called_attributes.intersection(
        {
            "forward",
            "generate",
            "backward",
            "step",
            "save",
            "write_text",
            "write_bytes",
            "requires_grad_",
        }
    )
    assert "transformers" not in source.lower()
    assert "optimizer" not in called_names
