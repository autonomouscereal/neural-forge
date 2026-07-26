from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge import analytic_coordinate_completion as completion  # noqa: E402
from neural_forge.analytic_coordinate_completion import (  # noqa: E402
    AnalyticCoordinateCompletionError,
    AnalyticCoordinateCompletionRankError,
    analytic_coordinate_completion_evidence_sha256,
    compile_analytic_coordinate_completion_frontier,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    ResidualFacingOperator,
    compile_role_balanced_operator_geometry,
)


def _orthonormal_columns(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(value, mode="reduced").Q


def _legacy_pivoted_projector_basis(
    projector: torch.Tensor,
    dimension: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Small-fixture reference for the superseded exhaustive implementation."""

    width = int(projector.shape[0])
    accepted: list[torch.Tensor] = []
    pivots: list[int] = []
    remaining = set(range(width))
    while len(accepted) < dimension:
        scored: list[tuple[float, int, torch.Tensor]] = []
        for axis in sorted(remaining):
            candidate = projector[:, axis].clone()
            for _pass in range(2):
                for basis in accepted:
                    candidate = candidate - basis * torch.dot(basis, candidate)
            norm = float(torch.linalg.vector_norm(candidate).item())
            scored.append((norm, axis, candidate))
        norm, axis, candidate = max(scored, key=lambda value: (value[0], -value[1]))
        pivot = int(torch.argmax(candidate.abs()).item())
        candidate = candidate / norm
        if float(candidate[pivot].item()) < 0.0:
            candidate = -candidate
        accepted.append(candidate)
        pivots.append(axis)
        remaining.remove(axis)
    return torch.stack(accepted, dim=1).contiguous(), tuple(pivots)


@pytest.mark.parametrize(
    ("width", "dimension", "seed"),
    ((5, 2, 13), (8, 5, 27), (11, 7, 41)),
)
def test_pivoted_projector_basis_matches_small_legacy_reference(
    width: int,
    dimension: int,
    seed: int,
) -> None:
    columns = _orthonormal_columns(width, dimension, seed)
    projector = (columns @ columns.T).contiguous()

    expected_basis, expected_pivots = _legacy_pivoted_projector_basis(
        projector,
        dimension,
    )
    actual_basis, actual_pivots = completion._pivoted_projector_basis(
        projector,
        dimension,
    )

    assert actual_pivots == expected_pivots
    assert torch.allclose(actual_basis, expected_basis, rtol=2e-14, atol=2e-14)


def test_pivoted_projector_basis_is_deterministic_orthogonal_and_span_exact() -> None:
    columns = _orthonormal_columns(23, 9, 611)
    projector = (columns @ columns.T).contiguous()

    first_basis, first_pivots = completion._pivoted_projector_basis(projector, 9)
    second_basis, second_pivots = completion._pivoted_projector_basis(projector, 9)

    assert first_pivots == second_pivots
    assert torch.equal(first_basis, second_basis)
    assert torch.allclose(
        first_basis.T @ first_basis,
        torch.eye(9, dtype=torch.float64),
        rtol=0.0,
        atol=2e-13,
    )
    assert torch.allclose(
        first_basis @ first_basis.T,
        projector,
        rtol=2e-13,
        atol=2e-13,
    )

    tied = torch.zeros((8, 8), dtype=torch.float64)
    tied[0, 0] = tied[2, 2] = tied[5, 5] = 1.0
    _tied_basis, tied_pivots = completion._pivoted_projector_basis(tied, 3)
    assert tied_pivots == (0, 2, 5)


def test_pivoted_projector_basis_h1024_path_is_viable() -> None:
    projector = torch.zeros((1_024, 1_024), dtype=torch.float64)
    projector.diagonal()[:128].fill_(1.0)

    started = time.perf_counter()
    basis, pivots = completion._pivoted_projector_basis(projector, 128)
    elapsed_seconds = time.perf_counter() - started

    assert pivots == tuple(range(128))
    assert torch.equal(
        basis.T @ basis,
        torch.eye(128, dtype=torch.float64),
    )
    assert elapsed_seconds < 20.0


def test_rectangular_polar_recovers_repeated_spectrum_svd_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The production exact-4x body bridge has a 653-dimensional completion.
    # A repeated singular spectrum at that size can make LAPACK's direct SVD
    # report non-convergence even though the polar factor is unambiguous.
    width = 653
    row = torch.arange(width, dtype=torch.float64).unsqueeze(1)
    column = torch.arange(width, dtype=torch.float64).unsqueeze(0)
    value = (
        torch.cos(torch.pi / width * (row + 0.5) * column)
        * (2.0 / width) ** 0.5
    )
    value[:, 0] /= 2.0**0.5

    def fail_direct_svd(*_args, **_kwargs):
        raise torch.linalg.LinAlgError(
            "forced repeated-singular-value convergence failure"
        )

    monkeypatch.setattr(completion.torch.linalg, "svd", fail_direct_svd)
    orientation, report = completion._rectangular_polar_orientation(
        value,
        relative_tolerance=1e-12,
    )

    assert torch.allclose(orientation, value, rtol=2e-13, atol=2e-13)
    assert torch.allclose(
        orientation.T @ orientation,
        torch.eye(width, dtype=torch.float64),
        rtol=2e-13,
        atol=2e-13,
    )
    assert report["input_rank"] == width
    assert report["rank_deficient_nullspace_completed_canonically"] is False
    assert report["direct_svd_convergence_failure_recovered"] is True
    assert report["decomposition"] == (
        "symmetric-right-gram-eigh-after-direct-svd-failure"
    )


def test_rectangular_polar_gram_fallback_completes_rank_deficiency() -> None:
    generator = torch.Generator().manual_seed(712)
    value = torch.randn(9, 3, generator=generator, dtype=torch.float64) @ torch.randn(
        3,
        5,
        generator=generator,
        dtype=torch.float64,
    )

    orientation, report = (
        completion._rectangular_polar_orientation_from_right_gram(
            value,
            relative_tolerance=1e-12,
        )
    )

    assert report["input_rank"] == 3
    assert report["rank_deficient_nullspace_completed_canonically"] is True
    assert torch.allclose(
        orientation.T @ orientation,
        torch.eye(5, dtype=torch.float64),
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.trace(orientation.T @ value).item() == pytest.approx(
        torch.linalg.svdvals(value).sum().item(),
        rel=2e-12,
        abs=2e-12,
    )


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


def _reseal_report(report: dict[str, object]) -> dict[str, object]:
    report.pop("result_sha256", None)
    report["result_sha256"] = completion.canonical_sha256(report)
    return report


@pytest.fixture(scope="module")
def authenticated_frontier():
    case = _case(seed=2871)
    return compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=case["expected_evidence_sha256"],
        bridge_prior=case["bridge_prior"],
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "recipe",
        "coordinate-map-shape",
        "coordinate-map-sha256",
        "nullspace-update-shape-drop",
        "nullspace-update-sha256",
    ),
)
def test_candidate_rejects_resealed_malformed_direct_reconstruction(
    authenticated_frontier,
    mutation: str,
) -> None:
    candidate = authenticated_frontier.candidates[0]
    report = candidate.as_dict()
    recipe = candidate.recipe
    if mutation == "schema":
        report["schema"] = "not-a-coordinate-completion-candidate"
    elif mutation == "recipe":
        report["recipe"] = "operator-fisher-balanced"
        if report["recipe"] == recipe:
            report["recipe"] = "operator-sensitive-reflected"
    elif mutation == "coordinate-map-shape":
        report["coordinate_map_shape"] = [1, 1]
    elif mutation == "coordinate-map-sha256":
        report["coordinate_map_sha256"] = "a" * 64
    elif mutation == "nullspace-update-shape-drop":
        report.pop("nullspace_update_shape")
    else:
        report["nullspace_update_sha256"] = "b" * 64
    _reseal_report(report)

    with pytest.raises(AnalyticCoordinateCompletionError):
        completion.AnalyticCoordinateCompletionCandidate(
            recipe=recipe,
            coordinate_map=candidate.coordinate_map,
            nullspace_update=candidate.nullspace_update,
            report=report,
        )


def test_candidate_rejects_resealed_unknown_recipe_direct_reconstruction(
    authenticated_frontier,
) -> None:
    candidate = authenticated_frontier.candidates[0]
    report = candidate.as_dict()
    report["recipe"] = "invented-resealed-recipe"
    _reseal_report(report)

    with pytest.raises(AnalyticCoordinateCompletionError, match="recipe"):
        completion.AnalyticCoordinateCompletionCandidate(
            recipe="invented-resealed-recipe",
            coordinate_map=candidate.coordinate_map,
            nullspace_update=candidate.nullspace_update,
            report=report,
        )


def _reconstruct_frontier(
    frontier,
    report: dict[str, object],
    *,
    candidates=None,
):
    return completion.AnalyticCoordinateCompletionFrontier(
        identified_map=frontier.identified_map,
        identified_target_basis=frontier.identified_target_basis,
        target_null_basis=frontier.target_null_basis,
        occupied_source_basis=frontier.occupied_source_basis,
        completion_source_basis=frontier.completion_source_basis,
        candidates=frontier.candidates if candidates is None else candidates,
        report=report,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "identified-map-sha256",
        "identified-target-basis-sha256-drop",
        "target-null-basis-sha256",
        "occupied-source-basis-sha256",
        "completion-source-basis-sha256",
        "candidate-count",
        "candidate-recipes",
        "candidate-results",
        "target-width",
        "trace-rank",
        "completion-rank",
    ),
)
def test_frontier_rejects_resealed_malformed_direct_reconstruction(
    authenticated_frontier,
    mutation: str,
) -> None:
    report = authenticated_frontier.as_dict()
    if mutation == "schema":
        report["schema"] = "not-a-coordinate-completion-frontier"
    elif mutation == "identified-map-sha256":
        report["identified_map_sha256"] = "1" * 64
    elif mutation == "identified-target-basis-sha256-drop":
        report.pop("identified_target_basis_sha256")
    elif mutation == "target-null-basis-sha256":
        report["target_null_basis_sha256"] = "2" * 64
    elif mutation == "occupied-source-basis-sha256":
        report["occupied_source_basis_sha256"] = "3" * 64
    elif mutation == "completion-source-basis-sha256":
        report["completion_geometry"]["completion_source_basis_sha256"] = "4" * 64
    elif mutation == "candidate-count":
        report["candidate_count"] += 1
    elif mutation == "candidate-recipes":
        report["candidate_recipes"] = list(reversed(report["candidate_recipes"]))
    elif mutation == "candidate-results":
        report["candidate_result_sha256s"] = list(
            reversed(report["candidate_result_sha256s"])
        )
    elif mutation == "target-width":
        report["target_width"] += 1
    elif mutation == "trace-rank":
        report["trace_identified_rank"] += 1
    else:
        report["completion_rank"] += 1
    _reseal_report(report)

    with pytest.raises(AnalyticCoordinateCompletionError):
        _reconstruct_frontier(authenticated_frontier, report)


def test_frontier_rejects_resealed_candidate_substitution() -> None:
    first_case = _case(seed=2872)
    second_case = _case(seed=2873)
    first = compile_analytic_coordinate_completion_frontier(
        first_case["trace_design"],
        first_case["fitted_map"],
        first_case["parent_operators"],
        expected_evidence_sha256=first_case["expected_evidence_sha256"],
        bridge_prior=first_case["bridge_prior"],
    )
    second = compile_analytic_coordinate_completion_frontier(
        second_case["trace_design"],
        second_case["fitted_map"],
        second_case["parent_operators"],
        expected_evidence_sha256=second_case["expected_evidence_sha256"],
        bridge_prior=second_case["bridge_prior"],
    )
    substituted = (second.candidates[0], *first.candidates[1:])
    report = first.as_dict()
    report["candidate_recipes"] = [value.recipe for value in substituted]
    report["candidate_result_sha256s"] = [
        value.report["result_sha256"] for value in substituted
    ]
    _reseal_report(report)

    with pytest.raises(
        AnalyticCoordinateCompletionError,
        match="contradicts",
    ):
        _reconstruct_frontier(first, report, candidates=substituted)


def test_operator_evidence_remains_the_exact_v1_byte_recipe() -> None:
    case = _case(seed=301)
    operators = completion._operator_manifest(case["parent_operators"])
    expected = completion.canonical_sha256(
        {
            "schema": completion.ANALYTIC_COORDINATE_COMPLETION_EVIDENCE_SCHEMA,
            "trace_design_sha256": completion._tensor_sha256(case["trace_design"]),
            "fitted_map_sha256": completion._tensor_sha256(case["fitted_map"]),
            "bridge_prior_sha256": completion._tensor_sha256(case["bridge_prior"]),
            "parent_operator_sha256s": [value.operator_sha256 for value in operators],
            "cached_source_fisher_gram_sha256": None,
            "cached_source_fisher_receipt_sha256": None,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "trace_capture_calls": 0,
        }
    )

    assert case["expected_evidence_sha256"] == expected


def test_preaggregated_geometry_evidence_is_the_bound_v2_recipe() -> None:
    case = _case(seed=352)
    geometry = compile_role_balanced_operator_geometry(case["parent_operators"])
    snapshot = completion._role_balanced_geometry_snapshot(geometry)
    expected = completion.canonical_sha256(
        {
            "schema": (
                completion.ANALYTIC_COORDINATE_COMPLETION_GEOMETRY_EVIDENCE_SCHEMA
            ),
            "trace_design_sha256": completion._tensor_sha256(case["trace_design"]),
            "fitted_map_sha256": completion._tensor_sha256(case["fitted_map"]),
            "bridge_prior_sha256": completion._tensor_sha256(case["bridge_prior"]),
            "parent_geometry_result_sha256": geometry.report["result_sha256"],
            "parent_geometry_sha256": geometry.geometry_sha256,
            "parent_geometry_raw_gram_sha256": completion._tensor_sha256(
                geometry.raw_gram
            ),
            "parent_geometry_balanced_gram_sha256": completion._tensor_sha256(
                geometry.balanced_gram
            ),
            "parent_operator_sha256s": list(geometry.ordered_operator_sha256s),
            "cached_source_fisher_gram_sha256": None,
            "cached_source_fisher_receipt_sha256": None,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "trace_capture_calls": 0,
        }
    )

    assert snapshot.geometry_sha256 == geometry.geometry_sha256
    assert (
        analytic_coordinate_completion_evidence_sha256(
            case["trace_design"],
            case["fitted_map"],
            parent_geometry=geometry,
            bridge_prior=case["bridge_prior"],
        )
        == expected
    )


@pytest.mark.parametrize("with_fisher", (False, True))
def test_preaggregated_cpu_geometry_is_map_exact_to_operator_path(
    with_fisher: bool,
) -> None:
    case = _case(seed=404 + int(with_fisher))
    geometry = compile_role_balanced_operator_geometry(
        case["parent_operators"],
        compute_device="cpu",
    )
    fisher = None
    fisher_receipt = None
    if with_fisher:
        source_width = int(case["fitted_map"].shape[1])
        generator = torch.Generator().manual_seed(990)
        factor = torch.randn(
            source_width + 3,
            source_width,
            generator=generator,
            dtype=torch.float64,
        )
        fisher = (factor.T @ factor).contiguous()
        fisher_receipt = "f" * 64
    evidence_kwargs = {
        "bridge_prior": case["bridge_prior"],
        "cached_source_fisher_gram": fisher,
        "cached_source_fisher_receipt_sha256": fisher_receipt,
    }
    legacy_evidence = analytic_coordinate_completion_evidence_sha256(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        **evidence_kwargs,
    )
    geometry_evidence = analytic_coordinate_completion_evidence_sha256(
        case["trace_design"],
        case["fitted_map"],
        parent_geometry=geometry,
        **evidence_kwargs,
    )
    legacy = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        case["parent_operators"],
        expected_evidence_sha256=legacy_evidence,
        **evidence_kwargs,
    )
    preaggregated = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        expected_evidence_sha256=geometry_evidence,
        parent_geometry=geometry,
        **evidence_kwargs,
    )

    assert geometry_evidence != legacy_evidence
    assert geometry_evidence == analytic_coordinate_completion_evidence_sha256(
        case["trace_design"],
        case["fitted_map"],
        parent_geometry=geometry,
        **evidence_kwargs,
    )
    preaggregated_geometry = dict(preaggregated.report["parent_geometry"])
    provenance_keys = {
        "source_mode",
        "parent_geometry_result_sha256",
        "parent_geometry_sha256",
        "parent_geometry_raw_gram_sha256",
        "parent_geometry_balanced_gram_sha256",
        "parent_operator_manifest_sha256",
        "parent_operator_sha256s",
        "preaggregated_parent_geometry",
        "parent_operator_reconstruction_calls",
    }
    navigation_geometry = {
        key: value
        for key, value in preaggregated_geometry.items()
        if key not in provenance_keys
    }
    assert navigation_geometry == legacy.report["parent_geometry"]
    assert (
        preaggregated_geometry["source_mode"] == "preaggregated-role-balanced-geometry"
    )
    assert (
        preaggregated_geometry["parent_geometry_result_sha256"]
        == (geometry.report["result_sha256"])
    )
    assert preaggregated_geometry["parent_geometry_sha256"] == geometry.geometry_sha256
    assert (
        preaggregated_geometry["parent_geometry_raw_gram_sha256"]
        == (geometry.report["hashes"]["raw_gram_sha256"])
    )
    assert (
        preaggregated_geometry["parent_geometry_balanced_gram_sha256"]
        == (geometry.report["hashes"]["balanced_gram_sha256"])
    )
    assert (
        preaggregated_geometry["parent_operator_manifest_sha256"]
        == (geometry.report["hashes"]["operator_manifest_sha256"])
    )
    assert preaggregated_geometry["parent_operator_sha256s"] == list(
        geometry.ordered_operator_sha256s
    )
    assert preaggregated_geometry["preaggregated_parent_geometry"] is True
    assert preaggregated_geometry["parent_operator_reconstruction_calls"] == 0
    assert (
        preaggregated.report["evidence_policy"]["cached_parent_operator_tensors_only"]
        is False
    )
    assert preaggregated.report["evidence_policy"][
        "authenticated_preaggregated_role_balanced_geometry_only"
    ]
    assert "source_mode" not in legacy.report["parent_geometry"]
    assert legacy.report["evidence_policy"]["cached_parent_operator_tensors_only"]
    assert (
        preaggregated.report["trace_identified_rank"]
        == legacy.report["trace_identified_rank"]
    )
    assert preaggregated.report["completion_rank"] == legacy.report["completion_rank"]
    assert [value.recipe for value in preaggregated.candidates] == [
        value.recipe for value in legacy.candidates
    ]
    assert torch.equal(
        preaggregated.completion_source_basis,
        legacy.completion_source_basis,
    )
    for compact, direct in zip(
        preaggregated.candidates,
        legacy.candidates,
        strict=True,
    ):
        assert torch.equal(compact.coordinate_map, direct.coordinate_map)
        assert torch.equal(compact.nullspace_update, direct.nullspace_update)
        assert torch.equal(
            case["trace_design"] @ compact.coordinate_map,
            case["trace_design"] @ direct.coordinate_map,
        )
        assert compact.report["materialized_rank"] == direct.report["materialized_rank"]


@pytest.mark.parametrize("gram_name", ("raw_gram", "balanced_gram"))
def test_preaggregated_geometry_mutation_is_rejected(gram_name: str) -> None:
    case = _case(seed=511)
    geometry = compile_role_balanced_operator_geometry(case["parent_operators"])
    getattr(geometry, gram_name)[0, 0].add_(1.0)

    with pytest.raises(
        AnalyticCoordinateCompletionError,
        match="authentication failed",
    ):
        analytic_coordinate_completion_evidence_sha256(
            case["trace_design"],
            case["fitted_map"],
            parent_geometry=geometry,
            bridge_prior=case["bridge_prior"],
        )


def test_preaggregated_geometry_wrong_width_is_rejected() -> None:
    case = _case(seed=612)
    wrong_width = int(case["fitted_map"].shape[1]) + 1
    wrong_geometry = compile_role_balanced_operator_geometry(
        (
            ResidualFacingOperator(
                role="consumer",
                name="wrong_width",
                tensor=torch.eye(wrong_width, dtype=torch.float64),
                orientation=FeatureOrientation.INPUT,
            ),
        )
    )

    with pytest.raises(
        AnalyticCoordinateCompletionError,
        match="geometry width",
    ):
        analytic_coordinate_completion_evidence_sha256(
            case["trace_design"],
            case["fitted_map"],
            parent_geometry=wrong_geometry,
            bridge_prior=case["bridge_prior"],
        )


def test_completion_requires_exactly_one_operator_geometry_source() -> None:
    case = _case(seed=713)
    geometry = compile_role_balanced_operator_geometry(case["parent_operators"])
    for parent_operators, parent_geometry in (
        (None, None),
        (case["parent_operators"], geometry),
    ):
        with pytest.raises(
            AnalyticCoordinateCompletionError,
            match="Exactly one",
        ):
            analytic_coordinate_completion_evidence_sha256(
                case["trace_design"],
                case["fitted_map"],
                parent_operators,
                parent_geometry=parent_geometry,
                bridge_prior=case["bridge_prior"],
            )
        with pytest.raises(
            AnalyticCoordinateCompletionError,
            match="Exactly one",
        ):
            compile_analytic_coordinate_completion_frontier(
                case["trace_design"],
                case["fitted_map"],
                parent_operators,
                parent_geometry=parent_geometry,
                expected_evidence_sha256="0" * 64,
                bridge_prior=case["bridge_prior"],
            )


def test_preaggregated_path_never_recomputes_operator_grams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(seed=814)
    geometry = compile_role_balanced_operator_geometry(case["parent_operators"])
    evidence = analytic_coordinate_completion_evidence_sha256(
        case["trace_design"],
        case["fitted_map"],
        parent_geometry=geometry,
        bridge_prior=case["bridge_prior"],
    )

    def forbidden_operator_gram(_operator: ResidualFacingOperator) -> torch.Tensor:
        raise AssertionError("preaggregated path recomputed an operator Gram")

    monkeypatch.setattr(completion, "_operator_gram", forbidden_operator_gram)
    result = compile_analytic_coordinate_completion_frontier(
        case["trace_design"],
        case["fitted_map"],
        expected_evidence_sha256=evidence,
        parent_geometry=geometry,
        bridge_prior=case["bridge_prior"],
    )

    assert result.candidates


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
        assert (
            torch.linalg.matrix_rank(candidate.coordinate_map).item() == expected_rank
        )
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
        assert (
            candidate.report["maximum_trace_prediction_error"]
            <= (candidate.report["trace_prediction_fp64_roundoff_bound"])
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
    assert not torch.equal(
        first.completion_source_basis, second.completion_source_basis
    )
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
    bridge_coordinates = target_null.T @ case["bridge_prior"] @ source_completion

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
    fitted = (torch.eye(2, dtype=torch.float64) - identified_projector) @ torch.randn(
        2, 2, generator=generator, dtype=torch.float64
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
    fitted = torch.diag(torch.tensor([1e-200, 2e-200], dtype=torch.float64))
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
