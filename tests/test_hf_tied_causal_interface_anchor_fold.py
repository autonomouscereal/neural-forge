from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_tied_causal_interface_anchor_fold as anchor_module  # noqa: E402
from neural_forge.hf_tied_causal_interface_anchor_fold import (  # noqa: E402
    TiedCausalAnchorCandidateRecipe,
    TiedCausalAnchorFoldError,
    TiedCausalAnchorFrontier,
    TiedCausalAnchorSplit,
    enumerate_tied_causal_anchor_candidates,
    fit_tied_causal_interface_anchor_residual,
    materialize_tied_causal_anchor_candidate,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _interface_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sealed_report(body: dict) -> dict:
    return {**body, "report_sha256": _canonical_sha256(body)}


def _interface() -> TiedCausalInterfaceFold:
    target_weight = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    producer_encoder = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    report = _sealed_report(
        {
            "schema": "neural-forge-tied-causal-interface-fold-v1",
            "source_width": 3,
            "target_width": 2,
            "vocabulary_size": 4,
            "target_weight_sha256": _interface_tensor_sha256(target_weight),
            "producer_encoder_sha256": _interface_tensor_sha256(producer_encoder),
            "source_table_sha256": _interface_tensor_sha256(_source_table()),
        }
    )
    return TiedCausalInterfaceFold(
        target_weight=target_weight,
        producer_encoder=producer_encoder,
        report=report,
    )


def _source_table() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )


def _split(
    name: str,
    prefix: str,
    *,
    validation_variant: bool = False,
) -> TiedCausalAnchorSplit:
    if validation_variant:
        rows = torch.tensor(
            [
                [0.2, -1.0, 0.0],
                [-0.2, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        targets = torch.tensor([2, 3])
    else:
        rows = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        # The mapped parent prefers 0 and 1 respectively.  Anchors request the
        # opposite row in the same causal coordinate.  The other vocabulary
        # axis remains untouched, so this repair fits inside a 100% table
        # drift cap instead of asking the test to reverse the whole table.
        targets = torch.tensor([1, 0])
    return TiedCausalAnchorSplit(
        name=name,
        row_ids=tuple(f"{prefix}:{index}" for index in range(int(rows.shape[0]))),
        prefix_positions=torch.arange(7, 7 + int(rows.shape[0])),
        source_producer_rows=rows,
        target_next_token_ids=targets,
    )


def _fold(
    *,
    validation_variant: bool = False,
    drift: float = 1.0,
    base_interface: TiedCausalInterfaceFold | None = None,
):
    return fit_tied_causal_interface_anchor_residual(
        _interface() if base_interface is None else base_interface,
        _source_table(),
        selection_split=_split("selection", "selection"),
        fit_split=_split("fit", "fit"),
        validation_split=_split(
            "validation",
            "validation",
            validation_variant=validation_variant,
        ),
        ridge_candidates=(1e-8, 1e-4),
        margin_candidates=(0.25, 1.0),
        maximum_relative_table_drift=drift,
        metric_chunk_rows=2,
    )


def test_effectful_anchor_survives_when_every_numeric_review_goal_regresses() -> None:
    result = _fold()
    report = result.as_dict()

    assert (
        report["selected_candidate_index"]
        == (report["diagnostic_candidate_indices_best_to_worst"][0])
    )
    assert report["selected_candidate_is_base_control"] is False
    assert report["active_anchor_residual_selected"] is True
    assert report["base_control_candidate_included"] is True
    assert report["base_control_fallback_triggered"] is False
    assert report["base_control_used_only_for_structural_fallback"] is True
    assert report["eligible_candidate_indices"] == [1, 2, 3, 4]
    assert report["structurally_eligible_candidate_indices"] == [1, 2, 3, 4]
    assert report["numeric_thresholds_used_for_candidate_eligibility"] is False
    assert report["numeric_metrics_used_for_base_fallback"] is False
    assert report["target_anchor_metrics_diagnostic_only"] is True
    assert report["target_anchor_metrics_used_for_candidate_selection"] is False
    assert (
        report["minimum_mapped_parent_behavior_retention_is_diagnostic_review_goal"]
        is True
    )
    assert (
        report[
            "minimum_mapped_parent_behavior_retention_used_as_candidate_eligibility_gate"
        ]
        is False
    )
    assert torch.count_nonzero(result.residual_weight).item() > 0
    assert not torch.equal(result.interface.target_weight, _interface().target_weight)
    base_candidate = report["candidate_reports"][0]
    assert base_candidate["candidate_kind"] == "unmodified_base_interface"
    assert base_candidate["candidate_eligible_for_selection"] is False
    assert base_candidate["tensor_identical_to_base"] is True
    assert base_candidate["selection_metrics"]["mapped_parent_top1_agreement"] == 1.0
    assert (
        base_candidate["selection_metrics"][
            "mapped_parent_raw_centered_logit_retained_energy_fraction"
        ]
        == 1.0
    )
    analytic = report["candidate_reports"][1:]
    assert all(candidate["candidate_eligible_for_selection"] for candidate in analytic)
    assert all(candidate["effectful_residual"] for candidate in analytic)
    assert all(
        candidate["all_mapped_parent_behavior_review_goals_met"] is False
        for candidate in analytic
    )
    assert all(
        candidate["selection_metrics"]["mapped_parent_top1_agreement"]
        < base_candidate["selection_metrics"]["mapped_parent_top1_agreement"]
        for candidate in analytic
    )
    assert all(
        candidate["selection_metrics"][
            "mapped_parent_raw_centered_logit_retained_energy_fraction"
        ]
        < base_candidate["selection_metrics"][
            "mapped_parent_raw_centered_logit_retained_energy_fraction"
        ]
        for candidate in analytic
    )
    assert all(
        candidate["selection_metrics"]["target_anchor_top1_fraction"] == 1.0
        for candidate in analytic
    )
    assert report["producer_encoder_fixed"] is True
    assert report["additional_runtime_parameter_count"] == 0
    assert report["dense_same_shape_tied_table"] is True
    base = _interface()
    assert report["base_interface_report_sha256"] == base.report["report_sha256"]
    assert report["base_interface_lineage"]["report"] == dict(base.report)
    assert report["base_interface_lineage"]["tensor_identity_verified"] is True
    assert (
        report["base_dense_tied_parameter_count"]
        == report["repaired_dense_tied_parameter_count"]
        == 8
    )
    assert tuple(result.residual_weight.shape) == (4, 2)
    assert tuple(result.interface.target_weight.shape) == (4, 2)
    assert torch.equal(
        result.interface.producer_encoder,
        _interface().producer_encoder,
    )
    assert report["operation_audit"] == {
        "parent_forward_count": 0,
        "parent_generation_count": 0,
        "parent_cache_regeneration_count": 0,
        "target_model_forward_count": 0,
        "training_step_count": 0,
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }


def test_raw_parent_energy_is_not_confused_with_rmsnorm_scale_error() -> None:
    metrics = _fold().report["candidate_reports"][0]["selection_metrics"]

    assert metrics["mapped_parent_raw_centered_logit_retained_energy_fraction"] == 1.0
    assert metrics["mapped_parent_centered_logit_retained_energy_fraction"] < 1.0
    assert metrics["mapped_parent_top1_agreement"] == 1.0
    assert metrics["raw_and_runtime_coordinates_have_same_target_top1"] is True
    assert metrics["raw_target_coordinates_used_for_parent_behavior_selection"] is True


def test_validation_cannot_select_or_change_the_residual() -> None:
    first = _fold(validation_variant=False)
    second = _fold(validation_variant=True)

    assert (
        first.report["selected_candidate_index"]
        == (second.report["selected_candidate_index"])
    )
    assert torch.equal(first.residual_weight, second.residual_weight)
    assert torch.equal(
        first.interface.target_weight,
        second.interface.target_weight,
    )
    assert first.report["candidate_reports"] == second.report["candidate_reports"]
    assert (
        first.report["split_metrics"]["validation"]
        != second.report["split_metrics"]["validation"]
    )
    assert first.report["validation_split_used_for_ridge_margin_selection"] is False
    assert (
        first.report["validation_metrics_first_computed_after_candidate_frozen"] is True
    )
    assert first.report["validation_values_used_for_parameters_or_selection"] is False


def test_drift_is_diagnostic_only_and_fold_is_deterministic_and_sealed() -> None:
    first = _fold(drift=0.05)
    second = _fold(drift=0.05)

    assert first.report["relative_table_drift"] > 0.05
    assert first.report["relative_table_drift_review_goal_met"] is False
    assert first.report["drift_bound_satisfied"] is False
    assert first.report["drift_bound_enforced"] is False
    assert (
        first.report["maximum_relative_table_drift_used_as_candidate_eligibility_gate"]
        is False
    )
    assert all(
        candidate["drift_clipped"] is False
        for candidate in first.report["candidate_reports"]
    )
    assert torch.equal(first.residual_weight, second.residual_weight)
    assert first.report == second.report
    assert first.report["report_sha256"]
    assert first.report["residual_weight_sha256"]
    assert first.report["target_weight_sha256"]
    for name in ("fit", "selection", "validation"):
        metrics = first.report["split_metrics"][name]
        assert "mapped_parent_centered_logit_retained_energy_fraction" in metrics
        assert "mapped_parent_raw_centered_logit_retained_energy_fraction" in metrics
        assert "base_interface_raw_centered_logit_retained_energy_fraction" in metrics
        assert "mapped_parent_top1_agreement" in metrics
        assert "mean_target_anchor_margin" in metrics


def test_base_fallback_requires_every_analytic_candidate_to_be_identical() -> None:
    def zero_coordinate_split(name: str) -> TiedCausalAnchorSplit:
        rows = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float64,
        )
        return TiedCausalAnchorSplit(
            name=name,
            row_ids=(f"{name}:0", f"{name}:1"),
            prefix_positions=torch.tensor([7, 8]),
            source_producer_rows=rows,
            target_next_token_ids=torch.tensor([1, 0]),
        )

    result = fit_tied_causal_interface_anchor_residual(
        _interface(),
        _source_table(),
        selection_split=zero_coordinate_split("selection"),
        fit_split=zero_coordinate_split("fit"),
        validation_split=zero_coordinate_split("validation"),
        ridge_candidates=(1e-8, 1e-4),
        margin_candidates=(0.25, 1.0),
        metric_chunk_rows=2,
    )

    assert result.report["selected_candidate_is_base_control"] is True
    assert result.report["base_control_fallback_triggered"] is True
    assert result.report["structurally_rejected_candidate_indices"] == []
    assert result.report["tensor_identical_candidate_indices"] == [1, 2, 3, 4]
    assert result.report["eligible_candidate_indices"] == [0]
    assert all(
        candidate["structurally_valid"] is True
        and candidate["tensor_identical_to_base"] is True
        and candidate["effectful_residual"] is False
        for candidate in result.report["candidate_reports"][1:]
    )
    assert torch.equal(result.interface.target_weight, _interface().target_weight)


def test_base_fallback_requires_every_analytic_candidate_to_be_invalid(
    monkeypatch,
) -> None:
    def wrong_dtype_residual(
        base_weight,
        _fit_coordinates,
        _target_ids,
        _competitor_ids,
        *,
        ridge,
        margin,
        maximum_relative_table_drift,
    ):
        return torch.ones_like(base_weight, dtype=torch.float32), {
            "ridge_fraction": ridge,
            "requested_margin": margin,
            "relative_table_drift": maximum_relative_table_drift,
        }

    monkeypatch.setattr(anchor_module, "_fit_residual", wrong_dtype_residual)
    result = _fold()

    assert result.report["selected_candidate_is_base_control"] is True
    assert result.report["base_control_fallback_triggered"] is True
    assert result.report["structurally_rejected_candidate_indices"] == [1, 2, 3, 4]
    assert result.report["tensor_identical_candidate_indices"] == []
    assert result.report["eligible_candidate_indices"] == [0]
    assert all(
        candidate["structural_rejection_reason"] == "residual_dtype_differs_from_base"
        for candidate in result.report["candidate_reports"][1:]
    )
    assert torch.equal(result.interface.target_weight, _interface().target_weight)


def test_split_ids_must_be_disjoint_and_target_ids_must_fit_vocabulary() -> None:
    selection = _split("selection", "shared")
    fit = TiedCausalAnchorSplit(
        name="fit",
        row_ids=selection.row_ids,
        prefix_positions=selection.prefix_positions,
        source_producer_rows=selection.source_producer_rows,
        target_next_token_ids=selection.target_next_token_ids,
    )
    with pytest.raises(TiedCausalAnchorFoldError, match="not disjoint"):
        fit_tied_causal_interface_anchor_residual(
            _interface(),
            _source_table(),
            selection_split=selection,
            fit_split=fit,
            validation_split=_split("validation", "validation"),
        )

    invalid = _split("selection", "invalid")
    invalid_targets = invalid.target_next_token_ids.clone()
    invalid_targets[0] = 4
    invalid = TiedCausalAnchorSplit(
        name="selection",
        row_ids=invalid.row_ids,
        prefix_positions=invalid.prefix_positions,
        source_producer_rows=invalid.source_producer_rows,
        target_next_token_ids=invalid_targets,
    )
    with pytest.raises(TiedCausalAnchorFoldError, match="exceeds"):
        fit_tied_causal_interface_anchor_residual(
            _interface(),
            _source_table(),
            selection_split=invalid,
            fit_split=_split("fit", "fit"),
            validation_split=_split("validation", "validation"),
        )


def test_fit_rows_above_legacy_ceiling_use_one_global_solve_without_rejection() -> None:
    result = fit_tied_causal_interface_anchor_residual(
        _interface(),
        _source_table(),
        selection_split=_split("selection", "selection"),
        fit_split=_split("fit", "fit"),
        validation_split=_split(
            "validation",
            "validation",
            validation_variant=True,
        ),
        maximum_fit_anchor_rows=1,
    )

    assert result.report["fit_anchor_row_ceiling_satisfied"] is False
    assert result.report["fit_solver_batch_count"] == 1
    assert result.report["fit_anchor_rows_scheduled"] == 2
    assert result.report["fit_anchor_rows_rejected"] == 0
    assert (
        result.report["maximum_fit_anchor_rows_used_as_evidence_rejection_gate"]
        is False
    )
    assert result.report["materializable_candidate_indices"] == list(range(10))
    analytic = result.report["candidate_reports"][1:]
    assert len(analytic) == 9
    assert all(candidate["fit_solver_batch_count"] == 1 for candidate in analytic)
    assert all(
        candidate["fit_solver_batch_row_counts"] == [2]
        for candidate in analytic
    )
    assert all(
        candidate["fit_rows_rejected_by_solver_batch_bound"] == 0
        for candidate in analytic
    )
    assert all(
        candidate["global_solution_independent_of_maximum_fit_anchor_rows"] is True
        and candidate["maximum_fit_anchor_rows_is_legacy_diagnostic_only"] is True
        for candidate in analytic
    )


def test_legacy_fit_row_ceiling_cannot_change_candidate_bytes_or_order() -> None:
    arguments = {
        "selection_split": _split("selection", "selection"),
        "fit_split": _split("fit", "fit"),
        "validation_split": _split("validation", "validation"),
        "ridge_candidates": (1e-8, 1e-4),
        "margin_candidates": (0.25, 1.0),
        "maximum_relative_table_drift": 1.0,
        "metric_chunk_rows": 2,
    }
    below_ceiling = fit_tied_causal_interface_anchor_residual(
        _interface(),
        _source_table(),
        maximum_fit_anchor_rows=2,
        **arguments,
    )
    above_ceiling = fit_tied_causal_interface_anchor_residual(
        _interface(),
        _source_table(),
        maximum_fit_anchor_rows=1,
        **arguments,
    )

    assert (
        below_ceiling.report["selected_candidate_index"]
        == above_ceiling.report["selected_candidate_index"]
    )
    assert torch.equal(
        below_ceiling.interface.target_weight,
        above_ceiling.interface.target_weight,
    )
    assert (
        below_ceiling.report["diagnostic_candidate_indices_best_to_worst"]
        == above_ceiling.report["diagnostic_candidate_indices_best_to_worst"]
    )
    for index in below_ceiling.frontier.materializable_candidate_indices:
        assert torch.equal(
            below_ceiling.materialize_candidate(index).interface.target_weight,
            above_ceiling.materialize_candidate(index).interface.target_weight,
        )


def test_complete_structural_frontier_is_lazy_and_materializable() -> None:
    result = _fold()
    frontier = result.frontier

    assert isinstance(frontier, TiedCausalAnchorFrontier)
    assert result.report["materializable_candidate_indices"] == [0, 1, 2, 3, 4]
    assert frontier.materializable_candidate_indices == (0, 1, 2, 3, 4)
    assert result.report["all_structurally_valid_candidates_retained"] is True
    assert result.report["numeric_metrics_used_for_candidate_admission"] is False
    assert result.report["numeric_metrics_have_final_intelligence_authority"] is False
    assert result.report["whole_response_review_selects_final_candidate"] is True
    assert all(
        isinstance(recipe, TiedCausalAnchorCandidateRecipe)
        for recipe in frontier.recipes
    )

    numeric_order = enumerate_tied_causal_anchor_candidates(
        result,
        review_order=False,
    )
    review_order = result.enumerate_candidate_recipes(review_order=True)
    assert [recipe["candidate_index"] for recipe in numeric_order] == list(range(5))
    assert {recipe["candidate_index"] for recipe in review_order} == set(range(5))
    assert [recipe["candidate_index"] for recipe in review_order[:4]] == (
        result.report["diagnostic_candidate_indices_best_to_worst"]
    )
    assert all("target_weight" not in recipe for recipe in numeric_order)
    assert all("residual_weight" not in recipe for recipe in numeric_order)

    materialized = {
        index: materialize_tied_causal_anchor_candidate(result, index)
        for index in frontier.materializable_candidate_indices
    }
    assert torch.equal(
        materialized[result.report["selected_candidate_index"]].residual_weight,
        result.residual_weight,
    )
    assert torch.equal(materialized[0].interface.target_weight, _interface().target_weight)
    assert all(
        tuple(candidate.interface.target_weight.shape) == (4, 2)
        and candidate.report["selected_candidate_index"] == index
        and candidate.report["operation_audit"]["training_step_count"] == 0
        and candidate.report["operation_audit"]["parent_forward_count"] == 0
        for index, candidate in materialized.items()
    )
    assert len(
        {
            candidate.report["candidate_recipe_sha256"]
            for candidate in materialized.values()
        }
    ) == 5


def test_tensor_identical_analytic_recipes_are_preserved_for_review() -> None:
    def zero_coordinate_split(name: str) -> TiedCausalAnchorSplit:
        rows = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float64,
        )
        return TiedCausalAnchorSplit(
            name=name,
            row_ids=(f"{name}:0", f"{name}:1"),
            prefix_positions=torch.tensor([7, 8]),
            source_producer_rows=rows,
            target_next_token_ids=torch.tensor([1, 0]),
        )

    result = fit_tied_causal_interface_anchor_residual(
        _interface(),
        _source_table(),
        selection_split=zero_coordinate_split("selection"),
        fit_split=zero_coordinate_split("fit"),
        validation_split=zero_coordinate_split("validation"),
        ridge_candidates=(1e-8, 1e-4),
        margin_candidates=(0.25, 1.0),
        metric_chunk_rows=2,
    )

    assert result.report["selected_candidate_index"] == 0
    assert result.report["materializable_candidate_indices"] == [0, 1, 2, 3, 4]
    assert result.report["tensor_identical_candidate_indices"] == [1, 2, 3, 4]
    assert all(
        recipe["original_tensor_identical_to_base"] is True
        for recipe in result.enumerate_candidate_recipes(review_order=False)[1:]
    )
    assert all(
        torch.equal(
            result.materialize_candidate(index).interface.target_weight,
            result.interface.target_weight,
        )
        for index in range(1, 5)
    )


def test_frontier_and_recipe_authentication_fail_closed() -> None:
    result = _fold()
    frontier = result.frontier
    assert isinstance(frontier, TiedCausalAnchorFrontier)

    with pytest.raises(TiedCausalAnchorFoldError, match="recipe authentication"):
        replace(frontier.recipes[1], recipe_sha256="0" * 64)
    with pytest.raises(TiedCausalAnchorFoldError, match="frontier authentication"):
        replace(frontier, frontier_sha256="0" * 64)
    with pytest.raises(TiedCausalAnchorFoldError, match="not materializable"):
        result.materialize_candidate(999)

    wrong_output_recipe = replace(
        frontier.recipes[1],
        original_target_weight_sha256="0" * 64,
        recipe_sha256="",
    )
    wrong_output_frontier = replace(
        frontier,
        recipes=(frontier.recipes[0], wrong_output_recipe, *frontier.recipes[2:]),
        frontier_sha256="",
    )
    with pytest.raises(
        TiedCausalAnchorFoldError,
        match="changed the authenticated candidate bytes",
    ):
        wrong_output_frontier.materialize(1)

    tampered_report = copy.deepcopy(dict(result.report))
    selected_index = tampered_report["selected_candidate_index"]
    tampered_report["candidate_reports"][selected_index]["requested_margin"] += 0.5
    tampered_body = dict(tampered_report)
    tampered_body.pop("report_sha256")
    tampered_report["report_sha256"] = _canonical_sha256(tampered_body)
    tampered_interface = TiedCausalInterfaceFold(
        target_weight=result.interface.target_weight,
        producer_encoder=result.interface.producer_encoder,
        report=tampered_report,
    )
    with pytest.raises(TiedCausalAnchorFoldError, match="candidate frontier"):
        type(result)(
            interface=tampered_interface,
            residual_weight=result.residual_weight,
            report=tampered_report,
            frontier=frontier,
        )


def test_base_interface_ancestry_tampering_fails_closed() -> None:
    base = _interface()
    invalid_body = dict(base.report)
    invalid_body.pop("report_sha256")
    invalid_body["target_weight_sha256"] = "0" * 64
    invalid_base = TiedCausalInterfaceFold(
        target_weight=base.target_weight,
        producer_encoder=base.producer_encoder,
        report=_sealed_report(invalid_body),
    )
    with pytest.raises(TiedCausalAnchorFoldError, match="authenticate"):
        _fold(base_interface=invalid_base)

    result = _fold()
    tampered = copy.deepcopy(dict(result.report))
    lineage = tampered["base_interface_lineage"]
    nested_body = dict(lineage["report"])
    nested_body.pop("report_sha256")
    nested_body["producer_encoder_sha256"] = "1" * 64
    nested_report = _sealed_report(nested_body)
    lineage["report"] = nested_report
    lineage["report_sha256"] = nested_report["report_sha256"]
    lineage["producer_encoder_sha256"] = "1" * 64
    tampered["base_interface_report_sha256"] = nested_report["report_sha256"]
    tampered_body = dict(tampered)
    tampered_body.pop("report_sha256")
    tampered["report_sha256"] = _canonical_sha256(tampered_body)
    tampered_interface = TiedCausalInterfaceFold(
        target_weight=result.interface.target_weight,
        producer_encoder=result.interface.producer_encoder,
        report=tampered,
    )
    with pytest.raises(TiedCausalAnchorFoldError, match="ancestry"):
        type(result)(
            interface=tampered_interface,
            residual_weight=result.residual_weight,
            report=tampered,
        )


def test_nonfinite_inputs_and_report_tampering_fail_closed() -> None:
    values = _split("fit", "fit").source_producer_rows.clone()
    values[0, 0] = torch.nan
    with pytest.raises(TiedCausalAnchorFoldError, match="NaN or infinity"):
        TiedCausalAnchorSplit(
            name="fit",
            row_ids=("a", "b"),
            prefix_positions=torch.arange(2),
            source_producer_rows=values,
            target_next_token_ids=torch.tensor([0, 1]),
        )

    result = _fold()
    tampered = copy.deepcopy(dict(result.report))
    tampered["additional_runtime_parameter_count"] = 1
    with pytest.raises(
        TiedCausalAnchorFoldError,
        match="report, tensor seal, or base ancestry",
    ):
        type(result)(
            interface=result.interface,
            residual_weight=result.residual_weight,
            report=tampered,
        )
