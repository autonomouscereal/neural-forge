from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_tied_causal_interface_anchor_fold import (  # noqa: E402
    TiedCausalAnchorFoldError,
    TiedCausalAnchorSplit,
    fit_tied_causal_interface_anchor_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)


def _interface() -> TiedCausalInterfaceFold:
    return TiedCausalInterfaceFold(
        target_weight=torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=torch.float64,
        ),
        producer_encoder=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        report={"report_sha256": "0" * 64},
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
        row_ids=tuple(
            f"{prefix}:{index}" for index in range(int(rows.shape[0]))
        ),
        prefix_positions=torch.arange(7, 7 + int(rows.shape[0])),
        source_producer_rows=rows,
        target_next_token_ids=targets,
    )


def _fold(
    *,
    validation_variant: bool = False,
    drift: float = 1.0,
):
    return fit_tied_causal_interface_anchor_residual(
        _interface(),
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


def test_anchor_residual_repairs_fit_decisions_without_extra_parameters() -> None:
    result = _fold()
    report = result.as_dict()

    assert report["split_metrics"]["fit"]["target_anchor_top1_fraction"] == 1.0
    assert (
        report["split_metrics"]["selection"]["target_anchor_top1_fraction"]
        == 1.0
    )
    assert report["producer_encoder_fixed"] is True
    assert report["additional_runtime_parameter_count"] == 0
    assert report["dense_same_shape_tied_table"] is True
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


def test_validation_cannot_select_or_change_the_residual() -> None:
    first = _fold(validation_variant=False)
    second = _fold(validation_variant=True)

    assert first.report["selected_candidate_index"] == (
        second.report["selected_candidate_index"]
    )
    assert torch.equal(first.residual_weight, second.residual_weight)
    assert torch.equal(
        first.interface.target_weight,
        second.interface.target_weight,
    )
    assert first.report["candidate_reports"] == second.report[
        "candidate_reports"
    ]
    assert first.report["split_metrics"]["validation"] != second.report[
        "split_metrics"
    ]["validation"]
    assert (
        first.report["validation_split_used_for_ridge_margin_selection"]
        is False
    )
    assert first.report["validation_first_read_after_candidate_frozen"] is True


def test_drift_is_bounded_and_fold_is_deterministic_and_sealed() -> None:
    first = _fold(drift=0.05)
    second = _fold(drift=0.05)

    assert first.report["relative_table_drift"] <= 0.05 + 1e-12
    assert first.report["drift_bound_satisfied"] is True
    assert any(
        candidate["drift_clipped"]
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
        assert "mapped_parent_top1_agreement" in metrics
        assert "mean_target_anchor_margin" in metrics


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
    with pytest.raises(TiedCausalAnchorFoldError, match="report or tensor seal"):
        type(result)(
            interface=result.interface,
            residual_weight=result.residual_weight,
            report=tampered,
        )
