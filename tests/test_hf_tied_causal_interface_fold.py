from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceCapacityError,
    TiedCausalInterfaceFoldError,
    fit_tied_causal_interface,
)


def _affine_low_rank_fixture():
    generator = torch.Generator().manual_seed(701)
    vocabulary = 17
    source_width = 9
    affine_rank = 4
    basis = torch.randn(
        affine_rank,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    coefficients = torch.randn(
        vocabulary,
        affine_rank,
        generator=generator,
        dtype=torch.float64,
    )
    mean = torch.randn(
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    table = mean.unsqueeze(0) + coefficients @ basis
    fit = torch.randn(
        41,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    validation = torch.randn(
        19,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    return table, fit, validation


def test_affine_rank_fold_installs_one_dense_tied_table_and_preserves_logits():
    table, fit, validation = _affine_low_rank_fixture()

    folded = fit_tied_causal_interface(
        table,
        fit,
        target_width=5,
        validation_producer_rows=validation,
        minimum_validation_top1_agreement=1.0,
        metric_chunk_rows=7,
    )

    assert folded.target_weight.shape == (17, 5)
    assert folded.producer_encoder.shape == (9, 5)
    assert folded.report["target_dense_parameter_count"] == 85
    assert folded.report["mapped_source_dense_parameter_count"] == 153
    assert folded.report["numeric_parent_weight_matching_objective"] is False
    assert folded.report["functional_mapped_parent_logit_objective"] is True
    assert folded.report["validation"]["raw_parent_top1_agreement"] == 1.0
    assert folded.report["validation"]["rmsnorm_parent_top1_agreement"] == 1.0
    assert folded.report["validation"]["raw_centered_relative_rmse"] < 1e-10
    assert len(folded.report["report_sha256"]) == 64

    parent_logits = validation @ table.T
    target_rows = folded.encode_producer_rows(validation)
    child_logits = target_rows @ folded.target_weight.T
    torch.testing.assert_close(
        child_logits - child_logits.mean(dim=1, keepdim=True),
        parent_logits - parent_logits.mean(dim=1, keepdim=True),
        rtol=1e-9,
        atol=1e-9,
    )
    normalized = folded.encode_producer_rows(
        validation,
        rms_normalize=True,
    )
    assert torch.equal(
        (normalized @ folded.target_weight.T).argmax(dim=1),
        parent_logits.argmax(dim=1),
    )


def test_severe_width_reports_real_parameter_contraction_without_training():
    generator = torch.Generator().manual_seed(719)
    table = torch.randn(23, 10, generator=generator)
    fit = torch.randn(37, 10, generator=generator)
    validation = torch.randn(13, 10, generator=generator)

    folded = fit_tied_causal_interface(
        table,
        fit,
        target_width=3,
        validation_producer_rows=validation,
        minimum_validation_top1_agreement=0.0,
    )

    assert folded.target_weight.numel() == 69
    assert table.numel() == 230
    assert folded.report["interface_parameter_reduction_fraction"] == pytest.approx(
        0.7
    )
    assert 0.0 <= folded.report["validation"][
        "rmsnorm_parent_top1_agreement"
    ] <= 1.0
    assert folded.report["training_steps"] == 0
    assert folded.report["optimizer_steps"] == 0
    assert folded.report["backward_calls"] == 0
    assert folded.report["quantization_used"] is False
    assert folded.report["pruning_used"] is False


def test_validation_gate_and_disjoint_evidence_fail_closed():
    table, fit, validation = _affine_low_rank_fixture()
    with pytest.raises(
        TiedCausalInterfaceCapacityError,
        match="held-out parent top-1",
    ):
        fit_tied_causal_interface(
            table,
            fit,
            target_width=2,
            validation_producer_rows=validation,
            minimum_validation_top1_agreement=1.0,
        )

    with pytest.raises(
        TiedCausalInterfaceFoldError,
        match="disjoint objects",
    ):
        fit_tied_causal_interface(
            table,
            fit,
            target_width=5,
            validation_producer_rows=fit,
        )
