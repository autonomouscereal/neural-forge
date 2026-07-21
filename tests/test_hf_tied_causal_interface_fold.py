from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_tied_causal_interface_fold as interface_module  # noqa: E402
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA,
    TiedCausalInterfaceCapacityError,
    TiedCausalInterfaceFold,
    TiedCausalInterfaceFoldError,
    fit_tied_causal_interface,
    fit_tied_causal_interface_from_statistics,
    fit_tied_causal_interface_with_statistics,
    whiten_tied_causal_interface,
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
    assert folded.report["interface_parameter_reduction_fraction"] == pytest.approx(0.7)
    assert 0.0 <= folded.report["validation"]["rmsnorm_parent_top1_agreement"] <= 1.0
    assert folded.report["training_steps"] == 0
    assert folded.report["optimizer_steps"] == 0
    assert folded.report["backward_calls"] == 0
    assert folded.report["quantization_used"] is False
    assert folded.report["pruning_used"] is False


def test_second_moment_statistics_replay_is_byte_identical_to_row_fit():
    table, fit, validation = _affine_low_rank_fixture()
    fit_weights = torch.linspace(0.25, 1.25, fit.shape[0], dtype=torch.float64)
    validation_weights = torch.linspace(
        1.5,
        0.5,
        validation.shape[0],
        dtype=torch.float64,
    )

    folded, statistics = fit_tied_causal_interface_with_statistics(
        table,
        fit,
        target_width=5,
        validation_producer_rows=validation,
        fit_weights=fit_weights,
        validation_weights=validation_weights,
        metric_chunk_rows=7,
    )
    replayed = fit_tied_causal_interface_from_statistics(table, statistics)

    assert torch.equal(replayed.target_weight, folded.target_weight)
    assert torch.equal(replayed.producer_encoder, folded.producer_encoder)
    assert dict(replayed.report) == dict(folded.report)
    assert statistics.fit_second_moment.shape == (9, 9)
    assert torch.equal(
        statistics.fit_second_moment,
        statistics.fit_second_moment.T,
    )
    assert statistics.fit_metric_witness.rows == fit.shape[0]
    assert statistics.validation_metric_witness.rows == validation.shape[0]
    assert statistics.source_table_sha256 == interface_module._tensor_sha256(table)
    assert statistics.source_table_sha256 == folded.report["source_table_sha256"]
    assert len(statistics.statistics_sha256) == 64


def test_statistics_replay_rejects_same_shape_changed_source_before_solver(
    monkeypatch,
):
    table, fit, validation = _affine_low_rank_fixture()
    _folded, statistics = fit_tied_causal_interface_with_statistics(
        table,
        fit,
        target_width=5,
        validation_producer_rows=validation,
        metric_chunk_rows=7,
    )
    changed = table.clone()
    changed[0, 0] += 1.0
    solver_called = False

    def unexpected_solver(*_args, **_kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("source binding must be checked before the eigensolve")

    monkeypatch.setattr(
        interface_module,
        "_solve_tied_causal_interface_second_moment",
        unexpected_solver,
    )

    with pytest.raises(
        TiedCausalInterfaceFoldError,
        match="differs from the sufficient-statistics binding",
    ):
        fit_tied_causal_interface_from_statistics(changed, statistics)
    assert solver_called is False


def test_self_dual_whitening_preserves_operator_and_orthonormalizes_encoder():
    table, fit, validation = _affine_low_rank_fixture()
    folded = fit_tied_causal_interface(
        table,
        fit,
        target_width=5,
        validation_producer_rows=validation,
    )
    scales = torch.diag(torch.tensor([1e-3, 0.05, 1.0, 20.0, 1e3], dtype=torch.float64))
    scaled_encoder = folded.producer_encoder @ scales
    scaled_target = folded.target_weight @ torch.linalg.inv(scales).T
    scaled_body = dict(folded.report)
    scaled_body.pop("report_sha256")
    scaled_body["producer_encoder_sha256"] = interface_module._tensor_sha256(
        scaled_encoder
    )
    scaled_body["target_weight_sha256"] = interface_module._tensor_sha256(scaled_target)
    scaled = TiedCausalInterfaceFold(
        target_weight=scaled_target,
        producer_encoder=scaled_encoder,
        report={
            **scaled_body,
            "report_sha256": interface_module._canonical_sha256(scaled_body),
        },
    )

    self_dual = whiten_tied_causal_interface(scaled)

    torch.testing.assert_close(
        self_dual.producer_encoder.T @ self_dual.producer_encoder,
        torch.eye(5, dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        self_dual.producer_encoder @ self_dual.target_weight.T,
        scaled.producer_encoder @ scaled.target_weight.T,
        rtol=1e-8,
        atol=1e-8,
    )
    assert self_dual.report["schema"] == TIED_CAUSAL_INTERFACE_SELF_DUAL_SCHEMA
    assert (
        self_dual.report["source_interface_report_sha256"]
        == (scaled.report["report_sha256"])
    )
    assert self_dual.report["source_to_vocabulary_operator_preserved"] is True
    assert self_dual.report["full_vocabulary_rows_preserved"] is True
    assert self_dual.report["one_global_residual_gauge_available"] is True
    assert self_dual.report["training_steps"] == 0
    assert self_dual.report["optimizer_steps"] == 0
    assert self_dual.report["backward_calls"] == 0
    assert self_dual.report["quantization_used"] is False
    assert self_dual.report["pruning_used"] is False


def test_self_dual_whitening_rejects_rank_deficient_encoder():
    table, fit, validation = _affine_low_rank_fixture()
    folded = fit_tied_causal_interface(
        table,
        fit,
        target_width=5,
        validation_producer_rows=validation,
    )
    encoder = folded.producer_encoder.clone()
    encoder[:, -1] = encoder[:, 0]
    body = dict(folded.report)
    body.pop("report_sha256")
    body["producer_encoder_sha256"] = interface_module._tensor_sha256(encoder)
    broken = TiedCausalInterfaceFold(
        target_weight=folded.target_weight,
        producer_encoder=encoder,
        report={**body, "report_sha256": interface_module._canonical_sha256(body)},
    )
    with pytest.raises(
        TiedCausalInterfaceFoldError,
        match="full-column-rank",
    ):
        whiten_tied_causal_interface(broken)


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
