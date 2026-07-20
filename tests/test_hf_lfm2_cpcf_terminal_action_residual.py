from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionAuthenticationError,
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionResidualError,
    Lfm2CpcfTerminalActionSeed,
    compile_lfm2_cpcf_terminal_action_residual,
    damped_rmsnorm_gauss_newton_target,
    rmsnorm_jvp,
)


def test_rmsnorm_woodbury_gauss_newton_matches_explicit_jacobian() -> None:
    generator = torch.Generator().manual_seed(7)
    rows = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    residual = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    vectors = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    gain = torch.randn(7, generator=generator, dtype=torch.float64)
    epsilon = 1e-5
    damping = 3e-4

    actual_target = damped_rmsnorm_gauss_newton_target(
        rows,
        residual,
        gain,
        epsilon=epsilon,
        damping=damping,
    )
    actual_jvp = rmsnorm_jvp(
        rows,
        vectors,
        gain,
        epsilon=epsilon,
    )
    explicit_targets = []
    explicit_jvps = []
    for row, output_residual, vector in zip(
        rows,
        residual,
        vectors,
        strict=True,
    ):
        scale = (row.square().mean() + epsilon).sqrt()
        jacobian = torch.diag(gain) / scale - torch.outer(gain * row, row) / (
            int(row.numel()) * scale.pow(3)
        )
        explicit_targets.append(
            torch.linalg.solve(
                jacobian.T @ jacobian
                + damping
                * torch.eye(
                    int(row.numel()),
                    dtype=torch.float64,
                ),
                jacobian.T @ output_residual,
            )
        )
        explicit_jvps.append(jacobian @ vector)

    assert torch.allclose(
        actual_target,
        torch.stack(explicit_targets),
        rtol=2e-10,
        atol=3e-12,
    )
    assert torch.allclose(
        actual_jvp,
        torch.stack(explicit_jvps),
        rtol=2e-13,
        atol=5e-15,
    )


def _synthetic_terminal_problem():
    generator = torch.Generator().manual_seed(123)
    hidden = 4
    intermediate = 6
    epsilon = 1e-5
    gate = (
        torch.randn(
            intermediate,
            hidden,
            generator=generator,
        )
        * 0.55
    ).to(torch.bfloat16)
    up = (
        torch.randn(
            intermediate,
            hidden,
            generator=generator,
        )
        * 0.55
    ).to(torch.bfloat16)
    incumbent = (
        torch.randn(
            hidden,
            intermediate,
            generator=generator,
        )
        * 0.12
    ).to(torch.bfloat16)
    target_delta = torch.zeros(
        hidden,
        intermediate,
        dtype=torch.float32,
    )
    target_delta[0] = torch.tensor([0.20, -0.16, 0.18, -0.14, 0.12, -0.10])
    target_delta[1] = torch.tensor([-0.09, 0.07, -0.08, 0.06, -0.05, 0.04])
    target = (incumbent.to(torch.float32) + target_delta).to(torch.bfloat16)
    tied_table = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    seed = Lfm2CpcfTerminalActionSeed(
        gate_weight=gate,
        up_weight=up,
        down_weight=incumbent,
        final_norm_weight=torch.ones(
            hidden,
            dtype=torch.bfloat16,
        ),
        producer_encoder=torch.eye(hidden, dtype=torch.float64),
        target_tied_weight=tied_table,
        source_tied_table=tied_table,
        norm_epsilon=epsilon,
    )

    def evidence(
        split,
        rows: int,
        random_seed: int,
        *,
        target_weight=target,
    ):
        split_generator = torch.Generator().manual_seed(random_seed)
        mlp_input = (
            torch.randn(
                rows,
                hidden,
                generator=split_generator,
            )
            * 0.75
        ).to(torch.bfloat16)
        pre_mlp = (
            torch.randn(
                rows,
                hidden,
                generator=split_generator,
            )
            * 0.30
        ).to(torch.float64)
        signs = torch.where(
            torch.arange(rows) % 2 == 0,
            1.0,
            -1.0,
        )
        pre_mlp[:, 0] = signs * 1.4
        features = torch.nn.functional.silu(
            mlp_input.to(torch.float32) @ gate.to(torch.float32).T
        ) * (mlp_input.to(torch.float32) @ up.to(torch.float32).T)
        target_post = (
            pre_mlp + features.to(torch.float64) @ target_weight.to(torch.float64).T
        )
        parent_final_norm = (
            target_post
            / (target_post.square().mean(dim=1, keepdim=True) + epsilon).sqrt()
        )
        return Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(f"{split}:{index}" for index in range(rows)),
            mlp_input=mlp_input,
            pre_mlp_residual=pre_mlp.to(torch.bfloat16),
            parent_final_norm_output=parent_final_norm.to(torch.bfloat16),
            sample_weights=torch.ones(rows, dtype=torch.float64),
        )

    evidence_banks = {
        "fit": evidence("fit", 32, 1),
        "fit_check": evidence("fit_check", 16, 2),
        "selection": evidence("selection", 16, 3),
        "validation": evidence("validation", 16, 4),
    }
    rejecting_validation = evidence(
        "validation",
        16,
        4,
        target_weight=incumbent,
    )
    return seed, evidence_banks, rejecting_validation


def _compile(seed, evidence_banks):
    return compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=evidence_banks["fit"],
        fit_check_evidence=evidence_banks["fit_check"],
        selection_evidence=evidence_banks["selection"],
        validation_evidence=evidence_banks["validation"],
    )


def test_exact_bfloat16_frontier_selects_and_validates_terminal_repair() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    original = seed.down_weight.clone()

    result = _compile(seed, evidence_banks)
    report = result.as_dict()

    assert result.accepted is True
    assert result.changed is True
    assert result.selected_candidate_id != "incumbent"
    assert result.verify_authentication() is True
    assert torch.equal(seed.down_weight, original)
    assert result.incumbent_down_weight.dtype == torch.bfloat16
    assert result.selected_down_weight.dtype == torch.bfloat16
    assert result.published_down_weight.dtype == torch.bfloat16
    assert result.incumbent_down_weight.data_ptr() != seed.down_weight.data_ptr()
    assert result.selected_down_weight.data_ptr() != seed.down_weight.data_ptr()
    assert result.published_down_weight.data_ptr() != seed.down_weight.data_ptr()
    assert result.selected_down_weight.requires_grad is False
    assert result.selected_down_weight.grad_fn is None
    assert result.published_down_weight.numel() == seed.down_weight.numel()

    assert report["frontier_count"] == 37
    assert report["frontier_count"] == report["expected_frontier_count"]
    assert all(
        candidate["exact_bfloat16_materialization"]
        and candidate["private_candidate_storage"]
        for candidate in report["frontier"]
    )
    assert report["split_row_id_sets_disjoint"] is True
    assert report["fit_weighted_ridge"]["fit_check_rows_used"] == 0
    assert report["fit_weighted_ridge"]["selection_rows_used"] == 0
    assert report["fit_weighted_ridge"]["validation_rows_used"] == 0
    assert report["validation"]["candidate_count"] == 1
    assert report["validation"]["used_for_candidate_ranking"] is False
    assert (
        report["validation"]["accessed_only_after_selected_id_and_state_sha256_frozen"]
        is True
    )
    assert (
        report["validation"]["evaluated_candidate_id"] == result.selected_candidate_id
    )
    assert report["validation"]["passed"] is True
    assert report["rollback_performed"] is False
    assert report["source_seed_writer_unchanged"] is True
    assert report["parameter_count_before"] == report["parameter_count_after"]
    assert set(report["operation_audit"].values()) == {0}

    for split in ("fit", "fit_check", "selection"):
        incumbent_rmse = report["incumbent_metrics"][split]["oracle"][
            "centered_logit_relative_rmse"
        ]
        selected_rmse = report["selected_metrics_before_validation"][split]["oracle"][
            "centered_logit_relative_rmse"
        ]
        assert selected_rmse < incumbent_rmse
    assert (
        report["validation"]["metrics"]["oracle"]["centered_logit_relative_rmse"]
        < report["incumbent_metrics"]["validation"]["oracle"][
            "centered_logit_relative_rmse"
        ]
    )


def test_validation_cannot_reselect_and_rejection_rolls_back_exactly() -> None:
    seed, evidence_banks, rejecting_validation = _synthetic_terminal_problem()
    accepted = _compile(seed, evidence_banks)
    rejected_banks = {
        **evidence_banks,
        "validation": rejecting_validation,
    }

    rejected = _compile(seed, rejected_banks)
    report = rejected.as_dict()

    assert accepted.selected_candidate_id != "incumbent"
    assert rejected.selected_candidate_id == accepted.selected_candidate_id
    assert torch.equal(
        rejected.selected_down_weight,
        accepted.selected_down_weight,
    )
    assert rejected.accepted is False
    assert rejected.changed is False
    assert report["validation"]["passed"] is False
    assert report["validation"]["candidate_count"] == 1
    assert report["validation"]["used_for_candidate_ranking"] is False
    assert report["validation"]["used_to_change_selected_candidate_id"] is False
    assert report["rollback_performed"] is True
    assert report["rollback_exact"] is True
    assert report["published_is_incumbent"] is True
    assert torch.equal(
        rejected.published_down_weight,
        rejected.incumbent_down_weight,
    )
    assert rejected.verify_authentication() is True


def test_split_overlap_fixed_frontier_and_post_authentication_tamper_fail() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    selection = evidence_banks["selection"]
    overlapping_ids = (
        evidence_banks["fit"].row_ids[0],
        *selection.row_ids[1:],
    )
    overlapping_selection = replace(
        selection,
        row_ids=overlapping_ids,
        authentication_sha256="",
    )

    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match="overlap",
    ):
        _compile(
            seed,
            {
                **evidence_banks,
                "selection": overlapping_selection,
            },
        )

    policy = Lfm2CpcfTerminalActionPolicy()
    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match="frontier is fixed",
    ):
        replace(
            policy,
            alpha_frontier=(0.0, 1.0),
            authentication_sha256="",
        )

    evidence_banks["fit"].mlp_input[0, 0] += 1.0
    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="changed after authentication",
    ):
        _compile(seed, evidence_banks)
