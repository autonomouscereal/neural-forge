from __future__ import annotations

import copy
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_terminal_action_residual as terminal  # noqa: E402

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


def test_non_power_of_two_normalized_weights_do_not_escape_probability_bounds() -> None:
    weights = torch.ones(7, dtype=torch.float64)
    weights = weights / weights.sum()
    logits = torch.tensor(
        [[1.0, -1.0]] * 7,
        dtype=torch.float64,
    )
    accumulator = terminal._PairMetricAccumulator()

    terminal._accumulate_pair_metrics(
        accumulator,
        logits,
        logits,
        weights,
    )
    metrics = terminal._finish_pair_metrics(accumulator, 7)

    assert metrics.top1_agreement_fraction == 1.0
    assert metrics.reference_winner_positive_margin_fraction == 1.0


def test_cold_start_structural_coverage_allows_individual_zero_coefficients() -> None:
    covered = terminal._terminal_writer_structural_coverage(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            dtype=torch.bfloat16,
        )
    )
    missing_source = terminal._terminal_writer_structural_coverage(
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.bfloat16,
        )
    )

    assert covered["all_elements_finite"] is True
    assert covered["all_elements_nonzero"] is False
    assert covered["all_target_outputs_covered"] is True
    assert covered["all_source_features_covered"] is True
    assert covered["complete_dense_writer_coverage"] is True
    assert missing_source["all_source_features_covered"] is False
    assert missing_source["complete_dense_writer_coverage"] is False


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


def _zero_terminal_seed(seed):
    return Lfm2CpcfTerminalActionSeed(
        gate_weight=seed.gate_weight,
        up_weight=seed.up_weight,
        down_weight=torch.zeros_like(seed.down_weight),
        final_norm_weight=seed.final_norm_weight,
        producer_encoder=seed.producer_encoder,
        target_tied_weight=seed.target_tied_weight,
        source_tied_table=seed.source_tied_table,
        norm_epsilon=seed.norm_epsilon,
    )


def _evidence_for_zero_writer(evidence, seed):
    pre_mlp = evidence.pre_mlp_residual.to(torch.float64)
    target = (
        pre_mlp
        * seed.final_norm_weight.to(torch.float64)
        / (pre_mlp.square().mean(dim=1, keepdim=True) + seed.norm_epsilon).sqrt()
    )
    return replace(
        evidence,
        parent_final_norm_output=target.to(torch.bfloat16),
        authentication_sha256="",
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
    assert "exact_zero_cold_start" not in report

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


def test_exact_zero_writer_forces_fit_only_dense_plant_before_heldout_diagnostics(
    monkeypatch,
):
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)
    fit_reference = _compile(seed, evidence_banks)
    heldout_rejecting = {
        **evidence_banks,
        **{
            split: _evidence_for_zero_writer(evidence_banks[split], seed)
            for split in ("fit_check", "selection", "validation")
        },
    }
    evaluated_splits = []
    original_evaluate_split = terminal._evaluate_split

    def observe_split(prepared, *args, **kwargs):
        evaluated_splits.append(prepared.evidence.split)
        return original_evaluate_split(prepared, *args, **kwargs)

    monkeypatch.setattr(terminal, "_evaluate_split", observe_split)

    result = _compile(seed, heldout_rejecting)
    report = result.as_dict()
    coverage = report["selected_writer_structural_coverage"]
    prepared_fit = terminal._prepare_split(heldout_rejecting["fit"], seed)
    fitted_branch_output = (
        prepared_fit.features @ result.published_down_weight.to(torch.float64).T
    )

    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert result.accepted is True
    assert result.changed is True
    assert result.selected_candidate_id != "incumbent"
    assert result.selected_candidate_id == fit_reference.selected_candidate_id
    assert torch.equal(result.selected_down_weight, fit_reference.selected_down_weight)
    assert torch.equal(result.published_down_weight, result.selected_down_weight)
    assert not torch.equal(result.published_down_weight, result.incumbent_down_weight)
    assert report["exact_zero_cold_start"] is True
    assert report["incumbent_noop_publication_forbidden"] is True
    assert report["published_is_incumbent"] is False
    assert report["candidate_selection_evidence_scope"] == "fit-only"
    assert report["relative_to_zero_weight_delta_gate_applied"] is False
    assert report["selected_before_any_heldout_publication_diagnostic"] is True
    first_heldout = next(
        index for index, split in enumerate(evaluated_splits) if split != "fit"
    )
    assert set(evaluated_splits[:first_heldout]) == {"fit"}
    assert "fit" not in evaluated_splits[first_heldout:]
    assert evaluated_splits.count("fit_check") == 2
    assert evaluated_splits.count("selection") == 2
    assert evaluated_splits.count("validation") == 2
    assert report["heldout_candidate_evaluation_count_before_fit_freeze"] == 0
    assert report["heldout_candidate_evaluation_count_after_fit_freeze"] == 6
    assert report["heldout_candidate_count_evaluated_after_fit_freeze"] == 2
    assert report["validation"]["passed"] is False
    assert report["validation"]["heldout_diagnostic_only"] is True
    assert report["validation"]["publication_gate_applied"] is False
    assert coverage["all_elements_finite"] is True
    assert coverage["all_target_outputs_covered"] is True
    assert coverage["all_source_features_covered"] is True
    assert coverage["complete_dense_writer_coverage"] is True
    assert coverage["nonzero_element_count"] > 0
    assert torch.isfinite(fitted_branch_output).all().item() is True
    assert torch.count_nonzero(fitted_branch_output).item() > 0
    assert fitted_branch_output.ne(0).any(dim=0).all().item() is True
    assert (
        report["selected_metrics_before_validation"]["fit"]["oracle"][
            "centered_logit_relative_rmse"
        ]
        < report["incumbent_metrics"]["fit"]["oracle"]["centered_logit_relative_rmse"]
    )
    assert report["parameter_count_before"] == report["parameter_count_after"]
    assert report["parameter_identity_or_topology_mutation"] is False
    assert report["zero_training_pruning_quantization"] is True
    assert set(report["operation_audit"].values()) == {0}
    assert result.verify_authentication() is True
    selected_id = result.selected_candidate_id
    for candidate in report["frontier"]:
        if candidate["candidate_id"] in {"incumbent", selected_id}:
            continue
        assert candidate["fit_check_metrics"] is None
        assert candidate["selection_metrics"] is None
        assert candidate["fit_check_gates"] == {
            "not_evaluated_before_fit_selection_freeze": True
        }
        assert candidate["selection_gates"] == {
            "not_evaluated_before_fit_selection_freeze": True
        }


def test_exact_zero_result_rejects_self_sealed_caller_forgery() -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(_zero_terminal_seed(source_seed), evidence_banks)

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="compiler-issued",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report=result.as_dict(),
            result_sha256=result.result_sha256,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        lambda body: body.__setitem__("schema", "caller-forged"),
        lambda body: body.__setitem__("selected_nonincumbent", False),
        lambda body: body["operation_audit"].pop("training_steps"),
        lambda body: next(
            value
            for value in body["frontier"]
            if value["candidate_id"] == body["selected_candidate_id"]
        ).__setitem__("state_sha256", "0" * 64),
    ),
)
def test_exact_zero_result_rejects_resealed_internal_lineage_tamper(tamper) -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(_zero_terminal_seed(source_seed), evidence_banks)
    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    tamper(body)
    forged_sha256 = terminal._canonical_sha256(body)

    with pytest.raises(Lfm2CpcfTerminalActionAuthenticationError):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report={**body, "result_sha256": forged_sha256},
            result_sha256=forged_sha256,
            _compiler_input_binding=result._compiler_input_binding,
            _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
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
