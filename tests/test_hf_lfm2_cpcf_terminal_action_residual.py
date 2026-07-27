from __future__ import annotations

import copy
import gc
import weakref
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_terminal_action_residual as terminal  # noqa: E402

from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionAuthenticationError,
    Lfm2CpcfTerminalActionDownWeightPrior,
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionResidualError,
    Lfm2CpcfTerminalActionSeed,
    compile_lfm2_cpcf_terminal_action_residual,
    damped_rmsnorm_gauss_newton_target,
    lfm2_cpcf_terminal_action_tensor_sha256,
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


def test_terminal_evidence_preserves_and_authenticates_exact_fp32_runtime_dtype() -> (
    None
):
    mlp_input = torch.tensor(
        [[0.25, -0.50, 0.75], [1.00, -1.25, 1.50]],
        dtype=torch.float32,
    )
    pre_mlp = torch.tensor(
        [[-0.10, 0.20, -0.30], [0.40, -0.50, 0.60]],
        dtype=torch.float32,
    )
    parent = torch.tensor(
        [[0.15, -0.25, 0.35, -0.45], [0.55, -0.65, 0.75, -0.85]],
        dtype=torch.float32,
    )

    evidence = Lfm2CpcfTerminalActionEvidence(
        split="fit",
        row_ids=("fit:0", "fit:1"),
        mlp_input=mlp_input,
        pre_mlp_residual=pre_mlp,
        parent_final_norm_output=parent,
        sample_weights=torch.ones(2, dtype=torch.float64),
    )
    body = evidence.authentication_body()

    assert evidence.verify_authentication() is True
    assert evidence.mlp_input.dtype == torch.float32
    assert evidence.pre_mlp_residual.dtype == torch.float32
    assert evidence.parent_final_norm_output.dtype == torch.float32
    assert torch.equal(evidence.mlp_input, mlp_input)
    assert torch.equal(evidence.pre_mlp_residual, pre_mlp)
    assert torch.equal(evidence.parent_final_norm_output, parent)
    assert evidence.mlp_input.data_ptr() != mlp_input.data_ptr()
    assert body["runtime_dtype"] == "float32"
    assert body["all_runtime_evidence_tensors_share_exact_dtype"] is True
    assert body["exact_requested_dtype_child_replay"] is True
    assert body["exact_requested_dtype_parent_final_norm_evidence"] is True
    assert body["exact_bfloat16_child_replay"] is False
    assert body["exact_bfloat16_parent_final_norm_evidence"] is False


@pytest.mark.parametrize(
    "mixed_field", ("pre_mlp_residual", "parent_final_norm_output")
)
def test_terminal_evidence_rejects_mixed_runtime_dtypes(mixed_field: str) -> None:
    values = {
        "mlp_input": torch.ones((2, 3), dtype=torch.float32),
        "pre_mlp_residual": torch.zeros((2, 3), dtype=torch.float32),
        "parent_final_norm_output": torch.ones((2, 4), dtype=torch.float32),
    }
    values[mixed_field] = values[mixed_field].to(torch.bfloat16)

    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match=r"exact torch\.float32",
    ):
        Lfm2CpcfTerminalActionEvidence(
            split="fit",
            row_ids=("fit:0", "fit:1"),
            **values,
            sample_weights=torch.ones(2, dtype=torch.float64),
        )


def test_terminal_evidence_preserves_legacy_bfloat16_receipt_contract() -> None:
    evidence = Lfm2CpcfTerminalActionEvidence(
        split="fit",
        row_ids=("fit:0",),
        mlp_input=torch.ones((1, 3), dtype=torch.bfloat16),
        pre_mlp_residual=torch.zeros((1, 3), dtype=torch.bfloat16),
        parent_final_norm_output=torch.ones((1, 4), dtype=torch.bfloat16),
        sample_weights=torch.ones(1, dtype=torch.float64),
    )
    body = evidence.authentication_body()

    assert evidence.verify_authentication() is True
    assert body["exact_bfloat16_child_replay"] is True
    assert body["exact_bfloat16_parent_final_norm_evidence"] is True
    assert "runtime_dtype" not in body
    assert "exact_requested_dtype_child_replay" not in body


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


def _compile(seed, evidence_banks, *, policy=None, down_weight_prior=None):
    return compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=evidence_banks["fit"],
        fit_check_evidence=evidence_banks["fit_check"],
        selection_evidence=evidence_banks["selection"],
        validation_evidence=evidence_banks["validation"],
        policy=policy,
        down_weight_prior=down_weight_prior,
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


def _qdeim_down_weight_prior(
    seed,
    down_weight,
    *,
    body_recipe_id="nonlinear_qdeim_structural",
    mlp_lane="nonlinear_qdeim_structural_priority",
):
    return Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=down_weight,
        body_recipe_id=body_recipe_id,
        body_recipe_sha256="1" * 64,
        mlp_lane=mlp_lane,
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256="2" * 64,
        preparation_sha256="3" * 64,
        mlp_installation_core_sha256="4" * 64,
        live_deferred_w2_sha256=lfm2_cpcf_terminal_action_tensor_sha256(
            seed.down_weight
        ),
    )


@pytest.mark.parametrize(
    ("body_recipe_id", "mlp_lane"),
    (
        (
            "nonlinear_qdeim_structural",
            "nonlinear_qdeim_structural_priority",
        ),
        (
            "nonlinear_qdeim_fit_corrected",
            "nonlinear_qdeim_fit_corrected_priority",
        ),
        (
            "nonlinear_qdeim_coupled_attention_structural",
            "nonlinear_qdeim_structural_priority",
        ),
        (
            "nonlinear_qdeim_coupled_attention_fit_corrected",
            "nonlinear_qdeim_fit_corrected_priority",
        ),
        (
            "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected",
            "nonlinear_qdeim_fit_corrected_priority",
        ),
        (
            "nonlinear_qdeim_coupled_attention_short_conv_residual_writer",
            "nonlinear_qdeim_structural_priority",
        ),
    ),
)
def test_qdeim_prior_accepts_exact_base_and_coupled_recipe_lane_pairs(
    body_recipe_id,
    mlp_lane,
) -> None:
    source_seed, _evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)

    prior = _qdeim_down_weight_prior(
        seed,
        source_seed.down_weight,
        body_recipe_id=body_recipe_id,
        mlp_lane=mlp_lane,
    )

    assert prior.body_recipe_id == body_recipe_id
    assert prior.mlp_lane == mlp_lane
    assert prior.verify_authentication() is True


def test_qdeim_prior_is_a_pure_frontier_state_and_center_for_generated_deltas(
    monkeypatch,
) -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)
    prior = _qdeim_down_weight_prior(seed, source_seed.down_weight)
    captured_deltas = []
    original_ridge_delta = terminal._ridge_delta

    def capture_ridge_delta(*args, **kwargs):
        delta = original_ridge_delta(*args, **kwargs)
        captured_deltas.append(delta.detach().clone())
        return delta

    monkeypatch.setattr(terminal, "_ridge_delta", capture_ridge_delta)

    result = _compile(
        seed,
        evidence_banks,
        down_weight_prior=prior,
    )
    report = result.as_dict()
    pure_prior_id = "qdeim-prior-alpha-00"
    pure_prior = result.frontier_down_weights[pure_prior_id]
    generated_records = [
        record
        for record in report["frontier"]
        if record["candidate_id"].startswith("gn-")
    ]

    assert prior.verify_authentication() is True
    assert torch.equal(pure_prior, prior.down_weight)
    assert pure_prior.data_ptr() != prior.down_weight.data_ptr()
    assert report["pure_prior_frontier_candidate_id"] == pure_prior_id
    assert report["authenticated_pure_prior_frontier_candidate_count"] == 1
    assert report["down_weight_prior_tensor_sha256"] == (
        lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
    )
    assert report["frontier_centered_on_authenticated_down_weight_prior"] is True
    assert report["live_seed_w2_retained_as_separate_incumbent_control"] is True
    assert report["down_weight_prior_installed_before_terminal"] is False
    assert report["down_weight_prior_installation_count_before_terminal"] == 0

    ridge_count = len(Lfm2CpcfTerminalActionPolicy().ridge_frontier)
    assert len(captured_deltas) == (
        len(Lfm2CpcfTerminalActionPolicy().gauss_newton_damping_frontier) * ridge_count
    )
    assert len(generated_records) == 36
    for record in generated_records:
        candidate_id = record["candidate_id"]
        _, damping_index, _, ridge_index, _, _ = candidate_id.split("-")
        delta = captured_deltas[int(damping_index) * ridge_count + int(ridge_index)]
        expected = (
            prior.down_weight.to(torch.float32)
            + record["alpha"] * delta.to(torch.float32)
        ).to(torch.bfloat16)
        candidate = result.frontier_down_weights[candidate_id]

        assert torch.equal(candidate, expected)
        assert record["candidate_centered_on_authenticated_down_weight_prior"] is True
        assert record["down_weight_prior_authentication_sha256"] == (
            prior.authentication_sha256
        )
        assert record["down_weight_prior_tensor_sha256"] == (
            lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
        )
        assert record["live_incumbent_remains_separate_control"] is True

    assert any(
        not torch.equal(
            result.frontier_down_weights[record["candidate_id"]], pure_prior
        )
        for record in generated_records
    )
    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert torch.count_nonzero(result.incumbent_down_weight).item() == 0
    assert torch.equal(result.incumbent_down_weight, seed.down_weight)
    assert torch.equal(result.selected_down_weight, seed.down_weight)
    assert torch.equal(result.published_down_weight, seed.down_weight)
    assert result.verify_authentication() is True


def test_qdeim_prior_adds_one_candidate_without_changing_legacy_frontier_ids() -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)
    prior = _qdeim_down_weight_prior(seed, source_seed.down_weight)

    legacy = _compile(seed, evidence_banks)
    with_prior = _compile(seed, evidence_banks, down_weight_prior=prior)
    legacy_ids = [record["candidate_id"] for record in legacy.report["frontier"]]
    prior_ids = [record["candidate_id"] for record in with_prior.report["frontier"]]
    expected_generated_ids = [
        f"gn-{damping:02d}-ridge-{ridge:02d}-alpha-{alpha:02d}"
        for damping in range(3)
        for ridge in range(3)
        for alpha in range(1, 5)
    ]

    assert legacy_ids == ["incumbent", *expected_generated_ids]
    assert legacy.report["frontier_count"] == 37
    assert legacy.report["expected_frontier_count"] == 37
    assert list(legacy.frontier_down_weights) == expected_generated_ids
    assert "down_weight_prior" not in legacy.report
    assert "pure_prior_frontier_candidate_id" not in legacy.report

    assert prior_ids == [
        "incumbent",
        "qdeim-prior-alpha-00",
        *expected_generated_ids,
    ]
    assert with_prior.report["frontier_count"] == 38
    assert with_prior.report["expected_frontier_count"] == 38
    assert prior_ids[2:] == legacy_ids[1:]
    assert list(with_prior.frontier_down_weights) == prior_ids[1:]
    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert legacy.verify_authentication() is True
    assert with_prior.verify_authentication() is True


@pytest.mark.parametrize(
    "tamper_kind",
    ("prior-tensor", "live-incumbent-lineage"),
)
def test_qdeim_prior_tamper_or_live_incumbent_mismatch_fails_closed(
    tamper_kind,
) -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)
    prior = _qdeim_down_weight_prior(seed, source_seed.down_weight)
    if tamper_kind == "prior-tensor":
        prior.down_weight.reshape(-1)[0] += torch.tensor(
            1.0,
            dtype=torch.bfloat16,
        )
        assert prior.verify_authentication() is False
    else:
        prior = replace(
            prior,
            live_deferred_w2_sha256="f" * 64,
            authentication_sha256="",
        )
        assert prior.verify_authentication() is True

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="prior|live deferred-W2 lineage",
    ):
        _compile(seed, evidence_banks, down_weight_prior=prior)


def test_qdeim_prior_rejects_cross_recipe_lane_mismatch() -> None:
    source_seed, _evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)
    prior = _qdeim_down_weight_prior(seed, source_seed.down_weight)

    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match="exact selected nonlinear Q-DEIM recipe/lane pair",
    ):
        replace(
            prior,
            mlp_lane="nonlinear_qdeim_fit_corrected_priority",
            authentication_sha256="",
        )


def test_exact_bfloat16_frontier_defers_selection_and_validates_diagnostic_lead() -> (
    None
):
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    original = seed.down_weight.clone()

    result = _compile(seed, evidence_banks)
    report = result.as_dict()

    assert result.accepted is False
    assert result.changed is False
    assert result.selected_candidate_id == "incumbent"
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
    assert len(result.frontier_down_weights) == 36
    nonincumbent_frontier = report["frontier"][1:]
    assert list(result.frontier_down_weights) == [
        candidate["candidate_id"] for candidate in nonincumbent_frontier
    ]
    assert (
        len({tensor.data_ptr() for tensor in result.frontier_down_weights.values()})
        == 36
    )
    for candidate in nonincumbent_frontier:
        tensor = result.frontier_down_weights[candidate["candidate_id"]]
        assert tensor.dtype == torch.bfloat16
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert torch.isfinite(tensor).all().item() is True
        assert terminal._tensor_sha256(tensor) == candidate["state_sha256"]
    diagnostic_lead_id = report["diagnostic_lead_candidate_id"]
    assert diagnostic_lead_id in result.frontier_down_weights
    assert torch.equal(result.selected_down_weight, result.incumbent_down_weight)
    assert torch.equal(result.published_down_weight, result.incumbent_down_weight)

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
    assert report["validation"]["used_for_diagnostic_ordering"] is False
    assert (
        report["validation"][
            "accessed_only_after_diagnostic_lead_id_and_state_sha256_frozen"
        ]
        is True
    )
    assert report["validation"]["evaluated_candidate_id"] == diagnostic_lead_id
    assert report["validation"]["passed"] is True
    assert report["rollback_performed"] is False
    assert report["source_seed_writer_unchanged"] is True
    assert report["parameter_count_before"] == report["parameter_count_after"]
    assert set(report["operation_audit"].values()) == {0}
    assert "exact_zero_cold_start" not in report
    assert report["final_selection_status"] == (
        "deferred-to-manual-whole-response-review"
    )
    assert report["final_selected_candidate_id"] is None
    assert report["final_selection_performed"] is False
    assert report["automatic_candidate_acceptance_performed"] is False
    assert report["candidate_published_before_manual_review"] is False
    assert report["manual_whole_response_review_required"] is True
    assert report["diagnostic_order_candidate_ids"][0] == diagnostic_lead_id
    assert report["diagnostic_order_candidate_count"] == len(
        report["diagnostic_order_candidate_ids"]
    )
    assert report["diagnostic_order_has_final_selection_authority"] is False
    assert report["diagnostic_order_is_provisional"] is True

    for split in ("fit", "fit_check", "selection"):
        incumbent_rmse = report["incumbent_metrics"][split]["oracle"][
            "centered_logit_relative_rmse"
        ]
        diagnostic_lead_rmse = report["diagnostic_lead_metrics"][split]["oracle"][
            "centered_logit_relative_rmse"
        ]
        assert diagnostic_lead_rmse < incumbent_rmse
    assert (
        report["validation"]["metrics"]["oracle"]["centered_logit_relative_rmse"]
        < report["incumbent_metrics"]["validation"]["oracle"][
            "centered_logit_relative_rmse"
        ]
    )


def test_exact_zero_writer_uses_structural_eligibility_for_diagnostic_ordering(
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
    diagnostic_lead_id = report["diagnostic_lead_candidate_id"]
    diagnostic_lead_weight = result.frontier_down_weights[diagnostic_lead_id]
    coverage = report["diagnostic_lead_writer_structural_coverage"]
    prepared_fit = terminal._prepare_split(heldout_rejecting["fit"], seed)
    fitted_branch_output = (
        prepared_fit.features @ diagnostic_lead_weight.to(torch.float64).T
    )

    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert result.accepted is False
    assert result.changed is False
    assert result.selected_candidate_id == "incumbent"
    assert diagnostic_lead_id == fit_reference.report["diagnostic_lead_candidate_id"]
    assert torch.equal(
        diagnostic_lead_weight,
        fit_reference.frontier_down_weights[diagnostic_lead_id],
    )
    assert torch.equal(result.selected_down_weight, result.incumbent_down_weight)
    assert torch.equal(result.published_down_weight, result.incumbent_down_weight)
    assert report["exact_zero_cold_start"] is True
    assert report["incumbent_is_noncandidate_control"] is True
    assert report["candidate_publication_deferred_until_manual_review"] is True
    assert report["published_is_incumbent"] is True
    assert report["candidate_diagnostic_ordering_evidence_scope"] == (
        "finite-dense-structural-eligibility-plus-fit-ordering"
    )
    assert report["candidate_eligibility_basis"] == (
        "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
    )
    assert report["fit_metrics_role"] == (
        "ordering-among-structurally-eligible-not-threshold-gating"
    )
    assert report["fit_metrics_used_for_diagnostic_ordering"] is True
    assert report["fit_metrics_used_for_pre_review_candidate_selection"] is False
    assert report["fit_metrics_used_for_structural_eligibility"] is False
    assert report["fit_metrics_have_final_intelligence_authority"] is False
    assert report["fit_metric_thresholds_used_for_candidate_eligibility"] is False
    assert report["fit_metric_thresholds_used_for_publication"] is False
    assert report["relative_to_zero_weight_delta_gate_applied"] is False
    assert report["diagnostic_lead_frozen_before_any_heldout_diagnostic"] is True
    first_heldout = next(
        index for index, split in enumerate(evaluated_splits) if split != "fit"
    )
    assert set(evaluated_splits[:first_heldout]) == {"fit"}
    assert "fit" not in evaluated_splits[first_heldout:]
    assert evaluated_splits.count("fit_check") == 2
    assert evaluated_splits.count("selection") == 2
    assert evaluated_splits.count("validation") == 2
    assert report["heldout_candidate_evaluation_count_before_lead_freeze"] == 0
    assert report["heldout_candidate_evaluation_count_after_lead_freeze"] == 6
    assert report["heldout_candidate_count_evaluated_after_lead_freeze"] == 2
    assert report["validation"]["passed"] is False
    assert report["validation"]["heldout_diagnostic_only"] is True
    assert report["validation"]["publication_gate_applied"] is False
    assert coverage["all_elements_finite"] is True
    assert coverage["all_target_outputs_covered"] is True
    assert coverage["all_source_features_covered"] is True
    assert coverage["complete_dense_writer_coverage"] is True
    assert coverage["complete_dense_writer_structural_safety"] is True
    assert coverage["owns_exact_storage_allocation"] is True
    assert coverage["storage_offset"] == 0
    assert coverage["storage_bytes"] == coverage["expected_storage_bytes"]
    assert coverage["nonzero_element_count"] > 0
    assert torch.isfinite(fitted_branch_output).all().item() is True
    assert torch.count_nonzero(fitted_branch_output).item() > 0
    assert fitted_branch_output.ne(0).any(dim=0).all().item() is True
    assert (
        report["diagnostic_lead_metrics"]["fit"]["oracle"][
            "centered_logit_relative_rmse"
        ]
        < report["incumbent_metrics"]["fit"]["oracle"]["centered_logit_relative_rmse"]
    )
    assert report["parameter_count_before"] == report["parameter_count_after"]
    assert report["parameter_identity_or_topology_mutation"] is False
    assert report["zero_training_pruning_quantization"] is True
    assert set(report["operation_audit"].values()) == {0}
    assert result.verify_authentication() is True
    for candidate in report["frontier"]:
        if candidate["candidate_id"] in {"incumbent", diagnostic_lead_id}:
            continue
        assert candidate["fit_check_metrics"] is None
        assert candidate["selection_metrics"] is None
        assert candidate["fit_check_gates"] == {
            "not_evaluated_before_diagnostic_lead_freeze": True
        }
        assert candidate["selection_gates"] == {
            "not_evaluated_before_diagnostic_lead_freeze": True
        }


def test_exact_zero_dense_candidate_remains_reviewable_despite_fit_proxy_gate_miss() -> (
    None
):
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)
    policy = Lfm2CpcfTerminalActionPolicy(
        maximum_post_layer_rms_ratio=1e-12,
    )

    result = _compile(seed, evidence_banks, policy=policy)
    report = result.as_dict()
    diagnostic_lead = next(
        value
        for value in report["frontier"]
        if value["candidate_id"] == report["diagnostic_lead_candidate_id"]
    )

    assert result.accepted is False
    assert result.changed is False
    assert result.selected_candidate_id == "incumbent"
    assert diagnostic_lead["structurally_eligible"] is True
    assert diagnostic_lead["eligible"] is True
    assert diagnostic_lead["fit_metric_gates_passed"] is False
    assert diagnostic_lead["fit_gates"]["post_layer_norm"] is False
    assert diagnostic_lead["fit_metrics_used_for_diagnostic_ordering"] is True
    assert (
        diagnostic_lead["fit_metrics_used_for_pre_review_candidate_selection"] is False
    )
    assert diagnostic_lead["fit_metrics_used_for_structural_eligibility"] is False
    assert diagnostic_lead["fit_metric_thresholds_used_for_publication"] is False
    assert diagnostic_lead["fit_metrics_have_final_intelligence_authority"] is False
    assert report["structurally_eligible_fit_metric_gate_miss_count"] > 0
    assert report["fit_metric_gate_miss_can_exclude_structural_candidate"] is False
    assert report["published_writer_is_finite_dense_nonzero_structural_plant"] is False
    assert result.verify_authentication() is True


def test_nonzero_dense_candidate_remains_reviewable_despite_numeric_diagnostics() -> (
    None
):
    seed, evidence_banks, rejecting_validation = _synthetic_terminal_problem()
    policy = Lfm2CpcfTerminalActionPolicy(
        minimum_selection_centered_rmse_improvement_fraction=1.0,
        minimum_selection_oracle_top1_gain=1.0,
        minimum_selection_parent_top1_gain=1.0,
        minimum_selection_parent_positive_margin_gain=1.0,
        metric_regression_tolerance=0.0,
        maximum_post_layer_rms_ratio=1e-12,
        maximum_weight_delta_rms_ratio=1e-12,
    )
    result = _compile(
        seed,
        {**evidence_banks, "validation": rejecting_validation},
        policy=policy,
    )
    report = result.as_dict()
    diagnostic_lead = next(
        value
        for value in report["frontier"]
        if value["candidate_id"] == report["diagnostic_lead_candidate_id"]
    )

    assert result.selected_candidate_id == "incumbent"
    assert result.accepted is False
    assert result.changed is False
    assert diagnostic_lead["eligible"] is True
    assert diagnostic_lead["structurally_eligible"] is True
    assert diagnostic_lead["numeric_diagnostic_gate_miss"] is True
    assert diagnostic_lead["legacy_weight_delta_threshold_would_pass"] is False
    assert diagnostic_lead["fit_thresholds_used_for_eligibility"] is False
    assert diagnostic_lead["fit_check_thresholds_used_for_eligibility"] is False
    assert diagnostic_lead["selection_thresholds_used_for_eligibility"] is False
    assert diagnostic_lead["numeric_thresholds_used_for_publication"] is False
    assert report["numeric_thresholds_used_for_candidate_eligibility"] is False
    assert report["numeric_thresholds_used_for_publication"] is False
    assert report["structural_safety_is_only_candidate_veto"] is True
    assert report["validation"]["passed"] is False
    assert report["validation"]["heldout_diagnostic_only"] is True
    assert report["validation"]["publication_gate_applied"] is False
    assert report["rollback_performed"] is False
    assert report["published_is_incumbent"] is True
    assert torch.equal(result.published_down_weight, result.incumbent_down_weight)
    assert result.verify_authentication() is True


@pytest.mark.parametrize("exact_zero", (False, True), ids=("nonzero", "exact-zero"))
def test_proxy_metrics_only_create_a_provisional_review_order(exact_zero) -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    if exact_zero:
        seed = _zero_terminal_seed(seed)

    result = _compile(seed, evidence_banks)
    report = result.as_dict()
    eligible = [
        record
        for record in report["frontier"][1:]
        if record["structurally_eligible"] is True
    ]
    expected_order = [
        record["candidate_id"]
        for record in sorted(
            eligible,
            key=lambda record: (
                tuple(record["diagnostic_ordering_key"]),
                record["candidate_id"],
            ),
        )
    ]

    assert len(result.frontier_down_weights) == 36
    assert report["diagnostic_order_candidate_ids"] == expected_order
    assert report["diagnostic_lead_candidate_id"] == expected_order[0]
    assert report["diagnostic_order_is_provisional"] is True
    assert report["diagnostic_order_has_final_selection_authority"] is False
    assert report["diagnostic_lead_has_final_selection_authority"] is False
    assert report["final_selected_candidate_id"] is None
    assert report["final_selection_performed"] is False
    assert report["automatic_candidate_selection_performed"] is False
    assert report["automatic_candidate_acceptance_performed"] is False
    assert report["candidate_published_before_manual_review"] is False
    assert result.selected_candidate_id == "incumbent"
    assert result.accepted is False
    assert result.changed is False
    assert torch.equal(result.selected_down_weight, result.incumbent_down_weight)
    assert torch.equal(result.published_down_weight, result.incumbent_down_weight)
    assert result.verify_authentication() is True


def test_exact_zero_incomplete_frontier_fails_closed(monkeypatch) -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)

    def incomplete_delta(*_args, **_kwargs):
        value = torch.zeros_like(seed.down_weight, dtype=torch.float64)
        value[0, 0] = 1.0
        return value

    monkeypatch.setattr(terminal, "_ridge_delta", incomplete_delta)

    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match="no finite, fully dense effectful",
    ):
        _compile(seed, evidence_banks)


def test_exact_zero_nonfinite_frontier_and_malformed_writer_fail_closed(
    monkeypatch,
) -> None:
    source_seed, evidence_banks, _ = _synthetic_terminal_problem()
    seed = _zero_terminal_seed(source_seed)

    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match="materialized dense BF16 matrix",
    ):
        terminal._terminal_writer_structural_coverage(
            torch.ones_like(seed.down_weight, dtype=torch.float32)
        )

    def nonfinite_delta(*_args, **_kwargs):
        return torch.full_like(seed.down_weight, float("nan"), dtype=torch.float64)

    monkeypatch.setattr(terminal, "_ridge_delta", nonfinite_delta)

    with pytest.raises(
        Lfm2CpcfTerminalActionResidualError,
        match="nonfinite",
    ):
        _compile(seed, evidence_banks)


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
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report=result.as_dict(),
            result_sha256=result.result_sha256,
        )


def test_nonzero_result_rejects_self_sealed_caller_forgery() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="compiler-issued",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report=result.as_dict(),
            result_sha256=result.result_sha256,
        )


@pytest.mark.parametrize(
    "variant",
    (
        "method_relabel_schema_omitted",
        "method_omitted_schema_altered",
        "lineage_forged_with_labels_removed",
        "accepted_without_report_or_tensor_frontier",
        "unaccepted_but_frontier_bearing",
    ),
)
def test_nonzero_result_cannot_downgrade_compiler_authentication_by_resealing(
    variant,
) -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    frontier_down_weights = result.frontier_down_weights
    accepted = result.accepted

    if variant == "method_relabel_schema_omitted":
        body["method"] = "caller-relabeled-legacy-result"
        body.pop("structural_nonveto_schema")
    elif variant == "method_omitted_schema_altered":
        body.pop("method")
        body["structural_nonveto_schema"] = "caller-forged-schema"
    elif variant == "lineage_forged_with_labels_removed":
        body.pop("method")
        body["structural_nonveto_schema"] = "caller-forged-schema"
        body["seed_sha256"] = "1" * 64
        body["split_evidence_sha256"] = {
            split: "2" * 64 for split in body["split_evidence_sha256"]
        }
        body["split_row_ids_sha256"] = {
            split: "3" * 64 for split in body["split_row_ids_sha256"]
        }
    elif variant == "accepted_without_report_or_tensor_frontier":
        body["method"] = "caller-relabeled-legacy-result"
        body.pop("structural_nonveto_schema")
        body.pop("frontier")
        body["accepted"] = True
        frontier_down_weights = None
        accepted = True
    else:
        body["method"] = "caller-relabeled-legacy-result"
        body.pop("structural_nonveto_schema")
        body["accepted"] = False
        accepted = False

    forged_sha256 = terminal._canonical_sha256(body)
    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="accepted or frontier-bearing terminal result must be compiler-issued",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            frontier_down_weights=frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=accepted,
            report={**body, "result_sha256": forged_sha256},
            result_sha256=forged_sha256,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        lambda body: body.__setitem__("method", "caller-relabeled-legacy-result"),
        lambda body: body.pop("method"),
        lambda body: body.pop("structural_nonveto_schema"),
        lambda body: body.__setitem__(
            "structural_nonveto_schema", "caller-forged-schema"
        ),
    ),
)
def test_verify_authentication_rejects_resealed_compiler_contract_downgrade(
    tamper,
) -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    tamper(body)
    forged_sha256 = terminal._canonical_sha256(body)
    object.__setattr__(
        result,
        "report",
        {**body, "result_sha256": forged_sha256},
    )
    object.__setattr__(result, "result_sha256", forged_sha256)

    assert result.verify_authentication() is False


@pytest.fixture(scope="module", params=(False, True), ids=("nonzero", "exact-zero"))
def compiler_issued_structural_nonveto_result(request):
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    if request.param:
        seed = _zero_terminal_seed(seed)
    return _compile(seed, evidence_banks)


def _forge_compiler_lineage(result):
    binding_body = copy.deepcopy(dict(result._compiler_input_binding))
    binding_body.pop("binding_sha256")
    binding_body["seed_sha256"] = "1" * 64
    binding_body["split_evidence_sha256"] = {
        split: "2" * 64 for split in binding_body["split_evidence_sha256"]
    }
    binding_body["split_row_ids_sha256"] = {
        split: "3" * 64 for split in binding_body["split_row_ids_sha256"]
    }
    forged_binding = {
        **binding_body,
        "binding_sha256": terminal._canonical_sha256(binding_body),
    }

    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    body["seed_sha256"] = binding_body["seed_sha256"]
    body["split_evidence_sha256"] = binding_body["split_evidence_sha256"]
    body["split_row_ids_sha256"] = binding_body["split_row_ids_sha256"]
    forged_facts = terminal._terminal_compiler_lineage_facts(
        forged_binding,
        context="forged_test_binding",
    )
    body["compiler_lineage_sha256"] = terminal._terminal_compiler_lineage_sha256(
        forged_facts
    )
    forged_result_sha256 = terminal._canonical_sha256(body)
    forged_report = {**body, "result_sha256": forged_result_sha256}
    return forged_binding, forged_report, forged_result_sha256


def test_compiler_lineage_proof_is_identity_only_and_not_constructible(
    compiler_issued_structural_nonveto_result,
) -> None:
    proof = compiler_issued_structural_nonveto_result._compiler_lineage_token

    assert type(proof) is terminal._TerminalCompilerLineageProof
    assert type(proof).__slots__ == ("__weakref__",)
    assert not hasattr(proof, "__dict__")
    assert not hasattr(proof, "facts")
    assert not hasattr(proof, "lineage_sha256")
    with pytest.raises(Lfm2CpcfTerminalActionAuthenticationError):
        terminal._TerminalCompilerLineageProof()
    with pytest.raises(AttributeError):
        proof.caller_data = "0" * 64
    with pytest.raises(AttributeError):
        object.__setattr__(proof, "caller_data", "0" * 64)


def test_compiler_lineage_registry_releases_unreachable_proofs() -> None:
    gc.collect()
    before = terminal._terminal_compiler_lineage_registry_size()
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    proof_reference = weakref.ref(result._compiler_lineage_token)

    assert proof_reference() is result._compiler_lineage_token
    assert terminal._terminal_compiler_lineage_registry_size() == before + 1

    del result
    gc.collect()

    assert proof_reference() is None
    assert terminal._terminal_compiler_lineage_registry_size() == before


def _reconstruct_with_lineage_proof(result, proof):
    return terminal.Lfm2CpcfTerminalActionResidualResult(
        policy=result.policy,
        incumbent_down_weight=result.incumbent_down_weight,
        selected_down_weight=result.selected_down_weight,
        published_down_weight=result.published_down_weight,
        frontier_down_weights=result.frontier_down_weights,
        selected_candidate_id=result.selected_candidate_id,
        accepted=result.accepted,
        report=result.as_dict(),
        result_sha256=result.result_sha256,
        _compiler_input_binding=result._compiler_input_binding,
        _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
        _compiler_lineage_receipt=proof,
    )


@pytest.mark.parametrize("copy_kind", ("new", "copy", "deepcopy"))
def test_unregistered_or_copied_lineage_proof_is_rejected(
    compiler_issued_structural_nonveto_result,
    copy_kind,
) -> None:
    result = compiler_issued_structural_nonveto_result
    original = result._compiler_lineage_token
    if copy_kind == "new":
        forged = object.__new__(terminal._TerminalCompilerLineageProof)
    elif copy_kind == "copy":
        forged = copy.copy(original)
    else:
        forged = copy.deepcopy(original)

    assert forged is not original
    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="unregistered compiler-lineage proof",
    ):
        _reconstruct_with_lineage_proof(result, forged)

    probe = copy.copy(result)
    object.__setattr__(probe, "_compiler_lineage_token", forged)
    assert probe.verify_authentication() is False


def test_registered_lineage_proof_substitution_is_rejected(
    compiler_issued_structural_nonveto_result,
) -> None:
    result = compiler_issued_structural_nonveto_result
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    if int(torch.count_nonzero(result.incumbent_down_weight).item()) == 0:
        alternate = _compile(seed, evidence_banks)
    else:
        alternate = _compile(_zero_terminal_seed(seed), evidence_banks)
    substituted = alternate._compiler_lineage_token

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="diverges from compiler-issued lineage",
    ):
        _reconstruct_with_lineage_proof(result, substituted)

    probe = copy.copy(result)
    object.__setattr__(probe, "_compiler_lineage_token", substituted)
    assert probe.verify_authentication() is False


@pytest.mark.parametrize("replacement", ("metrics", "gates_and_passed"))
def test_compiler_lineage_rejects_coordinated_validation_report_replacement(
    compiler_issued_structural_nonveto_result,
    replacement,
) -> None:
    result = compiler_issued_structural_nonveto_result
    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    if replacement == "metrics":
        body["validation"]["metrics"] = copy.deepcopy(
            body["incumbent_metrics"]["validation"]
        )
    else:
        body["validation"]["gates"] = {
            name: False for name in body["validation"]["gates"]
        }
        body["validation"]["passed"] = False
    forged_sha256 = terminal._canonical_sha256(body)

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="diverges from compiler-issued lineage",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report={**body, "result_sha256": forged_sha256},
            result_sha256=forged_sha256,
            _compiler_input_binding=result._compiler_input_binding,
            _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
            _compiler_lineage_receipt=result._compiler_lineage_token,
        )

    probe = copy.copy(result)
    object.__setattr__(probe, "report", {**body, "result_sha256": forged_sha256})
    object.__setattr__(probe, "result_sha256", forged_sha256)
    assert probe.verify_authentication() is False


def test_forged_resealed_binding_and_report_cannot_replace_compiler_lineage(
    compiler_issued_structural_nonveto_result,
) -> None:
    result = compiler_issued_structural_nonveto_result
    forged_binding, forged_report, forged_sha256 = _forge_compiler_lineage(result)

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="diverges from compiler-issued lineage",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report=forged_report,
            result_sha256=forged_sha256,
            _compiler_input_binding=forged_binding,
            _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
            _compiler_lineage_receipt=result._compiler_lineage_token,
        )

    probe = copy.copy(result)
    object.__setattr__(probe, "_compiler_input_binding", forged_binding)
    object.__setattr__(probe, "report", forged_report)
    object.__setattr__(probe, "result_sha256", forged_sha256)
    assert probe.verify_authentication() is False


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("used_for_fit", True),
        ("used_for_fit_check_veto", True),
        ("used_for_diagnostic_ordering", True),
        ("used_for_final_selection", True),
        ("used_for_candidate_acceptance", True),
        ("heldout_diagnostic_only", False),
        ("publication_gate_applied", True),
        ("numeric_thresholds_diagnostic_only", False),
        (
            "accessed_only_after_diagnostic_lead_id_and_state_sha256_frozen",
            False,
        ),
        ("frozen_diagnostic_lead_state_sha256", "0" * 64),
        ("__omit_used_for_fit", None),
        ("__extra_role", True),
    ),
)
def test_validation_diagnostic_schema_rejects_resealed_role_tamper(
    compiler_issued_structural_nonveto_result,
    field,
    forged_value,
) -> None:
    result = compiler_issued_structural_nonveto_result
    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    if field == "__omit_used_for_fit":
        body["validation"].pop("used_for_fit")
    elif field == "__extra_role":
        body["validation"]["caller_forged_role"] = forged_value
    else:
        body["validation"][field] = forged_value
    forged_sha256 = terminal._canonical_sha256(body)
    forged_report = {**body, "result_sha256": forged_sha256}

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="validation report|compiler-issued lineage",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report=forged_report,
            result_sha256=forged_sha256,
            _compiler_input_binding=result._compiler_input_binding,
            _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
            _compiler_lineage_receipt=result._compiler_lineage_token,
        )

    probe = copy.copy(result)
    object.__setattr__(probe, "report", forged_report)
    object.__setattr__(probe, "result_sha256", forged_sha256)
    assert probe.verify_authentication() is False


@pytest.mark.parametrize("tamper_kind", ("missing", "extra", "bytes", "nonfinite"))
def test_terminal_frontier_writer_bank_fails_closed(tamper_kind) -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    bank = {
        candidate_id: value.detach().clone()
        for candidate_id, value in result.frontier_down_weights.items()
    }
    first_id = next(iter(bank))
    if tamper_kind == "missing":
        bank.pop(first_id)
    elif tamper_kind == "extra":
        bank["unexpected"] = bank[first_id].clone()
    elif tamper_kind == "bytes":
        bank[first_id].reshape(-1)[0] += torch.tensor(
            1.0,
            dtype=torch.bfloat16,
        )
    else:
        bank[first_id].reshape(-1)[0] = float("nan")

    with pytest.raises(Lfm2CpcfTerminalActionResidualError):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=result.selected_down_weight,
            published_down_weight=result.published_down_weight,
            frontier_down_weights=bank,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report=result.as_dict(),
            result_sha256=result.result_sha256,
        )


def test_terminal_frontier_writer_bank_is_privately_owned_and_reauthenticated() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    compiled = _compile(seed, evidence_banks)
    supplied = {
        candidate_id: value.detach().clone()
        for candidate_id, value in compiled.frontier_down_weights.items()
    }
    result = terminal.Lfm2CpcfTerminalActionResidualResult(
        policy=compiled.policy,
        incumbent_down_weight=compiled.incumbent_down_weight,
        selected_down_weight=compiled.selected_down_weight,
        published_down_weight=compiled.published_down_weight,
        frontier_down_weights=supplied,
        selected_candidate_id=compiled.selected_candidate_id,
        accepted=compiled.accepted,
        report=compiled.as_dict(),
        result_sha256=compiled.result_sha256,
        _compiler_input_binding=compiled._compiler_input_binding,
        _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
        _compiler_lineage_receipt=compiled._compiler_lineage_token,
    )
    first_id = next(iter(supplied))
    assert (
        result.frontier_down_weights[first_id].data_ptr()
        != supplied[first_id].data_ptr()
    )
    supplied[first_id].zero_()
    assert result.verify_authentication() is True

    result.frontier_down_weights[first_id].reshape(-1)[0] += torch.tensor(
        1.0,
        dtype=torch.bfloat16,
    )
    assert result.verify_authentication() is False


def test_terminal_frontier_rejects_resealed_metric_lead_as_final_selection() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    assert result.selected_candidate_id == "incumbent"
    diagnostic_lead_id = result.report["diagnostic_lead_candidate_id"]
    diagnostic_lead_weight = result.frontier_down_weights[diagnostic_lead_id]
    body = copy.deepcopy(result.as_dict())
    body.pop("result_sha256")
    body["selected_candidate_id"] = diagnostic_lead_id
    body["selected_nonincumbent"] = True
    body["accepted"] = True
    body["published_is_incumbent"] = False
    body["final_selection_performed"] = True
    body["automatic_candidate_selection_performed"] = True
    body["automatic_candidate_acceptance_performed"] = True
    body["candidate_published_before_manual_review"] = True
    body["output_state_sha256"]["selected"] = body["diagnostic_lead_state_sha256"]
    body["output_state_sha256"]["published"] = body["diagnostic_lead_state_sha256"]
    forged_sha256 = terminal._canonical_sha256(body)

    with pytest.raises(
        Lfm2CpcfTerminalActionAuthenticationError,
        match="compiler-issued|decision semantics|selected frontier record",
    ):
        terminal.Lfm2CpcfTerminalActionResidualResult(
            policy=result.policy,
            incumbent_down_weight=result.incumbent_down_weight,
            selected_down_weight=diagnostic_lead_weight,
            published_down_weight=diagnostic_lead_weight,
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=diagnostic_lead_id,
            accepted=True,
            report={**body, "result_sha256": forged_sha256},
            result_sha256=forged_sha256,
        )


def test_terminal_frontier_rejects_distinct_views_into_shared_storage() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    candidate_ids = tuple(result.frontier_down_weights)[:2]
    shape = tuple(result.incumbent_down_weight.shape)
    shared = torch.empty((2, *shape), dtype=torch.bfloat16)
    shared[0].copy_(result.frontier_down_weights[candidate_ids[0]])
    shared[1].copy_(result.frontier_down_weights[candidate_ids[1]])
    assert shared[0].data_ptr() != shared[1].data_ptr()
    assert (
        shared[0].untyped_storage().data_ptr() == shared[1].untyped_storage().data_ptr()
    )
    forged_bank = dict(result.frontier_down_weights)
    forged_bank[candidate_ids[0]] = shared[0]
    forged_bank[candidate_ids[1]] = shared[1]
    object.__setattr__(result, "frontier_down_weights", forged_bank)

    assert result.verify_authentication() is False


def test_terminal_frontier_rejects_bank_storage_alias_to_main_selected_tensor() -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
    assert result.selected_candidate_id == "incumbent"
    diagnostic_lead_id = result.report["diagnostic_lead_candidate_id"]
    forged_bank = dict(result.frontier_down_weights)
    forged_bank[diagnostic_lead_id] = result.selected_down_weight
    object.__setattr__(result, "frontier_down_weights", forged_bank)

    assert forged_bank[diagnostic_lead_id].data_ptr() == (
        result.selected_down_weight.data_ptr()
    )
    assert result.verify_authentication() is False


@pytest.mark.parametrize(
    "tamper",
    (
        lambda body: body.__setitem__("schema", "caller-forged"),
        lambda body: body.__setitem__("selected_nonincumbent", True),
        lambda body: body["operation_audit"].pop("training_steps"),
        lambda body: next(
            value
            for value in body["frontier"]
            if value["candidate_id"] == body["diagnostic_lead_candidate_id"]
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
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report={**body, "result_sha256": forged_sha256},
            result_sha256=forged_sha256,
            _compiler_input_binding=result._compiler_input_binding,
            _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
            _compiler_lineage_receipt=result._compiler_lineage_token,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        lambda body: body.pop("structural_nonveto_schema"),
        lambda body: body.__setitem__("structural_nonveto_schema", "caller-forged"),
        lambda body: next(
            value
            for value in body["frontier"]
            if value["candidate_id"] == body["diagnostic_lead_candidate_id"]
        ).__setitem__("eligible", False),
        lambda body: next(
            value
            for value in body["frontier"]
            if value["candidate_id"] == body["diagnostic_lead_candidate_id"]
        ).__setitem__("numeric_thresholds_used_for_publication", True),
        lambda body: body["validation"].__setitem__("publication_gate_applied", True),
        lambda body: body["diagnostic_lead_writer_structural_coverage"].__setitem__(
            "state_sha256", "0" * 64
        ),
    ),
)
def test_nonzero_result_rejects_resealed_structural_nonveto_tamper(tamper) -> None:
    seed, evidence_banks, _ = _synthetic_terminal_problem()
    result = _compile(seed, evidence_banks)
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
            frontier_down_weights=result.frontier_down_weights,
            selected_candidate_id=result.selected_candidate_id,
            accepted=result.accepted,
            report={**body, "result_sha256": forged_sha256},
            result_sha256=forged_sha256,
            _compiler_input_binding=result._compiler_input_binding,
            _compiler_issuer=terminal._TERMINAL_RESULT_COMPILER_ISSUER,
            _compiler_lineage_receipt=result._compiler_lineage_token,
        )


def test_validation_cannot_make_a_final_selection_or_publication_decision() -> None:
    seed, evidence_banks, rejecting_validation = _synthetic_terminal_problem()
    reference = _compile(seed, evidence_banks)
    rejected_banks = {
        **evidence_banks,
        "validation": rejecting_validation,
    }

    rejected = _compile(seed, rejected_banks)
    report = rejected.as_dict()

    assert reference.selected_candidate_id == "incumbent"
    assert rejected.selected_candidate_id == "incumbent"
    assert (
        rejected.report["diagnostic_lead_candidate_id"]
        == (reference.report["diagnostic_lead_candidate_id"])
    )
    assert torch.equal(
        rejected.selected_down_weight,
        reference.selected_down_weight,
    )
    assert rejected.accepted is False
    assert rejected.changed is False
    assert report["validation"]["passed"] is False
    assert report["validation"]["candidate_count"] == 1
    assert report["validation"]["used_for_diagnostic_ordering"] is False
    assert report["validation"]["used_for_final_selection"] is False
    assert report["validation"]["used_for_candidate_acceptance"] is False
    assert report["validation"]["heldout_diagnostic_only"] is True
    assert report["validation"]["publication_gate_applied"] is False
    assert report["rollback_performed"] is False
    assert report["rollback_exact"] is True
    assert report["published_is_incumbent"] is True
    assert torch.equal(
        rejected.published_down_weight,
        rejected.selected_down_weight,
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
