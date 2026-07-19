from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    Lfm2CpcfV4ProjectionCase,
    replay_sampled_cp_volterra,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_matrix_free_bcx import (  # noqa: E402
    Lfm2CpcfV5MatrixFreeBcxPolicy,
    causal_horizon_row_mass,
    compare_lfm2_cpcf_v5_matrix_free_bcx_to_legacy_oracle,
    materialize_lfm2_cpcf_v5_bcx_legacy_oracle,
    solve_lfm2_cpcf_v5_matrix_free_bcx,
    verify_lfm2_cpcf_short_conv_v5_matrix_free_bcx_report,
)


def _small_problem(
    *,
    hidden: int = 2,
    sequence: int = 6,
    fit_bias: bool = True,
    case_count: int = 2,
):
    generator = torch.Generator().manual_seed(93017)
    in_weight = (
        torch.randn((3 * hidden, hidden), generator=generator, dtype=torch.float64)
        * 0.22
    )
    in_bias = (
        torch.randn((3 * hidden,), generator=generator, dtype=torch.float64) * 0.04
        if fit_bias
        else None
    )
    fir_kernel = (
        torch.randn((hidden, 3), generator=generator, dtype=torch.float64) * 0.3
    )
    fir_bias = torch.randn((hidden,), generator=generator, dtype=torch.float64) * 0.03
    writer_weight = torch.tensor(
        [[0.7, -0.45], [0.35, 0.8]],
        dtype=torch.float64,
    )
    if hidden != 2:
        writer_weight = (
            torch.randn((hidden, hidden), generator=generator, dtype=torch.float64)
            * 0.2
            + torch.eye(hidden, dtype=torch.float64) * 0.6
        )
    cases = []
    replays = []
    for case_index in range(case_count):
        child = torch.randn(
            (1, sequence, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        valid = torch.ones((1, sequence), dtype=torch.bool)
        replay = replay_sampled_cp_volterra(
            child,
            in_proj_weight=in_weight,
            in_proj_bias=in_bias,
            fir_kernel=fir_kernel,
            fir_bias=fir_bias,
            writer_weight=writer_weight,
            writer_bias=None,
            valid_prefix_mask=valid,
        )
        context_residual = (
            torch.randn(
                replay.context_history.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.025
        )
        endpoint_residual = (
            torch.randn(
                replay.write_history.shape,
                generator=generator,
                dtype=torch.float64,
            )
            * 0.04
        )
        cases.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=f"fit-{case_index:02d}",
                split="fit",
                child_input_history=child,
                desired_context_history=(replay.context_history + context_residual),
                desired_write_suffix=(replay.write_history + endpoint_residual),
                suffix_start=0,
                valid_prefix_mask=valid,
                valid_suffix_mask=valid,
            )
        )
        replays.append(replay)
    return {
        "cases": tuple(cases),
        "replays": tuple(replays),
        "in_proj_weight": in_weight,
        "in_proj_bias": in_bias,
        "fir_kernel": fir_kernel,
        "fir_bias": fir_bias,
        "writer_weight": writer_weight,
    }


def test_causal_horizon_mass_is_normalized_and_tail_sensitive() -> None:
    mask = torch.ones((1, 256), dtype=torch.bool)
    mass = causal_horizon_row_mass(mask)

    assert mass.dtype == torch.float64
    assert float(mass.sum().item()) == pytest.approx(1.0)
    assert float(mass[0, -1].item()) > 8.0 * float(mass[0, 0].item())
    assert float(mass[0, :128].sum().item()) < 0.3

    masked = torch.tensor([[True, True, False, False], [False, True, True, True]])
    masked_mass = causal_horizon_row_mass(masked)
    assert float(masked_mass.sum().item()) == pytest.approx(1.0)
    assert torch.equal(masked_mass == 0.0, ~masked)


@pytest.mark.parametrize("fit_bias", [False, True])
def test_matrix_free_global_pcg_matches_exact_legacy_oracle(
    fit_bias: bool,
) -> None:
    problem = _small_problem(fit_bias=fit_bias)
    policy = Lfm2CpcfV5MatrixFreeBcxPolicy(
        backend="cpu",
        condition_cap=10_000.0,
        pcg_iterations=96,
        parameter_trust_radius=4.0,
        effect_trust_radius=4.0,
    )

    comparison = compare_lfm2_cpcf_v5_matrix_free_bcx_to_legacy_oracle(
        **problem,
        policy=policy,
    )

    assert verify_lfm2_cpcf_short_conv_v5_matrix_free_bcx_report(comparison)
    assert comparison["relative_correction_error"] < 2e-7
    assert comparison["maximum_absolute_correction_error"] < 2e-7
    assert comparison["matrix_free_normal_residual_norm"] < 1e-8


def test_dense_oracle_contains_cross_channel_writer_coupling() -> None:
    problem = _small_problem(case_count=1)
    oracle = materialize_lfm2_cpcf_v5_bcx_legacy_oracle(**problem)
    hidden = 2
    feature_width = 3
    gram = oracle.design.T @ oracle.design
    first_b_channel = slice(0, feature_width)
    second_b_channel = slice(feature_width, 2 * feature_width)

    assert oracle.verify_authentication()
    assert float(gram[first_b_channel, second_b_channel].abs().max().item()) > 1e-10
    assert oracle.report["global_writer_channel_coupling"] is True
    assert oracle.report["legacy_per_channel_design_function"].endswith(
        "linearized_bcx_affine_design"
    )
    assert oracle.design.shape[1] == 3 * hidden * feature_width


def test_result_preserves_every_dense_coordinate_and_has_no_training() -> None:
    problem = _small_problem(hidden=3, sequence=9, case_count=3)
    policy = Lfm2CpcfV5MatrixFreeBcxPolicy(
        backend="cpu",
        pcg_iterations=12,
    )

    result = solve_lfm2_cpcf_v5_matrix_free_bcx(
        **problem,
        policy=policy,
    )
    report = result.diagnostics

    assert result.verify_authentication()
    assert result.in_proj_weight.shape == problem["in_proj_weight"].shape
    assert result.in_proj_bias is not None
    assert result.in_proj_bias.shape == problem["in_proj_bias"].shape
    assert result.raw_correction.shape == (3, 3, 4)
    assert report["physical_correction_coordinate_count"] == 36
    assert report["all_physical_correction_coordinates_preserved"] is True
    assert report["dense_transported_prior_preserved"] is True
    assert len(report["input_state_sha256"]) == 64
    assert set(report["input_state_tensor_sha256"]) == {
        "fir_bias",
        "fir_kernel",
        "in_proj_bias",
        "in_proj_weight",
        "writer_weight",
    }
    assert report["global_writer_channel_coupling"] is True
    assert report["solve"]["normal_design_materialized"] is False
    assert report["solve"]["gram_materialized"] is False
    assert report["solve"]["early_stopping"] is False
    assert len(report["solve"]["residual_norms"]) == 13
    assert report["conditioning"]["condition_cap_passed"] is True
    assert report["conditioning"][
        "regularized_condition_upper_bound"
    ] <= policy.condition_cap * (1.0 + 1e-12)
    assert report["row_mass_contract"]["suffix_start_used_as_decode_boundary"] is False
    assert report["execution"]["training_steps"] == 0
    assert report["execution"]["backward_calls"] == 0
    assert report["execution"]["optimizer_steps"] == 0
    assert report["execution"]["parameters_pruned"] == 0
    assert report["execution"]["parameters_quantized"] == 0
    assert math.isfinite(report["weighted_fit_objective_after"])
    assert report["weighted_fit_objective_improved"] is True


def test_matrix_free_solve_is_bitwise_repeatable_on_cpu() -> None:
    problem = _small_problem(hidden=3, sequence=8, case_count=2)
    policy = Lfm2CpcfV5MatrixFreeBcxPolicy(
        backend="cpu",
        pcg_iterations=10,
    )

    first = solve_lfm2_cpcf_v5_matrix_free_bcx(**problem, policy=policy)
    second = solve_lfm2_cpcf_v5_matrix_free_bcx(**problem, policy=policy)

    assert torch.equal(first.in_proj_weight, second.in_proj_weight)
    assert torch.equal(first.in_proj_bias, second.in_proj_bias)
    assert torch.equal(first.raw_correction, second.raw_correction)
    assert first.diagnostics["report_sha256"] == second.diagnostics["report_sha256"]
