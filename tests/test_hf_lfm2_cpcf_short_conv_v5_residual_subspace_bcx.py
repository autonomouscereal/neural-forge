from __future__ import annotations

import copy

import pytest
import torch

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfV4ProjectionCase,
    replay_sampled_cp_volterra,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import Lfm2CpcfV5RidgePolicy
from neural_forge.hf_lfm2_cpcf_short_conv_v5_residual_subspace_bcx import (
    Lfm2CpcfV5ResidualSubspaceBcxError,
    automatic_residual_subspace_rank,
    causal_horizon_balanced_row_weights,
    solve_lfm2_cpcf_v5_residual_subspace_bcx,
)


def _fixture(
    *,
    hidden: int = 4,
    include_bias: bool = True,
    endpoint_only: bool = False,
) -> tuple[
    tuple[Lfm2CpcfV4ProjectionCase, ...],
    dict[str, object],
    dict[str, torch.Tensor | None],
]:
    generator = torch.Generator().manual_seed(20260719)
    state: dict[str, torch.Tensor | None] = {
        "in_proj_weight": torch.randn(
            (3 * hidden, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.12,
        "in_proj_bias": (
            torch.randn(
                (3 * hidden,),
                generator=generator,
                dtype=torch.float64,
            )
            * 0.02
            if include_bias
            else None
        ),
        "fir_kernel": torch.tensor(
            [0.72, -0.19, 0.08],
            dtype=torch.float64,
        ).repeat(hidden, 1),
        "fir_bias": (
            torch.linspace(
                -0.01,
                0.01,
                hidden,
                dtype=torch.float64,
            )
            if include_bias
            else None
        ),
        "writer_weight": torch.randn(
            (hidden, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.3,
        "writer_bias": (
            torch.randn(
                (hidden,),
                generator=generator,
                dtype=torch.float64,
            )
            * 0.01
            if include_bias
            else None
        ),
    }
    target_state = {
        name: None if value is None else value.clone()
        for name, value in state.items()
    }
    direction = torch.linspace(
        -0.09,
        0.11,
        hidden,
        dtype=torch.float64,
    )
    target_state["in_proj_weight"][hidden + 1] += direction
    if include_bias:
        target_state["in_proj_bias"][hidden + 1] += 0.045
    cases = []
    replays = {}
    for index, case_id in enumerate(("case-a", "case-b", "case-c")):
        batch = 1 + int(index == 2)
        tokens = 7 + index
        suffix_start = 3
        child = torch.randn(
            (batch, tokens, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        valid = torch.ones((batch, tokens), dtype=torch.bool)
        if batch == 2:
            valid[1, 0] = False
        current = replay_sampled_cp_volterra(
            child,
            in_proj_weight=state["in_proj_weight"],
            in_proj_bias=state["in_proj_bias"],
            fir_kernel=state["fir_kernel"],
            fir_bias=state["fir_bias"],
            writer_weight=state["writer_weight"],
            writer_bias=state["writer_bias"],
            valid_prefix_mask=valid,
        )
        target = replay_sampled_cp_volterra(
            child,
            in_proj_weight=target_state["in_proj_weight"],
            in_proj_bias=target_state["in_proj_bias"],
            fir_kernel=target_state["fir_kernel"],
            fir_bias=target_state["fir_bias"],
            writer_weight=target_state["writer_weight"],
            writer_bias=target_state["writer_bias"],
            valid_prefix_mask=valid,
        )
        desired_context = (
            current.context_history
            if endpoint_only
            else target.context_history
        )
        cases.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=case_id,
                split="fit",
                child_input_history=child,
                desired_context_history=desired_context,
                desired_write_suffix=target.write_history[:, suffix_start:, :],
                suffix_start=suffix_start,
                valid_prefix_mask=valid,
                valid_suffix_mask=valid[:, suffix_start:],
            )
        )
        replays[case_id] = current
    return tuple(cases), replays, state


def _write_mse(
    cases: tuple[Lfm2CpcfV4ProjectionCase, ...],
    state: dict[str, torch.Tensor | None],
) -> float:
    values = []
    for case in cases:
        replay = replay_sampled_cp_volterra(
            case.child_input_history,
            in_proj_weight=state["in_proj_weight"],
            in_proj_bias=state["in_proj_bias"],
            fir_kernel=state["fir_kernel"],
            fir_bias=state["fir_bias"],
            writer_weight=state["writer_weight"],
            writer_bias=state["writer_bias"],
            valid_prefix_mask=case.valid_prefix_mask,
        )
        difference = (
            replay.write_history[:, case.suffix_start :, :]
            - case.desired_write_suffix
        )
        rows = difference[case.valid_suffix_mask]
        values.append(float(rows.square().mean().item()))
    return sum(values) / len(values)


def _solve(
    cases: tuple[Lfm2CpcfV4ProjectionCase, ...],
    replays: dict[str, object],
    state: dict[str, torch.Tensor | None],
):
    return solve_lfm2_cpcf_v5_residual_subspace_bcx(
        cases=cases,
        replays=replays,
        state=state,
        cap_id="kappa-01000",
        ridge_policy=Lfm2CpcfV5RidgePolicy(
            parameter_trust_radius=100.0,
            effect_trust_radius=100.0,
        ),
        compute_device="cpu",
        channel_chunk_size=2,
    )


def test_dynamic_rank_uses_small_full_rank_and_sqrt_power_of_two() -> None:
    assert automatic_residual_subspace_rank(3) == 3
    assert automatic_residual_subspace_rank(32) == 32
    assert automatic_residual_subspace_rank(33) == 8
    assert automatic_residual_subspace_rank(776) == 32
    assert automatic_residual_subspace_rank(4096) == 64
    assert automatic_residual_subspace_rank(16384) == 64


def test_causal_horizon_weights_mix_uniform_coverage_and_decode_tail() -> None:
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [False, True, True, True, True, True],
        ]
    )

    weights = causal_horizon_balanced_row_weights(mask)

    assert weights.dtype == torch.float64
    assert float(weights.sum().item()) == pytest.approx(1.0)
    assert torch.equal(weights == 0.0, ~mask)
    assert float(weights[0, -1]) > float(weights[0, 0])
    uniform = mask.to(torch.float64) / int(mask.sum())
    assert torch.all(weights[mask] >= 0.5 * uniform[mask])


def test_endpoint_residual_drives_coordinate_fold_when_context_target_is_current() -> None:
    cases, replays, state = _fixture(endpoint_only=True)
    before = _write_mse(cases, state)

    result = _solve(cases, replays, state)
    candidate_state = {
        **state,
        "in_proj_weight": result.in_proj_weight,
        "in_proj_bias": result.in_proj_bias,
    }
    after = _write_mse(cases, candidate_state)

    assert result.verify_authentication()
    assert after < before
    assert after <= 0.8 * before
    assert result.diagnostics["endpoint_objective_primary"] is True
    assert (
        result.diagnostics["writer_cross_channel_contract"]
        == "deterministic_channel_coordinate_updates_with_"
        "writer_projected_residual_sufficient_statistics_refresh"
    )
    assert result.diagnostics["independent_simultaneous_endpoint_solve_claimed"] is False
    assert result.diagnostics["sum_coordinate_objective_after"] <= (
        result.diagnostics["sum_coordinate_objective_before"]
    )


def test_solver_is_case_order_deterministic_and_keeps_bias_coordinate_physical() -> None:
    cases, replays, state = _fixture(include_bias=True)

    first = _solve(cases, replays, state)
    second = _solve(tuple(reversed(cases)), replays, state)

    assert torch.equal(first.in_proj_weight, second.in_proj_weight)
    assert torch.equal(first.in_proj_bias, second.in_proj_bias)
    assert first.receipt_sha256 == second.receipt_sha256
    assert dict(first.diagnostics) == dict(second.diagnostics)
    assert (
        first.diagnostics["physical_bias_coordinate_exact_and_unprojected"]
        is True
    )
    assert not torch.equal(first.in_proj_bias, state["in_proj_bias"])


def test_zero_residual_preserves_dense_prior_exactly() -> None:
    cases, replays, state = _fixture(include_bias=True)
    zero_cases = tuple(
        Lfm2CpcfV4ProjectionCase(
            case_id=case.case_id,
            split="fit",
            child_input_history=case.child_input_history,
            desired_context_history=replays[case.case_id].context_history,
            desired_write_suffix=replays[case.case_id].write_history[
                :, case.suffix_start :, :
            ],
            suffix_start=case.suffix_start,
            valid_prefix_mask=case.valid_prefix_mask,
            valid_suffix_mask=case.valid_suffix_mask,
        )
        for case in cases
    )
    before = copy.deepcopy(state)

    result = _solve(zero_cases, replays, state)

    assert result.verify_authentication()
    assert torch.equal(result.in_proj_weight, before["in_proj_weight"])
    assert torch.equal(result.in_proj_bias, before["in_proj_bias"])
    assert result.diagnostics["dense_prior_preserved_additively"] is True
    assert all(
        report["joint_trust_scale"] == pytest.approx(1.0)
        for report in result.diagnostics["channel_reports"]
    )


def test_solver_rejects_nonfit_or_misaligned_evidence() -> None:
    cases, replays, state = _fixture()
    bad = Lfm2CpcfV4ProjectionCase(
        case_id="selection-case",
        split="selection",
        child_input_history=cases[0].child_input_history,
        desired_context_history=cases[0].desired_context_history,
        desired_write_suffix=cases[0].desired_write_suffix,
        suffix_start=cases[0].suffix_start,
        valid_prefix_mask=cases[0].valid_prefix_mask,
        valid_suffix_mask=cases[0].valid_suffix_mask,
    )

    with pytest.raises(
        Lfm2CpcfV5ResidualSubspaceBcxError,
        match="FIT",
    ):
        solve_lfm2_cpcf_v5_residual_subspace_bcx(
            cases=(bad,),
            replays={"selection-case": replays["case-a"]},
            state=state,
            cap_id="kappa-01000",
            ridge_policy=Lfm2CpcfV5RidgePolicy(),
            compute_device="cpu",
        )
