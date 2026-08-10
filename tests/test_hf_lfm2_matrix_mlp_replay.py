from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from neural_forge.hf_lfm2_matrix_mlp_replay import (
    LFM2_MATRIX_MLP_REPLAY_SCHEMA,
    Lfm2MatrixMlpCopyExactError,
    Lfm2MatrixMlpPhase,
    Lfm2MatrixMlpProjectionBundle,
    Lfm2MatrixMlpProjectionError,
    Lfm2MatrixMlpReplayError,
    Lfm2MatrixMlpReplayTolerance,
    Lfm2MatrixMlpTopology,
    evaluate_lfm2_matrix_mlp_replay,
)


def _weight(rows: int, columns: int, *, scale: float, offset: float) -> torch.Tensor:
    values = torch.arange(rows * columns, dtype=torch.float64).reshape(rows, columns)
    return torch.sin(values * scale + offset) / max(columns, 1)


def _linear(weight: torch.Tensor):
    def project(value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight)

    return project


def _fixture(
    *,
    gate_weight: torch.Tensor | None = None,
    up_weight: torch.Tensor | None = None,
    down_weight: torch.Tensor | None = None,
    input_requires_grad: bool = False,
    dtype: torch.dtype = torch.float64,
):
    topology = Lfm2MatrixMlpTopology(hidden_size=4, intermediate_size=6)
    if gate_weight is None:
        gate_weight = _weight(6, 4, scale=0.17, offset=0.2).to(dtype=dtype)
    if up_weight is None:
        up_weight = _weight(6, 4, scale=0.11, offset=-0.4).to(dtype=dtype)
    if down_weight is None:
        down_weight = _weight(4, 6, scale=0.07, offset=0.8).to(dtype=dtype)
    mlp_input = torch.linspace(-1.1, 1.3, 24, dtype=dtype).reshape(2, 3, 4)
    if input_requires_grad:
        mlp_input.requires_grad_(True)
    residual = torch.linspace(0.3, -0.6, 24, dtype=dtype).reshape(2, 3, 4)
    gate = F.linear(mlp_input, gate_weight)
    up = F.linear(mlp_input, up_weight)
    post = F.silu(gate) * up
    output = F.linear(post, down_weight)
    phase = Lfm2MatrixMlpPhase(
        topology=topology,
        phase_id="fit.batch-000.phase-000",
        mlp_input=mlp_input,
        residual_post_operator=residual,
        gate=gate,
        up=up,
        post_swiglu=post,
        output_write=output,
        residual_post_layer=residual + output,
    )
    reference = Lfm2MatrixMlpProjectionBundle(
        bundle_id="reference-bundle",
        topology=topology,
        gate_projection=_linear(gate_weight),
        up_projection=_linear(up_weight),
        down_projection=_linear(down_weight),
        declared_logical_stored_parameter_count=(6 * 4 + 6 * 4 + 4 * 6),
        declared_physical_materialized_parameter_count=(6 * 4 + 6 * 4 + 4 * 6),
    )
    return {
        "topology": topology,
        "phase": phase,
        "reference": reference,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
        "input": mlp_input,
        "residual": residual,
    }


def _candidate(
    fixture,
    *,
    bundle_id: str = "candidate-bundle",
    gate_weight: torch.Tensor | None = None,
    up_weight: torch.Tensor | None = None,
    down_weight: torch.Tensor | None = None,
    logical_count: int = 37,
    physical_count: int = 72,
) -> Lfm2MatrixMlpProjectionBundle:
    return Lfm2MatrixMlpProjectionBundle(
        bundle_id=bundle_id,
        topology=fixture["topology"],
        gate_projection=_linear(
            fixture["gate_weight"] if gate_weight is None else gate_weight
        ),
        up_projection=_linear(fixture["up_weight"] if up_weight is None else up_weight),
        down_projection=_linear(
            fixture["down_weight"] if down_weight is None else down_weight
        ),
        declared_logical_stored_parameter_count=logical_count,
        declared_physical_materialized_parameter_count=physical_count,
        declared_accounting_scope="caller-declared-test-bundle",
    )


@pytest.mark.parametrize("cell", ["C11", "C12", "C13"])
def test_replays_exact_swiglu_isolation_semantics(cell: str) -> None:
    fixture = _fixture()
    candidate_gate = fixture["gate_weight"] + _weight(6, 4, scale=0.03, offset=1.1)
    candidate_up = fixture["up_weight"] - _weight(6, 4, scale=0.05, offset=-0.7)
    candidate_down = fixture["down_weight"] + _weight(4, 6, scale=0.09, offset=0.5)
    candidate = _candidate(
        fixture,
        gate_weight=candidate_gate,
        up_weight=candidate_up,
        down_weight=candidate_down,
    )

    result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate,
        cell=cell,
    )

    parent_gate = F.linear(fixture["input"], fixture["gate_weight"])
    parent_up = F.linear(fixture["input"], fixture["up_weight"])
    changed_gate = F.linear(fixture["input"], candidate_gate)
    changed_up = F.linear(fixture["input"], candidate_up)
    if cell == "C11":
        expected_gate = changed_gate
        expected_up = parent_up
        expected_post = F.silu(changed_gate) * parent_up
        expected_write = F.linear(expected_post, fixture["down_weight"])
    elif cell == "C12":
        expected_gate = parent_gate
        expected_up = changed_up
        expected_post = F.silu(parent_gate) * changed_up
        expected_write = F.linear(expected_post, fixture["down_weight"])
    else:
        expected_gate = parent_gate
        expected_up = parent_up
        expected_post = F.silu(parent_gate) * parent_up
        expected_write = F.linear(expected_post, candidate_down)

    replay = result.phases[0]
    torch.testing.assert_close(replay.gate, expected_gate, atol=0.0, rtol=0.0)
    torch.testing.assert_close(replay.up, expected_up, atol=0.0, rtol=0.0)
    torch.testing.assert_close(replay.post_swiglu, expected_post, atol=0.0, rtol=0.0)
    torch.testing.assert_close(replay.output_write, expected_write, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        replay.residual_post_layer,
        fixture["residual"] + expected_write,
        atol=0.0,
        rtol=0.0,
    )
    assert result.schema == LFM2_MATRIX_MLP_REPLAY_SCHEMA
    assert result.copy_exact_verified is True
    assert len(result.copy_exact_endpoint_checks) == 5
    assert all(check.passed for check in result.copy_exact_endpoint_checks)
    assert result.mlp_branch_composable is True
    assert result.residual_post_layer_composable is True
    assert result.downstream_trajectory_composable is False


def test_gate_replay_uses_nonlinear_swiglu_not_additive_delta_proxy() -> None:
    fixture = _fixture()
    candidate_gate = fixture["gate_weight"] * -1.7
    candidate = _candidate(fixture, gate_weight=candidate_gate)
    result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate,
        cell="C11",
    )
    replay = result.phases[0]
    changed_gate = F.linear(fixture["input"], candidate_gate)
    parent_up = F.linear(fixture["input"], fixture["up_weight"])
    exact_write = F.linear(
        F.silu(changed_gate) * parent_up,
        fixture["down_weight"],
    )
    naive_additive = fixture["phase"].output_write + F.linear(
        changed_gate - fixture["phase"].gate,
        fixture["down_weight"],
    )
    torch.testing.assert_close(replay.output_write, exact_write, atol=0.0, rtol=0.0)
    assert not torch.allclose(replay.output_write, naive_additive)


@pytest.mark.parametrize(
    ("projection_name", "weight_name"),
    [
        ("gate_projection", "gate_weight"),
        ("up_projection", "up_weight"),
        ("down_projection", "down_weight"),
    ],
)
def test_reference_bundle_must_reproduce_every_sealed_endpoint(
    projection_name: str,
    weight_name: str,
) -> None:
    fixture = _fixture()
    wrong_reference = replace(
        fixture["reference"],
        **{projection_name: _linear(fixture[weight_name] * 0.5)},
    )
    with pytest.raises(Lfm2MatrixMlpCopyExactError, match="Reference MLP replay"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=wrong_reference,
            candidate=_candidate(fixture),
            cell="C13",
        )


@pytest.mark.parametrize("endpoint", ["post_swiglu", "residual_post_layer"])
def test_corrupt_sealed_derived_endpoint_fails_copy_calibration(endpoint: str) -> None:
    fixture = _fixture()
    corrupted = getattr(fixture["phase"], endpoint).clone()
    corrupted[0, 0, 0] += 0.25
    phase = replace(fixture["phase"], **{endpoint: corrupted})
    with pytest.raises(Lfm2MatrixMlpCopyExactError, match=endpoint):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[phase],
            reference=fixture["reference"],
            candidate=_candidate(fixture),
            cell="C13",
        )


def test_projection_shape_dtype_and_finiteness_fail_closed() -> None:
    fixture = _fixture()

    def wrong_shape(value: torch.Tensor) -> torch.Tensor:
        return value

    with pytest.raises(Lfm2MatrixMlpProjectionError, match="shape"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=fixture["reference"],
            candidate=replace(
                _candidate(fixture),
                gate_projection=wrong_shape,
            ),
            cell="C11",
        )

    def wrong_dtype(value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, fixture["gate_weight"]).float()

    with pytest.raises(Lfm2MatrixMlpProjectionError, match="dtype"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=fixture["reference"],
            candidate=replace(
                _candidate(fixture),
                gate_projection=wrong_dtype,
            ),
            cell="C11",
        )

    def nonfinite(value: torch.Tensor) -> torch.Tensor:
        result = F.linear(value, fixture["gate_weight"])
        result[0, 0, 0] = torch.inf
        return result

    with pytest.raises(Lfm2MatrixMlpProjectionError, match="finite"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=fixture["reference"],
            candidate=replace(_candidate(fixture), gate_projection=nonfinite),
            cell="C11",
        )


def test_phase_validation_rejects_nonfinite_and_mixed_dtype_authority() -> None:
    fixture = _fixture()
    bad_gate = fixture["phase"].gate.clone()
    bad_gate[0, 0, 0] = torch.nan
    with pytest.raises(Lfm2MatrixMlpReplayError, match="non-finite"):
        replace(fixture["phase"], gate=bad_gate)
    with pytest.raises(Lfm2MatrixMlpReplayError, match="one runtime dtype"):
        replace(fixture["phase"], gate=fixture["phase"].gate.float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_copy_replay_uses_runtime_float32_and_bfloat16(dtype: torch.dtype) -> None:
    fixture = _fixture(dtype=dtype)
    result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=_candidate(fixture),
        cell="C13",
    )
    assert result.copy_exact_verified is True
    assert all(check.passed for check in result.copy_exact_endpoint_checks)
    assert all(tensor.dtype == dtype for tensor in replay_tensors(result.phases[0]))


def test_copy_tolerance_cannot_be_relaxed_or_mutated_after_construction() -> None:
    with pytest.raises(Lfm2MatrixMlpReplayError, match="ceiling"):
        Lfm2MatrixMlpReplayTolerance(absolute=1.0, relative=0.0)

    fixture = _fixture()
    policy = Lfm2MatrixMlpReplayTolerance()
    object.__setattr__(policy, "relative", 1e9)
    with pytest.raises(Lfm2MatrixMlpReplayError, match="ceiling"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=fixture["reference"],
            candidate=_candidate(fixture),
            cell="C11",
            tolerance=policy,
        )


def test_post_construction_authority_mutation_is_revalidated() -> None:
    fixture = _fixture()
    corrupted_gate = fixture["phase"].gate.clone()
    corrupted_phase = replace(fixture["phase"], gate=corrupted_gate)
    corrupted_gate[0, 0, 0] = torch.nan
    with pytest.raises(Lfm2MatrixMlpReplayError, match="non-finite"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[corrupted_phase],
            reference=fixture["reference"],
            candidate=_candidate(fixture),
            cell="C11",
        )

    bad_bundle = _candidate(fixture)
    object.__setattr__(bad_bundle, "gate_projection", None)
    with pytest.raises(Lfm2MatrixMlpProjectionError, match="callable"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=fixture["reference"],
            candidate=bad_bundle,
            cell="C11",
        )


def test_declared_counts_are_explicitly_non_authoritative() -> None:
    fixture = _fixture()
    result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=_candidate(fixture, logical_count=0, physical_count=0),
        cell="C13",
    )
    assert result.declared_logical_stored_parameter_count == 0
    assert result.declared_physical_materialized_parameter_count == 0
    assert result.parameter_accounting_authority is False
    assert result.parameter_accounting_evidence == (
        "caller_declared_unverified_by_evaluator"
    )


def test_seal_covers_inactive_attribution_paths() -> None:
    fixture = _fixture()
    changed_gate = fixture["gate_weight"] * 0.8
    candidate_a = _candidate(
        fixture,
        bundle_id="same-provenance-id",
        gate_weight=changed_gate,
        down_weight=fixture["down_weight"],
    )
    candidate_b = _candidate(
        fixture,
        bundle_id="same-provenance-id",
        gate_weight=changed_gate,
        down_weight=fixture["down_weight"] * -2.0,
    )
    result_a = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate_a,
        cell="C11",
    )
    result_b = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate_b,
        cell="C11",
    )
    torch.testing.assert_close(
        result_a.phases[0].output_write,
        result_b.phases[0].output_write,
        atol=0.0,
        rtol=0.0,
    )
    assert result_a.replay_sha256 != result_b.replay_sha256


def test_seal_covers_phase_policy_counts_scope_and_claims() -> None:
    fixture = _fixture()
    base_candidate = _candidate(fixture, bundle_id="stable-bundle")
    base = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=base_candidate,
        cell="C13",
    )
    tighter = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=base_candidate,
        cell="C13",
        tolerance=Lfm2MatrixMlpReplayTolerance(absolute=1e-4, relative=1e-3),
    )
    changed_counts = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=replace(
            base_candidate,
            declared_logical_stored_parameter_count=38,
        ),
        cell="C13",
    )
    changed_scope = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=replace(base_candidate, declared_accounting_scope="other-scope"),
        cell="C13",
    )
    shifted_input = fixture["input"].detach() + 0.125
    gate = F.linear(shifted_input, fixture["gate_weight"])
    up = F.linear(shifted_input, fixture["up_weight"])
    post = F.silu(gate) * up
    write = F.linear(post, fixture["down_weight"])
    shifted_phase = replace(
        fixture["phase"],
        phase_id="fit.batch-001.phase-000",
        mlp_input=shifted_input,
        gate=gate,
        up=up,
        post_swiglu=post,
        output_write=write,
        residual_post_layer=fixture["residual"] + write,
    )
    shifted = evaluate_lfm2_matrix_mlp_replay(
        phases=[shifted_phase],
        reference=fixture["reference"],
        candidate=base_candidate,
        cell="C13",
    )
    assert (
        len(
            {
                base.replay_sha256,
                tighter.replay_sha256,
                changed_counts.replay_sha256,
                changed_scope.replay_sha256,
                shifted.replay_sha256,
            }
        )
        == 5
    )
    assert base.parameter_accounting_authority is False
    assert base.mlp_branch_composable is True
    assert base.residual_post_layer_composable is True
    assert base.downstream_trajectory_composable is False


def test_zero_reference_nonzero_candidate_has_stable_infinite_metric() -> None:
    zero_down = torch.zeros(4, 6, dtype=torch.float64)
    fixture = _fixture(down_weight=zero_down)
    candidate_down = _weight(4, 6, scale=0.13, offset=0.4)
    candidate = _candidate(fixture, down_weight=candidate_down)
    first = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate,
        cell="C13",
    )
    second = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate,
        cell="C13",
    )
    assert math_is_positive_infinity(
        first.phase_metrics[0].output_write.aggregate_relative_l2
    )
    assert first.replay_sha256 == second.replay_sha256


def math_is_positive_infinity(value: float) -> bool:
    return value == float("inf") and value > 0.0


def test_evaluation_is_no_grad_and_projection_cannot_mutate_phase_input() -> None:
    gate_weight = _weight(6, 4, scale=0.17, offset=0.2).requires_grad_(True)
    up_weight = _weight(6, 4, scale=0.11, offset=-0.4).requires_grad_(True)
    down_weight = _weight(4, 6, scale=0.07, offset=0.8).requires_grad_(True)
    fixture = _fixture(
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        input_requires_grad=True,
    )
    original_input = fixture["phase"].mlp_input.detach().clone()

    def mutating_gate(value: torch.Tensor) -> torch.Tensor:
        value.add_(3.0)
        return F.linear(value, gate_weight)

    candidate = replace(_candidate(fixture), gate_projection=mutating_gate)
    result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate,
        cell="C11",
    )
    torch.testing.assert_close(
        fixture["phase"].mlp_input,
        original_input,
        atol=0.0,
        rtol=0.0,
    )
    assert all(not tensor.requires_grad for tensor in replay_tensors(result.phases[0]))
    assert gate_weight.grad is None
    assert up_weight.grad is None
    assert down_weight.grad is None

    stored_grad_output = F.linear(
        fixture["input"].detach(),
        gate_weight.detach(),
    ).requires_grad_(True)

    def return_stored_grad_tensor(_value: torch.Tensor) -> torch.Tensor:
        return stored_grad_output

    stored_result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=replace(
            _candidate(fixture),
            gate_projection=return_stored_grad_tensor,
        ),
        cell="C11",
    )
    assert all(
        not tensor.requires_grad for tensor in replay_tensors(stored_result.phases[0])
    )


def test_projection_results_are_isolated_from_reused_mutable_buffers() -> None:
    fixture = _fixture()
    shared_buffer = None

    def stateful_down(value: torch.Tensor) -> torch.Tensor:
        nonlocal shared_buffer
        projected = F.linear(value, fixture["down_weight"])
        if shared_buffer is None:
            shared_buffer = torch.empty_like(projected)
        shared_buffer.copy_(projected)
        return shared_buffer

    changed_gate = fixture["gate_weight"] * -0.75
    candidate = replace(
        _candidate(fixture, gate_weight=changed_gate),
        down_projection=stateful_down,
    )
    result = evaluate_lfm2_matrix_mlp_replay(
        phases=[fixture["phase"]],
        reference=fixture["reference"],
        candidate=candidate,
        cell="C13",
    )
    expected = F.linear(fixture["phase"].post_swiglu, fixture["down_weight"])
    torch.testing.assert_close(
        result.phases[0].output_write,
        expected,
        atol=0.0,
        rtol=0.0,
    )


def replay_tensors(replay):
    return (
        replay.gate,
        replay.up,
        replay.post_swiglu,
        replay.output_write,
        replay.residual_post_layer,
        replay.attribution.candidate_gate_parent_up_post,
        replay.attribution.parent_gate_candidate_up_post,
        replay.attribution.candidate_gate_candidate_up_post,
        replay.attribution.parent_down_candidate_gate_post,
        replay.attribution.parent_down_candidate_up_post,
        replay.attribution.parent_down_joint_post,
        replay.attribution.candidate_down_parent_post,
        replay.attribution.candidate_down_joint_post,
    )


def test_phase_identity_and_cell_validation_fail_closed() -> None:
    fixture = _fixture()
    duplicate = replace(fixture["phase"])
    with pytest.raises(Lfm2MatrixMlpReplayError, match="unique"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"], duplicate],
            reference=fixture["reference"],
            candidate=_candidate(fixture),
            cell="C11",
        )
    with pytest.raises(Lfm2MatrixMlpReplayError, match="cell must be"):
        evaluate_lfm2_matrix_mlp_replay(
            phases=[fixture["phase"]],
            reference=fixture["reference"],
            candidate=_candidate(fixture),
            cell="C10",  # type: ignore[arg-type]
        )
