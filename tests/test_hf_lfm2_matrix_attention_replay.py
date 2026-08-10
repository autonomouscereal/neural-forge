from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from neural_forge.hf_lfm2_matrix_attention_replay import (
    MAX_COPY_EXACT_ABSOLUTE_TOLERANCE,
    MAX_COPY_EXACT_RELATIVE_TOLERANCE,
    Lfm2MatrixAttentionContinuityError,
    Lfm2MatrixAttentionCopyExactError,
    Lfm2MatrixAttentionPhase,
    Lfm2MatrixAttentionProjectionBundle,
    Lfm2MatrixAttentionReplayError,
    Lfm2MatrixAttentionReplayTolerance,
    Lfm2MatrixAttentionTopology,
    evaluate_lfm2_matrix_attention_replay,
)


@dataclass(frozen=True)
class _Weights:
    query: Tensor
    key: Tensor
    value: Tensor
    output: Tensor
    query_norm: Tensor
    key_norm: Tensor


def _linear(weight: Tensor):
    def project(value: Tensor) -> Tensor:
        return F.linear(value, weight)

    return project


def _bundle(
    bundle_id: str,
    topology: Lfm2MatrixAttentionTopology,
    weights: _Weights,
    *,
    logical_count: int | None = None,
    physical_count: int | None = None,
) -> Lfm2MatrixAttentionProjectionBundle:
    total = sum(int(value.numel()) for value in weights.__dict__.values())
    return Lfm2MatrixAttentionProjectionBundle(
        bundle_id=bundle_id,
        topology=topology,
        query_projection=_linear(weights.query),
        key_projection=_linear(weights.key),
        value_projection=_linear(weights.value),
        output_projection=_linear(weights.output),
        query_norm_weight=weights.query_norm,
        key_norm_weight=weights.key_norm,
        declared_logical_stored_parameter_count=total
        if logical_count is None
        else logical_count,
        declared_physical_materialized_parameter_count=(
            total if physical_count is None else physical_count
        ),
    )


def _manual_rms_norm(value: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    input_dtype = value.dtype
    direction = value.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    return weight.to(dtype=input_dtype) * direction.to(dtype=input_dtype)


def _rotate_half(value: Tensor) -> Tensor:
    half = int(value.shape[-1]) // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope_terms(
    positions: Tensor, *, batch: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    angle = positions.to(torch.float64).reshape(1, -1, 1) * 0.37
    angle = torch.cat((angle, angle), dim=-1)
    return (
        angle.cos().to(dtype).expand(batch, -1, -1).contiguous(),
        angle.sin().to(dtype).expand(batch, -1, -1).contiguous(),
    )


def _manual_phase(
    *,
    topology: Lfm2MatrixAttentionTopology,
    phase_id: str,
    phase_kind: str,
    decode_step: int | None,
    operator_input: Tensor,
    residual_pre: Tensor,
    positions: Tensor,
    weights: _Weights,
    prior_key: Tensor | None,
    prior_value: Tensor | None,
) -> Lfm2MatrixAttentionPhase:
    batch, queries, _hidden = operator_input.shape
    query_linear_sequence = F.linear(operator_input, weights.query).view(
        batch,
        queries,
        topology.query_heads,
        topology.head_dim,
    )
    key_linear_sequence = F.linear(operator_input, weights.key).view(
        batch,
        queries,
        topology.key_value_heads,
        topology.head_dim,
    )
    value_current = (
        F.linear(operator_input, weights.value)
        .view(
            batch,
            queries,
            topology.key_value_heads,
            topology.head_dim,
        )
        .transpose(1, 2)
        .contiguous()
    )
    query_normalized = (
        _manual_rms_norm(
            query_linear_sequence,
            weights.query_norm,
            topology.norm_epsilon,
        )
        .transpose(1, 2)
        .contiguous()
    )
    key_normalized = (
        _manual_rms_norm(
            key_linear_sequence,
            weights.key_norm,
            topology.norm_epsilon,
        )
        .transpose(1, 2)
        .contiguous()
    )
    cosine, sine = _rope_terms(positions, batch=batch, dtype=operator_input.dtype)
    query = (
        query_normalized * cosine.unsqueeze(1)
        + _rotate_half(query_normalized) * sine.unsqueeze(1)
    ).contiguous()
    current_key = (
        key_normalized * cosine.unsqueeze(1)
        + _rotate_half(key_normalized) * sine.unsqueeze(1)
    ).contiguous()
    key_history = (
        current_key
        if prior_key is None
        else torch.cat((prior_key, current_key), dim=2).contiguous()
    )
    value_history = (
        value_current
        if prior_value is None
        else torch.cat((prior_value, value_current), dim=2).contiguous()
    )
    history = int(key_history.shape[2])
    repeated_key = key_history.repeat_interleave(topology.groups, dim=1)
    repeated_value = value_history.repeat_interleave(topology.groups, dim=1)
    scores = (query @ repeated_key.transpose(2, 3)) * float(topology.head_dim**-0.5)
    if phase_kind == "prefill":
        allowed = torch.ones((queries, history), dtype=torch.bool).tril()
    else:
        allowed = torch.ones((queries, history), dtype=torch.bool)
    minimum = torch.finfo(operator_input.dtype).min
    additive_mask = torch.where(
        allowed,
        torch.zeros((), dtype=operator_input.dtype),
        torch.full((), minimum, dtype=operator_input.dtype),
    ).reshape(1, 1, queries, history)
    probabilities = F.softmax(
        scores + additive_mask,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    context = (probabilities @ repeated_value).contiguous()
    context_flat = context.transpose(1, 2).reshape(
        batch,
        queries,
        topology.hidden_size,
    )
    output_write = F.linear(context_flat, weights.output).contiguous()
    return Lfm2MatrixAttentionPhase(
        topology=topology,
        phase_id=phase_id,
        phase_kind=phase_kind,
        decode_step=decode_step,
        operator_input=operator_input,
        residual_pre=residual_pre,
        query_linear=query_linear_sequence.transpose(1, 2).contiguous(),
        key_linear=key_linear_sequence.transpose(1, 2).contiguous(),
        query_normalized_pre_rope=query_normalized,
        key_normalized_pre_rope=key_normalized,
        rope_cos=cosine,
        rope_sin=sine,
        query_post_rope=query,
        key_post_rope_current=current_key,
        value_current=value_current,
        key_history=key_history,
        value_history=value_history,
        qk_scores=scores,
        additive_mask=additive_mask,
        probabilities=probabilities,
        context=context,
        context_from_probabilities=context.clone(),
        output_write=output_write,
        residual_post_operator=(residual_pre + output_write).contiguous(),
    )


def _fixture() -> tuple[
    tuple[Lfm2MatrixAttentionPhase, ...],
    Lfm2MatrixAttentionProjectionBundle,
    Lfm2MatrixAttentionProjectionBundle,
]:
    torch.manual_seed(1701)
    dtype = torch.float64
    topology = Lfm2MatrixAttentionTopology(
        hidden_size=8,
        query_heads=4,
        key_value_heads=2,
        head_dim=2,
        norm_epsilon=1e-5,
    )
    reference_weights = _Weights(
        query=torch.randn(8, 8, dtype=dtype) * 0.22,
        key=torch.randn(4, 8, dtype=dtype) * 0.19,
        value=torch.randn(4, 8, dtype=dtype) * 0.17,
        output=torch.randn(8, 8, dtype=dtype) * 0.13,
        query_norm=torch.tensor([0.91, 1.07], dtype=dtype),
        key_norm=torch.tensor([1.11, 0.94], dtype=dtype),
    )
    prefill_input = torch.randn(2, 3, 8, dtype=dtype)
    decode_input = torch.randn(2, 1, 8, dtype=dtype)
    prefill = _manual_phase(
        topology=topology,
        phase_id="prefill",
        phase_kind="prefill",
        decode_step=None,
        operator_input=prefill_input,
        residual_pre=prefill_input * 0.31 + 0.07,
        positions=torch.arange(3),
        weights=reference_weights,
        prior_key=None,
        prior_value=None,
    )
    decode = _manual_phase(
        topology=topology,
        phase_id="decode.000",
        phase_kind="decode",
        decode_step=0,
        operator_input=decode_input,
        residual_pre=decode_input * 0.29 - 0.04,
        positions=torch.tensor([3]),
        weights=reference_weights,
        prior_key=prefill.key_history,
        prior_value=prefill.value_history,
    )
    candidate_weights = _Weights(
        query=reference_weights.query.clone(),
        key=reference_weights.key.clone(),
        value=reference_weights.value.clone(),
        output=reference_weights.output.clone(),
        query_norm=reference_weights.query_norm.clone(),
        key_norm=reference_weights.key_norm.clone(),
    )
    candidate_weights.query[0, 0] += 0.73
    candidate_weights.query[3, 4] -= 0.31
    candidate_weights.key[1, 2] += 0.61
    candidate_weights.key[3, 6] -= 0.27
    candidate_weights.value[0, 1] -= 0.52
    candidate_weights.value[2, 7] += 0.44
    candidate_weights.output[0, 0] += 0.67
    candidate_weights.output[5, 3] -= 0.38
    candidate_weights.query_norm[0] *= 1.08
    candidate_weights.key_norm[1] *= 0.89
    reference = _bundle("reference", topology, reference_weights)
    candidate = _bundle(
        "candidate",
        topology,
        candidate_weights,
        logical_count=37,
        physical_count=113,
    )
    return (prefill, decode), reference, candidate


@pytest.mark.parametrize("cell", ["C06", "C07", "C08", "C09", "C10"])
def test_copy_exact_reproduces_structured_gqa_prefill_and_decode(cell: str) -> None:
    phases, reference, _candidate = _fixture()
    result = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=reference,
        cell=cell,
    )

    assert result.copy_exact_verified
    assert result.score_preservation.softmax_fisher_error_worst == pytest.approx(0.0)
    assert result.score_preservation.js_divergence_worst == pytest.approx(0.0)
    assert result.score_preservation.top1_key_agreement_fraction == pytest.approx(1.0)
    for phase, replay, metrics in zip(
        phases,
        result.phases,
        result.phase_metrics,
        strict=True,
    ):
        assert replay.qk_scores.shape[1] == 4
        assert replay.key_history.shape[1] == 2
        assert replay.probabilities.shape[-1] == phase.history_count
        assert torch.equal(replay.context, phase.context)
        assert metrics.context.maximum_absolute_error == pytest.approx(0.0)
        assert metrics.output_write.maximum_absolute_error == pytest.approx(0.0)
        assert metrics.key_cache.maximum_absolute_error == pytest.approx(0.0)
        assert metrics.value_cache.maximum_absolute_error == pytest.approx(0.0)


def test_one_side_interventions_obey_all_cell_boundaries_and_attribution() -> None:
    phases, reference, candidate = _fixture()
    results = {
        cell: evaluate_lfm2_matrix_attention_replay(
            phases=phases,
            reference=reference,
            candidate=candidate,
            cell=cell,
        )
        for cell in ("C06", "C07", "C08", "C09", "C10")
    }
    prefill = phases[0]

    query = results["C06"].phases[0]
    assert not torch.equal(query.query_post_rope, prefill.query_post_rope)
    assert torch.equal(query.key_history, prefill.key_history)
    assert torch.equal(query.value_history, prefill.value_history)
    assert torch.equal(query.context, query.attribution.qk_only_context)
    assert torch.equal(query.output_write, query.attribution.parent_o_candidate_context)
    assert not torch.equal(
        query.output_write,
        query.attribution.candidate_o_candidate_context,
    )

    key = results["C07"].phases[0]
    assert torch.equal(key.query_post_rope, prefill.query_post_rope)
    assert not torch.equal(key.key_history, prefill.key_history)
    assert torch.equal(key.value_history, prefill.value_history)
    assert torch.equal(key.context, key.attribution.qk_only_context)
    assert torch.equal(key.output_write, key.attribution.parent_o_candidate_context)

    value = results["C08"].phases[0]
    assert torch.equal(value.query_post_rope, prefill.query_post_rope)
    assert torch.equal(value.key_history, prefill.key_history)
    assert not torch.equal(value.value_history, prefill.value_history)
    assert torch.equal(value.qk_scores, prefill.qk_scores)
    assert torch.equal(value.probabilities, prefill.probabilities)
    assert torch.equal(value.context, value.attribution.value_only_context)
    assert torch.equal(value.output_write, value.attribution.parent_o_candidate_context)

    output = results["C09"].phases[0]
    assert torch.equal(output.query_post_rope, prefill.query_post_rope)
    assert torch.equal(output.key_history, prefill.key_history)
    assert torch.equal(output.value_history, prefill.value_history)
    assert torch.equal(output.context, prefill.context)
    assert torch.equal(
        output.output_write, output.attribution.candidate_o_parent_context
    )
    assert not torch.equal(output.output_write, prefill.output_write)

    joint = results["C10"].phases[0]
    assert not torch.equal(joint.query_post_rope, prefill.query_post_rope)
    assert not torch.equal(joint.key_history, prefill.key_history)
    assert not torch.equal(joint.value_history, prefill.value_history)
    assert torch.equal(joint.context, joint.attribution.joint_context)
    assert torch.equal(
        joint.output_write, joint.attribution.candidate_o_candidate_context
    )

    assert results["C07"].phase_metrics[1].key_cache_prefix_drift is not None
    assert (
        results["C07"].phase_metrics[1].key_cache_prefix_drift.aggregate_relative_l2
        > 0.0
    )
    assert results["C08"].phase_metrics[1].value_cache_prefix_drift is not None
    assert (
        results["C08"].phase_metrics[1].value_cache_prefix_drift.aggregate_relative_l2
        > 0.0
    )
    assert results["C10"].declared_logical_stored_parameter_count == 37
    assert results["C10"].declared_physical_materialized_parameter_count == 113
    assert results["C10"].parameter_accounting_authority is False
    assert (
        results["C10"].parameter_accounting_evidence
        == "caller_declared_unverified_by_evaluator"
    )


def test_broken_parent_cache_continuity_is_rejected_before_replay() -> None:
    phases, reference, candidate = _fixture()
    broken_history = phases[1].key_history.clone()
    broken_history[0, 0, 0, 0] += 1.0
    broken_decode = replace(phases[1], key_history=broken_history)

    with pytest.raises(
        Lfm2MatrixAttentionContinuityError,
        match="changed its sealed KV prefix",
    ):
        evaluate_lfm2_matrix_attention_replay(
            phases=(phases[0], broken_decode),
            reference=reference,
            candidate=candidate,
            cell="C10",
        )


def test_reference_bundle_must_pass_copy_exact_calibration() -> None:
    phases, _reference, candidate = _fixture()

    with pytest.raises(Lfm2MatrixAttentionCopyExactError):
        evaluate_lfm2_matrix_attention_replay(
            phases=phases,
            reference=candidate,
            candidate=candidate,
            cell="C10",
        )


def test_replay_is_deterministic() -> None:
    phases, reference, candidate = _fixture()
    first = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=candidate,
        cell="C10",
    )
    second = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=candidate,
        cell="C10",
    )

    assert first.replay_sha256 == second.replay_sha256
    assert first.phase_metrics == second.phase_metrics
    assert first.score_preservation == second.score_preservation
    for left, right in zip(first.phases, second.phases, strict=True):
        assert torch.equal(left.key_history, right.key_history)
        assert torch.equal(left.value_history, right.value_history)
        assert torch.equal(left.qk_scores, right.qk_scores)
        assert torch.equal(left.probabilities, right.probabilities)
        assert torch.equal(left.context, right.context)
        assert torch.equal(left.output_write, right.output_write)


def test_copy_exact_tolerance_cannot_be_relaxed_by_the_caller() -> None:
    with pytest.raises(
        Lfm2MatrixAttentionReplayError,
        match="absolute exceeds",
    ):
        Lfm2MatrixAttentionReplayTolerance(
            absolute=MAX_COPY_EXACT_ABSOLUTE_TOLERANCE * 2.0,
        )
    with pytest.raises(
        Lfm2MatrixAttentionReplayError,
        match="relative exceeds",
    ):
        Lfm2MatrixAttentionReplayTolerance(
            relative=MAX_COPY_EXACT_RELATIVE_TOLERANCE * 2.0,
        )


def test_copy_exact_policy_and_endpoint_checks_are_sealed() -> None:
    phases, reference, candidate = _fixture()
    default = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=candidate,
        cell="C10",
    )
    strict = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=candidate,
        cell="C10",
        tolerance=Lfm2MatrixAttentionReplayTolerance(
            absolute=1e-12,
            relative=1e-12,
        ),
    )

    assert default.copy_exact_tolerance.absolute == pytest.approx(
        MAX_COPY_EXACT_ABSOLUTE_TOLERANCE
    )
    assert default.copy_exact_tolerance.relative == pytest.approx(
        MAX_COPY_EXACT_RELATIVE_TOLERANCE
    )
    assert len(default.copy_exact_endpoint_checks) == len(phases) * 15
    assert all(check.passed for check in default.copy_exact_endpoint_checks)
    assert {check.endpoint for check in default.copy_exact_endpoint_checks} >= {
        "query_linear",
        "context_from_probabilities",
        "residual_post_operator",
    }
    assert default.replay_sha256 != strict.replay_sha256


def test_replay_seal_covers_candidate_only_attribution_paths() -> None:
    phases, reference, candidate = _fixture()
    baseline = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=candidate,
        cell="C06",
    )

    def zero_output(value: Tensor) -> Tensor:
        return torch.zeros_like(value)

    changed = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=replace(candidate, output_projection=zero_output),
        cell="C06",
    )

    assert torch.equal(baseline.phases[0].output_write, changed.phases[0].output_write)
    assert not torch.equal(
        baseline.phases[0].attribution.candidate_o_candidate_context,
        changed.phases[0].attribution.candidate_o_candidate_context,
    )
    assert baseline.replay_sha256 != changed.replay_sha256


def test_parameter_counts_are_explicitly_declarations_not_compaction_proof() -> None:
    phases, reference, candidate = _fixture()
    result = evaluate_lfm2_matrix_attention_replay(
        phases=phases,
        reference=reference,
        candidate=replace(
            candidate,
            declared_logical_stored_parameter_count=0,
            declared_physical_materialized_parameter_count=0,
        ),
        cell="C10",
    )

    assert result.declared_logical_stored_parameter_count == 0
    assert result.declared_physical_materialized_parameter_count == 0
    assert result.parameter_accounting_authority is False
    assert result.parameter_accounting_evidence.endswith("unverified_by_evaluator")


def test_zero_reference_nonzero_candidate_error_has_stable_seal() -> None:
    phases, reference, candidate = _fixture()

    def zero_output(value: Tensor) -> Tensor:
        return torch.zeros_like(value)

    zero_output_phases = tuple(
        replace(
            phase,
            output_write=torch.zeros_like(phase.output_write),
            residual_post_operator=phase.residual_pre.clone(),
        )
        for phase in phases
    )
    zero_output_reference = replace(reference, output_projection=zero_output)
    first = evaluate_lfm2_matrix_attention_replay(
        phases=zero_output_phases,
        reference=zero_output_reference,
        candidate=candidate,
        cell="C09",
    )
    second = evaluate_lfm2_matrix_attention_replay(
        phases=zero_output_phases,
        reference=zero_output_reference,
        candidate=candidate,
        cell="C09",
    )

    assert math.isinf(first.phase_metrics[0].output_write.aggregate_relative_l2)
    assert first.replay_sha256 == second.replay_sha256
