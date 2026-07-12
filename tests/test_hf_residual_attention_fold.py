from __future__ import annotations

import numpy as np
import pytest

from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionCandidateEvidence,
    ResidualAttentionFoldError,
    ResidualAttentionGatePolicy,
    ResidualAttentionRollout,
    ResidualAttentionRolloutPair,
    ResidualAttentionScoringConfig,
    ResidualAttentionStepEvidence,
    fit_fixed_attention_output_ridge,
    fit_residual_attention_branch_scale,
    gate_residual_attention_candidate,
    rank_residual_attention_candidates,
    score_residual_attention_candidate,
    score_residual_attention_interaction,
)


def _step(
    step_index: int,
    *,
    attention: tuple[float, ...] | None = None,
    cache_append: tuple[float, float] | None = None,
    delta_scale: float = 1.0,
    logits: tuple[float, ...] = (4.0, 2.0, 1.0, 0.0),
    attention_score_shift: float = 0.0,
) -> ResidualAttentionStepEvidence:
    length = step_index + 1
    pre = np.array([[1.0 + step_index, -0.5]], dtype=np.float64)
    delta = np.array([[0.2, -0.1]], dtype=np.float64) * delta_scale
    query = np.array([[0.5 + step_index, 0.25]], dtype=np.float64)
    key = np.array([[0.1 + step_index, -0.2]], dtype=np.float64)
    value = np.array([[0.3, 0.4 + step_index]], dtype=np.float64)
    context = np.array([[0.7, -0.1]], dtype=np.float64) * delta_scale
    if attention is None:
        probabilities = np.full(length, 1.0 / length, dtype=np.float64)
    else:
        probabilities = np.array(attention, dtype=np.float64)
    cache = np.zeros((1, 1, length, 2), dtype=np.float64)
    for index in range(length):
        cache[0, 0, index] = (index + 0.1, index + 0.2)
    if cache_append is not None:
        cache[0, 0, -1] = cache_append
    return ResidualAttentionStepEvidence(
        step_index=step_index,
        pre_attention_residual=pre,
        query=query,
        key=key,
        value=value,
        attention_scores=np.log(
            np.maximum(probabilities, np.finfo(np.float64).tiny)
        ).reshape(1, 1, -1)
        + attention_score_shift,
        attention_probabilities=probabilities.reshape(1, 1, -1),
        fixed_attention_value_readout=context,
        attention_output_input=context,
        attention_delta=delta,
        post_attention_residual=pre + delta,
        key_cache=cache,
        value_cache=cache * 0.5,
        action_logits=np.array(logits, dtype=np.float64).reshape(1, -1),
    )


def _rollout(
    sequence_id: str,
    steps: tuple[ResidualAttentionStepEvidence, ...],
) -> ResidualAttentionRollout:
    return ResidualAttentionRollout(sequence_id, steps)


def _evidence(
    candidate_ids: tuple[str, ...],
    reference_steps: tuple[ResidualAttentionStepEvidence, ...],
    folded_steps: tuple[ResidualAttentionStepEvidence, ...],
) -> ResidualAttentionCandidateEvidence:
    return ResidualAttentionCandidateEvidence(
        candidate_ids,
        (
            ResidualAttentionRolloutPair(
                _rollout("prompt-1", reference_steps),
                _rollout("prompt-1", folded_steps),
            ),
        ),
    )


def test_identical_atomic_attention_rollouts_have_zero_damage_and_pass() -> None:
    reference = (_step(0), _step(1))
    damage = score_residual_attention_candidate(
        _evidence(("layer35-group0",), reference, reference)
    )

    assert damage.risk_score == pytest.approx(0.0)
    assert damage.metrics["qkvo_bundle_error"].worst == pytest.approx(0.0)
    assert damage.metrics["attention_js_divergence"].worst == pytest.approx(0.0)
    assert damage.metrics["cache_transition_error"].worst == pytest.approx(0.0)
    assert damage.metrics["action_margin_damage"].worst == pytest.approx(0.0)
    assert damage.as_dict()["qkvo_residual_cache_atomic"] is True
    assert damage.as_dict()["gradient_free"] is True
    assert damage.as_dict()["cache_closed_loop"] is True
    assert damage.as_dict()["autoregressive_closed_loop"] is False
    assert damage.as_dict()["token_lane"] == "unspecified"

    decision = gate_residual_attention_candidate(
        damage,
        ResidualAttentionGatePolicy(
            quantile_limits={"atomic_attention_damage": 0.0},
            worst_limits={"atomic_attention_damage": 0.0},
            max_risk_score=0.0,
        ),
    )
    assert decision.accepted
    assert decision.reasons == ()


def test_gate_without_any_declared_limit_fails_closed() -> None:
    with pytest.raises(ResidualAttentionFoldError, match="must declare"):
        ResidualAttentionGatePolicy()


def test_late_attention_and_cache_divergence_defeats_accurate_first_decode() -> None:
    reference = (
        _step(0),
        _step(1, attention=(0.8, 0.2)),
    )
    folded = (
        _step(0),
        _step(
            1,
            attention=(0.1, 0.9),
            cache_append=(20.0, -20.0),
            delta_scale=-1.0,
            logits=(0.0, 5.0, 1.0, 0.0),
        ),
    )
    damage = score_residual_attention_candidate(
        _evidence(("fold-A",), reference, folded),
        ResidualAttentionScoringConfig(quantile=0.5, top_k=2),
    )

    for metric in (
        "attention_js_divergence",
        "residual_transition_error",
        "cache_transition_error",
        "action_margin_damage",
        "action_mismatch",
    ):
        assert damage.metrics[metric].first_step_worst == pytest.approx(0.0)
        assert damage.metrics[metric].final_step_worst > 0.0
    assert damage.metrics["attention_js_divergence"].worst > 0.3
    assert damage.metrics["cache_transition_error"].worst > 0.5
    assert damage.metrics["action_mismatch"].worst == pytest.approx(1.0)

    decision = gate_residual_attention_candidate(
        damage,
        ResidualAttentionGatePolicy(
            quantile_limits={
                "attention_js_divergence": 0.2,
                "cache_transition_error": 0.2,
            },
            worst_limits={
                "action_mismatch": 0.0,
                "residual_transition_error": 0.2,
            },
        ),
    )
    assert not decision.accepted
    assert any("attention_js_divergence" in reason for reason in decision.reasons)
    assert any("cache_transition_error" in reason for reason in decision.reasons)
    assert any("action_mismatch" in reason for reason in decision.reasons)


def test_attention_distribution_is_not_hidden_by_matching_qkvo_tensors() -> None:
    reference = (
        _step(0),
        _step(1, attention=(0.99, 0.01)),
    )
    folded = (
        _step(0),
        _step(1, attention=(0.01, 0.99)),
    )
    damage = score_residual_attention_candidate(
        _evidence(("attention-routing-only",), reference, folded)
    )

    for metric in (
        "query_error",
        "key_error",
        "value_error",
        "output_context_error",
        "output_delta_error",
    ):
        assert damage.metrics[metric].worst == pytest.approx(0.0)
    assert damage.metrics["residual_transition_error"].worst == pytest.approx(0.0)
    assert damage.metrics["qkvo_bundle_error"].worst > 0.8
    assert damage.metrics["attention_js_divergence"].worst > 0.8


def test_score_geometry_ignores_softmax_invisible_row_shift() -> None:
    reference = (_step(0), _step(1, attention=(0.8, 0.2)))
    shifted = (
        _step(0, attention_score_shift=17.0),
        _step(1, attention=(0.8, 0.2), attention_score_shift=-23.0),
    )
    damage = score_residual_attention_candidate(
        _evidence(("score-shift",), reference, shifted)
    )

    assert damage.metrics["key_error"].worst == pytest.approx(0.0)
    assert damage.metrics["attention_score_geometry_error"].worst == pytest.approx(
        0.0, abs=1e-12
    )
    assert damage.metrics["attention_js_divergence"].worst == pytest.approx(0.0)


def test_closed_form_o_branch_compensation_is_global_and_gradient_free() -> None:
    reference = (_step(0, delta_scale=2.0), _step(1, delta_scale=2.0))
    folded = (_step(0), _step(1))
    evidence = _evidence(("scaled-O",), reference, folded)

    branch = fit_residual_attention_branch_scale(evidence, ridge=0.0)
    output = fit_fixed_attention_output_ridge(evidence, ridge=1e-8)

    assert branch.scale == pytest.approx(2.0)
    assert branch.fitted_relative_l2 == pytest.approx(0.0, abs=1e-12)
    assert branch.improved
    assert output.output_matrix.shape == (2, 2)
    assert not output.output_matrix.flags.writeable
    assert output.fitted_relative_l2 < output.baseline_relative_l2
    assert output.as_dict()["fixed_reference_attention"] is True
    assert output.as_dict()["qk_edited"] is False


def test_measured_joint_rollout_exposes_destructive_fold_synergy() -> None:
    reference = (
        _step(0),
        _step(1, attention=(0.5, 0.5)),
    )
    singleton_a = (
        _step(0),
        _step(1, attention=(0.55, 0.45)),
    )
    singleton_b = (
        _step(0),
        _step(1, attention=(0.45, 0.55)),
    )
    joint = (
        _step(0),
        _step(1, attention=(0.99, 0.01)),
    )
    config = ResidualAttentionScoringConfig(
        metric_weights={"attention_js_divergence": 1.0},
        worst_weight=1.0,
    )
    damage_a = score_residual_attention_candidate(
        _evidence(("A",), reference, singleton_a), config
    )
    damage_b = score_residual_attention_candidate(
        _evidence(("B",), reference, singleton_b), config
    )
    group = score_residual_attention_candidate(
        _evidence(("A", "B"), reference, joint), config
    )
    interaction = score_residual_attention_interaction(group, (damage_a, damage_b))

    assert interaction.destructive_synergy
    assert interaction.risk_excess > 0.0
    assert interaction.metrics["attention_js_divergence"].worst_excess > 0.0
    assert interaction.as_dict()["measured_jointly"] is True


def test_ranking_prefers_candidates_that_pass_worst_and_quantile_gates() -> None:
    reference = (_step(0), _step(1, attention=(0.5, 0.5)))
    safe = score_residual_attention_candidate(
        _evidence(("safe",), reference, reference)
    )
    medium = score_residual_attention_candidate(
        _evidence(
            ("medium",),
            reference,
            (_step(0), _step(1, attention=(0.65, 0.35))),
        )
    )
    bad = score_residual_attention_candidate(
        _evidence(
            ("bad",),
            reference,
            (_step(0), _step(1, attention=(0.99, 0.01))),
        )
    )
    ranked = rank_residual_attention_candidates(
        (bad, medium, safe),
        ResidualAttentionGatePolicy(
            quantile_limits={"attention_js_divergence": 0.1},
            worst_limits={"attention_js_divergence": 0.1},
        ),
    )

    assert [value.damage.candidate_ids for value in ranked] == [
        ("safe",),
        ("medium",),
        ("bad",),
    ]
    assert ranked[0].decision.accepted
    assert not ranked[-1].decision.accepted


def test_evidence_rejects_broken_residual_identity_and_incompatible_heads() -> None:
    with pytest.raises(ResidualAttentionFoldError, match="must equal"):
        ResidualAttentionStepEvidence(
            step_index=0,
            pre_attention_residual=np.zeros((1, 2)),
            query=np.zeros((1, 2)),
            key=np.zeros((1, 2)),
            value=np.zeros((1, 2)),
            attention_scores=np.zeros((1, 1, 1)),
            attention_probabilities=np.ones((1, 1, 1)),
            fixed_attention_value_readout=np.zeros((1, 2)),
            attention_output_input=np.zeros((1, 2)),
            attention_delta=np.ones((1, 2)),
            post_attention_residual=np.zeros((1, 2)),
            key_cache=np.zeros((1, 1, 1, 2)),
            value_cache=np.zeros((1, 1, 1, 2)),
            action_logits=np.zeros((1, 3)),
        )

    reference = (_step(0), _step(1))
    incompatible = list(reference)
    original = incompatible[1]
    incompatible[1] = ResidualAttentionStepEvidence(
        step_index=1,
        pre_attention_residual=original.pre_attention_residual,
        query=np.zeros((1, 3)),
        key=original.key,
        value=original.value,
        attention_scores=original.attention_scores,
        attention_probabilities=original.attention_probabilities,
        fixed_attention_value_readout=original.fixed_attention_value_readout,
        attention_output_input=original.attention_output_input,
        attention_delta=original.attention_delta,
        post_attention_residual=original.post_attention_residual,
        key_cache=original.key_cache,
        value_cache=original.value_cache,
        action_logits=original.action_logits,
    )
    with pytest.raises(ResidualAttentionFoldError, match="canonicalize folded heads"):
        score_residual_attention_candidate(
            _evidence(("bad-chart",), reference, tuple(incompatible))
        )


def test_tensor_evidence_is_detached_copied_and_read_only() -> None:
    torch = pytest.importorskip("torch")
    source = torch.tensor([[0.5, 0.25]], requires_grad=True)
    base = _step(0)
    step = ResidualAttentionStepEvidence(
        step_index=base.step_index,
        pre_attention_residual=base.pre_attention_residual,
        query=source,
        key=base.key,
        value=base.value,
        attention_scores=base.attention_scores,
        attention_probabilities=base.attention_probabilities,
        fixed_attention_value_readout=base.fixed_attention_value_readout,
        attention_output_input=base.attention_output_input,
        attention_delta=base.attention_delta,
        post_attention_residual=base.post_attention_residual,
        key_cache=base.key_cache,
        value_cache=base.value_cache,
        action_logits=base.action_logits,
    )

    assert isinstance(step.query, np.ndarray)
    assert not step.query.flags.writeable
    assert source.grad is None
    assert step.query.shape == (1, 2)
