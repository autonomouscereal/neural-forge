from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import pytest


from neural_forge.hf_residual_attention_functional import (
    ResidualAttentionFunctionalError,
    build_residual_attention_functional_comparison,
    functional_ranking_key,
    rank_functional_reports,
    score_residual_attention_function,
)


@dataclass(frozen=True)
class _Step:
    step_index: int
    selected_token_id: int
    action_logits: object


@dataclass(frozen=True)
class _Trace:
    sequence_id: str
    lane: str
    prompt_token_ids: tuple[int, ...]
    consumed_token_ids: tuple[int, ...]
    selected_token_ids: tuple[int, ...]
    steps: tuple[_Step, ...]


def _row(top: int, *, vocab: int = 5, second: int | None = None) -> np.ndarray:
    value = np.linspace(-1.0, -2.0, vocab, dtype=np.float64)
    value[top] = 5.0
    if second is not None:
        value[second] = 4.0
    return value


def _trace(
    lane: str,
    sequence_id: str,
    selected: tuple[int, ...],
    rows: tuple[object, ...],
    *,
    prompt: tuple[int, ...] = (0, 1),
) -> _Trace:
    return _Trace(
        sequence_id=sequence_id,
        lane=lane,
        prompt_token_ids=prompt,
        consumed_token_ids=selected[:-1],
        selected_token_ids=selected,
        steps=tuple(
            _Step(index, token_id, row)
            for index, (token_id, row) in enumerate(zip(selected, rows, strict=True))
        ),
    )


def _base_traces():
    parent_forced = _trace(
        "forced",
        "prompt-0",
        (1, 2, 3),
        (_row(1), _row(2), _row(3)),
    )
    candidate_forced = _trace(
        "forced",
        "prompt-0",
        (1, 2, 3),
        (
            _row(1),
            np.array([5.0, 0.0, 4.0, -1.0, -2.0]),
            _row(3),
        ),
    )
    parent_greedy = _trace(
        "greedy",
        "prompt-0",
        (1, 2, 3, 4),
        (_row(1), _row(2), _row(3), _row(4)),
    )
    candidate_greedy = _trace(
        "greedy",
        "prompt-0",
        (1, 2, 4, 4),
        (_row(1), _row(2), _row(4), _row(4)),
    )
    parent = {"forced-site-a": parent_forced, "greedy-site-a": parent_greedy}
    candidate = {
        "forced-site-a": candidate_forced,
        "greedy-site-a": candidate_greedy,
    }
    return parent, candidate


def test_behavior_first_metrics_remain_valid_after_greedy_divergence() -> None:
    parent, candidate = _base_traces()

    report = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="candidate-a",
        split="calibration-selection",
        top_k=2,
    )

    assert report.forced.action_count == 3
    assert report.forced.parent_driver_top1_rate == pytest.approx(1.0)
    assert report.forced.candidate_parent_top1_agreement_count == 2
    assert report.forced.candidate_parent_top1_agreement_rate == pytest.approx(2 / 3)
    assert report.forced.candidate_parent_action_topk_rate == pytest.approx(1.0)
    assert report.forced.candidate_parent_action_mean_rank == pytest.approx(4 / 3)
    expected_logprobs = []
    for token, row in zip(
        (1, 2, 3),
        candidate["forced-site-a"].steps,
        strict=True,
    ):
        logits = np.asarray(row.action_logits)
        expected_logprobs.append(
            logits[token] - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        )
    assert report.forced.candidate_parent_action_mean_logprob == pytest.approx(
        np.mean(expected_logprobs)
    )

    assert report.greedy.positional_match_count == 3
    assert report.greedy.exact_token_retention_rate == pytest.approx(0.75)
    assert report.greedy.longest_common_prefix_tokens == 2
    assert report.greedy.longest_common_prefix_rate == pytest.approx(0.5)
    assert report.greedy.earliest_divergence_step == 2
    assert report.greedy.exact_rollout_match_rate == pytest.approx(0.0)
    assert report.greedy.candidate_mean_repeat_fraction == pytest.approx(0.25)
    assert report.greedy.candidate_mean_adjacent_repeat_fraction == pytest.approx(1 / 3)
    assert report.greedy.candidate_longest_run == 2
    assert report.selection_allowed is True
    assert report.heldout_evaluation is False
    payload = report.as_dict()
    assert payload["forced_parent_anchor_primary"] is True
    assert payload["valid_after_greedy_prefix_divergence"] is True
    assert payload["ranking_uses_output_function_only"] is True
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "attention_js",
        "cache_transition_error",
        "residual_transition_error",
        "qk_score",
    ):
        assert forbidden not in serialized


def test_duplicate_layer_site_traces_are_deduplicated_not_overweighted() -> None:
    parent, candidate = _base_traces()
    parent["forced-site-b"] = parent["forced-site-a"]
    parent["greedy-site-b"] = parent["greedy-site-a"]
    candidate["forced-site-b"] = candidate["forced-site-a"]
    candidate["greedy-site-b"] = candidate["greedy-site-a"]

    report = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="deduplicated",
        split="calibration-selection",
        top_k=2,
    )

    assert report.forced.sequence_count == 1
    assert report.forced.action_count == 3
    assert report.greedy.sequence_count == 1

    conflicting = dict(candidate)
    conflicting["forced-site-b"] = replace(
        candidate["forced-site-a"],
        steps=(
            replace(
                candidate["forced-site-a"].steps[0],
                action_logits=_row(0),
            ),
            *candidate["forced-site-a"].steps[1:],
        ),
    )
    with pytest.raises(ResidualAttentionFunctionalError, match="conflicting duplicate"):
        score_residual_attention_function(
            parent,
            conflicting,
            candidate_id="conflict",
            split="calibration-selection",
        )


def test_fixed_parent_anchor_metrics_precede_child_greedy_gate_in_ranking() -> None:
    parent, candidate = _base_traces()
    strong_forced_bad_greedy = dict(candidate)
    strong_forced_bad_greedy["forced-site-a"] = parent["forced-site-a"]
    strong_forced_bad_greedy["greedy-site-a"] = _trace(
        "greedy",
        "prompt-0",
        (1, 1, 1, 1),
        (_row(1), _row(1), _row(1), _row(1)),
    )
    weak_forced_good_greedy = dict(candidate)
    weak_forced_good_greedy["greedy-site-a"] = parent["greedy-site-a"]

    strong = score_residual_attention_function(
        parent,
        strong_forced_bad_greedy,
        candidate_id="strong-anchor",
        split="calibration-selection",
        top_k=2,
    )
    weak = score_residual_attention_function(
        parent,
        weak_forced_good_greedy,
        candidate_id="weak-anchor",
        split="calibration-selection",
        top_k=2,
    )

    assert strong.forced.candidate_parent_top1_agreement_rate > (
        weak.forced.candidate_parent_top1_agreement_rate
    )
    assert strong.greedy.candidate_longest_run > weak.greedy.candidate_longest_run
    assert functional_ranking_key(strong) < functional_ranking_key(weak)
    ranking = rank_functional_reports((weak, strong), purpose="selection")
    assert [value.candidate_id for value in ranking.candidates] == [
        "strong-anchor",
        "weak-anchor",
    ]


def test_calibration_selection_and_heldout_evaluation_are_not_interchangeable() -> None:
    parent, candidate = _base_traces()
    calibration_b = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="b",
        split="calibration-selection",
    )
    calibration_a = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="a",
        split="calibration-selection",
    )
    heldout_a = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="a",
        split="heldout-evaluation",
    )
    heldout_b = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="b",
        split="heldout-evaluation",
    )

    selection = rank_functional_reports(
        (calibration_b, calibration_a), purpose="selection"
    )
    assert [value.candidate_id for value in selection.candidates] == ["a", "b"]
    evaluation = rank_functional_reports((heldout_b, heldout_a), purpose="evaluation")
    assert [value.candidate_id for value in evaluation.candidates] == ["a", "b"]
    assert heldout_a.selection_allowed is False
    assert heldout_a.heldout_evaluation is True
    assert (
        "excluded from candidate fitting/selection"
        in heldout_a.as_dict()["split_semantics"]
    )

    with pytest.raises(ResidualAttentionFunctionalError, match="requires heldout"):
        rank_functional_reports((calibration_a,), purpose="evaluation")
    with pytest.raises(ResidualAttentionFunctionalError, match="requires calibration"):
        rank_functional_reports((heldout_a,), purpose="selection")
    with pytest.raises(ResidualAttentionFunctionalError, match="cannot mix"):
        rank_functional_reports((calibration_a, heldout_b), purpose="selection")


def test_calibration_selection_fails_closed_when_driver_is_not_parent_top1() -> None:
    parent, candidate = _base_traces()
    parent["forced-site-a"] = replace(
        parent["forced-site-a"],
        steps=(
            parent["forced-site-a"].steps[0],
            replace(
                parent["forced-site-a"].steps[1],
                action_logits=_row(0),
            ),
            parent["forced-site-a"].steps[2],
        ),
    )
    calibration = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="invalid-anchor",
        split="calibration-selection",
    )
    assert calibration.forced.parent_driver_top1_rate < 1.0
    assert calibration.fixed_parent_top1_anchor_valid is False
    assert calibration.selection_allowed is False
    assert calibration.as_dict()["fixed_parent_top1_anchor_valid"] is False
    with pytest.raises(
        ResidualAttentionFunctionalError,
        match="parent_driver_top1_rate == 1.0",
    ):
        rank_functional_reports((calibration,), purpose="selection")

    heldout = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="invalid-anchor",
        split="heldout-evaluation",
    )
    assert heldout.as_dict()["fixed_parent_top1_anchor_valid"] is False
    assert rank_functional_reports(
        (heldout,), purpose="evaluation"
    ).candidates[0].candidate_id == "invalid-anchor"


def test_functional_comparison_selects_from_calibration_and_only_orders_heldout() -> (
    None
):
    parent, weak = _base_traces()
    perfect = {
        "forced-site-a": parent["forced-site-a"],
        "greedy-site-a": parent["greedy-site-a"],
    }
    comparison = build_residual_attention_functional_comparison(
        parent,
        parent,
        {"calibration-winner": perfect, "heldout-winner": weak},
        {"calibration-winner": weak, "heldout-winner": perfect},
        top_k=2,
    )

    assert (
        comparison["calibration_selection"]["winner_candidate_id"]
        == "calibration-winner"
    )
    assert comparison["calibration_selection"]["heldout_used_for_selection"] is False
    assert comparison["heldout_evaluation"]["winner_named"] is False
    assert comparison["heldout_evaluation"]["used_for_selection"] is False
    assert [
        item["candidate_id"]
        for item in comparison["heldout_evaluation"]["ordering"]["candidates"]
    ] == ["heldout-winner", "calibration-winner"]


def test_single_split_exploration_cannot_be_ranked_or_select_a_winner() -> None:
    parent, candidate = _base_traces()
    exploratory = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="exploratory",
        split="calibration-exploratory",
    )
    assert exploratory.calibration_exploratory is True
    assert exploratory.selection_allowed is False
    assert "no disjoint heldout split" in exploratory.as_dict()["split_semantics"]
    with pytest.raises(ResidualAttentionFunctionalError, match="requires calibration"):
        rank_functional_reports((exploratory,), purpose="selection")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("nonfinite", "NaN or infinity"),
        ("integer", "floating-point dtype"),
        ("rank3", "must have shape"),
        ("vocab", "output vocabulary width"),
        ("token", "outside its output vocabulary"),
    ),
)
def test_strict_logit_shape_finite_and_vocab_contracts(mutation, message) -> None:
    parent, candidate = _base_traces()
    if mutation == "nonfinite":
        row = _row(1)
        row[0] = np.nan
        candidate["forced-site-a"] = replace(
            candidate["forced-site-a"],
            steps=(
                replace(candidate["forced-site-a"].steps[0], action_logits=row),
                *candidate["forced-site-a"].steps[1:],
            ),
        )
    elif mutation == "integer":
        candidate["forced-site-a"] = replace(
            candidate["forced-site-a"],
            steps=(
                replace(
                    candidate["forced-site-a"].steps[0],
                    action_logits=np.arange(5, dtype=np.int64),
                ),
                *candidate["forced-site-a"].steps[1:],
            ),
        )
    elif mutation == "rank3":
        candidate["forced-site-a"] = replace(
            candidate["forced-site-a"],
            steps=(
                replace(
                    candidate["forced-site-a"].steps[0],
                    action_logits=np.zeros((1, 1, 5)),
                ),
                *candidate["forced-site-a"].steps[1:],
            ),
        )
    elif mutation == "vocab":
        candidate["forced-site-a"] = replace(
            candidate["forced-site-a"],
            steps=tuple(
                replace(step, action_logits=_row(step.selected_token_id, vocab=6))
                for step in candidate["forced-site-a"].steps
            ),
        )
    else:
        invalid = _trace(
            "forced",
            "prompt-0",
            (1, 2, 7),
            (_row(1), _row(2), _row(3)),
        )
        candidate["forced-site-a"] = invalid

    with pytest.raises(ResidualAttentionFunctionalError, match=message):
        score_residual_attention_function(
            parent,
            candidate,
            candidate_id="invalid",
            split="calibration-selection",
        )


def test_forced_alignment_is_strict_but_greedy_parent_candidate_paths_may_differ() -> (
    None
):
    parent, candidate = _base_traces()
    bad_forced = replace(
        candidate["forced-site-a"],
        selected_token_ids=(1, 4, 3),
        consumed_token_ids=(1, 4),
        steps=(
            candidate["forced-site-a"].steps[0],
            replace(candidate["forced-site-a"].steps[1], selected_token_id=4),
            candidate["forced-site-a"].steps[2],
        ),
    )
    candidate["forced-site-a"] = bad_forced
    with pytest.raises(ResidualAttentionFunctionalError, match="parent-driven"):
        score_residual_attention_function(
            parent,
            candidate,
            candidate_id="misaligned",
            split="calibration-selection",
        )

    parent, candidate = _base_traces()
    report = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="diverged-is-valid",
        split="heldout-evaluation",
    )
    assert report.greedy.earliest_divergence_step == 2


def test_greedy_trace_must_contain_real_argmax_actions() -> None:
    parent, candidate = _base_traces()
    candidate["greedy-site-a"] = replace(
        candidate["greedy-site-a"],
        selected_token_ids=(1, 2, 3, 4),
        consumed_token_ids=(1, 2, 3),
        steps=(
            candidate["greedy-site-a"].steps[0],
            candidate["greedy-site-a"].steps[1],
            replace(candidate["greedy-site-a"].steps[2], selected_token_id=3),
            candidate["greedy-site-a"].steps[3],
        ),
    )
    with pytest.raises(ResidualAttentionFunctionalError, match="true greedy action"):
        score_residual_attention_function(
            parent,
            candidate,
            candidate_id="not-greedy",
            split="heldout-evaluation",
        )


def test_report_and_ranking_hashes_are_deterministic() -> None:
    parent, candidate = _base_traces()
    first = score_residual_attention_function(
        parent,
        candidate,
        candidate_id="stable",
        split="calibration-selection",
    )
    second = score_residual_attention_function(
        dict(reversed(tuple(parent.items()))),
        dict(reversed(tuple(candidate.items()))),
        candidate_id="stable",
        split="calibration-selection",
    )

    assert first.report_sha256 == second.report_sha256
    assert first.as_dict() == second.as_dict()
    first_ranking = rank_functional_reports((first,), purpose="selection")
    second_ranking = rank_functional_reports((second,), purpose="selection")
    assert first_ranking.ranking_sha256 == second_ranking.ranking_sha256
