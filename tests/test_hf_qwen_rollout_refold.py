from __future__ import annotations

from neural_forge.hf_qwen_rollout_refold import (
    _attempt_dual_gated_candidates,
    _build_mixed_trajectory_state,
)


class TinyDeterministicModel:
    def __init__(self, value: float = 0.0):
        self.value = float(value)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __len__(self):
        return 32


def _score(
    top1: float,
    logprob: float,
    *,
    repetition: float = 0.0,
    longest_run: int = 1,
    actions: int = 4,
):
    return {
        "parent_top1_agreement": float(top1),
        "mean_parent_action_logprob": float(logprob),
        "generated_repetition_rate": float(repetition),
        "generated_longest_run": int(longest_run),
        "generated_action_count": int(actions),
        "hashes": {"report_sha256": f"score-{top1}-{logprob}"},
    }


def _rollout(role: str, suffix_token: int):
    token_ids = [1, int(suffix_token)]
    record = {
        "rollout_index": 0,
        "source_index": 0,
        "token_ids": token_ids,
        "prompt_length": 1,
    }
    return {
        "prompts": ["prompt"],
        "token_id_sequences": [token_ids],
        "rollout_records": [record],
        "provenance": {"role": role},
    }


def _child_state(score, suffix_token: int = 3):
    return {
        "score": score,
        "rollout": _rollout("current-child", suffix_token),
    }


def test_self_action_gain_with_anchor_regression_is_rejected_and_restored():
    model = TinyDeterministicModel()
    baseline = model.value
    restore_calls = []
    incumbent_child = _child_state(_score(0.30, -3.0))
    incumbent_anchor = _score(0.60, -1.0)

    def apply_alpha(alpha):
        model.value = float(alpha)

    def restore():
        restore_calls.append(model.value)
        model.value = baseline

    def build_child_state():
        assert model.value == 1.0
        return _child_state(_score(0.45, -2.0), suffix_token=4)

    def score_fixed_anchors():
        assert model.value == 1.0
        return _score(0.50, -1.2)

    result = _attempt_dual_gated_candidates(
        alphas=(1.0,),
        apply_alpha=apply_alpha,
        restore=restore,
        build_child_state=build_child_state,
        score_fixed_anchors=score_fixed_anchors,
        incumbent_child_state=incumbent_child,
        incumbent_anchor_score=incumbent_anchor,
        min_top1_gain=0.0,
        min_action_logprob_gain=0.0,
        max_repetition_regression=0.0,
        max_longest_run_regression=0,
        min_action_count_ratio=1.0,
    )

    decision = result["attempts"][0]["dual_decision"]
    assert decision["regenerated_child_rollout_gate"]["accepted"] is True
    assert decision["fixed_parent_anchor_gate"]["accepted"] is False
    assert result["accepted"] is False
    assert result["restored"] is True
    assert restore_calls == [1.0]
    assert model.value == baseline


def test_mixed_trajectory_split_keeps_both_roles_in_one_prompt_partition():
    import pytest

    torch = pytest.importorskip("torch")
    child_rollout = {
        "prompts": ["prompt-0", "prompt-1"],
        "token_id_sequences": [[1, 10], [2, 11]],
        "rollout_records": [
            {
                "rollout_index": 0,
                "source_index": 0,
                "token_ids": [1, 10],
                "prompt_length": 1,
            },
            {
                "rollout_index": 1,
                "source_index": 1,
                "token_ids": [2, 11],
                "prompt_length": 1,
            },
        ],
        "provenance": {"role": "current-child"},
    }
    anchor_rollout = {
        "prompts": ["prompt-0", "prompt-1"],
        "token_id_sequences": [[1, 20], [2, 21]],
        "rollout_records": [
            {
                "rollout_index": 0,
                "source_index": 0,
                "token_ids": [1, 20],
                "prompt_length": 1,
            },
            {
                "rollout_index": 1,
                "source_index": 1,
                "token_ids": [2, 21],
                "prompt_length": 1,
            },
        ],
        "provenance": {"role": "fixed-parent-anchor"},
    }
    mixed = _build_mixed_trajectory_state(
        torch,
        {"rollout": child_rollout},
        {"rollout": anchor_rollout},
        TinyTokenizer(),
        {"train_indices": [0], "validation_indices": [1], "seed": 7},
        batch_size=2,
        max_length=8,
        device="cpu",
    )

    split = mixed["provenance"]["prompt_group_split"]
    assert split["group_to_trajectory_indices"] == [[0, 1], [2, 3]]
    assert mixed["train_indices"] == [0, 1]
    assert mixed["validation_indices"] == [2, 3]
    assert split["no_prompt_leakage"] is True
    assert mixed["trajectory_group_ids"] == [0, 0, 1, 1]
    assert sum(int(batch["trace_mask"].sum()) for batch in mixed["batches"]) == 4


def test_candidate_is_retained_only_after_both_global_gates_pass():
    model = TinyDeterministicModel()
    incumbent_child = _child_state(_score(0.30, -3.0))
    incumbent_anchor = _score(0.50, -1.0)
    restore_calls = []

    def apply_alpha(alpha):
        model.value = float(alpha)

    def restore():
        restore_calls.append(model.value)
        model.value = 0.0

    def build_child_state():
        if model.value == 1.0:
            return _child_state(_score(0.20, -3.2), suffix_token=5)
        return _child_state(_score(0.40, -2.5), suffix_token=6)

    def score_fixed_anchors():
        if model.value == 1.0:
            return _score(0.60, -0.8)
        return _score(0.55, -0.9)

    result = _attempt_dual_gated_candidates(
        alphas=(1.0, 0.5),
        apply_alpha=apply_alpha,
        restore=restore,
        build_child_state=build_child_state,
        score_fixed_anchors=score_fixed_anchors,
        incumbent_child_state=incumbent_child,
        incumbent_anchor_score=incumbent_anchor,
        min_top1_gain=0.0,
        min_action_logprob_gain=0.0,
        max_repetition_regression=0.0,
        max_longest_run_regression=0,
        min_action_count_ratio=1.0,
    )

    first = result["attempts"][0]["dual_decision"]
    accepted = result["attempts"][1]["dual_decision"]
    assert first["fixed_parent_anchor_gate"]["accepted"] is True
    assert first["regenerated_child_rollout_gate"]["accepted"] is False
    assert first["accepted"] is False
    assert accepted["fixed_parent_anchor_gate"]["accepted"] is True
    assert accepted["regenerated_child_rollout_gate"]["accepted"] is True
    assert accepted["accepted"] is True
    assert result["accepted"] is True
    assert result["accepted_alpha"] == 0.5
    assert result["attempts"][1]["post_commit_anchor_recheck"]["accepted"] is True
    assert restore_calls == []
    assert model.value == 0.5
