from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from neural_forge.hf_rollout_gate import (
    compare_rollout_gate_reports,
    evaluate_parent_rollouts,
    hash_rollout_records,
    normalize_rollout_records,
)


def _tiny_parent(torch):
    class TinyParent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []
            self.config = SimpleNamespace(vocab_size=8, pad_token_id=0)

        def forward(self, input_ids, attention_mask):
            self.calls.append(
                {
                    "input_ids": input_ids.detach().cpu().clone(),
                    "attention_mask": attention_mask.detach().cpu().clone(),
                    "training": self.training,
                    "grad_enabled": torch.is_grad_enabled(),
                    "inference_enabled": torch.is_inference_mode_enabled(),
                }
            )
            logits = torch.zeros(
                (*input_ids.shape, self.config.vocab_size),
                dtype=torch.float32,
                device=input_ids.device,
            )
            predicted = (input_ids + 1) % self.config.vocab_size
            logits.scatter_(2, predicted.unsqueeze(-1), 3.0)
            logits[..., 3] = torch.where(
                input_ids == 3,
                torch.tensor(2.0, device=input_ids.device),
                logits[..., 3],
            )
            return SimpleNamespace(logits=logits)

    return TinyParent()


def test_parent_gate_scores_exact_generated_actions_with_padding_and_no_gradients() -> None:
    torch = pytest.importorskip("torch")
    parent = _tiny_parent(torch)
    parent.train()
    parent.anchor.grad = torch.tensor(7.0)
    gradient_before = parent.anchor.grad
    rollouts = [
        {"token_ids": [1, 2, 3, 3], "prompt_length": 2},
        {"full_token_ids": [4, 5, 6], "prompt_length": 1},
    ]

    report = evaluate_parent_rollouts(
        parent,
        rollouts,
        batch_size=2,
        top_k=2,
        padding_side="right",
    )

    matching_logprob = 3.0 - math.log(math.exp(3.0) + 7.0)
    rank_two_logprob = 2.0 - math.log(math.exp(3.0) + math.exp(2.0) + 6.0)
    expected_logprob = (3.0 * matching_logprob + rank_two_logprob) / 4.0
    assert report["parent_top1_agreement"] == pytest.approx(0.75)
    assert report["mean_parent_action_logprob"] == pytest.approx(expected_logprob)
    assert report["mean_parent_action_rank"] == pytest.approx(1.25)
    assert report["parent_action_top_k_coverage"] == pytest.approx(1.0)
    assert report["generated_repetition_rate"] == pytest.approx(0.25)
    assert report["generated_unique_token_ratio"] == pytest.approx(0.75)
    assert report["generated_longest_run"] == 2
    assert report["prompt_count"] == 2
    assert report["prompt_token_count"] == 3
    assert report["generated_token_count"] == 4
    assert report["total_token_count"] == 7
    assert report["zero_gradient"] is True
    assert parent.training is True
    assert parent.anchor.grad is gradient_before
    assert torch.equal(parent.anchor.grad, torch.tensor(7.0))
    assert parent.calls[0]["training"] is False
    assert parent.calls[0]["grad_enabled"] is False
    assert parent.calls[0]["inference_enabled"] is True
    assert parent.calls[0]["input_ids"].tolist() == [[1, 2, 3, 3], [4, 5, 6, 0]]
    assert parent.calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_aggregate_rollout_result_and_left_padding_are_deterministic() -> None:
    torch = pytest.importorskip("torch")
    aggregate = {
        "token_id_sequences": [[1, 2, 3, 3], [4, 5, 6]],
        "prompt_token_id_sequences": [[1, 2], [4]],
        "generated_token_id_sequences": [[3, 3], [5, 6]],
        "provenance": {
            "sequences": [
                {"prompt_token_count": 2},
                {"prompt_token_count": 1},
            ]
        },
    }
    expected_records = [
        {"token_ids": [1, 2, 3, 3], "prompt_length": 2},
        {"token_ids": [4, 5, 6], "prompt_length": 1},
    ]

    assert normalize_rollout_records(aggregate) == expected_records
    assert hash_rollout_records(aggregate) == hash_rollout_records(expected_records)
    aggregate["rollout_records"] = [
        {**record, "rollout_id": f"rollout-{index}"}
        for index, record in enumerate(expected_records)
    ]
    assert normalize_rollout_records(aggregate) == expected_records

    first_parent = _tiny_parent(torch)
    second_parent = _tiny_parent(torch)
    first = evaluate_parent_rollouts(
        first_parent, aggregate, batch_size=2, top_k=2, padding_side="left"
    )
    second = evaluate_parent_rollouts(
        second_parent, aggregate, batch_size=2, top_k=2, padding_side="left"
    )

    assert first["metrics"] == second["metrics"]
    assert first["hashes"] == second["hashes"]
    assert first_parent.calls[0]["input_ids"].tolist() == [
        [1, 2, 3, 3],
        [0, 4, 5, 6],
    ]
    assert first_parent.calls[0]["attention_mask"].tolist() == [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
    ]


def test_record_attention_mask_removes_only_edge_padding() -> None:
    masked = [
        {
            "input_ids": [0, 0, 4, 5, 6],
            "attention_mask": [0, 0, 1, 1, 1],
            "prompt_length": 1,
        }
    ]
    assert normalize_rollout_records(masked) == [
        {"token_ids": [4, 5, 6], "prompt_length": 1}
    ]

    with pytest.raises(ValueError, match="padding only before or after"):
        normalize_rollout_records(
            [
                {
                    "input_ids": [4, 0, 6],
                    "attention_mask": [1, 0, 1],
                    "prompt_length": 1,
                }
            ]
        )


def test_non_finite_parent_logits_fail_and_restore_training_state() -> None:
    torch = pytest.importorskip("torch")

    class NonFiniteParent(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            assert self.training is False
            assert torch.is_grad_enabled() is False
            return SimpleNamespace(
                logits=torch.full(
                    (*input_ids.shape, 4),
                    float("nan"),
                    device=input_ids.device,
                )
            )

    parent = NonFiniteParent()
    parent.train()
    with pytest.raises(ValueError, match="non-finite logits"):
        evaluate_parent_rollouts(
            parent,
            [{"token_ids": [1, 2], "prompt_length": 1}],
        )
    assert parent.training is True


def _gate_metrics(
    *,
    top1: float,
    logprob: float,
    repetition: float = 0.1,
    longest_run: int = 2,
) -> dict[str, float | int]:
    return {
        "parent_top1_agreement": top1,
        "mean_parent_action_logprob": logprob,
        "generated_repetition_rate": repetition,
        "generated_longest_run": longest_run,
    }


def test_promotion_is_top1_first_then_logprob_with_degeneration_vetoes() -> None:
    incumbent = _gate_metrics(top1=0.5, logprob=-2.0)

    top1_winner = compare_rollout_gate_reports(
        incumbent,
        _gate_metrics(top1=0.6, logprob=-3.0),
    )
    assert top1_winner["accepted"] is True
    assert top1_winner["reverted"] is False
    assert top1_winner["accepted_reason"] == "parent_top1_agreement_improved"

    logprob_winner = compare_rollout_gate_reports(
        incumbent,
        _gate_metrics(top1=0.5, logprob=-1.9),
        min_action_logprob_gain=0.05,
    )
    assert logprob_winner["accepted"] is True
    assert (
        logprob_winner["reason"]
        == "parent_action_logprob_improved_after_top1_tie"
    )

    top1_loser = compare_rollout_gate_reports(
        incumbent,
        _gate_metrics(top1=0.49, logprob=-1.0),
    )
    assert top1_loser["accepted"] is False
    assert top1_loser["reverted_reason"] == "parent_top1_agreement_regressed"

    repeated = compare_rollout_gate_reports(
        incumbent,
        _gate_metrics(top1=0.7, logprob=-1.0, repetition=0.16),
        max_repetition_regression=0.05,
    )
    assert repeated["accepted"] is False
    assert repeated["reason"] == "generated_repetition_regression_above_threshold"
    assert repeated["thresholds"] == {
        "min_parent_top1_agreement_gain": 0.0,
        "min_parent_action_logprob_gain": 0.0,
        "max_generated_repetition_rate_regression": 0.05,
        "max_generated_longest_run_regression": 0,
    }

    long_run = compare_rollout_gate_reports(
        incumbent,
        _gate_metrics(top1=0.7, logprob=-1.0, longest_run=4),
        max_longest_run_regression=1,
    )
    assert long_run["accepted"] is False
    assert (
        long_run["reason"]
        == "generated_longest_run_regression_above_threshold"
    )
