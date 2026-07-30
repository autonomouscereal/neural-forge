from __future__ import annotations

import torch

from experiments import prepare_lfm2_a12_carrier_reclosure as prepare


def test_presence_counts_are_per_prompt_not_raw_frequency() -> None:
    audit = {
        "prompt_results": [
            {
                "prompt_id": "fit-a",
                "parent_reference_tokens": [
                    {"next_forced_token_id": 3},
                    {"next_forced_token_id": 3},
                    {"next_forced_token_id": None},
                ],
                "lanes": {
                    "exact_exact": {
                        "tokens": [
                            {"parent_winner_token_id": 4},
                            {"parent_winner_token_id": 4},
                            {"parent_winner_token_id": 4},
                        ]
                    }
                },
            },
            {
                "prompt_id": "fit-b",
                "parent_reference_tokens": [
                    {"next_forced_token_id": 3},
                    {"next_forced_token_id": None},
                ],
                "lanes": {
                    "exact_exact": {
                        "tokens": [
                            {"parent_winner_token_id": 4},
                            {"parent_winner_token_id": 5},
                        ]
                    }
                },
            },
        ]
    }
    manifest = {
        "fit_prompt_ids": ["fit-a", "fit-b"],
        "prompt_token_ids_by_prompt": {
            "fit-a": [1, 1, 1, 2],
            "fit-b": [1, 3],
        },
    }
    occurrence, winners, diagnostics = prepare._presence_counts(
        audit=audit,
        prompt_manifest=manifest,
        donor_count=8,
    )
    assert occurrence.tolist() == [0, 2, 1, 2, 0, 0, 0, 0]
    assert winners.tolist() == [0, 0, 0, 0, 2, 1, 0, 0]
    assert diagnostics["raw_occurrence_frequency_used"] is False
    assert diagnostics["aggregation"].startswith("maximum-one-count")


def test_demand_diagnostics_use_independent_input_and_output_views() -> None:
    parent = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    target = parent[:2].clone()
    offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    ids = torch.tensor([0, 1, 0], dtype=torch.int64)
    input_coefficients = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    output_coefficients = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
    occurrence = torch.tensor([0, 0, 1], dtype=torch.int64)
    winners = torch.tensor([0, 0, 1], dtype=torch.int64)
    errors, costs = prepare._demand_diagnostics(
        occurrence=occurrence,
        winners=winners,
        parent_table=parent,
        target_table=target,
        input_transform=torch.eye(2),
        output_transform=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        input_offsets=offsets,
        input_target_ids=ids,
        input_coefficients=input_coefficients,
        output_offsets=offsets,
        output_target_ids=ids,
        output_coefficients=output_coefficients,
        target_to_donor=torch.tensor([0, 1], dtype=torch.int64),
    )
    assert errors[2] > 0.0
    assert errors[:2].tolist() == [0.0, 0.0]
    assert costs[0] == 1.0
    assert costs[1] == 1.0
    assert costs[2] == 0.0


def test_relative_feature_error_is_zero_for_behaviorally_identical_rows() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.float64)
    transform = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    assert prepare._relative_feature_error(value, value.clone(), transform) == 0.0
