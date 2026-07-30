from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments import extract_lfm2_a12_causal_support_features as extractor


ROOT = Path(__file__).resolve().parents[1]
_D = "a" * 64


class _Session:
    evidence_sha256 = _D

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.requested_names: tuple[str, ...] = ()
        self.released = False

    def load_selected_tensors(self, index: int, names: tuple[str, ...]):
        assert index == 0
        self.requested_names = names
        return SimpleNamespace(
            tensors={name: self.tensors[name] for name in names},
            selection_sha256="b" * 64,
        )

    def release_selected_tensors(self, _payload: object) -> None:
        self.released = True


def _prompt_result() -> dict[str, object]:
    return {
        "batch_index": 0,
        "prompt_id": "fit-a",
        "prompt_token_count": 3,
        "forced_token_count": 2,
        "state_count": 3,
        "result_sha256": "c" * 64,
        "parent_trajectory_receipt_sha256": "d" * 64,
        "parent_reference_tokens": [
            {"prefix_index": 0, "next_forced_token_id": 7},
            {"prefix_index": 1, "next_forced_token_id": 8},
            {"prefix_index": 2, "next_forced_token_id": None},
        ],
    }


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "input.input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "input.attention_mask": torch.ones((1, 3), dtype=torch.int64),
        "input.forced_decode_ids": torch.tensor([[7, 8]], dtype=torch.int64),
        "prefill.output.lm_head_input": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.bfloat16
        ),
        "decode.000.output.lm_head_input": torch.tensor(
            [[[7.0, 8.0]]], dtype=torch.bfloat16
        ),
        "decode.000.forced_token_ids": torch.tensor([[7]], dtype=torch.int64),
        "decode.001.output.lm_head_input": torch.tensor(
            [[[9.0, 10.0]]], dtype=torch.bfloat16
        ),
        "decode.001.forced_token_ids": torch.tensor([[8]], dtype=torch.int64),
    }


def test_minimal_trace_extraction_never_loads_logits_and_resumes_checkpoint(
    tmp_path: Path,
) -> None:
    values = _tensors()
    session = _Session(values)
    reference = SimpleNamespace(
        index=0,
        input_sha256="e" * 64,
        record_sha256="f" * 64,
        ordered_tensor_names=tuple(sorted({*values, "prefill.logits"})),
    )
    tensors, receipt = extractor._extract_batch(
        session=session,
        reference=reference,
        prompt_result=_prompt_result(),
        hidden_size=2,
        request_cache_key="1" * 64,
    )
    assert "prefill.logits" not in session.requested_names
    assert tensors["parent_hidden_states"].tolist() == [
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ]
    assert tensors["parent_hidden_states"].dtype == torch.float32
    assert receipt["trace_logits_loaded"] is False
    assert receipt["operation_audit"]["parent_forward_calls"] == 0
    assert session.released

    published = extractor._publish_batch_checkpoint(tmp_path, tensors, receipt)
    loaded = extractor._load_batch_checkpoint(
        tmp_path,
        batch_index=0,
        prompt_result=_prompt_result(),
        request_cache_key="1" * 64,
        trace_session_sha256=_D,
        hidden_size=2,
    )
    assert loaded is not None
    assert loaded[1]["artifact_sha256"] == published["artifact_sha256"]
    torch.testing.assert_close(
        loaded[0]["parent_hidden_states"], tensors["parent_hidden_states"]
    )


def test_frozen_future_review_inventory_is_exactly_the_disjoint_2026_union() -> None:
    path = ROOT / "research" / "lfm2_causal_support_future_manual_review_v1.json"
    raw = json.loads(path.read_text("utf-8"))
    inventory = extractor.authenticate_manual_review_inventory(
        raw,
        expected_inventory_sha256=(
            "64d77f914617cdf927a12cae0aff49c32eb1dda8ce6f37c3d0169404d1ff6170"
        ),
    )
    source_paths = (
        ROOT
        / "experiments"
        / "p35_mlp_behavior_residual_v2_sealed_heldout_prompts_20260711.json",
        ROOT
        / "experiments"
        / "p35_mlp_behavior_residual_v3_sealed_heldout_prompts_20260711.json",
        ROOT / "experiments" / "p35_residual_attention_v_heldout_prompts_20260711.json",
    )
    source_texts = [
        text
        for source in source_paths
        for text in json.loads(source.read_text("utf-8"))
    ]
    assert [row["text"] for row in inventory["prompts"]] == source_texts
    a12 = json.loads(
        (ROOT / "research" / "lfm2_semantic_whole_response_prompts_v1.json").read_text(
            "utf-8"
        )
    )
    a12_ids = {row["id"] for row in a12["prompts"]}
    a12_texts = {
        "\n".join(message["content"] for message in row["messages"])
        for row in a12["prompts"]
    }
    assert not a12_ids.intersection(inventory["prompt_ids"])
    assert not a12_texts.intersection(source_texts)
    assert inventory["safety_review_prompt_count"] == 3
    assert inventory["intelligence_review_prompt_count"] == 15
    assert (
        inventory["review_contract"][
            "refusal_or_safety_policy_difference_counts_as_intelligence_failure"
        ]
        is False
    )
    assert (
        inventory["review_contract"][
            "output_truncation_at_an_evaluator_token_cap_counts_as_model_failure"
        ]
        is False
    )
