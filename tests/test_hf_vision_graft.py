from __future__ import annotations

import random

from neural_forge.hf_vision_graft import GRAFT_QUESTIONS
from neural_forge.hf_vision_graft import _encode_prompt_answer
from neural_forge.hf_vision_graft import _make_hf_vision_example
from neural_forge.hf_vision_graft import _normalize_task_keys
from neural_forge.hf_vision_graft import graft_answer_options
from neural_forge.hf_vision_graft import graft_prompt
from neural_forge.hf_vision_graft import train_hf_native_vision_graft
from neural_forge.native_multimodal import COLORS, DIGITS
import pytest


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [len(part) for part in text.split()]}


def test_graft_prompt_mentions_image_prefix_and_question() -> None:
    prompt = graft_prompt("What color is the square?")
    assert "Image patches" in prompt
    assert "What color is the square?" in prompt
    assert prompt.endswith("Answer:")


def test_graft_answer_options_match_task_spaces() -> None:
    assert graft_answer_options("square_color") == list(COLORS)
    assert graft_answer_options("circle_color") == list(COLORS)
    assert graft_answer_options("digit") == list(DIGITS)


def test_make_hf_vision_example_has_consistent_answer() -> None:
    for seed in range(20):
        example = _make_hf_vision_example(random.Random(seed), seed)
        assert example["image"].shape == (3, 16, 16)
        assert (example["question"], example["answer_key"]) in GRAFT_QUESTIONS
        if example["answer_key"] == "square_color":
            assert example["answer"] == example["square_color"]
        elif example["answer_key"] == "circle_color":
            assert example["answer"] == example["circle_color"]
        else:
            assert example["answer"] == example["digit"]


def test_encode_prompt_answer_masks_prompt_and_keeps_answer_labels() -> None:
    encoded = _encode_prompt_answer(_Tokenizer(), "What digit is shown?", "two")
    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert encoded["labels"].count(-100) == len(encoded["prompt_ids"])
    assert encoded["labels"][-1] in encoded["answer_ids"]
    assert encoded["answer_ids"]


def test_task_key_filter_limits_generated_examples() -> None:
    for seed in range(10):
        example = _make_hf_vision_example(random.Random(seed), seed, task_keys=["square_color"])
        assert example["answer_key"] == "square_color"
    assert _normalize_task_keys(["square_color", "digit"]) == ["square_color", "digit"]


def test_trainable_layer_fp16_is_rejected_before_model_load(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit non-fp16 dtype"):
        train_hf_native_vision_graft(
            "not-loaded",
            tmp_path,
            train_last_layers=1,
            torch_dtype="float16",
        )
