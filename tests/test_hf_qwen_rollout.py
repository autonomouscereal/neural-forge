from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_forge.hf_qwen_rollout import (
    build_exact_trace_batches,
    derive_prompt_only_records,
    derive_prompt_only_strings,
    generate_greedy_child_rollouts,
    hash_rollout_provenance,
    hash_token_id_sequence,
    hash_token_id_sequences,
    validate_token_id_sequences,
)


def test_prompt_only_derivation_uses_last_common_response_boundary() -> None:
    examples = [
        "Question: What is 2 + 2?\nAnswer: 4",
        "user: Explain folding.\nassistant: Fold matching channels.",
        "Write a function.\n```python\ndef answer():\n    return 4\n```",
        "No response boundary is present.",
        {"prompt": "Question: Pick one.\nAnswer:"},
        {
            "messages": [
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello."},
            ]
        },
        "Demo\nAnswer: first\nQuestion: final\nAnswer: second",
    ]

    prompts = derive_prompt_only_strings(examples)
    records = derive_prompt_only_records(examples)

    assert prompts == [
        "Question: What is 2 + 2?\nAnswer:",
        "user: Explain folding.\nassistant:",
        "Write a function.\n",
        "No response boundary is present.",
        "Question: Pick one.\nAnswer:",
        "user: Say hello.\nassistant:",
        "Demo\nAnswer: first\nQuestion: final\nAnswer:",
    ]
    assert [record["boundary_kind"] for record in records] == [
        "answer",
        "assistant",
        "code_fence",
        "none",
        "answer",
        "assistant",
        "answer",
    ]
    assert all(len(record["prompt_text_sha256"]) == 64 for record in records)
    assert derive_prompt_only_strings(examples, max_prompt_count=2) == prompts[:2]


class TinyTokenizer:
    def __init__(self, torch) -> None:
        self._torch = torch
        self.vocab_size = 64
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"
        self.padding_sides_seen: list[str] = []

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        return [3 + (ord(character) % 50) for character in str(text)] or [3]

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=32,
    ):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        self.padding_sides_seen.append(self.padding_side)
        if isinstance(texts, str):
            texts = [texts]
        rows = [self.encode(text)[: int(max_length)] for text in texts]
        width = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            count = width - len(row)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * count + row)
                masks.append([0] * count + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * count)
                masks.append([1] * len(row) + [0] * count)
        return {
            "input_ids": self._torch.tensor(padded, dtype=self._torch.long),
            "attention_mask": self._torch.tensor(masks, dtype=self._torch.long),
        }


def test_greedy_rollouts_are_exact_deterministic_and_zero_gradient() -> None:
    torch = pytest.importorskip("torch")

    class TinyChild(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.grad_enabled_seen: list[bool] = []
            self.training_seen: list[bool] = []

        def generate(
            self,
            input_ids,
            attention_mask,
            max_new_tokens,
            do_sample,
            pad_token_id,
            eos_token_id,
        ):
            self.grad_enabled_seen.append(torch.is_grad_enabled())
            self.training_seen.append(self.training)
            assert max_new_tokens == 2
            assert do_sample is False
            assert pad_token_id == 0
            assert eos_token_id == 2
            suffix = torch.tensor(
                [[40 + row, 2] for row in range(int(input_ids.shape[0]))],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat([input_ids, suffix], dim=1)

    tokenizer = TinyTokenizer(torch)
    child = TinyChild()
    child.train()
    examples = [
        "Q1\nAnswer: old",
        "user: Q2\nassistant: old",
        "Q3\nAnswer: old",
    ]

    first = generate_greedy_child_rollouts(
        child,
        tokenizer,
        examples,
        max_prompt_count=2,
        max_length=32,
        max_new_tokens=2,
        device="cpu",
    )
    second = generate_greedy_child_rollouts(
        child,
        tokenizer,
        examples,
        max_prompt_count=2,
        max_length=32,
        max_new_tokens=2,
        device="cpu",
    )

    expected_prompts = ["Q1\nAnswer:", "user: Q2\nassistant:"]
    expected = [
        tokenizer.encode(expected_prompts[0]) + [40, 2],
        tokenizer.encode(expected_prompts[1])[:30] + [41, 2],
    ]
    assert first["prompts"] == expected_prompts
    assert first["token_id_sequences"] == expected
    assert first["sequences"] == expected
    assert [record["token_ids"] for record in first["rollout_records"]] == expected
    assert [record["prompt_length"] for record in first["rollout_records"]] == [
        len(tokenizer.encode(prompt)[:30]) for prompt in expected_prompts
    ]
    assert [record["generated_length"] for record in first["rollout_records"]] == [
        2,
        2,
    ]
    assert [record["rollout_id"] for record in first["rollout_records"]] == [
        record["rollout_id"] for record in second["rollout_records"]
    ]
    assert all(
        record["rollout_id"] == f"sha256:{record['trajectory_sha256']}"
        and record["token_ids_sha256"] == hash_token_id_sequence(record["token_ids"])
        for record in first["rollout_records"]
    )
    assert first["token_id_sequences"] == second["token_id_sequences"]
    assert first["sequence_token_ids_sha256"] == hash_token_id_sequences(expected)
    assert (
        first["provenance"]["provenance_sha256"]
        == hash_rollout_provenance(first["provenance"])
        == second["provenance"]["provenance_sha256"]
    )
    assert first["provenance"]["zero_gradient"] is True
    assert first["provenance"]["selected_prompt_count"] == 2
    assert first["provenance"]["discarded_prompt_count"] == 1
    assert child.grad_enabled_seen == [False, False]
    assert child.training_seen == [False, False]
    assert child.training is True
    assert child.weight.grad is None
    assert tokenizer.padding_side == "right"
    assert tokenizer.padding_sides_seen == ["left", "left"]
    assert (
        hash_token_id_sequence(expected[0])
        == first["provenance"]["sequences"][0]["sequence_token_ids_sha256"]
    )


def test_generation_restores_model_and_tokenizer_state_after_error() -> None:
    torch = pytest.importorskip("torch")

    class FailingChild(torch.nn.Module):
        def generate(self, **_kwargs):
            assert self.training is False
            assert torch.is_grad_enabled() is False
            raise RuntimeError("generation failed")

    child = FailingChild()
    child.train()
    tokenizer = TinyTokenizer(torch)

    with pytest.raises(RuntimeError, match="generation failed"):
        generate_greedy_child_rollouts(
            child,
            tokenizer,
            ["Question\nAnswer: response"],
            max_prompt_count=1,
            max_length=16,
            max_new_tokens=2,
        )

    assert child.training is True
    assert tokenizer.padding_side == "right"


def test_generation_accepts_boolean_masks_and_rejects_non_integer_ids() -> None:
    torch = pytest.importorskip("torch")

    class BoolMaskTokenizer(TinyTokenizer):
        def __call__(self, *args, **kwargs):
            encoded = super().__call__(*args, **kwargs)
            encoded["attention_mask"] = encoded["attention_mask"].to(torch.bool)
            return encoded

    class Child(torch.nn.Module):
        def __init__(self, output_dtype) -> None:
            super().__init__()
            self.output_dtype = output_dtype

        def generate(self, input_ids, **_kwargs):
            suffix = torch.full(
                (int(input_ids.shape[0]), 1),
                2,
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat([input_ids, suffix], dim=1).to(self.output_dtype)

    tokenizer = BoolMaskTokenizer(torch)
    child = Child(torch.long)
    child.eval()
    result = generate_greedy_child_rollouts(
        child,
        tokenizer,
        ["Question\nAnswer: response"],
        max_prompt_count=1,
        max_length=16,
        max_new_tokens=1,
    )
    assert result["generated_token_id_sequences"] == [[2]]
    assert child.training is False
    assert tokenizer.padding_side == "right"

    invalid = Child(torch.float32)
    invalid.train()
    with pytest.raises(ValueError, match="integer token IDs"):
        generate_greedy_child_rollouts(
            invalid,
            tokenizer,
            ["Question\nAnswer: response"],
            max_prompt_count=1,
            max_length=16,
            max_new_tokens=1,
        )
    assert invalid.training is True
    assert tokenizer.padding_side == "right"


def test_exact_trace_batches_match_qwen_compose_contract() -> None:
    torch = pytest.importorskip("torch")
    sequences = [[4, 5, 6], [7], [8, 9]]

    batches = build_exact_trace_batches(
        sequences,
        batch_size=2,
        pad_token_id=0,
        device="cpu",
        max_length=3,
        vocab_size=10,
    )

    assert len(batches) == 2
    assert set(batches[0]) == {"start", "count", "attention_mask", "model_inputs"}
    assert batches[0]["start"] == 0
    assert batches[0]["count"] == 2
    assert batches[0]["model_inputs"]["input_ids"].tolist() == [
        [4, 5, 6],
        [7, 0, 0],
    ]
    assert batches[0]["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batches[0]["attention_mask"] is batches[0]["model_inputs"]["attention_mask"]
    assert batches[1]["start"] == 2
    assert batches[1]["count"] == 1
    assert batches[1]["model_inputs"]["input_ids"].tolist() == [[8, 9]]

    class Recorder:
        def __init__(self) -> None:
            self.rows = []

        def __call__(self, input_ids, attention_mask):
            self.rows.append((input_ids.clone(), attention_mask.clone()))
            return SimpleNamespace(last_hidden_state=input_ids)

    donor = Recorder()
    child = Recorder()
    for batch in batches:
        donor(**batch["model_inputs"])
        child(**batch["model_inputs"])
    for donor_row, child_row in zip(donor.rows, child.rows):
        torch.testing.assert_close(donor_row[0], child_row[0])
        torch.testing.assert_close(donor_row[1], child_row[1])

    left = build_exact_trace_batches(
        [[4], [5, 6]],
        batch_size=2,
        pad_token_id=0,
        padding_side="left",
    )[0]
    assert left["model_inputs"]["input_ids"].tolist() == [[0, 4], [5, 6]]
    assert left["attention_mask"].tolist() == [[0, 1], [1, 1]]


@pytest.mark.parametrize(
    ("padding_side", "expected_input_ids", "expected_trace_mask"),
    [
        (
            "right",
            [[4, 5, 6, 7], [8, 9, 0, 0]],
            [[0, 0, 1, 0], [1, 0, 0, 0]],
        ),
        (
            "left",
            [[4, 5, 6, 7], [0, 0, 8, 9]],
            [[0, 0, 1, 0], [0, 0, 1, 0]],
        ),
    ],
)
def test_trace_masks_align_one_generated_action_with_left_and_right_padding(
    padding_side, expected_input_ids, expected_trace_mask
) -> None:
    pytest.importorskip("torch")

    batch = build_exact_trace_batches(
        [[4, 5, 6, 7], [8, 9]],
        batch_size=2,
        pad_token_id=0,
        prompt_lengths=[3, 1],
        padding_side=padding_side,
    )[0]

    assert batch["model_inputs"]["input_ids"].tolist() == expected_input_ids
    assert batch["trace_mask"].tolist() == expected_trace_mask
    assert batch["trace_mask"].sum(dim=1).tolist() == [1, 1]
    assert batch["attention_mask"].sum(dim=1).tolist() == [4, 2]
    assert batch["model_inputs"]["attention_mask"] is batch["attention_mask"]
    assert "trace_mask" not in batch["model_inputs"]


@pytest.mark.parametrize(
    ("prompt_lengths", "message"),
    [
        ([1], "exactly one value"),
        ([0, 1], "positive integer"),
        ([1.0, 1], "positive integer"),
        ([3, 1], "exceeds sequence length"),
    ],
)
def test_trace_batch_prompt_lengths_are_validated(prompt_lengths, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_exact_trace_batches(
            [[4, 5], [6]],
            prompt_lengths=prompt_lengths,
        )


@pytest.mark.parametrize(
    ("sequences", "kwargs", "message"),
    [
        ([], {}, "must not be empty"),
        ([[1, -1]], {}, "negative token ID"),
        ([[True]], {}, "not booleans"),
        ([[1.0]], {}, "only integer token IDs"),
        ([[4]], {"vocab_size": 4}, "outside vocab_size"),
        ([[1, 2]], {"max_length": 1}, "exceeding max_length"),
    ],
)
def test_token_id_validation_rejects_invalid_sequences(
    sequences, kwargs, message
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_token_id_sequences(sequences, **kwargs)


def test_rollout_and_batch_limits_are_validated_before_use() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = TinyTokenizer(torch)

    with pytest.raises(ValueError, match="smaller than max_length"):
        generate_greedy_child_rollouts(
            object(),
            tokenizer,
            ["Question\nAnswer:"],
            max_length=4,
            max_new_tokens=4,
        )
    with pytest.raises(ValueError, match="padding_side"):
        build_exact_trace_batches([[1]], padding_side="middle")
    with pytest.raises(ValueError, match="outside vocab_size"):
        build_exact_trace_batches([[1]], pad_token_id=4, vocab_size=4)
