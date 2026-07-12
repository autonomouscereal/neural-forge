from __future__ import annotations

import math

import pytest

from neural_forge.cli import build_parser
import neural_forge.hf_qwen_logit_head as logit_head
from neural_forge.hf_qwen_logit_head import _bounded_aligned_prompt_rows
from neural_forge.hf_qwen_logit_head import _validate_aligned_tokenizers
from neural_forge.hf_qwen_logit_head import fit_qwen_logit_head
from neural_forge.hf_qwen_logit_head import materialize_full_vocabulary_head


torch = pytest.importorskip("torch")


def test_cli_routes_balanced_calibration_into_checkpoint_transplant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_transplant(**kwargs):
        captured.update(kwargs)
        return {"zero_gradient": True}

    monkeypatch.setattr(
        logit_head, "transplant_qwen_logit_head_checkpoint", fake_transplant
    )
    monkeypatch.setattr(
        "neural_forge.cli._load_qwen_pair_calibration_texts",
        lambda _path, _limit, _balance: ["math", "code", "general"],
    )
    args = build_parser().parse_args(
        [
            "hf-qwen-logit-head",
            "donor",
            "child",
            "out",
            "--calibration-file",
            "calibration.jsonl",
            "--candidate-alphas",
            "0,0.5,1",
            "--device",
            "cpu",
        ]
    )

    assert args.func(args) == {"zero_gradient": True}
    assert captured["donor_model"] == "donor"
    assert captured["child_model"] == "child"
    assert captured["calibration_texts"] == ["math", "code", "general"]
    assert captured["candidate_alphas"] == [0.0, 0.5, 1.0]
    assert captured["max_rows"] == 4096


def test_trace_row_bound_preserves_donor_child_alignment() -> None:
    child = [torch.arange(12, dtype=torch.float32).reshape(6, 2)] * 3
    donor = [torch.arange(18, dtype=torch.float32).reshape(6, 3)] * 3

    child_rows, donor_rows, prompt_ids, report = _bounded_aligned_prompt_rows(
        torch, child, donor, ["a", "b", "c"], max_rows=7
    )

    assert child_rows.shape == (7, 2)
    assert donor_rows.shape == (7, 3)
    assert len(prompt_ids) == 7
    assert report == {
        "original_rows": 18,
        "selected_rows": 7,
        "selection": "global-evenly-spaced",
        "max_rows": 7,
    }


def test_aligned_tokenizer_validation_checks_ids_and_special_tokens() -> None:
    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 2

        def __init__(self, offset=0):
            self.offset = offset

        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is True
            return [len(text) + self.offset, self.eos_token_id]

    _validate_aligned_tokenizers(Tokenizer(), Tokenizer(), ["one", "two"])
    with pytest.raises(ValueError, match="tokenization differs"):
        _validate_aligned_tokenizers(Tokenizer(), Tokenizer(offset=1), ["one"])


def _known_mapping_problem(*, dtype=torch.float64):
    generator = torch.Generator().manual_seed(41)
    child_rows = torch.randn(72, 3, generator=generator, dtype=dtype)
    decoder = torch.tensor(
        [
            [1.2, -0.3, 0.7],
            [-0.5, 0.8, 0.2],
            [0.1, -1.1, 0.6],
            [0.9, 0.4, -0.2],
            [-0.7, 0.3, 1.0],
        ],
        dtype=dtype,
    )
    donor_rows = child_rows.matmul(decoder.t())
    donor_head = torch.randn(29, 5, generator=generator, dtype=dtype)
    current_head = -0.5 * donor_head.matmul(decoder)
    prompt_ids = [f"prompt-{index // 3}" for index in range(72)]
    return child_rows, donor_rows, donor_head, current_head, decoder, prompt_ids


def test_recovers_known_low_dimensional_teacher_mapping() -> None:
    x, z, donor_head, current_head, expected_decoder, prompt_ids = (
        _known_mapping_problem()
    )

    result = fit_qwen_logit_head(
        x,
        z,
        donor_head,
        current_head,
        prompt_ids=prompt_ids,
        ridge=1e-12,
        split_seed=7,
        candidate_alphas=[0.0, 0.5, 1.0],
        vocab_chunk_size=7,
    )

    assert result["decoder"].shape == expected_decoder.shape
    assert torch.allclose(result["decoder"], expected_decoder, atol=1e-9, rtol=1e-9)
    expected_head = donor_head.matmul(expected_decoder)
    assert torch.allclose(
        result["fitted_weight"], expected_head, atol=1e-9, rtol=1e-9
    )
    assert result["report"]["chosen_alpha"] == 1.0
    assert result["report"]["chosen_metrics"]["top1_agreement"] == pytest.approx(
        1.0
    )
    assert result["report"]["chosen_metrics"][
        "teacher_top1_in_child_topk"
    ] == pytest.approx(1.0)
    assert result["report"]["chosen_metrics"][
        "teacher_margin_sign_retention"
    ] == pytest.approx(1.0)
    assert result["report"]["chosen_metrics"][
        "kl_teacher_child_mean"
    ] < 1e-12


def test_held_out_real_logits_choose_better_alpha() -> None:
    x, z, donor_head, current_head, _, prompt_ids = _known_mapping_problem()

    report = fit_qwen_logit_head(
        x,
        z,
        donor_head,
        current_head,
        prompt_ids=prompt_ids,
        ridge=1e-10,
        split_seed=13,
        candidate_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        top_k=3,
        vocab_chunk_size=5,
    )["report"]

    by_alpha = {item["alpha"]: item for item in report["candidate_metrics"]}
    assert report["chosen_alpha"] == 1.0
    assert by_alpha[1.0]["top1_agreement"] > by_alpha[0.0]["top1_agreement"]
    assert by_alpha[1.0]["teacher_top1_in_child_topk"] > by_alpha[0.0][
        "teacher_top1_in_child_topk"
    ]
    assert by_alpha[1.0]["teacher_margin_sign_retention"] > by_alpha[0.0][
        "teacher_margin_sign_retention"
    ]
    assert by_alpha[1.0]["kl_teacher_child_mean"] < by_alpha[0.0][
        "kl_teacher_child_mean"
    ]
    assert report["evaluation"]["actual_teacher_logits"] == "Z @ H.T"
    assert report["evaluation"]["actual_child_logits"] == "X @ W.T"
    assert report["evaluation"]["surrogate_logits"] is False
    assert report["selection"]["held_out_only"] is True


def test_materialization_and_evaluation_cover_chunked_full_vocabulary() -> None:
    x, z, donor_head, current_head, decoder, prompt_ids = _known_mapping_problem(
        dtype=torch.float32
    )
    expected = donor_head.matmul(decoder)

    chunked = materialize_full_vocabulary_head(
        donor_head,
        decoder,
        vocab_chunk_size=4,
        output_dtype=torch.float32,
        output_device="cpu",
    )
    assert torch.allclose(chunked, expected, atol=1e-6, rtol=1e-6)

    report = fit_qwen_logit_head(
        x,
        z,
        donor_head,
        current_head,
        prompt_ids=prompt_ids,
        ridge=1e-8,
        candidate_alphas=[0.0, 1.0],
        vocab_chunk_size=4,
    )["report"]
    assert report["chunking"]["vocabulary_chunk_size"] == 4
    assert report["chunking"]["vocabulary_chunk_count"] == math.ceil(29 / 4)
    assert report["chunking"]["materialized_vocabulary_rows"] == 29
    assert report["chunking"]["full_vocabulary_materialized"] is True
    assert report["chunking"]["full_vocabulary_evaluated"] is True
    assert report["evaluation"]["vocabulary_rows_evaluated"] == 29
    assert report["evaluation"]["passes_over_full_vocabulary"] == 2


def test_preserves_inputs_and_existing_gradients_without_autograd() -> None:
    x, z, donor_head, current_head, _, prompt_ids = _known_mapping_problem(
        dtype=torch.float32
    )
    inputs = [
        x.detach().clone().requires_grad_(True),
        z.detach().clone().requires_grad_(True),
        donor_head.detach().clone().requires_grad_(True),
        current_head.detach().clone().requires_grad_(True),
    ]
    for index, tensor in enumerate(inputs):
        tensor.grad = torch.full_like(tensor, float(index + 1))
    values_before = [tensor.detach().clone() for tensor in inputs]
    gradients_before = [tensor.grad.detach().clone() for tensor in inputs]

    result = fit_qwen_logit_head(
        *inputs,
        prompt_ids=prompt_ids,
        ridge=1e-8,
        candidate_alphas=[0.0, 1.0],
        vocab_chunk_size=6,
    )

    for tensor, value_before, gradient_before in zip(
        inputs, values_before, gradients_before, strict=True
    ):
        assert torch.equal(tensor.detach(), value_before)
        assert torch.equal(tensor.grad, gradient_before)
    assert result["weight"].requires_grad is False
    assert result["fitted_weight"].requires_grad is False
    assert result["decoder"].requires_grad is False
    report = result["report"]
    assert report["zero_gradient"] is True
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["autograd_enabled_during_fit"] is False
    assert report["inputs_mutated"] is False
    assert report["input_gradients_mutated"] is False
    assert report["hashes"]["inputs_before"] == report["hashes"]["inputs_after"]
    assert report["hashes"]["input_gradients_before"] == report["hashes"][
        "input_gradients_after"
    ]
    assert len(report["hashes"]["fitted_head_sha256"]) == 64


def test_rejects_invalid_shapes_and_nonfinite_data() -> None:
    x, z, donor_head, current_head, _, prompt_ids = _known_mapping_problem()

    with pytest.raises(ValueError, match="same row count"):
        fit_qwen_logit_head(
            x[:-1],
            z,
            donor_head,
            current_head,
            prompt_ids=prompt_ids[:-1],
        )
    with pytest.raises(ValueError, match="same vocabulary size"):
        fit_qwen_logit_head(
            x,
            z,
            donor_head[:-1],
            current_head,
            prompt_ids=prompt_ids,
        )
    bad = z.detach().clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_qwen_logit_head(
            x,
            bad,
            donor_head,
            current_head,
            prompt_ids=prompt_ids,
        )
    with pytest.raises(ValueError, match="decoder_prior shape"):
        fit_qwen_logit_head(
            x,
            z,
            donor_head,
            current_head,
            prompt_ids=prompt_ids,
            decoder_prior=torch.zeros(4, 3),
        )
