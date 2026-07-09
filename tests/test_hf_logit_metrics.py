from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_logit_metrics import aggregate_by_domain
from neural_forge.hf_logit_metrics import align_vocabulary_logits
from neural_forge.hf_logit_metrics import build_blank_baseline_cache
from neural_forge.hf_logit_metrics import build_logit_diagnostics_report
from neural_forge.hf_logit_metrics import calibration_hashes
from neural_forge.hf_logit_metrics import centered_logit_cosine
from neural_forge.hf_logit_metrics import collect_hf_calibration_logits
from neural_forge.hf_logit_metrics import collect_hf_logit_diagnostics
from neural_forge.hf_logit_metrics import compute_logit_diagnostics
from neural_forge.hf_logit_metrics import normalized_kl_retention
from neural_forge.hf_logit_metrics import paired_bootstrap_lower_bound
from neural_forge.hf_logit_metrics import paired_bootstrap_report
from neural_forge.hf_logit_metrics import teacher_child_kl
from neural_forge.hf_logit_metrics import teacher_margin_sign_retention
from neural_forge.hf_logit_metrics import teacher_margin_values
from neural_forge.hf_logit_metrics import teacher_top1_in_child_topk
from neural_forge.hf_logit_metrics import top1_agreement
from neural_forge.hf_logit_metrics import top_k_agreement


def test_centered_logit_cosine_is_shift_and_scale_invariant() -> None:
    teacher = torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    child = torch.tensor([[12.0, 14.0, 18.0], [9.0, 9.0, 9.0]])

    scores = centered_logit_cosine(teacher, child)

    assert scores.tolist() == pytest.approx([1.0, 0.0])
    assert centered_logit_cosine(teacher[:1], -teacher[:1]).item() == pytest.approx(-1.0)
    assert torch.equal(teacher, torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]]))


def test_top1_and_topk_agreement_have_explicit_set_semantics() -> None:
    teacher = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    child = torch.tensor([[1.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]])

    assert top1_agreement(teacher, child).tolist() == [0.0, 1.0]
    assert top_k_agreement(teacher, child, k=2).tolist() == pytest.approx([0.5, 1.0])
    assert top_k_agreement(teacher, child, k=2, mode="exact").tolist() == [0.0, 1.0]
    assert teacher_top1_in_child_topk(teacher, child, k=2).tolist() == [0.0, 1.0]
    assert top_k_agreement(teacher, child, k=99).shape == torch.Size([2])


def test_kl_is_teacher_to_child_and_uses_only_aligned_vocabulary() -> None:
    teacher = torch.tensor([[1.0, -4.0, 3.0, 0.0]])
    child = torch.tensor([[8.0, 11.0, -2.0, 1.0, 13.0]])
    aligned_teacher, aligned_child = align_vocabulary_logits(
        teacher,
        child,
        teacher_vocab_indices=[2, 0],
        child_vocab_indices=[4, 1],
    )

    assert aligned_teacher.tolist() == [[3.0, 1.0]]
    assert aligned_child.tolist() == [[13.0, 11.0]]
    assert teacher_child_kl(
        teacher,
        child,
        teacher_vocab_indices=[2, 0],
        child_vocab_indices=[4, 1],
    ).item() == pytest.approx(0.0, abs=1e-7)
    assert teacher_child_kl(teacher, teacher + 50.0).item() == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(ValueError, match="unique"):
        align_vocabulary_logits(
            teacher,
            child,
            teacher_vocab_indices=[0, 0],
            child_vocab_indices=[1, 2],
        )


def test_normalized_kl_retention_is_blank_relative_and_unclipped() -> None:
    values = normalized_kl_retention(
        torch.tensor([0.25, 0.5, 1.5]),
        torch.tensor([1.0, 1.0, 1.0]),
    )

    assert values.tolist() == pytest.approx([0.75, 0.5, -0.5])
    assert normalized_kl_retention(values.new_tensor([0.5, 1.5]), values.new_ones(2), reduction="ratio-of-means").item() == pytest.approx(0.0)


def test_teacher_margin_sign_retention_uses_teacher_token_pair() -> None:
    teacher = torch.tensor([[4.0, 3.0, 0.0], [4.0, 3.0, 0.0]])
    child = torch.tensor([[2.0, 1.0, 9.0], [1.0, 2.0, 0.0]])

    teacher_margin, child_margin = teacher_margin_values(teacher, child)

    assert teacher_margin.tolist() == [1.0, 1.0]
    assert child_margin.tolist() == [1.0, -1.0]
    assert teacher_margin_sign_retention(teacher, child).tolist() == [1.0, 0.0]
    explicit = teacher_margin_sign_retention(
        teacher,
        child,
        positive_indices=[1, 1],
        negative_indices=[0, 0],
    )
    assert explicit.tolist() == [1.0, 0.0]


def test_compute_logit_diagnostics_returns_observation_level_tensors() -> None:
    teacher = torch.tensor([[3.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
    child = torch.tensor([[4.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    metrics = compute_logit_diagnostics(
        teacher,
        child,
        top_k=2,
        blank_baseline_kl=torch.ones(2),
    )

    expected = {
        "centered_logit_cosine",
        "top1_agreement",
        "topk_agreement",
        "topk_exact_agreement",
        "teacher_top1_in_child_topk",
        "kl_teacher_child",
        "teacher_margin",
        "child_on_teacher_margin",
        "teacher_margin_sign_retention",
        "normalized_kl_retention",
    }
    assert set(metrics) == expected
    assert all(value.shape == torch.Size([2]) for value in metrics.values())


def test_paired_bootstrap_is_deterministic_and_resamples_pair_units() -> None:
    values = torch.tensor([1.0, 3.0, 5.0, 7.0])
    pair_ids = ["a", "a", "b", "b"]

    first = paired_bootstrap_report(values, pair_ids=pair_ids, n_resamples=200, seed=17)
    second = paired_bootstrap_report(values, pair_ids=pair_ids, n_resamples=200, seed=17)

    assert first == second
    assert first["estimate"] == pytest.approx(4.0)
    assert first["pair_count"] == 2
    assert paired_bootstrap_lower_bound(
        torch.ones(8),
        torch.full((8,), 0.25),
        n_resamples=100,
    ) == pytest.approx(0.75)


def test_domain_aggregation_reports_bootstrap_lower_bounds() -> None:
    result = aggregate_by_domain(
        {
            "cosine": torch.tensor([1.0, 0.5, 0.25, 0.75]),
            "agreement": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        },
        ["math", "math", "code", "code"],
        pair_ids=["m1", "m2", "c1", "c2"],
        n_resamples=100,
        seed=9,
    )

    assert result["math"]["row_count"] == 2
    assert result["math"]["metrics"]["cosine"]["mean"] == pytest.approx(0.75)
    assert "paired_bootstrap_lower_bound" in result["code"]["metrics"]["agreement"]


def test_blank_cache_and_report_bind_to_calibration_hashes() -> None:
    teacher = torch.tensor([[4.0, 2.0, 0.0], [0.0, 2.0, 4.0]])
    child = torch.tensor([[3.0, 2.0, 0.0], [0.0, 3.0, 2.0]])
    blank = torch.zeros_like(teacher)
    hashes = {"algorithm": "sha256", "calibration_layout_sha256": "calibration-a"}
    cache = build_blank_baseline_cache(
        teacher,
        blank,
        calibration_hashes=hashes,
        domains=["math", "code"],
        pair_ids=["m", "c"],
    )

    report = build_logit_diagnostics_report(
        teacher,
        child,
        domains=["math", "code"],
        pair_ids=["m", "c"],
        blank_baseline_cache=cache,
        calibration_hashes=hashes,
        bootstrap_resamples=100,
    )

    assert cache["zero_gradient"] is True
    assert cache["optimizer_steps"] == 0
    assert cache["closed_form"] is True
    assert report["zero_gradient"] is True
    assert report["optimizer_steps"] == 0
    assert report["closed_form"] is True
    assert report["blank_baseline_cached"] is True
    assert report["blank_baseline"]["kind"] == "neural_forge_blank_logit_baseline"
    assert report["blank_baseline"]["mean_kl_teacher_blank"] == pytest.approx(cache["mean_kl_teacher_blank"])
    assert report["calibration_hashes"] == hashes
    assert set(report["per_domain"]) == {"code", "math"}
    assert "normalized_kl_retention" in report["overall"]["metrics"]
    json.dumps(report, allow_nan=False)

    with pytest.raises(ValueError, match="calibration hashes"):
        build_logit_diagnostics_report(
            teacher,
            child,
            blank_baseline_cache=cache,
            calibration_hashes={"algorithm": "sha256", "calibration_layout_sha256": "calibration-b"},
            bootstrap_resamples=10,
        )


def test_flat_blank_cache_applies_to_batched_sequence_logits() -> None:
    teacher = torch.tensor(
        [
            [[4.0, 2.0, 0.0], [3.0, 1.0, 0.0]],
            [[0.0, 2.0, 4.0], [0.0, 1.0, 3.0]],
        ]
    )
    child = teacher * 0.75
    hashes = {"algorithm": "sha256", "calibration_layout_sha256": "batched"}
    cache = build_blank_baseline_cache(teacher, torch.zeros_like(teacher), calibration_hashes=hashes)

    report = build_logit_diagnostics_report(
        teacher,
        child,
        domains=["math", "math", "code", "code"],
        pair_ids=["m", "m", "c", "c"],
        blank_baseline_cache=cache,
        calibration_hashes=hashes,
        bootstrap_resamples=20,
    )

    assert report["row_count"] == 4
    assert report["overall"]["metrics"]["normalized_kl_retention"]["pair_count"] == 2


def test_calibration_hashes_are_order_sensitive_and_bounded() -> None:
    forward = calibration_hashes(
        [{"id": "a", "domain": "math", "text": "one"}, {"id": "b", "text": "two"}],
        max_samples=1,
    )
    reverse = calibration_hashes(
        [{"id": "b", "text": "two"}, {"id": "a", "domain": "math", "text": "one"}],
        max_samples=1,
    )

    assert forward["sample_count"] == 1
    assert forward["truncated"] is True
    assert forward["ordered_samples_sha256"] != reverse["ordered_samples_sha256"]


class _FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        return_tensors,
        padding,
        truncation,
        max_length,
        **kwargs,
    ):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        assert kwargs == {}
        rows = [[(ord(character) % 4) + 1 for character in text][:max_length] for text in texts]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int, offset: int = 0) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, vocab_size))
        self.vocab_size = vocab_size
        self.offset = offset
        self.grad_enabled = []
        self.inference_enabled = []
        self.training_seen = []

    def forward(self, input_ids, attention_mask=None, use_cache=True):
        assert use_cache is False
        self.grad_enabled.append(torch.is_grad_enabled())
        self.inference_enabled.append(torch.is_inference_mode_enabled())
        self.training_seen.append(self.training)
        target = (input_ids + self.offset) % self.vocab_size
        logits = torch.nn.functional.one_hot(target, num_classes=self.vocab_size).to(torch.float32) * 4.0
        return SimpleNamespace(logits=logits + self.bias)


def test_hf_collector_is_bounded_hashed_and_never_writes_gradients() -> None:
    teacher = _FakeCausalLM(6)
    child = _FakeCausalLM(6, offset=1)
    teacher.train()
    child.eval()
    teacher.bias.grad = torch.full_like(teacher.bias, 0.25)
    child.bias.grad = torch.full_like(child.bias, -0.5)
    teacher_grad_before = teacher.bias.grad.clone()
    child_grad_before = child.bias.grad.clone()
    teacher_weight_before = teacher.bias.detach().clone()
    child_weight_before = child.bias.detach().clone()
    samples = [
        {"id": "math-a", "domain": "math", "text": "abcd"},
        {"id": "code-a", "domain": "code", "text": "xyz"},
        {"id": "extra", "domain": "other", "text": "q"},
    ]

    collected = collect_hf_calibration_logits(
        teacher,
        child,
        _FakeTokenizer(),
        samples,
        max_samples=2,
        max_length=4,
        max_rows=4,
        max_logit_elements=100,
    )

    assert collected["teacher_logits"].shape == (4, 6)
    assert collected["child_logits"].shape == (4, 6)
    assert collected["teacher_logits"].requires_grad is False
    assert collected["zero_gradient"] is True
    assert collected["optimizer_steps"] == 0
    assert collected["metadata"]["truncated"] == {
        "max_samples": True,
        "max_rows": True,
        "max_logit_elements": False,
    }
    assert collected["metadata"]["retained_logit_elements"] <= 100
    assert len(collected["calibration_hashes"]["tokenized_inputs_sha256"]) == 64
    assert teacher.training is True
    assert child.training is False
    assert teacher.grad_enabled == [False]
    assert child.grad_enabled == [False]
    assert teacher.inference_enabled == [True]
    assert child.inference_enabled == [True]
    assert teacher.training_seen == [False]
    assert child.training_seen == [False]
    assert torch.equal(teacher.bias.grad, teacher_grad_before)
    assert torch.equal(child.bias.grad, child_grad_before)
    assert torch.equal(teacher.bias, teacher_weight_before)
    assert torch.equal(child.bias, child_weight_before)


def test_hf_collector_enforces_logit_element_bound_and_alignment() -> None:
    teacher = _FakeCausalLM(6)
    child = _FakeCausalLM(7, offset=1)
    collected = collect_hf_calibration_logits(
        teacher,
        child,
        _FakeTokenizer(),
        [{"domain": "math", "text": "abcd"}],
        max_rows=10,
        max_logit_elements=18,
        vocab_alignment=[(0, 1), (2, 3), (4, 5)],
    )

    assert collected["teacher_logits"].shape == (3, 3)
    assert collected["child_logits"].shape == (3, 3)
    assert collected["metadata"]["retained_logit_elements"] == 18
    assert collected["metadata"]["truncated"]["max_logit_elements"] is True
    cache = build_blank_baseline_cache(
        collected["teacher_logits"],
        collected["child_logits"],
        calibration_hashes=collected["calibration_hashes"],
    )
    assert cache["vocabulary_alignment_sha256"] == collected["metadata"]["vocabulary_alignment_sha256"]


def test_hf_diagnostic_collector_reports_provenance_and_domains() -> None:
    report = collect_hf_logit_diagnostics(
        _FakeCausalLM(6),
        _FakeCausalLM(6, offset=1),
        _FakeTokenizer(),
        [
            {"id": "m", "domain": "math", "text": "ab"},
            {"id": "c", "domain": "code", "text": "cd"},
        ],
        max_rows=4,
        max_logit_elements=100,
        top_k=2,
        bootstrap_resamples=50,
    )

    assert report["zero_gradient"] is True
    assert report["optimizer_steps"] == 0
    assert report["calibration"]["gradient_context"] == "torch.inference_mode"
    assert report["calibration"]["parameter_gradients_written"] is False
    assert set(report["per_domain"]) == {"code", "math"}
    assert report["calibration_hashes"]["tokenized_inputs_sha256"] == report["calibration"]["calibration_hashes"]["tokenized_inputs_sha256"]
