from __future__ import annotations

import pytest

from neural_forge.hf_tools import GENERATION_PROMPTS
from neural_forge.hf_tools import gate_text_report
from neural_forge.hf_tools import rank_evaluation_reports
from neural_forge.hf_tools import _assert_tokenizers_compatible
from neural_forge.hf_tools import _ensure_unique_ratio_tags
from neural_forge.hf_tools import _finite_float
from neural_forge.hf_tools import _generation_diagnostics
from neural_forge.hf_tools import _ratio_tag
from neural_forge.hf_tools import _score_behavior_probe


def _report(ppl: float, acc: float, continuation: str = " This is a healthy continuation with enough varied words.") -> dict:
    return {
        "model": "test",
        "param_count": 10,
        "perplexity": {"mean": ppl, "items": [ppl]},
        "choice_accuracy": {"accuracy": acc, "correct": int(acc * 4), "total": 4, "items": []},
        "generations": [{"prompt": "Prompt", "output": "Prompt" + continuation}],
    }


def test_gate_text_report_passes_near_baseline_candidate() -> None:
    verdict = gate_text_report(_report(100, 1.0), _report(110, 1.0), max_ppl_ratio=1.25)
    assert verdict["passed"] is True


def test_gate_text_report_rejects_ppl_regression() -> None:
    verdict = gate_text_report(_report(100, 1.0), _report(200, 1.0), max_ppl_ratio=1.25)
    assert verdict["passed"] is False
    assert any(check["name"] == "perplexity_ratio" and not check["passed"] for check in verdict["checks"])


def test_gate_text_report_rejects_blank_generation() -> None:
    verdict = gate_text_report(_report(100, 1.0), _report(100, 1.0, continuation="   "))
    assert verdict["passed"] is False
    assert any(check["name"] == "generation_health" and not check["passed"] for check in verdict["checks"])


def test_gate_text_report_rejects_irrelevant_diverse_generation() -> None:
    candidate = _report(
        100,
        1.0,
        continuation=" mercury cathedral envelope horizon lantern quartz invoice maple velvet orbit syntax.",
    )
    candidate["generations"][0]["prompt"] = GENERATION_PROMPTS[0]
    candidate["generations"][0]["output"] = (
        GENERATION_PROMPTS[0] + candidate["generations"][0]["output"][len("Prompt") :]
    )

    verdict = gate_text_report(_report(100, 1.0), candidate)

    assert verdict["passed"] is False
    generation_check = next(check for check in verdict["checks"] if check["name"] == "generation_health")
    assert generation_check["passed"] is False
    assert generation_check["value"][0]["relevant_enough"] is False


def test_gate_text_report_rejects_behavior_suite_regression() -> None:
    baseline = _report(100, 1.0)
    baseline["behavior_suite"] = {"pass_rate": 0.75, "correct": 3, "total": 4, "items": []}
    candidate = _report(100, 1.0)
    candidate["behavior_suite"] = {"pass_rate": 0.25, "correct": 1, "total": 4, "items": []}

    verdict = gate_text_report(baseline, candidate, min_behavior_pass_rate=0.75)

    assert verdict["passed"] is False
    behavior_check = next(check for check in verdict["checks"] if check["name"] == "behavior_suite")
    behavior_drop_check = next(check for check in verdict["checks"] if check["name"] == "behavior_suite_drop")
    assert behavior_check["passed"] is False
    assert behavior_drop_check["passed"] is False
    assert verdict["candidate"]["behavior_pass_rate"] == pytest.approx(0.25)


def test_gate_text_report_rejects_lower_ppl_worse_behavior_by_default() -> None:
    baseline = _report(100, 0.75)
    baseline["behavior_suite"] = {"pass_rate": 0.5, "correct": 3, "total": 6, "items": []}
    candidate = _report(80, 0.75)
    candidate["behavior_suite"] = {"pass_rate": 0.0, "correct": 0, "total": 6, "items": []}

    verdict = gate_text_report(baseline, candidate, max_ppl_ratio=1.25, min_choice_accuracy=0.75)

    assert verdict["passed"] is False
    ppl_check = next(check for check in verdict["checks"] if check["name"] == "perplexity_ratio")
    behavior_drop_check = next(check for check in verdict["checks"] if check["name"] == "behavior_suite_drop")
    assert ppl_check["passed"] is True
    assert behavior_drop_check["passed"] is False


def test_rank_evaluation_reports_prefers_behavior_over_ppl(tmp_path) -> None:
    low_ppl_collapsed = _report(90, 1.0)
    low_ppl_collapsed["model"] = "low-ppl-collapsed"
    low_ppl_collapsed["behavior_suite"] = {"pass_rate": 0.0, "correct": 0, "total": 6, "items": []}
    better_behavior = _report(150, 0.75)
    better_behavior["model"] = "better-behavior"
    better_behavior["behavior_suite"] = {"pass_rate": 0.5, "correct": 3, "total": 6, "items": []}
    output = tmp_path / "rank.json"

    report = rank_evaluation_reports([low_ppl_collapsed, better_behavior], output_file=output)

    assert output.exists()
    assert report["best"]["model"] == "better-behavior"
    assert report["candidates"][0]["behavior_pass_rate"] == pytest.approx(0.5)
    assert report["candidates"][1]["perplexity_mean"] == pytest.approx(90)


def test_behavior_probe_scores_first_number_exact_answer() -> None:
    probe = {
        "id": "multiply",
        "prompt": "Question: What is 12 * 13?\nAnswer:",
        "kind": "first_number",
        "answer": "156",
    }
    passed = _score_behavior_probe(probe, {"output": probe["prompt"] + " 156"})
    failed = _score_behavior_probe(probe, {"output": probe["prompt"] + " 155"})

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["details"]["prediction"] == "155"


def test_behavior_probe_scores_pattern_groups() -> None:
    probe = {
        "id": "tokenizer",
        "prompt": "Prompt:",
        "kind": "pattern_groups",
        "pattern_groups": [[r"\btokeni[sz]er\w*"], [r"\bmismatch\w*"], [r"\bembedding\w*"]],
    }
    result = _score_behavior_probe(
        probe,
        {"output": "Prompt: tokenizer mismatch can put embeddings and logits on the wrong rows."},
    )

    assert result["passed"] is True
    assert len(result["details"]["matched_groups"]) == 3


def test_generation_diagnostics_detects_repetition() -> None:
    diagnostics = _generation_diagnostics("the the the pruning pruning", [4, 4, 4, 7, 7])

    assert diagnostics["word_token_count"] == 5
    assert diagnostics["distinct_1"] == pytest.approx(0.4)
    assert diagnostics["top_token_share"] == pytest.approx(0.6)
    assert diagnostics["max_token_run"] == 3
    assert diagnostics["generated_max_token_id_run"] == 3


def test_ratio_tag_handles_alpha_values() -> None:
    assert _ratio_tag(1.25) == "1p2500"
    assert _ratio_tag(-0.5) == "m0p5000"


def test_finite_float_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="alpha must be finite"):
        _finite_float(float("nan"), "alpha")
    with pytest.raises(ValueError, match="alpha must be finite"):
        _finite_float(float("inf"), "alpha")


def test_unique_ratio_tags_rejects_collisions() -> None:
    with pytest.raises(ValueError, match="Duplicate alpha output tag"):
        _ensure_unique_ratio_tags([0.123440, 0.123441], "alpha")


class _Tokenizer:
    def __init__(self, vocab: dict[str, int], special_tokens_map: dict[str, str] | None = None) -> None:
        self._vocab = vocab
        self.special_tokens_map = special_tokens_map or {"eos_token": "</s>"}

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_tokenizer_compatibility_rejects_vocab_mismatch() -> None:
    left = _Tokenizer({"hello": 0, "world": 1})
    right = _Tokenizer({"hello": 0, "there": 1})
    with pytest.raises(ValueError, match="vocabularies differ"):
        _assert_tokenizers_compatible(left, right)


def test_tokenizer_compatibility_reports_special_token_differences() -> None:
    left = _Tokenizer({"hello": 0}, {"eos_token": "</s>"})
    right = _Tokenizer({"hello": 0}, {"eos_token": "<|endoftext|>"})
    report = _assert_tokenizers_compatible(left, right)
    assert report["vocab_match"] is True
    assert report["special_tokens_match"] is False
