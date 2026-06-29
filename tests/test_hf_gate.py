from __future__ import annotations

import pytest

from neural_forge.hf_tools import GENERATION_PROMPTS
from neural_forge.hf_tools import build_eval_suite
from neural_forge.hf_tools import _behavior_first_candidate_sort_key
from neural_forge.hf_tools import gate_text_report
from neural_forge.hf_tools import rank_evaluation_reports
from neural_forge.hf_tools import _assert_tokenizers_compatible
from neural_forge.hf_tools import _ensure_unique_ratio_tags
from neural_forge.hf_tools import _finite_float
from neural_forge.hf_tools import _generation_diagnostics
from neural_forge.hf_tools import _load_eval_suite
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


def test_gate_text_report_allows_advisory_ppl_regression_when_behavior_is_healthy() -> None:
    baseline = _report(100, 1.0)
    baseline["behavior_suite"] = {"pass_rate": 0.75, "correct": 3, "total": 4, "items": []}
    candidate = _report(300, 1.0)
    candidate["behavior_suite"] = {"pass_rate": 0.75, "correct": 3, "total": 4, "items": []}

    verdict = gate_text_report(
        baseline,
        candidate,
        max_ppl_ratio=1.25,
        ppl_gate_mode="advisory",
        min_behavior_pass_rate=0.75,
    )

    ppl_check = next(check for check in verdict["checks"] if check["name"] == "perplexity_ratio")
    assert verdict["passed"] is True
    assert ppl_check["passed"] is False
    assert ppl_check["hard"] is False
    assert verdict["candidate"]["generation_health_rate"] == pytest.approx(1.0)


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


def test_rank_evaluation_reports_uses_large_suite_correct_count_after_rate_tie(tmp_path) -> None:
    small = _report(100, 1.0)
    small["model"] = "small-suite"
    small["behavior_suite"] = {"pass_rate": 0.5, "correct": 5, "total": 10, "items": []}
    large = _report(120, 0.75)
    large["model"] = "large-suite"
    large["behavior_suite"] = {"pass_rate": 0.5, "correct": 500, "total": 1000, "items": []}

    report = rank_evaluation_reports([small, large], output_file=tmp_path / "rank.json")

    assert report["best"]["model"] == "large-suite"
    assert report["candidates"][0]["behavior_correct"] == 500


def test_rank_evaluation_reports_adds_category_breakdowns_and_suite_warnings(tmp_path) -> None:
    candidate = _report(120, 0.75)
    candidate["model"] = "candidate"
    candidate["eval_suite"] = {
        "source": "broad-a.json",
        "text_eval_count": 64,
        "choice_probe_count": 250,
        "generation_prompt_count": 2,
        "behavior_probe_count": 4,
    }
    candidate["behavior_suite"] = {
        "pass_rate": 0.5,
        "correct": 2,
        "total": 4,
        "items": [
            {"id": "synthetic_math_0000", "passed": True},
            {"id": "synthetic_python_add_0001", "passed": False},
            {"id": "synthetic_tokenizer_0002", "passed": True},
            {"id": "synthetic_merge_definition_0003", "passed": False},
        ],
    }
    candidate["generations"] = [
        {
            "id": "generation_python_add_0000",
            "prompt": "Write a tiny Python function that adds two numbers.",
            "output": "Write a tiny Python function that adds two numbers.\ndef add(a, b):\n    return a + b",
            "relevance_rules": [[r"\bdef\b"], [r"\+"], [r"\breturn\b"]],
        },
        {
            "id": "generation_tokenizer_mismatch_0001",
            "prompt": "In one sentence, why can tokenizer mismatch hurt model stitching?",
            "output": "In one sentence, why can tokenizer mismatch hurt model stitching?    ",
            "relevance_rules": [[r"\btokeni[sz]er\w*"]],
        },
    ]
    other_suite = _report(200, 0.5)
    other_suite["model"] = "other-suite"
    other_suite["eval_suite"] = {
        "source": "tiny.json",
        "text_eval_count": 4,
        "choice_probe_count": 4,
        "generation_prompt_count": 1,
        "behavior_probe_count": 6,
    }
    other_suite["behavior_suite"] = {"pass_rate": 0.0, "correct": 0, "total": 6, "items": []}

    report = rank_evaluation_reports([candidate, other_suite], output_file=tmp_path / "rank.json", min_continuation_chars=0)
    top = report["summary_rows"][0]

    assert top["model"] == "candidate"
    assert top["behavior"] == "2/4"
    assert top["generation"] == "1/2"
    assert top["behavior_category_breakdown"]["math"]["correct"] == 1
    assert top["behavior_category_breakdown"]["code"]["total"] == 1
    assert top["generation_category_breakdown"]["code"]["correct"] == 1
    assert any(warning["type"] == "mixed_eval_suite_counts" for warning in report["suite_warnings"])
    assert any(warning["type"] == "mixed_eval_suite_sources" for warning in report["suite_warnings"])


def test_rank_evaluation_reports_prefers_hard_category_coverage_after_behavior_tie(tmp_path) -> None:
    easy = _report(90, 1.0)
    easy["model"] = "easy-categories"
    easy["behavior_suite"] = {
        "pass_rate": 0.5,
        "correct": 2,
        "total": 4,
        "items": [
            {"id": "synthetic_tokenizer_0001", "passed": True},
            {"id": "synthetic_rain_0002", "passed": True},
            {"id": "synthetic_math_0000", "passed": False},
            {"id": "synthetic_python_add_0004", "passed": False},
        ],
    }
    hard = _report(130, 1.0)
    hard["model"] = "hard-categories"
    hard["behavior_suite"] = {
        "pass_rate": 0.5,
        "correct": 2,
        "total": 4,
        "items": [
            {"id": "synthetic_tokenizer_0001", "passed": False},
            {"id": "synthetic_rain_0002", "passed": False},
            {"id": "synthetic_math_0000", "passed": True},
            {"id": "synthetic_python_add_0004", "passed": True},
        ],
    }

    report = rank_evaluation_reports([easy, hard], output_file=tmp_path / "rank.json")

    assert report["best"]["model"] == "hard-categories"
    assert report["best"]["behavior_category_summary"]["hard_category_correct"] == 2
    assert report["summary_rows"][0]["behavior_category_summary"]["hard_category_correct"] == 2


def test_gate_text_report_reports_lost_behavior_and_generation_ids() -> None:
    baseline = _report(100, 1.0)
    baseline["behavior_suite"] = {
        "pass_rate": 1.0,
        "correct": 2,
        "total": 2,
        "items": [{"id": "kept", "passed": True}, {"id": "lost", "passed": True}],
    }
    baseline["generations"] = [
        {
            "id": "gen_lost",
            "prompt": "Custom prompt",
            "output": "Custom prompt relevant words",
            "relevance_rules": [[r"relevant"]],
        }
    ]
    candidate = _report(100, 1.0)
    candidate["behavior_suite"] = {
        "pass_rate": 0.5,
        "correct": 1,
        "total": 2,
        "items": [{"id": "kept", "passed": True}, {"id": "lost", "passed": False}],
    }
    candidate["generations"] = [
        {
            "id": "gen_lost",
            "prompt": "Custom prompt",
            "output": "Custom prompt off topic",
            "relevance_rules": [[r"relevant"]],
        }
    ]

    verdict = gate_text_report(baseline, candidate, max_behavior_drop=0.0, min_continuation_chars=0)

    assert verdict["passed"] is False
    assert verdict["candidate"]["lost_behavior_ids"] == ["lost"]
    assert verdict["candidate"]["lost_generation_ids"] == ["gen_lost"]


def test_eval_suite_builder_and_loader_round_trip(tmp_path) -> None:
    path = tmp_path / "suite.json"

    report = build_eval_suite(
        path,
        behavior_count=1000,
        generation_count=12,
        choice_count=20,
        text_count=8,
        seed=7,
    )
    loaded = _load_eval_suite(path)

    assert report["counts"]["behavior_probes"] == 1000
    assert len(loaded["behavior_probes"]) == 1000
    assert len(loaded["generation_prompts"]) == 12
    assert loaded["generation_prompts"][0]["relevance_rules"]
    assert len(loaded["choice_probes"]) == 20
    assert len(loaded["text_evals"]) == 8


def test_eval_suite_builder_preserves_explicit_zero_counts(tmp_path) -> None:
    path = tmp_path / "suite.json"

    build_eval_suite(
        path,
        behavior_count=0,
        generation_count=0,
        choice_count=0,
        text_count=0,
        seed=7,
    )
    loaded = _load_eval_suite(path)

    assert loaded["behavior_probes"] == []
    assert loaded["generation_prompts"] == []
    assert loaded["choice_probes"] == []
    assert loaded["text_evals"] == []


def test_eval_suite_loader_accepts_jsonl_rows(tmp_path) -> None:
    path = tmp_path / "suite.jsonl"
    rows = [
        {"type": "text_eval", "text": "A model should answer clearly."},
        {"type": "generation", "id": "gen", "prompt": "Say merge risk.", "relevance_rules": [[r"risk"]]},
        {"type": "choice", "id": "choice", "prompt": "Question: 1 + 1?\nAnswer:", "choices": [" 2", " 3"], "answer": 0},
        {"id": "math", "prompt": "Question: What is 2 + 3?\nAnswer:", "kind": "first_number", "answer": "5"},
    ]
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")

    suite = _load_eval_suite(path)

    assert suite["text_evals"] == ["A model should answer clearly."]
    assert suite["generation_prompts"][0]["id"] == "gen"
    assert suite["choice_probes"][0]["id"] == "choice"
    assert suite["behavior_probes"][0]["id"] == "math"


def test_behavior_first_candidate_sort_key_uses_shrink_and_ppl_only_after_output_ties() -> None:
    high_behavior_high_ppl = {
        "tag": "high-behavior",
        "shrink": 0.10,
        "metrics": {
            "perplexity_mean": 500.0,
            "choice_accuracy": 0.75,
            "behavior_pass_rate": 0.50,
            "generation_health_rate": 1.0,
        },
    }
    low_behavior_low_ppl = {
        "tag": "low-behavior",
        "shrink": 0.90,
        "metrics": {
            "perplexity_mean": 100.0,
            "choice_accuracy": 0.75,
            "behavior_pass_rate": 0.25,
            "generation_health_rate": 1.0,
        },
    }
    high_shrink_high_ppl_tie = {
        "tag": "high-shrink-tie",
        "shrink": 0.80,
        "metrics": {
            "perplexity_mean": 900.0,
            "choice_accuracy": 0.75,
            "behavior_pass_rate": 0.50,
            "generation_health_rate": 1.0,
        },
    }

    ranked = sorted([low_behavior_low_ppl, high_behavior_high_ppl], key=_behavior_first_candidate_sort_key)
    tied = sorted([high_behavior_high_ppl, high_shrink_high_ppl_tie], key=_behavior_first_candidate_sort_key)

    assert ranked[0]["tag"] == "high-behavior"
    assert tied[0]["tag"] == "high-shrink-tie"


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


def test_behavior_probe_scores_executable_python_add_function() -> None:
    probe = {
        "id": "python_add",
        "prompt": "```python\n",
        "kind": "python_function",
        "function_name": "add",
        "tests": [
            {"args": [2, 3], "expected": 5},
            {"args": [-1, 5], "expected": 4},
        ],
    }

    result = _score_behavior_probe(
        probe,
        {"output": "```python\ndef add(a, b):\n    return a + b\n```"},
    )

    assert result["passed"] is True
    assert all(item["passed"] for item in result["details"]["test_results"])


def test_behavior_probe_rejects_malformed_python_like_text() -> None:
    probe = {
        "id": "python_add",
        "prompt": "```python\n",
        "kind": "python_function",
        "function_name": "add",
        "tests": [{"args": [2, 3], "expected": 5}],
    }

    result = _score_behavior_probe(
        probe,
        {"output": "```python\npythondef add(numbers(a, b): return a + b 3110 + 3200"},
    )

    assert result["passed"] is False
    assert result["details"]["reason"] in {"syntax_error", "function_not_found"}


def test_behavior_probe_rejects_untrusted_python_body() -> None:
    probe = {
        "id": "python_add",
        "prompt": "```python\n",
        "kind": "python_function",
        "function_name": "add",
        "tests": [{"args": [2, 3], "expected": 5}],
    }

    result = _score_behavior_probe(
        probe,
        {"output": "```python\ndef add(a, b):\n    return __import__('os').system('echo bad')\n```"},
    )

    assert result["passed"] is False
    assert result["details"]["test_results"][0]["reason"] == "unsupported_call"


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
