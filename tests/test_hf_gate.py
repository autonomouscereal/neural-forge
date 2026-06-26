from __future__ import annotations

from neural_forge.hf_tools import gate_text_report


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

