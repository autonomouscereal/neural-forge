from __future__ import annotations

from neural_forge.hf_tools import _generation_from_cached_spec, _generation_health


def test_generation_from_cached_spec_preserves_expected_answer_metadata() -> None:
    spec = {
        "id": "generation_math_answer",
        "category": "math",
        "prompt": "Question: What is 6 + 4? Answer with only the number.\nAnswer:",
        "kind": "first_number",
        "answer": "10",
        "relevance_rules": [],
    }
    generated = {
        "prompt": spec["prompt"],
        "output": spec["prompt"] + " 6",
        "diagnostics": {
            "distinct_1": 1.0,
            "top_token_share": 0.0,
            "max_token_run": 0,
        },
    }

    item = _generation_from_cached_spec(generated, spec)
    health = _generation_health(item, min_continuation_chars=0, min_unique_token_ratio=0.0)

    assert item["kind"] == "first_number"
    assert item["answer"] == "10"
    assert health["expected_behavior"]["checked"] is True
    assert health["expected_behavior"]["passed"] is False
    assert health["expected_behavior"]["details"]["prediction"] == "6"
    assert health["passed"] is False


def test_generation_from_cached_spec_preserves_python_function_tests() -> None:
    spec = {
        "id": "generation_python_add_exact",
        "category": "code",
        "prompt": "Write a tiny Python function named add that returns a + b.\n```python\n",
        "kind": "python_function",
        "function_name": "add",
        "tests": [{"args": [2, 3], "expected": 5}],
        "relevance_rules": [],
    }
    generated = {
        "prompt": spec["prompt"],
        "output": spec["prompt"] + "def add(a, b):\n    return a - b",
        "diagnostics": {
            "distinct_1": 1.0,
            "top_token_share": 0.0,
            "max_token_run": 0,
        },
    }

    item = _generation_from_cached_spec(generated, spec)
    health = _generation_health(item, min_continuation_chars=0, min_unique_token_ratio=0.0)

    assert item["kind"] == "python_function"
    assert item["function_name"] == "add"
    assert health["expected_behavior"]["checked"] is True
    assert health["expected_behavior"]["passed"] is False
    assert health["expected_behavior"]["details"]["function_found"] is True
    assert health["passed"] is False
