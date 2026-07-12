from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "compare_eval_retention.py"
    spec = importlib.util.spec_from_file_location("compare_eval_retention", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_eval_retention_includes_manual_samples(tmp_path: Path) -> None:
    compare_module = _load_compare_module()
    parent = {
        "model": "parent",
        "param_count": 400,
        "behavior_suite": {
            "correct": 2,
            "total": 2,
            "items": [
                {
                    "id": "math_add",
                    "category": "math",
                    "kind": "first_number",
                    "prompt": "Question: 2+2?\nAnswer:",
                    "passed": True,
                    "continuation": "4",
                    "details": {"answer": "4", "prediction": "4"},
                },
                {
                    "id": "code_add",
                    "category": "code",
                    "kind": "python_function",
                    "prompt": "Write add.\n```python\n",
                    "passed": True,
                    "continuation": "def add(a, b):\n    return a + b",
                },
            ],
        },
        "choice_accuracy": {"correct": 0, "total": 0, "items": []},
        "generation_health": {"correct": 1, "total": 1},
        "generation_diagnostics": [
            {
                "id": "gen_math",
                "category": "math",
                "passed": True,
                "expected_behavior": {"details": {"answer": "7", "prediction": "7"}},
            }
        ],
        "generations": [
            {
                "id": "gen_math",
                "category": "math",
                "prompt": "Question: 3+4?\nAnswer:",
                "output": "Question: 3+4?\nAnswer: 7",
            }
        ],
    }
    child = {
        "model": "child",
        "param_count": 100,
        "behavior_suite": {
            "correct": 1,
            "total": 2,
            "items": [
                {
                    "id": "math_add",
                    "category": "math",
                    "kind": "first_number",
                    "prompt": "Question: 2+2?\nAnswer:",
                    "passed": False,
                    "continuation": "5",
                    "details": {"answer": "4", "prediction": "5"},
                },
                {
                    "id": "code_add",
                    "category": "code",
                    "kind": "python_function",
                    "prompt": "Write add.\n```python\n",
                    "passed": True,
                    "continuation": "def add(a, b):\n    return a + b",
                },
            ],
        },
        "choice_accuracy": {"correct": 0, "total": 0, "items": []},
        "generation_health": {"correct": 0, "total": 1},
        "generation_diagnostics": [
            {
                "id": "gen_math",
                "category": "math",
                "passed": False,
                "expected_behavior": {"details": {"answer": "7", "prediction": "8"}},
            }
        ],
        "generations": [
            {
                "id": "gen_math",
                "category": "math",
                "prompt": "Question: 3+4?\nAnswer:",
                "output": "Question: 3+4?\nAnswer: 8",
            }
        ],
    }
    parent_path = tmp_path / "parent.json"
    child_path = tmp_path / "child.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    child_path.write_text(json.dumps(child), encoding="utf-8")

    report = compare_module.compare(parent_path, child_path, max_samples=2)

    assert report["param_compression_ratio"] == 4
    assert report["behavior"]["overall"]["lost_parent_pass"] == 1
    behavior_sample = report["manual_samples"]["behavior"][0]
    generation_sample = report["manual_samples"]["generation"][0]
    assert behavior_sample["id"] == "math_add"
    assert behavior_sample["parent_continuation"] == "4"
    assert behavior_sample["child_continuation"] == "5"
    assert behavior_sample["child_details"]["prediction"] == "5"
    assert generation_sample["id"] == "gen_math"
    assert generation_sample["parent_continuation"] == "7"
    assert generation_sample["child_continuation"] == "8"
    assert generation_sample["child_details"]["prediction"] == "8"
