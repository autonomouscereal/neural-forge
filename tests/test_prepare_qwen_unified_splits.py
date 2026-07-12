from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "prepare_qwen_unified_splits.py"
SPEC = importlib.util.spec_from_file_location("prepare_qwen_unified_splits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _prompts(payload):
    result = set(payload["text_evals"])
    for collection in ("choice_probes", "generation_prompts", "behavior_probes"):
        result.update(str(item["prompt"]) for item in payload[collection])
    return result


def test_prepare_creates_four_deterministic_prompt_disjoint_splits(tmp_path) -> None:
    calibration_source = tmp_path / "calibration.jsonl"
    calibration_source.write_text(
        "".join(json.dumps(f"calibration prompt {index}") + "\n" for index in range(12)),
        encoding="utf-8",
    )
    suite_source = tmp_path / "suite.json"
    suite_source.write_text(
        json.dumps(
            {
                "text_evals": [f"text eval {index}" for index in range(8)],
                "choice_probes": [
                    {
                        "id": f"choice-{index}",
                        "prompt": f"choice prompt {index}",
                        "choices": ["a", "b"],
                        "answer": index % 2,
                        "category": f"category-{index % 2}",
                    }
                    for index in range(12)
                ],
                "generation_prompts": [
                    {
                        "id": f"generation-{index}",
                        "prompt": f"generation prompt {index}",
                        "category": f"category-{index % 3}",
                    }
                    for index in range(12)
                ],
                "behavior_probes": [
                    {
                        "id": f"behavior-{index}",
                        "prompt": f"behavior prompt {index}",
                        "kind": "contains",
                        "expected": "answer",
                        "category": f"category-{index % 2}",
                    }
                    for index in range(12)
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    args = module.build_parser().parse_args(
        [
            "--calibration-source",
            str(calibration_source),
            "--eval-suite-source",
            str(suite_source),
            "--output-dir",
            str(output),
            "--calibration-count",
            "3",
            "--sensitivity-gate-count",
            "3",
            "--candidate-text-count",
            "2",
            "--candidate-choice-count",
            "3",
            "--candidate-generation-count",
            "3",
            "--candidate-behavior-count",
            "3",
            "--transaction-text-count",
            "2",
            "--transaction-choice-count",
            "4",
            "--transaction-generation-count",
            "4",
            "--transaction-behavior-count",
            "4",
        ]
    )

    first = module.prepare(args)
    first_bytes = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    second = module.prepare(args)

    assert first == second
    assert first["prompt_overlap_count"] == 0
    assert first_bytes == {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    calibration = {
        json.loads(line)
        for line in (output / "calibration.jsonl").read_text(encoding="utf-8").splitlines()
    }
    sensitivity = {
        json.loads(line)
        for line in (output / "sensitivity_gate.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    candidate = json.loads(
        (output / "candidate_validation_suite.json").read_text(encoding="utf-8")
    )
    transaction = json.loads(
        (output / "transaction_gate_suite.json").read_text(encoding="utf-8")
    )
    candidate_prompts = _prompts(candidate)
    transaction_prompts = _prompts(transaction)
    assert calibration.isdisjoint(sensitivity)
    assert calibration.isdisjoint(candidate_prompts | transaction_prompts)
    assert sensitivity.isdisjoint(candidate_prompts | transaction_prompts)
    assert candidate_prompts.isdisjoint(transaction_prompts)
