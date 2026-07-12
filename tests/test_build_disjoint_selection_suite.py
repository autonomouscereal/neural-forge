from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "build_disjoint_selection_suite.py"
)
SPEC = importlib.util.spec_from_file_location("build_disjoint_selection_suite", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_selection_suite_namespaces_every_prompt_and_id(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    report = MODULE.build_selection_suite(
        output,
        namespace="selection-test",
        seed=17,
        text_count=3,
        choice_count=4,
        generation_count=5,
        behavior_count=6,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    marker = "[NEURAL_FORGE_SPLIT:selection-test]\n"

    assert report["counts"] == {
        "text_evals": 3,
        "choice_probes": 4,
        "generation_prompts": 5,
        "behavior_probes": 6,
    }
    assert report["suite_sha256"] == payload["suite_sha256"]
    assert all(item.startswith(marker) for item in payload["suite"]["text_evals"])
    for collection in ("choice_probes", "generation_prompts", "behavior_probes"):
        for item in payload["suite"][collection]:
            assert item["id"].startswith("selection-test:")
            assert item["prompt"].startswith(marker)


def test_build_selection_suite_rejects_unsafe_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="namespace"):
        MODULE.build_selection_suite(
            tmp_path / "selection.json",
            namespace="bad namespace",
            seed=0,
            text_count=1,
            choice_count=1,
            generation_count=1,
            behavior_count=1,
        )
