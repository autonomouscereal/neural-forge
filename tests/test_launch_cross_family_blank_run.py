from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "experiments" / "launch_cross_family_blank_run.py"
SPEC = importlib.util.spec_from_file_location("launch_cross_family_blank_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_text_corpus_accepts_text_prompt_and_string_rows(tmp_path) -> None:
    source = tmp_path / "calibration.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"text": "first"}),
                json.dumps({"prompt": "second"}),
                json.dumps("third"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert MODULE.load_text_corpus(source) == ["first", "second", "third"]


def test_prompt_split_is_deterministic_disjoint_and_spans_corpus() -> None:
    texts = [f"prompt-{index}" for index in range(100)]

    first = MODULE.deterministic_prompt_split(texts, 64, 16)
    second = MODULE.deterministic_prompt_split(texts, 64, 16)

    assert first == second
    calibration, validation, report = first
    assert len(calibration) == 64
    assert len(validation) == 16
    assert set(calibration).isdisjoint(validation)
    assert len(set(report["calibration_indices"] + report["validation_indices"])) == 80
    assert min(report["calibration_indices"] + report["validation_indices"]) == 0
    assert max(report["calibration_indices"] + report["validation_indices"]) == 99


def test_prompt_split_and_jsonl_fail_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="Need 5 prompts"):
        MODULE.deterministic_prompt_split(["one"], 4, 1)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"missing": "text"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="has no nonempty text"):
        MODULE.load_text_corpus(malformed)


def test_run_threads_nonlinear_merge_method_and_reports_it(
    tmp_path, monkeypatch
) -> None:
    calibration_file = tmp_path / "calibration.jsonl"
    calibration_file.write_text('"calibration"\n"validation"\n', encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_bootstrap(donor, **kwargs):
        captured["donor"] = donor
        captured.update(kwargs)
        return {
            "accepted": False,
            "compression": {"unique": {"source_to_child_ratio": 2.0}},
        }

    monkeypatch.setattr(MODULE, "bootstrap_blank_decoder_checkpoint", fake_bootstrap)
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "donor-model",
            "--run-root",
            str(tmp_path / "runs"),
            "--tag",
            "medoid",
            "--calibration-file",
            str(calibration_file),
            "--eval-suite",
            str(tmp_path / "eval.json"),
            "--cached-parent-eval",
            str(tmp_path / "parent.json"),
            "--calibration-count",
            "1",
            "--validation-count",
            "1",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "12",
            "--num-hidden-layers",
            "2",
            "--num-attention-heads",
            "2",
            "--num-key-value-heads",
            "1",
            "--head-dim",
            "4",
            "--max-position-embeddings",
            "32",
            "--nonlinear-merge-method",
            "real_medoid",
        ]
    )

    report = MODULE.run(args)

    assert captured["donor"] == "donor-model"
    assert captured["nonlinear_merge_method"] == "real_medoid"
    assert report["nonlinear_merge_method"] == "real_medoid"
    saved = json.loads(
        (tmp_path / "runs" / "medoid" / "run_report.json").read_text(encoding="utf-8")
    )
    assert saved == report
