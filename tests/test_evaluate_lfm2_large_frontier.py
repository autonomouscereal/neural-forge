from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.evaluate_lfm2_large_frontier import (
    Lfm2LargeFrontierEvaluationError,
    _checkpoint_accounting,
    _load_sealed_fold_report,
    build_parser,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256


def _write_report(path: Path, body: dict) -> str:
    seal = canonical_sha256(body)
    path.write_text(
        json.dumps({**body, "report_sha256": seal}),
        encoding="utf-8",
    )
    return seal


def test_loads_only_complete_canonically_sealed_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    body = {
        "status": "complete",
        "completed_layers": [0, 1],
        "final_checkpoint": {},
        "accounting": {},
    }
    seal = _write_report(path, body)

    document, observed = _load_sealed_fold_report(path)

    assert observed == seal
    assert document["completed_layers"] == [0, 1]

    document["completed_layers"].append(2)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        Lfm2LargeFrontierEvaluationError,
        match="seal",
    ):
        _load_sealed_fold_report(path)


def test_checkpoint_accounting_requires_real_positive_compaction(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    state = "1" * 64
    document = {
        "final_checkpoint": {
            "path": str(child),
            "state_sha256": state,
            "unique_parameters": 80,
        },
        "accounting": {
            "observed_parent_unique_parameters": 100,
            "observed_target_unique_parameters": 80,
        },
    }

    assert _checkpoint_accounting(document) == (child, state, 80, 100)

    document["accounting"]["observed_parent_unique_parameters"] = 80
    with pytest.raises(
        Lfm2LargeFrontierEvaluationError,
        match="positive parameter removal",
    ):
        _checkpoint_accounting(document)


def test_parser_requires_all_bound_evidence_paths(tmp_path: Path) -> None:
    digest = "a" * 64
    args = build_parser().parse_args(
        [
            "--parent-model",
            str(tmp_path / "parent"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--run-root",
            str(tmp_path / "run"),
            "--envelope",
            str(tmp_path / "envelope.json"),
            "--envelope-sha256",
            digest,
            "--parent-cache-key",
            digest,
            "--output",
            str(tmp_path / "evaluation.json"),
        ]
    )

    assert args.envelope_sha256 == digest
    assert args.parent_cache_key == digest
    assert args.device == "cuda:0"
