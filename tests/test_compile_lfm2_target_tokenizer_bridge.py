from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.compile_lfm2_target_tokenizer_bridge import (  # noqa: E402
    Lfm2TargetTokenizerCompilationError,
    collect_calibration_texts,
    resolve_donor_row_capacity,
)


def test_collect_calibration_texts_is_deterministic_and_sealed(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            (
                json.dumps({"z": "third", "a": ["first", "second"]}),
                json.dumps({"text": "duplicate"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"text": ["duplicate", "fourth"]}) + "\n",
        encoding="utf-8",
    )

    texts, report = collect_calibration_texts(
        jsonl_paths=(second, first),
        parquet_paths=(),
        maximum_texts=10,
        maximum_characters=100,
    )
    replay_texts, replay_report = collect_calibration_texts(
        jsonl_paths=(first, second),
        parquet_paths=(),
        maximum_texts=10,
        maximum_characters=100,
    )

    assert texts == ("first", "second", "third", "duplicate", "fourth")
    assert replay_texts == texts
    assert replay_report == report
    assert report["selected_texts"] == 5
    assert report["evaluation_prompts_used"] is False
    assert len(report["corpus_sha256"]) == 64


def test_collect_calibration_texts_enforces_limits(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"text": ["abcdef", "second"]}) + "\n",
        encoding="utf-8",
    )

    texts, report = collect_calibration_texts(
        jsonl_paths=(source,),
        parquet_paths=(),
        maximum_texts=10,
        maximum_characters=4,
    )

    assert texts == ("abcd",)
    assert report["selected_characters"] == 4
    assert report["limit_reached"] is True


def test_collect_calibration_texts_rejects_empty_sources() -> None:
    with pytest.raises(
        Lfm2TargetTokenizerCompilationError,
        match="At least one independent",
    ):
        collect_calibration_texts(
            jsonl_paths=(),
            parquet_paths=(),
            maximum_texts=10,
            maximum_characters=100,
        )


def test_donor_row_capacity_uses_physical_config_padding(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 65_536}),
        encoding="utf-8",
    )

    assert (
        resolve_donor_row_capacity(
            tmp_path,
            explicit_capacity=None,
            tokenizer_size=64_402,
        )
        == 65_536
    )
    assert (
        resolve_donor_row_capacity(
            tmp_path,
            explicit_capacity=70_000,
            tokenizer_size=64_402,
        )
        == 70_000
    )
    with pytest.raises(
        Lfm2TargetTokenizerCompilationError,
        match="cannot be below",
    ):
        resolve_donor_row_capacity(
            tmp_path,
            explicit_capacity=60_000,
            tokenizer_size=64_402,
        )
