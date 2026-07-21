from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.compile_lfm2_target_tokenizer_bridge import (  # noqa: E402
    Lfm2TargetTokenizerCompilationError,
    REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
    build_parser,
    collect_calibration_texts,
    load_required_donor_token_ids,
    resolve_donor_row_capacity,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_required_donor_ids_plain_list_is_file_hash_bound_and_normalized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "required.json"
    source.write_text(json.dumps([9, 2, 7]), encoding="utf-8")

    ids, report = load_required_donor_token_ids(
        source,
        expected_file_sha256=_file_sha256(source),
    )

    assert ids == (2, 7, 9)
    assert report["file_sha256"] == _file_sha256(source)
    assert report["required_donor_token_ids"] == [2, 7, 9]
    assert report["required_donor_token_ids_sha256"] == _canonical_sha256(
        [2, 7, 9]
    )


def test_required_donor_ids_sealed_manifest_is_verified(tmp_path: Path) -> None:
    body = {
        "schema": REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
        "source_caches": [{"cache_key": "a" * 64}],
        "required_donor_token_ids": [2, 7, 9],
        "required_donor_token_count": 3,
        "required_donor_token_ids_sha256": _canonical_sha256([2, 7, 9]),
    }
    source = tmp_path / "required-manifest.json"
    source.write_text(
        json.dumps(
            {**body, "manifest_sha256": _canonical_sha256(body)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ids, report = load_required_donor_token_ids(
        source,
        expected_file_sha256=_file_sha256(source),
    )

    assert ids == (2, 7, 9)
    assert report["payload_schema"] == REQUIRED_DONOR_TOKEN_IDS_SCHEMA


def test_required_donor_ids_fail_closed_on_binding_and_manifest_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "required.json"
    source.write_text(json.dumps([2, 7, 9]), encoding="utf-8")

    with pytest.raises(
        Lfm2TargetTokenizerCompilationError,
        match="requires an expected file SHA-256",
    ):
        load_required_donor_token_ids(source, expected_file_sha256=None)
    with pytest.raises(
        Lfm2TargetTokenizerCompilationError,
        match="differs from its expected SHA-256",
    ):
        load_required_donor_token_ids(
            source,
            expected_file_sha256="0" * 64,
        )

    body = {
        "schema": REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
        "required_donor_token_ids": [2, 7, 9],
        "required_donor_token_count": 3,
        "required_donor_token_ids_sha256": _canonical_sha256([2, 7, 9]),
    }
    source.write_text(
        json.dumps({**body, "manifest_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(
        Lfm2TargetTokenizerCompilationError,
        match="manifest seal is invalid",
    ):
        load_required_donor_token_ids(
            source,
            expected_file_sha256=_file_sha256(source),
        )


def test_required_donor_ids_cli_requires_explicit_hash_argument() -> None:
    args = build_parser().parse_args(
        [
            "--donor-tokenizer-directory",
            "donor",
            "--output-directory",
            "target",
            "--receipt",
            "receipt.json",
            "--target-vocab-size",
            "4096",
            "--required-donor-token-ids-json",
            "required.json",
            "--required-donor-token-ids-sha256",
            "a" * 64,
        ]
    )

    assert args.required_donor_token_ids_json == Path("required.json")
    assert args.required_donor_token_ids_sha256 == "a" * 64
