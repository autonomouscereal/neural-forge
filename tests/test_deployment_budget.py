from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from neural_forge.deployment_budget import (
    ArtifactAccountingError,
    ArtifactCategory,
    CategoryByteTotals,
    build_deployment_budget_report,
)


def _write(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_complete_artifact_tree_produces_immutable_budget_report(
    tmp_path: Path,
) -> None:
    contents = {
        "model/weights.bin": b"learned-weights",
        "model/assignments.idx": b"index",
        "tokenizer.json": b"tokenizer",
        "config.json": b"metadata",
        "runtime/kernel.bin": b"aux",
    }
    for relative_path, content in contents.items():
        _write(tmp_path, relative_path, content)

    categories = {
        "model/weights.bin": "learned_weights",
        "model/assignments.idx": ArtifactCategory.INDICES,
        "tokenizer.json": "tokenizer",
        "config.json": "metadata",
        "runtime/kernel.bin": "auxiliary",
    }
    deployable_bytes = sum(map(len, contents.values()))
    report = build_deployment_budget_report(
        tmp_path,
        categories,
        source_serialized_bytes=deployable_bytes * 8,
        unique_parameters=11,
        logical_parameters=37,
        flops_per_token=123.5,
        kv_cache_bytes_per_token=256,
        peak_load_bytes=4096,
        peak_runtime_bytes=3072,
    )

    assert report.artifact_root == str(tmp_path.resolve())
    assert report.file_count == len(contents)
    assert report.actual_total_serialized_bytes == deployable_bytes
    assert report.total_serialized_bytes == deployable_bytes
    assert report.category_totals.total_serialized_bytes == deployable_bytes
    assert report.source_to_deployable_byte_ratio == 8.0
    assert report.unique_parameters == 11
    assert report.logical_parameters == 37
    assert report.flops_per_token == 123.5
    assert report.kv_cache_bytes_per_token == 256
    assert report.peak_load_bytes == 4096
    assert report.peak_runtime_bytes == 3072
    with pytest.raises(FrozenInstanceError):
        report.actual_total_serialized_bytes = 1  # type: ignore[misc]


def test_category_sums_hashes_and_ratio_match_serialized_files(
    tmp_path: Path,
) -> None:
    weights = _write(tmp_path, "weights.bin", b"abc")
    _write(tmp_path, "nested/config.json", b"12345")

    report = build_deployment_budget_report(
        tmp_path,
        {
            weights: "learned_weights",
            "nested/config.json": "metadata",
        },
        source_serialized_bytes=32,
    )

    assert report.category_totals == CategoryByteTotals(
        learned_weights=3,
        indices=0,
        tokenizer=0,
        metadata=5,
        auxiliary=0,
    )
    assert report.category_totals["learned_weights"] == 3
    assert report.category_totals[ArtifactCategory.METADATA] == 5
    assert report.actual_total_serialized_bytes == 8
    assert report.source_to_deployable_byte_ratio == 4.0
    hashes = {file.relative_path: file.sha256 for file in report.files}
    assert hashes == {
        "nested/config.json": hashlib.sha256(b"12345").hexdigest(),
        "weights.bin": hashlib.sha256(b"abc").hexdigest(),
    }


def test_unclassified_hidden_file_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, "weights.bin", b"weights")
    _write(tmp_path, ".runtime-state", b"required")

    with pytest.raises(
        ArtifactAccountingError,
        match=r"Unclassified artifact files: \.runtime-state",
    ):
        build_deployment_budget_report(
            tmp_path,
            {"weights.bin": "learned_weights"},
            source_serialized_bytes=100,
        )


def test_symlink_escape_fails_without_hashing_target(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    outside = _write(tmp_path, "outside.bin", b"outside-secret")
    escape = artifact_root / "escape.bin"
    try:
        escape.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Filesystem does not permit symlink creation: {error}")

    with pytest.raises(
        ArtifactAccountingError,
        match="Unsafe symlink escapes artifact root",
    ):
        build_deployment_budget_report(
            artifact_root,
            {"escape.bin": "auxiliary"},
            source_serialized_bytes=100,
        )


def test_duplicate_path_aliases_are_rejected(tmp_path: Path) -> None:
    weights = _write(tmp_path, "weights.bin", b"weights")

    with pytest.raises(
        ArtifactAccountingError,
        match="Duplicate category mapping",
    ):
        build_deployment_budget_report(
            tmp_path,
            [
                ("weights.bin", "learned_weights"),
                (weights, "metadata"),
            ],
            source_serialized_bytes=100,
        )


def test_missing_classified_path_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "weights.bin", b"weights")

    with pytest.raises(
        ArtifactAccountingError,
        match="Classified artifact path does not exist",
    ):
        build_deployment_budget_report(
            tmp_path,
            {
                "weights.bin": "learned_weights",
                "missing.idx": "indices",
            },
            source_serialized_bytes=100,
        )


def test_outside_root_classification_is_rejected(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    _write(artifact_root, "weights.bin", b"weights")
    outside = _write(tmp_path, "outside.json", b"metadata")

    with pytest.raises(
        ArtifactAccountingError,
        match="outside artifact root",
    ):
        build_deployment_budget_report(
            artifact_root,
            {
                "weights.bin": "learned_weights",
                outside: "metadata",
            },
            source_serialized_bytes=100,
        )


def test_report_invariants_reject_mismatched_totals(tmp_path: Path) -> None:
    _write(tmp_path, "weights.bin", b"weights")
    report = build_deployment_budget_report(
        tmp_path,
        {"weights.bin": "learned_weights"},
        source_serialized_bytes=100,
    )
    mismatched_totals = replace(
        report.category_totals,
        metadata=report.category_totals.metadata + 1,
    )

    with pytest.raises(
        ArtifactAccountingError,
        match="Category byte totals do not match the per-file inventory",
    ):
        replace(report, category_totals=mismatched_totals)

    with pytest.raises(
        ArtifactAccountingError,
        match="Actual total serialized bytes do not match",
    ):
        replace(
            report,
            actual_total_serialized_bytes=(report.actual_total_serialized_bytes + 1),
        )
