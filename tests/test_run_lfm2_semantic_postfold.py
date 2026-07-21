from __future__ import annotations

from pathlib import Path

import pytest

import experiments.run_lfm2_semantic_postfold as runner


def _required_arguments(tmp_path: Path) -> list[str]:
    return [
        "--input-checkpoint",
        str(tmp_path / "input"),
        "--expected-input-artifact-sha256",
        "1" * 64,
        "--output-checkpoint",
        str(tmp_path / "output"),
        "--receipt-json",
        str(tmp_path / "receipt.json"),
        "--trace-cache",
        str(tmp_path / "cache"),
        "--expected-cache-key",
        "2" * 64,
        "--bridge-directory",
        str(tmp_path / "bridge"),
        "--expected-bridge-artifact-sha256",
        "3" * 64,
        "--donor-tokenizer-directory",
        str(tmp_path / "donor-tokenizer"),
        "--target-tokenizer-directory",
        str(tmp_path / "target-tokenizer"),
        "--parent-model-directory",
        str(tmp_path / "parent"),
        "--compiler-diagnostic-json",
        str(tmp_path / "foundation.json"),
        "--expected-compiler-diagnostic-sha256",
        "4" * 64,
        "--device",
        "cpu",
    ]


@pytest.mark.parametrize(
    "unpaired",
    (
        ["--interface-statistics-directory", "statistics"],
        ["--expected-interface-statistics-sha256", "a" * 64],
    ),
)
def test_cli_rejects_unpaired_statistics_before_loading_assets(
    tmp_path: Path,
    monkeypatch,
    unpaired: list[str],
) -> None:
    def unexpected_asset_load(*_args, **_kwargs):
        pytest.fail("unpaired CLI statistics reached asset loading")

    monkeypatch.setattr(runner, "load_local_tokenizer", unexpected_asset_load)
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        unexpected_asset_load,
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        runner.main([*_required_arguments(tmp_path), *unpaired])


def test_cli_forwards_exact_compact_statistics_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []
    written = []

    monkeypatch.setattr(
        runner,
        "load_local_tokenizer",
        lambda path: ("tokenizer", str(path)),
    )

    def run_postfold(**kwargs):
        calls.append(kwargs)
        return {"result_sha256": "f" * 64}

    monkeypatch.setattr(runner, "run_lfm2_semantic_postfold", run_postfold)
    monkeypatch.setattr(
        runner,
        "_atomic_json",
        lambda path, value: written.append((path, value)),
    )
    statistics_directory = tmp_path / "statistics" / ("a" * 64)

    status = runner.main(
        [
            *_required_arguments(tmp_path),
            "--interface-statistics-directory",
            str(statistics_directory),
            "--expected-interface-statistics-sha256",
            "a" * 64,
        ]
    )

    assert status == 0
    assert len(calls) == 1
    assert calls[0]["interface_statistics_directory"] == str(statistics_directory)
    assert calls[0]["expected_interface_statistics_sha256"] == "a" * 64
    assert written == [(tmp_path / "receipt.json", {"result_sha256": "f" * 64})]


def test_cli_omits_compact_statistics_keywords_by_default() -> None:
    args = runner._parser().parse_args(
        [
            "--input-checkpoint",
            "input",
            "--expected-input-artifact-sha256",
            "1" * 64,
            "--output-checkpoint",
            "output",
            "--receipt-json",
            "receipt.json",
            "--trace-cache",
            "cache",
            "--expected-cache-key",
            "2" * 64,
            "--bridge-directory",
            "bridge",
            "--expected-bridge-artifact-sha256",
            "3" * 64,
            "--donor-tokenizer-directory",
            "donor",
            "--target-tokenizer-directory",
            "target",
            "--parent-model-directory",
            "parent",
            "--compiler-diagnostic-json",
            "compiler.json",
            "--expected-compiler-diagnostic-sha256",
            "4" * 64,
        ]
    )

    assert args.interface_statistics_directory is None
    assert args.expected_interface_statistics_sha256 is None
