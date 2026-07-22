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


def _replace_argument(arguments: list[str], name: str, value: Path) -> list[str]:
    updated = list(arguments)
    updated[updated.index(name) + 1] = str(value)
    return updated


def test_atomic_receipt_publication_never_replaces_a_concurrent_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "receipt.json"

    def concurrent_publish(_source, destination, *, follow_symlinks):
        assert follow_symlinks is False
        Path(destination).write_bytes(b"concurrent-owner\n")
        raise FileExistsError(destination)

    monkeypatch.setattr(runner.os, "link", concurrent_publish)

    with pytest.raises(FileExistsError):
        runner._atomic_json(output, {"result_sha256": "f" * 64})

    assert output.read_bytes() == b"concurrent-owner\n"


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


@pytest.mark.parametrize("collision", ("same-output", "nested-source", "receipt"))
def test_cli_rejects_capsule_and_receipt_path_collisions_before_asset_load(
    tmp_path: Path,
    monkeypatch,
    collision: str,
) -> None:
    def unexpected_asset_load(*_args, **_kwargs):
        pytest.fail("path collision reached tokenizer or artifact loading")

    monkeypatch.setattr(runner, "load_local_tokenizer", unexpected_asset_load)
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        unexpected_asset_load,
    )
    arguments = _required_arguments(tmp_path)
    if collision == "same-output":
        arguments.extend(["--preterminal-capsule", str(tmp_path / "output")])
    elif collision == "nested-source":
        arguments.extend(["--preterminal-capsule", str(tmp_path / "input" / "capsule")])
    else:
        arguments = _replace_argument(
            arguments,
            "--receipt-json",
            tmp_path / "output",
        )

    with pytest.raises(ValueError, match="must be disjoint"):
        runner.main(arguments)


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
            "--body-recipe",
            "residual_writer",
        ]
    )

    assert status == 0
    assert len(calls) == 1
    assert calls[0]["interface_statistics_directory"] == str(statistics_directory)
    assert calls[0]["expected_interface_statistics_sha256"] == "a" * 64
    assert calls[0]["body_recipe"] == "residual_writer"
    assert written == [(tmp_path / "receipt.json", {"result_sha256": "f" * 64})]


@pytest.mark.parametrize(
    "missing",
    (
        "--interface-frontier-companion-json",
        "--expected-interface-frontier-companion-file-sha256",
        "--expected-interface-frontier-companion-sha256",
        "--anchor-candidate-index",
    ),
)
def test_cli_rejects_incomplete_frontier_companion_set_before_assets(
    tmp_path: Path,
    monkeypatch,
    missing: str,
) -> None:
    monkeypatch.setattr(
        runner,
        "load_local_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("incomplete companion loaded assets"),
    )
    companion = [
        "--interface-frontier-companion-json",
        str(tmp_path / "companion.json"),
        "--expected-interface-frontier-companion-file-sha256",
        "b" * 64,
        "--expected-interface-frontier-companion-sha256",
        "c" * 64,
        "--anchor-candidate-index",
        "9",
    ]
    position = companion.index(missing)
    del companion[position : position + 2]
    with pytest.raises(ValueError, match="must be supplied together"):
        runner.main(
            [
                *_required_arguments(tmp_path),
                "--interface-statistics-directory",
                str(tmp_path / "statistics"),
                "--expected-interface-statistics-sha256",
                "a" * 64,
                *companion,
            ]
        )


def test_cli_forwards_exact_frontier_companion_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(runner, "load_local_tokenizer", lambda path: str(path))
    monkeypatch.setattr(
        runner,
        "run_lfm2_semantic_postfold",
        lambda **kwargs: calls.append(kwargs) or {"result_sha256": "f" * 64},
    )
    monkeypatch.setattr(runner, "_atomic_json", lambda *_args, **_kwargs: None)
    companion_path = tmp_path / "companion.json"

    assert runner.main(
        [
            *_required_arguments(tmp_path),
            "--interface-statistics-directory",
            str(tmp_path / "statistics"),
            "--expected-interface-statistics-sha256",
            "a" * 64,
            "--interface-frontier-companion-json",
            str(companion_path),
            "--expected-interface-frontier-companion-file-sha256",
            "b" * 64,
            "--expected-interface-frontier-companion-sha256",
            "c" * 64,
            "--anchor-candidate-index",
            "9",
        ]
    ) == 0
    assert calls[0]["interface_frontier_companion_json"] == str(companion_path)
    assert calls[0]["expected_interface_frontier_companion_file_sha256"] == "b" * 64
    assert calls[0]["expected_interface_frontier_companion_sha256"] == "c" * 64
    assert calls[0]["anchor_candidate_index"] == 9


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
    assert args.interface_frontier_companion_json is None
    assert args.expected_interface_frontier_companion_file_sha256 is None
    assert args.expected_interface_frontier_companion_sha256 is None
    assert args.anchor_candidate_index is None
    assert args.body_recipe == "fit_guided"
