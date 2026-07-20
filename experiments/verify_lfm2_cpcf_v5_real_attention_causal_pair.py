#!/usr/bin/env python3
"""Authenticate an H776 off/off versus on/on residual-attention pair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
for _path in (_REPOSITORY_ROOT, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class CausalPairVerifierCliError(RuntimeError):
    """The causal-pair verifier CLI failed before issuing a verdict."""


def _local_file(value: Path, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be pathlib.Path.")
    raw = os.fspath(value.expanduser())
    if not raw or "://" in raw or raw.startswith(("\\\\", "//")):
        raise CausalPairVerifierCliError(f"{name} must be one explicit local file.")
    try:
        path = value.expanduser().resolve(strict=True)
    except OSError as exc:
        raise CausalPairVerifierCliError(f"{name} does not exist.") from exc
    if path.is_symlink() or not path.is_file():
        raise CausalPairVerifierCliError(f"{name} must be one nonsymlink regular file.")
    return path


def _destination(
    value: Path,
    *,
    inputs: Sequence[Path],
) -> Path:
    if not isinstance(value, Path):
        raise TypeError("verdict_json must be pathlib.Path.")
    raw = os.fspath(value.expanduser())
    if not raw or "://" in raw or raw.startswith(("\\\\", "//")):
        raise CausalPairVerifierCliError(
            "verdict_json must be one explicit local path."
        )
    path = value.expanduser().absolute().resolve(strict=False)
    if path in inputs or path.exists() or path.is_symlink():
        raise CausalPairVerifierCliError(
            "verdict_json must be a new path disjoint from both inputs."
        )
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise CausalPairVerifierCliError(
            "verdict_json parent directory does not exist."
        ) from exc
    if parent.is_symlink() or not parent.is_dir():
        raise CausalPairVerifierCliError(
            "verdict_json parent must be a nonsymlink directory."
        )
    return path


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CausalPairVerifierCliError(f"{name} is not readable JSON.") from exc
    if not isinstance(value, dict):
        raise CausalPairVerifierCliError(f"{name} must contain one JSON object.")
    return value


def _write_create_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise CausalPairVerifierCliError(
            "Could not create the verdict atomically."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-diagnostic-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--combined-diagnostic-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--verdict-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline_path = _local_file(
            args.baseline_diagnostic_json,
            "baseline_diagnostic_json",
        )
        combined_path = _local_file(
            args.combined_diagnostic_json,
            "combined_diagnostic_json",
        )
        if baseline_path == combined_path:
            raise CausalPairVerifierCliError(
                "Baseline and combined diagnostics must be distinct files."
            )
        output = _destination(
            args.verdict_json,
            inputs=(baseline_path, combined_path),
        )
        from neural_forge.hf_lfm2_cpcf_v5_real_attention_causal_pair import (
            verify_lfm2_cpcf_v5_real_attention_causal_pair,
        )

        verdict = verify_lfm2_cpcf_v5_real_attention_causal_pair(
            _read_json(baseline_path, "baseline_diagnostic_json"),
            _read_json(combined_path, "combined_diagnostic_json"),
        )
        _write_create_new(output, verdict.as_dict())
    except Exception as exc:
        print(
            f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "passed": verdict.passed,
                "report_sha256": verdict.report["report_sha256"],
                "verdict_json": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
