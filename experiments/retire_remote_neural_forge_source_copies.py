#!/usr/bin/env python3
"""Move legacy run evidence, then retire archived remote source copies safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE = Path("/home/cereal/model-surgery-lab")
CANONICAL = BASE / "neural-forge-src"
DESTINATION = CANONICAL / "runs" / "legacy-source-copies"


@dataclass(frozen=True, slots=True)
class SourceCopy:
    name: str
    archive_commit: str
    runs_sha256: str | None = None
    marker_name: str | None = None
    marker_value: str | None = None


COPIES = (
    SourceCopy(
        "neural-forge-behavior-vo-tuner-20260711",
        "eb9c37ad45927938946c120659136d7f8517159b",
        "d95078b389ba4ba7f4c125774cac0b3ade4d745a0d05671b10ea9e3f3a22eeb7",
        ".deployed_revision",
        "a9daf7b",
    ),
    SourceCopy(
        "neural-forge-cached-attention-smoke-20260711",
        "db34a1f0e6fb08545eca3cc60a0bbd689a4ac0e5",
    ),
    SourceCopy(
        "neural-forge-pivot",
        "fa9ca7cf34c043b73bc67c86392b3be820e821cf",
        "b858ccd9c64049eb717d9ff68fbdbc3b3af13fcc84663fa891f28940383322f6",
        ".deployed_revision",
        "a9daf7b",
    ),
    SourceCopy(
        "neural-forge-residual-attn-20260711",
        "ce492296cd3dd278673a54bd21d53d9aecb34004",
        "953fedf18d51354f21476f00afb997a88f04eaf5225cf20ab164979d56dc7876",
        ".deployed_revision",
        "a9daf7b",
    ),
    SourceCopy(
        "neural-forge-residual-attn-functional-20260711",
        "0d7d9150fe5e342b2654bc158a3d70462c928dda",
        "953fedf18d51354f21476f00afb997a88f04eaf5225cf20ab164979d56dc7876",
        ".deployed_revision",
        "a9daf7b",
    ),
    SourceCopy(
        "neural-forge-role-scale-functional-20260711",
        "c24fa1f430fc25a37f12e3ab7c01a3ea0490c813",
        "953fedf18d51354f21476f00afb997a88f04eaf5225cf20ab164979d56dc7876",
        ".deployed_revision",
        "a9daf7b",
    ),
    SourceCopy(
        "neural-forge-src-981feb9",
        "ff9755d21e45bfc05b86d041026b36cb26b58b9c",
        marker_name=".deployed_commit",
        marker_value="981feb9",
    ),
    SourceCopy(
        "neural-forge-unified-20260710",
        "bd9f807568dba70a7b3e16d70d1d8a88de29c403",
        "9cd68b1789e8f8afb9de5901510d762a1ce265b18237311c64e38a98e5e2a9e0",
        ".deployed_revision",
        "a9daf7b",
    ),
)


def tree_sha256(root: Path) -> str:
    """Hash relative names, directories, link targets, sizes, and file bytes."""

    lines: list[bytes] = []
    for directory_path, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        relative_directory = os.path.relpath(directory_path, root).replace(os.sep, "/")
        if relative_directory != ".":
            lines.append(f"D\t{relative_directory}\n".encode())
        for file_name in file_names:
            path = Path(directory_path) / file_name
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if path.is_symlink():
                lines.append(f"L\t{relative}\t{os.readlink(path)}\n".encode())
                continue
            stat = path.stat(follow_symlinks=False)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            lines.append(
                f"F\t{relative}\t{stat.st_size}\t{digest.hexdigest()}\n".encode()
            )
    return hashlib.sha256(b"".join(lines)).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(CANONICAL), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _archive_ref(copy: SourceCopy) -> str:
    return f"refs/tags/archive/remote-20260712/{copy.name}"


def _safe_source(copy: SourceCopy) -> Path:
    source = BASE / copy.name
    if source.exists():
        resolved = source.resolve(strict=True)
        if resolved.parent != BASE.resolve(strict=True) or resolved.name != copy.name:
            raise RuntimeError(f"Unsafe source root: {resolved}")
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"Source is not a real directory: {source}")
        if source.stat().st_dev != CANONICAL.stat().st_dev:
            raise RuntimeError(f"Source is not on the canonical filesystem: {source}")
    return source


def _assert_archive(copy: SourceCopy) -> None:
    actual = _git("rev-parse", _archive_ref(copy))
    if actual != copy.archive_commit:
        raise RuntimeError(
            f"Archive ref mismatch for {copy.name}: {actual} != {copy.archive_commit}"
        )


def _assert_no_process_references(sources: tuple[Path, ...]) -> None:
    existing_sources = tuple(source for source in sources if source.exists())
    needles = tuple(f"{source}/" for source in existing_sources)
    if not needles:
        return
    references: list[str] = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            command = (
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            command = ""
        if any(needle in command for needle in needles):
            references.append(f"pid={process.name} argv={command}")
        for name in ("cwd", "root"):
            try:
                target = os.readlink(process / name)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if any(
                target == str(source) or target.startswith(f"{source}/")
                for source in existing_sources
            ):
                references.append(f"pid={process.name} {name}={target}")
        try:
            descriptors = tuple((process / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            descriptors = ()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if any(
                target == str(source) or target.startswith(f"{source}/")
                for source in existing_sources
            ):
                references.append(
                    f"pid={process.name} fd={descriptor.name} target={target}"
                )
    if references:
        raise RuntimeError("Source copies are still in use:\n" + "\n".join(references))


def _marker_state(copy: SourceCopy, source: Path, destination: Path) -> str | None:
    if copy.marker_name is None:
        return None
    source_marker = source / copy.marker_name
    destination_marker = destination / copy.marker_name
    existing = [path for path in (source_marker, destination_marker) if path.exists()]
    if len(existing) != 1:
        raise RuntimeError(f"Expected exactly one marker for {copy.name}: {existing}")
    value = existing[0].read_text(encoding="utf-8").strip()
    if value != copy.marker_value:
        raise RuntimeError(f"Deployment marker mismatch for {copy.name}: {value}")
    return str(existing[0])


def _runs_state(copy: SourceCopy, source: Path, destination: Path) -> dict[str, Any]:
    source_runs = source / "runs"
    destination_runs = destination / "runs"
    if copy.runs_sha256 is None:
        if source_runs.exists() or destination_runs.exists():
            raise RuntimeError(f"Unexpected runs directory for {copy.name}")
        return {"expected": False}
    existing = [path for path in (source_runs, destination_runs) if path.exists()]
    if len(existing) != 1 or not existing[0].is_dir() or existing[0].is_symlink():
        raise RuntimeError(
            f"Expected exactly one real runs tree for {copy.name}: {existing}"
        )
    digest = tree_sha256(existing[0])
    if digest != copy.runs_sha256:
        raise RuntimeError(
            f"Run tree digest mismatch for {copy.name}: {digest} != {copy.runs_sha256}"
        )
    return {
        "expected": True,
        "location": str(existing[0]),
        "sha256": digest,
    }


def _state(copy: SourceCopy) -> dict[str, Any]:
    _assert_archive(copy)
    source = _safe_source(copy)
    destination = DESTINATION / copy.name
    if not source.exists() and not destination.is_dir():
        raise RuntimeError(
            f"Neither source nor receipt destination exists: {copy.name}"
        )
    return {
        "name": copy.name,
        "archive_ref": _archive_ref(copy),
        "archive_commit": copy.archive_commit,
        "source_exists": source.exists(),
        "destination_exists": destination.exists(),
        "runs": _runs_state(copy, source, destination),
        "marker": _marker_state(copy, source, destination),
    }


def retire(*, apply: bool, receipt: Path | None = None) -> dict[str, Any]:
    if CANONICAL.resolve(strict=True) != CANONICAL or not (CANONICAL / ".git").is_dir():
        raise RuntimeError("Canonical Git checkout is missing or redirected.")
    before = [_state(copy) for copy in COPIES]
    if not apply:
        return {"mode": "preflight", "copies": before}

    sources = tuple(_safe_source(copy) for copy in COPIES)
    _assert_no_process_references(sources)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for copy, source in zip(COPIES, sources):
        destination = DESTINATION / copy.name
        destination.mkdir(exist_ok=True)
        source_runs = source / "runs"
        destination_runs = destination / "runs"
        if source_runs.exists():
            if destination_runs.exists():
                raise RuntimeError(
                    f"Refusing run destination collision: {destination_runs}"
                )
            os.rename(source_runs, destination_runs)
        if copy.marker_name is not None:
            source_marker = source / copy.marker_name
            destination_marker = destination / copy.marker_name
            if source_marker.exists():
                if destination_marker.exists():
                    raise RuntimeError(
                        f"Refusing marker collision: {destination_marker}"
                    )
                os.rename(source_marker, destination_marker)
        _runs_state(copy, source, destination)
        _marker_state(copy, source, destination)
        if source.exists():
            shutil.rmtree(source)
        if source.exists():
            raise RuntimeError(f"Failed to retire source directory: {source}")

    after = [_state(copy) for copy in COPIES]
    payload = {
        "schema": "neural-forge-source-copy-retirement-v1",
        "mode": "applied",
        "canonical": str(CANONICAL),
        "destination": str(DESTINATION),
        "copies": after,
    }
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            retire(apply=args.apply, receipt=args.receipt), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
