#!/usr/bin/env python3
"""Move unique ignored project data and retire superseded local Git folders."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE = Path(r"C:\Users\cereal\Documents\ai training")
CANONICAL = WORKSPACE / "neural-forge"
DESTINATION = CANONICAL / "runs" / "legacy-projects"


@dataclass(frozen=True, slots=True)
class DataMove:
    source_relative: str
    destination_relative: str
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class Project:
    name: str
    head: str
    archive_tag: str
    moves: tuple[DataMove, ...]


PROJECTS = (
    Project(
        "chimera-omni",
        "925a083296630aec3477367a15419af74f79fe8d",
        "archive/projects/chimera-omni-main",
        (
            DataMove("artifacts", "artifacts", 129, 1_524_494),
            DataMove(
                "proof_outputs/multitask_interference/external_benchmarks/benchmark_lab_smoke.log",
                "proof_outputs/multitask_interference/external_benchmarks/benchmark_lab_smoke.log",
                1,
                25_484,
            ),
            DataMove(
                "proof_outputs/multitask_interference/external_benchmarks/evalplus_chimera_mini.log",
                "proof_outputs/multitask_interference/external_benchmarks/evalplus_chimera_mini.log",
                1,
                315,
            ),
            DataMove(
                "proof_outputs/multitask_interference/proof_run_stdout.log",
                "proof_outputs/multitask_interference/proof_run_stdout.log",
                1,
                186_920,
            ),
        ),
    ),
    Project(
        "ai-frontier-research-2026",
        "bafd73231a057312bda986c93f5e46b78b5887e7",
        "archive/projects/ai-frontier-research-2026-main",
        (
            DataMove("pdfs", "pdfs", 5_869, 36_535_248_604),
            DataMove("extracted_text", "extracted_text", 5_869, 477_146_988),
            DataMove("tmp", "tmp", 1_352, 46_746_804),
        ),
    ),
)


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    files = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RuntimeError(f"Refusing a data tree containing a symlink: {item}")
        if item.is_file():
            files += 1
            size += item.stat().st_size
    return files, size


def _safe_project(project: Project) -> Path:
    path = WORKSPACE / project.name
    if path.exists():
        resolved = path.resolve(strict=True)
        if (
            resolved.parent != WORKSPACE.resolve(strict=True)
            or resolved.name != project.name
        ):
            raise RuntimeError(f"Unsafe project root: {resolved}")
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"Project is not a real directory: {path}")
    return path


def _assert_history(
    project: Project,
    source: Path,
    *,
    data_already_moved: bool,
) -> str:
    archived = _git(CANONICAL, "rev-parse", f"refs/tags/{project.archive_tag}")
    if archived != project.head:
        raise RuntimeError(f"Canonical archive tag mismatch for {project.name}")
    if source.exists():
        try:
            if _git(source, "rev-parse", "HEAD") != project.head:
                raise RuntimeError(f"Source HEAD mismatch for {project.name}")
            if _git(source, "status", "--porcelain"):
                raise RuntimeError(f"Source repository is not clean: {project.name}")
            _git(source, "fsck", "--strict")
        except subprocess.CalledProcessError:
            if not data_already_moved:
                raise RuntimeError(
                    f"Source Git repository became unreadable before its unique "
                    f"data moved: {project.name}"
                ) from None
            return "resume-after-partial-read-only-deletion"
        return "verified-clean-source"
    return "archive-only"


def _remove_tree(path: Path) -> None:
    def make_writable_and_retry(function, failed_path, _error_info) -> None:
        os.chmod(failed_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        function(failed_path)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def _move_state(project: Project, move: DataMove) -> dict[str, Any]:
    source = WORKSPACE / project.name / Path(move.source_relative)
    destination = DESTINATION / project.name / Path(move.destination_relative)
    existing = [path for path in (source, destination) if path.exists()]
    if len(existing) != 1:
        raise RuntimeError(f"Expected one data location for {project.name}: {existing}")
    actual = _stats(existing[0])
    expected = (move.files, move.bytes)
    if actual != expected:
        raise RuntimeError(
            f"Data stats mismatch for {existing[0]}: {actual} != {expected}"
        )
    return {
        "source_relative": move.source_relative,
        "destination_relative": move.destination_relative,
        "location": str(existing[0]),
        "files": actual[0],
        "bytes": actual[1],
    }


def _state(project: Project) -> dict[str, Any]:
    source = _safe_project(project)
    moves = [_move_state(project, move) for move in project.moves]
    destination_prefix = str(DESTINATION / project.name)
    history_state = _assert_history(
        project,
        source,
        data_already_moved=all(
            str(move["location"]).startswith(destination_prefix) for move in moves
        ),
    )
    return {
        "name": project.name,
        "head": project.head,
        "archive_tag": project.archive_tag,
        "source_exists": source.exists(),
        "history_state": history_state,
        "moves": moves,
    }


def retire(*, apply: bool, receipt: Path | None = None) -> dict[str, Any]:
    if CANONICAL.resolve(strict=True) != CANONICAL or not (CANONICAL / ".git").is_dir():
        raise RuntimeError("Canonical local Git checkout is missing or redirected.")
    before = [_state(project) for project in PROJECTS]
    if not apply:
        return {"mode": "preflight", "projects": before}

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for project in PROJECTS:
        source_root = _safe_project(project)
        destination_root = DESTINATION / project.name
        destination_root.mkdir(exist_ok=True)
        for move in project.moves:
            source = source_root / Path(move.source_relative)
            destination = destination_root / Path(move.destination_relative)
            if source.exists():
                if destination.exists():
                    raise RuntimeError(
                        f"Refusing data destination collision: {destination}"
                    )
                if source.stat().st_dev != destination_root.stat().st_dev:
                    raise RuntimeError(f"Move would cross filesystems: {source}")
                source_inode = (source.stat().st_dev, source.stat().st_ino)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(source, destination)
                destination_inode = (
                    destination.stat().st_dev,
                    destination.stat().st_ino,
                )
                if destination_inode != source_inode:
                    raise RuntimeError(f"Move was not inode-preserving: {destination}")
            _move_state(project, move)
        if source_root.exists():
            _assert_history(project, source_root, data_already_moved=True)
            _remove_tree(source_root)
        if source_root.exists():
            raise RuntimeError(f"Failed to retire source project: {source_root}")

    after = [_state(project) for project in PROJECTS]
    payload = {
        "schema": "neural-forge-local-project-retirement-v1",
        "mode": "applied",
        "canonical": str(CANONICAL),
        "destination": str(DESTINATION),
        "projects": after,
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
