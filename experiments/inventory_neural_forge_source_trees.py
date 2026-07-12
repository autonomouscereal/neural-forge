#!/usr/bin/env python3
"""Create a content-addressed inventory of Neural Forge source trees.

This intentionally excludes checkpoints, run artifacts, environments, caches,
and ad-hoc transfer archives.  It is used to consolidate historical remote
source copies into the canonical Git checkout without copying model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_DIRECTORIES = frozenset(
    {
        "docs",
        "experiments",
        "research",
        "scripts",
        "src",
        "tests",
    }
)
SOURCE_FILES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "README.md",
        "pyproject.toml",
    }
)
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "runs",
        "venv",
    }
)
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".zip"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in sorted(SOURCE_FILES):
        path = root / name
        if path.is_file():
            candidates.append(path)
    for name in sorted(SOURCE_DIRECTORIES):
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and not any(part in EXCLUDED_PARTS for part in relative.parts)
                and path.suffix.lower() not in EXCLUDED_SUFFIXES
            ):
                candidates.append(path)
    return sorted(set(candidates), key=lambda value: value.relative_to(root).as_posix())


def inventory_root(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"source root does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in iter_source_files(resolved):
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": size,
                "sha256": file_sha256(path),
            }
        )
    tree_digest = hashlib.sha256()
    for row in rows:
        tree_digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        tree_digest.update(b"\n")
    return {
        "root": str(resolved),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = {
        "schema": "neural-forge-source-tree-inventory-v1",
        "roots": [inventory_root(root) for root in args.root],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
