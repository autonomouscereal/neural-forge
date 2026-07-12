#!/usr/bin/env python3
"""Import a copied source snapshot into the canonical repository as an archive ref.

The active worktree and index are never changed.  A temporary alternate index
builds an orphan tree from the snapshot, and ``git update-ref`` records it under
the explicitly supplied ref name.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def _git(
    repository: Path,
    worktree: Path,
    index: Path,
    *arguments: str,
    input_text: str | None = None,
) -> str:
    environment = dict(os.environ)
    environment["GIT_INDEX_FILE"] = str(index)
    completed = subprocess.run(
        [
            "git",
            f"--git-dir={repository / '.git'}",
            f"--work-tree={worktree}",
            *arguments,
        ],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=True,
    )
    return completed.stdout.strip()


def import_snapshot(
    repository: Path,
    snapshot: Path,
    ref: str,
    message: str,
) -> str:
    repository = repository.resolve()
    snapshot = snapshot.resolve()
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"canonical Git directory is missing: {repository}")
    if not snapshot.is_dir():
        raise FileNotFoundError(f"snapshot directory is missing: {snapshot}")
    if not ref.startswith("refs/archive/"):
        raise ValueError("archive refs must begin with refs/archive/")

    descriptor, index_name = tempfile.mkstemp(
        prefix="neural-forge-snapshot-index-", dir=repository / ".git"
    )
    os.close(descriptor)
    index = Path(index_name)
    index.unlink()
    try:
        _git(repository, snapshot, index, "read-tree", "--empty")
        _git(repository, snapshot, index, "add", "--all", "--force")
        tree = _git(repository, snapshot, index, "write-tree")
        commit = _git(
            repository,
            snapshot,
            index,
            "commit-tree",
            tree,
            input_text=message.rstrip() + "\n",
        )
        _git(repository, snapshot, index, "update-ref", ref, commit)
        observed = _git(repository, snapshot, index, "rev-parse", ref)
        if observed != commit:
            raise RuntimeError("archive ref did not resolve to the imported commit")
        return commit
    finally:
        index.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--message", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    commit = import_snapshot(
        args.repository,
        args.snapshot,
        args.ref,
        args.message,
    )
    print(commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
