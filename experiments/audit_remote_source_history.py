#!/usr/bin/env python3
"""Prove whether copied remote source bytes already exist in canonical Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from inventory_neural_forge_source_trees import inventory_root


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def git_blob_sha256s(root: Path) -> set[str]:
    object_ids = sorted(
        set(
            _git(root, "rev-list", "--objects", "--all", "--no-object-names")
            .decode("ascii")
            .splitlines()
        )
    )
    query = ("\n".join(object_ids) + "\n").encode("ascii")
    checked = _git(
        root,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype)",
        input_bytes=query,
    ).decode("ascii")
    blobs = [line.split()[0] for line in checked.splitlines() if line.endswith(" blob")]
    blob_query = ("\n".join(blobs) + "\n").encode("ascii")
    batch = _git(root, "cat-file", "--batch", input_bytes=blob_query)
    digests: set[str] = set()
    offset = 0
    for object_id in blobs:
        header_end = batch.index(b"\n", offset)
        header = batch[offset:header_end].decode("ascii").split()
        if len(header) != 3 or header[0] != object_id or header[1] != "blob":
            raise RuntimeError(f"unexpected git cat-file header: {header!r}")
        size = int(header[2])
        content_start = header_end + 1
        content_end = content_start + size
        digests.add(hashlib.sha256(batch[content_start:content_end]).hexdigest())
        if batch[content_end : content_end + 1] != b"\n":
            raise RuntimeError("git cat-file batch framing is malformed")
        offset = content_end + 1
    if offset != len(batch):
        raise RuntimeError("git cat-file batch has trailing bytes")
    return digests


def build_audit(canonical_root: Path, remote_inventory: Path) -> dict[str, Any]:
    canonical = inventory_root(canonical_root)
    current_hashes = {str(row["sha256"]) for row in canonical["files"]}
    historical_hashes = git_blob_sha256s(canonical_root)
    payload = json.loads(remote_inventory.read_text(encoding="utf-8"))

    occurrences: dict[tuple[str, str], list[str]] = {}
    for root in payload["roots"]:
        root_name = str(root["root"])
        for row in root["files"]:
            key = (str(row["path"]), str(row["sha256"]))
            occurrences.setdefault(key, []).append(root_name)

    rows: list[dict[str, Any]] = []
    for (path, digest), roots in sorted(occurrences.items()):
        if digest in current_hashes:
            status = "canonical-worktree"
        elif digest in historical_hashes:
            status = "canonical-git-history"
        else:
            status = "not-in-canonical-history"
        rows.append(
            {
                "path": path,
                "sha256": digest,
                "roots": sorted(roots),
                "status": status,
            }
        )

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "canonical-worktree",
            "canonical-git-history",
            "not-in-canonical-history",
        )
    }
    return {
        "schema": "neural-forge-remote-source-history-audit-v1",
        "canonical_root": str(canonical_root.resolve()),
        "canonical_tree_sha256": canonical["tree_sha256"],
        "git_blob_sha256_count": len(historical_hashes),
        "counts": counts,
        "unseen": [row for row in rows if row["status"] == "not-in-canonical-history"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--remote-inventory", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        json.dumps(
            build_audit(args.canonical_root, args.remote_inventory),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
