#!/usr/bin/env python3
"""Compare copied Neural Forge source trees with the canonical checkout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from inventory_neural_forge_source_trees import inventory_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--remote-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _file_map(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]): dict(row) for row in root["files"]}


def build_plan(canonical_root: Path, remote_inventory: Path) -> dict[str, Any]:
    canonical = inventory_root(canonical_root)
    remote_payload = json.loads(remote_inventory.read_text(encoding="utf-8"))
    if remote_payload.get("schema") != "neural-forge-source-tree-inventory-v1":
        raise ValueError("remote inventory schema is not supported")

    canonical_files = _file_map(canonical)
    path_versions: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    root_summaries: list[dict[str, Any]] = []
    for root in remote_payload["roots"]:
        root_name = str(root["root"])
        root_summaries.append(
            {
                "root": root_name,
                "file_count": int(root["file_count"]),
                "total_bytes": int(root["total_bytes"]),
                "tree_sha256": str(root["tree_sha256"]),
            }
        )
        for row in root["files"]:
            path_versions[str(row["path"])][str(row["sha256"])].append(root_name)

    remote_only: list[dict[str, Any]] = []
    differing: list[dict[str, Any]] = []
    identical_only: list[str] = []
    for path in sorted(path_versions):
        versions = path_versions[path]
        canonical_row = canonical_files.get(path)
        rendered_versions = [
            {"sha256": digest, "roots": sorted(roots)}
            for digest, roots in sorted(versions.items())
        ]
        if canonical_row is None:
            remote_only.append({"path": path, "remote_versions": rendered_versions})
            continue
        canonical_sha = str(canonical_row["sha256"])
        alternatives = [
            value for value in rendered_versions if value["sha256"] != canonical_sha
        ]
        if alternatives:
            differing.append(
                {
                    "path": path,
                    "canonical_sha256": canonical_sha,
                    "canonical_bytes": int(canonical_row["bytes"]),
                    "remote_alternatives": alternatives,
                    "remote_match_roots": sorted(versions.get(canonical_sha, [])),
                }
            )
        else:
            identical_only.append(path)

    canonical_only = sorted(set(canonical_files) - set(path_versions))
    return {
        "schema": "neural-forge-source-consolidation-plan-v1",
        "canonical": {
            key: canonical[key]
            for key in ("root", "file_count", "total_bytes", "tree_sha256")
        },
        "remote_roots": root_summaries,
        "summary": {
            "remote_root_count": len(root_summaries),
            "canonical_file_count": len(canonical_files),
            "remote_union_file_count": len(path_versions),
            "identical_path_count": len(identical_only),
            "remote_only_path_count": len(remote_only),
            "differing_path_count": len(differing),
            "canonical_only_path_count": len(canonical_only),
        },
        "remote_only": remote_only,
        "differing": differing,
        "canonical_only": canonical_only,
    }


def main() -> int:
    args = build_parser().parse_args()
    plan = build_plan(args.canonical_root, args.remote_inventory)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
