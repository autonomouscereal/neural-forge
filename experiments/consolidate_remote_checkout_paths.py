#!/usr/bin/env python3
"""Point every executable experiment at the one canonical AI-server checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANONICAL_ROOT = b"/home/cereal/model-surgery-lab/neural-forge-src"
STALE_ROOTS = tuple(
    value.encode("utf-8")
    for value in sorted(
        {
            "/home/cereal/model-surgery-lab/neural-forge-behavior-vo-tuner-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-cached-attention-smoke-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-functional-adapter-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-p35-multisite-ragged-qkvo-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-pivot",
            "/home/cereal/model-surgery-lab/neural-forge-qkvo-broad-consumer-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-qkvo-residual-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-qkvo-selected-pack-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-residual-attn-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-residual-attn-functional-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-role-scale-functional-20260711",
            "/home/cereal/model-surgery-lab/neural-forge-unified-20260710",
        },
        key=len,
        reverse=True,
    )
)
TEXT_SUFFIXES = frozenset({".json", ".md", ".ps1", ".py", ".sh", ".txt"})
INTENTIONAL_COPY_INVENTORIES = frozenset(
    {
        "remote_inventory_neural_forge_sources_20260712.sh",
        "remote_pack_neural_forge_sources_20260712.sh",
    }
)


def consolidate(root: Path, *, write: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "experiments").rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in TEXT_SUFFIXES
            or path.name in INTENTIONAL_COPY_INVENTORIES
            or path.name == Path(__file__).name
        ):
            continue
        original = path.read_bytes()
        updated = original
        replacements = 0
        for stale in STALE_ROOTS:
            count = updated.count(stale)
            if count:
                updated = updated.replace(stale, CANONICAL_ROOT)
                replacements += count
        if updated == original:
            continue
        if write:
            path.write_bytes(updated)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "replacement_count": replacements,
                "written": write,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--write", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = consolidate(args.root.resolve(), write=args.write)
    print(
        json.dumps(
            {
                "canonical_root": CANONICAL_ROOT.decode("utf-8"),
                "changed_file_count": len(rows),
                "replacement_count": sum(int(row["replacement_count"]) for row in rows),
                "files": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
