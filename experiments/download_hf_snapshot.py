#!/usr/bin/env python3
"""Download one immutable Hugging Face snapshot into an explicit cache root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--revision")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    resolved = Path(snapshot).resolve(strict=True)
    print(
        json.dumps(
            {
                "cache_dir": str(args.cache_dir.resolve()),
                "repo_id": args.repo_id,
                "requested_revision": args.revision,
                "snapshot": str(resolved),
                "snapshot_revision": resolved.name,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
