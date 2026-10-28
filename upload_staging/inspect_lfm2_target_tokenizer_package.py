#!/usr/bin/env python3
"""Read-only fingerprint of one already-local tokenizer package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.evaluate_lfm2_intelligence_retention import (  # noqa: E402
    fingerprint_bound_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    fingerprint, files = fingerprint_bound_local_tokenizer(args.directory)
    print(
        json.dumps(
            {
                "directory": str(args.directory.resolve()),
                "package_fingerprint": fingerprint,
                "file_count": len(files),
                "file_inventory_sha256": canonical_sha256(files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
