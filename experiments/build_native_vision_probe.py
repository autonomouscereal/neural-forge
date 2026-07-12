#!/usr/bin/env python3
"""Build the deterministic Neural Forge native-vision transfer probe."""

from __future__ import annotations

import argparse
import json

from neural_forge.native_vision_probe import build_native_vision_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--examples-per-category", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = build_native_vision_probe(
        args.output_dir,
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )
    summary = {
        key: result[key]
        for key in (
            "schema",
            "seed",
            "total",
            "calibration_count",
            "heldout_count",
            "categories",
            "manifest",
            "manifest_sha256",
        )
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
