#!/usr/bin/env python3
"""How far apart are two sealed bases, as subspaces?

The wave's central comparison is between children that differ in ONE thing --
which 768-dimensional subspace of the residual they keep -- and are otherwise the
same parameter count, the same corpus, the same solver and the same code.  For
that comparison to mean anything, the two subspaces have to actually differ.
Two bases can have wildly different construction stories and span very nearly the
same thing, in which case a matched review score says nothing about objectives
and everything about the fact that nothing moved.

This prices it, from the sealed artifacts alone and with no GPU:

* **the overlap** ``tr(E_a^T E_b E_b^T E_a) / H`` -- the mean squared cosine of
  the principal angles, 1.0 for identical subspaces and H'/H for a random
  H'-dimensional overlap;
* **the principal angles** themselves, summarised, so a subspace that agrees on
  most directions and disagrees on a few is not reported as "nearly the same";
* **the dimension exchanged**: ``H - overlap*H``, the number of directions one
  basis keeps and the other does not, which is the number a reader wants.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx28_build as mx  # noqa: E402


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    width = int(a.shape[1])
    cross = a.T @ b
    singular = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(singular.clamp(max=1.0)))
    overlap = float(singular.pow(2).sum() / width)
    return {
        "width": width,
        "subspace_overlap": overlap,
        "dimensions_exchanged": width * (1.0 - overlap),
        "principal_angle_deg": {
            "min": float(angles.min()),
            "median": float(angles.median()),
            "p90": float(angles.kthvalue(max(1, int(0.9 * width))).values),
            "max": float(angles.max()),
        },
        "directions_beyond_1deg": int((angles > 1.0).sum()),
        "directions_beyond_45deg": int((angles > 45.0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    loaded: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {}
    for path in args.basis:
        basis, census = mx.read_basis(Path(path))
        name = Path(path).stem
        loaded[name] = basis
        meta[name] = {
            "sha256": census["basis_sha256"],
            "mode": census.get("mode"),
            "width": int(basis.shape[1]),
        }
    report: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave28-subspace-comparison-v1",
        "bases": meta,
        "pairs": {},
    }
    for left, right in combinations(loaded, 2):
        if loaded[left].shape != loaded[right].shape:
            continue
        report["pairs"][f"{left} vs {right}"] = compare(loaded[left], loaded[right])
    print(json.dumps(report, indent=1))
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
