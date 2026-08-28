#!/usr/bin/env python3
"""Free read: does a sealed width basis keep the parent's massive-activation DIRECTION?

Wave 29's emission pre-read compared massive-activation dimensions by INDEX --
the parent's coordinate 294 against the child's own top-energy coordinates -- and
reported "MASSIVE-ACTIVATION DIMS LOST: 1 of 1" for ``W720`` and ``MM768``.  A
width fold rewrites the residual stream as ``z = E^T x``, so the child's
coordinate 294 is not the parent's coordinate 294 and an index comparison across
that rotation cannot mean what it says.

The quantity that does mean it is the parent massive direction's energy inside
the child's subspace,

    keep(E) = || E^T e_294 ||^2 = e_294^T (E E^T) e_294,

which is 1.0 when the fold keeps the direction intact and 0.0 when it deletes
it.  Every sealed reader-PCA basis is on the share, so the whole ladder costs one
eigenvector read per width and no GPU at all.  It is the cheapest possible test
of the field synthesis's #2 -- a carve-out can only buy what the truncation
currently throws away, and this is that number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MASSIVE = 294
SCHEMA = "neural-forge-matrix-wave32-massive-direction-probe-v1"

#: label -> (sealed basis artifact, the sealed review count of the child built
#: on it).  The review column is here so the retention is read against the width
#: cliff in one table rather than two.
BASES: tuple[tuple[str, str, int | None], ...] = (
    ("960", "/mnt/ai_data/neural-forge-matrix/wave19/basis/E-reader-pca-960.safetensors", None),
    ("896", "/mnt/ai_data/neural-forge-matrix/wave19/basis/E-reader-pca-896.safetensors", 4),
    ("768", "/mnt/ai_data/neural-forge-matrix/wave19/basis/E-reader-pca-768.safetensors", 3),
    ("720", "/mnt/ai_data/neural-forge-matrix/wave22/basis/E-reader-pca-720.safetensors", 0),
    ("704", "/mnt/ai_data/neural-forge-matrix/wave20/basis/E-reader-pca-704.safetensors", None),
    ("640", "/mnt/ai_data/neural-forge-matrix/wave19/basis/E-reader-pca-640.safetensors", 0),
    ("512", "/mnt/ai_data/neural-forge-matrix/wave19/basis/E-reader-pca-512.safetensors", 0),
    ("768ev", "/mnt/ai_data/neural-forge-matrix/wave19/basis/E-evidence-pca-768.safetensors", 1),
)


def main() -> int:
    from safetensors.torch import load_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out: dict[str, object] = {
        "schema": SCHEMA,
        "massive_dim": MASSIVE,
        "note": (
            "keep = ||E^T e_294||^2, the parent massive-activation direction's "
            "energy inside the child's residual subspace.  Index comparison "
            "across a basis rotation is not this quantity, and wave 29's "
            "`massive_dims_lost` reading for the width children is superseded "
            "by it."
        ),
        "widths": {},
    }
    for label, path, sealed in BASES:
        p = Path(path)
        if not p.exists():
            out["widths"][label] = {"error": "basis artifact absent"}
            print(f"{label:>6}  ABSENT {path}")
            continue
        E = load_file(str(p))["basis"].to(torch.float64)
        hidden, width = E.shape
        e = torch.zeros(hidden, dtype=torch.float64)
        e[MASSIVE] = 1.0
        coeff = E.T @ e
        keep = float((coeff ** 2).sum())
        top = int(coeff.abs().argmax())
        # A direction can be "kept" while being smeared over many columns, and
        # the smearing is what makes an index comparison fail downstream, so it
        # is reported next to the retention rather than left to be inferred.
        share = torch.sort(coeff ** 2, descending=True).values
        out["widths"][label] = {
            "hidden": hidden,
            "width": width,
            "massive_direction_kept": round(keep, 6),
            "massive_direction_lost": round(1.0 - keep, 6),
            "argmax_column": top,
            "argmax_column_share": round(float(coeff[top] ** 2), 6),
            "columns_for_90pct": int((torch.cumsum(share, 0) < 0.9 * keep).sum()) + 1,
            "sealed_review": sealed,
        }
        print(
            f"{label:>6}  keep {keep:.6f}  lost {1.0 - keep:.6f}  "
            f"argmax col {top} ({float(coeff[top] ** 2):.4f})  sealed {sealed}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
