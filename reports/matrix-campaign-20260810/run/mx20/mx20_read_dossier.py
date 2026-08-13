#!/usr/bin/env python3
"""Print one frozen-review dossier in a form a reviewer can read case by case.

ADDENDUM 9 §9.1 needs two readings per child and both of them are judgements of
verbatim text, so this prints the verbatim text.  What it adds is the mechanical
part of the degeneracy clause, computed the same way for every case: the length
of the response after **consecutive repeated tokens are collapsed**, beside the
raw length.  A lane that leaves under 350 characters when its repetitions are
removed is a lane the clause charges, and the reviewer decides that, not this
script.  Nothing here scores anything.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def collapsed(text: str) -> str:
    """The response with consecutive repeated words and phrases collapsed."""

    words = text.split()
    out: list[str] = []
    for word in words:
        if out and out[-1] == word:
            continue
        out.append(word)
    joined = " ".join(out)
    for period in range(2, 13):
        pattern = re.compile(r"(?:\b(\w+(?:\s+\w+){%d})\b\s+)(?:\1\s+)+" % (period - 1))
        joined = pattern.sub(r"\1 ", joined)
    return joined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", type=Path, required=True)
    ap.add_argument("--chars", type=int, default=380)
    ap.add_argument("--which", default="child", choices=["child", "parent"])
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    body = json.loads(args.dossier.read_text())
    records = body[f"{args.which}_responses"]
    # Wave 20's two slate probes are not ledger cells: they carry no `cell_id`
    # and no dossier seal because there is no ARCH receipt to seal them against.
    # They are read case by case exactly like a child, so the reader names them
    # by their probe class instead of refusing them.
    label = body.get("cell_id") or f"PROBE.{body.get('probe', 'unknown')}"
    seal = body.get("dossier_sha256") or body.get("prompt_surface_sha256") or ""
    print(f"# {label}  dossier {seal[:16]}")
    ss = body.get("standard_shape_child")
    if ss:
        print(f"# net_ratio {ss['net_ratio']:.6f}  config_diff "
              f"{[d['field'] for d in ss['config_diff']]}")
    caps = 0
    for index, record in enumerate(records, start=1):
        text = record.get("response_verbatim", record.get("response", ""))
        squeezed = collapsed(text)
        if record["termination"] == "censored-at-cap":
            caps += 1
        print(
            f"\n--- {index:2d} {record.get('id', record.get('case_id'))} "
            f"[{record['termination']}] {len(text)} chars, {len(squeezed)} collapsed"
            + ("  <350 COLLAPSED" if len(squeezed) < 350 else "")
        )
        print(text if args.full else text[: args.chars])
    print(f"\n# cap-hits {caps}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
