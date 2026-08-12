#!/usr/bin/env python3
"""Print a wave-14 narrow-child dossier for a human reading, with the degeneracy view.

ADDENDUM 8's degeneracy clause asks what a response leaves when its repetitions
are removed, so both readings are printed side by side and the reader is never
shown only the flattering one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def dedup(text: str) -> str:
    seen: list[str] = []
    out: list[str] = []
    for chunk in re.split(r"(?<=[.!?\n])\s+", text):
        key = chunk.strip()
        if key and key not in seen:
            seen.append(key)
            out.append(key)
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", type=Path, required=True)
    ap.add_argument("--which", default="child", choices=["child", "parent", "both"])
    ap.add_argument("--chars", type=int, default=400)
    args = ap.parse_args()
    d = json.loads(args.dossier.read_text())
    lanes = ["child_responses"] if args.which == "child" else (
        ["parent_responses"] if args.which == "parent" else ["child_responses", "parent_responses"]
    )
    receipt = d.get("child_build_receipt", {})
    print(f"net {receipt.get('net_ratio')}  keep {receipt.get('keep')}  cases {d.get('cases')}")
    print(f"protocol {d.get('protocol_sha256', '')[:16]}  pack {d.get('prompt_pack_sha256', '')[:16]}")
    for lane in lanes:
        print(f"\n########## {lane} ##########")
        for r in d[lane]:
            text = r["response_verbatim"]
            dd = dedup(text)
            print("=" * 96)
            print(f"{r['id']}  [{r['category']}]  {r['generated_tokens']} tok  {r['termination']}")
            print(f"  raw   : {text[:args.chars]!r}")
            print(f"  dedup : {dd[:args.chars]!r}")
            print(f"  survives-dedup: {len(dd)} of {len(text)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
