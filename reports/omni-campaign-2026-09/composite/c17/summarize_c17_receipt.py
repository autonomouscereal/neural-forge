#!/usr/bin/env python3
"""Lane E3: print one diagnostic line per trunk layer from a c17 receipt.json (read-only).

Numbers are DIAGNOSTICS (owner policy 2026-10-06); nothing here gates or drops a layer.
Columns: trunk layer, K3 layers, paired rows (per K3 layer), unique trunk tokens, gate rows,
fit/held rows, then held cos/relL2 for C_in (forward, K3->trunk) and C_out (backward,
trunk->K3) and round-trip cos, for the ridge chart of record and the svd / procrustes controls,
plus the ridge refit regime and seconds.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def fmt(arm: dict | None) -> str:
    hs = (arm or {}).get("held_summary")
    if not hs:
        return "n/a"
    f, b, r = hs["forward"], hs["backward"], hs["roundtrip"]
    return f"in {f['cos']:.4f}/{f['rel_l2']:.4f} out {b['cos']:.4f}/{b['rel_l2']:.4f} rt {r['cos']:.4f}"


def main() -> int:
    path = Path(sys.argv[1])
    raw = path.read_bytes()
    receipt = json.loads(raw)
    print(f"receipt {path} sha256 {hashlib.sha256(raw).hexdigest()} tool {receipt.get('tool_version')} "
          f"present {receipt.get('layers_present')} invalid {receipt.get('layers_invalid')}")
    print("k | K3 | paired (per K3) | uniq | gate | fit/held | ridge C_in cos/relL2, C_out cos/relL2, rt cos | svd | procrustes | regime | s")
    for rec in receipt["layers"]:
        if rec.get("status") == "invalid":
            print(f"L{rec['trunk_layer']:02d} INVALID {rec.get('reasons')}")
            continue
        arms = rec.get("arms", {})
        sp = rec.get("split", {})
        print(f"L{rec['trunk_layer']:02d} | {rec['k3_layers']} | {rec['paired_rows']} {rec['paired_rows_per_k3_layer']} | "
              f"{rec['unique_trunk_tokens']} | {rec['gate']['rows']} | {sp.get('fit_rows')}/{sp.get('held_rows')} | "
              f"ridge {fmt(arms.get('ridge'))} | svd {fmt(arms.get('svd'))} | procrustes {fmt(arms.get('procrustes'))} | "
              f"{rec.get('ridge_regime')} | {rec.get('seconds', 0):.0f}")
    print("ridge_held aggregate:", json.dumps(receipt.get("ridge_held")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
