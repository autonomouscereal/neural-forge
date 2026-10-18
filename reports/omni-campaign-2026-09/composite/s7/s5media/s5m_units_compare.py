#!/usr/bin/env python3
"""Compare a rebuilt S7 unit tree (units_s5) with the S3 units' pre-chart state.

S7-on-S5 media lane, Claude, 2026-10-13. For every unit: every non-chart file of the S3 unit (resolved
through symlinks) must have a byte-identical counterpart in the rebuilt unit, and the rebuilt
charted_block.json must equal the S3 unit's charted_block.json.pre-m1 in spec, accounting and declared
tensor sha256 table. All other json differences are listed (out paths are normalised first).
usage: s5m_units_compare.py <s3 units root> <rebuilt units root>
"""
import hashlib
import json
import sys
from pathlib import Path

UNITS = ["ace/ace_dit_L12", "h3/h3_dit_L25_curve6", "qimg/qimg_mmdit_L30_img", "hy3d/hy3d_dit_L10", "qvl32/qvl32_vit_L13"]
CHART_FILES = {"charts.safetensors", "charts_split_ridge.diagnostic.safetensors", "chart_receipt.json", "chart_receipt.json.sha256",
               "charted_block.json", "charted_block.json.pre-m1"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def flat(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = x
    return out


def norm(v, root, tag):
    if not isinstance(v, str):
        return v
    return v.replace(str(root), "<UNITS>").replace(f"/composite/s7/{tag}/", "/composite/s7/<UNITS>/")


def main():
    s3, s5 = Path(sys.argv[1]), Path(sys.argv[2])
    report = {"s3_root": str(s3), "rebuilt_root": str(s5), "units": {}}
    ok_all = True
    for u in UNITS:
        a, b = s3 / u, s5 / u
        r = {"files": {}, "missing": [], "mismatch": []}
        for f in sorted(a.iterdir()):
            if f.name in CHART_FILES:
                continue
            g = b / f.name
            if not g.exists():
                r["missing"].append(f.name)
                continue
            sa, sb = sha(f.resolve()), sha(g.resolve())
            r["files"][f.name] = {"sha256": sb, "identical": sa == sb, "symlink_s3": f.is_symlink(), "symlink_rebuilt": g.is_symlink(),
                                  "target_rebuilt": str(g.resolve()) if g.is_symlink() else None}
            if sa != sb:
                r["mismatch"].append(f.name)
        pre = json.loads((a / "charted_block.json.pre-m1").read_text())
        new = json.loads((b / "charted_block.json").read_text())
        fa, fb = flat(pre), flat(new)
        diffs = []
        for k in sorted(set(fa) | set(fb)):
            va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
            if norm(va, s3, "units") == norm(vb, s5, "units_s5"):
                continue
            diffs.append({"key": k, "s3_pre_m1": va if not isinstance(va, str) else va[:200], "rebuilt": vb if not isinstance(vb, str) else vb[:200]})
        r["json_diffs"] = diffs
        r["json_hard_diffs"] = [d["key"] for d in diffs if d["key"] != "s7_unit.tool_version"]
        r["tensor_sha256_equal"] = pre.get("tensor_sha256") == new.get("tensor_sha256")
        r["spec_equal"] = pre.get("spec") == new.get("spec")
        r["accounting_equal"] = pre.get("accounting") == new.get("accounting")
        r["charted_block_json_sha256"] = sha(b / "charted_block.json")
        r["verdict"] = ("IDENTICAL_BLOCK" if (not r["missing"] and not r["mismatch"] and r["tensor_sha256_equal"] and r["spec_equal"]
                                              and r["accounting_equal"]) else "DIFFERS")
        ok_all = ok_all and r["verdict"] == "IDENTICAL_BLOCK"
        report["units"][u] = r
    report["verdict"] = "all five rebuilt blocks identical to the S3 units' blocks" if ok_all else "DIFFERENCES (see units)"
    print(json.dumps(report, indent=1, default=str))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
