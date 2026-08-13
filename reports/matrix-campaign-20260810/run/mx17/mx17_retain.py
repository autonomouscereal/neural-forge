#!/usr/bin/env python3
"""Wave 17 retention: the best five by sealed score plus the deepest non-zero
child, kept; every other child deleted and named with its sealed tensor digest.
"""
import json, os, shutil, sys

W = "/mnt/ai_data/neural-forge-matrix/wave17"
KEEP = {"MX17-X06", "MX17-KV6JQ", "MX17-P1301", "MX17-P1303", "MX17-X10", "MX17-X13KV6JQ"}
REASON = {
    "MX17-X06": "8/15, the highest-scoring attention excision and the best benchmark row of any child in the campaign",
    "MX17-KV6JQ": "8/15, the joint-solve head merge, three cases above wave 16 average-in-gauge at the same size",
    "MX17-P1301": "7/15, the highest-scoring pair (blocks 13 and 1) and the deepest non-zero child of the wave",
    "MX17-P1303": "7/15, the second-widest pair, and the distance ladder needs both ends",
    "MX17-X10": "7/15 and the only child of the wave whose per-CHILD verdict is not dead",
    "MX17-X13KV6JQ": "4/15, the deepest child that stacks an excision on the joint merge",
}
index = {c["cell_id"]: c for c in json.load(open(W + "/wave17-child-index.json"))["children"]}
rows = []
for name in sorted(os.listdir(W + "/children")):
    path = os.path.join(W, "children", name)
    if not os.path.isdir(path):
        continue
    tag = name.replace("MX17-", "")
    entry = index.get("ARCH." + tag, {})
    receipt = json.load(open(os.path.join(path, "mx17-build-receipt.json")))
    row = {
        "child": name,
        "cell_id": "ARCH." + tag,
        "net_ratio": receipt["net_ratio"],
        "child_tensors_sha256": receipt["child_tensors_sha256"],
        "useful_or_partial": entry.get("useful_or_partial"),
        "per_child_verdict": entry.get("per_child_verdict"),
        "kept": name in KEEP,
        "reason": REASON.get(name, "not in the best five and not the deepest non-zero"),
        "bytes": sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)),
    }
    rows.append(row)
freed = 0
for row in rows:
    if not row["kept"]:
        shutil.rmtree(os.path.join(W, "children", row["child"]))
        freed += row["bytes"]
body = {
    "schema": "neural-forge-matrix-wave17-children-retention-v1",
    "policy": "best five by sealed ADDENDUM 6 score plus the deepest non-zero child",
    "kept": sorted(KEEP),
    "freed_bytes": freed,
    "children": rows,
}
open(W + "/children-retention-manifest-wave17.json", "w").write(json.dumps(body, indent=1))
print("kept", sum(1 for r in rows if r["kept"]), "deleted", sum(1 for r in rows if not r["kept"]), "freed", freed)
