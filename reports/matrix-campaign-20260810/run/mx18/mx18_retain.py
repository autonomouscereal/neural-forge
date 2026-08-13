#!/usr/bin/env python3
"""Wave 18 retention: the best five by sealed score plus the children a later
wave cannot re-derive, kept; every other child deleted and named with its sealed
tensor digest.

Wave 18 adds one class to the policy and says why: the pre-repair `X07KV6JQ` is
kept beside its repaired twin because the pair is the only behavioural price of
the causal-mask defect this campaign will ever measure, and a control you have
deleted is not a control.  The bit-identical rebuilds are NOT kept -- X13C, X00C,
X07C, P1301C and T010713C reproduce their originals to the digest, so keeping
them would be keeping the same bytes twice.
"""
import json, os, shutil, sys

W = "/mnt/ai_data/neural-forge-matrix/wave18"
KEEP = {
    "MX18-KV6JS",
    "MX18-X07KV6JQC",
    "MX18-X07KV6JQ",
    "MX18-X09KV6JQC",
    "MX18-T010713",
}
REASON = {
    "MX18-KV6JS": "7/15, the lambda fix: the value path costs one case where wave 17 unsplit lambda cost five, at the identical parameter count and an unchanged benchmark row",
    "MX18-X07KV6JQC": "7/15, the best composed child of the wave and the best excision-plus-merge composition in the campaign",
    "MX18-X07KV6JQ": "6/15, the SAME recipe built with the causal-mask defect -- the only behavioural price of that defect the campaign will ever measure, and a control you have deleted is not a control",
    "MX18-X09KV6JQC": "5/15, the second composition rung and the one that says the merge does not compose by distance",
    "MX18-T010713": "3/15, the highest-scoring triple and the deepest non-zero child of the wave at 0.842437x",
}
index = {c["cell_id"]: c for c in json.load(open(W + "/wave18-child-index.json"))["children"]}
rows = []
for name in sorted(os.listdir(W + "/children")):
    path = os.path.join(W, "children", name)
    if not os.path.isdir(path):
        continue
    tag = name.replace("MX18-", "")
    entry = index.get("ARCH." + tag, {})
    receipt = json.load(open(os.path.join(path, "mx18-build-receipt.json")))
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
    "schema": "neural-forge-matrix-wave18-children-retention-v1",
    "policy": "best five by sealed ADDENDUM 6 score plus the deepest non-zero child",
    "kept": sorted(KEEP),
    "freed_bytes": freed,
    "children": rows,
}
open(W + "/children-retention-manifest-wave18.json", "w").write(json.dumps(body, indent=1))
print("kept", sum(1 for r in rows if r["kept"]), "deleted", sum(1 for r in rows if not r["kept"]), "freed", freed)
