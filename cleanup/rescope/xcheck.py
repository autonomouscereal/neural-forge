#!/usr/bin/env python3
"""Every reference source in full_hits.json must have exactly one waiver, and no waiver may lack a hit."""
import json
R = "/home/cereal/omni-campaign/lab/cleanup/inputs/rescope-20261021"
ov = json.load(open(f"{R}/overrides_rescope.json"))["units"]
fh = json.load(open("/home/cereal/claude-scratch/cleanup-rescope-20261021/full_hits.json"))
bad = 0
for i, o in fh.items():
    srcs, w = set(o["hit_sources"]), set(ov.get(i, {}).get("ref_sources", {}))
    print(i, "unwaived:", sorted(srcs - w), "waiver-without-hit:", sorted(w - srcs), "datadirs:", len(o["datadirs"]),
          "parent_hits:", o["parent_hits"], "opened:", o["opened"])
    bad += len(srcs - w) + len(w - srcs)
print("XCHECK", "OK" if not bad else f"MISMATCH {bad}")
