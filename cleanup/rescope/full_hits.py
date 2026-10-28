#!/usr/bin/env python3
"""Full (source, token) reference hits + data dirs per approved unit, using cleanup_exec's own scanners."""
import json, os, sys
sys.path.insert(0, "/home/cereal/omni-campaign/lab/cleanup")
import cleanup_exec as X
R = "/home/cereal/omni-campaign/lab/cleanup/inputs/rescope-20261021"
units = json.load(open(f"{R}/approved_rescope.json"))
generic = set(json.load(open("/home/cereal/omni-campaign/lab/cleanup/inputs/generic_roots.json")))
scan = X.proc_scan()
toks = []
for pid, kind, text in scan["strings"]:
    for t in X.tokens_from_text(text):
        toks.append((f"pid{pid}:{kind}", t))
toks += X.crontab_tokens() + X.docker_mount_tokens() + X.RefIndex().file_tokens()
out = {}
for u in units:
    unit = u["path"]
    hits = sorted({(s, t) for s, t in toks if X.under(t, unit)})
    ph = sorted({(s, t) for s, t in toks if unit.startswith(t + "/") and t not in generic})
    srcs = {}
    for s, t in hits:
        srcs.setdefault(s, []).append(t)
    files, dirs = X.walk_unit(unit)
    dd = [os.path.relpath(d, unit) for d in dirs if os.path.basename(d) in X.DATA_DIR_NAMES]
    out[u["plan_index"]] = {"path": unit, "hit_sources": {s: {"n": len(v), "tokens_exist": sum(os.path.lexists(t) for t in v),
                            "sample": v[:3]} for s, v in srcs.items()}, "parent_hits": ph, "datadirs": dd,
                            "opened": [p for p in scan["open"] if X.under(p, unit)]}
json.dump(out, open("/home/cereal/claude-scratch/cleanup-rescope-20261021/full_hits.json", "w"), indent=1)
for i, o in out.items():
    print(f"== {i} {o['path']}  parent_hits={o['parent_hits']} opened={o['opened'][:3]}")
    for s, d in o["hit_sources"].items():
        print(f"   {d['n']:>4} hits ({d['tokens_exist']} exist)  {s}   e.g. {d['sample'][0]}")
    if o["datadirs"]:
        print(f"   datadirs {len(o['datadirs'])}: {o['datadirs'][:4]}")
