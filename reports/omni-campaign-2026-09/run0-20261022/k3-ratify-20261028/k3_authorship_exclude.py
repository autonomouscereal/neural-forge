#!/usr/bin/env python3
"""k3_authorship_exclude 1.0.0: list K3-row documents whose registry source is agent-authored (lead call L1).

Read-only over corpus/k3 row_meta + route_probe. Classes by registry source name: 'claude' (name contains
claude), 'memory_batch' (harvested agent-memory batches), 'memory_review' (other names mentioning memory).
Every doc in the first two classes is excluded from the run-0 K3 bucket (all loss terms masked);
'memory_review' docs are listed for area B's B-0 authorship audit. Numeric registry sources are left to
the run-0 authorship-field guard (it matches registry fields, not names)."""
import hashlib, json, sys, collections
K = "/mnt/lexar/omni-campaign/teacher_cache/k3/eb35851b2496a628-provisional-bf16"
C = "/home/cereal/omni-campaign/distill/corpus/k3"
out = sys.argv[1]
probe = json.load(open(K + "/route_probe_k3.json"))
docs = collections.defaultdict(list)
for d in probe["docs"]:
    docs[(d["pass"], d["row"])].append(d)
items, totals = [], collections.Counter()
for p in range(4):
    meta = json.load(open(f"{C}/pass-{p}/row_meta.json"))
    for r in meta["rows"]:
        ds = sorted(docs[(p, r["row_index"])], key=lambda x: x["doc"])
        assert len(ds) == len(r["rids"]), (p, r["row_index"])
        for rid, uid, d in zip(r["rids"], r["uids"], ds):
            name = rid.split(":")[1].lower()
            cls = ("claude" if "claude" in name else "memory_batch" if name.startswith("memory_batch")
                   else "memory_review" if "memory" in name else None)
            if cls is None:
                continue
            totals[cls + "_docs"] += 1
            totals[cls + "_k3_tokens"] += d["k3_tokens"]
            items.append({"class": cls, "action": "exclude" if cls != "memory_review" else "review",
                          "pass": p, "row": r["row_index"], "doc": d["doc"], "rid": rid, "uid": uid,
                          "route": d["route"], "k3_span": d["k3_span"], "q_span": d["q_span"],
                          "k3_tokens": d["k3_tokens"]})
body = {"schema": "omni-run0-k3-authorship-exclude-v1", "tool": "k3_authorship_exclude 1.0.0",
        "ruling": "k3-cache-ratify-run0-L1-2026-10-28",
        "rule": "exclude = mask every loss term (CE, KD, xtok) on the doc's q_span/k3_span in the run-0 K3 bucket",
        "k3_total_tokens": probe["all"]["k3_tokens"], "totals": dict(totals), "items": items}
raw = json.dumps(body, indent=1, sort_keys=True).encode()
open(out, "wb").write(raw)
print(json.dumps(body["totals"]), "sha256", hashlib.sha256(raw).hexdigest())
