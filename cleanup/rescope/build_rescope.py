#!/usr/bin/env python3
"""build_rescope.py - new allowlist + protected list for the 2026-10-21 post-fold-ruling rescope.

Reads the old inputs read-only and writes NEW files into inputs/rescope-20261021/ (old files untouched).
Approved = the 8 units of the lead review "APPROVED LIST" (2026-10-21, after Cyril's
cyril-v2-arch-and-fold-init-2026-10-21 ruling); everything else the old allowlist held is protected.
"""
import hashlib, json, os, sys

IN = "/home/cereal/omni-campaign/lab/cleanup/inputs"
OUT = f"{IN}/rescope-20261021"
APPROVED = {9, 125, 208, 210, 229, 132, 237, 241}
CLONE_D2 = "/mnt/lexar/omni-campaign/clones/OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2"
CLONE_D1 = "/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1"
P1_PLAN = "/mnt/lexar/omni-campaign/distill/d2/p1_plan.json"
FOLD_ROOTS = [  # PROTECTED NOW (fold artifacts / fold sources / records), existing roots only
    "/mnt/lexar/omni-campaign/s3-fold-sep07-codex", "/home/cereal/omni-campaign/composite/s7",
    "/home/cereal/omni-campaign/composite/c17", "/home/cereal/omni-campaign/composite/s5",
    "/home/cereal/omni-campaign/runtime", "/home/cereal/omni-campaign/qwen-fold",
    "/home/cereal/omni-campaign/k3-run-oct09-claude", "/mnt/ai_data/omni-campaign/k3-run-oct09-claude-archive",
    "/mnt/lexar/omni-campaign/distill/fullkd/run1", "/mnt/lexar/omni-campaign/teacher_cache",
    "/mnt/lexar/omni-campaign/k3", "/mnt/lexar/omni-campaign/omni1", "/mnt/lexar/omni-campaign/corpus",
    "/mnt/lexar/omni-campaign/clones", "/home/cereal/omni-campaign/distill/corpus",
    "/home/cereal/omni-campaign/distill/queues", "/home/cereal/omni-campaign/registry",
    "/home/cereal/omni-campaign/ledger", "/home/cereal/omni-campaign/ledger.jsonl",
]
UNSURE = {130: "rescope-UNSURE-media_b-corpora-teacher-cache-manifests",
          102: "rescope-UNSURE-trainer-lane-receipts-teacher-cache"}
FOLD_UNITS = {7: "never-delete-fold-k3-oct09-L61-92-only-copy"}

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

old = json.load(open(f"{IN}/approved_server.json"))
prot = json.load(open(f"{IN}/protected_server.json"))
os.makedirs(OUT, exist_ok=False)

approved = []
for u in old:
    if u["plan_index"] not in APPROVED:
        continue
    u = json.loads(json.dumps(u))
    u.setdefault("notes", []).append("rescope-20261021: approved by lead review after the fold-init ruling")
    if u["plan_index"] == 210:
        u["reason"] = (f"byte duplicate of {CLONE_D2} (directory_sha256 03da9ab938dab873f20474e634a6dc18060325b07a62e9121abf91b26184dece "
                       "on both sides, clone receipt ae3060ebb4ec3d47); that clone is the surviving copy of the D1 delta, is "
                       "never-delete and is the delta link of distill/d2/p1_plan.json; D1 delta superseded by full-KD run 1")
    approved.append(u)
assert sorted(u["plan_index"] for u in approved) == sorted(APPROVED), sorted(u["plan_index"] for u in approved)
assert len({u["path"] for u in approved}) == len(approved)

new_prot = list(prot)
for p in (CLONE_D2, CLONE_D2 + ".clone-receipt.json", CLONE_D1, CLONE_D1 + ".clone-receipt.json", P1_PLAN):
    assert os.path.exists(p), p
    new_prot.append({"class": "never-delete-run1-reassembly-base", "path": p, "plan_index": None})
for p in FOLD_ROOTS:
    assert os.path.exists(p), p
    new_prot.append({"class": "never-delete-fold-or-record-root", "path": p, "plan_index": None})
for u in old:
    i = u["plan_index"]
    if i in APPROVED:
        continue
    cls = UNSURE.get(i) or FOLD_UNITS.get(i) or "rescope-not-approved-20261021"
    new_prot.append({"class": cls, "path": u["path"], "plan_index": i})

# no protected path may overlap an approved unit (else the gate would refuse it anyway)
def under(c, p): return c == p or c.startswith(p.rstrip("/") + "/")
clash = [(u["plan_index"], p["path"], p["class"]) for u in approved for p in new_prot
         if under(u["path"], p["path"]) or under(p["path"], u["path"])]
assert not clash, clash

json.dump(approved, open(f"{OUT}/approved_rescope.json", "w"), indent=1)
json.dump(new_prot, open(f"{OUT}/protected_rescope.json", "w"), indent=1)
receipt = {"tool": "build_rescope.py", "inputs": {"approved_server.json": sha(f"{IN}/approved_server.json"),
           "protected_server.json": sha(f"{IN}/protected_server.json")},
           "approved_plan_indexes": sorted(APPROVED), "approved_units": len(approved),
           "approved_du_bytes": sum(u["size_bytes_du"] for u in approved),
           "protected_entries": len(new_prot), "protected_added": len(new_prot) - len(prot),
           "outputs": {n: sha(f"{OUT}/{n}") for n in ("approved_rescope.json", "protected_rescope.json")}}
json.dump(receipt, open(f"{OUT}/BUILD_RECEIPT.json", "w"), indent=1)
print(json.dumps(receipt, indent=1))
for u in sorted(approved, key=lambda x: x["plan_index"]):
    print(u["plan_index"], u["volume"], u["size_gb"], u["path"])
