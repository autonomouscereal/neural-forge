#!/usr/bin/env python3
"""d2_holdout_eos_audit.py -- two read-only D2 facts as one JSON receipt (research lane, Claude, 2026-10-15).

  1. Codex's recovery held-out splits inside D2.  Every row of Codex's corpus-schema-v2 validation / test is joined to
     the registry by exact identifier (the Codex `id` equals a registry record identifier), then to carve.json and to D2's
     documents.jsonl; each joined D2 document is classed by the sealed A2 masks (d2_masks_v1): excluded row, all_mask
     (every term off), gt_mask (CE/EOS/UL off, KL kept) or trained (CE + KL).
  2. End-of-turn targets after cut rows.  For every D2 document of a cut row (quality_scorer class 06 unclosed think /
     08 generic web cut mid-sentence / 14 cut mid-word) the position whose target is the document's <|im_end|> is
     classed the same way.

Nothing is written outside --out.  Inputs are hashed into the receipt.

    python3 d2_holdout_eos_audit.py --scores <quality run>/scores.jsonl --out <dir>/d2_holdout_eos_audit.json
"""
import argparse
import bisect
import collections
import hashlib
import json
import os
import time

OC = "/home/cereal/omni-campaign"
SEQ = 4096
TOOL = ("d2_holdout_eos_audit", "1.0.0")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spans(path):
    m = json.load(open(path))
    out = collections.defaultdict(list)
    for c in m["corpora"].values():
        for row, c0, c1 in c["spans"]:
            out[c["row_offset"] + row].append((c0, c1))
    for v in out.values():
        v.sort()
    return out


def covered(spans, pos):
    row, col = divmod(pos, SEQ)
    lst = spans.get(row, [])
    i = bisect.bisect_right(lst, (col, 10 ** 9)) - 1
    return i >= 0 and lst[i][0] <= col < lst[i][1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--rows", default=f"{OC}/distill/corpus/staging/rows.jsonl")
    p.add_argument("--carve", default=f"{OC}/distill/corpus/carve.json")
    p.add_argument("--documents", default=f"{OC}/distill/corpus/text/documents.jsonl")
    p.add_argument("--masks", default=f"{OC}/distill/corpus/d2_masks_v1")
    p.add_argument("--codex", default=f"{OC}/distill/codex-preparation-01a1326f-20261013/corpus-schema-v2")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    gt, al = load_spans(f"{a.masks}/gt_mask.json"), load_spans(f"{a.masks}/all_mask.json")
    excluded = set(json.load(open(f"{a.masks}/exclude_rows.json"))["global_rows"])

    def state(pos):
        return ("excluded_row" if pos // SEQ in excluded else "all_mask" if covered(al, pos)
                else "gt_mask_kl_kept" if covered(gt, pos) else "trained_ce_kl")

    cls = {}
    for line in open(a.scores):
        r = json.loads(line)
        cls[r["rid"]] = r["class"]
    ident = {}
    for line in open(a.rows):
        x = json.loads(line)
        for key in ("record_id", "uid", "evidence_key", "text_key", "rid"):
            if x.get(key):
                ident.setdefault(str(x[key]), x["rid"])
    carve = json.load(open(a.carve))["assignment"]
    codex = {}
    split_stats = {}
    for split in ("validation", "test"):
        n = joined = 0
        csplit = collections.Counter()
        for line in open(f"{a.codex}/{split}.jsonl"):
            n += 1
            rid = ident.get(json.loads(line)["id"])
            if rid:
                joined += 1
                codex[rid] = split
                csplit[str(carve.get(rid))] += 1
        split_stats[split] = {"rows": n, "joined_by_exact_id": joined, "carve_split_of_joined": dict(csplit)}
    hold_docs = collections.Counter()
    hold_tok = collections.Counter()
    hold_rows = collections.defaultdict(set)
    eos = collections.Counter()
    for line in open(a.documents):
        d = json.loads(line)
        s, e, kept = int(d["start"]), int(d["end"]), int(d["kept"])
        if d["rid"] in codex:
            st = state((s + e) // 2)
            key = f"{codex[d['rid']]}:{st}"
            hold_docs[key] += 1
            hold_tok[key] += kept
            hold_rows[codex[d["rid"]]].add(d["rid"])
        c = cls.get(d["rid"])
        if c in ("06", "08", "14") and kept == int(d["length"]):
            eos[f"{c}:{state(e - 4)}"] += 1
    out = {"schema": "omni-research.d2-holdout-eos-audit.v1", "tool": {"name": TOOL[0], "version": TOOL[1], "sha256": sha(__file__)},
           "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "inputs": {k: {"path": v, "sha256": sha(v)} for k, v in (("scores", a.scores), ("rows", a.rows), ("carve", a.carve),
                      ("documents", a.documents), ("gt_mask", f"{a.masks}/gt_mask.json"), ("all_mask", f"{a.masks}/all_mask.json"),
                      ("exclude_rows", f"{a.masks}/exclude_rows.json"), ("codex_validation", f"{a.codex}/validation.jsonl"),
                      ("codex_test", f"{a.codex}/test.jsonl"))},
           "codex_heldout_in_d2": {"splits": split_stats, "d2_rows_by_split": {k: len(v) for k, v in hold_rows.items()},
                                   "d2_docs_by_split_and_mask": dict(hold_docs), "d2_tokens_by_split_and_mask": dict(hold_tok),
                                   "d2_tokens_total": sum(hold_tok.values()), "rule": "document midpoint position classed by the A2 masks"},
           "cut_doc_im_end_target_by_class_and_mask": dict(eos),
           "rule_eos": "position end-4 of a document of a cut row predicts its <|im_end|> (packed tail <|im_end|> \\n <|endoftext|>)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({k: out[k] for k in ("codex_heldout_in_d2", "cut_doc_im_end_target_by_class_and_mask")}, indent=1))


if __name__ == "__main__":
    main()
