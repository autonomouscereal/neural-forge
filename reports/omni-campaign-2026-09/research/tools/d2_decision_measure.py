#!/usr/bin/env python3
"""d2_decision_measure.py -- the read-only numbers behind D2_DECISIONS_2026-10-15 (research lane, Claude, 2026-10-15).

  1. Codex's canonical recovery held-out split (chimera_text_jobs_oct13_v8: native-text-v3 short + v4 long + v7 executed
     tools, validation and test) as row ids, cross-checked against corpus-schema-v2 (the audit's input), joined to registry
     rids the way d2_holdout_eos_audit 1.0.0 does, then located in every D2 input: registry documents (positions classed by
     the A2 masks, packed rows touched), K3 rows, long documents, self-gen rescore rows, BFCL-native rows.
  2. Truncated registry documents (quality_scorer classes 06 / 08 / 14, tail not packed away): their end-of-turn target
     positions [e-4, e-1) (targets <|im_end|>, the newline, <|endoftext|>) classed by the A2 masks, with the tail ids read
     from the packed shards.
  3. Refill pool sizes (tool-call self-gen progress, Codex train-split sources outside the registry, registry spare).

Nothing is written outside --out.  Every input is hashed into the receipt.  CPU only.

    CUDA_VISIBLE_DEVICES= nice -n 15 python d2_decision_measure.py --scores <quality run>/scores.jsonl --out <file>
"""
import argparse
import bisect
import collections
import hashlib
import json
import os
import time

import numpy as np
from safetensors.numpy import load_file

OC = "/home/cereal/omni-campaign"
CP = f"{OC}/distill/codex-preparation-01a1326f-20261013"
SEQ = 4096
TOOL = ("d2_decision_measure", "1.0.0")
HELD_PACKS = {"short": "native-text-v3", "long": "native-text-v4", "executed_tools": "native-text-v7"}
TRAIN_PACKS = {"short": "native-text-v3", "long": "native-text-v4", "executed_tools": "native-text-v7",
               "verified_intelligence": "native-text-v8"}
CUT_CLASSES = ("06", "08", "14")
IM_END, NEWLINE, ENDOFTEXT = 248046, 198, 248044


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


def pack_ids(name, split):
    m = json.load(open(f"{CP}/{name}-{split}/manifest.json"))
    ids = [r["row_id"] for r in m["row_receipts"]]
    return ids, {"rows": m["rows"], "tokens": m["tokens"], "supervised_tokens": m["supervised_tokens"],
                 "admission": m["admission"], "split": m["split"], "manifest_sha256": sha(f"{CP}/{name}-{split}/manifest.json")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--rows", default=f"{OC}/distill/corpus/staging/rows.jsonl")
    p.add_argument("--carve", default=f"{OC}/distill/corpus/carve.json")
    p.add_argument("--documents", default=f"{OC}/distill/corpus/text/documents.jsonl")
    p.add_argument("--corpus-manifest", default=f"{OC}/distill/corpus/text/manifest.json")
    p.add_argument("--masks", default=f"{OC}/distill/corpus/d2_masks_v1")
    p.add_argument("--schema-v2", default=f"{CP}/corpus-schema-v2")
    p.add_argument("--k3-dir", default=f"{OC}/distill/corpus/k3")
    p.add_argument("--long-tasks", default=f"{OC}/distill/corpus/long/tasks.json")
    p.add_argument("--selfgen-rows", required=True, help="the self-gen rescore export's rows.json (row -> prompt rid)")
    p.add_argument("--bfcl-native-rows", default=f"{OC}/distill/corpus/bfcl_native_v1/rows.json")
    p.add_argument("--holdout", default=f"{OC}/distill/corpus/holdouts/bfcl62_rids.json")
    p.add_argument("--toolcall-progress", default=f"{OC}/distill/toolcall/run-q38-q4km-2026-10-14/progress.json")
    p.add_argument("--registry-draw", type=int, default=18_850_000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    inputs = {}

    def pin(key, path):
        inputs[key] = {"path": path, "sha256": sha(path)}
        return path

    # ---- 1. the canonical held-out ids
    held, held_meta = {}, {}
    for lane, name in HELD_PACKS.items():
        for split in ("validation", "test"):
            ids, meta = pack_ids(name, split)
            held_meta[f"{name}-{split}"] = meta
            for i in ids:
                held[i] = f"{lane}:{split}"
    schema = {}
    for split in ("validation", "test"):
        for line in open(pin(f"schema_v2_{split}", f"{a.schema_v2}/{split}.jsonl")):
            schema[json.loads(line)["id"]] = split
    short_long = {i for i, v in held.items() if not v.startswith("executed_tools")}
    cross = {"schema_v2_ids": len(schema), "v3_v4_heldout_ids": len(short_long),
             "v3_v4_equals_schema_v2": short_long == set(schema),
             "split_labels_agree": all(held[i].split(":")[1] == s for i, s in schema.items() if i in held)}
    train_meta = {}
    train_ids = {}
    for lane, name in TRAIN_PACKS.items():
        ids, meta = pack_ids(name, "train")
        train_meta[f"{name}-train"] = meta
        train_ids[lane] = set(ids)

    ident = {}
    for line in open(pin("rows", a.rows)):
        x = json.loads(line)
        for key in ("record_id", "uid", "evidence_key", "text_key", "rid"):
            if x.get(key):
                ident.setdefault(str(x[key]), x["rid"])
    held_rid = {}
    unjoined = collections.Counter()
    for i, lab in held.items():
        rid = ident.get(i)
        if rid:
            held_rid[rid] = lab
        else:
            unjoined[lab] += 1
    carve = json.load(open(pin("carve", a.carve)))["assignment"]
    carve_of_held = collections.Counter(str(carve.get(r)) for r in held_rid)

    # ---- BFCL-62 sources and twins
    hold = json.load(open(pin("holdout", a.holdout)))
    excl = json.load(open(pin("exclude_rows", f"{a.masks}/exclude_rows.json")))
    b62 = set(hold["rids"])
    twins = set(excl.get("text_twin_rids", []))
    b62_info = {"bfcl62_rids": len(b62), "bfcl62_rids_in_heldout": len(b62 & set(held_rid)),
                "text_twin_rids": len(twins), "text_twin_rids_in_heldout": len(twins & set(held_rid)),
                "text_twin_rids_codex_train": sum(1 for t in twins if any(ident.get(i) == t for s in train_ids.values() for i in s))}

    # ---- 2. registry documents
    gt = load_spans(pin("gt_mask", f"{a.masks}/gt_mask.json"))
    al = load_spans(pin("all_mask", f"{a.masks}/all_mask.json"))
    excluded = set(excl["global_rows"])

    def state(pos):
        return ("excluded_row" if pos // SEQ in excluded else "all_mask" if covered(al, pos)
                else "gt_mask_kl_kept" if covered(gt, pos) else "trained_ce_kl")

    cls = {}
    for line in open(pin("scores", a.scores)):
        r = json.loads(line)
        cls[r["rid"]] = r["class"]
    cman = json.load(open(pin("corpus_manifest", a.corpus_manifest)))
    rows_total = int(cman["rows"])
    shards = {}
    for s in cman["shards"]:
        shards[s["path"]] = (s["rows"][0], s["rows"][1], s["sha256"])
    ids_cache = {}

    def token_at(g):
        row, col = divmod(g, SEQ)
        for path, (r0, r1, _) in shards.items():
            if r0 <= row < r1:
                if path not in ids_cache:
                    t = load_file(path)
                    key = "input_ids" if "input_ids" in t else sorted(t)[0]
                    ids_cache[path] = (key, t[key])
                return int(ids_cache[path][1][row - r0, col])
        raise KeyError(g)

    held_pos = collections.Counter()
    held_docs = collections.Counter()
    row_held = collections.Counter()
    eos_pos = collections.Counter()
    eos_docs = collections.Counter()
    tails = collections.Counter()
    eos_spans = 0
    for line in open(pin("documents", a.documents)):
        d = json.loads(line)
        s, kept, length = int(d["start"]), int(d["kept"]), int(d["length"])
        e = s + kept
        rid = d["rid"]
        if rid in held_rid:
            held_docs[held_rid[rid]] += 1
            for g in range(max(s - 1, 0), e - 1):
                held_pos[state(g)] += 1
                row_held[g // SEQ] += 1
        c = cls.get(rid)
        if c in CUT_CLASSES and kept == length:
            eos_docs[f"{c}:{'heldout' if rid in held_rid else 'train'}"] += 1
            tails[f"{token_at(e - 3)},{token_at(e - 2)},{token_at(e - 1)}"] += 1
            eos_spans += 1
            for g in range(e - 4, e - 1):
                eos_pos[f"{'heldout' if rid in held_rid else 'train'}:{state(g)}"] += 1
    touched = sorted(row_held)
    share = collections.Counter()
    for r in touched:
        f = row_held[r] / SEQ
        share["100%" if f >= 1.0 else ">=90%" if f >= 0.9 else ">=50%" if f >= 0.5 else ">=10%" if f >= 0.1 else "<10%"] += 1
    new_rows = [r for r in touched if r not in excluded]
    registry_positions = rows_total * SEQ
    left_after_bfcl = registry_positions - len(excluded) * SEQ
    trained_lost = held_pos["trained_ce_kl"]
    registry = {
        "rows_total": rows_total, "positions_total": registry_positions,
        "positions_left_after_a2_3a": left_after_bfcl,
        "heldout_docs": dict(held_docs), "heldout_positions_by_mask": dict(held_pos),
        "heldout_positions_total": sum(held_pos.values()),
        "rows_touched": len(touched), "rows_touched_not_already_excluded": len(new_rows),
        "rows_touched_by_heldout_share": dict(share),
        "option_row_exclusion": {"rows": len(new_rows), "positions_removed": len(new_rows) * SEQ,
                                 "positions_left": left_after_bfcl - len(new_rows) * SEQ,
                                 "covers_registry_draw": left_after_bfcl - len(new_rows) * SEQ >= a.registry_draw},
        "option_span_mask": {"positions_masked_all_terms": sum(held_pos.values()) - held_pos["excluded_row"],
                             "trained_ce_kl_lost": trained_lost, "kl_only_lost": held_pos["gt_mask_kl_kept"],
                             "positions_left": left_after_bfcl,
                             "covers_registry_draw": left_after_bfcl >= a.registry_draw},
        "rule": "document positions [start-1, start+kept-1) (d2_mask_build POSITION CONVENTION), classed by the A2 masks"}

    # ---- other D2 inputs
    k3 = {}
    for pdir in sorted(os.listdir(a.k3_dir)):
        if not pdir.startswith("pass-"):
            continue
        meta = json.load(open(pin(f"k3_{pdir}_row_meta", f"{a.k3_dir}/{pdir}/row_meta.json")))
        hit = [r["row_index"] for r in meta["rows"] if any(x in held_rid for x in r.get("rids", []))]
        k3[pdir] = {"rows": len(meta["rows"]), "rows_holding_heldout": hit}
    lt = json.load(open(pin("long_tasks", a.long_tasks)))
    long_hits = [t["doc"] for t in lt["tasks"] if any(x in held_rid for x in t.get("rows", []))]
    sg = json.load(open(pin("selfgen_rows", a.selfgen_rows)))
    sg_rids = sg["row_ids"]  # rescore row n was sampled from prompt rid row_ids[n]
    sg_hits = [i for i, r in enumerate(sg_rids) if r in held_rid]
    sg_limit = [i for i, end in enumerate(sg["row_end"]) if end != "eog"]
    sg_limit_last = collections.Counter()
    corpus = sg["corpus"]["path"]
    if os.path.isfile(corpus):
        t = load_file(pin("selfgen_corpus", corpus))
        lengths = next((t[k] for k in ("lengths", "length", "row_length") if k in t), None)
        ids = next((t[k] for k in ("input_ids", "ids", "tokens") if k in t), None)
        if lengths is not None and ids is not None:
            for i in sg_limit:
                last = int(ids[i, int(lengths[i]) - 1])
                sg_limit_last["eos" if last in (IM_END, ENDOFTEXT) else "not_eos"] += 1
        else:
            sg_limit_last["unverified: tensors " + ",".join(sorted(t))] += 1
    bn = json.load(open(pin("bfcl_native_rows", a.bfcl_native_rows)))
    bn_hits = [r["out_row"] for r in bn["rows"] if r["row_id"] in held]
    others = {"k3": k3, "long_docs_holding_heldout": long_hits, "long_docs": len(lt["tasks"]),
              "selfgen_rescore_rows": len(sg_rids), "selfgen_rows_from_heldout_prompts": sg_hits,
              "selfgen_rows_limit_ended": len(sg_limit), "selfgen_limit_rows_last_id": dict(sg_limit_last),
              "bfcl_native_rows": len(bn["rows"]), "bfcl_native_rows_in_heldout": bn_hits,
              "toolcall_selfgen": "synthetic prompts (plan tc-*), no row id join; needs the 13-gram screen against the held-out text"}

    # ---- 3. refill pools
    tp = json.load(open(pin("toolcall_progress", a.toolcall_progress)))
    reg_rids = set()
    for line in open(a.documents):
        reg_rids.add(json.loads(line)["rid"])
    pools = {"toolcall_selfgen_progress": {k: tp.get(k) for k in ("accepted", "accepted_tokens", "token_target", "updated_utc")}}
    for lane, ids in train_ids.items():
        name = TRAIN_PACKS[lane]
        joined = {ident.get(i) for i in ids} - {None}
        pools[f"codex_{lane}_train"] = {**train_meta[f"{name}-train"], "rows_joined_to_registry": len(joined),
                                        "rows_already_in_d2_registry": len(joined & reg_rids),
                                        "rows_in_heldout": len(ids & set(held))}
    pools["registry_spare_after_span_mask"] = left_after_bfcl - a.registry_draw
    pools["registry_draw"] = a.registry_draw

    out = {"schema": "omni-research.d2-decision-measure.v1",
           "tool": {"name": TOOL[0], "version": TOOL[1], "sha256": sha(__file__)},
           "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "canonical_split": {"job_manifest": "proof_outputs/chimera_text_jobs_oct13_v8 (supersedes v1-v4, v7; v5-v8 carry identical sources)",
                               "packs": held_meta, "cross_check_vs_schema_v2": cross,
                               "heldout_ids": len(held), "heldout_ids_by_lane": dict(collections.Counter(held.values())),
                               "joined_to_registry": len(held_rid), "unjoined_by_lane": dict(unjoined),
                               "carve_split_of_joined": dict(carve_of_held)},
           "bfcl62": b62_info, "registry": registry, "other_inputs": others,
           "eos_truncated": {"docs": dict(eos_docs), "docs_total": eos_spans, "tail_ids": dict(tails.most_common(5)),
                             "positions_by_mask": dict(eos_pos),
                             "rule": f"classes {CUT_CLASSES} and kept == length; positions [e-4, e-1) predict <|im_end|>, newline, <|endoftext|>"},
           "refill_pools": pools, "inputs": inputs}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({k: out[k] for k in ("canonical_split", "bfcl62", "registry", "other_inputs", "eos_truncated",
                                          "refill_pools")}, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
