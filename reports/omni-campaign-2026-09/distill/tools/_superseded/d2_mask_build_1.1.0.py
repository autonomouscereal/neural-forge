#!/usr/bin/env python3
"""d2_mask_build.py -- D2-M1: the registry text's row exclusions and loss masks, from documents.jsonl x Codex's labels.

Card D2-M1 of STAGE_D_RECONCILED_PLAN_2026-10-14 (section 3.3, amendment A2 item 3).  The sealed D2 text set
(`distill/corpus/text`, 20 files x 256 rows x 4,096 positions, 4,883 rows) is never rewritten and its Q38 caches are
never recaptured: the exclusions and masks below are applied by streamed_kd_trainer (>= 1.3.4) at load time.

  exclude_rows.json  the packed rows that hold the source documents of Codex's 62 BFCL held-out tasks
                     (combined_oct13_v1 `bfcl-heldout:*` group ids -> normalized.jsonl recovery groups -> registry rids
                     -> document spans).  Whole rows go, so even attention exposure is removed and BFCL-62 stays a
                     valid secondary held-out (A2 3a; also excluded from D4 replay).
                     1.1.0 (BFCL-62 holdout decision, Claude lead 2026-10-14, option a): ALSO the packed rows of the
                     registry TWINS of those tasks -- other BFCL v4 records (simple_python vs multiple categories) that ask
                     the same request with the same function and argument values -- as listed by bfcl62_holdout.py
                     (`twin_exposure.twin_ids_by_source`, --holdout, pinned sha; its 62 rids and 69 rows must equal this
                     tool's own group-join derivation).  The same file carries the twin rows of the OTHER D2 inputs,
                     keyed by their corpus sha256 the way the trainer matches them: the K3 row holding a twin (--k3-dir,
                     pass-k/row_meta.json -> pass-k/qwen_input_ids.safetensors) and any self-gen rescore row whose prompt
                     is a twin (--selfgen-rows/--selfgen-corpus; none were rescored on 2026-10-14, which is recorded).
                     The native-v3 twins are dropped by d2_bfcl_native_copy.py >= 1.1.0, not here.
  all_mask.json      every loss term off (KD, CE, EOS, UL) on the other BFCL v4 JSON-in-prose documents of
                     `manual_cleanup_batch_000166_bfcl_v4_function_calls_ready_append_current` (A2 3b), i.e. not the 62
                     sources and not the text twins (whose rows are excluded).  Their rows stay: other documents share
                     them.
  gt_mask.json       the ground-truth terms off (CE, EOS, UL; Q38 KL kept) on the documents Codex's quarantine labels
                     `teacher_rewrite_exposes_current_answer` (CURRENT_TARGET rewrites) or
                     `tool_route_has_no_structured_calls` (flattened tool records) (A2 3c).  Where an all-mask span and a
                     gt-mask span overlap the trainer zeroes everything, i.e. the all-mask wins.

POSITION CONVENTION (the trainer reads exactly this).  The loss terms at position p of a row predict token p+1.  A
document occupying the global stream tokens [s, e) (documents.jsonl `start`, `start + kept`; the document's closing
separator is its last token) masks the global POSITIONS [s - 1, e - 1): every position whose target token belongs to
the document, including the one that predicts its first token and the one that predicts its closing separator.  The
separator position itself (predicting the next document's first token) is not the document's.  Global position g
lives in row g // 4,096 at column g % 4,096, so a document cut across rows masks the last column of the earlier row
(whose stored teacher distribution predicts the cut continuation).  Spans are written per corpus FILE, keyed by that
file's sha256 (the trainer matches it to its Q-text cache's corpus), with LOCAL row numbers and half-open column
ranges: [row, col_start, col_end).

INPUTS (each pinned by sha256 in the receipt; a different sha than --expect-* is a refusal):
  --documents         distill/corpus/text/documents.jsonl (ce6ff058...)
  --corpus-manifest   distill/corpus/text/manifest.json (its `shards` give each file's rows and sha256)
  --normalized        Codex chimera_corpus_recovery_oct13/normalized.jsonl (168c4dc9...): registry_origin.rid,
                      recovery_group_id, quality.blockers
  --eval-tasks        Codex chimera_recovery_eval_combined_oct13_v1/tasks.jsonl (3795f333...)
  --holdout           distill/corpus/holdouts/bfcl62_rids.json (bfcl62_holdout 1.0.0, 538678aa...)
  --k3-dir            distill/corpus/k3 (only needed when the holdout lists a K3 twin)
  --selfgen-rows      the self-gen rescore export's rows.json (row i -> prompt rid), with --selfgen-corpus the
                      rescore_target.safetensors the Q38 self-gen cache was built on (its metadata names the export)

`check-doc-index` (run where the corpus files are) re-derives every span from the files' own `doc_index` tensors and
refuses any disagreement, so the spans are proven against the packed corpus, not only against documents.jsonl.

Usage:
  python d2_mask_build.py build --documents D --corpus-manifest M --normalized N --eval-tasks T --holdout H \\
      --k3-dir K --selfgen-rows R --selfgen-corpus C --out DIR
  python d2_mask_build.py check-doc-index --out DIR --corpus-manifest M [--corpus-dir DIR]
  python d2_mask_build.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "d2_mask_build"
TOOL_VERSION = "1.1.0"  # 2026-10-14: BFCL-62 task twins excluded (text rows, K3 row, self-gen rows); --holdout cross-check
# 1.0.0 2026-10-14: first version (lane d2prep, card D2-M1); _superseded/d2_mask_build_1.0.0.py

EXCLUDE_SCHEMA = "omni-d2-row-exclusion-v1"
HOLDOUT_SCHEMA = "omni-holdout-bfcl62-rids-v1"
K3_PASSES = 4
MASK_SCHEMA = "omni-d2-position-mask-v1"
RECEIPT_SCHEMA = "omni-d2-mask-build-receipt-v1"
CONVENTION = ("position-indexed: the terms at position p predict token p+1; a document of global tokens [s, e) masks "
              "global positions [s-1, e-1); spans are [local_row, col_start, col_end) per corpus file")

SEQ = 4096
BFCL_DATASET = "manual_cleanup_batch_000166_bfcl_v4_function_calls_ready_append_current"
BFCL_TASK_PREFIX = "bfcl-heldout:"
GT_BLOCKERS = ("teacher_rewrite_exposes_current_answer", "tool_route_has_no_structured_calls")

#: what the X04 lane printed on the same inputs (2026-10-14); a build that disagrees is refused unless --no-expect
EXPECT = {
    "documents_sha256": "ce6ff058b175e7b0558917faaef737b7487b90e1854aa762571410fe43511fcf",
    "normalized_sha256": "168c4dc994db5858f0dcea3854c6983625f3c692876925c7c4b880b4f327c51e",
    "eval_tasks_sha256": "3795f333998627bb3b2d6fd460c19228469b2286284dfaadf30c5e12d23713f8",
    "holdout_sha256": "538678aacbdd1e9dfe83253b30d2276f4da3d3349724c7e7c78a3cbf5d594cb0",
    "documents": 35985, "rows": 4883, "bfcl_docs": 761, "bfcl_tokens": 293384,
    "bfcl62_docs": 62, "bfcl62_rows": 69, "gt_rewrite_docs": 2412, "gt_flat_docs": 5791, "gt_union_tokens": 3263593,
    # 1.1.0: the twin decision (bfcl62_rids.json twin_exposure, 2026-10-14T20:15:39Z)
    "text_twin_docs": 7, "text_twin_rows": 7, "excluded_rows": 76, "all_mask_docs": 692, "k3_twin_rows": 1,
}


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def log(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------
# Pure span arithmetic (tested on its own).
# --------------------------------------------------------------------------


def doc_positions(start: int, end: int, total: int) -> tuple[int, int]:
    """Global positions [s-1, e-1) of a document of global tokens [s, e), clipped to [0, total)."""

    if not 0 <= start < end <= total:
        raise Refusal(f"document span [{start}, {end}) is not inside the stream [0, {total})")
    return max(0, start - 1), end - 1


def split_by_row(lo: int, hi: int, seq: int | None = None) -> list[tuple[int, int, int]]:
    """Global positions [lo, hi) -> [(global_row, col_start, col_end)], one per row touched."""

    seq = SEQ if seq is None else seq
    out = []
    g = lo
    while g < hi:
        row, col = divmod(g, seq)
        stop = min(hi, (row + 1) * seq)
        out.append((row, col, stop - row * seq))
        g = stop
    return out


def merge_spans(spans: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Sort and merge overlapping / touching [row, a, b) spans of the same row."""

    merged: list[list[int]] = []
    for row, a, b in sorted(spans):
        if merged and merged[-1][0] == row and a <= merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], b)
        else:
            merged.append([row, a, b])
    return [tuple(x) for x in merged]  # type: ignore[misc]


def span_positions(spans: Iterable[tuple[int, int, int]]) -> int:
    return sum(b - a for _, a, b in spans)


def rows_of_docs(docs: Iterable[dict[str, Any]], seq: int | None = None) -> set[int]:
    seq = SEQ if seq is None else seq
    rows = set()
    for d in docs:
        rows.update(range(d["start"] // seq, (d["start"] + d["kept"] - 1) // seq + 1))
    return rows


def k3_twin_rows(row_metas: dict[int, list[dict[str, Any]]], twin_rids: Iterable[str]) -> dict[int, list[int]]:
    """{pass: [local row_index, ...]} of the K3 rows (pass-k/row_meta.json `rows`) that hold any twin rid."""

    twins = set(twin_rids)
    out: dict[int, list[int]] = {}
    for k, rows in sorted(row_metas.items()):
        hit = sorted(int(r["row_index"]) for r in rows if twins & set(r["rids"]))
        if hit:
            out[k] = hit
    return out


def selfgen_twin_rows(row_ids: list[str], twin_rids: Iterable[str]) -> list[int]:
    """Rows of the self-gen rescore corpus (row i was sampled from prompt rid row_ids[i]) whose prompt is a twin."""

    twins = set(twin_rids)
    return [i for i, rid in enumerate(row_ids) if rid in twins]


def safetensors_header(path: Path) -> dict[str, Any]:
    """The JSON header of a safetensors file (tensor shapes and __metadata__), read without torch / numpy."""

    with open(path, "rb") as handle:
        n = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(n))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def shard_table(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards = []
    for entry in manifest["shards"]:
        lo, hi = int(entry["rows"][0]), int(entry["rows"][1])
        shards.append({"file": Path(entry["path"]).name, "path": entry["path"], "sha256": entry["sha256"], "row_offset": lo, "rows": hi - lo})
    shards.sort(key=lambda s: s["row_offset"])
    expect = 0
    for s in shards:
        if s["row_offset"] != expect:
            raise Refusal(f"corpus manifest shards are not contiguous at row {expect} ({s['file']} starts at {s['row_offset']})")
        expect += s["rows"]
    return shards


def per_file(global_spans: Iterable[tuple[int, int, int]], shards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Global-row spans -> {corpus sha: {file, row_offset, spans: [[local_row, a, b], ...]}}."""

    out: dict[str, dict[str, Any]] = {}
    by_row = {}
    for s in shards:
        for r in range(s["row_offset"], s["row_offset"] + s["rows"]):
            by_row[r] = s
    for row, a, b in global_spans:
        s = by_row.get(row)
        if s is None:
            raise Refusal(f"global row {row} is in no corpus file")
        entry = out.setdefault(s["sha256"], {"file": s["file"], "row_offset": s["row_offset"], "rows_in_file": s["rows"], "spans": []})
        entry["spans"].append([row - s["row_offset"], a, b])
    return out


def mask_document(kind: str, spans: list[tuple[int, int, int]], shards: list[dict[str, Any]], docs: int, reason: str) -> dict[str, Any]:
    files = per_file(spans, shards)
    body = {"schema": MASK_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "kind": kind, "convention": CONVENTION,
            "terms_off": ["q38", "k3", "ce", "eos", "ul"] if kind == "all" else ["ce", "eos", "ul"], "reason": reason,
            "seq_len": SEQ, "documents": docs, "positions": span_positions(spans), "spans_total": len(spans), "corpora": files}
    body["spans_sha256"] = canonical_sha256({k: v["spans"] for k, v in sorted(files.items())})
    return body


def build(args: argparse.Namespace) -> int:
    shas = {"documents_sha256": sha256_file(args.documents), "normalized_sha256": sha256_file(args.normalized),
            "eval_tasks_sha256": sha256_file(args.eval_tasks), "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
            "holdout_sha256": sha256_file(args.holdout)}
    for key in ("documents_sha256", "normalized_sha256", "eval_tasks_sha256", "holdout_sha256"):
        log(f"{key} {shas[key]}")
        if not args.no_expect and shas[key] != EXPECT[key]:
            raise Refusal(f"{key} {shas[key][:16]} is not the pinned {EXPECT[key][:16]} (give --no-expect to build on new inputs)")
    manifest = json.loads(Path(args.corpus_manifest).read_text(encoding="utf-8"))
    shards = shard_table(manifest)
    rows_total = sum(s["rows"] for s in shards)
    total = rows_total * SEQ
    if manifest.get("documents_file", {}).get("sha256") not in (None, shas["documents_sha256"]):
        raise Refusal("the corpus manifest names another documents.jsonl sha256")
    docs = list(read_jsonl(args.documents))
    labels: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for r in read_jsonl(args.normalized):
        rid = ((r.get("provenance") or {}).get("registry_origin") or {}).get("rid")
        if rid:
            labels[rid] = {"uid": r.get("id"), "split": r.get("split"), "blockers": tuple((r.get("quality") or {}).get("blockers") or ())}
            groups[r["provenance"]["recovery_group_id"]].append(rid)
    missing = [d["rid"] for d in docs if d["rid"] not in labels or labels[d["rid"]]["uid"] != d["uid"]]
    if missing:
        raise Refusal(f"{len(missing)} documents have no matching normalized record (first {missing[:3]}); the join must be total")
    tasks = list(read_jsonl(args.eval_tasks))
    bfcl_tasks = [t for t in tasks if str(t.get("id", "")).startswith(BFCL_TASK_PREFIX)]
    other_tasks = [t for t in tasks if not str(t.get("id", "")).startswith(BFCL_TASK_PREFIX)]
    by_rid = {d["rid"]: d for d in docs}
    bfcl62_rids = sorted({rid for t in bfcl_tasks for rid in groups.get(t["group_id"], []) if rid in by_rid})
    other_hits = sorted({rid for t in other_tasks for rid in groups.get(t.get("group_id", ""), []) if rid in by_rid})
    bfcl_docs = [d for d in docs if d["dataset_id"] == BFCL_DATASET]
    bfcl62_docs = [by_rid[r] for r in bfcl62_rids]
    not_bfcl = [d["rid"] for d in bfcl62_docs if d["dataset_id"] != BFCL_DATASET]
    if not_bfcl:
        raise Refusal(f"{len(not_bfcl)} BFCL held-out source documents are not in {BFCL_DATASET}: {not_bfcl[:3]}")
    bfcl62_rows = sorted(rows_of_docs(bfcl62_docs))
    bfcl62_set = set(bfcl62_rids)

    # 1.1.0: the holdout file (bfcl62_holdout, task-provenance path) must name the same 62 documents and 69 rows as this
    # tool's group join; then its twin list decides the extra exclusions
    holdout = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
    if holdout.get("schema") != HOLDOUT_SCHEMA:
        raise Refusal(f"{args.holdout}: schema {holdout.get('schema')!r} != {HOLDOUT_SCHEMA}")
    if sorted(holdout.get("rids") or []) != bfcl62_rids or holdout.get("global_rows") != bfcl62_rows:
        raise Refusal(f"{args.holdout} names {len(holdout.get('rids') or [])} rids / {len(holdout.get('global_rows') or [])} rows "
                      f"that differ from this tool's group join ({len(bfcl62_rids)} / {len(bfcl62_rows)})")
    twins = (holdout.get("twin_exposure") or {}).get("twin_ids_by_source") or {}
    text_twin_rids = sorted(twins.get("text") or [])
    absent = [r for r in text_twin_rids if r not in by_rid]
    if absent:
        raise Refusal(f"{len(absent)} text twin rids are not documents of the registry text (first {absent[:2]})")
    if set(text_twin_rids) & bfcl62_set:
        raise Refusal("a text twin is itself a BFCL-62 source document")
    twin_docs = [by_rid[r] for r in text_twin_rids]
    twin_rows = sorted(rows_of_docs(twin_docs) - set(bfcl62_rows))
    want_twin_rows = sorted((holdout.get("twin_exposure") or {}).get("extra_text_rows_if_text_twins_excluded") or [])
    if twin_rows != want_twin_rows:
        raise Refusal(f"text twin rows {twin_rows} != the holdout's extra_text_rows_if_text_twins_excluded {want_twin_rows}")
    if twins.get("long"):
        raise Refusal(f"the holdout lists {len(twins['long'])} long-corpus twins; this tool has no long-row exclusion path")
    excluded_rows = sorted(set(bfcl62_rows) | set(twin_rows))
    twin_set = set(text_twin_rids)
    all_docs = [d for d in bfcl_docs if d["rid"] not in bfcl62_set and d["rid"] not in twin_set]

    # the other D2 inputs: their twin rows, keyed by the corpus sha256 their Q38 cache was built on
    other_corpora: dict[str, dict[str, Any]] = {}
    other_sources: dict[str, Any] = {}
    k3_twins = sorted(twins.get("k3") or [])
    if k3_twins:
        if not args.k3_dir:
            raise Refusal(f"the holdout lists {len(k3_twins)} K3 twin(s); give --k3-dir")
        metas = {k: json.loads((Path(args.k3_dir) / f"pass-{k}" / "row_meta.json").read_text(encoding="utf-8"))["rows"]
                 for k in range(K3_PASSES)}
        hits = k3_twin_rows(metas, k3_twins)
        found = {rid for k, rows in hits.items() for r in rows for rid in metas[k][r]["rids"]} & set(k3_twins)
        if found != set(k3_twins):
            raise Refusal(f"K3 twin rids not found in any pass row_meta: {sorted(set(k3_twins) - found)[:3]}")
        other_sources["k3"] = {"twin_rids": k3_twins, "passes": {}}
        for k, rows in hits.items():
            corpus = Path(args.k3_dir) / f"pass-{k}" / "qwen_input_ids.safetensors"
            shape = safetensors_header(corpus)["input_ids"]["shape"]
            if int(shape[0]) != len(metas[k]) or any(int(m["row_index"]) != i for i, m in enumerate(metas[k])):
                raise Refusal(f"K3 pass {k}: row_meta ({len(metas[k])} rows) does not index {corpus} ({shape[0]} rows) one to one")
            sha = sha256_file(corpus)
            other_corpora[sha] = {"file": f"k3/pass-{k}/qwen_input_ids.safetensors", "row_offset": None, "rows_in_file": int(shape[0]),
                                  "rows": rows, "source": "k3"}
            other_sources["k3"]["passes"][str(k)] = {"corpus_sha256": sha, "rows": rows, "rows_in_pass": int(shape[0]),
                                                     "row_rids": {str(r): metas[k][r]["rids"] for r in rows}}
    sg_twins = sorted(twins.get("selfgen") or [])
    if sg_twins:
        if not (args.selfgen_rows and args.selfgen_corpus):
            raise Refusal(f"the holdout lists {len(sg_twins)} self-gen twin prompt(s); give --selfgen-rows and --selfgen-corpus")
        rows_doc = json.loads(Path(args.selfgen_rows).read_text(encoding="utf-8"))
        header = safetensors_header(Path(args.selfgen_corpus))
        meta = header.get("__metadata__") or {}
        n_rows = int(header["input_ids"]["shape"][0])
        if meta.get("source_export_sha256") != (rows_doc.get("corpus") or {}).get("sha256") or len(rows_doc["row_ids"]) != n_rows:
            raise Refusal(f"{args.selfgen_corpus} (export {str(meta.get('source_export_sha256'))[:16]}, {n_rows} rows) is not "
                          f"built from {args.selfgen_rows} (export {str((rows_doc.get('corpus') or {}).get('sha256'))[:16]}, "
                          f"{len(rows_doc['row_ids'])} rows)")
        sg_rows = selfgen_twin_rows(rows_doc["row_ids"], sg_twins)
        sha = sha256_file(Path(args.selfgen_corpus))
        if sg_rows:
            other_corpora[sha] = {"file": Path(args.selfgen_corpus).name, "row_offset": None, "rows_in_file": n_rows,
                                  "rows": sg_rows, "source": "selfgen"}
        other_sources["selfgen"] = {"twin_prompt_rids": sg_twins, "corpus_sha256": sha, "rows_in_corpus": n_rows,
                                    "rows": sg_rows, "rows_json_sha256": sha256_file(Path(args.selfgen_rows)),
                                    "note": ("no rescored row was sampled from a twin prompt: nothing to exclude" if not sg_rows
                                             else "rows sampled from twin prompts are excluded")}
    other_sources["native_train"] = {"twin_uids": sorted(twins.get("native_train") or []),
                                     "handled_by": "d2_bfcl_native_copy.py >= 1.1.0 (dropped from the BFCL-native copy)"}
    rewrite_docs = [d for d in docs if GT_BLOCKERS[0] in labels[d["rid"]]["blockers"]]
    flat_docs = [d for d in docs if GT_BLOCKERS[1] in labels[d["rid"]]["blockers"]]
    gt_docs = [d for d in docs if any(b in labels[d["rid"]]["blockers"] for b in GT_BLOCKERS)]

    def spans_of(selected: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
        out = []
        for d in selected:
            lo, hi = doc_positions(int(d["start"]), int(d["start"]) + int(d["kept"]), total)
            if int(d["end"]) != int(d["start"]) + int(d["kept"]):
                raise Refusal(f"document {d['doc']}: end {d['end']} != start + kept")
            out.extend(split_by_row(lo, hi))
        return merge_spans(out)

    all_spans, gt_spans = spans_of(all_docs), spans_of(gt_docs)
    excluded_set = set(excluded_rows)
    counts = {
        "documents": len(docs), "rows": rows_total, "positions": total,
        "bfcl_docs": len(bfcl_docs), "bfcl_tokens": sum(d["kept"] for d in bfcl_docs),
        "bfcl_tasks": len(bfcl_tasks), "bfcl_tasks_with_a_source_doc": sum(1 for t in bfcl_tasks if any(r in by_rid for r in groups.get(t["group_id"], []))),
        "non_bfcl_tasks": len(other_tasks), "non_bfcl_task_source_docs_in_corpus": len(other_hits),
        "bfcl62_docs": len(bfcl62_docs), "bfcl62_tokens": sum(d["kept"] for d in bfcl62_docs),
        "bfcl62_rows": len(bfcl62_rows),
        "text_twin_docs": len(twin_docs), "text_twin_tokens": sum(d["kept"] for d in twin_docs), "text_twin_rows": len(twin_rows),
        "excluded_rows": len(excluded_rows), "excluded_positions": len(excluded_rows) * SEQ,
        "positions_left_after_exclusion": (rows_total - len(excluded_rows)) * SEQ,
        "docs_sharing_excluded_rows": sum(1 for d in docs if rows_of_docs([d]) & excluded_set),
        "other_tokens_lost_with_excluded_rows": sum(
            sum(max(0, min(d["start"] + d["kept"], (r + 1) * SEQ) - max(d["start"], r * SEQ)) for r in rows_of_docs([d]) & excluded_set)
            for d in docs if d["rid"] not in bfcl62_set and d["rid"] not in twin_set),
        "k3_twin_rows": sum(len(p["rows"]) for p in (other_sources.get("k3") or {}).get("passes", {}).values()),
        "selfgen_twin_rows": len((other_sources.get("selfgen") or {}).get("rows") or []),
        "native_train_twins": len(other_sources["native_train"]["twin_uids"]),
        "all_mask_docs": len(all_docs), "all_mask_tokens": sum(d["kept"] for d in all_docs), "all_mask_positions": span_positions(all_spans),
        "gt_rewrite_docs": len(rewrite_docs), "gt_rewrite_tokens": sum(d["kept"] for d in rewrite_docs),
        "gt_flat_docs": len(flat_docs), "gt_flat_tokens": sum(d["kept"] for d in flat_docs),
        "gt_union_docs": len(gt_docs), "gt_union_tokens": sum(d["kept"] for d in gt_docs), "gt_mask_positions": span_positions(gt_spans),
        "gt_and_all_overlap_docs": len({d["rid"] for d in gt_docs} & {d["rid"] for d in all_docs}),
    }
    for key, value in counts.items():
        log(f"COUNT {key} {value}")
    if not args.no_expect:
        bad = {k: (counts[k], EXPECT[k]) for k in ("documents", "rows", "bfcl_docs", "bfcl_tokens", "bfcl62_docs", "bfcl62_rows",
                                                    "gt_rewrite_docs", "gt_flat_docs", "gt_union_tokens", "text_twin_docs",
                                                    "text_twin_rows", "excluded_rows", "all_mask_docs", "k3_twin_rows")
               if counts[k] != EXPECT[k]}
        if bad:
            raise Refusal(f"counts differ from the X04 reading / the 2026-10-14 twin decision on the same inputs: {bad}")
    out = Path(args.out)
    exclusion = {"schema": EXCLUDE_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
                 "reason": ("A2 3(a): packed rows holding the source documents of Codex's 62 BFCL held-out tasks (combined_oct13_v1) "
                            "and of their task twins (same request, same function and argument values) in every D2 input; "
                            "BFCL-62 holdout decision, Claude lead 2026-10-14, option a"),
                 "seq_len": SEQ,
                 "global_rows": excluded_rows, "global_rows_sha256": canonical_sha256(excluded_rows),
                 "bfcl62_rows": bfcl62_rows, "bfcl62_rows_sha256": canonical_sha256(bfcl62_rows),
                 "bfcl62_rids": bfcl62_rids, "bfcl62_rids_sha256": canonical_sha256(bfcl62_rids),
                 "text_twin_rids": text_twin_rids, "text_twin_rows": twin_rows,
                 "holdout": {"path": str(args.holdout), "sha256": shas["holdout_sha256"], "tool": holdout.get("tool"),
                             "tool_version": holdout.get("tool_version")},
                 "other_sources": other_sources,
                 "corpora": {}}
    for s in shards:
        local = [r - s["row_offset"] for r in excluded_rows if s["row_offset"] <= r < s["row_offset"] + s["rows"]]
        if local:
            exclusion["corpora"][s["sha256"]] = {"file": s["file"], "row_offset": s["row_offset"], "rows_in_file": s["rows"], "rows": local,
                                                 "source": "registry"}
    clash = set(other_corpora) & set(exclusion["corpora"])
    if clash:
        raise Refusal(f"a K3 / self-gen corpus sha equals a registry file sha: {sorted(clash)[:2]}")
    exclusion["corpora"].update(other_corpora)
    files = {
        "exclude_rows.json": exclusion,
        "all_mask.json": mask_document("all", all_spans, shards, len(all_docs),
                                       "A2 3(b): BFCL v4 JSON-in-prose documents other than the BFCL-62 sources and their "
                                       "(row-excluded) text twins: every term off"),
        "gt_mask.json": mask_document("gt", gt_spans, shards, len(gt_docs),
                                      "A2 3(c): Codex quarantine labels " + " / ".join(GT_BLOCKERS) + ": CE, EOS, UL off; KL kept"),
    }
    written = {name: write_json_atomic(out / name, body) for name, body in files.items()}
    receipt = {"schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__)),
               "argv": sys.argv, "inputs": {**shas, "documents": str(args.documents), "normalized": str(args.normalized),
                                            "eval_tasks": str(args.eval_tasks), "corpus_manifest": str(args.corpus_manifest),
                                            "holdout": str(args.holdout), "k3_dir": str(args.k3_dir) if args.k3_dir else None,
                                            "selfgen_rows": str(args.selfgen_rows) if args.selfgen_rows else None,
                                            "selfgen_corpus": str(args.selfgen_corpus) if args.selfgen_corpus else None},
               "convention": CONVENTION, "counts": counts, "outputs": written,
               "corpus_files": [{k: s[k] for k in ("file", "sha256", "row_offset", "rows")} for s in shards],
               "doc_index_check": "not run (run check-doc-index where the corpus files are)"}
    rsha = write_json_atomic(out / "build_receipt.json", receipt)
    for name, sha in written.items():
        log(f"WROTE {out / name} sha256 {sha}")
    log(f"WROTE {out / 'build_receipt.json'} sha256 {rsha}")
    return 0


# --------------------------------------------------------------------------
# check-doc-index: the spans against the packed files' own doc_index tensors
# --------------------------------------------------------------------------


def expected_positions_from_doc_index(doc_index: Any, docs_wanted: set[int]) -> list[tuple[int, int, int]]:
    """From the stream's doc_index [rows, L] (document numbers, -1 pad): a flat bool over global positions, True where
    the TARGET token (p+1 in the stream) belongs to a wanted document.  The stream continues across rows, so the last
    column of row r predicts row r+1's first token."""

    import numpy as np

    flat = np.asarray(doc_index).reshape(-1)
    wanted = np.isin(flat, np.fromiter(docs_wanted, dtype=np.int64)) if docs_wanted else np.zeros_like(flat, dtype=bool)
    target = np.zeros_like(wanted)
    target[:-1] = wanted[1:]
    return target


def check_doc_index(args: argparse.Namespace) -> int:
    import numpy as np
    from safetensors.numpy import load_file

    out = Path(args.out)
    manifest = json.loads(Path(args.corpus_manifest).read_text(encoding="utf-8"))
    shards = shard_table(manifest)
    receipt_path = out / "build_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    docs = list(read_jsonl(Path(receipt["inputs"]["documents"]) if not args.documents else args.documents))
    if sha256_file(Path(args.documents) if args.documents else Path(receipt["inputs"]["documents"])) != receipt["inputs"]["documents_sha256"]:
        raise Refusal("documents.jsonl moved since the build")
    masks = {k: json.loads((out / f"{k}_mask.json").read_text(encoding="utf-8")) for k in ("all", "gt")}
    exclusion = json.loads((out / "exclude_rows.json").read_text(encoding="utf-8"))
    # the wanted document numbers per mask, from the spans' own documents (by global position -> doc)
    rid_to_doc = {d["rid"]: int(d["doc"]) for d in docs}
    doc_rows = {int(d["doc"]): d for d in docs}
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None
    # one stream array over all files (doc_index is per position, 4,883 x 4,096 int32 = 80 MB)
    arrays = []
    for s in shards:
        path = (corpus_dir / s["file"]) if corpus_dir else Path(s["path"])
        if sha256_file(path) != s["sha256"]:
            raise Refusal(f"{path} does not digest to the manifest's {s['sha256'][:16]}")
        arrays.append(load_file(str(path))["doc_index"].astype(np.int64))
    doc_index = np.concatenate(arrays, axis=0)
    flat = doc_index.reshape(-1)
    problems = []
    # documents.jsonl spans == doc_index
    for d in docs:
        lo, hi = int(d["start"]), int(d["start"]) + int(d["kept"])
        if not bool((flat[lo:hi] == int(d["doc"])).all()):
            problems.append(f"doc {d['doc']} span [{lo},{hi}) disagrees with doc_index")
            if len(problems) > 5:
                break
    result = {"documents_checked": len(docs), "documents_disagreeing": len(problems)}
    for kind, mask in masks.items():
        spans = []
        for sha, entry in mask["corpora"].items():
            off = int(entry["row_offset"])
            spans += [(off + r, a, b) for r, a, b in entry["spans"]]
        got = np.zeros_like(flat, dtype=bool)
        for row, a, b in spans:
            got[row * SEQ + a: row * SEQ + b] = True
        docs_in = set(np.unique(flat[np.concatenate([np.arange(row * SEQ + a, row * SEQ + b) + 1 for row, a, b in spans])]).tolist()) if spans else set()
        docs_in.discard(-1)
        want = expected_positions_from_doc_index(doc_index, docs_in)
        equal = bool((got == want).all())
        result[f"{kind}_mask"] = {"positions": int(got.sum()), "documents_by_doc_index": len(docs_in), "equal_to_doc_index_derivation": equal,
                                  "documents_in_file": mask["documents"]}
        if not equal or len(docs_in) != int(mask["documents"]):
            problems.append(f"{kind}_mask: doc_index derivation differs ({int((got != want).sum())} positions) or documents "
                            f"{len(docs_in)} != {mask['documents']}")
    excl_docs = set(np.unique(doc_index[exclusion["global_rows"]].reshape(-1)).tolist()) - {-1}
    bfcl62_docs = {rid_to_doc[r] for r in exclusion["bfcl62_rids"]}
    twin_docs = {rid_to_doc[r] for r in exclusion.get("text_twin_rids") or []}
    wanted = bfcl62_docs | twin_docs
    result["exclusion"] = {"rows": len(exclusion["global_rows"]), "bfcl62_docs_inside_excluded_rows": len(bfcl62_docs & excl_docs),
                           "bfcl62_docs": len(bfcl62_docs), "text_twin_docs": len(twin_docs),
                           "text_twin_docs_inside_excluded_rows": len(twin_docs & excl_docs),
                           "docs_touching_excluded_rows": len(excl_docs)}
    if not wanted <= excl_docs:
        problems.append(f"{len(wanted - excl_docs)} BFCL-62 / text twin documents are not inside the excluded rows")
    # every excluded row actually holds a BFCL-62 or twin token (no row excluded for nothing)
    for row in exclusion["global_rows"]:
        if not set(doc_index[row].tolist()) & wanted:
            problems.append(f"excluded row {row} holds no BFCL-62 or text twin document")
    # the all-mask never names a BFCL-62 or twin document (their rows are gone, not masked)
    all_docs_named = set()
    for sha, entry in masks["all"]["corpora"].items():
        off = int(entry["row_offset"])
        for r, a, b in entry["spans"]:
            g = (off + r) * SEQ
            all_docs_named |= set(np.unique(flat[g + a + 1: g + b + 1]).tolist())
    if all_docs_named & wanted:
        problems.append(f"the all-mask covers {len(all_docs_named & wanted)} BFCL-62 / twin documents")
    result["problems"] = problems
    result["ok"] = not problems
    for key, value in result.items():
        log(f"DOC-INDEX {key} {json.dumps(value)}")
    receipt["doc_index_check"] = {**result, "tool_version": TOOL_VERSION, "corpus_files_verified": len(shards)}
    write_json_atomic(receipt_path, receipt)
    log(f"DOC-INDEX CHECK {'OK' if not problems else 'FAILED'}; receipt {receipt_path} sha256 {sha256_file(receipt_path)}")
    return 0 if not problems else 1


# --------------------------------------------------------------------------
# self-test and CLI
# --------------------------------------------------------------------------


def self_test() -> int:
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    check("doc at stream start loses no position before 0", doc_positions(0, 5, 100) == (0, 4))
    check("doc positions predict its tokens", doc_positions(10, 15, 100) == (9, 14))
    check("row split", split_by_row(4090, 4100, 4096) == [(0, 4090, 4096), (1, 0, 4)], str(split_by_row(4090, 4100, 4096)))
    check("merge touching", merge_spans([(0, 5, 9), (0, 9, 12), (1, 0, 2), (0, 1, 3)]) == [(0, 1, 3), (0, 5, 12), (1, 0, 2)])
    check("rows of a doc crossing rows", rows_of_docs([{"start": 4095, "kept": 2}], 4096) == {0, 1})
    metas = {0: [{"row_index": 0, "rids": ["a", "b"]}], 2: [{"row_index": 0, "rids": ["c"]}, {"row_index": 1, "rids": ["d", "t"]}]}
    check("k3 twin row found by any rid in the packed row", k3_twin_rows(metas, ["t"]) == {2: [1]}, str(k3_twin_rows(metas, ["t"])))
    check("self-gen twin rows by prompt rid; absent prompt -> none",
          selfgen_twin_rows(["x", "t", "y", "t"], ["t"]) == [1, 3] and selfgen_twin_rows(["x"], ["t"]) == [])
    print(f"SELF-TEST {'PASS' if not failures else 'FAIL'} ({7 - len(failures)}/7)", flush=True)
    return 0 if not failures else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="command")
    b = sub.add_parser("build")
    b.add_argument("--documents", required=True, type=Path)
    b.add_argument("--corpus-manifest", required=True, type=Path)
    b.add_argument("--normalized", required=True, type=Path)
    b.add_argument("--eval-tasks", required=True, type=Path)
    b.add_argument("--holdout", required=True, type=Path, help="bfcl62_holdout.py output (bfcl62_rids.json): rid cross-check + twins")
    b.add_argument("--k3-dir", type=Path, help="distill/corpus/k3 (pass-k/row_meta.json, pass-k/qwen_input_ids.safetensors)")
    b.add_argument("--selfgen-rows", type=Path, help="self-gen rescore_export.rows.json (row i -> prompt rid)")
    b.add_argument("--selfgen-corpus", type=Path, help="self-gen rescore_target.safetensors (the Q38 self-gen cache's corpus)")
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--no-expect", action="store_true", help="build on inputs or counts other than the pinned X04 reading")
    c = sub.add_parser("check-doc-index")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--corpus-manifest", required=True, type=Path)
    c.add_argument("--corpus-dir", type=Path, help="where the text-*.safetensors are, if not at the manifest's paths")
    c.add_argument("--documents", type=Path, help="documents.jsonl, if not at the build receipt's path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.version:
        print(f"{TOOL_NAME} {TOOL_VERSION}")
        return 0
    if args.self_test:
        return self_test()
    try:
        if args.command == "build":
            return build(args)
        if args.command == "check-doc-index":
            return check_doc_index(args)
    except Refusal as refusal:
        print(f"REFUSED: {refusal.message}", file=sys.stderr, flush=True)
        return 2
    parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
