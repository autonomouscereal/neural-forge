#!/usr/bin/env python3
"""d2_clean_repack.py -- D2-X1 clean: the registry text re-packed WITHOUT any held-out document (lane d2clean).

Lead decision 2026-10-15 (Claude, under `cyril-claude-leads-all-lanes-2026-10-14` and the standing GO; an upgrade over the
span mask of `d2-codex-heldout-exclusion-2026-10-15`).  The 5,055 held-out registry documents (Codex val/test, canonical
split of `$Q/D2_DECISIONS_2026-10-15.json`) plus the BFCL-62 sources and their 7 text twins touch 3,386 of the 4,883
packed rows, and streamed_kd_trainer does not block attention at document boundaries, so the span mask still leaves
held-out text as attention CONTEXT for later documents of a shared row.  This tool builds the context-clean alternative
as a NEW immutable corpus version; the sealed `distill/corpus/text` and its Q38 caches are never touched.

THE RULE (one epoch, every non-held-out document exactly once, whole):
  X           = the held-out rids (the six heldout_packs of the decisions file, joined to registry rids by the
                d2_holdout_eos_audit rule) + the BFCL-62 source rids + the A2 3(a) text twin rids.
  A (affected)= every sealed packed row holding a token of an X document (3,386 rows).  K = the other rows (1,497).
                K rows are used AS THEY ARE: byte-identical, so their sealed Q38 cache rows stay valid.
  re-pack     = every non-X document with any token in an A row, WHOLE (its tokens read from the sealed shards, never
                re-tokenized), in the sealed allocation order (documents.jsonl `doc`, which keeps the route
                interleave), stream-packed into rows of 4,096 exactly like `distill_corpus_build.stream_pack`: concatenated,
                cut every 4,096 positions, a document may continue into the next row; the last row is right-padded
                with the separator (doc_index -1, route 255, `real_lengths` < 4,096; qtext stores no padded position).
                Shards of 256 rows, the sealed tensor names and metadata keys, plus provenance per document.
  boundary    = a re-packed document that also has tokens in a K row (1,737 documents): it is trained once, whole, in
                the re-pack; in its K row(s) its positions are loss-masked (all terms) and stay plain train-text context.
  K-row mask  = every position of a K row that belongs (d2_mask_build POSITION CONVENTION: a document of global tokens
                [s, e) owns positions [s-1, e-1)) to an X document or to a re-packed document.
So no held-out token is ever an input or a target, and no train document is trained twice.

MASKS CARRIED OVER (the A2 / D2 decisions are rules about documents, so they move with the document):
  A2 3(b) all-terms  documents whose positions the pinned v1 `all_mask.json` covers (read off its spans; the spans must
                     be an exact union of whole documents, else refused) -- K-row copies and re-pack positions;
  A2 3(c) gt         the same for the pinned v1 `gt_mask.json`;
  D2-X2 EOS (gt)     every flagged truncated document (quality_scorer classes 06/08/14 of the decisions file's scores
                     file, kept == length) gets CE/EOS/UL off on the three positions [e-4, e-1) predicting
                     <|im_end|>, newline, <|endoftext|> (guard: those three ids, else refused);
  A2 3(a)            the v1 `exclude_rows.json` rows are all inside A; its non-registry entries (the K3 twin row) are
                     copied unchanged into `clean_exclude_rows.json`, so the file replaces v1 for every D2 source.
Outputs (one dir, written to a temp name and renamed into place only when complete; an existing --out is refused):
  text-clean-NNNNN.safetensors   input_ids int32, route_ids uint8, doc_index int32, real_lengths int32 [rows, 4096]
  documents.jsonl                the sealed columns in re-pack coordinates + source_doc / source_start / source rows
  clean_exclude_rows.json        omni-d2-row-exclusion-v1: the A rows per sealed file + v1's non-registry entries
  clean_all_mask.json            omni-d2-position-mask-v1 kind all, keyed by corpus sha (sealed K rows + re-pack files)
  clean_gt_mask.json             omni-d2-position-mask-v1 kind gt, same keys
  manifest.json                  counts, every input pinned, shard shas, the token-conservation identity, the post-pack
                                 review screen, sealed with manifest_sha256 (distill_corpus_build.write_json)
`verify` re-derives everything from the written files and the sealed shards (ids per document, doc_index, routes,
masks, no X document anywhere, conservation) and writes `verify_receipt.json` beside them.  `state` prints whether the
clean version is built, verified and fully cached (exit 0) or not (exit 10): d2_plan uses the clean registry only when
`state` is 0 at data-config time and otherwise keeps the span mask (D2 never waits for this lane).

CPU only.  numpy + safetensors (+ tokenizers for the screen).  Runs from distill/tools (imports distill_corpus_build and
d2_mask_build from beside it, so the packer and the position convention are the sealed builders' own code).

    CUDA_VISIBLE_DEVICES= nice -n 15 $PY d2_clean_repack.py build --out $OC/distill/corpus/text_clean_v1
    CUDA_VISIBLE_DEVICES= nice -n 15 $PY d2_clean_repack.py verify --out $OC/distill/corpus/text_clean_v1
    $PY d2_clean_repack.py state --out $OC/distill/corpus/text_clean_v1
    $PY d2_clean_repack.py --self-test
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import d2_mask_build as MB  # noqa: E402  (position convention, span helpers, mask/exclusion schemas)
import distill_corpus_build as DCB  # noqa: E402  (stream_pack, write_safetensors, write_json, the review screen)

TOOL_NAME = "d2_clean_repack"
TOOL_VERSION = "1.0.1"  # 2026-10-15: the decisions file's eos_mask.flag.source is "<path> (<tool note>)": read the path
# 1.0.0 2026-10-15: first version (lane d2clean); refused at its first real build on that string; _superseded/d2_clean_repack_1.0.0.py
SCHEMA = "omni-d2-clean-repack-v1"
EXIT_WAITING = 10

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
LEX = Path(os.environ.get("LEX", "/mnt/lexar/omni-campaign"))
DEFAULT_DECISIONS = OC / "distill" / "queues" / "D2_DECISIONS_2026-10-15.json"
DEFAULT_CORPUS = OC / "distill" / "corpus" / "text"
DEFAULT_ROWS = OC / "distill" / "corpus" / "staging" / "rows.jsonl"
DEFAULT_MASKS_V1 = OC / "distill" / "corpus" / "d2_masks_v1"
DEFAULT_OUT = OC / "distill" / "corpus" / "text_clean_v1"
DEFAULT_CACHE_ROOT = LEX / "teacher_cache" / "qtext" / "q38-27bu"

SHARD_ROWS = 256  # distill_corpus_build.TEXT_SHARD_ROWS
FILE_PREFIX = "text-clean-"
JOIN_KEYS = ("record_id", "uid", "evidence_key", "text_key", "rid")  # d2_holdout_eos_audit 1.0.0 join rule
EOS_TAIL = (248046, 198, 248044)  # <|im_end|>, newline, <|endoftext|> (D2-X2 guard)

#: pins of the inputs this version was built for (a different input is refused unless --no-expect)
#: $Q/D2_DECISIONS_2026-10-15.json as recorded in STAGE_D_RULINGS (sha c8d47f22...)
DECISIONS_SHA256 = "c8d47f22784d265a2e9239f9274c062298d25278497ba21d01102aa45274d90d"
EXPECT = {
    "documents_sha256": MB.EXPECT["documents_sha256"],
    "heldout_ids": 5391, "heldout_rids_joined": 5055, "x_rids": 5061, "x_docs": 5061, "x_tokens": 3_180_411,
    "rows": 4883, "affected_rows": 3386, "kept_rows": 1497, "repack_docs": 22_970, "boundary_docs": 1737,
    "repack_tokens": 12_349_164,
}
#: v1 masks: the same pins d2_plan 1.2.0 MASK_SHAS carries (checked against it at build)
MASK_V1_SHAS = {
    "exclude_rows.json": "55d9c51030f742a4f5969ae8abd736b4226727f24f57113941da51f12d85c864",
    "all_mask.json": "23b0186e5104fc34c9437e4f3051059797caa36212f9c55ec1dbc0b7aabcf598",
    "gt_mask.json": "148bbebc3ab16e6469d271b41d088aabc36430714817f9106ce6d1cf8d4dfda2",
}


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path | str) -> str:
    return MB.sha256_file(Path(path))


def canonical_sha256(value: Any) -> str:
    return MB.canonical_sha256(value)


def write_json_atomic(path: Path, value: Any) -> str:
    return MB.write_json_atomic(Path(path), value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(MB.read_jsonl(Path(path)))


def tool_identity() -> dict[str, Any]:
    return {"tool": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__),
            "distill_corpus_build": {"version": DCB.VERSION, "sha256": sha256_file(DCB.__file__)},
            "d2_mask_build": {"version": MB.TOOL_VERSION, "sha256": sha256_file(MB.__file__)},
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# --------------------------------------------------------------------------
# Pure planning (tested on its own).
# --------------------------------------------------------------------------


def doc_rows(doc: dict[str, Any], seq: int) -> range:
    s, e = int(doc["start"]), int(doc["start"]) + int(doc["kept"])
    return range(s // seq, (e - 1) // seq + 1)


def plan_repack(docs: list[dict[str, Any]], rows_total: int, seq: int, x_rids: set[str]) -> dict[str, Any]:
    """The whole decision as data: affected rows, kept rows, the re-packed documents (in `doc` order), the X documents,
    the documents left whole in K rows, and the K-row all-terms spans (global [row, a, b), merged)."""

    total = rows_total * seq
    ordered = sorted(docs, key=lambda d: int(d["doc"]))
    x_docs = [d for d in ordered if d["rid"] in x_rids]
    affected = set()
    for d in x_docs:
        affected.update(doc_rows(d, seq))
    repack, kept_whole, boundary = [], [], []
    for d in ordered:
        if d["rid"] in x_rids:
            continue
        rows = set(doc_rows(d, seq))
        if rows & affected:
            repack.append(d)
            if rows - affected:
                boundary.append(d)
        else:
            kept_whole.append(d)
    kept_rows = [r for r in range(rows_total) if r not in affected]
    kept_set = set(kept_rows)
    spans = []
    for d in x_docs + repack:
        lo, hi = MB.doc_positions(int(d["start"]), int(d["start"]) + int(d["kept"]), total)
        spans.extend(s for s in MB.split_by_row(lo, hi, seq) if s[0] in kept_set)
    return {"affected_rows": sorted(affected), "kept_rows": kept_rows, "x_docs": x_docs, "repack": repack,
            "boundary": boundary, "kept_whole": kept_whole, "kept_row_all_spans": MB.merge_spans(spans)}


def pack_rows(lengths: list[int], seq: int) -> int:
    return max(1, math.ceil(sum(lengths) / seq))


def covered_docs(spans: Iterable[tuple[int, int, int]], docs: list[dict[str, Any]], seq: int, total: int, what: str) -> set[int]:
    """The documents a pinned v1 mask names: those whose whole convention span lies inside the mask.  Refuses a mask that
    is not an exact union of whole documents (then a per-document carry-over would be wrong)."""

    by_row: dict[int, list[tuple[int, int]]] = {}
    for row, a, b in MB.merge_spans(spans):
        by_row.setdefault(row, []).append((a, b))

    def inside(row: int, a: int, b: int) -> bool:
        lst = by_row.get(row, [])
        i = bisect.bisect_right(lst, (a, 1 << 62)) - 1
        return i >= 0 and lst[i][0] <= a and b <= lst[i][1]

    named, union = set(), []
    for d in docs:
        lo, hi = MB.doc_positions(int(d["start"]), int(d["start"]) + int(d["kept"]), total)
        parts = MB.split_by_row(lo, hi, seq)
        if parts and all(inside(*p) for p in parts):
            named.add(int(d["doc"]))
            union.extend(parts)
    if MB.merge_spans(union) != MB.merge_spans(spans):
        got, want = MB.span_positions(MB.merge_spans(union)), MB.span_positions(MB.merge_spans(spans))
        raise Refusal(f"{what}: its spans are not an exact union of whole documents ({want:,} positions in the file, "
                      f"{got:,} in the documents it covers whole)")
    return named


def eos_span(end: int) -> tuple[int, int]:
    """D2-X2: positions [e-4, e-1) predict the document's last three tokens (<|im_end|>, newline, <|endoftext|>)."""

    return end - 4, end - 1


# --------------------------------------------------------------------------
# Inputs.
# --------------------------------------------------------------------------


def pinned(path: Path, want: str | None, what: str, no_expect: bool) -> str:
    if not Path(path).is_file():
        raise Refusal(f"{what} {path} does not exist")
    got = sha256_file(path)
    if want and got != want and not no_expect:
        raise Refusal(f"{what} {path} hashes {got[:16]}, pinned {want[:16]}")
    return got


def heldout_rids(decisions: dict[str, Any], rows_path: Path, no_expect: bool) -> tuple[dict[str, str], dict[str, Any]]:
    """rid -> 'pack:split' for the canonical split, joined by the decisions' join rule; plus the pins and counts."""

    held: dict[str, str] = {}
    packs = []
    for pack in decisions["canonical_split"]["heldout_packs"]:
        mpath = Path(pack["manifest"])
        sha = pinned(mpath, pack["manifest_sha256"], f"held-out pack {pack['name']}", no_expect)
        m = json.loads(mpath.read_text(encoding="utf-8"))
        ids = [r["row_id"] for r in m["row_receipts"]]
        if len(ids) != int(pack["rows"]) and not no_expect:
            raise Refusal(f"held-out pack {pack['name']}: {len(ids)} rows, the decisions file says {pack['rows']}")
        for i in ids:
            held[str(i)] = pack["name"]
        packs.append({"name": pack["name"], "manifest": str(mpath), "sha256": sha, "rows": len(ids)})
    ident: dict[str, str] = {}
    for x in MB.read_jsonl(rows_path):
        for key in JOIN_KEYS:
            if x.get(key):
                ident.setdefault(str(x[key]), x["rid"])
    joined: dict[str, str] = {}
    unjoined = Counter()
    for i, label in held.items():
        rid = ident.get(i)
        if rid:
            joined[rid] = label
        else:
            unjoined[label] += 1
    return joined, {"packs": packs, "heldout_ids": len(held), "joined_rids": len(joined), "unjoined_by_pack": dict(unjoined)}


def load_mask_spans(path: Path, shards: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """A d2_mask_build mask file -> global [row, a, b) spans over the sealed registry files (other corpora ignored)."""

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    offsets = {s["sha256"]: s["row_offset"] for s in shards}
    out = []
    for sha, entry in (doc.get("corpora") or {}).items():
        if sha in offsets:
            out.extend((offsets[sha] + int(r), int(a), int(b)) for r, a, b in entry["spans"])
    return MB.merge_spans(out)


def load_sealed(shards: list[dict[str, Any]], seq: int) -> dict[str, np.ndarray]:
    ids, doc_index, routes = [], [], []
    for s in shards:
        from safetensors.numpy import load_file

        got = sha256_file(s["path"])
        if got != s["sha256"]:
            raise Refusal(f"{s['path']} hashes {got[:16]}, its manifest says {s['sha256'][:16]}")
        t = load_file(str(s["path"]))
        if t["input_ids"].shape != (s["rows"], seq):
            raise Refusal(f"{s['path']}: input_ids {t['input_ids'].shape} != ({s['rows']}, {seq})")
        ids.append(t["input_ids"].reshape(-1))
        doc_index.append(t["doc_index"].reshape(-1))
        routes.append(t["route_ids"].reshape(-1))
    return {"ids": np.concatenate(ids), "doc_index": np.concatenate(doc_index), "routes": np.concatenate(routes)}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    started = time.time()
    out = Path(args.out)
    if out.exists():
        raise Refusal(f"{out} exists: a clean corpus version is immutable (choose a new --out)")
    decisions_path = Path(args.decisions)
    pins: dict[str, Any] = {"decisions": {"path": str(decisions_path), "sha256": pinned(decisions_path, DECISIONS_SHA256, "decisions file", args.no_expect)}}
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions.get("schema") != "omni-campaign.d2-decisions.v1":
        raise Refusal(f"{decisions_path}: schema {decisions.get('schema')!r}")

    corpus = Path(args.corpus_dir)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seq = int(manifest["sequence_length"])
    pins["sealed_manifest"] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "manifest_sha256": manifest.get("manifest_sha256")}
    shards = MB.shard_table(manifest)
    for s in shards:  # the manifest's paths are the server's; a copied corpus is read beside its manifest
        local = corpus / s["file"]
        s["path"] = str(local) if local.is_file() else s["path"]
    rows_total = sum(s["rows"] for s in shards)
    total = rows_total * seq
    documents_path = corpus / "documents.jsonl"
    pins["documents"] = {"path": str(documents_path), "sha256": pinned(documents_path, (manifest.get("documents_file") or {}).get("sha256"), "documents.jsonl", False)}
    if not args.no_expect and pins["documents"]["sha256"] != EXPECT["documents_sha256"]:
        raise Refusal(f"documents.jsonl {pins['documents']['sha256'][:16]} is not the pinned {EXPECT['documents_sha256'][:16]}")
    docs = read_jsonl(documents_path)

    # ---- X: held-out + BFCL-62 + text twins, every list read from the decisions file's own pins
    joined, held_info = heldout_rids(decisions, Path(args.rows), args.no_expect)
    pins["rows_jsonl"] = {"path": str(args.rows), "sha256": sha256_file(args.rows)}
    sets = decisions["exclusion"]["sets"]
    b62_doc = json.loads(Path(sets["bfcl62"]["path"]).read_text(encoding="utf-8"))
    pins["bfcl62"] = {"path": sets["bfcl62"]["path"], "sha256": pinned(Path(sets["bfcl62"]["path"]), sets["bfcl62"]["sha256"], "bfcl62_rids.json", args.no_expect)}
    v1x_path = Path(sets["a2_3a_row_exclusion_unchanged"]["path"])
    masks_v1 = Path(args.masks_v1)
    if v1x_path.resolve() != (masks_v1 / "exclude_rows.json").resolve() and not args.no_expect:
        raise Refusal(f"the decisions file's A2 3(a) file {v1x_path} is not {masks_v1 / 'exclude_rows.json'}")
    for name, want in MASK_V1_SHAS.items():
        pins[f"v1_{name}"] = {"path": str(masks_v1 / name), "sha256": pinned(masks_v1 / name, None if args.no_expect else want, f"v1 {name}", args.no_expect)}
    if not args.no_expect and sets["a2_3a_row_exclusion_unchanged"]["sha256"] != MASK_V1_SHAS["exclude_rows.json"]:
        raise Refusal("the decisions file pins another A2 3(a) exclude_rows.json than d2_plan's MASK_SHAS")
    v1x = json.loads((masks_v1 / "exclude_rows.json").read_text(encoding="utf-8"))
    b62 = set(b62_doc["rids"])
    twins = set(v1x.get("text_twin_rids") or [])
    x_rids = set(joined) | b62 | twins
    by_rid = {d["rid"]: d for d in docs}
    absent = sorted(r for r in b62 | twins if r not in by_rid)
    if absent:
        raise Refusal(f"{len(absent)} BFCL-62 / twin rids are not registry documents (first {absent[:3]})")

    # ---- the plan
    plan = plan_repack(docs, rows_total, seq, x_rids)
    for d in docs:
        if int(d["end"]) != int(d["start"]) + int(d["kept"]):
            raise Refusal(f"document {d['doc']}: end != start + kept")
    if not set(v1x["global_rows"]) <= set(plan["affected_rows"]):
        raise Refusal("an A2 3(a) excluded row is not an affected row (its BFCL-62 / twin document must be in X)")
    repack = plan["repack"]
    affected_set, kept_set = set(plan["affected_rows"]), set(plan["kept_rows"])
    repack_tokens = sum(int(d["kept"]) for d in repack)
    x_tokens = sum(int(d["kept"]) for d in plan["x_docs"])
    whole_k_tokens = sum(int(d["kept"]) for d in plan["kept_whole"])
    counts = {
        "heldout_ids": held_info["heldout_ids"], "heldout_rids_joined": held_info["joined_rids"],
        "bfcl62_rids": len(b62), "text_twin_rids": len(twins), "x_rids": len(x_rids),
        "x_docs": len(plan["x_docs"]), "x_tokens": x_tokens, "rows": rows_total, "positions": total,
        "affected_rows": len(plan["affected_rows"]), "kept_rows": len(plan["kept_rows"]),
        "kept_row_runs": sum(1 for i, r in enumerate(plan["kept_rows"]) if i == 0 or plan["kept_rows"][i - 1] != r - 1),
        "repack_docs": len(repack), "boundary_docs": len(plan["boundary"]), "repack_tokens": repack_tokens,
        "boundary_tokens_in_kept_rows": sum(
            sum(max(0, min(int(d["start"]) + int(d["kept"]), (r + 1) * seq) - max(int(d["start"]), r * seq))
                for r in doc_rows(d, seq) if r not in affected_set) for d in plan["boundary"]),
        "kept_whole_docs": len(plan["kept_whole"]), "kept_whole_tokens": whole_k_tokens,
        "kept_row_all_mask_positions": MB.span_positions(plan["kept_row_all_spans"]),
        "documents": len(docs), "real_tokens_sealed": sum(int(d["kept"]) for d in docs),
    }
    if counts["real_tokens_sealed"] != x_tokens + repack_tokens + whole_k_tokens:
        raise Refusal("token conservation fails: X + re-pack + whole-in-K != every sealed document token")
    for key, value in counts.items():
        log(f"COUNT {key} {value}")
    if not args.no_expect:
        bad = {k: (counts[k], EXPECT[k]) for k in EXPECT if k in counts and counts[k] != EXPECT[k]}
        if bad:
            raise Refusal(f"counts differ from the 2026-10-15 reading on the same inputs: {bad}")

    # ---- the sealed shards: ids, doc_index, routes; every document's span proven against doc_index
    arrays = load_sealed(shards, seq)
    flat_ids, flat_doc, flat_route = arrays["ids"], arrays["doc_index"], arrays["routes"]
    routes = manifest["route_table"]
    pad_id = int(manifest["tokenizer"]["separator_id"])
    bad_docs = [int(d["doc"]) for d in docs if not bool((flat_doc[int(d["start"]): int(d["start"]) + int(d["kept"])] == int(d["doc"])).all())]
    if bad_docs:
        raise Refusal(f"{len(bad_docs)} documents disagree with the sealed doc_index (first {bad_docs[:3]})")
    if (flat_doc < 0).any():
        raise Refusal("the sealed registry has padded positions; this tool assumes the sealed stream is full")
    kept_flat = np.zeros(total, dtype=bool)
    for r in plan["kept_rows"]:
        kept_flat[r * seq: (r + 1) * seq] = True
    x_doc_ids = np.array(sorted(int(d["doc"]) for d in plan["x_docs"]), dtype=np.int64)
    if np.isin(flat_doc[kept_flat], x_doc_ids).any():
        raise Refusal("a kept row holds an X token (planner bug)")

    documents = []
    for d in repack:
        s, e = int(d["start"]), int(d["start"]) + int(d["kept"])
        ids = flat_ids[s:e]
        if not bool((flat_route[s:e] == int(routes[d["route"]])).all()):
            raise Refusal(f"document {d['doc']}: route ids disagree with route {d['route']}")
        if int(d["kept"]) == int(d["length"]):
            if int(ids[-1]) != pad_id or bool((ids[:-1] == pad_id).any()):
                raise Refusal(f"document {d['doc']}: not closed by exactly one separator {pad_id}")
        elif e != total:
            raise Refusal(f"document {d['doc']} is cut (kept {d['kept']} < length {d['length']}) but is not the stream's last")
        documents.append(ids.astype(np.int32).tolist())
    n_rows = pack_rows([len(x) for x in documents], seq)
    new_ids, new_doc_of, spans = DCB.stream_pack(documents, rows=n_rows, seq=seq, pad_id=pad_id)
    if len(spans) != len(repack) or any(sp["kept"] != sp["length"] for sp in spans):
        raise Refusal("the re-pack cut a document: rows were planned too short")
    new_route = np.full(new_ids.shape, 255, dtype=np.uint8)
    flat_new_route = new_route.reshape(-1)
    for sp in spans:
        flat_new_route[sp["start"]: sp["end"]] = int(routes[repack[sp["doc"]]["route"]])
    real_lengths_all = (new_doc_of >= 0).sum(axis=1).astype(np.int32)
    if int(real_lengths_all[:-1].min(initial=seq)) != seq:
        raise Refusal("a re-pack row other than the last is not full")

    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        files = []
        base_meta = {"schema": f"{DCB.SCHEMA}/text", "tool_version": DCB.VERSION, "sequence_length": str(seq),
                     "tokenizer_sha256": manifest["tokenizer"]["sha256"], "separator_id": str(pad_id), "pad_id": str(pad_id),
                     "route_table": json.dumps(routes, sort_keys=True), "route_pad": "255", "doc_index_pad": "-1",
                     "repack_tool": f"{TOOL_NAME} {TOOL_VERSION}", "repack_schema": SCHEMA,
                     "derived_from_manifest_sha256": str(manifest.get("manifest_sha256")),
                     "repack_rule": "registry text minus every held-out / BFCL-62 / twin document (D2_DECISIONS_2026-10-15 D2-X1, clean); non-held-out documents of affected rows re-packed whole"}
        for index, start in enumerate(range(0, n_rows, SHARD_ROWS)):
            stop = min(start + SHARD_ROWS, n_rows)
            name = f"{FILE_PREFIX}{index:05d}.safetensors"
            tensors = {"input_ids": new_ids[start:stop], "route_ids": new_route[start:stop], "doc_index": new_doc_of[start:stop],
                       "real_lengths": real_lengths_all[start:stop]}
            sha = DCB.write_safetensors(tmp / name, tensors, {**base_meta, "rows": f"{start}:{stop}"})
            files.append({"file": name, "path": str(out / name), "sha256": sha, "rows": [start, stop], "row_offset": start,
                          "tokens": int(real_lengths_all[start:stop].sum())})
        table = []
        for sp in spans:
            d = repack[sp["doc"]]
            s = int(d["start"])
            table.append({"doc": sp["doc"], "rid": d["rid"], "uid": d.get("uid"), "record_id": d.get("record_id"), "dataset_id": d["dataset_id"],
                          "route": d["route"], "render": d.get("render"), "start": sp["start"], "end": sp["end"], "length": int(d["length"]),
                          "kept": sp["kept"], "row_first": sp["start"] // seq, "row_last": (sp["end"] - 1) // seq,
                          "source_doc": int(d["doc"]), "source_start": s, "source_end": s + int(d["kept"]),
                          "source_row_first": s // seq, "source_row_last": (s + int(d["kept"]) - 1) // seq,
                          "source_class": "boundary" if any(r in kept_set for r in doc_rows(d, seq)) else "inside_affected"})
        docs_sha, _ = DCB.write_jsonl(tmp / "documents.jsonl", table)

        # ---- masks
        new_total = n_rows * seq
        new_files = [{"file": f["file"], "path": f["path"], "sha256": f["sha256"], "row_offset": f["row_offset"], "rows": f["rows"][1] - f["rows"][0]} for f in files]
        v1_all = load_mask_spans(masks_v1 / "all_mask.json", shards)
        v1_gt = load_mask_spans(masks_v1 / "gt_mask.json", shards)
        all_docs = covered_docs(v1_all, docs, seq, total, "v1 all_mask.json")
        gt_docs = covered_docs(v1_gt, docs, seq, total, "v1 gt_mask.json")
        eos = decisions["eos_mask"]["flag"]
        scores_path = Path(str(eos["source"]).split(" (")[0].strip())  # "<path> (quality_scorer 1.0.4)"
        pins["eos_scores"] = {"path": str(scores_path), "sha256": pinned(scores_path, eos["sha256"], "D2-X2 scores", args.no_expect)}
        cut_classes = set(eos["classes"])
        flag_rids = {r["rid"] for r in MB.read_jsonl(scores_path) if str(r.get("class")) in cut_classes}
        new_of_source = {t["source_doc"]: t for t in table}

        def k_spans(selected: set[int]) -> list[tuple[int, int, int]]:  # positions of whole-in-K documents
            outk = []
            for d in plan["kept_whole"]:
                if int(d["doc"]) in selected:
                    lo, hi = MB.doc_positions(int(d["start"]), int(d["start"]) + int(d["kept"]), total)
                    outk.extend(s for s in MB.split_by_row(lo, hi, seq) if s[0] in kept_set)
            return outk

        def r_spans(selected: set[int]) -> list[tuple[int, int, int]]:  # positions of re-packed documents
            outr = []
            for src in sorted(selected & set(new_of_source)):
                t = new_of_source[src]
                lo, hi = MB.doc_positions(t["start"], t["end"], new_total)
                outr.extend(MB.split_by_row(lo, hi, seq))
            return outr

        eos_k, eos_r, eos_docs, eos_refused = [], [], 0, []
        for d in docs:
            if d["rid"] not in flag_rids or d["rid"] in x_rids or int(d["kept"]) != int(d["length"]):
                continue
            e = int(d["start"]) + int(d["kept"])
            if tuple(int(v) for v in flat_ids[e - 3: e]) != EOS_TAIL:
                eos_refused.append(int(d["doc"]))
                continue
            eos_docs += 1
            if int(d["doc"]) in new_of_source:
                t = new_of_source[int(d["doc"])]
                eos_r.extend(MB.split_by_row(*eos_span(t["end"]), seq))
            else:
                eos_k.extend(s for s in MB.split_by_row(*eos_span(e), seq) if s[0] in kept_set)
        if eos_refused:
            raise Refusal(f"{len(eos_refused)} flagged documents do not end in {EOS_TAIL} (first {eos_refused[:3]})")
        all_k = MB.merge_spans(plan["kept_row_all_spans"] + k_spans(all_docs))
        all_r = MB.merge_spans(r_spans(all_docs))
        gt_k = MB.merge_spans(k_spans(gt_docs) + eos_k)
        gt_r = MB.merge_spans(r_spans(gt_docs) + eos_r)

        def mask_body(kind: str, k: list, r: list, reason: str, n_docs: int) -> dict[str, Any]:
            corpora = {**MB.per_file(k, shards), **MB.per_file(r, new_files)}
            body = {"schema": MB.MASK_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "kind": kind, "convention": MB.CONVENTION,
                    "terms_off": ["q38", "k3", "ce", "eos", "ul"] if kind == "all" else ["ce", "eos", "ul"], "reason": reason,
                    "seq_len": seq, "documents": n_docs, "positions": MB.span_positions(k) + MB.span_positions(r),
                    "positions_sealed_kept_rows": MB.span_positions(k), "positions_repack": MB.span_positions(r),
                    "spans_total": len(k) + len(r), "corpora": corpora}
            body["spans_sha256"] = canonical_sha256({key: v["spans"] for key, v in sorted(corpora.items())})
            return body

        all_body = mask_body("all", all_k, all_r,
                             "D2-X1 clean: in the sealed KEPT rows every position owned by an X (held-out / BFCL-62 / twin) or re-packed "
                             "document; plus A2 3(b) (the v1 all_mask documents, carried by document) in kept rows and in the re-pack",
                             len(all_docs - {int(d['doc']) for d in plan['x_docs']}))
        gt_body = mask_body("gt", gt_k, gt_r,
                            "A2 3(c) (the v1 gt_mask documents, carried by document) + D2-X2 (decisions eos_mask: classes "
                            f"{sorted(cut_classes)}, kept == length, positions [e-4, e-1)) in kept rows and in the re-pack; KL kept",
                            len(gt_docs - {int(d['doc']) for d in plan['x_docs']}))
        exclusion = {"schema": MB.EXCLUDE_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
                     "reason": ("D2-X1 clean: the sealed registry rows holding any token of a held-out (Codex canonical val/test), "
                                "BFCL-62 source or text twin document; their non-held-out documents are re-packed whole into the "
                                f"{FILE_PREFIX}* files. Non-registry entries are v1's A2 3(a) rows, unchanged."),
                     "seq_len": seq, "global_rows": plan["affected_rows"], "global_rows_sha256": canonical_sha256(plan["affected_rows"]),
                     "a2_3a_rows_inside": sorted(v1x["global_rows"]), "corpora": {}}
        for s in shards:
            local = [r - s["row_offset"] for r in plan["affected_rows"] if s["row_offset"] <= r < s["row_offset"] + s["rows"]]
            if local:
                exclusion["corpora"][s["sha256"]] = {"file": s["file"], "row_offset": s["row_offset"], "rows_in_file": s["rows"],
                                                     "rows": local, "source": "registry"}
        carried = {sha: e for sha, e in (v1x.get("corpora") or {}).items() if e.get("source") != "registry"}
        exclusion["corpora"].update(carried)
        exclusion["carried_from_v1"] = sorted(carried)
        written = {"clean_exclude_rows.json": write_json_atomic(tmp / "clean_exclude_rows.json", exclusion),
                   "clean_all_mask.json": write_json_atomic(tmp / "clean_all_mask.json", all_body),
                   "clean_gt_mask.json": write_json_atomic(tmp / "clean_gt_mask.json", gt_body)}

        # ---- the post-pack review screen (the sealed builder's own, over the new rows)
        if args.no_screen:
            screen = {"skipped": "--no-screen"}
        else:
            dargs = DCB.parse_args(["text"])
            index, _ = DCB.build_review_index(dargs)
            qwen = DCB.QwenText(Path(args.qwen_dir))
            screen = DCB.screen_packed(index, ((f"row {i}", qwen.decode(new_ids[i, : int(real_lengths_all[i])].tolist())) for i in range(n_rows)))
            if screen["units_hit"]:
                raise Refusal(f"re-packed rows hit the review surface: {screen['hits'][:3]}")

        kd_positions_kept = len(plan["kept_rows"]) * seq - MB.span_positions(all_k)
        repack_real = int(real_lengths_all.sum())
        body = {
            "schema": f"{DCB.SCHEMA}/text-clean", "repack_schema": SCHEMA, "identity": tool_identity(),
            "set": "(a) text, clean D2 version: registry text without any held-out document (D2-X1 clean)",
            "decision": "Claude-lead 2026-10-15 (lane d2clean), upgrade over d2-codex-heldout-exclusion-2026-10-15's span mask",
            "rule": __doc__.split("THE RULE")[1].split("MASKS CARRIED OVER")[0].strip(),
            "sequence_length": seq, "rows": n_rows, "positions": n_rows * seq, "real_tokens": repack_real,
            "pad_positions": n_rows * seq - repack_real, "documents": len(table),
            "documents_spanning_rows": sum(1 for t in table if t["row_first"] != t["row_last"]),
            "route_table": routes, "tensors": manifest.get("tensors"),
            "shards": files, "documents_file": {"path": str(out / "documents.jsonl"), "sha256": docs_sha},
            "sealed_corpus": {"manifest": str(manifest_path), "manifest_sha256": manifest.get("manifest_sha256"),
                              "use": "rows NOT in clean_exclude_rows.json, with clean_all_mask.json / clean_gt_mask.json; their sealed Q38 caches unchanged"},
            "masks": {name: {"path": str(out / name), "sha256": sha} for name, sha in written.items()},
            "counts": {**counts, "repack_rows": n_rows, "repack_files": len(files), "eos_flagged_docs_masked": eos_docs,
                       "a2_3b_docs": len(all_docs), "a2_3c_docs": len(gt_docs),
                       "all_mask_positions_kept_rows": MB.span_positions(all_k), "all_mask_positions_repack": MB.span_positions(all_r),
                       "gt_mask_positions_kept_rows": MB.span_positions(gt_k), "gt_mask_positions_repack": MB.span_positions(gt_r),
                       "kd_positions_kept_rows": kd_positions_kept, "kd_positions_repack": repack_real - MB.span_positions(all_r),
                       "kd_positions_clean_total": kd_positions_kept + repack_real - MB.span_positions(all_r)},
            "conservation": {"x_tokens": x_tokens, "repack_tokens": repack_tokens, "kept_whole_tokens": whole_k_tokens,
                             "sum": x_tokens + repack_tokens + whole_k_tokens, "sealed_real_tokens": counts["real_tokens_sealed"],
                             "holds": True},
            "heldout": held_info, "inputs": pins, "post_pack_screen": screen,
            "cache": {"teacher": "q38-27bu", "root": str(Path(args.cache_root)), "per_file": "one Q38 cache per text-clean file, "
                      "<root>/<file sha256>, built by the frozen qtext_teacher_cache 1.0.5 (the sealed registry cache's tool, f2465a11)"},
            "wall_seconds": round(time.time() - started, 2),
        }
        sha = DCB.write_json(tmp / "manifest.json", body, seal_field="manifest_sha256")
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"WROTE {out}: {n_rows} rows in {len(files)} files, {repack_real:,} real tokens, {len(table):,} documents; manifest_sha256 {sha}")
    for name, s in written.items():
        log(f"WROTE {out / name} sha256 {s}")
    log(f"KD positions: kept rows {kd_positions_kept:,} + re-pack {repack_real - MB.span_positions(all_r):,}")
    return 0


# --------------------------------------------------------------------------
# verify (from the written files and the sealed shards only)
# --------------------------------------------------------------------------


def verify(args: argparse.Namespace) -> int:
    from safetensors.numpy import load_file
    from safetensors import safe_open

    out = Path(args.out)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    body = {k: v for k, v in man.items() if k != "manifest_sha256"}
    if DCB.canonical_sha256(body) != man.get("manifest_sha256"):
        problems.append("manifest.json does not carry its own canonical sha (edited after the build)")
    seq = int(man["sequence_length"])
    sealed_man_path = Path(man["sealed_corpus"]["manifest"])
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else sealed_man_path.parent
    sealed_man = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    if sealed_man.get("manifest_sha256") != man["sealed_corpus"]["manifest_sha256"]:
        problems.append("the sealed corpus manifest moved since the build")
    shards = MB.shard_table(sealed_man)
    for s in shards:
        local = corpus_dir / s["file"]
        s["path"] = str(local) if local.is_file() else s["path"]
    total = sum(s["rows"] for s in shards) * seq
    sealed = load_sealed(shards, seq)
    ids_parts, doc_parts, route_parts, len_parts = [], [], [], []
    for f in man["shards"]:
        path = out / f["file"]
        if sha256_file(path) != f["sha256"]:
            problems.append(f"{f['file']} does not hash to its manifest sha")
            continue
        t = load_file(str(path))
        with safe_open(str(path), "np") as handle:
            meta = dict(handle.metadata())
        if meta.get("rows") != f"{f['rows'][0]}:{f['rows'][1]}" or meta.get("derived_from_manifest_sha256") != str(sealed_man.get("manifest_sha256")):
            problems.append(f"{f['file']}: metadata rows / derived_from disagree")
        ids_parts.append(t["input_ids"].reshape(-1)); doc_parts.append(t["doc_index"].reshape(-1))
        route_parts.append(t["route_ids"].reshape(-1)); len_parts.append(t["real_lengths"])
    for name, entry in (man.get("masks") or {}).items():
        if not (out / name).is_file() or sha256_file(out / name) != entry["sha256"]:
            problems.append(f"{name} does not hash to its manifest sha")
    if problems:  # a moved file: nothing below can be trusted
        return finish_verify(args, out, man, {"ok": False, "problems": problems[:20]}, problems)
    new_ids, new_doc = np.concatenate(ids_parts), np.concatenate(doc_parts)
    new_route, real_lengths = np.concatenate(route_parts), np.concatenate(len_parts)
    n_rows = len(real_lengths)
    if n_rows != int(man["rows"]) or new_ids.size != n_rows * seq:
        problems.append("row count disagrees with the manifest")
    table = read_jsonl(out / "documents.jsonl")
    if sha256_file(out / "documents.jsonl") != man["documents_file"]["sha256"]:
        problems.append("documents.jsonl moved")
    sealed_docs = {int(d["doc"]): d for d in read_jsonl(corpus_dir / "documents.jsonl")}
    routes = man["route_table"]
    position = 0
    for t in table:
        s, e = int(t["start"]), int(t["end"])
        src = sealed_docs[int(t["source_doc"])]
        if s != position or e - s != int(src["kept"]) or src["rid"] != t["rid"]:
            problems.append(f"doc {t['doc']}: span / source disagree")
        if not np.array_equal(new_ids[s:e], sealed["ids"][int(src["start"]): int(src["start"]) + int(src["kept"])]):
            problems.append(f"doc {t['doc']}: ids differ from sealed doc {src['doc']}")
        if not bool((new_doc[s:e] == int(t["doc"])).all()) or not bool((new_route[s:e] == int(routes[t["route"]])).all()):
            problems.append(f"doc {t['doc']}: doc_index / route_ids disagree")
        position = e
        if len(problems) > 20:
            break
    if not bool((new_doc[position:] == -1).all()) or int(real_lengths.sum()) != position:
        problems.append("padding / real_lengths disagree with the documents")
    excl = json.loads((out / "clean_exclude_rows.json").read_text(encoding="utf-8"))
    affected = set(excl["global_rows"])
    kept = [r for r in range(total // seq) if r not in affected]
    # X = what the sealed rows of `affected` hold minus what was re-packed: no X document may appear in a kept row
    repacked = {int(t["source_doc"]) for t in table}
    kept_docs = set(np.unique(sealed["doc_index"].reshape(-1, seq)[kept]).tolist()) if kept else set()
    affected_docs = set(np.unique(sealed["doc_index"].reshape(-1, seq)[sorted(affected)]).tolist())
    x_docs = affected_docs - repacked
    if kept_docs & x_docs:
        problems.append(f"{len(kept_docs & x_docs)} X documents appear in kept rows")
    x_tokens = sum(int(sealed_docs[d]["kept"]) for d in x_docs)
    whole_k = sum(int(sealed_docs[d]["kept"]) for d in kept_docs - repacked)
    conservation = x_tokens + position + whole_k == sum(int(d["kept"]) for d in sealed_docs.values())
    if not conservation:
        problems.append("token conservation fails")
    rows_by_x = {r for d in x_docs for r in doc_rows(sealed_docs[d], seq)}
    if rows_by_x != affected:
        problems.append(f"the excluded rows are not exactly the rows of the X documents ({len(rows_by_x ^ affected)} differ)")
    # the kept-row all mask covers exactly the positions owned by X / re-packed documents (plus carried A2 3(b) docs)
    allm = json.loads((out / "clean_all_mask.json").read_text(encoding="utf-8"))
    files = {s["sha256"]: s for s in shards}
    want = np.zeros(total, dtype=bool)
    for d in x_docs | repacked:
        dd = sealed_docs[d]
        lo, hi = MB.doc_positions(int(dd["start"]), int(dd["start"]) + int(dd["kept"]), total)
        want[lo:hi] = True
    kept_mask = np.zeros(total, dtype=bool)
    for r in kept:
        kept_mask[r * seq: (r + 1) * seq] = True
    want &= kept_mask
    got = np.zeros(total, dtype=bool)
    for sha, entry in allm["corpora"].items():
        if sha in files:
            off = files[sha]["row_offset"]
            for r, a, b in entry["spans"]:
                got[(off + r) * seq + a: (off + r) * seq + b] = True
    if (want & ~got).any():
        problems.append(f"the all mask misses {int((want & ~got).sum())} kept-row positions of X / re-packed documents")
    if (got & ~kept_mask).any():
        problems.append("the all mask names positions in excluded rows")
    result = {"rows": n_rows, "documents": len(table), "real_tokens": position, "affected_rows": len(affected), "kept_rows": len(kept),
              "x_docs": len(x_docs), "x_tokens": x_tokens, "conservation": conservation,
              "kept_row_x_positions_masked": int(want.sum()), "problems": problems[:20], "ok": not problems}
    return finish_verify(args, out, man, result, problems)


def finish_verify(args: argparse.Namespace, out: Path, man: dict[str, Any], result: dict[str, Any], problems: list[str]) -> int:
    receipt = {"schema": f"{SCHEMA}-verify", "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(__file__),
               "manifest_sha256": man.get("manifest_sha256"), "manifest_file_sha256": sha256_file(out / "manifest.json"),
               "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **result}
    path = Path(args.receipt) if args.receipt else out / "verify_receipt.json"
    rsha = write_json_atomic(path, receipt)
    for key, value in result.items():
        log(f"VERIFY {key} {json.dumps(value)}")
    log(f"VERIFY {'OK' if not problems else 'FAILED'}; receipt {path} sha256 {rsha}")
    return 0 if not problems else 1


# --------------------------------------------------------------------------
# state (read by the recache runner and by d2_plan)
# --------------------------------------------------------------------------


def cache_sealed(cache: Path) -> dict[str, Any]:
    manifest = cache / "manifest.json"
    if not manifest.is_file():
        return {"dir": str(cache), "sealed": False, "reason": "no manifest.json"}
    m = json.loads(manifest.read_text(encoding="utf-8"))
    ok = bool(m.get("all_finite")) and bool(m.get("all_structural_ok")) and bool((m.get("teacher") or {}).get("equal_before_and_after_every_session"))
    return {"dir": str(cache), "sealed": ok, "manifest_sha256": sha256_file(manifest), "tool": m.get("tool"),
            "kernel": ((m.get("cache_config") or {}).get("linear_attention_kernel") or {}).get("chunk_gated_delta_rule")}


def state(args: argparse.Namespace) -> int:
    out = Path(args.out)
    body: dict[str, Any] = {"out": str(out), "built": (out / "manifest.json").is_file()}
    if body["built"]:
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        body["manifest_sha256"] = man.get("manifest_sha256")
        rec = out / "verify_receipt.json"
        v = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else {}
        body["verified"] = bool(v.get("ok")) and v.get("manifest_sha256") == man.get("manifest_sha256") and v.get("manifest_file_sha256") == sha256_file(out / "manifest.json")
        caches = []
        for f in man["shards"]:
            caches.append({"file": f["file"], "corpus_sha256": f["sha256"], **cache_sealed(Path(args.cache_root) / f["sha256"])})
        body["caches"] = caches
        body["caches_sealed"] = sum(1 for c in caches if c["sealed"])
        body["files"] = len(caches)
        body["masks"] = man["masks"]
        body["ready"] = body["verified"] and body["caches_sealed"] == len(caches)
    else:
        body["ready"] = False
    print("D2CLEAN state " + json.dumps(body, sort_keys=True), flush=True)
    return 0 if body["ready"] else EXIT_WAITING


# --------------------------------------------------------------------------
# self-test and CLI
# --------------------------------------------------------------------------


def self_test() -> int:
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    seq = 8
    # stream of 4 rows x 8: d0 [0,5) d1 [5,13) X d2 [13,15) d3 [15,26) d4 [26,32)
    spec = [(0, 5, "a"), (5, 13, "x"), (13, 15, "b"), (15, 26, "c"), (26, 32, "d")]
    docs = [{"doc": i, "rid": r, "start": s, "end": e, "kept": e - s, "length": e - s} for i, (s, e, r) in enumerate(spec)]
    p = plan_repack(docs, 4, seq, {"x"})
    check("affected = rows of the X doc", p["affected_rows"] == [0, 1], str(p["affected_rows"]))
    check("kept rows", p["kept_rows"] == [2, 3])
    check("re-pack = non-X docs touching affected rows, in doc order", [d["rid"] for d in p["repack"]] == ["a", "b", "c"])
    check("boundary = re-packed doc that also lives in a kept row", [d["rid"] for d in p["boundary"]] == ["c"])
    check("kept whole", [d["rid"] for d in p["kept_whole"]] == ["d"])
    # c owns positions [14, 25): kept-row part = row 2 cols [0,8) + row 3 cols [0,1); x owns [4,12) -> rows 0/1 only
    check("kept-row all spans = c's positions in rows 2/3", p["kept_row_all_spans"] == [(2, 0, 8), (3, 0, 1)], str(p["kept_row_all_spans"]))
    check("pack rows rounds up", pack_rows([5, 2, 11], 8) == 3 and pack_rows([8], 8) == 1)
    check("eos span", eos_span(100) == (96, 99))
    try:
        covered_docs([(0, 0, 3)], docs, seq, 32, "t")
        check("a partial-document mask is refused", False)
    except Refusal:
        check("a partial-document mask is refused", True)
    check("a whole-document mask names the document", covered_docs([(0, 0, 4)], docs, seq, 32, "t") == {0})
    print(f"SELF-TEST {'PASS' if not failures else 'FAIL'} ({10 - len(failures)}/10)", flush=True)
    return 0 if not failures else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="command")
    b = sub.add_parser("build")
    b.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    b.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS))
    b.add_argument("--rows", default=str(DEFAULT_ROWS))
    b.add_argument("--masks-v1", default=str(DEFAULT_MASKS_V1))
    b.add_argument("--qwen-dir", default=DCB.DEFAULT_QWEN_DIR)
    b.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--no-screen", action="store_true", help="skip the post-pack review screen (tests only)")
    b.add_argument("--no-expect", action="store_true", help="build on inputs other than the 2026-10-15 ones (tests)")
    v = sub.add_parser("verify")
    v.add_argument("--out", default=str(DEFAULT_OUT))
    v.add_argument("--corpus-dir", default=None, help="the sealed corpus dir (default: the manifest's)")
    v.add_argument("--receipt", default=None)
    s = sub.add_parser("state")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.command:
        parser().print_help()
        return 2
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        return {"build": build, "verify": verify, "state": state}[args.command](args)
    except Refusal as refusal:
        log(f"REFUSED: {refusal.message}")
        return 2
    except MB.Refusal as refusal:
        log(f"REFUSED (d2_mask_build): {refusal.message}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
