#!/usr/bin/env python3
"""bfcl62_holdout.py -- the BFCL-62 held-out rid list: Codex's 62 BFCL v4 test tasks -> Claude's corpus rids/rows.

Card L-DATA of STAGE_D_RECONCILED_PLAN_2026-10-14 (sections 3.3(a), 3.4, 7.1: "BFCL-62 rid list for A2 3a"), now
Claude's lane under ruling cyril-claude-leads-all-lanes-2026-10-14.  Codex's combined held-out set
(proof_outputs/chimera_recovery_eval_combined_oct13_v1) holds 62 `bfcl-heldout:*` tasks.  They are a valid secondary
held-out for a D2-trained checkpoint only while every registry document behind them is out of D2 and D4 replay.

This tool derives that list from Codex's artifacts READ-ONLY, by a path independent of d2_mask_build's group join:

  path B (primary)  each task's own provenance: source_ids (registry rids) and source_row_id (normalized uid) ->
                    documents.jsonl, where the document's uid must equal source_row_id;
  path A (check)    task group_id -> normalized.jsonl recovery_group_id -> registry_origin.rid (the X04 /
                    d2_mask_build 1.0.0 logic).  A and B must give the same rid set.

Then it cross-checks: all 62 normalized rows are split=test; all docs sit in the BFCL v4 prose batch; the task ids
equal Codex's source pin (chimera_recovery_eval_bfcl_oct13_v3); no task group is in Codex's training groups; no task
uid is a row of Codex's native-v3 TRAIN shard (the BFCL-native 506 copy source); no rid is also a long / K3 / self-gen
source in allocation.json; the packed rows (4,096 positions) equal d2_mask_build's exclude_rows.json when given.
An exposure scan looks for each task's user request verbatim inside any other D2-sourced normalized record.

Writes ONE json (schema omni-holdout-bfcl62-rids-v1).  It refuses (exit 2) when a pinned input sha differs, when the
two derivation paths disagree, or when a hard cross-check fails; exposure findings are reported, not refused.

Usage:
  python bfcl62_holdout.py build --eval-tasks T --bfcl-tasks T3 --bfcl-training-groups G --documents D \
      --normalized N --allocation A --native-train-manifest M [--corpus-manifest CM] [--exclude-rows X] --out FILE
  python bfcl62_holdout.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TOOL_NAME = "bfcl62_holdout"
TOOL_VERSION = "1.0.0"  # 2026-10-14: first version (lane L-DATA under Claude, LANE guard workflow item 2)

SCHEMA = "omni-holdout-bfcl62-rids-v1"
SEQ = 4096
TASK_PREFIX = "bfcl-heldout:"
BFCL_DATASET = "manual_cleanup_batch_000166_bfcl_v4_function_calls_ready_append_current"

#: inputs as read on 2026-10-14 (X04 lane, d2_mask_build 1.0.0 receipt, server sha256sum)
PINS = {
    "eval_tasks": "3795f333998627bb3b2d6fd460c19228469b2286284dfaadf30c5e12d23713f8",
    "documents": "ce6ff058b175e7b0558917faaef737b7487b90e1854aa762571410fe43511fcf",
    "normalized": "168c4dc994db5858f0dcea3854c6983625f3c692876925c7c4b880b4f327c51e",
}
#: what the reconciled plan (X04, m) says the answer is
EXPECT = {"tasks": 62, "docs": 62, "rows": 69, "excluded_positions": 282624, "bfcl_docs": 761}


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
    """Same canonical form as d2_mask_build, so list shas compare across the two tools."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json_atomic(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


# --------------------------------------------------------------------------
# Pure helpers (tested on their own).
# --------------------------------------------------------------------------


def doc_rows(start: int, kept: int, seq: int | None = None) -> list[int]:
    """Packed rows holding the document's global tokens [start, start + kept)."""
    seq = seq or SEQ
    if kept <= 0:
        return []
    return list(range(start // seq, (start + kept - 1) // seq + 1))


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def user_request(messages: list[dict[str, Any]]) -> str:
    """The task's own request text: the BFCL 'User request:' block when present, else every pre-assistant message."""
    parts = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        content = message.get("content")
        parts.append(content if isinstance(content, str) else json.dumps(content, sort_keys=True))
    text = "\n".join(parts)
    match = re.search(r"User request:\n(.*?)\n\nAvailable tool schemas:", text, flags=re.S)
    return squash(match.group(1) if match else text)


def fold(text: str) -> str:
    """Case and punctuation folded: lowercase, every non-alphanumeric run one space."""
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def expected_calls(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    adaptation = (record.get("provenance") or {}).get("bfcl_schema_target_adaptation") or {}
    return (adaptation.get("original_acceptable_targets") or {}).get("expected_tool_calls")


def same_answer(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> bool:
    """Same function names in order, same argument keys, and every argument's accepted alternatives overlap."""
    if not left or not right or len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if list(a) != list(b):
            return False
        (name, args_a), = a.items()
        args_b = b[name]
        if set(args_a) != set(args_b):
            return False
        for key in args_a:
            if not {json.dumps(v, sort_keys=True) for v in args_a[key]} & {json.dumps(v, sort_keys=True) for v in args_b[key]}:
                return False
    return True


def record_text(record: dict[str, Any]) -> str:
    parts = []
    for message in record.get("messages") or []:
        content = message.get("content")
        parts.append(content if isinstance(content, str) else json.dumps(content, sort_keys=True))
    return squash("\n".join(parts))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    inputs = {name: Path(getattr(args, name)) for name in
              ("eval_tasks", "bfcl_tasks", "bfcl_training_groups", "documents", "normalized", "allocation",
               "native_train_manifest")}
    for name in ("corpus_manifest", "exclude_rows"):
        if getattr(args, name):
            inputs[name] = Path(getattr(args, name))
    shas = {name: sha256_file(path) for name, path in inputs.items()}
    for name, sha in shas.items():
        log(f"INPUT {name} {sha} {inputs[name]}")
    for name, pin in PINS.items():
        if not args.no_expect and shas[name] != pin:
            raise Refusal(f"{name} {shas[name][:16]} is not the pinned {pin[:16]} (give --no-expect to build on new inputs)")

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool, detail: Any = None, hard: bool = True) -> None:
        checks[name] = {"ok": bool(ok), "hard": hard, "detail": detail}
        log(f"CHECK {'PASS' if ok else 'FAIL'} {name}{' :: ' + json.dumps(detail) if detail is not None else ''}")

    tasks = [t for t in read_jsonl(inputs["eval_tasks"]) if str(t.get("id", "")).startswith(TASK_PREFIX)]
    pin_tasks = [t for t in read_jsonl(inputs["bfcl_tasks"]) if str(t.get("id", "")).startswith(TASK_PREFIX)]
    check("task ids equal Codex source pin chimera_recovery_eval_bfcl_oct13_v3",
          sorted(t["id"] for t in tasks) == sorted(t["id"] for t in pin_tasks),
          {"combined": len(tasks), "source_pin": len(pin_tasks)})
    check("distinct task groups", len({t["group_id"] for t in tasks}) == len(tasks), len({t["group_id"] for t in tasks}))

    docs = list(read_jsonl(inputs["documents"]))
    by_rid = {d["rid"]: d for d in docs}
    by_uid = defaultdict(list)
    for d in docs:
        by_uid[d["uid"]].append(d["rid"])
    span_bad = [d["rid"] for d in docs if int(d["end"]) != int(d["start"]) + int(d["kept"])
                or (int(d["kept"]) > 0 and (int(d["row_first"]), int(d["row_last"]))
                    != (int(d["start"]) // SEQ, (int(d["start"]) + int(d["kept"]) - 1) // SEQ))]
    check("documents.jsonl end == start + kept and row_first/row_last match span arithmetic", not span_bad,
          {"documents": len(docs), "disagreeing": len(span_bad), "first": span_bad[:3]})

    # path B: the task's own provenance
    path_b: dict[str, str] = {}
    problems_b = []
    for t in tasks:
        p = t.get("provenance") or {}
        rids = p.get("source_ids") or []
        uid = p.get("source_row_id")
        if len(rids) != 1 or rids[0] not in by_rid or by_rid[rids[0]]["uid"] != uid:
            problems_b.append({"task": t["id"], "source_ids": rids, "source_row_id": uid,
                               "uid_rids": by_uid.get(uid, [])})
            continue
        path_b[t["id"]] = rids[0]
    check("path B: every task has exactly one source rid in documents.jsonl whose uid == source_row_id",
          not problems_b, {"resolved": len(path_b), "problems": problems_b[:5]})

    # path A: group join through normalized.jsonl (the X04 / d2_mask_build logic); keep the 62 records for later checks
    task_groups = {t["group_id"]: t["id"] for t in tasks}
    wanted_uids = {(t.get("provenance") or {}).get("source_row_id") for t in tasks}
    group_rids: dict[str, list[str]] = defaultdict(list)
    held_records: dict[str, dict[str, Any]] = {}
    rid_of_uid: dict[str, str] = {}
    for r in read_jsonl(inputs["normalized"]):
        prov = r.get("provenance") or {}
        rid = (prov.get("registry_origin") or {}).get("rid")
        if rid:
            rid_of_uid[r["id"]] = rid
        if prov.get("recovery_group_id") in task_groups and rid:
            group_rids[prov["recovery_group_id"]].append(rid)
        if r.get("id") in wanted_uids:
            held_records[r["id"]] = r
    path_a = {task_groups[g]: sorted(set(rids) & set(by_rid)) for g, rids in group_rids.items()}
    multi = {k: v for k, v in path_a.items() if len(v) != 1}
    rids_a = sorted({rid for v in path_a.values() for rid in v})
    rids_b = sorted(set(path_b.values()))
    check("path A == path B (group join vs task provenance)", rids_a == rids_b and not multi,
          {"path_a": len(rids_a), "path_b": len(rids_b), "groups_not_one_doc": multi})

    rids = rids_b
    held_docs = [by_rid[r] for r in rids]
    check("every source document is in the BFCL v4 prose batch",
          all(d["dataset_id"] == BFCL_DATASET for d in held_docs),
          sorted({d["dataset_id"] for d in held_docs}))
    splits = sorted({held_records[u]["split"] for u in held_records})
    check("all 62 normalized source records are split=test and train_ready",
          len(held_records) == len(tasks) and splits == ["test"]
          and all((held_records[u].get("quality") or {}).get("train_ready") for u in held_records),
          {"records": len(held_records), "splits": splits})

    training_groups = set(json.loads(inputs["bfcl_training_groups"].read_text(encoding="utf-8")))
    overlap = sorted(set(task_groups) & training_groups)
    check("no task group in Codex's training groups (bfcl_oct13_v3/training_group_ids.json)", not overlap,
          {"training_groups": len(training_groups), "overlap": overlap[:3]})

    native = json.loads(inputs["native_train_manifest"].read_text(encoding="utf-8"))
    native_rows = {rec.get("row_id") for rec in native.get("row_receipts") or []}
    leaked = sorted(native_rows & wanted_uids)
    check("no task uid among Codex native-v3 TRAIN rows (source of the BFCL-native 506 copy)", not leaked,
          {"native_train_sha256": native.get("sha256"), "native_train_rows": len(native_rows), "overlap": leaked[:3]})

    allocation = json.loads(inputs["allocation"].read_text(encoding="utf-8"))
    other_sets = {"long": {x for d in allocation["long"]["docs"] for x in d["rows"]},
                  "k3": {x for row in allocation["k3"]["rows"] for x in row},
                  "selfgen": set(allocation["selfgen"]["prompts"])}
    text_set = set(allocation["text"]["docs"])
    in_other = {name: sorted(set(rids) & members) for name, members in other_sets.items()}
    check("no BFCL-62 rid is also a long / K3 / self-gen source (allocation.json)",
          not any(in_other.values()), {k: len(v) for k, v in in_other.items()})
    check("every BFCL-62 rid is a registry text-set doc (allocation.json text.docs)", set(rids) <= text_set,
          {"missing": sorted(set(rids) - text_set)[:3]})

    rows = sorted({row for d in held_docs for row in doc_rows(int(d["start"]), int(d["kept"]))})
    total_rows = max(int(d["row_last"]) for d in docs) + 1
    row_set = set(rows)
    row_docs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        for row in doc_rows(int(d["start"]), int(d["kept"])):
            if row in row_set:
                row_docs[row].append(d)
    held = set(rids)
    collateral = [d for row in rows for d in row_docs[row] if d["rid"] not in held]
    collateral_rids = sorted({d["rid"] for d in collateral})
    counts = {"tasks": len(tasks), "routes": dict(sorted(_count(t["route"] for t in tasks).items())),
              "docs": len(held_docs), "doc_tokens": sum(int(d["kept"]) for d in held_docs),
              "rows": len(rows), "excluded_positions": len(rows) * SEQ, "corpus_rows": total_rows,
              "positions_left": (total_rows - len(rows)) * SEQ,
              "bfcl_docs": sum(1 for d in docs if d["dataset_id"] == BFCL_DATASET),
              "bfcl_docs_other_than_62": sum(1 for d in docs if d["dataset_id"] == BFCL_DATASET and d["rid"] not in held),
              "collateral_docs_in_excluded_rows": len(collateral_rids),
              "collateral_tokens_in_excluded_rows": sum(int(d["kept"]) for d in {d["rid"]: d for d in collateral}.values()),
              "docs_starting_at_row_column_0": sum(1 for d in held_docs if int(d["start"]) % SEQ == 0)}
    for key, value in counts.items():
        log(f"COUNT {key} {json.dumps(value)}")
    mismatch = {k: (counts[k], v) for k, v in EXPECT.items() if counts[k] != v}
    check("counts equal the reconciled plan's X04 reading (62 docs, 69 rows, 282,624 positions, 761 BFCL docs)",
          not mismatch, mismatch or None)

    if "exclude_rows" in inputs:
        x = json.loads(inputs["exclude_rows"].read_text(encoding="utf-8"))
        same = (x.get("global_rows") == rows and sorted(x.get("bfcl62_rids") or []) == rids
                and x.get("global_rows_sha256") == canonical_sha256(rows) and x.get("bfcl62_rids_sha256") == canonical_sha256(rids))
        check("equal to d2_mask_build exclude_rows.json (rows, rids and their canonical shas)", same,
              {"tool": x.get("tool"), "tool_version": x.get("tool_version"), "rows": len(x.get("global_rows") or []),
               "rids": len(x.get("bfcl62_rids") or [])})

    per_file = None
    if "corpus_manifest" in inputs:
        manifest = json.loads(inputs["corpus_manifest"].read_text(encoding="utf-8"))
        per_file, offset = {}, 0
        for shard in sorted(manifest["shards"], key=lambda e: int(e["rows"][0])):
            lo, hi = int(shard["rows"][0]), int(shard["rows"][1])
            if lo != offset:
                raise Refusal(f"corpus manifest shards are not contiguous at row {offset}")
            local = [r - lo for r in rows if lo <= r < hi]
            if local:
                per_file[shard["sha256"]] = {"file": Path(shard["path"]).name, "row_offset": lo, "rows_in_file": hi - lo,
                                             "rows": local}
            offset = hi
        check("corpus manifest rows == documents.jsonl rows", offset == total_rows, {"manifest": offset, "documents": total_rows})

    # exposure: each held-out request, verbatim or case/punctuation-folded, inside any OTHER record that feeds training:
    # registry text docs, long / K3 / self-gen sources (allocation.json) and Codex's native-v3 TRAIN shard (the source
    # of the BFCL-native 506 copy).  BFCL v4 asks the same question in several categories (simple_python vs multiple),
    # and Codex's recovery groups put such twins in different groups, so a group-disjoint split can still leak them.
    requests = {u: user_request(held_records[u].get("messages") or []) for u in held_records}
    folded = {u: fold(r) for u, r in requests.items()}
    membership = (("text", lambda uid, rid: rid in by_rid), ("long", lambda uid, rid: rid in other_sets["long"]),
                  ("k3", lambda uid, rid: rid in other_sets["k3"]), ("selfgen", lambda uid, rid: rid in other_sets["selfgen"]),
                  ("native_train", lambda uid, rid: uid in native_rows))
    hits = []
    for r in read_jsonl(inputs["normalized"]):
        uid = r.get("id")
        rid = rid_of_uid.get(uid)
        if uid in held_records:
            continue
        sources = [name for name, member in membership if member(uid, rid)]
        if not sources:
            continue
        text, ftext = record_text(r), None
        for held_uid, request in requests.items():
            level = "verbatim" if request and request in text else None
            if level is None:
                ftext = fold(text) if ftext is None else ftext
                level = "folded" if folded[held_uid] and folded[held_uid] in ftext else None
            if level is None:
                continue
            twin_rows = doc_rows(int(by_rid[rid]["start"]), int(by_rid[rid]["kept"])) if rid in by_rid else []
            hits.append({"task": TASK_PREFIX + held_uid, "match": level, "uid": uid, "rid": rid, "sources": sources,
                         "split": r.get("split"), "train_ready": (r.get("quality") or {}).get("train_ready"),
                         "task_type": ((r.get("provenance") or {}).get("registry_origin") or {}).get("task_type"),
                         "same_answer": same_answer(expected_calls(held_records[held_uid]), expected_calls(r)),
                         "text_rows": twin_rows,
                         "covered_by_excluded_rows": sources == ["text"] and set(twin_rows) <= row_set})
    exposed = sorted({h["task"] for h in hits if not h["covered_by_excluded_rows"]})
    clean = sorted(set(t["id"] for t in tasks) - set(exposed))
    twin_rids = {name: sorted({h["rid"] if name != "native_train" else h["uid"] for h in hits if name in h["sources"]})
                 for name, _ in membership}
    extra_rows = sorted({row for h in hits if "text" in h["sources"] for row in h["text_rows"]} - row_set)
    twin_exposure = {"tasks_exposed": exposed, "clean_task_ids": clean, "clean_task_ids_sha256": canonical_sha256(clean),
                     "twin_ids_by_source": twin_rids, "extra_text_rows_if_text_twins_excluded": extra_rows,
                     "extra_positions_if_text_twins_excluded": len(extra_rows) * SEQ, "hits": hits}
    check("no held-out request (verbatim or folded) in another training-fed record outside the excluded rows",
          not exposed, {"hits": len(hits), "same_answer_hits": sum(1 for h in hits if h["same_answer"]),
                        "tasks_exposed": len(exposed), "clean_tasks": len(clean),
                        "twins_by_source": {k: len(v) for k, v in twin_rids.items()}}, hard=False)

    hard_fail = [k for k, v in checks.items() if v["hard"] and not v["ok"]]
    if hard_fail:
        raise Refusal(f"hard cross-checks failed: {hard_fail}")

    task_rows = []
    for t in sorted(tasks, key=lambda t: path_b[t["id"]]):
        d = by_rid[path_b[t["id"]]]
        task_rows.append({"task_id": t["id"], "route": t["route"], "split": t["split"], "group_id": t["group_id"],
                          "source_row_id": t["provenance"]["source_row_id"], "rid": d["rid"], "doc": int(d["doc"]),
                          "record_id": d["record_id"], "start": int(d["start"]), "kept": int(d["kept"]),
                          "rows": doc_rows(int(d["start"]), int(d["kept"]))})
    out = {
        "schema": SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": ("Registry rids and packed rows behind Codex's 62 BFCL v4 held-out tasks (combined_oct13_v1). "
                    "Excluding these rows from D2 and D4 replay keeps BFCL-62 a valid secondary held-out "
                    "(reconciled plan 3.3(a), prereg A2 3(a))."),
        "seq_len": SEQ,
        "rids": rids, "rids_sha256": canonical_sha256(rids),
        "uids": sorted(wanted_uids), "uids_sha256": canonical_sha256(sorted(wanted_uids)),
        "task_ids": sorted(t["id"] for t in tasks), "task_ids_sha256": canonical_sha256(sorted(t["id"] for t in tasks)),
        "global_rows": rows, "global_rows_sha256": canonical_sha256(rows),
        "corpora": per_file,
        "tasks": task_rows,
        "collateral_rids_in_excluded_rows": collateral_rids,
        "counts": counts,
        "cross_checks": checks,
        "twin_exposure": twin_exposure,
        "inputs": {name: {"path": str(path), "sha256": shas[name]} for name, path in inputs.items()},
        "provenance": {
            "codex_branch": "codex/work/codex/chimera-training-recovery-oct13-eae1b72e2c3f",
            "codex_head": "81f549884203193efdff6a842054937410ab0ddc",
            "codex_task_builder": "experiments/chimera_recovery/evaluation.py (sha256 = combined manifest grader_sha256)",
            "codex_artifacts": "read-only; nothing in Codex's tree was written",
            "derivation": "path B (task provenance source_ids/source_row_id) primary; path A (group join, X04/d2_mask_build) must agree",
            "ruling": "cyril-claude-leads-all-lanes-2026-10-14 (Claude leads L-DATA)",
            "plan": "STAGE_D_RECONCILED_PLAN_2026-10-14.md 3.3(a), 3.4, 7.1",
            "argv": sys.argv,
        },
    }
    sha = write_json_atomic(Path(args.out), out)
    log(f"WROTE {args.out} sha256 {sha}")
    log(f"rids_sha256 {out['rids_sha256']} global_rows_sha256 {out['global_rows_sha256']}")
    return 0


def _count(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


# --------------------------------------------------------------------------
# self-test and CLI
# --------------------------------------------------------------------------


def self_test() -> int:
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(name)

    check("doc inside one row", doc_rows(10, 5, 16) == [0])
    check("doc ending exactly at a row edge stays in that row", doc_rows(11, 5, 16) == [0])
    check("doc crossing a row edge", doc_rows(14, 5, 16) == [0, 1])
    check("empty doc has no rows", doc_rows(14, 0, 16) == [])
    msgs = [{"role": "user", "content": "x\nUser request:\nTurn 1: user:  find\n it\n\nAvailable tool schemas:\n[]"},
            {"role": "assistant", "content": "ignored"}]
    check("user request block", user_request(msgs) == "Turn 1: user: find it", user_request(msgs))
    check("fold", fold("Convert 150 Euros, to CAD!") == "convert 150 euros to cad")
    check("same answer despite alternative order",
          same_answer([{"f": {"a": [1, ""], "b": ["x"]}}], [{"f": {"a": ["", 1], "b": ["x", "y"]}}]))
    check("different argument value is not the same answer", not same_answer([{"f": {"a": [1]}}], [{"f": {"a": [2]}}]))
    print(f"SELF-TEST {'PASS' if not failures else 'FAIL'} ({8 - len(failures)}/8)", flush=True)
    return 0 if not failures else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="command")
    b = sub.add_parser("build")
    b.add_argument("--eval-tasks", required=True, help="Codex chimera_recovery_eval_combined_oct13_v1/tasks.jsonl")
    b.add_argument("--bfcl-tasks", required=True, help="Codex chimera_recovery_eval_bfcl_oct13_v3/tasks.jsonl (source pin)")
    b.add_argument("--bfcl-training-groups", required=True, help="Codex chimera_recovery_eval_bfcl_oct13_v3/training_group_ids.json")
    b.add_argument("--documents", required=True, help="distill/corpus/text/documents.jsonl")
    b.add_argument("--normalized", required=True, help="Codex chimera_corpus_recovery_oct13/normalized.jsonl")
    b.add_argument("--allocation", required=True, help="distill/corpus/allocation.json")
    b.add_argument("--native-train-manifest", required=True, help="Codex native-v3-manifests/train/manifest.json")
    b.add_argument("--corpus-manifest", help="distill/corpus/text/manifest.json (adds per-file local rows)")
    b.add_argument("--exclude-rows", help="d2_mask_build exclude_rows.json to cross-check")
    b.add_argument("--out", required=True)
    b.add_argument("--no-expect", action="store_true", help="build on inputs other than the pinned shas")
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
    except Refusal as refusal:
        log(f"REFUSED: {refusal.message}")
        return 2
    parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
