"""ingest.fold_receipts -- fold / K3-pass / import / chart receipts -> run (one per receipt file).

Walks $OC, $L and /home/cereal/model-surgery-lab for '*receipt*.json' under fold-work paths (k3, fold,
composite c10/c17/s3/s7, qwen-fold, import, media folds, LFM2 work) and records each as a run:
  kind     cache (k3_teacher_cache schemas) | probe (parity) | build (import) | fold (everything else)
  status   from verdict/status/ok/passed fields (sealed/pass/ok/completed/done -> pass, fail*, refus*);
           'unknown' when the receipt states none
  times    started/finished/created/generated fields; when a receipt has none, ended_at = file mtime
           (config.time_source says which)
  config   the receipt's small top-level fields (big blocks reduced to sizes) + argv/tool when present
Receipts owned by other ingesters (teacher_cache/, distill/fullkd, media_m1, m2, d1, d2) are skipped, and so
are content-addressed cache entries (neural_forge_phase_cache: ~26k per-entry receipt.json files) and
copied trees (deployments/, backup-pre-*, _superseded).  Runs this source wrote earlier whose receipt is
no longer found (or now excluded) are deleted when nothing else references them, so the table tracks
the current scope; the receipts themselves are never touched.
"""
from __future__ import annotations

import os
import re

from . import common as C
from .common import db
from .teacher_caches import summarise

SOURCE = "fold_receipts"
WANT = re.compile(r"(k3|fold|/composite/|c10|c17|/s3-|/s7|qwen-fold|/import|ace|hy3d|qimg|/h3|lfm2|LFM2|chart|union)", re.I)
EXCLUDE = re.compile(r"/teacher_cache/|/distill/(fullkd|media_m1|m2|d1|d2)/|pytest|scratch_review_tests|/\.smoke|"
                     r"/__pycache__/|/site-packages/|/venv/|/\.git/|/llama\.cpp|"
                     # content-addressed cache entries and copied trees are not runs:
                     r"neural_forge_phase_cache|/deployments/|/backup-pre-|/_superseded")
NAME = re.compile(r"receipt.*\.json$", re.I)
SKIP_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", "site-packages", "teacher_cache", "llama.cpp"}
TIME_START = ("started_utc", "start_utc", "started_at", "started", "launched_at_utc")
TIME_END = ("finished_utc", "ended_utc", "end_utc", "finished_at", "completed_utc", "sealed_utc", "sealed_at",
            "created_utc", "created_at_utc", "generated_utc", "utc", "written_utc")


def find_receipts(roots: list[tuple[str, int]]) -> list[str]:
    found = []
    for root, maxdepth in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for d, subdirs, files in os.walk(root):
            subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS]
            if d.count("/") - base >= maxdepth or len(files) > 4000:
                subdirs[:] = []
            for n in files:
                if NAME.search(n):
                    p = os.path.join(d, n)
                    if WANT.search(p) and not EXCLUDE.search(p):
                        found.append(p)
    return sorted(found)


def status_of(doc: dict) -> str:
    for key in ("verdict", "status", "state", "result", "outcome"):
        v = doc.get(key)
        if isinstance(v, str) and v:
            t = v.lower()
            if "refus" in t:
                return "refused"
            if "fail" in t or "error" in t or "abort" in t:
                return "fail"
            if any(w in t for w in ("seal", "pass", "ok", "complete", "done", "success", "succeeded", "finished",
                                    "verified", "written")):
                return "pass"
    for key in ("ok", "passed", "pass", "all_ok", "all_finite"):
        if isinstance(doc.get(key), bool):
            return "pass" if doc[key] else "fail"
    return "unknown"


def kind_of(doc: dict, path: str) -> str:
    schema = str(doc.get("schema") or "")
    if "teacher_cache" in schema:
        return "probe" if "parity" in schema else "cache"
    if "parity" in schema or "parity" in os.path.basename(path):
        return "probe"
    if "import" in schema or "/import" in path:
        return "build"
    return "fold"


def project_of(path: str) -> str:
    if re.search(r"k3", path, re.I):
        return "k3"
    if path.startswith(C.NF_LAB) or re.search(r"lfm2|/research/fold/|qwen-fold", path, re.I):
        return "neural-forge"
    if re.search(r"ace|hy3d|qimg|/h3|/s7|media", path, re.I):
        return "media"
    return "chimera-omni"


def prune(conn, batch: C.Batch, keep: set[str]) -> None:
    """Delete this source's runs whose receipt is out of scope now (unreferenced rows only)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, run_key FROM run WHERE run_key LIKE 'receipt:%%'")
        stale = [rid for rid, key in cur.fetchall() if key not in keep]
        if not stale:
            return
        cur.execute("DELETE FROM run r WHERE r.id = ANY(%s) "
                    "AND NOT EXISTS (SELECT 1 FROM run c WHERE c.parent_run_id = r.id) "
                    "AND NOT EXISTS (SELECT 1 FROM experiment e WHERE e.run_id = r.id) "
                    "AND NOT EXISTS (SELECT 1 FROM document d WHERE d.run_id = r.id) "
                    "AND NOT EXISTS (SELECT 1 FROM eval_run er WHERE er.baseline_run_id = r.id)", (stale,))
        batch.skip(f"pruned {cur.rowcount} out-of-scope receipt runs ({len(stale) - cur.rowcount} kept: referenced)")


def run(conn, batch: C.Batch, args) -> None:
    host_id = C.host_id(conn)
    keep: set[str] = set()
    for path in find_receipts([(C.OC, 9), (C.LX, 8), (C.NF_LAB, 7)]):
        batch.saw()
        doc = C.load_json(path, 16 << 20)
        if not isinstance(doc, dict):
            batch.skip(f"not a JSON object: {path}")
            continue
        prov = batch.prov(path)
        started = next((C.parse_ts(doc.get(k)) for k in TIME_START if C.parse_ts(doc.get(k))), None)
        ended = next((C.parse_ts(doc.get(k)) for k in TIME_END if C.parse_ts(doc.get(k))), None)
        time_source = "receipt" if (started or ended) else "file-mtime"
        if not (started or ended):
            ended = C.parse_ts(C.iso_mtime(path))
        tool = doc.get("tool")
        tname, tver, tsha, tpath = (None, None, None, None)
        if isinstance(tool, dict):
            tname, tver, tsha, tpath = tool.get("name"), tool.get("version"), tool.get("sha256"), tool.get("path")
        elif isinstance(tool, str):
            tname, tver, tsha = tool, doc.get("tool_version") or doc.get("version"), doc.get("tool_sha256")
        tool_vid = C.tool_version_for(conn, tname, str(tver) if tver else None,
                                      tsha if isinstance(tsha, str) else None, tpath) if tname else None
        config = summarise(doc)
        config["time_source"] = time_source
        kind = kind_of(doc, path)
        project = project_of(path)
        rel = path.replace(C.OC + "/", "$OC/").replace(C.LX + "/", "$L/").replace(C.NF_LAB + "/", "$NF/")
        keep.add(f"receipt:{rel}")
        env_id = db.ensure_env(conn, torch=doc.get("torch"), python=doc.get("python") if isinstance(doc.get("python"), str) else None) \
            if isinstance(doc.get("torch"), str) else None
        rid = db.upsert_run(conn, f"receipt:{rel}", project, kind, rel, status_of(doc), tool_version_id=tool_vid,
                            host_id=host_id, env_id=env_id, started_at=started, ended_at=ended,
                            wall_seconds=next((float(doc[k]) for k in ("seconds", "wall_seconds", "elapsed_s", "wall_s")
                                               if isinstance(doc.get(k), (int, float)) and not isinstance(doc.get(k), bool)),
                                              C.seconds_between(started, ended)),
                            config=config, receipt_path=path, receipt_sha256=prov[1], run_dir=os.path.dirname(path),
                            notes=None if time_source == "receipt" else "no timestamp in the receipt; ended_at = file mtime",
                            prov=prov, batch_id=batch.id)
        if project == "k3":
            C.set_run_lane(conn, rid, "L-K3")
        elif project == "neural-forge":
            C.set_run_lane(conn, rid, "L-FOLD")
        batch.wrote()
    prune(conn, batch, keep)
