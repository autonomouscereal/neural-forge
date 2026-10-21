"""ingest.documents -- the document index (inventory/<date>/docs.json) -> document.

One row per unique document content (sha256 is the natural key); identical copies elsewhere go into
other_paths.  Columns map 1:1 from the inventory except:
    location 'local:*' -> host 'workstation', 'server:*' -> host 'ai'
    date_source 'content' -> 'text', 'dirname' -> 'folder' (filename, mtime unchanged)
    status 'superseded-or-backup' -> 'backup' when the path is a backup/copy, else 'superseded'
    top30_rank -> rank (the top-30 reason is appended to the summary)
summary is the document's first sentence (inventory 'auto-excerpt'), type/arm are keyword-classified:
a first pass, not a reading.  src_path/src_sha256 point at the inventory snapshot.  Re-runnable.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "documents"

DATE_SOURCE = {"content": "text", "dirname": "folder", "filename": "filename", "mtime": "mtime"}
BACKUP_RE = re.compile(r"(/backups?/|\.bak|backup|[-.]before-|[-.]pre-|/orig/|local_copy|\.staged$|\.r\d+-[0-9a-f]{8})", re.I)


def host_of(location: str) -> str:
    if location.startswith("local:"):
        return "workstation"
    if location.startswith("server:"):
        return "ai"
    raise db.LabError(f"unknown document location {location!r}")


def status_of(doc: dict) -> str:
    s = doc.get("status")
    if s in ("current", "scratch", "superseded", "backup"):
        return s
    if s == "superseded-or-backup":
        return "backup" if BACKUP_RE.search(doc["path"]) else "superseded"
    raise db.LabError(f"unknown document status {s!r} for {doc.get('path')}")


def run(conn, batch: C.Batch, args) -> None:
    inv = K.inventory_dir(args) / "docs.json"
    prov = batch.prov(inv, host=db.this_host())
    doc = json.loads(Path(inv).read_text(encoding="utf-8"))
    docs = doc.get("docs") if isinstance(doc, dict) else None
    if not isinstance(docs, list):
        raise db.LabError(f"{inv}: expected an object with a 'docs' list")
    tally = K.Tally(batch)
    for d in docs:
        batch.saw()
        if not re.fullmatch(r"[0-9a-f]{64}", d.get("sha256") or ""):
            batch.skip(f"no sha256 for {d.get('path')}")
            continue
        summary = d.get("summary") or None
        if d.get("top30_why"):
            summary = (summary + " | " if summary else "") + "TOP-30: " + d["top30_why"]
        doc_date = date.fromisoformat(d["date"]) if d.get("date") else None
        _id, state = K.upsert(conn, "document", ("sha256",), {
            "sha256": d["sha256"], "path": d["path"], "other_paths": list(d.get("copies") or []),
            "host": host_of(d["location"]), "title": d.get("title") or None, "doc_date": doc_date,
            "date_source": DATE_SOURCE.get(d.get("date_source")) if doc_date else None,
            "doc_type": d.get("type"), "arm": d.get("arm"), "status": status_of(d), "summary": summary,
            "size_bytes": d.get("size"), "rank": d.get("top30_rank"),
            "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id})
        tally.add("document", state)
    K.finish(batch, tally)


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
