"""ingest.tool_pins -- $OC/distill/tools/TOOL_PINS.json (omni-campaign.tool-pins.v1) -> tool, tool_version.

Each pins['tools'][<file>] entry is the pinned (current) version; its 'superseded' list holds older
versions (superseded=true).  The tool name is the entry's 'name' (file stem).  Paths are stored
absolute (pins 'root' + relative path).
"""
from __future__ import annotations

import posixpath

from . import common as C
from .common import db

SOURCE = "tool_pins"


def pins_path() -> str:
    return C.OC + "/distill/tools/TOOL_PINS.json"


def _upsert(conn, name: str, entry: dict, root: str, pinned: bool, superseded: bool, prov, batch_id) -> int:
    rel = entry.get("path") or ""
    path = rel if rel.startswith("/") else posixpath.join(root, rel) if rel else None
    detail = {k: entry.get(k) for k in ("bytes", "mtime_utc", "version_string", "version_source", "kind") if k in entry}
    with conn.cursor() as cur:
        cur.execute("INSERT INTO tool (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                    "RETURNING id", (name,))
        tool_id = cur.fetchone()[0]
        cur.execute("INSERT INTO tool_version (tool_id, version, sha256, path, pinned, superseded, pinned_at, detail, "
                    "src_path, src_sha256, src_host, ingest_batch_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                    "ON CONFLICT (tool_id, sha256) DO UPDATE SET version = EXCLUDED.version, path = EXCLUDED.path, "
                    "pinned = EXCLUDED.pinned, superseded = EXCLUDED.superseded, pinned_at = EXCLUDED.pinned_at, "
                    "detail = EXCLUDED.detail, src_path = EXCLUDED.src_path, src_sha256 = EXCLUDED.src_sha256, "
                    "src_host = EXCLUDED.src_host, ingest_batch_id = EXCLUDED.ingest_batch_id RETURNING id",
                    (tool_id, str(entry.get("version") or "unknown"), entry["sha256"], path, pinned, superseded,
                     C.parse_ts(entry.get("mtime_utc")) if pinned else None, db.canonical_json(detail),
                     prov[0], prov[1], prov[2], batch_id))
        return cur.fetchone()[0]


def run(conn, batch: C.Batch, args) -> None:
    path = pins_path()
    doc = C.load_json(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("tools"), dict):
        raise db.LabError(f"{path}: not an omni-campaign.tool-pins.v1 document")
    prov = batch.prov(path)
    root = doc.get("root") or (C.OC + "/distill")
    with conn.cursor() as cur:   # a re-ingest re-derives the pin flags from this file only
        cur.execute("UPDATE tool_version SET pinned = false WHERE src_path = %s", (prov[0],))
    for fname, entry in doc["tools"].items():
        if not isinstance(entry, dict) or not entry.get("sha256"):
            batch.skip(f"no sha256: {fname}")
            continue
        name = entry.get("name") or fname.rsplit(".", 1)[0]
        for old in entry.get("superseded") or []:
            if isinstance(old, dict) and old.get("sha256") and old["sha256"] != entry["sha256"]:
                batch.saw()
                _upsert(conn, name, old, root, False, True, prov, batch.id)
                batch.wrote()
        batch.saw()
        _upsert(conn, name, entry, root, True, False, prov, batch.id)
        batch.wrote()
