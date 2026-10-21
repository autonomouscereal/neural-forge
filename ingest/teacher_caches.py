"""ingest.teacher_caches -- $L/teacher_cache/** -> dataset, dataset_version, and one cache run per pass.

Each manifest.json / pass_receipt.partial.json / plan.json under teacher_cache/<kind>/<name>/ is one
dataset_version of dataset 'teacher_cache/<kind>/<name>' (version = the path below <name>, e.g. the
content-address '<sha>/<sha>' or 'pass-0-of-4').  A manifest with a tool and timestamps also becomes
a run (kind 'cache') that produced that version; a partial pass receipt is a run with status 'unknown'
(no seal).  Only the small top-level fields of a manifest are stored (per-layer / per-shard blocks
are summarised by counts).  Shards are never read.
"""
from __future__ import annotations

import os

from . import common as C
from .common import db

SOURCE = "teacher_caches"
DOC_NAMES = ("manifest.json", "pass_receipt.partial.json", "plan.json")
PROJECT_BY_KIND = {"k3": "k3", "qtext": "corpus", "qtext_vl": "corpus"}
SMALL = 4096


def cache_root() -> str:
    return C.LX + "/teacher_cache"


def summarise(doc: dict) -> dict:
    """Top-level scalars + small objects verbatim; big blocks reduced to their size."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v if not isinstance(v, str) or len(v) <= SMALL else v[:SMALL]
        else:
            text = db.canonical_json(v)
            out[k] = v if len(text) <= SMALL else {"_omitted_bytes": len(text),
                                                   "_len": len(v) if isinstance(v, (list, dict)) else None}
    return out


def tool_of(doc: dict) -> tuple[str | None, str | None, str | None, str | None]:
    t = doc.get("tool")
    if isinstance(t, dict):
        return t.get("name"), t.get("version"), t.get("sha256"), t.get("path")
    if isinstance(t, str):
        return t, doc.get("tool_version"), doc.get("tool_sha256"), None
    return None, None, None, None


def _dir_bytes(d: str) -> int:
    total = 0
    try:
        for e in os.scandir(d):
            if e.is_file(follow_symlinks=False):
                total += e.stat(follow_symlinks=False).st_size
    except OSError:
        pass
    return total


def find_docs(root: str, maxdepth: int = 6) -> list[str]:
    found = []
    base_depth = root.rstrip("/").count("/")
    for d, subdirs, files in os.walk(root):
        if d.count("/") - base_depth >= maxdepth:
            subdirs[:] = []
        for n in DOC_NAMES:
            if n in files and not (n == "pass_receipt.partial.json" and "manifest.json" in files):
                found.append(os.path.join(d, n))   # a sealed manifest supersedes the partial receipt beside it
        if len(files) > 4000:      # shard leaf: no deeper manifests below a shard dump
            subdirs[:] = []
    return sorted(found)


def run(conn, batch: C.Batch, args) -> None:
    root = cache_root()
    host_id = C.host_id(conn)
    gpus_by_name: dict[str, list[int]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT g.name, g.id FROM gpu g JOIN host h ON h.id = g.host_id WHERE h.name = %s", (C.SERVER_HOST,))
        for name, gid in cur.fetchall():
            gpus_by_name.setdefault(name, []).append(gid)
    for path in find_docs(root):
        batch.saw()
        doc = C.load_json(path)
        if not isinstance(doc, dict):
            batch.skip(f"unreadable: {path}")
            continue
        relparts = os.path.relpath(path, root).split(os.sep)
        if len(relparts) < 3:
            batch.skip(f"not under <kind>/<name>/: {path}")
            continue
        kind, name = relparts[0], relparts[1]
        leaf_dir = os.path.dirname(path)
        version_parts = relparts[2:-1]
        fname = relparts[-1]
        version = "/".join(version_parts) or "."
        if fname != "manifest.json":
            version += "#" + fname
        prov = batch.prov(path)
        project = PROJECT_BY_KIND.get(kind, "media")
        dkey = f"teacher_cache/{kind}/{name}"
        with conn.cursor() as cur:
            cur.execute("INSERT INTO dataset (project, key, kind, description, src_path, src_sha256, src_host, ingest_batch_id) "
                        "VALUES (%s, %s, 'teacher-cache', %s, %s, %s, %s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET project = EXCLUDED.project RETURNING id",
                        (project, dkey, f"teacher cache {kind}/{name} under {root}", *prov, batch.id))
            dataset_id = cur.fetchone()[0]
        partial = fname == "pass_receipt.partial.json"
        sealed = (not partial and fname == "manifest.json"
                  and (doc.get("verdict") == "sealed" or bool(doc.get("sealed_utc"))
                       or os.path.isfile(os.path.join(leaf_dir, "SHA256SUMS"))))
        cache_block = doc.get("cache") if isinstance(doc.get("cache"), dict) else {}
        items = doc.get("positions_total") or cache_block.get("positions") or (
            len(doc["shards"]) if isinstance(doc.get("shards"), list) else None)
        size = doc.get("bytes_total") or cache_block.get("bytes_on_disk") or _dir_bytes(leaf_dir)
        created = C.parse_ts(doc.get("sealed_utc") or doc.get("finished_utc") or doc.get("started_utc"))
        splits = {}
        if isinstance(doc.get("shards"), list):
            splits["shards"] = len(doc["shards"])
        with conn.cursor() as cur:
            cur.execute("INSERT INTO dataset_version (dataset_id, version, path, host_id, manifest, splits, item_count, "
                        "size_bytes, sealed, created_at, src_path, src_sha256, src_host, ingest_batch_id) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (dataset_id, version) DO UPDATE SET path = EXCLUDED.path, manifest = EXCLUDED.manifest, "
                        "splits = EXCLUDED.splits, item_count = EXCLUDED.item_count, size_bytes = EXCLUDED.size_bytes, "
                        "sealed = EXCLUDED.sealed, created_at = EXCLUDED.created_at, src_sha256 = EXCLUDED.src_sha256, "
                        "ingest_batch_id = EXCLUDED.ingest_batch_id RETURNING id",
                        (dataset_id, version, leaf_dir, host_id, db.canonical_json(summarise(doc)), db.canonical_json(splits),
                         items if isinstance(items, int) else None, size if isinstance(size, int) else None, sealed, created,
                         *prov, batch.id))
            dv_id = cur.fetchone()[0]
        batch.wrote()

        tname, tver, tsha, tpath = tool_of(doc)
        started = C.parse_ts(doc.get("started_utc"))
        ended = C.parse_ts(doc.get("finished_utc") or doc.get("sealed_utc"))
        if fname == "plan.json" or not tname or not (started or ended):
            continue
        tool_vid = C.tool_version_for(conn, tname, tver, tsha, tpath)
        teacher_dir = None
        sessions = doc.get("sessions")
        if isinstance(sessions, list) and sessions and isinstance(sessions[0], dict):
            teacher_dir = (sessions[0].get("after") or {}).get("model_dir")
        teacher_id = C.model_id_by_path(conn, teacher_dir)
        config = {"tool": tname, "tool_version": tver, "tool_sha256": tsha, "cache_dir": leaf_dir,
                  "arguments": doc.get("arguments"), "cache_config": doc.get("cache_config"),
                  "cache_config_sha256": doc.get("cache_config_sha256"), "device": doc.get("device"),
                  "devices": doc.get("devices"), "torch": doc.get("torch"), "cuda": doc.get("cuda"),
                  "run_id": doc.get("run_id"), "resumed": doc.get("resumed"), "teacher_dir": teacher_dir,
                  "partial": partial}
        status = "unknown" if partial else ("pass" if sealed else "unknown")
        wall = doc.get("seal_wall_seconds_this_session") or C.seconds_between(started, ended)
        env_id = db.ensure_env(conn, torch=doc.get("torch"), cuda=doc.get("cuda"), python=doc.get("python")) \
            if doc.get("torch") else None
        run_id = db.upsert_run(conn, f"cache:{dkey}:{version}", project, "cache", f"cache {kind}/{name} {version}",
                               status, model_id_=teacher_id, tool_version_id=tool_vid, host_id=host_id, env_id=env_id,
                               started_at=started, ended_at=ended if not partial else None, wall_seconds=wall,
                               config={k: v for k, v in config.items() if v is not None}, receipt_path=prov[0],
                               receipt_sha256=prov[1], run_dir=leaf_dir,
                               notes="; ".join(n for n in (
                                   "partial pass receipt: no seal recorded" if partial else None,
                                   "the cache directory name marks it PROVISIONAL" if "provisional" in name else None) if n)
                               or None,
                               prov=prov, batch_id=batch.id)
        C.set_run_lane(conn, run_id, "L-K3" if kind == "k3" else None)
        C.link_run_dataset(conn, run_id, dv_id, "output")
        if teacher_id:
            db.link_run_model(conn, run_id, teacher_id, "teacher")
        for name_, value in (("positions_total", doc.get("positions_total")),
                             ("forward_tokens_total", doc.get("forward_tokens_total")),
                             ("bytes_total", size), ("peak_rss_bytes", doc.get("peak_rss_bytes")),
                             ("compute_seconds", (doc.get("totals") or {}).get("compute_seconds")
                              if isinstance(doc.get("totals"), dict) else None)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                db.put_metric(conn, run_id, name_, float(value), ts=ended)
        for dev in doc.get("devices") or []:
            if isinstance(dev, dict) and len(gpus_by_name.get(str(dev.get("name")), [])) == 1:
                db.add_run_device(conn, run_id, gpus_by_name[dev["name"]][0], "compute",
                                  detail={"mapping": "name-unique", "device": dev})
        batch.wrote()
