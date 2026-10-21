"""ingest.corpora -- corpora, holdout registries and training-data configs -> dataset, dataset_version.

Teacher caches are NOT read here (ingest.teacher_caches owns teacher_cache/**); evaluation banks are
test suites (ingest.banks).  This module covers the training-side data:

    dataset key                        source (AI server, read only)                          version
    corpus/distill                     $OC/distill/corpus/verify_receipt.json                 receipt sha16
    corpus/distill/<sub>[/<sub2>]      $OC/distill/corpus/<sub>/{manifest,build_receipt,       manifest_sha256[:16]
                                       stage_receipt,verify_receipt}.json                     (else file sha16)
    corpus/canonical                   chimera_corpus.corpus_version (raw SQL, read-only)     v<version_no>
    corpus/holdout_registry            $L/corpus/holdouts/HOLDOUT_REGISTRY_v<n>.json          v<n>
    corpus/external/<name>             $L/corpus/external/<name>/** (file listing only)       sha16 of the listing
    training_config/<stem>             $L/distill/{fullkd,media_m1}/<stem>_v<n>.json          v<n>

manifest = the document's top-level fields (big blocks reduced to their size), splits = the per-route /
per-tier / per-split counts the document itself states, item_count = the row/item/document count it
states (NULL when it states none), sealed = covered by a verify receipt that says ok / a sealed registry
row / a registry sha in the file.  chimera_corpus credentials come from the chimera-intake env file
(vault key chimera_corpus_pg); its absence skips that part with a note.  Re-runnable.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from . import common as C
from . import knowledge_common as K
from .common import db

SOURCE = "corpora"
CORPUS_DOCS = ("manifest.json", "build_receipt.json", "stage_receipt.json", "verify_receipt.json")
SPLIT_KEYS = ("splits", "rows_by_split", "stage_t_split", "rows_by_route", "documents_by_route",
              "prompts_by_route", "by_tier", "tokens_by_route")
COUNT_KEYS = ("rows", "items", "documents", "prompts", "n_items", "clips")
SMALL = 4096
INTAKE_ENV = os.environ.get("OMNI_LAB_CHIMERA_INTAKE_ENV", "/home/cereal/.config/chimera-intake/env")


def compact(doc, small: int = SMALL) -> dict:
    """Top-level scalars + small blocks verbatim; anything bigger reduced to its size."""
    if not isinstance(doc, dict):
        return {"_type": type(doc).__name__}
    out = {}
    for k, v in doc.items():
        if isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:small]
        else:
            text = db.canonical_json(v)
            out[k] = v if len(text) <= small else {"_omitted_bytes": len(text), "_len": len(v)
                                                   if isinstance(v, (list, dict)) else None}
    return out


def splits_of(doc: dict) -> dict:
    out = {}
    for k in SPLIT_KEYS:
        v = doc.get(k)
        if isinstance(v, dict) and len(db.canonical_json(v)) <= SMALL:
            out[k] = v
    return out


def count_of(doc: dict) -> int | None:
    for k in COUNT_KEYS:
        v = doc.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    for block in ("vision_in", "counts"):
        b = doc.get(block)
        if isinstance(b, dict):
            for k in COUNT_KEYS:
                v = b.get(k)
                if isinstance(v, int) and not isinstance(v, bool):
                    return v
    return None


def built_at(doc: dict):
    ident = doc.get("identity") if isinstance(doc.get("identity"), dict) else {}
    for v in (ident.get("built_at_utc"), doc.get("built_at_utc"), doc.get("created_utc"), doc.get("sealed_utc"),
              ident.get("utc")):
        ts = C.parse_ts(v)
        if ts:
            return ts
    return None


def dir_bytes(d: Path, prune: set[Path]) -> int:
    total = 0
    for dp, dns, fns in os.walk(d):
        dns[:] = [n for n in dns if Path(dp, n) not in prune]
        for f in fns:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def put_dataset(conn, tally: K.Tally, batch: C.Batch, project: str, key: str, kind: str, description: str | None,
                prov) -> int:
    ds_id, state = K.upsert(conn, "dataset", ("key",), {
        "project": project, "key": key, "kind": kind, "description": description,
        "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id},
        ignore_in_compare=("ingest_batch_id", "src_path", "src_sha256", "src_host"))
    tally.add("dataset", state)
    return ds_id


def put_version(conn, tally: K.Tally, batch: C.Batch, ds_id: int, version: str, path: str, manifest: dict,
                splits: dict, item_count, size_bytes, sealed: bool, created_at, prov) -> int:
    dv_id, state = K.upsert(conn, "dataset_version", ("dataset_id", "version"), {
        "dataset_id": ds_id, "version": version, "path": path, "host_id": C.host_id(conn),
        "manifest": manifest, "splits": splits, "item_count": item_count, "size_bytes": size_bytes,
        "sealed": sealed, "created_at": created_at,
        "src_path": prov[0], "src_sha256": prov[1], "src_host": prov[2], "ingest_batch_id": batch.id},
        jsonb_cols=("manifest", "splits"))
    tally.add("dataset_version", state)
    return dv_id


def project_for(rel: str) -> str:
    head = rel.split("/")[0]
    return {"media": "media", "vision_in": "media", "k3": "k3"}.get(head, "corpus")


# ---------------------------------------------------------------------------

def ingest_distill_corpus(conn, batch, tally) -> None:
    root = Path(C.OC, "distill", "corpus")
    if not root.is_dir():
        batch.skip(f"{root} missing")
        return
    root_receipt = C.load_json(root / "verify_receipt.json") or {}
    verified = {p: (v or {}).get("recomputed_equal") is True
                for p, v in (root_receipt.get("manifests") or {}).items()} if root_receipt.get("ok") is True else {}
    doc_dirs: dict[Path, list[str]] = {}
    for dp, dns, fns in os.walk(root):
        depth = len(Path(dp).relative_to(root).parts)
        if depth >= 3:
            dns[:] = []
        hits = [f for f in CORPUS_DOCS if f in fns]
        if hits:
            doc_dirs[Path(dp)] = hits
    for d, hits in sorted(doc_dirs.items()):
        batch.saw()
        rel = d.relative_to(root).as_posix()
        primary = next(h for h in CORPUS_DOCS if h in hits)
        doc = C.load_json(d / primary)
        if not isinstance(doc, dict):
            batch.skip(f"{d / primary}: unreadable")
            continue
        prov = batch.prov(d / primary)
        version = (doc.get("manifest_sha256") or doc.get("receipt_sha256") or prov[1])[:16]
        manifest = {name: compact(C.load_json(d / name)) for name in hits}
        own_verify = C.load_json(d / "verify_receipt.json") if "verify_receipt.json" in hits else None
        sealed = bool(verified.get((d / "manifest.json").as_posix())) or bool(isinstance(own_verify, dict)
                                                                               and own_verify.get("ok") is True)
        if rel == ".":
            key, project, desc = "corpus/distill", "corpus", "distill corpus build: verify receipt over every sub-manifest"
        else:
            key, project = f"corpus/distill/{rel}", project_for(rel)
            desc = str(doc.get("set") or doc.get("schema") or primary)[:300]
        prune = {p for p in doc_dirs if p != d and d in p.parents}
        ds_id = put_dataset(conn, tally, batch, project, key, "corpus", desc, prov)
        put_version(conn, tally, batch, ds_id, version, d.as_posix(), manifest, splits_of(doc), count_of(doc),
                    None if rel == "." else dir_bytes(d, prune), sealed, built_at(doc), prov)


def ingest_holdouts(conn, batch, tally) -> None:
    d = Path(C.LX, "corpus", "holdouts")
    files = sorted(d.glob("HOLDOUT_REGISTRY_v*.json"))
    ds_id = None
    for p in files:
        batch.saw()
        doc = C.load_json(p)
        m = re.search(r"_v(\d+)\.json$", p.name)
        if not isinstance(doc, dict) or not m:
            batch.skip(f"{p}: unreadable")
            continue
        prov = batch.prov(p)
        if ds_id is None:
            ds_id = put_dataset(conn, tally, batch, "corpus", "corpus/holdout_registry", "corpus",
                                "chimera-corpus intake holdout registry (rows never trained; eval/holdout banks)", prov)
        file_rows = {k: v.get("rows") for k, v in (doc.get("files") or {}).items() if isinstance(v, dict)}
        reg_dir = doc.get("dir") or (d / f"registry_v{m.group(1)}").as_posix()
        put_version(conn, tally, batch, ds_id, f"v{m.group(1)}", reg_dir, compact(doc), {"rows_by_file": file_rows},
                    file_rows.get("texts"), dir_bytes(Path(reg_dir), set()) if Path(reg_dir).is_dir() else None,
                    bool(doc.get("registry_sha256")), C.parse_ts(doc.get("created_utc")), prov)


def ingest_external(conn, batch, tally) -> None:
    base = Path(C.LX, "corpus", "external")
    if not base.is_dir():
        return
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        batch.saw()
        listing = []
        total = 0
        for dp, _dns, fns in os.walk(d):
            for f in fns:
                fp = Path(dp, f)
                size = fp.lstat().st_size
                listing.append((fp.relative_to(d).as_posix(), size))
                total += size
        fingerprint = hashlib.sha256("\n".join(f"{r}\t{n}" for r, n in sorted(listing)).encode()).hexdigest()
        audio = sum(1 for r, _n in listing if r.endswith((".flac", ".wav", ".mp3")))
        by_split: dict[str, int] = {}
        for r, _n in listing:                         # <corpus>/<split>/... e.g. LibriSpeech/test-clean/...
            parts = r.split("/")
            if len(parts) > 2:
                by_split[parts[1]] = by_split.get(parts[1], 0) + 1
        prov = (d.as_posix(), fingerprint, C.SERVER_HOST)
        batch.file_shas.append(fingerprint)
        ds_id = put_dataset(conn, tally, batch, "speech" if "speech" in d.name else "corpus",
                            f"corpus/external/{d.name}", "corpus", f"external corpus under {d.as_posix()}", prov)
        put_version(conn, tally, batch, ds_id, fingerprint[:16], d.as_posix(),
                    {"files": len(listing), "audio_files": audio, "splits_seen": sorted(by_split),
                     "version_rule": "sha256 of the sorted (relative path, size) listing"},
                    {"files_by_split": by_split},
                    audio or len(listing), total, False, None, prov)


def ingest_training_configs(conn, batch, tally) -> None:
    for sub, project in (("fullkd", "chimera-omni"), ("media_m1", "media")):
        for p in sorted(Path(C.LX, "distill", sub).glob("*_v*.json")):
            m = re.match(r"^(.+)_v(\d+)\.json$", p.name)
            if not m:
                continue
            batch.saw()
            doc = C.load_json(p)
            if not isinstance(doc, dict):
                batch.skip(f"{p}: unreadable")
                continue
            prov = batch.prov(p)
            ds_id = put_dataset(conn, tally, batch, project, f"training_config/{m.group(1)}", "training-config",
                                f"training data config under {p.parent.as_posix()} (schema {doc.get('schema')})", prov)
            sources = doc.get("sources") if isinstance(doc.get("sources"), (list, dict)) else doc.get("routes")
            item_count = len(sources) if isinstance(sources, (list, dict)) else None
            splits = {"routes": sorted(doc["routes"])} if isinstance(doc.get("routes"), dict) else {}
            put_version(conn, tally, batch, ds_id, f"v{m.group(2)}", p.as_posix(), compact(doc), splits, item_count,
                        p.stat().st_size, False, C.parse_ts(C.iso_mtime(p)), prov)


def intake_env() -> dict[str, str]:
    """KEY=VALUE lines of the chimera-intake env file (values never printed or logged)."""
    out: dict[str, str] = {}
    text = C.read_text(INTAKE_ENV, limit=1 << 16)
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def ingest_canonical(conn, batch, tally) -> None:
    env = intake_env()
    if not env.get("CHIMERA_CORPUS_PG_PASSWORD"):
        batch.skip(f"chimera_corpus registry skipped: no credential in {INTAKE_ENV} on {db.this_host()}")
        return
    import psycopg2
    src = psycopg2.connect(host=env.get("CHIMERA_CORPUS_PG_HOST", "127.0.0.1"), port=int(env.get("CHIMERA_CORPUS_PG_PORT", "25490")),
                           dbname=env.get("CHIMERA_CORPUS_PG_DB", "chimera_corpus"), user=env.get("CHIMERA_CORPUS_PG_USER", "chimera_corpus"),
                           password=env["CHIMERA_CORPUS_PG_PASSWORD"], connect_timeout=5,
                           options="-c default_transaction_read_only=on -c statement_timeout=20000",
                           application_name="omni_lab/ingest-corpora-readonly")
    try:
        with src.cursor() as cur:
            cur.execute("SELECT version_no, parent_version_no, status, manifest_path, manifest_sha256, archive_path, "
                        "holdout_registry_version, tool_version, tool_sha256, counts, sealed_utc, sealed_by "
                        "FROM corpus_version ORDER BY version_no")
            rows = cur.fetchall()
        src.rollback()
    finally:
        src.close()
    prov_ref = "chimera_corpus.corpus_version"
    for (no, parent, status, mpath, msha, apath, hrv, tver, tsha, counts, sealed_utc, sealed_by) in rows:
        batch.saw()
        prov = (f"{prov_ref}#{no}", (msha or "").strip() or None, C.SERVER_HOST)
        ds_id = put_dataset(conn, tally, batch, "corpus", "corpus/canonical", "corpus",
                            "canonical Chimera corpus versions (chimera_corpus.corpus_version, intake.py)", prov)
        counts = counts or {}
        put_version(conn, tally, batch, ds_id, f"v{no}", mpath, compact({
                        "parent_version_no": parent, "status": status, "manifest_sha256": (msha or "").strip(),
                        "archive_path": apath, "holdout_registry_version": hrv, "tool_version": tver,
                        "tool_sha256": (tsha or "").strip(), "sealed_by": sealed_by, "counts": counts}),
                    splits_of(counts), count_of(counts) or (counts.get("adopted") or {}).get("dedupe_index", {}).get("rows"),
                    None, status == "sealed", sealed_utc, prov)


def run(conn, batch: C.Batch, args) -> None:
    C.roots_from_env()
    tally = K.Tally(batch)
    ingest_distill_corpus(conn, batch, tally)
    ingest_holdouts(conn, batch, tally)
    ingest_external(conn, batch, tally)
    ingest_training_configs(conn, batch, tally)
    ingest_canonical(conn, batch, tally)
    K.finish(batch, tally)


if __name__ == "__main__":
    K.main_guard(sys.modules[__name__])
