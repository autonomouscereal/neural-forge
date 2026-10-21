#!/usr/bin/env python3
"""verify_deleted.py 1.0.0 - independent check of every server model_deletion row (read-only role).

For each row on host 'ai': the unit path is absent (or, when git-tracked files were kept, only those remain),
<recipe>/MANIFEST.tsv hashes to recipe_archive_sha256, DELETION_RECEIPT.json carries the same listing_sha256,
every recipe file listed is present in tree.tar.gz (or the 1.0.0 tree/), and every affected model row has status 'deleted' with
deleted_at, delete_reason and rebuild_recipe_path set. Prints one line per problem and a summary.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import omni_lab_db as db  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


conn = db.connect("ro", app="cleanup-verify")
cur = conn.cursor()
cur.execute("SELECT d.id, d.unit_path, d.status, d.recipe_archive_path, d.recipe_archive_sha256, d.listing_sha256, "
            "d.recipe_files, d.affected_model_ids, d.preflight->'git_tracked_kept', d.plan_wave FROM model_deletion d "
            "JOIN host h ON h.id = d.host_id WHERE h.name = 'ai' ORDER BY d.id")
rows = cur.fetchall()
problems, ok = [], 0
for did, path, status, rec, rec_sha, listing, files, affected, tracked, wave in rows:
    p = []
    if status != "deleted":
        p.append(f"status {status}")
    tracked = tracked or []
    if os.path.lexists(path):
        if not tracked:
            p.append("path still exists")
        else:
            left = [str(q) for q in Path(path).rglob("*") if q.is_file()]
            extra = [q for q in left if os.path.relpath(q, path) not in tracked]
            if extra:
                p.append(f"{len(extra)} untracked files remain")
    man = os.path.join(rec, "MANIFEST.tsv")
    if not os.path.isfile(man) or sha(man) != rec_sha:
        p.append("MANIFEST.tsv missing or sha mismatch")
    try:
        receipt = json.load(open(os.path.join(rec, "DELETION_RECEIPT.json")))
        if receipt.get("listing_sha256") != listing:
            p.append("receipt listing_sha256 mismatch")
    except Exception as exc:
        p.append(f"receipt unreadable: {exc}")
    tar = os.path.join(rec, "tree.tar.gz")
    if os.path.isfile(tar):
        import tarfile
        with tarfile.open(tar, "r:gz") as tf:
            names = {m.name for m in tf if m.isfile()}
        missing = [f for f in files if f not in names]
    else:
        missing = [f for f in files if not os.path.isfile(os.path.join(rec, "tree", f))]
    if missing:
        p.append(f"{len(missing)} recipe files missing from tree/")
    cur.execute("SELECT count(*) FROM model WHERE id = ANY(%s) AND NOT (status = 'deleted' AND deleted_at IS NOT NULL "
                "AND delete_reason IS NOT NULL AND rebuild_recipe_path IS NOT NULL)", (affected,))
    nbad = cur.fetchone()[0]
    if nbad:
        p.append(f"{nbad} affected model rows not fully marked deleted")
    if p:
        problems.append((did, wave, path, p))
    else:
        ok += 1
for did, wave, path, p in problems:
    print(f"PROBLEM {did} {wave} {path}: {'; '.join(p)}")
print(f"verified ok {ok} / {len(rows)} server model_deletion rows; problems {len(problems)}")
