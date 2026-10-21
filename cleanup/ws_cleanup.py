#!/usr/bin/env python3
"""ws_cleanup.py 1.0.0 - workstation wave (W4) of the 2026-10-21 weight cleanup.

Removes (a) clean Neural Forge agent worktrees whose only weights are the 7 fold-child blobs that neural-forge
history keeps, with `git worktree remove` (never --force; branches stay), and (b) the 7 fold-child
model.safetensors files from the neural-forge working tree with `git rm` + one pathspec-limited commit (the
blobs stay in history; config/tokenizer files stay in place as the recipe).

Records first, in omni_lab: 7 child rows (host 'workstation', git blob sha1 + sha256), one unit row per
worktree, one model_deletion row per removed item (status 'deleting' -> 'deleted' once verified absent).

Refuses a worktree when: it is not in the approved list, `git status --porcelain --ignored` shows anything
except Python caches, a child file's blob differs from the recorded blob, or a running process names it.

    C:/Python314/python.exe cleanup/ws_cleanup.py --plan cleanup_plan.json --skip-list cleanup/inputs/ws_codex_route.json [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TOOL = "ws_cleanup 1.0.0"
NF = "C:/Users/cereal/Documents/ai training/neural-forge"
WT_ROOT = "C:\\Users\\cereal\\Documents\\ai training\\.agent-worktrees\\neural-forge-a495ffdd2f\\"
OWNER_ORDER = "[Cyril quote withheld]"
CACHE_OK = ("__pycache__/", ".pyc", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args, cwd=NF) -> str:
    r = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} (in {cwd}) rc={r.returncode}: {r.stderr.strip()[-600:]}")
    return r.stdout


def free(drive: str) -> int:
    return shutil.disk_usage(drive + "\\").free


def process_cmdlines() -> list[str]:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine } | Where-Object { $_ }"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [line.lower().replace("\\", "/") for line in r.stdout.splitlines() if line.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def posix(p: str) -> str:
    return p.replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--skip-list", required=True, help="JSON list of worktree paths routed to Codex (never touched)")
    ap.add_argument("--log", default=str(HERE / "inputs" / "ws_cleanup_log.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import omni_lab_db as db
    from ingest import common as C

    plan = json.load(open(args.plan, encoding="utf-8"))
    skip = {p.lower() for p in json.load(open(args.skip_list, encoding="utf-8"))}
    children = [w for w in plan["workstation_delete"] if w["category"] == "DELETE_NF_EXPERIMENT"]
    worktrees = [w for w in plan["workstation_delete"] if w["category"] == "DELETE_DUPLICATE"
                 and w["path"].startswith(WT_ROOT) and w["path"].lower() not in skip]
    if args.limit:
        worktrees = worktrees[:args.limit]
    logf = open(args.log, "a", encoding="utf-8")

    def log(**ev):
        ev = {"ts": now(), **ev}
        print(json.dumps(ev, default=str), flush=True)
        logf.write(json.dumps(ev, default=str) + "\n")
        logf.flush()

    rel_children = [posix(os.path.relpath(c["path"], NF.replace("/", "\\"))) + "/model.safetensors" for c in children]
    head = git("rev-parse", "HEAD").strip()
    blobs = {}
    for line in git("ls-tree", "HEAD", "--", *rel_children).splitlines():
        meta, path = line.split("\t", 1)
        blobs[path] = meta.split()[2]
    if len(blobs) != 7:
        raise SystemExit(f"expected 7 child blobs in HEAD {head}, found {len(blobs)}")
    for c, rel in zip(children, rel_children):
        if blobs[rel] != c["git_blob"]:
            raise SystemExit(f"HEAD blob for {rel} is {blobs[rel]}, plan recorded {c['git_blob']}")
    log(step="start", tool=TOOL, head=head, worktrees=len(worktrees), dry_run=args.dry_run, free_c=free("C:"))

    conn = None if args.dry_run else db.connect("rw", app="ws-cleanup", statement_timeout_ms=60000)
    child_ids: dict[str, int] = {}
    if conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM host WHERE name = 'workstation'")
            host_id = cur.fetchone()[0]
        for c, rel in zip(children, rel_children):
            p = posix(c["path"])
            child_ids[rel] = db.upsert_model(
                conn, C.sha16_path_key(p, host="workstation"), "neural-forge", "fold-child", "hf", path=p, host_id=host_id,
                size_bytes=c["size_bytes"], config={"git_blob_sha1": c["git_blob"], "sha256": c["sha256"],
                                                    "git_repo": NF, "git_commit_with_blob": head,
                                                    "rebuild_source": c["rebuild_source"], "what": c["what"]},
                status="present", notes=f"Neural Forge fold child (git-tracked); recorded by {TOOL}")
        conn.commit()
        log(step="child-rows", ids=child_ids)

    cmdlines = process_cmdlines()
    done = skipped = 0
    for w in worktrees:
        wt = w["path"]
        wtp = posix(wt)
        try:
            if not os.path.isdir(wt):
                log(path=wt, step="absent-already")
                continue
            reasons = []
            if any(wtp.lower() in c for c in cmdlines):
                reasons.append("named by a running process command line")
            st = git("status", "--porcelain", "--ignored", "--untracked-files=all", cwd=wt)
            noise = [l for l in st.splitlines() if l and not any(k in posix(l) for k in CACHE_OK)]
            if noise:
                reasons.append(f"not clean: {noise[:5]}")
            found = {}
            for rel in rel_children:
                fp = os.path.join(wt, rel)
                if os.path.exists(fp):
                    found[rel] = git("hash-object", fp, cwd=wt).strip()
            mism = {r: b for r, b in found.items() if b != blobs[r]}
            if mism:
                reasons.append(f"child blob differs from neural-forge HEAD: {mism}")
            if reasons:
                skipped += 1
                log(path=wt, step="SKIPPED", reasons=reasons)
                continue
            branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt).strip()
            wt_head = git("rev-parse", "HEAD", cwd=wt).strip()
            if args.dry_run:
                log(path=wt, step="would-remove", branch=branch, head=wt_head, child_copies=len(found))
                continue
            before = free("C:")
            digests = {r: {"git_blob_sha1": b, "duplicate_of": child_ids[r]} for r, b in found.items()}
            listing = hashlib.sha256("\n".join(f"{r}\t{b}" for r, b in sorted(found.items())).encode()).hexdigest()
            biggest = max(found, key=lambda r: next(c["size_bytes"] for c, rr in zip(children, rel_children) if rr == r)) if found else None
            with conn.cursor() as cur:
                unit_id = db.upsert_model(
                    conn, C.sha16_path_key(wtp, host="workstation"), "neural-forge", "other", "git-worktree", path=wtp,
                    host_id=host_id, size_bytes=w["size_bytes"],
                    config={"git_worktree": {"branch": branch, "head": wt_head, "repo": NF},
                            "carries_child_blobs": digests}, status="present",
                    notes=f"Codex agent worktree of neural-forge; recorded by {TOOL} before removal")
                cur.execute("UPDATE model SET status='deleting', delete_reason=%s, rebuild_recipe_path=%s WHERE id=%s",
                            (f"cleanup 2026-10-21: clean worktree; its weights are the 7 fold-child blobs kept in "
                             f"neural-forge history; branch {branch} kept", f"git:{NF}#{branch}@{wt_head}", unit_id))
                cur.execute(
                    "INSERT INTO model_deletion (model_id, duplicate_of_model_id, host_id, unit_path, plan_wave, status, "
                    "size_bytes_du, weight_bytes, file_count, layout, digest_mode, listing_sha256, file_digests, "
                    "recipe_archive_path, recipe_files, affected_model_ids, reason, owner_order, preflight, df_before) "
                    "VALUES (%s,%s,%s,%s,'W4-WS','deleting',%s,%s,%s,'git-worktree','git-blob',%s,%s::jsonb,%s,'[]'::jsonb,"
                    "%s,%s,%s,%s::jsonb,%s::jsonb) ON CONFLICT (host_id, unit_path) DO UPDATE SET status='deleting', "
                    "error=NULL RETURNING id",
                    (unit_id, child_ids.get(biggest), host_id, wtp, w["size_bytes"], w["size_bytes"], len(found), listing,
                     db.canonical_json(digests), f"git:{NF}#{branch}@{wt_head}", [unit_id],
                     f"clean Codex agent worktree (duplicate weights); {w['reason'][:300]}", OWNER_ORDER,
                     db.canonical_json({"status_porcelain_ignored": st.splitlines()[:50], "process_cmdline_hits": 0,
                                        "checked_at": now(), "tool": TOOL}),
                     db.canonical_json({"drive": "C:", "free_bytes": before})))
                del_id = cur.fetchone()[0]
            conn.commit()
            r = subprocess.run(["git", "-C", NF, "worktree", "remove", wt], capture_output=True, text=True)
            gone = not os.path.exists(wt)
            with conn.cursor() as cur:
                if r.returncode != 0 or not gone:
                    cur.execute("UPDATE model_deletion SET status='failed', error=%s WHERE id=%s",
                                (f"rc={r.returncode} gone={gone}: {r.stderr[-800:]}", del_id))
                    cur.execute("UPDATE model SET status = CASE WHEN %s THEN 'deleting' ELSE 'present' END WHERE id=%s",
                                (gone, unit_id))
                    conn.commit()
                    log(path=wt, step="FAILED", rc=r.returncode, gone=gone, stderr=r.stderr[-800:])
                    if not gone and r.returncode != 0:
                        skipped += 1
                        continue
                    raise RuntimeError("worktree remove reported failure but path is gone")
                after = free("C:")
                cur.execute("UPDATE model_deletion SET status='deleted', executed_at=now(), verified_absent_at=now(), "
                            "df_after=%s::jsonb WHERE id=%s", (db.canonical_json({"drive": "C:", "free_bytes": after}), del_id))
                cur.execute("UPDATE model SET status='deleted', deleted_at=now() WHERE id=%s", (unit_id,))
            conn.commit()
            done += 1
            log(path=wt, step="REMOVED", branch=branch, model_deletion_id=del_id, freed=after - before)
        except Exception as exc:
            log(path=wt, step="ERROR", error=str(exc)[:1500])
            if conn:
                conn.rollback()
            return 1

    # (b) the 7 fold children in the neural-forge working tree
    if args.dry_run:
        log(step="would-git-rm", files=rel_children)
        log(step="end", removed=done, skipped=skipped)
        return 0
    present = [r for r in rel_children if os.path.exists(os.path.join(NF, r))]
    dirty = git("status", "--porcelain", "--", *present).strip()
    if dirty:
        log(step="children-SKIPPED", reason=f"modified in working tree: {dirty[:500]}")
        log(step="end", removed=done, skipped=skipped)
        return 0
    before = free("C:")
    with conn.cursor() as cur:
        for c, rel in zip(children, rel_children):
            if rel not in present:
                continue
            got = sha256(os.path.join(NF, rel))
            if got != c["sha256"]:
                raise RuntimeError(f"{rel} sha256 {got} != plan {c['sha256']}")
            mid = child_ids[rel]
            cur.execute("UPDATE model SET status='deleting', delete_reason=%s, rebuild_recipe_path=%s WHERE id=%s",
                        ("cleanup 2026-10-21: Neural Forge experiment child; bytes stay in neural-forge git history",
                         f"git:{NF}@{head}:{rel}", mid))
            cur.execute(
                "INSERT INTO model_deletion (model_id, host_id, unit_path, plan_wave, status, size_bytes_du, weight_bytes, "
                "file_count, layout, digest_mode, listing_sha256, file_digests, recipe_archive_path, recipe_files, "
                "affected_model_ids, reason, owner_order, preflight, df_before) VALUES (%s,%s,%s,'W4-WS','deleting',%s,%s,1,'hf',"
                "'git-blob',%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s::jsonb) "
                "ON CONFLICT (host_id, unit_path) DO UPDATE SET status='deleting', error=NULL RETURNING id",
                (mid, host_id, posix(os.path.join(NF, rel)), c["size_bytes"], c["size_bytes"], c["sha256"],
                 db.canonical_json({"model.safetensors": {"git_blob_sha1": c["git_blob"], "sha256": got, "size": c["size_bytes"]}}),
                 f"git:{NF}@{head}:{rel}", db.canonical_json(["config.json", "generation_config.json", "tokenizer.json",
                                                             "tokenizer_config.json (kept in place)"]),
                 [mid], c["reason"][:500], OWNER_ORDER, db.canonical_json({"checked_at": now(), "tool": TOOL, "head": head}),
                 db.canonical_json({"drive": "C:", "free_bytes": before})))
    conn.commit()
    git("rm", "-q", "--", *present)
    msg = ("cleanup 2026-10-21: git rm the 7 Neural Forge fold-child weights (bytes stay in history)\n\n"
           f"Cyril's order: \"{OWNER_ORDER}\"\nRecorded in omni_lab model_deletion (wave W4-WS). Config and tokenizer "
           f"files stay in each child dir. Restore any one with: git show {head}:<path> > <path>\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    r = subprocess.run(["git", "-C", NF, "commit", "-q", "-F", "-", "--", *present], input=msg, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"commit failed: {r.stderr[-800:]}")
    commit = git("rev-parse", "HEAD").strip()
    after = free("C:")
    with conn.cursor() as cur:
        for rel in present:
            cur.execute("UPDATE model_deletion SET status='deleted', executed_at=now(), verified_absent_at=now(), "
                        "df_after=%s::jsonb, ledger_receipt_sha=%s WHERE unit_path=%s",
                        (db.canonical_json({"drive": "C:", "free_bytes": after}), f"git-commit:{commit}",
                         posix(os.path.join(NF, rel))))
            cur.execute("UPDATE model SET status='deleted', deleted_at=now() WHERE id=%s", (child_ids[rel],))
    conn.commit()
    log(step="children-REMOVED", files=present, commit=commit, freed=after - before)
    log(step="end", removed=done, skipped=skipped, free_c=free("C:"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
