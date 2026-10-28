#!/usr/bin/env python3
"""nas_group_delete.py 1.1.0 - delete verified-safe NAS groups one unit at a time, recording each first.

Owner order (Cyril, 2026-10-27, relayed by the NAS-cleanup workflow): "[Cyril quote withheld]".
Groups and their evidence: neural-forge/reports/omni-campaign-2026-09/NAS_CLEANUP_2026-10-27.md. This tool never
chooses what to delete; it reads an explicit plan (groups -> units) and only refuses.

1.1.0 (2026-10-28, round 2) adds three optional plan fields, all off by default so 1.0.0 plans behave the same:
  - plan "owner_order" / "reason_prefix": the owner wording and row-reason prefix recorded for that plan
  - unit "unreadable_in_place" {relpath: written reason}: files the cereal user cannot open (e.g. 0600 nobody-owned
    files on the share). They must exist, must not be planned, and must really refuse open(); they are recorded by
    size and mtime only (digest kind "unreadable"), never hashed, never archived, and stay in place
  - unit "volume" ("nas" default, or "home" / "lexar"): which filesystem df_before/df_after measure; the unit must
    sit on that filesystem

Differences from cleanup_exec.py 1.3.0 (whose gate primitives it reuses):
  - a unit is either mode "tree" (the whole directory goes, rm -rf --one-file-system) or mode "files" (only the
    listed relpaths go, one os.unlink each, never rm -r; every other file stays in place as evidence and is
    recorded in model_deletion.preflight.kept_in_place, which verify_deleted.py 1.1.0 honours)
  - backups and public model stores are not "weights", so the >= 50 MB non-weight and data-dir checks do not
    apply; instead every planned file must still have its planned size and an mtime older than --cutoff
  - added gate checks: inbound symlinks (scan files of "link<TAB>target" lines; a link resolving into the unit
    blocks unless its link path is under a --dead-link-prefix given with a reason), and an omni_lab row scan
    (any row of any table except model/model_deletion whose JSON text contains the unit path blocks)
  - protected-list overlap: an entry inside the unit always blocks; an entry at or above the unit blocks unless
    the plan unit carries protected_waivers {exact protected path: written reason} (owner ruling recorded in the
    preflight). Used only where the most specific entry is plan-UNSURE ("owner decides") and the owner has ruled.

Per unit: gate -> measure (cleanup_exec.measure) -> archive (files < 50 MiB that are not weights, into
<recipe>/tree.tar.gz, plus MANIFEST.tsv, REBUILD.md, DELETION_RECEIPT.json) -> record (one transaction, status
'deleting') -> delete -> verify absent -> 'deleted'. Any refusal or failure stops the run (exit 1).
On ENOSPC (the share is NFS-backed ZFS behind Samba; at 100% it cannot record a remove) a file is truncated to
0 and then unlinked; the first truncation must show the space back in df within 180 s or the run stops.
A re-run after a failed delete refreshes the 'failed' row; a 'deleted' row is never reopened.

    PY=/home/cereal/.venvs/chimera-intake/bin/python
    set -a; . ~/.config/omni-lab/env; set +a
    nice -n 19 ionice -c3 $PY nas_group_delete.py --plan plan.json --group D3-... --protected protected.json \
        --generic generic_roots.json --symlinks a.tsv --symlinks b.tsv --run-dir RUN [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import cleanup_exec as CE  # noqa: E402

TOOL = "nas_group_delete 1.1.0"
OWNER_ORDER = "[Cyril quote withheld]"
SMALL = 50 * 1024 ** 2
SKIP_TABLES = {"model", "model_deletion"}


def resolve_link(link: str, target: str) -> str:
    t = target if target.startswith("/") else os.path.join(os.path.dirname(link), target)
    return os.path.normpath(t)


def load_links(paths: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "\t" in line:
                    link, target = line.rstrip("\n").split("\t", 1)
                    out.append((link, resolve_link(link, target)))
    return out


def omni_lab_refs(conn, unit: str) -> list[tuple[str, int]]:
    from psycopg2 import sql
    hits = []
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
                    "AND table_type = 'BASE TABLE' ORDER BY 1")
        tables = [r[0] for r in cur.fetchall() if r[0] not in SKIP_TABLES]
        for t in tables:
            cur.execute(sql.SQL("SELECT count(*) FROM {} x WHERE row_to_json(x)::text LIKE %s").format(sql.Identifier(t)),
                        ("%" + unit.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",))
            n = cur.fetchone()[0]
            if n:
                hits.append((t, n))
    conn.rollback()
    return hits


def unreadable_check(u: dict, by_rel: dict) -> list[str]:
    """unreadable_in_place entries must exist, stay unplanned, carry a reason and really refuse open()."""
    reasons = []
    for rel, why in u.get("unreadable_in_place", {}).items():
        if not isinstance(why, str) or len(why.strip()) < 20:
            reasons.append(f"unreadable_in_place {rel} has no written reason")
        if rel in u["planned"]:
            reasons.append(f"unreadable_in_place {rel} is also planned for deletion")
        if rel not in by_rel:
            reasons.append(f"unreadable_in_place {rel} is not in the unit")
            continue
        try:
            with open(os.path.join(u["path"], rel), "rb") as fh:
                fh.read(1)
            reasons.append(f"unreadable_in_place {rel} is readable now; hash it instead of skipping it")
        except PermissionError:
            pass
    return reasons


def volume_check(u: dict) -> list[str]:
    """The unit must sit on the filesystem its df figures are taken from."""
    vol = u.get("volume", "nas")
    mount = CE.VOLUME_MOUNT.get(vol)
    if mount is None:
        return [f"unknown volume {vol!r}"]
    if os.stat(u["path"]).st_dev != os.stat(mount).st_dev:
        return [f"unit is not on the {vol} filesystem ({mount})"]
    return []


def measure_unit(u: dict, files: list[dict], fh_log) -> dict:
    """CE.measure over the readable files; unreadable_in_place files are recorded by size and mtime only and their
    listing line carries the digest token 'unreadable' (same line format as CE.measure)."""
    skip = u.get("unreadable_in_place", {})
    m = CE.measure({"path": u["path"]}, [f for f in files if f["rel"] not in skip], "index+headers+listing", fh_log)
    if not skip:
        return m
    for f in files:
        if f["rel"] in skip:
            m["digests"][f["rel"]] = {"size": f["size"], "unreadable": skip[f["rel"]],
                                      "mtime": dt.datetime.fromtimestamp(f["mtime"], dt.timezone.utc).isoformat()}
    lines = []
    for rel in sorted(m["digests"]):
        d = m["digests"][rel]
        dig = (d.get("sha256") or d.get("header_sha256") or d.get("head64m_sha256")
               or ("->" + d["symlink"] if "symlink" in d else "") or ("unreadable" if "unreadable" in d else ""))
        lines.append(f"{rel}\t{d['size']}\t{dig}")
    m["listing_lines"] = lines
    m["listing_sha256"] = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    m["total_bytes"] = sum(f["size"] for f in files)
    return m


def gate(u: dict, protected: list[dict], generic: set[str], refs, links: list[tuple[str, str]], dead_prefixes: dict,
         conn, cutoff: float, prior_digests: dict | None = None) -> dict:
    unit = u["path"]
    reasons: list[str] = []
    checked: dict = {"at": CE.now(), "tool": TOOL, "cutoff": dt.datetime.fromtimestamp(cutoff, dt.timezone.utc).isoformat()}
    if not os.path.lexists(unit):
        return {"reasons": ["path absent"], "checked": checked, "absent": True}
    if os.path.islink(unit) or not os.path.isdir(unit) or os.path.realpath(unit) != unit:
        reasons.append(f"not a plain directory or symlinked component (realpath {os.path.realpath(unit)})")
    reasons += volume_check(u)
    checked["volume"] = u.get("volume", "nas")
    waivers = u.get("protected_waivers", {})
    waived = []
    for p in protected:
        if CE.under(p["path"], unit) and p["path"] != unit:
            reasons.append(f"protected {p['class']} inside the unit: {p['path']}")
        elif CE.under(unit, p["path"]):
            if p["path"] in waivers:
                waived.append({"class": p["class"], "path": p["path"], "reason": waivers[p["path"]]})
            else:
                reasons.append(f"protected {p['class']} overlaps: {p['path']}")
    checked["protected_waived"] = waived
    scan = CE.proc_scan()
    opened = [p for p in scan["open"] if CE.under(p, unit)]
    if opened:
        reasons.append(f"open by a live process: {opened[:5]}")
    checked["proc_scan"] = {"open_paths": len(scan["open"]), "strings": len(scan["strings"]), "errors": scan["errors"]}
    token_srcs: list[tuple[str, str]] = []
    for pid, kind, text in scan["strings"]:
        for t in CE.tokens_from_text(text):
            token_srcs.append((f"pid{pid}:{kind}", t))
    token_srcs += CE.crontab_tokens() + CE.docker_mount_tokens() + refs.file_tokens()
    hits = [(s, t) for s, t in token_srcs if CE.under(t, unit)]
    parent_hits = [(s, t) for s, t in token_srcs if unit.startswith(t + "/") and t not in generic]
    if hits:
        reasons.append(f"referenced (unit or inside): {sorted(set(hits))[:8]}")
    if parent_hits:
        reasons.append(f"referenced (non-generic parent): {sorted(set(parent_hits))[:8]}")
    checked["reference_tokens"] = len(token_srcs)
    inbound, dead = [], []
    for link, target in links:
        if CE.under(target, unit) and not CE.under(link, unit):
            pre = next((p for p in dead_prefixes if CE.under(link, p)), None)
            (dead if pre else inbound).append((link, target))
    if inbound:
        reasons.append(f"inbound symlinks into unit: {inbound[:8]}")
    checked["inbound_symlinks"] = {"scanned": len(links), "dead_tree_breadcrumbs": dead,
                                   "dead_prefix_reasons": {p: r for p, r in dead_prefixes.items()
                                                           if any(CE.under(l, p) for l, _ in dead)}}
    lab = omni_lab_refs(conn, unit)
    if lab:
        reasons.append(f"omni_lab rows name the unit: {lab}")
    checked["omni_lab_tables_with_refs"] = lab
    try:
        files, dirs = CE.walk_unit(unit)
    except Exception as exc:
        return {"reasons": reasons + [f"walk failed: {exc}"], "checked": checked}
    by_rel = {f["rel"]: f for f in files}
    links_inside = [f["rel"] for f in files if "symlink" in f]
    if links_inside:
        reasons.append(f"symlinks inside unit: {links_inside[:5]}")
    reasons += unreadable_check(u, by_rel)
    checked["unreadable_in_place"] = sorted(u.get("unreadable_in_place", {}))
    planned = u["planned"]
    gone = []
    for rel, want in planned.items():
        f = by_rel.get(rel)
        prior = (prior_digests or {}).get(rel)
        if prior and prior["size"] == want and (f is None or f["size"] == 0):
            gone.append(rel)
        elif f is None:
            reasons.append(f"planned file missing: {rel}")
        elif f["size"] != want:
            reasons.append(f"planned file changed size: {rel} {f['size']} != {want}")
    if u["mode"] == "tree" and set(by_rel) != set(planned):
        extra = sorted(set(by_rel) - set(planned))
        if extra:
            reasons.append(f"files appeared since the plan: {extra[:5]}")
    checked["resume_already_gone_or_truncated"] = gone
    newest = max([f["mtime"] for f in files] + [os.lstat(d).st_mtime for d in dirs])
    if newest >= cutoff:
        reasons.append(f"modified after cutoff (newest {dt.datetime.fromtimestamp(newest, dt.timezone.utc).isoformat()})")
    checked.update(files=len(files), dirs=len(dirs), newest_mtime=newest)
    return {"reasons": reasons, "checked": checked, "files": files, "dirs": dirs}


def archive(u: dict, m: dict, rec_dir: str, deleted: list[str], kept: list[str]) -> dict:
    unit = u["path"]
    os.makedirs(rec_dir, exist_ok=True)
    tar_path = os.path.join(rec_dir, "tree.tar.gz")
    copied = sorted(rel for rel, d in m["digests"].items()
                    if "symlink" not in d and not CE.is_weight(rel) and d["size"] < SMALL and "sha256" in d)
    tmp = tar_path + ".partial"
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
        for rel in copied:
            tf.add(os.path.join(unit, rel), arcname=rel, recursive=False)
    seen = 0
    with tarfile.open(tmp, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            h = hashlib.sha256()
            fh = tf.extractfile(member)
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
            if h.hexdigest() != m["digests"][member.name]["sha256"]:
                raise RuntimeError(f"recipe tar member mismatch {member.name}")
            seen += 1
    if seen != len(copied):
        raise RuntimeError(f"recipe tar holds {seen} files, expected {len(copied)}")
    os.replace(tmp, tar_path)
    dset, kset = set(deleted), set(kept)
    with open(os.path.join(rec_dir, "MANIFEST.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("relpath\tsize\tmtime_utc\tdigest_kind\tdigest\tarchived\tkept_in_place\tdeleted\n")
        for rel in sorted(m["digests"]):
            d = m["digests"][rel]
            kind = next((k for k in ("sha256", "header_sha256", "head64m_sha256", "symlink", "unreadable") if k in d), "none")
            fh.write(f"{rel}\t{d['size']}\t{d['mtime']}\t{kind}\t{d.get(kind, '')}\t{'yes' if rel in copied else 'no'}\t"
                     f"{'yes' if rel in kset else 'no'}\t{'yes' if rel in dset else 'no'}\n")
    with open(os.path.join(rec_dir, "REBUILD.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {u['group']} : {unit}\n\nWhat: {u['what']}\n\nReason: {u['reason']}\n\nEvidence:\n{u['evidence']}\n\n"
                 f"Rebuild:\n{u['rebuild']}\n")
    return {"recipe_files": copied, "manifest_sha256": CE.sha256_file(os.path.join(rec_dir, "MANIFEST.tsv")),
            "tar_sha256": CE.sha256_file(tar_path)}


def record(conn, db, C, u: dict, m: dict, pre: dict, arch: dict, rec_dir: str, df_before: dict, deleted_bytes: int,
           owner_order: str = OWNER_ORDER, reason_prefix: str = "NAS cleanup 2026-10-28") -> tuple[int, list[int]]:
    unit = u["path"]
    reason = f"{reason_prefix} ({u['group']}): {u['reason']}"
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM host WHERE name = 'ai'")
        host_id = cur.fetchone()[0]
        cur.execute("SELECT id FROM model WHERE rtrim(path, '/') = %s ORDER BY id LIMIT 1", (unit,))
        row = cur.fetchone()
        unit_id = row[0] if row else db.upsert_model(
            conn, C.sha16_path_key(unit), u["project"], u["role"], m["layout"], path=unit, host_id=host_id,
            parameter_count=m["parameter_count"], size_bytes=deleted_bytes,
            config={"cleanup_unit": {"group": u["group"], "mode": u["mode"], "listing_sha256": m["listing_sha256"]}},
            status="present", notes=f"unit row created by {TOOL} before deletion")
        cur.execute("SELECT id FROM model WHERE rtrim(path, '/') = %s OR left(path, %s) = %s",
                    (unit, len(unit) + 1, unit + "/"))
        affected = sorted({r[0] for r in cur.fetchall()} | {unit_id})
        cur.execute("UPDATE model SET status = 'deleting', delete_reason = %s, rebuild_recipe_path = %s WHERE id = ANY(%s)",
                    (reason, rec_dir, affected))
        cur.execute(
            "INSERT INTO model_deletion (model_id, host_id, unit_path, plan_wave, plan_index, status, size_bytes_du, weight_bytes, "
            "file_count, layout, parameter_count, digest_mode, listing_sha256, file_digests, recipe_archive_path, "
            "recipe_archive_sha256, recipe_files, affected_model_ids, reason, owner_order, preflight, df_before) "
            "VALUES (%s,%s,%s,%s,%s,'deleting',%s,%s,%s,%s,%s,'index+headers+listing',%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s,"
            "%s::jsonb,%s::jsonb) "
            # a re-run after a failed delete refreshes that row; a 'deleted' row is never reopened (no row returned)
            "ON CONFLICT (host_id, unit_path) DO UPDATE SET status = 'deleting', file_digests = EXCLUDED.file_digests, "
            "listing_sha256 = EXCLUDED.listing_sha256, preflight = EXCLUDED.preflight, df_before = EXCLUDED.df_before, "
            "recipe_files = EXCLUDED.recipe_files, recipe_archive_sha256 = EXCLUDED.recipe_archive_sha256, "
            "size_bytes_du = EXCLUDED.size_bytes_du, error = NULL WHERE model_deletion.status <> 'deleted' RETURNING id",
            (unit_id, host_id, unit, u["wave"], u["plan_index"], deleted_bytes, m["weight_bytes"], len(m["digests"]),
             m["layout"], m["parameter_count"], m["listing_sha256"], db.canonical_json(m["digests"]), rec_dir,
             arch["manifest_sha256"], db.canonical_json(arch["recipe_files"]), affected, reason, owner_order,
             db.canonical_json(pre), db.canonical_json(df_before)))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"model_deletion row for {unit} is already 'deleted'; refusing to reopen it")
        del_id = row[0]
    conn.commit()
    return del_id, affected


RECLAIM_WAIT = 180


def reclaim_check(size: int, free_before: int, fh_log, p: str, volume: str = "nas") -> None:
    """The share is NFS-backed ZFS behind Samba: a truncate must actually return space (no snapshot pinning)."""
    t0 = time.time()
    while time.time() - t0 < RECLAIM_WAIT:
        gained = CE.df(volume)["free_bytes"] - free_before
        if gained >= size // 2:
            CE.log(fh_log, step="reclaim-verified", rel=p, size=size, gained=gained, seconds=round(time.time() - t0, 1))
            return
        time.sleep(10)
    raise RuntimeError(f"space not reclaimed {RECLAIM_WAIT}s after truncating {p} ({size} B): snapshot pinning? stopping")


def unlink_or_truncate(p: str, fh_log, state: dict, volume: str = "nas") -> None:
    """unlink; on ENOSPC (a 100%-full ZFS dataset cannot record the remove) truncate to 0 first, then unlink.
    The first truncation of a run is followed by reclaim_check."""
    try:
        os.unlink(p)
        return
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
    size = os.stat(p).st_size
    free_before = CE.df(volume)["free_bytes"]
    try:
        os.truncate(p, 0)
    except OSError as exc:
        # seen 2026-10-28: the CIFS->Samba->NFS->ZFS chain reported ENOSPC for a truncate that did take effect
        if exc.errno != errno.ENOSPC or os.stat(p).st_size != 0:
            raise
    CE.log(fh_log, step="truncated-enospc", rel=p, size=size)
    if not state.get("reclaim_verified"):
        reclaim_check(size, free_before, fh_log, p, volume)
        state["reclaim_verified"] = True
    os.unlink(p)


def delete(u: dict, deleted: list[str], fh_log, state: dict, resume: bool = False) -> None:
    unit = u["path"]
    if u["mode"] == "tree":
        r = subprocess.run(["rm", "-rf", "--one-file-system", "--", unit], capture_output=True, text=True)
        if os.path.lexists(unit):
            raise RuntimeError(f"rm failed rc={r.returncode}: {r.stderr[-400:]}")
        return
    for rel in deleted:
        p = os.path.join(unit, rel)
        if resume and not os.path.lexists(p):
            CE.log(fh_log, unit=unit, step="absent-already-resume", rel=rel)
            continue
        if os.path.islink(p) or not os.path.isfile(p):
            raise RuntimeError(f"not a regular file at delete time: {p}")
        unlink_or_truncate(p, fh_log, state, u.get("volume", "nas"))
        CE.log(fh_log, unit=unit, step="unlinked", rel=rel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--protected", required=True)
    ap.add_argument("--generic", required=True)
    ap.add_argument("--symlinks", action="append", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--recipe-root", default=CE.RECIPE_ROOT)
    ap.add_argument("--cutoff", default="2026-10-27T00:00:00Z")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import omni_lab_db as db
    from ingest import common as C
    plan = json.load(open(args.plan, encoding="utf-8"))
    units = [u for u in plan["units"] if u["group"] == args.group]
    for u in units:
        for path, why in u.get("protected_waivers", {}).items():
            if not isinstance(why, str) or len(why.strip()) < 20:
                raise ValueError(f"waiver {u['path']} / {path} has no written reason")
    if not units:
        print(f"no units for group {args.group}")
        return 1
    owner_order = plan.get("owner_order", OWNER_ORDER)
    reason_prefix = plan.get("reason_prefix", "NAS cleanup 2026-10-28")
    dead_prefixes = plan.get("dead_link_prefixes", {})
    protected = json.load(open(args.protected, encoding="utf-8"))
    generic = set(json.load(open(args.generic, encoding="utf-8"))) | set(plan.get("extra_generic", []))
    links = load_links(args.symlinks)
    cutoff = dt.datetime.strptime(args.cutoff, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    os.makedirs(args.run_dir, exist_ok=True)
    fh = open(os.path.join(args.run_dir, "exec_nas_groups.jsonl"), "a", encoding="utf-8")
    conn = db.connect("rw", app="nas-group-delete", statement_timeout_ms=120000)
    refs = CE.RefIndex()
    state: dict = {}
    CE.log(fh, step="start", group=args.group, units=len(units), dry_run=args.dry_run, tool=TOOL, pid=os.getpid())
    for u in units:
        unit = u["path"]
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT d.id, d.status, d.file_digests, d.listing_sha256, d.recipe_archive_path, "
                            "d.recipe_archive_sha256, d.affected_model_ids, d.size_bytes_du FROM model_deletion d "
                            "JOIN host h ON h.id = d.host_id WHERE h.name = 'ai' AND d.unit_path = %s", (unit,))
                prior = cur.fetchone()
            conn.rollback()
            if prior and prior[1] == "deleted":
                print(f"STOP: {unit} already has a 'deleted' model_deletion row {prior[0]} (surprise)")
                return 1
            g = gate(u, protected, generic, refs, links, dead_prefixes, conn, cutoff, prior[2] if prior else None)
            if g.get("absent"):
                CE.log(fh, unit=unit, step="absent-already")
                print(f"STOP: {unit} is already absent (surprise)")
                return 1
            if g["reasons"]:
                CE.log(fh, unit=unit, step="BLOCKED", reasons=g["reasons"])
                print("BLOCKED " + unit + "\n  " + "\n  ".join(r[:600] for r in g["reasons"]))
                return 1
            all_rel = [f["rel"] for f in g["files"]]
            deleted = sorted(all_rel) if u["mode"] == "tree" else sorted(u["planned"])
            kept = [] if u["mode"] == "tree" else sorted(set(all_rel) - set(deleted))
            deleted_bytes = sum(f["size"] for f in g["files"] if f["rel"] in set(deleted))
            pre = {**g["checked"], "mode": u["mode"], "kept_in_place": kept, "deleted_files": len(deleted),
                   "deleted_bytes": deleted_bytes, "reasons": [], "group": u["group"],
                   "plan_sha256": CE.sha256_file(args.plan), "protected_sha256": CE.sha256_file(args.protected),
                   "protected_entries": len(protected), "evidence": u["evidence"]}
            CE.log(fh, unit=unit, step="gate-clear", files=len(all_rel), deleted=len(deleted), kept=len(kept),
                   deleted_bytes=deleted_bytes, dead_links=len(g["checked"]["inbound_symlinks"]["dead_tree_breadcrumbs"]))
            if args.dry_run:
                print(f"CLEAR {unit}: delete {len(deleted)} files {deleted_bytes:,} B, keep {len(kept)}")
                continue
            if prior:
                # resume a failed attempt: the record made before any byte was removed stays the truth
                del_id, _st, _dig, listing, rec_dir, rec_sha, affected, deleted_bytes = prior
                if CE.sha256_file(os.path.join(rec_dir, "MANIFEST.tsv")) != rec_sha:
                    raise RuntimeError(f"resume: MANIFEST.tsv of {rec_dir} does not match row {del_id}")
                if json.load(open(os.path.join(rec_dir, "DELETION_RECEIPT.json"), encoding="utf-8"))["listing_sha256"] != listing:
                    raise RuntimeError(f"resume: receipt listing of {rec_dir} does not match row {del_id}")
                df_before = CE.df(u.get("volume", "nas"))
                with conn.cursor() as cur:
                    cur.execute("UPDATE model_deletion SET status = 'deleting', error = NULL, preflight = preflight || "
                                "jsonb_build_object('resumed', %s::jsonb) WHERE id = %s AND status <> 'deleted' RETURNING id",
                                (db.canonical_json({"at": CE.now(), "tool": TOOL, "gate": g["checked"], "df": df_before}), del_id))
                    if cur.fetchone() is None:
                        raise RuntimeError(f"resume: row {del_id} could not be reopened")
                conn.commit()
                CE.log(fh, unit=unit, step="resumed", model_deletion_id=del_id, recipe=rec_dir,
                       already_gone=g["checked"]["resume_already_gone_or_truncated"])
            else:
                m = measure_unit(u, g["files"], fh)
                rec_dir = os.path.join(args.recipe_root, f"{C.sha16_path_key(unit).replace(':', '-')}__nas20261028-{u['slug']}")
                arch = archive(u, m, rec_dir, deleted, kept)
                df_before = CE.df(u.get("volume", "nas"))
                receipt = {"schema": "omni-lab.deletion-receipt.v1", "tool": TOOL, "owner_order": owner_order,
                           "unit": {k: u[k] for k in ("path", "group", "mode", "what", "reason", "plan_index")},
                           "wave": u["wave"], "digest_mode": "index+headers+listing", "listing_sha256": m["listing_sha256"],
                           "layout": m["layout"], "parameter_count": m["parameter_count"], "weight_bytes": m["weight_bytes"],
                           "file_count": len(m["digests"]), "deleted": deleted, "kept_in_place": kept,
                           "deleted_bytes": deleted_bytes, "recipe_files": arch["recipe_files"],
                           "manifest_tsv_sha256": arch["manifest_sha256"],
                           "recipe_tar": {"path": os.path.join(rec_dir, "tree.tar.gz"), "sha256": arch["tar_sha256"]},
                           "preflight": pre, "df_before": df_before, "rebuild": u["rebuild"]}
                with open(os.path.join(rec_dir, "DELETION_RECEIPT.json"), "w", encoding="utf-8") as rf:
                    json.dump(receipt, rf, indent=1, sort_keys=True)
                del_id, affected = record(conn, db, C, u, m, pre, arch, rec_dir, df_before, deleted_bytes, owner_order,
                                          reason_prefix)
                CE.log(fh, unit=unit, step="recorded", model_deletion_id=del_id, affected_models=affected, recipe=rec_dir)
            t0 = time.time()
            try:
                delete(u, deleted, fh, state, resume=bool(prior))
            except Exception as exc:
                with conn.cursor() as cur:
                    cur.execute("UPDATE model_deletion SET status = 'failed', error = %s WHERE id = %s", (str(exc)[:2000], del_id))
                conn.commit()
                raise
            left = [r for r in deleted if os.path.lexists(os.path.join(unit, r))] if u["mode"] == "files" else (
                [unit] if os.path.lexists(unit) else [])
            if left:
                with conn.cursor() as cur:
                    cur.execute("UPDATE model_deletion SET status = 'failed', error = %s WHERE id = %s",
                                (f"still present after delete: {left[:5]}", del_id))
                conn.commit()
                raise RuntimeError(f"still present after delete: {left[:5]}")
            df_after = CE.df(u.get("volume", "nas"))
            with conn.cursor() as cur:
                cur.execute("UPDATE model_deletion SET status = 'deleted', executed_at = to_timestamp(%s), "
                            "verified_absent_at = now(), df_after = %s::jsonb WHERE id = %s",
                            (t0, db.canonical_json(df_after), del_id))
                cur.execute("UPDATE model SET status = 'deleted', deleted_at = now() WHERE id = ANY(%s)", (affected,))
            conn.commit()
            CE.log(fh, unit=unit, step="DELETED", deleted_bytes=deleted_bytes, model_deletion_id=del_id,
                   free_delta=df_after["free_bytes"] - df_before["free_bytes"], seconds_rm=round(time.time() - t0, 1))
            print(f"DELETED {unit}: {len(deleted)} files {deleted_bytes:,} B, model_deletion {del_id}, recipe {rec_dir}")
        except Exception as exc:
            CE.log(fh, unit=unit, step="FAILED", error=str(exc)[:2000], trace=traceback.format_exc()[-3000:])
            conn.rollback()
            print(f"FAILED {unit}: {exc}")
            return 1
    CE.log(fh, step="end", group=args.group)
    return 0


if __name__ == "__main__":
    sys.exit(main())
