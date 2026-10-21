#!/usr/bin/env python3
"""cleanup_exec.py 1.2.0 - remove experimental weight directories one unit at a time, recording each first.

Owner order (Cyril, 2026-10-21): "[Cyril quote withheld]".

Plan: neural-forge/reports/omni-campaign-2026-09/CLEANUP_PLAN_2026-10-21.md + the lead review that moved 27
units out of DELETE. The approved list is a JSON array of plan units (cleanup_plan.json server_delete rows plus
"plan_index"); this tool never chooses what to delete, it only refuses.

Per unit, in order (a failure stops the worker; a refusal skips the unit and is reported as BLOCKED):
  1. gate      - nothing under the unit is open by any process (root /proc scan); no live reference to the unit,
                 to anything inside it, or to a non-generic parent of it, in crontabs, process cmdlines/environ,
                 queue/tool scripts, queue JSONs, Lexar distill JSONs or any JSON of the K3 runtime folder
                 ($VARS expanded); no protected path (keep/hold/unsure/excluded) inside or around it; no
                 non-weight file >= 50 MB, no corpus/holdouts/receipts dir, no teacher-cache manifest, nothing
                 modified in the last 24 h; the path is a real directory (no symlinked component)
  2. measure   - every file: size, mtime; sha256 of every file < 1 GB (all files when the unit is < 20 GB);
                 safetensors >= 1 GB: header sha256; other files >= 1 GB: sha256 of the first 64 MiB;
                 parameter count from safetensors headers / GGUF tensor infos; listing sha256
  3. archive   - every non-weight file into <recipe>/tree.tar.gz (members re-hashed), plus MANIFEST.tsv and
                 DELETION_RECEIPT.json (1.0.0 wrote loose files to <recipe>/tree/; converted by recipe_pack.py)
  4. record    - one transaction: unit model row (created if absent), every model row inside the unit set to
                 status 'deleting' with reason + recipe path, one model_deletion row (status 'deleting')
  5. ledger    - for registered checkpoints: oc_checkpoint.py set-status --cyril-ruling (status unchanged)
  6. delete    - rm -rf --one-file-system -- <path> (argument list, no shell); git-tracked files are left in place
  7. verify    - path absent; df recorded; rows -> 'deleted', model_deletion -> 'deleted'

    PY=/home/cereal/.venvs/chimera-intake/bin/python
    set -a; . ~/.config/omni-lab/env; set +a
    $PY cleanup_exec.py gate    --units approved.json --protected protected.json --generic generic_roots.json --out g.json
    $PY cleanup_exec.py execute --units approved.json --protected protected.json --generic generic_roots.json \
        --volume home --run-dir /mnt/lexar/omni-campaign/archive/cleanup-2026-10-21 [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TOOL = "cleanup_exec 1.2.0"
OWNER_ORDER = "[Cyril quote withheld]"
OC = "/home/cereal/omni-campaign"
LX = "/mnt/lexar/omni-campaign"
RECIPE_ROOT = f"{LX}/archive/recipes"
LEDGER_TOOL = f"{OC}/ledger/oc_checkpoint.py"
K3_RUNTIME = f"{OC}/runtime/k3-nas-window61-72-oct07-a151e1a8-39b6-486e-a4aa-124865332425"
VOLUME_MOUNT = {"home": "/", "lexar": "/mnt/lexar", "nas": "/mnt/ai_data"}
WAVE = {"nas": "W1-NAS", "home": "W2-HOME", "lexar": "W3-LEXAR"}
WEIGHT_EXT = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".npz", ".npy", ".onnx")
DATA_DIR_NAMES = {"corpus", "holdouts", "receipts"}
BIG_NON_WEIGHT = 50 * 1024 ** 2
FULL_UNIT_LIMIT = 20 * 10 ** 9
FULL_FILE_LIMIT = 1024 ** 3
HEAD_BYTES = 64 * 1024 ** 2
RECENT_SECONDS = 24 * 3600
GLOBAL_VARS = {"OC": [OC], "OC_ROOT": [OC], "L": [LX], "LX": [LX], "LEXAR": [LX], "HOME": ["/home/cereal"],
               "T": [f"{OC}/distill/tools"], "TOOLS": [f"{OC}/distill/tools"], "Q": [f"{OC}/distill/queues"],
               "NAS": ["/mnt/ai_data"], "NF": ["/home/cereal/model-surgery-lab"]}
PATH_RE = re.compile(r"/(?:home|mnt)/[^\s\"'`;|<>(){}\[\],\\]*")
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-=?+])([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
ASSIGN_RE = re.compile(r"^\s*(?:export\s+|local\s+|readonly\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|[^\s;#]*)",
                       re.M)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(fh, **ev) -> None:
    ev = {"ts": now(), **ev}
    line = json.dumps(ev, sort_keys=True, default=str)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def norm(p: str) -> str:
    p = p.strip().rstrip("/.")
    while "//" in p:
        p = p.replace("//", "/")
    # a glob or a $VAR left unexpanded: keep the literal directory part before it
    for mark in ("*", "?", "{"):
        if mark in p:
            p = p.split(mark, 1)[0].rsplit("/", 1)[0]
    return p.rstrip("/") or "/"


def under(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent + "/")


# ---------------------------------------------------------------- references
def expand(text: str, local: dict[str, list[str]], depth: int = 0) -> list[str]:
    """Every expansion of $VAR / ${VAR:-d} in text, using file-local assignments then the globals (max 16)."""
    if "$" not in text or depth > 4:
        return [text]
    parts, choices = [], []
    pos = 0
    for m in VAR_RE.finditer(text):
        parts.append(text[pos:m.start()])
        name = m.group(1) or m.group(4)
        op, default = m.group(2), m.group(3)
        vals = list(dict.fromkeys(local.get(name, []) + GLOBAL_VARS.get(name, [])))
        if op and op.lstrip(":") in ("-", "=") and default is not None:
            vals = vals + [default]
        choices.append(vals or [m.group(0)])
        pos = m.end()
    parts.append(text[pos:])
    out = []
    for combo in itertools.islice(itertools.product(*choices), 16):
        s = parts[0] + "".join(c + parts[i + 1] for i, c in enumerate(combo))
        out.extend(expand(s, local, depth + 1) if s != text else [s])
    return list(dict.fromkeys(out))[:16]


def tokens_from_text(text: str) -> set[str]:
    local: dict[str, list[str]] = {}
    for m in ASSIGN_RE.finditer(text):
        v = m.group(2).strip("\"'")
        if v and not v.startswith("$("):
            local.setdefault(m.group(1), []).append(v)
    out: set[str] = set()
    for line in text.splitlines():
        if "/home/" not in line and "/mnt/" not in line and "$" not in line and "~/" not in line:
            continue
        line = line.replace("~/", "/home/cereal/")
        for variant in expand(line, local) if "$" in line else [line]:
            for m in PATH_RE.finditer(variant):
                out.add(norm(m.group(0)))
    return out


class RefIndex:
    """Path tokens named by every static source; files are re-read only when their size/mtime changes."""

    def __init__(self):
        self.cache: dict[str, tuple[tuple[int, int], set[str]]] = {}

    def sources(self) -> list[str]:
        files: list[str] = []
        for pattern_dir, exts, recursive in ((f"{OC}/distill/tools", (".sh",), False),
                                             (f"{OC}/distill/queues", (".json",), False),
                                             (f"{LX}/distill", (".json",), True),
                                             (K3_RUNTIME, (".json",), True)):
            if not os.path.isdir(pattern_dir):
                continue
            if recursive:
                for root, dirs, names in os.walk(pattern_dir):
                    files.extend(os.path.join(root, n) for n in names if n.endswith(exts))
            else:
                files.extend(os.path.join(pattern_dir, n) for n in os.listdir(pattern_dir) if n.endswith(exts))
        files += ["/etc/crontab"] + [os.path.join("/etc/cron.d", n) for n in sorted(os.listdir("/etc/cron.d"))]
        return sorted(set(files))

    def file_tokens(self) -> list[tuple[str, str]]:
        out = []
        for f in self.sources():
            try:
                st = os.stat(f)
            except OSError:
                continue
            key = (st.st_size, st.st_mtime_ns)
            hit = self.cache.get(f)
            if not hit or hit[0] != key:
                try:
                    with open(f, "r", encoding="utf-8", errors="replace") as fh:
                        toks = tokens_from_text(fh.read())
                except OSError:
                    toks = set()
                self.cache[f] = (key, toks)
            out.extend((f, t) for t in self.cache[f][1])
        return out


def crontab_tokens() -> list[tuple[str, str]]:
    out = []
    for label, cmd in (("crontab:cereal", ["crontab", "-l"]), ("crontab:root", ["sudo", "-n", "crontab", "-l", "-u", "root"])):
        r = subprocess.run(cmd, capture_output=True, text=True)
        out.extend((label, t) for t in tokens_from_text(r.stdout))
    return out


def proc_scan() -> dict:
    r = subprocess.run(["sudo", "-n", "/usr/bin/python3", str(HERE / "proc_scan.py")], capture_output=True, text=True,
                       timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"proc_scan failed rc={r.returncode}: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def docker_mount_tokens() -> list[tuple[str, str]]:
    r = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True)
    ids = r.stdout.split()
    if not ids:
        return []
    r = subprocess.run(["docker", "inspect", "-f", "{{.Name}}|{{range .Mounts}}{{.Source}}|{{end}}"] + ids,
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        name, *srcs = line.split("|")
        out.extend((f"docker:{name}", norm(s)) for s in srcs if s.startswith("/"))
    return out


# ---------------------------------------------------------------- gate
def src_of(unit: str, rel: str) -> str:
    """Absolute path of a unit member; a single-file unit (a GGUF export) is its own only member."""
    return unit if os.path.isfile(unit) and rel == os.path.basename(unit) else os.path.join(unit, rel)


def walk_unit(unit: str) -> tuple[list[dict], list[str]]:
    """Files under the unit (no symlink following, one filesystem). Returns (files, dirs).
    A unit that is a regular file (e.g. QF-X00.gguf) yields itself as the only file."""
    st0 = os.lstat(unit)
    if stat.S_ISREG(st0.st_mode):
        return [{"rel": os.path.basename(unit), "size": st0.st_size, "mtime": st0.st_mtime}], []
    dev = st0.st_dev
    files, dirs = [], []
    stack = [unit]
    while stack:
        d = stack.pop()
        dirs.append(d)
        with os.scandir(d) as it:
            for e in it:
                st = e.stat(follow_symlinks=False)
                rel = os.path.relpath(e.path, unit)
                if stat.S_ISDIR(st.st_mode):
                    if st.st_dev != dev:
                        raise RuntimeError(f"mount point inside unit: {e.path}")
                    stack.append(e.path)
                elif stat.S_ISLNK(st.st_mode):
                    files.append({"rel": rel, "size": 0, "mtime": st.st_mtime, "symlink": os.readlink(e.path)})
                else:
                    files.append({"rel": rel, "size": st.st_size, "mtime": st.st_mtime})
    return files, dirs


def is_weight(rel: str) -> bool:
    return rel.lower().endswith(WEIGHT_EXT)


def git_tracked(unit: str) -> list[str]:
    """Files inside the unit tracked by an enclosing git repository (outside the unit)."""
    d = os.path.dirname(unit)
    while d and d != "/":
        if os.path.exists(os.path.join(d, ".git")):
            rel = os.path.relpath(unit, d)
            r = subprocess.run(["git", "-C", d, "ls-files", "-z", "--", rel], capture_output=True)
            return [os.path.relpath(os.path.join(d, p.decode()), unit) for p in r.stdout.split(b"\0") if p]
        d = os.path.dirname(d)
    return []


def gate(u: dict, protected: list[dict], generic: set[str], refs: RefIndex, scan: dict | None = None) -> dict:
    """Every reason this unit must not be deleted now ([] = clear), plus what was checked."""
    unit = u["path"]
    reasons: list[str] = []
    checked: dict = {"at": now(), "tool": TOOL}
    if not os.path.lexists(unit):
        return {"reasons": ["path absent"], "checked": checked, "absent": True}
    if os.path.islink(unit) or not (os.path.isdir(unit) or os.path.isfile(unit)) or os.path.realpath(unit) != unit:
        reasons.append(f"not a plain directory/file or symlinked component (realpath {os.path.realpath(unit)})")
    for p in protected:
        if under(p["path"], unit) or under(unit, p["path"]):
            reasons.append(f"protected {p['class']} overlaps: {p['path']}")
    scan = scan or proc_scan()
    opened = [p for p in scan["open"] if under(p, unit)]
    if opened:
        reasons.append(f"open by a live process: {opened[:5]}")
    checked["proc_scan"] = {"open_paths": len(scan["open"]), "strings": len(scan["strings"]), "errors": scan["errors"]}
    token_srcs: list[tuple[str, str]] = []
    for pid, kind, text in scan["strings"]:
        for t in tokens_from_text(text):
            token_srcs.append((f"pid{pid}:{kind}", t))
    token_srcs += crontab_tokens() + docker_mount_tokens() + refs.file_tokens()
    hits, parent_hits = [], []
    for src, t in token_srcs:
        if under(t, unit):
            hits.append((src, t))
        elif unit.startswith(t + "/") and t not in generic:
            parent_hits.append((src, t))
    if hits:
        reasons.append(f"referenced (unit or inside): {sorted(set(hits))[:8]}")
    if parent_hits:
        reasons.append(f"referenced (non-generic parent): {sorted(set(parent_hits))[:8]}")
    checked["reference_tokens"] = len(token_srcs)
    checked["reference_files"] = len(refs.cache)
    try:
        files, dirs = walk_unit(unit)
    except Exception as exc:
        return {"reasons": reasons + [f"walk failed: {exc}"], "checked": checked}
    big = [f["rel"] for f in files if "symlink" not in f and not is_weight(f["rel"]) and f["size"] >= BIG_NON_WEIGHT]
    if big:
        reasons.append(f"non-weight file >= 50 MB (may be data): {big[:5]}")
    datadirs = [os.path.relpath(d, unit) for d in dirs if os.path.basename(d) in DATA_DIR_NAMES]
    if datadirs:
        reasons.append(f"data directory inside: {datadirs[:5]}")
    tc = []
    for f in files:
        if os.path.basename(f["rel"]) in ("manifest.json", "plan.json", "pass_receipt.partial") and f["size"] < 4 * 1024 ** 2:
            try:
                with open(src_of(unit, f["rel"]), "r", encoding="utf-8", errors="replace") as fh:
                    head = fh.read(65536)
                if "teacher_cache" in head or "teacher-cache" in head:
                    tc.append(f["rel"])
            except OSError:
                pass
    if tc:
        reasons.append(f"teacher-cache manifest inside: {tc[:5]}")
    newest = max([f["mtime"] for f in files] + [os.lstat(d).st_mtime for d in dirs])
    if time.time() - newest < RECENT_SECONDS:
        reasons.append(f"modified within 24 h (newest {dt.datetime.fromtimestamp(newest, dt.timezone.utc).isoformat()})")
    checked.update(files=len(files), dirs=len(dirs), newest_mtime=newest)
    return {"reasons": reasons, "checked": checked, "files": files, "dirs": dirs}


# ---------------------------------------------------------------- measure
def sha256_file(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    left = limit
    with open(path, "rb", buffering=0) as fh:
        while True:
            want = 8 * 1024 * 1024 if left is None else min(8 * 1024 * 1024, left)
            if want <= 0:
                break
            b = fh.read(want)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def safetensors_header(path: str) -> tuple[str, int]:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        n = struct.unpack("<Q", raw)[0]
        if n > 512 * 1024 ** 2:
            raise ValueError(f"implausible safetensors header length {n}")
        body = fh.read(n)
    hdr = json.loads(body)
    numel = 0
    for k, v in hdr.items():
        if k != "__metadata__":
            c = 1
            for d in v.get("shape", []):
                c *= int(d)
            numel += c
    return hashlib.sha256(raw + body).hexdigest(), numel


def gguf_params(path: str) -> int | None:
    def rd(fh, fmt):
        return struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))[0]

    def rstr(fh):
        return fh.read(rd(fh, "Q"))

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(fh, t):
        if t in sizes:
            fh.seek(sizes[t], 1)
        elif t == 8:
            rstr(fh)
        elif t == 9:
            et, n = rd(fh, "I"), rd(fh, "Q")
            if et in sizes:
                fh.seek(sizes[et] * n, 1)
            else:
                for _ in range(n):
                    skip(fh, et)
        else:
            raise ValueError(f"gguf type {t}")

    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            return None
        if rd(fh, "I") < 2:
            return None
        n_t, n_kv = rd(fh, "Q"), rd(fh, "Q")
        for _ in range(n_kv):
            rstr(fh)
            skip(fh, rd(fh, "I"))
        total = 0
        for _ in range(n_t):
            rstr(fh)
            nd = rd(fh, "I")
            c = 1
            for _ in range(nd):
                c *= rd(fh, "Q")
            rd(fh, "I")
            rd(fh, "Q")
            total += c
        return total


def measure(u: dict, files: list[dict], mode: str, fh_log, hash_threads: int = 1) -> dict:
    unit = u["path"]
    params, n_params_known = 0, False
    weight_bytes = 0
    digests: dict[str, dict] = {}
    t0 = time.time()
    hashed = 0
    jobs: list[tuple] = []
    for f in sorted(files, key=lambda x: x["rel"]):
        rel, p = f["rel"], src_of(unit, f["rel"])
        d = {"size": f["size"], "mtime": dt.datetime.fromtimestamp(f["mtime"], dt.timezone.utc).isoformat()}
        if "symlink" in f:
            d["symlink"] = f["symlink"]
            digests[rel] = d
            continue
        if is_weight(rel):
            weight_bytes += f["size"]
        if rel.endswith(".safetensors") and f["size"] >= 8:
            try:
                d["header_sha256"], numel = safetensors_header(p)
                params += numel
                n_params_known = True
            except Exception as exc:
                d["header_error"] = str(exc)[:200]
        elif rel.endswith(".gguf"):
            try:
                g = gguf_params(p)
                if g:
                    params += g
                    n_params_known = True
            except Exception as exc:
                d["gguf_error"] = str(exc)[:200]
        if mode == "full-sha256" or f["size"] < FULL_FILE_LIMIT:
            jobs.append((rel, p, "sha256", None))
            hashed += f["size"]
        elif not rel.endswith(".safetensors"):
            jobs.append((rel, p, "head64m_sha256", HEAD_BYTES))
            hashed += HEAD_BYTES
        digests[rel] = d
    # content hashes: in parallel when asked (the NAS share serves more in total to 2-3 readers than to one)
    if hash_threads > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=hash_threads) as pool:
            results = list(pool.map(lambda j: sha256_file(j[1], j[3]), jobs))
    else:
        results = [sha256_file(j[1], j[3]) for j in jobs]
    for (rel, _p, key, _lim), digest in zip(jobs, results):
        digests[rel][key] = digest
    lines = []
    for rel in sorted(digests):
        d = digests[rel]
        dig = d.get("sha256") or d.get("header_sha256") or d.get("head64m_sha256") or ("->" + d["symlink"] if "symlink" in d else "")
        lines.append(f"{rel}\t{d['size']}\t{dig}")
    listing = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    names = {os.path.basename(f["rel"]) for f in files if os.path.dirname(f["rel"]) == ""}
    if any(r.endswith(".gguf") for r in digests):
        layout = "gguf"
    elif "composite_manifest.json" in names or "student_manifest.json" in names or "chimera_manifest.json" in names:
        layout = "chimera.checkpoint.v1"
    elif "config.json" in names:
        layout = "hf"
    elif weight_bytes:
        layout = "safetensors-raw"
    else:
        layout = "other"
    log(fh_log, unit=unit, step="measured", files=len(files), hashed_bytes=hashed, seconds=round(time.time() - t0, 1))
    return {"digests": digests, "listing_sha256": listing, "listing_lines": lines, "weight_bytes": weight_bytes,
            "parameter_count": params if n_params_known else None, "layout": layout,
            "total_bytes": sum(f["size"] for f in files)}


# ---------------------------------------------------------------- archive
def archive(u: dict, m: dict, g: dict, rec_dir: str, tracked: list[str]) -> dict:
    """Non-weight files -> <rec_dir>/tree.tar.gz (one file: Lexar is exFAT with 1 MiB clusters, so loose small
    files cost 1 MiB each), every member re-read and re-hashed against the measured sha256; plus MANIFEST.tsv."""
    import tarfile
    unit = u["path"]
    os.makedirs(rec_dir, exist_ok=True)
    tar_path = os.path.join(rec_dir, "tree.tar.gz")
    copied = sorted(rel for rel, d in m["digests"].items() if "symlink" not in d and not is_weight(rel))
    tmp = tar_path + ".partial"
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
        for rel in copied:
            tf.add(src_of(unit, rel), arcname=rel, recursive=False)
    with tarfile.open(tmp, "r:gz") as tf:
        seen = 0
        for member in tf:
            if not member.isfile():
                continue
            h = hashlib.sha256()
            fh = tf.extractfile(member)
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
            want = m["digests"][member.name]["sha256"]
            if h.hexdigest() != want:
                raise RuntimeError(f"recipe tar member mismatch {member.name}: {h.hexdigest()} != {want}")
            seen += 1
    if seen != len(copied):
        raise RuntimeError(f"recipe tar holds {seen} files, expected {len(copied)}")
    os.replace(tmp, tar_path)
    with open(os.path.join(rec_dir, "MANIFEST.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("relpath\tsize\tmtime_utc\tdigest_kind\tdigest\tarchived\tgit_tracked_kept\n")
        for rel in sorted(m["digests"]):
            d = m["digests"][rel]
            kind = next((k for k in ("sha256", "header_sha256", "head64m_sha256", "symlink") if k in d), "none")
            fh.write(f"{rel}\t{d['size']}\t{d['mtime']}\t{kind}\t{d.get(kind, '')}\t"
                     f"{'yes' if rel in copied else 'no'}\t{'yes' if rel in tracked else 'no'}\n")
    man_sha = sha256_file(os.path.join(rec_dir, "MANIFEST.tsv"))
    return {"recipe_files": copied, "manifest_sha256": man_sha, "tar_sha256": sha256_file(tar_path)}


# ---------------------------------------------------------------- database
def project_for(path: str) -> str:
    if any(k in path for k in ("neural-forge", "model-surgery-lab", "/nf-")):
        return "neural-forge"
    if re.search(r"/k3[-_/]|/k3$", path):
        return "k3"
    return "chimera-omni"


ROLE_FOR_WHAT = {"model": "fold-child", "media-unit": "media-unit", "delta": "delta"}


def record(conn, db, C, u: dict, m: dict, g: dict, arch: dict, rec_dir: str, wave: str, df_before: dict) -> tuple[int, list[int]]:
    unit = u["path"]
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM host WHERE name = 'ai'")
        host_id = cur.fetchone()[0]
        cur.execute("SELECT id FROM model WHERE rtrim(path, '/') = %s ORDER BY (ledger_checkpoint_id IS NOT NULL) DESC, id "
                    "LIMIT 1", (unit,))
        row = cur.fetchone()
        reason = f"cleanup 2026-10-21 ({u['what']}): {u['reason']}"
        if row:
            unit_id = row[0]
        else:
            unit_id = db.upsert_model(
                conn, C.sha16_path_key(unit), project_for(unit), ROLE_FOR_WHAT.get(u["what"], "intermediate"), m["layout"],
                path=unit, host_id=host_id, parameter_count=m["parameter_count"], size_bytes=m["weight_bytes"],
                config={"cleanup_unit": {"what": u["what"], "plan_index": u["plan_index"], "du_bytes": u["size_bytes_du"],
                                         "listing_sha256": m["listing_sha256"]}},
                status="present", notes=f"unit row created by {TOOL} before deletion")
        cur.execute("SELECT id FROM model WHERE rtrim(path, '/') = %s OR left(path, %s) = %s",
                    (unit, len(unit) + 1, unit + "/"))
        affected = sorted({r[0] for r in cur.fetchall()} | {unit_id})
        cur.execute("UPDATE model SET status = 'deleting', delete_reason = %s, rebuild_recipe_path = %s, "
                    "parameter_count = COALESCE(parameter_count, %s) WHERE id = ANY(%s)",
                    (reason, rec_dir, m["parameter_count"] if len(affected) == 1 else None, affected))
        cur.execute("SELECT DISTINCT id FROM run WHERE model_id = ANY(%s) UNION "
                    "SELECT DISTINCT run_id FROM run_model WHERE model_id = ANY(%s) ORDER BY 1", (affected, affected))
        runs = [r[0] for r in cur.fetchall()]
        ledger_ids = u["omni_lab"].get("ledger_checkpoint_ids") or []
        cur.execute(
            "INSERT INTO model_deletion (model_id, host_id, unit_path, plan_wave, plan_index, status, size_bytes_du, weight_bytes, "
            "file_count, layout, parameter_count, digest_mode, listing_sha256, file_digests, recipe_archive_path, "
            "recipe_archive_sha256, recipe_files, affected_model_ids, producing_run_ids, ledger_checkpoint_id, reason, "
            "owner_order, preflight, df_before) VALUES (%s,%s,%s,%s,%s,'deleting',%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,"
            "%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb) "
            "ON CONFLICT (host_id, unit_path) DO UPDATE SET status = 'deleting', file_digests = EXCLUDED.file_digests, "
            "listing_sha256 = EXCLUDED.listing_sha256, preflight = EXCLUDED.preflight, df_before = EXCLUDED.df_before, "
            "recipe_files = EXCLUDED.recipe_files, recipe_archive_sha256 = EXCLUDED.recipe_archive_sha256, error = NULL "
            "RETURNING id",
            (unit_id, host_id, unit, wave, u["plan_index"], u["size_bytes_du"], m["weight_bytes"], len(m["digests"]),
             m["layout"], m["parameter_count"], u["digest_mode_used"], m["listing_sha256"], db.canonical_json(m["digests"]),
             rec_dir, arch["manifest_sha256"], db.canonical_json(arch["recipe_files"]), affected, runs or None,
             ",".join(ledger_ids) or None, reason, OWNER_ORDER, db.canonical_json(g["preflight"]), db.canonical_json(df_before)))
        del_id = cur.fetchone()[0]
    conn.commit()
    return del_id, affected


def df(volume: str) -> dict:
    s = os.statvfs(VOLUME_MOUNT[volume])
    return {"mount": VOLUME_MOUNT[volume], "free_bytes": s.f_bavail * s.f_frsize, "size_bytes": s.f_blocks * s.f_frsize}


# ---------------------------------------------------------------- ledger
def ledger_note(cid: str, u: dict, rec_dir: str, del_id: int) -> str:
    r = subprocess.run(
        ["/usr/bin/python3", LEDGER_TOOL, "set-status", "--checkpoint-id", cid, "--cyril-ruling", OWNER_ORDER,
         "--notes", f"weights deleted by {TOOL} on {now()[:10]} (cleanup wave {WAVE[u['volume']]}); directory "
                    f"{u['path']}; reason: {u['reason']}; recipe archive {rec_dir}; omni_lab model_deletion id {del_id}; "
                    f"status left unchanged (record only)", "--confirm"],
        capture_output=True, text=True, cwd=OC, env={**os.environ, "OC_ROOT": OC})
    if r.returncode != 0:
        raise RuntimeError(f"oc_checkpoint set-status {cid} failed rc={r.returncode}: {r.stderr[-800:]} {r.stdout[-400:]}")
    m = re.search(r"^sealed (.+)$", r.stdout, re.M)
    if not m:
        raise RuntimeError(f"oc_checkpoint set-status {cid}: no 'sealed' line: {r.stdout[-400:]}")
    sealed = m.group(1).strip()
    if not os.path.isabs(sealed):
        sealed = os.path.join(OC, sealed)
    with open(sealed, "r", encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec.get("checkpoint_id") != cid:
        raise RuntimeError(f"sealed record {sealed} is for {rec.get('checkpoint_id')}, not {cid}")
    return rec["receipt_sha"]


# ---------------------------------------------------------------- delete
def delete_unit(unit: str, tracked: list[str], fh_log) -> None:
    if not tracked or os.path.isfile(unit):
        if tracked:
            raise RuntimeError(f"single-file unit is git-tracked: {unit}")
        r = subprocess.run(["rm", "-rf", "--one-file-system", "--", unit], capture_output=True, text=True)
        if os.path.lexists(unit):
            log(fh_log, unit=unit, step="rm-as-user-incomplete", rc=r.returncode, stderr=r.stderr[-400:])
            r = subprocess.run(["sudo", "-n", "rm", "-rf", "--one-file-system", "--", unit], capture_output=True, text=True)
            if os.path.lexists(unit):
                raise RuntimeError(f"rm failed rc={r.returncode}: {r.stderr[-400:]}")
        return
    keep = {os.path.normpath(os.path.join(unit, t)) for t in tracked}
    keep_dirs = set()
    for k in keep:
        d = os.path.dirname(k)
        while under(d, unit):
            keep_dirs.add(d)
            d = os.path.dirname(d)
    for root, dirs, names in os.walk(unit, topdown=False):
        for n in names:
            p = os.path.join(root, n)
            if p not in keep:
                os.unlink(p)
        for n in dirs:
            p = os.path.join(root, n)
            if os.path.islink(p):
                os.unlink(p)
            elif p not in keep_dirs:
                os.rmdir(p)


# ---------------------------------------------------------------- driver
def load(args):
    units = json.load(open(args.units, encoding="utf-8"))
    protected = json.load(open(args.protected, encoding="utf-8"))
    generic = set(json.load(open(args.generic, encoding="utf-8")))
    return units, protected, generic


def cmd_gate(args) -> int:
    units, protected, generic = load(args)
    refs = RefIndex()
    scan = proc_scan()
    out = []
    for u in units:
        g = gate(u, protected, generic, refs, scan)
        out.append({"plan_index": u["plan_index"], "path": u["path"], "volume": u["volume"], "size_gb": u["size_gb"],
                    "reasons": g["reasons"], "checked": g["checked"]})
        print(f"{'CLEAR' if not g['reasons'] else 'BLOCK'} {u['plan_index']:>3} {u['size_gb']:>7} {u['path']}"
              + ("" if not g["reasons"] else "\n      " + "\n      ".join(r[:400] for r in g["reasons"])), flush=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"clear {sum(1 for o in out if not o['reasons'])} / {len(out)}")
    return 0


def cmd_execute(args) -> int:
    import omni_lab_db as db
    from ingest import common as C
    units, protected, generic = load(args)
    units = [u for u in units if u["volume"] == args.volume]
    units.sort(key=lambda u: -u["size_bytes_du"])
    if args.only:
        units = [u for u in units if u["plan_index"] in set(args.only)]
    if args.limit:
        units = units[:args.limit]
    os.makedirs(args.run_dir, exist_ok=True)
    fh = open(os.path.join(args.run_dir, f"exec_{args.volume}.jsonl"), "a", encoding="utf-8")
    wave = WAVE[args.volume]
    refs = RefIndex()
    conn = None if args.dry_run else db.connect("rw", app=f"cleanup-exec-{args.volume}", statement_timeout_ms=120000)
    log(fh, step="start", volume=args.volume, units=len(units), dry_run=args.dry_run, tool=TOOL, pid=os.getpid())
    done = blocked = 0
    for u in units:
        unit = u["path"]
        if os.path.exists(os.path.join(args.run_dir, f"STOP_{args.volume}")):
            log(fh, step="stop", volume=args.volume, deleted=done, blocked=blocked, why="STOP file present")
            return 0
        try:
            g = gate(u, protected, generic, refs)
            if g.get("absent"):
                log(fh, unit=unit, plan_index=u["plan_index"], step="absent-already")
                continue
            if g["reasons"]:
                blocked += 1
                log(fh, unit=unit, plan_index=u["plan_index"], step="BLOCKED", reasons=g["reasons"])
                continue
            tracked = git_tracked(unit)
            mode = "full-sha256" if u["size_bytes_du"] < FULL_UNIT_LIMIT else "index+headers+listing"
            u["digest_mode_used"] = mode
            g["preflight"] = {**g["checked"], "git_tracked_kept": tracked, "reasons": [],
                              "generic_roots_sha256": hashlib.sha256(json.dumps(sorted(generic)).encode()).hexdigest(),
                              "protected_entries": len(protected)}
            log(fh, unit=unit, plan_index=u["plan_index"], step="gate-clear", digest_mode=mode, files=g["checked"]["files"],
                git_tracked=len(tracked))
            if args.dry_run:
                continue
            m = measure(u, g["files"], mode, fh, args.hash_threads)
            ledger_ids = u["omni_lab"].get("ledger_checkpoint_ids") or []
            rec_dir = os.path.join(args.recipe_root, (ledger_ids[0] if len(ledger_ids) == 1 else
                                                 os.path.basename(u["recipe_archive"].rstrip("/"))))
            arch = archive(u, m, g, rec_dir, tracked)
            df_before = df(args.volume)
            receipt = {"schema": "omni-lab.deletion-receipt.v1", "tool": TOOL, "owner_order": OWNER_ORDER,
                       "unit": {k: u[k] for k in ("path", "volume", "what", "reason", "plan_index", "size_bytes_du")},
                       "wave": wave, "digest_mode": mode, "listing_sha256": m["listing_sha256"], "layout": m["layout"],
                       "parameter_count": m["parameter_count"], "weight_bytes": m["weight_bytes"],
                       "file_count": len(m["digests"]), "ledger_checkpoint_ids": ledger_ids,
                       "recipe_files": arch["recipe_files"], "manifest_tsv_sha256": arch["manifest_sha256"],
                       "recipe_tar": {"path": os.path.join(rec_dir, "tree.tar.gz"), "sha256": arch["tar_sha256"]},
                       "git_tracked_kept": tracked, "preflight": g["preflight"], "df_before": df_before,
                       "rebuild": {"recipe_dir": rec_dir, "ledger": f"{OC}/ledger.jsonl" if ledger_ids else None,
                                   "omni_lab": "model_deletion.unit_path = this path; model rows status deleted"}}
            with open(os.path.join(rec_dir, "DELETION_RECEIPT.json"), "w", encoding="utf-8") as rf:
                json.dump(receipt, rf, indent=1, sort_keys=True)
            del_id, affected = record(conn, db, C, u, m, g, arch, rec_dir, wave, df_before)
            log(fh, unit=unit, step="recorded", model_deletion_id=del_id, affected_models=affected, recipe=rec_dir)
            receipts = []
            for cid in ledger_ids:
                receipts.append(ledger_note(cid, u, rec_dir, del_id))
            if receipts:
                with conn.cursor() as cur:
                    cur.execute("UPDATE model_deletion SET ledger_receipt_sha = %s WHERE id = %s", (",".join(receipts), del_id))
                conn.commit()
                log(fh, unit=unit, step="ledger", checkpoints=ledger_ids, receipt_sha=receipts)
            t0 = time.time()
            try:
                delete_unit(unit, tracked, fh)
            except Exception as exc:
                with conn.cursor() as cur:
                    cur.execute("UPDATE model_deletion SET status = 'failed', error = %s WHERE id = %s", (str(exc)[:2000], del_id))
                conn.commit()
                raise
            gone = not os.path.lexists(unit) if not tracked else all(
                not os.path.lexists(src_of(unit, r)) for r in m["digests"] if r not in tracked)
            if not gone:
                with conn.cursor() as cur:
                    cur.execute("UPDATE model_deletion SET status = 'failed', error = 'path still present after rm' "
                                "WHERE id = %s", (del_id,))
                conn.commit()
                raise RuntimeError(f"still present after rm: {unit}")
            df_after = df(args.volume)
            with conn.cursor() as cur:
                cur.execute("UPDATE model_deletion SET status = 'deleted', executed_at = to_timestamp(%s), "
                            "verified_absent_at = now(), df_after = %s::jsonb WHERE id = %s",
                            (t0, db.canonical_json(df_after), del_id))
                cur.execute("UPDATE model SET status = 'deleted', deleted_at = now() WHERE id = ANY(%s)", (affected,))
            conn.commit()
            done += 1
            log(fh, unit=unit, plan_index=u["plan_index"], step="DELETED", du_bytes=u["size_bytes_du"],
                free_delta=df_after["free_bytes"] - df_before["free_bytes"], seconds_rm=round(time.time() - t0, 1),
                model_deletion_id=del_id)
        except Exception as exc:
            log(fh, unit=unit, plan_index=u.get("plan_index"), step="FAILED", error=str(exc)[:2000],
                trace=traceback.format_exc()[-3000:])
            if conn:
                conn.rollback()
            log(fh, step="stop", volume=args.volume, deleted=done, blocked=blocked)
            return 1
    log(fh, step="end", volume=args.volume, deleted=done, blocked=blocked)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "execute"):
        p = sub.add_parser(name)
        p.add_argument("--units", required=True)
        p.add_argument("--protected", required=True)
        p.add_argument("--generic", required=True)
    sub.choices["gate"].add_argument("--out", required=True)
    e = sub.choices["execute"]
    e.add_argument("--volume", required=True, choices=sorted(VOLUME_MOUNT))
    e.add_argument("--run-dir", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--only", type=int, nargs="*", default=None, help="plan indexes")
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--recipe-root", default=RECIPE_ROOT)
    e.add_argument("--hash-threads", type=int, default=1)
    args = ap.parse_args()
    return cmd_gate(args) if args.cmd == "gate" else cmd_execute(args)


if __name__ == "__main__":
    sys.exit(main())
