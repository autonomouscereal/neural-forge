#!/usr/bin/env python3
"""codex_inflight_adopt.py -- lane inflight: adopt Codex's finished Omni artifacts into Claude's tree, byte-verified.

Written 2026-10-14 by Claude, lane inflight (Cyril's ruling cyril-claude-leads-all-lanes-2026-10-14: Claude leads every
lane, Codex is out of usage).  CODEX_INFLIGHT_DISPOSITION_2026-10-14.md decides, per Codex work item, FINISH / ADOPT /
PARK / SUPERSEDED.  This tool executes the ADOPT items that matter for v1: it copies each one (Codex's source is never
modified, moved or deleted) to /mnt/lexar/omni-campaign/codex-adopted-2026-10-14/<name>/payload and proves the copy:

  * every pin in the plan (relpath -> sha256, e.g. a manifest sha Codex reported on Relay) must match the SOURCE before
    anything is copied -- a different sha is a refusal, not a warning;
  * kind "tree": every source file is hashed, copied, and the copy re-hashed; any difference, unreadable file, special
    file or symlink leaving the tree fails the item;
  * kind "tar": the archive's sha256 must equal its index's `archive_sha256`; members are extracted safely (regular
    files and dirs only, no absolute paths, no `..`) and every file listed in the index must match its bytes + sha256,
    with no extra or missing members;
  * manifests named in the plan are scanned for {"path", "sha256"} references; each one that resolves inside the copy
    (relative to the manifest, or through the plan's container-prefix map such as /recovery -> payload/...) must match.
    References that resolve outside the copy (parent model files, other Codex dirs) are counted as external, not
    checked here.

Gates (lane rules, STAGE_D_RECONCILED_PLAN_2026-10-14 section 7.4): CPU only (CUDA_VISIBLE_DEVICES='' is forced),
run it under nice 15 / ionice idle, single-threaded hashing, and MemAvailable >= --min-mem-gib (default 40) before every
item (it waits up to --mem-wait-s, then refuses).  Re-running is idempotent: an item whose ADOPTION.json says PASS is
skipped; any other existing destination is a refusal (nothing is ever overwritten).

Usage:
  python3 codex_inflight_adopt.py run --plan PLAN.json --dest /mnt/lexar/omni-campaign/codex-adopted-2026-10-14
  python3 codex_inflight_adopt.py verify-item --dir DEST/<name>     # re-hash a finished copy against its SHA256SUMS
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

TOOL_NAME = "codex_inflight_adopt"
TOOL_VERSION = "1.0.0"  # 2026-10-14: first version (lane inflight)

PLAN_SCHEMA = "omni-codex-adopt-plan-v1"
ADOPTION_SCHEMA = "omni-codex-adoption-v1"
RECEIPT_SCHEMA = "omni-codex-adopt-receipt-v1"
CHUNK = 1 << 22


class Refusal(Exception):
    """A pin, gate or integrity rule failed: the item is not adopted."""


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def mem_available_bytes(meminfo: str = "/proc/meminfo") -> int:
    with open(meminfo) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise Refusal("MemAvailable missing from " + meminfo)


def wait_for_memory(min_gib: float, wait_s: float, meminfo: str = "/proc/meminfo", poll_s: float = 30.0) -> int:
    need = int(min_gib * (1 << 30))
    deadline = time.monotonic() + wait_s
    while True:
        have = mem_available_bytes(meminfo)
        if have >= need:
            return have
        if time.monotonic() >= deadline:
            raise Refusal(f"MemAvailable {have / (1 << 30):.1f} GiB < gate {min_gib} GiB after waiting {wait_s:.0f} s")
        time.sleep(min(poll_s, max(1.0, deadline - time.monotonic())))


# ---------------------------------------------------------------------------------------------------------- trees


def walk_tree(root: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """(files relpath -> sha256, symlinks relpath -> target, problems).  Never follows symlinks."""
    files: dict[str, str] = {}
    links: dict[str, str] = {}
    problems: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=lambda e: problems.append(f"unreadable dir: {e.filename}")):
        base = Path(dirpath)
        for name in sorted(dirnames):
            p = base / name
            if p.is_symlink():
                links[p.relative_to(root).as_posix()] = os.readlink(p)
        for name in sorted(filenames):
            p = base / name
            rel = p.relative_to(root).as_posix()
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                links[rel] = os.readlink(p)
            elif stat.S_ISREG(st.st_mode):
                try:
                    files[rel] = sha256_file(p)
                except OSError as exc:
                    problems.append(f"unreadable file: {rel} ({exc.strerror})")
            else:
                problems.append(f"special file: {rel}")
    for rel, target in links.items():
        resolved = (root / rel).parent / target
        if os.path.isabs(target) or not os.path.normpath(resolved).startswith(os.path.normpath(root) + os.sep):
            problems.append(f"symlink leaves the tree: {rel} -> {target}")
    return files, links, problems


def copy_tree(src: Path, dst: Path, links: dict[str, str]) -> None:
    def ignore(dirpath: str, names: list[str]) -> list[str]:
        return [n for n in names if os.path.islink(os.path.join(dirpath, n))]

    shutil.copytree(src, dst, symlinks=True, ignore=ignore, copy_function=shutil.copy2)
    for rel, target in sorted(links.items()):
        os.symlink(target, dst / rel)


# ---------------------------------------------------------------------------------------------------------- tarballs


def safe_member_name(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or name.startswith("/") or "\\" in name:
        raise Refusal(f"unsafe tar member name {name!r}")
    return p.as_posix()  # PurePosixPath drops a leading "./" itself


def extract_tar(archive: Path, dst: Path) -> list[str]:
    names: list[str] = []
    dst.mkdir(parents=True)
    with tarfile.open(archive, "r:*") as tf:
        for m in tf.getmembers():
            name = safe_member_name(m.name)
            target = dst / name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    raise Refusal(f"tar member {name} has no data")
                with src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
                os.chmod(target, 0o644)
                names.append(name)
            else:
                raise Refusal(f"tar member {name} is not a regular file or directory (type {m.type!r})")
    return names


# ---------------------------------------------------------------------------------------------------------- manifests


def manifest_refs(doc: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            p, s = x.get("path"), x.get("sha256")
            if isinstance(p, str) and isinstance(s, str) and len(s) == 64:
                out.append((p, s))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(doc)
    return out


def resolve_ref(path: str, manifest_dir: Path, payload: Path, prefix_map: dict[str, str]) -> Path | None:
    if not path.startswith("/"):
        return manifest_dir / path
    for prefix in sorted(prefix_map, key=len, reverse=True):
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            rest = path[len(prefix.rstrip("/")):].lstrip("/")
            return payload / prefix_map[prefix] / rest
    return None


def keyed_refs(doc: Any, files_key: str) -> list[tuple[str, str]]:
    """{"<files_key>": {"name": {"sha256": ...}}} style (qtext shard.json, collection indexes)."""
    table = doc.get(files_key) if isinstance(doc, dict) else None
    if not isinstance(table, dict):
        raise Refusal(f"manifest has no {files_key!r} table")
    return [(name, meta["sha256"]) for name, meta in table.items() if isinstance(meta, dict) and isinstance(meta.get("sha256"), str)]


def check_manifest(payload: Path, spec: str | dict[str, Any], prefix_map: dict[str, str]) -> dict[str, Any]:
    """spec: "rel/path.json" ({"path", "sha256"} refs anywhere) or {"rel": ..., "files_key": "files"} (keyed refs)."""
    rel = spec if isinstance(spec, str) else spec["rel"]
    mpath = payload / rel
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    refs = manifest_refs(doc) if isinstance(spec, str) else keyed_refs(doc, spec["files_key"])
    result = {"manifest": rel, "refs": 0, "ok": 0, "mismatch": [], "missing": [], "external": 0}
    seen: set[tuple[str, str]] = set()
    for path, sha in refs:
        if (path, sha) in seen:
            continue
        seen.add((path, sha))
        result["refs"] += 1
        target = resolve_ref(path, mpath.parent, payload, prefix_map)
        if target is None or not os.path.normpath(target).startswith(os.path.normpath(payload) + os.sep):
            result["external"] += 1
            continue
        if not target.is_file():
            result["missing"].append(path)
            continue
        if sha256_file(target) != sha:
            result["mismatch"].append(path)
        else:
            result["ok"] += 1
    return result


# ---------------------------------------------------------------------------------------------------------- items


def write_sums(item_dir: Path, files: dict[str, str]) -> str:
    text = "".join(f"{sha}  payload/{rel}\n" for rel, sha in sorted(files.items()))
    (item_dir / "SHA256SUMS").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def adopt_item(item: dict[str, Any], dest: Path, args: argparse.Namespace) -> dict[str, Any]:
    name = item["name"]
    if not name or "/" in name or name.startswith("."):
        raise Refusal(f"bad item name {name!r}")
    kind = item["kind"]
    src = Path(item["src"])
    final = dest / name
    record: dict[str, Any] = {"schema": ADOPTION_SCHEMA, "name": name, "kind": kind, "source": str(src),
                              "decision": item.get("decision", "ADOPT"), "why": item.get("why", ""),
                              "tool": f"{TOOL_NAME} {TOOL_VERSION}", "started_utc": utc()}
    done = final / "ADOPTION.json"
    if done.is_file():
        prior = json.loads(done.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS":
            prior["skipped_already_adopted_utc"] = utc()
            return prior
    if final.exists():
        raise Refusal(f"{final} exists without a PASS adoption record; refusing to overwrite")

    record["mem_available_bytes_at_start"] = wait_for_memory(args.min_mem_gib, args.mem_wait_s, args.meminfo)

    pins = item.get("pins", {})
    pin_results = {}
    for rel, want in pins.items():
        p = src / rel if rel else src
        got = sha256_file(p) if p.is_file() else None
        pin_results[rel or "."] = {"expected": want, "source_sha256": got}
        if got != want:
            raise Refusal(f"{name}: source pin {rel or src} is {got}, expected {want}")
    record["source_pins"] = pin_results

    staging = dest / (name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)  # our own interrupted staging only; never a finished item, never a Codex path
    staging.mkdir(parents=True)
    payload = staging / "payload"
    try:
        if kind == "tree":
            src_files, links, problems = walk_tree(src)
            if problems:
                raise Refusal(f"{name}: source problems: {problems[:5]}")
            copy_tree(src, payload, links)
            dst_files, dst_links, dst_problems = walk_tree(payload)
            if dst_problems or dst_links != links:
                raise Refusal(f"{name}: copy problems {dst_problems[:5]} or symlink drift")
            diff = sorted(set(src_files.items()) ^ set(dst_files.items()))
            if diff:
                raise Refusal(f"{name}: copy differs from source at {len(diff)} entries, e.g. {diff[:3]}")
            record["symlinks"] = links
            files = dst_files
        elif kind == "tar":
            index = json.loads((src.parent / item["index"]).read_text(encoding="utf-8")) if not os.path.isabs(item["index"]) \
                else json.loads(Path(item["index"]).read_text(encoding="utf-8"))
            archive_sha = sha256_file(src)
            if archive_sha != index.get("archive_sha256"):
                raise Refusal(f"{name}: archive sha {archive_sha} != index archive_sha256 {index.get('archive_sha256')}")
            record["archive_sha256"] = archive_sha
            extracted = extract_tar(src, payload)
            listed = index.get("files", {})
            if sorted(extracted) != sorted(listed):
                extra = sorted(set(extracted) - set(listed))[:5]
                missing = sorted(set(listed) - set(extracted))[:5]
                raise Refusal(f"{name}: tar members differ from index (extra {extra}, missing {missing})")
            files = {}
            bad = []
            for rel, meta in listed.items():
                p = payload / rel
                got = sha256_file(p)
                files[rel] = got
                if got != meta.get("sha256") or p.stat().st_size != meta.get("bytes"):
                    bad.append(rel)
            if bad:
                raise Refusal(f"{name}: {len(bad)} extracted files differ from the index, e.g. {bad[:3]}")
            record["index_files"] = len(listed)
        else:
            raise Refusal(f"{name}: unknown kind {kind!r}")

        inner = {}
        for rel, want in item.get("payload_pins", {}).items():
            got = files.get(rel)
            inner[rel] = {"expected": want, "payload_sha256": got}
            if got != want:
                raise Refusal(f"{name}: payload pin {rel} is {got}, expected {want}")
        record["payload_pins"] = inner

        checks = [check_manifest(payload, rel, item.get("prefix_map", {})) for rel in item.get("manifests", [])]
        for c in checks:
            if c["mismatch"] or c["missing"]:
                raise Refusal(f"{name}: manifest {c['manifest']} has {len(c['mismatch'])} mismatched and "
                              f"{len(c['missing'])} missing references, e.g. {(c['mismatch'] + c['missing'])[:3]}")
        record["manifest_checks"] = checks

        record["files"] = len(files)
        record["bytes"] = sum((payload / rel).stat().st_size for rel in files)
        record["sha256sums_sha256"] = write_sums(staging, files)
        record["status"] = "PASS"
        record["finished_utc"] = utc()
        (staging / "ADOPTION.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        for dirpath, _, filenames in os.walk(payload):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if not os.path.islink(p):
                    os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) & ~0o222)
        os.replace(staging, final)
        return record
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def cmd_run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"plan schema {plan.get('schema')!r} != {PLAN_SCHEMA}")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    only = set(args.only or [])
    results = []
    for item in plan["items"]:
        if only and item["name"] not in only:
            continue
        try:
            rec = adopt_item(item, dest, args)
            line = f"ADOPT {rec['status']} {item['name']}: files {rec.get('files')} bytes {rec.get('bytes')}"
            for c in rec.get("manifest_checks", []):
                line += f"; {c['manifest']} refs {c['refs']} ok {c['ok']} external {c['external']}"
            print(line, flush=True)
            results.append({"name": item["name"], "status": rec["status"], "sha256sums_sha256": rec.get("sha256sums_sha256"),
                            "files": rec.get("files"), "bytes": rec.get("bytes")})
        except Refusal as exc:
            print(f"ADOPT FAIL {item['name']}: {exc}", flush=True)
            results.append({"name": item["name"], "status": "FAIL", "reason": str(exc)})
    receipt = {"schema": RECEIPT_SCHEMA, "tool": f"{TOOL_NAME} {TOOL_VERSION}", "tool_sha256": sha256_file(Path(__file__)),
               "plan": str(plan_path), "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "dest": str(dest),
               "utc": utc(), "items": results, "pass": sum(r["status"] == "PASS" for r in results),
               "fail": sum(r["status"] == "FAIL" for r in results)}
    out = dest / f"RECEIPT-{receipt['utc'].replace(':', '')}.json"
    out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(f"RESULT {TOOL_NAME} {TOOL_VERSION}: {receipt['pass']} adopted, {receipt['fail']} failed; receipt {out}", flush=True)
    return 0 if receipt["fail"] == 0 else 1


def cmd_verify_item(args: argparse.Namespace) -> int:
    item_dir = Path(args.dir)
    bad = 0
    n = 0
    for line in (item_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        n += 1
        if sha256_file(item_dir / rel) != sha:
            bad += 1
            print(f"MISMATCH {rel}")
    print(f"VERIFY {item_dir.name}: {n - bad}/{n} files match SHA256SUMS")
    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--dest", required=True)
    r.add_argument("--only", nargs="*")
    r.add_argument("--min-mem-gib", type=float, default=40.0)
    r.add_argument("--mem-wait-s", type=float, default=1800.0)
    r.add_argument("--meminfo", default="/proc/meminfo")
    v = sub.add_parser("verify-item")
    v.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_verify_item(args)


if __name__ == "__main__":
    sys.exit(main())
