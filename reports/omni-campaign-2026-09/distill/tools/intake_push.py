#!/usr/bin/env python3
"""intake_push.py -- workstation side of the Chimera corpus intake (design A): copy files from a local staging tree
(default D:\\chimera-intake\\<modality>\\...) to the NAS drop zone \\\\192.168.50.250\\ai\\chimera-intake\\<modality>\\...

Written 2026-10-15 by Claude.  Standard library only; runs on Windows (C:/Python314/python.exe) or anywhere.

Each file is copied under a temporary "<name>.part" name (the server's settle rule ignores *.part), re-hashed on the
share, renamed into place, and a "<name>.sha256" sidecar is written beside it.  A file already on the share with the
same sha256 is skipped; a DIFFERENT file with the same name is never overwritten (it is pushed as
"<stem>.<sha8><ext>").  Local files are left alone unless --delete-after-verify is given (then only after the copy on
the share re-hashed equal).  Never deletes or changes anything on the share except its own .part files.

    C:/Python314/python.exe intake_push.py [--src D:/chimera-intake] [--dst //192.168.50.250/ai/chimera-intake] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

TOOL = "intake_push"
VERSION = "1.0.0"   # 1.0.0 (2026-10-15): first version
MODALITY_DIRS = ("speech", "audio", "music", "images", "video", "text", "documents", "3d", "mixed")
SKIP_NAMES = {"desktop.ini", "Thumbs.db", ".DS_Store", "_README.txt"}
SKIP_SUFFIXES = (".part", ".crdownload", ".tmp", "~", ".sha256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates(src: Path) -> list[Path]:
    out = []
    for modality in MODALITY_DIRS:
        base = src / modality
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.name not in SKIP_NAMES and not path.name.startswith(".") \
                    and not path.name.lower().endswith(SKIP_SUFFIXES):
                out.append(path)
    return out


def push_one(path: Path, src: Path, dst: Path, *, dry_run: bool, delete_after: bool, min_age_s: float) -> dict:
    rel = path.relative_to(src)
    if time.time() - path.stat().st_mtime < min_age_s:
        return {"file": str(rel), "action": "wait", "why": "modified in the last %d s" % min_age_s}
    sha = sha256_file(path)
    target = dst / rel
    if target.exists():
        if target.is_file() and sha256_file(target) == sha:
            if delete_after and not dry_run:
                path.unlink()
            return {"file": str(rel), "action": "present", "sha256": sha}
        target = target.with_name(f"{target.stem}.{sha[:8]}{target.suffix}")
        if target.exists() and sha256_file(target) == sha:
            return {"file": str(rel), "action": "present", "sha256": sha, "as": target.name}
    if dry_run:
        return {"file": str(rel), "action": "would_copy", "sha256": sha, "to": str(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    shutil.copyfile(path, part)
    if sha256_file(part) != sha:
        part.unlink(missing_ok=True)
        return {"file": str(rel), "action": "error", "why": "copy re-hash mismatch"}
    os.replace(part, target)
    target.with_name(target.name + ".sha256").write_text(f"{sha}  {target.name}\n", encoding="ascii")
    if delete_after and sha256_file(target) == sha:
        path.unlink()
    return {"file": str(rel), "action": "copied", "sha256": sha, "to": str(target)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", default="D:/chimera-intake")
    parser.add_argument("--dst", default="//192.168.50.250/ai/chimera-intake")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-after-verify", action="store_true")
    parser.add_argument("--min-age-s", type=float, default=60.0, help="skip files modified more recently (still being written)")
    args = parser.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    if not dst.is_dir():
        print(f"[{TOOL}] drop zone {dst} not reachable", file=sys.stderr)
        return 2
    results = [push_one(p, src, dst, dry_run=args.dry_run, delete_after=args.delete_after_verify, min_age_s=args.min_age_s)
               for p in candidates(src)] if src.is_dir() else []
    summary = {}
    for r in results:
        summary[r["action"]] = summary.get(r["action"], 0) + 1
    print(json.dumps({"tool": f"{TOOL} {VERSION}", "src": str(src), "dst": str(dst), "summary": summary,
                      "results": results[:200]}, indent=1))
    return 1 if summary.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
