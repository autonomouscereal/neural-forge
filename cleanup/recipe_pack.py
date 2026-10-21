#!/usr/bin/env python3
"""recipe_pack.py 1.0.0 - pack loose recipe trees (cleanup_exec 1.0.0 layout) into one tree.tar.gz each.

Lexar is exFAT with 1 MiB clusters: a recipe of 16k small JSON files occupied 16 GB as loose files. For every
<recipe>/tree/ directory: build tree.tar.gz.partial, re-read every member and compare its sha256 with the
MANIFEST.tsv row (digest_kind sha256), require the member set to equal the MANIFEST rows marked archived=yes,
rename to tree.tar.gz, append a line to PACKED.txt, and only then remove tree/. A mismatch leaves tree/ intact.

    python3 recipe_pack.py /mnt/lexar/omni-campaign/archive/recipes [--dry-run]
"""
import hashlib
import os
import shutil
import sys
import tarfile
from datetime import datetime, timezone


def manifest(rec):
    want = {}
    with open(os.path.join(rec, "MANIFEST.tsv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            rel, size, mtime, kind, digest, archived, tracked = line.rstrip("\n").split("\t")
            if archived == "yes":
                want[rel] = digest
    return want


def pack(rec, dry):
    tree = os.path.join(rec, "tree")
    want = manifest(rec)
    have = set()
    for root, dirs, names in os.walk(tree):
        for n in names:
            have.add(os.path.relpath(os.path.join(root, n), tree))
    if have != set(want):
        return f"SKIP {rec}: tree/ files != MANIFEST archived rows ({len(have)} vs {len(want)})"
    if dry:
        return f"would pack {rec}: {len(want)} files"
    tmp = os.path.join(rec, "tree.tar.gz.partial")
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
        for rel in sorted(want):
            tf.add(os.path.join(tree, rel), arcname=rel, recursive=False)
    seen = 0
    with tarfile.open(tmp, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            h = hashlib.sha256()
            f = tf.extractfile(m)
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
            if h.hexdigest() != want.get(m.name):
                os.remove(tmp)
                return f"FAIL {rec}: member {m.name} sha mismatch; tree/ kept"
            seen += 1
    if seen != len(want):
        os.remove(tmp)
        return f"FAIL {rec}: {seen} members != {len(want)}; tree/ kept"
    final = os.path.join(rec, "tree.tar.gz")
    os.replace(tmp, final)
    h = hashlib.sha256()
    with open(final, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    with open(os.path.join(rec, "PACKED.txt"), "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} recipe_pack 1.0.0: tree/ ({len(want)} files, every member "
                 f"sha256-verified against MANIFEST.tsv) -> tree.tar.gz sha256 {h.hexdigest()}; tree/ removed\n")
    shutil.rmtree(tree)
    return f"packed {rec}: {len(want)} files"


def main():
    root = sys.argv[1]
    dry = "--dry-run" in sys.argv
    rc = 0
    for name in sorted(os.listdir(root)):
        rec = os.path.join(root, name)
        if os.path.isdir(os.path.join(rec, "tree")) and os.path.isfile(os.path.join(rec, "MANIFEST.tsv")):
            msg = pack(rec, dry)
            print(msg, flush=True)
            if msg.startswith(("FAIL", "SKIP")):
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
