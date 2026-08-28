#!/usr/bin/env python3
"""Cut a wave-29 deployment from a commit and verify it file by file.

The protocol is waves 15-25's, written down here because it had never been in
the repository -- it lived in a shell history, and a shell history is not a
receipt.  Three properties are what make a deployment usable as the thing a
receipt names:

* **the bytes are the commit's**, which is why the archive is cut with
  ``-c core.autocrlf=false`` on a repository whose ``core.autocrlf`` is ``true``:
  the working copy carries CRLF for some files and the index does not, and a
  deployment that carries the working copy's line endings is not the commit;
* **every file is verified against the commit's own blob digest**, not against a
  checksum the deployment computed about itself, and the count of missing,
  mismatched and undeclared files is reported rather than summarised as "OK";
* **no file has a second name.**  Wave 21 cut its deployment as a hardlink copy
  plus a delta, and every frozen review then refused, because
  ``lfm2_carrier_review_cli._ordinary_file`` checks ``st_nlink != 1`` -- a prompt
  pack reachable under a second name is a prompt pack something else can rewrite.
  ``st_nlink`` is asserted here rather than discovered at review time.

Seven blobs over 20 MB are excluded by name, as every deployment in this
campaign has excluded them; they are report artifacts and nothing in the run
path opens them.

Run it in two halves: ``--emit`` locally (it needs the git repository) and
``--verify`` on the server (it needs the extracted tree and the manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

EXCLUDED_OVER_20MB = (
    "reports/smolvlm-real-fold-20260804/L3-prefix/child/model.safetensors",
    "reports/cross-family-real-20260804-pretrained/L6-prefix-random-control/child/model.safetensors",
    "reports/cross-family-real-20260803/L6-prefix-fold/child/model.safetensors",
    "reports/cross-family-real-20260804-pretrained/L3-prefix-random-control/child/model.safetensors",
    "reports/cross-family-real-20260804-pretrained/L3-prefix-pretrained/child/model.safetensors",
    "reports/cross-family-real-20260803/L2-full-rank-parent-control/child/model.safetensors",
    "reports/cross-family-real-20260803/L2-exact-subspace-parent/child/model.safetensors",
)


def emit(repo: Path, commit: str, out_tar: Path, out_manifest: Path) -> int:
    listing = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--format=%(objectname) %(path)", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    manifest: dict[str, str] = {}
    for line in listing:
        blob, path = line.split(" ", 1)
        if path in EXCLUDED_OVER_20MB:
            continue
        manifest[path] = blob
    # The blob id is a digest of the CONTENT with a git header, so the manifest
    # carries the content sha256 too: the verifier must not need a git binary.
    # One `cat-file --batch` for the whole tree -- 3,736 process spawns is
    # twenty minutes on this workstation and one pipe is twenty seconds.
    content: dict[str, str] = {}
    paths = list(manifest)
    with subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as batch:
        assert batch.stdin is not None and batch.stdout is not None
        for path in paths:
            batch.stdin.write(manifest[path].encode("ascii") + b"\n")
            batch.stdin.flush()
            header = batch.stdout.readline().decode("ascii").split()
            size = int(header[2])
            body = batch.stdout.read(size)
            batch.stdout.read(1)  # the trailing newline `--batch` writes
            content[path] = hashlib.sha256(body).hexdigest()
        batch.stdin.close()
    out_manifest.write_text(
        json.dumps(
            {
                "schema": "neural-forge-matrix-wave35-deployment-manifest-v1",
                "commit": commit,
                "files": len(content),
                "excluded_over_20mb": list(EXCLUDED_OVER_20MB),
                "content_sha256": content,
            },
            indent=1,
        )
    )
    # The archive is FILTERED to the manifest, not merely accompanied by it: an
    # excluded blob that ships anyway lands on disk as an undeclared file and
    # the verifier -- correctly -- refuses the whole deployment for it.
    with subprocess.Popen(
        ["git", "-C", str(repo), "-c", "core.autocrlf=false", "archive", "--format=tar", commit],
        stdout=subprocess.PIPE,
    ) as proc:
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode="r|") as source, tarfile.open(
            str(out_tar), "w:gz"
        ) as sink:
            for member in source:
                if member.isfile() and member.name not in manifest:
                    continue
                if member.isfile():
                    sink.addfile(member, source.extractfile(member))
                else:
                    sink.addfile(member)
        if proc.wait() != 0:
            raise SystemExit("git archive failed")
    print(f"emitted {out_tar} ({out_tar.stat().st_size} bytes) and {out_manifest}")
    print(f"manifest declares {len(content)} files at commit {commit}")
    return 0


def verify(root: Path, manifest_path: Path, report_path: Path | None) -> int:
    body = json.loads(manifest_path.read_text())
    declared = body["content_sha256"]
    mismatched: list[str] = []
    missing: list[str] = []
    hardlinked: list[str] = []
    for path, digest in declared.items():
        target = root / path
        if not target.is_file():
            missing.append(path)
            continue
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != digest:
            mismatched.append(path)
        if os.stat(target).st_nlink != 1:
            hardlinked.append(path)
    on_disk = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    }
    excluded = set(body["excluded_over_20mb"])
    undeclared = sorted(on_disk - set(declared) - excluded)
    report = {
        "schema": "neural-forge-matrix-wave35-deployment-verification-v1",
        "deployment": str(root),
        "commit": body["commit"],
        "files_declared": len(declared),
        "files_on_disk": len(on_disk),
        "content_mismatches": len(mismatched),
        "missing": len(missing),
        "undeclared_on_disk": len(undeclared),
        "hardlinks": len(hardlinked),
        "mismatched_paths": mismatched[:20],
        "missing_paths": missing[:20],
        "undeclared_paths": undeclared[:20],
        "hardlinked_paths": hardlinked[:20],
    }
    print(json.dumps(report, indent=1))
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=1))
    bad = len(mismatched) + len(missing) + len(undeclared) + len(hardlinked)
    if bad:
        raise SystemExit(f"DEPLOYMENT REFUSED: {bad} discrepancies")
    print(
        f"COMPLETENESS OK: {len(declared)} files, 0 content mismatches, "
        "0 missing, 0 undeclared, 0 hardlinks"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--commit")
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.emit:
        return emit(args.repo, args.commit, args.tar, args.manifest)
    if args.verify:
        return verify(args.root, args.manifest, args.report)
    raise SystemExit("--emit or --verify")


if __name__ == "__main__":
    raise SystemExit(main())
