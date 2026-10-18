#!/usr/bin/env python3
"""speech_fetch.py -- pinned Hugging Face speech-teacher downloads with per-file proof (Claude, 2026-10-15).

Design of record: distill/SPEECH_LANE_PLAN_v2_2026-10-14.md section 2.4 (decision 4), plan step 3.  Layout follows the
stage-teachers order of 2026-10-15: /mnt/lexar/models/<org>/<name>/ (the existing moonshotai/<name> convention), one
pinned revision per directory; the revision lives in the receipt, not the path.

What it does for one repo at one FULL commit sha:
  * reads https://huggingface.co/api/models/<repo>/revision/<sha>?blobs=true (bytes hashed into the receipt), refuses
    when the API answers with another sha, and refuses a gated repo unless HF_TOKEN is set in the environment (vault
    key `hf_token`; os.environ.get("HF_TOKEN", "") -- the token is never printed or written);
  * for every file: keeps it when the local copy already proves out, else copies it from --seed (an on-disk copy of
    the same repo, e.g. Codex's root parents) when size and hash prove out, else downloads
    https://huggingface.co/<repo>/resolve/<sha>/<path> into <file>.part with HTTP Range resume;
  * proof per file = LFS files: sha256 == the API's lfs.sha256; plain git files: git blob sha1 == the API's blobId.
    A file that does not prove out is deleted and the run fails; nothing unproven is renamed into place;
  * writes <dest>/.speech_fetch_receipt.json and the same JSON under --receipt-dir, with licence (API cardData plus the
    sha256 of any LICENSE* file in the repo), sizes, sha256 per file, source per file (present | seed | download),
    wall time and MB/s.
Never imports, loads or executes repository code.  Refuses to start when the destination filesystem would end below
--floor-gb free.  Stdlib only, so it runs under any python3.

Usage
    python3 speech_fetch.py fetch --repo Qwen/Qwen3-TTS-12Hz-1.7B-Base --revision <40-hex> \
        [--dest /mnt/lexar/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base] [--seed DIR] [--receipt-dir DIR] [--workers 4]
    python3 speech_fetch.py verify --dest DIR      (re-hash every file against its receipt; exit 1 on any mismatch)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

TOOL_VERSION = "1.0.0"
SCHEMA = "omni-campaign.speech-fetch-receipt.v1"
HF = "https://huggingface.co"
RECEIPT_NAME = ".speech_fetch_receipt.json"
CHUNK = 8 * 1024 * 1024


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def request(url: str, headers: dict | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": f"omni-speech-fetch/{TOOL_VERSION}"})
    token = os.environ.get("HF_TOKEN", "")
    if token and url.startswith(HF):
        req.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    return req


def file_digests(path: Path) -> tuple[str, str]:
    """(sha256, git blob sha1) of a file in one pass."""
    sha = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            sha.update(chunk)
            git.update(chunk)
    return sha.hexdigest(), git.hexdigest()


def proves_out(path: Path, spec: dict) -> tuple[bool, str, str]:
    """True when the file at path is byte-identical to the API's description of it."""
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False, "", ""
    sha256, git_sha1 = file_digests(path)
    if spec.get("lfs_sha256"):
        return sha256 == spec["lfs_sha256"], sha256, git_sha1
    return git_sha1 == spec["blob_id"], sha256, git_sha1


def safe_relative(name: str) -> PurePosixPath:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe repo path: {name!r}")
    return rel


def read_api(repo: str, revision: str) -> tuple[dict, str]:
    url = f"{HF}/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(request(url), timeout=60) as resp:
        raw = resp.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def download(url: str, part: Path, size: int) -> None:
    """Stream url into part, resuming from its current length; retries transient failures."""
    for attempt in range(8):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            return
        if have > size:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(request(url, headers), timeout=120) as resp:
                if have and resp.status != 206:
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                with part.open(mode) as out:
                    for chunk in iter(lambda: resp.read(CHUNK), b""):
                        out.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"  retry {attempt + 1}/8 {part.name}: {exc}", file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"download incomplete after retries: {url}")


def fetch_one(repo: str, revision: str, dest: Path, seed: Path | None, spec: dict) -> dict:
    rel = safe_relative(spec["path"])
    target = dest.joinpath(*rel.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok, sha256, git_sha1 = proves_out(target, spec)
    source = "present"
    if not ok:
        part = target.with_name(target.name + ".part")
        candidate = seed.joinpath(*rel.parts) if seed else None
        seeded = False
        if candidate is not None and candidate.is_file() and candidate.stat().st_size == spec["size"]:
            shutil.copyfile(candidate, part)
            ok, sha256, git_sha1 = proves_out(part, spec)
            seeded = ok
            if not ok:
                part.unlink()
        if not seeded:
            download(f"{HF}/{repo}/resolve/{revision}/{rel.as_posix()}", part, spec["size"])
            ok, sha256, git_sha1 = proves_out(part, spec)
            if not ok:
                part.unlink()
                raise RuntimeError(f"hash mismatch after download: {rel} (deleted)")
        os.replace(part, target)
        source = "seed" if seeded else "download"
    return {
        "path": rel.as_posix(), "size": spec["size"], "sha256": sha256, "git_blob_sha1": git_sha1,
        "lfs": bool(spec.get("lfs_sha256")), "verified_against": "lfs.sha256" if spec.get("lfs_sha256") else "blobId",
        "source": source, "seconds": round(time.time() - started, 2),
    }


def cmd_fetch(args: argparse.Namespace) -> int:
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        print("--revision must be a full 40-hex commit sha", file=sys.stderr)
        return 2
    org, name = args.repo.split("/", 1)
    dest = Path(args.dest or f"/mnt/lexar/models/{org}/{name}")
    seed = Path(args.seed) if args.seed else None
    api, api_sha256 = read_api(args.repo, args.revision)
    if api.get("sha") != args.revision:
        print(f"API answered sha {api.get('sha')} for {args.revision}", file=sys.stderr)
        return 3
    gated = api.get("gated") or False
    if gated and not os.environ.get("HF_TOKEN", ""):
        print(f"{args.repo} is gated ({gated}); set HF_TOKEN from vault key hf_token after the gate is accepted",
              file=sys.stderr)
        return 4
    specs = []
    for sib in api.get("siblings", []):
        lfs = sib.get("lfs") or {}
        specs.append({"path": sib["rfilename"], "size": int(sib.get("size", lfs.get("size", 0))),
                      "blob_id": sib.get("blobId", ""), "lfs_sha256": lfs.get("sha256", "")})
    total = sum(s["size"] for s in specs)
    dest.mkdir(parents=True, exist_ok=True)
    have = sum(min(s["size"], (dest / s["path"]).stat().st_size) for s in specs if (dest / s["path"]).is_file())
    free = shutil.disk_usage(dest).free
    if free - (total - have) < args.floor_gb * 1024**3:
        print(f"refusing: {free / 1e9:.1f} GB free, need {(total - have) / 1e9:.1f} GB + floor {args.floor_gb} GB",
              file=sys.stderr)
        return 5
    print(f"{args.repo}@{args.revision[:8]}: {len(specs)} files, {total / 1e9:.3f} GB -> {dest}", flush=True)
    started_utc, started = utc_now(), time.time()
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, args.repo, args.revision, dest, seed, s): s for s in specs}
        for fut in concurrent.futures.as_completed(futures):
            spec = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"  ok {row['source']:8s} {row['size'] / 1e6:10.1f} MB  {row['path']}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- every file failure is reported, then the run fails
                errors.append({"path": spec["path"], "error": repr(exc)})
                print(f"  FAIL {spec['path']}: {exc!r}", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["path"])
    seconds = time.time() - started
    downloaded = sum(r["size"] for r in rows if r["source"] == "download")
    card = api.get("cardData") or {}
    licence_files = [{"path": r["path"], "sha256": r["sha256"]} for r in rows
                     if PurePosixPath(r["path"]).name.upper().startswith(("LICENSE", "LICENCE"))]
    receipt = {
        "schema": SCHEMA, "tool": "speech_fetch.py", "tool_version": TOOL_VERSION,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repo": args.repo, "revision": args.revision, "dest": str(dest),
        "api_url": f"{HF}/api/models/{args.repo}/revision/{args.revision}?blobs=true", "api_sha256": api_sha256,
        "api_last_modified": api.get("lastModified"), "api_created_at": api.get("createdAt"),
        "gated": gated, "private": api.get("private"),
        "licence": {"card_license": card.get("license"), "card_license_name": card.get("license_name"),
                    "card_license_link": card.get("license_link"), "files": licence_files},
        "pipeline_tag": api.get("pipeline_tag"), "library_name": api.get("library_name"),
        "files": rows, "file_count": len(rows), "total_bytes": sum(r["size"] for r in rows),
        "downloaded_bytes": downloaded, "seed": str(seed) if seed else None,
        "started_utc": started_utc, "finished_utc": utc_now(), "seconds": round(seconds, 1),
        "download_mb_per_s": round(downloaded / 1e6 / seconds, 2) if seconds and downloaded else None,
        "errors": errors, "complete": not errors and len(rows) == len(specs),
    }
    text = json.dumps(receipt, indent=1, sort_keys=False) + "\n"
    (dest / RECEIPT_NAME).write_text(text)
    if args.receipt_dir:
        out = Path(args.receipt_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{org}__{name}.json").write_text(text)
    print(f"{'COMPLETE' if receipt['complete'] else 'INCOMPLETE'} {args.repo}@{args.revision[:8]} "
          f"{receipt['total_bytes'] / 1e9:.3f} GB, downloaded {downloaded / 1e9:.3f} GB in {seconds:.0f} s, "
          f"licence {card.get('license')}", flush=True)
    return 0 if receipt["complete"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    receipt = json.loads((dest / RECEIPT_NAME).read_text())
    bad = 0
    for row in receipt["files"]:
        path = dest / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"] or file_digests(path)[0] != row["sha256"]:
            bad += 1
            print(f"  MISMATCH {row['path']}")
    print(f"{'OK' if not bad else 'FAIL'} {receipt['repo']}@{receipt['revision'][:8]}: "
          f"{len(receipt['files']) - bad}/{len(receipt['files'])} files")
    return 0 if not bad else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--dest")
    fetch.add_argument("--seed")
    fetch.add_argument("--receipt-dir")
    fetch.add_argument("--workers", type=int, default=4)
    fetch.add_argument("--floor-gb", type=float, default=50.0)
    verify = sub.add_parser("verify")
    verify.add_argument("--dest", required=True)
    args = parser.parse_args()
    return cmd_fetch(args) if args.cmd == "fetch" else cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
