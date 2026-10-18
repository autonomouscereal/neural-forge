#!/usr/bin/env python3
"""media_a_corpus_items.py -- distill corpus media rows -> stage_c_media_a item JSONL.

Lane gpu4 (Claude, 2026-10-13). The corpus lane (`distill_corpus_build.py` 1.0.1) wrote the stage-C media sets as
`$OC/distill/corpus/media/<teacher>/items.jsonl` in ITS schema: `uid` (no `id`), the media as a dict
(`{"path", "stored_path", "sha256", ...}`) and `negative_prompt`. `stage_c_media_a.py` 1.0.2 `read_items` expects
`{"id", "prompt", <media_key>: <path>, ["negative"], ["sha256"]}`. This tool is the one-to-one adapter between them;
it changes no content: every corpus row becomes exactly one item, in corpus order, with

    id        <- uid                       (must be present and unique)
    prompt    <- prompt                    (must be a non-empty string)
    image     <- image.stored_path         (the corpus's own local copy under distill/corpus/media_store;
                                            `--media-path-key path` picks the NAS original instead)
    sha256    <- image.sha256              (stage_c_media_a re-hashes the file and drops the item on mismatch)
    negative  <- negative_prompt           (only when present; "" is kept: it IS the parent's default negative)

plus a `corpus` block with the remaining source fields (tier, task_type, rid, dataset_id, uid, the source line), so
the converted file is self-describing. `--check-files` stats every media path and (with `--hash`) re-hashes it, so a
missing or changed file is found here and not hours into a GPU job. A receipt JSON beside the output records the
source file's sha256, the output's sha256 and the counts.

    convert --route image --src $OC/distill/corpus/media/qimg/items.jsonl --out <items.jsonl> [--check-files [--hash]]

Exit codes: 0 written, 2 refusal (bad row, duplicate id, missing file, hash mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "media_a_corpus_items"
TOOL_VERSION = "1.0.0"
SCHEMA = "omni-campaign.stage_c.media_a.items-from-corpus.v1"
ROUTES = {"image": "image", "video": "video"}
CARRIED = ("uid", "rid", "dataset_id", "tier", "task_type", "route", "prompt_source", "negative_source", "target_resolution")


class Refusal(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_row(row: dict[str, Any], *, line: int, route: str, media_path_key: str) -> dict[str, Any]:
    """One corpus row -> one media-a item (pure)."""

    ident = row.get("uid")
    if not isinstance(ident, str) or not ident.strip():
        raise Refusal(f"line {line}: corpus row has no uid")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Refusal(f"line {line}: row {ident} has no prompt")
    media = row.get(route)
    if not isinstance(media, dict):
        raise Refusal(f"line {line}: row {ident} has no {route!r} dict (got {type(media).__name__})")
    path = media.get(media_path_key)
    if not isinstance(path, str) or not path:
        raise Refusal(f"line {line}: row {ident} {route}.{media_path_key} is missing")
    item: dict[str, Any] = {"id": ident, "prompt": prompt, ROUTES[route]: path}
    if isinstance(media.get("sha256"), str) and media["sha256"]:
        item["sha256"] = media["sha256"]
    if "negative_prompt" in row:
        if not isinstance(row["negative_prompt"], str):
            raise Refusal(f"line {line}: row {ident} negative_prompt is not a string")
        item["negative"] = row["negative_prompt"]
    item["corpus"] = {k: row[k] for k in CARRIED if k in row}
    item["corpus"]["line"] = line
    item["corpus"][f"{route}_source_path"] = media.get("path")
    return item


def convert(src: Path, *, route: str, media_path_key: str = "stored_path") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(src, "r", encoding="utf-8") as handle:
        for line, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            item = convert_row(json.loads(text), line=line, route=route, media_path_key=media_path_key)
            if item["id"] in seen:
                raise Refusal(f"line {line}: duplicate uid {item['id']}")
            seen.add(item["id"])
            items.append(item)
    if not items:
        raise Refusal(f"{src}: no rows")
    return items


def check_files(items: list[dict[str, Any]], *, route: str, rehash: bool) -> dict[str, Any]:
    missing, mismatched, total_bytes = [], [], 0
    for item in items:
        path = Path(item[ROUTES[route]])
        if not path.is_file():
            missing.append(item["id"])
            continue
        total_bytes += path.stat().st_size
        if rehash and item.get("sha256") and sha256_file(path) != item["sha256"]:
            mismatched.append(item["id"])
    if missing or mismatched:
        raise Refusal(f"{len(missing)} missing file(s) (first {missing[:3]}), {len(mismatched)} sha256 mismatch(es) (first {mismatched[:3]})")
    return {"files": len(items), "bytes": total_bytes, "rehashed": bool(rehash)}


def write_items(items: list[dict[str, Any]], out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return sha256_file(out)


def run_convert(args: argparse.Namespace) -> int:
    src, out = Path(args.src), Path(args.out)
    items = convert(src, route=args.route, media_path_key=args.media_path_key)
    files = check_files(items, route=args.route, rehash=bool(args.hash)) if args.check_files else None
    if out.exists() and not args.overwrite:
        existing = [json.loads(t) for t in out.read_text(encoding="utf-8").splitlines() if t.strip()]
        if existing != items:
            raise Refusal(f"{out} exists with different content; pass --overwrite (a velocity cache pins the items file sha)")
        out_sha = sha256_file(out)
        print(f"unchanged {out} sha256 {out_sha}")
    else:
        out_sha = write_items(items, out)
    receipt = {
        "schema": SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(__file__),
        "route": args.route, "media_path_key": args.media_path_key, "source": str(src), "source_sha256": sha256_file(src),
        "out": str(out), "out_sha256": out_sha, "items": len(items),
        "with_negative": sum(1 for i in items if "negative" in i), "with_sha256": sum(1 for i in items if "sha256" in i),
        "file_check": files, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rpath = out.with_name(out.name + ".receipt.json")
    rpath.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{TOOL_NAME} {TOOL_VERSION}: {len(items)} items -> {out} sha256 {out_sha} (source {receipt['source_sha256'][:16]}); "
          f"file check {files}; receipt {rpath}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("convert", help="corpus media rows -> stage_c_media_a items")
    p.add_argument("--route", required=True, choices=sorted(ROUTES))
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--media-path-key", default="stored_path", choices=("stored_path", "path"))
    p.add_argument("--check-files", action="store_true")
    p.add_argument("--hash", action="store_true", help="with --check-files: re-hash every media file against its sha256")
    p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_convert(args)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
