#!/usr/bin/env python3
"""d1_canonical_rungs.py -- the canonical OMNI1D1 rung directory (state doc 2026-10-14 step 4b).

Lane d1, Claude, 2026-10-14.  One directory holds every COMMITTED D1 rung of the r4 lineage, contiguous from L00,
whatever attempt trained it (v35 L00-L08, c36-r2 L09-L10, c37-r4 L11+ as they appear).  The hardened trainer
(d1_trunk_recovery >= 1.3.0, --prefix-dir) imports from it instead of from the attempt directories.

    sync   --canonical DIR --receipt RUN/d1_receipt.json --rungs-dir RUN/rungs [--label TEXT]
    verify --canonical DIR [--expect-layers N]

A rung is committed when the attempt's own d1_receipt.json carries a layer record with ``rung.sha256`` (written
after the rung, so a rung file without its record is not committed) AND the file on disk hashes to that sha AND
the rung's .json says the same ``file_sha256``.  Copies use ``cp -n`` (never overwrite), are re-hashed after the
copy and made read-only.  A layer the receipt imported (``import_provenance``) is a lineage check instead: the
canonical L{i}.json must hash to the receipt's ``old_record_pin`` -- the attempt trained on exactly these rungs.
1.1.0: the hardened trainer (d1_trunk_recovery >= 1.3.0, --prefix-dir) records an imported layer as
``imported_from_prefix`` {dir, sha256sums_sha256, json_sha256}; that is the same lineage check (the dir must BE this
canonical dir and json_sha256 its L{i}.json line).  1.0.0 stopped at such a record, so a hardened run's L11+ were
never synced.
1.2.0 (2026-10-14, d1seal review): the receipt of an attempt that RESUMED after a restart (the hardened unit restarts
on failure; the run has --resume --boundary-dir) records layers two more ways, and 1.1.0 refused or stopped on both:
  boundary resume  every layer up to the boundary gets ``rung`` {file, sha256} + ``from_boundary``, the prefix
                   layers with file = the canonical L{i}.safetensors (no copy in the run's rungs dir) -- a layer
                   already canonical is now compared by sha BEFORE its source file is looked for;
  replay resume    a layer replayed from the run's own rungs dir (load_rung) gets ``resumed_from`` and no sha; its
                   sha is the rung .json's file_sha256 (re-hashed here as for any committed rung), and
                   ``resumed_from`` must BE this run's rungs-dir file.
Refusals (exit 2): a sha mismatch anywhere, a canonical file that differs from a source, a gap (L{i} before
L{i-1}), a receipt whose sidecar never matches.  Sources are only read.

Files written in DIR: L{i:02d}.safetensors + L{i:02d}.json (copies), SHA256SUMS (``<sha>  <name>``, sorted; the
pin the trainer takes), PROVENANCE.json (per layer: source receipt + its sha, source file, sha, copy time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "d1_canonical_rungs"
TOOL_VERSION = "1.2.0"   # 1.0.0 (2026-10-14): first version, lane d1 (state doc step 4b); 1.1.0 (2026-10-14, lane
                         # d1seal): `imported_from_prefix` records (hardened trainer) are lineage checks, not the end;
                         # 1.2.0 (2026-10-14, d1seal review): boundary-resume and replay-resume layer records
SUMS = "SHA256SUMS"
PROVENANCE = "PROVENANCE.json"
RUNG_RE = re.compile(r"^L(\d{2})\.(safetensors|json)$")


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"[{TOOL_NAME}] REFUSED: {message}")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{TOOL_NAME}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_receipt(path: Path, tries: int = 10) -> tuple[dict[str, Any], str]:
    """The attempt's receipt + its sha; the .sha256 sidecar must agree (the writer replaces the receipt, then
    the sidecar, so a read between the two is retried)."""
    side = path.with_name(path.name + ".sha256")
    for _ in range(tries):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not side.is_file() or side.read_text().strip() == digest:
            return json.loads(raw), digest
        time.sleep(3)
    raise Refusal(f"{path}: the .sha256 sidecar never matched the receipt ({tries} reads)")


def canonical_state(canonical: Path) -> dict[int, dict[str, str]]:
    """{layer: {"safetensors": sha, "json": sha}} from SHA256SUMS (empty when absent)."""
    sums = canonical / SUMS
    out: dict[int, dict[str, str]] = {}
    if not sums.is_file():
        return out
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        m = RUNG_RE.match(name.strip())
        if not m:
            raise Refusal(f"{sums}: unexpected entry {name!r}")
        out.setdefault(int(m.group(1)), {})[m.group(2)] = digest
    return out


def write_sums(canonical: Path, state: dict[int, dict[str, str]]) -> str:
    lines = [f"{state[i][kind]}  L{i:02d}.{kind}" for i in sorted(state) for kind in ("json", "safetensors")]
    write_atomic(canonical / SUMS, "\n".join(lines) + "\n")
    return sha256_file(canonical / SUMS)


def contiguous(state: dict[int, dict[str, str]]) -> int:
    n = 0
    while n in state and set(state[n]) == {"json", "safetensors"}:
        n += 1
    if set(state) != set(range(n)):
        raise Refusal(f"canonical layers are not contiguous from L00: {sorted(state)}")
    return n


def cp_n(src: Path, dst: Path) -> None:
    """``cp -n`` (never overwrite), then read-only."""
    subprocess.run(["cp", "-n", "--", str(src), str(dst)], check=True)
    os.chmod(dst, 0o444)


def sync(canonical: Path, receipt_path: Path, rungs_dir: Path, label: str | None = None) -> dict[str, Any]:
    canonical.mkdir(parents=True, exist_ok=True)
    receipt, receipt_sha = read_receipt(receipt_path)
    state = canonical_state(canonical)
    have = contiguous(state)
    prov_path = canonical / PROVENANCE
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {
        "schema": "omni-campaign.d1-canonical-rungs.v1", "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "layers": {}, "syncs": []}
    layers = sorted(receipt.get("layers", []), key=lambda r: int(r["text_layer"]))
    added, checked, lineage = [], [], []
    for rec in layers:
        i = int(rec["text_layer"])
        rung = rec.get("rung")
        if (not rung and rec.get("trained") is True and rec.get("resumed_from") and not rec.get("import_provenance")
                and not rec.get("imported_from_prefix")):  # 1.2.0: replayed from this run's own rungs dir after a restart
            own = rungs_dir / f"L{i:02d}.safetensors"
            if Path(str(rec["resumed_from"])).resolve() != own.resolve():
                raise Refusal(f"L{i:02d}: resumed from {rec['resumed_from']}, not this run's {own}")
            if not own.with_suffix(".json").is_file():
                raise Refusal(f"L{i:02d}: resumed from {own} but {own.with_suffix('.json').name} is missing")
            rung = {"file": str(own), "sha256": json.loads(own.with_suffix(".json").read_text(encoding="utf-8")).get("file_sha256")}
        if rung:                                          # trained by this attempt (or an earlier attempt of this run)
            want = str(rung["sha256"])
            if i in state:                                # already canonical: must be the same bytes (1.2.0: checked
                if state[i]["safetensors"] != want:       # first -- a boundary-resumed prefix layer has no source copy)
                    raise Refusal(f"L{i:02d}: canonical {state[i]['safetensors']} != source receipt {want}")
                checked.append(i)
                continue
            src, src_json = rungs_dir / f"L{i:02d}.safetensors", rungs_dir / f"L{i:02d}.json"
            if not (src.is_file() and src_json.is_file()):
                raise Refusal(f"L{i:02d}: the receipt commits it but {src} / {src_json} is missing")
            doc = json.loads(src_json.read_text(encoding="utf-8"))
            if doc.get("file_sha256") != want:
                raise Refusal(f"L{i:02d}: {src_json} file_sha256 {doc.get('file_sha256')} != receipt {want}")
            if i != have:
                raise Refusal(f"L{i:02d}: canonical has L00..L{have - 1:02d}; refusing a gap")
            got = sha256_file(src)
            if got != want:
                raise Refusal(f"L{i:02d}: {src} hashes {got}, the receipt says {want}")
            json_sha = sha256_file(src_json)
            for s, d in ((src, canonical / src.name), (src_json, canonical / src_json.name)):
                if d.exists():
                    raise Refusal(f"{d} exists but is not in {SUMS}; inspect by hand")
                cp_n(s, d)
            if sha256_file(canonical / src.name) != want or sha256_file(canonical / src_json.name) != json_sha:
                raise Refusal(f"L{i:02d}: the copy does not hash like its source")
            state[i] = {"safetensors": want, "json": json_sha}
            prov["layers"][f"L{i:02d}"] = {"source_receipt": str(receipt_path), "source_receipt_sha256": receipt_sha,
                                           "source_file": str(src), "safetensors_sha256": want, "json_sha256": json_sha,
                                           "bytes": src.stat().st_size, "input_chain": doc.get("input_chain"),
                                           "tool": doc.get("tool"), "label": label,
                                           "copied_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            have += 1
            added.append(i)
            log(f"L{i:02d} copied + verified {want[:16]} ({label or receipt_path})")
        elif rec.get("import_provenance"):               # imported by this attempt: lineage check
            pin = rec["import_provenance"].get("old_record_pin", {}).get("sha256")
            if i not in state:
                raise Refusal(f"L{i:02d}: the attempt imported it but the canonical dir does not hold it")
            if pin != state[i]["json"]:
                raise Refusal(f"L{i:02d}: the attempt imported a rung record {pin} != canonical {state[i]['json']}")
            lineage.append(i)
        elif rec.get("imported_from_prefix"):            # 1.1.0: the hardened trainer's import record
            ifp = rec["imported_from_prefix"]
            if i not in state:
                raise Refusal(f"L{i:02d}: the attempt imported it from a prefix but the canonical dir does not hold it")
            if not ifp.get("dir") or Path(str(ifp["dir"])).resolve() != canonical.resolve():
                raise Refusal(f"L{i:02d}: imported from prefix dir {ifp.get('dir')!r}, not this canonical dir {canonical}")
            if ifp.get("json_sha256") != state[i]["json"]:
                raise Refusal(f"L{i:02d}: the attempt imported a rung record {ifp.get('json_sha256')} != canonical "
                              f"{state[i]['json']}")
            lineage.append(i)
        else:
            break                                         # an untrained / unfinished layer ends the committed prefix
    sums_sha = write_sums(canonical, state)
    prov["syncs"].append({"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "receipt": str(receipt_path),
                          "receipt_sha256": receipt_sha, "added": added, "already_equal": checked,
                          "lineage_checked": lineage, "sha256sums_sha256": sums_sha, "label": label})
    write_atomic(prov_path, json.dumps(prov, indent=1, sort_keys=True))
    os.chmod(prov_path, 0o644)
    n = contiguous(state)
    log(f"SYNC {label or receipt_path}: added {added} equal {checked} lineage {lineage}; canonical L00..L{n - 1:02d} "
        f"({n} layers); {SUMS} {sums_sha}")
    return {"layers": n, "added": added, "already_equal": checked, "lineage_checked": lineage, "sha256sums_sha256": sums_sha}


def verify(canonical: Path, expect_layers: int | None = None) -> dict[str, Any]:
    state = canonical_state(canonical)
    if not state:
        raise Refusal(f"{canonical}: no {SUMS}")
    n = contiguous(state)
    for i in range(n):
        for kind in ("safetensors", "json"):
            got = sha256_file(canonical / f"L{i:02d}.{kind}")
            if got != state[i][kind]:
                raise Refusal(f"L{i:02d}.{kind} hashes {got}, {SUMS} says {state[i][kind]}")
        doc = json.loads((canonical / f"L{i:02d}.json").read_text(encoding="utf-8"))
        if doc.get("file_sha256") != state[i]["safetensors"] or int(doc.get("text_layer", -1)) != i:
            raise Refusal(f"L{i:02d}.json does not describe its safetensors")
    extra = sorted(p.name for p in canonical.iterdir() if RUNG_RE.match(p.name) and int(p.name[1:3]) >= n)
    if extra:
        raise Refusal(f"rung files outside {SUMS}: {extra}")
    if expect_layers is not None and n != expect_layers:
        raise Refusal(f"{n} canonical layers, expected {expect_layers}")
    sums_sha = sha256_file(canonical / SUMS)
    log(f"VERIFY OK: L00..L{n - 1:02d} ({n} layers, {2 * n} files re-hashed); {SUMS} {sums_sha}")
    return {"layers": n, "sha256sums_sha256": sums_sha, "rung_sha256": [state[i]["safetensors"] for i in range(n)]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--canonical", required=True, type=Path)
    s.add_argument("--receipt", required=True, type=Path)
    s.add_argument("--rungs-dir", required=True, type=Path)
    s.add_argument("--label", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--canonical", required=True, type=Path)
    v.add_argument("--expect-layers", type=int, default=None)
    v.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "sync":
        sync(args.canonical, args.receipt, args.rungs_dir, args.label)
    else:
        res = verify(args.canonical, args.expect_layers)
        if args.json:
            print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
