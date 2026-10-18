#!/usr/bin/env python3
"""d1_seal_verify.py -- read-only verification of the HARDENED D1 run's final delta, before it is registered.

Lane d1seal, Claude, 2026-10-14.  The successor of Codex's verify_final_delta_v35.py / verify_final_delta_c36.py
(lineage: /mnt/lexar/omni-campaign/d1-readback-c36-r2-20261014-bad11a92/), rebuilt for the run that actually
finishes D1: d1_trunk_recovery >= 1.3.0 under d1_hardened_entry (systemd unit, no allocator, no container), the
canonical prefix L00..L{P-1} imported from d1_canonical_rungs' directory and L{P}..L63 trained by this run.
Those scripts are bound to Codex's allocator assignment, execution id and binding ledger, none of which exist
here, so the checks are re-derived from the hardened run's own records.  Standalone: stdlib + numpy, no torch, no
model load, nothing written except --out (+ its .sha256 sidecar).

    d1_seal_verify.py --run-dir RUN --canonical CAN --delta-dir DELTA --base-dir CLONE \\
        --base-sealed-sha256 SHA --out RECEIPT.json [--expect-layers 64]

CHECKS (any failure: exit 3, "REFUSED: ..." and a receipt with verified=false; nothing else written)
  1 terminal success  RUN/runtime-status.json: completed, state succeeded, exit_code 0, no alert, systemd's
                      SERVICE_RESULT=success recorded by the ExecStopPost finalize, launch_id == RUN/launch.json's.
                      (`completed` alone also means failure, and a transient unit's Result=success is printed
                      while it still runs, so neither is read alone.)
  2 receipt           trainer/d1_receipt.json == its .sha256 sidecar; run_identity_sha256 == sha256_json(identity);
                      the full-run verdict; 64 layers, text_layer i at index i, every one trained; layers
                      0..P-1 imported from THIS canonical dir (json sha == its SHA256SUMS line; P == receipt prefix
                      record), layers P..63 trained here with a rung {file, sha256}.
  3 rungs / chain     every new rung file hashes to the receipt; its .json names that sha, layer and block; the
                      chain sha256_json([chain, i, rung sha]) from run_identity is recomputed through all 64 layers
                      and each new rung's input_chain equals it (the trainer's own resume rule).
  4 canonical         d1_canonical_rungs.verify(CAN, 64): every file re-hashed; L{P}..L63 there == this run's rungs;
                      L00..L{P-1} == the receipt's prefix record rung shas; every rung file's tensor payloads hash to
                      the entries of its record (so step 5's "delta entry == rung entry" is a byte equality).
  5 delta manifest    DELTA/D1_delta.json == its sidecar; schema; producer == the receipt's tool sha; trained layers
                      0..63; 256 entries, each == the canonical rung record's entry + text_layer (key, base tensor,
                      part, slice, shape, dtype, payload sha, base tensor sha); parameters == the sum of the entries;
                      the receipt's delta summary names this file / manifest.
  6 delta payload     header keys == entries, BF16, shapes, spans contiguous and complete; every entry's payload
                      bytes hash to its sha; the whole file hashes to file_sha256; every element FINITE (bf16 exponent
                      bits not all ones); all-zero tensors counted (diagnostic).
  7 base              the manifest's base is --base-dir (mode full, config / index sha == the live files); the
                      sealed clone digest --base-sealed-sha256 is RE-DERIVED from the run's per-shard sha256 (hashed at
                      the end of the run) + the live small files (oc_seal.digest_of_manifest), so the delta is pinned to
                      the registered bytes without re-reading 55 GB (`delta-compose --deep` re-hashes them anyway).
Finite fit numbers (loss, relL2, KL) are diagnostics (owner policy 2026-10-06) and are only copied into the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "d1_seal_verify"
TOOL_VERSION = "1.0.0"   # 2026-10-14: first version, lane d1seal (successor of verify_final_delta_v35 / _c36)
SCHEMA = "omni-campaign.d1-seal-verify.v1"
DELTA_SCHEMA = "omni-campaign.d1-delta.v1"
TRAINER = "d1_trunk_recovery"
VERDICT = "{n} layer(s) trained, streams finite through text layer {last}; delta written"
ENTRY_FIELDS = ("key", "base_tensor", "part", "slice", "shape", "dtype", "sha256", "base_tensor_sha256")
PARTS = {"gate_trunk", "up_trunk", "down_trunk", "whole"}
MAX_JSON = 64 * 1024 ** 2
HERE = Path(__file__).resolve().parent
LEDGER_ROOT = Path(os.environ.get("OC_LEDGER_ROOT", "/home/cereal/omni-campaign/ledger"))


class Refusal(Exception):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise Refusal(message)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{TOOL_NAME}] {message}", flush=True)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(doc: Any) -> str:
    """d1_trunk_recovery.sha256_json, byte for byte (the chain and the run identity are made with it)."""
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def version_tuple(text: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(text).split("-")[0].split("."))
    except ValueError:
        return (0,)


def read_json(path: Path) -> tuple[Any, bytes]:
    require(path.is_file(), f"{path} is missing")
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_JSON, f"{path} is empty or larger than {MAX_JSON} bytes")
    return json.loads(raw), raw


def sidecar_matches(path: Path, raw: bytes) -> bool:
    side = path.with_name(path.name + ".sha256")
    return side.is_file() and side.read_text(encoding="utf-8").split()[:1] == [sha256_bytes(raw)]


# ---------------------------------------------------------------------------------------------------- 1 status
def check_status(run: Path) -> dict[str, Any]:
    status, raw = read_json(run / "runtime-status.json")
    launch, _ = read_json(run / "launch.json")
    require(status.get("completed") is True, f"runtime-status: completed={status.get('completed')!r}")
    require(status.get("state") == "succeeded" and status.get("exit_code") == 0,
            f"runtime-status: state {status.get('state')!r} exit {status.get('exit_code')!r} (completed also means failure)")
    require(not status.get("alert"), f"runtime-status carries an alert: {status.get('alert')}")
    systemd = status.get("systemd") or {}
    require(systemd.get("service_result") == "success",
            f"systemd finalize recorded SERVICE_RESULT={systemd.get('service_result')!r} (ExecStopPost must have run)")
    require(launch.get("launch_id") and status.get("launch_id") == launch.get("launch_id"),
            f"runtime-status launch {status.get('launch_id')!r} != launch.json {launch.get('launch_id')!r}")
    return {"runtime_status_sha256": sha256_bytes(raw), "state": status["state"], "exit_code": status["exit_code"],
            "attempt": status.get("attempt"), "finished_at": status.get("finished_at"), "launch_id": launch["launch_id"],
            "unit": launch.get("unit"), "systemd": systemd, "earlier_attempts": len(status.get("history") or [])}


# ---------------------------------------------------------------------------------------------------- 2 + 3 receipt
def check_receipt(run: Path, canonical: Path, n_layers: int) -> tuple[dict[str, Any], dict[str, Any], list[str], int]:
    """(receipt, record, per-layer rung sha, prefix layers)."""
    path = run / "trainer" / "d1_receipt.json"
    receipt, raw = read_json(path)
    require(sidecar_matches(path, raw), f"{path} does not match its .sha256 sidecar")
    require(receipt.get("schema") == "omni-campaign.d1-trunk-recovery.v1", f"receipt schema {receipt.get('schema')!r}")
    tool = receipt.get("tool") or {}
    require(tool.get("name") == TRAINER and version_tuple(tool.get("version")) >= (1, 3, 0),
            f"receipt tool {tool} is not {TRAINER} >= 1.3.0 (the hardened trainer)")
    require(receipt.get("run_identity_sha256") == sha256_json(receipt.get("identity")),
            "run_identity_sha256 is not sha256_json(identity)")
    require(receipt.get("verdict") == VERDICT.format(n=n_layers, last=n_layers - 1),
            f"receipt verdict {receipt.get('verdict')!r} is not the full-run verdict")
    require(receipt.get("child_text_blocks") == n_layers and receipt.get("stop_after_layer") == n_layers - 1
            and receipt.get("train_layers") == list(range(n_layers)), "receipt is not a full 64-layer selection")
    layers = receipt.get("layers")
    require(isinstance(layers, list) and len(layers) == n_layers, f"receipt holds {len(layers or [])} layers, not {n_layers}")
    prefix = receipt.get("prefix") or {}
    p = int(prefix.get("layers") or 0)
    require(Path(str(prefix.get("dir", canonical))).resolve() == canonical.resolve() or p == 0,
            f"the run imported its prefix from {prefix.get('dir')!r}, not {canonical}")
    sums = canonical_sums(canonical)
    chain = receipt["run_identity_sha256"]
    shas: list[str] = []
    new_rungs: list[dict[str, Any]] = []
    for i, rec in enumerate(layers):
        require(isinstance(rec, dict) and rec.get("text_layer") == i and rec.get("trained") is True,
                f"receipt layer {i}: text_layer {rec.get('text_layer')!r} trained {rec.get('trained')!r}")
        if i < p:
            ifp = rec.get("imported_from_prefix") or {}
            require(Path(str(ifp.get("dir"))).resolve() == canonical.resolve(), f"L{i:02d}: not imported from {canonical}")
            require(ifp.get("json_sha256") == sums.get(i, {}).get("json"),
                    f"L{i:02d}: imported record {ifp.get('json_sha256')} != canonical {sums.get(i, {}).get('json')}")
            require(ifp.get("sha256sums_sha256") == prefix.get("sha256sums_sha256"),
                    f"L{i:02d}: import pin {ifp.get('sha256sums_sha256')} != the run's prefix pin")
            sha = (prefix.get("rung_sha256") or [None] * p)[i]
            require(sha and sha == sums.get(i, {}).get("safetensors"), f"L{i:02d}: prefix rung sha {sha} != canonical")
        else:
            require("imported_from_prefix" not in rec, f"L{i:02d}: imported past the prefix ({p} layers)")
            rung = rec.get("rung") or {}
            sha = rung.get("sha256")
            rpath = run / "trainer" / "rungs" / f"L{i:02d}.safetensors"
            jpath = rpath.with_suffix(".json")
            require(isinstance(sha, str) and len(sha) == 64, f"L{i:02d}: no rung sha in the receipt")
            require(rpath.is_file() and jpath.is_file(), f"L{i:02d}: {rpath} / {jpath.name} missing")
            got = sha256_file(rpath)
            require(got == sha, f"L{i:02d}: {rpath} hashes {got}, the receipt says {sha}")
            doc, jraw = read_json(jpath)
            require(doc.get("file_sha256") == sha and doc.get("text_layer") == i and doc.get("child_block") == rec.get("child_block"),
                    f"L{i:02d}: rung record does not describe its file / layer / block")
            require(doc.get("input_chain") == chain, f"L{i:02d}: rung input_chain != the recomputed chain")
            new_rungs.append({"layer": i, "sha256": sha, "json_sha256": sha256_bytes(jraw)})
        shas.append(sha)
        chain = sha256_json([chain, i, sha])
    record = {"receipt": str(path), "receipt_sha256": sha256_bytes(raw), "tool": tool, "run_identity_sha256":
              receipt["run_identity_sha256"], "verdict": receipt["verdict"], "prefix_layers": p,
              "prefix_sha256sums_sha256": prefix.get("sha256sums_sha256"), "new_rungs": len(new_rungs),
              "final_chain": chain, "head_kl": receipt.get("head_kl"), "seconds": receipt.get("seconds")}
    record["_new_rungs"] = new_rungs
    return receipt, record, shas, p


def canonical_sums(canonical: Path) -> dict[int, dict[str, str]]:
    sys.path.insert(0, str(HERE))
    import d1_canonical_rungs as cr  # noqa: E402
    return cr.canonical_state(canonical)


def check_canonical(canonical: Path, n_layers: int, shas: list[str], new_rungs: list[dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import d1_canonical_rungs as cr  # noqa: E402
    try:
        res = cr.verify(canonical, expect_layers=n_layers)
    except SystemExit as exc:
        raise Refusal(f"canonical verify: {exc}") from exc
    require(res["rung_sha256"] == shas, "canonical rung shas differ from the run's 64-layer chain")
    state = cr.canonical_state(canonical)
    for r in new_rungs:
        require(state[r["layer"]]["json"] == r["json_sha256"], f"L{r['layer']:02d}: canonical json != this run's rung json")
    for i in range(n_layers):   # each rung file's tensors ARE the bytes its record names (the delta is checked against these)
        doc, _ = read_json(canonical / f"L{i:02d}.json")
        got = span_hashes(canonical / f"L{i:02d}.safetensors")
        want = {e["key"]: e["sha256"] for e in doc.get("entries") or []}
        require(got == want, f"canonical L{i:02d}: tensor payload hashes differ from its record's entries")
    return {"dir": str(canonical), "layers": res["layers"], "sha256sums_sha256": res["sha256sums_sha256"],
            "rung_tensor_hashes_verified": True}


def span_hashes(path: Path) -> dict[str, str]:
    """{tensor: sha256 of its payload bytes} of one safetensors file, streamed in offset order."""
    out: dict[str, str] = {}
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        for lo, hi, key in sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in header.items()):
            fh.seek(base + lo)
            h, remaining = hashlib.sha256(), hi - lo
            while remaining:
                chunk = fh.read(min(1 << 26, remaining))
                require(chunk, f"{path}: {key} truncated")
                h.update(chunk)
                remaining -= len(chunk)
            out[key] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------------------------------- 5 + 6 delta
def expected_entries(canonical: Path, n_layers: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i in range(n_layers):
        doc, _ = read_json(canonical / f"L{i:02d}.json")
        entries = doc.get("entries")
        require(isinstance(entries, list) and len(entries) == 4, f"canonical L{i:02d}.json: {len(entries or [])} entries, not 4")
        for e in entries:
            require(e.get("part") in PARTS, f"canonical L{i:02d}.json: part {e.get('part')!r}")
            key = e["key"]
            require(key not in out, f"{key} appears in two rung records")
            out[key] = {**{f: e.get(f) for f in ENTRY_FIELDS}, "text_layer": i}
    return out


def check_manifest(delta: Path, receipt: dict[str, Any], expected: dict[str, dict[str, Any]], n_layers: int) -> tuple[dict[str, Any], str]:
    path = delta / "D1_delta.json"
    manifest, raw = read_json(path)
    require(sidecar_matches(path, raw), f"{path} does not match its .sha256 sidecar")
    require(manifest.get("schema") == DELTA_SCHEMA, f"delta schema {manifest.get('schema')!r}")
    tool = manifest.get("tool") or {}
    require(tool.get("name") == TRAINER and tool.get("sha256") == (receipt.get("tool") or {}).get("sha256"),
            f"delta producer {tool} is not the receipt's trainer")
    require(manifest.get("trained_text_layers") == list(range(n_layers)), "delta does not carry every text layer")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == 4 * n_layers, f"delta has {len(entries or [])} entries")
    seen = set()
    for e in entries:
        key = e.get("key")
        require(key in expected and key not in seen, f"foreign or duplicate delta entry {key!r}")
        seen.add(key)
        got = {**{f: e.get(f) for f in ENTRY_FIELDS}, "text_layer": e.get("text_layer")}
        require(got == expected[key], f"{key}: delta entry differs from the canonical rung record")
        require(set(e) == set(ENTRY_FIELDS) | {"text_layer"}, f"{key}: unexpected entry fields {sorted(set(e) - set(ENTRY_FIELDS))}")
    require(seen == set(expected), "delta is missing canonical rung entries")
    params = sum(math.prod(e["shape"]) for e in entries)
    require(manifest.get("parameters") == params, f"manifest parameters {manifest.get('parameters')} != entries {params}")
    require(manifest.get("reader") == f"{TRAINER}.apply_delta (slice overwrite of the base tensor)", "delta reader semantics differ")
    file = delta / "D1_delta.safetensors"
    summary = receipt.get("delta") or {}
    want = {"file": str(file.absolute()), "file_sha256": manifest.get("file_sha256"), "bytes": manifest.get("bytes"),
            "parameters": params, "manifest_sha256": sha256_bytes(raw)}
    for k, v in want.items():
        if k == "file":
            require(Path(str(summary.get("file"))).resolve() == file.resolve(), f"receipt delta file {summary.get('file')} != {file}")
        else:
            require(summary.get(k) == v, f"receipt delta {k} {summary.get(k)!r} != {v!r}")
    return manifest, sha256_bytes(raw)


def check_payload(delta: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    path = delta / "D1_delta.safetensors"
    entries = {e["key"]: e for e in manifest["entries"]}
    size = path.stat().st_size
    require(size == manifest.get("bytes"), f"{path} is {size} bytes, the manifest says {manifest.get('bytes')}")
    whole = hashlib.sha256()
    nonfinite = zero_tensors = 0
    worst: list[str] = []
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        require(len(prefix) == 8, "short safetensors prefix")
        n = int.from_bytes(prefix, "little")
        require(0 < n <= 16 * 1024 ** 2 and n <= size - 8, f"header length {n} out of bounds")
        hraw = fh.read(n)
        whole.update(prefix + hraw)
        header = json.loads(hraw)
        meta = header.pop("__metadata__", {})
        require(meta.get("schema") == DELTA_SCHEMA, f"delta header metadata {meta}")
        require(set(header) == set(entries), "delta header keys != manifest entries")
        spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in header.items())
        payload = size - 8 - n
        end = 0
        for lo, hi, key in spans:
            spec, entry = header[key], entries[key]
            require(lo == end, f"{key}: gap or overlap at {lo} (expected {end})")
            require(spec["dtype"] == "BF16" and entry["dtype"] == "bfloat16", f"{key}: dtype {spec['dtype']} / {entry['dtype']}")
            require(spec["shape"] == entry["shape"] and hi - lo == math.prod(spec["shape"]) * 2, f"{key}: shape / span differ")
            own = hashlib.sha256()
            remaining = hi - lo
            bad = 0
            nonzero = False
            while remaining:
                chunk = fh.read(min(1 << 26, remaining))
                require(chunk, f"{key}: payload truncated")
                own.update(chunk)
                whole.update(chunk)
                bits = np.frombuffer(chunk, dtype="<u2")
                bad += int(np.count_nonzero((bits & 0x7F80) == 0x7F80))
                nonzero = nonzero or bool(np.any(bits & 0x7FFF))
                remaining -= len(chunk)
            require(own.hexdigest() == entry["sha256"], f"{key}: payload hash differs from its entry")
            if bad:
                nonfinite += bad
                worst.append(f"{key}:{bad}")
            zero_tensors += 0 if nonzero else 1
            end = hi
        require(end == payload, f"payload {payload} bytes, spans cover {end}")
        require(fh.read(1) == b"", "trailing bytes after the payload")
    require(whole.hexdigest() == manifest["file_sha256"], "whole-file sha256 differs from the manifest")
    require(nonfinite == 0, f"{nonfinite} non-finite element(s) in the delta: {worst[:6]}")
    return {"file": str(path), "bytes": size, "sha256": manifest["file_sha256"], "tensors": len(entries),
            "elements": sum(math.prod(e["shape"]) for e in entries.values()), "nonfinite_elements": 0,
            "all_zero_tensors": zero_tensors, "every_entry_payload_hash_verified": True}


# ---------------------------------------------------------------------------------------------------- 7 base
def check_base(manifest: dict[str, Any], base_dir: Path, sealed: str) -> dict[str, Any]:
    base = manifest.get("base") or {}
    require(Path(str(base.get("path"))).resolve() == base_dir.resolve(), f"D1 base {base.get('path')} is not {base_dir}")
    require(base.get("mode") == "full", f"D1 base digest mode {base.get('mode')!r} (full is required to re-derive the seal)")
    for name, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        require(base.get(key) == sha256_file(base_dir / name), f"{base_dir}/{name} changed since D1 recorded {key}")
    files = base.get("files") or {}
    shards = sorted(p.name for p in base_dir.glob("*.safetensors"))
    require(sorted(files) == shards and shards, f"D1 recorded shards {sorted(files)[:3]} != on disk {shards[:3]}")
    for name in shards:
        st = (base_dir / name).stat()
        require(files[name].get("bytes") == st.st_size and isinstance(files[name].get("sha256"), str),
                f"{name}: size changed or no sha256 recorded")
    sys.path.insert(0, str(LEDGER_ROOT))
    from oc_seal import _digest_files, digest_of_manifest  # noqa: E402
    entries = []
    for path in _digest_files(base_dir):
        rel = path.relative_to(base_dir).as_posix()
        entries.append({"path": rel, "sha256": files[rel]["sha256"] if rel in files else sha256_file(path)})
    derived = digest_of_manifest(entries)
    require(derived == sealed, f"re-derived base digest {derived[:16]} != the sealed clone digest {sealed[:16]}")
    return {"dir": str(base_dir), "sealed_directory_sha256": sealed, "rederived_from_run_shard_hashes": True,
            "shards": len(shards), "config_sha256": base.get("config_sha256"), "index_sha256": base.get("index_sha256")}


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run, canonical, delta, base_dir = (Path(a).absolute() for a in (args.run_dir, args.canonical, args.delta_dir, args.base_dir))
    n = int(args.expect_layers)
    out: dict[str, Any] = {"schema": SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(Path(__file__))},
                           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out["status"] = check_status(run)
    log(f"1 terminal success: {out['status']['state']} exit {out['status']['exit_code']} launch {out['status']['launch_id']}")
    receipt, rec, shas, p = check_receipt(run, canonical, n)
    new_rungs = rec.pop("_new_rungs")
    out["receipt"] = rec
    log(f"2-3 receipt + chain: prefix {p} imported, {len(new_rungs)} trained here, chain {rec['final_chain'][:16]}")
    out["canonical"] = check_canonical(canonical, n, shas, new_rungs)
    log(f"4 canonical: {out['canonical']['layers']} layers, SHA256SUMS {out['canonical']['sha256sums_sha256'][:16]}")
    manifest, msha = check_manifest(delta, receipt, expected_entries(canonical, n), n)
    out["delta_manifest"] = {"path": str(delta / "D1_delta.json"), "sha256": msha, "entries": len(manifest["entries"]),
                             "parameters": manifest["parameters"]}
    log(f"5 delta manifest {msha[:16]}: {len(manifest['entries'])} entries, {manifest['parameters']:,} parameters")
    out["delta_payload"] = check_payload(delta, manifest)
    log(f"6 delta payload {out['delta_payload']['sha256'][:16]}: every entry hash verified, 0 non-finite, "
        f"{out['delta_payload']['all_zero_tensors']} all-zero tensor(s)")
    out["base"] = check_base(manifest, base_dir, args.base_sealed_sha256)
    log(f"7 base {base_dir.name}: sealed {args.base_sealed_sha256[:16]} re-derived from the run's shard hashes")
    out.update(verified=True, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               scope="Terminal success of the hardened D1 run, its 64-layer rung chain, the canonical rung dir and the "
                     "serialized delta (every byte hashed, every element finite). No model was loaded and no quality "
                     "claim is made: functional checks only (owner policy 2026-10-06).")
    return out


def write_receipt(path: Path, doc: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--delta-dir", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--base-sealed-sha256", required=True)
    ap.add_argument("--expect-layers", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    try:
        doc = verify(args)
    except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        msg = f"{type(exc).__name__}: {exc}" if not isinstance(exc, Refusal) else str(exc)
        print(f"REFUSED: {msg}", flush=True)
        write_receipt(out, {"schema": SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION}, "verified": False,
                            "error": msg, "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return 3
    digest = write_receipt(out, doc)
    print(f"D1 SEAL VERIFY OK {out} sha256 {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
