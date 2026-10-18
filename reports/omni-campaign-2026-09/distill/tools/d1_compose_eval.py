#!/usr/bin/env python3
"""d1_compose_eval.py -- a TRANSIENT composed copy of base + delta for the tools that only read a Chimera dir.

Lane d1seal, Claude, 2026-10-14.  The D1 delta is registered as a delta (oc_checkpoint.py register --delta-base) and
is never materialised as a checkpoint; the P1 reader composes it per tensor for the trainer.  But the functional
check (s7_omni1_check) and the frozen review (run_omni_frozen_review) read a `chimera_omni.checkpoint.v1` DIRECTORY,
and neither reads a plan.  A copy-free overlay is not possible here either: D1 rewrites FFN tensors in 12 of the 14
shards, and /mnt/lexar is exFAT (no symlinks, no hardlinks; the registry refuses symlinks anyway).  So this tool
writes ONE composed copy into declared scratch (STAGE_D_RULINGS "Transient scratch": never registered, removed with
plain rm after use) and proves it is exactly base + delta:

    compose  --plan PLAN.json --out DIR [--scratch-root /mnt/lexar/omni-campaign/scratch] [--no-readback]
        PLAN.json is `oc_checkpoint.py delta-compose --plan-out` (sealed, schema neural-forge-omni-delta-plan-v1).
        Refused (exit 3, the partial dir removed): a plan whose seal fails, an op other than `replace` (lora / scale
        compose belongs to the P1 reader and the D4 merge), a delta file whose sha256 is not the plan's, a base tensor
        whose bytes are not its index sha256 pin, a delta tensor of another dtype / shape than its region, an --out
        outside --scratch-root or already present, a root whose re-derived directory digest (every byte read here)
        is not the plan's sealed root digest, a written shard that does not read back to the bytes written.
        Every shard keeps its header byte for byte (shapes and dtypes do not change); the index keeps its weight_map
        and total_size and gets the composed tensors' sha256 pins (chimera's dtype + shape + bytes digest), so the
        readers' sha-verified loads check the composed bytes.  Receipt: DIR.compose.json (+ .sha256), beside DIR.
    split    --model DIR --budgets-gib 21.5,22.5,22.5 [--head-extra-gib 2.5] [--block-extra-gib 0.0]
        The --split for run_omni_frozen_review on len(budgets) cards, head (embed / lm_head / norm / media heads) on
        the first: the contiguous block partition with the lowest worst-card fill, from the shard headers.  Prints
        `SPLIT a,b,c` and a JSON line; exit 4 when no partition fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "d1_compose_eval"
TOOL_VERSION = "1.0.0"   # 2026-10-14: first version, lane d1seal
PLAN_SCHEMA = "neural-forge-omni-delta-plan-v1"
INDEX = "model.safetensors.index.json"
ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
            "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
LEDGER_ROOT = Path(os.environ.get("OC_LEDGER_ROOT", "/home/cereal/omni-campaign/ledger"))
CHUNK = 1 << 26


class Refusal(Exception):
    pass


def require(ok: Any, message: str) -> None:
    if not ok:
        raise Refusal(message)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{TOOL_NAME}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def chimera_tensor_sha256(dtype: str, shape: list[int], buffer: bytes) -> str:
    """chimera_omni.importing.safetensors_io._tensor_sha256: dtype + compact shape JSON + raw bytes."""
    h = hashlib.sha256()
    h.update(dtype.encode("ascii"))
    h.update(json.dumps([int(x) for x in shape], separators=(",", ":")).encode("ascii"))
    h.update(buffer)
    return h.hexdigest()


def read_header(path: Path) -> tuple[dict[str, Any], bytes, int]:
    """(header without __metadata__, the raw 8-byte length + header bytes, payload base offset)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        require(len(prefix) == 8, f"{path}: short safetensors prefix")
        n = int.from_bytes(prefix, "little")
        require(0 < n <= 64 * 1024 ** 2, f"{path}: header length {n}")
        raw = fh.read(n)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, prefix + raw, 8 + n


def np_view(buffer: bytes | bytearray, dtype: str, shape: list[int]) -> Any:
    import numpy as np
    return np.frombuffer(buffer, dtype=np.dtype(f"<u{ITEMSIZE[dtype]}")).reshape(shape)


def region(shape: list[int], spec: Any) -> tuple[tuple[slice, ...], list[int]]:
    """oc_checkpoint._region: `[[dim, start, stop], ...]` -> numpy slices and the region shape (absent = whole)."""
    slices = [slice(0, int(n)) for n in shape]
    for item in spec or []:
        dim, start, stop = (int(v) for v in item)
        require(0 <= dim < len(shape) and 0 <= start < stop <= int(shape[dim]), f"slice {item} outside {shape}")
        slices[dim] = slice(start, stop)
    return tuple(slices), [s.stop - s.start for s in slices]


# ---------------------------------------------------------------------------------------------------- compose
def load_plan(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(LEDGER_ROOT))
    from oc_seal import OcSealError, verify_seal  # noqa: E402
    plan = json.loads(path.read_text(encoding="utf-8"))
    try:
        verify_seal(plan)
    except OcSealError as exc:
        raise Refusal(f"{path}: {exc}") from exc
    require(plan.get("schema") == PLAN_SCHEMA, f"{path} is not a {PLAN_SCHEMA}")
    require(plan.get("deltas"), f"{path} composes no delta")
    return plan


def delta_tensor(path: Path, header_cache: dict[str, Any], name: str) -> tuple[str, list[int], bytes]:
    if str(path) not in header_cache:
        header_cache[str(path)] = read_header(path)
    header, _raw, base = header_cache[str(path)]
    require(name in header, f"{path} has no tensor {name!r}")
    spec = header[name]
    lo, hi = spec["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + lo)
        buf = fh.read(hi - lo)
    require(len(buf) == hi - lo, f"{path}: {name} truncated")
    return spec["dtype"], list(spec["shape"]), buf


def compose(plan_path: Path, out: Path, scratch_root: Path, readback: bool = True) -> dict[str, Any]:
    plan = load_plan(plan_path)
    out = out.absolute()
    require(scratch_root.resolve() in out.resolve().parents, f"--out {out} is not under the declared scratch {scratch_root}")
    require(not out.exists(), f"{out} exists; a composed copy is never rewritten (rm it first)")
    root = Path(plan["root"]["directory"])
    plan_files = {f["path"]: f for f in plan["root"]["files"]}
    index = json.loads((root / INDEX).read_text(encoding="utf-8"))
    pins: dict[str, str] = dict((index.get("metadata") or {}).get("tensor_sha256") or {})
    weight_map: dict[str, str] = index["weight_map"]

    # every replace, by target tensor, in chain order then recipe order
    ops: dict[str, list[dict[str, Any]]] = {}
    delta_files: dict[str, Path] = {}
    for link in plan["deltas"]:
        directory = Path(link["directory"])
        holder: dict[str, Path] = {}
        for f in link["files"]:
            p = directory / f["path"]
            got = sha256_file(p)
            require(got == f["sha256"], f"{p} hashes {got[:16]}, the plan says {f['sha256'][:16]}")
            for name in read_header(p)[0]:
                require(name not in holder, f"delta tensor {name} is in two files of {link['checkpoint_id']}")
                holder[name] = p
        for number, entry in enumerate(link["entries"]):
            require(entry.get("op") == "replace", f"{link['checkpoint_id']} entry {number}: op {entry.get('op')!r} -- "
                                                  "only `replace` composes here (lora / scale: the P1 reader, the D4 merge)")
            require(entry["target"] in weight_map, f"target {entry['target']} is not a tensor of the root")
            require(entry.get("tensor") in holder, f"delta tensor {entry.get('tensor')!r} is in no file of {link['checkpoint_id']}")
            delta_files[entry["tensor"]] = holder[entry["tensor"]]
            ops.setdefault(entry["target"], []).append({**entry, "_file": holder[entry["tensor"]], "_link": link["checkpoint_id"]})
    log(f"plan {plan['receipt_sha'][:16]}: {sum(len(v) for v in ops.values())} replace op(s) on {len(ops)} tensor(s), "
        f"root {plan['root']['checkpoint_id']} {plan['root']['directory_sha256'][:16]}")

    partial = out.with_name(out.name + ".partial")
    require(not partial.exists(), f"{partial} exists (an earlier compose died); inspect and rm it")
    partial.mkdir(parents=True)
    headers: dict[str, Any] = {}
    shard_rows, changed, src_shas, out_shas = [], {}, {}, {}
    replaced_elements = 0
    try:
        sys.path.insert(0, str(LEDGER_ROOT))
        from oc_seal import _digest_files, digest_of_manifest  # noqa: E402
        for src in _digest_files(root):
            rel = src.relative_to(root).as_posix()
            dst = partial / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel == INDEX:
                src_shas[rel] = sha256_file(src)
                continue                                   # written last, with the composed pins
            if not rel.endswith(".safetensors") or not any(weight_map[t] == rel for t in ops):
                src_h = hashlib.sha256()
                with open(src, "rb") as fi, open(dst, "wb") as fo:
                    for chunk in iter(lambda: fi.read(CHUNK), b""):
                        src_h.update(chunk)
                        fo.write(chunk)
                src_shas[rel] = out_shas[rel] = src_h.hexdigest()
                if rel.endswith(".safetensors"):
                    shard_rows.append({"shard": rel, "patched_tensors": 0, "sha256": src_shas[rel]})
                continue
            header, raw_header, base = read_header(src)
            src_h, dst_h = hashlib.sha256(raw_header), hashlib.sha256(raw_header)
            patched = 0
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(raw_header)
                for lo, hi, name in sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in header.items()):
                    fi.seek(base + lo)
                    require(fi.tell() == base + lo, f"{rel}: seek")
                    if name not in ops:
                        remaining = hi - lo
                        while remaining:
                            chunk = fi.read(min(CHUNK, remaining))
                            require(chunk, f"{rel}: {name} truncated")
                            src_h.update(chunk)
                            dst_h.update(chunk)
                            fo.write(chunk)
                            remaining -= len(chunk)
                        continue
                    spec = header[name]
                    buf = bytearray(fi.read(hi - lo))
                    require(len(buf) == hi - lo, f"{rel}: {name} truncated")
                    src_h.update(buf)
                    old = chimera_tensor_sha256(spec["dtype"], spec["shape"], bytes(buf))
                    require(pins.get(name) in (None, old), f"{name}: base bytes {old[:16]} are not the index pin {str(pins.get(name))[:16]}")
                    arr = np_view(buf, spec["dtype"], spec["shape"])   # a writable view of buf (bytearray)
                    for entry in ops[name]:
                        slices, shape = region(spec["shape"], entry.get("slice"))
                        dtype, dshape, dbuf = delta_tensor(entry["_file"], headers, entry["tensor"])
                        require(dtype == spec["dtype"] and dshape == shape,
                                f"{name}: delta {entry['tensor']} is {dtype} {dshape}, the region is {spec['dtype']} {shape}")
                        arr[slices] = np_view(dbuf, dtype, dshape)
                        replaced_elements += int(math.prod(shape))
                    new = chimera_tensor_sha256(spec["dtype"], spec["shape"], bytes(buf))
                    changed[name] = {"shard": rel, "base_sha256": old, "composed_sha256": new, "ops": len(ops[name]),
                                     "unchanged_by_delta": old == new}
                    pins[name] = new
                    dst_h.update(buf)
                    fo.write(buf)
                    patched += 1
                require(fi.read(1) == b"", f"{rel}: trailing bytes after the last tensor")
            src_shas[rel], out_shas[rel] = src_h.hexdigest(), dst_h.hexdigest()
            shard_rows.append({"shard": rel, "patched_tensors": patched, "sha256": out_shas[rel], "base_sha256": src_shas[rel]})
            log(f"{rel}: {patched} tensor(s) composed")
        require(set(changed) == set(ops), f"targets never reached: {sorted(set(ops) - set(changed))[:3]}")
        # the root, every byte of which was just read, must be the plan's sealed root
        derived = digest_of_manifest([{"path": k, "sha256": v} for k, v in src_shas.items()])
        require(derived == plan["root"]["directory_sha256"],
                f"the root re-derives to {derived[:16]}, the plan seals {plan['root']['directory_sha256'][:16]}")
        for rel, f in plan_files.items():
            if f.get("sha256"):
                require(src_shas.get(rel) == f["sha256"], f"{rel}: {str(src_shas.get(rel))[:16]} != plan {f['sha256'][:16]}")
        index.setdefault("metadata", {})["tensor_sha256"] = dict(sorted(pins.items()))
        (partial / INDEX).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        out_shas[INDEX] = sha256_file(partial / INDEX)
        if readback:
            for rel, want in out_shas.items():
                got = sha256_file(partial / rel)
                require(got == want, f"{rel}: wrote {want[:16]}, reads back {got[:16]}")
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    composed = digest_of_manifest([{"path": k, "sha256": v} for k, v in out_shas.items()])
    return {"schema": "omni-campaign.d1-compose-eval.v1", "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
            "sha256": sha256_file(Path(__file__))}, "plan": str(plan_path), "plan_receipt_sha": plan["receipt_sha"],
            "plan_file_sha256": sha256_file(plan_path), "checkpoint_id": plan["checkpoint_id"],
            "root": {k: plan["root"][k] for k in ("checkpoint_id", "directory", "directory_sha256")},
            "root_rederived_from_every_byte": True, "deltas": [d["checkpoint_id"] for d in plan["deltas"]],
            "out": str(out), "out_directory_sha256": composed, "readback_verified": bool(readback),
            "tensors_composed": len(changed), "replaced_elements": replaced_elements,
            "tensors_unchanged_by_delta": sum(1 for c in changed.values() if c["unchanged_by_delta"]),
            "shards": shard_rows, "changed": changed, "transient": "declared scratch; never registered; rm -rf after use",
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# ---------------------------------------------------------------------------------------------------- split
def block_bytes(model: Path) -> tuple[list[int], int]:
    index = json.loads((model / INDEX).read_text(encoding="utf-8"))
    sizes: dict[str, int] = {}
    for shard in sorted(set(index["weight_map"].values())):
        header, _raw, _base = read_header(model / shard)
        for name, spec in header.items():
            sizes[name] = spec["data_offsets"][1] - spec["data_offsets"][0]
    blocks: dict[int, int] = {}
    head = 0
    for name, n in sizes.items():
        if name.startswith("blocks."):
            i = int(name.split(".", 2)[1])
            blocks[i] = blocks.get(i, 0) + n
        else:
            head += n
    require(sorted(blocks) == list(range(len(blocks))), "block indices are not contiguous from 0")
    return [blocks[i] for i in range(len(blocks))], head


def best_split(blocks: list[int], head: int, budgets: list[float], head_extra: float, block_extra: float) -> dict[str, Any] | None:
    """Contiguous partition into len(budgets) non-empty segments, head on the first; minimise the worst fill."""
    gib = float(1 << 30)
    n, k = len(blocks), len(budgets)
    prefix = [0]
    for b in blocks:
        prefix.append(prefix[-1] + b)
    best = None

    def rec(start: int, card: int, cuts: list[int]) -> None:
        nonlocal best
        if card == k - 1:
            segs = cuts + [n - start]
            loads, pos = [], 0
            for c, s in enumerate(segs):
                load = (prefix[pos + s] - prefix[pos]) / gib + block_extra + (head / gib + head_extra if c == 0 else 0.0)
                loads.append(load)
                pos += s
            fill = max(l / b for l, b in zip(loads, budgets))
            if fill <= 1.0 and (best is None or fill < best["worst_fill"]):
                best = {"split": segs, "loads_gib": [round(x, 3) for x in loads], "worst_fill": round(fill, 4)}
            return
        for s in range(1, n - start - (k - 1 - card) + 1):
            rec(start + s, card + 1, cuts + [s])

    rec(0, 0, [])
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compose")
    c.add_argument("--plan", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--scratch-root", type=Path, default=Path("/mnt/lexar/omni-campaign/scratch"))
    c.add_argument("--no-readback", action="store_true")
    s = sub.add_parser("split")
    s.add_argument("--model", required=True, type=Path)
    s.add_argument("--budgets-gib", required=True)
    s.add_argument("--head-extra-gib", type=float, default=2.5)
    s.add_argument("--block-extra-gib", type=float, default=0.0)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "split":
            blocks, head = block_bytes(args.model)
            budgets = [float(x) for x in args.budgets_gib.split(",") if x.strip()]
            best = best_split(blocks, head, budgets, args.head_extra_gib, args.block_extra_gib)
            if best is None:
                print(json.dumps({"fits": False, "blocks": len(blocks), "total_gib": round((sum(blocks) + head) / 2 ** 30, 3)}))
                return 4
            print(json.dumps({"fits": True, "blocks": len(blocks), "head_gib": round(head / 2 ** 30, 3), **best}))
            print("SPLIT " + ",".join(str(x) for x in best["split"]))
            return 0
        doc = compose(args.plan, args.out, args.scratch_root, readback=not args.no_readback)
        rpath = args.out.absolute().with_name(args.out.name + ".compose.json")
        rpath.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        digest = sha256_file(rpath)
        rpath.with_name(rpath.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
        print(f"COMPOSED {doc['out']} {doc['tensors_composed']} tensor(s), {doc['replaced_elements']:,} elements replaced; "
              f"dir digest {doc['out_directory_sha256']}; receipt {rpath} sha256 {digest}", flush=True)
        return 0
    except (Refusal, OSError, ValueError, KeyError) as exc:
        print(f"REFUSED: {type(exc).__name__ + ': ' if not isinstance(exc, Refusal) else ''}{exc}", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
