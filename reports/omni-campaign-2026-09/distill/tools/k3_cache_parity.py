#!/usr/bin/env python3
"""k3_cache_parity.py -- bit-identity of two k3_teacher_cache builds on identical inputs (lane k3accel, 2026-10-14).

k3_teacher_cache 1.3.0 claims its default (--accel on) path computes EXACTLY the numbers of the frozen 1.2.4
(2aae4dc3...).  This tool is how that claim is checked: each build runs its own ``run`` (imported by path, sha
checked, never edited) and every output tensor is hashed as raw bytes in its stored dtype (bf16 bit patterns, not
an fp32 conversion):

  per layer   the host prefix [N, H] after the layer, and every filled AttnRes block slot [N, H] (slot b of the
              host block stack; the pushed slot and every earlier one)
  head        every tensor of every output group the head writes (topk_ids, topk_logprobs, lse, target_logprob,
              row_file, row_index, real_lengths)

The digests are taken after ``CachePass.layer`` returns and after ``head_group`` returns, outside the tool's own
timers, so the tool's read / compute walls in its receipt are its own.

Subcommands
  digest   --tool PATH [--tool-sha SHA] --out JSON [--inject-ulp L] -- <k3_teacher_cache run arguments>
           ``--inject-ulp L`` moves prefix element [0, 0] by ONE ulp (bit pattern + 1) right after layer L, before
           its digest: the negative control -- ``compare`` must report DIFFER from L on.
  compare  A.json B.json [--out JSON] [--lines PATH]: every (layer, tensor) and (head group, tensor) sha256 of both
           sides, EQUAL / DIFFER; exit 0 only when both sides hashed the same tensors and every one is equal,
           3 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Any

TOOL_NAME = "k3_cache_parity"
TOOL_VERSION = "1.0.0"
CHUNK_BYTES = 64 << 20


def log(msg: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body).hexdigest()


def tensor_sha256(tensor) -> str:
    """sha256 of the tensor's raw bytes in its own dtype, row chunk by row chunk (a strided block slot included)."""
    import torch

    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    digest = hashlib.sha256()
    if t.numel() == 0:
        return digest.hexdigest()
    if t.dim() == 0:
        t = t.reshape(1)
    row_bytes = max(1, (t.numel() // t.shape[0]) * t.element_size())
    step = max(1, CHUNK_BYTES // row_bytes)
    for r0 in range(0, t.shape[0], step):
        piece = t[r0:r0 + step].contiguous()
        if piece.dtype == torch.bool:
            piece = piece.to(torch.uint8)
        digest.update(memoryview(piece.view(torch.uint8).reshape(-1).numpy()))
    return digest.hexdigest()


def one_ulp(tensor) -> None:
    """Element [0, 0] moved by one unit in the last place, in place (its bit pattern + 1)."""
    import torch

    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    flat = tensor.view(-1)[:1]
    flat.view(width).add_(1)


def load_tool(path: str, expected_sha: str | None):
    got = sha256_file(path)
    if expected_sha and got != expected_sha:
        raise SystemExit(f"{path} sha256 {got[:16]} != expected {expected_sha[:16]}; refusing another build")
    name = f"k3_teacher_cache_parity_{got[:12]}"
    if name in sys.modules:
        return sys.modules[name], got
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, got


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------
def digest_run(tool_path: str, tool_sha: str | None, tool_argv: list[str], inject_ulp: int | None = None,
               echo: bool = True) -> dict[str, Any]:
    tool, sha = load_tool(tool_path, tool_sha)
    targs = tool.build_parser().parse_args(tool_argv)
    if targs.command != "run":
        raise SystemExit("digest runs the tool's `run` subcommand only")
    record: dict[str, Any] = {"layers": {}, "head": {}}
    orig_layer, orig_head_group = tool.CachePass.layer, tool.CachePass.head_group
    hash_seconds = [0.0]

    def layer_digest(self, run, layer_idx, prefix, blocks, nb, plan_shards):
        nb_out = orig_layer(self, run, layer_idx, prefix, blocks, nb, plan_shards)
        if inject_ulp is not None and layer_idx == inject_ulp:
            one_ulp(prefix)
            log(f"NEGATIVE CONTROL: prefix[0, 0] moved by one ulp after L{layer_idx:02d}")
        t0 = time.time()
        entry = {"prefix": tensor_sha256(prefix), "dtype": str(prefix.dtype), "shape": list(prefix.shape),
                 "blocks": [tensor_sha256(blocks[:, b]) for b in range(nb_out)], "nb": nb_out}
        record["layers"][str(layer_idx)] = entry
        hash_seconds[0] += time.time() - t0
        if echo:
            print(f"L{layer_idx:02d} prefix {entry['prefix']}", flush=True)
            for b, h in enumerate(entry["blocks"]):
                print(f"L{layer_idx:02d} block{b} {h}", flush=True)
        return nb_out

    def head_group_digest(self, run, pair, seg, si, r0, r1):
        payload = orig_head_group(self, run, pair, seg, si, r0, r1)
        key = f"{seg.name}:{r0}-{r1}"
        record["head"][key] = {k: tensor_sha256(v) for k, v in sorted(payload.items())}
        if echo:
            for k, h in record["head"][key].items():
                print(f"head {key} {k} {h}", flush=True)
        return payload

    tool.CachePass.layer, tool.CachePass.head_group = layer_digest, head_group_digest
    pass_obj = tool.CachePass(targs)
    try:
        receipt = pass_obj.run()
    finally:
        tool.CachePass.layer, tool.CachePass.head_group = orig_layer, orig_head_group
    layers = receipt.get("layers") or {}
    return {"schema": "omni-campaign.k3_cache_parity.digest.v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
            "parity_tool_sha256": sha256_file(os.path.abspath(__file__)),
            "subject": {"path": os.path.abspath(tool_path), "sha256": sha, "version": tool.TOOL_VERSION},
            "argv": tool_argv, "inject_ulp": inject_ulp, "digests": record,
            "hash_seconds": round(hash_seconds[0], 1),
            "timings": {k: {f: v.get(f) for f in ("is_kda", "snapshot", "read_wall_seconds", "compute_seconds",
                                                  "compute_positions_per_s", "dequant_seconds_engine",
                                                  "device_peak_bytes", "rss_after_bytes", "accel", "stages")}
                        for k, v in layers.items()},
            "head_seconds": (receipt.get("head") or {}).get("seconds"),
            "corpus": {k: (receipt.get("corpus") or {}).get(k) for k in ("positions", "real_positions", "groups",
                                                                        "group_tokens", "corpus_sha256", "pass_sha256")},
            "accel": receipt.get("accel"), "verdict": receipt.get("verdict"),
            "pass_dir": pass_obj.pass_dir, "torch": receipt.get("torch"),
            "devices": receipt.get("devices"), "checkpoints": receipt.get("checkpoints")}


def cmd_digest(args, rest: list[str]) -> int:
    out = digest_run(args.tool, args.tool_sha, rest, args.inject_ulp)
    digest = write_json(args.out, out)
    n = sum(1 + len(v["blocks"]) for v in out["digests"]["layers"].values()) + \
        sum(len(v) for v in out["digests"]["head"].values())
    log(f"digest {out['subject']['version']} ({out['subject']['sha256'][:12]}): {n} tensors hashed -> {args.out} "
        f"(sha256 {digest[:12]})")
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def flatten(record: dict[str, Any]) -> dict[str, str]:
    out = {}
    for layer, v in record["layers"].items():
        out[f"L{int(layer):02d} prefix"] = v["prefix"]
        for b, h in enumerate(v["blocks"]):
            out[f"L{int(layer):02d} block{b}"] = h
    for group, tensors in record["head"].items():
        for name, h in tensors.items():
            out[f"head {group} {name}"] = h
    return out


def sort_key(name: str):
    kind, rest = (0, name) if name.startswith("L") else (1, name)
    if kind == 0:
        layer, what = name.split(" ", 1)
        return (0, int(layer[1:]), 0 if what == "prefix" else 1, int(what[5:]) if what.startswith("block") else 0, "")
    return (1, 0, 0, 0, rest)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    fa, fb = flatten(a["digests"]), flatten(b["digests"])
    names = sorted(set(fa) | set(fb), key=sort_key)
    rows = [{"tensor": n, "a": fa.get(n), "b": fb.get(n), "equal": fa.get(n) is not None and fa.get(n) == fb.get(n)}
            for n in names]
    only_a = sorted(set(fa) - set(fb), key=sort_key)
    only_b = sorted(set(fb) - set(fa), key=sort_key)
    differ = [r["tensor"] for r in rows if not r["equal"]]
    ok = bool(rows) and not differ and not only_a and not only_b
    return {"a": a["subject"], "b": b["subject"], "tensors": len(rows), "equal": len(rows) - len(differ),
            "differ": differ, "only_a": only_a, "only_b": only_b, "bit_identical": ok, "rows": rows}


def cmd_compare(args, _rest) -> int:
    with open(args.a, "r", encoding="utf-8") as handle:
        a = json.load(handle)
    with open(args.b, "r", encoding="utf-8") as handle:
        b = json.load(handle)
    res = compare(a, b)
    lines = [f"{r['tensor']:<40} a {r['a']} b {r['b']} {'EQUAL' if r['equal'] else 'DIFFER'}" for r in res["rows"]]
    if args.lines:
        os.makedirs(os.path.dirname(os.path.abspath(args.lines)), exist_ok=True)
        with open(args.lines, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    if not args.quiet:
        print("\n".join(lines))
    print(f"a = {res['a']['version']} {res['a']['sha256'][:16]} | b = {res['b']['version']} {res['b']['sha256'][:16]}")
    if res["bit_identical"]:
        print(f"BIT-IDENTICAL: {res['equal']} of {res['tensors']} tensors equal")
    else:
        first = (res["differ"] or res["only_a"] or res["only_b"])[:3]
        print(f"DIFFER: {len(res['differ'])} of {res['tensors']} tensors differ, only-a {len(res['only_a'])}, "
              f"only-b {len(res['only_b'])} (first {first})")
    if args.out:
        write_json(args.out, {k: v for k, v in res.items()} | {"inputs": [os.path.abspath(args.a), os.path.abspath(args.b)],
                                                             "tool": TOOL_NAME, "tool_version": TOOL_VERSION})
    return 0 if res["bit_identical"] else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("digest", help="run one build with every output tensor hashed; tool args after --")
    d.add_argument("--tool", required=True)
    d.add_argument("--tool-sha", default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--inject-ulp", type=int, default=None, help="NEGATIVE CONTROL: +1 ulp on prefix[0,0] after layer L")
    d.set_defaults(func=cmd_digest)
    c = sub.add_parser("compare", help="two digest receipts, tensor by tensor")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", default=None)
    c.add_argument("--lines", default=None, help="write every sha line here")
    c.add_argument("--quiet", action="store_true", help="only the verdict on stdout (the lines go to --lines)")
    c.set_defaults(func=cmd_compare)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)
    return int(args.func(args, rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
