#!/usr/bin/env python3
"""mix_teacher_cache.py -- two same-tokenizer teachers on ONE row: a qtext-format cache of their weighted mixture.

Why: the full-parameter KD trains text rows against ONE same-tokenizer teacher term, w * KL_bucket(teacher || student).
Cyril's teachers for those rows are Qwen 3.8 27B (small weight) AND Qwen3.8-Flash-Next (2026-10-17).  The cross-entropy
part of a KL is linear in the teacher distribution, so for any student s

    sum_t w_t * grad KL(p_t || s)  =  (sum_t w_t) * grad KL(pbar || s),   pbar = sum_t w_t p_t / sum_t w_t

exactly (the teacher entropies are constants).  A cache holding pbar therefore gives the trainer the two-teacher
gradient with NO change to the trainer (`streamed_kd_trainer.QTextCache` reads it as any qtext cache; the data config
points the text source at it; the source's w_q38 is the summed weight).  The only approximation is the cache's top-K:
pbar is formed on the union of the two teachers' top-K ids (an id outside a teacher's top-K contributes that teacher's
share to the tail bucket, where the teacher's own tail already sits) and cut back to the K largest, the rest joining
the tail.  Both inputs must be SEALED caches over the SAME corpus (sha256) with the same store rule (identical stored
positions, checked row by row), the same top_k and vocab_rows.

Stored per position: topk_ids = the K largest pbar ids (descending; ties: lower id), topk_logprobs = log pbar (bf16),
lse = 0 (a mixture has no logits; the trainer reads ids + logprobs only).  The manifest uses the qtext schema; its
teacher block names both input manifests (sha256) and the weights, and is "equal before and after" by construction
(the inputs are sealed files whose digests are checked before and after the build).

  mix  --a Q38_CACHE --b FLASH_CACHE --weight-a 0.25 --weight-b 0.75 --teacher NAME --out-root ROOT
  check --cache DIR  [--a ... --b ... --positions N]   re-derive N random positions from the inputs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

TOOL_NAME = "mix_teacher_cache"
TOOL_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent


def _q() -> Any:
    name = "qtext_teacher_cache"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "qtext_teacher_cache.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q = _q()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()),
            "sha256": q.file_sha256(Path(__file__).resolve())}


class Rows:
    """Row access to one sealed qtext-format cache (raw shard files, every shard verified once)."""

    def __init__(self, cache_dir: Path) -> None:
        self.dir = Path(cache_dir)
        self.manifest_path = self.dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise q.CacheError(f"{self.dir} is not sealed")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != q.MANIFEST_SCHEMA or not (self.manifest.get("all_finite") and self.manifest.get("all_structural_ok")):
            raise q.CacheError(f"{self.dir}: not a sealed, finite, structurally ok qtext cache")
        self.manifest_sha256 = q.file_sha256(self.manifest_path)
        self.top_k = int(self.manifest["cache_config"]["top_k"])
        self.vocab_rows = int(self.manifest["cache_config"]["vocab_rows"])
        self.index: dict[int, tuple[str, int, int, int]] = {}
        for entry in self.manifest["shards"]:
            record = json.loads((self.dir / entry["shard"] / "shard.json").read_text(encoding="utf-8"))
            if q.file_sha256(self.dir / entry["shard"] / "shard.json") != entry["shard_json_sha256"]:
                raise q.CacheError(f"{self.dir}/{entry['shard']}: shard.json differs from the manifest")
            for row in record["row_table"]:
                self.index[int(row["row"])] = (entry["shard"], int(row["offset"]), int(row["count"]), int(row["length"]))
        self._cache: tuple[str, dict[str, Any]] | None = None

    def row(self, r: int) -> dict[str, Any]:
        shard, offset, count, length = self.index[r]
        if self._cache is None or self._cache[0] != shard:
            checked = q.verify_shard(self.dir / shard)
            if not checked["ok"]:
                raise q.CacheError(f"{self.dir}/{shard}: {checked['problems']}")
            self._cache = (shard, q.read_shard(self.dir / shard))
        a = self._cache[1]
        sl = slice(offset, offset + count)
        return {"ids": a["ids"][sl], "pos": a["pos"][sl], "topk_ids": a["topk_ids"][sl].to(torch.int64),
                "lp": a["topk_logprobs"][sl].float(), "length": length}


def mix_rows(ra: dict[str, Any], rb: dict[str, Any], wa: float, wb: float, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """pbar on the union of the two top-K id sets, cut to the K largest: (ids [P, K] int32, logprobs [P, K] fp32)."""

    pa, pb = ra["lp"].exp(), rb["lp"].exp()
    ids = torch.cat([ra["topk_ids"], rb["topk_ids"]], dim=1)
    w = torch.cat([pa * (wa / (wa + wb)), pb * (wb / (wa + wb))], dim=1)
    sid, order = ids.sort(dim=1, stable=True)
    sw = w.gather(1, order)
    dup = sid[:, 1:] == sid[:, :-1]          # an id is at most once per teacher, so duplicates come in pairs
    merged = sw.clone()
    merged[:, 1:] = torch.where(dup, merged[:, 1:] + sw[:, :-1], merged[:, 1:])
    merged[:, :-1] = torch.where(dup, torch.full_like(merged[:, :-1], -1.0), merged[:, :-1])
    # top-K by probability, ties to the lower id: `sid` is ascending, so a STABLE sort on -p keeps lower ids first
    order2 = torch.sort(-merged, dim=1, stable=True).indices[:, :k]
    vals, out_ids = merged.gather(1, order2), sid.gather(1, order2)
    if bool((vals <= 0).any()):
        raise q.CacheError("a mixed position has fewer than K positive-mass ids (the inputs' top-K overlap too much for K)")
    return out_ids.to(torch.int32), vals.clamp_max(1.0).log()


def build(args: argparse.Namespace) -> int:
    a, b = Rows(Path(args.a)), Rows(Path(args.b))
    for key in ("sha256", "rows", "row_length", "path"):
        if a.manifest["corpus"].get(key) != b.manifest["corpus"].get(key):
            raise q.CacheError(f"the two caches index different corpora ({key}: {a.manifest['corpus'].get(key)} vs {b.manifest['corpus'].get(key)})")
    if a.top_k != b.top_k or a.vocab_rows != b.vocab_rows:
        raise q.CacheError("top_k / vocab_rows differ")
    if set(a.index) != set(b.index):
        raise q.CacheError("the two caches store different rows")
    k = a.top_k
    wa, wb = float(args.weight_a), float(args.weight_b)
    if not (wa >= 0 and wb >= 0 and wa + wb > 0):
        raise q.CacheError("weights must be >= 0 with a positive sum")
    corpus = a.manifest["corpus"]
    cache_dir = Path(args.out_root) / args.teacher / corpus["sha256"]
    if (cache_dir / "manifest.json").is_file():
        log(f"{cache_dir} already sealed")
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"a": {"dir": str(a.dir), "manifest_sha256": a.manifest_sha256, "teacher": a.manifest["teacher"]["name"], "weight": wa},
              "b": {"dir": str(b.dir), "manifest_sha256": b.manifest_sha256, "teacher": b.manifest["teacher"]["name"], "weight": wb}}
    config = {"schema": q.CONFIG_SCHEMA, "teacher": args.teacher, "model_dir": None, "corpus_kind": "text",
              "corpus_sha256": corpus["sha256"], "corpus_rows": int(corpus["rows"]), "top_k": k, "vocab_rows": a.vocab_rows,
              "dtype": "mixture", "attention_implementation": None, "linear_attention_kernel": None,
              "rows_per_shard": int(args.rows_per_shard), "store_rule": a.manifest["cache_config"]["store_rule"],
              "corpus_read": a.manifest["cache_config"].get("corpus_read"), "head_rows": a.manifest["cache_config"].get("head_rows"),
              "head_precision": "mixture of two sealed caches", "logit_precision": "log of the weighted mixture, rounded to bf16 once; lse = 0",
              "processor": None, "mixture": {"inputs": inputs, "rule": "pbar = (wa pa + wb pb) / (wa + wb) on the union of top-K ids, cut to K"},
              "tool": {"name": TOOL_NAME, "major": TOOL_VERSION.split(".")[0]}, "tool_major": q.TOOL_VERSION.split(".")[0]}
    config_path = cache_dir / "cache_config.json"
    if config_path.is_file():
        changed = q.compare_config(json.loads(config_path.read_text(encoding="utf-8")), config)
        if changed:
            raise q.CacheError(f"{config_path} locks different semantics for {changed}")
    else:
        q.write_json_atomic(config_path, config)
    before = {"a": q.file_sha256(a.manifest_path), "b": q.file_sha256(b.manifest_path)}
    rows = sorted(a.index)
    per = int(args.rows_per_shard)
    total = (int(corpus["rows"]) + per - 1) // per
    session = q.new_session_id()
    shards = []
    started = time.perf_counter()
    for index in range(total):
        start, stop = index * per, min(int(corpus["rows"]), (index + 1) * per)
        name = q.shard_name(index)
        if (cache_dir / name / "shard.json").is_file():
            shards.append(json.loads((cache_dir / name / "shard.json").read_text(encoding="utf-8")))
            continue
        parts = {key: [] for key in ("ids", "pos", "topk_ids", "topk_logprobs", "lse")}
        table, offset = [], 0
        for r in range(start, stop):
            if r not in a.index:
                table.append({"row": r, "offset": offset, "count": 0, "length": 0, "row_id": str(r)})
                continue
            ra, rb = a.row(r), b.row(r)
            if not torch.equal(ra["pos"], rb["pos"]) or not torch.equal(ra["ids"], rb["ids"]):
                raise q.CacheError(f"row {r}: the two caches store different positions / ids")
            ids, lp = mix_rows(ra, rb, wa, wb, k)
            n = int(ra["pos"].numel())
            parts["ids"].append(ra["ids"])
            parts["pos"].append(ra["pos"])
            parts["topk_ids"].append(ids)
            parts["topk_logprobs"].append(lp.to(torch.bfloat16))
            parts["lse"].append(torch.zeros(n, dtype=torch.float32))
            table.append({"row": r, "offset": offset, "count": n, "length": int(ra["length"]), "row_id": str(r)})
            offset += n
        arrays = {
            "ids": torch.cat(parts["ids"]) if parts["ids"] else torch.empty(0, dtype=torch.int64),
            "pos": torch.cat(parts["pos"]) if parts["pos"] else torch.empty(0, dtype=torch.int64),
            "topk_ids": torch.cat(parts["topk_ids"]) if parts["topk_ids"] else torch.empty((0, k), dtype=torch.int32),
            "topk_logprobs": torch.cat(parts["topk_logprobs"]) if parts["topk_logprobs"] else torch.empty((0, k), dtype=torch.bfloat16),
            "lse": torch.cat(parts["lse"]) if parts["lse"] else torch.empty(0, dtype=torch.float32),
        }
        record = {"rows": [start, stop], "row_table": table, "forward_tokens": int(sum(t["length"] for t in table)),
                  "padded_tokens": 0, "batches": 0, "micro_batch": 0, "max_batch_tokens": 0, "batch_order": "corpus",
                  "head_chunk_rows": 0, "device": "cpu (mixture)", "session": session, "seconds": 0.0,
                  "written_utc": q.utc_now()}
        shards.append(q.write_shard(cache_dir, index, arrays, record, a.vocab_rows, k))
        log(f"{name}: rows [{start},{stop}) {shards[-1]['positions']} positions ({index + 1}/{total})")
    after = {"a": q.file_sha256(a.manifest_path), "b": q.file_sha256(b.manifest_path)}
    if before != after:
        raise q.CacheError("an input manifest changed during the build")
    entries = []
    for i in range(total):
        res = q.verify_shard(cache_dir / q.shard_name(i))
        if not res["ok"]:
            raise q.CacheError(f"{q.shard_name(i)}: {res['problems']}")
        rec = json.loads((cache_dir / q.shard_name(i) / "shard.json").read_text(encoding="utf-8"))
        entries.append({"shard": rec["shard"], "rows": rec["rows"], "positions": rec["positions"], "forward_tokens": rec["forward_tokens"],
                        "files": rec["files"], "finite": rec["checks"]["finite"], "structural_ok": rec["checks"]["structural_ok"],
                        "nonfinite_lse": rec["checks"]["nonfinite_lse"], "nonfinite_logprobs": rec["checks"]["nonfinite_logprobs"],
                        "session": rec["session"], "device": rec["device"], "micro_batch": rec["micro_batch"],
                        "shard_json_sha256": q.file_sha256(cache_dir / q.shard_name(i) / "shard.json")})
    tree = q.canonical_sha256({"inputs": inputs, "before": before})
    manifest = {
        "schema": q.MANIFEST_SCHEMA, "sealed_utc": q.utc_now(), "tool": tool_identity(), "cache_format_tool": q.tool_identity(),
        "engine": {"kind": "mixture of two sealed teacher caches"}, "cache_config": config,
        "cache_config_sha256": q.file_sha256(config_path),
        "teacher": {"name": args.teacher, "model_dir": None, "digest_mode": "input manifests", "tree_sha256_before": tree,
                    "tree_sha256_after": q.canonical_sha256({"inputs": inputs, "before": after}),
                    "equal_before_and_after_every_session": before == after, "interrupted_sessions_vouched_for": 0,
                    "files": [{"path": "a/manifest.json", "sha256": before["a"]}, {"path": "b/manifest.json", "sha256": before["b"]}],
                    "mixture": inputs},
        "corpus": a.manifest["corpus"], "shard_read_order": [e["shard"] for e in entries], "shards": entries,
        "positions_total": sum(e["positions"] for e in entries), "forward_tokens_total": sum(e["forward_tokens"] for e in entries),
        "bytes_total": sum(f["bytes"] for e in entries for f in e["files"].values()),
        "all_finite": all(e["finite"] for e in entries), "all_structural_ok": all(e["structural_ok"] for e in entries),
        "sessions": [{"session": session, "unchanged": before == after}], "sessions_interrupted": [], "sessions_aborted_before_start": [],
        "seal_wall_seconds_this_session": round(time.perf_counter() - started, 2)}
    q.write_json_atomic(cache_dir / "manifest.json", manifest)
    log(f"SEALED {cache_dir}/manifest.json sha256 {q.file_sha256(cache_dir / 'manifest.json')} positions {manifest['positions_total']}")
    return 0


def check(args: argparse.Namespace) -> int:
    """Re-derive random positions: the stored log pbar equals log pbar computed here from the inputs within the bf16
    rounding of the stored value (|d log p| <= |log p| * 2^-8 + 1e-6: bf16 keeps 8 mantissa bits), and the stored mass
    never exceeds 1 beyond bf16 rounding."""

    m = Rows(Path(args.cache))
    inputs = m.manifest["teacher"]["mixture"]
    a, b = Rows(Path(inputs["a"]["dir"])), Rows(Path(inputs["b"]["dir"]))
    wa, wb = float(inputs["a"]["weight"]), float(inputs["b"]["weight"])
    rng = random.Random(int(args.seed))
    rows = [r for r in sorted(m.index) if m.index[r][2] > 0]
    worst, mass_max, checked = 0.0, 0.0, 0
    for _ in range(int(args.positions)):
        r = rng.choice(rows)
        rm, ra, rb = m.row(r), a.row(r), b.row(r)
        j = rng.randrange(int(rm["pos"].numel()))
        pa = {int(i): float(p) for i, p in zip(ra["topk_ids"][j], ra["lp"][j].exp())}
        pb = {int(i): float(p) for i, p in zip(rb["topk_ids"][j], rb["lp"][j].exp())}
        for i, lp in zip(rm["topk_ids"][j].tolist(), rm["lp"][j].tolist()):
            want = (wa * pa.get(i, 0.0) + wb * pb.get(i, 0.0)) / (wa + wb)
            import math
            log_want = math.log(max(want, 1e-30))
            worst = max(worst, abs(lp - log_want) / (abs(log_want) * 2 ** -8 + 1e-6))
        mass_max = max(mass_max, float(rm["lp"][j].exp().sum()))
        checked += 1
    result = {"positions_checked": checked, "worst_logprob_error_in_bf16_ulps": worst, "max_stored_mass": mass_max,
              "pass": worst <= 1.0 and mass_max <= 1.0 + 2 ** -8}
    print(json.dumps(result, indent=1), flush=True)
    return 0 if result["pass"] else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mix")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--weight-a", type=float, required=True)
    m.add_argument("--weight-b", type=float, required=True)
    m.add_argument("--teacher", required=True)
    m.add_argument("--out-root", required=True)
    m.add_argument("--rows-per-shard", type=int, default=64)
    c = sub.add_parser("check")
    c.add_argument("--cache", required=True)
    c.add_argument("--positions", type=int, default=400)
    c.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        return build(args) if args.command == "mix" else check(args)
    except q.CacheError as exc:
        log(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
