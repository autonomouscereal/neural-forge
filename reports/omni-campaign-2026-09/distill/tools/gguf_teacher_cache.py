#!/usr/bin/env python3
"""gguf_teacher_cache.py -- a text teacher cache in the qtext_teacher_cache format from ANY GGUF teacher.

Why: the Chimera Omni full-parameter KD (ruling cyril-full-distill-no-lora-2026-10-17) distills from Kimi K3 (primary),
Qwen 3.8 27B (small weight) AND Qwen3.8-Flash-Next ("[Cyril quote withheld]", Cyril 2026-10-17).  Flash Next runs here only as a GGUF (abliterated Strata IQ3_S, 125B MoE) through
llama.cpp, and llama-server returns probabilities for generated tokens only.  So `teacher_score` (teacher_score.cpp,
linked to the export lane's libllama) scores fixed rows teacher-forced, and this tool writes what it returns as a
cache the trainer reads unchanged (streamed_kd_trainer.QTextCache: same manifest/shard schemas, the same store rule,
top_k and vocab_rows).  Flash Next's tokenizer is Qwen 3.8 27B's (248,077 ids, the same merges and special ids), so its
caches index the same corpora as the Q38 caches and sit in a data config as ordinary text sources.

What one stored position is: exactly qtext_teacher_cache's rule -- position t of row r is stored iff t < length[r] and
store_mask[r, t] (a target_mask is shifted by one), and carries the teacher distribution over token t+1 given [0, t].
Per stored position:  lse = logsumexp(logits[0:V]) (double accumulation), topk_ids = top-K rows descending by logit
(ties: lower id), topk_logprobs = logit - lse rounded to bf16 once.  V = min(n_vocab of the GGUF, --vocab-rows).
The logits are llama.cpp's (the GGUF's own quantized output.weight and kernels): `head_precision` is "gguf-native"
and the config locks the GGUF sha256, the scorer binary sha256, the llama.cpp commit and every scorer argument.

Layout, locks, sessions, the teacher before/after digest and the seal are qtext_teacher_cache's own functions, imported
by path (never copied): <out-root>/<teacher>/<corpus sha256>/{cache_config.json, shard-NNNNN/, sessions/, manifest.json}.
The teacher digest is over the GGUF's directory (keep the GGUF alone in it).  After the seal the manifest gains a
`gguf_teacher` block (scorer identity, hello facts, GGUF sha) and names this tool; its schema stays the qtext one.

Commands:
  build   --gguf G --teacher NAME --corpus C [--corpus C2 ...] [--scorer-arg=-ngl --scorer-arg=999 ...]
  compare --cache A --cache B          agreement of two sealed caches over one corpus (top-1, top-K overlap, KL)
  verify  --cache-dir D                qtext_teacher_cache verify
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "gguf_teacher_cache"
TOOL_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
DEFAULT_SCORER = Path("/home/cereal/omni-campaign/export/llamacpp/bin/teacher-score")
MAGIC_HELLO, MAGIC_REQ, MAGIC_REPLY = 0x31485354, 0x31515354, 0x31525354
HEAD_PRECISION = "gguf-native"

_Q: Any = None


def qtext() -> Any:
    global _Q
    if _Q is None:
        path = HERE / "qtext_teacher_cache.py"
        spec = importlib.util.spec_from_file_location("qtext_teacher_cache", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("qtext_teacher_cache", module)
        spec.loader.exec_module(module)
        _Q = module
    return _Q


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tool_identity() -> dict[str, Any]:
    q = qtext()
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()),
            "sha256": q.file_sha256(Path(__file__).resolve())}


class ScorerError(RuntimeError):
    pass


class Scorer:
    """One teacher_score process: the GGUF is loaded once, rows are scored one at a time over the pipe."""

    def __init__(self, binary: Path, gguf: Path, extra: list[str], top_k: int, vocab_rows: int, stderr_path: Path) -> None:
        self.binary = Path(binary)
        self.argv = [str(self.binary), "-m", str(gguf), "--top-k", str(top_k), "--vocab-rows", str(vocab_rows), *extra]
        self.stderr = open(stderr_path, "ab")
        log(f"scorer: {shlex.join(self.argv)}")
        # the export lane's libggml*.so carry their own RUNPATH, so the binary's rpath does not reach libggml-cpu/cuda:
        # the lane's documented runtime is LD_LIBRARY_PATH=<its bin> (build-receipt.json runtime_env), as llama-server uses
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(p for p in (str(self.binary.parent), env.get("LD_LIBRARY_PATH", "")) if p)
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, env=env)
        magic, size = struct.unpack("<II", self._read(8))
        if magic != MAGIC_HELLO:
            raise ScorerError(f"scorer hello magic {magic:#x}")
        self.hello = json.loads(self._read(size).decode("utf-8"))
        self.top_k = int(self.hello["top_k"])
        if self.top_k != int(top_k):
            raise ScorerError(f"scorer top_k {self.top_k} != {top_k}")
        log(f"scorer ready: {json.dumps(self.hello, sort_keys=True)}")

    def _read(self, n: int) -> bytes:
        data = self.proc.stdout.read(n)
        if data is None or len(data) != n:
            code = self.proc.poll()
            raise ScorerError(f"scorer pipe closed after {0 if data is None else len(data)}/{n} bytes (exit {code}); "
                              f"see its stderr log")
        return data

    def score(self, tokens: Any, positions: Any) -> dict[str, Any]:
        import numpy as np

        tokens = np.ascontiguousarray(tokens, dtype="<i4")
        positions = np.ascontiguousarray(positions, dtype="<u4")
        self.proc.stdin.write(struct.pack("<III", MAGIC_REQ, tokens.size, positions.size))
        self.proc.stdin.write(tokens.tobytes())
        self.proc.stdin.write(positions.tobytes())
        self.proc.stdin.flush()
        magic, status, n, k = struct.unpack("<IIII", self._read(16))
        decode_s, post_s = struct.unpack("<dd", self._read(16))
        if magic != MAGIC_REPLY:
            raise ScorerError(f"scorer reply magic {magic:#x}")
        lse = np.frombuffer(self._read(4 * n), dtype="<f4").copy()
        ids = np.frombuffer(self._read(4 * n * k), dtype="<i4").reshape(n, k).copy()
        lp = np.frombuffer(self._read(4 * n * k), dtype="<f4").reshape(n, k).copy()
        if status != 0:
            raise ScorerError(f"scorer status {status} for a row of {tokens.size} tokens / {positions.size} positions")
        if n != positions.size or k != self.top_k:
            raise ScorerError(f"scorer replied n={n} k={k} for {positions.size} positions, top_k {self.top_k}")
        return {"lse": lse, "ids": ids, "lp": lp, "decode_seconds": decode_s, "post_seconds": post_s}

    def close(self) -> int:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(struct.pack("<III", MAGIC_REQ, 0, 0))
                self.proc.stdin.flush()
                self.proc.stdin.close()
                return self.proc.wait(timeout=300)
            return int(self.proc.returncode)
        except Exception:
            self.proc.kill()
            return -9
        finally:
            self.stderr.close()


def visible_devices() -> list[dict[str, Any]]:
    """nvidia-smi rows of the cards CUDA_VISIBLE_DEVICES names (PCI bus order, as every lane here sets it)."""

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        index, uuid, name, used = [p.strip() for p in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "name": name, "memory_used_mib_at_start": int(used)})
    wanted = os.environ.get("CUDA_VISIBLE_DEVICES")
    if wanted is None:
        return rows
    keep = [int(x) for x in wanted.split(",") if x.strip().isdigit()]
    return [r for r in rows if r["index"] in keep]


def cache_config(args: argparse.Namespace, corpus: dict[str, Any], gguf: dict[str, Any], scorer_id: dict[str, Any]) -> dict[str, Any]:
    q = qtext()
    return {
        "schema": q.CONFIG_SCHEMA,
        "teacher": args.teacher,
        "model_dir": str(Path(args.gguf).resolve().parent),
        "corpus_kind": "text",
        "corpus_sha256": corpus["sha256"],
        "corpus_rows": corpus["rows"],
        "top_k": int(args.top_k),
        "vocab_rows": int(args.vocab_rows),
        "dtype": "gguf",
        "attention_implementation": "llama.cpp",
        "linear_attention_kernel": {"engine": "llama.cpp", "commit": scorer_id["llama_cpp_commit"]},
        "rows_per_shard": int(args.rows_per_shard),
        "store_rule": (
            "position t of row r is stored iff t < length[r], store_mask[r,t] (text corpora, "
            "default all; a target_mask is shifted so store_mask[r,t] = target_mask[r,t+1]), and "
            "mm_token_type_ids[r,t] == 0 (VL rows); it carries the teacher distribution over the token at t+1"
        ),
        "corpus_read": {"ids_key": corpus["ids_key"], "lengths_key": corpus["lengths_key"],
                        "mask_source": corpus["mask_source"]},
        "head_rows": f"[0, min(n_vocab, {int(args.vocab_rows)}))",
        "head_precision": HEAD_PRECISION,
        "logit_precision": ("llama.cpp logits of the GGUF's own output.weight and kernels (fp32 output buffer); "
                            "logsumexp accumulated in double, top-k in fp32; stored logprobs rounded to bf16 once"),
        "processor": None,
        "gguf": gguf,
        "scorer": scorer_id,
        "tool": {"name": TOOL_NAME, "major": TOOL_VERSION.split(".")[0]},
        "tool_major": qtext().TOOL_VERSION.split(".")[0],
    }


def scorer_identity(args: argparse.Namespace) -> dict[str, Any]:
    q = qtext()
    binary = Path(args.scorer).resolve()
    receipt = Path(str(binary) + ".receipt.json")
    commit = "unknown"
    if receipt.is_file():
        commit = json.loads(receipt.read_text(encoding="utf-8")).get("llama_cpp_commit", "unknown")
    return {"binary": str(binary), "binary_sha256": q.file_sha256(binary), "llama_cpp_commit": commit,
            "argv_extra": list(args.scorer_arg or [])}


def load_corpus(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    q = qtext()
    corpus = q.load_text_corpus(Path(path), args.ids_key, args.lengths_key)
    return corpus


def build(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    q = qtext()
    started_wall = time.perf_counter()
    gguf_path = Path(args.gguf).resolve()
    model_dir = gguf_path.parent
    others = [p.name for p in q.teacher_files(model_dir) if p != gguf_path]
    if others and not args.allow_other_files:
        raise q.CacheError(f"{model_dir} holds more than the GGUF ({others[:5]}); the teacher digest covers the whole "
                           "directory -- keep the GGUF alone or pass --allow-other-files")
    scorer_id = scorer_identity(args)
    gguf_id = {"path": str(gguf_path), "bytes": gguf_path.stat().st_size, "sha256": args.gguf_sha256}
    per = int(args.rows_per_shard)
    entries = []
    sealed_codes = []
    for path in args.corpus:
        corpus = load_corpus(args, Path(path))
        cache_dir = Path(args.out_root) / args.teacher / corpus["sha256"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        wanted = cache_config(args, corpus, gguf_id, scorer_id)
        config_path = cache_dir / "cache_config.json"
        if config_path.is_file():
            changed = q.compare_config(json.loads(config_path.read_text(encoding="utf-8")), wanted)
            if changed:
                raise q.CacheError(f"{config_path} locks different semantics for {changed}; refusing to mix shards")
        else:
            q.write_json_atomic(config_path, wanted)
        if (cache_dir / "manifest.json").is_file():
            log(f"{cache_dir}/manifest.json already sealed; verifying instead of writing")
            sealed_codes.append(q.verify_cache(cache_dir))
            continue
        entries.append({"corpus": corpus, "cache_dir": cache_dir, "rows": corpus["rows"],
                        "total_shards": (corpus["rows"] + per - 1) // per})
    if not entries:
        return max(sealed_codes) if sealed_codes else 0

    devices = visible_devices()
    busy = [d for d in devices if d["memory_used_mib_at_start"] > q.CARD_BUSY_MIB]
    if busy:
        raise q.CacheError(f"visible cards already hold memory: {busy}; refusing to start")
    previous_sigterm = q.install_sigterm_handler()
    locks = []
    try:
        for entry in entries:
            lock = q.WriterLock(entry["cache_dir"])
            lock.acquire()
            locks.append(lock)
            entry["lock"] = lock
    except BaseException:
        for lock in locks:
            lock.release()
        q.restore_sigterm_handler(previous_sigterm)
        raise
    session_id = q.new_session_id()
    for entry in entries:
        sequence = q.next_session_sequence(entry["cache_dir"])
        entry["session_path"] = entry["cache_dir"] / "sessions" / f"session-{sequence:06d}-{session_id}.json"
        entry["session"] = {"session": session_id, "sequence": sequence, "started_utc": q.utc_now(),
                            "tool": tool_identity(), "cache_format_tool": q.tool_identity(), "scorer": scorer_id,
                            "lock_taken_over_from": entry["lock"].taken_over,
                            "caches_in_process": [str(e["cache_dir"]) for e in entries], "devices_at_start": devices}
    failure = None
    exit_code = 0
    scorer = None
    try:
        for entry in entries:
            entry["session"]["stale_staging_removed"] = q.remove_stale_staging(entry["cache_dir"])
        log(f"teacher digest before ({args.teacher_digest}) of {model_dir}")
        before = q.teacher_digest(model_dir, args.teacher_digest, args.digest_threads)
        if args.teacher_digest == "full" and args.gguf_sha256:
            got = next(f["sha256"] for f in before["files"] if f["path"] == gguf_path.name)
            if got != args.gguf_sha256:
                raise q.CacheError(f"{gguf_path} sha256 {got} != --gguf-sha256 {args.gguf_sha256}")
        for entry in entries:
            entry["session"]["teacher_digest_before"] = before
            q.write_json_atomic(entry["session_path"], entry["session"])
        log(f"teacher tree {before['tree_sha256'][:16]} ({before['file_count']} files, {before['total_bytes']:,} B, "
            f"{before['seconds']} s)")
        for entry in entries:
            entry["pending"] = [i for i in range(entry["total_shards"])
                                if not q.existing_shard_ok(entry["cache_dir"] / q.shard_name(i), False)]
            log(f"cache {entry['cache_dir']}: {entry['total_shards'] - len(entry['pending'])}/{entry['total_shards']} "
                "shards already present")
        if args.max_new_shards:
            budget = int(args.max_new_shards)
            for entry in entries:
                if len(entry["pending"]) > budget:
                    entry["session"]["partial_by_request"] = {"max_new_shards": args.max_new_shards,
                                                              "pending_at_start": len(entry["pending"])}
                    entry["pending"] = entry["pending"][:budget]
                budget -= len(entry["pending"])
        if any(entry["pending"] for entry in entries):
            stderr_path = Path(args.scorer_log) if args.scorer_log else entries[0]["cache_dir"] / f"scorer-{session_id}.log"
            load_started = time.perf_counter()
            scorer = Scorer(Path(args.scorer), gguf_path, list(args.scorer_arg or []), int(args.top_k),
                            int(args.vocab_rows), stderr_path)
            hello = dict(scorer.hello)
            hello["wall_load_seconds"] = round(time.perf_counter() - load_started, 2)
            if int(hello["vocab_rows_scored"]) < min(int(args.vocab_rows), int(hello["n_vocab"])):
                raise q.CacheError(f"scorer scores {hello['vocab_rows_scored']} rows < expected")
            for entry in entries:
                entry["session"]["scorer_hello"] = hello
                entry["current"] = True
                write_shards(args, entry, scorer, session_id)
                entry["current"] = False
    except q.NonFiniteShard as exc:
        failure, exit_code = str(exc), 2
        log(f"FUNCTIONAL FAILURE: {exc}")
    except BaseException as exc:
        failure, exit_code = repr(exc), q.failure_exit_code(exc)
        log(f"session stopped by {exc!r}; recording exit {exit_code}")
        raise
    finally:
        try:
            if scorer is not None:
                rc = scorer.close()
                for entry in entries:
                    entry["session"]["scorer_exit"] = rc
            log("teacher digest after")
            after = q.teacher_digest(model_dir, args.teacher_digest, args.digest_threads)
            for entry in entries:
                session = entry["session"]
                if failure is not None:
                    session["failure"] = failure if entry.get("current") else f"process stopped: {failure}"
                session["teacher_digest_after"] = after
                session["teacher_unchanged_in_session"] = (
                    session.get("teacher_digest_before", {}).get("tree_sha256") == after["tree_sha256"])
                session["finished_utc"] = q.utc_now()
                session["runtime"] = {"host": os.uname().nodename, "pid": os.getpid(), "python": sys.executable,
                                      "devices": devices, "exact_command": [sys.executable, *sys.argv]}
                q.write_json_atomic(entry["session_path"], session)
        finally:
            for lock in locks:
                lock.release()
            q.restore_sigterm_handler(previous_sigterm)
    if not all(entry["session"].get("teacher_unchanged_in_session") for entry in entries):
        log("FUNCTIONAL FAILURE: the teacher directory digest changed during this session")
        return 2
    if exit_code:
        return exit_code
    codes = list(sealed_codes)
    for entry in entries:
        cache_dir = entry["cache_dir"]
        missing = sum(1 for i in range(entry["total_shards"]) if not (cache_dir / q.shard_name(i) / "shard.json").is_file())
        if missing:
            log(f"PARTIAL: {cache_dir}: {missing} shards still pending; not sealed (exit 0)")
            codes.append(0)
            continue
        code = q.seal_manifest(cache_dir, entry["corpus"], model_dir, time.perf_counter() - started_wall)
        if code == 0:
            manifest_path = cache_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_format_tool"] = manifest.get("tool")
            manifest["tool"] = tool_identity()
            manifest["engine"] = {"kind": "teacher_score over libllama", **scorer_id}
            manifest["gguf_teacher"] = {"gguf": gguf_id, "scorer": scorer_id,
                                        "hello": entry["session"].get("scorer_hello")}
            q.write_json_atomic(manifest_path, manifest)
            log(f"SEALED (gguf teacher) {manifest_path} sha256 {q.file_sha256(manifest_path)}")
        codes.append(code)
    return max(codes) if codes else 0


def write_shards(args: argparse.Namespace, entry: dict[str, Any], scorer: Scorer, session_id: str) -> None:
    import numpy as np
    import torch

    q = qtext()
    corpus, cache_dir, rows, per = entry["corpus"], entry["cache_dir"], entry["rows"], int(args.rows_per_shard)
    ids_all, lengths, store = corpus["ids"], corpus["lengths"], corpus["store_mask"]
    session = entry["session"]
    s_tokens = s_positions = 0
    s_compute = 0.0
    for done, index in enumerate(entry["pending"], 1):
        t0 = time.perf_counter()
        row_start, row_stop = index * per, min(rows, (index + 1) * per)
        parts = {k: [] for k in ("ids", "pos", "topk_ids", "topk_logprobs", "lse")}
        table, offset, fwd, decode_s, post_s = [], 0, 0, 0.0, 0.0
        for r in range(row_start, row_stop):
            length = int(lengths[r])
            mask = torch.arange(length) < length
            if store is not None:
                mask &= store[r, :length]
            pos = torch.nonzero(mask, as_tuple=False).reshape(-1)
            count = int(pos.numel())
            if count:
                tokens = ids_all[r, :length]
                out = scorer.score(tokens.numpy().astype(np.int32), pos.numpy().astype(np.uint32))
                parts["ids"].append(tokens[pos].to(torch.int64))
                parts["pos"].append(pos.to(torch.int64))
                parts["topk_ids"].append(torch.from_numpy(out["ids"]).to(torch.int32))
                parts["topk_logprobs"].append(torch.from_numpy(out["lp"]).to(torch.bfloat16))
                parts["lse"].append(torch.from_numpy(out["lse"]).to(torch.float32))
                decode_s += out["decode_seconds"]
                post_s += out["post_seconds"]
                fwd += length
            table.append({"row": r, "offset": offset, "count": count, "length": length, "row_id": corpus["row_ids"][r]})
            offset += count
        k = int(args.top_k)
        arrays = {
            "ids": torch.cat(parts["ids"]) if parts["ids"] else torch.empty(0, dtype=torch.int64),
            "pos": torch.cat(parts["pos"]) if parts["pos"] else torch.empty(0, dtype=torch.int64),
            "topk_ids": torch.cat(parts["topk_ids"]) if parts["topk_ids"] else torch.empty((0, k), dtype=torch.int32),
            "topk_logprobs": (torch.cat(parts["topk_logprobs"]) if parts["topk_logprobs"]
                              else torch.empty((0, k), dtype=torch.bfloat16)),
            "lse": torch.cat(parts["lse"]) if parts["lse"] else torch.empty(0, dtype=torch.float32),
        }
        seconds = time.perf_counter() - t0
        record = {
            "rows": [row_start, row_stop], "row_table": table, "forward_tokens": fwd, "padded_tokens": fwd,
            "batches": row_stop - row_start, "micro_batch": 1, "max_batch_tokens": 0, "batch_order": "corpus",
            "head_chunk_rows": 0, "device": f"llama.cpp CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            "session": session_id, "seconds": round(seconds, 3),
            "forward_tokens_per_second": round(fwd / max(seconds, 1e-9), 2),
            "decode_seconds": round(decode_s, 3), "post_seconds": round(post_s, 3),
            "torch_peak_allocated_gb_so_far": None, "written_utc": q.utc_now(),
        }
        shard = q.write_shard(cache_dir, index, arrays, record, int(args.vocab_rows), k)
        s_tokens += fwd
        s_positions += shard["positions"]
        s_compute += seconds
        remaining = sum(min(rows, (i + 1) * per) - i * per for i in entry["pending"][done:])
        q.write_json_atomic(cache_dir / "progress.json", {
            "shards_total": entry["total_shards"], "shards_present": entry["total_shards"] - len(entry["pending"]) + done,
            "session": session_id, "session_forward_tokens": s_tokens, "session_positions": s_positions,
            "session_forward_tokens_per_second": round(s_tokens / max(s_compute, 1e-9), 2),
            "rows_remaining": remaining, "updated_utc": q.utc_now()})
        log(f"{shard['shard']}: rows [{row_start},{row_stop}) {shard['positions']} positions, {fwd} tokens in "
            f"{seconds:.1f} s = {record['forward_tokens_per_second']} tok/s (decode {decode_s:.1f} s, post {post_s:.1f} s); "
            f"finite={shard['checks']['finite']} ({done}/{len(entry['pending'])}) [{cache_dir.name[:16]}]")
    session["session_forward_tokens"] = s_tokens
    session["session_positions"] = s_positions
    session["session_forward_tokens_per_second"] = round(s_tokens / max(s_compute, 1e-9), 2)


def compare(args: argparse.Namespace) -> int:
    """Two sealed caches over ONE corpus, position by position: top-1 agreement, top-K id overlap, |dlse|, the largest
    logprob gap on shared ids, and KL(A || B) over A's top-K plus a tail bucket (B's mass on A's ids from B's own top-K,
    the rest of B's mass as the tail)."""

    import numpy as np
    import torch

    q = qtext()
    a_dir, b_dir = Path(args.cache[0]), Path(args.cache[1])
    ma = json.loads((a_dir / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b_dir / "manifest.json").read_text(encoding="utf-8"))
    if ma["corpus"]["sha256"] != mb["corpus"]["sha256"]:
        raise q.CacheError("the two caches index different corpora")

    def gather(cache_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        out = {k: [] for k in ("rowpos", "topk_ids", "topk_logprobs", "lse")}
        for entry in manifest["shards"]:
            arrays = q.read_shard(cache_dir / entry["shard"])
            rows = torch.empty(arrays["pos"].numel(), dtype=torch.int64)
            for row in arrays["record"]["row_table"]:
                rows[row["offset"]: row["offset"] + row["count"]] = row["row"]
            out["rowpos"].append(rows * 1_000_000 + arrays["pos"])
            out["topk_ids"].append(arrays["topk_ids"].to(torch.int64))
            out["topk_logprobs"].append(arrays["topk_logprobs"].float())
            out["lse"].append(arrays["lse"])
        return {k: torch.cat(v) for k, v in out.items()}

    a, b = gather(a_dir, ma), gather(b_dir, mb)
    if not torch.equal(a["rowpos"], b["rowpos"]):
        raise q.CacheError("the caches store different positions")
    n, k = a["topk_ids"].shape
    top1 = float((a["topk_ids"][:, 0] == b["topk_ids"][:, 0]).float().mean())
    overlap, kl, gap = [], [], 0.0
    for i in range(0, n, 4096):
        ai, bi = a["topk_ids"][i:i + 4096], b["topk_ids"][i:i + 4096]
        al, bl = a["topk_logprobs"][i:i + 4096], b["topk_logprobs"][i:i + 4096]
        eq = ai.unsqueeze(2) == bi.unsqueeze(1)  # [m, k, k]
        shared = eq.any(dim=2)
        overlap.append(shared.float().mean(dim=1))
        b_on_a = torch.where(eq, bl.unsqueeze(1), torch.full_like(eq, float("-inf"), dtype=torch.float32)).amax(dim=2)
        if bool(shared.any()):
            gap = max(gap, float((al - b_on_a).abs()[shared].max()))
        pa = al.exp()
        pb = torch.where(shared, b_on_a.exp(), torch.zeros_like(pa))
        tail_a = (1 - pa.sum(dim=1)).clamp_min(1e-12)
        tail_b = (1 - pb.sum(dim=1)).clamp_min(1e-12)
        pb_floor = torch.where(shared, pb, torch.full_like(pb, 1e-12))
        kl.append((pa * (pa.clamp_min(1e-30).log() - pb_floor.log())).sum(dim=1) + tail_a * (tail_a.log() - tail_b.log()))
    overlap_t, kl_t = torch.cat(overlap), torch.cat(kl)
    result = {"schema": "omni-gguf-teacher-cache-compare-v1", "tool": tool_identity(), "a": str(a_dir), "b": str(b_dir),
              "positions": int(n), "top_k": int(k), "top1_agreement": top1,
              "topk_overlap_mean": float(overlap_t.mean()), "topk_overlap_min": float(overlap_t.min()),
              "lse_abs_diff_max": float((a["lse"] - b["lse"]).abs().max()),
              "lse_abs_diff_mean": float((a["lse"] - b["lse"]).abs().mean()),
              "shared_logprob_abs_gap_max": gap, "kl_a_b_mean": float(kl_t.mean()), "kl_a_b_p99": float(kl_t.quantile(0.99)),
              "kl_a_b_max": float(kl_t.max())}
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    if args.receipt:
        q.write_json_atomic(Path(args.receipt), result)
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gguf", required=True)
    b.add_argument("--gguf-sha256", default="", help="checked against the 'before' digest (full mode)")
    b.add_argument("--teacher", required=True, help="cache namespace, e.g. flash-next-strata-iq3s")
    b.add_argument("--corpus", required=True, action="append")
    b.add_argument("--out-root", default="/mnt/lexar/omni-campaign/teacher_cache")
    b.add_argument("--scorer", default=str(DEFAULT_SCORER))
    b.add_argument("--scorer-arg", action="append", default=[], help="one teacher_score argument (repeat; use --scorer-arg=-ngl)")
    b.add_argument("--scorer-log", default=None)
    b.add_argument("--top-k", type=int, default=64)
    b.add_argument("--vocab-rows", type=int, default=248_320)
    b.add_argument("--rows-per-shard", type=int, default=64)
    b.add_argument("--ids-key", default="input_ids")
    b.add_argument("--lengths-key", default="auto")
    b.add_argument("--teacher-digest", default="full", choices=["full", "stat"])
    b.add_argument("--digest-threads", type=int, default=2)
    b.add_argument("--max-new-shards", type=int, default=0)
    b.add_argument("--allow-other-files", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--cache", action="append", required=True)
    c.add_argument("--receipt", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--cache-dir", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    q = qtext()
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        if args.command == "build":
            return build(args)
        if args.command == "compare":
            if len(args.cache) != 2:
                raise q.CacheError("compare takes exactly two --cache")
            return compare(args)
        if args.command == "verify":
            return q.verify_cache(Path(args.cache_dir))
    except (q.CacheError, ScorerError) as exc:
        log(f"REFUSED: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
