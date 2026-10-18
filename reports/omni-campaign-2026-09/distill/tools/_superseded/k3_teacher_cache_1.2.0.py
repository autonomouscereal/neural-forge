#!/usr/bin/env python3
"""k3_teacher_cache.py -- the stage-C K3 teacher cache (DISTILL_DESIGN_2026-10-12 section 3, K3 row
+ the K3 route probe row; consumed by D2 signal (2), "K3 top-32 KL on aligned boundaries only").

Subcommands

  run     A teacher-forced, layer-streamed Kimi-K3 pass over dual-tokenized K3 rows.  Per REAL
          position it writes the top-32 ids (int32) + logprobs (bf16) + lse (fp32) = 196 B/token,
          plus the realised next-token logprob (fp32, 4 B/token) that the route probe reads, so a
          shard costs 200 B/token on disk.  One PASS holds <= --max-pass-tokens positions of host
          state (prefix_sum + up to eight AttnRes block snapshots -- attn_res_block_size 12 over 93
          layers -- + the working h; design: 10 x 0.5M x 7,168 x 2 B = 71.7 GB at 0.5M), so the
          2.0M-token budget runs as ``--pass-count 4``.  The working set on the CARD is bounded by
          ``--group-tokens`` (whole rows; every row is independent of every other row), so the
          same command runs on a 24 GB card and on the RTX 8000.  Resumable at every layer
          boundary (``--checkpoint-every``, blocks written once, prefix per checkpoint) and at
          every output group.  Integrity per the design's K3 row: per-shard size/mtime before and
          after (and against the fold receipt's ``resume_context``), plus one full sha256 per
          shard hashed IN LINE by the prefetch thread (``--inline-digest``, pass 1) against
          ``k3_lexar_digest_verify.json`` -- the bytes are hashed while they are read into the
          page cache for the next layer, never re-read.  Refused up front: a missing / unusable
          fold receipt (unless ``--allow-source-drift``), and with ``--inline-digest`` a missing
          digest receipt or any shard of the pass that the receipt does not list as VERIFIED with
          hf_sha256 == local_sha256; refused at the seal: any shard of the pass left unhashed or
          unchecked.  A ``--resume`` never re-reads a shard only earlier layers needed (its digest,
          when one is due, comes from the checkpoint).
  xtok    The cross-tokenizer target.  At every aligned byte boundary (``boundary_map.json``
          pairs [qwen_pos, k3_pos]: both sides consumed the same bytes) the K3 top-32 at k3_pos is
          mapped to Qwen ids by IDENTICAL byte string, restricted to the text rows [0, 248,320).
          Unmatched K3 tokens and the out-of-top-32 mass go to the tail bucket
          (``tail_mass = 1 - matched_mass``).
  probe   Per-route bits-per-byte of K3 -- and of Q38 when the Q-text cache's per-position
          ``target_logprob`` is given -- on the SAME byte span of each paired row (first to last
          aligned boundary).  Informs lambda_K3 and the route mix only; it never skips K3.
  parity  The cache's top-k vs the fold engine's own teacher-forced review dump
          (``fold/review/*_teacher_forced.safetensors``, top-16 per position), against the
          PRE-REGISTERED ``PARITY_THRESHOLDS`` below (fixed before the first parity run).

Reuse (imported by path, never edited): the K3 fold engine ``k3_dense_pass`` (1.2.5, sha256
fd7dbf8d..., the tool recorded in ``$OC/k3-run-oct09-claude``) -- ``read_layer`` (shard reads,
MXFP4 experts kept PACKED on ``--experts-device``, the compute-dtype rule), ``attention_half`` /
``ffn_half`` (KDA / MLA, the AttnRes reads, the routed MoE with per-expert dequant), ``Pair``,
``require_finite``, ``ShardStager`` and the ``DualStreamRun`` embedding / output-head methods; and
through it the k3 lane (``k3_common``, ``k3_layer``).  ``k3_corpus_build`` (``bind_lanes``,
``load_sides``) supplies both tokenizers' byte spellings for ``xtok`` and ``probe``.

What this tool adds, and nothing else: row-group batching of those calls, the host state with
preallocated block slots, layer checkpoints, the prefetch/digest thread, the top-32 reduction, the
cache layout and manifest, the vocabulary byte map and the probe arithmetic.

Usage (server).  P40s need the Pascal venv (the lab venv's torch 2.11+cu128 has no sm_6x):

  PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 nice -n 15 $PY k3_teacher_cache.py run \\
      --input-ids $OC/composite/corpus/k3_review_input_ids.safetensors \\
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --group-tokens 32768
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import uuid
from typing import Any

import torch

TOOL_NAME = "k3_teacher_cache"
TOOL_VERSION = "1.2.0"  # 1.2.0: digest / fold receipt refusals, resume never re-reads finished shards
# what a checkpoint / head-progress resume must match: bumped only when the numbers could change
NUMERICS_VERSION = "k3_teacher_cache.numerics.v1"
SCHEMA = "omni-campaign.k3_teacher_cache.v1"
TOPK = 32
# ids int32 + logprobs bf16 + lse fp32: the design's 196 B/token; the probe's target logprob adds 4
CORE_BYTES_PER_TOKEN = TOPK * 4 + TOPK * 2 + 4
PROBE_BYTES_PER_TOKEN = 4
QWEN_TEXT_ROWS = 248_320
QWEN_SEPARATOR_ID = 248_044
MAX_PASS_TOKENS = 500_000

OC = "/home/cereal/omni-campaign"
DEFAULT_ENGINE_DIR = (f"{OC}/runtime/k3-continuation-oct08-claude/code/reports/omni-campaign-2026-09/"
                      "composite/tools")
ENGINE_EXPECTED_VERSION = "1.2.5"
ENGINE_EXPECTED_SHA256 = "fd7dbf8da04504696ebb0702372802d2869bd1c13ee8c50d2578cee8108a55b7"
DEFAULT_SRC_DIR = "/mnt/lexar/models/moonshotai/Kimi-K3"
DEFAULT_OUT_ROOT = f"{OC}/distill/teacher_cache/k3"
DEFAULT_DIGEST_RECEIPT = f"{OC}/k3/logs/k3_lexar_digest_verify.json"
DEFAULT_FOLD_RECEIPT = f"{OC}/k3-run-oct09-claude/receipts/k3_dense_pass_receipt.json"
DEFAULT_CORPUS_TOOLS_DIR = f"{OC}/composite/tools"

# Fixed BEFORE the first parity run (2026-10-12) and never tuned on its numbers.  The reference was
# computed bf16 on an RTX 8000 (sm75) with torch 2.11/cu128; a different card / cuBLAS changes bf16
# accumulation order through 93 layers, so bit equality is not the claim.  What must hold:
PARITY_THRESHOLDS = {
    "top1_agreement_min": 0.97,          # cache argmax == reference argmax
    "top16_overlap_mean_min": 0.90,      # |cache top-16 ∩ reference top-16| / 16, mean over positions
    "target_logprob_abs_diff_median_max": 0.10,   # nats, realised next token
    "target_logprob_abs_diff_p99_max": 1.00,
    "non_finite_max": 0,
}


def log(msg: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: Any) -> str:
    """Atomic canonical JSON; returns the sha256 of the bytes written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    body = canonical_json(payload).encode("utf-8")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body).hexdigest()


def vmrss_bytes() -> int | None:
    """CURRENT resident set (k3_common.peak_rss_bytes is the high-water mark)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------
# the engine, imported by path and never edited
# --------------------------------------------------------------------------
class Engine:
    """``k3_dense_pass`` + the k3 lane it binds, with the identity every receipt carries."""

    def __init__(self, engine_dir: str, allow_engine_sha: bool = False):
        path = os.path.join(engine_dir, "k3_dense_pass.py")
        if not os.path.isfile(path):
            raise SystemExit(f"the K3 fold engine was not found at {path} (--engine-dir / K3_ENGINE_DIR)")
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        import k3_dense_pass as kdp  # noqa: E402

        if os.path.realpath(kdp.__file__) != os.path.realpath(path):
            raise SystemExit(f"a different k3_dense_pass is already imported ({kdp.__file__}); refusing to mix engines")
        import k3_common  # noqa: E402  (the lane dir is on sys.path once the engine imported)
        import k3_layer  # noqa: E402

        self.kdp, self.common, self.layer = kdp, k3_common, k3_layer
        self.path = os.path.realpath(path)
        self.sha256 = sha256_file(path)
        self.version = kdp.TOOL_VERSION
        if not allow_engine_sha and (self.sha256 != ENGINE_EXPECTED_SHA256 or self.version != ENGINE_EXPECTED_VERSION):
            raise SystemExit(f"engine {path} is v{self.version} sha {self.sha256[:12]}; this tool is pinned to "
                             f"v{ENGINE_EXPECTED_VERSION} {ENGINE_EXPECTED_SHA256[:12]} (the fold engine of record); "
                             "--allow-engine-sha records an override")

    def record(self) -> dict[str, Any]:
        return {"path": self.path, "tool_version": self.version, "sha256": self.sha256,
                "pinned": {"version": ENGINE_EXPECTED_VERSION, "sha256": ENGINE_EXPECTED_SHA256},
                "lane_dir": self.kdp.LANE_DIR, "lane_sha256": self.kdp.lane_file_digests()}


# --------------------------------------------------------------------------
# the corpus: K3 ids files (k3_corpus_build v2 layout), split into passes and row groups
# --------------------------------------------------------------------------
def load_ids_file(path: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    got = load_file(path)
    if "input_ids" not in got or "real_lengths" not in got:
        raise SystemExit(f"{path}: a K3 ids file must carry input_ids [rows, T] and real_lengths [rows] "
                         "(k3_corpus_build v2 layout); refusing to guess where the padding starts")
    ids = got["input_ids"].to(torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    lengths = got["real_lengths"].to(torch.long).flatten()
    if ids.dim() != 2 or lengths.numel() != ids.shape[0]:
        raise SystemExit(f"{path}: input_ids {list(ids.shape)} and real_lengths {list(lengths.shape)} disagree")
    if bool((lengths < 1).any()) or bool((lengths > ids.shape[1]).any()):
        raise SystemExit(f"{path}: real_lengths outside [1, {ids.shape[1]}]")
    name = os.path.splitext(os.path.basename(path))[0]
    return {"path": os.path.abspath(path), "name": name, "ids": ids.contiguous(), "real_lengths": lengths,
            "rows": int(ids.shape[0]), "sequence_length": int(ids.shape[1]),
            "input_ids_sha256": hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
            "real_lengths_sha256": hashlib.sha256(lengths.contiguous().numpy().tobytes()).hexdigest(),
            "file_sha256": sha256_file(path), "real_tokens": int(lengths.sum())}


def corpus_sha256(files: list[dict[str, Any]]) -> str:
    return sha256_text(canonical_json([{"name": f["name"], "shape": [f["rows"], f["sequence_length"]],
                                        "input_ids_sha256": f["input_ids_sha256"],
                                        "real_lengths_sha256": f["real_lengths_sha256"]} for f in files]))


def plan_passes(files: list[dict[str, Any]], pass_count: int) -> list[list[tuple[int, int]]]:
    """Contiguous (file, row) ranges with ~equal REAL tokens: a row goes to the pass its token midpoint falls in."""
    if pass_count < 1:
        raise SystemExit("--pass-count must be >= 1")
    rows = [(fi, r, int(f["real_lengths"][r])) for fi, f in enumerate(files) for r in range(f["rows"])]
    total = sum(n for _, _, n in rows)
    passes: list[list[tuple[int, int]]] = [[] for _ in range(pass_count)]
    acc = 0
    for fi, r, n in rows:
        k = min(int((acc + n / 2.0) * pass_count / max(total, 1)), pass_count - 1)
        passes[k].append((fi, r))
        acc += n
    return passes


class CacheSegment:
    """One ids file restricted to this pass's rows.  ``T`` is trimmed to the longest REAL row when
    ``trim`` (every block is causal and every token-wise op is per token, so positions below a row's
    real length never read the trimmed padding)."""

    role = "teacher"

    def __init__(self, meta: dict[str, Any], file_index: int, rows: list[int], *, trim: bool, truncate: int = 0):
        self.name = meta["name"]
        self.path = meta["path"]
        self.file_index = file_index
        self.row_index = torch.tensor(rows, dtype=torch.long)
        self.rows = len(rows)
        self.real_lengths = meta["real_lengths"][self.row_index] if rows else torch.zeros(0, dtype=torch.long)
        if truncate:  # PROBE ONLY: the first `truncate` tokens of each row (a causal prefix is a valid input)
            self.real_lengths = self.real_lengths.clamp_max(int(truncate))
        self.T = int(self.real_lengths.max()) if (trim and rows) else int(meta["sequence_length"])
        if truncate:
            self.T = min(self.T, int(truncate))
        self.ids = meta["ids"][self.row_index][:, : self.T].contiguous() if rows else meta["ids"][:0]

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "rows": self.rows, "sequence_length_used": self.T,
                "real_tokens": int(self.real_lengths.sum()), "positions": self.rows * self.T,
                "first_row": int(self.row_index[0]) if self.rows else None,
                "last_row": int(self.row_index[-1]) if self.rows else None}


class TeacherCorpus:
    """The pass's segments flattened token-major, with the row groups the card processes at a time.
    ``row_chunks`` is ``k3_dense_pass.Corpus.row_chunks``'s contract (the engine iterates it)."""

    def __init__(self, segments: list[CacheSegment], group_tokens: int):
        self.segments = [s for s in segments if s.rows]
        self.offsets: list[int] = []
        total = 0
        for seg in self.segments:
            self.offsets.append(total)
            total += seg.rows * seg.T
        self.tokens = total
        self.groups: list[tuple[int, int, int, int, int]] = []
        for si, (seg, offset) in enumerate(zip(self.segments, self.offsets)):
            per = max(1, int(group_tokens) // max(seg.T, 1))
            for r0 in range(0, seg.rows, per):
                r1 = min(r0 + per, seg.rows)
                self.groups.append((si, r0, r1, offset + r0 * seg.T, offset + r1 * seg.T))

    def row_chunks(self, row_chunk: int):
        for seg, offset in zip(self.segments, self.offsets):
            for r0 in range(0, seg.rows, max(1, row_chunk)):
                r1 = min(r0 + max(1, row_chunk), seg.rows)
                yield seg, r0, r1, offset + r0 * seg.T, offset + r1 * seg.T


class GroupView:
    """Rows [r0, r1) of one segment as a corpus of their own: offsets relative to the group's pair."""

    def __init__(self, seg: CacheSegment, r0: int, r1: int):
        self.seg, self.r0, self.r1 = seg, r0, r1
        self.tokens = (r1 - r0) * seg.T

    def row_chunks(self, row_chunk: int):
        step = max(1, int(row_chunk))
        for a in range(self.r0, self.r1, step):
            b = min(a + step, self.r1)
            yield self.seg, a, b, (a - self.r0) * self.seg.T, (b - self.r0) * self.seg.T


# --------------------------------------------------------------------------
# teacher integrity: stat before/after, the fold receipt's metadata, the in-line digest
# --------------------------------------------------------------------------
def source_stat(index, src_dir: str) -> dict[str, Any]:
    files = {}
    for name in sorted(set(index.weight_map.values())):
        st = os.stat(os.path.join(src_dir, name))
        files[name] = {"bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    return {"config_sha256": sha256_file(os.path.join(src_dir, "config.json")),
            "index_sha256": sha256_file(os.path.join(src_dir, "model.safetensors.index.json")),
            "files": files}


def expected_from_fold_receipt(path: str | None) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        got = json.load(handle)
    model = ((got.get("resume_context") or {}).get("model") or {})
    if not model.get("source_file_metadata"):
        return None
    return {"path": path, "receipt_sha256": sha256_file(path), "config_sha256": model.get("config_sha256"),
            "index_sha256": model.get("index_sha256"), "files": model["source_file_metadata"]}


def compare_stat(now: dict[str, Any], ref: dict[str, Any], *, what: str) -> list[str]:
    problems = []
    for key in ("config_sha256", "index_sha256"):
        if ref.get(key) and ref[key] != now[key]:
            problems.append(f"{what}: {key} {now[key][:12]} != {ref[key][:12]}")
    for name, rec in ref["files"].items():
        cur = now["files"].get(name)
        if cur is None:
            problems.append(f"{what}: {name} missing")
        elif int(cur["bytes"]) != int(rec["bytes"]) or int(cur["mtime_ns"]) != int(rec["mtime_ns"]):
            problems.append(f"{what}: {name} bytes/mtime {cur} != {rec}")
    return problems


def expected_digests(path: str | None) -> dict[str, Any] | None:
    """The digest receipt's per-file record.  The reference digest is ``hf_sha256`` (the publisher's): a
    shard the verifier recorded as MISMATCH carries its own corrupt bytes' hash in ``local_sha256``."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        got = json.load(handle)
    files = {f["file"]: {"sha256": f.get("hf_sha256"), "bytes": f.get("hf_size"),
                         "local_sha256": f.get("local_sha256"), "status": f.get("status")}
             for f in got.get("files", []) if f.get("file")}
    return {"path": path, "receipt_sha256": sha256_file(path), "all_verified": got.get("all_verified"),
            "compares": "hf_sha256", "files": files}


def digest_receipt_problems(digests: dict[str, Any] | None, shards: list[str], path: str | None) -> list[str]:
    """What stops ``--inline-digest`` from verifying every shard of this pass -- checked BEFORE the pass."""
    if digests is None:
        return [f"no digest receipt at {path!r}"]
    problems = []
    for shard in shards:
        want = digests["files"].get(shard)
        if want is None:
            problems.append(f"{shard} is not in the digest receipt")
            continue
        if want.get("status") != "VERIFIED":
            problems.append(f"{shard} has status {want.get('status')!r} in the digest receipt")
        if not want.get("sha256"):
            problems.append(f"{shard} has no hf_sha256 in the digest receipt")
        elif want.get("local_sha256") and want["local_sha256"] != want["sha256"]:
            problems.append(f"{shard}: the digest receipt's local_sha256 != hf_sha256")
    return problems


class ShardWarmer(threading.Thread):
    """Reads the NEXT layer's shard(s) sequentially into the page cache while the card computes the
    current one, and (``digest``) hashes every byte as it goes -- the design's "one full sha256 hashed
    in line during pass 1 (no extra read)".  It never runs more than ``ahead`` shards past the shard the
    pass is consuming; the main thread's ``read_layer`` then reads from the page cache.

    On a resume, ``skip`` names the shards no remaining layer reads (and, with ``digest``, whose digest
    the checkpoint already holds -- passed in ``prior``): they are never opened, never waited on."""

    def __init__(self, src_dir: str, order: list[str], *, ahead: int, digest: bool, chunk: int = 64 << 20,
                 skip: set[str] | None = None, prior: dict[str, dict[str, Any]] | None = None):
        super().__init__(name="k3-shard-warmer", daemon=True)
        self.src_dir, self.order = src_dir, list(order)
        self.ahead, self.digest, self.chunk = max(0, int(ahead)), bool(digest), int(chunk)
        self.skip = set(skip or ())
        self.allowed = -1
        self.done_upto = -1
        self.results: dict[str, dict[str, Any]] = {s: dict(r) for s, r in (prior or {}).items() if s in self.skip}
        self.seeded = sorted(self.results)
        self.read_log: list[str] = []
        self.cv = threading.Condition()
        self.stop_flag = False
        self.error: BaseException | None = None

    def advance(self, shards: list[str]) -> None:
        positions = [self.order.index(s) for s in shards if s in self.order]
        if not positions:
            return
        with self.cv:
            self.allowed = max(self.allowed, max(positions) + self.ahead)
            self.cv.notify_all()

    def release_all(self) -> None:
        with self.cv:
            self.allowed = len(self.order)
            self.cv.notify_all()

    def run(self) -> None:
        try:
            for i, shard in enumerate(self.order):
                if shard in self.skip or shard in self.results:
                    continue  # nothing left reads it (resume): neither waited on nor opened
                with self.cv:
                    while not self.stop_flag and i > self.allowed:
                        self.cv.wait(timeout=1.0)
                    if self.stop_flag:
                        return
                self.read_log.append(shard)
                path = os.path.join(self.src_dir, shard)
                started = time.time()
                digest = hashlib.sha256() if self.digest else None
                size = 0
                with open(path, "rb", buffering=0) as handle:
                    while True:
                        block = handle.read(self.chunk)
                        if not block:
                            break
                        size += len(block)
                        if digest is not None:
                            digest.update(block)
                        if self.stop_flag:
                            return
                wall = max(time.time() - started, 1e-9)
                self.results[shard] = {"bytes": size, "seconds": round(wall, 1),
                                       "MB_per_s": round(size / 1e6 / wall, 1),
                                       "sha256": digest.hexdigest() if digest is not None else None}
                with self.cv:
                    self.done_upto = i
                    self.cv.notify_all()
        except BaseException as exc:  # surfaced by the pass, never swallowed
            self.error = exc

    def close(self) -> None:
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()

    def wait_all(self, timeout: float | None = None) -> None:
        self.release_all()
        self.join(timeout)


# --------------------------------------------------------------------------
# the pass
# --------------------------------------------------------------------------
class CachePass:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine = Engine(args.engine_dir, args.allow_engine_sha)
        kdp, common = self.engine.kdp, self.engine.common
        self.kdp, self.common = kdp, common
        self.index = common.K3Index(args.src_dir)
        self.cfg = self.index.text_config
        self.block_size = int(self.cfg.get("attn_res_block_size") or 0)
        if self.block_size <= 0:
            raise SystemExit("only the AttnRes variant (attn_res_block_size set) is implemented by the engine")
        self.device = common.canonical_device(args.device)
        self.cuda = self.device.type == "cuda"
        self.experts_device = common.canonical_device(args.experts_device or args.device)
        self.resid_device = common.canonical_device(args.resid_device)
        self.compute_dtype = kdp.torch_dtype(args.compute_dtype)
        self.num_layers = self.index.num_layers
        self.layers = self._layer_window()
        self.full_stack = self.layers == list(range(self.num_layers))

        self.files = [load_ids_file(p) for p in args.input_ids]
        self.corpus_sha = corpus_sha256(self.files)
        plan = plan_passes(self.files, args.pass_count)
        if not 0 <= args.pass_index < args.pass_count:
            raise SystemExit(f"--pass-index {args.pass_index} outside [0, {args.pass_count})")
        mine = plan[args.pass_index]
        if args.row_limit:
            mine = mine[: int(args.row_limit)]
        self.plan_record = [{"pass": k, "rows": len(p), "real_tokens": sum(int(self.files[fi]["real_lengths"][r])
                                                                           for fi, r in p),
                             "first": list(p[0]) if p else None, "last": list(p[-1]) if p else None}
                            for k, p in enumerate(plan)]
        if args.truncate_seq and not args.probe:
            raise SystemExit("--truncate-seq changes the teacher's input; it is a PROBE option (--probe)")
        segments = [CacheSegment(f, fi, [r for gi, r in mine if gi == fi], trim=not args.no_trim,
                                 truncate=args.truncate_seq)
                    for fi, f in enumerate(self.files)]
        self.corpus = TeacherCorpus(segments, args.group_tokens)
        if self.corpus.tokens == 0:
            raise SystemExit(f"pass {args.pass_index}/{args.pass_count} holds no rows")
        if self.corpus.tokens > args.max_pass_tokens:
            raise SystemExit(f"pass {args.pass_index}/{args.pass_count} holds {self.corpus.tokens:,} positions > "
                             f"--max-pass-tokens {args.max_pass_tokens:,}: its host state would be "
                             f"{self.state_bytes(self.corpus.tokens) / 1e9:.1f} GB (design rule: <= 0.5M per pass, "
                             "~71.7 GB); raise --pass-count")
        self.pass_sha = sha256_text(canonical_json({
            "corpus": self.corpus_sha, "pass_index": args.pass_index, "pass_count": args.pass_count,
            "row_limit": args.row_limit, "trim": not args.no_trim, "truncate_seq": args.truncate_seq, "segments": [s.record() for s in self.corpus.segments]}))
        self.run_id = uuid.uuid4().hex[:12]
        # a probe never lands in the cache's own pass dir
        leaf = f"probe-{self.run_id}" if args.probe else f"pass-{args.pass_index}-of-{args.pass_count}"
        self.pass_dir = os.path.abspath(args.pass_dir or os.path.join(args.out_root, self.corpus_sha, leaf))
        self.state_dir = os.path.abspath(args.state_dir or os.path.join(self.pass_dir, "state"))
        self.shard_dir = os.path.join(self.pass_dir, "shards")
        self.write = not args.no_write
        self.runtime_key = {"compute_dtype": args.compute_dtype, "kda_backend": args.kda_backend,
                            "row_chunk_kda": args.row_chunk_kda, "row_chunk_mla": args.row_chunk_mla,
                            "token_chunk": args.token_chunk, "engine_sha256": self.engine.sha256,
                            "numerics_version": NUMERICS_VERSION, "synthetic_blocks": args.synthetic_blocks,
                            "layers": [self.layers[0], self.layers[-1]]}
        self.receipt: dict[str, Any] = {
            "schema": SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
            "tool_sha256": sha256_file(os.path.abspath(__file__)), "run_id": self.run_id, "started_utc": utc(),
            "engine": self.engine.record(), "arguments": vars(args),
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
            "devices": common.device_inventory(), "device": str(self.device),
            "experts_device": str(self.experts_device), "resid_device": str(self.resid_device),
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "num_threads": torch.get_num_threads(),
            "corpus": {"corpus_sha256": self.corpus_sha,
                       "files": [{k: f[k] for k in ("path", "name", "rows", "sequence_length", "input_ids_sha256",
                                                    "real_lengths_sha256", "file_sha256", "real_tokens")}
                                 for f in self.files],
                       "pass_plan": self.plan_record, "pass_index": args.pass_index, "pass_count": args.pass_count,
                       "pass_sha256": self.pass_sha, "segments": [s.record() for s in self.corpus.segments],
                       "positions": self.corpus.tokens,
                       "real_positions": int(sum(int(s.real_lengths.sum()) for s in self.corpus.segments)),
                       "groups": len(self.corpus.groups), "group_tokens": args.group_tokens},
            "layout": {"topk": TOPK, "core_bytes_per_token": CORE_BYTES_PER_TOKEN,
                       "probe_bytes_per_token": PROBE_BYTES_PER_TOKEN,
                       "position_semantics": "position t of a row is K3's distribution over the token at t+1; every "
                                             "REAL position is stored; target_logprob is 0 at a row's last real "
                                             "position (no in-row target)",
                       "tensors": {"topk_ids": "int32 [P, 32]", "topk_logprobs": "bf16 [P, 32] = logit - lse",
                                   "lse": "fp32 [P] logsumexp over all 163,840 K3 logits",
                                   "target_logprob": "fp32 [P] (route probe)",
                                   "row_file": "int32 [R]", "row_index": "int32 [R]", "real_lengths": "int32 [R]"}},
            "host_state_model": {"bytes_per_position": self.state_bytes(1),
                                 "this_pass_bytes": self.state_bytes(self.corpus.tokens),
                                 "what": "prefix_sum + n_blocks_max AttnRes block slots + working h, compute dtype"},
            "layers": {}, "checkpoints": [], "events": []}

    # -- helpers -----------------------------------------------------------------------------
    def _layer_window(self) -> list[int]:
        layers = self.engine.kdp.parse_layers(self.args.layers, self.num_layers) if self.args.layers else \
            list(range(self.num_layers))
        if layers != list(range(layers[0], layers[-1] + 1)):
            raise SystemExit(f"--layers must be one contiguous range, got {self.args.layers}")
        if layers != list(range(self.num_layers)) and not self.args.probe:
            raise SystemExit("a layer window other than the whole stack is a PROBE (timing / memory only); pass --probe")
        if self.args.start_snapshot and (not self.args.probe or layers[0] == 0):
            raise SystemExit("--start-snapshot is a PARITY probe (--probe, a window that starts above layer 0)")
        if self.args.probe and layers[0] > 0 and not self.args.synthetic_blocks and not self.args.start_snapshot \
                and self.common.blocks_before(layers[0], self.block_size) > 0:
            log("probe window above layer 0 without --synthetic-blocks: the block stack starts empty")
        return layers

    def n_blocks_max(self) -> int:
        pushes = sum(1 for i in self.layers if i % self.block_size == 0)
        start = self.common.blocks_before(self.layers[0], self.block_size) if self.args.start_snapshot else 0
        return int(self.args.synthetic_blocks) + start + pushes

    def state_bytes(self, positions: int) -> int:
        dtype = self.compute_dtype or torch.bfloat16
        element = torch.tensor([], dtype=dtype).element_size()
        nb = self.common.blocks_before(self.num_layers, self.block_size) if not self.args.probe else self.n_blocks_max()
        return int(positions) * int(self.index.hidden_size) * element * (nb + 2)

    def runner(self):
        """The fold engine's ``DualStreamRun`` with only what its embedding / head methods read."""
        kdp = self.kdp
        outer = self

        class TeacherRun(kdp.DualStreamRun):
            def __init__(self):  # noqa: D401 -- deliberately not DualStreamRun.__init__ (no fold state)
                self.args = argparse.Namespace(src_dir=outer.args.src_dir, model_dir=outer.args.src_dir,
                                               token_chunk=int(outer.args.token_chunk))
                self.index, self.cfg = outer.index, outer.cfg
                self.device, self.experts_device = outer.device, outer.experts_device
                self.resid_device, self.compute_dtype = outer.resid_device, outer.compute_dtype
                self.corpus, self.stager = outer.corpus, outer.stager
                self.block_size = outer.block_size
                self._head_cache = None

        return TeacherRun()

    def partial(self) -> None:
        if self.write:
            write_json(os.path.join(self.pass_dir, "pass_receipt.partial.json"), self.receipt)

    # -- state: init / checkpoint / resume ---------------------------------------------------
    def init_state(self, run) -> tuple[torch.Tensor, torch.Tensor, int]:
        pair = run.initial_pair()  # the engine's embedding (StreamRun.embed per segment)
        prefix = pair.prefix.contiguous()
        hidden = int(prefix.shape[1])
        blocks = torch.empty((prefix.shape[0], max(self.n_blocks_max(), 1), hidden), dtype=prefix.dtype,
                             device=self.resid_device)
        nb = 0
        if self.args.synthetic_blocks:
            # PROBE ONLY: resident snapshots for the RSS / AttnRes-read cost, not teacher numerics
            for b in range(int(self.args.synthetic_blocks)):
                blocks[:, b].copy_(prefix)
            nb = int(self.args.synthetic_blocks)
        return prefix, blocks, nb

    def init_from_snapshot(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """PARITY ONLY: the fold engine's own TEACHER pair entering ``layers[0]`` (k3_dense_pass
        ``before_L{l}.safetensors``: prefix_sum [N, H], block_residual [N, b, H] over ITS corpus), sliced to
        this pass's rows by ``--snapshot-layout name=token_offset:row_length`` -- so the window and the head
        are compared from the reference's exact input state, with no upstream drift."""
        from safetensors import safe_open

        path = self.args.start_snapshot
        layout = {}
        for item in (self.args.snapshot_layout or "").split(","):
            if item.strip():
                name, rest = item.split("=")
                offset, length = rest.split(":")
                layout[name.strip()] = (int(offset), int(length))
        with safe_open(path, framework="pt") as handle:
            meta = handle.metadata() or {}
            if meta.get("stream", "teacher") != "teacher" or int(meta.get("layer", self.layers[0])) != self.layers[0]:
                raise SystemExit(f"{path} is stream {meta.get('stream')} layer {meta.get('layer')}; need the teacher "
                                 f"pair entering L{self.layers[0]}")
            p_slice, b_slice = handle.get_slice("prefix_sum"), handle.get_slice("block_residual")
            nb = int(b_slice.get_shape()[1])
            if nb != self.common.blocks_before(self.layers[0], self.block_size):
                raise SystemExit(f"{path} carries {nb} blocks; L{self.layers[0]} expects "
                                 f"{self.common.blocks_before(self.layers[0], self.block_size)}")
            hidden = int(p_slice.get_shape()[1])
            dtype = self.compute_dtype or torch.bfloat16
            prefix = torch.empty((self.corpus.tokens, hidden), dtype=dtype)
            blocks = torch.empty((self.corpus.tokens, self.n_blocks_max(), hidden), dtype=dtype)
            for seg, offset in zip(self.corpus.segments, self.corpus.offsets):
                if seg.name not in layout:
                    raise SystemExit(f"--snapshot-layout names no offset for {seg.name}")
                base, length = layout[seg.name]
                if seg.T > length:
                    raise SystemExit(f"{seg.name}: this pass's T {seg.T} exceeds the snapshot row length {length}")
                for r in range(seg.rows):
                    src = base + int(seg.row_index[r]) * length
                    dst = offset + r * seg.T
                    prefix[dst:dst + seg.T] = p_slice[src:src + seg.T].to(dtype)
                    blocks[dst:dst + seg.T, :nb] = b_slice[src:src + seg.T].to(dtype)
        self.receipt["start_snapshot"] = {"path": path, "sha256": sha256_file(path), "metadata_layer": meta.get("layer"),
                                          "metadata_tool": meta.get("tool"),
                                          "metadata_tool_version": meta.get("tool_version"),
                                          "input_ids_sha256": meta.get("input_ids_sha256"), "blocks": nb,
                                          "layout": {k: list(v) for k, v in layout.items()}}
        return prefix, blocks, nb

    def state_manifest_path(self) -> str:
        return os.path.join(self.state_dir, "state.json")

    def load_state(self, blocks_shape: tuple[int, int, int], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, int, int] | None:
        path = self.state_manifest_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("pass_sha256") != self.pass_sha or state.get("runtime_key") != canonical_json(self.runtime_key):
            raise SystemExit(f"{path} belongs to another pass / runtime (pass {state.get('pass_sha256', '')[:12]} vs "
                             f"{self.pass_sha[:12]}); refusing to resume onto it -- move it away or use a new --state-dir")
        from safetensors.torch import load_file

        blocks = torch.empty(blocks_shape, dtype=dtype, device=self.resid_device)
        for b, rec in enumerate(state["blocks"]):
            got = sha256_file(rec["path"])
            if got != rec["sha256"]:
                raise SystemExit(f"state block {b} {rec['path']} sha {got[:12]} != recorded {rec['sha256'][:12]}")
            blocks[:, b].copy_(load_file(rec["path"])["block"])
        rec = state["prefix"]
        got = sha256_file(rec["path"])
        if got != rec["sha256"]:
            raise SystemExit(f"state prefix {rec['path']} sha {got[:12]} != recorded {rec['sha256'][:12]}")
        prefix = load_file(rec["path"])["prefix"].to(self.resid_device)
        self.saved_blocks = list(state["blocks"])
        self.saved_prefix = dict(rec)
        self.receipt["resumed"] = {"from_layer_done": state["layer_done"], "blocks": len(state["blocks"]),
                                   "state_json_sha256": sha256_file(path), "at_utc": utc()}
        if state.get("digests"):
            self.warm_digests_prior = state["digests"]
        return prefix, blocks, int(state["n_blocks"]), int(state["layer_done"])

    def checkpoint(self, layer_done: int, prefix: torch.Tensor, blocks: torch.Tensor, nb: int) -> None:
        save = self.common.save_tensors
        element = prefix.element_size()
        new_blocks = list(range(len(self.saved_blocks), nb))
        need = (len(new_blocks) + 1) * int(prefix.shape[0]) * int(prefix.shape[1]) * element
        os.makedirs(self.state_dir, exist_ok=True)
        free = shutil.disk_usage(self.state_dir).free
        floor = int(self.args.min_free_gib * (1 << 30))
        if free - need < floor:
            event = {"layer_done": layer_done, "action": "checkpoint_skipped_low_disk", "free_bytes": free,
                     "needed_bytes": need, "floor_bytes": floor, "state_dir": self.state_dir}
            self.receipt["checkpoints"].append(event)
            log(f"WARNING checkpoint after L{layer_done:02d} skipped: {free/1e9:.1f} GB free - {need/1e9:.1f} GB "
                f"< {floor/1e9:.1f} GB floor on {self.state_dir}")
            return
        started = time.time()
        for b in new_blocks:
            path = os.path.join(self.state_dir, f"block_{b}.safetensors")
            digest = save(path, {"block": blocks[:, b]}, metadata={"tool": TOOL_NAME, "pass_sha256": self.pass_sha,
                                                                    "block": str(b)})
            self.saved_blocks.append({"path": path, "sha256": digest})
        path = os.path.join(self.state_dir, f"prefix_L{layer_done:02d}.safetensors")
        digest = save(path, {"prefix": prefix}, metadata={"tool": TOOL_NAME, "pass_sha256": self.pass_sha,
                                                          "layer_done": str(layer_done)})
        old = self.saved_prefix.get("path") if self.saved_prefix else None
        self.saved_prefix = {"path": path, "sha256": digest, "layer_done": layer_done}
        write_json(self.state_manifest_path(), {
            "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "pass_sha256": self.pass_sha,
            "runtime_key": canonical_json(self.runtime_key), "layer_done": layer_done, "n_blocks": nb,
            "blocks": self.saved_blocks, "prefix": self.saved_prefix,
            # a copy (atomic under the GIL): the warmer thread adds entries while this serialises; it carries the
            # checkpoint-seeded digests forward, so a second resume still has them
            "digests": dict(self.warmer.results) if self.warmer else None, "written_utc": utc()})
        if old and os.path.abspath(old) != os.path.abspath(path) and os.path.exists(old):
            os.remove(old)
        self.receipt["checkpoints"].append({"layer_done": layer_done, "blocks_written": new_blocks,
                                            "bytes": need, "seconds": round(time.time() - started, 1)})

    # -- the pass ---------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        args, kdp, common = self.args, self.kdp, self.common
        if self.write:
            os.makedirs(self.pass_dir, exist_ok=True)
        # teacher integrity BEFORE: stat, against the fold receipt's metadata
        self.integrity_before = source_stat(self.index, args.src_dir)
        fold_ref = expected_from_fold_receipt(args.fold_receipt)
        drift = compare_stat(self.integrity_before, fold_ref, what="before vs fold receipt") if fold_ref else []
        self.receipt["integrity"] = {"before": {k: v for k, v in self.integrity_before.items() if k != "files"},
                                     "files_stat_sha256": sha256_text(canonical_json(self.integrity_before["files"])),
                                     "fold_receipt": {k: v for k, v in (fold_ref or {}).items() if k != "files"} or None,
                                     "drift_vs_fold_receipt": drift,
                                     "fold_receipt_unusable_override": fold_ref is None}
        if fold_ref is None and not args.allow_source_drift:
            raise SystemExit(f"no usable fold receipt at {args.fold_receipt!r} (missing, or no "
                             "resume_context.model.source_file_metadata): the K3 source cannot be checked against the "
                             "fold of record -- --allow-source-drift records an override")
        if drift and not args.allow_source_drift:
            raise SystemExit("the K3 source differs from the fold receipt's resume_context (size/mtime): "
                             + "; ".join(drift[:4]) + " -- --allow-source-drift records an override")
        digests = expected_digests(args.digest_receipt)
        self.receipt["integrity"]["digest_receipt"] = {k: v for k, v in (digests or {}).items() if k != "files"} or None

        plan_shards = {i: self.index.shards_for(self.index.layer_keys(i)) for i in self.layers}
        model_level = sorted({self.index.weight_map[k] for k in self.index.weight_map
                              if any(k.endswith(s) for s in ("embed_tokens.weight", "lm_head.weight",
                                                             "model.norm.weight", "output_attn_res_proj.weight",
                                                             "output_attn_res_norm.weight"))})
        order: list[str] = []
        for shard in [s for i in self.layers for s in plan_shards[i]] + model_level:
            if shard not in order:
                order.append(shard)
        if args.inline_digest:
            unverifiable = digest_receipt_problems(digests, order, args.digest_receipt)
            if unverifiable:
                raise SystemExit("--inline-digest cannot verify this pass: " + "; ".join(unverifiable[:6])
                                 + (f" (+{len(unverifiable) - 6} more)" if len(unverifiable) > 6 else ""))

        self.stager = kdp.ShardStager(args.src_dir, None, enabled=False, window=1, prefetch=0,
                                      delete_after_consume=False, run_id=self.run_id)
        run = self.runner()
        self.saved_blocks: list[dict[str, Any]] = []
        self.saved_prefix: dict[str, Any] | None = None
        self.warmer = None
        self.warm_digests_prior = None
        hidden = int(self.index.hidden_size)
        dtype = self.compute_dtype or torch.bfloat16
        state = None
        if args.resume and self.write and not args.probe:
            state = self.load_state((self.corpus.tokens, max(self.n_blocks_max(), 1), hidden), dtype)
        if args.prefetch_next or args.inline_digest:
            # a resume reads only what the remaining layers (and the head) read; a shard only earlier layers
            # read is skipped -- with --inline-digest only when the checkpoint already holds its digest
            first = state[3] + 1 if state is not None else self.layers[0]
            needed = set(model_level) | {s for i in self.layers if i >= first for s in plan_shards[i]}
            prior = {s: r for s, r in (self.warm_digests_prior or {}).items()
                     if s in order and r and r.get("sha256")} if args.inline_digest else {}
            skip = {s for s in order if s not in needed and (not args.inline_digest or s in prior)}
            self.warmer = ShardWarmer(args.src_dir, order, ahead=1 if args.prefetch_next else 0,
                                      digest=args.inline_digest, skip=skip, prior=prior)
            self.warmer.start()
        try:
            self.run_layers_and_head(run, state, plan_shards, model_level)
        except BaseException:
            if self.warmer is not None:
                self.warmer.close()
            raise
        return self.finish(digests)

    def run_layers_and_head(self, run, state, plan_shards: dict[int, list[str]], model_level: list[str]) -> None:
        args = self.args
        with torch.no_grad():
            if state is None and args.start_snapshot:
                t0 = time.time()
                prefix, blocks, nb = self.init_from_snapshot()
                start = self.layers[0]
                self.receipt["start"] = {"from": "reference_snapshot", "layer": start,
                                         "seconds": round(time.time() - t0, 1)}
            elif state is None:
                t0 = time.time()
                prefix, blocks, nb = self.init_state(run)
                start = self.layers[0]
                self.receipt["start"] = {"from": "embedding", "seconds": round(time.time() - t0, 1),
                                         "synthetic_blocks": int(args.synthetic_blocks)}
            else:
                prefix, blocks, nb, done = state
                start = done + 1
                self.receipt["start"] = {"from": "checkpoint", "layer_done": done}
                log(f"resumed after L{done:02d} with {nb} blocks from {self.state_dir}")
            self.receipt["host_state"] = {"prefix_bytes": prefix.numel() * prefix.element_size(),
                                          "block_slots": int(blocks.shape[1]),
                                          "block_slot_bytes_reserved": blocks.numel() * blocks.element_size()}
            for layer_idx in [i for i in self.layers if i >= start]:
                nb = self.layer(run, layer_idx, prefix, blocks, nb, plan_shards)
                if self.warmer is not None and self.warmer.error is not None:
                    raise SystemExit(f"shard warmer failed: {self.warmer.error!r}")
                last = layer_idx == self.layers[-1]
                if self.write and not args.probe and args.checkpoint_every > 0 and \
                        ((layer_idx + 1) % args.checkpoint_every == 0 or last):
                    self.checkpoint(layer_idx, prefix, blocks, nb)
                self.partial()
            if self.cuda and not args.hold_device:
                torch.cuda.empty_cache()
            if self.head_runs():
                if self.warmer is not None:
                    self.warmer.advance(model_level)
                self.head(run, prefix, blocks, nb)

    def head_runs(self) -> bool:
        ends = self.layers[-1] == self.num_layers - 1
        return ends and not self.args.no_head and (self.full_stack or bool(self.args.start_snapshot))

    def layer(self, run, layer_idx: int, prefix: torch.Tensor, blocks: torch.Tensor, nb: int,
              plan_shards: dict[int, list[str]]) -> int:
        args, kdp, common = self.args, self.kdp, self.common
        if self.warmer is not None:
            self.warmer.advance(plan_shards[layer_idx])
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        rss_before = vmrss_bytes()
        t0 = time.time()
        weights, raw_body, timing = kdp.read_layer(self.index, layer_idx, self.stager, self.device,
                                                   self.experts_device, self.compute_dtype,
                                                   progress_tag=f"L{layer_idx:02d}" if args.verbose else None)
        del raw_body
        if self.cuda:
            torch.cuda.synchronize(self.device)
        read_wall = time.time() - t0
        runner = self.engine.layer.StreamedLayer(weights, self.cfg, kda_backend=args.kda_backend)
        snapshot = (layer_idx % self.block_size) == 0
        if snapshot and nb >= blocks.shape[1]:
            raise SystemExit(f"L{layer_idx}: block slot {nb} beyond the {blocks.shape[1]} reserved")
        row_chunk = args.row_chunk_kda if weights.is_kda else args.row_chunk_mla
        dequant = 0.0
        compute_start = time.time()
        for si, r0, r1, g0, g1 in self.corpus.groups:
            seg = self.corpus.segments[si]
            pair = kdp.Pair(prefix[g0:g1], blocks[g0:g1, :nb])
            view = GroupView(seg, r0, r1)
            carried, ffn_in, _ = kdp.attention_half(runner, pair, view, self.device,
                                                    row_chunk=row_chunk, token_chunk=args.token_chunk)
            new_prefix, taps = kdp.ffn_half(runner, carried, ffn_in, self.cfg, device=self.device,
                                            token_chunk=args.token_chunk)
            dequant += float(taps.get("dequant_seconds", 0.0))
            del carried, ffn_in, taps
            kdp.require_finite("teacher prefix after the layer", new_prefix, layer_idx)
            if snapshot:
                blocks[g0:g1, nb].copy_(prefix[g0:g1])  # the prefix ENTERING this layer is the pushed block
            prefix[g0:g1].copy_(new_prefix.to(self.resid_device))
            del new_prefix, pair
        if self.cuda:
            torch.cuda.synchronize(self.device)
        compute_wall = time.time() - compute_start
        if snapshot:
            nb += 1
        entry = {
            "layer": layer_idx, "is_kda": bool(weights.is_kda), "moe": bool(weights.is_moe), "snapshot": snapshot,
            "read": {k: timing[k] for k in ("bytes", "gigabytes", "seconds", "MB_per_s", "compute_dtype",
                                            "compute_dtype_source")},
            "read_wall_seconds": round(read_wall, 2),
            "compute_seconds": round(compute_wall, 2),
            "dequant_seconds_engine": round(dequant, 2),
            "dequant_note": ("k3_dense_pass.routed_y_lat's own wall around dequant_expert + .to(device); it is the "
                             "true dequant time only with CUDA_LAUNCH_BLOCKING=1 (see cuda_launch_blocking)"),
            "compute_minus_dequant_seconds": round(compute_wall - dequant, 2),
            "row_chunk": row_chunk, "groups": len(self.corpus.groups), "blocks_after": nb,
            "layer_bytes_on_device": weights.device_bytes(self.device),
            "device_peak_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.cuda else None,
            "rss_before_bytes": rss_before, "rss_after_bytes": vmrss_bytes(),
            "peak_rss_bytes": common.peak_rss_bytes(),
        }
        if args.prefix_digests:
            entry["prefix_out_sha256"] = common.sha256_tensor(prefix)
        if self.warmer is not None:
            entry["warmer"] = {s: self.warmer.results.get(s) for s in plan_shards[layer_idx]}
        self.receipt["layers"][str(layer_idx)] = entry
        log(f"L{layer_idx:02d} {'KDA' if weights.is_kda else 'MLA'}: read {read_wall:.1f}s "
            f"({timing['MB_per_s']} MB/s), compute {compute_wall:.1f}s (engine dequant {dequant:.1f}s), "
            f"blocks {nb}, device peak {entry['device_peak_bytes'] or 0:,} B, rss {entry['rss_after_bytes'] or 0:,} B")
        del weights, runner
        if self.cuda and not self.args.hold_device:
            torch.cuda.empty_cache()
        return nb

    # -- the output head: top-32 per real position ------------------------------------------
    def head(self, run, prefix: torch.Tensor, blocks: torch.Tensor, nb: int) -> None:
        args, kdp = self.args, self.kdp
        pair = kdp.Pair(prefix, blocks[:, :nb])
        progress_path = os.path.join(self.pass_dir, "head_progress.json")
        progress: dict[str, Any] = {}
        if args.resume and os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as handle:
                prior = json.load(handle)
            if prior.get("pass_sha256") == self.pass_sha and prior.get("runtime_key") == canonical_json(self.runtime_key):
                progress = prior.get("groups", {})
        started = time.time()
        self.shards: list[dict[str, Any]] = []
        for gi, (si, r0, r1, g0, g1) in enumerate(self.corpus.groups):
            seg = self.corpus.segments[si]
            path = os.path.join(self.shard_dir, f"group-{gi:05d}.safetensors")
            done = progress.get(str(gi))
            if done and os.path.exists(path) and sha256_file(path) == done["sha256"]:
                self.shards.append(done)
                continue
            payload = self.head_group(run, pair, seg, si, r0, r1)
            meta = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "corpus_sha256": self.corpus_sha,
                    "pass_sha256": self.pass_sha, "group": str(gi), "segment": seg.name,
                    "rows": f"{int(seg.row_index[r0])}-{int(seg.row_index[r1 - 1])}"}
            finite = self.finite_check(payload, gi)
            if self.write:
                digest = self.common.save_tensors(path, payload, metadata=meta)
                rec = {"group": gi, "path": path, "sha256": digest, "bytes": os.path.getsize(path),
                       "segment": seg.name, "file_index": seg.file_index, "rows": r1 - r0,
                       "positions": int(payload["lse"].numel()), "finite": finite}
                progress[str(gi)] = rec
                write_json(progress_path, {"pass_sha256": self.pass_sha,
                                           "runtime_key": canonical_json(self.runtime_key), "groups": progress})
            else:
                rec = {"group": gi, "positions": int(payload["lse"].numel()), "finite": finite}
            self.shards.append(rec)
        self.receipt["head"] = {"seconds": round(time.time() - started, 1), "groups": len(self.shards)}

    def head_group(self, run, pair, seg: CacheSegment, si: int, r0: int, r1: int) -> dict[str, torch.Tensor]:
        offset = self.corpus.offsets[si]
        ids_out, lp_out, lse_out, tlp_out = [], [], [], []
        step = max(1, int(self.args.head_row_chunk))
        for a in range(r0, r1, step):
            b = min(a + step, r1)
            start, stop = offset + a * seg.T, offset + b * seg.T
            logits = run.logits_rows(pair, start, stop, b - a, seg.T)  # the engine's head: fp32 [rows, T, V]
            for r in range(b - a):
                length = int(seg.real_lengths[a + r])
                row = logits[r, :length]
                lse = torch.logsumexp(row, dim=-1)
                top = row.topk(TOPK, dim=-1)
                target = seg.ids[a + r, 1:length].to(row.device)
                tlp = torch.zeros(length, dtype=torch.float32, device=row.device)
                if length > 1:
                    tlp[: length - 1] = row[: length - 1].gather(1, target.unsqueeze(1)).squeeze(1) - lse[: length - 1]
                ids_out.append(top.indices.to(torch.int32).cpu())
                lp_out.append((top.values - lse.unsqueeze(1)).to(torch.bfloat16).cpu())
                lse_out.append(lse.float().cpu())
                tlp_out.append(tlp.cpu())
                del row, lse, top, target, tlp
            del logits
        return {"topk_ids": torch.cat(ids_out).contiguous(), "topk_logprobs": torch.cat(lp_out).contiguous(),
                "lse": torch.cat(lse_out).contiguous(), "target_logprob": torch.cat(tlp_out).contiguous(),
                "row_file": torch.full((r1 - r0,), seg.file_index, dtype=torch.int32),
                "row_index": seg.row_index[r0:r1].to(torch.int32).contiguous(),
                "real_lengths": seg.real_lengths[r0:r1].to(torch.int32).contiguous()}

    def finite_check(self, payload: dict[str, torch.Tensor], gi: int) -> dict[str, Any]:
        bad = {k: int((~torch.isfinite(payload[k].float())).sum()) for k in ("topk_logprobs", "lse", "target_logprob")}
        vocab = int(self.cfg["vocab_size"])
        ids = payload["topk_ids"]
        out_of_range = int(((ids < 0) | (ids >= vocab)).sum())
        if any(bad.values()) or out_of_range:
            raise SystemExit(f"group {gi}: non-finite or out-of-range cache values {bad}, ids out of range "
                             f"{out_of_range}; nothing written for this group")
        return {"non_finite": bad, "ids_out_of_range": out_of_range,
                "positions": int(payload["lse"].numel()), "rows": int(payload["real_lengths"].numel()),
                "positions_match_real_lengths": int(payload["real_lengths"].sum()) == int(payload["lse"].numel())}

    # -- the seal ---------------------------------------------------------------------------
    def finish(self, digests: dict[str, Any] | None) -> dict[str, Any]:
        args = self.args
        if self.warmer is not None:
            if args.inline_digest:
                self.warmer.wait_all()
            else:
                self.warmer.close()
            if self.warmer.error is not None:
                raise SystemExit(f"shard warmer failed: {self.warmer.error!r}")
        after = source_stat(self.index, args.src_dir)
        changed = compare_stat(after, self.integrity_before, what="after vs before")
        self.receipt["integrity"]["after_equals_before"] = not changed
        self.receipt["integrity"]["changed"] = changed
        inline = None
        problems = list(changed)
        if self.warmer is not None:
            self.receipt["integrity"]["warmer"] = {"order": len(self.warmer.order), "skipped": sorted(self.warmer.skip),
                                                   "digests_from_checkpoint": self.warmer.seeded,
                                                   "read": list(self.warmer.read_log)}
        if args.inline_digest:
            # every shard of the pass hashed (here or, for a shard only earlier layers read, before the resume)
            # and every hash checked against the receipt's hf_sha256 -- or the pass is not sealed
            results = dict(self.warmer.results) if self.warmer is not None else {}
            order = self.warmer.order if self.warmer is not None else []
            files = (digests or {}).get("files") or {}
            mismatches, unchecked, checked = [], [], 0
            for shard, rec in sorted(results.items()):
                want = files.get(shard)
                if not want or not want.get("sha256") or not rec.get("sha256"):
                    unchecked.append(shard)
                    continue
                checked += 1
                if rec["sha256"] != want["sha256"] or (want.get("bytes") and int(want["bytes"]) != int(rec["bytes"])):
                    mismatches.append({"shard": shard, "got": rec["sha256"], "want": want["sha256"]})
            not_hashed = [s for s in order if s not in results]
            inline = {"hashed": len(results), "checked_against_receipt": checked, "compares": "hf_sha256",
                      "mismatches": mismatches, "unchecked": unchecked, "not_hashed": not_hashed,
                      "all_hashed_match": bool(results) and checked == len(results) and not mismatches
                      and not unchecked and not not_hashed,
                      "results": results}
            problems += [f"inline digest mismatch {m['shard']}" for m in mismatches]
            problems += [f"inline digest unchecked {s}" for s in unchecked]
            problems += [f"inline digest never hashed {s}" for s in not_hashed]
            if not results:
                problems.append("inline digest hashed nothing")
        self.receipt["integrity"]["inline_digest"] = inline
        self.receipt["finished_utc"] = utc()
        self.receipt["peak_rss_bytes"] = self.common.peak_rss_bytes()
        layers = self.receipt["layers"]
        if layers:
            self.receipt["totals"] = {
                "layers": len(layers),
                "read_wall_seconds": round(sum(v["read_wall_seconds"] for v in layers.values()), 1),
                "compute_seconds": round(sum(v["compute_seconds"] for v in layers.values()), 1),
                "dequant_seconds_engine": round(sum(v["dequant_seconds_engine"] for v in layers.values()), 1)}
        if getattr(self, "shards", None) is not None:
            positions = sum(s["positions"] for s in self.shards)
            self.receipt["shards"] = self.shards
            self.receipt["cache"] = {"positions": positions,
                                     "bytes_on_disk": sum(s.get("bytes", 0) for s in self.shards),
                                     "core_bytes": positions * CORE_BYTES_PER_TOKEN,
                                     "probe_bytes": positions * PROBE_BYTES_PER_TOKEN}
        self.receipt["verdict"] = "refused" if problems else "sealed"
        self.receipt["problems"] = problems
        if self.write:
            name = "manifest.json" if self.head_runs() else "probe_receipt.json"
            self.receipt["kind"] = ("teacher_cache" if not args.probe else
                                    "parity_from_reference_snapshot" if args.start_snapshot else "probe")
            path = os.path.join(self.pass_dir, name)
            digest = write_json(path, self.receipt)
            with open(path + ".sha256", "w", encoding="utf-8") as handle:
                handle.write(f"{digest}  {name}\n")
            log(f"{name} -> {path} (sha256 {digest[:12]}...)")
            if name == "manifest.json" and not problems and not args.keep_state and os.path.isdir(self.state_dir):
                shutil.rmtree(self.state_dir)
                self.receipt["state_removed"] = self.state_dir
        if problems:
            raise SystemExit("teacher integrity failed: " + "; ".join(problems[:6]))
        return self.receipt


# --------------------------------------------------------------------------
# reading a sealed cache back
# --------------------------------------------------------------------------
class CacheReader:
    """Every (file name, row) of one or more sealed pass dirs -> its per-position arrays."""

    def __init__(self, pass_dirs: list[str]):
        from safetensors.torch import load_file

        self.rows: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.manifests = []
        self.files: dict[str, dict[str, Any]] = {}
        for pass_dir in pass_dirs:
            path = os.path.join(pass_dir, "manifest.json")
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("verdict") != "sealed":
                raise SystemExit(f"{path} is not sealed (verdict {manifest.get('verdict')})")
            self.manifests.append({"path": path, "sha256": sha256_file(path), "pass_sha256": manifest["corpus"]["pass_sha256"],
                                   "corpus_sha256": manifest["corpus"]["corpus_sha256"]})
            names = [f["name"] for f in manifest["corpus"]["files"]]
            for f in manifest["corpus"]["files"]:
                self.files[f["name"]] = f
            for shard in manifest["shards"]:
                if sha256_file(shard["path"]) != shard["sha256"]:
                    raise SystemExit(f"{shard['path']} does not match its manifest sha256")
                got = load_file(shard["path"])
                pos = 0
                for i in range(int(got["real_lengths"].numel())):
                    n = int(got["real_lengths"][i])
                    key = (names[int(got["row_file"][i])], int(got["row_index"][i]))
                    self.rows[key] = {k: got[k][pos:pos + n] for k in ("topk_ids", "topk_logprobs", "lse",
                                                                        "target_logprob")}
                    pos += n


# --------------------------------------------------------------------------
# parity vs the fold engine's review dump
# --------------------------------------------------------------------------
def parity_metrics(cache: CacheReader, reference_path: str, file_name: str | None = None) -> dict[str, Any]:
    from safetensors.torch import load_file

    ref = load_file(reference_path)
    name = file_name or os.path.basename(reference_path).replace("_teacher_forced.safetensors", "")
    rows, positions = ref["row_index"].long(), ref["position"].long()
    k_ref = int(ref["teacher_topk_ids"].shape[1])
    mine_ids, mine_lp, mine_tlp, missing = [], [], [], 0
    keep = []
    for i in range(rows.numel()):
        got = cache.rows.get((name, int(rows[i])))
        if got is None or int(positions[i]) >= got["lse"].numel():
            missing += 1
            continue
        keep.append(i)
        p = int(positions[i])
        mine_ids.append(got["topk_ids"][p])
        mine_lp.append(got["topk_logprobs"][p].float())
        mine_tlp.append(got["target_logprob"][p])
    if not keep:
        raise SystemExit(f"no reference position of {name} is in the cache")
    idx = torch.tensor(keep)
    ids = torch.stack(mine_ids)
    lps = torch.stack(mine_lp)
    tlp = torch.stack(mine_tlp).float()
    r_ids = ref["teacher_topk_ids"][idx].long()
    r_lp = ref["teacher_topk_logprobs"][idx].float()
    r_tlp = ref["teacher_next_token_logprob"][idx].float()
    r_arg = ref["teacher_argmax"][idx].long()
    top1 = (ids[:, 0].long() == r_arg).float()
    k = min(k_ref, ids.shape[1])
    overlap = torch.tensor([len(set(ids[i, :k].tolist()) & set(r_ids[i, :k].tolist())) / k for i in range(ids.shape[0])])
    d_t = (tlp - r_tlp).abs()
    agree = top1.bool()
    d_top1 = (lps[agree, 0] - r_lp[agree, 0]).abs()
    return {"reference": reference_path, "reference_sha256": sha256_file(reference_path), "file": name,
            "positions_reference": int(rows.numel()), "positions_compared": len(keep), "positions_missing": missing,
            "reference_topk": k_ref, "compared_topk": k,
            "top1_agreement": float(top1.mean()),
            "top16_overlap_mean": float(overlap.mean()), "top16_overlap_min": float(overlap.min()),
            "target_logprob_abs_diff": {"mean": float(d_t.mean()), "median": float(d_t.median()),
                                        "p99": float(torch.quantile(d_t, 0.99)), "max": float(d_t.max())},
            "top1_logprob_abs_diff_where_agree": {"mean": float(d_top1.mean()) if d_top1.numel() else None,
                                                  "max": float(d_top1.max()) if d_top1.numel() else None},
            "non_finite": int((~torch.isfinite(tlp)).sum() + (~torch.isfinite(lps)).sum())}


def parity_verdict(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    t = PARITY_THRESHOLDS
    total = sum(m["positions_compared"] for m in metrics)

    def wmean(key):
        return sum(m[key] * m["positions_compared"] for m in metrics) / max(total, 1)

    pooled = {"positions": total, "top1_agreement": wmean("top1_agreement"),
              "top16_overlap_mean": wmean("top16_overlap_mean"),
              "target_logprob_abs_diff_median_max_over_files": max(m["target_logprob_abs_diff"]["median"] for m in metrics),
              "target_logprob_abs_diff_p99_max_over_files": max(m["target_logprob_abs_diff"]["p99"] for m in metrics),
              "non_finite": sum(m["non_finite"] for m in metrics),
              "positions_missing": sum(m["positions_missing"] for m in metrics)}
    gates = {
        "top1_agreement": pooled["top1_agreement"] >= t["top1_agreement_min"],
        "top16_overlap_mean": pooled["top16_overlap_mean"] >= t["top16_overlap_mean_min"],
        "target_logprob_median": pooled["target_logprob_abs_diff_median_max_over_files"] <= t["target_logprob_abs_diff_median_max"],
        "target_logprob_p99": pooled["target_logprob_abs_diff_p99_max_over_files"] <= t["target_logprob_abs_diff_p99_max"],
        "finite": pooled["non_finite"] <= t["non_finite_max"],
        "complete": pooled["positions_missing"] == 0,
    }
    return {"thresholds": t, "pooled": pooled, "gates": gates, "verdict": "pass" if all(gates.values()) else "fail"}


# --------------------------------------------------------------------------
# the cross-tokenizer target and the route probe
# --------------------------------------------------------------------------
def bind_tokenizers(args) -> tuple[Any, Any, dict[str, Any]]:
    """``k3_corpus_build.load_sides``: the exact byte spellings the boundary map was built with."""
    tools = args.corpus_tools_dir
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import k3_corpus_build as kcb  # noqa: E402

    kcb.bind_lanes(chimera_src=args.chimera_src)
    from pathlib import Path

    qwen, k3, report = kcb.load_sides(qwen_dir=Path(args.qwen_dir), k3_dir=Path(args.k3_dir),
                                      qwen_separator_id=int(args.qwen_separator_id), k3_separator_id=None)
    report["k3_corpus_build"] = {"path": os.path.realpath(kcb.__file__), "sha256": sha256_file(kcb.__file__)}
    report["chimera_src"] = str(kcb.resolve_chimera_src(args.chimera_src))
    return qwen, k3, report


def build_vocab_map(k3, qwen, k3_vocab: int, qwen_limit: int = QWEN_TEXT_ROWS) -> tuple[torch.Tensor, dict[str, Any]]:
    """K3 id -> Qwen id with the IDENTICAL byte string (or -1).  Qwen rows >= ``qwen_limit`` and Qwen
    specials (which spell their literal names) are never targets; the two separators map to each other
    (the boundary map pairs them explicitly); other K3 specials spell no bytes and map to -1."""
    reverse: dict[bytes, int] = {}
    collisions = 0
    for qid in range(min(int(qwen.vocab_size), int(qwen_limit))):
        if qid == qwen.separator_id or qid in qwen.special_ids:
            continue
        try:
            raw = qwen.token_bytes(qid)
        except (Exception, SystemExit):
            continue  # an undefined row
        if not raw:
            continue
        if raw in reverse:
            collisions += 1
            continue
        reverse[raw] = qid
    mapping = torch.full((int(k3_vocab),), -1, dtype=torch.int32)
    mapped = specials = 0
    for kid in range(int(k3_vocab)):
        if kid == k3.separator_id:
            if qwen.separator_id < qwen_limit:
                mapping[kid] = qwen.separator_id
                mapped += 1
            continue
        if kid in k3.special_ids:
            specials += 1
            continue
        try:
            raw = k3.token_bytes(kid)
        except (Exception, SystemExit):
            continue  # an id the encoding does not define
        qid = reverse.get(raw) if raw else None
        if qid is not None:
            mapping[kid] = qid
            mapped += 1
    stats = {"k3_vocab": int(k3_vocab), "k3_mapped": mapped, "k3_specials_unmapped": specials,
             "qwen_targets_considered": len(reverse), "qwen_limit": int(qwen_limit),
             "qwen_byte_collisions_skipped": collisions,
             "map_sha256": hashlib.sha256(mapping.numpy().tobytes()).hexdigest()}
    return mapping, stats


def qwen_name_for(k3_name: str) -> str:
    return "qwen_" + k3_name[len("k3_"):] if k3_name.startswith("k3_") else k3_name


def aligned_pairs(row: dict[str, Any]) -> torch.Tensor:
    """[qwen_pos, k3_pos] pairs where BOTH sides still have an in-row next token (both positions predict)."""
    pairs = torch.tensor(row["pairs"], dtype=torch.long).reshape(-1, 2)
    if pairs.numel() == 0:
        return pairs
    keep = (pairs[:, 0] < int(row["qwen_tokens_real"]) - 1) & (pairs[:, 1] < int(row["k3_tokens_real"]) - 1)
    return pairs[keep]


def xtok_rows(cache: CacheReader, boundary: dict[str, Any], mapping: torch.Tensor,
              qwen_ids: dict[str, torch.Tensor] | None, k3_ids: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    q_file, q_row, q_pos, k_pos, ids_out, lp_out, matched, topk_mass, next_match = ([] for _ in range(9))
    names: list[str] = []
    rows_used = rows_skipped = 0
    for row in boundary["rows"]:
        key = (row["file"], int(row["row_index"]))
        got = cache.rows.get(key)
        if got is None:
            rows_skipped += 1
            continue
        pairs = aligned_pairs(row)
        if pairs.numel() == 0:
            continue
        rows_used += 1
        qname = qwen_name_for(row["file"])
        if qname not in names:
            names.append(qname)
        k = pairs[:, 1]
        ids = got["topk_ids"][k].long()
        lp = got["topk_logprobs"][k].float()
        mapped = mapping[ids]
        p = lp.exp()
        m = (p * (mapped >= 0).float()).sum(1)
        q_file.append(torch.full((pairs.shape[0],), names.index(qname), dtype=torch.int32))
        q_row.append(torch.full((pairs.shape[0],), int(row["row_index"]), dtype=torch.int32))
        q_pos.append(pairs[:, 0].to(torch.int32))
        k_pos.append(k.to(torch.int32))
        ids_out.append(mapped.to(torch.int32))
        lp_out.append(got["topk_logprobs"][k])
        matched.append(m.float())
        topk_mass.append(p.sum(1).float())
        if qwen_ids is not None and k3_ids is not None and qname in qwen_ids and row["file"] in k3_ids:
            k_next = k3_ids[row["file"]][int(row["row_index"]), k + 1].long()
            q_next = qwen_ids[qname][int(row["row_index"]), pairs[:, 0] + 1].long()
            next_match.append((mapping[k_next].long() == q_next).to(torch.uint8))
    if not rows_used:
        raise SystemExit("no boundary-map row is in the cache")
    out = {"q_file": torch.cat(q_file), "q_row": torch.cat(q_row), "q_pos": torch.cat(q_pos),
           "k3_pos": torch.cat(k_pos), "qwen_ids": torch.cat(ids_out).contiguous(),
           "k3_logprobs": torch.cat(lp_out).contiguous(), "matched_mass": torch.cat(matched),
           "topk_mass": torch.cat(topk_mass)}
    out["tail_mass"] = (1.0 - out["matched_mass"]).clamp_min(0.0)
    if next_match:
        out["next_token_identical"] = torch.cat(next_match)
    return {"tensors": out, "qwen_files": names, "rows_used": rows_used, "rows_not_in_cache": rows_skipped}


def token_byte_lengths(side, vocab: int) -> torch.Tensor:
    lengths = torch.zeros(int(vocab), dtype=torch.long)
    for i in range(int(vocab)):
        try:
            raw = side.token_bytes(i)
        except (Exception, SystemExit):
            raw = None
        lengths[i] = len(raw) if raw else 0
    return lengths


def probe_rows(cache: CacheReader, boundary: dict[str, Any], k3_ids: dict[str, torch.Tensor],
               k3_len: torch.Tensor, routes: dict[tuple[str, int], str],
               q_target: dict[str, torch.Tensor] | None, qwen_ids: dict[str, torch.Tensor] | None,
               q_len: torch.Tensor | None) -> dict[str, Any]:
    """bits/byte on [first aligned boundary, last aligned boundary] of each paired row, both sides."""
    per_route: dict[str, dict[str, float]] = {}
    rows_out = []
    span_mismatch = 0
    for row in boundary["rows"]:
        key = (row["file"], int(row["row_index"]))
        got = cache.rows.get(key)
        if got is None:
            continue
        pairs = torch.tensor(row["pairs"], dtype=torch.long).reshape(-1, 2)
        pairs = pairs[(pairs[:, 0] <= int(row["qwen_tokens_real"]) - 1) & (pairs[:, 1] <= int(row["k3_tokens_real"]) - 1)]
        if pairs.shape[0] < 2:
            continue
        (qa, ka), (qb, kb) = pairs[0].tolist(), pairs[-1].tolist()
        ids = k3_ids[row["file"]][int(row["row_index"])]
        span_bytes = int(k3_len[ids[ka + 1: kb + 1].long()].sum())
        if span_bytes <= 0:
            continue
        k3_nats = float(-got["target_logprob"][ka:kb].double().sum())
        rec = {"file": row["file"], "row": int(row["row_index"]), "route": routes.get(key, "unknown"),
               "bytes": span_bytes, "k3_tokens": kb - ka, "k3_nats": k3_nats,
               "k3_bpb": k3_nats / math.log(2) / span_bytes}
        qname = qwen_name_for(row["file"])
        if q_target is not None and qname in q_target:
            q_nats = float(-q_target[qname][int(row["row_index"]), qa:qb].double().sum())
            rec.update({"q_tokens": qb - qa, "q_nats": q_nats, "q_bpb": q_nats / math.log(2) / span_bytes})
            if qwen_ids is not None and q_len is not None:
                q_bytes = int(q_len[qwen_ids[qname][int(row["row_index"]), qa + 1: qb + 1].long()].sum())
                if q_bytes != span_bytes:
                    span_mismatch += 1
                    rec["q_span_bytes"] = q_bytes
        rows_out.append(rec)
        agg = per_route.setdefault(rec["route"], {"rows": 0, "bytes": 0, "k3_nats": 0.0, "q_nats": 0.0, "q_rows": 0})
        agg["rows"] += 1
        agg["bytes"] += span_bytes
        agg["k3_nats"] += k3_nats
        if "q_nats" in rec:
            agg["q_nats"] += rec["q_nats"]
            agg["q_rows"] += 1
    for route, agg in per_route.items():
        agg["k3_bpb"] = agg["k3_nats"] / math.log(2) / max(agg["bytes"], 1)
        if agg["q_rows"] == agg["rows"] and agg["rows"]:
            agg["q_bpb"] = agg["q_nats"] / math.log(2) / max(agg["bytes"], 1)
            agg["k3_minus_q_bpb"] = agg["k3_bpb"] - agg["q_bpb"]
    return {"per_route": per_route, "rows": rows_out, "q_span_byte_mismatches": span_mismatch,
            "note": "informs lambda_K3 and the route mix only; K3 is distilled in every outcome"}


def load_ids_by_name(paths: list[str]) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    out = {}
    for p in paths or []:
        out[os.path.splitext(os.path.basename(p))[0]] = load_file(p)["input_ids"].to(torch.long)
    return out


def load_routes(path: str | None, key: str) -> dict[tuple[str, int], str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        got = json.load(handle)
    routes = {}
    for row in got.get("row_list", got.get("rows", [])):
        value = row
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            value = row.get("source", "unknown")
        routes[(row["file"], int(row["row_index"]))] = str(value)
    return routes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_run(args) -> int:
    torch.manual_seed(0)
    receipt = CachePass(args).run()
    log(f"verdict {receipt['verdict']}")
    return 0


def cmd_parity(args) -> int:
    cache = CacheReader(args.cache_dir)
    metrics = [parity_metrics(cache, ref) for ref in args.reference]
    verdict = parity_verdict(metrics)
    receipt = {"schema": SCHEMA + ".parity", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
               "tool_sha256": sha256_file(os.path.abspath(__file__)), "at_utc": utc(),
               "cache": cache.manifests, "files": metrics, **verdict}
    digest = write_json(args.out, receipt)
    for m in metrics:
        log(f"{m['file']}: positions {m['positions_compared']}/{m['positions_reference']} top1 {m['top1_agreement']:.4f} "
            f"top16 overlap {m['top16_overlap_mean']:.4f} |d target lp| median {m['target_logprob_abs_diff']['median']:.4f} "
            f"p99 {m['target_logprob_abs_diff']['p99']:.4f} max {m['target_logprob_abs_diff']['max']:.4f}")
    log(f"PARITY {verdict['verdict'].upper()} gates {verdict['gates']} -> {args.out} (sha256 {digest[:12]})")
    return 0 if verdict["verdict"] == "pass" else 3


def cmd_xtok(args) -> int:
    cache = CacheReader(args.cache_dir)
    qwen, k3, report = bind_tokenizers(args)
    mapping, stats = build_vocab_map(k3, qwen, int(args.k3_vocab), QWEN_TEXT_ROWS)
    with open(args.boundary_map, "r", encoding="utf-8") as handle:
        boundary = json.load(handle)
    got = xtok_rows(cache, boundary, mapping, load_ids_by_name(args.qwen_ids), load_ids_by_name(args.k3_ids))
    t = got["tensors"]
    os.makedirs(args.out, exist_ok=True)
    from safetensors.torch import save_file

    map_path = os.path.join(args.out, "k3_to_qwen_map.safetensors")
    save_file({"k3_to_qwen": mapping}, map_path, metadata={"tool": TOOL_NAME, "qwen_limit": str(QWEN_TEXT_ROWS)})
    path = os.path.join(args.out, "xtok_target.safetensors")
    save_file(t, path, metadata={"tool": TOOL_NAME, "tool_version": TOOL_VERSION,
                                 "qwen_files": json.dumps(got["qwen_files"]),
                                 "semantics": "row q_pos of the Qwen student <-> K3 position k3_pos (same bytes "
                                              "consumed); qwen_ids -1 = no identical-byte Qwen token in [0, 248320); "
                                              "tail bucket = 1 - matched_mass"})
    summary = {"pairs": int(t["q_pos"].numel()), "rows_used": got["rows_used"],
               "rows_not_in_cache": got["rows_not_in_cache"],
               "matched_mass_mean": float(t["matched_mass"].mean()), "tail_mass_mean": float(t["tail_mass"].mean()),
               "topk_mass_mean": float(t["topk_mass"].mean()),
               "top1_mapped_fraction": float((t["qwen_ids"][:, 0] >= 0).float().mean()),
               "next_token_identical_fraction": float(t["next_token_identical"].float().mean())
               if "next_token_identical" in t else None,
               "max_qwen_id": int(t["qwen_ids"].max())}
    receipt = {"schema": SCHEMA + ".xtok", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
               "tool_sha256": sha256_file(os.path.abspath(__file__)), "at_utc": utc(), "cache": cache.manifests,
               "boundary_map": {"path": args.boundary_map, "sha256": sha256_file(args.boundary_map)},
               "tokenizers": report, "vocab_map": {**stats, "path": map_path, "file_sha256": sha256_file(map_path)},
               "target": {"path": path, "sha256": sha256_file(path), **summary}, "qwen_files": got["qwen_files"]}
    digest = write_json(os.path.join(args.out, "xtok_receipt.json"), receipt)
    log(f"xtok: {summary} map {stats['k3_mapped']}/{stats['k3_vocab']} K3 ids -> receipt sha256 {digest[:12]}")
    return 0


def cmd_probe(args) -> int:
    cache = CacheReader(args.cache_dir)
    qwen, k3, report = bind_tokenizers(args)
    with open(args.boundary_map, "r", encoding="utf-8") as handle:
        boundary = json.load(handle)
    k3_len = token_byte_lengths(k3, int(args.k3_vocab))
    q_len = token_byte_lengths(qwen, int(qwen.vocab_size)) if args.q_target else None
    q_target = None
    if args.q_target:
        from safetensors.torch import load_file

        q_target = {os.path.splitext(os.path.basename(p))[0].replace("_target_logprob", ""): load_file(p)["target_logprob"].float()
                    for p in args.q_target}
    got = probe_rows(cache, boundary, load_ids_by_name(args.k3_ids), k3_len, load_routes(args.row_meta, args.route_key),
                     q_target, load_ids_by_name(args.qwen_ids) if args.qwen_ids else None, q_len)
    receipt = {"schema": SCHEMA + ".probe", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
               "tool_sha256": sha256_file(os.path.abspath(__file__)), "at_utc": utc(), "cache": cache.manifests,
               "boundary_map": {"path": args.boundary_map, "sha256": sha256_file(args.boundary_map)},
               "tokenizers": report, "q_target": args.q_target or None,
               "q_note": None if args.q_target else "no Q-text cache given: K3 side only",
               **got}
    digest = write_json(args.out, receipt)
    for route, agg in sorted(got["per_route"].items()):
        log(f"route {route}: rows {agg['rows']} bytes {agg['bytes']} K3 {agg['k3_bpb']:.4f} bpb"
            + (f" Q {agg['q_bpb']:.4f} bpb" if "q_bpb" in agg else ""))
    log(f"probe -> {args.out} (sha256 {digest[:12]})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="the teacher-forced streamed K3 pass")
    r.add_argument("--input-ids", action="append", required=True, help="K3 ids file(s), k3_corpus_build v2 layout")
    r.add_argument("--src-dir", default=DEFAULT_SRC_DIR)
    r.add_argument("--engine-dir", default=os.environ.get("K3_ENGINE_DIR", DEFAULT_ENGINE_DIR))
    r.add_argument("--allow-engine-sha", action="store_true")
    r.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="teacher_cache/k3; the pass dir is <corpus_sha>/pass-k-of-N")
    r.add_argument("--pass-dir", default=None)
    r.add_argument("--state-dir", default=None, help="layer checkpoints (default <pass-dir>/state)")
    r.add_argument("--pass-count", type=int, default=1)
    r.add_argument("--pass-index", type=int, default=0)
    r.add_argument("--max-pass-tokens", type=int, default=MAX_PASS_TOKENS)
    r.add_argument("--row-limit", type=int, default=0, help="first N rows of the pass (probes)")
    r.add_argument("--truncate-seq", type=int, default=0, help="PROBE: the first N tokens of every row")
    r.add_argument("--no-trim", action="store_true", help="keep each file's full T (default: trim to the longest real row)")
    r.add_argument("--group-tokens", type=int, default=32768, help="positions per card-resident row group")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--experts-device", default=None)
    r.add_argument("--resid-device", default="cpu")
    r.add_argument("--compute-dtype", default="auto", choices=("auto", "bf16", "fp32"))
    r.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    r.add_argument("--row-chunk-kda", type=int, default=1)
    r.add_argument("--row-chunk-mla", type=int, default=1)
    r.add_argument("--token-chunk", type=int, default=1024)
    r.add_argument("--head-row-chunk", type=int, default=1)
    r.add_argument("--checkpoint-every", type=int, default=1, help="layers between checkpoints; 0 = none")
    r.add_argument("--min-free-gib", type=float, default=60.0, help="a checkpoint that would leave less is skipped")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--hold-device", dest="hold_device", action="store_true", default=True,
                   help="keep the caching allocator's reservation between layers (default), so a shared card "
                        "never reads as free to another lane's nvidia-smi check mid-pass")
    r.add_argument("--release-device", dest="hold_device", action="store_false")
    r.add_argument("--keep-state", action="store_true")
    r.add_argument("--prefetch-next", dest="prefetch_next", action="store_true", default=True)
    r.add_argument("--no-prefetch-next", dest="prefetch_next", action="store_false")
    r.add_argument("--inline-digest", action="store_true", help="hash every shard read against the digest receipt (pass 1)")
    r.add_argument("--digest-receipt", default=DEFAULT_DIGEST_RECEIPT)
    r.add_argument("--fold-receipt", default=DEFAULT_FOLD_RECEIPT)
    r.add_argument("--allow-source-drift", action="store_true")
    r.add_argument("--prefix-digests", action="store_true", help="sha256 of the prefix after every layer (small runs)")
    r.add_argument("--probe", action="store_true", help="P3: a layer window, timing/memory only")
    r.add_argument("--layers", default=None)
    r.add_argument("--synthetic-blocks", type=int, default=0, help="PROBE: pre-fill N block slots (RSS with blocks resident)")
    r.add_argument("--no-head", action="store_true")
    r.add_argument("--start-snapshot", default=None,
                   help="PARITY: k3_dense_pass teacher snapshot before_L{l}.safetensors entering --layers' first layer")
    r.add_argument("--snapshot-layout", default=None,
                   help="PARITY: name=token_offset:row_length,... of every input file inside the snapshot's corpus")
    r.add_argument("--no-write", action="store_true")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("parity", help="cache top-k vs the fold engine's review dump")
    p.add_argument("--cache-dir", action="append", required=True)
    p.add_argument("--reference", action="append", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_parity)

    for name, func, helptext in (("xtok", cmd_xtok, "the cross-tokenizer target"),
                                 ("probe", cmd_probe, "per-route bits per byte")):
        x = sub.add_parser(name, help=helptext)
        x.add_argument("--cache-dir", action="append", required=True)
        x.add_argument("--boundary-map", required=True)
        x.add_argument("--k3-ids", action="append", default=[], help="the K3 ids files the cache was built from")
        x.add_argument("--qwen-ids", action="append", default=[], help="the paired Qwen ids files")
        x.add_argument("--qwen-dir", default="/home/cereal/models/Qwen3.8-27B-Uncensored")
        x.add_argument("--k3-dir", default=DEFAULT_SRC_DIR)
        x.add_argument("--k3-vocab", type=int, default=163840)
        x.add_argument("--qwen-separator-id", type=int, default=QWEN_SEPARATOR_ID)
        x.add_argument("--chimera-src", default=os.environ.get("CHIMERA_OMNI_SRC"))
        x.add_argument("--corpus-tools-dir", default=DEFAULT_CORPUS_TOOLS_DIR)
        if name == "xtok":
            x.add_argument("--out", required=True, help="output directory")
        else:
            x.add_argument("--out", required=True, help="receipt path")
            x.add_argument("--q-target", action="append", default=[],
                           help="<qwen file>_target_logprob.safetensors with target_logprob [rows, T] (Q-text cache)")
            x.add_argument("--row-meta", default=None)
            x.add_argument("--route-key", default="route")
        x.set_defaults(func=func)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
