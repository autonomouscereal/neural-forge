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
          2.0M-token budget runs as ``--pass-count 4``.  ``--max-pass-tokens`` counts REAL tokens (the
          budget is in real tokens); the padded positions a pass actually holds are bounded separately by
          ``--max-host-state-gb`` on the host state they cost.  The working set on the CARD is bounded by
          ``--group-tokens`` (whole rows; every row is independent of every other row), so the
          same command runs on a 24 GB card and on the RTX 8000.  Resumable at every layer
          boundary (``--checkpoint-every``, blocks written once, prefix per checkpoint) and at
          every output group.  The checkpoints live in ``<state-dir>/<pass_sha[:16]>/`` (default
          state-dir ``<pass-dir>/state``); after a seal only the files this pass wrote there are
          removed, never anything else under ``--state-dir``.  Integrity per the design's K3 row: per-shard size/mtime before and
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
  probe-distill
          ``probe`` over the stage-D K3 corpus (``distill_corpus_build``: one directory per pass, the
          aligned pairs as int32 arrays in ``boundary_map.safetensors`` with per-row offsets, rows packed
          with ~15 documents each closed by the separator on both sides, one route per document in
          ``row_meta.json``).  The unit is the DOCUMENT: from the boundary after the previous separator
          (the row's first pair for the first document) to the last aligned pair before its own separator,
          so no separator prediction is scored; bits per byte per route, K3 side, and Q38 when a Q-text
          ``target_logprob`` file is given for the pass.  Every input is checked against the sha256 the
          corpus's JSON names.
  parity  The cache's top-k vs the fold engine's own teacher-forced review dump
          (``fold/review/*_teacher_forced.safetensors``, top-16 per position), against the
          PRE-REGISTERED ``PARITY_THRESHOLDS`` below (fixed before the first parity run).
  export-reference
          A sealed cache written in that review-dump layout at EXACTLY the (row, position)s of a given
          fold dump, so ``parity`` -- its code and ``PARITY_THRESHOLDS`` unchanged -- can judge another
          cache against this one (the B2 re-baseline: a bf16 production-geometry cache against an fp32
          cache of the same geometry; ruling in $OC/distill/queues/k3_b2_rebaseline_ruling.txt).

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

1.3.0 -- bit-exact acceleration (lane k3accel, 2026-10-14; ``--accel on``, the default).  The 1.2.4 pass was
host-bound (k3_accel_profile 1.0.0 on the frozen 1.2.4: serial layer reads on the main thread, a Python-dispatched
KDA recurrence, a per-expert host sync in the routed MoE, pageable / non-contiguous copies, synchronous checkpoints).
Every change below keeps each floating-point op's inputs, kernel and order; only scheduling, copies and syncs move:
  * layer loader: a thread reads layer i+1's tensors -- the same byte ranges, in on-disk offset order, as
    ``k3_common.read_tensors`` -- into ONE reusable host arena (registered pinned memory, one copy instead of two)
    while the card computes layer i; the main thread then copies the arena to the card asynchronously and applies the
    engine's own ``resolve_compute_dtype`` casts.  Page-cache ranges it has consumed are dropped (fadvise DONTNEED)
    once the digest warmer has passed that shard.
  * KDA recurrence: ``k3_layer.naive_recurrent_kda`` (the engine's own function) is captured once per input shape in
    a CUDA graph and replayed per row.  The first row of EVERY KDA layer is also run eagerly and compared BITWISE
    (output and final state); any difference disables the graph for the rest of the pass and that row keeps the
    eager result (``--kda-graph-verify``).  This is the one runtime substitution inside the engine (a module
    attribute swapped for the pass and restored after; the engine file is never edited).
  * routed MoE: one stable sort per row group replaces 896 ``topk_idx == e`` scans and their per-expert ``nonzero``
    syncs (same rows, ascending; the weight is the single matching top-k weight that the engine's masked sum adds
    exact zeros to), and the MXFP4 dequant reads a 256-entry pair table built by the engine's own expression
    (``dequant_mxfp4`` re-staged its 8-value table on the card with a synchronising copy on every call); the power-of-
    two factor, the fp32 product and the cast are the engine's statements.  Each expert GEMM sees the same rows in
    the same order.  ``--moe-group-batch auto``: several row groups route first and then share ONE dequant per expert
    (the engine dequantized every expert again for every group); each group keeps its own rows, GEMMs and ascending-
    expert accumulation, so its y_lat is the one it gets alone.  The batch is sized from free card memory (one group
    on a 24 GB card at 65,536-token groups); an out-of-memory restarts the batch's accumulators from zero one group
    at a time.
  * attention: the prefix rows and the AttnRes block slice reach the card through pinned double buffers (same bytes,
    contiguous on the card as ``.to`` makes them); the host-side ``torch.cat`` of the group's whole block stack at a
    snapshot layer, which 1.2.4 discarded, is not built.  The new prefix is copied straight into the host state.
    The finite refusal is checked with ``isfinite().all()`` (the engine's ``require_finite`` sums the mask as int64:
    a 3.5 GiB transient at a 65,536-position group); a failure still raises through the engine's own check.
  * checkpoints: a streaming safetensors writer hashes the bytes as it writes them (no re-read) and runs beside the
    next layer; the main thread never overwrites a prefix row the writer has not written yet.  An exception waits
    for the checkpoint in flight; a SIGKILL inside the ~30 s write window resumes from the checkpoint before it.
``--accel off`` runs the 1.2.4 code path exactly.  NUMERICS_VERSION is unchanged, so checkpoints and head progress
carry over between the two.  Proof: ``k3_cache_parity`` (per-layer sha256 of the prefix and of every block slot, and
of every head output tensor, against the frozen 1.2.4 on identical inputs and geometry).

Usage (server).  P40s need the Pascal venv (the lab venv's torch 2.11+cu128 has no sm_6x):

  PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 nice -n 15 $PY k3_teacher_cache.py run \\
      --input-ids $OC/composite/corpus/k3_review_input_ids.safetensors \\
      --device cuda:0 --experts-device cuda:0 --kda-backend torch --group-tokens 32768
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import threading
import time
import uuid
from typing import Any

import torch

TOOL_NAME = "k3_teacher_cache"
# 1.2.0: digest / fold receipt refusals, resume never re-reads finished shards
# 1.2.1: --max-pass-tokens counts real tokens (+ --max-host-state-gb on the padded positions); checkpoints scoped
#        to <state-dir>/<pass_sha[:16]>/ and only this pass's own files removed after a seal; per-layer timings
#        name the positions they cover; two input files with one name are refused (the cache keys rows by name)
# 1.2.2: probe-distill -- the route probe over distill_corpus_build's K3 passes (pairs in boundary_map.safetensors,
#        one route per packed DOCUMENT in row_meta.json); the run path is unchanged from 1.2.1
# 1.2.3 (trainer-prep lane, E5): xtok / probe read a distill pass's boundary map (load_boundary_map: the pairs
#        come from load_distill_pass, every file sha-checked; 1.2.2's xtok raised KeyError 'file' on it); the K3
#        side of the join is proven (k3_join: the cache must have been built from the very K3 file the map names,
#        by sha -- every distill pass names its file k3_input_ids); CacheReader refuses a (file, row) seen twice
#        and one file name with two shas across pass dirs; the run path is unchanged
# 1.2.4 (lane gpu6, plan Step 2a): export-reference -- a sealed cache (the fp32 pass) in the fold review-dump layout
#        at the fold dump's own (row, position)s, for the B2 re-baseline against an fp32 reference at production
#        geometry (k3_b2_geometry_diag: bf16 batch-shape numerics in the routed MoE, not padding or state); parity,
#        PARITY_THRESHOLDS and the run path are unchanged from 1.2.1
# 1.3.0 (lane k3accel): --accel on (default) -- layer loader, CUDA-graph KDA recurrence verified bitwise per layer,
#        sorted-dispatch MoE with the exact pair-table dequant, pinned staging, streaming async checkpoints; every op's
#        inputs, kernel and order unchanged (see the docstring); --accel off is the 1.2.4 path; --stage-timers (probe)
TOOL_VERSION = "1.3.0"
# what a checkpoint / head-progress resume must match: bumped only when the numbers could change
NUMERICS_VERSION = "k3_teacher_cache.numerics.v1"
SCHEMA = "omni-campaign.k3_teacher_cache.v1"
TOPK = 32
# ids int32 + logprobs bf16 + lse fp32: the design's 196 B/token; the probe's target logprob adds 4
CORE_BYTES_PER_TOKEN = TOPK * 4 + TOPK * 2 + 4
PROBE_BYTES_PER_TOKEN = 4
QWEN_TEXT_ROWS = 248_320
QWEN_SEPARATOR_ID = 248_044
MAX_PASS_TOKENS = 500_000  # REAL tokens per pass (design: 4 passes of <= 0.5M)
MAX_HOST_STATE_GB = 80.0   # the padded positions' host state (design 71.7 GB at 0.5M); RAM rule 97-130 GB with RSS
STATE_FILE_PREFIXES = ("block_", "prefix_L")

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
# 1.3.0: bit-exact acceleration -- scheduling, copies and syncs only; every number is the engine's
# --------------------------------------------------------------------------
ARENA_ALIGN = 64
WRITE_CHUNK_BYTES = 64 << 20


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal BIT PATTERNS (``torch.equal`` calls -0.0 == +0.0 and NaN != NaN equal / unequal by value)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    as_int = {2: torch.int16, 4: torch.int32, 8: torch.int64}
    if a.is_floating_point() and a.element_size() in as_int:
        return bool(torch.equal(a.contiguous().view(as_int[a.element_size()]),
                                b.contiguous().view(as_int[b.element_size()])))
    return bool(torch.equal(a, b))


class StageClock:
    """``--stage-timers`` (probe only): wall seconds per named stage, the card synchronised at both ends of every
    stage so each wall is that stage's own.  Off (the default) a stage is a null context: nothing synchronises."""

    def __init__(self, enabled: bool, synchronize=None):
        self.enabled = bool(enabled)
        self.synchronize = synchronize or (lambda: None)
        self.acc: dict[str, list] = {}

    def stage(self, name: str):
        return self._stage(name) if self.enabled else contextlib.nullcontext()

    @contextlib.contextmanager
    def _stage(self, name: str):
        self.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            rec = self.acc.setdefault(name, [0, 0.0])
            rec[0] += 1
            rec[1] += time.perf_counter() - t0

    def take(self) -> dict[str, dict[str, float]]:
        out = {k: {"calls": c, "wall_s": round(s, 3)} for k, (c, s) in sorted(self.acc.items())}
        self.acc = {}
        return out


def e2m1_pair_table(common, device: torch.device) -> torch.Tensor:
    """[256, 2] fp32: the (low, high) nibble values of every packed byte, by ``k3_common.dequant_mxfp4``'s OWN
    expression (low nibble = even column; bit 3 = sign, so code 8 is -0.0), built on ``device``."""
    packed = torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8).reshape(256, 1)
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    table = common.E2M1.to(device)
    mag = table[(codes & 0x07).long()]
    return torch.where((codes & 0x08) != 0, -mag, mag).contiguous()


def dequant_mxfp4_exact(packed: torch.Tensor, scale: torch.Tensor, pair_table: torch.Tensor, group: int = 32,
                        dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``k3_common.dequant_mxfp4`` with the nibble decode read from ``pair_table``: the fp32 ``vals`` are the same bit
    patterns (gathered, not recomputed), and the factor, the fp32 product and the cast are its own statements -- so
    the output is bit-identical by construction (the unit test checks every byte x every scale)."""
    vals = pair_table.index_select(0, packed.reshape(-1).to(torch.int32)).reshape(packed.shape[0], -1)
    exponent = scale.to(torch.int32) - 127
    factor = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    vals = vals.reshape(packed.shape[0], -1, group) * factor.unsqueeze(-1)
    return vals.reshape(packed.shape[0], -1).to(dtype)


def register_pinned(tensor: torch.Tensor) -> bool:
    """Page-lock an existing host tensor (cudaHostRegister): its pages stay ordinary anonymous memory, so RssAnon --
    what the gpu6 queue's RAM ballast reads -- still counts them.  False (left pageable) if CUDA refuses."""
    if not torch.cuda.is_available() or tensor.numel() == 0:
        return False
    try:
        err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
    except Exception:  # noqa: BLE001 -- an older torch without the binding: stay pageable
        return False
    return int(getattr(err, "value", err)) == 0


def unregister_pinned(tensor: torch.Tensor) -> None:
    try:
        torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())
    except Exception:  # noqa: BLE001
        pass


class HostToDevice:
    """Host->card copies of row-sized slices through two reusable pinned buffers (a buffer is refilled only after its
    previous copy completed).  The card receives the same bytes as ``src.to(device)`` and, like it for these slices,
    a contiguous tensor; on the same device it is ``src.to(device)`` itself."""

    def __init__(self, device: torch.device, pinned: bool = True):
        self.device = device
        self.pinned = bool(pinned) and device.type == "cuda" and torch.cuda.is_available()
        self.bufs: list[torch.Tensor | None] = [None, None]
        self.events: list[Any] = [None, None]
        self.turn = 0
        self.bytes = 0

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        if not self.pinned or src.device.type != "cpu":
            return src.to(self.device)
        n = src.numel() * src.element_size()
        if n == 0:
            return torch.empty(src.shape, dtype=src.dtype, device=self.device)
        slot = self.turn
        self.turn ^= 1
        if self.events[slot] is not None:
            self.events[slot].synchronize()
        buf = self.bufs[slot]
        if buf is None or buf.numel() < n:
            self.bufs[slot] = None
            buf = self.bufs[slot] = torch.empty(n, dtype=torch.uint8).pin_memory()
        host = buf[:n].view(src.dtype).view(src.shape)
        host.copy_(src)
        out = torch.empty(src.shape, dtype=src.dtype, device=self.device)
        out.copy_(host, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.events[slot] = event
        self.bytes += n
        return out

    def close(self) -> None:
        for event in self.events:
            if event is not None:
                event.synchronize()
        self.bufs = [None, None]
        self.events = [None, None]


class KdaGraphRecurrence:
    """``k3_layer.naive_recurrent_kda`` replayed from a CUDA graph of ITSELF, one graph per input shape / strides /
    dtypes / scale (static inputs laid out like the call's own).  Verified bitwise against the eager function on the
    first row of every KDA layer (``verify`` = "layer"), or of the pass ("once"), or never ("off"); a difference
    disables the graph for the rest of the pass and the row keeps the eager result.  CPU tensors, a carried
    ``initial_state`` or a refused / failed capture run the eager function unchanged."""

    def __init__(self, eager, verify: str = "layer", min_free_bytes: int = 1 << 30):
        self.eager = eager
        self.verify = verify
        self.min_free_bytes = int(min_free_bytes)
        self.graphs: dict[tuple, dict[str, Any]] = {}
        self.layer: int | None = None
        self.verified: dict[int, bool] = {}
        self.disabled: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.counts = {"replays": 0, "eager": 0, "verify_runs": 0, "captures": 0, "releases": 0}

    def release(self) -> None:
        """Drop every graph and its private pool (a 24 GB card needs that room back for an MLA layer)."""
        if not self.graphs:
            return
        device = next(iter(self.graphs.values()))["out"].device
        self.graphs.clear()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        self.counts["releases"] += 1

    def __call__(self, q, k, v, g, beta, scale, initial_state=None):
        if self.disabled is not None or initial_state is not None or not q.is_cuda \
                or torch.cuda.is_current_stream_capturing():
            self.counts["eager"] += 1
            return self.eager(q, k, v, g, beta, scale, initial_state=initial_state)
        inputs = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        key = (str(q.device), float(scale),
               tuple((n, tuple(t.shape), tuple(t.stride()), str(t.dtype)) for n, t in inputs.items()))
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(key, inputs, scale)
            if entry is None:
                self.counts["eager"] += 1
                return self.eager(q, k, v, g, beta, scale)
        for name, tensor in inputs.items():
            entry["static"][name].copy_(tensor)
        entry["graph"].replay()
        self.counts["replays"] += 1
        out, state = entry["out"].clone(), entry["state"].clone()
        due = (self.verify == "layer" and self.layer not in self.verified) or \
            (self.verify == "once" and not self.verified)
        if due:
            e_out, e_state = self.eager(q, k, v, g, beta, scale)
            self.counts["verify_runs"] += 1
            same = bits_equal(out, e_out) and bits_equal(state, e_state)
            self.verified[self.layer] = same
            if not same:
                diff = float((out.float() - e_out.float()).abs().max())
                self.disabled = {"layer": self.layer, "max_abs_diff_out": diff, "at_utc": utc(),
                                 "action": "graph disabled for the rest of the pass; this row keeps the eager result"}
                log(f"WARNING KDA graph replay != eager at L{self.layer} (max |diff| {diff:.3e}): graph disabled")
                return e_out, e_state
        return out, state

    def _capture(self, key, inputs: dict[str, torch.Tensor], scale: float) -> dict[str, Any] | None:
        device = inputs["q"].device
        static = {n: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=device) for n, t in inputs.items()}
        for n, t in inputs.items():
            static[n].copy_(t)
        # the private pool holds q * scale, out, the state and per-step scratch; the eager check beside it needs as much
        # again.  The general pool's cached blocks go back to the driver first, so the free figure is the real one.
        q, v = inputs["q"], inputs["v"]
        state = q.shape[0] * q.shape[2] * q.shape[3] * v.shape[3] * 4
        need = 2 * (q.numel() * 4 + v.numel() * 4 + 16 * state) + self.min_free_bytes
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
        if free < need:
            self.events.append({"layer": self.layer, "action": "capture_refused_low_memory", "free_bytes": free,
                                "need_bytes": need, "shape_q": list(inputs["q"].shape)})
            return None
        started = time.time()
        try:
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):  # warm-up outside the capture (allocator, library handles on this stream)
                self.eager(static["q"], static["k"], static["v"], static["g"], static["beta"], scale)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, state = self.eager(static["q"], static["k"], static["v"], static["g"], static["beta"], scale)
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 -- never fatal: the pass continues eagerly
            self.disabled = {"layer": self.layer, "action": "capture_failed", "error": f"{type(exc).__name__}: {exc}"[:400]}
            log(f"WARNING KDA graph capture failed ({self.disabled['error']}): eager for the rest of the pass")
            torch.cuda.synchronize(device)
            return None
        entry = {"graph": graph, "static": static, "out": out, "state": state}
        self.graphs[key] = entry
        self.counts["captures"] += 1
        self.events.append({"layer": self.layer, "action": "captured", "shape_q": list(inputs["q"].shape),
                            "seconds": round(time.time() - started, 2)})
        return entry

    def record(self) -> dict[str, Any]:
        return {"verify": self.verify, "graphs": len(self.graphs), "counts": dict(self.counts),
                "verified_layers": {str(k): v for k, v in sorted(self.verified.items(), key=lambda kv: str(kv[0]))},
                "all_verified_equal": all(self.verified.values()) if self.verified else None,
                "disabled": self.disabled, "events": list(self.events)}

    def close(self) -> None:
        self.graphs.clear()


class LayerLoader(threading.Thread):
    """Reads one layer's tensors -- exactly ``k3_dense_pass.read_layer``'s set: per shard of
    ``index.shards_for(layer_keys)``, the layer's keys present in that shard, in on-disk offset order -- into ONE
    reusable host arena (64-byte aligned slots), while the main thread computes the previous layer.  ``take`` hands
    back views into the arena in the engine's dict order; the caller copies them out and ``release``-s the arena
    before the next layer is read into it."""

    def __init__(self, index, common, path_for, layers: list[int], *, pinned: bool, fadvise_drop: bool,
                 warmer: ShardWarmer | None, plan_shards: dict[int, list[str]]):
        super().__init__(name="k3-layer-loader", daemon=True)
        self.index, self.common, self.path_for = index, common, path_for
        self.fadvise_drop = bool(fadvise_drop) and hasattr(os, "posix_fadvise")
        self.warmer, self.plan_shards = warmer, plan_shards
        self.headers: dict[str, tuple[dict, int]] = {}
        self.layouts = {i: self.layout(i) for i in layers}
        self.arena_bytes = max((lay["bytes"] for lay in self.layouts.values()), default=0)
        self.arena = torch.empty(max(self.arena_bytes, 1), dtype=torch.uint8)
        self.arena_np = self.arena.numpy()
        self.pinned = register_pinned(self.arena) if pinned else False
        self.cv = threading.Condition()
        self.pending: int | None = None
        self.ready: tuple[int, dict[str, torch.Tensor], dict[str, Any]] | None = None
        self.busy = False  # the arena holds a layer the main thread has not released
        self.stop_flag = False
        self.error: BaseException | None = None
        self.read_log: list[str] = []
        self.dropped_bytes = 0

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self.headers:
            self.headers[shard] = self.common.read_header(self.path_for(shard))
        return self.headers[shard]

    def layout(self, layer_idx: int) -> dict[str, Any]:
        keys = self.index.layer_keys(layer_idx)
        if not keys:
            raise SystemExit(f"layer {layer_idx}: the index carries no tensors under "
                             f"{self.index.layer_prefix_for(layer_idx)}")
        shards, cursor = [], 0
        for shard in self.index.shards_for(keys):
            wanted = [k for k in keys if self.index.weight_map[k] == shard]
            header, base = self._header(shard)
            items = sorted(((k, header[k]) for k in wanted if k in header and k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            recs = []
            for key, meta in items:
                start, stop = meta["data_offsets"]
                cursor = -(-cursor // ARENA_ALIGN) * ARENA_ALIGN
                recs.append({"key": key, "dtype": meta["dtype"], "shape": list(meta["shape"]),
                             "file_offset": base + start, "bytes": stop - start, "arena_offset": cursor})
                cursor += stop - start
            shards.append({"shard": shard, "items": recs})
        missing = [k for k in keys if k not in {r["key"] for s in shards for r in s["items"]}]
        if missing:
            raise SystemExit(f"index promised {len(missing)} tensors the shards do not carry, e.g. {missing[:3]}")
        return {"layer": layer_idx, "shards": shards, "bytes": cursor}

    # -- main thread ------------------------------------------------------------------------------------------
    def request(self, layer_idx: int) -> None:
        with self.cv:
            self.pending = layer_idx
            self.cv.notify_all()

    def take(self, layer_idx: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], float]:
        started = time.time()
        with self.cv:
            if self.pending is None and self.ready is None and not self.busy:
                self.pending = layer_idx  # not requested ahead: read it now
                self.cv.notify_all()
            while self.error is None and (self.ready is None or self.ready[0] != layer_idx):
                if self.ready is not None:
                    raise SystemExit(f"layer loader holds L{self.ready[0]:02d}, L{layer_idx:02d} was asked for")
                self.cv.wait(timeout=1.0)
            if self.error is not None:
                raise SystemExit(f"layer loader failed at L{layer_idx:02d}: {self.error!r}")
            _, tensors, timing = self.ready
            self.ready = None
        return tensors, timing, time.time() - started

    def release(self) -> None:
        with self.cv:
            self.busy = False
            self.cv.notify_all()

    def close(self) -> None:
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()
        self.join(timeout=60)
        if self.pinned:
            unregister_pinned(self.arena)
            self.pinned = False

    # -- the thread ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        try:
            while True:
                with self.cv:
                    while not self.stop_flag and (self.pending is None or self.busy):
                        self.cv.wait(timeout=1.0)
                    if self.stop_flag:
                        return
                    layer_idx, self.pending = self.pending, None
                    self.busy = True
                tensors, timing = self.read(layer_idx)
                with self.cv:
                    self.ready = (layer_idx, tensors, timing)
                    self.cv.notify_all()
        except BaseException as exc:  # surfaced by take(), never swallowed
            with self.cv:
                self.error = exc
                self.cv.notify_all()

    def read(self, layer_idx: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        lay = self.layouts.get(layer_idx) or self.layout(layer_idx)
        if self.warmer is not None:
            self.warmer.advance(self.plan_shards.get(layer_idx, []))
        dtypes = self.common.DTYPES
        tensors: dict[str, torch.Tensor] = {}
        per_shard, total = [], 0
        started = time.time()
        for rec in lay["shards"]:
            shard, path = rec["shard"], self.path_for(rec["shard"])
            self.read_log.append(shard)
            shard_start = time.time()
            got = 0
            with open(path, "rb", buffering=0) as handle:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                for item in rec["items"]:
                    a, n = item["arena_offset"], item["bytes"]
                    view = memoryview(self.arena_np)[a:a + n]
                    handle.seek(item["file_offset"])
                    filled = 0
                    while filled < n:
                        r = handle.readinto(view[filled:])
                        if not r:
                            raise SystemExit(f"{path}: short read at {item['key']} ({filled} of {n} bytes)")
                        filled += r
                    if self.stop_flag:
                        return {}, {}
                    tensors[item["key"]] = self.arena[a:a + n].view(dtypes[item["dtype"]]).reshape(item["shape"])
                    got += n
                # drop what this read consumed from the page cache, once the digest warmer (if any) has hashed it
                if self.fadvise_drop and (self.warmer is None or shard in self.warmer.results):
                    for item in rec["items"]:
                        os.posix_fadvise(handle.fileno(), item["file_offset"], item["bytes"],
                                         os.POSIX_FADV_DONTNEED)
                        self.dropped_bytes += item["bytes"]
            elapsed = max(time.time() - shard_start, 1e-9)
            total += got
            per_shard.append({"shard": shard, "path": path, "tensors": len(rec["items"]),
                              "gigabytes": round(got / 1e9, 3), "seconds": round(elapsed, 1),
                              "MB_per_s": round(got / 1e6 / elapsed, 1), "staged": False})
        wall = max(time.time() - started, 1e-9)
        timing = {"tensors": len(tensors), "bytes": total, "gigabytes": round(total / 1e9, 3), "seconds": round(wall, 1),
                  "MB_per_s": round(total / 1e6 / wall, 1), "shards": per_shard, "reader": "layer_loader",
                  "arena_pinned": self.pinned}
        return tensors, timing


def write_safetensors_streaming(path: str, name: str, tensor: torch.Tensor, metadata: dict[str, str],
                                dtype_names: dict[torch.dtype, str], on_rows=None,
                                chunk_bytes: int = WRITE_CHUNK_BYTES) -> str:
    """One 2-D host tensor as a safetensors file (8-byte LE header length, JSON header padded with spaces to 8 bytes,
    raw little-endian data), written in row chunks and hashed AS IT IS WRITTEN; atomic (``.tmp<pid>`` + rename, the
    ``save_tensors`` naming).  ``on_rows(r)`` is called after rows [0, r) are on disk.  Returns the file sha256."""
    rows = int(tensor.shape[0])
    row_bytes = int(tensor[0].numel() * tensor.element_size()) if rows else 0
    total = rows * row_bytes
    header = {"__metadata__": dict(sorted(metadata.items())),
              name: {"dtype": dtype_names[tensor.dtype], "shape": [int(s) for s in tensor.shape],
                     "data_offsets": [0, total]}}
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body += b" " * ((8 - len(body) % 8) % 8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    digest = hashlib.sha256()
    step = max(1, chunk_bytes // max(row_bytes, 1))
    with open(tmp, "wb") as handle:
        head = struct.pack("<Q", len(body)) + body
        handle.write(head)
        digest.update(head)
        for r0 in range(0, rows, step):
            r1 = min(r0 + step, rows)
            piece = tensor[r0:r1].contiguous().view(torch.uint8).reshape(-1)
            data = memoryview(piece.numpy())
            handle.write(data)
            digest.update(data)
            del piece, data
            if on_rows is not None:
                on_rows(r1)
    os.replace(tmp, path)
    return digest.hexdigest()


class CheckpointJob(threading.Thread):
    """One layer checkpoint written beside the next layer: the new block slots (never written again once pushed),
    then the prefix in row order -- ``rows_done`` tells the main thread which prefix rows it may overwrite -- then
    ``state.json`` and the removal of the previous prefix file, in 1.2.4's order."""

    def __init__(self, owner: "CachePass", layer_done: int, prefix: torch.Tensor, blocks: torch.Tensor, nb: int,
                 new_blocks: list[int], need: int):
        super().__init__(name=f"k3-checkpoint-L{layer_done:02d}", daemon=True)
        self.owner, self.layer_done = owner, layer_done
        self.prefix, self.blocks, self.nb, self.new_blocks, self.need = prefix, blocks, nb, new_blocks, need
        self.cv = threading.Condition()
        self.rows_done = 0
        self.finished = False
        self.error: BaseException | None = None
        self.result: dict[str, Any] | None = None

    def _rows(self, r: int) -> None:
        with self.cv:
            self.rows_done = r
            self.cv.notify_all()

    def wait_rows(self, stop: int) -> float:
        started = time.time()
        with self.cv:
            while not self.finished and self.error is None and self.rows_done < stop:
                self.cv.wait(timeout=1.0)
        return time.time() - started

    def run(self) -> None:
        own = self.owner
        started = time.time()
        try:
            names = {v: k for k, v in own.common.DTYPES.items()}
            saved_blocks = list(own.saved_blocks)
            for b in self.new_blocks:
                path = os.path.join(own.state_dir, f"block_{b}.safetensors")
                digest = write_safetensors_streaming(path, "block", self.blocks[:, b], {
                    "tool": TOOL_NAME, "pass_sha256": own.pass_sha, "block": str(b)}, names)
                saved_blocks.append({"path": path, "sha256": digest})
            path = os.path.join(own.state_dir, f"prefix_L{self.layer_done:02d}.safetensors")
            digest = write_safetensors_streaming(path, "prefix", self.prefix, {
                "tool": TOOL_NAME, "pass_sha256": own.pass_sha, "layer_done": str(self.layer_done)}, names,
                on_rows=self._rows)
            old = own.saved_prefix.get("path") if own.saved_prefix else None
            saved_prefix = {"path": path, "sha256": digest, "layer_done": self.layer_done}
            write_json(own.state_manifest_path(), {
                "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "pass_sha256": own.pass_sha,
                "runtime_key": canonical_json(own.runtime_key), "layer_done": self.layer_done, "n_blocks": self.nb,
                "blocks": saved_blocks, "prefix": saved_prefix,
                "digests": dict(own.warmer.results) if own.warmer else None, "written_utc": utc()})
            if old and os.path.abspath(old) != os.path.abspath(path) and os.path.exists(old):
                os.remove(old)
            self.result = {"saved_blocks": saved_blocks, "saved_prefix": saved_prefix,
                           "record": {"layer_done": self.layer_done, "blocks_written": self.new_blocks,
                                      "bytes": self.need, "seconds": round(time.time() - started, 1),
                                      "writer": "streaming_async", "sha256_inline": True}}
        except BaseException as exc:  # surfaced by the next join, never swallowed
            self.error = exc
        finally:
            with self.cv:
                self.finished = True
                self.cv.notify_all()


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
        names = [f["name"] for f in self.files]
        if len(set(names)) != len(names):
            # the cache (CacheReader, xtok, probe, parity) keys every row by (file name, row index)
            raise SystemExit(f"two --input-ids files share a name ({sorted(n for n in set(names) if names.count(n) > 1)}): "
                             "their rows would collide in the cache; run one pass per file or rename one")
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
        # the budget is in REAL tokens; the padded positions (rows x trimmed T) are what the host state costs
        self.real_positions = int(sum(int(s.real_lengths.sum()) for s in self.corpus.segments))
        if self.real_positions > args.max_pass_tokens:
            raise SystemExit(f"pass {args.pass_index}/{args.pass_count} holds {self.real_positions:,} real tokens > "
                             f"--max-pass-tokens {args.max_pass_tokens:,} (design rule: <= 0.5M per pass); "
                             "raise --pass-count")
        state_gb = self.state_bytes(self.corpus.tokens) / 1e9
        if state_gb > args.max_host_state_gb:
            raise SystemExit(f"pass {args.pass_index}/{args.pass_count} holds {self.corpus.tokens:,} positions "
                             f"({self.real_positions:,} real): its host state would be {state_gb:.1f} GB > "
                             f"--max-host-state-gb {args.max_host_state_gb:.1f} (design ~71.7 GB at 0.5M); "
                             "raise --pass-count")
        self.pass_sha = sha256_text(canonical_json({
            "corpus": self.corpus_sha, "pass_index": args.pass_index, "pass_count": args.pass_count,
            "row_limit": args.row_limit, "trim": not args.no_trim, "truncate_seq": args.truncate_seq, "segments": [s.record() for s in self.corpus.segments]}))
        self.run_id = uuid.uuid4().hex[:12]
        # a probe never lands in the cache's own pass dir
        leaf = f"probe-{self.run_id}" if args.probe else f"pass-{args.pass_index}-of-{args.pass_count}"
        self.pass_dir = os.path.abspath(args.pass_dir or os.path.join(args.out_root, self.corpus_sha, leaf))
        # checkpoints of THIS pass only: <state-dir>/<pass_sha[:16]>/ (a shared --state-dir never mixes passes and
        # the seal never removes anything above this directory)
        self.state_base = os.path.abspath(args.state_dir or os.path.join(self.pass_dir, "state"))
        self.state_base_is_default = not args.state_dir
        self.state_dir = os.path.join(self.state_base, self.pass_sha[:16])
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
                       "real_positions": self.real_positions,
                       "fill": round(self.real_positions / max(self.corpus.tokens, 1), 4),
                       "positions_note": "positions = rows x trimmed T, what every layer computes and the host state "
                                         "holds; real_positions = real tokens, what the cache stores and what "
                                         "--max-pass-tokens counts",
                       "groups": len(self.corpus.groups), "group_tokens": args.group_tokens},
            "state": {"state_dir": self.state_dir, "state_base": self.state_base,
                      "scoped_by": "pass_sha256[:16]", "base_is_default": self.state_base_is_default},
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
        if args.stage_timers and not args.probe:
            raise SystemExit("--stage-timers synchronises the card at every stage: a PROBE option (--probe)")
        self.accel = args.accel == "on"
        self.receipt["accel"] = {
            "enabled": self.accel,
            "layer_loader": self.accel and args.layer_loader, "layer_buffer": args.layer_buffer,
            "kda_graph": self.accel and args.kda_graph and args.kda_backend == "torch",
            "kda_graph_verify": args.kda_graph_verify, "fadvise_drop": self.accel and args.fadvise_drop,
            "pinned_staging": self.accel and args.pinned_staging, "async_checkpoint": self.accel and args.async_checkpoint,
            "moe": "sorted dispatch + exact pair-table dequant" if self.accel else "engine routed_y_lat",
            "stage_timers": bool(args.stage_timers),
            "numerics": "unchanged: every floating-point op has the engine's inputs, kernel and order "
                        "(NUMERICS_VERSION unchanged); --accel off runs the 1.2.4 code path"}
        self.clock = StageClock(bool(args.stage_timers),
                                (lambda: torch.cuda.synchronize(self.device)) if self.cuda else None)

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
        self.join_checkpoint()  # 1.3.0: one checkpoint in flight; its files are in saved_blocks before the next
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
        if self.accel and self.args.async_checkpoint:
            # streaming + hashed in line, beside the next layer; the prefix rows are guarded by wait_checkpoint_rows
            self.ckpt_job = CheckpointJob(self, layer_done, prefix, blocks, nb, new_blocks, need)
            self.ckpt_job.start()
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

    def join_checkpoint(self) -> None:
        """Wait for the checkpoint in flight (1.3.0 async writer) and take its files into the pass's record."""
        job = getattr(self, "ckpt_job", None)
        if job is None:
            return
        started = time.time()
        job.join()
        self.ckpt_job = None
        if job.error is not None:
            raise SystemExit(f"checkpoint after L{job.layer_done:02d} failed: {job.error!r}")
        self.saved_blocks = job.result["saved_blocks"]
        self.saved_prefix = job.result["saved_prefix"]
        self.receipt["checkpoints"].append({**job.result["record"], "join_wait_seconds": round(time.time() - started, 1)})

    def wait_checkpoint_rows(self, stop: int) -> float:
        """Before prefix rows [.., stop) are overwritten: the checkpoint in flight must have written them."""
        job = getattr(self, "ckpt_job", None)
        return job.wait_rows(stop) if job is not None else 0.0

    def remove_own_state(self) -> dict[str, Any]:
        """After a seal: remove this pass's OWN checkpoint files (state.json, block_*/prefix_L*.safetensors and an
        interrupted state.json write) from ``<state-base>/<pass_sha[:16]>/``, then that directory if it is empty,
        then the default ``<pass-dir>/state`` base if it is empty.  Never a tree removal: anything else found there
        stays and is reported, and nothing outside the scoped directory is ever touched."""
        scope = os.path.realpath(self.state_dir)
        out: dict[str, Any] = {"state_dir": self.state_dir, "removed": [], "left": [], "dir_removed": False,
                               "base_removed": False, "at_utc": utc()}
        if os.path.basename(scope) != self.pass_sha[:16] or \
                os.path.dirname(scope) != os.path.realpath(self.state_base):
            out["refused"] = f"{self.state_dir} resolves to {scope}, outside <state-base>/<pass_sha[:16]>; nothing removed"
            log(f"WARNING state not removed: {out['refused']}")
            return out
        for entry in sorted(os.listdir(scope)):
            path = os.path.join(scope, entry)
            ours = entry == "state.json" or entry.startswith("state.json.tmp") or \
                (entry.startswith(STATE_FILE_PREFIXES) and entry.endswith(".safetensors"))
            if ours and os.path.isfile(path) and not os.path.islink(path):
                os.remove(path)
                out["removed"].append(entry)
            else:
                out["left"].append(entry)
        try:
            os.rmdir(scope)
            out["dir_removed"] = True
        except OSError:
            pass
        if self.state_base_is_default and out["dir_removed"]:
            try:
                os.rmdir(self.state_base)
                out["base_removed"] = True
            except OSError:
                pass
        log(f"state: removed {len(out['removed'])} file(s) of this pass from {scope}"
            + (f"; left {out['left']}" if out["left"] else ""))
        return out

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
        self.ckpt_job = None
        self.loader = None
        self.kda_graph = None
        self.h2d_prefix = self.h2d_blocks = None
        self.pair_tables: dict[str, torch.Tensor] = {}
        self.moe_batch_cap: int | None = None
        first = state[3] + 1 if state is not None else self.layers[0]
        restore = None
        try:
            if self.accel:
                restore = self.start_accel([i for i in self.layers if i >= first], plan_shards)
            self.run_layers_and_head(run, state, plan_shards, model_level)
            self.join_checkpoint()
        except BaseException:
            try:
                self.join_checkpoint()  # a checkpoint in flight completes before the exception leaves
            except BaseException as exc:  # noqa: BLE001 -- the original exception is the one to raise
                log(f"WARNING checkpoint in flight failed during an abort: {exc!r}")
            if self.warmer is not None:
                self.warmer.close()
            raise
        finally:
            self.stop_accel(restore)
        return self.finish(digests)

    # -- 1.3.0 acceleration: set up / tear down ----------------------------------------------
    def start_accel(self, layers: list[int], plan_shards: dict[int, list[str]]):
        """Loader thread (reads the first remaining layer at once), pinned staging, and the graphed KDA recurrence
        swapped into k3_layer for this pass.  Returns what ``stop_accel`` restores."""
        args = self.args
        if args.layer_loader and layers:
            self.loader = LayerLoader(self.index, self.common, self.stager.path_for, layers,
                                      pinned=args.layer_buffer == "pinned" and self.cuda,
                                      fadvise_drop=args.fadvise_drop, warmer=self.warmer, plan_shards=plan_shards)
            self.loader.start()
            self.loader.request(layers[0])
            self.receipt["accel"]["arena"] = {"bytes": self.loader.arena_bytes, "pinned": self.loader.pinned}
            log(f"layer loader: arena {self.loader.arena_bytes / 1e9:.2f} GB "
                f"({'pinned' if self.loader.pinned else 'pageable'}), first L{layers[0]:02d}")
        staging = args.pinned_staging and self.cuda
        self.h2d_prefix = HostToDevice(self.device, pinned=staging)
        self.h2d_blocks = HostToDevice(self.device, pinned=staging)
        restore = None
        if args.kda_graph and self.cuda and args.kda_backend == "torch":
            k3_layer = self.engine.layer
            eager = k3_layer.naive_recurrent_kda
            self.kda_graph = KdaGraphRecurrence(eager, verify=args.kda_graph_verify)
            k3_layer.naive_recurrent_kda = self.kda_graph
            restore = (k3_layer, eager)
            total = torch.cuda.get_device_properties(self.device).total_memory
            self.kda_graph_keep = args.kda_graph_keep == "always" or (args.kda_graph_keep == "auto" and total >= 32 << 30)
            self.receipt["accel"]["kda_graph_keep"] = {"policy": args.kda_graph_keep, "device_total_bytes": total,
                                                       "kept_across_non_kda_layers": self.kda_graph_keep}
        return restore

    def stop_accel(self, restore) -> None:
        if restore is not None:
            module, eager = restore
            module.naive_recurrent_kda = eager
        if self.kda_graph is not None:
            self.receipt["accel"]["kda_graph_record"] = self.kda_graph.record()
            self.kda_graph.close()
        if self.loader is not None:
            self.receipt["accel"]["loader_record"] = {"read": list(self.loader.read_log),
                                                      "fadvise_dropped_bytes": self.loader.dropped_bytes}
            self.loader.close()
        for h2d in (self.h2d_prefix, self.h2d_blocks):
            if h2d is not None:
                h2d.close()
        if self.h2d_prefix is not None:
            self.receipt["accel"]["staged_bytes"] = self.h2d_prefix.bytes + self.h2d_blocks.bytes

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
        if self.accel:
            return self.layer_accel(run, layer_idx, prefix, blocks, nb, plan_shards)
        return self.layer_reference(run, layer_idx, prefix, blocks, nb, plan_shards)

    # -- 1.3.0: the accelerated layer (same numbers as layer_reference) ------------------------
    def read_layer_accel(self, layer_idx: int, plan_shards: dict[int, list[str]]):
        """``k3_dense_pass.read_layer``'s result from the loader's arena: ``LayerWeights`` over the engine's tensor set
        in the engine's order, copied to the card (async from pinned memory, one sync), then the engine's own
        compute-dtype resolution and casts.  The next layer's read starts as soon as the arena is free."""
        kdp = self.kdp
        tensors, timing, waited = self.loader.take(layer_idx)
        t0 = time.time()
        weights = self.engine.layer.LayerWeights(layer_idx, tensors, self.index)

        def to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
            if device.type == "cpu":
                return tensor.clone()  # never alias the arena: it is refilled with the next layer
            out = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
            out.copy_(tensor, non_blocking=self.loader.pinned)
            return out

        for group in (weights.attn, weights.res, weights.body):
            for key, tensor in group.items():
                group[key] = to(tensor, self.device)
        for expert in weights.experts.values():
            for key, tensor in expert.items():
                expert[key] = to(tensor, self.experts_device)
        for dev in {self.device, self.experts_device}:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        del tensors
        self.loader.release()
        nxt = [i for i in self.layers if i > layer_idx]
        if nxt:
            self.loader.request(nxt[0])
        h2d = time.time() - t0
        resolved, dtype_source = kdp.resolve_compute_dtype(weights, self.compute_dtype)
        for group in (weights.attn, weights.res, weights.body):
            for key, tensor in group.items():
                if tensor.is_floating_point() and tensor.dtype != resolved:
                    group[key] = tensor.to(resolved)
        timing = dict(timing)
        timing.update({"compute_dtype": str(resolved), "compute_dtype_source": dtype_source,
                       "loader_wait_seconds": round(waited, 2), "h2d_seconds": round(h2d, 2)})
        return weights, timing

    def layer_accel(self, run, layer_idx: int, prefix: torch.Tensor, blocks: torch.Tensor, nb: int,
                    plan_shards: dict[int, list[str]]) -> int:
        args, kdp, common = self.args, self.kdp, self.common
        clock = self.clock
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        rss_before = vmrss_bytes()
        t0 = time.time()
        with clock.stage("read"):
            if self.loader is not None:
                weights, timing = self.read_layer_accel(layer_idx, plan_shards)
            else:
                if self.warmer is not None:
                    self.warmer.advance(plan_shards[layer_idx])
                weights, raw_body, timing = kdp.read_layer(self.index, layer_idx, self.stager, self.device,
                                                           self.experts_device, self.compute_dtype,
                                                           progress_tag=f"L{layer_idx:02d}" if args.verbose else None)
                del raw_body
        if self.cuda:
            torch.cuda.synchronize(self.device)
        read_wall = time.time() - t0
        runner = self.engine.layer.StreamedLayer(weights, self.cfg, kda_backend=args.kda_backend)
        if self.kda_graph is not None:
            if not weights.is_kda and not self.kda_graph_keep:
                self.kda_graph.release()  # a 24 GB card: the MLA layer gets the graph's pool back
            self.kda_graph.layer = layer_idx
            graph_before = dict(self.kda_graph.counts)
        snapshot = (layer_idx % self.block_size) == 0
        if snapshot and nb >= blocks.shape[1]:
            raise SystemExit(f"L{layer_idx}: block slot {nb} beyond the {blocks.shape[1]} reserved")
        row_chunk = args.row_chunk_kda if weights.is_kda else args.row_chunk_mla
        groups = self.corpus.groups
        w = weights
        m, batch_info = self.moe_group_batch(weights, max(s.T for s in self.corpus.segments),
                                             max(g1 - g0 for _, _, _, g0, g1 in groups), len(groups))
        batches: list[int] = []
        dequant = 0.0
        ckpt_wait = 0.0
        pending: list[dict[str, Any]] = []

        def finish(g0: int, g1: int, new_prefix: torch.Tensor) -> None:
            nonlocal ckpt_wait
            with clock.stage("finite_push_d2h"):
                # the same refusal as the engine's require_finite, without its int64 copy of the whole finite mask
                # (3.5 GiB at a 65,536-position group); on a failure the engine's own check names the count
                if not bool(torch.isfinite(new_prefix).all()):
                    kdp.require_finite("teacher prefix after the layer", new_prefix, layer_idx)
                if snapshot:
                    blocks[g0:g1, nb].copy_(prefix[g0:g1])  # the prefix ENTERING this layer is the pushed block
                ckpt_wait += self.wait_checkpoint_rows(g1)
                if new_prefix.dtype == prefix.dtype:
                    prefix[g0:g1].copy_(new_prefix)  # card -> host state directly: the same bytes as .to(cpu)
                else:
                    prefix[g0:g1].copy_(new_prefix.to(self.resid_device))

        def flush() -> None:
            """The pending groups' routed MoE with one dequant per expert, then each group's moe_write + residual
            and its copy-back, in group order."""
            nonlocal dequant
            plans = [p["plan"] for p in pending]
            try:
                dequant += self.moe_experts_accel(w, plans, self.cfg, self.device)
            except torch.cuda.OutOfMemoryError:
                if len(plans) == 1:
                    raise
                # every op so far was pure: restart each accumulator from zero, one group at a time (m = 1)
                self.moe_batch_cap = 1
                self.receipt["events"].append({"layer": layer_idx, "action": "moe_group_batch_oom_fallback",
                                               "groups": len(plans), "at_utc": utc()})
                log(f"WARNING L{layer_idx:02d}: MoE over {len(plans)} groups ran out of card memory; one group at "
                    "a time from here (same numbers)")
                torch.cuda.empty_cache()
                for plan in plans:
                    plan["y_lat"].zero_()
                    dequant += self.moe_experts_accel(w, [plan], self.cfg, self.device)
            batches.append(len(plans))
            plans.clear()
            while pending:
                p = pending.pop(0)
                with clock.stage("ffn.moe_write"):
                    out = kdp.moe_write(w, p["ffn_in"], p["plan"]["y_lat"], self.cfg, args.token_chunk)
                    new_prefix = p["carried"] + out
                g0, g1 = p["g0"], p["g1"]
                p.clear()  # this group's carried / FFN input / latent / accumulator go before its copy-back
                del out, p
                finish(g0, g1, new_prefix)
                del new_prefix

        compute_start = time.time()
        for gi, (si, r0, r1, g0, g1) in enumerate(groups):
            seg = self.corpus.segments[si]
            pair = kdp.Pair(prefix[g0:g1], blocks[g0:g1, :nb])
            view = GroupView(seg, r0, r1)
            try:
                carried, ffn_in = self.attention_half_accel(runner, pair, view, self.device,
                                                            row_chunk=row_chunk, token_chunk=args.token_chunk)
            except torch.cuda.OutOfMemoryError:
                if not pending:
                    raise
                # the held groups free their memory first; the attention reads only rows not yet written: retry
                self.moe_batch_cap = max(1, len(pending))
                self.receipt["events"].append({"layer": layer_idx, "action": "attention_oom_flush_and_retry",
                                               "held_groups": len(pending), "at_utc": utc()})
                flush()
                torch.cuda.empty_cache()
                m = 1
                carried, ffn_in = self.attention_half_accel(runner, pair, view, self.device,
                                                            row_chunk=row_chunk, token_chunk=args.token_chunk)
            if weights.is_moe:
                with clock.stage("ffn.latent_proj"):
                    latent = kdp._chunked_linear(ffn_in, w.body["block_sparse_moe.routed_expert_down_proj.weight"],
                                                 args.token_chunk)
                plan = self.moe_route_accel(w, ffn_in, latent, self.cfg)
                pending.append({"g0": g0, "g1": g1, "carried": carried, "ffn_in": ffn_in, "plan": plan})
                del latent, plan
                if len(pending) >= m or gi == len(groups) - 1:
                    flush()
            else:
                new_prefix, _ = self.ffn_half_accel(runner, carried, ffn_in, self.cfg, device=self.device,
                                                    token_chunk=args.token_chunk)
                finish(g0, g1, new_prefix)
                del new_prefix
            del carried, ffn_in, pair
        if self.cuda:
            torch.cuda.synchronize(self.device)
        compute_wall = time.time() - compute_start
        if snapshot:
            nb += 1
        positions, real = self.corpus.tokens, self.real_positions
        entry = {
            "layer": layer_idx, "is_kda": bool(weights.is_kda), "moe": bool(weights.is_moe), "snapshot": snapshot,
            "read": {k: timing[k] for k in ("bytes", "gigabytes", "seconds", "MB_per_s", "compute_dtype",
                                            "compute_dtype_source")},
            "read_wall_seconds": round(read_wall, 2),
            "compute_seconds": round(compute_wall, 2),
            "compute_positions": positions, "compute_real_positions": real,
            "compute_positions_per_s": round(positions / max(compute_wall, 1e-9), 1),
            "compute_real_tokens_per_s": round(real / max(compute_wall, 1e-9), 1),
            "dequant_seconds_engine": round(dequant, 2),
            "dequant_note": ("1.3.0: host-side launch time of the exact pair-table dequant (no sync inside the "
                             "routed loop); the true dequant time only under --stage-timers"),
            "compute_minus_dequant_seconds": round(compute_wall - dequant, 2),
            "row_chunk": row_chunk, "groups": len(self.corpus.groups), "blocks_after": nb,
            "layer_bytes_on_device": weights.device_bytes(self.device),
            "device_peak_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.cuda else None,
            "rss_before_bytes": rss_before, "rss_after_bytes": vmrss_bytes(),
            "peak_rss_bytes": common.peak_rss_bytes(),
            "accel": {"read_background_seconds": timing.get("seconds"),
                      "loader_wait_seconds": timing.get("loader_wait_seconds"),
                      "h2d_seconds": timing.get("h2d_seconds"), "reader": timing.get("reader", "engine.read_layer"),
                      "checkpoint_row_wait_seconds": round(ckpt_wait, 2),
                      "moe_group_batch": {**batch_info, "batches": batches}},
        }
        if self.kda_graph is not None:
            entry["accel"]["kda_graph"] = {k: self.kda_graph.counts[k] - graph_before[k] for k in graph_before}
            entry["accel"]["kda_graph"]["verified_equal"] = self.kda_graph.verified.get(layer_idx)
        if clock.enabled:
            entry["stages"] = clock.take()
        if args.prefix_digests:
            entry["prefix_out_sha256"] = common.sha256_tensor(prefix)
        if self.warmer is not None:
            entry["warmer"] = {s: self.warmer.results.get(s) for s in plan_shards[layer_idx]}
        self.receipt["layers"][str(layer_idx)] = entry
        log(f"L{layer_idx:02d} {'KDA' if weights.is_kda else 'MLA'}: read {read_wall:.1f}s "
            f"({timing['MB_per_s']} MB/s{', background' if self.loader is not None else ''}), "
            f"compute {compute_wall:.1f}s over {positions:,} positions "
            f"({real:,} real, {len(self.corpus.groups)} groups; {positions / max(compute_wall, 1e-9):,.0f} pos/s) "
            f"(dequant launch {dequant:.1f}s), "
            f"blocks {nb}, device peak {entry['device_peak_bytes'] or 0:,} B, rss {entry['rss_after_bytes'] or 0:,} B")
        del weights, runner
        if self.cuda and not self.args.hold_device:
            torch.cuda.empty_cache()
        return nb

    def attention_half_accel(self, runner, pair, corpus, device, *, row_chunk: int, token_chunk: int):
        """``k3_dense_pass.attention_half`` (1.2.5) statement for statement, except: the prefix rows and the block
        slice reach the card through pinned double buffers (same bytes, contiguous on the card as ``.to`` makes
        them), and the host-side ``torch.cat`` of the group's whole block stack at a snapshot layer -- discarded by
        1.2.4 (``carried, ffn_in, _ = ...``) -- is not built.  Returns (carried, ffn_in)."""
        kdp, clock = self.kdp, self.clock
        w = runner.w
        eps = runner.eps
        hidden = int(pair.prefix.shape[1])
        snapshot = (w.layer_idx % runner.block_size) == 0
        carried_all = torch.empty(corpus.tokens, hidden, dtype=pair.prefix.dtype, device=device)
        ffn_in_all = torch.empty_like(carried_all)
        for seg, r0, r1, start, stop in corpus.row_chunks(row_chunk):
            n_rows = r1 - r0
            with clock.stage("attn.h2d"):
                p = self.h2d_prefix(pair.prefix[start:stop])
                bl = self.h2d_blocks(pair.blocks[start:stop])
            with clock.stage("attn.attnres_pre"):
                if bl.shape[1] > 0:
                    hidden_states = kdp.apply_attn_res(p, bl, w.res["self_attention_res_proj.weight"],
                                                       w.res["self_attention_res_norm.weight"], eps,
                                                       token_chunk_size=token_chunk)
                else:
                    hidden_states = p
                if snapshot:
                    bl = torch.cat([bl, p.unsqueeze(1)], dim=1)
            with clock.stage("attn.core"):
                normed = kdp.rms_norm(hidden_states, w.res["input_layernorm.weight"], eps).view(n_rows, seg.T, hidden)
                attn_out, _ = runner.attention(normed, state=None)
            with clock.stage("attn.post"):
                attn_out = attn_out.reshape(-1, hidden)
                carried = attn_out if snapshot else p + attn_out
                h2 = kdp.apply_attn_res(carried, bl, w.res["mlp_res_proj.weight"], w.res["mlp_res_norm.weight"], eps,
                                        token_chunk_size=token_chunk)
                h2 = kdp.rms_norm(h2, w.res["post_attention_layernorm.weight"], eps)
                carried_all[start:stop] = carried
                ffn_in_all[start:stop] = h2
            del p, bl, hidden_states, normed, attn_out, carried, h2
        return carried_all, ffn_in_all

    def ffn_half_accel(self, runner, carried, ffn_in, cfg: dict, *, device, token_chunk: int):
        """``k3_dense_pass.ffn_half`` (teacher branch: no dense swap-in, latent / y_lat computed here) statement for
        statement, with ``routed_y_lat_accel`` for the routed accumulator.  Returns (new prefix, dequant launch s)."""
        kdp, clock = self.kdp, self.clock
        w = runner.w
        if not w.is_moe:
            with clock.stage("ffn.dense"):
                out = torch.empty_like(ffn_in)
                for start in range(0, ffn_in.shape[0], token_chunk):
                    stop = min(start + token_chunk, ffn_in.shape[0])
                    out[start:stop] = kdp.dense_mlp_forward(w, ffn_in[start:stop], cfg)
                return carried + out, 0.0
        with clock.stage("ffn.latent_proj"):
            latent = kdp._chunked_linear(ffn_in, w.body["block_sparse_moe.routed_expert_down_proj.weight"], token_chunk)
        y_lat, dequant_seconds = self.routed_y_lat_accel(w, ffn_in, latent, cfg, device)
        with clock.stage("ffn.moe_write"):
            out = kdp.moe_write(w, ffn_in, y_lat, cfg, token_chunk)
            new_prefix = carried + out
        return new_prefix, dequant_seconds

    def pair_table(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self.pair_tables:
            self.pair_tables[key] = e2m1_pair_table(self.common, device)
        return self.pair_tables[key]

    def moe_route_accel(self, weights, x: torch.Tensor, latent: torch.Tensor, cfg: dict) -> dict[str, Any]:
        """The routing half of ``k3_dense_pass.routed_y_lat`` for one row group, with ONE host sync: a stable sort
        of the flat top-k index gives, per expert in ascending id order, its rows ascending (``nonzero`` order; an
        expert appears at most once per token) and its single top-k weight (what the engine's masked sum over k adds
        exact zeros to).  Returns the group's plan; ``moe_experts_accel`` fills its ``y_lat``."""
        kdp = self.kdp
        with self.clock.stage("ffn.moe.route"):
            topk_idx, topk_weight, _ = kdp.route(x, weights, cfg)
            y_lat = torch.zeros(latent.shape[0], latent.shape[1], dtype=torch.float32, device=latent.device)
            k = topk_idx.shape[1]
            flat = topk_idx.flatten()
            order = torch.sort(flat, stable=True).indices
            tokens = order // k
            flat_w = topk_weight.flatten()
            counts = torch.bincount(flat, minlength=int(cfg["num_experts"])).tolist()  # the one host sync
        spans, offset = {}, 0
        for expert, n in enumerate(counts):
            if n:
                spans[expert] = (offset, n)
                offset += n
        return {"latent": latent, "y_lat": y_lat, "order": order, "tokens": tokens, "flat_w": flat_w, "spans": spans}

    def moe_experts_accel(self, weights, plans: list[dict[str, Any]], cfg: dict, device) -> float:
        """The expert loop of ``k3_dense_pass.routed_y_lat`` over one or more row groups' plans: experts in ascending
        id order, each dequantized ONCE (the exact pair-table form) for every group it serves; per group the
        latent[rows] gather, the SiTU, the three GEMMs, the fp32 cast, the weighting and the index add are the
        engine's statements on the same tensors, accumulated in the same (ascending expert) order -- so each group's
        y_lat is the one it gets alone.  Returns the host seconds spent issuing the dequants."""
        kdp, clock = self.kdp, self.clock
        F = torch.nn.functional
        beta = cfg.get("activation_situ_beta") or 1.0
        linear_beta = cfg.get("activation_situ_linear_beta")
        dtype = plans[0]["latent"].dtype
        dequant_seconds = 0.0
        for expert in sorted(set().union(*[p["spans"] for p in plans])):
            started = time.time()
            with clock.stage("ffn.moe.dequant"):
                packed = weights.experts[expert]
                ex = {}
                for name in ("w1", "w2", "w3"):
                    pk = packed[f"{name}.weight_packed"]
                    ex[name] = dequant_mxfp4_exact(pk, packed[f"{name}.weight_scale"], self.pair_table(pk.device),
                                                   dtype=dtype)
                ex = {kk: v.to(device) for kk, v in ex.items()}
            dequant_seconds += time.time() - started
            for plan in plans:
                span = plan["spans"].get(expert)
                if span is None:
                    continue
                offset, n = span
                with clock.stage("ffn.moe.expert_ffn"):
                    sel = plan["order"][offset:offset + n]
                    rows = plan["tokens"][offset:offset + n]
                    weight = plan["flat_w"][sel]
                    chunk = plan["latent"][rows]
                    activated = kdp.situ(F.linear(chunk, ex["w1"]), F.linear(chunk, ex["w3"]), beta, linear_beta)
                    plan["y_lat"][rows] += F.linear(activated, ex["w2"]).float() * weight.unsqueeze(-1)
                del chunk, activated
            del ex
        return dequant_seconds

    def routed_y_lat_accel(self, weights, x: torch.Tensor, latent: torch.Tensor, cfg: dict, device):
        """``k3_dense_pass.routed_y_lat``'s accumulator for one row group (route + expert loop above)."""
        plan = self.moe_route_accel(weights, x, latent, cfg)
        dequant_seconds = self.moe_experts_accel(weights, [plan], cfg, device)
        return plan["y_lat"], dequant_seconds

    def moe_group_batch(self, weights, seg_t: int, group_positions: int, n_groups: int) -> tuple[int, dict[str, Any]]:
        """How many row groups share one dequant per expert in this layer: ``--moe-group-batch N`` or, for ``auto``,
        as many as the card holds -- per group its carried + FFN input (compute dtype), latent (compute dtype),
        fp32 accumulator and routing indices -- after a reserve for one row's attention (an MLA row's math-SDPA
        scores and softmax at fp32 when the layer is MLA) and the dequant scratch."""
        args = self.args
        info: dict[str, Any] = {"policy": args.moe_group_batch}
        if not weights.is_moe or n_groups <= 1:
            return 1, info
        if self.moe_batch_cap is not None:
            info["capped_after_oom"] = self.moe_batch_cap
        if args.moe_group_batch != "auto":
            return max(1, min(int(args.moe_group_batch), n_groups, self.moe_batch_cap or n_groups)), info
        if not self.cuda:
            return min(n_groups, self.moe_batch_cap or n_groups), info
        if self.moe_batch_cap is not None and self.moe_batch_cap <= 1:
            return 1, info
        hidden = int(self.index.hidden_size)
        latent_dim = int(weights.body["block_sparse_moe.routed_expert_down_proj.weight"].shape[0])
        elem = torch.tensor([], dtype=self.compute_dtype or torch.bfloat16).element_size()
        topk = int(self.cfg["num_experts_per_token"])
        per_group = group_positions * (2 * hidden * elem + latent_dim * (elem + 4) + topk * 32)
        free = torch.cuda.mem_get_info(self.device)[0] +             torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        reserve = 3 << 30
        if not weights.is_kda:
            heads = int(self.cfg["num_attention_heads"])
            reserve += 2 * heads * seg_t * seg_t * 4
        m = int((free - reserve) // max(per_group, 1))
        m = max(1, min(m, n_groups, self.moe_batch_cap or n_groups))
        info.update({"free_bytes": int(free), "reserve_bytes": int(reserve), "per_group_bytes": int(per_group), "m": m})
        return m, info

    # -- 1.2.4: the reference layer (--accel off) --------------------------------------------
    def layer_reference(self, run, layer_idx: int, prefix: torch.Tensor, blocks: torch.Tensor, nb: int,
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
        positions, real = self.corpus.tokens, self.real_positions
        entry = {
            "layer": layer_idx, "is_kda": bool(weights.is_kda), "moe": bool(weights.is_moe), "snapshot": snapshot,
            "read": {k: timing[k] for k in ("bytes", "gigabytes", "seconds", "MB_per_s", "compute_dtype",
                                            "compute_dtype_source")},
            "read_wall_seconds": round(read_wall, 2),
            "compute_seconds": round(compute_wall, 2),
            # the compute covers every padded position of the pass, not only the real tokens
            "compute_positions": positions, "compute_real_positions": real,
            "compute_positions_per_s": round(positions / max(compute_wall, 1e-9), 1),
            "compute_real_tokens_per_s": round(real / max(compute_wall, 1e-9), 1),
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
            f"({timing['MB_per_s']} MB/s), compute {compute_wall:.1f}s over {positions:,} positions "
            f"({real:,} real, {len(self.corpus.groups)} groups; {positions / max(compute_wall, 1e-9):,.0f} pos/s) "
            f"(engine dequant {dequant:.1f}s), "
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
                "layers": len(layers), "compute_positions_per_layer": self.corpus.tokens,
                "compute_real_positions_per_layer": self.real_positions,
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
                self.receipt["state_removed"] = self.remove_own_state()
                write_json(os.path.join(self.pass_dir, "state_removal.json"), self.receipt["state_removed"])
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
                seen = self.files.get(f["name"])
                if seen is not None and seen.get("file_sha256") != f.get("file_sha256"):
                    # every distill pass names its file k3_input_ids: rows keyed by name would overwrite each other
                    raise SystemExit(f"{path}: file name {f['name']!r} is also used by another pass dir with a different "
                                     "file sha256; read each pass with its own boundary map")
                self.files[f["name"]] = f
            for shard in manifest["shards"]:
                if sha256_file(shard["path"]) != shard["sha256"]:
                    raise SystemExit(f"{shard['path']} does not match its manifest sha256")
                got = load_file(shard["path"])
                pos = 0
                for i in range(int(got["real_lengths"].numel())):
                    n = int(got["real_lengths"][i])
                    key = (names[int(got["row_file"][i])], int(got["row_index"][i]))
                    if key in self.rows:
                        raise SystemExit(f"{path}: row {key} is already in another shard / pass dir")
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


def export_reference(cache: CacheReader, positions_path: str, out_path: str) -> dict[str, Any]:
    """``cache`` in the fold engine's review-dump layout (``teacher_topk_ids`` / ``teacher_topk_logprobs`` top-k,
    ``teacher_argmax``, ``teacher_next_token_logprob``, ``row_index``, ``position``) at EXACTLY the (row, position)s
    of ``positions_path`` (a fold ``<file>_teacher_forced.safetensors``), k = that dump's k.  ``parity`` reads it like
    a fold dump; nothing about the comparison changes.  Logprobs are written fp32 (the cache's own values, never
    re-rounded); every position the dump names must be in the cache, or nothing is written."""
    from safetensors.torch import load_file, save_file

    src = load_file(positions_path)
    name = os.path.basename(positions_path).replace("_teacher_forced.safetensors", "")
    if os.path.basename(out_path) != f"{name}_teacher_forced.safetensors":
        raise SystemExit(f"--out must be named {name}_teacher_forced.safetensors (parity keys the file by that name)")
    if os.path.abspath(out_path) == os.path.abspath(positions_path):
        raise SystemExit("--out would overwrite the positions source")
    rows, positions = src["row_index"].long(), src["position"].long()
    k = int(src["teacher_topk_ids"].shape[1])
    if k > TOPK:
        raise SystemExit(f"{positions_path} carries top-{k}; the cache holds top-{TOPK}")
    ids, lps, tlp, missing = [], [], [], []
    for i in range(rows.numel()):
        got = cache.rows.get((name, int(rows[i])))
        p = int(positions[i])
        if got is None or p >= got["lse"].numel():
            missing.append((int(rows[i]), p))
            continue
        ids.append(got["topk_ids"][p, :k])
        lps.append(got["topk_logprobs"][p, :k].float())
        tlp.append(got["target_logprob"][p].float())
    if missing:
        raise SystemExit(f"{len(missing)} of {rows.numel()} (row, position)s of {positions_path} are not in the cache "
                         f"(first {missing[:3]}); nothing written")
    top = torch.stack(ids).to(torch.int32).contiguous()
    payload = {"teacher_topk_ids": top, "teacher_topk_logprobs": torch.stack(lps).contiguous(),
               "teacher_argmax": top[:, 0].contiguous(), "teacher_next_token_logprob": torch.stack(tlp).contiguous(),
               "row_index": src["row_index"].clone().contiguous(), "position": src["position"].clone().contiguous()}
    sources = []
    for m in cache.manifests:
        with open(m["path"], "r", encoding="utf-8") as handle:
            args = json.load(handle).get("arguments") or {}
        sources.append({**m, "compute_dtype": args.get("compute_dtype"), "group_tokens": args.get("group_tokens"),
                        "no_trim": args.get("no_trim")})
    meta = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "what": "k3_teacher_cache export-reference",
            "positions_from": os.path.abspath(positions_path), "positions_from_sha256": sha256_file(positions_path),
            "source_caches": json.dumps(sources, sort_keys=True), "topk": str(k)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_file(payload, out_path, metadata=meta)
    return {"file": name, "out": os.path.abspath(out_path), "sha256": sha256_file(out_path), "positions": int(rows.numel()),
            "topk": k, "positions_from": meta["positions_from"], "positions_from_sha256": meta["positions_from_sha256"],
            "source_caches": sources}


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
    pairs = torch.as_tensor(row["pairs"], dtype=torch.long).reshape(-1, 2)
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
        pairs = torch.as_tensor(row["pairs"], dtype=torch.long).reshape(-1, 2)
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


def load_distill_pass(corpus_dir: str) -> dict[str, Any]:
    """One ``distill_corpus_build`` K3 pass directory: the K3 / Qwen ids, the aligned pairs and the per-document
    routes, every file checked against the sha256 that ``boundary_map.json`` names."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with open(os.path.join(corpus_dir, "boundary_map.json"), "r", encoding="utf-8") as handle:
        bmap = json.load(handle)
    if "pairs_file" not in bmap or "rows" not in bmap:
        raise SystemExit(f"{corpus_dir}/boundary_map.json is not the distill layout (pairs_file + rows)")
    pairs_path = os.path.join(corpus_dir, "boundary_map.safetensors")
    checks = {pairs_path: bmap["pairs_file"]["sha256"]}
    for side in ("k3_input_ids", "qwen_input_ids"):
        checks[os.path.join(corpus_dir, f"{side}.safetensors")] = bmap["ids_files_sha256"][side]
    for path, want in checks.items():
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path} sha256 {got[:12]} != {want[:12]} named by {corpus_dir}/boundary_map.json")
    pairs = load_file(pairs_path)
    with safe_open(os.path.join(corpus_dir, "k3_input_ids.safetensors"), framework="pt") as handle:
        k3_meta = handle.metadata() or {}
    k3 = load_file(os.path.join(corpus_dir, "k3_input_ids.safetensors"))
    qwen = load_file(os.path.join(corpus_dir, "qwen_input_ids.safetensors"))
    with open(os.path.join(corpus_dir, "row_meta.json"), "r", encoding="utf-8") as handle:
        row_meta = json.load(handle)
    routes = {int(r["row_index"]): list(r.get("routes") or []) for r in row_meta["rows"]}
    offsets = pairs["row_offsets"].long()
    if offsets.numel() != int(k3["input_ids"].shape[0]) + 1 or int(offsets[-1]) != int(pairs["k3_position"].numel()):
        raise SystemExit(f"{pairs_path}: row_offsets {offsets.numel()} / {int(offsets[-1])} disagree with "
                         f"{int(k3['input_ids'].shape[0])} rows / {int(pairs['k3_position'].numel())} pairs")
    return {"dir": os.path.abspath(corpus_dir), "name": k3_meta.get("file", "k3_input_ids"),
            "k3_separator_id": int(k3_meta["separator_id"]), "k3_ids": k3["input_ids"].long(),
            "k3_real": k3["real_lengths"].long().flatten(), "qwen_ids": qwen["input_ids"].long(),
            "qwen_real": qwen["real_lengths"].long().flatten(), "offsets": offsets,
            "qwen_position": pairs["qwen_position"].long(), "k3_position": pairs["k3_position"].long(),
            "routes": routes, "boundary_map_sha256": sha256_file(os.path.join(corpus_dir, "boundary_map.json")),
            "row_meta_sha256": sha256_file(os.path.join(corpus_dir, "row_meta.json"))}


def probe_distill_docs(cache_rows: dict, d: dict[str, Any], k3_len: torch.Tensor, *, qwen_separator_id: int,
                       q_target: torch.Tensor | None = None, q_len: torch.Tensor | None = None) -> dict[str, Any]:
    """Per DOCUMENT of every packed row: the span from the boundary after the previous separator (the row's first
    aligned pair for the first document) to the last aligned pair before the document's own separator.  K3 nats
    are -sum target_logprob[ka:kb] (the tokens ka+1..kb, which consumed exactly the span's bytes); Q38's, when
    a Q-text target_logprob [rows, T] is given, -sum over [qa:qb] on the same bytes."""
    docs, problems = [], {"rows_not_in_cache": 0, "rows_cache_length_mismatch": 0, "rows_route_count_mismatch": 0,
                          "separator_pairs_not_qwen_separator": 0, "empty_documents": 0, "q_span_byte_mismatches": 0}
    sep = int(d["k3_separator_id"])
    for r in range(int(d["offsets"].numel()) - 1):
        got = cache_rows.get((d["name"], r))
        if got is None:
            problems["rows_not_in_cache"] += 1
            continue
        if int(got["target_logprob"].numel()) != int(d["k3_real"][r]):
            problems["rows_cache_length_mismatch"] += 1
            continue
        a0, a1 = int(d["offsets"][r]), int(d["offsets"][r + 1])
        kp, qp = d["k3_position"][a0:a1], d["qwen_position"][a0:a1]
        ids, qids = d["k3_ids"][r], d["qwen_ids"][r]
        seps = (ids[kp] == sep).nonzero().flatten().tolist()
        routes = d["routes"].get(r, [])
        if len(seps) != len(routes):
            problems["rows_route_count_mismatch"] += 1
            continue
        problems["separator_pairs_not_qwen_separator"] += int((qids[qp[seps]] != int(qwen_separator_id)).sum()) if seps else 0
        start = 0
        for doc, si in enumerate(seps):
            a, b = start, si - 1
            start = si
            if b <= a:
                problems["empty_documents"] += 1
                continue
            ka, kb, qa, qb = int(kp[a]), int(kp[b]), int(qp[a]), int(qp[b])
            span_bytes = int(k3_len[ids[ka + 1: kb + 1]].sum())
            if span_bytes <= 0:
                problems["empty_documents"] += 1
                continue
            k3_nats = float(-got["target_logprob"][ka:kb].double().sum())
            rec = {"row": r, "doc": doc, "route": routes[doc], "bytes": span_bytes, "k3_span": [ka, kb],
                   "q_span": [qa, qb], "k3_tokens": kb - ka, "k3_nats": k3_nats,
                   "k3_bpb": k3_nats / math.log(2) / span_bytes}
            if q_target is not None:
                q_nats = float(-q_target[r, qa:qb].double().sum())
                rec.update({"q_tokens": qb - qa, "q_nats": q_nats, "q_bpb": q_nats / math.log(2) / span_bytes})
                if q_len is not None:
                    q_bytes = int(q_len[qids[qa + 1: qb + 1]].sum())
                    if q_bytes != span_bytes:
                        problems["q_span_byte_mismatches"] += 1
                        rec["q_span_bytes"] = q_bytes
            docs.append(rec)
    return {"docs": docs, "problems": problems}


def aggregate_routes(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per: dict[str, dict[str, Any]] = {}
    for rec in docs:
        agg = per.setdefault(rec["route"], {"docs": 0, "bytes": 0, "k3_tokens": 0, "k3_nats": 0.0, "q_nats": 0.0,
                                            "q_docs": 0})
        agg["docs"] += 1
        agg["bytes"] += rec["bytes"]
        agg["k3_tokens"] += rec["k3_tokens"]
        agg["k3_nats"] += rec["k3_nats"]
        if "q_nats" in rec:
            agg["q_nats"] += rec["q_nats"]
            agg["q_docs"] += 1
    for agg in per.values():
        agg["k3_bpb"] = agg["k3_nats"] / math.log(2) / max(agg["bytes"], 1)
        if agg["q_docs"] == agg["docs"] and agg["docs"]:
            agg["q_bpb"] = agg["q_nats"] / math.log(2) / max(agg["bytes"], 1)
            agg["k3_minus_q_bpb"] = agg["k3_bpb"] - agg["q_bpb"]
    return per


def load_boundary_map(path: str) -> dict[str, Any]:
    """A boundary map in either layout, as rows ``{file, row_index, qwen_tokens_real, k3_tokens_real, pairs}``.

    * composite (k3_corpus_build): the JSON rows carry ``file`` and the ``pairs`` lists themselves;
    * distill (distill_corpus_build 1.0.1, one pass dir): the JSON names ``pairs_file`` and its rows carry neither;
      the pairs, both ids files and row_meta.json come from ``load_distill_pass`` (each sha-checked against this
      JSON), and every row's count / real lengths must agree with the JSON row.
    ``ids_files_sha256`` is returned for k3_join (the K3 side) and, through the xtok receipt, the trainer (Qwen side).
    """
    with open(path, "r", encoding="utf-8") as handle:
        bmap = json.load(handle)
    out: dict[str, Any] = {"path": os.path.abspath(path), "sha256": sha256_file(path),
                           "ids_files_sha256": dict(bmap.get("ids_files_sha256") or {})}
    if "pairs_file" in bmap:
        d = load_distill_pass(os.path.dirname(os.path.abspath(path)))
        meta = {int(r["row_index"]): r for r in bmap["rows"]}
        rows = []
        for r in range(int(d["offsets"].numel()) - 1):
            a0, a1 = int(d["offsets"][r]), int(d["offsets"][r + 1])
            m = meta.get(r)
            if m is None or int(m["aligned_pairs"]) != a1 - a0 or int(m["qwen_tokens_real"]) != int(d["qwen_real"][r]) \
                    or int(m["k3_tokens_real"]) != int(d["k3_real"][r]):
                raise SystemExit(f"{path}: row {r} of the JSON disagrees with the pairs / ids files ({m})")
            rows.append({"file": d["name"], "row_index": r, "qwen_tokens_real": int(d["qwen_real"][r]),
                         "k3_tokens_real": int(d["k3_real"][r]),
                         "pairs": torch.stack([d["qwen_position"][a0:a1], d["k3_position"][a0:a1]], dim=1)})
        out.update(format="distill-safetensors", rows=rows, row_meta_sha256=d["row_meta_sha256"],
                   pairs_file={**bmap["pairs_file"], "sha256_checked": True})
        return out
    if any("pairs" not in row or "file" not in row for row in bmap.get("rows", [])):
        raise SystemExit(f"{path}: rows carry no file / pairs and the map names no pairs_file -- neither layout")
    out.update(format="json-pairs", rows=bmap["rows"], pairs_file=None)
    return out


def k3_join(cache: "CacheReader", boundary: dict[str, Any]) -> dict[str, Any]:
    """The K3 side of the cross-tokenizer join, proven.  The cache keys rows by (file NAME, row), and all four
    distill passes name their K3 file k3_input_ids, so a pass-1 map on a pass-0 cache would pair every row with
    another row's positions, every value still finite.  The map names the K3 file's sha; the cache recorded the
    sha of the file it read: they must be equal."""
    out: dict[str, Any] = {}
    for name in sorted({str(row["file"]) for row in boundary["rows"]}):
        want = boundary["ids_files_sha256"].get(name)
        have = (cache.files.get(name) or {}).get("file_sha256")
        if have is None:
            out[name] = {"verified": False, "reason": "no such file in the cache"}
        elif want is None:
            out[name] = {"verified": False, "reason": "the boundary map names no sha for it", "cache_file_sha256": have}
        elif want != have:
            raise SystemExit(f"boundary map {boundary['path']} pairs K3 file {name!r} sha {want[:12]}, but the cache was "
                             f"built from {name!r} sha {have[:12]}: it is another pass's map")
        else:
            out[name] = {"verified": True, "file_sha256": have}
    return out


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


def cmd_export_reference(args) -> int:
    cache = CacheReader(args.cache_dir)
    out = export_reference(cache, args.positions_from, args.out)
    log(f"export-reference {out['file']}: {out['positions']} positions top-{out['topk']} from "
        f"{[(s['compute_dtype'], s['group_tokens']) for s in out['source_caches']]} -> {out['out']} (sha256 {out['sha256'][:12]})")
    return 0


def cmd_xtok(args) -> int:
    cache = CacheReader(args.cache_dir)
    boundary = load_boundary_map(args.boundary_map)
    join = k3_join(cache, boundary)
    qwen, k3, report = bind_tokenizers(args)
    mapping, stats = build_vocab_map(k3, qwen, int(args.k3_vocab), QWEN_TEXT_ROWS)
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
               "boundary_map": {"path": args.boundary_map, "sha256": sha256_file(args.boundary_map),
                                "format": boundary["format"], "pairs_file": boundary["pairs_file"]},
               "k3_join": join,
               "tokenizers": report, "vocab_map": {**stats, "path": map_path, "file_sha256": sha256_file(map_path)},
               "target": {"path": path, "sha256": sha256_file(path), **summary}, "qwen_files": got["qwen_files"]}
    digest = write_json(os.path.join(args.out, "xtok_receipt.json"), receipt)
    log(f"xtok: {summary} map {stats['k3_mapped']}/{stats['k3_vocab']} K3 ids -> receipt sha256 {digest[:12]}")
    return 0


def cmd_probe(args) -> int:
    cache = CacheReader(args.cache_dir)
    boundary = load_boundary_map(args.boundary_map)
    k3_join(cache, boundary)
    qwen, k3, report = bind_tokenizers(args)
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


def cmd_probe_distill(args) -> int:
    qwen, k3, report = bind_tokenizers(args)
    k3_len = token_byte_lengths(k3, int(args.k3_vocab))
    q_len = None
    passes, docs, problems = [], [], {}
    for spec in args.pass_spec:
        parts = spec.split("=")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--pass {spec!r}: want CACHE_PASS_DIR=CORPUS_PASS_DIR[=Q_TARGET.safetensors]")
        cache = CacheReader([parts[0]])
        d = load_distill_pass(parts[1])
        if cache.manifests[0]["corpus_sha256"] != corpus_sha256([load_ids_file(os.path.join(parts[1], "k3_input_ids.safetensors"))]):
            raise SystemExit(f"{parts[0]} was not built from {parts[1]}/k3_input_ids.safetensors (corpus sha differs)")
        q_target = None
        if len(parts) == 3:
            from safetensors.torch import load_file

            q_target = load_file(parts[2])["target_logprob"].float()
            if q_len is None:
                q_len = token_byte_lengths(qwen, int(qwen.vocab_size))
        got = probe_distill_docs(cache.rows, d, k3_len, qwen_separator_id=int(args.qwen_separator_id),
                                 q_target=q_target, q_len=q_len)
        for k, v in got["problems"].items():
            problems[k] = problems.get(k, 0) + v
        for rec in got["docs"]:
            rec["pass"] = len(passes)
        docs += got["docs"]
        passes.append({"cache": cache.manifests[0], "corpus_dir": d["dir"], "boundary_map_sha256": d["boundary_map_sha256"],
                       "row_meta_sha256": d["row_meta_sha256"], "q_target": parts[2] if len(parts) == 3 else None,
                       "docs": len(got["docs"]), "problems": got["problems"]})
    per_route = aggregate_routes(docs)
    total = aggregate_routes([dict(r, route="ALL") for r in docs]).get("ALL")
    receipt = {"schema": SCHEMA + ".probe_distill", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
               "tool_sha256": sha256_file(os.path.abspath(__file__)), "at_utc": utc(), "tokenizers": report,
               "unit": "document: boundary after the previous separator (or the row's first pair) to the last aligned "
                       "pair before its own separator; no separator prediction scored",
               "passes": passes, "problems": problems, "all": total, "per_route": per_route, "docs": docs,
               "note": "informs lambda_K3 and the route mix only; K3 is distilled in every outcome"}
    digest = write_json(args.out, receipt)
    for route, agg in sorted(per_route.items(), key=lambda kv: -kv[1]["bytes"]):
        log(f"route {route}: docs {agg['docs']} bytes {agg['bytes']:,} K3 {agg['k3_bpb']:.4f} bpb"
            + (f" Q {agg['q_bpb']:.4f} bpb" if "q_bpb" in agg else ""))
    if total:
        log(f"ALL: docs {total['docs']} bytes {total['bytes']:,} K3 {total['k3_bpb']:.4f} bpb; problems {problems}")
    log(f"probe-distill -> {args.out} (sha256 {digest[:12]})")
    return 0 if docs and not problems.get("rows_not_in_cache") and not problems.get("rows_cache_length_mismatch") else 4


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
    r.add_argument("--state-dir", default=None,
                   help="layer checkpoints go to <state-dir>/<pass_sha[:16]>/ (default state-dir <pass-dir>/state)")
    r.add_argument("--pass-count", type=int, default=1)
    r.add_argument("--pass-index", type=int, default=0)
    r.add_argument("--max-pass-tokens", type=int, default=MAX_PASS_TOKENS, help="REAL tokens per pass")
    r.add_argument("--max-host-state-gb", type=float, default=MAX_HOST_STATE_GB,
                   help="host state of the pass's padded positions (prefix + block slots + working h)")
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
    a = r.add_argument_group("1.3.0 acceleration (bit-exact: scheduling, copies and syncs only)")
    a.add_argument("--accel", default="on", choices=("on", "off"), help="off = the 1.2.4 code path exactly")
    a.add_argument("--no-layer-loader", dest="layer_loader", action="store_false", default=True,
                   help="read each layer on the main thread with the engine's read_layer (1.2.4)")
    a.add_argument("--layer-buffer", default="pinned", choices=("pinned", "pageable"),
                   help="the loader's host arena: registered page-locked memory (async H2D) or pageable")
    a.add_argument("--no-kda-graph", dest="kda_graph", action="store_false", default=True,
                   help="run the KDA recurrence eagerly (1.2.4)")
    a.add_argument("--kda-graph-verify", default="layer", choices=("layer", "once", "off"),
                   help="bitwise graph-vs-eager check on the first KDA row of every layer / of the pass / never")
    a.add_argument("--kda-graph-keep", default="auto", choices=("auto", "always", "kda-only"),
                   help="keep the graph across MLA layers (auto: on cards with >= 32 GiB; kda-only: release it "
                        "before every non-KDA layer and capture again at the next KDA layer)")
    a.add_argument("--moe-group-batch", default="auto",
                   help="row groups whose routed MoE shares one dequant per expert: auto (as many as the card holds; "
                        "1 on a 24 GB card at 65,536-token groups) or N; 1 = one group at a time (1.2.4's loop order)")
    a.add_argument("--no-fadvise-drop", dest="fadvise_drop", action="store_false", default=True,
                   help="keep the page cache of the ranges the loader consumed")
    a.add_argument("--no-pinned-staging", dest="pinned_staging", action="store_false", default=True,
                   help="pageable .to(device) for the prefix rows and block slices (1.2.4)")
    a.add_argument("--no-async-checkpoint", dest="async_checkpoint", action="store_false", default=True,
                   help="write checkpoints on the main thread with save_tensors + a re-read sha (1.2.4)")
    a.add_argument("--stage-timers", action="store_true",
                   help="PROBE: per-stage wall in the layer receipt (synchronises the card at every stage)")
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("parity", help="cache top-k vs the fold engine's review dump")
    p.add_argument("--cache-dir", action="append", required=True)
    p.add_argument("--reference", action="append", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_parity)

    e = sub.add_parser("export-reference", help="a sealed cache in the fold review-dump layout (a parity reference)")
    e.add_argument("--cache-dir", action="append", required=True)
    e.add_argument("--positions-from", required=True, help="the fold's <file>_teacher_forced.safetensors")
    e.add_argument("--out", required=True, help="<dir>/<file>_teacher_forced.safetensors")
    e.set_defaults(func=cmd_export_reference)

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

    x = sub.add_parser("probe-distill", help="per-route bits per byte over distill_corpus_build K3 passes, per document")
    x.add_argument("--pass", dest="pass_spec", action="append", required=True,
                   help="CACHE_PASS_DIR=CORPUS_PASS_DIR[=Q_TARGET.safetensors] (one per sealed K3 pass)")
    x.add_argument("--out", required=True, help="receipt path")
    x.add_argument("--qwen-dir", default="/home/cereal/models/Qwen3.8-27B-Uncensored")
    x.add_argument("--k3-dir", default=DEFAULT_SRC_DIR)
    x.add_argument("--k3-vocab", type=int, default=163840)
    x.add_argument("--qwen-separator-id", type=int, default=QWEN_SEPARATOR_ID)
    x.add_argument("--chimera-src", default=os.environ.get("CHIMERA_OMNI_SRC"))
    x.add_argument("--corpus-tools-dir", default=DEFAULT_CORPUS_TOOLS_DIR)
    x.set_defaults(func=cmd_probe_distill)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
