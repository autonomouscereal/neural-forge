#!/usr/bin/env python3
"""qtext_teacher_cache.py -- the Q-text / Q-long / self-gen-rescoring teacher cache writer,
and tool T(i), the streamed vision-language teacher.

WHY THIS FILE EXISTS.  Stage C of the distillation design
(`distill/DISTILL_DESIGN_2026-10-12.md` section 3) writes every Qwen3.8-27B-U teacher
output ONCE, sealed and content-addressed, before any student step runs: "No teacher
runs inside a training loop."  The design names the payload exactly -- top-64 bf16
logprobs + int32 ids + the full-row logsumexp, at text positions only, over vocab rows
[0, 248,320) only -- and names the engine: the layer-streamed forward in
`distill/tps_probe.py` (696.19 tok/s at L=4,096 mb8 on one 3090, bit-identical to a
stock full load on the 0.8B).  CAMPAIGN 5.2 lists the writer as NOT WRITTEN.  This is it.

WHAT IS REUSED, NOT RE-IMPLEMENTED.  The text trunk is
`tps_probe.build_streamed_text_model` -- the stock `Qwen3_5TextModel` prologue, the
stock decoder layer and the stock epilogue, with `layers` replaced by a generator that
loads one layer at a time.  This file imports that module by path and seals its sha256
in every manifest; it does not carry a second copy of the engine.

T(i): THE STREAMED VISION-LANGUAGE TEACHER.  The existing engine builds a
`Qwen3_5TextModel` only and the vision tower is dropped (design A3).  Vision-in and
video-in answer caches need the parent's own `model.visual`.  T(i) builds the STOCK
`Qwen3_5Model` on the meta device and swaps in two real submodules:
  * `visual`         -- the parent's vision tower, resident (0.45B params on the 27B),
                        built at fp32 ON THE CPU so its non-persistent `inv_freq` buffer
                        is the fp32 CPU value `from_pretrained` leaves (computed on a GPU
                        it differs in the last bit), moved, loaded strict, and only its
                        PARAMETERS cast to the run dtype;
  * `language_model` -- the streamed text trunk above.
The stock `Qwen3_5Model.forward` then runs unmodified: `get_image_features` /
`get_video_features` -> `get_placeholder_mask` + `masked_scatter` into `inputs_embeds`
-> `compute_3d_position_ids` (`get_rope_index`, M-RoPE) -> the streamed LLM.  Only the
module that holds the weights changed, so the parity arm can only be wrong about that.

WHAT ONE STORED POSITION IS.  Teacher-forced: position t of a row carries the teacher's
distribution over the token at t+1 given tokens [0, t].  A position is stored iff
  t < length(row)                 (right padding is never stored),
  store_mask[row, t]              (text corpora may carry one; default all ones),
  mm_token_type_ids[row, t] == 0  (VL rows: image/video placeholder positions are not
                                   text positions and are not stored).
A text corpus may instead (or as well) carry `target_mask[row, t]`, indexed by the
TARGET token: "token t is a target".  The tool shifts it itself -- position t is stored
iff target_mask[row, t+1] -- so a producer marks the completion tokens of a
prompt+completion row and the first stored position is the last prompt token (1.0.4,
finding L2: an input-indexed store_mask is an off-by-one hazard at the producer).  When a
corpus carries both, they must agree position for position or the build is refused.
Over the stored positions the head runs on rows [0, vocab_rows) of the lm_head only
(default 248,320: the ledger bands [248,320, 252,416) and the C18 music band
[252,416, 317,952) of a Chimera-vocab teacher are never part of a text target).  The
head precision is locked per cache (1.0.4, finding L3):
  fp32  (default) the head weight rows [0, V) are held as an exact fp32 upcast and the
        hidden state is upcast before the GEMM, with TF32 disabled, so the logits carry
        no bf16 rounding (a bf16 logit near 20 is rounded to a 1/16 step);
  model the stock causal-LM path: GEMM in the model dtype, then `logits.float()`;
and the shard keeps
    lse          = logsumexp(logits[0:V])                      fp32
    topk_ids     = top-64 row ids, descending by logit         int32
    topk_logprob = logit[topk_ids] - lse                       bf16
so logit = logprob + lse is recoverable to bf16 precision and the tail mass is
1 - sum(exp(topk_logprob)).

SHARD FILES (little-endian raw arrays, P = stored positions in the shard):
    ids.u32              [P]      input token id at the stored position
    pos.u32              [P]      position index of the stored position in its row
    topk_ids.i32         [P, K]
    topk_logprobs.bf16   [P, K]
    lse.f32              [P]
    shard.json           rows table (row index, row id, offset, count, length),
                         sha256 + byte count of every file, finite check, timings.
396 B per stored position at K=64 (the design's 384 B = ids + logprobs, plus the lse,
the input id and the position, which alignment needs).

CACHE LAYOUT:  <out-root>/<teacher>/<corpus_sha>/
    cache_config.json    the locked semantics (K, V, dtype, kernel path, store rule,
                         head precision, how the corpus tensors were read, rows per
                         shard, corpus identity, and for VL caches the processor
                         directory, the sha256 of every processor/tokenizer/template
                         file in it, the chat-template kwargs and the video decoder).
                         A resume with different semantics is refused.
    shard-NNNNN/         one directory per `--rows-per-shard` consecutive rows, written
                         to a temp name and renamed into place only when complete and
                         finite -- so resume = skip every shard directory that exists
                         and verifies.
    sessions/*.json      one per writer session: teacher digest before and after.  A
                         session hard-killed after its "before" digest (SIGKILL, the
                         OOM killer) has no "after"; the seal treats it as INTERRUPTED,
                         not as a teacher change, when a later complete session's
                         digests vouch for the same tree.  SIGTERM is turned into
                         SystemExit(143) so a plain `kill` still writes the "after".
    progress.json        rewritten after every shard (poll this).
    manifest.json        written only when every shard exists: the teacher's integrity
                         check before and after (a full directory digest), the corpus
                         sha, the shard read order, a finite check per shard.

FAILURES ARE FUNCTIONAL, NUMBERS ARE NOT.  A non-finite lse or logprob, an id outside
[0, V), a teacher whose digest moved between sessions, a shard whose bytes no longer
match its record: each one stops the run with a non-zero exit and a record.  The
parity arms print their logprob distances as diagnostics; what they gate on is
structural: identical top-k ids at every stored position.

HOST-RESIDENT TEACHER AND SHARED TEACHER LOADS (1.0.5).  `--host-resident pageable` (default)
reads every text-layer tensor once into owned host RAM (~46 GiB at 27B, inside the 56 GB
one-Qwen-streamer budget), so a pass no longer depends on the page cache, which the other
lanes' 55 GB children and the 1.56 TB K3 copy evict (one cold 27B pass measured 48.97 tok/s).
`--corpus` may be repeated: every corpus is its own cache (config, lock, sessions, seal), but
one process builds the teacher once and takes one full digest before its first read and one
after its last, written into every session it opened.

BATCHING (1.0.4).  Inside one shard the rows are planned into micro-batches by length
(longest first, ties in row order) under `--micro-batch` rows and `--max-batch-tokens`
padded tokens, so a shard of mixed lengths (self-gen rows, VL rows of 64..16k visual
tokens) is not padded to its longest row.  The outputs are put back in row order before
the shard is written, so the shard layout does not depend on the plan.  Both parity arms
use the same plan.

VL ROWS.  Either the native form {"row_id", "messages": [...]} or a record of the corpus
lane's `vision_in.jsonl` / `video_in.jsonl` (schema neural-forge-omni-distill-corpus-v1/
vision_in: "instruction", "media" [{media, stored_path, sha256, sampled_frames, ...}],
"reference_response", "uid"), mapped by one fixed rule: user turn = the media in order,
then the instruction; assistant turn = the reference response.  Every media file must
digest to the sha256 the record names.  A video item is either "frames" (image paths,
"fps", optional "frames_indices") or "path" to a video file plus "num_frames": the file is
decoded with the ffmpeg CLI (version locked in cache_config) and sampled at
round(linspace(0, total-1, num_frames)) -- the parent processor's own rule -- and the
processor stamps timestamps from the source fps and those indices.

Subcommands
-----------
  build-text   text / long / self-gen rescoring rows from a safetensors corpus
               (`input_ids` [N, L] int, optional `lengths` or `real_lengths` [N],
               `store_mask` [N, L], `target_mask` [N, L]).
  build-vl     vision-in / video-in rows from a JSONL corpus, through T(i).
  smoke-vl     greedy answers from T(i) for a few VL rows (user turn only, no KV cache,
               the whole prefix re-run per token), printed for a human read.
  verify       re-hash every shard file against its record and re-run the checks.
  parity-text  streamed engine (this writer, end to end) vs a stock full load.
  parity-vl    T(i) (this writer, end to end) vs a stock full-load VL forward.

    PY=/home/cereal/model-surgery-lab/venv/bin/python        # 3090 / RTX 8000
    PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python  # P40 (sm61)
    T=/home/cereal/omni-campaign/distill/tools
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<uuid> $PY $T/qtext_teacher_cache.py \\
        build-text --model-dir /home/cereal/models/Qwen3.8-27B-Uncensored \\
        --teacher q38-27bu --corpus corpus.safetensors --micro-batch 8 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TOOL_NAME = "qtext_teacher_cache"
TOOL_VERSION = "1.0.5"
MANIFEST_SCHEMA = "omni-distill-teacher-cache-manifest-v1"
SHARD_SCHEMA = "omni-distill-teacher-cache-shard-v1"
CONFIG_SCHEMA = "omni-distill-teacher-cache-config-v1"
PARITY_SCHEMA = "omni-distill-teacher-cache-parity-v1"

HERE = Path(__file__).resolve().parent
#: The streamed engine lives one directory up, beside its own receipts.  It is
#: loaded by path and its sha256 is sealed in every manifest; an override exists
#: only so the tests can point at the same file from a staging copy.
ENGINE_PATH = Path(os.environ.get("QTEXT_TPS_PROBE", str(HERE.parent / "tps_probe.py")))

#: Qwen3.5 / Qwen3.8 physical text vocabulary.  Every text loss in stage D is over
#: rows [0, 248,320) (design section 4, "Common to every training stage").
TEXT_VOCAB_ROWS = 248_320
DEFAULT_TOP_K = 64
SHARD_FILES = ("ids.u32", "pos.u32", "topk_ids.i32", "topk_logprobs.bf16", "lse.f32")
#: Largest card occupancy a launch accepts on its target (workflow GPU rule).
CARD_BUSY_MIB = 2000
#: Directory names under a teacher directory that are not the teacher: the HF
#: download cache keeps lock files that change on every read.
DIGEST_SKIP_DIRS = (".cache", ".git", "__pycache__")
#: Every file `AutoProcessor.from_pretrained` can read the processor, tokenizer or
#: chat template from.  Their sha256 is locked into a VL cache's config (finding L1):
#: the same model directory with an edited template or tokenizer is a different cache.
PROCESSOR_FILE_NAMES = (
    "added_tokens.json", "chat_template.jinja", "chat_template.json", "merges.txt",
    "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)
HEAD_PRECISIONS = ("fp32", "model")
BATCH_ORDERS = ("length", "corpus")


class CacheError(RuntimeError):
    """A refusal that is this tool's own, never a bare traceback."""


class NonFiniteShard(CacheError):
    """A shard whose teacher output is not finite: a functional failure."""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# The engine module.  Imported by path, never copied.
# --------------------------------------------------------------------------

_ENGINE: Any = None


def engine_module() -> Any:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if not ENGINE_PATH.is_file():
        raise CacheError(f"the streamed engine {ENGINE_PATH} is missing")
    spec = importlib.util.spec_from_file_location("tps_probe", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("tps_probe", module)
    spec.loader.exec_module(module)
    _ENGINE = module
    return module


def engine_identity() -> dict[str, Any]:
    return {"path": str(ENGINE_PATH), "sha256": file_sha256(ENGINE_PATH)}


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve())}


# --------------------------------------------------------------------------
# Kernel path for the Gated-DeltaNet layers.
# --------------------------------------------------------------------------


def select_linear_attention_kernel(mode: str) -> dict[str, Any]:
    """Pin the GDN kernel path BEFORE any layer is constructed.

    `Qwen3_5GatedDeltaNet.__init__` binds `chunk_gated_delta_rule or
    torch_chunk_gated_delta_rule` from module globals at construction time, so the
    path is decided by what those globals hold when the layer is built.  `auto` keeps
    whatever transformers found (fla on the server); `torch` clears the fla symbols so
    every layer built afterwards -- streamed buffers and a stock full load alike --
    runs the pure-torch reference.  The result is recorded, because the kernel path
    changes the low bits and is part of a cache's locked semantics.
    """

    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if mode == "torch":
        mq.chunk_gated_delta_rule = None
        mq.fused_recurrent_gated_delta_rule = None
        mq.FusedRMSNormGated = None
        mq.causal_conv1d_fn = None
        mq.causal_conv1d_update = None
    elif mode != "auto":
        raise CacheError(f"--linear-attn-kernel must be auto or torch, not {mode!r}")
    return {
        "requested": mode,
        "chunk_gated_delta_rule": "fla" if mq.chunk_gated_delta_rule is not None else "torch",
        "rms_norm_gated": "fla" if mq.FusedRMSNormGated is not None else "torch",
        "causal_conv1d": "causal_conv1d" if mq.causal_conv1d_fn is not None else "torch",
    }


# --------------------------------------------------------------------------
# Teacher integrity: a full directory digest before and after.
# --------------------------------------------------------------------------


def teacher_files(model_dir: Path) -> list[Path]:
    files = []
    for root, dirs, names in os.walk(model_dir):
        dirs[:] = sorted(d for d in dirs if d not in DIGEST_SKIP_DIRS)
        for name in sorted(names):
            path = Path(root) / name
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(model_dir).as_posix())


def teacher_digest(model_dir: Path, mode: str = "full", threads: int = 4) -> dict[str, Any]:
    """`full`: sha256 of every file + a tree digest over (relpath, bytes, sha256).
    `stat`: (relpath, bytes, mtime_ns) only -- for fixtures, never for a sealed cache.
    """

    model_dir = Path(model_dir).resolve()
    files = teacher_files(model_dir)
    started = time.perf_counter()
    if mode == "full":
        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
            shas = list(pool.map(file_sha256, files))
        entries = [
            {"path": f.relative_to(model_dir).as_posix(), "bytes": f.stat().st_size, "sha256": s}
            for f, s in zip(files, shas)
        ]
    elif mode == "stat":
        entries = [
            {"path": f.relative_to(model_dir).as_posix(), "bytes": f.stat().st_size,
             "mtime_ns": f.stat().st_mtime_ns}
            for f in files
        ]
    else:
        raise CacheError(f"unknown digest mode {mode!r}")
    return {
        "mode": mode,
        "model_dir": str(model_dir),
        "file_count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "tree_sha256": canonical_sha256(entries),
        "skipped_dirs": list(DIGEST_SKIP_DIRS),
        "seconds": round(time.perf_counter() - started, 2),
        "files": entries,
    }


def digest_summary(digest: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in digest.items() if k != "files"}


# --------------------------------------------------------------------------
# Corpora.
# --------------------------------------------------------------------------


#: `--lengths-key auto` reads the first of these that the corpus carries; with none,
#: every row is its full width.  The corpus lane's sets use `real_lengths`, the
#: self-gen export uses `lengths`.
AUTO_LENGTHS_KEYS = ("lengths", "real_lengths")


def store_mask_from_target_mask(target: Any, lengths: Any) -> Any:
    """target_mask[r, t] = "token t of row r is a target" -> store_mask[r, t] = "store
    position t" = target_mask[r, t+1].  Refused: a target at column 0 (no position
    predicts the first token) and a target at or past the row's length."""

    import torch

    rows, width = target.shape
    if bool(target[:, 0].any()):
        raise CacheError(f"target_mask marks column 0 in {int(target[:, 0].sum())} rows: no position predicts "
                         "the first token of a row (is this an input-indexed store_mask under the wrong name?)")
    beyond = target & ~(torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1))
    if bool(beyond.any()):
        raise CacheError(f"target_mask marks {int(beyond.sum())} positions at or past their row's length")
    store = torch.zeros((rows, width), dtype=torch.bool)
    store[:, :-1] = target[:, 1:]
    return store


def load_text_corpus(path: Path, ids_key: str = "input_ids", lengths_key: str = "auto") -> dict[str, Any]:
    """`input_ids` [N, L] (int32/int64), optional lengths [N] (`--lengths-key`; `auto` =
    `lengths`, else `real_lengths`, else full rows), `store_mask` [N, L] (input-indexed),
    `target_mask` [N, L] (target-indexed, shifted here; see the module docstring).

    The corpus sha is the sha256 of the file's bytes: the cache directory is
    addressed by content, so two corpora with one differing id can never share it.
    """

    import torch
    from safetensors.torch import load_file

    path = Path(path).resolve()
    if not path.is_file():
        raise CacheError(f"corpus {path} does not exist")
    tensors = load_file(str(path))
    if ids_key not in tensors:
        raise CacheError(f"corpus {path} has no {ids_key!r} tensor (keys {sorted(tensors)})")
    ids = tensors[ids_key]
    if ids.dim() != 2:
        raise CacheError(f"{ids_key} must be [rows, length], got {tuple(ids.shape)}")
    if ids.dtype not in (torch.int32, torch.int64):
        raise CacheError(f"{ids_key} must be int32 or int64, got {ids.dtype}")
    ids = ids.to(torch.int64)
    rows, length = ids.shape
    if lengths_key == "auto":
        used_lengths_key = next((k for k in AUTO_LENGTHS_KEYS if k in tensors), None)
    else:
        if lengths_key not in tensors:
            raise CacheError(f"corpus {path} has no {lengths_key!r} tensor (keys {sorted(tensors)})")
        used_lengths_key = lengths_key
    if used_lengths_key is None:
        lengths = torch.full((rows,), length, dtype=torch.int64)
    else:
        lengths = tensors[used_lengths_key]
    lengths = lengths.to(torch.int64).reshape(-1)
    if lengths.numel() != rows or int(lengths.min()) < 1 or int(lengths.max()) > length:
        raise CacheError(f"lengths must be [rows] in [1, {length}]")
    masks = {}
    for name in ("store_mask", "target_mask"):
        mask = tensors.get(name)
        if mask is not None:
            if tuple(mask.shape) != (rows, length):
                raise CacheError(f"{name} must be {(rows, length)}, got {tuple(mask.shape)}")
            masks[name] = mask.to(torch.bool)
    store_mask = masks.get("store_mask")
    mask_source = "none (every position below the row length)"
    if "target_mask" in masks:
        shifted = store_mask_from_target_mask(masks["target_mask"], lengths)
        if store_mask is not None:
            valid_cols = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
            disagree = int(((shifted != store_mask) & valid_cols).sum())
            if disagree:
                raise CacheError(f"store_mask and target_mask (shifted by one) disagree at {disagree} positions: "
                                 "the producer's shift is not the tool's; refusing to guess")
            mask_source = "store_mask, checked equal to target_mask shifted by one"
        else:
            mask_source = "target_mask shifted by one inside the tool (stored t iff target t+1)"
        store_mask = shifted
    elif store_mask is not None:
        mask_source = "store_mask (input-indexed, as given)"
    valid = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
    if int(ids[valid].min()) < 0:
        raise CacheError("negative token id in the corpus")
    return {
        "kind": "text",
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "ids_key": ids_key,
        "lengths_key": used_lengths_key,
        "mask_source": mask_source,
        "ids": ids,
        "lengths": lengths,
        "store_mask": store_mask,
        "rows": rows,
        "row_length": length,
        "tokens": int(lengths.sum()),
        "stored_positions_planned": int((valid if store_mask is None else (valid & store_mask)).sum()),
        "row_ids": [str(i) for i in range(rows)],
    }


def corpus_record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    """One record of the corpus lane's vision_in.jsonl / video_in.jsonl -> the native
    row.  The rule is fixed (and so covered by the JSONL sha): user turn = every media
    item in order, then the instruction; assistant turn = the reference response."""

    content: list[dict[str, Any]] = []
    for media in record["media"]:
        target = media.get("stored_path") or media.get("path")
        if not target:
            raise CacheError(f"record {record.get('uid')}: a media item has no path")
        if media.get("media") == "image":
            content.append({"type": "image", "path": target, "sha256": media.get("sha256")})
        elif media.get("media") == "video":
            frames = int(media.get("sampled_frames") or 0)
            if frames < 1:
                raise CacheError(f"record {record.get('uid')}: video item without sampled_frames")
            content.append({"type": "video", "path": target, "num_frames": frames, "sha256": media.get("sha256")})
        else:
            raise CacheError(f"record {record.get('uid')}: media kind {media.get('media')!r}")
    content.append({"type": "text", "text": str(record["instruction"])})
    messages = [{"role": "user", "content": content}]
    if record.get("reference_response"):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": str(record["reference_response"])}]})
    row_id = record.get("uid") or record.get("record_id") or record.get("rid")
    return {"row_id": str(row_id), "messages": messages, "source_format": "corpus-lane vision_in record"}


def media_targets(item: dict[str, Any]) -> list[str]:
    kind = item.get("type")
    if kind == "image":
        return [item["path"]]
    if kind == "video":
        return list(item["frames"]) if "frames" in item else [item["path"]]
    return []


def load_vl_corpus(path: Path) -> dict[str, Any]:
    """JSONL, one row per line: {"row_id": str, "messages": [chat messages]}, or a
    corpus-lane vision_in record (mapped by `corpus_record_to_row`).

    A content item is {"type": "text", "text": ...}, {"type": "image", "path": ...},
    {"type": "video", "frames": [paths...], "fps": float, "frames_indices"?: [...]} or
    {"type": "video", "path": video file, "num_frames": n}.  Relative media paths
    resolve against the JSONL's directory; an item that names a "sha256" must digest
    to it.  The corpus identity covers the JSONL bytes AND every media file's sha256:
    the same JSONL over a changed image is a different corpus.
    """

    path = Path(path).resolve()
    if not path.is_file():
        raise CacheError(f"corpus {path} does not exist")
    rows = []
    media: dict[str, str] = {}
    formats: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" not in row:
            if "instruction" in row and "media" in row:
                row = corpus_record_to_row(row)
            else:
                raise CacheError(f"{path}:{line_number} has no messages (and is not a corpus-lane vision_in record)")
        formats[row.get("source_format", "native")] = formats.get(row.get("source_format", "native"), 0) + 1
        row.setdefault("row_id", str(len(rows)))
        for message in row["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                for target in media_targets(item):
                    resolved = (path.parent / target).resolve() if not Path(target).is_absolute() else Path(target)
                    if not resolved.is_file():
                        raise CacheError(f"{path}:{line_number} media {target} does not exist")
                    if target not in media:
                        media[target] = file_sha256(resolved)
                    if item.get("sha256") and len(media_targets(item)) == 1 and media[target] != item["sha256"]:
                        raise CacheError(f"{path}:{line_number} media {target} digests to {media[target][:16]}, "
                                         f"the record says {str(item['sha256'])[:16]}")
        rows.append(row)
    if not rows:
        raise CacheError(f"corpus {path} has no rows")
    jsonl_sha = file_sha256(path)
    identity = {"jsonl_sha256": jsonl_sha, "media": sorted(media.items())}
    return {
        "kind": "vl",
        "path": str(path),
        "sha256": canonical_sha256(identity),
        "jsonl_sha256": jsonl_sha,
        "media_files": len(media),
        "row_formats": formats,
        "bytes": path.stat().st_size,
        "rows_data": rows,
        "rows": len(rows),
        "row_ids": [str(r["row_id"]) for r in rows],
    }


# --------------------------------------------------------------------------
# Engines: streamed text, streamed VL (T(i)), and the stock full load for parity.
# --------------------------------------------------------------------------


def resolve_dtype(name: str) -> Any:
    return engine_module().resolve_dtype(name)


class Teacher:
    """What the writer needs from any engine: hidden states and a head weight."""

    def __init__(self, kind: str, model_dir: Path, config: Any, head_weight: Any,
                 text_trunk: Any = None, vl_model: Any = None, facts: dict | None = None,
                 reader: Any = None, owner: Any = None) -> None:
        self.kind = kind
        self.model_dir = model_dir
        self.config = config
        self.head_weight = head_weight
        self.text_trunk = text_trunk
        self.vl_model = vl_model
        self.facts = facts or {}
        self.reader = reader
        self.owner = owner

    def text_hidden(self, ids: Any) -> Any:
        return self.text_trunk(input_ids=ids, use_cache=False).last_hidden_state

    def vl_hidden(self, batch: dict[str, Any]) -> Any:
        if self.vl_model is None:
            raise CacheError(f"the {self.kind} engine has no vision tower")
        self.vl_model.rope_deltas = None
        kwargs = {k: v for k, v in batch.items() if k in (
            "input_ids", "pixel_values", "image_grid_thw", "pixel_values_videos",
            "video_grid_thw", "mm_token_type_ids")}
        return self.vl_model(**kwargs, use_cache=False).last_hidden_state


def cast_parameters(module: Any, dtype: Any) -> None:
    """Cast parameters only.  Non-persistent fp32 buffers (rotary `inv_freq`) stay
    fp32, which is what `from_pretrained(dtype=...)` leaves them as."""

    for parameter in module.parameters():
        parameter.data = parameter.data.to(dtype)


def build_streamed_teacher(model_dir: Path, device: str, dtype: Any, attn: str,
                           with_vision: bool, vocab_rows: int = TEXT_VOCAB_ROWS,
                           head_precision: str = "model") -> Teacher:
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    tps = engine_module()
    trunk, lm_head, extra = tps.build_streamed_text_model(model_dir, device, dtype, attn)
    trunk.config.use_cache = False
    head = prepare_head(lm_head.weight, vocab_rows, head_precision)
    del lm_head
    facts = dict(extra["facts"])
    facts["head"] = {"precision": head_precision, "rows": int(head.shape[0]), "dtype": str(head.dtype),
                     "bytes_gb": round(head.numel() * head.element_size() / 1024**3, 4)}
    config = Qwen3_5Config.from_pretrained(str(model_dir))
    vl_model = None
    if with_vision:
        reader = extra["reader"]
        prefix = "model.visual."
        names = reader.names_under(prefix)
        if not names:
            raise CacheError(f"{model_dir} has no {prefix}* tensors: not a vision-language checkpoint")
        config.vision_config._attn_implementation = attn
        config.text_config._attn_implementation = attn
        with torch.device("meta"):
            vl_model = Qwen3_5Model(config)
        from transformers import AutoModel

        # Built on the CPU, then moved.  The rotary `inv_freq` buffer is computed at
        # construction; computed on a CUDA device it differs in the last bit from the
        # CPU value `from_pretrained` leaves (9.3e-10 on the 0.8B,
        # qtext-vl-divergence-diag-08b-fp32-det-p40.json), which moved the vision
        # output by 9e-5 and one top-64 order in the first parity-vl arm (v1.0.0).
        visual = AutoModel.from_config(config.vision_config).to(device)
        state = {name[len(prefix):]: reader.get(name) for name in names}
        visual.load_state_dict(state, strict=True, assign=False)
        del state
        cast_parameters(visual, dtype)
        visual.eval()
        vl_model.visual = visual
        vl_model.language_model = trunk
        vl_model.eval()
        facts["vision_tower_parameters"] = sum(p.numel() for p in visual.parameters())
        facts["vision_tower_class"] = type(visual).__name__
        facts["vision_buffers"] = {n: str(b.dtype) for n, b in visual.named_buffers()}
    return Teacher("streamed", Path(model_dir), config, head, text_trunk=trunk,
                   vl_model=vl_model, facts=facts, reader=extra["reader"])


def build_full_load_teacher(model_dir: Path, device: str, dtype: Any, attn: str,
                            vocab_rows: int = TEXT_VOCAB_ROWS, head_precision: str = "model") -> Teacher:
    tps = engine_module()
    model, facts = tps.load_resident_model(Path(model_dir), dtype, device, attn, None, None)
    model.config.text_config.use_cache = False
    trunk, head = tps.text_trunk_and_head(model)
    return Teacher("full-load", Path(model_dir), model.config, prepare_head(head.weight, vocab_rows, head_precision),
                   text_trunk=trunk, vl_model=model.model, facts=facts, owner=model)


# --------------------------------------------------------------------------
# Scoring: the head over stored positions, rows [0, V).
# --------------------------------------------------------------------------


def prepare_head(weight: Any, vocab_rows: int, precision: str) -> Any:
    """The head the writer scores with: rows [0, vocab_rows) of the lm_head weight,
    as an exact fp32 upcast (`fp32`) or as stored (`model`).  The caller drops its
    own reference to the full head, so at fp32 the card holds 4 B x V x H instead of
    2 B x (all rows) x H."""

    import torch

    if precision not in HEAD_PRECISIONS:
        raise CacheError(f"--head-precision must be one of {HEAD_PRECISIONS}, not {precision!r}")
    if weight.shape[0] < vocab_rows:
        raise CacheError(f"head has {weight.shape[0]} rows, fewer than --vocab-rows {vocab_rows}")
    sliced = weight.detach()[:vocab_rows]
    if precision == "fp32":
        return sliced.to(torch.float32).contiguous()
    return sliced


def configure_head_matmul(precision: str) -> dict[str, Any]:
    """At fp32 the head GEMM must be IEEE fp32: TF32 (a 10-bit mantissa) off."""

    import torch

    if precision == "fp32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    return {"head_precision": precision,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}


def score_hidden(hidden_rows: Any, head_weight: Any, vocab_rows: int, top_k: int,
                 chunk_rows: int) -> tuple[Any, Any, Any]:
    """hidden_rows [P, H] -> (topk_ids int32 [P,K], topk_logprobs bf16 [P,K], lse fp32 [P]), on CPU.

    The head GEMM runs in the head weight's dtype (`prepare_head`: fp32, or the model
    dtype as the stock `lm_head` does), then the logits are in fp32 -- the stock
    causal-LM path's `logits.float()` -- before logsumexp and top-k.  Rows
    [vocab_rows, ...) of a wider head never enter either.
    """

    import torch
    import torch.nn.functional as F

    if head_weight.shape[0] < vocab_rows:
        raise CacheError(f"head has {head_weight.shape[0]} rows, fewer than --vocab-rows {vocab_rows}")
    weight = head_weight[:vocab_rows]
    count = hidden_rows.shape[0]
    ids_out = torch.empty((count, top_k), dtype=torch.int32)
    lp_out = torch.empty((count, top_k), dtype=torch.bfloat16)
    lse_out = torch.empty((count,), dtype=torch.float32)
    for start in range(0, count, max(1, int(chunk_rows))):
        stop = min(count, start + int(chunk_rows))
        logits = F.linear(hidden_rows[start:stop].to(weight.dtype), weight).float()
        lse = torch.logsumexp(logits, dim=-1)
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        ids_out[start:stop] = indices.to(torch.int32).cpu()
        lp_out[start:stop] = (values - lse.unsqueeze(-1)).to(torch.bfloat16).cpu()
        lse_out[start:stop] = lse.cpu()
        del logits, lse, values, indices
    return ids_out, lp_out, lse_out


def stored_mask_for(lengths: Any, width: int, store_mask: Any = None, mm_types: Any = None) -> Any:
    import torch

    mask = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
    if store_mask is not None:
        mask &= store_mask[:, :width]
    if mm_types is not None:
        mask &= mm_types[:, :width] == 0
    return mask


# --------------------------------------------------------------------------
# Batches.
# --------------------------------------------------------------------------


def plan_batches(lengths: list[int], micro_batch: int, max_batch_tokens: int = 0,
                 order: str = "corpus") -> list[list[int]]:
    """Indices into `lengths` grouped into micro-batches.  `corpus`: consecutive.
    `length`: longest first (ties in index order).  A batch closes at `micro_batch`
    rows, or when one more row would take rows x width past `max_batch_tokens`
    (0 = no token cap); a single row longer than the cap runs alone."""

    if order not in BATCH_ORDERS:
        raise CacheError(f"--batch-order must be one of {BATCH_ORDERS}, not {order!r}")
    index = list(range(len(lengths)))
    if order == "length":
        index.sort(key=lambda i: (-int(lengths[i]), i))
    limit = max(1, int(micro_batch))
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for i in index:
        grown = max(width, int(lengths[i]))
        if current and (len(current) >= limit or (max_batch_tokens and (len(current) + 1) * grown > max_batch_tokens)):
            batches.append(current)
            current, grown = [], int(lengths[i])
        current.append(i)
        width = grown
    if current:
        batches.append(current)
    return batches


def text_batches(corpus: dict[str, Any], row_start: int, row_stop: int, micro_batch: int,
                 max_batch_tokens: int = 0, order: str = "corpus"):
    """Rows of [row_start, row_stop) in `plan_batches` groups, each right-trimmed to
    its longest row.  The pad value never matters: the model is causal and padded
    positions are never stored."""

    ids = corpus["ids"]
    lengths = corpus["lengths"]
    store = corpus["store_mask"]
    plan = plan_batches(lengths[row_start:row_stop].tolist(), micro_batch, max_batch_tokens, order)
    for group in plan:
        rows = [row_start + i for i in group]
        index = torch_index(rows)
        lens = lengths.index_select(0, index)
        width = int(lens.max())
        yield {
            "rows": rows,
            "input_ids": ids.index_select(0, index)[:, :width].clone(),
            "lengths": lens.clone(),
            "store_mask": None if store is None else store.index_select(0, index)[:, :width].clone(),
            "mm_token_type_ids": None,
        }


def torch_index(rows: list[int]) -> Any:
    import torch

    return torch.tensor(rows, dtype=torch.int64)


def load_processor(processor_dir: Path, max_pixels: int | None) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(processor_dir))
    if max_pixels:
        size = dict(processor.image_processor.size)
        size["longest_edge"] = int(max_pixels)
        processor.image_processor.size = size
    return processor


def processor_files(processor_dir: Path) -> dict[str, str]:
    """sha256 of every processor / tokenizer / chat-template file the directory holds."""

    processor_dir = Path(processor_dir).resolve()
    found = {name: file_sha256(processor_dir / name) for name in PROCESSOR_FILE_NAMES
             if (processor_dir / name).is_file()}
    if not found:
        raise CacheError(f"{processor_dir} holds none of the processor files {PROCESSOR_FILE_NAMES}")
    return found


_FFMPEG_VERSION: str | None = None


def ffmpeg_version() -> str:
    global _FFMPEG_VERSION
    if _FFMPEG_VERSION is None:
        try:
            out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=60)
            _FFMPEG_VERSION = out.stdout.splitlines()[0].strip() if out.returncode == 0 and out.stdout else "unavailable"
        except (OSError, subprocess.SubprocessError):
            _FFMPEG_VERSION = "unavailable"
    return _FFMPEG_VERSION


def processor_identity(processor: Any, processor_dir: Path | None = None,
                       template_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    image = getattr(processor, "image_processor", None)
    video = getattr(processor, "video_processor", None)

    def as_dict(value: Any) -> Any:
        try:
            return {k: v for k, v in dict(value).items() if v is not None}
        except Exception:  # noqa: BLE001 -- recorded as text when not a mapping
            return str(value)

    identity = {
        "class": type(processor).__name__,
        "image_processor": None if image is None else type(image).__name__,
        "image_size": None if image is None else as_dict(getattr(image, "size", None)),
        "video_processor": None if video is None else type(video).__name__,
        "video_size": None if video is None else as_dict(getattr(video, "size", None)),
    }
    if processor_dir is not None:
        identity["dir"] = str(Path(processor_dir).resolve())
        identity["files_sha256"] = processor_files(Path(processor_dir))
        identity["chat_template_kwargs"] = dict(template_kwargs or {})
        identity["video_decoder"] = {"tool": "ffmpeg CLI rawvideo rgb24, -threads 1", "version": ffmpeg_version(),
                                     "sampling": "round(linspace(0, total-1, num_frames))"}
    return identity


def decode_video(path: Path, num_frames: int) -> tuple[list[Any], dict[str, Any]]:
    """Decode a video file to RGB frames with the ffmpeg CLI and sample `num_frames`
    of them by the parent processor's rule, round(linspace(0, total-1, n))."""

    import numpy as np
    from PIL import Image

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        raise CacheError(f"ffprobe failed on {path}: {probe.stderr.strip()[:300]}")
    stream = (json.loads(probe.stdout).get("streams") or [None])[0]
    if not stream:
        raise CacheError(f"{path} has no video stream")
    width, height = int(stream["width"]), int(stream["height"])

    def rate(text: str) -> float:
        num, _, den = str(text).partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0

    fps = rate(stream.get("avg_frame_rate", "0/1")) or rate(stream.get("r_frame_rate", "0/1"))
    if fps <= 0:
        raise CacheError(f"{path}: no usable frame rate in {stream}")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=600)
    if raw.returncode != 0:
        raise CacheError(f"ffmpeg failed on {path}: {raw.stderr.decode(errors='replace').strip()[:300]}")
    frame_bytes = width * height * 3
    if not raw.stdout or len(raw.stdout) % frame_bytes:
        raise CacheError(f"{path}: decoded {len(raw.stdout)} B, not a whole number of {width}x{height} rgb24 frames")
    total = len(raw.stdout) // frame_bytes
    count = min(int(num_frames), total)
    indices = [int(i) for i in np.linspace(0, total - 1, count).round().astype(int)]
    frames = []
    for i in indices:
        array = np.frombuffer(raw.stdout, dtype=np.uint8, count=frame_bytes, offset=i * frame_bytes)
        frames.append(Image.fromarray(array.reshape(height, width, 3).copy(), "RGB"))
    return frames, {"fps": fps, "total_num_frames": total, "frames_indices": indices,
                    "width": width, "height": height}


def encode_vl_row(processor: Any, row: dict[str, Any], base_dir: Path,
                  template_kwargs: dict[str, Any] | None = None,
                  generation_prompt: bool = False) -> dict[str, Any]:
    """One JSONL row -> the processor's own tensors, through the parent's chat template.

    The template is rendered with media placeholders, then the processor expands the
    placeholders against the decoded media -- the documented processor call, so the
    image/video token counts and the M-RoPE token types are the parent's own.
    `generation_prompt` (smoke-vl) drops every trailing assistant turn and asks the
    template for the assistant header instead.
    """

    import torch
    from PIL import Image

    images: list[Any] = []
    videos: list[Any] = []
    metadata: list[dict[str, Any]] = []
    template_messages = []
    messages = list(row["messages"])
    if generation_prompt:
        while messages and messages[-1].get("role") == "assistant":
            messages.pop()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            template_messages.append(message)
            continue
        items = []
        for item in content:
            kind = item.get("type")
            if kind == "image":
                target = Path(item["path"])
                target = target if target.is_absolute() else base_dir / target
                with Image.open(target) as handle:
                    images.append(handle.convert("RGB"))
                items.append({"type": "image"})
            elif kind == "video":
                if "frames" in item:
                    frames = []
                    for frame in item["frames"]:
                        target = Path(frame)
                        target = target if target.is_absolute() else base_dir / target
                        with Image.open(target) as handle:
                            frames.append(handle.convert("RGB"))
                    fps = float(item.get("fps", 2.0))
                    indices = [int(i) for i in item.get("frames_indices", range(len(frames)))]
                    if len(indices) != len(frames):
                        raise CacheError(f"row {row.get('row_id')}: {len(indices)} frames_indices for {len(frames)} frames")
                    total = int(item.get("total_num_frames", max(indices) + 1))
                    facts = {"fps": fps, "total_num_frames": total, "frames_indices": indices}
                else:
                    target = Path(item["path"])
                    target = target if target.is_absolute() else base_dir / target
                    frames, decoded = decode_video(target, int(item["num_frames"]))
                    facts = {k: decoded[k] for k in ("fps", "total_num_frames", "frames_indices")}
                videos.append(frames)
                metadata.append(facts)
                items.append({"type": "video"})
            else:
                items.append({k: v for k, v in item.items() if k != "sha256"})
        template_messages.append({**message, "content": items})
    text = processor.apply_chat_template(template_messages, tokenize=False,
                                         add_generation_prompt=generation_prompt, **dict(template_kwargs or {}))
    kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "return_mm_token_type_ids": True}
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
        kwargs["video_metadata"] = metadata
        kwargs["do_sample_frames"] = False
    encoded = processor(**kwargs)
    out = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    if "mm_token_type_ids" not in out:
        raise CacheError("the processor returned no mm_token_type_ids; M-RoPE cannot be built")
    return out


def collate_vl(encoded_rows: list[dict[str, Any]], rows: list[int], pad_id: int) -> dict[str, Any]:
    """Right-pad ids and token types; concatenate media in row order.

    No attention mask: every pad sits after every real token, the model is causal, and
    `get_rope_index` without a mask numbers the pads as trailing text, so no real
    position sees a pad.  `masked_scatter` fills placeholder slots row-major, which is
    exactly the order the media tensors are concatenated in.
    """

    import torch

    lengths = torch.tensor([int(e["input_ids"].shape[1]) for e in encoded_rows], dtype=torch.int64)
    width = int(lengths.max())
    ids = torch.full((len(encoded_rows), width), int(pad_id), dtype=torch.int64)
    types = torch.zeros((len(encoded_rows), width), dtype=torch.int64)
    for index, encoded in enumerate(encoded_rows):
        n = int(encoded["input_ids"].shape[1])
        ids[index, :n] = encoded["input_ids"][0]
        types[index, :n] = encoded["mm_token_type_ids"][0]
    batch: dict[str, Any] = {"rows": rows, "input_ids": ids, "mm_token_type_ids": types,
                             "lengths": lengths, "store_mask": None}
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        parts = [e[key] for e in encoded_rows if key in e]
        if parts:
            batch[key] = torch.cat(parts, dim=0)
    return batch


def vl_batches(corpus: dict[str, Any], processor: Any, row_start: int, row_stop: int,
               micro_batch: int, pad_id: int, max_batch_tokens: int = 0, order: str = "corpus",
               template_kwargs: dict[str, Any] | None = None):
    """Every row of [row_start, row_stop) is encoded first (its token count decides the
    plan), then collated in `plan_batches` groups."""

    base = Path(corpus["path"]).parent
    data = corpus["rows_data"]
    encoded = [encode_vl_row(processor, data[i], base, template_kwargs) for i in range(row_start, row_stop)]
    lengths = [int(e["input_ids"].shape[1]) for e in encoded]
    for group in plan_batches(lengths, micro_batch, max_batch_tokens, order):
        yield collate_vl([encoded[i] for i in group], [row_start + i for i in group], pad_id)


# --------------------------------------------------------------------------
# Running one batch through a teacher.
# --------------------------------------------------------------------------


def run_batch(teacher: Teacher, batch: dict[str, Any], device: str, vocab_rows: int,
              top_k: int, head_chunk_rows: int) -> dict[str, Any]:
    """Forward one batch and score its stored positions.  Returns CPU arrays in
    row-major stored order plus the per-row counts."""

    import torch

    ids = batch["input_ids"]
    width = int(ids.shape[1])
    mask = stored_mask_for(batch["lengths"], width, batch.get("store_mask"), batch.get("mm_token_type_ids"))
    with torch.no_grad():
        if batch.get("mm_token_type_ids") is not None:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
                     if k not in ("rows", "lengths", "store_mask")}
            hidden = teacher.vl_hidden(moved)
        else:
            hidden = teacher.text_hidden(ids.to(device))
        flat = hidden.reshape(-1, hidden.shape[-1])
        index = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
        selected = flat.index_select(0, index.to(flat.device))
        topk_ids, topk_lp, lse = score_hidden(selected, teacher.head_weight, vocab_rows, top_k, head_chunk_rows)
        del hidden, flat, selected
    positions = torch.arange(width).unsqueeze(0).expand(ids.shape[0], width)
    return {
        "rows": list(batch["rows"]),
        "lengths": batch["lengths"].tolist(),
        "ids": ids[mask].to(torch.int64),
        "pos": positions[mask].to(torch.int64),
        "topk_ids": topk_ids,
        "topk_logprobs": topk_lp,
        "lse": lse,
        "counts": mask.sum(dim=1).tolist(),
        "forward_tokens": int(batch["lengths"].sum()),
        "padded_tokens": int(ids.shape[0]) * width,
    }


ROW_ARRAY_KEYS = ("ids", "pos", "topk_ids", "topk_logprobs", "lse")


class RowAssembler:
    """Collects batch outputs (in plan order) and hands them back in ROW order, so a
    shard's layout does not depend on how its rows were batched."""

    def __init__(self, keys: tuple[str, ...] = ROW_ARRAY_KEYS) -> None:
        self.keys = keys
        self.rows: dict[int, dict[str, Any]] = {}

    def add(self, out: dict[str, Any]) -> None:
        offset = 0
        for row, count, length in zip(out["rows"], out["counts"], out["lengths"]):
            count = int(count)
            if row in self.rows:
                raise CacheError(f"row {row} was planned into two batches")
            self.rows[row] = {"count": count, "length": int(length),
                              **{k: out[k][offset : offset + count] for k in self.keys}}
            offset += count

    def assemble(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import torch

        order = sorted(self.rows)
        arrays = {k: torch.cat([self.rows[r][k] for r in order], dim=0) if order else torch.empty(0)
                  for k in self.keys}
        table, offset = [], 0
        for r in order:
            table.append({"row": r, "offset": offset, "count": self.rows[r]["count"], "length": self.rows[r]["length"]})
            offset += self.rows[r]["count"]
        return arrays, table


# --------------------------------------------------------------------------
# Shards: write, read, check.
# --------------------------------------------------------------------------


def shard_name(index: int) -> str:
    return f"shard-{index:05d}"


def array_bytes(name: str, tensor: Any) -> bytes:
    import numpy as np
    import torch

    if name in ("ids.u32", "pos.u32"):
        values = tensor.to(torch.int64)
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= 2**32):
            raise CacheError(f"{name} value outside uint32")
        return values.numpy().astype("<u4").tobytes()
    if name == "topk_ids.i32":
        return tensor.to(torch.int32).contiguous().numpy().astype("<i4").tobytes()
    if name == "topk_logprobs.bf16":
        return tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
    if name == "lse.f32":
        return tensor.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    raise CacheError(f"unknown shard file {name}")


def check_arrays(arrays: dict[str, Any], vocab_rows: int, top_k: int) -> dict[str, Any]:
    """Structural + finite checks over one shard's arrays.  Every count here is a
    functional fact: a non-zero one refuses the shard."""

    import torch

    lse = arrays["lse"]
    lp = arrays["topk_logprobs"].float()
    ids = arrays["topk_ids"].to(torch.int64)
    positions = int(lse.numel())
    result = {
        "positions": positions,
        "nonfinite_lse": int((~torch.isfinite(lse)).sum()),
        "nonfinite_logprobs": int((~torch.isfinite(lp)).sum()),
        "topk_ids_out_of_range": int(((ids < 0) | (ids >= vocab_rows)).sum()),
        "shape_ok": bool(tuple(lp.shape) == (positions, top_k) and tuple(ids.shape) == (positions, top_k)
                         and arrays["ids"].numel() == positions and arrays["pos"].numel() == positions),
        # bf16 rounding of a top-1 logprob near 0 can land a hair above 0; anything
        # past 1e-2 is not rounding.
        "positive_logprobs": int((lp > 1e-2).sum()),
        "unsorted_rows": int((lp[:, 1:] > lp[:, :-1] + 1e-2).any(dim=1).sum()) if positions and top_k > 1 else 0,
        "duplicate_id_rows": int(
            ((ids.sort(dim=1).values[:, 1:] == ids.sort(dim=1).values[:, :-1]).any(dim=1)).sum()
        ) if positions and top_k > 1 else 0,
    }
    if positions:
        result["lse_min"] = float(lse[torch.isfinite(lse)].min()) if result["nonfinite_lse"] < positions else None
        result["lse_max"] = float(lse[torch.isfinite(lse)].max()) if result["nonfinite_lse"] < positions else None
        top1 = lp[:, 0]
        result["top1_logprob_mean"] = float(top1[torch.isfinite(top1)].mean()) if positions else None
        mass = torch.exp(lp).sum(dim=1)
        result["topk_mass_mean"] = float(mass[torch.isfinite(mass)].mean())
    result["finite"] = result["nonfinite_lse"] == 0 and result["nonfinite_logprobs"] == 0
    result["structural_ok"] = (result["shape_ok"] and result["topk_ids_out_of_range"] == 0
                               and result["positive_logprobs"] == 0 and result["unsorted_rows"] == 0
                               and result["duplicate_id_rows"] == 0)
    return result


def write_shard(cache_dir: Path, index: int, arrays: dict[str, Any], record: dict[str, Any],
                vocab_rows: int, top_k: int) -> dict[str, Any]:
    checks = check_arrays(arrays, vocab_rows, top_k)
    if not checks["finite"] or not checks["structural_ok"]:
        failure = {"shard": shard_name(index), "checks": checks, "record": record, "at_utc": utc_now()}
        write_json_atomic(cache_dir / "failures" / f"{shard_name(index)}.json", failure)
        raise NonFiniteShard(f"{shard_name(index)} refused: {json.dumps(checks, sort_keys=True)}")
    final = cache_dir / shard_name(index)
    staging = cache_dir / f".{shard_name(index)}.tmp-{os.getpid()}"
    if staging.exists():
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
    staging.mkdir(parents=True)
    files: dict[str, Any] = {}
    key_for = {"ids.u32": "ids", "pos.u32": "pos", "topk_ids.i32": "topk_ids",
               "topk_logprobs.bf16": "topk_logprobs", "lse.f32": "lse"}
    for name in SHARD_FILES:
        payload = array_bytes(name, arrays[key_for[name]])
        with open(staging / name, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    shard = {
        "schema": SHARD_SCHEMA,
        "shard": shard_name(index),
        "index": index,
        "top_k": top_k,
        "vocab_rows": vocab_rows,
        "positions": checks["positions"],
        "files": files,
        "checks": checks,
        **record,
    }
    write_json_atomic(staging / "shard.json", shard)
    fsync_directory(staging)
    if final.exists():
        raise CacheError(f"{final} appeared while this writer held the lock")
    os.replace(staging, final)
    fsync_directory(cache_dir)
    return shard


def read_shard(shard_dir: Path) -> dict[str, Any]:
    """The reader every consumer uses: arrays as torch tensors + the shard record."""

    import numpy as np
    import torch

    record = json.loads((Path(shard_dir) / "shard.json").read_text(encoding="utf-8"))
    k = int(record["top_k"])
    p = int(record["positions"])

    def load(name: str, dtype: str) -> Any:
        return np.fromfile(str(Path(shard_dir) / name), dtype=dtype)

    return {
        "record": record,
        "ids": torch.from_numpy(load("ids.u32", "<u4").astype(np.int64)),
        "pos": torch.from_numpy(load("pos.u32", "<u4").astype(np.int64)),
        "topk_ids": torch.from_numpy(load("topk_ids.i32", "<i4").reshape(p, k).copy()),
        "topk_logprobs": torch.from_numpy(load("topk_logprobs.bf16", "<i2").reshape(p, k).copy()).view(torch.bfloat16),
        "lse": torch.from_numpy(load("lse.f32", "<f4").copy()),
    }


def verify_shard(shard_dir: Path) -> dict[str, Any]:
    record = json.loads((Path(shard_dir) / "shard.json").read_text(encoding="utf-8"))
    problems = []
    for name in SHARD_FILES:
        path = Path(shard_dir) / name
        expected = record["files"][name]
        if not path.is_file():
            problems.append(f"{name} missing")
            continue
        if path.stat().st_size != expected["bytes"]:
            problems.append(f"{name} size {path.stat().st_size} != {expected['bytes']}")
            continue
        if file_sha256(path) != expected["sha256"]:
            problems.append(f"{name} sha256 mismatch")
    if not problems:
        arrays = read_shard(shard_dir)
        checks = check_arrays(arrays, int(record["vocab_rows"]), int(record["top_k"]))
        if not checks["finite"] or not checks["structural_ok"]:
            problems.append(f"checks failed {checks}")
        if checks["positions"] != record["positions"]:
            problems.append("position count differs from record")
    return {"shard": record["shard"], "ok": not problems, "problems": problems}


# --------------------------------------------------------------------------
# Locks, sessions, the device rule.
# --------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class WriterLock:
    def __init__(self, cache_dir: Path) -> None:
        self.path = cache_dir / "writer.lock"
        self.taken_over: dict[str, Any] | None = None

    def acquire(self) -> None:
        payload = json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "at_utc": utc_now()})
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                held = json.loads(self.path.read_text(encoding="utf-8") or "{}")
                if held.get("host") == socket.gethostname() and not pid_alive(int(held.get("pid", -1))):
                    self.taken_over = held
                    self.path.unlink()
                    continue
                raise CacheError(f"{self.path} is held by {held}; another writer owns this cache")
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
            return
        raise CacheError(f"could not take {self.path}")

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


STAGING_GLOB = ".shard-*.tmp-*"


def remove_stale_staging(cache_dir: Path) -> list[str]:
    """Call only while holding the writer lock: every staging directory is then a dead
    writer's half-written shard (a hard kill between mkdir and rename), up to one shard
    of bytes each, never published and never read."""

    removed = []
    for path in sorted(cache_dir.glob(STAGING_GLOB)):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


def raise_on_sigterm(signum: int, _frame: Any) -> None:
    # Python's default SIGTERM action ends the process without running `finally`, so a
    # plain `kill` would leave a session with no "after" digest and a stale allocator
    # receipt.  As SystemExit the session's `finally` runs and records the stop.
    raise SystemExit(128 + signum)


def install_sigterm_handler() -> Any:
    try:
        return signal.signal(signal.SIGTERM, raise_on_sigterm)
    except ValueError:  # not the main thread (an embedding caller): keep the default
        return None


def restore_sigterm_handler(previous: Any) -> None:
    if previous is None:
        return
    try:
        signal.signal(signal.SIGTERM, previous)
    except ValueError:
        pass


def failure_exit_code(exc: BaseException) -> int:
    """The process exit status a session's exception leads to, for its records."""

    if isinstance(exc, NonFiniteShard):
        return 2
    if isinstance(exc, SystemExit):
        return exc.code if isinstance(exc.code, int) and exc.code else 1
    if isinstance(exc, KeyboardInterrupt):
        return 130
    return 1


def visible_card_check(limit_mib: int = CARD_BUSY_MIB) -> list[dict[str, Any]]:
    """Refuse to start on a visible card that already holds more than `limit_mib`."""

    import torch

    if not torch.cuda.is_available():
        return []
    tps = engine_module()
    inventory = tps.nvidia_smi_inventory()
    used_by_uuid = {}
    for line in inventory.get("gpus", []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            used_by_uuid[parts[2]] = int(parts[3].split()[0])
    out = []
    for record in tps.device_records():
        used = used_by_uuid.get(record["uuid"])
        out.append({**record, "memory_used_mib_at_start": used})
        if used is not None and used > limit_mib:
            raise CacheError(f"card {record['uuid']} already holds {used} MiB (> {limit_mib}); refusing to start")
    return out


def allocator_receipt(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    existing = {}
    if Path(path).is_file():
        existing = json.loads(Path(path).read_text(encoding="utf-8"))
    existing.update(payload)
    write_json_atomic(Path(path), existing)


def runtime_record(device: str) -> dict[str, Any]:
    tps = engine_module()
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.executable,
        "platform": platform.platform(),
        "versions": tps.package_versions(),
        "devices": tps.device_records(),
        "requested_device": device,
        "exact_command": tps.exact_command(),
    }


# --------------------------------------------------------------------------
# The writer.
# --------------------------------------------------------------------------


def cache_config(args: argparse.Namespace, corpus: dict[str, Any], kernel: dict[str, Any],
                 processor_id: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "teacher": args.teacher,
        "model_dir": str(Path(args.model_dir).resolve()),
        "corpus_kind": corpus["kind"],
        "corpus_sha256": corpus["sha256"],
        "corpus_rows": corpus["rows"],
        "top_k": int(args.top_k),
        "vocab_rows": int(args.vocab_rows),
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "linear_attention_kernel": kernel,
        "rows_per_shard": int(args.rows_per_shard),
        "store_rule": (
            "position t of row r is stored iff t < length[r], store_mask[r,t] (text corpora, "
            "default all; a target_mask is shifted so store_mask[r,t] = target_mask[r,t+1]), and "
            "mm_token_type_ids[r,t] == 0 (VL rows); it carries the teacher distribution over the token at t+1"
        ),
        "corpus_read": None if corpus["kind"] != "text" else {
            "ids_key": corpus["ids_key"], "lengths_key": corpus["lengths_key"], "mask_source": corpus["mask_source"]},
        "head_rows": f"[0, {int(args.vocab_rows)})",
        "head_precision": args.head_precision,
        "logit_precision": (
            "lm_head rows [0, V) held as an exact fp32 upcast, hidden state upcast, IEEE fp32 GEMM (TF32 off); "
            "logsumexp/top-k in fp32; stored logprobs rounded to bf16 once"
            if args.head_precision == "fp32" else
            "lm_head GEMM in the model dtype (logits rounded to the model dtype), upcast to fp32 before "
            "logsumexp/top-k; stored logprobs rounded to bf16"),
        "processor": processor_id,
        "tool_major": TOOL_VERSION.split(".")[0],
    }


def compare_config(existing: dict[str, Any], wanted: dict[str, Any]) -> list[str]:
    return [key for key in sorted(set(existing) | set(wanted)) if existing.get(key) != wanted.get(key)]


def existing_shard_ok(path: Path, verify_hashes: bool) -> bool:
    record_path = path / "shard.json"
    if not record_path.is_file():
        return False
    if verify_hashes:
        result = verify_shard(path)
        if not result["ok"]:
            raise CacheError(f"existing {path.name} does not verify: {result['problems']}; "
                             "move it aside by hand after reading why")
        return True
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for name in SHARD_FILES:
        if not (path / name).is_file() or (path / name).stat().st_size != record["files"][name]["bytes"]:
            raise CacheError(f"existing {path.name}/{name} is missing or the wrong size")
    return True


#: Options added in 1.0.4, read through `option` so a Namespace built by an older
#: caller (or a test) keeps the 1.0.3 behaviour except where a default changed.
OPTION_DEFAULTS = {"head_precision": "fp32", "lengths_key": "auto", "max_batch_tokens": 0,
                   "batch_order": "length", "chat_template_kwargs": None, "max_new_shards": 0,
                   "host_resident": "pageable"}


def option(args: argparse.Namespace, name: str) -> Any:
    return getattr(args, name, OPTION_DEFAULTS[name])


def template_kwargs_of(args: argparse.Namespace) -> dict[str, Any]:
    raw = option(args, "chat_template_kwargs")
    if raw in (None, ""):
        return {}
    value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(value, dict):
        raise CacheError("--chat-template-kwargs must be a JSON object")
    return value


class HostResidentReader:
    """The engine `ShardReader` surface (get / has / names_under / weight_map / total_size /
    model_dir / bytes_read) over HOST-RAM copies of every tensor under `prefixes` (1.0.5).

    The streamed engine reads each decoder layer from the safetensors mmap on every pass,
    so its rate is the page cache's: on a host where other lanes stream 55 GB children and
    the 1.56 TB K3 copy, the 27B teacher's pages are evicted between passes and a pass
    becomes a 50 GB disk read (48.97 tok/s measured, j0 head probe, 2026-10-13).  Here
    every text-layer tensor is read ONCE into owned (anonymous) host memory -- the same
    bytes, `clone()`d off the mmap so eviction can no longer touch them -- and the
    engine's `_materialise` copies from there.  Names outside `prefixes` (embed, head,
    vision tower: read once at build) go to the wrapped reader unchanged."""

    def __init__(self, inner: Any, prefixes: list[str]) -> None:
        from safetensors import safe_open

        self.inner = inner
        self.model_dir = inner.model_dir
        self.weight_map = inner.weight_map
        self.total_size = inner.total_size
        self.device = getattr(inner, "device", None)
        self.bytes_read = 0
        names = sorted({n for prefix in prefixes for n in inner.names_under(prefix)})
        by_file: dict[str, list[str]] = {}
        for name in names:
            by_file.setdefault(self.weight_map[name], []).append(name)
        self.store: dict[str, Any] = {}
        started = time.perf_counter()
        for filename in sorted(by_file):
            with safe_open(str(Path(self.model_dir) / filename), framework="pt", device="cpu") as handle:
                for name in by_file[filename]:
                    self.store[name] = handle.get_tensor(name).clone()
        self.load_seconds = round(time.perf_counter() - started, 2)
        self.resident_bytes = sum(t.numel() * t.element_size() for t in self.store.values())
        self.resident_tensors = len(self.store)

    def get(self, name: str) -> Any:
        tensor = self.store.get(name)
        if tensor is None:
            tensor = self.inner.get(name)
            return tensor
        self.bytes_read += tensor.numel() * tensor.element_size()
        return tensor

    def has(self, name: str) -> bool:
        return self.inner.has(name)

    def names_under(self, prefix: str) -> list[str]:
        return self.inner.names_under(prefix)

    def facts(self) -> dict[str, Any]:
        return {"mode": "pageable", "tensors": self.resident_tensors,
                "bytes_gb": round(self.resident_bytes / 1024**3, 3), "load_seconds": self.load_seconds}


def make_teacher_resident(teacher: Teacher, mode: str) -> dict[str, Any] | None:
    """Swap the streamed trunk's layer reader for a `HostResidentReader` (mode `pageable`)."""

    if mode == "off" or teacher.text_trunk is None or not hasattr(teacher.text_trunk, "layers"):
        return None
    if mode != "pageable":
        raise CacheError(f"--host-resident must be pageable or off, not {mode!r}")
    tps = engine_module()
    layers = teacher.text_trunk.layers
    log(f"host-resident teacher: reading every tensor under {layers.prefix}. into host RAM once")
    resident = HostResidentReader(layers.reader, [f"{tps.TEXT_TRUNK_PREFIX}.layers."])
    layers.reader = resident
    teacher.reader = resident
    facts = resident.facts()
    log(f"host-resident teacher: {facts['tensors']} tensors, {facts['bytes_gb']} GiB in {facts['load_seconds']} s")
    return facts


def build_cache(args: argparse.Namespace, corpus_kind: str) -> int:
    """One or more corpora (`--corpus` repeated) through ONE teacher in ONE process.

    Each corpus is still its own content-addressed cache with its own locked config,
    writer lock, session record and seal.  What the corpora of one process share is the
    expensive part: the teacher is built (and made host-resident) once, and its full
    directory digest is taken once before the first read and once after the last, and
    both are written into every session of the process -- the "before" precedes every
    read of every cache and the "after" follows them all, so each session carries a
    true before/after pair.  A process killed mid-way leaves "before"-only sessions in
    every cache it opened; those are INTERRUPTED and vouched for by the next complete
    session, exactly as for one corpus."""

    import torch

    started_wall = time.perf_counter()
    allocator = Path(args.allocator_receipt) if args.allocator_receipt else None
    kernel = select_linear_attention_kernel(args.linear_attn_kernel)
    args.head_precision = option(args, "head_precision")
    head_matmul = configure_head_matmul(args.head_precision)
    template_kwargs = template_kwargs_of(args)
    corpus_paths = args.corpus if isinstance(args.corpus, (list, tuple)) else [args.corpus]
    if not corpus_paths:
        raise CacheError("no --corpus given")
    processor = None
    processor_id = None
    if corpus_kind != "text":
        processor_dir = Path(args.processor_dir or args.model_dir)
        processor = load_processor(processor_dir, args.max_pixels)
        processor_id = processor_identity(processor, processor_dir, template_kwargs)
    model_dir = Path(args.model_dir).resolve()
    embedding_rows = int(json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
                         .get("text_config", {}).get("vocab_size", 0))
    per = int(args.rows_per_shard)

    # Every corpus is loaded and its config checked BEFORE anything is locked: a refusal
    # for corpus k leaves corpora 0..k-1 untouched.
    entries: list[dict[str, Any]] = []
    sealed_codes: list[int] = []
    seen_dirs: set[str] = set()
    for path in corpus_paths:
        if corpus_kind == "text":
            corpus = load_text_corpus(Path(path), args.ids_key, option(args, "lengths_key"))
            if embedding_rows:
                valid = torch.arange(corpus["row_length"]).unsqueeze(0) < corpus["lengths"].unsqueeze(1)
                if int(corpus["ids"][valid].max()) >= embedding_rows:
                    raise CacheError(f"corpus {path}: id {int(corpus['ids'][valid].max())} >= the teacher's "
                                     f"{embedding_rows} embedding rows")
        else:
            corpus = load_vl_corpus(Path(path))
        cache_dir = Path(args.out_root) / args.teacher / corpus["sha256"]
        if str(cache_dir) in seen_dirs:
            raise CacheError(f"corpus {path} addresses {cache_dir} twice in one call")
        seen_dirs.add(str(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        wanted = cache_config(args, corpus, kernel, processor_id)
        config_path = cache_dir / "cache_config.json"
        if config_path.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            changed = compare_config(existing, wanted)
            if changed:
                raise CacheError(f"{config_path} locks different semantics for {changed}; refusing to mix shards")
        else:
            write_json_atomic(config_path, wanted)
        if (cache_dir / "manifest.json").is_file():
            log(f"{cache_dir / 'manifest.json'} already sealed; verifying instead of writing")
            sealed_codes.append(verify_cache(cache_dir))
            continue
        rows = corpus["rows"]
        entries.append({"corpus": corpus, "cache_dir": cache_dir, "rows": rows,
                        "total_shards": (rows + per - 1) // per})
    if not entries:
        return max(sealed_codes) if sealed_codes else 0

    # The card rule is checked BEFORE the locks and before any session record exists: a
    # refused start touches nothing in the caches (v1.0.1 wrote a session with no
    # "before" digest here, and the seal then read it as a teacher change).
    devices = visible_card_check() if str(args.device).startswith("cuda") else []
    previous_sigterm = install_sigterm_handler()
    locks: list[WriterLock] = []
    try:
        for entry in entries:
            lock = WriterLock(entry["cache_dir"])
            lock.acquire()
            locks.append(lock)
            entry["lock"] = lock
    except BaseException:
        for lock in locks:
            lock.release()
        restore_sigterm_handler(previous_sigterm)
        raise
    session_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    caches_in_process = [str(e["cache_dir"]) for e in entries]
    for entry in entries:
        entry["session_path"] = entry["cache_dir"] / "sessions" / f"session-{session_id}.json"
        entry["session"] = {"session": session_id, "started_utc": utc_now(), "tool": tool_identity(),
                            "engine": engine_identity(), "lock_taken_over_from": entry["lock"].taken_over,
                            "caches_in_process": caches_in_process}
    exit_code = 0
    failure: str | None = None
    process_facts: dict[str, Any] = {}
    try:
        for entry in entries:
            entry["session"]["devices_at_start"] = devices
            entry["session"]["stale_staging_removed"] = remove_stale_staging(entry["cache_dir"])
        allocator_receipt(allocator, {
            "name": Path(allocator).stem if allocator else None, "owner": "claude", "server": "ai",
            "kind": "host-venv", "pid": os.getpid(), "python": sys.executable,
            "gpu_uuids": [d["uuid"] for d in devices], "purpose": args.purpose,
            "tool": f"{TOOL_NAME} {TOOL_VERSION}", "started_utc": utc_now(), "state": "running",
            "cache_dir": caches_in_process[0] if len(caches_in_process) == 1 else caches_in_process,
        })
        log(f"teacher digest before ({args.teacher_digest}) of {model_dir}")
        before = teacher_digest(model_dir, args.teacher_digest, args.digest_threads)
        for entry in entries:
            entry["session"]["teacher_digest_before"] = before
            write_json_atomic(entry["session_path"], entry["session"])
        log(f"teacher tree {before['tree_sha256'][:16]} ({before['file_count']} files, "
            f"{before['total_bytes']:,} B, {before['seconds']} s)")

        for entry in entries:
            cache_dir = entry["cache_dir"]
            entry["pending"] = [i for i in range(entry["total_shards"])
                                if not existing_shard_ok(cache_dir / shard_name(i), args.verify_existing)]
            log(f"cache {cache_dir}: {entry['total_shards'] - len(entry['pending'])}/{entry['total_shards']} "
                "shards already present")
        max_new = int(option(args, "max_new_shards") or 0)
        budget = max_new if max_new else None
        for entry in entries:
            if budget is not None:
                if len(entry["pending"]) > budget:
                    entry["session"]["partial_by_request"] = {"max_new_shards": max_new,
                                                              "pending_at_start": len(entry["pending"])}
                    entry["pending"] = entry["pending"][:budget]
                budget -= len(entry["pending"])
            entry["session"]["head_matmul"] = head_matmul
            entry["session"]["batching"] = {"micro_batch": int(args.micro_batch),
                                            "max_batch_tokens": int(option(args, "max_batch_tokens")),
                                            "order": option(args, "batch_order")}
        if max_new:
            log(f"--max-new-shards {max_new}: this process writes at most {max_new} shards and leaves the rest "
                "unsealed (a later session resumes)")
        if any(entry["pending"] for entry in entries):
            dtype = resolve_dtype(args.dtype)
            tps = engine_module()
            sampler = tps.ResourceSampler(interval_seconds=2.0)
            sampler.start()
            build_started = time.perf_counter()
            teacher = build_streamed_teacher(model_dir, args.device, dtype, args.attn_implementation,
                                             with_vision=corpus_kind == "vl", vocab_rows=int(args.vocab_rows),
                                             head_precision=args.head_precision)
            resident = make_teacher_resident(teacher, option(args, "host_resident"))
            process_facts["teacher_build_seconds"] = round(time.perf_counter() - build_started, 2)
            process_facts["host_resident"] = resident
            pad_id = 0
            if processor is not None and getattr(processor.tokenizer, "pad_token_id", None) is not None:
                pad_id = int(processor.tokenizer.pad_token_id)
            for entry in entries:
                entry["session"]["engine_facts"] = teacher.facts
                entry["session"]["host_resident"] = resident
                entry["session"]["teacher_build_seconds"] = process_facts["teacher_build_seconds"]
            for entry in entries:
                entry["current"] = True
                write_cache_shards(args, corpus_kind, entry, teacher, processor, pad_id, template_kwargs, session_id)
                entry["current"] = False
            sampler.stop()
            process_facts["resources"] = sampler.report()
            process_facts["torch_memory"] = tps.torch_memory_report()
            if teacher.reader is not None:
                process_facts["bytes_read_gb"] = round(teacher.reader.bytes_read / 1024**3, 3)
            del teacher
    except NonFiniteShard as exc:
        failure = str(exc)
        exit_code = 2
        log(f"FUNCTIONAL FAILURE: {exc}")
    except BaseException as exc:
        # CUDA OOM, a refusal, SIGTERM (as SystemExit 143), ^C: every session record and the
        # allocator receipt say what stopped it (v1.0.2 left both reading "exit 0"), and
        # the exception still propagates to the caller.
        failure = repr(exc)
        exit_code = failure_exit_code(exc)
        log(f"session stopped by {exc!r}; recording exit {exit_code}")
        raise
    finally:
        try:
            log("teacher digest after")
            after = teacher_digest(model_dir, args.teacher_digest, args.digest_threads)
            for entry in entries:
                session = entry["session"]
                if failure is not None:
                    session["failure"] = failure if entry.get("current") else f"process stopped: {failure}"
                session.update(process_facts)
                session["teacher_digest_after"] = after
                session["teacher_unchanged_in_session"] = (
                    session.get("teacher_digest_before", {}).get("tree_sha256") == after["tree_sha256"])
                session["finished_utc"] = utc_now()
                session["runtime"] = runtime_record(args.device)
                write_json_atomic(entry["session_path"], session)
        finally:
            for lock in locks:
                lock.release()
            allocator_receipt(allocator, {"ended_utc": utc_now(), "state": "released (process exited)",
                                          "outcome": f"exit {exit_code}", "failure": failure})
            restore_sigterm_handler(previous_sigterm)
    if not all(entry["session"].get("teacher_unchanged_in_session") for entry in entries):
        log("FUNCTIONAL FAILURE: the teacher directory digest changed during this session")
        return 2
    if exit_code:
        return exit_code
    codes = list(sealed_codes)
    for entry in entries:
        cache_dir = entry["cache_dir"]
        missing = sum(1 for i in range(entry["total_shards"]) if not (cache_dir / shard_name(i) / "shard.json").is_file())
        if entry["session"].get("partial_by_request") or (max_new and missing):
            log(f"PARTIAL by request: {cache_dir}: {missing} shards still pending; not sealed (exit 0)")
            codes.append(0)
            continue
        codes.append(seal_manifest(cache_dir, entry["corpus"], model_dir, time.perf_counter() - started_wall))
    return max(codes) if codes else 0


def write_cache_shards(args: argparse.Namespace, corpus_kind: str, entry: dict[str, Any], teacher: Teacher,
                       processor: Any, pad_id: int, template_kwargs: dict[str, Any], session_id: str) -> None:
    """Every pending shard of one cache, through the shared teacher."""

    import torch

    tps = engine_module()
    corpus = entry["corpus"]
    cache_dir = entry["cache_dir"]
    rows = entry["rows"]
    per = int(args.rows_per_shard)
    pending = entry["pending"]
    session = entry["session"]
    session_tokens = 0
    session_positions = 0
    session_compute = 0.0
    for done, index in enumerate(pending, 1):
        shard_start = time.perf_counter()
        row_start, row_stop = index * per, min(rows, (index + 1) * per)
        assembler = RowAssembler()
        forward_tokens = 0
        padded_tokens = 0
        batch_count = 0
        plan = (int(args.micro_batch), int(option(args, "max_batch_tokens")), option(args, "batch_order"))
        batches = (text_batches(corpus, row_start, row_stop, *plan)
                   if corpus_kind == "text" else
                   vl_batches(corpus, processor, row_start, row_stop, plan[0], pad_id, plan[1], plan[2],
                              template_kwargs))
        for batch in batches:
            out = run_batch(teacher, batch, args.device, int(args.vocab_rows), int(args.top_k),
                            int(args.head_chunk_rows))
            assembler.add(out)
            forward_tokens += out["forward_tokens"]
            padded_tokens += out["padded_tokens"]
            batch_count += 1
        arrays, table = assembler.assemble()
        for row_entry in table:
            row_entry["row_id"] = corpus["row_ids"][row_entry["row"]]
        seconds = time.perf_counter() - shard_start
        peak = tps.torch_memory_report().get(str(torch.cuda.current_device()) if torch.cuda.is_available() else "", {})
        record = {
            "rows": [row_start, row_stop],
            "row_table": table,
            "forward_tokens": forward_tokens,
            "padded_tokens": padded_tokens,
            "batches": batch_count,
            "micro_batch": int(args.micro_batch),
            "max_batch_tokens": plan[1],
            "batch_order": plan[2],
            "head_chunk_rows": int(args.head_chunk_rows),
            "device": args.device,
            "session": session_id,
            "seconds": round(seconds, 3),
            "forward_tokens_per_second": round(forward_tokens / max(seconds, 1e-9), 2),
            "torch_peak_allocated_gb_so_far": peak.get("peak_allocated_gb"),
            "written_utc": utc_now(),
        }
        shard = write_shard(cache_dir, index, arrays, record, int(args.vocab_rows), int(args.top_k))
        session_tokens += forward_tokens
        session_positions += shard["positions"]
        session_compute += seconds
        rate = session_tokens / max(session_compute, 1e-9)
        remaining = sum(min(rows, (i + 1) * per) - i * per for i in pending[done:])
        write_json_atomic(cache_dir / "progress.json", {
            "shards_total": entry["total_shards"],
            "shards_present": entry["total_shards"] - len(pending) + done,
            "session": session_id,
            "session_forward_tokens": session_tokens,
            "session_positions": session_positions,
            "session_forward_tokens_per_second": round(rate, 2),
            "rows_remaining": remaining,
            "updated_utc": utc_now(),
        })
        log(f"{shard['shard']}: rows [{row_start},{row_stop}) {shard['positions']} positions, "
            f"{forward_tokens} fwd tokens ({padded_tokens} padded, {batch_count} batches) in {seconds:.1f} s = "
            f"{record['forward_tokens_per_second']} tok/s; peak alloc {record['torch_peak_allocated_gb_so_far']} GB; "
            f"finite={shard['checks']['finite']} ({done}/{len(pending)}) [{cache_dir.name[:16]}]")
    session["session_forward_tokens"] = session_tokens
    session["session_positions"] = session_positions
    session["session_forward_tokens_per_second"] = round(session_tokens / max(session_compute, 1e-9), 2)


def seal_manifest(cache_dir: Path, corpus: dict[str, Any], model_dir: Path, wall: float) -> int:
    config = json.loads((cache_dir / "cache_config.json").read_text(encoding="utf-8"))
    per = int(config["rows_per_shard"])
    total_shards = (int(config["corpus_rows"]) + per - 1) // per
    missing = [shard_name(i) for i in range(total_shards) if not (cache_dir / shard_name(i) / "shard.json").is_file()]
    if missing:
        log(f"not sealing: {len(missing)} shards missing (first {missing[:3]})")
        return 1
    log(f"sealing: re-verifying {total_shards} shards byte for byte")
    shards = []
    for i in range(total_shards):
        result = verify_shard(cache_dir / shard_name(i))
        if not result["ok"]:
            log(f"FUNCTIONAL FAILURE: {result}")
            return 2
        record = json.loads((cache_dir / shard_name(i) / "shard.json").read_text(encoding="utf-8"))
        shards.append({
            "shard": record["shard"], "rows": record["rows"], "positions": record["positions"],
            "forward_tokens": record["forward_tokens"], "files": record["files"],
            "finite": record["checks"]["finite"], "structural_ok": record["checks"]["structural_ok"],
            "nonfinite_lse": record["checks"]["nonfinite_lse"],
            "nonfinite_logprobs": record["checks"]["nonfinite_logprobs"],
            "session": record["session"], "device": record["device"], "micro_batch": record["micro_batch"],
            "shard_json_sha256": file_sha256(cache_dir / shard_name(i) / "shard.json"),
        })
    sessions = []
    judged_paths = []
    aborted = []
    interrupted = []
    shards_by_session: dict[str, int] = {}
    for entry in shards:
        shards_by_session[entry["session"]] = shards_by_session.get(entry["session"], 0) + 1
    # Session file names are session-<UTC second>-<pid>.json, so sorted = started order.
    for path in sorted((cache_dir / "sessions").glob("session-*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        has_before = "teacher_digest_before" in s
        has_after = "teacher_digest_after" in s
        if not has_before and s["session"] not in shards_by_session:
            # Stopped before its first digest and wrote nothing: recorded, not judged.
            aborted.append({"session": s["session"], "failure": s.get("failure"),
                            "record_sha256": file_sha256(path)})
            continue
        if has_before and not has_after:
            # Hard-killed (SIGKILL, the OOM killer, a second SIGTERM during the "after"
            # digest) after its "before" digest: its own "after" never ran.  It is judged
            # below by the sessions around it, not as a teacher that changed.
            interrupted.append({
                "session": s["session"],
                "before": digest_summary(s["teacher_digest_before"]),
                "shards_written": shards_by_session.get(s["session"], 0),
                "failure": s.get("failure"),
                "record_sha256": file_sha256(path),
                "order": len(sessions) + len(interrupted) + len(aborted),
            })
            judged_paths.append(path)
            continue
        sessions.append({
            "session": s["session"],
            "before": digest_summary(s.get("teacher_digest_before", {})),
            "after": digest_summary(s.get("teacher_digest_after", {})),
            "unchanged": s.get("teacher_unchanged_in_session"),
            "forward_tokens_per_second": s.get("session_forward_tokens_per_second"),
            "devices": [d.get("uuid") for d in (s.get("runtime", {}) or {}).get("devices", [])],
            "failure": s.get("failure"),
            "record_sha256": file_sha256(path),
            "order": len(sessions) + len(interrupted) + len(aborted),
        })
        judged_paths.append(path)
    if not sessions:
        log("FUNCTIONAL FAILURE: no session carries both teacher digests")
        return 2
    trees = {x["before"].get("tree_sha256") for x in sessions} | {x["after"].get("tree_sha256") for x in sessions}
    teacher_ok = len(trees) == 1 and None not in trees and all(x["unchanged"] for x in sessions)
    if not teacher_ok:
        log(f"FUNCTIONAL FAILURE: teacher digests differ across complete sessions: {sorted(map(str, trees))}")
        return 2
    (tree,) = trees
    for x in interrupted:
        later = [c for c in sessions if c["order"] > x["order"]]
        if x["before"].get("tree_sha256") != tree:
            log(f"FUNCTIONAL FAILURE: interrupted session {x['session']} started on teacher tree "
                f"{x['before'].get('tree_sha256')}, the complete sessions saw {tree}")
            return 2
        if not later:
            log(f"not sealing: session {x['session']} was interrupted (no 'after' digest) and no later "
                "complete session vouches for the teacher; run one more build session")
            return 1
        # The next complete session's "before" equals this session's "before" and every
        # other digest: the tree this session read from was the tree on both sides of it.
        x["vouched_by"] = later[0]["session"]
    for x in sessions + interrupted:
        x.pop("order")
    first = json.loads(judged_paths[0].read_text(encoding="utf-8"))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "sealed_utc": utc_now(),
        "tool": tool_identity(),
        "engine": engine_identity(),
        "cache_config": config,
        "cache_config_sha256": file_sha256(cache_dir / "cache_config.json"),
        "teacher": {
            "name": config["teacher"],
            "model_dir": str(model_dir),
            "digest_mode": first["teacher_digest_before"]["mode"],
            "tree_sha256_before": first["teacher_digest_before"]["tree_sha256"],
            "tree_sha256_after": sessions[-1]["after"]["tree_sha256"],
            "equal_before_and_after_every_session": teacher_ok,
            "interrupted_sessions_vouched_for": len(interrupted),
            "files": first["teacher_digest_before"]["files"],
        },
        "corpus": {k: v for k, v in corpus.items() if k in (
            "kind", "path", "sha256", "bytes", "rows", "row_length", "tokens", "ids_key",
            "lengths_key", "mask_source", "stored_positions_planned", "jsonl_sha256", "media_files",
            "row_formats")},
        "shard_read_order": [s["shard"] for s in shards],
        "shards": shards,
        "positions_total": sum(s["positions"] for s in shards),
        "forward_tokens_total": sum(s["forward_tokens"] for s in shards),
        "bytes_total": sum(f["bytes"] for s in shards for f in s["files"].values()),
        "all_finite": all(s["finite"] for s in shards),
        "all_structural_ok": all(s["structural_ok"] for s in shards),
        "sessions": sessions,
        "sessions_interrupted": interrupted,
        "sessions_aborted_before_start": aborted,
        "seal_wall_seconds_this_session": round(wall, 2),
    }
    write_json_atomic(cache_dir / "manifest.json", manifest)
    log(f"SEALED {cache_dir}/manifest.json sha256 {file_sha256(cache_dir / 'manifest.json')} "
        f"positions={manifest['positions_total']} bytes={manifest['bytes_total']} all_finite={manifest['all_finite']}")
    return 0


def verify_cache(cache_dir: Path) -> int:
    cache_dir = Path(cache_dir)
    if not (cache_dir / "manifest.json").is_file():
        raise CacheError(f"{cache_dir} is not sealed (no manifest.json); nothing to verify")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    passed = failed = 0
    for entry in manifest["shards"]:
        result = verify_shard(cache_dir / entry["shard"])
        sha_ok = file_sha256(cache_dir / entry["shard"] / "shard.json") == entry["shard_json_sha256"]
        if result["ok"] and sha_ok:
            passed += 1
        else:
            failed += 1
            log(f"FAIL {entry['shard']}: {result['problems']} shard.json sha ok={sha_ok}")
    print(f"verify {cache_dir}: {passed} shards PASS, {failed} FAIL "
          f"(teacher equal before/after: {manifest['teacher']['equal_before_and_after_every_session']})", flush=True)
    return 0 if failed == 0 else 2


# --------------------------------------------------------------------------
# Parity arms.
# --------------------------------------------------------------------------


def load_cache_arrays(cache_dir: Path) -> dict[str, Any]:
    import torch

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    parts: dict[str, list[Any]] = {k: [] for k in ("ids", "pos", "topk_ids", "topk_logprobs", "lse")}
    table = []
    base = 0
    for name in manifest["shard_read_order"]:
        shard = read_shard(cache_dir / name)
        for key in parts:
            parts[key].append(shard[key])
        for row in shard["record"]["row_table"]:
            table.append({**row, "offset": row["offset"] + base})
        base += shard["record"]["positions"]
    return {"manifest": manifest, "row_table": table,
            **{k: torch.cat(v, dim=0) for k, v in parts.items()}}


def compare_arrays(cache: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Streamed cache (read back from disk) vs the full-load reference, same positions."""

    import torch

    same_positions = bool(torch.equal(cache["pos"], ref["pos"]) and torch.equal(cache["ids"], ref["ids"]))
    ids_equal = torch.equal(cache["topk_ids"], ref["topk_ids"].to(torch.int32))
    mismatch_rows = int((cache["topk_ids"] != ref["topk_ids"].to(torch.int32)).any(dim=1).sum())
    set_equal_rows = int(sum(
        1 for a, b in zip(cache["topk_ids"].tolist(), ref["topk_ids"].tolist()) if set(a) == set(b)))
    top1_equal = int((cache["topk_ids"][:, 0] == ref["topk_ids"][:, 0].to(torch.int32)).sum())
    lp_cache = cache["topk_logprobs"].float()
    lp_ref_fp32 = ref["topk_logprobs_fp32"]
    lp_ref_bf16 = lp_ref_fp32.to(torch.bfloat16).float()
    aligned = cache["topk_ids"] == ref["topk_ids"].to(torch.int32)
    diff_fp32 = (lp_cache - lp_ref_fp32).abs()[aligned]
    diff_bf16 = (lp_cache - lp_ref_bf16).abs()[aligned]
    lse_diff = (cache["lse"] - ref["lse"]).abs()
    return {
        "positions": int(cache["lse"].numel()),
        "same_positions_and_input_ids": same_positions,
        "topk_ids_equal_all_positions": bool(ids_equal),
        "topk_id_mismatch_positions": mismatch_rows,
        "topk_id_set_equal_positions": set_equal_rows,
        "top1_equal_positions": top1_equal,
        "logprob_max_abs_diff_vs_fp32_reference": float(diff_fp32.max()) if diff_fp32.numel() else None,
        "logprob_max_abs_diff_vs_reference_rounded_to_bf16": float(diff_bf16.max()) if diff_bf16.numel() else None,
        "logprob_bitwise_equal_to_reference_bf16": bool(diff_bf16.numel() and float(diff_bf16.max()) == 0.0),
        "lse_max_abs_diff": float(lse_diff.max()) if lse_diff.numel() else None,
    }


def reference_arrays(teacher: Teacher, batches: list[dict[str, Any]], device: str, vocab_rows: int,
                     top_k: int, chunk_rows: int) -> dict[str, Any]:
    """The full-load arm, scored the same way but kept at fp32 before the bf16 store."""

    import torch
    import torch.nn.functional as F

    keys = ("ids", "pos", "topk_ids", "topk_logprobs_fp32", "lse")
    assembler = RowAssembler(keys)
    weight = teacher.head_weight[:vocab_rows]
    with torch.no_grad():
        for batch in batches:
            parts: dict[str, list[Any]] = {k: [] for k in keys}
            ids = batch["input_ids"]
            width = int(ids.shape[1])
            mask = stored_mask_for(batch["lengths"], width, batch.get("store_mask"), batch.get("mm_token_type_ids"))
            if batch.get("mm_token_type_ids") is not None:
                moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
                         if k not in ("rows", "lengths", "store_mask")}
                hidden = teacher.vl_hidden(moved)
            else:
                hidden = teacher.text_hidden(ids.to(device))
            flat = hidden.reshape(-1, hidden.shape[-1])
            index = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1).to(flat.device)
            selected = flat.index_select(0, index)
            for start in range(0, selected.shape[0], chunk_rows):
                logits = F.linear(selected[start:start + chunk_rows].to(weight.dtype), weight).float()
                lse = torch.logsumexp(logits, dim=-1)
                values, indices = torch.topk(logits, k=top_k, dim=-1)
                parts["topk_ids"].append(indices.cpu())
                parts["topk_logprobs_fp32"].append((values - lse.unsqueeze(-1)).cpu())
                parts["lse"].append(lse.cpu())
            positions = torch.arange(width).unsqueeze(0).expand(ids.shape[0], width)
            parts["ids"].append(ids[mask])
            parts["pos"].append(positions[mask])
            out = {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in parts.items()}
            out.update({"rows": list(batch["rows"]), "counts": mask.sum(dim=1).tolist(),
                        "lengths": batch["lengths"].tolist()})
            assembler.add(out)
    arrays, _table = assembler.assemble()
    return arrays


def parity_common_args(args: argparse.Namespace, out_root: Path, corpus: Path, teacher_name: str) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.out_root = str(out_root)
    ns.corpus = str(corpus)
    ns.teacher = teacher_name
    ns.allocator_receipt = None
    ns.verify_existing = True
    return ns


def make_parity_text_corpus(path: Path, tokenizer_dir: Path, rows: int, length: int, seed: int) -> dict[str, Any]:
    """Real text rows (the parent's tokenizer over fixed prose) + random-id rows, with
    variable lengths so the right-padding path is exercised."""

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    prose = (
        "The streamed teacher reads one decoder layer at a time from the safetensors shards, "
        "runs it on the whole batch, and frees it before the next layer is read. "
        "def fold(rows, keep):\n    return sorted(rows, key=lambda r: -r.energy)[:keep]\n"
        "A cache that cannot prove what wrote it is not a cache; it is a rumour. "
    ) * 64
    real = tokenizer(prose, add_special_tokens=False)["input_ids"]
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 248_044, (rows, length), generator=generator, dtype=torch.int64)
    for r in range(0, rows, 2):
        take = real[r * 7 : r * 7 + length]
        ids[r, : len(take)] = torch.tensor(take, dtype=torch.int64)
    lengths = torch.tensor([length - (r * 37) % (length // 2) for r in range(rows)], dtype=torch.int64)
    store = torch.ones((rows, length), dtype=torch.uint8)
    store[rows - 1, : length // 4] = 0
    save_file({"input_ids": ids, "lengths": lengths, "store_mask": store}, str(path))
    return {"rows": rows, "length": length, "lengths": lengths.tolist()}


def run_parity(args: argparse.Namespace, kind: str) -> int:
    """Arm A: this writer, end to end (streamed engine -> shards on disk -> sealed manifest),
    read back through `read_shard`.  Arm B: a stock `from_pretrained` load of the same
    checkpoint, the SAME collated batches, scored at fp32.  Gate: identical top-k ids at
    every stored position.  Printed: logprob max |diff| and lse max |diff|."""

    import gc

    import torch

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype)
    kernel = select_linear_attention_kernel(args.linear_attn_kernel)
    args.head_precision = option(args, "head_precision")
    head_matmul = configure_head_matmul(args.head_precision)
    template_kwargs = template_kwargs_of(args)
    started = utc_now()
    if kind == "text":
        corpus_path = work / "parity_text_corpus.safetensors"
        corpus_facts = make_parity_text_corpus(corpus_path, Path(args.model_dir), args.parity_rows,
                                               args.parity_length, args.seed)
    else:
        corpus_path = Path(args.corpus).resolve() if args.corpus else make_parity_vl_corpus(work, args)
        corpus_facts = {"jsonl": str(corpus_path)}
    build_args = parity_common_args(args, work / "cache", corpus_path, f"parity-{kind}")
    log(f"arm A: {TOOL_NAME} build-{kind} (streamed engine) into {work / 'cache'}")
    code = build_cache(build_args, kind)
    if code != 0:
        log(f"arm A build exited {code}")
        return code
    corpus = (load_text_corpus(corpus_path, args.ids_key, option(args, "lengths_key")) if kind == "text"
              else load_vl_corpus(corpus_path))
    cache_dir = work / "cache" / build_args.teacher / corpus["sha256"]
    cache = load_cache_arrays(cache_dir)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log("arm B: stock full load of the same checkpoint")
    reference = build_full_load_teacher(Path(args.model_dir), args.device, dtype, args.attn_implementation,
                                        int(args.vocab_rows), args.head_precision)
    # The same batches arm A ran: planned shard by shard with the same plan arguments.
    per = int(args.rows_per_shard)
    plan = (int(args.micro_batch), int(option(args, "max_batch_tokens")), option(args, "batch_order"))
    batches = []
    if kind != "text":
        processor = load_processor(Path(args.processor_dir or args.model_dir), args.max_pixels)
        pad_id = int(processor.tokenizer.pad_token_id or 0)
    for start in range(0, corpus["rows"], per):
        stop = min(corpus["rows"], start + per)
        if kind == "text":
            batches.extend(text_batches(corpus, start, stop, *plan))
        else:
            batches.extend(vl_batches(corpus, processor, start, stop, plan[0], pad_id, plan[1], plan[2], template_kwargs))
    ref = reference_arrays(reference, batches, args.device, int(args.vocab_rows), int(args.top_k),
                           int(args.head_chunk_rows))
    comparison = compare_arrays(cache, ref)

    # Diagnostic only: each row alone, unpadded, through the full load.
    single = []
    for batch in batches:
        for i, row in enumerate(batch["rows"]):
            n = int(batch["lengths"][i])
            one = {"rows": [row], "input_ids": batch["input_ids"][i : i + 1, :n],
                   "lengths": batch["lengths"][i : i + 1],
                   "store_mask": None if batch.get("store_mask") is None else batch["store_mask"][i : i + 1, :n],
                   "mm_token_type_ids": None if batch.get("mm_token_type_ids") is None
                   else batch["mm_token_type_ids"][i : i + 1, :n]}
            if kind == "vl":
                encoded = encode_vl_row(processor, corpus["rows_data"][row], Path(corpus["path"]).parent, template_kwargs)
                one = collate_vl([encoded], [row], pad_id)
            single.append(one)
    ref_single = reference_arrays(reference, single, args.device, int(args.vocab_rows), int(args.top_k),
                                  int(args.head_chunk_rows))
    unpadded = compare_arrays(cache, ref_single)

    vision_buffers = None
    if kind == "vl":
        vision_buffers = {n: str(b.dtype) for n, b in reference.vl_model.visual.named_buffers()}
    passed = bool(comparison["same_positions_and_input_ids"] and comparison["topk_ids_equal_all_positions"])
    receipt = {
        "schema": PARITY_SCHEMA,
        "arm": "T(i) streamed VL teacher vs stock full-load VL forward" if kind == "vl"
        else "streamed text teacher vs stock full-load forward",
        "gate": "identical top-k ids (order included) at every stored position, same positions and input ids",
        "passed": passed,
        "comparison_batched_same_inputs": comparison,
        "diagnostic_vs_unpadded_single_row_full_load": unpadded,
        "corpus": corpus_facts,
        "corpus_sha256": corpus["sha256"],
        "cache_dir": str(cache_dir),
        "manifest_sha256": file_sha256(cache_dir / "manifest.json"),
        "model_dir": str(Path(args.model_dir).resolve()),
        "dtype": args.dtype,
        "linear_attention_kernel": kernel,
        "attention_implementation": args.attn_implementation,
        "head_matmul": head_matmul,
        "batching": {"micro_batch": plan[0], "max_batch_tokens": plan[1], "order": plan[2], "rows_per_shard": per},
        "micro_batch": int(args.micro_batch),
        "top_k": int(args.top_k),
        "vocab_rows": int(args.vocab_rows),
        "full_load_facts": {k: v for k, v in reference.facts.items() if k != "hf_device_map_devices"},
        "full_load_vision_buffers": vision_buffers,
        "tool": tool_identity(),
        "engine": engine_identity(),
        "runtime": runtime_record(args.device),
        "started_utc": started,
        "finished_utc": utc_now(),
        "note": "logprob and lse distances are diagnostics (owner policy 2026-10-06); the gate is the id identity",
    }
    out = Path(args.receipt) if args.receipt else work / f"parity-{kind}.json"
    write_json_atomic(out, receipt)
    c = comparison
    print(f"PARITY {kind}: passed={passed} positions={c['positions']} topk_ids_equal={c['topk_ids_equal_all_positions']} "
          f"mismatch_positions={c['topk_id_mismatch_positions']} top1_equal={c['top1_equal_positions']}/{c['positions']}", flush=True)
    print(f"PARITY {kind}: logprob_max_abs_diff_vs_fp32_ref={c['logprob_max_abs_diff_vs_fp32_reference']} "
          f"vs_bf16_rounded_ref={c['logprob_max_abs_diff_vs_reference_rounded_to_bf16']} "
          f"lse_max_abs_diff={c['lse_max_abs_diff']}", flush=True)
    u = unpadded
    print(f"PARITY {kind} diagnostic (unpadded single-row full load): ids_equal={u['topk_ids_equal_all_positions']} "
          f"mismatch_positions={u['topk_id_mismatch_positions']} logprob_max_abs_diff={u['logprob_max_abs_diff_vs_fp32_reference']}",
          flush=True)
    print(f"PARITY {kind}: receipt {out} sha256 {file_sha256(out)}", flush=True)
    return 0 if passed else 2


def make_parity_vl_corpus(work: Path, args: argparse.Namespace) -> Path:
    """Two deterministic synthetic images (drawn, not downloaded) and two rows: one
    image + question + answer, and BOTH images in one row (multi-image M-RoPE)."""

    from PIL import Image, ImageDraw

    media = work / "media"
    media.mkdir(parents=True, exist_ok=True)
    a = Image.new("RGB", (448, 320), (235, 235, 225))
    draw = ImageDraw.Draw(a)
    draw.rectangle([40, 40, 200, 200], fill=(200, 30, 30))
    draw.ellipse([240, 80, 400, 240], fill=(30, 60, 200))
    draw.text((60, 260), "RED SQUARE  BLUE CIRCLE", fill=(0, 0, 0))
    b = Image.new("RGB", (256, 384), (20, 20, 20))
    draw = ImageDraw.Draw(b)
    for i in range(6):
        draw.rectangle([20, 20 + 58 * i, 236, 60 + 58 * i], fill=(40 * i, 200 - 30 * i, 90))
    draw.text((30, 360), "six bars", fill=(255, 255, 255))
    a.save(media / "parity_a.png")
    b.save(media / "parity_b.png")
    rows = [
        {"row_id": "vl-parity-0", "messages": [
            {"role": "user", "content": [{"type": "image", "path": "media/parity_a.png"},
                                         {"type": "text", "text": "What shapes are in this picture, and what colours are they?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "A red square on the left and a blue circle on the right, with a caption underneath."}]}]},
        {"row_id": "vl-parity-1", "messages": [
            {"role": "user", "content": [{"type": "image", "path": "media/parity_b.png"},
                                         {"type": "image", "path": "media/parity_a.png"},
                                         {"type": "text", "text": "How many bars are in the first image? Does the second image contain a circle?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "The first image has six horizontal bars. Yes, the second image has a blue circle."}]}]},
    ]
    path = work / "parity_vl_corpus.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# smoke-vl: what the teacher says about a few images / clips, for a human read.
# --------------------------------------------------------------------------

#: Qwen3.5 / Qwen3.8 end-of-turn ids (generation_config.json eos_token_id).
DEFAULT_EOS_IDS = (248046, 248044)


def greedy_vl_answers(teacher: Teacher, batch: dict[str, Any], device: str, vocab_rows: int,
                      max_new_tokens: int, eos_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    """Greedy decode for every row of a right-padded VL batch, no KV cache: each step
    re-runs the whole prefix through T(i) and reads the head at each row's last real
    position.  Exact (the same forward the cache writer runs), slow, for a smoke only."""

    import torch
    import torch.nn.functional as F

    ids = batch["input_ids"].clone()
    types = batch["mm_token_type_ids"].clone()
    lengths = [int(x) for x in batch["lengths"].tolist()]
    media = {k: v.to(device) for k, v in batch.items()
             if k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")}
    generated: list[list[int]] = [[] for _ in lengths]
    stop = ["max_new_tokens"] * len(lengths)
    done = [False] * len(lengths)
    weight = teacher.head_weight[:vocab_rows]
    with torch.no_grad():
        for _step in range(int(max_new_tokens)):
            if all(done):
                break
            hidden = teacher.vl_hidden({"input_ids": ids.to(device), "mm_token_type_ids": types.to(device), **media})
            rows = [r for r in range(len(lengths)) if not done[r]]
            last = torch.stack([hidden[r, lengths[r] - 1] for r in rows])
            logits = F.linear(last.to(weight.dtype), weight).float()
            picks = logits.argmax(dim=-1).tolist()
            del hidden, last, logits
            if max(lengths[r] for r in rows) >= ids.shape[1]:
                ids = torch.cat([ids, torch.zeros((ids.shape[0], 1), dtype=ids.dtype)], dim=1)
                types = torch.cat([types, torch.zeros((types.shape[0], 1), dtype=types.dtype)], dim=1)
            for r, token in zip(rows, picks):
                ids[r, lengths[r]] = int(token)
                lengths[r] += 1
                generated[r].append(int(token))
                if int(token) in eos_ids:
                    done[r], stop[r] = True, "eos"
    return [{"generated_ids": g, "stop": s} for g, s in zip(generated, stop)]


def run_smoke_vl(args: argparse.Namespace) -> int:
    import torch

    kernel = select_linear_attention_kernel(args.linear_attn_kernel)
    args.head_precision = option(args, "head_precision")
    head_matmul = configure_head_matmul(args.head_precision)
    template_kwargs = template_kwargs_of(args)
    corpus = load_vl_corpus(Path(args.corpus))
    picks = [int(x) for x in str(args.rows).split(",") if x.strip()]
    if not picks or max(picks) >= corpus["rows"]:
        raise CacheError(f"--rows {args.rows!r} must index rows of a {corpus['rows']}-row corpus")
    model_dir = Path(args.model_dir).resolve()
    processor_dir = Path(args.processor_dir or args.model_dir)
    processor = load_processor(processor_dir, args.max_pixels)
    devices = visible_card_check() if str(args.device).startswith("cuda") else []
    before = teacher_digest(model_dir, args.teacher_digest, args.digest_threads)
    log(f"teacher tree {before['tree_sha256'][:16]} ({before['seconds']} s)")
    base = Path(corpus["path"]).parent
    encoded = [encode_vl_row(processor, corpus["rows_data"][i], base, template_kwargs, generation_prompt=True)
               for i in picks]
    pad_id = int(processor.tokenizer.pad_token_id or 0)
    batch = collate_vl(encoded, picks, pad_id)
    tps = engine_module()
    sampler = tps.ResourceSampler(interval_seconds=2.0)
    sampler.start()
    started = time.perf_counter()
    teacher = build_streamed_teacher(model_dir, args.device, resolve_dtype(args.dtype), args.attn_implementation,
                                     with_vision=True, vocab_rows=int(args.vocab_rows), head_precision=args.head_precision)
    resident = make_teacher_resident(teacher, option(args, "host_resident"))
    built = time.perf_counter() - started
    eos_ids = tuple(int(x) for x in str(args.eos_ids).split(",") if x.strip())
    answers = greedy_vl_answers(teacher, batch, args.device, int(args.vocab_rows), int(args.max_new_tokens), eos_ids)
    seconds = time.perf_counter() - started - built
    sampler.stop()
    after = teacher_digest(model_dir, args.teacher_digest, args.digest_threads)
    rows_out = []
    for n, (row_index, enc, answer) in enumerate(zip(picks, encoded, answers)):
        row = corpus["rows_data"][row_index]
        prompt_ids = enc["input_ids"][0].tolist()
        types = enc["mm_token_type_ids"][0].tolist()
        text_ids = [t for t, k in zip(prompt_ids, types) if k == 0]
        reference = next((c.get("text") for m in row["messages"] if m.get("role") == "assistant"
                          for c in m.get("content", []) if isinstance(c, dict) and c.get("type") == "text"), None)
        media = [{k: v for k, v in c.items() if k in ("type", "path", "num_frames", "frames")}
                 for m in row["messages"] if isinstance(m.get("content"), list)
                 for c in m["content"] if c.get("type") in ("image", "video")]
        rows_out.append({
            "row": row_index, "row_id": row["row_id"], "media": media,
            "prompt_tokens": len(prompt_ids), "visual_tokens": sum(1 for k in types if k != 0),
            "image_grid_thw": enc.get("image_grid_thw", torch.empty(0)).tolist(),
            "video_grid_thw": enc.get("video_grid_thw", torch.empty(0)).tolist(),
            "prompt_text_tail": processor.tokenizer.decode(text_ids[-160:]),
            "answer": processor.tokenizer.decode(answer["generated_ids"]),
            "answer_ids": answer["generated_ids"], "stop": answer["stop"],
            "reference_response": reference,
        })
    receipt = {
        "schema": "omni-distill-teacher-cache-smoke-vl-v1",
        "what": "greedy answers of the streamed VL teacher T(i), user turn only, for a human read; not a gate",
        "tool": tool_identity(), "engine": engine_identity(), "model_dir": str(model_dir),
        "corpus": {k: corpus[k] for k in ("path", "sha256", "jsonl_sha256", "media_files", "rows", "row_formats")},
        "processor": processor_identity(processor, processor_dir, template_kwargs),
        "linear_attention_kernel": kernel, "head_matmul": head_matmul, "dtype": args.dtype,
        "max_new_tokens": int(args.max_new_tokens), "eos_ids": list(eos_ids),
        "teacher_tree_before": before["tree_sha256"], "teacher_tree_after": after["tree_sha256"],
        "teacher_unchanged": before["tree_sha256"] == after["tree_sha256"],
        "build_seconds": round(built, 2), "decode_seconds": round(seconds, 2), "host_resident": resident,
        "resources": sampler.report(), "torch_memory": tps.torch_memory_report(), "devices_at_start": devices,
        "engine_facts": teacher.facts, "rows": rows_out, "runtime": runtime_record(args.device), "finished_utc": utc_now(),
    }
    out = Path(args.receipt)
    write_json_atomic(out, receipt)
    for r in rows_out:
        print(f"SMOKE-VL row {r['row']} ({r['row_id']}) media={json.dumps(r['media'])} visual_tokens={r['visual_tokens']} "
              f"stop={r['stop']}", flush=True)
        print(f"  PROMPT ...{r['prompt_text_tail'][-300:]!r}", flush=True)
        print(f"  ANSWER {r['answer']!r}", flush=True)
        print(f"  REFERENCE {str(r['reference_response'])[:400]!r}", flush=True)
    print(f"SMOKE-VL teacher_unchanged={receipt['teacher_unchanged']} decode {seconds:.1f} s; "
          f"receipt {out} sha256 {file_sha256(out)}", flush=True)
    return 0 if receipt["teacher_unchanged"] else 2


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--processor-dir", default=None, help="defaults to --model-dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--linear-attn-kernel", default="auto", choices=["auto", "torch"])
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--vocab-rows", type=int, default=TEXT_VOCAB_ROWS)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--head-chunk-rows", type=int, default=2048)
    ap.add_argument("--rows-per-shard", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="override the processor's longest_edge pixel budget (default: the parent's own)")
    ap.add_argument("--teacher-digest", default="full", choices=["full", "stat"])
    ap.add_argument("--digest-threads", type=int, default=4)
    ap.add_argument("--allocator-receipt", default=None)
    ap.add_argument("--purpose", default="teacher cache")
    ap.add_argument("--ids-key", default="input_ids")
    ap.add_argument("--lengths-key", default="auto",
                    help="text corpora: the [rows] length tensor (auto = lengths, else real_lengths, else full rows)")
    ap.add_argument("--head-precision", default="fp32", choices=list(HEAD_PRECISIONS),
                    help="fp32: exact fp32 head GEMM, TF32 off (default, finding L3); model: the stock bf16 lm_head")
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="cap on rows x width per micro-batch (0 = rows only)")
    ap.add_argument("--batch-order", default="length", choices=list(BATCH_ORDERS),
                    help="plan micro-batches inside a shard longest-first (default) or in corpus order")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='VL: JSON passed to apply_chat_template, e.g. {"enable_thinking": false} (locked per cache)')
    ap.add_argument("--host-resident", default="pageable", choices=["pageable", "off"],
                    help="read every teacher layer into host RAM once (default; immune to page-cache eviction) or stream "
                         "from the safetensors mmap on every pass (off)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("build-text", "build-vl"):
        p = sub.add_parser(name)
        add_common(p)
        p.add_argument("--teacher", required=True, help="cache namespace, e.g. q38-27bu")
        p.add_argument("--corpus", required=True, action="append",
                       help="repeat for several corpora: one cache each, one teacher load and one digest pair shared")
        p.add_argument("--out-root", default="/home/cereal/omni-campaign/distill/teacher_cache")
        p.add_argument("--verify-existing", action="store_true",
                       help="re-hash already-present shards before skipping them (default: size check)")
        p.add_argument("--max-new-shards", type=int, default=0,
                       help="write at most this many shards this session, then stop unsealed with exit 0 (0 = all)")
    s = sub.add_parser("smoke-vl")
    add_common(s)
    s.add_argument("--corpus", required=True)
    s.add_argument("--rows", required=True, help="comma-separated row indexes of the corpus")
    s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument("--eos-ids", default=",".join(str(e) for e in DEFAULT_EOS_IDS))
    s.add_argument("--receipt", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--cache-dir", required=True)
    for name in ("parity-text", "parity-vl"):
        p = sub.add_parser(name)
        add_common(p)
        p.add_argument("--work-dir", required=True)
        p.add_argument("--receipt", default=None)
        p.add_argument("--corpus", default=None, help="parity-vl: a JSONL corpus (default: two drawn images)")
        p.add_argument("--parity-rows", type=int, default=4)
        p.add_argument("--parity-length", type=int, default=384)
        p.add_argument("--seed", type=int, default=20261012)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        if args.command == "build-text":
            return build_cache(args, "text")
        if args.command == "build-vl":
            return build_cache(args, "vl")
        if args.command == "smoke-vl":
            return run_smoke_vl(args)
        if args.command == "verify":
            return verify_cache(Path(args.cache_dir))
        if args.command == "parity-text":
            return run_parity(args, "text")
        if args.command == "parity-vl":
            return run_parity(args, "vl")
    except CacheError as exc:
        log(f"REFUSED: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
