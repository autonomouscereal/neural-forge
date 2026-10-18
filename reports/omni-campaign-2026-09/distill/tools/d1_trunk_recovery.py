#!/usr/bin/env python3
"""d1_trunk_recovery.py -- OMNI1 stage D1: per-layer trunk FFN recovery against the Qwen3.8 parent.

Lane d1, Claude, 2026-10-12.  Design of record: distill/DISTILL_DESIGN_2026-10-12.md, section 4 row D1 and
section 5.1; rulings distill/STAGE_D_RULINGS_2026-10-12.md.

WHAT ONE LAYER DOES (text layer i; the child's charted media blocks, if any, are identity on text rows and
are verified to be so, then skipped):

    parent   y_p(i)  = f_p,i(y_p(i-1))                        the parent PREFIX (Qwen3.8-27B-U, S0 Chimera import)
    child    x_m     = h_c(i) + attn_c,i(h_c(i))              the child PREFIX h_c(i); attention frozen
    predict  p       = x_m + FFN_trunk(norm2(x_m))            trunk columns [0, q) only -- NO K3 term
    loss     L       = mean_t [ ||p_t - y_t|| / ||y_t||  +  (1 - cos(p_t, y_t)) ]
    advance  h_c(i+1) = x_m + FFN_full(norm2(x_m))            trunk (trained) + K3 columns (frozen) -- the K3 term
                                                               stays in the prefix advance, outside the loss

Trainable per layer: the trunk gate rows [0, q), up rows [W, W+q) of the fused ``ffn.up.weight`` [2W, h], the
down columns [:, 0:q) of ``ffn.down.weight`` [h, W], and ``norm2.weight`` [h]; 64 x (3 x 5120 x 4352 + 5120)
= 4,278,517,760 on the S5/OMNI1 child.  Frozen: attention/KDA, K3 rows/columns [q, q+k3), zero-pad rows,
embed, head, media.  Because attention is frozen and h_c(i) is fixed while layer i trains, x_m is computed
ONCE per layer and the training is a per-token regression: tokens are shuffled freely.

Engine.  Blocks are built the way the reviewed streamed engine builds them (run_omni_frozen_review
StreamedChimera._build_block: meta construction, strict assign-load from the shard views, the source's fp32
RoPE table put back) and run the way c10_virtual_child_check.run_block runs them (keyword masks, all tokens
valid, where(valid)).  Two host buffers hold the parent stream and the child stream ([N, h] each, pinned);
the frozen forwards spread row batches over ``--devices`` (one thread per device, host-staged: no tensor ever
moves device to device).  Training is host-reduced data parallel: every device computes the analytic
gradient of its token shard (scaled by the GLOBAL token count; several CUDA cards run their shards on one thread
each, pinned non-blocking uploads), the shards are summed on the host in device order, Adam runs on the first device's fp32 master copy and the updated parameters are host-staged to the
other replicas.  The analytic gradient is checked against autograd in the tests.

Outputs (everything under --out, plus --delta-out when given; nothing else is written):

    <out>/rungs/L{i:02d}.safetensors (+ .json)   the layer's trained tensors in the child's dtype + record
    <out>/delta/D1_delta.safetensors (+ .json)   all trained layers (--delta-out overrides the directory)
    <out>/d1_receipt.json (+ .sha256)            rewritten after every layer
    <out>/d1.marker.json                         ownership marker (a foreign non-empty --out is refused);
                                                 a --delta-out outside --out gets the same guard and its own marker

DELTA FORMAT (schema ``omni-campaign.d1-delta.v1``, for ``oc_checkpoint.py --delta-base``): one safetensors
file whose keys are ``<base tensor name>::<part>``; the manifest's ``entries`` give for every key the base
tensor, the slice it overwrites (``[[row_begin, row_end], [col_begin, col_end]]``, a 1-D tensor uses one
pair), shape, dtype, the delta tensor's sha256 and the base tensor's sha256 (the chimera TensorView digest
of the WHOLE base tensor before the overwrite).  ``base`` pins the base checkpoint: config/index sha256 and,
unless --base-digest headers, the sha256 of every shard file.  ``apply_delta`` below is the reference
reader: base tensor -> clone -> slices overwritten -> the D1 child's tensor.  Parts:
    ffn.up.weight::gate_trunk  rows [0, q)        ffn.up.weight::up_trunk   rows [W, W+q)
    ffn.down.weight::down_trunk cols [0, q)       norm2.weight::whole

DIAGNOSTICS (owner policy 2026-10-06: finite fit numbers never refuse): per layer, on held tokens, before
(the child's own weights) and after (the trained weights as stored): residual relL2 / cos / D1 loss of the
trunk-only prediction and of the full child layer output (with K3) vs y_p(i); the FFN-level relL2
||FFN_trunk - (y_p - x_m)|| / ||y_p - x_m||; the K3-term energy relative to the trunk FFN output
sqrt(sum ||K3||^2 / sum ||FFN_trunk||^2); the loss curve.  After the last text layer: frozen-head logit
KL(parent || child) and top-1 agreement over vocab rows [0, 248,320), for the D1 child and for the
original child (a baseline stream of the KL rows advanced with the original weights).
REFUSALS (functional): structure / identity / shapes, non-finite streams, gradients or losses, the block
decomposition check (x_m + FFN(x_m) must reproduce the block's own forward), a charted block that is not
identity on text rows, the disk floor, --out or --delta-out overlapping a model directory / under /mnt/ai_data /
naming a foreign non-empty directory, ids outside the text rows.

Resume: a rung whose record carries the same input chain (run identity + every earlier trained rung's sha)
is loaded instead of trained (--resume); the prefix is recomputed through it.

HARDENED RUN (1.3.0, state doc 2026-10-14 step 4a; no allocator attestation, no Windows/SSH in the loop -- the
launcher holds the gpu0/gpu4 card locks for the process lifetime, see d1_hardened_entry.py):
  --prefix-dir DIR --prefix-sums-sha256 SHA --prefix-layers N   import committed rungs L00..L{N-1} from the
        canonical rung directory (d1_canonical_rungs.py): SHA256SUMS must hash to the pin; every imported rung
        file and record must hash to its SHA256SUMS line, name its text layer / child block, and carry the base
        tensor sha256 of the child being read.  Imported rungs are applied, never retrained.
  --boundary-dir DIR   after the last imported layer and after every newly trained layer (--boundary-every K:
        every K-th; never the last layer), the next-layer inputs -- the parent stream P, the child stream C (all
        rows: fit + held) and the KL-row baseline B -- go to DIR/L{i:02d}/ (raw bytes + sha256 + the rung list they
        were advanced through) on background threads that never block training (BoundaryCache); only the latest
        is kept.  On start, a boundary whose run identity and rung list equal this run's is loaded (every
        byte re-hashed) and layers 0..i are not recomputed.  A boundary that does not match is ignored.
  Per layer: process RSS, the cgroup's memory.current / memory.stat and MemAvailable are logged and recorded,
        then host buffers are freed (gc, malloc_trim, page cache of the shards / rungs / boundary dropped).
        Trained rungs are no longer held in RAM (they were 134 MB a layer, 8.6 GB at L63): the delta re-reads
        them from the verified rung files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TOOL_NAME = "d1_trunk_recovery"
TOOL_VERSION = "1.3.1"   # 1.0.0 (2026-10-12): first version, lane d1; 1.1.0: --attn-row-batch for full-attention
                         # blocks (the KDA reference recurrence is launch-bound per token, so it gets the wide
                         # --row-batch) + next-layer read-ahead of the parent/child block tensors + a deterministic
                         # safetensors writer (safetensors' own header order varies run to run; rung/delta bytes must not)
                         # 1.2.0: right-padded rows: an ids file's `lengths` [rows] (the qtext_teacher_cache corpus format)
                         # makes positions >= length invalid (zeroed stream, out of training, diagnostics and KL)
                         # 1.2.1: non-persistent buffers left on meta (a charted block's routed_type_ids) are taken
                         # from a CPU-built block -- found by the charted-child test
                         # 1.2.2: the stream init embeds in 64-row chunks (no whole-buffer temporaries at 4M tokens)
                         # 1.2.3: the CUDA cache is held between layers (--release-cache restores empty_cache): a card
                         # that dips below 2000 MiB between layers is taken by other lanes (an OOM at L04, 2026-10-13)
                         # 1.2.4 (review fixes): --delta-out gets the --out guard (model-dir overlap, /mnt/ai_data,
                         # foreign non-empty dir) and its own marker, checked before any write; the data-parallel step
                         # runs each CUDA card's shard on its own thread with lazy gathers + pinned non_blocking uploads
                         # (1.2.3 walked the cards one after another: ~1.33x, not 2x, on two cards)
                         # 1.2.5-1.2.10 (Codex, 2026-10-13/14, never on this path): --preregistration, the source
                         # successor contract (d1_source_contract), allocator attestation before every optimizer
                         # update, prefix continuation; the numerics are 1.2.4's
                         # 1.3.0 (2026-10-14, state doc step 4a): 1.2.10's --preregistration + source contract kept,
                         # attestation + continuation dropped; --prefix-dir (canonical rungs), --boundary-dir
                         # (next-layer input cache), per-layer memory log + host-buffer free, rungs not held in RAM,
                         # streaming deterministic writer (same bytes), PROGRESS for the entry's heartbeat
                         # 1.3.1 (2026-10-14 review): the cgroup's memory.pressure (PSI some / full avg10 / avg60) in
                         # every memory record + the per-layer line; read_stream_file drops each chunk's page cache as
                         # it goes (a 41 GB boundary load no longer fills the cgroup with clean cache before the drop)
SCHEMA ="omni-campaign.d1-trunk-recovery.v1"
DELTA_SCHEMA = "omni-campaign.d1-delta.v1"
RUNG_SCHEMA = "omni-campaign.d1-rung.v1"
MARKER = "d1.marker.json"
BOUNDARY_SCHEMA = "omni-campaign.d1-boundary.v1"
PREFIX_SUMS = "SHA256SUMS"     # d1_canonical_rungs.py
TEXT_ROWS = 248_320            # design section 4: every text loss / KL over vocab rows [0, 248,320)
PROGRESS: dict[str, Any] = {}  # {"layer", "phase", "at"}: read by d1_hardened_entry's heartbeat thread
TINY = 1e-30
PROTECTED_OUT_PREFIXES = ("/mnt/ai_data",)   # CP R6: nothing transient under the NAS


class Refusal(SystemExit):
    """A functional refusal (structure, identity, non-finite, disk floor)."""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{TOOL_NAME}] {message}", flush=True)


# --------------------------------------------------------------------------
# chimera binding (CHIMERA_SRC or CHIMERA_OMNI_SRC/src or PYTHONPATH)
# --------------------------------------------------------------------------
def bind_chimera() -> dict[str, Any]:
    src = os.environ.get("CHIMERA_SRC") or (os.path.join(os.environ["CHIMERA_OMNI_SRC"], "src")
                                            if os.environ.get("CHIMERA_OMNI_SRC") else None)
    if src and os.path.isdir(os.path.join(src, "chimera_omni")) and src not in sys.path:
        sys.path.insert(0, src)
    import chimera_omni  # noqa: F401

    loaded = Path(chimera_omni.__file__).resolve().parents[1]
    tree = loaded.parent
    commit = None
    for name in ("COMMIT",):
        path = tree / name
        if path.is_file():
            commit = path.read_text(encoding="utf-8").strip()
    return {"src": str(loaded), "tree": str(tree), "commit": commit, "requested": src}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def tensor_bytes_sha256(tensor: Any) -> str:
    """The tensor's own bytes in its own dtype (c10.sha256_tensor_bytes)."""
    import torch

    t = tensor.detach().cpu().contiguous()
    raw = t.view(torch.uint8) if t.dtype in (torch.bfloat16, torch.float16) else t
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def peak_rss_gb() -> float | None:
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 3)
    except Exception:  # noqa: BLE001 - windows / exotic platforms
        return None


def progress(layer: int | None, phase: str) -> None:
    PROGRESS.update({"layer": layer, "phase": phase, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def _cgroup_dir() -> Path | None:
    """This process's cgroup v2 directory (systemd unit / docker container), or None."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                d = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                return d if (d / "memory.current").is_file() else None
    except OSError:
        pass
    return None


def host_memory() -> dict[str, Any]:
    """GiB figures: process RSS split (anon / file / shmem) and its high-water mark, the cgroup's memory.current /
    max / peak and its memory.stat anon / file / shmem / dirty / writeback, host MemAvailable.  Missing sources
    are left out (non-Linux)."""
    gib = 2 ** 30
    out: dict[str, Any] = {}
    try:
        status = {l.split(":")[0]: l.split()[1:] for l in Path("/proc/self/status").read_text().splitlines() if ":" in l}
        for key, name in (("VmRSS", "rss"), ("RssAnon", "rss_anon"), ("RssFile", "rss_file"), ("RssShmem", "rss_shmem"),
                          ("VmHWM", "rss_peak")):
            if key in status:
                out[name] = round(int(status[key][0]) * 1024 / gib, 3)
    except OSError:
        pass
    cg = _cgroup_dir()
    if cg is not None:
        cgroup: dict[str, Any] = {"path": str(cg)}
        for name in ("memory.current", "memory.max", "memory.peak"):
            try:
                raw = (cg / name).read_text().strip()
                cgroup[name.split(".")[1]] = None if raw == "max" else round(int(raw) / gib, 3)
            except (OSError, ValueError):
                pass
        try:
            stat = dict(l.split() for l in (cg / "memory.stat").read_text().splitlines())
            for key in ("anon", "file", "shmem", "file_dirty", "file_writeback"):
                if key in stat:
                    cgroup[key] = round(int(stat[key]) / gib, 3)
        except (OSError, ValueError):
            pass
        try:
            events = dict(l.split() for l in (cg / "memory.events").read_text().splitlines())
            cgroup["oom_kill"] = int(events.get("oom_kill", 0))
        except (OSError, ValueError):
            pass
        try:   # 1.3.1: PSI, the signal systemd-oomd acts on (full avg10 over its limit for its duration)
            for line in (cg / "memory.pressure").read_text().splitlines():
                kind, *fields = line.split()
                vals = dict(f.split("=", 1) for f in fields)
                cgroup[f"pressure_{kind}"] = {"avg10": float(vals["avg10"]), "avg60": float(vals["avg60"])}
        except (OSError, ValueError, KeyError):
            pass
        out["cgroup"] = cgroup
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                out["host_mem_available"] = round(int(line.split()[1]) * 1024 / gib, 3)
    except OSError:
        pass
    return out


def memory_line(mem: dict[str, Any]) -> str:
    cg = mem.get("cgroup", {})
    full = cg.get("pressure_full", {})
    return (f"rss {mem.get('rss')} GiB (anon {mem.get('rss_anon')} file {mem.get('rss_file')} shmem {mem.get('rss_shmem')}, "
            f"peak {mem.get('rss_peak')}); cgroup {cg.get('current')}/{cg.get('max')} GiB (anon {cg.get('anon')} "
            f"file {cg.get('file')} dirty {cg.get('file_dirty')} oom_kill {cg.get('oom_kill')}; "
            f"psi full avg10 {full.get('avg10')} avg60 {full.get('avg60')}); "
            f"MemAvailable {mem.get('host_mem_available')} GiB")


def drop_file_cache(paths: list[Path] | tuple[Path, ...]) -> int:
    """posix_fadvise(DONTNEED) on whole files: their clean page cache leaves this cgroup's charge.  Returns the
    count of files advised (missing files / platforms without fadvise are skipped)."""
    if not hasattr(os, "posix_fadvise"):
        return 0
    done = 0
    for p in paths:
        try:
            fd = os.open(str(p), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            done += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return done


def free_host_buffers(paths: list[Path] | tuple[Path, ...] = ()) -> dict[str, Any]:
    """After a rung: Python garbage, glibc's free lists (malloc_trim) and the page cache of ``paths``."""
    import gc

    collected = gc.collect()
    trimmed = None
    try:
        import ctypes

        trimmed = bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        pass
    return {"gc_collected": collected, "malloc_trim_released": trimmed, "files_dropped": drop_file_cache(paths)}


_ST_DTYPES = {"float32": "F32", "float16": "F16", "bfloat16": "BF16", "float64": "F64", "int64": "I64", "int32": "I32",
              "int16": "I16", "int8": "I8", "uint8": "U8", "bool": "BOOL"}


def save_safetensors_deterministic(tensors: dict[str, Any], path: Path, metadata: dict[str, str]) -> None:
    """A safetensors file whose bytes depend only on the tensors and the metadata: names sorted, data laid out
    in that order, header JSON with sorted keys padded to 8 bytes; written to a temp file then renamed.
    (The library writer orders same-dtype tensors by a per-run hash, so equal states hashed unequal.)
    1.3.0: the tensors are streamed from their own memory (no second copy of the whole delta as bytes)."""
    import torch

    names = sorted(tensors)
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    for name in names:
        t = tensors[name]
        dtype = _ST_DTYPES[str(t.dtype).replace("torch.", "")]
        nbytes = int(t.numel()) * t.element_size()
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    text += b" " * ((8 - len(text) % 8) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(len(text).to_bytes(8, "little"))
        handle.write(text)
        for name in names:
            t = tensors[name].detach().to("cpu").contiguous()
            if t.numel():
                handle.write(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    os.replace(tmp, path)


def write_json_atomic(path: Path, doc: Any, with_sha: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    digest = sha256_file(path)
    if with_sha:
        (path.parent / (path.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    return digest


# --------------------------------------------------------------------------
# checkpoints: config, layout, block build (the StreamedChimera rules)
# --------------------------------------------------------------------------
@dataclass
class Checkpoint:
    root: Path
    doc: dict[str, Any]
    cfg: Any
    index: dict[str, Any]
    text_blocks: list[int]          # block indices of the non-charted (text) blocks, in order
    charted_blocks: list[int]
    dtype_on_disk: Any              # the FFN storage dtype
    _ref_gqa: Any = None

    @property
    def hidden(self) -> int:
        return int(self.cfg.hidden_size)

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.root), "config_sha256": sha256_file(self.root / "config.json"),
                "index_sha256": sha256_file(self.root / "model.safetensors.index.json"),
                "chimera_commit": self.doc.get("chimera_commit")}


def open_checkpoint(root: str | Path) -> Checkpoint:
    import torch
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.importing.safetensors_io import _torch_dtype, read_checkpoint_config, read_header, read_index

    root = Path(root).absolute()
    doc = read_checkpoint_config(root)
    cfg = ChimeraConfig.from_manifest(doc["config"])
    if cfg.residual_mode != "none":
        raise Refusal(f"{root}: residual_mode {cfg.residual_mode!r}; D1 mirrors the plain residual trunk only")
    index = read_index(root) or {}
    if not index.get("weight_map"):
        raise Refusal(f"{root}: no model.safetensors.index.json weight map")
    kinds = [cfg.block_kind(i) for i in range(int(cfg.num_layers))]
    text = [i for i, k in enumerate(kinds) if k != "charted"]
    charted = [i for i, k in enumerate(kinds) if k == "charted"]
    first = f"blocks.{text[0]}.ffn.up.weight"
    shard = root / index["weight_map"][first]
    header, _ = read_header(shard)
    dtype = _torch_dtype(str(header[first]["dtype"]))
    if not dtype.is_floating_point:
        raise Refusal(f"{root}: {first} is {dtype}")
    del torch
    return Checkpoint(root, doc, cfg, index, text, charted, dtype)


def block_tensor_names(ck: Checkpoint, k: int) -> list[str]:
    import torch
    from chimera_omni.modeling.layers import ChimeraBlock

    with torch.device("meta"):
        block = ChimeraBlock(ck.cfg, k)
    return [f"blocks.{k}.{n}" for n in block.state_dict()]


def read_block_state(ck: Checkpoint, k: int) -> tuple[dict[str, Any], dict[str, str]]:
    """The block's tensors as CPU tensors in their on-disk dtype + their chimera TensorView sha256
    (read_safetensors verify=True checks them against the index digests when the index records any)."""
    from chimera_omni.importing.safetensors_io import read_safetensors

    names = block_tensor_names(ck, k)
    views = read_safetensors(ck.root, names, verify=True)
    state, shas = {}, {}
    for name in names:
        view = views[name]
        state[name[len(f"blocks.{k}."):]] = view.torch(device="cpu")
        shas[name] = view.sha256()
    return state, shas


def _reference_gqa_inv_freq(ck: Checkpoint) -> Any:
    """GatedGQAAttention(cfg).inv_freq built on the CPU in fp32 (run_omni_frozen_review.rope_inv_freq)."""
    import torch
    from chimera_omni.modeling.layers import GatedGQAAttention

    if ck._ref_gqa is None:
        ref = GatedGQAAttention(ck.cfg)
        inv = getattr(ref, "inv_freq", None)
        ck._ref_gqa = None if inv is None else inv.detach().to(torch.float32).clone()
        del ref
    return ck._ref_gqa


def build_block(ck: Checkpoint, k: int, state: dict[str, Any], device: str, dtype: Any) -> Any:
    """StreamedChimera._build_block: meta construction, strict assign-load, fp32 RoPE table, no meta left."""
    import torch
    from chimera_omni.modeling.layers import ChimeraBlock, GatedGQAAttention

    with torch.device("meta"):
        block = ChimeraBlock(ck.cfg, k)
    target = block.state_dict()
    params = {n for n, _ in block.named_parameters()}
    loaded = {}
    for name, ref in target.items():
        if name not in state:
            raise Refusal(f"block {k}: no tensor {name!r}")
        t = state[name]
        t = t.to(device=device, dtype=dtype if ref.is_floating_point() else ref.dtype)
        if tuple(t.shape) != tuple(ref.shape):
            raise Refusal(f"block {k}: {name} shape {tuple(t.shape)} != {tuple(ref.shape)}")
        loaded[name] = torch.nn.Parameter(t, requires_grad=False) if name in params else t
    extra = sorted(set(state) - set(target))
    if extra:
        raise Refusal(f"block {k}: unexpected tensors {extra[:4]}")
    block.load_state_dict(loaded, strict=True, assign=True)
    if isinstance(getattr(block, "mixer", None), GatedGQAAttention) and getattr(block.mixer, "inv_freq", None) is not None:
        inv = _reference_gqa_inv_freq(ck)
        block.mixer.inv_freq = inv.to(device=device)
    meta_buffers = [n for n, t in block.named_buffers() if t.is_meta]
    if meta_buffers:
        # non-persistent buffers built from the config (e.g. a charted block's routed_type_ids): taken from a
        # CPU-constructed block, in the dtype the constructor gives them (never cast to the run dtype)
        ref = dict(ChimeraBlock(ck.cfg, k).named_buffers())
        for name in meta_buffers:
            owner, _, leaf = name.rpartition(".")
            module = block.get_submodule(owner) if owner else block
            module._buffers[leaf] = ref[name].detach().clone().to(device=device)
        del ref
    leftover =[n for n, t in list(block.named_parameters()) + list(block.named_buffers()) if t.is_meta]
    if leftover:
        raise Refusal(f"block {k}: tensors never materialised: {leftover[:3]}")
    for p in block.parameters():
        p.requires_grad_(False)
    return block.eval()


def run_block(block: Any, x: Any, valid: Any) -> Any:
    """c10_virtual_child_check.run_block: keyword masks, then where(valid)."""
    import torch

    x = block(x, native_media_slot_mask=None, valid_media_mask=None, enable_native_media_refinement=False,
              token_valid_mask=valid)
    return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))


def attention_half(block: Any, x: Any, valid: Any) -> Any:
    """x + attention_output(x): the first half of ChimeraBlock.forward, exactly."""
    return x + block.attention_output(x, token_valid_mask=valid)


def ffn_half(block: Any, xm: Any, valid: Any) -> Any:
    """where(valid, x_m + feed_forward_output(x_m)): the second half of ChimeraBlock.forward + run_block's mask."""
    import torch

    out = xm + block.feed_forward_output(xm, token_valid_mask=valid)
    return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


# --------------------------------------------------------------------------
# FFN layout of a union child (c10): fused up [2W, h] = [gate | up], down [h, W]
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UnionLayout:
    width: int       # W
    trunk: int       # q
    k3: int          # K3 channels
    pad: int         # zero rows

    def check(self) -> None:
        if self.trunk <= 0 or self.k3 < 0 or self.pad < 0 or self.trunk + self.k3 + self.pad != self.width:
            raise Refusal(f"union layout q {self.trunk} + k3 {self.k3} + pad {self.pad} != width {self.width}")


def union_layout(ck: Checkpoint, trunk: int | None, k3: int | None) -> UnionLayout:
    meta = dict(getattr(ck.cfg, "metadata", {}) or {})
    q = trunk if trunk is not None else int(meta.get("c10_trunk_intermediate_size", 0) or 0)
    kk = k3 if k3 is not None else int(meta.get("c10_k3_channels", 0) or 0)
    widths = {ck.cfg.resolved_intermediate_size(i) for i in ck.text_blocks}
    if len(widths) != 1:
        raise Refusal(f"{ck.root}: the child's text FFN widths are not uniform: {sorted(widths)}")
    if not q:
        raise Refusal(f"{ck.root}: no c10_trunk_intermediate_size in the config metadata; pass --trunk-width")
    w = widths.pop()
    layout = UnionLayout(w, q, kk, w - q - kk)
    layout.check()
    return layout


def trunk_slices(state: dict[str, Any], lay: UnionLayout) -> dict[str, Any]:
    up, down = state["ffn.up.weight"], state["ffn.down.weight"]
    w, q = lay.width, lay.trunk
    return {"gate": up[:q], "up": up[w:w + q], "down": down[:, :q], "norm": state["norm2.weight"]}


def k3_slices(state: dict[str, Any], lay: UnionLayout) -> dict[str, Any]:
    up, down = state["ffn.up.weight"], state["ffn.down.weight"]
    w, q, k = lay.width, lay.trunk, lay.k3
    return {"gate": up[q:q + k], "up": up[w + q:w + q + k], "down": down[:, q:q + k]}


def patched_state(state: dict[str, Any], lay: UnionLayout, new: dict[str, Any]) -> dict[str, Any]:
    """The block state with the trunk slices + norm2 overwritten (dtype of the base tensors)."""
    out = dict(state)
    w, q = lay.width, lay.trunk
    up = state["ffn.up.weight"].clone()
    down = state["ffn.down.weight"].clone()
    up[:q] = new["gate"].to(up.dtype)
    up[w:w + q] = new["up"].to(up.dtype)
    down[:, :q] = new["down"].to(down.dtype)
    out["ffn.up.weight"] = up
    out["ffn.down.weight"] = down
    out["norm2.weight"] = new["norm"].to(state["norm2.weight"].dtype).clone()
    return out


DELTA_PARTS = (("ffn.up.weight", "gate_trunk", "gate"), ("ffn.up.weight", "up_trunk", "up"),
               ("ffn.down.weight", "down_trunk", "down"), ("norm2.weight", "whole", "norm"))


def delta_slice(part: str, lay: UnionLayout, h: int) -> list[list[int]]:
    w, q = lay.width, lay.trunk
    return {"gate_trunk": [[0, q], [0, h]], "up_trunk": [[w, w + q], [0, h]],
            "down_trunk": [[0, h], [0, q]], "whole": [[0, h]]}[part]


def apply_delta(base: Any, entry: dict[str, Any], value: Any) -> Any:
    """Reference reader of the delta format: the base tensor with ``entry['slice']`` overwritten by ``value``."""
    out = base.clone()
    sl = tuple(slice(int(a), int(b)) for a, b in entry["slice"])
    if tuple(out[sl].shape) != tuple(value.shape):
        raise Refusal(f"{entry['key']}: slice {entry['slice']} of {tuple(base.shape)} != delta {tuple(value.shape)}")
    out[sl] = value.to(out.dtype)
    return out


# --------------------------------------------------------------------------
# the D1 objective: forward, analytic gradient, diagnostics
# --------------------------------------------------------------------------
def rms_scaled(x: Any, eps: float) -> Any:
    """x * rsqrt(mean(x^2) + eps): chimera RMSNorm before its weight (computed in x's float dtype)."""
    import torch

    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _mm(a: Any, b: Any, mm_dtype: Any) -> Any:
    if mm_dtype is None or a.dtype == mm_dtype:
        return a @ b
    return (a.to(mm_dtype) @ b.to(mm_dtype)).to(a.dtype)


def d1_loss_and_grads(xm: Any, y: Any, p: dict[str, Any], eps: float, scale: float, *, residual: bool = True,
                      mm_dtype: Any = None, grads: bool = True) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    """Sum over tokens of [relL2_t + 1 - cos_t] times ``scale`` (= 1 / global token count) and its
    analytic gradient w.r.t. gate [q,h], up [q,h], down [h,q], norm [h].  Works in p's dtype (fp32 / fp64).

    residual=True : prediction x_m + FFN_trunk(norm2(x_m)) vs y           (design D1)
    residual=False: prediction FFN_trunk(norm2(x_m))       vs y - x_m      (--loss-site delta)
    """
    import torch

    work = p["gate"].dtype
    x = xm.to(work)
    yt = y.to(work)
    xr = rms_scaled(x, eps)
    n = xr * p["norm"]
    a = _mm(n, p["gate"].T, mm_dtype)
    b = _mm(n, p["up"].T, mm_dtype)
    sg = torch.sigmoid(a)
    s = a * sg
    hh = s * b
    f = _mm(hh, p["down"].T, mm_dtype)
    if residual:
        pred, target = x + f, yt
    else:
        pred, target = f, yt - x
    e = pred - target
    ne = e.norm(dim=-1, keepdim=True)
    ny = target.norm(dim=-1, keepdim=True).clamp_min(TINY)
    npn = pred.norm(dim=-1, keepdim=True).clamp_min(TINY)
    rel = ne / ny
    cos = (pred * target).sum(dim=-1, keepdim=True) / (npn * ny)
    loss = (rel + 1.0 - cos).sum() * scale
    stats = {"rel_sum": rel.sum(), "cos_sum": cos.sum(), "tokens": int(x.shape[0])}
    if not grads:
        return loss, None, stats
    g = (e / (ne.clamp_min(TINY) * ny) - (target / (npn * ny) - cos * pred / npn.pow(2))) * scale
    d_down = _mm(g.T, hh, mm_dtype)
    dh = _mm(g, p["down"], mm_dtype)
    ds = dh * b
    db = dh * s
    da = ds * (sg * (1.0 + a * (1.0 - sg)))
    d_gate = _mm(da.T, n, mm_dtype)
    d_up = _mm(db.T, n, mm_dtype)
    dn = _mm(da, p["gate"], mm_dtype) + _mm(db, p["up"], mm_dtype)
    d_norm = (dn * xr).sum(dim=0)
    return loss, {"gate": d_gate, "up": d_up, "down": d_down, "norm": d_norm}, stats


def ffn_terms(xm: Any, trunk: dict[str, Any], k3: dict[str, Any] | None, eps: float) -> tuple[Any, Any | None]:
    """(FFN_trunk, K3 term) on x_m in the trunk dtype (fp32 diagnostics; SiLU-GLU, the child's activation)."""
    import torch.nn.functional as F

    x = xm.to(trunk["gate"].dtype)
    n = rms_scaled(x, eps) * trunk["norm"]
    f = (F.silu(n @ trunk["gate"].T) * (n @ trunk["up"].T)) @ trunk["down"].T
    if k3 is None or k3["gate"].shape[0] == 0:
        return f, None
    t = (F.silu(n @ k3["gate"].T) * (n @ k3["up"].T)) @ k3["down"].T
    return f, t


class DiagAccumulator:
    """Aggregated + per-token-mean scores of a prediction vs a target (float64 sums)."""

    def __init__(self) -> None:
        self.diff = 0.0
        self.tgt = 0.0
        self.rel_tok = 0.0
        self.cos_tok = 0.0
        self.n = 0
        self.finite = True

    def add(self, pred: Any, target: Any) -> None:
        e = (pred - target).double()
        t = target.double()
        pp = pred.double()
        nt = t.norm(dim=-1).clamp_min(TINY)
        self.finite = self.finite and bool(e.isfinite().all())
        self.diff += float(e.pow(2).sum())
        self.tgt += float(t.pow(2).sum())
        self.rel_tok += float((e.norm(dim=-1) / nt).sum())
        self.cos_tok += float(((pp * t).sum(-1) / (pp.norm(dim=-1).clamp_min(TINY) * nt)).sum())
        self.n += int(t.shape[0])

    def result(self) -> dict[str, Any]:
        n = max(self.n, 1)
        rel_tok, cos_tok = self.rel_tok / n, self.cos_tok / n
        return {"relL2": math.sqrt(self.diff / max(self.tgt, TINY)), "relL2_token_mean": rel_tok,
                "cos_token_mean": cos_tok, "d1_loss": rel_tok + 1.0 - cos_tok, "tokens": self.n, "finite": self.finite}


def layer_diagnostics(xm: Any, y: Any, token_ids: Any, trunk: dict[str, Any], k3: dict[str, Any] | None, eps: float,
                      device: str, chunk: int = 4096) -> dict[str, Any]:
    """Scores of one weight set on the given token rows of the host buffers (fp32 on ``device``)."""
    import torch

    res, full, delta = DiagAccumulator(), DiagAccumulator(), DiagAccumulator()
    trunk_sq = k3_sq = 0.0
    tr = {k: v.to(device=device, dtype=torch.float32) for k, v in trunk.items()}
    kk = None if k3 is None else {k: v.to(device=device, dtype=torch.float32) for k, v in k3.items()}
    with torch.no_grad():
        for lo in range(0, int(token_ids.numel()), chunk):
            idx = token_ids[lo:lo + chunk]
            x = xm.index_select(0, idx).to(device=device, dtype=torch.float32)
            t = y.index_select(0, idx).to(device=device, dtype=torch.float32)
            f, term = ffn_terms(x, tr, kk, eps)
            res.add(x + f, t)
            delta.add(f, t - x)
            full.add(x + f + (term if term is not None else 0.0), t)
            trunk_sq += float(f.double().pow(2).sum())
            if term is not None:
                k3_sq += float(term.double().pow(2).sum())
    return {"trunk_only_residual": res.result(), "full_layer_with_k3_residual": full.result(),
            "ffn_level_trunk_vs_needed_update": delta.result(),
            "k3_term_energy_rel_trunk_ffn": math.sqrt(k3_sq / trunk_sq) if trunk_sq > 0 else None}


# --------------------------------------------------------------------------
# host-reduced data-parallel Adam on the trunk slices
# --------------------------------------------------------------------------
PARAM_NAMES = ("gate", "up", "down", "norm")


class TrunkTrainer:
    def __init__(self, init: dict[str, Any], devices: list[str], *, lr: float, betas: tuple[float, float], eps: float,
                 clip: float, norm_eps: float, residual: bool, mm_dtype: Any, master_dtype: Any,
                 parallel: bool | None = None) -> None:
        import torch

        self.devices = devices
        self.lr, self.betas, self.adam_eps, self.clip = lr, betas, eps, clip
        self.norm_eps, self.residual, self.mm_dtype = norm_eps, residual, mm_dtype
        # one thread per device when there are several CUDA cards (auto); CPU replicas share the cores, so they
        # stay sequential unless a test forces the threaded path
        self.parallel = (len(devices) > 1 and all(d.startswith("cuda") for d in devices)) if parallel is None else parallel
        self.replicas = [{k: init[k].to(device=d, dtype=master_dtype).clone() for k in PARAM_NAMES} for d in devices]
        self.m = {k: torch.zeros_like(self.replicas[0][k]) for k in PARAM_NAMES}
        self.v = {k: torch.zeros_like(self.replicas[0][k]) for k in PARAM_NAMES}
        self.t = 0

    def _device_shard(self, d: int, micro: list[Any], n_total: int) -> tuple[dict[str, Any] | None, list[Any], list[Any]]:
        """Device d's micro-batches -> (its gradient sum, its losses, its stats), in micro-batch order.  A
        micro-batch is an (x_m, y) pair or a zero-argument callable returning one (the gather then runs here, on
        this device's thread, while the card works on the previous micro-batch).  CUDA uploads go through
        pinned staging with non_blocking=True, so the host never waits on the stream between micro-batches."""
        import torch

        dev = self.devices[d]
        cuda = dev.startswith("cuda")
        acc = None
        losses: list[Any] = []
        stats: list[Any] = []
        staged: list[Any] = []   # pinned sources stay referenced until the device has consumed them
        for item in micro:
            xm, y = item() if callable(item) else item
            if cuda:
                xm, y = xm.pin_memory(), y.pin_memory()
                staged.extend((xm, y))
            xd = xm.to(dev, non_blocking=cuda)
            yd = y.to(dev, non_blocking=cuda)
            loss, g, st = d1_loss_and_grads(xd, yd, self.replicas[d], self.norm_eps, 1.0 / n_total,
                                            residual=self.residual, mm_dtype=self.mm_dtype)
            losses.append(loss)
            stats.append(st)
            if acc is None:
                acc = g
            else:
                for k in PARAM_NAMES:
                    acc[k] += g[k]
        if cuda:
            torch.cuda.synchronize(torch.device(dev))
        del staged
        return acc, losses, stats

    def shard_grads(self, shards: list[list[Any]], n_total: int) -> tuple[dict[str, Any], float, dict[str, float]]:
        """Each device's micro-batches -> its gradient sum (scaled by 1/n_total); the host reduce in device order.
        With several CUDA cards every device runs its shard on its own thread (1.2.4); the per-device sums and
        the host reduce keep the sequential order, so the result does not depend on the threading."""
        import torch

        if self.parallel and len(self.devices) > 1:
            with concurrent.futures.ThreadPoolExecutor(len(self.devices)) as pool:
                futures = [pool.submit(self._device_shard, d, micro, n_total) for d, micro in enumerate(shards)]
                results = [f.result() for f in futures]
        else:
            results = [self._device_shard(d, micro, n_total) for d, micro in enumerate(shards)]
        per_dev = [r[0] for r in results]
        losses = [l for r in results for l in r[1]]
        stats_t = [s for r in results for s in r[2]]
        rel = cos = 0.0
        host: dict[str, Any] = {}
        for k in PARAM_NAMES:
            parts = [g[k].detach().to("cpu", torch.float64 if g[k].dtype == torch.float64 else torch.float32)
                     for g in per_dev if g is not None]
            total = parts[0].clone()
            for extra in parts[1:]:
                total += extra
            host[k] = total
        loss_total = float(sum(float(l) for l in losses))
        for st in stats_t:
            rel += float(st["rel_sum"])
            cos += float(st["cos_sum"])
        return host, loss_total, {"rel_mean": rel / max(n_total, 1), "cos_mean": cos / max(n_total, 1)}

    def apply(self, host_grads: dict[str, Any], lr: float) -> dict[str, float]:
        import torch

        sq = sum(float(host_grads[k].double().pow(2).sum()) for k in PARAM_NAMES)
        gnorm = math.sqrt(sq)
        if not math.isfinite(gnorm):
            raise Refusal(f"non-finite gradient norm at step {self.t + 1}")
        coef = min(1.0, self.clip / (gnorm + 1e-12)) if self.clip and self.clip > 0 else 1.0
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1.0 - b1 ** self.t, 1.0 - b2 ** self.t
        master = self.replicas[0]
        with torch.no_grad():
            for k in PARAM_NAMES:
                g = host_grads[k].to(device=master[k].device, dtype=master[k].dtype) * coef
                self.m[k].mul_(b1).add_(g, alpha=1.0 - b1)
                self.v[k].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                denom = (self.v[k] / bc2).sqrt_().add_(self.adam_eps)
                master[k].addcdiv_(self.m[k], denom, value=-lr / bc1)
            for rep in self.replicas[1:]:
                for k in PARAM_NAMES:
                    rep[k].copy_(master[k].detach().to("cpu").to(rep[k].device))
        return {"grad_norm": gnorm, "clip_coef": coef}

    def master(self) -> dict[str, Any]:
        return {k: v.detach().to("cpu") for k, v in self.replicas[0].items()}


def lr_at(step: int, total: int, lr: float, warmup: int, min_frac: float) -> float:
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    span = max(total - warmup, 1)
    progress = min(max(step - warmup, 0) / span, 1.0)
    return lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_ids(spec: str, seq_len: int | None, max_rows: int | None,
             lengths_key: str = "lengths") -> tuple[Any, Any, dict[str, Any]]:
    """(ids [rows, L] int64, valid lengths [rows] int64, record).  A .safetensors file may carry ``lengths``
    [rows] (right padding, the qtext_teacher_cache corpus format); otherwise every position is valid."""
    import numpy as np
    import torch

    lengths = None

    path, key = (spec.rsplit("::", 1) + [None])[:2] if "::" in spec else (spec, None)
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(p), "pt") as handle:
            keys = list(handle.keys())
            if key is None:
                cands = [k for k in keys if handle.get_slice(k).get_dtype() in ("I64", "I32", "U32", "I16", "U16")
                         and len(handle.get_slice(k).get_shape()) == 2]
                if "input_ids" in cands:
                    key = "input_ids"
                elif len(cands) == 1:
                    key = cands[0]
                else:
                    raise Refusal(f"{p}: name the ids tensor (FILE::KEY); 2-D integer tensors: {cands}")
            ids = handle.get_tensor(key).to(torch.int64)
            if lengths_key and lengths_key in keys:
                lengths = handle.get_tensor(lengths_key).to(torch.int64).reshape(-1)
    elif p.suffix in (".u32", ".bin"):
        if not seq_len:
            raise Refusal(f"{p}: a raw uint32 id file needs --seq-len")
        raw = np.fromfile(str(p), dtype=np.uint32)
        if raw.size % seq_len:
            raise Refusal(f"{p}: {raw.size} ids are not a multiple of --seq-len {seq_len}")
        ids = torch.from_numpy(raw.astype(np.int64)).view(-1, seq_len)
    elif p.suffix == ".npy":
        ids = torch.from_numpy(np.load(str(p)).astype(np.int64))
    else:
        raise Refusal(f"{p}: ids must be .safetensors[::KEY], .u32 (+ --seq-len) or .npy")
    if ids.ndim != 2:
        raise Refusal(f"{p}: ids must be [rows, seq], got {tuple(ids.shape)}")
    if seq_len and int(ids.shape[1]) != seq_len:
        raise Refusal(f"{p}: rows are {ids.shape[1]} tokens, --seq-len says {seq_len}")
    total_rows = int(ids.shape[0])
    if lengths is None:
        lengths = torch.full((total_rows,), int(ids.shape[1]), dtype=torch.int64)
        lengths_source = "none (every position valid)"
    else:
        lengths_source = f"{lengths_key} tensor"
        if tuple(lengths.shape) != (total_rows,) or int(lengths.min()) < 0 or int(lengths.max()) > int(ids.shape[1]):
            raise Refusal(f"{p}: {lengths_key} must be [rows] within [0, {ids.shape[1]}], got shape {tuple(lengths.shape)} "
                          f"range [{int(lengths.min())}, {int(lengths.max())}]")
    if max_rows:
        ids, lengths = ids[:max_rows], lengths[:max_rows]
    return ids.contiguous(), lengths.contiguous(), {
        "path": str(p), "key": key, "sha256": sha256_file(p), "rows_in_file": total_rows, "rows_used": int(ids.shape[0]),
        "seq_len": int(ids.shape[1]), "lengths": lengths_source, "valid_tokens": int(lengths.sum())}


def parse_layers(text: str | None, n: int) -> list[int]:
    if not text or text == "all":
        return list(range(n))
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise Refusal(f"--layers names text layers outside [0, {n}): {sorted(bad)}")
    return sorted(out)


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def guard_out(out: Path, protected: list[Path], resume: bool) -> None:
    resolved = str(out.resolve())
    for prefix in PROTECTED_OUT_PREFIXES:
        if resolved == prefix or resolved.startswith(prefix + "/"):
            raise Refusal(f"--out {out} is under {prefix} (CP R6: no transient files there)")
    for root in protected:
        if _inside(out, root) or _inside(root, out):
            raise Refusal(f"--out {out} overlaps the model directory {root}")
    if out.exists() and any(out.iterdir()) and not (out / MARKER).is_file():
        raise Refusal(f"--out {out} is a non-empty directory this tool did not create")
    if (out / MARKER).is_file() and not resume and any((out / "rungs").glob("L*.safetensors")):
        raise Refusal(f"--out {out} already holds rungs; pass --resume to continue it or pick a new --out")


def guard_delta_out(delta: Path, out: Path, protected: list[Path]) -> bool:
    """--delta-out gets the --out checks (1.2.4: it had none and could write into a registered checkpoint).
    Inside --out it is covered by --out's guard and marker.  Returns: the directory needs its own marker."""
    if _inside(delta, out):
        return False
    resolved = str(delta.resolve())
    for prefix in PROTECTED_OUT_PREFIXES:
        if resolved == prefix or resolved.startswith(prefix + "/"):
            raise Refusal(f"--delta-out {delta} is under {prefix} (CP R6: no transient files there)")
    for root in protected:
        if _inside(delta, root) or _inside(root, delta):
            raise Refusal(f"--delta-out {delta} overlaps the model directory {root}")
    if delta.exists() and not delta.is_dir():
        raise Refusal(f"--delta-out {delta} exists and is not a directory")
    if delta.exists() and any(delta.iterdir()) and not (delta / MARKER).is_file():
        raise Refusal(f"--delta-out {delta} is a non-empty directory this tool did not create")
    return True


def disk_floor(path: Path, min_free_gib: float, need_bytes: int = 0) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(str(probe)).free
    if (free - need_bytes) / 2 ** 30 < min_free_gib:
        raise Refusal(f"{probe}: {free / 2 ** 30:.1f} GiB free, writing {need_bytes / 2 ** 30:.2f} GiB would cross "
                      f"the --min-free-gib {min_free_gib} floor")
    return free / 2 ** 30


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
@dataclass
class RunState:
    args: argparse.Namespace
    child: Checkpoint
    parent: Checkpoint
    layout: UnionLayout
    ids: Any
    rows_fit: list[int]
    rows_held: list[int]
    seq: int
    P: Any = None                   # parent stream [N, h]
    C: Any = None                   # child stream  [N, h]
    B: Any = None                   # baseline child stream on the KL rows (original weights)
    kl_rows: list[int] = field(default_factory=list)
    receipt: dict[str, Any] = field(default_factory=dict)
    chain: str = ""
    trained: dict[int, dict[str, Any]] = field(default_factory=dict)   # text layer -> trained_entry (1.3.0: no tensors)
    lock: threading.Lock = field(default_factory=threading.Lock)
    devices: list[str] = field(default_factory=list)
    compute_dtype: Any = None
    text_rows: int = TEXT_ROWS
    lengths: list[int] = field(default_factory=list)   # valid positions per row (right padding)
    b_lengths: list[int] = field(default_factory=list)  # valid positions of the KL rows held in B
    prefix: Prefix | None = None
    boundary: BoundaryCache | None = None


def _row_tokens(rows: list[int], seq: int, lengths: list[int] | None = None) -> Any:
    """Flat buffer indices of the rows' tokens (only the first ``lengths[r]`` positions when given)."""
    import torch

    if not rows:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat([torch.arange(r * seq, r * seq + (seq if lengths is None else int(lengths[r]))) for r in rows])


def _valid_mask(lengths: list[int], seq: int, device: str) -> Any:
    import torch

    return torch.arange(seq, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)


def for_row_batches(rs: RunState, buf: Any, rows: list[int], blocks: dict[str, Any],
                    fn: Callable[[Any, Any, Any], Any], lengths: list[int] | None = None) -> bool:
    """Apply fn(block, x, valid) to every row batch of ``buf`` in place; devices in parallel (one thread each).
    ``rows`` index rows of ``buf`` (row r = tokens [r*seq, (r+1)*seq)); ``lengths`` are the valid positions of
    those buffer rows (default: the run's rows).  Returns: every output finite."""
    import torch

    seq = rs.seq
    lengths = rs.lengths if lengths is None else lengths
    devices = list(blocks)
    kind = getattr(next(iter(blocks.values())), "kind", "kda")
    rb = max(1, int(rs.args.row_batch if kind == "kda" else rs.args.attn_row_batch))
    batches = [rows[i:i + rb] for i in range(0, len(rows), rb)]
    finite = [True]

    def work(dev: str, mine: list[list[int]]) -> None:
        blk = blocks[dev]
        for batch in mine:
            lo, hi = batch[0] * seq, (batch[-1] + 1) * seq
            contiguous = batch == list(range(batch[0], batch[-1] + 1))
            if not contiguous:
                raise Refusal("row batches must be contiguous")
            x = buf[lo:hi].view(len(batch), seq, -1).to(device=dev, dtype=rs.compute_dtype)
            valid = _valid_mask([lengths[r] for r in batch], seq, dev)
            with torch.no_grad():
                out = fn(blk, x, valid)
            ok = bool(torch.isfinite(out).all())
            buf[lo:hi].copy_(out.reshape(hi - lo, -1).to("cpu", buf.dtype))
            if not ok:
                with rs.lock:
                    finite[0] = False

    if len(devices) == 1:
        work(devices[0], batches)
    else:
        assigned = {d: batches[i::len(devices)] for i, d in enumerate(devices)}
        with concurrent.futures.ThreadPoolExecutor(len(devices)) as pool:
            for fut in [pool.submit(work, d, assigned[d]) for d in devices]:
                fut.result()
    return finite[0]


class ReadAhead:
    """One background reader: the next layer's block tensors are read while the current layer computes
    (the parent sits on the NAS at ~20 MB/s).  ``get`` returns exactly read_block_state's result."""

    def __init__(self) -> None:
        self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.pending: dict[tuple[str, int], concurrent.futures.Future] = {}

    def prefetch(self, ck: Checkpoint, k: int) -> None:
        key = (str(ck.root), k)
        if key not in self.pending:
            self.pending[key] = self.pool.submit(read_block_state, ck, k)

    def get(self, ck: Checkpoint, k: int) -> tuple[dict[str, Any], dict[str, str]]:
        fut = self.pending.pop((str(ck.root), k), None)
        return fut.result() if fut is not None else read_block_state(ck, k)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()


def blocks_on_devices(ck: Checkpoint, k: int, state: dict[str, Any], devices: list[str], dtype: Any) -> dict[str, Any]:
    return {d: build_block(ck, k, state, d, dtype) for d in devices}


def free_blocks(blocks: dict[str, Any], release_cache: bool = False) -> None:
    """Drop the blocks.  The CUDA cache is KEPT by default: the card must not look free (< 2000 MiB) to the
    other lanes between two layers -- 1.2.3, after a K3 job started on a P40 in such a gap and this run hit OOM."""
    import gc

    import torch

    blocks.clear()
    gc.collect()
    if release_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_charted_identity(rs: RunState, j: int) -> dict[str, Any]:
    """A charted media block writes exact zeros on rows it does not own: on a text-only stream it is identity."""
    import torch

    state, _ = read_block_state(rs.child, j)
    dev = rs.devices[0]
    blk = build_block(rs.child, j, state, dev, rs.compute_dtype)
    x = rs.C[: rs.seq].view(1, rs.seq, -1).to(device=dev, dtype=rs.compute_dtype)
    valid = _valid_mask(rs.lengths[:1], rs.seq, dev)
    with torch.no_grad():
        out = run_block(blk, x, valid)
    same = bool(torch.equal(out, x))
    del blk, state
    if not same:
        raise Refusal(f"charted block {j} is not identity on text-only rows (max |diff| {float((out - x).abs().max())})")
    return {"block": j, "identity_on_text_rows": True, "checked_tokens": int(rs.lengths[0])}


def train_layer(rs: RunState, i: int, j: int, base_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Layer i (child block j): x_m already in rs.C, y_p(i) in rs.P.  Returns (stored tensors, record)."""
    import torch

    a = rs.args
    lay = rs.layout
    eps = float(rs.child.cfg.rms_norm_eps)
    init = {k: v.to(torch.float32) for k, v in trunk_slices(base_state, lay).items()}
    k3 = {k: v.to(torch.float32) for k, v in k3_slices(base_state, lay).items()}
    fit_tokens = _row_tokens(rs.rows_fit, rs.seq, rs.lengths)
    held_tokens = _row_tokens(rs.rows_held, rs.seq, rs.lengths)
    g = torch.Generator().manual_seed(int(a.seed) * 1000003 + i)
    diag_fit = fit_tokens[torch.randperm(fit_tokens.numel(), generator=g)[: int(a.diag_tokens)]].sort().values
    diag_held = held_tokens[: int(a.diag_tokens)] if held_tokens.numel() else diag_fit
    dev0 = rs.devices[0]
    before = {"held": layer_diagnostics(rs.C, rs.P, diag_held, init, k3, eps, dev0),
              "fit_sample": layer_diagnostics(rs.C, rs.P, diag_fit, init, k3, eps, dev0)}
    mm_dtype = None if a.train_dtype == "float32" else getattr(torch, a.train_dtype)
    master_dtype = torch.float64 if a.master_dtype == "float64" else torch.float32
    trainer = TrunkTrainer(init, rs.devices, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)), eps=float(a.adam_eps),
                           clip=float(a.clip), norm_eps=eps, residual=a.loss_site == "residual", mm_dtype=mm_dtype,
                           master_dtype=master_dtype)
    n_fit = int(fit_tokens.numel())
    per_step = max(1, int(a.tokens_per_step))
    order = torch.cat([fit_tokens[torch.randperm(n_fit, generator=g)] for _ in range(max(1, int(a.epochs)))]) \
        if a.epochs > 0 else torch.zeros(0, dtype=torch.long)
    total_steps = math.ceil(order.numel() / per_step) if order.numel() else 0
    if a.max_steps:
        total_steps = min(total_steps, int(a.max_steps))
    curve, started = [], time.time()
    D = len(rs.devices)
    micro = max(1, int(a.micro_tokens))
    for step in range(total_steps):
        idx = order[step * per_step:(step + 1) * per_step]
        n = int(idx.numel())
        bounds = [round(n * d / D) for d in range(D + 1)]
        shards = []
        for d in range(D):   # lazy gathers: each runs on its device's thread, overlapping that card's compute
            part = idx[bounds[d]:bounds[d + 1]]
            shards.append([(lambda sel=part[m:m + micro]: (rs.C.index_select(0, sel), rs.P.index_select(0, sel)))
                           for m in range(0, int(part.numel()), micro)])
        host, loss, st = trainer.shard_grads(shards, n)
        if not math.isfinite(loss):
            raise Refusal(f"layer {i}: non-finite D1 loss at step {step + 1}")
        lr = lr_at(step, total_steps, float(a.lr), int(a.warmup_steps), float(a.min_lr_frac))
        upd = trainer.apply(host, lr)
        progress(i, f"train step {step + 1}/{total_steps}")
        if step == 0 or (step + 1) % max(1, int(a.log_every)) == 0 or step + 1 == total_steps:
            curve.append({"step": step + 1, "loss": loss, "rel_mean": st["rel_mean"], "cos_mean": st["cos_mean"],
                          "lr": lr, **upd, "tokens": n, "seconds": round(time.time() - started, 2)})
            log(f"  L{i:02d} step {step + 1}/{total_steps} loss {loss:.6f} rel {st['rel_mean']:.5f} "
                f"cos {st['cos_mean']:.6f} |g| {upd['grad_norm']:.3e} lr {lr:.2e}")
    master = trainer.master()
    stored = {k: master[k].to(rs.child.dtype_on_disk).contiguous() for k in PARAM_NAMES}
    for k in PARAM_NAMES:
        if not bool(torch.isfinite(stored[k].float()).all()):
            raise Refusal(f"layer {i}: trained {k} carries non-finite values")
    after_w = {k: v.to(torch.float32) for k, v in stored.items()}
    after = {"held": layer_diagnostics(rs.C, rs.P, diag_held, after_w, k3, eps, dev0),
             "fit_sample": layer_diagnostics(rs.C, rs.P, diag_fit, after_w, k3, eps, dev0)}
    dp_threaded = trainer.parallel
    del trainer
    record = {
        "text_layer": i, "child_block": j, "steps": total_steps, "tokens_per_step": per_step, "fit_tokens": n_fit,
        "epochs": a.epochs, "train_seconds": round(time.time() - started, 2), "loss_curve": curve,
        "dp": {"devices": list(rs.devices), "threaded": bool(dp_threaded)},
        "before": before, "after": after,
        "weight_change_relL2": {k: float((after_w[k] - init[k]).norm() / init[k].norm().clamp_min(TINY)) for k in PARAM_NAMES},
        "diag_tokens": {"held": int(diag_held.numel()), "fit_sample": int(diag_fit.numel()),
                        "held_is_fit_sample": not bool(held_tokens.numel())},
    }
    return stored, record


def write_rung(rs: RunState, i: int, j: int, stored: dict[str, Any], record: dict[str, Any],
               base_shas: dict[str, str]) -> dict[str, Any]:
    rung_dir = Path(rs.args.out).absolute() / "rungs"
    rung_dir.mkdir(parents=True, exist_ok=True)
    tensors, entries = rung_tensors(rs, j, stored, base_shas)
    need = sum(int(t.numel()) * t.element_size() for t in tensors.values())
    disk_floor(rung_dir, float(rs.args.min_free_gib), need)
    path = rung_dir / f"L{i:02d}.safetensors"
    save_safetensors_deterministic(tensors, path, {"tool": f"{TOOL_NAME} {TOOL_VERSION}", "schema": RUNG_SCHEMA})
    doc = {"schema": RUNG_SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION}, "text_layer": i,
           "child_block": j, "input_chain": rs.chain, "file": str(path), "file_sha256": sha256_file(path),
           "entries": entries, "record": record}
    write_json_atomic(rung_dir / f"L{i:02d}.json", doc)
    return doc


def rung_tensors(rs: RunState, j: int, stored: dict[str, Any], base_shas: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h = rs.child.hidden
    tensors, entries = {}, []
    for base_name, part, key in DELTA_PARTS:
        name = f"blocks.{j}.{base_name}"
        dkey = f"{name}::{part}"
        t = stored[key].contiguous()
        tensors[dkey] = t
        entries.append({"key": dkey, "base_tensor": name, "part": part, "slice": delta_slice(part, rs.layout, h),
                        "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                        "sha256": tensor_bytes_sha256(t), "base_tensor_sha256": base_shas.get(name)})
    return tensors, entries


def load_rung(rs: RunState, i: int, j: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    from safetensors.torch import load_file

    rung_dir = Path(rs.args.out).absolute() / "rungs"
    path, jpath = rung_dir / f"L{i:02d}.safetensors", rung_dir / f"L{i:02d}.json"
    if not (path.is_file() and jpath.is_file()):
        return None
    doc = {**json.loads(jpath.read_text(encoding="utf-8")), "file": str(path)}
    if doc.get("input_chain") != rs.chain:
        log(f"  L{i:02d}: rung on disk was made from a different input chain; retraining")
        return None
    if sha256_file(path) != doc.get("file_sha256"):
        raise Refusal(f"{path}: sha256 does not match its record")
    got = load_file(str(path))
    stored = {}
    for base_name, part, key in DELTA_PARTS:
        stored[key] = got[f"blocks.{j}.{base_name}::{part}"]
    return stored, doc


def trained_entry(j: int, base_shas: dict[str, str], path: Path | str, file_sha256: str) -> dict[str, Any]:
    """rs.trained[i] (1.3.0): where the layer's stored tensors live, not the tensors (8.6 GB at L63)."""
    return {"block": j, "base_shas": dict(base_shas), "file": str(path), "file_sha256": file_sha256}


def trained_tensors(rs: RunState, i: int) -> dict[str, Any]:
    """Text layer i's stored tensors, re-read from its rung file (sha256 checked against the run's record)."""
    from safetensors.torch import load_file

    info = rs.trained[i]
    if sha256_file(info["file"]) != info["file_sha256"]:
        raise Refusal(f"{info['file']}: the rung file changed during the run")
    got = load_file(str(info["file"]))
    j = info["block"]
    return {key: got[f"blocks.{j}.{base_name}::{part}"] for base_name, part, key in DELTA_PARTS}


# --------------------------------------------------------------------------
# 1.3.0: the canonical rung prefix (d1_canonical_rungs.py) and the boundary cache
# --------------------------------------------------------------------------
@dataclass
class Prefix:
    root: Path
    layers: int                        # imported text layers L00..L{layers-1}
    sums_sha256: str
    shas: dict[int, dict[str, str]]    # layer -> {"safetensors": sha, "json": sha}

    def record(self) -> dict[str, Any]:
        return {"dir": str(self.root), "layers": self.layers, "sha256sums_sha256": self.sums_sha256,
                "rung_sha256": [self.shas[i]["safetensors"] for i in range(self.layers)]}


def open_prefix(args: argparse.Namespace, n_text: int) -> Prefix | None:
    if not args.prefix_dir:
        if args.prefix_sums_sha256 or args.prefix_layers:
            raise Refusal("--prefix-sums-sha256 / --prefix-layers need --prefix-dir")
        return None
    root = Path(args.prefix_dir).absolute()
    sums = root / PREFIX_SUMS
    if not args.prefix_sums_sha256 or args.prefix_layers is None:
        raise Refusal("--prefix-dir needs --prefix-sums-sha256 and --prefix-layers (the launcher pins both)")
    if not sums.is_file() or sha256_file(sums) != args.prefix_sums_sha256:
        raise Refusal(f"{sums} is missing or does not hash to --prefix-sums-sha256 {args.prefix_sums_sha256}")
    shas: dict[int, dict[str, str]] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            stem, _, kind = name.strip().partition(".")
            shas.setdefault(int(stem[1:]), {})[kind] = digest
    n = int(args.prefix_layers)
    if not 0 <= n <= n_text or any(set(shas.get(i, {})) != {"safetensors", "json"} for i in range(n)):
        raise Refusal(f"--prefix-layers {n}: {sums} does not list L00..L{n - 1:02d} (has {sorted(shas)})")
    return Prefix(root, n, args.prefix_sums_sha256, shas)


def import_prefix_rung(rs: RunState, i: int, j: int, base_shas: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical rung L{i}: both files hash to SHA256SUMS, the record names (i, j) and this child's base tensors."""
    from safetensors.torch import load_file

    pre = rs.prefix
    path, jpath = pre.root / f"L{i:02d}.safetensors", pre.root / f"L{i:02d}.json"
    if sha256_file(jpath) != pre.shas[i]["json"]:
        raise Refusal(f"{jpath} does not hash to its {PREFIX_SUMS} line")
    doc = json.loads(jpath.read_text(encoding="utf-8"))
    if sha256_file(path) != pre.shas[i]["safetensors"] or doc.get("file_sha256") != pre.shas[i]["safetensors"]:
        raise Refusal(f"{path} does not hash to its {PREFIX_SUMS} line / record")
    if int(doc.get("text_layer", -1)) != i or int(doc.get("child_block", -1)) != j:
        raise Refusal(f"{jpath}: text layer {doc.get('text_layer')} / child block {doc.get('child_block')}, expected {i} / {j}")
    for entry in doc["entries"]:
        if entry["base_tensor_sha256"] != base_shas.get(entry["base_tensor"]):
            raise Refusal(f"{jpath}: {entry['base_tensor']} was trained on another base tensor")
    got = load_file(str(path))
    stored = {key: got[f"blocks.{j}.{base_name}::{part}"] for base_name, part, key in DELTA_PARTS}
    doc = {**doc, "file": str(path), "imported_from_prefix": {"dir": str(pre.root), "sha256sums_sha256": pre.sums_sha256,
                                                             "json_sha256": pre.shas[i]["json"]}}
    return stored, doc


def _stream_bytes(t: Any) -> Any:
    """A writable uint8 numpy view of a contiguous CPU tensor's memory (no copy)."""
    import torch

    return t.reshape(-1).view(torch.uint8).numpy()


BOUNDARY_CHUNK = 1 << 28          # 256 MiB per write / read / hash update
BOUNDARY_INFLIGHT = 8             # chunks whose writeback may still be running (dirty cache <= ~2.25 GiB)
_SFR_WAIT_BEFORE, _SFR_WRITE, _SFR_WAIT_AFTER = 1, 2, 4


def _sync_file_range() -> Callable[[int, int, int, int], int] | None:
    """libc sync_file_range (Linux), or None: start / await writeback of a byte range without a full fdatasync."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        fn = libc.sync_file_range
        fn.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_uint]
        fn.restype = ctypes.c_int
        return fn
    except (OSError, AttributeError):
        return None


def write_stream_file(t: Any, path: Path) -> dict[str, Any]:
    """The tensor's bytes to ``path`` with their sha256; durable (fsync) and out of the page cache when done.
    Writeback is pipelined: each 256 MiB chunk's writeback is started at once (sync_file_range WRITE), the chunk
    two behind is awaited and its pages dropped -- the device stays busy while the run's cgroup never holds more
    than ~3 chunks of dirty cache.  (A synchronous fdatasync per GiB measured 108.7 MB/s on the exFAT Lexar, dd
    with one final sync 192 MB/s, 2026-10-14.)  Without sync_file_range: fdatasync per chunk."""
    mv = _stream_bytes(t)
    digest = hashlib.sha256()
    sfr = _sync_file_range()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    started = time.time()
    pending: list[tuple[int, int]] = []

    def settle(lo: int, n: int) -> None:
        if sfr is None or sfr(fd, lo, n, _SFR_WAIT_BEFORE | _SFR_WRITE | _SFR_WAIT_AFTER) != 0:
            os.fdatasync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, lo, n, os.POSIX_FADV_DONTNEED)

    try:
        for lo in range(0, mv.nbytes, BOUNDARY_CHUNK):
            part = mv[lo:lo + BOUNDARY_CHUNK]
            digest.update(part)
            view = memoryview(part)
            while view.nbytes:
                n = os.write(fd, view)
                view = view[n:]
            if sfr is not None:
                sfr(fd, lo, part.nbytes, _SFR_WRITE)
            pending.append((lo, part.nbytes))
            while len(pending) > BOUNDARY_INFLIGHT:
                settle(*pending.pop(0))
        while pending:
            settle(*pending.pop(0))
        os.fsync(fd)
    finally:
        os.close(fd)
    seconds = time.time() - started
    return {"file": path.name, "bytes": int(mv.nbytes), "sha256": digest.hexdigest(), "seconds": round(seconds, 2),
            "mb_per_s": round(mv.nbytes / 1e6 / max(seconds, 1e-9), 1)}


def read_stream_file(t: Any, path: Path, expect: dict[str, Any]) -> bool:
    """Fill the tensor's memory from ``path``; True when size and sha256 match ``expect``."""
    mv = _stream_bytes(t)
    if path.stat().st_size != int(expect["bytes"]) or mv.nbytes != int(expect["bytes"]):
        return False
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        for lo in range(0, mv.nbytes, BOUNDARY_CHUNK):
            part = mv[lo:lo + BOUNDARY_CHUNK]
            view, filled = memoryview(part), 0
            while filled < part.nbytes:
                n = handle.readinto(view[filled:])
                if not n:
                    return False
                filled += n
            digest.update(part)
            if hasattr(os, "posix_fadvise"):   # 1.3.1: the chunk's clean cache leaves the cgroup now, not at the end
                os.posix_fadvise(handle.fileno(), lo, part.nbytes, os.POSIX_FADV_DONTNEED)
    drop_file_cache([path])
    return digest.hexdigest() == expect["sha256"]


class BoundaryCache:
    """DIR/L{i:02d}/ {P.bin, C.bin, [B.bin], boundary.json} + DIR/LATEST.json; one boundary kept.

    No write blocks the run.  ``begin(i)`` right after layer i's parent forward: P = y_p(i) is final and nothing
    touches it until layer i+1's parent forward (~17 min later: child attention, training, advance), so P is written
    on a thread.  ``finish(i)`` after layer i's rung is committed and C advanced: P's thread is joined (long done),
    B (0.17 GB) is written, then C on a thread -- C is untouched until layer i+1's child pass, after its ~8 min
    parent forward -- which then seals the boundary.  ``wait`` joins that thread and must run before C changes.  A
    boundary becomes LATEST only after every file is fsynced and hashed; the older one is then removed.  A failed or
    skipped write is reported, never fatal: it only costs a recompute on restart.  Measured 2026-10-14 on the exFAT
    Lexar: 108-139 MB/s write, 123-145 MB/s read through these functions; one OMNI1 boundary is 82.3 GB."""

    def __init__(self, root: Path, min_free_gib: float) -> None:
        self.root = root
        self.min_free_gib = min_free_gib
        self.thread: threading.Thread | None = None      # C writer + sealing
        self.p_thread: threading.Thread | None = None    # P writer
        self.p_box: dict[str, Any] = {}
        self.tmp: Path | None = None
        self.tmp_layer: int | None = None
        self.started = 0.0
        self.result: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def key(self, rs: RunState, i: int) -> dict[str, Any]:
        rungs = [[k, rs.trained[k]["file_sha256"]] for k in sorted(rs.trained)]
        return {"run_identity_sha256": rs.receipt["run_identity_sha256"], "layer": i, "chain_after": rs.chain,
                "rungs": rungs, "rungs_sha256": sha256_json(rungs)}

    def _drop_tmp(self) -> None:
        if self.tmp is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp, self.tmp_layer, self.p_thread, self.p_box = None, None, None, {}

    def begin(self, rs: RunState, i: int) -> dict[str, Any]:
        """After layer i's parent forward: the P file starts on a thread (disk floor checked for the whole set)."""
        self.wait()
        self.abort()
        streams = [rs.P, rs.C] + ([rs.B] if rs.B is not None else [])
        need = sum(int(t.numel()) * t.element_size() for t in streams)
        try:
            disk_floor(self.root, self.min_free_gib, need)
        except Refusal as error:
            rec = {"layer": i, "written": False, "skipped": str(error)}
            log(f"  boundary L{i:02d} SKIPPED: {error}")
            self.history.append(rec)
            return rec
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".tmp-L{i:02d}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        self.tmp, self.tmp_layer, self.p_box, self.started = tmp, i, {}, time.time()

        def write_p() -> None:
            try:
                self.p_box["P"] = write_stream_file(rs.P, tmp / "P.bin")
            except Exception as error:  # noqa: BLE001 - reported by finish
                self.p_box["error"] = repr(error)

        self.p_thread = threading.Thread(target=write_p, name=f"d1-boundary-P-L{i:02d}", daemon=False)
        self.p_thread.start()
        return {"layer": i, "pending": True, "begun": "P"}

    def abort(self) -> None:
        """Drop an unfinished boundary (its layer did not reach finish)."""
        if self.p_thread is not None:
            self.p_thread.join()
        self._drop_tmp()

    def finish(self, rs: RunState, i: int) -> dict[str, Any] | None:
        """After layer i's rung is committed and C advanced: join P, write B, start C + sealing on a thread."""
        if self.tmp is None or self.tmp_layer != i:
            return None
        waited = time.time()
        self.p_thread.join()
        waited = round(time.time() - waited, 2)
        tmp, box = self.tmp, dict(self.p_box)
        if "error" in box:
            rec = {"layer": i, "written": False, "error": box["error"]}
            log(f"  boundary L{i:02d} NOT written (P): {box['error']}")
            self.history.append(rec)
            self._drop_tmp()
            return rec
        streams = [("P", rs.P), ("C", rs.C)] + ([("B", rs.B)] if rs.B is not None else [])
        doc: dict[str, Any] = {"schema": BOUNDARY_SCHEMA, "tool": [TOOL_NAME, TOOL_VERSION], "key": self.key(rs, i),
                               "b_lengths": rs.b_lengths, "kl_rows": rs.kl_rows, "files": {"P": box["P"]},
                               "streams": {name: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
                                           for name, t in streams},
                               "finish_waited_for_p_seconds": waited}
        try:
            if rs.B is not None:
                doc["files"]["B"] = write_stream_file(rs.B, tmp / "B.bin")
        except Exception as error:  # noqa: BLE001
            rec = {"layer": i, "written": False, "error": repr(error)}
            self.history.append(rec)
            self._drop_tmp()
            return rec
        self.tmp, self.tmp_layer, self.p_thread, self.p_box = None, None, None, {}
        started = self.started
        self.result = {"layer": i, "written": False, "pending": True}

        def seal() -> None:
            try:
                doc["files"]["C"] = write_stream_file(rs.C, tmp / "C.bin")
                doc["bytes"] = sum(f["bytes"] for f in doc["files"].values())
                doc["seconds"] = round(time.time() - started, 2)
                doc["written_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                write_json_atomic(tmp / "boundary.json", doc)
                final = self.root / f"L{i:02d}"
                shutil.rmtree(final, ignore_errors=True)
                os.replace(tmp, final)
                write_json_atomic(self.root / "LATEST.json", {"dir": final.name, "layer": i,
                                                              "boundary_sha256": sha256_file(final / "boundary.json")})
                for old in self.root.iterdir():   # keep only the latest (a newer begin() cannot run before wait())
                    if old.is_dir() and old.name != final.name and (old.name.startswith("L") or old.name.startswith(".tmp-")):
                        shutil.rmtree(old, ignore_errors=True)
                self.result = {"layer": i, "written": True, "dir": str(final), "bytes": doc["bytes"],
                               "seconds": doc["seconds"], "finish_waited_for_p_seconds": waited, "files": doc["files"]}
            except Exception as error:  # noqa: BLE001 - a failed cache costs a recompute, never the run
                shutil.rmtree(tmp, ignore_errors=True)
                self.result = {"layer": i, "written": False, "error": repr(error)}

        self.thread = threading.Thread(target=seal, name=f"d1-boundary-C-L{i:02d}", daemon=False)
        self.thread.start()
        return {"layer": i, "pending": True, "begun": "C", "finish_waited_for_p_seconds": waited,
                "sync_files": dict(doc["files"])}

    def wait(self) -> dict[str, Any] | None:
        if self.thread is None:
            return None
        self.thread.join()
        self.thread = None
        rec = dict(self.result)
        self.history.append(rec)
        if rec.get("written"):
            log(f"  boundary L{rec['layer']:02d} written: {rec['bytes'] / 1e9:.2f} GB, {rec['seconds']:.0f}s from begin, "
                f"finish waited {rec['finish_waited_for_p_seconds']}s for P "
                f"({', '.join(f'{k} {v['mb_per_s']} MB/s' for k, v in rec['files'].items())}) -> {rec['dir']}")
        else:
            log(f"  boundary L{rec['layer']:02d} NOT written: {rec.get('error')}")
        return rec

    def load(self, rs: RunState, expected_rungs: Callable[[int], list[list[Any]] | None]) -> dict[str, Any] | None:
        """Fill rs.P / rs.C / rs.B from the LATEST boundary when its key is this run's; None (with a log line)
        otherwise.  ``expected_rungs(i)`` gives the rung list this run would advance layers 0..i through."""
        latest = self.root / "LATEST.json"
        if not latest.is_file():
            return None
        try:
            ptr = json.loads(latest.read_text(encoding="utf-8"))
            bdir = self.root / ptr["dir"]
            if sha256_file(bdir / "boundary.json") != ptr["boundary_sha256"]:
                raise ValueError("boundary.json does not hash to LATEST.json")
            doc = json.loads((bdir / "boundary.json").read_text(encoding="utf-8"))
            key, i = doc["key"], int(doc["key"]["layer"])
            if doc.get("schema") != BOUNDARY_SCHEMA or key["run_identity_sha256"] != rs.receipt["run_identity_sha256"]:
                raise ValueError("another run identity")
            want = expected_rungs(i)
            if want is None or want != key["rungs"] or sha256_json(want) != key["rungs_sha256"]:
                raise ValueError(f"rung list through L{i:02d} differs from this run's")
            names = {"P", "C"} | ({"B"} if rs.B is not None else set())
            if set(doc["files"]) != names or doc["kl_rows"] != rs.kl_rows:
                raise ValueError(f"streams {sorted(doc['files'])} / KL rows differ from this run's {sorted(names)}")
            for name in sorted(names):
                t = getattr(rs, name)
                if doc["streams"][name] != {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}:
                    raise ValueError(f"{name} shape/dtype differs")
            started = time.time()
            for name in sorted(names):
                if not read_stream_file(getattr(rs, name), bdir / f"{name}.bin", doc["files"][name]):
                    raise ValueError(f"{name}.bin does not hash to its record")
        except (OSError, ValueError, KeyError, TypeError) as error:
            log(f"boundary cache {self.root} not used: {error}")
            return None
        seconds = time.time() - started
        log(f"boundary L{i:02d} loaded and verified: {doc['bytes'] / 1e9:.2f} GB in {seconds:.0f}s from {bdir}")
        return {"layer": i, "dir": str(bdir), "key": key, "b_lengths": doc["b_lengths"], "bytes": doc["bytes"],
                "load_seconds": round(seconds, 2), "written_at_utc": doc.get("written_at_utc")}


def head_kl(rs: RunState) -> dict[str, Any]:
    """Frozen-head logit KL(parent || child) and top-1 agreement over vocab rows [0, text_rows)."""
    import torch
    import torch.nn.functional as F
    from chimera_omni.importing.safetensors_io import read_safetensors
    from chimera_omni.modeling.layers import RMSNorm

    dev = rs.devices[0]
    dt = rs.compute_dtype
    rows = rs.text_rows

    def head(ck: Checkpoint) -> tuple[Any, Any]:
        names = ["norm.weight", "lm_head.weight" if "lm_head.weight" in ck.index["weight_map"] else "embed.weight"]
        views = read_safetensors(ck.root, names, verify=True)
        norm = RMSNorm(ck.hidden, ck.cfg.rms_norm_eps).to(device=dev, dtype=dt)
        with torch.no_grad():
            norm.weight.copy_(views["norm.weight"].torch(device="cpu").to(device=dev, dtype=dt))
        w = views[names[1]].torch(device="cpu")[:rows].to(device=dev, dtype=dt)
        del views
        return norm, w

    # B holds the KL rows back to back (all positions); keep the valid ones and their local B index
    tokens = _row_tokens(rs.kl_rows, rs.seq, rs.lengths)
    local_all = _row_tokens(list(range(len(rs.kl_rows))), rs.seq, [rs.lengths[r] for r in rs.kl_rows])
    p_norm, p_w = head(rs.parent)
    c_norm, c_w = head(rs.child)
    acc = {"d1": {"kl": 0.0, "agree": 0, "finite": True}, "original": {"kl": 0.0, "agree": 0, "finite": True}}
    n = 0
    chunk = int(rs.args.kl_chunk)
    with torch.no_grad():
        for lo in range(0, int(tokens.numel()), chunk):
            idx = tokens[lo:lo + chunk]
            xp = rs.P.index_select(0, idx).to(device=dev, dtype=dt)
            lp = F.log_softmax(F.linear(p_norm(xp), p_w).float(), dim=-1)
            top_p = lp.argmax(-1)
            streams = [("d1", rs.C.index_select(0, idx))]
            if rs.B is not None:
                streams.append(("original", rs.B.index_select(0, local_all[lo:lo + int(idx.numel())])))
            for name, xc in streams:
                lc = F.log_softmax(F.linear(c_norm(xc.to(device=dev, dtype=dt)), c_w).float(), dim=-1)
                acc[name]["finite"] = acc[name]["finite"] and bool(torch.isfinite(lc).all())
                acc[name]["kl"] += float((lp.exp() * (lp - lc)).sum(-1).double().sum())
                acc[name]["agree"] += int((lc.argmax(-1) == top_p).sum())
                del lc
            n += int(idx.numel())
            del lp
    out = {"positions": n, "vocab_rows": [0, rows], "rows": rs.kl_rows,
           "definition": "KL(parent || child) of fp32 log-softmax over rows [0, text_rows), frozen heads; top-1 = argmax agreement"}
    for name, a in acc.items():
        if name == "original" and rs.B is None:
            continue
        out[name] = {"kl_mean": a["kl"] / max(n, 1), "top1_agreement": a["agree"] / max(n, 1), "logits_finite": a["finite"]}
    del p_w, c_w
    return out


def base_digest(ck: Checkpoint, mode: str) -> dict[str, Any]:
    from chimera_omni.importing.safetensors_io import shard_paths

    doc = ck.identity()
    files = {}
    for p in shard_paths(ck.root):
        st = p.stat()
        files[p.name] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        if mode == "full":
            files[p.name]["sha256"] = sha256_file(p)
            drop_file_cache([p])   # 1.3.0: the 55 GB digest read leaves no page cache in the run's cgroup
    doc["files"] = files
    doc["mode"] = mode
    return doc


def write_delta(rs: RunState, base: dict[str, Any]) -> dict[str, Any] | None:
    if not rs.trained:
        return None
    out = Path(rs.args.delta_out).absolute() if rs.args.delta_out else Path(rs.args.out) / "delta"
    out.mkdir(parents=True, exist_ok=True)
    tensors, entries = {}, []
    for i in sorted(rs.trained):
        info = rs.trained[i]
        t, e = rung_tensors(rs, info["block"], trained_tensors(rs, i), info["base_shas"])
        tensors.update(t)
        for item in e:
            item["text_layer"] = i
        entries.extend(e)
    need = sum(int(t.numel()) * t.element_size() for t in tensors.values())
    disk_floor(out, float(rs.args.min_free_gib), need)
    path = out / "D1_delta.safetensors"
    save_safetensors_deterministic(tensors, path, {"tool": f"{TOOL_NAME} {TOOL_VERSION}", "schema": DELTA_SCHEMA})
    manifest = {"schema": DELTA_SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__)},
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "base": base, "union_layout": rs.layout.__dict__, "trained_text_layers": sorted(rs.trained),
                "parameters": int(sum(int(t.numel()) for t in tensors.values())),
                "file": str(path), "file_sha256": sha256_file(path), "bytes": path.stat().st_size,
                "entries": entries, "reader": f"{TOOL_NAME}.apply_delta (slice overwrite of the base tensor)"}
    write_json_atomic(out / "D1_delta.json", manifest, with_sha=True)
    return {"dir": str(out), "file": str(path), "file_sha256": manifest["file_sha256"], "bytes": manifest["bytes"],
            "parameters": manifest["parameters"], "manifest_sha256": sha256_file(out / "D1_delta.json")}


def save_receipt(rs: RunState) -> None:
    write_json_atomic(Path(rs.args.out) / "d1_receipt.json", rs.receipt, with_sha=True)


def preregistration_record(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind production runs to the frozen body and any dated amendments (1.2.10).  Small standalone experiments
    can omit the argument; the production launcher always supplies it.  Verified before any checkpoint is opened
    or any output directory is created."""
    if not args.preregistration:
        return None
    import stage2_training_receipt as s2
    try:
        record = s2.assert_preregistration_unedited(Path(args.preregistration))
    except s2.Stage2TrainingReceiptError as exc:
        raise Refusal(f"preregistration refused: {exc}") from exc
    return {**record, "path": str(Path(args.preregistration).resolve())}


def source_contract_record(args: argparse.Namespace) -> dict[str, Any] | None:
    """1.2.10's successor dataset contract (d1_source_contract.py, unchanged, its own sha pinned by
    --source-validator-sha256): every source pin, the exact real lengths and the fit / held split, verified
    before any checkpoint is opened.  None when no --source-* flag is given (tests, experiments)."""
    if not (args.source_manifest or args.source_manifest_sha256 or args.source_validator_sha256):
        return None
    import d1_source_contract as source_contract
    try:
        return source_contract.validate_args(args, Path(__file__))
    except ValueError as exc:
        raise Refusal(f"source contract refused: {exc}") from exc


def boundary_due(i: int, stop: int, trained_now: bool, next_replays: bool, last: int, every: int) -> bool:
    """Write the boundary after layer i?  Never after the last layer (nothing follows); after a replayed layer only
    when the next one has nothing to replay (the end of the prefix / of the run's own rungs: a long replay is not
    paid twice); after a newly trained layer when ``every`` layers have passed since the last boundary (1: every
    layer, the state doc's rule; a larger value trades boundary writes for replays after a crash)."""
    if i >= stop:
        return False
    if not trained_now:
        return not next_replays
    return i - last >= max(1, every)


def _layer_memory(rs: RunState, i: int, cache_paths: list[Path], extra: list[Path]) -> dict[str, Any]:
    """Per-layer host memory, before and after freeing host buffers (logged + recorded)."""
    before = host_memory()
    freed = free_host_buffers(cache_paths + extra)
    after = host_memory()
    log(f"L{i:02d} memory: {memory_line(after)} (before free: rss {before.get('rss')} cgroup "
        f"{before.get('cgroup', {}).get('current')}; freed {freed})")
    return {"before_free": before, "freed": freed, "after_free": after}


def run(argv: list[str] | None = None) -> RunState:
    args = build_parser().parse_args(argv)
    progress(None, "admission")
    preregistration = preregistration_record(args)
    source_record = source_contract_record(args)
    started = time.time()
    tree = bind_chimera()
    import torch
    from chimera_omni.importing.safetensors_io import shard_paths

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    out = Path(args.out).absolute()
    child = open_checkpoint(args.child)
    parent = open_checkpoint(args.parent)
    guard_out(out, [child.root, parent.root], bool(args.resume))
    delta_dir = Path(args.delta_out).absolute() if args.delta_out else None
    delta_marker = guard_delta_out(delta_dir, out, [child.root, parent.root]) if delta_dir is not None else False
    if child.hidden != parent.hidden:
        raise Refusal(f"hidden {child.hidden} (child) != {parent.hidden} (parent)")
    if parent.charted_blocks:
        raise Refusal("the parent carries charted blocks")
    if len(child.text_blocks) != len(parent.text_blocks):
        raise Refusal(f"child has {len(child.text_blocks)} text blocks, the parent {len(parent.text_blocks)}")
    for i, (j, k) in enumerate(zip(child.text_blocks, parent.text_blocks)):
        if child.cfg.block_kind(j) != parent.cfg.block_kind(k):
            raise Refusal(f"text layer {i}: child block {j} is {child.cfg.block_kind(j)}, parent block {k} {parent.cfg.block_kind(k)}")
    if str(child.cfg.activation_kind) != "silu":
        raise Refusal(f"the child FFN activation is {child.cfg.activation_kind!r}; D1's gradient is SiLU-GLU (c10 lifts K3 to SiLU)")
    layout = union_layout(child, args.trunk_width, args.k3_channels)
    text_rows = min(TEXT_ROWS, int(child.cfg.ledger_band_begin()), int(parent.cfg.ledger_band_begin()))
    ids, lengths, ids_record = load_ids(args.ids, args.seq_len, args.max_rows, args.lengths_key)
    R, L = int(ids.shape[0]), int(ids.shape[1])
    valid_pos = torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)
    valid_ids = ids[valid_pos]
    if valid_ids.numel() == 0 or int(valid_ids.min()) < 0 or int(valid_ids.max()) >= text_rows:
        raise Refusal(f"ids outside the text rows [0, {text_rows}) (or no valid position): "
                      f"min {int(valid_ids.min()) if valid_ids.numel() else None} "
                      f"max {int(valid_ids.max()) if valid_ids.numel() else None}")
    ids = torch.where(valid_pos, ids, torch.zeros_like(ids))   # padding ids are never used: id 0, zeroed below
    held_n = int(args.held_rows) if args.held_rows is not None else max(1, R // 16)
    if held_n >= R:
        raise Refusal(f"--held-rows {held_n} leaves no fit rows of {R}")
    rows_fit, rows_held = list(range(R - held_n)), list(range(R - held_n, R))
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(set(devices)) != len(devices) and not all(d == "cpu" for d in devices):
        raise Refusal(f"--devices repeats a card: {devices}")
    n_text = len(child.text_blocks)
    train_layers = set(parse_layers(args.layers, n_text))
    stop = int(args.stop_after_layer) if args.stop_after_layer is not None else n_text - 1
    if not 0 <= stop < n_text:
        raise Refusal(f"--stop-after-layer {stop} outside [0, {n_text})")
    prefix = open_prefix(args, n_text)
    if prefix is not None and any(i not in train_layers for i in range(prefix.layers)):
        raise Refusal(f"--prefix-layers {prefix.layers} imports layers that --layers {args.layers!r} does not train")
    compute_dtype = getattr(torch, args.dtype)
    buffer_dtype = getattr(torch, args.buffer_dtype or args.dtype)

    rs = RunState(args, child, parent, layout, ids, rows_fit, rows_held, L)
    rs.devices, rs.compute_dtype, rs.text_rows = devices, compute_dtype, text_rows
    rs.lengths = [int(v) for v in lengths.tolist()]
    rs.kl_rows = (rows_held or rows_fit)[: max(1, math.ceil(int(args.kl_tokens) / L))]
    rs.prefix = prefix
    # the chain identity leaves out code hashes (trainer / validator sha): an edit of this file must not orphan the
    # run's own rungs or its boundary; the tool version stays in, and the receipt keeps the full source record
    source_identity = None if source_record is None else {
        k: v for k, v in source_record.items() if k not in ("trainer_sha256", "validator_sha256")}
    identity = {"tool": [TOOL_NAME, TOOL_VERSION], "child": child.identity(), "parent": parent.identity(),
                "ids": ids_record, "held_rows": held_n, "layout": layout.__dict__,
                "preregistration": preregistration, "source_contract": source_identity,
                "hyper": {k: getattr(args, k) for k in ("dtype", "buffer_dtype", "train_dtype", "master_dtype", "lr", "beta1",
                                                        "beta2", "adam_eps", "clip", "epochs", "tokens_per_step", "micro_tokens",
                                                        "warmup_steps", "min_lr_frac", "seed", "loss_site", "max_steps")}}
    for owned in [out] + ([delta_dir] if delta_marker else []):   # every guard passed before anything is written
        owned.mkdir(parents=True, exist_ok=True)
        if not (owned / MARKER).is_file():
            write_json_atomic(owned / MARKER, {"tool": TOOL_NAME, "version": TOOL_VERSION,
                                               "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    disk_floor(out, float(args.min_free_gib))
    rs.chain = sha256_json(identity)
    rs.receipt = {"schema": SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__)},
                  "argv": list(argv) if argv is not None else sys.argv[1:], "chimera": tree, "torch": torch.__version__,
                  "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "identity": identity,
                  "run_identity_sha256": rs.chain, "devices": devices, "text_rows": text_rows,
                  "preregistration": preregistration, "source_contract": source_record,
                  "prefix": None if prefix is None else prefix.record(),
                  "rows": {"fit": len(rows_fit), "held": len(rows_held), "kl": rs.kl_rows, "seq": L, "tokens": R * L,
                           "valid_tokens": int(sum(rs.lengths)),
                           "fit_valid_tokens": int(sum(rs.lengths[r] for r in rows_fit)),
                           "held_valid_tokens": int(sum(rs.lengths[r] for r in rows_held))},
                  "child_text_blocks": n_text, "child_charted_blocks": child.charted_blocks,
                  "train_layers": sorted(train_layers), "stop_after_layer": stop, "layers": [], "charted_checks": [],
                  "host_memory_at_start": host_memory(),
                  "capability_claim": "none; per-layer fit numbers are diagnostics (owner policy 2026-10-06)"}
    for d in devices:
        if d.startswith("cuda"):
            props = torch.cuda.get_device_properties(torch.device(d))
            rs.receipt.setdefault("cards", {})[d] = {"name": props.name, "sm": f"{props.major}{props.minor}",
                                                     "total_mib": props.total_memory // 2 ** 20}
    log(f"child {child.root} ({n_text} text blocks, charted {child.charted_blocks}), parent {parent.root}; layout {layout}; "
        f"ids {R}x{L} (fit {len(rows_fit)} rows, held {len(rows_held)}); devices {devices}; dtype {args.dtype}; "
        f"train layers {sorted(train_layers)[:8]}{'...' if len(train_layers) > 8 else ''}; stop after {stop}; "
        f"prefix {'none' if prefix is None else f'L00..L{prefix.layers - 1:02d} from {prefix.root}'}; "
        f"boundary {args.boundary_dir or 'none'}")

    digest_box: dict[str, Any] = {}

    def digest_job() -> None:
        try:
            digest_box["doc"] = base_digest(child, args.base_digest)
        except Exception as error:  # noqa: BLE001 - reported, then refused at the end
            digest_box["error"] = repr(error)

    digest_thread = threading.Thread(target=digest_job, daemon=True)
    digest_thread.start()

    pin = (not args.no_pin) and any(d.startswith("cuda") for d in devices)
    h = child.hidden
    N = R * L
    progress(None, "streams")
    rs.P = torch.empty(N, h, dtype=buffer_dtype, pin_memory=pin)
    rs.C = torch.empty(N, h, dtype=buffer_dtype, pin_memory=pin)
    rs.receipt["host_buffers"] = {"pinned": pin, "dtype": str(buffer_dtype).replace("torch.", ""),
                                  "bytes_each": N * h * torch.empty(0, dtype=buffer_dtype).element_size()}
    first_train = min([i for i in train_layers if i <= stop], default=None)
    rs.boundary = BoundaryCache(Path(args.boundary_dir).absolute(), float(args.min_free_gib)) if args.boundary_dir else None
    resumed_at = None
    if rs.boundary is not None and args.resume:

        def expected_rungs(b: int) -> list[list[Any]] | None:
            """The rung list this run would advance layers 0..b through (prefix, then its own --out rungs)."""
            chain, rungs = rs.chain, []
            for k in range(b + 1):
                if k not in train_layers:
                    continue
                if prefix is not None and k < prefix.layers:
                    sha = prefix.shas[k]["safetensors"]
                else:
                    jpath = out / "rungs" / f"L{k:02d}.json"
                    if not jpath.is_file():
                        return None
                    doc = json.loads(jpath.read_text(encoding="utf-8"))
                    sha = doc.get("file_sha256")
                    if doc.get("input_chain") != chain or sha256_file(out / "rungs" / f"L{k:02d}.safetensors") != sha:
                        return None
                rungs.append([k, sha])
                chain = sha256_json([chain, k, sha])
            return rungs

        if first_train is not None and not args.no_baseline:
            rs.B = torch.empty(len(rs.kl_rows) * L, h, dtype=buffer_dtype)
        progress(None, "boundary-load")
        resumed_at = rs.boundary.load(rs, expected_rungs)
        if resumed_at is not None and resumed_at["layer"] >= stop:
            log(f"boundary L{resumed_at['layer']:02d} is at or past --stop-after-layer {stop}; not used")
            resumed_at = None
        if resumed_at is None:
            rs.B = None
    start_layer = 0
    if resumed_at is None:
        from chimera_omni.importing.safetensors_io import read_safetensors

        for ck, buf in ((parent, rs.P), (child, rs.C)):
            emb = read_safetensors(ck.root, ["embed.weight"], verify=True)["embed.weight"].torch(device="cpu")
            with torch.no_grad():   # row chunks: at 4M tokens a whole-buffer temporary would double the 84 GB host set
                for r0 in range(0, R, 64):
                    x0 = torch.nn.functional.embedding(ids[r0:r0 + 64].reshape(-1), emb).to(compute_dtype)
                    keep = valid_pos[r0:r0 + 64].reshape(-1, 1)
                    buf[r0 * L:(r0 + int(keep.shape[0]) // L) * L].copy_(
                        torch.where(keep, x0, torch.zeros_like(x0)).to(buffer_dtype))   # where(valid), as the engines do
                    del x0, keep
            del emb
    else:   # layers 0..b come from the boundary: their rungs are bookkept, nothing is recomputed
        b = int(resumed_at["layer"])
        for k in range(b + 1):
            j_k = child.text_blocks[k]
            rec_k: dict[str, Any] = {"text_layer": k, "child_block": j_k, "parent_block": parent.text_blocks[k],
                                     "kind": child.cfg.block_kind(j_k), "from_boundary": resumed_at["dir"]}
            if k in train_layers:
                stem = (prefix.root if prefix is not None and k < prefix.layers else out / "rungs") / f"L{k:02d}"
                doc = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
                base_shas = {e["base_tensor"]: e["base_tensor_sha256"] for e in doc["entries"]}
                rs.trained[k] = trained_entry(j_k, base_shas, stem.with_suffix(".safetensors"), doc["file_sha256"])
                rs.chain = sha256_json([rs.chain, k, doc["file_sha256"]])
                rec_k.update({"trained": True, "rung": {"file": str(stem.with_suffix(".safetensors")),
                                                        "sha256": doc["file_sha256"]}, "record": doc.get("record")})
            else:
                rec_k["trained"] = False
            rs.receipt["layers"].append(rec_k)
        if rs.chain != resumed_at["key"]["chain_after"]:
            raise Refusal(f"boundary L{b:02d}: the rung chain recomputed here differs from the boundary's record")
        rs.b_lengths = [int(v) for v in resumed_at["b_lengths"]]
        start_layer = b + 1
    rs.receipt["boundary"] = None if rs.boundary is None else {"dir": str(rs.boundary.root), "resumed_from": resumed_at,
                                                               "history": rs.boundary.history}
    save_receipt(rs)
    cache_paths = list(shard_paths(child.root)) + list(shard_paths(parent.root))
    last_boundary = start_layer - 1 if resumed_at is not None else -(10 ** 6)
    all_rows = list(range(R))
    charted_next = 0
    ahead = ReadAhead()
    for i in range(start_layer, stop + 1):
        t0 = time.time()
        j, k = child.text_blocks[i], parent.text_blocks[i]
        progress(i, "charted-checks")
        while charted_next < len(child.charted_blocks) and child.charted_blocks[charted_next] < j:
            rs.receipt["charted_checks"].append(check_charted_identity(rs, child.charted_blocks[charted_next]))
            charted_next += 1
        layer_rec: dict[str, Any] = {"text_layer": i, "child_block": j, "parent_block": k, "kind": child.cfg.block_kind(j)}
        # parent prefix -> y_p(i)
        progress(i, "parent")
        p_state, _ = ahead.get(parent, k)
        if i + 1 <= stop:
            ahead.prefetch(parent, parent.text_blocks[i + 1])
        blocks = blocks_on_devices(parent, k, p_state, devices, compute_dtype)
        del p_state
        if not for_row_batches(rs, rs.P, all_rows, blocks, run_block):
            raise Refusal(f"text layer {i}: non-finite parent stream")
        free_blocks(blocks, rs.args.release_cache)
        if rs.boundary is not None:      # the previous boundary's C file must be on disk before C changes
            progress(i, "boundary-wait")
            rs.boundary.wait()
        t_parent = time.time()
        c_state, c_shas = ahead.get(child, j)
        if i + 1 <= stop:
            ahead.prefetch(child, child.text_blocks[i + 1])
        if first_train is not None and i == first_train and not args.no_baseline:
            rs.B = rs.C.index_select(0, _row_tokens(rs.kl_rows, L)).clone()
            rs.b_lengths = [rs.lengths[r] for r in rs.kl_rows]
        if rs.B is not None:
            progress(i, "baseline")
            blocks = blocks_on_devices(child, j, c_state, devices, compute_dtype)
            for_row_batches(rs, rs.B, list(range(len(rs.kl_rows))), blocks, run_block, rs.b_lengths)
            free_blocks(blocks, rs.args.release_cache)
        if prefix is not None and i < prefix.layers:
            resumed = import_prefix_rung(rs, i, j, c_shas)
        else:
            resumed = load_rung(rs, i, j) if (args.resume and i in train_layers) else None
        rung_path: Path | None = None
        boundary_now = False
        if rs.boundary is not None and i in train_layers:   # P = y_p(i) is final now: its file starts on a thread
            next_replays = (prefix is not None and i + 1 < prefix.layers) or (
                bool(args.resume) and (out / "rungs" / f"L{i + 1:02d}.json").is_file())
            boundary_now = boundary_due(i, stop, resumed is None, next_replays, last_boundary, int(args.boundary_every))
            if boundary_now:
                layer_rec["boundary"] = rs.boundary.begin(rs, i)
        if i in train_layers and resumed is None:
            blocks = blocks_on_devices(child, j, c_state, devices, compute_dtype)
            # decomposition check on the first row batch: x_m + FFN half == the block's own forward
            dev0 = devices[0]
            x0 = rs.C[: L].view(1, L, -1).to(device=dev0, dtype=compute_dtype)
            v0 = _valid_mask(rs.lengths[:1], L, dev0)
            with torch.no_grad():
                ref = run_block(blocks[dev0], x0, v0)
                dec = ffn_half(blocks[dev0], attention_half(blocks[dev0], x0, v0), v0)
            diff = float((ref.float() - dec.float()).abs().max())
            scale = float(ref.float().abs().max())
            layer_rec["decomposition_check"] = {"max_abs_diff": diff, "bit_exact": bool(torch.equal(ref, dec)),
                                                "ref_abs_max": scale, "tokens": L}
            if not math.isfinite(diff) or diff > 1e-6 * max(scale, 1.0):
                raise Refusal(f"text layer {i}: x_m + FFN(x_m) does not reproduce the block forward (max |diff| {diff})")
            del x0, v0, ref, dec
            progress(i, "child-attention")
            if not for_row_batches(rs, rs.C, all_rows, blocks, attention_half):
                raise Refusal(f"text layer {i}: non-finite child attention stream")
            free_blocks(blocks, rs.args.release_cache)
            t_attn = time.time()
            progress(i, "train")
            stored, record = train_layer(rs, i, j, c_state)
            t_train = time.time()
            rung = write_rung(rs, i, j, stored, record, c_shas)
            rung_path = Path(rung["file"])
            progress(i, "advance")
            new_state = patched_state(c_state, layout, stored)
            blocks = blocks_on_devices(child, j, new_state, devices, compute_dtype)
            if not for_row_batches(rs, rs.C, all_rows, blocks, ffn_half):
                raise Refusal(f"text layer {i}: non-finite child stream after the trained FFN")
            free_blocks(blocks, rs.args.release_cache)
            del new_state, stored
            rs.trained[i] = trained_entry(j, c_shas, rung_path, rung["file_sha256"])
            rs.chain = sha256_json([rs.chain, i, rung["file_sha256"]])
            hb, ha = record["before"]["held"], record["after"]["held"]
            layer_rec.update({"trained": True, "rung": {"file": rung["file"], "sha256": rung["file_sha256"]},
                              "seconds": {"parent": round(t_parent - t0, 2), "child_attention": round(t_attn - t_parent, 2),
                                          "train": round(t_train - t_attn, 2), "advance": round(time.time() - t_train, 2)},
                              "record": record})
            log(f"L{i:02d} (block {j}, {layer_rec['kind']}) TRAINED: held residual relL2 trunk-only "
                f"{hb['trunk_only_residual']['relL2']:.6f} -> {ha['trunk_only_residual']['relL2']:.6f}; "
                f"with K3 {hb['full_layer_with_k3_residual']['relL2']:.6f} -> {ha['full_layer_with_k3_residual']['relL2']:.6f}; "
                f"FFN-level {hb['ffn_level_trunk_vs_needed_update']['relL2']:.6f} -> {ha['ffn_level_trunk_vs_needed_update']['relL2']:.6f}; "
                f"D1 loss {hb['trunk_only_residual']['d1_loss']:.6f} -> {ha['trunk_only_residual']['d1_loss']:.6f}; "
                f"K3/trunk energy {hb['k3_term_energy_rel_trunk_ffn']} -> {ha['k3_term_energy_rel_trunk_ffn']}")
        else:
            progress(i, "replay")
            if resumed is not None:
                stored, rung_doc = resumed
                state = patched_state(c_state, layout, stored)
                rung_path = Path(rung_doc["file"])
                rs.trained[i] = trained_entry(j, c_shas, rung_path, rung_doc["file_sha256"])
                rs.chain = sha256_json([rs.chain, i, rung_doc["file_sha256"]])
                layer_rec.update({"trained": True, "resumed_from": rung_doc["file"], "record": rung_doc.get("record")})
                if "imported_from_prefix" in rung_doc:
                    layer_rec["imported_from_prefix"] = rung_doc["imported_from_prefix"]
                del stored
            else:
                state = c_state
                layer_rec["trained"] = False
            blocks = blocks_on_devices(child, j, state, devices, compute_dtype)
            if not for_row_batches(rs, rs.C, all_rows, blocks, run_block):
                raise Refusal(f"text layer {i}: non-finite child stream")
            free_blocks(blocks, rs.args.release_cache)
            del state
            log(f"L{i:02d} (block {j}, {layer_rec['kind']}) "
                f"{('imported' if 'imported_from_prefix' in layer_rec else 'resumed') if resumed is not None else 'frozen'} "
                f"advance {time.time() - t0:.1f}s")
        del c_state
        layer_rec["seconds_total"] = round(time.time() - t0, 2)
        if any(d.startswith("cuda") for d in devices):
            layer_rec["cuda_peak_gib"] = {d: round(torch.cuda.max_memory_allocated(torch.device(d)) / 2 ** 30, 3)
                                          for d in devices if d.startswith("cuda")}
        layer_rec["peak_rss_gb"] = peak_rss_gb()
        progress(i, "free-host-buffers")
        layer_rec["memory"] = _layer_memory(rs, i, cache_paths, [rung_path] if rung_path is not None else [])
        if boundary_now and i in rs.trained:      # the rung is committed and C advanced: B now, C + seal on a thread
            progress(i, "boundary-finish")
            layer_rec["boundary"] = rs.boundary.finish(rs, i) or layer_rec.get("boundary")
            last_boundary = i
        rs.receipt["layers"].append(layer_rec)
        rs.receipt["seconds"] = round(time.time() - started, 2)
        save_receipt(rs)
    ahead.close()
    if rs.boundary is not None:
        rs.boundary.wait()
        rs.boundary.abort()
        save_receipt(rs)
    if stop == n_text - 1:
        progress(stop, "kl")
        while charted_next < len(child.charted_blocks):
            rs.receipt["charted_checks"].append(check_charted_identity(rs, child.charted_blocks[charted_next]))
            charted_next += 1
        if not args.no_kl:
            rs.receipt["head_kl"] = head_kl(rs)
            kl = rs.receipt["head_kl"]
            log(f"frozen-head KL(parent||child) rows [0,{text_rows}) over {kl['positions']} positions: D1 {kl['d1']['kl_mean']:.6f} "
                f"(top-1 {kl['d1']['top1_agreement']:.4f})" + (f"; original child {kl['original']['kl_mean']:.6f} "
                                                              f"(top-1 {kl['original']['top1_agreement']:.4f})" if "original" in kl else ""))
    progress(stop, "base-digest")
    digest_thread.join()
    if "error" in digest_box:
        raise Refusal(f"base digest failed: {digest_box['error']}")
    base = digest_box["doc"]
    for name, meta in base["files"].items():
        st = (child.root / name).stat()
        if st.st_size != meta["bytes"] or st.st_mtime_ns != meta["mtime_ns"]:
            raise Refusal(f"the base child changed during the run: {name}")
    rs.receipt["base"] = base
    progress(stop, "delta")
    rs.receipt["delta"] = write_delta(rs, base)
    rs.receipt["seconds"] = round(time.time() - started, 2)
    rs.receipt["peak_rss_gb"] = peak_rss_gb()
    rs.receipt["host_memory_at_end"] = host_memory()
    rs.receipt["verdict"] = (f"{len(rs.trained)} layer(s) trained, streams finite through text layer {stop}"
                             + ("; delta written" if rs.receipt["delta"] else "; no trained layer, no delta"))
    save_receipt(rs)
    progress(stop, "done")
    log(f"{rs.receipt['verdict']}; receipt {out / 'd1_receipt.json'}; {rs.receipt['seconds']:.0f}s")
    return rs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", required=True, help="the union child (S5 / OMNI1 clone), Chimera checkpoint dir; read only")
    ap.add_argument("--preregistration", default=None, help="frozen stage-D document (stage2_training_receipt pin); "
                                                            "the production launcher always gives it")
    ap.add_argument("--parent", required=True, help="the parent as a Chimera checkpoint (S0 import of Qwen3.8-27B-U); read only")
    ap.add_argument("--ids", required=True, help="token ids: FILE.safetensors[::KEY] [rows, seq], FILE.u32 (+ --seq-len) or FILE.npy")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--lengths-key", default="lengths", help="valid-length tensor [rows] in a .safetensors ids file ('' = ignore)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--held-rows", type=int, default=None, help="last N rows held out of training (default max(1, rows // 16))")
    ap.add_argument("--source-manifest", default=None, help="1.2.10 successor dataset manifest (d1_source_contract.py)")
    ap.add_argument("--source-manifest-sha256", default=None)
    ap.add_argument("--source-validator-sha256", default=None, help="sha256 of d1_source_contract.py")
    ap.add_argument("--prefix-dir", default=None, help="canonical rung directory (d1_canonical_rungs.py) to import from")
    ap.add_argument("--prefix-sums-sha256", default=None, help="sha256 of the prefix dir's SHA256SUMS (pinned by the launcher)")
    ap.add_argument("--prefix-layers", type=int, default=None, help="import text layers [0, N) from --prefix-dir")
    ap.add_argument("--boundary-dir", default=None,
                    help="next-layer input cache (P, C, B after a committed rung; the latest only); loaded with --resume")
    ap.add_argument("--boundary-every", type=int, default=1,
                    help="write the boundary after every K-th newly trained layer (1: after every one)")
    ap.add_argument("--layers", default="all", help="text layers to train, e.g. 'all', '39', '0-31,40'")
    ap.add_argument("--stop-after-layer", type=int, default=None, help="last text layer to process (default: the last)")
    ap.add_argument("--devices", default="cuda:0", help="comma list; frozen forwards and training shards spread over them")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"), help="frozen forward dtype")
    ap.add_argument("--buffer-dtype", default=None, choices=("bfloat16", "float16", "float32"), help="host stream dtype (default --dtype)")
    ap.add_argument("--train-dtype", default="float32", choices=("float32", "bfloat16", "float16"), help="training matmul dtype")
    ap.add_argument("--master-dtype", default="float32", choices=("float32", "float64"), help="master weights + Adam (float64: tests)")
    ap.add_argument("--row-batch", type=int, default=8, help="rows per frozen forward batch of a KDA block")
    ap.add_argument("--attn-row-batch", type=int, default=2,
                    help="rows per frozen forward batch of a full-attention block (its fp32 scores are [B, heads, L, L])")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tokens-per-step", type=int, default=32768)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-site", default="residual", choices=("residual", "delta"),
                    help="residual: x_m + FFN vs y_p (design D1); delta: FFN vs y_p - x_m")
    ap.add_argument("--diag-tokens", type=int, default=16384)
    ap.add_argument("--kl-tokens", type=int, default=16384)
    ap.add_argument("--kl-chunk", type=int, default=512)
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--no-baseline", action="store_true", help="no original-child stream on the KL rows")
    ap.add_argument("--trunk-width", type=int, default=None, help="default: config metadata c10_trunk_intermediate_size")
    ap.add_argument("--k3-channels", type=int, default=None, help="default: config metadata c10_k3_channels")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta-out", default=None)
    ap.add_argument("--min-free-gib", type=float, default=60.0)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--release-cache", action="store_true", help="empty the CUDA cache between layers (the card then looks free)")
    ap.add_argument("--base-digest", default="full", choices=("full", "headers"))
    ap.add_argument("--log-every", type=int, default=10)
    return ap


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
