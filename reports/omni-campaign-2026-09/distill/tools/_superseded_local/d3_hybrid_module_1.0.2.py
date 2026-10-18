#!/usr/bin/env python3
"""E1: the D3 hybrid module -- qwen3_5 text layers + the Chimera charted units at their depth +
the C15/C11 interface rows + the native media (flow) head, with the four D3 sub-stages.

WHY THIS FILE EXISTS.  DISTILL_DESIGN_2026-10-12 section 4 (D3) and section 5.2-5.6: the five
media units of OMNI1 are Chimera-only tensors (`blocks.32-36.charted.*`), the interface rows live
in Chimera's `native_media_bridge` / `native_media_head`, and the C18 music band is a slice of the
Chimera vocabulary -- none of them has a slot in the qwen3_5 layout the streamed engines train in.
So D3 trains a HYBRID: the text trunk runs as stock `Qwen3_5DecoderLayer`s fed by the S0-P1
zero-copy reader (`p1_qwen35_reader.py` >= 1.0.5, which serves OMNI1's 64 text layers out of its
69 blocks and every Chimera-only tensor as an extra), and the media parts run as the Chimera
tree's OWN modules (`ChartedSourceBlock`, `NativeContinuousMediaBridge`,
`UnifiedNativeMediaEmissionHead`) loaded from those extras, glued with the tree's own
`ChimeraOmniModel._add_native_media` / `_charted_layout` (called on a shim, never re-typed here).

WHERE THE UNITS SIT.  Read from the checkpoint, never assumed: a unit at Chimera block j reads the
output of the last text layer below j.  On OMNI1 (composite_manifest `placements`, insert arm,
"depth-fraction -> nearest gated_gqa refinement layer") ALL FIVE units are inserted after text
layer 31, so d = 32 for every unit and D3b backprops through text layers 32..63.  The names the
brief uses (ACE L12, H3 L25, QIMG L30, HY3D L10, QVL32 L13) are the SOURCE models' block indices
(the containers ace_dit_L12, h3_dit_L25_curve6, qimg_mmdit_L30_img, hy3d_dit_L10, qvl32_vit_L13),
not trunk depths.  When the manifest is present its `insert_after` must equal what the block
schedule says, or the module refuses.

THE FOUR SUB-STAGES (design section 4 D3):
  D3a  per-unit refit on host rows: chart_in(host row) -> the parent block's input row
       (relative MSE), plus, where the teacher cache has block in/out pairs (H3 block 25,
       QVL32 ViT block 13), the block on the parent's own input -> the parent's output delta.
       QVL32 is weighted 0.1 (design section 5.6).  No trunk layer runs.
  D3b  guided-flow KD.  x_d (the hidden state entering depth d, BEFORE the units) is recaptured
       once with layers 0..d-1 frozen and stored in <= 20 GB chunks (`XdChunkWriter`); a step
       applies the units to x_d, streams layers d..63 forward (no grad, boundaries on the host),
       runs the final norm and the flow head, and walks d..63 back with the weights reloaded
       (per-layer checkpointing, the T(ii) algorithm of streamed_kd_trainer.py, reused).
       Trainable: the unit, the emission rows of its types (native_media_head.out band rows +
       the type_affine rows of its types), and the shared LoRA r16 on layers d..63.
  D3c  the C18 music band rows [252,416, 317,952) of embed and lm_head by KL to the ACE LM's
       recorded top-64 (renormalised onto the 64,000 code rows), full depth.  The band rows are
       host fp32 masters updated by `LazyRowAdam` (only touched rows: embed rows of the codes in
       the batch; all 64,000 head rows, whose softmax gradient is dense); + the shared LoRA r16.
  D3d  vision-in / video-in answer KD through the QVL32 unit + the C11 rows (bridge
       feature_proj columns [640, 2176) and type_embed row 6), full depth, over text rows
       [0, 248,320) with the T(ii) objective (`chunked_objective`, Q-text cache format).
Every text loss is over vocab rows [0, 248,320) only (the head is sliced before the matmul, by
the trainer's engine); the D3c loss is over the band rows only.

FLOW TARGETS ARE x0 (2026-08-28 proof: pure velocity regression pins at the target variance at
small scale and does not train).  The flow head's emission band is read as the clean endpoint x0
(`--flow-param x0`, default; Chimera's own `native_media_target_kind="x0"`).  Teacher targets
become x0 with each parent's sign (s8_media_smoke.SIGN, read from that tool): x0 = x_t + SIGN *
sigma * v_parent_raw.  `--flow-param raw` keeps OMNI1's velocity-space rows and maps them
(x0 = x_t + SIGN * sigma * raw) -- recorded, not the default.

WHAT IS PROVEN WHERE.
  parity     OMNI1, CPU fp32, a few tokens per unit: every unit's held S5 pair segment
             (s7_omni1_check.media_segment, trimmed) through the hybrid AND through the Chimera
             reference (s7_omni1_check.Checkpoint / build_block / build_top: strict, sha-verified,
             layer-major), compared per text layer.  Functional gates: hybrid media modules
             bit-identical to the reference's; every unit's write bit-identical on the
             reference's own input; exact zeros on rows a unit does not own (every unit, every
             segment) and a non-zero write on its own segment; all finite; end-to-end relative L2
             (x_d, final hidden, own emission band, text logits rows [0, 248,320)) <= --tol-rel.
  synthetic  a tiny Chimera child built with the REAL importer + real ChartedSourceBlocks: the
             hybrid vs `ChimeraOmniModel.forward` on it, streamed grads vs full autograd, and
             D3a / D3b / D3c / D3d losses that decrease.  test_d3_hybrid_module.py beside this
             file holds the same checks as unit tests.
  rate       real OMNI1 weights on one card, lr = 0 (nothing changes, asserted): x_d recapture
             tok/s, D3b (or D3d-shaped) fwd+bwd tok/s, peak VRAM, host RSS.

WHAT IS NOT HERE.  The per-teacher cache adapters (cache tensors -> native feature rows / emission
x0 targets) for the production caches, which are not built yet; the D3 run driver (rungs,
safety checkpoints, delta export).  The functions they call are here and tested.

Usage (server; env: CHIMERA_OMNI_SRC=<tree ef1ffb9>, PYTHONPATH=<tree>/src:<this dir>):
    $PY d3_hybrid_module.py parity --omni1 DIR --parent-config PDIR --tap-rows DIR --receipt OUT
    $PY d3_hybrid_module.py synthetic --receipt OUT
    $PY d3_hybrid_module.py rate --omni1 DIR --parent-config PDIR --tap-rows DIR --device cuda:0 \\
        --stage d3b --units ace_dit_L12,hy3d_dit_L10 --receipt OUT
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

TOOL_NAME = "d3_hybrid_module"
#: 1.0.0 (2026-10-13): first version, lane d3 (wave 2, E1).
#: 1.0.1 (2026-10-13): the rate cell is a function (`rate_cell`), CPU-tested on the tiny child before
#: it runs on a card; every cell starts from a trainable LoRA bank (1.0.0's D3d branch relied on
#: running in a fresh process); a trainable tensor that gets no gradient is a refusal.
#: 1.0.2 (2026-10-13, review fixes): BandTrainer.head_rows returns a copy (on CPU fp32 `.to` aliased
#: the master and D3c stepped on the running sum of every earlier grad); the D3d rate cell puts its KD
#: positions in answer text appended AFTER the media rows (1.0.1 scored prompt rows only, so causal
#: attention gave the unit and the bridge rows exactly zero gradient); a unit / delta / LoRA group
#: whose gradient norm is exactly 0 is a refusal, and every cell records its group grad norms.
TOOL_VERSION = "1.0.2"
RECEIPT_SCHEMA = "neural-forge-omni-d3-hybrid-v1"
XD_SCHEMA = "neural-forge-omni-d3-xd-chunks-v1"
HERE = Path(__file__).resolve().parent
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
COMPOSITE_TOOLS = Path(os.environ.get("D3_COMPOSITE_TOOLS", str(OC / "composite" / "tools")))
MANIFEST_FILENAME = "composite_manifest.json"
#: the reader must know charted blocks (1.0.4 refuses OMNI1 at block 32)
P1_MIN_VERSION = (1, 0, 5)
#: ACE-Step 1.5 LM code ids (media_b_teacher_cache.py ACE_AUDIO_START/END: codes are stored as
#: indices into [151669, 215669)); the C18 band writes code k at band row k (c18_music_band.py)
ACE_CODES = 64_000
DEFAULT_XD_CHUNK_BYTES = 20 * 10**9


class D3Error(RuntimeError):
    """A structural refusal: wrong layout, missing tensor, non-finite value, failed gate."""


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": file_sha256(Path(__file__).resolve())}


# --------------------------------------------------------------------------
# Peers.  Every contract this file consumes is read through its own tool:
# the streamed engine + LoRA bank + objective from streamed_kd_trainer.py, the
# qwen3_5 map from p1_qwen35_reader.py (through the trainer's loader, so both
# see one module object), the Chimera reference and the S5 pair segments from
# composite/tools/s7_omni1_check.py, the flow signs from s8_media_smoke.py.
# --------------------------------------------------------------------------


def _load_by_path(name: str, path: Path) -> Any:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    if not path.is_file():
        raise D3Error(f"peer tool {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def trainer_module() -> Any:
    path = Path(os.environ.get("STREAMED_KD_TRAINER_PATH", str(HERE / "streamed_kd_trainer.py")))
    return _load_by_path("streamed_kd_trainer", path)


def p1_module() -> Any:
    module = trainer_module().load_peer_module("p1_qwen35_reader")
    version = str(getattr(module, "TOOL_VERSION", "")).split()[-1]
    try:
        parts = tuple(int(p) for p in version.split("."))
    except ValueError as error:
        raise D3Error(f"p1_qwen35_reader version {version!r} is not numeric") from error
    if parts < P1_MIN_VERSION:
        raise D3Error(
            f"p1_qwen35_reader {version} has no charted-block support (needs >= "
            f"{'.'.join(map(str, P1_MIN_VERSION))}: OMNI1's text layer t is not blocks.t)"
        )
    return module


def composite_tool(name: str) -> Any:
    return _load_by_path(f"{name}_d3", COMPOSITE_TOOLS / f"{name}.py")


def peer_record(module: Any) -> dict[str, Any]:
    path = Path(module.__file__).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "version": getattr(module, "TOOL_VERSION", None)}


def flow_signs() -> dict[int, float]:
    """x0 = x_t + SIGN * sigma * raw per media type id, from s8_media_smoke.SIGN (its comments
    say which parent output convention each sign comes from)."""

    s8 = composite_tool("s8_media_smoke")
    by_name = dict(s8.SIGN)
    return {4: by_name["ace"], 1: by_name["qimg"], 7: by_name["hy3d"], 2: by_name["h3_video"], 3: by_name["h3_audio"]}


# --------------------------------------------------------------------------
# Layout: where each unit sits, which bands each type owns.
# --------------------------------------------------------------------------


@dataclass
class UnitPlacement:
    block: int                  # Chimera block index (OMNI1: 32..36)
    container: str              # manifest container (ace_dit_L12, ...) or the spec's component
    source: str                 # spec.source (ace, h3, hy3d, qimg, qvl32)
    type_ids: tuple[int, ...]
    after_text_layer: int       # the text layer whose output the unit reads; -1 = the embedding
    spec: Any = field(repr=False, default=None)

    @property
    def depth(self) -> int:
        """d: the first text layer ABOVE the unit (D3b backprops through d .. L-1)."""

        return self.after_text_layer + 1


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def unit_layout(reader: Any, cfg: Any, manifest: dict[str, Any] | None) -> list[UnitPlacement]:
    """Placements in Chimera block order, cross-checked against the manifest when present."""

    text_blocks = list(reader.text_blocks)
    placements = {int(p["layer_index"]): p for p in ((manifest or {}).get("blocks") or {}).get("placements", [])}
    trunk_map = {int(k): int(v) for k, v in ((manifest or {}).get("blocks") or {}).get("trunk_layer_map", {}).items()}
    if trunk_map and [trunk_map[t] for t in range(len(text_blocks))] != text_blocks:
        raise D3Error("composite_manifest trunk_layer_map disagrees with the reader's text_blocks")
    out: list[UnitPlacement] = []
    for block in reader.charted_blocks:
        below = [t for t, b in enumerate(text_blocks) if b < block]
        after = below[-1] if below else -1
        spec = cfg.charted_spec(block)
        container = str(spec.component)
        if placements:
            if block not in placements:
                raise D3Error(f"charted block {block} has no manifest placement")
            placed = placements[block]
            container = str(placed["container"])
            declared = int((placed.get("pairing") or {}).get("insert_after", after))
            if declared != after:
                raise D3Error(f"{container}: manifest insert_after {declared}, block schedule says {after}")
            if sorted(int(t) for t in placed["type_ids"]) != sorted(int(t) for t in spec.type_ids):
                raise D3Error(f"{container}: manifest type_ids {placed['type_ids']} != spec {list(spec.type_ids)}")
        out.append(UnitPlacement(block=int(block), container=container, source=str(spec.source),
                                 type_ids=tuple(int(t) for t in spec.type_ids), after_text_layer=int(after), spec=spec))
    owners: dict[int, str] = {}
    for unit in out:
        for t in unit.type_ids:
            if t in owners:
                raise D3Error(f"type {t} is owned by both {owners[t]} and {unit.container}")
            owners[t] = unit.container
    return out


def interface_bands(manifest: dict[str, Any] | None) -> dict[int, dict[str, tuple[int, int]]]:
    """type id -> {"feature": (lo, hi), "emission": (lo, hi)} from the manifest's interface entries."""

    bands: dict[int, dict[str, tuple[int, int]]] = {}
    for entry in ((manifest or {}).get("interface") or {}).get("entries", []) or []:
        t = int(entry["type_id"])
        bands[t] = {"feature": (int(entry["feature_band"][0]), int(entry["feature_band"][1])),
                    "emission": (int(entry["emission_band"][0]), int(entry["emission_band"][1])),
                    "name": entry.get("name"), "set": entry.get("set")}
    return bands


# --------------------------------------------------------------------------
# Module construction from P1 extras (Chimera names).
# --------------------------------------------------------------------------


def load_strict_from(module: nn.Module, fetch: Callable[[str], torch.Tensor], prefix: str) -> nn.Module:
    """`load_state_dict(strict=True)` with every tensor fetched by `prefix + name` and cast to the
    dtype its reference parameter already has (so the H3 fp32 islands stay fp32)."""

    state = {}
    for name, reference in module.state_dict().items():
        value = fetch(prefix + name)
        if tuple(value.shape) != tuple(reference.shape):
            raise D3Error(f"{prefix}{name}: stored {tuple(value.shape)} != module {tuple(reference.shape)}")
        state[name] = value.to(device=reference.device, dtype=reference.dtype if reference.is_floating_point() else value.dtype)
    module.load_state_dict(state, strict=True)
    return module


def build_unit(reader: Any, placement: UnitPlacement, hidden: int, *, device: str, dtype: torch.dtype) -> nn.Module:
    from chimera_omni.modeling.charted import ChartedSourceBlock

    unit = ChartedSourceBlock(placement.spec, hidden).to(device=device, dtype=dtype)
    return load_strict_from(unit, reader.extra, f"blocks.{placement.block}.charted.").eval()


def build_bridge(reader: Any, cfg: Any, *, device: str, dtype: torch.dtype) -> nn.Module:
    from chimera_omni.modeling.chimera import NativeContinuousMediaBridge

    bridge = NativeContinuousMediaBridge(cfg).to(device=device, dtype=dtype)
    return load_strict_from(bridge, reader.extra, "native_media_bridge.").eval().requires_grad_(False)


def build_media_head(reader: Any, cfg: Any, *, device: str, dtype: torch.dtype) -> nn.Module:
    from chimera_omni.modeling.chimera import UnifiedNativeMediaEmissionHead

    head = UnifiedNativeMediaEmissionHead(cfg).to(device=device, dtype=dtype)
    return load_strict_from(head, reader.extra, "native_media_head.").eval().requires_grad_(False)


class _BridgeCall(nn.Module):
    """`functional_call` target: the bridge's own `embed`, so a delta never re-types its code."""

    def __init__(self, bridge: nn.Module) -> None:
        super().__init__()
        self.bridge = bridge

    def forward(self, *args: Any) -> torch.Tensor:
        return self.bridge.embed(*args)


class _HeadCall(nn.Module):
    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, hidden: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        return self.head(hidden, type_ids)


class SliceDelta(nn.Module):
    """A trainable additive delta on row (axis 0) or column (axis 1) ranges of a frozen weight.

    The master is fp32 and starts at exactly zero, so a fresh delta changes nothing; `full(base)`
    returns base + delta scattered into its ranges, differentiable with respect to the delta.
    Exported later as the trained subset of the named tensor (oc_checkpoint --delta-base)."""

    def __init__(self, name: str, shape: tuple[int, int], ranges: list[tuple[int, int]], axis: int,
                 device: str | torch.device) -> None:
        super().__init__()
        if axis not in (0, 1):
            raise D3Error("SliceDelta axis must be 0 (rows) or 1 (columns)")
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self.ranges = [(int(a), int(b)) for a, b in ranges]
        self.axis = axis
        for a, b in self.ranges:
            if not 0 <= a < b <= self.shape[axis]:
                raise D3Error(f"{name}: range [{a}, {b}) outside axis {axis} of {self.shape}")
        width = sum(b - a for a, b in self.ranges)
        other = self.shape[1 - axis]
        delta_shape = (width, other) if axis == 0 else (other, width)
        self.delta = nn.Parameter(torch.zeros(delta_shape, dtype=torch.float32, device=device))

    def full(self, base: torch.Tensor) -> torch.Tensor:
        scattered = torch.zeros(self.shape, dtype=torch.float32, device=base.device)
        offset = 0
        for a, b in self.ranges:
            n = b - a
            if self.axis == 0:
                scattered[a:b] = self.delta[offset:offset + n]
            else:
                scattered[:, a:b] = self.delta[:, offset:offset + n]
            offset += n
        return base + scattered.to(base.dtype)


# --------------------------------------------------------------------------
# Lazy row Adam (host fp32 masters of the C18 band rows).
# --------------------------------------------------------------------------


class LazyRowAdam:
    """Adam over the rows of a host fp32 table, updating only the rows a step names.

    Each row keeps its own step count, so its bias correction is the one a dense Adam would have
    used had the row been updated only on the steps it was (TF LazyAdam / torch SparseAdam
    semantics).  When every row is named every step this is exactly dense Adam (tested).  The
    caller passes unique row ids and their summed gradients."""

    def __init__(self, table: torch.Tensor, *, lr: float, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        if table.dtype != torch.float32 or table.device.type != "cpu":
            raise D3Error("LazyRowAdam keeps an fp32 host master")
        self.table = table
        self.lr = float(lr)
        self.beta1, self.beta2 = (float(b) for b in betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.m = torch.zeros_like(table)
        self.v = torch.zeros_like(table)
        self.steps = torch.zeros(table.shape[0], dtype=torch.int64)
        self.rows_updated = 0

    @torch.no_grad()
    def step(self, rows: torch.Tensor, grads: torch.Tensor) -> None:
        rows = rows.to(device="cpu", dtype=torch.long)
        grads = grads.to(device="cpu", dtype=torch.float32)
        if rows.ndim != 1 or grads.shape != (rows.numel(), self.table.shape[1]):
            raise D3Error(f"LazyRowAdam.step wants rows [n] and grads [n, {self.table.shape[1]}]")
        if rows.numel() != torch.unique(rows).numel():
            raise D3Error("LazyRowAdam.step rows must be unique (sum duplicate gradients first)")
        if not bool(torch.isfinite(grads).all()):
            raise D3Error("LazyRowAdam.step got a non-finite gradient")
        t = self.steps[rows] + 1
        self.steps[rows] = t
        m = self.m[rows].mul_(self.beta1).add_(grads, alpha=1.0 - self.beta1)
        v = self.v[rows].mul_(self.beta2).addcmul_(grads, grads, value=1.0 - self.beta2)
        self.m[rows] = m
        self.v[rows] = v
        tf = t.to(torch.float64)
        c1 = (1.0 - self.beta1 ** tf).to(torch.float32).unsqueeze(1)
        c2 = (1.0 - self.beta2 ** tf).to(torch.float32).unsqueeze(1)
        update = (m / c1) / ((v / c2).sqrt() + self.eps)
        current = self.table[rows]
        if self.weight_decay:
            current.mul_(1.0 - self.lr * self.weight_decay)
        self.table[rows] = current - self.lr * update
        self.rows_updated += int(rows.numel())

    def state(self) -> dict[str, torch.Tensor]:
        return {"m": self.m, "v": self.v, "steps": self.steps}


def unique_row_grads(rows: torch.Tensor, grads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum gradients of repeated rows: (unique rows, [n_unique, H] summed grads)."""

    unique, inverse = torch.unique(rows, return_inverse=True)
    summed = torch.zeros(unique.numel(), grads.shape[-1], dtype=torch.float32, device=grads.device)
    summed.index_add_(0, inverse.to(grads.device), grads.float())
    return unique, summed


# --------------------------------------------------------------------------
# x_d chunk store (D3b).
# --------------------------------------------------------------------------


class XdChunkWriter:
    """x_d samples ([L, H] each, stored in `dtype`) packed into safetensors chunks of at most
    `max_chunk_bytes`, with a sealed index (sha256 per chunk, the identity the x_d depends on:
    base checkpoint, LoRA state of the layers below d, inputs).  A sample bigger than the cap
    gets a chunk of its own (recorded).  The directory must be new or empty."""

    def __init__(self, directory: str | Path, *, identity: dict[str, Any], max_chunk_bytes: int = DEFAULT_XD_CHUNK_BYTES,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise D3Error(f"{self.directory} is not empty; an x_d store is written once")
        self.identity = identity
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.dtype = dtype
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.chunks: list[dict[str, Any]] = []
        self.entries: dict[str, dict[str, Any]] = {}
        self.oversize: list[str] = []

    def add(self, key: str, xd: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
        if key in self.entries or key in self.buffer:
            raise D3Error(f"x_d key {key!r} written twice")
        if xd.ndim != 2:
            raise D3Error("x_d samples are [L, H]")
        tensor = xd.detach().to(device="cpu", dtype=self.dtype).contiguous()
        if not bool(torch.isfinite(tensor.float()).all()):
            raise D3Error(f"x_d {key}: non-finite")
        nbytes = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_bytes + nbytes > self.max_chunk_bytes:
            self.flush()
        if nbytes > self.max_chunk_bytes:
            self.oversize.append(key)
        self.buffer[key] = tensor
        self.buffer_bytes += nbytes
        self.entries[key] = {"chunk": None, "shape": list(tensor.shape), "meta": meta or {}}

    def flush(self) -> None:
        if not self.buffer:
            return
        from safetensors.torch import save_file

        name = f"chunk-{len(self.chunks):05d}.safetensors"
        path = self.directory / name
        save_file(self.buffer, str(path), metadata={"schema": XD_SCHEMA})
        sha = file_sha256(path)
        self.chunks.append({"file": name, "sha256": sha, "bytes": path.stat().st_size, "keys": sorted(self.buffer)})
        for key in self.buffer:
            self.entries[key]["chunk"] = name
        self.buffer = {}
        self.buffer_bytes = 0

    def close(self) -> dict[str, Any]:
        self.flush()
        index = {"schema": XD_SCHEMA, "tool": tool_identity(), "identity": self.identity, "dtype": str(self.dtype).replace("torch.", ""),
                 "max_chunk_bytes": self.max_chunk_bytes, "chunks": self.chunks, "entries": self.entries,
                 "oversize_samples": self.oversize, "samples": len(self.entries)}
        trainer_module().write_json_atomic(self.directory / "index.json", index)
        return index


class XdChunkReader:
    def __init__(self, directory: str | Path, *, expect_identity: dict[str, Any] | None = None, verify: bool = True) -> None:
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        if self.index.get("schema") != XD_SCHEMA:
            raise D3Error(f"{self.directory} is not an x_d store")
        if expect_identity is not None and self.index["identity"] != expect_identity:
            raise D3Error("x_d store identity differs from this run's (base / lower-layer LoRA / inputs changed: recapture)")
        if verify:
            for chunk in self.index["chunks"]:
                if file_sha256(self.directory / chunk["file"]) != chunk["sha256"]:
                    raise D3Error(f"{chunk['file']} does not hash like its index")
        self._handles: dict[str, Any] = {}

    def keys(self) -> list[str]:
        return sorted(self.index["entries"])

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        entry = self.index["entries"].get(key)
        if entry is None:
            raise D3Error(f"x_d key {key!r} is not in the store")
        name = entry["chunk"]
        if name not in self._handles:
            self._handles[name] = safe_open(str(self.directory / name), framework="pt")
        return self._handles[name].get_tensor(key)


# --------------------------------------------------------------------------
# The hybrid.
# --------------------------------------------------------------------------


@dataclass
class HybridOptions:
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    head_device: str | None = None       # text head rows [0, base) (LayerwiseEngine); default = device
    mode: str = "streamed"                # LayerwiseEngine: streamed | resident
    attention: str = "sdpa"
    kernels: str = "torch"                # torch (CPU / Pascal / parity) or auto (fla on the 3090s)
    unit_dtype: torch.dtype = torch.float32  # units keep fp32 masters (trainable in D3a/b/d)
    lora: Any = None                      # streamed_kd_trainer.SitePlanConfig or None
    lora_seed: int = 0
    plan: str | None = None               # sealed oc_checkpoint delta-compose plan (P1 --plan)
    pin_boundaries: bool = True
    prefetch: bool = True


class HybridD3:
    """OMNI1 (or a delta child) as qwen3_5 layers + Chimera units / bridge / flow head.

    All rows must be valid (no padding): one sample per call, as the parity arm and the D3 rungs
    use it.  Chimera's per-block `torch.where(token_valid, ...)` is then the identity."""

    def __init__(self, omni_dir: str | Path, parent_config_dir: str | Path, options: HybridOptions | None = None) -> None:
        self.options = options or HybridOptions()
        o = self.options
        self.T = trainer_module()
        self.T.select_linear_attention_kernels(o.kernels)
        p1_module()  # version gate before the trainer opens it
        self.omni_dir = Path(omni_dir)
        self.source = self.T.P1Source(self.omni_dir, Path(parent_config_dir), Path(o.plan) if o.plan else None)
        self.reader = self.source.reader
        if not hasattr(self.reader, "text_blocks"):
            raise D3Error("the P1 reader has no text_blocks (needs >= 1.0.5)")
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.modeling.chimera import ChimeraOmniModel

        self._chimera_model = ChimeraOmniModel
        self.cfg = ChimeraConfig.from_manifest(self.reader.manifest["config"])
        self.manifest = read_manifest(self.omni_dir)
        self.layout = unit_layout(self.reader, self.cfg, self.manifest)
        self.bands = interface_bands(self.manifest)
        self.base_vocab = int(self.reader.base_vocab)
        self.hidden = int(self.reader.hidden)
        self.device = torch.device(o.device)
        self.dtype = o.dtype
        self.text_config = self.T.text_config_from_source(self.source, o.dtype, o.attention)
        self.num_layers = int(self.text_config.num_hidden_layers)
        self.bank = None
        if o.lora is not None:
            plan = self.T.build_site_plan(self.text_config, o.lora)
            self.bank = self.T.AdapterBank(plan, [str(t) for t in self.text_config.layer_types[: self.num_layers]],
                                           o.device, o.lora_seed)
        self.engine = self.T.LayerwiseEngine(
            self.source, self.text_config, mode=o.mode, dtype=o.dtype, layer_devices=[o.device],
            head_device=o.head_device or o.device, bank=self.bank, text_vocab_rows=self.base_vocab,
            pin_boundaries=o.pin_boundaries, prefetch=o.prefetch,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        self.final_norm = Qwen3_5RMSNorm(self.text_config.hidden_size, eps=self.text_config.rms_norm_eps).to(self.device, o.dtype)
        with torch.no_grad():
            self.final_norm.weight.copy_(self.source.get(f"{self.T.DEFAULT_TEXT_PREFIX}.norm.weight"))
        self.final_norm.requires_grad_(False)
        self.units: dict[str, nn.Module] = {
            p.container: build_unit(self.reader, p, self.hidden, device=o.device, dtype=o.unit_dtype).requires_grad_(False)
            for p in self.layout
        }
        self.bridge = build_bridge(self.reader, self.cfg, device=o.device, dtype=o.dtype)
        self.media_head = build_media_head(self.reader, self.cfg, device=o.device, dtype=o.dtype)
        self._bridge_call = _BridgeCall(self.bridge)
        self._head_call = _HeadCall(self.media_head)
        self.bridge_deltas: list[SliceDelta] = []
        self.head_deltas: list[SliceDelta] = []
        #: the band rows (host views of the extras; D3c swaps in trainable masters)
        self.band_embed = self.reader.extra("embed.weight[base:]")
        self.band_head = self.reader.extra("lm_head.weight[base:]") if self.reader.has("lm_head.weight") else None
        self.band_embed_override: torch.Tensor | None = None
        self.media_placeholder = int(self.cfg.ledger().range("media").begin)
        self.music_band_begin = int(self.cfg.ledger().range("music_codes").begin)

    # -- identity ------------------------------------------------------------

    def identity(self) -> dict[str, Any]:
        return {"source": self.source.identity(), "layout": [dataclasses.asdict(dataclasses.replace(p, spec=None)) for p in self.layout],
                "bands": {str(k): v for k, v in self.bands.items()}, "trainer": peer_record(self.T),
                "p1": peer_record(sys.modules["p1_qwen35_reader"]), "options": {
                    "dtype": str(self.dtype), "device": str(self.device), "mode": self.options.mode,
                    "kernels": self.options.kernels, "unit_dtype": str(self.options.unit_dtype),
                    "lora": None if self.options.lora is None else dataclasses.asdict(self.options.lora)}}

    def units_after(self, layer: int) -> list[UnitPlacement]:
        return [p for p in self.layout if p.after_text_layer == layer]

    # -- inputs --------------------------------------------------------------

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token rows: [0, base) from the P1 text table, >= base from the band rows."""

        ids = ids.to("cpu", torch.long)
        text = F.embedding(ids.clamp(max=self.base_vocab - 1), self.engine.embed_weight)
        band = ids >= self.base_vocab
        if bool(band.any()):
            table = self.band_embed_override if self.band_embed_override is not None else self.band_embed
            text[band] = table[ids[band] - self.base_vocab].to(text.dtype)
        return text.to(device=self.device, dtype=self.dtype)

    def _shim(self) -> SimpleNamespace:
        bridge = self._bridge_call
        deltas = self.bridge_deltas

        class _Embedder:
            def embed(_self, *args: Any) -> torch.Tensor:
                if not deltas:
                    return bridge(*args)
                params = {f"bridge.{d.name}": d.full(dict(bridge.named_parameters())[f"bridge.{d.name}"]) for d in deltas}
                return torch.func.functional_call(bridge, params, args)

        return SimpleNamespace(has_charted_blocks=bool(self.layout), native_media_bridge=_Embedder(), cfg=self.cfg)

    def prepare(self, segment: dict[str, Any], *, media_grad: bool = False) -> dict[str, Any]:
        """Embedding + native media (Chimera's own `_add_native_media` / `_charted_layout`)."""

        ids = segment["ids"].to(self.device)
        if ids.shape[0] != 1:
            raise D3Error("the hybrid runs one sample per call")
        x = self.embed(ids)
        shim = self._shim()
        model = self._chimera_model
        features = segment.get("features")
        context: dict[str, Any] = {"ids": ids, "valid": torch.ones(ids.shape, dtype=torch.bool, device=self.device),
                                   "row_types": None, "levels": None, "positions": None, "indices": None, "types": None,
                                   "media_rows": 0}
        if features is not None:
            types = segment["types"].to(self.device)
            valid_media = torch.ones(types.shape, dtype=torch.bool, device=self.device)
            sigma = segment.get("sigma")
            noise = None if sigma is None else sigma.to(self.device)
            indices = segment["indices"].to(self.device)
            with torch.set_grad_enabled(media_grad):
                x, count = model._add_native_media(shim, x, features.to(self.device), types,
                                                   segment["positions"].to(self.device), indices, valid_media, noise)
            row_types, levels = model._charted_layout(shim, ids, count, indices, types, valid_media, noise)
            context.update({"row_types": row_types, "levels": levels, "indices": indices, "types": types, "media_rows": int(count)})
            positions = segment.get("source_positions")
            context["positions"] = None if positions is None else {k: v.to(self.device) for k, v in positions.items()}
        context["x"] = x
        return context

    # -- units ---------------------------------------------------------------

    def unit_write(self, placement: UnitPlacement, x: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        unit = self.units[placement.container]
        positions = None if context["positions"] is None else context["positions"].get(placement.spec.source)
        x_unit = x.to(self.options.unit_dtype)
        write = unit(x_unit, context["row_types"], context["levels"], context["valid"], source_positions=positions)
        return write.to(x.dtype)

    def apply_units(self, x: torch.Tensor, after_layer: int, context: dict[str, Any],
                    writes: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        for placement in self.units_after(after_layer):
            write = self.unit_write(placement, x, context)
            if writes is not None:
                writes[placement.container] = write.detach()
            x = x + write
        return x

    # -- layers --------------------------------------------------------------

    def contexts(self, x: torch.Tensor) -> dict[str, dict[str, Any]]:
        return self.engine._context(int(x.shape[0]), int(x.shape[1]))

    def run_layers(self, x: torch.Tensor, start: int, stop: int, context: dict[str, Any], *,
                   bounds: list[torch.Tensor] | None = None, pre_units: dict[int, torch.Tensor] | None = None,
                   writes: dict[str, torch.Tensor] | None = None, per_layer: Callable[[int, torch.Tensor], None] | None = None,
                   units_at_stop: bool = True) -> torch.Tensor:
        """Text layers [start, stop) under no_grad, the units placed after each of them applied.

        `bounds[i - start]` = the input of layer i (host copy); `pre_units[i]` = layer i's output
        before the units that follow it (host copy), for the streamed backward.  With
        `units_at_stop=False` the units after layer stop-1 are NOT applied: that is how x_d (the
        hidden entering depth d, before the units) is captured."""

        contexts = self.contexts(x)
        with torch.no_grad():
            for i in range(start, stop):
                if bounds is not None:
                    bounds.append(x.detach().to("cpu", copy=True))
                layer = self.engine.layer(i, i + 1 if i + 1 < stop else None)
                x = self.engine._run(layer, i, x, contexts)
                if self.units_after(i) and (units_at_stop or i != stop - 1):
                    if pre_units is not None:
                        pre_units[i] = x.detach().to("cpu", copy=True)
                    x = self.apply_units(x, i, context, writes)
                if per_layer is not None:
                    per_layer(i, x)
        return x

    def backward_layers(self, grad: torch.Tensor, bounds: list[torch.Tensor], start: int, stop: int,
                        context: dict[str, Any], pre_units: dict[int, torch.Tensor]) -> torch.Tensor:
        """Walk layers stop-1 .. start back: units (recomputed with grad) then the layer (weights
        reloaded, boundary from the host).  Returns d loss / d (input of layer `start`)."""

        contexts = self.contexts(grad)
        for i in reversed(range(start, stop)):
            if i in pre_units:
                y = pre_units[i].to(self.device, copy=True).requires_grad_(True)
                with torch.enable_grad():
                    z = self.apply_units(y, i, context)
                z.backward(grad)
                grad = y.grad
                del y, z
            layer = self.engine.layer(i, i - 1 if i - 1 >= start else None)
            x = bounds[i - start].to(self.device, copy=True).requires_grad_(True)
            with torch.enable_grad():
                y = self.engine._run(layer, i, x, contexts)
            y.backward(grad)
            grad = x.grad
            del x, y
        return grad

    # -- heads ---------------------------------------------------------------

    def media_hidden(self, normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        index = context["indices"].unsqueeze(-1).expand(-1, -1, normed.shape[-1])
        return normed.gather(1, index)

    def emission(self, normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        """The flow head (`native_media_head`) on the media rows, with any trainable row deltas."""

        rows = self.media_hidden(normed, context)
        if not self.head_deltas:
            return self._head_call(rows, context["types"])
        named = dict(self._head_call.named_parameters())
        params = {f"head.{d.name}": d.full(named[f"head.{d.name}"]) for d in self.head_deltas}
        return torch.func.functional_call(self._head_call, params, (rows, context["types"]))

    def text_logits(self, normed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Logits over the text rows [0, base) only, at the given positions (fp32)."""

        head = self.engine.head_weight
        rows = normed[0, positions.to(normed.device)].to(head.device, head.dtype)
        return F.linear(rows, head).float()

    # -- whole forward -------------------------------------------------------

    def forward(self, segment: dict[str, Any]) -> dict[str, Any]:
        """The whole hybrid, no grad: `pre_units[i]` is the hidden entering the units placed after
        text layer i (x_d for d = i + 1), `writes` every unit's write, `emission` the flow head."""

        context = self.prepare(segment)
        x = context["x"]
        out: dict[str, Any] = {"writes": {}}
        with torch.no_grad():
            x = self.apply_units(x, -1, context, out["writes"])
            pre: dict[int, torch.Tensor] = {}
            x = self.run_layers(x, 0, self.num_layers, context, pre_units=pre, writes=out["writes"])
            normed = self.final_norm(x)
        out["pre_units"] = pre
        out["normed"] = normed
        out["context"] = context
        if context["media_rows"]:
            with torch.no_grad():
                out["emission"] = self.emission(normed, context)
        return out

    # -- trainable sets ------------------------------------------------------

    def unit_parameters(self, containers: list[str]) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for name in containers:
            unit = self.units[name]
            for pname, p in unit.named_parameters():
                # source-frozen coefficients (H3 curve table, H3 RoPE frequencies) stay frozen
                if pname in ("h3_time.curve_table", "attn.rope_inv_freq"):
                    continue
                p.requires_grad_(True)
                params.append(p)
        return params

    def add_emission_deltas(self, type_ids: list[int]) -> list[SliceDelta]:
        """Trainable emission rows of these types (out.weight band rows) + their type_affine rows."""

        out_w = self.media_head.out.weight
        ta = self.media_head.type_affine.weight
        ranges = []
        for t in sorted(set(type_ids)):
            if t not in self.bands:
                raise D3Error(f"type {t} has no interface band in the manifest")
            ranges.append(self.bands[t]["emission"])
        deltas = [SliceDelta("out.weight", tuple(out_w.shape), ranges, 0, self.device),
                  SliceDelta("type_affine.weight", tuple(ta.shape), [(t, t + 1) for t in sorted(set(type_ids))], 0, self.device)]
        self.head_deltas.extend(deltas)
        return deltas

    def add_bridge_deltas(self, type_ids: list[int]) -> list[SliceDelta]:
        """Trainable interface-in rows: feature_proj columns of these types' feature bands + type_embed rows."""

        fp = self.bridge.feature_proj.weight
        te = self.bridge.type_embed.weight
        cols = [self.bands[t]["feature"] for t in sorted(set(type_ids))]
        deltas = [SliceDelta("feature_proj.weight", tuple(fp.shape), cols, 1, self.device),
                  SliceDelta("type_embed.weight", tuple(te.shape), [(t, t + 1) for t in sorted(set(type_ids))], 0, self.device)]
        self.bridge_deltas.extend(deltas)
        return deltas

    def lora_parameters(self, layers: range | None = None) -> list[torch.Tensor]:
        if self.bank is None:
            return []
        keep = set(range(self.num_layers) if layers is None else layers)
        out = []
        for name in self.bank.names:
            tensor = self.bank.tensors[name]
            index = int(name.split(".")[1])
            if index in keep and tensor.requires_grad:
                out.append(tensor)
        return out

    def freeze_lora_outside(self, layers: range) -> None:
        """D3b: layers below d keep their LoRA but get no gradient (x_d is cached past them)."""

        if self.bank is None:
            return
        keep = set(layers)
        for name in self.bank.names:
            if int(name.split(".")[1]) not in keep:
                self.bank.tensors[name].requires_grad_(False)

    def lora_digest(self, layers: range) -> str:
        """sha256 over the bank tensors of `layers` (x_d validity: the layers below d)."""

        digest = hashlib.sha256()
        if self.bank is not None:
            for name in self.bank.names:
                if int(name.split(".")[1]) in set(layers):
                    digest.update(name.encode())
                    digest.update(self.bank.tensors[name].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# Objectives.
# --------------------------------------------------------------------------


def relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction.float(), target.float()
    return (prediction - target).pow(2).sum() / target.pow(2).sum().clamp_min(1e-30)


def teacher_x0(x_t: torch.Tensor, raw: torch.Tensor, sigma: torch.Tensor, sign: float) -> torch.Tensor:
    """x0 from a parent's raw output in its own convention: x0 = x_t + SIGN * sigma * raw."""

    s = sigma.float()
    while s.ndim < x_t.ndim:
        s = s.unsqueeze(-1)
    return x_t.float() + float(sign) * s * raw.float()


def flow_x0_loss(prediction: torch.Tensor, x0_target: torch.Tensor, types: torch.Tensor, mask: torch.Tensor,
                 bands: dict[int, dict[str, Any]], *, param: str = "x0", x_t: torch.Tensor | None = None,
                 sigma: torch.Tensor | None = None, signs: dict[int, float] | None = None) -> torch.Tensor:
    """Per row: mean over the row type's EMISSION band of (x0_pred - x0_target)^2; type-balanced
    mean over supervised rows (Chimera's `native_loss_reduction = type_balanced`).

    prediction / x0_target / x_t are [B, M, F] in the emission layout; types [B, M]; mask [B, M]."""

    if param not in ("x0", "raw"):
        raise D3Error("flow param is x0 or raw")
    per_type: list[torch.Tensor] = []
    for t in sorted(int(v) for v in torch.unique(types[mask > 0]).tolist()):
        if t not in bands:
            raise D3Error(f"type {t} has no emission band")
        lo, hi = bands[t]["emission"]
        rows = (types == t) & (mask > 0)
        pred = prediction[..., lo:hi][rows].float()
        if param == "raw":
            if x_t is None or sigma is None or signs is None:
                raise D3Error("flow param raw needs x_t, sigma and the parent signs")
            pred = teacher_x0(x_t[..., lo:hi][rows], pred, sigma[rows], signs[t])
        target = x0_target[..., lo:hi][rows].float()
        if not bool(torch.isfinite(target).all()):
            raise D3Error(f"type {t}: non-finite x0 target")
        weight = mask[rows].float()
        err = (pred - target).pow(2).mean(dim=-1)
        per_type.append((err * weight).sum() / weight.sum().clamp_min(1e-12))
    if not per_type:
        raise D3Error("flow loss with no supervised media row")
    return torch.stack(per_type).mean()


def band_teacher_logprobs(topk_logprobs: torch.Tensor, lse_full: torch.Tensor, lse_band: torch.Tensor) -> torch.Tensor:
    """ACE LM top-k log-probs (log-softmax over the FULL 217,204 vocab) renormalised onto the
    64,000 code rows: log p_band = log p_full + lse_full - lse_band."""

    return topk_logprobs.float() + (lse_full.float() - lse_band.float()).unsqueeze(-1)


def band_kl(normed_rows: torch.Tensor, band_head_rows: torch.Tensor, topk_codes: torch.Tensor,
            topk_logprobs_band: torch.Tensor) -> torch.Tensor:
    """Mean over positions of KL(teacher top-k + tail || student) with the student's softmax over
    the code rows only (the trainer's bucket KL, so the tail is the exact logsumexp of the rest)."""

    T = trainer_module()
    logits = F.linear(normed_rows.float(), band_head_rows.float())
    logq = torch.log_softmax(logits, dim=-1)
    kl = T.bucket_forward_kl(logq, topk_codes.long(), topk_logprobs_band)
    return kl.mean()


def d3a_loss(unit: nn.Module, host_rows: torch.Tensor, parent_in: torch.Tensor, *, levels: torch.Tensor | None = None,
             row_types: torch.Tensor | None = None, positions: torch.Tensor | None = None,
             parent_out: torch.Tensor | None = None, w_in: float = 1.0, w_block: float = 1.0, weight: float = 1.0
             ) -> tuple[torch.Tensor, dict[str, float]]:
    """D3a for one unit on packed rows [n, *]: chart_in(host) -> parent block input; and, when
    the cache has the parent's block output, block(parent_in) - parent_in -> parent_out - parent_in
    (the block teacher-forced on the parent's own input).  `weight` is the unit weight (QVL32 0.1)."""

    n = int(host_rows.shape[0])
    dtype = unit.chart_in.weight.dtype
    projected = unit.chart_in(host_rows.to(dtype).view(1, n, -1))
    loss_in = relative_mse(projected[0], parent_in)
    parts = {"chart_in_rel_mse": float(loss_in.detach())}
    loss = w_in * loss_in
    if parent_out is not None:
        key_valid = torch.ones(1, n, dtype=torch.bool, device=host_rows.device)
        lv = torch.zeros(1, n, device=host_rows.device) if levels is None else levels.view(1, n).float()
        rt = None if row_types is None else row_types.view(1, n).long()
        sp = None if positions is None else positions.view(1, n, 3)
        p_in = parent_in.to(dtype).view(1, n, -1)
        out = unit.block(p_in, lv, key_valid, row_types=rt, source_positions=sp)
        loss_block = relative_mse((out - p_in)[0], parent_out.float() - parent_in.float())
        parts["block_rel_mse"] = float(loss_block.detach())
        loss = loss + w_block * loss_block
    return weight * loss, parts


# --------------------------------------------------------------------------
# Streamed steps.
# --------------------------------------------------------------------------


def d3b_forward_backward(model: HybridD3, context: dict[str, Any], x_d: torch.Tensor, depth: int,
                         loss_fn: Callable[[torch.Tensor], torch.Tensor], *, loss_scale: float = 1.0) -> float:
    """One D3b micro-step from a cached x_d (the hidden entering depth d, before the units).

    Units (with graph) -> layers d..L-1 streamed (no grad, host boundaries) -> final norm + flow
    head (graph) -> loss -> layers walked back with reloaded weights -> units' backward."""

    x_in = x_d.to(device=model.device, dtype=model.dtype)
    with torch.enable_grad():
        z = model.apply_units(x_in, depth - 1, context)
    bounds: list[torch.Tensor] = []
    pre: dict[int, torch.Tensor] = {}
    h = model.run_layers(z.detach(), depth, model.num_layers, context, bounds=bounds, pre_units=pre)
    last = h.detach().requires_grad_(True)
    with torch.enable_grad():
        normed = model.final_norm(last)
        loss = loss_fn(model.emission(normed, context))
    if not bool(torch.isfinite(loss)):
        raise D3Error(f"D3b loss is not finite: {float(loss)}")
    (loss * float(loss_scale)).backward()
    grad = model.backward_layers(last.grad, bounds, depth, model.num_layers, context, pre)
    if z.requires_grad:
        z.backward(grad)
    return float(loss.detach())


def full_depth_forward_backward(model: HybridD3, segment: dict[str, Any],
                                head_loss: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor], *,
                                loss_scale: float = 1.0, input_grad: bool = False) -> tuple[float, torch.Tensor | None, dict[str, Any]]:
    """D3c / D3d: the whole stack.  The input (embedding + bridge, with any bridge deltas) keeps
    its graph; returns (loss, d loss / d x0 when `input_grad`, context)."""

    context = model.prepare(segment, media_grad=bool(model.bridge_deltas))
    x0 = context["x"]
    x_start = x0.detach()
    with torch.enable_grad():
        x_start = model.apply_units(x_start, -1, context) if model.units_after(-1) else x_start
    bounds: list[torch.Tensor] = []
    pre: dict[int, torch.Tensor] = {}
    h = model.run_layers(x_start.detach(), 0, model.num_layers, context, bounds=bounds, pre_units=pre)
    last = h.detach().requires_grad_(True)
    with torch.enable_grad():
        normed = model.final_norm(last)
        loss = head_loss(normed, context)
    if not bool(torch.isfinite(loss)):
        raise D3Error(f"loss is not finite: {float(loss)}")
    (loss * float(loss_scale)).backward()
    grad = model.backward_layers(last.grad, bounds, 0, model.num_layers, context, pre)
    if x0.requires_grad:
        x0.backward(grad, retain_graph=input_grad)
    return float(loss.detach()), (grad.detach() if input_grad else None), context


class BandTrainer:
    """D3c: host fp32 masters of the C18 code rows of embed and lm_head, LazyRowAdam on each."""

    def __init__(self, model: HybridD3, *, lr: float, codes: int = ACE_CODES) -> None:
        self.model = model
        self.offset = model.music_band_begin - model.base_vocab   # band row index of code 0
        self.codes = int(codes)
        if model.band_head is None:
            raise D3Error("D3c needs an untied lm_head (the band head rows)")
        full_embed = model.band_embed.float().clone()
        self.embed_master = full_embed
        self.head_master = model.band_head[self.offset:self.offset + self.codes].float().clone()
        self.embed_opt = LazyRowAdam(self.embed_master, lr=lr)
        self.head_opt = LazyRowAdam(self.head_master, lr=lr)
        model.band_embed_override = self.embed_master

    def head_rows(self) -> torch.Tensor:
        """A fresh copy of the head master on the model's device / dtype.  Never the master itself:
        `Tensor.to` returns the same object when device and dtype already match (CPU fp32), and the
        grad would then accumulate on the master across steps (1.0.1 trained D3c on a running sum)."""
        return self.head_master.detach().to(self.model.device, self.model.dtype, copy=True)

    def step(self, segment: dict[str, Any], prompt_len: int, topk_codes: torch.Tensor, topk_lp_band: torch.Tensor) -> float:
        """segment ids = [prompt (text ids) | band ids of the codes]; position prompt_len-1+k
        predicts code k (record k of the ACE LM cache)."""

        model = self.model
        n = int(topk_codes.shape[0])
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + n)
        head = self.head_rows().requires_grad_(True)

        def head_loss(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
            rows = normed[0, positions.to(normed.device)]
            return band_kl(rows, head, topk_codes.to(normed.device), topk_lp_band.to(normed.device))

        ids = segment["ids"][0]
        x_grad_needed = bool((ids >= model.base_vocab).any())
        loss, grad_x0, _ = full_depth_forward_backward(model, segment, head_loss, input_grad=x_grad_needed)
        self.head_opt.step(torch.arange(self.codes), head.grad.detach().float().cpu())
        if x_grad_needed and grad_x0 is not None:
            band = ids >= model.base_vocab
            rows, grads = unique_row_grads((ids[band] - model.base_vocab).cpu(), grad_x0[0][band.to(grad_x0.device)].float().cpu())
            self.embed_opt.step(rows, grads)
        return loss


def answer_kd_targets(length: int, positions: torch.Tensor, next_ids: torch.Tensor, q_ids: torch.Tensor,
                      q_lp: torch.Tensor, *, w_ce: float = 0.0) -> dict[str, torch.Tensor]:
    """The T(ii) targets dict (`chunked_objective`) for D3d: teacher top-k at the answer positions
    (position p carries the teacher's distribution for token p+1), zero weight elsewhere."""

    k = int(q_ids.shape[-1])
    out = {"next_ids": torch.full((length,), -1, dtype=torch.long), "ce_w": torch.zeros(length), "eos_w": torch.zeros(length),
           "q_w": torch.zeros(length), "q_ids": torch.full((length, k), -1, dtype=torch.long), "q_lp": torch.zeros(length, k)}
    out["next_ids"][positions] = next_ids.long()
    out["ce_w"][positions] = float(w_ce)
    out["q_w"][positions] = 1.0
    out["q_ids"][positions] = q_ids.long()
    out["q_lp"][positions] = q_lp.float()
    return out


def d3d_head_loss(model: HybridD3, targets: dict[str, torch.Tensor], *, denominator: float) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """The text KD over rows [0, base) through the trainer's `chunked_objective`; it backwards per
    chunk into a detached copy, so the returned surrogate carries that gradient to `normed`."""

    T = model.T
    objective = T.ObjectiveConfig(w_q38=1.0, lambda_k3=0.0, w_ce=0.0, w_eos=0.0, w_ul=0.0,
                                  text_vocab_rows=model.base_vocab, head_chunk_positions=1024)

    def loss_fn(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        head = model.engine.head_weight
        parts, grad = T.chunked_objective(normed.to(head.device, head.dtype), head, targets, objective,
                                          denominator=denominator, backward=True)
        surrogate = (normed * grad.to(normed.device, normed.dtype)).sum()
        value = torch.tensor(float(parts["total"]), device=normed.device)
        return surrogate - surrogate.detach() + value

    return loss_fn


# --------------------------------------------------------------------------
# Parity arm on OMNI1 (CPU fp32) against the Chimera reference.
# --------------------------------------------------------------------------


def trim_segment(segment: dict[str, Any], max_rows_per_type: int, placeholder: int) -> dict[str, Any]:
    """Keep the first `max_rows_per_type` media rows of every type (order kept; the HY3D time row
    is row 0 of the volume segment and so is kept).  0 = keep all."""

    if max_rows_per_type <= 0:
        return segment
    types = segment["types"][0]
    keep: list[int] = []
    for t in dict.fromkeys(int(v) for v in types.tolist()):
        keep.extend(torch.nonzero(types == t).flatten()[:max_rows_per_type].tolist())
    keep = sorted(keep)
    idx = torch.tensor(keep, dtype=torch.long)
    plen = int(segment["prompt_tokens"])
    m = len(keep)
    out = dict(segment)
    out["ids"] = torch.cat([segment["ids"][:, :plen], torch.full((1, m), int(placeholder), dtype=torch.long)], dim=1)
    out["features"] = segment["features"][:, idx]
    out["types"] = segment["types"][:, idx]
    out["positions"] = segment["positions"][:, idx]
    out["sigma"] = None if segment["sigma"] is None else segment["sigma"][:, idx]
    out["indices"] = torch.arange(plen, plen + m, dtype=torch.long).view(1, m)
    out["media_rows"] = m
    out["trimmed_from"] = int(segment["media_rows"])
    if "host_rows_s5" in segment:
        out["host_rows_s5"] = segment["host_rows_s5"][idx]
    return out


def with_text_suffix(segment: dict[str, Any], count: int) -> dict[str, Any]:
    """[prompt | media rows] -> [prompt | media rows | `count` answer-text tokens].  The answer text is
    the prompt's own token ids, cycled (real text rows, nothing past `prompt_tokens` is media); media
    rows, indices and features are unchanged; per-position source positions get zero rows (the H3
    zero plan of `h3_zero_positions`)."""

    if count <= 0:
        return segment
    prompt = int(segment["prompt_tokens"])
    if prompt <= 0:
        raise D3Error("a text suffix needs at least one prompt token to cycle")
    ids = segment["ids"]
    tail = ids[:, :prompt].repeat(1, -(-count // prompt))[:, :count]
    out = dict(segment)
    out["ids"] = torch.cat([ids, tail], dim=1)
    out["suffix_tokens"] = int(segment.get("suffix_tokens", 0)) + int(count)
    positions = segment.get("source_positions")
    if positions is not None:
        out["source_positions"] = {k: torch.cat([v, torch.zeros(v.shape[0], count, *v.shape[2:], dtype=v.dtype)], dim=1)
                                   for k, v in positions.items()}
    return out


def h3_zero_positions(cfg: Any, length: int) -> dict[str, torch.Tensor] | None:
    """s7_omni1_check's plan: H3 source positions zeros (the h3_thw_v1 positions have never run)."""

    out = {spec.source: torch.zeros(1, length, 3) for spec in cfg.charted_blocks if spec.modulation == "h3_curve6"}
    return out or None


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def command_parity(args: argparse.Namespace) -> int:
    torch.set_num_threads(int(args.threads))
    started = time.time()
    s7 = composite_tool("s7_omni1_check")
    dtype = torch.float32
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "parity", "tool": tool_identity(), "argv": sys.argv[1:],
                               "torch": torch.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "s7_omni1_check": peer_record(s7), "device": "cpu", "dtype": "float32"}
    refusals: list[str] = []
    omni_dir = Path(args.omni1)
    log("opening the hybrid (P1 reader, streamed qwen3_5 layers, Chimera units from extras)")
    hybrid = HybridD3(omni_dir, Path(args.parent_config), HybridOptions(dtype=dtype, device="cpu", kernels="torch",
                                                                         mode="streamed", prefetch=True))
    receipt["hybrid"] = hybrid.identity()
    receipt["layout"] = [{"block": p.block, "container": p.container, "after_text_layer": p.after_text_layer,
                          "depth_d": p.depth, "type_ids": list(p.type_ids)} for p in hybrid.layout]
    log("opening the Chimera reference (s7_omni1_check.Checkpoint, sha-verified)")
    ref = s7.Checkpoint(omni_dir, "omni1", verify=True)
    top = s7.build_top(ref, device="cpu")
    meta = top["meta"]
    bridge32 = top["native_media_bridge"].to(dtype=dtype)
    setattr(meta, "native_media_bridge", bridge32)
    head32 = top["native_media_head"].to(dtype=dtype)
    norm32 = top["norm"].to(dtype=dtype)

    # G-a: the hybrid's media modules are the reference's, bit for bit
    identity: dict[str, Any] = {}
    for name, a, b in [("native_media_bridge", hybrid.bridge, bridge32), ("native_media_head", hybrid.media_head, head32)]:
        sa, sb = a.state_dict(), b.state_dict()
        same = set(sa) == set(sb) and all(torch.equal(sa[k].float(), sb[k].float()) for k in sa)
        identity[name] = bool(same)
        if not same:
            refusals.append(f"G-a: hybrid {name} differs from the reference's")

    bands_feature = {t: v["feature"] for t, v in hybrid.bands.items()}
    segments = []
    for p in hybrid.layout:
        seg = s7.media_segment(p.container, Path(args.tap_rows), bands_feature, ref.cfg)
        seg = trim_segment(seg, int(args.max_rows_per_type), hybrid.media_placeholder)
        seg["source_positions"] = h3_zero_positions(ref.cfg, int(seg["ids"].shape[1]))
        segments.append(seg)
    receipt["segments"] = [{"unit": s["unit"], "forward": s["forward"], "prompt_tokens": s["prompt_tokens"],
                            "media_rows": s["media_rows"], "trimmed_from": s.get("trimmed_from"), "tokens": int(s["ids"].shape[1]),
                            "record": s["record"]} for s in segments]
    log(f"{len(segments)} segments: " + ", ".join(f"{s['unit']} {int(s['ids'].shape[1])} tok" for s in segments))

    # reference inputs (s7_omni1_check's own recipe, fp32)
    ref_state = []
    hyb_state = []
    with torch.no_grad():
        for seg in segments:
            ids = seg["ids"]
            x = top["embed"](ids).to(dtype)
            types = seg["types"]
            valid_media = torch.ones(types.shape, dtype=torch.bool)
            noise = None if seg["sigma"] is None else seg["sigma"]
            x, count = type(meta)._add_native_media(meta, x, seg["features"], types, seg["positions"], seg["indices"], valid_media, noise)
            row_types, levels = type(meta)._charted_layout(meta, ids, count, seg["indices"], types, valid_media, noise)
            valid = torch.ones(ids.shape, dtype=torch.bool)
            ref_state.append({"x": x, "valid": valid, "row_types": row_types, "levels": levels, "positions": seg["source_positions"]})
            ctx = hybrid.prepare(seg)
            hyb_state.append(ctx)
            identity.setdefault("input_rel_l2", []).append(rel_l2(ctx["x"], x))
    receipt["identity"] = identity

    text_blocks = list(hybrid.reader.text_blocks)
    per_layer: list[dict[str, Any]] = []
    units_report: dict[str, Any] = {}
    x_d_rel: dict[str, float] = {}
    contexts_cache: dict[int, Any] = {}

    def run_ref_block(block: Any, s: dict[str, Any]) -> torch.Tensor:
        y = block(s["x"], native_media_slot_mask=None, valid_media_mask=None, enable_native_media_refinement=False,
                  token_valid_mask=s["valid"], charted_row_types=s["row_types"], charted_noise_levels=s["levels"],
                  charted_source_positions=s["positions"])
        return torch.where(s["valid"].unsqueeze(-1), y, torch.zeros_like(y))

    with torch.no_grad():
        for t, j in enumerate(text_blocks):
            blk = s7.build_block(ref, j, device="cpu", dtype=dtype)
            for s in ref_state:
                s["x"] = run_ref_block(blk, s)
            del blk
            layer = hybrid.engine.layer(t, t + 1 if t + 1 < len(text_blocks) else None)
            for ctx in hyb_state:
                contexts = hybrid.engine._context(1, int(ctx["x"].shape[1]))
                ctx["x"] = hybrid.engine._run(layer, t, ctx["x"], contexts)
            row = {"layer": t, "block": j, "rel_l2": [rel_l2(c["x"], s["x"]) for c, s in zip(hyb_state, ref_state)],
                   "finite": all(bool(torch.isfinite(c["x"]).all()) for c in hyb_state)}
            per_layer.append(row)
            if not row["finite"]:
                refusals.append(f"layer {t}: hybrid hidden not finite")
            following = hybrid.units_after(t)
            if following:
                for si, (seg, c, s) in enumerate(zip(segments, hyb_state, ref_state)):
                    x_d_rel[seg["unit"]] = rel_l2(c["x"], s["x"])
            for placement in following:
                cblk = s7.build_block(ref, placement.block, device="cpu", dtype=dtype)
                unit = hybrid.units[placement.container]
                sa, sb = unit.state_dict(), cblk.charted.state_dict()
                same_params = set(sa) == set(sb) and all(torch.equal(sa[k].float(), sb[k].float()) for k in sa)
                rep: dict[str, Any] = {"block": placement.block, "params_bit_identical": bool(same_params), "segments": {}}
                if not same_params:
                    refusals.append(f"G-a: {placement.container} unit params differ from the reference block")
                for seg, c, s in zip(segments, hyb_state, ref_state):
                    w_ref = cblk.charted_output(s["x"], s["row_types"], s["levels"], s["valid"], s["positions"])
                    # G-b: the hybrid unit on the REFERENCE's own input
                    w_iso = hybrid.unit_write(placement, s["x"], {**c, "row_types": s["row_types"], "levels": s["levels"],
                                                                  "valid": s["valid"], "positions": s["positions"]})
                    # the hybrid's own write on its own stream
                    w_hyb = hybrid.unit_write(placement, c["x"], c)
                    owned = unit.owned_rows(c["row_types"], c["valid"])
                    n_owned = 0 if owned is None else int(owned.sum())
                    off = w_hyb if owned is None else w_hyb.masked_fill(owned.unsqueeze(-1), 0.0)
                    own_norm = 0.0 if owned is None else float(w_hyb[owned].norm())
                    own_segment = seg["unit"] == placement.container
                    r = {"rows_owned": n_owned, "write_iso_bit_identical": bool(torch.equal(w_iso, w_ref)),
                         "write_iso_max_abs": float((w_iso - w_ref).abs().max()) if w_ref.numel() else 0.0,
                         "zeros_off_owned": bool((off == 0).all()), "write_finite": bool(torch.isfinite(w_hyb).all()),
                         "write_norm_owned": own_norm, "own_segment": own_segment,
                         "write_rel_l2_vs_reference": rel_l2(w_hyb, w_ref) if float(w_ref.norm()) > 0 else (0.0 if float(w_hyb.norm()) == 0 else math.inf)}
                    if not r["write_iso_bit_identical"]:
                        refusals.append(f"G-b: {placement.container} write on {seg['unit']} differs from the reference on the same input")
                    if not r["zeros_off_owned"]:
                        refusals.append(f"G-c: {placement.container} writes off its own rows on {seg['unit']}")
                    if not r["write_finite"]:
                        refusals.append(f"G-c: {placement.container} non-finite write on {seg['unit']}")
                    if own_segment and (n_owned == 0 or own_norm == 0.0):
                        refusals.append(f"G-c: {placement.container} owns / writes nothing on its own segment")
                    if not own_segment and own_norm != 0.0:
                        refusals.append(f"G-c: {placement.container} writes on the {seg['unit']} segment")
                    rep["segments"][seg["unit"]] = r
                    c["x"] = c["x"] + w_hyb
                    s["x"] = torch.where(s["valid"].unsqueeze(-1), s["x"] + w_ref, torch.zeros_like(s["x"]))
                units_report[placement.container] = rep
                del cblk
            if t % 8 == 0 or t == len(text_blocks) - 1:
                worst = max(row["rel_l2"]) if row["rel_l2"] else 0.0
                log(f"text layer {t}/{len(text_blocks) - 1} (block {j}): worst rel_l2 {worst:.3e} "
                    f"({time.time() - started:.0f}s, rss {hybrid.T.host_rss_gb() or 0:.1f} GB)")

        # heads
        head_report: dict[str, Any] = {}
        lm_head = top["lm_head_weight"]
        base = hybrid.base_vocab
        for seg, c, s in zip(segments, hyb_state, ref_state):
            n_h = hybrid.final_norm(c["x"])
            n_r = norm32(s["x"])
            em_h = hybrid.emission(n_h, c)
            em_r = head32(n_r.gather(1, seg["indices"].unsqueeze(-1).expand(-1, -1, n_r.shape[-1])), seg["types"])
            band_rel = {}
            for t in sorted(set(int(v) for v in seg["types"][0].tolist())):
                lo, hi = hybrid.bands[t]["emission"]
                rows = seg["types"][0] == t
                band_rel[str(t)] = rel_l2(em_h[0, rows, lo:hi], em_r[0, rows, lo:hi])
            positions = torch.arange(int(seg["prompt_tokens"]))
            lg_h = hybrid.text_logits(n_h, positions)
            lg_r = torch.cat([F.linear(n_r[0, positions], lm_head[a:min(a + 32768, base)].to(dtype)) for a in range(0, base, 32768)], dim=-1)
            agree = int((lg_h.argmax(-1) == lg_r.argmax(-1)).sum())
            head_report[seg["unit"]] = {
                "final_hidden_rel_l2": rel_l2(n_h, n_r), "emission_band_rel_l2": band_rel,
                "text_logits_rel_l2": rel_l2(lg_h, lg_r), "text_top1_agree": agree, "text_positions": int(positions.numel()),
                "finite": bool(torch.isfinite(n_h).all() and torch.isfinite(em_h).all() and torch.isfinite(lg_h).all()),
            }
    tol = float(args.tol_rel)
    e2e: dict[str, Any] = {"tol_rel": tol, "x_d_rel_l2": x_d_rel, "heads": head_report}
    for unit, value in x_d_rel.items():
        if not value <= tol:
            refusals.append(f"G-d: x_d rel_l2 {value:.3e} > {tol} on {unit}")
    for unit, rep in head_report.items():
        if not rep["finite"]:
            refusals.append(f"G-d: non-finite hybrid output on {unit}")
        for key in ("final_hidden_rel_l2", "text_logits_rel_l2"):
            if not rep[key] <= tol:
                refusals.append(f"G-d: {key} {rep[key]:.3e} > {tol} on {unit}")
        for t, value in rep["emission_band_rel_l2"].items():
            if not value <= tol:
                refusals.append(f"G-d: emission band type {t} rel_l2 {value:.3e} > {tol} on {unit}")
    receipt.update({"per_layer": per_layer, "units": units_report, "end_to_end": e2e, "refusals": refusals,
                    "verdict": "PASS" if not refusals else "FAIL", "seconds": round(time.time() - started, 1),
                    "peak_rss_gb": hybrid.T.host_rss_gb()})
    sha = hybrid.T.write_json_atomic(Path(args.receipt), receipt)
    for line in refusals[:20]:
        log(f"REFUSAL {line}")
    worst_layer = max((max(r["rel_l2"]) for r in per_layer), default=0.0)
    log(f"D3 PARITY {'PASS' if not refusals else 'FAIL'}: {len(hybrid.layout)} units, {len(segments)} segments, "
        f"worst per-layer rel_l2 {worst_layer:.3e}, x_d {max(x_d_rel.values(), default=0.0):.3e}, "
        f"heads {max((max([r['final_hidden_rel_l2'], r['text_logits_rel_l2'], *r['emission_band_rel_l2'].values()]) for r in head_report.values()), default=0.0):.3e} "
        f"(tol {tol}); receipt {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


# --------------------------------------------------------------------------
# The synthetic arm: a tiny OMNI-shaped child, built with the real importer.
# --------------------------------------------------------------------------


TINY = SimpleNamespace(LAYERS=4, HIDDEN=64, WIDTH=24, TRUNK=8, K3=16, VOCAB=40, HEADS=2, KV_HEADS=1, HEAD_DIM=32,
                       V_HEADS=4, K_HEADS=2, K_DIM=16, V_DIM=16, FEATURE=48,
                       BANDS=[("control", 8), ("media", 8), ("music_codes", 16)])


def tiny_hf_config() -> dict[str, Any]:
    t = TINY
    return {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": t.HIDDEN, "intermediate_size": t.WIDTH, "num_hidden_layers": t.LAYERS,
                "num_attention_heads": t.HEADS, "num_key_value_heads": t.KV_HEADS, "head_dim": t.HEAD_DIM,
                "linear_num_value_heads": t.V_HEADS, "linear_num_key_heads": t.K_HEADS, "linear_key_head_dim": t.K_DIM,
                "linear_value_head_dim": t.V_DIM, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"], "vocab_size": t.VOCAB, "rms_norm_eps": 1e-6,
                "max_position_embeddings": 256, "hidden_act": "silu", "attn_output_gate": True, "attention_bias": False,
                "attention_dropout": 0.0, "tie_word_embeddings": False,
                "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25,
                                    "rope_theta": 10000.0, "rope_type": "default"},
                "dtype": "float32"}}


def tiny_unit_specs(insert_after: int) -> list[Any]:
    """An ACE-like (adaln6, RoPE, window), an H3-like (h3_curve6, reads text) and an HY3D-like
    (no modulation, layer norm, gelu, attention bias) unit, all after text layer `insert_after`."""

    from chimera_omni.charted import ChartedBlockSpec

    first = insert_after + 1
    return [
        ChartedBlockSpec(layer_index=first, source="ace", component="tiny.ace", type_ids=(4,), width=32, heads=2, kv_heads=1,
                         head_dim=16, mlp_width=40, mlp_kind="swiglu", modulation="adaln6", rope_theta=10000.0,
                         sliding_window=6, time_hidden=24, time_freq_dim=16, time_embed_r=True),
        ChartedBlockSpec(layer_index=first + 1, source="h3", component="tiny.h3", type_ids=(2, 3), width=48, heads=3, kv_heads=3,
                         head_dim=16, mlp_width=56, mlp_kind="swiglu", modulation="h3_curve6", norm_eps=1e-5,
                         h3_curve_grid=9, h3_curve_dim=4, h3_rope_freqs=2),
        ChartedBlockSpec(layer_index=first + 2, source="hy3d", component="tiny.hy3d", type_ids=(7,), width=32, heads=2, kv_heads=2,
                         head_dim=16, mlp_width=36, mlp_kind="gelu", modulation="none", norm_kind="layer", attn_bias=True),
    ]


TINY_BANDS = {4: ((0, 12), (0, 6)), 2: ((12, 20), (6, 10)), 3: ((20, 24), (10, 12)), 7: ((24, 32), (12, 16)), 6: ((32, 48), (16, 17))}


def build_tiny_child(directory: Path, *, seed: int = 5, insert_after: int = 1) -> dict[str, Any]:
    """A tiny Chimera child shaped like OMNI1: text layers imported from a random qwen3_5 state by
    the REAL importer, three charted units inserted after text layer `insert_after`, real bridge /
    head tensors (randomised so the interface is not the identity), a composite manifest with
    placements, the trunk map and interface bands.  Returns paths + the reference model."""

    from safetensors.torch import save_file
    from chimera_omni.importing.profiles import qwen35_chimera_config
    from chimera_omni.importing.qwen35 import import_qwen35_state
    from chimera_omni.modeling import ChimeraOmniModel

    t = TINY
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.08) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * scale

    key_dim, value_dim = t.K_HEADS * t.K_DIM, t.V_HEADS * t.V_DIM
    p = "model.language_model."
    qwen = {f"{p}embed_tokens.weight": rand(t.VOCAB, t.HIDDEN, scale=0.5), "lm_head.weight": rand(t.VOCAB, t.HIDDEN),
            f"{p}norm.weight": rand(t.HIDDEN, scale=0.1)}
    for i in range(t.LAYERS):
        q = f"{p}layers.{i}."
        qwen[f"{q}input_layernorm.weight"] = rand(t.HIDDEN, scale=0.1)
        qwen[f"{q}post_attention_layernorm.weight"] = rand(t.HIDDEN, scale=0.1)
        qwen[f"{q}mlp.gate_proj.weight"] = rand(t.WIDTH, t.HIDDEN)
        qwen[f"{q}mlp.up_proj.weight"] = rand(t.WIDTH, t.HIDDEN)
        qwen[f"{q}mlp.down_proj.weight"] = rand(t.HIDDEN, t.WIDTH)
        if i % 4 != 3:
            a = f"{q}linear_attn."
            qwen[f"{a}in_proj_qkv.weight"] = rand(2 * key_dim + value_dim, t.HIDDEN)
            qwen[f"{a}conv1d.weight"] = rand(2 * key_dim + value_dim, 1, 4)
            qwen[f"{a}in_proj_z.weight"] = rand(value_dim, t.HIDDEN)
            qwen[f"{a}in_proj_b.weight"] = rand(t.V_HEADS, t.HIDDEN)
            qwen[f"{a}in_proj_a.weight"] = rand(t.V_HEADS, t.HIDDEN)
            qwen[f"{a}dt_bias"] = rand(t.V_HEADS, scale=1.0)
            qwen[f"{a}A_log"] = rand(t.V_HEADS, scale=1.0)
            qwen[f"{a}norm.weight"] = 1.0 + rand(t.V_DIM, scale=0.1)
            qwen[f"{a}out_proj.weight"] = rand(t.HIDDEN, value_dim)
        else:
            s = f"{q}self_attn."
            qwen[f"{s}q_proj.weight"] = rand(t.HEADS * 2 * t.HEAD_DIM, t.HIDDEN)
            qwen[f"{s}k_proj.weight"] = rand(t.KV_HEADS * t.HEAD_DIM, t.HIDDEN)
            qwen[f"{s}v_proj.weight"] = rand(t.KV_HEADS * t.HEAD_DIM, t.HIDDEN)
            qwen[f"{s}o_proj.weight"] = rand(t.HIDDEN, t.HEADS * t.HEAD_DIM)
            qwen[f"{s}q_norm.weight"] = rand(t.HEAD_DIM, scale=0.1)
            qwen[f"{s}k_norm.weight"] = rand(t.HEAD_DIM, scale=0.1)
    base_cfg = qwen35_chimera_config(tiny_hf_config(), ledger_bands=t.BANDS, max_sequence_length=256, native_feature_dim=t.FEATURE,
                                     metadata={"c10_trunk_intermediate_size": str(t.TRUNK), "c10_k3_channels": str(t.K3)})
    imported, _receipt = import_qwen35_state(qwen, base_cfg, dtype=torch.float32)
    text_state = {n: v.detach().clone() for n, v in imported.state_dict().items()}
    specs = tiny_unit_specs(insert_after)
    total = t.LAYERS + len(specs)
    charted_at = [s.layer_index for s in specs]
    text_of = [b for b in range(total) if b not in charted_at]
    pattern = list(base_cfg.block_pattern)
    text_kinds = iter(pattern[i % len(pattern)] for i in range(t.LAYERS))
    full_pattern = tuple("charted" if b in charted_at else next(text_kinds) for b in range(total))
    cfg = dataclasses.replace(base_cfg, num_layers=total, block_pattern=full_pattern, charted_blocks=tuple(specs))
    cfg.validate()
    torch.manual_seed(seed + 1)
    model = ChimeraOmniModel(cfg).to(torch.float32)
    state = model.state_dict()
    for name, value in text_state.items():
        if name.startswith("blocks."):
            _, index, rest = name.split(".", 2)
            state[f"blocks.{text_of[int(index)]}.{rest}"] = value
        elif name in state:
            state[name] = value
    # non-trivial interface: random type rows / affine rows (the constructor zeroes some)
    for name in list(state):
        if name.startswith(("native_media_bridge.", "native_media_head.")) or ".charted." in name:
            if "curve_table" in name:
                state[name] = torch.linspace(-1.0, 1.0, state[name].numel()).view_as(state[name])
            elif "rope_inv_freq" in name:
                state[name] = 1.0 / (100.0 ** (torch.arange(state[name].numel(), dtype=torch.float32) / state[name].numel()))
            elif name.endswith("norm.weight") and ".charted." not in name:
                state[name] = 1.0 + rand(*state[name].shape, scale=0.1)
            else:
                state[name] = rand(*state[name].shape, scale=0.1)
    # band rows are written (C18-like) in embed and head: the importer leaves them zero, and a zero
    # band head makes the band logits independent of the hidden state (no trunk gradient in D3c)
    for name in ("embed.weight", "lm_head.weight"):
        rows = state[name][t.VOCAB:]
        state[name][t.VOCAB:] = rand(*rows.shape, scale=0.3)
    model.load_state_dict(state, strict=True)
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(state)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for filename, chunk in halves.items():
        save_file({n: state[n].contiguous() for n in chunk}, str(directory / filename),
                  metadata={"format": "pt", "schema": "chimera_omni.checkpoint.v1"})
        weight_map.update({n: filename for n in chunk})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"schema": "chimera_omni.checkpoint.v1", "chimera_commit": "tiny-d3",
                                                       "config": cfg.to_manifest()}), encoding="utf-8")
    manifest = {
        "schema": "chimera_omni.composite_manifest.v1",
        "blocks": {"trunk_layer_map": {str(i): b for i, b in enumerate(text_of)},
                   "placements": [{"layer_index": s.layer_index, "container": s.component.replace(".", "_"), "source": s.source,
                                   "type_ids": list(s.type_ids), "pairing": {"insert_after": insert_after}} for s in specs]},
        "interface": {"entries": [{"type_id": k, "feature_band": list(v[0]), "emission_band": list(v[1]), "name": f"tiny{k}", "set": "c15"}
                                  for k, v in TINY_BANDS.items()]},
    }
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    parent = directory.parent / f"{directory.name}-parent-config"
    parent.mkdir(exist_ok=True)
    (parent / "config.json").write_text(json.dumps(tiny_hf_config()), encoding="utf-8")
    return {"dir": directory, "parent_config": parent, "cfg": cfg, "model": model.eval(), "text_of": text_of, "specs": specs}


def h3_sigmas(sigma_v: float) -> tuple[float, float]:
    """The H3 plain AV schedule ChartedSourceBlock checks: video shift 12, audio shift 3."""

    base = sigma_v / (12.0 + sigma_v * (1.0 - 12.0))
    return float(sigma_v), float(3.0 * base / (1.0 + 2.0 * base))


def tiny_segment(cfg: Any, kind: str, *, prompt: int = 5, rows: int = 6, seed: int = 0, suffix: int = 0) -> dict[str, Any]:
    """[prompt | media rows | suffix text] of one kind ("ace", "h3", "hy3d", "ocr", "text"),
    features in the type's feature band, positions, sigma, H3 zero source positions."""

    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, TINY.VOCAB, (1, prompt), generator=g)
    if kind == "text":
        return {"ids": ids, "prompt_tokens": prompt}
    placeholder = int(cfg.ledger().range("media").begin)
    if kind == "h3":
        types = torch.tensor([[3, 3] + [2] * (rows - 2)])
        sv, sa = h3_sigmas(0.6)
        sigma = torch.where(types == 2, torch.tensor(sv), torch.tensor(sa)).float()
    else:
        t = {"ace": 4, "hy3d": 7, "ocr": 6}[kind]
        types = torch.full((1, rows), t)
        sigma = None if kind == "ocr" else torch.full((1, rows), 0.4)
    feats = torch.zeros(1, rows, TINY.FEATURE)
    for r in range(rows):
        lo, hi = TINY_BANDS[int(types[0, r])][0]
        feats[0, r, lo:hi] = torch.randn(hi - lo, generator=g)
    m = rows
    tail = torch.randint(0, TINY.VOCAB, (1, suffix), generator=g)
    seg = {"ids": torch.cat([ids, torch.full((1, m), placeholder), tail], dim=1), "features": feats, "types": types,
           "positions": torch.rand(1, m, 4, generator=g), "sigma": sigma, "indices": torch.arange(prompt, prompt + m).view(1, m),
           "prompt_tokens": prompt, "media_rows": m}
    seg["source_positions"] = h3_zero_positions(cfg, prompt + m + suffix)
    return seg


def chimera_reference(model: Any, segment: dict[str, Any]) -> dict[str, torch.Tensor]:
    """`ChimeraOmniModel.forward` itself (whole tiny model, fp32)."""

    kwargs: dict[str, Any] = {}
    if segment.get("features") is not None:
        kwargs = {"native_media_features": segment["features"], "native_media_type_ids": segment["types"],
                  "native_media_positions": segment["positions"], "native_media_indices": segment["indices"],
                  "native_media_valid_mask": torch.ones(segment["types"].shape, dtype=torch.bool),
                  "native_media_noise_levels": segment["sigma"], "charted_source_positions": segment.get("source_positions"),
                  "return_aux": True}
        if segment["sigma"] is not None:
            # the tree refuses noise levels under "reconstruction"; "x0" changes nothing else here
            # (native_bidirectional_refinement is off in the config, so no refinement layout)
            kwargs.update({"native_media_target_kind": "x0",
                           "native_media_update_mask": torch.ones(segment["types"].shape, dtype=torch.bool)})
    with torch.no_grad():
        out = model(segment["ids"], **kwargs)
    return {"hidden": out["hidden"], "logits": out["logits"], "emission": out["aux"].get("native_media_prediction") if kwargs else None}


def command_synthetic(args: argparse.Namespace) -> int:
    """Tiny OMNI-shaped child: hybrid == Chimera forward; streamed grads == autograd; D3a-d train."""

    import tempfile

    torch.manual_seed(0)
    started = time.time()
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "synthetic", "tool": tool_identity(), "argv": sys.argv[1:],
                               "torch": torch.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    refusals: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        built = build_tiny_child(Path(tmp) / "child")
        cfg, ref_model = built["cfg"], built["model"]
        T = trainer_module()
        lora = T.SitePlanConfig(rank=4, alpha=4.0, k3_rank=0, trunk_ffn_width=TINY.TRUNK, gain=False)
        hybrid = HybridD3(built["dir"], built["parent_config"], HybridOptions(dtype=torch.float32, device="cpu", kernels="torch",
                                                                              lora=lora, lora_seed=3, pin_boundaries=False))
        receipt["layout"] = [{"block": p.block, "container": p.container, "after_text_layer": p.after_text_layer} for p in hybrid.layout]
        # 1. forward parity vs ChimeraOmniModel.forward
        parity = {}
        for kind in ("text", "ace", "h3", "hy3d"):
            seg = tiny_segment(cfg, kind, seed=11)
            ref = chimera_reference(ref_model, seg)
            out = hybrid.forward(seg)
            r = {"hidden_rel_l2": rel_l2(out["normed"], ref["hidden"])}
            positions = torch.arange(int(seg["prompt_tokens"]))
            lg = hybrid.text_logits(out["normed"], positions)
            r["text_logits_rel_l2"] = rel_l2(lg, ref["logits"][0, positions, : hybrid.base_vocab])
            if ref["emission"] is not None:
                r["emission_rel_l2"] = rel_l2(out["emission"], ref["emission"])
            parity[kind] = r
            worst = max(r.values())
            if not worst <= 1e-4:
                refusals.append(f"tiny parity {kind}: rel_l2 {worst:.3e} > 1e-4")
        receipt["tiny_parity"] = parity
        log("tiny parity " + ", ".join(f"{k} {max(v.values()):.2e}" for k, v in parity.items()))
        # 2. D3b streamed grads vs full autograd on the same step
        receipt["grad_parity"] = d3b_grad_parity(hybrid)
        if not receipt["grad_parity"]["pass"]:
            refusals.append(f"D3b grad parity: worst rel {receipt['grad_parity']['worst_rel']:.3e}")
        log(f"D3b grad parity worst rel {receipt['grad_parity']['worst_rel']:.3e} over {receipt['grad_parity']['tensors']} tensors "
            f"({receipt['grad_parity']['zero_reference_grads']} with a zero reference gradient)")
        # 3. training: each sub-stage's loss decreases
        receipt["training"] = synthetic_training(hybrid, steps=int(args.steps))
        for stage, rep in receipt["training"].items():
            if not rep["last"] < rep["first"] * float(args.decrease):
                refusals.append(f"{stage}: loss {rep['first']:.4g} -> {rep['last']:.4g} (needs < x{args.decrease})")
            log(f"{stage}: loss {rep['first']:.4g} -> {rep['last']:.4g} in {rep['steps']} steps")
    receipt.update({"refusals": refusals, "verdict": "PASS" if not refusals else "FAIL", "seconds": round(time.time() - started, 1)})
    sha = trainer_module().write_json_atomic(Path(args.receipt), receipt)
    log(f"D3 SYNTHETIC {receipt['verdict']}: receipt {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


def _tiny_d3b_setup(hybrid: HybridD3, seed: int = 1) -> tuple[dict[str, Any], torch.Tensor, int, Callable[[torch.Tensor], torch.Tensor]]:
    cfg = hybrid.cfg
    seg = tiny_segment(cfg, "ace", rows=8, seed=seed)
    context = hybrid.prepare(seg)
    depth = hybrid.layout[0].depth
    with torch.no_grad():
        x_d = hybrid.run_layers(context["x"], 0, depth, context, units_at_stop=False) if depth > 0 else context["x"]
    g = torch.Generator().manual_seed(seed + 100)
    target = torch.randn(1, seg["media_rows"], TINY.FEATURE, generator=g)
    mask = torch.ones(1, seg["media_rows"])

    def loss_fn(prediction: torch.Tensor) -> torch.Tensor:
        return flow_x0_loss(prediction, target, context["types"], mask, hybrid.bands)

    return context, x_d, depth, loss_fn


def d3b_grad_parity(hybrid: HybridD3) -> dict[str, Any]:
    """Streamed D3b gradients vs one full autograd graph of the same computation (fp32)."""

    unit_name = hybrid.layout[0].container
    params = hybrid.unit_parameters([unit_name])
    deltas = hybrid.add_emission_deltas([4])
    with torch.no_grad():  # non-zero deltas and LoRA B so every gradient is exercised
        g = torch.Generator().manual_seed(9)
        for d in deltas:
            d.delta.copy_(torch.randn(d.delta.shape, generator=g) * 0.01)
        for name in hybrid.bank.names:
            if name.endswith(".b"):
                hybrid.bank.tensors[name].copy_(torch.randn(hybrid.bank.tensors[name].shape, generator=g) * 0.01)
    context, x_d, depth, loss_fn = _tiny_d3b_setup(hybrid)
    hybrid.freeze_lora_outside(range(depth, hybrid.num_layers))
    lora = hybrid.lora_parameters(range(depth, hybrid.num_layers))
    tensors = params + [d.delta for d in deltas] + lora
    for p in tensors:
        p.grad = None
    loss_streamed = d3b_forward_backward(hybrid, context, x_d, depth, loss_fn)
    streamed = [p.grad.detach().clone() for p in tensors]
    for p in tensors:
        p.grad = None
    # reference: one graph through units + layers d..L-1 + head (layers run with autograd on)
    contexts = hybrid.contexts(x_d)
    with torch.enable_grad():
        h = hybrid.apply_units(x_d, depth - 1, context)
        for i in range(depth, hybrid.num_layers):
            layer = hybrid.engine.layer(i)
            h = hybrid.engine._run(layer, i, h, contexts)
            if hybrid.units_after(i):
                h = hybrid.apply_units(h, i, context)
        loss_ref = loss_fn(hybrid.emission(hybrid.final_norm(h), context))
    loss_ref.backward()
    reference = [p.grad.detach().clone() for p in tensors]
    worst = 0.0
    zero = 0
    for a, b in zip(streamed, reference):
        worst = max(worst, float((a - b).norm() / b.norm().clamp_min(1e-30)))
        zero += int(float(b.norm()) == 0.0)  # a zero reference gradient would make the comparison vacuous
    for p in tensors:
        p.grad = None
        p.requires_grad_(False)
    hybrid.head_deltas.clear()
    for name in hybrid.bank.names:  # restore the bank to a fresh state (B = 0, all trainable)
        with torch.no_grad():
            if name.endswith(".b"):
                hybrid.bank.tensors[name].zero_()
        hybrid.bank.tensors[name].requires_grad_(True)
    return {"pass": worst <= 1e-5 and zero == 0, "worst_rel": worst, "tensors": len(tensors), "zero_reference_grads": zero,
            "loss_streamed": loss_streamed,
            "loss_reference": float(loss_ref.detach()), "threshold": 1e-5}


def synthetic_training(hybrid: HybridD3, *, steps: int) -> dict[str, Any]:
    """D3a, D3b, D3c, D3d on the tiny child; every loss must come down (in-memory only)."""

    report: dict[str, Any] = {}
    # D3a: the HY3D-like unit, chart_in -> a fixed random linear map of the host rows, block on pairs
    unit_place = [p for p in hybrid.layout if p.source == "hy3d"][0]
    unit = hybrid.units[unit_place.container]
    params = hybrid.unit_parameters([unit_place.container])
    g = torch.Generator().manual_seed(21)
    host = torch.randn(16, hybrid.hidden, generator=g)
    w_true = torch.randn(unit.chart_in.out_features, hybrid.hidden, generator=g) * 0.1
    p_in = host @ w_true.T
    p_out = p_in + 0.1 * torch.tanh(p_in)
    opt = torch.optim.Adam(params, lr=3e-3)
    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = d3a_loss(unit, host, p_in, parent_out=p_out, levels=torch.zeros(16), row_types=torch.full((16,), 7))
        loss.backward()
        opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    report["d3a"] = {"first": first, "last": last, "steps": steps}
    for p in params:
        p.requires_grad_(False)

    # D3b: ACE-like unit + its emission rows + LoRA on layers d..L-1, x0 targets
    context, x_d, depth, loss_fn = _tiny_d3b_setup(hybrid, seed=4)
    ace = hybrid.layout[0].container
    deltas = hybrid.add_emission_deltas([4])
    hybrid.freeze_lora_outside(range(depth, hybrid.num_layers))
    trainable = hybrid.unit_parameters([ace]) + [d.delta for d in deltas] + hybrid.lora_parameters(range(depth, hybrid.num_layers))
    opt = torch.optim.Adam(trainable, lr=1e-2)
    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = d3b_forward_backward(hybrid, context, x_d, depth, loss_fn)
        opt.step()
        first = loss if first is None else first
        last = loss
    report["d3b"] = {"first": first, "last": last, "steps": steps, "trainable": sum(p.numel() for p in trainable)}
    for p in trainable:
        p.requires_grad_(False)
    for name in hybrid.bank.names:
        hybrid.bank.tensors[name].requires_grad_(True)

    # D3c: band rows by KL to a synthetic "ACE LM" top-k over the code rows + LoRA all layers
    codes = 12
    band = BandTrainer(hybrid, lr=2e-2, codes=codes)
    lora_opt = torch.optim.Adam(hybrid.lora_parameters(), lr=5e-3)
    g = torch.Generator().manual_seed(33)
    prompt = torch.randint(0, TINY.VOCAB, (1, 4), generator=g)
    seq = torch.randint(0, codes, (6,), generator=g)
    ids = torch.cat([prompt, (seq + hybrid.music_band_begin).view(1, -1)], dim=1)
    k = 4
    teacher_logits = torch.randn(seq.numel(), codes, generator=g) * 2.0
    lp_full = torch.log_softmax(teacher_logits, -1)
    top_lp, top_ids = lp_full.topk(k, dim=-1)
    lse = torch.zeros(seq.numel())
    lp_band = band_teacher_logprobs(top_lp, lse, lse)
    first = last = None
    for step in range(steps):
        lora_opt.zero_grad(set_to_none=True)
        loss = band.step({"ids": ids}, int(prompt.shape[1]), top_ids, lp_band)
        lora_opt.step()
        first = loss if first is None else first
        last = loss
    report["d3c"] = {"first": first, "last": last, "steps": steps, "embed_rows_updated": band.embed_opt.rows_updated,
                     "head_rows_updated": band.head_opt.rows_updated}
    hybrid.band_embed_override = None

    # D3d: answer KD through an OCR (type 6) "vision" segment: the bridge's C11-like rows + LoRA
    hybrid.add_bridge_deltas([6])
    seg = tiny_segment(hybrid.cfg, "ocr", prompt=6, rows=5, seed=8, suffix=4)
    length = int(seg["ids"].shape[1])
    answer = torch.arange(length - 4, length - 1)  # the answer text after the image rows
    g = torch.Generator().manual_seed(44)
    q_logits = torch.randn(answer.numel(), hybrid.base_vocab, generator=g) * 3.0
    q_lp_all = torch.log_softmax(q_logits, -1)
    q_lp, q_ids = q_lp_all.topk(5, dim=-1)
    targets = answer_kd_targets(length, answer, torch.randint(0, TINY.VOCAB, (answer.numel(),), generator=g), q_ids, q_lp)
    loss_fn = d3d_head_loss(hybrid, targets, denominator=float(answer.numel()))
    trainable = [d.delta for d in hybrid.bridge_deltas] + hybrid.lora_parameters()
    opt = torch.optim.Adam(trainable, lr=1e-2)
    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = full_depth_forward_backward(hybrid, seg, loss_fn)
        opt.step()
        first = loss if first is None else first
        last = loss
    report["d3d"] = {"first": first, "last": last, "steps": steps}
    hybrid.bridge_deltas.clear()
    return report


# --------------------------------------------------------------------------
# Rate cell on real OMNI1 weights (lr = 0; nothing changes, asserted).
# --------------------------------------------------------------------------


#: answer-text tokens a D3d rate cell appends after the media rows (KD at D3D_RATE_SUFFIX - 1 positions)
D3D_RATE_SUFFIX = 8


def grad_group_norms(groups: dict[str, list[torch.Tensor]]) -> tuple[dict[str, float | None], list[str]]:
    """L2 norm of each named group's gradients (None when no tensor in it has a grad).  A group whose
    norm is exactly 0 or not finite is a problem: the path into it is dead (1.0.1's D3d cell reported
    `grads_finite_and_present` with all-zero unit and bridge gradients).  Groups, not tensors: a LoRA
    A factor legitimately gets an exact-zero gradient while its B factor is still zero."""

    norms: dict[str, float | None] = {}
    problems: list[str] = []
    for name, tensors in groups.items():
        grads = [p.grad.detach().float() for p in tensors if p.grad is not None]
        if not grads:
            norms[name] = None
            problems.append(f"{name}: no gradient")
            continue
        value = float(torch.sqrt(sum((g * g).sum().cpu() for g in grads)))
        norms[name] = value
        if not math.isfinite(value):
            problems.append(f"{name}: gradient norm not finite")
        elif value == 0.0:
            problems.append(f"{name}: gradient norm exactly 0 (dead path)")
    return norms, problems


def rate_cell(hybrid: HybridD3, placement: UnitPlacement, seg: dict[str, Any], *, stage: str, steps: int) -> tuple[dict[str, Any], list[str]]:
    """One lr = 0 rate cell: D3b (x_d recapture + units / layers d..L-1 / flow head fwd+bwd) or a
    D3d-shaped full-depth step (bridge deltas + every layer + the unit + text KD on answer text
    appended after the media rows, so the KD gradient reaches the unit and the bridge rows).
    AdamW at lr 0 runs every step; the first trainable tensors are asserted bit-unchanged after, and
    every trainable group (unit, each delta, LoRA) must get a finite non-zero gradient."""

    device = str(hybrid.device)
    cuda = device.startswith("cuda")
    container = placement.container
    if stage == "d3d":
        media_end = int(seg["ids"].shape[1])
        seg = with_text_suffix(seg, D3D_RATE_SUFFIX)
    tokens = int(seg["ids"].shape[1])
    refusals: list[str] = []
    context = hybrid.prepare(seg)
    cell: dict[str, Any] = {"unit": container, "stage": stage, "tokens": tokens, "media_rows": int(seg["media_rows"]),
                            "depth_d": placement.depth}
    hybrid.head_deltas.clear()
    hybrid.bridge_deltas.clear()
    if hybrid.bank is not None:  # a previous cell leaves the bank frozen; every cell starts from a trainable bank
        for name in hybrid.bank.names:
            hybrid.bank.tensors[name].requires_grad_(True)
    if stage == "d3b":
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            x_d = hybrid.run_layers(context["x"], 0, placement.depth, context, units_at_stop=False)
        if cuda:
            torch.cuda.synchronize()
        cell["recapture_seconds"] = time.perf_counter() - t0
        cell["recapture_tok_per_s"] = tokens / cell["recapture_seconds"]
        deltas = hybrid.add_emission_deltas(list(placement.type_ids))
        hybrid.freeze_lora_outside(range(placement.depth, hybrid.num_layers))
        groups = {"unit": hybrid.unit_parameters([container]),
                  **{f"emission_delta.{d.name}": [d.delta] for d in deltas},
                  "lora": hybrid.lora_parameters(range(placement.depth, hybrid.num_layers))}
        g = torch.Generator().manual_seed(1)
        target = torch.randn(1, int(seg["media_rows"]), int(hybrid.cfg.native_feature_dim), generator=g).to(device)
        mask = torch.ones(1, int(seg["media_rows"]), device=device)
        loss_fn = lambda pred: flow_x0_loss(pred, target, context["types"], mask, hybrid.bands)  # noqa: E731
        run = lambda: d3b_forward_backward(hybrid, context, x_d, placement.depth, loss_fn)  # noqa: E731
    elif stage == "d3d":
        groups = {**{f"bridge_delta.{d.name}": [d.delta] for d in hybrid.add_bridge_deltas(list(placement.type_ids))},
                  "lora": hybrid.lora_parameters(), "unit": hybrid.unit_parameters([container])}
        # KD only on the appended answer text: position p predicts token p + 1, every p after the media
        answer = torch.arange(media_end, tokens - 1)
        if answer.numel() == 0:
            raise D3Error(f"{container}: no answer-text position after the media rows for the D3d-shaped loss")
        cell.update({"answer_positions": [int(answer[0]), int(answer[-1])], "media_end": media_end,
                     "suffix_tokens": tokens - media_end})
        g = torch.Generator().manual_seed(2)
        q_ids = torch.randint(0, hybrid.base_vocab, (answer.numel(), 64), generator=g)
        q_lp = torch.log_softmax(torch.randn(answer.numel(), 64, generator=g), -1) - 1.0
        targets = answer_kd_targets(tokens, answer, seg["ids"][0, answer + 1], q_ids, q_lp)
        loss_fn2 = d3d_head_loss(hybrid, targets, denominator=float(answer.numel()))
        run = lambda: full_depth_forward_backward(hybrid, seg, loss_fn2)[0]  # noqa: E731
    else:
        raise D3Error(f"rate stage {stage!r}")
    groups = {name: tensors for name, tensors in groups.items() if tensors}  # no LoRA bank -> no LoRA group
    trainable = [p for tensors in groups.values() for p in tensors]
    opt = torch.optim.AdamW(trainable, lr=0.0, weight_decay=0.0)
    before = [p.detach().clone() for p in trainable[: min(4, len(trainable))]]
    cell["trainable_parameters"] = int(sum(p.numel() for p in trainable))
    times: list[float] = []
    losses: list[float] = []
    load_before = hybrid.engine.load_seconds
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run()
        missing = sum(1 for p in trainable if p.grad is None)
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainable)
        norms, dead = grad_group_norms(groups)
        cell.setdefault("grad_norms", []).append(norms)
        refusals.extend(f"{container} {stage} step {step}: {problem}" for problem in dead)
        opt.step()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss)
        if not finite:
            refusals.append(f"{container} {stage} step {step}: non-finite gradient")
        if missing:
            refusals.append(f"{container} {stage} step {step}: {missing} trainable tensors got no gradient")
        log(f"{container} {stage} step {step}: {times[-1]:.1f}s ({tokens / times[-1]:.1f} tok/s) loss {loss:.4g} grad norms "
            + ", ".join(f"{k} {'none' if v is None else f'{v:.3e}'}" for k, v in norms.items()))
    unchanged = all(torch.equal(a, p.detach()) for a, p in zip(before, trainable))
    if not unchanged:
        refusals.append(f"{container}: an lr = 0 step changed a parameter")
    steady = times[1:] if len(times) > 1 else times
    cell.update({"step_seconds": times, "losses": losses, "tok_per_s": tokens / (sum(steady) / len(steady)),
                 "params_unchanged_at_lr0": unchanged, "grads_finite_and_present": not refusals,
                 "layer_load_seconds": round(hybrid.engine.load_seconds - load_before, 1)})
    if cuda:
        cell["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(hybrid.device) / 2**20, 1)
    cell["host_rss_gb"] = hybrid.T.host_rss_gb()
    for p in trainable:
        p.requires_grad_(False)
        p.grad = None
    hybrid.head_deltas.clear()
    hybrid.bridge_deltas.clear()
    log(f"RATE {container} {stage}: {cell['tok_per_s']:.1f} tok/s steady over {len(steady)} step(s), {tokens} tokens"
        + (f", recapture {cell['recapture_tok_per_s']:.1f} tok/s" if "recapture_tok_per_s" in cell else "")
        + (f", peak VRAM {cell['peak_vram_mib']} MiB" if "peak_vram_mib" in cell else "") + f", RSS {cell['host_rss_gb']} GB")
    return cell, refusals


def command_rate(args: argparse.Namespace) -> int:
    started = time.time()
    torch.set_num_threads(int(args.threads))
    dtype = getattr(torch, args.dtype)
    device = args.device
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    s7 = composite_tool("s7_omni1_check")
    T = trainer_module()
    lora = T.SitePlanConfig(rank=16, alpha=16.0, k3_rank=0, trunk_ffn_width=int(args.trunk_ffn_width), gain=False)
    hybrid = HybridD3(Path(args.omni1), Path(args.parent_config),
                      HybridOptions(dtype=dtype, device=device, head_device="cpu" if args.stage == "d3b" else device,
                                    kernels=args.kernels, lora=lora, lora_seed=0))
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "rate", "stage": args.stage, "tool": tool_identity(),
                               "argv": sys.argv[1:], "torch": torch.__version__, "device": device, "dtype": args.dtype,
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hybrid": hybrid.identity(),
                               "lr": 0.0, "s7_omni1_check": peer_record(s7)}
    if device.startswith("cuda"):
        receipt["card"] = torch.cuda.get_device_name(torch.device(device))
    refusals: list[str] = []
    bands_feature = {t: v["feature"] for t, v in hybrid.bands.items()}
    cells = []
    for container in [u.strip() for u in args.units.split(",") if u.strip()]:
        placement = next((p for p in hybrid.layout if p.container == container), None)
        if placement is None:
            raise D3Error(f"unit {container} is not in the layout {[p.container for p in hybrid.layout]}")
        seg = s7.media_segment(container, Path(args.tap_rows), bands_feature, hybrid.cfg)
        seg = trim_segment(seg, int(args.max_rows_per_type), hybrid.media_placeholder)
        seg["source_positions"] = h3_zero_positions(hybrid.cfg, int(seg["ids"].shape[1]))
        cell, cell_refusals = rate_cell(hybrid, placement, seg, stage=args.stage, steps=int(args.steps))
        cells.append(cell)
        refusals.extend(cell_refusals)
    receipt.update({"cells": cells, "refusals": refusals, "verdict": "PASS" if not refusals else "FAIL",
                    "seconds": round(time.time() - started, 1)})
    sha = T.write_json_atomic(Path(args.receipt), receipt)
    log(f"D3 RATE {receipt['verdict']}: receipt {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parity", help="hybrid vs the Chimera charted forward on OMNI1, CPU fp32")
    p.add_argument("--omni1", required=True)
    p.add_argument("--parent-config", required=True, help="HF dir whose config.json P1 widens (the Q38 parent)")
    p.add_argument("--tap-rows", required=True, help="S5 host rows root (<unit>/paired_rows.*), s7_omni1_check's input")
    p.add_argument("--receipt", required=True)
    p.add_argument("--max-rows-per-type", type=int, default=24)
    p.add_argument("--tol-rel", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("synthetic", help="tiny OMNI-shaped child: parity, grad parity, D3a-d losses decrease")
    s.add_argument("--receipt", required=True)
    s.add_argument("--steps", type=int, default=40)
    s.add_argument("--decrease", type=float, default=0.7, help="last loss must be below first x this")
    r = sub.add_parser("rate", help="lr = 0 fwd+bwd rate on real OMNI1 weights")
    r.add_argument("--omni1", required=True)
    r.add_argument("--parent-config", required=True)
    r.add_argument("--tap-rows", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    r.add_argument("--kernels", default="torch", choices=("torch", "auto"))
    r.add_argument("--stage", default="d3b", choices=("d3b", "d3d"))
    r.add_argument("--units", default="ace_dit_L12,hy3d_dit_L10")
    r.add_argument("--max-rows-per-type", type=int, default=0)
    r.add_argument("--steps", type=int, default=3)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--trunk-ffn-width", type=int, default=4352)
    r.add_argument("--receipt", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "parity":
            return command_parity(args)
        if args.command == "synthetic":
            return command_synthetic(args)
        if args.command == "rate":
            return command_rate(args)
    except D3Error as error:
        log(f"REFUSED: {error}")
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
