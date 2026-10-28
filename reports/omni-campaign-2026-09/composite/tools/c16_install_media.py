#!/usr/bin/env python3
"""c16_install_media.py -- S7 of COMPOSITE step 1: attach the media sources to the union child.

Takes the S5 Chimera child (`--child`) and writes ONE new Chimera checkpoint (`--out`)
plus `composite_manifest.json`:

  * C15 / C11 interface rows (`--c15 DIR`, `--c11 DIR`) go INTO the counted native
    interface: `native_media_bridge.feature_proj.weight[:, band]`,
    `native_media_bridge.type_embed.weight[type]`, `native_media_head.out.weight[band, :]`
    and the head's `type_affine` shift for that type (fp64 least squares of the decoder
    bias through the written head rows). No accounting slack: those parameters exist
    already (PROBE.S0 counted them). Precondition, asserted per type: the bridge's
    `type_feature_scale` row over the band and the SCALE halves of the bridge's and the
    head's `type_affine` rows are zero (else the written rows would be pre-scaled without
    a receipt; `--allow-nonzero-type-affine` records instead of refusing).
  * C18 music-code rows (`--c18 FILE`) go into the DECLARED `music_codes` band of
    `embed.weight` and `lm_head.weight`; the rows must be zero before (S0 declared them
    zero) and the table must be exactly the band's width.
  * C16 charted 4x blocks (`--blocks DIR ...`) are inserted as `charted` blocks (branch
    claude/omni-composite of chimera-omni) right after the depth-fraction-paired
    gated-GQA refinement layer (`--pair-by depth-fraction`) or after the container's
    `insert_after` (`--pair-by explicit`). `--arm insert` is the fold of record.
    Every adaLN container ships the CAPTURED modulation (`time_reference.safetensors`:
    sigma -> [6, width], the parent's `temb` at the capture's noise level); the installer
    rebuilds the spec's time path (`ChartedSourceBlock.time_path`: `time_embed` plus, for
    ACE, `time_embed_r`) and refuses the block unless it reproduces the reference within
    `--time-reference-rel-tol` (1e-5). This is the gate that catches a dropped timestep
    path -- the ACE block with `time_embed` alone ran at ~10 % of the parent's modulation.
    H3 `h3_curve6` instead carries captured `[n,3,6,width]` modulation in video/text/audio
    order and `[n,curve_dim]` curve references. Both are checked at relL2 <= 1e-5.
    Its saved curve, projection and RoPE coefficients stay byte-exact F32 even inside
    a BF16 child. Counted shared coefficients, source tensor digests and explicit
    `h3_thw_v1` text/AV position requirements are checked and carried into the manifest;
    the installer does not claim the caller's coordinate mapping has been verified.
    `--arm blend` is a DIAGNOSTIC push, not a registrable fold: each block's SwiGLU rows
    are pushed through the charts into the paired trunk FFN (`ffn.up` / `ffn.down`, plain
    SiLU-GLU only; a `situ` trunk is refused). Its child is `3 x keep x hidden` per block,
    so at hidden 5,120 no source narrower than the trunk can meet the >= 4x component
    rule (ACE 0.625, H3 / QIMG 0.417); such components go on the register line as
    `--passed-through`, never as `--component`. The registered fallback arm is
    `c10_union_install.py --extra-blocks` (a union coreset at the S5 width).

Charts: a container without `charts.safetensors` is fitted from `--tap-rows DIR`
(`<block>/paired_rows.safetensors`, `host_rows` / `parent_rows`) with
`mediafold.c16_transplant.fit_charts` (ridge, fp64).

Memory and disk: the child is opened as LAZY views (index + shard headers; bytes are read
per tensor when touched and hashed in streaming chunks on verify), and the output shards
are written by streaming untouched tensors straight from the input shards -- only the
tensors this tool modifies (embed / lm_head rows, the interface tensors, blend FFNs) and
the charted state are materialised, so the 27B run stays far under the runbook's 80 GB
instead of holding the 55 GB child plus fp32 working copies. Before anything is written
the tool refuses when the output filesystem would drop below `--min-free-gib` (60) free.
Launch S7 with `--out` under `/mnt/lexar/omni-campaign/composite/...` (the disk rule); the
manifest records whether the ledger's protected-prefix guard would accept that directory.

Every count in the manifest is re-derived from the written safetensors headers: the
output's header total must equal the input's total plus the charted state (plus the
blend rows) EXACTLY, or the tool unlinks what it wrote and leaves `REFUSED.json`. With
nothing to install the output shards are byte-for-byte the input child's (the
zero-media control the tests assert; the streamed writer reproduces
`safetensors_io.write_safetensors` bytes exactly, asserted by test).

Register line: `--parent-checkpoint-id` names the LINEAGE parent (the S5 child); the
fold accounting is priced against `--accounting-parent-directory` (the PARENT-Q38-27BU
record's directory, 27,781,427,952 header parameters, derived by the ledger from its
headers) because every growth stage of the chain (S3, S5, S7) is smaller than the source
and not smaller than its lineage parent; `oc_checkpoint._fold_accounting` refuses a child
that did not shrink against the count it is given. The manifest's `register.reachable`
says whether the emitted line can pass; without an accounting parent it cannot.

Refusals (exit 3, nothing written): `--out` overlaps `--child`; `--out` already holds
safetensors without `--overwrite`, or with `--overwrite` but without this tool's own
`composite_manifest.json` (the flag only removes what this tool wrote); a tree whose
`chimera_omni` is not the one this process imported; a band that does not fit
`native_feature_dim`, overlaps another (without `--allow-band-overlap`) or a type id used
twice; a declared `tensor_sha256` that does not match the file; a band table whose width
is not the declared band, or non-zero target rows (without `--allow-nonzero-band`); a
container whose tensor set / shapes are not exactly `ChartedBlockSpec.parameter_shapes`;
an adaLN container without `time_reference.safetensors`, or whose time path does not
reproduce it; a container without charts when `--tap-rows` has none for it; a blend arm
on a non-SwiGLU block, a non-SiLU trunk, or a per-layer-width result without
`--blend-uniform pad` / `--allow-per-layer-widths`; a child that already carries charted
blocks; an output that would leave the filesystem under `--min-free-gib`.

Formats: `composite/COMPOSITE_S6_INPUT_FORMATS.md`; fixtures: `c16_media_fixtures.py`.
Standing rules: dense only; no quantization (the charted state is written in the
child's storage dtype, bf16 narrowing measured and recorded, `--charted-dtype fp32`
keeps it wide; H3 time/RoPE coefficients ALWAYS stay F32 and mixed storage is explicit);
storage counted in parameters from headers; every claim in the manifest
names the bytes it was read from.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "c16_install_media"
TOOL_VERSION = "1.2.0"
MANIFEST_SCHEMA = "chimera_omni.composite_manifest.v1"
MANIFEST_FILENAME = "composite_manifest.json"
REFUSED_FILENAME = "REFUSED.json"
INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"
TIME_REFERENCE_FILENAME = "time_reference.safetensors"
INTERFACE_SCHEMA = "chimera_omni.composite.interface_set.v1"
BLOCK_SCHEMA = "chimera_omni.composite.charted_block.v1"
DEFAULT_CHIMERA_SRC = "/home/cereal/omni-campaign/import/omni-composite/tree_2ec15a8"
DEFAULT_MEDIAFOLD_ROOT = "/home/cereal/omni-campaign/media"
DEFAULT_LEDGER_ROOT = "/home/cereal/omni-campaign/ledger"
DEFAULT_PARENT_ID = "OC02-F-Q38-27BU-C2K4352K3U"
DEFAULT_CHECKPOINT_ID = "OC02-F-Q38-27BU-OMNI1"
DEFAULT_MIN_FREE_GIB = 60.0
DEFAULT_TIME_REFERENCE_REL_TOL = 1.0e-5
EXIT_REFUSED = 3
CHART_KEYS = ("chart_in.weight", "chart_in.bias", "chart_out.weight")
H3_SHARED_COEFFICIENTS = {"h3_time.curve_table": "adaln_t_table", "attn.rope_inv_freq": "rope.inv_freq"}
H3_FP32_TENSORS = frozenset((*H3_SHARED_COEFFICIENTS, "h3_time.proj.weight", "h3_time.proj.bias"))
H3_MODALITY_ORDER = ["video", "text", "audio"]
H3_SOURCE_POSITIONS = "explicit h3_thw_v1 coordinates for text and AV rows"
H3_COORDINATE_INFERENCE = "forbidden: native thwr_v1 values are not H3 source coordinates"
STREAM_CHUNK = 1 << 24


class Refusal(RuntimeError):
    """A precondition the tool will not run past; exit code 3."""


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# hashing / io helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_with_sidecar(path: Path, body: dict[str, Any]) -> str:
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def elements(shape: tuple[int, ...]) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def header_counts(directory: Path) -> dict[str, int]:
    """{tensor name: elements} over every *.safetensors directly under `directory`."""

    from chimera_omni.importing.safetensors_io import read_header

    counts: dict[str, int] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        header, _base = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in counts:
                raise Refusal(f"tensor {name!r} appears in two shards under {directory}")
            counts[name] = elements(tuple(int(x) for x in meta["shape"]))
    return counts


def read_single(path: Path) -> dict[str, Any]:
    from chimera_omni.importing.safetensors_io import read_safetensors

    if not path.is_file():
        raise Refusal(f"{path} is not a file")
    return read_safetensors(path, verify=False)


def view_hash(view: Any) -> str:
    return str(view.sha256())


def verify_declared_sha256(views: dict[str, Any], declared: Any, *, where: str) -> dict[str, Any]:
    """Compare a file's tensors against the `tensor_sha256` table its JSON declares.

    `TensorView.sha256()` (safetensors_io) is the formula: dtype name + shape + bytes.
    A declared name the file lacks, or a digest that differs, is a refusal; provenance
    that travels with a file has to be checked where the file is consumed.
    """

    if not declared:
        return {"declared": 0, "verified": None, "note": "no tensor_sha256 declared"}
    if not isinstance(declared, dict):
        raise Refusal(f"{where}: tensor_sha256 must map tensor names to digests")
    for name, digest in declared.items():
        if name not in views:
            raise Refusal(f"{where}: tensor_sha256 declares {name!r}, which the file does not carry")
        actual = view_hash(views[name])
        if str(digest) != actual:
            raise Refusal(f"{where}: {name!r} sha256 {actual} != declared {digest}")
    return {"declared": len(declared), "verified": True}


def _torch():
    import torch

    return torch


def peak_rss_bytes() -> int | None:
    """Actual process-lifetime peak resident memory; unknown is never a fake zero."""
    if sys.platform != "win32":
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize) or None


# --------------------------------------------------------------------------
# lazy checkpoint views and the streamed writer
# --------------------------------------------------------------------------


def tensor_hasher(dtype: str, shape: tuple[int, ...]) -> Any:
    """The `safetensors_io._tensor_sha256` prefix (dtype + shape), ready for streamed bytes."""

    digest = hashlib.sha256()
    digest.update(str(dtype).encode("ascii"))
    digest.update(json.dumps([int(extent) for extent in shape], separators=(",", ":")).encode("ascii"))
    return digest


class LazyView:
    """One tensor as it sits in an input shard: read on demand, never held whole unless asked.

    A plain class (not a dataclass): the tests load this module by file path, outside
    ``sys.modules``, where ``dataclasses`` cannot resolve postponed annotations.
    """

    __slots__ = ("path", "offset", "nbytes", "dtype", "shape")

    def __init__(self, path: Path, offset: int, nbytes: int, dtype: str, shape: tuple[int, ...]) -> None:
        self.path = Path(path)
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.dtype = str(dtype)
        self.shape = tuple(int(x) for x in shape)

    def __repr__(self) -> str:
        return f"LazyView({self.path.name}, offset={self.offset}, nbytes={self.nbytes}, {self.dtype} {self.shape})"

    def chunks(self, size: int = STREAM_CHUNK):
        remaining = int(self.nbytes)
        with open(self.path, "rb") as handle:
            handle.seek(int(self.offset))
            while remaining > 0:
                chunk = handle.read(min(size, remaining))
                if not chunk:
                    raise Refusal(f"{self.path}: short read at offset {self.offset + self.nbytes - remaining}")
                remaining -= len(chunk)
                yield chunk

    def read(self) -> bytes:
        return b"".join(self.chunks())

    def view(self) -> Any:
        from chimera_omni.importing.safetensors_io import TensorView

        return TensorView(self.dtype, tuple(self.shape), self.read())

    def torch(self, *, dtype: Any = None) -> Any:
        return self.view().torch(dtype=dtype)

    def sha256(self) -> str:
        digest = tensor_hasher(self.dtype, self.shape)
        for chunk in self.chunks():
            digest.update(chunk)
        return digest.hexdigest()


def open_lazy_checkpoint(directory: Path) -> tuple[dict[str, LazyView], dict[str, Any]]:
    """Every tensor of a sharded Chimera checkpoint as a LazyView, from the index + headers."""

    from chimera_omni.importing.safetensors_io import read_header, read_index

    index = read_index(directory)
    if index is None:
        raise Refusal(f"{directory} has no {INDEX_FILENAME}")
    weight_map: dict[str, str] = dict(index["weight_map"])
    views: dict[str, LazyView] = {}
    for filename in sorted(set(weight_map.values())):
        shard = directory / filename
        if not shard.is_file():
            raise Refusal(f"{directory}: index names {filename}, which is missing")
        header, base = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != filename:
                raise Refusal(f"{shard}: carries {name!r}, which the index maps to {weight_map.get(name)!r}")
            begin, end = (int(x) for x in meta["data_offsets"])
            views[name] = LazyView(shard, int(base) + begin, end - begin, str(meta["dtype"]), tuple(int(x) for x in meta["shape"]))
    missing = sorted(set(weight_map) - set(views))
    if missing:
        raise Refusal(f"{directory}: the index names tensors no shard carries: {missing[:5]}")
    return views, index


def verify_lazy_checkpoint(views: dict[str, LazyView], index: dict[str, Any]) -> int:
    """Streamed per-tensor sha256 against the index (the read path's `verify=True`), shard order."""

    recorded: dict[str, str] = dict(index.get("metadata", {}).get("tensor_sha256", {}))
    checked = 0
    for name, view in sorted(views.items(), key=lambda item: (str(item[1].path), item[1].offset)):
        expected = recorded.get(name)
        if expected is None:
            continue
        actual = view.sha256()
        if actual != expected:
            raise Refusal(f"tensor {name!r} does not match the sha256 recorded in the index ({actual} != {expected})")
        checked += 1
    return checked


class _Sized:
    """`len()` of a tensor's bytes without the bytes: what `_pack_header` asks of a view."""

    __slots__ = ("nbytes",)

    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)

    def __len__(self) -> int:
        return self.nbytes


class _HeaderEntry:
    __slots__ = ("dtype", "shape", "buffer")

    def __init__(self, dtype: str, shape: tuple[int, ...], buffer: _Sized) -> None:
        self.dtype = dtype
        self.shape = shape
        self.buffer = buffer


def write_checkpoint_streamed(
    state: dict[str, Any],
    out_dir: Path,
    *,
    tied: dict[str, str],
    config: dict[str, Any],
    ledger: dict[str, Any],
    source: dict[str, Any],
    chimera_commit: str | None,
    shard_bytes: int,
    metadata: dict[str, str],
) -> dict[str, Any]:
    """`safetensors_io.write_safetensors`, byte for byte, without materialising the state.

    Values are `LazyView`s (copied from the input shards in chunks) or torch tensors /
    `TensorView`s (materialised). `tied` is {alias: owner} from the input index: aliases
    are not written, the index records them, exactly as the reference writer does for
    two names on one storage. Same sorted-name sharding rule, same header packing
    (`_pack_header` is the reference's own), same index / config documents, so the
    zero-media control is byte-identical to the input and the tests assert it.
    """

    from chimera_omni.importing.safetensors_io import CHECKPOINT_SCHEMA, TensorView, _as_view, _pack_header

    for alias, owner in tied.items():
        if owner not in state:
            raise Refusal(f"tied owner {owner!r} (of {alias!r}) is missing from the state")
        if owner in tied:
            raise Refusal(f"tied owner {owner!r} is itself an alias")
    entries: dict[str, Any] = {}
    for name in sorted(state):
        if name in tied:
            continue
        value = state[name]
        if isinstance(value, (LazyView, TensorView)):
            entries[name] = value
        else:
            entries[name], _storage = _as_view(value)

    def nbytes(view: Any) -> int:
        return int(view.nbytes) if isinstance(view, LazyView) else len(view.buffer)

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in sorted(entries):
        size = nbytes(entries[name])
        if current and current_bytes + size > int(shard_bytes):
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current or not shards:
        shards.append(current)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(shards)
    weight_map: dict[str, str] = {}
    tensor_sha256: dict[str, str] = {}
    total_size = 0
    for shard_index, names in enumerate(shards, start=1):
        filename = f"model-{shard_index:05d}-of-{total:05d}.safetensors"
        packed: list[tuple[str, _HeaderEntry, int]] = []
        offset = 0
        for name in names:
            view = entries[name]
            packed.append((name, _HeaderEntry(str(view.dtype), tuple(int(x) for x in view.shape), _Sized(nbytes(view))), offset))
            offset += nbytes(view)
        header = _pack_header(packed, {"format": "pt", "schema": CHECKPOINT_SCHEMA})
        with open(out_dir / filename, "wb", buffering=1 << 22) as handle:
            handle.write(header)
            for name in names:
                view = entries[name]
                digest = tensor_hasher(str(view.dtype), tuple(int(x) for x in view.shape))
                if isinstance(view, LazyView):
                    for chunk in view.chunks():
                        handle.write(chunk)
                        digest.update(chunk)
                else:
                    handle.write(view.buffer)
                    digest.update(view.buffer)
                tensor_sha256[name] = digest.hexdigest()
                weight_map[name] = filename
                total_size += nbytes(view)
    index = {
        "metadata": {
            "schema": CHECKPOINT_SCHEMA,
            "total_size": total_size,
            "tensor_sha256": tensor_sha256,
            "tied_weights": dict(sorted(tied.items())),
            **{str(key): str(value) for key, value in dict(metadata or {}).items()},
        },
        "weight_map": weight_map,
    }
    with open(out_dir / INDEX_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    document = {
        "schema": CHECKPOINT_SCHEMA,
        "config": dict(config) if config is not None else None,
        "ledger": dict(ledger) if ledger is not None else None,
        "source": dict(source) if source is not None else None,
        "chimera_commit": str(chimera_commit) if chimera_commit else None,
    }
    with open(out_dir / CONFIG_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return index


def state_nbytes(state: dict[str, Any], tied: dict[str, str]) -> int:
    from chimera_omni.importing.safetensors_io import ST_ITEMSIZE, TensorView, _torch_dtype_name

    total = 0
    for name, value in state.items():
        if name in tied:
            continue
        if isinstance(value, LazyView):
            total += int(value.nbytes)
        elif isinstance(value, TensorView):
            total += len(value.buffer)
        else:
            total += ST_ITEMSIZE[_torch_dtype_name(value.dtype)] * int(value.numel())
    return total


def free_bytes_under(path: Path) -> tuple[Path, int]:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return probe, int(shutil.disk_usage(probe).free)


# --------------------------------------------------------------------------
# tree identity
# --------------------------------------------------------------------------


def tree_identity(chimera_src: Path) -> dict[str, Any]:
    import chimera_omni

    package_dir = Path(chimera_omni.__file__).resolve().parent
    expected = (chimera_src / "src" / "chimera_omni").resolve()
    if package_dir != expected:
        raise Refusal(
            f"the imported chimera_omni ({package_dir}) is not the one under --chimera-src "
            f"({expected}); set PYTHONPATH to the tree the manifest will name"
        )
    commit = None
    commit_file = chimera_src / "COMMIT"
    if commit_file.is_file():
        for line in commit_file.read_text(encoding="utf-8").splitlines():
            token = line.strip().split()[0] if line.strip() else ""
            if token and not token.startswith("#"):
                commit = token.lower()
                break
    digest = hashlib.sha256()
    files = sorted(p for p in expected.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        digest.update(path.relative_to(expected).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "chimera_src": str(chimera_src),
        "commit": commit,
        "commit_source": "COMMIT file (a declaration, not proof)" if commit else None,
        "package_sha256": digest.hexdigest(),
        "package_files": len(files),
    }


# --------------------------------------------------------------------------
# interface row sets
# --------------------------------------------------------------------------


def load_interface_set(directory: Path, label: str, cfg: Any) -> list[dict[str, Any]]:
    manifest_path = directory / "companions.json"
    if not manifest_path.is_file():
        raise Refusal(f"{label}: {manifest_path} is missing")
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    if body.get("schema") != INTERFACE_SCHEMA:
        raise Refusal(f"{label}: companions.json schema {body.get('schema')!r} != {INTERFACE_SCHEMA!r}")
    if int(body.get("hidden", -1)) != int(cfg.hidden_size):
        raise Refusal(f"{label}: companions.json hidden {body.get('hidden')} != child hidden {cfg.hidden_size}")
    if int(body.get("native_feature_dim", -1)) != int(cfg.native_feature_dim):
        raise Refusal(
            f"{label}: companions.json native_feature_dim {body.get('native_feature_dim')} != child "
            f"native_feature_dim {cfg.native_feature_dim}"
        )
    entries: list[dict[str, Any]] = []
    for raw in body.get("entries", []):
        rows_path = directory / str(raw["rows"])
        if not rows_path.is_file():
            raise Refusal(f"{label}/{raw.get('name')}: rows file {rows_path} is missing")
        fb = [int(v) for v in raw["feature_band"]]
        eb = [int(v) for v in raw.get("emission_band", raw["feature_band"])]
        type_id = int(raw["type_id"])
        if not 1 <= type_id < int(cfg.native_type_count):
            raise Refusal(f"{label}/{raw.get('name')}: type_id {type_id} outside 1..{cfg.native_type_count - 1}")
        for band_name, (b, e) in (("feature_band", fb), ("emission_band", eb)):
            if not 0 <= b < e <= int(cfg.native_feature_dim):
                raise Refusal(
                    f"{label}/{raw.get('name')}: {band_name} [{b}, {e}) does not fit native_feature_dim "
                    f"{cfg.native_feature_dim}"
                )
        views = read_single(rows_path)
        sha_report = verify_declared_sha256(views, raw.get("tensor_sha256"), where=f"{label}/{raw.get('name')}")
        entries.append(
            {
                "set": label,
                "name": str(raw["name"]),
                "source": str(raw.get("source", raw["name"])),
                "type_id": type_id,
                "feature_band": fb,
                "emission_band": eb,
                "rows_path": rows_path,
                "rows_sha256": sha256_file(rows_path),
                "tensor_sha256": {name: view_hash(view) for name, view in sorted(views.items())},
                "tensor_sha256_check": sha_report,
                "rung": raw.get("rung"),
                "companion_decoder": raw.get("companion_decoder"),
                "psnr_absorbed_db": raw.get("psnr_absorbed_db"),
                "psnr_parent_roundtrip_db": raw.get("psnr_parent_roundtrip_db"),
            }
        )
    return entries


def check_bands(entries: list[dict[str, Any]], *, allow_overlap: bool) -> dict[str, Any]:
    """Disjoint bands per side (every pair compared) and one type id per install, unless waived."""

    overlaps: list[dict[str, Any]] = []
    for side in ("feature_band", "emission_band"):
        spans = [(e[side][0], e[side][1], e["name"]) for e in entries]
        for i in range(len(spans)):
            b0, e0, n0 = spans[i]
            for j in range(i + 1, len(spans)):
                b1, e1, n1 = spans[j]
                low, high = max(b0, b1), min(e0, e1)
                if low < high:
                    overlaps.append({"side": side, "a": n0, "b": n1, "span": [low, high]})
    types: dict[int, list[str]] = {}
    for entry in entries:
        types.setdefault(entry["type_id"], []).append(entry["name"])
    duplicate_types = {t: names for t, names in types.items() if len(names) > 1}
    if (overlaps or duplicate_types) and not allow_overlap:
        raise Refusal(
            "interface bands must be disjoint and every type id written once: overlaps "
            f"{overlaps}, duplicate type ids {duplicate_types}; pass --allow-band-overlap to superpose "
            "(the manifest records the superposition)"
        )
    return {"overlaps": overlaps, "duplicate_type_ids": {str(k): v for k, v in duplicate_types.items()}}


# --------------------------------------------------------------------------
# charted block containers
# --------------------------------------------------------------------------


def validate_h3_contract(body: dict[str, Any], spec: Any, views: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Validate the converter's counted H3 coefficients and explicit runtime contract.

    This verifies the installed bytes against carried source tensor digests; it
    does not pretend to reopen the full parent checkpoint or prove media quality.
    """
    metadata = spec.metadata
    requirements = body.get("runtime_requirements", {})
    if (not isinstance(metadata, dict) or metadata.get("source_position_schema") != "h3_thw_v1"
            or not isinstance(requirements, dict) or requirements.get("source_positions") != H3_SOURCE_POSITIONS
            or requirements.get("coordinate_inference") != H3_COORDINATE_INFERENCE):
        raise Refusal(f"{directory}: H3 requires explicit h3_thw_v1 source positions for text and AV; coordinate inference is forbidden")
    reference = body.get("time_reference", {})
    if not isinstance(reference, dict) or reference.get("modality_order") != H3_MODALITY_ORDER:
        raise Refusal(f"{directory}: H3 time reference must declare modality order video/text/audio")
    for name in H3_FP32_TENSORS:
        if views[name].dtype != "F32":
            raise Refusal(f"{directory}: H3 source time/RoPE coefficient {name} must be stored F32, not {views[name].dtype}")
    provenance = body.get("provenance", {})
    if not isinstance(provenance, dict):
        raise Refusal(f"{directory}: H3 source coefficient provenance must be an object")
    globals_sha = provenance.get("source_globals_tensor_sha256")
    if not isinstance(globals_sha, dict) or set(globals_sha) != set(H3_SHARED_COEFFICIENTS.values()):
        raise Refusal(f"{directory}: H3 requires source tensor digests for the retained curve and RoPE coefficients")
    coefficient_counts = {}
    for target, source in H3_SHARED_COEFFICIENTS.items():
        if view_hash(views[target]) != globals_sha[source]:
            raise Refusal(f"{directory}: retained H3 coefficient {target} differs from its source tensor digest")
        coefficient_counts[target] = elements(views[target].shape)
    shared = sum(coefficient_counts.values())
    total = sum(elements(view.shape) for view in views.values())
    accounting = body.get("accounting", {})
    if not isinstance(accounting, dict):
        raise Refusal(f"{directory}: H3 accounting must declare block and retained coefficient counts")
    expected = {"retained_shared_coefficients": shared, "child_block_parameters": total - shared,
                "child_mlp_parameters": spec.mlp_parameter_count()}
    for name, count in expected.items():
        if type(accounting.get(name)) is not int or accounting[name] != count:
            raise Refusal(f"{directory}: H3 accounting.{name} must equal its header-derived count {count}")
    for name in ("parent_mlp_parameters", "parent_block_parameters"):
        if type(accounting.get(name)) is not int or accounting[name] <= 0:
            raise Refusal(f"{directory}: H3 accounting.{name} must be a positive integer")
    if (accounting["parent_mlp_parameters"] < 4 * expected["child_mlp_parameters"]
            or accounting["parent_block_parameters"] != total - shared - expected["child_mlp_parameters"] + accounting["parent_mlp_parameters"]):
        raise Refusal(f"{directory}: H3 parent/child block counts or 4x MLP component counts are inconsistent")
    return {"coefficient_header_parameters": coefficient_counts, "retained_shared_coefficients": shared,
            "block_with_retained_coefficients_parameters": total, "source_coefficient_tensor_sha256_verified": True,
            "preserved_f32_tensors": sorted(H3_FP32_TENSORS),
            "source_position_schema": "h3_thw_v1", "source_positions_required": True,
            "position_mapping_verified": False, "note": "caller must supply and verify actual text/AV source coordinates; no media capability proof"}


def require_complete_hashes(views: dict[str, Any], declared: Any, *, where: str) -> None:
    if not isinstance(declared, dict) or set(declared) != set(views):
        raise Refusal(f"{where}: H3 requires complete declared tensor_sha256 evidence for every tensor")


def load_container(directory: Path, cfg: Any) -> dict[str, Any]:
    from chimera_omni.charted import ChartedBlockSpec

    body_path = directory / "charted_block.json"
    tensors_path = directory / "charted_block.safetensors"
    charts_path = directory / "charts.safetensors"
    reference_path = directory / TIME_REFERENCE_FILENAME
    if not body_path.is_file() or not tensors_path.is_file():
        raise Refusal(f"{directory}: charted_block.json and charted_block.safetensors are both required")
    body = json.loads(body_path.read_text(encoding="utf-8"))
    if body.get("schema") != BLOCK_SCHEMA:
        raise Refusal(f"{directory}: charted_block.json schema {body.get('schema')!r} != {BLOCK_SCHEMA!r}")
    fields = dict(body["spec"])
    if "layer_index" in fields:
        raise Refusal(f"{directory}: the container must not fix layer_index; the installer pairs the block")
    fields["type_ids"] = tuple(int(v) for v in fields["type_ids"])
    try:
        spec = ChartedBlockSpec(layer_index=0, **fields)
        spec.validate()
    except (TypeError, ValueError) as error:
        raise Refusal(f"{directory}: invalid charted spec: {error}") from error
    for type_id in spec.type_ids:
        if not 1 <= int(type_id) < int(cfg.native_type_count):
            raise Refusal(f"{directory}: type id {type_id} outside 1..{cfg.native_type_count - 1}")
    expected = spec.parameter_shapes("", int(cfg.hidden_size))
    block_expected = {k: v for k, v in expected.items() if not k.startswith("chart_")}
    chart_expected = {k: v for k, v in expected.items() if k.startswith("chart_")}
    declared = body.get("tensor_sha256") or {}
    if declared and not isinstance(declared, dict):
        raise Refusal(f"{directory}: tensor_sha256 must map file names to per-tensor digests")

    views = read_single(tensors_path)
    is_h3 = spec.modulation == "h3_curve6"
    if is_h3:
        require_complete_hashes(views, declared.get("charted_block.safetensors"), where=str(tensors_path))
    sha_checks = {"charted_block.safetensors": verify_declared_sha256(views, declared.get("charted_block.safetensors"), where=f"{directory}/charted_block.safetensors")}
    if set(views) != set(block_expected):
        raise Refusal(
            f"{directory}: charted_block.safetensors tensor set is not the canonical set for this spec; "
            f"missing {sorted(set(block_expected) - set(views))}, unexpected {sorted(set(views) - set(block_expected))}"
        )
    tensors: dict[str, Any] = {}
    for name, view in views.items():
        if tuple(view.shape) != tuple(block_expected[name]):
            raise Refusal(f"{directory}: {name} has shape {tuple(view.shape)}, spec says {block_expected[name]}")
        tensors[name] = view.torch(dtype=_torch().float32)
        if is_h3 and not bool(_torch().isfinite(tensors[name]).all()):
            raise Refusal(f"{directory}: non-finite H3 source tensor {name}")
    h3_contract = validate_h3_contract(body, spec, views, directory) if is_h3 else None
    charts: dict[str, Any] | None = None
    dropped_chart_keys: list[str] = []
    if charts_path.is_file():
        chart_views = read_single(charts_path)
        if is_h3:
            require_complete_hashes(chart_views, declared.get("charts.safetensors"), where=str(charts_path))
        sha_checks["charts.safetensors"] = verify_declared_sha256(chart_views, declared.get("charts.safetensors"), where=f"{directory}/charts.safetensors")
        charts = {}
        for name in chart_expected:
            if name not in chart_views:
                raise Refusal(f"{directory}: charts.safetensors lacks {name}")
            if tuple(chart_views[name].shape) != tuple(chart_expected[name]):
                raise Refusal(
                    f"{directory}: {name} has shape {tuple(chart_views[name].shape)}, spec says {chart_expected[name]}"
                )
            charts[name] = chart_views[name].torch(dtype=_torch().float32)
            if is_h3 and not bool(_torch().isfinite(charts[name]).all()):
                raise Refusal(f"{directory}: non-finite H3 chart tensor {name}")
        dropped_chart_keys = sorted(set(chart_views) - set(chart_expected))
    time_reference: dict[str, Any] | None = None
    if spec.modulation in ("adaln6", "h3_curve6"):
        if not reference_path.is_file():
            raise Refusal(
                f"{directory}: a {spec.modulation} container must ship {TIME_REFERENCE_FILENAME}; "
                "the installer refuses a time path it cannot check against the capture"
            )
        reference_views = read_single(reference_path)
        if is_h3:
            require_complete_hashes(reference_views, declared.get(TIME_REFERENCE_FILENAME), where=str(reference_path))
        sha_checks[TIME_REFERENCE_FILENAME] = verify_declared_sha256(reference_views, declared.get(TIME_REFERENCE_FILENAME), where=f"{directory}/{TIME_REFERENCE_FILENAME}")
        for key in ("sigma", "modulation"):
            if key not in reference_views:
                raise Refusal(f"{reference_path} lacks {key}")
        sigma = reference_views["sigma"].torch(dtype=_torch().float32)
        if is_h3 and (sigma.ndim != 1 or not bool(_torch().isfinite(sigma).all()) or bool(((sigma < 0) | (sigma > 1)).any())):
            raise Refusal(f"{reference_path}: H3 sigma must be a finite [n] vector in [0, 1]")
        sigma = sigma.reshape(-1)
        modulation = reference_views["modulation"].torch(dtype=_torch().float32)
        expected_shape = (int(sigma.numel()), *((3,) if is_h3 else ()), 6, int(spec.width))
        if sigma.numel() == 0 or tuple(modulation.shape) != expected_shape:
            raise Refusal(
                f"{reference_path}: modulation must be {expected_shape}, got {tuple(modulation.shape)}"
            )
        time_reference = {
            "file": str(reference_path),
            "sha256": sha256_file(reference_path),
            "sigma": sigma,
            "modulation": modulation,
            "declared": body.get("time_reference", {}),
        }
        if is_h3:
            if "curve" not in reference_views or tuple(reference_views["curve"].shape) != (len(sigma), spec.h3_curve_dim):
                raise Refusal(f"{reference_path}: H3 curve reference must be [n, {spec.h3_curve_dim}]")
            time_reference["curve"] = reference_views["curve"].torch(dtype=_torch().float32)
    elif reference_path.is_file():
        raise Refusal(f"{directory}: {TIME_REFERENCE_FILENAME} present on a block without adaLN modulation")
    accounting = dict(body.get("accounting", {}))
    if int(accounting.get("child_mlp_parameters", spec.mlp_parameter_count())) != spec.mlp_parameter_count():
        raise Refusal(
            f"{directory}: accounting.child_mlp_parameters {accounting.get('child_mlp_parameters')} != the spec's "
            f"MLP count {spec.mlp_parameter_count()}"
        )
    insert_after = body.get("insert_after")
    return {
        "directory": directory,
        "name": directory.name,
        "spec": spec,
        "spec_fields": fields,
        "tensors": tensors,
        "charts": charts,
        "chart_source": "container" if charts is not None else None,
        "dropped_chart_keys": dropped_chart_keys,
        "time_reference": time_reference,
        "tensor_sha256_check": sha_checks,
        "accounting": accounting,
        "h3_source_contract": h3_contract,
        "runtime_requirements": body.get("runtime_requirements", {}),
        "capture_deviations": body.get("capture_deviations", []),
        "provenance": body.get("provenance", {}),
        "holdout": body.get("holdout", {}),
        "insert_after": None if insert_after is None else int(insert_after),
        "files": {
            "charted_block.json": sha256_file(body_path),
            "charted_block.safetensors": sha256_file(tensors_path),
            "charts.safetensors": sha256_file(charts_path) if charts_path.is_file() else None,
            TIME_REFERENCE_FILENAME: sha256_file(reference_path) if reference_path.is_file() else None,
        },
    }


def build_source_block(spec: Any, hidden: int, tensors: dict[str, Any], charts: dict[str, Any]) -> Any:
    """The branch's ChartedSourceBlock carrying exactly the container's state (strict)."""

    from chimera_omni.modeling.charted import ChartedSourceBlock

    block = ChartedSourceBlock(dataclasses.replace(spec, layer_index=0), int(hidden)).eval()
    block.load_state_dict({**tensors, **charts}, strict=True)
    return block


def verify_time_reference(container: dict[str, Any], hidden: int, *, rel_tol: float) -> dict[str, Any] | None:
    """The spec's time path (as the installed block will run it) must reproduce the captured temb."""

    reference = container.get("time_reference")
    if reference is None:
        return None
    is_h3 = container["spec"].modulation == "h3_curve6"
    if not math.isfinite(rel_tol) or rel_tol <= 0 or (is_h3 and rel_tol > DEFAULT_TIME_REFERENCE_REL_TOL):
        raise Refusal("time reference tolerance must be finite and positive; H3 evidence cannot be loosened beyond 1e-5")
    torch = _torch()
    block = build_source_block(container["spec"], hidden, container["tensors"], container["charts"])
    with torch.no_grad():
        got = block.time_path(reference["sigma"]).to(torch.float32)
    expected = reference["modulation"]
    if got.shape != expected.shape:
        raise Refusal(f"{container['name']}: rebuilt time path shape differs from the captured modulation")
    if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(got).all()):
        raise Refusal(f"{container['name']}: non-finite values in the time reference or the time path")
    norm = float(expected.double().norm())
    if norm == 0.0:
        raise Refusal(f"{container['name']}: the captured modulation is all zero; a reference must carry the parent's temb")
    diff = (got - expected).abs()
    rel = float((got.double() - expected.double()).norm() / norm)
    chunk_norms = expected.double().norm(dim=(0, 3) if is_h3 else (0, 2)).tolist()
    report = {
        "file": reference["file"],
        "sha256": reference["sha256"],
        "sigma": [float(v) for v in reference["sigma"]],
        "rel_l2": rel,
        "max_abs": float(diff.max()),
        "reference_norm": norm,
        "reference_chunk_norms": chunk_norms,
        "reference_shape": list(expected.shape),
        "tolerance_rel_l2": float(rel_tol),
        "time_embed_r": bool(container["spec"].time_embed_r),
        "declared": reference.get("declared"),
        "verified": rel <= float(rel_tol),
    }
    if is_h3:
        with torch.no_grad():
            curve = block.h3_time.curve(reference["sigma"])
        expected_curve = reference["curve"]
        if (curve.shape != expected_curve.shape or not bool(torch.isfinite(curve).all())
                or not bool(torch.isfinite(expected_curve).all()) or float(expected_curve.double().norm()) == 0):
            raise Refusal(f"{container['name']}: invalid or empty H3 captured curve evidence")
        curve_rel = float((curve.double() - expected_curve.double()).norm() / expected_curve.double().norm())
        report.update(modality_order=H3_MODALITY_ORDER.copy(), curve_rel_l2=curve_rel,
                      curve_max_abs=float((curve - expected_curve).abs().max()))
        if not math.isfinite(curve_rel) or curve_rel > rel_tol:
            raise Refusal(f"{container['name']}: H3 curve does not reproduce captured reference (rel_l2 {curve_rel:.3e})")
    if not math.isfinite(rel) or rel > float(rel_tol):
        raise Refusal(
            f"{container['name']}: the spec's time path does not reproduce the captured modulation "
            f"(rel_l2 {rel:.3e} > {rel_tol:.1e}, max_abs {float(diff.max()):.3e}); the block would run in a "
            "modulation regime the holdout never measured (a dropped time_embed_r, a permuted chunk order or "
            "the wrong sinusoid). Fix the capture / conversion, not the tolerance."
        )
    container["time_reference_check"] = report
    return report


def fit_container_charts(container: dict[str, Any], tap_root: Path | None, *, seed: int, mediafold_root: Path) -> None:
    """Charts from paired tap rows through mediafold's fit_charts (ridge, fp64)."""

    if tap_root is None:
        raise Refusal(f"{container['name']}: no charts.safetensors and no --tap-rows to fit them from")
    rows_path = tap_root / container["name"] / "paired_rows.safetensors"
    if not rows_path.is_file():
        raise Refusal(f"{container['name']}: no charts.safetensors and {rows_path} is missing")
    if str(mediafold_root) not in sys.path:
        sys.path.insert(0, str(mediafold_root))
    try:
        from mediafold.c16_transplant import fit_charts
    except ImportError as error:
        raise Refusal(f"mediafold is not importable from {mediafold_root}: {error}") from error
    torch = _torch()
    views = read_single(rows_path)
    for key in ("host_rows", "parent_rows"):
        if key not in views:
            raise Refusal(f"{rows_path} lacks {key}")
    host = views["host_rows"].torch(dtype=torch.float64)
    parent = views["parent_rows"].torch(dtype=torch.float64)
    spec = container["spec"]
    if host.shape[1] != container["hidden"] or parent.shape[1] != int(spec.width):
        raise Refusal(
            f"{container['name']}: tap rows are [{host.shape[1]}] host / [{parent.shape[1]}] parent, the block "
            f"needs [{container['hidden']}] / [{spec.width}]"
        )
    pair = fit_charts(host, parent, method="ridge", seed=int(seed))
    charts = {
        "chart_in.weight": pair.chart_in.weight.to(torch.float32),
        "chart_in.bias": pair.chart_in.bias.to(torch.float32),
        "chart_out.weight": pair.chart_out.weight.to(torch.float32),
    }
    if not spec.chart_in_bias:
        charts.pop("chart_in.bias")
    container["charts"] = charts
    container["chart_source"] = "fit_charts(ridge) on --tap-rows"
    container["chart_fit"] = {
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "method": pair.method,
        "roundtrip_scores": pair.notes.get("roundtrip_scores"),
        "forward_scores": pair.notes.get("forward_scores"),
        "split": pair.notes.get("split"),
        "chart_out_bias_dropped": True,
    }
    container["dropped_chart_keys"] = ["chart_out.bias"]


def pair_containers(
    containers: list[dict[str, Any]], cfg: Any, *, pair_by: str,
    allow_existing_charted: bool = False,
) -> None:
    """Assign `insert_after` (an index in the CURRENT host's trunk) to every container.

    Existing charted children require an insert-only opt-in and explicit pairing.
    A new block after a trunk anchor precedes any old charted blocks at that anchor.
    Donor indices/context identities are preserved, never inferred from host indices.
    """

    has_charted = any(cfg.is_charted(index) for index in range(int(cfg.num_layers)))
    if has_charted and not (allow_existing_charted and pair_by == "explicit"):
        raise Refusal("the child already carries charted blocks; this tool installs onto a plain trunk")
    refinement = [index for index in range(int(cfg.num_layers)) if cfg.block_kind(index) == "gated_gqa"]
    depth = max(int(cfg.num_layers) - 1, 1)
    for container in containers:
        spec = container["spec"]
        if pair_by == "explicit":
            if container["insert_after"] is None:
                raise Refusal(f"{container['name']}: --pair-by explicit needs insert_after in charted_block.json")
            after = int(container["insert_after"])
            if not 0 <= after < int(cfg.num_layers):
                raise Refusal(f"{container['name']}: insert_after {after} outside 0..{cfg.num_layers - 1}")
            if cfg.is_charted(after):
                raise Refusal(f"{container['name']}: insert_after {after} names an existing charted block, not a trunk layer")
            container["pairing"] = {"rule": "explicit", "insert_after": after}
            container["insert_after"] = after
            continue
        if not refinement:
            raise Refusal("depth-fraction pairing needs gated_gqa refinement layers in the trunk; none found")
        fraction = float(spec.source_depth_fraction)
        if fraction < 0.0:
            raise Refusal(f"{container['name']}: source_depth_fraction is unknown; depth-fraction pairing needs it")
        best = min(refinement, key=lambda index: (abs(index / depth - fraction), index))
        container["insert_after"] = best
        container["pairing"] = {
            "rule": "depth-fraction -> nearest gated_gqa refinement layer",
            "source_depth_fraction": fraction,
            "trunk_depth_fraction": best / depth,
            "insert_after": best,
            "candidates": refinement,
        }
    containers.sort(key=lambda c: (int(c["insert_after"]), c["spec"].source, c["name"]))


# --------------------------------------------------------------------------
# the install
# --------------------------------------------------------------------------


class Installer:
    def __init__(self, args: argparse.Namespace, argv: list[str] | None = None) -> None:
        self.args = args
        self.argv = list(sys.argv[1:] if argv is None else argv)
        self.torch = _torch()
        self.child = _resolve(args.child)
        self.out = _resolve(args.out)
        self.work: dict[str, Any] = {}
        self.record: dict[str, Any] = {}

    # -- child ---------------------------------------------------------------

    def load_child(self) -> None:
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.importing.safetensors_io import read_checkpoint_config

        if _overlaps(self.child, self.out):
            raise Refusal(f"--out {self.out} overlaps --child {self.child}")
        if not (self.child / INDEX_FILENAME).is_file() or not (self.child / CONFIG_FILENAME).is_file():
            raise Refusal(f"{self.child} is not a Chimera checkpoint directory (index + config required)")
        if self.out.exists() and any(self.out.glob("*.safetensors")):
            if not self.args.overwrite:
                raise Refusal(f"--out {self.out} already holds safetensors; pass --overwrite")
            prior = self.out / MANIFEST_FILENAME
            prior_tool = None
            if prior.is_file():
                try:
                    prior_tool = json.loads(prior.read_text(encoding="utf-8")).get("tool", {}).get("name")
                except (ValueError, AttributeError):
                    prior_tool = None
            if prior_tool != TOOL_NAME:
                raise Refusal(
                    f"--overwrite only removes what this tool wrote, and {self.out} carries no {MANIFEST_FILENAME} "
                    f"from {TOOL_NAME} (found tool {prior_tool!r}); refusing to unlink another checkpoint"
                )
        document = read_checkpoint_config(self.child)
        self.cfg = ChimeraConfig.from_manifest(document["config"])
        self.child_document = document
        log(f"opening {self.child} (lazy views; verify={not self.args.skip_verify})")
        self.views, index = open_lazy_checkpoint(self.child)
        self.child_index = index
        self.tied: dict[str, str] = dict(index.get("metadata", {}).get("tied_weights", {}))
        verified = 0
        if not self.args.skip_verify:
            verified = verify_lazy_checkpoint(self.views, index)
        self.state: dict[str, Any] = dict(self.views)
        for alias, owner in self.tied.items():
            if owner not in self.state:
                raise Refusal(f"tied owner {owner!r} is missing from the child")
            self.state[alias] = self.state[owner]
        self.input_counts = {name: elements(view.shape) for name, view in self.views.items()}
        self.input_total = sum(self.input_counts.values())
        anchor = self.views.get("embed.weight")
        if anchor is None:
            raise Refusal("child has no embed.weight")
        self.storage_dtype_name = anchor.dtype
        self.storage_dtype = {"BF16": self.torch.bfloat16, "F16": self.torch.float16, "F32": self.torch.float32}.get(
            anchor.dtype
        )
        if self.storage_dtype is None:
            raise Refusal(f"unsupported child storage dtype {anchor.dtype}")
        self.record["child"] = {
            "directory": str(self.child),
            "index_sha256": sha256_file(self.child / INDEX_FILENAME),
            "config_sha256": sha256_file(self.child / CONFIG_FILENAME),
            "tensors": len(self.views),
            "header_parameters": self.input_total,
            "header_bytes": state_nbytes(self.views, {}),
            "tensors_sha256_verified": verified if not self.args.skip_verify else None,
            "storage_dtype": anchor.dtype,
            "tied_weights": dict(sorted(self.tied.items())),
            "chimera_commit": document.get("chimera_commit"),
            "shards": sorted(str(p.name) for p in self.child.glob("*.safetensors")),
        }
        log(f"child: {len(self.views)} tensors, {self.input_total:,} header parameters, {anchor.dtype}, {verified} tensors sha-verified")

    def working(self, name: str) -> Any:
        """fp32 working copy of a child tensor (read now); written back in the storage dtype at the end."""

        if name not in self.work:
            if name not in self.views:
                raise Refusal(f"child has no tensor named {name!r}")
            self.work[name] = self.views[name].torch(dtype=self.torch.float32).clone()
        return self.work[name]

    # -- interface rows --------------------------------------------------------

    def install_interface(self) -> None:
        entries: list[dict[str, Any]] = []
        for label, directory in (("c15", self.args.c15), ("c11", self.args.c11)):
            if directory:
                entries.extend(load_interface_set(_resolve(directory), label, self.cfg))
        if not entries:
            self.record["interface"] = {"entries": [], "note": "no interface row sets given"}
            return
        band_report = check_bands(entries, allow_overlap=self.args.allow_band_overlap)
        torch = self.torch
        hidden = int(self.cfg.hidden_size)
        feature_proj = self.working("native_media_bridge.feature_proj.weight")
        type_embed = self.working("native_media_bridge.type_embed.weight")
        head_out = self.working("native_media_head.out.weight")
        type_affine = self.working("native_media_head.type_affine.weight")
        # read-only preconditions (not written, so not working copies)
        type_feature_scale = self.views["native_media_bridge.type_feature_scale.weight"].torch(dtype=torch.float32)
        bridge_affine = self.views["native_media_bridge.type_affine.weight"].torch(dtype=torch.float32)
        written_feature = torch.zeros(feature_proj.shape[1], dtype=torch.bool)
        written_emission = torch.zeros(head_out.shape[0], dtype=torch.bool)
        reports: list[dict[str, Any]] = []
        for entry in entries:
            views = read_single(entry["rows_path"])
            b, e = entry["feature_band"]
            eb, ee = entry["emission_band"]
            needed = {
                "bridge.feature_proj.weight": (hidden, e - b),
                "bridge.type_embed_delta": (hidden,),
                "head.out.weight": (ee - eb, hidden),
                "head.type_affine_shift": (ee - eb,),
            }
            for key, shape in needed.items():
                if key not in views:
                    raise Refusal(f"{entry['set']}/{entry['name']}: composed interface lacks {key}")
                if tuple(views[key].shape) != shape:
                    raise Refusal(
                        f"{entry['set']}/{entry['name']}: {key} is {tuple(views[key].shape)}, the band needs {shape}"
                    )
            E = views["bridge.feature_proj.weight"].torch(dtype=torch.float32)
            delta = views["bridge.type_embed_delta"].torch(dtype=torch.float32)
            D = views["head.out.weight"].torch(dtype=torch.float32)
            bias = views["head.type_affine_shift"].torch(dtype=torch.float32)
            type_id = entry["type_id"]
            # the rows are composed for an interface that applies them as-is: the bridge's
            # per-type feature scale over this band and the SCALE halves of both type_affine
            # rows must be zero (S0 initialises them zero; the S5 child is not S0)
            preconditions = {
                "native_media_bridge.type_feature_scale.weight[type, band]": int(type_feature_scale[type_id, b:e].ne(0).sum()),
                "native_media_bridge.type_affine.weight[type, scale half]": int(bridge_affine[type_id, :hidden].ne(0).sum()),
                "native_media_head.type_affine.weight[type, scale half]": int(type_affine[type_id, :hidden].ne(0).sum()),
            }
            recorded_norms = {
                "native_media_bridge.type_affine.weight[type, shift half] norm": float(bridge_affine[type_id, hidden:].norm()),
                "native_media_head.type_affine.weight[type, shift half] previous norm": float(type_affine[type_id, hidden:].norm()),
            }
            violated = {k: v for k, v in preconditions.items() if v}
            if violated and not self.args.allow_nonzero_type_affine:
                raise Refusal(
                    f"{entry['set']}/{entry['name']} (type {type_id}): non-zero scale rows would pre-scale the composed "
                    f"interface rows without a receipt: {violated}; pass --allow-nonzero-type-affine to record and proceed"
                )
            previous_feature_norm = float(feature_proj[:, b:e].norm())
            previous_emission_norm = float(head_out[eb:ee, :].norm())
            if bool(written_feature[b:e].any()):
                feature_proj[:, b:e] += E
                feature_mode = "superposed (band overlap waived)"
            else:
                feature_proj[:, b:e] = E
                feature_mode = "replaced"
            written_feature[b:e] = True
            if bool(written_emission[eb:ee].any()):
                head_out[eb:ee, :] += D
                emission_mode = "superposed (band overlap waived)"
            else:
                head_out[eb:ee, :] = D
                emission_mode = "replaced"
            written_emission[eb:ee] = True
            type_embed[type_id] += delta
            # the decoder's bias lives in feature space; the head's shift in hidden space:
            # solve D s = bias in fp64 through the rows just written for this band
            rows = head_out[eb:ee, :].to(torch.float64)
            solution = torch.linalg.lstsq(rows, bias.to(torch.float64).unsqueeze(1)).solution.squeeze(1)
            residual = float((rows @ solution - bias.to(torch.float64)).norm() / bias.to(torch.float64).norm().clamp_min(1e-30))
            type_affine[type_id, hidden:] += solution.to(torch.float32)
            reports.append(
                {
                    "set": entry["set"],
                    "name": entry["name"],
                    "source": entry["source"],
                    "type_id": type_id,
                    "feature_band": [b, e],
                    "emission_band": [eb, ee],
                    "rows_sha256": entry["rows_sha256"],
                    "tensor_sha256": entry["tensor_sha256"],
                    "tensor_sha256_check": entry["tensor_sha256_check"],
                    "rung": entry["rung"],
                    "companion_decoder": entry["companion_decoder"],
                    "psnr_absorbed_db": entry["psnr_absorbed_db"],
                    "psnr_parent_roundtrip_db": entry["psnr_parent_roundtrip_db"],
                    "preconditions": {"nonzero_scale_values": preconditions, "violated": bool(violated), "waived": bool(violated) and bool(self.args.allow_nonzero_type_affine), **recorded_norms},
                    "writes": {
                        "native_media_bridge.feature_proj.weight": {"columns": [b, e], "mode": feature_mode, "previous_norm": previous_feature_norm, "parameters": hidden * (e - b)},
                        "native_media_bridge.type_embed.weight": {"row": type_id, "mode": "added", "parameters": hidden},
                        "native_media_head.out.weight": {"rows": [eb, ee], "mode": emission_mode, "previous_norm": previous_emission_norm, "parameters": hidden * (ee - eb)},
                        "native_media_head.type_affine.weight": {"row": type_id, "half": "shift", "mode": "added (fp64 lstsq of the decoder bias through the head rows)", "relative_residual": residual, "parameters": hidden},
                    },
                }
            )
            log(f"interface {entry['set']}/{entry['name']}: type {type_id}, feature band [{b},{e}), emission band [{eb},{ee}), shift residual {residual:.3e}")
        self.record["interface"] = {
            "entries": reports,
            "bands": band_report,
            "note": "rows written INTO the counted interface (PROBE.S0 counted them): no accounting slack",
        }

    # -- band rows -------------------------------------------------------------

    def install_band(self) -> None:
        if not self.args.c18:
            self.record["band"] = {"note": "no band table given"}
            return
        torch = self.torch
        path = _resolve(self.args.c18)
        views = read_single(path)
        try:
            band = self.cfg.ledger().range(self.args.band_name)
        except KeyError as error:
            raise Refusal(f"the child declares no ledger band named {self.args.band_name!r}") from error
        if "embed.weight" not in views:
            raise Refusal(f"{path}: band table lacks embed.weight")
        sidecar = Path(str(path) + ".json")
        sidecar_body: dict[str, Any] = {}
        if sidecar.is_file():
            try:
                sidecar_body = json.loads(sidecar.read_text(encoding="utf-8"))
            except ValueError as error:
                raise Refusal(f"{sidecar}: not JSON: {error}") from error
        sha_check = verify_declared_sha256(views, sidecar_body.get("tensor_sha256") if isinstance(sidecar_body, dict) else None, where=str(path))
        embed = views["embed.weight"].torch(dtype=torch.float32)
        if tuple(embed.shape) != (band.width, int(self.cfg.hidden_size)):
            raise Refusal(
                f"{path}: embed.weight is {tuple(embed.shape)}, the declared {band.name} band is "
                f"[{band.width}, {self.cfg.hidden_size}]"
            )
        head = views["lm_head.weight"].torch(dtype=torch.float32) if "lm_head.weight" in views else None
        if head is not None and tuple(head.shape) != tuple(embed.shape):
            raise Refusal(f"{path}: lm_head.weight is {tuple(head.shape)}, embed.weight is {tuple(embed.shape)}")
        tied = "lm_head.weight" in self.tied or bool(self.cfg.tie_word_embeddings)
        if tied and head is not None and not torch.equal(head, embed):
            raise Refusal("the child ties lm_head to embed; the band table's two rows sets must be equal")
        if not tied and head is None:
            raise Refusal("the child has an untied lm_head; the band table must carry lm_head.weight")
        targets = ["embed.weight"] + ([] if tied else ["lm_head.weight"])
        writes: dict[str, Any] = {}
        for name, table in zip(targets, [embed, head]):
            tensor = self.working(name)
            existing = tensor[band.begin : band.end]
            nonzero = int(existing.ne(0).sum())
            if nonzero and not self.args.allow_nonzero_band:
                raise Refusal(
                    f"{name}[{band.begin}:{band.end}] holds {nonzero} non-zero values; the declared band must be "
                    "zero before C18 fills it (pass --allow-nonzero-band to overwrite)"
                )
            tensor[band.begin : band.end] = table
            writes[name] = {"rows": [band.begin, band.end], "previous_nonzero_values": nonzero, "parameters": int(table.numel())}
        self.record["band"] = {
            "band": {"name": band.name, "begin": band.begin, "end": band.end, "width": band.width},
            "table": str(path),
            "table_sha256": sha256_file(path),
            "table_tensor_sha256": {name: view_hash(view) for name, view in sorted(views.items())},
            "table_tensor_sha256_check": sha_check,
            "sidecar_sha256": sha256_file(sidecar) if sidecar.is_file() else None,
            "tied": tied,
            "rows_written": band.width,
            "writes": writes,
            "note": "rows written into the DECLARED band (S0 ledger_bands change): not a component, no slack",
        }
        log(f"band {band.name}: {band.width} rows into {targets}")

    # -- charted blocks --------------------------------------------------------

    def load_blocks(self) -> list[dict[str, Any]]:
        containers = [load_container(_resolve(d), self.cfg) for d in (self.args.blocks or [])]
        names = [c["name"] for c in containers]
        if len(set(names)) != len(names):
            raise Refusal(f"container directory names must be unique: {names}")
        tap_root = _resolve(self.args.tap_rows) if self.args.tap_rows else None
        hidden = int(self.cfg.hidden_size)
        for container in containers:
            container["hidden"] = hidden
            if container["charts"] is None:
                fit_container_charts(container, tap_root, seed=int(self.args.seed), mediafold_root=_resolve(self.args.mediafold_root))
            report = verify_time_reference(container, hidden, rel_tol=float(self.args.time_reference_rel_tol))
            if report is not None:
                log(f"time reference {container['name']}: rel_l2 {report['rel_l2']:.3e} at sigma {report['sigma']} (time_embed_r={report['time_embed_r']})")
        return containers

    def cast_stats(self, name: str, tensor: Any) -> dict[str, Any]:
        if self.args.charted_dtype == "fp32":
            return {"dtype": "F32", "max_abs_narrowing": 0.0}
        narrowed = tensor.to(self.storage_dtype)
        diff = (narrowed.to(self.torch.float32) - tensor).abs().max()
        return {"dtype": self.storage_dtype_name, "max_abs_narrowing": float(diff)}

    def install_insert(self, containers: list[dict[str, Any]]) -> None:
        pair_containers(
            containers, self.cfg, pair_by=self.args.pair_by,
            allow_existing_charted=bool(getattr(self.args, "allow_existing_charted_insert", False)),
        )
        cfg = self.cfg
        trunk_layers = int(cfg.num_layers)
        kinds: list[str] = []
        widths: list[int] = []
        specs = []
        placements: list[dict[str, Any]] = []
        new_index_of: dict[int, int] = {}
        charted_state: dict[str, Any] = {}
        added = 0
        for old in range(trunk_layers):
            new_index_of[old] = len(kinds)
            kinds.append(cfg.block_kind(old))
            widths.append(int(cfg.resolved_intermediate_size(old)))
            if cfg.is_charted(old):
                specs.append(dataclasses.replace(cfg.charted_spec(old), layer_index=new_index_of[old]))
            for container in [c for c in containers if int(c["insert_after"]) == old]:
                position = len(kinds)
                spec = dataclasses.replace(container["spec"], layer_index=position)
                specs.append(spec)
                kinds.append("charted")
                widths.append(int(cfg.intermediate_size))
                prefix = f"blocks.{position}.charted."
                narrowing: dict[str, Any] = {}
                tensors = dict(container["tensors"])
                tensors.update(container["charts"])
                expected = spec.parameter_shapes("", int(cfg.hidden_size))
                if set(tensors) != set(expected):
                    raise Refusal(f"{container['name']}: tensor set {sorted(tensors)} != spec set {sorted(expected)}")
                block_params = 0
                for key, tensor in tensors.items():
                    preserve_h3_f32 = spec.modulation == "h3_curve6" and key in H3_FP32_TENSORS
                    stats = ({"dtype": "F32", "max_abs_narrowing": 0.0, "reason": "source H3 time/RoPE coefficients retained exactly"}
                             if preserve_h3_f32 else self.cast_stats(key, tensor))
                    narrowing[key] = stats
                    charted_state[prefix + key] = tensor if preserve_h3_f32 or self.args.charted_dtype == "fp32" else tensor.to(self.storage_dtype)
                    block_params += int(tensor.numel())
                added += block_params
                mlp_prefix = prefix + "mlp."
                placements.append(
                    {
                        "container": container["name"],
                        "source": spec.source,
                        "component": spec.component,
                        "layer_index": position,
                        "tensor_prefix": prefix,
                        "mlp_tensor_prefix": mlp_prefix,
                        "pairing": container["pairing"],
                        "type_ids": list(spec.type_ids),
                        "rung": spec.rung,
                        "spec": spec.to_manifest(),
                        "parameters": block_params,
                        "chart_parameters": int(sum(int(t.numel()) for k, t in tensors.items() if k.startswith("chart_"))),
                        "mlp_parameters": spec.mlp_parameter_count(),
                        "chart_source": container["chart_source"],
                        "chart_fit": container.get("chart_fit"),
                        "dropped_chart_keys": container["dropped_chart_keys"],
                        "time_reference": container.get("time_reference_check"),
                        "tensor_sha256_check": container["tensor_sha256_check"],
                        "narrowing": narrowing,
                        "files": container["files"],
                        "provenance": container["provenance"],
                        "holdout": container["holdout"],
                        "accounting": container["accounting"],
                        "h3_source_contract": container.get("h3_source_contract"),
                        "runtime_requirements": container.get("runtime_requirements", {}),
                        "capture_deviations": container.get("capture_deviations", []),
                    }
                )
                log(f"charted {container['name']} ({spec.source}) -> blocks.{position} after trunk layer {old}; {block_params:,} params")
        renamed: dict[str, Any] = {}
        for name, value in self.state.items():
            if name.startswith("blocks."):
                head, index, rest = name.split(".", 2)
                renamed[f"{head}.{new_index_of[int(index)]}.{rest}"] = value
            else:
                renamed[name] = value
        # working copies follow the rename too
        work: dict[str, Any] = {}
        for name, value in self.work.items():
            if name.startswith("blocks."):
                head, index, rest = name.split(".", 2)
                work[f"{head}.{new_index_of[int(index)]}.{rest}"] = value
            else:
                work[name] = value
        self.work = work
        renamed.update(charted_state)
        self.state = renamed
        new_widths = tuple(widths) if cfg.intermediate_sizes else ()
        self.new_cfg = dataclasses.replace(
            cfg,
            num_layers=len(kinds),
            block_pattern=tuple(kinds),
            intermediate_sizes=new_widths,
            charted_blocks=tuple(specs),
            metadata={**cfg.metadata, "composite_arm": "insert", "composite_tool": f"{TOOL_NAME} {TOOL_VERSION}"},
        )
        self.new_cfg.validate()
        self.record["blocks"] = {"arm": "insert", "placements": placements, "added_parameters": added, "trunk_layer_map": {str(k): v for k, v in new_index_of.items()}}
        self.added_parameters = added

    def install_blend(self, containers: list[dict[str, Any]]) -> None:
        pair_containers(containers, self.cfg, pair_by=self.args.pair_by)
        torch = self.torch
        cfg = self.cfg
        hidden = int(cfg.hidden_size)
        if str(cfg.activation_kind) != "silu":
            raise Refusal(
                f"the blend arm pushes SwiGLU rows into a plain SiLU-GLU trunk FFN; this child's activation_kind is "
                f"{cfg.activation_kind!r} (SiTU caps gate and up), so the push would not be the block's function"
            )
        widths = [int(cfg.resolved_intermediate_size(i)) for i in range(int(cfg.num_layers))]
        placements: list[dict[str, Any]] = []
        added = 0
        for container in containers:
            spec = container["spec"]
            if spec.modulation == "h3_curve6":
                raise Refusal(f"{container['name']}: H3 curve modulation, joint context and source RoPE cannot be represented by the blend arm; use insert")
            if spec.mlp_kind != "swiglu":
                raise Refusal(f"{container['name']}: the blend arm folds SwiGLU rows only; {spec.mlp_kind!r} cannot enter a silu-glu FFN")
            layer = int(container["insert_after"])
            tensors = container["tensors"]
            charts = container["charts"]
            block = build_source_block(spec, hidden, tensors, charts)
            with torch.no_grad():
                if spec.modulation == "adaln6":
                    modulation = block.time_path(torch.zeros(1)).view(6, int(spec.width)) + block.scale_shift_table
                    _, _, _, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(0)
                else:
                    shift_mlp = torch.zeros(int(spec.width))
                    scale_mlp = torch.zeros(int(spec.width))
                    gate_mlp = torch.ones(int(spec.width))
                B_in = charts["chart_in.weight"].to(torch.float64)
                B_out = charts["chart_out.weight"].to(torch.float64)
                gate = tensors["mlp.gate_proj.weight"].to(torch.float64) * (1.0 + scale_mlp.to(torch.float64))[None, :]
                up = tensors["mlp.up_proj.weight"].to(torch.float64) * (1.0 + scale_mlp.to(torch.float64))[None, :]
                down = tensors["mlp.down_proj.weight"].to(torch.float64) * gate_mlp.to(torch.float64)[:, None]
                gate_rows = (gate @ B_in).to(torch.float32)
                up_rows = (up @ B_in).to(torch.float32)
                down_cols = (B_out @ down).to(torch.float32)
            keep = int(spec.mlp_width)
            up_name = f"blocks.{layer}.ffn.up.weight"
            down_name = f"blocks.{layer}.ffn.down.weight"
            current_up = self.working(up_name)
            current_down = self.working(down_name)
            width = widths[layer]
            if tuple(current_up.shape) != (2 * width, hidden) or tuple(current_down.shape) != (hidden, width):
                raise Refusal(f"layer {layer} FFN shapes {tuple(current_up.shape)} / {tuple(current_down.shape)} do not match width {width}")
            self.work[up_name] = torch.cat((current_up[:width], gate_rows, current_up[width:], up_rows), dim=0)
            self.work[down_name] = torch.cat((current_down, down_cols), dim=1)
            widths[layer] = width + keep
            rows_params = 3 * keep * hidden
            added += rows_params
            placements.append(
                {
                    "container": container["name"],
                    "source": spec.source,
                    "component": spec.component,
                    "layer_index": layer,
                    "pairing": container["pairing"],
                    "appended_channels": keep,
                    "parameters": rows_params,
                    "mlp_parameters": rows_params,
                    "approximations": [
                        "the trunk's norm2 replaces the block's mlp_norm ahead of the chart",
                        "adaLN evaluated at sigma 0 through the reference-verified time path (shift_mlp dropped: the trunk FFN has no bias; scale_mlp and gate_mlp folded into the rows)",
                        "chart_in bias dropped",
                        "the block's attention half is not blended (MLP rows only)",
                    ],
                    "shift_mlp_norm_dropped": float(shift_mlp.norm()),
                    "time_reference": container.get("time_reference_check"),
                    "tensor_sha256_check": container["tensor_sha256_check"],
                    "files": container["files"],
                    "provenance": container["provenance"],
                    "holdout": container["holdout"],
                    "accounting": container["accounting"],
                    "chart_source": container["chart_source"],
                    "chart_fit": container.get("chart_fit"),
                }
            )
            log(f"blend {container['name']} ({spec.source}) -> {keep} channels appended to blocks.{layer}.ffn")
        padding = 0
        uniform = len(set(widths)) == 1
        padding_per_layer: dict[str, int] = {}
        if not uniform:
            if self.args.blend_uniform == "pad":
                target = max(widths)
                for layer, width in enumerate(widths):
                    if width == target:
                        continue
                    up_name = f"blocks.{layer}.ffn.up.weight"
                    down_name = f"blocks.{layer}.ffn.down.weight"
                    current_up = self.working(up_name)
                    current_down = self.working(down_name)
                    extra = target - width
                    zeros_up = torch.zeros(extra, hidden, dtype=torch.float32)
                    self.work[up_name] = torch.cat((current_up[:width], zeros_up, current_up[width:], zeros_up), dim=0)
                    self.work[down_name] = torch.cat((current_down, torch.zeros(hidden, extra, dtype=torch.float32)), dim=1)
                    padding_per_layer[str(layer)] = extra
                    padding += 3 * extra * hidden
                    widths[layer] = target
                added += padding
                uniform = True
            elif not self.args.allow_per_layer_widths:
                raise Refusal(
                    f"the blend arm leaves per-layer FFN widths {sorted(set(widths))}; the qwen35 exporter writes one "
                    "feed_forward_length. Pass --blend-uniform pad (zero-pad every layer, priced in the manifest) or "
                    "--allow-per-layer-widths (not exportable)"
                )
        self.new_cfg = dataclasses.replace(
            cfg,
            intermediate_size=int(widths[0]) if uniform else int(cfg.intermediate_size),
            intermediate_sizes=() if uniform else tuple(widths),
            metadata={**cfg.metadata, "composite_arm": "blend", "composite_tool": f"{TOOL_NAME} {TOOL_VERSION}"},
        )
        self.new_cfg.validate()
        self.record["blocks"] = {
            "arm": "blend",
            "activation_kind": str(cfg.activation_kind),
            "placements": placements,
            "added_parameters": added,
            "padding_parameters": padding,
            "padding_channels_per_layer": padding_per_layer,
            "uniform_width": uniform,
            "ffn_widths": widths,
            "note": (
                "DIAGNOSTIC arithmetic push of the coreset MLP rows through the charts; NOT the c10_union_install "
                "--extra-blocks union refit, and at trunk width the appended rows cannot meet the >= 4x component rule "
                "for a source narrower than the trunk -- those components register as --passed-through"
            ),
        }
        self.added_parameters = added

    # -- write -----------------------------------------------------------------

    def finalize_state(self) -> dict[str, Any]:
        narrowing: dict[str, float] = {}
        for name, tensor in self.work.items():
            cast = tensor.to(self.storage_dtype)
            narrowing[name] = float((cast.to(self.torch.float32) - tensor).abs().max())
            self.state[name] = cast
        for alias, owner in self.tied.items():
            if owner not in self.state:
                raise Refusal(f"tied owner {owner!r} is missing from the state")
            self.state[alias] = self.state[owner]
        self.record["narrowing"] = {"storage_dtype": self.storage_dtype_name, "max_abs_per_modified_tensor": narrowing}
        return self.state

    def write(self) -> None:
        from chimera_omni.export.gguf_names import chimera_parameter_names

        new_cfg = getattr(self, "new_cfg", None) or dataclasses.replace(
            self.cfg, metadata={**self.cfg.metadata, "composite_arm": "none", "composite_tool": f"{TOOL_NAME} {TOOL_VERSION}"}
        )
        self.new_cfg = new_cfg
        state = self.finalize_state()
        expected_names = set(chimera_parameter_names(new_cfg))
        written_names = set(state)
        if expected_names != written_names:
            raise Refusal(
                f"state names do not match the new config's parameter enumeration; missing "
                f"{sorted(expected_names - written_names)[:5]}, unexpected {sorted(written_names - expected_names)[:5]}"
            )
        estimated = state_nbytes(state, self.tied)
        probe, free_before = free_bytes_under(self.out)
        minimum = int(float(self.args.min_free_gib) * (1 << 30))
        if free_before - estimated < minimum:
            raise Refusal(
                f"writing ~{estimated / (1 << 30):.1f} GiB under {self.out} would leave {(free_before - estimated) / (1 << 30):.1f} GiB "
                f"free on the filesystem of {probe} (now {free_before / (1 << 30):.1f} GiB), below --min-free-gib {self.args.min_free_gib}; "
                "large artifacts belong under /mnt/lexar/omni-campaign/<lane>/"
            )
        self.record["disk"] = {
            "filesystem_probe": str(probe),
            "free_bytes_before": free_before,
            "estimated_output_bytes": estimated,
            "min_free_gib": float(self.args.min_free_gib),
            "materialised_bytes": state_nbytes({k: v for k, v in state.items() if not isinstance(v, LazyView)}, self.tied),
            "streamed_bytes": state_nbytes({k: v for k, v in state.items() if isinstance(v, LazyView)}, self.tied),
        }
        self.out.mkdir(parents=True, exist_ok=True)
        if self.args.overwrite:
            removed = []
            for path in list(self.out.glob("model-*.safetensors")) + [
                self.out / INDEX_FILENAME, self.out / CONFIG_FILENAME, self.out / MANIFEST_FILENAME,
                self.out / (MANIFEST_FILENAME + ".sha256"), self.out / REFUSED_FILENAME,
            ]:
                if path.is_file():
                    path.unlink()
                    removed.append(path.name)
            self.record["overwrite_removed"] = removed
        log(f"writing {self.out} (streamed {self.record['disk']['streamed_bytes'] / (1 << 30):.2f} GiB, materialised {self.record['disk']['materialised_bytes'] / (1 << 30):.2f} GiB)")
        index = write_checkpoint_streamed(
            state,
            self.out,
            tied=self.tied,
            config=new_cfg.to_manifest(),
            ledger=new_cfg.ledger().to_manifest(),
            source={
                "tool": f"{TOOL_NAME} {TOOL_VERSION}",
                "child": str(self.child),
                "child_index_sha256": self.record["child"]["index_sha256"],
                "arm": self.record.get("blocks", {}).get("arm", "none"),
                "composite_manifest": MANIFEST_FILENAME,
            },
            chimera_commit=self.record["tree"]["commit"],
            shard_bytes=int(self.args.shard_bytes),
            metadata={"composite_tool": f"{TOOL_NAME} {TOOL_VERSION}"},
        )
        self.output_index = index

    def verify_and_manifest(self) -> None:
        counts = header_counts(self.out)
        output_total = sum(counts.values())
        expected_total = self.input_total + int(getattr(self, "added_parameters", 0))
        index_shards = sorted(set(self.output_index["weight_map"].values()))
        disk_shards = sorted(p.name for p in self.out.glob("*.safetensors"))
        if index_shards != disk_shards:
            raise Refusal(f"written shards {disk_shards} != index shards {index_shards}")
        if output_total != expected_total:
            raise Refusal(
                f"output header count {output_total:,} != child {self.input_total:,} + additions "
                f"{getattr(self, 'added_parameters', 0):,} = {expected_total:,}"
            )
        oc_seal_total = None
        try:
            ledger_root = _resolve(self.args.ledger_root)
            if str(ledger_root) not in sys.path:
                sys.path.insert(0, str(ledger_root))
            from oc_seal import safetensors_tensor_counts  # type: ignore

            oc_counts = safetensors_tensor_counts(self.out)
            oc_seal_total = sum(oc_counts.values())
            if oc_seal_total != output_total:
                raise Refusal(f"oc_seal counts {oc_seal_total:,} != header walk {output_total:,}")
        except ImportError:
            oc_seal_total = None
        components = self.components(counts)
        register = self.register_line(components, output_total)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(Path(__file__).resolve()), "argv": self.argv, "recipe_command": self.recipe_command()},
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tree": self.record["tree"],
            "child": self.record["child"],
            "output": {
                "directory": str(self.out),
                "checkpoint_id": self.args.checkpoint_id,
                "shards": disk_shards,
                "shard_sha256": {name: sha256_file(self.out / name) for name in disk_shards},
                "index_sha256": sha256_file(self.out / INDEX_FILENAME),
                "config_sha256": sha256_file(self.out / CONFIG_FILENAME),
                "tensors": len(counts),
                "config": self.new_cfg.to_manifest(),
            },
            "accounting": {
                "child_header_parameters": self.input_total,
                "charted_or_blend_added": int(getattr(self, "added_parameters", 0)),
                "output_header_parameters": output_total,
                "oc_seal_header_parameters": oc_seal_total,
                "equality_asserted": True,
                "quantization": "none",
                "storage_dtype": self.storage_dtype_name,
                "charted_dtype": ("F32" if self.args.charted_dtype == "fp32" else
                                  "mixed" if self.storage_dtype_name != "F32" and any(c.get("h3_source_contract") for c in self.record.get("blocks", {}).get("placements", []))
                                  else self.storage_dtype_name),
                "charted_dtype_policy": {"default": "F32" if self.args.charted_dtype == "fp32" else self.storage_dtype_name,
                                         "h3_preserved_f32_tensors": sorted(H3_FP32_TENSORS)},
            },
            "sources": self.sources(components),
            "components": components,
            "interface": self.record.get("interface"),
            "band": self.record.get("band"),
            "blocks": self.record.get("blocks", {"arm": "none", "placements": []}),
            "narrowing": self.record.get("narrowing"),
            "disk": self.record.get("disk"),
            "register": register,
            "runtime": {"peak_rss_bytes": peak_rss_bytes(), "python": sys.version.split()[0],
                        "peak_rss_source": "GetProcessMemoryInfo.PeakWorkingSetSize" if sys.platform == "win32" else "resource.ru_maxrss"},
        }
        digest = write_json_with_sidecar(self.out / MANIFEST_FILENAME, manifest)
        self.manifest_sha256 = digest
        log(f"manifest {self.out / MANIFEST_FILENAME} sha256 {digest}")
        log(f"output: {len(counts)} tensors, {output_total:,} header parameters (= child + {getattr(self, 'added_parameters', 0):,}); register reachable: {register['reachable']}")

    def components(self, counts: dict[str, int]) -> list[dict[str, Any]]:
        blocks = self.record.get("blocks", {})
        out: list[dict[str, Any]] = []
        for placement in blocks.get("placements", []):
            parent = int(placement.get("accounting", {}).get("parent_mlp_parameters") or 0)
            if blocks.get("arm") == "insert":
                prefix = placement["mlp_tensor_prefix"]
                child = sum(v for k, v in counts.items() if k.startswith(prefix))
                if child != int(placement["mlp_parameters"]):
                    raise Refusal(f"{prefix}: headers say {child}, the spec says {placement['mlp_parameters']}")
                counts_source = "child derived from the written headers under the prefix; parent from the container's source-block accounting"
            else:
                prefix = None
                child = int(placement["mlp_parameters"])
                counts_source = "child = the appended trunk-width rows (3 x keep x hidden); parent from the container's source-block accounting"
            ratio = (child / parent) if parent else None
            meets = ratio is not None and ratio <= 0.25
            whole_parent = placement.get("accounting", {}).get("parent_block_parameters")
            whole_child = placement.get("accounting", {}).get("child_block_parameters")
            out.append(
                {
                    "component": placement["component"],
                    "source": placement["source"],
                    "parent_params": parent,
                    "child_params": child,
                    "ratio": ratio,
                    "meets_4x": meets,
                    "register_form": "--component" if meets else "--passed-through",
                    "tensor_prefix": prefix,
                    "counts_source": counts_source,
                    "register_spec": f"{placement['component']}:{parent}:{child}",
                    "whole_block": {
                        "parent_params": whole_parent,
                        "child_params": whole_child,
                        "ratio": (whole_child / whole_parent) if whole_parent and whole_child is not None else None,
                        "retained_shared_coefficients": placement.get("accounting", {}).get("retained_shared_coefficients", 0),
                        "child_with_retained_coefficients_params": (whole_child + placement.get("accounting", {}).get("retained_shared_coefficients", 0)) if whole_child is not None else None,
                        "installed_with_charts_parameters": placement["parameters"],
                    },
                    "note": (
                        "the ledger's 4-field derived form needs the PREFIX in the parent's headers too; the parent block "
                        "lives in the source model, so the register line uses the 3-field asserted form and this manifest "
                        "carries the derivation"
                        + ("" if meets else "; ratio > 0.25 so oc_checkpoint would refuse it as --component: it is carried as --passed-through")
                    ),
                }
            )
        return out

    def sources(self, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}

        def entry(source: str) -> dict[str, Any]:
            return table.setdefault(source, {"source": source, "mechanisms": []})

        for item in self.record.get("interface", {}).get("entries", []) or []:
            entry(item["source"])["mechanisms"].append(
                {
                    "mechanism": "C15/C11 interface rows into the counted native interface" if item["set"] == "c15" else "C11 patch-embed/merger rows into the image interface",
                    "rung": item["rung"],
                    "type_id": item["type_id"],
                    "parameters": sum(w["parameters"] for w in item["writes"].values()),
                    "sha256": item["rows_sha256"],
                }
            )
        band = self.record.get("band")
        if band and band.get("table_sha256"):
            entry("ACE-LM")["mechanisms"].append(
                {"mechanism": "C18 embedding bridge into the declared music_codes band", "rung": "linear", "parameters": sum(w["parameters"] for w in band["writes"].values()), "sha256": band["table_sha256"]}
            )
        for placement in self.record.get("blocks", {}).get("placements", []):
            entry(placement["source"])["mechanisms"].append(
                {
                    "mechanism": f"C16 4x block, {self.record['blocks']['arm']} arm",
                    "rung": placement.get("rung"),
                    "parameters": placement["parameters"],
                    "sha256": placement["files"].get("charted_block.safetensors"),
                    "component": placement["component"],
                }
            )
        return [table[key] for key in sorted(table)]

    def accounting_parent(self) -> dict[str, Any]:
        """The parent count the fold accounting is priced against, and where it was read from."""

        directory = _resolve(self.args.accounting_parent_directory) if self.args.accounting_parent_directory else None
        typed = self.args.parent_parameter_count
        if directory is not None:
            if not directory.is_dir():
                raise Refusal(f"--accounting-parent-directory {directory} is not a directory")
            try:
                ledger_root = _resolve(self.args.ledger_root)
                if str(ledger_root) not in sys.path:
                    sys.path.insert(0, str(ledger_root))
                from oc_seal import safetensors_parameter_count  # type: ignore

                count, _per_file = safetensors_parameter_count(directory)
                source = "oc_seal.safetensors_parameter_count over the directory's headers (the ledger's own reader)"
            except ImportError:
                count = sum(header_counts(directory).values())
                source = "header walk over the directory (oc_seal not importable)"
            if typed is not None and int(typed) != int(count):
                raise Refusal(f"--parent-parameter-count {typed:,} disagrees with the headers under {directory} ({count:,}); the bytes decide")
            return {"directory": str(directory), "parameter_count": int(count), "source": source, "register_args": ["--parent-directory", str(directory)]}
        if typed is not None:
            return {
                "directory": None,
                "parameter_count": int(typed),
                "source": "asserted (--parent-parameter-count); the ledger records it as asserted-by-operator",
                "register_args": ["--parent-parameter-count", str(int(typed)), "--allow-asserted-parameter-counts"],
            }
        return {"directory": None, "parameter_count": None, "source": None, "register_args": []}

    def directory_registerable(self) -> dict[str, Any]:
        """Would `oc_checkpoint register --directory <out>` pass the ledger's protected-prefix guard?"""

        try:
            ledger_root = _resolve(self.args.ledger_root)
            if str(ledger_root) not in sys.path:
                sys.path.insert(0, str(ledger_root))
            import oc_checkpoint  # type: ignore

            prefixes = tuple(oc_checkpoint.PROTECTED_PREFIXES)
        except (ImportError, AttributeError):
            return {"known": False, "note": "oc_checkpoint not importable; the ledger's protected prefixes could not be read"}
        parts = self.out.as_posix().split("/")
        hit = None
        for prefix in prefixes:
            prefix_parts = prefix.split("/")
            if parts[: len(prefix_parts)] == prefix_parts:
                hit = prefix
                break
        return {
            "known": True,
            "registerable": hit is None,
            "protected_prefixes": list(prefixes),
            "matched_prefix": hit,
            "note": (
                None if hit is None else
                f"{self.out} is under the ledger's protected prefix {hit!r} (its guard names the CIFS share; /mnt/lexar is a "
                "local SSD): register from a byte copy under $OC/oc02/children or narrow oc_checkpoint.PROTECTED_PREFIXES "
                "to /mnt/ai_data first (S7 row decision)"
            ),
        }

    def recipe_command(self) -> str:
        """The command that rebuilds this output: the ledger's R4 recipe (a checkpoint without one cannot be dropped)."""

        return " ".join([f"{TOOL_NAME}.py", *(shlex.quote(str(a)) for a in self.argv)])

    def register_line(self, components: list[dict[str, Any]], output_total: int) -> dict[str, Any]:
        parent = self.parent_accounting
        parent_count = parent["parameter_count"]
        folded = [c for c in components if c["meets_4x"]]
        passed = [c for c in components if not c["meets_4x"]]
        claimed_saving = sum(int(c["parent_params"]) - int(c["child_params"]) for c in folded)
        claimed_parent = sum(int(c["parent_params"]) for c in folded)
        slack = None
        measured_drop = None
        reachable = False
        notes = [
            "--parent-checkpoint-id is the LINEAGE parent (the S5 child); the fold accounting is priced against the "
            "accounting parent (PARENT-Q38-27BU's directory, derived from its headers by the ledger) because an "
            "insert-arm child GROWS by the charted state relative to the S5 child and oc_checkpoint._fold_accounting "
            "refuses measured_drop <= 0. Every growth stage of the chain (S3, S5, S7) is priced the same way.",
        ]
        if parent_count is not None:
            measured_drop = int(parent_count) - int(output_total)
            slack = max(0, claimed_saving - measured_drop)
            reachable = measured_drop > 0 and claimed_parent <= int(parent_count)
            notes.append(
                f"measured_drop against the accounting parent {int(parent_count):,} ({parent['source']}) is {measured_drop:,}; "
                f"claimed saving {claimed_saving:,} over {len(folded)} --component entries; slack required {slack:,}; "
                f"reachable={reachable}"
            )
        else:
            notes.append(
                "no accounting parent given (--accounting-parent-directory / --parent-parameter-count): the emitted line "
                "would be refused by the ledger (measured_drop <= 0 against the S5 child); reachable=False"
            )
        if passed:
            notes.append(
                "components with ratio > 0.25 are carried as --passed-through: " + ", ".join(c["component"] for c in passed)
            )
        if not folded:
            notes.append("no --component entry meets >= 4x: stage fold needs at least one, so the line is not reachable")
            reachable = False
        args = [
            "$L register",
            f"--checkpoint-id {self.args.checkpoint_id}",
            "--stage fold",
            f"--parent-checkpoint-id {self.args.parent_checkpoint_id}",
            f"--directory {self.out}",
        ]
        args.extend(parent["register_args"])
        args.extend(f"--component {c['register_spec']}" for c in folded)
        args.extend(f"--passed-through {c['component']}" for c in passed)
        if slack:
            args.append(f"--accounting-slack {slack}")
        # R4: the record carries how the checkpoint was built -- the manifest is the recipe
        # file (hashed into the record) and this tool's own command line the recipe command
        args.append(f"--recipe {self.out / MANIFEST_FILENAME}")
        args.append(f"--recipe-command {shlex.quote(self.recipe_command())}")
        args.append("--confirm")
        return {
            "command": " ".join(args),
            "reachable": bool(reachable),
            "claimed_saving": claimed_saving,
            "claimed_parent": claimed_parent,
            "measured_drop": measured_drop,
            "accounting_slack_required": slack,
            "accounting_parent": {k: v for k, v in parent.items() if k != "register_args"},
            "lineage_parent_checkpoint_id": self.args.parent_checkpoint_id,
            "passed_through": [c["component"] for c in passed],
            "directory": self.directory_registerable(),
            "notes": notes,
        }

    # -- driver ----------------------------------------------------------------

    def run(self) -> int:
        self.record["tree"] = tree_identity(_resolve(self.args.chimera_src))
        # resolved first: a typed count that disagrees with the directory refuses before any write
        self.parent_accounting = self.accounting_parent()
        self.load_child()
        self.install_interface()
        self.install_band()
        containers = self.load_blocks()
        if containers:
            if self.args.arm == "insert":
                self.install_insert(containers)
            else:
                self.install_blend(containers)
        else:
            self.record["blocks"] = {"arm": "none", "placements": [], "added_parameters": 0, "note": "no charted blocks given"}
            self.added_parameters = 0
        self.write()
        try:
            self.verify_and_manifest()
        except Refusal as error:
            self.refuse_after_write(str(error))
            raise
        return 0

    def refuse_after_write(self, reason: str) -> None:
        removed = []
        for path in list(self.out.glob("model-*.safetensors")) + [self.out / INDEX_FILENAME, self.out / CONFIG_FILENAME]:
            if path.is_file():
                path.unlink()
                removed.append(path.name)
        (self.out / REFUSED_FILENAME).write_text(
            json.dumps({"tool": TOOL_NAME, "reason": reason, "removed": removed, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n",
            encoding="utf-8",
        )


# --------------------------------------------------------------------------
# ACE c16_transplant_run output -> charted container
# --------------------------------------------------------------------------


def write_time_reference(out: Path, sigma: Any, modulation: Any) -> dict[str, Any]:
    from safetensors.torch import save_file

    torch = _torch()
    path = out / TIME_REFERENCE_FILENAME
    save_file({"sigma": sigma.to(torch.float32).reshape(-1).contiguous(), "modulation": modulation.to(torch.float32).contiguous()}, str(path))
    return {name: view_hash(view) for name, view in sorted(read_single(path).items())}


def convert_ace_run(
    run_dir: Path,
    ace_weights: Path,
    out: Path,
    *,
    chart: str = "ridge",
    type_ids: tuple[int, ...] = (4,),
    sliding_window: int = 128,
    source_depth: int = 24,
    rel_tol: float = DEFAULT_TIME_REFERENCE_REL_TOL,
) -> dict[str, Any]:
    """`c16_transplant_run.py` output (folded block + chart + receipt + capture) -> a charted container.

    Drops `cross_attn.*` (the trunk is the context), takes BOTH of the source's timestep
    paths from the ACE weights (`decoder.time_embed.*` and `decoder.time_embed_r.*`: the
    parent builds temb as `time_embed(t) + time_embed_r(t - t_r)` and the capture used
    `t_r = 0`), keeps every other tensor verbatim, copies the run's coreset accounting /
    holdout numbers into the container JSON, and writes `time_reference.safetensors` from
    the capture's own `temb` (`capture_L*.safetensors`) at the stimulus sigma. The
    conversion then rebuilds the charted block and refuses unless its time path
    reproduces that reference within `rel_tol` -- the same gate the installer applies.
    """

    torch = _torch()
    from safetensors import safe_open
    from safetensors.torch import save_file

    receipt_path = run_dir / "c16_receipt.json"
    if not receipt_path.is_file():
        raise Refusal(f"{run_dir} has no c16_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    folded = sorted(run_dir.glob("folded_block_L*_keep*.safetensors"))
    if len(folded) != 1:
        raise Refusal(f"{run_dir}: expected exactly one folded_block_L*_keep*.safetensors, found {len(folded)}")
    captures = sorted(run_dir.glob("capture_L*.safetensors"))
    if len(captures) != 1:
        raise Refusal(f"{run_dir}: expected exactly one capture_L*.safetensors (the captured temb), found {len(captures)}")
    chart_path = run_dir / f"chart_{chart}.safetensors"
    if not chart_path.is_file():
        raise Refusal(f"{run_dir}: {chart_path.name} is missing")
    block_state = read_single(folded[0])
    chart_state = read_single(chart_path)
    shell = receipt.get("capture", {}).get("shell", {}).get("config", {})
    stimulus = receipt.get("capture", {}).get("stimulus", {}) or {}
    block_meta = receipt.get("block", {})
    keep = int(receipt.get("coreset", {}).get("keep") or block_state["mlp.gate_proj.weight"].shape[0])
    width = int(shell.get("hidden_size") or block_state["mlp.gate_proj.weight"].shape[1])
    heads = int(shell.get("num_attention_heads") or 16)
    kv_heads = int(shell.get("num_key_value_heads") or 8)
    head_dim = int(shell.get("head_dim") or 128)
    layer_type = str(block_meta.get("layer_type", "full_attention"))
    prefix = str(block_meta.get("prefix", "decoder.layers.?."))
    try:
        block_index = int(prefix.rstrip(".").split(".")[-1])
    except ValueError:
        block_index = -1
    depth = int(receipt.get("capture", {}).get("shell", {}).get("depth") or 0)
    # the ACE turbo config: sliding_window 128, 24 layers (ace_dit.AceStepDiTConfig); the
    # run receipt's shell config does not carry the window, so it is an argument here
    sliding_window = int(shell.get("sliding_window") or sliding_window)
    source_depth = int(shell.get("num_hidden_layers") or source_depth)
    if "sigma" not in stimulus:
        raise Refusal(f"{receipt_path}: capture.stimulus carries no sigma; the time reference needs the capture's noise level")
    sigma_value = float(stimulus["sigma"])
    tensors: dict[str, Any] = {}
    rename = {
        "self_attn_norm.weight": "attn_norm.weight",
        "mlp_norm.weight": "mlp_norm.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "self_attn.q_norm.weight": "attn.q_norm.weight",
        "self_attn.k_norm.weight": "attn.k_norm.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
    }
    dropped = []
    for name, view in block_state.items():
        if name.startswith("cross_attn"):
            dropped.append(name)
            continue
        if name == "scale_shift_table":
            tensors[name] = view.torch(dtype=torch.float32).reshape(6, width)
            continue
        if name not in rename:
            raise Refusal(f"{folded[0].name}: unexpected tensor {name!r}")
        tensors[rename[name]] = view.torch(dtype=torch.float32)
    with safe_open(str(ace_weights), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for path_name in ("time_embed", "time_embed_r"):
            for short in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias", "time_proj.weight", "time_proj.bias"):
                key = f"decoder.{path_name}.{short}"
                if key not in keys:
                    raise Refusal(f"{ace_weights}: {key} is missing (both ACE timestep paths are required)")
                tensors[f"{path_name}.{short}"] = handle.get_tensor(key).to(torch.float32)
    charts = {
        "chart_in.weight": chart_state["chart_in.weight"].torch(dtype=torch.float32),
        "chart_in.bias": chart_state["chart_in.bias"].torch(dtype=torch.float32),
        "chart_out.weight": chart_state["chart_out.weight"].torch(dtype=torch.float32),
    }
    capture_views = read_single(captures[0])
    if "temb" not in capture_views:
        raise Refusal(f"{captures[0]}: no temb tensor; the capture must carry the parent's modulation")
    temb = capture_views["temb"].torch(dtype=torch.float32)
    if temb.ndim != 3 or tuple(temb.shape[1:]) != (6, width):
        raise Refusal(f"{captures[0]}: temb is {tuple(temb.shape)}, expected [rows, 6, {width}]")
    row_spread = float((temb - temb[:1]).abs().max())
    if row_spread != 0.0:
        raise Refusal(
            f"{captures[0]}: temb differs across rows (max {row_spread:.3e}) but the receipt names one sigma "
            f"({sigma_value}); a per-row sigma table is needed to build the reference"
        )
    reference_sigma = torch.tensor([sigma_value], dtype=torch.float32)
    reference_modulation = temb[:1].clone()
    spec_fields = {
        "source": "ace",
        "component": f"ace.dit.block{block_index}.mlp",
        "type_ids": list(type_ids),
        "width": width,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "mlp_width": keep,
        "mlp_kind": "swiglu",
        "modulation": "adaln6",
        "norm_kind": "rms",
        "norm_affine": True,
        "norm_eps": 1.0e-6,
        "qk_norm": True,
        "attn_bias": False,
        "rope_theta": 1.0e6,
        "sliding_window": sliding_window if layer_type == "sliding_attention" else 0,
        "time_freq_dim": int(tensors["time_embed.linear_1.weight"].shape[1]),
        "time_hidden": int(tensors["time_embed.linear_1.weight"].shape[0]),
        "time_factor": 1000.0,
        "time_embed_r": True,
        "delta_scale": 1.0,
        "chart_in_bias": True,
        "rung": str(stimulus.get("rung", "")),
        "source_depth_fraction": (block_index / (source_depth - 1)) if block_index >= 0 else -1.0,
        "metadata": {
            "layer_type": layer_type,
            "timestep_r": "0.0 as captured; time_embed_r is evaluated at the trunk noise level (t - 0)",
            "cross_attention": "dropped",
            "converted_from": str(run_dir),
        },
    }
    from chimera_omni.charted import ChartedBlockSpec

    spec = ChartedBlockSpec(layer_index=0, **{**spec_fields, "type_ids": tuple(type_ids)})
    spec.validate()
    expected_shapes = spec.parameter_shapes("", 1)
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise Refusal(f"{name}: shape {tuple(tensor.shape)} != spec {expected_shapes[name]}")
    probe = build_source_block(spec, 1, tensors, {**charts, "chart_in.weight": torch.zeros(width, 1), "chart_out.weight": torch.zeros(1, width)})
    with torch.no_grad():
        got = probe.time_path(reference_sigma)
    rel = float((got - reference_modulation).norm() / reference_modulation.norm().clamp_min(1e-30))
    if rel > float(rel_tol):
        raise Refusal(
            f"{run_dir}: the charted time path (time_embed + time_embed_r at sigma {sigma_value}) does not reproduce the "
            f"captured temb: rel_l2 {rel:.3e} > {rel_tol:.1e}"
        )
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "charted_block.safetensors"))
    save_file({k: v.contiguous() for k, v in charts.items()}, str(out / "charts.safetensors"))
    reference_table = write_time_reference(out, reference_sigma, reference_modulation)
    coreset = receipt.get("coreset", {})
    body = {
        "schema": BLOCK_SCHEMA,
        "spec": spec_fields,
        "provenance": {
            "source_weights": block_meta.get("source"),
            "source_prefix": prefix,
            "source_tensor_sha256": block_meta.get("tensor_sha256"),
            "capture_receipt": str(receipt_path),
            "capture_receipt_sha256": sha256_file(receipt_path),
            "capture_file": captures[0].name,
            "capture_sha256": sha256_file(captures[0]),
            "folded_block_file": folded[0].name,
            "folded_block_sha256": sha256_file(folded[0]),
            "chart_file": chart_path.name,
            "chart_sha256": sha256_file(chart_path),
            "ace_weights": str(ace_weights),
            "dropped_tensors": sorted(dropped),
            "prefix_depth_captured": depth,
        },
        "accounting": {
            "parent_mlp_parameters": int(coreset.get("component", {}).get("parent_parameters") or block_meta.get("mlp_parameters") or 0),
            "child_mlp_parameters": int(coreset.get("component", {}).get("child_parameters") or 3 * keep * width),
            "parent_block_parameters": coreset.get("whole_block", {}).get("parent_parameters"),
            "child_block_parameters": coreset.get("whole_block", {}).get("child_parameters"),
        },
        "holdout": {**(coreset.get("holdout_scores") or {}), "split_by": coreset.get("split_by")},
        "time_reference": {
            "file": TIME_REFERENCE_FILENAME,
            "source": f"{captures[0].name}:temb (rows identical; row 0)",
            "sigma": [sigma_value],
            "timestep_r": 0.0,
            "conversion_rel_l2": rel,
            "chunk_norms": [float(v) for v in reference_modulation[0].norm(dim=1)],
        },
        "tensor_sha256": {
            "charted_block.safetensors": {name: view_hash(view) for name, view in sorted(read_single(out / "charted_block.safetensors").items())},
            "charts.safetensors": {name: view_hash(view) for name, view in sorted(read_single(out / "charts.safetensors").items())},
            TIME_REFERENCE_FILENAME: reference_table,
        },
        "insert_after": None,
    }
    (out / "charted_block.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", help="S5 Chimera child checkpoint directory")
    parser.add_argument("--out", help="output checkpoint directory (S7: under /mnt/lexar/omni-campaign/composite/...)")
    parser.add_argument("--c15", help="S6A interface row set directory (companions.json)")
    parser.add_argument("--c11", help="QVL32 patch-embed/merger row set directory (companions.json)")
    parser.add_argument("--c18", help="music_codes band table safetensors")
    parser.add_argument("--band-name", default="music_codes")
    parser.add_argument("--blocks", nargs="*", default=[], help="charted block container directories")
    parser.add_argument("--tap-rows", help="paired rows root for fit_charts when a container has no charts")
    parser.add_argument("--pair-by", default="depth-fraction", choices=("depth-fraction", "explicit"))
    parser.add_argument("--allow-existing-charted-insert", action="store_true",
                        help="insert into an already charted child using explicit CURRENT host trunk indices; retain prior typed blocks")
    parser.add_argument("--arm", default="insert", choices=("insert", "blend"))
    parser.add_argument("--blend-uniform", default="refuse", choices=("refuse", "pad"))
    parser.add_argument("--allow-per-layer-widths", action="store_true")
    parser.add_argument("--charted-dtype", default="match", choices=("match", "fp32"))
    parser.add_argument("--allow-nonzero-band", action="store_true")
    parser.add_argument("--allow-band-overlap", action="store_true")
    parser.add_argument("--allow-nonzero-type-affine", action="store_true", help="record (instead of refuse) non-zero type scale rows under an interface write")
    parser.add_argument("--time-reference-rel-tol", type=float, default=DEFAULT_TIME_REFERENCE_REL_TOL)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB, help="refuse when the output filesystem would drop below this")
    parser.add_argument("--shard-bytes", type=int, default=4 << 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="remove a previous output of THIS tool in --out first")
    parser.add_argument("--skip-verify", action="store_true", help="skip the child's per-tensor sha256 check on read")
    parser.add_argument("--chimera-src", default=os.environ.get("CHIMERA_OMNI_SRC", DEFAULT_CHIMERA_SRC))
    parser.add_argument("--mediafold-root", default=os.environ.get("MEDIAFOLD_ROOT", DEFAULT_MEDIAFOLD_ROOT))
    parser.add_argument("--ledger-root", default=os.environ.get("OC_LEDGER_ROOT", DEFAULT_LEDGER_ROOT))
    parser.add_argument("--checkpoint-id", default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--parent-checkpoint-id", default=DEFAULT_PARENT_ID, help="the LINEAGE parent (the S5 child)")
    parser.add_argument("--accounting-parent-directory", default=None, help="the directory whose header count the fold accounting is priced against (PARENT-Q38-27BU's)")
    parser.add_argument("--parent-parameter-count", type=int, default=None, help="typed accounting-parent count (asserted); must agree with --accounting-parent-directory when both are given")
    parser.add_argument("--convert-ace-run", help="c16_transplant_run.py output dir to convert into a charted container (then exit)")
    parser.add_argument("--ace-weights", default="/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors")
    parser.add_argument("--convert-out")
    parser.add_argument("--convert-chart", default="ridge")
    parser.add_argument("--convert-sliding-window", type=int, default=128)
    parser.add_argument("--convert-source-depth", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.convert_ace_run:
            if not args.convert_out:
                raise Refusal("--convert-ace-run needs --convert-out")
            body = convert_ace_run(
                _resolve(args.convert_ace_run), _resolve(args.ace_weights), _resolve(args.convert_out),
                chart=args.convert_chart, sliding_window=args.convert_sliding_window, source_depth=args.convert_source_depth,
                rel_tol=float(args.time_reference_rel_tol),
            )
            print(json.dumps({"converted": str(_resolve(args.convert_out)), "component": body["spec"]["component"], "mlp_width": body["spec"]["mlp_width"], "time_reference_rel_l2": body["time_reference"]["conversion_rel_l2"]}))
            return 0
        if not args.child or not args.out:
            raise Refusal("--child and --out are required")
        if int(args.shard_bytes) <= 0:
            raise Refusal("--shard-bytes must be positive")
        if float(args.min_free_gib) < 0:
            raise Refusal("--min-free-gib must be non-negative")
        installer = Installer(args, argv=list(sys.argv[1:] if argv is None else argv))
        installer.run()
        print(json.dumps({"out": str(installer.out), "manifest_sha256": installer.manifest_sha256, "added_parameters": installer.added_parameters}))
        return 0
    except Refusal as error:
        log(f"REFUSED: {error}")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
