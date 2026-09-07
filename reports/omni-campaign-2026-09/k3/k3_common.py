#!/usr/bin/env python3
"""Shared plumbing for the K3 streamed harness (lane k3, Chimera Omni campaign).

Everything in here is family-neutral with respect to *our* code: it reads the
Kimi-K3 repo's own artefacts (`model.safetensors.index.json`, the 96 shards,
`config.json`) and reproduces the arithmetic of the repo's own modeling file
(`modeling_kimi_linear.py`), which is the only definition of "correct" this
lane has.  Nothing here imports transformers, so it runs on a CPU-only box.

Three facts drive the design and are worth stating once:

* The share is a CIFS mount that reads at 5.3 MB/s under contention (measured:
  layer 1 = 16.99 GB in 3207 s, `k3_layer1_probe_cpu.json`).  Every read is
  therefore sequential and offset-sorted, one shard at a time, with a 64 MiB
  buffer.  Nothing in this lane ever walks the share.
* One MoE layer is 29.6 G expert parameters = 59.2 GB at bf16 but only 14.8 GB
  packed.  Experts stay PACKED in RAM/VRAM and are dequantized per touched
  expert into a 22 MB scratch.  `dequant_mxfp4` is the whole cost model.
* K3's residual stream is a PAIR -- `prefix_sum` plus up to eight block
  snapshots -- because `attn_res_block_size = 12` turns the residual into a
  softmax read over snapshots (`_apply_attn_res`).  Carrying only the hidden
  state across a layer boundary is wrong and silently so.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import time
from typing import Any, Callable, Iterable

import torch

# --------------------------------------------------------------------------
# safetensors: header parse + offset-sorted sequential reads
# --------------------------------------------------------------------------
DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


def log(msg: str, tag: str = "k3") -> None:
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_header(path: str) -> tuple[dict, int]:
    """Return (header dict, byte offset where the tensor data starts)."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def read_metadata(path: str) -> dict[str, str]:
    """The `__metadata__` block a safetensors file was stamped with, or {}.

    Header-only: it never touches the tensor data.  Used by
    `k3_review_stream.py` to ask a baseline logits file what produced it --
    an unstamped file is an old one (or someone else's) and is reported as
    unstamped rather than assumed innocent.
    """
    header, _ = read_header(path)
    meta = header.get("__metadata__") or {}
    return {str(k): str(v) for k, v in meta.items()}


def read_tensors(
    path: str,
    wanted: Iterable[str],
    buffer_bytes: int = 64 << 20,
    progress: Callable[[int], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Read only the named tensors from one shard, in on-disk offset order.

    Sorting by offset is what turns 5,404 scattered reads into one sequential
    stream; on the CIFS share the unsorted version is several times slower.
    """
    header, base = read_header(path)
    wanted = list(wanted)
    items = sorted(
        ((k, header[k]) for k in wanted if k in header and k != "__metadata__"),
        key=lambda kv: kv[1]["data_offsets"][0],
    )
    out: dict[str, torch.Tensor] = {}
    read_bytes = 0
    with open(path, "rb", buffering=buffer_bytes) as handle:
        for key, meta in items:
            start, stop = meta["data_offsets"]
            handle.seek(base + start)
            buf = handle.read(stop - start)
            tensor = torch.frombuffer(bytearray(buf), dtype=DTYPES[meta["dtype"]])
            out[key] = tensor.reshape(meta["shape"])
            read_bytes += stop - start
            if progress is not None:
                progress(read_bytes)
    return out


def save_tensors(path: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> str:
    """Atomic safetensors write; returns the sha256 of the finished file."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata=metadata)
    os.replace(tmp, path)
    return sha256_file(path)


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Digest of the fp32 bytes -- dtype-stable across bf16/fp16 producers."""
    arr = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def write_receipt(path: str, payload: dict[str, Any]) -> str:
    """Canonical-JSON receipt, atomically written; returns its own sha256."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8")
    # Binary writes keep the digest identical to the committed bytes on Windows too.
    with tempfile.NamedTemporaryFile("wb", delete=False,
                                      dir=os.path.dirname(os.path.abspath(path))) as handle:
        handle.write(body)
    os.replace(handle.name, path)
    return hashlib.sha256(body).hexdigest()


# --------------------------------------------------------------------------
# MXFP4  (compressed-tensors "mxfp4-pack-quantized", group_size 32)
# --------------------------------------------------------------------------
#: E2M1 magnitudes for codes 0..7; bit 3 is the sign.  Identical to
#: compressed_tensors' kE2M1ToFloat; `mxfp4_reference_check` asserts that.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def dequant_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    group: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """packed uint8 [out, in/2] + E8M0 uint8 scale [out, in/group] -> [out, in].

    Layout (verified against compressed_tensors 0.18.0
    `compressors/.../unpack_fp4_from_uint8`): the LOW nibble is the EVEN column
    and the HIGH nibble the odd one; the scale is a power of two, 2**(s - 127),
    shared by 32 consecutive INPUT columns.
    """
    device = packed.device
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    table = E2M1.to(device)
    mag = table[(codes & 0x07).long()]
    vals = torch.where((codes & 0x08) != 0, -mag, mag)
    exponent = scale.to(torch.int32) - 127
    factor = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    vals = vals.reshape(packed.shape[0], -1, group) * factor.unsqueeze(-1)
    return vals.reshape(packed.shape[0], -1).to(dtype)


def _import_compressed_tensors_helpers() -> dict[str, Any]:
    """Resolve compressed_tensors' own unpack/scale helpers, if installed.

    The module path moved between releases, so we search instead of hardcoding
    one; whichever path resolves is recorded in the receipt.
    """
    import importlib

    candidates = [
        ("compressed_tensors.compressors.quantized_compressors.nvfp4_quantized", "unpack_fp4_from_uint8"),
        ("compressed_tensors.compressors.nvfp4.helpers", "unpack_fp4_from_uint8"),
        ("compressed_tensors.utils.helpers", "unpack_fp4_from_uint8"),
    ]
    found: dict[str, Any] = {"unpack": None, "unpack_module": None,
                             "descale": None, "descale_module": None, "table": None}
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, attr):
            found["unpack"] = getattr(module, attr)
            found["unpack_module"] = module_name
            for table_name in ("kE2M1ToFloat", "FP4_E2M1_TABLE", "E2M1_TABLE"):
                if hasattr(module, table_name):
                    found["table"] = getattr(module, table_name)
                    break
            break
    for module_name in ("compressed_tensors.compressors.mx_utils",
                        "compressed_tensors.compressors.quantized_compressors.mx_utils",
                        "compressed_tensors.utils.mx_utils"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "decompress_mx_scale"):
            found["descale"] = module.decompress_mx_scale
            found["descale_module"] = module_name
            break
    return found


def mxfp4_reference_check(packed: torch.Tensor, scale: torch.Tensor, group: int = 32) -> dict[str, Any]:
    """Cross-check `dequant_mxfp4` against compressed_tensors' own helpers.

    Returns a receipt fragment.  If the library is absent this records
    `available: false` and never raises -- the harness must run on a box that
    does not have it, and a missing cross-check is a gap, not a failure.
    """
    helpers = _import_compressed_tensors_helpers()
    fragment: dict[str, Any] = {
        "available": helpers["unpack"] is not None,
        "unpack_module": helpers["unpack_module"],
        "descale_module": helpers["descale_module"],
    }
    if helpers["table"] is not None:
        try:
            table = torch.as_tensor(helpers["table"], dtype=torch.float32).flatten()[:8]
            fragment["e2m1_table_matches"] = bool(torch.equal(table.cpu(), E2M1))
            fragment["e2m1_table_library"] = [float(x) for x in table.cpu()]
        except Exception as exc:  # pragma: no cover - table shape surprises
            fragment["e2m1_table_error"] = repr(exc)[:200]
    if helpers["unpack"] is None:
        return fragment

    out_rows = int(packed.shape[0])
    in_cols = int(packed.shape[1]) * 2
    ours = dequant_mxfp4(packed.cpu(), scale.cpu(), group=group, dtype=torch.float32)
    try:
        ref_codes = helpers["unpack"](packed.cpu(), out_rows, in_cols, torch.float32)
        if helpers["descale"] is not None:
            ref_scale = helpers["descale"](scale.cpu()).float()
        else:
            ref_scale = torch.ldexp(
                torch.ones_like(scale.cpu(), dtype=torch.float32), scale.cpu().to(torch.int32) - 127
            )
        ref = ref_codes * ref_scale.repeat_interleave(group, dim=1)
        fragment["max_abs_difference"] = float((ours - ref).abs().max())
        fragment["bit_exact"] = bool(torch.equal(ours, ref))
        fragment["reference_sha256"] = sha256_tensor(ref)
    except Exception as exc:
        fragment["error"] = repr(exc)[:300]
    fragment["ours_sha256"] = sha256_tensor(ours)
    return fragment


# --------------------------------------------------------------------------
# K3 arithmetic primitives (transcribed from modeling_kimi_linear.py)
# --------------------------------------------------------------------------
def blocks_before(layer_idx: int, block_size: int) -> int:
    """How many block snapshots the residual stack holds when layer `layer_idx` starts.

    `_forward_attn_residual` pushes a snapshot when ``layer_idx % block_size == 0``,
    so the stack entering layer L holds ``|{i in [0, L) : i % B == 0}| = ceil(L / B)``.
    The obvious ``L // B`` is only right when ``B == 1`` -- at K3's real
    ``attn_res_block_size = 12`` it undercounts every layer that is not a
    multiple of 12 (L=5, B=3 holds two snapshots, not one).  Real K3: 93 layers
    at B=12 tops out at ceil(92/12) = 8, which is the "up to eight" this lane
    keeps quoting.
    """
    if block_size <= 0:
        raise SystemExit(f"attn_res_block_size must be positive, got {block_size}")
    return -(-int(layer_idx) // int(block_size))


def _windows_peak_rss_bytes() -> int | None:
    """Native process-lifetime peak working set, in bytes (not commit size)."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize) or None


def peak_rss_source() -> str:
    if sys.platform == "win32":
        return "GetProcessMemoryInfo(GetCurrentProcess()).PeakWorkingSetSize (bytes)"
    if sys.platform == "linux":
        return "/proc/self/status VmHWM; fallback resource.ru_maxrss (KiB * 1024)"
    scale = "bytes" if sys.platform == "darwin" else "KiB * 1024"
    return f"resource.getrusage(RUSAGE_SELF).ru_maxrss ({scale})"


def peak_rss_bytes() -> int | None:
    """High-water-mark resident set size of THIS process; unknown is None, never zero.

    Used to turn "the Gram arithmetic says N GB" into a measurement.  `VmHWM`
    remains Linux's first source. Windows uses native PeakWorkingSetSize;
    `ru_maxrss` is the Unix fallback (KiB on Linux, bytes on macOS).
    """
    if sys.platform == "win32":
        try:
            return _windows_peak_rss_bytes()
        except (ImportError, AttributeError, OSError):
            return None
    try:
        if sys.platform == "linux":
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        value = int(line.split()[1])
                        if value > 0:
                            return value * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * (1 if sys.platform == "darwin" else 1024) if value > 0 else None
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return None


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """KimiRMSNorm: fp32 internally, weight applied after the cast back."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is None:
        return xf.to(dtype)
    return (weight * xf.to(dtype))


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """SituAndMul on an already-split gate/up pair.

    `beta * tanh(g/beta) * sigmoid(g) * (linear_beta * tanh(u/linear_beta))`,
    computed in fp32 and cast back to the input dtype -- byte-for-byte the
    repo's `SituAndMul.forward` on `cat([gate, up], -1)`.
    """
    dtype = gate.dtype
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    activated = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (activated * u).to(dtype)


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    token_chunk_size: int | None = 1024,
) -> torch.Tensor:
    """`_apply_attn_res`: a softmax read over the block snapshots + the prefix.

    prefix_sum      [tokens, hidden]
    block_residual  [tokens, blocks, hidden]
    proj_weight     [1, hidden]        (self_attention_res_proj / mlp_res_proj)
    norm_weight     [hidden]           (the matching res_norm)

    Slice the token axis BEFORE concatenating snapshots or upcasting to fp32.
    During inference/no_grad, only the [tokens, hidden] output is allocated at
    full length; autograd may retain per-chunk intermediates. None keeps
    the original unchunked expression for comparison; five positional callers
    receive the bounded path without changing their call sites.
    """
    if token_chunk_size is not None and (
        isinstance(token_chunk_size, bool)
        or not isinstance(token_chunk_size, int)
        or token_chunk_size <= 0
    ):
        raise ValueError("token_chunk_size must be a positive integer or None")
    if (
        prefix_sum.ndim != 2
        or block_residual.ndim != 3
        or block_residual.shape[0] != prefix_sum.shape[0]
        or block_residual.shape[2] != prefix_sum.shape[1]
    ):
        raise ValueError("AttnRes inputs must be [tokens, hidden] and [tokens, blocks, hidden] with matching dimensions")
    tokens = prefix_sum.shape[0]
    if token_chunk_size is None or tokens <= token_chunk_size:
        return _apply_attn_res_chunk(prefix_sum, block_residual, proj_weight, norm_weight, eps)

    result = prefix_sum.new_empty(
        prefix_sum.shape, dtype=torch.promote_types(prefix_sum.dtype, block_residual.dtype)
    )
    for start in range(0, tokens, token_chunk_size):
        stop = min(start + token_chunk_size, tokens)
        result[start:stop] = _apply_attn_res_chunk(
            prefix_sum[start:stop], block_residual[start:stop], proj_weight, norm_weight, eps
        )
    return result


def _apply_attn_res_chunk(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The original numerical kernel, called only on a bounded token slice."""

    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


# --------------------------------------------------------------------------
# the model index
# --------------------------------------------------------------------------
class K3Index:
    """`model.safetensors.index.json` + `config.json`, and nothing else read.

    Deliberately never lists the directory: `/mnt/ai_data` is a CIFS share and
    a recursive walk of it is the one thing this lane must not do.
    """

    LAYER_PREFIXES = ("language_model.model.layers.", "model.layers.")

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as handle:
            self.config_json: dict = json.load(handle)
        self.text_config: dict = self.config_json.get("text_config", self.config_json)
        # Which prefix this checkpoint actually uses (K3 = language_model.*).
        self.layer_prefix = next(
            (p for p in self.LAYER_PREFIXES if any(k.startswith(p) for k in self.weight_map)),
            self.LAYER_PREFIXES[0],
        )
        self.model_prefix = self.layer_prefix[: -len("layers.")]

    # -- layout ------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return int(self.text_config["num_hidden_layers"])

    @property
    def hidden_size(self) -> int:
        return int(self.text_config["hidden_size"])

    @property
    def attn_res_block_size(self) -> int | None:
        return self.text_config.get("attn_res_block_size")

    def is_kda_layer(self, layer_idx: int) -> bool:
        """`configuration_kimi_k3.is_kda_layer`: ONE-BASED membership test."""
        cfg = self.text_config.get("linear_attn_config")
        return bool(cfg) and (layer_idx + 1) in cfg["kda_layers"]

    def is_moe_layer(self, layer_idx: int) -> bool:
        cfg = self.text_config
        if cfg.get("num_experts") is None:
            return False
        return (
            layer_idx >= int(cfg.get("first_k_dense_replace", 0))
            and layer_idx % int(cfg.get("moe_layer_freq", 1)) == 0
        )

    # -- key groups --------------------------------------------------------
    def layer_prefix_for(self, layer_idx: int) -> str:
        return f"{self.layer_prefix}{layer_idx}."

    def layer_keys(self, layer_idx: int) -> list[str]:
        prefix = self.layer_prefix_for(layer_idx)
        return [k for k in self.weight_map if k.startswith(prefix)]

    def shards_for(self, keys: Iterable[str]) -> list[str]:
        return sorted({self.weight_map[k] for k in keys})

    def model_level_keys(self) -> list[str]:
        prefix = self.layer_prefix
        return [
            k
            for k in self.weight_map
            if not k.startswith(prefix) and ("vision" not in k and "mm_projector" not in k)
        ]

    # -- reads -------------------------------------------------------------
    def read_keys(self, keys: Iterable[str], progress_tag: str | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        """Read a key set shard by shard; returns (tensors, timing fragment)."""
        keys = list(keys)
        tensors: dict[str, torch.Tensor] = {}
        total_bytes = 0
        started = time.time()
        per_shard = []
        for shard in self.shards_for(keys):
            wanted = [k for k in keys if self.weight_map[k] == shard]
            shard_start = time.time()
            got = read_tensors(os.path.join(self.model_dir, shard), wanted)
            got_bytes = sum(t.numel() * t.element_size() for t in got.values())
            tensors.update(got)
            total_bytes += got_bytes
            elapsed = max(time.time() - shard_start, 1e-9)
            per_shard.append(
                {
                    "shard": shard,
                    "tensors": len(got),
                    "gigabytes": round(got_bytes / 1e9, 3),
                    "seconds": round(elapsed, 1),
                    "MB_per_s": round(got_bytes / 1e6 / elapsed, 1),
                }
            )
            if progress_tag:
                log(f"{progress_tag}: {shard} {len(got)} tensors "
                    f"{got_bytes/1e9:.2f} GB in {elapsed:.0f}s = {got_bytes/1e6/elapsed:.1f} MB/s")
        wall = max(time.time() - started, 1e-9)
        timing = {
            "tensors": len(tensors),
            # exact bytes as well as the rounded GB: at synthetic scale the GB
            # figure rounds to 0.0 and "did the fold actually skip the experts?"
            # becomes untestable
            "bytes": total_bytes,
            "gigabytes": round(total_bytes / 1e9, 3),
            "seconds": round(wall, 1),
            "MB_per_s": round(total_bytes / 1e6 / wall, 1),
            "shards": per_shard,
        }
        missing = [k for k in keys if k not in tensors]
        if missing:
            raise SystemExit(f"index promised {len(missing)} tensors the shards do not carry, e.g. {missing[:3]}")
        return tensors, timing


# --------------------------------------------------------------------------
# scoring (mirrors mx45_build.write_scores)
# --------------------------------------------------------------------------
def write_scores(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    diff = pred - target
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).flatten()
    # torch.quantile refuses a q tensor on a different device than its input
    quantile = torch.tensor(0.05, dtype=cos.dtype, device=cos.device)
    return {
        "relL2": float(diff.norm() / max(float(target.norm()), 1e-30)),
        "cos": float(cos.mean()),
        "cos_p05": float(torch.quantile(cos, quantile)),
    }


def canonical_device(device: torch.device | str | int) -> torch.device:
    """A `torch.device` with its index spelled out, so two names for one card compare equal.

    `torch.device('cuda') != torch.device('cuda:0')`, and `torch.device('cpu:0')
    != torch.device('cpu')`, even though every tensor that lands on the first of
    each pair reports the second.  Every device comparison in this lane was a
    raw `==` against the device the caller named, so `--device cuda` made
    `LayerWeights.device_bytes` and `CoresetFold.device_resident_bytes` count
    ZERO bytes for tensors sitting on that very card: the RESIDENT half of the
    C14 budget silently vanished and `--max-device-gb` went straight back to
    pricing ~4% of the truth, which is the failure t13d exists to stop.  A
    zeroed estimate does not fail loudly -- it admits the rung and then dies
    inside the allocator.

    Two rules, both taken from what a tensor actually reports:

    * `cuda` with no index means the CURRENT device -- what `.to('cuda')` uses --
      not device 0.  Those differ the moment anything calls `torch.cuda.set_device`,
      and under `CUDA_VISIBLE_DEVICES` the current device is what the card
      answers to anyway.  With no CUDA present it degrades to index 0 rather
      than raising, because the callers here also price CPU-only runs.
    * `cpu` and `meta` tensors NEVER carry an index (verified: a tensor built on
      `'cpu:0'` reports `device(type='cpu')`), so an index written on those is
      dropped rather than invented.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        index = dev.index
        if index is None:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    if dev.type in ("cpu", "meta") and dev.index is not None:
        return torch.device(dev.type)
    return dev


def same_device(a: torch.device | str | int, b: torch.device | str | int) -> bool:
    """Do these two names refer to the same device?  The only device test in this lane.

    Both sides are canonicalised, so `t.device` (always indexed on cuda, never
    indexed on cpu) compares correctly against whatever a `--device` flag spelt.
    """
    return canonical_device(a) == canonical_device(b)


def device_inventory() -> list[dict[str, Any]]:
    """What CUDA devices this process can see -- pinned into every receipt."""
    inventory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            inventory.append(
                {
                    "index": i,
                    "name": props.name,
                    "capability": f"sm{props.major}{props.minor}",
                    "total_GB": round(props.total_memory / 1e9, 1),
                }
            )
    return inventory
