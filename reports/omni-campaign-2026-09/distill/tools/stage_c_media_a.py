#!/usr/bin/env python3
"""stage_c_media_a.py -- stage C teacher caches, media lane A: MiniMax H3 (AV) and Qwen-Image.

Lane media-a (Claude, 2026-10-12). Design of record: distill/DISTILL_DESIGN_2026-10-12.md §3 (cache
plan: sealed, content-addressed, resumable), §5.2 (H3), §5.3 (QIMG); rulings STAGE_D_RULINGS_2026-10-12.md
(H3 included; 2,048 H3 targets unless P4 > 20 s/forward).

Every cache is written once to

    <cache-root>/<teacher>/<corpus_sha16>/shard-NNNNN.safetensors (+ .json sidecar)   manifest.json (+ .sha256)

The manifest records the teacher's integrity check BEFORE and AFTER (full sha256 by default), the corpus
sha (sha256 of the item JSONL), the plan and its sha, the shard read order, a finite check and sha256
per shard, the runtime and every named deviation. A rerun resumes: a shard whose sidecar json exists
and whose bytes still hash to it is skipped; a different plan into the same directory is a refusal.

Subcommands (the GPU ones run INSIDE comfyui:codex-minimax-h3-runtime-env-r5-20260808, ComfyUI 0.30.0,
the only place the parent code lives; launch lines are in run_media_a_wave2.sh):

    manifest        curated registry -> item JSONL (CPU, chimera_omni.data.curated_shards)       venv
    plan            counts, shapes and bytes of a planned cache, no GPU, no comfy                 any
    h3-encode       curated clips -> H3 video VAE (fp16) + audio VAE (fp32) latents              image
    h3-text         prompts (+ negatives) -> Qwen3-VL-32B H3 conditioning + minimax_token_tags   image
    h3-velocity     (x0, seed, sigma, guided v, x0_hat) per target + block-L in/out row pairs    image
    qimg-encode     curated images -> Qwen-Image VAE latents                                     image
    qimg-text       prompts + negatives -> qwen_2.5_vl_7b_fp8_scaled conditioning                image
    qimg-velocity   FP8 E4M3 storage, ComfyUI manual_cast compute -> guided v per target         image
    probe           S0 P4: s/forward, peak VRAM, host RSS of one teacher component              image
    verify          re-hash every shard of a sealed cache against its manifest                   any

Conventions (read out of comfy/ldm/minimax/model.py and comfy/model_sampling.py in the r5 image):

* Rectified flow, comfy CONST: x_t = (1 - sigma) * x0 + sigma * noise; the sampler's model output v
  satisfies denoised = x_t - sigma * v, so v = noise - x0 for a perfect model. Targets are taken through
  ComfyUI's own CFGGuider + KSAMPLER call path (process_latent_in, extra_conds, the H3 packed layout,
  pack/unpack of the nested AV latent), with a sampler function that evaluates the guided model at OUR
  x_t, so the cached values are exactly what a ComfyUI sampler would integrate.
* H3 runs one flat ODE on the video sigma. The audio stream lives on its own shifted schedule
  sigma_a = time_shift_sigma(sigma_v, 12, 3) and its returned velocity is scaled by
  slope = d sigma_a / d sigma_v. This tool noises audio at sigma_a (on the sampler's trajectory, the c16
  S6H convention), stores v in the FLAT convention (what comfy integrates) and the NATIVE audio
  quantities: v_audio_native = v_audio_flat / slope, x0hat_audio = x_a - sigma_a * v_audio_native.
  x0hat_video = denoised_video (exact). x0-parametrisation note (H3 hybrid AV flow proof, 2026-08-28):
  pure velocity regression did not train at small scale and the proof trained joint AV x0 prediction,
  so every target stores BOTH v and x0_hat, and every item stores its clean model-space x0.
* Guidance. H3's official ComfyUI route is BasicGuider (one positive branch, cfg 1): --cfg 1 runs ONE
  forward per target. cfg != 1 runs cond and uncond as two separate single-branch passes (H3 refuses
  batch 2) and combines uncond + cfg * (cond - uncond) on denoised in fp32 (comfy.samplers.cfg_function
  without model patches); v_cond and v_uncond are stored beside v.
* Noise: one CPU torch.Generator per target, seeded seed = base_seed * 1_000_003 + item * 64 + j,
  float32 randn in stream order (video, then audio). The noise is NOT stored; its sha256 is, so a trainer
  regenerates it and proves it bit-identical.

1.0.1 (2026-10-13, after the P40 integration runs): only a CPU media DECODE failure drops an item (recorded);
a CUDA / device error re-raises and a VAE encode failure is fatal (1.0.0 caught an OOM as "dropped" and sealed an
EMPTY cache); sealing refuses zero data shards and a drop fraction above --max-drop-fraction; shard sidecars are
matched exactly (shard-NNNNN.json, never the .empty.json markers); the VRAM guard subtracts this process's own
context + reservation on a second run in one process and can --guard-wait on a shared card; text encoders can be
pinned with --te-load-device (ComfyUI puts them on the CPU on a P40: 126.5 s/text measured).

1.0.2 (2026-10-13, review of 1.0.1): every shard sidecar and empty-shard marker carries a RUNTIME SIGNATURE (tool
version + sha, device name + capability, torch, comfy, the component's weight / inference / manual_cast dtypes);
a resume whose signature differs from any existing data shard is a refusal (checked once before the teacher loads
and again after), and sealing refuses a directory with more than one signature, so a cache is never a mix of card
classes, compute dtypes or tool versions. The manifest's "runtime_signature" comes from the shards, not from the
sealing process. Encode drops are stored PER SHARD (data sidecar "dropped" + the .empty.json marker, which a later
data shard for that index deletes) and re-aggregated at seal time against the planned item count: every planned
item must be kept or dropped, and --max-drop-fraction is judged on the whole cache, not on the last invocation.
Only an affirmative decode verdict drops an item; the media file is hashed OUTSIDE the drop guard and an
environment OSError (ENOENT, EIO, ESTALE, EACCES, ...), MemoryError or device error is fatal (rerun resumes).

Standing rules: finite fit numbers are diagnostics; a non-finite value, a shape/identity mismatch or a
changed teacher digest is a functional failure (SystemExit). No GPU work starts on a card that already
has more than --vram-guard-mib in use.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

TOOL_NAME = "stage_c_media_a"
TOOL_VERSION = "1.0.2"
SCHEMA = "omni-campaign.stage_c.media_a.v1"
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
CACHE_ROOT_DEFAULT = OC / "teacher_cache"
COMFY_ROOTS = ("/opt/ComfyUI", "/workspace/ComfyUI")
MODELS = Path("/opt/ComfyUI/models")

H3_DIT_BF16 = MODELS / "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
H3_DIT_INT8 = MODELS / "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_TE_BF16 = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
H3_TE_NVFP4 = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = MODELS / "vae/minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors"
QIMG_DIT = MODELS / "diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
QIMG_TE = MODELS / "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
QIMG_VAE = MODELS / "vae/qwen_image_vae.safetensors"

# H3 geometry (comfy_extras/nodes_minimax_h3.py + comfy/ldm/minimax/model.py, r5 image)
H3_FPS = 24
H3_AUDIO_LATENT_FPS = 40
H3_AUDIO_SAMPLE_RATE = 32000
H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32
H3_SPATIAL = 16
H3_PATCH = (1, 2, 2)
H3_HIDDEN = 5376
H3_SHIFT_VIDEO = 12.0
H3_SHIFT_AUDIO = 3.0
QIMG_SPATIAL = 8
QIMG_CHANNELS = 16
QIMG_SHIFT_DEFAULT = 1.15  # supported_models.QwenImage sampling_settings (no ModelSamplingAuraFlow node)
KIND_TEXT, KIND_AUDIO, KIND_VIDEO = 0, 1, 2
VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SEED_ITEM_STRIDE = 64
SEED_BASE_STRIDE = 1_000_003


# ======================================================================================
# small utilities (pure)
# ======================================================================================


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def write_json_atomic(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def _torch():
    import torch

    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(max(1, min(16, threads)))
    return torch


def tensor_digest(tensor: Any) -> str:
    """dtype + shape + raw bytes (the s7_pairs_parent / safetensors_io formula's inputs)."""

    torch = _torch()
    t = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(t.dtype).replace("torch.", "").encode())
    digest.update(json.dumps(list(t.shape)).encode())
    if t.dtype in (torch.bfloat16, torch.float16):
        digest.update(t.view(torch.int16).numpy().tobytes())
    elif t.dtype == torch.bool:
        digest.update(t.to(torch.uint8).numpy().tobytes())
    else:
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def rss_block() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                out[key] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass
    return out


# ======================================================================================
# geometry and schedules (pure, tested)
# ======================================================================================


def h3_align_frame_count(n: int) -> int:
    """nodes_minimax_h3.align_frame_count: the 17k + 5 frame grid."""

    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def h3_video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def h3_temporal_shape(length: int) -> tuple[int, int, int]:
    """(frame_count, video latent t, audio latent t) exactly as nodes_minimax_h3.temporal_shape."""

    frame_count = h3_align_frame_count(length)
    duration = frame_count / H3_FPS
    return frame_count, h3_video_latent_t(frame_count), int(round(duration * H3_AUDIO_LATENT_FPS))


def h3_latent_shapes(width: int, height: int, length: int) -> dict[str, Any]:
    if width % 32 or height % 32:
        raise SystemExit(f"H3 canvas {width}x{height}: both axes must be multiples of 32 (nodes_minimax_h3)")
    frame_count, latent_t, audio_t = h3_temporal_shape(length)
    video = (1, H3_VIDEO_CHANNELS, latent_t, height // H3_SPATIAL, width // H3_SPATIAL)
    audio = (1, H3_AUDIO_CHANNELS, 2, audio_t)
    video_tokens = latent_t * (height // H3_SPATIAL // H3_PATCH[1]) * (width // H3_SPATIAL // H3_PATCH[2])
    return {"frame_count": frame_count, "seconds": frame_count / H3_FPS, "video": video, "audio": audio,
            "video_tokens": video_tokens, "audio_tokens": 2 * audio_t, "av_tokens": video_tokens + 2 * audio_t}


def qimg_latent_shape(width: int, height: int) -> dict[str, Any]:
    if width % 16 or height % 16:
        raise SystemExit(f"QIMG canvas {width}x{height}: both axes must be multiples of 16 (VAE /8, patch 2)")
    latent = (1, QIMG_CHANNELS, 1, height // QIMG_SPATIAL, width // QIMG_SPATIAL)
    return {"latent": latent, "image_tokens": (height // QIMG_SPATIAL // 2) * (width // QIMG_SPATIAL // 2)}


def time_shift(u: float, shift: float) -> float:
    """comfy flow time shift: sigma = shift * u / (1 + (shift - 1) * u)."""

    return float(shift) * u / (1.0 + (float(shift) - 1.0) * u)


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """comfy/ldm/minimax/model.py: invert the from-shift to the base grid, re-apply the to-shift."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: float, from_shift: float, to_shift: float) -> float:
    """d sigma_to / d sigma_from at the same base-grid point (comfy/ldm/minimax/model.py)."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (from_shift * (1.0 + (to_shift - 1.0) * base) ** 2)


def target_seed(base_seed: int, item_index: int, j: int) -> int:
    if not 0 <= j < SEED_ITEM_STRIDE:
        raise ValueError(f"sigma index {j} outside [0, {SEED_ITEM_STRIDE})")
    return int(base_seed) * SEED_BASE_STRIDE + int(item_index) * SEED_ITEM_STRIDE + int(j)


def sigma_plan(n_items: int, *, per_item: int, mode: str, shift: float, seed: int,
               sigmas: Sequence[float] | None = None, lo: float = 0.02, hi: float = 0.98) -> list[dict[str, Any]]:
    """One row per target: item, j, sigma, seed.

    stratified-shifted: per item, `per_item` strata of u in [lo, hi], one seeded jittered u per stratum, mapped
                        through the model's time shift (the sampler's sigma distribution, every noise band
                        covered per item).
    fixed:              every item gets the explicit `sigmas` (len == per_item).
    cycle:              item k gets the single sigma sigmas[k % len] (per_item must be 1).
    """

    if per_item < 1 or per_item > SEED_ITEM_STRIDE:
        raise SystemExit(f"--sigmas-per-item {per_item} outside [1, {SEED_ITEM_STRIDE}]")
    rows: list[dict[str, Any]] = []
    if mode == "stratified-shifted":
        if not 0.0 < lo < hi < 1.0:
            raise SystemExit("stratified sigma bounds need 0 < lo < hi < 1")
        rng = random.Random(int(seed) * 7919 + 17)
        width = (hi - lo) / per_item
        for item in range(n_items):
            for j in range(per_item):
                u = lo + width * (j + rng.random())
                rows.append({"item": item, "j": j, "u": u, "sigma": time_shift(u, shift), "seed": target_seed(seed, item, j)})
    elif mode == "fixed":
        values = [float(s) for s in (sigmas or [])]
        if len(values) != per_item or any(not 0.0 < s < 1.0 for s in values):
            raise SystemExit(f"--sigma-mode fixed needs exactly {per_item} sigmas in (0, 1)")
        for item in range(n_items):
            for j, s in enumerate(values):
                rows.append({"item": item, "j": j, "u": None, "sigma": s, "seed": target_seed(seed, item, j)})
    elif mode == "cycle":
        values = [float(s) for s in (sigmas or [])]
        if per_item != 1 or not values or any(not 0.0 < s < 1.0 for s in values):
            raise SystemExit("--sigma-mode cycle needs --sigmas-per-item 1 and a non-empty --sigmas list in (0, 1)")
        for item in range(n_items):
            rows.append({"item": item, "j": 0, "u": None, "sigma": values[item % len(values)], "seed": target_seed(seed, item, 0)})
    else:
        raise SystemExit(f"unknown sigma mode {mode!r}")
    return rows


def make_noise(shapes: Sequence[Sequence[int]], seed: int) -> list[Any]:
    """float32 randn on a CPU generator, streams drawn in order (video, then audio)."""

    torch = _torch()
    generator = torch.Generator().manual_seed(int(seed))
    return [torch.randn(tuple(int(v) for v in shape), generator=generator, dtype=torch.float32) for shape in shapes]


def noise_sha(noises: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for n in noises:
        digest.update(tensor_digest(n).encode())
    return digest.hexdigest()


def flow_interp(x0: Any, noise: Any, sigma: float) -> Any:
    return (1.0 - float(sigma)) * x0.float() + float(sigma) * noise.float()


def velocity_from_denoised(x_t: Any, denoised: Any, sigma: float) -> Any:
    """v = (x_t - denoised) / sigma in float64, returned as float32 (comfy CONST inverse)."""

    torch = _torch()
    return ((x_t.to(torch.float64) - denoised.to(torch.float64)) / float(sigma)).to(torch.float32)


def cfg_combine(cond: Any, uncond: Any | None, scale: float) -> Any:
    """comfy.samplers.cfg_function without patches: uncond + (cond - uncond) * scale; cfg 1 -> cond."""

    if uncond is None or math.isclose(float(scale), 1.0):
        return cond
    return uncond + (cond - uncond) * float(scale)


def h3_native_targets(*, x_video: Any, x_audio: Any, den_video: Any, den_audio: Any, sigma_v: float) -> dict[str, Any]:
    """Flat (comfy) and native per-stream targets from one guided denoised H3 output."""

    sigma_a = time_shift_sigma(sigma_v, H3_SHIFT_VIDEO, H3_SHIFT_AUDIO)
    slope = time_shift_slope(sigma_v, H3_SHIFT_VIDEO, H3_SHIFT_AUDIO)
    v_video = velocity_from_denoised(x_video, den_video, sigma_v)
    v_audio = velocity_from_denoised(x_audio, den_audio, sigma_v)  # flat: the sampler's ODE on sigma_v
    torch = _torch()
    v_audio_native = (v_audio.to(torch.float64) / slope).to(torch.float32)
    x0hat_audio = (x_audio.to(torch.float64) - sigma_a * v_audio_native.to(torch.float64)).to(torch.float32)
    return {"v_video": v_video, "v_audio": v_audio, "v_audio_native": v_audio_native,
            "x0hat_video": den_video.float().clone(), "x0hat_audio": x0hat_audio,
            "sigma_audio": sigma_a, "slope_audio": slope}


def h3_packed_kinds(text_len: int, audio_t: int, video_tokens: int) -> list[int]:
    """Row kinds of the plain t2va packed sequence [text | audio (2 channels x T) | video]."""

    return [KIND_TEXT] * int(text_len) + [KIND_AUDIO] * (2 * int(audio_t)) + [KIND_VIDEO] * int(video_tokens)


def block_row_selection(kinds: Sequence[int], k: int, seed: int, *, include_text: bool = False) -> list[int]:
    """Seeded, kind-stratified, sorted row subset of one forward (audio/video proportional to their counts,
    at least one row of each present kind)."""

    pools: dict[int, list[int]] = {}
    for row, kind in enumerate(kinds):
        if kind == KIND_TEXT and not include_text:
            continue
        pools.setdefault(int(kind), []).append(row)
    total = sum(len(v) for v in pools.values())
    if total == 0:
        return []
    k = min(int(k), total)
    rng = random.Random(int(seed) * 104729 + 3)
    quotas = {kind: max(1, int(round(k * len(rows) / total))) for kind, rows in pools.items()}
    while sum(quotas.values()) > k:
        biggest = max(quotas, key=lambda q: quotas[q])
        quotas[biggest] -= 1
    while sum(quotas.values()) < k:
        room = [q for q in quotas if quotas[q] < len(pools[q])]
        quotas[max(room, key=lambda q: len(pools[q]) - quotas[q])] += 1
    chosen: list[int] = []
    for kind in sorted(pools):
        chosen.extend(rng.sample(pools[kind], min(quotas[kind], len(pools[kind]))))
    return sorted(chosen)


def frame_times_for_target(n_frames: int, fps: float, start: float = 0.0) -> list[float]:
    return [start + k / float(fps) for k in range(int(n_frames))]


def pick_frames(source_times: Sequence[float], target_times: Sequence[float]) -> list[int]:
    """For each target time the LATEST source frame with pts <= t (+1 us), i.e. sample-and-hold resampling."""

    picks: list[int] = []
    s = 0
    for t in target_times:
        while s + 1 < len(source_times) and source_times[s + 1] <= t + 1e-6:
            s += 1
        if source_times[s] > t + 1e-6:
            raise ValueError(f"no source frame at or before t={t:.4f}s")
        picks.append(s)
    return picks


def cover_crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """(resize_w, resize_h, left, top): scale to cover dst, then centre crop."""

    scale = max(dst_w / src_w, dst_h / src_h)
    rw, rh = max(dst_w, int(math.ceil(src_w * scale))), max(dst_h, int(math.ceil(src_h * scale)))
    return rw, rh, (rw - dst_w) // 2, (rh - dst_h) // 2


# ======================================================================================
# plan arithmetic (pure, tested)
# ======================================================================================


def plan_bytes(*, teacher: str, items: int, per_item: int, cfg: float, width: int, height: int, length: int = 0,
               block_rows: int = 0, block_store_bytes: int = 2) -> dict[str, Any]:
    targets = items * per_item
    if teacher == "h3":
        geo = h3_latent_shapes(width, height, length)
        floats = math.prod(geo["video"]) + math.prod(geo["audio"])
        per_target_arrays = 2 + (2 if not math.isclose(cfg, 1.0) else 0)  # v + x0hat (+ v_cond, v_uncond)
        block = 2 * block_rows * H3_HIDDEN * block_store_bytes
        forwards = targets * (1 if math.isclose(cfg, 1.0) else 2)
        out = {"geometry": geo, "floats_per_latent": floats}
    elif teacher == "qimg":
        geo = qimg_latent_shape(width, height)
        floats = math.prod(geo["latent"])
        per_target_arrays = 2 + (2 if not math.isclose(cfg, 1.0) else 0)
        block = 0
        forwards = targets * (1 if math.isclose(cfg, 1.0) else 2)
        out = {"geometry": geo, "floats_per_latent": floats}
    else:
        raise SystemExit(f"unknown teacher {teacher!r}")
    per_target = per_target_arrays * floats * 4 + block
    x0_bytes = items * floats * 4
    out.update({
        "items": items, "targets": targets, "forwards": forwards, "cfg": cfg,
        "bytes_per_target": per_target, "bytes_targets": per_target * targets, "bytes_x0_model": x0_bytes,
        "bytes_block_pairs": block * targets, "bytes_total": per_target * targets + x0_bytes,
        "gib_total": round((per_target * targets + x0_bytes) / 2**30, 3),
    })
    return out


# ======================================================================================
# items manifest (pure, tested)
# ======================================================================================


def read_items(path: str | Path, *, media_key: str, need_negative: bool = False) -> list[dict[str, Any]]:
    """Item JSONL: {"id", "prompt", media_key: path, ["negative"], ["audio"], ["split"], ["sha256"]}."""

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            ident = str(row.get("id") or "")
            if not ident:
                raise SystemExit(f"{path}:{line_number}: item without id")
            if ident in seen:
                raise SystemExit(f"{path}:{line_number}: duplicate id {ident!r}")
            seen.add(ident)
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise SystemExit(f"{path}:{line_number}: item {ident!r} has no prompt")
            if media_key and not row.get(media_key):
                raise SystemExit(f"{path}:{line_number}: item {ident!r} has no {media_key!r} path")
            if need_negative and "negative" in row and not isinstance(row["negative"], str):
                raise SystemExit(f"{path}:{line_number}: item {ident!r} negative must be a string")
            items.append(row)
    if not items:
        raise SystemExit(f"{path}: no items")
    return items


def bank_strings(paths: Iterable[str | Path]) -> set[str]:
    """Every string value (normalised) inside the given bank / probe / smoke JSON files."""

    out: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            norm = " ".join(value.split()).lower()
            if len(norm) >= 12:
                out.add(norm)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in paths:
        walk(json.loads(Path(path).read_text(encoding="utf-8")))
    return out


def contaminated(text: str, bank: set[str]) -> bool:
    norm = " ".join(str(text).split()).lower()
    if not norm:
        return False
    return norm in bank or any(len(b) >= 24 and b in norm for b in bank)


# ======================================================================================
# shards, caches, manifests (pure, tested)
# ======================================================================================


def corpus_identity(items_path: str | Path) -> dict[str, Any]:
    sha = sha256_file(items_path)
    return {"items_file": str(items_path), "sha256": sha, "sha16": sha[:16]}


def cache_dir(root: str | Path, teacher: str, corpus_sha16: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", teacher):
        raise SystemExit(f"teacher dir name {teacher!r} is not [a-z0-9._-]")
    return Path(root) / teacher / corpus_sha16


def claim_cache(directory: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Pin the plan of a cache directory; a different plan into the same directory is a refusal."""

    directory.mkdir(parents=True, exist_ok=True)
    plan_sha = sha256_text(canonical_json(plan))
    pin = directory / "plan.json"
    if pin.is_file():
        existing = json.loads(pin.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan_sha:
            raise SystemExit(f"{directory}: holds plan {existing.get('plan_sha256', '?')[:16]}, this run is {plan_sha[:16]}; "
                             "a different plan needs its own --cache-tag (no overwrite)")
    else:
        write_json_atomic(pin, {"plan_sha256": plan_sha, "plan": plan, "tool": TOOL_NAME, "tool_version": TOOL_VERSION})
    return {"plan_sha256": plan_sha, "pin": str(pin)}


_TOOL_SHA: list[str] = []


def tool_sha256() -> str:
    if not _TOOL_SHA:
        _TOOL_SHA.append(sha256_file(Path(__file__).resolve()))
    return _TOOL_SHA[0]


SIGNATURE_RUNTIME_KEYS = ("device_requested", "device_name", "capability", "torch", "comfy_version",
                          "should_use_bf16", "should_use_fp16")


def runtime_signature(runtime: dict[str, Any], component: dict[str, Any] | None = None) -> dict[str, Any]:
    """What makes two shards of one cache comparable: the tool, the card class, the stack and the component's
    dtypes. `component` is None for the pre-load check (compared on the runtime keys only)."""

    sig: dict[str, Any] = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": tool_sha256(),
                           **{k: runtime.get(k) for k in SIGNATURE_RUNTIME_KEYS}}
    if component is not None:
        sig["component"] = {k: component[k] for k in sorted(component)}
    return sig


def signature_differences(existing: dict[str, Any] | None, current: dict[str, Any], *, runtime_only: bool) -> list[str]:
    if existing is None:
        return ["<no runtime_signature: written by stage_c_media_a < 1.0.2>"]
    keys = [k for k in current if k != "component"] if runtime_only else sorted(set(existing) | set(current))
    return [f"{k}: {existing.get(k)!r} -> {current.get(k)!r}" for k in keys if existing.get(k) != current.get(k)]


class ShardStore:
    """Atomic, resumable, finite-checked safetensors shards with a json sidecar each. Every sidecar (and empty-shard
    marker) carries the runtime signature of the process that wrote it."""

    def __init__(self, directory: Path, signature: dict[str, Any]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.signature = signature

    def path(self, index: int) -> Path:
        return self.directory / f"shard-{index:05d}.safetensors"

    def empty_path(self, index: int) -> Path:
        return self.directory / f"shard-{index:05d}.empty.json"

    def check_resume(self, signature: dict[str, Any] | None = None, *, runtime_only: bool = False) -> int:
        """Refuse to add shards to a directory whose existing data shards were written under another signature.
        Empty-shard markers are not checked: they hold no data and are re-derived on every run."""

        current = signature or self.signature
        checked = 0
        for side in shard_sidecars(self.directory):
            existing = json.loads(side.read_text(encoding="utf-8")).get("runtime_signature")
            diff = signature_differences(existing, current, runtime_only=runtime_only)
            if diff:
                raise SystemExit(f"{side}: written under a different runtime ({'; '.join(diff)}); resuming here would mix "
                                 "teacher numerics in one cache. Rerun with the original card/tool or use a new --cache-tag")
            checked += 1
        return checked

    def done(self, index: int, *, verify: bool = True) -> dict[str, Any] | None:
        side = self.path(index).with_suffix(".json")
        if not side.is_file() or not self.path(index).is_file():
            return None
        meta = json.loads(side.read_text(encoding="utf-8"))
        if verify and sha256_file(self.path(index)) != meta.get("sha256"):
            raise SystemExit(f"{self.path(index)}: bytes do not hash to its sidecar (corrupt shard); move it aside and rerun")
        return meta

    def write(self, index: int, tensors: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        torch = _torch()
        from safetensors.torch import save_file

        finite: dict[str, bool] = {}
        for name, t in tensors.items():
            if t.is_floating_point():
                ok = bool(torch.isfinite(t.float()).all())
                finite[name] = ok
                if not ok:
                    raise SystemExit(f"shard {index}: non-finite values in {name}")
        path = self.path(index)
        tmp = path.with_name(path.name + ".tmp")
        save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, str(tmp))
        os.replace(tmp, path)
        record = {
            "shard": index, "file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "finite": all(finite.values()) if finite else True, "finite_checked": sorted(finite),
            "shapes": {k: list(v.shape) for k, v in tensors.items()},
            "dtypes": {k: str(v.dtype).replace("torch.", "") for k, v in tensors.items()},
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta,
            "runtime_signature": self.signature,
        }
        write_json_atomic(path.with_suffix(".json"), record)
        self.empty_path(index).unlink(missing_ok=True)  # an earlier all-dropped attempt of this shard is superseded
        return record

    def write_empty(self, index: int, dropped: list[dict[str, Any]]) -> None:
        """Marker for a shard whose every item was dropped; it is never 'done', so a rerun retries those items."""

        write_json_atomic(self.empty_path(index), {"shard": index, "dropped": dropped, "runtime_signature": self.signature,
                                                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


SHARD_SIDE_RE = re.compile(r"shard-\d{5}\.json")


def shard_sidecars(directory: Path) -> list[Path]:
    """Data-shard sidecars only (shard-NNNNN.json), never the shard-NNNNN.empty.json markers."""

    return sorted(p for p in Path(directory).glob("shard-*.json") if SHARD_SIDE_RE.fullmatch(p.name))


def empty_markers(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("shard-*.empty.json"))


def aggregate_drops(directory: Path, planned_items: int) -> dict[str, Any]:
    """Re-derive kept and dropped items of an encode cache from EVERY shard on disk (data sidecars + empty markers),
    so a resumed run accounts for the drops of earlier invocations. Every planned index must be kept or dropped."""

    kept: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for side in shard_sidecars(directory):
        meta = json.loads(side.read_text(encoding="utf-8"))
        kept.update(int(m["index"]) for m in meta.get("items", []))
        dropped.extend(meta.get("dropped", []))
    for marker in empty_markers(directory):
        dropped.extend(json.loads(marker.read_text(encoding="utf-8")).get("dropped", []))
    dropped_idx = [int(m["index"]) for m in dropped]
    both = kept & set(dropped_idx)
    if both or len(dropped_idx) != len(set(dropped_idx)):
        raise SystemExit(f"{directory}: items recorded twice (kept and dropped: {sorted(both)[:8]}); the shard records are inconsistent")
    missing = sorted(set(range(int(planned_items))) - kept - set(dropped_idx))
    extra = sorted((kept | set(dropped_idx)) - set(range(int(planned_items))))
    if missing or extra:
        raise SystemExit(f"{directory}: {len(missing)} planned items have no shard record (first {missing[:8]}), "
                         f"{len(extra)} records outside the plan; refusing to seal")
    return {"kept": len(kept), "dropped": sorted(dropped, key=lambda m: int(m["index"])), "planned": int(planned_items)}


def seal_manifest(directory: Path, body: dict[str, Any]) -> str:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in shard_sidecars(directory)]
    if not records:
        raise SystemExit(f"{directory}: no data shard was written; refusing to seal an empty cache")
    markers = [json.loads(p.read_text(encoding="utf-8")) for p in empty_markers(directory)]
    signatures = {canonical_json(r.get("runtime_signature")) for r in records + markers}
    if len(signatures) != 1 or records[0].get("runtime_signature") is None:
        raise SystemExit(f"{directory}: shards carry {len(signatures)} different runtime signatures (or none); "
                         "refusing to seal a cache that mixes runtimes")
    manifest = {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": tool_sha256()},
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "read_order": [r["file"] for r in records],
        "shards": [{k: r[k] for k in ("file", "sha256", "bytes", "finite") if k in r} for r in records],
        "all_finite": bool(records) and all(r.get("finite", False) for r in records),
        "empty_shard_markers": [p.name for p in empty_markers(directory)],
        "runtime_signature": records[0]["runtime_signature"],
        "runtime_signature_note": "identical on every shard and marker (checked at seal); 'runtime' below is the sealing process",
        **body,
    }
    path = directory / "manifest.json"
    sha = write_json_atomic(path, manifest)
    (directory / "manifest.json.sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def verify_cache(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{directory}: no manifest.json (not sealed)")
    sidecar = directory / "manifest.json.sha256"
    manifest_sha = sha256_file(manifest_path)
    if sidecar.is_file() and sidecar.read_text().strip() != manifest_sha:
        raise SystemExit(f"{manifest_path}: sha256 differs from manifest.json.sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for shard in manifest["shards"]:
        path = directory / shard["file"]
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            bad.append(shard["file"])
    teacher = manifest.get("teacher_integrity", {})
    return {"directory": str(directory), "manifest_sha256": manifest_sha, "shards": len(manifest["shards"]),
            "bad_shards": bad, "all_finite": manifest.get("all_finite"), "teacher_unchanged": teacher.get("unchanged"),
            "ok": bool(manifest["shards"]) and not bad and bool(manifest.get("all_finite")) and teacher.get("unchanged", True) is not False}


def teacher_identity(paths: Sequence[str | Path], mode: str) -> dict[str, Any]:
    out = {}
    for p in paths:
        st = os.stat(p)
        entry = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
        if mode == "full":
            started = time.time()
            entry["sha256"] = sha256_file(p)
            entry["hash_seconds"] = round(time.time() - started, 1)
        out[str(p)] = entry
    return out


def identity_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if set(before) != set(after):
        return False
    for path, b in before.items():
        a = after[path]
        for key in ("bytes", "mtime_ns", "inode", "sha256"):
            if key in b and key in a and b[key] != a[key]:
                return False
    return True


# ======================================================================================
# registry manifest builder (CPU; chimera_omni.data.curated_shards)
# ======================================================================================


def is_device_error(exc: BaseException) -> bool:
    """CUDA / device failures are environment faults, never a reason to drop one item."""

    name = type(exc).__name__
    text = str(exc)
    return name in ("OutOfMemoryError", "AcceleratorError", "CudaError") or "CUDA" in text or "cuda" in text or "out of memory" in text


ENVIRONMENT_ERRNOS = frozenset(getattr(errno, name) for name in (
    "ENOENT", "EIO", "ESTALE", "EACCES", "EPERM", "ENOTCONN", "ETIMEDOUT", "EHOSTDOWN", "EHOSTUNREACH", "ENETDOWN",
    "ENETUNREACH", "ECONNRESET", "ECONNABORTED", "ENXIO", "ENODEV", "ENOMEM", "ENOSPC", "EMFILE", "ENFILE", "EAGAIN",
    "EINTR", "EBUSY", "EISDIR", "ENOTDIR", "EROFS", "EDQUOT", "EREMOTEIO", "ENOLINK", "ECOMM") if hasattr(errno, name))


def is_environment_error(exc: BaseException) -> bool:
    """A mount, NAS, permission, host-memory or descriptor fault: never a verdict on the media itself.
    An OSError WITHOUT a POSIX errno (PIL's UnidentifiedImageError / 'image file is truncated', PyAV's
    InvalidDataError with its AVERROR code) is a decode verdict."""

    if isinstance(exc, (MemoryError, FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError,
                        ConnectionError, TimeoutError)):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in ENVIRONMENT_ERRNOS


def decode_or_drop(fn: Callable[[], Any], meta: dict[str, Any], dropped: list[dict[str, Any]]) -> Any:
    """Run one item's CPU media decode; an affirmative decode verdict drops that item (recorded), a device or
    environment failure re-raises (the run dies; a rerun resumes from the last sealed shard)."""

    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - classified below
        if is_device_error(exc) or is_environment_error(exc):
            raise
        meta["dropped"] = f"{type(exc).__name__}: {exc}"
        dropped.append(meta)
        log(f"item {meta.get('id')}: dropped ({meta['dropped']})")
        return None


def check_drop_fraction(dropped: int, total: int, limit: float) -> dict[str, Any]:
    fraction = (dropped / total) if total else 1.0
    if fraction > limit:
        raise SystemExit(f"{dropped} of {total} items dropped ({fraction:.0%} > --max-drop-fraction {limit:.0%}): refusing to seal")
    return {"dropped": dropped, "total": total, "fraction": fraction, "limit": limit}


def media_kind_of(path: str, kind: str) -> str | None:
    low = path.lower()
    if low.endswith(VIDEO_SUFFIXES) or kind.startswith("video"):
        return "video"
    if low.endswith(IMAGE_SUFFIXES) or kind.startswith("image"):
        return "image"
    return None


def run_manifest(args: argparse.Namespace) -> int:
    src = os.environ.get("CHIMERA_OMNI_SRC")
    if not src:
        raise SystemExit("set CHIMERA_OMNI_SRC to the chimera-omni tree (its src/ is imported)")
    sys.path.insert(0, str(Path(src) / "src") if (Path(src) / "src").is_dir() else src)
    from chimera_omni.data import curated_shards as cs  # noqa: E402

    want = args.route
    captions: dict[str, str] = {}
    if args.captions:
        for line in Path(args.captions).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                captions[str(row["sha256"])] = str(row["caption"])
    bank = bank_strings(args.exclude_bank or [])
    counts = {"rows_scanned": 0, "route_rows": 0, "no_media_sidecar": 0, "missing_file": 0, "excluded_bank": 0,
              "excluded_bank_media": 0, "no_caption": 0, "duplicate_media": 0, "kept_candidates": 0}
    candidates: list[dict[str, Any]] = []
    seen_media: set[str] = set()
    banned_media: set[str] = set()
    for entry in cs.iter_master_entries(args.pointer):
        for shard in cs.iter_shard_specs(entry):
            for row, line_number in cs.iter_rows_for_shard(shard):
                counts["rows_scanned"] += 1
                rec = cs.normalize_row(row, source_dataset=str(entry.get("dataset_id") or entry.get("name") or ""),
                                       training_status=str(entry.get("training_status", "")),
                                       shard_path=str(shard.get("path", "")), line_number=line_number)
                if rec is None or rec.route != want:
                    continue
                counts["route_rows"] += 1
                media = [(p, k) for p, k in zip(rec.sidecar_paths, rec.sidecar_kinds + ("",) * len(rec.sidecar_paths))
                         if media_kind_of(p, k) == want]
                if not media:
                    counts["no_media_sidecar"] += 1
                    continue
                path = media[0][0]
                if not os.path.isfile(path):
                    counts["missing_file"] += 1
                    continue
                if contaminated(rec.instruction, bank) or contaminated(rec.response, bank):
                    # the screen comes first and bans the MEDIA too: no other row may bring a bank item's clip back
                    counts["excluded_bank"] += 1
                    banned_media.add(path)
                    continue
                if path in seen_media:
                    counts["duplicate_media"] += 1
                    continue
                prompt_source = args.prompt_from
                if args.prompt_from == "captions":
                    media_sha = sha256_file(path)
                    prompt = captions.get(media_sha)
                    if not prompt:
                        counts["no_caption"] += 1
                        continue
                else:
                    media_sha = None
                    prompt = rec.instruction if args.prompt_from == "instruction" else rec.response
                    for prefix in ("user:", "assistant:"):
                        if prompt.lower().startswith(prefix):
                            prompt = prompt[len(prefix):].strip()
                if contaminated(prompt, bank):
                    counts["excluded_bank"] += 1
                    banned_media.add(path)
                    continue
                seen_media.add(path)
                candidates.append({"id": f"{want}-{rec.row_sha256[:16]}", "prompt": prompt, want: path,
                                   "prompt_source": prompt_source, "record_id": rec.record_id,
                                   "source_dataset": rec.source_dataset, "row_sha256": rec.row_sha256,
                                   **({"sha256": media_sha} if media_sha else {})})
    before_ban = len(candidates)
    candidates = [c for c in candidates if c[want] not in banned_media]
    counts["excluded_bank_media"] = before_ban - len(candidates)
    counts["kept_candidates"] = len(candidates)
    rng = random.Random(int(args.seed))
    rng.shuffle(candidates)
    chosen = candidates[: int(args.count)] if args.count > 0 else candidates
    held_every = int(args.held_every)
    for k, item in enumerate(chosen):
        item["split"] = "held" if held_every > 0 and k % held_every == held_every - 1 else "train"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for item in chosen:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, out)
    deviations = []
    if args.prompt_from != "captions":
        deviations.append({"name": "prompt_not_a_caption", "what": f"prompt = the registry row's {args.prompt_from} text",
                           "why_it_matters": "the 2026-10-12 sample of registry media rows carries task text ('Describe ... this real video "
                                             "artifact') and artifact descriptors, not captions of the media; a text-to-AV teacher "
                                             "conditioned on it is off its trained prompt distribution"})
    receipt = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "route": want, "pointer": str(args.pointer),
               "counts": counts, "chosen": len(chosen), "seed": int(args.seed), "out": str(out), "out_sha256": sha256_file(out),
               "exclude_bank": [str(p) for p in (args.exclude_bank or [])], "bank_strings": len(bank),
               "prompt_from": args.prompt_from, "deviations": deviations}
    write_json_atomic(out.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, indent=1))
    return 0


# ======================================================================================
# comfy runtime (container only)
# ======================================================================================


_BOOT: dict[str, Any] = {}


def release_gpu() -> None:
    """Drop every comfy-loaded model and return cached blocks to the driver (between runs in one process)."""

    import gc

    if "comfy.model_management" not in sys.modules:
        return
    import comfy.model_management as mm

    gc.collect()
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache(True)
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def comfy_boot(device: str, *, vram_guard_mib: int, expect_device_name: str | None, guard_wait: float = 0.0) -> dict[str, Any]:
    """Guarded boot; with guard_wait > 0 a busy card is re-read every 15 s for up to guard_wait seconds before the
    refusal (shared test cards), and the waited seconds are recorded."""

    started = time.time()
    while True:
        try:
            info = _comfy_boot_once(device, vram_guard_mib=vram_guard_mib, expect_device_name=expect_device_name)
            info["guard_waited_seconds"] = round(time.time() - started, 1)
            return info
        except _BusyCard as busy:
            if time.time() - started + 15.0 > guard_wait:
                raise SystemExit(str(busy)) from None
            log(f"{busy}; re-reading in 15 s (guard-wait {guard_wait:.0f} s)")
            time.sleep(15.0)


class _BusyCard(Exception):
    pass


def _comfy_boot_once(device: str, *, vram_guard_mib: int, expect_device_name: str | None) -> dict[str, Any]:
    """The VRAM guard refuses a card on which OTHER processes hold more than `vram_guard_mib`. On the first boot of a
    process that is the whole device reading; on a later run in the same process (tests) this process's own CUDA
    context (read at first boot) and its caching-allocator reservation are subtracted first."""

    torch = _torch()
    if _BOOT.get("device_requested") == device:
        info = dict(_BOOT)
        if device != "cpu" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            used = (total - free) >> 20
            own = int(_BOOT.get("context_mib", 0)) + (int(torch.cuda.memory_reserved(0)) >> 20)
            others = used - own
            info.update({"used_mib_at_start": used, "own_mib": own, "others_mib": others,
                         "guard": f"others <= {vram_guard_mib} MiB (own context + reservation subtracted)"})
            if others > vram_guard_mib:
                raise _BusyCard(f"other processes hold ~{others} MiB on the device (> {vram_guard_mib} MiB): refusing to start")
        return info
    for root in (os.environ.get("COMFYUI_ROOT", ""), *COMFY_ROOTS):
        if root and os.path.isdir(os.path.join(root, "comfy")) and root not in sys.path:
            sys.path.insert(0, root)
    import comfy.options  # noqa: F401
    from comfy.cli_args import args as comfy_args

    info: dict[str, Any] = {"torch": torch.__version__, "device_requested": device}
    if device == "cpu":
        comfy_args.cpu = True
    else:
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device visible: start the container with --gpus device=<UUID> -e CUDA_VISIBLE_DEVICES=0")
        props = torch.cuda.get_device_properties(0)
        arch = torch.cuda.get_arch_list()
        cc = f"sm_{props.major}{props.minor}"
        compatible = any(a == cc or (a.startswith(f"sm_{props.major}") and int(a[3:]) <= props.major * 10 + props.minor)
                         for a in arch if a.startswith("sm_") and a[3:].isdigit())
        free, total = torch.cuda.mem_get_info(0)
        used_mib = (total - free) >> 20
        info.update({"device_name": props.name, "capability": cc, "arch_list": arch, "arch_compatible": compatible,
                     "total_mib": total >> 20, "used_mib_at_start": used_mib,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES")})
        if not compatible:
            raise SystemExit(f"torch {torch.__version__} arch list {arch} has no kernel image for {props.name} ({cc})")
        if expect_device_name and props.name != expect_device_name:
            raise SystemExit(f"device 0 is {props.name!r}, expected {expect_device_name!r}")
        if used_mib > vram_guard_mib:
            raise _BusyCard(f"device already has {used_mib} MiB in use (> {vram_guard_mib} MiB): refusing to start")
        info["context_mib"] = used_mib  # our fresh context (+ any small foreign use the guard admitted)
    import comfy.model_management as mm
    import comfy.sd  # noqa: F401
    import comfy.utils  # noqa: F401

    dev = mm.get_torch_device()
    version = "unknown"
    for root in sys.path:
        vf = Path(root) / "comfyui_version.py"
        if vf.is_file():
            m = re.search(r'__version__\s*=\s*"([^"]+)"', vf.read_text())
            version = m.group(1) if m else version
            break
    info.update({"comfy_version": version, "torch_device": str(dev),
                 "should_use_bf16": bool(mm.should_use_bf16(dev)),
                 "should_use_fp16": bool(mm.should_use_fp16(dev, prioritize_performance=False)),
                 "vram_state": str(getattr(mm, "vram_state", None)), "guard": f"<= {vram_guard_mib} MiB at first boot"})
    _BOOT.clear()
    _BOOT.update(info)
    return info


def cuda_peak() -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info(0)
    return {"max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)), "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "device_used_bytes_now": int(total - free), "total_bytes": int(total)}


def load_dit(path: str | Path, weight_dtype: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    import comfy.model_management as mm
    import comfy.sd

    options: dict[str, Any] = {}
    if weight_dtype != "default":
        options["dtype"] = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
                            "fp8_e4m3fn": torch.float8_e4m3fn}[weight_dtype]
    started = time.time()
    patcher = comfy.sd.load_diffusion_model(str(path), model_options=options)
    model = patcher.model
    mm.load_models_gpu([patcher])
    info = {
        "path": str(path), "bytes": Path(path).stat().st_size, "weight_dtype_arg": weight_dtype,
        "model_class": type(model).__name__, "diffusion_model_class": type(model.diffusion_model).__name__,
        "weight_dtype": str(model.get_dtype()), "manual_cast_dtype": str(model.manual_cast_dtype),
        "inference_dtype": str(model.get_dtype_inference()), "model_size_bytes": int(patcher.model_size()),
        "loaded_size_bytes": int(patcher.loaded_size()) if hasattr(patcher, "loaded_size") else None,
        "sampling_shift": float(getattr(model.model_sampling, "shift", float("nan"))),
        "sampling_class": type(model.model_sampling).__name__, "load_seconds": round(time.time() - started, 2),
    }
    return patcher, info


def load_te(path: str | Path, clip_type: str, load_device: str) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    import comfy.model_management as mm
    import comfy.sd

    options: dict[str, Any] = {}
    if load_device == "gpu":
        options["load_device"] = mm.get_torch_device()
    elif load_device == "cpu":
        options["load_device"] = torch.device("cpu")
    started = time.time()
    clip = comfy.sd.load_clip(ckpt_paths=[str(path)], clip_type=getattr(comfy.sd.CLIPType, clip_type), model_options=options)
    try:
        model_dtype = str(clip.patcher.model_dtype())
    except Exception as exc:  # noqa: BLE001 - recorded, the signature still pins device + stack
        model_dtype = f"unknown ({type(exc).__name__})"
    return clip, {"path": str(path), "bytes": Path(path).stat().st_size, "clip_type": clip_type,
                  "load_device": str(clip.patcher.load_device), "offload_device": str(clip.patcher.offload_device),
                  "model_dtype": model_dtype, "load_seconds": round(time.time() - started, 2)}


def load_vae(path: str | Path) -> tuple[Any, dict[str, Any]]:
    import comfy.sd
    import comfy.utils

    started = time.time()
    sd, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae, {"path": str(path), "bytes": Path(path).stat().st_size, "vae_dtype": str(vae.vae_dtype),
                 "device": str(vae.device), "first_stage_model": type(vae.first_stage_model).__name__,
                 "load_seconds": round(time.time() - started, 2)}


def encode_text(clip: Any, text: str) -> tuple[Any, dict[str, Any]]:
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
    if len(cond) != 1:
        raise SystemExit(f"text encoder returned {len(cond)} scheduled conds (hooks?); expected 1")
    tensor, extra = cond[0][0], dict(cond[0][1])
    return tensor, extra


class BlockTap:
    """Pre/post hooks on diffusion_model.blocks[i] (the H3 block adds IN PLACE, so x_in is cloned in a PRE hook).
    `rows` (set before each forward) selects which packed rows to keep; `active` gates recording."""

    def __init__(self, dit: Any, block_ids: Sequence[int], store_dtype: Any):
        self.dit = dit
        self.block_ids = [int(b) for b in block_ids]
        self.store_dtype = store_dtype
        self.rows: Any = None
        self.active = False
        self.records: list[dict[int, dict[str, Any]]] = []
        self.handles: list[Any] = []

    def __enter__(self) -> "BlockTap":
        for index in self.block_ids:
            block = self.dit.blocks[index]
            self.handles.append(block.register_forward_pre_hook(self._pre(index)))
            self.handles.append(block.register_forward_hook(self._post(index)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def start_forward(self, rows: Any) -> None:
        self.rows = rows
        self.records.append({})

    def _keep(self, x: Any) -> Any:
        x = x.detach()
        if x.dim() == 3 and x.shape[0] == 1:
            x = x[0]
        sel = x.index_select(0, self.rows.to(x.device)) if self.rows is not None else x
        return {"rows_total": int(x.shape[0]), "data": sel.to(device="cpu", dtype=self.store_dtype).clone()}

    def _pre(self, index: int):
        def hook(module, args):
            if self.active and self.records:
                self.records[-1].setdefault(index, {})["x_in"] = self._keep(args[0])
        return hook

    def _post(self, index: int):
        def hook(module, args, output):
            if self.active and self.records:
                out = output[0] if isinstance(output, (tuple, list)) else output
                self.records[-1].setdefault(index, {})["x_out"] = self._keep(out)
        return hook


def guided_denoise(patcher: Any, positive: list, *, raw_streams: Sequence[Any], x_t_list: Sequence[Sequence[Any]],
                   sigmas: Sequence[float], seed: int, first_noise: Sequence[Any],
                   before_forward: Callable[[int], None] | None = None) -> dict[str, Any]:
    """Run ComfyUI's own CFGGuider (cfg 1, ONE branch = `positive`) with a sampler function that evaluates the
    wrapped model at OUR x_t for every sigma. `raw_streams` are the RAW VAE latents (comfy applies
    process_latent_in itself); `x_t_list` are model-space. Returns the per-sigma denoised streams plus comfy's
    own noised x for sigma[0] (noise_scaling(sigma0, first_noise, process_latent_in(raw)): the parity reading)."""

    torch = _torch()
    import comfy.nested_tensor
    import comfy.samplers
    import comfy.utils

    nested = len(raw_streams) > 1
    latent = comfy.nested_tensor.NestedTensor([t.float() for t in raw_streams]) if nested else raw_streams[0].float()
    noise = comfy.nested_tensor.NestedTensor([t.float() for t in first_noise]) if nested else first_noise[0].float()
    shapes = [tuple(t.shape) for t in raw_streams]
    flats = []
    for streams in x_t_list:
        flat, _ = comfy.utils.pack_latents([t.float() for t in streams]) if nested else (streams[0].float(), None)
        flats.append(flat)
    record: dict[str, Any] = {"denoised": [], "comfy_x0": None}

    def sampler_fn(model, x, sig, extra_args=None, callback=None, disable=None, **kwargs):
        record["comfy_x0"] = x.detach().float().cpu().clone()
        s_in = x.new_ones([x.shape[0]])
        for j, (flat, sigma) in enumerate(zip(flats, sigmas)):
            if before_forward is not None:
                before_forward(j)
            den = model(flat.to(device=x.device, dtype=x.dtype), torch.tensor(float(sigma), device=x.device) * s_in, **extra_args)
            record["denoised"].append(den.detach().float().cpu().clone())
        return x

    guider = comfy.samplers.CFGGuider(patcher)
    guider.set_conds(positive, positive)
    guider.set_cfg(1.0)
    sampler = comfy.samplers.KSAMPLER(sampler_fn)
    sigma_tensor = torch.tensor(list(float(s) for s in sigmas) + [0.0], dtype=torch.float32)
    guider.sample(noise, latent, sampler, sigma_tensor, seed=int(seed), disable_pbar=True)
    out = {"denoised": [], "comfy_x0": None}
    for den in record["denoised"]:
        out["denoised"].append(comfy.utils.unpack_latents(den, shapes) if nested else [den])
    if record["comfy_x0"] is not None:
        out["comfy_x0"] = comfy.utils.unpack_latents(record["comfy_x0"], shapes) if nested else [record["comfy_x0"]]
    return out


def model_space_x0(patcher: Any, raw_streams: Sequence[Any]) -> list[Any]:
    """What CFGGuider.inner_sample does to latent_image: process_latent_in on the (packed, for H3) latent."""

    import comfy.utils

    model = patcher.model
    if len(raw_streams) > 1:
        flat, shapes = comfy.utils.pack_latents([t.float() for t in raw_streams])
        return [t.float() for t in comfy.utils.unpack_latents(model.process_latent_in(flat), shapes)]
    return [model.process_latent_in(raw_streams[0].float()).float()]


def direct_velocity(patcher: Any, teacher: str, x_t: Sequence[Any], sigma: float, cond: Any, extra: dict[str, Any]) -> list[Any]:
    """Parity arm: the diffusion model called directly (no guider, no sampler, no CFG wrapper, no pack/unpack).
    Returns the model output in the comfy flat convention (what calculate_denoised consumes)."""

    torch = _torch()
    import comfy.model_management as mm

    mm.load_models_gpu([patcher])
    model = patcher.model
    dev = patcher.load_device
    dtype = model.get_dtype_inference()
    sig = torch.tensor([float(sigma)], device=dev)
    t = model.model_sampling.timestep(sig).float()
    with torch.no_grad():
        if teacher == "h3":
            ctx = model.diffusion_model.preprocess_text_embeds(cond.to(device=dev, dtype=dtype))
            payload = {"seed": 0}
            if extra.get("minimax_token_tags") is not None:
                payload["text_token_tags"] = extra["minimax_token_tags"]
            out = model.diffusion_model(x=[x_t[0].to(dev, dtype), x_t[1].to(dev, dtype)], timestep=t, context=ctx,
                                        transformer_options={}, minimax_payload=payload)
            return [o.float().cpu() for o in out]
        kwargs = {}
        if extra.get("attention_mask") is not None:
            kwargs["attention_mask"] = extra["attention_mask"].to(dev)
        out = model.diffusion_model(x_t[0].to(dev, dtype), t, context=cond.to(device=dev, dtype=dtype), transformer_options={}, **kwargs)
        return [out.float().cpu()]


def rel_l2(a: Any, b: Any) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ======================================================================================
# media decoding (container: PyAV + PIL)
# ======================================================================================


def resize_cover(frames: Any, width: int, height: int) -> Any:
    """[T, H, W, 3] float 0..1 -> [T, height, width, 3]: bilinear (antialiased) cover scale, centre crop."""

    torch = _torch()
    t, h, w, _ = frames.shape
    rw, rh, left, top = cover_crop_box(w, h, width, height)
    x = frames.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False, antialias=True)
    x = x[:, :, top: top + height, left: left + width]
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0).contiguous()


def read_video(path: str, *, frames: int, fps: float, width: int, height: int, start: float = 0.0) -> tuple[Any, dict[str, Any]]:
    import av
    import numpy as np

    torch = _torch()
    times: list[float] = []
    images: list[Any] = []
    need_until = start + (frames - 1) / fps
    with av.open(path) as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if times and t <= times[-1]:
                continue
            times.append(t)
            images.append(frame.to_ndarray(format="rgb24"))
            if t >= need_until + 1.0 / fps:
                break
    if not times:
        raise ValueError("no decodable video frames")
    if times[-1] + 1e-6 < need_until:
        raise ValueError(f"clip too short: last frame at {times[-1]:.3f}s, need {need_until:.3f}s")
    base = times[0]
    rel = [t - base for t in times]
    picks = pick_frames(rel, frame_times_for_target(frames, fps, start))
    stack = torch.from_numpy(np.stack([images[i] for i in picks])).float() / 255.0
    return resize_cover(stack, width, height), {"source_fps": src_fps, "source_frames_read": len(times),
                                                "source_size": [int(images[0].shape[1]), int(images[0].shape[0])],
                                                "picked_source_frames": [picks[0], picks[-1]], "start_seconds": start}


def read_audio(path: str, *, seconds: float, start: float = 0.0, rate: int = H3_AUDIO_SAMPLE_RATE) -> tuple[Any, dict[str, Any]]:
    import av
    import numpy as np

    torch = _torch()
    need = int(round(seconds * rate))
    with av.open(path) as container:
        if not container.streams.audio:
            return torch.zeros(1, 2, need), {"audio": "silence", "reason": "no audio stream"}
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        return torch.zeros(1, 2, need), {"audio": "silence", "reason": "audio stream decoded no samples"}
    wave = torch.from_numpy(np.concatenate(chunks, axis=1)).float()
    first = int(round(start * rate))
    wave = wave[:, first: first + need]
    padded = need - int(wave.shape[1])
    if padded > 0:
        wave = torch.nn.functional.pad(wave, (0, padded))
    return wave.unsqueeze(0), {"audio": "decoded", "padded_samples": max(0, padded), "rate": rate}


def read_image(path: str, *, width: int, height: int) -> Any:
    from PIL import Image
    import numpy as np

    torch = _torch()
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    x = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
    return resize_cover(x, width, height)


# ======================================================================================
# subcommands: encode
# ======================================================================================


def common_runtime(args: argparse.Namespace) -> dict[str, Any]:
    info = comfy_boot(args.device, vram_guard_mib=int(args.vram_guard_mib), expect_device_name=args.expect_device_name,
                      guard_wait=float(getattr(args, "guard_wait", 0.0) or 0.0))
    log(f"runtime: {json.dumps(info, default=str)}")
    return info


def finish(directory: Path, *, teacher_paths: Sequence[Path], before: dict[str, Any], args: argparse.Namespace,
           body: dict[str, Any]) -> dict[str, Any]:
    after = teacher_identity(teacher_paths, args.teacher_digest)
    unchanged = identity_unchanged(before, after)
    body["teacher_integrity"] = {"mode": args.teacher_digest, "before": before, "after": after, "unchanged": unchanged}
    if not unchanged:
        seal_manifest(directory, body)
        raise SystemExit(f"teacher files changed during the run: {directory}/manifest.json records before/after")
    sha = seal_manifest(directory, body)
    result = {"cache": str(directory), "manifest_sha256": sha, **verify_cache(directory)}
    log(f"sealed {directory} manifest {sha[:16]} shards {result['shards']} ok {result['ok']}")
    print(json.dumps(result, indent=1))
    return result


def run_h3_encode(args: argparse.Namespace) -> int:
    torch = _torch()
    items = read_items(args.items, media_key="video")
    if args.max_items:
        items = items[: int(args.max_items)]
    geo = h3_latent_shapes(args.width, args.height, args.length)
    corpus = corpus_identity(args.items)
    directory = cache_dir(args.cache_root, "h3-vae" + (f"-{args.cache_tag}" if args.cache_tag else ""), corpus["sha16"])
    plan = {"kind": "h3-encode", "corpus_sha256": corpus["sha256"], "items": len(items), "width": args.width,
            "height": args.height, "length": args.length, "geometry": geo, "start_seconds": args.start_seconds,
            "video_vae": str(args.video_vae), "audio_vae": str(args.audio_vae), "items_per_shard": args.items_per_shard}
    claim = claim_cache(directory, plan)
    runtime = common_runtime(args)
    ShardStore(directory, runtime_signature(runtime)).check_resume(runtime_only=True)  # before the teachers load
    teachers = [Path(args.video_vae), Path(args.audio_vae)]
    before = teacher_identity(teachers, args.teacher_digest)
    vvae, vinfo = load_vae(args.video_vae)
    avae, ainfo = load_vae(args.audio_vae)
    store = ShardStore(directory, runtime_signature(runtime, {"video_vae_dtype": vinfo["vae_dtype"], "video_vae_device": vinfo["device"],
                                                              "audio_vae_dtype": ainfo["vae_dtype"], "audio_vae_device": ainfo["device"]}))
    store.check_resume()
    timing: list[float] = []
    n_shards = math.ceil(len(items) / args.items_per_shard)
    for s in range(n_shards):
        group = items[s * args.items_per_shard: (s + 1) * args.items_per_shard]
        if store.done(s):
            log(f"shard {s}: done, skipped")
            continue
        vids, auds, metas = [], [], []
        dropped: list[dict[str, Any]] = []  # this shard's drops; the cache-wide count is re-derived at seal time
        for k, item in enumerate(group):
            index = s * args.items_per_shard + k
            path = item["video"]
            meta: dict[str, Any] = {"index": index, "id": item["id"], "video": path, "prompt_sha256": sha256_text(item["prompt"])}
            meta["video_sha256"] = sha256_file(path)  # outside the drop guard: an unreadable mount is fatal, not a drop

            def decode(path=path, item=item, meta=meta):
                if item.get("sha256") and item["sha256"] != meta["video_sha256"]:
                    raise ValueError(f"sha256 {meta['video_sha256'][:16]} != manifest {item['sha256'][:16]}")
                frames, vmeta = read_video(path, frames=geo["frame_count"], fps=H3_FPS, width=args.width, height=args.height,
                                           start=args.start_seconds)
                wave, ameta = read_audio(item.get("audio") or path, seconds=geo["seconds"], start=args.start_seconds)
                return frames, vmeta, wave, ameta

            started = time.time()
            decoded = decode_or_drop(decode, meta, dropped)
            if decoded is None:
                continue
            frames, vmeta, wave, ameta = decoded
            with torch.no_grad():  # a VAE / device failure here is fatal, never a dropped clip
                zv = vvae.encode(frames).float().cpu()
                za = avae.encode(wave[:1].movedim(1, -1)).float().cpu()
            timing.append(time.time() - started)
            if tuple(zv.shape) != geo["video"]:
                raise SystemExit(f"item {item['id']}: video latent {tuple(zv.shape)} != planned {geo['video']}")
            target_t = geo["audio"][-1]
            meta["audio_latent_frames_encoded"] = int(za.shape[-1])
            if za.shape[:3] != geo["audio"][:3]:
                raise SystemExit(f"item {item['id']}: audio latent {tuple(za.shape)} != planned {geo['audio']}")
            if za.shape[-1] >= target_t:
                za = za[..., :target_t]
            else:
                za = torch.nn.functional.pad(za, (0, target_t - za.shape[-1]))
                meta["audio_latent_padded"] = int(target_t - meta["audio_latent_frames_encoded"])
            meta.update({"video_meta": vmeta, "audio_meta": ameta})
            vids.append(zv[0])
            auds.append(za[0])
            metas.append(meta)
        if not metas:
            store.write_empty(s, dropped)
            continue
        store.write(s, {"x0_raw_video": torch.stack(vids), "x0_raw_audio": torch.stack(auds),
                        "item_index": torch.tensor([m["index"] for m in metas], dtype=torch.long)},
                    {"items": metas, "dropped": dropped})
        log(f"shard {s}/{n_shards}: {len(metas)} items, {sum(timing[-len(metas):]) / len(metas):.2f} s/item")
    agg = aggregate_drops(directory, len(items))
    drop = check_drop_fraction(len(agg["dropped"]), len(items), args.max_drop_fraction)
    body = {"kind": "h3-encode", "plan": plan, "plan_sha256": claim["plan_sha256"], "corpus": corpus, "runtime": runtime,
            "vae": {"video": vinfo, "audio": ainfo}, "dropped": agg["dropped"], "kept": agg["kept"], "drop_check": drop,
            "timing": {"items": len(timing), "seconds_per_item_mean": (sum(timing) / len(timing)) if timing else None},
            "latent_space": "raw VAE output (comfy LATENT samples); the velocity pass applies model.process_latent_in",
            "deviations": [{"name": "resample_sample_and_hold", "what": "source frames resampled to 24 fps by the latest frame at or before each target time"},
                           {"name": "cover_crop", "what": f"bilinear antialiased cover-resize and centre crop to {args.width}x{args.height}"}]}
    finish(directory, teacher_paths=teachers, before=before, args=args, body=body)
    return 0


def run_qimg_encode(args: argparse.Namespace) -> int:
    torch = _torch()
    items = read_items(args.items, media_key="image")
    if args.max_items:
        items = items[: int(args.max_items)]
    geo = qimg_latent_shape(args.width, args.height)
    corpus = corpus_identity(args.items)
    directory = cache_dir(args.cache_root, "qimg-vae" + (f"-{args.cache_tag}" if args.cache_tag else ""), corpus["sha16"])
    plan = {"kind": "qimg-encode", "corpus_sha256": corpus["sha256"], "items": len(items), "width": args.width,
            "height": args.height, "geometry": geo, "vae": str(args.vae), "items_per_shard": args.items_per_shard}
    claim = claim_cache(directory, plan)
    runtime = common_runtime(args)
    ShardStore(directory, runtime_signature(runtime)).check_resume(runtime_only=True)  # before the teacher loads
    teachers = [Path(args.vae)]
    before = teacher_identity(teachers, args.teacher_digest)
    vae, vinfo = load_vae(args.vae)
    store = ShardStore(directory, runtime_signature(runtime, {"vae_dtype": vinfo["vae_dtype"], "vae_device": vinfo["device"]}))
    store.check_resume()
    n_shards = math.ceil(len(items) / args.items_per_shard)
    for s in range(n_shards):
        group = items[s * args.items_per_shard: (s + 1) * args.items_per_shard]
        if store.done(s):
            log(f"shard {s}: done, skipped")
            continue
        lats, metas = [], []
        dropped: list[dict[str, Any]] = []  # this shard's drops; the cache-wide count is re-derived at seal time
        for k, item in enumerate(group):
            index = s * args.items_per_shard + k
            meta: dict[str, Any] = {"index": index, "id": item["id"], "image": item["image"], "prompt_sha256": sha256_text(item["prompt"])}
            meta["image_sha256"] = sha256_file(item["image"])  # outside the drop guard: an unreadable mount is fatal

            def decode(item=item, meta=meta):
                if item.get("sha256") and item["sha256"] != meta["image_sha256"]:
                    raise ValueError("sha256 differs from the manifest")
                return read_image(item["image"], width=args.width, height=args.height)

            pixels = decode_or_drop(decode, meta, dropped)
            if pixels is None:
                continue
            with torch.no_grad():  # fatal on failure
                z = vae.encode(pixels).float().cpu()
            if tuple(z.shape) != geo["latent"]:
                raise SystemExit(f"item {item['id']}: latent {tuple(z.shape)} != planned {geo['latent']}")
            lats.append(z[0])
            metas.append(meta)
        if not metas:
            store.write_empty(s, dropped)
            continue
        store.write(s, {"x0_raw": torch.stack(lats), "item_index": torch.tensor([m["index"] for m in metas], dtype=torch.long)},
                    {"items": metas, "dropped": dropped})
    agg = aggregate_drops(directory, len(items))
    drop = check_drop_fraction(len(agg["dropped"]), len(items), args.max_drop_fraction)
    body = {"kind": "qimg-encode", "plan": plan, "plan_sha256": claim["plan_sha256"], "corpus": corpus, "runtime": runtime,
            "vae": vinfo, "dropped": agg["dropped"], "kept": agg["kept"], "drop_check": drop,
            "latent_space": "raw VAE output; the velocity pass applies model.process_latent_in (Wan21 mean/std)"}
    finish(directory, teacher_paths=teachers, before=before, args=args, body=body)
    return 0


# ======================================================================================
# subcommands: text encoders
# ======================================================================================


def texts_for(items: Sequence[dict[str, Any]], *, negatives: bool, default_negative: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        for text in [item["prompt"]] + ([item.get("negative", default_negative)] if negatives else []):
            if text not in seen:
                seen.add(text)
                out.append(text)
    return out


def run_text(args: argparse.Namespace, *, teacher: str) -> int:
    torch = _torch()
    items = read_items(args.items, media_key="")
    if args.max_items:
        items = items[: int(args.max_items)]
    texts = texts_for(items, negatives=bool(args.with_negatives), default_negative=args.negative)
    corpus = corpus_identity(args.items)
    te_path = Path(args.te)
    name = f"{teacher}-te-{te_path.stem.lower()}" + (f"-{args.cache_tag}" if args.cache_tag else "")
    directory = cache_dir(args.cache_root, re.sub(r"[^a-z0-9._-]", "-", name), corpus["sha16"])
    clip_type = "MINIMAX" if teacher == "h3" else "QWEN_IMAGE"
    plan = {"kind": f"{teacher}-text", "corpus_sha256": corpus["sha256"], "texts": len(texts), "te": str(te_path),
            "clip_type": clip_type, "with_negatives": bool(args.with_negatives), "default_negative": args.negative,
            "texts_per_shard": args.texts_per_shard, "text_sha256": [sha256_text(t) for t in texts]}
    claim = claim_cache(directory, plan)
    runtime = common_runtime(args)
    ShardStore(directory, runtime_signature(runtime)).check_resume(runtime_only=True)  # before the encoder loads
    before = teacher_identity([te_path], args.teacher_digest)
    clip, cinfo = load_te(te_path, clip_type, args.te_load_device)
    store = ShardStore(directory, runtime_signature(runtime, {"te_load_device": cinfo["load_device"], "te_model_dtype": cinfo["model_dtype"]}))
    store.check_resume()
    timing: list[float] = []
    n_shards = math.ceil(len(texts) / args.texts_per_shard)
    for s in range(n_shards):
        group = texts[s * args.texts_per_shard: (s + 1) * args.texts_per_shard]
        if store.done(s):
            continue
        tensors: dict[str, Any] = {}
        metas = []
        for k, text in enumerate(group):
            key = sha256_text(text)
            started = time.time()
            with torch.no_grad():
                cond, extra = encode_text(clip, text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing.append(time.time() - started)
            tensors[f"{key}.cond"] = cond.detach().cpu()
            extras_json = {}
            for ek, ev in extra.items():
                if hasattr(ev, "shape"):
                    tensors[f"{key}.{ek}"] = ev.detach().cpu()
                elif ev is not None:
                    extras_json[ek] = ev
            if teacher == "h3" and f"{key}.minimax_token_tags" not in tensors:
                raise SystemExit("H3 text encoder returned no minimax_token_tags (not the minimax CLIP?)")
            metas.append({"text_sha256": key, "chars": len(text), "tokens": int(cond.shape[1]), "cond_shape": list(cond.shape),
                          "cond_dtype": str(cond.dtype), "extra_tensors": sorted(ek for ek in extra if hasattr(extra[ek], "shape")),
                          "extra_json": extras_json, "seconds": round(timing[-1], 3)})
        store.write(s, tensors, {"texts": metas})
        log(f"shard {s}/{n_shards}: {len(group)} texts, {sum(timing[-len(group):]) / len(group):.2f} s/text")
    index = {}
    for side in shard_sidecars(directory):
        meta = json.loads(side.read_text())
        for t in meta["texts"]:
            index[t["text_sha256"]] = {"file": meta["file"], "tokens": t["tokens"]}
    write_json_atomic(directory / "index.json", index)
    body = {"kind": f"{teacher}-text", "plan": plan, "plan_sha256": claim["plan_sha256"], "corpus": corpus, "runtime": runtime,
            "te": cinfo, "index_file": "index.json", "index_sha256": sha256_file(directory / "index.json"),
            "timing": {"texts": len(timing), "seconds_per_text_mean": (sum(timing) / len(timing)) if timing else None,
                       "peak": cuda_peak(), "rss": rss_block()},
            "keys": "<sha256(text)>.cond plus <sha256(text)>.<extra> (minimax_token_tags for H3, attention_mask if any)"}
    finish(directory, teacher_paths=[te_path], before=before, args=args, body=body)
    return 0


class TextCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if not manifest.get("teacher_integrity", {}).get("unchanged"):
            raise SystemExit(f"{directory}: text cache not sealed with an unchanged teacher")
        self.manifest_sha = sha256_file(self.directory / "manifest.json")
        self.index = json.loads((self.directory / "index.json").read_text())
        self.te = manifest.get("te", {})

    def get(self, text: str) -> tuple[Any, dict[str, Any]]:
        from safetensors import safe_open

        key = sha256_text(text)
        if key not in self.index:
            raise SystemExit(f"text {text[:60]!r} (sha {key[:16]}) is not in the text cache {self.directory}")
        extra: dict[str, Any] = {}
        with safe_open(str(self.directory / self.index[key]["file"]), "pt") as handle:
            cond = handle.get_tensor(f"{key}.cond")
            for name in handle.keys():
                if name.startswith(key + ".") and name != f"{key}.cond":
                    extra[name[len(key) + 1:]] = handle.get_tensor(name)
        return cond, extra


class EncodeCache:
    def __init__(self, directory: str | Path, keys: Sequence[str]):
        from safetensors import safe_open

        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if not manifest.get("teacher_integrity", {}).get("unchanged") or not manifest.get("all_finite"):
            raise SystemExit(f"{directory}: encode cache not sealed clean")
        self.manifest = manifest
        self.manifest_sha = sha256_file(self.directory / "manifest.json")
        self.items: dict[int, dict[str, Any]] = {}
        for shard in manifest["shards"]:
            meta = json.loads((self.directory / shard["file"]).with_suffix(".json").read_text())
            with safe_open(str(self.directory / shard["file"]), "pt") as handle:
                tensors = {k: handle.get_tensor(k) for k in keys}
            for row, m in enumerate(meta["items"]):
                self.items[int(m["index"])] = {"meta": m, **{k: tensors[k][row] for k in keys}}


# ======================================================================================
# subcommands: velocity
# ======================================================================================


def build_conditioning(cond: Any, extra: dict[str, Any]) -> list:
    return [[cond, dict(extra)]]


def run_velocity(args: argparse.Namespace, *, teacher: str) -> int:
    torch = _torch()
    items = read_items(args.items, media_key="")
    corpus = corpus_identity(args.items)
    enc_keys = ["x0_raw_video", "x0_raw_audio"] if teacher == "h3" else ["x0_raw"]
    enc = EncodeCache(args.encode_cache, enc_keys)
    if enc.manifest["corpus"]["sha256"] != corpus["sha256"]:
        raise SystemExit("encode cache was built from a different item manifest")
    present = sorted(enc.items)
    if args.max_items:
        present = present[: int(args.max_items)]
    texts = TextCache(args.te_cache)
    neg_needed = not math.isclose(args.cfg, 1.0)
    shift = float(args.shift)
    rows = sigma_plan(len(present), per_item=args.sigmas_per_item, mode=args.sigma_mode, shift=shift, seed=args.seed,
                      sigmas=args.sigmas)
    block_ids = [int(b) for b in args.block_pairs.split(",") if b.strip()] if (teacher == "h3" and args.block_pairs) else []
    dit_path = Path(args.dit)
    name = f"{teacher}-velocity-{dit_path.stem.lower()}" + (f"-{args.cache_tag}" if args.cache_tag else "")
    directory = cache_dir(args.cache_root, re.sub(r"[^a-z0-9._-]", "-", name), corpus["sha16"])
    first = enc.items[present[0]]
    plan = {"kind": f"{teacher}-velocity", "corpus_sha256": corpus["sha256"], "encode_cache_manifest": enc.manifest_sha,
            "te_cache_manifest": texts.manifest_sha, "items": present, "dit": str(dit_path), "weight_dtype": args.weight_dtype,
            "cfg": args.cfg, "negative_default": args.negative if neg_needed else None, "sigma_mode": args.sigma_mode,
            "sigmas_per_item": args.sigmas_per_item, "sigmas": args.sigmas, "shift": shift, "seed": args.seed,
            "targets": [[r["item"], r["j"], r["sigma"], r["seed"]] for r in rows], "block_pairs": block_ids,
            "block_rows": args.block_rows, "block_include_text": bool(args.block_include_text),
            "block_store_dtype": args.block_store_dtype, "items_per_shard": args.items_per_shard,
            "x0_shapes": {k: list(first[k].shape) for k in enc_keys}}
    claim = claim_cache(directory, plan)
    runtime = common_runtime(args)
    ShardStore(directory, runtime_signature(runtime)).check_resume(runtime_only=True)  # before a 20-40 GB teacher loads
    before = teacher_identity([dit_path], args.teacher_digest)
    patcher, dinfo = load_dit(dit_path, args.weight_dtype)
    if not math.isclose(dinfo["sampling_shift"], shift, rel_tol=1e-6) and args.sigma_mode == "stratified-shifted":
        if not args.shift_override:
            raise SystemExit(f"model sampling shift {dinfo['sampling_shift']} != --shift {shift}; pass --shift-override to "
                             "sample sigmas on a different schedule on purpose")
    log(f"loaded {dinfo['model_class']} weights {dinfo['weight_dtype']} inference {dinfo['inference_dtype']} "
        f"manual_cast {dinfo['manual_cast_dtype']} in {dinfo['load_seconds']} s")
    store_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.block_store_dtype]
    store = ShardStore(directory, runtime_signature(runtime, {k: dinfo[k] for k in ("weight_dtype", "inference_dtype", "manual_cast_dtype",
                                                                                     "model_class", "diffusion_model_class")}))
    store.check_resume()
    by_item: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_item.setdefault(r["item"], []).append(r)
    targets_done = 0
    n_shards = math.ceil(len(present) / args.items_per_shard)
    dit = patcher.model.diffusion_model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    with BlockTap(dit, block_ids, store_dtype) as tap:
        for s in range(n_shards):
            group = list(range(s * args.items_per_shard, min(len(present), (s + 1) * args.items_per_shard)))
            if store.done(s):
                targets_done += sum(len(by_item[g]) for g in group)
                log(f"shard {s}: done, skipped")
                continue
            acc: dict[str, list[Any]] = {}
            metas: list[dict[str, Any]] = []
            timings: list[dict[str, Any]] = []
            parity: list[dict[str, Any]] = []
            comfy_x_parity: list[dict[str, Any]] = []

            def put(key: str, value: Any) -> None:
                acc.setdefault(key, []).append(value)

            for g in group:
                entry = enc.items[present[g]]
                item = items[int(entry["meta"]["index"])]
                if item["id"] != entry["meta"]["id"]:
                    raise SystemExit(f"encode cache item {entry['meta']['id']} != manifest item {item['id']} at index {entry['meta']['index']}")
                raw = [entry[k].unsqueeze(0).float() for k in enc_keys]
                x0 = model_space_x0(patcher, raw)
                pos_cond, pos_extra = texts.get(item["prompt"])
                branches = [("cond", build_conditioning(pos_cond, pos_extra), pos_cond, pos_extra)]
                if neg_needed:
                    neg_cond, neg_extra = texts.get(item.get("negative", args.negative))
                    branches.append(("uncond", build_conditioning(neg_cond, neg_extra), neg_cond, neg_extra))
                plan_rows = by_item[g]
                sigmas = [r["sigma"] for r in plan_rows]
                x_t_list, noises_list, kinds = [], [], None
                for r in plan_rows:
                    shapes = [t.shape for t in x0]
                    noises = make_noise(shapes, r["seed"])
                    if teacher == "h3":
                        sa = time_shift_sigma(r["sigma"], H3_SHIFT_VIDEO, H3_SHIFT_AUDIO)
                        x_t_list.append([flow_interp(x0[0], noises[0], r["sigma"]), flow_interp(x0[1], noises[1], sa)])
                    else:
                        x_t_list.append([flow_interp(x0[0], noises[0], r["sigma"])])
                    noises_list.append(noises)
                if teacher == "h3" and block_ids:
                    video_tokens = math.prod(x0[0].shape[2:3]) * (x0[0].shape[3] // 2) * (x0[0].shape[4] // 2)
                    kinds = h3_packed_kinds(int(pos_cond.shape[1]), int(x0[1].shape[-1]), video_tokens)
                row_sel = {j: block_row_selection(kinds, args.block_rows, r["seed"], include_text=args.block_include_text)
                           for j, r in enumerate(plan_rows)} if kinds is not None else {}

                def before_forward(j: int) -> None:
                    if block_ids:
                        tap.start_forward(torch.tensor(row_sel[j], dtype=torch.long))

                outs: dict[str, Any] = {}
                for label, conditioning, cond_t, cond_x in branches:
                    tap.active = bool(block_ids) and label == "cond"
                    if label == "cond":
                        tap.records.clear()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started = time.time()
                    res = guided_denoise(patcher, conditioning, raw_streams=raw, x_t_list=x_t_list, sigmas=sigmas,
                                         seed=plan_rows[0]["seed"], first_noise=noises_list[0],
                                         before_forward=before_forward if label == "cond" else None)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    wall = time.time() - started
                    timings.append({"item": g, "branch": label, "forwards": len(sigmas), "seconds": wall,
                                    "seconds_per_forward": wall / len(sigmas)})
                    outs[label] = res["denoised"]
                    if label == "cond" and res["comfy_x0"] is not None:
                        # comfy's own noise_scaling of (latent, first noise) at sigma[0] vs our x_t (video / image stream)
                        comfy_x_parity.append({"item": g, "sigma": sigmas[0],
                                               "rel_l2_stream0": rel_l2(res["comfy_x0"][0], x_t_list[0][0]),
                                               "max_abs_stream0": float((res["comfy_x0"][0] - x_t_list[0][0]).abs().max())})
                tap.active = False
                for j, r in enumerate(plan_rows):
                    den_c = outs["cond"][j]
                    den = [cfg_combine(den_c[k], outs["uncond"][j][k] if neg_needed else None, args.cfg) for k in range(len(den_c))]
                    xt = x_t_list[j]
                    meta = {"target": targets_done, "item_index": int(entry["meta"]["index"]), "id": item["id"], "j": r["j"],
                            "sigma": r["sigma"], "u": r["u"], "seed": r["seed"], "noise_sha256": noise_sha(noises_list[j])}
                    if teacher == "h3":
                        tgt = h3_native_targets(x_video=xt[0], x_audio=xt[1], den_video=den[0], den_audio=den[1], sigma_v=r["sigma"])
                        put("v_video", tgt["v_video"][0])
                        put("v_audio", tgt["v_audio"][0])
                        put("x0hat_video", tgt["x0hat_video"][0])
                        put("x0hat_audio", tgt["x0hat_audio"][0])
                        put("sigma_audio", torch.tensor(tgt["sigma_audio"], dtype=torch.float32))
                        put("slope_audio", torch.tensor(tgt["slope_audio"], dtype=torch.float32))
                        if neg_needed:
                            for label in ("cond", "uncond"):
                                put(f"v_{label}_video", velocity_from_denoised(xt[0], outs[label][j][0], r["sigma"])[0])
                                put(f"v_{label}_audio", velocity_from_denoised(xt[1], outs[label][j][1], r["sigma"])[0])
                        for b in block_ids:
                            rec = tap.records[j].get(b, {}) if j < len(tap.records) else {}
                            if "x_in" not in rec or "x_out" not in rec:
                                raise SystemExit(f"block {b}: hooks recorded nothing for item {g} sigma {j}")
                            if rec["x_in"]["rows_total"] != len(kinds):
                                raise SystemExit(f"block {b}: {rec['x_in']['rows_total']} packed rows, layout expects {len(kinds)}")
                            put(f"blk{b}.x_in", rec["x_in"]["data"])
                            put(f"blk{b}.x_out", rec["x_out"]["data"])
                            put(f"blk{b}.rows", torch.tensor(row_sel[j], dtype=torch.long))
                            put(f"blk{b}.kind", torch.tensor([kinds[i] for i in row_sel[j]], dtype=torch.int8))
                    else:
                        put("v", velocity_from_denoised(xt[0], den[0], r["sigma"])[0])
                        put("x0hat", den[0][0].float())
                        if neg_needed:
                            for label in ("cond", "uncond"):
                                put(f"v_{label}", velocity_from_denoised(xt[0], outs[label][j][0], r["sigma"])[0])
                    put("sigma", torch.tensor(r["sigma"], dtype=torch.float32))
                    put("seed", torch.tensor(r["seed"], dtype=torch.long))
                    put("target_item", torch.tensor(int(entry["meta"]["index"]), dtype=torch.long))
                    # parity arm: the diffusion model called directly at the same x_t (cond branch)
                    if args.parity_every >= 0 and (targets_done == 0 or (args.parity_every > 0 and targets_done % args.parity_every == 0)):
                        direct = direct_velocity(patcher, teacher, xt, r["sigma"], pos_cond, pos_extra)
                        sampler_v = [velocity_from_denoised(xt[k], outs["cond"][j][k], r["sigma"]) for k in range(len(xt))]
                        reading = {"target": targets_done, "sigma": r["sigma"],
                                   "rel_l2": [rel_l2(sampler_v[k], direct[k]) for k in range(len(xt))],
                                   "max_abs": [float((sampler_v[k] - direct[k]).abs().max()) for k in range(len(xt))],
                                   "tolerance": args.parity_tolerance}
                        reading["pass"] = all(v <= args.parity_tolerance for v in reading["rel_l2"])
                        parity.append(reading)
                        log(f"parity target {targets_done}: rel_l2 {['%.3e' % v for v in reading['rel_l2']]} pass {reading['pass']}")
                        if not reading["pass"]:
                            raise SystemExit(f"parity arm failed at target {targets_done}: {reading}")
                    metas.append(meta)
                    targets_done += 1
                for k, name_x0 in enumerate(["x0_video", "x0_audio"] if teacher == "h3" else ["x0"]):
                    put(f"item.{name_x0}", x0[k][0])
                put("item.index", torch.tensor(int(entry["meta"]["index"]), dtype=torch.long))
            tensors = {k: torch.stack(v) for k, v in acc.items()}
            store.write(s, tensors, {"targets": metas, "items": [int(enc.items[present[g]]["meta"]["index"]) for g in group],
                                     "parity": parity, "timing": timings, "comfy_noise_scaling_parity": comfy_x_parity,
                                     "vram_peak_so_far": cuda_peak()})
            spf = sum(t["seconds"] for t in timings) / max(1, sum(t["forwards"] for t in timings))
            log(f"shard {s}/{n_shards}: {len(metas)} targets, {spf:.2f} s/forward, peak {cuda_peak().get('max_allocated_bytes', 0) >> 20} MiB")
    # evidence is aggregated from the shard sidecars, so a resumed run re-seals with every shard's record
    sides = [json.loads(p.read_text(encoding="utf-8")) for p in shard_sidecars(directory)]
    timings = [t for m in sides for t in m.get("timing", [])]
    parity = [r for m in sides for r in m.get("parity", [])]
    comfy_x_parity = [r for m in sides for r in m.get("comfy_noise_scaling_parity", [])]
    targets_done = sum(len(m.get("targets", [])) for m in sides)
    total_fw = sum(t["forwards"] for t in timings)
    body = {
        "kind": f"{teacher}-velocity", "plan": plan, "plan_sha256": claim["plan_sha256"], "corpus": corpus, "runtime": runtime,
        "dit": dinfo, "encode_cache": {"dir": str(enc.directory), "manifest_sha256": enc.manifest_sha},
        "te_cache": {"dir": str(texts.directory), "manifest_sha256": texts.manifest_sha, "te": texts.te},
        "targets": targets_done, "parity_arm": parity, "comfy_noise_scaling_parity": comfy_x_parity,
        "timing": {"forwards": total_fw, "seconds_total": sum(t["seconds"] for t in timings),
                   "seconds_per_forward_mean": (sum(t["seconds"] for t in timings) / total_fw) if total_fw else None,
                   "first": timings[:4]},
        "vram_peak": cuda_peak(), "vram_peak_by_shard": [m.get("vram_peak_so_far") for m in sides], "rss": rss_block(),
        "conventions": {
            "x_t": "(1 - sigma) * x0_model + sigma * noise per stream; H3 audio at sigma_audio = time_shift_sigma(sigma, 12, 3)",
            "v": "comfy CONST flat convention: v = (x_t - denoised) / sigma (float64), denoised from ComfyUI's CFGGuider/KSAMPLER path",
            "x0hat": "H3 video: denoised; H3 audio: x_a - sigma_a * v_audio / slope_audio; QIMG: denoised",
            "noise": "torch.Generator().manual_seed(seed); randn float32 per stream in order (video, audio); sha256 per target in shard json",
            "guidance": "cfg 1 = one positive branch (H3 BasicGuider); else uncond + cfg * (cond - uncond) on denoised in fp32",
            "targets_are": "predictions of THIS weight file under ComfyUI's compute dtype on ONE card class: runtime_signature (identical on every shard, checked at resume and seal) names the card, stack and dit dtypes",
        },
        "deviations": ([{"name": "fp8_teacher", "what": "Qwen-Image targets are FP8 E4M3 model predictions (manual_cast compute), not bf16-original"}]
                       if teacher == "qimg" else [])
        + ([{"name": "int8_teacher", "what": "H3 targets come from the int8 ConvRot file, not the bf16 DiT"}] if "int8" in dit_path.name else []),
    }
    finish(directory, teacher_paths=[dit_path], before=before, args=args, body=body)
    return 0


# ======================================================================================
# probe (S0 P4)
# ======================================================================================


def run_probe(args: argparse.Namespace) -> int:
    torch = _torch()
    runtime = common_runtime(args)
    receipt: dict[str, Any] = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": tool_sha256(),
                               "probe": "S0 P4", "label": args.label, "component": args.component, "teacher": args.teacher,
                               "runtime": runtime, "argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    if args.component == "dit":
        patcher, dinfo = load_dit(args.path, args.weight_dtype)
        receipt["load"] = dinfo
        receipt["vram_after_load"] = cuda_peak()
        if args.teacher == "h3":
            geo = h3_latent_shapes(args.width, args.height, args.length)
            shapes = [geo["video"], geo["audio"]]
        else:
            geo = qimg_latent_shape(args.width, args.height)
            shapes = [geo["latent"]]
        receipt["geometry"] = geo
        if args.te_cache and args.prompt:
            cond, extra = TextCache(args.te_cache).get(args.prompt)
            text_source = "te_cache"
        else:
            width = 5120 if args.teacher == "h3" else 3584
            cond = torch.randn(1, int(args.text_len), width, generator=torch.Generator().manual_seed(5)) * 0.5
            extra = {"minimax_token_tags": torch.ones(int(args.text_len), dtype=torch.long)} if args.teacher == "h3" else {}
            text_source = f"seeded gaussian [1, {args.text_len}, {width}] (timing only)"
        raw = [torch.randn(s, generator=torch.Generator().manual_seed(11)) * 0.5 for s in shapes]
        x0 = model_space_x0(patcher, raw)
        forwards = []
        for k in range(int(args.forwards)):
            sigma = [0.95, 0.75, 0.5, 0.25][k % 4]
            noise = make_noise([t.shape for t in x0], 1000 + k)
            xt = [flow_interp(x0[0], noise[0], sigma)] + ([flow_interp(x0[1], noise[1], time_shift_sigma(sigma, 12, 3))] if len(x0) > 1 else [])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.time()
            res = guided_denoise(patcher, build_conditioning(cond, extra), raw_streams=raw, x_t_list=[xt], sigmas=[sigma],
                                 seed=k, first_noise=noise)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.time() - started
            finite = all(bool(torch.isfinite(t).all()) for t in res["denoised"][0])
            forwards.append({"k": k, "sigma": sigma, "seconds": round(wall, 3), "finite": finite})
            log(f"forward {k}: {wall:.2f} s finite {finite}")
            if not finite:
                receipt["non_finite"] = True
                break
        steady = forwards[1:] if len(forwards) > 1 else forwards
        receipt.update({"text_source": text_source, "forwards": forwards,
                        "seconds_per_forward_steady": (sum(f["seconds"] for f in steady) / len(steady)) if steady else None,
                        "note": "one guided branch per forward (cfg 1); a cfg != 1 target costs 2 forwards"})
    elif args.component == "te":
        clip, cinfo = load_te(args.path, "MINIMAX" if args.teacher == "h3" else "QWEN_IMAGE", args.te_load_device)
        receipt["load"] = cinfo
        prompts = [args.prompt or "A red marble rolls across a wooden table and taps a silver bell."] * int(args.forwards)
        seconds = []
        for k, p in enumerate(prompts):
            started = time.time()
            with torch.no_grad():
                cond, extra = encode_text(clip, p)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds.append(time.time() - started)
        receipt.update({"texts": len(prompts), "seconds": [round(s, 3) for s in seconds], "cond_shape": list(cond.shape),
                        "seconds_per_text_steady": (sum(seconds[1:]) / len(seconds[1:])) if len(seconds) > 1 else seconds[0]})
    elif args.component == "vae":
        vae, vinfo = load_vae(args.path)
        receipt["load"] = vinfo
        if args.teacher == "h3":
            geo = h3_latent_shapes(args.width, args.height, args.length)
            pixels = torch.rand(geo["frame_count"], args.height, args.width, 3, generator=torch.Generator().manual_seed(3))
        else:
            pixels = torch.rand(1, args.height, args.width, 3, generator=torch.Generator().manual_seed(3))
        seconds = []
        for _ in range(int(args.forwards)):
            started = time.time()
            with torch.no_grad():
                z = vae.encode(pixels)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds.append(time.time() - started)
        receipt.update({"latent_shape": list(z.shape), "seconds": [round(s, 3) for s in seconds]})
    receipt.update({"vram_peak": cuda_peak(), "rss": rss_block(), "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out = Path(args.out)
    sha = write_json_atomic(out, receipt)
    print(json.dumps({"receipt": str(out), "sha256": sha, "seconds_per_forward_steady": receipt.get("seconds_per_forward_steady"),
                      "vram_peak": receipt["vram_peak"]}, indent=1))
    return 0


# ======================================================================================
# plan / verify / CLI
# ======================================================================================


def run_plan(args: argparse.Namespace) -> int:
    out = plan_bytes(teacher=args.teacher, items=args.items_count, per_item=args.sigmas_per_item, cfg=args.cfg,
                     width=args.width, height=args.height, length=args.length, block_rows=args.block_rows)
    rows = sigma_plan(min(args.items_count, 4), per_item=args.sigmas_per_item, mode=args.sigma_mode, shift=args.shift,
                      seed=args.seed, sigmas=args.sigmas)
    out["sigma_sample_first_items"] = [[r["item"], r["j"], round(r["sigma"], 4)] for r in rows]
    if args.seconds_per_forward:
        out["hours_at_measured_rate"] = round(out["forwards"] * float(args.seconds_per_forward) / 3600.0, 2)
    print(json.dumps(out, indent=1, default=str))
    return 0


def run_verify(args: argparse.Namespace) -> int:
    results = [verify_cache(Path(d)) for d in args.dirs]
    print(json.dumps(results, indent=1))
    return 0 if all(r["ok"] for r in results) else 1


def _common(p: argparse.ArgumentParser, *, gpu: bool = True) -> None:
    p.add_argument("--items", required=True, help="item JSONL (manifest subcommand output or hand-made)")
    p.add_argument("--cache-root", default=str(CACHE_ROOT_DEFAULT))
    p.add_argument("--cache-tag", default="", help="suffix for the teacher dir (a second plan over the same corpus)")
    p.add_argument("--max-items", type=int, default=0, help="first N items only (smoke)")
    p.add_argument("--teacher-digest", default="full", choices=("full", "stat"), help="full sha256 of the teacher before AND after (default)")
    if gpu:
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--expect-device-name", default=None)
        p.add_argument("--vram-guard-mib", type=int, default=2000)
        p.add_argument("--guard-wait", type=float, default=0.0, help="seconds to keep re-reading a busy card before refusing (0 = refuse at once)")


def _sigma_args(p: argparse.ArgumentParser, *, per_item: int, mode: str, shift: float) -> None:
    p.add_argument("--sigmas-per-item", type=int, default=per_item)
    p.add_argument("--sigma-mode", default=mode, choices=("stratified-shifted", "fixed", "cycle"))
    p.add_argument("--sigmas", type=lambda s: [float(v) for v in s.split(",") if v.strip()], default=None)
    p.add_argument("--shift", type=float, default=shift)
    p.add_argument("--seed", type=int, default=20261012)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="curated registry -> item JSONL")
    p.add_argument("--route", required=True, choices=("video", "image"))
    p.add_argument("--pointer", default="/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--held-every", type=int, default=16)
    p.add_argument("--prompt-from", default="instruction", choices=("instruction", "response", "captions"))
    p.add_argument("--captions", default=None, help="JSONL {sha256, caption} keyed by media sha256 (--prompt-from captions)")
    p.add_argument("--exclude-bank", nargs="*", default=[], help="bank/probe/smoke JSON files whose strings are carved out")

    p = sub.add_parser("plan", help="counts, shapes and bytes of a planned cache (no GPU)")
    p.add_argument("--teacher", required=True, choices=("h3", "qimg"))
    p.add_argument("--items-count", type=int, required=True)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--length", type=int, default=48)
    p.add_argument("--block-rows", type=int, default=0)
    p.add_argument("--seconds-per-forward", type=float, default=0.0)
    _sigma_args(p, per_item=4, mode="stratified-shifted", shift=H3_SHIFT_VIDEO)

    p = sub.add_parser("h3-encode", help="clips -> H3 video + audio VAE latents")
    _common(p)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--length", type=int, default=48, help="requested frames at 24 fps (snapped up to 17k+5: 48 -> 56)")
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--max-drop-fraction", type=float, default=0.5, help="refuse to seal when more items than this fraction failed to decode")
    p.add_argument("--video-vae", default=str(H3_VIDEO_VAE))
    p.add_argument("--audio-vae", default=str(H3_AUDIO_VAE))
    p.add_argument("--items-per-shard", type=int, default=16)

    p = sub.add_parser("qimg-encode", help="images -> Qwen-Image VAE latents")
    _common(p)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--max-drop-fraction", type=float, default=0.5)
    p.add_argument("--vae", default=str(QIMG_VAE))
    p.add_argument("--items-per-shard", type=int, default=64)

    for name, te, neg in (("h3-text", H3_TE_NVFP4, False), ("qimg-text", QIMG_TE, True)):
        p = sub.add_parser(name, help=f"{name.split('-')[0]} text-encoder conditioning cache")
        _common(p)
        p.add_argument("--te", default=str(te))
        p.add_argument("--te-load-device", default="auto", choices=("auto", "gpu", "cpu"))
        p.add_argument("--with-negatives", action=argparse.BooleanOptionalAction, default=neg)
        p.add_argument("--negative", default="", help="negative text for items without a 'negative' field")
        p.add_argument("--texts-per-shard", type=int, default=64)

    for name, dit, cfg, shift, per_item, mode in (("h3-velocity", H3_DIT_BF16, 1.0, H3_SHIFT_VIDEO, 4, "stratified-shifted"),
                                                  ("qimg-velocity", QIMG_DIT, 4.0, QIMG_SHIFT_DEFAULT, 4, "stratified-shifted")):
        p = sub.add_parser(name, help=f"{name.split('-')[0]} guided velocity targets")
        _common(p)
        p.add_argument("--dit", default=str(dit))
        p.add_argument("--weight-dtype", default="default", choices=("default", "bfloat16", "float16", "float32", "fp8_e4m3fn"))
        p.add_argument("--encode-cache", required=True)
        p.add_argument("--te-cache", required=True)
        p.add_argument("--cfg", type=float, default=cfg)
        p.add_argument("--negative", default="")
        _sigma_args(p, per_item=per_item, mode=mode, shift=shift)
        p.add_argument("--shift-override", action="store_true")
        p.add_argument("--block-pairs", default="25" if name.startswith("h3") else "")
        p.add_argument("--block-rows", type=int, default=256)
        p.add_argument("--block-include-text", action="store_true")
        p.add_argument("--block-store-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
        p.add_argument("--items-per-shard", type=int, default=8)
        p.add_argument("--parity-every", type=int, default=128, help="direct-call parity arm every N targets (0 = first only, -1 = off)")
        p.add_argument("--parity-tolerance", type=float, default=1e-3)

    p = sub.add_parser("probe", help="S0 P4: s/forward, peak VRAM and RSS of one teacher component")
    p.add_argument("--teacher", required=True, choices=("h3", "qimg"))
    p.add_argument("--component", required=True, choices=("dit", "te", "vae"))
    p.add_argument("--path", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weight-dtype", default="default", choices=("default", "bfloat16", "float16", "float32", "fp8_e4m3fn"))
    p.add_argument("--te-load-device", default="auto", choices=("auto", "gpu", "cpu"))
    p.add_argument("--te-cache", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--text-len", type=int, default=128)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--length", type=int, default=48)
    p.add_argument("--forwards", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expect-device-name", default=None)
    p.add_argument("--vram-guard-mib", type=int, default=2000)
    p.add_argument("--guard-wait", type=float, default=0.0)

    p = sub.add_parser("verify", help="re-hash sealed caches against their manifests")
    p.add_argument("dirs", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.cmd}")
    try:
        return _dispatch(args)
    finally:
        release_gpu()


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "manifest":
        return run_manifest(args)
    if args.cmd == "plan":
        return run_plan(args)
    if args.cmd == "verify":
        return run_verify(args)
    if args.cmd == "h3-encode":
        return run_h3_encode(args)
    if args.cmd == "qimg-encode":
        return run_qimg_encode(args)
    if args.cmd == "h3-text":
        return run_text(args, teacher="h3")
    if args.cmd == "qimg-text":
        return run_text(args, teacher="qimg")
    if args.cmd == "h3-velocity":
        return run_velocity(args, teacher="h3")
    if args.cmd == "qimg-velocity":
        return run_velocity(args, teacher="qimg")
    if args.cmd == "probe":
        return run_probe(args)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
