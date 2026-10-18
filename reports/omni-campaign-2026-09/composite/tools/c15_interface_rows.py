#!/usr/bin/env python3
"""c15_interface_rows.py -- Lane M2: C15 (+ C11) native media interface rows at trunk width H.

Writes the S6 interface-row sets that ``c16_install_media.py --c15 DIR --c11 DIR`` installs
into ``native_media_bridge`` / ``native_media_head`` (format of record:
``composite/COMPOSITE_S6_INPUT_FORMATS.md`` section 1), for the five media sources:

  music_ace    ACE-Step 1.5 DiT        type 4  (music)
  video_h3     MiniMax H3 DiT, video   type 2  (video)
  audio_h3     MiniMax H3 DiT, audio   type 3  (audio)
  image_qimg   Qwen-Image MMDiT        type 1  (image)
  volume_hy3d  Hunyuan3D-2.1 DiT       type 7  (volume)
  ocr_qvl32    Qwen3-VL-32B ViT (C11)  type 6  (ocr; image id 1 is QIMG's)

RUNG: "latent" -- one native row is exactly one parent DiT/ViT token, and its feature is
the parent's OWN patch-embed input (the VAE-latent patch; for ACE the DiT's 192-channel
[context | x_t] input over its 2-frame patch; for QVL32 the raw 2x16x16x3 pixel tubelet).
This is the only layout in which all five sources fit ``native_feature_dim`` disjointly
(raw-rung bands need 3,840 + 800 + 3,072 + ... > 6,144 and are not 1:1 with parent tokens
for ACE / H3 video). Below the bridge the trunk keeps each source's VAE as a companion
encoder/decoder (D5b) -- so ``companion_decoder`` is true for every C15 entry.

Composition (exact linear algebra; no training, no fitting of the installed rows):

  feature_proj[:, band]  =  s_e * B_in @ P_src            (E = identity at this rung)
  type_embed[t]         +=  s_e * B_in @ b_src
  head.out[band', :]     =  P_out @ B_out                 (D = identity at this rung)
  head shift  (c16)      :  lstsq(head.out[band'], b_out)

  B_in  = U_r V_r^T   [H, W]   (semi-orthogonal; B_out = B_in^+ = V_r U_r^T)
  V     = [ orth(range of every P_src, b_src, P_out^T of that source)  |
            complement ordered by the parent's block-0 attention read Gram ]
  U     = eigenvectors of the trunk's layer-0 tap second moment, descending energy

``V`` puts every direction the parent's patch embed WRITES and its output projection READS
first, so the bridge is injective on the parent's token subspace and the head reproduces
``P_out`` exactly even where W > H (H3: 5,376 > 5,120). The pairing of V's columns with
U's columns is a DESIGN CHOICE recorded in the receipt, not a fit: nothing pairs trunk and
parent states at layer 0. ``s_e`` is a scale heuristic (also not a fit): it makes the media
content carry ``--content-ratio`` x the norm of S0's random bridge terms (type_embed row,
position and noise projections) before the bridge's RMSNorm.

Diagnostics only (owner policy 2026-10-06: finite numeric values never gate):
  * bridge injectivity (singular values, pinv round trip of real latents) and head transport;
  * PSNR of the parent VAE round trip on held clips/images/volumes, the PSNR THROUGH the
    interface (decoder fed the interface's own reconstruction of the latent), and the
    media-lane raw-rung ladder (linear E/D, ``mediafold.c15_fit``) where a raw rung exists;
  * the c16 installer's own ``load_interface_set`` + ``check_bands`` on the written files,
    and a simulated install into the trunk's bridge/head tensors followed by a forward of
    the real ``NativeContinuousMediaBridge`` / ``UnifiedNativeMediaEmissionHead``.

Structural refusals (exit 2): a band outside ``native_feature_dim``, overlapping bands, a
duplicate type id, a non-finite composed tensor, a projection whose shape does not match the
declared band width, a parent tensor missing from its weights file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

TOOL_VERSION = "1.0.0"
INTERFACE_SCHEMA = "chimera_omni.composite.interface_set.v1"
RECEIPT_SCHEMA = "chimera_omni.composite.c15_interface_rows.v1"
HERE = Path(__file__).resolve().parent


class Refusal(RuntimeError):
    """A structural/identity failure. Numeric values never raise this."""


def log(message: str) -> None:
    print(f"[c15-rows {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

DEFAULT_PLAN: dict[str, Any] = {
    "hidden": 5120,
    "native_feature_dim": 6144,
    "native_type_count": 8,
    "sources": {
        "ace": {
            "weights": "diffusion_models/acestep_v1.5_turbo.safetensors",
            "width": 2048,
            "read_gram": [r"^decoder\.layers\.0\.self_attn\.[qkv]_proj\.weight$"],
        },
        "h3": {
            "weights": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
            "width": 5376,
            "read_gram": [r"^blocks\.0\.attn\.qkv_proj\.weight$"],
        },
        "qimg": {
            "weights": "diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
            "width": 3072,
            "read_gram": [r"^transformer_blocks\.0\.attn\.to_[qkv]\.weight$"],
        },
        "hy3d": {
            "weights": "checkpoints/hunyuan_3d_v2.1.safetensors",
            "width": 2048,
            "read_gram": [r"^model\.blocks\.0\.attn1\.to_[qkv]\.weight$"],
        },
        "qvl32": {
            "weights": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
            "width": 1152,
            "read_gram": [r"^visual\.blocks\.0\.attn\.qkv\.weight$"],
        },
    },
    "entries": [
        {
            "set": "c15", "name": "music_ace", "source": "ace", "type_id": 4,
            "feature_band": [0, 384], "emission_band": [0, 128],
            "in": {"weight": "decoder.proj_in.1.weight", "bias": "decoder.proj_in.1.bias", "kind": "conv1d_patch"},
            "out": {"weight": "decoder.proj_out.1.weight", "bias": "decoder.proj_out.1.bias", "kind": "conv_transpose1d_patch"},
            "feature_layout": "index c*2+k over the DiT's own 192-channel input x = cat([src_latents(64) | chunk_masks(64) | x_t(64)]) at the 2 latent frames k of one patch (ComfyUI ace_step15 forward: cat([context_latents, hidden_states]); Conv1d(192, 2048, k=2, s=2))",
            "emission_layout": "index c*2+k over the 64 latent channels x 2 frames that ConvTranspose1d(2048, 64, k=2, s=2) emits for one token (the parent's velocity/flow output space)",
        },
        {
            "set": "c15", "name": "video_h3", "source": "h3", "type_id": 2,
            "feature_band": [384, 480], "emission_band": [128, 224],
            "in": {"weight": "video_patch_proj.weight", "bias": "video_patch_proj.bias", "kind": "linear"},
            "out": {"weight": "final_layer.video_out.weight", "bias": "final_layer.video_out.bias", "kind": "linear"},
            "feature_layout": "the 96-wide H3 video token (24 latent channels x 1x2x2) in H3's own patchify order (video_patch_proj input)",
            "emission_layout": "final_layer.video_out output (96) in the same order",
        },
        {
            "set": "c15", "name": "audio_h3", "source": "h3", "type_id": 3,
            "feature_band": [480, 512], "emission_band": [224, 256],
            "in": {"weight": "audio_patch_proj.weight", "bias": "audio_patch_proj.bias", "kind": "linear"},
            "out": {"weight": "final_layer.audio_out.weight", "bias": "final_layer.audio_out.bias", "kind": "linear"},
            "feature_layout": "the 32-wide H3 audio token (audio_patch_proj input)",
            "emission_layout": "final_layer.audio_out output (32)",
        },
        {
            "set": "c15", "name": "image_qimg", "source": "qimg", "type_id": 1,
            "feature_band": [512, 576], "emission_band": [256, 320],
            "in": {"weight": "img_in.weight", "bias": "img_in.bias", "kind": "linear"},
            "out": {"weight": "proj_out.weight", "bias": "proj_out.bias", "kind": "linear"},
            "feature_layout": "index c*4+ph*2+pw over a 2x2 group of 16-channel DiT-space latents (ComfyUI qwen_image process_img order; Wan21 latent format)",
            "emission_layout": "proj_out output (64) in the same (c, ph, pw) order",
        },
        {
            "set": "c15", "name": "volume_hy3d", "source": "hy3d", "type_id": 7,
            "feature_band": [576, 640], "emission_band": [320, 384],
            "in": {"weight": "model.x_embedder.weight", "bias": "model.x_embedder.bias", "kind": "linear"},
            "out": {"weight": "model.final_layer.linear.weight", "bias": "model.final_layer.linear.bias", "kind": "linear"},
            "feature_layout": "one 64-channel ShapeVAE latent token in DiT space (x scale_factor 1.0039506158752403)",
            "emission_layout": "final_layer.linear output (64)",
        },
        {
            "set": "c11", "name": "ocr_qvl32", "source": "qvl32", "type_id": 6,
            "feature_band": [640, 2176], "emission_band": [384, 385],
            "in": {"weight": "visual.patch_embed.proj.weight", "bias": "visual.patch_embed.proj.bias", "kind": "conv3d_patch"},
            "out": None,
            "feature_layout": "index c*512+t*256+y*16+x over one 2x16x16 RGB tubelet, pixels normalised (x/255-0.5)/0.5 (Conv3d(3, 1152, (2,16,16)) input)",
            "emission_layout": "one zero row: an encoder-only band has no emission (c16 requires a non-empty band)",
        },
    ],
}


# ---------------------------------------------------------------------------
# parent projections
# ---------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def load_tensors(path: Path, names: list[str]) -> dict[str, Any]:
    from safetensors import safe_open

    torch = _torch()
    found: dict[str, Any] = {}
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        for name in names:
            if name not in keys:
                raise Refusal(f"{path}: tensor {name!r} is missing")
            found[name] = handle.get_tensor(name).to(torch.float32).to(torch.float64)
    return found


def match_tensors(path: Path, patterns: list[str]) -> list[str]:
    from safetensors import safe_open

    compiled = [re.compile(p) for p in patterns]
    with safe_open(str(path), framework="pt") as handle:
        return sorted(k for k in handle.keys() if any(c.search(k) for c in compiled))


def source_projection(kind: str, weight: Any, bias: Any) -> tuple[Any, Any]:
    """P_src [W, d_in] and b_src [W] in the feature layout the plan declares."""

    if kind == "linear":
        return weight, bias
    if kind == "conv1d_patch":  # [W, C, k] -> [W, C*k], index c*k_len + k
        return weight.reshape(weight.shape[0], -1), bias
    if kind == "conv3d_patch":  # [W, C, t, h, w] -> [W, C*t*h*w]
        return weight.reshape(weight.shape[0], -1), bias
    raise Refusal(f"unknown source projection kind {kind!r}")


def output_projection(kind: str, weight: Any, bias: Any) -> tuple[Any, Any]:
    """P_out [d_out, W] and b_out [d_out] in the emission layout the plan declares."""

    if kind == "linear":
        return weight, bias
    if kind == "conv_transpose1d_patch":  # ConvTranspose1d weight [W, C, k]: y[c*k_len+k] = sum_i W[i,c,k] h[i] + b[c]
        width, channels, k = weight.shape
        return weight.permute(1, 2, 0).reshape(channels * k, width), bias.repeat_interleave(k)
    raise Refusal(f"unknown output projection kind {kind!r}")


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def fix_signs(basis: Any) -> Any:
    """Deterministic column signs: the largest-|value| entry of every column is positive."""

    torch = _torch()
    index = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[index, torch.arange(basis.shape[1])])
    signs[signs == 0] = 1.0
    return basis * signs


def parent_basis(columns: list[Any], read_gram: Any | None, width: int) -> tuple[Any, dict[str, Any]]:
    """V [W, W]: orth(span of the given columns) first, then the complement by read energy."""

    torch = _torch()
    blocks = []
    for block in columns:
        b = block.reshape(width, -1).to(torch.float64)
        norm = float(b.norm())
        if norm > 0:
            blocks.append(b / norm)
    stacked = torch.cat(blocks, dim=1)
    u_full, singular, _ = torch.linalg.svd(stacked, full_matrices=True)
    tol = float(singular.max()) * max(stacked.shape) * 2.2e-16 * 16
    rank = int((singular > tol).sum())
    q0 = u_full[:, :rank]
    null = u_full[:, rank:]
    notes: dict[str, Any] = {"written_read_rank": rank, "written_read_columns": int(stacked.shape[1]), "singular_min_kept": float(singular[rank - 1]) if rank else 0.0}
    if null.shape[1] and read_gram is not None:
        reduced = null.transpose(0, 1) @ read_gram @ null
        reduced = 0.5 * (reduced + reduced.transpose(0, 1))
        evals, evecs = torch.linalg.eigh(reduced)
        order = torch.argsort(evals, descending=True)
        complement = null @ evecs[:, order]
        notes["complement_order"] = "eigenvectors of N^T G N, G = sum of the parent block-0 attention read weights' W^T W, descending"
        notes["complement_energy_top4"] = [float(v) for v in evals[order][:4]]
    else:
        complement = null
        notes["complement_order"] = "SVD null-space order (no read Gram available)"
    basis = torch.cat((q0, complement), dim=1)
    basis = torch.cat((fix_signs(basis[:, :rank]), fix_signs(basis[:, rank:])), dim=1) if rank < width else fix_signs(basis)
    orth_err = float((basis.transpose(0, 1) @ basis - torch.eye(width, dtype=basis.dtype)).abs().max())
    notes["orthonormality_max_abs_error"] = orth_err
    return basis, notes


def trunk_basis(rows_path: Path, key: str, hidden: int, chunk: int = 2048, centred: bool = True) -> tuple[Any, dict[str, Any]]:
    """U [H, H]: eigenvectors of the trunk tap covariance (centred, default) or second moment, descending.

    The layer-0 second moment is dominated by one direction (the rows' common mean: ~99 %
    of E[x x^T] on the S3 taps), so the centred covariance is the default: media directions
    land on the directions along which the trunk's own rows VARY, not on their shared offset.
    """

    from safetensors import safe_open

    torch = _torch()
    with safe_open(str(rows_path), framework="pt") as handle:
        rows = handle.get_tensor(key)
    if rows.ndim != 2 or rows.shape[1] != hidden:
        raise Refusal(f"{rows_path}:{key} is {tuple(rows.shape)}, expected [N, {hidden}]")
    second = torch.zeros(hidden, hidden, dtype=torch.float64)
    for start in range(0, rows.shape[0], chunk):
        block = rows[start : start + chunk].to(torch.float64)
        second += block.transpose(0, 1) @ block
    second /= float(rows.shape[0])
    mean = rows.to(torch.float64).mean(0)
    uncentred_top = float(torch.linalg.eigvalsh(second)[-1] / torch.diagonal(second).sum())
    if centred:
        second = second - torch.outer(mean, mean)
    evals, evecs = torch.linalg.eigh(second)
    order = torch.argsort(evals, descending=True)
    basis = fix_signs(evecs[:, order])
    total = float(evals.clamp_min(0).sum())
    energy = evals[order].clamp_min(0)
    notes = {
        "rows": int(rows.shape[0]),
        "key": key,
        "statistic": "centred covariance" if centred else "uncentred second moment",
        "uncentred_second_moment_top1_fraction": uncentred_top,
        "mean_norm": float(mean.norm()),
        "file_sha256": sha256_file(rows_path),
        "energy_top1_fraction": float(energy[0] / total) if total else 0.0,
        "energy_top64_fraction": float(energy[:64].sum() / total) if total else 0.0,
        "energy_top512_fraction": float(energy[:512].sum() / total) if total else 0.0,
        "row_norm_median": float(rows.to(torch.float32).norm(dim=1).median()),
    }
    return basis, notes


# ---------------------------------------------------------------------------
# trunk bridge tensors (for the scale heuristic and the simulated install)
# ---------------------------------------------------------------------------

BRIDGE_KEYS = (
    "native_media_bridge.feature_proj.weight",
    "native_media_bridge.type_embed.weight",
    "native_media_bridge.type_feature_scale.weight",
    "native_media_bridge.type_affine.weight",
    "native_media_bridge.position_proj.weight",
    "native_media_bridge.position_fourier_proj.weight",
    "native_media_bridge.noise_proj.0.weight",
    "native_media_bridge.noise_proj.0.bias",
    "native_media_bridge.noise_proj.2.weight",
    "native_media_bridge.norm.weight",
    "native_media_head.norm.weight",
    "native_media_head.out.weight",
    "native_media_head.type_affine.weight",
)


def load_trunk_bridge(native_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    torch = _torch()
    index = json.loads((native_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    recorded = index.get("metadata", {}).get("tensor_sha256", {})
    tensors: dict[str, Any] = {}
    shas: dict[str, Any] = {}
    for key in BRIDGE_KEYS:
        shard = weight_map.get(key)
        if shard is None:
            raise Refusal(f"{native_dir}: trunk lacks {key}")
        with safe_open(str(native_dir / shard), framework="pt") as handle:
            tensors[key] = handle.get_tensor(key).to(torch.float32)
        shas[key] = recorded.get(key)
    config = json.loads((native_dir / "config.json").read_text(encoding="utf-8"))
    return tensors, {"tensor_sha256_recorded": shas, "config": config}


def fourier(values: Any, frequencies: int) -> Any:
    torch = _torch()
    freqs = torch.pow(torch.tensor(2.0, dtype=values.dtype), torch.arange(frequencies, dtype=values.dtype))
    angles = values.unsqueeze(-1) * freqs * math.pi
    return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)


def s0_terms(bridge: dict[str, Any], type_id: int, count: int, seed: int, frequencies: int) -> Any:
    """type_embed[t] + position terms + noise terms at random positions in [0,1]^4 and sigma in [0,1]."""

    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    pos = torch.rand(count, bridge["native_media_bridge.position_proj.weight"].shape[1], generator=gen, dtype=torch.float64)
    sigma = torch.rand(count, 1, generator=gen, dtype=torch.float64)
    w = {k: v.to(torch.float64) for k, v in bridge.items()}
    x = w["native_media_bridge.type_embed.weight"][type_id].unsqueeze(0).expand(count, -1).clone()
    x += pos @ w["native_media_bridge.position_proj.weight"].transpose(0, 1)
    x += fourier(pos, frequencies) @ w["native_media_bridge.position_fourier_proj.weight"].transpose(0, 1)
    hidden = fourier(sigma, frequencies) @ w["native_media_bridge.noise_proj.0.weight"].transpose(0, 1) + w["native_media_bridge.noise_proj.0.bias"]
    x += torch.nn.functional.silu(hidden) @ w["native_media_bridge.noise_proj.2.weight"].transpose(0, 1)
    return x


# ---------------------------------------------------------------------------
# probes (real data where it exists on the box; Gaussian otherwise)
# ---------------------------------------------------------------------------


def read_pairs(directory: Path) -> dict[str, Any] | None:
    from safetensors import safe_open

    path = directory / "pairs.safetensors"
    if not path.is_file():
        return None
    with safe_open(str(path), framework="pt") as handle:
        state = {k: handle.get_tensor(k) for k in handle.keys()}
    receipt = json.loads((directory / "pairs_receipt.json").read_text(encoding="utf-8"))
    state["_geometry"] = receipt.get("geometry", {})
    state["_sha256"] = sha256_file(path)
    return state


def ace_tokens(latent: Any, clip: Any, frame: Any) -> Any:
    """Consecutive frame pairs of one clip -> [n, 384] in the plan's (c, k) layout.

    Context probe: src_latents 0, chunk_masks 1 (recorded; the ComfyUI text-to-music
    context is not reproduced here)."""

    torch = _torch()
    out = []
    for c in torch.unique(clip):
        rows = latent[clip == c][torch.argsort(frame[clip == c])].to(torch.float64)
        pairs = rows.shape[0] // 2
        if pairs == 0:
            continue
        x_t = rows[: pairs * 2].reshape(pairs, 2, 64).permute(0, 2, 1)  # [n, 64, 2]
        ctx_src = torch.zeros(pairs, 64, 2, dtype=torch.float64)
        ctx_mask = torch.ones(pairs, 64, 2, dtype=torch.float64)
        out.append(torch.cat((ctx_src, ctx_mask, x_t), dim=1).reshape(pairs, 384))
    return torch.cat(out)


def qimg_capture_tokens(files: list[Path]) -> tuple[Any, Any, list[dict[str, Any]]]:
    from safetensors import safe_open

    torch = _torch()
    tokens, states, meta = [], [], []
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            x = handle.get_tensor("model_input").to(torch.float64)[0, :, 0]  # [16, h, w]
            img_in = handle.get_tensor("img_in").to(torch.float64)[0]
        c, h, w = x.shape
        packed = x.reshape(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4).reshape((h // 2) * (w // 2), c * 4)
        tokens.append(packed)
        states.append(img_in)
        meta.append({"file": str(path), "sha256": sha256_file(path)})
    return torch.cat(tokens), torch.cat(states), meta


def choose_images(directory: Path, count: int, min_side: int) -> list[Path]:
    from PIL import Image

    suffixes = (".png", ".jpg", ".jpeg", ".webp")
    candidates = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    usable = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                if min(image.size) >= int(min_side):
                    usable.append(path)
        except Exception:
            continue
    if len(usable) < count:
        raise Refusal(f"only {len(usable)} usable images under {directory}")
    step = len(usable) / float(count)
    return [usable[int(i * step)] for i in range(count)]


def qvl_tubelets(paths: list[Path], size: int) -> tuple[Any, Any]:
    """[n, 1536] normalised tubelets (frame duplicated to temporal 2) and their image index."""

    from PIL import Image

    torch = _torch()
    rows, owner = [], []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            w, h = image.size
            side = min(w, h)
            image = image.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)).resize((size, size), Image.BICUBIC)
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
        pix = (data.to(torch.float64) / 255.0 - 0.5) / 0.5  # [H, W, 3]
        chw = pix.permute(2, 0, 1)
        grid = chw.reshape(3, size // 16, 16, size // 16, 16).permute(1, 3, 0, 2, 4)  # [gh, gw, c, y, x]
        grid = grid.unsqueeze(3).expand(-1, -1, -1, 2, -1, -1)  # [gh, gw, c, t, y, x]
        rows.append(grid.reshape(-1, 3 * 2 * 16 * 16))
        owner.append(torch.full((rows[-1].shape[0],), index, dtype=torch.long))
    return torch.cat(rows), torch.cat(owner)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------


def rel_l2(prediction: Any, target: Any) -> float:
    denom = float(target.norm())
    return float((prediction - target).norm()) / denom if denom > 0 else float((prediction - target).norm())


def psnr(prediction: Any, target: Any, data_range: float) -> float:
    mse = float((prediction.double() - target.double()).pow(2).mean())
    return float("inf") if mse <= 0 else 10.0 * math.log10(data_range * data_range / mse)


def finite_or_none(value: float) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


class Builder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else DEFAULT_PLAN
        self.hidden = int(self.plan["hidden"])
        self.feature_dim = int(self.plan["native_feature_dim"])
        self.models = Path(args.models_root)
        self.out = Path(args.out)
        self.c11_out = Path(args.c11_out)
        self.receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "tool": {"name": "c15_interface_rows.py", "version": TOOL_VERSION, "sha256": sha256_file(Path(__file__))},
            "argv": sys.argv[1:],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "owner_policy": "finite numeric values are diagnostics, never gates (Cyril 2026-10-06)",
            "rung": "latent (one native row = one parent token; companion VAE encoder/decoder kept, D5b)",
        }

    # -- checks ---------------------------------------------------------------
    def check_plan(self) -> None:
        names, types = set(), {}
        spans: dict[str, list[tuple[int, int, str]]] = {"feature_band": [], "emission_band": []}
        for entry in self.plan["entries"]:
            if entry["name"] in names:
                raise Refusal(f"duplicate entry name {entry['name']}")
            names.add(entry["name"])
            t = int(entry["type_id"])
            if not 1 <= t < int(self.plan["native_type_count"]):
                raise Refusal(f"{entry['name']}: type id {t} out of range")
            if t in types:
                raise Refusal(f"type id {t} written by {types[t]} and {entry['name']}")
            types[t] = entry["name"]
            for side in spans:
                b, e = (int(v) for v in entry[side])
                if not 0 <= b < e <= self.feature_dim:
                    raise Refusal(f"{entry['name']}: {side} [{b},{e}) outside [0,{self.feature_dim})")
                for b0, e0, n0 in spans[side]:
                    if max(b, b0) < min(e, e0):
                        raise Refusal(f"{side} of {entry['name']} overlaps {n0}")
                spans[side].append((b, e, entry["name"]))
            if entry["source"] not in self.plan["sources"]:
                raise Refusal(f"{entry['name']}: unknown source {entry['source']}")

    # -- main -------------------------------------------------------------------
    def run(self) -> int:
        torch = _torch()
        torch.manual_seed(int(self.args.seed))
        self.check_plan()
        self.receipt["plan"] = self.plan
        log("trunk basis from the layer-0 tap rows")
        U, u_notes = trunk_basis(Path(self.args.trunk_taps), self.args.taps_key, self.hidden, centred=(self.args.trunk_statistic == "centred"))
        self.receipt["trunk_basis"] = {"path": str(self.args.trunk_taps), **u_notes, "meaning": "U = eigenvectors of the trunk's layer-0 tap row statistic (see 'statistic'); the parent's written/read directions are mapped onto the trunk's top directions in order (design choice, not a fit)"}
        bridge, bridge_meta = load_trunk_bridge(Path(self.args.trunk_native))
        freq = int(bridge_meta["config"]["config"]["native_coordinate_frequencies"])
        self.receipt["trunk_bridge"] = {"native_dir": str(self.args.trunk_native), "tensor_sha256_recorded": bridge_meta["tensor_sha256_recorded"], "chimera_commit": bridge_meta["config"].get("chimera_commit")}
        cfg = bridge_meta["config"]["config"]
        if int(cfg["hidden_size"]) != self.hidden or int(cfg["native_feature_dim"]) != self.feature_dim:
            raise Refusal(f"trunk hidden/feature {cfg['hidden_size']}/{cfg['native_feature_dim']} != plan {self.hidden}/{self.feature_dim}")

        # parent projections per source
        projections: dict[str, dict[str, Any]] = {}
        source_notes: dict[str, Any] = {}
        for source, spec in self.plan["sources"].items():
            path = self.models / spec["weights"]
            entries = [e for e in self.plan["entries"] if e["source"] == source]
            names = []
            for e in entries:
                names += [e["in"]["weight"], e["in"]["bias"]]
                if e.get("out"):
                    names += [e["out"]["weight"], e["out"]["bias"]]
            read_names = match_tensors(path, spec.get("read_gram", []))
            log(f"{source}: {path.name} ({len(names)} projection tensors, {len(read_names)} read-Gram tensors)")
            tensors = load_tensors(path, names + read_names)
            width = int(spec["width"])
            per_entry = {}
            columns = []
            for e in entries:
                p_in, b_in = source_projection(e["in"]["kind"], tensors[e["in"]["weight"]], tensors[e["in"]["bias"]])
                if p_in.shape[0] != width:
                    raise Refusal(f"{e['name']}: P_src rows {p_in.shape[0]} != source width {width}")
                d_in = int(e["feature_band"][1]) - int(e["feature_band"][0])
                if p_in.shape[1] != d_in:
                    raise Refusal(f"{e['name']}: P_src columns {p_in.shape[1]} != feature band width {d_in}")
                record = {"P_src": p_in, "b_src": b_in}
                columns += [p_in, b_in.unsqueeze(1)]
                if e.get("out"):
                    p_out, b_out = output_projection(e["out"]["kind"], tensors[e["out"]["weight"]], tensors[e["out"]["bias"]])
                    d_out = int(e["emission_band"][1]) - int(e["emission_band"][0])
                    if tuple(p_out.shape) != (d_out, width):
                        raise Refusal(f"{e['name']}: P_out {tuple(p_out.shape)} != ({d_out}, {width})")
                    record.update({"P_out": p_out, "b_out": b_out})
                    columns.append(p_out.transpose(0, 1))
                per_entry[e["name"]] = record
            gram = None
            if read_names:
                gram = torch.zeros(width, width, dtype=torch.float64)
                for n in read_names:
                    w = tensors[n]
                    if w.shape[-1] != width:
                        continue
                    gram += w.reshape(-1, width).transpose(0, 1) @ w.reshape(-1, width)
            V, v_notes = parent_basis(columns, gram, width)
            r = min(width, self.hidden)
            Ur, Vr = U[:, :r], V[:, :r]
            chart_in = Ur @ Vr.transpose(0, 1)  # [H, W]
            chart_out = Vr @ Ur.transpose(0, 1)  # [W, H]
            projections[source] = {"entries": per_entry, "chart_in": chart_in, "chart_out": chart_out, "width": width, "rank": r}
            source_notes[source] = {
                "weights": str(path),
                "weights_sha256": sha256_file(path) if self.args.hash_parents else "not hashed (--hash-parents off)",
                "projection_tensors": names,
                "read_gram_tensors": read_names,
                "width": width,
                "chart_rank": r,
                "dropped_parent_directions": width - r,
                "basis": v_notes,
                "chart_kind": "B_in = U_r V_r^T (semi-orthogonal), B_out = V_r U_r^T = pinv(B_in); design choice, not a fit",
            }
            chart_path = self.out / "charts" / f"{source}.safetensors"
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file

            save_file({"chart_in": chart_in.to(torch.float32).contiguous(), "chart_out": chart_out.to(torch.float32).contiguous()}, str(chart_path))
            source_notes[source]["chart_file"] = {"path": str(chart_path), "sha256": sha256_file(chart_path)}
        self.receipt["sources"] = source_notes

        probes = self.load_probes()
        entries_out: dict[str, list[dict[str, Any]]] = {"c15": [], "c11": []}
        diagnostics: dict[str, Any] = {}
        for e in self.plan["entries"]:
            src = projections[e["source"]]
            rec = src["entries"][e["name"]]
            chart_in, chart_out = src["chart_in"], src["chart_out"]
            P, b = rec["P_src"], rec["b_src"]
            d_in = P.shape[1]
            # scale heuristic: Gaussian unit latents; content = P z + b (exactly preserved by B_in)
            gen = torch.Generator().manual_seed(int(self.args.seed) + int(e["type_id"]))
            z = torch.randn(int(self.args.scale_probes), d_in, generator=gen, dtype=torch.float64)
            content = z @ P.transpose(0, 1) + b
            noise_terms = s0_terms(bridge, int(e["type_id"]), int(self.args.scale_probes), int(self.args.seed), freq)
            content_norm = float(content.norm(dim=1).median())
            s0_norm = float(noise_terms.norm(dim=1).median())
            scale = float(self.args.content_ratio) * s0_norm / content_norm if content_norm > 0 else 1.0
            feature_proj = scale * (chart_in @ P)  # [H, d_in]
            type_delta = scale * (chart_in @ b)  # [H]
            if rec.get("P_out") is not None:
                head_out = rec["P_out"] @ chart_out  # [d_out, H]
                head_shift = rec["b_out"]
            else:
                width_e = int(e["emission_band"][1]) - int(e["emission_band"][0])
                head_out = torch.zeros(width_e, self.hidden, dtype=torch.float64)
                head_shift = torch.zeros(width_e, dtype=torch.float64)
            for name, tensor in (("feature_proj", feature_proj), ("type_delta", type_delta), ("head_out", head_out), ("head_shift", head_shift)):
                if not bool(torch.isfinite(tensor).all()):
                    raise Refusal(f"{e['name']}: composed {name} is not finite")
            diag = self.entry_diagnostics(e, rec, src, feature_proj, type_delta, head_out, scale, probes, bridge, freq)
            diag["scale"] = {"s": scale, "content_ratio_target": float(self.args.content_ratio), "gaussian_content_norm_median": content_norm, "s0_terms_norm_median": s0_norm, "meaning": "s makes median ||s B_in (P z + b)|| = ratio x median ||type_embed[t] + position + noise terms|| for z ~ N(0, I); a heuristic, not a fit"}
            diagnostics[e["name"]] = diag
            target_root = self.out if e["set"] == "c15" else self.c11_out
            rows_rel = f"{e['name']}/composed_interface.safetensors"
            rows_path = target_root / rows_rel
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file

            state = {
                "bridge.feature_proj.weight": feature_proj.to(torch.float32).contiguous(),
                "bridge.type_embed_delta": type_delta.to(torch.float32).contiguous(),
                "head.out.weight": head_out.to(torch.float32).contiguous(),
                "head.type_affine_shift": head_shift.to(torch.float32).contiguous(),
            }
            save_file(state, str(rows_path))
            tensor_sha = self.tensor_sha256(rows_path)
            vae = diag.get("vae_psnr", {})
            entries_out[e["set"]].append(
                {
                    "name": e["name"],
                    "source": e["source"],
                    "type_id": int(e["type_id"]),
                    "feature_band": [int(v) for v in e["feature_band"]],
                    "emission_band": [int(v) for v in e["emission_band"]],
                    "rows": rows_rel,
                    "rung": "latent",
                    "companion_decoder": None if e.get("out") is None else True,
                    "psnr_absorbed_db": finite_or_none(vae.get("psnr_through_interface_db")),
                    "psnr_parent_roundtrip_db": finite_or_none(vae.get("psnr_parent_roundtrip_db")),
                    "psnr_meaning": vae.get("meaning"),
                    "raw_rung_ladder": vae.get("raw_rung_ladder"),
                    "feature_layout": e.get("feature_layout"),
                    "emission_layout": e.get("emission_layout"),
                    "scale": scale,
                    "chart": source_notes[e["source"]]["chart_file"],
                    "tensor_sha256": tensor_sha,
                    "rows_file_sha256": sha256_file(rows_path),
                }
            )
            log(f"{e['name']}: type {e['type_id']} f{e['feature_band']} e{e['emission_band']} s={scale:.4g} bridge_recon_relL2={diag['bridge']['pinv_roundtrip_rel_l2_real']}")
        self.receipt["diagnostics"] = diagnostics
        written = {}
        for label, root in (("c15", self.out), ("c11", self.c11_out)):
            body = {
                "schema": INTERFACE_SCHEMA,
                "hidden": self.hidden,
                "native_feature_dim": self.feature_dim,
                "producer": {"tool": "c15_interface_rows.py", "version": TOOL_VERSION, "lane": "M2 2026-10-10"},
                "entries": entries_out[label],
            }
            root.mkdir(parents=True, exist_ok=True)
            path = root / "companions.json"
            path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written[label] = {"path": str(path), "sha256": sha256_file(path), "entries": [x["name"] for x in entries_out[label]]}
        self.receipt["companions"] = written
        self.receipt["installer_check"] = self.installer_check()
        self.receipt["simulated_install"] = self.simulated_install(bridge, probes, freq)
        self.receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt_path = self.out / "c15_interface_receipt.json"
        receipt_path.write_text(json.dumps(self.receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        log(f"receipt {receipt_path} sha256 {sha256_file(receipt_path)}")
        return 0

    # -- sha formula of record ------------------------------------------------
    def tensor_sha256(self, path: Path) -> dict[str, str]:
        from chimera_omni.importing.safetensors_io import read_safetensors

        views = read_safetensors(path, verify=False)
        return {name: str(view.sha256()) for name, view in sorted(views.items())}

    # -- probes -------------------------------------------------------------------
    def load_probes(self) -> dict[str, Any]:
        torch = _torch()
        a = self.args
        probes: dict[str, Any] = {"notes": {}}
        ace = read_pairs(Path(a.ace_pairs)) if a.ace_pairs else None
        if ace is not None:
            probes["music_ace"] = {"latent_tokens": ace_tokens(ace["latent"], ace["clip_index"], ace["frame_index"]), "pairs": ace}
        h3a = read_pairs(Path(a.h3_audio_pairs)) if a.h3_audio_pairs else None
        if h3a is not None:
            probes["audio_h3"] = {"latent_tokens": h3a["latent"].to(torch.float64), "pairs": h3a}
        h3v = read_pairs(Path(a.h3_video_pairs)) if a.h3_video_pairs else None
        if h3v is not None:
            lat = h3v["latent"].to(torch.float64)
            n = lat.shape[0] // 4
            probes["video_h3"] = {"latent_tokens": lat[: n * 4].reshape(n, 96), "pairs": h3v, "note": "probe vectors = 4 consecutive latent-pixel rows (24 ch each), NOT spatially exact 1x2x2 tokens"}
        if a.qimg_captures:
            files = sorted(Path(a.qimg_captures).glob(a.qimg_capture_glob))[: int(a.qimg_capture_files)]
            if files:
                tokens, states, meta = qimg_capture_tokens(files)
                probes["image_qimg"] = {"latent_tokens": tokens, "parent_states": states, "capture_files": meta}
        qimg_pairs = read_pairs(Path(a.qimg_pairs)) if a.qimg_pairs else None
        if qimg_pairs is not None:
            probes.setdefault("image_qimg", {})["pairs"] = qimg_pairs
        if a.hy3d_probes and (Path(a.hy3d_probes) / "volume_probes.safetensors").is_file():
            from safetensors import safe_open

            with safe_open(str(Path(a.hy3d_probes) / "volume_probes.safetensors"), framework="pt") as handle:
                lat = handle.get_tensor("latent_dit").to(torch.float64)
            receipt = json.loads((Path(a.hy3d_probes) / "volume_probe_receipt.json").read_text(encoding="utf-8"))
            probes["volume_hy3d"] = {"latent_tokens": lat.reshape(-1, lat.shape[-1]), "volume_receipt": receipt, "file_sha256": sha256_file(Path(a.hy3d_probes) / "volume_probes.safetensors")}
        if a.ace_capture and Path(a.ace_capture).is_file():
            from safetensors import safe_open

            with safe_open(str(a.ace_capture), framework="pt") as handle:
                probes.setdefault("music_ace", {})["parent_states"] = handle.get_tensor("x_in").to(torch.float64).reshape(-1, 2048)
        if a.images_dir:
            paths = choose_images(Path(a.images_dir), int(a.image_count), 256)
            tubelets, owner = qvl_tubelets(paths, int(a.qvl_size))
            gen = torch.Generator().manual_seed(int(a.seed))
            held = torch.randperm(len(paths), generator=gen)[: max(1, len(paths) // 4)]
            probes["ocr_qvl32"] = {"tubelets": tubelets, "owner": owner, "held_images": held, "images": [str(p) for p in paths]}
        return probes

    # -- diagnostics ----------------------------------------------------------------
    def entry_diagnostics(self, e, rec, src, feature_proj, type_delta, head_out, scale, probes, bridge, freq) -> dict[str, Any]:
        torch = _torch()
        name = e["name"]
        out: dict[str, Any] = {}
        sv = torch.linalg.svdvals(feature_proj)
        pinv = torch.linalg.pinv(feature_proj)
        gen = torch.Generator().manual_seed(int(self.args.seed) + 7)
        z_gauss = torch.randn(512, feature_proj.shape[1], generator=gen, dtype=torch.float64)
        probe = probes.get(name, {})
        real = probe.get("latent_tokens")
        if name == "ocr_qvl32" and "tubelets" in probe:
            real = probe["tubelets"]
        bridge_block = {
            "singular_max": float(sv.max()),
            "singular_min": float(sv.min()),
            "condition": float(sv.max() / sv.min()) if float(sv.min()) > 0 else None,
            "numerical_rank": int((sv > sv.max() * max(feature_proj.shape) * 2.2e-16 * 16).sum()),
            "columns": int(feature_proj.shape[1]),
            "pinv_roundtrip_rel_l2_gaussian": rel_l2(z_gauss @ (pinv @ feature_proj).transpose(0, 1), z_gauss),
            "pinv_roundtrip_rel_l2_real": None,
            "real_probe_rows": 0,
        }
        if real is not None:
            recon = real @ (pinv @ feature_proj).transpose(0, 1)
            bridge_block["pinv_roundtrip_rel_l2_real"] = rel_l2(recon, real)
            bridge_block["real_probe_rows"] = int(real.shape[0])
            # parent-only reference: the parent's own patch embed P (rank-limited for QVL32)
            P = rec["P_src"]
            p_recon = real @ (torch.linalg.pinv(P) @ P).transpose(0, 1)
            bridge_block["parent_patch_embed_pinv_rel_l2_real"] = rel_l2(p_recon, real)
            # energy share of the media content before the bridge's RMSNorm (first 8,192 rows)
            sample = real[:8192]
            content = sample @ feature_proj.transpose(0, 1) + type_delta
            noise = s0_terms(bridge, int(e["type_id"]), int(sample.shape[0]), int(self.args.seed) + 1, freq)
            total = content + noise
            bridge_block["content_energy_fraction_median"] = float((content.pow(2).sum(1) / total.pow(2).sum(1)).median())
            bridge_block["content_energy_fraction_at_scale_1_median"] = float(((content / scale).pow(2).sum(1) / ((content / scale) + noise).pow(2).sum(1)).median())
        out["bridge"] = bridge_block
        if rec.get("P_out") is not None:
            P_out = rec["P_out"]
            chart_in, chart_out = src["chart_in"], src["chart_out"]
            transport = P_out @ chart_out @ chart_in
            y = torch.randn(512, src["width"], generator=gen, dtype=torch.float64)
            head_block = {"transport_rel_l2_gaussian": rel_l2(y @ transport.transpose(0, 1), y @ P_out.transpose(0, 1))}
            states = probe.get("parent_states")
            if states is not None:
                head_block["transport_rel_l2_parent_states"] = rel_l2(states @ transport.transpose(0, 1), states @ P_out.transpose(0, 1))
                head_block["parent_state_rows"] = int(states.shape[0])
                head_block["parent_states_meaning"] = "parent residual states (ACE block-12 input / QIMG img_in capture) as a proxy for the parent's final hidden"
            head_block["head_singular_max"] = float(torch.linalg.svdvals(head_out).max())
            head_block["meaning"] = "|P_out B_out B_in y - P_out y| / |P_out y|: the head emits the parent's own output projection of a parent state carried into the trunk by B_in"
            out["head"] = head_block
        else:
            out["head"] = {"note": "encoder-only band: one zero emission row"}
        out["vae_psnr"] = self.vae_psnr(e, rec, probe, bridge_block, feature_proj, pinv)
        return out

    def vae_psnr(self, e, rec, probe, bridge_block, feature_proj, pinv) -> dict[str, Any]:
        torch = _torch()
        name = e["name"]
        exact = bridge_block.get("pinv_roundtrip_rel_l2_real")
        tol = float(self.args.exact_tol)
        block: dict[str, Any] = {}
        ladder_map = {"music_ace": ("c15_real_v3", "ace_1.5_vae_stacked"), "audio_h3": ("c15_real_v3", "h3_audio_ctx0"), "video_h3": ("c15_video_v2", "h3_video_group4")}
        if name in ladder_map and probe.get("pairs") is not None:
            pairs = probe["pairs"]
            geometry = pairs["_geometry"]
            held = pairs["split"].eq(1)
            data_range = 2.0 if geometry.get("modality") == "audio" else 1.0
            centre = pairs["raw"][held][:, int(geometry.get("centre_offset", 0)) : int(geometry.get("centre_offset", 0)) + int(geometry["d_centre"])]
            ceiling = psnr(pairs["roundtrip"][held], centre, data_range)
            block["psnr_parent_roundtrip_db"] = ceiling
            block["holdout_rows"] = int(held.sum())
            block["pairs_sha256"] = pairs["_sha256"]
            block["data_range"] = data_range
            block["psnr_through_interface_db"] = ceiling if (exact is not None and exact <= tol) else None
            receipt_root, ladder_name = ladder_map[name]
            ladder_path = Path(self.args.ladder_root) / receipt_root / "c15_ladder_receipt.json"
            if ladder_path.is_file():
                ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
                rows = [r for r in ladder.get("ladder", []) if r.get("vae") == ladder_name]
                block["raw_rung_ladder"] = {
                    "receipt": str(ladder_path),
                    "receipt_sha256": sha256_file(ladder_path),
                    "rows": [{k: r.get(k) for k in ("rung", "d_raw", "d_latent", "psnr_absorbed_db", "psnr_parent_roundtrip_db", "linear_rank_bound_psnr_db", "ceiling_below_linear_bound")} for r in rows],
                    "meaning": "media-lane raw-rung absorption (linear/nonlinear E,D fitted raw<->latent per VAE frame at hidden 5120); NOT what is installed (raw bands do not fit F)",
                }
        elif name == "image_qimg" and probe.get("pairs") is not None:
            sys.path.insert(0, str(Path(self.args.mediafold_root)))
            from mediafold.c15_fit import fit_absorption
            from mediafold.c15_pairs import load_pair_set

            pair_set = load_pair_set(self.args.qimg_pairs)
            held = pair_set.split.eq(1)
            ceiling = psnr(pair_set.roundtrip[held], pair_set.raw[held], 1.0)
            block["psnr_parent_roundtrip_db"] = ceiling
            block["holdout_rows"] = int(held.sum())
            block["pairs_sha256"] = probe["pairs"]["_sha256"]
            block["data_range"] = 1.0
            lat = pair_set.latent[held].to(torch.float64)
            recon = lat @ (pinv @ feature_proj).transpose(0, 1)
            block["latent_through_interface_rel_l2_held"] = rel_l2(recon, lat)
            exact = block["latent_through_interface_rel_l2_held"]
            block["psnr_through_interface_db"] = ceiling if exact <= tol else None
            maps = fit_absorption(pair_set, rung="linear", seed=int(self.args.seed))
            m = maps.metrics
            block["raw_rung_ladder"] = {
                "rows": [{"rung": "linear", "d_raw": 768, "d_latent": 64, "psnr_absorbed_db": m.get("psnr_absorbed_db"), "psnr_parent_roundtrip_db": m.get("psnr_parent_roundtrip_db"), "linear_rank_bound_psnr_db": (m.get("linear_rank_bound") or {}).get("psnr_db"), "ceiling_below_linear_bound": (m.get("linear_rank_bound") or {}).get("ceiling_below_linear_bound"), "encoder_cos": (m.get("encoder_scores") or {}).get("cos")}],
                "meaning": "media-lane raw-rung absorption fitted here (mediafold.c15_fit.fit_absorption, linear, ridge lambda ladder) on 16x16 RGB patch <-> 2x2x16 DiT-space latent token rows; held split by image; NOT installed",
            }
        elif name == "volume_hy3d" and probe.get("volume_receipt") is not None:
            summary = probe["volume_receipt"]["summary"]
            block["psnr_parent_roundtrip_db"] = summary.get("holdout_mean_psnr_db")
            block["psnr_parent_roundtrip_all_meshes_db"] = summary.get("mean_psnr_db")
            block["occupancy_accuracy_all_meshes"] = summary.get("mean_accuracy")
            vr = probe["volume_receipt"]
            key = "gt0" if str(vr.get("convention", "")).startswith("logit>0") else "lt0"
            held_idx = [int(i) for i in summary.get("holdout_meshes", [])]
            accs = [vr["per_mesh"][i][f"accuracy_logit_{key}_inside"] for i in held_idx]
            block["psnr_hard_occupancy_holdout_db"] = (sum(10.0 * math.log10(1.0 / max(1.0 - a, 1e-12)) for a in accs) / len(accs)) if accs else None
            block["probe_rule"] = vr.get("probe_rule")
            block["note"] = "near-surface probes (+/- eps along the normal) are the hardest occupancy queries; psnr_parent_roundtrip_db is over sigmoid(logit) vs {0,1}, psnr_hard_occupancy is -10 log10(error rate)"
            block["convention"] = probe["volume_receipt"].get("convention")
            block["volume_file_sha256"] = probe.get("file_sha256")
            block["psnr_through_interface_db"] = block["psnr_parent_roundtrip_db"] if (exact is not None and exact <= tol) else None
            block["raw_rung_ladder"] = {"rows": [], "meaning": "not applicable: the ShapeVAE is a perceiver over surface point sets; no raw row pairs with one latent token"}
            block["data_range"] = 1.0
        elif name == "ocr_qvl32" and probe.get("tubelets") is not None:
            held_images = probe["held_images"]
            mask = torch.isin(probe["owner"], held_images)
            x = probe["tubelets"][mask]
            recon_iface = x @ (pinv @ feature_proj).transpose(0, 1)
            block["psnr_parent_roundtrip_db"] = None
            P = rec["P_src"]
            block["psnr_parent_patch_embed_pinv_db"] = psnr(x @ (torch.linalg.pinv(P) @ P).transpose(0, 1), x, 2.0)
            block["psnr_through_interface_db"] = psnr(recon_iface, x, 2.0)
            block["holdout_rows"] = int(mask.sum())
            block["held_images"] = [probe["images"][int(i)] for i in held_images]
            block["data_range"] = 2.0
            block["raw_rung_ladder"] = {"rows": [], "meaning": "the installed QVL32 rung IS raw pixels (the patch embed is linear); no separate ladder"}
        else:
            block["not_measured"] = "no held pair set / probe file for this entry on this run"
        if name == "ocr_qvl32":
            block["meaning"] = "vision has no decoder: psnr_through_interface_db = PSNR of the minimum-norm pixel reconstruction pinv(F) F x on held images' tubelets (data range 2); the parent's own 1152-wide patch embed keeps the same rank, so this equals the parent's information bound"
        else:
            block["meaning"] = ("psnr_parent_roundtrip_db = parent VAE D(E(x)) vs x on held items; psnr_through_interface_db = the decoder fed "
                                "the interface's own reconstruction pinv(F) F z of the latent z -- equal to the round trip by construction when "
                                f"that reconstruction is exact (rel_l2 <= {tol:g}), so the decoder is not re-run; None otherwise")
        return block

    # -- the installer's own checks -------------------------------------------------
    def installer_check(self) -> dict[str, Any]:
        sys.path.insert(0, str(HERE))
        import c16_install_media as c16

        class Cfg:
            hidden_size = self.hidden
            native_feature_dim = self.feature_dim
            native_type_count = int(self.plan["native_type_count"])

        entries = c16.load_interface_set(self.out, "c15", Cfg) + c16.load_interface_set(self.c11_out, "c11", Cfg)
        bands = c16.check_bands(entries, allow_overlap=False)
        return {
            "c16_tool_sha256": sha256_file(Path(c16.__file__)),
            "entries": [{"set": x["set"], "name": x["name"], "type_id": x["type_id"], "feature_band": x["feature_band"], "emission_band": x["emission_band"], "tensor_sha256_check": x["tensor_sha256_check"]} for x in entries],
            "bands": bands,
            "verdict": "accepted by c16 load_interface_set + check_bands (no --allow-band-overlap)",
        }

    def simulated_install(self, bridge: dict[str, Any], probes: dict[str, Any], freq: int) -> dict[str, Any]:
        """c16 install_interface's writes on the trunk's own bridge/head tensors (in memory),
        then the real chimera bridge/head modules run on real probe rows of every type."""

        torch = _torch()
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.modeling.chimera import NativeContinuousMediaBridge, UnifiedNativeMediaEmissionHead

        cfg_body = json.loads((Path(self.args.trunk_native) / "config.json").read_text(encoding="utf-8"))["config"]
        cfg = ChimeraConfig.from_manifest(cfg_body)
        b = NativeContinuousMediaBridge(cfg).double()
        h = UnifiedNativeMediaEmissionHead(cfg).double()
        sd_b = {k.split("native_media_bridge.", 1)[1]: v.double() for k, v in bridge.items() if k.startswith("native_media_bridge.")}
        sd_h = {k.split("native_media_head.", 1)[1]: v.double() for k, v in bridge.items() if k.startswith("native_media_head.")}
        b.load_state_dict(sd_b, strict=True)
        h.load_state_dict(sd_h, strict=True)
        hidden = self.hidden
        report: dict[str, Any] = {"entries": {}}
        from safetensors import safe_open

        with torch.no_grad():
            for label, root in (("c15", self.out), ("c11", self.c11_out)):
                body = json.loads((root / "companions.json").read_text(encoding="utf-8"))
                for entry in body["entries"]:
                    with safe_open(str(root / entry["rows"]), framework="pt") as handle:
                        E = handle.get_tensor("bridge.feature_proj.weight").double()
                        delta = handle.get_tensor("bridge.type_embed_delta").double()
                        D = handle.get_tensor("head.out.weight").double()
                        bias = handle.get_tensor("head.type_affine_shift").double()
                    fb, fe = entry["feature_band"]
                    eb, ee = entry["emission_band"]
                    t = int(entry["type_id"])
                    pre = {
                        "type_feature_scale_nonzero": int(b.type_feature_scale.weight[t, fb:fe].ne(0).sum()),
                        "bridge_affine_scale_nonzero": int(b.type_affine.weight[t, :hidden].ne(0).sum()),
                        "head_affine_scale_nonzero": int(h.type_affine.weight[t, :hidden].ne(0).sum()),
                    }
                    b.feature_proj.weight[:, fb:fe] = E
                    b.type_embed.weight[t] += delta
                    h.out.weight[eb:ee, :] = D
                    rows = h.out.weight[eb:ee, :]
                    if float(bias.norm()) > 0:
                        solution = torch.linalg.lstsq(rows, bias.unsqueeze(1)).solution.squeeze(1)
                        residual = float((rows @ solution - bias).norm() / bias.norm())
                    else:
                        solution = torch.zeros(hidden, dtype=torch.float64)
                        residual = 0.0
                    h.type_affine.weight[t, hidden:] += solution
                    report["entries"][entry["name"]] = {"preconditions_nonzero": pre, "head_shift_lstsq_relative_residual": residual}
            # forward on real probes per type
            for label, root in (("c15", self.out), ("c11", self.c11_out)):
                body = json.loads((root / "companions.json").read_text(encoding="utf-8"))
                for entry in body["entries"]:
                    name = entry["name"]
                    probe = probes.get(name, {})
                    rows = probe.get("tubelets") if name == "ocr_qvl32" else probe.get("latent_tokens")
                    rows_kind = "real probe rows"
                    if rows is None:
                        fb, fe = entry["feature_band"]
                        rows = torch.randn(64, fe - fb, generator=torch.Generator().manual_seed(int(self.args.seed) + 11), dtype=torch.float64)
                        rows_kind = "gaussian N(0, I) rows (no real probe on this run)"
                    rows = rows[:256]
                    n = int(rows.shape[0])
                    fb, fe = entry["feature_band"]
                    feats = torch.zeros(1, n, self.feature_dim, dtype=torch.float64)
                    feats[0, :, fb:fe] = rows
                    types = torch.full((1, n), int(entry["type_id"]), dtype=torch.long)
                    gen = torch.Generator().manual_seed(int(self.args.seed) + 3)
                    pos = torch.rand(1, n, int(cfg.native_position_dim), generator=gen, dtype=torch.float64)
                    noise = torch.rand(1, n, generator=gen, dtype=torch.float64)
                    valid = torch.ones(1, n, dtype=torch.float64)
                    x = b.embed(feats, types, pos, valid, noise_levels=noise)
                    with safe_open(str(root / entry["rows"]), framework="pt") as handle:
                        delta_e = handle.get_tensor("bridge.type_embed_delta").double()
                    content = feats[0] @ b.feature_proj.weight.transpose(0, 1) + delta_e
                    cos = torch.nn.functional.cosine_similarity(x[0], content, dim=-1)
                    emitted = h(x, types)[0, :, entry["emission_band"][0] : entry["emission_band"][1]]
                    report["entries"][name]["forward"] = {
                        "rows": n,
                        "rows_kind": rows_kind,
                        "finite_bridge": bool(torch.isfinite(x).all()),
                        "finite_head": bool(torch.isfinite(emitted).all()),
                        "bridge_row_norm_median": float(x[0].norm(dim=-1).median()),
                        "cos_bridge_output_vs_media_content_median": float(cos.median()),
                        "emitted_rms": float(emitted.pow(2).mean().sqrt()),
                        "meaning": "real chimera NativeContinuousMediaBridge.embed (RMSNorm, type_affine, positions, noise) after the c16 writes; cos against the transported parent token state s*B_in(P z + b) says how much of each bridge row is the media content",
                    }
        report["modules"] = "chimera_omni.modeling.chimera NativeContinuousMediaBridge / UnifiedNativeMediaEmissionHead (fp64)"
        return report


def build_parser() -> argparse.ArgumentParser:
    oc = os.environ.get("OC", "/home/cereal/omni-campaign")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plan", help="JSON plan (default: the built-in five-source latent-rung plan)")
    p.add_argument("--models-root", default="/home/cereal/comfyui/models")
    p.add_argument("--trunk-native", default="/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native")
    p.add_argument("--trunk-taps", default=f"{oc}/composite/c17/trunk_taps/s3native-c2k4352-held-all/L00/ffn_input_rows.safetensors")
    p.add_argument("--taps-key", default="prenorm")
    p.add_argument("--trunk-statistic", default="centred", choices=("centred", "second-moment"))
    p.add_argument("--out", default=f"{oc}/composite/s7/interface")
    p.add_argument("--c11-out", default=f"{oc}/composite/s7/interface_c11")
    p.add_argument("--mediafold-root", default=f"{oc}/media")
    p.add_argument("--ladder-root", default=f"{oc}/media/out")
    p.add_argument("--ace-pairs", default=f"{oc}/media/pairs/ace_s_ctx0_merged")
    p.add_argument("--h3-audio-pairs", default=f"{oc}/media/pairs/h3a_d_ctx0_merged")
    p.add_argument("--h3-video-pairs", default=f"{oc}/media/pairs/h3v_g4_64_merged")
    p.add_argument("--ace-capture", default=f"{oc}/media/out/c16_L12_v4/capture_L12.safetensors")
    p.add_argument("--qimg-captures", default="/mnt/lexar/omni-campaign/qimg-broad-capture-sep07/run/capture_batch1")
    p.add_argument("--qimg-capture-glob", default="*_fit_*.safetensors")
    p.add_argument("--qimg-capture-files", type=int, default=4)
    p.add_argument("--qimg-pairs", default=f"{oc}/composite/s7/pairs_m2/qimg_vae_tok2")
    p.add_argument("--hy3d-probes", default=f"{oc}/composite/s7/pairs_m2/hy3d_volume_probe")
    p.add_argument("--images-dir", default="/home/cereal/comfyui/output")
    p.add_argument("--image-count", type=int, default=48)
    p.add_argument("--qvl-size", type=int, default=448)
    p.add_argument("--content-ratio", type=float, default=4.0)
    p.add_argument("--scale-probes", type=int, default=2048)
    p.add_argument("--exact-tol", type=float, default=1e-6)
    p.add_argument("--hash-parents", action="store_true", help="sha256 every parent weights file (tens of GB of reads)")
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--chimera-src", default=os.environ.get("CHIMERA_SRC", f"{oc}/import/rq27b/tree_ef1ffb9/src"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chimera_src and args.chimera_src not in sys.path:
        sys.path.insert(0, str(args.chimera_src))
    if args.mediafold_root and str(args.mediafold_root) not in sys.path:
        sys.path.insert(0, str(args.mediafold_root))
    try:
        return Builder(args).run()
    except Refusal as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
