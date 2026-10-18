#!/usr/bin/env python3
"""s7_pairs_parent.py -- PARENT side of the 5120-wide media chart pairs (lane M1, Claude, 2026-10-10).

For each S7 media unit this writes ONE pair-set file

    <out>/<unit>/parent_capture.safetensors   (+ parent_capture.json, sha256 sidecar)

with one row per PARENT TOKEN at the unit's block input:

    features      fp32 [N, Wf]  the parent's own patch-embed INPUT for that token (latent rung;
                                audio rows of H3 are zero-padded past their 32 columns, see
                                feature_width)
    feature_width int64 [N]     how many leading feature columns are real for that row
    parent_rows   bf16/fp32 [N, W]  the source block-L input for that token (what chart_in must hit)
    type_ids      int64 [N]     native media type id (image 1, video 2, audio 3, music 4, ocr 6, volume 7)
    sigma         fp32 [N]      the row's noise level (-1 = none, an encoder unit)
    positions     fp32 [N, 4]   thwr_v1 bridge positions (native_media.py conventions)
    forward       int64 [N]     index into the json `forwards` list (one trunk forward each)
    token         int64 [N]     the row's order inside that forward's media rows
    group         int64 [N]     holdout group (prompt / clip / image); held rows never share a group with fit rows
    held          bool  [N]     diagnostic holdout membership (charts of record are fitted on ~held only)
    time_row      bool  [N]     (1.1.0) the row is a source TIME-TOKEN row, not a content row (HY3D only:
                                the parent's row 0, first in its forward, zero features, feature_width 0,
                                positions TIME_ROW_POSITION, the segment sigma)

The json lists every forward with its trunk PROMPT string (the text the trunk reads before the
media rows), the parent source files with sha256, and the named deviations.

Subcommands (all parent work; nothing here touches the trunk):

    qimg   reuse /mnt/lexar/omni-campaign/qimg-broad-capture-sep07 (block-30 img_in, model_input, prompt)   CPU
    hy3d   reuse Codex's HY3D dense-child capture of block 10 (block_input, dit_latent_input)               CPU
    h3     reuse $OC/composite/media h3_blocks_v2 block-25 x_in and REBUILD the seeded stimulus exactly
           (c16_h3_capture.build_stimuli with the receipt's own argv) to recover each row's input latent  CPU
    ace    fresh capture: mediafold ACE-1.5 DiT prefix (blocks 0..11), latent_stimulus on the C15 pair
           corpus at several sigmas                                                                      GPU
    qvl    fresh capture: Qwen3-VL-32B vision tower (visual.* of the H3 text-encoder file) on the QIMG
           calibration images, block-13 input                                                            GPU

Standing rules: no training, nothing here is a gate; every numeric value is a diagnostic. CPU work
honours OMP_NUM_THREADS (set it, and run under nice -n 19).

1.1.0 (PREP-B, Claude, 2026-10-10): `hy3d` KEEPS the parent's time token. Each forward is now
[1 time row | 4096 latent rows]: the time row's parent row is block_input[s, 0] (the evolved time token
the parent's block 10 self-attends over), its features are zero (feature_width 0), its position is
TIME_ROW_POSITION and its sigma is the segment sigma -- exactly the unit's runtime contract
runtime_requirements.hy3d_time_token_row. 1.0.0 dropped row 0. `--no-time-row` reproduces 1.0.0.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s7_pairs_parent"
TOOL_VERSION = "1.1.0"
SCHEMA = "omni-campaign.s7_parent_pairs.v1"
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
TYPE_IDS = {"image": 1, "video": 2, "audio": 3, "music": 4, "tts": 5, "ocr": 6, "volume": 7}
# runtime_requirements.hy3d_time_token_row (s7_unit_build >= 1.2.0): thwr_v1 [-1, -1, -1, -1]; no latent row
# uses a negative coordinate
TIME_ROW_POSITION = (-1.0, -1.0, -1.0, -1.0)

QIMG_ROOT = Path("/mnt/lexar/omni-campaign/qimg-broad-capture-sep07/run")
HY3D_CAPTURE = Path(
    "/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/dense-child-blocks/"
    "dense-child-block-10-1c1709d9-8afb-47f0-931f-35e01f90f125/capture-block10"
)
HY3D_WEIGHTS = Path("/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors")
H3_CAPTURE = Path("/mnt/lexar/omni-campaign/media/h3_blocks_v2")
ACE_WEIGHTS = Path("/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors")
QVL_WEIGHTS = Path("/home/cereal/comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors")
IMAGE_ROOT = Path("/mnt/lexar/omni-campaign/qimg-broad-calibration-sep07")

PROMPT_ACE = "A short instrumental music clip."
PROMPT_H3 = "A short video clip with its soundtrack."
PROMPT_QVL = "Describe this image."


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    import torch

    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(max(1, threads))
    return torch


def tensor_digest(tensor: Any) -> str:
    """dtype + shape + bytes (the safetensors_io TensorView.sha256 formula's inputs)."""

    torch = _torch()
    t = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(t.dtype).replace("torch.", "").encode())
    digest.update(json.dumps(list(t.shape)).encode())
    if t.dtype == torch.bfloat16:
        digest.update(t.view(torch.int16).numpy().tobytes())
    else:
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# packing helpers (pure, tested)
# --------------------------------------------------------------------------


def qimg_pack(model_input: Any) -> Any:
    """ComfyUI QwenImage.process_img: [B,C,T,H,W] -> [B, T*(H/2)*(W/2), C*4] in (c, ph, pw) order."""

    b, c, t, h, w = model_input.shape
    x = model_input.reshape(b, c, t, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 3, 5, 1, 4, 6)
    return x.reshape(b, t * (h // 2) * (w // 2), c * 4)


def h3_patchify_video(latent: Any, patch: tuple[int, int, int] = (1, 2, 2)) -> Any:
    """comfy minimax patchify_video: [B,C,T,H,W] -> [B*t*h*w, C*pt*ph*pw] ('nctrhpwq->nthwcrpq')."""

    torch = _torch()
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def h3_pack_audio(latent: Any) -> Any:
    """comfy minimax pack_audio: [1, 32, ch, T] -> [ch*T, 32] channel-major."""

    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def ace_token_features(context_latents: Any, hidden_states: Any) -> Any:
    """ACE proj_in input per token: x = cat(context, hidden) [B, L, 192]; Conv1d k=s=2 -> token t reads
    frames 2t, 2t+1; feature[c*2 + k] = x[2t + k, c] so that proj_in.weight.reshape(out, 192*2) is P_src."""

    torch = _torch()
    x = torch.cat((context_latents, hidden_states), dim=-1)
    b, length, c = x.shape
    if length % 2:
        x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        length += 1
    x = x.reshape(b, length // 2, 2, c).permute(0, 1, 3, 2)
    return x.reshape(b, length // 2, c * 2)


def qwen_vl_patchify(pixels: Any, *, patch: int = 16, temporal: int = 2, merge: int = 2) -> tuple[Any, Any, Any]:
    """Qwen2/3-VL image-processor patch order for ONE image [C,H,W] duplicated to `temporal` frames.

    Returns (flat [gh*gw, C*T*P*P], grid_thw [1,3], yx [gh*gw, 2]) with rows in merge-window order.
    """

    torch = _torch()
    c, h, w = pixels.shape
    gh, gw = h // patch, w // patch
    frames = pixels.unsqueeze(0).expand(temporal, c, h, w)
    x = frames.reshape(1, temporal, c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = x.reshape(gh * gw, c * temporal * patch * patch)
    bh, bw, mh, mw = torch.meshgrid(
        torch.arange(gh // merge), torch.arange(gw // merge), torch.arange(merge), torch.arange(merge), indexing="ij"
    )
    yx = torch.stack(((bh * merge + mh).reshape(-1), (bw * merge + mw).reshape(-1)), dim=-1)
    grid = torch.tensor([[1, gh, gw]], dtype=torch.long)
    return flat.contiguous(), grid, yx


def hy3d_segment(block_input_s: Any, latent: Any, positions: Any, *, time_row: bool = True) -> dict[str, Any]:
    """One HY3D forward's pair rows from the parent's block-L input [1 + n, W] (row 0 = the time token).

    time_row=True (the unit's runtime contract): rows = [time row | n latent rows]; the time row's features
    are zero with feature_width 0, its position is TIME_ROW_POSITION, its parent row is block_input_s[0].
    time_row=False reproduces 1.0.0 (row 0 dropped).
    """

    torch = _torch()
    n, f = int(latent.shape[0]), int(latent.shape[1])
    if int(block_input_s.shape[0]) != n + 1:
        raise SystemExit(f"hy3d: block input has {int(block_input_s.shape[0])} rows, expected 1 time row + {n} latent rows")
    if not time_row:
        return {"features": latent, "widths": torch.full((n,), f, dtype=torch.long), "parent": block_input_s[1:],
                "positions": positions, "time_row": torch.zeros(n, dtype=torch.bool)}
    feats = torch.cat((torch.zeros(1, f, dtype=latent.dtype), latent), dim=0)
    widths = torch.cat((torch.zeros(1, dtype=torch.long), torch.full((n,), f, dtype=torch.long)))
    pos = torch.cat((torch.tensor([TIME_ROW_POSITION], dtype=torch.float32), positions.to(torch.float32)), dim=0)
    flag = torch.zeros(n + 1, dtype=torch.bool)
    flag[0] = True
    return {"features": feats, "widths": widths, "parent": block_input_s, "positions": pos, "time_row": flag}


def image_positions(yx: Any, gh: int, gw: int) -> Any:
    """native_media image/ocr convention: (0, y/(gh-1), x/(gw-1), 0)."""

    torch = _torch()
    pos = torch.zeros(yx.shape[0], 4, dtype=torch.float32)
    pos[:, 1] = yx[:, 0].float() / max(1, gh - 1)
    pos[:, 2] = yx[:, 1].float() / max(1, gw - 1)
    return pos


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------


class PairWriter:
    def __init__(self, unit: str, out: Path, *, feature_width: int, parent_width: int, parent_dtype: Any):
        self.unit = unit
        self.out = out / unit
        self.feature_width = int(feature_width)
        self.parent_width = int(parent_width)
        self.parent_dtype = parent_dtype
        self.chunks: dict[str, list[Any]] = {k: [] for k in ("features", "feature_width", "parent_rows", "type_ids", "sigma", "positions", "forward", "token", "group", "held", "time_row")}
        self.forwards: list[dict[str, Any]] = []

    def add_forward(self, *, prompt: str, features: Any, parent_rows: Any, type_ids: Any, sigma: Any, positions: Any,
                    group: int, held: bool, meta: dict[str, Any], time_row: Any = None) -> int:
        torch = _torch()
        n = int(features.shape[0])
        if parent_rows.shape != (n, self.parent_width):
            raise SystemExit(f"{self.unit}: parent rows {tuple(parent_rows.shape)} != ({n}, {self.parent_width})")
        if features.shape[1] > self.feature_width:
            raise SystemExit(f"{self.unit}: feature width {features.shape[1]} > {self.feature_width}")
        for name, t in (("features", features), ("parent_rows", parent_rows), ("positions", positions)):
            if not bool(torch.isfinite(t.float()).all()):
                raise SystemExit(f"{self.unit}: non-finite {name} in forward {len(self.forwards)}")
        padded = torch.zeros(n, self.feature_width, dtype=torch.float32)
        padded[:, : features.shape[1]] = features.float()
        index = len(self.forwards)
        widths = torch.full((n,), int(features.shape[1]), dtype=torch.long) if not isinstance(meta.get("_widths"), torch.Tensor) else meta.pop("_widths")
        self.chunks["features"].append(padded)
        self.chunks["feature_width"].append(widths)
        self.chunks["parent_rows"].append(parent_rows.to(self.parent_dtype).contiguous())
        self.chunks["type_ids"].append(type_ids.to(torch.long))
        self.chunks["sigma"].append(sigma.to(torch.float32))
        self.chunks["positions"].append(positions.to(torch.float32))
        self.chunks["forward"].append(torch.full((n,), index, dtype=torch.long))
        self.chunks["token"].append(torch.arange(n, dtype=torch.long))
        self.chunks["group"].append(torch.full((n,), int(group), dtype=torch.long))
        self.chunks["held"].append(torch.full((n,), bool(held), dtype=torch.bool))
        flag = torch.zeros(n, dtype=torch.bool) if time_row is None else time_row.to(torch.bool).reshape(-1)
        if int(flag.numel()) != n:
            raise SystemExit(f"{self.unit}: time_row flags {int(flag.numel())} != rows {n}")
        if bool(flag.any()):
            # structural: the time row is the forward's FIRST row, there is one, and it carries no content
            if int(flag.sum()) != 1 or not bool(flag[0]):
                raise SystemExit(f"{self.unit}: forward {index}: a time row must be the single first row")
            if bool(padded[0].ne(0).any()) or int(widths[0]) != 0:
                raise SystemExit(f"{self.unit}: forward {index}: the time row must carry zero features (width 0)")
        self.chunks["time_row"].append(flag)
        self.forwards.append({"index": index, "prompt": prompt, "rows": n, "group": int(group), "held": bool(held),
                              "time_rows": int(flag.sum()), **meta})
        return index

    def write(self, body: dict[str, Any]) -> dict[str, Any]:
        torch = _torch()
        from safetensors.torch import save_file

        self.out.mkdir(parents=True, exist_ok=True)
        state = {k: torch.cat(v, dim=0).contiguous() for k, v in self.chunks.items()}
        path = self.out / "parent_capture.safetensors"
        tmp = path.with_suffix(".tmp")
        save_file(state, str(tmp))
        os.replace(tmp, path)
        held = state["held"]
        types = state["type_ids"]
        receipt = {
            "schema": SCHEMA,
            "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__)},
            "unit": self.unit,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file": str(path),
            "file_sha256": sha256_file(path),
            "tensor_sha256": {k: tensor_digest(v) for k, v in sorted(state.items())},
            "shapes": {k: list(v.shape) for k, v in state.items()},
            "rows": int(held.numel()),
            "rows_fit": int((~held).sum()),
            "rows_held": int(held.sum()),
            "rows_by_type": {str(int(t)): int((types == t).sum()) for t in torch.unique(types)},
            "rows_time": int(state["time_row"].sum()),
            "rows_time_held": int((state["time_row"] & held).sum()),
            "groups": int(torch.unique(state["group"]).numel()),
            "held_groups": sorted(int(g) for g in torch.unique(state["group"][held])),
            "forwards": self.forwards,
            "parent_row_stats": {
                "rms": float(state["parent_rows"].float().pow(2).mean().sqrt()),
                "row_norm_median": float(state["parent_rows"].float().norm(dim=1).median()),
            },
            "capability_claim": "none; parent-side pair rows for a chart fit, a diagnostic input",
            **body,
        }
        jpath = self.out / "parent_capture.json"
        jpath.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str), encoding="utf-8")
        (self.out / "parent_capture.json.sha256").write_text(sha256_file(jpath) + "\n", encoding="utf-8")
        log(f"{self.unit}: wrote {path} rows {receipt['rows']} (fit {receipt['rows_fit']}, held {receipt['rows_held']})")
        return receipt


# --------------------------------------------------------------------------
# QIMG
# --------------------------------------------------------------------------


def run_qimg(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    from safetensors import safe_open

    prompts: dict[str, dict[str, Any]] = {}
    inputs_files = []
    for path in sorted(QIMG_ROOT.glob("encoded_batch*/capture_inputs.json")):
        inputs_files.append({"path": str(path), "sha256": sha256_file(path)})
        for entry in json.loads(path.read_text())["entries"]:
            prompts[entry["id"]] = entry
    batches = [int(b) for b in args.batches.split(",")]
    files = []
    for b in batches:
        files.extend(sorted((QIMG_ROOT / f"capture_batch{b}").glob(f"b{b}_*_sigma*.safetensors")))
    writer = PairWriter("qimg_mmdit_L30_img", Path(args.out), feature_width=64, parent_width=3072, parent_dtype=torch.bfloat16)
    sample_groups: dict[str, int] = {}
    sources = []
    indices_report = None
    for path in files:
        m = re.match(r"(b\d+_(fit|held)_.+)_sigma(\d+)\.safetensors", path.name)
        if not m:
            continue
        sample, split = m.group(1), m.group(2)
        entry = prompts[sample]
        group = sample_groups.setdefault(sample, len(sample_groups))
        with safe_open(str(path), "pt") as h:
            model_input = h.get_tensor("model_input").float()
            img_in = h.get_tensor("img_in")[0]
            sigma = float(h.get_tensor("sigma").reshape(-1)[0])
            idx = h.get_tensor("image_patch_indices")
        feats = qimg_pack(model_input)[0]
        t, gh, gw = model_input.shape[2], model_input.shape[3] // 2, model_input.shape[4] // 2
        if feats.shape[0] != img_in.shape[0]:
            raise SystemExit(f"{path}: {feats.shape[0]} packed tokens vs {img_in.shape[0]} img rows")
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        yx = torch.stack((ys.reshape(-1), xs.reshape(-1)), dim=-1).repeat(t, 1)
        if indices_report is None:
            # image_patch_indices are the parent's (t, h, w) RoPE ids per img row; the packed order is
            # row-major (t, h, w) iff they are monotone in that order
            dh = (idx[1:, 1] - idx[:-1, 1])
            dw = (idx[1:, 2] - idx[:-1, 2])
            indices_report = {
                "image_patch_indices_first": idx[:3].tolist(),
                "image_patch_indices_last": idx[-2:].tolist(),
                "h_nondecreasing": bool((dh >= 0).all()),
                "w_steps_within_row": sorted(set(int(v) for v in dw.unique())),
                "reading": "rows advance w fastest then h, i.e. the process_img (t, h/2, w/2) row-major order the packing assumes",
            }
        pos = image_positions(yx, gh, gw)
        writer.add_forward(
            prompt=entry["prompt"], features=feats, parent_rows=img_in, type_ids=torch.full((feats.shape[0],), TYPE_IDS["image"]),
            sigma=torch.full((feats.shape[0],), sigma), positions=pos, group=group, held=(split == "held"),
            meta={"sample": sample, "split": split, "sigma": sigma, "source_file": str(path), "prompt_sha256": entry.get("prompt_sha256")},
        )
        sources.append({"file": str(path), "sha256": sha256_file(path)})
    return writer.write({
        "source": "qimg broad capture sep07 (Codex), block 30 img_in = block input; model_input = DiT input x_t",
        "batches": batches,
        "captures": sources,
        "capture_inputs": inputs_files,
        "feature": {"width": 64, "layout": "ComfyUI QwenImage.process_img (c, ph, pw) of model_input", "p_src": "img_in.weight [3072, 64] + bias",
                    "packing_check": indices_report},
        "parent": {"block": 30, "width": 3072, "dtype": "bf16 as captured"},
        "positions": "image thwr_v1: (0, y/(gh-1), x/(gw-1), 0) over the 32x32 token grid",
        "trunk_prompt": "the parent's own prompt string (capture_inputs.json)",
        "holdout": "by prompt: the capture's own 'held' split (2 objects per batch)",
        "deviations": ["the parent's bf16 capture is the target (fp8 weights widened); packing order (c,ph,pw) read from comfy process_img source, not hook-verified"],
    })


# --------------------------------------------------------------------------
# HY3D
# --------------------------------------------------------------------------


def run_hy3d(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    from safetensors import safe_open

    captions: dict[str, str] = {}
    for path in sorted(QIMG_ROOT.glob("encoded_batch*/capture_inputs.json")):
        for entry in json.loads(path.read_text())["entries"]:
            captions[entry["image"]["sha256"]] = entry["prompt"]
    cap_path = HY3D_CAPTURE / "block-10" / "capture.safetensors"
    with safe_open(str(cap_path), "pt") as h:
        block_input = h.get_tensor("block_input")
        latent_xyz = h.get_tensor("latent_xyz").float()
        sigma_all = h.get_tensor("sigma").float()
    samples = sorted(HY3D_CAPTURE.glob("sample-*.safetensors"))
    if len(samples) != block_input.shape[0]:
        raise SystemExit(f"{len(samples)} sample files vs block_input batch {block_input.shape[0]}")
    verify = verify_hy3d_block0(args.verify_block0, samples) if args.verify_block0 else None
    keep_time = not bool(getattr(args, "no_time_row", False))
    writer = PairWriter("hy3d_dit_L10", Path(args.out), feature_width=64, parent_width=2048, parent_dtype=torch.float16)
    lo = latent_xyz.amin(dim=(0, 1))
    hi = latent_xyz.amax(dim=(0, 1))
    sources = []
    time_stats = []
    for s, path in enumerate(samples):
        meta = json.loads(path.with_suffix(".json").read_text())
        with safe_open(str(path), "pt") as h:
            latent = h.get_tensor("dit_latent_input")[0].float()
        caption = captions.get(meta["conditioning_sha256"])
        if caption is None:
            raise SystemExit(f"{path}: conditioning image sha {meta['conditioning_sha256']} matches no QIMG prompt image")
        xyz = (latent_xyz[s] - lo) / (hi - lo).clamp_min(1e-6)
        pos = torch.zeros(latent.shape[0], 4)
        pos[:, 0], pos[:, 1], pos[:, 2] = xyz[:, 2], xyz[:, 1], xyz[:, 0]
        seg = hy3d_segment(block_input[s], latent, pos, time_row=keep_time)
        n = int(seg["features"].shape[0])
        writer.add_forward(
            prompt=caption, features=seg["features"], parent_rows=seg["parent"], type_ids=torch.full((n,), TYPE_IDS["volume"]),
            sigma=torch.full((n,), float(sigma_all[s])), positions=seg["positions"], group=s, held=(meta["split"] == "held"),
            meta={"sample": path.name, "split": meta["split"], "sigma": float(sigma_all[s]), "object_sha256": meta["object_sha256"],
                  "conditioning_sha256": meta["conditioning_sha256"], "seed": meta.get("seed"), "_widths": seg["widths"]},
            time_row=seg["time_row"],
        )
        row0 = block_input[s, 0].float()
        time_stats.append({"sample": path.name, "row0_norm": float(row0.norm()),
                           "latent_row_norm_median": float(block_input[s, 1:].float().norm(dim=1).median()),
                           "cos_row0_vs_latent_mean": float(torch.nn.functional.cosine_similarity(row0, block_input[s, 1:].float().mean(0), dim=0))})
        sources.append({"file": str(path), "sha256": sha256_file(path)})
    if keep_time:
        parent_rows_text = "block_input[:, 0:]: row 0 = the parent's evolved TIME TOKEN (kept, first in each forward, time_row=True), rows 1: = latent tokens"
        time_block = {
            "kept": True,
            "contract": "runtime_requirements.hy3d_time_token_row of the hy3d_dit_L10 unit (s7_unit_build >= 1.2.0)",
            "layout": "each forward = [1 time row | 4096 latent rows], all type 7, time row FIRST",
            "time_row_parent_row": "block_input[s, 0] (the time token after blocks 0..9, what block 10 self-attends over)",
            "time_row_features": "zeros, feature_width 0",
            "time_row_position": list(TIME_ROW_POSITION),
            "time_row_sigma": "the segment sigma (the same value as its latent rows)",
            "per_sample_diagnostics": time_stats,
        }
        time_deviation = "the time token is paired at ONE sigma (0.5) only: 4 time rows (2 fit + 2 held objects); the chart's map of the trunk time row at other sigmas is an extrapolation"
    else:
        parent_rows_text = "block_input[:, 1:] (row 0 = the time token, dropped: --no-time-row, the 1.0.0 layout)"
        time_block = {"kept": False, "why": "--no-time-row"}
        time_deviation = "the parent's time token (row 0) is not a trunk row (--no-time-row)"
    return writer.write({
        "source": "Codex hy3d-native dense-child capture of block 10 (4 objects x 4096 latent tokens, one sigma each)",
        "capture": {"file": str(cap_path), "sha256": sha256_file(cap_path), "capture_run": str(HY3D_CAPTURE / "capture-run.json"),
                    "capture_run_sha256": sha256_file(HY3D_CAPTURE / "capture-run.json")},
        "samples": sources,
        "feature": {"width": 64, "layout": "dit_latent_input row (the x_embedder input); the time row has none", "p_src": "model.x_embedder.weight [2048, 64] + bias"},
        "parent": {"block": 10, "width": 2048, "rows": parent_rows_text, "dtype": "fp16 as captured"},
        "time_token": time_block,
        "block0_check": verify,
        "positions": "volume thwr_v1: (z, y, x) of latent_xyz min-max normalised to [0,1] over the capture, r = 0; time row TIME_ROW_POSITION",
        "trunk_prompt": "the caption (QIMG prompt) of the conditioning image the parent saw through DINO",
        "holdout": "by object: the capture's own fit/held split (2 + 2); each object's time row goes with its object",
        "deviations": [
            "the image-conditioned parent (DINO cross-attention) is paired with the image's CAPTION on the trunk side",
            time_deviation,
            "16,384 latent rows at one sigma (0.5): the only per-token block-10 capture on the box",
        ],
    })


def verify_hy3d_block0(block0_dir: str, samples: list[Path]) -> dict[str, Any]:
    """Row 0 of the block input is the time token and rows 1: are x_embedder(dit_latent_input)."""

    torch = _torch()
    from safetensors import safe_open

    root = Path(block0_dir)
    cap = root / "block-00" / "capture.safetensors"
    if not cap.is_file():
        return {"checked": False, "reason": f"{cap} missing"}
    ours = {json.loads(p.with_suffix(".json").read_text())["object_sha256"]: p for p in samples}
    theirs = sorted(root.glob("sample-*.safetensors"))
    with safe_open(str(HY3D_WEIGHTS), "pt") as h:
        w = h.get_tensor("model.x_embedder.weight").float()
        b = h.get_tensor("model.x_embedder.bias").float()
    with safe_open(str(cap), "pt") as h:
        bi = h.get_tensor("block_input").float()
        tt = h.get_tensor("time_token").float()
    report = {"checked": True, "block0_capture": str(cap), "per_sample": []}
    for s, path in enumerate(theirs):
        meta = json.loads(path.with_suffix(".json").read_text())
        with safe_open(str(path), "pt") as h:
            latent = h.get_tensor("dit_latent_input")[0].float()
        emb = latent @ w.T + b
        rows = bi[s, 1:]
        rel = float((emb - rows).norm() / rows.norm())
        rel_t = float((bi[s, 0] - tt[s, 0]).norm() / tt[s, 0].norm().clamp_min(1e-30))
        report["per_sample"].append({"object_sha256": meta["object_sha256"], "same_object_as_block10_capture": meta["object_sha256"] in ours,
                                     "rel_l2_x_embedder_vs_rows1": rel, "rel_l2_row0_vs_time_token": rel_t})
    return report


# --------------------------------------------------------------------------
# H3
# --------------------------------------------------------------------------


def run_h3(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    from safetensors import safe_open

    media = OC / "media"
    if str(media) not in sys.path:
        sys.path.insert(0, str(media))
    import c16_h3_capture as cap  # noqa: E402
    from mediafold.c16_h3 import KIND_AUDIO, KIND_TEXT, KIND_VIDEO  # noqa: E402

    receipt_path = H3_CAPTURE / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    argv = dict(receipt["argv"])
    for key in ("video_pairs", "audio_pairs"):
        argv[key] = str(argv[key]).replace("/work/", str(OC) + "/")
    ns = argparse.Namespace(**argv)
    sigmas = [float(s) for s in receipt["sigmas"]]
    plan, notes = cap.build_stimuli(ns, sigmas)
    geometry_audio = notes.get("audio_pairs_geometry", {})
    hop = float(geometry_audio.get("hop_samples") or 0) or None
    rate = float(geometry_audio.get("sample_rate_hz") or 0) or None
    audio_frame_seconds = (hop / rate) if (hop and rate) else None
    fps_assumed = 24.0
    temporal_stride = int((notes.get("video_pairs_geometry") or {}).get("temporal_stride") or 4)
    writer = PairWriter("h3_dit_L25_curve6", Path(args.out), feature_width=96, parent_width=5376, parent_dtype=torch.bfloat16)
    parent_of = {int(a): int(p) for a, p in notes["audio_parent_of_pairs"].items()}
    held_mod = int(args.held_mod)
    checks = {"forwards": 0, "token_kind_equal": 0, "token_frame_equal": 0, "token_clip_equal": 0, "sigma_equal": 0}
    sources = []
    for step, row in enumerate(plan):
        path = H3_CAPTURE / f"step{step:02d}" / "L25.safetensors"
        with safe_open(str(path), "pt") as h:
            x_in = h.get_tensor("x_in")
            kind = h.get_tensor("token_kind")
            frame = h.get_tensor("token_frame")
            clip = h.get_tensor("token_clip")
            fwd = h.get_tensor("forward_index")
            sv = h.get_tensor("sigma_video")
            sa = h.get_tensor("sigma_audio")
        sources.append({"file": str(path), "sha256": sha256_file(path)})
        starts = torch.cat((torch.tensor([0]), torch.nonzero(fwd[1:] != fwd[:-1]).flatten() + 1))
        if len(starts) != len(row):
            raise SystemExit(f"step {step}: {len(starts)} forwards in the file vs {len(row)} rebuilt stimuli")
        for k, stim in enumerate(row):
            a = int(starts[k])
            b = a + stim.seq_len
            checks["forwards"] += 1
            checks["token_kind_equal"] += int(torch.equal(kind[a:b], stim.token_kind))
            checks["token_frame_equal"] += int(torch.equal(frame[a:b], stim.token_frame))
            checks["token_clip_equal"] += int(torch.equal(clip[a:b], stim.token_clip))
            checks["sigma_equal"] += int(abs(float(sv[k]) - stim.sigma_video) < 1e-6 and abs(float(sa[k]) - stim.sigma_audio) < 1e-6)
            if not (torch.equal(kind[a:b], stim.token_kind) and torch.equal(fwd[a:b], torch.full((stim.seq_len,), k))):
                raise SystemExit(f"step {step} forward {k}: rebuilt layout does not match the capture rows")
            k_kind = stim.token_kind
            audio_rows = h3_pack_audio(stim.audio.float())
            video_rows = h3_patchify_video(stim.video.float())
            n_a, n_v = int((k_kind == KIND_AUDIO).sum()), int((k_kind == KIND_VIDEO).sum())
            if audio_rows.shape[0] != n_a or video_rows.shape[0] != n_v:
                raise SystemExit(f"step {step} forward {k}: packed {audio_rows.shape[0]}/{video_rows.shape[0]} vs layout {n_a}/{n_v}")
            av_mask = k_kind != KIND_TEXT
            parent = x_in[a:b][av_mask]
            # packed sequence order is [text | audio | video]
            feats = torch.zeros(n_a + n_v, 96)
            feats[:n_a, :32] = audio_rows
            feats[n_a:, :96] = video_rows
            widths = torch.cat((torch.full((n_a,), 32, dtype=torch.long), torch.full((n_v,), 96, dtype=torch.long)))
            types = torch.cat((torch.full((n_a,), TYPE_IDS["audio"]), torch.full((n_v,), TYPE_IDS["video"])))
            sig = torch.cat((torch.full((n_a,), stim.sigma_audio), torch.full((n_v,), stim.sigma_video)))
            pos = torch.zeros(n_a + n_v, 4)
            ta = n_a // 2
            frames_a = torch.arange(ta).repeat(2).float()
            ch = torch.arange(2).repeat_interleave(ta).float()
            sec = audio_frame_seconds if audio_frame_seconds else 1.0 / 25.0
            pos[:n_a, 0] = (frames_a + 0.5) * sec
            pos[:n_a, 2] = sec
            pos[:n_a, 3] = ch
            t, gh, gw = stim.notes["video_tokens_thw"]
            tt, yy, xx = torch.meshgrid(torch.arange(t), torch.arange(gh), torch.arange(gw), indexing="ij")
            pos[n_a:, 0] = (tt.reshape(-1).float() * temporal_stride + 0.5 * temporal_stride) / fps_assumed
            pos[n_a:, 1] = yy.reshape(-1).float() / max(1, gh - 1)
            pos[n_a:, 2] = xx.reshape(-1).float() / max(1, gw - 1)
            pos[n_a:, 3] = 1.0 / fps_assumed
            parent_audio = parent_of.get(int(stim.notes["audio_clip_id"]), int(stim.audio_clip_id))
            group = int(parent_audio)
            writer.add_forward(
                prompt=PROMPT_H3, features=feats, parent_rows=parent, type_ids=types, sigma=sig, positions=pos,
                group=group, held=(group % held_mod == held_mod - 1),
                meta={"step": step, "forward_in_step": k, "video_clip": stim.clip_id, "audio_window": int(stim.notes["audio_clip_id"]),
                      "audio_parent": parent_audio, "sigma_video": stim.sigma_video, "sigma_audio": stim.sigma_audio,
                      "seed": int(stim.notes["seed"]), "rows_audio": n_a, "rows_video": n_v, "_widths": widths},
            )
    return writer.write({
        "source": "h3_blocks_v2 block-25 capture (Sep-07, Claude S6H): x_in rows of the AUDIO and VIDEO tokens",
        "capture_receipt": {"file": str(receipt_path), "sha256": sha256_file(receipt_path)},
        "captures": sources,
        "stimulus_rebuild": {
            "how": "c16_h3_capture.build_stimuli(receipt argv with /work -> $OC, receipt sigmas): seeded, CPU, no parent run",
            "checks": checks,
            "c16_h3_capture_sha256": sha256_file(media / "c16_h3_capture.py"),
            "c16_h3_sha256": sha256_file(media / "mediafold" / "c16_h3.py"),
            "receipt_source_sha256": receipt.get("source_sha256"),
        },
        "feature": {"video": {"width": 96, "layout": "comfy patchify_video 'nctrhpwq->nthwcrpq' (c, pt, ph, pw)", "p_src": "video_patch_proj [5376, 96] + bias"},
                    "audio": {"width": 32, "layout": "comfy pack_audio channel-major [ch*T, 32]", "p_src": "audio_patch_proj [5376, 32] + bias"}},
        "parent": {"block": 25, "width": 5376, "dtype": "bf16 as captured (fp32 compute on sm75)"},
        "positions": {"video": f"(t_s, y, x, 1/fps) with fps {fps_assumed} ASSUMED and temporal stride {temporal_stride}",
                      "audio": f"(t_s, 0, frame_seconds, channel) with frame_seconds {audio_frame_seconds} from the pair geometry"},
        "trunk_prompt": PROMPT_H3,
        "holdout": f"by audio parent clip (group % {held_mod} == {held_mod - 1}); each video clip appears in one forward per sigma",
        "text_rows": "NOT paired: the parent text stream was conditioned on SEEDED GAUSSIAN states (capture deviation 'text_conditioning'); no trunk token corresponds to a gaussian row. The H3 block reads type-0 trunk rows through this AV-fitted chart_in at runtime (recorded deviation).",
        "deviations": receipt.get("deviations", []) + [
            {"name": "trunk_prompt_generic", "what": f"the trunk reads the fixed prompt {PROMPT_H3!r}; the parent saw gaussian text"},
            {"name": "text_pairs_absent", "what": "chart_in is fitted on AV rows only"},
        ],
    })


# --------------------------------------------------------------------------
# ACE (GPU)
# --------------------------------------------------------------------------


def run_ace(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    media = OC / "media"
    if str(media) not in sys.path:
        sys.path.insert(0, str(media))
    from mediafold.ace_dit import AceStepDiTConfig, load_ace_dit  # noqa: E402
    from mediafold.c16_capture import latent_stimulus  # noqa: E402

    device = args.device
    config = AceStepDiTConfig.from_json()
    model, load_receipt = load_ace_dit(weights=ACE_WEIGHTS, config=config, num_layers=12, device=device, dtype=torch.float32)
    model.eval()
    pairs = OC / "media" / "pairs" / args.ace_pairs
    sigmas = [float(s) for s in args.sigmas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    writer = PairWriter("ace_dit_L12", Path(args.out), feature_width=384, parent_width=2048, parent_dtype=torch.float32)
    held_mod = int(args.held_mod)
    frame_seconds = 1920.0 / 48000.0
    timing = []
    for seed in seeds:
        for sigma in sigmas:
            stim = latent_stimulus(config, pairs, batch=args.batch, length=args.length, condition_length=32, sigma=sigma, seed=seed,
                                   row_offset=0, device=device, dtype=torch.float32)
            started = time.time()
            with torch.no_grad():
                inputs = model.block_inputs(stim.hidden_states, stim.timestep, stim.timestep_r, stim.encoder_hidden_states, stim.context_latents)
                x = inputs["hidden"]
                for layer in model.layers:
                    x = layer(x, inputs["temb"], encoder_hidden_states=inputs["context"], position_embeddings=(inputs["cos"], inputs["sin"]))
            torch.cuda.synchronize() if device.startswith("cuda") else None
            timing.append(time.time() - started)
            feats = ace_token_features(stim.context_latents.float(), stim.hidden_states.float()).cpu()
            x = x.float().cpu()
            clips = stim.clip_index
            for b in range(int(x.shape[0])):
                tokens = int(x.shape[1])
                clip_ids = clips[b].reshape(-1)
                clip = int(torch.mode(clip_ids).values)
                pos = torch.zeros(tokens, 4)
                pos[:, 0] = (torch.arange(tokens).float() * 2 + 1.0) * frame_seconds
                pos[:, 2] = 2 * frame_seconds
                writer.add_forward(
                    prompt=PROMPT_ACE, features=feats[b], parent_rows=x[b], type_ids=torch.full((tokens,), TYPE_IDS["music"]),
                    sigma=torch.full((tokens,), sigma), positions=pos, group=clip, held=(clip % held_mod == held_mod - 1),
                    meta={"seed": seed, "sigma": sigma, "batch_row": b, "clip": clip, "clips_in_row": sorted(set(int(c) for c in clip_ids.tolist()))},
                )
            log(f"ace seed {seed} sigma {sigma}: {x.shape[0]} x {x.shape[1]} rows, {timing[-1]:.2f}s")
    return writer.write({
        "source": "fresh capture: mediafold.ace_dit (transcribed ComfyUI ace_step15) prefix blocks 0..11, block-12 input",
        "weights": {"file": str(ACE_WEIGHTS), "load": load_receipt},
        "stimulus": {"rung": "latents", "pairs": str(pairs), "batch": args.batch, "length": args.length, "sigmas": sigmas, "seeds": seeds,
                     "condition": "seeded gaussian encoder states [B, 32, 2048] through the parent's condition_embedder (the regime of the c16_L12_v4 unit capture)",
                     "timestep_r": 0.0},
        "feature": {"width": 384, "layout": "proj_in input frames 2t,2t+1 as (c*2 + k), c over cat(context 128 | noisy 64)", "p_src": "decoder.proj_in.1.weight [2048,192,2] -> [2048,384] + bias"},
        "parent": {"block": 12, "width": 2048, "dtype": "fp32 compute and store"},
        "positions": "music thwr_v1: (t_s centre of the 2-frame token, 0, 0.08 s, channel 0) at 25 Hz latent frames",
        "trunk_prompt": PROMPT_ACE,
        "holdout": f"by clip (clip % {held_mod} == {held_mod - 1})",
        "timing_seconds": timing,
        "deviations": [
            {"name": "condition_gaussian", "what": "the parent's cross-attention context is seeded gaussian (as in the c16_L12_v4 capture the unit's coreset was fitted on); the trunk reads a fixed generic prompt"},
            {"name": "procedural_clips", "what": "C15 procedural pair corpus (ace_s_ctx0_merged), not S1 music stimuli"},
        ],
    })


# --------------------------------------------------------------------------
# QVL32 (GPU)
# --------------------------------------------------------------------------


def run_qvl(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    from PIL import Image
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    device = args.device
    config = Qwen3VLVisionConfig(out_hidden_size=5120)
    config._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(config)
    state = {}
    with safe_open(str(QVL_WEIGHTS), "pt") as h:
        for key in h.keys():
            if key.startswith("visual."):
                state[key[len("visual."):]] = h.get_tensor(key)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if unexpected or [m for m in missing if not m.endswith("inv_freq")]:
        raise SystemExit(f"qvl load: missing {missing[:5]} unexpected {unexpected[:5]}")
    model = model.to(device=device, dtype=torch.float32).eval()
    identity = qvl_unit_identity(state)
    captured: dict[str, Any] = {}

    def hook(module, hook_args):
        captured["x"] = hook_args[0].detach()

    handle = model.blocks[13].register_forward_pre_hook(hook)
    images = sorted(IMAGE_ROOT.glob("batch*/b*.png"))
    sides = [int(s) for s in args.sides.split(",")]
    writer = PairWriter("qvl32_vit_L13", Path(args.out), feature_width=1536, parent_width=1152, parent_dtype=torch.float32)
    held_mod = int(args.held_mod)
    sources = []
    for i, path in enumerate(images):
        sources.append({"file": str(path), "sha256": sha256_file(path)})
        rgb = Image.open(path).convert("RGB")
        for side in sides:
            img = rgb.resize((side, side), Image.BICUBIC)
            pixels = torch.tensor(list(img.getdata()), dtype=torch.float32).view(side, side, 3).permute(2, 0, 1) / 255.0
            pixels = (pixels - 0.5) / 0.5
            flat, grid, yx = qwen_vl_patchify(pixels)
            with torch.no_grad():
                model(flat.to(device), grid.to(device))
            x = captured.pop("x").float().cpu()
            if x.shape[0] != flat.shape[0]:
                raise SystemExit(f"{path}: block-13 rows {x.shape[0]} vs {flat.shape[0]} patches")
            gh, gw = int(grid[0, 1]), int(grid[0, 2])
            writer.add_forward(
                prompt=PROMPT_QVL, features=flat, parent_rows=x, type_ids=torch.full((flat.shape[0],), TYPE_IDS["ocr"]),
                sigma=torch.full((flat.shape[0],), -1.0), positions=image_positions(yx, gh, gw), group=i, held=(i % held_mod == held_mod - 1),
                meta={"image": str(path), "side": side, "grid_thw": grid[0].tolist()},
            )
        log(f"qvl image {i} {path.name}")
    handle.remove()
    return writer.write({
        "source": "fresh capture: transformers Qwen3VLVisionModel with visual.* of qwen3vl_32b_minimax_h3_bf16.safetensors (fp32 compute)",
        "weights": {"file": str(QVL_WEIGHTS), "visual_tensors": len(state), "config": config.to_dict()},
        "unit_identity": identity,
        "images": sources,
        "preprocess": {"sides": sides, "resize": "PIL bicubic to side x side", "normalise": "(x/255 - 0.5)/0.5 (Qwen3-VL processor mean/std 0.5 ASSUMED)",
                       "temporal": "frame duplicated to temporal_patch_size 2", "order": "Qwen-VL processor merge-window order"},
        "feature": {"width": 1536, "layout": "flattened (C=3, T=2, P=16, P=16) tubelet", "p_src": "visual.patch_embed.proj.weight [1152,3,2,16,16] -> [1152,1536] + bias"},
        "parent": {"block": 13, "width": 1152, "rows": "block-13 input (after patch_embed + interpolated pos_embed + blocks 0..12 with the 2-D RoPE)", "dtype": "fp32"},
        "positions": "ocr thwr_v1 (image convention): (0, y/(gh-1), x/(gw-1), 0)",
        "trunk_prompt": PROMPT_QVL,
        "type": "ocr (6); R0(b): the unit's spec type_ids [1, 6] collide with QIMG's image type 1",
        "holdout": f"by image (index % {held_mod} == {held_mod - 1})",
        "deviations": [{"name": "no_sigma", "what": "encoder rows carry no noise level (native_media_noise_levels None)"}],
    })


def qvl_unit_identity(state: dict[str, Any]) -> dict[str, Any]:
    """Which QVL32 unit tensors are byte-equal (after widening) to this file's visual.blocks.13 tensors."""

    torch = _torch()
    from safetensors import safe_open

    unit = OC / "composite/s7/units/qvl32/qvl32_vit_L13/charted_block.safetensors"
    if not unit.is_file():
        return {"checked": False}
    block = {k[len("blocks.13."):]: v for k, v in state.items() if k.startswith("blocks.13.")}
    matches = {}
    with safe_open(str(unit), "pt") as h:
        for key in h.keys():
            t = h.get_tensor(key).float()
            hit = [name for name, v in block.items() if tuple(v.shape) == tuple(t.shape) and torch.equal(v.float(), t)]
            if not hit and t.dim() == 2:
                qkv = block.get("attn.qkv.weight")
                if qkv is not None and t.shape[1] == qkv.shape[1]:
                    for part, rows in zip(("q", "k", "v"), qkv.float().chunk(3, dim=0)):
                        if tuple(rows.shape) == tuple(t.shape) and torch.equal(rows, t):
                            hit.append(f"attn.qkv.weight[{part}]")
            matches[key] = hit
    equal = sum(1 for v in matches.values() if v)
    return {"checked": True, "unit_file": str(unit), "unit_tensors": len(matches), "equal_to_file_block13": equal, "matches": matches}


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("qimg", "hy3d", "h3", "ace", "qvl"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(OC / "composite/s7/pairs"))
        s.add_argument("--held-mod", type=int, default=5)
        if name == "qimg":
            s.add_argument("--batches", default="1,2")
        if name == "hy3d":
            s.add_argument("--no-time-row", action="store_true", help="1.0.0 layout: drop the parent's row 0 (the time token)")
            s.add_argument("--verify-block0", default="/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/initial-gpu4-attempt2/capture-block0")
        if name in ("ace", "qvl"):
            s.add_argument("--device", default="cuda:0")
        if name == "ace":
            s.add_argument("--ace-pairs", default="ace_s_ctx0_merged")
            s.add_argument("--sigmas", default="0.3,0.5,0.75,0.95")
            s.add_argument("--seeds", default="0,1")
            s.add_argument("--batch", type=int, default=96)
            s.add_argument("--length", type=int, default=100)
        if name == "qvl":
            s.add_argument("--sides", default="448,512")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = {"qimg": run_qimg, "hy3d": run_hy3d, "h3": run_h3, "ace": run_ace, "qvl": run_qvl}[args.cmd]
    started = time.time()
    receipt = runner(args)
    log(f"{args.cmd}: done in {time.time() - started:.1f}s rows {receipt['rows']} file sha {receipt['file_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
