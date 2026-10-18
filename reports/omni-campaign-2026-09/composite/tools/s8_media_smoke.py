#!/usr/bin/env python3
"""s8_media_smoke.py -- S8-MEDIA: per-modality output smoke of OMNI1 (OC02-F-Q38-27BU-OMNI1), host side.

S8-MEDIA lane, Claude, 2026-10-13.  OMNI1 is the S5 child + five charted media blocks (ACE L12, H3 L25 curve6,
HY3D L10, QIMG L30, QVL32 ViT L13) + the C15/C11 interface rows + the C18 band.  This tool asks one question per
modality: when a native media segment goes in, what comes out of OMNI1's own media head, and what does it decode to?

  build      for each generative source, ONE held object/clip at 2-4 sigmas, its native segment laid out exactly as
             ``s7_host_rows`` / ``s7_omni1_check`` lay it out ([prompt | media rows], the source's latent-rung
             feature in the type's C15 feature band, thwr_v1 positions, per-row sigma), plus its references in the
             parent's own DiT latent space:
               clean     the clean latent (ACE / H3: the pair corpus clip; QIMG: solved exactly from the capture's
                         four shared-noise model inputs; HY3D: the ShapeVAE posterior mode x scale_factor)
               x_t       the noisy DiT input (what the v=0 / "do nothing" baseline decodes to)
               parent    the PARENT's own full-model output on the same input where it exists: ACE (fresh: the whole
                         24-layer ACE-1.5 DiT, mediafold.ace_dit, same seeded stimulus as the pair capture), QIMG (the
                         capture's own `prediction`), HY3D at sigma 0.5 (the capture's own `dit_output`).  H3: none
                         (no full-H3 output exists on the box; a 33B run is out of this lane's GPU budget).
             Every captured-sigma segment is checked byte-for-byte against the S7 pair set it came from.
  forward    OMNI1 loaded strict from sha-verified tensors (s7_omni1_check.Checkpoint / build_block / build_top),
             all 69 blocks in order (trunk blocks streamed host->GPU per pass with torch.func.functional_call,
             the five charted blocks resident), each segment run through the WHOLE composite, the emission read
             through the native media head (final norm -> native_media_head -> the type's C15 emission band) at
               final     the model's own output path (after trunk layers 32-63), and
               site      a diagnostic read right after the last charted block (trunk layer 31 + the five writes).
             The emission band is the parent's own output-projection space (P_out @ B_out, c15), so the emitted
             latent is the parent-style raw prediction; x0 = x_t + sign * sigma * raw with the sign of that
             parent's flow parameterisation (checked against the parent's own output where one exists).
  qvl        Qwen3-VL ViT rows -> a TEXT answer through the trunk: a held image packetised as the C11 rows expect
             (Qwen-VL 2x16x16x3 tubelets, (x/255-0.5)/0.5, merge-window order), placed in the user turn after the
             question, then greedy decode with the ledger bands masked (argmax over [0, ledger_band_begin)).
             Controls: the same chat with NO image rows, and with the image rows' features zeroed.  1.1.0: cached
             decode (the tree's _forward_trunk_with_cache semantics; the charted blocks write on the prefill's media
             rows and own no decoded text row), checked by a preflight: case 0's prefill and first-step logits vs a
             full no-cache recompute.  1.1.0 also clones the saved latents (1.0.0 died in save_file on shared refs).

Outputs under --out-root (default $OC/composite/s8/media_out):
  <modality>/latents.safetensors   every reference / emission in the parent's DiT latent layout (decode-ready
                                   after the parent's latent_format.process_out), + latents.json (the decode plan)
  qvl/answers.json                 verbatim answers per case
  s8_media_forward.json (+.sha256) the receipt; every number in it is a DIAGNOSTIC (owner policy 2026-10-06)

Refusals (exit 3) are structural only: OMNI1 not strict-equivalent, a pair file / capture whose sha changed, a
rebuilt segment that does not equal its pair rows, non-finite inputs or emissions.  The decoded media are made by
``s8_media_decode.py`` inside the ComfyUI runtime image (the parent VAEs).

Env: CHIMERA_OMNI_SRC=<tree ef1ffb9> PYTHONPATH=<tree>/src (the tree the OMNI1 check used); MEDIAFOLD_ROOT=$OC/media.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s8_media_smoke"
TOOL_VERSION = "1.1.0"
SCHEMA = "omni-campaign.s8_media_smoke.v1"
HERE = Path(__file__).resolve().parent
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))

OMNI1_DIR = Path("/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1")
PAIRS = OC / "composite/s7/pairs"
TAP_ROWS = OC / "composite/s7/tap_rows_s5"
ACE_WEIGHTS = Path("/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors")
QIMG_RUN = Path("/mnt/lexar/omni-campaign/qimg-broad-capture-sep07/run")
QIMG_IMAGES = Path("/mnt/lexar/omni-campaign/qimg-broad-calibration-sep07")
HY3D_CAPTURE = Path(
    "/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/dense-child-blocks/"
    "dense-child-block-10-1c1709d9-8afb-47f0-931f-35e01f90f125/capture-block10"
)
H3_CAPTURE = Path("/mnt/lexar/omni-campaign/media/h3_blocks_v2")
HY3D_SCALE_FACTOR = 1.0039506158752403  # comfy latent_formats.Hunyuan3Dv2_1.scale_factor (DiT space = z * sf)

UNITS = {"ace": "ace_dit_L12", "h3": "h3_dit_L25_curve6", "qimg": "qimg_mmdit_L30_img", "hy3d": "hy3d_dit_L10",
         "qvl": "qvl32_vit_L13"}
# x0 = x_t + SIGN * sigma * raw, raw = the parent's output projection (what the C15 emission rows reproduce):
#   ACE   comfy ace_step15 returns the decoder output unchanged, ModelSamplingFlow CONST: x0 = x - sigma*out  -> -1
#   QIMG  comfy QwenImage (FLUX model type) returns proj_out unchanged, CONST                           -> -1
#   HY3D  comfy hunyuan3dv2_1 returns -final_layer(...) (`output * (-1.0)`), CONST                      -> +1
#   H3    comfy minimax returns [-video_out, -slope_a * audio_out]; video x0 = x + sigma_v*raw, audio
#         x0 = x_a + sigma_a*raw_a (the audio raw is the audio stream's own velocity, before the slope)  -> +1
SIGN = {"ace": -1.0, "qimg": -1.0, "hy3d": 1.0, "h3_video": 1.0, "h3_audio": 1.0}
H3_MODEL_PY_SHA256 = "882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024"  # comfy/ldm/minimax/model.py (r5 image)
H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
H3_FRAME_RESCALE = 5.0 / 3.0
TYPE_IDS = {"image": 1, "video": 2, "audio": 3, "music": 4, "ocr": 6, "volume": 7}


class Refusal(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_s8", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


# ----------------------------------------------------------------------------- pure layout helpers (tested)

def ace_emission_to_latent(rows: Any) -> Any:
    """[n, 128] emission rows (index c*2+k, ConvTranspose1d k=s=2) -> [2n, 64] latent frames (frame 2t+k)."""
    n = int(rows.shape[0])
    return rows.reshape(n, 64, 2).permute(0, 2, 1).reshape(2 * n, 64)


def qimg_unpack(rows: Any, t: int, h: int, w: int, c: int = 16) -> Any:
    """inverse of s7_pairs_parent.qimg_pack: [t*(h/2)*(w/2), c*4] in (c, ph, pw) order -> [1, c, t, h, w]."""
    x = rows.reshape(1, t, h // 2, w // 2, c, 2, 2)
    x = x.permute(0, 4, 1, 2, 5, 3, 6)
    return x.reshape(1, c, t, h, w)


def h3_unpatchify_video(rows: Any, t: int, h: int, w: int, c: int = 24) -> Any:
    """comfy minimax unpatchify_video for patch (1,2,2): [t*h*w, c*4] -> [1, c, t, 2h, 2w]."""
    import torch
    x = rows.reshape(-1, t, h, w, c, 1, 2, 2)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t, h * 2, w * 2)


def h3_unpack_audio(rows: Any, ch: int = 2) -> Any:
    """comfy minimax unpack_audio: [ch*T, 32] channel-major -> [1, 32, ch, T]."""
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _h3_axis(dim: int, patch: int, sqrt_area: float) -> Any:
    import torch
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def h3_source_positions(text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int) -> Any:
    """The parent's (t, h, w) RoPE coordinates of a plain t2va packed sequence [text | audio | video], transcribed
    from comfy/ldm/minimax/model.py PackedLayout (sha256 H3_MODEL_PY_SHA256; no keyframes, no refs)."""
    import torch
    area = math.sqrt(latent_h * latent_w)
    hh, ww = torch.meshgrid(_h3_axis(latent_h, 2, area), _h3_axis(latent_w, 2, area), indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    w_grid = _h3_axis(latent_w, 2, area)
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len, dtype=torch.float64)
    cursor = float(text_len)
    audio = torch.zeros(audio_t * 2, 3, dtype=torch.float64)
    audio[:, 0] = (cursor + torch.arange(audio_t, dtype=torch.float64)).repeat(2)
    audio[:audio_t, 2] = float(w_grid[0])
    audio[audio_t:, 2] = float(w_grid[-1])
    spans = torch.tensor([H3_FRAME_RESCALE * H3_FRAME_PER_TOKEN[k % 5] for k in range(latent_t)], dtype=torch.float64)
    t_grid = cursor + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
    video = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
    video[:, :, 0] = t_grid[:, None]
    video[:, :, 1:] = frame[None]
    return torch.cat([text, audio, video.reshape(-1, 3)], dim=0)


def solve_clean_shared_noise(xs: Any, sigmas: list[float]) -> tuple[Any, Any, float]:
    """x_s = (1 - s) z + s n at k >= 2 sigmas with ONE shared noise draw -> (z, n, max relative residual).
    Least squares over the k sigmas (exact when the inputs are exactly the interpolates)."""
    import torch
    k = len(sigmas)
    a = torch.tensor([[1.0 - s, s] for s in sigmas], dtype=torch.float64)
    flat = xs.reshape(k, -1).to(torch.float64)
    sol = torch.linalg.lstsq(a, flat).solution
    z, n = sol[0].reshape(xs.shape[1:]), sol[1].reshape(xs.shape[1:])
    resid = (a @ sol - flat).norm(dim=1) / flat.norm(dim=1).clamp_min(1e-30)
    return z.to(torch.float32), n.to(torch.float32), float(resid.max())


def x0_from_raw(x_t: Any, raw: Any, sigma: Any, sign: float) -> Any:
    return x_t + float(sign) * sigma * raw


def latent_metrics(x_t: Any, clean: Any, raw: Any, sigma: Any, sign: float, parent_raw: Any = None) -> dict[str, Any]:
    """Diagnostics in the parent's DiT latent space.  `ideal_raw` is the raw output that would give x0 == clean."""
    import torch
    import torch.nn.functional as F

    def cos(a: Any, b: Any) -> float:
        return float(F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))

    def rel(a: Any, b: Any) -> float:
        return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))

    x0 = x0_from_raw(x_t, raw, sigma, sign)
    ideal = (clean - x_t) / (float(sign) * sigma)
    rec: dict[str, Any] = {
        "raw_finite": bool(torch.isfinite(raw).all()), "raw_nonzero": bool((raw != 0).any()), "raw_norm": float(raw.norm()),
        "ideal_raw_norm": float(ideal.norm()),
        "cos_raw_vs_ideal": cos(raw, ideal), "rel_l2_raw_vs_ideal": rel(raw, ideal),
        "rel_l2_x0_vs_clean": rel(x0, clean), "rel_l2_xt_vs_clean_v0_baseline": rel(x_t, clean),
        "cos_x0_vs_clean": cos(x0, clean),
    }
    d = (raw.double() * raw.double()).sum()
    g = float((raw.double() * ideal.double()).sum() / d) if float(d) > 0 else 0.0
    rec["oracle_scalar_gain"] = g
    rec["rel_l2_x0_vs_clean_at_oracle_gain"] = rel(x0_from_raw(x_t, raw * g, sigma, sign), clean)
    if parent_raw is not None:
        px0 = x0_from_raw(x_t, parent_raw, sigma, sign)
        rec.update({"cos_raw_vs_parent": cos(raw, parent_raw), "rel_l2_raw_vs_parent": rel(raw, parent_raw),
                    "rel_l2_x0_vs_parent_x0": rel(x0, px0), "parent_rel_l2_x0_vs_clean": rel(px0, clean),
                    "parent_cos_raw_vs_ideal": cos(parent_raw, ideal)})
    return rec


def best_sign(x_t: Any, clean: Any, parent_raw: Any, sigma: Any) -> dict[str, Any]:
    """Which flow sign makes the PARENT's own output reconstruct the clean latent (convention check)."""
    out = {}
    for s in (1.0, -1.0):
        x0 = x0_from_raw(x_t, parent_raw, sigma, s)
        out[f"{s:+.0f}"] = float((x0.double() - clean.double()).norm() / clean.double().norm())
    out["better"] = 1.0 if out["+1"] < out["-1"] else -1.0
    return out


# ----------------------------------------------------------------------------- pair sets

class PairSet:
    """The S7 pair set of one unit (light keys only), sha-checked against its receipt."""

    KEYS = ("features", "feature_width", "type_ids", "sigma", "positions", "forward", "time_row")

    def __init__(self, unit: str) -> None:
        import torch
        from safetensors import safe_open

        self.unit = unit
        self.path = PAIRS / unit / "parent_capture.safetensors"
        self.receipt_path = PAIRS / unit / "parent_capture.json"
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.sha256 = sha256_file(self.path)
        if self.sha256 != self.receipt["file_sha256"]:
            raise Refusal(f"{unit}: pair set {self.path} does not hash like its receipt")
        with safe_open(str(self.path), "pt") as h:
            keys = set(h.keys())
            self.t = {k: h.get_tensor(k) for k in self.KEYS if k in keys}
        if "time_row" not in self.t:
            self.t["time_row"] = torch.zeros(self.t["forward"].shape[0], dtype=torch.bool)
        self.forwards = {int(f["index"]): f for f in self.receipt["forwards"]}

    def rows(self, forward: int) -> dict[str, Any]:
        import torch
        idx = torch.nonzero(self.t["forward"] == int(forward)).flatten()
        a, b = int(idx[0]), int(idx[-1]) + 1
        if b - a != int(idx.numel()):
            raise Refusal(f"{self.unit}: forward {forward} rows not contiguous")
        return {k: v[a:b] for k, v in self.t.items()}

    def find(self, **match: Any) -> int:
        hits = [i for i, f in self.forwards.items() if all(_close(f.get(k), v) for k, v in match.items())]
        if len(hits) != 1:
            raise Refusal(f"{self.unit}: {len(hits)} forwards match {match}")
        return hits[0]

    def record(self) -> dict[str, Any]:
        return {"file": str(self.path), "sha256": self.sha256, "receipt": str(self.receipt_path),
                "receipt_sha256": sha256_file(self.receipt_path)}


def _close(a: Any, b: Any) -> bool:
    if isinstance(b, float):
        return a is not None and abs(float(a) - b) < 1e-6
    return a == b


def features_in_band(nm: int, feature_dim: int, band: tuple[int, int], rows: Any, mask: Any = None) -> Any:
    import torch
    feats = torch.zeros(1, nm, feature_dim, dtype=torch.float32)
    lo, hi = band
    if mask is None:
        feats[0, :, lo:hi] = rows[:, : hi - lo].to(torch.float32)
    else:
        feats[0, mask, lo:hi] = rows[mask][:, : hi - lo].to(torch.float32)
    return feats


def max_abs(a: Any, b: Any) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


# ----------------------------------------------------------------------------- segment builders

def build_ace(args: argparse.Namespace, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    s7p = ctx["s7p"]
    pairs = PairSet(UNITS["ace"])
    ctx["pair_records"]["ace"] = pairs.record()
    media = OC / "media"
    if str(media) not in sys.path:
        sys.path.insert(0, str(media))
    from mediafold.ace_dit import AceStepDiTConfig, load_ace_dit
    from mediafold.c16_capture import latent_stimulus

    stim_cfg = pairs.receipt["stimulus"]
    config = AceStepDiTConfig.from_json()
    device = args.device
    model, load_receipt = load_ace_dit(weights=ACE_WEIGHTS, config=config, num_layers=None, device=device, dtype=torch.float32)
    model.eval()
    segs = []
    for sigma in [float(s) for s in args.ace_sigmas.split(",")]:
        f = pairs.find(clip=int(args.ace_clip), seed=int(args.ace_seed), sigma=sigma)
        entry = pairs.forwards[f]
        if not entry["held"]:
            raise Refusal(f"ace forward {f} is not a held forward")
        stim = latent_stimulus(config, Path(stim_cfg["pairs"]), batch=int(stim_cfg["batch"]), length=int(stim_cfg["length"]),
                               condition_length=32, sigma=sigma, seed=int(args.ace_seed), row_offset=0, device=device,
                               dtype=torch.float32)
        b = int(entry["batch_row"])
        feats_rows = s7p.ace_token_features(stim.context_latents.float(), stim.hidden_states.float())[b].cpu()
        rows = pairs.rows(f)
        eq = max_abs(feats_rows, rows["features"][:, :384])
        if eq != 0.0:
            raise Refusal(f"ace forward {f}: rebuilt features differ from the pair set by {eq}")
        with torch.no_grad():
            parent = model(stim.hidden_states[b:b + 1], stim.timestep[b:b + 1], stim.timestep_r[b:b + 1],
                           stim.encoder_hidden_states[b:b + 1], stim.context_latents[b:b + 1]).float().cpu()
        x_t = stim.hidden_states[b:b + 1].float().cpu()            # [1, 100, 64]
        clean = stim.context_latents[b:b + 1, :, :64].float().cpu()  # the source channels carry the clean latent
        nm = int(rows["features"].shape[0])
        segs.append({
            "name": f"ace_s{int(round(sigma * 100)):03d}", "modality": "ace", "unit": UNITS["ace"], "pair_forward": f,
            "sigma": sigma, "prompt": entry["prompt"], "types": torch.full((1, nm), TYPE_IDS["music"], dtype=torch.long),
            "rows_features": rows["features"][:, :384].float(), "band_type": TYPE_IDS["music"],
            "positions": rows["positions"].float().view(1, nm, -1), "sigma_rows": rows["sigma"].float().view(1, nm),
            "time_row": torch.zeros(nm, dtype=torch.bool),
            "ref": {"x_t": x_t.transpose(1, 2).contiguous(), "clean": clean.transpose(1, 2).contiguous(),
                    "parent_raw": parent.transpose(1, 2).contiguous()},   # VAE layout [1, 64, T]
            "layout": {"latent": "[1, 64, 100] ACE-1.5 VAE latent (ACEAudio15: DiT space = VAE space)", "frames": 100},
            "checks": {"features_equal_pair_rows_max_abs": eq, "clip": int(args.ace_clip), "seed": int(args.ace_seed),
                       "batch_row": b, "context_note": "src_latents = the CLEAN latent, chunk masks 0 (latent_stimulus): the clean "
                       "target is literally inside the input features"},
        })
        log(f"ace sigma {sigma}: forward {f}, rows {nm}, parent |v| {float(parent.norm()):.2f}")
    ctx["ace_parent"] = {"weights": str(ACE_WEIGHTS), "load": load_receipt, "regime": "the pair capture's: seeded gaussian "
                         "encoder states, timestep = sigma, timestep_r = 0 (ComfyUI passes timestep_r = timestep)"}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return segs


def build_h3(args: argparse.Namespace, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    s7p = ctx["s7p"]
    pairs = PairSet(UNITS["h3"])
    ctx["pair_records"]["h3"] = pairs.record()
    media = OC / "media"
    if str(media) not in sys.path:
        sys.path.insert(0, str(media))
    import c16_h3_capture as cap
    from mediafold.c15_pairs import load_pair_set
    from mediafold.c16_h3 import audio_clips_from_pairs, flow_noise, matched_audio_frames, video_clips_from_pairs

    receipt = json.loads((H3_CAPTURE / "capture_receipt.json").read_text())
    argv = dict(receipt["argv"])
    for key in ("video_pairs", "audio_pairs"):
        argv[key] = str(argv[key]).replace("/work/", str(OC) + "/")
    ns = argparse.Namespace(**argv)
    sigmas = [float(s) for s in receipt["sigmas"]]
    plan, notes = cap.build_stimuli(ns, sigmas)
    # the clean clips, exactly as build_stimuli draws them
    video_clips, _ = video_clips_from_pairs(load_pair_set(ns.video_pairs), frames=ns.video_frames)
    latent_frames = int(video_clips[sorted(video_clips)[0]].shape[1])
    matched = matched_audio_frames(latent_frames)
    audio_frames = matched if int(ns.audio_frames) < 0 else (0 if int(ns.audio_frames) == 0 else int(ns.audio_frames))
    windows = int(ns.audio_windows) if audio_frames > 0 else 1
    audio_clips, _ = audio_clips_from_pairs(load_pair_set(ns.audio_pairs), frames=audio_frames, windows=windows)
    segs = []
    k = int(args.h3_forward)
    for step in [int(s) for s in args.h3_steps.split(",")]:
        stim = plan[step][k]
        f = pairs.find(step=step, forward_in_step=k)
        entry = pairs.forwards[f]
        if not entry["held"]:
            raise Refusal(f"h3 forward {f} is not held")
        v_id, a_id = stim.clip_id, int(stim.notes["audio_clip_id"])
        v_clean, a_clean = video_clips[v_id].float(), audio_clips[a_id].float()
        g = torch.Generator().manual_seed(int(stim.notes["seed"]))
        v_re = flow_noise(v_clean, stim.sigma_video, g).unsqueeze(0)
        a_re = flow_noise(a_clean, stim.sigma_audio, g).unsqueeze(0)
        if not (torch.equal(v_re, stim.video.float()) and torch.equal(a_re, stim.audio.float())):
            raise Refusal(f"h3 step {step} forward {k}: the clean clips do not reproduce the stimulus")
        audio_rows = s7p.h3_pack_audio(stim.audio.float())
        video_rows = s7p.h3_patchify_video(stim.video.float())
        n_a, n_v = int(audio_rows.shape[0]), int(video_rows.shape[0])
        rows = pairs.rows(f)
        nm = n_a + n_v
        rebuilt = torch.zeros(nm, 96)
        rebuilt[:n_a, :32] = audio_rows
        rebuilt[n_a:, :96] = video_rows
        eq = max_abs(rebuilt, rows["features"][:, :96])
        if eq != 0.0:
            raise Refusal(f"h3 forward {f}: rebuilt features differ from the pair set by {eq}")
        t, gh, gw = stim.notes["video_tokens_thw"]
        types = rows["type_ids"].view(1, nm).to(torch.long)
        segs.append({
            "name": f"h3_step{step}", "modality": "h3", "unit": UNITS["h3"], "pair_forward": f, "sigma": stim.sigma_video,
            "sigma_audio": stim.sigma_audio, "prompt": entry["prompt"], "types": types,
            "rows_features": rows["features"][:, :96].float(), "feature_width": rows["feature_width"],
            "positions": rows["positions"].float().view(1, nm, -1), "sigma_rows": rows["sigma"].float().view(1, nm),
            "time_row": torch.zeros(nm, dtype=torch.bool), "h3_counts": (n_a, n_v), "h3_thw": (int(t), int(gh), int(gw)),
            "ref": {"video_x_t": stim.video.float(), "video_clean": v_clean.unsqueeze(0), "audio_x_t": stim.audio.float(),
                    "audio_clean": a_clean.unsqueeze(0)},
            "layout": {"video": f"[1, 24, {int(t)}, {2 * int(gh)}, {2 * int(gw)}] MiniMax H3 video latent (scale 1)",
                       "audio": f"[1, 32, 2, {n_a // 2}] MiniMax H3 audio latent"},
            "checks": {"features_equal_pair_rows_max_abs": eq, "clean_reproduces_stimulus": True, "video_clip": v_id,
                       "audio_window": a_id, "seed": int(stim.notes["seed"]), "sigma_video": stim.sigma_video,
                       "sigma_audio": stim.sigma_audio},
        })
        log(f"h3 step {step}: forward {f}, rows {n_a} audio + {n_v} video, sigma_v {stim.sigma_video:.4f} sigma_a {stim.sigma_audio:.4f}")
    return segs


def build_qimg(args: argparse.Namespace, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from safetensors import safe_open

    s7p = ctx["s7p"]
    pairs = PairSet(UNITS["qimg"])
    ctx["pair_records"]["qimg"] = pairs.record()
    sample = args.qimg_sample
    batch = sample.split("_")[0][1:]
    enc_path = QIMG_RUN / f"encoded_batch{batch}" / f"{sample}.safetensors"
    with safe_open(str(enc_path), "pt") as h:
        model_inputs = h.get_tensor("model_inputs").float()
        enc_sigmas = [float(s) for s in h.get_tensor("sigma").tolist()]
    clean, noise, resid = solve_clean_shared_noise(model_inputs, enc_sigmas)
    ctx.setdefault("qimg_clean_solve", {})[sample] = {"encoded": str(enc_path), "sha256": sha256_file(enc_path),
                                                      "sigmas": enc_sigmas, "max_rel_residual": resid,
                                                      "recipe": "x_s = s*n + (1-s)*Wan21.process_in(VAE.encode(pixels)), one shared n"}
    image = QIMG_IMAGES / f"batch{batch}" / f"{sample}.png"
    segs = []
    picks = [int(s) for s in args.qimg_sigma_files.split(",")]
    for idx in picks:
        cap_path = QIMG_RUN / f"capture_batch{batch}" / f"{sample}_sigma{idx:02d}.safetensors"
        with safe_open(str(cap_path), "pt") as h:
            x_t = h.get_tensor("model_input").float()
            parent = h.get_tensor("prediction").float()
            sigma = float(h.get_tensor("sigma").reshape(-1)[0])
        f = pairs.find(sample=sample, sigma=sigma)
        entry = pairs.forwards[f]
        if not entry["held"]:
            raise Refusal(f"qimg forward {f} is not held")
        rows = pairs.rows(f)
        packed = s7p.qimg_pack(x_t)[0]
        eq = max_abs(packed, rows["features"][:, :64])
        if eq != 0.0:
            raise Refusal(f"qimg forward {f}: packed model_input differs from the pair set by {eq}")
        nm = int(packed.shape[0])
        segs.append({
            "name": f"qimg_s{int(round(sigma * 100)):03d}", "modality": "qimg", "unit": UNITS["qimg"], "pair_forward": f,
            "sigma": sigma, "prompt": entry["prompt"], "types": torch.full((1, nm), TYPE_IDS["image"], dtype=torch.long),
            "rows_features": rows["features"][:, :64].float(), "positions": rows["positions"].float().view(1, nm, -1),
            "sigma_rows": rows["sigma"].float().view(1, nm), "time_row": torch.zeros(nm, dtype=torch.bool),
            "ref": {"x_t": x_t, "clean": clean.reshape(x_t.shape), "parent_raw": parent},
            "layout": {"latent": f"{list(x_t.shape)} Qwen-Image DiT space = Wan21.process_in(VAE latent)", "image": str(image)},
            "checks": {"features_equal_pair_rows_max_abs": eq, "capture": str(cap_path), "capture_sha256": sha256_file(cap_path),
                       "xt_vs_solved_interpolate_rel": float(((1 - sigma) * clean.reshape(x_t.shape) + sigma * noise.reshape(x_t.shape) - x_t).norm() / x_t.norm())},
        })
        log(f"qimg {sample} sigma {sigma}: forward {f}, rows {nm}")
    return segs


def build_hy3d(args: argparse.Namespace, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from safetensors import safe_open

    pairs = PairSet(UNITS["hy3d"])
    ctx["pair_records"]["hy3d"] = pairs.record()
    sample = args.hy3d_sample
    path = HY3D_CAPTURE / sample
    with safe_open(str(path), "pt") as h:
        x_cap = h.get_tensor("dit_latent_input")[0].float()   # [4096, 64] DiT space at the capture sigma
        noise = h.get_tensor("dit_noise")[0].float()
        parent = h.get_tensor("dit_output")[0].float()       # the parent's raw final-layer output (comfy negates it)
        post = h.get_tensor("posterior_mode")[0].float()
    meta = json.loads(path.with_suffix(".json").read_text())
    clean = post * HY3D_SCALE_FACTOR
    f = pairs.find(sample=sample)
    entry = pairs.forwards[f]
    if not entry["held"]:
        raise Refusal(f"hy3d forward {f} is not held")
    rows = pairs.rows(f)
    cap_sigma = float(entry["sigma"])
    tr = rows["time_row"].to(torch.bool)
    if int(tr.sum()) != 1 or not bool(tr[0]):
        raise Refusal("hy3d: the pair forward has no leading time row")
    eq = max_abs(x_cap, rows["features"][1:, :64])
    if eq != 0.0:
        raise Refusal(f"hy3d forward {f}: dit_latent_input differs from the pair rows by {eq}")
    interp = (1.0 - cap_sigma) * clean + cap_sigma * noise
    interp_rel = float((interp - x_cap).norm() / x_cap.norm())
    nm = int(rows["features"].shape[0])
    segs = []
    for sigma in [float(s) for s in args.hy3d_sigmas.split(",")]:
        captured = abs(sigma - cap_sigma) < 1e-9
        x_t = x_cap if captured else (1.0 - sigma) * clean + sigma * noise
        feats = torch.zeros(nm, 64)
        feats[1:] = x_t
        segs.append({
            "name": f"hy3d_s{int(round(sigma * 100)):03d}", "modality": "hy3d", "unit": UNITS["hy3d"], "pair_forward": f,
            "sigma": sigma, "prompt": entry["prompt"], "types": torch.full((1, nm), TYPE_IDS["volume"], dtype=torch.long),
            "rows_features": feats, "positions": rows["positions"].float().view(1, nm, -1),
            "sigma_rows": torch.full((1, nm), sigma), "time_row": tr.clone(),
            "ref": {"x_t": x_t.t().unsqueeze(0).contiguous(), "clean": clean.t().unsqueeze(0).contiguous(),
                    **({"parent_raw": parent.t().unsqueeze(0).contiguous()} if captured else {})},
            "layout": {"latent": "[1, 64, 4096] Hunyuan3D-2.1 ShapeVAE latent x scale_factor (DiT space); comfy layout [B, C, N]"},
            "checks": {"features_equal_pair_rows_max_abs": eq if captured else None, "captured_sigma": captured,
                       "capture_sigma": cap_sigma, "x_t_rebuilt_from_posterior_and_noise": not captured,
                       "capture_interpolate_rel_l2": interp_rel, "object_sha256": meta.get("object_sha256"),
                       "time_row": "row 0: zero features, TIME_ROW_POSITION, the segment sigma (unit contract)"},
        })
        log(f"hy3d {sample} sigma {sigma}: forward {f}, rows {nm} (captured={captured})")
    return segs


BUILDERS = {"ace": build_ace, "h3": build_h3, "qimg": build_qimg, "hy3d": build_hy3d}


# ----------------------------------------------------------------------------- the engine

def _CachedStep(blk: Any) -> Any:
    """nn.Module wrapper so torch.func.functional_call can drive the block's cached attention + FFN."""
    import torch
    from torch import nn

    class CachedStep(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.blk = inner

        def forward(self, x: Any, cache: Any, valid: Any) -> tuple[Any, Any]:
            attention, next_cache = self.blk.attention_output_with_cache(x, cache, valid)
            x = x + attention
            x = x + self.blk.feed_forward_output(x, valid)
            return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)), next_cache

    return CachedStep(blk)


class Engine:
    """OMNI1 in host RAM (trunk blocks) + on the GPU (the five charted blocks, top-level modules)."""

    def __init__(self, omni_dir: Path, *, device: str, dtype: Any, s7c: Any) -> None:
        import torch

        self.s7c = s7c
        self.device, self.dtype = device, dtype
        self.ck = s7c.Checkpoint(omni_dir, "omni1", verify=True)
        self.cfg = self.ck.cfg
        st = s7c.structural(self.ck)
        if not st["strict_equivalent"]:
            raise Refusal(f"OMNI1 is not strict-equivalent: missing {st['missing'][:3]} unexpected {st['unexpected'][:3]}")
        if str(self.cfg.residual_mode) != "none":
            raise Refusal(f"residual_mode {self.cfg.residual_mode!r}: this engine is the plain residual loop")
        self.structural = {k: st[k] for k in ("tensors", "strict_equivalent", "dtypes", "sha256_pinned_tensors")}
        self.kinds = st["block_kinds"]
        self.charted = [i for i, k in enumerate(self.kinds) if k == "charted"]
        manifest = json.loads((omni_dir / "composite_manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.bands = {int(e["type_id"]): (int(e["feature_band"][0]), int(e["feature_band"][1]))
                      for e in manifest["interface"]["entries"]}
        self.emission = {int(e["type_id"]): (int(e["emission_band"][0]), int(e["emission_band"][1]))
                         for e in manifest["interface"]["entries"]}
        self.placements = {int(p["layer_index"]): p for p in manifest["blocks"]["placements"]}
        log("building top-level modules (strict, sha-verified)")
        top = s7c.build_top(self.ck, device="cpu")
        self.meta = top["meta"]
        self.embed = top["embed"]
        self.bridge = top["native_media_bridge"].to(device=device, dtype=dtype)
        setattr(self.meta, "native_media_bridge", self.bridge)
        self.norm = top["norm"].to(device=device, dtype=dtype)
        self.head_media = top["native_media_head"].to(device=device, dtype=dtype)
        self.band_begin = int(self.cfg.ledger_band_begin())
        self.lm_head_cpu = top["lm_head_weight"]
        self.lm_head = None
        self.placeholder = int(self.cfg.ledger().range("media").begin)
        self.blocks: dict[int, Any] = {}
        self.wrappers: dict[int, Any] = {}
        started = time.time()
        for j in range(len(self.kinds)):
            on = device if j in self.charted else "cpu"
            self.blocks[j] = s7c.build_block(self.ck, j, device=on, dtype=dtype)
            if j % 8 == 0:
                log(f"block {j}/{len(self.kinds) - 1} built on {on} ({time.time() - started:.0f}s, rss {rss_gib():.1f} GiB)")
        expected = set(self.ck.views) | set(self.ck.tied)
        self.coverage = {"tensors_loaded": len(self.ck.loaded & expected), "tensors_expected": len(expected),
                         "complete": self.ck.loaded >= expected, "sha256_verified_on_read": self.ck.views.verify}
        if not self.coverage["complete"]:
            raise Refusal(f"strict load did not cover {len(expected - self.ck.loaded)} tensors")

    def head_weight(self) -> Any:
        if self.lm_head is None:
            self.lm_head = self.lm_head_cpu[: self.band_begin].to(device=self.device, dtype=self.dtype)
        return self.lm_head

    def prepare(self, ids: list[int], media: dict[str, Any] | None) -> dict[str, Any]:
        """embed + native media bridge + charted layout, exactly as s7_omni1_check prepares a segment."""
        import torch

        t_ids = torch.tensor([ids], dtype=torch.long)
        x = self.embed(t_ids).to(device=self.device, dtype=self.dtype)
        valid = torch.ones(t_ids.shape, dtype=torch.bool, device=self.device)
        rec: dict[str, Any] = {"valid": valid, "row_types": None, "levels": None, "positions": None, "media_count": 0}
        if media is not None:
            types = media["types"].to(self.device)
            valid_media = torch.ones(types.shape, dtype=torch.bool, device=self.device)
            noise = None if media.get("sigma_rows") is None else media["sigma_rows"].to(self.device)
            indices = media["indices"].to(self.device)
            x, count = type(self.meta)._add_native_media(self.meta, x, media["features"].to(self.device), types,
                                                         media["positions"].to(self.device), indices, valid_media, noise)
            if count != int(types.shape[1]):
                raise Refusal(f"media count {count} != {int(types.shape[1])}")
            row_types, levels = type(self.meta)._charted_layout(self.meta, t_ids.to(self.device), count, indices, types,
                                                                 valid_media, noise)
            positions = {spec.source: torch.zeros(1, len(ids), 3, device=self.device)
                         for spec in self.cfg.charted_blocks if spec.modulation == "h3_curve6"}
            if media.get("h3_positions") is not None:
                for key in positions:
                    positions[key] = media["h3_positions"].to(device=self.device, dtype=torch.float32).view(1, len(ids), 3)
            rec.update({"row_types": row_types, "levels": levels, "positions": positions or None, "media_count": count})
        rec["x"] = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        rec["input_finite"] = bool(torch.isfinite(rec["x"]).all())
        return rec

    def run(self, seqs: list[dict[str, Any]], *, site_after: int | None = None, quiet: bool = False) -> dict[str, Any]:
        """All blocks in order over every sequence.  Trunk blocks: host->GPU per pass (functional_call); charted
        blocks: resident, charted_output on media sequences, the identity call on text-only ones."""
        import torch
        from torch.func import functional_call

        started = time.time()
        writes: dict[str, list[dict[str, Any]]] = {}
        for j in range(len(self.kinds)):
            blk = self.blocks[j]
            if j in self.charted:
                for si, s in enumerate(seqs):
                    if s["row_types"] is None:
                        y = blk(s["x"], native_media_slot_mask=None, valid_media_mask=None,
                                enable_native_media_refinement=False, token_valid_mask=s["valid"])
                        same = bool(torch.equal(y, s["x"]))
                        writes.setdefault(str(j), []).append({"seq": si, "text_only_identity": same})
                        if not same:
                            raise Refusal(f"charted block {j} changed a text-only sequence")
                        continue
                    w = blk.charted_output(s["x"], s["row_types"], s["levels"], s["valid"], s["positions"])
                    wf = w.float()
                    if not bool(torch.isfinite(wf).all()):
                        raise Refusal(f"charted block {j}: non-finite write on sequence {si}")
                    owned = blk.charted.owned_rows(s["row_types"], s["valid"])
                    writes.setdefault(str(j), []).append({"seq": si, "rows_owned": 0 if owned is None else int(owned.sum()),
                                                          "write_norm": float(wf.norm())})
                    s["x"] = torch.where(s["valid"].unsqueeze(-1), s["x"] + w, torch.zeros_like(s["x"]))
            else:
                state = {n: t.to(device=self.device, non_blocking=True)
                         for n, t in itertools.chain(blk.named_parameters(), blk.named_buffers())}
                for s in seqs:
                    y = functional_call(blk, state, (s["x"],), {"native_media_slot_mask": None, "valid_media_mask": None,
                                                                 "enable_native_media_refinement": False,
                                                                 "token_valid_mask": s["valid"]})
                    s["x"] = torch.where(s["valid"].unsqueeze(-1), y, torch.zeros_like(y))
                del state
            if site_after is not None and j == site_after:
                for s in seqs:
                    s["x_site"] = s["x"].clone()
            if not quiet and (j % 8 == 0 or j == len(self.kinds) - 1):
                fin = all(bool(torch.isfinite(s["x"]).all()) for s in seqs)
                log(f"block {j}/{len(self.kinds) - 1} ({self.kinds[j]}) done, finite {fin} ({time.time() - started:.0f}s"
                    + (f", cuda peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)" if self.device.startswith("cuda") else ")"))
        return {"charted_writes": writes, "seconds": time.time() - started}

    def run_cached(self, seqs: list[dict[str, Any]], caches: list[list[Any]] | None) -> list[list[Any]]:
        """The tree's ``_forward_trunk_with_cache`` (residual_mode none) block by block: trunk blocks streamed with
        functional_call over a wrapper of ``attention_output_with_cache`` + ``feed_forward_output``; charted blocks
        write on the media rows of a chunk that carries them (the prefill) and are the identity otherwise (as the
        tree's cached path treats them)."""
        import torch
        from torch.func import functional_call

        out: list[list[Any]] = [[None] * len(self.kinds) for _ in seqs]
        for j in range(len(self.kinds)):
            blk = self.blocks[j]
            if j in self.charted:
                for s in seqs:
                    if s["row_types"] is not None:
                        w = blk.charted_output(s["x"], s["row_types"], s["levels"], s["valid"], s["positions"])
                        s["x"] = torch.where(s["valid"].unsqueeze(-1), s["x"] + w, torch.zeros_like(s["x"]))
                continue
            wrap = self.wrappers.setdefault(j, _CachedStep(blk))
            state = {f"blk.{n}": t.to(device=self.device, non_blocking=True)
                     for n, t in itertools.chain(blk.named_parameters(), blk.named_buffers())}
            for si, s in enumerate(seqs):
                prev = None if caches is None else caches[si][j]
                s["x"], out[si][j] = functional_call(wrap, state, (s["x"], prev, s["valid"]))
            del state
        return out

    def media_emission(self, x: Any, media: dict[str, Any], plen: int) -> Any:
        import torch
        hidden = self.norm(x)
        nm = int(media["types"].shape[1])
        mh = hidden[:, plen:plen + nm]
        return self.head_media(mh, media["types"].to(self.device)).float().cpu()[0]   # [nm, feature_dim]


# ----------------------------------------------------------------------------- forward pass over the segments

def tokenizer_for_prompts() -> tuple[Any, str]:
    from tokenizers import Tokenizer

    tap = json.loads((TAP_ROWS / UNITS["ace"] / "paired_rows.json").read_text(encoding="utf-8"))
    path = tap["sequence_layout"]["tokenizer"]
    if sha256_file(path) != tap["sequence_layout"]["tokenizer_sha256"]:
        raise Refusal(f"tokenizer {path} changed since the host-row capture")
    return Tokenizer.from_file(path), path


def segment_media(seg: dict[str, Any], engine: Engine, prompt_ids: list[int]) -> dict[str, Any]:
    import torch

    nm = int(seg["types"].shape[1])
    feats = torch.zeros(1, nm, int(engine.cfg.native_feature_dim), dtype=torch.float32)
    types = seg["types"][0]
    for t in torch.unique(types).tolist():
        lo, hi = engine.bands[int(t)]
        m = (types == t) & ~seg["time_row"]
        if seg["modality"] == "h3":
            width = 32 if int(t) == TYPE_IDS["audio"] else 96
            if hi - lo != width:
                raise Refusal(f"h3 type {t}: band width {hi - lo} != {width}")
        feats[0, m, lo:hi] = seg["rows_features"][m][:, : hi - lo]
    media = {"features": feats, "types": seg["types"], "positions": seg["positions"], "sigma_rows": seg["sigma_rows"],
             "indices": torch.arange(len(prompt_ids), len(prompt_ids) + nm, dtype=torch.long).view(1, nm)}
    if seg["modality"] == "h3":
        n_a, n_v = seg["h3_counts"]
        t, gh, gw = seg["h3_thw"]
        pos = h3_source_positions(len(prompt_ids), t, 2 * gh, 2 * gw, n_a // 2)
        if int(pos.shape[0]) != len(prompt_ids) + nm:
            raise Refusal(f"h3 source positions {int(pos.shape[0])} rows != sequence {len(prompt_ids) + nm}")
        media["h3_positions"] = pos.to(torch.float32)
    return media


def unpack_emission(seg: dict[str, Any], rows: Any, engine: Engine) -> dict[str, Any]:
    """emission rows [nm, feature_dim] -> {stream: raw latent in the parent's layout}."""
    types = seg["types"][0]
    keep = ~seg["time_row"]
    if seg["modality"] == "ace":
        lo, hi = engine.emission[TYPE_IDS["music"]]
        lat = ace_emission_to_latent(rows[:, lo:hi])                      # [100, 64]
        return {"": lat.t().unsqueeze(0).contiguous()}
    if seg["modality"] == "qimg":
        lo, hi = engine.emission[TYPE_IDS["image"]]
        shape = seg["ref"]["x_t"].shape
        return {"": qimg_unpack(rows[:, lo:hi], int(shape[2]), int(shape[3]), int(shape[4]))}
    if seg["modality"] == "hy3d":
        lo, hi = engine.emission[TYPE_IDS["volume"]]
        return {"": rows[keep][:, lo:hi].t().unsqueeze(0).contiguous()}   # [1, 64, 4096]
    if seg["modality"] == "h3":
        n_a, n_v = seg["h3_counts"]
        t, gh, gw = seg["h3_thw"]
        la, ha = engine.emission[TYPE_IDS["audio"]]
        lv, hv = engine.emission[TYPE_IDS["video"]]
        if not (bool((types[:n_a] == TYPE_IDS["audio"]).all()) and bool((types[n_a:] == TYPE_IDS["video"]).all())):
            raise Refusal("h3 rows are not [audio | video]")
        return {"audio_": h3_unpack_audio(rows[:n_a, la:ha]), "video_": h3_unpatchify_video(rows[n_a:, lv:hv], t, gh, gw)}
    raise Refusal(f"no emission unpacker for {seg['modality']}")


def segment_metrics(seg: dict[str, Any], raws: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """latent-space diagnostics + the decode-ready tensors of one segment."""
    import torch

    out_t: dict[str, Any] = {}
    rec: dict[str, Any] = {}
    ref = seg["ref"]
    streams = [("", "x_t", "clean", "parent_raw", SIGN[seg["modality"]], seg["sigma"])] if seg["modality"] != "h3" else [
        ("video_", "video_x_t", "video_clean", None, SIGN["h3_video"], seg["sigma"]),
        ("audio_", "audio_x_t", "audio_clean", None, SIGN["h3_audio"], seg["sigma_audio"])]
    for prefix, kx, kc, kp, sign, sigma in streams:
        x_t, clean = ref[kx], ref[kc]
        parent = ref.get(kp) if kp else None
        out_t[f"{prefix}clean"] = clean
        out_t[f"{prefix}x_t"] = x_t
        if parent is not None:
            out_t[f"{prefix}parent_raw"] = parent
            out_t[f"{prefix}x0_parent"] = x0_from_raw(x_t, parent, sigma, sign)
            rec[f"{prefix}parent_sign_check"] = best_sign(x_t, clean, parent, sigma)
        for read in ("final", "site"):
            raw = raws[read][prefix]
            if tuple(raw.shape) != tuple(x_t.shape):
                raise Refusal(f"{seg['name']} {prefix}{read}: emitted {tuple(raw.shape)} vs latent {tuple(x_t.shape)}")
            out_t[f"{prefix}raw_omni_{read}"] = raw
            out_t[f"{prefix}x0_omni_{read}"] = x0_from_raw(x_t, raw, sigma, sign)
            rec[f"{prefix}{read}"] = latent_metrics(x_t, clean, raw, sigma, sign, parent)
            if not rec[f"{prefix}{read}"]["raw_finite"]:
                raise Refusal(f"{seg['name']} {prefix}{read}: non-finite emission")
        rec[f"{prefix}sigma"] = float(sigma)
        rec[f"{prefix}sign"] = float(sign)
    return rec, {k: v.float().clone().contiguous() for k, v in out_t.items()}


def run_forward(args: argparse.Namespace, engine: Engine, segs: list[dict[str, Any]], receipt: dict[str, Any]) -> None:
    import torch
    from safetensors.torch import save_file

    tok, tok_path = tokenizer_for_prompts()
    receipt["prompt_tokenizer"] = {"path": tok_path, "sha256": sha256_file(tok_path)}
    seqs, medias = [], []
    for seg in segs:
        prompt_ids = tok.encode(seg["prompt"], add_special_tokens=False).ids
        media = segment_media(seg, engine, prompt_ids)
        ids = prompt_ids + [engine.placeholder] * int(seg["types"].shape[1])
        state = engine.prepare(ids, media)
        if not state["input_finite"]:
            raise Refusal(f"{seg['name']}: non-finite bridge input")
        state["plen"] = len(prompt_ids)
        seqs.append(state)
        medias.append(media)
        seg["prompt_tokens"] = len(prompt_ids)
    last_charted = max(engine.charted)
    log(f"media pass: {len(seqs)} segments, {sum(int(s['x'].shape[1]) for s in seqs)} rows in total")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    run = engine.run(seqs, site_after=last_charted)
    receipt["media_pass"] = {"seconds": run["seconds"], "site_after_block": last_charted,
                             "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else None}
    by_mod: dict[str, dict[str, Any]] = {}
    for si, (seg, s, media) in enumerate(zip(segs, seqs, medias)):
        with torch.no_grad():
            final_rows = engine.media_emission(s["x"], media, s["plen"])
            site_rows = engine.media_emission(s["x_site"], media, s["plen"])
            logits_finite = bool(torch.isfinite(torch.nn.functional.linear(engine.norm(s["x"][:, -1:]), engine.head_weight())).all())
        raws = {"final": unpack_emission(seg, final_rows, engine), "site": unpack_emission(seg, site_rows, engine)}
        metrics, tensors = segment_metrics(seg, raws)
        own = [w for j, ws in run["charted_writes"].items() for w in ws
               if w.get("seq") == si and engine.placements[int(j)]["container"] == seg["unit"]]
        rec = {"segment": seg["name"], "unit": seg["unit"], "pair_forward": seg["pair_forward"], "sigma": seg["sigma"],
               "prompt_tokens": seg["prompt_tokens"], "media_rows": int(seg["types"].shape[1]),
               "time_rows": int(seg["time_row"].sum()), "input_finite": s["input_finite"],
               "final_hidden_finite": bool(torch.isfinite(s["x"]).all()), "last_position_logits_finite": logits_finite,
               "emission_final": {"finite": bool(torch.isfinite(final_rows).all()), "norm": float(final_rows.norm()),
                                  "nonzero": bool((final_rows != 0).any())},
               "emission_site": {"finite": bool(torch.isfinite(site_rows).all()), "norm": float(site_rows.norm())},
               "own_charted_write": own, "checks": seg["checks"], "layout": seg["layout"], "latent": metrics}
        if seg["modality"] == "h3":
            rec["h3_source_positions"] = {"from": f"comfy/ldm/minimax/model.py PackedLayout, transcribed (sha256 {H3_MODEL_PY_SHA256})",
                                          "text_len": seg["prompt_tokens"], "note": "the OMNI1 check passed zeros here"}
        mod = by_mod.setdefault(seg["modality"], {"segments": [], "tensors": {}})
        mod["segments"].append(rec)
        for k, v in tensors.items():
            mod["tensors"][f"{seg['name']}.{k}"] = v
        log(f"{seg['name']}: emission final |e| {rec['emission_final']['norm']:.1f} finite {rec['emission_final']['finite']}; "
            + "; ".join(f"{k}: x0 relL2 {v['rel_l2_x0_vs_clean']:.3f} (v=0 {v['rel_l2_xt_vs_clean_v0_baseline']:.3f}, cos ideal {v['cos_raw_vs_ideal']:.3f})"
                        for k, v in metrics.items() if isinstance(v, dict) and "rel_l2_x0_vs_clean" in v))
    out_root = Path(args.out_root)
    receipt["modalities"] = {}
    for mod, data in by_mod.items():
        d = out_root / mod
        d.mkdir(parents=True, exist_ok=True)
        lat = d / "latents.safetensors"
        save_file({k: v.clone().contiguous() for k, v in data["tensors"].items()}, str(lat))
        plan = {"schema": "omni-campaign.s8_media_latents.v1", "modality": mod, "file": str(lat), "sha256": sha256_file(lat),
                "keys": {k: list(v.shape) for k, v in data["tensors"].items()},
                "space": "the parent's DiT latent space; decode = vae.decode(latent_format.process_out(x))",
                "segments": [{"segment": r["segment"], "sigma": r["sigma"], "layout": r["layout"]} for r in data["segments"]]}
        (d / "latents.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
        receipt["modalities"][mod] = {"latents": str(lat), "latents_sha256": plan["sha256"], "segments": data["segments"]}
        log(f"{mod}: wrote {lat} ({len(data['tensors'])} tensors)")


# ----------------------------------------------------------------------------- QVL: vision rows -> a text answer

QVL_DEFAULT_IMAGES = "batch2/b2_held_aquarium_room.png,batch1/b1_fit_travel_bag.png"


def qvl_image_rows(path: Path, side: int, s7p: Any) -> dict[str, Any]:
    import torch
    from PIL import Image

    rgb = Image.open(path).convert("RGB").resize((side, side), Image.BICUBIC)
    pixels = torch.tensor(list(rgb.getdata()), dtype=torch.float32).view(side, side, 3).permute(2, 0, 1) / 255.0
    pixels = (pixels - 0.5) / 0.5
    flat, grid, yx = s7p.qwen_vl_patchify(pixels)
    gh, gw = int(grid[0, 1]), int(grid[0, 2])
    return {"flat": flat, "positions": s7p.image_positions(yx, gh, gw), "grid": [1, gh, gw], "sha256": sha256_file(path)}


def image_subject(path: Path) -> str | None:
    """The QIMG prompt that generated a calibration image (capture_inputs.json), for the content check."""
    for inputs in sorted(QIMG_RUN.glob("encoded_batch*/capture_inputs.json")):
        for entry in json.loads(inputs.read_text())["entries"]:
            img = entry.get("image") or {}
            if Path(str(img.get("path", ""))).name == path.name or entry.get("id") == path.stem:
                return entry.get("prompt")
    return None


def run_qvl(args: argparse.Namespace, engine: Engine, receipt: dict[str, Any], s7p: Any) -> None:
    """Cached greedy decode (the tree's _forward_trunk_with_cache semantics + charted writes on the prefill), with a
    decode-path preflight: case 0's cached prefill / first step logits vs the full no-cache recompute."""
    import torch

    sys.path.insert(0, str(Path(args.calibration_dir)))
    import oc_calib  # type: ignore
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    marker = "⁣IMG⁣"
    text = oc_calib.render_prompt(tok, [{"role": "user", "content": args.qvl_question + "\n" + marker}], thinking="off")
    if text.count(marker) != 1:
        raise Refusal("the chat template did not keep the image marker exactly once")
    pre, post = text.split(marker)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    plain_text = oc_calib.render_prompt(tok, [{"role": "user", "content": args.qvl_question}], thinking="off")
    plain_ids = tok(plain_text, add_special_tokens=False)["input_ids"]
    stops = {int(i) for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
             if i is not None and int(i) >= 0}
    band = engine.bands[TYPE_IDS["ocr"]]
    cases: list[dict[str, Any]] = []
    for rel in [p for p in args.qvl_images.split(",") if p]:
        path = QIMG_IMAGES / rel
        rows = qvl_image_rows(path, int(args.qvl_side), s7p)
        nm = int(rows["flat"].shape[0])
        if int(rows["flat"].shape[1]) != band[1] - band[0]:
            raise Refusal(f"qvl tubelet width {int(rows['flat'].shape[1])} != C11 band width {band[1] - band[0]}")
        for variant in ("image", "zeroed_features"):
            feats = torch.zeros(1, nm, int(engine.cfg.native_feature_dim))
            if variant == "image":
                feats[0, :, band[0]:band[1]] = rows["flat"]
            cases.append({"case": f"{path.stem}.{variant}", "image": str(path), "image_sha256": rows["sha256"],
                          "variant": variant, "subject_prompt": image_subject(path), "prefix": pre_ids, "suffix": post_ids,
                          "media": {"features": feats, "types": torch.full((1, nm), TYPE_IDS["ocr"], dtype=torch.long),
                                    "positions": rows["positions"].view(1, nm, -1), "sigma_rows": None,
                                    "indices": torch.arange(len(pre_ids), len(pre_ids) + nm).view(1, nm)},
                          "media_rows": nm, "grid_thw": rows["grid"]})
    cases.append({"case": "no_image", "variant": "no_image", "prefix": plain_ids, "suffix": [], "media": None, "media_rows": 0})
    head = engine.head_weight()

    def logits_of(x: Any) -> Any:
        return torch.nn.functional.linear(engine.norm(x[:, -1:]), head).float()[0, 0]

    def ids_of(c: dict[str, Any]) -> list[int]:
        return c["prefix"] + ([engine.placeholder] * c["media_rows"]) + c["suffix"]

    started = time.time()
    seqs = [engine.prepare(ids_of(c), c["media"]) for c in cases]
    caches = engine.run_cached(seqs, None)
    prefill_seconds = time.time() - started
    log(f"qvl prefill: {len(cases)} cases, rows {[int(s['x'].shape[1]) for s in seqs]} ({prefill_seconds:.0f}s)")
    preflight: dict[str, Any] = {"case": cases[0]["case"]}
    ref = [engine.prepare(ids_of(cases[0]), cases[0]["media"])]
    engine.run(ref, quiet=True)
    a, b = logits_of(seqs[0]["x"]), logits_of(ref[0]["x"])
    preflight["prefill_last_logits_max_abs_diff_cached_vs_recompute"] = float((a - b).abs().max())
    preflight["prefill_argmax_equal"] = int(a.argmax()) == int(b.argmax())
    gen: list[list[int]] = [[int(logits_of(s["x"]).argmax())] for s in seqs]
    done = [g[0] in stops for g in gen]
    step_seconds = []
    for step in range(1, int(args.max_new_tokens)):
        if all(done):
            break
        t0 = time.time()
        steps = [engine.prepare([gen[i][-1]], None) for i in range(len(cases))]
        caches = engine.run_cached(steps, caches)
        for i, s in enumerate(steps):
            lg = logits_of(s["x"])
            if not bool(torch.isfinite(lg).all()):
                raise Refusal(f"qvl {cases[i]['case']}: non-finite logits at step {step}")
            if step == 1 and i == 0:
                ref = [engine.prepare(ids_of(cases[0]) + gen[0][:1], cases[0]["media"])]
                engine.run(ref, quiet=True)
                r = logits_of(ref[0]["x"])
                preflight["step1_logits_max_abs_diff_cached_vs_recompute"] = float((lg - r).abs().max())
                preflight["step1_argmax_equal"] = int(lg.argmax()) == int(r.argmax())
            if not done[i]:
                nxt = int(lg.argmax())
                gen[i].append(nxt)
                done[i] = nxt in stops
        step_seconds.append(time.time() - t0)
        if step % 8 == 0 or all(done):
            log(f"qvl step {step}: {sum(done)}/{len(cases)} stopped ({step_seconds[-1]:.1f}s/step)")
    answers = []
    for c, g in zip(cases, gen):
        body = [t for t in g if t not in stops]
        answers.append({"case": c["case"], "variant": c["variant"], "image": c.get("image"), "image_sha256": c.get("image_sha256"),
                        "media_rows": c["media_rows"], "grid_thw": c.get("grid_thw"), "subject_prompt": c.get("subject_prompt"),
                        "tokens": g, "answer_verbatim": tok.decode(body, skip_special_tokens=False),
                        "stopped": bool(g and g[-1] in stops), "new_tokens": len(g)})
    img = [x for x in answers if x["variant"] == "image"]
    zer = [x for x in answers if x["variant"] == "zeroed_features"]
    out = Path(args.out_root) / "qvl"
    out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "omni-campaign.s8_media_qvl_answers.v1", "question": args.qvl_question, "side": int(args.qvl_side),
           "tokenizer_dir": args.tokenizer_dir, "chat": "oc_calib.render_prompt thinking=off; image rows (C11 type 6) after the "
           "question inside the user turn; no_image = the same chat without rows; zeroed_features = the rows with zero features",
           "decode": f"greedy, cached (tree _forward_trunk_with_cache semantics; charted writes on the prefill), ledger bands "
                     f"masked (argmax over [0, {engine.band_begin}))", "preflight": preflight,
           "max_new_tokens": int(args.max_new_tokens), "stops": sorted(stops), "answers": answers,
           "answers_differ_between_images": len({x["answer_verbatim"] for x in img}) == len(img),
           "image_answer_equals_zeroed_answer": [a["answer_verbatim"] == z["answer_verbatim"] for a, z in zip(img, zer)],
           "prefill_seconds": prefill_seconds, "seconds": time.time() - started, "step_seconds": step_seconds}
    path = out / "answers.json"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "answers.txt").write_text("\n\n".join(f"[{a['case']}] ({a['new_tokens']} tokens, stopped={a['stopped']})\n"
                                                 f"subject: {a.get('subject_prompt')}\nanswer: {a['answer_verbatim']}"
                                                 for a in answers) + "\n", encoding="utf-8")
    receipt["qvl"] = {"answers": str(path), "answers_sha256": sha256_file(path), "preflight": preflight,
                      "answers_differ_between_images": doc["answers_differ_between_images"],
                      "image_answer_equals_zeroed_answer": doc["image_answer_equals_zeroed_answer"],
                      "cases": [{k: a[k] for k in ("case", "new_tokens", "stopped", "answer_verbatim")} for a in answers]}
    log(f"qvl preflight {json.dumps(preflight)}")
    for a in answers:
        log(f"qvl [{a['case']}] {a['answer_verbatim']!r}")


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omni1", default=str(OMNI1_DIR))
    ap.add_argument("--out-root", default=str(OC / "composite/s8/media_out"))
    ap.add_argument("--modalities", default="ace,h3,qimg,hy3d,qvl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--chimera-src", default=os.environ.get("CHIMERA_OMNI_SRC", ""))
    ap.add_argument("--ace-sigmas", default="0.3,0.5,0.75")
    ap.add_argument("--ace-clip", type=int, default=4)
    ap.add_argument("--ace-seed", type=int, default=0)
    ap.add_argument("--h3-steps", default="0,1,2")
    ap.add_argument("--h3-forward", type=int, default=12, help="forward_in_step of a held H3 clip pair")
    ap.add_argument("--qimg-sample", default="b1_held_glass_pitcher")
    ap.add_argument("--qimg-sigma-files", default="0,1,2,3")
    ap.add_argument("--hy3d-sample", default="sample-0002.safetensors")
    ap.add_argument("--hy3d-sigmas", default="0.25,0.5,0.75")
    ap.add_argument("--qvl-images", default=QVL_DEFAULT_IMAGES)
    ap.add_argument("--qvl-side", type=int, default=448)
    ap.add_argument("--qvl-question", default="Describe this image.")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--tokenizer-dir", default="/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07")
    ap.add_argument("--calibration-dir", default=str(OC / "calibration"))
    ap.add_argument("--out", default=None, help="receipt path (default <out-root>/s8_media_forward.json)")
    ap.add_argument("--build-only", action="store_true", help="build the segments and references, print them, stop (no OMNI1)")
    args = ap.parse_args(argv)
    started = time.time()
    import torch
    import chimera_omni

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    mods = [m for m in args.modalities.split(",") if m]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"schema": SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(Path(__file__).resolve())},
                               "argv": list(sys.argv[1:] if argv is None else argv), "torch": torch.__version__,
                               "device": args.device, "dtype": args.dtype, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "policy": "owner 2026-10-06: every finite number here is a diagnostic; refusals are structural / non-finite only"}
    src = Path(args.chimera_src).resolve() if args.chimera_src else None
    imported = Path(chimera_omni.__file__).resolve()
    receipt["chimera_omni"] = {"imported_from": str(imported), "chimera_src": str(src) if src else None,
                               "bound": bool(src and str(imported).startswith(str(src)))}
    if src and not receipt["chimera_omni"]["bound"]:
        raise Refusal(f"chimera_omni imported from {imported}, not from --chimera-src {src}")
    if args.device.startswith("cuda"):
        receipt["cuda_device_name"] = torch.cuda.get_device_name(0)
        receipt["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    s7c = load_sibling("s7_omni1_check")
    s7p = load_sibling("s7_pairs_parent")
    receipt["siblings"] = {n: sha256_file(HERE / f"{n}.py") for n in ("s7_omni1_check", "s7_pairs_parent")}
    ctx: dict[str, Any] = {"s7p": s7p, "pair_records": {}}
    segs: list[dict[str, Any]] = []
    for mod in mods:
        if mod in BUILDERS:
            segs.extend(BUILDERS[mod](args, ctx))
        elif mod != "qvl":
            raise Refusal(f"unknown modality {mod}")
    receipt["pairs"] = ctx["pair_records"]
    receipt["references"] = {k: v for k, v in ctx.items() if k in ("ace_parent", "qimg_clean_solve")}
    if args.build_only:
        # builder smoke: the references alone, with the parent's own sign check (no OMNI1 load)
        for seg in segs:
            streams = ([("", "x_t", "clean", "parent_raw", SIGN[seg["modality"]], seg["sigma"])] if seg["modality"] != "h3" else
                       [("video_", "video_x_t", "video_clean", None, SIGN["h3_video"], seg["sigma"]),
                        ("audio_", "audio_x_t", "audio_clean", None, SIGN["h3_audio"], seg["sigma_audio"])])
            for prefix, kx, kc, kp, sign, sigma in streams:
                x_t, clean = seg["ref"][kx], seg["ref"][kc]
                parent = seg["ref"].get(kp) if kp else None
                line = {"segment": seg["name"], "stream": prefix or "-", "sigma": round(float(sigma), 4), "rows": int(seg["types"].shape[1]),
                        "x_t_relL2_vs_clean": round(float((x_t - clean).norm() / clean.norm()), 4)}
                if parent is not None:
                    line["parent"] = latent_metrics(x_t, clean, parent, sigma, sign)
                    line["parent_sign_check"] = best_sign(x_t, clean, parent, sigma)
                print("build", json.dumps(line, default=str), flush=True)
        print("build_only checks", json.dumps({s["name"]: s["checks"] for s in segs}, default=str)[:4000])
        return 0
    omni_dir = Path(args.omni1)
    receipt["omni1"] = {"directory": str(omni_dir), "manifest_sha256": sha256_file(omni_dir / "composite_manifest.json"),
                        "index_sha256": sha256_file(omni_dir / "model.safetensors.index.json"),
                        "config_sha256": sha256_file(omni_dir / "config.json")}
    engine = Engine(omni_dir, device=args.device, dtype=getattr(torch, args.dtype), s7c=s7c)
    receipt["omni1"].update({"structural": engine.structural, "strict_load_coverage": engine.coverage,
                             "charted_blocks": {str(j): engine.placements[j]["container"] for j in engine.charted},
                             "band_begin": engine.band_begin, "placeholder_token_id": engine.placeholder})
    out = Path(args.out) if args.out else out_root / "s8_media_forward.json"

    def write_receipt(final: bool) -> None:
        receipt["seconds"] = round(time.time() - started, 1)
        receipt["peak_rss_gib"] = rss_gib()
        receipt["complete"] = final
        out.write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
        Path(str(out) + ".sha256").write_text(sha256_file(out) + "\n", encoding="utf-8")

    with torch.no_grad():
        if segs:
            run_forward(args, engine, segs, receipt)
            write_receipt(False)
        if "qvl" in mods:
            run_qvl(args, engine, receipt, s7p)
    write_receipt(True)
    print("out", out, sha256_file(out))
    for mod, data in (receipt.get("modalities") or {}).items():
        for r in data["segments"]:
            lat = r["latent"]
            parts = []
            for k, v in lat.items():
                if isinstance(v, dict) and "rel_l2_x0_vs_clean" in v:
                    parts.append(f"{k} x0relL2 {v['rel_l2_x0_vs_clean']:.4f} v0 {v['rel_l2_xt_vs_clean_v0_baseline']:.4f} "
                                 f"cosIdeal {v['cos_raw_vs_ideal']:.4f}" + (f" cosParent {v['cos_raw_vs_parent']:.4f}" if "cos_raw_vs_parent" in v else ""))
            print("segment", r["segment"], "finite", r["emission_final"]["finite"], "nonzero", r["emission_final"]["nonzero"], "|", "; ".join(parts))
    if receipt.get("qvl"):
        for c in receipt["qvl"]["cases"]:
            print("qvl", c["case"], json.dumps(c["answer_verbatim"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refusal as error:
        print(f"REFUSED: {error}", flush=True)
        sys.exit(3)
