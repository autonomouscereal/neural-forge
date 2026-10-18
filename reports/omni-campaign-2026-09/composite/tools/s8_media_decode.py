#!/usr/bin/env python3
"""s8_media_decode.py -- S8-MEDIA decode: OMNI1's emitted latents -> media, with each parent's OWN decoder.

S8-MEDIA lane, Claude, 2026-10-13.  Runs INSIDE the ComfyUI runtime image (comfyui:codex-minimax-h3-runtime-env-r5-
20260808: /opt/ComfyUI + /opt/environments/python/comfyui/bin/python).  Reads ``<root>/<modality>/latents.safetensors``
written by ``s8_media_smoke.py`` (sha-checked against latents.json) and decodes every variant of every segment:

    clean            the reference latent (the VAE ceiling: what a perfect prediction decodes to)
    x_t              the noisy input (the v=0 / "do nothing" baseline)
    x0_omni_final    OMNI1's emission at its own output path, turned into x0 with the parent's flow sign
    x0_omni_site     OMNI1's emission read right after the last charted block (diagnostic)
    x0_parent        the parent's own full-model x0 on the same input (where it exists)

Decoders (each the parent's own, through comfy.sd.VAE's auto-detection; latent -> latent_format.process_out first):
    ace   models/vae/ace_1.5_vae.safetensors            ACEAudio15     -> 48 kHz stereo wav
    h3    models/vae/minimax_h3_video_vae_fp16           MiniMaxH3Video -> frames (png strip)
          models/vae/minimax_h3_audio_vae_fp32           (scale 1)      -> 32 kHz stereo wav
    qimg  models/vae/qwen_image_vae.safetensors          Wan21          -> png
    hy3d  checkpoints/hunyuan_3d_v2.1.safetensors vae.*  Hunyuan3Dv2_1  -> occupancy logits on an octree grid
          -> comfy voxel_to_mesh_surfnet (threshold 0.6, octree 256: the shipped 'Image to Model (Hunyuan3d 2.1)'
          blueprint's VoxelToMesh / VAEDecodeHunyuan3D settings) -> glb (comfy save_glb)

Media-space diagnostics (never gates): image / frame PSNR vs the decoded clean (and, for QIMG, vs the calibration
PNG the latent was encoded from); audio SNR and STFT-magnitude cosine vs the decoded clean; occupancy IoU vs the
decoded clean grid; every one also vs the decoded parent where a parent output exists.  A decoded artifact whose
numbers sit at or below the x_t baseline is reported as noise, not as media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

TOOL_NAME = "s8_media_decode"
TOOL_VERSION = "1.0.0"
COMFY_ROOT = os.environ.get("COMFY_ROOT", "/opt/ComfyUI")
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
VARIANTS = ("clean", "x_t", "x0_omni_final", "x0_omni_site", "x0_parent")


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- metrics (pure)

def psnr(a: Any, b: Any, data_range: float = 1.0) -> float:
    mse = float((a.double() - b.double()).pow(2).mean())
    return float("inf") if mse <= 0 else 10.0 * math.log10(data_range * data_range / mse)


def snr_db(x: Any, ref: Any) -> float:
    num = float(ref.double().pow(2).sum())
    den = float((ref.double() - x.double()).pow(2).sum())
    return float("inf") if den <= 0 else 10.0 * math.log10(max(num, 1e-30) / den)


def stft_cos(x: Any, ref: Any, n_fft: int = 2048) -> float:
    import torch
    win = torch.hann_window(n_fft, dtype=torch.float32)
    def mag(s: Any) -> Any:
        return torch.stft(s.float().reshape(-1, s.shape[-1]), n_fft, hop_length=n_fft // 4, window=win, return_complex=True).abs()
    a, b = mag(x), mag(ref)
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


def iou(a: Any, b: Any) -> float | None:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return None if union == 0 else inter / union   # both empty: undefined, not a match


def write_wav(path: Path, audio: Any, rate: int) -> None:
    """[C, S] float in [-1, 1] -> 16-bit PCM wav (clipped)."""
    import numpy as np
    data = audio.detach().float().clamp(-1, 1).cpu().numpy()
    pcm = (data.T * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(data.shape[0]))
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


def write_png(path: Path, image: Any) -> None:
    """[H, W, 3] float in [0, 1] -> png."""
    import numpy as np
    from PIL import Image
    arr = (image.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def frames_strip(frames: Any) -> Any:
    """[T, C, H, W] -> [H, T*W, C] side by side."""
    import torch
    return torch.cat([f.permute(1, 2, 0) for f in frames], dim=1)


# ----------------------------------------------------------------------------- comfy plumbing

def bind_comfy(device: str = "cuda") -> dict[str, Any]:
    if COMFY_ROOT not in sys.path:
        sys.path.insert(0, COMFY_ROOT)
    media = OC / "media"
    if str(media) not in sys.path:
        sys.path.insert(0, str(media))
    import torch
    if device == "cpu" or not torch.cuda.is_available():
        # comfy picks its device once, at model_management import (see mediafold.vae_adapters): flip args.cpu first
        import comfy.options  # noqa: F401
        from comfy.cli_args import args as comfy_args
        comfy_args.cpu = True
    from mediafold.vae_adapters import ComfyVaeAdapter  # noqa: F401  (also installs the comfy_kitchen shim when needed)
    try:
        import comfy.sd  # noqa: F401
    except Exception:
        from mediafold.vae_adapters import _install_comfy_kitchen_stub
        _install_comfy_kitchen_stub()
        import comfy.sd  # noqa: F401
    import comfy.latent_formats  # noqa: F401
    info = {"comfy_root": COMFY_ROOT, "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({"device_name": props.name, "device_uuid": str(getattr(props, "uuid", "")), "device_count": torch.cuda.device_count()})
    for rel in ("comfy/sd.py", "comfy/latent_formats.py", "comfy/ldm/minimax/model.py", "comfy_extras/nodes_hunyuan3d.py"):
        p = Path(COMFY_ROOT) / rel
        if p.is_file():
            info.setdefault("sha256", {})[rel] = sha256_file(p)
    return info


def adapter(path: Path, device: str) -> Any:
    from mediafold.vae_adapters import ComfyVaeAdapter
    return ComfyVaeAdapter(str(path), device=device, path_mode="auto")


def load_latents(root: Path, modality: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file
    plan = json.loads((root / modality / "latents.json").read_text(encoding="utf-8"))
    path = Path(plan["file"])
    if sha256_file(path) != plan["sha256"]:
        raise SystemExit(f"{path} does not hash like latents.json")
    return plan, load_file(str(path))


def segments_of(tensors: dict[str, Any]) -> list[str]:
    return sorted({k.split(".", 1)[0] for k in tensors})


# ----------------------------------------------------------------------------- per modality

def audio_metrics(decoded: dict[str, Any], *, base: str = "clean") -> dict[str, Any]:
    out = {}
    ref = decoded.get(base)
    for name, a in decoded.items():
        rec = {"finite": bool(a.isfinite().all()), "rms": float(a.double().pow(2).mean().sqrt()), "samples": int(a.shape[-1])}
        if ref is not None and name != base:
            rec.update({"snr_db_vs_clean": snr_db(a, ref), "stft_cos_vs_clean": stft_cos(a, ref)})
        if "x0_parent" in decoded and name not in ("x0_parent",):
            rec["snr_db_vs_parent"] = snr_db(a, decoded["x0_parent"])
            rec["stft_cos_vs_parent"] = stft_cos(a, decoded["x0_parent"])
        out[name] = rec
    return out


def verdict_vs_baseline(metrics: dict[str, Any], key: str, higher_is_better: bool = True) -> dict[str, Any]:
    """Is OMNI1's decoded x0 any closer to the clean decode than the noisy input is?"""
    out = {}
    base = metrics.get("x_t", {}).get(key)
    for name in ("x0_omni_final", "x0_omni_site", "x0_parent"):
        v = metrics.get(name, {}).get(key)
        if v is None or base is None:
            continue
        better = (v > base) if higher_is_better else (v < base)
        out[name] = {key: v, "x_t_baseline": base, "beats_noisy_input": bool(better)}
    return out


def decode_ace(root: Path, models: Path, device: str, mirror: Path | None) -> dict[str, Any]:
    import comfy.latent_formats as lf
    plan, tensors = load_latents(root, "ace")
    vae = adapter(models / "vae/ace_1.5_vae.safetensors", device)
    fmt = lf.ACEAudio15()
    rate = int(vae.geometry.sample_rate_hz or 48000)
    out = {"vae": vae.geometry.notes, "sample_rate": rate, "segments": {}}
    for seg in segments_of(tensors):
        decoded, files = {}, {}
        for v in VARIANTS:
            key = f"{seg}.{v}"
            if key not in tensors:
                continue
            audio = vae.decode(fmt.process_out(tensors[key].float()))[0].float().cpu()   # [C, S]
            decoded[v] = audio
            path = root / "ace" / f"{seg}.{v}.wav"
            write_wav(path, audio, rate)
            files[v] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        m = audio_metrics(decoded)
        out["segments"][seg] = {"files": files, "metrics": m, "vs_noisy_input": verdict_vs_baseline(m, "snr_db_vs_clean")}
        log(f"ace {seg}: " + ", ".join(f"{k} snr {r.get('snr_db_vs_clean', float('nan')):.2f} dB" for k, r in m.items()))
    return out


def decode_h3(root: Path, models: Path, device: str, mirror: Path | None) -> dict[str, Any]:
    import torch
    import comfy.latent_formats as lf
    plan, tensors = load_latents(root, "h3")
    vvae = adapter(models / "vae/minimax_h3_video_vae_fp16.safetensors", device)
    avae = adapter(models / "vae/minimax_h3_audio_vae_fp32.safetensors", device)
    fmt = lf.MiniMaxH3Video()
    rate = int(avae.geometry.sample_rate_hz or 32000)
    out = {"video_vae": vvae.geometry.notes, "audio_vae": avae.geometry.notes, "audio_sample_rate": rate,
           "latent_format": {"class": "MiniMaxH3Video", "scale_factor": float(getattr(fmt, "scale_factor", 1.0)),
                             "shift_factor": float(getattr(fmt, "shift_factor", 0.0) or 0.0)}, "segments": {}}
    for seg in segments_of(tensors):
        vid, aud, files = {}, {}, {}
        for v in VARIANTS:
            kv, ka = f"{seg}.video_{v}", f"{seg}.audio_{v}"
            if kv in tensors:
                frames = vvae.decode(fmt.process_out(tensors[kv].float()))[0].float().cpu()   # [T, C, H, W]
                frames01 = frames.clamp(0, 1) if float(frames.min()) >= -0.05 else ((frames + 1) / 2).clamp(0, 1)
                vid[v] = frames01
                p = root / "h3" / f"{seg}.video_{v}.png"
                write_png(p, frames_strip(frames01))
                files[f"video_{v}"] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p), "frames": int(frames.shape[0]),
                                       "hw": [int(frames.shape[-2]), int(frames.shape[-1])], "raw_range": [float(frames.min()), float(frames.max())]}
            if ka in tensors:
                a = avae.decode(tensors[ka].float())[0].float().cpu()   # [C, S]
                aud[v] = a
                p = root / "h3" / f"{seg}.audio_{v}.wav"
                write_wav(p, a, rate)
                files[f"audio_{v}"] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        vm = {}
        for name, f in vid.items():
            rec = {"finite": bool(f.isfinite().all())}
            if "clean" in vid and name != "clean":
                rec["psnr_vs_clean"] = psnr(f, vid["clean"])
            vm[name] = rec
        am = audio_metrics(aud)
        out["segments"][seg] = {"files": files, "video_metrics": vm, "audio_metrics": am,
                                "video_vs_noisy_input": verdict_vs_baseline(vm, "psnr_vs_clean"),
                                "audio_vs_noisy_input": verdict_vs_baseline(am, "snr_db_vs_clean")}
        log(f"h3 {seg}: video " + ", ".join(f"{k} {r.get('psnr_vs_clean', float('nan')):.2f} dB" for k, r in vm.items())
            + " | audio " + ", ".join(f"{k} {r.get('snr_db_vs_clean', float('nan')):.2f} dB" for k, r in am.items()))
    return out


def decode_qimg(root: Path, models: Path, device: str, mirror: Path | None) -> dict[str, Any]:
    import torch
    import comfy.latent_formats as lf
    from PIL import Image
    plan, tensors = load_latents(root, "qimg")
    vae = adapter(models / "vae/qwen_image_vae.safetensors", device)
    fmt = lf.Wan21()
    source = None
    for s in plan.get("segments", []):
        img = (s.get("layout") or {}).get("image")
        if img:
            source = Path(img)
    original = None
    if source is not None and source.is_file():
        original = torch.tensor(list(Image.open(source).convert("RGB").getdata()), dtype=torch.float32).view(
            Image.open(source).size[1], Image.open(source).size[0], 3) / 255.0
    out = {"vae": vae.geometry.notes, "latent_format": "Wan21 (latents_mean / latents_std)", "source_image": str(source) if source else None,
           "source_image_sha256": sha256_file(source) if source and source.is_file() else None, "segments": {}}
    for seg in segments_of(tensors):
        imgs, files = {}, {}
        for v in VARIANTS:
            key = f"{seg}.{v}"
            if key not in tensors:
                continue
            dec = vae.decode(fmt.process_out(tensors[key].float()))[0].float().cpu()   # [T, C, H, W]
            img = dec[0].permute(1, 2, 0)
            img01 = img.clamp(0, 1) if float(img.min()) >= -0.05 else ((img + 1) / 2).clamp(0, 1)
            imgs[v] = img01
            p = root / "qimg" / f"{seg}.{v}.png"
            write_png(p, img01)
            files[v] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p), "hw": list(img.shape[:2]),
                        "raw_range": [float(img.min()), float(img.max())]}
        m = {}
        for name, im in imgs.items():
            rec = {"finite": bool(im.isfinite().all())}
            if "clean" in imgs and name != "clean":
                rec["psnr_vs_clean"] = psnr(im, imgs["clean"])
            if original is not None and tuple(original.shape) == tuple(im.shape):
                rec["psnr_vs_source_png"] = psnr(im, original)
            if "x0_parent" in imgs and name != "x0_parent":
                rec["psnr_vs_parent"] = psnr(im, imgs["x0_parent"])
            m[name] = rec
        out["segments"][seg] = {"files": files, "metrics": m, "vs_noisy_input": verdict_vs_baseline(m, "psnr_vs_clean")}
        log(f"qimg {seg}: " + ", ".join(f"{k} {r.get('psnr_vs_clean', float('nan')):.2f} dB" for k, r in m.items()))
    return out


def decode_hy3d(root: Path, models: Path, device: str, mirror: Path | None, *, octree: int, chunks: int, threshold: float) -> dict[str, Any]:
    import torch
    import comfy.latent_formats as lf
    import comfy.sd
    from safetensors import safe_open
    from comfy_extras.nodes_hunyuan3d import voxel_to_mesh_surfnet
    from comfy_extras.nodes_save_3d import save_glb
    plan, tensors = load_latents(root, "hy3d")
    ckpt = models / "checkpoints/hunyuan_3d_v2.1.safetensors"
    state = {}
    with safe_open(str(ckpt), framework="pt") as h:
        for key in h.keys():
            if key.startswith("vae."):
                state[key[4:]] = h.get_tensor(key)
    vae = comfy.sd.VAE(sd=state)
    fmt = lf.Hunyuan3Dv2_1()
    out = {"vae": {"checkpoint": str(ckpt), "vae_tensors": len(state), "class": type(vae.first_stage_model).__name__},
           "octree_resolution": octree, "num_chunks": chunks, "mesh_threshold": threshold,
           "latent_format": {"class": "Hunyuan3Dv2_1", "scale_factor": float(fmt.scale_factor)}, "segments": {}}
    for seg in segments_of(tensors):
        grids, files = {}, {}
        for v in VARIANTS:
            key = f"{seg}.{v}"
            if key not in tensors:
                continue
            logits = vae.decode(fmt.process_out(tensors[key].float()), vae_options={"num_chunks": chunks, "octree_resolution": octree})
            grid = logits[0].float().cpu()
            grids[v] = grid
            rec = {"grid": list(grid.shape), "finite": bool(grid.isfinite().all()), "logit_range": [float(grid.min()), float(grid.max())],
                   "occupied_fraction": float((grid > threshold).float().mean())}
            try:
                verts, faces = voxel_to_mesh_surfnet(grid, threshold=threshold, device=None)
                rec.update({"vertices": int(verts.shape[0]), "faces": int(faces.shape[0])})
                if int(verts.shape[0]) and int(faces.shape[0]):
                    p = root / "hy3d" / f"{seg}.{v}.glb"
                    save_glb(verts, faces, str(p))
                    rec.update({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
                else:
                    rec["mesh"] = "empty at this threshold"
            except Exception as error:  # a mesh failure is recorded, not hidden
                rec["mesh_error"] = f"{type(error).__name__}: {error}"
            files[v] = rec
        m = {}
        for name, g in grids.items():
            r = {}
            if "clean" in grids and name != "clean":
                r["iou_vs_clean"] = iou(g > threshold, grids["clean"] > threshold)
                r["logit_cos_vs_clean"] = float(torch.nn.functional.cosine_similarity(g.flatten().double(), grids["clean"].flatten().double(), dim=0))
            if "x0_parent" in grids and name != "x0_parent":
                r["iou_vs_parent"] = iou(g > threshold, grids["x0_parent"] > threshold)
            m[name] = r
        out["segments"][seg] = {"files": files, "metrics": m, "vs_noisy_input": verdict_vs_baseline(m, "iou_vs_clean")}
        log(f"hy3d {seg}: " + ", ".join(f"{k} iou {r.get('iou_vs_clean')} occ {files[k]['occupied_fraction']:.3f}"
                                         for k, r in m.items()))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(OC / "composite/s8/media_out"))
    ap.add_argument("--models-root", default="/home/cereal/comfyui/models")
    ap.add_argument("--modalities", default="ace,h3,qimg,hy3d")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hy3d-octree", type=int, default=256)
    ap.add_argument("--hy3d-chunks", type=int, default=20000)
    ap.add_argument("--hy3d-threshold", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    started = time.time()
    info = bind_comfy(args.device)
    import torch
    root, models = Path(args.root), Path(args.models_root)
    receipt: dict[str, Any] = {"schema": "omni-campaign.s8_media_decode.v1", "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
                               "sha256": sha256_file(Path(__file__).resolve())}, "argv": list(sys.argv[1:] if argv is None else argv),
                               "runtime": info, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "policy": "every number is a diagnostic (owner policy 2026-10-06)", "modalities": {}}
    out = Path(args.out) if args.out else root / f"s8_media_decode.{'-'.join(args.modalities.split(','))}.json"
    for mod in [m for m in args.modalities.split(",") if m]:
        t0 = time.time()
        with torch.no_grad():
            if mod == "ace":
                rec = decode_ace(root, models, args.device, None)
            elif mod == "h3":
                rec = decode_h3(root, models, args.device, None)
            elif mod == "qimg":
                rec = decode_qimg(root, models, args.device, None)
            elif mod == "hy3d":
                rec = decode_hy3d(root, models, args.device, None, octree=args.hy3d_octree, chunks=args.hy3d_chunks,
                                  threshold=args.hy3d_threshold)
            else:
                raise SystemExit(f"unknown modality {mod}")
        rec["seconds"] = round(time.time() - t0, 1)
        receipt["modalities"][mod] = rec
        out.write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    receipt["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
    Path(str(out) + ".sha256").write_text(sha256_file(out) + "\n", encoding="utf-8")
    print("out", out, sha256_file(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
