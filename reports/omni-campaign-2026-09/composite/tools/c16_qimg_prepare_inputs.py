#!/usr/bin/env python3
"""Encode real image files and prompt text with the pinned QIMG teachers.

This is frozen inference plus a declared rectified-flow calibration noise grid,
not training, a sampler trajectory, or final folded-child evidence. An existing
image may itself be teacher-generated; its provenance must say so explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c16_qimg_capture import (INPUT_SCHEMA, file_sha, read_header,
                              tensor_sha, validate_inputs, verified_file, write_tensors)

PLAN_SCHEMA = "neural_forge.qimg_encoding_plan.v1"
SOURCE_FILES = {
    "comfy/sd.py": "7efb10f22252446069a0b88f0a8bc2d94713696d9f946d7757a0981ae8101b58",
    "comfy/sd1_clip.py": "cada45bf2cd1e2e20d24a46b959d86fc9f5529953456bb21471735f036e2dc70",
    "comfy/model_management.py": "b7741d049c2c749c316e5769ef520846bc7dd444e67de08403a4ee13a9c034c2",
    "comfy/latent_formats.py": "f13259cbd815a339a474976522eb6aae10fb16c8397f13b41e50ddd107f8caf8",
    "comfy/model_sampling.py": "8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717",
    "comfy/text_encoders/qwen_image.py": "9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9",
    "comfy/text_encoders/llama.py": "ad7d59be8f4732eee5b898591330adc875d69fcbd9987b9aa6c8473aeb423a11",
    "comfy/ldm/wan/vae.py": "b18fdd6d95d9c7093fbc8f2c831dfb36322aa1951c2dd17d32733dac53d689e4",
}


def read_plan(path):
    if path.stat().st_size > 1 << 20:
        raise ValueError("encoding plan exceeds 1 MiB")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected encoding plan schema")
    sigmas = plan.get("sigmas", [])
    if not sigmas or len(sigmas) > 8 or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 < s < 1 for s in sigmas):
        raise ValueError("explicit calibration sigmas must be in (0,1), at most eight")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("duplicate calibration sigma")
    entries = plan.get("entries", [])
    if not entries or len(entries) * len(sigmas) > 32:
        raise ValueError("one to 32 total source forwards required")
    ids, splits, verified = set(), {}, []
    for entry in entries:
        name, split = entry.get("id", ""), entry.get("split")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) or name in ids or split not in {"fit", "holdout"}:
            raise ValueError("unique safe sample id and explicit fit/holdout split required")
        ids.add(name)
        seed = entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("each image requires an explicit nonnegative int63 seed")
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 8192 or "<|" in prompt:
            raise ValueError("bounded plain text prompt required; no template overrides")
        provenance = entry.get("image_provenance")
        if not isinstance(provenance, dict) or not provenance.get("kind") or not provenance.get("description"):
            raise ValueError("image origin must be explicit; generated images cannot be passed off as photographs")
        image = verified_file(path.parent, entry["image"], max_bytes=128 << 20)
        digest = entry["image"]["sha256"]
        if digest in splits and splits[digest] != split:
            raise ValueError("same original image appears in fit and holdout")
        splits[digest] = split
        # Decode and geometry-check before any teacher allocation.
        pixels, meta = image_pixels(image)
        verified.append({**entry, "image": {"path": str(image), "sha256": digest}, "pixel_preflight": meta})
        del pixels
    return {**plan, "entries": verified}


def image_pixels(path):
    with Image.open(path) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("image input must have exactly one frame")
        image = ImageOps.exif_transpose(original)
        width, height = image.size
        if min(width, height) < 16 or max(width, height) > 512 or width % 16 or height % 16:
            raise ValueError("use an explicit 16-aligned image at most 512x512; no implicit crop/resize")
        alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
        if alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).float().unsqueeze(0) / 255.0
        return pixels, {"width": width, "height": height, "exif_transpose": True,
            "alpha_composite": "white" if alpha else "none", "resize": "none", "crop": "none",
            "pixel_range": "RGB float32 [0,1], BHWC", "pixels_sha256": tensor_sha(pixels)}


def aligned_context_ids(tokens, context):
    """Mirror the pinned teacher's crop and prove alignment to its returned rows."""
    if set(tokens) != {"qwen25_7b"} or len(tokens["qwen25_7b"]) != 1:
        raise ValueError("plain single-sequence Qwen2.5 tokens required")
    pairs = tokens["qwen25_7b"][0]
    if any(not isinstance(p[0], numbers.Integral) or p[1] != 1.0 for p in pairs):
        raise ValueError("embedded/weighted/image tokens are outside plain capture")
    ids = [int(p[0]) for p in pairs]
    start, count = -1, 0
    for i, value in enumerate(ids):
        if value == 151644 and count < 2:
            start, count = i, count + 1
    if count != 2:
        raise ValueError("normal Qwen-Image system/user template markers missing")
    # Native code checks the pre-crop output length, which equals this single
    # token sequence length. Returned length alone is not used to infer a crop.
    if len(ids) > start + 3 and ids[start + 1:start + 3] == [872, 198]:
        start += 3
    cropped = torch.tensor([ids[start:]], dtype=torch.int64)
    if cropped.shape != context.shape[:2]:
        raise ValueError("teacher context rows do not match the exact source token crop")
    return cropped, {"template_crop_start": start, "original_token_count": len(ids),
                    "context_token_count": cropped.shape[1]}


def calibration_inputs(clean, sigmas, seed, noise_scaling):
    if clean.ndim != 5 or clean.shape[0:3] != (1, 16, 1) or not torch.isfinite(clean).all():
        raise ValueError("expected one finite native Wan21 image encoding")
    clean = clean.detach().float().cpu()
    sigma = torch.tensor(sigmas, dtype=torch.float32)
    # A shared per-image Gaussian draw makes the explicitly chosen noise levels
    # comparable. This is a calibration grid, never described as sampler states.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    values = torch.cat([noise_scaling(s[None], noise, clean) for s in sigma], 0)
    expected = sigma[:, None, None, None, None] * noise + (1 - sigma[:, None, None, None, None]) * clean
    if not torch.equal(values, expected):
        raise ValueError("source noise scaling differs from the audited unit-scale CONST recipe")
    return values, sigma, {"noise_sha256": tensor_sha(noise), "clean_model_latent_sha256": tensor_sha(clean),
        "seed": seed, "generator": "torch CPU", "noise_kind": "one shared standard-normal draw per image",
        "recipe": "sigma * noise + (1 - sigma) * Wan21.process_in(VAE.encode(pixels))",
        "sigma_semantics": "explicit calibration noise fractions, not captured sampler trajectory",
        "sigma_shift_applied": False, "noise_scale": 1.0}


def init_comfy(root, device):
    actual = {name: file_sha(root / name) for name in SOURCE_FILES}
    if actual != SOURCE_FILES:
        raise ValueError("encoding source differs from the reviewed r5 runtime")
    sys.path.insert(0, str(root))
    import comfy.options
    from comfy.cli_args import args
    args.cpu = device == "cpu"
    import comfy.model_management as management
    import comfy.sd as sd
    import comfy.utils as utils
    from comfy.latent_formats import Wan21
    from comfy.model_sampling import CONST
    if management.in_training or str(management.get_torch_device()) != device:
        raise ValueError("wrong teacher device or training path")
    return sd, utils, Wan21(), CONST(), actual


def encode_plan(plan_path, vae_path, te_path, comfy_root, device, out):
    if out.exists():
        raise ValueError("output already exists; no overwrite/cleanup option")
    plan = read_plan(plan_path)
    identities = {}
    for name, path in (("vae", vae_path), ("text_encoder", te_path)):
        _, identity = read_header(path)
        identities[name] = {"path": str(path.resolve()), **identity}
    probe = out.resolve().parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < 6 << 30:
        raise ValueError("encoding must leave bounded output space and 4 GiB disk reserve")
    argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        sd, utils, latent_format, flow, source = init_comfy(comfy_root, device)
    finally:
        sys.argv = argv
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "neural_forge.qimg_encoding_receipt.v1", "status": "incomplete",
        "kind": "real_image_vae_qwen25_context", "latent_space": "wan21_model_input",
        **identities, "plan_sha256": file_sha(plan_path), "source_code_sha256": source,
        "entries": {}, "details": {}, "runtime": {"torch": torch.__version__, "python": sys.version,
        "pillow": pillow_version, "device": device, "visible_gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES")},
        "scope": "frozen native teacher encodings; no gradient updates, sampling, or folded-child proof"}
    manifest = {"schema": INPUT_SCHEMA, "kind": receipt["kind"], "latent_space": receipt["latent_space"],
        "context_kind": "qwen25_vl_7b_hidden_states", "entries": []}
    started = time.time()
    try:
        clip = sd.load_clip([str(te_path)], clip_type=sd.CLIPType.QWEN_IMAGE, disable_dynamic=True)
        if type(clip.cond_stage_model).__module__ != "comfy.text_encoders.qwen_image":
            raise ValueError("actual Qwen-Image text teacher required")
        clip.cond_stage_model.eval()
        contexts = {}
        with torch.inference_mode():
            for entry in plan["entries"]:
                tokens = clip.tokenize(entry["prompt"])
                if set(tokens) != {"qwen25_7b"} or len(tokens["qwen25_7b"]) != 1 or len(tokens["qwen25_7b"][0]) > 1024:
                    raise ValueError("plain text encoding is bounded to one 1024-token input")
                encoded = clip.encode_from_tokens(tokens, return_dict=True)
                context = encoded["cond"].detach().cpu()
                # FP16 -> FP32 is lossless; it makes the capture input storage
                # contract explicit without changing the teacher computation.
                original_dtype = str(context.dtype)
                if context.dtype == torch.float16:
                    context = context.float()
                ids, align = aligned_context_ids(tokens, context)
                align["teacher_output_dtype"] = original_dtype
                align["stored_context_dtype"] = str(context.dtype)
                mask = encoded.get("attention_mask", torch.ones(context.shape[:2], dtype=torch.bool))
                mask = mask.detach().cpu().bool()
                contexts[entry["id"]] = (context, ids, mask, align)
        # Only this standalone process's teachers are managed here. No service
        # endpoint, shared runtime queue, or other process is unloaded.
        import comfy.model_management as management
        management.unload_all_models()
        del clip
        state, metadata = utils.load_torch_file(str(vae_path), safe_load=True, return_metadata=True)
        vae = sd.VAE(sd=state, device=torch.device(device), dtype=torch.bfloat16, metadata=metadata)
        if type(vae.first_stage_model).__module__ != "comfy.ldm.wan.vae" or vae.latent_channels != 16:
            raise ValueError("actual Wan21 VAE required")
        # For this native checkpoint there is no key remapping. Refuse silently
        # missing weights and record the actual precision conversion in use.
        actual_state = vae.first_stage_model.state_dict()
        if set(actual_state) != set(state) or any(actual_state[k].shape != state[k].shape for k in state):
            raise ValueError("VAE loaded state does not completely cover the native safe checkpoint")
        vae.first_stage_model.eval()
        receipt["vae_loaded_complete_key_shape_coverage"] = True
        receipt["vae_storage_dtypes"] = sorted({str(t.dtype) for t in actual_state.values()})
        del state, actual_state
        with torch.inference_mode():
            for entry in plan["entries"]:
                if file_sha(Path(entry["image"]["path"])) != entry["image"]["sha256"]:
                    raise ValueError("image changed after preflight")
                pixels, pixel_meta = image_pixels(Path(entry["image"]["path"]))
                if pixel_meta != entry["pixel_preflight"]:
                    raise ValueError("decoded pixels changed after preflight")
                raw = vae.encode(pixels).float().cpu()
                clean = latent_format.process_in(raw)
                values, sigma, noise = calibration_inputs(clean, plan["sigmas"], entry["seed"], flow.noise_scaling)
                context, ids, mask, align = contexts.pop(entry["id"])
                tensors = {"model_inputs": values, "sigma": sigma, "context": context,
                           "context_mask": mask, "context_token_ids": ids}
                validate_inputs(tensors, max_tokens=2048)
                target = out / (entry["id"] + ".safetensors")
                stored = write_tensors(target, tensors)
                prompt_sha = hashlib.sha256(entry["prompt"].encode()).hexdigest()
                receipt["entries"][entry["id"]] = {"image_sha256": entry["image"]["sha256"],
                    "prompt_sha256": prompt_sha, "tensor_file_sha256": stored["file_sha256"]}
                receipt["details"][entry["id"]] = {"pixels": pixel_meta, "alignment": align, "noise": noise,
                    "image_provenance": entry["image_provenance"], "tensors": stored,
                    "raw_vae_latent_sha256": tensor_sha(raw), "vae_posterior": "native encoder mu; no posterior sample"}
                manifest["entries"].append({"id": entry["id"], "split": entry["split"], "image": entry["image"],
                    "prompt": entry["prompt"], "prompt_sha256": prompt_sha,
                    "tensors": {"path": target.name, "sha256": stored["file_sha256"]}})
                print(json.dumps({"encoded": entry["id"], "sigmas": plan["sigmas"], "context_tokens": ids.shape[1]}), flush=True)
        for name, path in (("vae", vae_path), ("text_encoder", te_path)):
            _, identity = read_header(path)
            if identity != identities[name]:
                raise ValueError("teacher header identity changed during encoding")
        receipt["status"] = "complete"
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["seconds"] = time.time() - started
        if device.startswith("cuda"):
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        receipt_path = out / "encoding_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["encoding_receipt"] = {"path": receipt_path.name, "sha256": file_sha(receipt_path)}
    (out / "capture_inputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=Path("/opt/ComfyUI"))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-plan-only", action="store_true")
    args = parser.parse_args()
    if args.validate_plan_only:
        print(json.dumps(read_plan(args.plan), indent=2, sort_keys=True))
        return
    encode_plan(args.plan, args.vae, args.text_encoder, args.comfy_root, args.device, args.out)


if __name__ == "__main__":
    main()
