#!/usr/bin/env python3
"""Generate a bounded, preregistered QIMG parent calibration set, not child proof.

Uses native frozen Comfy text encoding, Euler/normal sampling and Wan VAE decode.
No service endpoint, gradient update, custom attention, selection or retry search.
Outputs are teacher-generated images, with a complete plan/seed/source receipt.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c16_qimg_capture import (PARENT_HEADER_SHA256, file_sha, load_parent,
                              read_header, tensor_sha, verify_source_code)
from c16_qimg_prepare_inputs import init_comfy

SCHEMA = "neural_forge.qimg_parent_image_plan.v1"
SAMPLING_SOURCE = {
    "comfy/sample.py": "ee3485747727f9a182b9e7dd5a32a81e9429c329f2dcd248996ea0eba191faea",
    "comfy/samplers.py": "b9f5e22d0d65b5e78cd4dcc763e702369869dc92cf4371dc52bbecaf0733fe49",
    "comfy/k_diffusion/sampling.py": "cc5f944efd85c566484c3999beb74e8c19c894f2a50ca574d090c3a46ac6bd06",
}


def read_plan(path):
    if path.stat().st_size > 1 << 20:
        raise ValueError("plan exceeds 1 MiB")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA:
        raise ValueError("unexpected parent image plan")
    settings = {"width": 512, "height": 512, "steps": 20, "cfg": 4.0,
                "sampler": "euler", "scheduler": "normal", "denoise": 1.0}
    if plan.get("sampling") != settings:
        raise ValueError("this reviewed runner requires the explicit 512px Euler20/normal/CFG4 recipe")
    entries = plan.get("entries", [])
    if not 1 <= len(entries) <= 8:
        raise ValueError("one to eight preregistered images required")
    ids, prompts, seeds = set(), set(), set()
    for entry in entries:
        name, prompt, seed = entry.get("id", ""), entry.get("prompt"), entry.get("seed")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) or name in ids:
            raise ValueError("safe unique image ID required")
        if not isinstance(prompt, str) or not prompt.strip() or "<|" in prompt or len(prompt.encode()) > 8192 or prompt in prompts:
            raise ValueError("bounded unique plain prompt required")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 or seed in seeds:
            raise ValueError("unique nonnegative int63 seed required")
        if entry.get("split") not in {"fit", "holdout"}:
            raise ValueError("predeclared fit/holdout split required")
        ids.add(name)
        prompts.add(prompt)
        seeds.add(seed)
    return plan


def png_pixels(pixels, size=(512, 512)):
    # Native Wan decode returns BTHWC; one image is exactly one temporal frame.
    if pixels.shape == (1, 1, size[1], size[0], 3):
        pixels = pixels[:, 0]
    if pixels.shape != (1, size[1], size[0], 3) or not torch.isfinite(pixels).all():
        raise ValueError("native decoder did not return one finite full-resolution RGB image")
    if bool(((pixels < 0) | (pixels > 1)).any()):
        raise ValueError("native decoded RGB must already be in [0,1]")
    # Matches native SaveImage's multiply/clip/uint8, with no resize or editing.
    return np.clip(255.0 * pixels[0].detach().float().cpu().numpy(), 0, 255).astype(np.uint8)


def generate(plan_path, weights, vae_path, te_path, comfy_root, out, selected_ids=()):
    if out.exists():
        raise ValueError("output exists; overwrite and automatic cleanup are forbidden")
    plan = read_plan(plan_path)
    entries = plan["entries"]
    if selected_ids:
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= {e["id"] for e in entries}:
            raise ValueError("selection must be unique IDs from the preregistered plan")
        entries = [e for e in entries if e["id"] in selected_ids]
    probe = out.resolve().parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < 6 << 30:
        raise ValueError("insufficient bounded output reserve")
    source = verify_source_code(comfy_root)
    sampling_source = {name: file_sha(comfy_root / name) for name in SAMPLING_SOURCE}
    if sampling_source != SAMPLING_SOURCE:
        raise ValueError("sampling code differs from reviewed native r5")
    identities = {name: read_header(path)[1] for name, path in
                  (("diffusion", weights), ("vae", vae_path), ("text_encoder", te_path))}
    if identities["diffusion"]["header_sha256"] != PARENT_HEADER_SHA256:
        raise ValueError("source header differs from the inspected Qwen-Image FP8 parent")
    argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        sd, utils, _, _, encoding_source = init_comfy(comfy_root, "cuda:0")
    finally:
        sys.argv = argv
    import comfy.model_management as management
    import comfy.sample as sample
    import comfy.samplers as samplers
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "neural_forge.qimg_parent_images.v1", "status": "incomplete",
               "scope": "frozen parent-generated calibration images; not folded-child output or acceptance",
               "plan_sha256": file_sha(plan_path), "sampling": plan["sampling"], "sources": identities,
               "source_code_sha256": {**source, **encoding_source, **sampling_source}, "images": []}
    started = time.time()
    try:
        with torch.inference_mode():
            clip = sd.load_clip([str(te_path)], clip_type=sd.CLIPType.QWEN_IMAGE, disable_dynamic=True)
            if type(clip.cond_stage_model).__module__ != "comfy.text_encoders.qwen_image":
                raise ValueError("native Qwen image text encoder required")
            clip.cond_stage_model.eval()
            conditions = {}
            for entry in entries:
                tokens = clip.tokenize(entry["prompt"])
                if len(tokens.get("qwen25_7b", [])) != 1 or len(tokens["qwen25_7b"][0]) > 1024:
                    raise ValueError("bounded plain native conditioning required")
                conditions[entry["id"]] = clip.encode_from_tokens_scheduled(tokens)
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
            management.unload_all_models()
            del clip
            gc.collect()
            patcher, dit, _, runtime = load_parent(weights, comfy_root, "cuda:0")
            if (runtime["parent_class"] != "comfy.model_base.QwenImage"
                    or len(dit.transformer_blocks) != 60):
                raise ValueError("the actual complete native Qwen-Image parent is required")
            receipt["runtime"] = runtime
            sampler = samplers.KSampler(patcher, steps=20, device=patcher.load_device,
                                        sampler="euler", scheduler="normal", denoise=1.0,
                                        model_options=patcher.model_options)
            receipt["native_sampling_sigmas"] = sampler.sigmas.detach().cpu().tolist()
            latents = {}
            for entry in entries:
                latent = torch.zeros(1, 16, 1, 64, 64)
                noise = sample.prepare_noise(latent, entry["seed"])
                ticks = []
                def callback(step, x0, x, total):
                    if not torch.isfinite(x0).all() or not torch.isfinite(x).all():
                        raise ValueError("non-finite native sampler state")
                    ticks.append(int(step))
                    print(json.dumps({"id": entry["id"], "step": int(step), "total": int(total)}), flush=True)
                value = sampler.sample(noise, conditions[entry["id"]], negative, cfg=4.0,
                                       latent_image=latent, callback=callback, disable_pbar=True, seed=entry["seed"])
                value = value.to(device=management.intermediate_device(), dtype=management.intermediate_dtype())
                if ticks != list(range(20)) or value.shape != latent.shape or not torch.isfinite(value).all():
                    raise ValueError("incomplete native sampling trajectory")
                latents[entry["id"]] = value.detach().cpu()
            management.unload_all_models()
            del sampler, dit, patcher, conditions, negative
            gc.collect()
            state, metadata = utils.load_torch_file(str(vae_path), safe_load=True, return_metadata=True)
            vae = sd.VAE(sd=state, device=torch.device("cuda:0"), dtype=torch.bfloat16, metadata=metadata)
            if type(vae.first_stage_model).__module__ != "comfy.ldm.wan.vae":
                raise ValueError("native Wan VAE required")
            loaded = vae.first_stage_model.state_dict()
            if set(loaded) != set(state) or any(loaded[k].shape != state[k].shape for k in state):
                raise ValueError("incomplete native VAE key/shape coverage")
            del loaded, state
            vae.first_stage_model.eval()
            for entry in entries:
                latent = latents.pop(entry["id"])
                pixels = vae.decode(latent)
                array = png_pixels(pixels)
                target = out / (entry["id"] + ".png")
                Image.fromarray(array).save(target, format="PNG")
                receipt["images"].append({**entry, "file": target.name, "sha256": file_sha(target),
                    "decoded_pixels_sha256": tensor_sha(pixels), "native_sample_latent_sha256": tensor_sha(latent),
                    "provenance": "native Qwen-Image parent generated; no selection; no edits; not a photograph",
                    "visual_review": "pending"})
        for role, path in (("diffusion", weights), ("vae", vae_path), ("text_encoder", te_path)):
            if read_header(path)[1] != identities[role]:
                raise ValueError("source header changed during parent generation")
        receipt["status"] = "complete"
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["seconds_including_teacher_loads"] = time.time() - started
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (out / "parent_images_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--text-encoder", type=Path, required=True)
    p.add_argument("--comfy-root", type=Path, default=Path("/opt/ComfyUI"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--id", action="append", default=[])
    a = p.parse_args()
    generate(a.plan, a.weights, a.vae, a.text_encoder, a.comfy_root, a.out, a.id)


if __name__ == "__main__":
    main()
