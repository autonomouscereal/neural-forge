#!/usr/bin/env python3
"""Preregistered native HY3D parent meshes from frozen QIMG images; not child proof.

Calls the complete original pinned flow pipeline, scheduler, DiT and decoder.
No training, selection/retry search, external decoder, mesh repair or deployment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import torch

from c16_capture_hy3d import (CAPTURE_DIMS, MODEL_PATH, SOURCE_ROOT,
    bounded_bytes, canonical_geometry, cpu, digest_file, finite, geometry_identity,
    inspect_source, native_modules, require, runtime_versions, sealed_json,
    sealed_tensors, stamp, strict_component, unique_json, verify_native_dependencies)

PLAN_SCHEMA = "neural-forge.hy3d-parent-calibration-images.v1"
SCHEMA = "neural-forge.hy3d-parent-meshes.v1"
EXTRA_SOURCE = {
    "hy3dshape/hy3dshape/schedulers.py": "d3b25b84dac26f0e862cfc866da08df2d6592f01a0aaeda965782ab0fc1d8bba",
    "nodes.py": "c5235211f0533352f08f92f4a4f50b98cdc8bd12ab11b596087f6a07f560e8f9",
}
COMFY_SOURCE = {
    "comfy/options.py": "a9d0bb41f5b8d393efc264e83e5f3003ce52c87aca585c0e8ab97bbaea87c8bb",
    "comfy/cli_args.py": "1fed1c93e9bed20ce069fc45b5997479b9229e19b3a4729463ad7ddf3d1403fa",
    "comfy/utils.py": "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
    "comfy/model_management.py": "b7741d049c2c749c316e5769ef520846bc7dd444e67de08403a4ee13a9c034c2",
}
FIXED = {"volume_decoder": "VanillaVolumeDecoder", "surface_extractor": "MCSurfaceExtractor",
         "mc_level": 0.0, "latent_scaling": "none_installed_Hy3D21VAEDecode",
         "seed_device": "cpu", "postprocess": "none", "dtype": "float16"}


def verify_files(root, expected):
    root = Path(root).resolve(strict=True)
    for name, digest in expected.items():
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and hashlib.sha256(bounded_bytes(path)).hexdigest() == digest,
                f"native generation dependency changed; re-audit required: {name}")


def read_plan(path):
    path = Path(path).resolve(strict=True)
    raw = bounded_bytes(path)
    plan = unique_json(raw)
    require(isinstance(plan, dict) and plan.get("schema") == PLAN_SCHEMA, "wrong HY3D parent image plan")
    settings = plan.get("generation")
    variable = {"steps", "guidance_scale", "bounds", "octree_resolution", "query_chunk", "capture_sigma"}
    require(isinstance(settings, dict) and set(settings) == set(FIXED) | variable, "explicit complete generation settings required")
    require(all(settings[k] == v and type(settings[k]) is type(v) for k, v in FIXED.items()), "unreviewed native generation variant")
    for name, low, high in (("steps", 2, 50), ("octree_resolution", 8, 256), ("query_chunk", 1, 2048)):
        require(type(settings[name]) is int and low <= settings[name] <= high, f"invalid bounded {name}")
    for name, low, high in (("guidance_scale", 0., 10.), ("bounds", 0.5, 2.), ("capture_sigma", 0., 1.)):
        require(type(settings[name]) in (int, float) and math.isfinite(settings[name]) and low <= settings[name] <= high,
                f"invalid explicit {name}")
    require(isinstance(plan.get("qimg_receipt_file"), str) and plan["qimg_receipt_file"], "explicit QIMG receipt path required")
    receipt_path = (path.parent / plan["qimg_receipt_file"]).resolve(strict=True)
    require(digest_file(receipt_path, 2 << 20) == plan.get("qimg_receipt_sha256"), "QIMG receipt digest mismatch")
    parent = unique_json(bounded_bytes(receipt_path))
    require(isinstance(parent, dict) and parent.get("schema") == "neural_forge.qimg_parent_images.v1" and parent.get("status") == "complete",
            "completed frozen native QIMG parent receipt required")
    parent_rows = parent.get("images")
    require(isinstance(parent_rows, list) and 1 <= len(parent_rows) <= 8, "invalid bounded QIMG image inventory")
    inventory = {}
    for row in parent_rows:
        require(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] not in inventory,
                "unique QIMG image IDs required")
        inventory[row["id"]] = row
    rows = plan.get("images")
    require(isinstance(rows, list) and 2 <= len(rows) <= 8, "two to eight preregistered images required")
    ids, splits, verified = set(), {}, []
    for row in rows:
        require(isinstance(row, dict), "invalid image entry")
        name = row.get("id")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) and name not in ids,
                "safe unique image ID required")
        require(name in inventory and row.get("split") in ("fit", "held"), "image must belong to frozen QIMG inventory")
        require(type(row.get("seed")) is int and 0 <= row["seed"] < 2**32, "one explicit uint32 mesh seed required")
        original = inventory[name]
        require({"fit": "fit", "holdout": "held"}.get(original.get("split")) == row["split"], "QIMG split cannot be reassigned")
        require(isinstance(row.get("conditioning_file"), str), "explicit conditioning PNG path required")
        image = (path.parent / row["conditioning_file"]).resolve(strict=True)
        require(image.suffix.lower() == ".png", "unmodified QIMG PNG required")
        digest = digest_file(image, 32 << 20)
        require(digest == row.get("conditioning_sha256") == original.get("sha256"), "conditioning is not the exact QIMG image")
        require(splits.setdefault(digest, row["split"]) == row["split"], "conditioning image leaked across fit/held")
        ids.add(name)
        verified.append({**row, "conditioning_file": str(image), "qimg_image": original})
    require(set(splits.values()) == {"fit", "held"}, "both frozen fit/held image partitions required")
    return plan, verified, {"plan_sha256": hashlib.sha256(raw).hexdigest(),
                           "qimg_receipt_sha256": plan["qimg_receipt_sha256"], "qimg_receipt_file": str(receipt_path)}


@contextmanager
def generation_modules(source_root, comfy_root, device):
    """Unchanged whole source modules; namespace-only bypass of UI __init__ exports."""
    verify_files(source_root, EXTRA_SOURCE)
    verify_files(comfy_root, COMFY_SOURCE)
    require(not any(n == "comfy" or n.startswith("comfy.") for n in sys.modules), "standalone fresh Comfy import required")
    old_path, old_argv = list(sys.path), sys.argv
    try:
        sys.path.insert(0, str(Path(comfy_root).resolve()))
        sys.argv = [sys.argv[0]]
        import comfy.options
        from comfy.cli_args import args
        args.cpu = device == "cpu"
        import comfy.model_management as management
        require(not management.in_training and str(management.get_torch_device()) == device, "wrong native device/training mode")
        with native_modules(source_root, include_dit=True) as native:
            prefix = native["vae"].__package__.rsplit(".models.autoencoders", 1)[0]
            # These are actual exports from unchanged computational modules,
            # not stubs or edited function/class bodies.
            exports = sys.modules[prefix + ".models.autoencoders"]
            exports.ShapeVAE = native["vae"].ShapeVAE
            exports.SurfaceExtractors = native["vae"].SurfaceExtractors
            native["pipeline"] = importlib.import_module(prefix + ".pipelines")
            native["scheduler"] = importlib.import_module(prefix + ".schedulers")
            yield native
    finally:
        sys.path[:] = old_path
        sys.argv = old_argv


@contextmanager
def observe_sampling(pipeline, steps):
    """Observe real forwards and scheduler calls; return every result untouched."""
    model, scheduler = pipeline.model, pipeline.scheduler
    require(not model.training and not any(p.requires_grad for p in model.parameters()), "frozen native DiT required")
    frames, evidence = [], {}
    old_step = scheduler.step
    local_step = scheduler.__dict__.get("step")
    had_step = "step" in scheduler.__dict__

    def before(module, args, kwargs):
        require(len(frames) < steps and (not frames or "latent_after" in frames[-1]), "unexpected repeated source forward")
        latent, timestep, context = args[:3]
        require(latent.shape == (2, CAPTURE_DIMS["tokens"], CAPTURE_DIMS["channels"]), "complete two-branch native CFG required")
        require(torch.equal(latent[0], latent[1]) and set(context) == {"main"}, "native CFG input/context changed")
        finite(latent, "native DiT input")
        finite(timestep, "native model time")
        finite(context["main"], "native image context")
        require(context["main"].shape == (2, CAPTURE_DIMS["context_tokens"], CAPTURE_DIMS["context_width"]), "complete image context required")
        value = cpu(context["main"])
        if "context" in evidence:
            require(torch.equal(value, evidence["context"]), "image context changed during trajectory")
        else:
            evidence["context"] = value
        frames.append({"latent_before": cpu(latent[:1]), "model_timestep": cpu(timestep),
                       "blocks": [0] * len(model.blocks)})

    def after(module, args, output):
        finite(output, "native DiT prediction")
        require(output.shape == args[0].shape, "native prediction shape changed")
        frames[-1]["model_output"] = cpu(output)

    def time_token(module, args, output):
        finite(output, "native time token")
        require("time_token" not in frames[-1], "repeated time embedder")
        frames[-1]["time_token"] = cpu(output)

    def block_hook(index):
        def hook(module, args, output):
            finite(output, f"native block {index}")
            frames[-1]["blocks"][index] += 1
        return hook

    def step(model_output, timestep, sample, *args, **kwargs):
        require(frames and "latent_after" not in frames[-1], "unexpected scheduler step")
        frame = frames[-1]
        require(torch.equal(cpu(sample), frame["latent_before"]), "scheduler/model source latent disagreement")
        finite(model_output, "actual CFG scheduler input")
        before_index = scheduler.step_index
        result = old_step(model_output, timestep, sample, *args, **kwargs)
        finite(result.prev_sample, "native scheduler state")
        index = scheduler.step_index - 1
        require(index == len(frames) - 1 and frame["blocks"] == [1] * len(model.blocks), "incomplete native trajectory/block inventory")
        frame.update(cfg_output=cpu(model_output), scheduler_timestep=cpu(timestep),
                     sigma_pair=cpu(scheduler.sigmas[index:index + 2]), latent_after=cpu(result.prev_sample),
                     step_index_before=-1 if before_index is None else before_index, step_index_after=scheduler.step_index)
        return result

    with ExitStack() as hooks:
        hooks.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        hooks.callback(model.register_forward_hook(after).remove)
        hooks.callback(model.t_embedder.register_forward_hook(time_token).remove)
        for index, block in enumerate(model.blocks):
            hooks.callback(block.register_forward_hook(block_hook(index)).remove)
        scheduler.step = step
        try:
            yield evidence, frames
            require(len(frames) == steps and all("latent_after" in f and "time_token" in f for f in frames), "incomplete native sampling")
        finally:
            if had_step:
                scheduler.step = local_step
            else:
                del scheduler.step


def stack_trajectory(evidence, frames):
    fields = ("latent_before", "model_timestep", "model_output", "time_token", "cfg_output",
              "scheduler_timestep", "sigma_pair", "latent_after")
    payload = {"context": evidence["context"]}
    payload.update({"trajectory_" + key: torch.stack([frame[key] for frame in frames]) for key in fields})
    payload["step_index_before"] = torch.tensor([f["step_index_before"] for f in frames], dtype=torch.int64)
    payload["step_index_after"] = torch.tensor([f["step_index_after"] for f in frames], dtype=torch.int64)
    return payload


@torch.inference_mode()
def sample_native(pipeline, image, seed, settings):
    observed_image = {}
    original = pipeline.image_processor

    def processor(*args, **kwargs):
        require(not observed_image, "one native image processing call required")
        result = original(*args, **kwargs)
        require(set(result) == {"image", "mask"}, "native single-image processing required")
        for key, value in result.items():
            finite(value, "processed " + key)
            observed_image["conditioning_" + key] = cpu(value)
        return result

    pipeline.image_processor = processor
    callbacks = []
    def callback(index, timestep, output):
        require(index == len(callbacks), "native callback trajectory changed")
        callbacks.append(index)
    try:
        with observe_sampling(pipeline, settings["steps"]) as (evidence, frames):
            latent = pipeline(image=image, num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                              generator=torch.Generator(device="cpu").manual_seed(seed),
                              enable_pbar=False, callback=callback, callback_steps=1)
        require(callbacks == list(range(settings["steps"])), "incomplete native scheduler callbacks")
        require(torch.equal(cpu(latent), frames[-1]["latent_after"]), "pipeline/scheduler final state disagreement")
        payload = {**stack_trajectory(evidence, frames), **observed_image, "final_latent": cpu(latent)}
        return latent, payload
    finally:
        pipeline.image_processor = original


@torch.inference_mode()
def extract_native(vae, latent, settings):
    import numpy as np
    require(not vae.training and not any(p.requires_grad for p in vae.parameters()), "frozen native VAE required")
    require(vae.geo_decoder.label_type == "binary", "native binary field logits required")
    vae.enable_flashvdm_decoder(enabled=False)
    require(type(vae.volume_decoder).__name__ == "VanillaVolumeDecoder" and
            type(vae.surface_extractor).__name__ == "MCSurfaceExtractor", "fixed native vanilla/MC extraction required")
    original, grid, query_calls = vae.surface_extractor, {}, []
    def surface(logits, **kwargs):
        require(not grid, "unexpected repeated native volume extraction")
        finite(logits, "native binary field grid")
        grid["field_logits"] = cpu(logits)
        return original(logits, **kwargs)
    def query_hook(module, args, output):
        finite(output, "native field query output")
        query_calls.append(output.shape[1])
    vae.surface_extractor = surface
    hook = vae.geo_decoder.register_forward_hook(query_hook)
    try:
        # Installed nodes.py decode path, deliberately NOT pipeline._export:
        # there is NO multiplication/division by the VAE scale_factor here.
        decoded = vae.decode(latent)
        finite(decoded, "native decoded latent")
        meshes = vae.latents2mesh(decoded, bounds=float(settings["bounds"]), mc_level=settings["mc_level"],
                                 num_chunks=settings["query_chunk"], octree_resolution=settings["octree_resolution"],
                                 mc_algo="mc", enable_pbar=False)
        require(len(meshes) == 1 and meshes[0] is not None, "native extraction did not produce one mesh; no retry/repair")
        expected = (settings["octree_resolution"] + 1) ** 3
        require(sum(query_calls) == expected and max(query_calls) <= settings["query_chunk"], "incomplete native dense field queries")
        require(grid.get("field_logits") is not None and grid["field_logits"].shape ==
                (1,) + (settings["octree_resolution"] + 1,) * 3, "complete native dense field grid required")
        mesh = meshes[0]
        vertices = np.asarray(mesh.mesh_v)
        # Native SurfaceExtractor.__call__ explicitly casts its rescaled F64
        # intermediate to F32 before returning Latent2MeshOutput. Preserve that
        # final source boundary; never introduce an additional rounding here.
        require(vertices.dtype == np.float32, "native final mesh dtype changed; re-audit required")
        faces = np.asarray(mesh.mesh_f)[:, ::-1].copy()  # actual installed node winding
        identity = canonical_geometry(vertices, faces)
        return vertices, faces, grid, {"geometry_identity": identity, "query_calls": len(query_calls),
                                      "query_count": expected, "vae_scale_factor_not_applied": vae.scale_factor,
                                      "native_winding_reversed": True, "trimesh_process": False}
    finally:
        hook.remove()
        vae.surface_extractor = original


def write_mesh(path, vertices, faces, expected_identity):
    import numpy as np
    import trimesh
    require(not path.exists(), "refusing to replace native mesh")
    vertices = np.asarray(vertices)
    require(np.array_equal(vertices, vertices.astype(np.float32).astype(vertices.dtype)),
            "PLY F32 storage would round native geometry; refusing lossy serialization")
    # Serialization only: no Trimesh's default process/repair/merge, decimation,
    # normals replacement, rescaling, extra model or external texture references.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    data = mesh.export(file_type="ply", encoding="binary_little_endian")
    require(isinstance(data, bytes) and len(data) <= 256 << 20, "bounded single-file PLY export required")
    with path.open("xb") as handle:
        handle.write(data)
    digest = digest_file(path)
    require(geometry_identity(path, digest) == expected_identity, "PLY serialization changed native triangle geometry")
    return {"object_file": str(path), "object_sha256": digest, "geometry_identity": expected_identity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-checkpoint", default=MODEL_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--comfy-root", default="/opt/ComfyUI")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    plan, rows, plan_identity = read_plan(args.plan)
    settings = plan["generation"]
    output = Path(args.output).resolve()
    require(not output.exists(), "fresh output directory required; preserve partial results")
    existing = output.parent
    while not existing.exists():
        existing = existing.parent
    require(shutil.disk_usage(existing).free >= 6 << 30, "capture must leave bounded disk reserve")
    source = inspect_source(args.parent_checkpoint, args.source_root)
    parent_stamp = stamp(Path(args.parent_checkpoint))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HY3DGEN_DEBUG="0", USE_SAGEATTN="0")
    import yaml
    from PIL import Image
    config = yaml.safe_load(bounded_bytes(Path(args.source_root) / "configs/dit_config_2_1.yaml"))
    with generation_modules(args.source_root, args.comfy_root, args.device) as native, torch.inference_mode():
        vae = strict_component(native["vae"].ShapeVAE(**config["vae"]["params"]), args.parent_checkpoint, "vae.", args.device)
        model = strict_component(native["dit"].HunYuanDiTPlain(**config["model"]["params"]), args.parent_checkpoint, "model.", args.device)
        conditioner = strict_component(native["conditioner"].SingleImageEncoder(**config["conditioner"]["params"]), args.parent_checkpoint, "conditioner.", args.device)
        require(len(model.blocks) == 21, "complete native DiT required")
        pipeline = native["pipeline"].Hunyuan3DDiTFlowMatchingPipeline(
            vae=vae, model=model, scheduler=native["scheduler"].FlowMatchEulerDiscreteScheduler(**config["scheduler"]["params"]),
            conditioner=conditioner, image_processor=native["image"].ImageProcessorV2(**config["image_processor"]["params"]),
            device=args.device, dtype=torch.float16)
        output.mkdir(parents=True, exist_ok=False)
        sealed_json(output / "preregistered-plan.json", {"plan": plan, **plan_identity, "source_identity": source["source_identity"],
                    "extra_source_sha256": EXTRA_SOURCE, "comfy_source_sha256": COMFY_SOURCE})
        receipts, objects, geometry_splits = [], [], {}
        for row in rows:
            require(digest_file(row["conditioning_file"], 32 << 20) == row["conditioning_sha256"], "conditioning image changed")
            with Image.open(row["conditioning_file"]) as image:
                require(image.format == "PNG" and image.mode in ("RGB", "RGBA") and image.size == (512, 512), "exact frozen 512px QIMG PNG required")
                latent, payload = sample_native(pipeline, image, row["seed"], settings)
            vertices, faces, field, extraction = extract_native(vae, latent, settings)
            item = write_mesh(output / (row["id"] + ".ply"), vertices, faces, extraction["geometry_identity"])
            identity = item["geometry_identity"]["sha256"]
            require(geometry_splits.setdefault(identity, row["split"]) == row["split"], "generated geometry crossed fit/held; no seed retry")
            require(digest_file(row["conditioning_file"], 32 << 20) == row["conditioning_sha256"], "conditioning changed during generation")
            tensors = {**payload, **field}
            require(sum(t.numel() * t.element_size() for t in tensors.values()) <= 256 << 20, "per-image source evidence exceeds bound")
            artifact = sealed_tensors(output / (row["id"] + ".safetensors"), tensors)
            receipt = {**row, **item, "trajectory": artifact, "extraction": extraction,
                       "generation": settings, "source_identity": source["source_identity"], **plan_identity,
                       "source_parent_generated": True, "folded_child_proven": False,
                       "field_kind": "native_binary_field_logits_not_metric_sdf"}
            sealed_json(output / (row["id"] + ".json"), receipt)
            receipts.append(receipt)
            objects.append({**item, "conditioning_file": row["conditioning_file"], "conditioning_sha256": row["conditioning_sha256"],
                            "seed": row["seed"], "sigma": settings["capture_sigma"], "split": row["split"],
                            "generation_receipt_file": row["id"] + ".json", "generation_receipt_sha256": digest_file(output / (row["id"] + ".json"))})
            del tensors, payload, field, latent
            print(json.dumps({"id": row["id"], "status": "source_mesh_sealed", "folded_child_proven": False}), flush=True)
        require(stamp(Path(args.parent_checkpoint)) == parent_stamp, "parent changed during generation")
        require(inspect_source(args.parent_checkpoint, args.source_root)["source_identity"] == source["source_identity"], "native source changed")
        verify_native_dependencies(args.source_root)
        verify_files(args.source_root, EXTRA_SOURCE)
        verify_files(args.comfy_root, COMFY_SOURCE)
        require(digest_file(plan_identity["qimg_receipt_file"]) == plan_identity["qimg_receipt_sha256"], "QIMG receipt changed")
        require(digest_file(args.plan) == plan_identity["plan_sha256"], "preregistered plan changed")
        runtime_device = {"device": args.device}
        if args.device.startswith("cuda"):
            runtime_device.update(name=torch.cuda.get_device_name(args.device),
                                  capability=list(torch.cuda.get_device_capability(args.device)))
        sealed_json(output / "objects.json", {"schema": "neural-forge.hy3d-capture-objects.v1", "objects": objects})
        sealed_json(output / "mesh-run.json", {"schema": SCHEMA, "status": "complete", "samples": receipts,
                    "source_identity": source["source_identity"], "runtime_versions": runtime_versions(),
                    "runtime_device": runtime_device, "extra_source_sha256": EXTRA_SOURCE, "comfy_source_sha256": COMFY_SOURCE,
                    "generation": settings, **plan_identity, "source_parent_generated": True,
                    "folded_child_proven": False, "mesh_sdf_pairs_complete": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
