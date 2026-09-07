#!/usr/bin/env python3
"""Native HY3D posterior/anchor/field-logit capture, optionally a complete DiT block.

No training, fixture mode, attention replacement, mesh-SDF claim, or installation.
Run only in an allocated runtime with read-only source, model and object mounts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from c16_hy3d_contract import (CAPTURE_DIMS, CAPTURE_SCHEMA, MODEL_PATH, SOURCE_ROOT,
                             bounded_bytes, inspect_source, require, stamp,
                             tensor_header, unique_json, validate_capture)

SCHEMA = "neural-forge.hy3d-native-posterior-subpath.v1"
PACKAGE_DIR = "hy3dshape/hy3dshape"
# Additional transitive native source involved in this producer, read 2026-09-07.
EXTRA_SHA = {
    "preprocessors.py": "284dc24fabbf974bd6793ebd58e95b7700550d90371cacb96ad0889ec558de1e",
    "utils/__init__.py": "0be86a4b0a9c7885550f80fd6f9900855246f6fb0d8df6000e07414258d6590f",
    "utils/utils.py": "09fda07cb7e7aafbbc2d0a0a808d312d6687bcb1a939c33cdf9f88dc07d42ca0",
    "utils/misc.py": "e8f41fef6db65993b87d64f9a540e75c3b4f8c85f6c703683929d6250ee3ec16",
    "models/autoencoders/attention_processors.py": "9a461ddde9a2bf5ca0cd3961c3c1b86f9c81772066fe5b815bc497e0b5a8f053",
    "models/autoencoders/surface_extractors.py": "203d1551ccef37e86f9809b9ac0fac68e8e4a2f633ca72a85408535595ae68e5",
    "models/autoencoders/volume_decoders.py": "95ffab6bd0a11034bc2b7415fef1379e27c4d2238638f831a0456019d90484eb",
}
PRODUCER = {"native_forward": True, "strict_weights": True, "eval": True,
            "modified_attention": False}
GEOMETRY_SCHEMA = "neural-forge.exact-triangle-geometry.v1"
MAX_GEOMETRY_FACES = 1_000_000


def canonical_geometry(vertices, faces):
    """Exact triangle-set identity, not a semantic/near-duplicate classifier.

    Ignore container metadata, vertex/face order, winding, unused/duplicated
    vertices and repeated triangles. No rounding, centering, rotation, scale
    normalization, retriangulation or source-mesh mutation is performed.
    """
    import numpy as np
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and 0 < len(vertices) <= 3 * MAX_GEOMETRY_FACES,
            "bounded XYZ vertex array required")
    require(vertices.dtype.kind in "fiu" and bool(np.isfinite(vertices).all()), "finite real geometry required")
    require(faces.ndim == 2 and faces.shape[1] == 3 and 0 < len(faces) <= MAX_GEOMETRY_FACES,
            "bounded nonempty triangle array required")
    require(faces.dtype.kind in "iu" and bool((faces >= 0).all()) and bool((faces < len(vertices)).all()),
            "valid integer triangle indices required")
    points = np.array(vertices[faces.reshape(-1)], dtype="<f8", copy=True)
    require(bool(np.isfinite(points).all()), "geometry cannot be represented as finite float64")
    points[points == 0] = 0.0  # canonicalize IEEE negative zero, without rounding
    points, inverse = np.unique(points, axis=0, return_inverse=True)
    triangles = np.unique(np.sort(inverse.reshape(-1, 3), axis=1), axis=0)
    digest = hashlib.sha256((GEOMETRY_SCHEMA + "\n").encode("ascii"))
    digest.update(np.asarray([len(points), len(triangles)], dtype="<u8").tobytes())
    digest.update(np.asarray(points, dtype="<f8").tobytes())
    digest.update(np.asarray(triangles, dtype="<u8").tobytes())
    return {"schema": GEOMETRY_SCHEMA, "sha256": digest.hexdigest(),
            "vertices": len(points), "triangles": len(triangles)}


def geometry_identity(path, expected_sha256):
    """Parse the exact hash-bound single-file input; never follow mesh sidecars."""
    import trimesh
    path = Path(path)
    require(path.suffix.lower() in (".ply", ".stl"), "single-file PLY/STL mesh required")
    before = stamp(path)
    require(before[2] <= 256 << 20, "geometry input too large")
    with path.open("rb") as handle:
        data = handle.read((256 << 20) + 1)
    require(len(data) <= 256 << 20 and before == stamp(path), "geometry input changed during read")
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "geometry input digest mismatch")
    mesh = trimesh.load_mesh(io.BytesIO(data), file_type=path.suffix[1:].lower(), process=False)
    require(isinstance(mesh, trimesh.Trimesh), "single triangle mesh required")
    return canonical_geometry(mesh.vertices, mesh.faces)


def digest_file(path, max_bytes=None):
    path = Path(path)
    before = stamp(path)
    require(path.is_file() and (max_bytes is None or before[2] <= max_bytes), "input file too large/invalid")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    require(before == stamp(path), "file changed while hashing")
    return digest.hexdigest()


def cpu(value):
    return value.detach().to("cpu").contiguous().clone()


def runtime_versions():
    versions = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("safetensors", "numpy", "trimesh", "torch-cluster", "diffusers", "transformers",
                 "torchvision", "opencv-python", "omegaconf", "einops", "scikit-image", "PyYAML", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def finite(value, name):
    for part in value.reshape(-1).split(1 << 20):
        require(bool(torch.isfinite(part).all()), f"nonfinite native tensor: {name}")


def sealed_tensors(path, tensors):
    require(not path.exists(), "refusing to replace capture artifact")
    for name, value in tensors.items():
        require(value.device.type == "cpu" and not value.requires_grad, "capture must be detached CPU evidence")
        finite(value, name)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    return {"file": path.name, "file_sha256": digest_file(path), "bytes": path.stat().st_size}


def sealed_json(path, body):
    # An incomplete directory is preserved on every failure; no resume/overwrite.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


class Proxy:
    """Forward all operations except named observational wrappers, no math shim."""
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


@contextmanager
def observe_sampler(encoder):
    """Observe the audited native randperm/FPS, including duplicate-coordinate rows.

    Only this native module's globals are scoped; torch.randperm itself is never
    monkeypatched. Intended for this standalone single-threaded inference driver.
    """
    require(threading.current_thread() is threading.main_thread(), "sampler observer needs main thread")
    require(encoder.pc_sharpedge_size == 0, "audited sampler requires no sharpedge branch")
    method = encoder.sample_points_and_latents
    namespace = method.__func__.__globals__
    native_torch, native_fps = namespace["torch"], namespace["fps"]
    require(native_torch is torch, "nested/replaced native torch namespace")
    trace = {}

    def randperm(*args, **kwargs):
        require("sampler_permutation" not in trace, "unexpected repeated native randperm")
        result = native_torch.randperm(*args, **kwargs)
        trace["sampler_permutation"] = cpu(result)
        return result

    def fps(*args, **kwargs):
        require("sampler_fps_flat_index" not in trace, "unexpected repeated native FPS")
        result = native_fps(*args, **kwargs)
        trace["sampler_fps_flat_index"] = cpu(result)
        trace["fps_input"] = cpu(args[0])
        return result

    def sampled(pc, feats=None):
        result = method(pc, feats)
        b, points, _ = pc.shape
        permutation, selected = trace["sampler_permutation"], trace["sampler_fps_flat_index"]
        require(points == encoder.pc_size and permutation.shape == (points,), "unexpected native point inventory")
        require(torch.equal(permutation.sort().values, torch.arange(points)), "invalid native permutation")
        require(selected.dtype == torch.int64 and selected.numel() == b * encoder.num_latents,
                "native FPS count changed")
        selected = selected.reshape(b, encoder.num_latents)
        require(torch.equal(selected // points, torch.arange(b)[:, None].expand_as(selected)),
                "native FPS changed batch/token ordering")
        source_ids = permutation[selected % points]
        original = cpu(pc)
        require(torch.equal(trace.pop("fps_input"), original[:, permutation].reshape(b * points, 3)),
                "FPS input no longer follows recorded source permutation")
        anchors = cpu(result[2][0])
        require(torch.equal(original.gather(1, source_ids[..., None].expand(-1, -1, 3)), anchors),
                "native anchor/source indices disagree")
        trace.update(query_source_index=source_ids, anchor_xyz=anchors)
        return result

    had_local = "sample_points_and_latents" in encoder.__dict__
    old_local = encoder.__dict__.get("sample_points_and_latents")
    namespace.update(torch=Proxy(native_torch, randperm=randperm), fps=fps)
    encoder.sample_points_and_latents = sampled
    try:
        yield trace
    finally:
        namespace.update(torch=native_torch, fps=native_fps)
        if had_local:
            encoder.sample_points_and_latents = old_local
        else:
            del encoder.sample_points_and_latents


def verify_native_dependencies(source_root):
    root = Path(source_root).resolve(strict=True) / PACKAGE_DIR
    for name, expected in EXTRA_SHA.items():
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and hashlib.sha256(bounded_bytes(path)).hexdigest() == expected,
                f"native dependency changed; re-audit required: {name}")
    return root


@contextmanager
def native_modules(source_root, include_dit=True):
    """Load genuine source under a private package namespace, skipping UI exports.

    Native top-level __init__ imports Comfy pipeline/postprocessors unnecessarily.
    Namespace containers below provide paths only, never replacement functions.
    All computational modules (including native utils) execute their actual files.
    """
    require(os.environ.get("USE_SAGEATTN", "0") == "0", "native unmodified attention required")
    root = verify_native_dependencies(source_root)
    prefix = "_nf_hy3d_capture_native"
    require(not any(n == prefix or n.startswith(prefix + ".") for n in sys.modules), "native namespace already loaded")
    try:
        for suffix in ("", ".models", ".models.autoencoders", ".models.denoisers"):
            name = prefix + suffix
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.strip(".").split("."))) if suffix else str(root)]
            module.__package__ = name
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            module.__spec__.submodule_search_locations = module.__path__
            sys.modules[name] = module
        result = {}
        imports = {"vae": "models.autoencoders.model", "surface": "surface_loaders"}
        if include_dit:
            imports.update(dit="models.denoisers.hunyuandit", conditioner="models.conditioner", image="preprocessors")
        for name, suffix in imports.items():
            result[name] = importlib.import_module(prefix + "." + suffix)
        result["extra_source_sha256"] = dict(EXTRA_SHA)
        yield result
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def strict_component(module, parent, prefix, device):
    """Read only this combined-safetensors component; never invoke pickle loaders."""
    parent = Path(parent)
    before = stamp(parent)
    header = tensor_header(parent)
    expected = module.state_dict()
    actual = {n[len(prefix):]: item for n, item in header["tensors"].items() if n.startswith(prefix)}
    require(set(actual) == set(expected), f"strict native {prefix} key mismatch")
    for name, value in expected.items():
        require(actual[name]["shape"] == list(value.shape), f"strict native shape mismatch: {prefix}{name}")
        require(actual[name]["dtype"] == "F16", f"unreviewed source storage dtype: {prefix}{name}")
    with safe_open(str(parent), framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(prefix + name) for name in expected}
    for name, value in state.items():
        finite(value, prefix + name)
    module.load_state_dict(state, strict=True, assign=True)
    require(before == stamp(parent), "parent changed during strict load")
    module.eval().requires_grad_(False).to(device=device, dtype=torch.float16)
    require(not any(p.is_meta for p in module.parameters()), "unmaterialized native parameter")
    return module


@torch.inference_mode()
def capture_posterior(vae, surface, seed, query_chunk=256):
    require(query_chunk > 0, "query chunk must be positive")
    require(not vae.training and not any(p.requires_grad for p in vae.parameters()), "frozen eval VAE required")
    require(vae.geo_decoder.label_type == "binary", "unreviewed native field interpretation")
    require(surface.ndim == 3 and surface.shape[1:] == (CAPTURE_DIMS["surface_points"], 7), "wrong native surface shape")
    finite(surface, "surface")
    torch.manual_seed(seed)
    with observe_sampler(vae.encoder) as trace:
        posterior = vae.encode(surface, sample_posterior=False)
    require(posterior.shape == (surface.shape[0], CAPTURE_DIMS["tokens"], CAPTURE_DIMS["channels"]),
            "wrong native posterior shape")
    decoded = vae.decode(posterior)
    anchors = trace["anchor_xyz"].to(device=surface.device, dtype=surface.dtype)
    logits = []
    for part in anchors.split(query_chunk, dim=1):
        logits.append(cpu(vae.geo_decoder(queries=part, latents=decoded)))
    payload = {**trace, "surface": cpu(surface), "posterior_mode": cpu(posterior),
               "field_queries": cpu(anchors), "parent_field_logits": torch.cat(logits, dim=1)}
    require(payload["parent_field_logits"].shape == (*anchors.shape[:2], 1), "wrong native field output shape")
    return payload, posterior


@torch.inference_mode()
def capture_dit(model, latent, sigma, context, layer):
    """Observe one block while running the complete source DiT, including all MoE."""
    require(type(layer) is int and 0 <= layer < len(model.blocks), "bad native block index")
    require(not model.training and not any(p.requires_grad for p in model.parameters()), "frozen eval DiT required")
    require(bool(((sigma >= 0) & (sigma <= 1)).all()), "sigma must be native normalized flow time")
    block, captured = model.blocks[layer], {}

    def put(name, value):
        require(name not in captured, f"unexpected repeated native hook: {name}")
        captured[name] = cpu(value)

    def output(name):
        return lambda module, args, result: put(name, result)

    def before(module, args, kwargs):
        put("block_input", args[0])
        put("context", args[2] if len(args) > 2 else kwargs["text_states"])
        skip = kwargs.get("skip_value", args[3] if len(args) > 3 else None)
        if skip is not None:
            put("skip_value", skip)

    with ExitStack() as stack:
        def hook(module, callback):
            stack.callback(module.register_forward_hook(callback).remove)
        stack.callback(block.register_forward_pre_hook(before, with_kwargs=True).remove)
        hook(block, output("block_output"))
        hook(model.t_embedder, output("time_token"))
        hook(block.attn1, output("self_attention_write"))
        hook(block.attn2, output("cross_attention_write"))
        mlp = block.moe if block.use_moe else block.mlp
        stack.callback(mlp.register_forward_pre_hook(lambda m, args: put("mlp_input", args[0])).remove)
        hook(mlp, output("mlp_output"))
        if block.use_moe:
            gate = mlp.gate
            namespace, native_f = gate.forward.__func__.__globals__, gate.forward.__func__.__globals__["F"]
            require(not isinstance(native_f, Proxy), "nested gate observer")

            def linear(x, weight, *args, **kwargs):
                result = native_f.linear(x, weight, *args, **kwargs)
                if weight is gate.weight:
                    put("router_logits", result.reshape(latent.shape[0], latent.shape[1] + 1, -1))
                return result

            namespace["F"] = Proxy(native_f, linear=linear)
            stack.callback(namespace.__setitem__, "F", native_f)

            def routed(module, args, result):
                b, n = args[0].shape[:2]
                put("router_topk_ids", result[0].reshape(b, n, -1))
                put("router_topk_weights", result[1].reshape(b, n, -1))
            hook(gate, routed)
            hook(mlp.shared_experts, output("shared_output"))
        final = model(latent, sigma, context)
        finite(final, "complete_native_dit_output")
    required = {"block_input", "block_output", "mlp_input", "mlp_output", "context", "time_token",
                "self_attention_write", "cross_attention_write"}
    if layer > 10:
        required.add("skip_value")
    if block.use_moe:
        required.update(("router_logits", "router_topk_ids", "router_topk_weights", "shared_output"))
    require(set(captured) == required, "native hook inventory changed")
    captured.update(sigma=cpu(sigma), timestep=cpu(sigma))
    return captured, cpu(final)


def read_objects(path):
    path = Path(path).resolve(strict=True)
    body = unique_json(bounded_bytes(path))
    require(isinstance(body, dict) and body.get("schema") == "neural-forge.hy3d-capture-objects.v1", "wrong objects manifest")
    rows = body.get("objects")
    require(isinstance(rows, list) and 2 <= len(rows) <= 8, "bounded capture needs 2..8 fit/held entries")
    splits, geometry_splits, result = {}, {}, []
    for row in rows:
        require(isinstance(row, dict) and row.get("split") in ("fit", "held"), "invalid object split")
        require(type(row.get("seed")) is int and 0 <= row["seed"] < 2**32, "explicit uint32 seed required")
        require(type(row.get("sigma")) in (int, float) and math.isfinite(row["sigma"]) and 0 <= row["sigma"] <= 1,
                "explicit sigma in [0,1] required")
        item = dict(row)
        for key, digest_key in (("object_file", "object_sha256"), ("conditioning_file", "conditioning_sha256")):
            require(isinstance(row.get(key), str), f"missing input path: {key}")
            file = (path.parent / row[key]).resolve(strict=True)
            require(digest_file(file, 256 << 20) == row.get(digest_key), f"input digest mismatch: {key}")
            item[key] = str(file)
        require(Path(item["object_file"]).suffix.lower() in (".ply", ".stl"), "single-file PLY/STL mesh required")
        require(splits.setdefault(row["object_sha256"], row["split"]) == row["split"], "object leaked across fit/held")
        geometry = geometry_identity(item["object_file"], item["object_sha256"])
        require("geometry_identity" not in row or row["geometry_identity"] == geometry,
                "declared geometry identity mismatch")
        require(geometry_splits.setdefault(geometry["sha256"], row["split"]) == row["split"],
                "identical geometry leaked across fit/held")
        item["geometry_identity"] = geometry
        result.append(item)
    require(set(splits.values()) == {"fit", "held"}, "fit and held objects required")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", default=MODEL_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block", type=int, choices=range(21))
    parser.add_argument("--query-chunk", type=int, default=256)
    args = parser.parse_args(argv)
    require(args.query_chunk > 0, "query chunk must be positive")
    rows = read_objects(args.objects)
    source = inspect_source(args.parent_checkpoint, args.source_root)
    parent_stamp = stamp(Path(args.parent_checkpoint))
    out = Path(args.output).resolve()
    require(not out.exists(), "fresh output directory required; partial captures must be preserved")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import yaml
    from PIL import Image
    config = yaml.safe_load(bounded_bytes(Path(args.source_root) / "configs/dit_config_2_1.yaml"))
    with native_modules(args.source_root, include_dit=args.block is not None) as native, torch.inference_mode():
        vae = strict_component(native["vae"].ShapeVAE(**config["vae"]["params"]),
                               args.parent_checkpoint, "vae.", args.device)
        model = conditioner = None
        if args.block is not None:
            model = strict_component(native["dit"].HunYuanDiTPlain(**config["model"]["params"]),
                                     args.parent_checkpoint, "model.", args.device)
            conditioner = strict_component(native["conditioner"].SingleImageEncoder(**config["conditioner"]["params"]),
                                           args.parent_checkpoint, "conditioner.", args.device)
        loader = native["surface"].SharpEdgeSurfaceLoader(num_uniform_points=81920, num_sharp_points=0)
        processor = native["image"].ImageProcessorV2(**config["image_processor"]["params"]) if model is not None else None
        out.mkdir(parents=True, exist_ok=False)
        sealed_json(out / "source.json", {**source, "extra_source_sha256": native["extra_source_sha256"]})
        receipts, captures, samples = [], [], []
        for index, row in enumerate(rows):
            mesh_stamp, image_stamp = stamp(Path(row["object_file"])), stamp(Path(row["conditioning_file"]))
            # Bind the actual later sampler inputs to the already checked split
            # inventory even if a file was replaced during model initialization.
            for key, digest_key in (("object_file", "object_sha256"), ("conditioning_file", "conditioning_sha256")):
                require(digest_file(row[key], 256 << 20) == row[digest_key], "capture input digest changed")
            surface = loader(row["object_file"]).to(device=args.device, dtype=torch.float16)
            payload, posterior = capture_posterior(vae, surface, row["seed"], args.query_chunk)
            sample = {key: row[key] for key in ("object_sha256", "conditioning_sha256", "seed", "split", "geometry_identity")}
            samples.append(sample)
            if model is not None:
                with Image.open(row["conditioning_file"]) as image:
                    inputs = processor(image)
                condition = conditioner(**{k: v.to(args.device) for k, v in inputs.items()})
                require(set(condition) == {"main"}, "native single-image conditioning required")
                sigma = torch.tensor([row["sigma"]], device=args.device, dtype=posterior.dtype)
                generator = torch.Generator(device=args.device).manual_seed(row["seed"])
                noise = torch.randn(posterior.shape, generator=generator, device=args.device, dtype=posterior.dtype)
                # Object-conditioned flow-path probe, NOT a generated scheduler trajectory.
                latent = (1 - sigma[:, None, None]) * (posterior * vae.scale_factor) + sigma[:, None, None] * noise
                captured, final = capture_dit(model, latent, sigma, condition, args.block)
                payload.update(dit_latent_input=cpu(latent), dit_noise=cpu(noise), dit_output=final,
                               conditioning_image=cpu(inputs["image"]), conditioning_mask=cpu(inputs["mask"]))
                for key in ("self_attention_write", "cross_attention_write"):
                    payload[key] = captured.pop(key)
                captured.update(surface=payload["surface"], query_source_index=payload["query_source_index"],
                                latent_xyz=payload["anchor_xyz"])
                captures.append(captured)
            require(mesh_stamp == stamp(Path(row["object_file"])) and image_stamp == stamp(Path(row["conditioning_file"])),
                    "object/conditioning input changed during capture")
            file = sealed_tensors(out / f"sample-{index:04d}.safetensors", payload)
            receipt = {**sample, **file, "requested_sigma": row["sigma"]}
            sealed_json(out / f"sample-{index:04d}.json", receipt)
            receipts.append(receipt)
            del payload, posterior, surface
        dit_receipt = None
        if captures:
            block_dir = out / f"block-{args.block:02d}"
            block_dir.mkdir()
            combined = {key: torch.cat([part[key] for part in captures], dim=0) for key in captures[0]}
            require(sum(v.numel() * v.element_size() for v in combined.values()) < 2 << 30, "native capture exceeds 2 GiB contract")
            file = sealed_tensors(block_dir / "capture.safetensors", combined)
            sealed_json(block_dir / "capture.json", {"schema": CAPTURE_SCHEMA, "kind": "dit_block", **file,
                        "source_identity": source["source_identity"], "samples": samples, "producer": PRODUCER,
                        "block_index": args.block, "attention": "self_then_image_cross",
                        "time_path": "sigma_prepended_token", "position_path": "sampled_anchors_no_dit_rope"})
            dit_receipt = validate_capture(block_dir, source)
        require(parent_stamp == stamp(Path(args.parent_checkpoint)), "parent changed during native capture")
        require(inspect_source(args.parent_checkpoint, args.source_root)["source_identity"] == source["source_identity"],
                "native source changed during capture")
        verify_native_dependencies(args.source_root)
        report = {"schema": SCHEMA, "source_identity": source["source_identity"], "producer": PRODUCER,
                  "samples": receipts, "dit_capture": dit_receipt, "posterior": "mode", "vae_scale_factor": vae.scale_factor,
                  "field_kind": "native_binary_field_logits_not_metric_sdf", "field_queries": "exact_encoder_anchors",
                  "seed_scope": "torch_randperm_fps_and_dit_noise",
                  "surface_replay": "sealed_native_surface; native_numpy_loader_also_uses_unseeded_entropy",
                  "dit_probe": "object_posterior_flow_interpolation_not_generated_scheduler_trajectory",
                  "d10_sdf_pairs_complete": False, "folded_child_proven": False,
                  "runtime_versions": runtime_versions()}
        sealed_json(out / "capture-run.json", report)
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
