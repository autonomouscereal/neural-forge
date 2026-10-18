"""C15 volume probes against the REAL Hunyuan3D-2.1 ShapeVAE, inside the ComfyUI runtime image.

Lane M2 (2026-10-10). The volume band's parent "VAE" is a perceiver: surface point samples
(xyz + normal + sharp-edge flag) are cross-attended into 4,096 latent tokens x 64 channels,
and the decoder answers occupancy logits at arbitrary query points. A raw-rung row (one
local raw patch <-> one latent token) does not exist for it, so this script measures the
parent ceiling only:

* encode: ``encoder(pc, feats) -> pre_kl -> mean`` (the posterior MEAN, deterministic; the
  sampled std is recorded), DiT space = ``latent * Hunyuan3Dv2_1.scale_factor``;
* decode at near-surface probes: ``post_kl -> transformer -> geo_decoder(queries)`` at
  ``p + eps*n`` (label outside) and ``p - eps*n`` (label inside) for surface samples ``p``
  with face normals ``n`` drawn by the parent's own ``load_surface_sharpedge``;
* score: occupancy accuracy and PSNR of sigmoid(logit) against the {0,1} labels (data
  range 1). The logit sign convention is reported both ways; ``convention`` names the one
  with the higher accuracy (a convention probe, not a fit).

Meshes: real GLB/OBJ files on the box (``--meshes-dir``), sorted names, evenly spaced.

    python c15_volume_probe_container.py --ckpt /models/checkpoints/hunyuan_3d_v2.1.safetensors \
        --meshes-dir /meshes --count 16 --out /work/pairs/hy3d_volume_probe --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for _comfy_root in ("/opt/ComfyUI",):
    if Path(_comfy_root).is_dir() and _comfy_root not in sys.path:
        sys.path.append(_comfy_root)

import numpy as np  # noqa: E402
import torch  # noqa: E402

MESH_SUFFIXES = (".glb", ".obj", ".ply")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose(directory: Path, count: int) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in MESH_SUFFIXES)
    if len(files) < count:
        raise SystemExit(f"only {len(files)} meshes under {directory}; need {count}")
    step = len(files) / float(count)
    return [files[int(i * step)] for i in range(count)]


def load_vae(ckpt: Path, device: str):
    import comfy.sd
    from safetensors import safe_open

    state = {}
    with safe_open(str(ckpt), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("vae."):
                state[key[4:]] = handle.get_tensor(key)
    vae = comfy.sd.VAE(sd=state)
    model = vae.first_stage_model
    notes = {}
    if not hasattr(model, "pre_kl") and "pre_kl.weight" in state:
        # ComfyUI's ShapeVAE is decode-only (encode() names self.pre_kl, __init__ never builds
        # it); the checkpoint ships the layer, so attach it from the checkpoint's own tensors
        weight, bias = state["pre_kl.weight"], state.get("pre_kl.bias")
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        with torch.no_grad():
            layer.weight.copy_(weight.float())
            if bias is not None:
                layer.bias.copy_(bias.float())
        model.pre_kl = layer
        notes["pre_kl"] = f"attached from checkpoint vae.pre_kl.* {tuple(weight.shape)} (absent from ComfyUI ShapeVAE)"
    model = model.float().to(device).eval()
    loaded = {k: v for k, v in model.state_dict().items()}
    missing = sorted(k for k in state if k not in loaded)
    mismatched = sorted(k for k in state if k in loaded and not torch.equal(loaded[k].detach().cpu().float(), state[k].float()))
    notes.update({"checkpoint_vae_tensors": len(state), "not_in_module": missing[:20], "not_in_module_count": len(missing), "value_mismatch_count": len(mismatched), "value_mismatch_first": mismatched[:10]})
    return model, notes


def psnr01(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = float((prediction.double() - target.double()).pow(2).mean())
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 volume probes against the Hunyuan3D-2.1 ShapeVAE")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--meshes-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--uniform-points", type=int, default=40960)
    parser.add_argument("--sharp-points", type=int, default=40960)
    parser.add_argument("--probe-points", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if str(args.device).startswith("cpu"):
        # ComfyUI picks its device at import from cli_args; without --cpu it asks CUDA
        import comfy.cli_args

        comfy.cli_args.args.cpu = True
    import comfy.latent_formats
    import comfy.ldm.hunyuan3d.vae as hvae

    # ComfyUI's fps() indexes a CPU arange with a CUDA index (RuntimeError on GPU); same
    # algorithm, arange on the points' device. Recorded in the receipt as a runtime patch.
    def fps_on_device(src, batch, sampling_ratio, start_random=True):
        batch_size = int(batch.max()) + 1
        deg = src.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))
        ptr_vec = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_vec[1:])
        sampled = []
        for b in range(batch_size):
            start, end = ptr_vec[b].item(), ptr_vec[b + 1].item()
            points = src[start:end]
            num_points = points.size(0)
            num_samples = max(1, math.ceil(num_points * sampling_ratio))
            selected = torch.zeros(num_samples, device=src.device, dtype=torch.long)
            distances = torch.full((num_points,), float("inf"), device=src.device)
            farthest = torch.randint(0, num_points, (1,), device=src.device) if start_random else torch.tensor([0], device=src.device, dtype=torch.long)
            for i in range(num_samples):
                selected[i] = farthest
                centroid = points[farthest].squeeze(0)
                distances = torch.minimum(distances, torch.norm(points - centroid, dim=1))
                farthest = torch.argmax(distances)
            sampled.append(torch.arange(start, end, device=src.device)[selected])
        return torch.cat(sampled, dim=0)

    hvae.fps = fps_on_device
    # the parent's surface sampler draws from an UNSEEDED numpy generator; pin it so the
    # probe set is reproducible (recorded in the receipt)
    seeded = np.random.default_rng(int(args.seed))
    hvae.np.random.default_rng = lambda *a, **k: seeded  # type: ignore[attr-defined]
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))

    ckpt = Path(args.ckpt)
    model, vae_tensors = load_vae(ckpt, args.device)
    print(json.dumps(vae_tensors), flush=True)
    expected = int(model.encoder.pc_size) + int(model.encoder.pc_sharpedge_size)
    if int(args.uniform_points) + int(args.sharp_points) != expected:
        raise SystemExit(f"the encoder splits exactly {expected} points (pc_size + pc_sharpedge_size); got "
                         f"{args.uniform_points} + {args.sharp_points}")
    scale_factor = float(comfy.latent_formats.Hunyuan3Dv2_1.scale_factor)
    meshes = choose(Path(args.meshes_dir), int(args.count))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    latents, latent_std, probes, labels, logits_rt = [], [], [], [], []
    per_mesh = []
    for index, path in enumerate(meshes):
        started = time.monotonic()
        mesh = hvae.SharpEdgeSurfaceLoader._load_mesh(str(path))
        surface = hvae.load_surface_sharpedge(
            mesh, num_points=int(args.uniform_points), num_sharp_points=int(args.sharp_points), device=args.device
        ).float()  # [1, P, 7]
        with torch.no_grad():
            pc, feats = surface[:, :, :3], surface[:, :, 3:]
            encoded = model.encoder(pc, feats)
            moments = model.pre_kl(encoded)
            mean, logvar = torch.chunk(moments, 2, dim=-1)
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
            # probes from the uniform (label 0) part of the parent's own samples
            uniform = surface[0, : int(args.uniform_points)]
            pick = torch.randperm(uniform.shape[0], device=uniform.device)[: int(args.probe_points)]
            points, normals = uniform[pick, :3], uniform[pick, 3:6]
            normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            query = torch.cat((points + args.eps * normals, points - args.eps * normals), dim=0)
            label = torch.cat((torch.zeros(points.shape[0]), torch.ones(points.shape[0]))).to(query.device)
            decoded = model.transformer(model.post_kl(mean))
            logit = model.geo_decoder(queries=query.unsqueeze(0), latents=decoded).reshape(-1).float()
        acc_pos = float(((logit > 0).float() == label).float().mean())
        acc_neg = float(((logit < 0).float() == label).float().mean())
        prob_pos = torch.sigmoid(logit)
        per_mesh.append(
            {
                "index": index,
                "file": str(path),
                "sha256": sha256_file(path),
                "faces": int(len(getattr(mesh, "faces", []))),
                "latent_shape": list(mean.shape),
                "latent_rms": float(mean.pow(2).mean().sqrt()),
                "posterior_std_rms": float(std.pow(2).mean().sqrt()),
                "accuracy_logit_gt0_inside": acc_pos,
                "accuracy_logit_lt0_inside": acc_neg,
                "psnr_sigmoid_gt0_inside_db": psnr01(prob_pos, label),
                "psnr_sigmoid_lt0_inside_db": psnr01(1.0 - prob_pos, label),
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        latents.append(mean[0].float().cpu())
        latent_std.append(std[0].float().cpu())
        probes.append(query.float().cpu())
        labels.append(label.float().cpu())
        logits_rt.append(logit.cpu())
        print(f"[{index + 1}/{len(meshes)}] {path.name}: acc(+)={acc_pos:.4f} acc(-)={acc_neg:.4f} {per_mesh[-1]['seconds']}s", flush=True)

    pos = sum(m["accuracy_logit_gt0_inside"] for m in per_mesh)
    neg = sum(m["accuracy_logit_lt0_inside"] for m in per_mesh)
    convention = "logit>0 inside" if pos >= neg else "logit<0 inside"
    key = "gt0" if pos >= neg else "lt0"
    generator = torch.Generator().manual_seed(int(args.seed))
    order = torch.randperm(len(meshes), generator=generator)
    held = max(1, int(round(float(args.holdout_fraction) * len(meshes))))
    split = torch.zeros(len(meshes), dtype=torch.long)
    split[order[:held]] = 1

    from safetensors.torch import save_file

    tensors = {
        "latent_mean": torch.stack(latents).contiguous(),  # VAE space [M, 4096, 64]
        "latent_dit": (torch.stack(latents) * scale_factor).contiguous(),
        "latent_std": torch.stack(latent_std).contiguous(),
        "probe_points": torch.stack(probes).contiguous(),
        "probe_labels": torch.stack(labels).contiguous(),
        "roundtrip_logits": torch.stack(logits_rt).contiguous(),
        "split": split,
    }
    save_file(tensors, str(out / "volume_probes.safetensors"))
    body = {
        "kind": "c15_volume_probes",
        "lane": "M2 C15 volume band (2026-10-10)",
        "ckpt": {"path": str(ckpt), "sha256": sha256_file(ckpt), "vae_tensors": vae_tensors},
        "latent_space": {"vae": "posterior mean of pre_kl(encoder(pc, feats))", "dit": f"vae * Hunyuan3Dv2_1.scale_factor ({scale_factor})"},
        "probe_rule": f"p +/- eps*n, eps {args.eps}, labels outside 0 / inside 1, {args.probe_points} surface samples per mesh",
        "convention": convention,
        "per_mesh": per_mesh,
        "summary": {
            "meshes": len(meshes),
            "mean_accuracy": (pos if key == "gt0" else neg) / len(meshes),
            "mean_psnr_db": sum(m[f"psnr_sigmoid_{key}_inside_db"] for m in per_mesh) / len(meshes),
            "holdout_meshes": [int(i) for i in torch.nonzero(split).flatten()],
            "holdout_mean_psnr_db": sum(per_mesh[int(i)][f"psnr_sigmoid_{key}_inside_db"] for i in torch.nonzero(split).flatten()) / max(1, int(split.sum())),
        },
        "volume_file_sha256": sha256_file(out / "volume_probes.safetensors"),
        "argv": vars(args),
        "runtime_patches": ["comfy.ldm.hunyuan3d.vae.fps replaced in-process by an identical FPS whose final torch.arange lives on the points' device (the shipped one raises on CUDA)", "numpy default_rng seeded for load_surface_sharpedge"],
        "raw_rung": "not applicable: the ShapeVAE is a perceiver over point sets; no local raw row pairs with one latent token",
    }
    (out / "volume_probe_receipt.json").write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(body["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
