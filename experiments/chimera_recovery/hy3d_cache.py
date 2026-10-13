"""Authentic HY3D trajectory math and fail-closed native preparation preflight.

Captured parent trajectories are genuine image-conditioned 3D teachers. Their
native conditioning/query-position adapter is a separate requirement: clean
posterior XYZ, a regular guessed grid and caption-only inputs are not substitutes.
This module proves cached velocity/time/Euler/mesh contracts without loading a
model, and exposes an explicitly experimental tensor packer for audited callers.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_hy3d_common", HERE / "music_cache.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
M, R, Error = C.M, C.R, C.Error
TEACHER = "hy3d_trajectories"
TIME_CONVENTION = "t_native: model timestep input, 0 = pure noise, 1 = data (HY3D 2.1 flow); noise_level = 1 - t_native"
PIPELINE_SHA = "cf56918d62d5bee11a80641638dbc590bd541a1264c384e302a87d68a4cd285b"
SOURCE_PINS = {
    "hy3dshape/hy3dshape/preprocessors.py": "284dc24fabbf974bd6793ebd58e95b7700550d90371cacb96ad0889ec558de1e",
    "configs/dit_config_2_1.yaml": "a04b6a62f7084e1490eaba825355d340983bcbc88809bb36cad37b35979845aa",
    "hy3dshape/hy3dshape/models/denoisers/hunyuandit.py": "27a74c5ba210468847aaf348bfd31ac2781dce094960ede0b2050d94233c531e",
}


def source_contract(capture_path):
    """Inspect exact captured source without importing its model or weights."""
    sources = M.read_json(capture_path)["hy3d_sources"]
    for name, expected in SOURCE_PINS.items():
        source = sources[name]
        if source.get("sha256") != expected or hashlib.sha256(source["text"].encode()).hexdigest() != expected:
            raise Error("HY3D captured preprocessor/config/denoiser source bytes differ")
    config = sources["configs/dit_config_2_1.yaml"]["text"]
    denoiser = sources["hy3dshape/hy3dshape/models/denoisers/hunyuandit.py"]["text"]
    # These facts are tied to full source SHA pins, not untrusted booleans.
    if "use_pos_emb: false" not in config or "use_attention_pooling: false" not in config or "x = torch.cat([c, x], dim=1)" not in denoiser:
        raise Error("HY3D source no-position/shared-time-token contract differs")
    return {"source_capture_sha256": R.sha(capture_path), "sources": SOURCE_PINS,
            "parent_latent_positions": "none", "parent_learned_query_slots": False,
            "parent_latent_permutation_equivariant": True, "shared_time_token": True,
            "conditioner": "actual DINO image+mask context; cross-attention in every parent block",
            "native_block_cross_attention": "dropped in existing composite; conditioning requires shared trunk",
            "native_bridge_spatial_features": "linear and Fourier position projections",
            "native_hybrid_trunk_positions": "sequential RoPE and causal/linear recurrence; not parent permutation equivariant",
            "zero_external_coordinate_policy_verified": False}


def recenter_image(image, border_ratio=.15):
    """Pinned ImageProcessorV2.recenter operations in their exact source order."""
    import cv2
    import numpy as np
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise Error("HY3D original image must decode as uint8 BGR/BGRA")
    if image.shape[-1] == 4:
        mask = image[..., 3]
    else:
        mask = np.ones_like(image[..., 0:1]) * 255
        image = np.concatenate([image, mask], axis=-1)
        mask = mask[..., 0]
    height, width, channels = image.shape
    size = max(height, width)
    result = np.zeros((size, size, channels), dtype=np.uint8)
    coords = np.nonzero(mask)
    if not len(coords[0]):
        raise Error("HY3D original image has an empty alpha mask")
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h, w = x_max - x_min, y_max - y_min
    if h == 0 or w == 0:
        raise Error("HY3D original image has an empty extent")
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2, w2 = int(h * scale), int(w * scale)
    if min(h2, w2) < 1:
        raise Error("HY3D original image has a degenerate recentered extent")
    x2_min, y2_min = (size - h2) // 2, (size - w2) // 2
    result[x2_min:x2_min + h2, y2_min:y2_min + w2] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    bg = np.ones((size, size, 3), dtype=np.uint8) * 255
    mask = result[..., 3:].astype(np.float32) / 255
    result = result[..., :3] * mask + bg * (1 - mask)
    return result.clip(0, 255).astype(np.uint8), (mask * 255).clip(0, 255).astype(np.uint8)


def preprocess_image(image_path, asset_sha):
    """Exact teacher string-path BGR→RGB preprocessing, not generic RGB resize."""
    import cv2
    import torch
    if R.sha(image_path) != R.require_sha(asset_sha, "asset_sha256"):
        raise Error("HY3D original image bytes changed")
    image, mask = recenter_image(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)[..., None]
    def tensor(value):
        return (torch.tensor(value).float() / 255 * 2 - 1).permute(2, 0, 1).contiguous()
    return tensor(image), tensor(mask)


def reconstruct_image(image_path, asset_sha, tensors):
    import torch
    image, mask = preprocess_image(image_path, asset_sha)
    if not torch.equal(image, tensors["image_input"]) or not torch.equal(mask, tensors["image_mask"]):
        raise Error("HY3D reconstructed image/mask differ from authentic teacher tensors")
    return {"preprocessor_sha256": SOURCE_PINS["hy3dshape/hy3dshape/preprocessors.py"], "asset_sha256": asset_sha,
            "image_input_sha256": R.tensor_sha256(image), "image_mask_sha256": R.tensor_sha256(mask),
            "image_bitwise_equal": True, "mask_bitwise_equal": True, "image_shape": list(image.shape),
            "normalization": "uint8 / 255 * 2 - 1", "border_ratio": .15,
            "conditioning_features_ready": False}


def validate_trajectory(tensors, sidecar, *, require_native_shape=True):
    """Check every real stored step, including last endpoint, geometry and CFG."""
    import torch
    required = {"x_t", "v_guided", "t_native", "noise_level", "sigma_next", "x0_final", "context_cond", "image_input", "image_mask", "mesh_vertices", "mesh_faces"}
    if set(tensors) != required or any(value.requires_grad or value.device.type != "cpu" or not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise Error("HY3D cached source requires actual detached finite CPU trajectory/conditioning/mesh tensors")
    xt, v, time = tensors["x_t"], tensors["v_guided"], tensors["t_native"]
    steps = sidecar.get("steps")
    if (steps != 30 or xt.ndim != 3 or v.shape != xt.shape or xt.shape[0] != steps or xt.shape[2] != 64 or
            (require_native_shape and xt.shape[1] != 4096) or tensors["x0_final"].shape != xt.shape[1:] or
            time.shape != (steps,) or tensors["noise_level"].shape != (steps,) or tensors["sigma_next"].shape != (steps,) or
            sidecar.get("time_convention") != TIME_CONVENTION or sidecar.get("guidance") != 5. or
            any(tensors[key].dtype != torch.float32 for key in required - {"mesh_faces"})):
        raise Error("HY3D source shape/dtype/time/guided CFG5 contract differs")
    if not torch.allclose(time, torch.linspace(0., 1., steps), atol=1e-7, rtol=1e-6):
        raise Error("HY3D actual model timestep is not the captured linspace0→1 schedule")
    if not torch.equal(tensors["noise_level"], 1. - time) or not torch.allclose(tensors["sigma_next"][:-1], time[1:], atol=1e-7, rtol=1e-6):
        raise Error("HY3D noise_level or sigma_next differs from actual teacher convention")
    next_time = tensors["sigma_next"]
    if bool((next_time < time).any()) or bool((next_time > 1.).any()):
        raise Error("HY3D scheduler does not increase parent flow time toward data")
    predicted = xt + (next_time - time)[:, None, None] * v
    expected = torch.cat((xt[1:], tensors["x0_final"][None, :]), 0)
    scale = max(1e-30, float(xt.abs().max()), float(tensors["x0_final"].abs().max()))
    residual = float((predicted - expected).abs().max()) / scale
    if residual > 2e-6 or sidecar.get("checks", {}).get("recording_euler_chain_residual_rel", float("inf")) > 2e-6:
        raise Error("HY3D actual stored trajectory is not its recorded Euler chain/final latent")
    reference = sidecar.get("checks", {}).get("reference_upstream_pipeline", {})
    if (reference.get("pipeline_sha256") != PIPELINE_SHA or reference.get("bitwise_equal") is not True or
            reference.get("max_abs_diff") != 0.):
        raise Error("HY3D source lacks actual upstream pipeline final-latent equality")
    vertices, faces = tensors["mesh_vertices"], tensors["mesh_faces"]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 4 or faces.ndim != 2 or faces.shape[1] != 3 or
            faces.shape[0] < 4 or faces.dtype != torch.int32 or bool(((faces < 0) | (faces >= vertices.shape[0])).any()) or
            sidecar.get("mesh", {}).get("vertices") != vertices.shape[0] or sidecar["mesh"].get("faces") != faces.shape[0] or
            sidecar["mesh"].get("octree_resolution") != 256 or sidecar["mesh"].get("num_chunks") != 8000):
        raise Error("HY3D targets lack genuine finite decoded vertex/face geometry")
    tri = vertices[faces.to(torch.int64)]
    if not bool((torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1) > 1e-10).any()):
        raise Error("HY3D decoded triangles are all degenerate")
    if int(torch.linalg.matrix_rank(vertices - vertices.mean(0))) < 3:
        raise Error("HY3D decoded geometry is planar; not a genuine volume target")
    if (tensors["image_input"].shape != (3, 512, 512) or tensors["image_mask"].shape != (1, 512, 512) or
            tensors["context_cond"].shape != (1370, 1024)):
        raise Error("HY3D image-conditioned teacher requires complete actual image/mask/DINO context")
    positive = (tensors["noise_level"] > 0).nonzero().flatten().tolist()
    return {"stored_steps": steps, "positive_noise_steps": positive, "excluded_clean_endpoint_steps": sorted(set(range(steps)) - set(positive)),
            "euler_chain_residual_rel": residual, "vertices": vertices.shape[0], "faces": faces.shape[0],
            "parent_sign": 1., "target_conversion": "canonical_velocity = -v_guided", "native_shape_verified": require_native_shape,
            "geometry_semantic_quality_claim": False, "image_conditioning_required": True}


def audit_source_shard(cache, items_path, index, source_sha, source_capture):
    """Bounded real CPU audit of a partial cache; never grants production admission."""
    import torch
    from safetensors.torch import load_file
    if torch.cuda.is_initialized():
        raise Error("HY3D source audit is CPU only")
    contract = source_contract(source_capture)
    corpus = M.read_json(items_path)
    if (corpus.get("schema") != "omni-campaign.teacher_corpus.v1" or corpus.get("kind") != "hy3d_images" or
            type(index) is not int or not 0 <= index < len(corpus["items"])):
        raise Error("HY3D actual source corpus/index differs")
    item = corpus["items"][index]
    path = cache / f"shard-{index:05d}.safetensors"
    sidecar_path = path.with_suffix(".json")
    sidecar = M.read_json(sidecar_path)
    config = M.read_json(cache / "cache_config.json")
    if (R.sha(path) != R.require_sha(source_sha, "source_shard_sha256") or sidecar.get("file_sha256") != source_sha or
            sidecar.get("schema") != "omni-campaign.teacher_cache_shard.v2" or sidecar.get("teacher") != TEACHER or
            sidecar.get("index") != index or sidecar.get("item_id") != item["id"] or sidecar.get("item_sha256") != item["sha256"] or
            sidecar.get("corpus_sha256") != corpus["corpus_sha256"] or sidecar.get("config_sha256") != R.digest(config) or
            config.get("teacher") != TEACHER or sidecar.get("finite") is not True or sidecar.get("item") != item):
        raise Error("HY3D raw source shard/sidecar/item/config identity differs")
    tensors = load_file(str(path), device="cpu")
    trajectory = validate_trajectory(tensors, sidecar)
    image = reconstruct_image(Path(item["path"]), item["sha256"], tensors)
    return {"schema": "chimera_recovery.hy3d_source_audit.v1", "sealed": False, "production_ready": False,
            "source_shard_sha256": source_sha, "source_sidecar_sha256": R.sha(sidecar_path), "source_items_sha256": R.sha(items_path),
            "item_id": item["id"], "held": item["held"], "contract": contract, "trajectory": trajectory, "image": image,
            "training_started": False, "model_weights_loaded": False, "geometry_training_admission_granted": False,
            "adapter_sha256": R.sha(Path(__file__))}


def geometry_admission(entry, item):
    if (entry.get("asset_sha256") != item["sha256"] or entry.get("teacher_item_sha256") != item["sha256"] or
            entry.get("geometry_source_category") not in {"physical_object_photo", "physical_object_render"} or
            entry.get("genuine_geometry_target") is not True or entry.get("source_is_chart_or_icon") is not False):
        raise Error("HY3D requires reviewed true physical-object geometry admission; captured charts/icons are excluded")
    if item.get("held") or entry.get("train_admitted") is not True or entry.get("split") != "train" or entry.get("blockers"):
        raise Error("HY3D source is held or has no training admission")
    return True


def experimental_native_pack(noisy, velocity, *, prefix_ids, latent_positions, image_features, image_positions, sigma, max_length):
    """Concrete native tensor allocation; caller-supplied positions are unproven.

    This returns an experimental batch/proof only, never a sealed ready receipt.
    It preserves leading time-row ownership and independent image conditioning.
    """
    import torch
    n, images = noisy.shape[0], image_features.shape[0]
    if (noisy.shape != (n, 64) or velocity.shape != noisy.shape or latent_positions.shape != (n, 4) or
            image_features.shape != (images, 1536) or images < 1 or image_positions.shape != (images, 4) or
            not 0 < sigma <= 1 or not prefix_ids or any(type(token) is not int or not 0 <= token < R.TEXT_VOCAB for token in prefix_ids) or
            any(value.requires_grad or value.device.type != "cpu" or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in (noisy, velocity, latent_positions, image_features, image_positions))):
        raise Error("experimental HY3D native noisy/velocity/image/position contract differs")
    media = images + 1 + n
    if len(prefix_ids) + media > max_length:
        raise Error("HY3D image+time+4096-volume rows exceed context; no truncation")
    features, target = torch.zeros(1, media, 6144), torch.zeros(1, media, 6144)
    types = torch.full((1, media), 7, dtype=torch.int64)
    types[:, :images] = 6
    features[0, :images, 640:2176] = image_features
    features[0, images + 1:, 576:640] = noisy
    target[0, images + 1:, 320:384] = -velocity
    positions = torch.cat((image_positions, torch.full((1, 4), -1.), latent_positions), 0)[None, :]
    weights = torch.zeros(1, media)
    weights[:, images + 1:] = 1.
    levels = torch.zeros(1, media)
    levels[:, images:] = sigma
    batch = {"input_ids": torch.tensor([prefix_ids + [M.MEDIA_PLACEHOLDER] * media], dtype=torch.int64),
             "native_media_features": features, "native_media_targets": target, "native_media_type_ids": types,
             "native_media_positions": positions, "native_media_indices": torch.arange(len(prefix_ids), len(prefix_ids) + media)[None, :],
             "native_media_valid_mask": torch.ones(1, media, dtype=torch.bool), "native_media_loss_mask": weights,
             "native_media_noise_levels": levels}
    return batch, {"production_ready": False, "experimental": True, "time_row": images, "time_row_loss_weight": 0.,
                   "image_conditioning_rows": images, "volume_target_rows": n, "input_layout": "feature", "target_layout": "emission",
                   "position_policy_established": False, "requires_native_route_conditioning_amendment": True}


def preflight(cache, items_path, manifest_sha, admission_path=None, *, source_audit_path=None):
    blockers, verified = [], {}
    try:
        manifest, corpus = C.sealed_source(cache, items_path, manifest_sha, teacher=TEACHER)
        verified["teacher_manifest_sha256"] = manifest_sha
        verified["teacher_items"] = corpus["n_items"]
    except (Error, OSError, KeyError, TypeError, ValueError) as error:
        blockers.append(str(error))
        manifest, corpus = None, None
    if admission_path is None:
        blockers.append("missing connected-group/complete-held HY3D geometry admission; charts/icons cannot be admitted")
    else:
        try:
            _, rows = M.media_admission(admission_path, R.sha(items_path), kind="hy3d")
            approved = []
            if corpus:
                for item in corpus["items"]:
                    row = rows.get(item["id"])
                    if row and row["train_admitted"]:
                        geometry_admission(row, item)
                        approved.append(item["id"])
            if not approved:
                raise Error("no approved true physical-object HY3D sources")
            verified["approved_geometry_items"] = len(approved)
        except (Error, OSError, KeyError, TypeError, ValueError) as error:
            blockers.append(str(error))
    if source_audit_path:
        try:
            verified["source_contract"] = source_contract(source_audit_path)
        except (Error, OSError, KeyError, TypeError, ValueError) as error:
            blockers.append(str(error))
    # The captured cache lacks latent_xyz and has no verified input-coordinate
    # contract. There is deliberately no boolean bypass or made-up zero grid.
    blockers.extend(["native image-conditioning route needs independently source-bound QVL image features; DINO cross-attention was dropped from native HY3D unit",
                     "native query-position policy is unresolved: parent has no query positions, existing bridge/trunk add positions and ordering; clean-posterior XYZ leaks targets",
                     "mixed type6 image conditioning/type7 volume objective admission and expanded context budget are not amended or parity tested"])
    return {"schema": "chimera_recovery.hy3d_preflight.v1", "production_ready": False, "training_started": False,
            "model_weights_loaded": False, "verified": verified, "blockers": blockers,
            "implemented": ["actual guided trajectory/time/Euler/mesh validation", "exact pinned ImageProcessorV2 image/mask reconstruction",
                            "explicit experimental native feature/target/image allocation"],
            "experimental_proposal": {"position_policy": "zero_spatial_tags_for_unordered_latents", "selected": False,
                                      "parent_learned_query_slots": False,
                                      "requires": "native bridge/trunk permutation analysis, finite guided-target parity, image-conditioned decode proof and dated amendment"},
            "adapter_sha256": R.sha(Path(__file__))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "audit-source"), default="preflight")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--source-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise Error("HY3D audit/preflight needs a new immutable output report")
    if args.mode == "audit-source":
        if args.index is None or args.source_sha256 is None or args.source_audit is None:
            parser.error("audit-source requires --index, --source-sha256 and --source-audit captured sources")
        result = audit_source_shard(args.cache, args.items, args.index, args.source_sha256, args.source_audit)
    else:
        if args.manifest_sha256 is None:
            parser.error("preflight requires --manifest-sha256")
        result = preflight(args.cache, args.items, args.manifest_sha256, args.admission, source_audit_path=args.source_audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(R.canonical(result) + b"\n")
    print(json.dumps(result))
    return 0 if args.mode == "audit-source" else 2


if __name__ == "__main__":
    raise SystemExit(main())
