"""Authentic QVL pixels/feature alignment for D3d, with answer-KD admission.

The sealed Stage-C ViT cache omits pixels and language answers. Pixels can be
reconstructed from its pinned original image; language targets require a new
independent native-vocabulary answer cache and cannot be invented from features.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_vision_cache_common", HERE / "train_media.py")
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
R, Error = T.R, T.Error
SOURCE_PINS = {"media_b_teacher_cache.py": "d62d646c852fe24a26223c5097cfeef8f9cb9c3307b342f09fb7b1fedf48fc48",
               "s7_pairs_parent.py": "4339b1952d6c84c112e311afc25a36a505f66888d833aa0fcd0924d5986d6ca6"}


def smart_resize(height, width, *, factor=32, min_pixels=64 * 32 * 32, max_pixels=512 * 512):
    """Exact captured media_b_teacher_cache smart_resize arithmetic."""
    if height <= 0 or width <= 0 or min_pixels > max_pixels:
        raise Error("invalid native vision image/resize allocation")
    h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return int(h), int(w)


def patchify(pixels):
    """Exact s7 QVL duplicated-frame, merge-window C*T*P*P packing."""
    import torch
    if pixels.ndim != 3 or pixels.shape[0] != 3 or pixels.shape[1] % 32 or pixels.shape[2] % 32:
        raise Error("native QVL pixels require RGB and factor32 geometry")
    c, h, w = pixels.shape
    gh, gw = h // 16, w // 16
    frames = pixels.unsqueeze(0).expand(2, c, h, w)
    flat = frames.reshape(1, 2, c, gh // 2, 2, 16, gw // 2, 2, 16).permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 1536)
    bh, bw, mh, mw = torch.meshgrid(torch.arange(gh // 2), torch.arange(gw // 2), torch.arange(2), torch.arange(2), indexing="ij")
    yx = torch.stack(((bh * 2 + mh).reshape(-1), (bw * 2 + mw).reshape(-1)), -1)
    return flat.contiguous(), torch.tensor([[1, gh, gw]], dtype=torch.int64), yx


def reconstruct_image(image_path, expected_sha, sidecar, teacher, config):
    import torch
    from PIL import Image
    if R.sha(image_path) != R.require_sha(expected_sha, "vision_asset_sha256"):
        raise Error("original vision image bytes changed")
    knobs = config["knobs"]
    if config.get("teacher") != "qvl32_vit_l13" or knobs.get("block") != 13 or knobs.get("resize_factor") != 32 or knobs.get("dtype") != "fp32":
        raise Error("QVL native recipe requires actual fp32 block13/factor32 source configuration")
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        h, w = smart_resize(rgb.height, rgb.width, max_pixels=knobs["max_pixels"])
        image = rgb.resize((w, h), Image.Resampling.BICUBIC)
        pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).view(h, w, 3).permute(2, 0, 1) / 255.
    flat, grid, yx = patchify((pixels - .5) / .5)
    positions = torch.zeros(flat.shape[0], 4, dtype=torch.float32)
    positions[:, 1] = yx[:, 0].float() / max(1, int(grid[0, 1]) - 1)
    positions[:, 2] = yx[:, 1].float() / max(1, int(grid[0, 2]) - 1)
    if (sidecar.get("resized_hw") != [h, w] or sidecar.get("grid_thw") != grid[0].tolist() or
            any(not torch.equal(actual, teacher[key].cpu()) for key, actual in (("grid_thw", grid), ("yx", yx), ("positions", positions))) or
            teacher["block_out"].shape != (flat.shape[0], 1152) or teacher["block_in"].shape != teacher["block_out"].shape or
            teacher["cu_seqlens"].tolist() != [0, flat.shape[0]] or any(not bool(torch.isfinite(value).all()) for value in teacher.values())):
        raise Error("reconstructed QVL image/patch order differs from sealed teacher grid/positions/block rows")
    if "patch_flat" in teacher and not torch.equal(flat, teacher["patch_flat"].cpu()):
        raise Error("reconstructed pixels differ from actual stored teacher patches")
    return flat, positions


def validate_vision_batch(batch, receipt, root):
    """Verify original pixels, features and independent answer targets pre-step."""
    import torch
    from safetensors.torch import load_file
    source = T.pinned_file(receipt["source_feature_shard"], root)
    sidecar = T.read_json(source.with_suffix(".json"))
    manifest = T.read_json(T.pinned_file(receipt["source_feature_manifest"], root))
    if (manifest.get("schema") != "omni-campaign.teacher_cache.v2" or manifest.get("teacher") != "qvl32_vit_l13" or
            manifest.get("finite") is not True or manifest.get("teacher_integrity", {}).get("equal") is not True or
            sidecar.get("file_sha256") != R.sha(source) or sidecar.get("finite") is not True or sidecar.get("item", {}).get("held") is not False or
            not any(entry.get("name") == source.name and entry.get("sha256") == R.sha(source) for entry in manifest.get("shards", []))):
        raise Error("D3d source features are not this sealed immutable held-excluded QVL cache")
    teacher = load_file(str(source), device="cpu")
    image = T.pinned_file(receipt["source_image"], root)
    if receipt["source_image"]["sha256"] != sidecar["item_sha256"]:
        raise Error("D3d original image differs from teacher item identity")
    flat, positions = reconstruct_image(image, sidecar["item_sha256"], sidecar, teacher, manifest["run_config"])
    features, types = batch["native_media_features"], batch["native_media_type_ids"]
    if (features.shape != (1, flat.shape[0], 6144) or types.shape != (1, flat.shape[0]) or not bool((types == 6).all()) or
            not torch.equal(features[0, :, 640:2176].cpu(), flat) or bool(features[..., :640].any()) or bool(features[..., 2176:].any()) or
            not torch.equal(batch["native_media_positions"][0].cpu(), positions) or
            not torch.equal(batch["teacher_features"].cpu(), teacher["block_out"][None, :])):
        raise Error("D3d native C11 pixels/positions or detached feature targets differ from actual teacher")
    answer_manifest_path = T.pinned_file(receipt["source_answer_manifest"], root)
    answer_manifest = T.read_json(answer_manifest_path)
    answer_path = T.pinned_file(receipt["source_answer_shard"], root)
    if (answer_manifest.get("schema") != "chimera_recovery.vision_answer_source.v1" or answer_manifest.get("sealed") is not True or
            answer_manifest.get("all_finite") is not True or answer_manifest.get("assistant_only") is not True or
            answer_manifest.get("native_text_vocab_rows") != 248320 or answer_manifest.get("teacher_weights_unchanged") is not True or
            not any(entry.get("sha256") == R.sha(answer_path) and entry.get("asset_sha256") == sidecar["item_sha256"] for entry in answer_manifest.get("shards", []))):
        raise Error("D3d missing independently sealed authentic native-vocabulary assistant-answer teacher cache")
    answer = load_file(str(answer_path), device="cpu")
    required = {"input_ids", "assistant_token_mask", "next_ids", "ce_w", "eos_w", "q_w", "q_ids", "q_lp"}
    if set(answer) != required or any(not bool(torch.isfinite(value).all()) for value in answer.values()):
        raise Error("D3d authentic answer source tensor contract differs")
    ids, mask = answer["input_ids"], answer["assistant_token_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != (ids.shape[1],) or mask.dtype != torch.bool or not torch.equal(ids, batch["input_ids"].cpu()):
        raise Error("D3d assistant history/template token alignment differs")
    shifted = torch.zeros_like(mask)
    shifted[:-1] = mask[1:]
    if (not torch.equal(answer["q_w"] > 0, shifted) or bool((answer["q_w"][shifted] != 1).any()) or
            int(shifted.sum()) != receipt["answer_positions"] or bool(answer["ce_w"].any()) or bool(answer["eos_w"].any()) or
            not torch.equal(answer["next_ids"][:-1][shifted[:-1]], ids[0, 1:][shifted[:-1]])):
        raise Error("D3d answer targets must supervise actual assistant next tokens only")
    for key in required - {"input_ids", "assistant_token_mask"}:
        if not torch.equal(answer[key], batch["answer." + key].cpu()):
            raise Error("D3d prepared answer targets differ from original sealed teacher")
    return {"image_reconstructed": True, "feature_rows": flat.shape[0], "answer_positions": int(shifted.sum()), "source_pins": SOURCE_PINS}
