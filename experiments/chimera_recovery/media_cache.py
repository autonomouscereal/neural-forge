"""Convert sealed Stage-C QIMG targets to actual native D3b input/target batches.

This performs CPU data preparation only. It requires connected source-group
admission, reconstructs the teacher's exact CPU noise, and verifies guidance and
denoising algebra before emitting any supervised target. No weights are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_cache_training", HERE / "training.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
Error = R.ReadinessError
QIMG_NOISE_BANDS = {"1": [512, 576]}
MEDIA_PLACEHOLDER = 248_832  # text248320 + fixed control512; checked against the loaded base by driver.


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_digest(tensor: Any) -> str:
    """Exact Stage-C tensor digest: dtype + json(shape) + contiguous CPU bytes."""
    value = tensor.detach().contiguous().cpu()
    out = hashlib.sha256()
    out.update(str(value.dtype).replace("torch.", "").encode())
    out.update(json.dumps(list(value.shape)).encode())
    out.update(value.numpy().tobytes())
    return out.hexdigest()


def qimg_pack(value: Any) -> Any:
    """Audited Comfy process_img order (t,h/2,w/2,c,ph,pw)."""
    b, c, t, h, w = value.shape
    if c != 16 or h % 2 or w % 2:
        raise Error("QIMG requires16 channels and even spatial latent dimensions")
    return value.reshape(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, -1, 64)


def reconstruct_qimg_target(tensors: dict[str, Any], sidecar: dict[str, Any], row: int, cfg: float) -> tuple[Any, Any, dict[str, Any]]:
    """Reproduce exact seeded x_t, with recorded noise and velocity parity checks."""
    import torch
    entry = sidecar["targets"][row]
    matching = (tensors["item.index"] == int(entry["item_index"])).nonzero().flatten()
    if matching.numel() != 1:
        raise Error("QIMG target has no unique source latent item")
    source_row = int(matching[0])
    x0 = tensors["item.x0"][source_row:source_row + 1].float()
    if x0.ndim != 5 or x0.shape[1] != 16 or not bool(torch.isfinite(x0).all()):
        raise Error("invalid cached QIMG model-space x0")
    seed, sigma = int(tensors["seed"][row]), float(entry["sigma"])
    if seed != entry["seed"] or not 0 < sigma <= 1 or not math.isfinite(sigma):
        raise Error("QIMG target seed or sigma differs from sidecar")
    if not math.isclose(float(tensors["sigma"][row]), sigma, rel_tol=1e-6, abs_tol=1e-7):
        raise Error("QIMG tensor sigma differs from full-precision recipe")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(x0.shape), generator=generator, dtype=torch.float32)
    noise_sha = hashlib.sha256(tensor_digest(noise).encode()).hexdigest()
    if noise_sha != R.require_sha(entry.get("noise_sha256"), "noise_sha256"):
        raise Error("exact CPU teacher noise digest disagrees; no approximate reconstruction allowed")
    noisy = (1. - sigma) * x0 + sigma * noise
    velocity = tensors["v"][row:row + 1].float()
    denoised = tensors["x0hat"][row:row + 1].float()
    if velocity.shape != x0.shape or denoised.shape != x0.shape or not bool(torch.isfinite(velocity).all()) or not bool(torch.isfinite(denoised).all()):
        raise Error("invalid QIMG guided velocity or denoised target")
    reconstructed = noisy.double() - sigma * velocity.double()
    error = float((reconstructed - denoised.double()).abs().max())
    if error > 2e-5 * max(1., float(denoised.abs().max())):
        raise Error("QIMG noisy input/velocity/x0hat algebra disagrees")
    guidance_error = 0.
    if cfg != 1.:
        cond, uncond = tensors["v_cond"][row:row + 1].float(), tensors["v_uncond"][row:row + 1].float()
        expected = uncond + cfg * (cond - uncond)
        guidance_error = float((expected - velocity).abs().max())
        # Stage C combines denoised fp32, then inverts in fp64; the velocity
        # combination consequently differs by fp32 rounding, amplified at sigma.
        tolerance = 2e-5 / sigma * max(1., float(velocity.abs().max()))
        if not math.isfinite(guidance_error) or guidance_error > tolerance:
            raise Error("QIMG guided velocity differs from recorded CFG branches")
    return noisy, velocity, {"seed": seed, "sigma": sigma, "noise_sha256": noise_sha,
                              "denoising_max_abs": error, "guidance_max_abs": guidance_error,
                              "packing": "Comfy process_img (t,h/2,w/2,c,ph,pw)", "clean_target_in_inputs": False}


def native_qimg_batch(noisy: Any, velocity: Any, *, sigma: float, prefix_ids: list[int], max_length: int) -> dict[str, Any]:
    import torch
    if not prefix_ids or any(type(token) is not int or not 0 <= token < R.TEXT_VOCAB for token in prefix_ids):
        raise Error("QIMG conditioning requires a real parent-template text prefix")
    inputs, targets = qimg_pack(noisy), qimg_pack(velocity)
    if inputs.shape != targets.shape or inputs.shape[0] != 1:
        raise Error("one QIMG sample per native HybridD3 call")
    rows, prefix = inputs.shape[1], len(prefix_ids)
    if prefix + rows > max_length:
        raise Error("QIMG prompt+latent grid exceeds native context; no row truncation")
    features = torch.zeros(1, rows, R.FEATURES)
    native_targets = torch.zeros_like(features)
    features[..., 512:576] = inputs
    native_targets[..., 256:320] = targets
    _, _, frames, height, width = noisy.shape
    gh, gw = height // 2, width // 2
    positions = torch.zeros(1, rows, 4)
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    positions[0, :, 1] = ys.flatten().repeat(frames).float() / max(1, gh - 1)
    positions[0, :, 2] = xs.flatten().repeat(frames).float() / max(1, gw - 1)
    positions[0, :, 0] = torch.arange(frames).repeat_interleave(gh * gw).float() / max(1, frames - 1)
    batch = {"input_ids": torch.tensor([prefix_ids + [MEDIA_PLACEHOLDER] * rows], dtype=torch.int64),
             "native_media_features": features, "native_media_targets": native_targets,
             "native_media_type_ids": torch.ones(1, rows, dtype=torch.int64),
             "native_media_indices": torch.arange(prefix, prefix + rows, dtype=torch.int64)[None, :],
             "native_media_valid_mask": torch.ones(1, rows, dtype=torch.bool), "native_media_positions": positions,
             "native_media_loss_mask": torch.ones(1, rows), "native_media_noise_levels": torch.full((1, rows), sigma)}
    R.validate_native_batch("image_generate", batch)
    return batch


def prompt_prefix(tokenizer: Any, prompt: str) -> list[int]:
    if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
        raise Error("ambiguous role delimiters in QIMG prompt")
    messages = [{"role": "user", "content": prompt}]
    options = {"tools": [], "add_generation_prompt": True, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    exact = R.native_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, **options))
    offset_ids = R.native_token_ids(tokenizer(rendered, add_special_tokens=False))
    if exact != offset_ids:
        raise Error("native QIMG prompt render/tokenization IDs disagree")
    return exact


def compact_native_batch(batch: dict[str, Any], bands: dict[int, dict[str, tuple[int, int]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lossless selected-row band storage; dense zero channels are never written."""
    import torch
    keys = {"native_media_features": "feature", "native_media_targets": "emission", "native_media_channel_loss_mask": "emission"}
    types = batch["native_media_type_ids"]
    stored = {key: value.contiguous() for key, value in batch.items() if key not in keys}
    allocations = {str(kind): {name: list(bounds) for name, bounds in band.items()} for kind, band in bands.items()}
    contract = {"schema": "chimera_recovery.native_band.v1", "width": R.FEATURES, "bands": allocations,
                "channel_loss_mask": "native_media_channel_loss_mask" in batch}
    for key, layout in keys.items():
        if key not in batch:
            continue
        value = batch[key]
        if value.shape != (*types.shape, R.FEATURES) or value.requires_grad:
            raise Error("compact native storage requires detached full-width bridge tensors")
        remainder = value.clone()
        for kind, band in bands.items():
            lo, hi = band[layout]
            selected = types == kind
            stored[f"{key}.band{kind}"] = value[..., lo:hi][selected].contiguous()
            remainder[..., lo:hi][selected] = 0
        if bool((remainder != 0).any()):
            raise Error("compact native storage would discard nonzero unallocated channels")
    return stored, contract


def inflate_native_batch(stored: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    import torch
    if contract.get("schema") != "chimera_recovery.native_band.v1" or contract.get("width") != R.FEATURES:
        raise Error("unsupported compact native storage contract")
    types = stored["native_media_type_ids"]
    bands = contract.get("bands")
    if not isinstance(bands, dict) or set(bands) != {str(kind) for kind in types.unique().tolist()}:
        raise Error("compact allocation differs from packed native type dispatch")
    keys = {"native_media_features": "feature", "native_media_targets": "emission"}
    if contract.get("channel_loss_mask") is True:
        keys["native_media_channel_loss_mask"] = "emission"
    elif contract.get("channel_loss_mask") is not False:
        raise Error("compact channel-mask presence must be explicit")
    consumed = set()
    result = {key: value for key, value in stored.items() if ".band" not in key}
    for key, layout in keys.items():
        result[key] = torch.zeros((*types.shape, R.FEATURES), dtype=torch.bool if key.endswith("channel_loss_mask") else torch.float32, device=types.device)
        for kind, band in bands.items():
            bounds = band.get(layout)
            if not isinstance(bounds, list) or len(bounds) != 2 or any(type(n) is not int for n in bounds) or not 0 <= bounds[0] < bounds[1] <= R.FEATURES:
                raise Error("invalid compact feature/emission allocation")
            lo, hi = bounds
            name = f"{key}.band{kind}"
            value, selected = stored[name], types == int(kind)
            if value.shape != (int(selected.sum()), hi - lo) or value.dtype != result[key].dtype:
                raise Error("compact native band shape/dtype differs from allocation")
            result[key][..., lo:hi][selected] = value
            consumed.add(name)
    if set(stored) - set(result) != consumed or any(key in stored for key in keys):
        raise Error("unrecognized or duplicate dense tensor in compact native storage")
    return result


def media_admission(path: Path, items_sha256: str, *, kind: str = "qimg") -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    admitted = read_json(path)
    if (admitted.get("schema") != "recovery_media_admission.v1" or
            admitted.get("sources", {}).get(kind + "_items", {}).get("sha256") != items_sha256 or admitted.get("eval_disjoint") is not True or
            admitted.get("connected_groups_verified") is not True):
        raise Error("QIMG requires verified connected source-group admission against frozen media eval")
    eval_sha = R.require_sha(admitted.get("sources", {}).get("eval_rows", {}).get("sha256"), "eval_index_sha256")
    output = admitted["outputs"]["admission"]
    filename = str(output["path"]).replace("\\", "/")
    rows_path = Path(filename)
    if not rows_path.is_file():
        rows_path = path.parent / Path(filename).name
    if not rows_path.is_file() or R.sha(rows_path) != R.require_sha(output.get("sha256"), "admission_rows_sha256"):
        raise Error("pinned connected admission rows missing or changed")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != output["rows"]:
        raise Error("admission row count differs from sealed output")
    by_id, groups = {}, {}
    for entry in rows:
        if entry.get("kind") != kind:
            continue
        if entry.get("schema") != "recovery_media_admission.v1":
            raise Error("media admission row schema differs")
        ident, group, split = entry["item_id"], entry["recovery_group_id"], entry["split"]
        if not isinstance(ident, str) or not ident or ident in by_id or not isinstance(group, str) or not group or split not in {"train", "validation", "test"}:
            raise Error("invalid media item/group/split admission")
        for key in ("asset_sha256", "prompt_sha256", "conditioning_sha256"):
            R.require_sha(entry.get(key), key)
        if type(entry.get("train_admitted")) is not bool or (entry["train_admitted"] and (split != "train" or entry.get("blockers"))):
            raise Error("media training admission contradicts split/blocker ruling")
        if group in groups and groups[group] != split:
            raise Error("media connected group crosses train/holdout splits")
        groups[group] = split
        by_id[ident] = entry
    admitted["eval_index_sha256"] = eval_sha
    return admitted, by_id


def prepare_qimg(cache: Path, items_path: Path, admission_path: Path, tokenizer: Any, amendment_sha256: str,
                 out: Path, *, expected_manifest_sha256: str, max_length: int = 4096, limit: int | None = None) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file
    if torch.cuda.is_initialized():
        raise Error("cache preparation refuses initialized CUDA")
    manifest_path = cache / "manifest.json"
    if R.sha(manifest_path) != R.require_sha(expected_manifest_sha256, "expected_manifest_sha256"):
        raise Error("QIMG sealed manifest digest moved")
    manifest = read_json(manifest_path)
    integrity = manifest.get("teacher_integrity", {})
    if manifest.get("schema") != "omni-campaign.stage_c.media_a.v1" or manifest.get("kind") != "qimg-velocity" or manifest.get("all_finite") is not True or integrity.get("unchanged") is not True:
        raise Error("QIMG source is not a finite sealed immutable velocity cache")
    identity = lambda record: {key: {"sha256": value["sha256"], "bytes": value["bytes"]} for key, value in record.items()}
    if identity(integrity.get("before", {})) != identity(integrity.get("after", {})):
        raise Error("QIMG teacher before/after digests disagree")
    teacher_sha = R.require_sha(integrity["after"][manifest["dit"]["path"]]["sha256"], "teacher_sha256")
    items_sha = R.sha(items_path)
    if items_sha != manifest["corpus"]["sha256"]:
        raise Error("QIMG source items are not the cache's exact corpus")
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    admission, admitted = media_admission(admission_path, items_sha)
    R.require_sha(amendment_sha256, "recovery_amendment_sha256")
    if out.exists() or (limit is not None and (type(limit) is not int or limit < 1)):
        raise Error("use a new output directory and positive optional proof limit")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".native-qimg-", dir=out.parent))
    accepted, excluded = [], []
    cfg = float(manifest["plan"]["cfg"])
    if not math.isfinite(cfg) or cfg <= 0:
        raise Error("QIMG teacher guidance must be finite and positive")
    traversal_complete = True
    seen_targets = set()
    for shard in manifest["shards"]:
        filename = Path(shard["file"])
        if filename.name != str(filename) or filename.suffix != ".safetensors":
            raise Error("QIMG shard path escapes cache")
        source = cache / filename
        if R.sha(source) != shard["sha256"] or shard.get("finite") is not True:
            raise Error("QIMG sealed shard digest or finite flag moved")
        sidecar_path = source.with_suffix(".json")
        sidecar = read_json(sidecar_path)
        if sidecar.get("sha256") != shard["sha256"] or sidecar.get("finite") is not True:
            raise Error("QIMG shard sidecar disagrees with sealed manifest")
        tensors = load_file(str(source), device="cpu")
        for row, target in enumerate(sidecar["targets"]):
            if target["target"] in seen_targets:
                raise Error("QIMG sealed cache repeats a target index")
            seen_targets.add(target["target"])
            item_index = int(target["item_index"])
            item = items[item_index]
            if item["id"] != target["id"] or int(tensors["target_item"][row]) != item_index:
                raise Error("QIMG target/source item linkage disagrees")
            entry = admitted.get(item["id"])
            if entry is None:
                raise Error("QIMG cached item has no reviewed media source-group admission")
            media_sha = item.get("sha256", item.get("image_sha256"))
            prompt_sha = hashlib.sha256(item["prompt"].encode()).hexdigest()
            negative = item.get("negative", manifest["plan"].get("negative_default"))
            if not isinstance(negative, str):
                raise Error("QIMG cache must pin explicit negative conditioning, including an empty string")
            conditioning_sha = R.digest({"prompt": item["prompt"], "negative": negative})
            if entry["asset_sha256"] != media_sha or entry["prompt_sha256"] != prompt_sha or entry["conditioning_sha256"] != conditioning_sha:
                raise Error("QIMG admission asset/prompt identity differs from source item")
            if not entry["train_admitted"]:
                excluded.append({"item_id": item["id"], "split": entry["split"], "recovery_group_id": entry["recovery_group_id"], "blockers": entry["blockers"]})
                continue
            noisy, velocity, proof = reconstruct_qimg_target(tensors, sidecar, row, cfg)
            student_prompt = item["prompt"] if not negative else "Image prompt:\n" + item["prompt"] + "\nNegative prompt:\n" + negative
            batch = native_qimg_batch(noisy, velocity, sigma=proof["sigma"], prefix_ids=prompt_prefix(tokenizer, student_prompt), max_length=max_length)
            ident = f"qimg-{int(target['target']):08d}"
            tensor_name, receipt_name = ident + ".safetensors", ident + ".json"
            stored, storage = compact_native_batch(batch, {1: {"feature": (512, 576), "emission": (256, 320)}})
            save_file(stored, str(stage / tensor_name))
            receipt = {"schema": "chimera_recovery.native_flow_cache.v1", "sealed": True, "all_finite": True,
                       "all_structural_ok": True, "guided_teacher": True, "no_clean_target_in_inputs": True,
                       "cache_sha256": R.sha(stage / tensor_name), "corpus_sha256": R.sha(admission_path), "teacher_sha256": teacher_sha,
                       "recovery_amendment_sha256": amendment_sha256, "target_kind": "canonical_velocity", "prediction_kind": "raw",
                       "route": "image_generate", "input_layout": "feature", "target_layout": "emission",
                       "parent_signs": {"1": -1.},
                       "noisy_feature_bands": QIMG_NOISE_BANDS, "noisy_feature_bands_sha256": R.digest(QIMG_NOISE_BANDS),
                       "conditioning_contract": {"kind": "native_parent_chat_prompt", "prompt_sha256": prompt_sha,
                                                 "template_sha256": R.digest(tokenizer.chat_template), "reference_image_in_inputs": False},
                       "source": {"manifest_sha256": expected_manifest_sha256, "shard_sha256": shard["sha256"],
                                  "sidecar_sha256": R.sha(sidecar_path), "target_index": int(target["target"]), "item_index": item_index,
                                  "item_id": item["id"], "media_sha256": media_sha, "conditioning_sha256": conditioning_sha,
                                  "cfg": cfg, "recovery_group_id": entry["recovery_group_id"]},
                       "reconstruction": proof, "media_placeholder": MEDIA_PLACEHOLDER, "storage": storage}
            R.ReadyObjectiveAdapter(receipt).bind("image_generate", batch, {1: {"feature": (512, 576), "emission": (256, 320)}}, parent_signs={1: -1.})
            (stage / receipt_name).write_bytes(R.canonical(receipt) + b"\n")
            accepted.append({"id": ident, "unit": "qimg_mmdit_L30_img", "route": "image_generate", "input_layout": "feature", "target_layout": "emission",
                             "recovery_group_id": entry["recovery_group_id"], "parent_signs": {"1": -1.},
                             "storage": storage, "tensors": {"path": tensor_name, "sha256": receipt["cache_sha256"]},
                             "teacher_receipt": {"path": receipt_name, "sha256": R.sha(stage / receipt_name)}})
            if limit is not None and len(accepted) >= limit:
                traversal_complete = False
                break
        if not traversal_complete:
            break
    if traversal_complete and len(seen_targets) != int(manifest["targets"]):
        raise Error("QIMG complete traversal target count differs from sealed parent cache")
    result = {"schema": "chimera_recovery.native_batches.v1", "sealed": traversal_complete and bool(accepted), "split": "train",
              "heldout_groups_excluded": True, "complete_source_traversal": traversal_complete,
              "corpus_sha256": R.sha(admission_path), "source_cache_manifest_sha256": expected_manifest_sha256,
              "source_items_sha256": items_sha, "eval_index_sha256": admission["eval_index_sha256"],
              "batches": accepted, "excluded_heldout_targets": excluded, "production_training_started": False,
              "adapter_sha256": R.sha(Path(__file__)), "adapter_kind": "exact_seeded_qimg_stage_c_velocity"}
    (stage / "manifest.json").write_bytes(R.canonical(result) + b"\n")
    stage.rename(out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="bounded CPU proof; output manifest deliberately remains unsealed")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    report = prepare_qimg(args.cache, args.items, args.admission, tokenizer, args.amendment_sha256, args.out,
                          expected_manifest_sha256=args.manifest_sha256, max_length=args.max_length, limit=args.limit)
    print(json.dumps({"accepted_batches": len(report["batches"]), "sealed": report["sealed"], "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
