"""Authentic ACE code conditioning and compact native guided-velocity preparation.

The condition command loads only the exact teacher quantizer/detokenizer on CPU.
The prepare command loads no model weights. Teacher noise was sampled on CUDA;
its sealed x_t is consumed verbatim, never replaced with a CPU RNG approximation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_ace_media", HERE / "media_cache.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)
R, Error = M.R, M.Error
SOURCE_PINS = {"comfy/ldm/ace/ace_step15.py": "2230ec1f84aa9f4a38cca534ec79e5ad96baecb687e3d4c67b1b83d47cf3f87b",
               "comfy/model_base.py": "b7e425e0325c632249de429ebfc96d540aa0d87d313fedf89ebd00ed78238722"}
TEACHER_SHA = "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7"
TEACHER_TOOL_SHA = "d62d646c852fe24a26223c5097cfeef8f9cb9c3307b342f09fb7b1fedf48fc48"
NOISE_BANDS = {"4": [256, 384]}
BANDS = {4: {"feature": (0, 384), "emission": (0, 128)}}


def checked_ace_source(cache: Path, items: Path, admission: Path, manifest_sha: str):
    path = cache / "manifest.json"
    if R.sha(path) != R.require_sha(manifest_sha, "ACE manifest SHA"):
        raise Error("ACE sealed manifest moved")
    manifest = M.read_json(path)
    integrity = manifest.get("teacher_integrity", {})
    if (manifest.get("schema") != "omni-campaign.teacher_cache.v2" or manifest.get("teacher") != "ace_dit_velocity" or
            manifest.get("finite") is not True or not manifest.get("sealed_at_utc") or integrity.get("equal") is not True):
        raise Error("ACE requires a sealed finite immutable native velocity cache")
    if manifest.get("tool", {}).get("sha256") != TEACHER_TOOL_SHA:
        raise Error("ACE target writer differs from audited CUDA noise and conditioning recipe")
    after = integrity.get("after", {})
    identity = lambda entry: {file["path"]: [file["sha256"], file["bytes"]] for file in entry["files"]}
    if not integrity.get("before") or any(identity(entry) != identity(after) for entry in integrity["before"]):
        raise Error("ACE teacher before/after weight identity differs")
    if not any(file.get("sha256") == TEACHER_SHA for file in after.get("files", [])):
        raise Error("ACE teacher weights differ from audited conditioning components")
    items_sha = R.sha(items)
    corpus = M.read_json(items)
    if items_sha != manifest["corpus"]["file_sha256"] or corpus["corpus_sha256"] != manifest["corpus"]["sha256"]:
        raise Error("ACE source items differ from sealed teacher corpus")
    admitted, entries = M.media_admission(admission, items_sha, kind="ace")
    by_id = {item["id"]: item for item in corpus["items"]}
    if len(by_id) != len(corpus["items"]):
        raise Error("ACE source items repeat an ID")
    knobs = manifest["run_config"]["knobs"]
    if (knobs.get("dtype") != "fp32" or knobs.get("cfg") != 1. or knobs.get("uncond") != "zeroed text embedding, same lyrics + codes" or
            knobs.get("steps") != 8 or knobs.get("shift") != 3. or knobs.get("sigmas_per_clip") != 4):
        raise Error("ACE converter supports only the audited fp32 CFG1 code-conditioned teacher run")
    return manifest, by_id, admitted, entries


def source_shards(cache, manifest, items, admission):
    seen = set()
    for shard in manifest["shards"]:
        name = shard["name"]
        if Path(name).name != name or Path(name).suffix != ".safetensors" or shard.get("finite") is not True:
            raise Error("invalid finite ACE shard path")
        path, sidecar_path = cache / name, (cache / name).with_suffix(".json")
        if R.sha(path) != shard["sha256"]:
            raise Error("ACE teacher shard digest moved")
        sidecar = M.read_json(sidecar_path)
        if (sidecar.get("schema") != "omni-campaign.teacher_cache_shard.v2" or sidecar.get("file_sha256") != shard["sha256"] or
                sidecar.get("finite") is not True or sidecar.get("teacher") != "ace_dit_velocity" or
                sidecar.get("config_sha256") != manifest["config_sha256"] or sidecar.get("corpus_sha256") != manifest["corpus"]["sha256"]):
            raise Error("ACE sidecar identity differs from sealed manifest")
        ident = sidecar["item_id"]
        if ident in seen or ident != shard["item_id"] or ident not in items or ident not in admission:
            raise Error("ACE duplicate or unadmitted source item")
        seen.add(ident)
        item, entry = items[ident], admission[ident]
        prompt_sha = hashlib.sha256(item["caption"].encode()).hexdigest()
        condition_sha = R.digest({"caption": item["caption"], "lyrics": item.get("lyrics") or ""})
        if (entry["asset_sha256"] != item["sha256"] or entry["prompt_sha256"] != prompt_sha or entry["conditioning_sha256"] != condition_sha or
                any(sidecar["item"].get(key) != item.get(key) for key in ("id", "sha256", "caption", "lyrics", "seed", "held"))):
            raise Error("ACE admission/source/teacher conditioning identity differs")
        if (sidecar["item"].get("held") or shard.get("meta", {}).get("held")) and entry["train_admitted"]:
            raise Error("ACE held teacher item was incorrectly admitted")
        yield shard, sidecar, item, entry, path, sidecar_path


def validate_ace_tensor_recipe(tensors, sidecar, *, index, item):
    """Check audited scheduler/seeds and consume the exact stored CUDA x_t."""
    import torch
    frames = int(sidecar["latent_frames"])
    table = [3. * ((k + 1) / 1000.) / (1. + 2. * ((k + 1) / 1000.)) for k in range(1000)]
    schedule = [table[-(1 + int(point * 125))] for point in range(8)]
    sigmas = [schedule[(index + 2 * row) % 8] for row in range(4)]
    if (sidecar.get("cfg") != 1. or len(sidecar.get("sigmas", [])) != 4 or
            any(not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12) for observed, expected in zip(sidecar["sigmas"], sigmas)) or
            tensors["sigma"].tolist() != torch.tensor(sigmas, dtype=torch.float32).tolist()):
        raise Error("ACE tensor/sidecar sigmas differ from audited shifted scheduler")
    seeds = [int(hashlib.sha256(f"ace-noise-{row}:{item['sha256']}".encode()).hexdigest()[:8], 16) for row in range(4)]
    if tensors["noise_seed"].tolist() != seeds:
        raise Error("ACE cached noise seeds differ from audited item-hash recipe")
    if (frames < 1 or tensors["x0"].shape != (64, frames) or tensors["x_t"].shape != (4, 64, frames) or
            tensors["v_guided"].shape != (4, 64, frames) or tensors["codes"].ndim != 1 or
            tensors["codes"].numel() != sidecar["code_frames"] or tensors["codes"].dtype != torch.int32 or
            bool(((tensors["codes"] < 0) | (tensors["codes"] >= 64000)).any()) or
            any(not bool(torch.isfinite(value).all()) for value in tensors.values())):
        raise Error("ACE sealed native tensor shape/finite contract failed")
    return frames


def load_cpu_conditioner(comfy_root: Path, weights: Path):
    """Use authentic classes and exactly32 audited subset tensors, not full DiT."""
    import torch
    from safetensors import safe_open
    if torch.cuda.is_initialized():
        raise Error("ACE component preparation requires a fresh CPU-only process")
    for relative, expected in SOURCE_PINS.items():
        if R.sha(comfy_root / relative) != expected:
            raise Error("authentic ACE conditioning source pin moved")
    if R.sha(weights) != TEACHER_SHA:
        raise Error("ACE component weight file is not the sealed teacher")
    sys.path.insert(0, str(comfy_root))
    # Same CPU-only optional-backend policy as audited media_b_teacher_cache
    # comfy_modules(): kitchen expects ImportError for unavailable Triton,
    # whereas installed GPU Triton may raise RuntimeError with no active driver.
    # FP32 quantizer/detokenizer use native Torch CPU ops, not those GPU kernels.
    if "triton" not in sys.modules:
        sys.modules["triton"] = None
    argv = sys.argv
    try:
        sys.argv = [argv[0], "--cpu"]
        cli = importlib.import_module("comfy.cli_args")
        cli.args.cpu = True
        operations = importlib.import_module("comfy.ops").manual_cast
        source = importlib.import_module("comfy.ldm.ace.ace_step15")
    finally:
        sys.argv = argv
    if Path(source.__file__).resolve() != (comfy_root / "comfy/ldm/ace/ace_step15.py").resolve() or torch.cuda.is_initialized():
        raise Error("ACE source imported from another checkout or initialized CUDA")
    quantizer = source.ResidualFSQ(levels=[8, 8, 8, 5, 5, 5], num_quantizers=1, dim=2048,
                                   dtype=torch.float32, device="cpu", operations=operations)
    decoder = source.AudioTokenDetokenizer(hidden_size=2048, pool_window_size=5, audio_acoustic_hidden_dim=64,
                                           num_layers=2, head_dim=128, dtype=torch.float32, device="cpu", operations=operations)
    loaded = {}
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = {key for key in handle.keys() if key.startswith(("tokenizer.quantizer.", "detokenizer."))}
        if len(keys) != 32:
            raise Error("ACE subset header differs from audited32 tensor components")
        for prefix, module in (("tokenizer.quantizer.", quantizer), ("detokenizer.", decoder)):
            state = {key[len(prefix):]: handle.get_tensor(key).float() for key in sorted(keys) if key.startswith(prefix)}
            module.load_state_dict(state, strict=True)
            module.eval().requires_grad_(False)
            loaded.update({prefix + key: M.tensor_digest(value) for key, value in state.items()})
    component_sha = R.digest(loaded)
    def decode(codes, frames):
        if codes.ndim != 1 or codes.dtype not in (torch.int32, torch.int64) or bool(((codes < 0) | (codes >= 64000)).any()) or frames < 1:
            raise Error("invalid authentic ACE audio codes")
        code_input = codes.to(torch.int64)[None, :]
        needed = math.ceil(frames / 5)
        if code_input.shape[1] < needed:
            code_input = torch.nn.functional.pad(code_input, (0, needed - code_input.shape[1]), value=35847)
        with torch.inference_mode():
            result = decoder(quantizer.get_output_from_indices(code_input, dtype=torch.float32))[:, :frames, :]
        if result.shape != (1, frames, 64) or not bool(torch.isfinite(result).all()) or torch.cuda.is_initialized():
            raise Error("ACE code conditioning output shape/finite/CPU contract failed")
        return result.transpose(1, 2).contiguous()[0]
    return decode, {"source_pins": SOURCE_PINS, "teacher_sha256": TEACHER_SHA, "component_tensors": loaded,
                    "component_sha256": component_sha, "dtype": "float32", "device": "cpu", "padding_code": 35847}


def condition_ace(cache, items_path, admission_path, manifest_sha, comfy_root, weights, out, *, limit=None):
    import torch
    from safetensors.torch import load_file, save_file
    manifest, items, admission, entries = checked_ace_source(cache, items_path, admission_path, manifest_sha)
    if out.exists() or (limit is not None and (type(limit) is not int or limit < 1)):
        raise Error("use a new conditioning directory and positive optional proof limit")
    decode, component = load_cpu_conditioner(comfy_root, weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ace-conditioning-", dir=out.parent))
    accepted, complete, seen = [], True, 0
    for shard, sidecar, item, entry, path, sidecar_path in source_shards(cache, manifest, items, entries):
        seen += 1
        if not entry["train_admitted"]:
            continue
        tensors = load_file(str(path), device="cpu")
        frames = validate_ace_tensor_recipe(tensors, sidecar, index=shard["index"], item=item)
        codes, hints = tensors["codes"], decode(tensors["codes"], frames)
        name = f"condition-{shard['index']:05d}.safetensors"
        save_file({"hints": hints}, str(stage / name))
        accepted.append({"item_id": item["id"], "path": name, "sha256": R.sha(stage / name), "frames": frames,
                         "hints_sha256": M.tensor_digest(hints), "codes_sha256": M.tensor_digest(codes),
                         "source_shard_sha256": shard["sha256"], "source_sidecar_sha256": R.sha(sidecar_path),
                         "recovery_group_id": entry["recovery_group_id"]})
        if limit is not None and len(accepted) >= limit:
            complete = False
            break
    if complete and seen != len(manifest["shards"]):
        raise Error("ACE conditioning traversal incomplete")
    result = {"schema": "chimera_recovery.ace_conditioning.v1", "sealed": complete and bool(accepted), "all_finite": True,
              "source_manifest_sha256": manifest_sha, "corpus_sha256": R.sha(admission_path), "component": component,
              "items": accepted, "complete_source_traversal": complete, "adapter_sha256": R.sha(Path(__file__))}
    (stage / "manifest.json").write_bytes(R.canonical(result) + b"\n")
    stage.rename(out)
    return result


def ace_pack(value):
    import torch
    if value.ndim != 2 or value.shape[0] != 64:
        raise Error("ACE latent requires64 channels")
    padded = torch.nn.functional.pad(value, (0, value.shape[1] % 2))
    return padded.reshape(64, -1, 2).permute(1, 0, 2).reshape(1, -1, 128)


def native_ace_batch(noisy, velocity, hints, codes, *, sigma, prefix_ids, max_length):
    import torch
    frames = noisy.shape[1]
    if velocity.shape != noisy.shape or hints.shape != noisy.shape or not 0 < sigma <= 1:
        raise Error("ACE noisy/velocity/code-hint dimensions differ")
    if not prefix_ids or any(type(token) is not int or not 0 <= token < R.TEXT_VOCAB for token in prefix_ids):
        raise Error("ACE requires authentic native parent text prompt")
    if codes.ndim != 1 or bool(((codes < 0) | (codes >= 64000)).any()):
        raise Error("ACE code conditioning exceeds actual64000 written rows")
    rows, prefix = math.ceil(frames / 2), len(prefix_ids) + codes.numel()
    if prefix + rows > max_length:
        raise Error("ACE text+codes+latent sequence exceeds context; no truncation")
    features = torch.zeros(1, rows, R.FEATURES)
    target, channels = torch.zeros_like(features), torch.zeros_like(features, dtype=torch.bool)
    features[..., :128], features[..., 128:256], features[..., 256:384] = ace_pack(hints), ace_pack(torch.ones_like(noisy)), ace_pack(noisy)
    target[..., :128] = ace_pack(velocity)
    channels[..., :128] = ace_pack(torch.ones_like(noisy)).bool()
    positions = torch.zeros(1, rows, 4)
    positions[0, :, 0], positions[0, :, 2] = (torch.arange(rows).float() * 2 + 1) * .04, .08
    ids = prefix_ids + (codes.to(torch.int64) + 252416).tolist() + [M.MEDIA_PLACEHOLDER] * rows
    batch = {"input_ids": torch.tensor([ids], dtype=torch.int64), "native_media_features": features,
             "native_media_targets": target, "native_media_channel_loss_mask": channels,
             "native_media_type_ids": torch.full((1, rows), 4, dtype=torch.int64),
             "native_media_indices": torch.arange(prefix, prefix + rows, dtype=torch.int64)[None, :],
             "ace_code_indices": torch.arange(len(prefix_ids), prefix, dtype=torch.int64)[None, :],
             "native_media_valid_mask": torch.ones(1, rows, dtype=torch.bool), "native_media_positions": positions,
             "native_media_loss_mask": channels[..., :128].sum(-1).float() / 128.,
             "native_media_noise_levels": torch.full((1, rows), sigma)}
    R.validate_native_batch("music_generate", batch)
    return batch


def prepare_ace(cache, items_path, admission_path, manifest_sha, conditioning_path, tokenizer, amendment_sha, out, *, max_length=4096, limit=None):
    import torch
    from safetensors.torch import load_file, save_file
    if torch.cuda.is_initialized():
        raise Error("ACE preparation requires CPU only")
    manifest, items, admission, entries = checked_ace_source(cache, items_path, admission_path, manifest_sha)
    conditioning = M.read_json(conditioning_path)
    if (conditioning.get("schema") != "chimera_recovery.ace_conditioning.v1" or conditioning.get("all_finite") is not True or
            conditioning.get("source_manifest_sha256") != manifest_sha or conditioning.get("corpus_sha256") != R.sha(admission_path) or
            conditioning.get("component", {}).get("source_pins") != SOURCE_PINS or conditioning["component"].get("teacher_sha256") != TEACHER_SHA):
        raise Error("ACE conditioning does not bind exact teacher/source/admitted corpus")
    if limit is None and conditioning.get("sealed") is not True:
        raise Error("full ACE preparation requires completely traversed sealed conditioning")
    conditions = {item["item_id"]: item for item in conditioning["items"]}
    if len(conditions) != len(conditioning["items"]):
        raise Error("ACE conditioning repeats an item")
    if out.exists() or (limit is not None and (type(limit) is not int or limit < 1)):
        raise Error("use a new ACE output directory and positive optional proof limit")
    R.require_sha(amendment_sha, "amendment SHA")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".native-ace-", dir=out.parent))
    accepted, excluded, complete, seen = [], [], True, 0
    for shard, sidecar, item, entry, source, sidecar_path in source_shards(cache, manifest, items, entries):
        seen += 1
        if not entry["train_admitted"]:
            excluded.append({"item_id": item["id"], "split": entry["split"], "blockers": entry["blockers"]})
            continue
        condition = conditions.get(item["id"])
        if condition is None or condition["source_shard_sha256"] != shard["sha256"] or condition["source_sidecar_sha256"] != R.sha(sidecar_path):
            raise Error("ACE admitted shard lacks exact authentic code conditioning")
        hint_path = conditioning_path.parent / condition["path"]
        if hint_path.parent.resolve() != conditioning_path.parent.resolve() or R.sha(hint_path) != condition["sha256"]:
            raise Error("ACE code hints path/digest moved")
        tensors, hints = load_file(str(source), device="cpu"), load_file(str(hint_path), device="cpu")["hints"]
        frames = validate_ace_tensor_recipe(tensors, sidecar, index=shard["index"], item=item)
        if (M.tensor_digest(hints) != condition["hints_sha256"] or M.tensor_digest(tensors["codes"]) != condition["codes_sha256"] or
                hints.shape != (64, frames) or condition["frames"] != frames or not all(bool(torch.isfinite(v).all()) for v in tensors.values()) or
                tensors["x_t"].shape != tensors["v_guided"].shape or tensors["x_t"].shape[1:] != (64, frames) or
                tensors["sigma"].tolist() != [float(torch.tensor(value)) for value in sidecar["sigmas"]]):
            raise Error("ACE native sealed tensor/conditioning shape or identity differs")
        knobs = manifest["run_config"]["knobs"]
        prompt = "Music caption:\n" + item["caption"] + "\nLyrics:\n" + (item.get("lyrics") or "") + "\nTeacher settings:\n" + R.canonical(knobs["node_defaults"]).decode()
        prefix_ids = M.prompt_prefix(tokenizer, prompt)
        for row, sigma in enumerate(tensors["sigma"].tolist()):
            batch = native_ace_batch(tensors["x_t"][row], tensors["v_guided"][row], hints, tensors["codes"],
                                     sigma=sigma, prefix_ids=prefix_ids, max_length=max_length)
            name = f"ace-{shard['index']:05d}-{row:02d}"
            stored, storage = M.compact_native_batch(batch, BANDS)
            save_file(stored, str(stage / (name + ".safetensors")))
            condition_contract = {"ace_source_latents": "authentic_code_detokenizer", "ace_chunk_masks": "ones_with_right_padding_zero",
                                  "regime": "teacher_audio_code_conditioned_generation", "full_clean_latent_in_inputs": False,
                                  "conditioning_manifest_sha256": R.sha(conditioning_path), "codes_sha256": condition["codes_sha256"],
                                  "source_band_sha256": M.tensor_digest(batch["native_media_features"][..., :128][batch["native_media_type_ids"] == 4]),
                                  "chunk_band_sha256": M.tensor_digest(batch["native_media_features"][..., 128:256][batch["native_media_type_ids"] == 4]),
                                  "ace_step15_sha256": SOURCE_PINS["comfy/ldm/ace/ace_step15.py"], "model_base_sha256": SOURCE_PINS["comfy/model_base.py"],
                                  "inference_code_generation_requires_D3c": True, "semantic_codes_derived_from_teacher_clean_clip": True,
                                  "teacher_node_defaults": knobs["node_defaults"], "frames": frames, "frame_seconds": .04, "timestep_r": "sigma"}
            receipt = {"schema": "chimera_recovery.native_flow_cache.v1", "sealed": True, "all_finite": True, "all_structural_ok": True,
                       "guided_teacher": True, "no_clean_target_in_inputs": True, "route": "music_generate", "teacher_sha256": TEACHER_SHA,
                       "cache_sha256": R.sha(stage / (name + ".safetensors")), "corpus_sha256": R.sha(admission_path),
                       "recovery_amendment_sha256": amendment_sha, "prediction_kind": "raw", "target_kind": "canonical_velocity",
                       "input_layout": "feature", "target_layout": "emission", "parent_signs": {"4": -1.},
                       "noisy_feature_bands": NOISE_BANDS, "noisy_feature_bands_sha256": R.digest(NOISE_BANDS), "storage": storage,
                       "conditioning_contract": condition_contract, "source": {"manifest_sha256": manifest_sha, "shard_sha256": shard["sha256"],
                           "sidecar_sha256": R.sha(sidecar_path), "item_id": item["id"], "sigma_row": row, "noise_seed": int(tensors["noise_seed"][row]),
                           "noise_recipe": "sealed CUDA-generated x_t consumed verbatim; CPU RNG reproduction deliberately not asserted", "cfg": 1.,
                           "recovery_group_id": entry["recovery_group_id"]}}
            R.ReadyObjectiveAdapter(receipt).bind("music_generate", batch, BANDS, parent_signs={4: -1.})
            (stage / (name + ".json")).write_bytes(R.canonical(receipt) + b"\n")
            accepted.append({"id": name, "unit": "ace_dit_L12", "route": "music_generate", "recovery_group_id": entry["recovery_group_id"],
                             "input_layout": "feature", "target_layout": "emission", "parent_signs": {"4": -1.}, "storage": storage,
                             "tensors": {"path": name + ".safetensors", "sha256": receipt["cache_sha256"]},
                             "teacher_receipt": {"path": name + ".json", "sha256": R.sha(stage / (name + ".json"))}})
            if limit is not None and len(accepted) >= limit:
                complete = False
                break
        if not complete:
            break
    if complete and seen != len(manifest["shards"]):
        raise Error("ACE source traversal incomplete")
    result = {"schema": "chimera_recovery.native_batches.v1", "sealed": complete and bool(accepted), "split": "train",
              "heldout_groups_excluded": True, "complete_source_traversal": complete, "corpus_sha256": R.sha(admission_path),
              "source_cache_manifest_sha256": manifest_sha, "source_items_sha256": R.sha(items_path),
              "eval_index_sha256": admission["eval_index_sha256"], "batches": accepted, "excluded_heldout_items": excluded,
              "adapter_sha256": R.sha(Path(__file__)), "production_training_started": False,
              "adapter_kind": "authentic_code_conditioned_ACE_velocity", "inference_prompt_to_codes_requires_D3c": True}
    (stage / "manifest.json").write_bytes(R.canonical(result) + b"\n")
    stage.rename(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["condition", "prepare"])
    for flag in ("cache", "items", "admission", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--conditioning", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--amendment-sha256")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "condition":
        if not args.comfy_root or not args.weights:
            parser.error("condition requires --comfy-root and --weights")
        result = condition_ace(args.cache, args.items, args.admission, args.manifest_sha256, args.comfy_root, args.weights, args.out, limit=args.limit)
    else:
        if not args.conditioning or not args.tokenizer or not args.amendment_sha256:
            parser.error("prepare requires --conditioning --tokenizer --amendment-sha256")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
        result = prepare_ace(args.cache, args.items, args.admission, args.manifest_sha256, args.conditioning, tokenizer,
                             args.amendment_sha256, args.out, max_length=args.max_length, limit=args.limit)
    print(json.dumps({"sealed": result["sealed"], "items": len(result.get("items", result.get("batches", []))), "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
