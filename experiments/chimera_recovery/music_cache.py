"""Faithful CPU conversion of sealed Stage-C ACE-LM teacher records to D3c.

The source cache is conditional-branch top64 over audio codes, normalized over
the parent's full vocabulary. D3c restores its recorded code-band normalization;
it never reuses parent text token IDs or admits an unsealed partial cache.
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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_music_cache_common", HERE / "media_cache.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)
R, Error = M.R, M.Error
TEACHER = "ace_lm_codes"
RECORDED = "conditional branch logits (row 0) at every emitted position: top-k over the band, log-softmax over the full vocab"


def sealed_source(cache, items_path, expected_sha, *, teacher=TEACHER):
    """Verify actual media_b v2 seal/config/corpus identity before tensor load."""
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise Error(f"{teacher} full teacher cache is not sealed; partial shards cannot enter production")
    if R.sha(manifest_path) != R.require_sha(expected_sha, "source_manifest_sha256"):
        raise Error("sealed teacher manifest digest changed")
    manifest = M.read_json(manifest_path)
    integrity = manifest.get("teacher_integrity", {})
    after = integrity.get("after", {})
    before = integrity.get("before", [])
    if (manifest.get("schema") != "omni-campaign.teacher_cache.v2" or manifest.get("teacher") != teacher or
            manifest.get("finite") is not True or integrity.get("equal") is not True or integrity.get("mode") != "full" or
            not before or after.get("mode") != "full" or any(entry.get("digest") != after.get("digest") or
            entry.get("files") != after.get("files") for entry in before)):
        raise Error("source requires a finite full-integrity immutable media_b v2 seal")
    R.require_sha(after.get("digest"), "teacher_integrity.digest")
    for entry in after.get("files", []):
        R.require_sha(entry.get("sha256"), "teacher_weight_sha256")
    if not after.get("files"):
        raise Error("source seal lacks teacher weight identities")
    config = manifest.get("run_config", {})
    if (config.get("schema") != "omni-campaign.teacher_cache_config.v1" or config.get("teacher") != teacher or
            R.digest(config) != manifest.get("config_sha256") or M.read_json(cache / "cache_config.json") != config):
        raise Error("source run configuration is not its sealed canonical digest")
    corpus = M.read_json(items_path)
    kind = "ace_prompts" if teacher == TEACHER else "qvl_images"
    if (corpus.get("schema") != "omni-campaign.teacher_corpus.v1" or corpus.get("kind") != kind or
            R.sha(items_path) != manifest.get("corpus", {}).get("file_sha256") or
            corpus.get("corpus_sha256") != manifest["corpus"].get("sha256") or
            corpus.get("n_items") != len(corpus.get("items", [])) or manifest["corpus"].get("n_items") != corpus["n_items"] or
            len(manifest.get("shards", [])) != corpus["n_items"] or
            corpus.get("contamination_screen", {}).get("applied") is not True):
        raise Error("source sealed corpus/count/contamination identity differs")
    if manifest.get("shard_read_order") != [entry["name"] for entry in manifest["shards"]]:
        raise Error("source sealed shard order differs")
    return manifest, corpus


def source_rows(cache, manifest, corpus):
    """Validate every sealed item, sidecar, held bit and shard digest."""
    seen = set()
    for shard in manifest["shards"]:
        index = shard["index"]
        if type(index) is not int or index in seen or not 0 <= index < len(corpus["items"]):
            raise Error("sealed teacher item index duplicate/outside corpus")
        seen.add(index)
        item = corpus["items"][index]
        filename = Path(shard["name"])
        if filename.name != str(filename) or filename.suffix != ".safetensors":
            raise Error("teacher shard path escapes cache")
        path = cache / filename
        sidecar_path = path.with_suffix(".json")
        sidecar = M.read_json(sidecar_path)
        if (R.sha(path) != shard["sha256"] or path.stat().st_size != shard["bytes"] or shard.get("finite") is not True or
                sidecar.get("schema") != "omni-campaign.teacher_cache_shard.v2" or sidecar.get("teacher") != manifest["teacher"] or
                sidecar.get("file") != filename.name or sidecar.get("file_sha256") != shard["sha256"] or
                sidecar.get("item_id") != item["id"] or shard.get("item_id") != item["id"] or
                sidecar.get("index") != index or sidecar.get("item_sha256") != item["sha256"] or sidecar.get("finite") is not True or
                sidecar.get("corpus_sha256") != corpus["corpus_sha256"] or sidecar.get("config_sha256") != manifest["config_sha256"] or
                sidecar.get("tool_version") not in manifest["compat"] or type(item.get("held")) is not bool or
                sidecar.get("item", {}).get("held") != item["held"]):
            raise Error("sealed teacher shard/item/config/held identity differs")
        yield shard, sidecar, item, path, sidecar_path
    if len(seen) != len(corpus["items"]):
        raise Error("source traversal omitted teacher corpus rows")


def native_music_prompt(tokenizer, item, sidecar, config):
    """Native parent template, with exact semantic conditioning and duration."""
    knobs = config["knobs"]
    seconds = min(knobs["max_seconds"], max(knobs["min_seconds"], item["seconds"]))
    if not math.isfinite(seconds) or sidecar.get("seconds") != seconds or sidecar.get("sampling") != knobs["sampling"]:
        raise Error("ACE-LM duration/sampling differs from frozen source configuration")
    metadata = {"caption": item["caption"], "lyrics": item.get("lyrics") or "", "seconds": seconds,
                "musical_metadata": knobs["node_defaults"]}
    message = [{"role": "user", "content": "Generate audio codes for this music request:\n" + R.canonical(metadata).decode()}]
    options = {"add_generation_prompt": True, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(message, tokenize=False, **options)
    ids = R.native_token_ids(tokenizer.apply_chat_template(message, tokenize=True, **options))
    if ids != R.native_token_ids(tokenizer(rendered, add_special_tokens=False)) or not ids or any(not 0 <= token < R.TEXT_VOCAB for token in ids):
        raise Error("ACE-LM native parent chat IDs disagree or leave text vocabulary")
    return ids, {"template_sha256": R.digest(tokenizer.chat_template), "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                 "semantic_conditioning_sha256": R.digest(metadata), "enable_thinking": False,
                 "parent_prompt_ids_reused": False, "prompt_tokens": len(ids)}


def native_music_batch(tensors, sidecar, prefix_ids, *, max_length=4096):
    """Restore recorded band KL and teacher-force only previous sampled codes."""
    import torch
    required = {"tokens", "topk_ids", "topk_logprobs", "lse_full", "lse_band", "eos_logprob", "prompt_ids"}
    if set(tensors) != required or any(value.requires_grad or value.device.type != "cpu" or not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise Error("ACE-LM actual detached finite CPU tensor contract differs")
    n = tensors["tokens"].numel()
    if (n < 1 or tensors["tokens"].shape != (n,) or tensors["tokens"].dtype != torch.int32 or
            tensors["topk_ids"].shape != (n, 64) or tensors["topk_ids"].dtype != torch.int32 or
            tensors["topk_logprobs"].shape != (n, 64) or tensors["topk_logprobs"].dtype != torch.bfloat16 or
            any(tensors[key].shape != (n,) or tensors[key].dtype != torch.float32 for key in ("lse_full", "lse_band", "eos_logprob")) or
            tensors["prompt_ids"].ndim != 1 or tensors["prompt_ids"].dtype != torch.int32 or
            sidecar.get("recorded") != RECORDED or sidecar.get("band", {}).get("start") != 151669 or sidecar["band"].get("end") != 215669 or
            sidecar.get("lm_metadata", {}).get("min_tokens") != n):
        raise Error("ACE-LM source shapes/dtypes/conditional/code-band contract differs")
    if (not prefix_ids or any(type(token) is not int or not 0 <= token < R.TEXT_VOCAB for token in prefix_ids) or
            len(prefix_ids) + n - 1 > max_length or
            any(bool(((tensors[key] < 0) | (tensors[key] >= 64000)).any()) for key in ("tokens", "topk_ids")) or
            any(row.unique().numel() != 64 for row in tensors["topk_ids"])):
        raise Error("ACE-LM context/code IDs invalid; no truncation or inferred remapping")
    full, band = tensors["lse_full"], tensors["lse_band"]
    if bool((band > full + 1e-5).any()) or bool((tensors["topk_logprobs"] > 0).any()):
        raise Error("ACE-LM recorded full/band normalizers invalid")
    normalized = tensors["topk_logprobs"].float() + (full - band)[:, None]
    if bool((normalized > 1e-5).any()) or bool((normalized.exp().sum(-1) > 1.00001).any()):
        raise Error("ACE-LM BF16 recorded band mass exceeds one; quarantine rather than invent renormalization")
    ids = prefix_ids + (252416 + tensors["tokens"][:-1].to(torch.int64)).tolist()
    return {"input_ids": torch.tensor([ids], dtype=torch.int64), "topk_codes": tensors["topk_ids"].to(torch.int64),
            "topk_logprobs_band": normalized.contiguous()}, {"prompt_length": len(prefix_ids), "teacher_positions": n,
            "normalization": "recorded_full_vocab_logprobs + lse_full - lse_band", "teacher_branch": "conditional",
            "target_code_in_same_input_position": False, "written_code_rows": [252416, 316416], "allocated_music_rows": [252416, 317952],
            "bf16_mass_policy": "quarantine invalid rows; never rescale", "eos_targets_used": False}


def prepare_music(cache, items_path, admission_path, tokenizer, amendment_sha, out, *, manifest_sha, max_length=4096, limit=None):
    import torch
    from safetensors.torch import load_file, save_file
    if torch.cuda.is_initialized() or out.exists() or (limit is not None and (type(limit) is not int or limit < 1)):
        raise Error("CPU preparation needs a new output directory and positive proof limit")
    manifest, corpus = sealed_source(cache, items_path, manifest_sha)
    admission, entries = M.media_admission(admission_path, R.sha(items_path), kind="ace_lm")
    R.require_sha(amendment_sha, "amendment_sha256")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".native-music-", dir=out.parent))
    accepted, excluded, quarantine, complete = [], [], [], True
    for shard, sidecar, item, source, sidecar_path in source_rows(cache, manifest, corpus):
        entry = entries.get(item["id"])
        conditioning = R.digest({"caption": item["caption"], "lyrics": item.get("lyrics") or "", "seconds": item["seconds"]})
        if (entry is None or entry.get("teacher_item_sha256") != item["sha256"] or entry["asset_sha256"] != item.get("source_clip_sha256") or
                entry["conditioning_sha256"] != conditioning or entry["prompt_sha256"] != hashlib.sha256(item["caption"].encode()).hexdigest()):
            raise Error("ACE-LM reviewed source asset/teacher item/conditioning admission differs")
        if item["held"] or not entry["train_admitted"]:
            excluded.append({"item_id": item["id"], "split": entry["split"], "held": item["held"], "blockers": entry["blockers"]})
            continue
        try:
            prefix, prompt_proof = native_music_prompt(tokenizer, item, sidecar, manifest["run_config"])
            tensors, proof = native_music_batch(load_file(str(source), device="cpu"), sidecar, prefix, max_length=max_length)
        except (Error, ValueError, TypeError, KeyError) as error:
            quarantine.append({"item_id": item["id"], "source_shard_sha256": shard["sha256"], "reason": str(error)})
            continue
        ident = f"ace-lm-{shard['index']:05d}"
        name = ident + ".safetensors"
        save_file(tensors, str(stage / name))
        receipt = {"schema": "chimera_recovery.music_cache.v1", "sealed": True, "all_finite": True,
                   "source_manifest_sha256": manifest_sha, "source_shard_sha256": shard["sha256"], "source_sidecar_sha256": R.sha(sidecar_path),
                   "teacher_integrity_digest": manifest["teacher_integrity"]["after"]["digest"], "corpus_sha256": R.sha(admission_path),
                   "source_items_sha256": R.sha(items_path), "item_id": item["id"], "teacher_item_sha256": item["sha256"],
                   "asset_sha256": entry["asset_sha256"], "conditioning_sha256": conditioning, "recovery_group_id": entry["recovery_group_id"],
                   "split": "train", "held": False, "amendment_sha256": amendment_sha, "tensors_sha256": R.sha(stage / name),
                   "source_cache": str(cache.resolve()), "source_shard": shard["name"], "adapter_sha256": R.sha(Path(__file__)),
                   "prompt": prompt_proof, **proof}
        receipt_name = ident + ".json"
        (stage / receipt_name).write_bytes(R.canonical(receipt) + b"\n")
        accepted.append({"id": ident, "route": "music_generate", "recovery_group_id": entry["recovery_group_id"],
                         "tensors": {"path": name, "sha256": receipt["tensors_sha256"]},
                         "teacher_receipt": {"path": receipt_name, "sha256": R.sha(stage / receipt_name)}})
        if limit is not None and len(accepted) >= limit:
            complete = False
            break
    result = {"schema": "chimera_recovery.music_batches.v1", "stage": "D3c", "sealed": complete and bool(accepted),
              "complete_source_traversal": complete, "split": "train", "heldout_groups_excluded": True,
              "source_manifest_sha256": manifest_sha, "source_items_sha256": R.sha(items_path), "corpus_sha256": R.sha(admission_path),
              "amendment_sha256": amendment_sha, "adapter_sha256": R.sha(Path(__file__)), "batches": accepted,
              "excluded": excluded, "quarantine": quarantine, "source_teacher": TEACHER, "capability_claim": "none"}
    (stage / "manifest.json").write_bytes(R.canonical(result) + b"\n")
    stage.rename(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "items", "admission", "tokenizer", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    result = prepare_music(args.cache, args.items, args.admission, tokenizer, args.amendment_sha256, args.out,
                           manifest_sha=args.manifest_sha256, max_length=args.max_length, limit=args.limit)
    print(json.dumps({"sealed": result["sealed"], "accepted": len(result["batches"]), "quarantined": len(result["quarantine"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
