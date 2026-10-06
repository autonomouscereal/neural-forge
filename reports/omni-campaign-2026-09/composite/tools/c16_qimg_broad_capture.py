#!/usr/bin/env python3
"""Seal receipt-derived broad QIMG plans, then encode/capture without a launcher.

No image selection, prompt editing, retry, fold scoring, training, GPU allocation,
container launch or cleanup. Actual native subprocesses require an independently
managed runtime. Run outputs must be fresh; interrupted outputs are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from PIL import Image
import torch
from safetensors.torch import load_file

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import c16_qimg_capture as capture
import c16_qimg_prepare_inputs as prepare
import c16_qimg_parent_images as parent
import c16_qimg_fold as fold
import c16_qimg_joint_replay as joint

SCHEMA = "neural_forge.qimg_broad_capture_bundle.v1"
RUN_SCHEMA = "neural_forge.qimg_broad_capture_run.v1"
SIGMAS = [.15, .35, .65, .9]
F32_SIGMAS = [struct.unpack("<f", struct.pack("<f", s))[0] for s in SIGMAS]
SEED_RULE = "sha256(qimg-broad-calibration-v1 NUL plan_sha256 NUL id NUL generation_seed); first8 big-endian masked int63"
TOOL_NAMES = (Path(__file__).name, "c16_qimg_prepare_inputs.py", "c16_qimg_capture.py",
              "c16_qimg_parent_images.py", "c16_qimg_preflight.py", "c16_qimg_fold.py", "c16_qimg_joint_replay.py")


def json_pinned(path, digest):
    path = Path(path)
    fold.digest(digest)
    if path.stat().st_size > 2 << 20:
        raise ValueError("bounded receipt/plan exceeds 2 MiB")
    raw = path.read_bytes()
    if len(raw) > 2 << 20 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("receipt/plan SHA256 or size mismatch")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def reference(path, digest=None):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": fold.digest(digest or capture.file_sha(path))}


def exclusive_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as handle:
        handle.write(raw)
    return reference(path, hashlib.sha256(raw).hexdigest())


def tool_hashes():
    return {name: capture.file_sha(TOOLS / name) for name in TOOL_NAMES}


def source_content(identity):
    """Mount paths are provenance, not a tensor/header byte identity."""
    if (not isinstance(identity, dict) or set(identity) != {"path", "file_bytes", "header_bytes", "header_sha256", "full_file_sha256", "scope"}
            or not isinstance(identity["path"], str) or not identity["path"]
            or type(identity["file_bytes"]) is not int or type(identity["header_bytes"]) is not int
            or not 2 <= identity["header_bytes"] < identity["file_bytes"]
            or identity["full_file_sha256"] is not None or identity["scope"] != "header only; tensor bodies not read"):
        raise ValueError("exact truthful header-only source identity required")
    fold.digest(identity["header_sha256"])
    return {key: value for key, value in identity.items() if key != "path"}


def calibration_seed(plan_sha, entry):
    text = f"qimg-broad-calibration-v1\0{plan_sha}\0{entry['id']}\0{entry['seed']}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def excluded_images(refs):
    if not 1 <= len(refs) <= 32:
        raise ValueError("explicit old capture receipt exclusions required")
    seen, hashes = set(), set()
    for ref in refs:
        digest = fold.digest(ref["sha256"])
        if digest in seen:
            raise ValueError("duplicate exclusion receipt")
        seen.add(digest)
        body = json_pinned(ref["path"], digest)
        if body.get("schema") != capture.CAPTURE_SCHEMA or body.get("status") != "complete" or body.get("fixture") is not False:
            raise ValueError("old exclusions must be complete real capture receipts")
        rows = body.get("captures")
        if not isinstance(rows, list) or not rows:
            raise ValueError("empty old image exclusion evidence")
        hashes.update(fold.digest(item["image_sha256"]) for item in rows)
    if len(hashes) < 8:
        raise ValueError("exclusions must cover at least the eight prior development images")
    return hashes


def derive_plans(batches, exclusions):
    """Legacy contract: only the four committed September 7 broad plans."""
    return _derive_plans_against(batches, exclusions, None)


def _derive_plans_against(batches, exclusions, canonical_plans):
    """Shared native audits; fresh callers must separately prove plan authority.

    This private helper grants no preregistration, launch or acceptance authority.
    The legacy entry point still supplies only its committed September 7 plans.
    """
    if canonical_plans is not None and (not isinstance(canonical_plans, list) or len(canonical_plans) != 4):
        raise ValueError("four independently verified canonical generation plans required")
    if not isinstance(batches, list) or len(batches) != 4:
        raise ValueError("exactly four complete broad batches required")
    old = excluded_images(exclusions)
    ids, prompts, pngs, seeds = set(), set(), set(), set()
    receipt_hashes, plan_hashes, sources = set(), set(), None
    result = []
    expected_source_code = {**capture.SOURCE_FILES, **prepare.SOURCE_FILES, **parent.SAMPLING_SOURCE}
    for number, batch in enumerate(batches, 1):
        if set(batch) != {"number", "generation_plan", "parent_receipt"} or batch["number"] != number or type(batch["number"]) is not int:
            raise ValueError("ordered exact batch identities 1..4 required")
        gp, rp = batch["generation_plan"], batch["parent_receipt"]
        plan = json_pinned(gp["path"], gp["sha256"])
        # Compare semantic content with the committed preregistration, while the
        # explicit caller hash binds exact archived bytes (including LF/CRLF).
        if canonical_plans is None:
            canonical_path = TOOLS / f"qimg_parent_broad_sep07_batch{number}.json"
            canonical = parent.read_plan(canonical_path)
        else:
            canonical = canonical_plans[number - 1]
        if plan != canonical or len(plan["entries"]) != 8 or Counter(e["split"] for e in plan["entries"]) != {"fit": 6, "holdout": 2}:
            raise ValueError("generation plan differs from the complete committed broad batch")
        body = json_pinned(rp["path"], rp["sha256"])
        if (body.get("schema") != "neural_forge.qimg_parent_images.v1" or body.get("status") != "complete"
                or body.get("plan_sha256") != gp["sha256"] or body.get("sampling") != plan["sampling"]
                or body.get("source_code_sha256") != expected_source_code):
            raise ValueError("parent receipt is incomplete or not linked to pinned generation/source")
        if gp["sha256"] in plan_hashes or rp["sha256"] in receipt_hashes:
            raise ValueError("duplicate generation plan or parent receipt")
        plan_hashes.add(gp["sha256"]); receipt_hashes.add(rp["sha256"])
        identity = body.get("sources", {})
        if set(identity) != {"diffusion", "vae", "text_encoder"}:
            raise ValueError("all native teacher source identities required")
        for role in identity:
            source_content(identity[role])
        if identity["diffusion"]["header_sha256"] != capture.PARENT_HEADER_SHA256:
            raise ValueError("wrong source diffusion header")
        if sources is not None and any(source_content(identity[role]) != source_content(sources[role]) for role in identity):
            raise ValueError("parent batches used different teacher source identities")
        sources = identity
        runtime = body.get("runtime", {})
        if (runtime.get("parent_class") != "comfy.model_base.QwenImage"
                or runtime.get("manual_cast_dtype") != "torch.bfloat16" or runtime.get("fp8_optimizations") is not False
                or runtime.get("full_residency_verified") is not True or runtime.get("dynamic_patcher") is not False):
            raise ValueError("complete frozen native parent runtime required")
        images = body.get("images", [])
        if not isinstance(images, list) or len(images) != 8:
            raise ValueError("partial/selected parent batches are forbidden")
        entries = []
        for planned, actual in zip(plan["entries"], images):
            if {key: actual.get(key) for key in planned} != planned:
                raise ValueError("parent image order, prompt, split, generation seed or ID changed")
            if actual.get("provenance") != "native Qwen-Image parent generated; no selection; no edits; not a photograph":
                raise ValueError("teacher-generated origin must remain explicit")
            fold.digest(actual["decoded_pixels_sha256"]); fold.digest(actual["native_sample_latent_sha256"])
            if actual.get("file") != planned["id"] + ".png":
                raise ValueError("unexpected parent PNG filename")
            path = fold.artifact(Path(rp["path"]).parent, actual["file"])
            digest = fold.digest(actual["sha256"])
            if path.stat().st_size > 128 << 20 or capture.file_sha(path) != digest:
                raise ValueError("original parent PNG bytes changed")
            with Image.open(path) as image:
                image.verify()  # Must precede getexif/load, which closes PNG.fp.
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (512, 512) or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1 or image.getexif() or "transparency" in image.info:
                    raise ValueError("plain original RGB 512x512 parent PNG required; no transforms")
            seed = calibration_seed(gp["sha256"], planned)
            if planned["id"] in ids or planned["prompt"] in prompts or digest in pngs or digest in old or seed in seeds:
                raise ValueError("duplicate/leaked/old image, prompt, ID or calibration seed")
            ids.add(planned["id"]); prompts.add(planned["prompt"]); pngs.add(digest); seeds.add(seed)
            entries.append({"id": planned["id"], "split": planned["split"], "prompt": planned["prompt"], "seed": seed,
                "image": {"path": str(path), "sha256": digest},
                "image_provenance": {"kind": "native_qwen_image_parent_generated",
                    "description": actual["provenance"], "parent_receipt": dict(rp), "generation_plan": dict(gp),
                    "generation_seed": planned["seed"], "decoded_pixels_sha256": actual["decoded_pixels_sha256"],
                    "native_sample_latent_sha256": actual["native_sample_latent_sha256"],
                    "calibration_seed_rule": SEED_RULE}})
        result.append({"schema": prepare.PLAN_SCHEMA, "sigmas": SIGMAS, "entries": entries})
    return result, sources, sorted(old)


def seal(batches, exclusions, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError("seal output exists; no overwrite or cleanup")
    plans, sources, old = derive_plans(batches, exclusions)
    out.mkdir(parents=True, exist_ok=False)
    refs = [exclusive_json(out / f"encoding_batch{i}.json", plan) for i, plan in enumerate(plans, 1)]
    bundle = {"schema": SCHEMA, "status": "sealed_before_encoding_and_capture", "batches": batches,
        "exclusions": exclusions, "excluded_image_sha256": old, "encoding_plans": refs,
        "source_identities": sources, "tool_sha256": tool_hashes(), "calibration_seed_rule": SEED_RULE,
        "sigmas": SIGMAS, "expected": {"images": 32, "fit_images": 24, "holdout_images": 8,
            "forwards": 128, "image_rows": {"fit": 98304, "holdout": 32768},
            "image_row_cap": 131072, "minimum_actual_fit_text_rows": 6142},
        "scope": "sealed calibration/capture inputs only; no fold scoring or acceptance"}
    return exclusive_json(out / "broad_capture_bundle.json", bundle)


def validate_bundle(path, digest):
    bundle = json_pinned(path, digest)
    if (set(bundle) != {"schema", "status", "batches", "exclusions", "excluded_image_sha256", "encoding_plans",
                        "source_identities", "tool_sha256", "calibration_seed_rule", "sigmas", "expected", "scope"}
            or bundle.get("schema") != SCHEMA or bundle.get("status") != "sealed_before_encoding_and_capture"
            or bundle.get("tool_sha256") != tool_hashes()):
        raise ValueError("unsealed bundle or changed production runner/tool source")
    plans, sources, old = derive_plans(bundle["batches"], bundle["exclusions"])
    if (bundle["source_identities"] != sources or bundle["excluded_image_sha256"] != old
            or bundle["sigmas"] != SIGMAS or bundle["calibration_seed_rule"] != SEED_RULE
            or bundle["expected"] != {"images": 32, "fit_images": 24, "holdout_images": 8, "forwards": 128,
                "image_rows": {"fit": 98304, "holdout": 32768}, "image_row_cap": 131072, "minimum_actual_fit_text_rows": 6142}
            or len(bundle["encoding_plans"]) != 4):
        raise ValueError("sealed bundle contract changed")
    for expected, ref in zip(plans, bundle["encoding_plans"]):
        if json_pinned(ref["path"], ref["sha256"]) != expected:
            raise ValueError("sealed encoding plan differs from its exact source receipts")
    return bundle, plans


def audit_encoding(out, plan, plan_ref, sources):
    entries, linked = capture.load_inputs(out / "capture_inputs.json", max_tokens=2048, max_forwards=32)
    body = json_pinned(out / "encoding_receipt.json", linked["encoding_receipt_sha256"])
    if (body.get("schema") != "neural_forge.qimg_encoding_receipt.v1" or body.get("plan_sha256") != plan_ref["sha256"]
            or body.get("source_code_sha256") != prepare.SOURCE_FILES or body.get("vae_loaded_complete_key_shape_coverage") is not True
            or linked["forwards"] != 32 or len(entries) != 8
            or set(body.get("entries", {})) != {e["id"] for e in plan["entries"]}
            or set(body.get("details", {})) != set(body["entries"])):
        raise ValueError("encoding did not preserve the complete sealed plan/source contract")
    for role in ("vae", "text_encoder"):
        if source_content(linked["sources"][role]) != source_content(sources[role]):
            raise ValueError("encoded teacher identity differs from parent generation")
    counts = {s: {"fit": 0, "holdout": 0} for s in ("img", "txt")}
    actual = {}
    for item, expected in zip(entries, plan["entries"]):
        entry, name = item["entry"], expected["id"]
        if ({k: entry.get(k) for k in ("id", "split", "prompt", "image")} != {k: expected[k] for k in ("id", "split", "prompt", "image")}):
            raise ValueError("encoded image/prompt/split order differs from sealed plan")
        detail = body["details"][name]
        if (detail.get("image_provenance") != expected["image_provenance"] or detail.get("noise", {}).get("seed") != expected["seed"]
                or detail["noise"].get("sigma_shift_applied") is not False or detail["noise"].get("noise_scale") != 1.0):
            raise ValueError("encoding changed origin, noise seed or source sigma recipe")
        state, _ = fold.verified_tensors(out, detail["tensors"], 64 << 20)
        capture.validate_inputs(state, max_tokens=2048)
        if state["model_inputs"].shape != (4, 16, 1, 64, 64) or not torch.equal(state["sigma"], torch.tensor(SIGMAS, dtype=torch.float32)):
            raise ValueError("actual encoding grid or FP32 sigmas differ from the sealed 512px experiment")
        if detail["tensors"]["file_sha256"] != entry["tensors"]["sha256"]:
            raise ValueError("encoding detailed tensor identities do not match capture inputs")
        valid = int(state["context_mask"].sum())
        counts["img"][expected["split"]] += 4 * 1024
        counts["txt"][expected["split"]] += 4 * valid
        actual[name] = {"valid_text_rows_per_forward": valid, "text_tokens": state["context"].shape[1],
                        "sigma_sha256": capture.tensor_sha(state["sigma"]),
                        "context_valid_sha256": capture.tensor_sha(state["context_mask"])}
        del state
    return {"inputs": linked, "row_counts": counts, "actual_rows": actual,
            "encoding_receipt": reference(out / "encoding_receipt.json")}


def audit_capture(out, plan, encoding, sources, previous_frozen=None):
    ref = reference(out / "capture_receipt.json")
    body = json_pinned(ref["path"], ref["sha256"])
    if (body.get("schema") != capture.CAPTURE_SCHEMA or body.get("status") != "complete" or body.get("fixture") is not False
            or body.get("block") != 30 or body.get("inputs") != encoding["inputs"]
            or body.get("source_code_sha256") != capture.SOURCE_FILES or body.get("source_header_stable_after_capture") is not True
            or source_content(body.get("source", {}).get("source_identity")) != source_content(sources["diffusion"])):
        raise ValueError("capture differs from its sealed upstream/source identity")
    runtime = body.get("runtime", {})
    if (runtime.get("parent_class") != "comfy.model_base.QwenImage" or runtime.get("full_residency_verified") is not True
            or runtime.get("selected_block_and_time_match_source_bytes") is not True
            or runtime.get("manual_cast_dtype") != "torch.bfloat16" or runtime.get("fp8_optimizations") is not False
            or runtime.get("dynamic_patcher") is not False):
        raise ValueError("actual native capture source/cast/residency contract incomplete")
    records = body.get("captures", [])
    if len(records) != 32:
        raise ValueError("exactly 32 complete source forwards per batch required")
    frozen, _ = fold.verified_tensors(out, body["frozen_source"], 2 << 30)
    if ({k: tuple(v.shape) for k, v in frozen.items()} != fold.frozen_shapes(30, 3072, 12288, 128)
            or any(v.dtype != torch.float8_e4m3fn for v in frozen.values())):
        raise ValueError("capture frozen source block/time geometry or dtype changed")
    for value in frozen.values():
        if any(not bool(torch.isfinite(chunk.float()).all()) for chunk in value.reshape(-1).split(1 << 20)):
            raise ValueError("nonfinite source parameter")
    frozen_sha = body["frozen_source"]["file_sha256"]
    if previous_frozen is not None and frozen_sha != previous_frozen:
        raise ValueError("capture batches changed selected source block/time bytes")
    counts = {s: {"fit": 0, "holdout": 0} for s in ("img", "txt")}
    seen = set()
    input_entries = {e["id"]: e for e in encoding["inputs"]["entries"]}
    encoded = None
    for record, (entry, index) in zip(records, [(e, i) for e in plan["entries"] for i in range(4)]):
        name = entry["id"]
        if (record.get("sample_id") != name or record.get("split") != entry["split"]
                or record.get("image_sha256") != entry["image"]["sha256"]
                or record.get("prompt_sha256") != hashlib.sha256(entry["prompt"].encode()).hexdigest()
                or record.get("sigma") != F32_SIGMAS[index]
                or record.get("file") != f"{name}_sigma{index:02d}.safetensors"
                or record.get("image_tokens") != 1024 or record.get("fixture") is not False
                or record.get("complete_context") is not True or record.get("row_order") != "text_then_image"):
            raise ValueError("captured forward does not match exact planned image/sigma/order")
        if record["file_sha256"] in seen:
            raise ValueError("duplicate captured forward")
        seen.add(record["file_sha256"])
        state, _ = fold.verified_tensors(out, record, 256 << 20)
        if (set(state) != fold.CAPTURE_KEYS or state["sigma"].dtype != torch.float32
                or state["sigma"].shape != (1,) or float(state["sigma"][0]) != F32_SIGMAS[index]):
            raise ValueError("missing native capture tensors or changed actual sigma")
        if index == 0:
            encoded_path = capture.verified_file(Path(encoding["inputs"]["manifest"]).parent, input_entries[name]["tensors"], max_bytes=64 << 20)
            encoded = load_file(str(encoded_path), device="cpu")
            capture.validate_inputs(encoded, max_tokens=2048)
        if (not torch.equal(state["model_input"], encoded["model_inputs"][index:index + 1].bfloat16())
                or not torch.equal(state["context_token_ids"], encoded["context_token_ids"])
                or bool((state["text_attention_mask"] > 0).any())):
            raise ValueError("captured actual inputs/token IDs/mask differ from the sealed native encoding")
        joint.context_contract(state, record, {"width": 3072, "frozen": frozen, "block": 30})
        if capture.tensor_sha(state["context_valid"]) != encoding["actual_rows"][name]["context_valid_sha256"]:
            raise ValueError("capture validity differs from its actual encoding")
        for tensor_name, value in state.items():
            if value.is_floating_point() and tensor_name != "text_attention_mask" and not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite captured source tensor")
        counts["img"][entry["split"]] += 1024
        counts["txt"][entry["split"]] += int(state["context_valid"].sum())
        del state
    if counts != encoding["row_counts"]:
        raise ValueError("actual captured rows differ from the validated encoding rows")
    return {"capture_receipt": ref, "row_counts": counts, "forwards": 32, "source_frozen_sha256": frozen_sha,
            "forward_file_sha256": sorted(seen)}


def total_rows(reports):
    return {s: {split: sum(r["row_counts"][s][split] for r in reports) for split in ("fit", "holdout")} for s in ("img", "txt")}


def require_rows(reports):
    rows = total_rows(reports)
    if rows["img"] != {"fit": 98304, "holdout": 32768} or sum(rows["img"].values()) > 131072:
        raise ValueError("actual full image-row count differs from the sealed 131072-row cap")
    if rows["txt"]["fit"] < 6142 or rows["txt"]["holdout"] < 1 or sum(rows["txt"].values()) > 131072:
        raise ValueError("actual text rows fail the unchanged keep3071 row floor/cap")
    return rows


def run(bundle_path, bundle_sha, out, weights, vae, text_encoder, comfy_root, threads):
    """Legacy command and sealed September 7 bundle contract are unchanged."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError("run output exists; no overwrite, retry or cleanup")
    if type(threads) is not int or not 1 <= threads <= 22:
        raise ValueError("explicit bounded CPU thread count 1..22 required")
    bundle, plans = validate_bundle(bundle_path, bundle_sha)
    return _run_audited(bundle_path, bundle_sha, out, weights, vae, text_encoder,
                        comfy_root, threads, bundle, plans,
                        lambda: validate_bundle(bundle_path, bundle_sha))


def _run_audited(bundle_path, bundle_sha, out, weights, vae, text_encoder,
                 comfy_root, threads, bundle, plans, revalidate, before_dispatch=None):
    """Shared native subprocess/audit pipeline; authorization belongs to caller."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise ValueError("run output exists; no overwrite, retry or cleanup")
    if type(threads) is not int or not 1 <= threads <= 22:
        raise ValueError("explicit bounded CPU thread count 1..22 required")
    for role, path in (("diffusion", weights), ("vae", vae), ("text_encoder", text_encoder)):
        if source_content(capture.read_header(Path(path))[1]) != source_content(bundle["source_identities"][role]):
            raise ValueError("runtime source header/size differs from actual generation receipt")
    torch.set_num_threads(threads)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
    out.mkdir(parents=True, exist_ok=False)
    encodings, captures, commands = [], [], []
    def execute(argv):
        # Every subprocess runs a committed existing tool; no shell or launcher.
        if tool_hashes() != bundle["tool_sha256"]:
            raise ValueError("production source changed after seal")
        if before_dispatch is not None:
            before_dispatch()
        commands.append(argv)
        subprocess.run(argv, check=True, env=env)
    try:
        for i, (plan, ref) in enumerate(zip(plans, bundle["encoding_plans"]), 1):
            json_pinned(ref["path"], ref["sha256"])
            target = out / f"encoded_batch{i}"
            execute([sys.executable, str(TOOLS / "c16_qimg_prepare_inputs.py"), "--plan", ref["path"],
                     "--vae", str(vae), "--text-encoder", str(text_encoder), "--comfy-root", str(comfy_root),
                     "--device", "cuda:0", "--out", str(target)])
            encodings.append(audit_encoding(target, plan, ref, bundle["source_identities"]))
        require_rows(encodings)  # actual tokenized fit rows BEFORE all parent captures
        for i, (plan, encoding) in enumerate(zip(plans, encodings), 1):
            target = out / f"capture_batch{i}"
            execute([sys.executable, str(TOOLS / "c16_qimg_capture.py"), "--weights", str(weights),
                     "--inputs", encoding["inputs"]["manifest"], "--out", str(target), "--comfy-root", str(comfy_root),
                     "--device", "cuda:0", "--block", "30", "--max-tokens", "2048", "--max-forwards", "32"])
            captures.append(audit_capture(target, plan, encoding, bundle["source_identities"],
                                         captures[0]["source_frozen_sha256"] if captures else None))
        rows = require_rows(captures)
        if sum(c["forwards"] for c in captures) != 128:
            raise ValueError("actual capture did not complete all 128 forwards")
        forward_hashes = [digest for batch in captures for digest in batch["forward_file_sha256"]]
        if len(forward_hashes) != 128 or len(set(forward_hashes)) != 128:
            raise ValueError("captured forward bytes were duplicated across batches")
        revalidate()
        result = {"schema": RUN_SCHEMA, "status": "complete", "bundle": reference(bundle_path, bundle_sha),
                  "encodings": encodings, "captures": captures, "actual_row_counts": rows, "actual_forwards": 128,
                  "commands": commands, "torch_threads": threads,
                  "scope": "actual source encodings/captures audited; no fold fitting, scoring, threshold change or child claim",
                  "runtime_image_and_gpu_allocation": "external coordinator obligation; this runner neither allocates nor attests either"}
        return exclusive_json(out / "broad_capture_run_receipt.json", result)
    except BaseException as exc:
        exclusive_json(out / "broad_capture_failure.json", {"schema": RUN_SCHEMA, "status": "incomplete",
            "bundle": reference(bundle_path, bundle_sha), "completed_encodings": encodings, "completed_captures": captures,
            "commands": commands, "error": {"type": type(exc).__name__, "message": str(exc)},
            "scope": "historical outputs preserved; no retry, overwrite, cleanup or acceptance"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seal")
    p.add_argument("--batch", action="append", nargs=4, required=True, metavar=("PLAN", "PLAN_SHA", "RECEIPT", "RECEIPT_SHA"))
    p.add_argument("--exclude-capture", action="append", nargs=2, required=True, metavar=("RECEIPT", "SHA"))
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--bundle-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    for name in ("weights", "vae", "text-encoder", "comfy-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--threads", type=int, required=True)
    a = parser.parse_args()
    if a.command == "seal":
        batches = [{"number": i, "generation_plan": reference(gp, gs), "parent_receipt": reference(rp, rs)}
                   for i, (gp, gs, rp, rs) in enumerate(a.batch, 1)]
        result = seal(batches, [reference(p, s) for p, s in a.exclude_capture], a.out)
    else:
        result = run(a.bundle, a.bundle_sha256, a.out, a.weights, a.vae, a.text_encoder, a.comfy_root, a.threads)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
