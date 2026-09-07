#!/usr/bin/env python3
"""Capture an authentic plain-T2I Qwen-Image source block, without a Comfy server.

Production input is an identity-checked bundle of real-image model inputs and
Qwen2.5-VL prompt context. This program generates no replacement stimuli, random
context, encodings or sampler trajectory. It records what the existing parent
actually computes. A capture is not a folded container or a capability result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c16_qimg_preflight import SOURCE_CODE_SHA256, inspect_weights, read_header


INPUT_SCHEMA = "neural_forge.qimg_capture_inputs.v1"
CAPTURE_SCHEMA = "neural_forge.qimg_source_capture.v1"
SOURCE_FILES = {
    "comfy/ldm/qwen_image/model.py": SOURCE_CODE_SHA256,
    "comfy/ldm/flux/math.py": "ee3473e262894884b12eb3af4caa22d069273eb87db3549fb46c42c37f910ece",
    "comfy/ldm/flux/layers.py": "35f2dfbadb8b59de79c306f365f26c2baa0a7d54836414737cd781fc24e6a2bc",
    "comfy/ldm/lightricks/model.py": "c9ae1a8505aa0eabefdffe409083141e5e8461dd9d55f452c13f0f7b0c6466d8",
    "comfy/model_sampling.py": "8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717",
    "comfy/model_management.py": "b7741d049c2c749c316e5769ef520846bc7dd444e67de08403a4ee13a9c034c2",
}
PARENT_HEADER_SHA256 = "bd90f4d6af8d235cb3b46ccf5f426abb351995e28e852b04fc678a541d87158e"
INPUT_KEYS = {"model_inputs", "sigma", "context", "context_mask", "context_token_ids"}


def file_sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_file(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def verified_file(root, body, *, field="path", max_bytes=None):
    path = resolve_file(root, body[field])
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError(f"input artifact exceeds the bounded file size: {path}")
    expected = body.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or file_sha(path) != expected:
        raise ValueError(f"file identity mismatch: {path}")
    return path


def load_inputs(manifest_path: Path, *, max_tokens=2048, max_forwards=32):
    """Verify artifact identities and an explicit actual-input contract, not encodings."""
    if not 1 <= max_tokens <= 2048 or not 1 <= max_forwards <= 32:
        raise ValueError("capture is bounded to 2048 full-context tokens and 32 forwards")
    if manifest_path.stat().st_size > 1 << 20:
        raise ValueError("input manifest exceeds the bounded file size")
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    if body.get("schema") != INPUT_SCHEMA or body.get("kind") != "real_image_vae_qwen25_context":
        raise ValueError("production capture requires real image/VAE/Qwen context provenance; fixture inputs are forbidden")
    root = manifest_path.resolve().parent
    if body.get("latent_space") != "wan21_model_input" or body.get("context_kind") != "qwen25_vl_7b_hidden_states":
        raise ValueError("inputs must declare actual Wan21 model inputs and Qwen2.5-VL hidden-state context")
    # Verify the small upstream receipt and its referenced model headers, without
    # hashing multi-GB parent/VAE/text-encoder tensor bodies.
    upstream_path = verified_file(root, body["encoding_receipt"], max_bytes=1 << 20)
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if (upstream.get("status") != "complete" or upstream.get("kind") != body["kind"]
            or upstream.get("latent_space") != body["latent_space"]):
        raise ValueError("upstream receipt has an incompatible encoding contract")
    sources = {}
    for role in ("vae", "text_encoder"):
        entry = upstream[role]
        source_path = resolve_file(upstream_path.parent, entry["path"])
        _, identity = read_header(source_path)
        if identity["header_sha256"] != entry.get("header_sha256"):
            raise ValueError(f"{role} source header does not match the upstream receipt")
        sources[role] = identity
    entries = body.get("entries", [])
    if not entries or len(entries) > max_forwards:
        raise ValueError("one to max_forwards actual image entries are required")
    ids, image_splits, result, total = set(), {}, [], 0
    for entry in entries:
        sample_id = entry.get("id", "")
        split = entry.get("split")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sample_id) or sample_id in ids or split not in {"fit", "holdout"}:
            raise ValueError("sample IDs must be unique safe names with explicit fit/holdout roles")
        ids.add(sample_id)
        image_path = verified_file(root, entry["image"], max_bytes=128 << 20)
        image_digest = entry["image"]["sha256"]
        if image_digest in image_splits and image_splits[image_digest] != split:
            raise ValueError("one image identity appears on both sides of the fit/holdout split")
        image_splits[image_digest] = split
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != entry.get("prompt_sha256"):
            raise ValueError("prompt identity mismatch")
        tensor_path = verified_file(root, entry["tensors"], max_bytes=64 << 20)
        linkage = upstream.get("entries", {}).get(sample_id)
        expected_link = {"image_sha256": image_digest, "prompt_sha256": entry["prompt_sha256"],
                         "tensor_file_sha256": entry["tensors"]["sha256"]}
        if linkage != expected_link:
            raise ValueError("upstream image/prompt/tensor linkage does not match the actual entry")
        input_header, _ = read_header(tensor_path)
        if set(input_header) != INPUT_KEYS:
            raise ValueError("input header has an unexpected tensor set")
        input_shape, context_shape = input_header["model_inputs"]["shape"], input_header["context"]["shape"]
        if len(input_shape) != 5 or len(context_shape) != 3 or input_shape[0] + total > max_forwards:
            raise ValueError("input header exceeds forward count or has invalid geometry")
        if context_shape[1] + input_shape[-2] * input_shape[-1] // 4 > max_tokens:
            raise ValueError("input header exceeds the full-context token count")
        state = load_file(str(tensor_path), device="cpu")
        validate_inputs(state, max_tokens=max_tokens)
        total += len(state["sigma"])
        if total > max_forwards:
            raise ValueError("input bundle exceeds the bounded forward count")
        result.append({"entry": entry, "tensor_path": tensor_path, "image_path": image_path,
                       "forwards": len(state["sigma"]), "token_count": state["context"].shape[1] +
                       state["model_inputs"].shape[-2] * state["model_inputs"].shape[-1] // 4,
                       "latent_shape": list(state["model_inputs"].shape[1:])})
    return result, {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_sha(manifest_path),
                    "encoding_receipt": str(upstream_path), "encoding_receipt_sha256": file_sha(upstream_path), "sources": sources,
                    "forwards": total, "entries": entries,
                    "scope": "artifact identities and upstream linkage verified; upstream encodings not numerically re-run"}


def validate_inputs(state, *, max_tokens, context_width=3584):
    if set(state) != INPUT_KEYS:
        raise ValueError(f"input tensor keys must be exactly {sorted(INPUT_KEYS)}")
    x, sigma, context = state["model_inputs"], state["sigma"], state["context"]
    if x.ndim != 5 or x.shape[1:3] != (16, 1) or min(x.shape[-2:]) < 2 or any(n % 2 for n in x.shape[-2:]):
        raise ValueError("model_inputs must be [forwards,16,1,even_height,even_width] for plain image capture")
    if sigma.ndim != 1 or len(sigma) != len(x) or not len(sigma):
        raise ValueError("sigma must contain one actual model timestep per forward")
    if sigma.dtype != torch.float32 or not bool(torch.isfinite(sigma).all()) or bool(((sigma < 0) | (sigma > 1)).any()):
        raise ValueError("actual sigma must be finite FP32 in [0,1]")
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[2] != context_width or not context.shape[1]:
        raise ValueError("context must be [1,text_tokens,3584] actual Qwen2.5-VL hidden states")
    if context.shape[1] + x.shape[-2] * x.shape[-1] // 4 > max_tokens:
        raise ValueError("full text/image context exceeds the bounded token count")
    for name in ("context_mask", "context_token_ids"):
        if state[name].shape != context.shape[:2]:
            raise ValueError(f"{name} must align one-to-one with context rows")
    if state["context_mask"].dtype != torch.bool or not bool(state["context_mask"].any()):
        raise ValueError("context_mask must be boolean with at least one valid text token")
    if state["context_token_ids"].dtype != torch.int64 or bool((state["context_token_ids"] < 0).any()):
        raise ValueError("context_token_ids must be nonnegative int64 aligned encoder token IDs")
    for name in ("model_inputs", "context"):
        if state[name].dtype not in {torch.float32, torch.bfloat16} or not bool(torch.isfinite(state[name]).all()):
            raise ValueError(f"{name} must be finite FP32/BF16 actual inputs")


class QimgBlockTap:
    """Hooks only: source tensors are cloned before any possible in-place mutation."""

    def __init__(self, dit, block_index, *, fixture=False):
        if not fixture and (type(dit).__module__ != "comfy.ldm.qwen_image.model" or
                            type(dit).__name__ != "QwenImageTransformer2DModel"):
            raise ValueError("production tap requires the actual Qwen-Image source class")
        self.dit, self.block_index, self.fixture = dit, block_index, fixture
        self.records, self.handles = {}, []
        self.mask_present = None

    def put(self, name, value):
        if name in self.records:
            raise ValueError(f"source hook fired more than once in one forward: {name}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"source hook did not receive a tensor: {name}")
        value = value.detach().clone().to("cpu").contiguous()
        if value.is_floating_point():
            if name == "text_attention_mask":
                # Native Qwen accepts an additive mask unchanged. Negative
                # infinity is a valid blocked key, not a failed activation.
                if bool((torch.isnan(value) | (value > 0)).any()):
                    raise ValueError("invalid additive source text attention mask")
            elif not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite source capture: {name}")
        self.records[name] = value

    def __enter__(self):
        if self.handles:
            raise ValueError("capture tap cannot be entered twice")
        block = self.dit.transformer_blocks[self.block_index]
        try:
            self.handles.append(self.dit.pe_embedder.register_forward_pre_hook(lambda m, a: self.put("source_ids", a[0])))
            self.handles.append(self.dit.pe_embedder.register_forward_hook(lambda m, a, out: self.put("rotation_generated", out)))
            self.handles.append(block.register_forward_pre_hook(self.block_pre, with_kwargs=True))
            self.handles.append(block.register_forward_hook(self.block_post))
            self.handles.append(block.attn.register_forward_hook(self.attn_post))
            for stream in ("img", "txt"):
                mlp = getattr(block, stream + "_mlp")
                mod = getattr(block, stream + "_mod")
                self.handles.append(mlp.register_forward_pre_hook(self.mlp_pre(stream)))
                self.handles.append(mlp.register_forward_hook(self.store_output(stream + "_mlp_out")))
                self.handles.append(mod.register_forward_hook(self.store_output(stream + "_modulation")))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def block_pre(self, module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values.get("timestep_zero_index") is not None:
            raise ValueError("reference-image / zero-timestep source variants are not supported")
        options = values.get("transformer_options", {})
        if any(options.get(k) for k in ("patches", "patches_replace", "reference_image_num_tokens")):
            raise ValueError("patched or reference-image source context is not supported")
        for key, target in (("hidden_states", "img_in"), ("encoder_hidden_states", "txt_in"),
                            ("temb", "temb"), ("image_rotary_emb", "rotation_used")):
            self.put(target, values[key])
        mask = values["encoder_hidden_states_mask"]
        self.mask_present = mask is not None
        if mask is not None:
            self.put("text_attention_mask", mask)

    def block_post(self, module, args, output):
        self.put("txt_out", output[0])
        self.put("img_out", output[1])

    def attn_post(self, module, args, output):
        self.put("img_attn_out", output[0])
        self.put("txt_attn_out", output[1])

    def mlp_pre(self, stream):
        return lambda module, args: self.put(stream + "_mlp_in", args[0])

    def store_output(self, name):
        return lambda module, args, output: self.put(name, output)

    def finish(self):
        required = {"source_ids", "rotation_generated", "rotation_used", "temb"}
        for stream in ("img", "txt"):
            required |= {stream + suffix for suffix in ("_in", "_out", "_attn_out", "_mlp_in", "_mlp_out", "_modulation")}
        if self.mask_present:
            required.add("text_attention_mask")
        if set(self.records) != required:
            raise ValueError(f"incomplete source hooks: missing {sorted(required - set(self.records))}")
        r = self.records
        img, txt = r["img_in"], r["txt_in"]
        if img.ndim != 3 or txt.ndim != 3 or img.shape[0] != 1 or txt.shape[0] != 1 or img.shape[2] != txt.shape[2]:
            raise ValueError("capture must retain one complete image/text forward")
        total = img.shape[1] + txt.shape[1]
        if r["source_ids"].shape != (1, total, 3):
            raise ValueError("source IDs do not align with full text-first image context")
        if not torch.equal(r["rotation_generated"].to(r["rotation_used"].dtype), r["rotation_used"]):
            raise ValueError("block rotation differs from the actual source position embedder")
        head_dim = self.dit.transformer_blocks[self.block_index].attention_head_dim
        if r["rotation_used"].shape != (1, 1, total, head_dim // 2, 2, 2):
            raise ValueError("rotation table does not cover every full-context attention head pair")
        for stream, original in (("img", img), ("txt", txt)):
            for suffix in ("_out", "_attn_out", "_mlp_in", "_mlp_out"):
                if r[stream + suffix].shape != original.shape:
                    raise ValueError("captured stream lost context rows or width")
            if r[stream + "_modulation"].shape != (1, 6 * original.shape[-1]):
                raise ValueError("unexpected source modulation geometry")
        if r["temb"].shape != (1, img.shape[-1]):
            raise ValueError("unexpected source time embedding geometry")
        return r, {"fixture": self.fixture, "text_mask_present": self.mask_present,
                   "row_order": "text_then_image", "complete_context": True,
                   "text_tokens": txt.shape[1], "image_tokens": img.shape[1], "capability_claim": "none"}


def capture_forward(dit, block_index, x, sigma, context, context_mask, *, fixture=False):
    if dit.training:
        raise ValueError("source capture requires eval mode")
    if sigma.dtype != torch.float32 or sigma.shape != (1,) or not bool(torch.isfinite(sigma).all()) or not 0 <= float(sigma[0]) <= 1:
        raise ValueError("capture_forward requires one actual finite FP32 sigma in [0,1]")
    with torch.inference_mode(), QimgBlockTap(dit, block_index, fixture=fixture) as tap:
        prediction = dit(x, sigma, context, attention_mask=context_mask, transformer_options={})
        tap.put("prediction", prediction)
        # Prediction is retained separately; finish verifies exactly the block taps.
        prediction = tap.records.pop("prediction")
        if not fixture and prediction.shape != x.shape:
            raise ValueError("parent prediction does not match its complete model input shape")
        records, meta = tap.finish()
        records["prediction"] = prediction
        records["sigma"] = sigma.detach().clone().float().cpu()
        records["model_input"] = x.detach().clone().cpu()
        nt, ni = meta["text_tokens"], meta["image_tokens"]
        h, w = x.shape[-2] // 2, x.shape[-1] // 2
        if ni != h * w or x.shape[-3] != 1:
            raise ValueError("captured image token count is not the complete plain 2x2 latent patch grid")
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        patch_indices = torch.stack((torch.zeros(h * w, dtype=torch.int64), rows.flatten(), cols.flatten()), 1)
        records["image_patch_indices"] = patch_indices
        records["joint_row_kind"] = torch.tensor([0] * nt + [1] * ni, dtype=torch.int8)
        if not fixture:
            want_img = patch_indices.float()
            want_img[:, 1] -= h // 2
            want_img[:, 2] -= w // 2
            start = max(h // 2, w // 2)
            want_txt = torch.arange(start, start + nt).float().view(-1, 1).expand(-1, 3)
            if not torch.equal(records["source_ids"], torch.cat((want_txt, want_img)).unsqueeze(0)):
                raise ValueError("actual source IDs differ from the plain image/text row-alignment contract")
    return records, meta


def verify_source_code(root):
    actual = {name: file_sha(root / name) for name in SOURCE_FILES}
    if actual != SOURCE_FILES:
        raise ValueError("Comfy source does not match the reviewed Qwen-Image contract")
    return actual


def output_size_estimate(preflight, entries):
    # Pinned source coefficients are FP8. Allow FP32 captures, both rotation
    # tables, all full-context taps, inputs/prediction, and per-forward overhead.
    total = preflight["block_parameters"] + preflight["retained_global_time_parameters"] + (1 << 20)
    for entry in entries:
        tokens = entry["token_count"]
        one = (24 * tokens * preflight["width"] + 16 * tokens * preflight["head_dim"] +
               8 * math.prod(entry["latent_shape"]) + (1 << 20))
        total += entry["forwards"] * one
    return total


def load_parent(weights, comfy_root, device):
    sys.path.insert(0, str(comfy_root))
    import comfy.options
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = device == "cpu"
    import comfy.model_management as management
    import comfy.sd
    if management.in_training:
        raise ValueError("production capture requires the source inference kernel path")
    actual_device = management.get_torch_device()
    if str(actual_device) != device and not (device == "cuda" and str(actual_device) == "cuda:0"):
        raise ValueError(f"Comfy selected {actual_device}, not requested {device}")
    patcher = comfy.sd.load_diffusion_model(str(weights), model_options={"dtype": torch.float8_e4m3fn,
                                          "fp8_optimizations": False}, disable_dynamic=True)
    # memory_required is EXTRA inference scratch, not the weight size. Comfy
    # already budgets model_memory_required * 1.1 separately. Keep its normal
    # inference reserve and verify residency below, without double-counting the
    # 20 GB parent in the requested free-memory target.
    management.load_models_gpu([patcher], force_full_load=True)
    model = patcher.model
    dit = model.diffusion_model.eval()
    off_device = [name for name, p in dit.named_parameters() if p.device != actual_device]
    if off_device:
        raise ValueError(f"source is not fully resident on the requested device: {off_device[:3]}")
    compute_dtype = getattr(model.model_config, "manual_cast_dtype", None) or model.get_dtype()
    if compute_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(f"unsupported source compute dtype: {compute_dtype}")
    dtype_numel = Counter()
    for parameter in dit.parameters():
        dtype_numel[str(parameter.dtype)] += parameter.numel()
    return patcher, dit, compute_dtype, {"device": str(actual_device), "torch": torch.__version__,
        "weight_dtype_requested": "torch.float8_e4m3fn", "fp8_optimizations": False,
        "dynamic_patcher": False, "full_residency_verified": True,
        "loader_memory_policy": "Comfy default inference reserve; model bytes counted by Comfy once",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "parent_class": f"{type(model).__module__}.{type(model).__name__}",
        "manual_cast_dtype": str(compute_dtype), "storage_dtype_numel": dict(dtype_numel),
        "storage_dtype_tensor_counts": dict(Counter(str(p.dtype) for p in dit.parameters())),
        "kernel_policy": "existing pinned Comfy inference attention/RoPE; no patches or custom operations requested"}


def write_tensors(path, tensors):
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, str(path))
    return {"file": path.name, "file_sha256": file_sha(path),
            "tensor_sha256_schema": "torch-dtype-name + compact JSON shape + raw contiguous bytes; v1",
            "tensor_sha256": {name: tensor_sha(tensor) for name, tensor in tensors.items()},
            "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
            "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="identity-linked capture_inputs.json")
    parser.add_argument("--out", type=Path, required=True, help="new directory; no overwrite option")
    parser.add_argument("--comfy-root", type=Path, default=Path("/opt/ComfyUI"))
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda:0")
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-forwards", type=int, default=32)
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 2048 or not 1 <= args.max_forwards <= 32:
        raise ValueError("at most 2048 full-context tokens and 32 forwards are supported")
    if args.out.exists():
        raise ValueError("output already exists; historical captures are never overwritten")
    entries, input_receipt = load_inputs(args.inputs, max_tokens=args.max_tokens, max_forwards=args.max_forwards)
    preflight = inspect_weights(args.weights, args.block)
    if preflight["source_identity"]["header_sha256"] != PARENT_HEADER_SHA256:
        raise ValueError("source parent header does not match the inspected FP8 parent")
    estimate = output_size_estimate(preflight, entries)
    if args.validate_inputs_only:
        print(json.dumps({"validated": True, "inputs": input_receipt, "source": preflight,
                          "estimated_output_upper_bound_bytes": estimate,
                          "source_code_verified": False,
                          "capture_launched": False}, sort_keys=True, indent=2))
        return
    probe = args.out.resolve().parent
    while not probe.exists():
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    if free_bytes < estimate + (4 << 30):
        raise ValueError("capture output estimate would leave less than 4 GiB free; no cleanup is attempted")
    source_code = verify_source_code(args.comfy_root)
    # Do not let Comfy parse this program's unrelated arguments.
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        patcher, dit, dtype, runtime = load_parent(args.weights, args.comfy_root, args.device)
    finally:
        sys.argv = saved_argv
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": CAPTURE_SCHEMA, "fixture": False, "status": "incomplete",
               "block": args.block, "source": preflight, "source_code_sha256": source_code,
               "inputs": input_receipt, "runtime": runtime, "captures": [],
               "estimated_output_upper_bound_bytes": estimate, "free_bytes_before": free_bytes,
               "capability_claim": "none; authentic source capture is not a fold or all-modalities proof"}
    try:
        names, _ = read_header(args.weights)
        selected = [name for name in names if name.startswith(f"transformer_blocks.{args.block}.") or
                    name.startswith("time_text_embed.")]
        with safe_open(str(args.weights), framework="pt", device="cpu") as handle:
            frozen = {name: handle.get_tensor(name) for name in selected}
        runtime_state = dit.state_dict()
        for name, tensor in frozen.items():
            actual = runtime_state[name].detach().cpu().contiguous()
            if actual.dtype != tensor.dtype or not torch.equal(actual.view(torch.uint8), tensor.contiguous().view(torch.uint8)):
                raise ValueError(f"loaded source tensor differs from stored FP8 source bytes: {name}")
        runtime["selected_block_and_time_match_source_bytes"] = True
        receipt["frozen_source"] = write_tensors(args.out / "source_block_and_time.safetensors", frozen)
        del frozen, runtime_state
        for item in entries:
            if file_sha(item["tensor_path"]) != item["entry"]["tensors"]["sha256"]:
                raise ValueError("input tensor file changed after initial validation")
            state = load_file(str(item["tensor_path"]), device="cpu")
            for index in range(item["forwards"]):
                x = state["model_inputs"][index:index + 1].to(args.device, dtype)
                sigma = state["sigma"][index:index + 1].to(args.device)
                context = state["context"].to(args.device, dtype)
                # Source _forward converts this integer 0/1 mask to additive
                # activation-dtype mask. Avoid bool subtraction in source code.
                mask = state["context_mask"].to(args.device, torch.int64)
                tensors, meta = capture_forward(dit, args.block, x, sigma, context, mask)
                tensors["context_token_ids"] = state["context_token_ids"]
                tensors["context_valid"] = state["context_mask"]
                name = f"{item['entry']['id']}_sigma{index:02d}.safetensors"
                record = write_tensors(args.out / name, tensors)
                receipt["captures"].append({**record, **meta, "sample_id": item["entry"]["id"],
                    "split": item["entry"]["split"], "image_sha256": item["entry"]["image"]["sha256"],
                    "prompt_sha256": item["entry"]["prompt_sha256"], "sigma": float(sigma[0]),
                    "model_input_cast_max_abs": float((x.float().cpu() - state["model_inputs"][index:index + 1].float()).abs().max()),
                    "context_cast_max_abs": float((context.float().cpu() - state["context"].float()).abs().max()),
                    "source_ids_schema": "qimg_text_first_centered_3d_v1"})
                print(json.dumps({"captured": name, "full_context_tokens": meta["image_tokens"] + meta["text_tokens"]}), flush=True)
        _, after = read_header(args.weights)
        if after != preflight["source_identity"]:
            raise ValueError("source parent header/size changed during capture")
        receipt["source_header_stable_after_capture"] = True
        receipt["status"] = "complete"
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["seconds"] = time.time() - started
        try:
            import resource
            receipt["peak_cpu_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:
            receipt["peak_cpu_rss_bytes"] = None
        if args.device.startswith("cuda"):
            receipt["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        (args.out / "capture_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    # Keep patcher alive until all forwards and source receipts are complete.
    del patcher


if __name__ == "__main__":
    main()
