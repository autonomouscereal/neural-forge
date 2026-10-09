#!/usr/bin/env python3
"""Frozen standalone plain-QIMG backbone continuation, never a child claim.

The existing reviewed native two-stream block/time implementation is reused.
This module adds strict complete-stack accounting and the source global path.
It never imports Comfy, reads a teacher checkpoint, calls a service, or trains.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn
import torch.nn.functional as F

from chimera_omni.qimg import QimgNativeSpec, SCHEMA as COMPONENT_SCHEMA, SOURCE_FILES, PARENT_HEADER_SHA256, sha256_string
from chimera_omni.importing.qimg import component_from_state, tensor_sha256
from chimera_omni.modeling.qimg import (
    QimgNativeComponent, _FrozenStorage, _Linear, _RMSNorm, _Block, _Time,
    qimg_rotation,
)

SCHEMA = "neural_forge.qimg_native_stack.v1"
DEPTH = 60
FOLD_STACK_SCHEMA = "neural_forge.qimg_folded_stack.v1"
# Names/shapes recovered from the actual 234040-byte source header, not a
# count-based inference. The raw header SHA is PARENT_HEADER_SHA256. Global
# equations/order were read from the 26510-byte model.py pinned in SOURCE_FILES.
GLOBAL_NAMES = (
    "img_in.bias", "img_in.weight", "norm_out.linear.bias", "norm_out.linear.weight",
    "proj_out.bias", "proj_out.weight", "txt_in.bias", "txt_in.weight", "txt_norm.weight",
)
CONSUMER_RUNTIME_SHA256 = {
    "src/chimera_omni/qimg.py": "f0ea5485d169db2e2a5f9d9b284d5ffee769c969ab96211160c05230571d568d",
    "src/chimera_omni/importing/qimg.py": "24af87e9ea18d53fca579b4cde01badc30e6c081fd345533b5494724ddbe6c3c",
    "src/chimera_omni/importing/qimg_fold.py": "b1f261c443c8671a0bf05fb73b6c9001dfa75d907982ea77b04d22d34f237c1e",
    "src/chimera_omni/modeling/qimg.py": "cccf19f9f525ec7b4059cb9199c07e28cd6c07a4bbdf47360e39cbd0d45ada35",
}
MAX_JSON_BYTES = 16 << 20


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.absolute() != path.resolve():
        raise ValueError("canonical regular nonsymlink artifact required")
    return path


def _json(path, sha256):
    path = _regular(path)
    sha256_string(sha256)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("stack JSON exceeds metadata bound")
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("stack JSON identity/size mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate stack JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def _finite(name, value):
    if not isinstance(value, torch.Tensor) or value.is_meta or value.layout != torch.strided:
        raise ValueError("actual dense tensor required: " + name)
    if value.is_floating_point():
        for chunk in value.detach().reshape(-1).split(1 << 20):
            if not bool(torch.isfinite(chunk.float()).all()):
                raise ValueError("nonfinite native tensor: " + name)


def verify_consumer_source():
    """Bind reused executable modules to the accepted V1 source bytes."""
    import chimera_omni.qimg as contract
    root = Path(contract.__file__).resolve().parents[2]
    actual = {name: file_sha256(_regular(root / name)) for name in CONSUMER_RUNTIME_SHA256}
    if actual != CONSUMER_RUNTIME_SHA256:
        raise ValueError("native stack requires the exact accepted consumer executable source")
    return actual


def _geometry(specs, fixture):
    if type(fixture) is not bool or len(specs) != DEPTH:
        raise ValueError("explicit fixture flag and exactly 60 native blocks required")
    if [s.source_layer for s in specs] != list(range(DEPTH)):
        raise ValueError("complete ordered source block coverage 0..59 required")
    first = specs[0]
    for spec in specs:
        if (not isinstance(spec, QimgNativeSpec) or spec.fixture is not fixture
                or (spec.width, spec.head_dim, spec.axes, spec.storage) !=
                   (first.width, first.head_dim, first.axes, first.storage)
                or (not fixture and spec.storage != "folded_fp8_bf16")):
            raise ValueError("native stack block geometry/storage differs")
    return first


def component_from_verified_fold(fold_path, fold_sha256, data, *, parent_header_sha256,
                                 policy, policy_reference, recipe_id, fixture_axes=(2, 2, 2)):
    """Explicit capture-V4 adapter; never fabricates legacy capture receipts.

    ``data`` must come from stack_fold.load_block_inputs. Candidate/source and
    selected-channel identity are checked by the actual unchanged joint reader.
    The existing component reader then checks all 36 owned tensor bytes again.
    """
    import c16_qimg_fold as fold
    import c16_qimg_joint_replay as joint
    if Path(fold_path).name != "fold_receipt.json":
        raise ValueError("canonical published fold_receipt.json required; provisional is not completion")
    body = _json(fold_path, fold_sha256)
    if (body.get("schema") != fold.OWNER_FOLD_SCHEMA
            or body.get("owner_policy") != fold.current_owner_policy()
            or body.get("structural_artifact_valid") is not True
            or body.get("candidate_written") is not True
            or body.get("selection_tensors_written") is not True
            or body.get("artifact_state") != "unevaluated_development"
            or any(body.get(k) is not False for k in ("accepted", "acceptance_enabled", "capability_accepted"))):
        raise ValueError("published current owner development fold required")
    def capture_refs(rows):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise ValueError("explicit bounded stack capture references required")
        result = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise ValueError("exact stack capture reference required")
            sha256_string(row["sha256"])
            result.append((str(Path(row["path"]).resolve()), row["sha256"]))
        return sorted(result)
    if capture_refs(body.get("capture_receipts")) != capture_refs(data["capture_receipts"]):
        raise ValueError("fold and supplied stack capture path/SHA identities differ")
    if (body.get("policy") != policy or body.get("policy_reference") != policy_reference
            or body.get("recipe_id") != recipe_id or body.get("law", {}).get("selection") != recipe_id
            or any(body["streams"][s]["keep"] != policy["streams"][s]["keep"] for s in ("img", "txt"))):
        raise ValueError("fold policy/recipe identity differs from the verified stack recipe")
    candidate, candidate_evidence = joint.load_candidate(Path(fold_path), fold_sha256, data)
    head = data["frozen"][f"transformer_blocks.{data['block']}.attn.norm_q.weight"].numel()
    fixture = data["fixture"]
    spec = QimgNativeSpec(data["block"], body["streams"]["img"]["keep"],
        body["streams"]["txt"]["keep"], "folded_fp8_bf16", data["width"], head,
        fixture_axes if fixture else (16, 56, 56), fixture)
    manifest = {"schema": COMPONENT_SCHEMA, "installed": False, "spec": spec.to_dict(),
        "weights": {"file": body["output"]["file"], "sha256": body["output"]["file_sha256"],
                    "tensor_sha256": {k: tensor_sha256(v) for k, v in candidate.items()}},
        "source": {"parent_header_sha256": parent_header_sha256,
                   "frozen_file_sha256": data["frozen_file_sha256"], "code_sha256": dict(SOURCE_FILES),
                   "capture_receipt_sha256": [r["sha256"] for r in data["capture_receipts"]],
                   "fold_receipt_sha256": fold_sha256,
                   "tensor_sha256": {k: tensor_sha256(v) for k, v in data["frozen"].items()}}}
    component = component_from_state(candidate, manifest, allow_fixture=fixture)
    component.verification = MappingProxyType({**dict(component.verification),
        "stack_capture_v4_adapter": True, "candidate_source_selection_verified": True,
        "candidate_tensor_sha256": dict(manifest["weights"]["tensor_sha256"]),
        "fold_receipt_sha256_verified": True, "fold_receipt": candidate_evidence,
        "component_manifest_origin": "constructed only from verified source/candidate; no legacy receipt synthesized"})
    return component


def global_shapes(width=3072, context_width=3584):
    return {"img_in.weight": (width, 64), "img_in.bias": (width,),
        "txt_norm.weight": (context_width,), "txt_in.weight": (width, context_width),
        "txt_in.bias": (width,), "norm_out.linear.weight": (2 * width, width),
        "norm_out.linear.bias": (2 * width,), "proj_out.weight": (64, width), "proj_out.bias": (64,)}


def source_global_shapes(spec, context_width):
    result = global_shapes(spec.width, context_width)
    result.update({k: shape for k, (shape, _) in spec.tensors().items() if k.startswith("time_text_embed.")})
    return result


def plain_source_ids(image_shape, text_tokens, *, device):
    """Exact plain source process_img positions, including BF16 linspace.

    Large/asymmetric grids can round BF16 positions. F32 arange would silently
    change that native operation boundary, so this follows the actual source.
    """
    h, w = image_shape[0] // 2, image_shape[1] // 2
    image = torch.zeros((1, h, w, 3), dtype=torch.float32, device=device)
    image[..., 1] = image[..., 1] + torch.linspace(0, h - 1, h, dtype=torch.bfloat16, device=device)[None, :, None] - h // 2
    image[..., 2] = image[..., 2] + torch.linspace(0, w - 1, w, dtype=torch.bfloat16, device=device)[None, None, :] - w // 2
    start = round(max(w // 2, h // 2))
    text = torch.arange(start, start + text_tokens, device=device).reshape(1, -1, 1).repeat(1, 1, 3)
    return torch.cat((text, image.reshape(1, h * w, 3)), dim=1)


def patchify(model_input):
    """Source process_img: [B,C,T,H/2,2,W/2,2] -> [B,T*H/2*W/2,C*4]."""
    b, c, t, h, w = model_input.shape
    return model_input.view(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t * (h // 2) * (w // 2), c * 4)


def unpatchify(patches, image_shape):
    h, w = image_shape
    return patches.view(1, 1, h // 2, w // 2, 16, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(1, 16, 1, h, w)


class _OutputNorm(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.linear = _Linear(state, "norm_out.linear")

    def forward(self, image, temb):
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=1)
        return torch.addcmul(shift[:, None, :], F.layer_norm(image, (image.shape[-1],), eps=1e-6), (1 + scale)[:, None, :])


class QimgNativeStack(_FrozenStorage):
    """Owned complete two-stream source backbone; frozen standalone inference.

    Build through native_stack_from_components/load_qimg_native_stack so every
    source/refit/global tensor is checked before a usable object is returned.
    The source namespace is retained physically: 60*32 + shared4 + global9.
    """
    def __init__(self, global_state, context_width, verification):
        super().__init__()
        self.specs = ()
        self.spec = None
        self.context_width = context_width
        self.verification = MappingProxyType(copy.deepcopy(verification))
        self.img_in = _Linear(global_state, "img_in")
        self.txt_norm = _RMSNorm(global_state["txt_norm.weight"])
        self.txt_in = _Linear(global_state, "txt_in")
        self.time_text_embed = _Time(global_state)
        self.norm_out = _OutputNorm(global_state)
        self.proj_out = _Linear(global_state, "proj_out")
        self.transformer_blocks = nn.ModuleDict()
        self.eval()

    def _install(self, component):
        spec = component.spec
        state = component.state_dict()
        if set(state) != set(spec.tensors()):
            raise ValueError("complete native component state required at stack installation")
        frozen_hashes = component.verification["source_lineage"]["tensor_sha256"]
        candidate_hashes = component.verification.get("candidate_tensor_sha256")
        if spec.storage == "folded_fp8_bf16" and (not isinstance(candidate_hashes, dict) or set(candidate_hashes) != set(state)):
            raise ValueError("folded component requires complete verified candidate tensor hash pins")
        for name, (shape, dtype) in spec.tensors().items():
            value = state[name]
            _finite(name, value)
            if value.device.type != "cpu" or tuple(value.shape) != shape or str(value.dtype) != dtype:
                raise ValueError("component shape/storage changed before stack installation")
            digest = tensor_sha256(value)
            changed = spec.storage == "folded_fp8_bf16" and (".img_mlp." in name or ".txt_mlp." in name)
            if (not changed and digest != frozen_hashes[name]) or (candidate_hashes is not None and digest != candidate_hashes[name]):
                raise ValueError("component source/candidate bytes changed before stack installation")
        # Reuse the actual accepted _Block implementation and clone its verified
        # selected FP8/BF16 storage. The caller cannot mutate this stack by alias.
        self.transformer_blocks[str(spec.source_layer)] = _Block(state, f"transformer_blocks.{spec.source_layer}", spec)
        self.eval()

    @torch.no_grad()
    def forward(self, model_input, context, sigma, *, text_mask=None, return_taps=False):
        device = next(self.parameters()).device
        if self.training or torch.is_autocast_enabled(device.type):
            raise ValueError("native stack requires frozen inference without external autocast")
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("CPU/CUDA native numeric path required")
        if (not isinstance(model_input, torch.Tensor) or model_input.ndim != 5
                or tuple(model_input.shape[:3]) != (1, 16, 1)
                or min(model_input.shape[-2:]) < 2 or any(n % 2 for n in model_input.shape[-2:])
                or model_input.dtype != torch.bfloat16 or model_input.device != device):
            raise ValueError("complete even plain BF16 model input [1,16,1,H,W] required")
        if (not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[0] != 1
                or context.shape[1] < 1 or context.shape[2] != self.context_width
                or context.dtype != torch.bfloat16 or context.device != device):
            raise ValueError("complete native BF16 single-sample source text context required")
        if model_input.shape[-2] // 2 * (model_input.shape[-1] // 2) + context.shape[1] > 2048:
            raise ValueError("joint context exceeds the existing 2048-token source capture contract")
        if (not isinstance(sigma, torch.Tensor) or sigma.shape != (1,) or sigma.dtype != torch.float32
                or sigma.device != device or not bool(torch.isfinite(sigma).all())
                or bool(((sigma < 0) | (sigma > 1)).any())):
            raise ValueError("native F32 sigma in [0,1] required; no double source shift")
        if text_mask is not None:
            if (not isinstance(text_mask, torch.Tensor) or text_mask.device != device
                    or text_mask.dtype != torch.bfloat16 or text_mask.shape != context.shape[:2]
                    or bool(torch.isnan(text_mask).any()) or bool((text_mask > 0).any())
                    or not bool((text_mask > torch.finfo(torch.bfloat16).min).any())):
                raise ValueError("native nonpositive BF16 additive text mask with a valid row required")
        _finite("model_input", model_input); _finite("context", context)
        if list(self.transformer_blocks) != [str(i) for i in range(DEPTH)]:
            raise ValueError("native stack is missing complete ordered 0..59 modules")
        image_shape = tuple(model_input.shape[-2:])
        patches = patchify(model_input)
        image = self.img_in(patches)
        text = self.txt_in(self.txt_norm(context))
        temb = self.time_text_embed(sigma)
        ids = plain_source_ids(image_shape, context.shape[1], device=device)
        rotation = qimg_rotation(ids, self.spec.axes)
        pe = rotation.to(dtype=model_input.dtype).contiguous()
        global_taps = {"image_patches": patches, "img_in": image, "txt_in": text,
            "source_ids": ids, "time_recomputed": temb, "rotation_recomputed": rotation, "rotation_used": pe}
        for name, value in global_taps.items(): _finite(name, value)
        all_taps = {} if return_taps else None
        for index in range(DEPTH):
            taps = {}
            text, image = self.transformer_blocks[str(index)](image, text, temb, pe, text_mask, taps)
            for name, value in taps.items(): _finite(f"block{index}/{name}", value)
            if return_taps:
                all_taps[str(index)] = {**taps, "time_recomputed": temb, "rotation_recomputed": rotation}
        normalized = self.norm_out(image, temb)
        projected = self.proj_out(normalized)
        prediction = unpatchify(projected, image_shape)
        for name, value in (("norm_out", normalized), ("proj_out", projected), ("prediction", prediction)):
            _finite(name, value)
        if return_taps:
            return {"globals": {**global_taps, "norm_out": normalized, "proj_out": projected, "prediction": prediction},
                    "blocks": all_taps, "text_out": text, "image_out": image, "prediction": prediction}
        return prediction

    def expected_layout(self):
        result = {k: (s, "torch.float8_e4m3fn") for k, s in source_global_shapes(self.spec, self.context_width).items()}
        for spec in self.specs:
            result.update({k: value for k, value in spec.tensors().items() if k.startswith("transformer_blocks.")})
        return result

    def runtime_report(self):
        params = list(self.named_parameters())
        layout = self.expected_layout()
        expected = {"parameters": sum(math.prod(s) for s, _ in layout.values()),
            "storage_bytes": sum(math.prod(s) * (2 if d == "torch.bfloat16" else 1) for s, d in layout.values()),
            "tensor_count": len(layout)}
        actual = {"parameters": sum(v.numel() for _, v in params),
            "storage_bytes": sum(v.numel() * v.element_size() for _, v in params), "tensor_count": len(params)}
        return {"schema": SCHEMA, **actual, "expected_counts": expected, "source_block_count": DEPTH,
            "folded_mlp_count": 2 * DEPTH if self.spec.storage == "folded_fp8_bf16" else 0,
            "block_tensor_count": 32 * DEPTH, "shared_time_tensor_count": 4, "remaining_global_tensor_count": 9,
            "shared_time_instances": 1, "trainable_parameters": sum(v.numel() for _, v in params if v.requires_grad),
            "kind": "standalone_plain_qimg_native_backbone", "fixture": self.spec.fixture,
            "artifact_state": "standalone_unevaluated_development_backbone", "full_backbone_loaded": len(params) == 1933,
            "installed": False, "accepted": False, "capability_accepted": False, "child_accepted": False,
            "mlp_diagnostics": {str(s.source_layer): s.counts()["mlp_compression_x"] for s in self.specs},
            "verification": copy.deepcopy(dict(self.verification)),
            "claims": "plain source backbone only; no sampler/VAE, ImageEdit, paired host/source charts, Chimera child or intelligence claim"}


def native_stack_from_components(components, global_state, *, global_tensor_sha256,
                                 parent_header_sha256, fixture=False, verification=None):
    """Construct an owned native backbone from verified components and globals.

    In-memory callers do not obtain file/header authenticity merely by supplying
    hashes. The production file adapter records its separately verified evidence.
    Source-FP8 components are allowed only for an explicit tiny fixture oracle.
    """
    verify_consumer_source()
    if type(fixture) is not bool or not isinstance(global_state, dict) or not isinstance(global_tensor_sha256, dict):
        raise ValueError("explicit fixture flag and complete global state/hash dictionaries required")
    sha256_string(parent_header_sha256)
    if not fixture and parent_header_sha256 != PARENT_HEADER_SHA256:
        raise ValueError("wrong plain QIMG parent header")
    items = iter(components)
    first = next(items, None)
    if not isinstance(first, QimgNativeComponent):
        raise ValueError("verified native component required")
    context_width = global_state.get("txt_norm.weight", torch.empty(0)).numel()
    if (not fixture and context_width != 3584) or (fixture and not 1 <= context_width <= 128):
        raise ValueError("native source context width required")
    shapes = source_global_shapes(first.spec, context_width)
    if not isinstance(global_state, dict) or set(global_state) != set(shapes) or set(global_tensor_sha256) != set(shapes):
        raise ValueError("exact nine global and four shared-time tensors required")
    for name, shape in shapes.items():
        value = global_state[name]
        if value.device.type != "cpu" or value.shape != shape or value.dtype != torch.float8_e4m3fn:
            raise ValueError("global source shape/FP8/CPU mismatch: " + name)
        _finite(name, value); sha256_string(global_tensor_sha256[name])
        if tensor_sha256(value) != global_tensor_sha256[name]:
            raise ValueError("global source tensor hash mismatch: " + name)
    specs = []
    # Allocate globals once. Per-block construction remains sequential: callers
    # may yield one loaded component at a time without retaining all candidates.
    result = None
    evidence = []
    for component in _prepend(first, items):
        if not isinstance(component, QimgNativeComponent) or component.spec.source_layer != len(specs):
            raise ValueError("complete ordered verified native source layers 0..59 required")
        specs.append(component.spec)
        if len(specs) > DEPTH or component.spec.fixture is not fixture:
            raise ValueError("native stack fixture/depth mismatch")
        if result is None:
            # Geometry is finalized before return; partial construction is
            # never exposed as a complete/loadable backbone.
            result = QimgNativeStack(global_state, context_width, {})
        state = component.state_dict()
        for name in shapes:
            if name.startswith("time_text_embed.") and tensor_sha256(state[name]) != global_tensor_sha256[name]:
                raise ValueError("shared time source coefficients differ across native blocks")
        result._install(component)
        evidence.append(copy.deepcopy(dict(component.verification)))
    spec = _geometry(specs, fixture)
    result.specs, result.spec = tuple(specs), spec
    result.verification = MappingProxyType({**copy.deepcopy(verification or {}),
        "consumer_runtime_sha256": dict(CONSUMER_RUNTIME_SHA256), "parent_header_sha256": parent_header_sha256,
        "global_tensor_sha256": dict(global_tensor_sha256), "block_verification": evidence,
        "in_memory_hash_verification_is_file_authenticity": False})
    result.eval()
    report = result.runtime_report()
    if any(report[k] != report["expected_counts"][k] for k in ("parameters", "storage_bytes", "tensor_count")) or report["tensor_count"] != 1933:
        raise ValueError("complete native runtime accounting differs from exact source layout")
    result.verification = MappingProxyType({**dict(result.verification),
        "runtime_tensor_sha256": {k: tensor_sha256(v) for k, v in result.state_dict().items()}})
    return result


def _prepend(first, rest):
    yield first
    yield from rest


def _reference(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("exact externally pinned path/SHA256 reference required")
    sha256_string(value["sha256"])
    return _regular(value["path"]), value["sha256"]


def load_qimg_native_stack(fold_stack_path, expected_sha256, *, allow_fixture=False,
                           fixture_axes=(2, 2, 2), device="cpu"):
    """Read a counted full60 folded stack and its authentic capture V4 source.

    Source coefficients and selected/frozen/candidate bytes are checked once per
    block. The folding driver already verified full per-block activation taps;
    this loader reads their bound metadata, not the 282GB estimated activation
    expansion again. Its report records that limit explicitly.
    """
    body = _json(fold_stack_path, expected_sha256)
    fixture = body.get("fixture")
    coverage = body.get("coverage", {})
    if (body.get("schema") != FOLD_STACK_SCHEMA or body.get("status") != "complete"
            or type(fixture) is not bool or type(allow_fixture) is not bool or (fixture and not allow_fixture)
            or body.get("functional_validation_complete") is not True or "error" in body
            or not isinstance(body.get("standalone_state"), dict)
            or body.get("accepted") is not False or body.get("capability_accepted") is not False
            or coverage.get("required_blocks") != list(range(DEPTH))
            or coverage.get("requested_blocks") != list(range(DEPTH))
            or coverage.get("completed_blocks") != list(range(DEPTH))
            or coverage.get("missing_blocks") != [] or coverage.get("full_stack_complete") is not True):
        raise ValueError("complete explicit native folded stack 0..59 required; fixture is not production evidence")
    import c16_qimg_stack_fold as stack_fold
    import c16_qimg_fold as fold
    import c16_qimg_development_refold as development
    parent = body.get("parent_header_sha256")
    if body.get("owner_policy") != fold.current_owner_policy():
        raise ValueError("stack owner diagnostic-only policy identity differs")
    if body.get("recipe_id") != development.RECIPE_ID:
        raise ValueError("stack recipe differs from the actual corrected fitter")
    sha256_string(parent)
    if not fixture and parent != PARENT_HEADER_SHA256:
        raise ValueError("wrong plain QIMG stack source header")
    blocks = body.get("blocks")
    if (not isinstance(blocks, list) or len(blocks) != DEPTH
            or any(not isinstance(r, dict) or set(r) != {"block", "fold_receipt"} for r in blocks)
            or [r["block"] for r in blocks] != list(range(DEPTH))
            or any(type(r["block"]) is not int for r in blocks)):
        raise ValueError("each exact native source block must be referenced once")
    refs = [_reference(r["fold_receipt"]) for r in blocks]
    if len({sha for _, sha in refs}) != DEPTH:
        raise ValueError("duplicate fold receipt identity cannot count as distinct source blocks")
    capture_path, capture_sha = _reference(body.get("stack_capture"))
    policy_path, policy_sha = _reference(body.get("policy"))
    policy = fold.load_policy(policy_path, policy_sha)
    policy_reference = {"path": str(policy_path), "sha256": policy_sha}
    index = stack_fold.CaptureIndex(capture_path, capture_sha, fixture=fixture)
    if index.request["blocks"] != list(range(DEPTH)):
        raise ValueError("capture source coverage must include exact 0..59")
    if index.request["source"]["header_sha256"] != parent or body.get("global_source") != index.body["global_source"]:
        raise ValueError("folded stack and actual source/global capture identities differ")
    globals_ = index.global_state()
    record = index.body["global_source"]
    def components():
        for layer, (path, sha) in enumerate(refs):
            data = stack_fold.load_block_inputs(capture_path, capture_sha, layer, fixture=fixture,
                                                index=index, load_forwards=False)
            yield component_from_verified_fold(path, sha, data, parent_header_sha256=parent,
                                               policy=policy, policy_reference=policy_reference,
                                               recipe_id=body["recipe_id"],
                                               fixture_axes=fixture_axes)
    model = native_stack_from_components(components(), globals_, global_tensor_sha256=record["tensor_sha256"],
        parent_header_sha256=parent, fixture=fixture,
        verification={"fold_stack_sha256_verified": True, "fold_stack": {"path": str(Path(fold_stack_path).resolve()), "sha256": expected_sha256},
            "stack_capture": {"path": str(capture_path), "sha256": capture_sha},
            "global_source_record": copy.deepcopy(record), "source_globals_preserved": True,
            "policy": {"path": str(policy_path), "sha256": policy_sha}, "recipe_id": body.get("recipe_id"),
            "all_120_folded_mlps_loaded": True, "capture_forward_payloads_loaded_by_native_reader": False,
            "lineage_scope": "actual complete selected/frozen source coefficients and captured metadata; parent full-file digest not newly established"})
    index.revalidate()
    _json(fold_stack_path, expected_sha256)
    _json(policy_path, policy_sha)
    for path, sha in refs: _json(path, sha)
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("CPU/CUDA native target required")
    readback_consolidated_state(model, body["standalone_state"], root=Path(fold_stack_path).resolve().parent)
    return model.to(device=target)


def readback_consolidated_state(model, record, *, root):
    """Load an actual 1933-key saved state against already verified components.

    This is a physical strict readback/dispatch path, not a receipt-only count.
    All source names, shapes, FP8/BF16 islands, complete finite bytes and file
    identity are checked before assigning them to the real native modules.
    """
    from safetensors.torch import load_file
    if not isinstance(model, QimgNativeStack) or not isinstance(record, dict):
        raise ValueError("verified full native stack and consolidated state record required")
    required = {"file_sha256", "tensor_sha256", "shapes", "dtypes", "tensor_sha256_schema"}
    if not required.issubset(record) or record["tensor_sha256_schema"] != "torch-dtype-name + compact JSON shape + raw contiguous bytes; v1":
        raise ValueError("complete native consolidated tensor identity record required")
    root = Path(root).resolve()
    if "path" in record:
        path = _regular(record["path"])
    elif "file" in record and Path(record["file"]).name == record["file"]:
        path = _regular(root / record["file"])
    else:
        raise ValueError("contained consolidated state path required")
    if not path.is_relative_to(root):
        raise ValueError("consolidated state escapes its folded stack directory")
    sha256_string(record["file_sha256"])
    layout = model.expected_layout()
    hashes = dict(model.verification["runtime_tensor_sha256"])
    if (any(set(record[k]) != set(layout) for k in ("tensor_sha256", "shapes", "dtypes"))
            or record["tensor_sha256"] != hashes):
        raise ValueError("consolidated checkpoint differs from complete verified native components")
    for name, (shape, dtype) in layout.items():
        if record["shapes"][name] != list(shape) or record["dtypes"][name] != dtype:
            raise ValueError("consolidated state shape/storage differs from native source layout")
    bound = model.runtime_report()["storage_bytes"] + (8 << 20)
    if path.stat().st_size > bound or file_sha256(path) != record["file_sha256"]:
        raise ValueError("consolidated checkpoint file size/SHA256 mismatch")
    values = load_file(str(path), device="cpu")
    if set(values) != set(layout):
        raise ValueError("actual checkpoint must contain exactly 1933 native source keys")
    for name, (shape, dtype) in layout.items():
        value = values[name]
        _finite(name, value)
        if tuple(value.shape) != shape or str(value.dtype) != dtype or tensor_sha256(value) != hashes[name]:
            raise ValueError("consolidated physical tensor readback mismatch: " + name)
    if file_sha256(path) != record["file_sha256"]:
        raise ValueError("consolidated state changed during readback")
    # Every destination already has the verified FP8/BF16 dtype. Copy into its
    # owned allocation, avoiding aliases to a safetensors file-backed buffer.
    model.load_state_dict(values, strict=True, assign=False)
    model.eval()
    for name, value in model.state_dict().items():
        if tensor_sha256(value) != hashes[name]:
            raise ValueError("native module load altered consolidated bytes: " + name)
    if file_sha256(path) != record["file_sha256"]:
        raise ValueError("consolidated state changed during native module readback")
    model.verification = MappingProxyType({**dict(model.verification),
        "consolidated_state_sha256_verified": True, "consolidated_state_file": str(path),
        "consolidated_state_sha256": record["file_sha256"], "consolidated_tensor_count": len(values),
        "consolidated_state_loaded_into_actual_native_modules": True})
    return model
