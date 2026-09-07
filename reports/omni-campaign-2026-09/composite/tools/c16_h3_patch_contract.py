#!/usr/bin/env python3
"""Read-only H3 plain-T2VA native patch capture validation, not a child gate.

No inference, training, capture generation, install, or checkpoint-body hashing.
Only four small FP32 source projection tensors are read from the parent. Geometry
functions are executed verbatim from the hash-pinned native source (no Comfy shim).
The producer and real clip/text/codec evidence are intentionally still required.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import torch
import torch.nn.functional as F
from safetensors import safe_open

SCHEMA = "neural-forge.h3-native-patch-capture.v1"
SOURCE_SHA = "882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024"
PARENT_BYTES = 40225724176
PARENT_HEADER_SHA = "d55386f89bed7f9719ae315b4d70d104b9000e7a839d7702c82a97c1835324e6"
PROJECTIONS = {"video_patch_proj.weight": (5376, 96), "video_patch_proj.bias": (5376,),
               "audio_patch_proj.weight": (5376, 32), "audio_patch_proj.bias": (5376,)}
PROJECTION_SHA = {
    "video_patch_proj.weight": "26ff2714f21c25539a4f80040735274cc72e3c4a9da82769925df6a1102a301d",
    "video_patch_proj.bias": "e022b31f38e18f2ad4f0eaa99679d22b39ef70375ba16a47783a9f6ceed47cb1",
    "audio_patch_proj.weight": "902248bff928268be54c81aa1550cdb74af2f8a6ee49de1981dad9648d3acec3",
    "audio_patch_proj.bias": "db724caaa45a0a1a69786ab4cc2a0bb8f4c5e525f90e3b8cf64b65a97165e420",
}
GEOMETRY = {"patchify_video", "pack_audio", "_axis_from_sqrt_area", "_frame_grid",
            "_video_t_spans", "_video_t_grid", "_audio_grid", "_video_grid",
            "PackedLayout", "time_shift_sigma"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def stamp(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def raw_sha(tensor):
    """SHA256 of contiguous native tensor bytes only, with dtype/shape checked separately."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Pins:
    def __init__(self):
        self.seen = {}

    def file(self, ref, *, max_bytes=512 << 20):
        require(isinstance(ref, dict), "missing file reference")
        p = Path(ref.get("path", ""))
        expected = ref.get("sha256", "")
        require(p.is_absolute() and p.is_file(), "reference must name an existing absolute file")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", expected)), "missing file SHA256")
        before = stamp(p)
        require(before[2] <= max_bytes, "referenced file exceeds bounded read budget")
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        require(stamp(p) == before, f"file changed while hashing: {p.name}")
        require(digest.hexdigest() == expected, f"file hash mismatch: {p.name}")
        key = str(p.resolve())
        require(key not in self.seen or self.seen[key] == before, "previously pinned file changed")
        self.seen[key] = before
        return p

    def json(self, ref):
        p = self.file(ref)
        value = json.loads(p.read_text(encoding="utf-8"))
        self.stable()
        return value

    def stable(self):
        for path, before in self.seen.items():
            require(stamp(path) == before, "pinned input changed during validation")


def native_geometry(source):
    """Load only reviewed pure native geometry nodes; never import the Comfy runtime."""
    data = Path(source).read_bytes()
    require(hashlib.sha256(data).hexdigest() == SOURCE_SHA, "native H3 source identity mismatch")
    tree = ast.parse(data, filename=str(source))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in GEOMETRY]
    require({n.name for n in nodes} == GEOMETRY, "native geometry definitions incomplete")
    # These constants were read from the same pinned module. Do not confuse this
    # positional clock with the empirically measured VAE ownership groups.
    scope = {"torch": torch, "math": math, "FRAME_PER_TOKEN": (1, 4, 4, 4, 4), "FRAME_RESCALE": 5.0 / 3.0}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def inspect_parent(path, expected_projection_sha):
    """Pin header and four selected payloads; explicitly NOT a fresh whole-file SHA."""
    path = Path(path)
    require(path.is_absolute() and path.is_file(), "parent checkpoint must be an absolute file")
    before = stamp(path)
    require(before[2] == PARENT_BYTES, "unexpected H3 parent file size")
    with path.open("rb") as stream:
        n = struct.unpack("<Q", stream.read(8))[0]
        require(0 < n <= 16 << 20, "invalid safetensors header size")
        data = stream.read(n)
    require(len(data) == n and hashlib.sha256(data).hexdigest() == PARENT_HEADER_SHA,
            "H3 parent header identity mismatch")
    header = json.loads(data)
    require(expected_projection_sha == PROJECTION_SHA, "all four pinned source projection payload hashes required")
    projections = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name, shape in PROJECTIONS.items():
            meta = header.get(name, {})
            require(meta.get("dtype") == "F32" and tuple(meta.get("shape", ())) == shape,
                    f"native FP32 projection shape/dtype changed: {name}")
            t = handle.get_tensor(name)
            require(raw_sha(t) == expected_projection_sha[name], f"source projection identity mismatch: {name}")
            require(bool(torch.isfinite(t).all()), f"nonfinite projection: {name}")
            projections[name] = t
    require(stamp(path) == before, "parent changed during selected-tensor inspection")
    return projections, before


def validate_groups(probe, raw_frames, latent_frames):
    require(probe.get("frames") == raw_frames and probe.get("latent_frames") == latent_frames,
            "ownership probe does not cover this full clip")
    influence = torch.tensor(probe.get("influence", []), dtype=torch.float64)
    require(tuple(influence.shape) == (raw_frames, latent_frames), "full measured influence matrix required")
    require(bool(torch.isfinite(influence).all()) and bool((influence >= 0).all())
            and bool((influence.max(dim=1).values > 0).all()), "invalid or zero ownership evidence")
    owner = influence.argmax(dim=1).tolist()
    require(probe.get("owner_per_frame") == owner, "ownership is not measured argmax")
    groups = {str(i): [r for r, k in enumerate(owner) if k == i] for i in range(latent_frames)}
    require(probe.get("groups") == groups, "ownership groups omit/relabel raw or latent frames")
    # An empty argmax group does NOT authorize dropping that latent frame.
    return {"all_latent_frames_required": list(range(latent_frames)),
            "group_sizes": sorted({len(g) for g in groups.values()}),
            "ownership_means": "measured argmax influence, not exclusive causal support"}


def tensor(state, name, shape, dtype):
    t = state.get(name)
    require(isinstance(t, torch.Tensor) and tuple(t.shape) == tuple(shape) and t.dtype == dtype,
            f"{name} must preserve native {shape}/{dtype}")
    require(bool(torch.isfinite(t).all()), f"nonfinite {name}")
    return t


def exact(a, b, label):
    require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), label)


def validate_forward(meta, state, geometry, projections, probe):
    """Validate full native patch/position rows; no parent DiT or refiner replay."""
    require(meta.get("route") == "plain_t2va" and meta.get("minimax_payload") == {},
            "first contract supports plain T2VA only; keyframe/reference/tag routes need separate evidence")
    require(meta.get("transformer_options") == {}, "schedule overrides, replacement blocks and wrappers not supported")
    require(meta.get("stimuli") == "s1_media" and meta.get("deviations") == [],
            "legacy/procedural/deviated capture is not authentic S1 patch evidence")
    require(meta.get("compute_dtype") == "float32" and meta.get("allow_tf32") is False,
            "initial capture contract requires recorded FP32 compute without TF32")
    vshape, ashape = meta.get("video_shape"), meta.get("audio_shape")
    require(isinstance(vshape, list) and len(vshape) == 5 and all(type(x) is int and x > 0 for x in vshape), "video shape missing")
    require(isinstance(ashape, list) and len(ashape) == 4 and all(type(x) is int and x > 0 for x in ashape), "audio shape missing")
    require(vshape[:2] == [1, 24] and vshape[3] % 2 == vshape[4] % 2 == 0,
            "native full video must be [1,24,T,even H,even W]; silent padding/crop unsupported")
    require(ashape[:3] == [1, 32, 2], "native audio must preserve both stereo channels and width32")
    vt, vh, vw, at = vshape[2], vshape[3], vshape[4], ashape[-1]
    require(at == round(sum(geometry["_video_t_spans"](vt))), "audio span must match full native video clock")
    raw_frames = meta.get("raw_frames")
    require(type(raw_frames) is int and raw_frames > 0, "original raw frame count missing")
    groups = validate_groups(probe, raw_frames, vt)
    frame_ids = tensor(state, "video_latent_frame_ids", [vt], torch.int64)
    exact(frame_ids, torch.arange(vt), "every original latent frame including0 is required without relabeling")
    video = tensor(state, "video_x", vshape, torch.float32)
    audio = tensor(state, "audio_x", ashape, torch.float32)
    sv = tensor(state, "sigma_video", [1], torch.float32)
    sa = tensor(state, "sigma_audio", [1], torch.float32)
    timestep = tensor(state, "timestep", [1], torch.float32)
    require(bool(((sv > 0) & (sv <= 1)).all()), "invalid actual sigma")
    exact(sv, (timestep / 1000).clamp(min=1e-6), "actual timestep/video sigma mismatch")
    exact(sa, geometry["time_shift_sigma"](sv, 12.0, 3.0), "actual audio shifted sigma mismatch")
    for band, shape, sigma, x in (("video", vshape, sv, video), ("audio", ashape, sa, audio)):
        clean = tensor(state, band + "_clean", shape, torch.float32)
        noise = tensor(state, band + "_noise", shape, torch.float32)
        exact(x, (1 - sigma) * clean + sigma * noise, f"{band} noising changed or full clean latent absent")
    vr = geometry["patchify_video"](video)
    ar = geometry["pack_audio"](audio)
    scores = {}
    for band, rows, width in (("video", vr, 96), ("audio", ar, 32)):
        captured = tensor(state, band + "_patch_input", [rows.shape[0], width], torch.float32)
        exact(captured, rows, f"{band} native patch values/order/coverage changed")
        output = tensor(state, band + "_patch_output", [rows.shape[0], 5376], torch.float32)
        difference2, target2 = 0.0, 0.0
        for start in range(0, rows.shape[0], 512):
            expected = F.linear(rows[start:start + 512], projections[band + "_patch_proj.weight"], projections[band + "_patch_proj.bias"])
            reference = expected.double()
            difference2 += float((output[start:start + 512].double() - reference).square().sum())
            target2 += float(reference.square().sum())
        error = math.sqrt(difference2 / max(target2, 1e-60))
        require(error <= 1e-5, f"{band} source projection replay exceeds FP32 arithmetic tolerance")
        scores[band] = error
    text_len = meta.get("text_len")
    require(type(text_len) is int and text_len > 0, "real text context length missing")
    tensor(state, "text_states", [1, text_len, 5120], torch.float32)
    refined = tensor(state, "text_refined", [text_len, 5376], torch.float32)
    layout = geometry["PackedLayout"](text_len, vt, vh, vw, at)
    pos = tensor(state, "position_ids", [layout.seq_len, 3], torch.float64)
    exact(pos, layout.position_ids, "native text/audio/video THW positions changed")
    packed = tensor(state, "block0_input", [layout.seq_len, 5376], torch.float32)
    offset = 0
    for rows in (refined, state["audio_patch_output"], state["video_patch_output"]):
        exact(packed[offset:offset + rows.shape[0]], rows, "packed block0 boundary is not full [text|audio|video]")
        offset += rows.shape[0]
    return {"tokens": layout.seq_len, "video_patch_width": 96, "audio_patch_width": 32,
            "temporal_coverage": groups, "projection_rel_l2": scores,
            "text_refiner_forward_replayed": False, "parent_dit_forward_replayed": False}


def validate_capture(manifest_ref):
    pins = Pins()
    manifest = pins.json(manifest_ref)
    require(manifest.get("schema") == SCHEMA and manifest.get("fixture") is False, "production native capture schema required; no fixture CLI")
    source = pins.file(manifest.get("native_source"))
    geometry = native_geometry(source)
    pins.file(manifest.get("producer"))
    parent = Path(manifest.get("parent_checkpoint", ""))
    projections, parent_stamp = inspect_parent(parent, manifest.get("projection_raw_bytes_sha256", {}))
    entries = manifest.get("forwards", [])
    require(isinstance(entries, list) and entries, "no authentic forwards")
    budget = manifest.get("max_capture_bytes")
    require(type(budget) is int and 0 < budget <= 8 << 30, "explicit per-forward file budget <=8GiB required")
    ids, membership, split_seen, reports = set(), {}, set(), []
    for item in entries:
        identity = item.get("id")
        require(isinstance(identity, str) and identity and identity not in ids, "forward id missing/duplicated")
        ids.add(identity)
        split = item.get("split")
        require(split in ("fit", "held"), "explicit original fit/held split required")
        split_seen.add(split)
        for band in ("video", "audio"):
            ref = item.get(band + "_asset")
            pins.file(ref)
            group = item.get(band + "_parent_group")
            require(isinstance(group, str) and group, "original AV parent group is required")
            # Content hashes ALSO group shared clips even if labels were changed.
            for key in (("group", group), ("sha256", ref["sha256"])):
                require(key not in membership or membership[key] == split, "joint-attention AV parent crosses fit/held")
                membership[key] = split
        sft = pins.file(item.get("tensors"), max_bytes=budget)
        with safe_open(str(sft), framework="pt", device="cpu") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        probe = pins.json(item.get("temporal_probe"))
        for band in ("video", "audio", "text"):
            receipt = pins.json(item.get(band + "_encoder_receipt"))
            require(receipt.get("kind") == "h3_native_" + band + "_encode" and receipt.get("actual_parent_forward") is True,
                    f"actual {band} encoder receipt required (not Gaussian conditioning)")
            pins.file(receipt.get("source_code"))
            encoder_identity = pins.json(receipt.get("source_identity_receipt"))
            require(encoder_identity.get("schema") == "neural-forge.h3-encoder-identity.v1"
                    and encoder_identity.get("source_code_sha256") == receipt["source_code"]["sha256"]
                    and encoder_identity.get("actual_tensor_loader_verified") is True,
                    "typed, source-bound encoder weight-loading receipt required")
            require(Path(encoder_identity.get("checkpoint_path", "")).is_absolute()
                    and type(encoder_identity.get("checkpoint_bytes")) is int and encoder_identity["checkpoint_bytes"] > 0
                    and bool(re.fullmatch(r"[0-9a-f]{64}", encoder_identity.get("checkpoint_header_sha256", ""))),
                    "encoder checkpoint identity missing")
            expected_family = {"video": "minimax_h3_video_vae", "audio": "minimax_h3_audio_vae", "text": "qwen3_vl"}[band]
            require(encoder_identity.get("family") == expected_family, "wrong encoder family identity")
            if band == "text":
                require(receipt.get("encoder_family") == "qwen3_vl" and receipt.get("hidden_layer") == 50,
                        "H3 requires actual Qwen3-VL layer50 text states")
                pins.file(receipt.get("prompt"))
                key = "text_states"
            else:
                require(receipt.get("input_sha256") == item[band + "_asset"]["sha256"], "codec receipt/clip identity mismatch")
                require(receipt.get("normalization") == "native_h3_model_input", "VAE normalization provenance missing")
                key = band + "_clean"
            require(key in state and receipt.get("output_raw_bytes_sha256") == raw_sha(state[key]), f"{band} encoder output hash mismatch")
            if band != "text":
                content_key = (band + "_clean_latent", tuple(state[key].shape), receipt["output_raw_bytes_sha256"])
                require(content_key not in membership or membership[content_key] == split,
                        "identical encoded AV content crosses fit/held under different file/group names")
                membership[content_key] = split
            require(receipt.get("output_shape") == list(state[key].shape) and receipt.get("output_dtype") == "float32",
                    "encoder output shape/dtype identity missing")
            if band == "video":
                require(receipt.get("raw_frames") == item.get("raw_frames"), "original raw frame count differs from encoder receipt")
        require(probe.get("input_sha256") == item["video_asset"]["sha256"] and probe.get("encoder_receipt_sha256") == item["video_encoder_receipt"]["sha256"],
                "measured temporal probe is not tied to the actual clip and encoder")
        reports.append({"id": identity, "split": split, **validate_forward(item, state, geometry, projections, probe)})
    require(split_seen == {"fit", "held"}, "both clip-disjoint splits required")
    pins.stable()
    require(stamp(parent) == parent_stamp, "parent changed during validation")
    return {"schema": SCHEMA + ".validation", "contract_valid": True, "forwards": reports,
            "source_identity_scope": "pinned native code/header and four FP32 projection payloads; not fresh full-weight SHA",
            "producer_executed_here": False, "encoders_independently_replayed": False,
            "folded_child_proven": False, "production_install_eligible": False,
            "remaining": ["independent parent block/time/write replay", "paired host charts", "installed-child and behavioral gates"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 22, "threads must be1..22")
    torch.set_num_threads(args.threads)
    with torch.no_grad():
        print(json.dumps(validate_capture({"path": args.manifest, "sha256": args.manifest_sha256}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
