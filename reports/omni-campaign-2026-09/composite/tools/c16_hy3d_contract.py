#!/usr/bin/env python3
"""Read-only native Hunyuan3D 2.1 source/header and capture-contract validator.

No model inference, pickle loading, checkpoint-body hashing, conversion, or install.
A valid capture contract is NOT a fidelity/quality verdict or an installed child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

SCHEMA = "neural-forge.hy3d-native-inputs.v1"
CAPTURE_SCHEMA = "neural-forge.hy3d-native-capture.v1"
MODEL_PATH = "/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors"
SOURCE_ROOT = "/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1"
PARENT_BYTES = 7365943290
PARENT_HEADER_SHA = "0b44c4852a3bd475352accc9db5bc78e678bd4b36c9a313b6592a7e4aabc908a"
CAPTURE_DIMS = {"tokens": 4096, "hidden": 2048, "channels": 64, "context_tokens": 1370,
                "context_width": 1024, "surface_points": 81920}
BASE = "hy3dshape/hy3dshape/models/"
# Read from this installed source, 2026-09-07; a changed source requires re-audit.
SOURCE_SHA = {
    "configs/dit_config_2_1.yaml": "a04b6a62f7084e1490eaba825355d340983bcbc88809bb36cad37b35979845aa",
    BASE + "denoisers/hunyuandit.py": "27a74c5ba210468847aaf348bfd31ac2781dce094960ede0b2050d94233c531e",
    BASE + "denoisers/moe_layers.py": "a8d756a5fbe6941ad0dd68e4d1b0ea4fc6a84418f4c2e2b0a28cb4f2144ec240",
    BASE + "autoencoders/model.py": "b81a4586cc1148def7fe157c18071fb7dc50753cfbf1930932bbcbecbcb35829",
    BASE + "autoencoders/attention_blocks.py": "4be3dccc02da6f773c979175280ee0618ce5cc58917016a16c554745fc051537",
    "hy3dshape/hy3dshape/pipelines.py": "cf56918d62d5bee11a80641638dbc590bd541a1264c384e302a87d68a4cd285b",
    BASE + "conditioner.py": "c735ef7abbdd0fad089e684a237ef30a6cb7616c2faa49a30571d3640c2865c9",
    "hy3dshape/hy3dshape/surface_loaders.py": "c4a821fffd9ea9f121ac07a034faaa9e52350859cc667db955ae91c17ee1c001",
}
ELEMENT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
                 "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))


def stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def bounded_bytes(path, limit=2 << 20):
    path = Path(path)
    before = stamp(path)
    require(before[2] <= limit, f"metadata too large: {path}")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    require(len(raw) <= limit and before == stamp(path), f"metadata changed/grew: {path}")
    return raw


def tensor_header(path):
    """Validate every offset/byte count but never read tensor bodies."""
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "checkpoint must be a regular file")
    before = stamp(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8 and not prefix.startswith(b"PK"),
                "safetensors required; ZIP/pickle ShapeVAE is not a header source")
        size = struct.unpack("<Q", prefix)[0]
        require(2 <= size <= 16 << 20 and size + 8 <= before[2], "invalid safetensors header length")
        raw = handle.read(size)
    header = unique_json(raw)
    require(isinstance(header, dict), "header must be an object")
    spans, tensors = [], {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        require(isinstance(item, dict), f"bad tensor metadata: {name}")
        shape, dtype, offsets = item.get("shape"), item.get("dtype"), item.get("data_offsets")
        require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape), f"bad shape: {name}")
        require(dtype in ELEMENT_BYTES, f"unsupported dtype: {name}")
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(v) is int for v in offsets), f"bad offsets: {name}")
        start, end = offsets
        require(0 <= start <= end <= before[2] - size - 8 and
                end - start == math.prod(shape) * ELEMENT_BYTES[dtype], f"bad tensor byte range: {name}")
        spans.append((start, end, name))
        tensors[name] = {"shape": shape, "dtype": dtype, "parameters": math.prod(shape)}
    cursor = 0
    for start, end, name in sorted(spans):
        require(start == cursor, f"overlapping/gapped tensor payload: {name}")
        cursor = end
    require(cursor == before[2] - size - 8, "unaccounted checkpoint payload")
    require(before == stamp(path), "checkpoint changed while reading headers")
    return {"path": str(path), "bytes": before[2], "mtime_ns": before[3],
            "header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": size,
            "full_weights_hashed": False, "tensors": tensors}


def geometry(tensors):
    """Refuse a Flux fixture, missing experts, or decoder-only VAE as this parent."""
    def shape(name, expected):
        require(tensors.get(name, {}).get("shape") == list(expected), f"native tensor shape mismatch: {name}")
        require(tensors[name]["dtype"] == "F16", f"audited parent must retain F16 storage: {name}")
    shape("model.x_embedder.weight", (2048, 64))
    shape("model.x_embedder.bias", (2048,))
    shape("model.t_embedder.mlp.0.weight", (8192, 2048))
    shape("model.t_embedder.mlp.0.bias", (8192,))
    shape("model.t_embedder.mlp.2.weight", (2048, 8192))
    shape("model.t_embedder.mlp.2.bias", (2048,))
    layers = {int(m[1]) for name in tensors if (m := re.match(r"model\.blocks\.(\d+)\.", name))}
    require(layers == set(range(21)), "native DiT must contain all 21 blocks")
    require(not any(name.startswith(("model.pos_embed", "model.pooler", "model.extra_embedder"))
                    or ".default_modulation." in name for name in tensors),
            "unreviewed positional/pooling/adaLN variant")
    for layer in range(21):
        p = f"model.blocks.{layer}."
        for norm in ("norm1", "norm2", "norm3"):
            for suffix in ("weight", "bias"):
                shape(p + norm + "." + suffix, (2048,))
        for attn, context in (("attn1", 2048), ("attn2", 1024)):
            for proj, width in (("to_q", 2048), ("to_k", context), ("to_v", context)):
                shape(p + attn + "." + proj + ".weight", (2048, width))
                require(p + attn + "." + proj + ".bias" not in tensors, "native qkv_bias=false")
            for norm in ("q_norm", "k_norm"):
                shape(p + attn + "." + norm + ".weight", (128,))
            shape(p + attn + ".out_proj.weight", (2048, 2048))
            shape(p + attn + ".out_proj.bias", (2048,))
        if layer > 10:
            shape(p + "skip_linear.weight", (2048, 4096))
            shape(p + "skip_linear.bias", (2048,))
            shape(p + "skip_norm.weight", (2048,))
            shape(p + "skip_norm.bias", (2048,))
        else:
            require(not any(n.startswith(p + "skip_") for n in tensors), "unexpected early skip connection")
        if layer >= 15:
            shape(p + "moe.gate.weight", (8, 2048))
            experts = {int(m[1]) for n in tensors if (m := re.match(re.escape(p) + r"moe\.experts\.(\d+)\.", n))}
            require(experts == set(range(8)), "native MoE requires all 8 experts")
            prefixes = [p + f"moe.experts.{i}.net." for i in range(8)] + [p + "moe.shared_experts.net."]
            require(not any(n.startswith(p + "mlp.") for n in tensors), "MoE block replaced by dense fixture")
            names = ("0.proj", "2")
        else:
            require(not any(n.startswith(p + "moe.") for n in tensors), "unexpected early MoE block")
            prefixes, names = [p + "mlp."], ("fc1", "fc2")
        for prefix in prefixes:
            for name, dims in zip(names, ((8192, 2048), (2048, 8192))):
                shape(prefix + name + ".weight", dims)
                shape(prefix + name + ".bias", (dims[0],))
    shape("conditioner.main_image_encoder.model.embeddings.position_embeddings", (1, 1370, 1024))
    for name, dims in {"encoder.input_proj.weight": (1024, 55), "pre_kl.weight": (128, 1024),
                       "post_kl.weight": (1024, 64), "encoder.cross_attn.attn.c_q.weight": (1024, 1024),
                       "encoder.cross_attn.attn.c_kv.weight": (2048, 1024),
                       "geo_decoder.query_proj.weight": (1024, 51),
                       "geo_decoder.output_proj.weight": (1, 1024)}.items():
        shape("vae." + name, dims)
    for prefix, count in (("vae.encoder.self_attn.resblocks.", 8), ("vae.transformer.resblocks.", 16)):
        found = {int(m[1]) for n in tensors if (m := re.match(re.escape(prefix) + r"(\d+)\.", n))}
        require(found == set(range(count)), "ShapeVAE encoder/decoder layer inventory incomplete")
    return {"depth": 21, "width": 2048, "latent_channels": 64, "latent_tokens": 4096,
            "context_tokens": 1370, "context_width": 1024, "heads": 16, "head_dim": 128,
            "moe_layers": list(range(15, 21)), "experts": 8, "top_k": 2,
            "attention": "native self-attention followed by image-context cross-attention",
            "time": "native GELU timestep embedding prepended as token 0; no adaln6",
            "positions": "no DiT RoPE/position embedding; preserve sampled ShapeVAE query anchors and token order",
            "skip_layers": list(range(11, 21)), "vae_prefix": "vae.", "surface_shape": [81920, 7]}


def inspect_source(parent, source_root):
    header = tensor_header(parent)
    require(header["bytes"] == PARENT_BYTES and header["header_sha256"] == PARENT_HEADER_SHA,
            "not the audited parent header; changed source requires re-audit")
    code = {}
    root = Path(source_root).resolve(strict=True)
    for name, expected in SOURCE_SHA.items():
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root), f"source escapes root: {name}")
        code[name] = hashlib.sha256(bounded_bytes(path)).hexdigest()
        require(code[name] == expected, f"source revision changed; re-audit required: {name}")
    geom = geometry(header["tensors"])
    counts = {prefix: sum(t["parameters"] for n, t in header["tensors"].items() if n.startswith(prefix))
              for prefix in ("model.", "vae.", "conditioner.")}
    identity = {"checkpoint_header_sha256": header["header_sha256"], "checkpoint_bytes": header["bytes"],
                "source_sha256": code}
    return {"schema": SCHEMA, "source_identity": identity,
            "checkpoint": {k: v for k, v in header.items() if k != "tensors"}, "geometry": geom,
            "component_parameters": counts, "validated": "native header/config/code identity only",
            "full_weights_hashed": False, "capture_present": False, "folded_child_proven": False}


def validate_capture(directory, source):
    """Validate sealed native evidence shape/value/ownership, not model quality."""
    import torch
    from safetensors import safe_open
    directory = Path(directory).resolve(strict=True)
    body = unique_json(bounded_bytes(directory / "capture.json"))
    require(isinstance(body, dict), "capture metadata must be an object")
    require(body.get("schema") == CAPTURE_SCHEMA, "wrong native capture schema")
    require(body.get("source_identity") == source["source_identity"], "capture belongs to another source")
    name = body.get("file")
    require(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")
            and not any(c in name for c in ("/", "\\", ":")), "unsafe capture filename")
    path = directory / name
    require(not path.is_symlink(), "capture file must not be a symlink")
    header = tensor_header(path)
    require(header["bytes"] <= 2 << 30, "capture exceeds bounded 2 GiB validation budget; shard by object")
    digest = hashlib.sha256()
    before = stamp(path)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    require(digest.hexdigest() == body.get("file_sha256"), "capture tensor file digest mismatch")
    samples = body.get("samples")
    require(isinstance(samples, list) and len(samples) >= 2, "capture needs fit and held objects")
    object_split = {}
    for row in samples:
        require(isinstance(row, dict) and row.get("split") in ("fit", "held"), "invalid object split")
        for key in ("object_sha256", "conditioning_sha256"):
            require(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key]), "missing object/condition identity")
        require(object_split.setdefault(row["object_sha256"], row["split"]) == row["split"], "object leaked across fit/held")
        require(type(row.get("seed")) is int and row["seed"] >= 0, "capture needs explicit sampling seed")
    require(set(object_split.values()) == {"fit", "held"}, "missing fit/held object partition")
    require(body.get("producer") == {"native_forward": True, "strict_weights": True,
                                      "eval": True, "modified_attention": False}, "native producer contract required")
    b, keys = len(samples), set(header["tensors"])
    n, h, ch = (CAPTURE_DIMS[k] for k in ("tokens", "hidden", "channels"))
    context_n, context_h, pc = (CAPTURE_DIMS[k] for k in ("context_tokens", "context_width", "surface_points"))
    kind = body.get("kind")
    if kind == "dit_block":
        layer = body.get("block_index")
        require(type(layer) is int and 0 <= layer < 21, "invalid native block index")
        require(body.get("attention") == "self_then_image_cross" and
                body.get("time_path") == "sigma_prepended_token" and
                body.get("position_path") == "sampled_anchors_no_dit_rope", "wrong native attention/time/position contract")
        shapes = {key: [b, n + 1, h] for key in ("block_input", "block_output", "mlp_input", "mlp_output")}
        shapes.update({"context": [b, context_n, context_h], "sigma": [b], "timestep": [b],
                       "time_token": [b, 1, h], "latent_xyz": [b, n, 3],
                       "surface": [b, pc, 7], "query_source_index": [b, n]})
        if layer > 10:
            shapes["skip_value"] = [b, n + 1, h]
        if layer >= 15:
            shapes.update({"router_logits": [b, n + 1, 8], "router_topk_ids": [b, n + 1, 2],
                           "router_topk_weights": [b, n + 1, 2], "shared_output": [b, n + 1, h]})
    elif kind == "shape_vae_pairs":
        require(body.get("posterior") == "mode" and body.get("anchor_source") == "native_encoder_query_pc"
                and body.get("feature_kind") == "anchor_local_sdf", "native posterior/anchor/SDF pairing required")
        q = body.get("feature_samples")
        require(type(q) is int and q > 0, "invalid local SDF sample count")
        shapes = {"surface": [b, pc, 7], "query_source_index": [b, n],
                  "anchor_xyz": [b, n, 3], "posterior_mode": [b, n, ch],
                  "feature_offsets": [q, 3], "feature_queries": [b, n, q, 3],
                  "feature_sdf": [b, n, q], "parent_roundtrip_sdf": [b, n, q]}
    else:
        raise ValueError("unknown native capture kind")
    require(keys == set(shapes), "capture has missing/extra native tensors")
    for name, shape in shapes.items():
        require(header["tensors"][name]["shape"] == shape, f"capture shape mismatch: {name}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in sorted(keys):
            value = handle.get_tensor(name)
            if name in ("query_source_index", "router_topk_ids"):
                require(value.dtype == torch.int64, f"index must be I64: {name}")
            else:
                require(value.is_floating_point(), f"capture must be floating point: {name}")
                for part in value.split(1):
                    require(bool(torch.isfinite(part).all()), f"nonfinite native capture: {name}")
            del value
        surface, ids = handle.get_tensor("surface"), handle.get_tensor("query_source_index")
        anchors = handle.get_tensor("latent_xyz" if kind == "dit_block" else "anchor_xyz")
        require(bool(((ids >= 0) & (ids < pc)).all()), "anchor source index out of range")
        require(torch.equal(surface[..., :3].gather(1, ids[..., None].expand(-1, -1, 3)), anchors),
                "anchors are not the captured source surface queries")
        if kind == "dit_block":
            sigma, timestep = handle.get_tensor("sigma"), handle.get_tensor("timestep")
            require(bool(((sigma >= 0) & (sigma <= 1)).all()) and torch.equal(sigma, timestep),
                    "native flow timestep must equal captured sigma in [0,1], not sigma*1000")
            if layer >= 15:
                logits = handle.get_tensor("router_logits")
                ids, weights = handle.get_tensor("router_topk_ids"), handle.get_tensor("router_topk_weights")
                require(bool(((ids >= 0) & (ids < 8)).all()) and bool((ids[..., 0] != ids[..., 1]).all()), "bad top-k ids")
                require(logits.dtype == weights.dtype, "native router scores must retain their source dtype")
                probs = logits.softmax(-1)
                require(torch.equal(weights, probs.gather(-1, ids)),
                        "native unnormalized top-k masses changed")
                require(torch.equal(weights.sort(-1).values, probs.topk(2, dim=-1).values.sort(-1).values),
                        "selected experts are not native top-k")
        else:
            offsets, queries = handle.get_tensor("feature_offsets"), handle.get_tensor("feature_queries")
            for index in range(b):
                require(torch.allclose(queries[index].float(), anchors[index, :, None].float() + offsets.float(),
                                       rtol=0, atol=1e-6), "local SDF features lost anchor ownership")
    require(before == stamp(path), "capture changed during validation")
    return {"kind": kind, "samples": b, "objects": len(object_split), "file_sha256": digest.hexdigest(),
            "contract_valid": True, "native_forward_independently_replayed": False,
            "folded_child_proven": False, "tensor_shapes": shapes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", default=MODEL_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--capture")
    args = parser.parse_args(argv)
    report = inspect_source(args.parent_checkpoint, args.source_root)
    if args.capture:
        report["capture"] = validate_capture(args.capture, report)
        report["capture_present"] = True
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
