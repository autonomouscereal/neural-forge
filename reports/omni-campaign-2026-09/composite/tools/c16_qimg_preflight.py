#!/usr/bin/env python3
"""Read-only Qwen-Image source geometry audit; never emits a folded container.

Reads the safetensors header, not the 20 GB tensor bodies. Header identity is not
full-file identity, numerical validation, captured behavior or media capability.
The report is printed to stdout so callers need not retain every invocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


SOURCE_CODE_SHA256 = "14c805af8da13d31094c2c704e413cc282ef43f4afe17f77de9070bdca301f28"
DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8}
MAX_HEADER_BYTES = 64 << 20


def read_header(path: Path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError("invalid or excessively large safetensors header")
        raw = handle.read(length)
    body = json.loads(raw)
    tensors = {name: value for name, value in body.items() if name != "__metadata__"}
    spans = []
    for name, value in tensors.items():
        shape, dtype, offsets = value.get("shape"), value.get("dtype"), value.get("data_offsets")
        if (not isinstance(shape, list) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
                or dtype not in DTYPE_BYTES or not isinstance(offsets, list) or len(offsets) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) for n in offsets)):
            raise ValueError(f"invalid tensor header: {name}")
        begin, end = offsets
        if begin < 0 or end - begin != math.prod(shape) * DTYPE_BYTES[dtype] or end > size - 8 - length:
            raise ValueError(f"invalid tensor byte range: {name}")
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError("tensor ranges overlap or leave an unaccounted gap")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("file has unaccounted tensor bytes")
    return tensors, {"path": str(path.resolve()), "file_bytes": size,
                     "header_bytes": length, "header_sha256": hashlib.sha256(raw).hexdigest(),
                     "full_file_sha256": None, "scope": "header only; tensor bodies not read"}


def biased_gelu_mlp_parameters(width: int, inner: int) -> int:
    return 2 * width * inner + inner + width


def maximum_fourfold_keep(width: int, parent_inner: int) -> int:
    if width < 1 or parent_inner < 1:
        raise ValueError("MLP dimensions must be positive")
    parent = biased_gelu_mlp_parameters(width, parent_inner)
    return max(0, (parent - 4 * width) // (4 * (2 * width + 1)))


def inspect_weights(path: Path, block_index: int = 30):
    tensors, identity = read_header(path)
    indices = sorted({int(k.split(".")[1]) for k in tensors if k.startswith("transformer_blocks.")})
    if not indices or indices != list(range(len(indices))) or block_index not in indices:
        raise ValueError("source blocks must be contiguous and include the selected block")
    prefix = f"transformer_blocks.{block_index}."
    block = {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}
    required = {}
    try:
        width = tensors["img_in.weight"]["shape"][0]
        head_dim = block["attn.norm_q.weight"]["shape"][0]
        inner_img = block["img_mlp.net.0.proj.weight"]["shape"][0]
        inner_txt = block["txt_mlp.net.0.proj.weight"]["shape"][0]
    except (KeyError, IndexError) as exc:
        raise ValueError("not a Qwen-Image dual-stream source header") from exc
    if min(width, head_dim, inner_img, inner_txt) < 1 or width % head_dim:
        raise ValueError("invalid Qwen-Image width/head geometry")
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out"):
        required[f"attn.{name}.weight"] = [width, width]
        required[f"attn.{name}.bias"] = [width]
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        required[f"attn.{name}.weight"] = [head_dim]
    for stream, inner in (("img", inner_img), ("txt", inner_txt)):
        required.update({f"{stream}_mlp.net.0.proj.weight": [inner, width],
                         f"{stream}_mlp.net.0.proj.bias": [inner],
                         f"{stream}_mlp.net.2.weight": [width, inner],
                         f"{stream}_mlp.net.2.bias": [width],
                         f"{stream}_mod.1.weight": [6 * width, width],
                         f"{stream}_mod.1.bias": [6 * width]})
    if {k: v["shape"] for k, v in block.items()} != required:
        raise ValueError("source block does not match the complete dual-stream geometry")
    time = {name: value for name, value in tensors.items() if name.startswith("time_text_embed.")}
    expected_time = {f"time_text_embed.timestep_embedder.linear_{i}.{part}": shape
                     for i, inputs in ((1, 256), (2, width))
                     for part, shape in (("weight", [width, inputs]), ("bias", [width]))}
    if {k: v["shape"] for k, v in time.items()} != expected_time:
        raise ValueError("unsupported source time path (including additional conditioning)")
    mlps = {}
    for stream, inner in (("image", inner_img), ("text", inner_txt)):
        keep = maximum_fourfold_keep(width, inner)
        parent = biased_gelu_mlp_parameters(width, inner)
        child = biased_gelu_mlp_parameters(width, keep)
        mlps[stream] = {"parent_inner": inner, "parent_parameters": parent,
                       "maximum_legal_keep": keep, "child_parameters_at_maximum": child,
                       "fourfold_count_verified": 4 * child <= parent,
                       "quarter_inner_parameters": biased_gelu_mlp_parameters(width, inner // 4)}
    return {"schema": "neural_forge.qimg_source_preflight.v1", "source_identity": identity,
            "reference_source_code_sha256": SOURCE_CODE_SHA256,
            "source_code_scope": "pinned reviewed contract; caller source file not verified by this header audit",
            "block": block_index, "depth": len(indices), "width": width, "heads": width // head_dim,
            "head_dim": head_dim, "tensor_count": len(tensors),
            "source_dtypes": dict(sorted(Counter(v["dtype"] for v in tensors.values()).items())),
            "total_header_parameters": sum(math.prod(v["shape"]) for v in tensors.values()),
            "block_parameters": sum(math.prod(v["shape"]) for v in block.values()),
            "retained_global_time_parameters": sum(math.prod(v["shape"]) for v in time.values()),
            "mlps": mlps, "fold_ready": False, "capability_claim": "none",
            "missing_evidence": ["real image/prompt identities and image-disjoint fit/holdout split",
                "parent block image AND text input/output plus per-stream MLP taps",
                "actual sigma, temb, both source modulations and dtype/kernel policy",
                "text mask, explicit source text/image coordinates and actual rotation table",
                "measured folded MLP weights with grouped holdout scores and immutable provenance",
                "paired host/source charts on aligned image AND text rows",
                "native dual-stream runtime and canonical container (generic adaln6 is insufficient)",
                "real QIMG VAE/raw-patch interface absorption; no companion decoder substitution"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--block", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(inspect_weights(args.weights, args.block), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
