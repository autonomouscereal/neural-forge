#!/usr/bin/env python3
"""Read-only S3 input/runtime check. Never loads parent tensors or starts a job.

The header/stat inventory is explicitly NOT a replacement for qf_stream27's
full source digest and its two subsequent integrity checks. Output is stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

GIB = 1 << 30
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EVIDENCE_SHA = "8a3422d89d245b79a152ffb1de44d077d39ec5574dc063bf4e8482a5107b4870"
MAX_SMALL_FILE = 64 << 20


def small_bytes(path: Path, limit: int = MAX_SMALL_FILE) -> bytes:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or unexpectedly large metadata file: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"metadata grew while reading: {path}")
    return data


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def safe_child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name or ":" in name or name in (".", ".."):
        raise ValueError(f"unsafe checkpoint filename: {name!r}")
    path = root / name
    if path.is_symlink() or path.resolve(strict=True).parent != root.resolve(strict=True):
        raise ValueError(f"escaping or symlinked checkpoint file: {path}")
    return path


def header_inventory(parent: Path) -> dict:
    """Read only the safetensors header, not a shard's tensor body."""
    index_path = parent / "model.safetensors.index.json"
    index_raw = small_bytes(index_path)
    mapping = json.loads(index_raw).get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("missing/empty indexed weight_map")
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ValueError("invalid tensor name")
    if any(not isinstance(name, str) or not name for name in mapping.values()):
        raise ValueError("invalid shard name")
    if (parent / "model.safetensors").exists() and set(mapping.values()) != {"model.safetensors"}:
        raise ValueError("ambiguous single-file and indexed checkpoint")
    names = set(mapping.values())
    files, tensors = [], {}
    for name in sorted(names):
        path = safe_child(parent, name)
        before = path.stat()
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors prefix: {path}")
            length = int.from_bytes(prefix, "little")
            if not 2 <= length <= 16 << 20 or length + 8 > before.st_size:
                raise ValueError(f"invalid safetensors header length: {path}")
            raw = handle.read(length)
        header = json.loads(raw)
        for tensor_name, shard in mapping.items():
            if shard != name:
                continue
            row = header.get(tensor_name)
            if not isinstance(row, dict):
                raise ValueError(f"indexed tensor missing from header: {tensor_name}")
            shape = row.get("shape")
            offsets = row.get("data_offsets")
            if not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
                raise ValueError(f"invalid tensor shape: {tensor_name}")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets) or not 0 <= offsets[0] <= offsets[1] <= before.st_size - length - 8:
                raise ValueError(f"invalid tensor byte offsets: {tensor_name}")
            tensors[tensor_name] = {"parameters": math.prod(shape), "bytes": offsets[1] - offsets[0]}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise ValueError(f"checkpoint stat changed while inspecting header: {path}")
        files.append({"name": name, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                      "header_sha256": hashlib.sha256(raw).hexdigest()})
    mtp = {n: r for n, r in tensors.items() if n.startswith("mtp.")}
    return {"method": "indexed headers and file stats ONLY; NOT a full weight digest",
            "index_sha256": hashlib.sha256(index_raw).hexdigest(), "files": files,
            "file_bytes": sum(r["bytes"] for r in files), "indexed_tensors": len(tensors),
            "stock_model_parameters": sum(r["parameters"] for n, r in tensors.items() if n not in mtp),
            "mtp_parameters_excluded_from_fold_ratio": sum(r["parameters"] for r in mtp.values())}


def inspect_evidence(parent: Path, evidence: Path, *, expected=EVIDENCE_SHA) -> dict:
    raw = small_bytes(evidence)
    body = json.loads(raw)
    declared = body.pop("manifest_sha256", None)
    digest = hashlib.sha256(canonical(body)).hexdigest()
    if declared != digest or digest != expected:
        raise ValueError("evidence seal or pinned evidence identity mismatch")
    if body.get("schema") != "neural-forge-omni-evidence-v1":
        raise ValueError("unexpected evidence schema")
    tokenizer = hashlib.sha256(small_bytes(parent / "tokenizer.json")).hexdigest()
    if tokenizer != body.get("tokenizer_sha256"):
        raise ValueError("evidence and parent tokenizer mismatch")
    rows, length = body.get("rows"), body.get("sequence_length")
    if not isinstance(rows, list) or not rows or type(length) is not int or length < 1:
        raise ValueError("invalid evidence rows or sequence length")
    ids_seen, splits = set(), {"fit": 0, "held": 0}
    config = json.loads(small_bytes(parent / "config.json"))
    vocab = config.get("text_config", config)["vocab_size"]
    for row in rows:
        index, ids, split = row.get("batch_index"), row.get("ids"), row.get("split")
        if type(index) is not int or index in ids_seen or split not in splits:
            raise ValueError("duplicate/invalid batch identity or split")
        if not isinstance(ids, list) or len(ids) != length or any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError("invalid evidence token ids or row length")
        ids_seen.add(index)
        splits[split] += 1
    if splits["fit"] < 3 or splits["held"] < 1:
        raise ValueError("missing fit/check/held partition")
    return {"manifest_sha256": digest, "file_sha256": hashlib.sha256(raw).hexdigest(),
            "tokenizer_sha256": tokenizer, "batches": len(rows), "sequence_length": length,
            "fit_rows": splits["fit"] * length, "held_rows": splits["held"] * length,
            "internal_check_rows": (splits["fit"] // 3) * length}


def memory_components(config: dict, evidence: dict, stock_parameters: int) -> dict:
    tc = config.get("text_config", config)
    h, w, layers = (int(tc[k]) for k in ("hidden_size", "intermediate_size", "num_hidden_layers"))
    k, group, tokens = 4352, 3, evidence["batches"] * evidence["sequence_length"]
    if (h, w, layers, int(tc["vocab_size"])) != (5120, 17408, 64, 248320):
        raise ValueError("not the audited S3 parent geometry")
    removed = layers * 3 * h * (w - k)
    components = {
        "parent_bf16_stock_weights": stock_parameters * 2,
        "child_bf16_stock_weights": (stock_parameters - removed) * 2,
        "child_fp32_reload_weights": (stock_parameters - removed) * 4,
        "host_final_folded_tensor_cache": layers * 3 * h * k * 2,
        "host_residual_cache": tokens * h * 2,
        "host_group_fp64_parent_mlp_weights": group * 3 * h * w * 8,
        "host_group_inner_check_gram_and_rhs": group * 2 * (w * w + w * h) * 8,
        "host_group_bf16_xy_stash": group * 2 * tokens * h * 2,
        "gpu_bf16_untied_embedding_and_head": 2 * int(tc["vocab_size"]) * h * 2,
        "gpu_one_fp32_full_attention_score_matrix": int(tc["num_attention_heads"]) * evidence["sequence_length"] ** 2 * 4,
    }
    return {"parameters_removed_by_C2": removed, "component_gib": {n: round(b / GIB, 3) for n, b in components.items()},
            "limits": "components, NOT peak RSS/VRAM proof; allocator/workspace/GEMM/attention temporaries add memory",
            "operational_budget": "reserve 128 GiB host for S3; require at least 128 GiB available at admission; 22 CPU threads; validate GPU4 full-sequence mechanics before a long fold"}


def runtime_check() -> dict:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
    versions = {n: importlib.metadata.version(n) for n in ("torch", "numpy", "safetensors", "transformers")}
    expected = {"torch": "2.11.0+cu128", "numpy": "2.4.4", "safetensors": "0.8.0", "transformers": "5.12.1"}
    if versions != expected or torch.get_num_threads() != 22:
        raise ValueError(f"runtime/thread pin mismatch: {versions}, threads={torch.get_num_threads()}")
    return {"versions": versions, "threads": torch.get_num_threads(), "cuda_initialized": torch.cuda.is_initialized(),
            "classes": [Qwen3_5ForConditionalGeneration.__name__, Qwen3_5ForCausalLM.__name__]}


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    campaign = here.parents[1]
    files = [here / "qf_stream27.py", campaign / "distill/tps_probe.py"]
    files += [campaign / "qwen-fold" / f"qf_{name}.py" for name in
              ("arch", "attn", "build", "corpus", "excise", "export", "gdn", "harness", "mlp", "solve", "termgain", "vocab")]
    return {str(path.relative_to(campaign)): hashlib.sha256(small_bytes(path)).hexdigest() for path in files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--check-memory", action="store_true", help="refuse if host MemAvailable is below the 128 GiB S3 admission budget")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError("child output already exists; S3 never overwrites a child")
    config = json.loads(small_bytes(args.parent / "config.json"))
    evidence = inspect_evidence(args.parent, args.evidence)
    if evidence["tokenizer_sha256"] != TOKENIZER_SHA:
        raise ValueError("not the pinned S3 tokenizer")
    headers = header_inventory(args.parent)
    available = None
    if args.check_memory:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(values["MemAvailable"].split()[0]) * 1024
        if available < 128 * GIB:
            raise ValueError("less than the 128 GiB S3 host admission budget is available")
    report = {"schema": "qf.s3-read-only-preflight.v1", "parent": str(args.parent), "evidence": evidence,
              "checkpoint": headers, "memory": memory_components(config, evidence, headers["stock_model_parameters"]),
              "cpu_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "full_parent_weight_hash_performed": False,
              "host_available_gib": None if available is None else round(available / GIB, 3),
              "source_sha256": source_hashes(),
              "runtime": runtime_check() if args.check_runtime else "not requested"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
