#!/usr/bin/env python3
"""Frozen native HY3D dense-block replay on sealed complete-context captures.

No fitting, training, scheduler generation, chart installation or quality gate.
Candidate residuals are reported only after every baseline boundary is exact.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

import c16_capture_hy3d as capture
from c16_hy3d_contract import (CAPTURE_DIMS, MODEL_PATH, SOURCE_ROOT, bounded_bytes,
    inspect_source, require, stamp, tensor_header, unique_json, validate_capture)

SCHEMA = "neural-forge.hy3d-dense-block-replay.v1"
FOLD_SCHEMA = "neural-forge.hy3d-frozen-dense-mlp.v1"
BOUNDARIES = ("time_token", "self_attention_write", "cross_attention_write", "mlp_input", "mlp_output", "block_output")


def sealed_path(directory, name):
    require(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")
            and not any(c in name for c in ("/", "\\", ":")), "unsafe evidence filename")
    path = directory / name
    require(not path.is_symlink() and path.is_file(), "regular sealed evidence file required")
    return path


def load_capture(run_directory, block_index, source):
    require(type(block_index) is int and 0 <= block_index < 15, "only native dense blocks 0..14; MoE refused")
    root = Path(run_directory).resolve(strict=True)
    run_path = root / "capture-run.json"
    run = unique_json(bounded_bytes(run_path))
    require(isinstance(run, dict) and run.get("schema") == capture.SCHEMA and
            run.get("source_identity") == source["source_identity"] and run.get("producer") == capture.PRODUCER,
            "matching sealed native capture run required")
    directory = root / f"block-{block_index:02d}"
    checked = validate_capture(directory, source)
    body = unique_json(bounded_bytes(directory / "capture.json"))
    require(body.get("kind") == "dit_block" and body.get("block_index") == block_index, "wrong captured native block")
    require(run.get("dit_capture") == checked, "capture-run/block validation identity disagreement")
    rows = run.get("samples")
    require(isinstance(rows, list) and 2 <= len(rows) <= 8 and len(rows) == len(body["samples"]), "bounded matching sample inventory required")
    path = sealed_path(directory, body["file"])
    tracked = {run_path: capture.digest_file(run_path), directory / "capture.json": capture.digest_file(directory / "capture.json"),
               path: body["file_sha256"]}
    geometry_splits, samples = {}, []
    identity_keys = ("object_sha256", "conditioning_sha256", "seed", "split", "geometry_identity")
    for index, (row, block_row) in enumerate(zip(rows, body["samples"])):
        require(isinstance(row, dict) and all(k in row and row[k] == block_row.get(k) for k in identity_keys), "sample order/object provenance mismatch")
        geometry = row.get("geometry_identity")
        require(isinstance(geometry, dict) and geometry.get("schema") == capture.GEOMETRY_SCHEMA and
                isinstance(geometry.get("sha256"), str), "canonical source geometry provenance required")
        require(geometry_splits.setdefault(geometry["sha256"], row["split"]) == row["split"], "canonical geometry crossed fit/held")
        sample_path = sealed_path(root, row.get("file"))
        require(tensor_header(sample_path)["bytes"] <= 512 << 20, "bounded per-object evidence required")
        require(capture.digest_file(sample_path) == row.get("file_sha256"), "sample evidence digest mismatch")
        tracked[sample_path] = row["file_sha256"]
        sample_json = root / f"sample-{index:04d}.json"
        require(unique_json(bounded_bytes(sample_json)) == row, "sample receipt disagrees with sealed run")
        tracked[sample_json] = capture.digest_file(sample_json)
        with safe_open(str(sample_path), framework="pt", device="cpu") as handle:
            required = {"self_attention_write", "cross_attention_write", "dit_latent_input", "surface", "query_source_index", "anchor_xyz"}
            require(required <= set(handle.keys()), "missing actual native attention writes/DiT input; producer recapture required")
            samples.append({name: handle.get_tensor(name) for name in required})
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = {"block_input", "context", "sigma", "timestep", "time_token", "mlp_input", "mlp_output", "block_output"}
        if block_index > 10:
            names.add("skip_value")
        tensors = {name: handle.get_tensor(name) for name in names}
        for sample_name, combined_name in (("surface", "surface"), ("query_source_index", "query_source_index"), ("anchor_xyz", "latent_xyz")):
            joined = torch.cat([sample[sample_name] for sample in samples], dim=0)
            reference = handle.get_tensor(combined_name)
            require(joined.dtype == reference.dtype and torch.equal(joined, reference), "per-object/combined source surface ownership mismatch")
    for index, row in enumerate(rows):
        require(type(row.get("requested_sigma")) in (int, float) and
                torch.equal(torch.tensor([row["requested_sigma"]], dtype=tensors["sigma"].dtype), tensors["sigma"][index:index + 1]),
                "requested sigma does not reproduce stored native time")
    for name in ("self_attention_write", "cross_attention_write", "dit_latent_input"):
        tensors[name] = torch.cat([sample[name] for sample in samples], dim=0)
    b, n, h = len(rows), CAPTURE_DIMS["tokens"], CAPTURE_DIMS["hidden"]
    for name in ("self_attention_write", "cross_attention_write"):
        require(tensors[name].shape == (b, n + 1, h), "incomplete source attention write")
    require(tensors["dit_latent_input"].shape == (b, n, CAPTURE_DIMS["channels"]), "incomplete actual DiT input")
    for name, tensor in tensors.items():
        require(tensor.dtype == torch.float16, f"native F16 capture storage required: {name}")
        capture.finite(tensor, name)
    return tensors, rows, tracked, body


def native_components(native, config, block_index):
    require(0 <= block_index < 15, "native MoE is not a dense block")
    p = config["model"]["params"]
    require(p["depth"] == 21 and p["num_moe_layers"] == 6 and not p["use_pos_emb"] and
            not p["use_attention_pooling"] and not p["with_decoupled_ca"], "audited native geometry required")
    dit = native["dit"]
    width = p["hidden_size"]
    block = dit.HunYuanDiTBlock(hidden_size=width, c_emb_size=width, num_heads=p["num_heads"],
        text_states_dim=p["context_dim"], qk_norm=p["qk_norm"], norm_layer=torch.nn.LayerNorm,
        qk_norm_layer=torch.nn.RMSNorm, qkv_bias=p["qkv_bias"], skip_connection=block_index > 10,
        with_decoupled_ca=False, timested_modulate=False, use_moe=False)
    time = dit.TimestepEmbedder(width, width * 4, cond_proj_dim=None)
    embedding = torch.nn.Linear(p["in_channels"], width, bias=True)
    return block, time, embedding


def residual(actual, reference):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype, "replay shape/dtype mismatch")
    capture.finite(actual, "replayed boundary")
    capture.finite(reference, "captured boundary")
    delta, ref = actual.double() - reference.double(), reference.double()
    denominator = float(ref.square().sum())
    squared = float(delta.square().sum())
    return {"exact": torch.equal(actual, reference), "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "relative_l2": (squared / denominator) ** .5 if denominator else (0. if squared == 0 else None),
            "reference_l2": denominator ** .5, "elements": actual.numel()}


@torch.inference_mode()
def replay_one(block, time_embedder, embedding, tensors, index, block_index, device):
    """Run the original block.forward unchanged, with the full captured sequence."""
    for module in (block, time_embedder, embedding):
        require(not module.training and not any(p.requires_grad for p in module.parameters()), "frozen native replay required")
        require(all(p.dtype == torch.float16 for p in module.parameters()), "native F16 replay weights required")
    require(not block.use_moe and not block.timested_modulate, "unsupported native block variant")
    require((block.skip_linear is not None) == (block_index > 10), "native skip ownership changed")
    row = {name: tensor[index:index + 1].to(device) for name, tensor in tensors.items()}
    observed = {}
    def put(name, value):
        require(name not in observed, "repeated native boundary hook")
        capture.finite(value, name)
        observed[name] = capture.cpu(value)
    with ExitStack() as hooks:
        for name, module in (("self_attention_write", block.attn1), ("cross_attention_write", block.attn2),
                             ("mlp_output", block.mlp)):
            hooks.callback(module.register_forward_hook(lambda m, a, out, name=name: put(name, out)).remove)
        hooks.callback(block.mlp.register_forward_pre_hook(lambda m, a: put("mlp_input", a[0])).remove)
        token = time_embedder(row["sigma"], condition=None)
        put("time_token", token)
        if block_index == 0:
            initial = torch.cat((token, embedding(row["dit_latent_input"])), dim=1)
            put("initial_block_input", initial)
        output = block(row["block_input"], token, row["context"], skip_value=row.get("skip_value"))
        put("block_output", output)
    references = {name: tensors[name][index:index + 1] for name in BOUNDARIES}
    if block_index == 0:
        references["initial_block_input"] = tensors["block_input"][index:index + 1]
    return observed, {name: residual(observed[name], ref) for name, ref in references.items()}


def load_candidate(receipt_path, source, body, rows, width, device):
    """A supplied frozen artifact only; no fitting or implicit precision changes."""
    receipt_path = Path(receipt_path).resolve(strict=True)
    body_raw = bounded_bytes(receipt_path)
    receipt = unique_json(body_raw)
    require(isinstance(receipt, dict) and receipt.get("schema") == FOLD_SCHEMA and
            receipt.get("source_identity") == source["source_identity"] and
            receipt.get("block_index") == body["block_index"] and
            receipt.get("capture_sha256") == body["file_sha256"], "folded MLP source/capture provenance mismatch")
    fit = sorted({r["object_sha256"] for r in rows if r["split"] == "fit"})
    require(receipt.get("fit_object_sha256") == fit, "explicit exact fit object inventory required; held objects cannot be fitted")
    path = sealed_path(receipt_path.parent, receipt.get("file"))
    require(tensor_header(path)["bytes"] <= 256 << 20 and capture.digest_file(path) == receipt.get("file_sha256"), "folded artifact hash/bound mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        require(set(handle.keys()) == names, "folded artifact must contain precisely both native biased linears")
        state = {name: handle.get_tensor(name) for name in names}
    first = state["fc1.weight"]
    require(first.ndim == 2 and first.shape[1] == width and 0 < first.shape[0] <= width * 4, "invalid frozen MLP width")
    keep = first.shape[0]
    shapes = {"fc1.weight": (keep, width), "fc1.bias": (keep,), "fc2.weight": (width, keep), "fc2.bias": (width,)}
    for name, value in state.items():
        require(value.dtype == torch.float16 and tuple(value.shape) == shapes[name], "native F16 folded MLP geometry required")
        capture.finite(value, name)
    # Retain native MLP.forward and its exact GELU; replace only its two linears.
    first = torch.nn.Linear(width, keep, bias=True, device=device, dtype=torch.float16)
    second = torch.nn.Linear(keep, width, bias=True, device=device, dtype=torch.float16)
    for module, prefix in ((first, "fc1"), (second, "fc2")):
        module.load_state_dict({k: state[prefix + "." + k] for k in ("weight", "bias")}, strict=True)
        module.eval().requires_grad_(False)
    return first, second, {"receipt_sha256": hashlib.sha256(body_raw).hexdigest(), "file_sha256": receipt["file_sha256"],
                          "parameters": sum(v.numel() for v in state.values()), "width": keep,
                          "fit_provenance": "receipt declaration; no independent fitting audit", "quality_accepted": False}, {
                              receipt_path: hashlib.sha256(body_raw).hexdigest(), path: receipt["file_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-run", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--parent-checkpoint", default=MODEL_PATH)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--folded-mlp-receipt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    require(not output.exists(), "fresh replay report required")
    source = inspect_source(args.parent_checkpoint, args.source_root)
    parent_stamp = stamp(Path(args.parent_checkpoint))
    tensors, rows, tracked, body = load_capture(args.capture_run, args.block, source)
    os.environ.update(USE_SAGEATTN="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HY3DGEN_DEBUG="0")
    import yaml
    config = yaml.safe_load(bounded_bytes(Path(args.source_root) / "configs/dit_config_2_1.yaml"))
    baseline, folded = [], []
    candidate = None
    with capture.native_modules(args.source_root, include_dit=True) as native, torch.inference_mode():
        block, time, embedding = native_components(native, config, args.block)
        block = capture.strict_component(block, args.parent_checkpoint, f"model.blocks.{args.block}.", args.device)
        time = capture.strict_component(time, args.parent_checkpoint, "model.t_embedder.", args.device)
        embedding = capture.strict_component(embedding, args.parent_checkpoint, "model.x_embedder.", args.device)
        parent_count = sum(p.numel() for p in block.mlp.parameters())
        for index, row in enumerate(rows):
            _, metrics = replay_one(block, time, embedding, tensors, index, args.block, args.device)
            baseline.append({"sample": row, "boundaries": metrics})
        exact = all(m["exact"] for row in baseline for m in row["boundaries"].values())
        if exact and args.folded_mlp_receipt:
            first, second, candidate, candidate_files = load_candidate(args.folded_mlp_receipt, source, body, rows, CAPTURE_DIMS["hidden"], args.device)
            tracked.update(candidate_files)
            old = block.mlp.fc1, block.mlp.fc2
            block.mlp.fc1, block.mlp.fc2 = first, second
            try:
                for index, row in enumerate(rows):
                    _, metrics = replay_one(block, time, embedding, tensors, index, args.block, args.device)
                    require(all(metrics[name]["exact"] for name in ("time_token", "self_attention_write", "cross_attention_write", "mlp_input")),
                            "fold changed frozen non-MLP source boundary")
                    folded.append({"sample": row, "boundaries": metrics})
            finally:
                block.mlp.fc1, block.mlp.fc2 = old
            candidate.update(parent_mlp_parameters=parent_count, parameter_reduction=parent_count / candidate["parameters"])
    for path, digest in tracked.items():
        require(capture.digest_file(path) == digest, "replay evidence changed during execution")
    require(stamp(Path(args.parent_checkpoint)) == parent_stamp and
            inspect_source(args.parent_checkpoint, args.source_root)["source_identity"] == source["source_identity"], "parent/source changed during replay")
    capture.verify_native_dependencies(args.source_root)
    report = {"schema": SCHEMA, "status": "baseline_exact" if exact else "baseline_mismatch",
        "source_identity": source["source_identity"], "block_index": args.block,
        "baseline": baseline, "baseline_all_boundaries_exact": exact, "candidate": candidate, "folded": folded,
        "capture_sha256": body["file_sha256"], "input_file_sha256": {str(p): s for p, s in tracked.items()},
        "runtime_versions": capture.runtime_versions(), "device": args.device,
        "full_context": True, "source_dtype": "float16", "new_quality_thresholds": None,
        "host_charts_installed": False, "folded_child_proven": False,
        "scope": "one native source block; no parent mesh-quality or child capability claim"}
    capture.sealed_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output), "folded_child_proven": False}), flush=True)
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
