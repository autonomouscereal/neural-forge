#!/usr/bin/env python3
"""Actual source-bound full21 per-block dense folding, with bounded FIT-only solve.

Existing capture --block already supports all21. Admit capture windows separately;
this driver never owns GPU allocations, source captures or production state.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import re
import torch
from safetensors import safe_open
import c16_capture_hy3d as capture
import hy3d_full21_native as teacher
from hy3d_full21_contract import block_contract, source_groups, require
from hy3d_full21_linalg import scores
from hy3d_dense_child_contract import SCHEMA, validate_policy, child_contract
from hy3d_dense_child_fit import fit_dense
from hy3d_dense_child_native import installed_child, replay_child, native_route_masses


def inventory(rows):
    require(isinstance(rows, list) and len(rows) == 4, "four actual source objects required")
    parts = {"fit": [], "held": []}; seen = {key: set() for key in ("object", "conditioning", "geometry")}
    for i, row in enumerate(rows):
        require(isinstance(row, dict) and row.get("split") in parts, "invalid source split")
        values = (row.get("object_sha256"), row.get("conditioning_sha256"), row.get("geometry_identity", {}).get("sha256"))
        for key, value in zip(seen, values):
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) and value not in seen[key], "duplicate/missing original source identity")
            seen[key].add(value)
        parts[row["split"]].append(i)
    require(all(len(ids) == 2 for ids in parts.values()), "exact two FIT and two HELD objects required")
    return parts


@torch.inference_mode()
def fold_loaded(tensors, rows, block, time, embedding, index, policy, device):
    validate_policy(policy, index); parts = inventory(rows); topology = block_contract(index)
    require(tensors["block_input"].shape == (4, 4097, 2048), "complete actual source context required")
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"), "external autocast forbidden")
    branches = teacher.branch_modules(block, index)
    baseline = []
    for i, row in enumerate(rows):
        observed, metrics = teacher.replay_one(block, time, embedding, tensors, i, index, device)
        baseline.append({"object_sha256": row["object_sha256"], "split": row["split"], "boundaries": metrics})
        del observed
    basis = []
    for info, _, first, second in branches:
        basis.append({"id": info["id"], "first.weight": capture.cpu(first.weight), "first.bias": capture.cpu(first.bias),
                      "second.weight": capture.cpu(second.weight), "second.bias": capture.cpu(second.bias)})
    inputs = [tensors["mlp_input"][i].reshape(-1, 2048) for i in parts["fit"]]
    targets = [tensors["mlp_output"][i].reshape(-1, 2048) for i in parts["fit"]]
    masses, routing_diagnostics = native_route_masses(tensors, parts["fit"]) if topology["use_moe"] else (None, [])
    state, selected, fit = fit_dense(inputs, targets, basis, policy["keep"], policy, masses,
                                      splits=[rows[i]["split"] for i in parts["fit"]])
    contract = child_contract(index, 2048, policy["keep"], selected.tolist(), [b["first.weight"].shape[0] for b in basis])
    del basis, inputs, targets, masses
    results = []
    with installed_child(block, index, state, contract, device):
        for i, row in enumerate(rows):
            observed, metrics = replay_child(block, time, embedding, tensors, i, index, device)
            results.append({"object_sha256": row["object_sha256"], "split": row["split"], "boundaries": metrics,
                            "diagnostics": {name: scores(observed[name].reshape(-1, 2048), tensors[name][i])
                            for name in ("mlp_output", "block_output")}})
    parent_ffn = sum(p.numel() for p in (block.moe if index >= 15 else block.mlp).parameters())
    child_ffn = sum(t.numel() for t in state.values())
    return state, selected, {"dense_child": contract, "fit": fit, "baseline": baseline,
            "routing_reconstruction_diagnostics": routing_diagnostics,
            "persistent_capture_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "persistent_child_bytes": sum(t.numel() * t.element_size() for t in state.values()),
            "baseline_all_boundaries_byte_exact": all(m["byte_exact"] for r in baseline for m in r["boundaries"].values()),
            "actual_source_forward_completed": True, "baseline_finite_mismatch_veto": False,
            "candidate_diagnostics": results, "held_candidate_diagnostic_passes": 1,
            "parent_ffn_parameters": parent_ffn, "child_ffn_parameters": child_ffn,
            "compression_x": parent_ffn / child_ffn, "numeric_veto": False}


def convert(run, index, parent, source_root, policy_file, policy_sha256, output, device):
    raw = capture.bounded_bytes(Path(policy_file)); require(hashlib.sha256(raw).hexdigest() == policy_sha256, "explicit policy hash mismatch")
    policy = validate_policy(capture.unique_json(raw), index)
    source = capture.inspect_source(parent, source_root); before = capture.stamp(Path(parent))
    accounting = source_groups(capture.tensor_header(parent)["tensors"])
    tensors, rows, tracked, body = teacher.load_capture(run, index, source); parts = inventory(rows)
    out = Path(output).resolve(); require(not out.exists(), "fresh output required; retain partial and historical runs")
    tracked[Path(policy_file)] = policy_sha256
    for path in Path(__file__).parent.glob("*.py"): tracked[path] = capture.digest_file(path)
    report = {"schema": SCHEMA, "block_index": index, "source_identity": source["source_identity"],
              "capture_sha256": body["file_sha256"], "policy": policy, "policy_sha256": policy_sha256,
              "fit_object_sha256": sorted(rows[i]["object_sha256"] for i in parts["fit"]),
              "held_object_sha256": sorted(rows[i]["object_sha256"] for i in parts["held"]),
              "fixture": False, "training": False, "capability_evaluation": False, "source_group_inventory": accounting,
              "all_source_child_proven": False, "folded_child_proven": False, "device": device}
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(USE_SAGEATTN="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HY3DGEN_DEBUG="0")
    try:
        import yaml
        config = yaml.safe_load(capture.bounded_bytes(Path(source_root) / "configs/dit_config_2_1.yaml"))
        with capture.native_modules(source_root, include_dit=True) as modules, torch.inference_mode():
            block, time, embedding = teacher.native_components(modules, config, index)
            block = teacher.strict_load(block, parent, f"model.blocks.{index}.", device)
            time = teacher.strict_load(time, parent, "model.t_embedder.", device)
            embedding = teacher.strict_load(embedding, parent, "model.x_embedder.", device)
            identities = teacher.component_identity(block, time, embedding, index)
            state, selected, measured = fold_loaded(tensors, rows, block, time, embedding, index, policy, device)
            report.update(measured)
            require(teacher.component_identity(block, time, embedding, index) == identities, "source component changed")
        with safe_open(str(parent), framework="pt", device="cpu") as handle:
            require(all(teacher.tensor_identity(handle.get_tensor(k)) == v for k, v in identities.items()), "source body changed")
        require(capture.stamp(Path(parent)) == before and capture.inspect_source(parent, source_root)["source_identity"] == source["source_identity"], "source changed during fold")
        capture.verify_native_dependencies(source_root)
        require(all(capture.digest_file(p) == h for p, h in tracked.items()), "fold evidence/code changed")
        selection = capture.sealed_tensors(out / "selection.safetensors", {"selected_source_units": selected})
        artifact = capture.sealed_tensors(out / "dense_child.safetensors", state)
        tensor_ids = {k: teacher.tensor_identity(v) for k, v in state.items()}
        with safe_open(str(out / artifact["file"]), framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == set(state) and all(teacher.tensor_identity(handle.get_tensor(k)) == tensor_ids[k] for k in state), "stored child readback mismatch")
        report.update(**artifact, selection=selection, tensor_identity=tensor_ids, source_tensor_identity=identities,
                      status="saved_finite_explicit_dense_child", candidate_written=True, stored_tensor_readback_verified=True,
                      input_file_sha256={str(p): h for p, h in tracked.items()})
        capture.sealed_json(out / "fold_receipt.json", report); return report
    except Exception as exc:
        capture.sealed_json(out / "incomplete.json", {**report, "status": "incomplete", "exception_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-run", "parent-checkpoint", "source-root", "policy", "policy-sha256", "output"): parser.add_argument("--" + name, required=True)
    parser.add_argument("--block", type=int, required=True, choices=range(21))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(); torch.set_num_threads(args.cpu_threads)
    result = convert(args.capture_run, args.block, args.parent_checkpoint, args.source_root, args.policy, args.policy_sha256, args.output, args.device)
    print(capture.json.dumps({"status": result["status"], "receipt": str(Path(args.output) / "fold_receipt.json")}))


if __name__ == "__main__": main()
