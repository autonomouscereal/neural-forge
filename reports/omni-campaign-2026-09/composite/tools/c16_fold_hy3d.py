#!/usr/bin/env python3
"""One frozen, provenance-bound HY3D block0 dense-MLP fold. No training/install.

The original complete-context native block must replay bit-exact before any
selection. Only FIT objects enter the unchanged mediafold pivot/ridge solver.
Failure leaves a sealed rejection, never an accepted tensor artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import torch
from safetensors import safe_open

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parents[1] / "media"))
import c16_replay_hy3d as replay
from c16_qimg_preflight import biased_gelu_mlp_parameters, maximum_fourfold_keep
from mediafold import linalg

capture = replay.capture
require = replay.require
POLICY_SCHEMA = "neural-forge.hy3d-dense-fold-policy.v1"
REPORT_SCHEMA = "neural-forge.hy3d-dense-fold-report.v1"
SCORED = ("mlp_output", "block_output")
FROZEN = ("time_token", "self_attention_write", "cross_attention_write", "mlp_input", "initial_block_input")
WEIGHTS = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")


class FoldRejected(ValueError):
    def __init__(self, status, **evidence):
        super().__init__(status)
        self.receipt = {"status": status, "numerical_policy_passed": False,
                        "candidate_written": False, **evidence}


def load_policy(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    raw = replay.bounded_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "explicit policy SHA256 mismatch")
    body = replay.unique_json(raw)
    keys = {"schema", "policy_id", "description", "preregistered_before_scoring", "block_index",
            "keep", "fit_objects", "held_objects", "gate_scope", "baseline", *SCORED}
    require(isinstance(body, dict) and set(body) == keys, "exact implemented policy fields required")
    require(body["schema"] == POLICY_SCHEMA and body["preregistered_before_scoring"] is True,
            "frozen preregistered policy required")
    require(isinstance(body["policy_id"], str) and bool(body["policy_id"]) and
            isinstance(body["description"], str), "policy identity/description required")
    require(type(body["block_index"]) is int and body["block_index"] == 0, "only native dense block0 is preregistered")
    require(type(body["keep"]) is int and body["keep"] > 0, "positive declared channel count required")
    require(type(body["fit_objects"]) is int and body["fit_objects"] == 2 and
            type(body["held_objects"]) is int and body["held_objects"] == 2, "exactly two fit and two held objects required")
    require(body["gate_scope"] == "pooled_and_each_held_object_full_context" and
            body["baseline"] == "all_native_boundaries_bit_exact", "unsupported policy scope/baseline")
    for name in SCORED:
        gate = body[name]
        require(isinstance(gate, dict) and set(gate) == {"max_relative_l2", "min_mean_cosine"}, "unknown or missing numerical gate")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in gate.values()), "finite numerical gates required")
        require(gate["max_relative_l2"] >= 0 and -1 <= gate["min_mean_cosine"] <= 1, "invalid numerical gate range")
    return body


def split_inventory(rows):
    require(isinstance(rows, list) and len(rows) == 4, "one complete context for each of two fit and two held objects required")
    inventory = {"fit": [], "held": []}
    seen = {key: set() for key in ("object_sha256", "conditioning_sha256", "geometry_sha256")}
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and row.get("split") in inventory, "invalid object split")
        geometry = row.get("geometry_identity", {})
        require(isinstance(geometry, dict) and geometry.get("schema") == capture.GEOMETRY_SCHEMA, "canonical source geometry required")
        identities = {"object_sha256": row.get("object_sha256"), "conditioning_sha256": row.get("conditioning_sha256"),
                      "geometry_sha256": geometry.get("sha256")}
        for key, value in identities.items():
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "explicit source object/image/geometry SHA256 required")
            require(value not in seen[key], f"duplicate object/conditioning/geometry identity: {key}")
            seen[key].add(value)
        require(type(row.get("seed")) is int and row["seed"] >= 0 and
                type(row.get("requested_sigma")) in (int, float) and math.isfinite(row["requested_sigma"]) and
                0 <= row["requested_sigma"] <= 1, "actual seed and sigma required")
        inventory[row["split"]].append(index)
    require(all(len(v) == 2 for v in inventory.values()), "exactly two fit and two held objects required")
    return inventory


def tensor_identity(value):
    value = value.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel(),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def source_tensor_identity(block, time, embedding):
    return {prefix + name: tensor_identity(value)
            for prefix, module in (("model.blocks.0.", block), ("model.t_embedder.", time), ("model.x_embedder.", embedding))
            for name, value in module.state_dict().items()}


def verify_parent_tensors(path, identities):
    # Hash only the selected native block + time + input embedding, never the full parent.
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name, expected in identities.items():
            require(tensor_identity(handle.get_tensor(name)) == expected, "source tensor changed during fold: " + name)


def preflight(tensors, rows, block, policy):
    require(not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
            "external autocast is forbidden; preserve native F16 math")
    inventory = split_inventory(rows)
    require(policy["block_index"] == 0 and block.skip_linear is None and not block.use_moe and
            not block.timested_modulate, "only original native dense block0 is supported")
    mlp = block.mlp
    require(isinstance(mlp.gelu, torch.nn.GELU) and mlp.gelu.approximate == "none", "native exact GELU required; QIMG tanh GELU is different")
    require(set(mlp.state_dict()) == set(WEIGHTS), "precisely both native biased MLP linears required")
    width, inner = mlp.fc1.in_features, mlp.fc1.out_features
    require(inner == 4 * width and mlp.fc2.in_features == inner and mlp.fc2.out_features == width,
            "native 4x expansion geometry required")
    keep = policy["keep"]
    require(0 < keep <= maximum_fourfold_keep(width, inner), "bias-inclusive 4x compression floor violated")
    n = tensors["block_input"].shape[1]
    require(n > 1 and all(v.shape[0] == len(rows) and v.dtype == torch.float16 for v in tensors.values()), "complete F16 contexts required")
    require(all(tensors[k].shape == (4, n, width) for k in (*SCORED, "mlp_input", "self_attention_write", "cross_attention_write")),
            "every token, including time token0, must be present")
    require("skip_value" not in tensors and torch.equal(tensors["sigma"], tensors["timestep"]), "unexpected skip or changed native time")
    for i, row in enumerate(rows):
        require(torch.equal(torch.tensor([row["requested_sigma"]], dtype=torch.float16), tensors["sigma"][i:i + 1]), "source sigma mismatch")
    for name, value in tensors.items():
        capture.finite(value, name)
    fit_count = len(inventory["fit"]) * n
    require(2 * keep <= fit_count <= 131072, "bounded fit rows must be at least twice the selected channels")
    return inventory, {"width": width, "parent_inner": inner, "keep": keep, "tokens_per_object": n,
        "fit_rows": fit_count, "held_rows": 2 * n, "time_token_included": True,
        "parent_parameters": biased_gelu_mlp_parameters(width, inner),
        "child_parameters": biased_gelu_mlp_parameters(width, keep)}


def passes(scores, gate):
    return (all(math.isfinite(scores[k]) for k in ("rel_l2", "cos")) and
            scores["rel_l2"] <= gate["max_relative_l2"] and scores["cos"] >= gate["min_mean_cosine"])


def bit_exact(actual, reference):
    return (actual.dtype == reference.dtype and actual.shape == reference.shape and
            torch.equal(actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)))


def byte_exact_metrics(observed, tensors, index, metrics):
    # torch.equal alone treats +0 and -0 as equal. This converter's frozen policy
    # says bit-exact, so retain the oracle residuals but compare literal F16 bytes.
    for name, row in metrics.items():
        reference_name = "block_input" if name == "initial_block_input" else name
        row["exact"] = bit_exact(observed[name], tensors[reference_name][index:index + 1])
    return metrics


@torch.inference_mode()
def fold_loaded(tensors, rows, block, time, embedding, policy, device="cpu"):
    """Internal testable mechanics. Production calls only through strict convert()."""
    inventory, counts = preflight(tensors, rows, block, policy)
    baseline = []
    # Every object, context, time path and write is replayed BEFORE any fit work.
    for index, row in enumerate(rows):
        observed, metrics = replay.replay_one(block, time, embedding, tensors, index, 0, device)
        metrics = byte_exact_metrics(observed, tensors, index, metrics)
        baseline.append({"object_sha256": row["object_sha256"], "split": row["split"], "boundaries": metrics})
    if not all(m["exact"] for result in baseline for m in result["boundaries"].values()):
        raise FoldRejected("rejected_source_replay", baseline=baseline, counts=counts)
    mlp = block.mlp
    original = {name: capture.cpu(value) for name, value in mlp.state_dict().items()}
    features = []
    # Keep original B=1/full-sequence GEMM grouping; only the F64 Gram is chunked.
    for index in inventory["fit"]:
        x = tensors["mlp_input"][index:index + 1].to(device)
        hidden = mlp.gelu(mlp.fc1(x))
        capture.finite(hidden, "native fit GELU features")
        require(bit_exact(capture.cpu(mlp.fc2(hidden)), tensors["mlp_output"][index:index + 1]), "fit feature replay changed native F16 grouping")
        features.append(capture.cpu(hidden).squeeze(0))
    features = torch.cat(features, dim=0)
    gram = torch.zeros(counts["parent_inner"], counts["parent_inner"], dtype=torch.float64)
    for chunk in features.split(512):
        values = chunk.double()
        gram.add_(values.T @ values)
    selected = linalg.pivoted_columns_gram(gram, counts["keep"])
    del gram
    if selected.numel() != counts["keep"] or selected.unique().numel() != counts["keep"]:
        raise FoldRejected("rejected_fit_rank", selected=int(selected.numel()), counts=counts, baseline=baseline)
    selected = selected.sort().values
    prior = torch.cat((original["fc2.weight"][:, selected].double(), original["fc2.bias"].double()[:, None]), dim=1)
    design = torch.cat((features[:, selected].double(), torch.ones(len(features), 1, dtype=torch.float64)), dim=1)
    target = torch.cat([tensors["mlp_output"][i].double() for i in inventory["fit"]], dim=0)
    fitted, solve = linalg.law_abiding_solve(design, target, prior)
    capture.finite(fitted, "F64 prior-anchored solve")
    stored = fitted.to(torch.float16)
    update = float(((stored.double() - prior).norm() / prior.norm().clamp_min(1e-30)).item())
    if not bool(torch.isfinite(stored).all()) or not math.isfinite(update) or update > 1:
        raise FoldRejected("rejected_stored_update", stored_update_scale=update if math.isfinite(update) else None,
                           counts=counts, baseline=baseline, solve=solve)
    state = {"fc1.weight": original["fc1.weight"][selected].contiguous(), "fc1.bias": original["fc1.bias"][selected].contiguous(),
             "fc2.weight": stored[:, :-1].contiguous(), "fc2.bias": stored[:, -1].contiguous()}
    require(sum(v.numel() for v in state.values()) == counts["child_parameters"] and
            4 * counts["child_parameters"] <= counts["parent_parameters"], "stored biased MLP count disagreement")
    # No held values reached the selection or ridge solver. Report F64 fit separately.
    fit_f64 = linalg.write_scores(design @ fitted.T, target)
    del features, design, target, fitted
    replacements = [torch.nn.Linear(a, b, bias=True, device=device, dtype=torch.float16)
                    for a, b in ((counts["width"], counts["keep"]), (counts["keep"], counts["width"]))]
    for prefix, module in zip(("fc1", "fc2"), replacements):
        module.load_state_dict({k: state[prefix + "." + k] for k in ("weight", "bias")}, strict=True)
        module.eval().requires_grad_(False)
    old = mlp.fc1, mlp.fc2
    mlp.fc1, mlp.fc2 = replacements
    grouped = {split: {name: [] for name in SCORED} for split in inventory}
    results = []
    try:
        for index, row in enumerate(rows):
            observed, metrics = replay.replay_one(block, time, embedding, tensors, index, 0, device)
            metrics = byte_exact_metrics(observed, tensors, index, metrics)
            require(all(metrics[name]["exact"] for name in FROZEN), "fold changed frozen non-MLP source boundary")
            scores = {name: linalg.write_scores(observed[name].reshape(-1, counts["width"]), tensors[name][index]) for name in SCORED}
            results.append({"object_sha256": row["object_sha256"], "split": row["split"], "boundaries": metrics, "scores": scores})
            for name in SCORED:
                grouped[row["split"]][name].append(observed[name].squeeze(0))
    finally:
        mlp.fc1, mlp.fc2 = old
    require(all(tensor_identity(v) == tensor_identity(original[k]) for k, v in mlp.state_dict().items()), "source MLP was mutated")
    pooled = {split: {name: linalg.write_scores(torch.cat(grouped[split][name]),
                 torch.cat([tensors[name][i] for i in inventory[split]])) for name in SCORED} for split in inventory}
    accepted = all(passes(pooled["held"][name], policy[name]) for name in SCORED) and all(
        passes(result["scores"][name], policy[name]) for result in results if result["split"] == "held" for name in SCORED)
    report = {"baseline": baseline, "baseline_all_boundaries_exact": True, "counts": counts,
        "compression_x": counts["parent_parameters"] / counts["child_parameters"], "solve": solve,
        "stored_update_scale": update, "fp64_fit_only_scores": fit_f64, "runtime_f16": {"pooled": pooled, "objects": results},
        "numerical_policy_passed": accepted, "law": {"update_scale_max": 1, "fit_only_selection_and_ridge": True,
            "held_used_for_selection_or_refit": False, "token_shuffle": False, "solver": "mediafold.linalg.law_abiding_solve unchanged",
            "prior": "selected native fc2 columns plus native output bias", "fallback_to_prior_permitted_by_unchanged_law": True}}
    if not accepted:
        raise FoldRejected("rejected_numerical_policy", **report)
    return state, selected, report


def verify_tracked(tracked):
    for path, expected in tracked.items():
        require(capture.digest_file(path) == expected, "fold input changed: " + str(path))


def convert(capture_run, parent, source_root, policy_path, policy_sha256, output, device="cpu"):
    """Production path: real pinned native source, exact parent, sealed captures only."""
    output = Path(output).resolve()
    require(not output.exists(), "fresh output directory required; historical and partial results are preserved")
    policy_path = Path(policy_path).resolve(strict=True)
    policy = load_policy(policy_path, policy_sha256)
    source = replay.inspect_source(parent, source_root)
    parent_stamp = replay.stamp(Path(parent))
    tensors, rows, tracked, body = replay.load_capture(capture_run, 0, source)
    inventory = split_inventory(rows)
    tracked[policy_path] = policy_sha256
    tool_paths = (Path(__file__), Path(replay.__file__), Path(capture.__file__), TOOLS / "c16_hy3d_contract.py",
                  TOOLS / "c16_qimg_preflight.py", Path(linalg.__file__))
    for path in tool_paths:
        tracked[path] = capture.digest_file(path)
    report = {"schema": REPORT_SCHEMA, "policy": policy, "policy_sha256": policy_sha256,
        "source_identity": source["source_identity"], "block_index": 0, "capture_sha256": body["file_sha256"],
        "fit_object_sha256": sorted(rows[i]["object_sha256"] for i in inventory["fit"]),
        "held_object_sha256": sorted(rows[i]["object_sha256"] for i in inventory["held"]), "samples": rows,
        "input_file_sha256": {str(p): s for p, s in tracked.items()}, "runtime_versions": capture.runtime_versions(),
        "device": device, "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "source_dtype": "float16", "activation": "GELU approximate=none", "solve_dtype": "float64",
        "full_context": True, "fixture": False, "host_charts_installed": False, "folded_child_proven": False,
        "capability_claim": "none; one source block engineering candidate, not mesh quality, native interface or child proof"}
    os.environ.update(USE_SAGEATTN="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HY3DGEN_DEBUG="0")
    output.mkdir(parents=True, exist_ok=False)
    accepted_receipt = output / "fold_receipt.json"
    try:
        import yaml
        config = yaml.safe_load(replay.bounded_bytes(Path(source_root) / "configs/dit_config_2_1.yaml"))
        rejected = None
        with capture.native_modules(source_root, include_dit=True) as native, torch.inference_mode():
            block, time, embedding = replay.native_components(native, config, 0)
            block = capture.strict_component(block, parent, "model.blocks.0.", device)
            time = capture.strict_component(time, parent, "model.t_embedder.", device)
            embedding = capture.strict_component(embedding, parent, "model.x_embedder.", device)
            source_tensors = source_tensor_identity(block, time, embedding)
            report["source_tensor_identity"] = source_tensors
            try:
                state, selection, measured = fold_loaded(tensors, rows, block, time, embedding, policy, device)
                report.update(measured)
            except FoldRejected as exc:
                rejected = exc
                report.update(exc.receipt)
            require(source_tensor_identity(block, time, embedding) == source_tensors, "source block/time/input tensors mutated")
        verify_tracked(tracked)
        require(replay.stamp(Path(parent)) == parent_stamp and
                replay.inspect_source(parent, source_root)["source_identity"] == source["source_identity"], "parent/source changed")
        capture.verify_native_dependencies(source_root)
        verify_parent_tensors(parent, source_tensors)
        require(replay.stamp(Path(parent)) == parent_stamp, "parent changed during final selected-tensor verification")
        if rejected:
            capture.sealed_json(output / "rejection.json", report)
            rejected.receipt = report
            raise rejected
        selection_file = capture.sealed_tensors(output / "selection.safetensors", {"selected_channels": selection})
        candidate = capture.sealed_tensors(output / "folded_mlp.safetensors", state)
        report.update(schema=replay.FOLD_SCHEMA, status="accepted_component_numerical_policy", candidate_written=True,
                      **candidate, selection=selection_file, tensor_identity={k: tensor_identity(v) for k, v in state.items()})
        # Read stored bytes back before publishing the final receipt. Integration
        # tests exercise the existing standalone replay loader on this contract.
        with safe_open(str(output / candidate["file"]), framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == set(state) and all(tensor_identity(handle.get_tensor(k)) == report["tensor_identity"][k]
                    for k in state), "written candidate changed")
        verify_tracked(tracked)
        require(replay.stamp(Path(parent)) == parent_stamp, "parent changed before final receipt")
        capture.sealed_json(accepted_receipt, report)
        return report
    except FoldRejected:
        raise
    except Exception as exc:
        # Preserve partial artifacts; no candidate receipt is published before all
        # numerical, lineage and stored-byte checks above complete.
        capture.sealed_json(output / "incomplete.json", {**report, "schema": REPORT_SCHEMA, "status": "incomplete",
            "candidate_written": (output / "folded_mlp.safetensors").exists(), "numerical_policy_passed": False,
            "exception_type": type(exc).__name__, "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-run", required=True)
    parser.add_argument("--parent-checkpoint", default=replay.MODEL_PATH)
    parser.add_argument("--source-root", default=replay.SOURCE_ROOT)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 33), default=4,
                        help="CPU F64 selection/ridge threads, also used with GPU native replay")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    try:
        result = convert(args.capture_run, args.parent_checkpoint, args.source_root, args.policy,
                         args.policy_sha256, args.output, args.device)
    except FoldRejected as exc:
        print(json.dumps({"status": exc.receipt["status"], "report": str(Path(args.output) / "rejection.json")}), flush=True)
        return 2
    print(json.dumps({"status": result["status"], "receipt": str(Path(args.output) / "fold_receipt.json"), "folded_child_proven": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
