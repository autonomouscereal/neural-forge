"""Candidate-only, through-origin charts of actual native FFN inputs and writes.

No training, identity fallback, held-selected regularization, legacy row keys,
chart installation or source-width changes. Both maps have explicit zero bias:
C10's bias-free child cannot faithfully retain either learned affine offset.
The numerical objective/ladder follows mediafold.linalg; accumulation is chunked
and the Cholesky solve refuses instead of silently increasing regularization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import c10_union_install as installer
import c17_capture_host as host
import c17_capture_k3 as k3
import c17_chart_fit as legacy

SCHEMA = "neural-forge.c17-native-chart-candidate.v1"
PLAN_SCHEMA = "neural-forge.c17-native-chart-plan.v1"
IDENTITIES = ("evidence_row_index", "batch_index", "host_position", "k3_position", "held",
              "host_flat_position", "k3_flat_position")
require, Refusal = k3.require, k3.Refusal


def original_vectors(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    values = {key: [] for key in IDENTITIES}
    for row in rows:
        rank, seq = row["row_index"], row["sequence_length"]
        for q, k in row["pairs"]:
            item = (rank, row["batch_index"], q, k, int(row["split"] == "held"), rank * seq + q, rank * seq + k)
            for name, value in zip(IDENTITIES, item):
                values[name].append(value)
    return {key: torch.tensor(value, dtype=torch.int64) for key, value in values.items()}


def read_capture(ref, family: str, layer: int, pins: k3.Pins, vectors: dict[str, torch.Tensor],
                 evidence_sha: str, boundary_sha: str, fixture: bool) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    rec = pins.json(ref)
    require(rec.get("schema") == (host.SCHEMA if family == "host" else k3.SCHEMA), "wrong native capture schema")
    require(rec.get("fixture", False) is fixture and rec.get("gradients_or_training") is False,
            "native capture is synthetic or used gradients")
    require(rec.get("evidence_sha256") == evidence_sha and rec.get("boundary_sha256") == boundary_sha,
            "native capture evidence/boundary identity mismatch")
    code = rec.get("code_sha256")
    require(isinstance(code, dict) and bool(code) and all(isinstance(s, str) and host.SHA.fullmatch(s) for s in code.values()),
            "native capture source identity missing")
    if family == "host":
        require(rec.get("consumed_tensor_hashes_verified") is True and rec.get("checkpoint_stat_stable") is True,
                "host capture tensor/source verification missing")
    else:
        require(rec.get("stream") == "folded" and rec.get("accepted_prefix_replay_exact") is True and
                rec.get("teacher_reference_verified") is True and rec.get("teacher_forward_replayed") is False and
                rec.get("parent_experts_loaded") is False and rec.get("new_fit") is False and
                rec.get("input_stat_stable") is True, "accepted folded native source gate missing")
    layers = [x for x in rec.get("layers", []) if x.get("layer") == layer]
    require(len(layers) == 1, "native capture layer missing or ambiguous")
    entry = layers[0]
    directory = Path(ref["path"]).resolve().parent
    path = (directory / entry["path"]).resolve(strict=True)
    require(path.is_relative_to(directory), "native row path escapes capture directory")
    pins.file(path, entry["sha256"])
    count = len(vectors["held"])
    require(type(entry.get("rows")) is int and entry["rows"] == count, "captured row count differs from original pairs")
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        meta = sf.metadata() or {}
        require(meta.get("schema") == rec["schema"] and meta.get("layer") == str(layer), "native row metadata mismatch")
        for key, expected in vectors.items():
            require(key in sf.keys(), f"native row identity missing: {key}")
            got = sf.get_tensor(key)
            require(got.dtype == torch.int64 and torch.equal(got, expected), f"original paired positions/groups differ: {key}")
        payload = {}
        for boundary in ("ffn_input", "ffn_output"):
            require(boundary in sf.keys(), f"actual native boundary missing: {boundary}")
            value = sf.get_tensor(boundary)
            require(value.ndim == 2 and value.shape[0] == count and value.shape[1] > 0 and
                    value.dtype in (torch.float16, torch.bfloat16, torch.float32) and
                    entry.get("boundaries", {}).get(boundary, {}).get("shape") == list(value.shape),
                    "native boundary shape/dtype declaration mismatch")
            payload[boundary] = value
    require(payload["ffn_input"].shape == payload["ffn_output"].shape, "native input/write widths differ")
    return rec, payload


def split_fit_groups(vectors: dict[str, torch.Tensor], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    outer_fit = vectors["held"] == 0
    groups = torch.unique(vectors["evidence_row_index"][outer_fit], sorted=True)
    require(len(groups) >= 2, "lambda selection needs at least two ORIGINAL fit evidence groups")
    order = groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed))]
    check_count = max(1, min(len(groups) - 1, round(0.2 * len(groups))))
    check = outer_fit & torch.isin(vectors["evidence_row_index"], order[:check_count])
    return outer_fit & ~check, check


def pieces(pairs, mask, chunk):
    selected = mask.nonzero().flatten()
    for x, y in pairs:
        require(x.shape[0] == y.shape[0] == len(mask), "paired native row shape mismatch")
        for index in selected.split(chunk):
            if not len(index):
                continue
            a, b = x[index].double(), y[index].double()
            require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "nonfinite native fit/held rows")
            yield a, b


def gram(pairs, mask, chunk):
    x, y = pairs[0]
    require(bool(mask.any()), "empty fit partition")
    a = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    b = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64)
    for design, target in pieces(pairs, mask, chunk):
        a.addmm_(design.T, design)
        b.addmm_(design.T, target)
    return a, b


def solve(a: torch.Tensor, b: torch.Tensor, ridge_rel: float) -> tuple[torch.Tensor, float]:
    scale = float(a.diagonal().mean())
    require(math.isfinite(scale) and scale > 0, "native fit has no finite design energy; no identity fallback")
    lam = scale * ridge_rel
    shifted = a.clone()
    shifted.diagonal().add_(lam)
    factor, info = torch.linalg.cholesky_ex(shifted)
    require(int(info) == 0, "ridge factorization failed; regularization is not silently increased")
    weight = torch.cholesky_solve(b, factor)
    require(bool(torch.isfinite(weight).all()), "nonfinite chart solve")
    return weight, lam


def scores(pairs, mask, weight, chunk):
    """The reviewed mediafold FP64 diagnostics, accumulated in bounded row chunks."""
    require(bool(mask.any()), "empty score partition")
    count = int(mask.sum()) * len(pairs)
    width = pairs[0][1].shape[1]
    mean = torch.zeros(width, dtype=torch.float64)
    for _, target in pieces(pairs, mask, chunk):
        mean += target.sum(0)
    mean /= count
    error = energy = variance = 0.0
    cosines = []
    weight64 = weight.double()
    for design, target in pieces(pairs, mask, chunk):
        prediction = design @ weight64
        require(bool(torch.isfinite(prediction).all()), "nonfinite chart prediction")
        error += float((prediction - target).square().sum())
        energy += float(target.square().sum())
        variance += float((target - mean).square().sum())
        cosines.append((prediction * target).sum(-1) /
                       (prediction.norm(dim=-1).clamp_min(1e-30) * target.norm(dim=-1).clamp_min(1e-30)))
    cosines = torch.cat(cosines)
    ordered = cosines.sort().values
    return {"rel_l2": math.sqrt(error) / max(math.sqrt(energy), 1e-30), "cos": float(cosines.mean()),
            "cos_p05": float(ordered[min(count - 1, math.floor(0.05 * count))]),
            "r2": 1.0 - error / max(variance, 1e-30), "rows": count, "width": width}


@torch.no_grad()
def fit_direction(pairs, vectors, *, seed=0, chunk=512):
    fit = vectors["held"] == 0
    held = ~fit
    inner_fit, inner_check = split_fit_groups(vectors, seed)
    a, b = gram(pairs, inner_fit, chunk)
    trail = []
    best = None
    for ridge_rel in legacy.linalg.LAMBDA_LADDER:
        weight, lam = solve(a, b, ridge_rel)
        measured = scores(pairs, inner_check, weight, chunk)
        trail.append({"ridge_rel": ridge_rel, "ridge_lambda": lam, "inner_check": measured})
        if best is None or measured["rel_l2"] < best[0]:
            best = measured["rel_l2"], ridge_rel
        del weight
    del a, b
    a, b = gram(pairs, fit, chunk)
    weight, lam = solve(a, b, best[1])
    del a, b
    # Installed C10 chart loader consumes F32. Measure those exact stored coefficients too.
    stored = weight.float()
    require(bool(torch.isfinite(stored).all()), "F32 chart narrowing overflow")
    zero = torch.zeros_like(weight)
    notes = {"constraint": "through-origin; no learned affine offset", "fit_bias": False,
        "ridge_rel_selected": best[1], "ridge_lambda": lam, "lambda_ladder": trail,
        "lambda_selection": "original fit evidence groups only; outer held groups never select a rung",
        "inner_fit_groups": torch.unique(vectors["evidence_row_index"][inner_fit]).tolist(),
        "inner_check_groups": torch.unique(vectors["evidence_row_index"][inner_check]).tolist(),
        "fit_rows": int(fit.sum()) * len(pairs), "held_rows": int(held.sum()) * len(pairs),
        "design_width": pairs[0][0].shape[1],
        "underdetermined": int(fit.sum()) * len(pairs) < pairs[0][0].shape[1],
        "fp64_fit": scores(pairs, fit, weight, chunk), "fp64_held": scores(pairs, held, weight, chunk),
        "stored_f32_fit": scores(pairs, fit, stored, chunk), "stored_f32_held": scores(pairs, held, stored, chunk),
        "zero_held": scores(pairs, held, zero, chunk), "regularization_escalated": False,
        "held_threshold_policy": None, "production_install_eligible": False}
    return stored, notes


def endpoint(ref, family, layer, boundary):
    return {"family": family, "layer": layer, "boundary": boundary, "capture_receipt": ref}


def run(plan_path: Path, plan_sha: str, out: Path, *, synthetic_fixture_root: Path | None = None):
    pins = k3.Pins()
    plan = pins.json({"path": str(plan_path), "sha256": plan_sha})
    require(plan.get("schema") == PLAN_SCHEMA and not out.exists(), "plan schema invalid or output is not fresh")
    fixture = synthetic_fixture_root is not None
    require(plan.get("fixture") is fixture, "explicit fixture/production mode mismatch")
    torch.set_num_threads(host.integer(plan["threads"], "threads", minimum=1))
    chunk = host.integer(plan["chunk"], "chunk", minimum=1)
    seed = host.integer(plan["seed"], "seed")
    evidence, boundary = pins.json(plan["evidence"]), pins.json(plan["boundary_map"])
    vectors = original_vectors(host.evidence_rows(evidence, boundary, plan["evidence"]["sha256"]))
    layer = host.integer(plan["host_layer"], "host_layer")
    hc, hp = read_capture(plan["host_capture"], "host", layer, pins, vectors,
            plan["evidence"]["sha256"], plan["boundary_map"]["sha256"], fixture)
    imported = pins.json(plan["host_import_receipt"])
    require(imported.get("schema") == "chimera_omni.import.qwen35.v1" and
            (hc.get("host_receipt") or {}).get("sha256") == plan["host_import_receipt"]["sha256"],
            "host source gate must be the actual pinned native import, not a preflight")
    checkpoint = imported.get("checkpoint", {})
    for key in ("config_sha256", "index_sha256"):
        require(isinstance(hc.get(key), str) and bool(host.SHA.fullmatch(hc[key])) and checkpoint.get(key) == hc[key],
                "host import/capture checkpoint identity mismatch")
    require(imported.get("output_gate_waiver") is None and checkpoint.get("directory_shard_files_match_index") is True,
            "host native import has an unverified output gate or unmatched shard set")
    ks, source_records = [], []
    provenance = {"schema": installer.NATIVE_CHART_SCHEMA, "fixture": fixture,
                  "maps": {"C_out": {"domain": [], "codomain": []}, "C_in": {"domain": [], "codomain": []}}}
    layers = [host.integer(item["layer"], "K3 layer", minimum=1) for item in plan["k3_captures"]]
    require(bool(layers) and layers == sorted(set(layers)), "K3 pool must be nonempty, sorted and unique")
    contexts = []
    for item, source_layer in zip(plan["k3_captures"], layers):
        kc, kp = read_capture(item["capture"], "k3", source_layer, pins, vectors,
                plan["evidence"]["sha256"], plan["boundary_map"]["sha256"], fixture)
        accepted = pins.json(item["accepted_receipt"])
        require(accepted.get("schema") == k3.native.RECEIPT_SCHEMA and
                (kc.get("accepted_receipt") or {}).get("sha256") == item["accepted_receipt"]["sha256"] and
                kc.get("accepted_resume_context") == accepted.get("resume_context") and
                k3.acceptance(accepted.get("layers", {}).get(str(source_layer), {}).get("acceptance", {})) and
                accepted["layers"][str(source_layer)]["ffn"]["sha256"] == kc["installed_ffn_sha256"][str(source_layer)],
                "K3 accepted source gate/FFN identity mismatch")
        contexts.append(kc["accepted_resume_context"])
        ks.append(kp)
        source_records.append({"layer": source_layer, "capture": item["capture"], "accepted_receipt": item["accepted_receipt"],
                               "installed_ffn_sha256": kc["installed_ffn_sha256"][str(source_layer)]})
        for name, hb, kb in (("C_out", "ffn_input", "ffn_input"), ("C_in", "ffn_output", "ffn_output")):
            he, ke = endpoint(plan["host_capture"], "host", layer, hb), endpoint(item["capture"], "k3", source_layer, kb)
            provenance["maps"][name]["domain"].append(he if name == "C_out" else ke)
            provenance["maps"][name]["codomain"].append(ke if name == "C_out" else he)
    require(all(c == contexts[0] for c in contexts), "pooled sources have different original native contexts")
    h, p = hp["ffn_input"].shape[1], ks[0]["ffn_input"].shape[1]
    require(all(x["ffn_input"].shape[1] == p for x in ks), "pooled K3 widths differ")
    if fixture:
        root = synthetic_fixture_root.resolve(strict=True)
        require(max(h, p) <= 128 and len(vectors["held"]) <= 4096 and layer < 8 and max(layers) < 8 and
                out.resolve().is_relative_to(root) and all(path.is_relative_to(root) for path in pins.records),
                "fixture data/output must be tiny and confined to its explicit root")
    else:
        # Independently consume the exact contract production C10 understands.
        installer.native_chart_boundaries({"arms": {"ridge": {"native_boundary_provenance": provenance}}},
                str(out / f"L{layer:02d}" / "candidate_layer.json"), layer, "ridge", trunk_hidden=h, k3_hidden=p)
    overhead = 8 * (5 * max(h, p)**2 + 5 * h * p + chunk * (h + p) * 8)
    require(overhead <= host.integer(plan["max_working_bytes"], "max_working_bytes", minimum=1),
            "projected numerical workspace exceeds explicit budget (capture mappings excluded)")
    code = [Path(__file__), Path(host.__file__), Path(k3.__file__), Path(installer.__file__),
            Path(legacy.__file__), Path(legacy.linalg.__file__)]
    for path in code:
        pins.file(path.resolve(), host.digest(path))
    pins.stable()
    print(f"[c17-native-fit] host L{layer:02d} / K3 {layers}: original paired groups verified; fitting two constrained maps", flush=True)
    c_out, out_notes = fit_direction([(hp["ffn_input"], x["ffn_input"]) for x in ks], vectors, seed=seed, chunk=chunk)
    c_in, in_notes = fit_direction([(x["ffn_output"], hp["ffn_output"]) for x in ks], vectors, seed=seed, chunk=chunk)
    pins.stable()
    out.mkdir(parents=True, exist_ok=False)
    folder = out / f"L{layer:02d}"
    folder.mkdir()
    chart = folder / "ridge.safetensors"
    save_file({"C_out": c_out, "C_in": c_in, "C_out_bias": torch.zeros(p), "C_in_bias": torch.zeros(h)}, str(chart),
              metadata={"schema": SCHEMA, "status": "candidate_only", "fixture": str(fixture).lower()})
    arm = {"path": str(chart), "sha256": host.digest(chart), "native_boundary_provenance": provenance,
           "constraints": {"C_out_bias": "fixed zero during fit", "C_in_bias": "fixed zero during fit"},
           "C_out": out_notes, "C_in": in_notes, "production_install_eligible": False}
    candidate = {"schema": SCHEMA, "fixture": fixture, "host_layer": layer, "k3_layers": layers,
        "status": "candidate_only", "production_install_eligible": False, "gradients_or_training": False,
        "host_index_sha256": hc["index_sha256"], "host_config_sha256": hc["config_sha256"],
        "host_import_receipt": plan["host_import_receipt"], "host_capture": plan["host_capture"], "k3_sources": source_records,
        "evidence_sha256": plan["evidence"]["sha256"], "boundary_sha256": plan["boundary_map"]["sha256"],
        "plan": {"path": str(plan_path), "sha256": plan_sha}, "arms": {"ridge": arm},
        "code_sha256": {str(path.resolve()): host.digest(path) for path in code},
        "numerics": {"dtype_solve": "float64", "dtype_stored": "float32", "threads": torch.get_num_threads(),
                     "chunk": chunk, "seed": seed, "torch": str(torch.__version__), "workspace_upper_estimate": overhead},
        "acceptance": {"source_import_integrity_verified": True, "original_groups_preserved": True,
            "post_s3_host_behavioral_acceptance": "unavailable; not inferred from import/preflight",
            "chart_held_threshold_policy": None, "exact_installed_computation_gate": "not_run",
            "all_source_child_capability": "not_demonstrated"},
        "limitations": ["independently fitted directed maps are not proved inverse or chart-law preserving",
            "zero chart biases remove affine-offset loss only; downstream installation remains an unmeasured approximation",
            "no layer.json is emitted: an explicit accepted installed-computation gate is required before promotion"]}
    pins.stable()
    k3.native.write_receipt(str(folder / "candidate_layer.json"), candidate)
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-fixture-root", type=Path)
    args = parser.parse_args(argv)
    run(args.plan, args.plan_sha256, args.out, synthetic_fixture_root=args.synthetic_fixture_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
