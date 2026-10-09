"""FULL21 structural contract. This module needs only the standard library.

No model/capture/quality result is implied by a validated metadata contract.
"""
from __future__ import annotations

import math
import re

SCHEMA = "chimera.hy3d-full21-native-branch-union.v1"
FOLD_SCHEMA = "chimera.hy3d-full21-finite-component.v1"
CAPTURE_SCHEMA = "chimera.hy3d-full21-sealed-block-capture.v1"
POLICY_SCHEMA = "chimera.hy3d-full21-fit-only-policy.v1"
EXPERTS = 8
SOURCE_PRESERVATION = {
    "model.blocks.0..14": "native biased exact GELU dense MLPs; all attention/norm/skip paths retained",
    "model.blocks.15..20": "native router plus eight routed FFNs and shared FFN; no source branch omitted",
    "model.non_ffn": "time/input/output embeddings, attention, norms and authentic skip paths retained",
    "vae.encoder": "complete ShapeVAE encoder retained as an explicit source component",
    "vae.decoder": "post_kl, complete transformer and geo_decoder retained as explicit source components",
    "conditioner": "complete native image conditioner retained as an explicit source component",
    "absorption": "none; no SDF absorption or all-source child claim",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def block_contract(index):
    require(type(index) is int and 0 <= index < 21, "native block index must be 0..20")
    prefix = f"model.blocks.{index}."
    if index < 15:
        branches = [{"id": "dense", "source_prefix": prefix + "mlp.",
                     "first": "fc1", "second": "fc2", "route": "all_tokens"}]
    else:
        branches = [{"id": f"expert{i}", "source_prefix": prefix + f"moe.experts.{i}.net.",
                     "first": "0.proj", "second": "2", "route": i} for i in range(EXPERTS)]
        branches.append({"id": "shared", "source_prefix": prefix + "moe.shared_experts.net.",
                         "first": "0.proj", "second": "2", "route": "all_tokens"})
    return {"block_index": index, "skip_required": index > 10,
            "use_moe": index >= 15, "branches": branches,
            "routing": {"experts": 8, "top_k": 2, "score": "softmax",
                        "renormalize_topk": False, "shared_mass": 1.0} if index >= 15 else None,
            "native_forward": True, "source_preservation": SOURCE_PRESERVATION}


def biased_parameters(width, inner):
    require(type(width) is int and type(inner) is int and width > 0 and inner > 0,
            "positive integer biased MLP geometry required")
    return inner * width + inner + width * inner + width


def maximum_fourfold_keep(width, inner):
    return (biased_parameters(width, inner) // 4 - width) // (2 * width + 1)


def union_manifest(index, widths):
    contract = block_contract(index)
    branches, offset = [], 0
    require(set(widths) == {b["id"] for b in contract["branches"]}, "dense union dropped/added a native source branch")
    for branch in contract["branches"]:
        width = widths[branch["id"]]
        require(type(width) is int and width > 0, "positive union branch width required")
        branches.append({**branch, "offset": offset, "width": width,
                         "bias_semantics": "native output bias multiplied by original route mass"
                         if type(branch["route"]) is int else "native unscaled output bias"})
        offset += width
    return {"schema": SCHEMA, "block_index": index, "branches": branches,
            "union_width": offset, "routing": contract["routing"],
            "execution": "unchanged_native_forward_with_all_native_branches",
            "union_semantics": "concatenated branch feature bank with independent native output biases; no cross-branch feature mixing",
            "inference_buffers": "native branch grouping/scatter/accumulation; not one regrouped GEMM",
            "source_preservation": SOURCE_PRESERVATION}


def validate_union(body):
    require(isinstance(body, dict) and body.get("schema") == SCHEMA, "wrong native dense union schema")
    contract = block_contract(body.get("block_index"))
    rows = body.get("branches")
    require(isinstance(rows, list) and len(rows) == len(contract["branches"]), "native branch inventory incomplete")
    widths = {}
    for actual, expected in zip(rows, contract["branches"]):
        require(isinstance(actual, dict) and all(actual.get(k) == v for k, v in expected.items()),
                "native branch identity/order/source changed")
        widths[expected["id"]] = actual.get("width")
    require(body == union_manifest(contract["block_index"], widths), "dense union offsets/router/bias contract changed")
    return body


def validate_policy(body, index):
    expected = {"schema", "block_index", "keep", "selection", "ridge_ladder", "fit_objects", "held_objects",
                "max_fit_rows", "max_union_width", "max_feature_bytes", "max_statistics_bytes", "row_chunk",
                "numeric_policy", "unvisited_branch"}
    require(isinstance(body, dict) and set(body) == expected and body["schema"] == POLICY_SCHEMA,
            "exact explicit FULL21 policy fields required")
    require(type(body["block_index"]) is int and body["block_index"] == index, "policy belongs to another block")
    for name in ("keep", "max_fit_rows", "max_union_width", "max_feature_bytes", "max_statistics_bytes", "row_chunk"):
        require(type(body[name]) is int and body[name] > 0, "positive policy bound required: " + name)
    require(type(body["fit_objects"]) is type(body["held_objects"]) is int and
            body["fit_objects"] == body["held_objects"] == 2, "exact two FIT/two HELD source objects required")
    require(body["max_fit_rows"] <= 8194 and body["max_union_width"] <= 73728 and
            body["max_feature_bytes"] <= 1 << 30 and body["max_statistics_bytes"] <= 1 << 30 and
            body["row_chunk"] <= body["max_fit_rows"], "policy exceeds bounded single-block implementation contract")
    require(body["selection"] == "fit_only_uncentered_native_feature_energy_stable_index_ties",
            "unsupported preregistered bounded selection")
    require(body["numeric_policy"] == "finite_diagnostics_only_no_numeric_veto" and
            body["unvisited_branch"] == "preserve_complete_source_branch_and_report",
            "finite diagnostics/source branch retention required")
    ladder = body["ridge_ladder"]
    require(isinstance(ladder, list) and ladder and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in ladder),
            "explicit positive finite FIT-only ridge ladder required")
    return body


def default_policy(index):
    block_contract(index)
    return {"schema": POLICY_SCHEMA, "block_index": index, "keep": 2047,
            "selection": "fit_only_uncentered_native_feature_energy_stable_index_ties",
            "ridge_ladder": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            "fit_objects": 2, "held_objects": 2, "max_fit_rows": 8194,
            "max_union_width": 73728, "max_feature_bytes": 1 << 30,
            "max_statistics_bytes": 512 << 20, "row_chunk": 256,
            "numeric_policy": "finite_diagnostics_only_no_numeric_veto",
            "unvisited_branch": "preserve_complete_source_branch_and_report"}


def source_groups(names):
    """Partition every parent tensor exactly; missing VAE half is a hard error."""
    names = list(names)
    require(len(names) == len(set(names)), "duplicate source tensor identity")
    groups = {key: [] for key in ("dit", "shape_vae_encoder", "shape_vae_decoder", "conditioner")}
    for name in names:
        if name.startswith("model."):
            group = "dit"
        elif name.startswith(("vae.encoder.", "vae.pre_kl.")):
            group = "shape_vae_encoder"
        elif name.startswith(("vae.post_kl.", "vae.transformer.", "vae.geo_decoder.")):
            group = "shape_vae_decoder"
        elif name.startswith("conditioner."):
            group = "conditioner"
        else:
            raise ValueError("unaccounted native parent tensor: " + name)
        groups[group].append(name)
    require(all(groups.values()), "complete DiT/ShapeVAE encoder/decoder/conditioner source groups required")
    require(any(n.startswith("vae.encoder.") for n in groups["shape_vae_encoder"]) and
            all(any(n.startswith(prefix) for n in groups["shape_vae_decoder"])
                for prefix in ("vae.post_kl.", "vae.transformer.", "vae.geo_decoder.")), "complete ShapeVAE source paths required")
    for names in groups.values():
        names.sort()
    return groups
