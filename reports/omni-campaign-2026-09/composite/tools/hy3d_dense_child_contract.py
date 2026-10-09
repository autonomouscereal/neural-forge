"""Explicit approximate dense-child contract, separate from native MoE unions.

This contract does not claim source-native fidelity or retained intelligence.
The native teacher executes all original branches. The child replaces only the
FFN computation; source attention/time/skip/VAE/conditioner are retained.
"""
from __future__ import annotations
import math
import json
from hy3d_full21_contract import block_contract, require

SCHEMA = "chimera.hy3d-full21-dense-child.v1"
POLICY_SCHEMA = "chimera.hy3d-full21-dense-child-fit-policy.v1"
NUMERIC_POLICY = "chimera_omni.owner_finite_diagnostics_only.2026-10-06.v1"
STATE_NAMES = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def memory_plan(width, keep, row_chunk, branch_inner):
    for value in (width, keep, row_chunk, branch_inner):
        require(type(value) is int and value > 0, "positive integer memory geometry required")
    k = keep + 1
    # Conservative live FP64 statistics/factor/solution/prior copies plus
    # selected-feature and largest single source-branch activation chunks.
    statistics = 8 * (6 * k * k + 6 * width * k + 2 * width)
    chunks = row_chunk * (2 * branch_inner + 2 * keep + 8 * (2 * k + 2 * width))
    return {"statistics_bytes": statistics, "chunk_bytes": chunks,
            "working_bytes": statistics + chunks, "union_gram_allocated": False}


def default_policy(index):
    block_contract(index)
    return {"schema": POLICY_SCHEMA, "block_index": index, "keep": 2047,
            "row_chunk": 128, "max_working_bytes": 768 << 20,
            "ridge_ladder": [1e-6, 1e-4, 1e-2, 1, 100],
            "selection": "fit_only_source_unit_output_energy_stable_global_index",
            "numeric_policy": NUMERIC_POLICY, "fit_objects": 2, "held_objects": 2}


def validate_policy(body, index):
    require(isinstance(body, dict) and set(body) == set(default_policy(index)), "exact dense-child policy fields required")
    require(body["schema"] == POLICY_SCHEMA and type(body["block_index"]) is int and body["block_index"] == index,
            "policy block/schema mismatch")
    for name in ("keep", "row_chunk", "max_working_bytes"):
        require(type(body[name]) is int and body[name] > 0, "positive explicit bound required: " + name)
    require(body["keep"] <= 8192 and body["row_chunk"] <= 512, "bounded dense-child geometry required")
    require(body["fit_objects"] == body["held_objects"] == 2 and
            body["selection"] == default_policy(index)["selection"] and body["numeric_policy"] == NUMERIC_POLICY,
            "FIT-only partition and diagnostic-only owner policy required")
    ladder = body["ridge_ladder"]
    require(isinstance(ladder, list) and 1 <= len(ladder) <= 16 and all(type(v) in (int, float) and
            math.isfinite(v) and v > 0 for v in ladder), "positive finite bounded ridge ladder required")
    require(memory_plan(2048, body["keep"], body["row_chunk"], 8192)["working_bytes"] <= body["max_working_bytes"],
            "declared working cap below conservative plan")
    return body


def child_contract(index, width, keep, selected, branch_widths):
    source = block_contract(index)
    require(type(width) is int and width == 2048 and type(keep) is int and 0 < keep <= 8192, "actual audited dense child dimensions required")
    require(isinstance(branch_widths, list) and len(branch_widths) == len(source["branches"]) and
            all(type(v) is int and v == 8192 for v in branch_widths), "complete source branch basis inventory required")
    require(isinstance(selected, list) and len(selected) == keep and len(set(selected)) == keep and
            selected == sorted(selected) and all(type(v) is int and 0 <= v < sum(branch_widths) for v in selected),
            "unique bounded source-unit identities required")
    return {"schema": SCHEMA, "block_index": index, "width": width, "keep": keep,
            "activation": "GELU approximate=none", "state_names": sorted(STATE_NAMES),
            "selected_source_units": selected, "source_branch_widths": branch_widths,
            "source_branch_ids": [r["id"] for r in source["branches"]],
            "skip_required": source["skip_required"], "teacher_routing": source["routing"],
            "teacher": "unchanged native block; original top2 unnormalized masses, routed output biases and unscaled shared branch",
            "child": "one static biased dense GELU FFN; no router, expert dispatch or shared branch at inference",
            "basis": "selected exact source first-layer units; output readout fitted to full native FFN output",
            "held_used_for_selection_or_refit": False, "numeric_policy": NUMERIC_POLICY,
            "source_native_fidelity_claim": False, "retained_intelligence_claim": False}


def validate_child(body):
    require(isinstance(body, dict), "dense child metadata must be an object")
    expected = child_contract(body.get("block_index"), body.get("width"), body.get("keep"),
                              body.get("selected_source_units"), body.get("source_branch_widths"))
    require(body == expected, "dense-child topology/activation/routing/basis contract changed")
    return body


def coverage21(rows):
    require(isinstance(rows, list) and len(rows) == 21, "all21 child receipts required")
    require([r.get("block_index") for r in rows] == list(range(21)), "exact ordered unique block0..20 coverage required")
    for row in rows:
        require(validate_child(row.get("dense_child"))["block_index"] == row["block_index"], "inner/outer block identity mismatch")
    identities = {json.dumps(r.get("source_identity"), sort_keys=True, allow_nan=False) for r in rows}
    require(len(identities) == 1 and "null" not in identities, "all21 must bind one actual source")
    require(all(r.get("fixture") is False and r.get("actual_source_forward_completed") is True and
                r.get("candidate_written") is True for r in rows), "actual source replay and saved children required")
    fit = rows[0].get("fit_object_sha256")
    held = rows[0].get("held_object_sha256")
    require(isinstance(fit, list) and len(fit) == 2 and isinstance(held, list) and len(held) == 2 and
            len(set(fit + held)) == 4 and all(r.get("fit_object_sha256") == fit and
            r.get("held_object_sha256") == held for r in rows), "one disjoint source object split required across all21")
    return rows
