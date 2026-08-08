#!/usr/bin/env python3
"""Stage-A attribution: extract the arm-A per-layer solver/validation profile.

Reads the sealed arm-A closed-loop telemetry that was copied off the server
read-only (`remote/closed_loop_attempt0.json`, taken from
`published-child/provenance/run.json` -> `attempts[0].transaction_report
.closed_loop`) plus the run receipt, and writes:

* `per_layer_profile.json` — one record per folded layer, every numeric field
  the item-8 telemetry actually carries.
* `per_layer_profile.md`  — the rendered attribution table.
* `branch_decomposition.json` — the operator-vs-MLP endpoint split for Q2.

Nothing is written back to the sealed run directory; this script only reads the
local copies under `remote/`.
"""

from __future__ import annotations

import json
import math
import os


HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = os.path.join(HERE, "remote")
CLOSED_LOOP = os.path.join(REMOTE, "closed_loop_attempt0.json")
RECEIPT = os.path.normpath(
    os.path.join(HERE, "..", "armA-review-20260808", "receipt.json")
)

ROLES = ("mlp_gate", "mlp_up", "mlp_down")


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _prior_reports(metadata):
    """The Lfm2PriorCorrectionReport list the MLP compile emitted, by role."""
    raw = metadata.get("mlp_prior_correction_reports_json")
    if not raw:
        return {}
    return {report["role"]: report for report in json.loads(raw)}


def _gate_point(metadata):
    raw = metadata.get("mlp_gate_operating_point_json")
    return json.loads(raw) if raw else {}


def _requested_ratio(report):
    """Correction norm the solver asked for, in units of the prior's norm.

    The clamp writes back ``correction_norm = update_scale * raw_norm`` and
    ``update_scale = min(1, max_prior_delta_ratio * prior_norm / raw_norm)``,
    so the pre-clamp request is recoverable exactly.
    """
    scale = report.get("update_scale")
    norm = report.get("correction_norm")
    prior = report.get("prior_norm")
    if not scale or not prior:
        return None
    return (norm / scale) / prior


def main() -> int:
    closed = _load(CLOSED_LOOP)
    receipt = _load(RECEIPT)
    telemetry = receipt["fold"]["closed_loop_depth_telemetry"]
    decisions = {
        int(entry["layer_index"]): entry
        for entry in telemetry["per_layer_validation_decisions"]
    }

    rows = []
    for report in closed["layer_reports"]:
        index = int(report["scope"]["layer_index"])
        producer_md = report["producer"]["compiled"]["metadata"]
        consumer_md = report["consumer"]["compiled"]["metadata"]
        downstream_md = report["downstream"]["compiled"]["metadata"]
        priors = _prior_reports(downstream_md)
        gate = _gate_point(downstream_md)
        metrics = decisions[index]["metrics"]

        gate_report = priors.get("mlp_gate", {})
        conditioning = gate_report.get("solver_conditioning") or {}

        record = {
            "layer_index": index,
            "operator_kind": producer_md.get("operator_kind"),
            "mlp_intermediate_channels": gate.get("channels"),
            # --- solver conditioning (the mlp_gate/mlp_up normal system) ---
            "solver_path": conditioning.get("solver_path"),
            "silent_gelsd_fallback_taken": conditioning.get(
                "silent_gelsd_fallback_taken"
            ),
            "conditioning_measured_on": conditioning.get("conditioning_measured_on"),
            "system_condition_number": conditioning.get("system_condition_number"),
            "system_singular_value_maximum": conditioning.get(
                "system_singular_value_maximum"
            ),
            "system_singular_value_minimum": conditioning.get(
                "system_singular_value_minimum"
            ),
            # --- prior-correction reports, per role ---
            "roles": {
                role: {
                    "rows": priors.get(role, {}).get("rows"),
                    "input_width": priors.get(role, {}).get("input_width"),
                    "output_width": priors.get(role, {}).get("output_width"),
                    "underdetermined": priors.get(role, {}).get("underdetermined"),
                    "prior_provided": priors.get(role, {}).get("prior_provided"),
                    "prior_norm": priors.get(role, {}).get("prior_norm"),
                    "correction_norm": priors.get(role, {}).get("correction_norm"),
                    "update_scale": priors.get(role, {}).get("update_scale"),
                    "clamp_bound": (
                        None
                        if priors.get(role, {}).get("update_scale") is None
                        else priors[role]["update_scale"] < 1.0
                    ),
                    "requested_correction_over_prior": _requested_ratio(
                        priors.get(role, {})
                    ),
                    "input_error_covariance_applied": priors.get(role, {}).get(
                        "input_error_covariance_applied"
                    ),
                    "solver_conditioning_present": bool(
                        priors.get(role, {}).get("solver_conditioning")
                    ),
                }
                for role in ROLES
            },
            # --- MLP branch fit quality ---
            "mlp_candidate_fit_relative_rmse": downstream_md.get(
                "candidate_fit_relative_rmse"
            ),
            "mlp_current_fit_relative_rmse": downstream_md.get(
                "current_fit_relative_rmse"
            ),
            "mlp_input_source": downstream_md.get("mlp_input_source"),
            # --- gate operating point (Q2 branch evidence) ---
            "gate_max_abs_channel_mean_drift": (gate.get("drift") or {}).get(
                "maximum_absolute_channel_mean_drift"
            ),
            "gate_max_abs_channel_sigma_drift": (gate.get("drift") or {}).get(
                "maximum_absolute_channel_standard_deviation_drift"
            ),
            "gate_correction_applied": gate.get("correction_applied"),
            "gate_child_mean_of_channel_sigma": (gate.get("child") or {}).get(
                "mean_of_channel_standard_deviation"
            ),
            "gate_parent_mean_of_channel_sigma": (gate.get("parent") or {}).get(
                "mean_of_channel_standard_deviation"
            ),
            "gate_rows": gate.get("rows"),
            # --- operator branch (what the telemetry does and does not carry) ---
            "operator_consumer_mode": consumer_md.get("attention_consumer_mode"),
            "operator_value_context_solver": consumer_md.get(
                "attention_value_context_solver"
            ),
            "operator_fit_nonincreasing": consumer_md.get("attention_fit_nonincreasing"),
            "short_conv_writer_topology": producer_md.get("short_conv_writer_topology"),
            "short_conv_factor_relative_frobenius_error": consumer_md.get(
                "short_conv_factor_relative_frobenius_error"
            ),
            "operator_prior_correction_reports_present": bool(
                producer_md.get("mlp_prior_correction_reports_json")
                or consumer_md.get("mlp_prior_correction_reports_json")
            ),
            "operator_solver_conditioning_present": any(
                "conditioning" in key or "condition_number" in key
                for key in tuple(producer_md) + tuple(consumer_md)
            ),
            # --- per-layer validation (composed, held out) ---
            "validation_accepted": decisions[index]["accepted"],
            "post_layer_candidate_relative_rmse": metrics.get(
                "candidate_relative_rmse"
            ),
            "post_layer_baseline_relative_rmse": metrics.get(
                "baseline_relative_rmse"
            ),
            "post_layer_relative_rmse_improvement": metrics.get(
                "relative_rmse_improvement"
            ),
            "post_layer_candidate_mean_cosine": metrics.get(
                "candidate_mean_cosine_similarity"
            ),
            "terminal_relative_rmse": metrics.get("terminal_relative_rmse"),
            "terminal_mean_cosine": metrics.get("terminal_mean_cosine_similarity"),
            "terminal_relative_rmse_regression": metrics.get(
                "terminal_relative_rmse_regression"
            ),
            "terminal_rows": metrics.get("terminal_rows"),
            "terminal_free_run_tripwire_evaluated": metrics.get(
                "terminal_free_run_tripwire_evaluated"
            ),
        }
        rows.append(record)

    rows.sort(key=lambda entry: entry["layer_index"])
    out = {
        "schema": "neural-forge-stage-a-attribution-profile-v1",
        "source_run": (
            "/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/"
            "catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold"
        ),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "configuration_sha256": receipt.get("configuration_sha256"),
        "solver_conditioning_diagnostics_enabled": telemetry.get(
            "solver_conditioning_diagnostics_enabled"
        ),
        "gate_operating_point_diagnostics_enabled": telemetry.get(
            "gate_operating_point_diagnostics_enabled"
        ),
        "input_error_covariance_enabled": telemetry.get(
            "input_error_covariance_enabled"
        ),
        "margin_row_weights_enabled": telemetry.get("margin_row_weights_enabled"),
        "layers": rows,
    }
    with open(
        os.path.join(HERE, "per_layer_profile.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(out, handle, indent=1)

    # ----- rendered table -----
    def fmt(value, spec="{:.6f}"):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
                return f"{value:.3e}"
            return spec.format(value)
        return str(value)

    lines = []
    lines.append(
        "| L | operator | MLP h | solve rows | cond(XᵀX) | σ_max | σ_min | "
        "solver path | gelsd | update_scale gate/up/down | requested Δ/prior "
        "gate/up/down | MLP fit relL2 | post-layer relL2 | post-layer cos | "
        "terminal relL2 |"
    )
    lines.append("|" + "---|" * 15)
    for record in rows:
        roles = record["roles"]
        scales = "/".join(
            fmt(roles[role]["update_scale"], "{:.3f}") for role in ROLES
        )
        requested = "/".join(
            fmt(roles[role]["requested_correction_over_prior"], "{:.1f}")
            for role in ROLES
        )
        lines.append(
            "| {L} | {op} | {h} | {rows} | {cond} | {smax} | {smin} | {path} | "
            "{gelsd} | {scales} | {req} | {fit} | {post} | {cos} | {term} |".format(
                L=record["layer_index"],
                op=record["operator_kind"],
                h=record["mlp_intermediate_channels"],
                rows=roles["mlp_gate"]["rows"],
                cond=fmt(record["system_condition_number"]),
                smax=fmt(record["system_singular_value_maximum"]),
                smin=fmt(record["system_singular_value_minimum"], "{:.6f}"),
                path=record["solver_path"] or "—",
                gelsd=fmt(record["silent_gelsd_fallback_taken"]),
                scales=scales,
                req=requested,
                fit=fmt(record["mlp_candidate_fit_relative_rmse"], "{:.4f}"),
                post=fmt(record["post_layer_candidate_relative_rmse"], "{:.4f}"),
                cos=fmt(record["post_layer_candidate_mean_cosine"], "{:.4f}"),
                term=fmt(record["terminal_relative_rmse"], "{:.4f}"),
            )
        )
    with open(
        os.path.join(HERE, "per_layer_profile.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")

    # ----- branch decomposition (Q2) -----
    #
    # What the run actually records per layer:
    #   baseline (held out)     post-layer relL2 BEFORE any of this layer's
    #                           three installs (producer/consumer/downstream),
    #                           captured in the producer stage.
    #   current_fit (fit rows)  post-layer relL2 with the NEW operator installed
    #                           and the OLD (structural-transport) MLP still in
    #                           place — the MLP cases are built after the
    #                           operator installs.
    #   candidate_fit (fit)     post-layer relL2 after the MLP refit too.
    #   candidate (held out)    same endpoint as candidate_fit, held-out rows.
    #
    # So `current_fit - candidate_fit` is exactly what refitting the MLP bought
    # at that boundary, and `baseline - candidate` is what the whole layer
    # bought.  The operator share is the remainder; it mixes a fit-row delta
    # with a held-out delta, so the fit/held-out gap at the shared candidate
    # point is carried alongside it as the error bar.
    branch = {
        "schema": "neural-forge-stage-a-branch-decomposition-v1",
        "note": (
            "The item-8 telemetry instruments the downstream (MLP) compile only. "
            "The producer/consumer (operator) compiles emit no prior-correction "
            "report and no conditioning block into the run record, and the MLP "
            "compiler's own branch-local `residual_endpoint_relative_rmse` is "
            "computed and discarded. The decomposition below is therefore a "
            "stage-delta decomposition of the composed post-layer endpoint, not "
            "a per-branch endpoint split."
        ),
        "operator_branch_prior_correction_reports_present": any(
            record["operator_prior_correction_reports_present"] for record in rows
        ),
        "operator_branch_conditioning_present": any(
            record["operator_solver_conditioning_present"] for record in rows
        ),
        "layers": [
            {
                "layer_index": record["layer_index"],
                "operator_kind": record["operator_kind"],
                "mlp_intermediate_channels": record["mlp_intermediate_channels"],
                "baseline_heldout": record["post_layer_baseline_relative_rmse"],
                "current_fit_operator_installed_old_mlp": record[
                    "mlp_current_fit_relative_rmse"
                ],
                "candidate_fit_all_installed": record[
                    "mlp_candidate_fit_relative_rmse"
                ],
                "candidate_heldout_all_installed": record[
                    "post_layer_candidate_relative_rmse"
                ],
                "mlp_stage_delta_fit": (
                    record["mlp_current_fit_relative_rmse"]
                    - record["mlp_candidate_fit_relative_rmse"]
                ),
                "whole_layer_delta_heldout": (
                    record["post_layer_baseline_relative_rmse"]
                    - record["post_layer_candidate_relative_rmse"]
                ),
                "operator_stage_delta_residual": (
                    (
                        record["post_layer_baseline_relative_rmse"]
                        - record["post_layer_candidate_relative_rmse"]
                    )
                    - (
                        record["mlp_current_fit_relative_rmse"]
                        - record["mlp_candidate_fit_relative_rmse"]
                    )
                ),
                "fit_minus_heldout_at_candidate": (
                    record["mlp_candidate_fit_relative_rmse"]
                    - record["post_layer_candidate_relative_rmse"]
                ),
                "gate_max_abs_channel_mean_drift": record[
                    "gate_max_abs_channel_mean_drift"
                ],
                "gate_child_over_parent_sigma": (
                    None
                    if not record["gate_parent_mean_of_channel_sigma"]
                    else record["gate_child_mean_of_channel_sigma"]
                    / record["gate_parent_mean_of_channel_sigma"]
                ),
            }
            for record in rows
        ],
    }
    with open(
        os.path.join(HERE, "branch_decomposition.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(branch, handle, indent=1)

    # ----- console summary -----
    print("\n".join(lines))
    print()
    gate_scales = [record["roles"]["mlp_gate"]["update_scale"] for record in rows]
    clamped = [
        record["layer_index"]
        for record in rows
        if all(record["roles"][role]["clamp_bound"] for role in ROLES)
    ]
    print(f"layers where all three MLP roles hit the prior-delta clamp: {clamped}")
    print(
        "gate update_scale: L0 {:.4f} -> min {:.4f} at L{}".format(
            gate_scales[0],
            min(gate_scales),
            rows[gate_scales.index(min(gate_scales))]["layer_index"],
        )
    )
    conds = [
        (record["layer_index"], record["system_condition_number"]) for record in rows
    ]
    print("condition numbers:", [(i, f"{c:.3e}") for i, c in conds])
    smins = [
        (record["layer_index"], record["system_singular_value_minimum"])
        for record in rows
    ]
    print("sigma_min:", [(i, f"{s:.6e}") for i, s in smins])
    print(
        "rows per solve:",
        sorted({record["roles"]["mlp_gate"]["rows"] for record in rows}),
    )
    print(
        "underdetermined anywhere:",
        any(
            record["roles"][role]["underdetermined"] for record in rows for role in ROLES
        ),
    )
    print(
        "mlp_down conditioning present anywhere:",
        any(record["roles"]["mlp_down"]["solver_conditioning_present"] for record in rows),
    )
    print(
        "operator-branch prior reports present:",
        branch["operator_branch_prior_correction_reports_present"],
    )

    # ----- does the depth trend track the validation degradation? -----
    def spearman(left, right):
        def rank(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            out = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i
                while (
                    j + 1 < len(order) and values[order[j + 1]] == values[order[i]]
                ):
                    j += 1
                shared = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    out[order[k]] = shared
                i = j + 1
            return out

        a, b = rank(left), rank(right)
        n = len(a)
        mean_a, mean_b = sum(a) / n, sum(b) / n
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        var_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
        var_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
        return cov / (var_a * var_b) if var_a and var_b else float("nan")

    depth = [record["layer_index"] for record in rows]
    log_cond = [math.log10(record["system_condition_number"]) for record in rows]
    scale = [record["roles"]["mlp_gate"]["update_scale"] for record in rows]
    post = [record["post_layer_candidate_relative_rmse"] for record in rows]
    delta = [
        record["post_layer_baseline_relative_rmse"]
        - record["post_layer_candidate_relative_rmse"]
        for record in rows
    ]
    print()
    print(f"spearman(depth, log10 cond)          = {spearman(depth, log_cond):+.4f}")
    print(f"spearman(depth, gate update_scale)   = {spearman(depth, scale):+.4f}")
    print(f"spearman(depth, post-layer relL2)    = {spearman(depth, post):+.4f}")
    print(f"spearman(log10 cond, post-layer relL2)= {spearman(log_cond, post):+.4f}")
    print(f"spearman(update_scale, post-layer relL2)= {spearman(scale, post):+.4f}")
    print(f"spearman(log10 cond, layer improvement)= {spearman(log_cond, delta):+.4f}")
    worse = [
        record["layer_index"]
        for record in rows
        if record["post_layer_candidate_relative_rmse"]
        > record["post_layer_baseline_relative_rmse"]
    ]
    print(f"layers whose install made the held-out post-layer residual worse: {worse}")
    above = [
        record["layer_index"]
        for record in rows
        if record["post_layer_candidate_relative_rmse"] > 1.0
    ]
    print(f"layers finishing above the 1.0 emit-zero floor at their own boundary: {above}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
