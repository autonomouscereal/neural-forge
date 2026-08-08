#!/usr/bin/env python3
"""Two independent checks on the Stage-A attribution, from the sealed telemetry.

1. Clamp identity.  `_fit_linear` writes back
   ``correction_norm = update_scale * raw_norm`` and
   ``update_scale = min(1, max_prior_delta_ratio * prior_norm / raw_norm)``
   (`hf_lfm2_structural_residual_attention_fold.py:1204-1216`).  With the
   hard-coded body default ``max_prior_delta_ratio = 1.0``
   (`hf_lfm2_dynamic_closed_loop_adapter.py:1262`) a bound clamp must leave
   ``correction_norm == prior_norm`` to machine precision.  If that identity
   holds on every solve, the clamp bound on every solve.

2. Ridge authority.  The conditioning block is measured on the solved system
   ``X^T X + ridge*I`` (`:1184`, `:1193-1195`), so ``sigma_min - ridge`` is an
   upper bound on the data's own smallest singular value in that direction.
   Where that difference is the same order as the ridge, the solve is
   inverting the regularizer rather than the evidence.
"""

from __future__ import annotations

import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "per_layer_profile.json")
RIDGE = 1e-6  # hf_lfm2_dynamic_closed_loop_adapter.py:1260, no CLI override
MAX_PRIOR_DELTA_RATIO = 1.0  # :1262, no CLI override
ROLES = ("mlp_gate", "mlp_up", "mlp_down")


def main() -> int:
    with open(PROFILE, "r", encoding="utf-8") as handle:
        profile = json.load(handle)

    print("=== check 1: does the prior-delta clamp bind on every MLP solve? ===")
    solves = 0
    bound = 0
    worst_identity = 0.0
    rows = []
    for layer in profile["layers"]:
        for role in ROLES:
            report = layer["roles"][role]
            solves += 1
            scale = report["update_scale"]
            prior = report["prior_norm"]
            correction = report["correction_norm"]
            relative = abs(correction - MAX_PRIOR_DELTA_RATIO * prior) / prior
            if scale < 1.0:
                bound += 1
                worst_identity = max(worst_identity, relative)
            rows.append(
                (
                    layer["layer_index"],
                    role,
                    scale,
                    prior,
                    correction,
                    relative,
                    report["requested_correction_over_prior"],
                    report["underdetermined"],
                    report["rows"],
                    report["input_width"],
                )
            )
    print(f"solves inspected: {solves}   clamp bound (update_scale < 1): {bound}")
    print(
        "worst |correction_norm - 1.0*prior_norm| / prior_norm over the bound "
        f"solves: {worst_identity:.3e}"
    )
    print()
    print(
        "| L | role | rows | in_w | underdet | update_scale | prior_norm | "
        "correction_norm | requested / prior |"
    )
    print("|" + "---|" * 9)
    for (
        index,
        role,
        scale,
        prior,
        correction,
        _relative,
        requested,
        underdetermined,
        row_count,
        width,
    ) in rows:
        print(
            f"| {index} | {role} | {row_count} | {width} | "
            f"{'true' if underdetermined else 'false'} | {scale:.6f} | "
            f"{prior:.6f} | {correction:.6f} | {requested:.1f}x |"
        )

    print()
    print("=== check 2: is the ridge doing any regularizing? ===")
    print(
        "| L | sigma_max | sigma_min(X^T X + ridge I) | sigma_min - ridge | "
        "ridge share of sigma_min | relative ridge (ridge/sigma_max) |"
    )
    print("|" + "---|" * 6)
    for layer in profile["layers"]:
        smax = layer["system_singular_value_maximum"]
        smin = layer["system_singular_value_minimum"]
        if smax is None or smin is None:
            continue
        print(
            f"| {layer['layer_index']} | {smax:.6e} | {smin:.6e} | "
            f"{smin - RIDGE:.6e} | {100.0 * RIDGE / smin:6.2f}% | "
            f"{RIDGE / smax:.3e} |"
        )

    print()
    unreached = [
        layer["layer_index"]
        for layer in profile["layers"]
        if layer["system_singular_value_minimum"] is not None
        and RIDGE / layer["system_singular_value_minimum"] > 0.01
    ]
    print(
        "layers where the ridge supplies more than 1% of the smallest singular "
        f"value (i.e. the data is numerically rank-deficient there): {unreached}"
    )
    print(
        "roles with no conditioning telemetry at all:",
        sorted(
            {
                role
                for layer in profile["layers"]
                for role in ROLES
                if not layer["roles"][role]["solver_conditioning_present"]
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
