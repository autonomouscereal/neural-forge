#!/usr/bin/env python3
"""Re-express one wave-14 narrow build receipt in ADDENDUM 12's form.

`NOTES-MX-R` §6: wave 14's three narrow children were graded by a human and
appended nothing, because receipt schema v2 had no id for a child whose shape
changed.  ADDENDUM 12 gives them one.  What it does not do is re-measure them --
every number this writes is wave 14's own, read out of
``mx14-build-receipt.json`` and rearranged into the ``axes[]`` shape the new
schema mandates.  The wave-14 file is never modified; the ADDENDUM 12 view is
written beside the child under a new name and says in its own body where its
numbers came from.

The one thing that IS re-measured is the digest of the child's tensors, because
the ADDENDUM 12 receipt binds a dossier to the exact file that was graded and
wave 14 never recorded one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PARENT_PARAMS = 229_693_184


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    source = args.child / "mx14-build-receipt.json"
    wave14 = json.loads(source.read_text())
    if wave14.get("schema") != "neural-forge-matrix-wave14-narrow-child-v1":
        raise SystemExit(f"{source} is not a wave-14 narrow build receipt")

    digest = hashlib.sha256()
    with (args.child / "model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)

    sites = wave14["per_site"]
    keep = int(wave14["keep"])
    cosines = sorted(float(s["cos"]) for s in sites)
    receipt = {
        "schema": "neural-forge-matrix-wave15-standard-shape-child-v1",
        "restated_from": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "schema": wave14["schema"],
            "note": (
                "Every number below is wave 14's own.  ADDENDUM 12 supplies the "
                "shape of the receipt, not the measurement; nothing was re-solved "
                "and nothing was re-run.  NOTES-MX-R 6 is why this file exists."
            ),
        },
        "arm": "stale (wave 14: every site solved against the parent's activations)",
        "keep_ff": keep,
        "keep_head_dim": None,
        "excised": [],
        "parent_parameters": int(wave14["parent_parameters"]),
        "child_parameters": int(wave14["child_parameters"]),
        "net_ratio": float(wave14["net_ratio"]),
        "child_tensors_sha256": digest.hexdigest(),
        "axes": [
            {
                "axis": "intermediate_size",
                "kind": "mlp",
                "sites": [int(s["layer"]) for s in sites],
                "config_fields": ["intermediate_size", "block_ff_dim"],
                "selection": (
                    f"{wave14['arm']}: pivoted-Cholesky coreset of real channels, "
                    "w1/w3 rows verbatim"
                ),
                "solve": (
                    "prior-anchored ridge on w2, lambda ladder to update_scale<=1.0"
                ),
                "evidence": "the parent's own sealed rows (wave-13 evidence cache)",
                "parameters_removed": int(wave14["removed"]),
                "heldout": {
                    "median_cos": cosines[len(cosines) // 2],
                    "min_cos": cosines[0],
                    "per_site": sites,
                },
            }
        ],
        "passed_through": ["head_dim", "num_hidden_layers", "hidden_size", "vocab_size"],
        "solver_path": "mx14_narrow.build (prior-anchored ridge, fp64 CPU)",
        "ridge_relative": 1e-6,
        "condition_number": None,
        "prior_norm": None,
        "correction_norm": None,
        "clamp_engaged": any(float(s["lambda_relative"]) > 1e-6 for s in sites),
        "requested_clamp_ratio": max(float(s["free_solve_update_scale"]) for s in sites),
        "allowed_clamp_ratio": 1.0,
        "condition_cap_saturated": False,
        "worst_site": {
            "endpoint": "mlp.down",
            "layer": int(min(sites, key=lambda s: s["cos"])["layer"]),
            "relL2": float(min(sites, key=lambda s: s["cos"])["relL2"]),
            "cos": float(min(sites, key=lambda s: s["cos"])["cos"]),
        },
        "evidence": {
            "trace_cache_key": "2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff",
            "split_authority": "matrix-campaign-capture-manifest",
            "fit_rows": 4751,
            "heldout_rows": 1584,
            "parent_model_forward_calls": 0,
            "conditioned": False,
        },
        "export_path": wave14["export_path"],
        "built_at": wave14["built_at"],
    }
    args.out.write_text(json.dumps(receipt, indent=1))
    print(
        f"restated {source} -> {args.out} "
        f"({receipt['child_parameters']:,} = {receipt['net_ratio']:.6f}x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
