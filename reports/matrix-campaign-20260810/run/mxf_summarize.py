#!/usr/bin/env python3
"""Pull the wave-2 comparison numbers out of every sealed matrix cell receipt.

Same shape as wave 1's `summarize_cells.py`, extended with the fields wave 2
added: the compute architecture (the P40 leak's missing input), the storage
counts and their ratio (the MX-E trap), and the two declared-inapplicability
flags.  Walks one level of subdirectories so a cell re-run into `<cell>/wave2`
is found without its wave-1 receipt being opened for writing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/ai_data/neural-forge-matrix/cells")
WAVE = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def get(body, *path, default=None):
    node = body
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def receipts(root):
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        for candidate in (
            directory / "cell-receipt.json",
            directory / "wave2" / "cell-receipt.json",
        ):
            if candidate.is_file():
                yield candidate


rows = []
for receipt in receipts(ROOT):
    body = json.loads(receipt.read_text(encoding="utf-8"))
    if int(body.get("campaign_wave", 0)) != WAVE:
        continue
    stored = get(body, "solve", "stored_parameter_count")
    parent = get(body, "solve", "parent_parameter_count")
    rows.append(
        {
            "cell": body["cell_id"],
            "component": body["component"],
            "technique": body["technique"],
            "level": body["level"],
            "site": body["site"],
            "arch": get(body, "code_identity", "compute_architecture"),
            "status": get(body, "outcome", "status"),
            "link": get(body, "outcome", "first_divergent_metric"),
            "calib": get(body, "calibration", "copy_exact_reproduction_relL2"),
            "calib_passed": get(body, "calibration", "passed"),
            "calib_applicable": get(body, "calibration", "applicable"),
            "write_cos": get(body, "boundary", "write_heldout_cos"),
            "write_rel": get(body, "boundary", "write_heldout_relL2"),
            "boundary_applicable": get(body, "boundary", "applicable"),
            "local_solve_applicable": get(body, "boundary", "local_solve_applicable"),
            "macro_rel": get(body, "boundary", "post_operator_heldout_relL2"),
            "macro_endpoint": get(body, "boundary", "macrocell_endpoint"),
            "in_band": get(body, "boundary", "endpoint_band_le_0p4"),
            "vs_control": get(body, "composed", "vs_copy_exact_control_delta"),
            "emit_zero": get(body, "composed", "emit_zero_floor_passed"),
            "free_run": get(body, "behavior", "free_run", "token_match_vs_control"),
            "lcp": get(body, "behavior", "free_run", "mean_lcp"),
            "cond": get(body, "solve", "condition_number"),
            "stored": stored,
            "parent_params": parent,
            "stored_ratio": (
                None
                if not isinstance(stored, int)
                or not isinstance(parent, int)
                or parent <= 0
                else round(stored / parent, 6)
            ),
            "kept": get(body, "solve", "kept_width"),
            "active_k": get(body, "solve", "omp", "k"),
            "child_head_dim": get(body, "solve", "child_head_dim"),
            "chart_cycle_fit": get(body, "solve", "chart_cycle_defect_fit"),
            "chart_cycle_heldout": get(body, "solve", "chart_cycle_defect_heldout"),
            "chart_decoder_dev": get(
                body, "solve", "chart_decoder_vs_encoder_transpose_relative_deviation"
            ),
            "balanced_fit_energy": get(body, "solve", "fit_retained_energy_fraction"),
            "balanced_obs_energy": get(
                body, "solve", "observability_retained_energy_fraction"
            ),
            "solver": get(body, "solve", "solver_path"),
            "wall": body.get("wall_time_seconds"),
            "review": get(body, "outcome", "decode_smoke_review"),
            "smoke": [
                {
                    "terminated": e.get("terminated"),
                    "chars": len(e.get("output_verbatim") or ""),
                }
                for e in (get(body, "behavior", "decode_smoke") or [])
            ],
            "notes": (get(body, "outcome", "notes") or "")[:300],
            "receipt": body["receipt_sha"][:12],
            "instance": body["cell_instance_id"][:8],
            "path": str(receipt),
            "tree": get(body, "code_identity", "src_tree_sha", default="")[:12],
        }
    )

print(json.dumps(rows, indent=1, sort_keys=False))
