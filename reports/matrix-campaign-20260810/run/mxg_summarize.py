#!/usr/bin/env python3
"""Pull the wave-3 comparison numbers out of every sealed matrix cell receipt.

Wave 2's `mxf_summarize.py` with the fields wave 3 added: the two new
contraction axes report their own geometry (kept FIR taps, dropped attention
heads) and their own parent gates, which are the numbers the harness's generic
calibration arm could not produce for those components; the progressive
schedule reports its rungs.  Walks one level of subdirectories so a cell re-run
into `<cell>/wave2` or `<cell>/wave3` is found without any sealed receipt being
opened for writing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/ai_data/neural-forge-matrix/cells")
WAVE = int(sys.argv[2]) if len(sys.argv) > 2 else 3


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
            directory / "wave3" / "cell-receipt.json",
        ):
            if candidate.is_file():
                yield candidate


rows = []
for receipt in receipts(ROOT):
    body = json.loads(receipt.read_text(encoding="utf-8"))
    if int(body.get("campaign_wave", 0)) != WAVE:
        continue
    solve = body.get("solve") or {}
    stored = solve.get("stored_parameter_count")
    parent = solve.get("parent_parameter_count")
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
            "calib_applicable": get(body, "calibration", "applicable"),
            "write_cos": get(body, "boundary", "write_heldout_cos"),
            "macro_rel": get(body, "boundary", "post_operator_heldout_relL2"),
            "macro_endpoint": get(body, "boundary", "macrocell_endpoint"),
            "local_solve_applicable": get(body, "boundary", "local_solve_applicable"),
            "in_band": get(body, "boundary", "endpoint_band_le_0p4"),
            "vs_control": get(body, "composed", "vs_copy_exact_control_delta"),
            "emit_zero": get(body, "composed", "emit_zero_floor_passed"),
            "free_run": get(body, "behavior", "free_run", "token_match_vs_control"),
            "lcp": get(body, "behavior", "free_run", "mean_lcp"),
            "rank": solve.get("contraction_basis_rank"),
            "stored": stored,
            "parent_params": parent,
            "stored_ratio": (
                None
                if not isinstance(stored, int) or not isinstance(parent, int) or parent <= 0
                else round(stored / parent, 4)
            ),
            # wave-3 axes
            "kept_taps": solve.get("kept_taps"),
            "kept_tap_indices": solve.get("kept_tap_indices"),
            "fir_replay": solve.get("parent_operator_replay_relative_l2"),
            "fir_discrimination": solve.get("segmentation_discrimination_ratio"),
            "kept_heads": solve.get("kept_heads"),
            "dropped_heads": solve.get("dropped_heads"),
            "head_replay": solve.get("parent_matmul_relative_l2"),
            "identity_gate": solve.get("identity_gate_max_abs_deviation"),
            "schedule": solve.get("progressive_schedule"),
            "schedule_base": solve.get("progressive_base_technique"),
            "final_step_change": solve.get("progressive_final_step_relative_change"),
            "t3_fit_energy": solve.get("fit_retained_energy_fraction"),
            "t3_obs_energy": solve.get("observability_retained_energy_fraction"),
            "wall": body.get("wall_time_seconds"),
            "receipt": body.get("receipt_sha", "")[:12],
            "path": str(receipt),
        }
    )

rows.sort(key=lambda r: r["cell"])
print(json.dumps(rows, indent=1, sort_keys=True))
print(f"\n{len(rows)} wave-{WAVE} receipts", file=sys.stderr)
