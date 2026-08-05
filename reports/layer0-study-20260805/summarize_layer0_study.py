#!/usr/bin/env python3
"""Render the layer-0 representability study's per-arm records as one table.

Reads every ``arm-*.json`` under the given directories and prints the held-out
branch-write and composed post-operator metrics, plus the two solver knobs the
study turned out to hinge on: the prior-anchored update scale of each closed-form
fit.  ``relative_l2 = 1.0`` is the emit-zero baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(record: dict) -> dict:
    metrics = record.get("metrics") or {}
    heldout = metrics.get("heldout") or {}
    last = heldout.get("last_token") or {}
    every = heldout.get("all_rows") or {}
    write = last.get("operator.output") or {}
    post = last.get("residual.post_operator") or {}
    write_all = every.get("operator.output") or {}
    producer = (record.get("producer") or {}).get("in_proj_fit") or {}
    consumer = (record.get("consumer") or {}).get("out_proj_fit") or {}
    return {
        "arm": record.get("arm"),
        "layer": record.get("layer_index"),
        "width": record.get("target_width"),
        "prior": record.get("prior_mode"),
        "ridge": record.get("ridge"),
        "delta": record.get("max_prior_delta_ratio"),
        "dtype": record.get("solve_dtype"),
        "write_cos": write.get("mean_cosine"),
        "write_relL2": write.get("mean_relative_l2"),
        "write_norm_ratio": write.get("mean_norm_ratio"),
        "post_relL2": post.get("mean_relative_l2"),
        "write_cos_all": write_all.get("mean_cosine"),
        "write_relL2_all": write_all.get("mean_relative_l2"),
        "in_scale": producer.get("update_scale"),
        "out_scale": consumer.get("update_scale"),
        "in_prior_norm": producer.get("prior_norm"),
        "in_corr_norm": producer.get("correction_norm"),
        "out_prior_norm": consumer.get("prior_norm"),
        "out_corr_norm": consumer.get("correction_norm"),
        "failed": record.get("failed", False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob("arm-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            row = _row(record)
            row["source"] = f"{directory.name}/{path.name}"
            rows.append(row)
    header = (
        f"{'arm':38s} {'L':>2s} {'width':>5s} {'prior':>13s} {'delta':>6s} "
        f"{'w.cos':>7s} {'w.relL2':>8s} {'w.nrat':>7s} {'post.relL2':>10s} "
        f"{'in_s':>6s} {'out_s':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["failed"]:
            print(f"{row['arm']:38s} FAILED")
            continue

        def fmt(value: object, spec: str) -> str:
            if value is None:
                return "-"
            return format(value, spec)

        print(
            f"{str(row['arm']):38s} {fmt(row['layer'], '>2d')} "
            f"{fmt(row['width'], '>5d')} {str(row['prior'] or '-'):>13s} "
            f"{str(row['delta']):>6s} "
            f"{fmt(row['write_cos'], '>7.4f')} "
            f"{fmt(row['write_relL2'], '>8.4f')} "
            f"{fmt(row['write_norm_ratio'], '>7.4f')} "
            f"{fmt(row['post_relL2'], '>10.4f')} "
            f"{fmt(row['in_scale'], '>6.3f')} "
            f"{fmt(row['out_scale'], '>6.3f')}"
        )
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
