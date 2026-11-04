#!/usr/bin/env python3
"""C19 -- the v2 fold-init depth alignment (V2_FOLD_INIT_PLAN_2026-10-21 section 4.2).

Two maps, both written BEFORE any fit so a fold cannot choose its alignment after seeing numbers:

* trunk -> v2, period-aligned and kind-preserving.  Source and target share the
  `(kda, kda, kda, gated_gqa)` period, so PERIODS are aligned by C17's floor-interval
  convention and slot j of a v2 period pools slot j of every source period it receives:

      v2 period p  <-  source periods [floor(P_src p / P_v2), floor(P_src (p + 1) / P_v2))
      v2 layer 4p + j  <-  source layers {4q + j : q in that range}

  64 -> 24 layers (16 -> 6 periods) pools 2-3 source layers per v2 layer; 64 -> 16 (the fold
  proxy) pools exactly 4; 64 -> 64 is the identity (the exactness test).

* K3 -> v2, floor-interval over the MoE layers: v2 layer k receives K3 layers
  `[offset + floor(n k / N), offset + floor(n (k + 1) / N))` with the dense K3 layer 0 joining
  v2 layer 0 (C17's `--dense-layer0 trunk0`).

`alignment.json` uses C17's `{schema, summary, table}` layout so `c10_union_install.load_alignment`
reads the K3 half unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

TOOL_NAME = "c19_v2_alignment"
TOOL_VERSION = "1.0.0"
SCHEMA = "chimera_omni.composite.v2_fold_alignment.v1"
C17_ALIGNMENT_SCHEMA = "chimera_omni.composite.c17_alignment.v1"


def floor_interval(n_source: int, n_target: int) -> list[list[int]]:
    """C17's floor-interval map: target t receives source [floor(n t / N), floor(n (t + 1) / N)).

    Refuses a map that would leave a target with no source (n_source < n_target): a fold layer
    with nothing to fold from would have to be placed or invented.
    """

    if n_source <= 0 or n_target <= 0:
        raise ValueError("floor_interval needs positive sizes")
    if n_source < n_target:
        raise ValueError(f"{n_source} source units cannot cover {n_target} targets without an empty target")
    return [list(range((n_source * t) // n_target, (n_source * (t + 1)) // n_target)) for t in range(n_target)]


def period_alignment(source_layers: int, target_layers: int, period: int,
                     kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    """Kind-preserving trunk -> v2 map: one row per v2 layer with its pooled source layers."""

    if len(kinds) != period:
        raise ValueError(f"kinds {kinds} must name every slot of the period {period}")
    if source_layers % period or target_layers % period:
        raise ValueError(f"source {source_layers} and target {target_layers} layers must be whole periods of {period}")
    periods = floor_interval(source_layers // period, target_layers // period)
    rows = []
    for p, source_periods in enumerate(periods):
        for j in range(period):
            rows.append({
                "target_layer": p * period + j,
                "target_period": p,
                "slot": j,
                "kind": kinds[j],
                "source_periods": list(source_periods),
                "source_layers": [q * period + j for q in source_periods],
            })
    return rows


def k3_alignment(k3_moe_layers: int, target_layers: int, *, first_moe_layer: int = 1,
                 dense_layer0: bool = True) -> list[dict[str, Any]]:
    """K3 MoE layers -> v2 layers by floor interval; the dense K3 L0 joins v2 layer 0."""

    rows = []
    for k, block in enumerate(floor_interval(k3_moe_layers, target_layers)):
        layers = [first_moe_layer + i for i in block]
        if dense_layer0 and k == 0:
            layers = [0] + layers
        rows.append({"trunk": k, "k3_layers": layers})
    return rows


def alignment_document(*, source_name: str, source_layers: int, target_profile: str, target_layers: int,
                       period: int, kinds: tuple[str, ...], k3_moe_layers: int | None = None) -> dict[str, Any]:
    trunk = period_alignment(source_layers, target_layers, period, kinds)
    pools = [len(row["source_layers"]) for row in trunk]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "source": source_name,
        "target_profile": target_profile,
        "rule": {
            "trunk": f"period:{source_layers // period}->{target_layers // period} (floor interval, kind-preserving)",
            "k3": None if k3_moe_layers is None else f"floor:{k3_moe_layers}->{target_layers} (+ dense L0 on layer 0)",
        },
        "summary": {"source_layers": source_layers, "target_layers": target_layers, "period": period,
                    "kinds": list(kinds), "pool_min": min(pools), "pool_max": max(pools)},
        "trunk_table": trunk,
    }
    if k3_moe_layers is not None:
        doc["k3"] = {"schema": C17_ALIGNMENT_SCHEMA,
                     "summary": {"k3_moe_layers": k3_moe_layers, "trunk_layers": target_layers},
                     "table": k3_alignment(k3_moe_layers, target_layers)}
    return doc


def document_sha256(doc: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-layers", type=int, required=True)
    parser.add_argument("--target-profile", required=True)
    parser.add_argument("--target-layers", type=int, required=True)
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--k3-moe-layers", type=int, default=None, help="92 for Kimi K3; omit for no K3 table")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    kinds = tuple(["kda"] * (args.period - 1) + ["gated_gqa"])
    doc = alignment_document(source_name=args.source_name, source_layers=args.source_layers,
                             target_profile=args.target_profile, target_layers=args.target_layers,
                             period=args.period, kinds=kinds, k3_moe_layers=args.k3_moe_layers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "sha256": document_sha256(doc), "summary": doc["summary"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
