#!/usr/bin/env python3
"""Resolve the S-level storage width for every wave-8 candidate cell.

ADDENDUM 6 §6.2: an S level bounds what a component STORES, not the width it
keeps, so the width is searched against the technique's storage algebra and
then re-checked by the runner against the realized count.  This script is the
planning half of that pair; the receipt is still the proof.

Cyril's hard rule is checked here too, before a GPU is spent: a candidate whose
storage_width_for_ratio is None cannot express 4x and is dropped, never folded
shallower.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16")
sys.path.insert(0, str(_DEPLOY / "src"))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_registry import (  # noqa: E402
    component_spec,
    storage_model,
    storage_width_for_ratio,
)

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors")

RATIOS = {"S4": 0.25, "S8": 0.125}

CANDIDATES = [
    # step 0 controls + step 1 C05 S8 ladder across every short_conv block
    *[f"C05.T8.S8.L{n:02d}" for n in (0, 1, 3, 5, 7, 9, 11, 13)],
    *[f"C05.T8.S4.L{n:02d}" for n in (5, 7)],
    # step 2 MLP folds -- FFN exists in all fourteen blocks
    *[f"C13.T8.S4.L{n:02d}" for n in range(14)],
    *[f"C11.T8.S4.L{n:02d}" for n in range(14)],
    *[f"C13.T8.S8.L{n:02d}" for n in (2, 4, 6, 8)],
    *[f"C11.T8.S8.L{n:02d}" for n in (2, 5, 8)],
    # attention lane, proven sites only
    "C07.T5.S4.L12", "C07.T5.S8.L12", "C06.T1.S4.L12",
]


def main() -> int:
    raw = load_file(str(PARENT))
    parent_weights = {k: v.to(torch.float64) for k, v in raw.items()}
    out = {}
    for text in CANDIDATES:
        cell = CellId.parse(text)
        try:
            spec = component_spec(cell, parent_weights)
        except Exception as exc:  # noqa: BLE001
            out[text] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        ratio = RATIOS[cell.level]
        width = storage_width_for_ratio(spec, cell.technique, ratio)
        if width is None:
            out[text] = {
                "cannot_express": True,
                "parent_parameter_count": spec.parent_parameter_count,
                "own_width": spec.own_width,
            }
            continue
        model = storage_model(cell.technique, cell.component)
        stored = int(model(spec, width))
        out[text] = {
            "width": int(width),
            "stored": stored,
            "parent_parameter_count": int(spec.parent_parameter_count),
            "stored_over_owned": stored / float(spec.parent_parameter_count),
            "bound": ratio,
            "satisfies": stored <= ratio * spec.parent_parameter_count,
            "weight_keys": list(spec.weight_keys),
            "input_endpoint": spec.input_endpoint,
            "write_endpoint": spec.write_endpoint,
        }
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
