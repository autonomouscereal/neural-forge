#!/usr/bin/env python3
"""Second width pass: the MLP lane at L05, and the S8 arms of the components in it.

The two wave-8 controls falsified the strong form of the block-conflict rule
(both DIFFERENT-block third components cost 3-4 review cases, the same as the
same-block ones), so the surviving cheap-composition signal in the campaign is
the SAME-LANE one: COMP.MA.MXJ2 -> COMP.MA.MXK5 added mlp_gate beside mlp_down
in one block for a cost of one case.  C12 mlp_up is the third tensor of that same
SwiGLU lane and has never been composed in any wave.  This asks whether it can
express a true quarter at L05 before any GPU time is spent on it.
"""

from __future__ import annotations

import sys

sys.path.insert(
    0, "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16/src"
)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from neural_forge.matrix_cell_receipt import COMPONENTS, CellId  # noqa: E402
from neural_forge.matrix_technique_registry import (  # noqa: E402
    component_spec,
    storage_model,
    storage_width_for_ratio,
)

CANDIDATES = [
    ("C12.T8.S4.L05", 0.25),
    ("C12.T8.S8.L05", 0.125),
    ("C12.T8.S4.L00", 0.25),
    ("C11.T8.S8.L05", 0.125),
    ("C13.T8.S8.L05", 0.125),
    ("C11.T8.S4.L05", 0.25),
    ("C13.T8.S4.L05", 0.25),
]


def main() -> int:
    raw = load_file("/home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors")
    parent_weights = {key: value.to(torch.float64) for key, value in raw.items()}
    for text, ratio in CANDIDATES:
        cell = CellId.parse(text)
        try:
            spec = component_spec(cell, parent_weights)
        except Exception as exc:  # noqa: BLE001
            print(f"{text:16} SPEC ERROR {type(exc).__name__}: {exc}")
            continue
        width = storage_width_for_ratio(spec, cell.technique, ratio)
        if width is None:
            print(f"{text:16} CANNOT-EXPRESS-4X (own_width={spec.own_width})")
            continue
        stored = int(storage_model(cell.technique, cell.component)(spec, width))
        owned = int(spec.parent_parameter_count)
        kind = str(COMPONENTS[cell.component]["kind"])
        print(
            f"{text:16} kind={kind:14} width={width:4d} stored={stored:8d} "
            f"owned={owned:8d} s/o={stored / owned:.6f} "
            f"keys={list(spec.weight_keys)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
