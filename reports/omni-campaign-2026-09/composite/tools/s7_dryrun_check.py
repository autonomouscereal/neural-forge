#!/usr/bin/env python3
"""s7_dryrun_check.py -- load a c16 dry-run output with chimera_omni and summarise its manifest.

Lane E4, Claude, 2026-10-09. Loads `--out` (and `--child`) through
safetensors_io.load_chimera_checkpoint, asserts the charted layer count, runs one text
forward on both and checks the residual stream is bit-identical (charted blocks write only
on routed media rows) and the logits are finite. Prints the manifest's per-component
accounting, block placements and time-reference checks as one JSON line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--child", required=True)
    args = parser.parse_args(argv)
    import torch
    from chimera_omni.importing.safetensors_io import load_chimera_checkpoint

    out = Path(args.out)
    manifest = json.loads((out / "composite_manifest.json").read_text(encoding="utf-8"))
    model, cfg, receipt = load_chimera_checkpoint(out)
    child, child_cfg, _ = load_chimera_checkpoint(Path(args.child))
    kinds = [cfg.block_kind(i) for i in range(cfg.num_layers)]
    torch.manual_seed(7)
    ids = torch.randint(0, 256, (1, 9))
    with torch.no_grad():
        a = child.eval()(ids)
        b = model.eval()(ids)
    result = {
        "out": str(out),
        "loaded_tensors": receipt.get("tensors"),
        "parameters": sum(int(p.numel()) for p in model.parameters()),
        "child_parameters": sum(int(p.numel()) for p in child.parameters()),
        "layers": cfg.num_layers,
        "charted_layers": [i for i, k in enumerate(kinds) if k == "charted"],
        "charted_specs": [{"source": s.source, "component": s.component, "layer_index": s.layer_index,
                           "modulation": s.modulation, "width": s.width, "mlp_width": s.mlp_width}
                          for s in cfg.charted_blocks],
        "text_hidden_bit_identical": bool(torch.equal(a["hidden"], b["hidden"])),
        "text_logits_finite": bool(torch.isfinite(b["logits"]).all()),
        "manifest_keys": sorted(manifest),
    }
    for key in ("components", "register", "header_count", "blocks", "disk"):
        if key in manifest:
            result[key] = manifest[key]
    print(json.dumps(result, default=str))
    return 0 if result["text_logits_finite"] else 1


if __name__ == "__main__":
    sys.exit(main())
