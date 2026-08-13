#!/usr/bin/env python3
"""The falsifying test for the wave-17 anomaly, in thirty seconds and no merge.

`Lfm2Model.forward` chooses each block's mask by POSITION in a config list:

    layer_mask = causal_mask if self.config.layer_types[i] == "full_attention" \\
                 else linear_attention

The excision path deletes entries from ``model.model.layers`` and renumbers
``layer_idx``, and leaves ``config.layer_types`` at the parent's fourteen.  So
after an excision that SHIFTS an attention block, index ``i`` reads the parent's
type at that position, and a full-attention block whose parent slot held a conv
is handed the padding mask -- which is ``None`` here -- instead of the causal
one.  It attends to the future.

The prediction is exact and falsifiable:

* excise block 13 (the last) and nothing shifts: every block still reads its own
  type, and the recomputation agrees with the module at the bf16 floor;
* excise block 1 as well and every attention block shifts down one slot into a
  conv's, so every one of them runs uncaused, and the recomputation -- which
  masks -- disagrees by O(1);
* repair ``config.layer_types`` after the deletion and the second case becomes
  the first.

Three loads, no solve, no export.

    python mx18_mask_probe.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mx18_build import PARENT, Harness, attention_probabilities  # noqa: E402


def measure(excise: list[int], repair: bool, device: str) -> dict:
    harness = Harness(PARENT, device, torch.bfloat16)
    config = harness.model.config
    parent_types = list(config.layer_types)
    for index in sorted(excise, reverse=True):
        del harness.model.model.layers[index]
    for position, layer in enumerate(harness.layers()):
        if getattr(layer, "is_attention_layer", False):
            layer.self_attn.layer_idx = position
        elif hasattr(layer, "conv"):
            layer.conv.layer_idx = position
    if repair:
        config.layer_types = [t for i, t in enumerate(parent_types) if i not in excise]
        config.num_hidden_layers = len(config.layer_types)

    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    heads, kv_heads = int(config.num_attention_heads), int(config.num_key_value_heads)
    inv_freq = harness.model.model.rotary_emb.inv_freq.detach().to("cpu", torch.float64)

    rows = []
    surviving = [l for l in range(len(parent_types)) if l not in excise]
    for position, layer in enumerate(harness.layers()):
        if not getattr(layer, "is_attention_layer", False):
            continue
        attn = layer.self_attn
        seen: dict = {}
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []

        def pre_hook(module, inputs, kwargs):
            hidden = inputs[0] if inputs else kwargs["hidden_states"]
            seen["x"] = hidden.detach().reshape(-1, hidden.shape[-1]).to("cpu", torch.float64)
            return None

        def post_hook(module, inputs, kwargs, output):
            captured.append((seen.pop("x"), output[1].detach()[0].to("cpu", torch.float64)))
            return None

        handles = [
            attn.register_forward_pre_hook(pre_hook, with_kwargs=True),
            attn.register_forward_hook(post_hook, with_kwargs=True),
        ]
        try:
            harness.model(input_ids=harness.batches[0][1].to(device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        X, A_module = captured[0]
        A = attention_probabilities(
            X,
            attn.q_proj.weight.data.to(torch.float64).cpu(),
            attn.k_proj.weight.data.to(torch.float64).cpu(),
            attn.q_layernorm.weight.data.to(torch.float64).cpu(),
            attn.k_layernorm.weight.data.to(torch.float64).cpu(),
            inv_freq,
            head_dim**-0.5,
            float(attn.q_layernorm.variance_epsilon),
            heads,
            kv_heads,
            head_dim,
        )
        width = A_module.shape[-1]
        future = torch.triu(torch.ones(width, width, dtype=torch.float64), diagonal=1)
        rows.append(
            {
                "position": position,
                "parent_block": surviving[position],
                "config_type_at_position": config.layer_types[position],
                "probabilities_relL2": float((A - A_module).norm() / A_module.norm()),
                # The whole diagnosis in one number: how much of the module's own
                # attention mass sits on tokens that have not happened yet.
                "module_mass_on_future": float((A_module * future).sum() / A_module.shape[1] / heads),
            }
        )
    del harness
    torch.cuda.empty_cache()
    return {"excise": excise, "repair": repair, "sites": rows}


def main() -> int:
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = [
        measure([13], False, args.device),
        measure([13, 1], False, args.device),
        measure([13, 1], True, args.device),
    ]
    for case in report:
        label = f"excise {case['excise']}" + (" REPAIRED" if case["repair"] else "")
        print(f"--- {label}")
        for row in case["sites"]:
            print(
                f"    position {row['position']:2d} = parent block "
                f"{row['parent_block']:2d}  config says "
                f"{row['config_type_at_position']:16s} relL2 "
                f"{row['probabilities_relL2']:.3e}  future mass "
                f"{row['module_mass_on_future']:.4f}"
            )
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
