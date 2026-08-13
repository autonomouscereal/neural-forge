#!/usr/bin/env python3
"""Where, exactly, the fp64 recomputation and the module's own attention differ.

The gate probe inside the builder returns aggregates, and on the 13+1 child its
aggregates disagree with each other: the module's probabilities are half as
peaked at the median, the argmax agrees on barely half the rows, the whole
tensor is relL2 0.95 apart -- and a per-row centred regression of the two sets
of log-probabilities returns slope 1.0000 with a residual one percent of the
logit spread.  Both cannot be a summary of the same disagreement, so this walks
the disagreement itself: per layer, per batch, per row.

It runs on an EXPORTED child, not inside a build.  The gate is evaluated after
q and k are installed and the attention probabilities depend on q and k alone,
so the exported checkpoint carries exactly the tensors the gate read -- which
also makes this reproducible by anyone holding the child, without the harness.

    python mx18_gate_forensics.py --child <dir> [--rows 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mx18_build import Harness, attention_probabilities  # noqa: E402


def entropy(P: torch.Tensor) -> float:
    safe = P.clamp_min(1e-300)
    return float((-(safe * safe.log()).sum(-1)).mean())


def main() -> int:
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    harness = Harness(args.child, args.device, torch.bfloat16)
    config = harness.model.config
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    inv_freq = harness.model.model.rotary_emb.inv_freq.detach().to("cpu", torch.float64)
    scaling = head_dim**-0.5

    sites = [
        (position, layer)
        for position, layer in enumerate(harness.model.model.layers)
        if getattr(layer, "is_attention_layer", False)
    ]
    report: list[dict] = []
    for position, layer in sites:
        attn = layer.self_attn
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []

        state: dict = {}

        def pre_hook(module, inputs, kwargs):
            hidden = inputs[0] if inputs else kwargs["hidden_states"]
            state["x"] = hidden.detach().reshape(-1, hidden.shape[-1]).to("cpu", torch.float64)
            return None

        def post_hook(module, inputs, kwargs, output):
            captured.append((state.pop("x"), output[1].detach()[0].to("cpu", torch.float64)))
            return None

        handles = [
            attn.register_forward_pre_hook(pre_hook, with_kwargs=True),
            attn.register_forward_hook(post_hook, with_kwargs=True),
        ]
        try:
            for _, ids in harness.batches:
                harness.model(input_ids=ids.to(harness.device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        wq = attn.q_proj.weight.data.to(torch.float64).cpu()
        wk = attn.k_proj.weight.data.to(torch.float64).cpu()
        gq = attn.q_layernorm.weight.data.to(torch.float64).cpu()
        gk = attn.k_layernorm.weight.data.to(torch.float64).cpu()
        eps = float(attn.q_layernorm.variance_epsilon)

        rows_out = []
        worst = None
        for index, (X, A_module) in enumerate(captured):
            A = attention_probabilities(
                X, wq, wk, gq, gk, inv_freq, scaling, eps, heads, kv_heads, head_dim
            )
            relL2 = float((A - A_module).norm() / A_module.norm())
            agree = float((A.argmax(-1) == A_module.argmax(-1)).to(torch.float64).mean())
            rows_out.append(
                {
                    "batch": index,
                    "relL2": relL2,
                    "argmax_agreement": agree,
                    "entropy_fp64": entropy(A),
                    "entropy_module": entropy(A_module),
                    "max_prob_fp64": float(A.max(-1).values.median()),
                    "max_prob_module": float(A_module.max(-1).values.median()),
                }
            )
            if worst is None or relL2 > worst[0]:
                worst = (relL2, index, A, A_module, X)

        relL2, index, A, A_module, X = worst
        head = int((A - A_module).pow(2).sum(dim=(1, 2)).argmax())
        per_row = (A[head] - A_module[head]).pow(2).sum(-1)
        picked = per_row.topk(args.rows).indices.tolist()
        detail = []
        for query in picked:
            top_fp64 = A[head, query].topk(4)
            top_mod = A_module[head, query].topk(4)
            detail.append(
                {
                    "query_row": int(query),
                    "valid_keys": int(query) + 1,
                    "fp64_top": [
                        [int(i), round(float(v), 6)]
                        for i, v in zip(top_fp64.indices, top_fp64.values)
                    ],
                    "module_top": [
                        [int(i), round(float(v), 6)]
                        for i, v in zip(top_mod.indices, top_mod.values)
                    ],
                    "fp64_mass_on_module_top1": float(A[head, query, top_mod.indices[0]]),
                    "module_mass_on_fp64_top1": float(A_module[head, query, top_fp64.indices[0]]),
                }
            )
        site = {
            "parent_layer_position": position,
            "worst_batch": index,
            "worst_head": head,
            "worst_relL2": relL2,
            "per_batch": rows_out,
            "worst_rows": detail,
            "row_relL2_quantiles": {
                q: float(
                    torch.quantile(
                        (A - A_module).pow(2).sum(-1).sqrt().flatten()
                        / A_module.pow(2).sum(-1).sqrt().flatten().clamp_min(1e-300),
                        torch.tensor(q, dtype=torch.float64),
                    )
                )
                for q in (0.5, 0.9, 0.99)
            },
        }
        report.append(site)
        print(
            f"site@{position:02d} worst batch {index} head {head} relL2 {relL2:.3e} | "
            f"per-batch relL2 "
            f"{min(r['relL2'] for r in rows_out):.3e}..{max(r['relL2'] for r in rows_out):.3e} | "
            f"row relL2 median {site['row_relL2_quantiles'][0.5]:.3e} "
            f"p99 {site['row_relL2_quantiles'][0.99]:.3e}",
            flush=True,
        )
        for row in detail:
            print(f"    q{row['query_row']:4d} ({row['valid_keys']} keys)")
            print(f"      fp64   {row['fp64_top']}")
            print(f"      module {row['module_top']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
