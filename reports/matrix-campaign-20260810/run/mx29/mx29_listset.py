#!/usr/bin/env python3
"""The LIST-to-SET autopsy: read the case swap instead of inferring it.

`PLAN_GAPS_2026-08-14.md` GAP 3.  The terminal-gain regression is this estate's
only *deterministic* death mechanism -- six bases out of six, sealed "beyond
reasonable doubt" -- and it has been routed around six times and autopsied zero
times, despite the sealed twin pair differing in exactly 1,024 receipt-verified
numbers.  Idea #6 derives its mask from non-review probes and #7 searches blends;
both are inference about a mechanism that can simply be READ.

This file reads it, on sealed checkpoints, with zero builds:

1. diff the two twins' `model.embedding_norm` diagonals and rank coordinates by
   ``|delta|``;
2. teacher-force the dedup prompt through both twins and capture the terminal
   logit trajectory at every position;
3. find the FIRST position where the argmax differs -- the token at which the
   solved twin stops writing the control's answer;
4. project that position's logit delta back through the diagonal onto the
   ``delta`` coordinates, so the flip is attributed to coordinates rather than to
   the solve as a whole.

Three outcomes, all useful and all cheap:

* **coordinate-sparse** -- a surgical mask for #6 that is MEASURED rather than
  probe-inferred, and a third arm for #7 cheaper than the alpha grid;
* **distributed** -- #7's per-entry partition arm is pre-doomed and one review is
  saved;
* either way, the first mechanism-level law for a case swap in this campaign
  ("terminal gain trades case X's margin at token t"), which is the template for
  reading every one-case swap the extension bank is about to surface.

It grades nothing.  The mask it emits is a *pre-registration artifact*: it is
written before the masked build runs, and the build cites this file's output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def load_case(pack: Path, case_id: str) -> dict[str, Any]:
    doc = json.loads(pack.read_text(encoding="utf-8"))
    for row in doc["prompts"]:
        if row["id"] == case_id:
            return row
    raise SystemExit(f"REFUSED: no case {case_id} in {pack}")


@torch.no_grad()
def trajectory(model: Any, tokenizer: Any, row: dict[str, Any], device: str,
               steps: int) -> dict[str, Any]:
    messages = [dict(m) for m in row["messages"]]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    cur, past = ids, None
    argmax: list[int] = []
    top_logit: list[float] = []
    hidden_last: list[torch.Tensor] = []
    for _ in range(steps):
        out = model(input_ids=cur, past_key_values=past, use_cache=True,
                    output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        nxt = int(logits.argmax())
        argmax.append(nxt)
        top_logit.append(float(logits.max()))
        hidden_last.append(out.hidden_states[-1][0, -1].float().cpu())
        if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[nxt]], device=device, dtype=ids.dtype)
    return {
        "argmax": argmax,
        "top_logit": top_logit,
        "text": tokenizer.decode(argmax, skip_special_tokens=False),
        "hidden": torch.stack(hidden_last) if hidden_last else torch.zeros(0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--control", type=Path, required=True, help="the zero-solve twin")
    ap.add_argument("--solved", type=Path, required=True, help="the terminal-gain twin")
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--case", default="coding-deduplicate")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--mask-quantile", type=float, default=0.90,
                    help="coordinates above this |delta| quantile go into the emitted mask")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-mask", type=Path)
    args = ap.parse_args()

    import os
    import sys
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for sub in ("src", "experiments"):
        p = str(args.deployment / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from safetensors.torch import load_file  # noqa: E402
    from transformers import AutoTokenizer, Lfm2ForCausalLM  # noqa: E402

    torch.set_grad_enabled(False)
    started = time.monotonic()
    key = "model.embedding_norm.weight"
    dc = load_file(str(args.control / "model.safetensors"))[key].to(torch.float64)
    ds = load_file(str(args.solved / "model.safetensors"))[key].to(torch.float64)
    delta = ds - dc
    absd = delta.abs()
    order = torch.argsort(absd, descending=True)
    cutoff = float(torch.quantile(absd, args.mask_quantile))
    mask_coords = sorted(int(i) for i in (absd > cutoff).nonzero().flatten().tolist())

    # concentration: how few coordinates carry how much of the change
    total = float((delta ** 2).sum())
    cumulative = torch.cumsum(delta[order] ** 2, dim=0) / max(total, 1e-30)
    def coords_for(fraction: float) -> int:
        hit = (cumulative >= fraction).nonzero()
        return int(hit[0]) + 1 if len(hit) else len(order)

    row = load_case(args.pack, args.case)
    tok = AutoTokenizer.from_pretrained(str(args.control))
    runs = {}
    for name, path in (("control", args.control), ("solved", args.solved)):
        model = Lfm2ForCausalLM.from_pretrained(str(path), dtype=torch.float32).to(args.device).eval()
        runs[name] = trajectory(model, tok, row, args.device, args.steps)
        del model
        torch.cuda.empty_cache()

    a, b = runs["control"]["argmax"], runs["solved"]["argmax"]
    flip = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    flip_report: dict[str, Any] = {"first_divergence_step": flip}
    if flip is not None:
        flip_report.update({
            "prefix_shared": tok.decode(a[:flip], skip_special_tokens=False),
            "control_token": tok.decode([a[flip]], skip_special_tokens=False),
            "solved_token": tok.decode([b[flip]], skip_special_tokens=False),
            "control_top_logit": round(runs["control"]["top_logit"][flip], 5),
            "solved_top_logit": round(runs["solved"]["top_logit"][flip], 5),
        })
        # Attribution: at the flip position the two models' final hidden states
        # are scaled by their own diagonals before the head reads them, so the
        # per-coordinate contribution to the logit change is h_d * delta_d
        # weighted by the head rows of the two competing tokens.  Both terms are
        # already on disk; nothing is fitted.
        hc = runs["control"]["hidden"][flip].to(torch.float64)
        hs = runs["solved"]["hidden"][flip].to(torch.float64)
        contribution = (0.5 * (hc + hs)) * delta
        c_order = torch.argsort(contribution.abs(), descending=True)
        top = [int(i) for i in c_order[:24].tolist()]
        flip_report["coordinate_attribution_top"] = [
            {"dim": d, "delta": round(float(delta[d]), 8),
             "contribution": round(float(contribution[d]), 6)}
            for d in top
        ]
        flip_report["attribution_in_mask"] = sum(1 for d in top if d in set(mask_coords))
        flip_report["attribution_concentration_top24_fraction"] = round(
            float(contribution[c_order[:24]].abs().sum() / contribution.abs().sum().clamp_min(1e-30)), 4
        )

    body = {
        "schema": "neural-forge-matrix-wave29-listset-forensic-v1",
        "purpose": (
            "GAP 3: read the terminal-gain case swap on sealed checkpoints instead of "
            "inferring it.  Zero builds, zero solves, pure inference.  Emits a "
            "PRE-REGISTERED mask for the masked terminal-gain arm."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "control": str(args.control),
        "solved": str(args.solved),
        "case": args.case,
        "diagonal": {
            "coordinates": int(delta.numel()),
            "nonzero_delta": int((delta != 0).sum()),
            "delta_l2": round(float(total ** 0.5), 8),
            "delta_max_abs": round(float(absd.max()), 8),
            "coords_for_50pct_energy": coords_for(0.50),
            "coords_for_90pct_energy": coords_for(0.90),
            "top_coordinates": [
                {"dim": int(i), "delta": round(float(delta[int(i)]), 8)}
                for i in order[:24].tolist()
            ],
        },
        "separability_reading": (
            "COORDINATE-SPARSE: a handful of coordinates carry the change, so a mask is "
            "meaningful and #7's partition arm is worth its review"
            if coords_for(0.90) <= 128 else
            "DISTRIBUTED: the change is spread across the diagonal, so #7's per-entry "
            "partition arm is pre-doomed and the mask is a blunt instrument"
        ),
        "control_text": runs["control"]["text"][:4000],
        "solved_text": runs["solved"]["text"][:4000],
        "flip": flip_report,
        "mask": {
            "quantile": args.mask_quantile,
            "cutoff_abs_delta": round(cutoff, 8),
            "coordinates": mask_coords,
            "count": len(mask_coords),
        },
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    if args.out_mask is not None:
        args.out_mask.write_text(json.dumps({
            "schema": "neural-forge-matrix-wave29-termgain-mask-v1",
            "derived_from": str(args.out),
            "case": args.case,
            "coordinates": mask_coords,
            "note": (
                "Coordinates the masked terminal-gain arm holds at the source's prior "
                "value.  Pre-registered: written before the masked build runs."
            ),
        }, indent=1))
    print(json.dumps({k: v for k, v in body.items()
                      if k not in ("control_text", "solved_text", "mask")}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
