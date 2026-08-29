#!/usr/bin/env python3
"""The free forensic pre-read for tax 3: does emission collapse live in the terminal read?

`INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` #5 makes this the first thing wave 29
spends a forward pass on, and makes it decisive in both directions: **no margin
inversion in the dying children kills MARGIN-RESTORE for free** and seals
"degeneracy lives upstream of the terminal read"; a confirmed inversion buys one
build.  Nothing here grades a child, and nothing here is a gate on a child.

ADDENDUM 16.6 adds the mechanism the 2026 literature says to look for.  The
published account (arXiv:2505.06708, NeurIPS oral; 2601.22966; 2603.05498) is
that **attention sinks** and **massive-activation channels** are a model's
emission-regulation machinery rather than its knowledge storage, so a fold that
disturbs those few channels breaks generation while the bulk channels still hold
the answer.  That is this campaign's tax 3 named from outside, and wave 14
already measured one unit carrying 97% of block 5's gated-output energy on this
parent, so the object is not hypothetical here.  This pre-read therefore measures
four things rather than one, and the fourth decides where a repair would aim:

1. **copy margin** -- teacher-forced on the probe text, the mean gap between the
   logit of the token that actually comes next and the best alternative.  This is
   the content-agnostic copy circuit; wave 28 sealed that the last surviving
   behaviour in a dying child is copying.
2. **generative margin** -- the mean top1-minus-top2 gap over the child's OWN
   free-running continuation.  The plan's inversion is: copy margin holds while
   generative margin flattens.
3. **terminal moments** -- entropy of the final distribution and the second-moment
   trace of the reader-projected final state, which are the two statistics a
   moment-matching recalibration would match.
4. **sink and massive-activation disturbance** -- per attention head, the share of
   attention mass landing on position 0; per residual dimension, the magnitude
   outliers.  Reported as parent-vs-child set overlap and rank correlation, so a
   confirmed inversion can be aimed at *those* channels instead of smeared over
   all 1,024.

Usage::

    mx44_emission.py --deployment D --parent P --child C --label V19CL896 --out J
    mx44_emission.py --deployment D --parent P --parent-only --label PARENT --out J
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch


def load_probes(pack: Path, limit: int | None) -> list[dict[str, Any]]:
    doc = json.loads(pack.read_text(encoding="utf-8"))
    rows = doc["prompts"] if "prompts" in doc else [
        r for lane in doc["triplets"].values() for r in lane
    ]
    return rows[:limit] if limit else rows


def repetition_rate(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are not the first occurrence.  A loop reads high."""
    words = re.findall(r"\S+", text)
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


@torch.no_grad()
def measure(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: str,
            gen_tokens: int) -> dict[str, Any]:
    per_prompt: list[dict[str, Any]] = []
    sink_by_head: dict[str, list[float]] = {}
    massive_dim_energy: torch.Tensor | None = None

    for row in rows:
        messages = [dict(m) for m in row["messages"]]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover
            text = "\n".join(m["content"] for m in messages)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        out = model(input_ids=ids, output_hidden_states=True,
                    output_attentions=True, use_cache=False)
        logits = out.logits[0].float()          # [T, V]
        hidden = out.hidden_states[-1][0].float()  # [T, H]

        # --- 1. copy margin, teacher-forced over the probe's own tokens -------
        target = ids[0, 1:]
        step = logits[:-1]
        chosen = step.gather(1, target.unsqueeze(1)).squeeze(1)
        masked = step.clone()
        masked.scatter_(1, target.unsqueeze(1), float("-inf"))
        best_other = masked.max(dim=1).values
        copy_margin = float((chosen - best_other).mean())

        # --- 3. terminal moments at the generation position -------------------
        last = logits[-1]
        probs = torch.softmax(last, dim=-1)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
        top2 = torch.topk(last, 2).values
        prompt_end_margin = float(top2[0] - top2[1])
        reader_second_moment = float((hidden[-1] ** 2).mean())

        # --- 4a. massive-activation dimensions --------------------------------
        energy = (hidden ** 2).mean(dim=0)  # [H]
        massive_dim_energy = energy if massive_dim_energy is None else massive_dim_energy + energy

        # --- 4b. attention sinks ---------------------------------------------
        if out.attentions:
            for layer_index, attn in enumerate(out.attentions):
                if attn is None:
                    continue
                a = attn[0].float()  # [heads, T, T]
                mass = a[:, :, 0].mean(dim=1)  # attention mass on position 0
                for head, value in enumerate(mass.tolist()):
                    sink_by_head.setdefault(f"L{layer_index:02d}H{head:02d}", []).append(value)

        # --- 2. generative margin over the model's own continuation -----------
        cur = ids
        past = None
        margins: list[float] = []
        produced: list[int] = []
        for _ in range(gen_tokens):
            step_out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = step_out.past_key_values
            step_logits = step_out.logits[0, -1].float()
            pair = torch.topk(step_logits, 2).values
            margins.append(float(pair[0] - pair[1]))
            nxt = int(step_logits.argmax())
            produced.append(nxt)
            if tokenizer.eos_token_id is not None and nxt == tokenizer.eos_token_id:
                break
            cur = torch.tensor([[nxt]], device=device, dtype=ids.dtype)
        decoded = tokenizer.decode(produced, skip_special_tokens=False)

        per_prompt.append({
            "id": row["id"],
            "prompt_tokens": int(ids.shape[1]),
            "copy_margin": round(copy_margin, 5),
            "generative_margin": round(sum(margins) / max(len(margins), 1), 5),
            "generated_tokens": len(produced),
            "terminal_entropy": round(entropy, 5),
            "prompt_end_margin": round(prompt_end_margin, 5),
            "reader_second_moment": round(reader_second_moment, 6),
            "repetition_rate_5gram": round(repetition_rate(decoded), 4),
            "emitted_empty": len(decoded.strip()) == 0,
        })

    n = max(len(per_prompt), 1)
    agg = {
        key: round(sum(p[key] for p in per_prompt) / n, 5)
        for key in ("copy_margin", "generative_margin", "terminal_entropy",
                    "prompt_end_margin", "reader_second_moment",
                    "repetition_rate_5gram")
    }
    agg["empty_lanes"] = sum(1 for p in per_prompt if p["emitted_empty"])
    agg["copy_over_generative"] = round(
        agg["copy_margin"] / agg["generative_margin"], 5
    ) if agg["generative_margin"] else None

    energy = (massive_dim_energy / n) if massive_dim_energy is not None else torch.zeros(1)
    order = torch.argsort(energy, descending=True)
    median = float(energy.median())
    massive = [int(i) for i in order[:32].tolist()
               if float(energy[int(i)]) > 20.0 * max(median, 1e-9)]
    sinks = {k: sum(v) / len(v) for k, v in sink_by_head.items()}
    top_sinks = sorted(sinks.items(), key=lambda kv: -kv[1])[:16]

    return {
        "aggregate": agg,
        "per_prompt": per_prompt,
        "massive_activation_dims": massive,
        "massive_dim_energy": {str(int(i)): round(float(energy[int(i)]), 4)
                               for i in order[:32].tolist()},
        "residual_energy_median": round(median, 6),
        "sink_heads_top": [{"head": k, "first_token_mass": round(v, 5)} for k, v in top_sinks],
        "sink_head_count": len(sinks),
        "sink_heads_all": {k: round(v, 5) for k, v in sinks.items()},
    }


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return round(num / den, 4) if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--child", type=Path)
    ap.add_argument("--child-tokenizer", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import os
    import sys
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for sub in ("src", "experiments"):
        p = str(args.deployment / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from transformers import AutoTokenizer, Lfm2ForCausalLM  # noqa: E402

    torch.set_grad_enabled(False)
    started = time.monotonic()
    rows = load_probes(args.probes, args.limit)
    print(f"probes: {len(rows)}")

    # eager attention is REQUIRED: sdpa returns no attention weights, and a sink
    # measurement that silently reports nothing is worse than no measurement.
    parent = Lfm2ForCausalLM.from_pretrained(
        str(args.parent), dtype=torch.float32, attn_implementation="eager"
    ).to(args.device).eval()
    parent_tok = AutoTokenizer.from_pretrained(str(args.parent))
    parent_stats = measure(parent, parent_tok, rows, args.device, args.gen_tokens)
    del parent
    torch.cuda.empty_cache()

    child_stats: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    if args.child is not None:
        child = Lfm2ForCausalLM.from_pretrained(
            str(args.child), dtype=torch.float32, attn_implementation="eager"
        ).to(args.device).eval()
        child_tok = AutoTokenizer.from_pretrained(
            str(args.child_tokenizer or args.child)
        ) if args.child_tokenizer else parent_tok
        child_stats = measure(child, child_tok, rows, args.device, args.gen_tokens)
        del child
        torch.cuda.empty_cache()

        pa, ca = parent_stats["aggregate"], child_stats["aggregate"]
        shared_heads = sorted(set(parent_stats["sink_heads_all"]) & set(child_stats["sink_heads_all"]))
        p_massive = set(parent_stats["massive_activation_dims"])
        c_massive = set(child_stats["massive_activation_dims"])
        comparison = {
            "copy_margin_retained": round(ca["copy_margin"] / pa["copy_margin"], 4) if pa["copy_margin"] else None,
            "generative_margin_retained": round(ca["generative_margin"] / pa["generative_margin"], 4) if pa["generative_margin"] else None,
            "margin_inversion": (
                None if not (pa["copy_margin"] and pa["generative_margin"]) else
                round((ca["copy_margin"] / pa["copy_margin"]) - (ca["generative_margin"] / pa["generative_margin"]), 4)
            ),
            "entropy_ratio": round(ca["terminal_entropy"] / pa["terminal_entropy"], 4) if pa["terminal_entropy"] else None,
            "second_moment_ratio": round(ca["reader_second_moment"] / pa["reader_second_moment"], 4) if pa["reader_second_moment"] else None,
            "repetition_delta": round(ca["repetition_rate_5gram"] - pa["repetition_rate_5gram"], 4),
            "massive_dims_parent": sorted(p_massive),
            "massive_dims_child": sorted(c_massive),
            "massive_dim_overlap": len(p_massive & c_massive),
            "massive_dims_lost": sorted(p_massive - c_massive),
            "sink_head_rank_spearman": spearman(
                [parent_stats["sink_heads_all"][h] for h in shared_heads],
                [child_stats["sink_heads_all"][h] for h in shared_heads],
            ),
            "sink_mass_ratio": (
                round(sum(child_stats["sink_heads_all"][h] for h in shared_heads)
                      / max(sum(parent_stats["sink_heads_all"][h] for h in shared_heads), 1e-9), 4)
                if shared_heads else None
            ),
        }
        # The pre-registered reading, computed rather than argued.
        inv = comparison["margin_inversion"]
        sink = comparison["sink_mass_ratio"]
        overlap = comparison["massive_dim_overlap"]
        verdict = []
        if inv is not None and inv >= 0.10:
            verdict.append("MARGIN INVERSION CONFIRMED: copy retained above generative by >=0.10")
        elif inv is not None:
            verdict.append("no margin inversion at the 0.10 threshold")
        if sink is not None and (sink < 0.75 or sink > 1.35):
            verdict.append(f"SINK MASS DISTURBED: child/parent first-token mass {sink}")
        if p_massive and overlap < len(p_massive):
            verdict.append(f"MASSIVE-ACTIVATION DIMS LOST: {len(p_massive) - overlap} of {len(p_massive)}")
        comparison["reading"] = verdict or ["no pre-registered signature fired"]

    body = {
        "schema": "neural-forge-matrix-wave44-emission-forensic-v1",
        "purpose": (
            "The free pre-read for MARGIN-RESTORE (#5) with ADDENDUM 16.6's sink and "
            "massive-activation channels added.  It grades nothing and gates nothing; "
            "it decides where a repair would aim, or kills the repair for free."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "probe_pack": str(args.probes),
        "probes_used": len(rows),
        "generation_tokens": args.gen_tokens,
        "parent": parent_stats,
        "child": child_stats,
        "comparison": comparison,
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(json.dumps({"label": args.label, "aggregate_child": (child_stats or {}).get("aggregate"),
                      "comparison": comparison}, indent=1)[:2400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
