#!/usr/bin/env python3
"""Which coordinates of the terminal diagonal carry which capability?

Wave 29 weaponised nothing and learned two things that point opposite ways.
``ARCH.BP64`` -- 64 of the 1,024 ``model.embedding_norm`` coordinates taken from
the solved twin and the other 960 left at the control's -- reproduces the ENTIRE
capability swap: it reads 10/15 with exactly the control's case set, dedup lost
like every other point on the segment.  So the swap is carried by a small
coordinate set.  And ``mx29_listset.py`` found that the top 24 coordinates by
contribution carry only **15.8%** of the attribution, and that the LIST-to-SET
regression is not a token flip at all but a trajectory divergence eight tokens
in.  So the swap is *not* local.

Both readings are about the same object measured two different ways, and neither
of them is the measurement this file makes.  The terminal diagonal is the last
thing that happens before the reader:

    z = W_head @ (RMSNorm(x) * d)

which is **exactly linear and exactly separable in d**.  The logit change from
swapping one coordinate is closed form and needs no gradient, no ablation sweep
and no second forward pass:

    dz = W_head[:, j] * x_hat_j * (d_B_j - d_A_j)

So for a reference continuation -- the sealed dossier text of the child that HELD
the lane -- the contribution of coordinate ``j`` to the margin between the token
that child actually emitted and its strongest competitor is a number, per
position, per lane.  Summed over a lane it is a 1,024-vector saying which
coordinates of the diagonal are holding that lane up and which are pushing it
over.

That makes the surgical construction obvious and free: take the solved twin's
diagonal everywhere it GAINS, and restore the control's values only on the
coordinates carrying the lane it LOSES.  Zero solve, zero capture, zero added
parameter, byte-identical storage -- the same legality argument
``mx37_blend.py`` already makes for a convex blend, on a set chosen by
measurement rather than by ``top-|delta|``.

**The gate that makes any of this mean anything.**  The reconstruction
``W_head @ (x_hat * d)`` is checked against the model's OWN logits before a
single attribution is written, and refused past 1e-3 relative.  An attribution
computed through a reader that is not the model's reader is a story about a
different model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

DIAG = "model.embedding_norm.weight"
SCHEMA = "neural-forge-matrix-wave37-coordinate-attribution-v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="the child whose diagonal is A -- the one that HOLDS the protected lane")
    ap.add_argument("--other", type=Path, required=True,
                    help="the sibling whose diagonal is B -- the one that gains lanes and loses it")
    ap.add_argument("--dossier-source", type=Path, required=True,
                    help="the sealed dossier of --source; its child_responses are the reference text")
    ap.add_argument("--dossier-other", type=Path,
                    help="the sealed dossier of --other, for the lanes B gains")
    ap.add_argument("--protect", action="append", default=None,
                    help="lane id whose capability is to be preserved; repeatable")
    ap.add_argument("--gain", action="append", default=None,
                    help="lane id B gains and which the surgery must not give back; repeatable")
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    protect = args.protect or ["coding-deduplicate"]
    gain = args.gain or ["writing-concise-email", "creative-continuity",
                         "uncertainty-insufficient-data"]

    import os
    import sys

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for sub in ("src", "experiments"):
        p = str(args.deployment / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402
    from transformers import AutoTokenizer, Lfm2ForCausalLM  # noqa: E402

    torch.set_grad_enabled(False)
    started = time.monotonic()

    a_tensors = load_file(str(args.source / "model.safetensors"))
    b_tensors = load_file(str(args.other / "model.safetensors"))
    differing = [
        k for k in sorted(a_tensors)
        if k not in b_tensors
        or a_tensors[k].shape != b_tensors[k].shape
        or not torch.equal(a_tensors[k], b_tensors[k])
    ]
    if differing != [DIAG]:
        raise SystemExit(
            f"REFUSED: the two children differ in {differing[:6]} (n={len(differing)}), "
            f"not in {DIAG} alone.  A coordinate attribution of these is not an "
            f"attribution of a diagonal."
        )
    d_a = a_tensors[DIAG].to(torch.float64)
    d_b = b_tensors[DIAG].to(torch.float64)
    delta = d_b - d_a
    hidden = int(d_a.shape[0])

    model = Lfm2ForCausalLM.from_pretrained(
        str(args.source), dtype=torch.float32
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.source))
    head = model.lm_head.weight.data.to(torch.float64)          # [V, H]
    eps = float(model.config.norm_eps)

    # The reader's own input, captured rather than reconstructed.
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()[0].to(torch.float64)

    handle = model.model.embedding_norm.register_forward_hook(hook)

    dossier_a = json.loads(args.dossier_source.read_text(encoding="utf-8"))
    responses = {r["id"]: r for r in dossier_a["child_responses"]}
    prompts = {r["id"]: r["prompt_messages"] for r in dossier_a["child_responses"]}

    per_lane: dict[str, Any] = {}
    gate_worst = 0.0
    for lane in sorted(set(protect) | set(gain)):
        row = responses.get(lane)
        if row is None:
            per_lane[lane] = {"absent": True}
            continue
        text = tokenizer.apply_chat_template(
            [dict(m) for m in prompts[lane]], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids
        cont_ids = tokenizer(
            row["response_verbatim"], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        ids = torch.cat([prompt_ids, cont_ids], dim=1).to(args.device)
        out = model(input_ids=ids, use_cache=False)
        x = captured["x"]                                        # [T, H], pre-norm
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
        x_hat = x / rms                                          # [T, H]

        # THE GATE.  Rebuild the model's own logits from the captured reader
        # input and refuse if the reconstruction is not the model's.
        rebuilt = (x_hat * d_a.to(x_hat.device)) @ head.to(x_hat.device).T
        actual = out.logits[0].to(torch.float64)
        rel = float((rebuilt - actual).norm() / actual.norm())
        gate_worst = max(gate_worst, rel)
        if rel > 1e-3:
            handle.remove()
            raise SystemExit(
                f"REFUSED on lane {lane}: the reconstruction W_head @ (x_hat * d) "
                f"differs from the model's own logits by {rel:.3e} relative.  An "
                f"attribution computed through a reader that is not the model's "
                f"reader is a story about a different model."
            )

        # Attribution over the CONTINUATION positions only: the prompt is shared
        # by both children and carries no capability claim.
        start = int(prompt_ids.shape[1]) - 1
        stop = int(ids.shape[1]) - 1
        contribution = torch.zeros(hidden, dtype=torch.float64)
        positions = 0
        flips = 0
        margin_a_total = margin_b_total = 0.0
        for t in range(start, stop):
            target = int(ids[0, t + 1])
            z_a = actual[t]
            masked = z_a.clone()
            masked[target] = float("-inf")
            rival = int(masked.argmax())
            margin_a = float(z_a[target] - z_a[rival])
            # The exact closed-form change of that margin under d_A -> d_B, and
            # its exact decomposition over coordinates.  No approximation: the
            # diagonal enters the logit linearly.
            w = (head[target] - head[rival]).to(x_hat.device)     # [H]
            g = (w * x_hat[t] * delta.to(x_hat.device)).cpu()     # [H]
            contribution += g
            margin_b = margin_a + float(g.sum())
            margin_a_total += margin_a
            margin_b_total += margin_b
            if margin_a > 0 >= margin_b:
                flips += 1
            positions += 1

        order = torch.argsort(contribution)                       # most negative first
        per_lane[lane] = {
            "reference_child": str(args.source),
            "continuation_tokens": positions,
            "reconstruction_relative_error": round(rel, 8),
            "mean_margin_under_A": round(margin_a_total / max(positions, 1), 5),
            "mean_margin_under_B": round(margin_b_total / max(positions, 1), 5),
            "positions_whose_margin_B_destroys": flips,
            "total_contribution": round(float(contribution.sum()), 5),
            "most_negative_coordinates": [int(i) for i in order[:64].tolist()],
            "most_positive_coordinates": [int(i) for i in order.flip(0)[:64].tolist()],
            "top24_share_of_absolute": round(
                float(contribution.abs().sort(descending=True).values[:24].sum()
                      / contribution.abs().sum()), 6
            ),
            "contribution": [round(float(v), 8) for v in contribution.tolist()],
        }
        print(
            f"{lane:>30}  n={positions:<4d} margin {per_lane[lane]['mean_margin_under_A']:>8.3f} "
            f"-> {per_lane[lane]['mean_margin_under_B']:>8.3f}  destroyed {flips:<3d} "
            f"top24 {per_lane[lane]['top24_share_of_absolute']:.3f}  gate {rel:.2e}",
            flush=True,
        )
    handle.remove()

    # ---- the surgical sets.  Coordinates that hurt the protected lanes most,
    # minus any coordinate that is also carrying a lane B gains -- restoring a
    # coordinate that the gain depends on would hand the gain back, which is the
    # whole failure mode of a top-|delta| partition.
    protect_score = torch.zeros(hidden, dtype=torch.float64)
    gain_score = torch.zeros(hidden, dtype=torch.float64)
    for lane in protect:
        row = per_lane.get(lane) or {}
        if "contribution" in row:
            protect_score += torch.tensor(row["contribution"], dtype=torch.float64)
    for lane in gain:
        row = per_lane.get(lane) or {}
        if "contribution" in row:
            gain_score += torch.tensor(row["contribution"], dtype=torch.float64)

    # A coordinate is a restore candidate when moving it A->B COSTS the protected
    # lane (negative protect score) and does not pay a gain lane (non-positive
    # gain score).  Ranked by how much it costs net of what it buys.
    net = protect_score - gain_score.clamp(min=0.0)
    ranked = torch.argsort(net)
    candidates = [int(i) for i in ranked.tolist() if float(protect_score[int(i)]) < 0.0]

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.source),
        "other": str(args.other),
        "protected_lanes": protect,
        "gain_lanes": gain,
        "hidden": hidden,
        "diagonal_delta_norm": round(float(delta.norm()), 6),
        "diagonal_coordinates_moved": int((delta != 0).sum()),
        "reconstruction_gate_worst_relative": round(gate_worst, 8),
        "per_lane": per_lane,
        "restore_candidates_ranked": candidates[:256],
        "restore_sets": {
            str(m): sorted(candidates[:m]) for m in (8, 24, 64, 128) if len(candidates) >= m
        },
        "wall_time_seconds": round(time.monotonic() - started, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"restore candidates: {len(candidates)} of {hidden}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
