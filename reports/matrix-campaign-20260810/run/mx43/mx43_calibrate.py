#!/usr/bin/env python3
"""C4.  Calibrate the extension-v2 surface against BOTH readings of the estate.

Wave 33 retired the extension-v1 verdict string, and it retired it for a reason
worth restating: pooled over the four waves that ran it, v1's rank correlation
against the sealed fifteen was **0.6926** over sixty observations, and its
per-wave rho ran 0.85 / 0.65 / 0.74 / **0.35**.  The last of those is not an
instrument that broke; wave 33's own reproduction control showed every repeated
child scoring identically across the wave boundary.  It is an instrument whose
rho is a property of the POPULATION's spread, which means a per-wave rho on a
narrow population is a number that cannot be acted on.

So this file does three things v1's reporting never did.

**1. It scores against two references, not one.**  The sealed frozen-fifteen
count is one ordering of this estate.  The canonical intelligence count -- lanes
of signal out of fifteen with loops, echoes, cap-rides and empties struck as
compression artifacts -- is a different ordering, and on the re-read population
the two disagree by up to five cases.  A second surface that tracks the sealed
ordering and not the canonical one is useful for one purpose and misleading for
the other, so both rhos are reported and neither is called "the" rho.

**2. It reports the spread it was measured on, in the same line as the rho.**
The wave-32 lesson, mechanised: a rho quoted without the range of its reference
column is uninterpretable.  Every rho below carries ``n``, the reference range
and a permutation null, and a rho whose observed value sits inside the null's
central 90 per cent is reported as NOT DISTINGUISHABLE FROM CHANCE in those
words rather than as a number.

**3. It refuses to adopt on a tie.**  The bar is v1's pooled 0.6926.  Beating it
requires the point estimate to exceed it AND the leave-one-out minimum to stay
above the null's 95th percentile -- because a correlation that one child can
delete is a correlation about that child.  A surface that does not clear that is
reported NO-BETTER, which is a result and not a failure.

This file computes nothing about folding.  It prices an instrument.

    mx43_calibrate.py --score <extension2-score.json> --reference <references.json>
                      --v1-pooled-rho 0.6926 --out <calibration.json>

``--reference`` is ``{"LABEL": {"sealed_15": int, "canonical_15": int,
"net_ratio": float, "provenance": "..."}}``, and every label in the score file
must appear in it or this refuses: a calibration that silently drops the rows it
has no reference for is a calibration of whatever is left.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

SCHEMA = "neural-forge-matrix-wave43-extension2-calibration-v1"

#: The estimator is IMPORTED rather than rewritten, from the file wave 33 used to
#: compute the 0.6926 this wave is measured against.  Two estimators that agree
#: on ties by accident are two estimators.
try:  # pragma: no cover - import shape depends on how this is invoked
    from mx43_instrument import spearman
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mx43_instrument import spearman


def permutation_null(
    reference: list[float], observed_scores: list[float], draws: int, seed: int
) -> dict[str, Any]:
    """The rho this surface would report if it carried no information at all.

    The reference column is held FIXED and the surface's own scores are shuffled,
    so the null inherits the reference's tie structure -- which matters here,
    because a population of deep children is mostly ties at the bottom and a null
    that ignores that is optimistic about how hard the ordering was.
    """
    rng = random.Random(seed)
    pool = list(observed_scores)
    draws_out: list[float] = []
    for _ in range(draws):
        rng.shuffle(pool)
        value = spearman(list(zip(reference, pool)))
        if value is not None:
            draws_out.append(value)
    draws_out.sort()

    def q(p: float) -> float | None:
        if not draws_out:
            return None
        idx = min(len(draws_out) - 1, max(0, int(round(p * (len(draws_out) - 1)))))
        return round(draws_out[idx], 4)

    return {
        "draws": len(draws_out),
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "seed": seed,
    }


def leave_one_out(pairs: list[tuple[float, float]], labels: list[str]) -> dict[str, Any]:
    """Which single row the correlation depends on, named."""
    rows = []
    for i, label in enumerate(labels):
        rest = pairs[:i] + pairs[i + 1:]
        rows.append({"dropped": label, "rho": spearman(rest)})
    values = [r["rho"] for r in rows if r["rho"] is not None]
    return {
        "per_row": rows,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "most_influential": (
            min(rows, key=lambda r: (r["rho"] is None, r["rho"]))["dropped"] if rows else None
        ),
    }


def against(
    label: str,
    reference_key: str,
    scored: list[dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    labels = [c["label"] for c in scored]
    ref = [float(reference[l][reference_key]) for l in labels]
    obs = [float(c["held"]) for c in scored]
    pairs = list(zip(ref, obs))
    rho = spearman(pairs)
    null = permutation_null(ref, obs, draws=draws, seed=seed)
    loo = leave_one_out(pairs, labels)
    distinguishable = (
        rho is not None
        and null["p95"] is not None
        and rho > null["p95"]
    )
    return {
        "reference": label,
        "reference_key": reference_key,
        "n": len(pairs),
        "reference_range": [min(ref), max(ref)] if ref else None,
        "reference_distinct_values": len(set(ref)),
        "surface_range": [min(obs), max(obs)] if obs else None,
        "surface_distinct_values": len(set(obs)),
        "rho": rho,
        "permutation_null": null,
        "distinguishable_from_chance": bool(distinguishable),
        "leave_one_out": loo,
        "verdict_line": (
            f"rho {rho} on n={len(pairs)} over reference range "
            f"{min(ref) if ref else None}-{max(ref) if ref else None}; "
            f"null p95 {null['p95']}; "
            + ("ABOVE CHANCE" if distinguishable else "NOT DISTINGUISHABLE FROM CHANCE")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, required=True,
                    help="mx43_extrubric2.py's output for the population")
    ap.add_argument("--reference", type=Path, required=True,
                    help="sealed and canonical counts per label")
    ap.add_argument("--v1-pooled-rho", type=float, default=0.6926,
                    help="the retired v1 surface's pooled rho, the bar to beat")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    score = json.loads(args.score.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    refs = reference["children"] if "children" in reference else reference
    scored = score["children"]

    missing = sorted({c["label"] for c in scored} - set(refs))
    if missing:
        raise SystemExit(
            "REFUSED: no sealed/canonical reference for "
            f"{missing}.  A calibration that drops the rows it cannot reference "
            "is a calibration of whatever is left."
        )
    for key in ("sealed_15", "canonical_15"):
        absent = sorted(l for l in {c["label"] for c in scored} if key not in refs[l])
        if absent:
            raise SystemExit(f"REFUSED: {absent} carry no {key}")

    vs_sealed = against("sealed frozen fifteen", "sealed_15", scored, refs,
                        args.draws, args.seed)
    vs_canon = against("canonical intelligence count", "canonical_15", scored, refs,
                       args.draws, args.seed + 1)

    # The bar, applied in the two clauses the docstring states.
    def clears(block: dict[str, Any]) -> bool:
        return bool(
            block["rho"] is not None
            and block["rho"] > args.v1_pooled_rho
            and block["leave_one_out"]["minimum"] is not None
            and block["permutation_null"]["p95"] is not None
            and block["leave_one_out"]["minimum"] > block["permutation_null"]["p95"]
        )

    verdicts = {
        "vs_sealed": "BETTER THAN V1" if clears(vs_sealed) else "NO-BETTER",
        "vs_canonical": "BETTER THAN V1" if clears(vs_canon) else "NO-BETTER",
    }
    body = {
        "schema": SCHEMA,
        "purpose": (
            "Price the extension-v2 surface against BOTH the sealed frozen-fifteen "
            "ordering and the canonical intelligence ordering, on a population "
            "whose spread is reported in the same line as every rho.  It prices an "
            "instrument.  It says nothing about folding and moves no frontier."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank2_sha256": score.get("bank_sha256"),
        "prompt_rules": score.get("prompt_rules"),
        "population": [
            {
                "label": c["label"],
                "extension2_30": c["held"],
                "held_A": c.get("held_A"),
                "held_B": c.get("held_B"),
                "charged_prompts": c.get("charged_prompts"),
                "lanes_with_a_hold": c.get("lanes_with_a_hold"),
                "sealed_15": refs[c["label"]]["sealed_15"],
                "canonical_15": refs[c["label"]]["canonical_15"],
                "net_ratio": refs[c["label"]].get("net_ratio"),
                "provenance": refs[c["label"]].get("provenance"),
            }
            for c in scored
        ],
        "v1_bar": {
            "pooled_rho": args.v1_pooled_rho,
            "source": (
                "wave 33 instrument-diagnosis.json, 60 observations / 58 distinct "
                "children over waves 29-32, single bank, reproduction control passed"
            ),
            "per_wave": {"wave29": 0.8517, "wave30": 0.6544, "wave31": 0.7411, "wave32": 0.3537},
        },
        "vs_sealed": vs_sealed,
        "vs_canonical": vs_canon,
        "adoption_rule": (
            "point estimate above the v1 bar AND leave-one-out minimum above the "
            "permutation null's p95.  A correlation one child can delete is a "
            "correlation about that child."
        ),
        "verdicts": verdicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")

    print(f"population {len(scored)}")
    print("  vs sealed    :", vs_sealed["verdict_line"])
    print("  vs canonical :", vs_canon["verdict_line"])
    print(f"  v1 bar {args.v1_pooled_rho}")
    print(f"  VERDICT vs sealed    {verdicts['vs_sealed']}")
    print(f"  VERDICT vs canonical {verdicts['vs_canonical']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
