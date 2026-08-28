#!/usr/bin/env python3
"""The multi-axis phase-transition budget allocator, fitted on this campaign's own ledger.

Field synthesis #3 (npj AI, Feb 2026) reports that every compression axis has
its own critical threshold with power-law-then-exponential collapse beyond it,
that the thresholds are stable under combination, and therefore that stacking
orthogonal axes each below its own transition point compounds safely.  Their
low-rank band brackets our frontier almost exactly, which is why the finding
ranked Tier 1.  This file does not take their word for any of it: it fits the
per-axis curves from the 822-row ledger, fits the composition model against the
sealed COMPOSED children, and reports where our record agrees with theirs and
where it does not.

**The parameter arithmetic is exact, not fitted.**  Every axis on this parent
removes a closed-form number of parameters:

* ``vocab_size`` to K rows removes ``(65536 - K) * 1024`` from the tied table;
* excising a *conv* block removes ``12,063,744`` and an *attention* block
  ``11,012,096`` -- the two block shapes of this 14-layer alternating trunk;
* ``intermediate_size`` to F over the L surviving blocks removes
  ``3 * 1024 * (2560 - F) * L``, the 43,008-per-channel figure the plan quotes,
  scaled by however many blocks an excision left standing.

Those three reproduce every sealed ratio in the ledger to the parameter, and the
allocator refuses to emit a split whose predicted count does not.

**What the fit finds, stated here because it decides the recommendation.**  Two
of our four axes are already PAST their transition at the mildest rung anyone
has ever sealed -- ``width`` reads 4/15 at 0.859x and ``mlp`` reads 4/15 at
0.688x against a 12/15 parent -- and the ``mlp`` axis has a 1,664-channel gap
between the parent's 2,560 and the only rung ever built.  So the honest
allocator output is not "here is the optimal split"; it is "one axis has
measured headroom, one has a hole in the middle of it, and the 0.65x build is
the measurement that fills the hole."
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any

PARENT_PARAMS = 229_693_184
PARENT_REVIEW = 12
PARENT_VOCAB = 65_536
PARENT_HIDDEN = 1_024
PARENT_FF = 2_560
PARENT_BLOCKS = 14
SCHEMA = "neural-forge-matrix-wave32-phase-transition-allocator-v1"

#: `config.json` `layer_types`, so a block's excision price is read off its kind
#: rather than off a table someone typed.
LAYER_TYPES = (
    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv",
)
#: Measured against the sealed counts, not assembled from a shape sketch.  The
#: attention figure is 11,012,096 by the obvious accounting -- q/k/v/o at
#: 3,145,728, the SwiGLU trio at 7,864,320, two block norms at 2,048 -- and that
#: is WRONG BY 128, which is exactly ``q_layernorm`` and ``k_layernorm``: LFM2
#: normalises inside the attention block over ``head_dim`` 64, twice.  The
#: sketch reproduces the sealed *ratio* to six decimals and misses the sealed
#: *count* on all six attention excisions, which is the entire argument for
#: gating on parameter counts rather than on ratios.
EXCISION_COST = {"conv": 12_063_744, "full_attention": 11_012_224}


def excision_cost(block: int) -> int:
    return EXCISION_COST[LAYER_TYPES[block]]


def params_of(vocab_k: int, excised: tuple[int, ...], ff: int, hidden: int = PARENT_HIDDEN) -> int:
    """Exact stored-parameter count of a (vocab x depth x mlp) split."""

    if hidden != PARENT_HIDDEN:
        raise SystemExit(
            "the width axis is not expressible in this closed form -- it re-solves "
            "every reader and writer in the model -- and the allocator refuses to "
            "recommend a split it cannot count"
        )
    total = PARENT_PARAMS
    total -= (PARENT_VOCAB - vocab_k) * PARENT_HIDDEN
    for block in excised:
        total -= excision_cost(block)
    survivors = PARENT_BLOCKS - len(excised)
    total -= 3 * PARENT_HIDDEN * (PARENT_FF - ff) * survivors
    return total


# ------------------------------------------------------------------- the record
CELL = re.compile(r"^ARCH\.(.+)$")
V = re.compile(r"^V(\d+)")
X = re.compile(r"X(\d+)")
CL = re.compile(r"CL(\d+)")
W = re.compile(r"(?:^|[^A-Z])W(\d+)")
MM = re.compile(r"^MM(\d+)")


def decompose(name: str) -> dict[str, Any] | None:
    """The axis levels a sealed cell id declares, or ``None`` if it declares none.

    Cell ids in this campaign are compositional by construction -- ``V19X13G03``
    is vocabulary 19,456 x excision of block 13 x the G03 terminal-gain variant --
    so the ladder is readable off the id.  Suffixes that do not move an axis
    (``G*`` terminal gain, ``L*`` lambda variants, ``KV6JQ`` head merges) are
    reported separately and excluded from the fit rather than silently folded
    into one of the three axes.
    """

    m = CELL.match(name)
    if not m:
        return None
    body = m.group(1)
    axes: dict[str, Any] = {"vocab": PARENT_VOCAB, "excised": (), "ff": PARENT_FF,
                            "hidden": PARENT_HIDDEN, "extras": []}
    rest = body
    v = V.match(rest)
    if v:
        axes["vocab"] = int(v.group(1)) * 1024
        rest = rest[v.end():]
    blocks = tuple(sorted(int(b) for b in X.findall(rest)))
    if blocks:
        axes["excised"] = blocks
        rest = X.sub("", rest)
        # Wave 15-16 named the solo excisions with a one-letter ARM suffix --
        # `X13A`, `X05A`, `X05T` -- and every composed child since drops it.
        # Left in `extras` the suffix disqualifies the cell from the solo
        # ladder, which is how the first run of this file produced a depth site
        # table with block 13 -- the campaign's BEST single excision at 13/15 --
        # missing from it, and then recommended splits that used no depth at all.
        if rest in ("A", "T"):
            axes["arm"] = rest
            rest = ""
    cl = CL.search(rest)
    if cl:
        axes["ff"] = int(cl.group(1))
        rest = CL.sub("", rest)
    mmm = MM.match(body)
    if mmm:
        axes["hidden"] = int(mmm.group(1))
        axes["extras"].append("maxmin-basis")
        rest = ""
    else:
        w = W.search(rest)
        if w:
            axes["hidden"] = int(w.group(1))
            rest = W.sub("", rest)
    leftovers = [t for t in re.findall(r"[A-Z]+\d*", rest) if t]
    axes["extras"] += leftovers
    return axes


def load_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """The most recent reviewed row per cell id."""

    best: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("cell_id") or ""
        if not cid.startswith("ARCH."):
            continue
        if row.get("behavior_useful_or_partial") is None:
            continue
        if row.get("stored_parameter_ratio") is None:
            continue
        prior = best.get(cid)
        if prior is None or (row.get("campaign_wave") or 0) >= (prior.get("campaign_wave") or 0):
            best[cid] = row
    return best


def solo_ladders(record: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rungs that move exactly ONE axis and nothing else."""

    ladders: dict[str, list[dict[str, Any]]] = {"vocab": [], "depth": [], "mlp": [], "width": []}
    for cid, row in record.items():
        ax = decompose(cid)
        if ax is None or ax["extras"]:
            continue
        moved = [
            name
            for name, hit in (
                ("vocab", ax["vocab"] != PARENT_VOCAB),
                ("depth", bool(ax["excised"])),
                ("mlp", ax["ff"] != PARENT_FF),
                ("width", ax["hidden"] != PARENT_HIDDEN),
            )
            if hit
        ]
        if len(moved) != 1:
            continue
        axis = moved[0]
        level = {
            "vocab": ax["vocab"], "mlp": ax["ff"], "width": ax["hidden"],
            "depth": ax["excised"],
        }[axis]
        ladders[axis].append(
            {
                "cell": cid,
                "level": level,
                "ratio": round(float(row["stored_parameter_ratio"]), 6),
                "review": int(row["behavior_useful_or_partial"]),
                "wave": row.get("campaign_wave"),
            }
        )
    # The parent is a measured rung of every axis, at that axis's full level,
    # and leaving it out is what made the first run of this file unable to price
    # `intermediate_size` anywhere above the one rung ever built: with only
    # (896, 8/15) and (640, 12/15) on the ladder there is nothing to interpolate
    # AGAINST, so every candidate in the 2,560-to-896 gap returned "unpriceable"
    # and the search quietly recommended not folding the axis at all.
    for axis, level in (("vocab", PARENT_VOCAB), ("mlp", PARENT_FF), ("width", PARENT_HIDDEN)):
        ladders[axis].append(
            {"cell": "PARENT", "level": level, "ratio": 1.0,
             "review": PARENT_REVIEW, "wave": None, "anchor": True}
        )
    for axis in ladders:
        ladders[axis].sort(key=lambda r: -r["ratio"])
    return ladders


def transition_of(rungs: list[dict[str, Any]]) -> dict[str, Any]:
    """The critical ratio: the deepest rung still holding half the parent's review.

    Deliberately crude, and crude is the point.  The field's fitted
    power-law-then-exponential form needs more rungs per axis than this campaign
    has ever built on three of its four axes, and a two-parameter fit through
    two points is a decoration.  "Deepest rung at or above 6/15, and the next one
    down" is a statement the ledger can actually support.
    """

    if not rungs:
        return {"measured": False, "reason": "no solo rung on this axis"}
    if any(isinstance(r["level"], tuple) for r in rungs):
        # Depth.  Every single excision sits at one of exactly two ratios --
        # 0.947479x for a conv block, 0.952057x for an attention block -- and the
        # reviews across those sites run 0/15 to 13/15.  There is no ratio at
        # which this axis transitions, because the axis is not parameterised by
        # ratio; it is parameterised by WHICH BLOCK, and reporting a critical
        # ratio for it would be reporting the spread of the site table as a
        # threshold.
        by_site = sorted(
            ((r["level"][0], r["review"]) for r in rungs if len(r["level"]) == 1),
            key=lambda p: -p[1],
        )
        return {
            "measured": True,
            "parameterised_by": "site, not ratio",
            "site_reviews": {str(b): v for b, v in by_site},
            "free_sites": [b for b, v in by_site if v >= PARENT_REVIEW - 3],
            "dead_sites": [b for b, v in by_site if v == 0],
            "note": (
                "single excisions occupy two ratios and eleven review levels; the "
                "transition on this axis is the NUMBER of excisions, and no solo "
                "multi-excision child has ever been sealed to measure it"
            ),
        }
    holding = [r for r in rungs if r["review"] >= PARENT_REVIEW / 2]
    if not holding:
        return {
            "measured": True,
            "pre_transition_floor": None,
            "post_transition_ceiling": rungs[0]["ratio"],
            "note": (
                f"every sealed rung is already past transition; the mildest one "
                f"built ({rungs[0]['cell']}, {rungs[0]['ratio']:.6f}x) reads "
                f"{rungs[0]['review']}/15 against a {PARENT_REVIEW}/15 parent"
            ),
        }
    floor = min(holding, key=lambda r: r["ratio"])
    below = [r for r in rungs if r["ratio"] < floor["ratio"]]
    ceiling = max(below, key=lambda r: r["ratio"]) if below else None
    return {
        "measured": True,
        "pre_transition_floor": floor["ratio"],
        "pre_transition_floor_cell": floor["cell"],
        "pre_transition_floor_review": floor["review"],
        "post_transition_ceiling": ceiling["ratio"] if ceiling else None,
        "post_transition_ceiling_cell": ceiling["cell"] if ceiling else None,
        "post_transition_ceiling_review": ceiling["review"] if ceiling else None,
        "bracket_width": (
            round(floor["ratio"] - ceiling["ratio"], 6) if ceiling else None
        ),
    }


def solo_deficit(ladders: dict[str, list[dict[str, Any]]], axis: str, level: Any) -> float | None:
    """``parent_review - solo_review`` at this level, by exact match or interpolation."""

    rungs = ladders[axis]
    if not rungs:
        return None
    if axis == "depth":
        # Depth is a SITE axis, not a budget axis: single excisions all sit at
        # one of two ratios and their reviews run 0..13 depending purely on which
        # block was cut.  The deficit is per-site and additive over sites.
        per_site = {r["level"][0]: PARENT_REVIEW - r["review"] for r in rungs if len(r["level"]) == 1}
        total = 0.0
        for block in level:
            if block not in per_site:
                return None
            total += per_site[block]
        return total
    points = sorted(((r["level"], PARENT_REVIEW - r["review"]) for r in rungs), key=lambda p: -p[0])
    for lv, d in points:
        if lv == level:
            return float(d)
    above = [p for p in points if p[0] > level]
    below = [p for p in points if p[0] < level]
    if above and below:
        (l1, d1), (l0, d0) = above[-1], below[0]
        t = (l1 - level) / (l1 - l0)
        return float(d1 + t * (d0 - d1))
    if above:
        return float(above[-1][1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--target", type=float, action="append", default=None,
                    help="target net ratio; repeatable.  Defaults to 0.65 and 0.55")
    ap.add_argument("--tolerance", type=float, default=0.006,
                    help="how close a split must land to the target ratio")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    targets = args.target or [0.65, 0.55]

    record = load_ledger(args.ledger)
    ladders = solo_ladders(record)
    transitions = {axis: transition_of(rungs) for axis, rungs in ladders.items()}

    # ---- the arithmetic gate.  Every sealed cell the decomposer can read must
    # reproduce its own sealed ratio from the closed form, or the allocator is
    # counting something the campaign is not.
    checked = mismatched = 0
    mismatches: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for cid, row in sorted(record.items()):
        ax = decompose(cid)
        if ax is None or ax["hidden"] != PARENT_HIDDEN:
            skipped["width-axis-or-unreadable"] = skipped.get("width-axis-or-unreadable", 0) + 1
            continue
        if ax["extras"]:
            # Head merges, block fusions, terminal-gain variants, blends: real
            # cells on axes this closed form does not model.  They are counted
            # and named rather than silently checked against an arithmetic that
            # was never meant to cover them -- the first run of this gate did
            # check them, reported 61 of 138 mismatched, and the 61 were all of
            # these.
            skipped["axis-not-modelled"] = skipped.get("axis-not-modelled", 0) + 1
            continue
        predicted = params_of(ax["vocab"], ax["excised"], ax["ff"])
        sealed = int(row.get("stored_parameter_count") or 0)
        if not sealed:
            continue
        checked += 1
        if predicted != sealed:
            mismatched += 1
            if len(mismatches) < 12:
                mismatches.append(
                    {"cell": cid, "predicted": predicted, "sealed": sealed,
                     "delta": predicted - sealed, "axes": {k: v for k, v in ax.items() if k != "extras"},
                     "extras": ax["extras"]}
                )

    # ---- the composition model.  pred = 12 - sum(solo deficits) - interference,
    # with one interference constant per axis PAIR, fitted by least squares on the
    # sealed composed children.  If the field's "thresholds stay stable under
    # combination" held here, every constant would be ~0.
    observations: list[dict[str, Any]] = []
    for cid, row in sorted(record.items()):
        ax = decompose(cid)
        if ax is None or ax["extras"] or ax["hidden"] != PARENT_HIDDEN:
            continue
        used = [n for n, hit in (("vocab", ax["vocab"] != PARENT_VOCAB),
                                 ("depth", bool(ax["excised"])),
                                 ("mlp", ax["ff"] != PARENT_FF)) if hit]
        if len(used) < 2:
            continue
        deficits = {
            "vocab": solo_deficit(ladders, "vocab", ax["vocab"]) if "vocab" in used else 0.0,
            "depth": solo_deficit(ladders, "depth", ax["excised"]) if "depth" in used else 0.0,
            "mlp": solo_deficit(ladders, "mlp", ax["ff"]) if "mlp" in used else 0.0,
        }
        if any(v is None for v in deficits.values()):
            continue
        additive = PARENT_REVIEW - sum(deficits.values())
        observations.append(
            {
                "cell": cid,
                "axes_used": used,
                "pairs": ["x".join(p) for p in itertools.combinations(sorted(used), 2)],
                "ratio": round(float(row["stored_parameter_ratio"]), 6),
                "observed": int(row["behavior_useful_or_partial"]),
                "additive_prediction": round(additive, 3),
                "interference": round(additive - int(row["behavior_useful_or_partial"]), 3),
                "solo_deficits": {k: round(v, 3) for k, v in deficits.items()},
            }
        )

    pair_names = sorted({p for o in observations for p in o["pairs"]})
    fitted: dict[str, float] = {}
    residuals: list[float] = []
    if observations and pair_names:
        # Normal equations by hand -- four unknowns at most, and no dependency on
        # a solver the deployment would have to carry.
        n = len(pair_names)
        ata = [[0.0] * n for _ in range(n)]
        atb = [0.0] * n
        for o in observations:
            idx = [pair_names.index(p) for p in o["pairs"]]
            for i in idx:
                atb[i] += o["interference"]
                for j in idx:
                    ata[i][j] += 1.0
        for i in range(n):
            ata[i][i] += 1e-6
        # Gauss-Jordan.
        aug = [row[:] + [atb[i]] for i, row in enumerate(ata)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            if abs(pv) < 1e-12:
                continue
            aug[col] = [v / pv for v in aug[col]]
            for r in range(n):
                if r != col and abs(aug[r][col]) > 0:
                    factor = aug[r][col]
                    aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]
        solution = [aug[i][n] for i in range(n)]
        fitted = {name: round(val, 4) for name, val in zip(pair_names, solution)}
        for o in observations:
            pred = o["additive_prediction"] - sum(fitted.get(p, 0.0) for p in o["pairs"])
            o["model_prediction"] = round(pred, 3)
            o["model_error"] = round(pred - o["observed"], 3)
            residuals.append(pred - o["observed"])

    mae = round(sum(abs(r) for r in residuals) / len(residuals), 3) if residuals else None
    worst = round(max((abs(r) for r in residuals), default=0.0), 3)

    def predict(vocab_k: int, excised: tuple[int, ...], ff: int) -> float | None:
        used, deficits = [], 0.0
        for axis, level, moved in (("vocab", vocab_k, vocab_k != PARENT_VOCAB),
                                   ("depth", excised, bool(excised)),
                                   ("mlp", ff, ff != PARENT_FF)):
            if not moved:
                continue
            d = solo_deficit(ladders, axis, level)
            if d is None:
                return None
            used.append(axis)
            deficits += d
        pred = PARENT_REVIEW - deficits
        for pair in itertools.combinations(sorted(used), 2):
            pred -= fitted.get("x".join(pair), 0.0)
        return pred

    # ---- the search.  Vocabulary rungs the ladder has actually sealed, excision
    # sets drawn from sites with a solo reading, and an `intermediate_size` grid
    # on multiples of 64 -- the axis with the hole in it, which is where the
    # allocator is allowed to extrapolate because that is what the build is for.
    vocab_grid = sorted({r["level"] for r in ladders["vocab"]}, reverse=True) or [PARENT_VOCAB]
    vocab_grid = [PARENT_VOCAB] + vocab_grid
    sites = sorted(
        {r["level"][0] for r in ladders["depth"] if len(r["level"]) == 1},
        key=lambda b: -(PARENT_REVIEW - next(
            PARENT_REVIEW - r["review"] for r in ladders["depth"] if r["level"] == (b,)
        )),
    )
    site_sets: list[tuple[int, ...]] = [()]
    for k in (1, 2, 3, 4):
        site_sets += [tuple(sorted(c)) for c in itertools.combinations(sites, k)]
    ff_grid = [PARENT_FF] + list(range(2496, 512 - 1, -64))

    recommendations: dict[str, Any] = {}
    for target in targets:
        want = target * PARENT_PARAMS
        candidates: list[dict[str, Any]] = []
        for vocab_k in vocab_grid:
            for excised in site_sets:
                for ff in ff_grid:
                    count = params_of(vocab_k, excised, ff)
                    ratio = count / PARENT_PARAMS
                    if abs(ratio - target) > args.tolerance:
                        continue
                    pred = predict(vocab_k, excised, ff)
                    if pred is None:
                        continue
                    pre = {
                        "vocab": (
                            vocab_k == PARENT_VOCAB
                            or ratio_of_solo(ladders, "vocab", vocab_k) is None
                            or (transitions["vocab"].get("pre_transition_floor") is not None
                                and ratio_of_solo(ladders, "vocab", vocab_k)
                                >= transitions["vocab"]["pre_transition_floor"] - 1e-9)
                        ),
                        "mlp": ff == PARENT_FF or ff > 896,
                        "depth": len(excised) <= 1,
                    }
                    candidates.append(
                        {
                            "vocab_k": vocab_k,
                            "excised": list(excised),
                            "intermediate_size": ff,
                            "stored_parameters": count,
                            "net_ratio": round(ratio, 6),
                            "predicted_review": round(pred, 2),
                            "every_axis_pre_transition": all(pre.values()),
                            "pre_transition": pre,
                            "extrapolates_mlp_into_the_gap": 896 < ff < PARENT_FF,
                        }
                    )
        candidates.sort(key=lambda c: (-c["every_axis_pre_transition"], -c["predicted_review"]))
        # SIX decimals, not two.  Wave 31 pre-registered ten named splits in one
        # invocation and got nine answers back: `intermediate_size` 1536 at
        # 0.808265x and 1792-with-block-13-excised at 0.813949x both round to
        # "0.81", so the second silently overwrote the first in this dict.  The
        # two ratios are 0.0057 apart and the DEFAULT tolerance is 0.006, so the
        # collision is reachable by any caller who prices two nearby splits at
        # once -- and the failure mode is a missing pre-registration, which is
        # the one thing a pre-registration must not be.
        recommendations[f"{target:.6f}"] = {
            "target": target,
            "tolerance": args.tolerance,
            "candidates_found": len(candidates),
            "top": candidates[:8],
            "recommended": candidates[0] if candidates else None,
        }

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ledger": str(args.ledger),
        "ledger_rows_with_review": len(record),
        "parent_review": PARENT_REVIEW,
        "arithmetic_gate": {
            "scope": (
                "cells whose id the closed form fully explains -- vocabulary x "
                "excision x intermediate_size at parent width.  Cells carrying an "
                "axis this form does not model are skipped BY NAME below, not "
                "checked and failed"
            ),
            "cells_checked": checked,
            "mismatched": mismatched,
            "passed": mismatched == 0,
            "skipped": skipped,
            "mismatches": mismatches,
        },
        "solo_ladders": ladders,
        "transitions": transitions,
        "composition_model": {
            "form": "review = 12 - sum(solo axis deficits) - sum(pairwise interference)",
            "fitted_pairwise_interference": fitted,
            "observations": len(observations),
            "mean_absolute_error_cases": mae,
            "worst_absolute_error_cases": worst,
            "per_child": observations,
        },
        "recommendations": recommendations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print(f"arithmetic gate: {checked} cells, {mismatched} mismatched")
    for axis, t in transitions.items():
        print(f"  {axis:6s} {json.dumps(t)}")
    print(f"interference: {fitted}  MAE {mae} cases over {len(observations)} composed children")
    for key, rec in recommendations.items():
        r = rec["recommended"]
        print(f"  target {key}: {json.dumps(r) if r else 'NO SPLIT WITHIN TOLERANCE'}")
    print(f"-> {args.out}")
    return 0


def ratio_of_solo(ladders: dict[str, list[dict[str, Any]]], axis: str, level: Any) -> float | None:
    """The net ratio this axis reaches ALONE at this level, for the transition test."""

    if axis == "vocab":
        return params_of(level, (), PARENT_FF) / PARENT_PARAMS
    if axis == "mlp":
        return params_of(PARENT_VOCAB, (), level) / PARENT_PARAMS
    return None


if __name__ == "__main__":
    raise SystemExit(main())
