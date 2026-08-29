#!/usr/bin/env python3
"""I1: what the extension surface actually is, measured over four waves of it.

Wave 32 read Spearman **0.3537** against sealed counts and its own scorer
printed `SCRAMBLED`.  The three waves before it read 0.8517, 0.6544 and 0.7411.
A number that walks that far is either an instrument that broke or an instrument
that was never characterised, and the difference is decidable from artifacts
that already exist: every wave since 29 committed `waveNN/extension-score.json`
with a per-prompt `held` flag for every child and the sealed 15-case count
beside it.  **Sixty child observations, four waves, one frozen bank
(`c9d37004c901d757...`), one frozen rubric.**  Nothing here decodes a token.

**Why a per-wave Spearman was the wrong statistic to publish.**  Spearman is a
property of the SAMPLE as much as of the instrument.  Wave 29 scored sixteen
children spanning sealed 0 to 14; wave 32 scored seventeen spanning 0 to 8 with
thirteen of them inside a five-case window.  A rank correlation computed on a
sample with no spread is small even when every measurement is as good as it ever
was, and reporting it per wave means reporting a different population each time
and attributing the difference to the instrument.

So this file computes three things a per-wave rho cannot say:

1. **Band-restricted rho.**  The hypothesis the wave was sent with: the surface
   may be valid above some sealed count and garbage below it, where children are
   degenerate and the rubric is scoring fluent noise.  Computed on the pooled
   sixty at every cut point, so the band is read off the data rather than
   guessed.
2. **Gap-concordance**, which is the honest form of the question anyone actually
   asks the surface.  For every ORDERED PAIR of children with sealed counts g
   apart, how often does the extension score put them in the same order?  This
   is a resolution curve, it does not depend on the sample's spread the way rho
   does, and it answers "may I use this to break a one-case tie" directly.
3. **Per-prompt and per-family discrimination.**  Each of the thirty rubric
   entries is scored on its own: the rank correlation between that prompt's
   `held` indicator and the sealed count over the pooled sixty.  A prompt whose
   correlation is negative is not a weak instrument, it is an ANTI-instrument,
   and it is contributing to the total with the wrong sign.

Plus one control that costs nothing and is worth more than any of them: two
children -- `V12CL1792` and `V16CL1792` -- were decoded on this bank in wave 31
AND in wave 32, from checkpoints retained across the wave boundary.  If their
scores differ, the surface is not reproducible and nothing above matters.

Usage::

    mx39_instrument.py --campaign <reports/matrix-campaign-20260810> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

WAVES = ("wave29", "wave30", "wave31", "wave32")


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Wave 29's estimator, byte-for-byte, so the numbers are comparable."""
    if len(pairs) < 3:
        return None

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    xs = rank([p[0] for p in pairs])
    ys = rank([p[1] for p in pairs])
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return round(num / den, 4) if den else None


def family(prompt_id: str) -> str:
    m = re.match(r"ext-([a-z]+)-", prompt_id)
    return m.group(1) if m else prompt_id


def collect(campaign: Path) -> list[dict[str, Any]]:
    """One row per (wave, child): sealed count, extension count, per-prompt held."""
    rows: list[dict[str, Any]] = []
    for wave in WAVES:
        doc = json.loads((campaign / wave / "extension-score.json").read_text(encoding="utf-8"))
        sealed = {r["label"]: r["sealed_15"] for r in doc["retro_score"]["table"]}
        for child in doc["children"]:
            label = child["label"]
            if label not in sealed:
                continue
            rows.append({
                "wave": wave,
                "label": label,
                "sealed_15": sealed[label],
                "extension_30": child["held"],
                "charged_lanes": child["charged_lanes"],
                "bank_sha256": child.get("bank_sha256"),
                "held": {p["id"]: bool(p["held"]) for p in child["per_prompt"]},
                # The SAME hold with the degeneracy veto closed.  `score_one`
                # charges cap, empty, echo, loop and word-substitution and then
                # vetoes a hold on `empty` or `echo` ONLY, so a capped 1,024-token
                # repetition of the word "not" satisfies any rubric entry whose
                # requirement list is empty and whose word floor it clears.  That
                # is not hypothetical: it is three of `ARCH.BM50N`'s three.
                "held_strict": {p["id"]: bool(p["held"]) and not p["charged"]
                                for p in child["per_prompt"]},
            })
    return rows


def reproduction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The free control: any child decoded on this bank in two different waves."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    repeated = {k: v for k, v in by_label.items() if len(v) > 1}
    checks = []
    for label, obs in sorted(repeated.items()):
        scores = sorted({o["extension_30"] for o in obs})
        held_sets = {frozenset(k for k, v in o["held"].items() if v) for o in obs}
        checks.append({
            "label": label,
            "waves": [o["wave"] for o in obs],
            "extension_30": [o["extension_30"] for o in obs],
            "sealed_15": [o["sealed_15"] for o in obs],
            "score_identical": len(scores) == 1,
            "held_set_identical": len(held_sets) == 1,
        })
    return {
        "children_scored_twice": len(checks),
        "checks": checks,
        "verdict": (
            "REPRODUCIBLE: every repeated child scores identically and holds the "
            "identical prompt set across the wave boundary, so a change in rho is a "
            "change in the POPULATION and not in the instrument"
            if checks and all(c["score_identical"] and c["held_set_identical"] for c in checks)
            else "NOT REPRODUCIBLE: a repeated child moved; the instrument is not stable"
            if checks else "no repeated child"
        ),
    }


def bands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rho above and below every cut point, on the pooled set."""
    out = []
    for cut in range(0, 12):
        above = [(r["sealed_15"], r["extension_30"]) for r in rows if r["sealed_15"] >= cut]
        below = [(r["sealed_15"], r["extension_30"]) for r in rows if r["sealed_15"] < cut]
        out.append({
            "cut": cut,
            "n_at_or_above": len(above),
            "rho_at_or_above": spearman(above),
            "n_below": len(below),
            "rho_below": spearman(below),
        })
    return out


def gap_concordance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For every ordered pair g sealed cases apart, does the surface agree?

    Ties in the extension score are counted as their own outcome and never
    silently credited: a surface that cannot separate two children is not a
    surface that ordered them correctly.
    """
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: {"agree": 0, "tie": 0, "invert": 0})
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["label"] == b["label"]:
                continue           # the reproduction control, not an ordering test
            gap = a["sealed_15"] - b["sealed_15"]
            if gap == 0:
                continue
            hi, lo = (a, b) if gap > 0 else (b, a)
            bucket = buckets[abs(gap)]
            if hi["extension_30"] > lo["extension_30"]:
                bucket["agree"] += 1
            elif hi["extension_30"] == lo["extension_30"]:
                bucket["tie"] += 1
            else:
                bucket["invert"] += 1
    table = []
    for gap in sorted(buckets):
        b = buckets[gap]
        n = b["agree"] + b["tie"] + b["invert"]
        table.append({
            "sealed_gap": gap,
            "pairs": n,
            "agree": b["agree"],
            "tie": b["tie"],
            "invert": b["invert"],
            "agree_pct": round(100.0 * b["agree"] / n, 1),
            "invert_pct": round(100.0 * b["invert"] / n, 1),
        })
    return {"table": table}


def pair_spread(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per wave: how far apart, in sealed cases, are the pairs it asked about?

    This is the statistic a per-wave rho hides.  Gap-concordance says the surface
    resolves a large sealed difference and not a small one; if a wave's own
    population is mostly small differences, its rho is low BY CONSTRUCTION and
    says nothing about the instrument.
    """
    out = []
    for wave in WAVES:
        sub = [r for r in rows if r["wave"] == wave]
        gaps = [abs(a["sealed_15"] - b["sealed_15"])
                for i, a in enumerate(sub) for b in sub[i + 1:]]
        gaps = [g for g in gaps if g > 0]
        n = len(gaps) or 1
        out.append({
            "wave": wave,
            "children": len(sub),
            "ordered_pairs": len(gaps),
            "mean_gap": round(sum(gaps) / n, 2),
            "pairs_gap_ge_4_pct": round(100.0 * sum(1 for g in gaps if g >= 4) / n, 1),
            "pairs_gap_le_2_pct": round(100.0 * sum(1 for g in gaps if g <= 2) / n, 1),
            "sealed_span": [min(r["sealed_15"] for r in sub), max(r["sealed_15"] for r in sub)],
            "rho": spearman([(r["sealed_15"], r["extension_30"]) for r in sub]),
        })
    return out


def held_out_trim(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the rubric entries on waves 29-31; score wave 32 on the survivors.

    Trimming a rubric on the same rows it is then evaluated against is circular,
    and the circular number would be the flattering one.  So the selection is
    made on the THREE EARLIER WAVES ONLY -- drop every entry no child there ever
    held, and every entry whose held-indicator correlates NEGATIVELY with the
    sealed count there -- and the trimmed score is computed on wave 32, which the
    selection has never seen.  If wave 32's rho moves, the rubric is part of the
    scramble.  If it does not, the population is the whole of it.
    """
    fit = [r for r in rows if r["wave"] != "wave32"]
    test = [r for r in rows if r["wave"] == "wave32"]
    ids = sorted({pid for r in rows for pid in r["held"]})
    keep, dropped = [], []
    for pid in ids:
        pairs = [(float(r["sealed_15"]), 1.0 if r["held"].get(pid) else 0.0)
                 for r in fit if pid in r["held"]]
        held_n = sum(1 for _, h in pairs if h)
        rho = spearman(pairs)
        if held_n == 0 or held_n == len(pairs):
            dropped.append({"id": pid, "why": "no variance on waves 29-31", "rho": rho})
        elif rho is not None and rho < 0.0:
            dropped.append({"id": pid, "why": "negative rho on waves 29-31", "rho": rho})
        else:
            keep.append({"id": pid, "rho_fit": rho})
    kept_ids = {k["id"] for k in keep}
    trimmed = [(r["sealed_15"], float(sum(1 for pid in kept_ids if r["held"].get(pid))))
               for r in test]
    full = [(r["sealed_15"], float(r["extension_30"])) for r in test]
    return {
        "fit_waves": ["wave29", "wave30", "wave31"],
        "test_wave": "wave32",
        "rubric_entries_kept": len(keep),
        "rubric_entries_dropped": len(dropped),
        "dropped": dropped,
        "wave32_rho_full_rubric": spearman(full),
        "wave32_rho_trimmed_rubric": spearman(trimmed),
        "wave32_trimmed_scores": sorted(
            [{"label": r["label"], "sealed_15": r["sealed_15"],
              "trimmed": sum(1 for pid in kept_ids if r["held"].get(pid)),
              "full": r["extension_30"]} for r in test],
            key=lambda x: -x["sealed_15"]),
    }


def expected_rho(rows: list[dict[str, Any]], draws: int = 20000, seed: int = 1234) -> dict[str, Any]:
    """Is a wave's rho anomalous, or exactly what its own population predicts?

    Wave 31 and wave 32 have almost the same sealed spread -- mean pair gap 2.86
    against 2.79 -- and rho 0.7411 against 0.3537, so range restriction on its own
    does not close the gap and saying "the population changed" would be a story
    rather than a measurement.  This is the measurement.

    For the wave under test, keep its OWN sealed counts and replace each child's
    extension score with a draw from the empirical conditional distribution
    ``P(extension | sealed)`` estimated on the OTHER three waves, in a +/-1
    sealed-case window so the conditional is not estimated off two points.  That
    is a null in which the instrument behaves exactly as it behaves everywhere
    else and only the population is this wave's.  Twenty thousand draws give the
    distribution of rho such a wave WOULD produce; the observed rho is then a
    quantile in it rather than a number with no scale.
    """
    import random

    out = []
    for wave in WAVES:
        test = [r for r in rows if r["wave"] == wave]
        other = [r for r in rows if r["wave"] != wave]
        pools: dict[int, list[int]] = {}
        for r in test:
            s = r["sealed_15"]
            pool = [o["extension_30"] for o in other if abs(o["sealed_15"] - s) <= 1]
            pools[s] = pool
        if any(not p for p in pools.values()) or len(test) < 3:
            out.append({"wave": wave, "verdict": "not estimable: an empty conditional pool"})
            continue
        rng = random.Random(seed)
        observed = spearman([(r["sealed_15"], r["extension_30"]) for r in test])
        sims: list[float] = []
        for _ in range(draws):
            sim = [(r["sealed_15"], float(rng.choice(pools[r["sealed_15"]]))) for r in test]
            value = spearman(sim)
            if value is not None:
                sims.append(value)
        sims.sort()
        below = sum(1 for v in sims if v < (observed or 0.0))
        out.append({
            "wave": wave,
            "observed_rho": observed,
            "null_median_rho": round(sims[len(sims) // 2], 4),
            "null_p05_rho": round(sims[int(0.05 * len(sims))], 4),
            "null_p95_rho": round(sims[int(0.95 * len(sims))], 4),
            "observed_quantile": round(below / len(sims), 4),
            "anomalous_at_5pct": below / len(sims) < 0.05,
        })
    return {
        "draws": draws,
        "seed": seed,
        "per_wave": out,
        "note": (
            "The null holds the INSTRUMENT fixed at its behaviour on the other three "
            "waves and varies only the population.  A wave whose observed rho sits "
            "inside its own null needs no explanation beyond who it measured."
        ),
    }


def leave_one_out(rows: list[dict[str, Any]], wave: str) -> list[dict[str, Any]]:
    """Which single child is wave 32's rho standing on?"""
    sub = [r for r in rows if r["wave"] == wave]
    base = spearman([(r["sealed_15"], r["extension_30"]) for r in sub])
    out = []
    for drop in sub:
        rest = [(r["sealed_15"], r["extension_30"]) for r in sub if r is not drop]
        rho = spearman(rest)
        out.append({
            "dropped": drop["label"],
            "sealed_15": drop["sealed_15"],
            "extension_30": drop["extension_30"],
            "rho_without": rho,
            "delta": round((rho or 0.0) - (base or 0.0), 4),
        })
    return sorted(out, key=lambda r: -r["delta"])


def degeneracy_veto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The one-line rubric fix, priced on all sixty children.

    NOT a fitted change and not selected on this data: the fix is that the veto
    already written in `score_one` is applied to the charged set it already
    computes, instead of to two of its five members.  Whether that helps is then
    a measurement, and it is reported whichever way it comes out.
    """
    strict = [(r["sealed_15"], float(sum(1 for v in r["held_strict"].values() if v)))
              for r in rows]
    loose = [(r["sealed_15"], float(r["extension_30"])) for r in rows]
    moved = sorted(
        [{"label": r["label"], "wave": r["wave"], "sealed_15": r["sealed_15"],
          "loose": r["extension_30"],
          "strict": sum(1 for v in r["held_strict"].values() if v),
          "lost": r["extension_30"] - sum(1 for v in r["held_strict"].values() if v)}
         for r in rows],
        key=lambda x: -x["lost"])
    strict_rows = [dict(r, extension_30=sum(1 for v in r["held_strict"].values() if v))
                   for r in rows]
    return {
        "pooled_rho_loose": spearman(loose),
        "pooled_rho_strict": spearman(strict),
        "per_wave_rho_strict": {
            w: spearman([(r["sealed_15"], float(sum(1 for v in r["held_strict"].values() if v)))
                         for r in rows if r["wave"] == w])
            for w in WAVES
        },
        "gap_concordance_strict": gap_concordance(strict_rows)["table"],
        "biggest_losses": moved[:8],
        "zero_and_one_case_children": [
            m for m in moved if m["sealed_15"] <= 1
        ],
    }


def prompt_satisfiable(bank_path: Path, discrimination: dict[str, Any]) -> dict[str, Any]:
    """Which extension rubric entries does their OWN PROMPT satisfy?

    `ext-logic-four-seats` requires the four names in order and the prompt states
    its constraints in that order.  That is not a curiosity: an entry a child can
    hold by quoting the question back is an entry that measures echo, and echo is
    what a degenerate child produces most of.  The check runs the wave-32 rubric
    against the wave-29 bank with `mx39_pararubric.apply_rule`, which is the same
    matcher the paraphrase rubric's own self-test uses, so a hole found here and
    a hole refused there are found by one piece of code.
    """
    import mx39_extscore
    import mx39_pararubric

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    rho = {e["id"]: e["rho"] for e in discrimination["prompts"]}
    hits = []
    for row in bank["prompts"]:
        rule = mx39_extscore.RUBRIC.get(row["id"])
        if not rule:
            continue
        ask = "\n".join(m["content"] for m in row["messages"])
        if not mx39_pararubric.apply_rule(rule, ask):
            hits.append({"id": row["id"], "pooled_rho": rho.get(row["id"])})
    live = [h for h in hits if h["pooled_rho"] is not None]
    return {
        "entries_checked": len(mx39_extscore.RUBRIC),
        "prompt_satisfiable": len(hits),
        "entries": hits,
        "mean_rho_of_satisfiable": (
            round(sum(h["pooled_rho"] for h in live) / len(live), 4) if live else None
        ),
        "mean_rho_of_rest": (
            lambda rest: round(sum(rest) / len(rest), 4) if rest else None
        )([e["rho"] for e in discrimination["prompts"]
           if e["rho"] is not None and e["id"] not in {h["id"] for h in hits}]),
    }


def per_prompt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = sorted({pid for r in rows for pid in r["held"]})
    entries = []
    for pid in ids:
        pairs = [(float(r["sealed_15"]), 1.0 if r["held"].get(pid) else 0.0)
                 for r in rows if pid in r["held"]]
        held_n = sum(1 for _, h in pairs if h)
        entries.append({
            "id": pid,
            "family": family(pid),
            "held_of_n": [held_n, len(pairs)],
            "rho": spearman(pairs),
        })
    fam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by_fam: dict[str, int] = defaultdict(int)
        for pid, ok in r["held"].items():
            if ok:
                by_fam[family(pid)] += 1
        for f in {family(p) for p in r["held"]}:
            fam[f].append((float(r["sealed_15"]), float(by_fam.get(f, 0))))
    families = [{"family": f, "rho": spearman(v), "n": len(v)} for f, v in sorted(fam.items())]
    return {
        "prompts": sorted(entries, key=lambda e: (e["rho"] is None, e["rho"] or 0.0)),
        "families": sorted(families, key=lambda e: (e["rho"] is None, e["rho"] or 0.0)),
        "anti_instruments": [e["id"] for e in entries if (e["rho"] or 0.0) < 0.0],
        "dead_prompts": [e["id"] for e in entries
                         if e["held_of_n"][0] in (0, e["held_of_n"][1])],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True,
                    help="research/lfm2_extension_bank_prompts_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = collect(args.campaign)
    banks = sorted({r["bank_sha256"] for r in rows})
    pooled = [(r["sealed_15"], r["extension_30"]) for r in rows]
    # Deduplicated: a child decoded twice contributes one row, so the pooled rho
    # is not inflated by the reproduction control agreeing with itself.
    seen: set[str] = set()
    deduped_rows = []
    for r in rows:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        deduped_rows.append(r)
    deduped = [(r["sealed_15"], r["extension_30"]) for r in deduped_rows]

    band_table = bands(rows)
    body = {
        "schema": "neural-forge-matrix-wave39-extension-instrument-v1",
        "purpose": (
            "I1.  Characterise the ADDENDUM 16.1 extension surface over the four waves "
            "that ran it, rather than re-reporting a per-wave rank correlation whose "
            "sample changes every wave.  Reads only committed artifacts."
        ),
        "observations": len(rows),
        "distinct_children": len(deduped_rows),
        "banks_seen": banks,
        "single_bank": len(banks) == 1,
        "per_wave_rho": {
            w: spearman([(r["sealed_15"], r["extension_30"]) for r in rows if r["wave"] == w])
            for w in WAVES
        },
        "pooled_rho": spearman(pooled),
        "pooled_rho_deduplicated": spearman(deduped),
        "sealed_range_per_wave": {
            w: [min(r["sealed_15"] for r in rows if r["wave"] == w),
                max(r["sealed_15"] for r in rows if r["wave"] == w)]
            for w in WAVES
        },
        "reproduction_control": reproduction(rows),
        "bands": band_table,
        "gap_concordance": gap_concordance(rows),
        "pair_spread_per_wave": pair_spread(rows),
        "expected_rho_null": expected_rho(rows),
        "leave_one_out_wave32": leave_one_out(rows, "wave32"),
        "held_out_rubric_trim": held_out_trim(rows),
        "degeneracy_veto_fix": degeneracy_veto(rows),
        "discrimination": per_prompt(rows),
        "rows": [{k: v for k, v in r.items() if k != "held"} for r in rows],
    }
    body["prompt_satisfiable_entries"] = prompt_satisfiable(args.bank, body["discrimination"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")

    print(f"observations {len(rows)}  distinct children {len(deduped_rows)}  "
          f"banks {len(banks)}")
    print("per-wave rho:", body["per_wave_rho"])
    print("pooled rho:", body["pooled_rho"], " deduped:", body["pooled_rho_deduplicated"])
    print(body["reproduction_control"]["verdict"])
    print("\ncut  n>=cut  rho>=cut   n<cut  rho<cut")
    for b in band_table:
        print(f"{b['cut']:>3}  {b['n_at_or_above']:>6}  {str(b['rho_at_or_above']):>8}   "
              f"{b['n_below']:>5}  {str(b['rho_below']):>7}")
    print("\ngap  pairs  agree  tie  invert  agree%")
    for g in body["gap_concordance"]["table"]:
        print(f"{g['sealed_gap']:>3}  {g['pairs']:>5}  {g['agree']:>5}  {g['tie']:>3}  "
              f"{g['invert']:>6}  {g['agree_pct']:>5}")
    print("\nwave  children  pairs  mean gap  gap>=4%  gap<=2%  span      rho")
    for s in body["pair_spread_per_wave"]:
        print(f"{s['wave']:<6} {s['children']:>8}  {s['ordered_pairs']:>5}  "
              f"{s['mean_gap']:>8}  {s['pairs_gap_ge_4_pct']:>7}  "
              f"{s['pairs_gap_le_2_pct']:>7}  {str(s['sealed_span']):<9} {s['rho']}")

    live = [e for e in body["discrimination"]["prompts"] if e["rho"] is not None]
    dead = [e for e in body["discrimination"]["prompts"] if e["rho"] is None]
    print(f"\n{len(dead)} DEAD rubric entries (no child in 60 ever held them):")
    for e in dead:
        print(f"  {e['id']:<40} held {e['held_of_n'][0]}/{e['held_of_n'][1]}")
    print("\nlive rubric entries, rho ascending:")
    for e in live:
        flag = "  <-- ANTI" if (e["rho"] or 0) < 0 else ""
        print(f"  {e['id']:<40} rho {e['rho']:>8}  held {e['held_of_n'][0]:>2}/{e['held_of_n'][1]}{flag}")

    print("\nwave    observed  null p05  null med  null p95  quantile  anomalous")
    for e in body["expected_rho_null"]["per_wave"]:
        if "observed_rho" not in e:
            print(f"{e['wave']:<7} {e['verdict']}"); continue
        print(f"{e['wave']:<7} {str(e['observed_rho']):>8}  {str(e['null_p05_rho']):>8}  "
              f"{str(e['null_median_rho']):>8}  {str(e['null_p95_rho']):>8}  "
              f"{e['observed_quantile']:>8}  {e['anomalous_at_5pct']}")

    print("\nwave 32 leave-one-out (rho without that child):")
    for e in body["leave_one_out_wave32"][:4] + body["leave_one_out_wave32"][-3:]:
        print(f"  drop {e['dropped']:<14} sealed {e['sealed_15']:>2} ext {e['extension_30']:>2}  "
              f"rho -> {e['rho_without']}  delta {e['delta']:+.4f}")

    print("\nper-family rho:")
    for e in body["discrimination"]["families"]:
        print(f"  {e['family']:<16} rho {str(e['rho']):>8}")

    trim = body["held_out_rubric_trim"]
    print(f"\nheld-out trim: kept {trim['rubric_entries_kept']} of 30 entries "
          f"(selected on waves 29-31 only)")
    print(f"  wave 32 rho, full rubric    : {trim['wave32_rho_full_rubric']}")
    print(f"  wave 32 rho, trimmed rubric : {trim['wave32_rho_trimmed_rubric']}")

    ps = body["prompt_satisfiable_entries"]
    print(f"\nrubric entries their OWN PROMPT satisfies: {ps['prompt_satisfiable']} of "
          f"{ps['entries_checked']}")
    for e in ps["entries"]:
        print(f"    {e['id']:<40} pooled rho {e['pooled_rho']}")
    print(f"  mean pooled rho, prompt-satisfiable {ps['mean_rho_of_satisfiable']}  "
          f"vs the rest {ps['mean_rho_of_rest']}")

    dv = body["degeneracy_veto_fix"]
    print("\ndegeneracy-veto fix (charge any of cap/empty/echo/loop/word-sub -> no hold):")
    print(f"  pooled rho  loose {dv['pooled_rho_loose']}  ->  strict {dv['pooled_rho_strict']}")
    print(f"  per wave strict: {dv['per_wave_rho_strict']}")
    print("  biggest score losses:")
    for m in dv["biggest_losses"]:
        print(f"    {m['label']:<14} sealed {m['sealed_15']:>2}  {m['loose']:>2} -> {m['strict']:>2}  (-{m['lost']})")
    print("  every child sealed <= 1:")
    for m in dv["zero_and_one_case_children"]:
        print(f"    {m['label']:<14} sealed {m['sealed_15']:>2}  {m['loose']:>2} -> {m['strict']:>2}")
    print("  gap-concordance, strict:")
    for g in dv["gap_concordance_strict"][:6]:
        print(f"    gap {g['sealed_gap']}  agree {g['agree_pct']}%  invert {g['invert_pct']}%  ties {g['tie']}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
