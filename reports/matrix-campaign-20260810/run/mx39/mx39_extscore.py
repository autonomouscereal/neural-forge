#!/usr/bin/env python3
"""The extension surface's MACHINE score, and the retro-score comparison.

`INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` #1 specifies the retro-score as
"pure inference, machine-scored", and it has to be: a human per-lane reading of
eight checkpoints times thirty prompts is two hundred and forty readings, and a
reading budget that large is exactly the budget the extension bank exists to
economise.  So this file carries a **pre-registered rubric**, one entry per
extension prompt, written in the same commit as the bank and therefore before
any wave-29 token was generated -- the deployment manifest digests both, so the
ordering is provable rather than asserted.

**This score is not a verdict and cannot become one.**  ADDENDUM 16.1 sentence 2
stands: the sealed fifteen, read by hand, remain the only judge of any frontier
claim.  What this file produces is a second coordinate for ordering children and
for pricing one-case margins.

**What a machine can honestly check.**  Each rubric entry is a conjunction of
cheap surface facts that a correct answer must exhibit and an incorrect one
usually does not: required literals or regexes, forbidden literals, an optional
minimum length, and for the coding lanes an AST check rather than execution
(model-generated code is not run).  This is deliberately a LOW-RESOLUTION
instrument: it under-credits good answers phrased unusually and it can be fooled
by a lucky string.  That is acceptable for *ordering* eight children and
unacceptable for grading one, which is precisely the split ADDENDUM 16.1 draws.

**The degeneracy readings are the sealed ones**, computed rather than read: cap
hit, empty string, prompt echo, loop, and the word self-substitution signature
wave 23 charged as its own failure class.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED RUBRIC.  Written with the bank, before any decode.
#   req      : every regex must match (case-insensitive, DOTALL)
#   forbid   : no regex may match
#   min_words: a floor, because a two-word reply is not an answer
#   ast_fn   : the code must parse and define this function name
# Every pattern is matched with re.I | re.S already applied, so no pattern
# carries an inline (?s): Python refuses a global flag that is not at the
# start of an expression, and an alternation with one on each side of the |
# is exactly that refusal.  The flags live in one place, at the match site.
# ---------------------------------------------------------------------------
RUBRIC: dict[str, dict[str, Any]] = {
    "ext-conversation-moving-week": {"req": [r"\?"], "forbid": [r"\?[^?]*\?"], "min_words": 12},
    "ext-conversation-competing-promises": {"req": [], "forbid": [r"(?m)^\s*(\d[\.\)]|[-*])\s"], "min_words": 15},
    "ext-instruction-two-prose-tips": {"req": [], "forbid": [r"(?m)^\s*(\d[\.\)]|[-*])\s", r"\bwater"], "min_words": 15},
    "ext-instruction-five-alphabetical": {"req": [r"1[\.\)].*2[\.\)].*3[\.\)].*4[\.\)].*5[\.\)]"], "forbid": [r"6[\.\)]"], "min_words": 10},
    "ext-explanation-dough-rises": {"req": [r"yeast|ferment", r"carbon dioxide|co2|gas"], "forbid": [], "min_words": 30},
    "ext-explanation-two-tides": {"req": [r"two|both|opposite|far side|other side", r"gravit|pull|tidal force"], "forbid": [], "min_words": 30},
    "ext-arithmetic-fuel-range": {"req": [r"\b486\b", r"\b242\b"], "forbid": [], "min_words": 10},
    "ext-arithmetic-order-of-operations": {"req": [r"\b204\b", r"220\.32|220,32|\b220\b"], "forbid": [], "min_words": 10},
    "ext-logic-four-seats": {"req": [r"ada.*bo.*cy.*dee", r"dee.*ada.*bo.*cy"], "forbid": [], "min_words": 15},
    "ext-logic-three-switches": {"req": [r"heat|warm|hot|temperature"], "forbid": [], "min_words": 30},
    "ext-context-clinic-note": {"req": [r"11[:.]00|eleven|11 a", r"stick|sticker|validat"], "forbid": [], "min_words": 15},
    "ext-context-conflicting-note": {"req": [r"contradict|conflict|inconsist|two different|both.*and|unclear|does not settle"], "forbid": [], "min_words": 15},
    "ext-summarization-three-bullets": {"req": [r"(night|contractor).*(reader|badge).*(november|monday)"], "forbid": [r"cafeteria refit, which everyone keeps asking about"], "min_words": 15},
    "ext-summarization-loudest-is-irrelevant": {"req": [r"tuesday", r"audit|inspect"], "forbid": [], "min_words": 8},
    "ext-comparison-map-and-traffic": {"req": [r"(analogy|comparison|like|as if).*(break|fall|limit|imperfect|wrong|fail)"], "forbid": [], "min_words": 40},
    "ext-comparison-vaccine-antibiotic": {"req": [r"bacteri", r"immun|antibod|vaccin"], "forbid": [], "min_words": 40},
    "ext-coding-group-by-initial": {"req": [r"def\s+group_by_initial", r"lower\(\)", r"append|setdefault|defaultdict"], "forbid": [r"set\(\)"], "min_words": 15, "ast_fn": "group_by_initial"},
    "ext-coding-merge-ranges": {"req": [r"def\s+merge_ranges", r"sort"], "forbid": [], "min_words": 15, "ast_fn": "merge_ranges"},
    "ext-debugging-mutable-default": {"req": [r"none", r"default.*(shared|mutable|same|once|persist)"], "forbid": [], "min_words": 20},
    "ext-debugging-reset-inside-loop": {"req": [r"(total\s*=\s*0).*(before|outside|above|move)|(before|outside|above|move).*(total\s*=\s*0)"], "forbid": [], "min_words": 20},
    "ext-writing-decline-invitation": {"req": [r"april", r"\?"], "forbid": [], "min_words": 25},
    "ext-writing-two-audiences": {"req": [r"four hours.*four hours|4 hours.*4 hours|(engineer|technical).*(customer|client)"], "forbid": [], "min_words": 20},
    "ext-planning-first-bookkeeper": {"req": [r"1[\.\)].*2[\.\)].*3[\.\)].*4[\.\)].*5[\.\)]", r"undo|revers|cheapest|back out|unwind"], "forbid": [], "min_words": 30},
    "ext-planning-around-a-freeze": {"req": [r"1[\.\)].*2[\.\)].*3[\.\)].*4[\.\)].*5[\.\)].*6[\.\)]", r"freeze"], "forbid": [], "min_words": 40},
    "ext-uncertainty-two-hospitals": {"req": [r"case mix|casemix|severity|referral|patient population|sicker|complexity"], "forbid": [], "min_words": 30},
    "ext-uncertainty-confident-source": {"req": [r"nine|9 (people|participants)|small (sample|study|trial)|sample size"], "forbid": [], "min_words": 30},
    "ext-creative-returned-glove": {"req": [r"[\"“].+[\"”]|'[^']{6,}'"], "forbid": [], "min_words": 60},
    "ext-creative-no-adjectives": {"req": [], "forbid": [], "min_words": 30},
    "ext-multiturn-budget-revision": {"req": [r"1[\.\)].*2[\.\)].*3[\.\)]|first.*second.*third"], "forbid": [], "min_words": 25},
    "ext-multiturn-contradiction-flag": {"req": [r"one page|single page|fit|space|too much|cannot|can't|not possible|trade|prioriti|which"], "forbid": [], "min_words": 20},
}

WORD_SUBSTITUTION = re.compile(r"\b(\w{4,})\b[^.]{0,30}\b\1[- ]?like\b[^.]{0,30}\b\1\b", re.I)


def ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\S+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def degeneracy(record: dict[str, Any]) -> list[str]:
    """The sealed reading standard's charged set, computed."""
    text = record.get("response_verbatim", "") or ""
    charged: list[str] = []
    if record.get("termination") == "censored-at-cap":
        charged.append("cap")
    if not text.strip():
        charged.append("empty")
    prompt = "\n".join(m["content"] for m in record.get("prompt_messages", []))
    resp_grams = ngram_set(text, 8)
    if resp_grams:
        shared = resp_grams & ngram_set(prompt, 8)
        if len(shared) / len(resp_grams) >= 0.40:
            charged.append("echo")
    words = re.findall(r"\S+", text)
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    if grams and 1.0 - len(set(grams)) / len(grams) >= 0.50:
        charged.append("loop")
    if WORD_SUBSTITUTION.search(text):
        charged.append("word-substitution")
    return charged


def score_one(record: dict[str, Any]) -> dict[str, Any]:
    rule = RUBRIC.get(record["id"])
    text = record.get("response_verbatim", "") or ""
    charged = degeneracy(record)
    if rule is None:
        return {"id": record["id"], "held": False, "charged": charged, "reason": "no rubric"}
    reasons: list[str] = []
    for pattern in rule.get("req", []):
        if not re.search(pattern, text, re.I | re.S):
            reasons.append(f"missing {pattern}")
    for pattern in rule.get("forbid", []):
        if re.search(pattern, text, re.I | re.S):
            reasons.append(f"forbidden {pattern}")
    words = len(re.findall(r"\S+", text))
    if words < rule.get("min_words", 0):
        reasons.append(f"short ({words} words)")
    fn = rule.get("ast_fn")
    if fn:
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S) or [text]
        ok = False
        for block in blocks:
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            if any(isinstance(node, ast.FunctionDef) and node.name == fn
                   for node in ast.walk(tree)):
                ok = True
                break
        if not ok:
            reasons.append(f"no parsable def {fn}")
    # A charged lane can still hold a case -- the sealed standard says the two
    # readings are independent -- EXCEPT for empty and echo, where there is no
    # answer of the child's own to credit.
    if "empty" in charged or "echo" in charged:
        reasons.append("charged: " + ",".join(charged))
    return {
        "id": record["id"],
        "held": not reasons,
        "charged": charged,
        "reason": "; ".join(reasons) if reasons else "held",
        "words": words,
    }


def score_dossier(path: Path, which: str = "child_responses") -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    records = doc.get(which) or []
    per = [score_one(r) for r in records]
    return {
        "label": doc.get("label"),
        "cell_id": doc.get("cell_id"),
        "surface": doc.get("surface"),
        "bank_sha256": doc.get("bank_sha256"),
        "prompts": len(per),
        "held": sum(1 for p in per if p["held"]),
        "charged_lanes": sum(1 for p in per if p["charged"]),
        "charged_kinds": sorted({k for p in per for k in p["charged"]}),
        "per_prompt": per,
    }


def spearman(pairs: list[tuple[float, float]]) -> float | None:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", type=Path, action="append", default=[])
    ap.add_argument("--sealed", type=Path, help="JSON: label -> sealed 15-case count")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scored = [score_dossier(p) for p in args.dossier]
    body: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave39-extension-score-v1",
        "purpose": (
            "Machine score on the ADDENDUM 16.1 extension surface, plus the retro-score "
            "comparison against sealed 15-case counts.  Orders and prices; never grades."
        ),
        "rubric_prompts": len(RUBRIC),
        "children": scored,
    }

    if args.sealed is not None:
        sealed = json.loads(args.sealed.read_text(encoding="utf-8"))
        pairs: list[tuple[float, float]] = []
        table = []
        for row in scored:
            label = row["label"]
            if label in sealed:
                pairs.append((float(sealed[label]), float(row["held"])))
                table.append({
                    "label": label,
                    "sealed_15": sealed[label],
                    "extension_30": row["held"],
                    "extension_pct": round(100.0 * row["held"] / max(row["prompts"], 1), 1),
                    "charged_lanes": row["charged_lanes"],
                })
        table.sort(key=lambda r: -r["sealed_15"])
        rho = spearman(pairs)
        sealed_vals = sorted({r["sealed_15"] for r in table})
        ext_vals = [r["extension_30"] for r in table]
        body["retro_score"] = {
            "table": table,
            "spearman_sealed_vs_extension": rho,
            "sealed_distinct_levels": len(sealed_vals),
            "extension_distinct_levels": len(set(ext_vals)),
            "extension_range": [min(ext_vals), max(ext_vals)] if ext_vals else None,
            "verdict": (
                "VALIDATED: the sealed rank order reproduces on the extension surface"
                if rho is not None and rho >= 0.70 else
                "PARTIAL: rank order reproduces weakly -- read the table before leaning on any one-case margin"
                if rho is not None and rho >= 0.40 else
                "SCRAMBLED: the sealed ordering does NOT reproduce.  Under the plan's "
                "pre-registered go/no-go every one-case play pauses and this goes to Cyril."
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    for row in scored:
        print(f"{row['label']:<22} extension {row['held']:>2}/{row['prompts']}  charged {row['charged_lanes']:>2}  {','.join(row['charged_kinds'])}")
    if "retro_score" in body:
        print(json.dumps(body["retro_score"], indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
