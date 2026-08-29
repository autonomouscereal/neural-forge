#!/usr/bin/env python3
"""The paraphrase surface's machine score: a LANE-keyed rubric, with the holes shut.

Wave 32 decoded 45 paraphrase prompts against five children and could not score
them: `mx32_extscore.py`'s `RUBRIC` is keyed on the extension bank's thirty
prompt ids, so it returned `no rubric` and `held: false` for all 45 and the
finding in that wave's section 5 is a hand reading.  This file is the rubric it
needed.  It is keyed by LANE -- `para-<lane>-1/2/3` all resolve to `<lane>` --
because the fifteen paraphrase lanes are one-to-one with the sealed fifteen
cases, so one rule per lane is one rule per case and the surface can be read
against the reading standard rather than against a second, unrelated bank.

**Three defects in the extension rubric are fixed here by construction, and each
one was measured on that rubric before it was fixed** (`mx40_instrument.py`).

1. **THE DEGENERACY VETO IS CLOSED.**  `score_one` computes five charged kinds
   -- cap, empty, echo, loop, word-substitution -- and then vetoes a hold on
   `empty` or `echo` only.  So `ARCH.BM50N`, the 0/15 child with sixteen of
   sixteen lanes charged, HELD THREE extension entries, and the three held texts
   are a 1,024-token repetition of the word "not", a 1,024-token repetition of
   an apostrophe, and 75 repetitions of "to" followed by a run of ">".  Here a
   charged lane cannot hold anything, ever.

2. **NO REQUIRED PATTERN MAY BE SATISFIED BY THE PROMPT ITSELF.**  The extension
   rubric's `ext-logic-four-seats` requires `ada.*bo.*cy.*dee`, and the prompt
   states its own constraints in exactly that order -- so a child that quotes the
   question back satisfies the answer test.  `ARCH.V12CL1792` holds that entry on
   a response that never produces a seating order.  `--self-test` runs every lane
   rule against that lane's own three prompts and REFUSES if any prompt passes
   its own rule.  It is the first thing this file does and there is no flag to
   skip it.

3. **CONTENT, NOT FORM.**  Six of the thirty extension entries carry an empty
   `req` list and grade on formatting and length alone; the worst of them,
   `ext-conversation-competing-promises`, is held by 39 of 60 children and
   correlates NEGATIVELY with the sealed count.  Every rule below names at least
   one thing the answer has to contain.

**Two lanes are declared UNANSWERABLE and are not scored.**  The sealed
`context-retrieval` prompt carries a NOTE inline (garden hours, tool checkout at
4:15 p.m., dogs on Saturdays, north gate closed) and the sealed
`summarization-priorities` prompt carries the four-supplier UPDATE inline.  All
six of their paraphrase restatements say "the note below" / "the update" and
carry NO note and NO update.  Nothing can answer them, no child ever could have,
and scoring them zero would charge six lanes of the surface to the children
rather than to the bank.  They are reported as a bank defect with the count.

**The verdict this file produces is a ROBUSTNESS reading and never a grade.**
Per lane: held on 3 of 3 restatements is ROBUST, 2 of 3 MOSTLY, 1 of 3 FRAGILE,
0 of 3 ABSENT.  A sealed case whose lane reads ABSENT under three restatements
of its own ask is a case that does not survive rewording -- which is the only
question this surface was built to answer.

Usage::

    mx40_pararubric.py --self-test --bank <lfm2_paraphrase_triplets_v1.json>
    mx40_pararubric.py --bank <bank> --dossier <extension-paraphrase-X.json> ... \\
        --sealed <label -> 15-count json> --out <json>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

#: A lane is UNANSWERABLE when the restatements dropped the source material the
#: sealed prompt supplies inline.  Declared, counted, and never scored.
UNANSWERABLE: dict[str, str] = {
    "context-retrieval": (
        "all three restatements say 'the note below' / 'from the note alone' / "
        "'strictly from the note' and carry NO note.  The sealed prompt supplies it "
        "inline -- garden hours, tool checkout 4:15 p.m., dogs on Saturdays, north "
        "gate closed -- and the paraphrase bank dropped it"
    ),
    "summarization-priorities": (
        "all three restatements say 'the update below' / 'the update' and carry NO "
        "update.  The sealed prompt supplies the four-supplier comparison inline and "
        "the paraphrase bank dropped it"
    ),
}

#: One rule per lane.  ``req`` is a conjunction of regexes (``re.I | re.S``),
#: ``forbid`` a disjunction that fails the lane, ``order`` a list of literals that
#: must appear in that ORDER by position, ``min_words`` a floor, ``max_words`` a
#: ceiling where the ask states one, ``ast_fn`` a parsable definition, and
#: ``distinct_items`` the number of numbered items that must not be near-copies
#: of one another (the sealed `instruction-three-actions` calibration: three items
#: that are two items repeated is a miss).
RUBRIC: dict[str, dict[str, Any]] = {
    "conversation-natural-greeting": {
        # Sealed ask: respond like a person, EXACTLY one question.
        "req": [r"\?", r"\b(tired|worn|exhaust|rough|rest|sleep|long day|drain|hard day|tough)\b"],
        "forbid": [r"\?[^?]*\?"],
        "min_words": 12,
    },
    "instruction-three-actions": {
        "req": [],
        "order": ["1", "2", "3"],
        "forbid": [r"(?m)^\s*4[\.\)]\s"],
        "min_words": 12,
        "distinct_items": 3,
    },
    "explanation-seasons": {
        # The ask has two halves and the second one is the whole point: correct
        # the distance misconception.  A tilt answer that never touches distance
        # answered half the question.
        "req": [r"\b(tilt|tilted|axis|axial|inclin)", r"\b(distance|closer|nearer|farther|further)\b"],
        "forbid": [],
        "min_words": 30,
    },
    "arithmetic-multi-step": {
        # 17*6 = 102, less 8 = 94, 94/5 = 18 remainder 4.  The sealed calibration
        # makes the DIVISION the discriminator: a child that never forms 17*6 is a
        # miss, one that reaches the division and botches the remainder is Partial,
        # and Useful needs 18 and 4.
        "req": [r"\b94\b", r"\b18\b", r"\b4\b"],
        "forbid": [],
        "min_words": 10,
    },
    "logic-ordering": {
        # Mira, Theo, Jun, Sal.  The ORDER of the four names is not enough on its
        # own -- every prompt states its constraints in that order, so a response
        # that RESTATES THE CONSTRAINTS passes a positional check.  The parent
        # does exactly that on two of the three restatements and is sealed a MISS
        # on this lane, which is how the hole was found.  The lane asks for the
        # full finishing order, so the discriminator is the END of it: Sal has to
        # be named last.  No prompt says Sal is last or fourth.
        "req": [
            r"\b(first|1st)\b",
            r"\bsal\b[^.]{0,40}\b(last|fourth|4th|bottom|behind (all|everyone|the others))\b"
            r"|\b(last|fourth|4th)\b[^.]{0,40}\bsal\b",
        ],
        "order": ["mira", "theo", "jun", "sal"],
        "forbid": [],
        "min_words": 15,
    },
    "comparison-ram-rom": {
        # One analogy plus two ways it misleads.  The analogy needs a VEHICLE --
        # a concrete object outside the domain -- because "analogy", "comparison"
        # and "misleads" are all words the ask itself supplies.
        "req": [
            r"\b(desk|workbench|bench|table|counter|whiteboard|notebook|filing|cabinet|"
            r"shelf|shelves|library|kitchen|fridge|warehouse|drawer|scratch\s*pad|"
            r"blackboard|chalkboard|workspace|worktop)\b",
            r"\b(break|breaks|fail|fails|limit|misle|imperfect|wrong|inaccurat|unlike)",
            r"\b(ram|working memory|short.term)\b",
            r"\b(disk|drive|storage|long.term|rom|hard\s*disk|ssd)\b",
        ],
        "forbid": [],
        "min_words": 40,
    },
    "coding-deduplicate": {
        # The sealed calibration: a SET is the one thing the lane forbids, because
        # first-seen order is the one thing the lane asks for.
        "req": [r"def\s+unique_in_order", r"\b(append|setdefault|fromkeys|not in|seen)\b"],
        "forbid": [r"return\s+set\(", r"=\s*set\(\)[\s\S]{0,200}return\s+\w+\s*$"],
        "min_words": 15,
        "ast_fn": "unique_in_order",
    },
    "debugging-off-by-one": {
        # The fault is `start - 1`; the fix is `values[start:]`.  The faulty slice
        # is in the prompt, so only the CORRECTED slice can be required.
        "req": [
            r"(values\[start:\]|\[-\s*3\s*:\]|values\[len\(values\)\s*-\s*3\s*:\])",
            r"\b(off.by.one|start\s*-\s*1|start-1|minus one|subtract|extra|one too many|fourth)\b",
        ],
        "forbid": [],
        "min_words": 20,
    },
    "writing-concise-email": {
        # Three named items and the ask, in an email.  All three items are in the
        # prompt, so the EMAIL FORM is the discriminator, and the sealed
        # calibration adds one more: requesting the tracking number is Useful,
        # offering to supply it has inverted the ask.
        "req": [
            r"\bfriday\b", r"\bmonday\b", r"\btrack",
            r"\b(dear|hello|hi\b|good morning|good afternoon|regards|sincerely|"
            r"best wishes|many thanks|thank you|thanks)\b",
        ],
        "forbid": [r"\b(we will (send|provide|supply)|I will (send|provide|supply)) "
                   r"(you )?(a |the |an )?(new |updated )?tracking"],
        "min_words": 25,
        "max_words": 120,
    },
    "planning-dependencies": {
        # The sealed calibration keys on the ITEMS: five of five in six steps is
        # Useful, four is Partial, three or fewer is a miss.  All five item words
        # are in the prompt, so the SIX NUMBERED STEPS are the discriminator.
        "req": [
            r"\b(backup|back up|snapshot|dump)\b",
            r"\bdns\b",
            r"\b(certificat|tls|ssl|https)\b",
            r"\b(verif|check|test|validat|smoke)\b",
            r"\b(roll ?back|back out|revert|undo|restore)\b",
        ],
        "order": ["1", "2", "3", "4", "5", "6"],
        "forbid": [],
        "min_words": 40,
    },
    "uncertainty-insufficient-data": {
        # Name what is missing and say what is still safe.  The ask lists the
        # missing items itself, so the rule needs the HEDGE and the length the
        # prompts do not have.
        "req": [
            r"\b(period|time ?frame|time range|date range|window|units|sample|"
            r"sample size|price|pricing|baseline|seasonal)\b",
            r"\b(cannot|can't|not enough|insufficient|no basis|unsafe|only|"
            r"without knowing|unclear|do not know|don't know)\b",
            r"\?|\b(would (need|ask|want|request)|ask for|request)\b",
        ],
        "forbid": [],
        "min_words": 45,
    },
    "creative-continuity": {
        # Wave 32's calibration, applied as written: THREE motifs held is Partial,
        # TWO is a miss.  The three are the keeper, the foghorn, and the answering.
        # The prompts carry all three, so the SINGLE LINE OF DIALOGUE -- which no
        # prompt contains -- is the discriminator, and the ask names it.
        "req": [
            r"\b(keeper|lighthouse)\b",
            r"\b(foghorn|fog horn|\bhorn\b)",
            r"\b(answer|answering|answers|repl(y|ies|ying)|respond|responds|responding)\b",
            r"[\"“][^\"”]{4,}[\"”]|'[^']{6,}'",
        ],
        "forbid": [],
        "min_words": 60,
    },
    "multi-turn-correction": {
        # Four revised options, peanut-free, at least one vegan.  The revision is
        # the whole lane: the user has disclosed TWO SEVERE PEANUT ALLERGIES, so
        # an option list that still contains peanut butter has not revised
        # anything.  The parent offers "Peanut Butter & Veggie Pudding" as option
        # one and is sealed a MISS on this lane; without `forbid_in_items` the
        # rule credited it, which is how the hole was found.
        "req": [
            r"\b(vegan|plant.based|dairy.free)\b",
            r"\b(no peanut|peanut.free|without peanut|avoid(ing)? peanut|free of peanut|"
            r"nut.free|instead of peanut)\b",
        ],
        "order": ["1", "2", "3", "4"],
        "forbid": [],
        "forbid_in_items": [r"peanut butter", r"\bpeanuts?\b(?!.{0,12}(free|allerg))"],
        "min_words": 20,
    },
}

WORD_SUBSTITUTION = re.compile(r"\b(\w{4,})\b[^.]{0,30}\b\1[- ]?like\b[^.]{0,30}\b\1\b", re.I)
NUMBERED = re.compile(r"(?m)^\s*(\d)[\.\)]\s*(.+)$")


def lane_of(prompt_id: str) -> str:
    return re.sub(r"-\d+$", "", prompt_id).removeprefix("para-")


def ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\S+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def degeneracy(record: dict[str, Any]) -> list[str]:
    """The sealed reading standard's charged set, computed.  Wave 29's, unchanged."""
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


def _distinct_items(text: str, needed: int) -> bool:
    """Are the numbered items different from one another?

    The sealed `instruction-three-actions` calibration says three items that are
    two items repeated is a miss, and every wave since 30 has hit children whose
    numbered list is one item three times.  Two items are 'the same' when their
    word sets overlap at 70 per cent or more, which is loose on purpose: this is
    meant to catch 'Prep and prepare' three times, not to referee near-synonyms.
    """
    items = [m.group(2).strip().lower() for m in NUMBERED.finditer(text)]
    if len(items) < needed:
        return False
    picked = items[:needed]
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            a = set(re.findall(r"[a-z]{3,}", picked[i]))
            b = set(re.findall(r"[a-z]{3,}", picked[j]))
            if not a or not b:
                continue
            if len(a & b) / min(len(a), len(b)) >= 0.70:
                return False
    return True


def apply_rule(rule: dict[str, Any], text: str) -> list[str]:
    """Every reason this text fails this rule.  Empty means it holds."""
    reasons: list[str] = []
    for pattern in rule.get("req", []):
        if not re.search(pattern, text, re.I | re.S):
            reasons.append(f"missing {pattern[:44]}")
    for pattern in rule.get("forbid", []):
        if re.search(pattern, text, re.I | re.S):
            reasons.append(f"forbidden {pattern[:44]}")
    # Forbidden INSIDE THE ANSWER'S OWN ITEMS.  A response may legitimately name
    # the thing it is removing while explaining the revision; what it may not do
    # is put it in one of the four options.  The item region starts at the first
    # numbered marker, so a preamble that says "no peanuts" is not charged and an
    # option that says "peanut butter base" is.
    forbid_items = rule.get("forbid_in_items") or []
    if forbid_items:
        first = NUMBERED.search(text)
        region = text[first.start():] if first else ""
        for pattern in forbid_items:
            if region and re.search(pattern, region, re.I | re.S):
                reasons.append(f"forbidden inside the items: {pattern[:40]}")
    order = rule.get("order")
    if order:
        cursor, ok = 0, True
        low = text.lower()
        for literal in order:
            needle = literal.lower()
            found = -1
            if literal.isdigit():
                m = re.compile(rf"(?m)^\s*{literal}[\.\)]\s").search(low, cursor)
                found = m.start() if m else -1
            else:
                found = low.find(needle, cursor)
            if found < 0:
                ok = False
                break
            cursor = found + len(needle)
        if not ok:
            reasons.append("order " + ">".join(order) + " not present")
    words = len(re.findall(r"\S+", text))
    if words < rule.get("min_words", 0):
        reasons.append(f"short ({words} words)")
    if "max_words" in rule and words > rule["max_words"]:
        reasons.append(f"long ({words} words)")
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
    need = rule.get("distinct_items")
    if need and not _distinct_items(text, need):
        reasons.append(f"fewer than {need} DISTINCT numbered items")
    return reasons


def score_one(record: dict[str, Any]) -> dict[str, Any]:
    lane = lane_of(record["id"])
    text = record.get("response_verbatim", "") or ""
    charged = degeneracy(record)
    if lane in UNANSWERABLE:
        return {"id": record["id"], "lane": lane, "held": False, "scored": False,
                "charged": charged, "reason": "UNANSWERABLE LANE: " + UNANSWERABLE[lane]}
    rule = RUBRIC.get(lane)
    if rule is None:
        return {"id": record["id"], "lane": lane, "held": False, "scored": False,
                "charged": charged, "reason": "no rubric for this lane"}
    reasons = apply_rule(rule, text)
    # THE VETO IS CLOSED.  Any charged kind ends the lane.  The extension rubric
    # vetoed on `empty` and `echo` only, and a 1,024-token repetition of the word
    # "not" therefore held three of its entries on the campaign's 0/15 child.
    if charged:
        reasons.append("charged: " + ",".join(charged))
    return {
        "id": record["id"], "lane": lane, "scored": True,
        "held": not reasons, "charged": charged,
        "words": len(re.findall(r"\S+", text)),
        "reason": "; ".join(reasons) if reasons else "held",
    }


def self_test(bank_path: Path) -> int:
    """Refuse if any lane's own prompts satisfy its own rule.

    This is the guard the extension rubric did not have.  `ext-logic-four-seats`
    requires the four names in order and the prompt states its constraints in
    that order, so quoting the question back passes the answer test -- and one
    frontier child holds that entry on a response that never produces a seating
    order.  A required pattern that the ASK already satisfies is not a
    requirement, it is a formality.
    """
    doc = json.loads(bank_path.read_text(encoding="utf-8"))
    triplets = doc["triplets"]
    problems: list[str] = []
    for lane, rows in triplets.items():
        if lane in UNANSWERABLE:
            print(f"  UNANSWERABLE (declared, not scored): {lane}")
            continue
        rule = RUBRIC.get(lane)
        if rule is None:
            problems.append(f"{lane}: the bank carries this lane and the rubric does not")
            continue
        for row in rows:
            ask = "\n".join(m["content"] for m in row["messages"])
            reasons = apply_rule(rule, ask)
            if not reasons:
                problems.append(
                    f"{row['id']}: THE PROMPT SATISFIES ITS OWN RULE -- a child that "
                    f"quotes the question back would hold this lane")
    for lane in RUBRIC:
        if lane not in triplets:
            problems.append(f"{lane}: the rubric carries this lane and the bank does not")
    covered = set(RUBRIC) | set(UNANSWERABLE)
    missing = sorted(set(triplets) - covered)
    if missing:
        problems.append(f"lanes with neither a rule nor a declared refusal: {missing}")
    if problems:
        for p in problems:
            print("SELF-TEST FAILED:", p, file=sys.stderr)
        return 1
    print(f"self-test: {len(RUBRIC)} lane rules, {len(UNANSWERABLE)} lanes declared "
          f"unanswerable, {len(triplets)} lanes in the bank; no prompt satisfies its "
          f"own rule; the degeneracy veto is closed")
    return 0


def score_dossier(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    # The parent's dossier carries `parent_responses` and an empty
    # `child_responses`; a scorer that reads only the second key reports the
    # PARENT at 0/45 and silently loses the one row that says whether the rubric
    # is calibrated or merely strict.
    records = doc.get("child_responses") or doc.get("parent_responses") or []
    per = [score_one(r) for r in records]
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in per:
        by_lane.setdefault(row["lane"], []).append(row)
    lanes = []
    for lane in sorted(by_lane):
        rows = by_lane[lane]
        if not rows[0]["scored"]:
            lanes.append({"lane": lane, "verdict": "UNSCORED", "held": 0,
                          "of": len(rows), "reason": rows[0]["reason"]})
            continue
        held = sum(1 for r in rows if r["held"])
        lanes.append({
            "lane": lane, "held": held, "of": len(rows),
            "charged": sum(1 for r in rows if r["charged"]),
            "verdict": ("ROBUST" if held == len(rows) else
                        "MOSTLY" if held == len(rows) - 1 else
                        "FRAGILE" if held > 0 else "ABSENT"),
            "reasons": [r["reason"][:120] for r in rows],
        })
    scored = [x for x in lanes if x["verdict"] != "UNSCORED"]
    return {
        "label": doc.get("label"),
        "cell_id": doc.get("cell_id"),
        "surface": doc.get("surface"),
        "bank_sha256": doc.get("bank_sha256"),
        "prompts": len(per),
        "prompts_scored": sum(x["of"] for x in scored),
        "prompts_unscorable": sum(x["of"] for x in lanes if x["verdict"] == "UNSCORED"),
        "lanes_scored": len(scored),
        "lanes_robust": sum(1 for x in scored if x["verdict"] == "ROBUST"),
        "lanes_mostly": sum(1 for x in scored if x["verdict"] == "MOSTLY"),
        "lanes_fragile": sum(1 for x in scored if x["verdict"] == "FRAGILE"),
        "lanes_absent": sum(1 for x in scored if x["verdict"] == "ABSENT"),
        "restatements_held": sum(x["held"] for x in scored),
        "restatements_of": sum(x["of"] for x in scored),
        "lanes": lanes,
        "per_prompt": per,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dossier", type=Path, action="append", default=[])
    ap.add_argument("--sealed", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    # The self-test runs on EVERY invocation, not only when asked for.  A rubric
    # whose guard is optional is a rubric with no guard.
    if self_test(args.bank) != 0:
        return 1
    if args.self_test and not args.dossier:
        return 0
    if not args.out:
        raise SystemExit("--out is required when scoring dossiers")

    scored = [score_dossier(p) for p in args.dossier]
    body: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave40-paraphrase-score-v1",
        "purpose": (
            "Robustness reading on the paraphrase-triplet surface.  Lane-keyed, "
            "content-required, degeneracy-vetoed, and self-tested against its own "
            "prompts.  It reports whether a sealed case survives rewording; it is "
            "never a grade and never a gate."
        ),
        "bank_sha256": __import__("hashlib").sha256(args.bank.read_bytes()).hexdigest(),
        "lane_rules": len(RUBRIC),
        "unanswerable_lanes": UNANSWERABLE,
        "children": scored,
    }
    if args.sealed is not None:
        sealed = json.loads(args.sealed.read_text(encoding="utf-8"))
        body["against_sealed"] = [
            {
                "label": c["label"],
                "sealed_15": sealed.get(c["label"]),
                "lanes_scored": c["lanes_scored"],
                "robust": c["lanes_robust"],
                "mostly": c["lanes_mostly"],
                "fragile": c["lanes_fragile"],
                "absent": c["lanes_absent"],
                "restatements": f"{c['restatements_held']}/{c['restatements_of']}",
            }
            for c in scored
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")

    for c in scored:
        print(f"\n{c['label']}  {c['restatements_held']}/{c['restatements_of']} restatements, "
              f"{c['lanes_scored']} lanes scored, {c['prompts_unscorable']} prompts unscorable")
        for lane in c["lanes"]:
            if lane["verdict"] == "UNSCORED":
                continue
            print(f"   {lane['lane']:<32} {lane['held']}/{lane['of']}  {lane['verdict']}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
