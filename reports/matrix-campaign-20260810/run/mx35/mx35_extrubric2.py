#!/usr/bin/env python3
"""The extension-v2 surface's machine score: a PROMPT-keyed rubric with the holes shut.

Wave 33 measured five defects in the v1 extension rubric and repaired none of
them, on purpose: retro-fitting a guard onto a rubric that had already produced
sixty committed observations would rewrite four waves of output after seeing it.
So v1 stays frozen and this is the second instrument, sealed with its own bank
in one commit before a single wave-34 token was generated.

**Every one of the five defects is closed here by construction, and each one was
measured on v1 before it was closed** (`mx35_instrument.py`, wave 33's I1).

1. **THE DEGENERACY VETO IS CLOSED.**  v1's `score_one` computed five charged
   kinds and vetoed a hold on `empty` and `echo` only, so `ARCH.BM50N` -- the
   campaign's 0/15 child, sixteen of sixteen lanes charged -- HELD THREE entries
   on a 1,024-token repetition of the word "not", a 1,024-token repetition of an
   apostrophe, and 75 repetitions of "to".  Here any charged kind ends the entry.

2. **NO REQUIRED PATTERN MAY BE SATISFIED BY THE PROMPT ITSELF.**  v1's
   `ext-logic-four-seats` required `ada.*bo.*cy.*dee` and its prompt stated the
   constraints in exactly that order, so quoting the question back passed the
   answer test; a frontier child holds that entry on a response that never
   produces a seating order.  Six of v1's thirty were prompt-satisfiable and
   those six pooled to rho 0.0781 against 0.2455 for the other twenty-four.
   `self_test` runs every rule against its OWN prompt and REFUSES if any prompt
   passes.  It runs on every invocation and there is no flag to skip it.

3. **CONTENT, NOT FORM.**  Six of v1's thirty carried an empty `req` list and
   graded on formatting and length alone; the worst,
   `ext-conversation-competing-promises`, was held by 39 of 60 children and
   correlated NEGATIVELY with the sealed count.  Every rule below names at least
   one thing the answer has to contain, and `--audit-rules` asserts it.

4. **NO DEAD FAMILIES.**  v1's `context` and `explanation` families were never
   held by any of sixty children -- four prompts contributing nothing.  Both are
   redesigned in the v2 bank rather than dropped, and the rules below are written
   against what the redesign made available: an inline payload with a length
   ceiling for context, a named misconception plus a mechanism word for
   explanation.

5. **DIFFICULTY IS STAGGERED INSIDE THE LANE.**  Each lane has an A rule and a B
   rule and the B rule carries the added constraint.  A lane that both prompts
   pass or both fail contributes one bit; v1 had no lane where the two prompts
   were meant to differ.

**This score is not a verdict and cannot become one.**  ADDENDUM 16.1 sentence 2
stands: the sealed fifteen, read by hand, remain the only judge of any frontier
claim.  What this file produces is a second coordinate.

Usage::

    mx35_extrubric2.py --bank <lfm2_extension_bank_prompts_v2.json> --self-test
    mx35_extrubric2.py --bank <bank> --dossier <extension2-X.json> ... \\
        --sealed <label -> 15-count json> --out <json>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

#: One rule per PROMPT -- v2's two prompts in a lane carry different content and
#: different difficulty, so a lane-keyed rule would grade the pair as one.
#:
#:   req             a conjunction of regexes (re.I | re.S)
#:   forbid          a disjunction that fails the entry
#:   forbid_in_items forbidden inside the answer's OWN numbered items only
#:   order           literals that must appear in this ORDER by position
#:   min_words       a floor
#:   max_words       a ceiling, where the ask states one or where echoing is the
#:                   failure mode the entry exists to catch
#:   ast_fn          a parsable function definition of this name
#:   distinct_items  numbered items that must not be near-copies of one another
RUBRIC: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------- conversation
    "ext2-conversation-dead-battery": {
        # The register word is the content requirement and the prompt supplies
        # none of them; the question mark is the second, and the ask carries no
        # question mark at all.
        "req": [
            r"\?",
            r"\b(rough|awful|miserable|horrible|stressful|exhaust\w*|frustrat\w*|"
            r"glad|relief|relieved|sorry|ugh|nightmare|grim|painful|drag|hassle)\b",
        ],
        "forbid": [r"\?[^?]*\?", r"(?m)^\s*(\d[.)]|[-*•])\s"],
        "min_words": 15,
        "max_words": 200,
    },
    "ext2-conversation-two-promises": {
        "req": [
            r"\?",
            r"\b(tough|hard|awful|rough|stuck|bind|torn|clash|overlap|double.?booked|"
            r"impossible|unfair|horrible|no good options|squeez\w*)\b",
        ],
        "forbid": [
            r"\?[^?]*\?",
            r"(?m)^\s*(\d[.)]|[-*•])\s",
            r"\b(you should|I would (pick|choose|go)|go with the|pick the|choose the|"
            r"my advice is)\b",
        ],
        "min_words": 20,
        "max_words": 220,
    },
    # ---------------------------------------------------------- instruction
    "ext2-instruction-first-week": {
        "req": [
            r"\b(ask|read|shadow|meet|introduce|schedul\w*|list|map|write down|note|"
            r"review|attend|observ\w*|find out|sit with|request|draw)\w*\b"
        ],
        "order": ["1", "2", "3"],
        "forbid": [r"(?m)^\s*4[.)]\s"],
        "min_words": 20,
        "distinct_items": 3,
    },
    "ext2-instruction-power-restored": {
        "req": [
            r"\b(check|verif\w*|inspect|restart|reboot|switch|test|confirm|walk|"
            r"reset|open|read)\w*\b",
            r"\b(server|computer|fridge|freezer|router|alarm|light|breaker|clock|"
            r"network|equipment|machine|system|ups|switchboard|socket|thermostat)s?\b",
        ],
        "order": ["1", "2", "3"],
        "forbid": [
            r"(?m)^\s*4[.)]\s",
            r"\b(electricity company|utility company|power company|energy supplier|"
            r"call the utility|ring the electricity)\b",
        ],
        "min_words": 25,
        "distinct_items": 3,
    },
    # ---------------------------------------------------------- explanation
    "ext2-explanation-second-tide": {
        # The ask supplies "two", "second" and "moon".  What it does not supply
        # is the mechanism: the far-side water is held LESS strongly, so it is
        # left behind.  That word set is the whole requirement.
        "req": [
            r"\b(inertia|centrifugal|weaker|weakest|less|least|farther|further|"
            r"left behind|lags|bulge|bulges|flung|thrown outward|orbit\w*)\b",
            r"\b(far side|opposite side|other side|away from|facing away|back side|"
            r"second bulge|both sides)\b",
        ],
        "min_words": 45,
    },
    "ext2-explanation-blue-then-red": {
        "req": [
            r"\b(scatter|scatters|scattered|scattering|rayleigh)\b",
            r"\b(short\w* wavelength|shorter|blue light|smaller wavelength|high\w* "
            r"frequency)\b",
            r"\b(longer path|more atmosphere|thicker|greater distance|low angle|"
            r"lower angle|through more|near the horizon|long\w* wavelength)\b",
            r"\b(not|isn'?t|does not|doesn'?t|nothing to do with|has no)\b"
            r"[^.]{0,60}\b(reflect\w*|ocean|sea)\b",
        ],
        "min_words": 55,
    },
    # ----------------------------------------------------------- arithmetic
    "ext2-arithmetic-bolt-bags": {
        "req": [r"\b149\b", r"\b18\b", r"\b5\b"],
        "min_words": 8,
    },
    "ext2-arithmetic-faulty-run": {
        "req": [r"\b216\b", r"\b306\b"],
        "min_words": 10,
    },
    # ---------------------------------------------------------------- logic
    "ext2-logic-shelf-order": {
        # The constraints are stated in the answer order, so a positional scan
        # alone is satisfied by the question.  The END of the order is not.
        "req": [
            r"\bprimer\b[^.]{0,40}\b(last|fourth|4th|rightmost|right.most|far right)\b"
            r"|\b(last|fourth|4th|rightmost|right.most|far right)\b[^.]{0,40}\bprimer\b"
        ],
        "order": ["atlas", "ledger", "manual", "primer"],
        "min_words": 12,
    },
    "ext2-logic-relay-finish": {
        "req": [
            r"\bfitz\b[^.]{0,40}\b(last|fifth|5th)\b"
            r"|\b(last|fifth|5th)\b[^.]{0,40}\bfitz\b"
        ],
        "order": ["gil", "dara", "erin", "hana", "fitz"],
        "min_words": 15,
    },
    # -------------------------------------------------------------- context
    "ext2-context-workshop-notice": {
        # Both facts are IN the notice, which is what a retrieval lane is.  The
        # ceiling is the discriminator and the ask states it: two sentences.  The
        # notice is 78 words, so returning it fails on length and on the echo
        # charge, and answering fails on neither.
        "req": [r"4:?15", r"\byellow\b"],
        "min_words": 5,
        "max_words": 60,
    },
    "ext2-context-clinic-notice": {
        "req": [
            r"\bfriday\b",
            r"\b10\b",
            r"\bthursday\b[^.]{0,60}\b(no|not|cannot|can'?t|unable|closed|unavailable)\b"
            r"|\b(no|not|cannot|can'?t|unable)\b[^.]{0,60}\bthursday\b",
        ],
        "min_words": 12,
        "max_words": 110,
    },
    # -------------------------------------------------------- summarization
    "ext2-summary-vendor-choice": {
        "req": [
            r"\bmarlin\b",
            r"\b(chose|chosen|selected|picked|decision|went with|settled on|"
            r"decided)\b",
            r"\b(retention|legal|confirm\w*|unresolved|open|outstanding|pending|"
            r"unconfirmed|not yet)\b",
        ],
        "min_words": 12,
        "max_words": 75,
    },
    "ext2-summary-friday-backlog": {
        "req": [
            r"\b(certificat\w*|gateway)\b[^.]{0,60}\b(first|immediat\w*|urgent|today|"
            r"priority|soonest|now)\b"
            r"|\b(first|immediat\w*|urgent|priority|soonest)\b[^.]{0,60}"
            r"\b(certificat\w*|gateway)\b",
            r"\b(hero|image|marketing)\b[^.]{0,60}\b(wait|later|defer\w*|last|lowest|"
            r"least|postpone\w*)\b"
            r"|\b(wait|later|defer\w*|postpone\w*|lowest|least)\b[^.]{0,60}"
            r"\b(hero|image|marketing)\b",
        ],
        "min_words": 15,
        "max_words": 110,
    },
    # ------------------------------------------------------------ comparison
    "ext2-comparison-cache-memory": {
        "req": [
            r"\b(desk|workbench|bench|table|counter|worktop|pocket|wallet|backpack|"
            r"bag|drawer|shelf|shelves|cupboard|pantry|fridge|kitchen|toolbox|"
            r"notebook|scratch\s*pad|whiteboard|library|warehouse)\b",
            r"\b(cache)\b",
            r"\b(main memory|ram|memory)\b",
            r"\b(mislead\w*|break\w*|fail\w*|wrong|imperfect|unlike|does not|doesn'?t|"
            r"limit\w*|inaccurat\w*)\b",
            r"\b(two|second|secondly|another|also|both)\b",
        ],
        "min_words": 60,
    },
    "ext2-comparison-bandwidth-latency": {
        "req": [
            r"\b(mislead\w*|break\w*|fail\w*|wrong|imperfect|unlike|does not|doesn'?t|"
            r"limit\w*|inaccurat\w*)\b",
            r"\b(bandwidth|throughput|how much|amount|volume|capacity)\b",
            r"\b(latency|delay|how long|time before|waiting|lag)\b",
            r"\b(two|second|secondly|another|also|both)\b",
        ],
        "forbid": [
            r"\b(road|roads|traffic|highway|motorway|freeway|pipe|pipes|piping|water|"
            r"hose|plumb\w*|lane|lanes)\b"
        ],
        "min_words": 60,
    },
    # ---------------------------------------------------------------- coding
    "ext2-coding-collapse-runs": {
        "req": [
            r"def\s+collapse_runs",
            r"(\[-1\]|\bprev\w*\b|\blast\b|\bprevious\b|zip\(|enumerate\()",
        ],
        "forbid": [r"\bset\(", r"\bitertools\b", r"dict\.fromkeys"],
        "min_words": 10,
        "ast_fn": "collapse_runs",
    },
    "ext2-coding-first-unique": {
        "req": [
            r"def\s+first_unique",
            r"\b(count|counts|counter|tally|freq\w*|occurrence\w*)\b",
            r"\bNone\b",
        ],
        "forbid": [r"\.sort\(", r"\bsorted\("],
        "min_words": 12,
        "ast_fn": "first_unique",
    },
    # ------------------------------------------------------------- debugging
    "ext2-debug-tail-slice": {
        "req": [
            r"items\[-\s*n\s*:\]|items\[len\(items\)\s*-\s*n\s*:\]|\[-\s*n\s*:\]",
            r"\b(off.by.one|one too many|extra|minus one|subtract\w*|too far|"
            r"one more|n\s*\+\s*1)\b",
        ],
        "min_words": 12,
    },
    "ext2-debug-adjacent-pairs": {
        "req": [
            r"range\(\s*len\(values\)\s*-\s*1\s*\)|range\(\s*0\s*,\s*len\(values\)\s*-\s*1\s*\)"
            r"|zip\(values,\s*values\[1:\]\)",
            r"\b(last|final|bound|index|out of range|indexerror|one past|beyond)\b",
        ],
        "min_words": 12,
    },
    # --------------------------------------------------------------- writing
    "ext2-email-wrong-colour": {
        "req": [
            r"\b7734\b",
            r"\brefund\w*\b",
            r"\bthursday\b",
            r"\b(address|where|which branch|which shop|which store|location)\b",
            r"\b(dear|hello|hi\b|good morning|good afternoon|regards|sincerely|"
            r"best wishes|many thanks|thank you|thanks|yours)\b",
            r"\?|\b(confirm|let me know|could you|please could|would you)\b",
        ],
        "min_words": 25,
        "max_words": 145,
    },
    "ext2-email-reschedule": {
        "req": [
            r"\bmonday\b",
            r"\b(10:?00|10 ?a\.?m|ten)\b",
            r"\bwednesday\b",
            r"\bfriday\b",
            r"\b(dear|hello|hi\b|good morning|good afternoon|regards|sincerely|"
            r"best wishes|many thanks|thank you|thanks|yours)\b",
            r"\?|\b(reply|confirm|let me know|get back|respond)\b",
        ],
        # More than one apology.  The ask says once; a model reaching for
        # politeness says it three times, and that is the failure this entry
        # exists to catch.
        "forbid": [r"(sorry|apolog\w*)[\s\S]{0,4000}?(sorry|apolog\w*)"],
        "min_words": 40,
        "max_words": 160,
    },
    # -------------------------------------------------------------- planning
    "ext2-planning-office-move": {
        "req": [
            r"\b(phone|number|line|telephone)\b",
            r"\b(rack|server)\b",
            r"\b(bank|address)\b",
            r"\b(clean\w*|contract|cancel\w*)\b",
            r"\b(check|verif\w*|test\w*|confirm\w*)\b",
        ],
        "order": ["1", "2", "3", "4", "5"],
        "forbid": [r"(?m)^\s*6[.)]\s"],
        "min_words": 40,
        "distinct_items": 5,
    },
    "ext2-planning-db-cutover": {
        "req": [
            r"\b(backup|back up|dump|snapshot)\b",
            r"\b(freeze|read.only|readonly|lock\w*)\b",
            r"\b(copy|transfer|replicat\w*|restor\w*|migrat\w*)\b",
            r"\b(dns|connection.string|endpoint|cname|connection string)\b",
            r"\b(smoke|check|verif\w*|test\w*)\b",
            r"\b(roll ?back|revert|back out|undo|fail ?back|switch back|restore the "
            r"old|point back)\b",
        ],
        "order": ["1", "2", "3", "4", "5", "6"],
        "forbid": [r"(?m)^\s*7[.)]\s"],
        "min_words": 50,
        "distinct_items": 6,
    },
    # ----------------------------------------------------------- uncertainty
    "ext2-uncertainty-churn-jump": {
        "req": [
            r"\b(cohort|segment\w*|sample size|how many customers|absolute number|"
            r"denominator|time ?frame|period|since when|over what|base|churn "
            r"reason|which customers|why)\b",
            r"\b(cannot|can'?t|not enough|insufficient|no basis|too early|unclear|"
            r"would need|need to know|without|before)\b",
            r"\?|\b(would ask|ask|request|find out|check)\b",
        ],
        "min_words": 45,
    },
    "ext2-uncertainty-first-hour": {
        "req": [
            r"\b(sample|sample size|underpowered|significan\w*|noise|noisy|chance|"
            r"random|confidence|too few|too small|variance|luck)\b",
            r"\b(not|don'?t|do not|wait|hold|too early|premature|cannot|shouldn'?t|"
            r"should not|hold off)\b",
            r"\b(how long|run it|more data|longer|another day|until|keep running|"
            r"continue|let it run|a week)\b",
        ],
        "min_words": 40,
    },
    # -------------------------------------------------------------- creative
    "ext2-creative-cold-cellar": {
        "req": [
            r"\b(cellar|basement)\b",
            r"\b(furniture|upstairs|above|dragging|scraping|shifting|moving)\b",
            r"\b(count\w*|number\w*|tally|one,? two|counting)\b",
            r"[\"“][^\"”]{4,}[\"”]|'[^']{6,}'",
        ],
        "min_words": 60,
    },
    "ext2-creative-own-handwriting": {
        "req": [
            r"\b(note|paper|handwriting|writing|script)\b",
            r"\b(fog|mist|haze)\b",
            r"\b(boat|supply|ship|vessel|tender)\b",
            r"[\"“][^\"”]{4,}[\"”]|'[^']{6,}'",
        ],
        "forbid": [
            r"\b(it turned out|the note was written by|she had written it|it was her "
            r"own hand|revealed that|explained that|the answer was)\b"
        ],
        "min_words": 70,
    },
    # ------------------------------------------------------------ multi-turn
    "ext2-multiturn-venue-revision": {
        "req": [
            r"\b(vegetarian|plant.based|meat.free|veggie|vegan)\b",
            r"\b(wheelchair|step.free|accessib\w*|level access|ramp|ground floor|lift)\b",
            r"\b(chang\w*|swapp\w*|replac\w*|instead|dropped|removed|out|new)\b",
        ],
        "order": ["1", "2", "3", "4"],
        # The two venues the new constraints rule out, BY NAME.  Not the cuisine:
        # a barbecue place with a real vegetarian menu and level access is a
        # legitimate answer, and forbidding the word would charge it.
        "forbid_in_items": [r"\bthe iron grill\b", r"\bthe smokehouse\b"],
        "min_words": 30,
        "distinct_items": 4,
    },
    "ext2-multiturn-dinner-revision": {
        "req": [
            r"\b(gluten.free|gluten free|no gluten|without gluten|rice|corn|buckwheat|"
            r"quinoa|polenta|potato)\b",
            r"\b(because|works|since|avoids|contains no|free of|instead|so it)\b",
        ],
        "order": ["1", "2", "3"],
        # The three dishes the new constraints rule out, plus carbonara's two
        # meats.  NOT the bare words `fish` and `pork`: a correct item may say
        # "no fish or pork" inside its own line, and charging that would charge
        # the answers that took the constraint most seriously.
        "forbid_in_items": [
            r"\bsalmon\b", r"\bcarbonara\b", r"\bcouscous\b",
            r"\bpancetta\b", r"\bbacon\b",
        ],
        "min_words": 30,
        "distinct_items": 3,
    },
}

#: TWO ENTRIES ARE HELD OFF THE PROMPT BY A LENGTH BOUND AND NOTHING ELSE, and
#: they are named here rather than discovered.  Retrieval and summarisation are
#: the two lanes whose answer content is, by definition, inside the prompt: the
#: notice contains "4:15" and "yellow" because a retrieval lane that hid them
#: would not be a retrieval lane.  What separates an answer from an echo on those
#: two is COMPRESSION, so the ceiling carries the entry -- backed by the echo
#: charge, which ends any entry independently.  A third entry landing in this set
#: is a rule that has quietly become form-only, and the self-test refuses it.
LENGTH_ONLY_BY_DESIGN = {
    "ext2-context-workshop-notice": "the two facts are in the notice; two sentences is the test",
    "ext2-summary-vendor-choice": "the decision and the risk are in the paragraph; three sentences is the test",
}

WORD_SUBSTITUTION = re.compile(r"\b(\w{4,})\b[^.]{0,30}\b\1[- ]?like\b[^.]{0,30}\b\1\b", re.I)
NUMBERED = re.compile(r"(?m)^\s*(\d)[.)]\s*(.+)$")


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
    two items repeated is a miss.  Two items are 'the same' when their word sets
    overlap at 70 per cent or more, which is loose on purpose: this catches
    'Prep and prepare' three times and does not referee near-synonyms.
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
    # what it is removing, before the list or after it; what it may not do is put
    # it back in one of the options.
    #
    # THIS IS A DELIBERATE DIFFERENCE FROM THE WAVE-33 PARAPHRASE RUBRIC, which
    # took the region from the first numbered marker to the END of the text.  On
    # this bank that is wrong in a way that matters: both multi-turn prompts ask
    # the answer to SAY WHAT CHANGED, and a correct answer names the dropped
    # option in a trailing sentence.  Under the tail rule the best answers would
    # be the ones charged.  So the region is the numbered items themselves,
    # `NUMBERED`'s own captures, and an item that runs past its first line is
    # checked on that first line only.
    forbid_items = rule.get("forbid_in_items") or []
    if forbid_items:
        items = [m.group(2) for m in NUMBERED.finditer(text)]
        region = "\n".join(items)
        for pattern in forbid_items:
            if region and re.search(pattern, region, re.I | re.S):
                reasons.append(f"forbidden inside the items: {pattern[:40]}")
    order = rule.get("order")
    if order:
        cursor, ok = 0, True
        low = text.lower()
        for literal in order:
            needle = literal.lower()
            if literal.isdigit():
                m = re.compile(rf"(?m)^\s*{literal}[.)]\s").search(low, cursor)
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
    prompt_id = record["id"]
    text = record.get("response_verbatim", "") or ""
    charged = degeneracy(record)
    rule = RUBRIC.get(prompt_id)
    if rule is None:
        return {"id": prompt_id, "held": False, "scored": False, "charged": charged,
                "reason": "no rubric for this prompt"}
    reasons = apply_rule(rule, text)
    # THE VETO IS CLOSED.  Any charged kind ends the entry.
    if charged:
        reasons.append("charged: " + ",".join(charged))
    return {
        "id": prompt_id,
        "lane": record.get("lane"),
        "scored": True,
        "held": not reasons,
        "charged": charged,
        "words": len(re.findall(r"\S+", text)),
        "reason": "; ".join(reasons) if reasons else "held",
    }


def audit_rules() -> list[str]:
    """Every rule must name something the answer has to CONTAIN.

    v1's six form-only entries are the reason this exists.  `req`, `ast_fn`,
    `order` on word literals and `forbid_in_items` all name content; `order` on
    bare digits, `min_words`, `max_words` and `forbid` do not, on their own.
    """
    problems: list[str] = []
    for key, rule in RUBRIC.items():
        content = bool(rule.get("req")) or bool(rule.get("ast_fn")) or bool(
            rule.get("forbid_in_items")
        ) or any(not str(x).isdigit() for x in rule.get("order", []))
        if not content:
            problems.append(f"{key}: form-only rule -- it names nothing the answer must contain")
    return problems


def self_test(bank_path: Path) -> int:
    """Refuse if any prompt satisfies its own rule, or if a rule grades only form."""
    doc = json.loads(bank_path.read_text(encoding="utf-8"))
    prompts = doc["prompts"]
    problems: list[str] = audit_rules()
    lanes: dict[str, list[str]] = {}
    for row in prompts:
        lanes.setdefault(row["lane"], []).append(row["id"])
        rule = RUBRIC.get(row["id"])
        if rule is None:
            problems.append(f"{row['id']}: the bank carries this prompt and the rubric does not")
            continue
        ask = "\n".join(m["content"] for m in row["messages"])
        reasons = apply_rule(rule, ask)
        if not reasons:
            problems.append(
                f"{row['id']}: THE PROMPT SATISFIES ITS OWN RULE -- a child that "
                f"quotes the question back would hold this entry"
            )
            continue
        # A prompt kept off its own rule by a word count alone is one edit away
        # from being prompt-satisfiable, so the two entries where that is the
        # design are DECLARED and any third one refuses.
        substantive = [r for r in reasons if not r.startswith(("short (", "long ("))]
        if not substantive and row["id"] not in LENGTH_ONLY_BY_DESIGN:
            problems.append(
                f"{row['id']}: the only thing keeping the prompt off its own rule "
                f"is a word count.  Declare it in LENGTH_ONLY_BY_DESIGN with a "
                f"reason or give the rule some content"
            )
        if substantive and row["id"] in LENGTH_ONLY_BY_DESIGN:
            problems.append(
                f"{row['id']}: declared LENGTH_ONLY_BY_DESIGN but the prompt fails "
                f"its rule for {len(substantive)} other reason(s); the declaration "
                f"is stale"
            )
    ids = {row["id"] for row in prompts}
    for key in RUBRIC:
        if key not in ids:
            problems.append(f"{key}: the rubric carries this prompt and the bank does not")
    for lane, members in sorted(lanes.items()):
        if len(members) != 2:
            problems.append(f"{lane}: {len(members)} prompts, the design is two per lane")
    steps = {row["id"]: row.get("step") for row in prompts}
    for lane, members in sorted(lanes.items()):
        if sorted(steps.get(m) or "" for m in members) != ["A", "B"]:
            problems.append(f"{lane}: difficulty is not staggered A/B inside the lane")
    if problems:
        for p in problems:
            print("SELF-TEST FAILED:", p, file=sys.stderr)
        return 1
    print(f"self-test: {len(RUBRIC)} prompt rules over {len(lanes)} lanes, two per lane, "
          f"A/B staggered; no prompt satisfies its own rule; no rule grades form "
          f"alone; the degeneracy veto is closed")
    return 0


def score_dossier(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    # The parent's dossier carries `parent_responses` and an empty
    # `child_responses`; a scorer that reads only the second key reports the
    # PARENT at 0/30 and silently loses the row that says whether the rubric is
    # calibrated or merely strict.
    records = doc.get("child_responses") or doc.get("parent_responses") or []
    per = [score_one(r) for r in records]
    by_lane: dict[str, list[dict[str, Any]]] = {}
    for row, rec in zip(per, records):
        lane = rec.get("lane") or "unknown"
        row["lane"] = lane
        by_lane.setdefault(lane, []).append(row)
    lanes = []
    for lane in sorted(by_lane):
        rows = by_lane[lane]
        held = sum(1 for r in rows if r["held"])
        lanes.append({
            "lane": lane, "held": held, "of": len(rows),
            "charged": sum(1 for r in rows if r["charged"]),
            "which": [r["id"] for r in rows if r["held"]],
        })
    return {
        "label": doc.get("label"),
        "cell_id": doc.get("cell_id"),
        "surface": doc.get("surface"),
        "bank_sha256": doc.get("bank_sha256"),
        "prompts": len(per),
        "held": sum(1 for r in per if r["held"]),
        "charged_prompts": sum(1 for r in per if r["charged"]),
        "lanes_with_a_hold": sum(1 for x in lanes if x["held"] > 0),
        "lanes_both": sum(1 for x in lanes if x["held"] == 2),
        "held_A": sum(1 for r in per if r["held"] and _step_of(r["id"]) == "A"),
        "held_B": sum(1 for r in per if r["held"] and _step_of(r["id"]) == "B"),
        "lanes": lanes,
        "per_prompt": per,
    }


#: Filled by `main` from the bank, so `score_dossier` can split A from B without
#: re-reading the bank per child.
_STEPS: dict[str, str] = {}


def _step_of(prompt_id: str) -> str:
    return _STEPS.get(prompt_id, "?")


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
    doc = json.loads(args.bank.read_text(encoding="utf-8"))
    _STEPS.update({row["id"]: row.get("step", "?") for row in doc["prompts"]})
    if args.self_test and not args.dossier:
        return 0
    if not args.out:
        raise SystemExit("--out is required when scoring dossiers")

    scored = [score_dossier(p) for p in args.dossier]
    body: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave35-extension2-score-v1",
        "purpose": (
            "The extension-v2 surface's machine score.  Prompt-keyed, "
            "content-required, degeneracy-vetoed, difficulty-staggered, and "
            "self-tested against its own prompts on every invocation.  It prices "
            "and orders; it is never a grade and never a gate."
        ),
        "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        "prompt_rules": len(RUBRIC),
        "children": scored,
    }
    if args.sealed is not None:
        sealed = json.loads(args.sealed.read_text(encoding="utf-8"))
        body["against_sealed"] = [
            {
                "label": c["label"],
                "sealed_15": sealed.get(c["label"]),
                "extension2_30": c["held"],
                "held_A": c["held_A"],
                "held_B": c["held_B"],
                "lanes_with_a_hold": c["lanes_with_a_hold"],
                "charged_prompts": c["charged_prompts"],
            }
            for c in scored
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")

    for c in scored:
        print(f"\n{c['label']}  {c['held']}/{c['prompts']} held "
              f"(A {c['held_A']}, B {c['held_B']}), {c['charged_prompts']} charged, "
              f"{c['lanes_with_a_hold']} lanes with a hold")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
