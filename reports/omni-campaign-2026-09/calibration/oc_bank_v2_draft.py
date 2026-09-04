#!/usr/bin/env python3
"""DRAFT bank v2 for the omni campaign -- authored, self-tested, digested, UNSEALED.

READ THIS FIRST.  This bank is a **DRAFT PENDING CYRIL**.  The plan (section 5.1)
says bank v2 is authored WITH him; nothing here is sealed, nothing here is a gate,
and every threshold in every rubric is a PROPOSED DECISION LINE.  The file writes
`status: DRAFT` into the bank body and into every single case, so a bank that
leaks into a runner cannot pretend to be the frozen surface.  The frozen 15-case
V5 pack (`0ee6bf14...`) is untouched and remains the only sealed review bank.

WHAT A RUBRIC IS FOR HERE.  It is a MACHINE DIAGNOSTIC and an aid to reading, not
a score.  The estate's standing rule is that only measured intelligence read by a
human grades, and the artifact-tolerant lens says a miss must be sorted into
artifact-miss and content-miss by someone who read the text.  A rubric that
matches tells a reader "the literal the case asked for is present"; a rubric that
does not match tells a reader "go read this one".  Neither is a verdict, and
`oc_assess.py` never turns one into a count.

THE DESIGN RULES THE CASES ARE BUILT TO (from the sealed bank-v2 design rules,
which the extension bank v2 already passes):

  1. **Self-contained.**  A case carries everything needed to answer it.
  2. **Content, not form.**  The rubric asks for a fact or a name, never a
     phrasing.  Word budgets exist only where the ASK is a budget.
  3. **Not prompt-quotable.**  `--self-test` renders each case's ASK SURFACE --
     the prompt with the supplied payload cut away -- and refuses to emit the
     bank if any case's own rubric is satisfied by it.  A case a copier passes is
     a case that measures copying.
  4. **A/B difficulty.**  Every lane exists at step A (the reachable rung) and,
     where it is meaningful, step B (the rung that separates).
  5. **7-gram disjoint, over all forty cases, with one stated exclusion.**
     `--self-test` refuses any 7-gram shared between two different cases and
     between a case and the frozen v1 pack, so a model cannot carry an answer
     across lanes.  The text compared is `comparable_prompt_text`: the prompt as
     the bank stores it -- PLACEHOLDER form for a long-context case, since
     `<<HAYSTACK>>` is not expanded in the bank -- plus every needle string.  The
     GENERATED FILLER IS EXCLUDED and the reason is in that function: it is drawn
     from a deliberately tiny vocabulary in one sentence template, so any two
     haystacks share thousands of 7-grams by construction and always will.  A
     separate refusal keeps needle words out of the filler vocabulary.
     This rule previously exempted every long-context case with an uncommented
     filter, and under that exemption three pairs of rungs shipped BYTE-IDENTICAL
     prompts while the lane claimed "0 failures over 40 cases".  Each rung now
     asks its own question about its own needle: the six retrieval rungs are
     MATCHED IN SHAPE (find one notice among distractors, state three facts from
     it) and DISJOINT IN CONTENT, which is what makes an 8k-versus-200k
     comparison a length effect instead of a memory of the previous rung.
  6. **Reference answers reachable.**  Every ground truth is derivable from the
     prompt by a reader in under a minute; the long-context needles are literal
     strings this file placed.

LONG CONTEXT IS SPECIFIED, NOT INLINED.  A 200,000-token haystack in a JSON bank
would be about a megabyte of filler per case and would make the bank unreadable.
Instead each long-context case carries a `haystack_spec` -- generator name, seed,
sentence count, needle strings and their fractional positions -- and the sha256 of
the text that spec produces.  `materialise_haystack` is deterministic, the digest
is checked at run time, and the bank digest therefore covers the haystack text
transitively.  The nominal token counts (8,192 / 32,768 / 65,536 / 200,000) are
TARGETS: the true count depends on the parent's tokenizer and is recorded per run
in the dossier, never assumed.

WHAT IS NOT HERE, AND IS AN OPEN QUESTION FOR CYRIL:
  * No safety lane.  The frozen v1 pack's `safety-household-chemical` remains the
    charged-not-scored lane.  Adding a second safety surface changes what
    "charged" means and that is his call, not mine.
  * The five AV cases are SPEC ONLY (`runnable: false`).  They need a generated
    clip with a known script and an ASR reference; the recipe is written into
    each case so the assets can be built the day H3 output is authorised for
    review use.
  * The four curated photographs R6 sketched beside the five generated cards are
    not included; the generated cards have exact ground truth and the photographs
    do not until someone labels them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oc_calib import (  # noqa: E402
    CalibrationError,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    user_turns_text,
)
from oc_testcards import build_cards  # noqa: E402

BANK_SCHEMA = "neural-forge-omni-review-bank-v2-draft"
DRAFT_STATUS = (
    "DRAFT -- PENDING CYRIL'S REVIEW.  Not sealed, not a gate, not a grade.  "
    "Every threshold in every rubric is a PROPOSED DECISION LINE for Cyril.  "
    "The frozen 15-case V5 pack remains the only sealed review surface."
)

# --------------------------------------------------------------------------
# Deterministic haystacks.
# --------------------------------------------------------------------------

#: Nominal tokens a filler sentence costs on a Qwen-class BPE.  Used ONLY to turn
#: a token target into a sentence count; the real count is measured per run.
TOKENS_PER_SENTENCE = 21

#: The filler vocabulary.  Disjoint from every needle's content words by
#: construction -- `--self-test` asserts it -- so a needle can never be confused
#: with filler, and the filler can never accidentally answer a question.
_ADJECTIVES = (
    "quiet", "amber", "narrow", "second", "outer", "steady", "hollow",
    "eastern", "spare", "level", "coastal", "upper",
)
_NOUNS = (
    "ledger", "conduit", "pallet", "reservoir", "manifold", "spindle",
    "annex", "culvert", "carriage", "lantern", "trestle", "cistern",
)
_PLACES = (
    "bay eleven", "the sorting shed", "dock nine", "the pump house",
    "corridor three", "the west yard", "store twelve", "the relay hut",
)
_VERBS = (
    "logged", "rotated", "inspected", "sealed", "re-weighed", "indexed",
    "swapped", "counted",
)


def _lcg(seed: int):
    """A 64-bit linear congruential generator.  Deterministic across platforms.

    `random.Random` would also be deterministic, but its stream is a promise the
    standard library makes about ITS implementation, and a bank whose text is
    pinned by digest should not depend on that promise across Python versions.
    """

    state = int(seed) & ((1 << 64) - 1)
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        yield state >> 11


def materialise_haystack(spec: dict[str, Any]) -> str:
    """The exact document a long-context case supplies.  Digest-pinned.

    Needles are placed at fractional positions in the sentence stream; each is
    its own sentence, so no needle can be split by the filler.
    """

    if spec.get("generator") != "oc-filler-v1":
        raise CalibrationError(f"unknown haystack generator {spec.get('generator')!r}")
    sentences_target = int(spec["sentences"])
    stream = _lcg(int(spec["seed"]))
    lines: list[str] = []
    for index in range(sentences_target):
        adjective = _ADJECTIVES[next(stream) % len(_ADJECTIVES)]
        noun = _NOUNS[next(stream) % len(_NOUNS)]
        place = _PLACES[next(stream) % len(_PLACES)]
        verb = _VERBS[next(stream) % len(_VERBS)]
        batch = next(stream) % 900 + 100
        lines.append(
            f"Record {index + 1}: the {adjective} {noun} in {place} was {verb} "
            f"by team {chr(65 + (next(stream) % 6))} during window {batch}."
        )
    for needle in spec["needles"]:
        position = int(float(needle["position_fraction"]) * (len(lines) - 1))
        lines[position] = str(needle["text"])
    return "\n".join(lines)


def _haystack_case(
    *,
    case_id: str,
    seed: int,
    target_tokens: int,
    needles: list[dict[str, Any]],
    instruction: str,
    rubric: dict[str, Any],
    ask: str,
    step: str,
    difficulty: str,
    context: str,
    lane: str,
) -> dict[str, Any]:
    spec = {
        "generator": "oc-filler-v1",
        "seed": seed,
        "sentences": max(8, round(target_tokens / TOKENS_PER_SENTENCE)),
        "target_tokens_nominal": target_tokens,
        "needles": needles,
    }
    text = materialise_haystack(spec)
    spec["haystack_sha256"] = canonical_sha256({"text": text})
    spec["haystack_characters"] = len(text)
    return {
        "id": case_id,
        "lane": lane,
        "category": "long-context",
        "step": step,
        "difficulty": difficulty,
        "ask": ask,
        "modality_in": "text",
        "modality_out": "text",
        "prompt_tokens_target": target_tokens,
        "status": DRAFT_STATUS,
        "context": context,
        "haystack_spec": spec,
        "messages": [
            {
                "role": "user",
                "content": (
                    instruction
                    + "\n\nDOCUMENT:\n<<HAYSTACK>>\n\nQuestion: "
                    + rubric["question"]
                ),
            }
        ],
        "payload_anchor": ["DOCUMENT:", "Question:"],
        "rubric": {
            key: value for key, value in rubric.items() if key != "question"
        },
    }


# --------------------------------------------------------------------------
# The thirty text cases.
# --------------------------------------------------------------------------


def _case(
    case_id: str,
    lane: str,
    category: str,
    step: str,
    difficulty: str,
    ask: str,
    messages: list[dict[str, str]],
    context: str,
    rubric: dict[str, Any],
    *,
    payload_anchor: list[str] | None = None,
    modality_in: str = "text",
    modality_out: str = "text",
    tokens_target: int = 512,
    runnable: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "id": case_id,
        "lane": lane,
        "category": category,
        "step": step,
        "difficulty": difficulty,
        "ask": ask,
        "modality_in": modality_in,
        "modality_out": modality_out,
        "prompt_tokens_target": tokens_target,
        "runnable": runnable,
        "status": DRAFT_STATUS,
        "context": context,
        "messages": messages,
        "payload_anchor": payload_anchor,
        "rubric": rubric,
    }
    if extra:
        body.update(extra)
    return body


def long_context_cases() -> list[dict[str, Any]]:
    """Nine rungs, MATCHED IN SHAPE AND DISJOINT IN CONTENT.

    The six retrieval rungs (8k/64k/200k at step A and step B) ask the same KIND
    of question -- find one notice among distractors and state three facts from it
    -- so an 8k score and a 200k score are comparable.  Every rung asks it about
    its own subject, with its own instruction, its own needle strings and its own
    question, because the first draft of this file reused one notice and one
    question across all three A rungs and all three B rungs, which made the 8k and
    200k prompts BYTE-IDENTICAL apart from the filler.  Three identical asks are
    one case measured three times only if the model cannot remember the first, and
    nothing here can promise that.  The 7-gram rule now covers these cases in
    placeholder form (see `comparable_prompt_text`) and it is what keeps them
    apart.
    """

    cases = [
        _haystack_case(
            case_id="lc-8k-notice-select-A",
            seed=8101,
            target_tokens=8192,
            needles=[
                {
                    "text": (
                        "NOTICE: the winter delivery window moves to Thursday at "
                        "6:40 a.m. and all drivers must enter through the north "
                        "gate."
                    ),
                    "position_fraction": 0.10,
                },
                {
                    "text": "NOTICE: the loading ramp inspection falls due in the same week.",
                    "position_fraction": 0.50,
                },
                {
                    "text": "NOTICE: drivers must carry the winter route card.",
                    "position_fraction": 0.90,
                },
            ],
            instruction=(
                "The document below is an operations log. Read it and answer the "
                "question after it."
            ),
            rubric={
                "question": (
                    "State the day, the time and the entrance named by the "
                    "delivery notice in this document. Answer in under 80 words."
                ),
                "required_all": [r"Thursday", r"6:40", r"north gate"],
                "forbidden_any": [],
                "max_words": 120,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="A",
            difficulty="reachable",
            context="day Thursday, time 6:40 a.m., entrance the north gate",
            lane="long-context-retrieval",
        ),
        _haystack_case(
            case_id="lc-8k-notice-select-B",
            seed=8102,
            target_tokens=8192,
            needles=[
                {
                    "text": (
                        "BULLETIN: the night watch rota starts Friday at 10:20 "
                        "p.m. from the tannery office. THIS BULLETIN IS CANCELLED "
                        "BY A LATER ONE PRINTED FURTHER DOWN."
                    ),
                    "position_fraction": 0.02,
                },
                {
                    "text": "BULLETIN: the boiler pressure check keeps its usual slot.",
                    "position_fraction": 0.60,
                },
                {
                    "text": (
                        "BULLETIN: the night watch rota starts Wednesday at 4:45 "
                        "a.m. from the millwright shop."
                    ),
                    "position_fraction": 0.98,
                },
            ],
            instruction=(
                "Below is a shift record. One bulletin in it is cancelled by a "
                "later bulletin that says so plainly."
            ),
            rubric={
                "question": (
                    "Which day, which clock time and which starting point govern "
                    "the night watch rota once the cancellation is applied? Keep "
                    "it under 60 words."
                ),
                "required_all": [r"Wednesday", r"4:45", r"millwright"],
                "forbidden_any": [r"Friday", r"10:20", r"tannery"],
                "max_words": 90,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="B",
            difficulty="separating",
            context="the surviving bulletin governs: Wednesday, 4:45 a.m., the millwright shop",
            lane="long-context-retrieval",
        ),
        _haystack_case(
            case_id="lc-8k-crossref-two-facts-B",
            seed=8103,
            target_tokens=8192,
            needles=[
                {
                    "text": "NOTE 12: consignment 417 was assigned to the twilight crew.",
                    "position_fraction": 0.15,
                },
                {
                    "text": "NOTE 31: the twilight crew works out of the granary loft.",
                    "position_fraction": 0.85,
                },
            ],
            instruction=(
                "Two separate entries in the record that follows have to be put "
                "together before its closing question can be answered."
            ),
            rubric={
                "question": (
                    "Which building does the crew handling consignment 417 work "
                    "out of, and what is that crew called? Reply in under 40 "
                    "words."
                ),
                "required_all": [r"granary", r"twilight"],
                "forbidden_any": [],
                "max_words": 80,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="B",
            difficulty="separating",
            context="consignment 417 -> twilight crew -> the granary loft",
            lane="long-context-crossref",
        ),
        _haystack_case(
            case_id="lc-32k-conflicting-updates-B",
            seed=32101,
            target_tokens=32768,
            needles=[
                {
                    "text": (
                        "UPDATE 1: the master key for the granary is held by "
                        "Marta Okonjo."
                    ),
                    "position_fraction": 0.20,
                },
                {
                    "text": (
                        "UPDATE 2 (this replaces UPDATE 1): the master key for "
                        "the granary is now held by Desmond Uhl."
                    ),
                    "position_fraction": 0.75,
                },
            ],
            instruction=(
                "Where two updates disagree in the material printed here, the "
                "later update replaces the earlier one."
            ),
            rubric={
                "question": (
                    "Who holds the master key for the granary at the end of this "
                    "material? Use fewer than 30 words."
                ),
                "required_all": [r"Desmond|Uhl"],
                "forbidden_any": [r"Marta|Okonjo"],
                "max_words": 60,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="B",
            difficulty="separating",
            context="UPDATE 2 governs: Desmond Uhl",
            lane="long-context-conflict",
        ),
        _haystack_case(
            case_id="lc-64k-notice-select-A",
            seed=64101,
            target_tokens=65536,
            needles=[
                {
                    "text": (
                        "NOTICE: the frozen-goods audit is rescheduled to Monday "
                        "at 3:05 p.m. and every auditor must sign in at the "
                        "archive vault."
                    ),
                    "position_fraction": 0.10,
                },
                {
                    "text": "NOTICE: the fuse cabinet was restocked that afternoon.",
                    "position_fraction": 0.50,
                },
                {
                    "text": "NOTICE: auditors bring their own hi-vis jacket.",
                    "position_fraction": 0.90,
                },
            ],
            instruction=(
                "What follows is a long depot journal. Use only what it contains "
                "when you answer the final question."
            ),
            rubric={
                "question": (
                    "Give the day, the clock time and the sign-in location for "
                    "the audit, in fewer than 80 words."
                ),
                "required_all": [r"Monday", r"3:05", r"archive vault"],
                "forbidden_any": [],
                "max_words": 120,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="A",
            difficulty="reachable",
            context="day Monday, time 3:05 p.m., sign-in at the archive vault",
            lane="long-context-retrieval",
        ),
        _haystack_case(
            case_id="lc-64k-notice-select-B",
            seed=64102,
            target_tokens=65536,
            needles=[
                {
                    "text": (
                        "ORDER: the quarry blast schedule opens Sunday at 1:30 "
                        "p.m. under foreman Iris Alvarado. THIS ORDER IS "
                        "WITHDRAWN LOWER IN THE SAME JOURNAL."
                    ),
                    "position_fraction": 0.02,
                },
                {
                    "text": "ORDER: the canteen menu board is refreshed every fortnight.",
                    "position_fraction": 0.60,
                },
                {
                    "text": (
                        "ORDER: the quarry blast schedule opens Tuesday at 7:15 "
                        "a.m. under foreman Piotr Nadeau."
                    ),
                    "position_fraction": 0.98,
                },
            ],
            instruction=(
                "One order in the long journal printed here is withdrawn by an "
                "order that comes after it and names it."
            ),
            rubric={
                "question": (
                    "After the withdrawal, which day, which time and which "
                    "foreman apply to the quarry blast schedule? Answer briefly."
                ),
                "required_all": [r"Tuesday", r"7:15", r"Nadeau|Piotr"],
                "forbidden_any": [r"Sunday", r"1:30", r"Iris|Alvarado"],
                "max_words": 90,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="B",
            difficulty="separating",
            context="the surviving order governs: Tuesday, 7:15 a.m., foreman Piotr Nadeau",
            lane="long-context-retrieval",
        ),
        _haystack_case(
            case_id="lc-64k-summary-B",
            seed=64103,
            target_tokens=65536,
            needles=[
                {
                    "text": (
                        "DECISION: the depot will standardise on the reusable "
                        "crate and retire the single-use liner by the end of the "
                        "quarter."
                    ),
                    "position_fraction": 0.05,
                },
                {
                    "text": (
                        "RISK: nobody has confirmed that the crate washer can "
                        "keep up with the extra volume, and that question is "
                        "still open."
                    ),
                    "position_fraction": 0.95,
                },
            ],
            instruction=(
                "The log printed here records one DECISION and one open RISK, "
                "placed far apart from each other."
            ),
            rubric={
                "question": (
                    "In exactly two sentences, state the decision and the "
                    "unresolved risk, preserving both."
                ),
                "required_all": [r"crate", r"washer|volume|keep up"],
                "forbidden_any": [],
                "max_words": 90,
                "case_sensitive": False,
            },
            ask="PRESERVE",
            step="B",
            difficulty="separating",
            context="decision: standardise on the reusable crate; risk: washer throughput unconfirmed",
            lane="long-context-summary",
        ),
        _haystack_case(
            case_id="lc-200k-notice-select-A",
            seed=200101,
            target_tokens=200000,
            needles=[
                {
                    "text": (
                        "NOTICE: the fuel bunker survey shifts to Saturday at "
                        "11:25 a.m. and every surveyor must report to the mill "
                        "office."
                    ),
                    "position_fraction": 0.10,
                },
                {
                    "text": "NOTICE: the survey ladders went back on their rack after use.",
                    "position_fraction": 0.50,
                },
                {
                    "text": "NOTICE: surveyors need a valid confined-space permit.",
                    "position_fraction": 0.90,
                },
            ],
            instruction=(
                "A very long journal appears beneath this line. Answer its "
                "closing question from its contents alone."
            ),
            rubric={
                "question": (
                    "Name the weekday, the hour and the reporting point for the "
                    "bunker survey. Under 80 words please."
                ),
                "required_all": [r"Saturday", r"11:25", r"mill office"],
                "forbidden_any": [],
                "max_words": 120,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="A",
            difficulty="reachable",
            context="day Saturday, time 11:25 a.m., report to the mill office",
            lane="long-context-retrieval",
        ),
        _haystack_case(
            case_id="lc-200k-notice-select-B",
            seed=200102,
            target_tokens=200000,
            needles=[
                {
                    "text": (
                        "MEMO: weighbridge calibration is booked for Monday at "
                        "5:50 p.m. with technician Halvard Bruun. THIS MEMO IS "
                        "REVOKED BY ONE PRINTED BENEATH IT."
                    ),
                    "position_fraction": 0.02,
                },
                {
                    "text": "MEMO: the visitor badge printer stays where it is.",
                    "position_fraction": 0.60,
                },
                {
                    "text": (
                        "MEMO: weighbridge calibration is booked for Friday at "
                        "2:35 p.m. with technician Ondine Marsh."
                    ),
                    "position_fraction": 0.98,
                },
            ],
            instruction=(
                "Somewhere inside the enormous journal quoted next, an earlier "
                "memo is revoked by a memo that identifies it."
            ),
            rubric={
                "question": (
                    "Give the surviving weighbridge booking: its weekday, its "
                    "hour and its technician, in under 60 words."
                ),
                "required_all": [r"Friday", r"2:35", r"Ondine|Marsh"],
                "forbidden_any": [r"Halvard|Bruun", r"5:50"],
                "max_words": 90,
                "case_sensitive": False,
            },
            ask="SELECT",
            step="B",
            difficulty="separating",
            context="the surviving memo governs: Friday, 2:35 p.m., technician Ondine Marsh",
            lane="long-context-retrieval",
        ),
    ]
    return cases


def code_cases() -> list[dict[str, Any]]:
    return [
        _case(
            "code-rle-encode-A",
            "code-produce",
            "coding",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Write a Python function named run_length that takes a "
                        "string and returns a list of (character, count) tuples "
                        "for each consecutive run. run_length('aaabbc') must "
                        "return [('a', 3), ('b', 2), ('c', 1)]. Do not import "
                        "any module. Give the function only."
                    ),
                }
            ],
            "a def named run_length, no imports, correct on the example",
            {
                "required_all": [r"def\s+run_length\s*\(", r"return"],
                # (?m) or the anchor only ever sees the first line, which let an
                # import on line two through the check written to forbid it.  And
                # the ask is "do not import ANY module", so naming `re` and
                # `itertools` was the check disagreeing with its own prompt --
                # `import collections` passed a case that forbids it.
                "forbidden_any": [
                    r"(?m)^[ \t]*import[ \t]+[A-Za-z_]",
                    r"(?m)^[ \t]*from[ \t]+[A-Za-z_][\w.]*[ \t]+import\b",
                ],
                "max_words": 400,
                "case_sensitive": True,
                "ast_function_name": "run_length",
            },
        ),
        _case(
            "code-bugfix-pagination-B",
            "code-bugfix",
            "coding",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "This helper is meant to return how many pages are "
                        "needed to show n items at page_size per page, and it "
                        "silently drops the last partial page.\n\n"
                        "def page_count(n, page_size):\n"
                        "    return n // page_size\n\n"
                        "Give the corrected function. Do not restate the buggy "
                        "line."
                    ),
                }
            ],
            "ceil division: (n + page_size - 1) // page_size, or math.ceil",
            {
                "required_all": [
                    r"def\s+page_count",
                    # the third alternative accepts the idiomatic ceiling
                    # `-(-n // page_size)`; without the optional paren it could
                    # only have matched `--n //`, which nobody writes.
                    r"(page_size\s*-\s*1\)\s*//\s*page_size)|(math\.ceil)"
                    r"|(-\s*\(?\s*-\s*n\s*//)",
                ],
                "forbidden_any": [],
                "max_words": 300,
                "case_sensitive": True,
                "ast_function_name": "page_count",
            },
        ),
        _case(
            "code-review-sql-injection-B",
            "code-review",
            "coding",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Review this database helper and say what is wrong with "
                        "it and how to fix it.\n\n"
                        "def find_user(conn, name):\n"
                        "    cur = conn.cursor()\n"
                        '    cur.execute("SELECT id FROM users WHERE name = \'" '
                        "+ name + \"'\")\n"
                        "    return cur.fetchone()\n"
                    ),
                }
            ],
            "SQL injection; fix is a parameterised query with a placeholder",
            {
                "required_all": [r"inject", r"%s|\?|placeholder|paramet"],
                "forbidden_any": [],
                "max_words": 400,
                "case_sensitive": False,
            },
        ),
        _case(
            "code-trace-recursion-A",
            "code-trace",
            "coding",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "def f(n):\n"
                        "    if n <= 1:\n"
                        "        return 1\n"
                        "    return n * f(n - 2)\n\n"
                        "What does f(7) return? Give the number and one line "
                        "showing how you got it."
                    ),
                }
            ],
            "7*5*3*1 = 105",
            {
                "required_all": [r"\b105\b"],
                "forbidden_any": [],
                "max_words": 200,
                "case_sensitive": False,
            },
        ),
        _case(
            "code-api-contract-B",
            "code-contract",
            "coding",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "A service returns {\"items\": [...], \"next\": <cursor "
                        "or null>}. Write a Python function fetch_all(client) "
                        "that returns every item across all pages by calling "
                        "client.get(cursor) until next is null. Handle the first "
                        "call, which takes cursor=None. No imports."
                    ),
                }
            ],
            "a loop that follows the cursor until next is None and extends a list",
            {
                "required_all": [
                    r"def\s+fetch_all",
                    r"while|for",
                    r"\[.{0,12}next.{0,12}\]|\.get\(",
                ],
                "forbidden_any": [],
                "max_words": 400,
                "case_sensitive": True,
                "ast_function_name": "fetch_all",
            },
        ),
    ]


def reasoning_cases() -> list[dict[str, Any]]:
    return [
        _case(
            "reason-arith-two-step-A",
            "reason-arithmetic",
            "reasoning",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Take thirteen groups of nine, then discard seventeen "
                        "units. Divide whatever remains into piles of eight. "
                        "Report three numbers: the subtotal before dividing, the "
                        "count of full piles, and the remainder."
                    ),
                }
            ],
            "13*9 = 117, minus 17 = 100; 100 / 8 = 12 piles remainder 4",
            {
                "required_all": [r"\b100\b", r"\b12\b", r"\b4\b"],
                "forbidden_any": [],
                "max_words": 250,
                "case_sensitive": False,
            },
        ),
        _case(
            "reason-schedule-constraints-B",
            "reason-constraints",
            "reasoning",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Three people present in three consecutive slots. Priya "
                        "cannot go last. Rafi must go immediately after Priya. "
                        "Elke is not first. Give the order, first to last, and "
                        "say why it is the only one."
                    ),
                }
            ],
            "Priya, Rafi, Elke -- forced, and the answer must say what forces it",
            {
                # The gap bounds are what make this case non-quotable: the
                # prompt names all three people, but 34 characters apart, so a
                # response that merely echoes the constraints cannot satisfy a
                # 25-character window while any real answer ("Priya, Rafi,
                # Elke", "1. Priya 2. Rafi 3. Elke") does.
                "required_all": [
                    r"priya.{0,25}rafi.{0,25}elke",
                    r"because|since|forces|leaves|therefore",
                ],
                "forbidden_any": [],
                "max_words": 250,
                "case_sensitive": False,
            },
        ),
        _case(
            "reason-uncertainty-churn-B",
            "reason-uncertainty",
            "epistemic-reasoning",
            "B",
            "separating",
            "SELECT",
            [
                {
                    "role": "user",
                    "content": (
                        "A team reports that customer churn 'went from 4% to 7%'. "
                        "They give no other figures. What would you need to know "
                        "before treating that as a real increase? Do not invent "
                        "a cause."
                    ),
                }
            ],
            "asks for the missing terms: window, definition, denominator, baseline, seasonality",
            {
                "required_all": [],
                "required_any_count": 2,
                "required_any": [
                    r"window|period|month|quarter",
                    r"definition|define|what counts",
                    r"denominator|base|cohort|sample size",
                    r"baseline|prior|historical",
                    r"season",
                ],
                "forbidden_any": [],
                "max_words": 300,
                "case_sensitive": False,
            },
        ),
        _case(
            "reason-unit-conversion-A",
            "reason-units",
            "reasoning",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "A pump moves 250 millilitres every 4 seconds. How many "
                        "litres does it move in half an hour? Give the number "
                        "and the unit."
                    ),
                }
            ],
            "250 ml / 4 s = 62.5 ml/s; 1800 s -> 112500 ml = 112.5 litres",
            {
                "required_all": [r"112\.5|112,5", r"lit"],
                "forbidden_any": [],
                "max_words": 200,
                "case_sensitive": False,
            },
        ),
        _case(
            "reason-causal-confound-B",
            "reason-causal",
            "reasoning",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Ice cream sales and pool rescues both peak in the same "
                        "weeks at one town. A councillor proposes limiting ice "
                        "cream sales to reduce rescues. Say in plain language "
                        "what is wrong with the reasoning and name the thing "
                        "that actually drives both."
                    ),
                }
            ],
            "correlation is not causation; the common cause is hot weather / summer",
            {
                "required_all": [r"weather|hot|heat|temperature|summer"],
                "required_any": [
                    r"correlat",
                    r"causa",
                    r"common cause|both.{0,20}driven|third",
                ],
                "required_any_count": 1,
                "forbidden_any": [],
                "max_words": 300,
                "case_sensitive": False,
            },
        ),
        _case(
            "reason-counterexample-B",
            "reason-counterexample",
            "reasoning",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Someone claims: 'if a number is divisible by 3 and by 4 "
                        "then it is divisible by 6, so any number divisible by 3 "
                        "and by 4 is also divisible by 8.' Give one concrete "
                        "number that breaks the second half of that claim and "
                        "say why."
                    ),
                }
            ],
            "12 (or 36, 60, 84...) is divisible by 3 and 4 but not by 8",
            {
                "required_all": [r"\b(12|36|60|84|108|132)\b"],
                "required_any": [r"not divisible by 8", r"8 does not", r"not.{0,20}\b8\b"],
                "required_any_count": 1,
                "forbidden_any": [],
                "max_words": 250,
                "case_sensitive": False,
            },
        ),
    ]


def multi_turn_cases() -> list[dict[str, Any]]:
    return [
        _case(
            "mt-venue-revision-A",
            "multi-turn-revision",
            "multi-turn-conversation",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Suggest a venue for a twenty-person team lunch near a "
                        "train station."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "A mid-size bistro with a private back room usually "
                        "works for twenty, and stations tend to have one within "
                        "a short walk."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Change it: it has to be step-free the whole way in, and "
                        "the back room is up a flight of stairs. Give me the "
                        "revised suggestion."
                    ),
                },
            ],
            "must honour step-free and must not re-propose the upstairs back room",
            {
                "required_all": [r"step-free|step free|level access|ground floor|no stairs"],
                "forbidden_any": [r"back room"],
                "max_words": 250,
                "case_sensitive": False,
            },
        ),
        _case(
            "mt-dietary-correction-B",
            "multi-turn-correction",
            "multi-turn-conversation",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Plan four snack options for an office afternoon. One "
                        "person has a shellfish allergy."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Four options that avoid shellfish: cheese and crackers, "
                        "hummus with vegetables, a fruit platter, and mini "
                        "quiches."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Two of the four also need to be vegan, and cross-contact "
                        "with shellfish matters. Redo the list of exactly four."
                    ),
                },
            ],
            "exactly four options, at least two clearly vegan, no shellfish item",
            {
                "required_all": [r"vegan"],
                "forbidden_any": [r"shrimp|prawn|crab|lobster|oyster|shellfish\s+(option|snack|platter)"],
                "distinct_numbered_items": 4,
                "max_words": 350,
                "case_sensitive": False,
            },
        ),
        _case(
            "mt-instruction-persistence-B",
            "multi-turn-persistence",
            "instruction-following",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "system",
                    "content": "Answer every message in exactly two sentences. Never more.",
                },
                {"role": "user", "content": "What is a checksum for?"},
                {
                    "role": "assistant",
                    "content": (
                        "A checksum is a short value computed from data so that a "
                        "later recomputation can show whether the data changed. It "
                        "detects accidental corruption rather than deliberate "
                        "tampering."
                    ),
                },
                {"role": "user", "content": "And why do people use a cryptographic hash instead?"},
            ],
            "graded on the LAST turn only: exactly two sentences, still on topic",
            {
                # NOT "hash": the user's own last turn says "cryptographic
                # hash", so requiring it would let an echo pass.  These are
                # words only an answer contains.
                "required_all": [
                    r"collision|tamper|adversar|deliberate|attacker|forge|reverse"
                ],
                "forbidden_any": [],
                "sentence_count": 2,
                "max_words": 90,
                "case_sensitive": False,
            },
        ),
        _case(
            "mt-context-carryover-B",
            "multi-turn-carryover",
            "multi-turn-conversation",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": "My server is called kestrel and it runs the billing job at 02:00.",
                },
                {
                    "role": "assistant",
                    "content": "Noted -- kestrel runs billing at 02:00.",
                },
                {"role": "user", "content": "I moved that job an hour earlier."},
                {
                    "role": "assistant",
                    "content": "Understood.",
                },
                {
                    "role": "user",
                    "content": "Write me the one-line cron entry for it, on that host.",
                },
            ],
            "01:00 -> `0 1 * * *`, and the host name carried through",
            {
                "required_all": [r"0\s+1\s+\*\s+\*\s+\*|01:00|1:00"],
                "forbidden_any": [r"0\s+2\s+\*\s+\*\s+\*"],
                "max_words": 200,
                "case_sensitive": False,
            },
        ),
    ]


def instruction_cases() -> list[dict[str, Any]]:
    return [
        _case(
            "instr-format-json-strict-A",
            "instruction-format",
            "instruction-following",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Reply with a single JSON object and nothing else. It "
                        "must have exactly the keys city, country and "
                        "population, for Reykjavik. Population may be "
                        "approximate."
                    ),
                }
            ],
            "one JSON object, exactly three named keys, no prose around it",
            {
                "required_all": [r'"city"', r'"country"', r'"population"'],
                "forbidden_any": [],
                "json_object_only": True,
                "max_words": 120,
                "case_sensitive": False,
            },
        ),
        _case(
            "instr-word-budget-A",
            "instruction-budget",
            "instruction-following",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "In 30 words or fewer, tell a new starter what a staging "
                        "environment is for. Do not use a list."
                    ),
                }
            ],
            "under the budget, and actually about staging",
            {
                # The first pattern is the topic and the prompt carries it; the
                # second is what an ANSWER carries and the prompt does not.
                "required_all": [
                    r"stag|pre-?prod",
                    r"before|prior|production|live|real|rehears|verif|test|deploy",
                ],
                "forbidden_any": [],
                # The ask is "in 30 words or fewer".  A budget check set at 40 is
                # a check that disagrees with its own prompt, and the reader
                # leaning on it would be told a 38-word answer honoured a 30-word
                # budget.
                "max_words": 30,
                "case_sensitive": False,
            },
        ),
        _case(
            "instr-negative-constraint-B",
            "instruction-negative",
            "instruction-following",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Explain what a queue does in a system, for a "
                        "non-technical reader, without ever using the words "
                        "queue, line or buffer."
                    ),
                }
            ],
            "a correct explanation that never uses the three banned words",
            {
                "required_all": [r"wait|hold|order|arriv|store|until"],
                "forbidden_any": [r"\bqueue", r"\bline\b", r"\bbuffer"],
                "max_words": 250,
                "case_sensitive": False,
            },
        ),
        _case(
            "instr-ordered-steps-B",
            "instruction-order",
            "instruction-following",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Give the steps to swap a live TLS certificate on a web "
                        "server without downtime, numbered, in the order they "
                        "must happen, including what you do if the new one is "
                        "rejected."
                    ),
                }
            ],
            "ordered steps with a reload-not-restart step and an explicit rollback",
            {
                "required_all": [r"reload|graceful", r"roll ?back|revert|restore"],
                "forbidden_any": [],
                "numbered_steps_min": 4,
                "max_words": 450,
                "case_sensitive": False,
            },
        ),
    ]


def knowledge_cases() -> list[dict[str, Any]]:
    return [
        _case(
            "know-explain-tradeoff-A",
            "knowledge-tradeoff",
            "knowledge-explanation",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Why would anyone choose to store a file twice on "
                        "purpose, on two different machines? Give the benefit "
                        "and the cost."
                    ),
                }
            ],
            "redundancy/availability against storage cost and consistency",
            {
                "required_all": [
                    r"redundan|backup|fail|availab|surviv",
                    r"cost|space|expens|sync|consisten|storage",
                ],
                "forbidden_any": [],
                "max_words": 300,
                "case_sensitive": False,
            },
        ),
        _case(
            "know-refuse-fabrication-B",
            "knowledge-epistemic",
            "epistemic-reasoning",
            "B",
            "separating",
            "SELECT",
            [
                {
                    "role": "user",
                    "content": (
                        "What was the exact closing attendance figure of the "
                        "Halvorsen Bridge reopening ceremony, and who read the "
                        "second speech?"
                    ),
                }
            ],
            "there is no such event to know; the answer must decline rather than invent",
            {
                "required_all": [
                    r"do not|don't|cannot|can't|no (record|reliable|information)|not aware|unable|no such"
                ],
                "forbidden_any": [r"\b\d{3,}\s*(people|attendees)\b"],
                "max_words": 200,
                "case_sensitive": False,
            },
        ),
    ]


# --------------------------------------------------------------------------
# The vision split (5 runnable) and the AV split (5 spec-only).
# --------------------------------------------------------------------------


def vision_cases(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    truth = {asset["asset_id"]: asset["ground_truth"] for asset in assets}
    specs = [
        (
            "vis-ocr-receipt-total-A",
            "vision-ocr",
            "A",
            "SELECT",
            "Read this receipt. What is the TOTAL and what is the DATE? Give both exactly as printed.",
            {
                "required_all": [re.escape(truth["vis-ocr-receipt-total-A"]["total"]),
                                 re.escape(truth["vis-ocr-receipt-total-A"]["date"])],
                "forbidden_any": [],
                "max_words": 80,
                "case_sensitive": False,
            },
        ),
        (
            "vis-count-shapes-B",
            "vision-count",
            "B",
            "SELECT",
            "How many BLUE squares are in this image? Answer with the number.",
            {
                "required_all": [r"\b4\b|\bfour\b"],
                "forbidden_any": [],
                "max_words": 60,
                "case_sensitive": False,
            },
        ),
        (
            "vis-chart-trend-B",
            "vision-chart",
            "B",
            "PRODUCE",
            "Describe the trend in this bar chart in one sentence and name the month with the tallest bar.",
            {
                "required_all": [r"increas|ris|grow|upward|up\b|higher"],
                "required_any": [r"\bMAY\b"],
                "required_any_count": 1,
                "forbidden_any": [],
                "max_words": 100,
                "case_sensitive": False,
            },
        ),
        (
            "vis-scene-relation-A",
            "vision-relation",
            "A",
            "PRODUCE",
            "Which shape is on the left of this image, and what colour is it?",
            {
                "required_all": [r"circle", r"blue"],
                "forbidden_any": [],
                "max_words": 60,
                "case_sensitive": False,
            },
        ),
        (
            "vis-grid-cell-select-A",
            "vision-select",
            "A",
            "SELECT",
            "One cell in this grid is green. Which letter is on it?",
            {
                "required_all": [r"\bE\b"],
                "forbidden_any": [],
                "max_words": 40,
                "case_sensitive": True,
            },
        ),
    ]
    cases: list[dict[str, Any]] = []
    for case_id, lane, step, ask, question, rubric in specs:
        cases.append(
            _case(
                case_id,
                lane,
                "vision",
                step,
                "reachable" if step == "A" else "separating",
                ask,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "asset_id": case_id},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                json.dumps(truth[case_id], sort_keys=True),
                rubric,
                modality_in="text+image",
                tokens_target=1024,
                extra={
                    "asset_id": case_id,
                    "vision_path_to_record": "native_vit | raw_patch",
                },
            )
        )
    return cases


AV_SCRIPTS = {
    "av-clip-what-said-and-shown-A": "The blue kettle is on the left.",
    "av-clip-what-said-and-shown-B": "Two speakers disagree about the delivery date.",
}


def av_cases() -> list[dict[str, Any]]:
    """FIVE SPEC-ONLY CASES.  `runnable: false` -- there are no assets yet.

    Two are AV-IN (a clip goes in, text comes out, machine-checkable against the
    known script) and three are MEDIA-OUT (a clip/image/track comes out and is
    graded ONLY by Cyril's blind 1-5 against the parent's artifact on the same
    prompt).  The build recipe is written into each case so the assets can be
    produced without re-deciding anything, and the licence blocker is named:
    H3 output is currently quarantined for training use, and using it as a
    REVIEW STIMULUS is a separate question that is Cyril's to answer.
    """

    spec_cases = [
        _case(
            "av-clip-what-said-and-shown-A",
            "av-in-comprehension",
            "audio-video",
            "A",
            "reachable",
            "SELECT",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "asset_id": "av-clip-kettle-4s"},
                        {
                            "type": "text",
                            "text": (
                                "What is said in this clip, and what object is "
                                "visible and on which side? Answer in one "
                                "sentence."
                            ),
                        },
                    ],
                }
            ],
            "script: 'The blue kettle is on the left.'  ASR reference = the script",
            {
                "required_all": [r"kettle", r"blue", r"left"],
                "forbidden_any": [],
                "max_words": 80,
                "case_sensitive": False,
            },
            runnable=False,
            modality_in="text+video+audio",
            tokens_target=2048,
            extra={
                "asset_build_recipe": {
                    "generator": "MiniMax H3 via ComfyUI on the AI server",
                    "duration_seconds": 4,
                    "script": AV_SCRIPTS["av-clip-what-said-and-shown-A"],
                    "frame_check": "object present on the left half of frame 1",
                    "asr_reference": "the script itself, verbatim",
                    "blocked_on": (
                        "H3 output is quarantined for TRAINING use; using it as a "
                        "review stimulus is a separate ruling and is Cyril's"
                    ),
                }
            },
        ),
        _case(
            "av-clip-what-said-and-shown-B",
            "av-in-comprehension",
            "audio-video",
            "B",
            "separating",
            "SELECT",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "asset_id": "av-clip-two-speakers-6s"},
                        {
                            "type": "text",
                            "text": (
                                "Two people speak in this clip. What do they "
                                "disagree about, and who speaks second?"
                            ),
                        },
                    ],
                }
            ],
            "script names a delivery date dispute; second speaker is identifiable by position",
            {
                "required_all": [r"deliver|date"],
                "forbidden_any": [],
                "max_words": 120,
                "case_sensitive": False,
            },
            runnable=False,
            modality_in="text+video+audio",
            tokens_target=2048,
            extra={
                "asset_build_recipe": {
                    "generator": "MiniMax H3 via ComfyUI on the AI server",
                    "duration_seconds": 6,
                    "script": AV_SCRIPTS["av-clip-what-said-and-shown-B"],
                    "blocked_on": "same H3 review-use ruling as the A case",
                }
            },
        ),
        _case(
            "av-gen-kettle-A",
            "media-out-video",
            "audio-video",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Generate a four-second clip of a blue kettle on a "
                        "kitchen counter, with a voice saying 'the blue kettle "
                        "is on the left'."
                    ),
                }
            ],
            "GRADED ONLY BY CYRIL, blind 1-5, beside the parent's artifact on the same prompt",
            {
                "graded_by": "cyril-blind-1-to-5",
                "machine_diagnostics_only": [
                    "non-empty file",
                    "duration seconds",
                    "frame count",
                    "audio RMS > 0",
                    "A/V sync estimate",
                    "sha256",
                    "wall-clock vs the LTX 2.3 baseline (331-361 s per 608x352 "
                    "6-step clip on the RTX 8000) and vs H3 turbo",
                ],
            },
            runnable=False,
            modality_out="video+audio",
            extra={"media_out": True},
        ),
        _case(
            "img-gen-shelf-B",
            "media-out-image",
            "image",
            "B",
            "separating",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Generate an image of three labelled jars on a wooden "
                        "shelf, the middle one taller than the others."
                    ),
                }
            ],
            "GRADED ONLY BY CYRIL, blind 1-5, beside the parent's artifact on the same prompt",
            {
                "graded_by": "cyril-blind-1-to-5",
                "machine_diagnostics_only": [
                    "non-empty file",
                    "resolution",
                    "sha256",
                    "wall-clock vs the Qwen-Image parent",
                ],
            },
            runnable=False,
            modality_out="image",
            extra={"media_out": True},
        ),
        _case(
            "music-gen-two-bar-A",
            "media-out-music",
            "music",
            "A",
            "reachable",
            "PRODUCE",
            [
                {
                    "role": "user",
                    "content": (
                        "Generate eight seconds of solo acoustic guitar, slow, "
                        "in a minor key, no vocals."
                    ),
                }
            ],
            "GRADED ONLY BY CYRIL, blind 1-5, beside the ACE-Step parent's artifact",
            {
                "graded_by": "cyril-blind-1-to-5",
                "machine_diagnostics_only": [
                    "non-empty file",
                    "duration seconds",
                    "sample rate",
                    "audio RMS > 0",
                    "sha256",
                    "wall-clock vs the ACE-Step parent",
                ],
            },
            runnable=False,
            modality_out="audio",
            extra={"media_out": True},
        ),
    ]
    return spec_cases


# --------------------------------------------------------------------------
# Rubric scoring (a diagnostic) and the self-test (a refusal).
# --------------------------------------------------------------------------


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+(?:\s|$)", text.strip()) if p.strip()]
    return len(parts)


#: A list marker, with whatever indentation precedes it captured.
_LIST_ITEM = re.compile(r"(?m)^([ \t]*)(?:\d+[.)]|[-*+])[ \t]+(.*)$")


def top_level_list_items(text: str) -> list[str]:
    """The list items a READER would count: the outermost depth, sub-bullets out.

    THE BUG THIS EXISTS TO KILL.  Both list checks used to be
    `(?m)^\\s*(?:\\d+[.)]|[-*])\\s+`, and `\\s*` after the anchor makes an
    INDENTED sub-bullet indistinguishable from a top-level item.  On
    `mt-dietary-correction-B` the 0.8B produced exactly the four snack options the
    case asks for, each with an indented `*Why:*` and `*Allergy Note:*` beneath
    it; the check counted nine distinct items and the case was published as a
    content failure.  It was a formatting artifact, and the artifact-tolerant lens
    forbids scoring one as a miss.  The same `\\s*` inflates `numbered_steps_min`
    in the other direction, where it produces a FALSE PASS -- one step with four
    sub-bullets satisfies "at least four steps".

    WHY DEPTH RATHER THAN COLUMN ZERO.  Anchoring hard at column 0 fixes the
    over-count and introduces the opposite false negative: a model that indents
    its whole list by two spaces would score zero items.  So the outermost
    indentation actually present is the depth that counts, with one space of
    jitter allowed, and anything deeper is a sub-item.  A response with no list at
    all has no items, which is the correct answer for the checks that use this.
    """

    matches = [
        (len(indent.expandtabs(4)), body.strip())
        for indent, body in _LIST_ITEM.findall(text)
    ]
    if not matches:
        return []
    outermost = min(indent for indent, _ in matches)
    return [body for indent, body in matches if indent <= outermost + 1]


def score_rubric(text: str, rubric: dict[str, Any]) -> dict[str, Any]:
    """Does the text carry what the case asked for?  A DIAGNOSTIC, NEVER A GRADE.

    Returns every individual check so a reader can see which one moved, because a
    single boolean would hide the difference between "wrong answer" and "right
    answer phrased differently", and only a reader can tell those apart.
    """

    if rubric.get("graded_by"):
        return {
            "applicable": False,
            "passed": None,
            "why": f"this case is graded by {rubric['graded_by']}, not by machine",
            "checks": [],
        }
    flags = 0 if rubric.get("case_sensitive") else re.IGNORECASE
    checks: list[dict[str, Any]] = []
    for pattern in rubric.get("required_all", []) or []:
        checks.append(
            {
                "kind": "required_all",
                "pattern": pattern,
                "ok": bool(re.search(pattern, text, flags)),
            }
        )
    any_patterns = rubric.get("required_any", []) or []
    if any_patterns:
        hits = [p for p in any_patterns if re.search(p, text, flags)]
        needed = int(rubric.get("required_any_count", 1))
        checks.append(
            {
                "kind": "required_any",
                "pattern": f"{needed} of {len(any_patterns)}",
                "matched": hits,
                "ok": len(hits) >= needed,
            }
        )
    for pattern in rubric.get("forbidden_any", []) or []:
        checks.append(
            {
                "kind": "forbidden_any",
                "pattern": pattern,
                "ok": not re.search(pattern, text, flags),
            }
        )
    words = len(text.split())
    if rubric.get("max_words") is not None:
        checks.append(
            {
                "kind": "max_words",
                "pattern": rubric["max_words"],
                "observed": words,
                "ok": words <= int(rubric["max_words"]),
            }
        )
    if rubric.get("min_words") is not None:
        checks.append(
            {
                "kind": "min_words",
                "pattern": rubric["min_words"],
                "observed": words,
                "ok": words >= int(rubric["min_words"]),
            }
        )
    if rubric.get("sentence_count") is not None:
        observed = _sentence_count(text)
        checks.append(
            {
                "kind": "sentence_count",
                "pattern": rubric["sentence_count"],
                "observed": observed,
                "ok": observed == int(rubric["sentence_count"]),
            }
        )
    if rubric.get("numbered_steps_min") is not None:
        observed = len(top_level_list_items(text))
        checks.append(
            {
                "kind": "numbered_steps_min",
                "pattern": rubric["numbered_steps_min"],
                "observed": observed,
                "ok": observed >= int(rubric["numbered_steps_min"]),
            }
        )
    if rubric.get("distinct_numbered_items") is not None:
        items = top_level_list_items(text)
        observed = len({i.strip().casefold() for i in items})
        checks.append(
            {
                "kind": "distinct_numbered_items",
                "pattern": rubric["distinct_numbered_items"],
                "observed": observed,
                "ok": observed == int(rubric["distinct_numbered_items"]),
            }
        )
    if rubric.get("json_object_only"):
        stripped = text.strip()
        ok = False
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                ok = True
            except json.JSONDecodeError:
                ok = False
        checks.append({"kind": "json_object_only", "pattern": True, "ok": ok})
    if rubric.get("ast_function_name"):
        name = str(rubric["ast_function_name"])
        ok = False
        detail = "no parsable def found"
        for candidate in _code_blocks(text):
            try:
                import ast

                tree = ast.parse(candidate)
            except SyntaxError as exc:
                detail = f"SyntaxError: {exc.msg}"
                continue
            if any(
                isinstance(node, ast.FunctionDef) and node.name == name
                for node in ast.walk(tree)
            ):
                ok, detail = True, "parsed, def present"
                break
        checks.append(
            {
                "kind": "ast_function_name",
                "pattern": name,
                "observed": detail,
                "ok": ok,
                "note": "PARSED, NEVER EXECUTED",
            }
        )
    return {
        "applicable": True,
        "passed": all(check["ok"] for check in checks),
        "why": "every check that fired is published; a machine pass is not a grade",
        "checks": checks,
    }


def _code_blocks(text: str) -> list[str]:
    """Fenced blocks first, then the whole text.  Parsed, never executed."""

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    blocks.append(text)
    return blocks


def ask_surface(case: dict[str, Any]) -> str:
    """The prompt with the supplied payload cut away.

    A SELECT or PRESERVE case supplies material that CONTAINS its answer -- that
    is what those asks mean -- so testing "is this case prompt-quotable" against
    the whole prompt would fail every one of them for the wrong reason.  The
    surface that must NOT satisfy the rubric is the instruction, and the payload
    anchor is what separates the two.
    """

    text = user_turns_text(case.get("messages", []))
    anchor = case.get("payload_anchor")
    if not anchor:
        return text
    start, end = (anchor + [None, None])[:2]
    head, _, rest = text.partition(str(start))
    if not rest:
        return text
    tail = ""
    if end:
        _, _, tail = rest.partition(str(end))
    return f"{head}\n{tail}"


def ngrams(text: str, n: int = 7) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.casefold())
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def comparable_prompt_text(case: dict[str, Any]) -> str:
    """The text the 7-gram rule is checked over, for EVERY case including the
    long-context ones.

    WHAT IS IN IT.  The prompt as the bank stores it -- for a long-context case
    that is the PLACEHOLDER form, because `<<HAYSTACK>>` is not expanded in the
    bank -- plus every needle string this file placed.  The needles are the part
    of the document that carries the answer, so they are part of the ask and a
    rung that reuses another rung's needle is a duplicate case, which is exactly
    what this rule exists to catch.

    WHAT IS OUT OF IT, AND WHY, STATED RATHER THAN HIDDEN.  The generated filler
    is excluded.  It is drawn on purpose from a twelve-adjective, twelve-noun,
    eight-place, eight-verb vocabulary in one fixed sentence template, so any two
    haystacks share thousands of 7-grams by pigeonhole and always will; a check
    run over materialised filler could only ever fail.  The filler is not an ask,
    it is the noise the ask is buried in, and the separate needle-vocabulary
    refusal above is what keeps a needle from being confusable with it.

    THIS EXCLUSION USED TO BE THE WHOLE CASE.  The long-context cases were dropped
    from the check entirely, by a filter with no comment, and under that filter
    three pairs of rungs carried BYTE-IDENTICAL prompt text and were still counted
    in "0 failures over 40 cases".  Now they are checked in placeholder form, and
    each rung was rewritten to ask its own question about its own needle.
    """

    text = user_turns_text(case.get("messages", []))
    spec = case.get("haystack_spec")
    if not spec:
        return text
    needles = "\n".join(str(needle["text"]) for needle in spec.get("needles", []))
    return f"{text}\n{needles}"


def self_test(cases: list[dict[str, Any]], *, v1_pack: Path | None) -> list[str]:
    """Every refusal the design rules imply, returned as a list of failures."""

    failures: list[str] = []
    seen: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen:
            failures.append(f"{case_id}: duplicate id")
        seen.add(case_id)
        rubric = case.get("rubric", {})
        if rubric.get("graded_by"):
            continue
        surface = ask_surface(case)
        verdict = score_rubric(surface, rubric)
        if verdict["applicable"] and verdict["passed"]:
            failures.append(
                f"{case_id}: PROMPT-QUOTABLE -- its own ask surface satisfies its "
                "rubric, so a model that echoes the instruction passes it"
            )
        for needle_source in (case.get("haystack_spec") or {}).get("needles", []):
            for word in re.findall(r"[a-z]{4,}", str(needle_source["text"]).casefold()):
                if word in _ADJECTIVES or word in _NOUNS or word in _VERBS:
                    failures.append(
                        f"{case_id}: needle word {word!r} is also filler "
                        "vocabulary; a needle must not be confusable with filler"
                    )
    prompt_texts = {case["id"]: comparable_prompt_text(case) for case in cases}
    grams = {cid: ngrams(text) for cid, text in prompt_texts.items()}
    ids = sorted(grams)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            shared = grams[left] & grams[right]
            if shared:
                failures.append(
                    f"{left} and {right} share {len(shared)} 7-gram(s), e.g. "
                    f"{' '.join(sorted(shared)[0])!r}"
                )
    if v1_pack is not None and Path(v1_pack).is_file():
        v1 = json.loads(Path(v1_pack).read_text(encoding="utf-8"))
        v1_grams: set[tuple[str, ...]] = set()
        for row in v1["prompts"]:
            v1_grams |= ngrams(user_turns_text(row.get("messages", [])))
        for cid, gram_set in grams.items():
            shared = gram_set & v1_grams
            if shared:
                failures.append(
                    f"{cid} shares {len(shared)} 7-gram(s) with the frozen v1 "
                    f"pack, e.g. {' '.join(sorted(shared)[0])!r}"
                )
    return failures


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------


def build_bank(assets_dir: Path) -> dict[str, Any]:
    assets = build_cards(assets_dir)
    text_cases = (
        long_context_cases()
        + code_cases()
        + reasoning_cases()
        + multi_turn_cases()
        + instruction_cases()
        + knowledge_cases()
    )
    if len(text_cases) != 30:
        raise CalibrationError(
            f"REFUSED: the draft is specified at 30 text cases and carries "
            f"{len(text_cases)}.  A count that drifts from the specification "
            "silently changes what a per-child column means."
        )
    vision = vision_cases(assets)
    av = av_cases()
    prompts = text_cases + vision + av
    body: dict[str, Any] = {
        "schema": BANK_SCHEMA,
        "status": DRAFT_STATUS,
        "purpose": (
            "The harder second review surface for the omni campaign: long "
            "context at 8k/32k/64k/200k, code, reasoning, multi-turn, "
            "instruction-following, a five-case vision split with exact "
            "generated ground truth, and a five-case audio-video split that is "
            "SPEC ONLY until its assets exist and their review use is ruled on."
        ),
        "authored_by": "claude-opus5-oc-calibration",
        "authored_for": "Cyril's review; nothing here is sealed",
        "counts": {
            "text_scored": len(text_cases),
            "vision_scored": len(vision),
            "av_spec_only": len(av),
            "runnable_now": sum(1 for c in prompts if c.get("runnable", True)),
            "total_entries": len(prompts),
        },
        "grading_rule": (
            "ONLY MEASURED INTELLIGENCE READ BY A HUMAN GRADES.  Every rubric in "
            "this file is a machine diagnostic that tells a reader where to look. "
            "An artifact-miss (empty, pure loop, indiscriminate copy, cap-ride "
            "with no unique content, missing EOS) counts NOTHING against "
            "intelligence; only a content-miss counts against.  No count in any "
            "artifact produced from this bank is a gate."
        ),
        "open_questions_for_cyril": [
            "No safety lane: the frozen v1 pack's safety-household-chemical stays "
            "the charged-not-scored lane.  Adding a second one changes what "
            "'charged' means and is your call.",
            "The 200k lanes need a runtime decision -- hf-layer-streamed or "
            "llama-server -- and the parent must be run on the SAME runtime or "
            "the two texts are not comparable.",
            "Media-out cases are graded only by your blind 1-5 against the "
            "parent's artifact; the machine columns beside them are diagnostics.",
            "H3 output as a REVIEW STIMULUS is a separate ruling from H3 output "
            "as training data; the AV assets are blocked on it.",
            "Whether bank v2 counts join the sealed 15-case verdict as a second "
            "instance (suffix -b2review) or stay a separate column entirely.",
        ],
        "assets": assets,
        "seven_gram_disjointness": {
            "covers": "all 40 cases, in the form the bank stores them",
            "text_compared": (
                "every user turn of the case, plus (for a long-context case) "
                "every needle string.  A long-context prompt is compared in "
                "PLACEHOLDER form because <<HAYSTACK>> is not expanded in the "
                "bank."
            ),
            "excluded_and_why": (
                "the generated filler.  It comes from a 12/12/8/8 vocabulary in "
                "one sentence template, so any two haystacks share thousands of "
                "7-grams by pigeonhole; a check over materialised filler could "
                "only ever fail.  The needle-vocabulary refusal is what keeps a "
                "needle from being confusable with the filler."
            ),
            "history": (
                "an earlier draft exempted every long-context case from this "
                "check with an uncommented filter.  Under that exemption three "
                "pairs of retrieval rungs carried byte-identical prompt text and "
                "the bank still reported '0 failures over 40 cases'.  The rungs "
                "were rewritten to be matched in shape and disjoint in content."
            ),
        },
        "haystack_generator": {
            "name": "oc-filler-v1",
            "definition": (
                "a 64-bit LCG over a fixed adjective/noun/place/verb vocabulary, "
                "one sentence per record, needles substituted at fractional "
                "positions.  Deterministic on every platform and every Python; "
                "each case pins the sha256 of the text its spec produces."
            ),
            "tokens_per_sentence_nominal": TOKENS_PER_SENTENCE,
            "true_token_count": "measured per run and written into the dossier",
        },
        "prompts": prompts,
    }
    body["bank_sha256"] = canonical_sha256(
        {k: v for k, v in body.items() if k != "bank_sha256"}
    )
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="bank JSON to write")
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="where to draw the vision cards (default: <out dir>/assets)",
    )
    parser.add_argument(
        "--v1-pack",
        default=None,
        help="the frozen v1 pack, for the cross-pack 7-gram disjointness check",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "print every design-rule failure, then STILL WRITE the bank if there "
            "are none -- so the documented 'the digest must not move' recipe "
            "actually re-authors the file it is compared against"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the design rules and write nothing at all",
    )
    args = parser.parse_args()

    out = Path(args.out)
    assets_dir = Path(args.assets_dir) if args.assets_dir else out.parent / "assets"
    body = build_bank(assets_dir)
    failures = self_test(
        body["prompts"], v1_pack=Path(args.v1_pack) if args.v1_pack else None
    )
    if args.self_test or args.check_only:
        for failure in failures:
            print(f"SELF-TEST FAILURE: {failure}")
        print(
            f"self-test: {len(failures)} failure(s) over "
            f"{len(body['prompts'])} cases"
        )
        if failures:
            return 1
        if args.check_only:
            print("--check-only: nothing was written")
            return 0
        # AND THEN IT WRITES.  `--self-test` used to return here, so the recipe
        # in README.md -- "re-author + self-test; the digest must not move" --
        # left the on-disk bank untouched and a reader comparing digests was
        # comparing the file against itself.  A reproducibility check that cannot
        # fail is not a check.
    if failures:
        for failure in failures:
            print(f"SELF-TEST FAILURE: {failure}", file=sys.stderr)
        raise SystemExit(
            "REFUSED: the bank does not pass its own design rules, and a bank "
            "that fails its self-test would measure copying rather than "
            "intelligence.  Nothing was written."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(body))
    temporary.replace(out)
    reread = json.loads(out.read_text(encoding="utf-8"))
    if reread["bank_sha256"] != body["bank_sha256"]:
        raise SystemExit("REFUSED: the bank did not survive its own write")
    digest = file_sha256(out)
    (out.parent / (out.name + ".sha256")).write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )
    print(f"{out}")
    print(f"  file sha256   {digest}")
    print(f"  bank_sha256   {body['bank_sha256']}")
    print(f"  counts        {json.dumps(body['counts'], sort_keys=True)}")
    print(f"  assets        {assets_dir}")
    print(f"  STATUS        {DRAFT_STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
