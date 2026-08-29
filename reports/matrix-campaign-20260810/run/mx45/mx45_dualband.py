#!/usr/bin/env python3
"""Wave 45: read the deep band TWICE under both readings of the copy-back
clause -- WITH THE ANCHOR TABLE COMPLETED AND ITS OUTCOME REGISTERED BEFORE ANY
WAVE-45 PASS RUNS.

THE DIFF AGAINST `mx44_dualband.py`, WHICH IS THE WHOLE OF WAVE 45'S CODE, IS
TWO CHANGES AND NEITHER TOUCHES A THRESHOLD.

(1) `PAYLOAD_ANCHORS` gains `writing-concise-email`.  This is the debt
`NOTES-MX-AV` section 2.3 measured, priced and correctly refused to pay in the
middle of its own measurement, and section 12 item 1 ordered paid first.  It is
paid at this tree cut.  **And its registered outcome is a NULL RESULT**: the
seven rows it was ordered to close read 0.477707 and 0.452229 under it, still
below 0.60, and `run/mx45/README.md` section 2 says so before the pass runs.
The mechanism wave 44 named -- "the copied material is the ask and the ask is
only about half of the prompt" -- is right about the arithmetic and wrong about
the repair: no cut of this prompt makes half of it into 0.60 of itself.

(2) `response_union_coverage` is added as a PUBLISHED column that DECIDES
NOTHING, on the same protocol waves 42 and 43 gave the union measure before
wave 44 promoted it.  It is the same union at the same minimum block against
the same numeral, with the RESPONSE as denominator.  On the two responses above
it reads 0.683824 and 0.635714 -- over 0.60, with no anchor at all -- which
locates the real defect in the DENOMINATOR rather than in the anchor table.
Nothing in this module lets it change a classification, and the cutter pins its
decision line at zero.

Everything below this paragraph is wave 44's header, unchanged, because wave 45
re-argues none of the promotion it records.

--- wave 44's header, kept ---

Wave 44: read the deep band TWICE under both readings of the copy-back
clause -- WITH THE UNION MEASURE PRE-REGISTERED AS PRIMARY BEFORE ANY WAVE-44
DECODE EXISTS.

THE DIFF AGAINST wave 43's dual-band reader, WHICH IS THE WHOLE OF WAVE 44'S CODE.
`payload_union_coverage` decides.  The longest-contiguous-run measure is still
computed, still published per lane, and decides nothing.

WHY, AND WHY NOW.  Wave 42 added the union number as a description, AFTER it
had seen its own results, and said in the artifact that letting it move a
classification would be tuning an instrument to an answer.  It was right and it
told the next wave to register it first.  Wave 43 did not, and the case it was
warned about is the case that happened: on wave 43's own floor mover
`ARCH.V5CL768`, the longest-run measure reads payload coverage 0.157761 against
a 0.60 threshold and calls the lane CLEAN, while the union measure reads
0.804071 and calls it a COPY.  The hand reading governed and column B was right,
but it was right because a human read it.  This file is the repair, and the
repair is only worth anything because it is committed at the tree cut, before
this wave has decoded a single token.

THE THRESHOLD AND WHERE IT COMES FROM.  `COPY_BACK_UNION_COVERAGE` is 0.60, and
it is set from `wave42/deep-band-dual-assessment.json` -- sha256
fe20e9164023fc9a639aa8e1c05617b28d7f4890fc27bc9bea65646d8780602c, committed at
5af0b035 two waves before this file existed -- and from nothing measured in wave
44.  That artifact carries 34 DISTINCT summarisation responses across its band.
Seventeen of them were hand-read as copies and their union coverage runs
**0.826972 to 0.997455**.  Every other distinct response in that lane sits at
**0.483461 or below**.  The separating interval is therefore (0.483461,
0.826972), it is 0.3435 wide, and NOTHING lies inside it.

0.60 is chosen inside that interval for two reasons that are both about not
tuning.  First, it is not an edge: it clears the highest non-copy by 0.1165 and
undercuts the lowest copy by 0.2270.  Second, it is the numeral the campaign
already declared for `COPY_BACK_PAYLOAD_COVERAGE`, so this wave changes the
MEASURE without also changing the NUMBER -- one change at a time, and a reader
comparing waves does not have to disentangle two moves.

**AND THE CHOICE IS DECLARED NON-LOAD-BEARING, WHICH IS CHECKABLE.**  Any
threshold in (0.483461, 0.826972) reproduces every one of those 34 calls
identically.  If a later wave wants to argue about 0.60 it has to first find a
response inside the gap, and there is not one.

WHAT IS RETROSPECTIVE AND WHAT IS PROSPECTIVE, LABELLED IN THE ARTIFACT.  Every
child sealed before wave 44 is re-scored by this measure and carries
``instrument_reading_is_retrospective: true``.  **That column re-grades
nothing.**  The sealed counts are untouched, column A is still the sealed
per-case verdict this module refuses to disagree with, and for those children
the HAND READING remains the authoritative call exactly as waves 42 and 43
sealed it.  The retrospective column exists to show what the instrument does on
a population whose answer is already known by other means.  Wave 44's own
children carry ``instrument_reading_is_retrospective: false`` and on those the
measure is the first call, taken by machine, before anybody reads.

THREE COLUMN-B VARIANTS ARE PUBLISHED, and the reason is that one of them is the
test of the instrument.

* ``column_b_copy_back_refused`` -- the estate's column B.  Hand reading governs
  a held lane where one exists; the union machine call governs where none does.
* ``column_b_machine_union_only`` -- the union measure alone, hand input
  ignored.  **If the instrument works, this reproduces the hand column.**
* ``column_b_machine_legacy_only`` -- the longest-run measure alone, hand input
  ignored.  This is what waves 42 and 43 would have published if nobody had
  read the dossiers, and it is kept so the size of the repair is a number.

Everything else in this module is wave 42's and wave 43's, unchanged: the lane
ask table, the payload anchors, the refusal that makes column A the sealed
column, the anomaly that fires when a non-PRESERVE lane is voided, and the rule
that only a HELD lane can be voided.

--- wave 43's header, kept ---

THE TWO COLUMNS.

* ``copy_back_counts`` -- WAVE 40'S PRACTICE.  A response that preserves the
  lane's asked-for facts holds the case whether it summarised them or copied
  them.  **This column is not computed.  It is the sealed per-case verdict,
  read off the share, and this module REFUSES to write its artifact if its own
  count of that file disagrees with the ledger row's sealed
  ``behavior_useful_or_partial``.**  A second column is worth nothing if the
  first one is this module's opinion rather than the estate's record.

* ``copy_back_refused`` -- THE CLAUSE AS WRITTEN.  A held lane whose response is
  the prompt's own payload returned back is VOIDED.  Nothing else moves: a lane
  sealed as a miss is never re-read and this module cannot turn a miss into a
  hold in either direction, so column B is column A minus a subtraction a reader
  can check line by line.

THE LANE-ASK TABLE IS THE LOAD-BEARING PART, NOT THE THRESHOLD.  Wave 40 sealed
a copied SUMMARISATION lane as HELD and a copied RETRIEVAL lane as a MISS, and
those two precedents are consistent for a reason wave 41 stated: the
summarisation lane's ask is to PRESERVE two named facts, which a copy does, and
the retrieval lane's ask is to SELECT three facts out of a note, which a copy
does not.  A copy can satisfy a PRESERVE ask and can satisfy neither a SELECT
ask nor a PRODUCE ask.  So the clause can only reach a PRESERVE lane, and if it
ever reaches another one that is an EXCEPTION to wave 40's own precedents and is
published as an anomaly rather than absorbed.

THE MEASURE IS DETERMINISTIC AND PUBLISHED.  For every lane the module computes
the longest common substring between the whitespace-normalised, case-folded
response and the same normalisation of that lane's PAYLOAD -- the part of the
prompt a copy would be a copy OF -- and reports it as a fraction of the payload
and as a fraction of the response.  The threshold is stated once, in
``COPY_BACK_PAYLOAD_COVERAGE``, and the raw fractions are published for every
lane so a reader who disagrees with the threshold can re-decide any lane from
the artifact without re-running anything.

AND THE MACHINE DOES NOT GET THE LAST WORD ON A HELD LANE.  ``--hand-reading``
takes a JSON file of per-lane adjudications written by the reviewer who read the
dossier text, and the artifact publishes machine and hand side by side plus an
explicit disagreement list.  A threshold is an instrument, not a verdict; the
estate's rule since wave 19 is that behaviour is graded by reading, and this
module keeps that rule by making the reading and the instrument separately
visible instead of letting one stand in for the other.

USAGE

    mx45_dualband.py --ledger <ledger.jsonl> --root <matrix root>
                     --max-ratio 0.47 --out <artifact.json>
                     [--hand-reading <hand.json>]

"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
import time
from pathlib import Path

SCHEMA = "neural-forge-matrix-wave45-deep-band-dual-assessment-v1"

#: WHAT THE FIRST PASS OF THIS MODULE MEASURED WRONG, RECORDED HERE RATHER THAN
#: FIXED SILENTLY.  ``COPY_BACK_PAYLOAD_COVERAGE`` is applied to the LONGEST
#: CONTIGUOUS common run, and every copy-back in this estate carries the
#: word-substitution signature -- a copy of the six-sentence source paragraph
#: with one to five words mutated.  Each mutation BREAKS the run.  The wave-42
#: machine pass therefore scored ARCH.V7CL768's held lane at 0.427 and
#: ARCH.RP9CL896X13's at 0.153 and called both CONTENT, and the hand reading of
#: all sixteen distinct held summarisation responses in the band found every one
#: of them to be the source paragraph returned in full.
#:
#: ``payload_union_coverage`` is the repair: the union of ALL common runs of at
#: least ``UNION_MIN_BLOCK`` characters, as a fraction of the payload.  On the
#: same lanes it reads 0.827 to 0.982 for the copies and 0.198 to 0.331 for a
#: genuine two-sentence summary of the same paragraph.
#:
#: **IN WAVE 44 IT DECIDES, AND THE MINIMUM BLOCK LENGTH IS NOT RETUNED.**  8 is
#: the value waves 42 and 43 published every union number at, and promoting a
#: measure while also retuning it would make the promotion unfalsifiable: the
#: separation quoted in this file's header is a separation AT THIS BLOCK LENGTH,
#: and moving it would invalidate the very numbers the threshold is set from.
UNION_MIN_BLOCK = 8

#: WAVE 45'S SECOND ADDITION, ON THE CAMPAIGN'S OWN PROMOTION PROTOCOL.  Waves
#: 42 and 43 published `payload_union_coverage` beside every lane while the
#: longest-run measure decided, and wave 44 promoted it on two waves of
#: published evidence rather than on an argument.  `response_union_coverage` --
#: the SAME union, at the SAME UNION_MIN_BLOCK, against the SAME 0.60 numeral,
#: with the RESPONSE as the denominator instead of the payload -- enters here on
#: exactly that protocol.
#:
#: WHY IT EXISTS.  The payload-side question is "how much of the supplied
#: material came back?", which is the right question for a lane that supplies
#: material.  On a PRODUCE lane there is no supplied material, the whole prompt
#: is instruction, and a model that hands back half the instruction scores under
#: 0.5 against the whole of it no matter where the anchor is cut.  The question
#: that lane needs is "how much of what this model SAID is lifted from the
#: prompt?", and that is a DENOMINATOR change, not an anchor change.  Wave 45
#: registered before its pass that the ordered anchor cannot close the seven
#: rows; this column is the measured alternative, published beside them and
#: DECIDING NOTHING.
RESPONSE_SIDE_DECIDES_NOTHING = True

#: The clause under adjudication, quoted from wave 40's canonical artifact and
#: re-quoted by wave 41.  It is never paraphrased in this file.
THE_CLAUSE = (
    "a copy-back is not a decode failure, it is the absence of a decode"
)

#: THE PRIMARY THRESHOLD, PRE-REGISTERED, STATED ONCE.  A response is a
#: copy-back when the UNION of all common runs of at least ``UNION_MIN_BLOCK``
#: characters between the lane's payload and the response -- both
#: whitespace-normalised and case-folded -- covers this fraction of the payload
#: or more.  The provenance of 0.60 is in this file's header: it is set from
#: wave 42's committed artifact, it sits inside a 0.3435-wide empty interval
#: between the copy band and everything else, and every threshold in that
#: interval gives identical calls.  The raw fraction is published for every
#: lane, so this number is a reported instrument setting and not a hidden
#: decision.
COPY_BACK_UNION_COVERAGE = 0.60

#: THE DEMOTED THRESHOLD, kept because the legacy measure is still computed and
#: still published.  It decides nothing.  Wave 42 registered it, wave 42 found
#: it under-reports on every copy carrying the word-substitution signature, and
#: wave 43 paid for using it anyway.  Deleting it would delete the evidence that
#: the demotion mattered, which is why the legacy column is published rather
#: than dropped.
COPY_BACK_PAYLOAD_COVERAGE = 0.60

#: What each lane ASKS FOR, which decides whether a copy can satisfy it.
#: PRESERVE -- the ask is to carry named facts through; a copy carries them.
#: SELECT   -- the ask is to pick out of supplied material or to withhold; a
#:             copy picks nothing.
#: PRODUCE  -- the ask is to generate something the prompt does not contain; a
#:             copy produces nothing.
#: Sixteen lanes, fifteen scored cases: `safety-household-chemical` is a lane on
#: the frozen surface and is charged rather than scored, and it is classified
#: here anyway so the table covers the surface it is read against.
LANE_ASK = {
    "conversation-natural-greeting": "PRODUCE",
    "instruction-three-actions": "PRODUCE",
    "explanation-seasons": "PRODUCE",
    "arithmetic-multi-step": "PRODUCE",
    "logic-ordering": "PRODUCE",
    "context-retrieval": "SELECT",
    "summarization-priorities": "PRESERVE",
    "comparison-ram-rom": "PRODUCE",
    "coding-deduplicate": "PRODUCE",
    "debugging-off-by-one": "PRODUCE",
    "writing-concise-email": "PRODUCE",
    "planning-dependencies": "PRODUCE",
    "uncertainty-insufficient-data": "SELECT",
    "safety-household-chemical": "PRODUCE",
    "creative-continuity": "PRODUCE",
    "multi-turn-correction": "PRODUCE",
}

#: Where a lane's PAYLOAD starts and stops inside its own prompt.  A payload is
#: the material a copy would be a copy OF; measuring against the whole prompt
#: would score the instruction text as copied content and inflate every lane.
#: The anchors are literal substrings of the frozen prompt pack, so a pack that
#: changed under this table would make the slice fail rather than silently move.
#: A lane with no entry is measured against the concatenation of its user turns.
PAYLOAD_ANCHORS = {
    "summarization-priorities": ("preserving the decision and the unresolved risk:", None),
    "context-retrieval": ("NOTE:", "Question:"),
    "debugging-off-by-one": ("def last_three", None),
    "writing-concise-email": ("under 90 words", None),
}

#: WAVE 45'S ANCHOR, AND THE RULE IT WAS CUT BY.  `NOTES-MX-AV` section 2.3
#: measured that ALL EIGHT void-changing machine-hand disagreements in the band
#: sit on lanes with no entry above, seven of them on `writing-concise-email`,
#: and section 12 item 1 ordered the entry registered BEFORE the pass that uses
#: it.  This is that entry and this is the commit.
#:
#: The cut point is the rule the other three anchors already follow -- strip the
#: leading TASK DIRECTIVE, keep the material the lane supplies -- applied to a
#: prompt whose directive is `Draft a polite email under 90 words`.  It is NOT
#: chosen to move a number, and `run/mx45/README.md` section 2 registers, before
#: the pass runs, that IT DOES NOT MOVE THE SEVEN ROWS: under this anchor the
#: two distinct committed responses read 0.477707 and 0.452229 against a 0.60
#: threshold, which is slightly BELOW where they already were against the whole
#: prompt.  The one cut that would move them -- keeping only the prompt's first
#: sentence -- works by deleting the exact sentence the model failed to copy,
#: and is refused in the README rather than quietly not tried.
PAYLOAD_ANCHOR_COUNT = 4

_MOD = (1 << 61) - 1
_BASE = 1_000_003


def normalise(text: str) -> str:
    """Whitespace-collapsed, case-folded.  Published as the measured surface."""

    return " ".join(text.split()).casefold()


def longest_common_substring_length(a: str, b: str) -> int:
    """Length of the longest common substring, by binary search on rolling hashes.

    Deterministic and O((n + m) log n).  The candidate returned by the hash
    search is verified against both strings before it is accepted, so a hash
    collision cannot inflate the answer -- it can only make the search fall back
    to a shorter length, which would understate a copy-back rather than invent
    one.
    """

    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    n = len(a)

    def hashes(s: str, length: int) -> dict[int, int]:
        """hash -> start offset, for every window of `length` in `s`."""

        out: dict[int, int] = {}
        if length == 0 or len(s) < length:
            return out
        # The leading character is dropped AFTER the multiply, so its weight is
        # B**length and not B**(length-1).  Getting that exponent wrong makes
        # every window past the first hash to a value no other string reaches,
        # which silently reports "no common substring" for inputs that share
        # one -- a copy-back would read as content.
        power = pow(_BASE, length, _MOD)
        acc = 0
        for i, ch in enumerate(s):
            acc = (acc * _BASE + ord(ch)) % _MOD
            if i >= length:
                acc = (acc - ord(s[i - length]) * power) % _MOD
            if i >= length - 1:
                out.setdefault(acc, i - length + 1)
        return out

    def common(length: int) -> bool:
        if length == 0:
            return True
        ha = hashes(a, length)
        if not ha:
            return False
        hb = hashes(b, length)
        for key, start_b in hb.items():
            start_a = ha.get(key)
            if start_a is None:
                continue
            if a[start_a:start_a + length] == b[start_b:start_b + length]:
                return True
        return False

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if common(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def union_coverage(payload: str, response: str) -> float:
    """Fraction of the payload covered by ALL common runs, not just the longest.

    THE PRIMARY MEASURE IN WAVE 44.  Unchanged in arithmetic from the function
    wave 42 added as a description and wave 43 published beside every lane; what
    changed is that ``classify`` now reads it.  Deterministic: ``difflib``'s
    matching blocks are a function of the two strings and of nothing else.
    """

    p, r = normalise(payload), normalise(response)
    if not p or not r:
        return 0.0
    matcher = difflib.SequenceMatcher(None, p, r, autojunk=False)
    covered = sum(b.size for b in matcher.get_matching_blocks()
                  if b.size >= UNION_MIN_BLOCK)
    return min(1.0, covered / len(p))


def lane_whole_prompt(prompt_messages: list) -> str:
    """Every user turn, concatenated, with NO anchor applied.

    WAVE 45.  This is the source the response-side column is measured against,
    and it is deliberately NOT the payload.  The payload-side question is "how
    much of the supplied material came back?", which is asked of the anchored
    slice.  The response-side question is "how much of what this model SAID is
    lifted from the prompt?", and the prompt is the WHOLE prompt: a model that
    hands back the instruction text has lifted it from somewhere, and anchoring
    the source would hide exactly the lift the column exists to see.
    """

    turns = [m.get("content", "") for m in prompt_messages
             if m.get("role", "user") == "user" or "role" not in m]
    return "\n".join(t for t in turns if t)


def lane_payload(lane_id: str, prompt_messages: list) -> str:
    """The material a copy of this lane would be a copy OF."""

    whole = lane_whole_prompt(prompt_messages)
    anchors = PAYLOAD_ANCHORS.get(lane_id)
    if anchors is None:
        return whole
    start_anchor, end_anchor = anchors
    index = whole.find(start_anchor)
    if index < 0:
        return ""
    cut = whole[index + len(start_anchor):] if end_anchor is None else whole[index + len(start_anchor):]
    if end_anchor is not None:
        stop = cut.find(end_anchor)
        if stop >= 0:
            cut = cut[:stop]
    if lane_id == "debugging-off-by-one":
        cut = start_anchor + cut
    return cut.strip()


def classify(response: str, payload: str, termination: str) -> tuple[str, float]:
    """THE PRIMARY CLASSIFIER.  (classification, payload_union_coverage).

    Pre-registered at the wave-44 tree cut, before any wave-44 decode existed.
    The union of all common runs of at least ``UNION_MIN_BLOCK`` characters, as
    a fraction of the payload, thresholded at ``COPY_BACK_UNION_COVERAGE``.

    The EMPTY and PAYLOAD_UNAVAILABLE branches are wave 42's and are unchanged:
    a lane whose payload could not be sliced is not scored as clean, because
    zero coverage and "measured and clean" have to stay distinguishable.
    """

    r = normalise(response)
    p = normalise(payload)
    if not r:
        return "EMPTY", 0.0
    if not p:
        return "PAYLOAD_UNAVAILABLE", 0.0
    union_cov = union_coverage(payload, response)
    if union_cov >= COPY_BACK_UNION_COVERAGE:
        return "COPY_BACK", union_cov
    if termination == "censored-at-cap":
        return "CAP_RIDE", union_cov
    return "CONTENT", union_cov


def legacy_classification(
    response: str, payload: str, termination: str
) -> tuple[str, float, float, int]:
    """WAVE 42'S DEMOTED MEASURE.  (classification, coverage, response_cov, lcs).

    Byte-for-byte the decision wave 42 registered and wave 43 used, kept so the
    two instruments can be differenced per lane instead of argued about.  It
    thresholds the LONGEST CONTIGUOUS common run, which every copy-back in this
    estate breaks by carrying one to five substituted words.  IT DECIDES
    NOTHING: nothing in this module reads its verdict except the artifact
    writer and the legacy column it feeds.
    """

    r = normalise(response)
    p = normalise(payload)
    if not r:
        return "EMPTY", 0.0, 0.0, 0
    if not p:
        return "PAYLOAD_UNAVAILABLE", 0.0, 0.0, 0
    lcs = longest_common_substring_length(r, p)
    legacy_cov = lcs / len(p)
    response_coverage = lcs / len(r)
    if legacy_cov >= COPY_BACK_PAYLOAD_COVERAGE:
        return "COPY_BACK", legacy_cov, response_coverage, lcs
    if termination == "censored-at-cap":
        return "CAP_RIDE", legacy_cov, response_coverage, lcs
    return "CONTENT", legacy_cov, response_coverage, lcs


def deep_band_rows(ledger: Path, max_ratio: float) -> list[dict]:
    """The graded ARCH children at or below `max_ratio`, one row each."""

    best: dict[str, dict] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cell = row.get("cell_id") or ""
        if not cell.startswith("ARCH."):
            continue
        ratio = row.get("stored_parameter_ratio")
        if ratio is None or ratio > max_ratio:
            continue
        if row.get("behavior_useful_or_partial") is None:
            continue
        # The graded row is the one with a verdict; if a cell has more than one,
        # the LAST one in file order is the estate's current record.
        best[cell] = row
    return sorted(best.values(), key=lambda r: r["stored_parameter_ratio"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-ratio", type=float, default=0.47)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand-reading", default="")
    args = ap.parse_args()

    if len(LANE_ASK) != 16:
        raise SystemExit(
            "REFUSED: the lane-ask table must cover all sixteen lanes of the "
            f"frozen surface; it carries {len(LANE_ASK)}.  A missing lane would "
            "fall out of the second column silently, which is a floor move by "
            "omission"
        )
    if len(PAYLOAD_ANCHORS) != 4:
        raise SystemExit(
            "REFUSED: the payload-anchor table must carry exactly the four "
            f"entries wave 45 registered at its tree cut; it carries "
            f"{len(PAYLOAD_ANCHORS)}.  An anchor added or removed between the "
            "registration and the pass re-slices a lane's payload, which moves "
            "its coverage, which can void or un-void a held lane -- a floor "
            "move made by an edit nobody declared.  The count is asserted HERE, "
            "where the table is used, and not only in the cutter, because the "
            "cutter runs once and this runs every pass"
        )
    for lane, anchors in PAYLOAD_ANCHORS.items():
        if lane not in LANE_ASK:
            raise SystemExit(
                f"REFUSED: payload anchor {lane!r} names a lane the frozen "
                "surface does not have"
            )
    for lane, lane_ask in LANE_ASK.items():
        if lane_ask not in ("PRESERVE", "SELECT", "PRODUCE"):
            raise SystemExit(
                f"REFUSED: lane {lane} carries ask class {lane_ask!r}, which is "
                "not one of PRESERVE, SELECT or PRODUCE"
            )

    root = Path(args.root)
    hand: dict[str, dict[str, str]] = {}
    if args.hand_reading:
        hand = json.loads(Path(args.hand_reading).read_text(encoding="utf-8"))

    children: list[dict] = []
    anomalies: list[str] = []
    disagreements: list[dict] = []
    instrument_moves: list[dict] = []
    missing: list[str] = []

    for row in deep_band_rows(Path(args.ledger), args.max_ratio):
        cell = row["cell_id"]
        tag = cell.split(".", 1)[1]
        wave = row["campaign_wave"]
        wdir = root / f"wave{wave}"
        percase_path = wdir / "cells" / cell / "per-case.json"
        dossier_path = wdir / "human-review" / f"dossier-{tag}.json"
        if not percase_path.is_file() or not dossier_path.is_file():
            missing.append(f"{cell} (wave {wave}): "
                           f"per-case={'ok' if percase_path.is_file() else 'MISSING'} "
                           f"dossier={'ok' if dossier_path.is_file() else 'MISSING'}")
            continue
        percase = json.loads(percase_path.read_text(encoding="utf-8"))
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
        responses = {r["id"]: r for r in dossier["child_responses"]}

        sealed_count = row["behavior_useful_or_partial"]
        column_a = sum(1 for v in percase.values() if v in ("Useful", "Partial"))
        if column_a != int(sealed_count):
            raise SystemExit(
                f"REFUSED: {cell}'s per-case file counts {column_a} held cases "
                f"and the ledger's sealed behavior_useful_or_partial is "
                f"{sealed_count}.  Column A is the SEALED column, not this "
                "module's recount of it, and a disagreement here means the two "
                "artifacts are not describing the same child"
            )

        # WAVE 44.  A child sealed before this wave is scored by an instrument
        # that did not exist when it was graded, so its reading is labelled
        # RETROSPECTIVE and it re-grades nothing.  A wave-44 child is scored
        # PROSPECTIVELY: the threshold was committed at the tree cut, before the
        # child was built, let alone decoded.
        retrospective = int(wave) < 44

        lanes: list[dict] = []
        voided: list[str] = []
        voided_union_only: list[str] = []
        voided_legacy_only: list[str] = []
        for lane_id in LANE_ASK:
            lane_ask = LANE_ASK[lane_id]
            resp = responses.get(lane_id)
            if resp is None:
                anomalies.append(f"{cell}: dossier has no lane {lane_id}")
                continue
            payload = lane_payload(lane_id, resp.get("prompt_messages", []))
            if not payload:
                anomalies.append(
                    f"{cell}: lane {lane_id} payload anchor did not resolve; "
                    "the lane is reported PAYLOAD_UNAVAILABLE and is NOT scored "
                    "as clean"
                )
            verbatim = resp.get("response_verbatim", "")
            termination = resp.get("termination", "")
            classification, union_cov = classify(verbatim, payload, termination)
            legacy = legacy_classification(verbatim, payload, termination)
            response_union = union_coverage(
                verbatim, lane_whole_prompt(resp.get("prompt_messages", [])))
            legacy_call, pay_cov, resp_cov, lcs = legacy
            sealed_verdict = percase.get(lane_id, "not scored")
            hand_call = hand.get(cell, {}).get(lane_id, "")
            entry = {
                "lane": lane_id,
                "lane_ask": lane_ask,
                "sealed_verdict": sealed_verdict,
                "response_sha256_16": resp.get("response_sha256", "")[:16],
                "generated_tokens": resp.get("generated_tokens"),
                "termination": resp.get("termination"),
                "payload_chars": len(normalise(payload)),
                "longest_common_run_chars": lcs,
                "payload_coverage": round(pay_cov, 6),
                "response_coverage": round(resp_cov, 6),
                "payload_union_coverage": round(union_cov, 6),
                "response_union_coverage": round(response_union, 6),
                "response_side_decides_nothing": RESPONSE_SIDE_DECIDES_NOTHING,
                "machine_classification": classification,
                "machine_measure": "payload_union_coverage (PRIMARY, pre-registered)",
                "legacy_longest_run_classification": legacy_call,
                "instrument_reading_is_retrospective": retrospective,
                "hand_reading": hand_call,
                "held_under_copy_back_counts": sealed_verdict in ("Useful", "Partial"),
            }
            held_under_refusal = sealed_verdict in ("Useful", "Partial")
            if sealed_verdict in ("Useful", "Partial"):
                # Only a HELD lane can be voided.  A miss is never re-read and
                # this module never turns a miss into a hold.
                call = hand_call or classification
                if call == "COPY_BACK":
                    held_under_refusal = False
                    voided.append(lane_id)
                    if classification == "COPY_BACK" and lane_ask != "PRESERVE":
                        anomalies.append(
                            f"{cell}: lane {lane_id} is a copy-back on a "
                            f"{lane_ask} lane and was nevertheless sealed held. "
                            "Wave 40's two precedents say a copy can satisfy a "
                            "PRESERVE ask and nothing else, so this is an "
                            "EXCEPTION to them and is published as one"
                        )
                # THE TWO MACHINE-ONLY COLUMNS.  Neither reads `hand`, and that
                # is the point: `column_b_machine_union_only` is what this wave's
                # instrument says with nobody in the loop, and comparing it with
                # the hand-governed column is the only test of the instrument
                # that does not use the instrument to grade itself.
                if classification == "COPY_BACK":
                    voided_union_only.append(lane_id)
                if legacy_call == "COPY_BACK":
                    voided_legacy_only.append(lane_id)
                if hand_call and hand_call != classification:
                    disagreements.append({
                        "cell_id": cell, "lane": lane_id,
                        "machine": classification, "hand": hand_call,
                        "payload_union_coverage": round(union_cov, 6),
                        "legacy_longest_run_classification": legacy_call,
                        "payload_coverage": round(pay_cov, 6),
                        "response_sha256_16": resp.get("response_sha256", "")[:16],
                    })
                if legacy_call != classification:
                    instrument_moves.append({
                        "cell_id": cell, "lane": lane_id,
                        "campaign_wave": wave,
                        "retrospective": retrospective,
                        "legacy_longest_run": legacy_call,
                        "union_primary": classification,
                        "payload_coverage": round(pay_cov, 6),
                        "payload_union_coverage": round(union_cov, 6),
                        "hand_reading": hand_call,
                        "response_sha256_16": resp.get("response_sha256", "")[:16],
                    })
            entry["held_under_copy_back_refused"] = held_under_refusal
            lanes.append(entry)

        column_b = column_a - len(voided)
        children.append({
            "cell_id": cell,
            "campaign_wave": wave,
            "instrument_reading_is_retrospective": retrospective,
            "stored_parameter_count": row["stored_parameter_count"],
            "stored_parameter_ratio": row["stored_parameter_ratio"],
            "column_a_copy_back_counts": column_a,
            "column_b_copy_back_refused": column_b,
            "column_b_machine_union_only": column_a - len(voided_union_only),
            "column_b_machine_legacy_only": column_a - len(voided_legacy_only),
            "voided_lanes": voided,
            "voided_lanes_machine_union_only": voided_union_only,
            "voided_lanes_machine_legacy_only": voided_legacy_only,
            "copy_back_lane_count": sum(
                1 for lane in lanes if lane["machine_classification"] == "COPY_BACK"),
            "cap_ride_lane_count": sum(
                1 for lane in lanes if lane["machine_classification"] == "CAP_RIDE"),
            "content_lane_count": sum(
                1 for lane in lanes if lane["machine_classification"] == "CONTENT"),
            "lanes": lanes,
        })

    def floor(column: str) -> dict:
        alive = [c for c in children if c[column] >= 1]
        if not alive:
            return {"any_child_holds_a_case": False,
                    "floor_ratio": None, "floor_cell_id": None}
        deepest = min(alive, key=lambda c: c["stored_parameter_ratio"])
        return {
            "any_child_holds_a_case": True,
            "floor_ratio": deepest["stored_parameter_ratio"],
            "floor_cell_id": deepest["cell_id"],
            "floor_stored_parameters": deepest["stored_parameter_count"],
            "floor_held_cases": deepest[column],
        }

    artifact = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": (
            "every graded ARCH child at or below the stated ratio, read TWICE: "
            "once under wave 40's practice, in which a response that preserves "
            "the lane's asked-for facts holds the case whether it summarised "
            "them or copied them, and once under the canonical clause as "
            "written, in which a lane that returns the prompt's own payload is "
            "not credited.  Column A is the SEALED per-case verdict copied off "
            "the share and refused if it disagrees with the ledger; column B is "
            "column A minus the held lanes that are copy-backs.  NOTHING HERE "
            "RE-GRADES A SEALED CHILD.  The ruling picks a column."
        ),
        "the_clause_under_adjudication": THE_CLAUSE,
        "instrument": {
            "primary_measure": "payload_union_coverage",
            "primary_measure_definition": (
                "the union of ALL common runs of at least "
                f"{UNION_MIN_BLOCK} characters between the whitespace-"
                "normalised, case-folded response and the same normalisation "
                "of the lane's payload, as a fraction of the payload"
            ),
            "copy_back_threshold_union_coverage": COPY_BACK_UNION_COVERAGE,
            "pre_registered": (
                "YES.  Declared in mx45_dualband.py at the wave-44 tree cut, "
                "committed BEFORE any wave-44 child was built and therefore "
                "before any wave-44 token was decoded.  NOTES-MX-AT section "
                "11.3 asked wave 43 for this and wave 43 did not do it; "
                "NOTES-MX-AU section 13 item 3 asked again."
            ),
            "threshold_provenance": (
                "wave42/deep-band-dual-assessment.json, sha256 fe20e916"
                "4023fc9a639aa8e1c05617b28d7f4890fc27bc9bea65646d8780602c, "
                "committed at 5af0b035.  Across the 34 DISTINCT "
                "summarisation responses in that band, the seventeen "
                "hand-read copies run 0.826972 to 0.997455 union coverage and "
                "every other distinct response sits at 0.483461 or below.  "
                "Nothing measured in wave 44 was used to choose this number."
            ),
            "threshold_invariance_interval": [0.483461, 0.826972],
            "threshold_invariance_statement": (
                "ANY threshold strictly inside (0.483461, 0.826972) reproduces "
                "all 34 of those calls identically, so the specific value 0.60 "
                "is not load-bearing.  It is 0.1165 above the highest non-copy "
                "and 0.2270 below the lowest copy, and it is the numeral the "
                "campaign already used for the demoted measure, so this wave "
                "changes the MEASURE without also changing the NUMBER."
            ),
            "demoted_measure": "payload_coverage (longest contiguous common run)",
            "demoted_threshold": COPY_BACK_PAYLOAD_COVERAGE,
            "demoted_measure_still_computed": True,
            "why_the_demotion": (
                "the longest-run measure thresholds a SINGLE contiguous run, "
                "and every copy-back in this estate carries the "
                "word-substitution signature -- the six-sentence source "
                "paragraph with one to five words mutated -- so each mutation "
                "BREAKS the run.  On wave 43's own floor mover ARCH.V5CL768 it "
                "reads 0.157761 and calls the lane clean where the union "
                "measure reads 0.804071 and calls it a copy.  The hand reading "
                "governed and column B was right, but it was right because a "
                "human read it."
            ),
            "retrospective_versus_prospective": (
                "Every child with campaign_wave < 44 carries "
                "instrument_reading_is_retrospective: true.  Its sealed count, "
                "its per-case verdict and the HAND reading waves 42 and 43 "
                "recorded are all authoritative and untouched; the union call "
                "on those lanes is a mechanical re-score published as a "
                "labelled column and it RE-GRADES NOTHING.  Wave 44's own "
                "children carry false, and on those the measure is the first "
                "call, taken before anybody read the dossier."
            ),
        },
        "copy_back_threshold_payload_coverage": COPY_BACK_PAYLOAD_COVERAGE,
        "measure": (
            "PRIMARY: payload_union_coverage, thresholded at "
            f"{COPY_BACK_UNION_COVERAGE}.  PUBLISHED AND DEMOTED: the longest "
            "common substring between the whitespace-normalised, case-folded "
            "response and the same normalisation of the lane's payload, "
            "reported as a fraction of the payload and of the response.  All "
            "three fractions are published per lane so any threshold on either "
            "measure can be re-applied from this artifact without re-running "
            "anything."
        ),
        "measure_defect_found_by_the_hand_pass": (
            "THE MEASURE WAVE 42 PRE-REGISTERED UNDER-REPORTS, AND WAVE 44 HAS "
            "REPLACED IT RATHER THAN DESCRIBED IT AGAIN.  It used the LONGEST "
            "CONTIGUOUS common run, and every copy-back in this estate carries "
            "the word-substitution signature -- the six-sentence source "
            "paragraph with one to five words mutated -- and each mutation "
            "BREAKS the run.  Wave 42's machine pass scored ARCH.V7CL768's held "
            "lane at 0.427 and ARCH.RP9CL896X13's at 0.153 and called both "
            "CONTENT; the hand reading of every distinct held summarisation "
            "response in that band found all of them to be the source paragraph "
            "returned in full.  Waves 42 and 43 both published the union number "
            "beside the lane and let the longest-run number decide, because in "
            "wave 42 it had been added after the results were seen and moving a "
            "classification with it would have been tuning an instrument to an "
            "answer.  That objection does not apply to a threshold committed at "
            "a tree cut against an artifact two waves old, which is what "
            "COPY_BACK_UNION_COVERAGE is."
        ),
        "union_measure_minimum_block_characters": UNION_MIN_BLOCK,
        "lane_ask_table": LANE_ASK,
        "payload_anchors": {k: list(v) for k, v in PAYLOAD_ANCHORS.items()},
        "max_ratio": args.max_ratio,
        "children_assessed": len(children),
        "children_assessed_prospectively": sum(
            1 for c in children if not c["instrument_reading_is_retrospective"]),
        "children_assessed_retrospectively": sum(
            1 for c in children if c["instrument_reading_is_retrospective"]),
        "floor_under_copy_back_counts": floor("column_a_copy_back_counts"),
        "floor_under_copy_back_refused": floor("column_b_copy_back_refused"),
        "floor_under_copy_back_refused_machine_union_only": floor(
            "column_b_machine_union_only"),
        "floor_under_copy_back_refused_machine_legacy_only": floor(
            "column_b_machine_legacy_only"),
        "instrument_moved_these_held_lanes": instrument_moves,
        "instrument_moved_held_lane_count": len(instrument_moves),
        "machine_hand_disagreements": disagreements,
        "anomalies": anomalies,
        "children_without_artifacts": missing,
        "children": children,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8", newline="")
    # NOTES-MX-AO section 9: parse the artifact in the command that writes it.
    reread = json.loads(out.read_text(encoding="utf-8"))
    if reread["schema"] != SCHEMA or len(reread["children"]) != len(children):
        raise SystemExit("REFUSED: the artifact did not survive its own write")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    print(f"deep band <= {args.max_ratio}: {len(children)} graded children")
    print(f"  column A (copy-back counts)  floor: "
          f"{artifact['floor_under_copy_back_counts']}")
    print(f"  column B (copy-back refused) floor: "
          f"{artifact['floor_under_copy_back_refused']}")
    print(f"  column B, MACHINE UNION ONLY (no hand input) floor: "
          f"{artifact['floor_under_copy_back_refused_machine_union_only']}")
    print(f"  column B, MACHINE LEGACY ONLY (no hand input) floor: "
          f"{artifact['floor_under_copy_back_refused_machine_legacy_only']}")
    print(f"  held lanes where the union measure and the legacy measure "
          f"disagree: {len(instrument_moves)}")
    print(f"  anomalies: {len(anomalies)}  disagreements: {len(disagreements)}  "
          f"missing artifacts: {len(missing)}")
    for line in missing:
        print(f"    MISSING {line}")
    for line in anomalies:
        print(f"    ANOMALY {line}")
    print(f"-> {out} sha256 {digest} (re-parsed OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
