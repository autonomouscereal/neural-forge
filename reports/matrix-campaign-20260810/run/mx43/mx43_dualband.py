#!/usr/bin/env python3
"""Wave 42: read the deep band TWICE, under both readings of the copy-back
clause, and publish the two columns side by side with the digests that make
every call checkable.

WHY THIS FILE EXISTS.  `NOTES-MX-AS` §4.1 recorded a contradiction and refused
to resolve it.  The canonical clause is quoted from wave 40's own canonical
artifact -- *a copy-back is not a decode failure, it is the absence of a decode*
-- and applied to wave 41's ten children it reads every one of them 0/15,
because every held case in that wave is the summarisation lane and every one of
those lanes is the source paragraph returned in full.  Applied to wave 40's
sealed floor child ARCH.V9CL896X13 it gives the same answer, and wave 40
published canonical ONE.  The two children's held lanes are byte-identical
(`111e342d9cb5438c`).  So either the clause does not mean what it says, or the
canonical 1/15 rung has been zero since wave 39.

That is an adjudication.  No wave can settle it by building a child, and wave 41
was right not to re-grade a sealed count to settle it.  What a wave CAN do is
make BOTH readings complete over the whole deep band, so that whichever way the
ruling goes, nothing needs re-grading and the floor is already computed under
the reading that wins.  This module does that and nothing else.  It builds
nothing, it solves nothing, it opens no model and it never writes into a sealed
artifact.

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

    mx43_dualband.py --ledger <ledger.jsonl> --root <matrix root>
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

SCHEMA = "neural-forge-matrix-wave43-deep-band-dual-assessment-v1"

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
#: **IT DECIDES NOTHING IN THIS WAVE AND IT IS NOT A THRESHOLD.**  It was added
#: after the results were seen, so letting it move a classification would be
#: tuning an instrument to an answer.  ``classify`` is untouched and the
#: pre-registered longest-run measure still produces every
#: ``machine_classification`` in the artifact; the union number is published
#: beside it as a description, and the HAND READING is what voids a held lane.
#: The next wave should pre-register the union measure as primary.
UNION_MIN_BLOCK = 8

#: The clause under adjudication, quoted from wave 40's canonical artifact and
#: re-quoted by wave 41.  It is never paraphrased in this file.
THE_CLAUSE = (
    "a copy-back is not a decode failure, it is the absence of a decode"
)

#: The threshold, stated ONCE.  A response is a copy-back when a single
#: contiguous run of the lane's payload, whitespace-normalised and case-folded,
#: survives in it at this fraction or above.  0.60 is chosen so that the
#: campaign's two named precedents both land where wave 40 sealed them: the
#: summarisation copy (the whole six-sentence paragraph, one word mutated)
#: scores far above it, and a response that merely re-uses the prompt's nouns
#: scores far below it.  The raw fraction is published for every lane, so this
#: number is a reported instrument setting and not a hidden decision.
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
}

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

    DESCRIPTIVE ONLY in wave 42.  See ``UNION_MIN_BLOCK``'s note: this was added
    after the pre-registered longest-run measure was found to under-report a
    copy that carries the word-substitution signature, so it is published and it
    changes no classification.
    """

    p, r = normalise(payload), normalise(response)
    if not p or not r:
        return 0.0
    matcher = difflib.SequenceMatcher(None, p, r, autojunk=False)
    covered = sum(b.size for b in matcher.get_matching_blocks()
                  if b.size >= UNION_MIN_BLOCK)
    return min(1.0, covered / len(p))


def lane_payload(lane_id: str, prompt_messages: list) -> str:
    """The material a copy of this lane would be a copy OF."""

    turns = [m.get("content", "") for m in prompt_messages
             if m.get("role", "user") == "user" or "role" not in m]
    whole = "\n".join(t for t in turns if t)
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


def classify(response: str, payload: str, termination: str) -> tuple[str, float, float, int]:
    """(classification, payload_coverage, response_coverage, lcs_chars)."""

    r = normalise(response)
    p = normalise(payload)
    if not r:
        return "EMPTY", 0.0, 0.0, 0
    if not p:
        # A lane whose payload could not be sliced is NOT scored as clean.  Zero
        # coverage and "measured and clean" have to be distinguishable.
        return "PAYLOAD_UNAVAILABLE", 0.0, 0.0, 0
    lcs = longest_common_substring_length(r, p)
    payload_coverage = lcs / len(p)
    response_coverage = lcs / len(r)
    if payload_coverage >= COPY_BACK_PAYLOAD_COVERAGE:
        return "COPY_BACK", payload_coverage, response_coverage, lcs
    if termination == "censored-at-cap":
        return "CAP_RIDE", payload_coverage, response_coverage, lcs
    return "CONTENT", payload_coverage, response_coverage, lcs


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

        lanes: list[dict] = []
        voided: list[str] = []
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
            classification, pay_cov, resp_cov, lcs = classify(
                resp.get("response_verbatim", ""), payload,
                resp.get("termination", ""))
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
                "payload_union_coverage": round(
                    union_coverage(payload, resp.get("response_verbatim", "")), 6),
                "machine_classification": classification,
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
                if hand_call and hand_call != classification:
                    disagreements.append({
                        "cell_id": cell, "lane": lane_id,
                        "machine": classification, "hand": hand_call,
                        "payload_coverage": round(pay_cov, 6),
                        "response_sha256_16": resp.get("response_sha256", "")[:16],
                    })
            entry["held_under_copy_back_refused"] = held_under_refusal
            lanes.append(entry)

        column_b = column_a - len(voided)
        children.append({
            "cell_id": cell,
            "campaign_wave": wave,
            "stored_parameter_count": row["stored_parameter_count"],
            "stored_parameter_ratio": row["stored_parameter_ratio"],
            "column_a_copy_back_counts": column_a,
            "column_b_copy_back_refused": column_b,
            "voided_lanes": voided,
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
        "copy_back_threshold_payload_coverage": COPY_BACK_PAYLOAD_COVERAGE,
        "measure": (
            "longest common substring between the whitespace-normalised, "
            "case-folded response and the same normalisation of the lane's "
            "payload, reported as a fraction of the payload and of the "
            "response.  Both fractions are published per lane so any threshold "
            "can be re-applied from this artifact without re-running anything."
        ),
        "measure_defect_found_by_the_hand_pass": (
            "THE PRE-REGISTERED MEASURE UNDER-REPORTS.  It uses the LONGEST "
            "CONTIGUOUS common run, and every copy-back in this estate carries "
            "the word-substitution signature -- the six-sentence source "
            "paragraph with one to five words mutated -- and each mutation "
            "BREAKS the run.  The machine pass scored ARCH.V7CL768's held lane "
            "at 0.427 and ARCH.RP9CL896X13's at 0.153 and called both CONTENT; "
            "the hand reading of all sixteen distinct held summarisation "
            "responses in this band found every one of them to be the source "
            "paragraph returned in full.  payload_union_coverage is the repair "
            "-- the union of all common runs of at least 8 characters -- and on "
            "the same lanes it reads 0.827 to 0.982 for the copies against "
            "0.198 to 0.331 for a genuine two-sentence summary of the same "
            "paragraph.  IT DECIDES NOTHING HERE: it was added after the "
            "results were seen, so letting it move a classification would be "
            "tuning an instrument to an answer.  Every machine_classification "
            "in this artifact is still the pre-registered longest-run measure's, "
            "and the HAND READING is what voids a held lane.  The next wave "
            "should pre-register the union measure as primary."
        ),
        "union_measure_minimum_block_characters": UNION_MIN_BLOCK,
        "lane_ask_table": LANE_ASK,
        "payload_anchors": {k: list(v) for k, v in PAYLOAD_ANCHORS.items()},
        "max_ratio": args.max_ratio,
        "children_assessed": len(children),
        "floor_under_copy_back_counts": floor("column_a_copy_back_counts"),
        "floor_under_copy_back_refused": floor("column_b_copy_back_refused"),
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
