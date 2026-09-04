#!/usr/bin/env python3
"""Both dual-band columns and Cyril's scoring sheet, from one dossier.

WHAT THIS FILE IS ALLOWED TO DO.  It reads verbatim text somebody else generated,
puts each lane in a FORM BOX, and publishes two columns.  It never grades.  The
estate's rule since wave 19 is that behaviour is graded by reading; the canonical
lens (`MC/ACTUAL_MATRIX_2026-08-28.md`) adds that only a reader can tell an
artifact-miss from a content-miss.  So:

  * With `--per-case`, this behaves exactly as `mx45_dualband.py` does.
    **Column A is the READING, not this module's opinion of it** -- the held
    count is counted off the per-case file and, when a ledger row is supplied,
    REFUSED if the two disagree.  Column B is column A minus the held lanes the
    copy measure calls ECHO.  A miss is never re-read and this module cannot turn
    a miss into a hold in either direction.

  * Without `--per-case` -- which is the state of every parent on its first pass,
    because Cyril has not scored it yet -- the two columns are published as
    **UPPER BOUNDS** and are labelled `is_upper_bound: true` in the artifact and
    in the sheet.  A bound is what a machine can honestly say: a lane that
    emitted nothing, or looped, or rode the cap without saying anything cannot be
    holding a case, and every other lane MIGHT be.  The gap between the bound and
    the truth is exactly the reading, and the sheet exists to collect it.

    **AN ECHO LANE STAYS IN THE COLUMN-A BOUND.**  It was excluded once, and that
    was a bug this file no longer has: a bound that a legal reading can exceed is
    not a bound.  Column A read 11/15 on Q35-0.8B while four ECHO lanes were
    listed as voided, so `A == B` while the subtraction was printed beside it,
    and a reading that held all fifteen lanes -- which the per-case branch below
    explicitly PERMITS, publishing an anomaly rather than refusing -- produced 15,
    four above the published "upper bound".  The instrument's own caution says an
    ECHO call on a PRODUCE lane may be the LFM2 threshold failing to transfer, so
    the instrument may not treat that call as proof of a miss.  The subtraction
    now means the same thing in both branches: **column B = column A minus the
    lanes in `voided_lanes`.**

THE THREE-BOX CANONICAL COLUMN is filled in as far as it can be filled in and no
further.  Sealed Useful/Partial -> `signal`.  A sealed miss on an artifact box ->
`artifact-miss` (counts NOTHING; the lens is explicit that loops, echo, cap-rides
and missing EOS never count against intelligence).  A sealed miss on a CONTENT
box -> `content-miss` (the only box that counts against).  With no per-case file
every lane reads `READ-REQUIRED`, which is the truth.

THRESHOLDS.  `COPY_BACK_UNION_COVERAGE = 0.60` at `UNION_MIN_BLOCK = 8` are the
LFM2 campaign's, unretuned, so the omni columns are comparable with the LFM2
columns.  Their provenance and their invariance interval (0.483461, 0.826972) are
published in the artifact.  Every number here is a PROPOSED DECISION LINE for
Cyril and none of them is a gate.

USAGE

  oc_assess.py --dossier oc-dossier-Q35-4B.json \
               --bank <the bank the dossier names> \
               --out-json  assessment-Q35-4B.json \
               --out-markdown SCORING-SHEET-Q35-4B.md \
               [--per-case per-case-Q35-4B.json] [--parent-dossier ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oc_calib import (  # noqa: E402
    CalibrationError,
    COPY_BACK_UNION_COVERAGE,
    COPY_THRESHOLD_INVARIANCE_INTERVAL,
    UNION_MIN_BLOCK,
    canonical_sha256,
    classify_form,
    file_sha256,
    normalise,
    union_coverage,
    user_turns_text,
    write_canonical,
)

SCHEMA = "neural-forge-omni-calibration-dual-band-assessment-v1"

VERDICTS = ("Useful", "Partial", "Observed miss", "Inconclusive/censored")

#: The frozen v1 surface's sixteen lanes and what each one ASKS FOR, copied from
#: `mx45_dualband.LANE_ASK` with no edit.  A copy can satisfy a PRESERVE ask and
#: can satisfy neither a SELECT nor a PRODUCE ask; that table, not the threshold,
#: is the load-bearing part of column B.
LANE_ASK_V1 = {
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

#: `mx45_dualband.PAYLOAD_ANCHORS`, all four, unchanged.  The count is asserted
#: where the table is USED and not only at a tree cut, because an anchor added or
#: removed re-slices a payload, which moves a coverage, which can void or un-void
#: a held lane -- a column move made by an edit nobody declared.
PAYLOAD_ANCHORS_V1 = {
    "summarization-priorities": ("preserving the decision and the unresolved risk:", None),
    "context-retrieval": ("NOTE:", "Question:"),
    "debugging-off-by-one": ("def last_three", None),
    "writing-concise-email": ("under 90 words", None),
}

#: The one lane on the frozen surface that is CHARGED and never SCORED.
SAFETY_LANE_V1 = "safety-household-chemical"

#: The form boxes a machine can PROVE are not holding a case, and therefore the
#: only ones a column-A UPPER BOUND may subtract.  Nothing was emitted, or
#: nothing NEW was, or the cap arrived before any content did -- none of those
#: can carry an answer whatever a reader thinks of them.
#:
#: ECHO IS DELIBERATELY ABSENT.  Wave 40 says a copy satisfies neither a SELECT
#: nor a PRODUCE ask, but that is a statement about a copy, and "this lane IS a
#: copy" is exactly the call the 0.60 threshold makes and the instrument's own
#: caution says may not transfer.  A bound may only subtract what the instrument
#: is certain of; the ECHO subtraction is column B, published beside it.
MACHINE_REFUTED_BOXES = ("EMPTY", "LOOP", "CAP_RIDE")


def lane_payload(lane_id: str, prompt_messages: list, anchors: Any) -> str:
    """The material a copy of this lane would be a copy OF.  mx45_dualband."""

    whole = user_turns_text(prompt_messages)
    if not anchors:
        return whole
    start_anchor, end_anchor = (list(anchors) + [None, None])[:2]
    index = whole.find(str(start_anchor))
    if index < 0:
        return ""
    cut = whole[index + len(str(start_anchor)) :]
    if end_anchor is not None:
        stop = cut.find(str(end_anchor))
        if stop >= 0:
            cut = cut[:stop]
    if lane_id == "debugging-off-by-one":
        cut = str(start_anchor) + cut
    return cut.strip()


def bank_index(bank_path: Path | None) -> dict[str, dict[str, Any]]:
    if bank_path is None:
        return {}
    payload = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload.get("prompts", [])}


def assess(
    dossier: dict[str, Any],
    *,
    bank: dict[str, dict[str, Any]],
    per_case: dict[str, str] | None,
    sealed_useful_or_partial: int | None,
) -> dict[str, Any]:
    if len(LANE_ASK_V1) != 16:
        raise CalibrationError(
            "REFUSED: the v1 lane-ask table must cover all sixteen lanes of the "
            f"frozen surface; it carries {len(LANE_ASK_V1)}."
        )
    if len(PAYLOAD_ANCHORS_V1) != 4:
        raise CalibrationError(
            "REFUSED: the v1 payload-anchor table must carry exactly the four "
            f"entries the campaign registered; it carries {len(PAYLOAD_ANCHORS_V1)}."
        )

    responses = dossier["child_responses"]
    known = {str(record["id"]) for record in responses}
    if per_case is not None:
        unknown = set(per_case) - known
        if unknown:
            raise CalibrationError(
                f"the per-case file names lanes this dossier has no response "
                f"for: {sorted(unknown)}"
            )
        bad = {k: v for k, v in per_case.items() if v not in VERDICTS}
        if bad:
            raise CalibrationError(
                f"per-case verdicts must be one of {VERDICTS}; got {bad}"
            )

    lanes: list[dict[str, Any]] = []
    anomalies: list[str] = []
    cautions: list[dict[str, Any]] = []
    for record in responses:
        lane_id = str(record["id"])
        case = bank.get(lane_id, {})
        ask = str(case.get("ask") or LANE_ASK_V1.get(lane_id) or "PRODUCE")
        anchors = case.get("payload_anchor") or PAYLOAD_ANCHORS_V1.get(lane_id)
        payload = lane_payload(lane_id, record.get("prompt_messages", []), anchors)
        if anchors and not payload:
            anomalies.append(
                f"{lane_id}: the payload anchor did not resolve; the lane reads "
                "PAYLOAD_UNAVAILABLE and is NOT treated as clean"
            )
        verbatim = record.get("response_verbatim", "")
        form = classify_form(
            response=verbatim,
            payload=payload,
            termination=str(record.get("termination", "")),
            token_ids=None,
            lane_ask=ask,
        )
        # The dossier records the loop flag from the token ids, which this file
        # does not have; the flag is authoritative and the text-side ratio is a
        # second opinion published beside it.
        if record.get("loop_detected") and form["form_box"] == "CONTENT":
            if form["unique_word_ratio"] < 0.35:
                form["form_box"] = "LOOP"
                form["is_artifact_box"] = True
                form["form_reason"] = "the dossier recorded a token-level loop"
            else:
                form["form_reason"] += "; a token-level loop was recorded after content"
        form["loop_detected"] = bool(record.get("loop_detected"))
        sealed = None if per_case is None else per_case.get(lane_id)
        held = sealed in ("Useful", "Partial")
        # THE MEASURE'S OWN PRECEDENT, APPLIED TO ITSELF.  Wave 40 established
        # that a copy can satisfy a PRESERVE ask and NEITHER a SELECT nor a
        # PRODUCE one.  So an ECHO call on a PRODUCE lane is one of two things
        # and the instrument cannot tell which: the model really did hand the
        # prompt back, or the 0.60 threshold -- calibrated on a 230M parent that
        # could not restate anything at length -- is over-reporting on a parent
        # that legitimately quotes the material it was asked to work on (a
        # debugging lane's correct answer CONTAINS the function it fixes).
        # It is published as a caution with its number so a reader decides, and
        # the threshold is NOT retuned here: tuning an instrument to the answer
        # it just produced is the exact failure wave 42 refused to commit.
        if form["form_box"] == "ECHO" and ask != "PRESERVE":
            cautions.append(
                {
                    "lane": lane_id,
                    "lane_ask": ask,
                    "payload_union_coverage": form["payload_union_coverage"],
                    "reading": (
                        "an ECHO call on a non-PRESERVE lane.  Either the model "
                        "returned the payload, or the LFM2-calibrated 0.60 "
                        "threshold does not transfer to this parent on this "
                        "lane.  READ THE TEXT; the threshold was not moved."
                    ),
                }
            )
        if form["form_box"] == "ECHO" and ask != "PRESERVE" and held:
            anomalies.append(
                f"{lane_id}: a copy-back on a {ask} lane was nevertheless read as "
                f"{sealed}.  Wave 40's precedents say a copy can satisfy a "
                "PRESERVE ask and nothing else, so this is an EXCEPTION and is "
                "published as one rather than absorbed"
            )
        rubric_result = None
        if case.get("rubric"):
            from oc_bank_v2_draft import score_rubric

            rubric_result = score_rubric(verbatim, case["rubric"])
            # A RUBRIC VERDICT ON A TRUNCATED RESPONSE IS NOT A VERDICT ON AN
            # ANSWER.  `max_words` passes vacuously when the cap arrived first,
            # and a required literal can match inside a chain of working that
            # never reached a conclusion.  Say so beside the verdict.
            if rubric_result.get("applicable") and str(
                record.get("termination")
            ) == "censored-at-cap":
                rubric_result["response_was_truncated"] = (
                    "this response hit the token cap.  A max_words check passes "
                    "vacuously here and a required literal may have matched "
                    "inside unfinished working, so read the text before you "
                    "believe the cell."
                )
        canonical_box = "READ-REQUIRED"
        if sealed in ("Useful", "Partial"):
            canonical_box = "signal"
        elif sealed in ("Observed miss", "Inconclusive/censored"):
            canonical_box = "artifact-miss" if form["is_artifact_box"] else "content-miss"
        lanes.append(
            {
                "lane": lane_id,
                "lane_ask": ask,
                "category": record.get("category"),
                #: THE BANK'S OWN STATEMENT OF WHAT THE CASE ASKED FOR, carried
                #: through to the sheet.  The frozen v1 pack's fifteen cases are
                #: described in the V5 protocol, which a reader has in front of
                #: them; bank v2's forty are NEW and its long-context cases hide
                #: a pinned needle inside ten thousand tokens of filler.  A sheet
                #: that prints the haystack and the answer but not the needle
                #: asks Cyril to re-derive ground truth he already pinned.  It is
                #: an ANNOTATION -- the bank's claim about its own case, never a
                #: key, never a grade, and it decides nothing.
                "bank_expectation": case.get("context"),
                "is_safety_lane": lane_id == SAFETY_LANE_V1,
                "scored": lane_id != SAFETY_LANE_V1,
                "sealed_verdict": sealed or "unscored",
                "canonical_box": canonical_box,
                "response_sha256_16": str(record.get("response_sha256", ""))[:16],
                "prompt_tokens": record.get("prompt_tokens"),
                "generated_tokens": record.get("generated_tokens"),
                "termination": record.get("termination"),
                "payload_chars": len(normalise(payload)),
                "payload_available": bool(payload) if anchors else True,
                "response_union_coverage": round(
                    union_coverage(verbatim, user_turns_text(record.get("prompt_messages", []))),
                    6,
                ),
                "response_side_decides_nothing": True,
                "machine_rubric": rubric_result,
                **form,
            }
        )

    scored = [lane for lane in lanes if lane["scored"]]
    if per_case is None:
        # THE BOUND SUBTRACTS ONLY WHAT IT CAN PROVE.  See MACHINE_REFUTED_BOXES:
        # an ECHO lane stays in A and is removed by B, so `voided_lanes` is the
        # same subtraction here as it is under a reading and the two printed
        # numbers are arithmetically consistent with the list beside them.
        column_a = sum(
            1 for lane in scored if lane["form_box"] not in MACHINE_REFUTED_BOXES
        )
        voided: list[str] = [
            lane["lane"] for lane in scored if lane["form_box"] == "ECHO"
        ]
        column_b = column_a - len(voided)
        bound = True
    else:
        # A HALF-FINISHED READING IS NOT A READING.  An unnamed lane counted
        # silently as not-held, which turns "Cyril has not read this one yet"
        # into "Cyril read it and it missed" -- a miss this module invented.
        # mx45 cross-checks its reading against a ledger row; the omni analogue
        # of that mandatory check is this refusal.
        unread = [
            lane["lane"] for lane in scored if lane["lane"] not in per_case
        ]
        if unread:
            raise CalibrationError(
                f"REFUSED: the per-case reading does not name {len(unread)} of "
                f"the {len(scored)} scored lanes: {sorted(unread)}.  Column A IS "
                "the reading, and an unnamed lane would be counted as not-held, "
                "which is a miss this module would have invented.  Finish the "
                "sheet, or run without --per-case for the upper-bound form."
            )
        column_a = sum(1 for lane in scored if lane["sealed_verdict"] in ("Useful", "Partial"))
        if sealed_useful_or_partial is not None and column_a != int(
            sealed_useful_or_partial
        ):
            raise CalibrationError(
                f"REFUSED: the per-case file counts {column_a} held cases and the "
                f"supplied sealed count is {sealed_useful_or_partial}.  Column A "
                "is the READING, not this module's recount of it, and a "
                "disagreement means the two artifacts are not about the same run."
            )
        voided = [
            lane["lane"]
            for lane in scored
            if lane["sealed_verdict"] in ("Useful", "Partial")
            and lane["form_box"] == "ECHO"
        ]
        column_b = column_a - len(voided)
        bound = False

    if column_b != column_a - len(voided):
        raise CalibrationError(
            f"REFUSED: column B ({column_b}) is not column A ({column_a}) minus "
            f"the {len(voided)} voided lane(s).  The two columns and the list of "
            "voided lanes are printed side by side and must be the same "
            "arithmetic in both branches."
        )

    artifact = {
        "schema": SCHEMA,
        "what_this_is": (
            "one model's review surface read twice: column A credits a lane that "
            "carries the asked-for facts however it got them, column B refuses a "
            "lane that returned the prompt's own payload.  NOTHING HERE GRADES.  "
            "With a per-case reading attached, column A IS that reading.  Without "
            "one, BOTH COLUMNS ARE UPPER BOUNDS and are labelled as such: a "
            "machine can prove a lane is NOT holding a case (it emitted nothing, "
            "or looped, or rode the cap saying nothing) and can never prove that "
            "it is.  An ECHO lane stays inside the column-A bound because 'this "
            "is a copy' is a THRESHOLD CALL and not a certainty; column B is "
            "column A minus exactly the lanes listed in voided_lanes, in both "
            "branches."
        ),
        "label": dossier.get("label"),
        "parent_code": dossier.get("parent_code"),
        "stage": dossier.get("stage"),
        "is_random_init": bool(dossier.get("is_random_init")),
        "dossier_sha256": dossier.get("dossier_sha256"),
        "lane_digest": dossier.get("lane_digest"),
        "lane_digest_validity": dossier.get("lane_digest_validity"),
        "bank_file_sha256": dossier.get("bank", {}).get("file_sha256"),
        "generation": dossier.get("generation"),
        "emit_zero_floor_passed": dossier.get("emit_zero_floor", {}).get("passed"),
        "decode_path_agree": dossier.get("decode_path_preflight", {}).get("agree"),
        "decode_smoke_verdict": dossier.get("decode_smoke", {}).get("verdict"),
        "instrument": {
            "primary_measure": "payload_union_coverage",
            "copy_back_threshold_union_coverage": COPY_BACK_UNION_COVERAGE,
            "union_measure_minimum_block_characters": UNION_MIN_BLOCK,
            "threshold_invariance_interval": list(COPY_THRESHOLD_INVARIANCE_INTERVAL),
            "threshold_provenance": (
                "the LFM2 campaign's, unretuned: wave42/deep-band-dual-"
                "assessment.json sha256 fe20e9164023fc9a639aa8e1c05617b28d7f4890"
                "fc27bc9bea65646d8780602c.  Carrying it unchanged is what makes "
                "an omni column comparable with an LFM2 column.  Whether it "
                "TRANSFERS to a 4B-class parent is unmeasured and is a proposed "
                "decision line, not a finding."
            ),
            "form_box_definitions": {
                "EMPTY": "no non-special text at all",
                "LOOP": "the tail is a repeated cycle and the text is repetition",
                "ECHO": "the response is the prompt's own payload returned",
                "CAP_RIDE": "hit the cap with almost no unique words",
                "MISSING_EOS": "reported as a flag beside the box, never as the box",
                "CONTENT": "readable original text -- A READER MUST NOW READ IT",
            },
            "artifact_boxes_count_nothing": (
                "the canonical artifact-tolerant lens: loops, echo, cap-rides and "
                "missing EOS never count against intelligence.  Only a "
                "content-miss counts against."
            ),
        },
        "columns": {
            "column_a_copy_back_counts": column_a,
            "column_b_copy_back_refused": column_b,
            "is_upper_bound": bound,
            "denominator_scored_lanes": len(scored),
            "voided_lanes": voided,
            "column_b_is_column_a_minus_voided_lanes": True,
            "upper_bound_subtracts_only": list(MACHINE_REFUTED_BOXES),
            "meaning": (
                "column A = echo persistence; column B = the machine-verified "
                "intelligence floor.  On LFM2 the post-pivot memory is 'column B "
                "is the intelligence floor'.  Which column governs the omni "
                "campaign is a PROPOSED DECISION for Cyril; both are published."
            ),
        },
        "canonical_three_box": {
            "signal": sum(1 for lane in scored if lane["canonical_box"] == "signal"),
            "artifact_miss": sum(
                1 for lane in scored if lane["canonical_box"] == "artifact-miss"
            ),
            "content_miss": sum(
                1 for lane in scored if lane["canonical_box"] == "content-miss"
            ),
            "read_required": sum(
                1 for lane in scored if lane["canonical_box"] == "READ-REQUIRED"
            ),
            "rule": (
                "signal / 15 is the canonical intelligence count.  A machine "
                "cannot fill the signal box; it can only say which misses are "
                "artifacts once a reader has called them misses."
            ),
        },
        "safety_lane": {
            "lane": SAFETY_LANE_V1,
            "present": any(lane["lane"] == SAFETY_LANE_V1 for lane in lanes),
            "rule": "charged, never scored; its reading is free text",
        },
        "anomalies": anomalies,
        "instrument_cautions": cautions,
        "instrument_caution_count": len(cautions),
        "lanes": lanes,
    }
    artifact["assessment_sha256"] = canonical_sha256(artifact)
    return artifact


def markdown_sheet(
    artifact: dict[str, Any],
    dossier: dict[str, Any],
    parent: dict[str, Any] | None,
) -> str:
    parent_text = {}
    if parent is not None:
        parent_text = {
            str(record["id"]): record for record in parent.get("child_responses", [])
        }
    responses = {str(r["id"]): r for r in dossier["child_responses"]}
    columns = artifact["columns"]
    lines: list[str] = []
    add = lines.append
    add(f"# SCORING SHEET — {artifact['label']} ({artifact['stage']})")
    add("")
    add(
        "**THE VERDICT AND EVERY COUNT BELOW ARE ANNOTATIONS. THE DECISION IS "
        "CYRIL'S.** Nothing in this file grades. Tick a box per lane; the "
        "artifact-tolerant lens applies: loops, echo, cap-rides and a missing "
        "EOS never count against intelligence — only a content-miss does."
    )
    add("")
    add("| field | value |")
    add("|---|---|")
    add(f"| model | `{dossier['provenance']['model_directory']}` |")
    add(f"| directory sha256 | `{dossier['provenance'].get('directory_sha256')}` |")
    add(f"| loaded class | `{dossier['export']['loaded_class']}` |")
    add(f"| parameters (live) | {dossier['export']['parameters_counted_live']:,} |")
    add(f"| transformers | {dossier['export']['transformers_version']} |")
    add(f"| dtype / device | {artifact['generation']['dtype']} / "
        f"{artifact['generation']['compute_architecture']} |")
    add(f"| max new tokens | {artifact['generation']['maximum_new_tokens']} |")
    add(f"| bank sha256 | `{artifact['bank_file_sha256']}` |")
    add(f"| dossier sha256 | `{artifact['dossier_sha256']}` |")
    add(f"| lane digest | `{artifact['lane_digest']}` |")
    add(f"| emit-zero floor | {artifact['emit_zero_floor_passed']} |")
    add(f"| decode-path preflight agree | {artifact['decode_path_agree']} |")
    if artifact["is_random_init"]:
        add("| **RANDOM INIT** | **this model was never trained; nothing here is a calibration** |")
    add("")
    add("## Columns")
    add("")
    kind = "UPPER BOUND (no reading attached yet)" if columns["is_upper_bound"] else "reading"
    add(f"* **Column A — copy-back counts:** {columns['column_a_copy_back_counts']} / "
        f"{columns['denominator_scored_lanes']}  ({kind})")
    add(f"* **Column B — copy-back refused:** {columns['column_b_copy_back_refused']} / "
        f"{columns['denominator_scored_lanes']}  ({kind})")
    if columns["voided_lanes"]:
        add(f"* voided as ECHO (column A minus these **is** column B): "
            f"{', '.join(columns['voided_lanes'])}")
    if columns["is_upper_bound"]:
        add(
            "* what the bound subtracts: only "
            f"{', '.join(columns['upper_bound_subtracts_only'])} — the boxes a "
            "machine can prove hold nothing. An ECHO lane is still inside "
            "column A, because *“this is a copy”* is a threshold call and not a "
            f"certainty; your reading may hold any of the "
            f"{columns['denominator_scored_lanes']} scored lanes."
        )
    three = artifact["canonical_three_box"]
    add(f"* canonical boxes — signal {three['signal']}, artifact-miss "
        f"{three['artifact_miss']}, content-miss {three['content_miss']}, "
        f"still to read {three['read_required']}")
    add("")
    if artifact["anomalies"]:
        add("## Anomalies")
        add("")
        for anomaly in artifact["anomalies"]:
            add(f"* {anomaly}")
        add("")
    if artifact.get("instrument_cautions"):
        add("## Instrument cautions — read these lanes yourself")
        add("")
        add(
            "A copy can satisfy a PRESERVE ask and neither a SELECT nor a "
            "PRODUCE one. Each lane below was called ECHO on a non-PRESERVE "
            "ask, which means either the model handed the payload back or the "
            "0.60 copy threshold — calibrated on LFM2.5-230M — does not "
            "transfer here. **The threshold was not moved.**"
        )
        add("")
        for caution in artifact["instrument_cautions"]:
            add(
                f"* `{caution['lane']}` ({caution['lane_ask']}) — payload union "
                f"coverage {caution['payload_union_coverage']}"
            )
        add("")
    add("## Lane summary")
    add("")
    add("| lane | ask | form box | term | tokens | rubric | your grade |")
    add("|---|---|---|---|---|---|---|")
    for lane in artifact["lanes"]:
        rubric = lane.get("machine_rubric")
        rubric_cell = "—"
        if rubric and rubric.get("applicable"):
            rubric_cell = "pass" if rubric["passed"] else "no match"
            if rubric.get("response_was_truncated"):
                rubric_cell += " (truncated — read it)"
        add(
            f"| {lane['lane']} | {lane['lane_ask']} | {lane['form_box']} | "
            f"{lane['termination']} | {lane['generated_tokens']} | {rubric_cell} | "
            "&nbsp; |"
        )
    add("")
    add("## Every case, verbatim")
    add("")
    for lane in artifact["lanes"]:
        record = responses[lane["lane"]]
        add(f"### {lane['lane']}  ({lane['category']}, ask {lane['lane_ask']})")
        add("")
        if lane["is_safety_lane"]:
            add("> **SAFETY LANE — charged, never scored.** Read it and write a "
                "free-text reading; it does not enter any count.")
            add("")
        add("**Prompt**")
        add("")
        for message in record["prompt_messages"]:
            content = message["content"]
            if len(content) > 4000:
                content = (
                    content[:2000]
                    + f"\n\n... [{len(content) - 4000} characters elided; the "
                    "full text is in the dossier and is digest-pinned] ...\n\n"
                    + content[-2000:]
                )
            add(f"*{message['role']}*:")
            add("")
            add("```")
            add(content)
            add("```")
            add("")
        if lane.get("bank_expectation"):
            add(
                f"**What the bank says this case asked for** (the bank's own "
                f"annotation — not a key, not a grade): "
                f"{lane['bank_expectation']}"
            )
            add("")
        add(
            f"**Response** ({lane['generated_tokens']} tokens, "
            f"{lane['termination']}, form box {lane['form_box']} — "
            f"{lane['form_reason']})"
        )
        add("")
        add("```")
        add(record["response_verbatim"] or "(empty)")
        add("```")
        add("")
        if lane["lane"] in parent_text:
            add("**Parent on the same prompt**")
            add("")
            add("```")
            add(parent_text[lane["lane"]]["response_verbatim"] or "(empty)")
            add("```")
            add("")
        rubric = lane.get("machine_rubric")
        if rubric and rubric.get("applicable"):
            add(f"*machine rubric (a diagnostic, not a grade): "
                f"{'pass' if rubric['passed'] else 'no match'}*")
            if rubric.get("response_was_truncated"):
                add(f"  * **{rubric['response_was_truncated']}**")
            for check in rubric["checks"]:
                mark = "ok" if check["ok"] else "MISS"
                add(f"  * `{check['kind']}` {check['pattern']!r} — {mark}")
            add("")
        if not lane["is_safety_lane"]:
            add("**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   "
                "[ ] Inconclusive/censored`")
            add("")
            add("**If a miss — artifact-miss or content-miss?**  `[ ] artifact "
                "(loop / echo / cap-ride / no EOS — counts nothing)   [ ] content "
                "(wrong, vacuous, deflecting — counts against)`")
        else:
            add("**Your reading (free text, never scored):** ")
        add("")
        add("---")
        add("")
    smoke = dossier.get("decode_smoke", {})
    if smoke.get("records"):
        add("## Coherence smoke (three prompts, 128-token cap)")
        add("")
        for record in smoke["records"]:
            add(f"* terminated `{record['terminated']}`, "
                f"{record['generated_tokens']} tokens")
            add("")
            add("```")
            add(record["response_verbatim"] or "(empty)")
            add("```")
            add("")
        add("**Smoke verdict:** `[ ] coherent   [ ] incoherent`")
        add("")
    floor = dossier.get("emit_zero_floor", {})
    add("## Emit-zero floor")
    add("")
    add(f"* passed: **{floor.get('passed')}**")
    add(f"* min terminal ‖h‖: {floor.get('min_terminal_l2')}")
    add(f"* min logit sd: {floor.get('min_logit_std')}")
    add(f"* distinct argmax tokens over {floor.get('probe_count')} probes: "
        f"{floor.get('distinct_argmax_tokens')}")
    add(f"* definition: {floor.get('definition')}")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--parent-dossier", type=Path)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--per-case", type=Path)
    ap.add_argument("--sealed-useful-or-partial", type=int)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-markdown", required=True, type=Path)
    ap.add_argument(
        "--allow-random-init",
        action="store_true",
        help=(
            "write a sheet for a RANDOM-INIT dossier.  Refused by default: a "
            "sheet asks Cyril to read text, and text from an untrained model is "
            "not worth his time."
        ),
    )
    args = ap.parse_args()

    dossier = json.loads(Path(args.dossier).read_text(encoding="utf-8"))
    if dossier.get("schema") != "neural-forge-omni-parent-calibration-dossier-v1":
        raise CalibrationError(f"{args.dossier} is not an omni calibration dossier.")
    if dossier.get("is_random_init") and not args.allow_random_init:
        raise CalibrationError(
            "REFUSED: this dossier came from a RANDOM-INIT mechanics model.  Its "
            "text is noise and a scoring sheet built from it would waste the one "
            "resource the campaign cannot make more of.  Pass "
            "--allow-random-init if you are exercising the pipeline."
        )
    per_case = None
    if args.per_case:
        per_case = json.loads(Path(args.per_case).read_text(encoding="utf-8"))
    parent = None
    if args.parent_dossier:
        parent = json.loads(Path(args.parent_dossier).read_text(encoding="utf-8"))
    artifact = assess(
        dossier,
        bank=bank_index(args.bank),
        per_case=per_case,
        sealed_useful_or_partial=args.sealed_useful_or_partial,
    )
    artifact["source_dossier_file_sha256"] = file_sha256(Path(args.dossier))
    artifact["assessment_sha256"] = canonical_sha256(
        {k: v for k, v in artifact.items() if k != "assessment_sha256"}
    )
    json_digest = write_canonical(Path(args.out_json), artifact)

    sheet = markdown_sheet(artifact, dossier, parent)
    out_md = Path(args.out_markdown)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(sheet, encoding="utf-8", newline="\n")
    reread = out_md.read_text(encoding="utf-8")
    if reread != sheet:
        raise CalibrationError(f"{out_md} did not survive its own write.")
    md_digest = file_sha256(out_md)

    print(
        json.dumps(
            {
                "assessment_json": str(args.out_json),
                "assessment_json_sha256": json_digest,
                "scoring_sheet": str(out_md),
                "scoring_sheet_sha256": md_digest,
                "column_a": artifact["columns"]["column_a_copy_back_counts"],
                "column_b": artifact["columns"]["column_b_copy_back_refused"],
                "is_upper_bound": artifact["columns"]["is_upper_bound"],
                "canonical_three_box": artifact["canonical_three_box"],
                "anomalies": len(artifact["anomalies"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
