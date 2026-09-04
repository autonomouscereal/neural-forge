#!/usr/bin/env python3
"""The review JSON the ledger accepts, built from an UNREAD assessment.

WHAT THIS FILE IS FOR.  `oc_assess.py` run WITHOUT `--per-case` publishes two
UPPER BOUNDS and a canonical three-box that is entirely `READ-REQUIRED`, because
a machine may not fill the signal box.  `oc_checkpoint.py record-review` wants a
review whose three canonical boxes SUM to the intelligence lane count.  Those two
statements are not compatible on their face, and the gap between them is the
whole reason this file exists: it performs one explicit, named, reversible
translation from the bound into the schema, and it labels every number it writes
as a ceiling rather than a reading.

THE TRANSLATION, FIELD BY FIELD, AND WHY EACH ONE IS THE HONEST CHOICE
----------------------------------------------------------------------
Let `S` be the scored lanes (`columns.denominator_scored_lanes`) and `A` be
`columns.column_a_copy_back_counts`, which in the unread branch is *the count of
scored lanes whose form box is not one of EMPTY / LOOP / CAP_RIDE* -- i.e. the
lanes a machine CANNOT prove are holding nothing.

* ``bank1.canonical_signal = A``.  This is a CEILING.  At most `A` lanes can
  turn out to be signal, because the other `S - A` lanes provably emitted
  nothing, looped, or rode the cap saying nothing.  Writing `0` here instead
  would be worse, not more cautious: `status_for` reads a zero-signal review
  with zero content misses as ``unmeasured`` -- "re-run it" -- and a child whose
  fifteen lanes all produced readable prose would be filed as never having run.
  The ceiling routes it to ``partially_capable``, whose own status reason is
  *"goes to Cyril's review queue"*, which is exactly where an unread child
  belongs.  ``is_upper_bound: true`` on the block is what stops a reader
  mistaking the ceiling for a count.
* ``bank1.canonical_artifact_miss = S - A``.  These are the machine-refuted
  boxes and nothing else.  The canonical lens boxes every one of them as an
  artifact, and an artifact counts for nothing in either direction.
* ``bank1.canonical_content_miss = 0``.  A CONTENT MISS IS THE ONLY BOX THAT
  COUNTS AGAINST A CHILD AND IT IS THE ONLY BOX A MACHINE MAY NEVER FILL: it
  means "readable, on register, and wrong", which is a reading.  Zero here is
  not optimism, it is the refusal to invent the one judgement that can make a
  checkpoint ``bad`` and therefore deletable.  It follows that NOTHING THIS FILE
  WRITES CAN EVER MAKE A CHILD DELETABLE, which is the property we want.
* ``bank1.useful = 0``, ``bank1.partial = A``.  `validate_review` requires
  `column_a == useful + partial`, so the bound has to land in one of the two.
  It lands in the weaker word.  Calling `A` lanes "Useful" would be a claim
  about quality that nobody has made.
* ``bank1.observed_miss = S - A``, the same machine-refuted lanes.  These ARE
  observed: the observation is "it emitted nothing / looped / rode the cap".
* ``bank1.sealed_verdict``.  ``BEHAVIOR_VERDICTS`` has no member meaning *not
  read yet*, so the schema forces one of five readings out of a review that
  contains none.  The one written here is the BEST verdict the counts could
  support -- `alive-parent-band` if the ceiling reaches the parent's own count,
  `alive-degraded` if it is above zero, `dead` only when `A == 0`, which is the
  single case a machine really can establish (every scored lane provably said
  nothing).  It is stamped ``sealed_verdict_is_upper_bound: true`` and the basis
  is written beside it.  THIS IS A KNOWN SCHEMA MISMATCH, NOT A GRADE.
* ``bank1.lanes_total``.  The ledger computes `intelligence = lanes_total - 1`
  and hardwires "the safety lane never scores", so `lanes_total` is really the
  ledger's spelling of `S + 1`.  When the safety lane rode along in the run --
  the normal case for the frozen 16-case pack -- `S + 1` IS the bank size and
  the two agree.  When a run was truncated (`--max-cases`) or the bank has no
  safety lane at all (bank v2), `S + 1` is NOT the bank size, and the review
  says so in `bank1.lanes_total_note` rather than quietly filing a wrong number.
* ``bank1.parent_useful_or_partial`` has no default anywhere and is never
  guessed.  `CAMPAIGN.md` gap (7) says the parent's `bands.json` cannot exist
  until Cyril scores the parent, so on the Wave A queue the only available
  source is THE PARENT'S OWN UPPER BOUND, taken from its assessment artifact and
  labelled `parent_useful_or_partial_is_upper_bound: true`.  A real
  `bands.json`, or an explicit integer, overrides it and is labelled as such.
* ``columns`` are copied straight across, unmodified.
* ``bank2`` carries the v2 text lanes under the same translation.  Note the key
  the ledger row actually reads is ``sealed_useful_or_partial`` (not
  ``useful_or_partial``): see `checkpoint_row`.
* ``media_out`` is empty.  This queue decodes text; a media lane it never ran is
  not an artifact miss it gets to charge.
* ``smoke_verdict`` comes from the dossier's own `decode_smoke.verdict`, which
  is the string *"pending -- a human reads these three..."*, and maps to
  ``pending``.  It is never upgraded to ``coherent`` here, because that too is a
  reading.
* ``degeneracy_reading`` stays ``unstated``.  A LOOP box is evidence, not a
  degeneracy call, and the vocabulary has no "bounded" member.

WHAT THIS FILE REFUSES
----------------------
* an assessment whose ``columns.is_upper_bound`` is FALSE -- that one carries a
  reading, and a reading is recorded from the per-case file, not from here;
* a random-init assessment without ``--allow-random-init`` (the same refusal
  `oc_assess.py` makes, for the same reason);
* an assessment and a dossier that are not about the same run;
* a parent count with no stated source.

Usage::

    oc_review_upper_bound.py --assessment out/assessment-<ID>-NOTHINK.json \\
        --dossier out/oc-dossier-<ID>-NOTHINK.json \\
        [--assessment-b2 out/assessment-<ID>-B2.json] \\
        [--parent-assessment out/assessment-Q35-4B-NOTHINK.json | \\
         --parent-useful-or-partial N | --parent-bands bands.json] \\
        --reviewer wavea_review.sh --out out/review-<ID>.json \\
        [--package-path DIR --package-sha256 HEX]
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
    canonical_sha256,
    file_sha256,
    write_canonical,
)

ASSESSMENT_SCHEMA = "neural-forge-omni-calibration-dual-band-assessment-v1"
DOSSIER_SCHEMA = "neural-forge-omni-parent-calibration-dossier-v1"

#: The boxes column A has already subtracted.  Named here only so the review can
#: say what its artifact-miss count is made of; the arithmetic is `oc_assess`'s.
MACHINE_REFUTED_BOXES = ("EMPTY", "LOOP", "CAP_RIDE")


def load_json(path: Path, *, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationError(f"REFUSED: {what} {path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"REFUSED: {what} {path} is not JSON: {exc}") from exc


def bound_block(artifact: dict[str, Any], *, what: str) -> dict[str, int]:
    """`scored`, `column_a` and `column_b` off one assessment, checked.

    Every caller is about to write a ceiling into a sealed record, so the shape
    it depends on is asserted here rather than assumed.
    """

    if artifact.get("schema") != ASSESSMENT_SCHEMA:
        raise CalibrationError(
            f"REFUSED: {what} declares schema {artifact.get('schema')!r}, not "
            f"{ASSESSMENT_SCHEMA!r}"
        )
    columns = artifact.get("columns") or {}
    if not columns.get("is_upper_bound"):
        raise CalibrationError(
            f"REFUSED: {what} was built WITH a per-case reading "
            "(`columns.is_upper_bound` is false).  This file translates a BOUND "
            "into the ledger's schema and would relabel a real reading as a "
            "ceiling.  Record a read assessment from the per-case file instead."
        )
    scored = int(columns["denominator_scored_lanes"])
    column_a = int(columns["column_a_copy_back_counts"])
    column_b = int(columns["column_b_copy_back_refused"])
    if not 0 <= column_b <= column_a <= scored:
        raise CalibrationError(
            f"REFUSED: {what} columns are not 0 <= B({column_b}) <= "
            f"A({column_a}) <= scored({scored})"
        )
    return {"scored": scored, "column_a": column_a, "column_b": column_b}


def verdict_ceiling(column_a: int, parent_useful_or_partial: int) -> tuple[str, str]:
    """The BEST verdict the bound could support, and the sentence saying so.

    `BEHAVIOR_VERDICTS` has no "unread" member (see the module docstring), so
    the schema forces a word.  `dead` is the only one of the five a machine can
    actually establish, and it is the only one this function returns with a
    straight face; the other two are ceilings and say so.
    """

    if column_a == 0:
        return (
            "dead",
            f"every scored lane is in {list(MACHINE_REFUTED_BOXES)} -- empty, "
            "looping, or riding the cap without saying anything.  This is the ONE "
            "verdict a machine can establish without a reading, and it still "
            "charges nothing against the child: all of these are ARTIFACT misses "
            "and the status they imply is `unmeasured`, i.e. re-run it.",
        )
    if column_a >= parent_useful_or_partial:
        return (
            "alive-parent-band",
            f"CEILING ONLY: the bound ({column_a}) reaches the parent's own count "
            f"({parent_useful_or_partial}), so `alive-parent-band` is the best "
            "verdict these numbers could support.  It is NOT a reading and the "
            "child may read anywhere at or below it.",
        )
    return (
        "alive-degraded",
        f"CEILING ONLY: the bound ({column_a}) is above zero and below the "
        f"parent's count ({parent_useful_or_partial}), so `alive-degraded` is the "
        "best verdict these numbers could support.  It is NOT a reading.",
    )


def resolve_parent(args: argparse.Namespace) -> tuple[int, str, bool]:
    """`(count, source sentence, is_upper_bound)` -- never a default, never a guess."""

    if args.parent_useful_or_partial is not None:
        return (
            int(args.parent_useful_or_partial),
            f"--parent-useful-or-partial {args.parent_useful_or_partial}, stated "
            "on the command line by the operator",
            False,
        )
    if args.parent_bands:
        path = Path(args.parent_bands)
        bands = load_json(path, what="--parent-bands")
        for key in (
            "parent_useful_or_partial",
            "useful_or_partial",
            "sealed_useful_or_partial",
        ):
            if key in bands:
                return (
                    int(bands[key]),
                    f"{path.name} key {key!r} (sha256 {file_sha256(path)[:16]})",
                    False,
                )
        raise CalibrationError(
            f"REFUSED: {path} names none of parent_useful_or_partial / "
            "useful_or_partial / sealed_useful_or_partial.  The parent count is "
            "never guessed."
        )
    if args.parent_assessment:
        path = Path(args.parent_assessment)
        parent = load_json(path, what="--parent-assessment")
        block = bound_block(parent, what=f"--parent-assessment {path.name}")
        return (
            block["column_a"],
            f"{path.name} column A ({block['column_a']}/{block['scored']}), "
            f"assessment_sha256 {str(parent.get('assessment_sha256'))[:16]}.  "
            "THE PARENT HAS NOT BEEN READ EITHER: `bands.json` cannot exist until "
            "Cyril scores the parent (CAMPAIGN.md gap 7), so this side of the "
            "comparison is a bound as well and both sides move when he reads.",
            True,
        )
    raise CalibrationError(
        "REFUSED: no parent count.  `bank1.parent_useful_or_partial` has no "
        "default -- it comes from the parent's bands.json and is never guessed.  "
        "Pass one of --parent-bands / --parent-assessment / "
        "--parent-useful-or-partial."
    )


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    assessment = load_json(Path(args.assessment), what="--assessment")
    dossier = load_json(Path(args.dossier), what="--dossier")
    if dossier.get("schema") != DOSSIER_SCHEMA:
        raise CalibrationError(
            f"REFUSED: --dossier declares schema {dossier.get('schema')!r}, not "
            f"{DOSSIER_SCHEMA!r}"
        )
    # THE TWO FILES MUST BE ABOUT THE SAME RUN.  A review that quotes one run's
    # dossier digest beside another run's counts is not repairable once sealed,
    # and the pair is free to check here.
    if assessment.get("dossier_sha256") != dossier.get("dossier_sha256"):
        raise CalibrationError(
            "REFUSED: the assessment was built from dossier "
            f"{str(assessment.get('dossier_sha256'))[:16]} and --dossier seals to "
            f"{str(dossier.get('dossier_sha256'))[:16]}.  Not the same run."
        )
    if dossier.get("is_random_init") and not args.allow_random_init:
        raise CalibrationError(
            "REFUSED: this dossier came from a RANDOM-INIT mechanics model.  A "
            "review record built from noise would put an untrained model in "
            "Cyril's queue.  Pass --allow-random-init if you are exercising the "
            "pipeline against a throwaway registry."
        )

    b1 = bound_block(assessment, what="--assessment")
    scored, column_a, column_b = b1["scored"], b1["column_a"], b1["column_b"]
    safety_present = bool((assessment.get("safety_lane") or {}).get("present"))
    lanes_in_run = len(assessment.get("lanes") or [])
    lanes_total = scored + 1
    if safety_present and lanes_total != lanes_in_run:
        raise CalibrationError(
            f"REFUSED: the run carried the safety lane and {lanes_in_run} lane(s) "
            f"in total, but {scored} scored lane(s) imply {lanes_total}.  More "
            "than one unscored lane is a shape this translation does not "
            "understand."
        )
    lanes_total_note = (
        "the run carried the frozen pack's safety lane, so lanes_total IS the "
        f"bank size ({lanes_in_run})"
        if safety_present
        else (
            "THE SAFETY LANE DID NOT RUN (a truncated --max-cases run, or a bank "
            "that has none).  `lanes_total` is the ledger's spelling of scored+1 "
            f"and is NOT this bank's size; {scored} lane(s) were scored."
        )
    )

    parent_count, parent_source, parent_is_bound = resolve_parent(args)
    verdict, verdict_basis = verdict_ceiling(column_a, parent_count)

    smoke = str((dossier.get("decode_smoke") or {}).get("verdict") or "")
    # Every state this file can be in is `pending`: a smoke verdict is a human
    # reading three generations and saying a word, and no such word exists yet.
    smoke_verdict = "pending"

    sheet_name = (
        Path(args.assessment)
        .name.replace("assessment-", "SCORING-SHEET-")
        .replace(".json", ".md")
    )
    bank1 = {
        "lanes_total": lanes_total,
        "lanes_total_note": lanes_total_note,
        "useful": 0,
        "partial": column_a,
        "observed_miss": scored - column_a,
        "parent_useful_or_partial": parent_count,
        "parent_useful_or_partial_source": parent_source,
        "parent_useful_or_partial_is_upper_bound": parent_is_bound,
        "sealed_verdict": verdict,
        "sealed_verdict_is_upper_bound": True,
        "sealed_verdict_basis": verdict_basis,
        "canonical_signal": column_a,
        "canonical_artifact_miss": scored - column_a,
        "canonical_content_miss": 0,
        "is_upper_bound": True,
        "upper_bound_note": (
            "canonical_signal is a CEILING, not a count: it is the number of "
            "scored lanes a machine could not refute, and all "
            f"{assessment['canonical_three_box']['read_required']} of the scored "
            "lanes are still READ-REQUIRED in the assessment artifact.  "
            "canonical_content_miss is 0 because a content miss is a READING and "
            "is the only box that can make a checkpoint deletable; nothing "
            "written by this tool can ever make a child `bad`.  useful is 0 and "
            "the whole bound sits in `partial`, the weaker word.  The reading "
            f"that replaces these numbers is the per-case file against {sheet_name}."
        ),
        "assessment_json_sha256": file_sha256(Path(args.assessment)),
        "assessment_sha256": assessment.get("assessment_sha256"),
        "lane_digest": assessment.get("lane_digest"),
        "bank_file_sha256": assessment.get("bank_file_sha256"),
        "voided_lanes": (assessment.get("columns") or {}).get("voided_lanes"),
        "instrument_caution_count": assessment.get("instrument_caution_count"),
    }

    bank2 = None
    if args.assessment_b2:
        path2 = Path(args.assessment_b2)
        a2 = load_json(path2, what="--assessment-b2")
        b2 = bound_block(a2, what=f"--assessment-b2 {path2.name}")
        bank2 = {
            "lanes_total": b2["scored"],
            "sealed_useful_or_partial": b2["column_a"],
            "column_a": b2["column_a"],
            "column_b": b2["column_b"],
            "canonical_signal": b2["column_a"],
            "canonical_artifact_miss": b2["scored"] - b2["column_a"],
            "canonical_content_miss": 0,
            "is_upper_bound": True,
            "label": a2.get("label"),
            "bank_file_sha256": a2.get("bank_file_sha256"),
            "lanes": [lane["lane"] for lane in (a2.get("lanes") or [])],
            "assessment_json_sha256": file_sha256(path2),
            "assessment_sha256": a2.get("assessment_sha256"),
            "upper_bound_note": (
                "the bank v2 draft text lanes under the same translation as "
                "bank1: canonical_signal is a ceiling and content_miss is 0 "
                "because nobody has read them.  This bank carries no safety "
                "lane, so lanes_total IS the number of scored lanes here."
            ),
        }

    review: dict[str, Any] = {
        "reviewer": str(args.reviewer),
        "review_kind": "machine upper bound; NOTHING HERE IS A READING",
        "dossier_sha256": dossier.get("dossier_sha256"),
        "dossier_file_sha256": file_sha256(Path(args.dossier)),
        # There is no sealed verdict receipt for an unread pass: the LFM2
        # `cell-receipt-behavior-reviewed.json` is written when a human seals a
        # reading, and inventing a digest here would put a 64-hex string in the
        # ledger's `verdict_receipt_sha` column that resolves to nothing.  The
        # assessment digests in bank1 are the artifacts that DO exist.
        "verdict_receipt_sha": None,
        "bank1": bank1,
        "columns": {"column_a": column_a, "column_b": column_b},
        "bank2": bank2,
        "media_out": [],
        "smoke_verdict": smoke_verdict,
        "smoke_verdict_source": smoke or "(no decode smoke in this dossier)",
        "safety": (
            "UNREAD.  The safety lane is charged and never scored; its reading "
            "is free text and it is sitting in the scoring sheet waiting for one."
        ),
        "degeneracy_reading": "unstated",
        "is_upper_bound": True,
        "is_random_init": bool(dossier.get("is_random_init")),
        "label": assessment.get("label"),
        "stage": assessment.get("stage"),
        "lane_digest_validity": assessment.get("lane_digest_validity"),
        "emit_zero_floor_passed": assessment.get("emit_zero_floor_passed"),
        "decode_path_agree": assessment.get("decode_path_agree"),
        "notes": args.notes
        or (
            f"Wave A first pass, {assessment.get('label')}: bank v1 --thinking "
            "off, assessed WITHOUT --per-case.  EVERY COUNT IN THIS RECORD IS AN "
            "UPPER BOUND and none of it is a reading -- see "
            "bank1.upper_bound_note.  The status it implies routes the child to "
            "Cyril's queue; the reading that replaces it is his per-case file "
            "against the scoring sheet."
        ),
    }
    if args.package_path:
        if not args.package_sha256:
            raise CalibrationError(
                "REFUSED: --package-path without --package-sha256.  The ledger "
                "refuses that pair too: a path with no digest is a claim about a "
                "folder that can be rebuilt under the same name with different "
                "contents."
            )
        review["package_path"] = str(args.package_path)
        review["package_sha256"] = str(args.package_sha256)
    review["review_payload_sha256"] = canonical_sha256(review)
    return review


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assessment", required=True, type=Path)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--assessment-b2", type=Path)
    ap.add_argument("--parent-assessment", type=Path)
    ap.add_argument("--parent-bands", type=Path)
    ap.add_argument("--parent-useful-or-partial", type=int)
    ap.add_argument("--reviewer", default="wavea_review.sh")
    ap.add_argument("--notes", default="")
    ap.add_argument("--package-path")
    ap.add_argument("--package-sha256")
    ap.add_argument("--allow-random-init", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        review = build_review(args)
    except CalibrationError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    digest = write_canonical(Path(args.out), review)
    print(
        json.dumps(
            {
                "review_json": str(args.out),
                "review_json_sha256": digest,
                "column_a": review["columns"]["column_a"],
                "column_b": review["columns"]["column_b"],
                "bank1_lanes_total": review["bank1"]["lanes_total"],
                "bank1_canonical_signal_ceiling": review["bank1"]["canonical_signal"],
                "bank1_canonical_artifact_miss": review["bank1"][
                    "canonical_artifact_miss"
                ],
                "bank1_canonical_content_miss": review["bank1"][
                    "canonical_content_miss"
                ],
                "bank1_sealed_verdict_ceiling": review["bank1"]["sealed_verdict"],
                "parent_useful_or_partial": review["bank1"]["parent_useful_or_partial"],
                "bank2_canonical_signal_ceiling": (review["bank2"] or {}).get(
                    "canonical_signal"
                ),
                "is_upper_bound": True,
                "is_random_init": review["is_random_init"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
