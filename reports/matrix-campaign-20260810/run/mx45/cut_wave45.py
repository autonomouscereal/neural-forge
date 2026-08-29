#!/usr/bin/env python3
"""Cut wave 45's tree from wave 44's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-44 cutter, in full.**

1. ``SRC`` is the wave-44 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  Wave 40's seal-call pin -- ``audit()``
   counts ``^seal <TAG>`` and requires **zero at cut time** -- is carried, and
   wave 45 is the FIFTH cut where it had work to do from the start.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README.  **Wave
   45 authors no new module and EDITS ONE CARRIED MODULE**, which is the same
   shape as wave 44 and is stated here rather than left to be inferred from a
   diff.
4. **THE EDIT IS ``mx45_dualband.py`` AND IT IS THE NAMED DEBT WAVE 44 REFUSED
   TO PAY MID-MEASUREMENT.**  `NOTES-MX-AV` §2.3 measured, published and
   deliberately did not repair a single defect: all eight void-changing
   machine-hand disagreements sit on lanes with **no ``PAYLOAD_ANCHORS``
   entry**, seven of them on ``writing-concise-email``.  §12 item 1 told the
   next wave to register the anchor BEFORE the pass that uses it.  This cut is
   the first thing wave 45 does, before any wave-45 decode exists, and the
   anchor is added under the SAME RULE the three existing anchors follow --
   strip the leading task directive, keep the material -- and not under a rule
   chosen to move a number.

   **AND THE REGISTERED PREDICTION FOR IT IS A NULL RESULT.**  The arithmetic
   is stated in `run/mx45/README.md` §2 against the two distinct committed
   responses: under the anchor the seven rows read 0.477707 and 0.452229, which
   is BELOW the pre-registered 0.60 and slightly below where they already were.
   The ordered repair is made and it is registered as not closing the rows it
   was ordered to close, because the mechanism wave 44 named is not the
   mechanism that is there.  Registering a prediction one expects to hit is not
   the point of registering.

5. **THE SECOND EDIT TO THE SAME MODULE IS A PUBLISHED, NON-DECIDING COLUMN**
   and it follows the campaign's own promotion protocol exactly.  Waves 42 and
   43 published ``payload_union_coverage`` beside every lane while the
   longest-run measure decided, and wave 44 promoted it on two waves of
   published evidence.  ``response_union_coverage`` -- the same union, with the
   RESPONSE as the denominator instead of the payload -- is added here as a
   published column that decides nothing, so that a later wave can promote it
   with evidence behind it rather than with an argument.  Its decision line is
   pinned at ZERO.

**WAVE 45'S OWN ADDITION TO THIS AUDIT, AND IT IS THE FILE `NOTES-MX-AV` §12.6
NAMED.**  Wave 43 found the sealer's own ``BUILD_RECEIPT_SCHEMAS`` after it had
already cost a deployment; wave 44 pinned the three path resolutions inside
``sealq.sh``.  §12.6 named the next one: ``seal_matrix_behavior_review.py``'s
own constants, which ``seal_verdict.sh`` passes flags to and which **no cut has
ever read**.  Reading it turns up something narrower and worse than a constant
in that file: the numbers that turn a reviewer's count into a VERDICT do not
live in that file at all.  ``PARENT_REVIEW_BAND``, ``A9_REVIEW_BAND`` and
``PARENT_BAND_TOLERANCE`` live in ``src/neural_forge/matrix_cell_receipt.py``,
outside the cut tree and outside the sealer, and ``behavior_verdict_for`` reads
them to decide `alive-parent-band` / `alive-degraded` / `impaired` for every
child this campaign has ever sealed.  **Nothing in forty-four waves has ever
asserted their values.**  ``audit()`` now pins all three by name and value,
together with the sealer's three argparse defaults and both verdict
vocabularies.

**THE NEEDLES CARRY THEIR INDENTATION** -- wave 38's lesson 6, carried, and
applied to the edited module's new lines from the start rather than after the
first cut eats one.

**WHAT THIS CUT DOES NOT DO.**  It does not author a NORM-ESTATE Arm-M/Arm-S
solver, and it does not unblock ALIAS or ROWSET Arms A and B -- those wait on a
ruling and on a corpus respectively, which are not authoring problems.  It also
builds nothing on the PLANE axis: wave 43 re-sealed that axis under the slate's
own kill gate, and ``mx45_plane.py`` is carried, pinned and dormant.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  **And it parses what it writes, in the invocation that
writes it** -- `NOTES-MX-AO` §9's rule, obeyed by the first file wave 45 writes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx44"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx44_verify_seals.py",
    "mx44_retain.py",
    "mx44_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files -- this
#: cutter and the README.  Wave 45's deliverable is a DEBT PAYMENT inside a
#: carried module plus a close-out document, not a new module, and declaring two
#: files here rather than six is how that difference is stated in code.
AUTHORED_THIS_WAVE = {
    "cut_wave45.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt.
NOT_CARRIED = {"wave44-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 44", "--campaign-wave 45"),
    ('"campaign_wave": 44', '"campaign_wave": 45'),
    ("wave44", "wave45"),
    ("mx44", "mx45"),
    ("MX44", "MX45"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.
FORBIDDEN_AFTER = {
    "mx44_": "a wave-44 module reference survived the copy",
    "MX44-": "a wave-44 child directory name survived the copy",
    "/wave44/": "a wave-44 artifact path survived the copy",
    "--campaign-wave 44": "a driver would seal this wave's build as wave 44",
}
#: DELIBERATE historical references, declared with a count.  The third field is
#: the comparison: an exact pin where the count is knowable before the wave
#: runs, a bound where it is not.
ALLOWED_HISTORICAL = {
    ("mx45_build.py", "mx44_"): (
        1,
        "==",
        "the wave-45 header names `mx44_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx45_vocab.py", "mx44_"): (
        1,
        "<=",
        "the wave-45 header may name `mx44_vocab.py` as the file it states its "
        "diff against.  Bounded rather than pinned because this wave makes no "
        "edit to that module and may add no header at all",
    ),
    ("mx45_dualband.py", "mx44_"): (
        1,
        "<=",
        "THIS WAVE'S ONE EDITED MODULE.  Its new header states its diff against "
        "`mx44_dualband.py`, which is the file it was copied from and the file "
        "whose anchor table it completes.  Bounded at one and not pinned, "
        "because the header is prose and a second sentence naming the same file "
        "would be a style change and not a defect",
    ),
    ("mx45_retain.py", "MX44-"): (
        3,
        "<=",
        "the retention rule requires each DROP reason to name the retained "
        "source that rebuilds the deleted child.  Wave 44 retained MX44-V3CL768 "
        "and wave 45's CL768 rungs sit directly below it on the same ladder, so "
        "a legitimate cross-reference is a note and not a refusal",
    ),
    ("sealq.sh", "MX44-"): (
        4,
        "<=",
        "a seal note naming a checkpoint a child is read against.  Wave 45's "
        "vocabulary source is wave 40's retained MX40-CL768 and its nearest "
        "sealed control is wave 44's retained MX44-V3CL768 -- the rung "
        "immediately above both of this wave's cells -- which a seal note "
        "legitimately carries",
    ),
    ("mx45_plane.py", "MX44-"): (
        0,
        "==",
        "the plane solver folds the PARENT and names no wave-44 child at all.  "
        "Carried from wave 44's cutter unchanged and pinned at zero rather than "
        "left undeclared, although wave 45 does not run this module: the axis is "
        "re-sealed and a dormant module still has to survive a cut intact",
    ),
}


def rename(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for old, new in SUBS:
        n = text.count(old)
        if n:
            counts[old] = n
            text = text.replace(old, new)
    return text, counts


def main() -> int:
    if not SRC.is_dir():
        raise SystemExit(f"REFUSED: no source tree at {SRC}")
    entries: list[dict[str, object]] = []
    for path in sorted(SRC.iterdir()):
        if not path.is_file():
            continue
        if path.name in NOT_CARRIED:
            entries.append({"source": path.name, "target": "",
                            "action": "NOT CARRIED", "substitutions": {}})
            continue
        target_name, _ = rename(path.name)
        target = DST / target_name
        if target_name in AUTHORED_THIS_WAVE or target.exists():
            entries.append({"source": path.name, "target": target_name,
                            "action": "left alone", "substitutions": {}})
            continue
        raw = path.read_bytes()
        if path.name in HAND_EDIT:
            target.write_bytes(raw)
            target.chmod(path.stat().st_mode)
            entries.append({"source": path.name, "target": target_name,
                            "action": "VERBATIM (hand-edit set)", "substitutions": {}})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            target.write_bytes(raw)
            entries.append({"source": path.name, "target": target_name,
                            "action": "verbatim (binary)", "substitutions": {}})
            continue
        renamed, counts = rename(text)
        target.write_text(renamed, encoding="utf-8", newline="")
        target.chmod(path.stat().st_mode)
        entries.append({"source": path.name, "target": target_name,
                        "action": "renamed", "substitutions": counts})

    manifest = {
        "schema": "neural-forge-matrix-wave45-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 44` / `Wave 44` / `WAVE 44` in English are NOT substituted, "
            "so a docstring recording what wave 44 measured still says wave 44 "
            "did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "ONE MODULE EDITED, TWO CHANGES INSIDE IT, AND NO NEW MODULE.  "
            "`mx45_dualband.py` gains (a) the `writing-concise-email` entry in "
            "PAYLOAD_ANCHORS, which is the named debt `NOTES-MX-AV` section 2.3 "
            "measured and section 12 item 1 ordered, registered at the tree cut "
            "before any wave-45 pass runs, and (b) `response_union_coverage`, a "
            "PUBLISHED column with the response as denominator that DECIDES "
            "NOTHING -- the same shape waves 42 and 43 gave the union measure "
            "before wave 44 promoted it.  Every other carried module keeps the "
            "bytes wave 44 shipped.  Wave 39's six vocabulary-gather lines are "
            "live code this wave -- the build list is TWO `mx45_vocab.py` "
            "gathers -- and wave 36's composed width x MLP path, wave 37's "
            "live-mask repair, wave 38's three multi-index excision lines, wave "
            "40's five tie-builder lines, wave 41's six rowset lines and wave "
            "43's two solver modules are carried and asserted but NOT run."
        ),
        "files": entries,
    }
    out = DST / "wave45-tree-cut-manifest.json"
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="")

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    # NOTES-MX-AO section 9: parse every artifact in the command that writes it.
    reread = json.loads(out.read_text(encoding="utf-8"))
    assert reread["schema"] == manifest["schema"], "manifest schema did not survive the write"
    assert len(reread["files"]) == len(entries), "manifest file list did not survive the write"
    print(f"-> {out} ({len(reread['files'])} files, re-parsed OK)")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-44 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave45.py", "wave45-tree-cut-manifest.json"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for needle, why in FORBIDDEN_AFTER.items():
            hits = text.count(needle)
            allowed = ALLOWED_HISTORICAL.get((path.name, needle))
            if allowed is not None:
                bound, how, reason = allowed
                ok = hits == bound if how == "==" else hits <= bound
                if ok:
                    print(f"  declared historical reference: {path.name} "
                          f"{needle!r} x{hits} ({how}{bound}) -- {reason}")
                    continue
                problems.append(
                    f"{path.name}: {needle!r} appears {hits} times, declared "
                    f"{how}{bound} -- {reason}"
                )
                continue
            if hits:
                problems.append(f"{path.name}: {why} ({needle!r} x{hits})")
    # The seal verifier must STILL carry every earlier wave, or the hand edit did
    # not happen and the rename did.  TWELVE roots are checked by name.
    for name, needle in (("mx45_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave39" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave40" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave41" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave42" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave43" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave44" / "cells"'),
                         ("mx45_verify_seals.py", 'ROOT / "wave45" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx45_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave45-residual-basis-v1",
                   "neural-forge-matrix-wave44-residual-basis-v1",
                   "neural-forge-matrix-wave43-residual-basis-v1",
                   "neural-forge-matrix-wave42-residual-basis-v1",
                   "neural-forge-matrix-wave41-residual-basis-v1",
                   "neural-forge-matrix-wave40-residual-basis-v1",
                   "neural-forge-matrix-wave39-residual-basis-v1",
                   "neural-forge-matrix-wave38-residual-basis-v1",
                   "neural-forge-matrix-wave37-residual-basis-v1",
                   "neural-forge-matrix-wave36-residual-basis-v1",
                   "neural-forge-matrix-wave35-residual-basis-v1",
                   "neural-forge-matrix-wave34-residual-basis-v1",
                   "neural-forge-matrix-wave33-residual-basis-v1",
                   "neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx45_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # ------------------------------------------------------------------------
    # WAVE 43'S ADDITION, CARRIED AND WIDENED BY THIS WAVE'S SEVEN STRINGS.
    # There are TWO append-only accept lists in this campaign.
    # `BASIS_SCHEMAS_ACCEPTED` lives in the builder and is checked above; the
    # SEALER's `BUILD_RECEIPT_SCHEMAS` lives in
    # `experiments/seal_matrix_architecture_child.py`, is OUTSIDE the tree the
    # cutter copies, and every wave since 36 appended it by hand at the cut.
    # Wave 43's README claimed the append had happened when it had not, the
    # sealer refused all three of that wave's reviews, and wave 43 paid an extra
    # deployment for it before adding this check.  Checked here for the third
    # time, ten strings deep.
    sealer = (Path(__file__).resolve().parents[4]
              / "experiments" / "seal_matrix_architecture_child.py")
    if not sealer.is_file():
        problems.append(
            f"{sealer} is missing: the sealer's own accept list cannot be audited "
            "and the only thing standing behind this wave's append would be prose"
        )
    else:
        body = sealer.read_text(encoding="utf-8")
        for needle in ("neural-forge-matrix-wave45-standard-shape-child-v1",
                       "neural-forge-matrix-wave45-blend-build-receipt-v1",
                       "neural-forge-matrix-wave45-logit-metric-writer-v1",
                       "neural-forge-matrix-wave45-body-blend-build-receipt-v1",
                       "neural-forge-matrix-wave45-tied-vocabulary-child-v1",
                       "neural-forge-matrix-wave45-rowset-vocabulary-child-v1",
                       "neural-forge-matrix-wave45-rope-band-child-v1",
                       "neural-forge-matrix-wave44-standard-shape-child-v1",
                       "neural-forge-matrix-wave43-standard-shape-child-v1",
                       "neural-forge-matrix-wave42-standard-shape-child-v1"):
            if needle not in body:
                problems.append(
                    f"seal_matrix_architecture_child.py: BUILD_RECEIPT_SCHEMAS is "
                    f"missing {needle}.  This list is append-only and lives "
                    "OUTSIDE the cut tree; a wave that forgets it gets every "
                    "review refused after the builds have already run"
                )
    # ------------------------------------------------------------------------
    # WAVE 45'S OWN ADDITION, AND IT IS THE ITEM `NOTES-MX-AV` SECTION 12.6
    # NAMED.  Waves 43 and 44 closed the two accept lists and the three sealq.sh
    # path resolutions.  The next file a wave depends on and no cut has ever
    # read is `experiments/seal_matrix_behavior_review.py` -- and reading it
    # leads somewhere worse.  The flags `seal_verdict.sh` passes are pinned by
    # wave 34's check below, but the NUMBERS THAT TURN A COUNT INTO A VERDICT
    # are not in the sealer at all: `behavior_verdict_for` reads
    # PARENT_REVIEW_BAND, A9_REVIEW_BAND and PARENT_BAND_TOLERANCE out of
    # `src/neural_forge/matrix_cell_receipt.py`, which is outside the cut tree,
    # outside the sealer, and has never been asserted by anything in this
    # campaign.  Every `alive-parent-band` / `alive-degraded` / `impaired`
    # label in 1,052 ledger rows rests on those three integers.
    #
    # They are pinned BY VALUE and not merely by presence, because a constant
    # that is present at a different value is exactly the failure a presence
    # check cannot see.
    reviewer = (Path(__file__).resolve().parents[4]
                / "experiments" / "seal_matrix_behavior_review.py")
    if not reviewer.is_file():
        problems.append(
            f"{reviewer} is missing: the sealer this wave's every verdict goes "
            "through cannot be audited"
        )
    else:
        body = reviewer.read_text(encoding="utf-8")
        for needle, want, why in (
            ('        default="coherent",\n', 1,
             "--smoke-verdict's default.  seal_verdict.sh passes it explicitly, "
             "so a changed default would not show up there -- but reviewonly.sh "
             "and any future caller inherit it, and `coherent` is one of the "
             "four readings behavior_verdict_for turns into `dead`"),
            ('        default="a6review",\n', 1,
             "--instance-suffix's default.  It is the SECOND half of every "
             "sealed cell_instance_id in this campaign; the ledger's identity "
             "column is `<tag>-0001-a6review` and a changed default silently "
             "forks the id space"),
            ("        default=16,\n", 1,
             "--lanes-total's default.  Fifteen graded lanes plus safety; "
             "per_lane_degeneracy_reading compares the charged count against it "
             "to decide `general`, which is death under either reading"),
            ('    parser.add_argument("--parent-useful-or-partial", type=int)\n', 1,
             "the parent calibration flag has NO default and must not acquire "
             "one: a wave that forgets to pass it must fail, not inherit a "
             "number"),
        ):
            got = body.count(needle)
            if got != want:
                problems.append(
                    f"seal_matrix_behavior_review.py: {needle!r} appears {got} "
                    f"times, expected {want} -- {why}"
                )
    receipt = (Path(__file__).resolve().parents[4]
               / "src" / "neural_forge" / "matrix_cell_receipt.py")
    if not receipt.is_file():
        problems.append(
            f"{receipt} is missing: the three band constants every behavioural "
            "verdict in this campaign is computed from cannot be audited"
        )
    else:
        body = receipt.read_text(encoding="utf-8")
        for needle, want, why in (
            ("PARENT_REVIEW_BAND = 12\n", 1,
             "THE PARENT BAND.  With the tolerance below it sets the "
             "alive-parent-band floor at 10/15.  Never asserted by any cut "
             "before wave 45"),
            ("A9_REVIEW_BAND = 8\n", 1,
             "the alive-degraded floor.  A child at 8..9 is alive-degraded and "
             "at 7 is impaired, and nothing has ever checked the 8"),
            ("PARENT_BAND_TOLERANCE = 2\n", 1,
             "ADDENDUM 2's measured reviewer-sensitivity interval.  It is the "
             "one constant in the chain that is an EMPIRICAL claim rather than "
             "a definition, which makes it the one most likely to be 'improved'"),
            ("    if score >= PARENT_REVIEW_BAND - PARENT_BAND_TOLERANCE:\n", 1,
             "the band arithmetic itself, pinned at its indented form so a "
             "refactor cannot move the comparison while leaving the constants"),
            ('    if degenerate or not coherent or emit_zero_floor_passed is False:\n', 1,
             "THE DEATH LINE.  Degeneracy, incoherence and the sealed wreck "
             "detector are each independently fatal, and every deep-band child "
             "in this campaign is `dead` through it"),
            ('BEHAVIOR_VERDICTS = (\n', 1,
             "the verdict vocabulary is ONE tuple; the sealer's --verdict "
             "choices are this tuple and nothing else"),
            ('DEGENERACY_READINGS = ("none", "lane-scoped", "general", "unstated")\n', 1,
             "ADDENDUM 9 9.1's four scopes, pinned as a single line so an "
             "append cannot pass a presence check"),
        ):
            got = body.count(needle)
            if got != want:
                problems.append(
                    f"matrix_cell_receipt.py: {needle!r} appears {got} times, "
                    f"expected {want} -- {why}"
                )
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.
    for needle, want, why in (
        ('(outdir / "mx45-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
        ('(outdir / "mx44-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx43-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx42-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx41-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx40-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx39-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx38-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx37-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx36-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx35-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx34-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit, exactly as it "
         "survived waves 32 and 33"),
        ('(outdir / "mx32-build-receipt.json").write_text', 0,
         "the ORIGINAL stale name, kept pinned at zero because it is the one "
         "that actually shipped for two waves"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx45_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # WAVE 39'S ONE EDIT, CARRIED AND STILL PINNED IN BOTH DIRECTIONS.  Wave 45
    # does not run this file either; the pin is kept for the reason waves 41
    # through 44 kept it, which is that the module is in the tree and a later
    # wave runs whatever this cut leaves behind.
    for needle, want, why in (
        ('    print(f"[mx45] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave even in a wave that does "
         "not run it, or the next wave that does inherits a stale tag"),
        ('    print(f"[mx44] {message}", flush=True)\n', 0,
         "last wave's prefix, which the substitution list retargets; pinned at "
         "zero so a hand edit cannot leave it behind"),
        ('    print(f"[mx43] {message}", flush=True)\n', 0,
         "the wave-43 prefix, pinned at zero"),
        ('    print(f"[mx42] {message}", flush=True)\n', 0,
         "the wave-42 prefix, pinned at zero"),
        ('    print(f"[mx32] {message}", flush=True)\n', 0,
         "the wave-32 prefix that shipped for seven waves; pinned at zero "
         "permanently"),
        ('    print(f"[mx16] {message}", flush=True)\n', 0,
         "wave 17's form of the same defect, pinned at zero so the repair cannot "
         "regress to an even older tag"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx45_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 45" not in body:
            problems.append(f"{name} does not carry --campaign-wave 45")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 45 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # ------------------------------------------------------------------------
    # WAVE 44'S ADDITION, CARRIED AND EXERCISED FOR THE SECOND TIME.  `sealq.sh`
    # is copied VERBATIM, so the three paths it resolves for itself -- the
    # wave's env.sh, the wave's per-case writer and the wave's seal-verdict
    # script -- have to be retargeted by hand every single cut, and until wave
    # 44 only one of the three was reachable by any needle.
    for needle, want, why in (
        (". /home/cereal/nf-mx45/env.sh\n", 1,
         "sealq.sh must source THIS wave's env.sh.  env.sh exports $W, the wave "
         "root every artifact is written under, so a stale source line writes "
         "this wave's cells into last wave's wave root"),
        (". /home/cereal/nf-mx44/env.sh\n", 0,
         "last wave's env.sh, pinned at zero.  This is the pin wave 43 said it "
         "had and did not, and wave 44 built"),
        ("$MX45/mx45_percase.py", 1,
         "the per-case writer must be THIS wave's; the per-case file is the "
         "artifact the sealer refuses a disagreeing count against"),
        ("$MX44/", 0,
         "last wave's module directory variable, pinned at zero.  `$MX44` is "
         "not reachable by the `MX44-` needle, which is why this pin exists"),
        ("$MX45/seal_verdict.sh", 1,
         "the seal-verdict script must be THIS wave's; it carries the wave root "
         "and the parent calibration flag"),
    ):
        got = seals.count(needle)
        if got != want:
            problems.append(
                f"sealq.sh: {needle!r} appears {got} times, expected {want} -- {why}"
            )
    # WAVE 40'S PIN, CARRIED.  `--audit` runs immediately after the cut and
    # before anything builds, when the correct number of seal calls is ZERO; the
    # wave appends its own after it reads its dossiers and does not re-run this.
    # `seal () {` is the function definition and is not a call, so the needle
    # requires an identifier char.  WAVE 44 CARRIED THREE CALLS HERE.
    calls = re.findall(r"(?m)^seal [A-Za-z0-9]", seals)
    if calls:
        problems.append(
            f"sealq.sh carries {len(calls)} seal CALL(s) at cut time, expected 0. "
            "A verbatim copy brings last wave's calls forward with every count "
            "intact, and rewriting the marker above them does not remove them"
        )
    # WAVE 42'S ADDITION, EXERCISED FOR THE THIRD TIME.  The three literal
    # needles are the NAMED REGRESSION -- wave 39's exact sentence, dossier and
    # counter, which survived two cuts -- and they are kept because a regression
    # pin that stops being checked is a pin that stops working.  They are NOT
    # the protection: the general pin below is.
    for needle, why in (
        ("# AND IT WAS RUN AND IT IS EXACT.",
         "an inherited CALIBRATION RESULT survived the cut.  This wave has not "
         "run its calibration at cut time, so a sentence asserting that it did "
         "is last wave's -- or, as it turned out in waves 40 and 41, wave 39's"),
        ("ARCH.V14CL896X13's dossier",
         "wave 39's named calibration dossier survived the cut.  The wave writes "
         "the dossier it actually read"),
        ("TWELFTH consecutive calibration",
         "an inherited calibration COUNTER survived the cut; wave 41's own "
         "NOTES record the fourteenth while this file still said the twelfth"),
    ):
        hits = seals.count(needle)
        if hits:
            problems.append(
                f"sealq.sh carries {needle!r} x{hits} at cut time, expected 0 -- {why}"
            )
    # THE GENERAL PIN, exercised for the THIRD time.  Wave 43 reported 78 lines,
    # wave 44 reported 62.  The marker block ends with a line that says the
    # wave's own calibration is written BELOW it after the dossiers are read, and
    # at cut time the only thing that may follow that line is blank lines and the
    # closing echo.
    sentinel = ("# --- wave 45's own calibration, byte comparison and reading order "
                "are written\n# --- below this line AFTER the dossiers are read, and "
                "not before.\n")
    if sentinel not in seals:
        problems.append(
            "sealq.sh is missing the calibration sentinel.  Without it there is "
            "no line to measure 'what this wave has not written yet' against, "
            "and the three literal needles above are the only protection -- "
            "which is exactly the protection that failed for two waves"
        )
    else:
        tail = seals.split(sentinel, 1)[1]
        residue = [line for line in tail.splitlines()
                   if line.strip() and line.strip() != 'echo "SEALQ DONE"']
        if residue:
            problems.append(
                f"sealq.sh carries {len(residue)} line(s) of prose or code after "
                "the calibration sentinel at cut time, expected 0.  Everything "
                "below that line is a reading THIS wave has not taken yet, so at "
                "the cut it can only be last wave's.  First line: "
                f"{residue[0].strip()[:80]!r}"
            )
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    if not (DST / "tieq.sh").is_file():
        problems.append("tieq.sh is missing: wave 40's build module has no driver")
    if not (DST / "rowsetq.sh").is_file():
        problems.append("rowsetq.sh is missing: wave 41's build module has no driver")
    if not (DST / "dualbandq.sh").is_file():
        problems.append("dualbandq.sh is missing: this wave's edited module has no driver")
    if not (DST / "planeq.sh").is_file():
        problems.append("planeq.sh is missing: wave 43's plane solver has no driver")
    if not (DST / "xchangeq.sh").is_file():
        problems.append("xchangeq.sh is missing: wave 43's go-gate census has no driver")
    if not (DST / "extq.sh").is_file():
        problems.append(
            "extq.sh is missing: the ADDENDUM 16.1 extension surface has no "
            "driver, and wave 45 RUNS it -- the debt NOTES-MX-AV section 10.6 "
            "declined twice"
        )
    # WAVE 34's ADDITION, carried unchanged.
    for name in ("mx45_extscore.py", "mx45_extrubric2.py", "mx45_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx45_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx45_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.
    pen = (DST / "mx45_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx45_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S, 37'S AND 38'S ADDITIONS.  Wave 45 builds NO body through this
    # module, so these pins guard dormant code.  They are kept for wave 41's
    # stated reason: the module is in the tree and a future wave will run it, and
    # dropping a pin because a wave does not exercise it is how wave 30 deleted a
    # seal root.
    for needle, want, why in (
        ("--kv-groups and --head-dim both rewrite the attention block", 1,
         "the head-count/head-dim refusal must survive"),
        ("--hidden-size already re-solves every reader and every writer", 1,
         "the SURVIVING half of wave 35's refusal -- --hidden-size with "
         "--head-dim -- must still fire in those words"),
        ("def narrow_mlp_site(", 1,
         "the composed width x MLP site is wave 36's whole build; without it "
         "the guard was lifted and nothing was put in its place"),
        ("COMPOSED_READER_EXACTNESS = ", 1,
         "the composed selection rests on the reader fold being EXACT against "
         "the parent's gated intermediate at the lifted row"),
        ("the intermediate-axis identity does not hold", 2,
         "wave 14 3's identity precondition must be checked on the CHILD's own "
         "gated intermediate too, not only on the parent's"),
        ("harness.model.config.layer_types = [", 1,
         "wave 18's live-mask repair re-derives layer_types after an excision so "
         "the LIVE model's causal mask follows the LIVE blocks"),
        ("so the live mask follows the live block", 1,
         "the repair must still SAY what it is doing in the log line the build "
         "log carries, or a wave that reads the log cannot tell whether it ran"),
        ("        for index in sorted(excised, reverse=True):\n", 1,
         "a multi-index excision must delete in DESCENDING order or the second "
         "del addresses a list the first one shortened"),
        ("                layer.self_attn.layer_idx = position\n", 1,
         "every surviving attention module must be renumbered to its NEW slot or "
         "it addresses the KV cache the parent had"),
        ("                layer.conv.layer_idx = position\n", 1,
         "every surviving conv module must be renumbered to its NEW slot or it "
         "addresses the conv state the parent had"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx45_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ADDITION, CARRIED -- and wave 45 runs this module TWICE, at the
    # two rungs that take the CL768 ladder to within a factor of 1.34 of its own
    # structural floor.
    vocab = (DST / "mx45_vocab.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "the identity gate for a ZERO-SOLVE arm: every kept row is the source's "
         "own bytes, asserted with torch.equal and not a tolerance"),
        ("        if not torch.equal(gathered[probe], table[order[probe]]):\n", 1,
         "the four-row spot check that runs before the whole-table comparison"),
        ("    closure_state = merge_closure_intact(vocab, keptset)\n", 1,
         "the STRUCTURAL half of the census gate -- the half with no flag.  Wave "
         "45 builds at K=1,024 against a 765-row mandatory set, which leaves 259 "
         "discretionary rows and is the closest this campaign has ever come to "
         "that floor, so this is the gate that decides whether the axis ends "
         "by measurement or by refusal"),
        ("    if lost and not args.below_floor:\n", 1,
         "the COVERAGE half, and the guard the --below-floor flag disarms.  "
         "Wave 45 passes that flag on both rungs it builds, and a wave that "
         "passes a flag must assert the guard the flag disarms or it cannot tell "
         "disarming from absence"),
        ("    if mandatory_size > k:\n", 1,
         "THE STRUCTURAL FLOOR'S OWN REFUSAL, and wave 45 is the first wave to "
         "build close enough to it that this line is a live risk rather than a "
         "formality.  It is pinned at its indented form because a wave that "
         "wants the refusal to be the result has to be able to prove the "
         "refusal was reachable"),
        ('            "the child tokenizer is not lossless on text; the byte alphabet or "\n', 1,
         "the refusal behind decode(child_encode(t)) == t"),
        ('    ranking = sorted(\n', 1,
         "`choose_kept`'s discretionary ranking.  Wave 41 closed the question of "
         "WHICH rows with three built arms; this wave moves HOW MANY twice more "
         "on the same call, and the call has to be the same one"),
    ):
        got = vocab.count(needle)
        if got != want:
            problems.append(
                f"mx45_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVES 40 AND 41'S ADDITIONS, CARRIED.  Neither runs this wave; both are
    # pinned, for the reason stated above.
    tie = (DST / "mx45_vocabtie.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(head_rows, table[head_index]):\n", 1,
         "the HEAD of a tie child is a verbatim gather; asserted with "
         "torch.equal and not a tolerance"),
        ("    if covered != len(tail_ids):\n", 1,
         "the tie's WHOLE CLAIM is that every dropped id has an image"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control"),
        ("    if decoded_probe != decoded_probe_again:\n", 1,
         "the id->token map must be a FUNCTION"),
        ("    if merges_kept != expected_merges:\n", 1,
         "the tokenizer-identity claim, checked by two implementations agreeing"),
    ):
        got = tie.count(needle)
        if got != want:
            problems.append(
                f"mx45_vocabtie.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    rowset = (DST / "mx45_vocabrowset.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "a rowset child is a VERBATIM gather exactly as the deletion arm is"),
        ("    if len(order) != len(sealed_order):\n", 1,
         "the arms are compared at FIXED K"),
        ("    if mandatory - keptset:\n", 1,
         "the mandatory set is held EXACTLY as sealed"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control"),
        ("    if closure_state[\"intact\"] is not True:\n", 1,
         "the STRUCTURAL half of the census gate is not releasable on a rowset "
         "arm any more than on a gather"),
        ("    if arm == \"C\":\n", 2,
         "the leverage arm is BUILT, not predicted; two load-bearing sites"),
    ):
        got = rowset.count(needle)
        if got != want:
            problems.append(
                f"mx45_vocabrowset.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 43'S TWO SOLVER MODULES, CARRIED AND DORMANT.  Every pin wave 43 put
    # on them is carried unchanged: a module whose axis is closed is exactly the
    # module a later cut is most likely to damage silently.
    plane = (DST / "mx45_plane.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(new_q.weight.data, wq_src[q_rows]):\n", 1,
         "THE Q ROWS ARE VERBATIM.  A band fold's entire legality argument is "
         "that the kept planes are the parent's own rows, asserted with "
         "torch.equal and not a tolerance"),
        ("    if not torch.equal(new_k.weight.data, wk_src[k_rows]):\n", 1,
         "the same assertion on the k side, which is a DIFFERENT row map because "
         "there are eight kv heads and sixteen query heads"),
        ("    if not torch.equal(new_v.weight.data, wv_src[v_rows]):\n", 1,
         "THE VALUE CHANNELS ARE A VERBATIM SUBSET TOO"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control"),
        ("        if abs(theta_child / expected_theta - 1.0) > THETA_TOLERANCE:\n", 1,
         "THE LADDER IS THE FOLD, checked over TWO independent sources"),
        ("            if float(scale) <= 1.0:\n", 1,
         "THE LAW is a veto on update scale and it is fail-closed"),
        ("    if census_only:\n", 1,
         "THE CENSUS RUNS WITHOUT THE BUILD"),
        ("    energy = torch.zeros(planes, dtype=torch.float64)\n", 1,
         "the census accumulator is fp64 and is wave 15's own measure, unchanged"),
    ):
        got = plane.count(needle)
        if got != want:
            problems.append(
                f"mx45_plane.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    xchange = (DST / "mx45_xchange.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("GO_GATE = {\n", 1,
         "THE THRESHOLDS ARE ONE DECLARED TABLE, and this one closed a class"),
        ("    if not torch.allclose(delta_check, delta, rtol=1e-4, atol=1e-4):\n", 1,
         "THE RECONSTRUCTION GATE"),
        ("        if verdict is False:\n", 1,
         "the census REPORTS A VERDICT"),
        ("    \"this census MAY close the XCHANGE class\"", 1,
         "and it says so in its own artifact, not only in the wave's notes"),
        ("        rho_2 = float(eps.pow(2).sum() / delta.pow(2).sum())\n", 1,
         "rho_2 is the share of the conv delta that is PROVABLY outside the "
         "linear-value attention class"),
    ):
        got = xchange.count(needle)
        if got != want:
            problems.append(
                f"mx45_xchange.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 44'S PINS ON THE DUAL-BAND READER, CARRIED UNCHANGED, PLUS WAVE 45'S
    # TWO ADDITIONS.  Everything wave 44 pinned about the union measure's
    # promotion stays pinned: the promotion is not re-argued by the wave that
    # completes its anchor table.
    dual = (DST / "mx45_dualband.py").read_text(encoding="utf-8")
    for needle, want, why in (
        # --- wave 44's pins, carried verbatim ---------------------------------
        ("        if column_a != int(sealed_count):\n", 1,
         "THE REFUSAL THAT MAKES COLUMN A THE SEALED COLUMN"),
        ("        if lane_ask not in (\"PRESERVE\", \"SELECT\", \"PRODUCE\"):\n", 1,
         "every lane must carry a DECLARED ask class"),
        ("                    if classification == \"COPY_BACK\" and lane_ask != \"PRESERVE\":\n", 1,
         "THE EXCEPTION CHECK against wave 40's two sealed precedents"),
        ("    if len(LANE_ASK) != 16:\n", 1,
         "all sixteen lanes of the frozen surface must be classified"),
        ("            if not payload:\n", 1,
         "an unresolvable payload anchor must SAY so rather than score zero"),
        ("            if sealed_verdict in (\"Useful\", \"Partial\"):\n", 1,
         "only a HELD lane can be voided by the second column"),
        ("COPY_BACK_UNION_COVERAGE = 0.60", 1,
         "THE PRE-REGISTERED PRIMARY THRESHOLD, declared ONCE, and NOT retuned "
         "by the wave that changes what it is applied to.  Moving an anchor and "
         "a threshold in one wave would make neither falsifiable"),
        ("    if union_cov >= COPY_BACK_UNION_COVERAGE:\n", 1,
         "THE PRIMARY DECISION LINE.  The union measure decides, and it is "
         "pinned at its indented form so a later cut cannot eat the branch and "
         "leave the constant behind"),
        ("    if payload_coverage >= COPY_BACK_PAYLOAD_COVERAGE:\n", 0,
         "THE DEMOTED LINE, pinned at ZERO.  The longest-run measure is still "
         "computed and still published as `legacy_longest_run_classification`, "
         "but it must not decide anything"),
        ("            legacy = legacy_classification(verbatim, payload, termination)\n", 1,
         "the legacy measure must still be COMPUTED on every lane, pinned at "
         "its INDENTED form -- wave 38's lesson 6"),
        ("    if legacy_cov >= COPY_BACK_PAYLOAD_COVERAGE:\n", 1,
         "the legacy classifier keeps its own threshold line, in its own "
         "function, under its own variable name"),
        ("UNION_MIN_BLOCK = 8", 1,
         "the union measure's minimum block length, unchanged from the value "
         "waves 42, 43 and 44 published their numbers at"),
        # --- WAVE 45'S FIRST ADDITION: THE ORDERED ANCHOR ---------------------
        ('    "writing-concise-email": ("under 90 words", None),\n', 1,
         "THE NAMED DEBT, PAID AT THE CUT.  NOTES-MX-AV section 2.3 measured "
         "seven void-changing disagreements on this lane and section 12 item 1 "
         "ordered the anchor registered BEFORE the pass that uses it.  The cut "
         "point is the campaign's own anchor rule -- strip the leading task "
         "directive, keep the material -- and NOT a cut chosen to move a "
         "number; run/mx45/README.md section 2 registers, before the pass, that "
         "it does NOT close the seven rows and states the alternative cut it "
         "refuses and why"),
        ('    "summarization-priorities": ("preserving the decision and the unresolved risk:", None),\n', 1,
         "the anchor whose payload decides the deep band, unchanged.  A wave "
         "that adds an anchor must assert it did not edit the existing ones, or "
         "'the anchor table changed' and 'the floor moved' are the same event"),
        ('    "context-retrieval": ("NOTE:", "Question:"),\n', 1,
         "the two-sided anchor, unchanged"),
        ('    "debugging-off-by-one": ("def last_three", None),\n', 1,
         "the code anchor, unchanged"),
        ("    if len(PAYLOAD_ANCHORS) != 4:\n", 1,
         "THE ANCHOR TABLE'S OWN SIZE PIN, live in the module and not only in "
         "the cutter.  Wave 44 shipped three anchors and this wave ships four; "
         "the count is asserted where the table is used, because a fifth anchor "
         "added quietly would re-classify a lane in a wave that never mentioned "
         "it"),
        # --- WAVE 45'S SECOND ADDITION: A PUBLISHED, NON-DECIDING COLUMN ------
        ("            response_union = union_coverage(\n"
         "                verbatim, lane_whole_prompt(resp.get(\"prompt_messages\", [])))\n", 1,
         "THE RESPONSE-SIDE COLUMN, computed on every lane, at its indented "
         "form.  Same union, same UNION_MIN_BLOCK, same 0.60 numeral -- the "
         "denominator is the RESPONSE instead of the payload, which is the "
         "question 'how much of what this model said is lifted from the "
         "prompt?' rather than 'how much of the supplied material came back?'. "
         "AND ITS SOURCE IS THE WHOLE PROMPT, NOT THE ANCHORED PAYLOAD, pinned "
         "here because measuring it against the anchored slice would hide the "
         "instruction-text lift that is the only thing it exists to see -- a "
         "defect this wave made and caught before the cut was committed"),
        ("def lane_whole_prompt(prompt_messages: list) -> str:\n", 1,
         "the unanchored source is a NAMED function and not an inline "
         "re-derivation, so the payload path and the response path cannot drift "
         "apart about what 'the prompt' is"),
        ("RESPONSE_SIDE_DECIDES_NOTHING = True", 1,
         "AND IT DECIDES NOTHING, DECLARED IN ONE CONSTANT.  Waves 42 and 43 "
         "published the union measure beside the deciding one for two waves "
         "before wave 44 promoted it; this column enters on exactly that "
         "protocol and a wave that wants to promote it has to say so at a cut"),
        ("    if response_union >= COPY_BACK_UNION_COVERAGE:\n", 0,
         "THE NON-DECISION, PINNED AT ZERO.  There is no branch anywhere in "
         "this module that lets the response-side number change a "
         "classification, and the only way to assert that is to assert the "
         "branch does not exist"),
    ):
        got = dual.count(needle)
        if got != want:
            problems.append(
                f"mx45_dualband.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-44 identifier survived; TWELVE seal roots declared; "
          "the builder's accept list kept seventeen basis schemas AND THE "
          "SEALER'S OWN BUILD_RECEIPT_SCHEMAS carries all seven of this wave's "
          "strings; THE BEHAVIOURAL SEALER'S THREE ARGPARSE DEFAULTS AND THE "
          "THREE BAND CONSTANTS EVERY VERDICT IN 1,052 LEDGER ROWS IS COMPUTED "
          "FROM ARE PINNED BY VALUE FOR THE FIRST TIME IN FORTY-FIVE WAVES; "
          "sealq.sh sources THIS wave's env.sh, per-case writer and "
          "seal-verdict script and carries this wave's marker, ZERO seal calls "
          "and ZERO lines after the calibration sentinel; gainq.sh, tieq.sh, "
          "rowsetq.sh, dualbandq.sh, planeq.sh, xchangeq.sh and extq.sh exist; "
          "both extension surfaces have a scorer and v1 still names the v1 "
          "bank; the penalty diagnostic kept its NOT_A_FOLD banner and its "
          "identity refusal; wave 36's composed width x MLP path, wave 37's "
          "live-mask repair, wave 38's three multi-index excision lines, wave "
          "40's five tie-builder lines, wave 41's six rowset lines and wave "
          "43's thirteen solver lines are present but DORMANT this wave; the "
          "builder's log prefix names THIS wave; all seven vocabulary-gather "
          "lines are present at their indented form and ARE live code this "
          "wave, INCLUDING the structural-floor refusal this wave builds close "
          "enough to reach; and THE ANCHOR TABLE IS FOUR ENTRIES with the "
          "three carried ones asserted unedited, while the response-side "
          "column is computed, published and pinned at ZERO decisions")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
