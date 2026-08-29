#!/usr/bin/env python3
"""Cut wave 44's tree from wave 43's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-43 cutter, in full.**

1. ``SRC`` is the wave-43 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  Wave 40's seal-call pin -- ``audit()``
   counts ``^seal <TAG>`` and requires **zero at cut time** -- is carried, and
   wave 44 is the FOURTH cut where it had work to do from the start: this
   file's copy of ``sealq.sh`` arrived carrying wave 43's **three** seal calls
   with every count intact, and wave 42's general calibration pin reported
   **62** residue lines below the sentinel.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README.  **Wave
   44 authors no new module and EDITS ONE CARRIED MODULE**, which is the
   opposite shape from wave 43 and is stated here rather than left to be
   inferred from a diff.
4. **THE EDIT IS ``mx44_dualband.py`` AND IT IS THE WHOLE POINT OF THE WAVE.**
   ``payload_union_coverage`` becomes the PRIMARY copy-back measure and the
   longest-contiguous-run measure is demoted to a published legacy column.
   `NOTES-MX-AT` §11.3 told wave 43 to do this, `NOTES-MX-AU` §8.1 and §13
   item 3 recorded that it did not and what that cost, and this cut is the
   first thing wave 44 does, BEFORE any wave-44 decode exists.  Every
   load-bearing line of the new classifier is pinned below at its INDENTED
   form, and the demoted line is pinned at the count that proves it no longer
   decides anything.

**WAVE 44'S OWN ADDITION TO THIS AUDIT, AND IT IS A DEFECT NOBODY HAD MEASURED.**
`NOTES-MX-AU` §13 item 6 said the natural next place to look is every file a
wave edits BY HAND each time with nothing but prose behind it.  The nearest one
is not another file: it is three lines INSIDE ``sealq.sh``.  Wave 43's own
sealq.sh records that "THE THREE PATH RESOLUTIONS a verbatim copy cannot fix
for itself -- last wave's env.sh, last wave's per-case writer and last wave's
seal-verdict script -- were retargeted by hand at pinned counts of one each",
and **two of those three pins do not exist**.  ``FORBIDDEN_AFTER``'s needles are
``mx43_`` and ``MX43-``; ``nf-mx43/env.sh`` carries ``mx43/`` and
``$MX43/seal_verdict.sh`` carries ``MX43/``, and neither needle reaches either
one.  A verbatim ``sealq.sh`` that kept last wave's ``env.sh`` would source last
wave's ``$W`` and write this wave's cells into last wave's wave root, and no
check in the campaign would have said so.  ``audit()`` now pins all three by
name, in both directions.

**THE NEEDLES CARRY THEIR INDENTATION** -- wave 38's lesson 6, carried, and
applied to the edited module's new lines from the start rather than after the
first cut eats one.

**WHAT THIS CUT DOES NOT DO.**  It does not author a NORM-ESTATE Arm-M/Arm-S
solver, and it does not unblock ALIAS or ROWSET Arms A and B -- those wait on a
ruling and on a corpus respectively, which are not authoring problems.  It also
builds nothing on the PLANE axis: wave 43 re-sealed that axis under the slate's
own kill gate, and ``mx44_plane.py`` is carried, pinned and dormant.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  **And it parses what it writes, in the invocation that
writes it** -- `NOTES-MX-AO` §9's rule, obeyed by the first file wave 44 writes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx43"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx43_verify_seals.py",
    "mx43_retain.py",
    "mx43_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files -- this
#: cutter and the README.  Wave 44's deliverable is an INSTRUMENT CHANGE inside
#: a carried module, not a new module, and declaring two files here rather than
#: six is how that difference is stated in code.
AUTHORED_THIS_WAVE = {
    "cut_wave44.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt.
NOT_CARRIED = {"wave43-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 43", "--campaign-wave 44"),
    ('"campaign_wave": 43', '"campaign_wave": 44'),
    ("wave43", "wave44"),
    ("mx43", "mx44"),
    ("MX43", "MX44"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.
FORBIDDEN_AFTER = {
    "mx43_": "a wave-43 module reference survived the copy",
    "MX43-": "a wave-43 child directory name survived the copy",
    "/wave43/": "a wave-43 artifact path survived the copy",
    "--campaign-wave 43": "a driver would seal this wave's build as wave 43",
}
#: DELIBERATE historical references, declared with a count.  The third field is
#: the comparison: an exact pin where the count is knowable before the wave
#: runs, a bound where it is not.
ALLOWED_HISTORICAL = {
    ("mx44_build.py", "mx43_"): (
        1,
        "==",
        "the wave-44 header names `mx43_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx44_vocab.py", "mx43_"): (
        1,
        "<=",
        "the wave-44 header may name `mx43_vocab.py` as the file it states its "
        "diff against.  Bounded rather than pinned because this wave makes no "
        "edit to that module and may add no header at all",
    ),
    ("mx44_dualband.py", "mx43_"): (
        1,
        "<=",
        "THIS WAVE'S ONE EDITED MODULE.  Its new header states its diff against "
        "`mx43_dualband.py`, which is the file it was copied from and the file "
        "whose measure it demotes.  Bounded at one and not pinned, because the "
        "header is prose and a second sentence naming the same file would be a "
        "style change and not a defect",
    ),
    ("mx44_retain.py", "MX43-"): (
        3,
        "<=",
        "the retention rule requires each DROP reason to name the retained "
        "source that rebuilds the deleted child.  Wave 43 retained two children "
        "-- MX43-V5CL768 and MX43-HD48 -- and wave 44's CL768 rungs sit directly "
        "below MX43-V5CL768 on the same ladder, so a legitimate cross-reference "
        "is a note and not a refusal",
    ),
    ("sealq.sh", "MX43-"): (
        4,
        "<=",
        "a seal note naming a checkpoint a child is read against.  Wave 44's "
        "vocabulary sources are wave 40's retained MX40-CL768 and wave 35's "
        "MX35-CL896X13, and its nearest sealed control is wave 43's retained "
        "MX43-V5CL768 -- the rung immediately above two of this wave's three "
        "cells -- which a seal note legitimately carries",
    ),
    ("mx44_plane.py", "MX43-"): (
        0,
        "==",
        "the plane solver folds the PARENT and names no wave-43 child at all.  "
        "Carried from wave 43's cutter unchanged and pinned at zero rather than "
        "left undeclared, although wave 44 does not run this module: the axis is "
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
        "schema": "neural-forge-matrix-wave44-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 43` / `Wave 43` / `WAVE 43` in English are NOT substituted, "
            "so a docstring recording what wave 43 measured still says wave 43 "
            "did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "ONE EDIT AND NO ADDITION, which is the opposite shape from wave "
            "43.  `mx44_dualband.py` promotes `payload_union_coverage` to the "
            "PRIMARY copy-back measure and demotes the longest-contiguous-run "
            "measure to a published legacy column.  The threshold is declared "
            "in one constant, its justification is the separation wave 42 "
            "published and committed at 5af0b035 -- copies 0.826972 to "
            "0.997455, everything else at or below 0.483461 over the 34 "
            "distinct summarisation responses in that band -- and the change is "
            "made in the FIRST commit of the wave, before any wave-44 decode "
            "exists, so the measure decides this wave's own children "
            "prospectively.  Every other carried module keeps the bytes wave 43 "
            "shipped.  Wave 39's six vocabulary-gather lines are live code this "
            "wave -- the build list is THREE `mx44_vocab.py` gathers -- and "
            "wave 36's composed width x MLP path, wave 37's live-mask repair, "
            "wave 38's three multi-index excision lines, wave 40's five "
            "tie-builder lines, wave 41's six rowset lines and wave 43's two "
            "solver modules are carried and asserted but NOT run."
        ),
        "files": entries,
    }
    out = DST / "wave44-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-43 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave44.py", "wave44-tree-cut-manifest.json"):
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
    # not happen and the rename did.  ELEVEN roots are checked by name.
    for name, needle in (("mx44_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave39" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave40" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave41" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave42" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave43" / "cells"'),
                         ("mx44_verify_seals.py", 'ROOT / "wave44" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx44_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave44-residual-basis-v1",
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
            problems.append(f"mx44_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # ------------------------------------------------------------------------
    # WAVE 43'S ADDITION, CARRIED AND WIDENED BY THIS WAVE'S SEVEN STRINGS.
    # There are TWO append-only accept lists in this campaign.
    # `BASIS_SCHEMAS_ACCEPTED` lives in the builder and is checked above; the
    # SEALER's `BUILD_RECEIPT_SCHEMAS` lives in
    # `experiments/seal_matrix_architecture_child.py`, is OUTSIDE the tree the
    # cutter copies, and every wave since 36 appended it by hand at the cut.
    # Wave 43's README claimed the append had happened when it had not, the
    # sealer refused all three of that wave's reviews, and wave 43 paid an extra
    # deployment for it before adding this check.  It is checked here for the
    # second time, ten strings deep.
    sealer = (Path(__file__).resolve().parents[4]
              / "experiments" / "seal_matrix_architecture_child.py")
    if not sealer.is_file():
        problems.append(
            f"{sealer} is missing: the sealer's own accept list cannot be audited "
            "and the only thing standing behind this wave's append would be prose"
        )
    else:
        body = sealer.read_text(encoding="utf-8")
        for needle in ("neural-forge-matrix-wave44-standard-shape-child-v1",
                       "neural-forge-matrix-wave44-blend-build-receipt-v1",
                       "neural-forge-matrix-wave44-logit-metric-writer-v1",
                       "neural-forge-matrix-wave44-body-blend-build-receipt-v1",
                       "neural-forge-matrix-wave44-tied-vocabulary-child-v1",
                       "neural-forge-matrix-wave44-rowset-vocabulary-child-v1",
                       "neural-forge-matrix-wave44-rope-band-child-v1",
                       "neural-forge-matrix-wave43-standard-shape-child-v1",
                       "neural-forge-matrix-wave42-standard-shape-child-v1",
                       "neural-forge-matrix-wave41-standard-shape-child-v1"):
            if needle not in body:
                problems.append(
                    f"seal_matrix_architecture_child.py: BUILD_RECEIPT_SCHEMAS is "
                    f"missing {needle}.  This list is append-only and lives "
                    "OUTSIDE the cut tree; a wave that forgets it gets every "
                    "review refused after the builds have already run"
                )
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.
    for needle, want, why in (
        ('(outdir / "mx44-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx44_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # WAVE 39'S ONE EDIT, CARRIED AND STILL PINNED IN BOTH DIRECTIONS.  Wave 44
    # does not run this file either; the pin is kept for the reason waves 41
    # through 43 kept it, which is that the module is in the tree and a later
    # wave runs whatever this cut leaves behind.
    for needle, want, why in (
        ('    print(f"[mx44] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave even in a wave that does "
         "not run it, or the next wave that does inherits a stale tag"),
        ('    print(f"[mx43] {message}", flush=True)\n', 0,
         "last wave's prefix, which the substitution list retargets; pinned at "
         "zero so a hand edit cannot leave it behind"),
        ('    print(f"[mx42] {message}", flush=True)\n', 0,
         "the wave-42 prefix, pinned at zero"),
        ('    print(f"[mx41] {message}", flush=True)\n', 0,
         "the wave-41 prefix, pinned at zero"),
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
                f"mx44_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 44" not in body:
            problems.append(f"{name} does not carry --campaign-wave 44")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 44 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # ------------------------------------------------------------------------
    # WAVE 44'S FIRST ADDITION, AND IT CLOSES A HOLE THAT HAS BEEN OPEN SINCE
    # WAVE 32 PUT `sealq.sh` IN `HAND_EDIT`.  `sealq.sh` is copied VERBATIM, so
    # the three paths it resolves for itself -- the wave's env.sh, the wave's
    # per-case writer and the wave's seal-verdict script -- have to be retargeted
    # by hand every single cut.  Wave 43's own copy says they "were retargeted by
    # hand at pinned counts of one each" and TWO OF THE THREE PINS DID NOT EXIST:
    # `FORBIDDEN_AFTER`'s needles are `mx43_` and `MX43-`, and neither reaches
    # `nf-mx43/env.sh` (which carries `mx43/`) or `$MX43/seal_verdict.sh` (which
    # carries `MX43/`).  Only the per-case writer was covered, and only
    # incidentally, because its filename happens to carry the underscore.
    #
    # What that would have cost is not cosmetic.  `env.sh` exports `$W`, the
    # wave root every cell, shard, log and dossier is written under.  A sealq.sh
    # that sourced last wave's env.sh would have sealed this wave's children
    # into last wave's wave root, under this wave's cell ids, and the seal
    # verifier -- which audits declared roots against disk -- would have found
    # them exactly where it expected not to look.
    for needle, want, why in (
        (". /home/cereal/nf-mx44/env.sh\n", 1,
         "sealq.sh must source THIS wave's env.sh.  env.sh exports $W, the wave "
         "root every artifact is written under, so a stale source line writes "
         "this wave's cells into last wave's wave root"),
        (". /home/cereal/nf-mx43/env.sh\n", 0,
         "last wave's env.sh, pinned at zero.  This is the pin wave 43 said it "
         "had and did not"),
        ("$MX44/mx44_percase.py", 1,
         "the per-case writer must be THIS wave's; the per-case file is the "
         "artifact the sealer refuses a disagreeing count against"),
        ("$MX43/", 0,
         "last wave's module directory variable, pinned at zero.  `$MX43` is "
         "not reachable by the `MX43-` needle, which is why this pin exists"),
        ("$MX44/seal_verdict.sh", 1,
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
    # requires an identifier char.  WAVE 43 CARRIED THREE CALLS HERE.
    calls = re.findall(r"(?m)^seal [A-Za-z0-9]", seals)
    if calls:
        problems.append(
            f"sealq.sh carries {len(calls)} seal CALL(s) at cut time, expected 0. "
            "A verbatim copy brings last wave's calls forward with every count "
            "intact, and rewriting the marker above them does not remove them"
        )
    # WAVE 42'S ADDITION, EXERCISED FOR THE SECOND TIME.  The three literal
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
    # THE GENERAL PIN, exercised for the SECOND time.  Wave 43 was the first cut
    # to run it and it reported exactly the 78 lines `NOTES-MX-AT` section 11.4
    # predicted.  Against `run/mx43/sealq.sh` as shipped it reports 62.  The
    # marker block ends with a line that says the wave's own calibration is
    # written BELOW it after the dossiers are read, and at cut time the only
    # thing that may follow that line is blank lines and the closing echo.
    sentinel = ("# --- wave 44's own calibration, byte comparison and reading order "
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
    # WAVE 34's ADDITION, carried unchanged.
    for name in ("mx44_extscore.py", "mx44_extrubric2.py", "mx44_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx44_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx44_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.
    pen = (DST / "mx44_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx44_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S, 37'S AND 38'S ADDITIONS.  Wave 44 builds NO body through this
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
                f"mx44_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ADDITION, CARRIED -- and wave 44 runs this module THREE TIMES.
    # It is the only CARRIED builder this wave runs.
    vocab = (DST / "mx44_vocab.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "the identity gate for a ZERO-SOLVE arm: every kept row is the source's "
         "own bytes, asserted with torch.equal and not a tolerance"),
        ("        if not torch.equal(gathered[probe], table[order[probe]]):\n", 1,
         "the four-row spot check that runs before the whole-table comparison"),
        ("    closure_state = merge_closure_intact(vocab, keptset)\n", 1,
         "the STRUCTURAL half of the census gate -- the half with no flag.  Wave "
         "44 walks the CL768 ladder to K=3,072, which is the first wave to build "
         "within a factor of four of the 765-row structural floor, so this is "
         "the gate that decides where the axis actually ends"),
        ("    if lost and not args.below_floor:\n", 1,
         "the COVERAGE half, and the guard the --below-floor flag disarms.  "
         "Wave 44 passes that flag on all three rungs it builds, and a wave that "
         "passes a flag must assert the guard the flag disarms or it cannot tell "
         "disarming from absence"),
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
                f"mx44_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVES 40 AND 41'S ADDITIONS, CARRIED.  Neither runs this wave; both are
    # pinned, for the reason stated above.
    tie = (DST / "mx44_vocabtie.py").read_text(encoding="utf-8")
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
                f"mx44_vocabtie.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    rowset = (DST / "mx44_vocabrowset.py").read_text(encoding="utf-8")
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
                f"mx44_vocabrowset.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 43'S TWO SOLVER MODULES, CARRIED AND DORMANT.  Wave 43 re-sealed the
    # frequency-band axis with the child that opened it and closed XCHANGE by its
    # own pre-registered gate, so wave 44 runs neither.  Every pin wave 43 put on
    # them is carried unchanged: a module whose axis is closed is exactly the
    # module a later cut is most likely to damage silently.
    plane = (DST / "mx44_plane.py").read_text(encoding="utf-8")
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
                f"mx44_plane.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    xchange = (DST / "mx44_xchange.py").read_text(encoding="utf-8")
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
                f"mx44_xchange.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 44'S SECOND ADDITION AND THE WAVE'S WHOLE DELIVERABLE.  The dual-band
    # reader's copy-back measure changes from the longest CONTIGUOUS common run
    # to the UNION of all common runs, and the change is pinned in BOTH
    # directions: the new classifier line must exist at its indented form, and
    # the demoted line must be gone from `classify`'s decision.  Waves 42 and 43
    # both published the union number beside every lane and let the longest-run
    # number decide; wave 43's own floor mover is the case where they disagree,
    # 0.157761 against 0.804071, and the hand reading had to break the tie.
    dual = (DST / "mx44_dualband.py").read_text(encoding="utf-8")
    for needle, want, why in (
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
         "THE PRE-REGISTERED PRIMARY THRESHOLD, declared ONCE.  A threshold "
         "spread through the code is a threshold that can be moved after the "
         "measurement without anybody noticing"),
        ("    if union_cov >= COPY_BACK_UNION_COVERAGE:\n", 1,
         "THE PRIMARY DECISION LINE.  The union measure now DECIDES, and it is "
         "pinned at its indented form so a later cut cannot eat the branch and "
         "leave the constant behind"),
        ("    if payload_coverage >= COPY_BACK_PAYLOAD_COVERAGE:\n", 0,
         "THE DEMOTED LINE, pinned at ZERO.  The longest-run measure is still "
         "computed and still published as `legacy_longest_run_classification`, "
         "but it must not decide anything, and the only way to assert that is "
         "to assert its decision line is gone"),
        ("            legacy = legacy_classification(verbatim, payload, termination)\n", 1,
         "the legacy measure must still be COMPUTED on every lane, or the wave "
         "that demoted it has also deleted the evidence that the demotion "
         "mattered.  Pinned at its INDENTED form -- wave 38's lesson 6 -- "
         "because this call sits inside the per-lane loop and a cut that "
         "re-indented it would silently stop scoring some lanes"),
        ("    if legacy_cov >= COPY_BACK_PAYLOAD_COVERAGE:\n", 1,
         "the legacy classifier keeps its own threshold line, in its own "
         "function, under its own variable name.  The rename from "
         "`payload_coverage` to `legacy_cov` is what makes the pin above -- the "
         "OLD decision line at zero -- an assertion about `classify` and not a "
         "spelling accident"),
        ("UNION_MIN_BLOCK = 8", 1,
         "the union measure's minimum block length, unchanged from the value "
         "waves 42 and 43 published their numbers at.  Promoting a measure and "
         "retuning it in the same wave would make the promotion unfalsifiable"),
    ):
        got = dual.count(needle)
        if got != want:
            problems.append(
                f"mx44_dualband.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-43 identifier survived; ELEVEN seal roots declared; "
          "the builder's accept list kept sixteen basis schemas AND THE SEALER'S "
          "OWN BUILD_RECEIPT_SCHEMAS carries all seven of this wave's strings; "
          "sealq.sh sources THIS wave's env.sh, per-case writer and "
          "seal-verdict script -- the three hand-retargeted paths, pinned in "
          "both directions for the first time -- and carries this wave's "
          "marker, ZERO seal calls and ZERO lines after the calibration "
          "sentinel; gainq.sh, tieq.sh, rowsetq.sh, dualbandq.sh, planeq.sh and "
          "xchangeq.sh exist; both extension surfaces have a scorer and v1 "
          "still names the v1 bank; the penalty diagnostic kept its NOT_A_FOLD "
          "banner and its identity refusal; wave 36's composed width x MLP "
          "path, wave 37's live-mask repair, wave 38's three multi-index "
          "excision lines, wave 40's five tie-builder lines, wave 41's six "
          "rowset lines and wave 43's thirteen solver lines are present but "
          "DORMANT this wave; the builder's log prefix names THIS wave and no "
          "stale tag survives; all six vocabulary-gather lines are present at "
          "their indented form and ARE live code this wave; and THE UNION COPY "
          "MEASURE IS PRIMARY -- its threshold is declared once, its decision "
          "line is pinned at its indented form, the longest-run decision line "
          "is pinned at ZERO, and the legacy measure is still computed")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
