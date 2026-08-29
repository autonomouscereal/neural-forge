#!/usr/bin/env python3
"""Cut wave 42's tree from wave 41's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-41 cutter, in full.**

1. ``SRC`` is the wave-41 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  Wave 40's seal-call pin -- ``audit()``
   counts ``^seal <TAG>`` and requires **zero at cut time** -- is carried, and
   wave 42 is the SECOND cut where it had work to do from the start: this file's
   copy of ``sealq.sh`` arrived carrying wave 41's **ten** seal calls with every
   count intact.
3. ``AUTHORED_THIS_WAVE`` is FOUR files -- this cutter, the README, and the two
   files that are wave 42's one behavioural addition, ``mx42_dualband.py`` and
   ``dualbandq.sh``.
4. **WAVE 42 ADDS ONE MEASUREMENT MODULE AND EDITS NO CARRIED ONE.**  Wave 41
   added a build module and edited one; wave 42 adds a module that builds
   nothing and reads what is already sealed, and touches no carried module at
   all.  Every carried file keeps the bytes wave 41 shipped, modulo the five
   substitutions and the four hand edits.

**WHY A NEW MODULE, AND WHAT IT IS.**  ``NOTES-MX-AS`` §4.1 recorded a
contradiction wave 41 refused to resolve: the canonical clause ("a copy-back is
not a decode failure, it is the absence of a decode") reads all ten wave-41
children 0/15, and the SAME clause applied to wave 40's sealed floor child
``ARCH.V9CL896X13`` -- whose held lane is byte-identical to wave 41's, digest
``111e342d9cb5438c`` -- gives canonical zero where wave 40 published canonical
one.  Either the clause does not mean what it says, or the canonical 1/15 rung
has been zero since wave 39.

That is an adjudication and not a measurement, and no wave can settle it by
building a child.  What a wave CAN do is make both readings complete and
checkable, so that whichever way the ruling goes, nothing needs re-grading.
``mx42_dualband.py`` does exactly that and nothing else:

* it enumerates every ``ARCH.*`` child in the ledger at or below a stated ratio;
* it reads that child's SEALED per-case verdicts and its FROZEN dossier;
* it classifies every one of the sixteen lanes against the lane's own prompt
  payload by a deterministic, published, thresholded string measure, and prints
  the response digest beside it so every call is checkable from the share;
* it publishes TWO columns -- ``copy_back_counts`` (wave 40's practice, which is
  the sealed count verbatim and is never recomputed) and ``copy_back_refused``
  (the canonical clause as written) -- and it derives the second from the first
  by VOIDING held cases whose held lane is a copy-back, so the derivation is one
  subtraction a reader can check by hand.

**THE MODULE RE-GRADES NOTHING.**  Column A is copied out of the sealed
``per-case.json`` byte for byte; the module refuses to write an artifact if its
own count of column A disagrees with the ledger's sealed
``behavior_useful_or_partial`` for that child.  That refusal is the whole reason
the artifact is worth anything: a second column is only interesting if the first
one is provably the sealed one.

**AND THE LANE-ASK TABLE IS THE LOAD-BEARING PART, NOT THE THRESHOLD.**  Wave
40 sealed a copied SUMMARISATION lane as held and a copied RETRIEVAL lane as a
miss, and wave 41 showed those two precedents are consistent: the summarisation
lane's ask is to PRESERVE two facts, which a copy does, and the retrieval lane's
ask is to SELECT three facts, which a copy does not.  So the copy-back clause
can only change a verdict on a PRESERVE-class lane.  ``LANE_ASK`` names the
class of all sixteen lanes explicitly, and ``audit()`` pins it, because a table
that quietly reclassified a lane would move the floor by relabelling rather than
by measuring.

**WAVE 42'S OWN AUDIT ADDITIONS.**  Every cut since wave 31 asserts the lines
this wave's build list depends on.  Wave 42's build list is three
``mx42_vocab.py`` gathers and no body, no solve and no capture, so the
vocabulary module's six lines are asserted exactly as wave 41 asserted them --
including ``--below-floor``'s guard, which every rung this wave builds passes.
Six new needles cover the dual-band reader.  Three are worth naming here:

* ``if column_a != int(sealed_count):`` -- the refusal that makes column A the
  SEALED column and not this module's opinion of it.
* ``if lane_ask not in ("PRESERVE", "SELECT", "PRODUCE"):`` -- every lane must
  carry a declared ask class; an unclassified lane cannot be silently dropped
  out of the second column.
* ``if classification == "COPY_BACK" and lane_ask != "PRESERVE":`` -- the
  exception check.  A copy can satisfy a PRESERVE ask and neither of the other
  two, so the clause should only ever reach a PRESERVE lane; if it reaches
  another one, that is published as an anomaly rather than absorbed.

**THE NEEDLES CARRY THEIR INDENTATION** -- wave 38's lesson 6, carried.

**WHAT THIS CUT DOES NOT DO.**  It does not add a PLANE or an XCHANGE solver.
``NOTES-MX-AS`` §13 items 5 and 6 named those as the honest cost of wave 41 and
the reason was authoring time for a new solver; that reason has not changed and
naming it again is cheaper than a tree that carries a driver for a module that
does not exist.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  **And it parses what it writes, in the invocation that
writes it** -- `NOTES-MX-AO` §9's rule, obeyed by the first file wave 42 writes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx41"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx41_verify_seals.py",
    "mx41_retain.py",
    "mx41_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  FOUR files, the
#: same count waves 40 and 41 declared and for the same reason -- this cutter,
#: the README, and the two files that are this wave's one addition,
#: ``mx42_dualband.py`` and ``dualbandq.sh``.
AUTHORED_THIS_WAVE = {
    "cut_wave42.py",
    "README.md",
    "mx42_dualband.py",
    "dualbandq.sh",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt.
NOT_CARRIED = {"wave41-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 41", "--campaign-wave 42"),
    ('"campaign_wave": 41', '"campaign_wave": 42'),
    ("wave41", "wave42"),
    ("mx41", "mx42"),
    ("MX41", "MX42"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.
FORBIDDEN_AFTER = {
    "mx41_": "a wave-41 module reference survived the copy",
    "MX41-": "a wave-41 child directory name survived the copy",
    "/wave41/": "a wave-41 artifact path survived the copy",
    "--campaign-wave 41": "a driver would seal this wave's build as wave 41",
}
#: DELIBERATE historical references, declared with a count.  The third field is
#: the comparison: an exact pin where the count is knowable before the wave
#: runs, a bound where it is not.
ALLOWED_HISTORICAL = {
    ("mx42_build.py", "mx41_"): (
        1,
        "==",
        "the wave-42 header names `mx41_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx42_vocab.py", "mx41_"): (
        1,
        "<=",
        "the wave-42 header may name `mx41_vocab.py` as the file it states its "
        "diff against.  Bounded rather than pinned because this wave makes no "
        "edit to that module and may add no header at all",
    ),
    ("mx42_vocabrowset.py", "mx41_"): (
        4,
        "<=",
        "wave 41 authored the rowset builder and it CITES by name the module "
        "whose ranking key it re-orders.  Those citations were written as "
        "`mx40_vocab.py` and the wave-42 substitution does not touch them; this "
        "entry is the bound for any citation that names wave 41's own copy "
        "instead.  It resolves no path -- the shared machinery is imported from "
        "`mx42_vocab` by module name",
    ),
    ("mx42_retain.py", "MX41-"): (
        3,
        "<=",
        "the retention rule requires each DROP reason to name the retained "
        "source that rebuilds the deleted child.  Wave 41 retained two children "
        "-- MX41-V8CL896X13 and MX41-V7CL768 -- and wave 42 grades its own "
        "children AGAINST both of them, so a legitimate cross-reference is a "
        "note and not a refusal",
    ),
    ("sealq.sh", "MX41-"): (
        4,
        "<=",
        "a seal note naming a checkpoint a child is read against.  Wave 42's "
        "sources are wave 35's MX35-CL896X13 and wave 40's retained MX40-CL768, "
        "and its controls are wave 41's retained MX41-V8CL896X13 and "
        "MX41-V7CL768, which a seal note legitimately carries",
    ),
    ("mx42_dualband.py", "MX41-"): (
        2,
        "<=",
        "the dual-band reader's docstring names the two wave-41 children whose "
        "held lanes are the contradiction it exists to make checkable.  It "
        "resolves no path: every dossier and per-case file it opens is located "
        "from the ledger row's own wave number",
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
        "schema": "neural-forge-matrix-wave42-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 41` / `Wave 41` / `WAVE 41` in English are NOT substituted, "
            "so a docstring recording what wave 41 measured still says wave 41 "
            "did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "ONE ADDITION AND NO EDIT.  `mx42_dualband.py` and its driver "
            "`dualbandq.sh` are NEW files and they BUILD NOTHING: they read the "
            "sealed per-case verdicts and the frozen dossiers of every ARCH "
            "child at or below a stated ratio and publish TWO readings of each "
            "-- `copy_back_counts`, which is the sealed column copied verbatim "
            "and refused if it disagrees with the ledger, and "
            "`copy_back_refused`, which voids a held case whose held lane "
            "returns the prompt's own payload.  `NOTES-MX-AS` section 4.1 "
            "flagged that contradiction and left the adjudication to the "
            "caller; this module makes both readings complete and checkable so "
            "the ruling costs no re-grading either way.  Every other carried "
            "module keeps the bytes wave 41 shipped.  Wave 39's five "
            "vocabulary-gather lines are live code this wave -- the build list "
            "is THREE `mx42_vocab.py` gathers -- and wave 36's composed width x "
            "MLP path, wave 37's live-mask repair, wave 38's three multi-index "
            "excision lines, wave 40's seven tie-builder lines and wave 41's six "
            "rowset lines are carried and asserted but NOT run."
        ),
        "files": entries,
    }
    out = DST / "wave42-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-41 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave42.py", "wave42-tree-cut-manifest.json"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for needle, why in FORBIDDEN_AFTER.items():
            hits = text.count(needle)
            if not hits:
                continue
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
            problems.append(f"{path.name}: {why} ({needle!r} x{hits})")
    # The seal verifier must STILL carry every earlier wave, or the hand edit did
    # not happen and the rename did.  Nine roots are checked by name.
    for name, needle in (("mx42_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave39" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave40" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave41" / "cells"'),
                         ("mx42_verify_seals.py", 'ROOT / "wave42" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx42_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave42-residual-basis-v1",
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
            problems.append(f"mx42_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.
    for needle, want, why in (
        ('(outdir / "mx42-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx42_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # WAVE 39'S ONE EDIT, CARRIED AND STILL PINNED IN BOTH DIRECTIONS.  Wave 39
    # repaired the builder's log prefix from `[mx32]`; wave 40 was the first wave
    # to RUN the repaired file.  Wave 42 does not run it at all, and the pin is
    # kept for the reason wave 41 kept the dormant tie pins: the module is in the
    # tree and a later wave will run it.
    for needle, want, why in (
        ('    print(f"[mx42] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave even in a wave that does "
         "not run it, or the next wave that does inherits a stale tag"),
        ('    print(f"[mx41] {message}", flush=True)\n', 0,
         "last wave's prefix, which the substitution list retargets; pinned at "
         "zero so a hand edit cannot leave it behind"),
        ('    print(f"[mx40] {message}", flush=True)\n', 0,
         "the wave-40 prefix, pinned at zero"),
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
                f"mx42_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 42" not in body:
            problems.append(f"{name} does not carry --campaign-wave 42")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 42 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # WAVE 40'S PIN, CARRIED.  The marker assertion above is ONE line at the top;
    # last wave's seal CALLS are anywhere below it, so a hand edit that rewrites
    # the marker and misses the calls passes.  `--audit` runs immediately after
    # the cut and before anything builds, when the correct number of seal calls
    # is ZERO; the wave appends its own after it reads its dossiers and does not
    # re-run this.  `seal () {` is the function definition and is not a call, so
    # the needle requires an identifier char.  WAVE 41 CARRIED TEN CALLS HERE.
    calls = re.findall(r"(?m)^seal [A-Za-z0-9]", seals)
    if calls:
        problems.append(
            f"sealq.sh carries {len(calls)} seal CALL(s) at cut time, expected 0. "
            "A verbatim copy brings last wave's calls forward with every count "
            "intact, and rewriting the marker above them does not remove them"
        )
    # WAVE 42'S ADDITION, AND IT IS THE SAME DEFECT ONE LEVEL UP.  Wave 40's pin
    # counts CALLS.  It does not count CLAIMS, and this file is where the reading
    # standard a seal was graded under is written down.  The block below the
    # marker arrived carrying WAVE 39'S CALIBRATION RESULT word for word --
    # "re-read blind from ARCH.V14CL896X13's dossier -- the first dossier of the
    # wave", a wave-39 child -- through wave 40's cut AND wave 41's, while both
    # of those waves ran their own calibration and recorded a different first
    # dossier in their own NOTES.  Nothing was mis-sealed; each wave's counts
    # were its own.  What was wrong is that the file a reader opens to find out
    # HOW a seal was graded described a measurement two waves old.  A result
    # sentence cannot be true before the wave has taken the measurement, so its
    # correct count at cut time is ZERO, exactly like a seal call.
    for needle, why in (
        ("# AND IT WAS RUN AND IT IS EXACT.",
         "an inherited CALIBRATION RESULT survived the cut.  This wave has not "
         "run its calibration at cut time, so a sentence asserting that it did "
         "is last wave's -- or, as it turned out in waves 40 and 41, wave 39's"),
        ("ARCH.V14CL896X13's dossier",
         "wave 39's named calibration dossier survived the cut, for the third "
         "consecutive wave.  The wave writes the dossier it actually read"),
        ("TWELFTH consecutive calibration",
         "an inherited calibration COUNTER survived the cut; wave 41's own "
         "NOTES record the fourteenth while this file still said the twelfth"),
    ):
        hits = seals.count(needle)
        if hits:
            problems.append(
                f"sealq.sh carries {needle!r} x{hits} at cut time, expected 0 -- {why}"
            )
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    if not (DST / "tieq.sh").is_file():
        problems.append("tieq.sh is missing: wave 40's build module has no driver")
    if not (DST / "rowsetq.sh").is_file():
        problems.append("rowsetq.sh is missing: wave 41's build module has no driver")
    if not (DST / "dualbandq.sh").is_file():
        problems.append("dualbandq.sh is missing: this wave's new module has no driver")
    # WAVE 34's ADDITION, carried unchanged.
    for name in ("mx42_extscore.py", "mx42_extrubric2.py", "mx42_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx42_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx42_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.
    pen = (DST / "mx42_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx42_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S, 37'S AND 38'S ADDITIONS.  Wave 42 builds NO body, so these pins
    # guard dormant code.  They are kept for wave 41's stated reason: the module
    # is in the tree and a future wave will run it, and dropping a pin because a
    # wave does not exercise it is how wave 30 deleted a seal root.
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
                f"mx42_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ADDITION, CARRIED -- and wave 42 runs this module THREE times.
    # It is the ONLY builder this wave runs.
    vocab = (DST / "mx42_vocab.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "the identity gate for a ZERO-SOLVE arm: every kept row is the source's "
         "own bytes, asserted with torch.equal and not a tolerance"),
        ("        if not torch.equal(gathered[probe], table[order[probe]]):\n", 1,
         "the four-row spot check that runs before the whole-table comparison"),
        ("    closure_state = merge_closure_intact(vocab, keptset)\n", 1,
         "the STRUCTURAL half of the census gate -- the half with no flag"),
        ("    if lost and not args.below_floor:\n", 1,
         "the COVERAGE half, and the guard the --below-floor flag disarms.  "
         "Wave 42 passes that flag on all three rungs it builds, and a wave that "
         "passes a flag must assert the guard the flag disarms or it cannot tell "
         "disarming from absence"),
        ('            "the child tokenizer is not lossless on text; the byte alphabet or "\n', 1,
         "the refusal behind decode(child_encode(t)) == t"),
        ('    ranking = sorted(\n', 1,
         "`choose_kept`'s discretionary ranking.  Wave 41 closed the question of "
         "WHICH rows with three built arms; this wave moves HOW MANY again, on "
         "the same call, and the call has to be the same one"),
    ):
        got = vocab.count(needle)
        if got != want:
            problems.append(
                f"mx42_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 40'S AND WAVE 41'S ADDITIONS, CARRIED.  Neither builder runs this
    # wave; both are pinned, for the reason stated above.
    tie = (DST / "mx42_vocabtie.py").read_text(encoding="utf-8")
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
                f"mx42_vocabtie.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    rowset = (DST / "mx42_vocabrowset.py").read_text(encoding="utf-8")
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
                f"mx42_vocabrowset.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 41'S EDIT TO A CARRIED MODULE, CARRIED AND STILL PINNED IN BOTH
    # DIRECTIONS.  `NOTES-MX-AR` §11.6.  Wave 42 runs this generator.
    index_py = (DST / "mx42_child_index.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ('    ap.add_argument("--out")\n', 1,
         "the generator must take an explicit output path"),
        ("    reread = json.loads(open(path, encoding=\"utf-8\").read())\n", 1,
         "parse-what-you-write lives INSIDE the generator"),
        ('    print("->", path)\n', 1,
         "the stdout summary every earlier wave read is KEPT"),
    ):
        got = index_py.count(needle)
        if got != want:
            problems.append(
                f"mx42_child_index.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 42'S OWN ADDITION.  `mx42_dualband.py` is authored this wave, so it
    # has no history of surviving a cut and every load-bearing line in it is
    # pinned at its indented form from the start rather than after the first cut
    # eats one.  Wave 38's lesson 6.
    dual = (DST / "mx42_dualband.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("        if column_a != int(sealed_count):\n", 1,
         "THE REFUSAL THAT MAKES COLUMN A THE SEALED COLUMN.  This module's "
         "count of the sealed per-case verdicts must equal the ledger's own "
         "behavior_useful_or_partial for that child, or the artifact is not "
         "written.  A second column is only worth reading if the first one is "
         "provably the sealed one and not this module's opinion of it"),
        ("        if lane_ask not in (\"PRESERVE\", \"SELECT\", \"PRODUCE\"):\n", 1,
         "every lane must carry a DECLARED ask class.  An unclassified lane "
         "would fall out of the second column silently, which is a floor move "
         "by omission"),
        ("                    if classification == \"COPY_BACK\" and lane_ask != \"PRESERVE\":\n", 1,
         "THE EXCEPTION CHECK, and it is the reason the ask table is the "
         "load-bearing part.  A copy can satisfy a PRESERVE ask and can satisfy "
         "neither a SELECT ask nor a PRODUCE ask, which is why wave 40's two "
         "sealed precedents -- a copied summarisation HELD, a copied retrieval "
         "a MISS -- are consistent.  If the second column ever voids a held "
         "lane whose ask is not PRESERVE, that is an EXCEPTION to those "
         "precedents and is published as an anomaly rather than absorbed"),
        ("    if len(LANE_ASK) != 16:\n", 1,
         "all sixteen lanes of the frozen surface must be classified.  Fifteen "
         "cases and sixteen lanes -- the safety lane is charged and not scored "
         "-- and a table that lost one would silently change a column"),
        ("            if not payload:\n", 1,
         "a lane whose payload anchor does not resolve cannot be measured for "
         "copy-back and must say so rather than score zero, because zero is "
         "indistinguishable from 'measured and clean'"),
        ("            if sealed_verdict in (\"Useful\", \"Partial\"):\n", 1,
         "only a HELD lane can be voided by the second column.  A lane already "
         "sealed as a miss is not re-read, and this module never turns a miss "
         "into a hold in either direction"),
    ):
        got = dual.count(needle)
        if got != want:
            problems.append(
                f"mx42_dualband.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-41 identifier survived; NINE seal roots declared; "
          "the accept list kept fourteen basis schemas; the drivers name wave 42; "
          "sealq.sh carries this wave's marker and ZERO seal calls; gainq.sh, "
          "tieq.sh, rowsetq.sh and dualbandq.sh exist; both extension surfaces "
          "have a scorer and v1 still names the v1 bank; the penalty diagnostic "
          "kept its NOT_A_FOLD banner and its identity refusal; wave 36's "
          "composed width x MLP path, wave 37's live-mask repair and wave 38's "
          "three multi-index excision lines are present but DORMANT this wave; "
          "the builder's log prefix names THIS wave and no stale tag survives; "
          "all six vocabulary-gather lines are present at their indented form "
          "and ARE live code this wave; wave 40's five tie-builder lines and "
          "wave 41's six rowset lines are carried; the child-index edit is "
          "pinned in both directions; and all six lines the NEW dual-band "
          "reader depends on are pinned at their indented form")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
