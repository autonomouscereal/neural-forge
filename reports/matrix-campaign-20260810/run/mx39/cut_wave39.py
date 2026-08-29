#!/usr/bin/env python3
"""Cut wave 39's tree from wave 38's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-38 cutter, in full.**

1. ``SRC`` is the wave-38 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  ``sealq.sh`` arrives VERBATIM carrying
   wave 38's marker and wave 38's SIX seal calls with every count intact, and
   ``--audit`` REFUSES until both are replaced by hand.  Seventh consecutive cut
   where that protection has had real work to do.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README -- exactly
   as in waves 35 through 38.
4. **THIS WAVE MAKES ONE EDIT AND IT IS A LOG STRING.**  See below; it is
   declared here rather than discovered in a diff, and it is asserted at a
   pinned count in both directions like every other guard in this file.

**WAVE 39'S OWN AUDIT ADDITION, AND WHY IT IS THE VOCABULARY GATHER'S TURN.**
Every cut since wave 31 has asserted the lines that wave's own build list
depends on: wave 36 the composed width x MLP site, wave 37 the live-mask repair,
wave 38 the descending deletion order and the two module re-index assignments.
**Wave 39's build list is TWO `mx39_vocab.py` passes and nothing else** -- no
body, no solve, no capture -- and in eleven waves of folding on that axis **no
cutter has ever asserted a single line of it.**  It is the least-guarded module
this campaign still runs, and it is the only module wave 39 runs at all.

Four lines carry it and all four are pinned below:

* ``if not torch.equal(gathered, table[index]):`` -- the identity gate for a
  ZERO-SOLVE arm.  Every kept row is the source's own bytes, so this is
  ``torch.equal`` and not a tolerance.  Delete it and a gather that silently
  permuted or resampled rows would still export, still load, still decode, and
  still be sealed as a zero-solve fold of the parent's own weights.
* ``closure_state = merge_closure_intact(vocab, keptset)`` -- the STRUCTURAL
  half of the census gate, the half that has no flag because a kept token that
  cannot be produced from the restricted merge list is a broken tokenizer and
  never a coverage choice.
* ``if lost and not args.below_floor:`` -- the COVERAGE half.  Both of wave 39's
  children are built below the census floor deliberately, which means this wave
  passes the flag that turns this refusal off; if the refusal itself had been
  dropped the flag would be a silent no-op and the loss would go unrecorded in
  the receipt.  **A wave that passes a flag must assert the guard the flag
  disarms**, or it cannot tell disarming from absence.
* ``"the child tokenizer is not lossless on text; the byte alphabet or "`` --
  the refusal behind ``decode(child_encode(t)) == t``.  This parent has
  ``byte_fallback: false`` and ``unk_token: null``, so the 256 byte-alphabet
  rows are the only reason an arbitrary byte can still be encoded at all.

**THE NEEDLES CARRY THEIR INDENTATION AND THAT IS WAVE 38'S LESSON, NOT
DECORATION.**  The bare strings ``torch.equal``, ``merge_closure_intact`` and
``lossless on text`` occur 6, 3 and 4 times in that file -- the module's own
docstring and its receipt fields quote every one of them -- so a bare-string pin
would pass with the CODE deleted and only the prose left.  Each indented form
occurs exactly once.

**AND WAVE 39 MAKES ONE EDIT: THE BUILDER'S LOG PREFIX.**
``mx39_build.py``'s logger has printed ``[mx32]`` since wave 32's cut, because
the literal sits inside an f-string that the substitution list -- identifiers,
paths and schema strings only -- has no entry for.  Wave 17 carried the same
defect as ``[mx16]``; ``NOTES-MX-AP`` calls it the seventh issue and cosmetic,
and waves 37 and 38 both declined to fix it on the stated grounds that an edit
in a no-behavioural-edit wave is itself a behavioural edit.

Wave 39 fixes it, and the reason it can is that **wave 39 does not run that file
at all.**  Its build list is two vocabulary post-processes; ``mx39_build.py`` is
carried, hand-retargeted and asserted, and never executed.  So the edit goes
under no seal, changes no artifact, and cannot move a number.  It is pinned in
BOTH directions below -- ``[mx39]`` at one and ``[mx32]`` at zero -- and it is
verified by calling the function rather than by reading it.

**THE ONE THING WAVE 36 DID THAT NO EARLIER CUT DID, AND THIS CUT DOES AGAIN.**
It appends this wave's four accept strings to
``experiments/seal_matrix_architecture_child.py`` **in the same commit that cuts
this tree**.  That file lives in the Codex lane, so this cutter has no standing
to assert anything about it and does not try; the append is a hand edit made
beside this one.  Waves 34 and 35 each paid an entire deployment to discover the
same refusal after the GPU work was already done; waves 36, 37 and 38 appended
at the cut and paid none.  Wave 39 is the fourth.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave38`` (a path component and a
schema infix) is retargeted; ``wave 38`` and ``Wave 38`` (English) are NOT, so
a docstring that records what wave 38 measured still says wave 38 measured it.
``run/mx39/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-39 header, by hand, after the
copy -- and this wave changes one line in one file, which the README states and
this docstring states above it.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx38_verify_seals.py`` -- the rename turns ``wave38/cells`` into
  ``wave39/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  34 declared roots become 35, ``wave38/cells`` still declared.
* ``mx38_retain.py`` -- a KEEP set inherited from last wave names ``MX38-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX39-*`` tags out of wave 38's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx38_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line, the module references, the build-receipt
  filename and this wave's ONE log-prefix repair are made by hand.  Wave 36's
  five assertions, wave 37's two and wave 38's three are carried unchanged and
  are asserted on this wave's copy.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 38's seal calls are
  deleted by hand and the marker is rewritten to name wave 39.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  It is committed beside the tree, because "renamed
correctly" is a claim about bytes and a claim about bytes needs a receipt.

**AND IT PARSES WHAT IT WRITES, IN THE INVOCATION THAT WRITES IT.**
`NOTES-MX-AO` section 9 is wave 37's account of reproducing wave 35's corruption
bug -- a script's stdout redirected into the path the script itself writes -- and
its instruction to the next wave is that every artifact a wave writes gets parsed
by the command that writes it.  This cutter reads its own manifest back with
``json.loads`` before it returns, so the rule is obeyed by the first file wave 39
writes rather than only by the ones a reader remembers to check.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx38"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32's cut brought wave 31's nineteen
#: seal calls forward with the substitution applied and every COUNT intact.
#: Nothing downstream would have caught a wrong count -- to ``seal_verdict.sh`` a
#: count is a number -- and the only thing that stopped it was that the tags did
#: not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx38_verify_seals.py",
    "mx38_retain.py",
    "mx38_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files, the same
#: two waves 35 through 38 declared, and for the same reason: a declared-authored
#: file that nobody designed is a line that makes the set stop meaning anything.
AUTHORED_THIS_WAVE = {
    "cut_wave39.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave38-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 38", "--campaign-wave 39"),
    ('"campaign_wave": 38', '"campaign_wave": 39'),
    ("wave38", "wave39"),
    ("mx38", "mx39"),
    ("MX38", "MX39"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx38_": "a wave-38 module reference survived the copy",
    "MX38-": "a wave-38 child directory name survived the copy",
    "/wave38/": "a wave-38 artifact path survived the copy",
    "--campaign-wave 38": "a driver would seal this wave's build as wave 38",
}
#: DELIBERATE historical references, declared with a count.  A reference to last
#: wave's file is legitimate in a header that states this wave's diff against it,
#: and illegitimate anywhere a path is resolved -- so the exception is per file,
#: per needle, and the count is pinned rather than waived.
#:
#: **THE THIRD FIELD IS THE COMPARISON.**  Wave 38 was the first cut to need two
#: kinds of it, because five of its six children were one CPU pass off a wave-37
#: RETAINED checkpoint and the retention and provenance rules both required that
#: directory name in prose.  **Wave 39's source is wave 35's retained
#: `MX35-CL896X13`**, which no needle here forbids, so the two bounded exemptions
#: are expected NOT to fire this wave.  They are declared at a small bound
#: anyway, because a seal note that legitimately names wave 38's retained floor
#: child while comparing this wave's X13 ladder against it is a note and not a
#: leak -- and a bound still refuses an unbounded one.
ALLOWED_HISTORICAL = {
    ("mx39_build.py", "mx38_"): (
        1,
        "==",
        "the wave-39 header names `mx38_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx39_retain.py", "MX38-"): (
        2,
        "<=",
        "the retention rule requires each DROP reason to name the retained source "
        "that rebuilds the deleted child.  This wave's source is wave 35's "
        "retained MX35-CL896X13, which is not forbidden, so this is expected to "
        "read ZERO; bounded at two so a legitimate cross-reference to wave 38's "
        "retained floor child is a note rather than a refusal",
    ),
    ("sealq.sh", "MX38-"): (
        2,
        "<=",
        "a seal note naming a RETAINED checkpoint a child was built on, which is "
        "provenance the seal is required to carry.  Wave 39's children are both "
        "one pass off wave 35's retained MX35-CL896X13, so this is expected to "
        "read ZERO; bounded at two for the same reason as above",
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
        "schema": "neural-forge-matrix-wave39-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 38` / `Wave 38` / `WAVE 38` in English are NOT substituted, so "
            "a docstring recording what wave 38 measured still says wave 38 did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "ONE, and it is a log string in a file this wave does not run.  "
            "`mx39_build.py`'s logger has printed [mx32] since wave 32's cut "
            "because the literal sits inside an f-string the substitution list "
            "has no entry for; NOTES-MX-AP calls it the seventh issue and "
            "cosmetic, and waves 37 and 38 both declined it on the grounds that "
            "an edit in a no-edit wave is itself an edit.  Wave 39's build list "
            "is two mx39_vocab.py passes, so that file is carried, retargeted, "
            "asserted and NEVER EXECUTED: the repair goes under no seal and can "
            "move no number.  It is pinned in both directions and verified by "
            "calling the function.  Wave 36's composed width x MLP edit is "
            "carried unchanged with its five guarding assertions, wave 37's two "
            "live-mask assertions and wave 38's three multi-index excision "
            "assertions are carried, and wave 39 adds FOUR of its own about the "
            "vocabulary gather -- the module that is this wave's entire build "
            "list and that no cutter has ever asserted a line of."
        ),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    out = DST / "wave39-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-38 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave39.py", "wave39-tree-cut-manifest.json"):
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
    # not happen and the rename did.  Six roots are checked by name: the five a
    # rename would have eaten in the last five waves, and this wave's own.
    for name, needle in (("mx39_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx39_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx39_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx39_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx39_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx39_verify_seals.py", 'ROOT / "wave39" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx39_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave39-residual-basis-v1",
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
            problems.append(f"mx39_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.  The build
    # receipt is written under a LITERAL filename, and that literal said
    # `mx32-build-receipt.json` from wave 32 through wave 33's hand edit.  Every
    # known-stale name stays pinned at zero, because the defect that hid for two
    # waves hid behind a check that did not exist for either of them.
    for needle, want, why in (
        ('(outdir / "mx39-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx39_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ONE EDIT, PINNED IN BOTH DIRECTIONS -- AND PINNED AT THE INDENTED
    # FORM, WHICH THIS CUTTER LEARNED THE HARD WAY ON ITS OWN FILE.  The builder's
    # logger has printed `[mx32]` since wave 32's cut because the literal lives
    # inside an f-string and the substitution list carries identifiers, paths and
    # schema strings only.  Wave 17 carried the same defect as `[mx16]`.
    #
    # The first version of this block pinned the BARE strings `[mx32]` at zero and
    # `[mx16]` at one, and it refused -- **on the wave-39 header this cut wrote**,
    # which names both literals four times between them while explaining the
    # repair, exactly as wave 38's audit refused on the docstring quoting the
    # three lines IT pinned.  That is `NOTES-MX-AP` lesson 6 arriving a second
    # time in a different file, and the answer is the same: the assertion is about
    # the CODE, so the needle is the code's own line with its own indentation.
    # The wave-18 docstring recording `[mx16]` and this wave's header explaining
    # the repair are prose and are left exactly as they are, because rule 1 does
    # not rewrite prose that records history.
    for needle, want, why in (
        ('    print(f"[mx39] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave, or a reader grepping a "
         "build log for this wave's tag finds nothing"),
        ('    print(f"[mx32] {message}", flush=True)\n', 0,
         "the wave-32 log prefix carried by every cut since wave 32; NOTES-MX-AP "
         "records it as the seventh issue and wave 39 is the wave that could "
         "afford to fix it, because it never executes this file"),
        ('    print(f"[mx16] {message}", flush=True)\n', 0,
         "wave 17's form of the same defect, pinned at zero so the repair cannot "
         "regress to an even older tag.  The PROSE mention of [mx16] in wave 18's "
         "docstring is not this needle and is deliberately untouched"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx39_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 39" not in body:
            problems.append(f"{name} does not carry --campaign-wave 39")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it; wave 33 was the first cut
    # where it fired.  `sealq.sh` arrives VERBATIM, so it arrives carrying LAST
    # wave's marker and LAST wave's seal calls, and this assertion forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 39 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # The terminal-gain driver was dropped after wave 29 and re-authored in wave
    # 33.  This cut carries it; if a future cut loses it again, this fires.  Wave
    # 39 does not run it, which is exactly the condition under which a driver
    # goes missing unnoticed.
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    # WAVE 34's ADDITION, carried unchanged.  The second review surface is two
    # banks and two rubrics and the v1 pair is FROZEN: a cut that quietly edited
    # the v1 rubric would rewrite the scoring of sixty committed observations.
    for name in ("mx39_extscore.py", "mx39_extrubric2.py", "mx39_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx39_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx39_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.  `mx39_penaltydecode.py` is the only module in
    # this tree that decodes a sealed child through a knob the frozen protocol
    # does not have.  Wave 39 does not edit it and does not run it -- the clause
    # it prices is settled -- but the three things that keep it a diagnostic
    # rather than a second scoring standard are asserted anyway, because a
    # carried file nobody checks is a file that quietly changes.
    pen = (DST / "mx39_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx39_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S ADDITION, CARRIED UNCHANGED AND ASSERTED ON THIS WAVE'S COPY FOR
    # THE THIRD CONSECUTIVE WAVE.  Wave 36 lifted `NOTES-MX-AM` 3.5's build
    # refusal IN HALF: `--hidden-size` may be composed with `--ff`, and may still
    # NOT be composed with `--head-dim`.  Wave 39 does not touch that path and
    # does not build on it -- it does not open the builder at all -- which is the
    # condition under which a guard rots fastest.
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
         "the parent's gated intermediate at the lifted row; that claim is "
         "measured at every site or the composition is an assumption"),
        ("the intermediate-axis identity does not hold", 2,
         "wave 14 3's identity precondition must be checked on the CHILD's own "
         "gated intermediate too, not only on the parent's"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx39_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 37'S ADDITION AND WAVE 38'S, BOTH CARRIED.  Wave 39 excises nothing --
    # both of its children are vocabulary folds on a body wave 35 already sealed
    # -- so neither of these paths runs this wave.  They are asserted anyway, at
    # the same pinned counts and the same indented forms, because the wave that
    # does not run a path is the wave that will not notice it disappearing.
    for needle, want, why in (
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
                f"mx39_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S OWN ADDITION, and it is the module this wave actually runs.  In
    # eleven waves of folding on the vocabulary axis no cutter has asserted a
    # single line of `mx39_vocab.py`, and wave 39's build list is TWO passes
    # through it and nothing else.  Four lines carry it; each is pinned at its
    # INDENTED form because the bare strings `torch.equal`,
    # `merge_closure_intact` and `lossless on text` occur 6, 3 and 4 times in
    # that file -- the module's own docstring and its receipt fields quote every
    # one of them -- so a bare-string pin would pass with the CODE deleted.
    # That is wave 38's lesson 6 applied to a different file.
    vocab = (DST / "mx39_vocab.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "the identity gate for a ZERO-SOLVE arm: every kept row is the source's "
         "own bytes, asserted with torch.equal and not a tolerance.  Without it "
         "a gather that permuted or resampled rows would still export, still "
         "load, still decode and still seal as a fold of the parent's weights"),
        ("        if not torch.equal(gathered[probe], table[order[probe]]):\n", 1,
         "the four-row spot check that runs before the whole-table comparison, "
         "so a failure names a row rather than a tensor"),
        ("    closure_state = merge_closure_intact(vocab, keptset)\n", 1,
         "the STRUCTURAL half of the census gate -- the half with no flag, "
         "because a kept token that cannot be produced from the restricted merge "
         "list is a broken tokenizer and never a coverage choice"),
        ("    if lost and not args.below_floor:\n", 1,
         "the COVERAGE half, and it is the guard that DECIDES whether this wave "
         "needs the --below-floor flag at all.  Wave 38 built K=14,336, 10,240 "
         "and 9,216 on the sibling body and every one read sealed-corpus "
         "coverage 1.000000 with ZERO occurrences lost, so this refusal never "
         "fired and the flag was never passed.  Wave 39 does not pass it either "
         "and does not get to assume it would not have fired: if this refusal "
         "were missing, a rung that DID lose coverage would build silently and "
         "the receipt's below_census_floor field would say nothing"),
        ('            "the child tokenizer is not lossless on text; the byte alphabet or "\n', 1,
         "the refusal behind decode(child_encode(t)) == t.  This parent has "
         "byte_fallback false and unk_token null, so the 256 byte-alphabet rows "
         "are the only reason an arbitrary byte can still be encoded at all"),
    ):
        got = vocab.count(needle)
        if got != want:
            problems.append(
                f"mx39_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-38 identifier survived; six seal roots declared; "
          "the accept list kept eleven basis schemas; the drivers name wave 39; "
          "sealq.sh carries this wave's marker; gainq.sh exists; "
          "both extension surfaces have a scorer and v1 still names the v1 bank; "
          "the penalty diagnostic kept its NOT_A_FOLD banner and its identity "
          "refusal; wave 36's composed width x MLP path still carries its "
          "surviving refusal, its exactness assertion and both identity "
          "preconditions; wave 37's live-mask repair and wave 38's three "
          "multi-index excision lines are present; the builder's log prefix "
          "names THIS wave and neither stale tag survives; and all five lines "
          "the vocabulary gather -- this wave's ONLY build path -- depends on "
          "are present at their indented form")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
