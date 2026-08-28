#!/usr/bin/env python3
"""Cut wave 36's tree from wave 35's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-35 cutter, in full.**

1. ``SRC`` is the wave-35 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  ``sealq.sh`` arrives VERBATIM carrying
   wave 35's marker and wave 35's EIGHT seal calls with every count intact, and
   ``--audit`` REFUSES until both are replaced by hand.  Fourth consecutive cut
   where that protection has had real work to do.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README -- exactly
   as in wave 35.  Wave 36's one behavioural edit is made ON the carried copy of
   the builder, by hand, after the copy, under a hand-written wave-36 header:
   **the composed width x MLP path**, which lifts half of the refusal
   `NOTES-MX-AM` §3.5 recorded and leaves the other half standing.
4. ``ALLOWED_HISTORICAL`` carries two entries rather than three.  Wave 35's
   ``mx35_penaltydecode.py`` exemption is dropped because this wave does not
   re-author that file and its wave-35 header names a wave-34 module, which is
   not a needle this cut forbids; wave 35's ``mx35_retain.py`` exemption is
   replaced by this wave's own, at this wave's own pinned count.

**THE ONE THING WAVE 36 DOES THAT NO EARLIER CUT DID.**  It appends this wave's
four accept strings to ``experiments/seal_matrix_architecture_child.py`` **in the
same commit that cuts this tree**.  That file lives in the Codex lane, so this
cutter has no standing to assert anything about it and does not try; the append
is a hand edit made beside this one, and the reason it is made now rather than
at the first seal is that waves 34 and 35 each paid an entire deployment to
discover the same refusal after the GPU work was already done.  This paragraph
is the third consecutive wave to write that sentence and the first to act on it.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave35`` (a path component and a
schema infix) is retargeted; ``wave 35`` and ``Wave 35`` (English) are NOT, so
a docstring that records what wave 35 measured still says wave 35 measured it.
``run/mx36/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-36 header, by hand, after the
copy.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx35_verify_seals.py`` -- the rename turns ``wave35/cells`` into
  ``wave36/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  31 declared roots become 32, ``wave35/cells`` still declared.
* ``mx35_retain.py`` -- a KEEP set inherited from last wave names ``MX35-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX36-*`` tags out of wave 35's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx35_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line, the module references and the build-receipt
  filename are made by hand.  **This wave also makes its ONE behavioural edit
  here**: the composed width x MLP build, its two surviving refusals and the
  identity gate that covers it.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 35's seal calls are
  deleted by hand and the marker is rewritten to name wave 36.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  It is committed beside the tree, because "renamed
correctly" is a claim about bytes and a claim about bytes needs a receipt.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx35"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32's cut brought wave 31's nineteen
#: seal calls forward with the substitution applied and every COUNT intact.
#: Nothing downstream would have caught a wrong count -- to ``seal_verdict.sh`` a
#: count is a number -- and the only thing that stopped it was that the tags did
#: not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx35_verify_seals.py",
    "mx35_retain.py",
    "mx35_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files, the same
#: two wave 35 declared, and for the same reason: a declared-authored file that
#: nobody designed is a line that makes the set stop meaning anything.
AUTHORED_THIS_WAVE = {
    "cut_wave36.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave35-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 35", "--campaign-wave 36"),
    ('"campaign_wave": 35', '"campaign_wave": 36'),
    ("wave35", "wave36"),
    ("mx35", "mx36"),
    ("MX35", "MX36"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx35_": "a wave-35 module reference survived the copy",
    "MX35-": "a wave-35 child directory name survived the copy",
    "/wave35/": "a wave-35 artifact path survived the copy",
    "--campaign-wave 35": "a driver would seal this wave's build as wave 35",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
ALLOWED_HISTORICAL = {
    ("mx36_build.py", "mx35_"): (
        1,
        "the wave-36 header names `mx35_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    # DECLARED IN ADVANCE THIS TIME, because wave 35 was refused here and the
    # refusal was correct.  The estate's retention rule requires every DROP
    # reason to name WHAT REBUILDS the deleted child AND FROM WHICH RETAINED
    # SOURCE, and on this wave every dropped child rebuilds from a checkpoint
    # wave 35 retained -- MX35-V9CL896, MX35-V12CL896X13 -- or from a body this
    # wave builds.  So the retention file cannot obey the retention rule without
    # naming a wave-35 child directory.  The reference is PROSE INSIDE A REASON
    # STRING; the only place that file resolves a path is its own `W` root and
    # its own `MX36-*` glob, both of which the audit's other assertions cover.
    # The count is pinned rather than waived, so a wave that accidentally
    # reintroduces an executable wave-35 name still gets refused.
    ("mx36_retain.py", "MX35-"): (
        2,
        "the DROP and KEEP reasons name the wave-35 checkpoints that rebuild the "
        "deleted children, which the retention rule requires them to do; they are "
        "prose in a reason string and this file resolves no wave-35 path",
    ),
    # DECLARED AFTER THE GUARD REFUSED, and the refusal was correct -- it is the
    # SECOND consecutive wave where this needle has fired on a legitimate prose
    # reference rather than on a live path.  Two of this wave's seal notes name
    # `MX35-V9CL896` as the RETAINED SOURCE the terminal-gain arm was solved on,
    # which is provenance the seal is required to carry: an arm whose source is
    # unnamed is an arm nobody can re-derive.  `sealq.sh` resolves exactly two
    # paths, `$W` and `$MX36`, and both are this wave's; the count is pinned
    # rather than waived so an accidentally reintroduced executable wave-35 name
    # still gets refused.
    ("sealq.sh", "MX35-"): (
        2,
        "two seal notes name the RETAINED wave-35 checkpoint each terminal-gain "
        "arm was solved on, which is provenance the seal is required to carry; "
        "they are prose in a note string and this file resolves no wave-35 path",
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
        "schema": "neural-forge-matrix-wave36-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 35` / `Wave 35` / `WAVE 35` in English are NOT substituted, so "
            "a docstring recording what wave 35 measured still says wave 35 did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave36-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave36-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-35 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave36.py", "wave36-tree-cut-manifest.json"):
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
            if allowed is not None and hits == allowed[0]:
                print(f"  declared historical reference: {path.name} {needle!r} "
                      f"x{hits} -- {allowed[1]}")
                continue
            problems.append(f"{path.name}: {why} ({needle!r} x{hits})")
    # The seal verifier must STILL carry every earlier wave, or the hand edit did
    # not happen and the rename did.  Five roots are checked by name: the four a
    # rename would have eaten in the last four waves, and this wave's own.
    for name, needle in (("mx36_verify_seals.py", 'ROOT / "wave32" / "cells"'),
                         ("mx36_verify_seals.py", 'ROOT / "wave33" / "cells"'),
                         ("mx36_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx36_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx36_verify_seals.py", 'ROOT / "wave36" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx36_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave36-residual-basis-v1",
                   "neural-forge-matrix-wave35-residual-basis-v1",
                   "neural-forge-matrix-wave34-residual-basis-v1",
                   "neural-forge-matrix-wave33-residual-basis-v1",
                   "neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx36_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.  The build
    # receipt is written under a LITERAL filename, and that literal said
    # `mx32-build-receipt.json` from wave 32 through wave 33's hand edit.  Every
    # known-stale name stays pinned at zero, because the defect that hid for two
    # waves hid behind a check that did not exist for either of them.
    for needle, want, why in (
        ('(outdir / "mx36-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx36_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 36" not in body:
            problems.append(f"{name} does not carry --campaign-wave 36")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it; wave 33 was the first cut
    # where it fired.  `sealq.sh` arrives VERBATIM, so it arrives carrying LAST
    # wave's marker and LAST wave's seal calls, and this assertion forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 36 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # The terminal-gain driver was dropped after wave 29 and re-authored in wave
    # 33.  This cut carries it; if a future cut loses it again, this fires.
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    # WAVE 34's ADDITION, carried unchanged.  The second review surface is two
    # banks and two rubrics and the v1 pair is FROZEN: a cut that quietly edited
    # the v1 rubric would rewrite the scoring of sixty committed observations.
    for name in ("mx36_extscore.py", "mx36_extrubric2.py", "mx36_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx36_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx36_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.  `mx36_penaltydecode.py` is the only module in
    # this tree that decodes a sealed child through a knob the frozen protocol
    # does not have.  Wave 36 does not edit it and does not run it -- the clause
    # it prices is settled -- but the three things that keep it a diagnostic
    # rather than a second scoring standard are asserted anyway, because a
    # carried file nobody checks is a file that quietly changes.
    pen = (DST / "mx36_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx36_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S OWN ADDITION, and it guards the ONE file this wave edits
    # behaviourally.  `NOTES-MX-AM` §3.5 records a build refusal that this wave
    # LIFTS IN HALF: `--hidden-size` may now be composed with `--ff`, and may
    # still NOT be composed with `--head-dim`.  Everything that makes the lifted
    # half legitimate is text in the builder -- the surviving refusal, the
    # sequential-order comment that says WHY the composition is well posed, the
    # exactness assertion on the reader fold that the composed selection rests
    # on, and the identity precondition on the child's own gated intermediate --
    # so each is asserted by name.  A hand edit that lifted the guard and dropped
    # any of them would produce a builder that composes two axes with nothing
    # checking that the composition means anything.
    for needle, want, why in (
        ("--kv-groups and --head-dim both rewrite the attention block", 1,
         "the head-count/head-dim refusal must survive"),
        ("--hidden-size already re-solves every reader and every writer", 1,
         "the SURVIVING half of wave 35's refusal -- --hidden-size with "
         "--head-dim -- must still fire in those words"),
        ("def narrow_mlp_site(", 1,
         "the composed width x MLP site is this wave's whole build; without it "
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
                f"mx36_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-35 identifier survived; five seal roots declared; "
          "the accept list kept eight schemas; the drivers name wave 36; "
          "sealq.sh carries this wave's marker; gainq.sh exists; "
          "both extension surfaces have a scorer and v1 still names the v1 bank; "
          "the penalty diagnostic kept its NOT_A_FOLD banner and its identity "
          "refusal; the composed width x MLP path carries its surviving refusal, "
          "its exactness assertion and both identity preconditions")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
