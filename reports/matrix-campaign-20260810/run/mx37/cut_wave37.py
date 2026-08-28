#!/usr/bin/env python3
"""Cut wave 37's tree from wave 36's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-36 cutter, in full.**

1. ``SRC`` is the wave-36 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  ``sealq.sh`` arrives VERBATIM carrying
   wave 36's marker and wave 36's SIX seal calls with every count intact, and
   ``--audit`` REFUSES until both are replaced by hand.  Fifth consecutive cut
   where that protection has had real work to do.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README -- exactly
   as in waves 35 and 36.
4. **THIS WAVE MAKES NO BEHAVIOURAL EDIT ANYWHERE IN THE TREE.**  Wave 36's one
   behavioural edit -- the composed width x MLP build -- is CARRIED, and every
   assertion wave 36 wrote to guard it is carried with it and widened by one
   line.  Wave 37 builds an ``--ff`` + ``--excise`` body and two post-processes
   that have existed since waves 15 and 25 respectively; nothing it was ordered
   to do needs a new code path, and inventing one to look busy would put an
   unreviewed edit under six seals.  The mechanical diff on ``mx37_build.py`` is
   therefore the usual four schema strings, fourteen identifier references, two
   receipt-filename sites and one appended accept-list line.

**THE ONE THING WAVE 36 DID THAT NO EARLIER CUT DID, AND THIS CUT DOES AGAIN.**
It appends this wave's four accept strings to
``experiments/seal_matrix_architecture_child.py`` **in the same commit that cuts
this tree**.  That file lives in the Codex lane, so this cutter has no standing
to assert anything about it and does not try; the append is a hand edit made
beside this one.  Waves 34 and 35 each paid an entire deployment to discover the
same refusal after the GPU work was already done; wave 36 appended at the cut
and paid none.  This is the second wave to act on that paragraph rather than
write it again.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave36`` (a path component and a
schema infix) is retargeted; ``wave 36`` and ``Wave 36`` (English) are NOT, so
a docstring that records what wave 36 measured still says wave 36 measured it.
``run/mx37/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-37 header, by hand, after the
copy -- and this wave changes none, which is stated in the README rather than
demonstrated by an absence.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx36_verify_seals.py`` -- the rename turns ``wave36/cells`` into
  ``wave37/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  32 declared roots become 33, ``wave36/cells`` still declared.
* ``mx36_retain.py`` -- a KEEP set inherited from last wave names ``MX36-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX37-*`` tags out of wave 36's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx36_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line, the module references and the build-receipt
  filename are made by hand.  **This wave makes NO behavioural edit here**, and
  the five assertions wave 36 added to guard its own edit are carried unchanged
  and are asserted on this wave's copy.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 36's seal calls are
  deleted by hand and the marker is rewritten to name wave 37.

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

SRC = Path(__file__).resolve().parent.parent / "mx36"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32's cut brought wave 31's nineteen
#: seal calls forward with the substitution applied and every COUNT intact.
#: Nothing downstream would have caught a wrong count -- to ``seal_verdict.sh`` a
#: count is a number -- and the only thing that stopped it was that the tags did
#: not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx36_verify_seals.py",
    "mx36_retain.py",
    "mx36_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files, the same
#: two waves 35 and 36 declared, and for the same reason: a declared-authored
#: file that nobody designed is a line that makes the set stop meaning anything.
AUTHORED_THIS_WAVE = {
    "cut_wave37.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave36-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 36", "--campaign-wave 37"),
    ('"campaign_wave": 36', '"campaign_wave": 37'),
    ("wave36", "wave37"),
    ("mx36", "mx37"),
    ("MX36", "MX37"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx36_": "a wave-36 module reference survived the copy",
    "MX36-": "a wave-36 child directory name survived the copy",
    "/wave36/": "a wave-36 artifact path survived the copy",
    "--campaign-wave 36": "a driver would seal this wave's build as wave 36",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
ALLOWED_HISTORICAL = {
    ("mx37_build.py", "mx36_"): (
        1,
        "the wave-37 header names `mx36_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    # DECLARED IN ADVANCE, as wave 36 declared it, because the estate's retention
    # rule requires every DROP reason to name WHAT REBUILDS the deleted child AND
    # FROM WHICH RETAINED SOURCE, and a wave whose dropped children rebuild off
    # last wave's checkpoints cannot obey that rule without naming a wave-36 child
    # directory in prose.
    #
    # **THIS DECLARATION WENT UNUSED AND THE FACT IS RECORDED RATHER THAN THE
    # DECLARATION DELETED.**  Wave 37's retention was written after its verdicts
    # were sealed and every one of its seven dropped children rebuilds from wave
    # 34's `MX34-CL896` or from a wave-37 checkpoint the KEEP set holds; no
    # wave-36 checkpoint is named.  `MX34-` is not a needle this cut forbids, so
    # the retention file contains ZERO occurrences of the exempted string and the
    # audit never consults this entry.  It is left in place, at its pinned count,
    # because a pre-declaration that turned out to be unnecessary is a correct
    # prediction to have made and deleting it would hide that the wave checked.
    ("mx37_retain.py", "MX36-"): (
        2,
        "PRE-DECLARED AND UNUSED.  Reserved for KEEP/DROP reasons naming a "
        "wave-36 checkpoint as the retained source that rebuilds a deleted child, "
        "which the retention rule requires when it applies.  Wave 37's drops all "
        "rebuild from wave 34's body or from this wave's own kept body, so the "
        "count on disk is zero and this entry never fires",
    ),
    # DECLARED IN ADVANCE for the same reason wave 36 declared it AFTER being
    # refused: a terminal-gain arm whose retained source is unnamed is an arm
    # nobody can re-derive, so the seal is REQUIRED to carry the source's name.
    #
    # **ALSO UNUSED.**  Wave 37's six seals are all wave-37 cells solved on
    # wave-37 sources; the F3 adjudication that DOES read a retained wave-36
    # checkpoint produced no seal, because `ARCH.V9CL896G10` is wave 36's cell and
    # this wave does not re-seal it.  `sealq.sh` resolves exactly two paths, `$W`
    # and `$MX37`, and both are this wave's.
    ("sealq.sh", "MX36-"): (
        1,
        "PRE-DECLARED AND UNUSED.  Reserved for a seal note naming the RETAINED "
        "wave-36 checkpoint an arm was solved on, which is provenance the seal is "
        "required to carry when it applies.  Wave 37's arms are all solved on "
        "wave-37 sources, so the count on disk is zero and this entry never fires",
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
        "schema": "neural-forge-matrix-wave37-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 36` / `Wave 36` / `WAVE 36` in English are NOT substituted, so "
            "a docstring recording what wave 36 measured still says wave 36 did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "NONE.  Wave 37's build list is an --ff + --excise body and two "
            "post-processes that have existed since waves 15 and 25; no order it "
            "was given needs a new code path.  Wave 36's composed width x MLP "
            "edit is carried unchanged and its five guarding assertions are "
            "asserted on this wave's copy."
        ),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave37-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave37-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-36 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave37.py", "wave37-tree-cut-manifest.json"):
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
    # not happen and the rename did.  Six roots are checked by name: the five a
    # rename would have eaten in the last five waves, and this wave's own.
    for name, needle in (("mx37_verify_seals.py", 'ROOT / "wave32" / "cells"'),
                         ("mx37_verify_seals.py", 'ROOT / "wave33" / "cells"'),
                         ("mx37_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx37_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx37_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx37_verify_seals.py", 'ROOT / "wave37" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx37_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave37-residual-basis-v1",
                   "neural-forge-matrix-wave36-residual-basis-v1",
                   "neural-forge-matrix-wave35-residual-basis-v1",
                   "neural-forge-matrix-wave34-residual-basis-v1",
                   "neural-forge-matrix-wave33-residual-basis-v1",
                   "neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx37_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.  The build
    # receipt is written under a LITERAL filename, and that literal said
    # `mx32-build-receipt.json` from wave 32 through wave 33's hand edit.  Every
    # known-stale name stays pinned at zero, because the defect that hid for two
    # waves hid behind a check that did not exist for either of them.
    for needle, want, why in (
        ('(outdir / "mx37-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx37_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 37" not in body:
            problems.append(f"{name} does not carry --campaign-wave 37")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it; wave 33 was the first cut
    # where it fired.  `sealq.sh` arrives VERBATIM, so it arrives carrying LAST
    # wave's marker and LAST wave's seal calls, and this assertion forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 37 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # The terminal-gain driver was dropped after wave 29 and re-authored in wave
    # 33.  This cut carries it; if a future cut loses it again, this fires.  It is
    # not decoration on THIS wave: four of wave 37's six children are its output.
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    # WAVE 34's ADDITION, carried unchanged.  The second review surface is two
    # banks and two rubrics and the v1 pair is FROZEN: a cut that quietly edited
    # the v1 rubric would rewrite the scoring of sixty committed observations.
    for name in ("mx37_extscore.py", "mx37_extrubric2.py", "mx37_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx37_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx37_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.  `mx37_penaltydecode.py` is the only module in
    # this tree that decodes a sealed child through a knob the frozen protocol
    # does not have.  Wave 37 does not edit it and does not run it -- the clause
    # it prices is settled -- but the three things that keep it a diagnostic
    # rather than a second scoring standard are asserted anyway, because a
    # carried file nobody checks is a file that quietly changes.
    pen = (DST / "mx37_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx37_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S ADDITION, CARRIED UNCHANGED AND ASSERTED ON THIS WAVE'S COPY.
    # Wave 36 lifted `NOTES-MX-AM` 3.5's build refusal IN HALF: `--hidden-size`
    # may be composed with `--ff`, and may still NOT be composed with
    # `--head-dim`.  Wave 37 does not touch that path and does not build on it,
    # which is exactly the condition under which a guard rots -- so the five
    # assertions are run on this wave's copy rather than retired as last wave's
    # business.  A cut that silently dropped any of them would leave a builder
    # that composes two axes with nothing checking that the composition means
    # anything, and the next wave to use that path would not know.
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
                f"mx37_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 37'S OWN ADDITION, and it is about the ONE code path this wave's build
    # list actually exercises that no recent wave has: `--excise` on a block that
    # is NOT the last one.  Every excision this campaign has sealed since wave 26
    # has been block 13 -- the final conv -- and removing the final block shifts
    # nothing.  Wave 37 excises block 9, which shifts blocks 10-13 down one slot
    # and therefore moves an attention block into a conv's index.  Wave 29's
    # `bodyq.sh` records what happens when a builder does not re-derive
    # `layer_types` after that kind of excision: wave 17's builder left the live
    # model's causal mask following the PARENT's block types, and 28.6-48.0% of
    # one module's attention mass landed on tokens that had not happened yet.
    # Wave 18 repaired it; the repair is a two-line branch and nothing since has
    # asserted that it is still there.  This wave excises into that exact case,
    # so it asserts it by name BEFORE it builds rather than discovering it in a
    # dossier.
    for needle, want, why in (
        ("harness.model.config.layer_types = [", 1,
         "wave 18's live-mask repair re-derives layer_types after an excision so "
         "the LIVE model's causal mask follows the LIVE blocks; wave 37 excises "
         "block 9, which shifts an attention block into a conv's slot, and that "
         "is the exact case the repair exists for"),
        ("so the live mask follows the live block", 1,
         "the repair must still SAY what it is doing in the log line the build "
         "log carries, or a wave that reads the log cannot tell whether it ran"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx37_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-36 identifier survived; six seal roots declared; "
          "the accept list kept nine schemas; the drivers name wave 37; "
          "sealq.sh carries this wave's marker; gainq.sh exists; "
          "both extension surfaces have a scorer and v1 still names the v1 bank; "
          "the penalty diagnostic kept its NOT_A_FOLD banner and its identity "
          "refusal; wave 36's composed width x MLP path still carries its "
          "surviving refusal, its exactness assertion and both identity "
          "preconditions; and the live-mask repair this wave's excision depends "
          "on is present")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
