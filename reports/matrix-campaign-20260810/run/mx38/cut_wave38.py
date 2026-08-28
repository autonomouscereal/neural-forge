#!/usr/bin/env python3
"""Cut wave 38's tree from wave 37's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-37 cutter, in full.**

1. ``SRC`` is the wave-37 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  ``sealq.sh`` arrives VERBATIM carrying
   wave 37's marker and wave 37's SIX seal calls with every count intact, and
   ``--audit`` REFUSES until both are replaced by hand.  Sixth consecutive cut
   where that protection has had real work to do.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README -- exactly
   as in waves 35, 36 and 37.
4. **THIS WAVE MAKES NO BEHAVIOURAL EDIT ANYWHERE IN THE TREE, FOR THE SECOND
   CONSECUTIVE WAVE.**  Wave 36's composed width x MLP edit is CARRIED and its
   five guarding assertions are run on this wave's copy, exactly as wave 37 ran
   them.  Wave 38's build list is one ``--ff 896 --excise 9 --excise 13`` body
   and four post-processes; ``--ff`` has been in the builder since wave 15,
   ``--excise`` since wave 16, multi-index excision since wave 17, and the
   vocabulary gather and terminal-gain solve since waves 25 and 24.  Nothing it
   was ordered to do needs a code path that does not exist, and inventing one to
   look busy would put an unreviewed edit under six seals.  The mechanical diff
   on ``mx38_build.py`` is therefore the usual four schema strings, the
   identifier references, two receipt-filename sites and one appended
   accept-list line.

**WAVE 38'S OWN AUDIT ADDITION, AND WHY IT IS A DIFFERENT LINE FROM WAVE 37'S.**
Wave 37 excised block 9 -- ONE block, not the last -- and asserted wave 18's
live-mask repair by name because that is the branch a single shifting excision
depends on.  Wave 38 excises blocks 9 AND 13 in one build, which is the first
MULTI-INDEX excision this campaign has run since wave 28 and the first ever
composed with ``--ff``.  A multi-index excision depends on two further lines
that a single one does not exercise at all:

* the deletion loop must run in DESCENDING index order.  ``del layers[9]``
  followed by ``del layers[13]`` on a list that is now thirteen long either
  raises or, for a pair that both sit below the end, silently deletes the wrong
  module -- ``--excise 9 --excise 11`` in ascending order deletes 9 and then
  what used to be 12.  Descending order is the whole of the fix and it is one
  ``sorted(..., reverse=True)``.
* every surviving module's own ``layer_idx`` must be rewritten to its NEW
  position, or the KV cache and the conv state address the slots the parent had.
  One excision shifts at most the tail; two shift a longer tail and shift some
  of it twice.

Both are asserted by name at a pinned count, BEFORE the build, and wave 37's two
assertions are carried unchanged because this wave excises into that case twice
over.

**THE ONE THING WAVE 36 DID THAT NO EARLIER CUT DID, AND THIS CUT DOES AGAIN.**
It appends this wave's four accept strings to
``experiments/seal_matrix_architecture_child.py`` **in the same commit that cuts
this tree**.  That file lives in the Codex lane, so this cutter has no standing
to assert anything about it and does not try; the append is a hand edit made
beside this one.  Waves 34 and 35 each paid an entire deployment to discover the
same refusal after the GPU work was already done; waves 36 and 37 appended at the
cut and paid none.  This is the third wave to act on that paragraph rather than
write it again, which is the point at which it stops being a suggestion.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave37`` (a path component and a
schema infix) is retargeted; ``wave 37`` and ``Wave 37`` (English) are NOT, so
a docstring that records what wave 37 measured still says wave 37 measured it.
``run/mx38/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-38 header, by hand, after the
copy -- and this wave changes none, which is stated in the README rather than
demonstrated by an absence.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx37_verify_seals.py`` -- the rename turns ``wave37/cells`` into
  ``wave38/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  33 declared roots become 34, ``wave37/cells`` still declared.
* ``mx37_retain.py`` -- a KEEP set inherited from last wave names ``MX37-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX38-*`` tags out of wave 37's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx37_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line, the module references and the build-receipt
  filename are made by hand.  **This wave makes NO behavioural edit here**, and
  wave 36's five assertions and wave 37's two are carried unchanged and are
  asserted on this wave's copy alongside wave 38's own two.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 37's seal calls are
  deleted by hand and the marker is rewritten to name wave 38.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  It is committed beside the tree, because "renamed
correctly" is a claim about bytes and a claim about bytes needs a receipt.

**AND IT PARSES WHAT IT WRITES, IN THE INVOCATION THAT WRITES IT.**
`NOTES-MX-AO` section 9 is wave 37's account of reproducing wave 35's corruption
bug -- a script's stdout redirected into the path the script itself writes -- and
its instruction to the next wave is that every artifact a wave writes gets parsed
by the command that writes it.  This cutter reads its own manifest back with
``json.loads`` before it returns, so the rule is obeyed by the first file wave 38
writes rather than only by the ones a reader remembers to check.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx37"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32's cut brought wave 31's nineteen
#: seal calls forward with the substitution applied and every COUNT intact.
#: Nothing downstream would have caught a wrong count -- to ``seal_verdict.sh`` a
#: count is a number -- and the only thing that stopped it was that the tags did
#: not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx37_verify_seals.py",
    "mx37_retain.py",
    "mx37_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files, the same
#: two waves 35, 36 and 37 declared, and for the same reason: a declared-authored
#: file that nobody designed is a line that makes the set stop meaning anything.
AUTHORED_THIS_WAVE = {
    "cut_wave38.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave37-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 37", "--campaign-wave 38"),
    ('"campaign_wave": 37', '"campaign_wave": 38'),
    ("wave37", "wave38"),
    ("mx37", "mx38"),
    ("MX37", "MX38"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx37_": "a wave-37 module reference survived the copy",
    "MX37-": "a wave-37 child directory name survived the copy",
    "/wave37/": "a wave-37 artifact path survived the copy",
    "--campaign-wave 37": "a driver would seal this wave's build as wave 37",
}
#: DELIBERATE historical references, declared with a count.  A reference to last
#: wave's file is legitimate in a header that states this wave's diff against it,
#: and illegitimate anywhere a path is resolved -- so the exception is per file,
#: per needle, and the count is pinned rather than waived.
#:
#: **THE THIRD FIELD IS THE COMPARISON, AND WAVE 38 IS THE FIRST CUT TO NEED TWO
#: KINDS OF IT.**  Waves 36 and 37 pinned every exemption at an exact count and
#: both left the retention and seal exemptions UNUSED, so the question never
#: arose.  Wave 38 will use both: five of its six children are one CPU pass off
#: wave 37's RETAINED `MX37-CL896X09`, so the retention rule (*name what rebuilds
#: each deleted child, and from which retained source*) and the provenance rule
#: (*name the retained source a child was built on*) both require that directory
#: name in prose.  How MANY times is not knowable at the tree cut -- it depends
#: on which children the verdicts keep -- and an exact pin that has to be guessed
#: before the measurement is a pin that gets waived the moment it is wrong.  So
#: those two are declared as an UPPER BOUND, which still refuses an unbounded
#: leak, and the header reference, whose count IS knowable, stays exact.
ALLOWED_HISTORICAL = {
    ("mx38_build.py", "mx37_"): (
        1,
        "==",
        "the wave-38 header names `mx37_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    # DECLARED IN ADVANCE, as waves 36 and 37 declared it, because the estate's
    # retention rule requires every DROP reason to name WHAT REBUILDS the deleted
    # child AND FROM WHICH RETAINED SOURCE, and this wave's dropped children all
    # rebuild off a wave-37 checkpoint -- `MX37-CL896X09` is the retained body
    # every G1 child is one CPU pass from.  Unlike wave 37's, this declaration is
    # EXPECTED TO FIRE: the source of five of this wave's six children is a
    # wave-37 directory by name.
    ("mx38_retain.py", "MX37-"): (
        8,
        "<=",
        "the retention rule requires each DROP reason to name the retained source "
        "that rebuilds the deleted child, and five of this wave's six children "
        "rebuild from wave 37's retained MX37-CL896X09 body; bounded at eight so "
        "a ninth occurrence is a refusal rather than a shrug",
    ),
    # DECLARED IN ADVANCE for the reason wave 36 declared it AFTER being refused:
    # a terminal-gain arm whose retained source is unnamed is an arm nobody can
    # re-derive, so the seal is REQUIRED to carry the source's name.  Wave 38's
    # G1 children are vocabulary folds on a RETAINED WAVE-37 BODY, so unlike
    # waves 36 and 37 this one is expected to fire too.
    ("sealq.sh", "MX37-"): (
        6,
        "<=",
        "a seal note naming the RETAINED wave-37 checkpoint a child was built "
        "on, which is provenance the seal is required to carry.  Wave 38's G1 "
        "and G3 children are all one pass off wave 37's retained MX37-CL896X09, "
        "so the name belongs in the seal; bounded at six, one per child sealed",
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
        "schema": "neural-forge-matrix-wave38-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 37` / `Wave 37` / `WAVE 37` in English are NOT substituted, so "
            "a docstring recording what wave 37 measured still says wave 37 did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "NONE, for the second consecutive wave.  Wave 38's build list is one "
            "--ff 896 --excise 9 --excise 13 body and four post-processes; --ff "
            "has existed since wave 15, --excise since wave 16, multi-index "
            "excision since wave 17 and the vocabulary gather since wave 25.  "
            "Wave 36's composed width x MLP edit is carried unchanged, its five "
            "guarding assertions and wave 37's two live-mask assertions are "
            "asserted on this wave's copy, and wave 38 adds TWO of its own about "
            "the multi-index deletion order and the module re-index."
        ),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    out = DST / "wave38-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-37 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave38.py", "wave38-tree-cut-manifest.json"):
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
    for name, needle in (("mx38_verify_seals.py", 'ROOT / "wave33" / "cells"'),
                         ("mx38_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx38_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx38_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx38_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx38_verify_seals.py", 'ROOT / "wave38" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx38_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave38-residual-basis-v1",
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
            problems.append(f"mx38_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.  The build
    # receipt is written under a LITERAL filename, and that literal said
    # `mx32-build-receipt.json` from wave 32 through wave 33's hand edit.  Every
    # known-stale name stays pinned at zero, because the defect that hid for two
    # waves hid behind a check that did not exist for either of them.
    for needle, want, why in (
        ('(outdir / "mx38-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx38_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 38" not in body:
            problems.append(f"{name} does not carry --campaign-wave 38")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it; wave 33 was the first cut
    # where it fired.  `sealq.sh` arrives VERBATIM, so it arrives carrying LAST
    # wave's marker and LAST wave's seal calls, and this assertion forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 38 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # The terminal-gain driver was dropped after wave 29 and re-authored in wave
    # 33.  This cut carries it; if a future cut loses it again, this fires.  It is
    # not decoration on THIS wave: G3 is its output.
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    # WAVE 34's ADDITION, carried unchanged.  The second review surface is two
    # banks and two rubrics and the v1 pair is FROZEN: a cut that quietly edited
    # the v1 rubric would rewrite the scoring of sixty committed observations.
    for name in ("mx38_extscore.py", "mx38_extrubric2.py", "mx38_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx38_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx38_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.  `mx38_penaltydecode.py` is the only module in
    # this tree that decodes a sealed child through a knob the frozen protocol
    # does not have.  Wave 38 does not edit it and does not run it -- the clause
    # it prices is settled -- but the three things that keep it a diagnostic
    # rather than a second scoring standard are asserted anyway, because a
    # carried file nobody checks is a file that quietly changes.
    pen = (DST / "mx38_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx38_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S ADDITION, CARRIED UNCHANGED AND ASSERTED ON THIS WAVE'S COPY FOR
    # THE SECOND CONSECUTIVE WAVE.  Wave 36 lifted `NOTES-MX-AM` 3.5's build
    # refusal IN HALF: `--hidden-size` may be composed with `--ff`, and may still
    # NOT be composed with `--head-dim`.  Wave 38 does not touch that path and
    # does not build on it, which is exactly the condition under which a guard
    # rots -- and it is now the SECOND consecutive wave in that condition, which
    # is when a carried guard is most likely to have been quietly dropped.
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
                f"mx38_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 37'S ADDITION, CARRIED, AND THIS WAVE LEANS ON IT HARDER THAN WAVE 37
    # DID.  Wave 37 excised ONE non-final block; wave 38 excises TWO in the same
    # build, so `layer_types` must be re-derived against a body that has shifted
    # in two places.  Carried at the same pinned counts.
    for needle, want, why in (
        ("harness.model.config.layer_types = [", 1,
         "wave 18's live-mask repair re-derives layer_types after an excision so "
         "the LIVE model's causal mask follows the LIVE blocks; wave 38 excises "
         "blocks 9 AND 13, so an attention block lands in a conv's slot and the "
         "repair is load-bearing twice over"),
        ("so the live mask follows the live block", 1,
         "the repair must still SAY what it is doing in the log line the build "
         "log carries, or a wave that reads the log cannot tell whether it ran"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx38_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 38'S OWN ADDITION, and it is about the part of `--excise` that only a
    # MULTI-INDEX excision exercises.  Wave 28 was the last wave to build one and
    # no wave has ever composed one with `--ff`; wave 38's G2 body is
    # `--ff 896 --excise 9 --excise 13`.  Two lines carry it:
    #
    #   * the deletion loop must run in DESCENDING index order.  Ascending, the
    #     second `del` addresses a list that the first one already shortened --
    #     `--excise 9 --excise 11` would delete block 9 and then whatever block
    #     12 used to be, silently, with a correct-looking parameter count and a
    #     correct-looking depth.  This wave's own pair would raise instead of
    #     lying, which is luck rather than a check.
    #   * every surviving module's `layer_idx` must be rewritten to its NEW
    #     position.  `Lfm2Attention` and the conv block both address the KV cache
    #     and the conv state by that integer, and with two excisions the tail is
    #     renumbered further than any single excision renumbers it.
    #
    # Asserted BEFORE the build, at a pinned count, like every guard above.
    # The needles carry their INDENTATION, and that is not decoration.  The
    # wave-38 header of `mx38_build.py` quotes all three of these lines by name,
    # so a bare-string count would read 2 for each and a pin of 2 would still
    # pass with the CODE deleted and only the prose left.  The indented form
    # exists exactly once, inside `build()`'s `if excised:` branch.
    for needle, want, why in (
        ("        for index in sorted(excised, reverse=True):\n", 1,
         "a multi-index excision must delete in DESCENDING order or the second "
         "del addresses a list the first one shortened; wave 38 is the first "
         "wave since 28 to excise two blocks and the first ever to compose that "
         "with --ff"),
        ("                layer.self_attn.layer_idx = position\n", 1,
         "every surviving attention module must be renumbered to its NEW slot or "
         "it addresses the KV cache the parent had"),
        ("                layer.conv.layer_idx = position\n", 1,
         "every surviving conv module must be renumbered to its NEW slot or it "
         "addresses the conv state the parent had; with two excisions the tail "
         "is renumbered further than one excision renumbers it"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx38_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-37 identifier survived; six seal roots declared; "
          "the accept list kept ten schemas; the drivers name wave 38; "
          "sealq.sh carries this wave's marker; gainq.sh exists; "
          "both extension surfaces have a scorer and v1 still names the v1 bank; "
          "the penalty diagnostic kept its NOT_A_FOLD banner and its identity "
          "refusal; wave 36's composed width x MLP path still carries its "
          "surviving refusal, its exactness assertion and both identity "
          "preconditions; wave 37's live-mask repair is present; and the "
          "descending deletion order and both module re-index lines this wave's "
          "DOUBLE excision depends on are present")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
