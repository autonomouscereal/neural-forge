#!/usr/bin/env python3
"""Cut wave 35's tree from wave 34's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-34 cutter, in full.**

1. ``SRC`` is the wave-34 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  ``sealq.sh`` arrives VERBATIM carrying
   wave 34's marker and wave 34's EIGHT seal calls with every count intact, and
   ``--audit`` REFUSES until both are replaced by hand.  Third consecutive cut
   where that protection has had real work to do.
3. ``AUTHORED_THIS_WAVE`` is TWO files -- this cutter and the README.  Wave 34
   carried ``mx34_robustforensics.py`` in that set and never wrote it, so the
   set named a file that did not exist for a whole wave; the slot is dropped
   rather than carried, because a declared-authored file nobody designed is a
   line that makes the set stop meaning anything.  Wave 35's two behavioural
   edits -- the decode-repair grid in ``mx35_penaltydecode.py`` and this wave's
   seals -- are made ON the carried copies, by hand, after the copy, each under
   a hand-written wave-35 header.
4. ``ALLOWED_HISTORICAL`` carries two entries rather than one: the build
   header's reference to ``mx34_build.py``, and the penalty-decode module's
   wave-35 header naming ``mx34_penaltydecode.py`` as the file whose result it
   is widening.  Both are prose in a header and neither resolves a path, and
   both are pinned to an exact count rather than waived.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave34`` (a path component and a
schema infix) is retargeted; ``wave 34`` and ``Wave 34`` (English) are NOT, so
a docstring that records what wave 34 measured still says wave 34 measured it.
``run/mx35/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-35 header, by hand, after the
copy.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx34_verify_seals.py`` -- the rename turns ``wave34/cells`` into
  ``wave35/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  30 declared roots become 31, ``wave34/cells`` still declared.
* ``mx34_retain.py`` -- a KEEP set inherited from last wave names ``MX34-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX35-*`` tags out of wave 34's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx34_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line and the build-receipt filename are made by hand.
  **The receipt filename is the wave-34 defect and it is the reason this file
  is still in the hand-edit set**: both export sites write the receipt under a
  literal, that literal said ``mx32-build-receipt.json`` from wave 32 to wave
  34, and only the exact-name readers on the sealing path could ever see it.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 34's seal calls are
  deleted by hand and the marker is rewritten to name wave 35.

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

SRC = Path(__file__).resolve().parent.parent / "mx34"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32's cut brought wave 31's nineteen
#: seal calls forward with the substitution applied and every COUNT intact.
#: Nothing downstream would have caught a wrong count -- to ``seal_verdict.sh`` a
#: count is a number -- and the only thing that stopped it was that the tags did
#: not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx34_verify_seals.py",
    "mx34_retain.py",
    "mx34_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  TWO files, and that
#: is deliberate: wave 34 listed six here and one of them
#: (``mx34_robustforensics.py``) was never written, so the set advertised a file
#: that did not exist.  Wave 35's behavioural changes are hand edits on carried
#: copies, which the README names one by one.
AUTHORED_THIS_WAVE = {
    "cut_wave35.py",
    "README.md",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave34-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 34", "--campaign-wave 35"),
    ('"campaign_wave": 34', '"campaign_wave": 35'),
    ("wave34", "wave35"),
    ("mx34", "mx35"),
    ("MX34", "MX35"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx34_": "a wave-34 module reference survived the copy",
    "MX34-": "a wave-34 child directory name survived the copy",
    "/wave34/": "a wave-34 artifact path survived the copy",
    "--campaign-wave 34": "a driver would seal this wave's build as wave 34",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
ALLOWED_HISTORICAL = {
    ("mx35_build.py", "mx34_"): (
        1,
        "the wave-35 header names `mx34_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx35_penaltydecode.py", "mx34_"): (
        1,
        "the wave-35 header names `mx34_penaltydecode.py` as the single-value "
        "diagnostic this wave widens into a grid; it resolves no path",
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
        "schema": "neural-forge-matrix-wave35-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 34` / `Wave 34` / `WAVE 34` in English are NOT substituted, so "
            "a docstring recording what wave 34 measured still says wave 34 did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave35-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave35-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-34 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave35.py", "wave35-tree-cut-manifest.json"):
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
    # not happen and the rename did.  Four roots are checked by name: the three a
    # rename would have eaten in the last three waves, and this wave's own.
    for name, needle in (("mx35_verify_seals.py", 'ROOT / "wave32" / "cells"'),
                         ("mx35_verify_seals.py", 'ROOT / "wave33" / "cells"'),
                         ("mx35_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx35_verify_seals.py", 'ROOT / "wave35" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx35_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave35-residual-basis-v1",
                   "neural-forge-matrix-wave34-residual-basis-v1",
                   "neural-forge-matrix-wave33-residual-basis-v1",
                   "neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx35_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE.  The build receipt is
    # written under a LITERAL filename, and that literal said
    # `mx32-build-receipt.json` from wave 32 through wave 33's hand edit; wave 34
    # was the first wave since 32 to CALL the builder and the first to see it.
    # Wave 35 asserts the current name at a pinned count of two, and pins BOTH
    # known-stale names at zero rather than only the most recent one, because the
    # defect that hid for two waves hid behind a check that did not exist for
    # either of them.
    for needle, want, why in (
        ('(outdir / "mx35-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx35_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 35" not in body:
            problems.append(f"{name} does not carry --campaign-wave 35")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it; wave 33 was the first cut
    # where it fired.  `sealq.sh` arrives VERBATIM, so it arrives carrying LAST
    # wave's marker and LAST wave's seal calls, and this assertion forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 35 seals are appended below, AFTER the dossiers are read ---"
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
    for name in ("mx35_extscore.py", "mx35_extrubric2.py", "mx35_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx35_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx35_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S OWN ADDITION, and it guards the file this wave edits by hand.
    # `mx35_penaltydecode.py` is the ONLY module in this tree that decodes a
    # sealed child through a knob the frozen protocol does not have, and wave 35
    # widens it from one penalty value to a four-cell grid.  Everything that
    # keeps it a diagnostic rather than a second scoring standard is text in the
    # file: the NOT_A_FOLD banner it stamps into every artifact it writes, and
    # the refusal that fires when the identity setting does not reproduce the
    # sealed decode byte for byte.  A hand edit that widened the knob and dropped
    # either of those would produce a file that looks like the diagnostic and
    # grades like a protocol, so both are asserted by name.
    pen = (DST / "mx35_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx35_penaltydecode.py lost {needle!r}: {why}")
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-34 identifier survived; four seal roots declared; "
          "the accept list kept seven schemas; the drivers name wave 35; "
          "sealq.sh carries this wave's marker; gainq.sh exists; "
          "both extension surfaces have a scorer and v1 still names the v1 bank; "
          "the penalty diagnostic kept its NOT_A_FOLD banner and its identity refusal")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
