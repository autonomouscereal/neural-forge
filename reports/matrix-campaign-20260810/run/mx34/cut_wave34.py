#!/usr/bin/env python3
"""Cut wave 34's tree from wave 33's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-33 cutter, in full.**

1. ``SRC`` is the wave-33 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files and every one of them is exercised.
   ``sealq.sh`` arrives VERBATIM carrying wave 33's marker and wave 33's six
   seal calls, and ``--audit`` REFUSES until both are replaced by hand -- the
   second wave in a row that protection does real work.
3. ``AUTHORED_THIS_WAVE`` carries the five files wave 34 wrote for itself.  The
   terminal-gain driver ``gainq.sh`` is NOT in it this time: wave 33 re-authored
   it into the tree, so this cut carries it forward like any other driver, and
   the audit still asserts it survived.
4. ``ALLOWED_HISTORICAL`` starts with the build header only.  Wave 33's waiver
   for the paraphrase rubric's docstring is not needed: that docstring names
   ``mx32_extscore.py``, and ``mx32_`` is not a forbidden needle for a cut whose
   ancestor is wave 33.  The needle that matters here is ``mx33_``.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave33`` (a path component and a
schema infix) is retargeted; ``wave 33`` and ``Wave 33`` (English) are NOT, so
a docstring that records what wave 33 measured still says wave 33 measured it.
``run/mx34/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-34 header, by hand, after the
copy.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx33_verify_seals.py`` -- the rename turns ``wave33/cells`` into
  ``wave34/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  29 declared roots become 30, ``wave33/cells`` still declared.
* ``mx33_retain.py`` -- a KEEP set inherited from last wave names ``MX33-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX34-*`` tags out of wave 33's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx33_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line and this wave's ``--select-rows`` arm are made
  by hand.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 33's seal calls are
  deleted by hand and the marker is rewritten to name wave 34.

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

SRC = Path(__file__).resolve().parent.parent / "mx33"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32's cut brought wave 31's nineteen
#: seal calls forward with the substitution applied and every COUNT intact.
#: Nothing downstream would have caught a wrong count -- to ``seal_verdict.sh`` a
#: count is a number -- and the only thing that stopped it was that the tags did
#: not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx33_verify_seals.py",
    "mx33_retain.py",
    "mx33_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  ``gainq.sh`` is NOT
#: here: wave 33 wrote it into ``run/mx33``, so it is carried by the cut like
#: every other driver and the audit asserts it arrived.
AUTHORED_THIS_WAVE = {
    "cut_wave34.py",
    "README.md",
    "mx34_extbank2.py",
    "mx34_extrubric2.py",
    "mx34_calibrate.py",
    "mx34_robustforensics.py",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave33-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 33", "--campaign-wave 34"),
    ('"campaign_wave": 33', '"campaign_wave": 34'),
    ("wave33", "wave34"),
    ("mx33", "mx34"),
    ("MX33", "MX34"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx33_": "a wave-33 module reference survived the copy",
    "MX33-": "a wave-33 child directory name survived the copy",
    "/wave33/": "a wave-33 artifact path survived the copy",
    "--campaign-wave 33": "a driver would seal this wave's build as wave 33",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
ALLOWED_HISTORICAL = {
    ("mx34_build.py", "mx33_"): (
        1,
        "the wave-34 header names `mx33_build.py` as the file it states its diff "
        "against; it resolves no path",
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
        "schema": "neural-forge-matrix-wave34-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 33` / `Wave 33` / `WAVE 33` in English are NOT substituted, so "
            "a docstring recording what wave 33 measured still says wave 33 did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave34-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave34-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-33 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave34.py", "wave34-tree-cut-manifest.json"):
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
    # not happen and the rename did.  Three roots are checked by name: the two a
    # rename would have eaten in the last two waves, and this wave's own.
    for name, needle in (("mx34_verify_seals.py", 'ROOT / "wave32" / "cells"'),
                         ("mx34_verify_seals.py", 'ROOT / "wave33" / "cells"'),
                         ("mx34_verify_seals.py", 'ROOT / "wave34" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx34_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave34-residual-basis-v1",
                   "neural-forge-matrix-wave33-residual-basis-v1",
                   "neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx34_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 34" not in body:
            problems.append(f"{name} does not carry --campaign-wave 34")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it; wave 33 was the first cut
    # where it fired.  `sealq.sh` arrives VERBATIM, so it arrives carrying LAST
    # wave's marker and LAST wave's seal calls, and this assertion forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 34 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # The terminal-gain driver was dropped after wave 29 and re-authored in wave
    # 33.  This cut carries it; if a future cut loses it again, this fires.
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    # WAVE 34's OWN ADDITION.  The second review surface is now two banks and two
    # rubrics, and the v1 pair is FROZEN: its rubric was sealed in one commit with
    # its bank before any decode, and four waves of output have been read off it.
    # A cut that quietly edited the v1 rubric would rewrite the scoring of sixty
    # committed observations.  So all three files are asserted present and two of
    # the v1 rubric's own thirty keys are asserted by name -- the substitution list
    # cannot reach an `ext-` id, and if one ever went missing the sixty
    # observations would silently lose an entry.
    for name in ("mx34_extscore.py", "mx34_extrubric2.py", "mx34_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx34_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx34_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-33 identifier survived; three seal roots declared; "
          "the accept list kept six schemas; the drivers name wave 34; "
          "sealq.sh carries this wave's marker; gainq.sh exists; "
          "both extension surfaces have a scorer and v1 still names the v1 bank")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
