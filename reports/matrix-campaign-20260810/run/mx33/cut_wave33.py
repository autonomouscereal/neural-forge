#!/usr/bin/env python3
"""Cut wave 33's tree from wave 32's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED`` list, and it rewrote four docstrings that recorded
what wave 29 did into claims that wave 30 did it.  The design is unchanged and
the reasons are unchanged, so this file states its diff against its ancestor
rather than restating the argument.

**The diff against `cut_wave32.py`, in full.**

1. ``SRC`` is ``mx32``, the substitution list moves one wave, and the forbidden
   list moves with it.
2. ``HAND_EDIT`` carries ``sealq.sh`` and this is the FIRST cut where that
   membership does anything.  Wave 32 discovered from its own tree that the
   copy had brought wave 31's nineteen seal calls forward with every COUNT
   intact onto cell ids in wave 32's tree; it deleted them by hand, added the
   file to the set for the next cut, and added the marker assertion below.
   This is the next cut.  ``sealq.sh`` therefore arrives VERBATIM, still
   carrying wave 32's marker and wave 32's seal calls, and ``--audit`` REFUSES
   until both are replaced by hand.  A cutter whose audit passes on the first
   run has not proved anything about the file it copied verbatim.
3. ``gainq.sh`` is AUTHORED, not carried.  The terminal-gain driver was dropped
   from the tree after wave 29 -- waves 30, 31 and 32 kept ``mx*_termgain.py``
   and lost the queue that drives it -- so wave 33 writes it again from wave
   29's, against this wave's frontier sources.  It is in
   ``AUTHORED_THIS_WAVE`` so the copier cannot overwrite it, and it is written
   before the cut runs.
4. ``ALLOWED_HISTORICAL`` starts EMPTY except for the build header, for the
   reason wave 32 gave: last wave's waiver was pinned to prose this wave has
   not written yet.  When this wave's retention policy cites a retained
   ancestor by name the audit fires and the count is pinned then.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave32`` (a path component and a
schema infix) is retargeted; ``wave 32`` and ``Wave 32`` (English) are NOT, so
a docstring that records what wave 32 measured still says wave 32 measured it.
``run/mx33/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-33 header, by hand, after the
copy.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx32_verify_seals.py`` -- the rename turns ``wave32/cells`` into
  ``wave33/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  28 declared roots become 29, ``wave32/cells`` still declared.
* ``mx32_retain.py`` -- a KEEP set inherited from last wave names ``MX32-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX33-*`` tags out of wave 32's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx32_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings
  and the appended accept-list line are made by hand.
* ``sealq.sh`` -- see (2) above.  Copied verbatim; wave 32's seal calls are
  deleted by hand and the marker is rewritten to name wave 33.

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

SRC = Path(__file__).resolve().parent.parent / "mx32"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` is in this set because wave 32 put it here after its own cut
#: brought wave 31's nineteen seal calls forward with the substitution applied
#: and every COUNT intact.  Nothing downstream would have caught a wrong count --
#: to ``seal_verdict.sh`` a count is a number -- and the only thing that stopped
#: it was that the tags did not exist in that wave's ``cells`` directory.
HAND_EDIT = {
    "mx32_verify_seals.py",
    "mx32_retain.py",
    "mx32_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  ``gainq.sh`` is
#: here because it does not exist in ``run/mx32`` at all: the terminal-gain
#: driver was last cut in wave 29 and three waves carried the solver without it.
AUTHORED_THIS_WAVE = {
    "cut_wave33.py",
    "README.md",
    "gainq.sh",
    "mx33_instrument.py",
    "mx33_pararubric.py",
    "mx33_mforensics.py",
    "mx33_condinventory.py",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave32-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 32", "--campaign-wave 33"),
    ('"campaign_wave": 32', '"campaign_wave": 33'),
    ("wave32", "wave33"),
    ("mx32", "mx33"),
    ("MX32", "MX33"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx32_": "a wave-32 module reference survived the copy",
    "MX32-": "a wave-32 child directory name survived the copy",
    "/wave32/": "a wave-32 artifact path survived the copy",
    "--campaign-wave 32": "a driver would seal this wave's build as wave 32",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
#:
#: Wave 32's two-count waiver on the retention script is NOT carried: it was
#: pinned to prose this wave has not written yet.  If this wave's retention
#: policy cites a retained ancestor, the audit fires and the count is pinned
#: then, against the prose that actually exists.
ALLOWED_HISTORICAL = {
    ("mx33_build.py", "mx32_"): (
        1,
        "the wave-33 header names `mx32_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx33_pararubric.py", "mx32_"): (
        1,
        "the paraphrase rubric's docstring names `mx32_extscore.py` as the scorer "
        "that could not key on this bank, which is the whole reason the file "
        "exists; it resolves no path and imports nothing from it.  The wave-33 "
        "code path imports `mx33_extscore` by its own name in "
        "`mx33_instrument.prompt_satisfiable`",
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
        "schema": "neural-forge-matrix-wave33-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 32` / `Wave 32` / `WAVE 32` in English are NOT substituted, so "
            "a docstring recording what wave 32 measured still says so"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave33-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave33-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-32 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave33.py", "wave33-tree-cut-manifest.json"):
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
    for name, needle in (("mx33_verify_seals.py", 'ROOT / "wave31" / "cells"'),
                         ("mx33_verify_seals.py", 'ROOT / "wave32" / "cells"'),
                         ("mx33_verify_seals.py", 'ROOT / "wave33" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx33_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave33-residual-basis-v1",
                   "neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx33_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 33" not in body:
            problems.append(f"{name} does not carry --campaign-wave 33")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # Wave 32 added this after its own cut missed it, and this is the first cut
    # where the HAND_EDIT membership that goes with it is actually exercised:
    # `sealq.sh` now arrives VERBATIM, so it arrives carrying LAST wave's marker
    # and LAST wave's seal calls, and this assertion is what forces them out.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 33 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # The terminal-gain driver was dropped after wave 29 and re-authored here.
    # If a future cut carries it forward the rename handles it; if a future cut
    # loses it again, this fires.
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-32 identifier survived; three seal roots declared; "
          "the accept list kept five schemas; the drivers name wave 33; "
          "sealq.sh carries this wave's marker; gainq.sh exists")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
