#!/usr/bin/env python3
"""Cut wave 32's tree from wave 31's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED`` list, and it rewrote four docstrings that recorded
what wave 29 did into claims that wave 30 did it.  The design is unchanged and
the reasons are unchanged, so this file states its diff against its ancestor
rather than restating the argument.

**The diff against `cut_wave31.py`, in full.**

1. ``SRC`` is ``mx31``, the substitution list moves one wave, and the forbidden
   list moves with it.
2. ``NOT_CARRIED`` is new.  Wave 31's tree contains
   ``wave31-tree-cut-manifest.json``; a rename would copy it forward as
   ``wave32-tree-cut-manifest.json`` and this script would then overwrite it at
   the end with the real manifest.  The final bytes would be right and the
   manifest's own ``files`` list would carry a fictional entry saying it renamed
   itself.  It is declared not-carried instead: wave 31's manifest is wave 31's
   receipt and stays in wave 31's tree.
3. ``ALLOWED_HISTORICAL`` starts EMPTY except for the build header.  Wave 31's
   seven-count waiver on ``mx31_retain.py`` was pinned to a retention policy
   that does not exist yet this wave; carrying the count forward would waive a
   file whose contents are about to be emptied by hand.  When this wave's
   retention policy is written and cites a retained ancestor by name, the count
   is pinned then -- the audit is supposed to fire first and be answered, which
   is what happened in wave 31.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave31`` (a path component and a
schema infix) is retargeted; ``wave 31`` and ``Wave 31`` (English) are NOT, so
a docstring that records what wave 31 measured still says wave 31 measured it.
``run/mx32/README.md`` carries the provenance instead.  Every file this wave
CHANGES BEHAVIOURALLY gets a hand-written wave-32 header, by hand, after the
copy.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx31_verify_seals.py`` -- the rename turns ``wave31/cells`` into
  ``wave32/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  27 declared roots become 28, ``wave31/cells`` still declared.
* ``mx31_retain.py`` -- a KEEP set inherited from last wave names ``MX31-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX32-*`` tags out of wave 31's
  verdicts; KEEP and DROP are emptied by hand and the wave root retargeted by
  hand, and the script refuses until this wave's verdicts are sealed.
* ``mx31_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and the
  substitution rewrites its first entry.  Copied verbatim; the schema strings
  and the appended accept-list line are made by hand.

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

SRC = Path(__file__).resolve().parent.parent / "mx31"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
#:
#: ``sealq.sh`` BELONGS IN THIS SET AND WAS NOT IN IT.  Wave 32 discovered that
#: from its own tree: the copy brought wave 31's nineteen seal calls forward with
#: `mx31`->`mx32` applied and every COUNT intact, onto cell ids in wave 32's
#: tree, and wave 31's own comment in that file says wave 30's calls were
#: DELETED rather than renamed and says why.  The calls were removed by hand
#: before any wave-32 seal was written, and ``audit()`` now asserts the seal
#: block's marker names this wave.  It is listed here for the next cut, which
#: reads this dict rather than the paragraph.
HAND_EDIT = {
    "mx31_verify_seals.py",
    "mx31_retain.py",
    "mx31_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.
AUTHORED_THIS_WAVE = {"cut_wave32.py", "README.md"}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt: renaming it forward and then overwriting it would leave a manifest
#: whose own file list says it renamed itself.
NOT_CARRIED = {"wave31-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 31", "--campaign-wave 32"),
    ('"campaign_wave": 31', '"campaign_wave": 32'),
    ("wave31", "wave32"),
    ("mx31", "mx32"),
    ("MX31", "MX32"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx31_": "a wave-31 module reference survived the copy",
    "MX31-": "a wave-31 child directory name survived the copy",
    "/wave31/": "a wave-31 artifact path survived the copy",
    "--campaign-wave 31": "a driver would seal this wave's build as wave 31",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
#:
#: Wave 31's seven-count waiver on the retention script is NOT carried: it was
#: pinned to prose this wave has not written yet.  If this wave's retention
#: policy cites a retained ancestor, the audit fires and the count is pinned
#: then, against the prose that actually exists.
ALLOWED_HISTORICAL = {
    ("mx32_build.py", "mx31_"): (
        1,
        "the wave-32 header names `mx31_build.py` as the file it states its diff "
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
        "schema": "neural-forge-matrix-wave32-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 31` / `Wave 31` / `WAVE 31` in English are NOT substituted, so "
            "a docstring recording what wave 31 measured still says so"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave32-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {str(e['target']):32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave32-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-31 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave32.py", "wave32-tree-cut-manifest.json"):
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
    for name, needle in (("mx32_verify_seals.py", 'ROOT / "wave30" / "cells"'),
                         ("mx32_verify_seals.py", 'ROOT / "wave31" / "cells"'),
                         ("mx32_verify_seals.py", 'ROOT / "wave32" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx32_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave32-residual-basis-v1",
                   "neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx32_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 32" not in body:
            problems.append(f"{name} does not carry --campaign-wave 32")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    # ADDED AFTER THIS WAVE'S OWN CUT MISSED IT.  `sealq.sh` is not in HAND_EDIT,
    # so the copy brought wave 31's nineteen seal calls forward with the
    # substitution applied and every COUNT intact, onto cell ids in this wave's
    # tree.  Nothing downstream would have caught a wrong count -- to
    # `seal_verdict.sh` a count is a number -- and the only thing that would have
    # stopped it is that the tags do not exist in `wave32/cells`.  The block is
    # delimited by a marker naming the wave it belongs to; asserting the marker
    # names THIS wave turns a copied block into a refusal.
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 32 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-31 identifier survived; three seal roots declared; "
          "the accept list kept four schemas; the drivers name wave 32")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
