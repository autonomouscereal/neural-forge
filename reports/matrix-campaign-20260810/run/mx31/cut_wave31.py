#!/usr/bin/env python3
"""Cut wave 31's tree from wave 30's, by an EXPLICIT substitution list.

Wave 30 cut its tree with a blanket rename and the rename broke three things:
it deleted ``wave29/cells`` from the seal verifier's declared roots (wave 20's
mistake, wave 25's mistake, made a third time), it dropped wave 29's string
from the append-only ``BASIS_SCHEMAS_ACCEPTED`` list, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it.
`NOTES-MX-AF` §9 named the fix and did not apply it.  This file is the fix.

Two rules, and they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave30`` (a path component and a
schema infix) is retargeted; ``wave 30`` and ``Wave 30`` (English) are NOT, so
a docstring that records what wave 30 measured still says wave 30 measured it.
The cost of that rule is that some headers below now describe their ancestor's
wave rather than this one -- which is true, and ``run/mx31/README.md`` carries
the provenance instead.  Every file this wave CHANGES BEHAVIOURALLY gets a
hand-written wave-31 header, by hand, after the copy.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx31_verify_seals.py`` -- the rename turns ``wave30/cells`` into
  ``wave31/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Copied verbatim; the only edit is ONE APPENDED LINE, made by hand.
  26 declared roots become 27, ``wave30/cells`` still declared.
* ``mx31_retain.py`` -- a KEEP set inherited from last wave names ``MX30-*``
  tags that do not exist this wave, and everything unnamed is DELETED.  Copied
  verbatim so the rename cannot manufacture ``MX31-*`` tags out of wave 30's
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

SRC = Path(__file__).resolve().parent.parent / "mx30"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx30_verify_seals.py",
    "mx30_retain.py",
    "mx30_build.py",
}
#: Authored by this wave; the copier never overwrites them.
AUTHORED_THIS_WAVE = {"cut_wave31.py", "README.md"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 30", "--campaign-wave 31"),
    ('"campaign_wave": 29', '"campaign_wave": 31'),
    ("wave30", "wave31"),
    ("mx30", "mx31"),
    ("MX30", "MX31"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.  Checked
#: after the copy on every file, hand-edit set included.
FORBIDDEN_AFTER = {
    "mx30_": "a wave-30 module reference survived the copy",
    "MX30-": "a wave-30 child directory name survived the copy",
    "/wave30/": "a wave-30 artifact path survived the copy",
    "--campaign-wave 30": "a driver would seal this wave's build as wave 30",
}
#: DELIBERATE historical references, declared with an exact count.  A reference
#: to last wave's file is legitimate in a header that states this wave's diff
#: against it, and illegitimate anywhere a path is resolved -- so the exception
#: is per file, per needle, and the count is pinned rather than waived.
ALLOWED_HISTORICAL = {
    ("mx31_build.py", "mx30_"): (
        1,
        "the wave-31 header names `mx30_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx31_retain.py", "MX30-"): (
        7,
        "seven DROP reasons cite `MX30-CL1792` as the retained wave-30 source the "
        "dropped child is rebuilt from.  That is the whole justification for "
        "deleting them, so the name has to be there -- and it appears only inside "
        "prose, never in KEEP, never in DROP's keys, and never in a path this "
        "script resolves.  The audit caught it on the first run after the "
        "retention policy was written, which is the mechanism working",
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
        "schema": "neural-forge-matrix-wave31-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 30` / `Wave 30` / `WAVE 30` in English are NOT substituted, so "
            "a docstring recording what wave 30 measured still says so"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "files": entries,
    }
    # `newline=""` because this workstation is Windows and the deployment is cut
    # with `core.autocrlf=false`: a manifest written with CRLF is a manifest
    # whose committed bytes differ from the bytes that ship.
    (DST / "wave31-tree-cut-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline=""
    )

    for e in entries:
        subs = e["substitutions"]
        total = sum(subs.values()) if isinstance(subs, dict) else 0
        print(f"  {e['action']:26s} {e['target']:32s} {total:4d} substitutions "
              f"{json.dumps(subs) if total else ''}")
    print(f"-> {DST / 'wave31-tree-cut-manifest.json'}")
    return 0


def audit() -> int:
    """Read every file in the destination back and refuse a surviving wave-30 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave31.py", "wave31-tree-cut-manifest.json"):
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
    # The seal verifier and the retention script must STILL carry last wave, or
    # the hand edit did not happen and the rename did.
    for name, needle in (("mx31_verify_seals.py", 'ROOT / "wave30" / "cells"'),
                         ("mx31_verify_seals.py", 'ROOT / "wave31" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx31_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave31-residual-basis-v1",
                   "neural-forge-matrix-wave30-residual-basis-v1",
                   "neural-forge-matrix-wave29-residual-basis-v1"):
        if needle not in build:
            problems.append(f"mx31_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 31" not in body:
            problems.append(f"{name} does not carry --campaign-wave 31")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-30 identifier survived; both seal roots declared; "
          "the accept list kept three schemas; the drivers name wave 31")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
