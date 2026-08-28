#!/usr/bin/env python3
"""Cut wave 29's tree from wave 28's, with the hand-edit set named BY FILE.

Waves 26, 27 and 28 each cut their tree this way and each wrote down why: a
blanket rename over the whole directory is correct for thirty-odd files and
actively destructive for exactly two.

* ``mx29_verify_seals.py`` -- the rename turns ``wave28/cells`` into
  ``wave29/cells`` and thereby DELETES last wave's seal root from the declared
  list.  Wave 20 did this and lost a root; wave 25 did it again.  The file is
  copied VERBATIM and the only edit is ONE APPENDED LINE.  24 declared roots
  become 25, ``wave28/cells`` still declared.
* ``mx29_retain.py`` -- a KEEP set inherited from last wave names tags that
  either do not exist this wave or name different children, and everything
  unnamed is deleted.  KEEP and DROP are emptied so the script REFUSES until
  this wave's verdicts are sealed.

Two files are NOT copied because wave 29 authored them fresh:
``mx29_author_bank.py`` and ``mx29_extreview.py``.  Anything already present in
the destination is left alone; the copier never overwrites this wave's own work.

The assertions at the end are the ones waves 27 and 28 added after the failures
that motivated each: a driver that still names last wave's campaign number seals
last wave's build, and a preread whose triples resolve under ``$W`` prices this
wave's children against paths that do not exist yet.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx28"
DST = Path(__file__).resolve().parent

HAND_EDIT = {"mx28_verify_seals.py", "mx28_retain.py"}
AUTHORED_THIS_WAVE = {"mx29_author_bank.py", "mx29_extreview.py", "cut_wave29.py"}

SUBS = [
    ("mx28", "mx29"),
    ("MX28", "MX29"),
    ("Mx28", "Mx29"),
    ("wave28", "wave29"),
    ("wave-28", "wave-29"),
    ("Wave 28", "Wave 29"),
    ("wave 28", "wave 29"),
    ("WAVE 28", "WAVE 29"),
    ("--campaign-wave 28", "--campaign-wave 29"),
    ('"campaign_wave": 28', '"campaign_wave": 29'),
]


def rename(text: str) -> str:
    for old, new in SUBS:
        text = text.replace(old, new)
    return text


def main() -> int:
    if not SRC.is_dir():
        raise SystemExit(f"REFUSED: no source tree at {SRC}")
    copied: list[str] = []
    hand: list[str] = []
    skipped: list[str] = []
    for path in sorted(SRC.iterdir()):
        if not path.is_file():
            continue
        target_name = rename(path.name)
        target = DST / target_name
        if target_name in AUTHORED_THIS_WAVE:
            skipped.append(target_name)
            continue
        if target.exists():
            skipped.append(target_name)
            continue
        raw = path.read_bytes()
        if path.name in HAND_EDIT:
            # Copied VERBATIM.  The edit is applied below, by hand, once.
            target.write_bytes(raw)
            target.chmod(path.stat().st_mode)
            hand.append(target_name)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            target.write_bytes(raw)
            copied.append(target_name + " (binary, verbatim)")
            continue
        target.write_text(rename(text), encoding="utf-8", newline="")
        target.chmod(path.stat().st_mode)
        copied.append(target_name)

    print(f"renamed copies : {len(copied)}")
    print(f"hand-edit set  : {hand}  (VERBATIM -- edit these by hand, once each)")
    print(f"left alone     : {skipped}")

    problems: list[str] = []
    sar = (DST / "seal_and_review.sh").read_text(encoding="utf-8")
    if "--campaign-wave 29" not in sar:
        problems.append("seal_and_review.sh does not carry --campaign-wave 29")
    if "--campaign-wave 28" in sar:
        problems.append("seal_and_review.sh still carries --campaign-wave 28")
    if "wave28" in sar:
        problems.append("seal_and_review.sh still points at wave28")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    if "--parent-useful-or-partial 12" not in sv:
        problems.append("seal_verdict.sh lost --parent-useful-or-partial 12")
    if "--lanes-total 16" not in sv:
        problems.append("seal_verdict.sh lost --lanes-total 16")
    rho = (DST / "rhoq.sh").read_text(encoding="utf-8")
    for line in rho.splitlines():
        if "--triple" in line and "$W" in line:
            problems.append(f"rhoq.sh triple resolves under $W: {line.strip()[:80]}")
    for name in ("mx29_verify_seals.py", "mx29_retain.py"):
        body = (DST / name).read_text(encoding="utf-8")
        if "wave28" not in body:
            problems.append(f"{name}: the hand-edit file lost its wave-28 text -- it was renamed")

    if problems:
        for p in problems:
            print("ASSERTION FAILED:", p, file=sys.stderr)
        return 1
    print("assertions: campaign wave, verdict flags, preread paths, hand-edit set -- all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
