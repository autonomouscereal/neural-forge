#!/usr/bin/env python3
"""Cut wave 43's tree from wave 42's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-42 cutter, in full.**

1. ``SRC`` is the wave-42 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  Wave 40's seal-call pin -- ``audit()``
   counts ``^seal <TAG>`` and requires **zero at cut time** -- is carried, and
   wave 43 is the THIRD cut where it had work to do from the start: this file's
   copy of ``sealq.sh`` arrived carrying wave 42's **three** seal calls with
   every count intact.
3. ``AUTHORED_THIS_WAVE`` is SIX files -- this cutter, the README, and the four
   files that are wave 43's two behavioural additions, ``mx43_plane.py`` /
   ``planeq.sh`` and ``mx43_xchange.py`` / ``xchangeq.sh``.
4. **WAVE 43 ADDS TWO SOLVER MODULES AND EDITS NO CARRIED ONE.**  Waves 41 and
   42 each added one module; wave 43 adds two, because its deliverable IS the
   two solvers that waves 41 and 42 both deferred for authoring time.  Every
   carried file keeps the bytes wave 42 shipped, modulo the five substitutions
   and the four hand edits.

**WHY THESE TWO MODULES, AND WHAT THEY ARE.**  `NOTES-MX-AS` §13 items 5 and 6
and `NOTES-MX-AT` §9 items 1 and 2 recorded PLANE and XCHANGE as deferred for
the SAME reason twice running -- authoring time for a new solver, not any
finding about either -- and `NOTES-MX-AT` §11.7 said the choice was to budget a
wave whose deliverable is a solver or take them off the slate.  This is that
wave.

* ``mx43_plane.py`` is `docs/ROUND3_SLATE_2026-08-28.md` §3.4.  It has TWO
  modes and the census mode runs FIRST, by the slate's own instruction: the
  plane-energy census over the d=48 and d=32 plane sets **publishes either way
  and never gates**, and its retention numbers are PRE-REGISTERED before the
  build so a death attributes to the mechanism rather than to the rung.  The
  build mode folds ``head_dim`` to a band of whole RoPE planes with verbatim
  q/k rows, a verbatim value-channel subset, and two solved objects: the q/k
  layernorm diagonals and ``out_proj``.
* ``mx43_xchange.py`` is §3.3's **go-gate census** and only that.  It builds
  nothing.  It measures the three quantities `docs/ATTENTION_SHORTCONV_
  TRANSLATION_DESIGN_2026-08-01.md` §B names as the preconditions for the
  substitution being expressible at all, against thresholds this wave registers
  BEFORE it runs them, and **its registration says a failure closes the class**.
  A census may close a class only when its registration says so; this one does,
  and it says so in its own docstring as well as in the wave's pre-registration.

**THE NEEDLES CARRY THEIR INDENTATION** -- wave 38's lesson 6, carried, and
applied to both new modules from the start rather than after the first cut eats
one.

**WAVE 43 IS THE FIRST CUT TO EXERCISE WAVE 42'S GENERAL CALIBRATION PIN.**
`NOTES-MX-AT` §11.4 predicted it would report 78 residue lines against
``run/mx42/sealq.sh`` as shipped, and said the general pin is the one that
matters.  It is exercised here, it fired, and the hand edit deleted the block it
named.

**WHAT THIS CUT DOES NOT DO.**  It does not author a NORM-ESTATE Arm-M/Arm-S
solver, and it does not unblock ALIAS or ROWSET Arms A and B -- those wait on a
ruling and on a corpus respectively, which are not authoring problems.  Naming
them is cheaper than a tree that carries a driver for a module that does not
exist.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  **And it parses what it writes, in the invocation that
writes it** -- `NOTES-MX-AO` §9's rule, obeyed by the first file wave 43 writes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx42"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx42_verify_seals.py",
    "mx42_retain.py",
    "mx42_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  SIX files -- two
#: more than waves 40, 41 and 42 declared, and the two extra are the second new
#: module and its driver.  Wave 43's deliverable is the solvers.
AUTHORED_THIS_WAVE = {
    "cut_wave43.py",
    "README.md",
    "mx43_plane.py",
    "planeq.sh",
    "mx43_xchange.py",
    "xchangeq.sh",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt.
NOT_CARRIED = {"wave42-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 42", "--campaign-wave 43"),
    ('"campaign_wave": 42', '"campaign_wave": 43'),
    ("wave42", "wave43"),
    ("mx42", "mx43"),
    ("MX42", "MX43"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.
FORBIDDEN_AFTER = {
    "mx42_": "a wave-42 module reference survived the copy",
    "MX42-": "a wave-42 child directory name survived the copy",
    "/wave42/": "a wave-42 artifact path survived the copy",
    "--campaign-wave 42": "a driver would seal this wave's build as wave 42",
}
#: DELIBERATE historical references, declared with a count.  The third field is
#: the comparison: an exact pin where the count is knowable before the wave
#: runs, a bound where it is not.
ALLOWED_HISTORICAL = {
    ("mx43_build.py", "mx42_"): (
        1,
        "==",
        "the wave-43 header names `mx42_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx43_vocab.py", "mx42_"): (
        1,
        "<=",
        "the wave-43 header may name `mx42_vocab.py` as the file it states its "
        "diff against.  Bounded rather than pinned because this wave makes no "
        "edit to that module and may add no header at all",
    ),
    ("mx43_retain.py", "MX42-"): (
        3,
        "<=",
        "the retention rule requires each DROP reason to name the retained "
        "source that rebuilds the deleted child.  Wave 42 retained two children "
        "-- MX42-V6CL768 and MX42-V7CL896X13 -- and wave 43 builds BOTH of its "
        "vocabulary rungs against bodies those two children's own sources "
        "provide, so a legitimate cross-reference is a note and not a refusal",
    ),
    ("sealq.sh", "MX42-"): (
        4,
        "<=",
        "a seal note naming a checkpoint a child is read against.  Wave 43's "
        "vocabulary sources are wave 40's retained MX40-CL768 and wave 35's "
        "MX35-CL896X13, and its nearest sealed controls are wave 42's retained "
        "MX42-V6CL768 and MX42-V7CL896X13, which a seal note legitimately "
        "carries",
    ),
    ("mx43_plane.py", "MX42-"): (
        0,
        "==",
        "the plane solver folds the PARENT and names no wave-42 child at all; "
        "pinned at zero rather than left undeclared so that a later wave that "
        "adds a control cannot do it silently",
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
        "schema": "neural-forge-matrix-wave43-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 42` / `Wave 42` / `WAVE 42` in English are NOT substituted, "
            "so a docstring recording what wave 42 measured still says wave 42 "
            "did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "TWO ADDITIONS AND NO EDIT.  `mx43_plane.py` / `planeq.sh` and "
            "`mx43_xchange.py` / `xchangeq.sh` are NEW files.  The plane module "
            "has a CENSUS mode that publishes the per-plane share of the score "
            "operator's energy for every (head_dim, stride, band-slide) set the "
            "RoPE ladder can express and GATES NOTHING, and a BUILD mode that "
            "folds head_dim to a band of whole RoPE planes with verbatim q/k "
            "rows and a verbatim value-channel subset.  The xchange module is a "
            "GO-GATE CENSUS ONLY -- it builds nothing, it measures the three "
            "preconditions the 2026-08-01 translation design names for the "
            "conv-to-attention direction, and its pre-registration states that "
            "a failure CLOSES the class.  Every other carried module keeps the "
            "bytes wave 42 shipped.  Wave 39's five vocabulary-gather lines are "
            "live code this wave -- the build list is TWO `mx43_vocab.py` "
            "gathers -- and wave 36's composed width x MLP path, wave 37's "
            "live-mask repair, wave 38's three multi-index excision lines, wave "
            "40's five tie-builder lines, wave 41's six rowset lines and wave "
            "42's six dual-band lines are carried and asserted but NOT run."
        ),
        "files": entries,
    }
    out = DST / "wave43-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-42 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave43.py", "wave43-tree-cut-manifest.json"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for needle, why in FORBIDDEN_AFTER.items():
            hits = text.count(needle)
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
            if hits:
                problems.append(f"{path.name}: {why} ({needle!r} x{hits})")
    # The seal verifier must STILL carry every earlier wave, or the hand edit did
    # not happen and the rename did.  TEN roots are checked by name.
    for name, needle in (("mx43_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave39" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave40" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave41" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave42" / "cells"'),
                         ("mx43_verify_seals.py", 'ROOT / "wave43" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx43_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave43-residual-basis-v1",
                   "neural-forge-matrix-wave42-residual-basis-v1",
                   "neural-forge-matrix-wave41-residual-basis-v1",
                   "neural-forge-matrix-wave40-residual-basis-v1",
                   "neural-forge-matrix-wave39-residual-basis-v1",
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
            problems.append(f"mx43_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.
    for needle, want, why in (
        ('(outdir / "mx43-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
        ('(outdir / "mx42-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx41-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx40-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
        ('(outdir / "mx39-build-receipt.json").write_text', 0,
         "a stale build-receipt filename survived the hand edit"),
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
                f"mx43_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # WAVE 39'S ONE EDIT, CARRIED AND STILL PINNED IN BOTH DIRECTIONS.  Wave 43
    # does not run this file either; the pin is kept for the reason waves 41 and
    # 42 kept it, which is that the module is in the tree and a later wave runs
    # whatever this cut leaves behind.
    for needle, want, why in (
        ('    print(f"[mx43] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave even in a wave that does "
         "not run it, or the next wave that does inherits a stale tag"),
        ('    print(f"[mx42] {message}", flush=True)\n', 0,
         "last wave's prefix, which the substitution list retargets; pinned at "
         "zero so a hand edit cannot leave it behind"),
        ('    print(f"[mx41] {message}", flush=True)\n', 0,
         "the wave-41 prefix, pinned at zero"),
        ('    print(f"[mx40] {message}", flush=True)\n', 0,
         "the wave-40 prefix, pinned at zero"),
        ('    print(f"[mx32] {message}", flush=True)\n', 0,
         "the wave-32 prefix that shipped for seven waves; pinned at zero "
         "permanently"),
        ('    print(f"[mx16] {message}", flush=True)\n', 0,
         "wave 17's form of the same defect, pinned at zero so the repair cannot "
         "regress to an even older tag"),
    ):
        got = build.count(needle)
        if got != want:
            problems.append(
                f"mx43_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 43" not in body:
            problems.append(f"{name} does not carry --campaign-wave 43")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 43 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # WAVE 40'S PIN, CARRIED.  `--audit` runs immediately after the cut and
    # before anything builds, when the correct number of seal calls is ZERO; the
    # wave appends its own after it reads its dossiers and does not re-run this.
    # `seal () {` is the function definition and is not a call, so the needle
    # requires an identifier char.  WAVE 42 CARRIED THREE CALLS HERE.
    calls = re.findall(r"(?m)^seal [A-Za-z0-9]", seals)
    if calls:
        problems.append(
            f"sealq.sh carries {len(calls)} seal CALL(s) at cut time, expected 0. "
            "A verbatim copy brings last wave's calls forward with every count "
            "intact, and rewriting the marker above them does not remove them"
        )
    # WAVE 42'S ADDITION, AND WAVE 43 IS THE FIRST CUT TO EXERCISE IT.  The three
    # literal needles are the NAMED REGRESSION -- wave 39's exact sentence, its
    # exact dossier and its exact counter, which survived two cuts -- and they
    # are kept because a regression pin that stops being checked is a pin that
    # stops working.  They are NOT the protection: they are literal, and wave
    # 42's own calibration paragraph opens with different words, so none of them
    # would have fired on it.
    for needle, why in (
        ("# AND IT WAS RUN AND IT IS EXACT.",
         "an inherited CALIBRATION RESULT survived the cut.  This wave has not "
         "run its calibration at cut time, so a sentence asserting that it did "
         "is last wave's -- or, as it turned out in waves 40 and 41, wave 39's"),
        ("ARCH.V14CL896X13's dossier",
         "wave 39's named calibration dossier survived the cut.  The wave writes "
         "the dossier it actually read"),
        ("TWELFTH consecutive calibration",
         "an inherited calibration COUNTER survived the cut; wave 41's own "
         "NOTES record the fourteenth while this file still said the twelfth"),
    ):
        hits = seals.count(needle)
        if hits:
            problems.append(
                f"sealq.sh carries {needle!r} x{hits} at cut time, expected 0 -- {why}"
            )
    # THE GENERAL PIN, and wave 43 is the FIRST cut to exercise it.  `NOTES-MX-AT`
    # section 11.4 predicted it would report 78 residue lines against
    # `run/mx42/sealq.sh` as shipped.  The marker block ends with a line that
    # says the wave's own calibration is written BELOW it after the dossiers are
    # read, and at cut time the only thing that may follow that line is blank
    # lines and the closing echo.  Anything else is last wave's reading, whatever
    # it says and however it is worded.
    sentinel = ("# --- wave 43's own calibration, byte comparison and reading order "
                "are written\n# --- below this line AFTER the dossiers are read, and "
                "not before.\n")
    if sentinel not in seals:
        problems.append(
            "sealq.sh is missing the calibration sentinel.  Without it there is "
            "no line to measure 'what this wave has not written yet' against, "
            "and the three literal needles above are the only protection -- "
            "which is exactly the protection that failed for two waves"
        )
    else:
        tail = seals.split(sentinel, 1)[1]
        residue = [line for line in tail.splitlines()
                   if line.strip() and line.strip() != 'echo "SEALQ DONE"']
        if residue:
            problems.append(
                f"sealq.sh carries {len(residue)} line(s) of prose or code after "
                "the calibration sentinel at cut time, expected 0.  Everything "
                "below that line is a reading THIS wave has not taken yet, so at "
                "the cut it can only be last wave's.  First line: "
                f"{residue[0].strip()[:80]!r}"
            )
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    if not (DST / "tieq.sh").is_file():
        problems.append("tieq.sh is missing: wave 40's build module has no driver")
    if not (DST / "rowsetq.sh").is_file():
        problems.append("rowsetq.sh is missing: wave 41's build module has no driver")
    if not (DST / "dualbandq.sh").is_file():
        problems.append("dualbandq.sh is missing: wave 42's module has no driver")
    if not (DST / "planeq.sh").is_file():
        problems.append("planeq.sh is missing: this wave's plane solver has no driver")
    if not (DST / "xchangeq.sh").is_file():
        problems.append("xchangeq.sh is missing: this wave's go-gate census has no driver")
    # WAVE 34's ADDITION, carried unchanged.
    for name in ("mx43_extscore.py", "mx43_extrubric2.py", "mx43_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx43_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx43_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.
    pen = (DST / "mx43_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx43_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S, 37'S AND 38'S ADDITIONS.  Wave 43 builds NO body through this
    # module, so these pins guard dormant code.  They are kept for wave 41's
    # stated reason: the module is in the tree and a future wave will run it, and
    # dropping a pin because a wave does not exercise it is how wave 30 deleted a
    # seal root.
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
         "the parent's gated intermediate at the lifted row"),
        ("the intermediate-axis identity does not hold", 2,
         "wave 14 3's identity precondition must be checked on the CHILD's own "
         "gated intermediate too, not only on the parent's"),
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
                f"mx43_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ADDITION, CARRIED -- and wave 43 runs this module TWICE.  It is
    # the only CARRIED builder this wave runs.
    vocab = (DST / "mx43_vocab.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "the identity gate for a ZERO-SOLVE arm: every kept row is the source's "
         "own bytes, asserted with torch.equal and not a tolerance"),
        ("        if not torch.equal(gathered[probe], table[order[probe]]):\n", 1,
         "the four-row spot check that runs before the whole-table comparison"),
        ("    closure_state = merge_closure_intact(vocab, keptset)\n", 1,
         "the STRUCTURAL half of the census gate -- the half with no flag"),
        ("    if lost and not args.below_floor:\n", 1,
         "the COVERAGE half, and the guard the --below-floor flag disarms.  "
         "Wave 43 passes that flag on both rungs it builds, and a wave that "
         "passes a flag must assert the guard the flag disarms or it cannot tell "
         "disarming from absence"),
        ('            "the child tokenizer is not lossless on text; the byte alphabet or "\n', 1,
         "the refusal behind decode(child_encode(t)) == t"),
        ('    ranking = sorted(\n', 1,
         "`choose_kept`'s discretionary ranking.  Wave 41 closed the question of "
         "WHICH rows with three built arms; this wave moves HOW MANY again, on "
         "the same call, and the call has to be the same one"),
    ):
        got = vocab.count(needle)
        if got != want:
            problems.append(
                f"mx43_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVES 40, 41 AND 42'S ADDITIONS, CARRIED.  None of the three runs this
    # wave; all three are pinned, for the reason stated above.
    tie = (DST / "mx43_vocabtie.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(head_rows, table[head_index]):\n", 1,
         "the HEAD of a tie child is a verbatim gather; asserted with "
         "torch.equal and not a tolerance"),
        ("    if covered != len(tail_ids):\n", 1,
         "the tie's WHOLE CLAIM is that every dropped id has an image"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control"),
        ("    if decoded_probe != decoded_probe_again:\n", 1,
         "the id->token map must be a FUNCTION"),
        ("    if merges_kept != expected_merges:\n", 1,
         "the tokenizer-identity claim, checked by two implementations agreeing"),
    ):
        got = tie.count(needle)
        if got != want:
            problems.append(
                f"mx43_vocabtie.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    rowset = (DST / "mx43_vocabrowset.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "a rowset child is a VERBATIM gather exactly as the deletion arm is"),
        ("    if len(order) != len(sealed_order):\n", 1,
         "the arms are compared at FIXED K"),
        ("    if mandatory - keptset:\n", 1,
         "the mandatory set is held EXACTLY as sealed"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control"),
        ("    if closure_state[\"intact\"] is not True:\n", 1,
         "the STRUCTURAL half of the census gate is not releasable on a rowset "
         "arm any more than on a gather"),
        ("    if arm == \"C\":\n", 2,
         "the leverage arm is BUILT, not predicted; two load-bearing sites"),
    ):
        got = rowset.count(needle)
        if got != want:
            problems.append(
                f"mx43_vocabrowset.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    dual = (DST / "mx43_dualband.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("        if column_a != int(sealed_count):\n", 1,
         "THE REFUSAL THAT MAKES COLUMN A THE SEALED COLUMN"),
        ("        if lane_ask not in (\"PRESERVE\", \"SELECT\", \"PRODUCE\"):\n", 1,
         "every lane must carry a DECLARED ask class"),
        ("                    if classification == \"COPY_BACK\" and lane_ask != \"PRESERVE\":\n", 1,
         "THE EXCEPTION CHECK against wave 40's two sealed precedents"),
        ("    if len(LANE_ASK) != 16:\n", 1,
         "all sixteen lanes of the frozen surface must be classified"),
        ("            if not payload:\n", 1,
         "an unresolvable payload anchor must SAY so rather than score zero"),
        ("            if sealed_verdict in (\"Useful\", \"Partial\"):\n", 1,
         "only a HELD lane can be voided by the second column"),
    ):
        got = dual.count(needle)
        if got != want:
            problems.append(
                f"mx43_dualband.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 43'S OWN ADDITIONS.  Both modules are authored this wave, so neither
    # has a history of surviving a cut, and every load-bearing line in both is
    # pinned at its INDENTED form from the start rather than after the first cut
    # eats one.  Wave 38's lesson 6.
    plane = (DST / "mx43_plane.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(new_q.weight.data, wq_src[q_rows]):\n", 1,
         "THE Q ROWS ARE VERBATIM.  A band fold's entire legality argument is "
         "that the kept planes are the parent's own rows, asserted with "
         "torch.equal and not a tolerance"),
        ("    if not torch.equal(new_k.weight.data, wk_src[k_rows]):\n", 1,
         "the same assertion on the k side, which is a DIFFERENT row map because "
         "there are eight kv heads and sixteen query heads"),
        ("    if not torch.equal(new_v.weight.data, wv_src[v_rows]):\n", 1,
         "THE VALUE CHANNELS ARE A VERBATIM SUBSET TOO.  Wave 15's dead "
         "head_dim axis contracted v by an evidence-weighted LRA; this axis does "
         "not, and the difference has to be asserted rather than described"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control, exactly as on every vocabulary "
         "child: an arithmetic prediction the build either lands on or fails"),
        ("        if abs(theta_child / expected_theta - 1.0) > THETA_TOLERANCE:\n", 1,
         "THE LADDER IS THE FOLD.  If the child's rope_theta is not "
         "theta^(stride*d/64) then its plane j is not the parent's plane "
         "stride*j + slide and the kept rows are rows of a different operator.  "
         "The check runs over TWO sources -- the value this build computed and "
         "the value that landed in the written config -- against an independent "
         "re-derivation, because a build that only checks its own variable "
         "against itself is checking nothing"),
        ("            if float(scale) <= 1.0:\n", 1,
         "THE LAW is a veto on update scale and it is fail-closed here as "
         "everywhere else in the campaign"),
        ("    if census_only:\n", 1,
         "THE CENSUS RUNS WITHOUT THE BUILD.  The slate requires the plane "
         "energy to be published and PRE-REGISTERED before any child exists, so "
         "the census cannot be a by-product of a build that has already chosen"),
        ("    energy = torch.zeros(planes, dtype=torch.float64)\n", 1,
         "the census accumulator is fp64 and is wave 15's own measure, "
         "unchanged, so this wave's numbers are comparable to HD16's receipt"),
    ):
        got = plane.count(needle)
        if got != want:
            problems.append(
                f"mx43_plane.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    xchange = (DST / "mx43_xchange.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("GO_GATE = {\n", 1,
         "THE THRESHOLDS ARE ONE DECLARED TABLE.  A go-gate whose numbers are "
         "spread through the code is a go-gate that can be moved after the "
         "measurement without anybody noticing"),
        ("    if not torch.allclose(delta_check, delta, rtol=1e-4, atol=1e-4):\n", 1,
         "THE RECONSTRUCTION GATE.  Every number this census reports is a "
         "decomposition of the conv branch's own delta, and the decomposition is "
         "worthless unless the sum of its parts reproduces the delta the model "
         "actually computed"),
        ("        if verdict is False:\n", 1,
         "the census REPORTS A VERDICT.  It is the only census in the campaign "
         "that may close a class, and it may do that only because its "
         "pre-registration says so, so it has to state the verdict rather than "
         "leave it to a reader"),
        ("    \"this census MAY close the XCHANGE class\"", 1,
         "and it says so in its own artifact, not only in the wave's notes"),
        ("        rho_2 = float(eps.pow(2).sum() / delta.pow(2).sum())\n", 1,
         "rho_2 is the share of the conv delta that is PROVABLY outside the "
         "linear-value attention class -- the design document's own quantity, "
         "computed on the campaign's own evidence"),
    ):
        got = xchange.count(needle)
        if got != want:
            problems.append(
                f"mx43_xchange.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-42 identifier survived; TEN seal roots declared; "
          "the accept list kept fifteen basis schemas; the drivers name wave 43; "
          "sealq.sh carries this wave's marker, ZERO seal calls and ZERO lines "
          "after the calibration sentinel; gainq.sh, tieq.sh, rowsetq.sh, "
          "dualbandq.sh, planeq.sh and xchangeq.sh exist; both extension "
          "surfaces have a scorer and v1 still names the v1 bank; the penalty "
          "diagnostic kept its NOT_A_FOLD banner and its identity refusal; wave "
          "36's composed width x MLP path, wave 37's live-mask repair and wave "
          "38's three multi-index excision lines are present but DORMANT this "
          "wave; the builder's log prefix names THIS wave and no stale tag "
          "survives; all six vocabulary-gather lines are present at their "
          "indented form and ARE live code this wave; wave 40's five "
          "tie-builder lines, wave 41's six rowset lines and wave 42's six "
          "dual-band lines are carried; and all thirteen lines the TWO NEW "
          "solver modules depend on are pinned at their indented form")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
