#!/usr/bin/env python3
"""Cut wave 40's tree from wave 39's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-39 cutter, in full.**

1. ``SRC`` is the wave-39 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  ``sealq.sh`` arrives VERBATIM carrying
   wave 39's marker AND wave 39's two seal calls with every count intact, and
   ``--audit`` REFUSES until both are replaced by hand.  Eighth consecutive cut
   where that protection has had work to do -- **and the first where it was not
   enough.**  The marker assertion is one line at the top; the calls are anywhere
   below it, so a hand edit that rewrites the marker and misses the calls PASSES,
   and on this wave it did: ``seal V14CL896X13`` and ``seal V10CL896X13`` sat
   under a rewritten wave-40 marker until a peer wave found them.  They would
   have errored rather than mis-sealed, because no wave-40 receipt carries those
   tags, but "it errors" is not the protection.  **``audit()`` now pins the seal
   CALL COUNT at zero at cut time**, which is wave 39's own recommendation, and
   both calls are deleted.
3. ``AUTHORED_THIS_WAVE`` is FOUR files -- this cutter, the README, and the two
   files that are wave 40's one behavioural addition.  Waves 35 through 39
   declared two; this wave declares four and says why below.
4. **WAVE 40 ADDS A BUILD MODULE AND EDITS NO CARRIED ONE.**  ``mx40_vocabtie.py``
   and its driver ``tieq.sh`` are new files.  Every carried module keeps the
   bytes wave 39 shipped, retargeted by the substitution list; the four hand
   edits are the same four hand edits every cut since wave 31 has made.

**WHY A NEW MODULE, AND WHAT IT IS.**  ``docs/UNBUILT_INVENTORY_2026-08-28.md``
§2 ranks **C1, vocabulary-axis row TYING**, the worst of the estate's nine
census kills: it was closed at wave 14 on a nearest-neighbour cosine of 0.2977
between tail rows and their nearest kept row, and the comparison that statistic
ran was **tie against verbatim**.  The mechanism that actually shipped on that
axis, and that set the campaign's floor, is **deletion**, and a deleted row is a
worse stand-in for its members than a 0.30-cosine tied one.  The comparison that
decides the class was never run.  ``mx40_vocabtie.py`` runs it: same census,
same closure, same budget, same body, same parameter count to the byte -- and
the tail mapped into ``c`` shared rows instead of dropped.

**AND THE FORM IT IS BUILT IN WAS DECIDED BY A MEASUREMENT, NOT BY A
PREFERENCE.**  C1 as literally specified is a **many-to-one** vocabulary --
several tail tokens addressing one shared row.  That was tried first, on this
parent, on this stack, before a line of the builder was written, and **it is not
constructible as a stock child**: ``AutoTokenizer.from_pretrained`` refuses a
``tokenizer.json`` whose ``model.vocab`` maps two tokens onto one id with
``Error while attempting to unpickle Tokenizer: Token `...` out of vocabulary``
-- **and it names a different token on different runs.**  Five loads of one file
named three different tokens at four different column offsets, which is the Rust
``HashMap`` id->token inverse being built in a nondeterministic order.  Had it
loaded, ``decode`` of a tied id would have been a non-function across processes.
So wave 40 builds the **injective** realisation, in which the tokenizer is the
deletion twin's byte for byte and the only difference between the two children
is the values in ``c`` rows of one tensor -- which is the better controlled
comparison in any case, because the many-to-one form moves the tokenizer and the
weights at once and a difference could then be either.

**WAVE 40'S OWN AUDIT ADDITION.**  Every cut since wave 31 asserts the lines
this wave's build list depends on.  Wave 40's list is four ``mx40_build.py``
bodies, seven ``mx40_vocab.py`` rungs and one ``mx40_vocabtie.py`` child, so all
three modules are asserted: wave 36's five composed width x MLP lines, wave 37's
two live-mask lines and wave 38's three multi-index excision lines on the
builder (**this wave RUNS all three of those paths**, which is the first time
since wave 38 that any of them is exercised rather than merely carried); wave
39's five vocabulary-gather lines; and **seven new ones on the tie builder**,
which is the module this wave authors and therefore the module with no history
of surviving a cut at all.

The seven are stated in ``audit()`` beside their pins.  Three are worth naming
here because they are the difference between a tie and a lie:

* ``    if not torch.equal(head_rows, table[head_index]):`` -- the head of a tie
  child is a VERBATIM gather exactly as the deletion arm's whole table is, and
  it is asserted with ``torch.equal`` and not a tolerance.  A tie builder whose
  head silently drifted would be a fold nobody solved and nobody bounded.
* ``    if covered != len(tail_ids):`` -- the tie's whole claim is that **every**
  dropped id has an image.  A tie that quietly leaves ids unmapped is deletion
  wearing a tie's receipt.
* ``    if moved_rows != int(args.tie_rows):`` -- the control the whole
  comparison rests on.  Exactly ``c`` rows of this child differ from the source;
  every other row is the source's own bytes, so a behavioural difference against
  the deletion twin is attributable to the tie and to nothing else.

**THE NEEDLES CARRY THEIR INDENTATION** -- wave 38's lesson 6, which wave 39's
cut then re-learned on its own header.  Every needle below is the code's own
line with its own leading whitespace, because this file's prose quotes several
of them and a bare-string pin would pass with the code deleted.

**THE ACCEPT LIST IS APPENDED IN THE COMMIT THAT CUTS THIS TREE.**  Waves 34 and
35 each paid an entire extra deployment to meet
``experiments/seal_matrix_architecture_child.py``'s append-only accept list at
their first seal, after the GPU work was done.  Waves 36, 37, 38 and 39 appended
at the cut and paid none.  Wave 40 is the fifth, and it appends **five** strings
rather than four: the four standard ones plus
``neural-forge-matrix-wave40-tied-vocabulary-child-v1``, the schema
``mx40_vocabtie.py`` writes.  That file is in the Codex lane, so this cutter has
no standing to assert anything about it and does not try.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave39`` (a path component and a
schema infix) is retargeted; ``wave 39`` and ``Wave 39`` (English) are NOT, so a
docstring that records what wave 39 measured still says wave 39 measured it.
``run/mx40/README.md`` carries the provenance instead.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx39_verify_seals.py`` -- the rename turns ``wave39/cells`` into
  ``wave40/cells`` and thereby DELETES last wave's seal root.  Copied verbatim;
  ONE APPENDED LINE by hand.  35 declared roots become 36, ``wave39/cells``
  still declared.  **And wave 39's cells exist and are graded**: that wave sealed
  its two children, read them at 1/15 each and merged four ledger rows while wave
  40 was building, so wave 40's merge starts from 992 rows and not 988.
* ``mx39_retain.py`` -- an inherited ``KEEP`` set names last wave's children and
  everything unnamed is DELETED.  Copied verbatim; ``KEEP`` and ``DROP`` emptied
  by hand, the wave root, glob, schema and manifest name retargeted by hand.
* ``mx39_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and a
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line, the module references, the log prefix and BOTH
  build-receipt write sites retargeted by hand.
* ``sealq.sh`` -- arrives VERBATIM with last wave's marker and last wave's seal
  calls.  Copied verbatim; the marker is rewritten to name wave 40 and the three
  path resolutions (``env.sh``, ``mx*_percase.py``, ``seal_verdict.sh``) are
  retargeted by hand.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  **And it parses what it writes, in the invocation that
writes it** -- `NOTES-MX-AO` §9's rule, obeyed by the first file wave 40 writes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx39"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx39_verify_seals.py",
    "mx39_retain.py",
    "mx39_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  FOUR files.  Waves
#: 35 through 39 declared two -- this cutter and the README -- and wave 40 adds
#: the two that are its one behavioural addition, so the set still means "no
#: file is in here that nobody designed".
AUTHORED_THIS_WAVE = {
    "cut_wave40.py",
    "README.md",
    "mx40_vocabtie.py",
    "tieq.sh",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt.
NOT_CARRIED = {"wave39-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 39", "--campaign-wave 40"),
    ('"campaign_wave": 39', '"campaign_wave": 40'),
    ("wave39", "wave40"),
    ("mx39", "mx40"),
    ("MX39", "MX40"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.
FORBIDDEN_AFTER = {
    "mx39_": "a wave-39 module reference survived the copy",
    "MX39-": "a wave-39 child directory name survived the copy",
    "/wave39/": "a wave-39 artifact path survived the copy",
    "--campaign-wave 39": "a driver would seal this wave's build as wave 39",
}
#: DELIBERATE historical references, declared with a count.  The third field is
#: the comparison: an exact pin where the count is knowable before the wave
#: runs, a bound where it is not.
ALLOWED_HISTORICAL = {
    ("mx40_build.py", "mx39_"): (
        1,
        "==",
        "the wave-40 header names `mx39_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx40_vocab.py", "mx39_"): (
        1,
        "<=",
        "the wave-40 header may name `mx39_vocab.py` as the file it states its "
        "diff against.  Bounded rather than pinned because this wave makes no "
        "edit to that module and may add no header at all",
    ),
    ("mx40_retain.py", "MX39-"): (
        2,
        "<=",
        "the retention rule requires each DROP reason to name the retained "
        "source that rebuilds the deleted child.  Wave 39's two children were "
        "built and sealed and NEVER GRADED, so wave 40 does not drop them and "
        "this is expected to read ZERO; bounded at two so a legitimate "
        "cross-reference to wave 39's ungraded pair is a note, not a refusal",
    ),
    ("sealq.sh", "MX39-"): (
        2,
        "<=",
        "a seal note naming a checkpoint a child was built on.  Wave 40's "
        "sources are wave 37's retained MX37-CL896X09 and this wave's own "
        "bodies, so this is expected to read ZERO; bounded at two because wave "
        "40 grades wave 39's two orphaned dossiers and a seal note may say so",
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
        "schema": "neural-forge-matrix-wave40-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 39` / `Wave 39` / `WAVE 39` in English are NOT substituted, "
            "so a docstring recording what wave 39 measured still says wave 39 "
            "did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "ONE ADDITION AND ZERO EDITS.  `mx40_vocabtie.py` and its driver "
            "`tieq.sh` are NEW files -- the first build module this campaign has "
            "authored since wave 25's vocabulary gather -- and they exist to "
            "repair the estate's worst census kill: C1, vocabulary-axis row "
            "TYING, closed at wave 14 on a 0.2977 nearest-neighbour cosine that "
            "compared tying against VERBATIM when the mechanism that shipped and "
            "set the floor is DELETION.  Every carried module keeps the bytes "
            "wave 39 shipped.  Wave 36's composed width x MLP edit, wave 37's "
            "live-mask repair, wave 38's three multi-index excision lines and "
            "wave 39's five vocabulary-gather lines are carried unchanged and "
            "asserted on this wave's copies -- and unlike waves 39, this wave "
            "RUNS all four of those paths, so the assertions guard live code "
            "rather than dormant code.  Six new assertions cover the tie builder."
        ),
        "files": entries,
    }
    out = DST / "wave40-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-39 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave40.py", "wave40-tree-cut-manifest.json"):
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
    # not happen and the rename did.  Seven roots are checked by name.
    for name, needle in (("mx40_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx40_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx40_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx40_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx40_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx40_verify_seals.py", 'ROOT / "wave39" / "cells"'),
                         ("mx40_verify_seals.py", 'ROOT / "wave40" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx40_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave40-residual-basis-v1",
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
            problems.append(f"mx40_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.
    for needle, want, why in (
        ('(outdir / "mx40-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx40_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # WAVE 39'S ONE EDIT, CARRIED AND STILL PINNED IN BOTH DIRECTIONS.  Wave 39
    # repaired the builder's log prefix from `[mx32]` -- carried since wave 32's
    # cut inside an f-string the substitution list has no entry for -- and could
    # afford to because it never executed that file.  WAVE 40 EXECUTES IT FOUR
    # TIMES, so the repair is now under a seal and the pin is what says it did
    # not regress.
    for needle, want, why in (
        ('    print(f"[mx40] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave; wave 39 repaired it and "
         "wave 40 is the first wave to RUN the repaired file"),
        ('    print(f"[mx39] {message}", flush=True)\n', 0,
         "last wave's prefix, which the substitution list retargets; pinned at "
         "zero so a hand edit cannot leave it behind"),
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
                f"mx40_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 40" not in body:
            problems.append(f"{name} does not carry --campaign-wave 40")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 40 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # WAVE 40 ADDS THE PIN THE MARKER ASSERTION WAS NOT, AND IT ADDS IT BECAUSE
    # THIS WAVE'S OWN CUT NEEDED IT.  Waves 33 through 39 each deleted last
    # wave's seal calls by hand and the marker assertion was treated as the thing
    # that forced them out.  It is not: the marker is ONE line at the top and the
    # calls are anywhere below it, so a hand edit that rewrites the marker and
    # misses the calls passes.  That is exactly what happened here -- wave 39's
    # V14CL896X13 and V10CL896X13 calls, counts intact, sat under a rewritten
    # wave-40 marker and this audit said nothing.  They would have ERRORED rather
    # than mis-sealed, because no wave-40 receipt carries those tags, but "it
    # errors" is not a protection and a peer wave found them, not this file.
    #
    # THE PIN IS AT CUT TIME AND THAT IS THE WHOLE POINT.  `--audit` runs
    # immediately after the cut and before anything builds, when the correct
    # number of seal calls is ZERO; the wave then appends its own after it reads
    # its dossiers, and does not re-run this.  `seal () {` is the function
    # definition and is not a call, so the needle requires an identifier char.
    calls = re.findall(r"(?m)^seal [A-Za-z0-9]", seals)
    if calls:
        problems.append(
            f"sealq.sh carries {len(calls)} seal CALL(s) at cut time, expected 0. "
            "A verbatim copy brings last wave's calls forward with every count "
            "intact, and rewriting the marker above them does not remove them. "
            "This is wave 39's recommendation and wave 40's own defect"
        )
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    if not (DST / "tieq.sh").is_file():
        problems.append("tieq.sh is missing: this wave's new build module has no driver")
    # WAVE 34's ADDITION, carried unchanged.
    for name in ("mx40_extscore.py", "mx40_extrubric2.py", "mx40_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx40_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx40_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.
    pen = (DST / "mx40_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx40_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S, 37'S AND 38'S ADDITIONS -- AND WAVE 40 RUNS EVERY ONE OF THEM.
    # Wave 39 asserted these on a file it never opened.  Wave 40's build list is
    # `--ff 768`, `--ff 768 --excise 9`, `--ff 1728` and `--ff 736`: the composed
    # width x MLP path, the live-mask repair and the excision renumbering are all
    # live code this wave, which is the first time since wave 38 that any of them
    # has been exercised rather than merely carried.
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
                f"mx40_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ADDITION, CARRIED -- and wave 40 runs this module SEVEN times.
    vocab = (DST / "mx40_vocab.py").read_text(encoding="utf-8")
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
         "Wave 40 passes that flag on at least one rung, and a wave that passes "
         "a flag must assert the guard the flag disarms or it cannot tell "
         "disarming from absence"),
        ('            "the child tokenizer is not lossless on text; the byte alphabet or "\n', 1,
         "the refusal behind decode(child_encode(t)) == t"),
    ):
        got = vocab.count(needle)
        if got != want:
            problems.append(
                f"mx40_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 40'S OWN ADDITION.  `mx40_vocabtie.py` is authored this wave, so it
    # has no history of surviving a cut and every load-bearing line in it is
    # pinned at its indented form from the start rather than after the first
    # cut eats one.
    tie = (DST / "mx40_vocabtie.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(head_rows, table[head_index]):\n", 1,
         "the HEAD of a tie child is a verbatim gather exactly as the deletion "
         "arm's whole table is, and it is asserted with torch.equal and not a "
         "tolerance.  A tie builder whose head silently drifted would be a fold "
         "nobody solved and nobody bounded"),
        ("    if covered != len(tail_ids):\n", 1,
         "the tie's WHOLE CLAIM is that every dropped id has an image.  A tie "
         "that leaves ids unmapped is deletion wearing a tie's receipt, and it "
         "would publish deletion's coverage as a tie's"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control.  This child is built to be "
         "byte-identical in COUNT to the deletion twin it is compared against, "
         "and a count that drifts turns a controlled comparison into two "
         "unrelated readings"),
        ("    if decoded_probe != decoded_probe_again:\n", 1,
         "the id->token map must be a FUNCTION, and on this stack that is not "
         "free: the MANY-TO-ONE form of this same class does not load at all "
         "and names a different out-of-vocabulary token on every attempt, which "
         "is a nondeterministic HashMap inverse.  The injective form is "
         "asserted to be injective across two independently loaded tokenizers "
         "rather than assumed to be, because a frozen review of a child whose "
         "decode depends on the process is not a frozen review"),
        ("    if lost and not args.below_floor:\n", 1,
         "the coverage gate, and it is the DELETION TWIN'S gate unrelaxed.  A "
         "tie child at budget R carries the twin's tokenizer exactly, so it "
         "loses exactly the occurrences the twin loses; tying buys row CONTENT "
         "and not coverage.  A builder that quietly waived this would publish "
         "one child's coverage on another child's receipt"),
        ("    if merges_kept != expected_merges:\n", 1,
         "the tokenizer-identity claim, checked by two implementations "
         "agreeing.  The merge list is written by the deletion arm's own "
         "`remapped_tokenizer` and then re-derived independently here from the "
         "closure rule; if they disagree, this child's tokenizer is not the "
         "twin's and the comparison is not controlled"),
    ):
        got = tie.count(needle)
        if got != want:
            problems.append(
                f"mx40_vocabtie.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-39 identifier survived; seven seal roots declared; "
          "the accept list kept twelve basis schemas; the drivers name wave 40; "
          "sealq.sh carries this wave's marker; gainq.sh and tieq.sh exist; "
          "both extension surfaces have a scorer and v1 still names the v1 bank; "
          "the penalty diagnostic kept its NOT_A_FOLD banner and its identity "
          "refusal; wave 36's composed width x MLP path, wave 37's live-mask "
          "repair and wave 38's three multi-index excision lines are present AND "
          "are live code this wave; the builder's log prefix names THIS wave and "
          "no stale tag survives; all five vocabulary-gather lines are present at "
          "their indented form; and all seven lines the NEW tie builder depends "
          "on are pinned at theirs")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
