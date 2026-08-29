#!/usr/bin/env python3
"""Cut wave 41's tree from wave 40's, by an EXPLICIT substitution list.

Wave 31 wrote this file (as `cut_wave31.py`) after wave 30's blanket rename
broke three things: it deleted ``wave29/cells`` from the seal verifier's
declared roots, it dropped wave 29's string from the append-only
``BASIS_SCHEMAS_ACCEPTED``, and it rewrote four docstrings that recorded what
wave 29 did into claims that wave 30 did it.  The design is unchanged and the
reasons are unchanged, so this file states its diff against its ancestor rather
than restating the argument.

**The diff against the wave-40 cutter, in full.**

1. ``SRC`` is the wave-40 tree, the substitution list moves one wave, and the
   forbidden list moves with it.
2. ``HAND_EDIT`` is the same four files.  Wave 40's own cut discovered that the
   ``sealq.sh`` marker assertion is not the protection it was treated as -- the
   marker is ONE line at the top and last wave's seal CALLS are anywhere below
   it -- and added ``audit()``'s pin of the seal-call count at **zero at cut
   time**.  That pin is carried here and it is the first thing this cut ran.
3. ``AUTHORED_THIS_WAVE`` is FOUR files -- this cutter, the README, and the two
   files that are wave 41's one behavioural addition, ``mx41_vocabrowset.py``
   and ``rowsetq.sh``.
4. **WAVE 41 ADDS A BUILD MODULE AND EDITS EXACTLY ONE CARRIED ONE.**  Wave 40
   added a module and edited none; wave 41 adds ``mx41_vocabrowset.py`` and
   makes ONE edit, to ``mx41_child_index.py``, which is wave 40's own §11.6
   recommendation and is pinned in both directions below.

**WHY A NEW MODULE, AND WHAT IT IS.**  ``docs/ROUND3_SLATE_2026-08-28.md`` §3.2
ranks **ROWSET** second of nine.  More than half the kept vocabulary at the
campaign's floor is chosen by a TIE-BREAK -- ``mx41_vocab.py``'s ranking key is
``(-census_count, parent_token_id)`` and that file's own docstring says "with a
67,584-position census and a 16,384-row budget, the tie-break decides most of
the table".  Sixteen waves have moved **how many** rows are kept and not one has
moved **which**.  ``mx41_vocabrowset.py`` moves which, at fixed K, on the same
body, with the same mandatory set, the same closure and a parameter count
identical to the deletion twin's to the byte.

**AND THE ARMS IT IS BUILT WITH WERE DECIDED BY A CORPUS AUDIT TAKEN BEFORE IT
WAS WRITTEN.**  The slate's Arms A and B rank the discretionary fill by a SECOND
census over held-out text "sealed by hash, disjoint from the frozen 15 AND the
30-prompt extension bank".  Every corpus this estate holds fails that test, and
each fails it for its own reason, checked in the file rather than assumed:

* ``lfm2_paraphrase_triplets_v1.json`` -- its own ``purpose`` field reads "three
  restatements of each sealed case's ask".  It is a paraphrase OF the frozen 15.
  Not disjoint; selecting on it is selecting on the review surface.
* ``lfm2_causal_support_future_manual_review_v1.json`` -- its own ``purpose``
  field reads "never support selection, fitting, training, or an automatic
  numeric gate".  Contractually barred, and 2,069 characters besides.
* ``lfm2_extension_bank_prompts_v1/v2.json`` -- the rider bank.  The slate voids
  any rider whose scoring text overlaps it.
* the 256-row sealed corpus and the 24 captured batches -- these ARE the
  evidence census that supplies the primary key.  Not held out from itself.
* wikitext / lambada -- the benchmark column.  Selecting a vocabulary on them
  contaminates the only external instrument the campaign publishes.

So wave 41 builds the three arms that need **no corpus at all**, which is a
change of ARM and not a waiver of the class, and it is registered as a
substitution with its cause before anything is built:

| arm | discretionary key | what it tests |
|---|---|---|
| ``P`` | ``(-census, -parent_id)`` | the sealed tie-break RUN BACKWARDS |
| ``R`` | ``(-census, blake2b(seed, id))`` | a seeded permutation, the null ordering |
| ``C`` | ``(-census, -leverage, parent_id)`` | the slate's Arm C, verbatim |

Arm P is the maximal perturbation of the one ordering the estate has never
varied and Arm R brackets it; if the sealed forward order, its exact reverse and
a seeded permutation all read the same count at the same K, "row identity is
inert at fixed K" is supported by three independent orderings rather than by one
thin second census.  **Arm C is built with the wave-9 prediction that it FAILS**
(``NOTES-MX-M`` §2.3: row-norm ordering reached 2.34% coverage at M=4,096),
predicted AND built, never skipped.

**WAVE 41'S OWN AUDIT ADDITIONS.**  Every cut since wave 31 asserts the lines
this wave's build list depends on.  Wave 41's list is one ``mx41_build.py`` body,
five ``mx41_vocab.py`` rungs, three ``mx41_vocabrowset.py`` arms and three
``mx41_termgain.py`` gain children, so all four modules are asserted.  Six new
needles cover the rowset builder.  Three are worth naming here:

* ``    if not torch.equal(gathered, table[index]):`` -- a rowset child is a
  VERBATIM gather exactly as the deletion arm is.  Only the SELECTION moves; if
  a row's bytes moved too, a behavioural difference could be either.
* ``    if len(order) != len(sealed_order):`` -- the arms are compared at FIXED
  K.  An arm that lands on a different row count is not a re-ranking, it is a
  different rung, and the comparison would be uncontrolled.
* ``    if mandatory - keptset:`` -- the mandatory set is held EXACTLY as sealed,
  so coverage cannot move and the below-floor gate cannot fire differently
  between arms.  An arm that quietly dropped a mandatory row would publish a
  selection result that was really a coverage result.

**THE NEEDLES CARRY THEIR INDENTATION** -- wave 38's lesson 6, carried.

**WAVE 41'S ONE EDIT TO A CARRIED MODULE, AND WHY IT IS SAFE TO MAKE.**
``NOTES-MX-AR`` §11.6: ``mx40_child_index.py`` prints its index to stdout and
takes no ``--out``, so the obvious invocation -- redirecting stdout into the
file the generator ALSO writes -- corrupts it.  Wave 40 reproduced wave 35's bug
that way and caught it only because ``NOTES-MX-AO`` §9's parse-what-you-write
rule made it parse the file in the same command.  This wave adds ``--out`` and
makes the file parse what it writes inside the generator itself, so the rule is
no longer the only thing standing between the estate and a corrupt index.  Both
the new flag and the re-parse are pinned below, and the old stdout print is
pinned as still present: the fix ADDS a path, it does not remove the one every
earlier wave used.

**THE ACCEPT LIST IS APPENDED IN THE COMMIT THAT CUTS THIS TREE.**  Waves 36
through 40 appended at the cut and paid no extra deployment; wave 41 is the
sixth.  It appends **six** strings -- the four standard ones, the tie schema
carried from wave 40 (that module is carried and not run this wave, and a list
completed only where it was needed is a list the next wave has to reason about),
and ``neural-forge-matrix-wave41-rowset-vocabulary-child-v1``, which is
``mx41_vocabrowset.py``'s schema.

Two rules, carried verbatim from wave 31, because they are the whole design.

**1. The substitution list is identifiers, paths and schema strings ONLY.**  It
contains no prose form of the wave number.  ``wave40`` (a path component and a
schema infix) is retargeted; ``wave 40`` and ``Wave 40`` (English) are NOT, so a
docstring that records what wave 40 measured still says wave 40 measured it.
``run/mx41/README.md`` carries the provenance instead.

**2. The hand-edit set is copied with NO substitution at all.**

* ``mx40_verify_seals.py`` -- the rename turns ``wave40/cells`` into
  ``wave41/cells`` and thereby DELETES last wave's seal root.  Copied verbatim;
  ONE APPENDED LINE by hand.  36 declared roots become 37, ``wave40/cells``
  still declared, and wave 40's eleven children live under it.
* ``mx40_retain.py`` -- an inherited ``KEEP`` set names last wave's children and
  everything unnamed is DELETED.  Copied verbatim; ``KEEP`` and ``DROP`` emptied
  by hand, the wave root, glob, schema and manifest name retargeted by hand.
* ``mx40_build.py`` -- ``BASIS_SCHEMAS_ACCEPTED`` is APPEND-ONLY and a
  substitution rewrites its first entry.  Copied verbatim; the schema strings,
  the appended accept-list line, the module references, the log prefix and BOTH
  build-receipt write sites retargeted by hand.
* ``sealq.sh`` -- arrives VERBATIM with last wave's marker and last wave's ELEVEN
  seal calls.  Copied verbatim; the marker is rewritten to name wave 41, the
  eleven calls are deleted, and the three path resolutions (``env.sh``,
  ``mx*_percase.py``, ``seal_verdict.sh``) are retargeted by hand.  The seal-call
  count is pinned at ZERO by ``audit()`` -- wave 40's addition, and the first cut
  where it had work to do from the start.

The manifest this writes is the explicit diff: per file, how many of each
substitution fired.  **And it parses what it writes, in the invocation that
writes it** -- `NOTES-MX-AO` §9's rule, obeyed by the first file wave 41 writes.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mx40"
DST = Path(__file__).resolve().parent

#: Copied VERBATIM.  Every edit to these is made by hand, once, and read back.
HAND_EDIT = {
    "mx40_verify_seals.py",
    "mx40_retain.py",
    "mx40_build.py",
    "sealq.sh",
}
#: Authored by this wave; the copier never overwrites them.  FOUR files, the
#: same count wave 40 declared and for the same reason -- this cutter, the
#: README, and the two files that are this wave's one behavioural addition.
AUTHORED_THIS_WAVE = {
    "cut_wave41.py",
    "README.md",
    "mx41_vocabrowset.py",
    "rowsetq.sh",
}
#: NOT carried forward at all.  Last wave's tree-cut manifest is last wave's
#: receipt.
NOT_CARRIED = {"wave40-tree-cut-manifest.json"}

#: IDENTIFIERS, PATHS AND SCHEMA STRINGS.  No prose.  Ordered longest-first so a
#: shorter key cannot eat a longer one's prefix.
SUBS = [
    ("--campaign-wave 40", "--campaign-wave 41"),
    ('"campaign_wave": 40', '"campaign_wave": 41'),
    ("wave40", "wave41"),
    ("mx40", "mx41"),
    ("MX40", "MX41"),
]

#: Substrings that must NOT appear in any renamed copy, and the reason.
FORBIDDEN_AFTER = {
    "mx40_": "a wave-40 module reference survived the copy",
    "MX40-": "a wave-40 child directory name survived the copy",
    "/wave40/": "a wave-40 artifact path survived the copy",
    "--campaign-wave 40": "a driver would seal this wave's build as wave 40",
}
#: DELIBERATE historical references, declared with a count.  The third field is
#: the comparison: an exact pin where the count is knowable before the wave
#: runs, a bound where it is not.
ALLOWED_HISTORICAL = {
    ("mx41_build.py", "mx40_"): (
        1,
        "==",
        "the wave-41 header names `mx40_build.py` as the file it states its diff "
        "against; it resolves no path",
    ),
    ("mx41_vocab.py", "mx40_"): (
        1,
        "<=",
        "the wave-41 header may name `mx40_vocab.py` as the file it states its "
        "diff against.  Bounded rather than pinned because this wave makes no "
        "edit to that module and may add no header at all",
    ),
    ("mx41_vocabrowset.py", "mx40_"): (
        4,
        "<=",
        "the rowset builder is authored this wave and it CITES the module it "
        "re-ranks -- `mx40_vocab.py`'s ranking key is the object under test, so "
        "the file names it.  Bounded rather than pinned because the citation is "
        "prose and may be made more than once; it resolves no path, because the "
        "shared machinery is imported from `mx41_vocab` by module name",
    ),
    ("mx41_retain.py", "MX40-"): (
        3,
        "<=",
        "the retention rule requires each DROP reason to name the retained "
        "source that rebuilds the deleted child.  Wave 40 retained three "
        "children -- MX40-CL1728, MX40-CL768 and MX40-V9CL896X13 -- and wave 41 "
        "builds ON the third of them, so a legitimate cross-reference is a note "
        "and not a refusal",
    ),
    ("sealq.sh", "MX40-"): (
        3,
        "<=",
        "a seal note naming a checkpoint a child was built on.  Wave 41's "
        "sources are wave 35's MX35-CL896X13, wave 37's MX37-CL896X09 and wave "
        "40's retained MX40-V9CL896X13, and the third of those is a wave-40 "
        "name that a seal note legitimately carries",
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
        "schema": "neural-forge-matrix-wave41-tree-cut-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tree": str(SRC),
        "destination_tree": str(DST),
        "substitutions_applied": [{"from": a, "to": b} for a, b in SUBS],
        "prose_left_alone": (
            "`wave 40` / `Wave 40` / `WAVE 40` in English are NOT substituted, "
            "so a docstring recording what wave 40 measured still says wave 40 "
            "did"
        ),
        "hand_edit_set": sorted(HAND_EDIT),
        "not_carried": sorted(NOT_CARRIED),
        "authored_this_wave": sorted(AUTHORED_THIS_WAVE),
        "behavioural_edits_this_wave": (
            "ONE ADDITION AND ONE EDIT.  `mx41_vocabrowset.py` and its driver "
            "`rowsetq.sh` are NEW files: sixteen waves have moved HOW MANY "
            "vocabulary rows are kept and none has moved WHICH, although "
            "`mx41_vocab.py`'s own docstring records that the tie-break decides "
            "most of the table.  The ONE edit is to `mx41_child_index.py`, which "
            "gains `--out` and parses what it writes -- `NOTES-MX-AR` §11.6, "
            "wave 40's own recommendation after it reproduced wave 35's "
            "stdout-redirect corruption on that exact file.  Every other carried "
            "module keeps the bytes wave 40 shipped.  Wave 36's composed width x "
            "MLP edit, wave 37's live-mask repair, wave 38's three multi-index "
            "excision lines, wave 39's five vocabulary-gather lines and wave 40's "
            "seven tie-builder lines are carried unchanged and asserted on this "
            "wave's copies; this wave RUNS the builder once (`--ff 768 --excise "
            "9`), the gather five times and the terminal-gain solver three times."
        ),
        "files": entries,
    }
    out = DST / "wave41-tree-cut-manifest.json"
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
    """Read every file in the destination back and refuse a surviving wave-40 name."""

    problems: list[str] = []
    for path in sorted(DST.iterdir()):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json"):
            continue
        if path.name in ("cut_wave41.py", "wave41-tree-cut-manifest.json"):
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
    # not happen and the rename did.  Eight roots are checked by name.
    for name, needle in (("mx41_verify_seals.py", 'ROOT / "wave34" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave35" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave36" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave37" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave38" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave39" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave40" / "cells"'),
                         ("mx41_verify_seals.py", 'ROOT / "wave41" / "cells"')):
        body = (DST / name).read_text(encoding="utf-8")
        if needle not in body:
            problems.append(f"{name}: missing {needle}")
    build = (DST / "mx41_build.py").read_text(encoding="utf-8")
    for needle in ("neural-forge-matrix-wave41-residual-basis-v1",
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
            problems.append(f"mx41_build.py: BASIS_SCHEMAS_ACCEPTED is missing {needle}")
    # WAVE 34'S ADDITION, CARRIED AND WIDENED BY ONE LINE EACH WAVE.
    for needle, want, why in (
        ('(outdir / "mx41-build-receipt.json").write_text', 2,
         "the builder must write THIS wave's build-receipt filename at both "
         "export sites, or every exact-name reader in the tree misses it"),
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
                f"mx41_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # WAVE 39'S ONE EDIT, CARRIED AND STILL PINNED IN BOTH DIRECTIONS.  Wave 39
    # repaired the builder's log prefix from `[mx32]`; wave 40 was the first wave
    # to RUN the repaired file and put it under four seals.  Wave 41 runs it once.
    for needle, want, why in (
        ('    print(f"[mx41] {message}", flush=True)\n', 1,
         "the builder's log prefix must name THIS wave; wave 39 repaired it, "
         "wave 40 ran it four times, and wave 41 runs it once more"),
        ('    print(f"[mx40] {message}", flush=True)\n', 0,
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
                f"mx41_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    for name in ("seal_and_review.sh",):
        body = (DST / name).read_text(encoding="utf-8")
        if "--campaign-wave 41" not in body:
            problems.append(f"{name} does not carry --campaign-wave 41")
    sv = (DST / "seal_verdict.sh").read_text(encoding="utf-8")
    for flag in ("--parent-useful-or-partial 12", "--lanes-total 16"):
        if flag not in sv:
            problems.append(f"seal_verdict.sh lost {flag}")
    seals = (DST / "sealq.sh").read_text(encoding="utf-8")
    marker = "# --- wave 41 seals are appended below, AFTER the dossiers are read ---"
    if marker not in seals:
        problems.append(
            "sealq.sh does not carry this wave's seal marker, so its seal calls "
            "are last wave's counts on this wave's cell ids"
        )
    # WAVE 40'S PIN, CARRIED, AND THE FIRST CUT WHERE IT HAD WORK TO DO FROM THE
    # START.  The marker assertion above is ONE line at the top; last wave's seal
    # CALLS are anywhere below it, so a hand edit that rewrites the marker and
    # misses the calls passes.  That is what happened on wave 40's own cut, and a
    # peer wave found it rather than this file.  `--audit` runs immediately after
    # the cut and before anything builds, when the correct number of seal calls is
    # ZERO; the wave appends its own after it reads its dossiers and does not
    # re-run this.  `seal () {` is the function definition and is not a call, so
    # the needle requires an identifier char.  WAVE 40 CARRIED ELEVEN CALLS HERE.
    calls = re.findall(r"(?m)^seal [A-Za-z0-9]", seals)
    if calls:
        problems.append(
            f"sealq.sh carries {len(calls)} seal CALL(s) at cut time, expected 0. "
            "A verbatim copy brings last wave's calls forward with every count "
            "intact, and rewriting the marker above them does not remove them"
        )
    if not (DST / "gainq.sh").is_file():
        problems.append("gainq.sh is missing: the terminal-gain solver has no driver")
    if not (DST / "tieq.sh").is_file():
        problems.append("tieq.sh is missing: wave 40's build module has no driver")
    if not (DST / "rowsetq.sh").is_file():
        problems.append("rowsetq.sh is missing: this wave's new build module has no driver")
    # WAVE 34's ADDITION, carried unchanged.
    for name in ("mx41_extscore.py", "mx41_extrubric2.py", "mx41_extbank2.py"):
        if not (DST / name).is_file():
            problems.append(f"{name} is missing: the second surface has no scorer")
    v1 = (DST / "mx41_extscore.py").read_text(encoding="utf-8")
    for key in ("ext-conversation-moving-week", "ext-conversation-competing-promises"):
        if f'"{key}"' not in v1:
            problems.append(
                f"mx41_extscore.py lost the frozen v1 rubric entry {key!r}: the "
                "frozen surface's rubric must keep scoring the frozen surface's bank"
            )
    # WAVE 35'S ADDITION, CARRIED.
    pen = (DST / "mx41_penaltydecode.py").read_text(encoding="utf-8")
    for needle, why in (
        ("NOT_A_FOLD", "the diagnostic must stamp its own artifacts as not-a-fold"),
        ("reproduces_the_sealed_decode",
         "the identity control must still be recorded in the artifact"),
        ("REFUSED: at penalty 1.0 this loop must be byte-identical",
         "the identity refusal must still fire, or the grid is unanchored"),
    ):
        if needle not in pen:
            problems.append(f"mx41_penaltydecode.py lost {needle!r}: {why}")
    # ------------------------------------------------------------------------
    # WAVE 36'S, 37'S AND 38'S ADDITIONS.  Wave 41's build list is `--ff 768
    # --excise 9`, so the composed width x MLP path, the live-mask repair and the
    # excision renumbering are all live code this wave, exactly as they were for
    # wave 40.
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
                f"mx41_build.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 39'S ADDITION, CARRIED -- and wave 41 runs this module FIVE times, and
    # imports it from the rowset builder besides.
    vocab = (DST / "mx41_vocab.py").read_text(encoding="utf-8")
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
         "Wave 41 passes that flag on every rung it builds, and a wave that "
         "passes a flag must assert the guard the flag disarms or it cannot tell "
         "disarming from absence"),
        ('            "the child tokenizer is not lossless on text; the byte alphabet or "\n', 1,
         "the refusal behind decode(child_encode(t)) == t"),
        ('    ranking = sorted(\n', 1,
         "THE OBJECT UNDER TEST THIS WAVE.  `choose_kept`'s discretionary "
         "ranking is what `mx41_vocabrowset.py` re-orders, and the rowset "
         "builder holds the mandatory set, the closure and K fixed against it.  "
         "If this call moved, the sealed control the arms are read against would "
         "not be the control they were sealed as"),
    ):
        got = vocab.count(needle)
        if got != want:
            problems.append(
                f"mx41_vocab.py: {needle!r} appears {got} times, expected {want} "
                f"-- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 40'S ADDITION, CARRIED.  This wave does not run the tie builder, so
    # these pins guard dormant code -- which is the case wave 39 was criticised
    # for and the reason the pins are kept rather than dropped: the module is
    # still in the tree and a future wave will run it.
    tie = (DST / "mx41_vocabtie.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(head_rows, table[head_index]):\n", 1,
         "the HEAD of a tie child is a verbatim gather; asserted with "
         "torch.equal and not a tolerance"),
        ("    if covered != len(tail_ids):\n", 1,
         "the tie's WHOLE CLAIM is that every dropped id has an image"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control"),
        ("    if decoded_probe != decoded_probe_again:\n", 1,
         "the id->token map must be a FUNCTION, and wave 41's own construction "
         "probe measured again that on this stack that is not free"),
        ("    if merges_kept != expected_merges:\n", 1,
         "the tokenizer-identity claim, checked by two implementations agreeing"),
    ):
        got = tie.count(needle)
        if got != want:
            problems.append(
                f"mx41_vocabtie.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 41'S OWN ADDITION.  `mx41_vocabrowset.py` is authored this wave, so it
    # has no history of surviving a cut and every load-bearing line in it is
    # pinned at its indented form from the start rather than after the first cut
    # eats one.
    rowset = (DST / "mx41_vocabrowset.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ("    if not torch.equal(gathered, table[index]):\n", 1,
         "a rowset child is a VERBATIM gather exactly as the deletion arm is.  "
         "Only the SELECTION moves; if a row's bytes moved too, a behavioural "
         "difference against the sealed twin could be either"),
        ("    if len(order) != len(sealed_order):\n", 1,
         "the arms are compared at FIXED K.  An arm that lands on a different "
         "row count is not a re-ranking, it is a different rung, and the "
         "comparison would be uncontrolled"),
        ("    if mandatory - keptset:\n", 1,
         "the mandatory set is held EXACTLY as sealed, so coverage cannot move "
         "and the below-floor gate cannot fire differently between arms.  An arm "
         "that quietly dropped a mandatory row would publish a selection result "
         "that was really a coverage result"),
        ("    if int(live) != int(expected_parameters):\n", 1,
         "the parameter count is the control, stated on the command line and "
         "asserted before anything is written -- wave 40's tie-builder rule, "
         "carried to the arm that needs it for the same reason"),
        ("    if closure_state[\"intact\"] is not True:\n", 1,
         "the STRUCTURAL half of the census gate is not releasable on a rowset "
         "arm any more than on a gather.  A re-ranked kept set that cannot be "
         "PRODUCED is a broken tokenizer wearing a selection receipt"),
        ("    if arm == \"C\":\n", 2,
         "Arm C is the slate's own leverage arm and it is BUILT, with the "
         "wave-9 prediction that it fails.  A predict-and-skip would close the "
         "class on a prediction, which is the thing this campaign refuses.  TWO "
         "sites, both load-bearing and both pinned: the key constructor, which "
         "refuses to hand back a leverage key with no leverage computed, and the "
         "builder, which computes the tied Gram and SAYS SO IN THE LOG so a "
         "reader of the build log can see the arm was built rather than "
         "predicted"),
    ):
        got = rowset.count(needle)
        if got != want:
            problems.append(
                f"mx41_vocabrowset.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    # ------------------------------------------------------------------------
    # WAVE 41'S ONE EDIT TO A CARRIED MODULE, PINNED IN BOTH DIRECTIONS.
    # `NOTES-MX-AR` §11.6.  The fix ADDS a path and removes none, so the stdout
    # print that every earlier wave used is pinned as STILL PRESENT.
    index_py = (DST / "mx41_child_index.py").read_text(encoding="utf-8")
    for needle, want, why in (
        ('    ap.add_argument("--out")\n', 1,
         "the generator must take an explicit output path.  It printed to stdout "
         "and wrote the same file itself, so the obvious invocation -- redirect "
         "stdout into that file -- corrupted it, on wave 35 and again on wave 40"),
        ("    reread = json.loads(open(path, encoding=\"utf-8\").read())\n", 1,
         "parse-what-you-write moves INSIDE the generator.  `NOTES-MX-AO` §9's "
         "rule caught the corruption twice from the caller; the third time it "
         "should not need a caller who remembers the rule"),
        ('    print("->", path)\n', 1,
         "the stdout summary every earlier wave read is KEPT.  This edit adds a "
         "path and removes none"),
    ):
        got = index_py.count(needle)
        if got != want:
            problems.append(
                f"mx41_child_index.py: {needle!r} appears {got} times, expected "
                f"{want} -- {why}"
            )
    if problems:
        for p in problems:
            print("AUDIT FAILED:", p, file=sys.stderr)
        return 1
    print("audit: no wave-40 identifier survived; eight seal roots declared; "
          "the accept list kept thirteen basis schemas; the drivers name wave 41; "
          "sealq.sh carries this wave's marker and ZERO seal calls; gainq.sh, "
          "tieq.sh and rowsetq.sh exist; both extension surfaces have a scorer "
          "and v1 still names the v1 bank; the penalty diagnostic kept its "
          "NOT_A_FOLD banner and its identity refusal; wave 36's composed width x "
          "MLP path, wave 37's live-mask repair and wave 38's three multi-index "
          "excision lines are present AND are live code this wave; the builder's "
          "log prefix names THIS wave and no stale tag survives; all six "
          "vocabulary-gather lines are present at their indented form INCLUDING "
          "the ranking call this wave re-orders; wave 40's five tie-builder lines "
          "are carried; all six lines the NEW rowset builder depends on are "
          "pinned at theirs; and the child-index edit is pinned in both "
          "directions")
    return 0


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    raise SystemExit(main())
