# `run/mx33` — how this tree was cut, and what in it is wave 33's

Every file here except `cut_wave33.py`, `gainq.sh`, `mx33_instrument.py`,
`mx33_pararubric.py`, `mx33_mforensics.py`, `mx33_condinventory.py`, this README
and `wave33-tree-cut-manifest.json` was produced from `run/mx32` by
`cut_wave33.py`. **The docstrings are wave 32's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 and 33 run it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 32` | `--campaign-wave 33` | the sealer's wave argument |
| `"campaign_wave": 32` | `"campaign_wave": 33` | the extension surface's own field |
| `wave32` | `wave33` | a path component and a schema infix |
| `mx32` | `mx33` | module names, the build-receipt filename, the log tag |
| `MX32` | `MX33` | the child-directory prefix and the `$MX33` shell variable |

`wave 32`, `Wave 32` and `WAVE 32` **in English are not substituted**, so a
header that records what wave 32 measured still says wave 32 measured it. The
provenance is here, once, rather than rewritten thirty times.

**2. Four files are copied with no substitution at all**, and every edit to
them was made by hand and read back:

* `mx33_verify_seals.py` — the rename turns `wave32/cells` into `wave33/cells`
  and thereby deletes last wave's seal root. One appended line: **28 declared
  roots become 29, `wave32/cells` and `wave31/cells` still declared.**
* `mx33_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. `KEEP` and `DROP` are emptied; the wave root,
  the glob, the schema and the manifest name are retargeted by hand.
* `mx33_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. Five schema strings retargeted by hand, **wave 32's
  basis schema appended** (the accept list now carries wave 33, 32, 31, 30, 29,
  23, 22, 21, 20 and 19), eight module references retargeted by name.
* **`sealq.sh` — and this is the first cut where that membership does anything.**
  Wave 32 discovered that its own cut had carried wave 31's nineteen seal calls
  forward with every count intact, added this file to `HAND_EDIT` for the next
  cut, and made `--audit` assert that the seal marker names the current wave.
  Wave 33 is the next cut. The file arrived verbatim carrying **wave 32's marker
  and wave 32's fifteen seal calls**, the audit refused on the marker before
  anything ran, and the fifteen were deleted by hand. **A protection whose first
  exercise is also its first refusal is a protection that has been tested.**

`cut_wave33.py --audit` reads every file back and refuses a surviving `mx32_`,
`MX32-`, `/wave32/` or `--campaign-wave 32`, with one declared exception at a
pinned count: `mx33_build.py`'s wave-33 header names `mx32_build.py` as the file
it states its diff against. It also asserts three seal roots are declared, that
the accept list still carries five basis schemas, that `seal_verdict.sh` still
passes `--parent-useful-or-partial 12` and `--lanes-total 16`, that `sealq.sh`
carries this wave's marker, and that `gainq.sh` exists.

**The diff against `cut_wave32.py` is three things.** `HAND_EDIT` is exercised on
`sealq.sh` for the first time (above); `gainq.sh` is in `AUTHORED_THIS_WAVE`
because it does not exist in `run/mx32` to be carried; and `ALLOWED_HISTORICAL`
starts with only the build header, so wave 32's two-count retention waiver is not
inherited by prose this wave has not written yet.

## What wave 33 authored, beyond the cut

* **`mx33_instrument.py`** — I1. Four waves of committed `extension-score.json`
  pooled into sixty child observations against one frozen bank, to answer whether
  the surface broke in wave 32 or was never characterised. It computes the
  band-restricted rho the wave was sent to look for (flat: the band hypothesis is
  falsified), the **gap-concordance resolution curve** that is the honest form of
  the question, a **null distribution for each wave's rho** holding the instrument
  fixed and varying only the population, leave-one-out influence, a **held-out
  rubric trim** selected on waves 29-31 and evaluated on wave 32, the
  **degeneracy-veto counterfactual**, and the **prompt-satisfiable census** — the
  rubric entries a child can hold by quoting the question back.
* **`mx33_pararubric.py`** — the rubric wave 32 could not write. Keyed by LANE,
  because the fifteen paraphrase lanes are one-to-one with the sealed fifteen
  cases. It closes the degeneracy veto, requires content rather than form, and
  **self-tests every rule against its own three prompts on every invocation**,
  refusing if any prompt satisfies its own rule. Two lanes are declared
  UNANSWERABLE and not scored: their restatements say "the note below" / "the
  update" and carry neither.
* **`mx33_mforensics.py`** — F1. Why the 8/15 body composes at 3/15 and the 7/15
  body composes at 7-8, read off sealed per-case verdicts and recovered channel
  selections.
* **`mx33_condinventory.py`** — A2. Every solve-bearing sealed child's build arm,
  read out of the receipts rather than remembered.
* **`gainq.sh`** — re-authored from wave 29's, because waves 30, 31 and 32 each
  carried `mx*_termgain.py` through the cut and lost the queue that drives it.
