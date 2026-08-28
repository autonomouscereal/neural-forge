# `run/mx34` — how this tree was cut, and what in it is wave 34's

Every file here except `cut_wave34.py`, `mx34_extbank2.py`, `mx34_extrubric2.py`,
`mx34_calibrate.py`, this README and `wave34-tree-cut-manifest.json` was produced
from `run/mx33` by `cut_wave34.py`. **The docstrings are wave 33's and earlier, on
purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32, 33 and 34 run it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 33` | `--campaign-wave 34` | the sealer's wave argument |
| `"campaign_wave": 33` | `"campaign_wave": 34` | the extension surface's own field |
| `wave33` | `wave34` | a path component and a schema infix |
| `mx33` | `mx34` | module names, the build-receipt filename, the log tag |
| `MX33` | `MX34` | the child-directory prefix and the `$MX34` shell variable |

`wave 33`, `Wave 33` and `WAVE 33` **in English are not substituted**, so a
header that records what wave 33 measured still says wave 33 measured it. The
provenance is here, once, rather than rewritten thirty times.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back. **Three of the four were finished after the wave
34 that cut this tree died mid-edit**; the state that agent left is written up in
`NOTES-MX-AK` §1.3 and the finishing is `NOTES-MX-AL` §1.

* `mx34_verify_seals.py` — the rename turns `wave33/cells` into `wave34/cells`
  and thereby deletes last wave's seal root. One appended line: **29 declared
  roots become 30, `wave33/cells` and `wave32/cells` still declared.** *(Done by
  the wave that cut the tree; verified, not redone.)*
* `mx34_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. `KEEP` and `DROP` are **emptied**, the wave root,
  the glob, the schema and the manifest name retargeted. *(Done by the wave that
  cut the tree. It is empty BY DECISION, not by substitution accident, and with
  no `wave34/children/` on disk at the time it could not have deleted anything.)*
* `mx34_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 33's
  basis schema appended** (the accept list now carries wave 34, 33, 32, 31, 30,
  29, 23, 22, 21, 20 and 19), **nine module references retargeted by name**. Nine
  rather than wave 33's eight because wave 33 authored one more of them into its
  own header. *(Outstanding at the crash; finished this wave.)*
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. It arrived with **wave 33's marker and wave 33's six terminal-gain
  calls**, `cut_wave34.py --audit` refused on the marker before anything ran, and
  the six were deleted by hand. **Second consecutive cut where this protection
  refused.** *(Outstanding at the crash; finished this wave.)*

`cut_wave34.py --audit` reads every file back and refuses a surviving `mx33_`,
`MX33-`, `/wave33/` or `--campaign-wave 33`, with one declared exception at a
pinned count: `mx34_build.py`'s wave-34 header names `mx33_build.py` as the file
it states its diff against. It also asserts three seal roots are declared, that
the accept list carries six basis schemas, that `seal_verdict.sh` still passes
`--parent-useful-or-partial 12` and `--lanes-total 16`, that `sealq.sh` carries
this wave's marker, that `gainq.sh` exists, and — wave 34's own addition — that
both extension surfaces have a scorer and that the FROZEN v1 rubric still names
two of its own thirty entries by id.

**The diff against `cut_wave33.py` is three things.** `gainq.sh` moves out of
`AUTHORED_THIS_WAVE` and is carried by the cut like any other driver, with the
audit asserting it arrived; `ALLOWED_HISTORICAL` starts with the build header
only, because `mx32_` is not a forbidden needle for a cut whose ancestor is wave
33; and the audit gained the extension-surface assertions above.

## What wave 34 authored, beyond the cut

* **`mx34_extrubric2.py` / `mx34_extbank2.py` / `research/lfm2_extension_bank_prompts_v2.json`**
  — the replacement second review surface. Thirty prompts, two per frozen lane,
  A/B difficulty staggered, zero seven-gram overlap with the sealed pack, the v1
  extension bank or the paraphrase triplets, and every reference answer reachable.
  `--self-test` runs on every invocation and refuses if any prompt satisfies its
  own rule, if any rule grades form alone, or if the degeneracy veto is open.
  Bank sha256 `340694a64f7e886ecf1576dd9245a1445d8e763036d842f3730c4c8ab5c9c592`,
  rubric sha256 `fc7564d7f0ebc9344ee02eb359d39f6d13edc5127df2915c70328a3ce540cfe9`.
  **Authored by the wave that cut this tree, verified intact and carried.**
* **`mx34_calibrate.py`** — the calibration this wave owes the v2 surface. It
  scores a population under the v2 bank and reports Spearman rho against **both**
  the sealed frozen-fifteen counts and the canonical intelligence counts, with
  the pooled 0.69 of the retired v1 surface as the number to beat, a permutation
  null, and leave-one-out influence. A surface that cannot be shown to rank
  better than the one it replaces is reported as no-better rather than adopted.

## What is NOT in this tree

`mx34_robustforensics.py` is named in `AUTHORED_THIS_WAVE` and was never written.
The cutter therefore leaves that slot alone and the audit does not assert it,
because asserting a file that was never designed would be a guard that fires on
an intention. Wave 33's robustness probe stands as the last word on rewording.
