# `run/mx35` — how this tree was cut, and what in it is wave 35's

Every file here except `cut_wave35.py`, this README and
`wave35-tree-cut-manifest.json` was produced from `run/mx34` by `cut_wave35.py`.
**The docstrings are wave 34's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32, 33, 34 and 35 run it with a small stated diff each
time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 34` | `--campaign-wave 35` | the sealer's wave argument |
| `"campaign_wave": 34` | `"campaign_wave": 35` | the extension surface's own field |
| `wave34` | `wave35` | a path component and a schema infix |
| `mx34` | `mx35` | module names, the build-receipt filename, the log tag |
| `MX34` | `MX35` | the child-directory prefix and the `$MX35` shell variable |

`wave 34`, `Wave 34` and `WAVE 34` **in English are not substituted**, so a
header that records what wave 34 measured still says wave 34 measured it. The
provenance is here, once, rather than rewritten thirty times.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx35_verify_seals.py` — the rename turns `wave34/cells` into `wave35/cells`
  and thereby deletes last wave's seal root. One appended line: **30 declared
  roots become 31, `wave34/cells` and `wave33/cells` still declared.**
* `mx35_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. Wave 34's **five keeps and six drops** arrived
  here verbatim. `KEEP` and `DROP` are **emptied by decision**, the wave root,
  the glob, the schema and the manifest name retargeted by hand. The inherited
  justification paragraphs were deleted rather than renamed, because prose
  arguing for last wave's rows is how a retention decision gets made by the
  previous wave.
* `mx35_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 34's
  basis schema appended** (the accept list now carries wave 35, 34, 33, 32, 31,
  30, 29, 23, 22, 21, 20 and 19), **thirteen module references retargeted by
  name**, and **both build-receipt write sites** moved to
  `mx35-build-receipt.json`. That last item is wave 34's defect, and the reason
  this file stays in the hand-edit set: the literal said
  `mx32-build-receipt.json` from wave 32 until wave 34 became the first wave
  since 32 to call the builder.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. It arrived with **wave 34's marker and wave 34's eight seal calls**,
  `cut_wave35.py --audit` refused on the marker before anything ran, and the
  eight were deleted by hand. It also arrived sourcing **`nf-mx34/env.sh`** and
  calling **`$MX34/mx34_percase.py`** — three path resolutions a verbatim copy
  cannot fix for itself — and those were retargeted by hand too. **Third
  consecutive cut where this protection refused.**

`cut_wave35.py --audit` reads every file back and refuses a surviving `mx34_`,
`MX34-`, `/wave34/` or `--campaign-wave 34`, with two declared exceptions at
pinned counts: `mx35_build.py`'s and `mx35_penaltydecode.py`'s wave-35 headers
each name the wave-34 file they state their diff against. It also asserts four
seal roots are declared, that the accept list carries seven basis schemas, that
`seal_verdict.sh` still passes `--parent-useful-or-partial 12` and
`--lanes-total 16`, that `sealq.sh` carries this wave's marker, that `gainq.sh`
exists, and that both extension surfaces have a scorer with the FROZEN v1 rubric
still naming two of its own thirty entries by id.

**The diff against `cut_wave34.py` is four things.**

1. `AUTHORED_THIS_WAVE` drops from six files to two. Wave 34 listed
   `mx34_robustforensics.py` in that set and never wrote it, so the set
   advertised a file that did not exist for a whole wave. Wave 35's behavioural
   changes are hand edits on carried copies, named below.
2. `ALLOWED_HISTORICAL` carries two entries rather than one.
3. The build-receipt assertion pins **both** known-stale filenames at zero —
   `mx34-` and `mx32-` — rather than only the most recent. A defect that
   survived two hand edits survived them behind a check that existed for
   neither of them.
4. **The audit gained a guard on `mx35_penaltydecode.py`**, which is the one
   file this wave edits behaviourally. It is the only module in the tree that
   decodes a sealed child through a knob the frozen protocol does not have, and
   everything that keeps it a diagnostic rather than a second scoring standard
   is text in the file. The audit asserts the `NOT_A_FOLD` banner, the recorded
   identity control and the byte-identity refusal are all still present, by
   name.

## What wave 35 changed, beyond the cut

* **`mx35_penaltydecode.py`** — the decode-repair grid. Wave 34 moved one knob
  to one value (repetition penalty 1.3) and found it eliminated the artifact
  class entirely, recovered nothing, and destroyed all thirteen lanes of signal
  the population held. That priced **one setting**, and 1.3 is the aggressive
  end. Wave 35 adds `--no-repeat-ngram N` — a **hard** constraint that removes
  a repeating token from the argmax and leaves every other token at exactly its
  sealed logit — and runs the mild end of the penalty beside it. The n-gram
  bank includes the **prompt**, as `transformers`' own processor does, which is
  what lets the arm stop a prompt echo and also what lets it forbid an answer
  that quotes the context; both directions are counted. The identity setting is
  now **both** knobs off, and the refusal checks both.
  The incremental ban map was cross-checked against a brute-force rescan on
  4,000 random sequences and 50,448 decode steps before it was run on a GPU.
* **`sealq.sh`** — this wave's marker, calibrations and seals.
* **`mx35_retain.py`** — this wave's retention policy, written after the
  verdicts.

## What is NOT in this tree

No new module. Wave 35 builds one body, composes vocabulary folds and
terminal-gain arms on it and on wave 34's retained bodies, and re-runs an
existing diagnostic with one more knob. Every number it produces is produced by
code that has already been self-tested, which is the only way its children are
comparable to the sealed rungs beside them.
