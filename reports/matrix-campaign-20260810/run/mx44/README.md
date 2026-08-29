# `run/mx44` — how this tree was cut, what in it is wave 44's, and the instrument this wave registers BEFORE it reads anything

Every file here except `cut_wave44.py` and this README was produced from
`run/mx43` by `cut_wave44.py`. **The docstrings are wave 43's and earlier, on
purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 43 ran it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 43` | `--campaign-wave 44` | the sealer's wave argument |
| `"campaign_wave": 43` | `"campaign_wave": 44` | the extension surface's own field |
| `wave43` | `wave44` | a path component and a schema infix |
| `mx43` | `mx44` | module names, the build-receipt filename, the log tag |
| `MX43` | `MX44` | the child-directory prefix and the `$MX44` shell variable |

`wave 43`, `Wave 43` and `WAVE 43` **in English are not substituted**, so a
header that records what wave 43 measured still says wave 43 measured it.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx44_verify_seals.py` — the rename turns `wave43/cells` into `wave44/cells`
  and thereby deletes last wave's seal root. One appended line: **39 declared
  roots become 40, `wave43/cells` and `wave42/cells` still declared.** A wave-44
  paragraph was added to the header stack, which is the second consecutive wave
  to write one after four waves of appending a LINE without a PARAGRAPH.
* `mx44_retain.py` — an inherited `KEEP` set names last wave's children and
  everything unnamed is deleted. It arrived carrying **wave 43's two keeps and
  one drop with every justification paragraph intact**; both mappings were
  emptied by hand and the wave root, glob, schema and manifest name retargeted
  by hand.
* `mx44_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **One schema string retargeted by hand, wave 43's
  basis schema appended** (the accept list now carries wave 44 back to 19),
  **sixteen module references retargeted by name**, **both build-receipt write
  sites** moved to `mx44-build-receipt.json`, the log prefix carried to `[mx44]`,
  and the header's own diff reference retargeted to `mx43_build.py`, which is the
  file it was actually copied from. **Wave 44 does not run this file.** The edits
  are still made, because the next wave that runs it inherits whatever this cut
  leaves behind.

  This edit **broke the file once before it worked**, and the header says so:
  the first attempt spliced the new docstring in at the OPENING quotes instead
  of the closing ones and left wave 43's docstring body loose at module level.
  `py_compile` refused it, the file was re-copied verbatim and the edit redone.
  The refusal cost nothing because it fired before anything was committed.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. **Twelfth consecutive cut where this protection refused.** The file
  arrived with wave 43's **three** seal calls, every count intact, and wave 42's
  general calibration pin reported **62** residue lines below the sentinel. All
  62 deleted by hand.

**3. WAVE 44 AUTHORS NO MODULE AND EDITS EXACTLY ONE.** That is the opposite
shape from wave 43, which authored two and edited none, and it is stated rather
than left to a diff. The edit is `mx44_dualband.py` and it is §4.

**4. THE THREE PATH RESOLUTIONS INSIDE `sealq.sh` ARE PINNED FOR THE FIRST
TIME, AND TWO OF THE THREE PINS NEVER EXISTED.** `NOTES-MX-AU` §13 item 6 said
the natural next place to look is every file a wave edits by hand each time with
nothing but prose behind it. The nearest one is not another file — it is three
lines inside a file that is already in `HAND_EDIT`.

Because `sealq.sh` is copied verbatim, every cut has to retarget by hand the
wave's `env.sh`, the wave's per-case writer and the wave's `seal_verdict.sh`.
Wave 43's own copy says all three "were retargeted by hand at pinned counts of
one each". `FORBIDDEN_AFTER`'s needles are `mx43_` and `MX43-`. `nf-mx43/env.sh`
carries `mx43/` and `$MX43/seal_verdict.sh` carries `MX43/`, and **neither needle
reaches either one** — measured, not asserted. Only the per-case writer was
covered, and only because its filename happens to carry an underscore.

The cost of that hole is not cosmetic: `env.sh` exports `$W`, the wave root every
cell, shard, log and dossier is written under. A `sealq.sh` that kept last wave's
source line would have written this wave's seals into last wave's wave root under
this wave's cell ids, and nothing in the campaign would have said so.
`cut_wave44.py --audit` now pins all three in both directions.

---

## 2. THE INSTRUMENT, REGISTERED HERE, BEFORE IT IS RUN

`mx44_dualband.py`'s `payload_union_coverage` is the **primary** copy-back
measure at `COPY_BACK_UNION_COVERAGE = 0.60`. The longest-contiguous-run measure
is computed on every lane, published per lane as
`legacy_longest_run_classification`, and decides nothing.

**Where 0.60 comes from, and it is not from anything this wave measured.**
`wave42/deep-band-dual-assessment.json` — sha256
`fe20e9164023fc9a639aa8e1c05617b28d7f4890fc27bc9bea65646d8780602c`, committed at
`5af0b035` two waves before this file existed — carries **34 distinct
summarisation responses** in its band. Seventeen were hand-read as copies and
their union coverage runs **0.826972 to 0.997455**. Every other distinct response
in that lane sits at **0.483461 or below**. The separating interval is
**(0.483461, 0.826972)**, it is 0.3435 wide, and nothing lies inside it. 0.60
clears the highest non-copy by 0.1165 and undercuts the lowest copy by 0.2270,
and it is the numeral the campaign already declared for the demoted measure, so
this wave changes the MEASURE without also changing the NUMBER.

**The threshold is declared non-load-bearing and that is checkable**: any value
strictly inside the interval reproduces all 34 calls identically.

### 2.1 THE PREDICTIONS. Registered in this commit, before the pass that tests them runs.

`mx44_dualband.py` publishes three column-B variants. Two of them read no hand
input at all, which is what makes this a test of the instrument rather than the
instrument grading itself.

| # | quantity | registered prediction |
|---|---|---|
| **P1** | `floor_under_copy_back_refused_machine_union_only`, machine pass, no `--hand-reading` | **0.4116484014x, `ARCH.V10CL896X09`** — the union measure ALONE reproduces the floor that until now only a human could produce |
| **P2** | `floor_under_copy_back_refused_machine_legacy_only`, same pass | **0.3951221469x, `ARCH.V5CL768`** — wave 43's machine-only artifact reproduced exactly, which is the regression check that the demoted path is unchanged |
| **P3** | `machine_hand_disagreements` on the hand-merged pass | **falls from 23 to 3 or fewer** |

**AND THE WAY P1 FAILS IS NAMED BEFORE THE MEASUREMENT.** The lanes that survive
column B are not summarisation lanes: they are `uncertainty-insufficient-data`
and `context-retrieval`, both hand-read CONTENT. `uncertainty-insufficient-data`
has **no payload anchor**, so its payload is the concatenation of its user turns
— the whole prompt — and a response that quotes the question back could score
high union coverage against it. If either lane clears 0.60, the machine voids it,
`column_b_machine_union_only` collapses below 0.4116484014x, and P1 is wrong. A
union measure that is right about copies and wrong about selection lanes is a
worse instrument than the one it replaces, and this wave would say so.

If P1 misses, the hand reading still governs — that rule is wave 42's and this
wave does not touch it — and the estate's column B does not move. What moves is
the claim that the machine can be trusted with it.

### 2.2 What the retrospective column is, and what it is not

Every child with `campaign_wave < 44` carries
`instrument_reading_is_retrospective: true`. **Nothing about those children is
re-graded.** Their sealed counts stand, column A is still the sealed per-case
verdict the module refuses to disagree with, and the hand calls waves 42 and 43
recorded remain authoritative. The retrospective column is published to show what
the instrument does on a population whose answer is already known by other means.

Wave 44's own three children carry `false`. On those the union measure is the
**first** call, taken by machine, before any dossier is read.

---

## 3. What this tree carries and does not run

Wave 44's build list is **three `mx44_vocab.py` gathers** — `V4CL768`,
`V3CL768` and `V5CL896X13`. Wave 39's six vocabulary-gather lines are live code
this wave and are pinned at their indented form.

Carried and **run**: `mx44_dualband.py`, twice — once machine-only, which is the
pass P1 and P2 are registered against, and once with the hand reading merged.

Carried and **not run**: `mx44_build.py` (wave 36's composed width × MLP path,
wave 37's live-mask repair, wave 38's descending-order excision),
`mx44_vocabtie.py`, `mx44_vocabrowset.py` and `mx44_termgain.py` with their
drivers, and **wave 43's two solver modules `mx44_plane.py` and
`mx44_xchange.py`**. The PLANE axis was re-sealed by the child that opened it and
XCHANGE was closed by its own pre-registered gate, so neither has anything wave
44 can legitimately spend a review on; both keep every pin wave 43 put on them,
because a module whose axis is closed is exactly the module a later cut is most
likely to damage silently.

**NORM-ESTATE is still not authored, ALIAS is still blocked on the PROBE storage
ruling, and ROWSET Arms A and B are still blocked on a corpus.** None of those
three is an authoring problem, which is why the wave whose deliverable is an
instrument did not spend itself on them.
