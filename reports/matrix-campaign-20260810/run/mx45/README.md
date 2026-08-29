# `run/mx45` — how this tree was cut, the anchor debt paid at the cut, and the outcome registered before the pass

Every file here except `cut_wave45.py` and this README was produced from
`run/mx44` by `cut_wave45.py`. **The docstrings are wave 44's and earlier, on
purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 44 ran it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 44` | `--campaign-wave 45` | the sealer's wave argument |
| `"campaign_wave": 44` | `"campaign_wave": 45` | the extension surface's own field |
| `wave44` | `wave45` | a path component and a schema infix |
| `mx44` | `mx45` | module names, the build-receipt filename, the log tag |
| `MX44` | `MX45` | the child-directory prefix and the `$MX45` shell variable |

`wave 44`, `Wave 44` and `WAVE 44` **in English are not substituted**.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx45_verify_seals.py` — **40 declared roots become 41, `wave44/cells` still
  declared, `wave45/cells` appended.** `cut_wave45.py --audit` asserts twelve
  roots by name. A wave-45 paragraph was added to the header stack — the third
  consecutive wave to write one.
* `mx45_retain.py` — arrived carrying **wave 44's one keep and two drops with
  every justification paragraph intact**. Both mappings emptied by hand; the
  wave root, glob, schema and default manifest name retargeted by hand at pinned
  counts of one each. Sixteenth consecutive wave.
* `mx45_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only. **One schema string
  retargeted by hand, wave 44's basis schema appended** (the list now carries
  wave 45 back to 19, seventeen strings deep), **eighteen module references
  retargeted by name**, **both build-receipt write sites** moved to
  `mx45-build-receipt.json`, the log prefix carried to `[mx45]`, and the header's
  own diff reference retargeted to `mx44_build.py`. **Wave 45 does not run this
  file.** The edits are still made.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. **Twelfth consecutive cut where this protection refused.** The file
  arrived with wave 44's **three** seal calls, every count intact, and wave 42's
  general calibration pin reported **83 residue lines below the sentinel — the
  largest count it has ever reported** (wave 43: 78, wave 44: 62). All 83
  deleted by hand. The block named carried wave 44's seventeenth consecutive
  calibration with its 6/6/3 split, its lane byte comparison and its adjudication
  of a held lane that returns the whole prompt.

**3. WAVE 45 AUTHORS NO MODULE AND EDITS EXACTLY ONE.** Same shape as wave 44.
The edit is `mx45_dualband.py` and it is §2 and §3.

**4. THE SEALER'S OWN BAND CONSTANTS ARE PINNED FOR THE FIRST TIME IN
FORTY-FIVE WAVES.** `NOTES-MX-AV` §12.6 named
`experiments/seal_matrix_behavior_review.py`'s constants as the next place to
look, because `seal_verdict.sh` passes flags to that file and no cut has ever
read it. Reading it leads somewhere narrower and worse: **the numbers that turn
a reviewer's count into a VERDICT are not in that file at all.**
`behavior_verdict_for` reads `PARENT_REVIEW_BAND = 12`, `A9_REVIEW_BAND = 8` and
`PARENT_BAND_TOLERANCE = 2` out of `src/neural_forge/matrix_cell_receipt.py` —
outside the cut tree, outside the sealer, and **never asserted by anything in
this campaign**. Every `alive-parent-band` / `alive-degraded` / `impaired` label
in 1,052 ledger rows rests on those three integers, and the `dead` line that
every deep-band child in this campaign is sealed through is one `if` beside them.

`cut_wave45.py --audit` now pins all three **by value**, plus the band
arithmetic and the death line at their indented forms, plus both verdict
vocabularies, plus the sealer's three argparse defaults (`--smoke-verdict`
`coherent`, `--instance-suffix` `a6review`, `--lanes-total` `16`) and the
*absence* of a default on `--parent-useful-or-partial`, which must stay absent
so a wave that forgets the calibration flag fails instead of inheriting a number.

Nothing was found wrong. That is the result: a fourth external surface is now
checked, and it was checked before it cost anything, which is not how the first
three were found.

---

## 2. THE ANCHOR DEBT, PAID AT THE CUT — AND ITS OUTCOME REGISTERED HERE, BEFORE THE PASS

`NOTES-MX-AV` §2.3 measured a defect and refused to repair it mid-measurement:
**all eight void-changing machine–hand disagreements in the band sit on lanes
with no `PAYLOAD_ANCHORS` entry**, seven of them on `writing-concise-email`.
§12 item 1 ordered the entry registered *before* the pass that uses it. This
commit is that registration.

```
"writing-concise-email": ("under 90 words", None),
```

**The cut point is the rule the other three anchors already follow** — strip the
leading task directive, keep the material the lane supplies — applied to a
prompt whose directive is `Draft a polite email under 90 words`. It is not
chosen to move a number.

### 2.1 THE REGISTERED PREDICTION IS A NULL RESULT, AND IT IS REGISTERED ANYWAY

The seven rows carry exactly **two distinct responses**, both committed in
`wave44/deep-band-dual-assessment.json` and both readable in the frozen
dossiers:

| sha256/16 | children | union coverage vs whole prompt (wave 44) |
|---|---:|---:|
| `39fdfd521947817c` | `V9CL896G03`, `V9CL896G10`, `V10CL896G03`, `V10CL896G10` | 0.481865 |
| `6022740936b6da1e` | `V12CL896G03`, `V14CL896G03`, `V16CL896G03` | 0.461140 |

Both are the same failure: *"I'm here here. Please. I'm a polite email that's
asking a vendor to confirm whether a delayed shipment will arrange by Friday.
I'm here."* — the model hands back the prompt's **first sentence** and drops the
second entirely.

Under the registered anchor the payload is 157 characters instead of 193 and the
two responses read **0.477707** and **0.452229**. Both are still below 0.60.
They are also *slightly lower than before*, because the string `a polite email`
lies inside the stripped directive and the strip removes matched union along
with denominator.

| # | quantity | registered prediction |
|---|---|---|
| **P1** | `writing-concise-email` void-changing disagreements, 57-child band, hand-merged pass | **stays at 7.** The ordered anchor does NOT close the rows it was ordered to close |
| **P2** | total void-changing disagreements | **stays at 8**, not the 0-or-1 `NOTES-MX-AV` §12.1 predicted |
| **P3** | both floors | **unmoved by the anchor.** Column A 0.3859919152x, column B 0.4116484014x. No lane held under column B is a `writing-concise-email` lane, and the anchor touches no other lane's slice |
| **P4** | `response_union_coverage` on those two responses | **0.683824 and 0.635714** — over 0.60, with no anchor at all |

**Why register a prediction you expect to hit.** Because the alternative is
worse. The one cut that *would* move the seven rows is keeping only the prompt's
first sentence: payload 112 characters, coverage **0.821429** and **0.785714**.
It works by deleting the exact sentence the model failed to copy. Registering
that cut would be fitting the instrument to the answer — the thing wave 42
refused, wave 44 refused, and this wave refuses in writing rather than by not
trying it. The full candidate table is published in §2.2 so the refusal is
checkable and not merely stated.

### 2.2 THE SENSITIVITY SWEEP, PUBLISHED WITH THE REGISTRATION

Payload = the normalised user turn under each candidate cut. Coverage is the
same union at `UNION_MIN_BLOCK = 8` against the same 0.60.

| candidate | payload chars | `39fdfd…` | `6022…` | verdict |
|---|---:|---:|---:|---|
| none (wave 44's fallback: whole prompt) | 193 | 0.481865 | 0.461140 | CONTENT |
| **`("under 90 words", None)` — REGISTERED** | 157 | 0.477707 | 0.452229 | CONTENT |
| `("Draft a polite email", None)` | 172 | 0.447674 | 0.424419 | CONTENT |
| `("asking", None)` | 150 | 0.453333 | 0.426667 | CONTENT |
| first sentence only — **REFUSED** | 112 | 0.821429 | 0.785714 | COPY_BACK |
| second sentence only | 80 | 0.000000 | 0.000000 | CONTENT |

**Four of the five defensible cuts agree with each other and disagree with the
hand.** That is not an anchor that needs tuning; it is a measure asking the
wrong question, and §3 is where that goes.

---

## 3. THE SECOND EDIT: A PUBLISHED COLUMN THAT DECIDES NOTHING

`response_union_coverage` is the same union, at the same minimum block, against
the same 0.60 numeral, with the **response** as the denominator instead of the
payload.

**The question changes and that is the whole point.** Payload-side asks *how
much of the supplied material came back* — right for a lane that supplies
material. `writing-concise-email` is a `PRODUCE` lane: it supplies nothing, the
whole prompt is instruction, and a model that hands back half the instruction
cannot score 0.60 against the whole of it no matter where the anchor is cut.
The question that lane needs is *how much of what this model said is lifted from
the prompt*, and that is a **denominator** change, not an anchor change.

`RESPONSE_SIDE_DECIDES_NOTHING = True` is declared in one constant and
`cut_wave45.py --audit` pins the decision line `if response_union >=
COPY_BACK_UNION_COVERAGE:` at **zero occurrences**. There is no branch anywhere
in the module that lets this number change a classification.

**This is the campaign's own promotion protocol, followed deliberately.** Waves
42 and 43 published the union measure beside the deciding one for two waves;
wave 44 promoted it on that published evidence rather than on an argument, and
it reproduced the hand column with nobody in the loop. A wave that wants to
promote the response-side column has to say so at a cut, with two waves of
published numbers behind it. Wave 45 publishes the first.

### 3.1 A DEFECT THIS WAVE MADE, AND CAUGHT, INSIDE THE CUT

The first version of the column read `union_coverage(verbatim, payload)` — the
response against the **anchored** slice. It ran, it compiled, it published, and
it was wrong: on the two responses above it returned **0.551471** and
**0.507143**, both under 0.60, and it would have shipped a column that appeared
to say the denominator change buys nothing.

The reason is the one this section is about. Anchoring the source removes the
instruction text, and the instruction text is exactly what a `PRODUCE`-lane copy
lifts. A response-side measure taken against an anchored source cannot see the
lift it exists to see.

It was caught by checking the README's registered numbers against the module
that would produce them, *before* the cut was committed — the discipline
`NOTES-MX-AO` §9 states for artifacts, applied to a registration. The fix is
`lane_whole_prompt()`, a named accessor both paths share so they cannot drift
apart about what "the prompt" is, and `cut_wave45.py --audit` pins the call site
at its indented form *including its source argument*. Recorded here rather than
silently corrected, because a wave that only reports the defects it found in
other waves is not measuring itself.

---

## 4. What this tree carries and does not run

Wave 45's build list is **two `mx45_vocab.py` gathers** — `V2CL768` (K=2,048)
and `V1CL768` (K=1,024). Wave 39's seven vocabulary-gather lines are live code
this wave and are pinned at their indented form, **including
`if mandatory_size > k:`, the structural floor's own refusal**, which wave 45 is
the first wave to build close enough to reach: the mandatory set is 765 rows and
K=1,024 leaves 259 discretionary rows. If that line fires, the refusal is the
result and this wave reports it as one.

Carried and **run**: `mx45_dualband.py` twice (machine-only, then hand-merged),
and `extq.sh` — the ADDENDUM 16.1 extension surface, which `NOTES-MX-AV` §10.6
declined and §12.4 called the cheapest unclaimed item on the board.

Carried and **not run**: `mx45_build.py`, `mx45_vocabtie.py`,
`mx45_vocabrowset.py`, `mx45_termgain.py`, `mx45_plane.py` and
`mx45_xchange.py`, all with their drivers and every pin intact.

**NORM-ESTATE is still not authored, ALIAS is still blocked on the PROBE storage
ruling, and ROWSET Arms A and B are still blocked on a corpus.**
