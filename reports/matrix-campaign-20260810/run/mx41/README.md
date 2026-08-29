# `run/mx41` — how this tree was cut, and what in it is wave 41's

Every file here except `cut_wave41.py`, this README, `mx41_vocabrowset.py` and
`rowsetq.sh` was produced from `run/mx40` by `cut_wave41.py`. **The docstrings
are wave 40's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 40 ran it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 40` | `--campaign-wave 41` | the sealer's wave argument |
| `"campaign_wave": 40` | `"campaign_wave": 41` | the extension surface's own field |
| `wave40` | `wave41` | a path component and a schema infix |
| `mx40` | `mx41` | module names, the build-receipt filename, the log tag |
| `MX40` | `MX41` | the child-directory prefix and the `$MX41` shell variable |

`wave 40`, `Wave 40` and `WAVE 40` **in English are not substituted**, so a
header that records what wave 40 measured still says wave 40 measured it.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx41_verify_seals.py` — the rename turns `wave40/cells` into `wave41/cells`
  and thereby deletes last wave's seal root. One appended line: **36 declared
  roots become 37, `wave40/cells` and `wave39/cells` still declared.**
  `wave40/cells` holds eleven graded children.
* `mx41_retain.py` — an inherited `KEEP` set names last wave's children and
  everything unnamed is deleted. It arrived carrying **wave 40's three keeps and
  eight drops with every justification paragraph intact**; both mappings were
  emptied by hand and the wave root, glob, schema and manifest name retargeted
  by hand.
* `mx41_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 40's
  basis schema appended** (the accept list now carries wave 41, 40, 39, 38, 37,
  36, 35, 34, 33, 32, 31, 30, 29, 23, 22, 21, 20 and 19), **the module
  references retargeted by name**, **both build-receipt write sites** moved to
  `mx41-build-receipt.json`, and the log prefix carried to `[mx41]`.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. **Ninth consecutive cut where this protection refused, and the first
  where wave 40's new pin is what caught it rather than the marker.** The file
  arrived with wave 40's **eleven** seal calls, every count intact, under a
  marker the hand edit would have rewritten anyway; `audit()` counts `^seal
  <TAG>` and requires **zero at cut time**, so all eleven had to go. They were
  deleted, the marker rewritten, and the three path resolutions a verbatim copy
  cannot fix for itself (last wave's `env.sh`, `mx*_percase.py` and
  `seal_verdict.sh`) retargeted by hand at pinned counts of one each.

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
Waves 36 through 40 appended at the cut and paid no extra deployment; wave 41 is
the sixth. It appends **six** strings — the four standard ones, wave 40's tie
schema carried forward (that module is in the tree and not run this wave; a list
completed only where it was needed is a list the next wave has to reason about),
and `neural-forge-matrix-wave41-rowset-vocabulary-child-v1`, which is
`mx41_vocabrowset.py`'s schema.

---

## 2. THE CORPUS AUDIT THAT DECIDED WHICH ROWSET ARMS EXIST, TAKEN BEFORE THE BUILDER WAS WRITTEN

`docs/ROUND3_SLATE_2026-08-28.md` §3.2 specifies ROWSET's Arms A and B as
re-rankings by a **second census over held-out text**, "sealed by hash, disjoint
from the frozen 15 AND the 30-prompt extension bank". Every corpus this estate
holds was checked against that requirement, in its own words, and every one
fails:

| corpus | why it cannot be the second census |
|---|---|
| `lfm2_paraphrase_triplets_v1.json` | its own `purpose`: "three restatements of each sealed case's ask". It is a paraphrase **of the frozen 15**. Selecting on it is selecting on the review surface |
| `lfm2_causal_support_future_manual_review_v1.json` | its own `purpose`: "never support selection, fitting, training, or an automatic numeric gate". Contractually barred — and 2,069 characters besides |
| `lfm2_extension_bank_prompts_v1/v2.json` | the rider bank. The slate voids any rider whose scoring text overlaps it |
| `mapped_donor_calibration.json` (256 rows) and the 24 captured batches | this **is** the evidence census that supplies the primary ranking key. Not held out from itself |
| wikitext / lambada | the benchmark column. Selecting a vocabulary on them contaminates the only external instrument the campaign publishes |

So wave 41 builds the three arms that need **no corpus at all** — `P` (the
sealed tie-break run backwards), `R` (a seeded permutation) and `C` (the slate's
leverage arm, verbatim). **This is a change of ARM registered with its cause
before anything was built, and not a waiver of the class.** Arms A and B remain
open and are re-proposable the moment a corpus exists that passes the test.

The three arms bracket the space of orderings rather than sampling two points of
it: the sealed forward order at one extreme, its exact reverse at the other, and
a seeded permutation between them. If all three read the sealed twin's count at
the same K and the same parameter count, "row identity is inert at fixed K" is
supported by three independent orderings.

---

## 3. Wave 41 ADDS a build module and EDITS exactly one carried one

### 3.1 What is new: `mx41_vocabrowset.py` and `rowsetq.sh`

Sixteen waves have moved **how many** vocabulary rows are kept. Not one has
moved **which**. `mx41_vocab.py`'s own docstring says why that matters — "with a
67,584-position census and a 16,384-row budget, the tie-break decides most of
the table" — and every sealed receipt agrees, admitting 4,633 observed tokens at
every K ever built and choosing the rest by parent id order.

The builder holds the mandatory set, the closure, K, the parameter count, the
weights and the tokenizer writer fixed, and moves only the order the
discretionary candidates are considered in. Each of those is asserted rather
than intended, and `mx41_vocab`'s own functions are **imported** rather than
reimplemented, so the arm and its control cannot drift apart about what the
mandatory set is.

**The identity gate is what makes a null readable.** `--arm SEALED` runs this
file's ranker with the sealed key and asserts the result equals
`mx41_vocab.choose_kept`'s kept set element for element; it runs on every build,
not only on the self-test. Without it a null is indistinguishable from a
re-ranker that quietly did nothing.

**The divergence pre-read runs first and gates nothing.** An arm whose kept set
is the sealed child's *is* the sealed child, and building it spends a frozen
review re-reading a sealed result. `--mode divergence` publishes, per arm, the
symmetric difference against the sealed kept set and the normalised footrule of
the whole candidate order, for arms that are built and arms that are not.

### 3.2 What is edited: `mx41_child_index.py`

`NOTES-MX-AR` §11.6, wave 40's own recommendation. The generator wrote its index
itself **and** printed to stdout and took no `--out`, so the obvious invocation —
redirecting stdout into the file the generator also writes — produced a file that
would not parse. Wave 35 shipped that corruption; wave 40 reproduced it and
caught it only because `NOTES-MX-AO` §9's parse-what-you-write rule made it parse
the artifact in the same command.

Two lines fix it and **neither removes a path**: `--out` names the destination,
and the re-parse moves inside the generator. The stdout summary every earlier
wave read is kept, and `cut_wave41.py --audit` pins all three lines so the fix
cannot be undone by a later cut and the old path cannot be lost by this one.

---

## 4. What this tree carries and does not run

`mx41_vocabtie.py` and `tieq.sh` are wave 40's addition, carried unchanged and
**not run this wave**. Their five load-bearing lines are still pinned. Wave 39
was criticised for asserting lines in a file it never opened; the answer is not
to drop the pins when the file is dormant — it is to say which is which, which
is what this section does.

Wave 41's build list is: **one `mx41_build.py` body** (`--ff 768 --excise 9`,
checked against wave 40's sealed digest), **`mx41_vocab.py` gathers**, **three
`mx41_vocabrowset.py` arms** and **`mx41_termgain.py` gain children**. Wave 36's
composed width × MLP path, wave 37's live-mask repair and wave 38's
descending-order excision with both re-index assignments are therefore live code
this wave.

**NORM-ESTATE runs on the arms that already exist in code**, and that is a
decision with a stated cost. `mx41_termgain.py` implements the slate's Arm D —
the reader-Gram-weighted diagonal — at both granularities: terminal-only (1,024
unknowns, the sealed G-arm, the estate's only repeatable +1) and `--block-gains`
(30,464 unknowns, S6's 29-diagonal, conditioned in depth order with a re-capture
per endpoint). Those are built, at the λ ladder, on the FLOOR child. The slate's
**Arm M (moment-match, zero fitted parameters) and Arm S (27 scalars per
endpoint) are NOT built this wave and need a new solver**; the grid therefore
straddles the wave-29 free-parameter hazard on its hazard side only, and a null
on the D arms does not by itself close the estate. That limitation is named here
rather than discovered in the reading.
