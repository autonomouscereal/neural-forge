# NOTES-MX-L — the block rule breaks, the lane rule holds, and the metric changes underneath the wave

**Wave 8 (MX-L), sealed 2026-08-11. ADDENDUM 6 grading, ADDENDUM 7 receipt
schema v2, and — from partway through — ADDENDUM 8, which Cyril registered
mid-run and which changes what counts as success and what prunes a rung.
Every folded component in this wave stores at or below a true quarter of its
own parent tensor and five of them at or below an eighth. No sub-4x level was
solved, sealed, or spent budget on, and the machine re-checks that bound on
every row rather than trusting the width it asked for.**

Twelve composed children, twelve sealed ADDENDUM 6 verdicts, one M-B
conditioning shot with its own M-A control, and the first benchmark columns the
campaign has ever had.

Three headlines, in the order they arrived:

1. **The wave-7 block-conflict rule is wrong in its strong form.** Both
   falsifying controls it demanded were run and both came back against it.
2. **What actually composes cheaply is a LANE, not a block** — and the proof is
   a matched pair at *identical* net storage that differs by three review cases.
3. **The same-block conflict is not a property of the pairing at all.** It
   disappears when the conv writer is folded deeper, which the conditioning
   shot's own M-A control found before conditioning was even applied.

---

## 1. ADDENDUM 8 arrived mid-wave, and was committed before it was applied

Cyril, at roughly the wave's halfway point: *"[Cyril quote withheld]"*

That is registered as **ADDENDUM 8** in
`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md`, committed as `91ff4351` **before a
single wave-8 child was graded under it**, so the rule cannot be read as having
been fitted to results already on the table. What it changes:

* **Success** = every folded component at 4x+ **and** retained coherence.
  In the ADDENDUM 6 vocabulary that is exactly
  `verdict ∈ {alive-parent-band, alive-degraded, impaired}`, so it needs **no
  new receipt field** and no schema version moves. `impaired` is promoted from
  a failure band to a success band.
* **Pruning** = collapse only. The ≥8/15 rung floor is abolished; a lane stops
  when it goes `dead`.
* **Accuracy** stays sealed as the intelligence-quality diagnostic and the
  tie-break.

The flag is computed from sealed receipts by `run/mxl/mxl_addendum8.py`, which
refuses a hand-supplied success value and re-derives the storage bound per
component rather than trusting the width the recipe asked for.

**Wave 7's `COMP.MA.MXK10` and `COMP.MA.MXK11` are retroactively live rungs**
(both 0.9726×, impaired, coherent). They were pruned by a floor that no longer
exists.

### 1.1 Every child of the wave, ranked the way ADDENDUM 8 ranks them

Deepest first among coherent children; `A8` is the machine-computed success
flag. Wave-7 children are included where they are the matched comparison.

| child | n | net stored | Δ params | worst stored/owned | review | verdict | A8 |
|---|---:|---:|---:|---:|---:|---|---|
| `COMP.MA.MXL11` | 6 | **0.955444×** | −10,234,122 | 0.249853 | 9/15 | alive-degraded | **PASS** |
| `COMP.MA.MXL5` | 4 | **0.972579×** | −6,298,505 | 0.249853 | 10/15 | alive-parent-band | **PASS** |
| `COMP.MA.MXK10` (w7) | 4 | 0.972579× | −6,298,505 | 0.249853 | 7/15 | impaired | **PASS** |
| `COMP.MA.MXK11` (w7) | 4 | 0.972590× | −6,295,818 | 0.249853 | 6/15 | impaired | **PASS** |
| `COMP.MA.MXL6` | 3 | 0.978304× | −4,983,497 | 0.249008 | 9/15 | alive-degraded | **PASS** |
| `COMP.MA.MXK13` (w7) | 3 | 0.979732× | −4,655,497 | 0.249853 | 9/15 | alive-degraded | **PASS** |
| `COMP.MA.MXK5` (w7) | 3 | 0.981152× | −4,329,353 | 0.249853 | 10/15 | alive-parent-band | **PASS** |
| `COMP.MA.MXL1` | 3 | 0.985726× | −3,278,602 | 0.249853 | 10/15 | alive-parent-band | **PASS** |
| `COMP.MA.MXL10` | 3 | 0.985726× | −3,278,602 | 0.249853 | **11/15** | alive-parent-band | **PASS** |
| `COMP.MB.MXL12` | 3 | 0.985726× | −3,278,602 | 0.249853 | 8/15 | alive-degraded | **PASS** |
| `COMP.MA.MXL2` | 3 | 0.990289× | −2,230,538 | 0.249008 | 9/15 | alive-degraded | **PASS** |
| `COMP.MA.MXL4` | 3 | 0.990864× | −2,098,569 | 0.250000 | 10/15 | alive-parent-band | **PASS** |
| `COMP.MA.MXK14` (w7) | 2 | 0.994287× | −1,312,137 | 0.249008 | **13/15** | alive-parent-band | **PASS** |
| `COMP.MA.MXL8` | 13 | 0.895457× | −24,012,812 | 0.249853 | 3/15 | **dead** | fail |
| `COMP.MA.MXL9` | 10 | 0.921164× | −18,108,043 | 0.249853 | 5/15 | **dead** | fail |
| `COMP.MA.MXL7` | 7 | 0.946871× | −12,203,274 | 0.249853 | 9/15 | **dead** | fail |
| `COMP.MA.MXL3` | 3 | 0.990289× | −2,230,538 | 0.249008 | **11/15** | **dead** | fail |

**Twelve of sixteen are ADDENDUM 8 successes.** The `worst stored/owned` column
is the invariant made visible: the largest per-component storage ratio anywhere
in this table is **0.250000** (`C06.T1.S4.L12`, exactly a quarter) and every
other component is at or below 0.249853. **No component in any child of this
wave stores more than a true quarter of the tensor it replaced**, and
`mxl_addendum8.py` re-derives that from the sealed counts rather than trusting
the width the recipe asked for.

The net whole-model ratio is reported separately and is always larger, for the
reason ADDENDUM 7 §7.4 gives: the **67,108,864-parameter tied embedding table is
29.2% of the parent** and no component in the matrix touches it.

---

## 2. The two falsifying controls, and what they did to the rule

`NOTES-MX-K` §2.1 predicted: C05 composes **freely** with folds in *other*
blocks, and stops composing with a fold in *its own* block. It named the two
children that would settle it. Both were run first, before anything was built.

The base for both is `COMP.MA.MXK14` — C05 `conv_out_proj` at a true eighth in
block L05 plus C07 `attn_k` at L12 — which wave 7 sealed at **13/15**.

| child | third component | block relationship | review | Δ vs its own base |
|---|---|---|---:|---:|
| `COMP.MA.MXL1` | C13 `mlp_down` S4 @ **L04** | different block | **10 / 15** | **−3** |
| `COMP.MA.MXL2` | C05 `conv_out_proj` S8 @ **L07** | different block, second conv writer | **9 / 15** | **−4** |

Now put those beside the same-block pairs wave 6 and 7 sealed, all measured
from a 10/15 base (`COMP.MA.MXK6`, C05 at a *quarter*):

| child | third component | block relationship | review | Δ vs its own base |
|---|---|---|---:|---:|
| `COMP.MA.MXK9` | C11 `mlp_gate` S4 @ L05 | **same block** | 7 / 15 | −3 |
| `COMP.MA.MXJ3` | C13 `mlp_down` S4 @ L05 | **same block** | 6 / 15 | −4 |

**The costs are the same: −3 and −4 on both sides.** A third component costs
three or four review cases whether or not it shares a block with the conv
writer. The strong form of the block-conflict rule — "composes freely in a
different block" — is falsified, and it is falsified in the direction that
matters, because it was the *permissive* half of the rule that the wave's
assembly strategy was going to lean on.

C03 and C04 are the componentwise nearest conv-lane neighbours and **both are
cannot-express-4x**, so control (b) used the nearest *expressible* second conv
writer, C05 at another `short_conv` block. No sub-4x fallback was considered at
any point.

### 2.1 What survives, and the matched pair that shows it

One addition in the whole campaign has ever been cheap: `COMP.MA.MXJ2` →
`COMP.MA.MXK5`, where `mlp_gate` joined `mlp_down` **in the same block and the
same lane**, for **−1**. Wave 8 tested that directly and got a controlled
result at *identical* storage:

| child | four folded components | net stored | review | verdict |
|---|---|---:|---:|---|
| `COMP.MA.MXK10` (wave 7) | C11@L05, C13@L05, **C11@L07**, C07@L12 | **0.972579×** | 7 / 15 | impaired |
| `COMP.MA.MXL5` (wave 8) | **C11@L05, C12@L05, C13@L05**, C07@L12 | **0.972579×** | **10 / 15** | **alive-parent-band** |

Same net ratio to six decimals, same component count, same technique, same
level. The only difference is that `MXL5` keeps all three MLP folds **in one
lane of one block** — the entire SwiGLU triple `w1`/`w3`/`w2` of L05 — while
`MXK10` spreads its MLP folds across L05 and L07. **Three review cases, bought
for nothing.** `C12 mlp_up` had never been composed in any wave.

Wave 7 recorded that *"no four-component child at S4 reached 8/15."*
`COMP.MA.MXL5` reaches 10/15.

**The honest limit of the lane reading.** The wave also tested the same-lane
idea on the *other* lane: `COMP.MA.MXL4` puts C06 `attn_q` beside C07 `attn_k`
in block L12 — same operator lane, no lane crossing — and it cost the usual
**−3** (13/15 → 10/15). So "same lane is free" is not a law either. What the
evidence supports is narrower and is stated as such: **the L05 SwiGLU lane
absorbs its three tensors nearly for free; every other addition this campaign
has measured costs three or four cases.** One lane, measured twice.

---

## 3. The coherence frontier under ADDENDUM 8

With pruning changed to collapse-only, the wave stopped walking and started
bracketing: seven components and thirteen components were launched **in
parallel** to bound the frontier in one round rather than approach it one rung
at a time.

| child | folded | what | net stored | Δ params | review | verdict | A8 |
|---|---:|---|---:|---:|---:|---|---|
| `COMP.MA.MXL8` | 13 | SwiGLU lanes L02+L04+L05+L06, C07@L12 | **0.895457×** | −24,012,812 | 3 / 15 | **dead** | fail |
| `COMP.MA.MXL9` | 10 | SwiGLU lanes L04+L05+L06, C07@L12 | 0.921164× | −18,108,043 | 5 / 15 | **dead** | fail |
| `COMP.MA.MXL7` | 7 | SwiGLU lanes L04+L05, C07@L12 | 0.946871× | −12,203,274 | 9 / 15 | **dead** | fail |
| **`COMP.MA.MXL11`** | **6** | **L05 lane + C11/C13@L04 + C07@L12** | **0.955444×** | **−10,234,122** | **9 / 15** | **alive-degraded** | **PASS** |
| `COMP.MA.MXL5` | 4 | L05 SwiGLU lane + C07@L12 | 0.972579× | −6,298,505 | 10 / 15 | alive-parent-band | PASS |

**The frontier sits between 0.955444× and 0.946871×** — six folded components
coherent, seven dead. `COMP.MA.MXL11` is the deepest coherent child either wave
has produced, and its coherence is not marginal: **all sixteen of its lanes
terminate on their own and none rides the 1,024-token cap**, which is a cleaner
signal than several shallower children give.

Every one of the four dead children died the same way, and it is worth naming
because it is the mechanism the clause exists to catch: a construct latches and
rides the cap. `MXL7` nests `def is_ordered(items):` with an identical docstring
until the cap; `MXL9` redefines `unique_in_order` returning `sorted(items)` over
and over; `MXL8` does it in **four separate lanes** at once — `The box with 6
boxes left over is the same as the box with 6 boxes left over` (~40×),
`**Subject:** To: vendor@example.com` (~75×), `Sales volume is calculated by
multiplying the sales volume by the sales volume` (~45×).

### 3.1 The count and the collapse detector measure different things

`COMP.MA.MXL3` is the wave's cleanest demonstration and the reason ADDENDUM 8's
split is right: **11/15 on the count — second best in the wave — and dead on the
clause**, charged on a single arithmetic lane that repeats *"Each box has 17
kits, and there are 5 boxes, so each box has 17 kits"* to the cap. A child can
be second-best on accuracy and still be collapsed.

The clause was applied in **both directions and recorded either way**, as
`NOTES-MX-I` §3.3 requires. Repetition was recorded and **not charged** in
`MXL1`, `MXL2`, `MXL5`, `MXL6` and `MXL11` — duplicated plan steps, a bullet
repeated four times, a two-bullet block cycled six times — because in each case
a structured, progressing response survives its removal.

**One borderline call, flagged so it can be overturned deliberately.**
`COMP.MA.MXL6`'s `writing-concise-email` rides the cap cycling a signature block
(`— Your Email Address` / `— Your Phone Number`), but the complete, correct,
usable email precedes the loop, so under the stated test — *remove the
repetitions and ask whether a response remains* — a response plainly remains and
the clause is not charged. A reviewer who charges cap-riding regardless of what
precedes it would call `MXL6` dead and would take 0.978304× off the frontier.

---

## 4. Benchmarks: the continuous scale ADDENDUM 8 asked for

The 15-prompt band cannot separate two children it calls the same thing, so
wave 8 added sampled `lm-eval` columns from the server's benchmark lab
(`/home/cereal/ai-benchmark-lab`, `envs/lm-eval`, lm_eval 0.4.13.dev0,
transformers 5.12.1, which loads `Lfm2ForCausalLM` natively).

Each child is materialized as a real HF model directory by
`run/mxl/mxl_materialize_child.py`, which **refuses to write a directory it
cannot prove is the sealed child**: the `.pt` file digest is checked against
`composed_child_tensors_sha256` and every folded tensor against the per-tensor
digest the receipt sealed. Materialization is at **float32**, byte-for-byte the
reconstitution `run_lfm2_composition_frozen_review.py` performs at its own
default, and the parent baseline is benchmarked from its own directory at the
same dtype — so the benchmark is of the object the review graded, on both sides.

Pinned and reported with every number: `--limit 200`, `--seed 1234`,
`--batch_size 4`, `max_length=2048`, one PCI-pinned sm86 RTX 3090.

**`max_length=2048` is not a default and is load-bearing.** LFM2.5-230M declares
`max_position_embeddings=128000`; lm-eval sizes the `wikitext` rolling window
from that and asks for a **67.86 GiB** attention allocation, which OOMs a 24 GB
3090 on the first request. 2048 is the window every model in the comparison is
measured at. It changes the absolute perplexity and not the parent-child
comparison.

These are **partial evaluations by construction** (200 documents per task).
They rank models against each other under identical settings and are not
comparable to published full-split numbers.

| model | n | net stored | review | verdict | arc_easy acc | hellaswag acc_n | piqa acc_n | wino acc | lambada acc | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **parent (unfolded)** | 0 | 1.000000 | 12/15 | — | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.3912 | 81.7976 |
| `COMP.MA.MXL11` | 6 | **0.955444×** | 9/15 | alive-degraded | 0.5250 | 0.4600 | 0.6350 | **0.5700** | 0.2250 | 99.2209 | 118.0216 |
| `COMP.MA.MXL5` | 4 | **0.972579×** | 10/15 | alive-parent-band | 0.5100 | 0.4850 | 0.6200 | 0.5500 | 0.3200 | 55.1916 | 105.5257 |
| `COMP.MA.MXL6` | 3 | **0.978304×** | 9/15 | alive-degraded | 0.5150 | 0.4600 | 0.6700 | 0.5400 | 0.3350 | 59.7803 | 112.2877 |
| `COMP.MA.MXL10` | 3 | **0.985726×** | 11/15 | alive-parent-band | 0.5300 | 0.4450 | 0.6050 | **0.5750** | 0.3200 | 63.8849 | 109.0789 |
| `COMP.MA.MXL4` | 3 | **0.990864×** | 10/15 | alive-parent-band | 0.5450 | 0.4650 | 0.6350 | **0.5800** | 0.3700 | 38.6083 | 92.2921 |

Random-chance floors: arc_easy 0.25, hellaswag 0.25, piqa 0.50, winogrande 0.50,
lambada 0.

**Read the first four accuracy columns against the last three.** At
**0.955444×** with six folded components, `COMP.MA.MXL11` is within **0.02** of
the parent on `arc_easy` and `piqa` accuracy, within 0.045 on `hellaswag`
normalized accuracy, and **above** the parent on `winogrande`. On the same
child, `lambada_openai` accuracy falls from 0.4600 to 0.2250 and its perplexity
goes 24.4 → 99.2, with `wikitext` word perplexity 81.8 → 118.0.

Three of the five children score **above** the parent on `winogrande`, which at
0.55 against a 0.50 floor is the task the parent itself barely does — that is
noise around a near-chance baseline and is not evidence of improvement.

**What the benchmark says that the review cannot.** The multiple-choice tasks
barely move while the exact-prediction tasks collapse. That is the signature of
a fold-init: the model still *discriminates* plausible continuations from
implausible ones at near-parent rates, and it has lost the precision to *emit*
the exact next token. Under ADDENDUM 8 that is the interesting half — retrieval
of "intelligence of any kind" is exactly what survives, and it is what
distillation and retraining would have to build on.

---

## 5. The conditioning shot, and the control that stole its result

Step 3 was the one M-B order never tried: install `C05` (S8@L05) **first**, then
re-solve `C13.T8.S4@L05` **conditioned on the C05-folded child**, against the
sealed unconditioned same-block result of **6/15** (`COMP.MA.MXJ3`, wave 6).

The recipe is a deliberate block-exclusivity violation and says so in the file:
`mxl_recipe.py` **refuses** a two-lane block unless `--allow-block-collision
REASON` is passed, and the reason is written into the recipe. (The refusal was
tested negatively before the wave ran: the same cell list without the flag exits
non-zero and writes nothing.)

It was run **twice on the same recipe** — once M-A, once M-B — because a
conditioned arm with no unconditioned arm at the same levels cannot separate
"conditioning helped" from "S8 helped".

| child | mode | C05@L05 level | C13@L05 solved against | review | verdict | wall |
|---|---|---|---|---:|---|---:|
| `COMP.MA.MXJ3` (wave 6) | M-A | **S4** | parent evidence | 6 / 15 | impaired | — |
| `COMP.MA.MXL10` | M-A | **S8** | parent evidence | **11 / 15** | **alive-parent-band** | 67.5 s |
| `COMP.MB.MXL12` | **M-B** | **S8** | **the C05-folded child** | **8 / 15** | alive-degraded | **2,437 s** |

The M-B arm is genuinely conditioned and the receipt proves it: `C13.T8.S4.L05`
and `C07.T5.S4.L12` both seal `conditioned=true` with 1,060 s and 1,333 s of
capture behind them, against a fresh 16 GB conditioned evidence tree keyed on
child identity `d928b94864d10c62` (parent base `96a6779d927a8aca`).

**Conditioning does not cure the same-block conflict. It made this child three
review cases worse than its own unconditioned control**, at **36× the wall
clock** — 2,437 s against 67.5 s, which reproduces the 27× wave 6 measured.

Against the sealed 6/15 the conditioned arm is +2, and that is the number a
wave without a control would have reported as a conditioning win. It is not one.

Install drift moved the *other* way, which is one more reminder that drift does
not predict the review: conditioned `C13@L05` drifts **0.322174** against the
unconditioned **0.335975**, and the conditioned child's global drift is
**0.375996** against **0.385172**. Better drift, worse behaviour.

**The M-A control is the result.** `COMP.MA.MXL10` — the same three components,
same block collision, *unconditioned* — reads **11/15 alive-parent-band** where
wave 6's `COMP.MA.MXJ3` read 6/15. The two differ in exactly one thing: the conv
fold is at a true **eighth** instead of a quarter. **Five review cases, from
folding the conv writer deeper.**

So the same-block conflict `NOTES-MX-K` §2.1 describes is not a property of the
pairing. It is a property of that pairing *at that depth*, and it disappears
when C05 is taken from S4 to S8 — which is the same direction wave 7 already saw
once and did not generalize: `C05@L05` S4→S8 took `MXK6`→`MXK14` from 10/15 to
13/15, and its solve condition number *improved* from 1,199 to 448. Deeper is
better for this component, twice, in two different composition contexts.

Had the control not been run, conditioning would have been credited with a
five-case gain it did not produce.

---

## 6. What this wave did not settle

* **Why the L05 SwiGLU lane is cheap and the L12 attention lane is not.** Two
  lanes, one measurement each. The mechanism is not established, only the
  contrast.
* **Whether C12 `mlp_up` helps anywhere else.** It was composed for the first
  time in this wave, at exactly one site.
* **The frontier between six and seven components is one child wide.** No
  intermediate was run; the wave stopped at the bracket.
* **The S8 arms of the deep children were never built.** `COMP.MA.MXL6` shows
  two MLP folds at an eighth surviving at 0.978304×, but no SwiGLU *lane* was
  taken to S8, and that is the obvious next move for depth: three tensors at an
  eighth is 0.9700× for one block against 0.9726× for a quarter-folded lane.
* **`C09 attn_o`, `C08 attn_v` were never composed**, for the fourth wave
  running.
* **The 15-prompt review is one reviewer, one pass**, ADDENDUM 2's ±2 interval
  applies, and under ADDENDUM 8 the counts are a diagnostic — the coherence call
  is what the frontier rests on, and it is a judgement about text.

---

## 7. Ledger, receipts, disk, provenance

| what | where |
|---|---|
| ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.pre-mxl-20260811T203806` (`f5994b53…`) |
| ledger before the merge | `…/backup/ledger.jsonl.before-mxl-merge-20260811T210142` (`f5994b53…`) |
| `LEDGER.md` before the render | `…/backup/LEDGER.md.before-mxl-render-20260811T210142` |

**The ledger.** 378 → **402 rows (+24)**: 12 composed-child instances and 12
review instances. `sha ebd6d76466b0ccd0…`, **402 distinct
`(cell_instance_id, receipt_sha)` pairs**, 402 distinct receipt digests, **24 of
24 wave-8 rows on `sm86 NVIDIA GeForce RTX 3090`** and **24 of 24 at
`receipt_schema_version` 2**. The merger ran **exactly once** over all 24 shards
and its refuse-unless-prefix check passed byte for byte.

**Seals.** 398 of 402 ledger rows resolve to a receipt on disk and **all 398
verify; 0 fail**. All **24 of 24** wave-8 seals verify. 337 receipts satisfy
today's mandatory field set and **61 wave-1 receipts do not**, for the documented
reason — they predate `code_identity.compute_architecture` — the same 61 waves 4
through 7 reported.

**Four ledger rows do not resolve, and none is wave 8's**: the same three wave-1
`CAL.C01.T5.X4` calibration rows and the one wave-5 `C06.T2.S4.L12` review
instance that is also the ledger's only duplicated `cell_instance_id`. Wave 8
found them by checking, as waves 6 and 7 did, and did not touch them.

**Mirror.** `reports/matrix-campaign-20260810/ledger.jsonl` and `LEDGER.md` are
**byte-identical** to the server authority:

```
ebd6d76466b0ccd0bef239c46fa74d55bec5bbc2a727504f09a142499212ad16  ledger.jsonl
1e4b86dd38ff4303dfe24f81ce949264bcf50321eb70476ee456db94649c9a9a  LEDGER.md
```

**Disk.** Root `/` was 7.3 GB free at the start of the wave and 6.8 GB at its
low point; every heavy artifact — twelve composed children, nine new site
evidence caches, the **16 GB** M-B conditioned capture tree, seven materialized
float32 HF child directories at ~920 MB each, and every benchmark output — was
written under `/mnt/ai_data/neural-forge-matrix/wave8/`. Each materialized
child's `model.safetensors` was deleted immediately after its benchmark ran; the
directory keeps its `materialization.json`, which carries the digest chain
needed to re-derive it.

**No new deployment, and that is the stronger claim.** Wave 8 changes no source
file, so it reuses wave 7's `neural-forge-matrix-17377c16` after re-verifying
that **all 605 of its Python modules are byte-identical to the git blobs of this
wave's base commit `5898d8d0`** (whose own diff against `17377c16` touches
`reports/` only). The deployed surface therefore *is* this wave's committed
surface, without a second `git archive` and without re-inflicting the re-lining
of 34 non-Python assets that wave 7 had to repair by hand.

40 non-Python assets in the deployment still differ from their git blobs — the
same re-lining — and **none of them is a review-surface file**. The three that
matter were checked by name and by digest and all three verify byte-identical to
the blobs at `5898d8d0`:

```
e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b  V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md
7b859172fa48c008eb68d5ea070458ac82e2149c19c5141230d45c061aaf00e5  PROTOCOL_SEAL.json
0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b  lfm2_semantic_whole_response_prompts_v1.json
```

`run-mxl-review.sh` re-checks the protocol and pack digests on **every single
review** and refuses the run if either has moved.

**The knob file is byte-identical to wave 7's** (`e355e4e7bb8a3058…`), so a
wave-8 rung and a wave-7 rung differ in the recipe and nothing else.

**The parent control is byte-identical across both waves.** The sixteen parent
responses in every wave-7 and wave-8 dossier digest to the same
`865f006e3813712e…`, checked by `run/mxl/mxl_parent_identity.py` on every
dossier the wave sealed. Its calibration reproduced exactly before any child was
graded: **5 useful / 7 partial / 3 observed miss = 12/15**, misses at
`arithmetic-multi-step`, `logic-ordering`, `multi-turn-correction`.

**The technique surface is inert to the reuse.** The first wave-8 child re-solved
`C05.T8.S8.L05` and reproduced wave 7 to the digit — install drift
**0.223583**, rank 127, gram 0.44067, cos 0.82669 — and `C07.T5.S4.L12` kept
**121 of 512** exactly as sealed. `COMP.MA.MXK14`'s and `COMP.MA.MXK5`'s seals
were re-verified before anything was built on them, and every one of their
composed tensors re-digested to the value its receipt sealed.

Every verdict in this wave was computed by `behavior_verdict_for` from counts
sealed through `seal_matrix_behavior_review.py`. No verdict was stated by hand,
and the ADDENDUM 8 success flag is likewise computed rather than asserted.
