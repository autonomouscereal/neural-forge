# NOTES-MX-AI — wave 33: the instrument never broke, and the frontier is thinner than the scorecard says

Wave 33 was sent to repair an instrument, do forensics on a composition failure,
and try two untried arms at the robust frontier. It moved **no frontier row at
any of the fifteen thresholds**: the deepest stored ratio at every one of the
fifteen is the number wave 32's table already carried, and wave 32 moved none
either. That is the second headline rather than the first.

*(The seal commit for this note says "the sixth consecutive wave that has not."
That count is not supported by the record — only wave 32's note states it
explicitly for itself — and the sentence above is the claim this file stands
behind. The commit message is immutable and the correction lives here, which is
the campaign's standing practice for a number that moved after it was written.)*

The first is that two of this wave's measurements say the campaign has been
reading two of its own numbers wrong.

1. **The extension surface never broke.** It is reproducible to the byte across
   a wave boundary, its stable value is ρ ≈ 0.69 over sixty observations, and
   the per-wave ρ the campaign has been publishing has a sampling interval
   **wider than the gap between the scorer's own three verdict thresholds.**
   Wave 32's `SCRAMBLED` was the 8th percentile of its own null; wave 31's
   `VALIDATED` was the 89th. Neither is anomalous. What the surface actually
   cannot do is the job ADDENDUM 16.1 gave it.
2. **The sealed fifteen overstate the frontier by a factor of three to five**,
   and the paraphrase surface now says so with a number. `ARCH.B75` holds
   eleven sealed cases and **two** of them survive a restatement of their own
   ask. `ARCH.CL1792M` holds eight and **none** do.

---

## 0. What this wave's own discipline caught, stated before the results

**The first pass at grading `ARCH.GV12L10` was wrong by three cases and the
check that caught it is new.** Every child in this wave is a terminal-gain
post-process: 1,024 values overwritten in place at update scale 0.022–0.059,
every other byte the source's. So lanes decode to text that is *byte-identical*
to the source's own sealed dossier — five of sixteen on `GV12L10`. The first
reading graded `conversation-natural-greeting`, `debugging-off-by-one` and
`writing-concise-email` as **misses** on strings wave 31 sealed **Partial**,
byte for byte. That is grader drift, not a result, and uncorrected it would have
published a three-case loss that did not happen.

A per-lane byte comparison against each source was run *before* any seal was
written; where the bytes agree the sealed verdict is carried, and where they
differ the lane is read. **This wave recommends that comparison become standing
practice for every post-process arm**, because a post-process is exactly the
build class where a grader's memory is least reliable and the check is free.

Two other things are stated rather than buried: **I1 was run before this wave's
pre-registration existed**, which the pre-registration says in its own last key
along with why that is not the wave-32 defect repeated (I1 reads only committed
artifacts, opens no dossier and produces no count any frontier claim rides on;
its three hypotheses were named in the dispatch). And **`mx33_condinventory.py`
under-reported its own coverage by forty-eight cells on its first run**, because
its marker list carried wave 30's and wave 32's phrasings and not waves 16 and
17's plain `conditioned capture`. The record was complete; the reader was not.
Fixed, and the count is in §4.

---

## 1. I1 — the extension surface, bounded rather than retired

Sixty child observations from four committed `waveNN/extension-score.json`
files, one frozen bank (`c9d37004c901d757…`), one frozen rubric. Nothing decoded.

**The free control first.** `V12CL1792` and `V16CL1792` were decoded on this bank
in wave 31 *and* wave 32, from checkpoints retained across the boundary. Both
score identically and hold the **identical prompt set** both times. The
instrument is reproducible, so any movement in ρ is movement in the population.

**Hypothesis (a), the band, is falsified.** ρ restricted to sealed ≥ cut:

| cut | 0 | 2 | 4 | 5 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ρ at or above | 0.6926 | 0.7184 | 0.7591 | 0.7058 | 0.6066 | 0.7306 | 0.6682 |
| ρ below | — | 0.8402 | 0.4528 | 0.2461 | 0.2081 | 0.5530 | 0.5878 |

There is no band where this is a good instrument and none where it collapses.

**What replaces the per-wave ρ is a resolution curve.** Over all 1,618 ordered
pairs, by sealed gap:

| sealed gap | 1 | 2 | 3 | **4** | 5 | 6 | 8 | ≥10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agree | 48.1% | 62.2% | 67.9% | **81.6%** | 89.2% | 86.5% | 92.1% | **100%** |
| **invert** | **34.5%** | 24.3% | 18.5% | 10.9% | 6.4% | 7.8% | 5.3% | 0% |

> **The bounded-validity statement.** The extension surface orders two children
> correctly ≥80% of the time when their sealed counts differ by **four or
> more**, and it is noise at one — 48.1% agreement with a 34.5% inversion rate.
> ADDENDUM 16.1 built it "for pricing one-case margins." **One case is exactly
> the resolution it does not have.** It is not retired; it is restricted to
> gap ≥ 4, where it has never disagreed with anything that mattered.

**And the per-wave verdict string is retired.** A null that holds the instrument
fixed at its behaviour on the other three waves and varies only the population:

| wave | observed ρ | null p05 | null median | null p95 | quantile | anomalous at 5%? |
|---|---:|---:|---:|---:|---:|---|
| 30 | 0.6544 | 0.1375 | 0.7914 | 0.9571 | 0.36 | no |
| 31 | 0.7411 | 0.2106 | 0.5531 | 0.7865 | 0.89 | no |
| 32 | **0.3537** | 0.3119 | 0.5758 | 0.7885 | **0.08** | **no** |

The 90% interval on wave 32's own population is **0.31 to 0.79**. The scorer's
thresholds are 0.70 and 0.40. **Both live inside the noise.** Wave 29 is not
estimable — its population reaches sealed 14 and no other wave has a child within
one case of that to draw a conditional from, which is itself the range-restriction
story in one sentence.

Wave 31 and wave 32 have nearly identical spread (mean pair gap 2.86 vs 2.79,
pairs ≥4 apart 33.8% vs 31.1%), so range restriction alone does *not* explain the
0.39 drop; the null does, and leave-one-out says where the rest of it sits:
dropping `ARCH.V4CL1792` alone — sealed 1/15, extension 10/30 — takes wave 32
from 0.3537 to **0.5897**.

**Hypothesis (b), the rubric, is a real defect that is not the cause.**

* **Eight of thirty entries were never held by any of sixty children.**
* **Six more correlate negatively** with the sealed count. The worst,
  `ext-conversation-competing-promises` (ρ −0.2841, held by 39 of 60), has an
  **empty requirement list**: it grades formatting and length and nothing else.
* **Six can be satisfied by quoting the prompt back**, found by running the
  wave-32 rubric against its own bank. Those six average pooled ρ **0.0781**
  against **0.2455** for the other twenty-four. `ARCH.V12CL1792` holds
  `ext-logic-four-seats` on a response that never produces a seating order.
* Two whole families — `context` and `explanation` — are dead: neither prompt in
  either was ever held.

**And repairing it buys nothing.** A trim selected on waves 29–31 only (17 of 30
entries kept) and evaluated on wave 32 moves its ρ from 0.3537 to **0.3577**.

**The degeneracy gate wave 32 asked for was built, priced, and does not catch the
child that motivated it.** `score_one` computes five charged kinds and vetoes a
hold on two of them, so `ARCH.BM50N` — 0/15, sixteen of sixteen lanes charged —
**held three entries**, on a 1,024-token repetition of the word "not", a
1,024-token repetition of an apostrophe, and 75 repetitions of "to". Closing the
veto correctly zeroes both 0/15 children and moves pooled ρ **0.6926 → 0.6802**.
`ARCH.V4CL1792` still scores **10/30**: its ten held responses are fluent,
non-degenerate and wrong.

**The hand audit, hypothesis (b) at full resolution.** Five dossiers, 39 held
prompts read by hand against the sealed standard:

| child | sealed | extension | held prompts defensible by hand |
|---|---:|---:|---:|
| `V12CL1792` | 7 | 9 | **2** |
| `CL1792S` | 4 | 8 | **1** |
| `V16CL1792M` | 3 | 9 | **0** |
| `V4CL1792` | 1 | 10 | **0** |
| `BM50N` | 0 | 3 | **0** |

**Three of thirty-nine.** The true-positive count varies by 2 across a seven-case
sealed range while the false-positive count varies by 7 to 10 between children.
Signal 2, noise 10. That is the whole finding in two numbers. (The hand grading
is mine and is a judgement; the counts above are the strict reading, and the
per-prompt text is packaged so somebody can disagree with each one.)

---

## 2. The rubric wave 32 could not write

`mx33_pararubric.py` is keyed by **lane**, because the fifteen paraphrase lanes
are one-to-one with the sealed fifteen cases. Three defects above are fixed by
construction: the degeneracy veto is closed; every rule names something the
answer must contain; and **every rule is run against its own three prompts on
every invocation**, refusing if any prompt satisfies its own rule. That guard
fires on 6 of the 30 extension entries, which is how it was validated.

**Two lanes are declared UNANSWERABLE and are not scored.** All six restatements
of `context-retrieval` and `summarization-priorities` say "the note below" /
"the update" and **carry neither**; the sealed prompts supply both inline. That
is a defect in the bank, six of forty-five prompts, and it is charged to the bank
rather than to the children. Wave 32's hand reading of this surface was made
across those six.

**The rubric is calibrated on the parent, and the calibration cost two rule
holes.** The first run credited the parent on `logic-ordering` and
`multi-turn-correction`, both sealed **misses** — the ordering rule was satisfied
by restating the constraints, and the snacks rule was satisfied by a list whose
first option is "Peanut Butter & Veggie Pudding" for two severe peanut allergies.
Both were closed from the reading standard, not from the number: Sal must be
named last, and a revised option list may not contain peanut butter. After the
fix the parent reads **12/39 restatements, 6 of its 10 scorable sealed lanes
robust**, against children at 2–7/39. Two lanes still credit the parent where the
sealed record misses (`comparison-ram-rom`, `logic-ordering`, one restatement
each) and that residual is stated rather than tuned away.

---

## 3. F1 — the M-composition mechanism, or rather its three falsifications

The fact:

| | body alone | K=16,384 | 12,288 | 9,216 | 8,192 |
|---|---:|---:|---:|---:|---:|
| `ARCH.CL1792` (stream-selected) | 7 | **8** | **7** | **6** | **5** |
| `ARCH.CL1792M` (reader-metric) | **8** | 3 | 4 | 2 | 3 |

**The lane reading rewrites wave 32's shape.** At every one of the four rungs the
M family's held set is a **strict subset** of the plain family's — `metric_only`
is empty four times out of four. No lane is ever traded; lanes are only lost. And
the loss is **a step, not a slope**: the M body loses the same five lanes from
its own body count at 16,384 and 12,288, six at 9,216, five at 8,192 —
essentially constant — while the plain body loses 0, 1, 2, 3 as K falls. Wave 32
read the shrinking delta as "the penalty shrinks and the two bodies converge."
**The M penalty does not shrink. The plain body descends toward it.**

The five lanes are the same five every time: `conversation-natural-greeting`,
`creative-continuity`, `debugging-off-by-one`, `planning-dependencies`,
`uncertainty-insufficient-data`.

**Candidate 1 — the M selection keeps channels the retokenization disturbs.
FALSIFIED.** The parent was run over the sealed corpus twice, once on its own
segmentation and once on the K = 16,384 child's segmentation injected back into
the parent's id space, capturing the SwiGLU intermediate at all fourteen
feed-forward sites. Median retokenization sensitivity: shared channels 0.2720,
plain-unique **0.3377**, metric-unique **0.2981** — the M-unique channels are
**0.883×**, marginally *less* exposed than the plain-unique ones. (Pre-registered as "not supported, ratio
in [0.90, 1.10]"; observed 0.883, so the prediction was right in direction and
slightly wrong in interval, in the direction that strengthens the falsification.)

**Candidate 2 — the M selection prefers logit-heavy channels whose leverage the
vocabulary fold destroys. FALSIFIED, in closed form.** With
`a_j = w2[:,j]ᵀ HᵀH w2[:,j]` (full 65,536-row head), `b_j` the same through the
16,384 kept rows and `s_j = ‖w2[:,j]‖²`:

| group | n | logit-per-stream (median) | leverage surviving restriction |
|---|---:|---:|---:|
| shared | 23,015 | 52.4705 | 0.2590 |
| plain-only | 2,073 | 52.8392 | 0.2562 |
| **metric-only** | 2,073 | **53.1025** | **0.2585** |

A 0.5% difference in logit weighting and a **three-decimal-identical** restriction
ratio. The M-unique channels are not logit-heavy and their leverage is not more
damaged by the gather.

**Candidate 3 — the metric was built against the wrong reader. FALSIFIED BY A
CONTROL THE CAMPAIGN ALREADY BUILT AND NEVER READ THIS WAY.**
`mx33_build.reader_metric`'s own docstring, written in wave 31, states the
mechanism outright: *"the metric is built from the PARENT's 65,536-row head, and
a composed child cannot emit 46,080 of those rows at all… the selection is
optimised against the reader of a model that will not exist."* Wave 31 shipped
the fix — `emittable`, restricting the probabilities to the composed child's own
keep-set — and built `ARCH.CL1792MV` with `emittable_rows: 19454` taken from
`MX27-V19`. Composed at exactly that K:

| body | selection metric | alone | at K = 19,456 |
|---|---|---:|---:|
| `CL1792` | stream | 7 | **7** |
| `CL1792M` | parent's full head | 8 | 6 |
| `CL1792ME` | head, recensused | 2 | 3 |
| `CL1792MV` | **the composition's own kept rows** | 7 | **5** |

**The matched fix composes two below the stream body and one below the unfixed
metric body.** The metric is not the variable.

> **The mechanism, as far as three measurements can put it.** What fails to
> compose is *being reader-metric-selected at all*, not any measurable property
> of the channels the metric picks and not the identity of the reader it is
> picked for. Three reader-metric bodies with three different metrics compose at
> 6, 3 and 5 against the stream body's 7, at one K, at byte-identical storage.
>
> **So the answer to the question this arm was sent with — can any future body
> improvement be expected to compose? — is: only if it is obtained in a metric
> the second fold does not touch.** The stream selection composes; every reader
> selection tried does not. The +1 case a reader metric buys standalone is
> borrowed from the composition, and the paraphrase surface says the loan is
> worse than that: `ARCH.CL1792M`'s eight sealed cases include **zero** that
> survive a restatement, the only child in the campaign to read robust 0.

---

## 4. A2 — the conditioning inventory, and a prediction that was wrong

220 cell receipts read, three independent witnesses each: the builder's
`evidence.conditioned` boolean, the axis prose, and `max_input_drift`, which is
**exactly 0.0** on a stale capture and 0.17–0.27 on a conditioned one.

* **101 solve-bearing cells, 119 zero-solve. 101 of 101 solve-bearing rows are
  classified by at least one witness. Zero witness disagreements.**
* **Exactly three solve-bearing rows carry a stale axis**, and all three are
  declared controls: `ARCH.CL1792S` (wave 32, 4/15 — B4's own control),
  `ARCH.W768ST` (wave 20, 2/15), `ARCH.S640` (wave 15, 0/15).
* **No frontier-adjacent row is stale.** Every row above 4/15 is conditioned,
  including all five of the campaign's best solve-bearing children
  (`X13A` 13/15, `X07` 11, `X09` 10, `X01` 9, `X11` 9).

**The pre-registered prediction was "zero other than `ARCH.CL1792S`" and it was
wrong: there are three.** It was right in substance — there is nothing to
rebuild, and wave 32's three-case dividend is already banked everywhere it could
pay. The two rows it missed are old stale controls whose names say so
(`W768ST`, `S640`) and which the inventory found without being told to look.

---

## 5. A1 — the terminal-gain ladder at the robust frontier

Six children, two sources, `--lambda-relative 1.0 / 0.3 / 0.1`. Zero storage
delta: 1,024 values overwritten in place, `config.json` untouched.

**The gate question wave 27 left open is closed.** The terminal-gain identity
gate refuses a *width* fold because that projects the embedding; a **vocabulary**
gather keeps its rows verbatim, so both sources passed and the arm is available
on the whole vocabulary axis. Nobody had tried.

| cell | λ | update | source | held-out relSSE | **sealed** | predicted |
|---|---:|---:|---|---|---:|---:|
| `ARCH.GV12L10` | 1.0 | 0.0220 | `V12CL1792` 7/15 | 0.1139 → 0.1118 | **7** | 7 ✓ |
| `ARCH.GV12L03` | 0.3 | 0.0416 | `V12CL1792` 7/15 | 0.1139 → 0.1113 | **7** | 7 ✓ |
| `ARCH.GV12L01` | 0.1 | 0.0585 | `V12CL1792` 7/15 | 0.1139 → **0.1111** | **6** | 7 ✗ |
| `ARCH.GV16L10` | 1.0 | 0.0222 | `V16CL1792` 8/15 | 0.1224 → 0.1199 | **7** | 8 ✗ |
| `ARCH.GV16L03` | 0.3 | 0.0419 | `V16CL1792` 8/15 | 0.1224 → 0.1194 | **7** | 8 ✗ |
| `ARCH.GV16L01` | 0.1 | 0.0588 | `V16CL1792` 8/15 | 0.1224 → **0.1192** | **8** | 7 ✗ |

**Six rungs, two sources, zero gains.** Two exact predictions, four off by one,
mean signed error +0.33 — a large improvement on wave 32's −2.25, and the
allocator is still leaning optimistic.

Three things inside it are worth more than the counts.

* **The objective and the review disagree, again.** The held-out logit relSSE
  improves monotonically with falling λ on both sources, and the review is flat,
  flat, down on one and down, down, level on the other. This is the fifth wave to
  record it.
* **`GV16L10` and `GV16L03` hold the identical seven lanes** a factor of three
  apart in λ. Two rungs landing on the same seven lanes is the strongest evidence
  in this ladder that the terminal diagonal is not what decides these cases.
* **The one rung that ties its source ties it on a case that is already
  retired.** `GV16L01` reads 8/15, equal to `ARCH.V16CL1792`, holding the same
  eight lanes — and the eighth is `creative-continuity`, the exact lane wave 32's
  paraphrase triplets retired on that source. Its **robust** count is seven,
  which `ARCH.V12CL1792` already holds 0.018 of stored ratio deeper. **No
  triplets were run on the six**, because the standing rule is triplets on any
  gain and there is no gain.

---

## 6. A3 — the Cyril-proofed frontier

Four more children decoded on the paraphrase bank (`f9ac00e6dd5212f0…`, 45
prompts), all ten scored with the repaired rubric. **Robust** = sealed-held lanes
that still answer under at least one of three restatements of that lane's own
ask; two lanes are unscorable for everyone, so the denominator is 13 and the
parent is the ceiling.

| child | ratio | sealed 15 | restatements /39 | **ROBUST** | predicted restat |
|---|---:|---:|---:|---:|---:|
| **PARENT** | 1.000000 | 12 | **12** | **6** | — |
| `ARCH.B75` | 0.742049 | 11 | **7** | **2** | 7 ✓ |
| `ARCH.V19X13G03` | 0.742049 | 11 | **6** | **2** | 7 |
| `ARCH.V18X13` | 0.737484 | 10 | **5** | **3** | 6 |
| `ARCH.V16CL1792` | 0.637073 | 8 | 3 | **2** | — |
| `ARCH.CL1792M` | 0.856199 | 8 | 2 | **0** | 4 |
| `ARCH.V12CL1792` | 0.618813 | 7 | 2 | **1** | — |
| `ARCH.V9CL1792` | 0.605117 | 6 | 2 | **1** | — |
| `ARCH.V8CL1792` | 0.600552 | 5 | 3 | **2** | — |
| `ARCH.V6CL1792` | 0.591422 | 3 | 2 | **1** | — |

**The sealed ordering does not survive.** `V18X13` at sealed 10 is more robust
than `B75` at sealed 11, and it is **deeper** — 0.737484x against 0.742049x.
`V8CL1792` at sealed **5** is as robust as `B75` at sealed **11**. And `ARCH.CL1792M`, the reader-metric body, is the only child in
the campaign to read **robust 0** — not one of its eight sealed cases survives a
single restatement of its own ask, which is §3's mechanism arriving from a
completely different direction.

The parent holds 6, so the children's 0–3 is not an impossible rubric. Spearman
between sealed count and restatements held is **0.8331** over ten (0.7671 without
the parent) — on a *narrower* population than the extension surface's sixty, at
which that surface reads 0.6926. **The paraphrase surface is the better
instrument and it costs 45 prompts per child instead of 30.**

---

## 7. Benchmarks

`limit=200 seed=1234 batch=4 max_length=2048 float32`, sm86. The parent row
reproduces the sealed reference **exactly** — 0.5450 / 0.5050 / 0.6800 / 0.5500 /
0.4600, ppl 24.39, wiki 81.80 — the sixth calibration. No child in this wave is
a looping child (three cap hits of sixteen on `GV12L01`, zero to one on the
others), so all six are benchmarked.

| model | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 | 12/15 |
| `GV16L01` | 0.4050 | 0.3950 | 0.5450 | 0.5250 | 0.2000 | 500.95 | **519.32** | **8/15** |
| `GV16L03` | 0.4000 | 0.4000 | 0.5450 | 0.5200 | 0.2000 | 490.28 | 520.03 | 7/15 |
| `GV16L10` | 0.4000 | 0.4050 | 0.5500 | 0.5200 | 0.1950 | 479.09 | 522.31 | 7/15 |
| `GV12L03` | 0.3700 | 0.3750 | 0.5350 | 0.5350 | 0.1900 | 568.44 | 762.82 | 7/15 |
| `GV12L10` | 0.3750 | 0.3850 | 0.5350 | 0.5350 | 0.1900 | 556.42 | 767.06 | 7/15 |
| `GV12L01` | 0.3650 | 0.3750 | 0.5350 | 0.5250 | 0.2000 | 580.50 | **761.11** | **6/15** |

**The inversion's sign flips between the two families.** In the `GV16` family the
lowest wiki perplexity carries the highest count; in the `GV12` family the lowest
wiki perplexity carries the *lowest*. Both families sit inside a 3-point and a
6-point wppl band while spanning a one-case difference. Twelfth wave.

---

## 8. What was ordered and not run

* **No extension decode.** §1 bounds that surface to sealed gaps ≥ 4, and this
  wave's six children are all within one case of their sources and of each other.
  Running it would have produced a number the wave's own instrument section says
  cannot be read at that resolution. The GPU went into the paraphrase decode
  instead, which §6 shows is the better instrument on the same children.
* **No paraphrase triplets on the six termgain children**, because the standing
  rule is triplets on any gain and there is no gain (§5).
* **The extension rubric is diagnosed and NOT repaired.** The degeneracy veto and
  the anti-echo guard are both implemented and validated — in
  `mx33_pararubric.py`, on the paraphrase surface. Retro-fitting them onto the
  extension rubric would rewrite the bank's frozen scoring after seeing four
  waves of its output, and the campaign's own rule is that the bank and its
  rubric were sealed in one commit before any decode. The extension surface keeps
  its rubric and loses its verdict string.

---

## 9. Estate

**Nine commits, five deployments**, and this paragraph is the ninth's whole
content because wave 32 had to make the same correction and the precedent is to
make it rather than to let the note and the log disagree. The tree cut and I1;
two guard refusals answered (the axis-schema crash and the logit-leverage arm);
the inventory's coverage fix; the seal calls; the wave seal with this note; the
scorecard refresh with two artifact corrections; the two closing refusals; the
withdrawal of an unsupported count; and this. Five carried run-path code or
shipped artifacts and each became a deployment: `da2f9e73`, `24366f3d`, `e002181e`,
`15ef17ef`, `c3e00fe4`. **Every one verified file by file against the commit's own
blob digests: 0 content mismatches, 0 missing, 0 undeclared, 0 hardlinks**, five
times — 6,179 files on the first four and 6,257 on the last. The builds and seals
ran under `da2f9e73` and `15ef17ef`; the two analysis arms under `e002181e`. The
sealer's four wave-33 schema strings shipped **in the first commit, before the
first build**.

**The post-wave re-verification refused twice and was right twice.** First on
`c3e00fe4` with one undeclared file: `--report` had been pointed at the
deployment root, so the verification wrote itself into the tree it was verifying
and the next run saw a file the commit does not declare. Reports moved out to the
wave directory. Then on `15ef17ef` with **124 undeclared files, 0 content
mismatches, 0 missing, 0 hardlinks — every one a `__pycache__/*.pyc`.** That is a
standing property of this protocol nobody had written down: `PYTHONPATH=$NF_DEPLOY/src`
makes CPython write bytecode into the deployment, so **a deployment that has been
RUN cannot pass its own completeness check until the caches are removed.** They
were removed and both survivors re-verify clean at 6,257 and 6,179 files with the
four zeros. Neither refusal was a corrupted byte; both were the verifier doing
exactly what it exists for.

**The tree cut.** `run/mx33/cut_wave33.py` is wave 32's with three changes.
`sealq.sh` is in `HAND_EDIT` for the first time and **the protection worked on
its first exercise**: the file arrived verbatim carrying wave 32's marker and
wave 32's fifteen seal calls, `--audit` refused on the marker before anything
ran, and the fifteen were deleted by hand. `gainq.sh` is **authored, not
carried** — waves 30, 31 and 32 each carried `mx*_termgain.py` through the cut
and lost the queue that drives it, so the solver sat in three trees with nothing
able to call it. `ALLOWED_HISTORICAL` starts with the build header and gained one
pinned entry when the audit fired on the paraphrase rubric's docstring. **28
declared seal roots became 29** with `wave31/cells` and `wave32/cells` still
declared.

**Ledger: 908 → 920, 12 rows appended, 0 superseded, in ONE PASS** — both shard
families in one invocation, which is the fix for wave 32's two-pass defect.
Backup `backup/ledger-pre-wave33.jsonl` at the pre-merge digest
`48fb010284bdec15…`, `cmp` identical; the authority's pre-wave 1,553,089 bytes
are a byte prefix of the 1,574,624-byte result. New digest
`c9f6cbe5971dd122…`.

**AND THE REPOSITORY MIRROR WAS TWO WAVES STALE.** `reports/matrix-campaign-
20260810/ledger.jsonl` stood at **878 rows / `85d5f12674169787…`** — wave 31's
state. Wave 32 merged 30 rows into the authority, wrote both merge reports, and
**never copied the result back**; its seal commit touches the merge reports and
not the ledger. This is *the same defect wave 26 recorded against wave 25*, in
those words: "the merge ran, the report was written, the copy-back did not
happen… the lesson is that merge the ledger is two steps and only one of them is
on the server." Fixed here for both waves at once: the repo mirror is now 920
rows at `c9f6cbe5971dd122…`, its 878-row predecessor verified as a **byte
prefix** of the result before the write, and backed up to
`backup/ledger.jsonl.pre-wave33-repo-sync-20260828`.

**Seal verification: 29 roots audited against disk, 0 undeclared; 920 rows, 916
resolve and all 916 verify, 0 failures**, with the same four unresolved rows
waves 25–33 carry (`CAL.C01.T5.X4` ×3, `COMP.MC.MXO0`).

`sealq.sh` ran **once**, behind `.SEALQ_RAN`; the eight-argument guard fired on
none of the six. The ledger merge ran once behind `.MERGED_ONCE`. Parent lane
calibration re-read blind from `ARCH.GV12L10`'s own dossier reproduces the sealed
**5 Useful / 7 Partial / 3 miss = 12/15** exactly, including the sealed parent
arithmetic defect — sixth consecutive calibration, and the benchmark row
reproduces on the same wave.

Both sm86 3090s ran the wave: GPU 0 took the terminal-gain builds, the six sealed
reviews and three benchmark rows; GPU 4 took the paraphrase decode and three
benchmark rows. Both H3 containers were **queue-empty** (`queue_running` and
`queue_pending` both `[]` on ports 28191 and 28195) and idle-holding at 256 MiB
each; wave 31 established by restarting them that this is the free-standing floor
rather than a leak, so this wave read that result and did not re-run a settled
experiment. `/mnt/ai_data` at **1.7 TB free, 64% used, unchanged across the
wave**; both cards free-standing at 288 and 279 MiB when it closed.

Retention ran **after** the package, `--plan` read before `--apply`: **two
children kept (0.54 GiB), four deleted (1.08 GiB)**. The two keeps are
`MX33-GV16L01`, the only child in the wave to reach eight and the only one that
ties its source, and `MX33-GV12L01`, the only rung of six that moved at all —
and it moved down, by a cap hit rather than by content, which is the claim its
bytes carry. The four drops are each one `gainq.sh` invocation from a retained
source (`MX31-V12CL1792`, `MX31-V16CL1792`), with dossier, receipt, per-case
verdicts and build receipt all packaged first. Deployments pruned from **seven to
two** (current + the one `sealq.sh` ran under, so every wave-33 receipt's
`repo-head` still resolves), and the prune manifest refuses to run if any
deployment on disk is unclassified.

---

## 10. What the next wave should do with this

1. **Stop reading the fifteen as if it were the frontier.** §6 says the top of
   the scorecard carries two robust cases out of eleven. The campaign's real
   deep-frontier question is now *robust ≥ 3*, and exactly one child in ten has
   ever reached it (`ARCH.V18X13`, at 0.737484x). Every future claim should carry
   the robustness column.
2. **Fix the paraphrase bank's two dead lanes.** Six of forty-five prompts are
   unanswerable because the restatements dropped the note and the update the
   sealed prompts supply inline. Repairing them costs one edit and buys back two
   lanes of resolution on every child already decoded — including the four the
   sealed record leans on hardest.
3. **Never select a body in the reader's metric again**, and if somebody wants
   to, §3 is the three measurements they have to beat. The corollary is the
   useful half: a body improvement measured in the *stream* is the only kind that
   has ever survived a second fold, so the next body arm should be a different
   stream objective and not a different reader.
4. **Retire the extension surface's verdict string, keep the surface at gap ≥ 4.**
   Its per-wave ρ cannot cross its own thresholds meaningfully (§1) and three
   waves of go/no-go decisions were taken on a coin toss. The paraphrase surface
   is strictly better on a narrower population and should replace it wherever a
   second coordinate is wanted.
