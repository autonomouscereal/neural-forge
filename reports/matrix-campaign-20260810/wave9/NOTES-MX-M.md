# NOTES-MX-M — the embedding is 29.2% of the parent, it can be folded at a true quarter, and the model that comes out cannot stop talking

**Wave 9 (MX-M), sealed 2026-08-11. ADDENDUM 6 grading, ADDENDUM 7 receipt
schema v2, ADDENDUM 8 coherence-first success. Ten composed children, ten
sealed verdicts, three new kernels for a component the campaign has never
folded, and one arithmetic fact that decides the whole phase.**

This is the wave that went after Cyril's number — **0.25× of the parent, still
answering questions coherently.** It did not get there. It got much closer than
any wave before it, it found out exactly what stops it, and the stopping point
is not where waves 1–8 would have predicted.

Three headlines:

1. **The tied token embedding is now foldable, and folding it moves the net
   ratio five times further than eight waves of body folding did.** Wave 8's
   deepest child removed 10,234,122 parameters. `COMP.MA.MXM9` removes
   **60,568,842**, and lands at **0.736305×**.
2. **Every one of the ten children is `dead` on the degeneracy clause**, and
   the best of them, `COMP.MA.MXM6`, scores **9/15 — exactly wave 8's frontier
   child's score** — while thirteen of its sixteen lanes terminate on their own
   with real answers. The clause is charged on three lanes that loop to the cap
   and leave nothing behind.
3. **0.25× is not reachable, and the reason is arithmetic rather than
   behavioural.** The embedding at a true quarter is 0.780857× of the whole
   model *on its own*. Every remaining parameter is body, wave 8 measured the
   body's coherent frontier at six folded components out of about a hundred
   available sites, and six components are worth 0.0446×. **0.25× needs the
   body folded essentially everywhere at a quarter, and seven components is
   already dead.**

---

## 1. The arithmetic that forced this wave, before any GPU ran

The parent is 229,693,184 unique parameters. **67,108,864 of them — 29.2% — are
one tensor**: `model.embed_tokens.weight`, 65,536 × 1,024, tied to the readout.
Every component waves 1–8 folded lives in the other 70.8%.

Body-wide S4, *every* body component at a true quarter, is
`(229,693,184 − 67,108,864) × 0.25 + 67,108,864 = 107,754,944` = **0.4691×**.
That is the floor of eight waves of work and it is nearly twice Cyril's number.
**No body-only recipe reaches 0.25×.** The embedding had to become a cell or
phase 1 was over before it started.

With the embedding at a true quarter the same recipe is
`(229,693,184 − 67,108,864) × 0.25 + 16,777,216 = 57,423,296` = **0.2500×**,
which is the whole target — and it requires *both*: the embedding at 4× **and**
the body at 4× essentially everywhere.

---

## 2. Three kernels, because the first two were wrong and the third is only
   less wrong

C01 had a layout entry, a level axis (`route`), an X4 width (4,096) and a
registration example in `matrix_technique_registry`'s own docstring since wave
1. It had no kernel. The A9 route that slot was written for could not be
reproduced on this estate — calibration cell 0b failed twice (`NOTES-MX-A` §5.2,
`NOTES-MX-C` §1) — and nothing has folded the token interface since.

### 2.1 T4: the plain and the weighted factorization

`matrix_technique_token_embedding.factor_token_embedding` minimizes a weighted
row error in closed form:

```
minimize  sum_v  w_v * || E_v - Ehat_v ||^2      over rank(Ehat) <= r
Ehat = E V_r V_r^T ,   V_r = top-r eigenvectors of  E^T W E
```

`w = 1` is the plain truncated SVD; `w = corpus token counts` is a genuine
weighted least squares. One 1,024×1,024 eigendecomposition, **no gradients, no
optimizer, no parent forward**. At full rank it reconstructs the table
identically, which is the tiny-rank identity gate the tests pin.

It registers under **T4** because T4 already publishes `r * (in + out)`, exactly
what the two factors store, so `storage_width_for_ratio` resolves **r = 252** at
S4 and **r = 126** at S8 with no new storage model and no schema move.

**The measurement that killed it, before either child was reviewed:**

| arm | rank | Gram energy retained | row cosine mean | weighted row cosine |
|---|---:|---:|---:|---:|
| uniform (plain SVD) | 252 | **0.4297** | 0.6640 | 0.6640 |
| token-frequency | 252 | 0.8570 *(weighted)* | 0.5401 | **0.9193** |
| uniform | 126 | 0.2901 | 0.5487 | 0.5487 |

**The table is not low rank.** At a true quarter the plain SVD keeps 43% of its
own Gram energy and reproduces its rows at a mean cosine of 0.66. Spending the
whole budget on rank buys a uniformly mediocre table.

| child | arm | net | install drift | review | verdict |
|---|---|---:|---:|---:|---|
| `COMP.MA.MXM1` | uniform SVD, r=252 | 0.780857× | **1.183802** | 0/15 | dead |
| `COMP.MA.MXM2` | frequency-weighted, r=252 | 0.780857× | 0.877127 | 0/15 | dead |

`MXM1` fails the **emit-zero floor outright** — terminal drift 1.1838 against a
1.0 bound — and its text is the cleanest collapse the campaign has recorded: the
CJK character `了` several hundred times, the literal string `assistant_`
repeated to the cap, and three lanes that emit nothing at all. `MXM2` produces
real English (`I can send a follow up and follow up…`) and loops in all sixteen
lanes. The weighting demonstrably buys back the token distribution and buys back
nothing else.

### 2.2 T5 at C01: the route, which is what the slot always meant

`_LAYOUT` has called C01's level axis a `route` since wave 1. A route spends the
budget unevenly:

* **head** — the `head_size` most important token ids, stored **verbatim**: the
  row values *and the ids*, because a route that stores rows and not the ids it
  routes them by is a table nobody can index (the same omission commit
  `8d59c4d2` found in the composition accounting);
* **tail** — every other row, at the deepest rank the rest of the budget allows.

```
stored = head*(hidden + 1) + rank*((vocab - head) + hidden)
```

At S4 with the registered head of 4,096 that resolves tail rank **201** and
16,753,664 stored (0.249649×); at S8, rank **67**. The head rows come out at row
cosine **1.000000** — they are a copy.

### 2.3 The head selection was wrong, and the measurement is the reason it changed

The first route kernel chose the head by corpus count, ties by **row L2 norm**.
`COMP.MA.MXM3` — that kernel at S4 — was qualitatively the best embedding child
of the wave to that point and still dead in all sixteen lanes. It provoked the
one measurement of this wave that changed a kernel rather than a recipe:
**how much of the frozen review surface's own 726 prompt tokens does a head of
size M actually cover?**

| head selection | M=4,096 | M=8,192 | M=12,288 | M=16,000 |
|---|---:|---:|---:|---:|
| token-id (BPE merge order) | 62.95% | 74.24% | 80.99% | **83.61%** |
| **row-norm descending** | **2.34%** | 7.85% | 11.98% | 15.01% |
| corpus count then token-id | **66.80%** | **74.93%** | 80.99% | 83.20% |
| corpus count then row-norm | 65.01% | 66.80% | 68.04% | 69.70% |

The sealed corpus observes **2,026 of 65,536 token types**, so at any head worth
having *most of the head is decided by the tie-break*, and the tie-break that
shipped was the worst one available. **Embedding row norm anti-correlates with
token usage in this table.** Token id ascending is a BPE vocabulary's own
descending merge frequency — bytes first, then the commonest merges — so it is a
usage prior that costs no evidence at all.

`beb699f2` changes the rule to counts-then-id. Every child from `COMP.MA.MXM5`
on is under the new rule and **every receipt seals which rule it used**, in
`solve.route.head_selection_rule`, so `MXM3`/`MXM4` are never confused with the
rest.

---

## 3. The head/tail sweep, and where the budget wants to go

All at S4, all embedding-only, all one folded component, all within a true
quarter of the table:

| child | head | tail rank | stored/owned | install drift | lanes at cap | lanes empty | review | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `COMP.MA.MXM3` | 4,096 *(old rule)* | 201 | 0.249649 | 0.606259 | **16** | 0 | 3/15 | dead |
| `COMP.MA.MXM5` | 8,192 | 143 | 0.249496 | 0.658222 | 4 | 0 | 6/15 | dead |
| **`COMP.MA.MXM6`** | **12,288** | **77** | 0.249954 | **0.498567** | **3** | 0 | **9/15** | dead |
| `COMP.MA.MXM7` | 15,000 | 27 | 0.249850 | 0.598189 | 4 | **3** | 5/15 | dead |

**The optimum is interior and it was measured, not assumed.** Pushing the head
from 12,288 to 15,000 costs 50 ranks of tail and the drift goes back *up*, with
three lanes emitting nothing at all: rows 15,000–65,535 at rank 27 are a
subspace so thin that the readout loses the ability to start some responses.
The tail is not free even though nothing in the corpus ever uses it.

### 3.1 `COMP.MA.MXM6`, and the judgement call of the wave

At **0.780861×** with the tied embedding stored at 0.249954 of itself,
`COMP.MA.MXM6` reads **9/15 — the same score as wave 8's frontier child** — and
**thirteen of its sixteen lanes terminate on their own.** It writes a usable
email. It gives three genuinely actionable items with no preamble. It plans a
domain migration. It asks exactly the right four follow-up questions about an
unlabelled sales graph. Its creative continuation is simply good. It computes
17 × 6 = 102 correctly and then loses the thread.

**It is sealed `dead`, and the call is deliberate.** Two of its three capped
lanes are pure loops that leave nothing behind — `context-retrieval` repeats
*the latest available information on* to the cap, `debugging-off-by-one` repeats
`max(0,` to the cap — and under the stated test (*remove the repetitions and ask
whether a response remains*) no response remains in either. That is the same
test that charged wave 8's `COMP.MA.MXL3` on **one** lane and spared
`COMP.MA.MXL6` on one.

**A reviewer who charges the clause per LANE rather than per CHILD would read
this child alive-degraded at 9/15 and 0.780861×, and that is the single most
consequential grading decision in this wave.** Both readings are in the
human-review package with the verbatim text; Cyril scores it.

---

## 4. Stage B: the embedding and the body, together, for the first time

| child | folded | what | net stored | Δ params | install drift | review | verdict |
|---|---:|---|---:|---:|---:|---:|---|
| **`COMP.MA.MXM9`** | **7** | MXL11's six-component body **+ C01 route S4** | **0.736305×** | **−60,568,842** | 0.650704 | 4/15 | dead |
| `COMP.MA.MXM10` | 5 | MXL5's four-component body + C01 route S4 | 0.753440× | −56,633,225 | 0.627954 | 6/15 | dead |
| `COMP.MA.MXM6` | 1 | C01 route S4 alone | 0.780861× | −50,334,720 | 0.498567 | 9/15 | dead |

**They do not add, they multiply the damage.** `COMP.MA.MXL11`'s body alone is
9/15 alive-degraded. `COMP.MA.MXM6`'s embedding alone is 9/15. Put together they
are **4/15**, and the shallower pairing (four body folds instead of six) is
6/15. Four body folds cost three review cases on top of the embedding; six cost
five.

`COMP.MA.MXM9` is nevertheless the deepest child the campaign has ever built and
it is not empty: it retrieves *the community garden is open from 7:30 a.m. to
4:15 p.m. Visitors may bring dogs only on Saturdays, and dogs must remain
le…* — correct, from the context — and it gives three real household actions.

---

## 5. S8: the eighth is worse in the way the sweep predicts

| child | head | tail rank | stored/owned | net | install drift | review | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `COMP.MA.MXM4` | 4,096 *(old rule)* | 67 | 0.124924 | 0.744331× | 0.845060 | 0/15 | dead |
| `COMP.MA.MXM8` | 8,000 | 3 | 0.124807 | 0.744297× | 0.888808 | 3/15 | dead |

At a true eighth the whole budget buys a head of about 8,000 rows and a tail of
rank 3. `COMP.MA.MXM8` is the interesting failure: it *barely loops* — fourteen
of sixteen lanes terminate — and it has almost nothing to say, with four empty
lanes, a greeting that echoes the prompt back verbatim, and 17 × 6 = 98. There
is no configuration of this route at an eighth that keeps a head worth having
**and** a tail worth having.

---

## 6. Every child of the wave, ranked the way ADDENDUM 8 ranks them

Machine-computed by `run/mxm/mxm_addendum8.py`, which refuses a hand-supplied
success value and re-derives the storage bound per component from the sealed
counts. Wave-8 rows included where they are the comparison.

| child | n | net stored | Δ params | worst stored/owned | review | verdict | A8 |
|---|---:|---:|---:|---:|---:|---|---|
| `COMP.MA.MXM9` | 7 | **0.736305×** | −60,568,842 | 0.249954 | 4/15 | dead | fail |
| `COMP.MA.MXM8` | 1 | 0.744297× | −58,733,184 | **0.124807** | 3/15 | dead | fail |
| `COMP.MA.MXM4` | 1 | 0.744331× | −58,725,376 | **0.124924** | 0/15 | dead | fail |
| `COMP.MA.MXM10` | 5 | 0.753440× | −56,633,225 | 0.249954 | 6/15 | dead | fail |
| `COMP.MA.MXM5` | 1 | 0.780727× | −50,365,440 | 0.249496 | 6/15 | dead | fail |
| `COMP.MA.MXM3` | 1 | 0.780772× | −50,355,200 | 0.249649 | 3/15 | dead | fail |
| `COMP.MA.MXM7` | 1 | 0.780830× | −50,341,744 | 0.249850 | 5/15 | dead | fail |
| `COMP.MA.MXM1` | 1 | 0.780857× | −50,335,744 | 0.249939 | 0/15 | dead | fail |
| `COMP.MA.MXM2` | 1 | 0.780857× | −50,335,744 | 0.249939 | 0/15 | dead | fail |
| `COMP.MA.MXM6` | 1 | 0.780861× | −50,334,720 | 0.249954 | **9/15** | dead | fail |
| `COMP.MA.MXL11` (w8) | 6 | 0.955444× | −10,234,122 | 0.249853 | 9/15 | alive-degraded | **PASS** |
| `COMP.MA.MXL5` (w8) | 4 | 0.972579× | −6,298,505 | 0.249853 | 10/15 | alive-parent-band | **PASS** |

**No component in any child of this wave stores more than a true quarter of the
tensor it replaced**, and five of the ten store at or below a true eighth. The
storage half of ADDENDUM 8 passes everywhere; the coherence half passes nowhere.

**The campaign's deepest COHERENT child is still `COMP.MA.MXL11` at 0.955444×**,
sealed in wave 8 and unmoved by this wave.

---

## 7. Benchmarks

Same wiring as wave 8, unchanged and pinned so the columns are comparable:
lm-eval from `/home/cereal/ai-benchmark-lab` (`envs/lm-eval`), tasks
`arc_easy, hellaswag, piqa, winogrande, lambada_openai, wikitext`,
`--limit 200 --seed 1234 --batch_size 4`, `max_length=2048`, float32, one
PCI-pinned sm86 RTX 3090. **Partial evaluations by construction** (200 documents
per task); they rank models against each other and are not comparable to
published full-split numbers.

The parent baseline is wave 8's, reused rather than re-measured: same model
directory, same dtype, same task list, same limit and seed, and no wave-9 change
touches the parent.

| model | n | net stored | review | verdict | arc_easy acc | hellaswag acc_n | piqa acc_n | wino acc | lambada acc | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **parent (unfolded)** | 0 | 1.000000 | 12/15 | - | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.3912 | 81.7976 |
| `COMP.MA.MXM9` | 7 | **0.736305x** | 4/15 | dead | 0.3700 | 0.3150 | 0.5100 | 0.5150 | 0.0700 | 15185.3986 | 515.0661 |
| `COMP.MA.MXM8` | 1 | **0.744297x** | 3/15 | dead | 0.2750 | 0.2800 | 0.5150 | 0.4750 | 0.0550 | 2315739.1765 | 11527.7892 |
| `COMP.MA.MXM10` | 5 | **0.753440x** | 6/15 | dead | 0.3950 | 0.3200 | 0.5400 | 0.4650 | 0.0800 | 11613.2573 | 501.3418 |
| `COMP.MA.MXM5` | 1 | **0.780727x** | 6/15 | dead | 0.3800 | 0.3600 | 0.5400 | 0.5500 | 0.0900 | 2628.3725 | 313.9674 |
| `COMP.MA.MXM3` | 1 | **0.780772x** | 3/15 | dead | 0.3850 | 0.4400 | 0.5600 | 0.5100 | 0.0600 | 5496.3102 | 471.2088 |
| `COMP.MA.MXM6` | 1 | **0.780861x** | 9/15 | dead | 0.3650 | 0.3200 | 0.5200 | 0.5350 | 0.1100 | 7337.1925 | 421.6808 |
| `COMP.MA.MXL11` *(wave 8)* | 6 | 0.955444x | 9/15 | alive-degraded | 0.5250 | 0.4600 | 0.6350 | 0.5700 | 0.2250 | 99.2209 | 118.0216 |
| `COMP.MA.MXL5` *(wave 8)* | 4 | 0.972579x | 10/15 | alive-parent-band | 0.5100 | 0.4850 | 0.6200 | 0.5500 | 0.3200 | 55.1916 | 105.5257 |

Random-chance floors: arc_easy 0.25, hellaswag 0.25, piqa 0.50, winogrande
0.50, lambada 0.

**This table carries the most important finding of the wave and the review
cannot see it.** Wave 8 established the fold-init signature: a body fold keeps
the parent's *discrimination* almost intact and loses the precision to *emit* --
`MXL11` at 0.955444x is within 0.02 of the parent on `arc_easy` and `piqa` and
**above** it on `winogrande`, while its `lambada` accuracy falls 0.4600 to
0.2250.

**A 4x embedding fold loses both halves.** Every wave-9 child sits at or barely
above chance on all four multiple-choice tasks -- `arc_easy` 0.2750-0.3950
against a 0.25 floor, `piqa` 0.5100-0.5600 against a 0.50 floor, `winogrande`
0.4650-0.5500 against a 0.50 floor -- and its perplexities are three to five
orders of magnitude worse than the parent's.  `COMP.MA.MXM8` at a true eighth
reports a `lambada` perplexity of **2,315,739**.

That is not "the same failure, further along".  A body fold degrades a model
that still knows what a plausible continuation looks like.  **The embedding fold
replaces the coordinate system all fourteen blocks were compiled against**, and
the blocks do not survive it -- even though `COMP.MA.MXM6`, on the same weights,
writes a usable email.

Note the one place the two measures disagree hardest: `COMP.MA.MXM3` scores
3/15 on the review and has the wave's **best** `hellaswag` (0.4400) and `piqa`
(0.5600); `COMP.MA.MXM6` scores 9/15 and is worse on both.  A head of 4,096 with
a tail of rank 201 keeps more of the table's geometry and cannot use it; a head
of 12,288 with a tail of rank 77 keeps less and writes better English.

---

## 8. What this wave did not settle

* **Whether the clause should be charged per child or per lane.** It decides
  whether this wave produced a 0.780861× coherent child or none at all, and it
  is a judgement about text. Both readings are in the package.
* **Whether a bigger head with a *different* tail would move it.** The sweep
  moved head and tail together because the budget couples them; a tail built
  some other way (not a projection) was never tried.
* **The 4,096 head constant is still the registered one** even though 12,288
  measured better, because the storage model has to be a function of one width
  and 4,096 is `_X4_WIDTHS["C01"]`, the campaign's own settled token-interface
  budget. Every child's realized head size and realized stored count are sealed
  and bound-checked; the constant is a planning aid.
* **`C02` and `C15` were never folded.** They share the tied tensor with C01 in
  this architecture, so folding C01 folds them, but that is an architectural
  accident rather than a measured result about either.
* **No M-B (conditioned) arm was run.** Three waves of evidence say conditioning
  is dead and wave 9 spent nothing on it.
* **The 15-prompt review is one reviewer, one pass.** ADDENDUM 2's ±2 interval
  applies.

---

## 9. Ledger, receipts, disk, provenance

**Deployments.** Wave 9 changes source for the first time since wave 7, three
times: `a395e024` (C01 as a T4 factorization), `ea4d06e3` (the T5 route),
`beb699f2` (the head tie-break). Each was cut with
**`git archive -c core.autocrlf=false`**, which is new and is why this wave needed
**no hand-restored assets at all**: all 815 files, Python and not, are
byte-identical to the git blobs of their commit. Waves 7 and 8 repaired 3 of 40
re-lined assets by hand every time; cutting the archive without the working-tree
line-ending conversion removes the defect instead of repairing it.

The review surface verifies on every review, as always:

```
e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b  V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md
7b859172fa48c008eb68d5ea070458ac82e2149c19c5141230d45c061aaf00e5  PROTOCOL_SEAL.json
0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b  lfm2_semantic_whole_response_prompts_v1.json
```

**Which child ran on which deployment** is in each receipt's `code_identity`.
`MXM1`/`MXM2` ran on `a395e024`; `MXM3`/`MXM4` on `ea4d06e3`; `MXM5`–`MXM10` on
`beb699f2`. The two superseded deployment trees were removed from the root
filesystem after use and are re-derivable from the archived tarballs under
`/mnt/ai_data/neural-forge-matrix/wave9/`.

**The knob file is byte-identical to waves 7 and 8's** (`e355e4e7bb8a3058…`), so
a wave-9 body rung and a wave-8 body rung differ in the recipe and nothing else.

**The parent control is byte-identical across waves 7, 8 and 9.** Its
calibration reproduced before any child was graded: **5 useful / 7 partial /
3 observed miss = 12/15**.

**The token-frequency table** is a plain JSON file counted over every record of
the mapped-donor calibration corpus, and `mxm_token_frequency.py` **refuses**
unless the sealed evidence base's 24 ordered batch digests are exactly the
prefix of that corpus's own. 65,535 tokens, **2,026 of 65,536 token types
observed**, `sha256 9581ef469c45cba0…`, re-checked by the kernel on every solve
against the digest the recipe names.

**Backups, before anything was written:**

| what | where |
|---|---|
| ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.pre-mxm-20260811T223442` (`ebd6d764...`) |
| `LEDGER.md` before the wave | `.../backup/LEDGER.md.pre-mxm-20260811T223442` (`1e4b86dd...`) |
| `LEDGER.md` before the render | `.../backup/LEDGER.md.before-mxm-render-20260811T223640` |

**The ledger.** 402 -> **422 rows (+20)**: 10 composed-child instances and 10
review instances.  The merger ran **exactly once** over all 20 shards and its
refuse-unless-prefix check passed byte for byte (`rows 402 -> 422 (+20)`,
`prefix verified byte-for-byte`).

```
87cc0cac57b92e7426610b7fd92c0f09b7c045deeab1d87c2a3a3fbfb67e77f2  ledger.jsonl
caaf7cdd345d82be077ee02f5001856cb6c6cfbd96a8f61ba724f967e2993e8b  LEDGER.md
```

**422 distinct `(cell_instance_id, receipt_sha)` pairs, 422 distinct receipt
digests, 20 of 20 wave-9 rows on `sm86 NVIDIA GeForce RTX 3090`, 20 of 20 at
`receipt_schema_version` 2, 20 of 20 verdicts `dead`.**

**Seals.** 418 of 422 ledger rows resolve to a receipt on disk and **all 418
verify; 0 fail.**  All **20 of 20** wave-9 seals verify.  357 receipts satisfy
today's mandatory field set and **61 wave-1 receipts do not**, for the
documented reason -- they predate `code_identity.compute_architecture` -- the
same 61 waves 4 through 8 reported.

**Four ledger rows do not resolve, and none is wave 9's**: the same three wave-1
`CAL.C01.T5.X4` calibration rows and the one wave-5 `C06.T2.S4.L12` review
instance that is also the ledger's only duplicated `cell_instance_id`.  Wave 9
found them by checking, as waves 6 through 8 did, and did not touch them.

**Mirror.** `reports/matrix-campaign-20260810/ledger.jsonl` and `LEDGER.md` are
**byte-identical to the server authority** at the two digests above.

**Disk.** Root `/` was 6.9 GB free at the start of the wave and 6.7 GB at its
low point.  Every heavy artifact -- ten composed children (each persisting a
268 MB float32 embedding tensor), six materialized float32 HF child directories
at ~920 MB each, three deployment tarballs, three test-tree tarballs and every
benchmark output -- was written under
`/mnt/ai_data/neural-forge-matrix/wave9/`.  Each materialized child's
`model.safetensors` was deleted immediately after its benchmark ran; the
directory keeps its `materialization.json`, which carries the digest chain
needed to re-derive it.  The two superseded deployment trees were removed from
`/` after use and are re-derivable from the archived tarballs.

**Retention.** Five children's composed tensors are kept -- `MXM6` (the
head/tail optimum and the wave's headline), `MXM9` (the deepest child ever
built), `MXM10` (the shallower stage-B bracket, worthless singly), `MXM5` (the
head-sweep neighbour that makes 12,288 a comparison rather than an assertion)
and `MXM8` (the only S8 arm under the corrected head rule).  **Five files,
1.25 GiB** were sha256-digested into `wave9/pruned-tensors-manifest.json` and
deleted -- the two SVD arms, the two route arms under the superseded row-norm
head rule, and the over-long 15,000 head -- and **all five digests matched the
digest their own receipt had sealed before deletion.**  Wave 9's tensors are
heavier than any wave before: folding C01 persists the whole 65,536 x 1,024
table at float32, so every child's `.pt` is 256 MiB against wave 8's tens of
megabytes.  Every pruned child keeps its receipt, diagnostics, state digest and
**full frozen-review dossier**, all committed, so the evidence of every child
survives where its tensors do not.

---

## 10. Verification

The matrix test subset, run server-side on a `git archive` of **this wave's own
commit** (`beb699f2`, `src` + `experiments` + `tests`, tarball
`6f5a617cfad95753...`) unpacked to
`/mnt/ai_data/neural-forge-matrix/wave9/testtree` -- no worktree, and therefore
none of the Codex lane's uncommitted files:

```
python -m pytest tests/test_matrix_composition.py tests/test_matrix_cell_receipt.py \
                 tests/test_run_lfm2_matrix_cell.py tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py tests/test_matrix_technique_new_axes.py \
                 tests/test_matrix_addendum6.py tests/test_matrix_addendum7.py \
                 tests/test_matrix_technique_token_embedding.py -q
# -> 10 failed, 205 passed, 1 skipped
```

**The same ten failures as waves 7 and 8, by name, and thirty more passes.**
Wave 8 reported `10 failed, 175 passed, 1 skipped`; the +30 is
`tests/test_matrix_technique_token_embedding.py`, all thirty of which pass
server-side.  The ten are the documented set, reported and not fixed:

* the `blank_receipt` `compute_architecture` `TypeError` family -- four in
  `test_matrix_technique_registry`
  (`test_t1_seals_an_unbounded_condition_number_and_keeps_the_cap_flag`,
  `test_t1_treats_an_unreported_condition_number_as_unbounded`,
  `test_t1_unclamped_solve_seals_null_clamp_ratios`,
  `test_c14_norm_refit_solve_block_is_sealable`) and one in
  `test_run_lfm2_matrix_cell_fresh_evidence`
  (`test_a_completed_route_solve_block_is_sealable`) -- whose fix is
  **uncommitted in the Codex lane** and which a `git archive` cannot see;
* `test_most_specific_registration_wins`, which monkeypatches the global
  technique dispatch table and never restores it, so every later test in the
  same process that resolves T3 gets its stub -- this is what turns 6 real
  failures into 10, and its three collateral victims are
  `test_unregistered_technique_names_the_registration_point`,
  `test_t1_refuses_components_it_does_not_model` and
  `test_both_techniques_are_wired_into_the_dispatch_table`;
* `test_factor_pair_storage_model_reproduces_a_sealed_wave2_receipt`,
  pre-existing and failing at wave-6, wave-7 and wave-8 HEAD as well.

`test_matrix_composition` -- the module the whole wave rides on -- is **33 of
33**.  `test_matrix_cell_receipt` is 31 of 31.  `test_matrix_addendum7` is 15 of
15.  `test_matrix_technique_token_embedding` is **30 of 30**.

Two of those ten also fail on the workstation when the four files are run
*alone* rather than as the campaign's subset, because
`test_t1_refuses_components_it_does_not_model` depends on
`matrix_technique_conv_fir` having been imported by an earlier file.  That is
the same shared-process import-order defect, seen from the other side, and it is
not wave 9's.

**The Codex lane was not touched.** Its uncommitted paths — including the frozen
`src/neural_forge/hf_lfm2_circuit_trace_cache.py` and
`src/neural_forge/hf_lfm2_component_method_matrix.py` — were present throughout
and are present unchanged. One wave-9 edit was made to
`hf_lfm2_component_method_matrix.py` (registering C01 as a T4 canonical kernel in
the pair-coverage table), discovered to be a Codex-lane file, and **reverted
byte-for-byte before any commit**. The pair-coverage table therefore still
reports `T4 × C01` as `kernel_missing`; it is a reporting surface, nothing reads
it, and wave 9 does not own that file. Every commit staged by explicit path.
