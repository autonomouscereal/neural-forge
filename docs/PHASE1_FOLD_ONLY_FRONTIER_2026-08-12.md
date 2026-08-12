# Phase 1 — the fold-only frontier on LFM2.5-230M

**Date:** 2026-08-12. **Branch:** `codex/lrc-blank-compression`. **Campaign:**
`reports/matrix-campaign-20260810/` (waves 1–13, `NOTES-MX-A` … `NOTES-MX-Q`).
**Ledger:** `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`.

This is the closing document of the **fold-only** matrix campaign: what
closed-form folding, with no gradient step and no distillation, reached on one
real 230M-parameter model, how it was measured, and what it did not reach.

Two standing rules from Cyril govern how it is written, and they overturn how
waves 11 and 12 reasoned:

1. **Nothing is gated on drift.** The drift law below has an R² of 0.62 and its
   residuals pay: `COMP.MA.MXO3` is coherent at a global drift of 0.4618 against
   a nominal budget of 0.3765. Drift is a **search heuristic** — it says which
   lane to try first — and never a verdict. Children get built and read.
2. **Nothing is gated on efficiency.** A mechanism that buys review cases
   expensively is still a mechanism that buys review cases. Wave 12 retired
   corrective routes on a 355× parameter-efficiency comparison; that comparison
   is a recorded diagnostic here, not a killer. 2.1M parameters is 0.9% of this
   model.

Consequently **this report does not declare fold-only closed.** It reports where
the deepest behaviourally-tested coherent child sits today, and it names the
structural mechanism classes that are queued and unplayed.

**Storage is counted in parameters, everywhere, always.** Precision reduction —
quantization, codebooks — is *not folding* and is out of scope for phase 1 by
Cyril's ruling; it is reserved for after structural folding is proven.

---

## 1. The deliverables — every frontier child, with its receipt

The parent is `LFM2.5-230M`, **229,693,184 parameters**, sealed 12/15 on the
frozen 15-prompt review (5 useful / 7 partial / 3 observed miss; it misses
`arithmetic-multi-step`, `logic-ordering`, `multi-turn-correction`). Parent
review lane `865f006e3813712e…`.

Grading is ADDENDUM 6 §6.1 (band arithmetic computed by
`matrix_cell_receipt.behavior_verdict_for`, never supplied by the reviewer), with
ADDENDUM 8's frontier definition and ADDENDUM 9's per-lane degeneracy reading.
**Both readings are printed for every child; neither is collapsed into the
other.** Benchmarks are `lm-eval` (`arc_easy`, `hellaswag`, `piqa`,
`winogrande`, `lambada_openai`, `wikitext`) at `--limit 200 --seed 1234
--batch_size 4 max_length=2048 dtype=float32` — partial evaluations by
construction, identical settings for every row.

### 1.1 The children

| child | wave | net stored | review | per-child verdict | per-lane verdict (reading) | tensors |
|---|---:|---:|---:|---|---|---|
| **`COMP.MA.MXN2`** | 10 | **0.780518x** | **10/15** | **alive-parent-band** | `unstated` (pre-ADDENDUM 9) | `/mnt/ai_data/neural-forge-matrix/wave10/cells/COMP.MA.MXN2/composed-child-tensors.pt` |
| `COMP.MA.MXN4` | 10 | 0.754811x | 8/15 | dead (degeneracy) | `unstated` | `…/wave10/cells/COMP.MA.MXN4/composed-child-tensors.pt` |
| **`COMP.MA.MXO3`** | 11 | **0.729103x** | **8/15** | dead (degeneracy) | alive-degraded (lane-scoped, stated in `NOTES-MX-O`) | `…/wave11/cells/COMP.MA.MXO3/composed-child-tensors.pt` |
| `COMP.MA.MXP1` | 12 | 0.712544x | 7/15 | dead (degeneracy) | `unstated` | `…/wave12/cells/COMP.MA.MXP1/rerun-170355/composed-child-tensors.pt` |
| **`COMP.MA.MXP2`** | 12 | **0.703409x** | **6/15** | dead (degeneracy) | impaired (lane-scoped) | `…/wave12/cells/COMP.MA.MXP2/composed-child-tensors.pt` |
| `COMP.MA.MXO4` | 11 | 0.703396x | 5/15 | dead | — | `…/wave11/cells/COMP.MA.MXO4/` |
| `COMP.MA.MXO2` | 11 | **0.497739x** | **0/15** | dead | — | pruned; manifest in `wave11/` |
| **`COMP.MA.MXQ5`** | **13** | **0.710270x** | **7/15** | dead (degeneracy) | impaired (lane-scoped, 2 of 16) | `…/wave13/cells/COMP.MA.MXQ5/composed-child-tensors.pt` + `-corrections.pt` |
| `COMP.MA.MXQ4` | 13 | 0.682581x | 0/15 | dead | dead (lane-scoped) | `…/wave13/cells/COMP.MA.MXQ4/composed-child-tensors.pt` |
| `COMP.MA.MXQ6` | 13 | 0.497777x | 0/15 | dead | dead (lane-scoped) | `…/wave13/cells/COMP.MA.MXQ6/composed-child-tensors.pt` |
| `COMP.MA.MXQ7` | 13 | 0.504638x | 0/15 | dead | dead (lane-scoped) | `…/wave13/cells/COMP.MA.MXQ7/composed-child-tensors.pt` + `-corrections.pt` |
| `COMP.MA.MXQ1` / `MXQ2` / `MXQ3` | 13 | 0.989586x | 6 / 6 / 4 /15 | dead | impaired (lane-scoped) | `…/wave13/cells/COMP.MA.MXQ{1,2,3}/composed-child-tensors.pt` |

### 1.2 The recipes, in words

* **`COMP.MA.MXN2` — 0.780518x, 10/15 alive-parent-band.** The token interface
  alone: `C01.T5.S4` at a **12,288-row verbatim head** (selected by token
  frequency) plus a **sparse-OMP tail** at rank 38 over the remaining 53,248
  rows. Nothing in the body is folded. This is the deepest child in the campaign
  that reads at the parent's own band, and it is one component.
* **`COMP.MA.MXN4` — 0.754811x, 8/15.** MXN2's interface plus one SwiGLU lane.
* **`COMP.MA.MXO3` — 0.729103x, 8/15, per-lane alive-degraded.** MXN2's
  interface plus **two** whole SwiGLU lanes at L05 and L06, every component
  `T8` (Gram relation matching) at `S4`: `C11@182`, `C12@182`, `C13@639` per
  block. Body drift 0.421325; the interface adds +0.040485 on top of it.
* **`COMP.MA.MXP2` — 0.703409x, 6/15.** MXO3 plus a **third** SwiGLU lane, and
  the lane was **shopped rather than taken in depth order**: `L13` at `T5`
  (sparse routes) rather than `L07` at `T8`, for +0.034129 marginal drift
  instead of +0.076198 — 12,950 parameters of difference in storage and half the
  lambada perplexity.
* **`COMP.MA.MXP1` — 0.712544x, 7/15.** `COMP.MA.MXO4`'s exact fold set plus
  **four ADDENDUM 10 corrective routes** at k=256 (L05/L06/L07/L13), whose
  2,101,252 added parameters are charged in full — which is why a 7/15 child
  appears at a *higher* net ratio than the 6/15 one.
* **`COMP.MA.MXO2` — 0.497739x, 0/15.** The set that meets the 0.5x target
  exactly: eleven whole SwiGLU lanes plus the token route, 34 components, every
  one at 4x. Fourteen of sixteen review lanes ride a single phrase to the token
  cap. **The 0.5x storage target is reachable; that child is not a model.**
* **Wave 13's M5 rungs** are in §2.4 and `NOTES-MX-Q`.

### 1.3 Benchmarks

| model | net stored | review | arc_easy | hellaswag (n) | piqa (n) | winogrande | lambada acc | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **parent** | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `MXN2` | 0.780518 | 10/15 | 0.4550 | 0.4450 | 0.6400 | 0.5550 | 0.2000 | 230.31 | 139.21 |
| `MXN4` | 0.754811 | 8/15 | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.09 | 168.70 |
| `MXO3` | 0.729103 | 8/15 | 0.4300 | 0.4300 | 0.5900 | 0.5450 | 0.1650 | 527.91 | 190.15 |
| `MXP2` | 0.703409 | 6/15 | 0.4250 | 0.4250 | 0.5850 | 0.5400 | 0.1450 | 857.20 | 231.96 |

Read the last two columns, not the first four. The multiple-choice tasks move
from 0.545 → 0.425 on `arc_easy` while a 230M parent is only 0.045 above chance
on `hellaswag` to begin with; **perplexity is the column that ranks these
children**, and it degrades monotonically and steeply: 24 → 230 → 370 → 528 →
857 on lambada while storage falls 1.00 → 0.70. `COMP.MA.MXP1` has **no
benchmark row at all**, because its corrective routes are forward hooks rather
than `state_dict` entries and the materializer cannot write an HF child that
carries them; that gap is stated in `NOTES-MX-P` §6a and is still open.

---

## 2. The quantitative case — where the fold-only frontier sits, and why

### 2.1 The drift law, and what it is for

Across 55 sealed rungs, the composed child's global drift against the
`copy_exact` control predicts its review score:

```
review(useful-or-partial, /15) = 12.9554 − 13.1626 × global_drift     R² = 0.6243, n = 55
review = 8   ->   drift 0.376476        (the nominal coherence budget)
```

**R² = 0.62 means 38% of the variance in behaviour is not drift**, and the
residuals are not noise around the line — they are the interesting cases:

| child | drift | predicted | sealed | residual |
|---|---:|---:|---:|---:|
| `COMP.MA.MXO3` | 0.461810 | 6.88 | **8** | +1.1 |
| `COMP.MA.MXO4` | 0.540899 | 5.84 | 5 | −0.8 |
| `COMP.MA.MXO2` | 0.814086 | 2.24 | 0 | −2.2 |
| **wave 13 `COMP.MA.MXQ2`** | **0.399369** | 7.70 | **6** | −1.7 |
| **wave 13 `COMP.MA.MXQ1`** | **0.429852** | 7.30 | **6** | −1.3 |

Wave 13 added the cleanest counter-example the campaign has: `MXQ2` (head merge
at L02) has **lower** drift than `MXQ1` (head merge at L12) — 0.399 vs 0.430 —
and **four** review lanes that loop to the 1024-token cap against `MXQ1`'s
**one**. Same score, worse text, lower drift. **The law ranks them backwards.**

So the law is used here for what it is good for — ordering a search — and never
to refuse a child a reading.

### 2.2 The lane price table, and saturation

A "lane" is one whole component group folded at one block. The prices below are
**marginal composed drift measured on a fixed base in one process**, every
candidate against the same base, with the parent's own tensors written back and
the base re-measured after each one (`base_restored_exactly: true` for all of
them). The method is exact for M-A because every component is solved against the
*parent's* sealed evidence, so the composed child is a function of the **set**,
not the install order — and the probe proves it rather than asserting it: the
`MXO3` base re-reads **0.421325**, its own sealed body drift, and `L07.T8.S4`
prices at **+0.076198** against `COMP.MA.MXO4`'s sealed **+0.076199**.

**Third lane, on the `COMP.MA.MXO3` base (wave 12, `wave12/probes/lane-shopping.json`):**

| rank | lane | marginal drift | removes | drift / M-param |
|---:|---|---:|---:|---:|
| 1 | `L13.T5.S4` | **+0.034129** | 5,901,819 | 0.005783 |
| 2 | `L13.T8.S4` | +0.043967 | 5,904,769 | 0.007446 |
| 9 | `L07.T8.S4` | +0.076198 | 5,904,769 | 0.012905 |
| 16 | `L00.T8.S4` | +0.182174 | 5,904,769 | 0.030852 |

**Fourth lane, on the `COMP.MA.MXP2` base (wave 13, `wave13/probes/deep-ladder.json`):**

| rank | lane | marginal drift | removes | drift / M-param |
|---:|---|---:|---:|---:|
| 1 | `L02.T8.S4` | **+0.050518** | 5,904,769 | 0.008555 |
| 2 | `L09.T8.S4` | +0.055169 | 5,904,769 | 0.009343 |
| 3 | `L08.T8.S4` | +0.057867 | 5,904,769 | 0.009800 |
| 4 | `L11.T5.S4` | +0.061364 | 5,901,819 | 0.010398 |
| … | `L02.T7.S4` (**head merge**) | +0.092359 | 2,392,064 | 0.038610 |
| … | `L00.T5.S4` | +0.211109 | 5,901,819 | 0.035770 |

**Saturation, measured across seven rounds — and it is not what wave 12
inferred.** `NOTES-MX-P` §6a listed the compounding rate as inferred from
`COMP.MA.MXO2`'s install trace, and waves 11–12 argued from it that the per-lane
allowance "gets harder, not easier". Wave 13's greedy ladder measured it: price
every remaining lane on the current base, install the cheapest **permanently**,
re-price, seven times (`wave13/probes/deep-ladder.json`, 3,982 s).

| the lane being bought | base lanes | base drift | cheapest marginal | drift / M-param | winner |
|---:|---:|---:|---:|---:|---|
| 3rd (wave 12, MXO3 base) | 2 | 0.421325 | +0.034129 | **0.005783** | `L13.T5` |
| 4th | 3 | 0.455455 | +0.050518 | **0.008555** | `L02.T8` |
| 5th | 4 | 0.505972 | +0.043186 | 0.007314 | `L08.T8` |
| 6th | 5 | 0.549159 | +0.038762 | 0.006565 | `L11.T8` |
| 7th | 6 | 0.587921 | +0.030083 | **0.005095** | `L09.T8` |
| 8th | 7 | 0.618004 | +0.046781 | 0.007923 | `L12.T8` |
| 9th | 8 | 0.664785 | +0.041657 | 0.007055 | `L07.T8` |
| 10th | 9 | 0.706443 | +0.037158 | 0.006293 | (not installed) |

**The marginal price does not run away.** It rises 48% from lane 3 to lane 4,
then *falls* through lane 7 — the seventh lane is the cheapest per parameter the
campaign has ever measured, on a six-lane base — then oscillates between 0.0051
and 0.0079. Over seven rounds the total drift accumulates **roughly linearly at
about 0.042 per lane**, and the shape of the curve is flat, not exponential.

**So the reason 0.5x is hard is not that lanes get expensive. It is that the
coherence budget is small and drift accumulates steadily against it.** Nine
shopped lanes already sit at drift 0.706 — 1.5× the deepest *coherent* drift
ever measured (0.4618) — while removing 53.1M parameters, which is only 23% of
the parent. That is the real arithmetic, and it is an argument about a budget,
not about a price curve.

`COMP.MA.MXO2`'s own eleven-lane install trace shows where the curve ends up —
0.3659, 0.1139, 0.0421, 0.0687, 0.0685, 0.0530, 0.0518, 0.0297, 0.0054, 0.0105,
**−0.0061** — the last lane *reduced* drift. Once the child is far enough from
the parent, moving it further barely registers on a distance-to-parent metric.
**That is another reason drift cannot be a gate: at depth it stops
discriminating**, which is exactly the depth where a gate would be doing the
most work.

### 2.3 The repair exchange rate — a diagnostic, not a verdict

ADDENDUM 10's corrective routes are additive sparse maps from `mlp.input` into
`mlp.down`, solved in closed form against a capture taken *through* the composed
child. Measured on `COMP.MA.MXO4`'s identical fold set:

* review **5/15 → 7/15**; drift **0.540899 → 0.517480**;
* cost **2,101,252 stored parameters**, charged in full (a route replaces
  nothing, so it is never subject to the 4x floor);
* **0.011146 drift per million stored parameters**, against lane shopping's
  **3.961** — the 355× ratio wave 12 closed the mechanism on;
* out of sample, the explained fraction of the block gap decays with depth —
  28.0% at L05, 10.4% at L06, 10.0% at L07, and **0.0% at L13, a block that was
  never folded**, where drift went *up*.

**Wave 13's reading of the same numbers.** 2.1M parameters is **0.91% of this
model** and it bought two review cases and a 4.3% drift reduction with no
gradient step. The efficiency ratio is a true statement about *where the next
parameter is best spent*; it is not a reason to stop carrying routes, and the
comparison that matters — against full distillation — was never run.

**So wave 13 built the rung wave 12 designed and never ran: routes on the
SHOPPED child.** `COMP.MA.MXQ5` is `COMP.MA.MXP2`'s exact fold set plus three
k=256 routes at L05/L06/L13:

| child | net stored | drift | review | per-child | per-lane (reading) | lanes charged |
|---|---:|---:|---:|---|---|---:|
| `COMP.MA.MXP2` (no routes) | 0.703409x | 0.489603 | 6/15 | dead | impaired | — |
| **`COMP.MA.MXQ5` (+3 routes)** | **0.710270x** | **0.487874** | **7/15** | dead | impaired (lane-scoped) | 2 of 16 |

**Repair pays on the shopped child too: +1 review case for 1,575,939 added
parameters — 0.69% of the model — charged in full.** It is also the healthiest
text wave 13 produced: one lane at the token cap, against `MXP2`'s profile, and
the only child in the wave with two `Useful` gradings (`context-retrieval` and
`summarization-priorities`, both near-complete against the source).

And the route trace repeats wave 12's locality result with a new twist:
`0.489603 → 0.474727` (L05 route, −0.014876), `→ 0.472629` (L06, −0.002098),
`→ 0.487874` (**L13, +0.015245 — worse**). Wave 12 saw a route make drift worse
only at a block that was *never folded*. Here L13 **is** folded, and its route
still hurt. Repair's purchase is not "wherever a fold is"; it is where the
fold's own error is still linear in that block's input, and at the end of a
stack it is not.

**And at depth it buys nothing at all.** `COMP.MA.MXQ7` is the eleven-lane
shopped set plus three k=256 routes at L10/L12/L13:

| step | drift |
|---|---:|
| `COMP.MA.MXQ6` (no routes) | 0.821803 |
| + L10 route | 0.801589 (**−0.020214**) |
| + L12 route | 0.818947 (**+0.017358**) |
| + L13 route | **0.821895** (**+0.002948**) |

Net **+0.000092 — worse than the unrepaired child** — for 1,575,939 added
parameters, and 0/15 either way. The locality result stands on its own and is
the transferable part: **repair has purchase only where a fold's own error is
still linear in that block's input, and that purchase collapses with depth**;
on a deep stack the later routes actively re-break what the earlier ones fixed.

### 2.4 What wave 13 measured: M5, the head merge

`C10` (all four attention projections at one block) had **never produced a
number at 4x**. Its only kernel was head *drop*, which stores `k_proj`/`v_proj`
whole and bottoms out near 0.5x — ADDENDUM 6 calls that `cannot-express-4x` and
no behavioural claim follows from it. Wave 13 built the merge kernel from
`NOTES-MX-P` §8: group heads, align each to its group with a closed-form
orthogonal Procrustes rotation, store `g` shared bases plus one rotation per
head, and re-solve the O group bases against the sealed attention write.

```
stored = g_q*(D*hidden + hidden*D) + g_kv*2*(D*hidden) + (H + H_kv)*D*D
g_q = 4, g_kv = 2  ->  884,736  = 0.281250x   (over the 4x bar: diagnostic only)
g_q = 3, g_kv = 2  ->  753,664  = 0.239583x   (the qualifying config)
g_q = 1, g_kv = 1  ->  360,448  = 0.114583x   (S8 IS expressible, unlike head drop)
```

**Identity gate:** at one basis per head the whole pipeline must return the
parent. Measured across five sites: **6.7e-16 to 8.7e-15** max absolute
deviation over all four projections.

**Price, on the same `COMP.MA.MXO3` base as the wave-12 table:**

| lane | marginal drift | removes | drift / M-param | ×  the cheapest SwiGLU lane |
|---|---:|---:|---:|---:|
| `L13.T5.S4` (SwiGLU) | +0.034129 | 5,901,819 | 0.005783 | 1.0 |
| `L02.T7.S4` (merge) | +0.110941 | 2,392,064 | 0.046379 | **8.0×** |
| `L08.T7.S4` | +0.157162 | 2,392,064 | 0.065701 | 11.4× |
| `L12.T7.S4` | +0.157983 | 2,392,064 | 0.066045 | 11.4× |
| `L10.T7.S4` | +0.176884 | 2,392,064 | 0.073946 | 12.8× |
| `L04.T7.S4` | +0.183834 | 2,392,064 | 0.076852 | 13.3× |

**And the behavioural reading, which is the one that counts.** One merged
attention block, alone, nothing else folded — net 0.989586x, 1.04% of the model
removed:

| child | site | drift | review | per-child | per-lane (reading) | lanes charged |
|---|---|---:|---:|---|---|---:|
| `COMP.MA.MXQ1` | L12 | 0.429852 | **6/15** | dead | impaired (lane-scoped) | 1 of 16 |
| `COMP.MA.MXQ2` | L02 | 0.399369 | **6/15** | dead | impaired (lane-scoped) | 4 of 16 |
| `COMP.MA.MXQ3` | L08 | 0.468446 | **4/15** | dead | impaired (lane-scoped) | 6 of 16 |

**A single merged attention block costs more behaviour than two whole SwiGLU
lanes plus the entire token interface.** `COMP.MA.MXO3` removes 27.1% of the
model and reads 8/15; `COMP.MA.MXQ1` removes 1.04% and reads 6/15.

**Stacked, and read rather than predicted.** Per the directive, the stack was
built regardless of the solo price. `COMP.MA.MXQ4` is `COMP.MA.MXP2`'s exact
fold set plus head merges at the two cheapest sites (L02 and L12):

| child | net stored | drift | review | per-child | per-lane (reading) | lanes charged |
|---|---:|---:|---:|---|---|---:|
| `COMP.MA.MXQ4` | **0.682581x** | 0.679394 | **0/15** | dead | dead (lane-scoped) | 9 of 15 graded |

It is **deeper than every previous coherent child** — 0.682581x against
`COMP.MA.MXP2`'s 0.703409x — and it is a wreck: five lanes run to the
1024-token cap on a single repeated phrase (`instruction-three-actions` emits
"1. Plan a meal plan:" 90+ times), and the safety lane emits "The room is still
clean" four times. Its own install trace says where the budget went: the L02
merge, installed first, is 0.399369 of the child's 0.679394 terminal drift by
itself, and the L12 merge adds **+0.111632** on top of six SwiGLU lanes.

**M5 does not reopen the road to 0.5x, and that is now a behavioural finding
rather than an arithmetic one.**

**The continuous axis says the same thing, independently.** `COMP.MA.MXQ1` — one
merged attention block, 1.04% of the parent removed — was materialized and
benchmarked under the identical settings as every other row:

| model | net stored | review | arc_easy | hellaswag (n) | piqa (n) | winogrande | lambada acc | **lambada ppl** | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | 81.80 |
| `MXQ1` (one merged block) | **0.989586** | 6/15 | 0.5000 | 0.4750 | 0.6650 | 0.5600 | 0.1450 | **341.58** | 105.94 |
| `MXP2` (10 folds, 29.7% removed) | 0.703409 | 6/15 | 0.4250 | 0.4250 | 0.5850 | 0.5400 | 0.1450 | **857.20** | 231.96 |

**Merging one attention block multiplies lambada perplexity by 14× while
removing 1% of the model.** Ten SwiGLU/interface folds that remove 30× more
storage multiply it by 35×. Per parameter removed, M5 is roughly **12× worse on
perplexity as well as on drift** — two independent instruments, the same
verdict.

**Why, mechanically.** The merge's own reconstruction says it: the held-out
write relL2 of the merged O map is **0.35–0.56** at S4, and the head partitions
the objective-linkage clustering finds are wildly unbalanced — at L12,
`[[0,8,9,15], [1,2,3,4,5,6,7,10,11,12,13], [14]]`, and at every site the kv
partition is 7 heads + 1 outlier. **LFM2's attention heads are not redundant up
to rotation.** There is no low-rank head structure to exploit at this scale; the
clustering is merging things that are not alike because the level told it to.

### 2.5 The mechanism results that are load-bearing for the frontier

* **Locality (ALS, wave 11).** Re-solving every component against the *composed
  child's own* activations — the obvious fix for "every body fold was solved
  against activations the rest of the composition then invalidated" — made the
  composition **worse**, not better. The fold's error is local to its block; the
  composition's error is not the sum of local errors that a local re-solve can
  reach.
* **Sharing falsified (wave 11).** Parameter sharing across blocks was measured
  and does not pay on this model.
* **Interface × body multiplies (wave 10).** The token interface's cost and the
  body's cost compose multiplicatively, not additively: MXN2's interface is
  10/15 alone, and it costs +0.040485 drift when installed on top of MXO3's
  body.
* **The block rule (wave 7, `NOTES-MX-K` §2.1).** A block hosts at most one
  folded **lane**. Two MLP components in one block cost nothing; a conv or
  attention writer *plus* an MLP fold in the same block costs −3 to −4 review
  cases, because they sit on opposite sides of that block's FFN norm. Machine-
  enforced in the recipe generator, overridable only with a written reason.
* **Deeper-is-safer is conv-specific (wave 12).** `S8` beats `S4` on the short
  conv writer (13/15 vs 10/15 — one case *above* the same-process parent) and
  **loses on every SwiGLU site measured**. The level ladder is per component and
  is not monotone.
* **The min-drift variant is per site (wave 12).** `T5` beats `T8` at L13 and
  loses at L07. A pooled per-site median ranks nothing.

### 2.6 Per-component-class safety table (updated with M5)

| class | component | 4x in isolation | 4x in composition | 8x | note |
|---|---|---|---|---|---|
| MLP gate `w1` | C11 | yes, 10–11/15 | yes at L05 (10/15, drift 0.168344 — the smallest single install measured) | 9/15 at L05 | the safest body fold known |
| MLP up `w3` | C12 | yes, 9/15 | yes, with C11 in the same lane | untested alone | |
| MLP down `w2` | C13 | yes, 12/15 at L05 | yes at L04/L05; 5–7/15 at L00/L03/L06/L13 | only inside a dead all-S8 child | strongest site dependence measured |
| short conv out | C05 | yes, 12/15 | 10/15 at S4, **13/15 at S8** | better than its own S4 arm | the one component where 8x beat 4x |
| attn q | C06 | 10/15 | one site only | — | |
| attn k | C07 | 11–12/15 at L12 | yes at L12; 8/15 as a second; **4/15 as a third — five lanes become single-token loops** | — | three stacked attention folds collapse the decoder |
| attn v / out | C08 / C09 | 8–9/15 | degrade at S4 | — | |
| **heads** | **C10** | **expressible at 4x from wave 13 (0.239583x) and at 8x (0.114583x); 6/15 solo at the two cheapest sites, 4/15 at L08** | **8–13× the per-parameter drift of a SwiGLU lane** | expressible, unmeasured | head *drop* remains `cannot-express-4x`; the merge is expressible and **behaviourally expensive** |
| norms | C14 | n/a (diagonal) | — | — | |
| token table | C01 | see §3.1 | **10/15 alive at 0.780518x** | 7/15 dead at S8 | the only 4x fold that pays 29% of this model |

---

## 3. What transfers off this model

Full derivations in `docs/FOLDING_SCALING_THEORY_2026-08-12.md`; this is the
summary a 7B/70B attempt should start from.

### 3.1 The interface rule and the interface share formula

The token interface (`embed_tokens`, tied to `lm_head`) is **29.2%** of
LFM2.5-230M and folds to 0.25x while reading 10/15. That is the single biggest
fold-only win in the campaign, and **it is a shrinking prize**: interface share
is `vocab × hidden / total`, which falls roughly as `1/depth` at fixed vocab.

```
interface share ≈ (V·H) / (V·H + L·c·H²)          c ≈ 12–16 for a standard block
LFM2.5-230M   V=65,536  H=1,024  L=14   ->  29.2%     measured
Llama-3-8B    V=128,256 H=4,096  L=32   ->  ~13%      projected
Llama-3-70B   V=128,256 H=8,192  L=80   ->  ~4%       projected
```

**So the technique that pays best here pays least at scale**, and the body is
where a large-model fold has to win. What transfers is the *method*: a verbatim
head selected by token frequency plus a sparse-route tail, with the route
indices counted in the storage (`_c01_token_route_storage`) — a sparse tail beat
a low-rank tail everywhere it was measured.

### 3.2 Attention share moves the other way — and M5 is the reason to care

Attention is **11.61%** of LFM2.5-230M's body (six attention blocks in fourteen
layers, GQA 16/8). On a standard dense transformer it is ~⅓. That is why M5 was
worth building even though the head-drop verdict was negative: a mechanism that
is worth 1% here could be worth 8% at 70B.

**Wave 13's result therefore transfers as a warning, and it is a specific one.**
The merge is exact where exactness is checkable (identity gate ≤ 8.7e-15), the
storage claim is honest (0.239583x per attention block, all four projections),
and the mechanism still costs **8–13× more drift per parameter** than the MLP
lanes it competes with, because **the heads are not rotations of a few shared
bases**. The measurement to repeat first on a larger model is not the merged
child — it is the *cluster objective* (`solve.query_family.objective_relative`
in any wave-13 receipt): if a 70B model's heads cluster tightly, M5's price
falls and the mechanism becomes the largest single fold-only prize available at
scale; if they scatter the way LFM2's do, it does not.

### 3.3 Drift-budget methodology (as a heuristic)

Re-derive, do not import, the constants:

1. Seal ≥ 20 composed rungs with both drift and a frozen review score.
2. Regress review on drift; publish the R² and the residuals.
3. Use the fit to **order** candidate lanes, and shop them with a fixed-base
   probe (~40 s per candidate) rather than taking them in depth order.
4. **Then build and review.** Wave 13's `MXQ1`/`MXQ2` pair is the standing
   demonstration that the order is not the verdict.

### 3.4 Per-class safety, transferable shape

MLP mass folds; the last attention block's key projection folds; value/output
projections and stacked attention folds do not; head merging is expressible but
expensive; mid-depth sites are safest; one component folds *better* at 8x than
at 4x. Re-measure the ladder per component and never assume it is monotone.

---

## 4. What fold-only did not reach, and what happens next

### 4.1 The honest statement

**Fold-only did not produce a coherent 0.5x child on this model.** The deepest
behaviourally-tested children are:

* **0.780518x — 10/15, alive-parent-band** (`COMP.MA.MXN2`): the deepest child
  that reads at the parent's own band.
* **0.729103x — 8/15** (`COMP.MA.MXO3`): the deepest child at the 8/15 alive
  line the drift budget is calibrated on.
* **0.703409x — 6/15** (`COMP.MA.MXP2`): the deepest child whose output is
  coherent, on-prompt language, which is ADDENDUM 8 §8.2's success band.
* **0.497739x — 0/15** (`COMP.MA.MXO2`): the storage target is reachable and
  that artifact is not a model.
* **0.497777x — 0/15** (`COMP.MA.MXQ6`, wave 13): **the same depth, bought in
  the measured-cheapest order instead of by depth**, and it is the same wreck.
  Ten of sixteen lanes run to the 1024-token cap, one emits zero tokens, and its
  drift is **0.821803** against `MXO2`'s 0.814086 — shopping the order did not
  help at this depth. That is the directive's test executed: the depth the
  allowance arithmetic wrote off was built and read rather than predicted, and
  the reading agrees with the prediction *for a reason the prediction did not
  give* (see §2.2 — the lane price does not run away; the budget simply runs
  out).

### 4.2 The measured reason — stated as measurement, not as closure

From 0.703409x, reaching 0.5x needs 46.7M more parameters removed ≈ **7.9 more
SwiGLU lanes**. The measured marginal price of the *fourth* lane is 0.008555
drift per million removed, 48% above the third's 0.005783, and the campaign has
no measurement of lanes 5–14 on a healthy base (wave 13's ladder is measuring
them as this report is written; see `NOTES-MX-Q`). Head merging, the one
remaining unplayed fold class in the matrix, prices 8–13× worse per parameter
and cannot supply the missing storage anyway: **all six attention blocks merged
at once remove only 6.25% of the parent** (6 × 2,392,064 = 14.35M), which moves
0.703409x to 0.640920x at best.

**What that arithmetic licenses, and what it does not.** It licenses an ordering:
SwiGLU lanes first, then everything else. It does **not** license the sentence
"0.5x is arithmetically closed", which waves 11 and 12 wrote and which this
report withdraws — because the arithmetic extrapolates a drift heuristic with
R² = 0.62 past the region where it was fit, into a region where its own
saturation result says it stops discriminating, and because three whole classes
of structural fold have never been priced at all (§4.4).

### 4.3 The designed next stage: fold-init → distillation

The stage-2 estate is built and preserved at
**`/mnt/ai_data/neural-forge-stage2/`**: a trainer, **26 corpus shards**, and
admission gates. Its design document is
`docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`, and the stage-1 ceiling it
was written against is `docs/STAGE1_CEILING_REPORT_2026-08-08.md`.

The thesis stage 2 tests is that **a fold is an initializer, not a product**:
`COMP.MA.MXO2` at 0.497739x is a model-shaped object whose weights are
individually well-solved and whose composition is far from the parent's
function. Distillation from the parent onto that initialization is the designed
experiment; the fold-only campaign's contribution is a *ranked family of
initializations* with sealed receipts, so stage 2 can ask which initialization
distils fastest rather than guessing one.

### 4.4 Queued and unplayed: the structural classes (wave 14)

The matrix priced **factorization** (T1–T4), **sparse routing** (T5), **relation
matching** (T7/T8), **progressive multi-pass** (T9), **repair** (X01) and now
**head merging** (M5). It has never priced:

* **CUR / interpolative decomposition** — a factorization whose factors are
  *actual rows and columns of the parent*, so the basis is not a new dense
  object that has to be stored at full width. Storage: `k` column ids + a `k×n`
  coefficient block, and the ids are integers.
* **Nearest-Kronecker-product (Van Loan)** — approximate `W ≈ A ⊗ B` by an SVD
  of the rearranged matrix. A `1024×1024` weight becomes `32×32` + `32×32` in
  the exact rank-1 case, which is a 512× contraction, and the residual is
  measurable rank by rank. This is the highest-leverage unplayed class by
  storage arithmetic alone.
* **Monarch / block-diagonal + low-rank** — permutation-sandwiched block-diagonal
  factors; expressly designed to be a drop-in for a dense linear layer, and the
  closest structural relative of what the campaign already knows folds (MLP
  mass).
* **Tensor-train / tensor-ring on the reshaped weight** — the natural
  generalization of the low-rank family the campaign has already priced, with a
  storage model that is linear rather than quadratic in the mode widths.

Each of these is closed-form, gradient-free, and expressible in the existing
kernel contract (parent-shaped installed tensors, a published storage model, an
identity gate). **The fold-only frontier is open until they are priced
behaviourally.**

**Explicitly out of scope for phase 1:** quantization and codebook/precision
reduction of any kind. Per Cyril's ruling, that is not folding; it is applied
after real structural folding is proven, and storage in this campaign is counted
in parameters and never in bytes.

---

## 5. Corrections of record — the epistemic trail

Every one of these was a claim the campaign made, then measured wrong, then
withdrew in a receipt rather than in a summary.

1. **A9 storage accounting (`8d59c4d2`).** Route storage counted the coefficient
   values and not the indices that route them. A table nobody can index is not
   storage. Fixed in `_t5_route_storage` and `_c01_token_route_storage`, and the
   same class of error was caught one level down in the composition accounting.
2. **The block rule (wave 7).** "Two folds in one block" was the hypothesis;
   the measurement said the conflict is **two folds on opposite sides of one
   block's operator/FFN boundary** — two MLP folds in one block cost nothing.
   The rule was rewritten on the lane axis and machine-enforced.
3. **C05 "toxicity" (wave 7).** The short-conv writer was believed to poison a
   block. It folds at 13/15 at **S8** — one case above the same-process parent —
   and the earlier reading was a block-collision artifact, not a component
   property.
4. **Token match retired as a gate (ADDENDUM 6).** Free-run token match against
   the control was the campaign's original pass/fail. It was demoted to a
   diagnostic column and replaced by the frozen 15-prompt review, because a
   child can match tokens and be incoherent and vice versa. Every dossier still
   prints token match, labelled `[DIAGNOSTIC ONLY]`.
5. **The per-child verdict was the wrong object for a lane-scoped failure
   (ADDENDUM 9, wave 12).** `COMP.MA.MXO3` printed `dead` in the calculator and
   `alive-degraded` in the note. Fixed by sealing a **per-lane** reading computed
   from reviewer-supplied per-lane readings, with four refusals — including "a
   review that names no lanes seals `unstated`", which keeps every wave-1..11
   review meaning exactly what it meant.
6. **`NOTES-MX-P` §8's head-merge arithmetic (wave 13).** §8 priced `g=3` at
   688,128 = 0.21875x by evaluating its kv term at `g/2 = 1.5`. Eight kv heads
   split into two groups; the implementable config stores **753,664 =
   0.239583x**. Still under the 4x bar — the verdict survived — but the number
   was 8.7% low, and the kernel and its tests now pin both.
7. **"Impaired" is the calculator's floor band (wave 13).** `behavior_verdict_for`
   returns `impaired` for *any* non-degenerate score below 8/15, so a per-lane
   `impaired` at 4/15 and one at 7/15 print the same word. ADDENDUM 8 §8.2's
   prose treats 6–7/15 as the success band. **A per-lane verdict must always be
   quoted with its score**, and every table in this report does.
8. **"0.5x is arithmetically out of reach" (waves 11–12) — withdrawn.** It was
   an extrapolation of an R² = 0.62 heuristic past its fitted range, used as a
   conclusion. What is measured is a lane price and its saturation; what is not
   measured is three structural mechanism classes and every deep shopped set
   built with repair on top.
9. **"Routes are not the road" (wave 12) — reframed.** The 355× efficiency ratio
   is real and is retained as a diagnostic. It was used as a kill decision; the
   comparison that would justify a kill — against the cost of the distillation
   it replaces — was never run.

---

## 6. Where everything is

| artifact | location |
|---|---|
| ledger (447 rows) | `/mnt/ai_data/neural-forge-matrix/ledger.jsonl` |
| wave notes | `reports/matrix-campaign-20260810/wave{5..13}/NOTES-MX-{I..Q}.md` |
| design + addenda 1–10 | `docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` |
| scaling theory | `docs/FOLDING_SCALING_THEORY_2026-08-12.md` |
| frontier child tensors | `/mnt/ai_data/neural-forge-matrix/wave{10,11,12,13}/cells/<CHILD>/composed-child-tensors.pt` |
| human-review packages | `reports/matrix-campaign-20260810/wave{12,13}/human-review/` |
| benchmarks | `/mnt/ai_data/neural-forge-matrix/wave{11,12,13}/bench/` |
| stage-2 estate | `/mnt/ai_data/neural-forge-stage2/` (trainer, 26 corpus shards, admission gates) |
| M5 kernel | `src/neural_forge/matrix_technique_head_merge.py`, tests in `tests/test_matrix_technique_head_merge.py` |
