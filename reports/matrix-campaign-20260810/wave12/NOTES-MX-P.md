# NOTES-MX-P — wave 12: the two unplayed cards

Wave 12, branch `codex/lrc-blank-compression`, source commit **`4d8774f6`**
("Register ADDENDUM 10, and give M2 the table it could not have") on top of
wave-11 HEAD `425db1d7`. Deployment `neural-forge-matrix-4d8774f6`, cut with
`git -c core.autocrlf=false archive`, review surface verified on extraction
**before any child was generated**: protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, seal
`7b859172fa48c008eb68d5ea070458ac82e2149c19c5141230d45c061aaf00e5`, prompt pack
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`. Knob file
`e355e4e7bb8a3058…`, byte-identical to waves 7-11.

Wave 11 reduced the target to arithmetic and then handed wave 12 two cards it
had not played: **drift REPAIR** (M2, blocked on a schema change wave 11
correctly refused to make mid-wave) and **drift SHOPPING** (M6, the min-drift
variant per lane, and the "which lane third" decision `NOTES-MX-O` §10.2 says
wave 11 took on autopilot).

*§0's five sentences are at the bottom, because they could not be written until
the last receipt was sealed.*

---

## 1. The allowance, computed before anything ran

Wave 11's drift model is the budget:

```
review(useful-or-partial, /15) = 12.9554 - 13.1626 x global drift    R2 0.6243, n 55
review = 8  ->  drift budget 0.376476
```

The sealed rungs it has to explain, and what they actually did:

| child | net | drift | predicted review | sealed review |
|---|---:|---:|---:|---:|
| `COMP.MA.MXO3` | 0.729103x | 0.461810 | 6.88 | **8** |
| `COMP.MA.MXO4` | 0.703396x | 0.540899 | 5.84 | **5** |
| `COMP.MA.MXO2` | 0.497739x | 0.814086 | 2.24 | **0** |

So the *empirical* alive line sits between **0.4618 (8/15)** and **0.5409
(5/15)** — above the model's own 0.3765, which is the model's residual and not a
licence to spend past it. Wave 12 uses **0.461810, the deepest measured coherent
drift**, as the working ceiling, and states both.

**The 0.5x arithmetic, from MXO3.** 0.729103x - 0.5x = 52,623,487 parameters, and
one whole SwiGLU lane at S4 removes 5,904,769 (0.025707x), so **0.5x is 8.91 —
nine — more lanes on top of the frontier**. The composed interface costs
**+0.040485** on this base (MXO3's own install trace), so the eleven-lane body of
a 0.5x child has to fit in `0.461810 - 0.0405`:

```
per-lane allowance to land at MXO3's own 0.461810   0.038037
MXO2's realized per-lane average                    0.070062
required reduction factor                           0.5429
per-lane allowance at the model's 0.376476 budget   0.030280
```

**Per-lane drift has to be roughly halved.** That is the number wave 12's two
mechanisms are measured against, and it was written down before either ran.

**What a corrective route costs, so its price was known before its value.**
`stored = 2*nnz + (targets+1)` on 1,024 targets:

| k | stored per route | as a fraction of the parent | eleven routes |
|---:|---:|---:|---:|
| 32 | 66,561 | 0.00028978x | 0.003188x |
| 64 | 132,097 | 0.00057510x | 0.006326x |
| 256 | 525,313 | 0.00228702x | 0.025157x |

At k=256 eleven routes cost **just under one whole SwiGLU lane**. So repair is
cheap in storage and the whole question is whether it is worth anything in
drift — which is exactly what ADDENDUM 10 §10.6 registers as the measurement.

## 2. What was registered before anything ran

**ADDENDUM 10** (`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md`), committed in
`4d8774f6` before the first cell, seven parts. The load-bearing ones:

* **A second, disjoint table.** `CORRECTIVE_COMPONENTS` lives in its own module
  with an `X..` id space that `CellId` refuses. `COMPONENTS`, `TECHNIQUES`,
  `known_cell_ids` and `component_sites` are byte-identical to wave 11's, so all
  442 sealed rows re-verify against exactly the objects they were sealed against.
* **`X01` reads `mlp.input` and writes `residual.post_layer`.** The frozen
  collector proves `residual.post_layer = residual.post_operator + mlp.down` for
  every block, so a tensor added to the `feed_forward` module's output is added
  to the block's output residual *exactly*. No parent tensor maps `mlp.input`
  linearly into the residual, which is what makes this a new degree of freedom
  rather than M1's re-solve renamed.
* **A corrective route replaces nothing**, so it is never subject to the 4x floor
  — there is nothing it is a quarter of — and its storage is **added** to the
  child in full. A repaired rung is comparable to an unrepaired one at a *higher*
  net ratio, never a discounted one.
* **Versioned on its own axis.** Receipt schema stays v2; `corrections` is a new
  optional key validated only when present. A receipt sealed before this addendum
  is not read against one line of the new validator, and a test says so.
* **The review reconstitutes the whole child or refuses.** A route is a forward
  hook, not a `state_dict` entry; a review that loaded only the folded tensors
  would grade a different model and call it the sealed one.

26 cases in `tests/test_matrix_corrective_routes.py`.

**The one open item wave 11 left, closed the way wave 11 said to close it.**
`NOTES-MX-O` §7a recorded that `addendum6.verdict` is the *per-child* verdict,
that the ADDENDUM 8 calculator reads it, and that wave 11's frontier child
therefore printed `dead / fail` in the calculator and `alive-degraded` in the
note. It refused to fix that by re-labelling, and specified the fix: *a sealed
per-lane verdict computed from per-lane readings the reviewer supplies case by
case*. Commit `dcfe22ff` is that, and nothing more:

* the reviewer names which lanes the clause charges (`--degenerate-lane`,
  repeatable, one case id each);
* the **reading** is derived from the charged set — `none` / `lane-scoped` /
  `general`;
* the **per-lane verdict** is computed by the same band arithmetic with the
  clause off, because the charged lanes are already observed misses in the
  useful-or-partial total and the per-lane reading re-scores nothing;
* four refusals keep it from becoming a discretion: a live per-child verdict
  cannot be sealed beside a charged lane, a lane nobody graded cannot be charged,
  charging every lane is `general` and is death under both readings, and **a
  review that names no lanes seals `unstated` and claims no per-lane verdict at
  all** — which is every wave-1..11 review, so none of them changes meaning.

12 cases in `tests/test_matrix_addendum9_per_lane.py`. Because this changes the
grading tooling and not the fold surface, it was deployed as a **second,
grading-only deployment** (`neural-forge-matrix-dcfe22ff`) used exclusively by
`seal_one.sh`: every wave-12 child is generated under `4d8774f6` and therefore
shares one `src_tree_sha`, and the verdicts are sealed by the commit that can
express both readings.

## 3. M6 — the shopping probe, and the lane the campaign was buying wrong

### 3.1 Why a probe and not sixteen cells

`COMP.MA.MXO2`'s own install trace is the reason the ledger's pooled per-site
medians cannot choose a lane. Its eleven lanes cost, in install order,

```
0.3659  0.1139  0.0421  0.0687  0.0685  0.0530  0.0518  0.0297  0.0054  0.0105  -0.0061
```

— the last lane *reduced* drift — so a lane's marginal is a function of **where
in the stack it lands**, and a median pooled over positions ranks nothing.
`reports/matrix-campaign-20260810/run/mxp/mxp_shop_probe.py` measures the number
the decision actually needs: the marginal drift of one candidate lane installed
on a **fixed** base, every candidate against the same base, in one process.

**It is exact for M-A, and that is what makes it affordable.** In M-A every
component is solved against the *parent's* sealed evidence, so a solved tensor
does not depend on what else is installed and the composed child is a function of
the **set**, not of the install order. A candidate installed after the base is
therefore the child a full run on the union recipe would build. Two things check
that rather than assert it:

* the base reads **0.421325**, which is `COMP.MA.MXO3`'s own sealed body drift to
  six decimals;
* `L07.T8.S4` — the lane `COMP.MA.MXO4` actually added — prices at **+0.076198**
  against MXO4's sealed **+0.076199**.

After every candidate the probe writes the parent's own tensors back and
re-measures the base; **all sixteen restored to 0.421325 exactly**
(`base_restored_exactly: true`, sixteen of sixteen). It seals no receipt and
appends no ledger row: it is a measurement that *chooses a recipe*, and the
recipe it chose was then run, sealed and reviewed the normal way.

### 3.2 The table

Base `COMP.MA.MXO3` body (L05 + L06 SwiGLU lanes at T8/S4), drift 0.421325.
617.2 s for all sixteen.

| rank | candidate lane | marginal drift | removes | note |
|---:|---|---:|---:|---|
| 1 | **`L13.T5.S4`** | **+0.034129** | 5,901,819 | the min-drift variant on the board |
| 2 | `L13.T8.S4` | +0.043967 | 5,904,769 | |
| 3 | `L13.T8.S8` | +0.053174 | 6,885,057 | S8 removes more and costs more |
| 4 | `L02.T8.S4` | +0.059242 | 5,904,769 | |
| 5 | `L09.T8.S4` | +0.061324 | 5,904,769 | |
| 6 | `L08.T8.S4` | +0.066225 | 5,904,769 | |
| 7 | `L11.T5.S4` | +0.072263 | 5,901,819 | |
| 8 | `L11.T8.S4` | +0.072910 | 5,904,769 | |
| 9 | `L07.T8.S4` | **+0.076198** | 5,904,769 | **what MXO4 chose** (sealed 0.076199) |
| 10 | `L11.T8.S8` | +0.078334 | 6,885,057 | |
| 11 | `L07.T8.S8` | +0.081732 | 6,885,057 | |
| 12 | `L10.T8.S4` | +0.087733 | 5,904,769 | |
| 13 | `L12.T8.S4` | +0.097154 | 5,904,769 | |
| 14 | `L07.T5.S4` | +0.101077 | 5,901,819 | T5 **loses** here |
| 15 | `L04.T8.S4` | +0.110090 | 5,904,769 | |
| 16 | `L00.T8.S4` | +0.182174 | 5,904,769 | |

Every component of every candidate stores at or below its S bound
(`storage_bound_satisfied: true`, forty-eight of forty-eight).

### 3.3 Four results, and three of them are transferable

1. **The autopilot lane was more than twice the price of the cheapest one.**
   MXO4 added L07 because it was next in depth, at +0.076198. `L13.T5.S4` costs
   **+0.034129** — 45% of it — at the same 4x bound and 5,901,819 removed against
   5,904,769. *The third lane was the wave-11 decision `NOTES-MX-O` §10.2 flagged
   as taken on autopilot, and it was measurably the wrong one.*
2. **The min-drift technique is lane-dependent, which is the whole point of
   shopping.** T5 beats T8 at L13 (0.034129 against 0.043967, −22%), ties it at
   L11 (0.072263 against 0.072910), and **loses badly** at L07 (0.101077 against
   0.076198, +33%). There is no globally better kernel; there is a better kernel
   *per site*, and it has to be measured per site.
3. **S8 does not carry from conv to MLP.** Wave 8's law — conv at S8 drifts less
   than conv at S4 — reverses on the SwiGLU lane at every site measured: L13
   0.053174 against 0.043967, L11 0.078334 against 0.072910, L07 0.081732 against
   0.076198. S8 removes ~17% more and costs 15-21% more drift each time. **The
   deeper-is-cheaper storage level is a short-conv property, not a general one.**
4. **The "cheap deep lanes" of `COMP.MA.MXO2` were a saturation artifact.** In
   MXO2, L11 cost 0.0054 and L13 −0.0061; on the frontier base the same lanes
   cost 0.0729 and 0.0440. A lane added to an already-broken child looks free
   because there is little left to break. **Marginal drift measured deep in a
   dying stack does not transfer to a healthy one**, and any allocator fitted on
   pooled ledger marginals inherits that bias.

### 3.4 The rebuilt set: `COMP.MA.MXP2`

The shopping table's own recommendation, run as a sealed child. `COMP.MA.MXP2`
is `COMP.MA.MXO3`'s frontier body plus **`L13` at `T5`/`S4`** — the min-drift
lane — where `COMP.MA.MXO4` added **`L07` at `T8`/`S4`**, the next one in depth.
Same component count, same 4x bound on every component, and 12,950 more
parameters stored (0.000056x), so the two are comparable at storage:

| | `COMP.MA.MXO4` (wave 11, autopilot) | **`COMP.MA.MXP2`** (wave 12, shopped) |
|---|---:|---:|
| folded components | 10 | 10 |
| net stored | 0.703396x | **0.703409x** |
| third lane | `L07` `T8`/`S4` | **`L13` `T5`/`S4`** |
| that lane's marginal drift | +0.076199 | **+0.034130** |
| global drift | 0.540899 | **0.489603** |
| frozen review | 5 / 15 | **6 / 15** |
| per-child reading | dead | dead |
| **per-lane reading** | dead | **impaired** |
| charged lanes | general | 3 of 16 |

**The lane cost exactly what the probe said it would.** The probe priced
`L13.T5.S4` at +0.034129 on the base; in the sealed child the same lane took the
composition from 0.421325 to 0.455455, a marginal of **+0.034130**. One
ten-thousandth of a drift unit between a probe and a sealed cell.

**What that bought, and what it did not.** 0.051 of composed drift, at the same
storage, from a *recipe decision* — no new mechanism, no extra parameters. The
review moved 5/15 to 6/15 and the degeneracy went from general to three
identifiable lanes, which is the difference between `dead` and `impaired` under
ADDENDUM 9 §9.1. **It is still not coherent**: 6/15 is below ADDENDUM 8's 8/15
alive band, so the frontier does **not** move and `COMP.MA.MXO3` at 0.729103x
remains it. The drift model predicted **6.51/15** from 0.489603 before the review
was run.

Six lanes stand on their own: the three-supplier summary keeps all three
suppliers, the decision *and* the residual risk; plus the household-task list,
the RAM-versus-storage comparison, the delay email carrying Monday and a tracking
request, the five-factor sales list, and a one-sentence creative continuation.
The three charged lanes are the greeting (rides *sleep* to the cap),
`context-retrieval` (zero tokens) and `multi-turn-correction` (an empty
`Meal 1A1..1A6` scaffold).

### 3.5 Benchmarks: the same storage, measured continuously

lm-eval from `/home/cereal/ai-benchmark-lab` (`envs/lm-eval`), tasks
`arc_easy, hellaswag, piqa, winogrande, lambada_openai, wikitext`,
`--limit 200 --seed 1234 --batch_size 4`, `max_length=2048`, float32, one
PCI-pinned sm86 RTX 3090. Partial evaluations by construction. Parent baseline
reused from wave 8; no wave-12 change touches the parent.

| model | n | net stored | review | arc_easy | hellaswag_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **parent** | 0 | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `COMP.MA.MXN4` *(w10)* | 4 | 0.754811x | 8/15 | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.09 | 168.70 |
| `COMP.MA.MXO3` *(w11, frontier)* | 7 | 0.729103x | 8/15 | 0.4300 | 0.4300 | 0.5900 | 0.5450 | 0.1650 | 527.91 | 190.15 |
| **`COMP.MA.MXP2`** | **10** | **0.703409x** | **6/15** | 0.4250 | 0.4250 | **0.5850** | **0.5400** | **0.1450** | **857.20** | **231.96** |
| `COMP.MA.MXO4` *(w11)* | 10 | 0.703396x | 5/15 | 0.4200 | 0.4350 | 0.5600 | 0.5250 | 0.1000 | 1477.86 | 257.83 |

**At the same storage and the same component count, the shopped lane wins five
of seven columns and ties a sixth**: `piqa_n` 0.5850 against 0.5600, `winogrande`
0.5400 against 0.5250, `lambada` accuracy 0.1450 against 0.1000, lambada
perplexity **857.20 against 1477.86** (42% lower), `wikitext` word-perplexity
231.96 against 257.83, `arc_easy` 0.4250 against 0.4200. It loses one:
`hellaswag_n` 0.4250 against 0.4350.

**The continuous columns say the same thing the review does and say it more
sharply.** A halved lambada perplexity at identical storage is the whole M6
result in one number, and it came from a recipe decision rather than a
mechanism. MXP2 also sits within noise of the *frontier* child `COMP.MA.MXO3` on
`arc_easy` (0.4250 / 0.4300), `hellaswag_n` (0.4250 / 0.4300) and `winogrande`
(0.5400 / 0.5450) while storing **0.0257x less** — and still scores 6/15 against
its 8/15, which is exactly the campaign's oldest lesson: benchmark accuracy is
the tie-break, coherence is the gate.

## 4. M2 — repair

*(filled in below)*

## 0. The five sentences

*(filled in last)*
