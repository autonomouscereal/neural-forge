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
* **`X01` reads `mlp.input` and writes into `mlp.down`.** The frozen collector
  proves `residual.post_layer = residual.post_operator + mlp.down` for every
  block, so an addition inside `mlp.down` is an addition to the block's output
  residual *exactly*. No parent tensor maps `mlp.input` linearly into the
  residual, which is what makes this a new degree of freedom rather than M1's
  re-solve renamed. (The addendum was registered with the write at the
  `feed_forward` module and the frozen collector refused it on a live run; §4.1
  is what happened and why the write moved one module deeper. The operator did
  not change.)
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

## 4. M2 — repair, and the exchange rate it trades at

`COMP.MA.MXP1` is `COMP.MA.MXO4`'s **exact** ten-component fold set — verified
byte-equal to MXO4's recipe before the run, and its install trace reproduces
MXO4's per-component drift to six decimals at all ten rungs (0.168344, 0.271744,
0.311616, 0.330763, 0.384159, 0.421325, 0.430738, 0.494235, 0.497524, 0.540899)
— plus four ADDENDUM 10 corrective routes at k=256.

### 4.1 The collector refused the first design, and it was right

The first attempt died four minutes into its second route:

```
COMPOSED CHILD FAILED Lfm2CircuitTraceCaptureError:
Layer 5 residual endpoints do not reproduce its output.
```

`hf_lfm2_circuit_trace_cache` verifies, per layer, that `layer_output ==
residual.post_operator + mlp.down`, and it captures `mlp.down` on
`feed_forward.w2`. A tensor added to the *feed_forward module's* output lands in
`layer_output` and not in `mlp.down`, so the identity breaks — and with it every
capture after the first, which is fatal, because each route is solved from a
capture taken through the child the previous routes made.

**The collector is frozen and it was not the thing that was wrong.** The write
moved one module deeper: X01 attaches as a pre-hook on `feed_forward` that caches
`mlp.input` plus a forward hook on `feed_forward.w2` that adds `cached @ R` to
`mlp.down`. The operator is unchanged — a linear map from `mlp.input` into the
block's output residual, which no parent tensor expresses. The refused run is
sealed as an `infra` row and its log is kept; **the refusal is a result about the
mechanism's shape, not an accident.**

### 4.2 What repair bought

| route | k | nnz | stored (ADDED) | held-out gap **after** | explained out of sample | drift before → after |
|---|---:|---:|---:|---:|---:|---|
| `X01.T5.K256.L05` | 256 | 262,144 | 525,313 | **0.719555** | 28.0% | 0.540899 → **0.531210** |
| `X01.T5.K256.L06` | 256 | 262,144 | 525,313 | 0.896034 | 10.4% | 0.531210 → **0.516254** |
| `X01.T5.K256.L07` | 256 | 262,144 | 525,313 | 0.900164 | 10.0% | 0.516254 → **0.511490** |
| `X01.T5.K256.L13` | 256 | 262,144 | 525,313 | **1.014878** | **0.0%** | 0.511490 → **0.517480** |

```
folded child (= COMP.MA.MXO4)      0.703396x   drift 0.540899
+ four corrective routes           0.712544x   drift 0.517480
drift bought back                              0.023419
storage spent                                  2,101,252 parameters (0.009148x)
EXCHANGE RATE                      0.011146 drift per MILLION stored parameters
```

**Repair works, is real, and is far too expensive to matter.** Every number in
that table is out of sample, and three of them say the same thing:

1. **The gap is mostly not linear in the block's own input.** The best route —
   at `L05`, the shallowest fold, where the block's own approximation error
   dominates — explains **28%** of the block's residual gap on rows it never saw.
   By `L06` and `L07` that falls to 10%.
2. **At a block that was never folded, the route explains nothing and does
   harm.** `L13` was included deliberately, to test repair *at the end of the
   stack* rather than at a fold: its held-out gap after the route is **1.014878**
   — the route made the gap **larger** out of sample — and the composed drift went
   **up** by 0.005990. That is a clean, sealed instance of overfitting a
   correction to a residual that has no linear structure left in it, and it is
   the reason a repair pass must be measured out of sample and never in.
3. **The exchange rate is the verdict.** M2 bought 0.023419 of drift for
   2,101,252 parameters. **M6 bought 0.051296 of drift for 12,950** — twice the
   drift for one one-hundred-sixty-second of the storage:

```
M2 corrective routes   0.011146 drift per million stored
M6 lane shopping       3.961    drift per million stored     ~355x
```

**Repair does not revive `COMP.MA.MXO4`.** It moves it from 0.540899 to
0.517480 — past `COMP.MA.MXO2`'s wreckage, nowhere near `COMP.MA.MXO3`'s
0.461810 — and it does it at a *higher* net ratio (0.712544x against 0.703396x),
so the repaired child is strictly dominated by the shopped child `COMP.MA.MXP2`,
which sits at 0.703409x and 0.489603 drift. **A rung that spends routes must beat
0.729103x net and it does; it must also be coherent, and it is not.**

### 4.3 What the exchange rate means for 0.5x, in one line

To carry `COMP.MA.MXO2` (0.497739x, drift 0.814086) down to the empirical alive
line of 0.461810 would require buying back **0.352276** of drift. At M2's
measured rate that is **31.6 million** corrective parameters — **0.1376x of the
parent**, which would put the "0.5x" child at 0.635x and make the exercise
pointless. And that estimate is *optimistic*: it assumes the rate holds, when
the four routes measured show it decaying with depth and going negative at an
unfolded block.

**M2 is measured, positive, and closed on economics rather than on failure.**

## 5. The frontier, and the two lines it can be read against

```
COMP.MA.MXP2   0.703409x   drift 0.489603   6/15   per-child dead   PER-LANE impaired
COMP.MA.MXP1   0.712544x   drift 0.517480   7/15   per-child dead   PER-LANE impaired
COMP.MA.MXO3   0.729103x   drift 0.461810   8/15   per-child dead   per-lane alive-degraded  (wave 11)
```

**Under ADDENDUM 8 as registered, the frontier moved to 0.703409x.** §8.2 makes
`impaired` a success band — *"a 6/15 or 7/15 child whose output is coherent,
on-prompt language is a successful fold-init"* — and §8.3 abolished the 8/15 rung
floor and defined the frontier as **the deepest net stored ratio at which output
is still coherent**. `mxp_addendum8.py` computes it from the sealed per-lane
verdict and prints `deepest coherent: COMP.MA.MXP2 at 0.703409x`.

**Under the 8/15 alive band — the line the drift budget is calibrated on, and the
line wave 11 reported its frontier against — neither wave-12 child qualifies and
`COMP.MA.MXO3` at 0.729103x remains the frontier.**

Both sentences are true about different lines and this note refuses to collapse
them. The human-review package puts the choice to Cyril with the verbatim text of
both children in front of him, which is what that package is for.

**One caveat, stated so it cannot be read as a thumb on the scale.** Wave-11's
children seal `unstated` per-lane readings — the sealed-per-lane machinery did
not exist when they were reviewed — so the calculator falls back to their
per-child verdicts. Wave 12 did **not** re-derive per-lane readings for them,
because re-reading earlier children in the wave that wants the frontier is
exactly the move `NOTES-MX-O` §7a refused. It is a wave-13 item.

**Distance to 0.5x from the new deepest coherent rung**: 0.203409x = 46,721,900
parameters = **7.91 more SwiGLU lanes**, against a per-lane drift allowance of
0.038037 and a measured cheapest lane of 0.034129 *as the third lane* — a
marginal that §3.3's own saturation result says will rise, not fall, as the stack
fills.

## 6. What wave 12 says, in the campaign's own terms

| mechanism | registered as | result |
|---|---|---|
| **M2** corrective routes | **ADDENDUM 10**, written and committed before the first cell | **Measured, positive, and closed on economics.** Repair moved `COMP.MA.MXO4`'s identical fold set from 5/15 to **7/15** and its drift from 0.540899 to 0.517480, at **0.011146 drift per million stored parameters** — 355x worse than M6's exchange rate. At an unfolded block it explained **zero** of the gap out of sample and made drift worse. |
| **M6** drift shopping | wave-12 directive | **The one that paid, again.** Sixteen variants priced against one fixed base; the autopilot lane cost 2.2x the cheapest; the shopped rebuild took the same storage from 5/15 to 6/15, halved lambada perplexity, and cost **12,950** parameters. |
| **M5** head merge | ADDENDUM 9 §9.4 | **Not reached, for the second wave running.** |
| ADDENDUM 9 §7a | `NOTES-MX-O` open item | **Closed as specified**: the per-lane verdict is a sealed field computed from reviewer-supplied per-lane readings, with four refusals and 12 tests. |

**Why M5 was not reached, as a budget statement rather than an excuse.** M2 cost
two full children (one refused by the collector, one re-run) plus the ADDENDUM 10
schema work; M6 cost a probe and a child; the per-lane sealing cost a commit and
a test file. A Procrustes head merge is not a knob — it needs a new kernel, a
registry entry, an identity gate, tests, a solo cell and a review before it may
enter a stack, and starting it with what was left would have produced an unrun
kernel rather than a measurement. **The design is written down in §8 so wave 13
starts from a specification rather than from this sentence.**

## 6a. What was NOT measured, stated as a gap

* **`COMP.MA.MXP1` was not benchmarked.** Its corrective routes are forward
  hooks, not `state_dict` entries, so `mxp_materialize_child.py` cannot write an
  HF child that carries them without custom modeling code (`trust_remote_code`).
  Every other child in the wave was benchmarked. **A repaired child's
  intelligence-per-compression is therefore unmeasured on the continuous axis**,
  and any claim about repair's benchmark effect would be a claim about a model
  lm-eval never loaded.
* **The greedy path past the third lane.** The shopping probe prices the *third*
  lane against the frontier base. It was not re-run against `COMP.MA.MXP2`'s own
  base to price the fourth, so the compounding rate is inferred from
  `COMP.MA.MXO2`'s trace rather than measured on a healthy stack.
* **Repair at smaller `k`.** Every route ran at k=256 to give the mechanism its
  best shot. The exchange rate at k=32 or k=64 is unmeasured; it can only be
  *better* per parameter if the first 32 atoms carry most of the explained
  fraction, and the realized supports — every route used its full budget,
  262,144 = 1024 x 256 nonzeros, with nothing stopped early — suggest they do not.

## 7. Estate

* **Ledger.** 442 rows sha `f2d945512a13a0917f24ac0b40bdb8fe9db0ea1cb1740d6b6f5c80d832167f00`
  before; backed up **twice** (`backup/ledger.jsonl.pre-wave12-20260812T180543`
  and `wave12/ledger.jsonl.pre-merge-backup`), both verified byte-identical to
  the original by sha256 **and** by `cmp` *before* the merge; merged **exactly
  once** behind a `.MERGED_ONCE` marker; **447 rows** after, sha
  `a1c6b924238cfecf5f48a49baafcb8e77d2dc32ff071f89526c5e2e57845355a`. The merger
  verified the prior bytes are a prefix, and the check was repeated independently
  afterwards (695,128 bytes before, 705,287 after, 10,159 appended).
* **Seals.** **442 of 442** resolvable rows verify, **0 fail**. Five rows do not
  resolve: the same five as wave 11 (three `CAL.C01.T5.X4`, one
  `C06.T2.S4.L12` a6review instance, one `COMP.MC.MXO0` infra row).
* **The seal verifier's search roots nearly repeated wave 11's own defect.** The
  wave10-to-wave11 rewrite cost wave 11 eleven rows and a 427/442 reading
  (`NOTES-MX-O` §8); the wave11-to-wave12 rewrite did the same thing here and was
  caught by the count — 14 unresolved instead of 5 — before anything was
  reported. Wave 12's verifier now keeps **every** earlier root and says why in
  its docstring.
* **Wave-12 rows.** 5, all `sm86 NVIDIA GeForce RTX 3090`, all receipt schema v2:
  one `infra` (the collector's refusal), two composed children, two ADDENDUM 6
  review instances carrying ADDENDUM 9 per-lane blocks.
* **Retention.** Nothing deleted: two reviewed children's tensor sets are kept
  (362,811,893 and 362,811,829 bytes, plus 50,370,709 bytes of corrective
  routes), the `infra` row died before its first tensor save, and the
  materialized HF child was deleted after its benchmark.
  `wave12/pruned-tensors-manifest.json` records digests, sizes and the
  re-derivation path.
* **Disk.** Server `/` at 4.5-4.9 GB free throughout; every heavy artifact under
  `/mnt/ai_data/neural-forge-matrix/wave12/` (684 MB). The five conditioned
  captures ran in `/dev/shm`, were deleted as their rows were read, and the tmpfs
  was cleared at wave end (126 GB free, 4.0K used).
* **Deployments.** Three, and the reason each exists is on the record:
  `4d8774f6` (ADDENDUM 10 + the corrective class), `dcfe22ff` (the sealed
  per-lane verdict, grading only), `f753b9e7` (the corrective write moved into
  `mlp.down`). `COMP.MA.MXP2` was generated under `4d8774f6`; `COMP.MA.MXP1`
  under `f753b9e7`; every verdict was sealed under `f753b9e7`. The frozen review
  script is **byte-identical** across `4d8774f6` and `f753b9e7`
  (`4a7a956605c4e205…`), which is why `COMP.MA.MXP2`'s dossier is unaffected by
  the fix.

## 8. M5's specification, so wave 13 does not start from zero

The `cannot-express-4x` verdict on C10 was about head **drop**, where k/v are
stored whole. A Procrustes merge is a different claim and the arithmetic works on
this parent: six attention blocks of `1,048,576 + 2 x 524,288 + 1,048,576 =
3,145,728`, i.e. 16 query heads and 8 kv heads at head_dim 64.

Merge the 16 q heads into `g` groups and the 8 kv heads into `g/2`, storing per
group one `64 x 1024` q basis and one `1024 x 64` o basis, per kv group one
`64 x 1024` k and one `64 x 1024` v, and per head a `64 x 64` orthogonal
Procrustes alignment `R_h` — which is what makes it a *merge* and not a drop:

```
stored(g) = g*(64*1024 + 1024*64) + (g/2)*2*(64*1024) + 24*(64*64)
g = 4  ->  884,736   = 0.281x   (over the 4x bar)
g = 3  ->  688,128   = 0.219x   (under it)
```

So `g = 3` is the deepest admissible merge and `g = 4` is the one to measure
against it. The installed tensors stay parent-shaped — `q_h = R_h Q_g`,
`o_h = O_g R_h^T` — so no config change, no architecture edit, and the frozen
review reconstitutes it exactly as it reconstitutes every other child. The
storage claim is over the *stored* basis-plus-alignment set, which is the
campaign's existing convention (`net_storage`'s coverage rule). Solve each `R_h`
in closed form by SVD of `Q_h Q_g^T`. **Attention is 11.6% of this body and about
a third of a large model's, which is the only reason it is still on the list.**

## 9. The test result

Run on the Windows checkout of this wave's own HEAD with waves 9-11's invocation,
same file order, plus wave 12's two new files appended:

```
python -m pytest tests/test_matrix_composition.py tests/test_matrix_cell_receipt.py \
                 tests/test_run_lfm2_matrix_cell.py tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py tests/test_matrix_technique_new_axes.py \
                 tests/test_matrix_addendum6.py tests/test_matrix_addendum7.py \
                 tests/test_matrix_technique_token_embedding.py \
                 tests/test_matrix_composition_als.py \
                 tests/test_matrix_corrective_routes.py \
                 tests/test_matrix_addendum9_per_lane.py -q
# -> 289 passed, 1 skipped
```

The wave's own two files are 27 + 12 = 39 of those. **Wave 11's ten known
failures do not appear here because this is the worktree, not a `git archive` —
the four-case `blank_receipt` `compute_architecture` family and its relative are
fixed in the Codex lane's uncommitted files, which an archive cannot see.** A
whole-tree run of the same checkout collects and executes without error —
`4360 passed, 15 skipped, 1 failed` — and the one failure,
`test_hf_lfm2_child_replay.py::test_single_pass_rejects_class_level_lm_head_forward_spoof_before_forward`,
is in the Codex lane and untouched by this wave.

One order-dependence worth naming because it will bite the next wave that runs a
subset: `test_matrix_technique_registry.py::test_t1_refuses_components_it_does_not_model`
passes in the campaign's invocation and fails when its file runs without
`tests/test_run_lfm2_matrix_cell.py` ahead of it, because it asserts the dispatch
table routes `C04` past T1's reference kernel and that registration happens on
import of the conv-FIR module. It is one of the four collateral victims
`NOTES-MX-O` §9 names.

## 0. The five sentences

1. **Closed-form repair works, and it is 355x too expensive to matter.**
   ADDENDUM 10's additive routes moved `COMP.MA.MXO4`'s identical fold set from
   **5/15 to 7/15** and its drift from 0.540899 to 0.517480 — a real result for a
   mechanism wave 11 could not even express — at **0.011146 drift per million
   stored parameters**, against lane shopping's **3.961**.
2. **Repair only has purchase where a fold's own error is still linear in the
   block's input, and that purchase collapses with depth.** Out of sample: 28.0%
   of the block gap explained at L05, 10.4% at L06, 10.0% at L07, and at L13 — a
   block that was never folded — **0.0%, a held-out gap of 1.014878, and drift
   that went up**.
3. **The cheapest lane was never the next one in depth, and the campaign had been
   buying the wrong one.** Sixteen variants priced against one fixed base:
   `COMP.MA.MXO4`'s `L07` at T8/S4 cost +0.076198 and `L13` at T5/S4 costs
   **+0.034129** at the same 4x bound. Swapping them, for **12,950 parameters**,
   took the same storage from 5/15 to 6/15 and **halved lambada perplexity**.
4. **Two laws that looked general are not.** S8 beats S4 on short conv and
   **loses on every SwiGLU site measured**; and the "cheap deep lanes" of the 0.5x
   child were a saturation artifact — the same lanes cost an order of magnitude
   more on a healthy base, so a marginal pooled over stack positions ranks
   nothing.
5. **The frontier moved to 0.703409x under ADDENDUM 8 as written, and did not
   move under the 8/15 band it was last reported against.** Both are printed,
   neither is collapsed into the other, and 0.5x is still **7.91 lanes** away from
   the deepest coherent child — against a per-lane allowance that this wave's own
   saturation result says gets harder, not easier, from here.

## 10. What wave 13 should measure

1. **Shop the fourth lane against `COMP.MA.MXP2`'s own base, then the fifth
   against that.** The probe costs ~40 s per candidate and the compounding rate is
   the single number that decides whether a greedy shopped ladder reaches 0.65x.
   It is currently inferred, not measured.
2. **Combine the two mechanisms in one child.** Repair was measured on the
   *autopilot* fold set. `COMP.MA.MXP2` plus routes at L05/L06/L13 is one run and
   it is the only rung that carries both of wave 12's positives.
3. **Seal per-lane readings for the wave-10 and wave-11 children.** They read
   `unstated`, which makes cross-wave frontier comparisons rest on two different
   readings. Do it in a wave that is not simultaneously trying to move the
   frontier.
4. **M5, from §8's specification.** `g = 3` at 0.219x and `g = 4` at 0.281x, solo
   cell first, into a stack only if the solo is parent-band.
5. **Make a repaired child materializable**, or state that repair is a research
   instrument and not a shippable fold. Right now `COMP.MA.MXP1` cannot be
   benchmarked, which means the campaign cannot rank it on the continuous axis it
   uses to break ties.
