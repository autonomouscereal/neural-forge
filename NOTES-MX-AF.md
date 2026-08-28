# NOTES-MX-AF — wave 30: the field slate's top two bets are void, and the axis nobody measured moved four frontier rungs

Wave 30 ran the field-research slate against the two-surface instrument wave 29
validated. Two of the slate's Tier-1 bets died before a GPU was allocated, for
the price of arithmetic and one CPU pass each, and both died with a mechanism
rather than a null count. The budget they freed went to the axis the allocator
pointed at, which had never been measured, and that is where the wave's result
is.

**The one-line result.** `ARCH.V19CL1792` — vocabulary 19,456 rows composed on
`intermediate_size` 1792, no excision — reads **7/15 at 0.650769x**, and is
simultaneously the deepest child this campaign has sealed at ≥4/15, ≥5/15,
≥6/15 and ≥7/15. The previous deepest ≥7 sat at 0.691832x. The plan's **M2**
milestone ("first >4/15 between 0.685x and 0.49x") is met.

---

## 1. The emission mechanism wave 29 adopted is a coordinate-system artifact

`NOTES-MX-AE` §6 read a correlation off six children and adopted it as this
campaign's emission story: the parent carries exactly one massive-activation
dimension, coordinate **294**, and the only two children that lose it — `W720`
and `MM768` — are exactly the two whose generative margin collapses to a third
of the parent's. Field synthesis #2 was scheduled against it.

`mx29_emission.py` compares massive-activation dimensions **by index**. A width
fold rewrites the residual stream as `z = E^T x` — `fold_writer` is literally
`E^T W` — so the child's coordinate 294 is not the parent's coordinate 294.
`W720` and `MM768` are the only two children in that table whose stream is
rotated, so they are the only two an index comparison *can* fail on, and it
failed on both. The correlation is with **"is a rotating fold"**, not with
"loses the massive channel".

Two invariant quantities replace it, and they say different things.

**Is the direction still representable?** `||E^T e_294||^2`, free from the
sealed basis artifacts, no GPU:

| basis | width | direction kept | direction lost | sealed review |
|---|---:|---:|---:|---:|
| `E-reader-pca-960` | 960 | 0.999478 | 0.000522 | — |
| `E-reader-pca-896` | 896 | 0.998695 | 0.001305 | 4/15 |
| `E-reader-pca-768` | 768 | 0.996690 | 0.003310 | 3/15 |
| `E-reader-pca-720` | 720 | 0.996005 | 0.003994 | 0/15 |
| `E-reader-pca-704` | 704 | 0.995683 | 0.004317 | 0/15 |
| `E-reader-pca-640` | 640 | 0.994482 | 0.005518 | 0/15 |
| `E-reader-pca-512` | 512 | 0.991717 | 0.008283 | 0/15 |

Every basis on the ladder keeps the direction to three decimals. **A carve-out
that exempts the massive dimension from the truncation can buy at most 0.8% of
one direction at width 512 and 0.4% at 720.** The field synthesis's #2(a) is
dead on arithmetic, and the width cliff cannot move because of it.

**Is the direction still written?** The child's stream lifted back through its
own basis, `x_hat = E z`, measured in parent coordinates:

| child | width | representable | energy on e_294 retained | child peakedness | rank of 294 | dims >20x median | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `W896` | 896 | 0.998695 | **0.1137** | 38.0 | 2 | 4 | 4/15 |
| `W768` | 768 | 0.996690 | **0.0435** | 18.2 | 6 | 5 | 3/15 |
| `W720` | 720 | 0.996005 | **0.0405** | 19.0 | 7 | 5 | 0/15 |
| `W704` | 704 | 0.995683 | **0.0416** | 19.1 | 6 | 6 | 0/15 |
| `W512` | 512 | 0.991717 | **0.0151** | 8.5 | 22 | 7 | 0/15 |

The parent's peakedness on that coordinate is **165.8**.

So the massive channel *is* destroyed — by 89% to 98.5% — but **at every width,
including the one that reads 4/15**, and **not by truncation**: the subspace
holds the direction essentially perfectly and the folded model stops driving it.
It is a write failure, not a representation failure, and a basis carve-out
protects the thing that was never broken.

It also does not track the review. `W768` and `W720` write 0.0435 and 0.0405 of
the parent's energy on that channel — a 7% difference — and read 3/15 and 0/15.
Whatever separates those two children, this is not it.

One observation kept because it is odd and cheap: **every width child's top
residual coordinate in parent coordinates is 767**, at five widths, through five
different bases. The fold does not merely damp the parent's massive channel; it
concentrates a weaker one somewhere else, and somewhere else is the same place
every time.

`mx30_emission_invariant.py` **refused** `W736` and `MM768`: their bases were
sealed by wave 21, which is not on the share, and the script will not measure a
rotated stream by index. That is the refusal working, and it is why those two
rows are absent rather than wrong.

## 2. RETOK-BODY is void at every K the campaign has ever built

The plan's #2 — the top mechanism bet, deferred by wave 29 because the evidence
path was not parameterised — costs one new capture to solve the `CL896` body
against `ARCH.V19`'s retokenized stream, on the theory that a composed child
reads a token sequence its body was never solved for. Wave 29's
`mx29_retok.py` computed that stream correctly and **wrote only its metadata**:
the positions were counted, the identity gate passed, and the ids themselves
were dropped when the process exited. Nothing could be solved against them at
any price, which is the whole of why the item sat unexecuted for two waves.

Wave 30 seals the stream (`stream_sha256`, re-sealed on read) and
`mx30_build.py --evidence` consumes it and names it in every receipt. Then it
measured what the stream actually is:

> The V19 tokenizer's segmentation of all twenty-four evidence batches is
> **identical to the parent's own, at every one of 4,181 positions**, differing
> only in what the ids are called.

Compared in one id space — child ids lifted through the kept-row order, because
comparing two id spaces compares the renaming. Zero differing positions.

`mx30_retokcensus.py` then ran the whole ladder:

| cell | keep rows | positions | vs parent | batches identical | positions differing | sealed review |
|---|---:|---:|---:|---:|---:|---:|
| V32 | 32,766 | 4,181 | 1.0000 | 24/24 | **0** | 11/15 |
| V22 | 22,526 | 4,181 | 1.0000 | 24/24 | **0** | 13/15 |
| V20 | 20,478 | 4,181 | 1.0000 | 24/24 | **0** | 13/15 |
| V19 | 19,454 | 4,181 | 1.0000 | 24/24 | **0** | 14/15 |
| V18 | 18,430 | 4,181 | 1.0000 | 24/24 | **0** | 13/15 |
| V17 | 17,406 | 4,181 | 1.0000 | 24/24 | **0** | 11/15 |
| V14 | 14,334 | 4,181 | 1.0000 | 24/24 | **0** | 9/15 |
| V12 | 12,286 | 4,181 | 1.0000 | 24/24 | **0** | 11/15 |
| V10 | 10,238 | 4,181 | 1.0000 | 24/24 | **0** | 10/15 |
| V9 | 9,214 | 4,181 | 1.0000 | 24/24 | **0** | 9/15 |
| V8 | 8,190 | 4,181 | 1.0000 | 24/24 | **0** | 4/15 |

**The keep-set is segmentation-complete on this corpus at every rung, including
the 4/15 cliff child.** A capture through the V19 interface is the parent's
capture with the rows renamed, the re-solve is the sealed solve, and the plan's
#2 has nothing to find at any K. The evidence path is parameterised, receipted
and available; it is the hypothesis that is empty.

**One thing the census found on the way that is not about vocabulary.** Decode
the sealed 6,335-position trace and re-encode it with the **parent's own**
tokenizer and you get 4,181 positions — 0.66x. The sealed ids are not the
canonical tokenization of their own text. Every arm here, parent control
included, runs on the re-encode, so it is a property of the corpus and not of
any fold; it is written down because it is what made the first reading of the
retokenized stream look like a 34% sequence shift, and because it bounds what a
retokenized capture could ever have measured.

**The question the null hands over, and its answer.** A folded-out row cannot be
*emitted* — the table is tied, so the row leaves the unembedding too. So
`mx30_emitcoverage.py` asks the output-side question against the only text that
matters for it: the parent's own sealed answers to the review's own prompts.

| cell | keep rows | token coverage | tokens lost | type coverage | types lost | sealed review |
|---|---:|---:|---:|---:|---:|---:|
| V32 | 32,766 | 0.94005 | 195 | 0.88691 | 114 | 11/15 |
| V22 | 22,526 | 0.91454 | 278 | 0.82738 | 174 | 13/15 |
| V20 | 20,478 | 0.90778 | 300 | 0.80853 | 193 | 13/15 |
| V19 | 19,454 | 0.90563 | 307 | 0.80258 | 199 | 14/15 |
| V18 | 18,430 | 0.89517 | 341 | 0.78869 | 213 | 13/15 |
| V17 | 17,406 | 0.88657 | 369 | 0.76885 | 233 | 11/15 |
| V14 | 14,334 | 0.86597 | 436 | 0.73115 | 271 | 9/15 |
| V12 | 12,286 | 0.85183 | 482 | 0.70536 | 297 | 11/15 |
| V10 | 10,238 | 0.83492 | 537 | 0.67559 | 327 | 10/15 |
| V9 | 9,214 | **0.81402** | 605 | 0.63591 | 367 | **9/15** |
| V8 | 8,190 | **0.80049** | 649 | 0.60813 | 395 | **4/15** |

Coverage falls **smoothly** — no cliff in it anywhere — and the review's cliff
sits exactly where coverage crosses **0.81**. The campaign's "~81% coverage
floor" has been an asserted number since wave 23; it is now located on a
measured quantity, and it gives the vocabulary axis a gate that costs a
tokenizer pass and no GPU: keep token coverage of the parent's own emissions
above 0.81.

## 3. The allocator, and the hole it found

`mx30_allocator.py` fits the per-axis curves on the ledger rather than taking
the field's word for any of them (npj AI, Feb 2026: per-axis critical
thresholds, stable under combination, so orthogonal axes each kept
pre-transition compound safely).

**The arithmetic gate first, because it caught a real error.** Every axis on
this parent removes a closed-form number of parameters, and the allocator
refuses to emit a split whose count it cannot reproduce. The obvious accounting
for an attention block — q/k/v/o at 3,145,728, the SwiGLU trio at 7,864,320, two
block norms at 2,048 — gives 11,012,096, and that is **wrong by 128**, which is
`q_layernorm` and `k_layernorm`: LFM2 normalises inside the attention block over
`head_dim` 64, twice. The sketch reproduces the sealed **ratio** to six decimals
and misses the sealed **count** on all six attention excisions. Gating on counts
rather than ratios is the only reason it was found. Final: **54 cells checked, 0
mismatched.**

**What the fit says.**

* **vocabulary** — pre-transition floor 0.748919x (`ARCH.V9`, 9/15),
  post-transition ceiling 0.744354x (`ARCH.V8`, 4/15). A **0.004565**-wide
  bracket: five cases across 1,024 rows. The sharpest transition in the campaign,
  and §2 now has its mechanism.
* **depth** — *not parameterised by ratio at all.* Single excisions occupy two
  ratios (0.947479x conv, 0.952057x attention) and eleven review levels, 13/15 to
  0/15, decided entirely by which block. Free sites {13, 7, 9, 11, 1}, dead sites
  {12, 2, 5, 0}. Reporting a critical ratio for this axis would be reporting the
  spread of the site table as a threshold.
* **mlp** and **width** — **every sealed rung is already past transition.**
  `ARCH.CL896` reads 4/15 at 0.688431x and `ARCH.W896` reads 4/15 at 0.859023x,
  against a 12/15 parent. And the mlp axis has a **1,664-channel hole** in it:
  between the parent's 2,560 and the only rung ever built, nobody has measured
  anything.

**Fitted pairwise interference**, least squares over 25 sealed composed children
(MAE 2.156 cases):

| pair | cases |
|---|---:|
| depth × vocabulary | **4.44** |
| depth × mlp | 1.39 |
| mlp × vocabulary | **1.17** |

That number is the recommendation. Depth-with-vocabulary — the composition every
deep child in this campaign is built from — costs nearly four times what
mlp-with-vocabulary costs, so the allocator **refused to spend a single
excision** and put the whole budget on the axis with the hole in it.

| target | split | stored | ratio | predicted | **observed** |
|---|---|---:|---:|---:|---:|
| 0.65x | V19 + no excision + `intermediate_size` 1792 | 149,477,120 | 0.650769 | 9.14 | **7/15** |
| 0.55x | V19 + no excision + `intermediate_size` 1280 | 127,457,024 | 0.554901 | 6.68 | **1/15** |

Both counts were emitted to the parameter before either was built. The 0.55x
split carried the allocator's own **EXTRAPOLATES-INTO-THE-GAP** flag, because
1280 sits below every measurement the axis had; it was built anyway, precisely to
fill the hole, and it missed by 5.68 cases. The 0.65x split, which does not
extrapolate past the first new rung, missed by 2.14.

## 4. The MLP axis, measured, and the head-to-head nobody had run

| cell | `intermediate_size` | ratio | review | charged lanes |
|---|---:|---:|---:|---:|
| PARENT | 2,560 | 1.000000 | 12/15 | — |
| `ARCH.CL1792` | 1,792 | 0.856199 | **7/15** | 3/16 |
| `ARCH.CL1280` | 1,280 | 0.760331 | **4/15** | 8/16 |
| `ARCH.CL896` (w15) | 896 | 0.688431 | 4/15 | 3/16 |
| `ARCH.CL640` (w15) | 640 | 0.640497 | 0/15 | — |

The axis falls fastest between 1792 and 1280 and then flattens onto the sealed
896 reading — power-law-then-plateau, which is the shape the field's model
predicts and not the cliff the vocabulary axis has.

**The comparison that had never been made.** At effectively the same ratio:

| axis | cell | ratio | review |
|---|---|---:|---:|
| mlp | `ARCH.CL1792` | 0.856199 | **7/15** |
| width | `ARCH.W896` | 0.859023 | 4/15 |
| mlp (maxmin basis) | `ARCH.MM896` | 0.859023 | 2/15 |

Three cases apart at three decimals of matched ratio. Twelve waves of this
campaign's parameter budget went into the width axis; it is the worse axis, and
until this wave there was no rung on the other one shallow enough to say so.

**And the composition price.** `ARCH.V19` costs, on the body it is composed onto:

| body | body review | composed | composed review | vocabulary's price |
|---|---:|---|---:|---:|
| `CL1792` | 7/15 | `V19CL1792` | 7/15 | **0** |
| `CL1280` | 4/15 | `V19CL1280` | 1/15 | −3 |
| `CL896` | 4/15 | `V19CL896` | 2/15 | −2 |
| any excision base | — | — | — | −3 to −4 (every wave 26–28) |

**Zero.** The 14/15 summit vocabulary composed onto the 1792 body costs nothing
at all, on a body that is itself pre-transition. That is the allocator's thesis
landing: keep every axis out of its own collapse and the composition tax
collapses with it.

## 5. The reader metric cancels, and then it does not pay

Field synthesis #6 (LFQ, ICML 2026; IO-SVD) says to solve the final block
against parent logits. `mx30_logitsolve.py` did, by a Sylvester ridge that
**asserts at startup** it reproduces `mx30_build.prior_anchored` at `M = I`
(max difference 4.4e-16 — an A/B whose control is not the sealed control is not
an A/B).

| | held-out relL2 | reader-metric SSE | stream SSE | λ | update scale |
|---|---:|---:|---:|---:|---:|
| logit-metric arm | 0.200287 | 325.5163 | 396.3423 | 1e-6 | 0.6280 |
| stream control | 0.200310 | 325.5198 | 396.4361 | 1e-6 | 0.6281 |

The weight moved by **0.34%**, and then:

> **All sixteen review lanes are byte-identical.** Every `response_sha256` of
> `ARCH.CL1792LG` equals `ARCH.CL1792`'s.

The benchmark column separates them by 0.01 of wiki word-perplexity (192.85 vs
192.84) and by nothing else. This is not a weak effect; it is algebra. For a
full-column-rank design and a vanishing anchor, `argmin ||X W^T − Y||²_M` is
`(X^T X)^{-1} X^T Y` for **every** positive-definite `M` — the metric cancels
out of an unconstrained writer solve, and the 0.34% is entirely the ridge
anchor, which is measured in the Frobenius norm and not in `M` at all. LFQ's and
IO-SVD's gains come from **rank-constrained** steps, where the metric decides
which subspace survives.

On this axis that step is the coreset **selection**, and there the factorisation
is exact: dropping channel `j` unrepaired writes `phi[:,j] w2[:,j]^T` into the
block's output, whose squared reader cost is `||phi[:,j]||²` times
`w2[:,j]^T M w2[:,j]`. So the reader-aware coreset is the sealed pivot on
columns rescaled by one scalar each — one line, closed form, solve unchanged. It
swapped 130 to 233 channels of 1,792 per site, with the scale spread widening
from 0.42–1.40 at block 0 to 0.08–2.52 at block 13, exactly where the metric is
exact rather than propagated.

| | stream selection | reader-metric selection |
|---|---:|---:|
| solo, 0.856199x | `CL1792` **7/15** | `CL1792M` **8/15** |
| composed, 0.650769x | `V19CL1792` **7/15** | `V19CL1792M` **6/15** |

Plus one at the shallow rung, minus one at the deep one, at byte-identical
storage on both. **Zero cases give or take one, with the two rungs disagreeing
in sign** — and both margins are one-case margins, so the concordance rider
applies and the extension surface adjudicates them:

| pair | sealed 15 | extension 30 | second surface says |
|---|---|---|---|
| `CL1792M` vs `CL1792` | 8 vs 7 (+1) | 7 vs 10 (**−3**) | **inverted** |
| `V19CL1792M` vs `V19CL1792` | 6 vs 7 (−1) | 9 vs 9 (**0**) | **no difference** |

The bank refuses the difference in both directions, which is the cleanest form
the answer could take: it is not that the reader-metric coreset wins a little at
one rung and loses a little at another, it is that neither surface can find a
difference to attribute. Findings #5 and #6 are both answered on this
instrument — the reader metric cancels out of the solve as a matter of algebra,
and where it does not cancel, it does not pay.

## 6. The case swap is not carried by a small coordinate set, measured directly

Wave 29 left `BP64` — 64 of 1,024 terminal-diagonal coordinates reproducing the
entire capability swap — beside `mx29_listset.py`'s finding that the top 24
coordinates carry only 15.8% of the attribution. Wave 30 measured the object
those two readings disagree about.

The terminal diagonal is the last thing before the reader, `z = W_head (RMSNorm(x) * d)`,
so it is **exactly linear and exactly separable in `d`**: the contribution of
coordinate `j` to the margin between the token a child actually emitted and its
strongest rival is `(W[r,j] − W[v,j]) · x_hat_j · Δd_j`, closed form, no
gradient, no ablation sweep. The reconstruction `W_head @ (x_hat * d)` was gated
against the model's own logits at **2.9e-07 to 5.5e-07** before a single
coordinate was ranked.

| lane | continuation tokens | mean margin under V19X13 | under G01 | positions destroyed | top-24 share of \|contribution\| |
|---|---:|---:|---:|---:|---:|
| `coding-deduplicate` | 230 | 4.356 | 4.118 | **9** | 0.191 |
| `uncertainty-insufficient-data` | 137 | 3.966 | 3.630 | 16 | 0.167 |
| `writing-concise-email` | 34 | 1.834 | 1.493 | 9 | 0.176 |
| `creative-continuity` | 43 | 1.853 | 1.884 | 6 | 0.183 |

558 of 1,024 coordinates are restore candidates. Two surgical children were
built — the sibling's diagonal whole, with the source's values returned only on
the top-ranked candidates:

| child | restored coordinates | ratio | review | `coding-deduplicate` |
|---|---:|---:|---:|---|
| `ARCH.SR24` | 24 | 0.742049 | 10/15 | `set()` |
| `ARCH.SR64` | 64 | 0.742049 | 10/15 | `set()` |

With wave 29's α ∈ {0.25, 0.50, 0.75} and its top-|Δ| partition, that is **seven
points on and around this segment, none above 11/15, and the LIST container held
at exactly one of them: the unmodified endpoint.** The attribution says why in
its own numbers — 9 of 230 positions, 19.1% in the top 24 — there is no small
set to find because the loss is not carried by a small set. Wave 29's
zero-dimensional reading stands, now with a coordinate-level measurement behind
it rather than a search that failed to find one.

## 7. The frontier

`ARCH.V19CL1792`, 149,477,120 parameters, **0.650769x**, **7/15**, **3 of 16
lanes charged** — tied with `ARCH.CL896` for the lowest charged-lane count
anywhere in the deep band, where 9 to 15 of 16 is the norm.

| threshold | previous deepest | wave 30 | moved by |
|---|---|---|---:|
| ≥7/15 | `V8X13GL01` / `V8X13GL03` 0.691832x | **`V19CL1792` 0.650769x** | 0.041063 |
| ≥6/15 | same | **`V19CL1792`** | 0.041063 |
| ≥5/15 | same | **`V19CL1792`** | 0.041063 |
| ≥4/15 | `V18X13X01G03` 0.684962x | **`V19CL1792`** | 0.034193 |

Four rungs, one child. Above it the frontier is unchanged: ≥12/15 at 0.790005x
(`ARCH.V18`), ≥11/15 at 0.742049x (`ARCH.V19X13G03`), ≥10/15 at 0.737484x
(`ARCH.V18X13`), ≥8/15 at 0.729103x (`COMP.MA.MXO3`, parent-shaped).

**Plan milestone M2 — "first >4/15 between 0.685x and 0.49x" — is met.** M1
(≥12/15 at ≤0.75x) and M3 (alive-degraded ≤0.51x) are not: `V19CL1792`'s
per-child verdict is `dead` and its per-lane reading is `impaired`, and the
readings disagree as they have on every child below 0.75x.

Benchmarks, `limit=200 seed=1234 batch=4 max_length=2048 float32`, parent row
re-measured this wave and matching the sealed reference exactly (0.5450 / 0.5050
/ 0.6800 / 0.5500 / 0.4600, ppl 24.39, wiki 81.80):

| model | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `CL1792` | 0.3850 | 0.4200 | 0.5800 | 0.5700 | 0.2500 | 201.92 | 192.84 |
| `CL1792LG` | 0.3850 | 0.4200 | 0.5800 | 0.5700 | 0.2500 | 201.92 | 192.85 |
| `CL1792M` | 0.4350 | 0.4600 | 0.5350 | 0.5600 | 0.2200 | 268.13 | 191.31 |
| `V19CL1792` | 0.3900 | 0.4000 | 0.5800 | 0.5450 | 0.2300 | 417.34 | 443.17 |
| `V19CL1792M` | 0.3650 | 0.3850 | 0.5400 | 0.5500 | 0.1850 | 573.95 | 440.80 |
| `SR64` | 0.4050 | 0.3800 | 0.6200 | 0.5100 | 0.3000 | 233.01 | 740.82 |
| `SR24` | 0.4050 | 0.3900 | 0.6050 | 0.4950 | 0.3000 | 229.11 | 718.91 |

The inversion is on this table too, for the ninth time: `SR64` and `SR24` read
10/15 with wiki word-perplexity near 730, and `V19CL1792` reads 7/15 at 443.

**Both surfaces on every child**, bank `c9d37004c901d757…` (the wave-29 pinned
bank, unchanged), parent run once:

| child | sealed 15 | extension 30 | extension % | charged lanes (30) | charged kinds |
|---|---:|---:|---:|---:|---|
| `SR64` | 10 | 13 | 43.3 | 3 | cap, loop, word-substitution |
| `SR24` | 10 | 9 | 30.0 | 3 | cap, loop, word-substitution |
| `CL1792M` | 8 | 7 | 23.3 | 5 | cap, echo, loop |
| `CL1792` | 7 | 10 | 33.3 | 4 | cap, echo, loop |
| `V19CL1792` | 7 | 9 | 30.0 | 3 | cap, echo, loop |
| `V19CL1792M` | 6 | 9 | 30.0 | 3 | cap, echo, loop |
| `CL1280` | 4 | 5 | 16.7 | 8 | cap, echo, loop |
| `V19CL1280` | 1 | 6 | 20.0 | 9 | cap, echo, loop |
| PARENT | — | 0 charged | — | 0 | — |

Spearman **0.6544** over eight children, against wave 29's 0.85 over sixteen —
the bank's own verdict line reads *"PARTIAL: rank order reproduces weakly — read
the table before leaning on any one-case margin"*, and §5 is exactly the place
that rider is spent. It does not touch the frontier claim, which is a
three-case margin over the deepest prior ≥4/15 and a 0.041-ratio-point move at
≥7/15, not a one-case margin. The one place the bank clearly over-credits is
`V19CL1280` at 6/30 against a sealed 1/15 — with **9 charged lanes**, the worst
of the wave, which is the rubric being fooled by lucky strings exactly as its
docstring says it can be.

## 8. What was ordered and not run

Stated plainly rather than folded into a summary.

* **B4, evidence-refresh discipline** (field #1, the slate's top-ranked finding)
  — **not run.** It needs a three-fold composition solved twice with per-state
  evidence caches, and the budget went to §1's correction and to the MLP ladder
  the allocator opened. The interference coefficient it was meant to move is now
  *measured* per axis pair in §3 (4.44 / 1.39 / 1.17), which is the number that
  experiment would have to beat, so it is better posed than it was.
* **B7** — the decision-margin analyzer landed inside §6 (per-coordinate margin
  attribution from cached logits, gated). **LACE shrinkage, EoRA-weighted
  truncation inside a width rebuild, and the paraphrase triplets did not run.**
  The triplets have now been authored and unrun for two waves.
* **B1(b), targeted diagonal repair on sink/massive channels** — not built, and
  deliberately: §1 removed its premise. The channels are not truncated away, and
  the one child that reads 4/15 loses as much of the channel as the ones that
  read 0/15.

## 9. Estate

Eight commits, eight deployments, every one verified file by file against the
commit's own blob digests: 5,544 files at the first and 5,548 at the rest, **0
content mismatches, 0 missing, 0 undeclared, 0 hardlinks** every time. Six of the
eight exist because something refused or something was found wrong, and the
estate's rule is that the fix ships as a commit and a verified deployment rather
than as an edit to a deployed file — a rule this wave broke once, by uploading a
census script directly into deployment `fb2c3768`. It was removed, the file
count re-checked at 5,544, and the script was run from scratch space until it
could ship in `74a8c871`.

**The wave's own rename broke three things and each is repaired in the first
commit rather than discovered later.** `mx30_verify_seals.py` lost
`wave29/cells` — wave 20's mistake and wave 25's, made a third time on the file
whose docstring exists to forbid it, and caught only by reading the list before
the file was ever run. `BASIS_SCHEMAS_ACCEPTED` lost wave 29's string; the list
is append-only. Four docstrings that recorded what wave 29 did were rewritten to
say wave 30 did it. The lesson waves 26 and 27 acted on — carry these files in an
explicit hand-edit set and never let the rename touch them — was not applied, and
should be next wave.

Ledger: **822 → 840 rows**, 18 appended, 0 superseded. Backup
`backup/ledger-pre-wave30.jsonl` at the pre-merge digest
`0d6ce61ff5dca9c5…`, `cmp` identical; merge applied once behind `.MERGED_ONCE`;
the authority's old bytes are a **byte prefix** of the new (1,399,367 of
1,433,106); new digest `e1de3fd9853a6bb2…`. Seal verification: **26 roots
audited against disk, 0 undeclared; 836 of 840 rows resolve and all 836 verify,
0 failures**, with the same four unresolved rows waves 25–29 carry
(`CAL.C01.T5.X4` ×3, `COMP.MC.MXO0`).

`sealq.sh` ran **once**, behind `.SEALQ_RAN`; the wave-29 guard that refuses a
`seal` call with anything other than eight arguments is unchanged and fired on
none of the nine. Parent lane calibration is unmoved. The parent's benchmark row
was re-measured and reproduces the sealed reference exactly, which is the third
calibration and the reason the child rows above are comparable to waves 8–29's.

Both sm86 3090s ran the wave. No ComfyUI contention: wave 29 restarted the two
H3 containers and they were still free-standing at 288 MiB and 279 MiB, so
nothing had to be checked, waited on, killed or restarted. `/mnt/ai_data` at
1.7 TB free, 64% used, unchanged across the wave.
