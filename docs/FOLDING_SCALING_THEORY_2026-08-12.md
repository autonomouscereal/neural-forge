# Folding scaling theory — what transfers off LFM2.5-230M

**2026-08-12, waves 11-12 of the matrix campaign.** Cyril's instruction: *"[Cyril quote withheld]"*

Everything below is stated **scale-invariantly**: as a fraction, a ratio, a rule
about a component *class*, or a formula in `(V, L, d)` — never as a row count, a
rank, or a parameter total that only means something on a 229M model. Every
claim carries the sealed receipt it comes from. Where a claim is measured on one
model and *cannot* be shown to transfer, it says so.

The model everything was measured on: **LFM2.5-230M**, 229,693,184 unique
parameters, 14 blocks (8 short-conv, 6 full-attention), hidden `d = 1024`,
SwiGLU intermediate 2,560, tied 65,536-row token table.

---

## 0. The five sentences

1. **Coherence is bought at the token interface, and it is bought with
   *coverage of token occurrences*, not with reconstruction error.** The best
   table reconstruction the campaign measured produced the *worse* child at the
   same budget (`COMP.MA.MXN5` 8/15 against `COMP.MA.MXN2` 10/15,
   `NOTES-MX-N` §6).
2. **The interface is a shrinking prize.** Its share of a model is
   `V / (V + L·c·d)`, which is 29.2% here and ~1.6% at 70B scale. Folding it 4x
   removes 21.9% of this model and would remove ~1.2% of that one.
3. **So the body is the whole game at scale, and the body's law is
   additive interference:** each independently-solved body component costs the
   composed child roughly 3 review cases, and interface damage × body damage
   *multiplies* (9/15 body + 10/15 route = 6/15 together, `NOTES-MX-N` §5).
4. **Local fit quality does not predict composed behavior. Re-solving an
   existing component cannot repair composition damage; an ADDITIVE tensor the
   parent lacks can, but only where the damage is still linear in that block's
   own input.** Wave 11 aimed every component directly at the parent's own
   activations, improved five of six local objectives, and made the composed
   child worse (§5). Wave 12's corrective routes explained **28%** of a folded
   block's residual gap out of sample at the shallowest fold, **10%** two blocks
   deeper, and **0%** — with a held-out gap above 1.0 and drift that rose — at a
   block that was never folded (§5a).
5. **Coherence is a drift budget, and the way to spend it is to price every
   candidate against the base you will actually install it on.** `review =
   12.96 − 13.16 × drift` over 55 sealed children; a *pooled* per-site marginal
   ranks nothing, because drift saturates and a lane measured deep in a dying
   stack looks free. Priced against the real base, the lane the campaign was
   about to buy cost **2.2x** the cheapest one, and swapping them — for 12,950
   parameters — halved lambada perplexity at identical storage (§3.4a, §5a).
   Nothing about the token interface transfers to a large model except the rule
   that some verbatim interface must exist.

---

## 1. The interface rule

### 1.1 What actually decides

A folded token interface keeps a **verbatim head** — a set of rows stored exactly
— and reaches the rest of the vocabulary with a coded tail (sparse route or
low-rank). The measured decision rule:

> **Coherence tracks the fraction of *prompt-token occurrences* the verbatim
> head covers, not the fraction of vocabulary rows it stores, and not the
> reconstruction error of the coded tail.**

Sealed evidence, one budget, one mechanism, one variable (`NOTES-MX-N` §6):

| arm | head | coverage of prompt-token occurrences | tail | rel. Frobenius | tail row cosine | review |
|---|---:|---:|---|---:|---:|---:|
| `COMP.MA.MXN5` | 4,096 rows (6.25% of vocab) | **66.80%** | k=101 | **0.5761** (best) | **0.8078** (best) | 8/15 |
| `COMP.MA.MXN2` | 12,288 rows (18.75% of vocab) | **80.99%** | k=38 | 0.6005 | 0.7476 | **10/15** |

Same storage (0.780411x against 0.780518x net), same solver, same evidence, same
GPU. The arm that reconstructs the table better is the worse child. What moved
is where the parameters went: **into covering occurrences, not into fidelity**.

**Scale-invariant statement.** Spend the interface budget to maximize
`P(token occurrence is verbatim)` under a natural-text frequency distribution.
On LFM2 the alive floor observed was **~81% occurrence coverage**; 66.8% was
measurably not enough. Because token frequency is Zipf-like, occurrence coverage
is a strongly concave function of head size, so this is a *cheap* thing to buy
at any vocabulary size: covering ~80% of occurrences costs a small fraction of
the rows, and the fraction *falls* as vocabularies grow, because large
vocabularies spend their extra rows on rare tokens.

### 1.2 Sparse tail beats low-rank tail, everywhere measured

At identical storage, at four different (head, level) points, on reconstruction,
on install drift, and on all seven benchmarks (`NOTES-MX-N` §3, §7):

| level | head | low-rank tail rel. Frobenius | sparse tail rel. Frobenius |
|---|---:|---:|---:|
| S8 | 4,096 | 0.8619 | **0.7397** |
| S4 | 4,096 | 0.7644 | **0.5761** |
| S4 | 12,288 | 0.7897 | **0.6005** |
| S8 | 8,000 | 0.9051 | **0.8523** |

and the head-to-head child comparison, one variable changed: install drift
0.498567 → **0.207468**, review 9/15 → **10/15**, lambada perplexity 7,337 →
**230** (`NOTES-MX-N` §4.1). **A token table is not low-rank. It is sparse in a
dictionary of its own frequent rows.** That is a statement about what word
embeddings *are*, and it should transfer.

### 1.3 Count the indices

A sparse code you did not store the atom ids for is a code nobody can decode.
A9's published 0.734512x stored counted its coefficients and not its atom ids or
its CSR offsets; honestly counted it is **0.743217x**, and A11 is **0.760056x**
rather than 0.742932x (`NOTES-MX-N` §1.2). Any route, at any scale, costs
`values + ids + offsets`. The convention is conservative (one index = one
parameter, where 12 bits would do) and is applied to every wave-11 number too.

---

## 2. The interface is a shrinking prize

Let `V` be vocabulary rows, `d` hidden width, `L` blocks, and `c` the block's
parameter coefficient (`body = L·c·d²`; LFM2 measures `c = 11.08`, a standard
transformer block is ~12).

```
interface share  =  V·d / (V·d + L·c·d²)  =  V / (V + L·c·d)
```

| model class | V | L | d | interface share | max removal by folding the interface at 4x |
|---|---:|---:|---:|---:|---:|
| LFM2.5-230M (measured) | 65,536 | 14 | 1,024 | **29.2%** | **21.9%** |
| 7B class | 32,000 | 32 | 4,096 | 2.0% | 1.5% |
| 70B class | 128,256 | 80 | 8,192 | 1.6% | 1.2% |
| 70B class, untied head | 128,256 | 80 | 8,192 | 3.2% | 2.4% |

The share falls as `1/(L·d)` while `V` saturates around 128k–256k. **Every
interface result in §1 is, at 70B scale, a 1–2% story.** They are still worth
having — they are nearly free and they protect coherence — but a campaign that
wants a 2x model at scale gets ~99% of the parameters it needs from the body.

Correspondingly, the body's share rises: 70.8% here, ~98.4% at 70B. **Folding
the whole body at 4x removes 53.1% of this model and 73.8% of that one.** The
0.5x target is a body target at every scale, and *more* of one as models grow.

The body's own composition, measured exactly on this parent (safetensors header,
one line per class), is the other half of the map:

| class | parameters | share of body |
|---|---:|---:|
| SwiGLU MLP (`w1`+`w3`+`w2`, 14 blocks) | 110,100,480 | **67.72%** |
| short conv (`in_proj` 25,165,824 + taps 24,576 + `out_proj` 8,388,608) | 33,579,008 | 20.65% |
| attention (`q`,`k`,`v`,`o`, 6 blocks) | 18,874,368 | 11.61% |
| block norms + final norm | 30,464 | 0.02% |

**MLP is two-thirds of the body here and is two-thirds of the body in every
dense transformer** (the SwiGLU triple is `3·(8/3)d² = 8d²` against attention's
`4d²`, before GQA shrinks the attention side further). A campaign that can fold
MLP safely can reach 0.5x; a campaign that cannot, cannot, at any scale.

---

## 3. The composition laws

These are the transferable results. Each is measured across many children on one
model; each is stated as a rule about *composition*, not about LFM2.

### 3.1 Additive interference (bodies)

**Each independently-solved body component costs the composed child roughly
three review cases relative to its isolation reading, and the costs add.**

| child | components | net | review | isolation readings of the same cells |
|---|---:|---:|---:|---|
| `COMP.MA.MXL5` | 4 (L05 SwiGLU lane + attn_k@L12) | 0.972579x | 10/15 | 12, 12, 12, 12 |
| `COMP.MA.MXL11` | 6 (adds C11/C13@L04) | 0.955444x | 9/15 | + 11, 12 |
| `COMP.MA.MXN3` | 7 (adds the folded interface) | 0.735962x | 6/15 | + 10 |

The mechanism is not mysterious and it is the thing wave 11 attacks: **every one
of those components was solved against the parent's activations, and then the
other components moved them.**

### 3.2 Interface × body multiplies

`COMP.MA.MXN2` (interface alone) is 10/15. `COMP.MA.MXL11`'s body alone is 9/15.
Together they are **6/15** (`NOTES-MX-N` §5). Not 9, not 10 — below both. A
folded interface changes the *input distribution* every body fold was solved
against, so interface damage and body damage compose multiplicatively rather
than additively. **Fold the interface last, or fold it first and re-solve the
body against it** — which is exactly what ADDENDUM 9's mode M-C does.

### 3.3 Lane concentration

Three MLP folds **in one block** (`COMP.MA.MXL5`, 10/15) beat the same three
folds spread across blocks by three review cases (`NOTES-MX-L` §"the same three
folds"). The block-exclusivity rule that generalizes it: **a block hosts at most
one folded lane, where a lane is a side of the operator/FFN-norm boundary.**
Measured: C05 alone at L05 is 10/15; C05 + mlp_gate at L05 is 7/15; C05 +
mlp_down at L05 is 6/15; **mlp_gate + mlp_down at L05 — both on the same side —
is 10/15** (`NOTES-MX-K` §2.1).

### 3.4 Deeper is sometimes safer, and mid-depth is safest

Site tolerance is not monotone in depth and must be measured per component
class. On LFM2, MLP folds live at mid-depth (L04/L05) and die at the ends
(`C13.T8.S4`: L04 8/15, L05 12/15 isolation, L06 7/15, L03 6/15, L00 5/15, L13
7/15 — `NOTES-MX-K` §3.1). `attn_k` lives at L12, the *last* attention block
(`NOTES-MX-K` §3.2).

### 3.4a Marginal drift is a property of (site, technique, **and position**)

**Wave 12, and this one invalidates a whole class of allocator.** A lane's
marginal drift measured deep in an already-damaged stack does **not** transfer to
a healthy one. `COMP.MA.MXO2`'s eleven lanes, in install order, cost

```
0.3659  0.1139  0.0421  0.0687  0.0685  0.0530  0.0518  0.0297  0.0054  0.0105  -0.0061
```

— the last lane *reduced* drift — and the same lanes priced against the **frontier
child's** base cost 0.0729 (L11) and 0.0440 (L13), an order of magnitude more.
Drift saturates: there is only so much of a model left to break, so a late lane
in a dying child looks free and is not.

**Consequence for anyone building a fold allocator at scale: price every
candidate against the base you will actually install it on, and re-price after
every acceptance.** A per-site marginal pooled over positions — which is what a
ledger of past runs gives you — is biased toward whatever positions those runs
happened to use. The measurement is cheap and does not need a sealed cell: in an
independently-solved composition (parent-evidence M-A), a component's solved
tensor does not depend on what else is installed, so the composed child is a
function of the **set**; one process can install a base once and then price
candidate after candidate by installing, measuring, and writing the parent's own
tensors back (verified: sixteen of sixteen restorations reproduced the base drift
exactly).

### 3.4b The min-drift variant is per site, and the storage level does not
generalize across component classes

Wave 12 priced sixteen `(lane, technique, level)` variants against one base:

* **The better kernel is per site, not global.** Sparse-OMP routes (T5) beat
  Gram relation matching (T8) by 22% of drift at one site, tie at a second, and
  **lose by 33% at a third**, all at the same 4x storage bound.
* **The deeper storage level reverses between component classes.** Wave 8
  measured that short-conv at S8 (1/8) drifts *less* than the same conv at S4
  (1/4). On the SwiGLU lane this reverses at **every** site measured: S8 removes
  ~17% more and costs 15-21% *more* drift. **"Fold it harder and it hurts less"
  is a short-conv property, not a law.**

The transferable procedure is therefore: for each candidate site, solve every
admissible (technique, level) whose stored form meets the bound, install it on
the real base, measure composed drift, and take the minimum. Nothing about which
kernel wins transfers; the *procedure* does.

### 3.5 Reconstruction does not predict behavior; conditioning anti-predicts

Three independent measurements:

* the token-interface arm with the best reconstruction is the worse child (§1.1);
* one-pass sequential conditioning (M-B) produces the same band, worse composed
  diagnostics, and 27x the wall clock (`NOTES-MX-J` §4);
* July's shared-basis fold reached mean held-out per-layer cosine **0.9625**
  (worst layer 0.9452) and its composed trajectory score fell to 0.3876 with
  zero reliable answers retained
  (`docs/DYNAMIC_LFM2_RESULTS_2026-07-16.md`, run seal `160bfc7f…`).

**Corollary for anyone scaling this work: do not select folds by per-layer
reconstruction error.** Select them by composed behavior, or by a drift model
calibrated on composed behavior.

### 3.6 Repetition is the collapse signal, and it is lane-scoped

The operational death test — *remove the repetitions and ask whether a response
remains* — is what separates a fold-init worth retraining from a dead tensor
(ADDENDUM 8 §8.5). ADDENDUM 9 §9.1 scopes it **per lane**: degeneracy confined
to identifiable lanes is charged to those lanes, general degeneracy is death.

---

## 4. Per-component-class safety table

What folds, where, at what level, on this architecture — stated by class so it
can be re-measured on another one. "4x" is `S4` (stored ≤0.25x of the component's
own parent tensor), "8x" is `S8`.

| class | component | 4x in isolation | 4x in composition | 8x | notes |
|---|---|---|---|---|---|
| MLP gate (`w1`) | C11 | **yes**, 10–11/15 | **yes** at L05 (10/15 in MXK5, the smallest single install measured: drift 0.168344) | 9/15 at L05 | the safest body fold known |
| MLP up (`w3`) | C12 | yes, 9/15 | yes, with C11 in the same lane | untested alone | |
| MLP down (`w2`) | C13 | **yes**, 12/15 at L05 | yes at L04/L05; 5–7/15 at L00/L03/L06/L13 | only inside a dead all-S8 child | strongest site dependence measured |
| short conv out (`C05`) | C05 | yes, 12/15 | 10/15 at S4, **13/15 at S8** — one case *above* the same-process parent | **better than its own S4 arm** | the one component where 8x beat 4x |
| conv in (`C03`) | C03 | measured | not composed at depth | — | writes `operator.output`; conflicts with an MLP fold in the same block |
| attn q (`C06`) | C06 | 10/15 | one site only | — | |
| attn k (`C07`) | C07 | 11–12/15 at L12 | **yes at L12**; 8/15 as a second fold at L08; **4/15 as a third at L10 — five lanes become single-token loops** | untested alone | three stacked attention folds collapse the decoder |
| attn v (`C08`) | C08 | 8/15 | degrades at S4 | — | |
| attn out (`C09`) | C09 | 9/15 | degrades at S4 | — | |
| heads (`C10`) | C10 | **cannot-express-4x** | — | — | for head *drop*, where k/v are stored whole; head *merge* is untried |
| norms (`C14`) | C14 | n/a (diagonal) | — | — | no level axis |
| token table (`C01`) | C01 | see §1 | 10/15 alive at 0.780518x | 7/15 dead at S8 | the only 4x fold that pays 29% of this model |

**Transferable shape of the table, independent of LFM2:** MLP mass folds; the
last attention block's key projection folds; value/output projections and
stacked attention folds do not; head-count contraction has never been expressed
at 4x; the safest sites are mid-depth; and one component (a conv writer) folds
*better* at 8x than at 4x, which means the level ladder must be measured per
component and never assumed monotone.

---

## 5. What wave 11's mechanisms added

The wall this wave attacked is §3.1: **every body fold in the campaign was
solved against activations the rest of the composition then invalidated.** Four
mechanisms were registered in ADDENDUM 9 and three of them now have answers.

### 5.0 The results, first

| mechanism | result | the number |
|---|---|---|
| **M1** ALS joint co-solve | **negative** | five of six components improved on their own held-out objective; composed drift went 0.487722 → 0.505337, **3.6% worse** |
| **M3** cross-layer shared basis | **closed by falsification** | shared loses to independent in 55 of 56 layer-comparisons at equal storage, even at 2.98x rank uplift |
| **M4** drift-budget allocation | **the one that paid** | `review = 12.9554 − 13.1626 × drift`, R² 0.624, n = 55; frontier moved 0.754811x → **0.729103x** |
| **M2** body corrective routes | not expressible without a schema change | a corrective tensor the parent does not have needs a new component id in a frozen table |
| **M5** head merge | not reached | — |

**The transferable statement from M1 is the sharpest thing wave 11 produced.**
A per-component closed form aimed *directly at the parent's own activations*,
solved against the inputs the composed child actually delivers, is still a
**local** objective: it minimizes one write's distance to the parent's write,
and the composed model is sensitive to the *direction* that residual points in
as it traverses the remaining depth. Two components whose residuals partially
cancelled can each be improved into residuals that no longer cancel. Seeing that
requires the downstream Jacobian, which is a gradient. **So: no gradient-free,
per-component objective can repair composition damage, and the campaign's whole
family of closed-form folds is bounded by that.** A12's law is not about stale
evidence; it is about locality.

There is also a structural result about the *schedule*: in a feed-forward
composition, a **depth-ordered ALS pass converges in one sweep, exactly**,
because nothing downstream of a component can move that component's input rows
and its own re-solve cannot either. Alternation buys nothing here. What could
make it pay is an objective that couples downstream state back into an upstream
solve — which is end-to-end, which is gradients.

### 5.1 The mechanisms as registered

* **M1 — ALS joint co-solve (mode M-C).** Install M-A, then repeatedly re-solve
  each component's own closed form against a capture taken through the current
  composed child, with the **parent's own write rows** as the target so the step
  cancels drift rather than re-fitting to it. Zero gradients. **Measured; see
  above.**
* **M2 — body corrective routes.** Sparse routes at block outputs to cancel
  composed residual drift; storage counted in full including atom ids and CSR
  offsets. **Not expressible under receipt schema v2** — every folded component
  binds to parent `state_dict` keys, and an additive corrective tensor has none.
* **M3 — cross-layer shared basis. Closed by falsification, in weight space.**
  The July estate had already rejected three behavioral forms of it (whole-FFN
  sharing removed 11.13–12.50% at whole-output retention 0.125–0.333;
  coupled-neuron banking rejected at 15% and 20% with mean coupled-weight cosine
  **0.014** between activation-correlated pairs; a single shared residual output
  basis at 15% rejected at per-layer cosine 0.9625 —
  `docs/DYNAMIC_LFM2_RESULTS_2026-07-16.md`). Wave 11 measured the *ceiling*
  instead of another instance, at the S4 budget, with the shared basis solved by
  alternating closed form on the real parent tensors
  (`reports/matrix-campaign-20260810/run/mxo/mxo_sharing_probe.py`):

  | component | rank at equal storage | independent | shared | layers sharing wins |
  |---|---|---:|---:|---:|
  | `w1` 2560x1024 x14 | 182 → 248 (1.363x) | **0.7219** | 0.7875 | 0 / 14 |
  | `w3` 2560x1024 x14 | 182 → 248 (1.363x) | **0.7650** | 0.8090 | 0 / 14 |
  | `w2` 1024x2560 x14 | 182 → **543 (2.984x)** | **0.7615** | 0.7946 | 1 / 14 |
  | `q_proj` 1024x1024 x6 | 128 → 219 (1.711x) | **0.5709** | 0.6619 | 0 / 6 |

  **Even a 2.984x rank uplift at equal storage loses to independent
  truncation, in every layer, in the metric most favourable to sharing.**
  Corresponding layers do not live in a shared subspace. The transferable part
  is the ceiling: sharing buys at most `1 + n/m` in rank (1.4x for a SwiGLU
  writer, 2x for a square attention projection), and on this model that was not
  enough to pay for the per-layer subspace freedom it costs.
* **M4 — drift-budget allocation.** Fit drift → review-damage per component
  class from the sealed children and water-fill a global budget instead of
  guessing fold-sets. **This is the mechanism that moved the frontier**, and it
  is the most transferable thing in this document because it is a *method*, not
  a result:

  ```
  review(useful-or-partial, out of 15)  =  12.9554  -  13.1626 x global drift
                                           R2 = 0.6243,  n = 55 sealed children
  ```

  Two consequences that should re-derive on any model: **coherence is a drift
  budget** (here 0.3765 of terminal relative-L2 against a copy_exact control, at
  the alive band), and **a fold-set should be chosen by spending that budget on
  the components with the lowest measured marginal drift**, not by depth order
  or by reconstruction error. On LFM2 that meant adding the L06 SwiGLU lane
  rather than the next one down, and it bought 0.0257x of the parent at no
  review cost (`COMP.MA.MXN4` 8/15 at 0.754811x → `COMP.MA.MXO3` 8/15 at
  **0.729103x**). One lane further, and the same budget model correctly predicted
  the collapse (`COMP.MA.MXO4`, 5/15 at 0.703396x).

**The storage arithmetic of sharing, stated scale-invariantly**, because it
bounds what M3 could ever buy. For a component of shape `(m, n)` replicated
across `L` layers, independent rank-`r` factors store `r(m+n)` each; a shared
`n`-side basis with per-layer coefficients stores `r_s(m + n/L)` each. At equal
storage:

```
r_s / r_ind  =  (m + n) / (m + n/L)   ->   1 + n/m   as L -> infinity
```

Sharing is a bounded win, not an unbounded one, and the bound is set by the
aspect ratio of the weight, not by the number of layers: 1.4x for a SwiGLU gate
or up projection, 2x for a square attention projection, 3.5x for a SwiGLU down
projection shared on its wide side. Wave 11 measured all four and **none of them
paid** — see the table above.

---

## 5a. What wave 12's mechanisms added

| mechanism | result | the number |
|---|---|---|
| **M2** additive corrective routes | **positive, and closed on economics** | +2 review cases and −0.023419 drift on an identical fold set, at **0.011146 drift per million stored parameters** |
| **M6** min-drift lane shopping | **the one that paid** | −0.051296 drift at the same storage, for **12,950** parameters: **3.961 drift per million**, ~355x M2's rate |

### 5a.1 Closed-form repair is real, local, and priced out

An **additive** route — a tensor the parent does not have, solved in closed form
from the composed child's own rows against the parent's rows at the same corpus
positions — **can** cancel composed drift. That is the first positive result the
campaign has for repairing composition damage, and it matters because wave 11's
M1 established that *re-solving an existing component* cannot (§5).

The transferable statement is about **where** it works, and it is a decay law:

```
fraction of a block's residual gap a linear route explains, OUT OF SAMPLE
  at the shallowest folded block          28.0%
  one block deeper                        10.4%
  two blocks deeper                       10.0%
  at a block that was never folded         0.0%   (held-out gap 1.0149: WORSE)
```

**A corrective route has purchase only where a fold's own approximation error is
still linear in that block's input.** Accumulated upstream drift is not: by the
time it has passed through another block's nonlinearity it has no linear
structure left for the route to find, and the solve overfits — measurably, out of
sample, in the direction that makes the model worse.

Two consequences for anyone scaling this:

* **Measure a repair out of sample or do not measure it.** The in-sample fit at
  the unfolded block was fine; the held-out gap was 1.0149 and the composed drift
  went up. A repair pass graded on its own objective would have shipped it.
* **Price repair against the recipe.** Storage spent on corrective tensors buys
  drift at a rate; storage *not* spent on a better fold choice buys it at a much
  better one. On this model the ratio was 355x in favour of the recipe. Repair is
  the right instrument only when the fold set is already the best available.

### 5a.2 Marginal drift is a property of (site, technique, **and position**)

See §3.4a. The short form: a lane's cost depends on the base it is installed on,
drift saturates, and a per-site marginal pooled over the positions past runs
happened to use ranks nothing. Price every candidate against the base you will
actually install it on, and re-price after every acceptance. In an
independently-solved composition this costs one install and one drift measurement
per candidate — no sealed cell, no capture, ~40 s each on a 230M model — because
the composed child is a function of the **set** and the base restores exactly.

### 5a.3 A registry note that is really a scaling note

Expressing an additive corrective tensor required a **second component table**
disjoint from the frozen one, because every folded component in this campaign is
bound to the parent `state_dict` keys it replaces and a corrective route replaces
nothing. Anyone building a fold compiler will hit the same wall the moment they
try to add a tensor the parent lacks, and the cheap fix is the one taken here:
a disjoint id space, an optional receipt block, storage **added** rather than
swapped, and no 4x claim — because there is nothing a corrective route is a
quarter *of*.

The second wall is the observability contract. A route attached at the wrong
module boundary broke the trace collector's own residual identity
(`layer_output == post_operator + mlp.down`) and made every subsequent capture
impossible. **If your evidence collector verifies structural identities — and it
should — then every new additive mechanism has to be attached inside an endpoint
those identities are written over, not beside it.**

---

## 6. What is not known, stated so nobody assumes it

* **Whether any of this holds on a different architecture.** Everything is one
  model. The claims are *shaped* to transfer; none has been re-measured on a
  second model.
* **Whether coherence at 0.5x is reachable by folding alone. Measured, and on
  this model it is not.** The 0.5x set exists and meets the storage target
  exactly — eleven whole SwiGLU lanes plus the token route, thirty-four
  components, every one at 4x, **0.497739x** — and it scores **0/15** with
  fourteen of sixteen lanes riding a single phrase to the token cap. The deepest
  coherent child is **0.703409x** (wave 12, `impaired`, 6/15) and the deepest
  `alive-degraded` one is **0.729103x**. The gap between those and 0.5x is the
  campaign's real open problem, and wave 12 measured that neither of the two
  remaining closed-form mechanisms closes it: repair buys drift at 1/355th of the
  rate a better recipe does, and a better recipe still leaves **7.91 lanes** to
  find at a per-lane allowance that saturation makes harder, not easier.
* **What the review surface measures beyond coherence.** A 15-prompt band with a
  ±2 reviewer-sensitivity interval (ADDENDUM 2) is a coarse instrument; the
  benchmark columns (arc_easy / hellaswag / piqa / winogrande / lambada /
  wikitext, `--limit 200`) are the continuous check, and they are partial
  evaluations by construction.
* **Whether ALS's fixed point is unique or ordering-dependent.** Coordinate
  descent on a non-convex composition has no such guarantee; the sweep order is
  pre-registered (depth-ascending) and the receipts record it.
