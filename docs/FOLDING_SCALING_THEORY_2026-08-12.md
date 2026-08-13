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
| heads (`C10`) | C10 | **drop: cannot-express-4x. merge (wave 13): expressible at 0.239583x** — 6/15 solo at L12 and L02, 4/15 at L08 | **8–13x the per-parameter drift of a SwiGLU lane** | merge expresses 0.114583x; unmeasured | head *drop* stores k/v whole and bottoms out near 0.5x; the *merge* is exact at its identity gate (≤8.7e-15) and behaviourally expensive because the heads are not rotations of a few shared bases (§5b) |
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

## 5b. What wave 13's mechanism added — M5, the head merge

The last unplayed fold class in the matrix, and the one with the strongest
*a priori* scaling argument: attention is **11.61%** of this body and roughly a
third of a standard dense transformer's, so a mechanism worth 1% here could be
worth 8% at 70B. Wave 13 built it and measured it.

### 5b.1 The mechanism, and why it can express 4x when head drop cannot

`C10` owns all four attention projections. Head **drop** stores `k_proj` and
`v_proj` whole, so its storage bottoms out near 0.5x and ADDENDUM 6 calls every
level of it `cannot-express-4x`. A **merge** stores `g` shared head bases plus
one orthogonal `D×D` alignment per head, and reconstructs every head
parent-shaped as `q_h = R_h Q_g`, `o_h = O_g R_hᵀ`:

```
stored = g_q*(D*hidden + hidden*D) + g_kv*2*(D*hidden) + (H + H_kv)*D*D
g_q=3, g_kv=2  ->  753,664 = 0.239583x     (S4, the qualifying config)
g_q=1, g_kv=1  ->  360,448 = 0.114583x     (S8 is expressible too)
```

The transferable part of the storage model is the **alignment tax**: the
per-head rotations are `(H + H_kv)·D²` and do not shrink with `g`. At `g_q=3`
they are 13% of what is stored; at `g_q=1` they are 27%. **A head merge on an
architecture with many small heads pays a larger fixed tax than one with few
wide heads**, because the tax scales with `H·D² = (H·D)·D = hidden·D`. Wide
heads are cheaper to align.

### 5b.2 The result: expressible, exact where checkable, and expensive

* **Identity gate** (one basis per head must reproduce the parent over all four
  projections): **6.7e-16 – 8.7e-15** across five sites. The mechanism is what
  it claims to be.
* **The evidence-anchored refit earns its place.** Re-solving the O group bases
  against the sealed attention write cuts held-out write error from 0.52–0.91
  (Procrustes bases alone) to **0.35–0.56**. Anyone implementing this should do
  the same: solve the alignment in weight space, then re-solve the one basis
  whose input and output are both observable against the observation.
* **Price:** 0.046–0.077 drift per million parameters removed, against a SwiGLU
  lane's 0.0058–0.0086 on the same base. **8–13× worse per parameter.**
* **Behaviour:** one merged attention block alone reads 6/15 (L12), 6/15 (L02),
  4/15 (L08) against a 12/15 parent, while removing 1.04% of the model.

### 5b.3 The diagnostic to run first on a larger model

Do not start by building a merged child. Start by running the kernel's
**clustering objective** on the target model's attention blocks and reading
`solve.query_family.objective_relative` and the partition it produces. On
LFM2.5-230M the partitions are degenerate — one large cluster plus singletons,
and a 7+1 kv split at *every* site — which says the heads are not rotations of a
few shared bases. That single number decides whether M5 is the largest fold-only
prize available at scale or a mechanism that is merely expressible.

**And note what it does not decide.** M5's price is a search-ordering fact. Wave
13 built the stacked children anyway, because the campaign's own outliers
(`COMP.MA.MXO3` alive at drift 0.4618 against a 0.3765 budget; wave 13's
`MXQ2`/`MXQ1` pair, where the lower-drift child has four looping lanes to the
other's one) say a drift ranking with R² = 0.62 is not a verdict.

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


---

## Wave-14 additions to the sealed physics

* **A storage-format ratio is not a smaller model.**  Every child through wave 13
  is parent-shaped by a refusal in the materializer.  Scaling claims must say
  which quantity they quote, because only three axes on this architecture are
  config-standard (`intermediate_size`, `head_dim`, depth) and conv width is not.
* **The simultaneous-installation fidelity floor.**  Measured on LFM2 at
  fourteen MLP sites at once, the boundary between a structured child and a
  degenerate one sits between per-site held-out write cosine **0.93** and
  **0.86**.  Single-lane fidelity in the sealed alive band does not survive being
  applied everywhere at once, and this is the first number the campaign has for
  how much tighter the simultaneous requirement is.  **The transferable
  prediction:** a width fold applied uniformly across depth needs materially
  higher per-site fidelity than the same fold applied at a few sites, and the
  gap is a few hundredths of cosine, not a factor.
* **Verbatim beats basis in gated feature space, and not in the conv context.**
  Matched-storage control, 40/42 vs 12/15.  Expected to transfer to any gated FFN
  and expected **not** to transfer to channel-mixing context streams.
* **Massive activations break share-of-total censuses.**  One unit at L05 carries
  97.15% of the layer's gated-output energy, which manufactured a false
  1,915-unit OFF certificate, a false six-fold Frobenius advantage, a kurtosis of
  2.5e6 and a passing HODLR-V coverage number.  Any scale-free screen must be
  scale-free; rank on cosine, not Frobenius, at such a site.
* **Terminal transfer is not linearly predictable from shallow residual.**
  Held-out R² of `post_layer[l] -> post_layer[13]` climbs smoothly 0.046 (L00) to
  0.874 (L12); only L10-L12 clear 0.5.  Terminal-metric methods are a late-depth
  tool on this model, and the 2,560-dim endpoint transfers are negative.

### Wave 15 additions

* **Conditioned evidence is not the fix for compositional death.** Re-solving
  every site against activations captured *through the partially built child* —
  the exact distribution the fold meets — improves the late-layer solves and the
  terminal drift and changes the behavioural verdict by nothing. Expect this to
  transfer: if a fold at a given width is dead composed, the reason is the width,
  not the evidence, and sequential conditioning is a refinement rather than a
  rescue.
* **One narrowed FFN moves the next layer's input by ~0.57 relative, and the
  drift then plateaus.** The residual stream's normalization bounds the
  accumulation. Damage from a per-site fold is immediate and local, not
  cumulative — which is why per-site fidelity measured on the parent's stream
  stays *approximately* right and still predicts nothing.
* **Trajectory targets are harmful even at zero cost.** Solving a writer against
  where the parent's stream should have been, rather than against the parent's map
  on the child's own input, damages the model at 4x width *and* at unchanged width
  with full capacity. Any method whose objective is a composed terminal residual
  (amortized correctors, terminal KKT re-solves, Ω-style steering) inherits this
  and should be priced against it first.
* **RoPE is a hard constraint on head-width folds, and it is expensive.** The
  rotation is block diagonal over frequency planes, so `q^T R_Δ k` decomposes
  exactly into per-plane terms and a smaller `head_dim` can only keep whole planes
  on a uniform stride (`theta' = theta^(s·d'/d)`). Score energy on this model is
  flat across the ladder: keeping 8 of 32 planes keeps 24% of it. A rank census
  taken at `Δ=0` overstates what a head-width fold can keep by **4x** here, and
  the same gap should be expected on any RoPE model.
* **A tied embedding is a floor, and it should be stated before a target is set.**
  65,536 × 1024 tied = 0.2922x of a 229.7M-parameter model. Any "standard-export"
  compression target below ~0.35x on such a checkpoint is a claim about the body
  alone and should be written that way.

## Wave-17 additions to the sealed physics — the depth census, closed

Wave 16 opened a behavioural single-block excision census and wave 17 closed it:
all fourteen blocks of LFM2.5-230M excised one at a time, each with every
surviving FFN writer re-solved at unchanged width, each read on the same frozen
16-lane surface. Then eight pairs at one parameter count. **This is the most
scale-transferable data the campaign has produced, because it is a map of the
model's own depth redundancy taken at the only level that decides anything.**

* **Depth redundancy is real, large, and unevenly distributed — and the
  distribution is not "by kind".** Removing one block of fourteen (5.3% of
  parameters) leaves a model that answers **13 of 15** prompts at one site and
  **0 of 15** at another. The cheap sites are the middle-to-late interior; the
  expensive ones are the **first block of each kind** (conv 0 → 0/15, attention 2
  → 0/15) and the **last attention block** (12 → 0/15). A census that treats
  "attention blocks" or "conv blocks" as a class gets this wrong in both
  directions: attention 6/8/10 read 8/7/7, inside the conv band, while conv 0 is
  the most expensive block in the model.
* **Rank blocks behaviourally or not at all.** Wave 14's absorbed-fraction census
  ranked block 5 first and block 13 second; behaviourally they are 6/15 and
  13/15. Terminal drift does not order the living children either (the 13/15
  child has the *second worst* drift of the eight). On a larger model, spend the
  budget on N single-block behavioural excisions before spending it on any
  activation-space redundancy statistic — the statistic will not survive.
* **The pair law: cost is set by DISTANCE, not by the components' solo scores.**
  Eight pairs at one parameter count, four distances, two anchors: distance 2
  reads 2, 2, 4 of 15; distance 4 reads 6, 6; distance 6 reads 6; distances 10
  and 12 read 7 and 7. Monotone with no inversions. And the two *best* single
  blocks (11/15 and 10/15, two apart) compose to **4/15**, while a 13/15 and a
  9/15 twelve apart compose to 7/15. **Adjacent blocks do related work and the
  survivors cannot absorb both.** The practical rule for a larger model: if you
  intend to remove k blocks, spread them as far apart as the depth allows and
  expect the *pair* interaction to dominate any per-block ranking.
* **Depth still stops at two, and the second step is fatal in every direction
  tried.** Two blocks 0.894958x is 2–7 of 15; three blocks is 0/15; two blocks
  plus a head merge is 0/15; two blocks plus a 40% narrower SwiGLU (0.730614x)
  echoes the prompt back verbatim. Nothing in seventeen waves has composed two
  independently-alive standard-shape changes without losing most of the score.
* **A better objective on an attention component does buy behaviour — once.**
  Merging 16 → 12 query heads by averaging two heads' weights in the exact
  per-plane sign gauge reads 5/15; solving the merged pair's *relation* instead
  (per RoPE plane, one rank-one matrix serving both parents, shared key across
  the GQA group, closed form through a 2×2 Hermitian generalised eigenproblem)
  reads **8/15** at the identical parameter count. This is the first component
  objective in the campaign whose improvement showed up behaviourally. It does
  not extend: 16 → 8 with the same solve is 0/15.
* **Two gauge facts that generalise to any RoPE + per-head-RMSNorm model.**
  (i) The only exact head-alignment gauge is the **per-plane sign** — the shared
  per-head norm weight is measurably not plane-symmetric, so a continuous plane
  rotation does not commute with it. (ii) A rank-one relation factorisation has
  its **own** per-plane unit-modulus gauge, and if it is left free the solver
  lands anywhere on that circle — here, on the exact reflection of the prior at
  −0.9998 post-norm cosine, which then destroys a prior-anchored ridge. Fix the
  factorisation's phase against whatever the ridge is anchored on.
* **A fit-side improvement is not a generalisation improvement, and on this
  problem it is anti-correlated.** The value/`out_proj` alternation (exact
  alternating least squares with the attention probabilities held fixed) improves
  the fit sum-of-squares by 16–29% at every site, lowers the held-out per-site
  cosine, and costs five of fifteen behavioural cases. λ chosen by a trust region
  does not see generalisation. Any two-sided solve on second moments of hidden
  states needs a validation split of its own.
* **Ill-conditioning is the default for these operators.** The two-sided Stein
  normal equations built from a second moment of hidden states and a Gram of
  `out_proj` columns needed **20,000** unpreconditioned CG iterations to reach
  the closed form on a *synthetic* problem two orders of magnitude better
  conditioned. Precondition with the single-term two-`eigh` solve, or a capped
  solver will silently return the prior.
* **The two measurement surfaces disagree in both directions, so quote both.**
  Wave 16 found a child the 15-prompt review ranked above what benchmarks
  supported; wave 17 found the reverse — one attention block excised
  (`ARCH.X06`, 4.6M parameters) costs **0.015 arc_easy and 1.6x lambada
  perplexity** while the child the review ranks five cases higher costs 4.4x, and
  a pair child reading 7/15 has a lambada perplexity of **2,870** (118x the
  parent's). A depth-reduction claim on a larger model that quotes only one
  surface is unfalsifiable in whichever direction it happens to be wrong.

## Wave-18 additions to the sealed physics — where the distance law ends, and one harness law

Wave 17 produced the campaign's one clean law (pair cost tracks block distance)
and one unexplained refusal. Wave 18 pushed the law to three bodies, built the
regularisation fix wave 17 named, and attributed the refusal. Two of the three
change what transfers.

* **The distance law is a two-body law. State it that way.** Spread still helps
  at three blocks — the tightest triple available (mutual distances 2, 2, 4) is
  0 of 15 and the widest (6, 6, 12) is 3 of 15 — but the magnitude collapses and
  the ordering inside the middle is not monotone in either the minimum or the
  total pairwise distance. At one block the best site keeps **13/15**, at two
  **7/15**, at three **3/15**: roughly **three to five review cases per block
  removed**, which is the additive-interference rule, with distance as a
  modulation on top of it rather than a replacement for it. The transferable
  planning rule for a larger model is: *spread the blocks you remove, and price
  the count, not the spread, as the budget.*
* **Any solve whose regulariser is chosen by a trust region needs a validation
  split of its own, and the fix is worth as much as the solve was.** Wave 17
  measured the value/`out_proj` alternation improving its own fit objective by
  16–29% while losing held-out cosine and five of fifteen behavioural cases.
  Selecting λ on an **internal split of the fit rows** — every third fit batch
  withheld from each rung's solve and used only to score it, the admitted rung
  with the lowest check objective re-solved on everything and installed, the
  trust region kept as a veto and stripped of its vote — recovers **four of the
  five cases** (3/15 → 7/15) at an identical parameter count and a benchmark row
  that matches the value-path-off child on all seven columns. The selected λ is
  **three to five orders of magnitude larger** than the trust region's pick, and
  the fit improvement it gives up (down to 0.004–0.09%) *was* the overfit. Note
  what this implies generally: the right answer was "move this component almost
  not at all", and only an out-of-sample criterion could discover that. A norm
  bound cannot, because staying near the prior and moving far from it can both
  be small in norm.
* **A better objective on a component still buys behaviour only once.** The same
  fixed solve at the axis's second step (16 → 8 query heads) is **0/15** with
  eight of sixteen lanes looping a refusal to the token cap — the same reading
  the unfixed solve got. Regularisation quality does not extend an axis past the
  step where it dies.
* **Composition prefers the component that is BEST, not the component that is
  FAR.** A head merge across all six attention blocks composed with a single
  block excision reads 7/15 when the excised block is the campaign's best single
  (one block from the nearest attention site), 5/15 for the next best, and 4/15
  for the highest-scoring single of all when it sits past the last attention
  block. The distance intuition inverts here, and this is the first two-axis
  child in the campaign that sits *at* its better component's level rather than
  far below both. **Distance is a within-axis law; across axes, rank by solo
  behaviour.**
* **A harness law, and it is not specific to this model.** Modern decoder
  implementations select per-layer behaviour by **position in a config list**,
  not by an attribute of the module object. In HF's Lfm2 the line is
  `layer_mask = causal_mask if config.layer_types[i] == "full_attention" else
  linear_attention`. Deleting a block from `model.model.layers` without deleting
  its entry from `config.layer_types` therefore re-points every later block at
  the wrong list entry, and a full-attention block that lands in a convolution's
  slot is handed the padding mask — `None` on an unpadded corpus — instead of the
  causal one. Measured here: **28.6% to 48.0% of the live model's attention mass
  on tokens that had not happened yet**, at every attention site, invisible in
  any loss, any norm, any drift statistic, and invisible in the exported
  checkpoint (which is written from a correctly re-derived config and is
  causal). **Any depth-reduction harness on any architecture with per-layer type
  dispatch — sliding-window/full alternation, MoE/dense alternation, Mamba/
  attention hybrids — must re-derive every per-layer config list when it deletes
  a layer, and should assert it.** The cheap universal assertion is the one used
  here: recompute one block's attention in fp64 from the live weights and compare
  against the module's own returned probabilities; or, simpler still, sum the
  module's attention mass strictly above the diagonal and require zero.
* **Corollary worth stating because the first estimate of the damage was wrong.**
  The blast radius of that defect was **one review case**, and only on children
  that both excise and solve. A pure excision runs no solve at all, so five
  rebuilds against the repaired harness came back **bit-identical** to the
  children they reproduce. When a harness bug is found late, measure its price by
  rebuilding rather than by reasoning about it: the reasoning here predicted a
  campaign-wide invalidation and the digests said the census was untouched.

## Wave-19 additions to the sealed physics — the residual axis, and the end of the standard-axis inventory

Wave 19 built `hidden_size`, the last config-standard axis nobody had moved, and
with it the standard-axis inventory for this architecture is **closed**:
`intermediate_size`, `head_dim`, `num_attention_heads`, `num_hidden_layers`,
`hidden_size`. Conv width is not a live field (`Lfm2ShortConv` hardwires
`hidden_size`), and `vocab_size` is a tokenizer change, not a fold.

* **The residual axis is the only one that reaches the tied embedding, and that
  is worth 29.2% of this model.** One config field takes the embedding to
  `0.75x` and every `d×d` and `d×ff` tensor toward `0.56x` at H=768. The
  arithmetic is exact and reproducible: with `head_dim` pinned, an LFM2-shaped
  child is `32*H² + 191541*H + 768` parameters. **Generalisation:** on any
  decoder whose embedding is tied and whose vocabulary is large relative to its
  depth, the residual width is the only standard axis with the *leverage* to
  reach half the parameters, and every other axis is bounded above by
  `1 − embedding_fraction`.
* **Pin `head_dim` explicitly whenever you narrow a residual.** Every HF
  attention module in this family computes `head_dim = getattr(config,
  "head_dim", hidden_size // num_attention_heads)`. Narrowing `hidden_size`
  without writing `head_dim` therefore silently *also* narrows the head and
  changes the RoPE frequency ladder — composing a second, differently-behaved
  axis into what was supposed to be one variable. The cost of pinning is small
  and computable (1.5 points of net ratio at 768 here); the cost of not pinning
  is an uninterpretable child.
* **A basis fold is exact against a local target, and the gauge is where the
  identity lives.** With one shared orthonormal `E` and the norm gauge
  `g_c = diag(E^T diag(g) E)`, readers fold as `sqrt(D/H)·W diag(g) E diag(1/g_c)`
  and writers as `E^T W`; the lift preserves the row norm, so the two RMSNorms
  differ by exactly `sqrt(D/H)` and nothing row-dependent is smuggled in. The
  gauge is not cosmetic: it is what makes the fold **bit-exact at `E = I`**,
  because a bf16 product divided by its own bf16 factor returns the other factor
  exactly in fp64. **Any width-folding harness should carry this gate and should
  compare bytes, not tolerances.**
* **A tied embedding has two roles and only one of them can absorb a gain.** As
  a writer it wants `E^T e_t`; as a reader it wants `W_emb diag(g_f) E`. The
  writer's form wins, so the final norm's diagonal is the only home for the final
  gain, and solving it there is closed form and worth a factor of three:
  relative logit SSE **0.54 → 0.17** at 768, where the transported gain alone
  leaves 0.54 and a single scalar leaves 0.41.
* **Channelwise-gated operators cannot be basis-folded and must be coreset.**
  A short-conv block computes `B*x` and `C*conv(·)` elementwise, so its internal
  channel space has no rotational freedom; its channels must be *selected*. This
  is the one lossy piece of an otherwise exact construction and it is where the
  per-site error concentrates (conv write cos 0.99 against attention and MLP at
  0.9999). **Generalisation:** any gated-elementwise operator — SSM/Mamba gates,
  GLU-style conv branches — is coreset-only under a width change.
* **THE CENTRAL RESULT, and it revises the wave-14 reading. Per-site install
  quality does not predict a width fold's behaviour.** At H=512 (0.463481x) every
  attention and MLP write lands at held-out cosine **0.9999** against its own
  local target and the conv writes at 0.9916 median — *better* than the living
  H=896 child's conv — and the child returns the **empty string** in half its
  lanes. Wave 14 located a cliff in per-site write cosine (0.93 → 0.86) and every
  wave since read per-site quality as the predictor. On this axis the predictor
  is elsewhere. The two quantities that move monotonically with the score are
  **the fraction of the parent's residual the subspace retains** (0.9705 → 0.8258
  from 896 to 512) and **the drift the child's own stream accumulates against
  `E^T r`** (0.267 → 0.557). A fold can be locally excellent everywhere and
  globally dead; measure the stream, not only the writes.
* **The residual stream of a trained decoder is not low-rank.** Pooled over all
  29 residual-write endpoints with each equalised, the top 768 of 1024 directions
  hold **0.9389** of the energy a reader sees (0.9729 of the raw energy) and the
  top 512 hold 0.8465 (0.9301). There is no knee anywhere in the spectrum. Do not
  plan a width fold around finding one; plan it around a price per direction.
* **The behavioural cliff on this axis sits at roughly 0.92 of retained read
  energy.** 896 (0.9627 retained, measured on held-out rows) is 4/15; 768
  (0.9193) is 3/15; 640 (0.8697) and 512 (0.8124) are 0/15. The transition is
  sharp — 3/15 to 0/15 across 128 dimensions — and the deeper failure is *quieter*
  than the shallower one: 640 returns the empty string in fifteen of sixteen
  lanes where 512 fills eight with punctuation.
* **The composition law survives contact with a new axis.** Width composed with
  the head merge is 2/15 and width composed with the census's best single
  excision is 1/15, from a 4/15 width child and a 13/15 excision. **The first
  composition step is survivable and the second is fatal** now holds across five
  axes, not four.
* **The excision ladder is complete and monotone: 13/15 single → 7/15 pair →
  3/15 triple → 0/15 quad.** The max-spread quad (blocks 1+5+9+13, 0.789916x)
  charges all sixteen lanes. Additive interference, run to its end.
* **The 15-case review surface is 89.6% one-dimensional.** Across 63 sealed
  children and 15 cases, a single strict difficulty order predicts **0.8963** of
  the cells. That is why per-child scores compare at all. The remaining tenth is
  per-axis and legible: the head merge preferentially keeps debugging
  (0.43 against a campaign 0.13), excision keeps instruction-following (0.59
  against 0.35), and the residual axis keeps summarisation (0.50, and 1.00 in
  both of its compositions) while taking context-retrieval to zero. **When
  reporting a fold class, report which cases it keeps, not only how many.**
* **A record-keeping law, found by sorting the ledger instead of reading the
  notes.** Three consecutive waves quoted "the frontier" as the deepest child
  with a non-zero behavioural score and named a child that was not it: sorted by
  net ratio, a wave-15 child (`intermediate_size` 896, conditioned-local) sits at
  **0.688431x with 4/15** and the child three waves promoted sits at 0.769439x
  with the same 4/15. The wave that built the deeper one declined to call it a
  frontier claim under an explicit eligibility rule — it is 2.86x per site and
  that wave's rule was the 4x class — and the next wave then promoted a child
  that is 1.67x per site, i.e. *less* eligible under the same rule. **A frontier
  is a query over a ledger, not a sentence in a note.** State the eligibility
  predicate with the number, and recompute it from the sealed rows every wave;
  a frontier that is inherited by quotation drifts silently, and this one drifted
  for three waves in the conservative direction, which is the direction nobody
  checks.

## Wave-20 additions to the sealed physics — the basis objective, and the shape of the energy law

Wave 20 held the width, the parameter count, the corpus, the solver and the code
fixed and moved only **which subspace of the residual the child keeps**. Five
children at `hidden_size` 768 are 165,978,624 parameters each. That makes the
review score a controlled measurement of the basis objective, which is the one
variable wave 19 never moved.

* **THE CENTRAL RESULT, and it bounds wave 19's law from the other side.
  Retained read energy is a CEILING, not a predictor.** Wave 19 found behaviour
  tracking the retained fraction and concluded the fraction was the variable. It
  is only half of one. Below roughly 0.90 of the read energy every child is dead
  (0.8953, 0.8967, 0.4780 — all 0/15, three different constructions). Above it
  the fraction stops predicting: a basis retaining **0.9178** scores **0/15**
  where one retaining **0.9193** scores **3/15**, at the same parameter count,
  the same accumulated drift (0.379 vs 0.375), the same per-site write cosines
  (median 0.9998), and a subspace that differs in **17 of 768 directions**.
  **Generalisation:** treat a retained-energy figure as a necessary condition to
  clear and never as an objective to optimise. Two subspaces of equal energy are
  not interchangeable, and the difference between them is not visible in any
  aggregate the fold computes about itself.
* **The currency is the energy a READER consumes, not the energy the stream
  carries, and this is now a measurement.** The two candidate second moments can
  be made to disagree: an evidence-PCA (raw) basis retains **more** raw energy
  than a reader-PCA basis (0.9636 vs 0.9522) and **less** read energy (0.9122 vs
  0.9193). It scores 1/15 against 3/15. **Generalisation:** on any architecture
  whose consumers see a normalised, gain-scaled residual — every pre-norm
  transformer — fit the basis to the post-norm, gain-weighted second moment.
  The raw stream's own principal directions are the wrong ones and the error is
  worth review cases, not decimal places.
* **Maximising reader energy per unit of raw energy is a trap, and it is the
  natural next idea.** The generalised eigenproblem `M_read v = mu M_raw v`
  selects the directions the gain amplifies most relative to what the stream
  carries — which are the directions the stream barely carries. Orthonormalised
  over its span at H=768 it retains **0.4780** of the read energy and **0.1862**
  of the raw, accumulates drift 3.749 against a living child's 0.375, and emits
  multilingual token salad in all sixteen lanes. **Generalisation:** ratio
  objectives over a spectrum with near-null directions concentrate on the null
  space. If a ratio objective is wanted, floor the denominator by the energy the
  behaviour actually needs, not by a numerical ridge.
* **Reweighting a basis toward the capabilities a fold kills does not recover
  them, and the reason is that it was never discarding them.** Rows implicated in
  the killed cases — selected from the evidence by closed-form token rules, never
  from the review prompts — are retained by the plain basis at the pooled rate or
  better (retrieval rows 0.9265 against a pooled 0.9193). **Generalisation:** the
  free version of this check is cheap on any model and any axis. Compute what
  your fold's basis retains on rows carrying the capability you lost *before*
  building a basis that claims to fix it; the answer is often that the subspace
  is not where the loss is.
* **A capability can be recovered by the SOLVER when it cannot be recovered by
  the basis.** Choosing the width writers' regularisation on an internal split of
  the fit rows — instead of taking the first λ a norm-based trust region admits —
  leaves the score unchanged (3/15) and changes *which* cases survive, recovering
  the context-retrieval case that the per-case attribution names as this axis's
  signature loss. **Generalisation:** report the capability mix, not only the
  count; and when a fold loses a specific capability, look at the estimator's
  variance before you look at its subspace.
* **The conditioned re-solve is worth one review case on a residual-width axis.**
  Installing the closed form and re-solving nothing reads 2/15 against 3/15. The
  closed-form projection `E^T W` is the L2-optimal write and is most of the
  child; the sequential solve is a correction. **Generalisation:** budget
  accordingly — on an axis where the fold is exact against a local target, the
  conditioning machinery buys far less than it does where the fold is a
  selection.
* **Weight TYING is not a cheap substitute for rank truncation, at least in
  channel space.** A block-Toeplitz tied ridge on `attn_o` — generically full
  rank, 245,824 stored numbers against 1,048,576 (**4.27x**) — reconstructs the
  parent's map **worse** (held-out cos 0.9108) than the sealed rank-128
  truncation at the same site storing *more* numbers (262,144, 4.00x, cos
  0.9434), and worse than all four other 4x techniques there (0.9349–0.9605).
  **Generalisation:** structured-matrix families borrowed from signal processing
  assume a translation structure the channel axis of a trained network does not
  have. Full rank is not the property that matters; agreement with the parent's
  map on the evidence is.
* **A frozen behavioural review at ONE site is below the instrument's
  resolution, and a distribution measurement is not.** A single tied tensor at
  one of fourteen blocks read near-parent on all sixteen review lanes while
  moving lambada accuracy from 0.4600 to 0.1800 and its perplexity from 24.39 to
  134.66. **Generalisation:** single-site cells need a perplexity or accuracy
  column, or they cannot be told from a null result; whole-model children need
  the review, because that is where the distribution stops discriminating.
* **A selection grid that lands on its own boundary has not converged.** A
  landmark-memory temperature selected on a fit split pinned to the smallest
  value offered on the first grid; widened, it converged interior on
  `head_dim^-0.5` — the parent's own attention scaling, recovered without being
  told. **Generalisation:** always report whether a selected hyperparameter is
  interior, and widen until it is; a boundary hit is the mechanism declining
  itself, or the grid being wrong, and those look identical in a receipt.

---

## Wave-21 additions to the sealed physics — the reader is not the unit, and the cliff has a slope

Wave 20 closed with the energy law bounded into a ceiling and one axis of the
basis objective unmoved: the objective is an **endpoint-equalised average** over
the 29 places a stock LFM2 reads the residual, and the top-k eigenvectors of an
average maximise the MEAN retained read energy over those readers, which says
nothing about the worst of them. Wave 21 moved that axis, and it lost like the
other four. Four things generalise.

* **A pooled retention number hides its own readers, and the hiding is
  structural rather than incidental.** The living 3/15 width child sits at
  0.9193 pooled read energy with **four of its 29 readers below 0.90** — the
  level at which three separate children read 0/15 — and the worst is 0.8801.
  The badly-retained endpoints are the same ones at every width: blocks 0 to 3,
  argmin at `B01.ffn` in all five sealed bases, while blocks 6 to 9 sit near
  0.99. **Generalisation:** publish the per-endpoint retention, not only the
  pool, on any axis whose law is stated in retained energy; the spread at the
  width where a model dies is 0.065 and rising, and a mean over 29 numbers that
  span 0.065 is not a description of any of them.
* **Fixing the worst reader is not the same as fixing the model, and the
  experiment is cheap enough that nobody should assume either way.**
  `max_E min_i tr(E^T M_i E)` over the endpoint moments is a concave problem on
  the convex hull of rank-k projections, so by Sion it is the convex
  `min_w [top-k eigenvalue sum of Σ w_i M_i]` over the endpoint simplex, whose
  subgradient is the vector of per-endpoint retentions — an eigensolve and an
  exponentiated-gradient step per round, no capture of its own, and **uniform
  weights return the pooled basis bit for bit**. It succeeded at its objective
  (worst reader 0.8801 → 0.8997, readers under the floor 4 → 1) and produced a
  child that reconstructs the parent's own logits BETTER than the living one
  (final-norm relative SSE 0.1522 against 0.1752) and returns loops in all
  sixteen review lanes. **Generalisation:** a basis objective that is provably
  better on the quantity your law is stated in can still be behaviourally dead;
  the law names a necessary condition, not the thing being optimised.
* **The subspace is a point, not a direction to search in.** Five objectives
  have now been measured against the pooled reader moment's own top eigenvectors
  on one model at one width and identical parameter count — case-weighted at two
  strengths, evidence (raw) PCA, reader-energy-per-raw, and max-min over readers
  — exchanging between **17 and 231 of 768 directions**, and every one of them
  scores at or below 1/15 where the unexchanged constructions score 3/15.
  **Generalisation:** on a residual-width fold, spend the search budget on the
  width and the solver, not on the objective. The eigenvectors of the pooled
  reader moment are not one good choice among many; they are, so far, the only
  choice that produces a model.
* **A cliff located by bisection may be a slope, and the difference matters for
  planning.** Wave 19 saw 3/15 at 768 and 0/15 at 640 and called it a step; two
  bisections later the ladder reads 4 → 3 → 1 → 0 across 896/768/736/704 and the
  interval that could still hide a step is 32 dimensions wide. **Generalisation:**
  bisect a behavioural cliff at least twice before describing its shape, because
  "step" and "slope" imply different strategies — a step says find the edge and
  stop, a slope says every rung costs something and the question is what you can
  afford.
* **And one estate law, from an operational failure worth repeating.** A
  deployment cut as a hardlink copy plus a verified delta is cheap and byte-true,
  and it breaks the frozen review, because the prompt-pack guard refuses a file
  with `st_nlink != 1`. The guard is right: a pack reachable under a second name
  is a pack something else can rewrite, and a review's whole authority is that
  the pack it read is the pack the seal names. **Generalisation:** integrity
  guards on evidence files should test reachability, not just content, and a
  cheap deployment trick that defeats one is a bug in the trick.
