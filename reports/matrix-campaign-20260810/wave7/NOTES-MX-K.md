# NOTES-MX-K — full model depth: the registry opens, and the wall moves one rung

**Wave 7 (MX-K), sealed 2026-08-11. ADDENDUM 6 grading, ADDENDUM 7 receipt
schema v2. Every folded component in this wave stores at or below a true
quarter of its own parent tensor; five of them store at or below an eighth.
No sub-4x level was solved, sealed, or spent budget on.**

Wave 6 asked for a fourteen-layer ladder and got three rungs, because
`matrix_cell_receipt.COMPONENTS` registered three sites per component and
`_validate_cell` enforced it (`NOTES-MX-J` §5). This wave removes that cap,
versions the schema so the removal cannot reach backwards, and then climbs.

**Fifteen composed children, fifteen sealed ADDENDUM 6 verdicts, thirty new
ledger rows. Three children hold the parent's band. Three more hold the
alive-degraded band. Nine do not.**

The headline is not the ladder. It is that the wave-6 headline was wrong:
`conv_out_proj` is not composition-toxic. **The L05 block is.**

---

## 1. The frontier

`net stored ratio` is the composed child's whole-model unique-parameter count
over the parent's 229,693,184. `stored/owned` is each folded component's own
stored form over the parent tensor it replaced — printed on every row, because
the campaign's whole-model number is now allowed to grow only by *coverage*,
never by folding anything shallower than 4x.

| child | folded (level @ site) | stored/owned per component | review | verdict | net stored | Δ params | global drift |
|---|---|---|---:|---|---:|---:|---:|
| `COMP.MA.MXK14` | C05 **S8**@L05, C07 S4@L12 | 0.124145 / 0.249008 | **13 / 15** | **alive-parent-band** | 0.994287× | −1,312,137 | 0.296834 |
| `COMP.MA.MXK6` | C05 S4@L05, C07 S4@L12 | 0.249267 / 0.249008 | **10 / 15** | **alive-parent-band** | 0.994859× | −1,180,937 | 0.269518 |
| `COMP.MA.MXK5` | C11 S4@L05, C13 S4@L05, C07 S4@L12 | 0.248828 / 0.249853 / 0.249008 | **10 / 15** | **alive-parent-band** | **0.981152×** | **−4,329,353** | 0.345775 |
| `COMP.MA.MXK13` | C11 **S8**@L05, C13 S4@L05, C07 S4@L12 | 0.124414 / 0.249853 / 0.249008 | **9 / 15** | **alive-degraded** | **0.979732×** | **−4,655,497** | 0.347091 |
| `COMP.MA.MXK1` | C13 S4@L04, C13 S4@L05, C07 S4@L12 | 0.249853 / 0.249853 / 0.249008 | **8 / 15** | alive-degraded | 0.981163× | −4,326,666 | 0.391302 |
| `COMP.MA.MXK4` | C13 S4@L05, C07 S4@L08, C07 S4@L12 | 0.249853 / 0.249008 / 0.249008 | **8 / 15** | alive-degraded | 0.988010× | −2,753,937 | 0.363435 |
| `COMP.MA.MXK9` | C05 S4@L05, C11 S4@L05, C07 S4@L12 | 0.249267 / 0.248828 / 0.249008 | 7 / 15 | impaired | 0.986286× | −3,150,089 | 0.314022 |
| `COMP.MA.MXK10` | C11 S4@L05, C13 S4@L05, C11 S4@L07, C07 S4@L12 | 0.248828 ×2 / 0.249853 / 0.249008 | 7 / 15 | impaired | 0.972579× | −6,298,505 | 0.362743 |
| `COMP.MA.MXK15` | C11 **S8**@L05, C13 **S8**@L05, C07 **S8**@L12 | 0.124414 / 0.124731 / 0.124163 | 7 / 15 | **dead** (degeneracy) | 0.978019× | −5,048,952 | 0.428846 |
| `COMP.MA.MXK2` | C13 S4@L05, C13 S4@L06, C07 S4@L12 | 0.249853 ×2 / 0.249008 | 7 / 15 | **dead** (degeneracy) | 0.981163× | −4,326,666 | 0.405338 |
| `COMP.MA.MXK7` | MXK5 + C11 S4@L13 | 0.248828 ×2 / 0.249853 / 0.249008 | 7 / 15 | **dead** (degeneracy) | 0.972579× | −6,298,505 | 0.371356 |
| `COMP.MA.MXK8` | MXK5 + C07 S4@L08 | 0.248828 / 0.249853 / 0.249008 ×2 | 7 / 15 | **dead** (degeneracy) | 0.979437× | −4,723,089 | 0.377027 |
| `COMP.MA.MXK3` | C13 S4@L03, C13 S4@L05, C07 S4@L12 | 0.249853 ×2 / 0.249008 | 6 / 15 | **dead** (degeneracy) | 0.981163× | −4,326,666 | 0.469745 |
| `COMP.MA.MXK11` | C13 S4@L04, C11 S4@L05, C13 S4@L05, C07 S4@L12 | 0.249853 ×2 / 0.248828 / 0.249008 | 6 / 15 | impaired | 0.972590× | −6,295,818 | 0.410367 |
| `COMP.MA.MXK12` | C13 S4@L05, C07 S4@L08, C07 S4@L10, C07 S4@L12 | 0.249853 / 0.249008 ×3 | 4 / 15 | **dead** (degeneracy) | 0.986296× | −3,147,673 | 0.430054 |

*Parent control, same process, every dossier: **12 / 15** — 5 useful, 7 partial,
3 observed miss, misses at `arithmetic-multi-step`, `logic-ordering`,
`multi-turn-correction`. Its sixteen responses are **byte-identical** to wave
6's, which is how this wave's calibration was spent: the control was checked
before a single child was graded and it reproduced exactly.*

**The two frontier children.**

> **Deepest alive-parent-band: `COMP.MA.MXK5`** — `mlp_gate` + `mlp_down` at L05
> and `attn_k` at L12, all three at a true quarter, **10 of 15** against the
> parent's 12, net **0.981152×**.
>
> **Deepest at or above the A9 band: `COMP.MA.MXK13`** — the same child with
> `mlp_gate` taken to a true **eighth**, **9 of 15**, net **0.979732×**. It is
> both deeper in storage and higher in review than any other survivor.

`COMP.MA.MXJ2` from wave 6 (11/15, 0.989725×) is still the highest-scoring
*multi-component* child at a quarter; wave 7 goes 0.0085 deeper on storage and
pays one review case for it.

---

## 2. What wave 6 got wrong about C05, and the experiment that shows it

`NOTES-MX-J` §2 is titled *"the component with the lowest local drift is the one
that destroys the composition"*, and it concludes that `C05 conv_out_proj` is
**composition-toxic** — a property of the component — because adding it to
`COMP.MA.MXJ2` cost five review cases under all three of its
alive-parent-band techniques and under M-B conditioning.

Every one of those four experiments had `C13 mlp_down` installed at **L05**,
the same block C05 was installed in. Nothing in wave 6 varied that. Wave 7 did.

| child | what is in block L05 | review | verdict | net stored |
|---|---|---:|---|---:|
| `COMP.MA.MXK6` | **C05 alone** (no MLP fold at L05) | **10 / 15** | **alive-parent-band** | 0.994859× |
| `COMP.MA.MXK9` | C05 + `mlp_gate` | 7 / 15 | impaired | 0.986286× |
| `COMP.MA.MXJ3` (wave 6) | C05 + `mlp_down` | 6 / 15 | impaired | 0.986297× |
| `COMP.MA.MXK5` | `mlp_gate` + `mlp_down`, **no C05** | **10 / 15** | **alive-parent-band** | 0.981152× |

Every one of these also folds `C07 attn_k` at L12, which is held fixed.

**`conv_out_proj` composes.** On its own with `attn_k` it holds the parent band
at 10/15, one case below the two-component `MXJ2` and two below the parent — and
its S8 arm reads 13/15, above the parent's own control. What does not compose is
**a conv writer and an MLP tensor folded in the same block**: `mlp_gate` costs
three cases on top of C05, `mlp_down` costs four, and the last row rules out the
obvious alternative — the two MLP tensors fold together in L05 at no cost at
all, so it is not "two folds in one block", it is **the conv writer plus an MLP
fold in one block**.

### 2.1 The mechanism, read off the sealed diagnostics

The three residual-stream facts that separate the toxic pairs from the safe one:

**C05 is the only one of the four that writes the block's *operator* output.**
`matrix_technique_registry._LAYOUT` gives C05 `input = short_conv.context`,
`write = operator.output`, `macrocell = residual.post_layer`; C11 and C13 both
take `mlp.input` / `mlp.post_swiglu` and write `mlp.gate` / `mlp.down` into the
same `residual.post_layer`. In an LFM2 block the operator writes into the
residual **before** the FFN norm reads it, so a C05 error is not one of two
independent contributions to the block's output — it is an input perturbation to
the MLP, and the MLP's own folded error is then computed against a residual that
has already moved. C11 and C13 sit on the same side of that norm as each other
and do not have this relationship.

**The drift numbers say the same thing and the single-component review cannot.**
Installed first, C05@L05 drifts 0.181179 and C11@L05 drifts 0.168344 — the two
smallest single installs the campaign has measured. Installed *second, behind
C05*, C11@L05 drifts to 0.241203, an increase of 0.072859 for a component whose
own solve did not change by a digit. In `COMP.MA.MXK5`, where C11 goes in first
and C13 second, the second install adds only 0.121619 to a stack with no conv
writer in it. Composition is superadditive across the operator/FFN boundary of a
block and roughly additive within it.

**The failure has the shape of a writer-scale error, not a capability loss.**
The five cases wave 6 watched fall (`summarization`, `comparison-ram-rom`,
`coding-deduplicate`, `creative-continuity`, `arithmetic`) are the same five
lanes that degrade first in every impaired wave-7 child, and they degrade the
same way — the response keeps its shape and loses its *content commitments*: the
decision disappears from the summary, the analogy contradicts itself in adjacent
clauses, the loop body loses `seen.add`. That is what a residual whose magnitude
is right and whose direction has rotated looks like downstream: the format
survives because it is carried by the strongest directions, and the specific
bindings die because they are carried by the ones the fold truncated.

**What this predicts, and what would falsify it.** Predicted: C05 at any conv
layer composes freely with folds in *other* blocks, and stops composing with any
fold in *its own* block, including `C03 conv_in_proj` and `C04 conv_kernel`.
Falsified if C05@L05 + C11@L07 (different blocks) reads below the parent band,
or if C05@L05 + C03@L05 reads inside it. **Neither was run — the wave spent its
budget on the ladder — and both are one four-minute child each.**

---

## 3. The per-layer ladder, rung by rung

Order was fixed before the wave: nearest-to-the-proven-site first, stop a
component after two consecutive failing sites, back off any site that drops the
child below the A9 band.

### 3.1 `C13 mlp_down` across the MLP layers (all fourteen are valid under v2)

On top of `COMP.MA.MXJ2`:

| site added | install drift | solve condition | review | verdict | net |
|---|---:|---:|---:|---|---:|
| L04 | 0.258949 | 3,528.6 | **8 / 15** | alive-degraded | 0.981163× |
| L06 | 0.355062 | 3,536.7 | 7 / 15 | **dead** | 0.981163× |
| L03 | 0.363202 | 1,747.2 | 6 / 15 | **dead** | 0.981163× |

Two consecutive dead sites, so the C13 ladder stops at three rungs — the same
number wave 6 got, for a completely different reason. L00 was not retried: wave
6 measured it at 0.422844 drift and 5/15, the ladder had already failed twice
before reaching it, and the escalation budget went to the S8 probes instead
under the mid-run directive.

### 3.2 `C07 attn_k` across the attention layers

| stack | site added | install drift | condition | review | verdict | net |
|---|---|---:|---:|---:|---|---:|
| MXJ2 | L08 | 0.308949 | 9.198 | **8 / 15** | alive-degraded | 0.988010× |
| MXK4 | + L10 | 0.383855 | 8.230 | 4 / 15 | **dead** | 0.986296× |
| MXK5 | L08 | 0.326709 | 9.198 | 7 / 15 | **dead** | 0.979437× |

`COMP.MA.MXK12` — three attn_k sites plus mlp_down at L05 — is the worst child
either wave has produced: **zero useful cases**, and **five separate lanes are
token-level loops** that ride a single token into the 1,024-token cap
(`and`, `sanitized`, `vegan`, `[Your Email Address]`, `1 year`). Two attn_k
folds degrade; three collapse the decoder.

### 3.3 `C11 mlp_gate`

| stack | site added | install drift | condition | review | verdict | net |
|---|---|---:|---:|---:|---|---:|
| MXJ2 | **L05** | **0.168344** | 17,769.8 | **10 / 15** | **alive-parent-band** | 0.981152× |
| MXK5 | + L13 | 0.371356 | 13,038.7 | 7 / 15 | **dead** | 0.972579× |
| MXK5 | + L07 | 0.312499 | 12,828.4 | 7 / 15 | impaired | 0.972579× |

Two consecutive failures, ladder stopped. **L05 is the wave's one real win**:
the same three components wave 6 assembled as `COMP.MA.MXJ3C` (9/15) at exactly
the same net storage, with `mlp_gate` moved from L00 to L05, reads 10/15. A site
change bought a review case at zero storage cost, which is the clearest possible
demonstration of why the registry had to open.

### 3.4 Four components at 4x is past the wall

Three attempts, all sealed:

| child | four folded components | review | net |
|---|---|---:|---:|
| `COMP.MA.MXK11` | the two cheapest installs by drift, stacked | 6 / 15 | 0.972590× |
| `COMP.MA.MXK10` | MXK5 + C11@L07 | 7 / 15 | 0.972579× |
| `COMP.MA.MXK7` | MXK5 + C11@L13 | 7 / 15 (dead) | 0.972579× |

**No four-component child at S4 reached 8/15.** `COMP.MA.MXK11` is the one worth
reading rather than counting: it produces the only exactly-two-sentence, fully
anchored `summarization` in either wave and the most complete
`context-retrieval`, and then refuses the greeting outright and loses every
reasoning and coding lane. The capability does not fade uniformly; it goes out
in lanes.

---

## 4. The S8 probes (Cyril's mid-run directive, 2026-08-11)

*"[Cyril quote withheld]"* Both parent-band sites were probed,
and the whole stack was probed once.

| probe | what changed | stored/owned | review | verdict | net |
|---|---|---:|---:|---|---:|
| `COMP.MA.MXK6` → `COMP.MA.MXK14` | C05@L05 quarter → **eighth** | 0.249267 → **0.124145** | 10 → **13 / 15** | parent-band both | 0.994859× → 0.994287× |
| `COMP.MA.MXK5` → `COMP.MA.MXK13` | C11@L05 quarter → **eighth** | 0.248828 → **0.124414** | 10 → **9 / 15** | band → degraded | 0.981152× → 0.979732× |
| `COMP.MA.MXK5` → `COMP.MA.MXK15` | **all three** → eighth | 0.1244 / 0.1247 / 0.1242 | 10 → 7 / 15 | band → **dead** | 0.981152× → 0.978019× |

**One site at an eighth holds. Three sites at an eighth at once does not**, and
the storage is not what separates them: 0.978019× against 0.979732× is 0.0017 of
the parent, and one child is dead by token-level collapse while the other is a
usable model.

`COMP.MA.MXK14` deserves its own sentence. It is a two-component child at
0.994287× and it reads **13/15 — one case above the parent control measured in
the same process on the same sixteen prompts**. It is the only child in either
wave to correct the off-by-one to `values[start:]`, one of two with a fully
correct first-seen-order dedup function with its complexity claim, and it states
the finish order *Mira, Theo, Jun, Sal* correctly inside its `logic-ordering`
explanation before answering the wrong question. **The +1 over the parent is
inside ADDENDUM 2's ±2 reviewer interval and is reported as parent-band, not as
an improvement.** The +3 over its own S4 arm at the same site is at the edge of
that interval, and is the only reason the row is asserted at all.

---

## 5. Conditioning is not the mechanical predictor

The wave was asked to record the per-site solve conditioning at unexplored
depths and to install anyway. It did. The conditioning is anti-correlated with
the behaviour, sometimes strongly.

| cell | condition number | first-install drift | what the review said |
|---|---:|---:|---|
| `C11.T8.S4.L05` | **17,769.8** — the worst in the wave | **0.168344** — the best in the wave | **alive-parent-band** |
| `C11.T8.S4.L13` | 13,038.7 | 0.371356 | dead |
| `C11.T8.S4.L07` | 12,828.4 | 0.312499 | impaired |
| `C13.T8.S4.L04` | 3,528.6 | 0.258949 | alive-degraded |
| `C13.T8.S4.L06` | 3,536.7 | 0.355062 | dead |
| `C13.T8.S4.L03` | **1,747.2** — better than L04 and L06 | 0.363202 | **dead** |
| `C05.T8.S4.L05` | 1,199.0 | 0.181179 | parent-band |
| `C05.T8.S8.L05` | **447.8** — better than its own S4 | 0.223583 | parent-band, 13/15 |
| `C07.T5.S4.L08/L10/L12` | 9.198 / 8.230 / 7.770 | 0.309 / 0.384 / 0.270 | degraded / dead / band |

Two readings worth keeping. **The best-conditioned MLP solve in the wave
(`C13@L03`, 1,747) is dead and the worst-conditioned solve in the wave
(`C11@L05`, 17,770) holds the parent band** — the rank order is not merely
noisy, it is inverted at the extremes, which is the same relationship
`NOTES-MX-J` §2 found between local drift and behaviour, now for the solve's own
conditioning. And **deepening to S8 *improves* the condition number** at C05
(1,199 → 448) and C13 (1,764 → 798), because a narrower truncation keeps fewer
and better-separated directions; the S8 stack is dead anyway. No ridge was
engaged anywhere: `ridge_relative` is 0.0 for every T8 solve and 1e-10 for every
T5 route.

**Install drift is the better predictor and is still not good.** Every child
whose global drift stayed under 0.35 is alive (0.270, 0.297, 0.314, 0.346,
0.347); every child over 0.40 is dead or impaired (0.405, 0.410, 0.429, 0.430,
0.470); the band 0.35–0.40 contains 8/15, 8/15, 7/15 and 7/15 and decides
nothing. Drift is a screen, not a gate, and the review is still the grade.

---

## 6. The degeneracy clause earned its keep six more times

Six children are sealed `dead` on the clause. Four of them would have been
**impaired** on count alone and one — `COMP.MA.MXK8` at 7/15 — sits one case
below a band the clause makes irrelevant.

| child | count alone | why it is dead |
|---|---|---|
| `COMP.MA.MXK2` | impaired (7/15) | `planning-dependencies` steps 6–20 cycle four blocks verbatim into the cap; nothing progresses between step 6 and step 20 |
| `COMP.MA.MXK3` | impaired (6/15) | `logic-ordering` restates one filler clause four times and nothing remains without it; `planning` gives sections 2–5 an identical two-line body |
| `COMP.MA.MXK7` | impaired (7/15) | `planning` collapses at the token level — the word `backend` several hundred times to the cap |
| `COMP.MA.MXK8` | impaired (7/15) | `multi-turn-correction` is three options whose bodies are the same two bullets; removing the repeats leaves one self-contradictory line |
| `COMP.MA.MXK15` | impaired (7/15) | `drawer` ~400 times and `seen_order_items = []` ~100 times, both to the cap; the smoke also terminated on `loop` |
| `COMP.MA.MXK12` | impaired (4/15) | **five** lanes are single-token loops to the cap |

The test applied, unchanged from `NOTES-MX-I` §3.3 and stated so it can be
overturned deliberately: **remove the repetitions and ask whether a response
remains.** It was applied in both directions this wave. Repetition was
*recorded and not charged* in `COMP.MA.MXK1`, `MXK4`, `MXK5`, `MXK6`, `MXK9`,
`MXK10`, `MXK13` and `MXK14` — duplicated plan steps, a doubled sentence in a
scene, a template repeated across list items — because in each case a
structured, progressing response survives the removal.

---

## 7. Schema v2 (ADDENDUM 7), and what it did not touch

The parent's real layer-type map, read off `config.json`'s `layer_types` and
re-checked against the shipped `model.safetensors` key set (which sums to
229,693,184 exactly):

| block kind | layers | count |
|---|---|---:|
| `short_conv` | 0, 1, 3, 5, 7, 9, 11, 13 | 8 |
| `full_attention` | 2, 4, 6, 8, 10, 12 | 6 |
| `feed_forward` (SwiGLU w1/w2/w3) | **all fourteen** | 14 |
| `operator_norm` / `ffn_norm` (+ global `embedding_norm`) | all fourteen (+ FINAL) | 15 |

v2 widens each component to exactly its own block kind's layers and nothing
else: `C07 attn_k` is still refused at L05, `C05 conv_out_proj` is still refused
at L04, and nothing reaches a fifteenth layer.
`tests/test_matrix_addendum7.py` (15 tests, all passing) pins the widening, the
frozen v1 registry, the refusal of a v2 site on a v1 body, the equality of the
failure chain across versions, and — the check that mattered most in practice —
that all 34 v1 evidence-cache names are unchanged, so the 21 authenticated site
caches on the run host stayed reachable and a reused site cost nothing.

**Both versions are live.** All 348 rows from waves 1–6 declare v1 and validate
against v1's narrow registry; all 30 wave-7 rows declare v2. The ledger's new
`receipt_schema_version` column reports exactly that, and `LEDGER_SCHEMA` did
not move, so the old rows are still read by the same reader.

### 7.1 The honest new ceiling

At a true S4 bound against the parent's 229,693,184:

| registry | foldable parent mass | stored | net |
|---|---:|---:|---:|
| v1, five APB components × 3 registered sites | 23,592,960 | 5,898,240 | 0.9230× |
| v1, every registered body site | 45,625,344 | 11,406,336 | 0.8510× |
| **v2, five APB components × every valid site** | **91,226,112** | 22,806,528 | **0.7021×** |
| v2, the four that compose (C05 dropped) × every valid site | 82,837,504 | 20,709,376 | 0.7295× |
| **v2, every body component × every valid site** | **162,583,552** | 40,645,888 | **0.4691×** |
| v2, every body component × every valid site at **S8** | 162,583,552 | 20,322,944 | **0.3806×** |

C10 is excluded from the mass everywhere — it owns the same four tensors
C06–C09 do. The floor under all of these is the same **67,108,864-parameter
tied embedding table**, 29.2% of the parent, which no body component touches.
**A whole-model 4× is unreachable without C01/C02/C15**, and ADDENDUM 7 does not
change that; what it changes is that the body is no longer capped at 0.92×.

**And the measured number is 0.9797×, not 0.4691×.** The registry was the cap on
what could be *said*; it was never the cap on what the model would *survive*.
The real cap is that this parent, at this technique surface, tolerates **three
folded components** at 4x-or-deeper and no more — and that is now a measurement
across nine sites and four components rather than an artefact of a site policy.

---

## 8. What this wave did not settle

* **The C05 mechanism has one clean bracket and two unrun controls.**
  C05@L05 + a fold in a *different* block, and C05@L05 + `C03`/`C04` in its own
  block, are the two children that would turn §2.1 from a strong reading into a
  law. Each is one four-minute run.
* **M-B was not run at all.** ADDENDUM 6's escalation arm (b) was skipped: wave
  6 measured it at 27× the wall clock for zero review cases, the mid-run
  directive redirected the budget to S8, and no wave-7 rung was close enough to
  a band boundary for a second-order correction to matter. Recorded as a
  deliberate omission, not an oversight.
* **`C13@L00` was not retried** (see §3.1), and no site above L08 was tried for
  `C13` at all — the ladder failed twice before reaching them.
* **`C12 mlp_up`, `C09 attn_o`, `C08 attn_v` were never composed**, for the
  third wave running.
* **S8 was probed at two sites, not at every site that holds.** `C13@L05` and
  `C07@L12` were only taken to S8 inside the all-S8 child that died; neither has
  a single-site S8 arm.
* **The frozen review is one reviewer, one pass.** ADDENDUM 2's ±2 is the
  measured interval. `MXK5`'s −2 and `MXK14`'s +1 are inside it; `MXK9`'s −5,
  `MXK3`'s −6 and `MXK12`'s −8 are not, and that is the only reason the C05
  bracket and the attn_k collapse can be asserted.
* **No claim about a site that was not folded**, per ADDENDUM 5 rule 2-S — which
  v2 makes load-bearing rather than decorative. v1 registered 34 component-site
  pairs; v2 registers 111. This wave measured **seven of the 77 pairs v2 adds**
  (`C13`@L03/L04/L06, `C11`@L05/L07/L13, `C07`@L10) plus four S8 levels. The
  other 70 have no evidence attached to them whatsoever, and being registered is
  not evidence.

---

## 9. Ledger, receipts, disk

| what | where |
|---|---|
| ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.pre-mxk-20260811T175820` (`5fc36c1b…`) |
| ledger before the merge | `…/backup/ledger.jsonl.before-mxk-merge-20260811T190204` (`5fc36c1b…`) |
| `LEDGER.md` before the render | `…/backup/LEDGER.md.before-mxk-render-20260811T190220` |
| pruned heavy artifacts | `reports/matrix-campaign-20260810/wave7/pruned-tensors-manifest.json` |

**The ledger.** 348 → **378 rows (+30)**: 15 composed-child instances and 15
review instances. `sha f5994b537726036b…`, **378 distinct
`(cell_instance_id, receipt_sha)` pairs**, 378 distinct receipt digests, **30 of
30 wave-7 rows on `sm86 NVIDIA GeForce RTX 3090`** and **30 of 30 at
`receipt_schema_version` 2**, with all 348 earlier rows carrying no such key —
which is the v1 statement, made by absence, exactly as ADDENDUM 7 §7.3 designs
it. The merge's refuse-unless-prefix check passed byte for byte and the merger
was run exactly once, because `NOTES-MX-H` §9 records that it is not idempotent.

**Seals.** 374 of 378 ledger rows resolve to a receipt on disk and **all 374
verify; 0 fail**. All **30 of 30** wave-7 seals verify. 313 receipts satisfy
today's mandatory field set and **61 wave-1 receipts do not**, for the
documented reason — they predate `code_identity.compute_architecture` — the same
61 waves 4, 5 and 6 reported.

**Four ledger rows do not resolve, and none is wave 7's**: the same three wave-1
`CAL.C01.T5.X4` calibration rows and the one wave-5 `C06.T2.S4.L12` review
instance whose file was overwritten 21 seconds later, which is also the ledger's
only duplicated `cell_instance_id`. Wave 7 found them by checking, as wave 6
did, and did not touch them.

**Mirror.** `reports/matrix-campaign-20260810/ledger.jsonl` and `LEDGER.md` are
**byte-identical** to the server authority:

```
f5994b537726036b5c7e195c1a0d69f2f3fffee6ee5c961968124351a9f2a6be  ledger.jsonl
e7e435c9df05b9bab0ebbc6a0f083c25ccc65649bbf5651b9ee89ecc491ec7dd  LEDGER.md
```

**Retention.** Five children's composed tensors are kept — `MXK5` (deepest
alive-parent-band), `MXK13` (deepest at/above the A9 band, and the S8 arm of the
MXK5 pair), `MXK14` (highest-scoring child of either wave, and the S8 arm of the
MXK6 pair), `MXK6` and `MXK9` (the two legs of the C05 separating bracket, which
cannot be re-read with one leg missing). **Ten files, 0.23 GiB** were
sha256-digested into `wave7/pruned-tensors-manifest.json` and deleted; **all ten
digests matched the digest their own receipt had sealed before deletion**, and
every entry carries the sentence that says how to re-derive it.

**Disk.** Root `/` was **7.2 GB free at the start of the wave and 7.3 GB free at
the end.** Every wave-7 artifact, every one of the seven new site evidence caches
(L03-C13, L04-C13, L05-C11, L06-C13, L07-C11, L10-C07, L13-C11 — 142 MB each)
and every composed child was written under
`/mnt/ai_data/neural-forge-matrix/wave7/`, and the 21 already-built caches were
reached through symlinks so reuse cost nothing.

---

## 10. Provenance

**A new deployment was built, and it had to be**: this wave changes the receipt
schema, and a wave-6 deployment cannot seal a v2 site.
`/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16` is a
`git archive` of **this wave's own commit** (tarball `9414879768c3e274…`).

* **All 605 Python modules verify byte-identical to the git blobs of `17377c16`**
  (`sha256sum -c` against a manifest built locally from `git show`).
* The deployment differs from wave 5/6's `eb32153e` surface **in exactly the two
  files the commit touched** — `src/neural_forge/matrix_cell_receipt.py` and
  `experiments/run_lfm2_matrix_composition.py` — across all 814 shared paths.
* `git archive` re-lined 34 non-Python assets (19 json, 15 md), including
  `V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md`, which archived to `cdf5d927…`. **All 34
  were restored byte-for-byte from the wave-6 deployment**, and the protocol was
  re-verified at `e5e3776885976b1b…` and the prompt pack at `0ee6bf14cef51baf…`
  **before a single token was generated**. `run-mxk-review.sh` re-checks both
  digests as a preflight and refuses the review if either has moved.

Review surface, digest-pinned in every one of the fifteen dossiers: protocol
`e5e37768…`, prompt pack `0ee6bf14…`, greedy decoding at the 1,024-token cap,
the parent decoded on the same sixteen prompts **in the same process** as every
child. GPUs `sm86 NVIDIA GeForce RTX 3090` indices 0 and 4, pinned by PCI bus id
in the shell preflight **and** re-checked by `--require-compute-architecture
sm86` inside every sealing process. Evidence base `2dec87ac…`, opened read-only.
Knob file byte-identical to waves 5 and 6's.

**The technique surface is proven inert to the schema change.** Wave 7's first
child re-solved `C13.T8.S4.L05` and `C07.T5.S4.L12` from scratch under the new
deployment and reproduced wave 6 to the digit: calibration relL2
`0.0006382105222036804`, rank 639, gram 0.20058, cos 0.92045, stored 654,975,
and `C07` keeping 121 of 512 — identical to the sealed `COMP.MA.MXJ2` values.
`COMP.MA.MXJ2`'s own seal was verified before anything was built on it, and both
of its installed tensors re-digested to the values its receipt sealed.

Every verdict was computed by `behavior_verdict_for` from counts the reviewer
sealed through `seal_matrix_behavior_review.py`; no verdict in this wave was
stated by hand.

---

## 11. Verification

The matrix test subset, run server-side on a `git archive` of **this wave's own
commit** (`17377c16`, tarball `ffb51afab401bda8…`) unpacked to
`/mnt/ai_data/neural-forge-matrix/wave7/testtree` — no worktree, and therefore
none of the Codex lane's uncommitted files:

```
python -m pytest tests/test_matrix_composition.py tests/test_matrix_cell_receipt.py \
                 tests/test_run_lfm2_matrix_cell.py tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py tests/test_matrix_technique_new_axes.py \
                 tests/test_matrix_addendum6.py tests/test_matrix_addendum7.py -q
# -> 10 failed, 175 passed, 1 skipped
```

One file per process: **6 failed, 179 passed, 1 skipped**. The four-failure
difference is `test_most_specific_registration_wins`, which monkeypatches the
global technique dispatch table and never restores it, so every later test in
the same process that resolves T3 gets its stub. Pre-existing, in the Codex
lane's files, reported and not fixed.

**The same nine files at wave-6 HEAD `670ce320` fail identically: 10 failed, 160
passed, 1 skipped**, with the identical ten test names. This wave's commit adds
15 passing tests and **zero failures**. The six real failures are the documented
`blank_receipt` `compute_architecture` `TypeError` family (four in
`test_matrix_technique_registry`, one in `test_run_lfm2_matrix_cell_fresh_evidence`)
whose fix is uncommitted in the Codex lane and which a `git archive` cannot see,
plus `test_factor_pair_storage_model_reproduces_a_sealed_wave2_receipt`, which
also fails at wave-6 HEAD and is not this wave's.

`test_matrix_composition` — the module the whole wave rides on — is 33 of 33.
`test_matrix_cell_receipt` is 31 of 31 with the schema change in it.
`test_matrix_addendum7` is 15 of 15.
