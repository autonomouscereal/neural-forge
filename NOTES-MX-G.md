# NOTES-MX-G — matrix campaign, wave 3

Wave MX-G, the last isolation wave before the composition modes. 31 cells
against the same fresh evidence base `2dec87ac…` at the same pre-registered
5e-3 calibration tolerance, on both RTX 3090s, plus the two missing technique
kernels `NOTES-MX-F` §4 localized and the T9 schedule nobody had ever run.

Repo baseline HEAD `9165e45c`, branch `codex/lrc-blank-compression`. The frozen
collector, the compatibility policy and Codex's live-edit files were **not**
touched. No `git add`, no commit.

**Verdicts up front:**

| deliverable | verdict | one line |
|---|---|---|
| 31 cells, wave 3 | **ALL SEALED** | **7 PASS**, 23 FAIL, 1 INFRA; 20:33→20:53 UTC on two GPUs, 23.5 GPU-minutes total |
| T3's wave-2 surprise | **it is a SITE result, not a component result** | `C05.T3` at M16 reads **0.914 / 0.699 / 0.646** at L05 / L00 / L13 — the 0.914 does not survive its own rule-2 obligation |
| T3 on the attention writer | **generalizes only at composition scale** | C09: T1 wins at M16 (0.820 vs 0.760) and M8 (0.736 vs 0.707), T3 wins at 768 by **0.264** (0.709 vs 0.445) |
| the two missing kernels | **both built, both gated, both produce first-ever evidence** | C04 replays its parent operator at **0.00295** and C10 at **0.00165**; identity gates **5.3e-15** and **exactly 0** |
| C10 drop-head | **the campaign's first technique that behaves AND contracts** | free-run **0.736** at 0.958× storage, 0.654 at 0.917× — every dense marginal cell in this campaign stores *more* than its parent |
| the X4 probes | **only the attention readers survive composition-scale width, and only at L12** | `C06.T1.X4.L12` **0.977 PASS**, `C07.T1.X4.L12` **0.918 PASS**; the conv writer collapses to 0.33 and the attention writer sits at 0.71 |
| T9 progressive, first probe ever | **stepping does not beat jumping — over T1 it is provably the identity** | `C05.T9.M8.L05` is **bit-identical** to the direct `C05.T1.M8.L05` on every sealed field and all three smoke outputs; the rung-2 correction norm is 2.96e-13 |
| the S8 band | **it failed for the first time in 50 measured cells, and it was right** | `C04.T1.M2.L00` macrocell **0.4432**, out of band, free-run 0.113, the wave's only incoherent review |
| rule 2 at M8 | **C06/T1 is 3 of 3 and `works to M8` outright; C07/T1 is 2 of 3 and its verdict depends on a rule-2 ambiguity (§4.6)** | and `C06.T5` matches dense ridge at L12 (0.977) while storing **0.938×**, the first route pass at an attention component |

---

## 1. Three new kernels, one registry change

### 1.1 C04 — the depthwise FIR tap contraction (`matrix_technique_conv_fir.py`)

`NOTES-MX-F` §4 moved C04 from "harness failure" to "technique refusal": no
kernel was registered for the `kernel` axis and T1/T5 both declined it by name.
That is a smaller hole than it looks — the axis is three FIR taps per channel,
so the contraction is "keep `t` of them and refit the survivors" and the whole
technique is 500 lines. Three things made it real work rather than a one-liner.

**The level axis is not the axis the harness resolves.** `resolve_level_width`
measures `ComponentSpec.own_width`, which for a `(1024, 1, 3)` tensor is the
*channel* count. M16 on C04 hands the kernel **960** and means **2 taps of 3**.
The kernel re-resolves the level against `LEVELS` on the tap axis and seals both
numbers (`harness_level_width: 960` beside `kept_taps: 2`). On a 3-tap parent
the marginal family collapses — M16, M8 and M4 all floor to 2 taps and only M2
reaches 1 — and the receipt says so instead of pretending four levels exist.

**A FIR needs the token axis and the evidence rows do not carry one.**
`TraceSiteEvidence` concatenates every batch and phase into one `(rows, width)`
matrix. Row adjacency is real *inside* a captured sequence (prefill tokens in
order, then the forced decode steps, whose conv state continues the prefill's)
and meaningless across one, and the blocks are **not even uniform**: the sealed
base's FIT half is 4,751 rows, one short of 18×264.

So the kernel derives the segmentation and then proves it. Every captured
sequence starts on the same first token and a position-0 row is a function of
that token alone, so the rows bitwise equal to row 0 are the sequence starts.
That is a hypothesis; it is tested the way §2.1 tests everything — by
reproducing the parent. The full short conv is replayed from `operator.input`
(`in_proj` → `B*x` → the FIR under the derived segmentation → `*C` → `out_proj`)
and scored against the sealed `operator.output`:

```
fit sequences   18, lengths {263, 264}      <- the missing row, found
heldout         6,  lengths {264}
parent replay relL2            0.00294585   (tolerance 0.02)
unsegmented replay relL2       0.02385623   (the deliberately wrong hypothesis)
discrimination ratio           8.10         (refuses below 3.0)
```

The kernel refuses if the replay misses the tolerance **and** refuses if the
wrong hypothesis is not materially worse, because a gate nothing can fail is
not a gate. The 0.02 tolerance is the estate's own band for this exact
reconstruction (`hf_lfm2_circuit_trace_cache._finalize_short_conv` accepts the
FIR replay at `rtol=2e-2`), pre-registered in the knob file before the cell ran.

**The refit target is the parent operator, which buys an exact identity gate.**
No sealed endpoint carries the FIR output alone, and recovering it from
`operator.output` would mean inverting `out_proj` and dividing by a gate that
passes through zero. The parent's taps and the reconstructed `B*x` give it
directly, so the least-squares target is the parent filter's response on the
evidence's own input distribution — and at zero drop the residual is
identically zero, so a prior-anchored solve returns the prior. Every C04 receipt
seals that measurement: `identity_gate_max_abs_deviation` **5.33e-15**.

Channels are weighted by the gate `C` by default (`conv_fir_channel_weighting`),
because the FIR output is multiplied elementwise by `C` before it leaves the
operator and an unweighted fit optimizes a quantity the model never reads.

This is also the **first parent-calibration measurement C04 has ever had**. The
runner still declares its generic gate inapplicable for C04 — correctly,
`held_input @ parent_weight.T` is not this component — and the number that gate
could not produce is sealed under `solve` where it belongs.

### 1.2 C10 — drop-head with O-projection absorption (`matrix_technique_head_drop.py`)

An attention write is `o = context @ out_proj.T` with `context` the
concatenation of per-head blocks, so a head is a contiguous block of `out_proj`
columns and of `q_proj` rows. Dropping head *j* zeroes both, and then the
surviving columns are **re-solved by least squares against the parent's own
write**, so the dropped head's contribution is absorbed rather than deleted.
Deleting a head is a one-line edit and a guaranteed loss; absorbing it asks
whether the remaining heads' context span already carries what the dropped head
was writing, which is the question nobody had asked.

Selection is exhaustive over drop sets, scored by the least-squares residual
computed from the full Gram (one Cholesky per candidate, not one pass over the
evidence per candidate) — 16 candidates for one head, 120 for two. The final
coefficients come from the production `_fit_linear` with the parent's surviving
columns as the prior, so C10's telemetry is the same `Lfm2PriorCorrectionReport`
every dense cell seals.

**One registry change was required and it is the point of the component.**
`_LAYOUT["C10"]`'s endpoints moved from `operator.input`/`operator.output` to
`attention.context`/`attention.o` (`matrix_technique_registry.py:179`). The old
pair put a softmax between the input and the write and left no endpoint at
which a dropped head is a nameable object — which is exactly why wave 1's
generic gate produced a meaningless 4.4955. The new pair is a single exact
matmul between two sealed endpoints with one channel block per head. These are
C09's endpoints, so a C10 cell **re-uses C09's authenticated site evidence**
rather than building a second copy of it, and no new evidence cache was cut.

The receipts:

```
parent write replay relL2      0.00165326   (== C09's own wave-1 calibration to 7 digits)
identity gate max |dW|         0.0          (exactly, at full head count)
M16 -> drop head 13   residual/energy 1.70e-4   worst single drop (head 1) 1.19e-2
M8  -> drop 11 and 13 residual/energy 4.80e-4   worst pair (1+6)        2.42e-2
```

Heads are not interchangeable: a **70× spread** between the cheapest and the
dearest head to absorb at the same level, and the two-head optimum is not the
two cheapest singletons naively stacked (11+13 beats 13+15). GQA is respected
and not exploited — the kernel reports any KV group whose queries are all gone
under `dead_kv_groups` (empty at both levels here) and does **not** count the
`k_proj`/`v_proj` rows it could then drop as saved storage, because this child
still stores them.

### 1.3 T9 — the progressive schedule (`matrix_technique_progressive.py`)

T9 owns no linear algebra. It resolves a base technique out of the same dispatch
table the harness uses, calls it once per rung, and between rungs does the only
two things "child_{t-1} becomes parent_t" can mean: the **prior** becomes the
previous rung's materialized weight, and the **target** becomes the previous
rung's own write (an evidence view answering the write endpoint with
`rows @ child.T` and passing every other endpoint through untouched).

The schedule's final width is always the cell's own level width, so a T9 cell
and its direct twin store the same rank and differ in exactly one thing: what
taught the fit. `C05.T9.M8.L05` runs 1024 → M16(960) → M8(896), the same 896 the
direct `C05.T1.M8.L05` stores.

Scope stated up front and repeated in §4.4: **T9 conditions the teacher, not the
input distribution.** In a single-component cell those coincide at the
component's own boundary, because every upstream neighbour is copy_exact. That
is the exact sense in which this cell is clean and a composed M-B run is not.

### 1.4 Tests, lint, and two pre-existing tests whose meaning changed

`tests/test_matrix_technique_new_axes.py` is new at **16 tests** and pins the
five ways a new kernel can be wrong in a way the receipt would not show: the
identity gate, the axis being the axis the cell id names, the stored form being
the child's, a refusal being a refusal, and T9's teacher substitution touching
only the write endpoint.

Two pre-existing tests changed meaning because a kernel now exists where one did
not, and were updated with the reason rather than deleted — the same treatment
wave 2 gave the `T3`→`T6` borrow:

* `test_unregistered_technique_names_the_registration_point` borrowed
  `("T9", "C13")` as a pair nothing registers. T9 now covers the nine body
  components, so it borrows `("T9", "C01")` — T9 still has no bare entry and
  C01 is not a body component.
* `test_t1_refuses_components_it_does_not_model` asserted T1's refusal *through
  the dispatch table* at C04. The reference kernel's refusal is unchanged, so
  the test now calls it directly and additionally pins that dispatch routes C04
  past it and still lands other components on it.

```
tests/test_matrix_cell_receipt.py, test_run_lfm2_matrix_cell.py,
test_matrix_technique_registry.py, test_run_lfm2_matrix_cell_fresh_evidence.py,
test_matrix_technique_charts.py, test_matrix_technique_new_axes.py
  -> 122 passed, 1 skipped        (106+1 before this wave; +16 new)
tests/test_matrix_technique_{body_routes,lowrank_writers,relation_matching}.py
  -> 76 passed                    (courtesy check, nothing in that lane edited)
ruff on the four changed/added source files and the two test files -> clean
```

---

## 2. The results

31 cells. Boundary/composed numbers are diagnostic; `free-run` and the review
are the only fields §5 rule 1 may read. **Grade** is ADDENDUM 3's interpretation
layer. **stored** is stored parameters over the dense parent — above 1.0 the
fold costs more storage than the tensor it replaced.

| # | cell | comp | tech | site | write cos | macro | vs control | free-run | **stored** | link | status | **grade** |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `C05.T3.M16.L00` | conv_out | T3 | L00 | 0.99904 | 0.0332 | 0.02692 | 0.699 | 1.875 | composition | fail | promising |
| 2 | `C05.T3.M16.L13` | conv_out | T3 | L13 | 0.99768 | 0.0125 | 0.02461 | 0.646 | 1.875 | composition | fail | promising |
| 3 | `C05.T3.M8.L05` | conv_out | T3 | L05 | 0.99468 | 0.0321 | 0.03959 | 0.777 | 1.750 | composition | fail | promising |
| 4 | `C05.T3.M4.L05` | conv_out | T3 | L05 | 0.98758 | 0.0492 | 0.05573 | 0.506 | 1.500 | composition | fail | weak |
| 5 | `C05.T3.M2.L05` | conv_out | T3 | L05 | 0.96302 | 0.0852 | 0.09700 | 0.277 | **1.000** | composition | fail | weak |
| 6 | `C05.T3.X4.L05` | conv_out | T3 | L05 | 0.98305 | 0.0576 | 0.06558 | 0.330 | 1.375 | composition | fail | weak |
| 7 | `C05.T1.X4.L05` | conv_out | T1 | L05 | 0.97401 | 0.0700 | 0.08716 | 0.314 | 1.375 | composition | fail | weak |
| 8 | `C05.T9.M8.L05` | conv_out | **T9**/T1 | L05 | 0.99236 | 0.0376 | 0.05392 | 0.746 | 1.750 | composition | fail | promising |
| 9 | `C05.T9.M4.L05` | conv_out | **T9**/T3 | L05 | 0.98466 | 0.0546 | 0.05978 | 0.525 | 1.500 | composition | fail | weak |
| 10 | `C13.T3.M16.L13` | mlp_down | T3 | L13 | 0.99774 | 0.0254 | 0.04642 | 0.582 | 3.281 | composition | fail | weak |
| 11 | `C13.T3.M8.L05` | mlp_down | T3 | L05 | 0.98532 | 0.0600 | 0.08764 | 0.410 | 3.062 | composition | fail | weak |
| 12 | `C13.T3.M4.L05` | mlp_down | T3 | L05 | 0.97351 | 0.0806 | 0.11037 | 0.166 | 2.625 | composition | fail | **dead** |
| 13 | `C04.T1.M16.L00` | conv_kernel | **T1/C04** | L00 | 0.99693 | 0.0752 | 0.07985 | 0.398 | **0.667** | composition | fail | weak |
| 14 | `C04.T1.M2.L00` | conv_kernel | **T1/C04** | L00 | 0.93668 | **0.4432** | 0.48293 | 0.113 | **0.333** | **macrocell-band** | fail | **dead** |
| 15 | `C06.T1.M8.L08` | attn_q | T1 | L08 | 0.99999 | *cat* | 0.00321 | **0.977** | 1.750 | — | **PASS** | strong |
| 16 | `C06.T1.M8.L12` | attn_q | T1 | L12 | 0.99999 | *cat* | 0.00228 | **1.000** | 1.750 | — | **PASS** | strong |
| 17 | `C06.T1.X4.L02` | attn_q | T1 | L02 | 0.99983 | *cat* | 0.00820 | 0.871 | 1.500 | free-run | fail | promising |
| 18 | `C06.T1.X4.L08` | attn_q | T1 | L08 | 0.99993 | *cat* | 0.00810 | 0.889 | 1.500 | free-run | fail | promising |
| 19 | `C06.T1.X4.L12` | attn_q | T1 | L12 | 0.99994 | *cat* | 0.00760 | **0.977** | 1.500 | — | **PASS** | strong |
| 20 | `C06.T5.M16.L08` | attn_q | T5 | L08 | 0.99955 | *cat* | 0.00767 | 0.883 | **0.938** | free-run | fail | promising |
| 21 | `C06.T5.M16.L12` | attn_q | T5 | L12 | 0.99961 | *cat* | 0.00440 | **0.977** | **0.938** | — | **PASS** | strong |
| 22 | `C07.T1.M8.L08` | attn_k | T1 | L08 | 0.99986 | *n/m* | 0.01515 | **0.984** | 1.312 | — | **PASS** | strong |
| 23 | `C07.T1.M8.L12` | attn_k | T1 | L12 | 0.99993 | *n/m* | 0.00946 | **0.994** | 1.312 | — | **PASS** | strong |
| 24 | `C07.T1.X4.L08` | attn_k | T1 | L08 | 0.99949 | *n/m* | 0.02072 | 0.746 | 1.125 | composition | fail | promising |
| 25 | `C07.T1.X4.L12` | attn_k | T1 | L12 | 0.99971 | *n/m* | 0.01772 | **0.918** | 1.125 | — | **PASS** | strong |
| 26 | `C08.T3.M16.L02` | attn_v | T3 | L02 | — | — | — | — | — | infra | **infra** | — |
| 27 | `C09.T3.M16.L02` | attn_o | T3 | L02 | 0.99989 | 0.0073 | 0.01161 | 0.760 | 1.875 | free-run | fail | promising |
| 28 | `C09.T3.M8.L02` | attn_o | T3 | L02 | 0.99959 | 0.0153 | 0.02302 | 0.707 | 1.750 | composition | fail | promising |
| 29 | `C09.T3.X4.L02` | attn_o | T3 | L02 | 0.99801 | 0.0346 | 0.04740 | 0.709 | 1.500 | composition | fail | promising |
| 30 | `C10.T1.M16.L02` | attn_per_head | **T1/C10** | L02 | 0.99942 | 0.0166 | 0.03047 | 0.736 | **0.958** | composition | fail | promising |
| 31 | `C10.T1.M8.L02` | attn_per_head | **T1/C10** | L02 | 0.99837 | 0.0309 | 0.04335 | 0.654 | **0.917** | composition | fail | promising |

*n/m* = the write is narrower than the residual and the runner records the
`#not-composable-from-write` sentinel. *cat* = dimensionally composable and
mechanically a category error (`NOTES-MX-E` §4.4).

**Grades:** 7 strong, 13 promising, 8 weak, 2 dead, 1 infra. 30 of 31 cells
carried decode-smoke text; 29 were reviewed **coherent** and one **incoherent**
(§4.5). Every calibration relL2 that was measured landed in
`0.000638 – 0.002227` against the 5e-3 tolerance, on 26 cells; the other five
declare the generic gate inapplicable (C04 ×2, C10 ×2) or never reached it
(C08.T3).

---

## 3. The T3 generalization verdict

### 3.1 Rule 2: the wave-2 surprise is a site result

`C05.T3.M16.L05` was wave 2's headline — free-run 0.914, the only non-attention
cell in the campaign to reach the 0.90 band, 0.17 above T1/T2 and 0.12 above T5
at the same component, level and site. `NOTES-MX-F` §5.2 flagged its rule-2
obligation (L00 and L13) as unrun and said so in the caveats. It has now run.

| conv_out_proj @ M16 | L00 | L05 | L13 | mean | rule 2 |
|---|---:|---:|---:|---:|---|
| T1 dense ridge | 0.686 | 0.744 | 0.734 | 0.721 | 0 of 3 |
| T5 sparse routes | 0.662 | 0.791 | 0.688 | 0.714 | 0 of 3 |
| **T3 balanced** | **0.699** | **0.914** | **0.646** | **0.750** | **0 of 3** |

**The 0.914 does not generalize.** T3 beats T1 by 0.170 at L05, by 0.013 at L00,
and *loses* by 0.088 at L13. On the mean it is 0.029 ahead of dense ridge, which
is inside the 0.13 site-to-site spread `NOTES-MX-E` §4.1 measured within a
single technique. The honest reading of wave 2's best cell is that the balanced
basis is worth 0.17 free-run **at one quiet mid conv layer** and nothing
measurable elsewhere on this component.

At `mlp_down` T3 is the opposite — consistent and consistently short: 0.586 at
L05 and 0.582 at L13, a 0.004 spread across two sites, both failing at
composition with drift 0.046–0.063. The wall is a property of the component and
T3 does not move it.

### 3.2 The T3 ladder, and where the balanced basis pays

Every rung at one site, so nothing here is a cross-site comparison.

| conv_out_proj @ L05 | M16 (960) | M8 (896) | M4 (768) | X4 (704) | M2 (512) |
|---|---:|---:|---:|---:|---:|
| T3 free-run | **0.914** | **0.777** | 0.506 | 0.330 | 0.277 |
| T1 free-run | 0.744 | 0.746 | — | 0.314 | 0.246 |
| T3 − T1 | **+0.170** | +0.031 | — | +0.016 | +0.031 |
| T3 observability energy | 0.9637 | 0.9282 | 0.8520 | 0.8096 | 0.6550 |

T3 leads at every rung it shares with T1 and the lead collapses from 0.170 to
0.031 the moment the level moves. Whatever the balanced criterion is buying at
M16, it is not a property that survives contraction.

### 3.3 Does balanced truncation generalize to attention writers? Only at depth

`C09` is the first attention component T3 has ever run on (its reader arm
refuses, §3.4).

| attn_o @ L02 | M16 (960) | M8 (896) | 768 |
|---|---:|---:|---:|
| T1 dense ridge | **0.820** | **0.736** | 0.445 *(M4, wave 2)* |
| T3 balanced | 0.760 | 0.707 | **0.709** *(X4)* |
| T3 − T1 | −0.060 | −0.029 | **+0.264** |

**The ordering inverts with depth of contraction.** Dense ridge wins the shallow
rungs; at 768 channels — the width the canonical exact-4x layout actually
assigns this component — T1 has fallen off a cliff to 0.445 and T3 is still at
0.709, essentially where it was at M8. That is the single most useful thing this
wave learned about technique selection: **the technique that wins at a marginal
level is not the technique that wins at composition scale**, and the matrix's
own priority order (marginal levels before X4) is exactly the order that hides
it.

The receipts say what the trade is. T3's retained *controllability* energy at
C09 is 0.99999 / 0.99988 / 0.99768 across the three rungs — it gives up
essentially no input energy — while its *observability* energy falls 0.933 →
0.855 → 0.758. Within one (component, site) the observability fraction orders
the cells the way free-run does at every component measured here. It does not
order across components (C09 at 0.758 behaves at 0.709; C05 at 0.810 behaves at
0.330), so it is a per-component diagnostic, not a criterion — the same shape of
finding as the composed gate in `NOTES-MX-F` §3.5.

### 3.4 T3's reader arm still refuses, by name

`C08.T3.M16.L02` seals `infra` with the refusal text:

> T3 balanced truncation needs a controllability lane and an observability lane
> over one shared state. C08 contracts its output axis, where the observability
> operator is the downstream consumer this cell does not own; substituting an
> identity observability would reduce balanced truncation to plain POD and
> duplicate T1 under a different name.

That is a result under the campaign's own rule, and it is the reason T3 has no
column at C06/C07/C08/C11/C12: three of the campaign's fourteen components are
writers and T3 covers those; the readers need an observability operator a
single-component cell does not own.

---

## 4. What the new cells said

### 4.1 C04 has evidence for the first time, and one tap is worth two

| cell | taps kept | subset residual (relative) | write cos | macro | free-run | stored |
|---|---|---:|---:|---:|---:|---:|
| `C04.T1.M16.L00` | **{1,2}** of {0,1,2} | 0.00661 | 0.99693 | 0.0752 | 0.398 | **0.667×** |
| `C04.T1.M2.L00` | **{1}** | 0.18470 | 0.93668 | **0.4432** | 0.113 | **0.333×** |

The tap-subset scores are the interesting part, because they are a statement
about the parent's filter that nothing in this estate had measured:

```
keep 2 of 3:  {1,2} 45.98      {0,1} 1161.0     {0,2} 1319.7
keep 1 of 3:  {1}  1283.9      {2}   1474.6     {0}   2687.6
```

Tap index 2 multiplies the current token and tap 0 the oldest. Dropping the
**oldest** tap costs 25× less than dropping either of the other two — the third
lag is nearly free. But when only one tap survives, the best one to keep is the
**middle** lag, not the current token: `{1}` beats `{2}` by 15%. The short conv
at L0 is not dominated by the current token; its one-step delay carries more of
the filter's response than the token in front of it.

Behaviourally the cell is `weak` (0.398) and the one-tap cell is `dead`, so C04
is not a component the composition wave should contract far. What it *is* is the
only component in the campaign whose marginal contraction is an unambiguous
storage win — 0.667× the parent tensor at M16, with a global 3-bit tap mask as
the only side information.

### 4.2 C10 is the campaign's first cell that behaves and contracts

Every dense marginal cell in this campaign stores more than the tensor it
replaced: T1/T2/T3's stored form is `r·(in+out)` against a dense `in·out`, which
on a square component only contracts below half width (`NOTES-MX-E` §4.1). The
storage column of this wave's table is 1.125× to 3.281× for 24 of its 30 cells.

A dropped head is not a factorization. It removes `head_dim` rows of `q_proj`
and `head_dim` columns of `out_proj` outright:

| cell | heads | dropped | free-run | vs control | **stored** |
|---|---|---|---:|---:|---:|
| `C10.T1.M16.L02` | 15 of 16 | {13} | **0.736** | 0.03047 | **0.958×** |
| `C10.T1.M8.L02` | 14 of 16 | {11, 13} | **0.654** | 0.04335 | **0.917×** |

Both grade `promising`, both fail at composition, and both are real
contractions. For the composition endgame that combination is worth more than a
higher free-run at 1.875× storage, because the composed child's budget is a
parameter count.

Two further readings. **Heads are not interchangeable** — the residual after
absorbing the cheapest head is 1.70e-4 of the write energy and the dearest is
1.19e-2, a 70× spread, and the two-head optimum {11,13} is not the two cheapest
singletons. And **the absorption is doing real work**: at drop-2 the solver's
condition number is 3.7e5 with the clamp not engaged, so the surviving columns
genuinely re-solve rather than sitting at the prior.

### 4.3 The X4 story: winners survive composition-scale width only at the deepest site

X4 is "the width the canonical exact-4x layout assigns this component", which
the registry refuses to guess. The widths used here are read off the **sealed
exact-4x materialization** `reports/exact4x-heterogeneous-blank-20260729/physical-materialization.json`
— 57,420,144 parameters, THE canonical budget of `docs/CURRENT_FOLDING_GOAL.md`,
descriptor `9733c1b5…`:

* `attention_geometry_in_causal_order` gives L2 and L12 both `(12 heads, 6 kv,
  64 dim)` → a query projection keeps **768**, a key/value projection **384**,
  and an O projection reads **768**;
* `conv_internal_widths_in_causal_order` gives L5 = **704**.

That artifact declares `fixture_is_not_selection_policy: true`, so these are the
widths of one sealed, materialized, authenticated witness at the canonical
budget and not a policy. Recorded, because a later layout could move them.

| component @ site | X4 width | free-run | verdict |
|---|---:|---:|---|
| `C06.T1` attn_q @ L12 | 768 | **0.977** | **PASS** |
| `C07.T1` attn_k @ L12 | 384 | **0.918** | **PASS** |
| `C06.T1` attn_q @ L08 | 768 | 0.889 | fail free-run |
| `C06.T1` attn_q @ L02 | 768 | 0.871 | fail free-run |
| `C07.T1` attn_k @ L08 | 384 | 0.746 | fail composition |
| `C09.T3` attn_o @ L02 | 768 | 0.709 | fail composition |
| `C05.T3` conv_out @ L05 | 704 | 0.330 | fail composition |
| `C05.T1` conv_out @ L05 | 704 | 0.314 | fail composition |

**Two components survive composition-scale contraction and both are attention
readers at the deepest site.** The depth gradient is monotone and steep for
attn_q — 0.871 / 0.889 / 0.977 at L02 / L08 / L12 — which is the same ordering
M16 and M8 show, only with 0.1 more spread. Neither clears rule 2 at X4: C06/T1
is 1 of 3 and C07/T1 is 1 of 2 attempted. `works to X4` is **not** claimable for
anything.

The conv writer is the other end of the story: at 704 channels both techniques
land at ~0.32, i.e. the component the exact-4x layout contracts by 31% loses
two thirds of its behaviour when contracted alone.

**One arithmetic coincidence, verified rather than assumed.** At L02 and L12 the
exact-4x layout keeps 12 of 16 heads, which is exactly 3/4 — the M4 level. So
`C06.T1.X4.L02` and `C06.T1.M4.L02` are the same rank, and the two receipts
agree to the last digit on write cosine (0.9998343147245656), composed drift
(0.008203554502268041) and free-run (0.87109375). Because that identity holds,
three planned X4 cells were **deliberately not run** — `C07.T1.X4.L02`,
`C08.T1.X4.L02` and `C09.T1.X4.L02` would have re-measured wave 2's M4 cells at
0.631, 0.553 and 0.445 under a different level label. The freed slots went to
the T3 ladder, the second T9 arm, and the C04 M2 floor.

### 4.4 The T9 verdict: over T1 it is provably the identity

`C05.T9.M8.L05` runs 1024 → 960 → 896, each rung solved against the previous
rung's child. The direct `C05.T1.M8.L05` runs 1024 → 896 against the parent.
Same rank, same site, same evidence, same GPU, one difference: the teacher.

| field | T9 two-step | direct one-step |
|---|---|---|
| `boundary.write_heldout_cos` | 0.9923561688351844 | 0.9923561688351844 |
| `boundary.write_heldout_relL2` | 0.11638884940453499 | 0.11638884940453499 |
| `composed.vs_copy_exact_control_delta` | 0.05391639576246887 | 0.05391639576246887 |
| `behavior.free_run.token_match_vs_control` | 0.74609375 | 0.74609375 |
| all three decode-smoke outputs | byte-identical | byte-identical |

**Bit-identical**, on every sealed field but one (the macrocell relL2 differs in
the last ULP, 0.03757709396274105 against …106). The receipt says why without
being asked: the rung-2 prior-anchored correction norm is **2.96e-13** against
rung 1's 4.68e-2. The second pass did nothing.

It could not have. On the input axis T1's basis is the eigenbasis of the design
Gram, so the rung bases are **nested**: `V₈₉₆ ⊂ span V₉₆₀`. Writing `G = XᵀX`
and `V` for a basis of `G`-eigenvectors, the prior-anchored normal system's
right-hand side is `VᵀG W₁ᵀ − VᵀGV VᵀW₁ᵀ = ΛVᵀW₁ᵀ − ΛVᵀW₁ᵀ = 0`, so every rung
returns exactly its projected prior and

```
W₂ = W₁ V₈₉₆ V₈₉₆ᵀ = (W_p V₉₆₀ V₉₆₀ᵀ) V₈₉₆ V₈₉₆ᵀ = W_p V₈₉₆ V₈₉₆ᵀ = W_direct
```

**For any technique whose contraction basis is a function of the inputs alone
and whose fit is the exact projection, progressive level scheduling is
identically a no-op.** That is not a null result about T9; it is a theorem the
cell confirms, and it means the entire progressive family has to be judged on
techniques whose basis depends on the parent.

So the second arm was run. T3's observability lane *is* the parent weight, which
the schedule replaces with the previous rung's rank-896 child, so its rung-2
basis genuinely differs from the direct one. `C05.T9.M4.L05` (1024 → M8(896) →
M4(768)) against the direct `C05.T3.M4.L05` (1024 → 768):

| field | T9/T3 two-step | direct T3 | Δ |
|---|---:|---:|---:|
| write cos | 0.98466 | 0.98758 | **−0.0029** |
| macrocell relL2 | 0.05465 | 0.04923 | **+0.0054** |
| vs control | 0.05978 | 0.05573 | **+0.0041** |
| free-run | **0.525** | 0.506 | **+0.020** |

The schedule moved every diagnostic in the wrong direction and free-run in the
right one by 0.020 — against a level step (M8→M4) that is worth 0.271 at the
same component. Two of three smoke outputs differ textually; none degrades.

**Verdict: stepping does not beat jumping.** Over a POD-basis technique the two
are the same object; over a parent-dependent basis the schedule is worth about
7% of one level's own effect, with a worse boundary and worse composition to pay
for it. Whatever value the progressive idea has is **not** in level scheduling
at a single component. Its remaining hypothesis is that the gain comes from
conditioning the *inputs* — which is composition mode M-B, and which an
isolation cell cannot test by construction, because every upstream neighbour
here is copy_exact and the input rows are the parent's either way. M-B is still
worth running; this wave says its mechanism cannot be the schedule.

### 4.5 The S8 band failed for the first time, and it was right

`NOTES-MX-F` §3.4 left the campaign's §5 rule-4 tally at "in band and behaved 1,
in band and did not behave 30, out of band 0" — the band had never been observed
to fail. Wave 3 adds 19 genuinely measured cells (both new kernels produce real
band readings: C04's write is `operator.output` and C10's is `attention.o`, and
both compose additively into `residual.post_layer`).

| class | wave-3 cells | reading |
|---|---:|---|
| **measured** | 19 | C05 ×9, C13 ×3, C09 ×3, C10 ×2, C04 ×2 |
| category error | 7 | C06 ×7 (`q_linear` matches the residual width and enters through a softmax) |
| sentinel | 4 | C07 ×4 — write narrower than the residual |
| none | 1 | C08.T3 never reached the boundary |

Of the 19 measured, **18 are in band and none behaved** (the best is 0.777), and
one is **out of band**: `C04.T1.M2.L00`, macrocell **0.4432** against the 0.4
threshold, free-run 0.113, and the wave's only `incoherent` review. It is the
first cell in the campaign to stop at chain link 4.

Campaign tally, all three waves: **in band and behaved 1; in band and did not
behave 48; out of band 1, and that one did not behave.** The band's positive
predictive value is 2% and remains useless as a criterion. Its *negative*
predictive value is 1 for 1 — one observation, worth exactly what one
observation is worth, and worth recording because it is the first time the
band has said anything at all.

### 4.6 Rule 2 at M8, and an ambiguity in rule 2 itself

| attn_q / T1 | L02 | L08 | L12 | passes |
|---|---:|---:|---:|---|
| M16 | 0.977 ✔ | 0.977 ✔ | 1.000 ✔ | **3/3** |
| M8 | 0.971 ✔ | **0.977 ✔** | **1.000 ✔** | **3/3** |
| M4 | 0.871 | — | — | 0/3 |
| X4 (768) | 0.871 | **0.889** | **0.977 ✔** | 1/3 |

| attn_k / T1 | L02 | L08 | L12 | passes |
|---|---:|---:|---:|---|
| M16 | 0.941 ✔ | 0.936 ✔ | 0.994 ✔ | **3/3** |
| M8 | 0.793 | **0.984 ✔** | **0.994 ✔** | **2/3** |
| M4 | 0.631 | — | — | 0/3 |
| X4 (384) | — | **0.746** | **0.918 ✔** | 1/3 |

`NOTES-MX-F` §3.1 named C06/T1's M8 rule-2 obligation (L08 and L12) as a wave-3
item and warned the rendered ledger's `works to M8` was credited off `1/1`. It
is now credited off **3/3**. Neither component is rule-2 complete at M4 or X4.

**C07/T1 at M8 exposes an ambiguity in rule 2 that this wave is the first to
hit, and it is reported rather than resolved.** §5 rule 2 reads "passes at ℓ
*and every shallower level* at ≥2 of its 3 sites". C07/T1 is 3/3 at M16 and 2/3
at M8, so the literal clause says `works to M8`. The renderer implements a
strictly stronger test — `good >= min(2, n) and good == len(rows_at)`
(`render_matrix_ledger.py:298`), i.e. **no attempted site may fail** — and
therefore prints `works to M16`. Both readings are defensible: the design's
number tolerates a known-failing site, the renderer's does not, and until now no
technique×component had ever produced a mixed rung for the difference to show
up in. The rendered ledger is the campaign artifact, so its `works to M16`
stands as the sealed verdict; changing rule 2 mid-campaign would need a dated
addendum, which is exactly what §5's preamble requires and is a coordinator
decision, not this wave's. What is not ambiguous either way: C07/T1's shallow
site is its worst (0.793 at L02 against 0.984 and 0.994 at L08 and L12), the
same depth gradient every attention cell in this campaign shows.

**T5 routes at an attention component.** `C06.T5.M16.L12` passes at 0.977 with
composed drift 0.00440 and stores **0.938×** — the first cell in the campaign
that passes §5 rule 1 *and* contracts storage. Its second site fails: 0.883 at
L08, so 1 of 2 attempted and rule 2 is not met. Dense ridge at the same site and
level reads 1.000 and stores 1.875×; the route gives up 0.023 free-run and
halves the storage bill relative to the factorization.

### 4.7 Smaller things the receipts said

* **All 31 cells ran one deployment tree.** `src_tree_sha 0642acc5837f5904`,
  `repo_head 9165e45c`, `sm86 NVIDIA GeForce RTX 3090`, 31 of 31. Zero P40 rows.
* **The C04 segmentation found the missing row.** Derived FIT sequence lengths
  are `{263, 264}` over 18 sequences — 4,751 rows, one short of 18×264 — which
  is exactly the arithmetic that made a fixed block length unusable. Nothing was
  told to the kernel; it read the anchor rows and the parent replay confirmed
  them at 0.00295.
* **Wall clock**: 31 cells in 20 minutes on two GPUs plus a 4-cell pilot, 1,408
  cell-seconds total, longest 177.7 s (`C13.T3.M8.L05`, the 2,560-wide balanced
  canonicalization). The 20-minute cap was never approached.
* **`C13.T3.M4.L05` is the wave's worst behaved cell at 0.166** and was still
  reviewed coherent with the degradation recorded: the dinner plan keeps its
  headings, lists broth and stock twice and herbs by the cup, and never reaches
  the cooking order the prompt asked for. Free-run grades it `dead` on its own;
  the rubric is not doing that work.

---

## 5. Winners per component — the composition wave's input

ADDENDUM 1's selection rule is the highest free-run-vs-control at the largest
level that passes, ties broken by composed-terminal margin; ADDENDUM 3 adds that
a component whose best is `promising` still ships its winner. Across all three
waves, over every sealed non-infra cell:

| component | winner | technique | level | free-run | vs control | **stored** | grade | cells run |
|---|---|---|---|---:|---:|---:|---|---:|
| C03 conv_in_proj | `C03.T5.M16.L00` | T5 routes | M16 | 0.787 | 0.02758 | **0.938** | promising | 1 |
| C04 conv_kernel | `C04.T1.M16.L00` | T1/C04 taps | M16 (2 of 3) | 0.398 | 0.07985 | **0.667** | weak | 2 |
| C05 conv_out_proj | `C05.T3.M16.L05` | T3 balanced | M16 | **0.914** | 0.02677 | 1.875 | strong | 22 |
| C06 attn_q | `C06.T1.M16.L12` | T1 ridge | M16 | **1.000** | 0.00148 | 1.875 | strong | 18 |
| C07 attn_k | `C07.T1.M16.L12` | T1 ridge | M16 | **0.994** | 0.00541 | 1.406 | strong | 10 |
| C08 attn_v | `C08.T1.M16.L02` | T1 ridge | M16 | 0.738 | 0.01539 | 1.406 | promising | 4 |
| C09 attn_o | `C09.T1.M16.L02` | T1 ridge | M16 | 0.820 | 0.01322 | 1.875 | promising | 11 |
| C10 attn_per_head | `C10.T1.M16.L02` | T1/C10 heads | M16 (15 of 16) | 0.736 | 0.03047 | **0.958** | promising | 2 |
| C11 mlp_gate | `C11.T5.M16.L00` | T5 routes | M16 | 0.400 | 0.09837 | **0.938** | weak | 1 |
| C12 mlp_up | `C12.T5.M16.L00` | T5 routes | M16 | 0.307 | 0.19453 | **0.938** | weak | 1 |
| C13 mlp_down | `C13.T3.M16.L05` | T3 balanced | M16 | 0.586 | 0.06250 | 3.281 | weak | 13 |
| C14 norms | `C14.T1.NA.L00` | T1 norm refit | NA | **1.000** | 0.00010 | 0.500† | strong | 1 |

† a bookkeeping artifact, not a compression (`NOTES-MX-F` §4.1).

Three storage-aware alternates the composition wave should have in view, because
the winner column above optimizes behaviour alone and a composed child is
budgeted in parameters:

| component | behaviour winner | storage-honest alternate | cost |
|---|---|---|---|
| C06 attn_q @ L12 | `T1.M16` 1.000, 1.875× | **`T5.M16` 0.977, 0.938×** | −0.023 free-run, half the storage |
| C09 attn_o @ 768 | `T1.M4` 0.445, 1.500× | **`T3.X4` 0.709, 1.500×** | +0.264 free-run at the same width |
| C05 conv_out @ M16 | `T3` 0.914 (L05 only), 1.875× | **`T5` 0.791 (L05), 0.938×** | −0.123 free-run, half the storage |

And four component-level cautions, all sealed:

1. **C05's winner is a single site.** T3 at M16 reads 0.699 / 0.914 / 0.646
   across L00 / L05 / L13. A composed child folds *every* conv layer; the
   winner's number is the best of three, not the expectation of thirteen.
2. **C06/C07's winners are the deepest site.** Both components' free-run falls
   monotonically toward L02 at every level. A composed child pays the shallow
   sites' numbers, not L12's.
3. **No winner passes at X4 except at L12**, and the two that do are 1 of 3 and
   1 of 2 on rule 2. Nothing in this campaign licenses `works to X4`.
4. **§5 rule 6 still binds.** Twelve single-component winners are twelve
   isolation results. ADDENDUM 4's M-A/M-B/M-C comparison is the experiment that
   decides whether they combine, and this wave's T9 result says the mechanism
   M-B is betting on cannot be level scheduling.

---

## 6. Ledger, backups and the destructive-change check

Three runners, three shards, one serialized appender — the discipline waves 1
and 2 established, for the same reason (`append_ledger` issues two `write()`
calls per row and `O_APPEND` makes each atomic, not the pair).

```
rows 126 -> 157 (+31)    cells, merged from ledger-{a,b,c}.jsonl
prefix verified byte-for-byte
rows 157 -> 187 (+30)    reviews, sealed from one process
```

**Two wave-3 cells re-run a cell id waves 1 and 2 had already sealed** —
`C04.T1.M16.L00` and `C10.T1.M16.L02`, both previously `infra`. The ledger keeps
every instance by design; the *files* would have collided, so the batch driver
writes a repeat into a `wave3/` subdirectory beside the earlier artifacts. No
wave-1 or wave-2 receipt was opened for writing, and `git status` over the
report tree shows exactly two modified files — `ledger.jsonl` and `LEDGER.md` —
plus 31 new cell directories.

**Batch C was adaptive, not planned.** Three cells (`C06.T1.X4.L08`,
`C07.T1.X4.L08`, `C06.T5.M16.L08`) were added after the first 28 sealed, because
three of them had cleared 0.90 at a single site and §5 rule 2 needs two. The
addition is recorded in the batch script with that reason.

**Codex's concurrent work, checked and not used.** During this wave Codex
advanced the branch to `59558ddc` ("Authenticate T5 sparse feature route
promotion"), which rewrites `src/neural_forge/matrix_technique_body_routes.py`
from 66,813 to 84,793 bytes, and left further uncommitted edits on top of it
(86,490 bytes in the working tree). **No wave-3 cell ran any of it.** The
deployed copy digests `055082c8a081317c`, which is the file at the wave-3
baseline `9165e45c` (identical at `347e8d93` and `c59be064`) — the same value
`NOTES-MX-E` §1 and `NOTES-MX-F` §6 recorded. This wave's two T5 cells
(`C06.T5.M16.L08`, `C06.T5.M16.L12`) executed that committed kernel, and so did
every earlier T5 cell quoted in §5. Nothing in that lane was edited or uploaded,
and Codex's new file was never deployed.

| what | where |
|---|---|
| deployment + ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/deployment-before-mxg-20260810T202634Z/` — 6 technique/receipt modules, `run_lfm2_matrix_cell.py` (`c5039a2a26ecbb5a`), `render_matrix_ledger.py` (`f903dc53dedd538a`), `ledger.jsonl.before-mxg` (`147656cb399013b3`) |
| ledger before the shard merge | `…/backup/ledger.jsonl.before-mxg-merge-20260810T205*Z` |
| ledger before the 30 review appends | `…/backup/ledger.jsonl.before-mxg-reviews-20260810T205*Z` |
| repo mirror ledger + LEDGER.md, pre-wave | scratch (`ledger.jsonl.before-mxg` `147656cb399013b3`, `LEDGER.md.before-mxg` `aa16d000c9efded9`) |

| check | result |
|---|---|
| ledger rows | 126 → 157 (cells) → 187 (reviews); **no row rewritten**, prefix verified after every append |
| mirror vs authority | byte-identical, `f26fd22bd7d6dfe1`; the pre-wave mirror is a byte prefix of it |
| receipt seals re-verified | **184 / 184** on the server, **185 / 185** in the repo, no failures |
| deployed file digests vs local | 4/4 identical after upload (`5dcd046ab019845e`, `86696b9fa1dc98e1`, `5ea35dc54d235982`, `a5513a89821ef5ff`); the five files this wave did not change still digest their MX-F values |
| sealed evidence base `2dec87ac…` | opened read-only; `find … -newermt` over the sealed tree returns nothing written during the wave. **Zero new evidence caches were cut** — every cell reused an existing authenticated per-(site, component) cache after re-deriving its payload digest against the recorded binding, and the C10 cells reused C09's cache because the two components now bind the same endpoint set |
| compute architecture in every wave-3 receipt | **31 / 31** `sm86 NVIDIA GeForce RTX 3090` |
| pilot runs | 4 cells into `/home/cereal/nf-mxg/pilot/` with their own ledger before the batches, to prove the three new kernels and the `--x4-width` path; that ledger was **not** merged |
| `/` free | 8.1 GB → 8.0 GB (no new caches) |

---

## 7. Verification

```
# local, .venv (python 3.11.9)
python -m pytest tests/test_matrix_cell_receipt.py tests/test_run_lfm2_matrix_cell.py \
                 tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py \
                 tests/test_matrix_technique_new_axes.py -q
# -> 122 passed, 1 skipped        (106 passed before this wave; +16 new, 2 rewritten)

python -m pytest tests/test_matrix_technique_body_routes.py \
                 tests/test_matrix_technique_lowrank_writers.py \
                 tests/test_matrix_technique_relation_matching.py -q
# -> 76 passed                    (courtesy check against the registry change)

python -m ruff check src/neural_forge/matrix_technique_{conv_fir,head_drop,progressive,registry}.py \
                     tests/test_matrix_technique_{new_axes,registry}.py
# -> All checks passed

python experiments/render_matrix_ledger.py --ledger reports/matrix-campaign-20260810/ledger.jsonl
# -> 187 rows render.  The repo's LEDGER.md is the SERVER render; re-rendering
#    locally reproduces it exactly except for the two lines that are supposed to
#    differ -- the render timestamp and the source path -- which is a 5-line
#    unified diff over 1,151 lines and was checked, not assumed.
```

Server: the eleven-file deployment digest check, a 4-cell pilot of the three new
kernels, 31 cell runs across two SM86 GPUs, the three-shard merge, 30 review
seals, the ledger render, 184/184 seal re-verification, and the free-space and
read-only-evidence checks.

**Not verified.**

* **Nothing is rule-2 complete at X4.** C06/T1 is 1 of 3 and C07/T1 is 1 of 2
  attempted; `works to X4` is not claimable for any technique×component.
* **C04 and C10 each have one site.** Both kernels ran only at the site their
  wave-1 cell named (L00 and L02). Their rule-2 obligations — C04 at L05/L13,
  C10 at L08 — are unrun, and the C10 kernel has never been asked to drop a KV
  head, which is the design's own second rung for that component.
* **C13/C11/C12 have no X4 cell and deliberately so.** The sealed exact-4x
  materialization records `mlp_intermediate_width_sum: 7926` over fourteen
  layers and no per-layer MLP width, so C13's X4 width is not determinable from
  the artifact. Guessing it would invent a layout the campaign never adopted,
  which is the one thing `_X4_WIDTHS` exists to prevent.
* **The T3+T5 hybrid at C13 was not run, by design.** The dispatch table
  resolves exactly one kernel per `(technique, component)` and has no stacking
  mechanism; a hybrid would be a new technique nobody registered, and inventing
  one mid-wave is the failure mode `NOTES-MX-F` §4 named for C04/C10. It is a
  technique-lane item with a cell id of its own.
* **T3 still has no reader arm**, so C06/C07/C08/C11/C12 have no T3 column and
  `C08.T3.M16.L02` is a refusal, not a measurement.
* **T9 has two arms at one component.** The no-op result over T1 is a proof and
  generalizes to any nested-basis exact-projection technique; the +0.020 over T3
  is one cell at one component at one level and is not a trend.
* **No composition-mode cell of any kind.** ADDENDUM 4 is the next wave and
  nothing here licenses a multi-component claim (§5 rule 6).

## 8. Paths

* wave-3 cells: `reports/matrix-campaign-20260810/cells/<cell-id>/` (29 new
  directories) and `…/cells/<cell-id>/wave3/` (2 re-runs: `C04.T1.M16.L00`,
  `C10.T1.M16.L02`) — each with `cell-receipt.json`,
  `cell-receipt-reviewed.json`, `cell-diagnostics.json`, `cell-<cell-id>.log`
* ledger + rendered report:
  `reports/matrix-campaign-20260810/{ledger.jsonl,LEDGER.md}` (mirrors
  `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`)
* run inputs: `reports/matrix-campaign-20260810/run/{mxg-knobs.json,mxg-knobs-t9.json,mxg-knobs-t9-t3.json,mxg-knobs-c04.json,run-mxg-batch.sh,run-mxg-pilot.sh,mxg-reviews.json,mxg_summarize.py,mxg_dump_smoke.py,mxg_seal_reviews.py}`
* new technique kernels: `src/neural_forge/matrix_technique_conv_fir.py`,
  `src/neural_forge/matrix_technique_head_drop.py`,
  `src/neural_forge/matrix_technique_progressive.py`, with
  `tests/test_matrix_technique_new_axes.py`
* registry change: `src/neural_forge/matrix_technique_registry.py:179` (C10's
  endpoints), plus two updated tests in `tests/test_matrix_technique_registry.py`
* X4 widths, sealed source:
  `reports/exact4x-heterogeneous-blank-20260729/physical-materialization.json`
* campaign store: `/mnt/ai_data/neural-forge-matrix/cells/`, run logs
  `/home/cereal/nf-mxg/logs/cell-*.log`, batch logs
  `/home/cereal/nf-mxg/batch-{a,b}.log`, backups
  `/mnt/ai_data/neural-forge-matrix/backup/` (see §6)
