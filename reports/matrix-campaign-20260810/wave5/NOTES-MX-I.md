# NOTES-MX-I — the X4 sweep: what folds at a real quarter, and what does not

**Wave 5 (MX-I), sealed 2026-08-11. Behaviour-first grading under ADDENDUM 6.**
Wave 5A answered the MLP question (`INTERIM-MLP-ANSWER.md`) and swept nine cells
before it ran out of budget. Wave 5B finished the sweep: eleven ungraded
dossiers graded, six S8 probes solved and reviewed, and every remaining
component taken to a true `S4` under every technique that can express it.

**Ninety-three cells now carry a sealed ADDENDUM 6 verdict.** The campaign has
its answer, and it is not the one the token-match era predicted.

---

## 1. The headline

**Nine of the fifteen component slots the campaign can measure fold at a real
quarter of their stored parameters with the whole model still inside the
parent's own quality band.** Three cells score **12 of 15 — dead level with the
parent** — at storage ratios of 0.2499×, 0.2490× and 0.2493×, on free-run token
matches of 0.066, 0.249 and 0.156.

| cell | component | stored / parent | **review** | vs parent | token match (diag) |
|---|---|---:|---:|---:|---:|
| `C13.T8.S4.L05` | mlp_down | **0.249853×** | **12 / 15** | **0** | 0.0664 |
| `C07.T5.S4.L12` | attn_k | **0.249008×** | **12 / 15** | **0** | 0.2363 |
| `C05.T8.S4.L05` | conv_out_proj | **0.249267×** | **12 / 15** | **0** | 0.1563 |
| `C05.T3.S4.L05` | conv_out_proj | 0.250000× | 12 / 15 | 0 | 0.2148 |
| `C05.T4.S4.L05` | conv_out_proj | 0.250000× | 12 / 15 | 0 | 0.2090 |
| `C11.T8.S4.L00` | mlp_gate | 0.248828× | 11 / 15 | −1 | 0.1934 |
| `C07.T1/T2.S4.L12` | attn_k | 0.249023× | 11 / 15 | −1 | 0.4141 |
| — *parent control* | *unfolded, same process* | 1.0000× | **12 / 15** | — | 1.0000 |

`C13.T8.S4.L05` is the cell to look at first. It agrees with the copy_exact
control on **one token in fifteen** and it is the only lane in the entire wave,
*the parent included*, that names the 23-degree axial tilt **and** calls the
distance idea incorrect. The metric said this child was further from the control
than almost anything else measured. The reading says it is the parent.

---

## 2. The complete S4 table

Verdicts are `behavior_verdict_for`'s, computed from counts the reviewer sealed
through `seal_matrix_behavior_review.py`. `stored/parent` is the receipt's own
`solve.storage_realized_ratio`, enforced by the runner against the 0.25 bound.

| component | site | T1 | T2 | T3 | T4 | T5 | T8 |
|---|---|---|---|---|---|---|---|
| C03 conv_in_proj | L00 | infra | infra | infra | — | infra | **dead** 0/15 |
| C05 conv_out_proj | L05 | **APB** 10 | **APB** 10 | **APB** 12 | **APB** 12 | **APB** 10 | **APB** 12 |
| C06 attn_q | L12 | **APB** 10 | **APB** 10 | infra | — | **APB** 10 | **APB** 10 |
| C07 attn_k | L12 | **APB** 11 | **APB** 11 | infra | — | **APB** 12 | **APB** 10 |
| C08 attn_v | L02 | **dead** 5 | **dead** 5 | infra | — | degraded 8 | **dead** 9 |
| C09 attn_o | L02 | degraded 8 | degraded 8 | degraded 9 | degraded 9 | degraded 9 | **dead** 5 |
| C10 attn_per_head | L02 | *cannot-express-4x* | — | — | — | — | — |
| C11 mlp_gate | L00 | **APB** 10 | **APB** 10 | infra | — | degraded 8 | **APB** 11 |
| C12 mlp_up | L00 | degraded 8 | degraded 8 | infra | — | **dead** 4 | degraded 9 |
| C13 mlp_down | L05 | **dead** 8 | **dead** 8 | impaired 7 | **APB** 10 | **APB** 11 | **APB** 12 |

`APB` = `alive-parent-band`. `—` = the technique is not registered for that
component. `infra` = the technique could not run the cell and nothing about the
model was measured. T7 and T9 publish no storage model at all and are registered
as refusals, not results: a technique that cannot state what it stores cannot be
planned at a storage-defined level.

`C04` conv_kernel and `C14` norms remain `cannot-express-4x` on geometry, from
wave 5A.

---

## 3. Five findings

### 3.1 Compressibility is a property of the component, not of the technique

The same solver is a triumph at one component and a catastrophe at another, at
the same bound, in the same wave. Balanced basis truncation (`T3`) scores **12 of
15** on `conv_out_proj` and **7 of 15** — impaired — on `mlp_down`. Gram relation
matching (`T8`) scores **12 of 15** on `mlp_down` and is **dead** at both
`attn_v` and `attn_o`. Dense prior-anchored ridge (`T1`), the reference
implementation, is `alive-parent-band` at five components and dead at two.

There is no ranking of techniques here. There is a ranking of *components*, and
it is stable across every technique that can express them:

**conv_out_proj > attn_k > mlp_down > attn_q > mlp_gate > mlp_up ≈ attn_o >
attn_v.**

### 3.2 `attn_v` is where the model actually breaks

The value projection is the only component whose technique row is mostly dead:
`T1` dead, `T2` dead, `T8` dead, `T5` alive-degraded at the A9 floor. And the
deaths are not close calls — `C08.T8.S4.L02` answers `multi-turn-correction` by
emitting `1 cup peanut butter` approximately **one hundred and forty times**
consecutively into the token cap, and `C08.T1`/`C08.T2` both terminate
`creative-continuity` on a bare `Keeper: (to the foghorn)` speaker tag with no
content at all.

Meanwhile `attn_q` and `attn_k` — the two projections whose output passes
through a softmax before anything reads it — fold at a quarter under *every*
technique that can express them. The write that survives contraction is the one
the model consumes as a relation; the write that does not is the one it consumes
as a value.

### 3.3 Counts and degeneracy disagree, and the clause earns its keep

Six cells are sealed `dead` in this wave. **Only one of them is dead on its
count.** `C08.T8.S4.L02` would have scored 9 of 15 — inside `alive-degraded`, one
case off the parent band — and it is dead because a response is nothing but a
loop. `C13.T1`/`C13.T2` would have been 8, `C09.T8` and `C08.T1`/`C08.T2` 5.

The standard applied uniformly across wave 5, and stated here so it can be
overturned deliberately rather than drifted:

> **The degeneracy clause fires when a RESPONSE IS THE LOOP** — the response
> carries no material other than its own repetition, as in `C13.T1.S4.L05`'s
> eight repetitions of "Good morning, sir", `C12.T5.S4.L00`'s sixty consecutive
> `than`, `C03.T8.S4.L00`'s sixteen identical responses, `C08.T8.S4.L02`'s
> hundred-and-forty peanut butters, or `C08.T1`'s empty speaker tag.
>
> **It does not fire on local phrase doubling inside otherwise-progressing
> text**, however heavy. `C06.T5.S4.L12` repeats "bell peppers" about
> twenty-two times across four ingredient lists and is sealed
> `alive-parent-band` at 10/15; the parent's own graded-12/15 lane cycles list
> phrases on the same prompt, and charging a fold for a failure mode the parent
> shows on the same case would be charging it for the model it was folded from.

### 3.4 T2 reproducing T1 is a per-cell fact, not a law

Wave 5A recorded that cross-width residual charts reproduce dense
prior-anchored ridge "to the last digit" on the MLP axes. That is true of the
*solve* and **not reliably true of the text**, and wave 5B checked it per cell
with a digest of the concatenated response lane rather than assuming it:

| component | child-lane digests | outcome |
|---|---|---|
| C05, C06, C09, C13 | **identical** | graded once, equivalence recorded in both receipts |
| C12 | 14 of 16 identical | graded separately, same 8/15 by different text |
| C11 | **completely different** | graded separately, same 10/15 by different text |
| C07, C08 | 13–14 of 16 identical | graded separately, same score |

Identical solve telemetry produced sixteen different responses at `C11`. The
equivalence has to be measured, not inherited.

### 3.5 A 4× of head count is a 2× of storage

`C10` drops sixteen query heads to **four** — a real four-fold reduction in head
count, with least-squares absorption of the twelve dropped heads into the
surviving O columns and a sealed parent-replay relL2 of 0.00165 — and the model
scores 8 of 15, `alive-degraded`. It stores **exactly 0.5×**.

`S4` is unreachable at this component *at any head count*, and the arithmetic is
worth stating because it is the coverage rule doing its job. The child still
stores `k_proj` and `v_proj` in full — 1,048,576 parameters — against an `S4`
budget of 786,432. Keeping **zero** query heads would still store 0.333×. The
kernel reports the KV groups a full head-drop orphans under `dead_kv_groups` and
deliberately declines to bank them, because claiming a saving the installed
child does not take is exactly the dishonesty §6.2 exists to catch.

`C10.T1.S4.L02` is therefore sealed `cannot-express-4x` from the kernel's own
count, not from a reading of the model.

One trap for the next wave: `--plan-storage-width` suggested width 384 for this
cell, because the registry has no per-component storage model for `(T1, C10)`
and falls through to the generic factor-pair algebra, which is the wrong algebra
for a head drop. The **planner** was wrong and the **runner** was right — the
enforced bound comes from `solve.stored_parameter_count`, and it refused. The
planner should get a `_STORAGE_MODELS_BY_COMPONENT` entry for C10 before anyone
plans that component again.

---

## 4. The S8 probes

ADDENDUM 6 §6.3 says to climb from an `alive-parent-band` S4. Six cells
qualified and all six were solved, installed, reviewed and sealed at
stored ≤ 0.125×.

| cell | component | stored / parent | S4 → S8 | verdict at S8 |
|---|---|---:|---:|---|
| `C11.T1.S8.L00` | mlp_gate | 0.124414× | 10 → **10** | **alive-parent-band** |
| `C11.T2.S8.L00` | mlp_gate | 0.124414× | 10 → **10** | **alive-parent-band** |
| `C11.T8.S8.L00` | mlp_gate | 0.124414× | 11 → 8 | alive-degraded |
| `C13.T4.S8.L05` | mlp_down | 0.124414× | 10 → 9 | alive-degraded |
| `C13.T5.S8.L05` | mlp_down | 0.124731× | 11 → 9 | alive-degraded |
| `C13.T8.S8.L05` | mlp_down | 0.124731× | 12 → **6** | **impaired** |

**`mlp_gate` reaches S8 inside the parent's band.** Rank 91 of 2560 — a 28×
width contraction, an 8× storage contraction — and the whole model still scores
10 of 15.

And the technique order **inverts** between rungs. At S4, `T8` is the best
mlp_gate solver (11) and `T1`/`T2` trail (10). At S8, `T1`/`T2` hold at 10 and
`T8` falls to 8. The same inversion is sharper at `mlp_down`, where `T8` is the
wave's best cell at S4 (12) and its steepest fall at S8 (6). **A technique's
rank at one level does not predict its rank at the next**, which means the
ladder has to be climbed, not extrapolated.

What S8 costs, consistently, is **long-horizon structure rather than local
fluency**. Both mlp_gate S8 lanes still write clean emails, correct
deduplication and calibrated uncertainty answers, and both collapse
`planning-dependencies` onto TLS — three of six steps become TLS steps and the
website backup disappears entirely.

---

## 5. The level ladder (the assembly wave's input)

Maximum surviving level per component, site-scoped per ADDENDUM 5 rule 2-S.

| component | max surviving level | at | verdict | stored |
|---|---|---|---|---:|
| C11 mlp_gate | **S8** | `T1`/`T2` @ L00 | alive-parent-band 10/15 | 0.1244× |
| C05 conv_out_proj | **S4** | `T8` @ L05 | alive-parent-band 12/15 | 0.2493× |
| C07 attn_k | **S4** | `T5` @ L12 | alive-parent-band 12/15 | 0.2490× |
| C13 mlp_down | **S4** | `T8` @ L05 | alive-parent-band 12/15 | 0.2499× |
| C06 attn_q | **S4** | `T1` @ L12 | alive-parent-band 10/15 | 0.2500× |
| C12 mlp_up | S4 (degraded) | `T8` @ L00 | alive-degraded 9/15 | 0.2488× |
| C09 attn_o | S4 (degraded) | `T3`/`T4`/`T5` @ L02 | alive-degraded 9/15 | 0.2500× |
| C08 attn_v | S4 (degraded) | `T5` @ L02 | alive-degraded 8/15 | 0.2490× |
| C10 attn_per_head | **S2** | `T1` @ L02 | alive-degraded 8/15 | 0.5000× |
| C03 conv_in_proj | none measurable | — | T8 dead, rest infra | — |
| C04 conv_kernel | *cannot-express-4x* | — | geometry | — |
| C14 norms | *cannot-express-4x* | — | geometry (diagonal) | — |

### Survivors — one per component, the assembly wave's input

1. `C05.T8.S4.L05` — conv_out_proj, **12/15**, 0.249267×
2. `C07.T5.S4.L12` — attn_k, **12/15**, 0.249008×
3. `C13.T8.S4.L05` — mlp_down, **12/15**, 0.249853×
4. `C11.T8.S4.L00` — mlp_gate, **11/15**, 0.248828×
5. `C06.T1.S4.L12` — attn_q, **10/15**, 0.250000×
6. `C11.T1.S8.L00` — mlp_gate at **S8**, 10/15, 0.124414× *(deepest rung reached)*

These six are the only cells whose installed tensors wave 5B retained; every
other wave-5 tensor artifact was digested into
`pruned-tensors-manifest.json` and deleted under Cyril's mid-wave cleanup
directive. All of them re-derive bit-exact from their sealed receipts.

**What assembly should expect to fight.** `attn_v` and `attn_o` will not follow
`attn_q` and `attn_k` down to a quarter, and `C03` has no measurable S4 at all.
A whole-model 4× claim built from this ladder is a *mixed-level* claim by
construction — §6.3 explicitly allows that, and this wave is what makes the
mixture concrete rather than hypothetical.

---

## 6. What this wave did not settle

- **`C03` is unmeasured, not refused.** Four of its five techniques failed
  before the model was touched: the generic contraction resolves
  `conv_in_proj`'s output width as 1024 when the tensor is 3072×1024 — three
  stacked B/C/x projection blocks over one internal channel axis — and dies on
  `mat1 and mat2 shapes cannot be multiplied (192x1024 and 3072x1024)`. That is
  a harness limitation, and calling it `cannot-express-4x` would be a lie about
  the storage algebra, which permits the width fine. The one technique that did
  run (`T8`, on the input axis) produced a child that failed the emit-zero floor
  and repeated one garbage string across all sixteen responses. C03 needs a
  kernel that understands the stacked layout before it can be graded.
- **S8 was not probed at `conv_out_proj`, `attn_k` or `attn_q`**, all of which
  cleared S4 at 10–12 of 15. On the mlp_gate evidence, at least one of them
  probably survives S8, and `conv_out_proj` — alive-parent-band under all six
  techniques — is the obvious next probe.
- **S2 was not measured for the degraded components.** §6.3 says to seal a
  maximum surviving level for components that die at S4; `attn_v`, `attn_o` and
  `mlp_up` all *survive* S4 in the degraded band, so their entries above are
  honest but they may well sit at parent band one rung up at S2.
- **Nothing here is a composition claim.** Every cell is one component in an
  otherwise-parent model, per §6.4. The six survivors have never been installed
  together.

---

## 7. Receipts and provenance

Review surface unchanged and digest-pinned before a token was generated:
protocol `e5e37768…`, prompt pack `0ee6bf14…`, greedy decoding at the
1,024-token cap, the parent decoded on the same sixteen prompts **in the same
process** as every child. The parent's response lane digests to
`aed8365ef86ac656` in **all sixty-one** dossiers of waves 5A and 5B — the
control did not move once.

Wave 5B re-graded the parent lane from scratch before grading any child and
reproduced wave 5A's calibration exactly: **5 useful / 7 partial / 3 observed
miss = 12/15**, with the misses at `arithmetic-multi-step` ("3 kits per box, 94
kits left over"), `logic-ordering` and `multi-turn-correction` (four peanut
butter options for a peanut-free request). Those three cases fail in nearly
every lane in the wave and are the parent's, not the fold's.

Every verdict was computed by `behavior_verdict_for` inside
`seal_matrix_behavior_review.py` from counts derived programmatically from the
per-case grading file, so a grading and the count sealed beside it cannot
disagree. The only verdict stated unilaterally is `C10.T1.S4.L02`'s
`cannot-express-4x`, which is a statement about storage algebra and involves no
text.

Campaign ledger: **322 rows**, `e212e7f6…`, byte-identical between
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl` and this repo. Backup of the
pre-wave-5B state at `ledger.jsonl.pre-mxi5b-20260811T154657`.

Deployment `neural-forge-matrix-eb32153e`, GPUs `sm86 NVIDIA GeForce RTX 3090`
indices 0 and 4 pinned by PCI bus id and re-checked by
`--require-compute-architecture` inside every sealing process, evidence base
`2dec87ac…`, knobs identical to waves 1–3.
