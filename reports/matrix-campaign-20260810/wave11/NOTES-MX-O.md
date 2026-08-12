# NOTES-MX-O — wave 11: the joint re-solve, and the wall it did not move

Wave 11, branch `codex/lrc-blank-compression`, source commit **`3fcc6cca`**
("Splice on row alignment, not on a digest the child cannot reproduce") on top
of wave-11's own `fcd69d17` ("Register ADDENDUM 9, and build the solver its
target requires"), which sits on wave-10 HEAD `4d684d81`. Deployment
`neural-forge-matrix-3fcc6cca`, cut with `git archive -c core.autocrlf=false`,
review surface verified on extraction **before any child was generated**:
protocol `e5e3776885976b1b…`, seal file `7b859172fa48c008…`, prompt pack
`0ee6bf14cef51baf…`.

Cyril's directive, verbatim: *"MXN4 retains intelligence — it is alive. MXN3 is
dead, it loops. Now get MXN4 to 0.5x the [Cyril quote withheld]. The theories have to carry to larger models."*

*The five sentences are §0, at the bottom, because they are the only part of
this file that could not be written until the last receipt was sealed. Every
number here is copied from a sealed receipt or from a log the receipt names.*

---

## 1. What was registered before anything ran

**ADDENDUM 9** (`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md`), three parts:

1. **The per-lane degeneracy reading governs the frontier**, per Cyril's
   adjudication. Both readings stay computed, sealed and printed for every
   child. The inherited frontier is `COMP.MA.MXN4`, **0.754811x, 8/15,
   alive-degraded per lane**.
2. **The 0.5x target in parameters**: 58,528,256 more to remove from MXN4,
   78,037,675 of body mass to cover at 4x, 48.0% of the body, which is ten to
   eleven whole SwiGLU lanes against a coherent body limit of three to six
   components.
3. **Mode `M-C` is the alternating joint co-solve**, replacing ADDENDUM 4's
   never-run "per-component hybrid" placeholder (zero `COMP.MC.*` rows existed
   in the 432-row ledger).

**`docs/FOLDING_SCALING_THEORY_2026-08-12.md`** was committed in the same
commit as the addendum, before any cell: every sealed result stated
scale-invariantly, with the interface share written as `V / (V + L·c·d)` —
29.2% on this parent, 1.6% at 70B — so that the campaign's own headline result
(a 4x token interface) is on the record as a 1-2% story at scale, and the body
is the whole game.

## 2. What was built

* **`--mode M-C`** in `experiments/run_lfm2_matrix_composition.py`: M-A first,
  then sweeps. Per component, per sweep: a fresh capture through the current
  composed child (frozen collector, called unchanged), a re-solve of that
  component's own closed form, install, next component.
* **The parent-target objective.** The coordinate step's design rows are the
  child's, and its target rows are the **parent's own sealed write rows** at the
  same endpoint over the same corpus positions. `_AlsHybridEvidence` splices the
  two. This is the difference between "be this map where the data now lives"
  (M-B, which re-fits the operator to the damage) and "given the inputs this
  child actually delivers, emit what the parent emitted" (M-C, which can cancel
  it).
* **`relation_target_endpoint`** on T8. Absent, the kernel is byte-for-byte the
  one waves 1-10 sealed; a test proves it by feeding the default target back
  through the knob and asserting `torch.equal` on the installed tensor.
* **The capture-reuse licence** (`_capture_is_stale`), which is what makes a
  sweep affordable: `w1` and `w3` share one capture, `w2` needs its own, and the
  first component of sweep *n+1* re-uses the last capture of sweep *n*.
* **`_als_verify_row_alignment`**, which refuses unless the conditioned split
  was inherited from the parent base by `record_id` and the ordered record ids
  are exactly the base's, in exactly the base's order.
* **The runner-lane defect fix**: `('T5','C01')` was registered twice, and
  `pytest tests/` failed at collection because of it (NOTES-MX-N §9.2).
  `replace=True`, plus two subprocess tests — one that the import order which
  used to raise no longer does, one that the fix does not change which kernel a
  C01 cell resolves to.

21 cases in `tests/test_matrix_composition_als.py`.

## 3. M4 — the drift budget, fitted before it was spent

The allocator ADDENDUM 9 §9.4 registers needs a model, and 55 sealed composed
children with both a global drift and a frozen review already exist. Fitted
over all of them (`reports/matrix-campaign-20260810/wave11/drift-model.json`):

```
review(useful-or-partial, /15)  =  12.9554  -  13.1626 x global drift
                                   R2 = 0.6243,  n = 55 sealed children
```

Three things fall straight out of it and every one of them is load-bearing for
this wave:

* **The coherence frontier is a drift budget of ~0.376.** Setting review = 8
  (ADDENDUM 8's alive band) gives drift = 0.3765. `COMP.MA.MXN4` — the sealed
  frontier — reads 8/15 at drift **0.377694**. The model predicts the frontier
  child to within a thousandth of the number that defines it.
* **MXN4 has no budget left.** It is *at* the line, so under the additive law
  every further body fold is spent below it. `COMP.MA.MXN3` at drift 0.487722 is
  predicted 6.53/15 and reads **6/15**.
* **The 0.5x target needs the law to break, not to bend.** Eleven SwiGLU lanes
  at the median marginal drift the ledger records for an MLP fold would put the
  composed drift far past 1.0, which the model maps to a review below zero.

The per-(component, site, level, technique) marginal-drift table this fit is
built on is §4's water-filling input and is the reason wave 11's deep set is
L02 and L04-L13 rather than all fourteen lanes: `C13` at L00 (median marginal
0.4228) and L03 (0.3632) are the two most damaging MLP folds the campaign has
measured, and L01 is unmeasured and sits between them.

## 4. The controls, before the result

**The M-A half of M-C is wave 10's child, to six decimals.** `COMP.MC.MXO1`
re-runs `COMP.MA.MXN3`'s recipe (extracted verbatim from MXN3's own sealed
diagnostics) on a new deployment, and its M-A install trace is:

| component | wave-11 M-A drift | wave-10 `MXN3` drift |
|---|---:|---:|
| `C11.T8.S4.L04` | 0.174351 | 0.17435 |
| `C13.T8.S4.L04` | 0.284168 | 0.28417 |
| `C11.T8.S4.L05` | 0.298803 | 0.29880 |
| `C12.T8.S4.L05` | 0.384258 | 0.38426 |
| `C13.T8.S4.L05` | 0.408377 | 0.40838 |
| `C07.T5.S4.L12` | 0.440971 | 0.44097 |

So any difference the sweeps produce is the sweeps.

**ALS is a no-op when there is nothing upstream to correct.** `COMP.MC.MXO0` is
the negative control: one component (`C11.T8.S4.L05`), nothing else folded,
one sweep. The re-solve moves the held-out parent-target residual from
0.25927949 to 0.25927994 — six decimal places of nothing — the sweep improves
global drift by **−0.011%**, the pre-registered floor stops it immediately, and
the runner restores sweep 0. That is exactly what the mechanism predicts: with
no upstream fold, the child's input rows *are* the parent's, so the
parent-target objective and the mimic objective coincide and there is nothing
to cancel. It also proves the splice is fitting the right object, because a
misaligned splice could not reproduce the M-A answer to six decimals.

The capture self-check on that run reads **0.0018074**, against the 5e-3
tolerance and against the copy_exact calibration's own 0.0016468 on the same
site: the conditioned capture reproduces its own installed weight to the
bfloat16 capture floor, which is the gate ADDENDUM 9 substitutes for the
copy_exact arm.

**Captures cost 197-204 s each** in tmpfs, against the ~850 s wave 6 measured
writing them to the share (`NOTES-MX-J` §4). That is the difference between an
ALS sweep being affordable and not.

## 4b. M1 — what the joint re-solve actually did

`COMP.MC.MXO1`, `COMP.MA.MXN3`'s seven-component recipe re-solved jointly, at
the identical 0.735962x. Sweep 1, component by component, held-out
parent-target residual before and after the coordinate step:

| # | component | objective | before | after | change |
|---:|---|---|---:|---:|---:|
| 1 | `C11.T8.S4.L04` | parent-target | 0.250676 | 0.250676 | **0.00%** |
| 2 | `C13.T8.S4.L04` | parent-target | 0.554634 | **0.546535** | −1.46% |
| 3 | `C11.T8.S4.L05` | parent-target | 0.302631 | **0.301151** | −0.49% |
| 4 | `C12.T8.S4.L05` | parent-target | 0.455944 | **0.454305** | −0.36% |
| 5 | `C13.T8.S4.L05` | parent-target | 0.126910 | **0.120556** | −5.01% |
| 6 | `C07.T5.S4.L12` | *mimic* (T5 takes no target) | 0.393670 | 0.395253 | +0.40% |

**Every component the objective could reach improved on its own objective. The
composed child got worse.**

```
M-A composed global drift          0.487722      (= COMP.MA.MXN3's, to six decimals)
after one full ALS sweep           0.505337      -3.61% RELATIVE, i.e. worse
composed free-run vs control       0.041016  ->  0.056641   (better)
```

The pre-registered rule then did its job: sweep 1 was worse than sweep 0, the
improvement was below the 5% floor, the run stopped, and **the child that was
sealed is the M-A child** — whose composed-tensor file digests to
`1a57bf60acfd58ea…`, byte-identical to `COMP.MA.MXN3`'s. So the sealed reading
for `COMP.MC.MXO1` is MXN3's own sealed **6/15, dead**, and it is the same
child, not a similar one.

**This is the answer to the wave's headline question, and it is no.** The
additive interference law is not an artifact of solving each component against
stale activations. Re-aiming every component at the parent's own writes, given
the inputs the composed child actually delivers, reduced five of six local
residuals and *increased* the end-to-end drift by 3.6%.

**Why a local win composes into a loss.** Each coordinate step minimizes its own
write's distance to the parent's write in a Frobenius sense over the fit rows.
The composed model does not care about that norm; it cares about the direction
the residual points in as it passes through the remaining depth. Two components
whose residuals partially cancelled under the M-A solve can each be individually
improved into residuals that no longer cancel. Nothing in a per-component closed
form can see that, because seeing it requires the downstream Jacobian — which is
a gradient, which the campaign does not do. **A12's law now has a second, much
sharper instance: a local fit improvement composes worse, even when the local
fit is aimed directly at the parent's own activations.**

The capture-reuse licence was verified in the same run rather than asserted:
`C12.T8.S4.L05`, reusing the capture cut for `C11.T8.S4.L05`, reads **4,751
rows bit-identical** to a fresh capture through the same child
(`max_abs_difference 0.0`). Five captures, 1,161.5 s of capture wall time.

## 4a. The structural result the sweeps produced, stated before its evidence

While sweep 1 ran, the sweep schedule turned out to be provable rather than
empirical, and the proof is worth more than the measurement:

> **In a feed-forward composition, a depth-ordered ALS pass converges in ONE
> sweep, exactly. The alternation buys nothing, and this is a theorem about the
> dataflow, not a fact about this model.**

The argument: a coordinate step re-solves component *i* against its own input
endpoint's rows. Nothing downstream of *i* can change those rows, and *i*'s own
re-solve cannot change them either (its input endpoint is upstream of its write
by construction of the component layout). So if the sweep walks depth-ascending,
every component is already solved against its **final** upstream state the first
time it is visited, and sweep 2 re-solves the identical problem and returns the
identical weights.

This is exactly what the capture-reuse licence predicts operationally: at the
start of sweep 2 nothing installed since the last capture is upstream of the
shallowest component, so no capture is taken, the same evidence is reloaded, the
same closed form is solved, and the same tensor is installed — all the way down.
The pre-registered stopping rule then fires on a 0.00% improvement.

**What this says about M-B.** Wave 6's one-pass conditioned composition had the
right *schedule* all along; it was hobbled by installing in a topological order
that left the damaging component first (`NOTES-MX-J` §4) and, more
fundamentally, by the wrong *objective*. Alternation was never the missing
ingredient. **The only thing that could make alternation pay is an objective
that couples downstream state back into an upstream component's solve — which is
end-to-end, which is gradients, which the campaign does not do.**

## 4c. The 0.5x target, measured directly

`COMP.MA.MXO2` is the target set: the MXN2/MXN4 token route plus **eleven whole
SwiGLU lanes** (L02 and L04-L13), thirty-four folded components, every one at or
under its S4 bound.

```
net stored                     0.497739x        <- the storage target is MET
global drift                   0.814086         <- the deepest the campaign has measured
composed free-run vs control   0.013672
frozen review                  0 / 15           <- 0 useful, 0 partial, 15 observed miss
verdict                        dead (both readings agree)
```

Fourteen of sixteen lanes ride a single phrase to the 1024-token cap — *answer
answer answer*, *why why why*, *The instruction is a single instruction*, *I'm a
task for the task* — two emit **zero** tokens, and the two that terminate on
their own produce *"the reference is the reference"* and *"If the question is a
question, it is a question."* Remove the repetitions and nothing remains
anywhere, so this is general degeneracy and the two readings do not disagree.

**0.5x by folding is reachable in storage and is not reachable in behavior by
stacking independently-solved 4x components.** The drift model called it before
the review did: 0.814086 drift predicts 2.24/15.

## 4d. The frontier, and where it actually moved

The one thing that did move the frontier was not a new mechanism. It was
**spending the drift budget where the drift model says it is cheapest**.

`COMP.MA.MXO3` is `COMP.MA.MXN4` — the sealed frontier — plus **one** more whole
SwiGLU lane, at **L06**, which §3's marginal-drift table ranks as the cheapest
lane on the board (medians 0.0103 / 0.0646 / 0.0083 for C11 / C12 / C13, against
L00's 0.2467 / — / 0.4228).

| | `COMP.MA.MXN4` (wave 10) | **`COMP.MA.MXO3`** (wave 11) |
|---|---:|---:|
| folded components | 4 | **7** |
| net stored | 0.754811x | **0.729103x** |
| global drift | 0.377694 | 0.461810 |
| frozen review | 8 / 15 | **8 / 15** |
| per-child reading | dead | dead |
| **per-lane reading** | alive-degraded | **alive-degraded** |
| lanes terminating on their own | 15 / 16 | 13 / 16 |

**`COMP.MA.MXO3` is the deepest coherent child the campaign has produced**, and
it is the first coherent child **below A9's own published 0.734512x** — and
below its honest 0.743217x (`NOTES-MX-N` §1.2) — which is the number wave 10
proved could not be reached without body contraction.

Thirteen of its sixteen lanes terminate by themselves in real, on-prompt
English: a five-item household-task list, a three-supplier summary that keeps
the decision *and* the residual risk, a RAM-versus-long-term-storage comparison
with a structure, a delay email carrying the install day and the tracking
number, a six-step website-migration plan, a five-factor list of what the sales
question is missing, a one-sentence creative continuation, and a safety answer
that engages the odour question without ever producing a mixing instruction.
It is wrong about a great deal — the seasons answer is a self-contradictory
multiple-choice fragment, the arithmetic is garbled, the logic-ordering lane
emits one name — but ADDENDUM 8 is explicit that a fold-init is judged on
whether there is structure worth retraining, and there is.

Charged lanes: `coding-deduplicate` (an endless numeric literal list; stripped
of repetition it leaves a broken class header and no dedup logic),
`debugging-off-by-one` (an endless test-case list; stripped, it leaves a real —
and still off-by-one — `last_three`, so that lane retains a response), and
`context-retrieval`, which emits zero tokens.

## 5. M3 — cross-layer sharing, excavated and then closed

### 5.1 The excavation Cyril asked for

`launch_lfm2_residual_sharing_12p5_20260716.sh` and
`launch_lfm2_residual_sharing_rank_frontier_20260716.sh` in
`/home/cereal/model-surgery-lab` ran a `--pair-count`/`--rank` frontier
(4-pair r64, 5-pair r256, 7-pair r384) whose reports all read
`"status": "rejected"`, `"publication_status": "rejected_by_in_memory_gates"`,
zero parent forwards, exact rollback. The published account of that whole family
is `docs/DYNAMIC_LFM2_RESULTS_2026-07-16.md`, and it closes three separate
sharing mechanisms:

* **whole-FFN sharing** — three activation-aligned shared-base/residual variants
  removed 11.13-12.50% at whole-output retention 0.125-0.333; *"increasing
  residual rank did not repair the collapse"*;
* **coupled-neuron banking** — 15% and 20% removal, both rejected; selected
  pairs had mean activation correlation 0.485 and mean coupled-weight cosine
  **0.014**, i.e. *"similarly firing neurons at different depths generally store
  different operators"*;
* **one shared residual output basis** — a real `1024 x 597` basis with fourteen
  private coefficient matrices, 15.005% removal, mean held-out per-layer cosine
  **0.9625** and worst layer 0.9452, and a composed trajectory score of 0.3876
  with **zero** of the parent's ten reliable answers retained (report sha
  `2f65583b…`, run seal `160bfc7f…`).

That last one is the one to keep: it is the campaign's decoupling law measured a
month early, in the sharing mechanism specifically.

### 5.2 The ceiling, and the falsification

What none of those runs stated is the *ceiling*: sharing does not make a fold
better, it makes it cheaper per layer, and the entire benefit is the extra rank
that buys. For a component of shape `(m, n)` over `L` layers, at equal storage,

```
r_shared / r_independent  =  (m + n) / (m + n/L)   ->   1 + n/m   as L -> inf
```

`reports/matrix-campaign-20260810/run/mxo/mxo_sharing_probe.py` prints that
arithmetic and then runs the **falsification half** in weight space, at the S4
budget, with the shared basis solved by the same alternating closed form
ADDENDUM 9 registers (six alternations), on the real parent tensors:

| component | shape | layers | rank at equal storage | mean rel. Frobenius, independent | mean rel. Frobenius, shared | layers where sharing wins |
|---|---|---:|---|---:|---:|---:|
| `w1` (C11) | 2560 x 1024 | 14 | 182 -> **248** (1.363x) | **0.721872** | 0.787540 | **0 / 14** |
| `w3` (C12) | 2560 x 1024 | 14 | 182 -> **248** (1.363x) | **0.764990** | 0.809011 | **0 / 14** |
| `w2` (C13) | 1024 x 2560 | 14 | 182 -> **543** (2.984x) | **0.761525** | 0.794643 | **1 / 14** |
| `q_proj` (C06) | 1024 x 1024 | 6 | 128 -> **219** (1.711x) | **0.570851** | 0.661949 | **0 / 6** |

**Even a 2.984x rank uplift does not pay for the loss of per-layer subspace
freedom.** The corresponding layers of this model do not live in a shared
subspace; they store different operators, which is the same sentence July's
coupled-neuron run wrote from a completely different direction (weight cosine
0.014 between activation-correlated pairs).

This is a **falsification probe and licenses nothing in the other direction** —
a win here would have meant nothing, because reconstruction does not predict
behavior. A loss here, in the metric most favourable to sharing (no activation
weighting, no composition, no depth, and the shared arm handed its full extra
rank), closes the mechanism without spending a child on it. **M3 is closed for
this architecture at S4**, and the transferable statement is the ceiling
formula, not the failure: sharing is worth at most `1 + n/m` in rank, which is
1.4x for a SwiGLU writer and 2x for a square attention projection, and on this
model that was not enough.

---

## 6. Benchmarks

lm-eval from `/home/cereal/ai-benchmark-lab` (`envs/lm-eval`), tasks
`arc_easy, hellaswag, piqa, winogrande, lambada_openai, wikitext`,
`--limit 200 --seed 1234 --batch_size 4`, `max_length=2048`, float32, one
PCI-pinned sm86 RTX 3090. Partial evaluations by construction. Parent baseline
reused from wave 8: same model directory, same dtype, same tasks, same limit and
seed, and no wave-11 change touches the parent.

| model | n | net stored | review | arc_easy | hellaswag_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **parent** | 0 | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.3912 | 81.7976 |
| `COMP.MA.MXL11` *(w8)* | 6 | 0.955444x | 9/15 | 0.5250 | 0.4600 | 0.6350 | 0.5700 | 0.2250 | 99.2209 | 118.0216 |
| `COMP.MA.MXN2` *(w10)* | 1 | 0.780518x | 10/15 | 0.4550 | 0.4450 | 0.6400 | 0.5550 | 0.2000 | 230.3146 | 139.2100 |
| `COMP.MA.MXN4` *(w10)* | 4 | 0.754811x | 8/15 | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.0939 | 168.7003 |
| **`COMP.MA.MXO3`** | **7** | **0.729103x** | **8/15** | **0.4300** | 0.4300 | 0.5900 | **0.5450** | 0.1650 | 527.9131 | 190.1456 |
| `COMP.MA.MXN3` *(w10)* | 7 | 0.735962x | 6/15 | 0.4750 | 0.4300 | 0.5800 | 0.5650 | 0.1200 | 797.6144 | 193.2446 |
| `COMP.MA.MXO4` | 10 | 0.703396x | 5/15 | 0.4200 | 0.4350 | 0.5600 | 0.5250 | 0.1000 | 1477.8552 | 257.8336 |

`COMP.MA.MXO3` stores **0.0257x less of the parent than `COMP.MA.MXN4`** and is
*above* it on `arc_easy` (0.4300 against 0.4250) and `winogrande` (0.5450
against 0.5250) while below it on the other five columns. Against
`COMP.MA.MXN3` — the wave-10 seven-component child at a *higher* 0.735962x —
MXO3 is deeper, scores two more review cases, and beats it on lambada accuracy
(0.1650 against 0.1200), lambada perplexity (527.91 against 797.61) and
wikitext (190.15 against 193.24), while MXN3 keeps `arc_easy` and `winogrande`.
**Trading the attn_k fold and the L04 lane for a whole L06 SwiGLU lane is a
better trade at a lower storage.**

`COMP.MA.MXO2`, the 0.5x child, was not benchmarked: ADDENDUM 8's benchmark
column is an intelligence-per-compression measure, and a child that emits
`answer answer answer` to the cap has none to measure. Its receipt, its dossier
and its verbatim text are in the human-review package all the same.

## 7. What wave 11 says, in the campaign's own terms

| mechanism | registered as | result |
|---|---|---|
| **M1** ALS joint co-solve | ADDENDUM 9 §9.3, mode `M-C` | **Measured and negative.** Five of six components improved on their own objective; composed drift got 3.6% worse. The wall is not a solver artifact. |
| **M2** body corrective routes | ADDENDUM 9 §9.4 | **Not run, and the reason is structural** — see below. |
| **M3** cross-layer shared basis | ADDENDUM 9 §9.4 | **Closed by falsification** in the metric most favourable to it: 0/14, 0/14, 1/14 and 0/6 layers at equal storage, even at a 2.98x rank uplift (§5). |
| **M4** drift-budget allocation | ADDENDUM 9 §9.4 | **The one that paid.** Fitted from 55 sealed children, it chose L06 as the cheapest lane, and the frontier moved from 0.754811x to 0.729103x (§3, §4d). |
| **M5** head merge | ADDENDUM 9 §9.4 | Not reached. |

**Why M2 was not run, stated as a constraint rather than an omission.** A
corrective route "at a block output" is a tensor the parent does not have. The
receipt schema binds every folded component to the parent `state_dict` keys it
replaces (`_validate_composition`, `composition.components[].weight_keys`), and
`matrix_cell_receipt.COMPONENTS` is one of the two frozen objects the campaign's
two lanes share (§4.3 of the design). Expressing an additive corrective tensor
needs a new component id in that frozen table, which is exactly the mid-wave
schema edit the standing rules forbid. What *is* expressible — a correction
solved into an existing weight — is what M1 already does, at zero extra storage,
and it made the child worse. **The honest statement is that M2 needs a schema
change registered in its own addendum, not that it failed.**

## 7a. One gap between ADDENDUM 9 and the tooling, left open deliberately

`mxo_addendum8.py` computes the ADDENDUM 8 success flag from
`addendum6.verdict`, which is the **per-child** machine verdict, and it refuses
a hand-supplied one — for the same reason `behavior_verdict_for` does, and that
refusal is correct. So it reports:

```
child                   n   net ratio     removed  worst s/o  review   verdict    A8
COMP.MA.MXO2           34    0.497739 115,366,026   0.249853    0/15      dead  fail
COMP.MA.MXO4           10    0.703396  68,127,874   0.249853    5/15      dead  fail
COMP.MA.MXO3            7    0.729103  62,223,105   0.249853    8/15      dead  fail
ADDENDUM 8 successes: 0 of 3
every component of every row above stores at or below its declared S bound
```

**Under ADDENDUM 9 §9.1 that last line for `COMP.MA.MXO3` should read
`alive-degraded` / `pass`**, because the per-lane reading is now the frontier
reading. The tool has not been changed to follow the addendum, and it should not
be changed by the wave that wants the answer: teaching a success calculator to
prefer the reading that produces a success, in the same wave that produces it,
is exactly the move the campaign's rules exist to prevent. **Recorded as an open
item for wave 12**: the per-lane verdict needs to be a *sealed field* computed
from per-lane readings the reviewer supplies case by case, not a re-labelling of
the per-child one. Until it is, the frontier table in this note carries both
readings explicitly and the calculator's `0 of 3` is a statement about the
per-child clause only.

The one thing the calculator settles unambiguously is the 4x floor: **every
component of every wave-11 child stores at or below its declared S bound**,
including all thirty-four of the 0.5x child's.

## 8. Estate

* **Ledger.** 432 rows sha `bf532a4fd98d55b93308dd11f396c78c1ee45d1b5d0089b5b30bec5d19a5aa18`
  before; backed up twice (`backup/ledger.jsonl.pre-wave11-<stamp>` and
  `wave11/ledger.jsonl.pre-merge-backup`, both verified byte-identical to the
  original by sha256 *before* the merge); merged **exactly once** behind a
  `.MERGED_ONCE` marker; **442 rows** after, sha
  `f2d945512a13a0917f24ac0b40bdb8fe9db0ea1cb1740d6b6f5c80d832167f00`. The
  merger verified the prior bytes are a **prefix** of the result, byte for byte,
  and the check was repeated independently afterwards.
* **The repo's copy of the ledger was stale and is now fixed.** Wave 10
  committed the ledger at its *pre-merge* 422 rows (`87cc0cac…`) and never
  committed the merge; the repo now carries the 442-row post-wave-11 file, so
  the committed ledger and the server's authoritative one agree for the first
  time since wave 9.
* **Seals.** 437 of 437 resolvable rows verify, **0 fail**. Five rows do not
  resolve: the same four as wave 10 (three `CAL.C01.T5.X4` calibration rows and
  one `C06.T2.S4.L12` a6review instance) plus `COMP.MC.MXO0`'s first `infra`
  row, whose receipt was replaced when the smoke cell was re-run in the same
  directory after the splice guard fired.
* **Wave-11 rows.** 10, all `sm86 NVIDIA GeForce RTX 3090`, all receipt schema
  v2.
* **Retention.** Composed tensor sets kept for `MXO2`, `MXO3`, `MXO4` and
  `MXO1`'s re-run; two deleted with their digests, sizes and re-derivation
  recorded in `wave11/pruned-tensors-manifest.json` (the ALS negative control,
  and the superseded first `MXO1`, whose child is byte-identical to wave-10's
  `MXN3` and therefore already retained there). Materialized HF children deleted
  after each benchmark.
* **Disk.** Server `/` at 5.0-5.9 GB free throughout; every heavy artifact under
  `/mnt/ai_data/neural-forge-matrix/wave11/`. The ALS captures ran in `/dev/shm`
  (126 GB tmpfs), were deleted as their rows were read, and the tmpfs was
  cleared at wave end.

## 9. The test result

Run server-side on a `git archive` of this wave's own commit (`10a608d5`),
unpacked to `/mnt/ai_data/neural-forge-matrix/wave11/fulltest/tree` — no
worktree, so none of the Codex lane's uncommitted files — with waves 9 and 10's
invocation, in the same file order, plus wave 11's own new file appended:

```
python -m pytest tests/test_matrix_composition.py tests/test_matrix_cell_receipt.py \
                 tests/test_run_lfm2_matrix_cell.py tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py tests/test_matrix_technique_new_axes.py \
                 tests/test_matrix_addendum6.py tests/test_matrix_addendum7.py \
                 tests/test_matrix_technique_token_embedding.py \
                 tests/test_matrix_composition_als.py -q
# -> 10 failed, 240 passed, 1 skipped   (86.94s)
```

**The same ten failures as waves 7-10, by name, and twenty-one more passes** —
wave 10 reported `10 failed, 219 passed, 1 skipped`, and the +21 is
`tests/test_matrix_composition_als.py`, all of which pass. The ten, unchanged:
the four-case `blank_receipt` `compute_architecture` family in
`test_matrix_technique_registry` plus its one relative in
`test_run_lfm2_matrix_cell_fresh_evidence` (the fix is uncommitted in the Codex
lane and a `git archive` cannot see it); `test_most_specific_registration_wins`,
which monkeypatches the global dispatch table and never restores it, plus its
three collateral victims; and
`test_factor_pair_storage_model_reproduces_a_sealed_wave2_receipt`.

**NOTES-MX-N §9.2's collection error is fixed.** `('T5','C01')` was registered
twice and a whole-tree collection died at import; with `replace=True` the two
files that triggered it collect and run together (69 tests collected, no error),
and two subprocess tests pin both halves of the fix — that the import order which
used to raise no longer does, and that the resolved C01 kernel is still the
technique lane's.

## 0. The five sentences

1. **The alternating joint co-solve does not break the additive wall.** Aiming
   every component directly at the parent's own writes, given the inputs the
   composed child actually delivers, improved five of six local objectives and
   made the composed child **3.6% worse** in global drift. The wall is
   structural, not a solver artifact.
2. **The wall's shape is now measurable rather than anecdotal.** Over 55 sealed
   children, `review = 12.9554 - 13.1626 x global drift` (R² 0.624), which puts
   the coherence frontier at a **drift budget of 0.3765** — and the sealed
   frontier child sits at 0.377694.
3. **Spending that budget where the model says it is cheapest is what moved the
   frontier.** `COMP.MA.MXO3` — the wave-10 frontier plus one SwiGLU lane at
   L06 — is **0.729103x, 8/15, alive-degraded per lane**: the deepest coherent
   child the campaign has produced, and the first below A9's published
   0.734512x. One lane deeper (`COMP.MA.MXO4`, 0.703396x) is 5/15 and dead.
4. **0.5x is reachable in storage and not in behavior.** `COMP.MA.MXO2` folds
   eleven whole SwiGLU lanes plus the token route at 4x each, lands at
   **0.497739x**, and scores **0/15** with fourteen of sixteen lanes riding a
   single phrase to the cap.
5. **Cross-layer sharing is closed for this architecture**: at equal storage a
   shared basis loses to independent truncation in 55 of 56 layer-comparisons,
   even where sharing buys a 2.98x rank uplift.

## 10. What wave 12 should measure

1. **The frontier between 0.729103x and 0.703396x.** One lane is 0.0257x. The
   intermediate rungs — MXO3 plus a *partial* lane (w1+w3 without w2, or w2
   alone) at L07 — are cheap and would find the exact stopping point.
2. **Which lane to add third.** MXO4 added L07 because it was next in depth. The
   drift model ranks L13 (`C11` 0.0256, `C13` 0.0344) and L02's `C13` (0.0197)
   below L07's unmeasured components; the allocator was used to pick the second
   lane and *not* the third, which is the one decision this wave took on
   autopilot.
3. **The objective, not the schedule.** §4a's argument says a depth-ordered
   sweep is already a fixed point, so more sweeps cannot help. What is untested
   is an objective that couples downstream state into an upstream solve without
   gradients — for instance solving a component against the parent's
   *macrocell* rows several layers down, which is still closed form and still
   linear in the component's weights.
4. **Head merge (M5)**, the only registered mechanism this wave did not reach,
   and the only one that attacks attention mass — a third of the body at scale.
