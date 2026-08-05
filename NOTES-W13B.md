# NOTES-W13B — layer-0 representability vs solver, single-boundary refit study

Wave 13B, follow-up to NOTES-W13A. NOTES-W13A located the sealed r12 exact-4x
child's wall at **layer 0**: heldout post-layer relative L2 1.159 against a
1.0 emit-zero baseline, 93% irreducible direction error, and a layer-0
short-conv **write** with cosine 0.171 against the chart-projected parent write.
This wave answers the question that diagnosis left open:

> Is that deficit **representational** — r12's exact-4x layout cannot express
> the parent's layer-0 function under any closed-form solve — or **solver-side**
> — a representable target that was fit badly?

**Verdict: solver-side, decisively, and not by any hyper-parameter.** The
production closed-form solve, run at r12's exact width on r12's exact sealed
evidence and sealed chart, reproduces the parent's layer-0 write to **held-out
cosine 0.9995 and relative L2 0.0245**, with the composed post-operator residual
at **0.0181**. That is 46x better than the sealed child achieved (0.836) and
5.3x better than the capacity bank's own predicted per-layer defect (0.0965).
Contraction to 816 is not the binding constraint, and no solver knob is either —
every arm at width 816 lands on the same number.

The r12 child's layer-0 write is destroyed in two identifiable stages, and the
larger of the two is **the moving-gauge body reclosure**, which left layer 0's
`in_proj` byte-identical and rewrote only its output writer, multiplying that
writer's Frobenius norm by 4.73x while dropping the write's cosine from 0.911
to 0.171.

**Zero training, zero gradients, zero parent forwards, zero child forwards.**
Every arm is one closed-form producer fit plus one closed-form consumer fit
through the production compilers. Every record carries
`parent_forward_calls: 0`, `child_forward_calls: 0`, `training_steps: 0`,
`backward_calls: 0`, `optimizer_steps: 0`. Nothing under the sealed r12 run
directory, the sealed trace cache, or the sealed balanced-basis phase cache was
written: `find … -newermt "2026-08-05 04:00"` returns 0 entries for all three.

---

## 1. Harness

`experiments/run_lfm2_layer0_representability_study.py` (new),
`tests/test_run_lfm2_layer0_representability_study.py` (new). No existing
`src/` or `experiments/` file was modified; the adapter and the catalog runner
are imported, never edited.

### What one arm does

1. Open the sealed parent trace cache through
   `Lfm2CircuitTraceSelectiveAuthenticatedSession` (content-hashes all 85 GB;
   ~13 min) and load the sealed balanced-basis phase bound to that session's
   digest, with the HMAC authority the r12 launcher used. FIT/heldout comes
   from `authoritative_split_by_batch` — 192 fit / 64 heldout, the same split
   the first-divergence lane used.
2. Take the layer's sealed parent rows: `operator.input`, `operator.output`,
   `residual.pre`, `residual.post_operator`, `short_conv.{b,c,x,context}`.
3. **Construct the child's layer-0 input as `encode(parent residual.pre)`.**
   Justified by the first-divergence ledger's `c_in` row (relL2 0.000000) and
   independently re-verified here against the r12 fold's *own* child replay
   cache (`cache/child-final`): over batches 1/5/29/100/205, all 256 prefill
   rows each, mean relative L2 **2.53e-08** and cosine 1.0. The fold's layer-0
   solve saw exactly the (input, target) pairs this study reconstructs.
4. Fit the child operator RMSNorm with `fit_diagonal_norm` (rms_norm, eps 1e-5,
   ridge 1e-6) on FIT rows only; materialize the child's `operator.input`.
5. Build cases with the production adapter
   `cached_short_conv_case_from_lfm2_trace`, and the sparse operator-local
   transport with the production rule
   `Lfm2DynamicClosedLoopAdapter._build_short_conv_assignment` (imported
   unbound, applied to the chart — assignment digest
   `9f1160809f…`).
6. `compile_lfm2_structural_short_conv_producer_fold` → run the compiled
   B*x → depthwise FIR → C*v path with the production
   `_run_compiled_short_conv_contexts` → `compile_lfm2_structural_short_conv_consumer_fold`.
7. Score the branch write `out_proj(context)` and the composed
   `residual.pre + write` against the chart-projected parent, on heldout
   batches only, in two row sets: **`last_token`** (one row per record-phase,
   the exact 576 states the first-divergence ledger aggregates) and
   **`all_rows`** (all 16,896 full-prefix heldout rows).

### Geometry that matters

r12's layer 0 contracts the **residual** 1024 → 816 but keeps the short-conv's
**internal channel width at 1024** (`short_conv_internal_width: 1024`,
`short_conv_writer_rank: null`, dense 816x1024 writer). Only layer 7 has a
factor writer in the whole child. So layer 0's operator-local channel
assignment is the identity and the entire contraction at this boundary is the
residual chart.

### Evidence accounting

The sealed cache stores prefill as up to 256 rows and each of the 8 forced
decode phases as exactly **one** row: 264 rows per record, 50,687 FIT rows and
16,896 heldout rows at layer 0. Because decode phases are single rows, the FIR
history has to carry across phases exactly as
`cached_short_conv_case_from_lfm2_trace` declares it
(`reset_state = not phase.startswith("decode")`); resetting per phase runs every
decode row against an empty filter window. Getting this wrong was the harness's
one real bug and it moved the calibration number from 0.204 to 0.171 — see §2.

### Cost

Sealed-session open ~13 min (once; then reused through an authenticated
evidence cache that re-derives the payload digest and the sealed chart tensor
digest). Each arm ~14 min, dominated by the production
`_weighted_group_rows` Python loop, which is O(target x source) per case and
does nothing at this boundary because the assignment is the identity. All arms
are dense fp64 **CPU** linear algebra; `CUDA_VISIBLE_DEVICES=""` throughout, so
no GPU was taken from the live on-policy round.

---

## 2. Reproduction check — the harness measures what the ledger measured

Arm `published_child` runs no solve. It loads the sealed published child's own
layer-0 `in_proj`/`conv`/`out_proj`/`operator_norm` and measures them under the
construction above. Held-out, 576 last-token states:

| metric | this harness | sealed first-divergence ledger |
|---|---|---|
| write mean cosine | **0.1712** | 0.1711472944060249 |
| write mean relative L2 | **1.0986** | 1.0986789052632018 |
| write mean norm ratio | **0.6506** | 0.651 (NOTES-W13A §2) |
| `residual.post_operator` mean relative L2 | **0.8358** | 0.8358 |
| row count | 576 | 576 |

Four independent statistics to four significant figures, from parent-cache rows
plus the chart alone. That simultaneously authenticates the exact-input framing,
the chart projection, the RMSNorm and short-conv semantics, the row selection
and the metric. Everything below is believable to the same standard.

The synthetic tests add the other half of the calibration: on a boundary that is
representable by construction the same code path solves to cosine > 0.999, and
on a chart that truncates live directions it does not — so a low measured cosine
is a real result, not a harness artifact.

---

## 3. The r12 configuration, re-solved

Arm `r12_reproduction`: width 816, the sealed `df7f6e88…` chart, the production
sparse argmax residual assignment, source-derived priors, `ridge = 1e-6`,
`max_prior_delta_ratio = 1.0` — r12's adapter defaults, unchanged.

| row set | split | write cos | write relL2 | write norm ratio | post_operator relL2 |
|---|---|---|---|---|---|
| last_token (576) | heldout | **0.99953** | **0.02452** | 0.99752 | **0.01807** |
| last_token (1,728) | fit | 0.99943 | 0.02636 | 0.99665 | 0.01930 |
| all_rows (16,896) | heldout | 0.99774 | 0.04394 | 0.99856 | 0.03248 |
| all_rows (50,687) | fit | 0.99841 | 0.03975 | 0.99838 | 0.02944 |

Heldout tracks fit; there is no overfitting to find. **100% of heldout rows beat
the emit-zero baseline** (the sealed child manages 9.5%). Both prior-anchored
fits report `update_scale = 1.0` — the prior-delta clamp does not bind.

Set against the sealed child at the same boundary, same rows, same metric:

| | write cos | write relL2 | post_operator relL2 | rows beating zero baseline |
|---|---|---|---|---|
| **this study's solve** | 0.9995 | 0.0245 | 0.0181 | 100.0% |
| r12 published child | 0.1712 | 1.0986 | 0.8358 | 9.5% |
| ratio | — | **44.8x** | **46.3x** | — |

R13 item 4's own prediction for this layout was a defect of 0.0965 per layer.
The solve lands at **0.0181** — 5.3x *better* than predicted, not 8.7x worse.

**Item 4's premise does not hold at layer 0.** The "~70% unrepresentable
post-activation tail" figure is about the 2560 → 321 MLP contraction, and it was
carried over to explain an operator-write failure it does not govern: at layer 0
the short-conv's internal width is not contracted at all.

---

## 4. Width ladder — the representability curve

The ladder must isolate width from chart identity, so it does not rebuild a
balanced basis per width (that re-canonicalizes the subspace and confounds the
two). It **extends the sealed r12 chart**: chart(W) = [sealed 816 columns |
top (W−816) residual-energy directions of the sealed chart's orthogonal
complement, measured on the same FIT rows, Gram-Schmidt'd twice against the
retained columns]. Every rung strictly contains the one below it and the bottom
rung is bit-identical to the sealed chart. Solver held at r12's settings.

| target width | contraction | write cos | write relL2 | post_operator relL2 |
|---|---|---|---|---|
| **816 (r12)** | 1024 → 816 | 0.9995 | **0.0245** | **0.0181** |
| 864 | 1024 → 864 | 0.9997 | 0.0194 | 0.0143 |
| 912 | 1024 → 912 | 0.9998 | 0.0143 | 0.0106 |
| 960 | 1024 → 960 | 0.9999 | 0.0101 | 0.0075 |
| 1008 | 1024 → 1008 | 1.0000 | 0.0054 | 0.0040 |
| 1023 | 1024 → 1023 | 1.0000 | 0.0031 | 0.0024 |

Smooth, monotone, and *entirely inside the noise of the question being asked*.
Widening from 816 to 1023 — giving up essentially the whole 4x parameter
budget — buys a factor of 7.5 on a quantity that is already 46x better than the
failure. **The minimum width at which heldout relative L2 beats the 1.0 floor
with margin is 816, r12's own width**, by a factor of 41 on the write and 55 on
the composed post-operator residual.

Item 4's functional-width contraction cap is therefore **not** the lever that
moves layer 0. It may still be the right call for the MLP boundary (untested
here) and for the deeper layers, but it cannot be justified by layer 0.

Caveat: this ladder measures the *chart's* residual contraction only. It holds
the short-conv internal channel width at the parent's 1024, exactly as r12's
layout does, and it says nothing about the 2560 → 321 MLP contraction.

---

## 5. Solver levers — none of them move anything

All at width 816, all heldout last-token, all reported as
`write cos / write relL2 / post_operator relL2`:

| arm | knob | result |
|---|---|---|
| `r12_reproduction` | source-derived priors, delta 1.0, ridge 1e-6 | 0.9995 / 0.0245 / 0.0181 |
| `solver_prior_delta_unclamped` | delta = None | 0.9995 / 0.0245 / 0.0181 |
| `solver_prior_delta_4` | delta = 4.0 | 0.9995 / 0.0245 / 0.0181 |
| `solver_chart_priors_clamped` | dense-chart priors, delta 1.0 | 0.9995 / 0.0245 / 0.0181 |
| `solver_chart_priors_unclamped` | dense-chart priors, delta None | 0.9995 / 0.0245 / 0.0181 |
| `solver_zero_prior_unclamped` | zero priors | 0.9995 / 0.0245 / 0.0181 |
| `capsule_prior_clamped` | **the r12 capsule's own layer-0 weights as prior**, delta 1.0 | 0.9995 / 0.0245 / 0.0181 |
| `capsule_prior_unclamped` | same, delta None | 0.9995 / 0.0245 / 0.0181 |
| `solver_ridge_0` | ridge 0 (gelsd) | 0.9995 / 0.0245 / 0.0181 |
| `solver_ridge_1e-08` / `1e-04` / `1e-02` | ridge sweep | 0.9995 / 0.0245 / 0.0181 |
| `r12_repro_fp32_storage` | fp32 weight templates, exactly as production stores them | 0.9995 / 0.0245 / 0.0181 |
| `width_768_r12_solver` (misnamed; ran at 816, second independent process) | determinism replica | 0.9995 / 0.0245 / 0.0181 |

The boundary is 50,687 rows against an 816-wide design; the least-squares
solution is over-determined, well conditioned, and prior-independent. Ridge over
seven orders of magnitude, prior choice, prior-delta clamp and fp32 storage all
land on the same solution. **`capsule_prior_clamped` is the pointed one:**
anchored on the r12 child's *own* layer-0 weights, with production's clamp
active, one pass of the production solver walks straight from relL2 0.6425 to
0.0245 without the clamp ever binding. The weights r12 shipped are not a point
the production solver could not leave.

The only knob that moves the number moves it the wrong way, and it is worth
recording because it is the shape of the damage seen in §6:

| `max_prior_delta_ratio` | in_proj update_scale | out_proj update_scale | write cos | write relL2 | write norm ratio | post_operator relL2 |
|---|---|---|---|---|---|---|
| 1.0 / 4.0 / None | 1.000 | 1.000 | 0.9995 | 0.0245 | 0.998 | 0.0181 |
| 0.50 | 0.643 | 0.625 | 0.9848 | 0.1884 | 0.916 | 0.1430 |
| 0.25 | 0.321 | 0.292 | 0.9313 | 0.3681 | 0.881 | 0.2791 |
| 0.10 | 0.129 | 0.111 | 0.8693 | 0.4947 | 0.906 | 0.3743 |

Shrinking the correction produces exactly the signature of *under-writing with
retained direction*: cosine stays high, norm ratio falls, relL2 climbs.

### Item 7's EIV ridge and margin weights cannot touch this boundary

This is a structural fact about the wiring, not a measurement, and it is worth
stating plainly because item 7 was nominated as a layer-0 lever:

- `Lfm2DynamicClosedLoopAdapter._mlp_input_error_covariance` returns
  `(None, 1.0, None)` whenever `self._accepted_layer_input_error is None`
  (`hf_lfm2_dynamic_closed_loop_adapter.py:6589-6591`), and that stash is only
  written by `_record_accepted_layer_input_error` after a layer is accepted
  (`:10893`). **At the first folded layer it is always `None`,** so the EIV
  multiplier sweep {0, 0.25, 1, 4} is a strict no-op at layer 0 by construction.
- Even if it were seeded, the covariance is estimated from the previous layer's
  parent-vs-child post-layer drift, and layer 0's input drift is *measured at
  2.53e-08* on the fold's own child replay. There is no input error to be
  robust to.
- The covariance only ever reaches the MLP W1/W3 compile
  (`:10659-10666`); it is never passed to
  `compile_lfm2_structural_short_conv_producer_fold` or
  `…_consumer_fold`, so no multiplier can change an operator write at any depth.
- Margin row weights replace the uniform profile in `_moving_gauge_row_manifest`
  (`:5086`), i.e. the **body-reclosure** row alignment, not the per-layer
  boundary solve.

Item 7 is not refuted — it is simply pointed at a different site than the one
that failed. Its margin-weight half lands squarely in the reclosure lane, which
§6 identifies as the dominant damage, so it should be judged **there**.

---

## 6. Stage attribution — where the write actually dies

The r12 run kept two restart capsules. `capsule-000013` holds the child exactly
as the fourteen per-layer closed-loop boundary solves left it, **before**
`child_closed_loop_body_reclosure_started`. Measuring its layer 0 with the same
harness, and comparing against both ends:

| pipeline stage | write cos | write relL2 | write norm ratio | post_operator relL2 | heldout rows beating the zero baseline |
|---|---|---|---|---|---|
| this study's closed-form solve | **0.9995** | **0.0245** | 0.998 | **0.0181** | 100.0% |
| r12 after the 14 boundary solves (capsule 12 == capsule 13) | 0.9106 | 0.6425 | **0.4185** | 0.4936 | 100.0% |
| r12 published child (post body reclosure) | **0.1712** | **1.0986** | 0.6506 | **0.8358** | 9.5% |

Two distinct failures, with different signatures:

**Stage A — boundary solve, 26x.** The per-layer pass ended at relL2 0.6425
where 0.0245 was available. Its signature is *scale*, not direction: cosine
0.911, norm ratio 0.419, and every heldout row still beats the emit-zero
baseline. That is the shape §5's clamp sweep produces, at an effective update
scale of roughly 0.4.

**Stage B — moving-gauge body reclosure, direction collapse.** Frobenius norms
of the layer-0 weights, from the sealed artifacts
(`reports/layer0-study-20260805/remote/clamp-forensics.txt`):

| tensor | capsule 13 | published child |
|---|---|---|
| `model.layers.0.conv.in_proj.weight` | 34.10311231 | **34.10311231** (byte-identical) |
| `model.layers.0.conv.out_proj.weight` | 26.8826 | **127.2283** (4.733x) |

The reclosure did not touch layer 0's input projection at all. It re-solved only
the writer, made it 4.73x larger, and in doing so took the write's cosine from
0.911 to 0.171 and pushed the composed residual from 0.494 through the 1.0
emit-zero floor to 0.836. It converted a *directionally correct, under-scaled*
boundary into an *uncorrelated, over-scaled* one.

This is the same object NOTES-W13A saw from the other side. Its §4 measured
that restoring the parent's true layer-0 write makes the terminal state 7.6x
**worse**, and its §6 concluded "the child is an internally self-consistent
function that has been solved against its own wrong intermediate states." The
reclosure is the step that performs that solve. It fits every writer, layer 0
included, against targets measured in the child's own already-drifted downstream
regime — so a writer that agreed with the parent gets replaced by one that
agrees with the child's own error.

**What I could not attribute.** Stage A's 26x is not explained by anything I was
able to vary: not width, not ridge, not the prior mode, not the prior-delta
clamp (which does not bind from any prior I tried, including the capsule's own
weights), not fp32 storage, and not evidence misalignment — the fold's own child
replay is at the projected parent input to 2.5e-08. The remaining candidates are
the structural transport pass that ran before the closed loop and set the priors
the production fit was anchored on (those weights were never checkpointed), and
whatever row subsampling `--max-fit-rows 40656 --max-selection-rows 13728`
applied to the production fit. **Both are exactly what R13 item 8's telemetry
would have shown**, and neither is reconstructable from r12's receipt, which
carries no `Lfm2PriorCorrectionReport` at all: no `update_scale`, no
`prior_norm`, no `correction_norm`, no condition number, for any of the 14
layers. Named follow-ups in §8.

---

## 7. Layer 1 — the deficit is not layer-0-specific, and neither is the fix

Same harness at layer 1 (also `short_conv`, also internal width 1024).

**Read this table with its framing in mind.** Every row uses the *exact-input*
construction — layer 1's input is the chart-projected parent's, not the
already-broken residual the child actually carried into layer 1. For the solve
arm that is the point. For the `published`/`capsule` arms it makes the number a
counterfactual ("how does the installed layer-1 operator behave on a correct
input"), *not* comparable to the first-divergence ledger's L01 rows, which were
measured on the child's real trajectory.

| | write cos | write relL2 | write norm ratio | post_operator relL2 |
|---|---|---|---|---|
| this study's solve (r12 config; `free_solver` identical) | **0.9858** | **0.1503** | 0.981 | **0.0400** |
| r12 capsule 13, pre-reclosure, on ideal input | 0.1722 | 1.0480 | 0.4755 | 0.2966 |
| r12 published child, post-reclosure, on ideal input | 0.0457 | 6.4223 | 6.3701 | 1.5289 |

Layer 1 is representable too: the production solve reaches composed relL2
0.0400, 25x below the floor. It is 2.2x harder than layer 0 on the write (0.150
vs 0.025), which is a real and expected effect — layer 1's input is a genuinely
richer residual than the embedding — but nowhere near a wall.

The published-vs-capsule contrast repeats layer 0's stage B and amplifies it:
the reclosure took layer 1's writer from norm ratio 0.48 (under-writing) to 6.37
(**13x over-writing**) and cosine from 0.17 to 0.05. The over-write signature
NOTES-W13A reported across L01–L13 (norm ratios up to 18.3) is a reclosure
signature.

---

## 8. Verdict and what r13 should do

**Representational or solver-side? Solver-side, and not by a hyper-parameter.**
r12's exact-4x layout expresses the parent's layer-0 function to held-out
relative L2 0.018 composed / 0.025 on the write, under the production compilers
with r12's own defaults on r12's own sealed evidence. The published child sits
at 0.836 / 1.099. Nothing about the layout, the chart, the width, the ridge, the
priors, the clamp or the storage dtype accounts for the gap.

**Which r13 lever moves layer 0 past the floor?** None of the three that were
nominated.

- **Item 4 (functional-width contraction cap): does not move layer 0.** 816 is
  already 41x inside the floor; 1023 buys 7.5x on a number that is not the
  problem. Do not spend the 4x budget here on layer 0's account. Keep the cap on
  the agenda for the 2560 → 321 MLP boundary, which this study did not measure.
- **Item 7 (EIV ridge): structurally cannot move layer 0**, and cannot move any
  operator write at any depth — it is wired only into the MLP W1/W3 compile and
  it is a no-op at the first folded layer by construction. Its margin-row-weight
  half belongs to the reclosure lane and should be judged there.
- **Item 2 (absolute composition-aware terminal floor): confirmed as the highest
  value change, and now with a second use.** A per-boundary floor evaluated on
  the *branch write* would have caught stage A at layer 0 (relL2 0.64 against an
  available 0.02) and would have caught stage B the instant the reclosure pushed
  the composed residual through 1.0.

**The lever that actually matters is item 6, inverted.** The single largest,
best-attributed loss is the moving-gauge body reclosure: it left layer 0's
`in_proj` byte-identical, multiplied its writer by 4.73x, and turned a cosine of
0.911 into 0.171. Item 6 proposed *improving* the reverse reclosure sweep with
recaptured states. This study says the prior question is whether the reclosure
should be allowed to overwrite a boundary that was already good.

### Recommended layer-0 configuration for the r13 arms

1. **Width 816. Do not widen for layer 0's sake.** Sealed chart `df7f6e88…`,
   source-derived priors, `ridge = 1e-6`, `max_prior_delta_ratio = 1.0`,
   fp32 storage — i.e. **r12's exact configuration, unchanged**. Expected
   held-out: write cosine 0.9995, write relL2 0.0245, post-operator relL2 0.0181.
   Anything materially worse than that in an r13 run is a pipeline defect, and
   these three numbers are the acceptance target to hold it to.
2. **Gate the boundary itself, not just the terminal.** Per-branch held-out write
   relL2 with a hard ceiling; 0.05 at layer 0 is generous against a measured
   0.018 and would have failed r12's 0.64 at the boundary that produced it.
3. **Make the body reclosure prove it improved the boundary it rewrote.** Any
   writer replacement that lowers the write cosine or raises the composed
   post-operator relL2 for its own layer should be rejected and the pre-reclosure
   writer kept. At layer 0 that single rule turns 0.836 back into 0.494 with no
   new solving, and combined with (2) into 0.018.
4. **Emit `Lfm2PriorCorrectionReport` per fit into the receipt** (R13 item 8):
   `update_scale`, `prior_norm`, `correction_norm`, `solver_path`, condition
   number. r12's receipt has none of these, which is the only reason stage A's
   26x is still unattributed after this study.
5. **Checkpoint the child after the structural transport pass, before the closed
   loop.** The two existing restart capsules made stage attribution possible at
   all; a third one at that boundary would have closed the remaining gap.

### Named follow-ups (not done here)

- **Attribute stage A.** Needs the post-structural-pass weights and the
  production fit's actual row selection — i.e. items 8 and (5) above. Cheap once
  those exist; impossible from r12's artifacts.
- **The MLP boundary.** `mlp.down` at layer 0 has sealed cosine 0.4014 / relL2
  0.9445 and is where item 4's 2560 → 321 tail argument actually applies. Same
  harness, `compile_lfm2_structural_mlp_*`, an intermediate-size ladder, and the
  EIV multiplier sweep — which *is* live at that site from layer 1 on. This is
  the obvious next wave and it is hours, not days.
- **The `full_attention` boundaries** (layers 2/4/6/8/10/12), where NOTES-W13A
  measured the two biggest amplifiers.
- **A reclosure-off arm.** The cheapest possible test of §6: fold the body,
  publish from the capsule, skip `compile_body_reclosure` entirely, and compare
  the terminal metric. If the reclosure is net-negative the way layer 0 says it
  is, that is a one-flag intelligence recovery.

## 9. Caveats

- **Exact-input framing.** Layer-0 arms are exact by measurement (child input
  = projected parent input to 2.5e-08, verified on the fold's own replay).
  **Layer-1 arms are counterfactual**: they feed layer 1 an input the child never
  had. §7's published/capsule rows are therefore not comparable to the
  first-divergence ledger's L01 rows.
- **The consumer fit uses replayed contexts, not recaptured ones.** Production
  refits `out_proj` on contexts recaptured from the installed fp32 child; this
  study computes them from the same weights in fp64. The `r12_repro_fp32_storage`
  arm shows the storage dtype is worth nothing here, but a live recapture was not
  performed.
- **The width ladder extends the sealed chart rather than rebuilding a balanced
  basis per width.** That is deliberate (it isolates width from chart identity
  and keeps the bottom rung bit-identical to r12's) but it means the ladder
  measures the best case for widening, not what a re-derived balanced basis at
  each width would give.
- **Evidence rows differ slightly from production's.** This study uses all
  50,687 FIT rows at layer 0; r12's launcher capped the fold at
  `--max-fit-rows 40656`. Both are ~50x the design width and §5 shows the solve
  is insensitive to everything else, but they are not the identical row set.
- **Numeric only.** Per
  `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`, nothing here is an
  intelligence verdict; the manual whole-response review remains the authority.
  This explains a mechanism, it does not overturn the 0/15.
- **Not committed.** No `git add`, no commit, no push.

## 10. Artifacts

Remote (new root, nothing else written):
`/home/cereal/model-surgery-lab/artifacts/neural_forge_layer0_study_v1/` — 4.2 GB,
dominated by the two evidence-cache payloads. Runtime
`/home/cereal/model-surgery-lab/deployments/neural-forge-layer0-3953ede4`,
`src/neural_forge` byte-identical to local HEAD `3953ede4` over all 383 files.

Local: `reports/layer0-study-20260805/` — every per-arm record, inventory,
progress journal, the two forensic dumps, the rendered table, the renderer and
the launcher. See its `README.md`.

New source: `experiments/run_lfm2_layer0_representability_study.py`,
`tests/test_run_lfm2_layer0_representability_study.py` (4 passed).
