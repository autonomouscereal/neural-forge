# NOTES-W13R — the three Stage-A remedies, implemented

Wave 13R. Implements R1/R2/R3 of
`reports/stage1-attribution-20260808/STAGE_A_ATTRIBUTION.md` §4.1 — the last
never-tried fold-only levers before the stage-1 ceiling question resolves.

The attributed cause, restated so this file stands alone: the body boundary
solve's ridge is *absolute* `1e-6` while the normal system it is added to has
σ_max 1.4e5–7.1e5, so its relative weight is 1.4e-12 to 7.3e-12 — inert. The
resulting near-singular solves asked for corrections of **1.4x–140x the prior's
norm at all 42 body MLP fits** of the r13 refold, and a hard-coded
`max_prior_delta_ratio = 1.0` truncated every one of them to a unit-relative
step. `update_scale` was never "how much the solver corrected"; it was how much
of the solve survived a budget set by the structural-transport prior's own norm.

Local CPU only. No training, no gradients, no model calls, no server writes.
Nothing committed.

---

## 1. Sites

Runtime for everything below is the repo venv (`.venv/Scripts/python.exe`,
CPython 3.11.9, torch 2.13.0+cpu) — `C:/Python314/python.exe` has no torch.

### R1 — condition-capped relative ridge

| what | where |
|---|---|
| `_shifted_system_conditioning` (new helper) | `src/neural_forge/hf_lfm2_structural_residual_attention_fold.py:1118` |
| `_fit_linear(..., relative_ridge_lambda=None)` | `hf_lfm2_structural_residual_attention_fold.py:1149` |
| the one Gram + one `svdvals`, and `ridge = λ·σ_max` | `hf_lfm2_structural_residual_attention_fold.py:1238-1261` |
| branch reuse of that Gram (covariance / dual / primal) | `hf_lfm2_structural_residual_attention_fold.py:1262-1310` |
| conditioning read off the reused spectrum | `hf_lfm2_structural_residual_attention_fold.py:1311-1327` |
| MLP compile passthrough (prepared basis) | `hf_lfm2_structural_residual_attention_fold.py:4899`, fits at `:5047`, `:5059`, `:5093` |
| MLP compile passthrough (fresh basis) | `hf_lfm2_structural_residual_attention_fold.py:6468`, fits at `:6776`, `:6788`, `:6818` |
| adapter config field | `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py:1320` |
| adapter → compiler | `hf_lfm2_dynamic_closed_loop_adapter.py:14099` |
| catalog field / validator / identity / CLI / passthrough | `experiments/run_lfm2_dynamic_catalog.py:1292`, `:2828`, `:3413`, `:14372`, `:10340` |

σ_max is **not** recomputed. The relative arm builds the Gram once, takes one
`svdvals` of it, sets the ridge from its maximum, reuses the same matrix in the
solve branch, and reads the solved system's conditioning off the same spectrum
shifted by the ridge. That is exact for a symmetric PSD Gram (its singular
values are its eigenvalues, and `ridge·I` shifts all of them equally); measured
worst relative deviation from a direct second `svdvals` is **8.3e-16** across
four squeeze levels × two λ (test
`test_the_reused_spectrum_measures_what_a_second_svd_would`). The shortcut is
skipped, and a real measurement taken, whenever the solved system is not
`Gram + ridge·I` — the errors-in-variables branch (Σ is not a multiple of I) and
a zero effective ridge (which solves the design matrix directly).

Per-solve seal (`Lfm2PriorCorrectionReport`, fold file `:874-889`, emitted at
`:954` **only** in relative mode): `mode`, `relative_ridge_lambda`,
`unregularized_system_singular_value_maximum`, `effective_ridge`,
`absolute_ridge_configured`, `effective_condition_number_cap`.

### R2 — body `max_prior_delta_ratio`, hard-coupled to R1

| what | where |
|---|---|
| adapter config field | `hf_lfm2_dynamic_closed_loop_adapter.py:1321` |
| adapter-side refusal | `hf_lfm2_dynamic_closed_loop_adapter.py:1486-1497` |
| `mlp_max_prior_delta_ratio` (unset ⇒ the shared 1.0 lane) | `hf_lfm2_dynamic_closed_loop_adapter.py:1897` |
| catalog refusal citing the closed path | `experiments/run_lfm2_dynamic_catalog.py:2856-2871` |
| CLI flag | `experiments/run_lfm2_dynamic_catalog.py:14387` |
| requested/allowed/engaged computed | `hf_lfm2_structural_residual_attention_fold.py:1344-1350` |
| `prior_delta_clamp` block | `hf_lfm2_structural_residual_attention_fold.py:943` |

**Scope, deliberately narrower than the flag name suggests.** Both R1 and R2
reach the three body MLP fits (`mlp_gate`, `mlp_up`, `mlp_down`) and nothing
else. The operator producer/consumer solves keep the absolute ridge and the 1.0
clamp, at every depth, and a test asserts it. If R2 had been wired to the shared
`max_prior_delta_ratio` it would also have relaxed the clamp on the five
operator `_fit_linear` sites — which R1 does not regularize — and that is
precisely the unbounded-step-on-an-unregularized-system configuration the
coupling exists to forbid. A body flag that quietly reopened the closed path on
the other branch would have been the bug, not the feature.

The refusal is **only** about *relaxing*: `body_max_prior_delta_ratio ≤ 1.0`
needs no companion (tightening a bound is not the closed path), and both
validators say so. Refusal text names A12/W12B and their numbers (pinv step 40x
for 1% fit gain, held out 6x worse; unregularized reclosure +515%).

### R3 — serialize the telemetry that already existed

| part | what | where |
|---|---|---|
| (a) | `_operator_prior_correction_metadata` helper | `hf_lfm2_dynamic_closed_loop_adapter.py:1133` |
| (a) | producer stage metadata | `hf_lfm2_dynamic_closed_loop_adapter.py:13511` |
| (a) | consumer stage metadata | `hf_lfm2_dynamic_closed_loop_adapter.py:13877` |
| (b) | `mlp_residual_endpoint_relative_rmse` | `hf_lfm2_dynamic_closed_loop_adapter.py:14130` |
| (b) | `operator_residual_endpoint_relative_rmse` | `hf_lfm2_dynamic_closed_loop_adapter.py:13872` |
| (c) | `mlp_down` gets `solver_diagnostics` | `hf_lfm2_structural_residual_attention_fold.py:5098` and `:6823` |
| (c) | materialization check now covers all three roles | `hf_lfm2_dynamic_closed_loop_adapter.py:14147` |

(a) handles both artifact shapes: `prior_correction_reports` (tuple — attention
producer, coupled V/O) and `prior_correction_report` (singular — short-conv
producer, both consumers).

(c) was fixed at **two** call sites, not one: the prepared-basis compile the
adapter actually uses (`:5098`, the `:4943` of the attribution's line numbering)
and the identical fresh-basis compile (`:6823`). Both produce role `mlp_down`.

---

## 2. The digest-stability decision, stated

**Unconditional, and default-run receipt digests move.** Specifically:

* `prior_delta_clamp` is emitted on **every** prior-anchored solve in the
  program, not behind a telemetry flag. Every `Lfm2PriorCorrectionReport` for a
  fit with a prior and a ratio grows one nested object.
* `operator_prior_correction_reports_json` is added to every producer and
  consumer stage's metadata, unconditionally.
* `mlp_residual_endpoint_relative_rmse` and
  `operator_residual_endpoint_relative_rmse` are added to downstream and
  consumer metadata, unconditionally.
* `mlp_down` now carries `solver_conditioning` whenever the item-8 flag is on,
  which changes `mlp_prior_correction_reports_json` for diagnostics runs.

Consequences, honestly: stage metadata feeds the sealed
`accepted_layer_completion_receipt` preimages, so **any run from this source
seals different stage digests than a run from the parent commit**, with or
without the new flags. What does *not* move: one compiled byte, any
`compiled_state_sha256`, any evidence/replay/split digest, or the configuration
digest (`identity()`) of a run that sets neither new flag — the identity entry
is conditional, following the install-verification and reclosure-ceiling
precedent in the same method.

Why unconditional was the right call here rather than the file's usual
"emitted only on request":

1. The gap this closes was *created* by an opt-in nobody set. The operator
   branch has been unattributable since W13B for exactly one reason — the
   numbers were computed and dropped — and re-gating them behind a flag
   reproduces the failure mode on the next run that forgets it.
2. Nothing downstream compares these digests to a frozen constant. There is no
   sealed protocol digest, test literal, or acceptance gate over stage metadata
   (checked: no 64-hex literal in `src/`, `experiments/` or `tests/` names one),
   and the receipt fingerprint rotates with any source edit regardless.
3. `requested/allowed` is *the* diagnostic. The attribution had to recover it by
   inverting `update_scale` after the fact, and could only do so because the
   clamp happened to bind at exactly 1.0x. That reconstruction is not available
   at any other ratio.

The two genuinely conditional blocks are the ones that describe a configuration
rather than a measurement: `relative_ridge` (absent in absolute mode) and the
catalog's `dense_closed_loop_body_solve` identity entry (absent when neither
flag is set).

---

## 3. The toy, and what it measures

`tests/test_hf_lfm2_r13_retention_lane.py` (`_ill_conditioned_bundle`): one
240-row boundary, target hidden 4 / intermediate 5, source 8 / 10. The design's
last direction is squeezed by 1e-5 (cond(XᵀX) ≈ 2.5e8), the parent gate/up
targets carry a 5e-2 unrepresentable component — the toy analogue of "the
child's post-operator activations are numerically rank-deficient in the 816-wide
chart at depth" — and the structural-transport prior is scaled to 0.05 so that
even a well-regularized correction exceeds it. Held-out rows are 240 more rows
from the same generator, never seen by any fit.

Fit = `residual_endpoint_relative_rmse` (the branch-local endpoint error R3(b)
now seals). Held-out = the same endpoint recomputed on the unseen rows.

| arm | mlp_gate req / clamped | mlp_up req | mlp_down req | cond(gate) | fit relL2 | held-out relL2 |
|---|---|---|---|---|---|---|
| absolute 1e-6, clamp 1.0 (**today**) | 112.1x / yes | 34.8x | 9415x | 2.495e+08 | 0.999908 | 0.999887 |
| R1 λ=1e-4, clamp 1.0 | 17.3x / yes | 15.8x | 1.62e4x | 1.0e+04 | 0.999989 | 0.999980 |
| R1 λ=1e-4 + R2 ratio 8 | 17.3x / yes | 15.8x | 70.9x | 1.0e+04 | 0.911353 | 0.906443 |
| **R1 λ=1e-4 + R2 ratio 64** | 17.3x / **no** | 15.8x | 15.7x | 1.0e+04 | **0.392793** | **0.524867** |
| R2 ratio 64 alone (refused by the validator) | 112.1x / **yes** | 34.8x | 29.7x | 2.495e+08 | 0.396161 | 0.540818 |

Five things this establishes, and one it does not:

1. **The shape reproduces.** Requested 112x/35x/9415x against an allowed 1.0x,
   all three roles clamped, and a boundary sitting at 0.9999 — i.e. no better
   than emitting zero, which is where arm A's fourteen boundaries sat (post-layer
   relL2 0.78–1.39, six of them *above* 1.0).
2. **R1 alone is not the remedy.** It does exactly what it claims — the
   unregularized request falls 112x → 17.3x and the solved condition number is
   capped at 1/λ — and the boundary does not move at all, because the clamp is
   still the operative limiter. Anyone reading a λ-sweep alone would conclude
   the ridge does nothing.
3. **R2 alone is not the remedy either, and is worse than the pair.** It leaves
   cond at 2.5e8, is *still* clamped on the largest request (112x exceeds even a
   64x budget), and generalizes worse than the coupled arm (0.5408 vs 0.5249)
   despite a comparable fit. That is the A12/W12B signature in miniature.
4. **Together they recover the boundary**: 0.9999 → 0.3928 fit, 0.9999 → 0.5249
   held out, with no clamp engaged anywhere.
5. **The clamp identity holds exactly.** `|‖correction‖ − ratio·‖prior‖| /
   (ratio·‖prior‖) = 0.0` on every bound solve, and
   `requested × update_scale = 1.000000000000` — the same identity
   `verify_clamp_and_ridge.py` checked at 1.46e-16 across the run's 42.

What it does not establish: that a 4x LFM2 refold's *composed* retention moves.
This is a 240-row synthetic boundary, not a fold. It proves the mechanism is
real, that the remedy engages it, and that the coupling is not superstition.

One property found while writing it, worth carrying: **the three MLP roles are
not independent**. `mlp_down` is solved against the post-SwiGLU block produced by
the gate/up weights, so its design matrix is a different matrix in every arm —
its request can *rise* under R1 (9415x → 1.62e4x) while the boundary improves.
Only `mlp_gate`/`mlp_up` share a design across arms and are directly comparable.
Any future λ-sweep read-out has to say which role it is reading.

---

## 4. Recommended arm flags

```
--dense-closed-loop-relative-ridge-lambda 1e-5 \
--dense-closed-loop-body-max-prior-delta-ratio 8.0 \
--dense-closed-loop-solver-conditioning-diagnostics \
--dense-closed-loop-depth-telemetry
```

on top of the existing `nf-launch-r12-round-refold.sh` arm-A line.

**λ = 1e-5.** The attribution's band is [1e-6, 1e-4]; 1e-5 is its geometric
centre and caps the solved condition number at ~1e5. Against the run's measured
σ_max (1.4e5–7.1e5) that is an effective ridge of **1.4–7.1**, i.e. seven to
thirteen orders above today's inert 1e-6, and — the number that matters — it
sits *above* σ_min at every one of the 14 layers (σ_min ranges 1.3e-6 to 0.23),
so every direction the solve inverts is a direction the ridge, not the noise,
controls. λ=1e-4 caps at 1e4 and is the safer arm if the first read-out shows
the requests still running past the budget; λ=1e-6 caps at 1e6, which is above
the *conditioning of the best layer in the run* (L0 at 1.26e6) and would
therefore be inert at exactly the layers that fit best.

**ratio = 8.0.** Two independent reasons for this rung rather than the toy's 64:

* Measured. With λ=1e-5 the run's requests should land in the low tens (the toy
  fell 112x → 17.3x at λ=1e-4; a smaller λ regularizes less and leaves a larger
  request). 8.0 buys back roughly an order of magnitude of the discarded
  correction while still refusing a 140x step outright, so the clamp stays a
  backstop instead of being removed.
* Policy. The audit's do-NOT list forbids unbounded steps. A ratio that never
  binds is an unbounded step by another name; one that binds at 8x is a bound.

The toy's own ladder is the evidence for treating this as a rung and not a
constant: ratio 8 recovered 0.9999 → 0.9114 and ratio 64 recovered it to 0.3928
*on that toy*, whose prior is deliberately tiny. The real priors are not tiny —
‖prior‖ runs 10.3–16.7 across the run's 42 solves — so the right production rung
is the smallest one at which the clamp stops binding on most roles. **Read
`requested_prior_delta_ratio` from the first two layers of the arm and set the
rung from it**; that number now exists in the receipt, which is the entire point
of R3.

Do **not** run `--dense-closed-loop-body-max-prior-delta-ratio` above 1.0 without
the λ flag. Both validators refuse it, and the toy says why: same fit, worse
held-out, condition number untouched at 2.5e8.

Also worth one control arm: `λ=1e-5` with the ratio left unset. It is the
cleanest possible test of point 2 above — if the boundary does not move, the
clamp is confirmed as the limiter on the real fold, not just on the toy.

---

## 5. Tests, and what was run

New (`tests/test_hf_lfm2_r13_retention_lane.py`, section "W13R"):

* `test_relative_ridge_default_off_solves_the_identical_system` — the
  byte-identity pin: unset ⇒ identical `compiled_state_sha256` and identical
  gate/up/down tensors, and no `relative_ridge` key anywhere in the report.
* `test_the_relative_ridge_seals_lambda_sigma_max_and_the_mode` — mode, λ,
  σ_max, `effective_ridge == λ·σ_max` to 1e-12, and the cap honoured.
* `test_the_reused_spectrum_measures_what_a_second_svd_would` — 8 cases, the
  1e-15 fidelity result above.
* `test_every_solve_seals_its_requested_and_allowed_delta_ratio` — including
  `update_scale == 1/requested` on bound solves.
* `test_mlp_down_now_carries_conditioning_like_the_other_two_roles` — R3(c),
  with the compiled state pinned identical.
* `test_the_clamp_not_the_evidence_limits_the_unregularized_boundary`,
  `test_the_relative_ridge_shrinks_the_request_it_cannot_release`,
  `test_the_two_remedies_together_recover_the_boundary`,
  `test_relaxing_the_clamp_alone_leaves_the_singular_system_uncapped` — §3's
  four arms.
* `test_the_operator_branch_seals_its_prior_correction_reports` — R3(a)/(b)
  through a real toy fold: reports present with real norms and a real clamp
  block on producer *and* consumer, plus both endpoint errors.
* `test_the_body_solve_remedy_reaches_a_toy_layer_and_records_itself` — R1+R2
  end to end: the body state and artifact digest move, the seal names λ and the
  ratio, the operator branch never sees either.
* `test_the_adapter_refuses_a_relaxed_clamp_without_a_relative_ridge`.

New (`tests/test_run_lfm2_dynamic_catalog_r13_retention.py`): defaults off and
absent from `identity()`, both flags require `--dense-run-closed-loop`, the
`>1.0`-without-λ refusal and its message, the `≤1.0` carve-out, negative-value
refusal, and the conditional identity + depth-telemetry entries.

```
.venv/Scripts/python.exe -m pytest tests/test_hf_lfm2_r13_retention_lane.py \
    tests/test_run_lfm2_dynamic_catalog_r13_retention.py -q      # 70 passed
.venv/Scripts/python.exe -m pytest tests/test_hf_lfm2_dynamic_closed_loop_adapter.py -q
                                                                 # 74 passed
.venv/Scripts/python.exe -m pytest tests/test_hf_lfm2_moving_gauge_input_producer_reclosure.py \
    tests/test_hf_lfm2_moving_gauge_residual_reclosure.py \
    tests/test_hf_lfm2_post_causal_o_global_reclosure.py \
    tests/test_hf_lfm2_postfold_reclosure_seed_archive.py \
    tests/test_hf_lfm2_postfold_terminal_w2_reclosure.py \
    tests/test_run_lfm2_exact4x_causal_reclosure.py \
    tests/test_prepare_lfm2_a12_carrier_reclosure.py \
    tests/test_run_lfm2_carrier_reclosure.py -q                  # 135 passed
.venv/Scripts/python.exe -m pytest tests/test_hf_lfm2_structural_residual_attention_fold.py -q
                                                                 # 40 passed, 1 skipped
.venv/Scripts/python.exe -m pytest tests/test_run_lfm2_dynamic_catalog.py -q
                                                                 # 119 passed, 1 skipped
.venv/Scripts/python.exe -m pytest tests/test_hf_lfm2_closed_loop_fold.py \
    tests/test_dynamic_closed_loop_fold.py \
    tests/test_hf_lfm2_closed_loop_restart_capsule.py \
    tests/test_hf_lfm2_child_closed_loop_refinement.py -q        # 78 passed
.venv/Scripts/python.exe -m pytest tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py \
    tests/test_hf_lfm2_semantic_mlp_refold_adapter.py -q         # 69 passed, 1 skipped
.venv/Scripts/python.exe -m pytest tests/test_dynamic_lfm2_dense_budget_fold_stack.py \
    tests/test_run_lfm2_dynamic_catalog_portable_runtime.py \
    tests/test_run_lfm2_weight_complete_closed_loop.py -q        # 49 passed
.venv/Scripts/python.exe -m ruff check <the five edited files>   # All checks passed
```

Known-red, unchanged: `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` —
3 failed, 6 passed (d296b602). Verified pre-existing by stashing every W13R edit
and re-running `test_carrier_dense_core_alternates_fold_round_and_refold`: same
failure, `Selected parent component payload changed after authenticated load`,
`status == 'failed_closed'`. `experiments/run_lfm2_dynamic_catalog.py` carries
3 pre-existing ruff `F401`s (unused imports at `:161`, `:163`, `:174`); the count
is unchanged by this work and they were left alone.

---

## 6. Deliberately left out

* **The operator branch does not get R1 or R2.** §1. It gets R3's telemetry, so
  the *next* run can attribute it; changing its solve without the measurement is
  the mistake this whole wave is a response to.
* **`_fit_affine` has no `relative_ridge_lambda`.** It is the short-conv and
  attention path; wiring a parameter that nothing may pass would be dead code.
* **`compile_lfm2_real_medoid_structural_residual_mlp_fold` keeps its
  instrumentation gap.** Its `mlp_multi_anchor_down` fit
  (`hf_lfm2_structural_residual_attention_fold.py:5585`) omits
  `solver_diagnostics` exactly the way `mlp_down` did. Same defect, different
  role and a different lane; R3(c) named `mlp_down` and that is what was fixed.
  Recorded here so it is a decision and not an oversight.
* **The frozen collector, `dense_budget`, `closed_loop_fold`, the v5 files, the
  on-policy round, and the reclosure/ceiling internals are untouched.**
* **No `git add`, no commit, no push.**
