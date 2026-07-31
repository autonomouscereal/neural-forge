# NOTES-1G — solver-math pack wiring plan (post-merge)

Branch `codex/lrc-blank-compression`. Everything landed in this pass is
**additive and opt-in**: with no new argument supplied, every compile lane
produces byte-identical weights and byte-identical receipts. Nothing below is
wired into the adapters, runners, or the dynamic catalog yet — that is
deliberate, because five other lanes are compiling against these modules
concurrently. This file is the exact wiring recipe for whoever merges.

Owned files touched:

- `src/neural_forge/hf_lfm2_structural_residual_attention_fold.py`
- `src/neural_forge/hf_lfm2_semantic_layer_stack_refold_adapter.py`
- `src/neural_forge/hf_lfm2_balanced_circuit_basis.py` (additive only)
- `tests/test_hf_lfm2_solver_math_pack.py` (new)

`src/neural_forge/hf_lfm2_swiglu_tangent_bank.py` was deliberately **not**
modified; see "Deliberately left out" at the end.

---

## 1. Noise-aware ridge (errors-in-variables Tikhonov)

### What exists now

`_fit_linear` (structural fold) accepts two new keyword-only arguments,
`input_error_covariance` and `input_error_scale`. With `P` the source-derived
prior, `S` the solved correction (installed operator `W = P + S.T`),
`R = Y - X P.T` and `Sigma` the observed upstream input-error covariance, the
objective is

```
min_S  ||X S - R||^2 + ridge*||S||^2 + scale * trace(W Sigma W.T)
```

with stationary point

```
(X.T X + ridge*I + scale*Sigma) S  =  X.T R  -  scale * Sigma @ P.T
```

The Tikhonov term penalizes the **complete** operator, not the correction, so
the solve shrinks toward robustness against generation-time input drift instead
of shrinking toward the prior. Supplying a covariance always selects the primal
system (the dual/kernel shortcut is unavailable because `Sigma` is not a
multiple of the identity). `_fit_affine` extends the covariance with one exactly
zero row and column for the appended constant bias column.

`lfm2_estimate_input_error_covariance(clean_rows, perturbed_rows, ...)` returns
`(Sigma, report)` — the exactly symmetric FP64 empirical second moment of the
row difference, uncentered by default (a systematic upstream bias hurts composed
generation exactly like zero-mean drift does), with a rank report.

Already threaded into the W1/W3 fits of
`compile_lfm2_structural_residual_mlp_fold` (both its prepared-basis and direct
paths) and `compile_lfm2_real_medoid_structural_residual_mlp_fold`. Deliberately
**not** applied to the W2 endpoint fit: its design is the post-SwiGLU block, a
different space from the layer input, so a layer-input covariance is not
type-correct there.

### Wiring recipe

1. In the layer loop of `hf_lfm2_semantic_mlp_refold_adapter.py` (and the CPCF
   layer compiler), you already hold two aligned row sets at each site: the
   parent endpoint transported into the child's coordinates, and the actual
   child rows the live replay produced at the same positions. Call

   ```python
   covariance, covariance_report = lfm2_estimate_input_error_covariance(
       parent_rows_in_child_coordinates,
       actual_child_rows,
       name=f"layer{layer_index:03d}.mlp_input",
   )
   ```

   Use the **previous** accepted layer's observed error, not this layer's — the
   point is to make layer `k` robust to the drift layers `< k` will inject at
   generation time. For layer 0 there is no upstream drift, so pass nothing.

2. Pass `input_error_covariance=covariance` and
   `input_error_scale=float(effective_row_count) * robustness_multiplier` into
   `compile_lfm2_structural_residual_mlp_fold` /
   `compile_lfm2_real_medoid_structural_residual_mlp_fold`.

   **Scale units matter.** `X.T X` for row-weighted designs is the *weighted*
   Gram, so `scale` must be commensurate with the sum of row weights, not with
   the raw row count. Start with `scale = sum(row_weights)` (which is the row
   count under the uniform default) and sweep the multiplier over
   `{0, 0.25, 1.0, 4.0}`. `multiplier = 0` reproduces the current fit exactly
   and is the control arm.

3. Store `covariance_report` next to `fold-fit-selection-evidence.json`. It
   carries `rank`, `rank_deficient`, `trace` and `covariance_sha256`. A
   rank-deficient estimate constrains only the observed error subspace; that is
   legitimate but must be visible in the receipt.

4. **Expected direction.** Teacher-forced fit error on the fit rows gets
   *worse*; composed/endpoint error under drift should get better. If the
   endpoint error does not improve, the multiplier is wrong or the covariance
   was estimated at the wrong site — do not conclude the term is useless from
   the fit-error regression alone. `tests/test_hf_lfm2_solver_math_pack.py
   ::test_noise_aware_ridge_beats_plain_ridge_under_real_input_drift` asserts
   both halves of that trade explicitly.

---

## 2. Margin-weighted rows

### What exists now

- `lfm2_vocabulary_margin_vulnerability(margins, logit_values=..., ...)` in
  `hf_lfm2_balanced_circuit_basis.py` — the exact unnormalized inverse-margin
  vulnerability the observability Gram already applies to its margin
  directions, lifted to a public, side-effect-free function. Purely additive;
  the internal `_build_vocabulary_margin_direction_set` path is untouched.

- `lfm2_vocabulary_margin_row_weights(margins, ...)` in the structural fold —
  sums a position's competitor vulnerabilities (its total mass in the
  balanced-basis Gram), clamps the ratio-to-mean into
  `[weight_floor, weight_ceiling]`, and renormalizes to **exactly unit mean**.
  Unit mean is what makes a constant profile reproduce the uniform default
  bit-for-bit, and the clamp is what stops one near-zero margin from owning the
  whole fit (`max/min <= ceiling/floor`).

- `compile_lfm2_real_medoid_structural_residual_mlp_fold` and
  `compile_lfm2_normalized_input_real_medoid_structural_mlp_fold` accept
  `case_row_sample_weights` (flat over concatenated rows, or one block per
  case) plus `row_weight_floor` / `row_weight_ceiling`. The profile multiplies
  the repeated case weight, so the total row mass is preserved and only its
  distribution changes. It reaches the multi-anchor **plan** as well as all
  three fits, because both read `sqrt_row_weights`.

- `hf_lfm2_semantic_layer_stack_refold_adapter.py`:
  `_build_attention_source_local_output_fit_cases` and
  `_build_short_conv_source_local_writer_fit_cases` accept
  `fit_row_weight_profiles: Mapping[int, Sequence[float]] | None`, and
  `lfm2_semantic_margin_fit_row_weights(...)` builds those profiles from cached
  per-donor-position margin tables, selecting rows by the alignment's donor
  positions so the profile is aligned to exactly the rows the builders emit.

### Wiring recipe

1. The margin tables come from the same cache the balanced basis reads:
   `_CachedVocabularyMarginPhase.margins` is `[positions, top_k]`
   (top-1 minus each competitor) and `.logit_values` is the matching logit
   table. Surface them per FIT batch as `Mapping[int, Tensor]` keyed by
   `source_batch_index`, rows ordered by donor position.

2. In the semantic layer stack, once per stack (not per layer):

   ```python
   profiles, profile_receipt = lfm2_semantic_margin_fit_row_weights(
       alignments=alignments,
       batch_margins=batch_margins,
       fit_batch_indices=fit_batch_indices,
       margin_floor_fraction=0.05,   # matches the balanced-basis default
       weight_floor=0.25,
       weight_ceiling=4.0,
   )
   ```

   then thread `fit_row_weight_profiles=profiles` into both source-local
   fit-case builders and attach `profile_receipt` to the layer receipt.

3. For the MLP lanes, convert the same margins with
   `lfm2_vocabulary_margin_row_weights` and pass the result as
   `case_row_sample_weights` (one block per `Lfm2CachedMlpFoldCase`, aligned to
   that case's `target_input` rows).

4. **Start with floor/ceiling `0.25 / 4.0`.** That is a 16x dynamic range,
   which is already aggressive for a closed-form fit. If a sweep is wanted,
   move the ceiling first (`2.0`, `4.0`, `8.0`) and keep `floor * ceiling = 1`
   so the clamp stays symmetric in log space.

5. **Do not** apply margin weights to the terminal/lexical writer fits. Those
   already sit at the decision surface; re-weighting them by their own decision
   margin double-counts.

---

## 3. Gate operating point

### What exists now

- `lfm2_gate_preactivation_statistics(rows, label=..., saturation_thresholds=...)`
  → `Lfm2GatePreActivationStatistics` with per-channel mean, population
  variance, standard deviation, and per-threshold fractions below `-t` and
  above `+t`. `as_dict()` emits aggregate scalars plus digests of the
  per-channel tensors (the full per-channel vectors stay on the object).

- `lfm2_gate_operating_point_report(...)` — read-only parent-versus-child
  comparison with absolute and relative per-channel drift.

- `lfm2_gate_operating_point_correction(...)` — the closed-form two-moment
  rescale. Per channel `j`:

  ```
  a_j = std(T_j) / std(Z_j)
  b_j = mean(T_j) - a_j * mean(Z_j)
  W'_j = a_j * W_j + b_j * c
  ```

  where `Z = X @ W.T` is the fitted child pre-activation, `T` the parent
  targets, and `c` the ridge least-squares solution of `X @ c ~ 1`. LFM2's `w1`
  has no bias, so the mean shift is carried by that one shared constant
  direction. **When `1` lies in the row space of `X` both moments match
  exactly**, which the receipt reports as
  `constant_direction.exact_constant_response == True` and
  `constant_direction.relative_residual == 0`. A real RMSNorm output has no
  constant coordinate, so on production rows expect a nonzero residual and a
  correspondingly approximate mean match — read that number before trusting the
  correction. `a_j` is clamped to `[1/maximum_channel_scale,
  maximum_channel_scale]` and forced to 1 on numerically constant channels.

  `match_channel_mean=False` (compile flag:
  `gate_operating_point_match_channel_mean=False`) drops the shift entirely and
  applies the pure per-channel scale `W'_j = a_j * W_j`, which a bias-free
  projection represents exactly. **This is the mode to reach for first**, for
  the reason measured below.

### Measured on a random 8-row / width-4 / bf16 target fixture

The diagnostics reproduce the documented failure mode immediately. Parent gate
pre-activations: 25% of rows below `-4`, 31% above `+4`. The fitted child before
correction: 6% below `-4`, 12% above `+4` — the child's gate distribution is
compressed into the linear region and a quarter of the parent's "off" neurons
are no longer off. After the affine correction: 25% below `-4`, 19% above `+4`.

On that same fixture `constant_direction.relative_residual` was `0.966` (the
all-ones vector is essentially orthogonal to the design row space), so the
affine mode left a max per-channel mean error of `1.37` while the
standard-deviation error was `0.16`. Scale-only mode matches the standard
deviation to `< 1e-10` and makes no mean claim at all. Prefer it unless that
residual is genuinely near zero.

- Compile lanes: `compile_lfm2_structural_residual_mlp_fold` (both paths) and
  `compile_lfm2_real_medoid_structural_residual_mlp_fold` accept
  `gate_operating_point_diagnostics` and `gate_operating_point_correction`;
  `compile_lfm2_normalized_input_real_medoid_structural_mlp_fold` accepts
  diagnostics only, because its W1/W3 are planted directly from the decoder and
  a rescale would falsify its `producer_prior_sha256` claim.
  `Lfm2CompiledStructuralMlpFold` / `Lfm2CompiledRealMedoidStructuralMlpFold`
  gained `gate_operating_point` and `uncorrected_gate_weight` (both FP64,
  pre-target-dtype), emitted in `as_dict()` only when present.

### Wiring recipe

1. **Turn diagnostics on unconditionally first.** They cost one extra GEMM per
   layer, never touch the weights (asserted by
   `test_gate_diagnostics_do_not_change_the_compiled_state_but_correction_does`)
   and give the missing observable: how far the fitted child's gate
   pre-activation distribution has moved from the parent's regime. Log
   `gate_operating_point.drift.*` and the per-threshold saturation fractions
   into the per-layer fold receipt.

2. Look for the failure signature before enabling any correction: layers where
   `child.saturation[t].mean_fraction_below_negative_threshold` differs
   materially from the parent's. Those are neurons the child has switched off
   (or on) wholesale — a distribution shift no residual RMSE on the fit rows can
   surface.

3. **The correction is behavior-changing and stays off by default.** Gate it
   like this and never the other way round:

   - compile twice per candidate layer (`correction=False`, `correction=True`),
   - keep both `uncorrected_gate_weight` and the installed weight,
   - run the **autoregressive generation** eval, not the teacher-forced fit
     error — the correction is a distribution repair and can legitimately raise
     fit-row residual while lowering composed endpoint error,
   - adopt per layer only if the composed endpoint error improves and the
     behavioral eval does not regress; otherwise install the uncorrected weight.

   The receipt already states
   `requires_behavioral_validation_before_production_use: true` and
   `applied_automatically: false`. Do not remove either.

4. Suggested thresholds for the first sweep: `(2.0, 4.0, 8.0)` (the default).
   `+/-4` is the practically meaningful SiLU frontier; `2.0` and `8.0` bracket
   it.

5. Sweep order: diagnostics only -> scale-only correction -> affine correction,
   and reach the third rung only if `constant_direction.relative_residual` is
   small on the real rows. Skipping to affine on production rows spends the
   experiment budget on a mean match the geometry cannot deliver.

---

## Verification status at hand-off

Command:

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" -m pytest tests/<file> -q
```

- `tests/test_hf_lfm2_solver_math_pack.py` — 25 passed (new).
- `tests/test_hf_lfm2_structural_residual_attention_fold.py` — 39 passed,
  1 skipped (unchanged from baseline).
- `tests/test_hf_lfm2_swiglu_tangent_bank.py` — 13 passed, 1 skipped.
- `tests/test_hf_lfm2_semantic_layer_stack_refold_adapter.py` and the three
  sibling semantic suites — 131 passed.
- `tests/test_dynamic_balanced_circuit_basis.py` — 13 passed.
- `tests/test_refine_lfm2_dynamic_structural_residual_attention.py` — 9 passed.

Ruff: `python -m ruff check` clean on all four source files and the new test
file. The three pre-existing `E501` lines in the structural fold and the fifteen
in the semantic adapter are unchanged in count and identity.

Beyond the suites, the default-equivalence claim was proven directly rather than
asserted: the pre-change module was loaded from `git show HEAD:` side by side
with the current one, and both were driven through
`compile_lfm2_structural_residual_mlp_fold` (direct and prepared-basis reuse),
`compile_lfm2_real_medoid_structural_residual_mlp_fold` and
`compile_lfm2_normalized_input_real_medoid_structural_mlp_fold` for FP64, FP32
and BF16 targets. All ten comparisons agreed on `compiled_state_sha256`, on the
full `as_dict()` receipt, and on `torch.equal` over every state tensor.

Blast-radius check: every test file in `tests/` that imports any of the four
owned modules was run in one serial process -- 54 files, **1143 passed, 8
skipped**.

A whole-repo sweep (`pytest tests/ -q -p no:randomly`) reports 70 failures and
16 errors, none of them mine. Those were triaged rather than waved away: the
tree was copied to a scratch directory, the three source files this pass
touched were replaced with their `git show HEAD:` versions, and the failing
files were re-run there. Every failure reproduces identically with this pass's
code reverted, so all of them belong to the other lanes' in-flight work on
`hf_lfm2_circuit_trace_cache.py`, `dynamic_behavior_gate.py`,
`dynamic_circuit_ir.py` and the CPCF v5 frontier runners. Concretely
reproduced-when-reverted:
`test_hf_lfm2_global_topology_phase_cache_exact4x.py::test_no_carrier_exact4x_control_h816_b3_cold_hit_frontier_equality`,
`test_hf_lfm2_long_horizon_terminal_refold.py::test_replays_each_complete_trajectory_once_and_installs_only_accepted_w2`,
three `test_run_lfm2_dynamic_catalog.py` cases,
`test_run_dynamic_qwen3_codebook_fold.py::test_tiny_production_runner_publishes_and_freshly_gates_without_training`
and
`test_run_lfm2_cpcf_v5_selective_blank_fold.py::test_existing_attention_profile_hashes_remain_byte_stable`.
The remaining `test_run_lfm2_cpcf_v5_frontier.py` /
`test_validate_lfm2_cpcf_v5_quarantine.py` / `test_surgery_meta_compiler.py`
entries pass when re-run serially and were CPU-contention artifacts of the
sweep, which ran alongside three other pytest processes.

---

## Deliberately left out

- **`hf_lfm2_swiglu_tangent_bank.py` is unmodified.** Adding a second gate
  statistics surface there would duplicate
  `lfm2_gate_preactivation_statistics` for no new information, and any addition
  to `advisory_diagnostics` perturbs `diagnostic_sha256`. If anchor-regime
  evidence is wanted later, call the structural fold's statistics function on
  `cached_gate_rows[anchor_indices]` from the caller rather than changing the
  bank.
- **No runner/adapter/catalog wiring.** `hf_lfm2_semantic_mlp_refold_adapter.py`,
  `hf_lfm2_cpcf_layer_compiler.py`, `hf_lfm2_dynamic_closed_loop_adapter.py`,
  `experiments/run_lfm2_dynamic_catalog.py` and the dynamic fold registry are
  out of scope for this pass and were not opened for editing.
- **W2/endpoint fits get no input-error covariance.** Wrong space; see §1.
- **Attention Q/K/V/O fits get no covariance yet.** `_fit_linear` and
  `_fit_affine` both accept it, and the attention lanes call them, but the
  arguments are not threaded through
  `compile_lfm2_structural_residual_attention_fold`'s public signature. That
  is a mechanical follow-up once §1's scale calibration is settled on the MLP
  lanes, which are the cheaper experiment.
- **No behavioral validation was run.** Every claim here is numerical and
  test-backed; none of it has been shown to improve a real child's generation.
