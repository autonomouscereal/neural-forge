# NOTES-W13M — the composed R13 verdict, the stale smoke assertion, and a first look at the child

Wave 13M, branch `codex/lrc-blank-compression`. The first complete composed r13
refold sealed at **2026-08-07T10:45:47Z** after 16 h 34 m. It **published its
child** and then **failed closed 55 ms later** on a publication-phase assertion.

```
run      …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
source   b65c81f5d086830247b0a418a78d54fb12975ca6   (W13L's rollback; d296b602 excluded)
status   failed_closed
receipt  c9a8f7468934f55a97f37cd14e7a7fb91721657dfa2f54f4218d9a962d98e4c9
message  ProductionLfm2CatalogError: Carrier route smoke did not execute on the
         in-memory child, stage reload, and final reload.
child    published-child/{child, certificate.json, provenance} — PRESENT
```

Three things this wave establishes:

1. **The floor held and the composition improved.** Terminal held-out relL2
   went from r12's **4.4728** to **0.9699 at layer 0** and **0.9711 after the
   body reclosure** — under the 1.0 emit-zero baseline at every one of 14
   layers, which is the first time this arm has ever been under it (§2).
2. **The reclosure gate refused nothing, and structurally cannot refuse the r12
   failure mode.** The sealed per-boundary ledger was **never persisted** (§1.1),
   but the outcome is directly measurable from this run's own tensors: the body
   reclosure rewrote **82** weights, **81 of them upward** — median norm ratio
   **33.1x**, 64 of 82 above 10x, from **2.45x at layer 0 to 974x at layer 7**
   (§1.2). r12's diagnosed 4.73x layer-0 blow-up is now the *mildest* case in
   the run.
3. **The failure is the exact stale-count class `a8b65e15` fixed for tests, in
   production code** (§4). Fixed here by naming the subjects and by having the
   catalog smoke the published final destination itself.

Predecessor: `NOTES-W13L` (attempt #5a/#5b). The `d296b602` regression it
attributed is **reproduced locally in a 4-minute toy test** this wave (§6.3).

---

## 1. Reclosure gate — `writer_reclosure_acceptance`

Configured, active, and running the guarded mode:

```
--dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement
--dense-moving-gauge-boundary-regression-tolerance 1e-6
--dense-body-reclosure-mode moving_gauge_v3
```

The reclosure executed once, after all fourteen ascending layers and before the
final normalization, and accepted:

```
body_reclosure.mode                          compiled_body_reclosure
body_reclosure.compiled.metadata.body_reclosure_mode   moving_gauge_v3
identity_preservation                        false
boundary_count                               15
transition_count                             14
input_producer_count                         54
physical_writer_count                        28
fresh_primal_operator_consumer_reclosure_count  14
fresh_primal_w2_reclosure_count                 14
install_fingerprint_group_count              57
changed_physical_parameter_count             82
install_verification_mode                    fingerprint
parent_forward_calls / training_steps        0 / 0
validation.accepted                          true
moving_gauge_replacement_state_sha256        82a51fbe171e9d8374d68eae9082e5f80d9b57bcbfb42107f6697a5568b65059
```

### 1.1 The acceptance ledger was not persisted — and this is a code defect

`refused_transition_indices`, which `NOTES-W13I` §11 called this arm's headline
question, **cannot be read from this run.** A recursive grep for
`boundary_acceptance`, `refused_transition_indices` and `write_norm_ratio` over
the whole sealed run directory — `receipt.json`, `diagnostics.json`,
`run.log`, `published-child/provenance/run.json` (24 MB), `certificate.json`,
both restart-capsule manifests — returns **zero files**.

The reason is not that the ledger was skipped. It is that nothing lifts it:

* `hf_lfm2_moving_gauge_residual_reclosure.py:5203-5245` seals the full ledger
  (`refused_by_sweep`, `refused_transition_indices`, per-decision
  `incumbent`/`proposed` write errors, `write_norm_ratio`, and the
  per-parameter `weight_frobenius_norm_ratio_by_parameter` — the literal 4.73x
  signature) into `report["diagnostics"]["boundary_acceptance"]`, and
  `:5439-5462` puts a compact `boundary_acceptance_summary` on the publication
  body.
* `hf_lfm2_dynamic_closed_loop_adapter.py:13532` (at `b65c81f5`; `:14633` at
  local HEAD) lifts **only** `replacement_state.report["report_sha256"]`,
  `["replacement_weight_sha256"]` and `["changed_coordinate_count"]` out of
  that report. `acceptance_schema`, `writer_reclosure_acceptance`,
  `boundary_regression_tolerance`, `boundary_acceptance_sha256` and
  `boundary_acceptance_summary` are all dropped on the floor.

So the ledger exists only as the digest `82a51fbe…`, with no preimage on disk.
**Deliberately not fixed here** — it is outside the scoped fix and it touches
the closed-loop adapter, which is where `d296b602` currently has a live
regression. It is the single highest-value one-line-class change for r14: five
keys, already computed, already sealed, thrown away at the boundary.

### 1.2 What the gate did, measured from the run's own tensors

`restart-capsules/capsule-000013-1929267…` is this run's child exactly as the
fourteen per-layer boundary solves left it, **before**
`child_closed_loop_body_reclosure_capture_started` (21:17:31Z). The published
child is after the body reclosure, the final normalization and the output
interface. That is the same pair `NOTES-W13B` §"pipeline stage" measured on r12.

Frobenius norms, fp64, read-only
(`/home/cereal/nf-w13m/claude-r13-read/writer-norms.txt`):

| tensor | capsule 13 | published | ratio |
|---|---|---|---|
| `model.layers.0.conv.in_proj.weight` | 33.834669 | **33.834669** | **1.0000 — byte-identical** |
| `model.layers.0.conv.out_proj.weight` | 26.7082 | 65.4264 | **2.450** |
| `model.layers.0.feed_forward.w2.weight` | 17.6956 | 124.8230 | **7.054** |
| `model.layers.1.conv.out_proj.weight` | 33.2701 | 548.1270 | **16.475** |
| `model.layers.1.feed_forward.w2.weight` | 16.8979 | 259.3325 | **15.347** |
| `model.layers.2.self_attn.out_proj.weight` | 23.3639 | 71.3388 | 3.053 |
| `model.layers.4.self_attn.out_proj.weight` | 23.4615 | 184.2161 | 7.852 |
| `model.layers.5.conv.out_proj.weight` | 31.3378 | 3857.0475 | 123.080 |
| `model.layers.7.conv.out_proj.0.weight` | 6.3246 | 6161.0258 | **974.144** |
| `model.layers.7.conv.out_proj.1.weight` | 12.0006 | 6.3246 | 0.527 |
| `model.layers.12.self_attn.out_proj.weight` | 37.8831 | 5866.0034 | 154.845 |
| `model.layers.13.conv.out_proj.weight` | 33.5691 | 3265.1855 | 97.268 |
| `model.layers.13.feed_forward.w2.weight` | 16.5794 | 1591.5065 | 95.993 |
| `lm_head.weight` | 53.073653 | 55.367774 | 1.043 *(output interface, not reclosure)* |
| `model.embed_tokens.weight` | 53.073653 | **53.073653** | 1.0000 |

**83 of 93 non-norm weights changed. 83 − 1 (`lm_head`, the output interface) =
82, exactly `install.metadata.changed_physical_parameter_count: 82`.** So every
changed weight below the head is the body reclosure's, and the accounting closes.

The ten unchanged: `embed_tokens`, and every `conv.conv.weight` at layers
0/1/3/5/7/9/11/13. Layer 0's `conv.in_proj.weight` is unchanged to the last
digit — **the exact r12 signature reproduced**: the reclosure leaves layer 0's
input projection alone and rewrites only its writer.

**Answer to the headline question: no, the gate did not refuse the layer-0/1
rewrites — it accepted them, along with all 82.** A refused boundary keeps the
incumbent writer, i.e. ratio exactly 1.0; only the ten unchanged tensors have
that, and none of them is a writer.

Distribution over the 82 the body reclosure rewrote: **81 up, 1 down**
(`layers.7.conv.out_proj.1` at 0.527, the other half of layer 7's factorized
pair). Median ratio **33.147x**; 77 of 82 above 2x; 64 of 82 above 10x.

**And the gate as specified could not have refused them.** Its metric is
`_BOUNDARY_ACCEPTANCE_METRIC` — the boundary's own *relative write error* in
its own solve frame (`hf_lfm2_moving_gauge_residual_reclosure.py:4262-4271`).
`write_norm_ratio` and `weight_frobenius_norm_ratio_by_parameter` are
**recorded as evidence, never gated on** (`NOTES-W13C` §"Receipt" says so
explicitly). A writer can grow 974x and still cut its own boundary's write
error. `require_boundary_improvement` is doing exactly what it says; the r12
failure mode is a boundary-local improvement that is globally destructive, and
this gate is the wrong instrument for it. That is the finding, not a bug report
against the gate.

Two caveats stated plainly:

* Layer 7's writer is **factorized** (`out_proj.0` / `out_proj.1`). The
  974.144x and 0.527x are per-factor; the composed operator's scale is roughly
  their product (~513x), not 974x. Every other row is a single dense writer.
* Norm ratio is not the gate's metric, so "the gate accepted a 974x rewrite" is
  a statement about outcome, not about a gate malfunction. The per-boundary
  write errors that *were* the decision are the ones §1.1 could not recover.

### 1.3 What the reclosure cost, in the one number that does compose

| | held-out terminal relL2 |
|---|---|
| after layer 13, before body reclosure | **0.806258** |
| after body reclosure (`body_reclosure.validation.metrics.heldout_terminal_relative_rmse`) | **0.971148** |

The reclosure moved the terminal metric **0.165 worse** and stayed under the
1.0 floor by 0.029. It was accepted; the body reclosure is not covered by the
per-layer terminal non-regression tolerance (0.25), which only guards layer
installs.

---

## 2. Terminal composition floor — it passed, at every layer

`--dense-closed-loop-terminal-composition-floor`,
`--dense-closed-loop-terminal-regression-tolerance 0.25`,
`--dense-closed-loop-terminal-absolute-floor-minimum-layer 0`,
`--dense-closed-loop-terminal-free-run-tripwire`.

All 14 layers `validation_accepted: true`. `terminal_layer_index 13`,
`terminal_rows 13464`, `terminal_no_branch_relative_rmse_baseline 1.0`,
`validation_tolerance 1e-08`.

| L | cand relL2 | base relL2 | cand cos | improvement | **terminal relL2** | terminal cos | terminal reg | floor failed | abs floor | non-reg |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.921592 | 0.828806 | 0.593407 | −0.092786 | **0.969881** | 0.384979 | −0.032388 | false | pass | pass |
| 1 | 1.148792 | 1.198004 | 0.478405 | +0.049211 | **0.932394** | 0.457169 | −0.037488 | false | pass | pass |
| 2 | 1.390184 | 1.470696 | 0.359531 | +0.080512 | 0.913080 | 0.484940 | −0.019314 | false | pass | pass |
| 3 | 1.387393 | 1.384911 | 0.320777 | −0.002483 | 0.889468 | 0.519139 | −0.023611 | false | pass | pass |
| 4 | 1.374215 | 1.419129 | 0.323304 | +0.044914 | 0.905411 | 0.486690 | **+0.015943** | false | pass | pass |
| 5 | 1.039856 | 1.043712 | 0.334244 | +0.003856 | 0.885287 | 0.516386 | −0.020124 | false | pass | pass |
| 6 | 1.017758 | 1.060358 | 0.356271 | +0.042600 | 0.840180 | 0.581237 | −0.045107 | false | pass | pass |
| 7 | 0.996585 | 1.003575 | 0.354848 | +0.006990 | 0.847366 | 0.570204 | **+0.007186** | false | pass | pass |
| 8 | 0.948188 | 1.017795 | 0.426990 | +0.069607 | 0.809602 | 0.618652 | −0.037764 | false | pass | pass |
| 9 | 0.946036 | 0.960000 | 0.385365 | +0.013963 | 0.791426 | 0.632105 | −0.018176 | false | pass | pass |
| 10 | 0.881436 | 0.923700 | 0.488935 | +0.042265 | 0.789816 | 0.624546 | −0.001610 | false | pass | pass |
| 11 | 0.886032 | 0.941343 | 0.453229 | +0.055310 | 0.721728 | 0.704487 | −0.068087 | false | pass | pass |
| 12 | 0.780132 | 0.834959 | 0.663907 | +0.054827 | **0.664374** | **0.765223** | −0.057354 | false | pass | pass |
| 13 | 0.806258 | 0.664374 | 0.697337 | −0.141884 | **0.806258** | 0.697337 | **+0.141884** | false | pass | pass |

**The number the program has been waiting for: terminal held-out relL2
0.806258 after fourteen layers, 0.971148 after the body reclosure, against the
1.0 emit-zero baseline and against r12's 4.4728.** That is a **4.6x**
improvement on the composed terminal metric, and the first time this arm has
finished a fold under the emit-zero floor.

Reading notes on the table:

* The terminal metric improves **monotonically to layer 12** (0.9699 →
  0.6644), then layer 13's install gives back +0.1419 — the single largest
  regression event, inside the 0.25 tolerance, so accepted.
* Four layers have a **negative per-layer improvement** (0, 3, 13) or a
  positive terminal regression (4, 7, 13). All were accepted:
  `performance_metrics_are_decision_authority: false` and
  `counterfactual_performance_threshold_passed: false` at layer 0. The floor
  and the non-regression tolerance are the only clauses with authority
  (`terminal_floor_retains_authority_under_diagnostic_only: true`), and neither
  tripped.
* **Regression tolerance events: 0 fired.** Largest observed +0.141884 against
  a 0.25 budget; second largest +0.015943.
* **Tripwire: never evaluated.** `terminal_free_run_tripwire_enabled: true`,
  `terminal_free_run_tripwire_evaluated: **false**` at all 14 layers,
  `terminal_free_run_tripwire_observed: null`,
  `terminal_free_run_chance_match_fraction: 0.0`,
  `terminal_free_run_tripwire_stops_acceptance: false`. The advisory free-run
  tripwire produced **no observation at all this run**. That is worth chasing:
  it is the one clause that was supposed to shout at chance-level free-run, and
  §4's free-run number is chance-level.

Post-body stages, same held-out rows:

```
final_normalization  candidate_relative_rmse 0.939250  cos 0.352587   improvement +0.388361  accepted
output_interface     body_latent_relative_rmse 0.904021
                     interpolation_alpha 0.032840  materialized_delta_ratio 0.247500 (cap 0.25)
                     transform_condition_number 2.536632  min singular value 0.640935
                     full-vocab logit relL2  0.765063 -> 0.749703
                     parent-winner top1      0.076426 -> 0.083556
                     known-seq token NLL     6.894938 -> 6.782162
                     full-vocab parent KL    3.970743 -> 3.848456
                     eos_margin_rmse         5.673538 -> 5.700403   (FAILED, +0.026865)
                     failed_gate_count 1, failed_gates "eos_margin_rmse", recorded
                     diagnostically without numeric decision authority
```

The output interface is at its **delta cap** (0.2475 of 0.25) — it wanted more
transform than it was allowed. Worth noting for r14.

---

## 3. Per-layer depth telemetry (item 8)

**Item 8's receipt table does not exist for this run, and would not have
existed even on success.** Two independent defects, both verified:

1. `_closed_loop_depth_telemetry` is called with
   `result.published.compilation.plan` (`run_lfm2_dynamic_catalog.py:12784`),
   and `DynamicFoldPlan` is a `slots=True` frozen dataclass whose fields are
   `parent_graph_fingerprints, parent_state_fingerprints,
   child_graph_fingerprint, child_state_fingerprint, evidence_fingerprint,
   exact_target_unique_parameters, actions, gaps, registry_fingerprint,
   ledger_fingerprint`. **There is no `report`.** `getattr(plan, "report", None)`
   returns `None`.
2. Even given a report, the function reads `closed_loop["layers"]`
   (`:10951`), while the closed-loop report emits **`layer_reports`**
   (`dynamic_closed_loop_fold.py:2501` and `:3290`).

`per_layer_validation_decisions` and `body_reclosure_validation` would both
have come back empty. `tests/test_run_lfm2_dynamic_catalog_r13_retention.py:184`
passes because its `_plan_with_closed_loop_report()` fixture is a
`SimpleNamespace(report={"closed_loop": {"layers": [...]}})` — a hand-made
shape that matches neither the real plan object nor the real key.

**Deliberately not fixed here** (out of the scoped fix, and it needs a decision
about which object the catalog should read — `compilation.run.attempts[0]
.transaction_report["closed_loop"]` is the one that actually carries it).

The data itself is intact and was recovered directly from
`published-child/provenance/run.json` →
`attempts[0].transaction_report.closed_loop` (extracted read-only to
`/home/cereal/nf-w13m/claude-r13-read/closed_loop.json`). §2's table *is*
`per_layer_validation_decisions`. The conditioning and gate profile:

| L | operator | cand fit relL2 | solver cond number | σ_min | gelsd fallback | max abs gate mean drift | max abs gate σ drift | gate saturation (all 3 thresholds) | update scale gate/up/down |
|---|---|---|---|---|---|---|---|---|---|
| 0 | short_conv | 0.920737 | 1.26e+06 | 0.1945 | false | 0.2922 | 0.3953 | **0.0000** | 0.428 / 0.691 / 0.064 |
| 1 | short_conv | 1.147524 | 1.21e+06 | 0.2262 | false | 0.5740 | 0.2915 | **0.0000** | 0.417 / 0.633 / 0.036 |
| 2 | full_attention | 1.373792 | 3.48e+07 | 0.0040 | false | 0.3850 | 0.2162 | 0.0000 | 0.149 / 0.220 / 0.029 |
| 3 | short_conv | 1.383037 | 3.37e+07 | 0.0054 | false | 0.5309 | 0.1767 | 0.0000 | 0.143 / 0.194 / 0.021 |
| 4 | full_attention | 1.359065 | **3.76e+10** | 0.0000 | false | 2.0974 | 0.1898 | 0.0000 | 0.010 / 0.011 / 0.018 |
| 5 | short_conv | 1.041715 | **1.16e+11** | 0.0000 | false | 0.5316 | 0.2644 | 0.0000 | 0.009 / 0.010 / 0.007 |
| 6 | full_attention | 1.018322 | 9.01e+08 | 0.0002 | false | 1.5700 | 0.1352 | 0.0000 | 0.034 / 0.042 / 0.007 |
| 7 | short_conv | 0.996459 | 1.28e+10 | 0.0000 | false | 0.7350 | 0.4442 | 0.0000 | 0.012 / 0.012 / 0.009 |
| 8 | full_attention | 0.946857 | 5.73e+09 | 0.0001 | false | 0.5874 | 0.1024 | 0.0000 | 0.028 / 0.028 / 0.011 |
| 9 | short_conv | 0.946977 | 1.79e+09 | 0.0002 | false | 0.7758 | 0.4665 | 0.0000 | 0.025 / 0.027 / 0.016 |
| 10 | full_attention | 0.874351 | 6.49e+08 | 0.0009 | false | 1.2975 | 0.2810 | 0.0000 | 0.065 / 0.058 / 0.021 |
| 11 | short_conv | 0.877656 | **4.17e+10** | 0.0000 | false | 2.0109 | 0.6336 | 0.0000 | 0.007 / 0.008 / 0.020 |
| 12 | full_attention | 0.768971 | 4.89e+09 | 0.0001 | false | 1.8455 | 0.4329 | 0.0000 | 0.021 / 0.024 / 0.028 |
| 13 | short_conv | 0.799676 | 9.39e+08 | 0.0007 | false | **3.2814** | **1.0419** | 0.0000 | 0.026 / 0.039 / 0.023 |

Three readings:

* **Gate saturation is dead as a hypothesis.** At thresholds 2.0, 4.0 and 8.0,
  every layer reports `mean_fraction_above_positive_threshold 0.0`,
  `mean_fraction_below_negative_threshold 0.0`, and
  `channels_more_than_half_above/below_threshold 0`. Nothing saturates
  anywhere. `correction_applied: false`, `decision_role: diagnostic_only`.
* **The MLP normal system goes effectively singular from layer 4.** σ_min falls
  below 1e-4 and the condition number reaches 1.16e+11 at layer 5. The
  `solver_path` is `primal_solve` at every layer and
  `silent_gelsd_fallback_taken` is **false** everywhere — no silent fallback
  masked it.
* **The correction collapses with it.** `update_scale` on `mlp_gate` goes
  0.428 (L0) → 0.010 (L4) → 0.007 (L11). From layer 4 on, the solver is barely
  correcting at all. That is `NOTES-W13B` §"Stage A — boundary solve, 26x" —
  the directionally-correct, under-scaled signature — now visible as a
  *depth-dependent* collapse, not a layer-0 quirk.
* Gate distribution drift grows monotonically-ish with depth: max abs channel
  mean drift 0.29 (L0) → 3.28 (L13).

### 3.1 Layer 0 and 1 against the layer-0 study's targets

| quantity | layer-0 study solve (W13B) | r12 published (W13B) | **r13 layer 0** | **r13 layer 1** |
|---|---|---|---|---|
| write cosine | **0.9995** | 0.1712 | *not recoverable* (§1.1) | *not recoverable* |
| write relL2 | **0.0245** | 1.0986 | *not recoverable* | *not recoverable* |
| writer norm ratio (post-reclosure) | 0.998 | **4.733** | **2.450** | **16.475** |
| post-operator / post-layer held-out relL2 | **0.0181** | 0.8358 (post-operator) | 0.921592 (post-layer) | 1.148792 |
| held-out post-layer relL2 (W13A framing) | — | **1.159** | **0.921592** | 1.148792 |
| post-layer residual cosine | — | — | 0.593407 | 0.478405 |
| MLP fit relL2 (candidate / current) | — | — | 0.920737 / 0.944303 | 1.147524 / 1.167645 |

The honest comparison is the last-but-two row, which is the same construction
on both sides: **r12's layer 0 sat at 1.159, above the emit-zero baseline;
r13's sits at 0.9216, below it.** Layer 1 is still above at 1.1488.

The **write**-level quantities the study targeted — cosine 0.9995, relL2
0.0245 — live in the boundary acceptance ledger and are unrecoverable from this
run (§1.1). `candidate_mean_cosine_similarity 0.5934` is the *post-layer
residual* cosine, not the write cosine, and quoting it against 0.9995 or
against r12's 0.171 would be comparing different objects. **Not claimed.**

What can be said: r13 is **48x** off the study's demonstrated solve at layer 0
(0.9216 vs 0.0181, different-but-adjacent constructions), so the solver-side
headroom `NOTES-W13B` proved is still almost entirely unclaimed.

---

## 4. `free_run_token_match 0.00102` — context and comparison

The string occurs **once** in the 1.5 MB `run.log`, inside the `final_receipt`
line. Its home is `prepublication_gate_evidence.trajectory_and_combined_events`
and `diagnostics.behavior_gate_events` — the **cached-parent teacher-forced
trajectory gate**, not the routed decode.

It appears at **two phases with byte-identical metrics**:
`accepted_in_memory_child` and `freshly_reloaded_child`.

```
teacher_top1                     0.1022584438227304     (10.226 %)
free_run_token_match             0.0010179843909060061  ( 0.102 %)
mean_residual_cosine             0.71934603908029
mean_residual_relative_rmse      0.676963836981147
trajectory_score                 0.25322590066516126
child_forward_count              6636
parent_forward_count             0        <- cached parent, never a live forward
cached_full_self_consistent      true
span_mapped_residual_layers      true
manual_intelligence_review_required  true
numeric_decision_authority       false
performance_metrics_diagnostic_only  true
```

Definitions, so the comparison is exact: `teacher_top1` is
`report["teacher_forced"]["top1_agreement"]` and `free_run_token_match` is
`report["free_run"]["token_match_fraction"]`
(`run_lfm2_dynamic_catalog.py:6974-6975`). Both are **child vs. the cached
parent's trajectory**, not child-vs-child.

| | r12 (`reports/r12-certified-20260804/R13_PLAN.md`) | **r13** | |
|---|---|---|---|
| teacher-top1 | 1.86 % | **10.23 %** | **5.5x better** |
| free-run token match | 0.44 % | **0.102 %** | **4.3x worse** |
| mean residual cosine | 0.574 | **0.719** | better |
| mean residual relL2 | 0.809 | **0.677** | better |
| teacher / free-run ratio | 4.2x | **100x** | **gap widened 24x** |

**The R13 plan's item-1 success criterion is not met.** It reads: "free-run
token match rises decisively above 0.44 % (target: within 3x of teacher-top1
rather than 4x below it)." Free-run *fell* to 0.102 %, and the exposure-bias
gap the on-policy round was built to close **widened from 4.2x to 100x**, even
though every other trajectory metric improved. Teacher-forced agreement,
residual cosine, residual rRMSE and the composed terminal floor all moved the
right way; the one metric that measures the child on its own visited states
moved the wrong way.

No argmax-retention diagnostic is sealed at trajectory level. The nearest
argmax evidence in the run is the output interface's teacher-forced
`parent_winner_top1_agreement` 0.076426 → 0.083556 (§2), explicitly flagged
`parent_winner_top1_and_margin_are_diagnostics: true`. `free_run_lcp` and
`cached_full_top1_diagnostic_fraction` are computed by `_gate_summary` but are
**not** in the sealed event summary.

---

## 5. On-policy extension evidence — consumed, and here is the proof

```
receipt.parent.on_policy_extension_evidence_consumed        true
receipt.parent.on_policy_round_count                        1
receipt.parent.on_policy_parent_capture_invocations         0
receipt.on_policy_evidence_policy.consumes_extension_evidence      true
receipt.on_policy_evidence_policy.consumed_round_manifest_count    1
receipt.on_policy_evidence_policy.all_rows_are_fit_rows            true
receipt.on_policy_evidence_policy.extension_evidence_consumer_gap  null
receipt.on_policy_evidence_policy.review_rows_are_never_on_policy  true
```

Evidence-set members in the receipt (`parent.on_policy_evidence_set`):

```
member_count                     2
base_cache_key                   c29b67370d5e33be47a11daccb1e6c2144d6e85efb024621458dc14ca5bd5ef7
base_batch_count                 256
extension_cache_keys             ["084ab7dc8914097d04d0ab4ec0da8db75fa909db9f0b2269055c33692e11a561"]
extension_batch_counts           [115]
extension_fit_batch_count        115
extension_selection_or_validation_batch_count   0
batch_count                      371        (256 base + 115 extension)
fit_batch_count                  269        (of 371; 102 held out)
every_extension_carries_a_target_preimage       true
all_extension_rows_are_fit_rows                 true
extension_target_preimage_sha256 ["0ac1ecd64958854ffe790e25f7ec70e57d3ea7338c0d00055356fdc0303a8d7f"]
round_manifest_sha256            ["75a690219ad8821fbef7f1ffc8dc66611ece0f5f71c57bdbc8123b760b76433c"]
evidence_set_cache_key           f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
evidence_set_corpus_sha256       878cbd931e7f785e8a665de08044ae1d4f059cf7cb6feec86dc4f977082fbb89
on_policy_evidence_identity_sha256  75217282d5f3d7ca397abcc14ba645d118c7833d93c12a3ebe25aecd381067e5
parent_forward_calls             0
```

Cross-checked against the tokenizer bridge in `diagnostics.json`:
`on_policy_extension_record_count 115`, `on_policy_merged_record_count 371`,
`on_policy_merged_paired_record_count 371`,
`on_policy_extension_replay_is_the_sealed_target_preimage true`,
`parent_cache_input_mode target-tokenizer-canonical-then-one-row-mapped-to-donor`,
`parent_model_forwards 0`. And against
`certificate.json.evidence_fingerprint = f62110d7…` — the published artifact is
bound to the merged evidence set, not to the base cache.

**Yes: the on-policy extension evidence was consumed in the solves.** 115 of
371 batches (31 %) are the child's own greedy rollout rows, all of them fit
rows, and the rollout policy was `greedy: true, seed 0, max_prompts 64,
max_new_tokens 16, max_context_tokens 512, max_capture_batches 256`. The
`evidence_fingerprint` on the certificate is the merged key, so this is the
first r13 arm publication that is provably solved on child-visited states.

That makes §4's result sharper, not weaker: **on-policy evidence was genuinely
consumed and free-run agreement still fell.** 16 new-token rollouts at 31 % of
the corpus were not enough to move the metric they were aimed at.

Phase caches (unchanged from `NOTES-W13L` §5): global topology 6 hits / 0
misses; balanced basis hit `52cbe3fd…`, completion `975e8286…`, 0 parent
forwards / 0 tensor loads / 0 training steps. The merged basis is still the
peer's `a0025` payload, inherited not rebuilt.

Replay: `prefix_by_prefix_canonical`, merge-adjacent ON and admitted, sharded
over `cuda:0,cuda:4`, 108/108 captures sharded, 0 fallbacks, observed speedup
**1.971x**, replay wall 7566.3 s against 14913.7 s of worker time.

---

## 6. The failure: root cause and fix

### 6.1 What actually happened

```
2026-08-07T10:45:47.174Z  system_fold_completed   published=true
                          accepted_action_count 1
                          exact_physical_parameter_elements 53,552,368
2026-08-07T10:45:47.229Z  failure   ProductionLfm2CatalogError  fail_closed=true
2026-08-07T10:45:47.259Z  final_receipt  failed_closed  c9a8f746…
```

55 ms. The child was published — `publication_present: true`, certificate,
provenance and carrier package all on disk and all authenticated — and then the
very next statement in the `uses_carrier_routes` branch refused.

`standalone_child_authenticated` and `manual_review_started` never emitted, so
**the routed manual dossier never ran.** That is why there is no dossier.

### 6.2 The assertion, and why it was wrong

`experiments/run_lfm2_dynamic_catalog.py:8656` (pre-fix):

```python
phases = tuple(str(value.get("phase")) for value in events)
in_memory = phases.count("accepted_in_memory_child")
fresh_reload = phases.count("freshly_reloaded_child")
if in_memory < 1 or fresh_reload < 2:
    raise ProductionLfm2CatalogError(
        "Carrier route smoke did not execute on the in-memory child, "
        "stage reload, and final reload."
    )
```

Observed: `in_memory = 1`, `fresh_reload = 1`. The sealed
`diagnostics.whole_output_events` has exactly two entries, both
`route_smoke_accepted: true`, with **distinct** runtime identities
(`2fb6ac00…` in-memory, `b65bb71f…` fresh reload), and `run.log` confirms only
two evaluator ordinals ever ran (`evaluator_ordinal 0 accepted_in_memory_child`
at 09:42:07Z, `evaluator_ordinal 1 freshly_reloaded_child` at 10:37:22Z).

**The hypothesis is confirmed.** `dynamic_fold_artifact.py:3102-3106`:

```python
reused_staged_evaluation = bool(
    reuse_proven_identical_fresh_reload_evaluation
    and final_evaluation_fingerprint == staged_evaluation_fingerprint
)
```

When the fingerprint matches, the publisher carries the stage decision forward
and never calls `fresh_reload_evaluator(final_reloaded)`. Every *structural*
final-destination check still runs — the model is loaded, its topology
fingerprint, state fingerprint, inventory, frozen-parameter state and gradient
absence are all re-verified — only the behaviour callback is skipped. The
fingerprint (`:762`) binds the final destination's own artifact tree, portable
child state sha, child state fingerprint, state inventory, structural topology
fingerprint, loader id + implementation + resource digests, evaluator
implementation + resource digests, the behaviour gate manifest, and the
inference execution contract. So the skipped subject is *proven identical by
digest* in every input the decision depends on.

This is exactly the class `a8b65e15` fixed — "Assert gate subjects, not
evaluation counts" — except that commit fixed five **test** suites and this one
is **production code** in the catalog. The toy carrier e2e lane cannot catch
it: `test_authenticated_carrier_routes_stop_at_the_sealed_production_abi`
proves the routed lane is inexpressible at toy scale (the sealed 4096-row/H816
ABI), so `_verify_carrier_route_smoke_phase_coverage` has **no end-to-end
coverage at all**. It only ever ran for the first time in production.

### 6.3 Why I did not just turn the dedup off

`reuse_proven_identical_fresh_reload_evaluation=False` is the obvious fix and
it is the wrong one. The evaluator the publisher calls is
`CombinedLfm2FinalEvaluator` — trajectory gate **plus** carrier smoke. The
carrier smoke really is cheap (`--routed-smoke-max-new-tokens 2`, one prompt,
`child_body_forward_calls: 2`, docstring "Cheap routed ABI proof"), but the
trajectory gate next to it is `child_forward_count 6636` and, in this run, the
tokenizer-bridge parent projection alone took **55 minutes** on the first
evaluator. Disabling the dedup buys one 2-token generation for roughly an hour
of recomputation, and contradicts `a8b65e15`'s proof that the dedup is correct.

Lifting the dedup's fingerprint into the certificate is also wrong: `a8b65e15`
deliberately made the certificate **invariant** to the dedup ("the certificates
were already proven identical either way"), and `fresh_reload_source` /
`final_destination_reload_performed` are hardcoded constants at
`dynamic_fold_artifact.py:349-350` precisely so it stays that way.

### 6.4 The fix

Take the task's stronger branch, but pay for it where it is actually cheap:
**the catalog smokes the published final destination itself.**

* `experiments/run_lfm2_dynamic_catalog.py:351-380` — subject names as module
  constants. The publisher labels both of its reloads with one phase string,
  which is precisely why counting that string went stale; the subject is now a
  separate, explicit field.
* `:8568-8684` — `CarrierRouteSmokeBehaviorEvaluator.evaluate_model` and a new
  `evaluate_published_child_reload` share one `_route_smoke` body. The new
  method **refuses to run on a model this evaluator registered**
  (`has_packaged_runtime` must be false), so it can only ever smoke a genuine
  standalone reload carrying the routes that were written to disk, not the
  build-time registration.
* `:8656-8666` — every event now carries `subject` and `runtime_source`.
* `:12600-12634` — after the counted-state verification, after the portable
  receipt (so it cannot move that identity) and before the dossier, the catalog
  runs the 2-token smoke on `final_routed_child` — the standalone reload of
  `published-child/child` it already performs and already authenticates — then
  verifies coverage and emits a new
  `carrier_route_smoke_coverage_verified` progress record.
* `:8685-8749` — `_verify_carrier_route_smoke_phase_coverage` asserts on the
  **set of subjects covered**, not on any count, and additionally refuses any
  recorded-but-unaccepted subject. Schema bumped to `…-phase-coverage-v2`.
* `:13104-13121` — the receipt drops the hardcoded
  `"in_memory_stage_and_final_reload_smoke_expected": 3` and records
  `in_memory_stage_and_final_reload_smoke_subjects`,
  `publisher_fresh_reload_smokes` (1 or 2) and
  `publisher_final_reload_reused_by_proof` — **so the receipt records which**,
  as required.

Net effect: the three-subject claim becomes literally true and stops depending
on the dedup; a widened dedup that dropped the final-destination subject
entirely still fails closed and now names the subject it lost; and the run
costs one extra 2-token generation.

`dynamic_fold_artifact.py` is **unchanged**. The dedup is untouched. Nothing
was weakened: the assertion is strictly stronger than the count it replaces,
because it now proves a final-destination subject was smoked instead of
inferring it from a phase tally.

### 6.5 Verification

```
ruff check experiments/run_lfm2_dynamic_catalog.py tests/test_run_lfm2_dynamic_catalog.py
   3 errors — byte-identical to the pre-change baseline (3 pre-existing F401
   unused imports; verified by running ruff against the saved .orig)

pytest tests/test_run_lfm2_dynamic_catalog.py tests/test_dynamic_fold_artifact.py \
       tests/test_run_lfm2_dynamic_catalog_r13_retention.py \
       tests/test_run_lfm2_dynamic_catalog_portable_runtime.py \
       tests/test_run_lfm2_dynamic_catalog_tokenizer_bridge.py -q -p no:randomly
   207 passed, 1 skipped in 62.03s

pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py \
       tests/test_lfm2_carrier_route_runtime.py \
       tests/test_lfm2_carrier_child_package_fresh_process.py \
       tests/test_lfm2_carrier_portable_runtime.py \
       tests/test_lfm2_carrier_bridge_publication.py -q -p no:randomly
   3 failed, 49 passed in 278.99s
```

**Those 3 failures are pre-existing and unrelated.** The identical 3 fail on the
untouched tree (baseline run before any edit: `3 failed, 6 passed` on the e2e
file alone), and all three fail with

```
AssertionError: {'category': 'alternation_pass_failed_closed', 'pass_index': 1,
 'pass_failure': {... 'type': 'Lfm2DynamicClosedLoopAdapterError',
 'message': 'Selected parent component payload changed after authenticated load.'}}
```

which is `d296b602`'s regression, exactly as `NOTES-W13L` §7 attributed it.
**New datum: it reproduces in the toy carrier e2e lane in ~4 minutes on CPU.**
W13L could only attribute it by blob archaeology across two 20-minute
production runs; `test_carrier_dense_core_alternates_fold_round_and_refold`,
`test_bridged_carrier_lane_alternates_fold_round_and_refold` and
`test_carrier_production_margin_pin_composes_through_a_round` are a local
reproducer. Whoever fixes `d296b602` should start there.

New/changed tests, following `a8b65e15`'s pattern:

* `test_carrier_smoke_coverage_requires_memory_stage_and_final_reload` now
  accepts **both** shapes — publisher evaluated both reloads, and publisher
  reload deduped — asserts they are still distinguishable
  (`coverage_sha256` differs, `publisher_final_reload_reused_by_proof`
  flips), and fails closed on four regressions: a missing
  final-destination subject, a missing publisher-fresh-reload subject, an
  unlabelled legacy ledger, and a recorded-but-rejected subject. The
  simulated widened-cache regression that motivated `a8b65e15` — dropping a
  subject entirely — is caught by name.
* `test_carrier_route_smoke_is_structural_and_enforces_every_count` gains
  coverage of `evaluate_published_child_reload`, including that it does **not**
  register a build-time runtime (`len(installs)` stays 1) and that it refuses a
  build-time-registered model.

**Not committed.** No `git add`, no commit, no push. Backups of both files
before editing:
`…/scratchpad/backup-20260807/{run_lfm2_dynamic_catalog.py.orig,
test_run_lfm2_dynamic_catalog.py.orig}` plus `HEAD.txt`. The pre-edit runner
digest was `2af6de96d8f147dbaf03db91a41545c3724179ba53859dfbb312a391aea04ee7`,
**identical to the digest `NOTES-W13L` §1 recorded as deployed**, so the file I
fixed is byte-for-byte the file that failed. `hf_lfm2_moving_gauge_residual_reclosure.py`,
`dynamic_fold_artifact.py` and `experiments/run_lfm2_dynamic_catalog.py` are
also identical between the deployed `b65c81f5` and local HEAD `6f276ade`
(`git diff --stat b65c81f5 HEAD --` on those three paths is empty).

One self-inflicted mistake, recorded: a `pathlib.write_text` round-trip while
renaming a receipt key converted the whole runner file from LF to CRLF. Caught
by git's `CRLF will be replaced by LF` warning and reverted in binary mode
before any test ran; the final diff is 190 changed lines, not 15,190.

---

## 7. First decode read of the published child (Task 3)

**Not the frozen manual review. Nothing scored.** The routed dossier never
generated (§6.1), so this is a deliberately minimal smoke taken so the program
can see the child speak at all.

```
loader        Lfm2CarrierChildCheckpointIO.from_child_directory
              over DirectLocalArtifactIO(tokenizer_directory=<child>,
              tokenizer_fingerprint=fingerprint_local_tokenizer(<child>),
              device="cuda:0", dtype=torch.float32)
              -> loader.registration is None  (child-local, read-only)
entrypoint    semantic_step_generate           (greedy, torch.argmax, no sampling)
prompts       first 10 non-safety rows of research/lfm2_semantic_whole_response_prompts_v1.json
              sha256 0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b  (verified on the server)
caps          maximum_new_tokens 256, maximum_prompt_tokens 8192
child         certificate_sha256 27450bcd…  child_state_fingerprint 53f96b38…
              artifact_tree_sha256 376417b1…  carrier package 503f2898…
counts        physical dense body 53,552,368   effective counted 57,420,144
timing        standalone load 14.4 s, decode 60.0 s for 10 prompts
audit         training_steps 0, backward_calls 0, optimizer_steps 0,
              parent_forward_calls 0, child_generate_calls 0,
              child_lm_head_forward_calls 0,
              child_body_forward_calls 2305 (10 prefill + 2295 generated)
```

That is the same standalone path
`run_lfm2_resumable_review_from_packaged_child` and the catalog's own
post-publication reload use — the certificate's documented portable-load route.
**No authentication wall was hit.** The carrier child package authenticated
first try against its own sealed receipt.

Raw decoded responses, verbatim and unscored:
`reports/r13-first-read-20260807/responses.md`.

Circumstances worth flagging for whoever runs the real review:

* 9 of 10 hit the 256-token cap (`finish_reason: length`,
  `completion_status: censored`). The frozen protocol uses a 1024-token initial
  cap with a 2048 retry, so these are all truncated by *my* cap, not the
  protocol's.
* 1 of 10 (`explanation-seasons`) emitted **EOS immediately** — zero response
  tokens, `finish_reason: eos`, `completion_status: complete`.
* `target_roundtrip_exact: true` on all 10 — the target-tokenizer round trip is
  clean, so the interface is not what is being seen here.
* Donor token counts exceed target token counts on every row (the 16x bridge
  collapses runs of donor tokens onto single target rows).

**First-glance impression, hedged and non-authoritative:** the outputs look
degenerate rather than coherent — short repeating runs of punctuation and
sub-word fragments (`","`, `")`, `"]`, `codecode`, `othoth`) with no
sentence-level structure in any of the ten. One 10-prompt greedy read at a
256-token cap is a smoke, not a measurement, and it is emphatically not the
frozen review's verdict.

---

## 8. Caveats

* **`refused_transition_indices` is still unanswered, and now provably
  unanswerable for this run** (§1.1). §1.2 answers the *question behind it*
  from the tensors; it does not recover the ledger.
* **Norm ratios are outcome, not the gate's metric.** §1.2 does not show the
  gate malfunctioning; it shows the gate measuring something that does not
  constrain the failure.
* **Layer 7's writer is factorized**, so its 974x is a per-factor number
  (§1.2).
* **Item 8's telemetry is broken in two places and was not fixed** (§3). The
  data was recovered by hand out of `provenance/run.json`.
* **The free-run tripwire never evaluated** (§2) and nobody knows why yet.
* **The merged balanced basis was inherited, not built** — still the peer's
  `a0025` payload, content-addressed and HMAC-chained, nothing pinned.
* **The 3 red carrier e2e tests are `d296b602`'s, not this change's** (§6.5),
  proven by a pre-edit baseline run.
* **`_verify_carrier_route_smoke_phase_coverage` has no e2e coverage**, before
  or after this fix, because the routed lane is inexpressible at toy scale
  (§6.2). The new behaviour is unit-tested; the call-site wiring is verified
  only by reading and by ordering checks. That is the weakest link in this
  fix and it should be stated to whoever launches r14.
* **Nothing scored** (§7). Per
  `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`, nothing in this file
  is an intelligence verdict.
* **Nothing committed**, and the only server writes are new read-only outputs
  under `/home/cereal/nf-w13m/claude-r13-read/`. The sealed run directory, the
  restart capsules, the phase caches and the peer's `nf-w13m` files were read
  and never modified.

---

## 9. Artifacts

Local:

| path | |
|---|---|
| `NOTES-W13M.md` | this file |
| `reports/r13-first-read-20260807/responses.md` | the 10 verbatim decoded responses (§7) |
| `experiments/run_lfm2_dynamic_catalog.py` | the fix (§6.4), **uncommitted** |
| `tests/test_run_lfm2_dynamic_catalog.py` | the subject-assertion tests, **uncommitted** |

Server, `/home/cereal/nf-w13m/claude-r13-read/` (all new, read-only outputs):

| file | |
|---|---|
| `closed_loop.json` | 1.76 MB, `transaction_report.closed_loop` extracted from `provenance/run.json` — all 14 layer reports, the body reclosure, output interface, final normalization |
| `residual_basis_slim.json` | the residual-basis report minus the two multi-MB sub-blocks |
| `run_json_head.json` | `provenance/run.json` minus `attempts` |
| `r13_writer_norms.py`, `writer-norms.txt` | §1.2's fp64 Frobenius comparison, 93 tensors |
| `r13_first_read.py`, `first-read.json`, `first-read.log` | §7's decode read |
| `probe.json` | the 1-prompt / 4-token dry run that validated the load path |

`/home/cereal/nf-w13m/` also holds the peer's `d296b602` deploy files from
W13M's own wave; **none were touched** — everything above is under the
`claude-r13-read/` subdirectory.
