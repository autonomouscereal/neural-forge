# Stage-A attribution — arm A (r13 refold), from the item-8 telemetry

Wave 13Q. Executes the audit fork's branch (b) orders
(`docs/OVERLAP_AUDIT_2026-08-07.md` §Realignment, "ARM A read-out plan"), which
`reports/armA-review-20260808/ARMA_MANUAL_INTELLIGENCE_REVIEW.md` selected on
the evidence of 0/15. Two questions: (Q1) why the healthy per-boundary solve
degrades ~26x by capsule time and whether the layer-4+ singularity is the
mechanism; (Q2) whether the 2560 → ~320 MLP contraction is the dominant
per-layer error source.

**Both answers come out against the framing the audit inherited.** The
layer-4+ singularity is real and is *not* the mechanism of the composed
degradation. The MLP contraction is real, quantified, and unrelievable inside
exact-4x, and is *not* the dominant per-layer error source in this run. The
mechanism that actually limited every boundary solve in arm A is a hard-coded
`max_prior_delta_ratio = 1.0` clamp interacting with an effectively
unregularized near-singular normal system — a cause neither of the audit's two
named suspects, and one with a concrete, never-tried, fold-only remedy.

Zero training, zero gradients, zero model calls. Everything below is read from
sealed run artifacts; the sealed run directory was accessed read-only and
nothing was written into it.

---

## 0. Provenance

| item | value |
|---|---|
| Run directory (read-only) | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold` |
| `receipt_sha256` | `1cdec0a61cd94712bc3b26fffb26dce970291acd5b21817f970a030ad3f38168` |
| `configuration_sha256` | `38d1766b8ad0228c9511d1399331b7c9a5343addc68de02a40bdf17248ebb102` |
| Telemetry flags | `solver_conditioning_diagnostics_enabled: true`, `gate_operating_point_diagnostics_enabled: true`, `input_error_covariance_enabled: false`, `margin_row_weights_enabled: false` |
| Item-8 depth telemetry | **present in `receipt.json`** at `.fold.closed_loop_depth_telemetry` (14 `per_layer_validation_decisions` + `body_reclosure_validation` + 2 `trajectory_gate_depth_tables`) — the two defects `NOTES-W13M` §3 found are fixed in this run's source |
| Solver/conditioning per-solve records | `published-child/provenance/run.json` → `attempts[0].transaction_report.closed_loop.layer_reports[*].downstream.compiled.metadata.mlp_prior_correction_reports_json` |
| Local extraction | `reports/stage1-attribution-20260808/remote/closed_loop_attempt0.json` (2,858,162 B), `diagnostics.json`, `ledger.jsonl`, `child_graph.json` |
| Launcher | `/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh` — `--max-selection-rows 13728 --max-fit-rows 40656 --max-validation-rows 13199`, `--dense-moving-gauge-writer-norm-ratio-ceiling 2.0`, `--dense-closed-loop-input-error-multiplier 0`; **no ridge flag and no prior-delta flag exist for the body solve** |
| Restart capsules kept | `capsule-000012`, `capsule-000013` only — `NOTES-W13B` §8 recommendation 5 (a post-structural-transport checkpoint) was **not** implemented, so the priors themselves are still not on disk |

---

## 1. The per-layer collapse profile

Rendered by `extract_stage_a_profile.py` into `per_layer_profile.json` /
`per_layer_profile.md`. `cond`, `σ_max`, `σ_min` are of the solved normal
system `XᵀX + ridge·I` for the `mlp_gate`/`mlp_up` fit; `update_scale` and
`requested Δ/prior` are per role `gate/up/down`; "post-layer relL2" and
"post-layer cos" are the held-out composed post-layer residual against the
chart-projected parent; "terminal relL2" is the held-out terminal metric after
that layer was installed.

| L | operator | MLP h | solve rows | cond(XᵀX) | σ_max | σ_min | solver path | gelsd | update_scale gate/up/down | requested Δ/prior gate/up/down | post-layer relL2 | post-layer cos | terminal relL2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | short_conv | 321 | 60,525 | 1.257e+06 | 2.444e+05 | 1.945e-01 | primal_solve | false | 0.428/0.691/0.064 | 2.3/1.4/15.6 | 0.9216 | 0.5934 | 0.9699 |
| 1 | short_conv | 323 | 60,525 | 1.212e+06 | 2.742e+05 | 2.262e-01 | primal_solve | false | 0.417/0.633/0.036 | 2.4/1.6/27.5 | **1.1488** | 0.4784 | 0.9324 |
| 2 | full_attention | 320 | 60,525 | 3.479e+07 | 1.378e+05 | 3.961e-03 | primal_solve | false | 0.149/0.220/0.029 | 6.7/4.5/34.3 | **1.3902** | 0.3595 | 0.9131 |
| 3 | short_conv | 330 | 60,525 | 3.366e+07 | 1.822e+05 | 5.414e-03 | primal_solve | false | 0.143/0.194/0.021 | 7.0/5.2/48.0 | **1.3874** | **0.3208** | 0.8895 |
| 4 | full_attention | 320 | 60,525 | **3.761e+10** | 2.027e+05 | 5.391e-06 | primal_solve | false | 0.010/0.011/0.018 | 103.0/91.8/56.6 | **1.3742** | 0.3233 | 0.9054 |
| 5 | short_conv | 330 | 60,525 | **1.159e+11** | 1.524e+05 | **1.314e-06** | primal_solve | false | 0.009/0.010/0.007 | 106.8/101.1/140.1 | **1.0399** | 0.3342 | 0.8853 |
| 6 | full_attention | 320 | 60,525 | 9.005e+08 | 2.167e+05 | 2.407e-04 | primal_solve | false | 0.034/0.042/0.007 | 29.7/23.7/139.1 | **1.0178** | 0.3563 | 0.8402 |
| 7 | short_conv | 320 | 60,525 | 1.275e+10 | 2.279e+05 | 1.787e-05 | primal_solve | false | 0.012/0.012/0.009 | 84.9/80.9/110.5 | 0.9966 | 0.3548 | 0.8474 |
| 8 | full_attention | 320 | 60,525 | 5.733e+09 | 3.307e+05 | 5.769e-05 | primal_solve | false | 0.028/0.028/0.011 | 35.7/35.1/91.6 | 0.9482 | 0.4270 | 0.8096 |
| 9 | short_conv | 320 | 60,525 | 1.788e+09 | 4.048e+05 | 2.264e-04 | primal_solve | false | 0.025/0.027/0.016 | 40.0/37.4/64.1 | 0.9460 | 0.3854 | 0.7914 |
| 10 | full_attention | 320 | 60,525 | 6.490e+08 | 5.969e+05 | 9.198e-04 | primal_solve | false | 0.065/0.058/0.021 | 15.3/17.2/48.6 | 0.8814 | 0.4889 | 0.7898 |
| 11 | short_conv | 320 | 60,525 | **4.170e+10** | 4.380e+05 | 1.050e-05 | primal_solve | false | **0.007**/0.008/0.020 | **140.4**/128.0/49.3 | 0.8860 | 0.4532 | 0.7217 |
| 12 | full_attention | 320 | 60,525 | 4.894e+09 | 7.088e+05 | 1.448e-04 | primal_solve | false | 0.021/0.024/0.028 | 48.1/41.9/35.9 | **0.7801** | 0.6639 | **0.6644** |
| 13 | short_conv | 320 | 60,525 | 9.390e+08 | 6.571e+05 | 6.998e-04 | primal_solve | false | 0.026/0.039/0.023 | 37.8/25.6/43.0 | 0.8063 | 0.6973 | 0.8063 |

Four facts to read off before any interpretation:

* **Every one of the 42 MLP solves (14 layers × {gate, up, down}) hit the
  prior-delta clamp.** `update_scale < 1` at all 42, and the clamp identity
  `correction_norm == 1.0 × prior_norm` holds to a worst relative error of
  **1.46e-16** across them (`verify_clamp_and_ridge.py` check 1,
  `clamp_and_ridge.txt`). There is no ambiguity here: the clamp bound, always.
* **Nothing was under-determined and nothing was subsampled by depth.** All 42
  solves used exactly **60,525 rows** against input widths 816 (gate/up) and
  320–330 (down) — 74x and ~190x over-determined — with
  `underdetermined: false` everywhere and an identical row count at every
  depth.
* **The solver never fell back.** `solver_path: primal_solve` and
  `silent_gelsd_fallback_taken: false` at all 14 layers.
* **Six boundaries (L1–L6) finished *above* the 1.0 emit-zero floor at their
  own endpoint**, and three (L0, L3, L13) left the boundary worse than before
  they were folded. All 14 were accepted, because
  `performance_metrics_are_decision_authority: false`.

---

## 2. Q1 — Stage-A attribution

### 2.1 The mechanism: a 1.0×prior clamp on an unregularized near-singular system

`_fit_linear` (`src/neural_forge/hf_lfm2_structural_residual_attention_fold.py:1204-1216`)
computes the closed-form correction, then does:

```
update_scale = min(1.0, max_prior_delta_ratio * prior_norm / correction_norm)
if update_scale < 1.0:
    correction      = correction * update_scale
    correction_norm = correction_norm * update_scale
```

so with the body default `max_prior_delta_ratio = 1.0` a bound clamp leaves
`correction_norm == prior_norm` exactly, and `1/update_scale` recovers the
pre-clamp request. That is what the run records:

| L | role | update_scale | raw correction the solver computed, in units of ‖prior‖ |
|---|---|---|---|
| 0 | mlp_gate / up / down | 0.428 / 0.691 / 0.064 | 2.3x / 1.4x / **15.6x** |
| 5 | mlp_gate / up / down | 0.009 / 0.010 / 0.007 | 106.8x / 101.1x / **140.1x** |
| 11 | mlp_gate / up / down | 0.007 / 0.008 / 0.020 | **140.4x** / 128.0x / 49.3x |

The solver asked to move 2.3x–140x the prior's norm and was allowed 1.0x, at
every layer, in every role. **`update_scale` is not "how much the solver
corrected"; it is how much of the solve survived a clamp whose budget is set
by the structural-transport prior's own norm.**

Why the requests are enormous is the second half of the mechanism: the solve
is effectively unregularized. `ridge = 1e-6` is *absolute*, and it is added to
`XᵀX` whose largest singular value is 1.4e5–7.1e5. The relative ridge is
therefore **1.4e-12 to 7.3e-12** — twelve orders below σ_max. At layers 4, 5,
7, 8 and 11 the ridge supplies more than 1% of the *smallest* singular value,
and at layer 5 it supplies **76%** of it: the data's own smallest singular
value there is ≈ 3.14e-7, so the solve is inverting the regularizer, not the
evidence, in that direction.

| L | σ_max | σ_min(XᵀX + ridge·I) | σ_min − ridge | ridge share of σ_min | relative ridge |
|---|---|---|---|---|---|
| 0 | 2.444e+05 | 1.945e-01 | 1.945e-01 | 0.00% | 4.09e-12 |
| 4 | 2.027e+05 | 5.391e-06 | 4.391e-06 | 18.55% | 4.93e-12 |
| 5 | 1.524e+05 | 1.314e-06 | **3.143e-07** | **76.09%** | 6.56e-12 |
| 7 | 2.279e+05 | 1.787e-05 | 1.687e-05 | 5.60% | 4.39e-12 |
| 11 | 4.380e+05 | 1.050e-05 | 9.503e-06 | 9.52% | 2.28e-12 |

Full table in `clamp_and_ridge.txt`. Chain, stated plainly:

> the child's own post-operator activations are numerically rank-deficient in
> the 816-wide chart at depth → the normal system is near-singular → an
> effectively unregularized least-squares solution has an enormous norm → the
> hard 1.0×prior clamp truncates it to a unit-relative step → the installed
> MLP is the structural-transport prior plus ~1–43% of the correction the
> evidence asked for.

Neither the ridge (`hf_lfm2_dynamic_closed_loop_adapter.py:1260`) nor the
clamp (`:1262`) has a CLI override in `experiments/run_lfm2_dynamic_catalog.py`;
the only prior-delta and ridge flags there are `--endpoint-max-prior-delta-ratio`
and `--endpoint-ridge`, which govern the output interface, not the body. **Every
production run in this program has therefore used ridge 1e-6 and clamp 1.0 at
every body boundary.**

### 2.2 (a) Is the depth trend monotone, and does it track validation degradation? No, and no.

Spearman rank correlations over the 14 layers (`extract_stage_a_profile.py`):

| pair | ρ |
|---|---|
| depth vs log₁₀ cond | **+0.486** |
| depth vs gate `update_scale` | −0.565 |
| depth vs post-layer relL2 | **−0.767** |
| log₁₀ cond vs post-layer relL2 | **−0.165** |
| gate `update_scale` vs post-layer relL2 | +0.222 |
| log₁₀ cond vs per-layer improvement | +0.147 |

* **Not monotone.** cond steps up at L2 (1.2e6 → 3.5e7), jumps at L4–L5
  (3.8e10, 1.16e11), then falls to 9.0e8 at L6, back up to 1.3e10 at L7, down
  to 6.5e8 at L10, up to 4.2e10 at L11. It oscillates over two and a half
  orders of magnitude below the L5 peak.
* **It does not track the validation degradation, and the weak association it
  does have runs the wrong way.** ρ(log cond, post-layer relL2) = −0.165 —
  worse conditioning is, if anything, associated with slightly *lower*
  relative error. The worst boundaries are L2 and L3 (relL2 1.390 / 1.387,
  cosine 0.360 / 0.321) at cond ~3.4e7; the two most singular layers, L4 and
  L5, sit at 1.374 and 1.040 and are on the *improving* side of the curve.
* **Per-layer error improves with depth** (ρ = −0.767), and this is not a
  normalization artifact of relative RMSE: the scale-free `post-layer cosine`
  has the same shape — worst at L3 (0.3208), then monotone-ish up to 0.6973 at
  L13.

So: the layer-4+ singularity is a real property of this fold, it is faithfully
recorded, and it is **not** the mechanism that produced the composed
degradation. It is a property of the deep layers, and the deep layers are the
*better*-fitting ones.

### 2.3 (b) The two named suspects

**Row subsampling — refuted, decisively.** Every solve used the same 60,525
rows; `underdetermined: false` on all 42; the ratio of rows to unknowns is 74x
at the gate/up boundary and ~190x at the down boundary; there is no
depth-dependent variation in row count at all. `--max-fit-rows 40656` caps the
evidence partition's fit **path** count (`deterministic_path_partition`,
`experiments/run_lfm2_residual_attention_fold.py:2430` — its `paths` are
`batch-XXXXXX/lfm2.full_prefix` circuit rows), not the activation rows a solve
consumes, and the activation-row count each solve did consume is recorded
independently and is constant across depth. Row subsampling
cannot explain a depth-dependent anything in this run, and it does not explain
the layer-0 deficit either.

**Structural transport priors — convicted, but as the *budget*, not as the
anchor.** The clamp's allowance is literally `max_prior_delta_ratio × ‖prior‖`,
so the structural-transport prior sets both the starting point of every body
solve and the maximum distance that solve is permitted to travel from it. With
the requested correction running to 140x the prior norm, the prior is the
operative constraint on every MLP boundary in the fold. The priors' own weights
are still not on disk (no post-structural-transport capsule), so the prior
cannot be scored against the parent directly — but its norm is recorded on all
42 solves and that is enough to convict it as the limiter.

**A third cause, which the audit did not name and which is mechanically prior
to both: the ridge is inert.** §2.1. This is the cause with the remedy.

### 2.4 (c) Where does the solve stop making meaningful updates?

`update_scale` on `mlp_gate`: **0.428 at L0, 0.417 at L1, then 0.149/0.143 at
L2/L3, then ≤ 0.034 from L4 onward, bottoming at 0.0071 at L11.** By the
strict reading of the question — "the solver is effectively returning the
prior" — solving stops at **layer 4**, and from there the installed MLP is the
structural-transport prior plus ≤3.4% of the computed correction.

Two qualifications that matter more than the number:

1. **The `mlp_down` role stops much earlier and much harder.** It is at
   `update_scale` 0.064 already at layer 0 (a 15.6x request) and 0.0071 at
   layer 5 (a 140x request). The 321→816 down projection — the write that
   actually puts the MLP's output into the residual, and the read-out of the
   2560→321 contraction — is the least-solved object in the entire fold at
   *every* depth including layer 0. It is also the only role with **no
   conditioning telemetry at all**: the `mlp_down` call at
   `hf_lfm2_structural_residual_attention_fold.py:4943` is the one `_fit_linear`
   invocation in the MLP compile that does not forward
   `solver_diagnostics=solver_conditioning_diagnostics`. Item 8 instrumented
   two of the three MLP roles.
2. **"Where solving stops" is not "where the damage is."** Layer 0 has the best
   conditioning in the run (cond 1.26e6, σ_min 0.194), the second-largest
   `update_scale`, and its install still moved the held-out post-layer residual
   from **0.8288 to 0.9216** — 11% *worse* than not folding the layer at all.
   The audit's expectation was that the deep singular layers are where the
   fold dies; the run says the fold is barely alive at every layer and dies
   hardest at layer 0 and layer 13, both well-conditioned.

### 2.5 What this telemetry structurally cannot say

The `Lfm2PriorCorrectionReport` objects exist for the **operator** branch too —
`prior_correction_reports=(in_fit, out_fit)` for the short-conv producer
(`hf_lfm2_structural_residual_attention_fold.py:7569`) and the attention
equivalents (`:8401`, `:8565`) — but the adapter serializes only the MLP triple
(`hf_lfm2_dynamic_closed_loop_adapter.py:14027`). The operator reports are
computed and dropped. Consequently:

* There is **no** `update_scale`, `prior_norm`, `correction_norm`,
  `solver_path` or condition number for any operator solve at any layer in
  this run.
* `NOTES-W13B`'s Stage-A 26x was measured on the layer-0 **operator write**
  (capsule relL2 0.6425 against an available 0.0245). **This run's telemetry
  does not instrument that object.** What it establishes is the analogous —
  and, per §3, larger — story on the MLP branch, plus the fact that the same
  clamp and the same inert ridge are wired into the operator compiles
  (`hf_lfm2_dynamic_closed_loop_adapter.py:13113`, `:13166`, `:13583`,
  `:13612`, `:13647`), which is why the same mechanism is the leading
  hypothesis there.

Stated without hedging: **item 8, as deployed, attributes the MLP branch. The
operator branch is still unattributed, for the same reason it was unattributed
after W13B — nobody wrote the numbers down.** Closing that is one serialization
change (§4.1 R3), not an experiment.

---

## 3. Q2 — the 2560 → ~320 MLP contraction

### 3.1 The per-branch endpoint split asked for is not in this run

Enumerating every numeric leaf of every layer report, the only per-layer error
objects the sealed record carries are:

* `downstream.compiled.metadata.candidate_fit_relative_rmse` /
  `current_fit_relative_rmse` — the **composed post-layer** endpoint on fit
  rows, before/after the MLP refit (`compute_lfm2_candidate_post_layer`,
  `hf_lfm2_child_closed_loop_refinement.py:1982-2031`, target
  `parent_post_mlp_residual_target_space`);
* `validation.metrics.{baseline,candidate}_relative_rmse` and
  `candidate_mean_cosine_similarity` — the same composed endpoint, held out;
* the terminal metrics.

The MLP compiler *does* compute a branch-local endpoint error —
`residual_endpoint_relative_rmse` at
`hf_lfm2_structural_residual_attention_fold.py:4970`, returned on the compiled
artifact at `:5010` — and it is **discarded**: the string occurs zero times in
`receipt.json`, in the extracted `closed_loop` record, in `diagnostics.json`
and in `ledger.jsonl`. The operator branch emits no error metric at all. So a
true "operator endpoint vs MLP endpoint" decomposition **cannot be produced
from this run**, and any number presented as one would be fabricated.

### 3.2 What *is* recoverable: the stage-delta decomposition

The MLP cases are built after the producer and consumer installs, so
`current_fit` is the composed post-layer error with the **new operator and the
old (structural-transport) MLP**, and `candidate_fit` is the same after the MLP
refit. `baseline` (held out) is the composed post-layer error with **none** of
the layer's three installs. That gives an exact MLP-stage delta, an exact
whole-layer delta, and an operator-stage remainder. The fit-vs-held-out gap at
the shared candidate point is carried as the error bar; its worst value is
**0.0164** (`branch_decomposition.md`).

| L | operator | baseline (held out) | operator installed, old MLP (fit) | after MLP refit (fit) | after everything (held out) | MLP-stage Δ | whole-layer Δ | operator-stage Δ (residual) |
|---|---|---|---|---|---|---|---|---|
| 0 | short_conv | 0.8288 | 0.9443 | 0.9207 | 0.9216 | +0.0236 | **−0.0928** | **−0.1164** |
| 1 | short_conv | 1.1980 | 1.1676 | 1.1475 | 1.1488 | +0.0201 | +0.0492 | +0.0291 |
| 2 | full_attention | 1.4707 | 1.3837 | 1.3738 | 1.3902 | +0.0099 | +0.0805 | +0.0706 |
| 3 | short_conv | 1.3849 | 1.3779 | 1.3830 | 1.3874 | −0.0052 | −0.0025 | +0.0027 |
| 4 | full_attention | 1.4191 | 1.3568 | 1.3591 | 1.3742 | −0.0022 | +0.0449 | +0.0472 |
| 5 | short_conv | 1.0437 | 1.0451 | 1.0417 | 1.0399 | +0.0034 | +0.0039 | +0.0005 |
| 6 | full_attention | 1.0604 | 1.0226 | 1.0183 | 1.0178 | +0.0043 | +0.0426 | +0.0383 |
| 7 | short_conv | 1.0036 | 1.0030 | 0.9965 | 0.9966 | +0.0066 | +0.0070 | +0.0004 |
| 8 | full_attention | 1.0178 | 0.9545 | 0.9469 | 0.9482 | +0.0077 | +0.0696 | +0.0619 |
| 9 | short_conv | 0.9600 | 0.9593 | 0.9470 | 0.9460 | +0.0124 | +0.0140 | +0.0016 |
| 10 | full_attention | 0.9237 | 0.8913 | 0.8744 | 0.8814 | +0.0170 | +0.0423 | +0.0253 |
| 11 | short_conv | 0.9413 | 0.9015 | 0.8777 | 0.8860 | +0.0239 | +0.0553 | +0.0315 |
| 12 | full_attention | 0.8350 | 0.7918 | 0.7690 | 0.7801 | +0.0228 | +0.0548 | +0.0320 |
| 13 | short_conv | 0.6644 | 0.8118 | 0.7997 | 0.8063 | +0.0121 | **−0.1419** | **−0.1540** |
| **sum** | | | | | | **+0.1562** | **+0.2269** | **+0.0707** |

Three readings:

1. **The MLP refit is the *productive* stage, not the destructive one.** It
   improved the composed post-layer endpoint at 12 of 14 layers, mean +0.0112,
   total +0.1562 — **69% of the entire per-layer improvement the fold
   achieved** (+0.2269), against +0.0707 for the two operator stages combined.
2. **Where a layer install net-damaged its own boundary, the damage is in the
   operator stage.** L0: whole-layer −0.0928 with an operator-stage residual of
   −0.1164 while the MLP refit repaired +0.0236. L13: −0.1419 whole-layer,
   −0.1540 operator-stage, +0.0121 from the MLP.
3. **The absolute magnitudes are the real finding.** The mean whole-layer
   improvement is **+0.0162 relL2** against boundaries sitting at 0.78–1.39. A
   fold whose boundaries are barely moved by being solved is not a fold whose
   problem is which branch is worse; it is a fold in which the solve is not
   engaging at all — which is exactly §2.1's clamp.

### 3.3 The contraction, measured as capacity

The contraction's fingerprint is unmistakable, but it is a *capacity*
measurement, not a per-layer solve failure. From the run's own planner evidence
(`heterogeneous_capacity_option_bank.layer_evidence`, cached, zero forwards) at
the selected layout — `capacity_risk.md`:

| L | type | selected MLP width | post-SwiGLU tail | collision fraction | tangent tail | predicted defect |
|---|---|---|---|---|---|---|
| 0 | conv | 321 | **0.6953** | 0.8746 | 0.7064 | 0.09645 |
| 1 | conv | 323 | 0.6550 | 0.8738 | 0.6976 | 0.08002 |
| 2 | full_attention | 320 | 0.6830 | 0.8750 | 0.7008 | 0.06792 |
| 3 | conv | 330 | 0.6114 | 0.8711 | 0.6551 | 0.06177 |
| … | | | | | | |
| 12 | full_attention | 320 | 0.3631 | 0.8750 | 0.4735 | 0.04257 |
| 13 | conv | 320 | 0.2824 | 0.8750 | 0.5425 | 0.04154 |

R13 item 4's "~70% unrepresentable post-activation tail" is exactly the layer-0
row: post-SwiGLU tail 0.6953, tangent tail 0.7064, collision fraction 0.8746 —
i.e. 87% of the parent's 2,560 neurons share a child channel. And the gate
operating point confirms the consequence downstream: the child's mean
per-channel σ falls from **1.200x** the parent target's at L0 to **0.512x** at
L13, while max absolute channel-mean drift climbs **0.292 → 3.281**. The
contracted MLP progressively loses the parent's channel statistics with depth.

Two things it does *not* do, both important:

* It does not track the per-layer error. Post-SwiGLU tail *falls* with depth
  (0.6953 at L0 → 0.2824 at L13, with L5 an outlier at 0.0377 whose predicted
  defect is nonetheless the run's largest at 0.29869) — the contraction gets
  *less* severe deeper — while the per-layer composed error also falls. The two
  move in the same direction, so the tail cannot explain why L2/L3 are the
  worst boundaries, and it points the wrong way for a depth-driven collapse.
* The gate operating point compares the child's 320-channel pre-activations
  against **the parent's targets already projected into the child's contracted
  neuron basis** (`parent.channels == child.channels == 321` at layer 0). It
  therefore measures how well the child hits its own contracted target, not how
  much of the parent's 2,560-channel behaviour the contraction threw away. The
  latter lives only in the planner's cached tail fractions, which are
  predictions, not measurements of this child.

### 3.4 Q2 verdict

**The evidence does not support the MLP contraction as a primary cause of the
composed collapse in this run.** It supports three things instead:

* The MLP contraction is a real, quantified, ~70%-post-activation-tail capacity
  defect at layer 0 (0.28–0.70 across depth), and §4.1 shows it is
  arithmetically unrelievable inside exact-4x.
* The MLP *solve* is the better-behaved of the two branches in the stage-delta
  ledger: it produced 69% of the fold's per-layer improvement and repaired part
  of the operator's damage at both layers where the layer net-regressed.
* The predicted MLP defect (0.032–0.30 per layer) is 8–14x smaller than the
  measured per-layer composed error (0.78–1.39). Whatever is setting the
  measured error, it is not the contraction's predicted defect.

The audit's "consistently-worst object across three independent lanes" claim is
not contradicted as a *capacity* statement — but this run gives it no support as
the per-layer error source, and the one number that could have adjudicated it
directly (`residual_endpoint_relative_rmse`) was computed and thrown away.

---

## 4. Consequence

### 4.1 A concrete, never-tried, fold-only remedy exists for the attributed cause

The attributed cause is §2.1: **every body boundary solve in every production
run this program has ever executed delivered ‖prior‖ of correction, because an
effectively unregularized near-singular normal system produced a solution 1.4x–140x
that size and a hard-coded 1.0×prior clamp truncated it.** Three changes, in
order, all fold-only, all closed-form, no gradients:

**R1 — condition-capped (relative) ridge at the body boundary solves.**
Replace the absolute `ridge = 1e-6` with `ridge = λ·σ_max` for the solved
system, λ ∈ [1e-6, 1e-4], capping the effective condition number at 1/λ and
bounding the solution norm by the evidence rather than by the prior. This is
the "condition-capped solve" R13 item 8's own note named. Cost: one expression
in `_fit_linear` (`hf_lfm2_structural_residual_attention_fold.py:1184`) plus a
`--dense-closed-loop-ridge` flag — **σ_max is already computed and recorded**
by the telemetry this report reads, so the cap needs no new linear algebra.
Never tried: W13B §5's ridge sweep {0, 1e-8, 1e-6, 1e-4, 1e-2} ran on the
layer-0 **operator** boundary, which is well conditioned (cond 1.26e6,
σ_min 0.19) and where the clamp never binds; the MLP boundary, where the clamp
binds 42/42, has never been swept.

**R2 — a CLI override for the body `max_prior_delta_ratio`, paired with R1.**
`hf_lfm2_dynamic_closed_loop_adapter.py:1262` hard-codes 1.0 and
`experiments/run_lfm2_dynamic_catalog.py` exposes no flag for it. Cost: one
argument plus one config passthrough. **This must not ship without R1** —
relaxing the clamp alone on a cond-1e11 system reproduces the W12B/A12
unbounded-step failure exactly (pinv step 40x for 1% gain, held out 6x worse;
unregularized reclosure +515%), and the audit's own do-NOT list forbids
unbounded steps. With R1 the bound comes from the conditioning; the clamp then
becomes a backstop rather than the operative limiter.

**R3 — serialize the three telemetry objects that already exist (prerequisite,
near-zero cost).** (a) Emit the operator producer/consumer
`prior_correction_reports` (`:7569`, `:8401`, `:8565`) into the compiled
metadata alongside the MLP triple (`hf_lfm2_dynamic_closed_loop_adapter.py:14027`).
(b) Emit `residual_endpoint_relative_rmse` (`:5010`) — the branch-local MLP
endpoint error — so Q2 can be answered directly next time. (c) Pass
`solver_diagnostics=solver_conditioning_diagnostics` to the `mlp_down` fit
(`:4943`), the least-solved role in the fold and the only one with no
conditioning block. Without R3 the next run reproduces this report's central
gap: an unattributable operator stage.

**Total cost.** Three small source changes, a test pass, and one re-run of the
existing arm-A launcher with two new flags — ~17 h wall clock on the same
hardware, the same sealed evidence, the same 4x budget, zero new caches. It is
the cheapest remaining stage-1 experiment and the only one aimed at a cause
that is measured rather than hypothesized.

**What it is expected to buy, honestly.** W13B demonstrated 46x of unclaimed
solver headroom at layer 0 under the production compilers (composed
post-operator relL2 0.0181 against the published 0.8358); arm A's layer 0
delivered a composed post-layer 0.9216, *worse* than the 0.8288 it started
from. No production run has ever been executed in a configuration that lets a
boundary claim that headroom. That is a real, unexploited gap. It is also the
last one.

### 4.2 What the remedy cannot do — the MLP contraction is arithmetically closed

`budget_and_lattice.py`, from the child's own graph and the run's own option
lattice:

| component | parameters | share of the 53,552,368 exact budget |
|---|---|---|
| short_conv | 23,910,400 | 44.65% |
| attention | 11,907,840 | 22.24% |
| **operator stack** | **35,818,240** | **66.88%** |
| gated_mlp (14 layers) | 11,025,792 | 20.59% |
| token embedding + output head | 6,684,672 | 12.48% |
| norms | 23,664 | 0.04% |

Selected MLP widths: `[321, 323, 320, 330, 320, 330, 320, 320, 320, 320, 320,
320, 320, 320]`, sum 4,504. One intermediate unit on one layer costs
3 × 816 = **2,448** parameters. The heterogeneous option lattice already offers
MLP widths {320, 321, 322, 323, 330, 354, 483, 578, 1280, 1920, 2559} and
short-conv internal widths {704, 1024} at every layer — the global exact-budget
solver saw all of them and chose these. So:

* The only lattice move that frees parameters is dropping every short-conv to
  the narrow 704 rung: **+5,510,656 parameters**, which buys a uniform
  **+160** intermediate width (to ~480). At layer 0 that moves post-SwiGLU tail
  0.6953 → 0.6153 and predicted defect 0.09645 → 0.08567 — an 11% reduction in
  a predicted defect that W13B already showed is not the binding constraint,
  in exchange for 24% of the short-conv width on the branch that owns the
  damage at L0 and L13.
* Reaching the 1280 rung (tail 0.318) costs **+32,842,368** parameters —
  **61.3% of the entire budget**, against an operator stack of 35.8M.
  1920 costs +54.8M (102% of budget); 2559 costs +76.7M (143%).
* **Absolute arithmetic ceiling:** spend every non-interface, non-norm
  parameter on the MLP and give the operator stack literally nothing —
  intermediate width **1,366**, tail ≈ 0.30. That is not a model.

**There is no fold-only remedy for the 2560 → 320 contraction at exact-4x.**
The "wider MLP-only layout within the exact-4x global budget via the
heterogeneous solver" option is already implemented, already offered to the
solver, and is arithmetically incapable of materially moving the tail. This
should be recorded as a closed path.

### 4.3 If R1+R2+R3 does not move decode, stage 1 is complete

Under `docs/CURRENT_FOLDING_GOAL.md`'s staged policy, stage 1 ends when we can
state with sealed evidence "the best composed retention achievable at each
compaction level with every procedural remedy exhausted." After R1+R2+R3 the
list of fold-only remedies aimed at a *measured* cause is empty: per-boundary
gates (built, W13C), terminal floor (built, r13/arm A), on-policy rounds
(consumed, r13), EIV/margin (proven structural no-op at the operator, W13B §5),
reclosure modes (convicted and defanged, arm A), layout arms (closed by §4.2
and by W13B §4), and now the solver's own regularization and step bound.

The sealed stage-1 ceiling report must then contain, at minimum:

1. **Best composed retention at each compaction level, with its manual-review
   verdict.** 1.35x: A9/A11 dual-route, 10–12 of 16 useful-or-partial — the
   proven behavioral frontier. 4x: arm A, terminal 0.806258, **0/15**. Any
   intermediate level (~2x) that gets run. Frozen V5 protocol digest
   `e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` on every
   verdict; no cached-frame metric anywhere in an acceptance path.
2. **The exhaustion ledger** — every fold-only remedy, what it was aimed at,
   what it measured, and its sealed artifact. Including the three negative
   results this report adds: the depth singularity does not track the composed
   degradation; row subsampling is refuted at 60,525 rows and 74x
   over-determination; the MLP contraction is arithmetically unrelievable at
   exact-4x (§4.2 numbers).
3. **The novel assets, stated as results rather than as failures.** The layer-0
   representability forensics (W13B: 46x of solver headroom demonstrated under
   production compilers on sealed evidence). The reclosure conviction and its
   quantified fix (arm A: 27 of 28 writer overwrites refused, terminal
   regression 0.165 recovered, decode unchanged — a real defect removed that
   was necessary but not sufficient). The Stage-A clamp/ridge attribution in
   this report. The A12 cached-fit non-predictivity rule and its two
   re-derivations.
4. **The zero-training provenance chain** for every claim: `training_steps: 0`,
   `backward_calls: 0`, `optimizer_steps: 0`, `parent_model_forward_calls: 0`,
   with receipt digests.
5. **The field comparison** (2026-08-06 taxonomy): no published coherent 4x
   closed-form-only LLM, and this program's 1.35x-works / 4x-fails cliff
   reproducing that boundary — so the ceiling is a property of the technique at
   this ratio, not of this implementation alone.
6. **The explicit authorization sentence** for stage 2, naming which stage-1
   artifacts (fold-as-initialization at 4x, the 1.35x dual-route baseline) the
   distillation stage inherits and what "compute saved versus
   distillation-from-scratch" will be measured against.
7. **The open instrumentation defects that were never closed**, so the ceiling
   is not over-claimed: `terminal_free_run_tripwire_evaluated: false` at all 14
   layers in arm A as in r13 — the advisory clause built to shout at
   chance-level free-run has still never produced an observation; and, unless
   R3 ships, the operator branch's prior-correction reports remain computed and
   discarded.

---

## 5. Caveats

* **Stage-A's original object was the operator write; this telemetry covers the
  MLP.** §2.5. The mechanism is shared by construction (same clamp, same ridge,
  same call site pattern) but the operator numbers are not in this run and are
  not claimed.
* **`current_fit` vs `baseline` mixes fit rows with held-out rows** in the
  operator-stage residual of §3.2. The gap at the shared candidate point is
  measured and small (worst absolute 0.01639, median absolute 0.00311), and is
  carried in `branch_decomposition.json` as `fit_minus_heldout_at_candidate`.
* **Absolute (unnormalized) per-layer errors are not recorded** — only
  `relative_rmse` and `mean_cosine_similarity`. The depth trend in §2.2 is
  therefore cross-checked on the scale-free cosine, which agrees.
* **The capacity tail fractions are planner predictions**, computed from cached
  parent rows before the fold, not measurements of this child. They are used
  here only to cost the width remedy, which is what they were built for.
* **The gate operating point's "parent" is already contracted** to the child's
  neuron basis (§3.3), so it does not measure the 2560-channel loss.
* **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`
  nothing here is an intelligence verdict; the frozen whole-response manual
  review remains the authority, and arm A's is 0/15.
* **Not committed.** No `git add`, no commit, no push. The sealed run directory
  was read only; the only server writes were to `/home/cereal/nf-w13q/`.

## 6. Artifacts

All under `reports/stage1-attribution-20260808/`:

| file | content |
|---|---|
| `STAGE_A_ATTRIBUTION.md` | this report |
| `extract_stage_a_profile.py` | builds the per-layer profile and the Spearman read-out |
| `per_layer_profile.json` | full per-layer record: conditioning, all three prior-correction reports, gate operating point, validation metrics |
| `per_layer_profile.md` | §1's table |
| `branch_decomposition.json` / `.md` | §3.2's stage-delta decomposition |
| `verify_clamp_and_ridge.py` / `clamp_and_ridge.txt` | the 42/42 clamp identity check and the ridge-authority table |
| `budget_and_lattice.py` / `budget_and_lattice.json` | §4.2's budget accounting and width costing |
| `render_tables.py` / `capacity_risk.md` | §3.3's per-layer capacity evidence |
| `remote/closed_loop_attempt0.json` | the extracted sealed closed-loop record (2,858,162 B) |
| `remote/closed_loop_attempt0_map.json` | its structural key map |
| `remote/diagnostics.json`, `remote/ledger.jsonl`, `remote/child_graph.json` | supporting sealed artifacts, byte copies |

Download integrity — local `sha256sum` recomputed against the server after
transfer, all four agree byte-for-byte:

| file | sha256 |
|---|---|
| `remote/closed_loop_attempt0.json` | `9a6e9313e2a8290a29c16d2f34837a6d0dc277d9d9f50ec812f36760c6ec0375` |
| `remote/diagnostics.json` | `cc8d1e64acab127aae6a729d3060dcd3c922424adbdad6c256673397b67b7774` |
| `remote/ledger.jsonl` | `b7d916ae989e460c7c0325cac7ffadd63508592c1c6df15df21e49613d4150bf` |
| `remote/child_graph.json` | `f14b244d002a087b2d972f342dc853c9098d4079c35bdfd3bf27677c1cae5958` |

The sealed run was re-checked after all reads:
`find <run> -newermt "2026-08-08 00:20"` returns **0 entries**, and
`sha256sum receipt.json` is still
`45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10`, the value
the arm-A review recorded. Server-side (new root, nothing else written):
`/home/cereal/nf-w13q/` — `extract_closed_loop.py` and `read/`.
