# NOTES-W13Q — Stage-A attribution from arm A's item-8 telemetry

Wave 13Q, the final attribution phase of the stage-1 ceiling program. Executes
the audit fork's branch (b) orders after
`reports/armA-review-20260808/ARMA_MANUAL_INTELLIGENCE_REVIEW.md` selected that
branch on 0/15. The report is
`reports/stage1-attribution-20260808/STAGE_A_ATTRIBUTION.md`; this file is the
working record — what was read, what was run, what was checked, and what could
not be answered.

Zero training, zero gradients, zero parent or child forwards. The sealed run
directory was opened read-only; the only server writes were under
`/home/cereal/nf-w13q/`.

---

## 1. What was read, in order

1. `docs/OVERLAP_AUDIT_2026-08-07.md` — Realignment, and the three
   "genuinely new ground" items in Suspicion 3 (depth-dependent solver
   collapse; the 2560→321 MLP boundary; the output interface at its delta cap).
2. `NOTES-W13B.md` — Stage A (solve 0.0245 available, capsule 0.6425 delivered,
   unattributed) and its two named suspects.
3. `NOTES-W13M.md` §3 — the r13 conditioning table and item 8's two defects.
4. `reports/armA-review-20260808/ARMA_MANUAL_INTELLIGENCE_REVIEW.md` — the
   0/15 and the branch selection.
5. `docs/CURRENT_FOLDING_GOAL.md` — the staged-escalation policy, for §4.3 of
   the report.

Then the sealed artifacts, and then the source that produced them
(`hf_lfm2_structural_residual_attention_fold.py`,
`hf_lfm2_dynamic_closed_loop_adapter.py`,
`hf_lfm2_child_closed_loop_refinement.py`,
`experiments/run_lfm2_dynamic_catalog.py`,
`experiments/run_lfm2_residual_attention_fold.py`). Reading the source was not
optional: three of the four load-bearing findings are semantics questions
(what `update_scale` means, what `candidate_fit_relative_rmse` measures, which
`_fit_linear` calls get conditioning diagnostics) that the JSON alone answers
wrongly.

## 2. Extraction

`published-child/provenance/run.json` is 24.6 MB, so it was reduced on the
server rather than downloaded whole.

```
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai \
  --upload  ".../extract_closed_loop.py" "/home/cereal/nf-w13q/extract_closed_loop.py"
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai \
  --script 'cd /home/cereal/nf-w13q && python3 extract_closed_loop.py /home/cereal/nf-w13q/read'
```

`attempts[0].transaction_report.closed_loop` → 2,858,162 B, downloaded to
`reports/stage1-attribution-20260808/remote/closed_loop_attempt0.json`.
`diagnostics.json`, `ledger.jsonl` and `published-child/provenance/child_graph.json`
were downloaded byte-for-byte alongside it.

Note for whoever uses `--download` next: the client treats the destination as a
directory and creates `<dest>/<basename>`, so a "file path" destination lands as
`foo.json/foo.json`. Flattened by hand each time here.

Local analysis, all `C:/Python314/python.exe`:

| script | what it does |
|---|---|
| `extract_stage_a_profile.py` | per-layer profile + Spearman read-out → `per_layer_profile.{json,md}`, `branch_decomposition.json` |
| `verify_clamp_and_ridge.py` | the 42/42 clamp identity and the ridge-authority table → `clamp_and_ridge.txt` |
| `budget_and_lattice.py` | budget accounting from the child graph + width costing from the option lattice → `budget_and_lattice.json` |
| `render_tables.py` | `branch_decomposition.md`, `capacity_risk.md` |

## 3. Item 8's actual state in this run

**Better than W13M reported, and still incomplete.**

* `receipt.json` **does** carry `.fold.closed_loop_depth_telemetry` with 14
  `per_layer_validation_decisions`, `body_reclosure_validation` and 2
  `trajectory_gate_depth_tables`. The two defects W13M §3 found
  (`getattr(plan, "report")` returning `None`, and reading `closed_loop["layers"]`
  against an emitted `layer_reports`) are fixed in this run's source. The receipt
  table W13M said "would not have existed even on success" exists.
* `solver_conditioning_diagnostics_enabled: true` and
  `gate_operating_point_diagnostics_enabled: true` in the same block; the
  launcher passes all three `--dense-closed-loop-*-diagnostics` flags.
* **Three holes remain.**
  1. Operator (`producer`/`consumer`) compiles emit no prior-correction report
     into the record. The objects exist —
     `prior_correction_reports=(in_fit, out_fit)` at
     `hf_lfm2_structural_residual_attention_fold.py:7569` and the attention
     equivalents at `:8401`, `:8565` — but
     `hf_lfm2_dynamic_closed_loop_adapter.py:14027` serializes only the MLP
     triple. **Stage A's original object is the layer-0 operator write, so the
     one thing this wave was sent to attribute is still not instrumented.**
  2. `mlp_down` gets no conditioning: its `_fit_linear` call at `:4943` is the
     only one in the MLP compile that does not forward
     `solver_diagnostics=solver_conditioning_diagnostics`. It is also the
     harshest-clamped role in the run (`update_scale` 0.0071–0.0642, i.e. a
     requested correction of 15.6x–140.1x the prior norm at every layer).
  3. `residual_endpoint_relative_rmse` — the MLP compiler's own branch-local
     endpoint error, computed at `:4970` and returned at `:5010` — is dropped.
     `grep -c residual_endpoint_relative_rmse` returns **0** on `receipt.json`
     and on the extracted `closed_loop` record. That is the number Q2 needed.

Also unchanged from r13: `terminal_free_run_tripwire_evaluated: false` at all
14 layers. The advisory clause has still never produced an observation.

## 4. The four findings, and how each was checked

### 4.1 The prior-delta clamp bound on 42 of 42 MLP solves

`_fit_linear` (`:1204-1216`) writes `correction_norm ← update_scale × raw`, and
`update_scale = min(1, ratio × prior_norm / raw)`. With the body default
`max_prior_delta_ratio = 1.0` (`hf_lfm2_dynamic_closed_loop_adapter.py:1262`) a
bound clamp must leave `correction_norm == prior_norm` exactly. Checked on all
42 solves: worst relative deviation **1.46e-16**. `update_scale < 1` on all 42.
`1/update_scale` recovers the pre-clamp request: **1.4x to 140.4x the prior
norm**.

This overturns W13M §3's reading of the same field ("the correction collapses…
from layer 4 on the solver is barely correcting at all"). The solver was not
barely correcting; it computed a correction up to 140x the prior and was
allowed 1x.

### 4.2 The ridge is inert

Conditioning is measured on the solved system `XᵀX + ridge·I`
(`_solved_system_conditioning`, `:1043`; called at `:1193-1195`). `ridge = 1e-6`
is absolute (`hf_lfm2_dynamic_closed_loop_adapter.py:1260`) against
σ_max 1.4e5–7.1e5 → relative ridge **1.4e-12 to 7.3e-12**. At L4/5/7/8/11 the
ridge supplies >1% of σ_min; at **L5 it supplies 76%** (data σ_min ≈ 3.14e-7).
The solve is inverting the regularizer there, not the evidence.

Neither `ridge` nor `max_prior_delta_ratio` has a CLI override for the body:
`experiments/run_lfm2_dynamic_catalog.py` has exactly one ridge argument
(`--endpoint-ridge`) and one prior-delta argument
(`--endpoint-max-prior-delta-ratio`), both for the output interface. **Every
production run in this program has used ridge 1e-6 and clamp 1.0 at every body
boundary.**

### 4.3 The depth singularity does not track the degradation

ρ(depth, log₁₀ cond) = **+0.486** (not monotone: 1.16e11 at L5 → 9.0e8 at L6 →
1.3e10 at L7 → 6.5e8 at L10 → 4.2e10 at L11).
ρ(log₁₀ cond, post-layer relL2) = **−0.165** — no association, wrong sign.
ρ(depth, post-layer relL2) = **−0.767** — per-layer error *improves* with depth.
Cross-checked on the scale-free `candidate_mean_cosine_similarity`, which has
the same shape (worst 0.3208 at L3, best 0.6973 at L13), so it is not a
relative-RMSE normalization artifact.

Row subsampling refuted: 60,525 rows on all 42 solves, `underdetermined: false`
everywhere, identical at every depth.

### 4.4 The MLP refit is the productive stage; the operator stages own the damage

The stage-delta decomposition (possible because MLP cases are built *after* the
operator installs, so `current_fit` = new operator + old MLP):
MLP stage **+0.1562** total, operator stages **+0.0707**, whole layer
**+0.2269**. The MLP refit improved 12 of 14 boundaries. At the two layers whose
install net-damaged the boundary — L0 (−0.0928) and L13 (−0.1419) — the damage
sits in the operator stage (−0.1164, −0.1540) and the MLP partly repaired it.

Mean whole-layer improvement: **+0.0162 relL2** against boundaries at
0.78–1.39. That number is the finding: the solves are barely engaging.

## 5. Budget arithmetic (the width remedy, costed)

From the child's own graph: short_conv 23,910,400 (44.65%), attention
11,907,840 (22.24%), gated_mlp 11,025,792 (20.59%), embed+head 6,684,672
(12.48%), norms 23,664. Operator stack **66.88%** of the 53,552,368 exact
budget; MLP **20.59%**.

One intermediate unit on one layer = 3 × 816 = **2,448** parameters. Selected
widths sum to 4,504 over 14 layers.

* Every short_conv → the lattice's 704 rung frees **5,510,656** → uniform
  **+160** width (to ~480). Layer-0 post-SwiGLU tail 0.6953 → 0.6153, predicted
  defect 0.09645 → 0.08567.
* Uniform 1280 costs **+32,842,368** (61.3% of the whole budget); 1920 costs
  +54,776,448 (102%); 2559 costs +76,676,256 (143%).
* Absolute ceiling with the operator stack zeroed: width **1,366**.

The lattice already contains widths to 2559 and the global exact-budget solver
saw all of them. **The "wider MLP within exact-4x via the heterogeneous solver"
option is implemented, was offered, and is arithmetically incapable of moving
the tail. Closed path.**

## 6. What this wave could not answer

* **The operator branch is still unattributed.** §3 hole 1. W13B's Stage-A 26x
  was measured on the layer-0 operator write; this run carries no operator
  solve telemetry. The mechanism is shared by construction (same clamp, same
  ridge, same call sites at `:13113/:13166/:13583/:13612/:13647`) but the
  numbers are not claimed.
* **The priors themselves are still not on disk.** Arm A kept
  `capsule-000012` and `capsule-000013` only; W13B §8 recommendation 5 (a
  post-structural-transport checkpoint) was not implemented, so the
  structural-transport prior cannot be scored against the parent — only its
  norm, which the reports do carry.
* **A per-branch endpoint split for Q2** — blocked by hole 3.
* **Absolute (unnormalized) per-layer error** — the validation metrics carry
  only `relative_rmse` and `mean_cosine_similarity`.
* **Whether the remedy moves decode.** Nothing here is a behavior claim. Arm
  A's manual verdict stands at 0/15.

## 7. Recommendation to the next wave

Run R1+R2+R3 from the report §4.1 as **the last stage-1 experiment**: a
condition-capped relative ridge (`ridge = λ·σ_max`, λ ∈ [1e-6, 1e-4], using the
σ_max the telemetry already records), a body `max_prior_delta_ratio` flag that
ships only with it, and the three-object serialization fix. Three small source
changes plus one re-run of the existing launcher, ~17 h, same evidence, same
budget, zero new caches.

If it does not move the frozen manual review, stage 1's ceiling evidence is
complete and §4.3 of the report enumerates what the sealed ceiling report has
to contain.

Two standing constraints this wave leaves untouched and does not authorize
relaxing: R2 must never ship without R1 (unbounded step = A12/W12B, twice
sealed), and no further on-policy rollout variants before the solver arm
resolves (W13A ordering).

## 8. Artifacts

`reports/stage1-attribution-20260808/` — report, four scripts, five derived
tables/JSON, and `remote/` byte copies of the sealed inputs. Server:
`/home/cereal/nf-w13q/` only. No `git add`, no commit, no push.
