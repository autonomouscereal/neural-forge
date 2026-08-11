# Stage-1 Matrix Campaign — design (2026-08-10)

Status: DESIGN ONLY. No implementation in this commit. Authorized by Cyril's
stage-1 reopening directive (docs/CURRENT_FOLDING_GOAL.md, addendum
2026-08-10): *test every folding technique at every component level
INDIVIDUALLY — one component folded at a time, everything else copy_exact —
across marginal contraction levels, with per-cell sealed metrics.*

Repo baseline: HEAD `d53cd5e4`. Parent: LFM2.5-230M (`229,693,184` params,
H=1024, 14 layers: short_conv at L0,1,3,5,7,9,11,13; full_attention at
L2,4,6,8,10,12; Q16/KV8/D64; SwiGLU intermediate 2560; vocab 65,536 physical /
64,402 defined). All sealed evidence identities referenced below are the
existing caches — the closed-path ban on re-capturing sealed evidence binds
this campaign.

Governing sealed facts (citations used throughout):

- **S1** cross-width charts near-lossless, attention transport ~free at L0,
  gated-MLP 8.8x worse (`reports/kernel-stage2-20260801/`).
- **S2** layer-0/1 representability exonerated: solver hits cos 0.9995 on the
  sealed 4x evidence, 13 arms identical, no hyper-parameter matters
  (`reports/layer0-study-20260805/`, NOTES-W13B).
- **S3** Arm-A attribution: MLP refit is the productive stage (+0.156 of
  +0.227); operator installs at L0/L13 are the damaging component
  (`reports/stage1-attribution-20260808/`).
- **S4** A9/A11 sparse-OMP token routes: the only production configuration
  with behavioral retention (10–12/16), split C_in/C_out on shared supports.
- **S5** CPCF 100x: macrocell (operator+MLP) beats operator alone.
- **S6** kind translation compiles both directions at 0.65–0.85 val error.
- **S7** cross-family: composition must be scored at depth on pretrained
  weights; the depth wall is real (refusal at L3 on the real parent).
- **S8** chimera: composition succeeds when branch endpoints stay ≤ ~0.4 and
  the channel dimension is untouched.
- **S9** 2x arm: per-component numeric wins do not compose into behavior at
  ≥1.559x body; cached-frame metrics are banned from acceptance.
- **F1–F5** the ceiling report's physics (composition law; direction dies;
  TF/free-run anti-correlation; chart-frame rank deficiency; per-fold body
  contraction is the governing variable).
- **N1–N10** the never-measured list (operator-branch solves dropped, branch
  MLP endpoint never serialized, `mlp_down` unconditioned, free-run tripwire
  never fired, etc.).

The metric policy of the closed-path ledger binds every cell: **free-run token
match/lcp and decode smoke are the behavioral signals; boundary/cached metrics
are diagnostic only and never appear in an acceptance clause.**

---

## 1. The matrix

### 1.1 Axes

**Component axis (rows)** — the unit is one named weight group at one named
site; everything else in the child is `copy_exact` (bit-identical parent
weights, identity charts, untouched norms):

| id | component | parent geometry | sites measured |
|---|---|---|---|
| C01 | token_interface_in (embedding / C_in route) | 65,536×1024 | global |
| C02 | token_interface_out (readout / C_out route) | 1024×65,536 | global |
| C03 | conv_in_proj (short-conv B/C/x projections) | 1024→3×1024 | L0, L5, L13 |
| C04 | conv_kernel (FIR state/kernel width) | 1024-ch conv state | L0, L5, L13 |
| C05 | conv_out_proj (short-conv output writer) | 1024→1024 | L0, L5, L13 |
| C06 | attn_q | 1024→16×64 | L2, L8, L12 |
| C07 | attn_k | 1024→8×64 | L2, L8, L12 |
| C08 | attn_v | 1024→8×64 | L2, L8, L12 |
| C09 | attn_o | 1024→1024 | L2, L8, L12 |
| C10 | attn_per_head (head-count / head_dim contraction) | Q16/KV8/D64 | L2, L8 |
| C11 | mlp_gate | 1024→2560 | L0, L5, L13 |
| C12 | mlp_up | 1024→2560 | L0, L5, L13 |
| C13 | mlp_down | 2560→1024 | L0, L5, L13 |
| C14 | norms (RMSNorm gains incl. final norm) | diag(1024) | L0, final |
| C15 | lm_head (dense head distinct from C02's routed readout) | 1024×65,536 | global |

Site selection rationale: L0 and L13 are the measured operator-damage extremes
(S3); L5 is a quiet mid conv layer; L2/L8/L12 are the largest attention
whole-layer gains (S3). A cell is `component × technique × level × site`; the
site column is folded into the cell id, not a fourth free axis — three sites
per body component, chosen, not exhaustive. Depth-generalization (all 14
layers) is a wave-3 escalation for surviving cells only (S7 rule: composed at
depth or it doesn't count — here "depth" is satisfied because the rest of the
network is the *real parent*, so every cell is automatically scored through 13
untouched downstream layers).

**Technique axis (columns):**

| id | technique | mechanism source |
|---|---|---|
| T1 | dense prior-anchored ridge contraction | production `_fit_linear` (:1138), the S2-exonerated solver |
| T2 | cross-width residual charts | `ResidualBoundaryChart` POD (S1) |
| T3 | balanced controllability/observability basis truncation | balanced-basis phase machinery (S2 inputs) |
| T4 | low-rank factor writers | closed-form SVD writer factorization (the L13 writer-rank-31 feasibility witness, CURRENT_FOLDING_GOAL §evidence) |
| T5 | sparse residual-OMP routes **extended to body components** | A9/A11 mechanics: planted anchor basis → residual OMP (double-MGS, deterministic ties) → **split in/out coefficient solves on shared supports** → local reclosure (`hf_lfm2_sparse_semantic_codebook.py`, `hf_lfm2_causal_semantic_support.py`) — **Cyril's explicit ask** |
| T6 | kind translation (attention↔short-conv) | S6 compilers |
| T7 | attention-relation matching (per-head lifted Gram, Stein solve, Procrustes gauge) | distillation-analog memo #1/#2 |
| T8 | Gram/relation matching at operator outputs (direction/magnitude split) | distillation-analog memo #5 — the only technique whose objective is invariant to the failure's symmetry group, and the only relation technique that covers **short_conv outputs**, the measured wall |
| T9 | progressive multi-pass (schedule ρ_t, child_{t-1} → parent_t) | distillation-analog memo #6; a *level-scheduling* technique orthogonal to T1–T8 |

**Level axis:** marginal contractions of the component's *own* width, plus the
canonical 4x-layout width for that component:

| level | meaning | example (mlp: 2560) | example (conv writer: 1024) |
|---|---|---|---|
| M16 | contract 1/16 (keep 15/16) | 2400 | 960 |
| M8 | contract 1/8 (keep 7/8) | 2240 | 896 |
| M4 | contract 1/4 (keep 3/4) | 1920 | 768 |
| X4 | the width the canonical exact-4x layout assigns this component | per option-bank | per option-bank |

For C01/C02 the levels are anchor-table sizes (M16 = 61,440-row equivalent
route budget down to the proven 4,096 at X4 — the 16x interface is a **solved,
separable cell** per S4 and is entered as a calibration row, not re-proven).
For C10 the levels are head drops (−1 Q head / −1 KV head / −4 Q heads /
4x-layout head plan). For C14 levels are n/a (norms are diagonal; the cell
tests standalone refit quality, N10, at identity size).

### 1.2 Cell prior marks

Legend: **G** expected-good (cite), **U** unknown, **B** expected-bad (cite),
**—** n/a. Marks below are per technique×component; the level modulates
(marginal levels shift marks one notch toward good; X4 one notch toward bad —
F5 is the reason the matrix exists).

| | T1 ridge | T2 chart | T3 balanced | T4 low-rank | T5 OMP routes | T6 kind | T7 attn-rel | T8 Gram | T9 multi-pass |
|---|---|---|---|---|---|---|---|---|---|
| C01 tok-in | B (unanchored maps closed, GOAL closed-paths) | G (S1 token bridge) | U | B (rank-64 fact. 15.6%, GOAL) | **G (S4, proven 16x)** | — | — | U | U |
| C02 tok-out | B (same) | U | U | B (same) | **G (S4 A10/A11 C_out)** | — | — | U | U |
| C03 conv_in_proj | U (never measured, N1) | U | U | U | **U (the ask)** | — | — | U | U |
| C04 conv_kernel | B at X4 (S2: shipped wall; but solver exonerated → M-levels U) | — | G-prior (2x arm kept width 1024, S9) | U | U | B (S6: ρ_far 0.53–0.70) | — | U | U |
| C05 conv_out_proj | U (N4 — never branch-scored beyond L0/L1) | U | U | G-prior (L13 writer-rank-31 witness) | **U (the ask)** | — | — | **U — highest-value unknown: covers the measured wall** | U |
| C06 attn_q | G at M-levels (S1: attn crossing 0.0759; S7 L0 0.131) | G (S1) | U | U | U | B (S6) | U (memo #1: score-kernel gauge) | U | U |
| C07 attn_k | G (same) | G | U | U | U | B | U (jointly with Q — the bilinear cell) | U | U |
| C08 attn_v | G (same); MiniLMv2 prior says V matters most | G | U | U | U | B | U | U | U |
| C09 attn_o | U (writer side never isolated, N1) | G | U | U | U | B | U | U | U |
| C10 per-head | U (never measured, N-class) | — | U | G-prior (head merge = rank op) | U | — | U (relation targets are head-count-free) | U | U |
| C11 mlp_gate | B at X4 (S1: MLP 8.8x worse; S8: channel dim is the loss center) → M-levels U | B (charts don't cover channel dim) | U | U | **U (the ask)** | — | — | U | U |
| C12 mlp_up | same as C11 | B | U | U | **U (the ask)** | — | — | U | U |
| C13 mlp_down | U (least-solved role, N3 — no conditioning telemetry ever) | B | U | U | **U (the ask)** | — | — | U | U |
| C14 norms | U (N10 — fitted inside interfaces, never standalone) | — | — | — | — | — | — | U | — |
| C15 lm_head | B (dense head correction of incoherent body closed; but on coherent 1-component child: U) | U | U | B (GOAL) | G-prior (S4) | — | — | U | U |

T9 applies to any row whose T1–T8 mark survives at M16: its cells are
schedules, not solves, and enter in wave 3.

**Priority order (Cyril's directive):** (1) T5 body routes first — C05, C13,
C11/C12, C03, C08/C09; (2) marginal levels (M16 → M8 → M4) before X4 always;
(3) single-component always — no cell ever folds two components; pairs are a
separate post-campaign design.

### 1.3 Cell count

Executable (non-n/a) cells: body components C03–C13 ≈ 11 rows × ~6 applicable
techniques × 3 sites-or-global × 4 levels, minus n/a and minus marks already
**G-cited** (cited cells are not re-run; the citation is the receipt) — the
enumerated executable set is **≈ 300 cells**. The adaptive policy (§5) expects
to *run* 120–180 of them: X4 and M4 levels are locked until the shallower
level in the same row passes, and expected-bad cells run only as 1-cell
falsification probes per row.

---

## 2. The harness

### 2.1 Pattern: generalize `run_lfm2_layer0_representability_study.py`

The layer-0 study is the proven shape: open sealed caches through
authenticated sessions, construct the boundary input as
`encode(parent residual.pre)` (exact per the first-divergence ledger), run the
**production** compilers for exactly one boundary, score against
chart-projected parent targets on the authoritative held-out split, seal one
JSON record per arm, fail closed on any calibration mismatch. New runner:

`experiments/run_lfm2_matrix_cell.py` — one process, one cell:

```
--cell-id C13.T5.M16.L05          # component.technique.level.site
--parent-trace-cache <sha>        # the sealed 76ff3cbc… cache (reused, never recaptured)
--phase-cache / --hmac-key        # as layer-0 study
--out <dir>                       # writes cell-receipt.json (§3)
```

Stages, all inside one process:

1. **Materialize** the single-component child: full parent `copy_exact`, one
   component replaced per the cell's technique+level. No chart anywhere except
   the component's own (T2 cells); all other boundaries identity.
2. **Solve** with the technique kernel against the sealed evidence rows for
   that site (the trace cache already carries every boundary's activations —
   S2 proved the caches suffice). Calibration arm first: `copy_exact`
   reproduction of the parent's own component must score ≈0 error or the run
   stops before any solve is believed (the layer-0 study's
   `published_child`-gate pattern).
3. **Install + boundary score**: held-out relL2/cos at the component write and
   at `residual.post_operator` (macrocell — S5/S3 rule: never operator alone
   when an MLP shares the layer; the cell still only *modifies* one component,
   but the score is taken at the composed layer output).
4. **Composed terminal**: one child forward pass over the held-out prompt set
   (child forwards at fold time are legal); terminal relL2 vs the emit-zero
   floor AND vs the all-copy-exact control (§3).
5. **Free-run probe**: fixed 8-prompt probe set (sealed once for the whole
   campaign, disjoint from review prompts), 64-token greedy free-run; token
   match / lcp vs the same prompts run on the all-copy-exact control child.
   This finally exercises the never-fired tripwire (N5).
6. **Decode smoke**: 3 fixed prompts, 128-token cap, decoded verbatim into the
   receipt. No scoring at fold time — recorded for the frozen review protocol.
7. **Seal** the receipt (§3), content-addressed, with the cell id and every
   input identity.

### 2.2 Estate reuse (no new lanes where one exists)

- T1: `_fit_linear` unchanged; the cell harness adds **serialization of the
  already-computed** `Lfm2PriorCorrectionReport`s (N1), the branch-local
  `residual_endpoint_relative_rmse` (N2), and conditioning on the `mlp_down`
  solve (N3). Three serialization changes, no solver change — the R3 pattern.
- T2/T3: `ResidualBoundaryChart` / balanced-basis phase, as-is.
- T4: layout accounting + blank materialization already support factored
  writers (GOAL §evidence); the cell enables the guarded path for one writer.
- T5: `hf_lfm2_sparse_semantic_codebook.py` OMP selection generalized from
  vocab rows to component output channels: anchor basis = the kept channels
  (exact one-hot), dropped channels get k-sparse codes over kept ones; C_in
  route = read-side coefficients, C_out route = write-side coefficients,
  shared supports, separate solves — the literal A11 split transplanted.
  `hf_lfm2_causal_semantic_support.py` (built, unrun) supplies the
  causal-damage reselection arm.
- T6: the S6 translation compilers, as-is.
- T7/T8: one new primitive — the two-sided Stein ridge solve
  (`(AᵀA)S(BBᵀ)+λS = AᵀEBᵀ` via two `eigh`) + PSD-project/rank-truncate/factor
  + the dormant Procrustes lane
  (`hf_cpcf_boundary_chart.py::_weighted_orthogonal_procrustes`,
  `analytic_coordinate_completion.py` bridge-procrustes-isometric). One new
  module, `hf_stein_relation_solve.py`, shared by both techniques.
- T9: a driver that re-points "parent" at a published cell child; wave 3.

### 2.3 Runtime budget

Layer-0 study arms ran in minutes each on the server against warm caches. Per
cell: solve (1–5 min; T5 OMP the slowest, but the A9 chunk-cache replay trick
applies), one composed forward over held-out prompts (~1 min), free-run +
decode smoke (~2–3 min at 3×128 + 8×64 tokens). **Target ≤ 12 min/cell, hard
cap 20 min** (a cell that exceeds the cap seals `timeout` and yields).
Evidence caches load once per worker process and are shared across cells at
the same site — batch cells by site.

Wall-clock: wave 1 (30 cells) ≈ **6–8 h on one GPU**. Full adaptive campaign
(120–180 run cells) ≈ **30–40 GPU-hours ≈ 2–3 server-days**, excluding frozen
review time on cells that reach the review gate.

---

## 3. Per-cell sealed receipt schema

One schema, every cell comparable. `schema: "neural-forge-matrix-cell-receipt-v1"`.

```json
{
  "schema": "neural-forge-matrix-cell-receipt-v1",
  "cell_id": "C13.T5.M16.L05",
  "component": "mlp_down", "technique": "sparse_omp_routes",
  "level": "M16", "site": "L05",
  "campaign_wave": 1,
  "code_identity": {"repo_head": "…", "src_tree_sha": "…"},
  "evidence": {
    "parent_trace_cache_sha": "76ff3cbc…",
    "chart_sha": null,
    "probe_set_sha": "…", "smoke_prompt_sha": "…",
    "heldout_split": {"fit_rows": 0, "heldout_rows": 0},
    "parent_model_forward_calls": 0, "backward_calls": 0
  },
  "calibration": {"copy_exact_reproduction_relL2": 0.0, "passed": true},
  "solve": {
    "solver_path": "…", "ridge_relative": 0.0,
    "condition_number": 0.0, "condition_cap_saturated": false,
    "prior_norm": 0.0, "correction_norm": 0.0,
    "requested_clamp_ratio": 0.0, "allowed_clamp_ratio": 0.0,
    "clamp_engaged": false,
    "omp": {"k": 32, "support_reuse_prefix_steps": 0, "double_mgs": true}
  },
  "boundary": {
    "write_heldout_relL2": 0.0, "write_heldout_cos": 0.0,
    "post_operator_heldout_relL2": 0.0, "post_operator_heldout_cos": 0.0,
    "endpoint_band_le_0p4": false
  },
  "composed": {
    "terminal_heldout_relL2": 0.0,
    "emit_zero_floor_passed": true,
    "vs_copy_exact_control_delta": 0.0,
    "norm_ratio_per_layer_minmax": [0.0, 0.0]
  },
  "behavior": {
    "free_run": {"probe_prompts": 8, "tokens_per_prompt": 64,
                 "token_match_vs_control": 0.0, "mean_lcp": 0.0},
    "decode_smoke": [
      {"prompt_sha": "…", "output_verbatim": "…",
       "terminated": "eos|cap|loop|error"}
    ]
  },
  "outcome": {
    "status": "pass|fail|timeout|infra",
    "first_divergent_metric": null,
    "notes": ""
  },
  "wall_time_seconds": 0.0,
  "sealed_at": "…", "receipt_sha": "self"
}
```

Rules:

- **Behavioral fields are the only ones an acceptance clause may read**
  (closed-path metric policy). `boundary.*` and `composed.terminal_*` are
  diagnostic; `emit_zero_floor` is a wreck detector, never progress.
- Decode smoke outputs are stored **verbatim, uncapped-metadata style**
  (eos/cap/loop/error distinct, per the GOAL censoring rule).
- Every solve serializes the full correction report — the N1/N2/N3 gaps are
  closed by construction because the schema makes the fields mandatory.

**Failure-localization convention.** Metrics form a fixed causal chain; the
failing cell's `first_divergent_metric` is the **first** link to cross its
pre-registered threshold, checked in this order:

1. `calibration.passed` → `infra` (cell void, not evidence)
2. `solve.condition_cap_saturated` or `clamp_engaged` with requested/allowed > 4 → `solve-conditioning`
3. `boundary.write_heldout_cos` < 0.95 → `local-solve`
4. `boundary.post_operator_heldout_relL2` > 0.4 → `macrocell-band` (S8 band)
5. `composed.vs_copy_exact_control_delta` > 0.02 → `composition`
6. `behavior.free_run.token_match_vs_control` < 0.90 → `free-run`
7. decode smoke incoherent under review → `behavior`

A cell can pass 1–5 and fail 6–7: that outcome is the F3 signature and is
tagged `free-run` — the most information-dense failure class in the campaign.

**Campaign ledger.** `reports/matrix-campaign-20260810/LEDGER.md` +
`ledger.jsonl` (one line per sealed receipt: cell_id, status,
first_divergent_metric, key numbers, receipt sha). The ledger is
append-only; a re-run of a cell gets a new cell instance id, never an
overwrite. A rendered matrix table (component × technique, best surviving
level per cell) regenerates from the jsonl by
`experiments/render_matrix_ledger.py`.

---

## 4. Execution waves and work split

### 4.1 Wave 1 — highest-information ~30 cells

Two calibration cells first, then Cyril's priority order (body T5 routes,
marginal levels, single-component):

| # | cell | why |
|---|---|---|
| 0a | CTRL.copy_exact | all-copy-exact control child: defines behavioral baselines for every cell |
| 0b | CAL.C01.T5.X4 | A9 reproduction under the harness: proves the harness reproduces the one known behavioral success |
| 1–3 | C05.T5.M16.{L00,L05,L13} | conv writer OMP routes at the measured wall + extremes — the directive's core |
| 4–6 | C13.T5.M16.{L00,L05,L13} | mlp_down OMP (least-solved role, N3) |
| 7–8 | C11.T5.M16.L00, C12.T5.M16.L00 | gate/up OMP at the loss center |
| 9 | C03.T5.M16.L00 | conv in_proj OMP |
| 10–11 | C08.T5.M16.L02, C09.T5.M16.L02 | attn V/O OMP |
| 12–14 | C05.T1.M16.{L00,L05,L13} | dense-ridge baseline for the same writer cells (technique contrast) |
| 15–16 | C13.T1.M16.{L00,L05} | dense-ridge mlp_down + first-ever conditioning telemetry (N3) |
| 17–20 | C06.T1.M16.L02, C07.T1.M16.L02, C08.T1.M16.L02, C09.T1.M16.L02 | first branch-local attention operator measurements ever (N4) |
| 21 | C04.T1.M16.L00 | conv kernel-width marginal (the shipped wall's component, solver now exonerated) |
| 22–24 | C05.T8.M16.L00, C13.T8.M16.L00, C09.T8.M16.L02 | Gram matching at the wall components — the invariance bet |
| 25–26 | C06+C07.T7.M16.{L02,L08} | bilinear QK score-kernel cells (Q,K solved jointly is still one attention-score component) |
| 27 | C05.T4.M16.L13 | low-rank writer at the feasibility-witness site |
| 28 | C09.T4.M16.L02 | low-rank attn_o writer |
| 29 | C10.T1.M16.L02 | drop one Q head (never measured) |
| 30 | C14.T1.—.L00 | standalone norm refit (N10) |

Deferred from wave 1 with cause: T6 kind translation (expected-bad, S6 — one
falsification probe in wave 2); T2/T3 (expected-good, S1 — cited, spot-check
in wave 2); T9 (needs the pass driver; enters wave 3 on surviving rows).

### 4.2 Adaptive escalation (waves 2+)

- A cell that **passes** (per §5) unlocks the next level in its row
  (M16→M8→M4→X4) and its neighboring sites.
- A row whose M16 cell fails with `first_divergent_metric` ∈ {local-solve,
  macrocell-band} is parked; ∈ {composition, free-run} escalates instead to a
  T8/T7 retry of the same cell (invariance-class change) before parking.
- Wave 2 also carries: one T6 falsification probe, T2/T3 spot-checks, the T5
  causal-damage-reselection arm (`hf_lfm2_causal_semantic_support.py`, N-item
  "built and unrun") on the best wave-1 T5 cell.
- Wave 3: T9 multi-pass schedules on rows with ≥2 surviving levels; depth
  generalization (all sites) for surviving technique×component pairs.

### 4.3 Codex work split (disjoint by construction)

**Codex — instrumentation and receipts (no technique code):**
- `experiments/run_lfm2_matrix_cell.py` harness skeleton: cell CLI, cache
  opening, copy_exact materializer, calibration gate, composed-forward /
  free-run / decode-smoke stages, receipt writer, ledger appender, renderer.
- The three serialization closures N1/N2/N3 inside the fold modules
  (report plumbing only — solver math untouched).
- Probe-set + smoke-prompt sealing; `CTRL.copy_exact` and `CAL.C01.T5.X4`
  runs (they exercise only harness + existing A9 code).

**Claude — technique kernels and runs:**
- T5 body-route generalization (OMP over component channels, split in/out
  solves, local reclosure) — the directive's centerpiece.
- `hf_stein_relation_solve.py` (T7/T8 shared primitive + Procrustes lane).
- T4 factored-writer cell enablement behind the existing guards.
- Wave-1 cells 1–30, receipt review, ledger analysis, wave-2 selection.

Interface contract between the halves: the receipt schema (§3) and the cell
CLI — frozen before either side starts. Neither side edits the other's files;
the harness calls technique kernels through one dispatch table
(`technique_registry: dict[str, CellTechnique]`).

---

## 5. Pre-registered interpretation rules

Registered before any cell runs; changing them mid-campaign requires a new
dated addendum, never an edit.

1. **A cell PASSES** iff all of:
   - `behavior.free_run.token_match_vs_control ≥ 0.90` on the fixed probe
     (control = CTRL.copy_exact, same prompts, same seeds), and
   - all 3 decode-smoke outputs are judged coherent-and-on-task under the
     frozen review rubric (reviewed from the verbatim receipt text; cap-hit
     outputs are censored, not failed), and
   - `composed.emit_zero_floor_passed` (wreck detector only — necessary,
     never sufficient, never headlined).
   Boundary metrics appear **nowhere** in this clause.
2. **A technique+component is marked `works at level ℓ`** iff its cells pass
   at ℓ *and every shallower level* at ≥2 of its 3 sites. Non-monotone
   outcomes (pass deep, fail shallow) are flagged `non-monotone` and treated
   as not-working pending a replication cell.
3. **A technique+component is marked `closed`** iff it fails at M16 at all
   attempted sites with `first_divergent_metric` ∈ {local-solve,
   macrocell-band} after the one permitted invariance-class retry (§4.2). A
   `closed` mark cites its receipts and enters the closed-path ledger only via
   the normal GOAL amendment process.
4. **Diagnostic-only readings** (recorded, never verdicts): boundary
   relL2/cos vs the S8 ≤0.4 band; whether the S8 band *predicts* the
   behavioral outcome is itself a campaign output — tallied as
   band-vs-behavior agreement across all sealed cells.
5. **Failure attribution is the schema's chain (§3), mechanically applied.**
   No narrative failure causes: the ledger records
   `first_divergent_metric` and nothing softer.
6. **No cell result generalizes across components, techniques, levels, or
   sites** beyond rules 2–3. In particular a T5 pass at C13 licenses nothing
   about C11, and no set of single-component passes licenses any claim about
   a composed multi-component fold — that claim requires its own
   post-campaign design.
7. **Teacher-forced top-1 never appears in any receipt headline** and, if
   recorded, carries its calibration band (closed-path rule).
8. The campaign's terminal deliverable is the rendered matrix + a dated
   ceiling-report addendum stating, per technique×component: works-to-level,
   closed-with-receipts, or untested-with-reason. Stage-2's gate reopens only
   on that addendum.

---

## 6. Cost summary

| item | estimate |
|---|---|
| wave 1 (32 incl. calibration) | 6–8 GPU-hours, 1 server-day incl. review |
| enumerated executable cells | ≈ 300 |
| expected run count (adaptive) | 120–180 |
| full campaign compute | 30–40 GPU-hours ≈ 2–3 server-days |
| new code | 1 harness runner, 1 Stein module, T5 body generalization, 3 serialization closures, ledger renderer |
| new evidence captures | 0 (probe/smoke prompt sealing only; all activation evidence is the existing sealed caches) |

---

## ADDENDUM 1 — 2026-08-10 (Cyril): the composition endgame

The matrix's deliverable is not a per-technique ranking for its own sake.
Per Cyril's direct clarification: **the child does not need one technique
everywhere — different techniques may fold different components,
dynamically, and all that matters is that the composed child works.**
A9/A11-style routes are first-class candidates at ANY component, not
only the token interface.

Therefore the campaign's final wave is the **heterogeneous composition
run**: for each component, select the technique+level the matrix verdicts
name as best (pre-registered selection rule: highest free-run-vs-control
at the largest level that passes; ties broken by composed-terminal
margin), assemble the full-child recipe from those winners, and execute
it as one planned transaction — the dynamic fold planner already selects
operators per branch, so the recipe is a planner selection policy fed by
the matrix ledger. The composed child then takes the frozen review. The
matrix tells us which techniques work where; the composition run tests
whether the winners compose — and its per-component receipts inherit the
same first_divergent_metric chain so a composition failure localizes to
the component that broke it.

---

## ADDENDUM 2 — 2026-08-10: the calibration criterion, final form

Cell 0b's criterion moved twice, each fall forced by a measured wall and
each documented:
1. **Bit-identity of the A9 route artifact** (original): unreachable —
   the dtype-resolution v2 makes A9's sealed bf16-on-Pascal
   configuration unreproducible on any host that can read evidence.
2. **Behavioral token-match ≥0.90 vs the sealed A9 dossier** (first
   recalibration): unreachable in principle — on SM86 all eleven dtype
   roles reproduce A9's contract exactly, but the fp32 OMP candidate
   search is architecture-non-portable, so any recompilation is the
   same mechanism at a nearby point; token-matching two chaotic
   decoders measures decode divergence, not harness fidelity
   (NOTES-MX-C).
3. **Quality-band (final)**: the recompiled child's dossier is graded
   under the frozen V5 protocol (digest verified) and must land within
   ±2 useful-or-partial cases of the sealed A9 child's 8/15 at the same
   cap. The ±2 is measured, not chosen: it is the frontier re-score's
   own reviewer-sensitivity interval (6–9 around 8) on this exact
   15-case surface. VERDICT: PASS with band delta 0 — identical
   category on all fifteen cases, two responses byte-identical
   (NOTES-MX-D §2, sealed review 5b8e6ee3…, receipt 28564b71…).

The harness faithfully implements the mechanism; the campaign's cells
are calibrated.

---

## ADDENDUM 3 — 2026-08-10 (Cyril): graded outcomes, 0.90 is a goal not a gate

Per Cyril's direct instruction: 0.90 free-run is the aspirational "works
outright" threshold, not a dismissal line. Cell outcomes become GRADED:
- **strong** — free-run ≥0.90 vs control AND coherent smoke (works outright)
- **promising** — free-run ≥0.60 AND coherent smoke (the best-baseline
  band; cells 1-3 live here; these cells get DEEPER adaptive exploration,
  not pruning: more levels, technique hybrids, and priority in the
  composition recipe when they are the best available at their component)
- **weak** — free-run 0.20-0.60 or degraded smoke
- **dead** — free-run <0.20 or degenerate smoke
first_divergent_metric and all sealed fields are unchanged; only the
interpretation layer and the adaptive policy read the grade. The
composition endgame (Addendum 1) selects the best available grade per
component — a component whose best is "promising" still ships its winner.
Scope discipline: single-component cell numbers must never be quoted as
whole-model results; the composed child's own receipts are the only
whole-model claim.

---

## ADDENDUM 4 — 2026-08-10 (Cyril): composition modes, tested not assumed

Isolation cells prove each component's best technique against
PARENT-context inputs. A composed child's downstream components see
FOLDED upstream outputs instead — so isolation winners may not combine
directly. ONLY AFTER the per-component winners are proven, the
composition wave tests the modes as an experiment, not an assumption:

- **M-A direct assembly**: install every component's isolation-solved
  winner as-is. Cheapest; measures how much the parent-frame/child-frame
  mismatch costs.
- **M-B sequential conditioned composition**: fold components in
  topological order; each component uses ITS winner technique but
  re-solves against the actually-folded upstream child state (layer 2's
  winner refits on folded layer 1's outputs, and so on downstream). The
  closed-loop machinery already implements sequence-conditioned solving;
  the new element is heterogeneous per-component techniques inside it.
- **M-C (if A and B disagree per component)**: hybrid — components whose
  M-A install survives keep it; components that degrade under M-A are
  re-solved M-B style in order.

All three carry the standard cell receipts (per-component
first_divergent_metric) so mode differences localize. The mode
comparison is itself a sealed campaign result: "winners combine
directly" and "winners need conditioning" are both findings.

---

## ADDENDUM 5 — 2026-08-10: a level claim is SITE-SCOPED

Wave 3 was the first wave to produce a technique×component whose rungs
disagree across sites, and it exposed that §5 rule 2's number and the
renderer's implementation of it are two different tests
(`NOTES-MX-G` §4.6). `C07.T1` at M8 is 3/3 at M16 and 2/3 at M8, so the
literal clause ("passes at ℓ and every shallower level at ≥2 of its 3
sites") says `works to M8` while `render_matrix_ledger.py:298`'s
`good >= min(2, n) and good == len(rows_at)` says `works to M16`. The
same wave produced `C06.T1.X4.L12` at 0.977 and `C07.T1.X4.L12` at
0.918 — passes that hold at **L12 only**, against 0.871/0.889 and
0.746 at the shallower sites of the same component, technique and
level. §5 rule 6 already forbids a result from generalizing across
sites; nothing in the rendered artifact said so on its face.

This addendum does not change what passes. It changes what a pass may
be **called**, and it makes the renderer say it.

**Rule 2 is superseded by rule 2-S, registered here.** A level claim is
site-scoped:

> **2-S.** A technique+component is marked `works at level ℓ at sites
> S` where **S is exactly the set of sites with a sealed PASSING receipt
> at ℓ and at every shallower level attempted at that site**. The claim
> is never written without its site set. A technique+component with a
> pass at one site and a fail at another at the same level is neither
> `works` nor `non-monotone` in general — it works at the sites in S and
> is closed or untested elsewhere, and the two statements are printed
> together. The ≥2-of-3 count survives as a *breadth* annotation
> (`3/3`, `2/3`, `1/3`) reported beside S, never as the licence for an
> unqualified claim.

Three consequences, stated so they cannot be re-litigated later:

1. **The C07/T1 ambiguity dissolves rather than being decided.** The
   old question — does 2 of 3 license `works to M8`? — asked for a
   whole-component verdict that rule 2-S does not issue. The sealed
   answer is now `C07/T1 works to M8 at L08 and L12 (2/3); at L02 it
   fails free-run at M8 (0.793) and works to M16 only`. Both readings
   of the old clause are recoverable from that sentence; neither is
   needed.
2. **No claim in this campaign extends past its sealed sites.** In
   particular `C05.T3.M16` is a claim **at L05 only** (0.914 there,
   0.699/0.646 at L00/L13), and `C06.T1.X4` / `C07.T1.X4` are claims
   **at L12 only**. The composition wave installs winners at their
   proven sites and passes every other site through unfolded; an
   extrapolation to an unsealed site is a new cell, not a deduction.
3. **The rendered ledger is the campaign artifact, so it carries the
   scope.** `render_matrix_ledger.py` prints `PASS@<level>@<sites>` in
   the matrix, a per-site rung breakdown in the ladders, and a verdict
   of the form `works to <level> at <sites> (<good>/<attempted> sites)`.
   Receipts and the ledger are unchanged — the site was always a sealed
   field; only the *aggregation* was site-blind.

Earlier renders of `LEDGER.md` that print an unqualified `works to ℓ`
are superseded by the site-scoped render, not amended in place; the
ledger rows they were produced from are untouched and re-render under
2-S without a re-run.


---

## ADDENDUM 6 — 2026-08-11 (Cyril): behaviour-first grading, and a real 4x bar

Wave 4 settled two questions the campaign had been answering wrong, and
Cyril's ruling reopens phase 1 on both.

**Token match is dead as a gate, as a grade, and as an optimization
target.** Tokens cannot match through a fold by construction, and the
campaign now has its own proof: `COMP.MA.RSTO` free-ran at **0.551**
against the copy_exact control and scored **12 of 15** useful-or-partial
under the frozen review — *the same score its own parent got in the same
process on the same prompts* (`NOTES-MX-H` §6). §5 rule 1 read 0.551 and
said `fail`; the campaign's own authoritative behavioural judge read the
text and said parent-level. Both are sealed and both are about the same
child. A metric that calls a parent-band child dead is not a gate.

**Marginal-level results do not prove folding.** The M16-class winners of
waves 1–3 prove that the *mechanics* work — a component can be replaced
and the model survives. They prove nothing about compression: at M16 a
technique whose stored form is a basis plus a solved factor stores
**more** than the dense tensor it replaced (`NOTES-MX-E` §4.1), and
`C13.T3.M16.L05`'s sealed 3.28× is the campaign's own worked example.
A fold that costs storage is not a fold.

### 6.1 Behaviour-first grading (supersedes ADDENDUM 3's grade ladder)

The **primary grade of a cell is the frozen 15-prompt review of the whole
model with that cell's folded component installed and every other
component copy_exact.** The review surface is the one wave 4 pinned: the
digest-pinned V5 protocol `e5e37768…`, the prompt pack's exact byte-form
`0ee6bf14…`, greedy decoding at the 1,024-token cap, and the parent
decoded on the same prompts **in the same process** as the control.
Grading is done on the verbatim text under the protocol's own per-case
anchors, and is sealed as a separate instance by a review script — a
runner still may not certify its own behaviour.

Reference bands, both measured on this exact surface:

| reference | useful-or-partial |
|---|---:|
| the parent itself, same process, same prompts | **12 / 15** |
| the sealed A9 child (ADDENDUM 2's band) | **8 / 15** |

Verdicts, in descending order:

- **`alive-parent-band`** — within 2 cases of the parent's own 12/15
  (i.e. ≥10/15), no degenerate output, coherence smoke coherent, and the
  emit-zero floor passed. The ±2 is not chosen: it is the same measured
  reviewer-sensitivity interval ADDENDUM 2 registered for this surface.
- **`alive-degraded`** — ≥8/15, i.e. at or above the A9 band, but outside
  the parent's ±2.
- **`impaired`** — <8/15 but the responses are still coherent, on-task
  language: the model lost capability without losing its mind.
- **`dead`** — incoherent output, emit-zero floor failed, or degenerate
  looping/collapse in the smoke or the review, at any score.
- **`cannot-express-4x`** — see §6.2. Distinct from `dead`: nothing was
  measured about the model, because the technique cannot state the
  component that small.

`token_match_vs_control`, `mean_lcp`, `vs_copy_exact_control_delta`, the
boundary write/macrocell metrics and the S8 band all remain sealed and
all remain **diagnostic columns for manual reading**. None of them is an
eligibility criterion for anything, and no cell is pruned, skipped or
deprioritized on any of them. In particular, the MLP cells `C11`/`C12`/
`C13` at token-match 0.31–0.59 were never behaviour-reviewed before this
addendum, and their exclusion by the old gate is void.

`outcome.status` and `first_divergent_metric` are **not** redefined:
every receipt sealed before this addendum re-verifies its seal and
re-computes the same chain it always did. The behaviour verdict rides in
its own sealed block (`addendum6`) on a review instance, and the renderer
leads with it.

### 6.2 The X4 proof bar

A claim that technique **T** *folds* component **C** requires, at
minimum, that the replacement **stores no more than 0.25× that
component's own parent parameter count**, verified from the actual
tensor shapes recorded in the receipt.

The stored count covers **every tensor the child must keep** to evaluate
the component: factors, bases, routes, expansion matrices, retained
coefficients, per-head blocks and any auxiliary table — the coverage rule
derived in commit `8d59c4d2` applies with full force, and its
`installed`/`owned` derivation is unchanged. A technique that installs a
parent-shaped tensor materialized from a smaller stored form declares
that smaller form and nothing else; a technique that keeps a tensor it
did not shrink still stores it.

Levels are therefore extended with a **storage-defined family**, whose
name is a bound on what is stored rather than a fraction of a width:

| level | bound |
|---|---|
| `S2` | stored ≤ 0.5× the component's parent parameters |
| `S4` | stored ≤ 0.25× — **the X4 proof bar** |
| `S8` | stored ≤ 0.125× |

The marginal family (`M16`/`M8`/`M4`/`M2`) and the layout-derived `X4`
are untouched and keep their meaning; `X4` remains "the width the
canonical exact-4x layout assigns this component", which is a *width*
and, as waves 1–3 measured, frequently not a 4× *storage* claim at all.
Where the two disagree, `S4` is the one that supports a folding claim.

The bound is enforced inside the runner, from the receipt's own
`solve.stored_parameter_count` and `solve.parent_parameter_count`, and
sealed as `solve.storage_bound_ratio` / `solve.storage_bound_satisfied`.
A technique×component that cannot express the component at or under the
bound at any admissible width returns the verdict
**`cannot-express-4x`**, which is a statement about the technique's
storage algebra and never about the model's behaviour.

### 6.3 The level ladder, and mixed levels for assembly

A folding claim for the whole model does not require every component at
the same level. The endgame target is global, so the campaign builds a
**per-component level ladder**:

- where `S4` is `alive-parent-band` under some technique, **probe `S8`**
  at that technique and site;
- where `S4` is `dead` under **every** applicable technique, seal the
  component's **maximum surviving level** (`S2`, else the deepest
  marginal level that survives) so the assembly wave knows what that
  component actually costs;
- a component whose best surviving level is `S2` and a component that
  survives `S8` are both legitimate members of one assembly, provided the
  assembled child hits the global target.

The ladder is the wave's deliverable, and the survivors are the input to
the assembly wave (compress → test → combine the best → test again).

### 6.4 Protocol, per cell

One component, one technique, one level, one site: solve it, install it
in the **otherwise-parent** model, run the frozen review and the
coherence smoke on the **whole** model, seal, append. Never two
components in one cell — §5 rule 6 is unchanged and ADDENDUM 4's
composition modes remain the only path to a multi-component claim.

Rule 2-S (ADDENDUM 5) is unchanged and applies to the storage levels
exactly as it applies to the marginal ones: a level claim is site-scoped,
always printed with its site set.

---

## ADDENDUM 7 — 2026-08-11: the site registry opens to full model depth

### 7.1 Why

Wave 6 climbed a per-layer ladder and it stopped after three rungs. Not
because the model refused — `NOTES-MX-J` §5 records the actual reason:

> The brief asked for `C13 T8` across L00–L13. **That is not expressible in
> this campaign**: `matrix_cell_receipt.COMPONENTS` registers `C13`, `C11`,
> `C05`, `C12` at sites `(0, 5, 13)` and `C06`–`C09` at `(2, 8, 12)`, and
> `_validate_cell` enforces it.

Three sites per body component was the right call in wave 1, when ~300 cells
had to fit inside a budget and the sites were chosen for what they would teach:
L0/L13 the measured operator-damage extremes (S3), L5 a quiet mid conv layer,
L2/L8/L12 the largest attention whole-layer gains. It stopped being a sampling
policy and became a ceiling the moment the campaign started assembling children.
`NOTES-MX-J` §5.1 put a number on the ceiling:

> Even a hypothetical child that folded all five alive-parent-band components at
> all three of their registered sites at a true quarter would store 23,592,960
> parent parameters → 5,884,002 stored → net **0.9229×** … eleven of the
> fourteen layers have no registered site at all.

**Eleven of fourteen layers unreachable is a statement about the registry, not
about the technique surface or the model.** ADDENDUM 7 removes it.

### 7.2 What changed

**One thing: the site tuples.** `COMPONENTS` now registers every
*architecturally valid* layer per component, read off the parent's own
`config.json` `layer_types` and re-checked against the shipped
`model.safetensors` key set:

| block kind | layers | count |
|---|---|---:|
| `short_conv` (`conv.in_proj`, `conv.conv`, `conv.out_proj`) | 0, 1, 3, 5, 7, 9, 11, 13 | 8 |
| `full_attention` (`self_attn.{q,k,v,out}_proj`) | 2, 4, 6, 8, 10, 12 | 6 |
| `feed_forward` SwiGLU (`w1`, `w2`, `w3`) | **all fourteen** | 14 |
| `operator_norm` / `ffn_norm` | all fourteen (+ the global `embedding_norm` at `FINAL`) | 14 + 1 |

so:

| component | v1 sites | v2 sites |
|---|---|---|
| C03 `conv_in_proj`, C04 `conv_kernel`, C05 `conv_out_proj` | 0, 5, 13 | 0, 1, 3, 5, 7, 9, 11, 13 |
| C06 `attn_q`, C07 `attn_k`, C08 `attn_v`, C09 `attn_o` | 2, 8, 12 | 2, 4, 6, 8, 10, 12 |
| C10 `attn_per_head` | 2, 8 | 2, 4, 6, 8, 10, 12 |
| C11 `mlp_gate`, C12 `mlp_up`, C13 `mlp_down` | 0, 5, 13 | 0 … 13 |
| C14 `norms` | 0, FINAL | 0 … 13, FINAL |
| C01, C02, C15 | global | global (unchanged) |

The widening is to the architecture and **not** to `range(14)`: a short_conv
component is still refused at an attention layer, an attention component is
still refused at a short_conv layer, and nothing reaches a fifteenth layer.
`tests/test_matrix_addendum7.py` asserts all three, because a blanket widening
would pass a "C13 at L07 is now legal" test and fail these.

**Nothing else moves.** The mandatory field set, `first_divergent_metric`'s
seven links, `campaign_pass`, the degeneracy clause, the storage levels, the
composition block, the canonical JSON and the seal are byte-for-byte what they
were. `test_a_v1_receipt_still_seals_verifies_and_walks_the_same_chain` pins
that: two bodies differing only in their schema string walk the same chain,
resolve the same outcome and are equal field-for-field.

### 7.3 The schema is versioned, and the widening does not reach backwards

`SCHEMA_V1 = "…matrix-cell-receipt-v1"` and
`SCHEMA_V2 = "…matrix-cell-receipt-v2"` are both live.
`RECEIPT_SCHEMA_VERSIONS` maps each to an integer, `validate_receipt` reads a
body's version **off the body**, and `CellId.parse(..., schema_version=)`
enforces that version's registry.

- Every receipt sealed in waves 1–6 declares v1 and is validated against v1's
  narrow registry **for ever**. Re-verifying any of the 348 rows recomputes
  exactly the clauses it was sealed under. There is no migration, no rewrite and
  no retro-validation.
- A body that declares v1 and names a v2-only site is **refused**. A wave-1 row
  cannot acquire a meaning it was not sealed with, in either direction.
- New work defaults to v2 — recipes, runners, and the renderer's enumeration of
  pending positions.

The ledger gains one column, `receipt_schema_version`, and `LEDGER_SCHEMA`
deliberately does **not** move: waves 1–6's rows stay readable by the same
reader, and a row without the key is a v1 row, which is the same statement the
key makes explicitly for every row appended from wave 7 on.

The 21 authenticated site evidence caches on the run host are named by
`evidence_cache_name`, which picks the lowest component id sharing an endpoint
set among the components *legal at that site* — so widening the legal set is
exactly the kind of change that could silently rename a cache and invalidate it.
It does not: every component v2 adds at an already-registered site has a higher
id than the canonical one already there. `test_every_v1_site_keeps_its_evidence_cache_name`
checks all 34 of them by literal rather than by argument.

### 7.4 What the ceiling becomes

Registered-site mass at a true `S4` (stored ≤ 0.25× the component's parent
parameters), against the parent's 229,693,184:

| registry | foldable parent mass | stored at S4 | net |
|---|---:|---:|---:|
| v1, the five alive-parent-band components at their 3 registered sites | 23,592,960 | 5,898,240 | **0.9230×** |
| v1, every registered site of every body component | 45,625,344 | 11,406,336 | 0.8510× |
| **v2, the five alive-parent-band components at every valid site** | **91,226,112** | 22,806,528 | **0.7021×** |
| v2, the four that compose (C05 dropped, `NOTES-MX-J` §2) at every valid site | 82,837,504 | 20,709,376 | 0.7295× |
| **v2, every body component at every valid site** | **162,583,552** | 40,645,888 | **0.4691×** |

(C10 `attn_per_head` is excluded from the mass everywhere: it owns the same four
tensors C06–C09 do, and counting it again would count them twice.
`NOTES-MX-J` §5.1's 5,884,002 is the *measured* stored count of the five
realized cells; 5,898,240 above is the same mass at the exact S4 bound.)

The floor under all five is the same 67,108,864-parameter tied embedding table,
which no body component in the matrix touches and which is 29.2% of the parent
on its own. **A whole-model 4× is not reachable without C01/C02/C15**, and
ADDENDUM 7 does not change that; what it changes is that the body is no longer
capped at 0.92×.

### 7.5 What v2 does *not* license

Rule 2-S (ADDENDUM 5) is unchanged and is now load-bearing: **a level claim is
site-scoped and is always printed with its site set.** A site being *registered*
is not evidence about that site. Wave 6 measured C13 at L00 as hostile
(0.422844 install drift, 5/15) and at L13 as impaired (7/15) while L05 held the
parent band — three sites of one component, three different answers — so no
result at one layer may be extrapolated to its neighbour. Every rung of a v2
ladder is a measurement or it is nothing.

§5 rule 6 is unchanged: one component per cell, and ADDENDUM 4's composition
modes remain the only path to a multi-component claim.
