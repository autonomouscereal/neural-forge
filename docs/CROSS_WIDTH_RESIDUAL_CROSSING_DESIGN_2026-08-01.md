# Cross-Width Residual Crossing Design (Rectangular Boundary Charts)

Updated: 2026-08-01
Status: design only.  No code in this document is implemented.  A
deep-implementer builds from this document; the staged proof plan defines the
first acceptable evidence.

## Problem

The family-neutral per-branch operators fit in one shared residual coordinate
system: the promotion preamble fails closed when parent and child hidden
widths differ (`src/neural_forge/dynamic_causal_operator_promotion.py:220-231`
— the error text itself says "no boundary-map operator is registered for this
crossing"), and the planner's contract returns `None` on width mismatch with
the comment "cross-width folding still needs its own boundary-map operator"
(`src/neural_forge/dynamic_fold_planner.py:759-763`).  Folding an H1024 parent
into an H816 cross-family child therefore has no executable path.  This
document designs the family-neutral boundary-map operator: per-boundary
rectangular encoder/decoder charts, chart-transition absorption into ordinary
physical weights, chart selection from sealed evidence, and the exact
fail-closed envelope.

A decisive existing fact shapes the whole design: **the branch kernels are
already width-agnostic**.  The attention kernel's Q/K/V fits are plain ridge
maps from child rows of any width to desired head targets
(`src/neural_forge/hf_causal_gqa_attention_fold.py:605-613`; the replay only
checks self-consistency of the fitted width, `:432-433`), and
`fit_functional_swiglu` constrains only row counts, not input/target widths
(`src/neural_forge/hf_closed_form_causal_circuit_fold.py:1199-1223`).  The
width-equality requirement lives exclusively in the promotion preamble and the
planner contract.  Cross-width folding is therefore a *target-construction*
problem — build the right chart-transformed endpoint targets — not a new
solver problem.

## Prior art this design builds on, not duplicates

- **Moving-gauge boundary charts (LFM2 closed loop).**  The reclosure lane
  already fits cycle-coupled per-boundary encoder/decoder pairs from sealed
  FIT evidence: the encoder is transported by orthogonal Procrustes, the
  decoder is a deterministic ridge with a regularized-reciprocal
  cycle-consistency anchor, and interface-locked boundaries are restricted to
  orthogonal rotation
  (`src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py:991-1056`).
  The chart-transition residual is contract-named
  `encoded_parent_endpoint_minus_actual_child_pre_state`, with the identity
  skip path explicitly accounted for
  (`hf_lfm2_moving_gauge_residual_reclosure.py:976-979`).  Moving-gauge v3 is
  a registered body-reclosure mode with sealed rank-factored writer artifacts
  (`src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py:171-180`,
  `:1585-1660`).
- **Balanced circuit basis.**  Family-neutral balanced residual coordinates
  from three evidence lanes — residual-facing weights balanced by semantic
  role, cached residual snapshots as an empirical controllability Gram, and
  downstream consumers as observability — solved by regularized balanced POD
  and canonicalized into one orthonormal `B[source, target]`, transporting
  consumers as `W @ B` and producers as `B.T @ W`
  (`src/neural_forge/dynamic_balanced_circuit_basis.py:1-22`).  Rank
  insufficiency is a typed error (`dynamic_balanced_circuit_basis.py:87-89`).
- **The A0025 lesson.**  Independently fitted components with small local
  errors landed in incompatible evolving residual coordinates; complete
  endpoint relative RMSE reached 4.99-14.37 while local action RMSE was
  0.005-0.05.  The audit's prescription is exactly this design's shape:
  deterministic moving charts at every internal boundary with distinct
  encoder/decoder maps and cycle consistency; whole-macrocell transport with
  the transition defect `E_(l+1) D_l - I` absorbed into ordinary physical
  writers; one forward analytic sweep plus one reverse reconciliation sweep
  (`reports/92b2e091-a0025-live-prefix/CAUSAL_GAP_AUDIT.md:26-34`, `:67-98`;
  `docs/CURRENT_FOLDING_GOAL.md:78-81`).
- **Active mechanism.**  Boundary-local encoder/decoder charts are mechanism
  item 4 of the current goal, with an explicit ban on collapsing them into
  one global chart (`docs/CURRENT_FOLDING_GOAL.md:152-158`).
- **Distinct prior art, not to be confused.**  The LFM2 variable-width fold
  varies per-layer *FFN* width while the residual width is unchanged
  (`src/neural_forge/hf_lfm2_variable_width_fold.py:1-19`); heterogeneous
  ShortConv contracts vary *internal channel* width against a constant
  external residual width (`src/neural_forge/hf_lfm2_canonical.py:111-156`).
  Neither crosses the residual width; this design does.

## Chart algebra

Notation: row vectors; parent hidden width `Hp`, child hidden width `Hc`
(v1 scope: `Hc <= Hp`).  Layers `l = 0..N-1`.  Charts live at every
*residual-write boundary*, i.e. per branch, matching the evidence granularity
(`attention_delta`, `post_attention_residual`, `mlp_delta` per layer —
`src/neural_forge/hf_causal_circuit_evidence.py:219-231`).  Boundary `b` has
an encoder `E_b in R^{Hp x Hc}` (parent state to child state, `hhat = h E_b`)
and a decoder `D_b in R^{Hc x Hp}` (child state to parent state estimate).

### Per-branch chart-transformed endpoint target

Child branch at boundary `b` (input) writing into boundary `b+1` (output).
Cached parent states `h_b`, parent branch delta `Delta`, and the *actual*
sequentially recaptured child state `hhat_b_actual`.  The branch's fit target
is

```text
Y_t = h_{b+1,t} E_{b+1} - hhat_b_actual,t
    = Delta_t E_{b+1} + h_b,t (E_{b+1} - E_b) - e_b,t
where  e_b,t = hhat_b_actual,t - h_b,t E_b     (accumulated upstream error)
```

One equation, three responsibilities, all absorbed by the ordinary branch
writer with **no runtime adapter**:

1. `Delta E_{b+1}`: the parent's functional write transported into the
   downstream chart;
2. `h_b (E_{b+1} - E_b)`: the chart transition applied to the through-state.
   The child skip path is identity, so when adjacent charts differ the layer's
   physical writers must carry the difference — this is chart-transition
   absorption, the same contract as the moving-gauge formula
   `encoded_parent_endpoint_minus_actual_child_pre_state` with the identity
   skip path accounted for
   (`hf_lfm2_moving_gauge_residual_reclosure.py:976-979`) and the A0025
   requirement that `E_(l+1) D_l - I` be absorbed into ordinary physical
   writers (`reports/92b2e091-a0025-live-prefix/CAUSAL_GAP_AUDIT.md:84-88`);
3. `-e_b`: correction of accumulated upstream error, which is what makes the
   fit *sequential* rather than parent-coordinate-local — the direct repair of
   the A0025 failure mode.

The sequential form `Y = h_{b+1} E_{b+1} - hhat_b_actual` is the one to
implement; the expansion above exists so each responsibility can be measured
separately in diagnostics.

### Compatibility conditions between adjacent charts

Three measured defects, each with a policy bound:

```text
cycle defect        c_b   = || h_b - (h_b E_b) D_b ||_F / || h_b ||_F        (fit rows)
transition defect   T_b   = E_{b+1}^T D_b^T - I_Hc   (child-chart mismatch, per A0025)
absorption demand   A_b   = D_b (E_{b+1} - E_b)      (in R^{Hc x Hc})
norm commutation    nu_b  = row-variance of ||h_b E_b|| / ||h_b||            (see below)
```

`A_b` is the linear map the branch must additionally represent on top of its
functional job, because `h_b (E_{b+1} - E_b) ~= hhat_b D_b (E_{b+1} - E_b)`
up to decode error.  Its representability is bounded by the writer's rank:
the attention branch writes through `W_o` (rank at most
`min(Hc, query_heads * head_dim)`), the MLP branch through the down
projection (rank at most `min(Hc, intermediate_size)`).  Fail closed when the
required energy of `A_b` outside the writer's representable rank exceeds the
policy bound, or when `c_b` exceeds its bound (the decode-error `eta` makes
absorption itself unreliable).

**Norm commutation.**  RMSNorm does not commute with a rectangular chart:
`rmsnorm(h E) != rmsnorm(h) E` in general.  For orthonormal `E` (chosen
below), `||h E|| <= ||h||`, and the child RMS rescales by
`(||h E|| / ||h||) * sqrt(Hp / Hc)`.  When the captured-energy fraction is
near-constant across rows, this is a single scalar absorbed exactly by the
fitted RMSNorm diagonal (`fit_rmsnorm_diagonal` fits target-coordinate gains
directly from rows — `src/neural_forge/hf_closed_form_causal_circuit_fold.py:
793-801`).  The row-wise fluctuation `nu_b` is the part no diagonal can
absorb; it is measured on fit rows and bounded, and its residual lands in the
branch endpoint fit like any other model error.

### Interface-locked boundaries

Boundary 0 (embedding output) and the terminal boundary (final-norm input)
are owned by the token/head interface, not by chart selection.  Their charts
are fixed by the token bridge and head construction, or restricted to
orthogonal rotation exactly as the deployed reclosure does for interface-locked
evidence (`hf_lfm2_moving_gauge_residual_reclosure.py:1054-1056`).  Chart
selection never modifies an interface-locked boundary; it fails closed if
asked to.

## Chart selection

### Objective and evidence

Per boundary `b`, the encoder `E_b` is the balanced controllability /
observability basis of `dynamic_balanced_circuit_basis`, generalized off the
LFM2 lane (its inputs are already family-neutral):

- **Controllability**: the empirical Gram of cached parent residual rows at
  boundary `b`, fit split only (`block_input` / `post_attention_residual` /
  `block_output` from `hf_causal_circuit_evidence.py:219-231`, or
  `residual.pre` / `residual.post_operator` / `residual.post_layer` from
  `hf_lfm2_circuit_trace_cache.py:2811-2820`).
- **Observability**: the residual-facing weights of every *downstream*
  consumer — attention Q/K/V input columns, MLP gate/up input columns, norms,
  and the output head — transported per the module's `W @ B` / `B.T @ W`
  convention (`dynamic_balanced_circuit_basis.py:1-22`).
- Solve: regularized balanced POD; canonical orthonormal `E_b` (so
  `E_b^T E_b = I_Hc`), which is what makes the norm-commutation scalar
  well-behaved above.  Not raw activation PCA, not raw weight SVD — the 100x
  failure record's sources already established both as inadequate objectives
  (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:32-42`).

The decoder `D_b` is the deterministic cycle-anchored ridge of the deployed
moving-gauge fit: ridge from encoded rows back to parent rows with a
regularized-reciprocal anchor as the closed-form cycle-consistency term
(`hf_lfm2_moving_gauge_residual_reclosure.py:996-1034`).

### Solve order: greedy forward + one reverse sweep (not global)

1. Fit all `E_b` independently from *parent-side* sealed evidence (they do
   not depend on child state).
2. Compile child layers front-to-back.  Each branch fit consumes the
   sequential target `Y = h_{b+1} E_{b+1} - hhat_b_actual` with
   `hhat_b_actual` recaptured from the live child after every accepted
   install — the sequential-recapture discipline A0025 proved necessary.
3. After the forward pass, run one reverse reclosure sweep re-solving
   *writers only* (charts frozen) against terminal-anchored targets, as the
   audit prescribes (`CAUSAL_GAP_AUDIT.md:85-89`) and as moving-gauge v3
   already implements structurally for the LFM2 lane.

**Rejected: joint global optimization** of all charts and writers.  It is a
large nonconvex bilinear program with no closed form, violating the
deterministic-solve doctrine; and a *single* shared global chart is both
banned (`docs/CURRENT_FOLDING_GOAL.md:152-158`) and empirically refuted
(A0012/A0025 evidence chain, `docs/CURRENT_FOLDING_GOAL.md:77-81`,
`:109-117`).  Greedy-plus-reverse is the boring, already-deployed pattern.

## RoPE and head-dim constraints under width change

Width change and head geometry are orthogonal, with one real coupling:

- RoPE acts per head after projection; `head_dim`, head counts, theta, and
  pair selection are governed by the existing kernel's rules regardless of
  `Hc`: even `head_dim` (`hf_causal_gqa_attention_fold.py:147-148`),
  contraction-only heads (`:113-127`), pair-energy RoPE contraction with no
  expansion (`:369-389`), default single-theta RoPE transport only
  (`dynamic_causal_operator_promotion.py:199-217`).
- The Q/K/V ridge maps `R^{Hc} -> R^{heads*head_dim}`; `Hc` only bounds the
  *rank* of these maps.  Policy requirement: `min(Hc, kv_width)` must exceed
  the effective rank demanded by the desired K/V targets on fit rows, else
  the fit is silently rank-starved — measured via the retained-energy
  diagnostics the kernel already emits (`:685-702`) plus a new explicit
  effective-rank check in the operator report.
- `attention_scaling` follows the *target* head_dim
  (`dynamic_causal_operator_promotion.py:500-505`); nothing width-related
  changes it.
- The real coupling is the absorption-demand rank bound of `A_b` against
  `query_heads * head_dim` (attention writer) described above: a child with
  narrow attention (`q_width << Hc`) may be unable to absorb a large chart
  transition at that boundary.  This is a plannable, checkable constraint —
  it belongs in the planner contract, not in a runtime surprise.

## API shape

New module `src/neural_forge/dynamic_residual_boundary_charts.py`:

```python
@dataclass(frozen=True)
class ResidualBoundaryChart:
    boundary_id: str            # e.g. "layer07.attention_in"
    encoder: Tensor             # [Hp, Hc], orthonormal columns
    decoder: Tensor             # [Hc, Hp]
    interface_locked: bool
    report: Mapping[str, Any]   # sealed: cycle/transition/nu diagnostics,
                                # evidence shas, policy sha, tensor shas

@dataclass(frozen=True)
class ResidualBoundaryChartSet:
    parent_evidence_sha256: str
    child_topology_sha256: str
    charts: tuple[ResidualBoundaryChart, ...]   # one per residual-write boundary
    report: Mapping[str, Any]                   # self-sealed canonical JSON

def compile_residual_boundary_chart_set(evidence, child_layout, policy)
    -> ResidualBoundaryChartSet   # closed form, fit rows only
```

New operator `CrossWidthCausalBranchOperator` wrapping the existing two
branch operators (`dynamic_causal_operator_promotion.py:434-602`, `:605-...`):

- `_bind` replaces `_require_shared_residual_width` with chart-set
  authentication (both boundary charts present, sealed, lineage-bound to the
  same evidence and child topology);
- computes `Y` from sealed parent rows, the bound charts, and the live-child
  recapture; feeds `compile_hf_causal_gqa_attention_fold` /
  `fit_functional_swiglu` / `fit_rmsnorm_diagonal` **unchanged** (they are
  width-agnostic, see Problem section);
- extends the transaction report with chart lineage, `c_b`, `T_b` energy,
  `A_b` absorbed/unabsorbed energy split, and `nu_b`;
- keeps the identical snapshot/commit/rollback transaction
  (`dynamic_causal_operator_promotion.py:298-362`).

Planner change: `_family_neutral_causal_layer_contract` keeps every clause
except width equality (`dynamic_fold_planner.py:759-763`); when widths differ
it emits the same branch method ids with
`configuration["boundary_chart_required"] = True`, and the executor refuses
to run such an action without an authenticated chart set.  When no chart set
can be compiled (rank failure), the planner keeps the existing gap — the
crossing stays fail-closed exactly as today.

## Fail-closed conditions

Each aborts before any child mutation:

1. Width *expansion* (`Hc > Hp`): out of the v1 envelope.  The balanced basis
   is a contraction construction; expansion needs its own design and evidence.
2. Missing, unauthenticated, or lineage-mismatched chart for either boundary
   of an action (wrong evidence sha, wrong child topology sha, tampered
   tensor fingerprints).
3. Balanced-basis rank failure: the controllable/observable intersection
   cannot support `Hc` (`dynamic_balanced_circuit_basis.py:87-89` is already
   the typed error).
4. Cycle defect `c_b`, norm-commutation fluctuation `nu_b`, or the
   unabsorbable part of `A_b` above their policy bounds.
5. A chart-selection request touching an interface-locked boundary with
   anything other than the interface-fixed or orthogonal-rotation chart.
6. Sequential recapture unavailable (the operator must not silently fall back
   to parent-coordinate inputs — that is precisely A0025).
7. Branch validation endpoint relative RMSE at or above the no-branch
   baseline (zero-write versus `Y`), after BF16 materialization replay.
8. Fit/validation/selection partition overlap; non-finite tensors;
   condition-cap exhaustion — identical policy to the branch operators and
   the v4/v5 conv lane.

## Persistence

No database.  Sealed canonical-JSON artifacts with content-addressed tensor
fingerprints, following the moving-gauge factor-writer precedent
(`hf_lfm2_dynamic_closed_loop_adapter.py:1585-1660`).  New schema
identifiers:

```text
neural-forge-residual-boundary-chart-v1      (one chart: boundary id, shapes,
    encoder/decoder sha256, cycle/transition/nu diagnostics, evidence lineage)
neural-forge-residual-boundary-chart-set-v1  (ordered charts, parent evidence
    sha, child topology sha, policy sha, interface-locked map, set self-seal)
neural-forge-cross-width-branch-fold-v1      (per-action report: chart shas,
    Y-construction term energy split, absorption diagnostics, kernel report
    shas, BF16 replay deltas)
```

A resumed compile must be byte-identical to a cold compile, matching the
existing sealed-artifact contract
(`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:271-275`).

## Staged proof plan

1. **Toy exact-subspace control.**  Synthetic 2-layer causal decoder,
   `Hp = 32 -> Hc = 24`, parent states constructed to lie exactly in a 24-dim
   subspace: charts must recover near-zero endpoint error (up to
   conditioning).  Then full-rank stress, signed cancellation, and a
   deliberately mismatched adjacent-chart pair to verify transition
   absorption and the `A_b` rank fail-closed path.
2. **Same-family cross-width.**  Small all-attention parent to a same-family
   child at reduced width (e.g. H1024 -> H816) using only charts plus the
   unchanged branch kernels; teacher-forced checkpoint metrics at every
   boundary, then manual whole-response review.  This isolates the chart
   machinery from KIND translation.
3. **LFM2 lane.**  H1024 LFM2 parent to an H816 LFM2-shaped child with
   kind-preserved layers (the H816 geometry is already a known exact-budget
   option-bank shape, `docs/CURRENT_FOLDING_GOAL.md:44-52`), combined with
   the forward sweep and the reverse reclosure sweep.
4. **Cross-family cross-width.**  Qwen3-style parent to an LFM2-shaped child
   at different width, composing this design with the KIND-translation design
   (`docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md`).  This is
   the last stage because it stacks both new mechanisms; it must not be the
   first evidence for either.

Ordered implementation steps for the deep-implementer:

1. `src/neural_forge/dynamic_residual_boundary_charts.py`: chart + chart-set
   dataclasses, sealed reports, `compile_residual_boundary_chart_set` on top
   of `dynamic_balanced_circuit_basis` and the moving-gauge decoder solve.
2. Tests: `tests/test_dynamic_residual_boundary_charts.py` (stage-1 toys,
   authentication/tamper, fail-closed cases).
3. `CrossWidthCausalBranchOperator` in
   `src/neural_forge/dynamic_causal_operator_promotion.py` (or a sibling
   module importing its transaction helpers), plus registry wiring in
   `src/neural_forge/dynamic_fold_registry.py`.
4. Planner relaxation with `boundary_chart_required` in
   `src/neural_forge/dynamic_fold_planner.py:728-807`, gap preserved when no
   chart set exists.
5. Stage-2 runner plus sealed receipts; then stage-3 integration behind the
   existing LFM2 catalog runner.

## Trade-offs

- **Per-boundary charts with writer absorption (chosen)** versus one global
  residual chart (rejected): banned by the active mechanism and refuted by
  A0012/A0025 (`docs/CURRENT_FOLDING_GOAL.md:77-81`, `:152-158`).
- **Greedy forward + one reverse sweep (chosen)** versus global joint solve
  (rejected): no closed form; nondeterministic iteration counts; the deployed
  moving-gauge/closed-loop lane already embodies forward-plus-reverse.
- **Orthonormal balanced-POD encoders (chosen)** versus oblique or
  Frobenius-optimal shared subspaces (rejected): oblique encoders break the
  norm-commutation scalar; shared Frobenius subspaces distort heterogeneous
  function-bearing geometry per the 2026 tensor-decomposition finding cited
  in the failure record
  (`research/2026-07-18-lfm2-cpcf-100x-shortconv-failure.md:36-42`).
- **Sequential targets `h_(b+1) E_(b+1) - hhat_actual` (chosen)** versus
  parent-coordinate local targets (rejected): the latter is the measured
  A0025 failure; the former folds upstream-error correction into every solve
  at zero extra parameter cost.
- **Reusing the unchanged branch kernels (chosen)** versus new cross-width
  kernels (rejected): the kernels are already width-agnostic; duplicating
  them would fork audited solver code for no representational gain.

## Risks

- **Error accumulation across many boundaries remains the dominant risk.**
  Sequential targets correct upstream error only within each branch's
  representable class; rank-starved writers leave a growing un-absorbed
  residue that the reverse sweep may not fully reconcile.  This is the same
  physics that produced A0025's 14.37 endpoint error; the design reduces and
  measures it per boundary, but does not prove it away.  Mitigation beyond
  the sweeps: on-policy evidence extension rounds on the states the child
  actually visits (`docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md`).
- Chart quality is bounded by teacher-forced evidence coverage; charts can be
  excellent on cached rows and wrong on child-visited states.
- The absorption-demand rank constraint may make some desirable child
  topologies (narrow attention, wide residual) infeasible; that is surfaced
  at plan time rather than discovered at run time, but it constrains topology
  search.
- Compute and storage: per-boundary Grams over the sealed caches; bounded by
  the precomputed-Gram schemas the balanced-basis module already defines
  (`dynamic_balanced_circuit_basis.py:59-73`).  Runs belong on the AI server;
  no production service, credential, or parent snapshot is touched, and
  publication stays transactional.

## What this does NOT claim

- It does not claim cross-width folding preserves intelligence.  It provides
  the coordinate machinery; the verdict remains complete decoded responses
  under manual review (`docs/CURRENT_FOLDING_GOAL.md:22-27`).
- It does not claim the charts are information-preserving: `Hc < Hp` discards
  a measured subspace; the design chooses *which* subspace by balanced
  controllability/observability and reports what was discarded.
- It does not cover width expansion, non-RoPE position policies, biased
  projections, or sliding-window attention — all remain fail-closed exactly
  as in the current operators.
- It does not replace the token/head interface work: interface-locked
  boundaries are consumed here, not designed here.
- It does not authorize any gradient, optimizer, distillation, pruning,
  quantization, parameter sharing, sidecar, or unsealed parent resampling.
