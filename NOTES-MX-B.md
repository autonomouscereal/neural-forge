# NOTES-MX-B — matrix-campaign TECHNIQUE lane

Date: 2026-08-10. Branch `codex/lrc-blank-compression`, base HEAD `dee1c952`.
Design of record: `docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md`.

Three new modules, four techniques, all fold-only. Nothing in the frozen
collector, the harness, `matrix_technique_registry.py`, `matrix_cell_receipt.py`
or any stage-1 production file was edited. No git add/commit.

| file | technique |
|---|---|
| `src/neural_forge/matrix_technique_body_routes.py` | **T5** sparse residual-OMP routes, generalized to body components |
| `src/neural_forge/matrix_technique_relation_matching.py` | **T7** attention-relation matching + **T8** Gram/relation matching + the shared two-sided Stein primitive |
| `src/neural_forge/matrix_technique_lowrank_writers.py` | **T4** factored (low-rank) writers |

| test file | count |
|---|---|
| `tests/test_matrix_technique_body_routes.py` | 25 |
| `tests/test_matrix_technique_relation_matching.py` | 24 |
| `tests/test_matrix_technique_lowrank_writers.py` | 17 |
| `tests/test_matrix_technique_registry_adapters.py` | 14 |

```
.venv/Scripts/python.exe -m pytest tests/test_matrix_technique_body_routes.py \
  tests/test_matrix_technique_relation_matching.py \
  tests/test_matrix_technique_lowrank_writers.py \
  tests/test_matrix_technique_registry_adapters.py \
  tests/test_matrix_technique_registry.py -q
# 101 passed in 14.26s   (the last file is the harness lane's own suite)

.venv/Scripts/python.exe -m ruff check src/neural_forge/matrix_technique_*.py \
  tests/test_matrix_technique_*.py
# All checks passed!
```

---

## 1. T5 — what transplants from the token interface, and what does not

### 1.1 The A9/A11 mechanism, restated

Read from `hf_lfm2_sparse_semantic_codebook.py`
(`discover_lfm2_residual_omp_semantic_candidates`, `_omp_append_orthonormal_atom`,
`_ridge_solve_batched`) and `hf_lfm2_functional_output_refit.py`
(`refit_lfm2_functional_output_coefficients` — the literal A11 split):

1. a **planted anchor basis** the child owns exactly (the 4,096 injective
   target rows);
2. **full-dictionary residual OMP** per donor row: search *all* anchors against
   the current orthogonal residual every step, double-MGS the selected atom,
   reproject, tie-break on higher |residual cosine| then lower target id;
3. **k-sparse coefficients** by batched ridge least squares on the selected
   support;
4. the **A11 split**: `C_out` re-solves on the *same* supports against a
   different (functional-metric-weighted) objective, with the direct one-hot
   anchor rows frozen and `C_in` provably unmutated.

Only step 1's *content* is vocabulary-specific. Steps 2–4 are indifferent to
what a "row" means.

### 1.2 What maps to body channels

| token interface | body component | transplanted? |
|---|---|---|
| anchor table (planted child vocab rows) | the **kept channels** of the component axis, as exact one-hot routes | yes — the anchor basis is now *selected*, not planted (§1.3) |
| donor rows (parent vocab embeddings) | the **dropped channels** of the same axis | yes |
| row direction = embedding vector | channel direction = that channel's **column of sealed evidence activations** | yes |
| direct bridge rows (exact one-candidate) | kept channels (support = itself, coefficient 1) | yes, and they are structural rather than solved |
| residual OMP over the full anchor dictionary | identical, over kept-channel evidence columns | yes |
| `C_in` = read-side coefficients | value-reconstruction coefficients for a dropped channel | yes |
| `C_out` = write-side coefficients on shared supports | induced-**output**-error coefficients on the same supports | yes, and it becomes a *coupled* solve (§1.4) |
| local norm reclosure (`reclose_..._norms`) | — | **no** (§1.6) |
| chunked candidate/coefficient checkpoint artifacts | — | **no** (§1.6) |

### 1.3 The one thing the token lane did not have to decide: which rows are anchors

A9's anchor table is *given* — 4,096 injective target rows exist by
construction. A body component has no planted basis, so the kept set has to be
chosen, deterministically, from sealed evidence alone.

`select_kept_channels` implements three policies, all sorted-ascending output
so the child's channel order is the parent's:

* `evidence-energy` (default) — descending sealed-evidence column energy, stable
  ties on the lower channel id. One line, and it is the rule
  `matrix_technique_relation_matching.select_kept_channels` restates verbatim so
  a **T5-vs-T8 contrast at the same level folds the same kept channels and
  differs only in objective**. A cross-module test asserts the two agree.
* `pivoted-qr` — greedy maximum-residual-norm pivots with an exact norm
  downdate. Favours a well-conditioned dictionary over a high-energy one, and
  refuses (`T5BodyRouteConditioningError`) when the evidence rank runs out
  before the level does.
* `explicit` — the option bank drives it.

This is a genuine new degree of freedom the token lane never had, and it is the
first thing to vary if a wave-1 T5 cell fails on `local-solve`.

### 1.4 The A11 split becomes a *coupled* solve — and that is the point

For a writer, the dropped channel `d` contributes `x_d · W[d,:]` to the output.
Absorbing `x_d ≈ Σ c_{d,i} x_i` into the kept columns gives residual
`E = A_drop − A_kept Cᵀ` and induced output error

```
‖E · M_drop‖²_F  =  Σ_{d,e} G_{de} ⟨r_d, r_e⟩ ,   G = M_drop M_dropᵀ
```

The dropped channels are **coupled through `G`** — they all write into the same
kept columns. That is a genuinely different system from the decoupled
value-reconstruction solve, which is exactly why A11's `C_out` was worth having.

* `read` coefficients: one exact ridge solve per dropped channel, grouped by
  support width, entirely in Gram space.
* `write` coefficients: seeded from `read`, then bounded **damped block-Newton
  (Jacobi) sweeps** — `write_solve_sweeps` (default 4), damping ladder
  `(1.0, 0.5, 0.25)`, each sweep accepted only on a **strict decrease** of the
  coupled objective, best iterate retained.
* When `G` is diagonal (identity write metric, i.e. a reader with no downstream
  matrix supplied), the coupled solve collapses to the decoupled one and the
  receipt seals `write_solve.coupled = false` and `sweeps_run = 0` rather than
  pretending a second lane ran.

Measured at real `C13` shape (2560→1024, M16, k=8): `coupled=true`,
`objective_improved=true`, and installing the write coefficients instead of the
read ones cuts the writer's own output relL2 (test
`test_write_coefficients_beat_read_coefficients_at_the_writer_output`).

### 1.5 Role geometry — the second thing the token lane did not have

A body component has one *component axis* (the width the level contracts) and
one *interface axis* (the residual width a single-component cell never touches).

* **writer** (component axis = the map's input): `conv_out_proj`, `attn_o`,
  `mlp_down`. The route is **absorbed into the weight** —
  `W'[i,:] = W[kept_i,:] + Σ_d c_{d,i} W[drop_d,:]` — and no runtime route
  object survives. The child gathers the kept input channels.
* **reader** (component axis = the map's output): `conv_in_proj`, `attn_q/k/v`,
  `mlp_gate`, `mlp_up`. The kept rows of `W` are copied **bit-exactly** and a
  sparse `m×N` expansion carries the narrow output back to the width the
  copy_exact downstream consumer still expects.

Both coefficient sets are always solved; `install_coefficients` picks which one
is installed (writer → `write`, reader → `read` by default) and the other is
sealed as the counterfactual.

**`conv_in_proj` block structure.** `in_proj` emits B/C/x as three contiguous
blocks over *one shared internal channel axis*. Selecting channels on the flat
3072 axis would shred that. `block_count=3` selects and routes on the internal
channel: every block's rows stack into one OMP dictionary (`(3R, 1024)`), one
support and one coefficient set is shared by all three blocks, and the child
gathers `{b*1024 + kept_i}`. A writer with `block_count != 1` is refused — a
writer reads one axis.

### 1.6 What deliberately did **not** transplant

* **Local norm reclosure.** `reclose_lfm2_sparse_semantic_codebook_norms` fixes
  the *embedding norm* distribution, which is a token-interface pathology
  (vocabulary rows have a meaningful norm law). A body channel's norm is
  already carried by the coefficients; adding a reclosure would silently make
  the cell a two-mechanism cell. Not implemented, deliberately.
* **Chunk-checkpointed candidate/coefficient artifacts.** A9 needed them because
  it ran 65,536 donor rows against 4,096 anchors. The largest body cell is
  ~1,800 dropped channels; every wave-1 shape solves in seconds (§5), so the
  checkpoint machinery would be pure ceremony. The prefix-reuse fields stay in
  the receipt (`support_reuse_prefix_steps: 0`) so the schema matches A9's.
* **The R-sized Q basis.** The estate's OMP keeps an explicit
  `(rows, k, hidden)` double-MGS basis. At body scale that is up to 944 MB.
  Replaced with an **incremental Cholesky in Gram space**: the pivot
  `G_jj − ‖L⁻¹G_Sj‖²` *is* the squared double-MGS orthogonal residual of the
  candidate atom, so the stability test is preserved exactly while nothing
  larger than `m×m` is ever allocated. Sealed as
  `"incremental-cholesky-pivot-equivalent-to-double-mgs"`. Dependent atoms are
  masked and retried up to 4 times, then the row stops short and the receipt
  counts it.
* **C01/C02/C15.** The token interface is A9/A11's own already-sealed lane
  (S4). `resolve_component` refuses them by name rather than re-deriving a
  solved cell.
* **C04 / C10 / C14.** Depthwise FIR taps, head counts and diagonal norms have
  no channel-route geometry. Refused by name.

---

## 2. T7 — attention-relation matching

### 2.1 The whole head is one bilinear form

The scores are `X (Wq_h Wk_hᵀ) Xᵀ / √d` and nothing else about the factors is
observable through the softmax. So contracting the head dimension is not
"shrink Q and K", it is *find the best rank-`d'` bilinear form under the
evidence metric*, and the factors are whatever gauge you pick afterwards.

With `M = XᵀX`, the score-kernel error is exactly `‖M^{1/2}(B'−B)M^{1/2}‖_F`,
so **the `R×R` score matrix is never formed**.

### 2.2 The Stein primitive

`solve_two_sided_stein_ridge_gram(left_gram, right_gram, cross, ridge=…)` solves
the design's literal normal equations `(AᵀA) S (BBᵀ) + λS = Aᵀ E Bᵀ` via two
`eigh`. Directions whose eigenvalue product falls below a machine-epsilon floor
get **pseudo-inverse semantics** (coefficient zero) rather than an invented
regularizer, so `ridge=0` stays exact wherever the right-hand side vanishes too
— which is what makes the no-contraction gate exact.

`left_eigen=` lets the caller hand over a precomputed `eigh(M)`. T7 reuses one
residual-width decomposition across every head and every half-sweep; without it
an `O(H³)` term sits inside the iteration and the cell blows its budget.

This subsumes the planned `hf_stein_relation_solve.py`; it is exported
(`solve_two_sided_stein_ridge`, `..._gram`, `STEIN_ALGORITHM`) so that module can
be a thin re-export if the harness lane wants the path.

### 2.3 Alternating closed form — and the finding that it does not need to iterate

Alternating half-sweeps run to `iteration_cap` (registered default **8**; the
design names a cap but no number), each accepted only on a strict decrease of
the *unregularized* score-kernel objective, best iterate retained.

**Finding: at `ridge_relative = 0` the closed-form initializer is already the
global optimum, so the alternating lane correctly accepts nothing.** The
initializer takes the top-`d'` eigenspace of `A S Aᵀ` with `A = F·Wk_g`,
`S = Σ_h (F·Wq_h)ᵀ(F·Wq_h)` — which is Eckart–Young on the lifted kernel with
the right subspace shared across a GQA group, i.e. the exact minimizer of
`Σ_h ‖C_h − A_h Vᵀ‖²` over orthonormal `V`. The receipt seals
`alternating_improved_on_closed_form: false`, and a test asserts the iterations
actually ran. This is a confirmed optimum, not a stalled solver. The lane earns
its keep only under ridge.

### 2.4 GQA, gauge, and softmax scale

* One child K per key/value head, fitted against the **summed** normal equations
  of its whole query group. Verified: `per_head` maps 6 query heads onto 2 kv
  heads correctly.
* The residual `O(d')` gauge is fixed by the dormant Procrustes lane (a compact
  deterministic restatement of
  `hf_cpcf_boundary_chart::_weighted_orthogonal_procrustes`), plus an optional
  per-group scalar rebalance. Both leave `B'` invariant; the module **verifies**
  that with an absolute tolerance against the parent's own lifted energy (a
  relative tolerance trips on rounding noise when the objective is ~0 — that
  bug was found and fixed by the identity gate).
* **Softmax scale.** Changing `d'` changes `1/√d'`. `compensate_softmax_scale`
  (default `True` for standalone use) folds `√(d'/d)` into the child Q so
  `q'k'ᵀ/√d'` reproduces `qkᵀ/√d`. The registry adapter sets it **False**,
  because a parent-shaped install keeps the parent's `head_dim` and the parent's
  scale is already correct. Getting this backwards would silently rescale every
  attention logit in the cell.

### 2.5 The `C06+C07` cell versus the registry's one-component rule

The design writes the cell as `C06+C07` (Q and K jointly). The registry
(correctly, per §1.2 and §5 rule 6) lets a cell install only the tensors its own
component owns. **Both are satisfied without compromise.**

The fitted kernel's row space already lies inside the parent partner's column
space — it was built from it — so a single side can carry the joint optimum:
`realize_sides=("query",)` re-solves a **parent-shaped** `Wq''` (one Stein
solve, partner = the untouched full parent `Wk`) with
`Wq'' Wk_pᵀ = B*` exactly.

Verified numerically: at `d'=6` of `d=8`, joint score-kernel error `0.304183`
and realized-single-side error `0.304183`, with `rank(Wq''_h) = 6`. The adapter
asserts this equality in `test_t7_installs_one_side_against_a_bit_identical_partner`.

Consequences the ledger should know:
* the partner stays **bit-identical**;
* the softmax scale is the parent's;
* the contraction shows up as the **rank** of the installed tensor, not its
  shape, and `stored_parameter_count = heads · d' · (H + d)`;
* running the `C06` cell and the `C07` cell measures the same relation from two
  sides, which is strictly more information than one two-component cell.

---

## 3. T8 — Gram/relation matching

### 3.1 Two mechanisms, only one of them always live

With `Ỹ = D·Y` the row-normalized parent output, `‖Ỹ'Ỹ'ᵀ − ỸỸᵀ‖_F` over
rank-`≤r` children is minimized by the rank-`r` truncation of `Ỹ` itself in the
parent's own right-singular gauge (that *is* the Procrustes gauge choice),
making the target `T = Y V_rV_rᵀ`. One `D²`-weighted ridge solve finishes it.

**The bug I found and fixed in my own first formulation:** un-normalizing the
truncated target cancels the normalization, so when `r` reaches the parent's
relation rank the target is just `Y` and T8 is numerically **identical to a
dense ridge**. That is the common case for a wide reader feeding a narrow writer
— i.e. it would have made wave-1 cells 22–24 meaningless.

The fix is to put the scale-freeness in the **weighting**, not only the target:
fit against `D·X_c W' ≈ Ŷ`, i.e. normal equations `X_cᵀD²X_c` / `X_cᵀD²T`. Every
evidence row then enters with its *direction* at equal weight instead of its
magnitude squared. This is live at every level and is genuinely different from
T1. Verified: rescaling one evidence row by 50× moves the T8 fit by `<1e-8`
Frobenius while it moves the unweighted ridge by `>1e-3`
(`test_scale_free_weighting_ignores_a_rescaled_evidence_row`).

The receipt distinguishes the two mechanisms so an inert projector cannot look
like a technique: `solve.row_weighting`
(`inverse-parent-row-norm-squared` | `uniform`), `solve.scale_free_objective`,
`solve.gram_truncation_active`, `solve.relation_rank`,
`solve.requested_relation_rank`, `solve.relation_rank_clamped`.
`normalization="none"` is the deliberate dense-ridge control arm.

### 3.2 Reported split

`direction_cosine_mean` / `direction_cosine_min` / `magnitude_ratio_mean` /
`value_relative_l2` / `gram_relative_error`, on fit and held-out rows. The Gram
error is computed through `‖Ỹ'ᵀỸ'‖²−2‖ỸᵀỸ'‖²+‖ỸᵀỸ‖²`, so the `R×R` Gram is
never materialized.

### 3.3 T8 covers the measured wall

`_T8_COMPONENTS` is C03, C05–C09, C11–C13 — including `conv_out_proj`, which is
the design's stated reason for the technique. Both contraction axes are the same
closed form with a different contracted object: `input` axis contracts the kept
design channels, `output` axis caps the relation rank.

---

## 4. T4 — factored writers

### 4.1 Wired, not reinvented

The child geometry is the one `hf_lfm2_dynamic_blank_materializer` already
materializes: `out_proj` becomes `Sequential(Linear(internal, rank),
Linear(rank, hidden))` at `conv.out_proj.0.weight` / `conv.out_proj.1.weight`,
bridged from fold-evidence names by
`lfm2_factorized_short_conv_evidence_state_key_mapping`.
`lfm2_short_conv_writer_state_entries()` emits exactly those keys with exactly
those shapes, and `test_state_key_layout_still_matches_the_owning_materializer`
reads that function's source and fails if the key layout ever moves.

The parameter algebra is `dynamic_parameter_frontier`'s:
`writer_rank * (hidden + internal)` against a dense `hidden * internal`.

### 4.2 The fit is the new part, and the evidence lane wins

`weighting="none"` is the plain truncated SVD of the writer.
`weighting="evidence"` (default) minimizes `‖XW − XED‖_F`, whose optimum is
`ED = M^{-1/2}·trunc_r(M^{1/2}W)` — one `eigh`, one SVD, no iteration.

On correlated evidence at rank 8, the evidence lane writes the component at
relL2 **0.289** against the plain SVD's **0.720**, and the same ordering holds
on held-out rows (**0.286** vs **0.723**). The trade is sealed rather than
silent: `solve.weight_energy_outside_evidence_range` records how much weight
energy was dropped for lying in directions the evidence never excites.

### 4.3 The hard edge: marginal levels do not contract a square writer

A factored writer only costs less than a dense one below `in·out/(in+out)` —
for a square writer, just under **half** the full rank.
`maximum_shrinking_rank(1024, 1024) == 511`; `maximum_shrinking_rank(2560, 1024)
== 731`.

The campaign's marginal levels keep 15/16, 7/8, 3/4. Therefore:

* **wave-1 cells 27 (`C05.T4.M16.L13`) and 28 (`C09.T4.M16.L02`) are not
  contractions at all** — at rank 960 a 1024×1024 writer stores 1,966,080
  against a dense 1,048,576. The kernel **fails closed** with the usable ceiling
  in the message; the knob `t4_allow_parameter_increase` is the deliberate
  override for a probe that wants the numeric answer anyway.
* on a rectangular writer the level can also exceed the rank ceiling
  (`C13.T4.M4`: level width 1920, full rank 1024). The adapter **clamps** to the
  full rank and seals `requested_writer_rank` / `writer_rank_clamped`.
* the usable T4 regime on those components is a **deep explicit rank** —
  the goal doc's witness. Drive it with the `writer_rank` knob.

Real-shape witness check (1024×1024 writer, rank 31): 63,488 stored against
1,048,576 dense, ratio **0.0605**.

---

## 5. Gate numbers

### 5.1 Exact-identity gates (all required `< 1e-6`)

| gate | measured |
|---|---|
| T5 no contraction, writer: `max|W_child − W_parent|` | **0.0** (bit-identical) |
| T5 no contraction, writer: component write relL2 | **0.0** |
| T5 no contraction, reader: `torch.equal(child, parent)` | **True** |
| T5 `k = full` on spanned evidence: channel reconstruction relL2 | **1.18e-10** |
| T5 `k = full` on spanned evidence: component write relL2 | **4.58e-11** |
| Stein primitive, exactly representable solution: relF | **1.72e-15** |
| T7 no contraction: score-kernel relF (max over heads) | **4.06e-08** |
| T7 no contraction: published-weight score reproduction on held-out rows | **2.49e-15** (small case) |
| T7 no contraction, realized single side | **4.06e-08** |
| T8 no contraction: `max|W_child − W_parent|` | **2.43e-11** |
| T8 no contraction: Gram relF | **1.28e-08** |
| T4 full rank, `weighting="none"`: `max|DE − W|` | **7.33e-15** |
| T4 full rank, `weighting="evidence"`: `max|DE − W|` | **1.95e-13** |

T7's and T8's residuals are set by the conditioning of `M^{-1/2}` on the test
evidence, not by the fit; both are three to four decades inside the gate. The
`test_contraction_degrades_monotonically...` case measures 3.85e-08 for the same
reason and the test threshold says so in a comment.

### 5.2 Wave-1 cell shapes, real LFM2.5-230M geometry, CPU FP64

`H=1024, intermediate=2560, Q16/KV8/D64, 2048 evidence rows.` Synthetic
correlated evidence (latent 384–512), so the *quality* numbers are shape
sanity, not predictions — the timings and the mechanism flags are the point.

| cell | wall | result |
|---|---|---|
| `C13.T5.M16.L05` (k=8) | **2.4 s** | kept 2400 / dropped 160, chan relL2 0.9023, write relL2 0.2098, `coupled=True`, `improved=True` |
| `C05.T5.M16.L00` (k=32) | **1.1 s** | kept 960, chan relL2 0.7025, write relL2 0.1612 |
| `C11.T5.M16.L00` (reader) | **2.5 s** | kept 2400, child (2400, 1024), chan relL2 0.8496 |
| `C06+C07.T7.M16.L02` | **4.3 s** | `d'=60`, score relF max 0.1307 / mean 0.1254, realized-query 0.1307, stored 1,044,480 |
| `C05.T8.M16.L00` | **1.5 s** | kept 960, rank 960, Gram relF 0.0098, cos 0.9999, truncation active |
| `C05.T4.rank31.L13` | **1.5 s** | 63,488 / 1,048,576 params, ratio 0.0605 |

Every wave-1 shape runs in **1–5 s on one CPU core**, against a 12-minute
target and a 20-minute hard cap. Solve time is not the campaign's bottleneck;
the composed forward, free-run and decode smoke are.

Note the T7 stored count: at M16 the installed q_proj drops from 1,048,576 to
1,044,480 — **0.4%**. T7's contraction is in the *kernel rank*, and the
parameter win only materializes if the harness stores the factored form. Worth
saying out loud before anyone reads a T7 pass as compression.

---

## 6. Registry wiring (harness lane)

The three modules register themselves at import time, exactly as
`matrix_technique_registry`'s docstring specifies. **Importing the module is the
registration.**

```python
import neural_forge.matrix_technique_body_routes        # T5 x 9 components
import neural_forge.matrix_technique_relation_matching  # T7 x 2, T8 x 9
import neural_forge.matrix_technique_lowrank_writers    # T4 x 3
```

Resulting `registered_pairs()`:

```
('T4','C05') ('T4','C09') ('T4','C13')
('T5','C03') ('T5','C05') ('T5','C06') ('T5','C07') ('T5','C08')
('T5','C09') ('T5','C11') ('T5','C12') ('T5','C13')
('T7','C06') ('T7','C07')
('T8','C03') ('T8','C05') ('T8','C06') ('T8','C07') ('T8','C08')
('T8','C09') ('T8','C11') ('T8','C12') ('T8','C13')
```

All four `TECHNIQUES[...]` canonical names match
`matrix_cell_receipt.TECHNIQUES` (T4 was `lowrank_factor_writers` in my first
draft and is now `low_rank_factor_writers`).

### 6.1 Entry points

| callable | registered as |
|---|---|
| `matrix_technique_body_routes.t5_body_route_technique` | `T5` at C03, C05–C09, C11–C13 |
| `matrix_technique_relation_matching.t7_attention_relation_technique` | `T7` at C06, C07 |
| `matrix_technique_relation_matching.t8_gram_relation_technique` | `T8` at C03, C05–C09, C11–C13 |
| `matrix_technique_lowrank_writers.t4_lowrank_writer_technique` | `T4` at C05, C09, C13 |

Every adapter returns `installed_weights` that are **parent-shaped and
functionally identical to the contracted child**, so the copy_exact neighbours
really are copy_exact:

* T5 writer — scatter the child's columns into the kept positions, zero the
  dropped ones. The dropped channels multiply zero, so it is the same function.
  `stored = out × kept`.
* T5 reader — `expansionᵀ @ child` per block; expansion is linear so the
  composition is a parent-shaped tensor computing the same function.
  `stored = kept·blocks·in + route_support_entries`.
* T7 — the single-side realization (§2.5). `stored = heads · d' · (H + d)`.
* T8 — `input` axis scatters kept columns (`stored = out × kept`); `output` axis
  installs the rank-capped product (`stored = rank × (out + in)`).
* T4 — `decoder @ encoder`. `stored = rank × (in + out)`.

Each adapter passes the registry's own `validate_result` in
`tests/test_matrix_technique_registry_adapters.py`, which drives real
`CellTechniqueRequest` objects through `component_spec` /
`resolve_level_width` / `resolve`.

### 6.2 Knobs the harness can set per cell

| technique | knob | default |
|---|---|---|
| T5 | `active_k` | 8 |
| T5 | `channel_selection` | `evidence-energy` |
| T5 | `write_solve_sweeps` | 4 |
| T5 | `install_coefficients` | role default (`write` / `read`) |
| T5 | `refit_ridge_relative` | 1e-10 |
| T7 | `head_dim`, `query_heads`, `key_value_heads` | 64, 16, 8 |
| T7 | `ridge_relative` | 0.0 |
| T7 | `iteration_cap` | 8 |
| T7 | `procrustes_gauge`, `balance_factors` | True, True |
| T8 | `normalization` | `row_unit` (`none` = dense-ridge control) |
| T8 | `ridge_relative` | 0.0 |
| T4 | `weighting` | `evidence` |
| T4 | `writer_rank` | `level_width`, clamped to the full rank |
| T4 | `t4_allow_parameter_increase` | False |

### 6.3 Three contract bugs the registry's validator caught

Recorded because they are the kind of thing that would have burned GPU cells:

1. **T7 sealed no `prior_norm` / `correction_norm`.** Both are mandatory (§3).
   Fixed by reporting the *relation's* norms — parent lifted kernel energy and
   the kernel's move in the evidence metric — because factor norms carry a gauge
   the relation does not see.
2. **T8 on an output axis could be asked for a rank wider than the component's
   own input** (`C11.T8.M4`: level width 1920, achievable 1024). Now clamps and
   seals `relation_rank_clamped` instead of refusing a cell whose scale-free
   weighting still has content — matching the reference T1 kernel's own
   accounting behaviour on that axis.
3. **T4's level can exceed the writer's rank ceiling on a rectangular writer.**
   Now clamps to the full rank and seals `writer_rank_clamped`.

---

## 7. Fold-only discipline

Every technique is OMP / least squares / `eigh` / SVD / Procrustes on sealed
evidence and already-materialized parent weights.

* Every input tensor carrying autograd state is refused
  (`*DisciplineError`), tested in all three modules.
* All linear algebra runs inside `torch.no_grad()`.
* Every sealed record carries the estate's prohibited-operation audit with
  `backward_calls`, `optimizer_steps`, `parent_forward_calls`,
  `parent_trace_recaptures`, `model_forward_calls` = 0 and
  `network_access = False`, asserted in the seal tests.
* No parent forward anywhere. Where a component's own output rows are needed
  and no sealed endpoint carries them, they are **derived** as
  `fit_rows(input_endpoint) @ parentᵀ` — pure arithmetic on sealed rows and a
  materialized weight — and the receipt seals
  `component_rows_source: "<endpoint> @ parent^T"`.

Sealing: each module canonically seals its technique record
(`canonical_sha256` over a JSON-plain body, non-finite floats refused) and
exposes a `verify_sealed_*` that re-checks seal and schema. Records are
reproducible: the same request twice gives the same `record_sha256`
(`test_records_are_reproducible_for_the_same_request`). The adapters put the
sha into `CellTechniqueResult.artifact_identities` and mirror the full
calibration block into `solve` for the receipt.

---

## 8. Refusal ladders

| module | ladder |
|---|---|
| T5 | `T5BodyRouteError` → `Geometry` / `Evidence` / `Conditioning` / `Discipline` / `Authentication` |
| T7/T8 | `RelationMatchError` → `Geometry` / `Evidence` / `Conditioning` / `Discipline` / `Authentication` |
| T4 | `LowRankWriterError` → `Geometry` / `Evidence` / `Conditioning` / `Budget` / `Discipline` / `Authentication` |

Each adapter wraps its module's error in `CellTechniqueError` with the cell id
prefixed, so the harness seals `infra` with the real reason attached rather than
a stack trace.

---

## 9. Deliberately left out

* **No `hf_stein_relation_solve.py`.** Design §2.3 names it as a separate module;
  the primitive lives in `matrix_technique_relation_matching` and is exported.
  Creating a second file for a re-export would have added a lane boundary
  nobody asked for. Say the word and it becomes a four-line shim.
* **No T5 causal-damage reselection arm.** `hf_lfm2_causal_semantic_support.py`
  is built-and-unrun and the design schedules it for wave 2 on the *best*
  wave-1 T5 cell. Nothing here blocks it — it reselects supports, and supports
  are already a first-class sealed object (`support_offsets`,
  `support_channel_ids`).
* **No T9, T2, T3, T6.** Out of scope for this lane; the design defers them.
* **No `C09`/`C13` two-factor install path for T4.** Only the short-conv writer
  has materializer support today; `factor_writer` produces correct factors for
  `attn_o` and `mlp_down` and the record seals `materializer_backed: false`, but
  the harness would have to build the `Sequential` itself. The T4 adapter
  installs the dense product `decoder @ encoder`, which is functionally exact
  and parent-shaped — the *storage* win at C09/C13 needs that install path.
* **No GPU run and no real sealed-cache run.** Everything here is verified on
  synthetic captured-evidence-shaped tensors at real LFM2 shapes on CPU FP64.
  I could not verify against the actual sealed trace cache (`76ff3cbc…`) —
  opening it is the harness lane's stage, and the closed-path ban forbids
  re-capturing. First contact with real evidence is wave-1 cell 0b.
