# NOTES-W4C — cross-width residual crossing, stage 1 (rectangular boundary charts)

Author: wave W4C (cross-width residual crossing)
Date: 2026-08-01
Branch: `codex/lrc-blank-compression` (HEAD at start: `c69d5a64`)
Design: `docs/CROSS_WIDTH_RESIDUAL_CROSSING_DESIGN_2026-08-01.md`

Files changed:

- `src/neural_forge/hf_causal_cross_width_charts.py` (new)
- `src/neural_forge/dynamic_causal_operator_promotion.py`
- `src/neural_forge/dynamic_fold_planner.py` (surgical, condition-gated)
- `tests/test_hf_causal_cross_width_charts.py` (new)
- `tests/test_cross_width_causal_fold_execution.py` (new)

Deliberately untouched: `hf_causal_gqa_attention_fold.py`,
`hf_closed_form_causal_circuit_fold.py`, `hf_causal_circuit_evidence.py`,
`dynamic_balanced_circuit_basis.py` (consumed read-only),
`dynamic_fold_registry.py`, `hf_causal_kind_translation.py`, `experiments/*`,
every `hf_lfm2_*` module.

---

## 1. What the crossing needed, and what it did not

The design's decisive finding held under implementation: **the branch kernels
are already width-agnostic**. `compile_hf_causal_gqa_attention_fold` never
compares the child row width to anything — Q/K/V are ridge maps from child rows
to parent head targets and `output_weight` is a ridge from the replayed context
to the endpoint target — and `fit_functional_swiglu` constrains only row counts.
Neither kernel was modified, and neither needed to be.

Width equality was enforced in exactly the two places the design named:

- `dynamic_causal_operator_promotion.py` `_require_shared_residual_width`, now
  called only when no chart binding is supplied;
- `dynamic_fold_planner.py` `_family_neutral_causal_layer_contract`, now
  relaxed only when `_cross_width_boundary_chart_crossing` proves a
  contraction-only rope/causal-decoder crossing.

Everything else in this wave is target construction and provenance.

## 2. `hf_causal_cross_width_charts.py`

**Boundary identity.** One boundary per residual write plus the terminal one:
`layer{NNN}.attention_in`, `layer{NNN}.mlp_in`, `output.final_norm_in`
(`2N + 1` total). Boundary 0 and the terminal boundary are interface locked.

**`ResidualBoundaryChart`** — sealed `boundary_id`, `encoder [Hp, Hc]`,
`decoder [Hc, Hp]`, `interface_locked`, and a canonical-JSON report carrying
encoder/decoder sha256, the measured defects, the policy sha, the evidence
lineage, and a `chart_sha256` self-seal. It also implements
`source_width`/`target_width`/`map_sha256`/`fold_consumer_weight`, which is
exactly the dense-vocabulary-projector protocol the token-interface operators
already accept, so an interface-locked chart is directly usable as a
coordinate map without a wrapper.

**`ResidualBoundaryChartSet`** — ordered charts, `parent_evidence_sha256`,
`child_topology_sha256`, policy sha, per-chart shas, self-sealed `set_sha256`.
`verify()` re-hashes every tensor and the seal; `authenticate()` additionally
binds the crossing (evidence sha + child topology sha) and is what every
operator calls before touching anything.

**Encoder selection** is the regularized balanced POD of
`dynamic_balanced_circuit_basis`, consumed read-only and generalized off the
LFM2 lane:

- controllability: the empirical Gram of the cached parent rows at that
  boundary, **fit split only**, with the validation rows supplied as the
  disjoint held-out block;
- observability: the residual-facing weights of every *downstream* consumer —
  this boundary's Q/K/V or gate/up input columns, every later layer's, plus the
  final norm and the output head;
- structural lane: the complete residual-facing weight inventory of the parent.

`build_balanced_circuit_basis(...).basis` is orthonormal by construction, so
`E_b^T E_b = I_Hc` and the norm-commutation rescale stays a single scalar.

**Encoder frame transport (deviation, see §6).** The balanced solve fixes each
boundary's *subspace*; the orthonormal frame inside it is canonical but
arbitrary. Solved independently, two nearly identical subspaces were handed
frames differing by a large rotation, and that rotation showed up entirely as
chart-transition absorption demand. Measured on the H64→H48 toy before the fix:
transition energy `0.44` against a transported-parent-delta energy of `0.36`,
and attention endpoint relative RMSE of **2.85** — worse than writing nothing.
The fix is the orthogonal Procrustes transport the design already cites from
the moving-gauge lane (`R = U V^T` from `SVD(E_b^T E_{b-1})`, applied to the
frame, span unchanged, orthonormality unchanged). After it: transition energy
`0.0048` (90x down) and attention RMSE `0.91`. Interface charts *supplied* by a
caller are never rotated; they anchor the chain instead.

**Decoder** is the deployed moving-gauge solve, reimplemented rather than
imported so no `hf_lfm2_*` module is touched: a regularized reciprocal of the
encoder (`min ||E W - I||^2 + ridge||W||^2`) used as the closed-form
cycle-consistency anchor, then an anchored ridge from encoded fit rows back to
parent fit rows with `ridge_ratio = chart_ridge_ratio + cycle_consistency_weight`.

**Measured defects, each policy-bounded and each a fail-closed condition:**

| defect | formula | policy default |
|---|---|---|
| cycle `c_b` | `\|\|h - (hE)D\|\|_F / \|\|h\|\|_F` on fit rows | `<= 0.75` |
| transition `T_b` | `E_{b+1}^T D_b^T - I_Hc`, Frobenius + max-abs | reported |
| absorption `A_b` | `D_b (E_{b+1} - E_b)`, SVD-split against the writer rank | unabsorbed `<= 0.25` |
| norm commutation `nu_b` | row std of `\|\|hE\|\|/\|\|h\|\|` | `<= 0.25` |

Writer rank is `min(Hc, query_heads*head_dim)` for an attention boundary and
`min(Hc, intermediate_size)` for an MLP boundary; the terminal boundary has no
writer, which is why it is interface locked. Rows with exactly zero energy (a
padded embedding row is the ordinary case) are excluded from `nu_b` and counted
rather than failing the crossing.

**`build_sequential_branch_target`** implements the load-bearing equation in the
sequential form the design demands, `Y = h_{b+1} E_{b+1} - hhat_b_actual`, and
separately measures the three-term expansion
`Delta E_{b+1} + h_b (E_{b+1} - E_b) - (hhat - h_b E_b)`, reporting per-term fit
energies plus `sequential_expansion_max_abs_error` (about `3e-9` on the toy).
The upstream-error term is the A0025 repair and is unit-tested against a
hand-computed two-boundary case.

**`LiveChildBoundaryRecapture`** runs `collect_hf_transport_trace` on the *live
child* and returns `block_input` / `post_attention_residual` / final
`block_output` per boundary. It is a required collaborator: a missing state is a
typed failure, never a fallback to parent coordinates.

**`plan_reverse_reclosure_sweep`** seals the reverse writer-only schedule
(charts frozen, terminal-anchored, reverse boundary order) and reports
`executed: False`. Every branch report carries its sha, so no artifact can
claim a reconciliation that has not run. **The sweep itself is not implemented
in this wave.**

**`identity_residual_boundary_chart_set`** is the width-preserving control:
`E = D = I` at every boundary, every defect exactly `0.0`.

## 3. Operator changes

`_CausalBranchOperator._bind` takes an optional `CrossWidthBranchBinding`. With
`None` it calls `_require_shared_residual_width` exactly as before; with a
binding it calls `_require_chart_bound_residual_widths`, which refuses
expansion, refuses a real width change on an action the planner did not mark
`boundary_chart_required`, and checks the binding's shapes.

Both `prepare` methods gained a keyword-only `cross_width_binding` and one seam
each: `branch_input` / `branch_normalized_target` / `branch_delta_target` are
the cached parent sequences when the binding is `None` and the chart-transformed
sequences otherwise. Nothing else in either method changed, and the kernels
below the seam are called with identical arguments in the default path.

`CrossWidthCausalBranchOperator` wraps either branch operator: authenticate the
chart set against the crossing, bind both boundary charts, recapture the live
child at the input boundary, build `Y`, delegate, then enforce the endpoint
condition. `CrossWidthTokenHeadInterfaceOperator` claims `token_interface_bridge`
and installs the two interface-locked boundaries as one transaction —
`W_embed @ E_0`, `W_head @ E_T`, and the final-norm gains fitted in terminal
chart coordinates.

**Fail-closed conditions implemented** (design §"Fail-closed conditions"):

1. width expansion — refused in chart compile and in the operator preamble;
2. missing / unauthenticated / lineage-mismatched chart — `authenticate()`
   plus per-boundary `chart()` lookup;
3. balanced-basis rank failure — `BalancedCircuitBasisError` is re-raised as
   `CrossWidthResidualChartRankError`;
4. `c_b`, `nu_b`, unabsorbed `A_b` above policy — raised at compile time,
   before any child mutation;
5. interface-locked boundary touched with anything but an orthogonal chart —
   refused, as is a supplied chart at an ordinary boundary;
6. sequential recapture unavailable — typed failure naming A0025, never a
   parent-coordinate fallback;
7. endpoint relative RMSE at or above the no-branch baseline — the zero-write
   baseline is exactly `1.0` by construction, the transaction is rolled back
   and the crossing refused;
8. fit/validation overlap, non-finite tensors, condition cap — inherited from
   the sealed evidence, `_matrix`/`_sequences`, and `max_condition_number`.

## 4. Planner change (condition-gated)

`_cross_width_boundary_chart_crossing(parent, child)` is the single gate: both
graphs `hf-causal-decoder`, both `rope`, and `0 < child_hidden < parent_hidden`.
Nothing below fires when widths are equal, so every same-width plan is
byte-identical to before.

1. `_family_neutral_causal_layer_contract` keeps every clause and drops only
   width equality behind that gate; the branch actions get
   `configuration["boundary_chart_required"] = True`.
2. `_cross_width_terminal_interface_bundle` extends the token-interface bundle
   with the child's single post-layer norm, so `token_interface_bridge` covers
   `{embedding, final_norm, output_head}` — the two interface-locked residual
   boundaries in one transaction. It returns `()` unless the child owns exactly
   one global norm ordered after every layer.
3. The same method is selected for a *shared* tokenizer under a width crossing
   (an exact copy is impossible at either interface, and the registry's own
   description of the method is "identity, strict row, or byte-span token
   interface compilation").
4. `_same_parameter_contract` / `validate_fold_method_contract` / `_action`
   gained an optional `residual_width_substitution`, used only for a
   **parameter-free** global bundle whose sole difference is the residual port
   width (`input.position`). A node with parameters keeps its real shapes, so a
   norm diagonal or an embedding matrix can never pass a width crossing off as
   an exact copy.

Without the chart operators registered, a cross-width plan is now *ready* but
unexecutable: the plain branch operator refuses the action and the executor has
no operator for it. The gap moved from plan time to a typed operator refusal;
nothing silently folds.

## 5. Measured results — H64 → H48, real executor run

Toy: llama parent `hidden 64`, llama-shaped child `hidden 48`, identical head
geometry (8 query / 4 KV heads, `head_dim 8`) and identical `intermediate_size
96`, 2 layers, vocab 48, 24x24 = 576 tokens (288 fit / 288 validation). The
parent is the design's stage-1 exact-subspace control: every residual-facing
weight is confined to one 48-dimensional subspace and every RMSNorm gain is
constant, so the parent's residual states lie in 48 dimensions to `<1e-6`
relative (asserted in the test). Charts compile in ~0.3 s; the whole fold runs
in ~11 s on CPU.

Plan: `ready`, **zero gaps**, 6 actions, every one of the child's 21 learned
tensors owned exactly once.

| boundary | cycle `c_b` | `nu_b` | incoming `T_b` (Fro) | unabsorbed `A_b` |
|---|---|---|---|---|
| `layer000.attention_in` | 1.66e-08 | 3.4e-16 | — | 0.0 |
| `layer000.mlp_in` | 4.33e-08 | 7.6e-16 | 2.5e-08 | 0.0 |
| `layer001.attention_in` | 4.57e-08 | 8.0e-16 | 1.7e-08 | 0.0 |
| `layer001.mlp_in` | 5.58e-08 | 9.9e-16 | 1.7e-08 | 0.0 |
| `output.final_norm_in` | 5.75e-08 | 1.0e-15 | 1.7e-08 | n/a |

| action | held-out endpoint relative RMSE | upstream-error energy ratio |
|---|---|---|
| `token_interface_bridge` (final norm) | 1.507e-04 | — |
| layer 0 attention | 6.783e-05 | 1.7e-15 |
| layer 0 MLP | 9.454e-04 | 3.8e-08 |
| layer 1 attention | 2.679e-04 | 1.5e-08 |
| layer 1 MLP | 1.590e-03 | 1.2e-06 |

- executor `status == "complete"`, every attempt accepted;
- **0 parent forwards** (measured with a forward hook and by the executor's own
  observer), 0 optimizer constructions/steps, 0 backward calls, 0 autograd
  calls, 0 pruning/quantization ops, static zero-training audit passed;
- **4 live-child recaptures**, one per branch action — the sequential discipline;
- child logits finite, shape `(24, 24, 48)`;
- **whole-model teacher-forced logit relative RMSE against the H64 parent:
  1.93e-04**, through a completely different residual coordinate system;
- rejecting the second attention branch rolls the whole run back to the blank
  child bit-for-bit (`torch.equal` on every state-dict entry).

Width-preserving control (identity charts, `hhat = h_b`): the input norm and all
of Q/K/V (or gate/up) are installed **bit-identically** to the parent-coordinate
operator; only the branch's write projection differs, by `<1e-5` relative,
because the sequential target `h_{b+1} - h_b` and the cached float32
`attention_delta` are the same quantity at different precision (the drift is
asserted against `2 * eps_float32 * max|h_{b+1}|`). Chart transition and
upstream-error energies are exactly `0.0`.

Narrow-attention fail-closed: a child with one 4-wide query head against the
48-wide residual leaves `0.33` of its `A_b` outside the writer's rank-4 span, so
the **default** policy refuses the crossing at chart-compile time; admitting it
with a `0.5` bound produces the same chart set and shows `writer_representable_rank == 4`.

## 6. Deviations from the design

1. **Module name.** The design proposes
   `src/neural_forge/dynamic_residual_boundary_charts.py`; the wave task
   specified `src/neural_forge/hf_causal_cross_width_charts.py`, which is what
   was built. The name is also more honest: the module consumes
   `HFCausalCircuitEvidence` and an HF `DecoderLayout`.
2. **`compile_residual_boundary_chart_set` takes the parent model.** The design
   signature is `(evidence, child_layout, policy)`, but the observability lane
   is defined over the parent's residual-facing *weights*, which the sealed
   causal evidence does not carry (it holds activations plus the token
   interface). The parent model is read for already-materialized tensors only —
   no forward, hook, trace, or gradient.
3. **Orthogonal Procrustes frame transport was added** (§2). It is not spelled
   out in the design's "Chart selection" section, but it is the design's own
   cited prior art and without it the chart transition swamps the branch write.
   Numbers for both sides are in §2.
4. **Interface-locked charts are derived, not supplied, by default.** The design
   says interface charts are "fixed by the token bridge and head construction,
   or restricted to orthogonal rotation". This wave has no token/head interface
   that owns a cross-width chart, so chart selection derives orthonormal charts
   at those boundaries and `CrossWidthTokenHeadInterfaceOperator` *consumes*
   them. Supplying them is fully wired
   (`interface_locked_encoders=`) and validated (orthogonality, shape,
   ordinary-boundary refusal) for the day the interface work lands.
5. **Terminal-norm gains are fitted in ideal chart coordinates**, not against a
   live recapture. The token/head transaction is planned before any layer, so a
   recapture there would read a blank child. A diagonal cannot correct
   accumulated upstream error anyway; that is the reverse sweep's job.
6. **Condition 7 is checked in float64, not "after BF16 materialization
   replay".** No BF16 replay exists in this lane yet.
7. **Depth must be preserved.** `CrossWidthCausalBranchOperator` refuses
   `source_layer_index != target_layer_index`; a depth-changing crossing needs
   its own boundary evidence and is out of this wave.
8. **Planner reach.** The design's planner section only mentions the layer
   contract. Delivering a genuinely gap-free cross-width plan also required the
   two global-interface changes in §4 — the embedding, final norm, and head all
   change width at once and none of them can be an exact copy. Both are gated
   on the same predicate.

## 7. Not done / known limits

1. **The reverse reclosure sweep is specced and sealed, not implemented.** Only
   the greedy forward pass runs. Every branch report says
   `reverse_reclosure_sweep_executed: false`.
2. **Stage 1 only.** The evidence here is the design's toy exact-subspace
   control plus a full-rank random-parent smoke run. Stages 2-4 (same-family
   H1024→H816, the LFM2 lane, cross-family cross-width) are untouched.
3. **The full-rank random-parent toy does not clear the endpoint gate with any
   margin.** On an untrained H64 parent with no subspace structure the branch
   errors sit at `0.75-1.00`, which is what the *same-width* kernels already do
   on random weights (measured: attention `0.82/0.34`, MLP `0.96/0.93`). The
   toy in the test suite is therefore the exact-subspace control, and the
   full-rank case is reported here rather than asserted.
4. **No `dynamic_fold_registry.py` wiring.** The cross-width operators must be
   constructed and registered explicitly by a caller, so the default registry
   still cannot execute a width crossing. That is deliberate fail-closed
   behaviour for an unproven lane, and it is why the toy builds its own
   `FoldOperatorRegistry`.
5. **Chart compile cost is O(boundaries x downstream consumers) Grams.** Fine at
   toy scale (0.3 s) and bounded by the precomputed-Gram schemas the balanced
   basis module already defines, but it has not been run at H1024 x 24 layers.
6. **No persistence yet.** The artifacts are sealed canonical-JSON in memory
   with content-addressed tensor fingerprints and a resumed compile is proven
   byte-identical, but nothing writes them to disk; the three schema identifiers
   the design names are defined and used, not serialized.
7. **`hf_causal_kind_translation.py` is another wave's file** and is not
   composed with here. Stage 4 stacks both mechanisms and must not be the first
   evidence for either.

## 8. Verification

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" -m pytest \
    tests/test_hf_causal_cross_width_charts.py \
    tests/test_cross_width_causal_fold_execution.py \
    tests/test_dynamic_causal_operator_promotion.py \
    tests/test_cross_family_causal_fold_execution.py \
    tests/test_hf_causal_gqa_qk_norm_fold.py \
    tests/test_dynamic_fold_planner.py -q -p no:randomly
```

`72 passed` (~37 s). New coverage: 19 chart tests (boundary identity,
orthonormality, roundtrip against the measured cycle defect, seal tampering,
lineage authentication, identity-chart control, the hand-computed A0025 target
guard, expansion refusal, interface-chart refusals, the narrow-attention
absorption bound, policy sealing, byte-identical resume, sweep sealing,
recapture behaviour) and 15 execution tests (subspace precondition, gap-free
plan and ownership partition, the still-gapped plan without the experimental
lane, complete execution with zero parent forwards, per-boundary defect and
endpoint bounds, whole-model logit fidelity, exact rollback, the plain
operator's refusal, foreign chart set, unplanned crossing, endpoint gate
rollback, missing recapture, unshared tokenizer identity, and the two
width-preserving control cases).

Also re-run green and unchanged: `tests/test_dynamic_fold_compiler.py`,
`tests/test_dynamic_fold_executor.py`, `tests/test_dynamic_fold_registry.py`,
`tests/test_dynamic_fold_artifact.py`, `tests/test_dynamic_system.py`,
`tests/test_dynamic_model_inspector.py`, `tests/test_dynamic_circuit_ir.py`,
`tests/test_dynamic_balanced_circuit_basis.py`,
`tests/test_dynamic_weight_complete_basis.py`,
`tests/test_dynamic_fold_diagnostics.py`, `tests/test_dynamic_fold_budget.py`
(180 passed, 1 skipped), plus `tests/test_dynamic_lfm2_system.py`,
`tests/test_dynamic_lfm2_adapter.py`, `tests/test_dynamic_qwen_adapter.py`,
`tests/test_dynamic_neo_native_vision.py`,
`tests/test_dynamic_lfm2_hybrid_catalog.py`,
`tests/test_dynamic_production_catalog.py`, `tests/test_hf_arch_layout.py`
(52 passed).

**Pre-existing failure, not caused by this wave:** `tests/test_dynamic_search.py`
reports `29 failed, 13 passed` both with and without this wave's
`dynamic_fold_planner.py` / `dynamic_causal_operator_promotion.py` changes
stashed (verified by `git stash push` on exactly those two files, re-running,
and `git stash pop`).
