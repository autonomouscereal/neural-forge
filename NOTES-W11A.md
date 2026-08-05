# NOTES-W11A — the chimera attention-family fold kernels (KDA, CSA/HCA, windowed NoPE)

Design contract: `docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`
Target platform: `C:/Users/cereal/Documents/ai training/chimera-omni` (read-only, not vendored)
Branch: `codex/lrc-blank-compression`. Started from `a9abb688`; a parallel agent
committed `9b25032c` and `01bea537` underneath during this wave, and the final
suite run below is against that HEAD with their uncommitted
planner/promotion/registry edits present in the tree. Nothing here is committed.

Stages 0–2 of the design's ladder are built and green. Stages 3–5 (real LFM2,
real SmolLM2, trained chimera, Kimi-Linear) are untouched: they need sealed
real-model caches and are explicitly out of this wave.

## What shipped

| file | what |
|---|---|
| `src/neural_forge/hf_causal_kda_fold.py` (new, 2321 lines) | the KDA fold: exact effective-weight unroll, barycentric head contraction, balanced-ranked channel-aligned selection, exact SVD decay rank, inverse-chain decay/beta/gate ridges, installed-recurrence replay, endpoint-coupled writer |
| `src/neural_forge/hf_causal_sparse_latent_fold.py` (new, 1534 lines) | the CSA/HCA fold, one stride-parameterized module: exact sub-branch split, extended GQA kernel for the local branch, score-ridge/probability-replay/endpoint V–O for the summary branch, bounded alternation, exact three-way translation split |
| `src/neural_forge/hf_causal_gqa_attention_fold.py` (extended, additive) | `attention_window` and `position_encoding` policy fields |
| `tests/test_hf_causal_kda_fold.py` (new, 21 tests) | stage 0/1 KDA acceptance |
| `tests/test_hf_causal_sparse_latent_fold.py` (new, 24 tests) | stage 0/1 CSA/HCA acceptance |
| `tests/test_hf_causal_windowed_nope_attention_fold.py` (new, 10 tests) | window/NoPE flags plus the byte-for-byte default reproduction |

Every parent in every test is a **real** `chimera_omni.modeling.layers` module
built through chimera-omni's own config API and imported from the checkout via
`experiments/run_chimera_omni_toy_fold.py`'s existing resolver. Nothing from
that repository is vendored, copied, or modified.

## The additive GQA flags — defaults are byte-identical

`attention_window: int | None = None` threads a band into the same site that
builds the causal tril (`hf_causal_gqa_attention_fold.py:334-347`).
`position_encoding: "rope" | "nope"` makes `apply_rope`/`invert_rope` identity
passthroughs, swaps RoPE pair selection for plain channel selection, and drops
the even-head-dim requirement (which exists only to pair rotations).

Both are **omitted from `as_dict()` and from the receipt when they sit at their
defaults**, the same discipline the Q/K-norm lane uses. The proof that nothing
moved is the pre-existing golden-hash suite: `tests/test_hf_causal_gqa_qk_norm_fold.py`
carries `attention_report_sha256` captured at commit `29939aea` and still
passes. `test_a_default_topology_serializes_to_its_pre_lane_bytes` and
`test_a_default_receipt_carries_no_window_or_encoding_keys` assert the same
thing directly.

Changing the window or the position-encoding class between source and target
now **fails closed** inside the kernel: band restriction and NoPE↔RoPE are kind
translations (design verdict matrix), not folds.

## Numbers

All at toy scale, chimera random initialization, CPU, FP64 solves. `val` is the
held-out endpoint relative RMSE; the no-branch baseline is 1.0 by construction
(predicting zero delta), so `val < 1.0` is the design's stage acceptance and
`energy` is the retained validation energy fraction.

### KDA — `hidden 32, 4 heads, Dk=Dv=8, decay_rank 4, conv K=4`, 8×24 rows, 18/6 fit/validation

| child geometry | val | val (BF16) | energy | child params |
|---|---|---|---|---|
| identity `4×8×8 r4` | **3.72e-07** | 0.0140 | 1.0000 | 6052 |
| decay rank 4→3 | 0.2224 | 0.2253 | 0.9506 | 5988 |
| Dv 8→6 | 0.3198 | 0.3203 | 0.8977 | 5236 |
| Dk 8→6 | 0.6406 | 0.6444 | 0.5897 | 5420 |
| Dk,Dv 8→6 | 0.6626 | 0.6659 | 0.5609 | 4604 |
| heads 4→3 | 0.7403 | 0.7404 | 0.4520 | 4571 |
| heads 4→3 + Dk,Dv→6 + r→3 | 0.8906 | 0.8917 | 0.2069 | 3435 |
| heads 4→2 | 0.9678 | 0.9677 | 0.0633 | 3090 |

Reading: the **value axis is cheap, the key axis is expensive, and head merging
is the expensive one** — exactly the ordering the design predicted when it
warned that merged recurrences do not commute and are refit, not conjugated.
At other seeds the 4→2 merge fails the BF16 no-branch gate outright and the
kernel refuses to emit a child; that refusal is the machinery working, not a
bug. The decay-rank cut is the only genuinely cheap structural cut (95%
retained for a 4→3 rank drop).

### CSA / HCA — `hidden 32, 4 q-heads, 2 kv-heads, d=8, window 8`, 12×32 rows, 22/10 fit/validation

| family / child | val | val (BF16) | energy | local sub-branch | summary sub-branch |
|---|---|---|---|---|---|
| CSA (stride 4) identity | **4.19e-06** | 0.0047 | 1.0000 | 1.11e-06 | 1.03e-05 |
| CSA heads 4→2, kv 2→1, d 8→6 | 0.8239 | 0.8237 | 0.3212 | 0.7432 | 1.4034 |
| CSA cross-width 32→16 under an orthonormal chart | 0.9143 | 0.9186 | 0.1641 | — | — |
| HCA (stride 16) identity | 6.06e-07 | 0.0045 | 1.0000 | 4.00e-07 | 1.54e-06 |
| HCA heads 4→2, kv 2→1, d 8→6 | 0.8960 | 0.8957 | 0.1972 | 0.7487 | 1.7956 |
| HCA cross-width 32→16 under a chart | **fails closed** (BF16 endpoint 1.0021 ≥ 1.0) | | | | |

Two honest readings. First, the summary sub-branch's own RMSE goes *above 1* in
the contracted folds: the bounded alternation deliberately trades mass between
the sub-branches to serve the shared endpoint, which is why the design demands
per-sub-branch energy be reported separately — it is, and it says exactly this.
Second, the HCA cross-width child is refused rather than emitted. On an HCA
layer 95.4% of the branch energy is the local sub-branch, and a simultaneous
width, head, and head-dim cut leaves the child worse than no branch at BF16.
The kernel fails closed with the measured numbers instead of installing it.

### The exact softmax-GQA → CSA/HCA three-way split (stage-2 comparability hook)

Source: an *unwindowed* causal softmax over the same chimera projections (the
translation's source is a global attention; splitting the branch's own already
windowed probabilities would put 100% in `rho_window` and prove nothing).

| stride | window | `rho_window` | `rho_far_blockmean` | `rho_far_deviation` | `rho_coverage_gap` | representable |
|---|---|---|---|---|---|---|
| 4 (CSA) | 8 | 0.8507 | 0.1463 | 0.0275 | 0.0000 | **0.9952** |
| 16 (HCA) | 8 | 0.8507 | 0.0795 | 0.0277 | **0.0738** | 0.9412 |

This is the design's "honest, promising" verdict quantified: with a stride no
wider than the window, 99.5% of a global attention delta is inside the CSA
operator class. Widening the stride past the window opens the predicted
coverage gap (7.4% here) — an architecture property, not a fold defect, and it
is reported as its own energy fraction rather than folded into the remainder.

### Exactness gates (all fail closed)

| gate | measured | tolerance |
|---|---|---|
| KDA effective-weight unroll vs. cached `o_t`, float32 parent | 7.90e-08 | 1e-06 |
| KDA unroll vs. cached `o_t`, float64 parent | < 1e-12 | 1e-12 |
| KDA unroll acausal leakage (`s > t`) | exactly 0.0 | — |
| KDA carried conv taps replay cached q/v rows | 6.59e-08 | 1e-05 |
| KDA decay inverse chain (probability space) | 4.87e-08 | 1e-05 |
| KDA decay-rank SVD truncation, full rank | 1.96e-08 | — |
| KDA state snapshots vs. parent, identity fold | 1.6e-07 at all 3 boundaries | drift-growth cap 4× |
| CSA/HCA `local + summary == sealed write` | 2.32e-08 | 1e-06 |
| CSA/HCA three-way split reconstructs the parent contraction | < 1e-12 | — |
| capture reconstruction vs. `module(x)` | KDA **0.0 exactly**, CSA 4.4e-08 | — |

The KDA capture is bit-exact because it steps the parent's *own*
`delta_rule_update`, `output_norm`, `output_gate` and `out` rather than
re-implementing them. The residual float32-epsilon numbers above are the
parent's own precision, not the kernel's: chimera normalizes q/k and computes
retention/beta in float32 even inside a float64 module (`layers.py:139-149`),
and it sums the two CSA sub-branches in float32 (`layers.py:228`).

### Zero-training counters

Every receipt from both kernels seals `training_steps = optimizer_steps =
backward_calls = parent_forward_calls = 0` and
`numeric_weight_matching_objective = false`, and each report self-seals
(`canonical_sha256` of the body equals `report_sha256`; asserted in both
suites). Evidence capture seals `parent_forward_calls = 1` — one forward, as
the contract allows; the *fold* performs none.

### Rollback

`HFCausalKDAFold.child_state_dict()` and `HFCausalSparseLatentFold.child_state_dict()`
emit exactly the chimera parameter names (`set(emitted) == set(child.state_dict())`
is asserted). Both suites install the emitted child into a real chimera module,
confirm the state digest changed, confirm the installed module reproduces the
kernel's own replay to < 1e-4, then restore the snapshot and assert the digest
is **bit-identical** to the pre-install digest. Both suites also assert the
*parent* module's digest is unchanged by compilation.

## Suites

```
.venv/Scripts/python.exe -m pytest \
  tests/test_hf_causal_gqa_qk_norm_fold.py \
  tests/test_cross_family_causal_fold_execution.py \
  tests/test_hf_causal_kind_translation.py \
  tests/test_hf_causal_kda_fold.py \
  tests/test_hf_causal_sparse_latent_fold.py \
  tests/test_hf_causal_windowed_nope_attention_fold.py -q
→ 91 passed (127s idle, 547s under load from the parallel agent's runs)

.venv/Scripts/python.exe -m pytest \
  tests/test_dynamic_causal_operator_promotion.py \
  tests/test_cross_width_causal_fold_execution.py -q
→ 27 passed in 49.33s

.venv/Scripts/python.exe -m ruff check <the six owned files>
→ All checks passed
```

Repo-wide `ruff check src tests` reports 33 errors, **all pre-existing and all
in files this wave does not own**: `hf_causal_kind_translation.py:419` (F841),
`hf_lfm2_weight_complete_basis.py:685` (F841), and E402/F401 in
`test_hf_lfm2_cached_heterogeneous_capacity.py`, `test_hf_logit_metrics.py`,
`test_hf_lfm2_crevf_checkpoint.py`. None were touched.

## For the coordinator — registry and promotion wiring (NOT done here)

`dynamic_fold_planner.py`, `dynamic_causal_operator_promotion.py`,
`dynamic_fold_registry.py`, `dynamic_model_inspector.py` and
`hf_causal_cross_width_charts.py` are owned elsewhere, so nothing was wired.
What the wiring needs:

1. **Method ids** (design section 4): `causal_kda_recurrent_layer`,
   `causal_summary_attention_layer`, and `causal_windowed_attention_layer` as a
   *policy variant* of the existing GQA method — the kernel is the same
   function, only the two topology fields change, so it does not need a new
   compile path.
2. **Branch-contract arms.** `RECURRENT_MIXER` needs an arm keyed on the KDA
   attributes (`heads, key_head_dim, value_head_dim, decay_rank,
   conv_kernel_size, gate_lower_bound`). CSA/HCA stays **one** composite
   `ATTENTION` node with `summary_stride`, `local_window`,
   `position_encoding = nope` — the current inspector emits *two* attention
   nodes plus a residual add for CSA/HCA
   (`docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md` §1.1), which contradicts the
   design's one-operator-per-branch contract (`dynamic_fold_planner.py:773-786`).
   **That discrepancy is unresolved and is the first thing to settle before
   planner wiring** — this kernel folds the composite as one branch with one
   residual write, matching the design, not the current IR.
3. **`FoldOperator` classes** following `_CausalBranchOperator`
   (bind/compile/snapshot/install/transaction). Both kernels already expose the
   two pieces such an operator needs: a capture function and a
   `child_state_dict()` in chimera parameter names.
4. **Charts.** KDA state is chart-invariant; only its residual-facing maps are
   consumers, and every consumer fit here is a ridge from child rows of any
   width, so no chart-specific code is needed — pass the chart-transported rows
   and endpoint. The CSA/HCA *summary* branch is the exception: its score is an
   inner product in the residual width, so it needs an explicit orthonormal
   `score_chart`; without one, a width change fails closed. Pooling commutes
   with the chart exactly (asserted:
   `test_the_pooling_operator_is_the_parents_and_commutes_with_a_chart`).
5. **AttnRes global-chart flag** (`global_chart_only`) in
   `hf_causal_cross_width_charts.py` is untouched and still required before any
   `residual_mode = "block_attnres"` child is folded per-boundary.

## Deviations from the design document

Each is also documented at the top of the module it applies to.

1. **Module name.** The design calls the second module
   `hf_causal_summary_attention_fold.py`; the work order named it
   `hf_causal_sparse_latent_fold.py`. Used the work order's name. Schema
   strings are the design's unchanged
   (`neural-forge-chimera-summary-attention-fold-v1`).
2. **Evidence dataclasses live in the two kernels**, not in a new
   `hf_chimera_circuit_evidence.py` (not owned this wave). They follow the
   `HFCausalLayerEvidence` field style and carry rows only; conv taps, decay
   bias/log-scale, head-norm gain and writer arrive in a separate sealed
   `ChimeraKDAParameters` carry, because they are *weights*. Fitting a step the
   design calls an exact carry out of activations would have made it
   approximate.
3. **The decay inverse chain is exact, not inverted.** The parent caches the
   pre-scale logits, so `u = exp(A_h) * z` identically. The chain is *checked*
   in probability space, where the comparison is well-conditioned; checking it
   in logit space measures the conditioning of the check (it read 1.6e-02 on a
   consistent cache) rather than the consistency of the cache.
4. **Alpha saturation gets its own threshold.** Beta and the gate are genuinely
   inverted, so the clamp policy gates them at 5% of rows. Alpha is not
   inverted, so its saturation is a parent property; it is reported and gated
   separately (`maximum_decay_saturated_row_fraction`, default 0.35, measured
   0.16% on the toy).
5. **Child `A_h` carries the parent's log-scale** (barycentric over a merged
   group) instead of being re-estimated from the target RMS. For a singleton
   group the child's linear target is then literally the parent's `z`, which is
   what makes the identity fold exact rather than exact-up-to-a-rescale.
6. **Conv taps are barycentric, not carried, when heads merge.** Exact carry is
   asserted for singleton groups (`torch.equal`, tested);
   `merged_*_conv_tap_dispersion` is sealed for merged ones.
7. **Balanced coordinates rank, a deterministic QDEIM cuts.** The balanced
   report's `canonical_pivot_indices` are sign-canonicalization pivots and can
   repeat, so selection is greedy row-pivoted Gram–Schmidt over the balanced
   decoder (the ShortConv channel-QDEIM discipline). The canonical pivots and
   the Hankel singular values are sealed beside the selection.
8. **State-drift superlinearity is measured against the horizon and a noise
   floor.** A drift ratio computed from a 1e-08 baseline measures float64
   rounding; the gate declares itself inapplicable below
   `state_drift_floor = 1e-3` and says so in the receipt
   (`state_drift_gate_applicable`).
9. **"Mass on assigned blocks" is defined** as the total-variation overlap
   `mean_t sum_j min(P_child, P_parent)`. Every masked replay trivially places
   all of its mass on *allowed* blocks, so the literal reading is vacuous.
10. **Merged-head dispersion defaults are loose enough to admit a chimera
    initialization.** `decay_log_scale` is drawn from `log(U(1,16))`, so two
    randomly paired heads routinely sit a relative 0.7 apart. The default is
    1.0 for decay and 0.5 for beta; the design says these are calibrated on the
    selection partition of a real parent, and they should be.
11. **Local-branch alternation refits by recompiling** the GQA fold against the
    updated target rather than reaching inside the kernel for a V/O-only refit.
    Q/K do not depend on the endpoint target, so only the writer moves.
    Monotonicity is *guarded* (apply-and-revert on the unregularized objective),
    not assumed.

## Deferred, with reasons

- **`compile_short_conv_to_kda_translation`, `compile_gqa_to_kda_translation`,
  `compile_kda_to_gqa_translation`, `compile_gqa_to_summary_attention_translation`**
  are precise fail-closed stubs citing the design document and their verdict.
  The closed-form primitives each is built from **are implemented and tested**:
  `constant_key_kda_pole` (the scalar-IIR collapse, with an `exact` flag
  because the single-pole reading is exact only when alpha is uniform on the
  key's support — the design states it unconditionally),
  `fir_taps_to_kda_poles` (Vandermonde pole match with conditioning cap and
  measured tail), `kda_effective_weights` (the exact signed weight matrix the
  KDA→GQA lane consumes), and `gqa_to_summary_three_way_split` (exact, with the
  coverage gap). What is missing in each case is the real-model hand-off —
  gate linearization, carrier routing, checkpoint gates — which needs sealed
  real caches and a downstream carrier, i.e. stages 3–4.
- **Gated MLA** is adjacent and out of scope by the design's own scoping.
- **`hf_chimera_circuit_evidence.py`, planner/promotion/registry wiring, and
  the `global_chart_only` chart flag** — not owned; see the coordinator section.
- **The IR discrepancy for CSA/HCA** (two attention nodes today vs. one
  composite node in the design) is flagged, not resolved.
- **No trained chimera parent** exists at nontrivial scale, so every number
  above proves machinery, not intelligence — the design says this explicitly
  and it stays true here.
