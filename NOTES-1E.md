# NOTES-1E — toy-scale dress rehearsal for the exact4x integrated-carrier lane

Author: wave 1E (carrier-lane end-to-end test)
Date: 2026-07-31
Status: **test only — no runner, adapter, or `src/` file was modified by this wave**

New file: `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`.

It builds every input an `nf-launch-integrated-carrier-exact4x-r*.sh` run needs,
at toy scale on CPU, and drives `run_production_catalog` — the same entrypoint
`main()` calls — with the r10 flag set:

* real ByteLevel-BPE donor tokenizer package (280 rows) and a 284-row parent,
* real strict tokenizer bridge compiled by
  `experiments/compile_lfm2_target_tokenizer_bridge.compile_target_tokenizer`
  (272 target rows, one-row donor transport),
* target-canonical calibration + one-row-mapped donor calibration,
* one sealed `Lfm2CircuitTraceCache` captured on the mapped donor IDs,
* the sealed canonical-parent and mapped-parent whole-output envelopes,
* a sealed routed manual-review prompt artifact,
* a real 2-layer `Lfm2ForCausalLM` parent (8,032,384 parameters).

Whole file: **~60 s on CPU** (three tests, one shared module-scoped pack).

Verified against a clean export of `9c53a6e8` (`git archive HEAD`), because the
live working tree was being edited by other waves while this was written:

```
PYTHONPATH="<snapshot>/src;<snapshot>" .venv/Scripts/python.exe -m pytest \
    tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q
3 passed in 60.45s
```

## Reached depth at HEAD

160 progress-ledger rows. The rehearsal executes the complete fold:

`run_started` → `tokenizer_bridge_complete` → `carrier_verified` →
global topology capacity/functional/selection phases (cold, then published to
the derived phase cache) → `parent_rows_materialized` → `wiring_planned` →
`system_fold_started` → `dense_structural_short_conv_action_evidence_completed`
→ `dense_structural_token_transport_completed` → `dense_structural_fold_completed`
→ per-layer `child_closed_loop_layer_{producer,consumer,downstream}_installed` →
`child_closed_loop_body_reclosure_{started,installed,completed}` with
`moving_gauge_{preview,install}_recapture_*` → `child_closed_loop_output_interface_completed`
→ `dense_closed_loop_completed` → **first trajectory gate, where blocker 4 stops it**.

That is exactly the window run r7 died in (`fold_progress_sequence 75`), and the
test asserts the ledger is strictly sequenced and scalar-only across all of it.

## Blockers

### 1. The routed configuration is pinned to the real 4096-row H816 child (config time)

`experiments/run_lfm2_dynamic_catalog.py`

* line 279-283 — `LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS = 57_420_144`,
  `LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS = 53_552_368`,
  `LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE = 1_933_888`,
  `LFM2_CARRIER_TARGET_ROWS = 4_096`, `LFM2_CARRIER_HIDDEN_SIZE = 816`.
* line 1737-1751 — carrier mode forces `required_target_hidden_size == 816`.
* line 1802-1825 — the counted total, the dense body budget, and
  `carrier_expected_{target_rows,hidden_size}` must equal those constants exactly.
* line 1826-1837 — routed manual review must use 1024/2048 new tokens.
* `src/neural_forge/lfm2_carrier_child_package.py:46-49` —
  `current_exact4x_counted_plan()` is bound to the 229,693,184-element parent, and
  `experiments/run_lfm2_dynamic_catalog.py:762-767` rejects any other parent.

**Effect:** no toy routed child exists. A rehearsal cannot fold 53.5 M dense
parameters on CPU in minutes, and cannot present a 229.7 M-parameter parent.

**Minimal change to make it rehearsable:** make the five ABI numbers a named
profile (for example `CarrierRouteProfile`) selected by a
`--carrier-route-profile` flag, defaulting to the current
`current_exact4x` values, plus one `toy` profile whose numbers come from the
supplied publication. `current_exact4x_counted_plan()` needs the same treatment
(a `counted_plan(parent_parameter_elements=..., target_parameter_elements=...)`
call already exists on the registration, so only the *current* wrapper is pinned).

The test asserts both config-time rejections (`equal carrier_expected_hidden_size`
and `reserve both independent route channels`) so the pins stay documented, then
runs the derived H816 carrier configuration and proves the run fails closed before
`carrier_verified`, `system_fold_started`, and `manual_review_started`.

### 2. The routed manual-review dossier is unreachable at toy scale

`experiments/run_lfm2_dynamic_catalog.py:9272-9509`. The dossier, the routed smoke
coverage, the counted-state verification of the published child, and the
`standalone_child_authenticated` / `manual_review_{started,completed}` stages all
sit behind `Lfm2CarrierChildPackageRegistration.current_exact4x(...).verify()`,
which needs a real authenticated carrier publication with two 1,933,888-element
channels bound to a 4096-row H816 child. Blocker 1 is the prerequisite.

**Covered instead:** `test_toy_carrier_lane_counts_route_state_inside_the_target`
runs the real counted-state verifier
(`build_counted_child_state_plan` / `verify_counted_child_state`) over two
independent toy route channels sized `4 * H * target_rows`, proving the routes are
counted inside the target, that the realized compaction ratio is strictly worse
than the dense-body-only ratio, and that dropping a channel or crediting the
routes to the dense body is rejected.

### 3. The global-frontier operator catalogs are not configurable

`experiments/run_lfm2_dynamic_catalog.py:7170-7196` builds the frontier without
passing `short_conv_internal_widths` or `attention_head_options`, so
`src/neural_forge/hf_lfm2_cached_global_topology_frontier.py:2156-2161` defaults
apply: ShortConv internal widths `(704, 1024)` and GQA options
`(12, 6, 64)` / `(16, 8, 64)`. `hf_lfm2_cached_heterogeneous_capacity.py:1365`
then requires every candidate geometry to "preserve head_dim and remain a legal
source-subset GQA topology".

**Effect:** the parent of any rehearsal must keep head_dim 64, ≥16 query / ≥8
key-value heads, and a residual width that can host a 704/1024-wide ShortConv.
A hidden-8 or hidden-32 parent (the shape the existing tiny dense e2e uses) has
no legal child at all.

**Worked around, not blocked:** the test's parent keeps the production residual
width (1024), head geometry (16/8/64), and conv kernel (3), and shrinks only
depth (2 layers), vocabulary (284 rows), and MLP width (64). The child is H32.
Making the two catalogs CLI-configurable would let a genuinely tiny parent
rehearse the same lane.

### 4. Bridge-aligned parent rows are discarded before the span-mapped gate — **live defect**

`experiments/run_lfm2_dynamic_catalog.py:5611-5650`.
`CachedTrajectoryEvaluator._evaluate_model` computes `trajectory_parent_rows` from
`_bridge_aligned_parent_rows(...)` (which rebinds the donor-ID parent rows onto the
target-ID child's paths and corpus fingerprint), and then, when the child config
carries the dense span markers, overwrites it with

```python
build_span_mapped_parent_trajectory_rows(
    self.payload.rows,        # <- unaligned donor-corpus rows
    rows,
    ...
)
```

`build_span_mapped_parent_trajectory_rows` (line 5215-5222) requires
`parent_rows.corpus_fingerprint == child_rows.corpus_fingerprint`, so every
bridge run whose child carries `_neural_forge_dense_ordered_source_spans` fails
closed with:

```
Parent and hybrid-child trajectory paths, splits, or corpus differ.
```

Observed at toy scale: `paths_equal=True`, `splits_equal=True`,
`corpus_equal=False` (donor `02ff0d2e…` vs target `2a86ed39…`).

Those markers are written for **every** dense-budgeted child
(`src/neural_forge/dynamic_lfm2_adapter.py:4666-4690`), regardless of depth
compaction, so this is on the carrier lane's path. Runs r1-r10 never reached it —
they died earlier, in evidence or basis phases.

**Minimal change:** pass the bridge-aligned rows into the span mapping, i.e.
`build_span_mapped_parent_trajectory_rows(trajectory_parent_rows, rows, ...)`.
Everything else (`assert_aligned`, the projected trace-rows digest) already
works on the rebound index.

### 5. The per-block FP64 Gram must be bitwise symmetric — **live defect**

`src/neural_forge/hf_lfm2_balanced_circuit_basis.py:1576-1585` computes

```python
contribution = (block.rows.T @ block.rows * block.gram_weight).contiguous()
```

and `src/neural_forge/dynamic_balanced_circuit_basis.py:640-648` rejects it unless
`torch.equal(contribution, contribution.T)`. A threaded CPU FP64 GEMM does not
guarantee bitwise symmetry: at toy scale, rows `(12, 1024)` produced a maximum
asymmetry of `3.469446951953614e-18` (one ULP) and the receipt raised

```
Snapshot contribution receipt requires its exact symmetric FP64 Gram.
```

`src/neural_forge/dynamic_balanced_circuit_basis.py:1589-1592` has the same
producer pattern. Both aggregates already symmetrize afterwards
(`raw = ((raw + raw.T) * 0.5)`), so only the per-block receipt is strict.

**Effect:** `--dense-residual-basis-mode circuit_balanced` — the default, and what
every carrier launcher passes — cannot complete on this CPU. Because
`--dense-vocabulary-margin-observability` and
`--dense-streaming-vocabulary-margin-v2` require `circuit_balanced`
(`experiments/run_lfm2_dynamic_catalog.py:2114`), the reachable rehearsal drops all
three flags and uses `weight_complete`.

**Minimal change:** symmetrize the block contribution before sealing
(`contribution = ((g + g.T) * 0.5).contiguous()`) or compare within an FP64
tolerance in `_snapshot_block_report_and_receipt`.

### 6. `prepare_lfm2_tokenizer_bridge_evidence.py` cannot build a pack at HEAD — **live defect**

`experiments/prepare_lfm2_tokenizer_bridge_evidence.py:910-930`
(`_normalization_config`) builds a `SimpleNamespace` that has no
`final_performance_metrics_diagnostic_only`, but
`experiments/run_lfm2_dynamic_catalog.py:4299-4315`
(`_load_bridge_mapped_parent_baseline`) now reads that attribute:

```
AttributeError: 'types.SimpleNamespace' object has no attribute
'final_performance_metrics_diagnostic_only'
```

Any invocation of `prepare_bridge_evidence` therefore dies after it has already
captured the parent trace cache. The existing production packs predate this drift.

**Minimal change:** add
`final_performance_metrics_diagnostic_only=<bool>` (and a matching
`BridgeEvidencePreparationConfig` field, defaulting to the value the sealed pack
should record) to `_normalization_config`.

**Worked around:** the test seals its own pack from the same primitives the prep
script uses (`_target_tokenized_records`,
`build_lfm2_strict_tokenizer_bridge_trace_adapter`,
`build_lfm2_circuit_trace_cache`, `seal_lfm2_cached_whole_output_envelope`,
`seal_lfm2_tokenizer_bridge_mapped_parent_envelope`,
`resolve_production_tokenizer_bridge`), so the runner still authenticates every
digest end to end.

### 7. Working-tree state while this wave ran (not a HEAD defect)

Two transient breakages from parallel waves were observed in the dirty tree and
are recorded only so the next reader does not chase them:

* `src/neural_forge/hf_lfm2_circuit_trace_cache.py` renamed `_tensor_bytes` to
  `_update_tensor_bytes`, while
  `src/neural_forge/hf_lfm2_cached_global_topology_frontier.py:1048` still binds
  `module._tensor_bytes` for its implementation fingerprint →
  `AttributeError: module 'neural_forge.hf_lfm2_circuit_trace_cache' has no
  attribute '_tensor_bytes'`, which also fails
  `tests/test_hf_lfm2_cached_global_topology_frontier.py`.
* the same file briefly carried a `MappingProxyType({})` dataclass default
  (`Lfm2ChildReplay._capture_timing`), which is rejected outright by the
  Python 3.11 interpreter in `.venv`.

Neither exists at `9c53a6e8`; both were reproduced only against the uncommitted
tree.

## How the test behaves when a blocker is fixed

`test_toy_carrier_lane_folds_through_every_reachable_stage` asserts the ledger,
the bridge receipt, and the zero-training invariants unconditionally, then
branches: `status == "complete"` adds the publication, fresh-reload, and exact
parameter assertions; otherwise the failure must be one of the two messages in
`_KNOWN_TOY_SCALE_BLOCKERS` (blockers 4 and 5). Fixing blocker 4 flips the test
into the publication branch with no edit. A *different* failure fails the test
with the real message, which is the point of the rehearsal.
