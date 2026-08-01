# NOTES-W4A — the dense/carrier lane consumes evidence sets, and the merge stops doubling

Author: wave W4A (dense-lane evidence-set consumption + streaming merge)
Date: 2026-07-31
Branch: `codex/lrc-blank-compression` (HEAD at start: `c69d5a64`)

Closes NOTES-W3C §6 items 1 and 2 for the lanes named below, and leaves the
rest of §6 exactly as it was.

Files changed:

- `experiments/run_lfm2_residual_attention_fold.py` — new `merge_trace_rows_into`
  entry point (`SealedEvidenceMergeMember`, `MergedTraceRows`,
  `MERGED_TRACE_ROWS_ACCOUNTING_SCHEMA`)
- `experiments/run_lfm2_dynamic_catalog.py`
- `src/neural_forge/hf_lfm2_sealed_evidence_set.py` (additive:
  `Lfm2MergedSealedEvidenceCache`, `Lfm2MergedSealedEvidenceSession`,
  `SEALED_EVIDENCE_CACHE_TYPES`, `SEALED_EVIDENCE_SESSION_TYPES`)
- `src/neural_forge/hf_lfm2_dense_budget_model_fold.py`
- `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`
- `src/neural_forge/dynamic_lfm2_system.py` (two small additive edits, §5)
- `src/neural_forge/hf_lfm2_child_closed_loop_refinement.py` (one type widening, §5)
- `tests/test_run_lfm2_residual_attention_fold.py`,
  `tests/test_hf_lfm2_sealed_evidence_set.py`,
  `tests/test_run_lfm2_dynamic_catalog.py`,
  `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` — all additive

Deliberately untouched: `hf_lfm2_circuit_trace_cache.py`,
`dynamic_closed_loop_fold.py`, every `hf_causal_*`,
`dynamic_causal_operator_promotion.py`, `docs/`,
`hf_lfm2_on_policy_evidence_round.py`, `hf_lfm2_tokenizer_bridge_trace_adapter.py`,
`hf_lfm2_cached_heterogeneous_capacity.py`,
`hf_lfm2_cached_global_topology_frontier.py`.

`src/neural_forge/dynamic_causal_operator_promotion.py` and
`src/neural_forge/dynamic_fold_planner.py` already carried uncommitted working-tree
changes when this wave started. Neither was touched here.

---

## 1. Two merged read-only views instead of one sealed handle

`Lfm2SealedEvidenceSet` gained two memoized views, both built only when the set
actually has an extension member:

* `merged_cache()` → `Lfm2MergedSealedEvidenceCache`. It exposes the sealed
  cache surface a closed-form writer uses — `cache_key`, `parent_fingerprint`,
  `corpus_fingerprint`, `implementation_fingerprint`,
  `layer_topology_fingerprint`, `batch_count`, `manifest`, `load_batch`,
  `iter_batches`, `verify` — over the merged batch order. `load_batch(i)`
  routes to the member that sealed batch `i`, compares the loaded batch's
  content hash against the set's own inventory, and restates only the batch
  number in merged coordinates.
* `merged_session()` → `Lfm2MergedSealedEvidenceSession`. Every selection is
  routed to the member session that authenticated it, so the per-file hash
  passes, the pinned descriptors, and the per-load descriptor comparison stay
  the member's own. The view **owns nothing and closes nothing**; the evidence
  set keeps owning every member session's lifetime, and `close()` is a no-op.

Neither is an `Lfm2CircuitTraceCache` and neither pretends to be. A merged
corpus has no single sealed directory, and subclassing the sealed handle would
let a consumer read files from a path whose manifest it never authenticated.
The two type tuples `SEALED_EVIDENCE_CACHE_TYPES` /
`SEALED_EVIDENCE_SESSION_TYPES` are the single place a writer states "one
sealed cache or one merged evidence-set view".

`merged_manifest()` is the base manifest with `corpus` replaced by the merged
corpus body and `cache_key` replaced by the evidence-set key. The sealed
`manifest_sha256` is **dropped**, not restated: no file on disk carries that
body, and restating a digest nobody can verify is the kind of small lie this
tree exists to avoid. A `merged_sealed_evidence_set` block records the member
keys, the on-policy evidence identity, and the extension batch indices.

`extension_batch_indices` is new on both the set and the merged views. A
consumer whose fit/validation split is derived from trajectory identity rather
than from sealed split labels needs it, which is exactly the closed-loop
adapter's situation (§3).

## 2. `merge_trace_rows_into` — one allocation, filled member by member

NOTES-W3C §6.2 recorded the fix as a "merge into" entry point in
`run_lfm2_residual_attention_fold.py`, because `TraceRowIndex.layer_fields` is a
`MappingProxyType` and the source cannot be released field by field from
outside. That entry point now exists:

```python
merge_trace_rows_into(
    members,                       # ordered SealedEvidenceMergeMember values
    cache_key=...,                 # the evidence-set key
    corpus_fingerprint=...,        # the evidence-set merged corpus digest
    merged_trace_rows_sha256=...,  # caller-supplied seal over member digests
) -> MergedTraceRows
```

It never materializes a member index and then joins it. It plans the exact
final storage for **every** member from authenticated cache metadata only
(`_plan_preallocated_trace_storage`, no payload loads), allocates each merged
per-layer field exactly once at the merged row count, and streams every
member's authenticated batches straight into their final row slices through
`_iter_trace_tensor_batches`, which releases each selected payload as it is
consumed. `torch.cat` is never called on circuit rows.

`MergedTraceRows.member_rows` are **zero-copy narrow views** into that single
allocation, each carrying its own member's standalone identity — cache key,
corpus fingerprint, unrenumbered paths, splits, and a `trace_rows_sha256`
recomputed over its own slice that reproduces the digest an independent
`extract_trace_rows` on that member would seal. So the merged index can still
name what produced each row, and the runner can hand the base corpus alone to a
consumer without retaining a second copy (§4).

The merged identity is **not** sealed inside this module. The caller supplies
`merged_trace_rows_sha256(member_rows, paths, splits)`; the runner passes
`_sealed_evidence_trace_rows_sha256(evidence_set)`, whose body is the same
`SEALED_EVIDENCE_TRACE_ROWS_SCHEMA` document W3C sealed, so the merged
`parent_trace_rows_sha256` is unchanged in shape.

### Peak accounting, recorded

`MergedTraceRows.accounting` (`MERGED_TRACE_ROWS_ACCOUNTING_SCHEMA`) is derived
from the sealed tensor descriptors, not estimated:

| field | meaning |
|---|---|
| `retained_bytes` | the merged payload actually held |
| `maximum_transient_selection_bytes` | the largest single-batch selected-tensor set across every member |
| `streaming_merge_peak_bytes` | `retained + transient` — what this merge holds |
| `separate_materialization_peak_bytes` | `2 × retained` — what the discarded per-member `torch.cat` join held |
| `peak_bytes_avoided` | `retained − transient` |
| `peak_bytes_ratio_to_retained` | `(retained + transient) / retained` |
| `preallocated_once`, `member_sources_released_as_consumed`, `member_rows_are_zero_copy_views`, `parent_forward_calls` | the invariants, stated |

The runner brackets the call with `_process_memory_progress_metadata()` and adds
`before_/after_process_rss_bytes` plus `observed_process_rss_growth_bytes`, so
the receipt carries a *measured* resident-set reading next to the predicted
accounting instead of only an assertion about intent. All of it lands in the
`on_policy_extension_evidence_materialized` progress stage (prefixed `merge_`)
and in the receipt as `parent.on_policy_merged_row_accounting`.

On the exact-4x lane's ~25 GiB of retained rows, `separate_materialization_peak`
is ~50 GiB and `streaming_merge_peak` is ~25 GiB plus one batch's selected
tensors.

## 3. What now consumes base+extension evidence

`Lfm2DenseBudgetModelOperator`, `Lfm2DenseBudgetEvidenceProvider`, and
`Lfm2DynamicClosedLoopAdapter` each grew one additive `evidence_set=`
parameter. With no evidence set, or a base-only one, every handle stays the
caller's own and the configuration is byte-identical. With extension members
the memoized merged views replace the pair, which is why the provider's sealed
evidence and the operator's trace cache still compare `is`-identical.

The adapter is where the hard rule lives. Its fit/validation split does **not**
come from sealed split labels: `ChildClosedLoopCompiler` calls
`deterministic_whole_trajectory_split` over whatever
`parent_trajectory_records()` returns and holds a hash-ranked fraction of it
out. If extension batches were simply appended to that list, roughly a fifth of
every child-visited state would become a held-out validation row. So:

* `parent_trajectory_records()` offers **only the teacher-forced batches**. The
  compiler's split therefore ranks exactly the corpus it ranked before, and the
  sealed `split_sha256` — which every restart capsule is bound to — is
  unchanged in shape.
* `_split()` unions `self._always_fit_batch_indices` (read from the merged
  cache's `extension_batch_indices`) into the fit side before taking the
  complement, so an extension batch can never land in validation no matter how
  the generic split ranked the base corpus. With no extension evidence that set
  is empty and `_split` is byte-identical.
* `self._records` still covers every merged batch, because
  `payload.fit_batch_indices` and the final-normalization validator index it by
  merged batch index.

`dynamic_lfm2_system` resolves the merged handles from
`payload.evidence_set` so the stack evidence fingerprint (the merged evidence
identity) and the handle the dense writers bind stay consistent.

## 4. Selection stays base-only, on purpose

The cached heterogeneous capacity option bank and the global topology frontier
are **not** wired to the merged evidence, and that is the correct answer rather
than a gap. They are *selection* consumers: they choose the child's own
topology. A child-visited row is a fit row precisely because selection would
let the child pick its own frontier. Feeding rollout evidence into the capacity
search is that capture.

So `ProductionParentCachePayload` gained `selection_rows` (exposed as
`base_rows`), which `build_production_catalog_wiring` hands to
`retained_parent_rows`. With extension evidence it is
`MergedTraceRows.base_rows` — the zero-copy base prefix carrying the sealed base
cache identity, so `Lfm2RetainedTraceRowAdapter.build` authenticates it against
the base cache and the base session exactly as before, at no extra memory.
`payload.verify()` now also proves the base rows are the sealed base prefix of
the merged rows.

Their derived phase keys still move with the evidence identity — the
global-topology phase context is built from `resolved.evidence_split_sha256`,
which folds in the on-policy identity — so a base-only selection is never
silently reused under a different evidence request.

## 5. The two edits outside the owned set

Both are type widenings with no behavior change when no evidence set is
present:

1. `dynamic_lfm2_system.Lfm2DynamicFoldStackAdapter` — new `_dense_evidence_set()`
   helper (`getattr(payload, "evidence_set", None)`, `None` unless it has
   extension members), `_dense_parent_trace_cache` resolves the merged view and
   checks the base handle belongs to the set, and `build()` passes
   `evidence_set=` plus the merged session to the provider and operator. A
   payload without the attribute behaves exactly as before.
2. `hf_lfm2_child_closed_loop_refinement.pair_lfm2_parent_child_replay` —
   `isinstance(parent_cache, Lfm2CircuitTraceCache)` widened to
   `SEALED_EVIDENCE_CACHE_TYPES`. Everything it then uses (`len`,
   `corpus_fingerprint`, `load_batch`, batch `index`/`input_sha256`) is on the
   merged view.

## 6. Lanes that still fail closed, and why

`_unconsumed_extension_evidence_reason` now names four:

| lane | why it cannot consume |
|---|---|
| `--hybrid-model-fold` | binds one sealed parent cache key |
| `--producer-first-model-fold` | `hf_lfm2_producer_first_model_fold` asserts `parent_rows.cache_key` equals the sealed parent cache key |
| strict tokenizer bridge (and therefore the routed carrier lane) | the sealed bridge trace adapter pairs exactly one target-token record with every sealed parent batch; an on-policy round re-tokenizes the child's decoded text with the **parent** tokenizer, so its sealed batches have no target-token preimage and no child replay can be built for them |
| `--dense-macro-cell-attractor-gate`, `--dense-vocabulary-margin-candidate-gate` | both bind one sealed cache handle |

The bridge gate is the one that matters and it is not a wiring gap: closing it
means teaching `run_lfm2_on_policy_evidence_round` to tokenize the rollout with
the *target* tokenizer and map through the transport profile, so the sealed
extension batch is the bridge image of a real target-token record. That module
is outside this wave. The gate is re-asserted at the dense operator
(`Lfm2DenseBudgetFoldError`, same words) as well as at configuration time, so a
caller that bypasses the runner still fails closed.

**Consequence for the production carrier lane (r12 and successors): the
alternation is available on its dense core — same H1024 parent, same exact-4x
dense budget, same heterogeneous layout with the global hidden frontier and
factor-writer rank discovery, same moving_gauge_v3 closed loop — and is still
refused with the bridge on.**

## 7. Verification

Commands (repo root, `-p no:randomly`, `.venv` interpreter):

```
-m pytest tests/test_run_lfm2_dynamic_catalog.py \
    tests/test_hf_lfm2_sealed_evidence_set.py \
    tests/test_hf_lfm2_on_policy_evidence_round.py -q -p no:randomly
```
→ `141 passed, 1 skipped` (~60 s)

```
-m pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly
```
→ `6 passed` (~221 s), including the two new tests

```
-m pytest tests/test_dynamic_lfm2_dense_budget_fold_stack.py \
    tests/test_dynamic_lfm2_cpcf_production.py \
    tests/test_hf_lfm2_dynamic_closed_loop_adapter.py \
    tests/test_hf_lfm2_circuit_trace_cache.py -q -p no:randomly
```
→ `132 passed, 1 skipped` (~242 s)

```
-m pytest tests/test_run_lfm2_residual_attention_fold.py \
    tests/test_hf_lfm2_child_closed_loop_refinement.py \
    tests/test_hf_lfm2_child_closed_loop_token_bridge.py \
    tests/test_hf_lfm2_dense_budget_model_fold_heterogeneous.py \
    tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py -q -p no:randomly
```
→ `52 passed` before the merge tests were added, `19 passed` for the residual
module alone afterwards

```
-m pytest tests/test_run_lfm2_dynamic_catalog_portable_runtime.py \
    tests/test_run_lfm2_dynamic_catalog_tokenizer_bridge.py \
    tests/test_dynamic_lfm2_producer_first_catalog.py \
    tests/test_prepare_lfm2_tokenizer_bridge_evidence.py \
    tests/test_run_lfm2_fused_attention_diagnostic.py -q -p no:randomly
```
→ `50 passed` (~18 s)

### Measured no-extension byte-identity

The tiny dense lane (`_tiny_dense_lane_arguments`, no extension evidence) was
run at `c69d5a64` and on this working tree against the *same* pack and the same
run root, and the two receipts compared field by field:

* `configuration_sha256` **identical** — so the fold identity, every derived
  phase-cache key, and the derived restart-capsule root
  (`<cache-root>/closed-loop-restart-capsules/<child>-<run-scope>`) are
  unchanged;
* the only semantic difference in the whole receipt is
  `on_policy_evidence_policy.extension_evidence_consumer_gap`, which moves from
  the old dense-lane refusal string to `null` because the lane is now supported.
  That field is reported policy; it is not in `config.identity()`;
* `run_fingerprint`, `certificate_sha256`, `receipt_sha256`,
  `provenance_run_sha256`, and `provenance_tree_sha256` differ between *any two
  runs at the same revision* (verified by running this tree twice), so they are
  run-nondeterministic, not a regression;
* `saver_/loader_implementation_fingerprint` and their resource fingerprints move
  because they are source-content digests over the runner module. Any edit to
  `run_lfm2_dynamic_catalog.py` moves them by design; `988cb7fc` already did.

### New coverage

* `tests/test_run_lfm2_residual_attention_fold.py` (3 tests, real sealed toy
  caches): the streamed merge equals the separate join tensor for tensor and
  reproduces each member's standalone `trace_rows_sha256`; the merge is proven
  to never concatenate by making `torch.cat` raise for the duration of the call,
  with every member view asserted to share the merged allocation's storage
  pointer and to sit at the right storage offset, plus the full accounting
  contract; fail-closed on a wrong batch offset, a short split assignment, a
  missing session, and a non-digest merged seal.
* `tests/test_hf_lfm2_sealed_evidence_set.py` (3 tests): the merged views are
  memoized and agree on identity/manifest/batch inventory; a selection through
  the merged session is byte-equal to the member's own and is released back
  through the member; the merged views are refused outright for a base-only set.
* `tests/test_run_lfm2_dynamic_catalog.py`: the lane-gate test now asserts the
  dense budget + heterogeneous lane consumes, and that producer-first, the
  bridge, the macro-attractor gate, and the vocabulary-margin candidate gate
  still refuse with their stated reasons.
* `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`:
  `test_bridged_carrier_lane_refuses_extension_evidence_with_a_reason` and
  `test_carrier_dense_core_alternates_fold_round_and_refold` (~80 s). The
  alternation e2e is monkeypatch-light — the only patches are two pure observers
  (a counting forward hook around `_load_cache_bound_parent` and a sampler
  around `run_lfm2_on_policy_evidence_round`); the fold, the round, and the
  refold are all real. It asserts publication → real sealed round → refold on
  base+extension → second publication with both passes `complete`; the extension
  contributes only fit batches
  (`extension_selection_or_validation_batch_count == 0`); the merge preallocated
  once and its streaming peak is strictly below the separate-materialization
  peak with `peak_bytes_ratio_to_retained < 1.5`; `evidence_split_sha256`,
  `parent_trace_rows_sha256`, and the published `evidence_fingerprint` all move
  while `parent_cache_key` does not; zero parent forwards outside the round's
  single sanctioned capture, measured with the hook rather than read from a
  receipt field; both children reload standalone at the exact counted budget;
  each pass journals its own ordered progress into one shared ledger.

## 8. Deferred / known limitations

1. **The bridged carrier lane still cannot consume extension evidence** (§6).
   The fix is in `hf_lfm2_on_policy_evidence_round`, not here: the round must
   tokenize the rollout with the target tokenizer and map through the transport
   profile so every sealed extension batch is the bridge image of a real
   target-token record. Until then the routed exact-4x lane can *produce* rounds
   and cannot *consume* them.
2. **`circuit_balanced` + the balanced-basis phase cache are not exercised by
   the new e2e.** The toy carrier lane can only reach `weight_complete`
   (NOTES-1E blocker 1 pins the routed configuration out of reach), so the
   on-policy balanced-basis phase dependency is still only covered by the unit
   test `test_on_policy_phase_dependency_is_added_only_when_evidence_exists`.
3. **The base payload session is no longer deferred when extension evidence is
   consumed.** The merged views are built from every member's authenticated
   session, base included, so `resolve_production_parent_cache` skips the
   deferral in that case. A base-only run still defers exactly as before.
4. **`supports()` in `dynamic_lfm2_system` still degrades a bridged+extension
   target to "unsupported" rather than reporting the reason**, because it
   catches `DynamicLfm2AdapterError`. Unreachable from the runner (the config
   gate fires first), but a direct stack caller would see a vague message.
5. **NOTES-W3C §6 items 3 through 7 are unchanged.** Alternation depth > 1 is
   still untested, the toy refold still produces a byte-identical child because
   the endpoint correction is null at that scale, `--on-policy-child-artifact-directory`
   is still refused with an alternation, round manifests are still consumed by
   absolute path, and the alternation summary is still not a receipt.
6. **The merged manifest is not a sealed document.** It is derived per call from
   the members' own sealed manifests and deliberately carries no
   `manifest_sha256`. Anything that wants to persist it must seal it under its
   own schema.
7. **`observed_process_rss_growth_bytes` is best effort.** It reads
   `/proc/self/status` or `psutil`; on a host with neither, only the
   descriptor-derived accounting is recorded, and the e2e asserts only the
   descriptor-derived fields.
