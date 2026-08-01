# NOTES-W3C — consuming base+extension evidence and alternating fold with round

Author: wave W3C (on-policy evidence consumption + fold/round alternation)
Date: 2026-07-31
Branch: `codex/lrc-blank-compression` (HEAD at start: `29939aea`)

Scope of files changed:

- `src/neural_forge/hf_lfm2_sealed_evidence_set.py` (new)
- `experiments/run_lfm2_dynamic_catalog.py`
- `tests/test_hf_lfm2_sealed_evidence_set.py` (new)
- `tests/test_run_lfm2_dynamic_catalog.py` (additive)

Deliberately untouched: `hf_lfm2_circuit_trace_cache.py`,
`hf_lfm2_dynamic_closed_loop_adapter.py`, `hf_lfm2_dense_budget_model_fold.py`,
`dynamic_closed_loop_fold.py`, `dynamic_lfm2_system.py`, every `dynamic_*`
registry/planner/inspector, and `experiments/run_lfm2_residual_attention_fold.py`
(its public `TraceRowIndex`, `extract_trace_rows`, `calibration_batches_from_cache`,
and `deterministic_path_partition` are consumed, not modified).

This closes NOTES-2A §1.1 items 1, 2, 3, and 5, and item 4 **only for the lanes
whose closed-form writers read the merged evidence rows**. Section 4 below is the
honest statement of what is still not consumable and why the runner now refuses
it instead of solving base-only.

---

## 1. `Lfm2SealedEvidenceSet`

`Lfm2SealedEvidenceSet.open(base_request, extension_requests, ...)` holds
`(base, *extensions)` as one ordered aggregate.

* **One fail-closed lifecycle.** It opens the selective authenticated session of
  every member it owns; if any later member fails to open, every session it
  already opened is closed (with `add_note` on the propagating error) before the
  error escapes. A session the caller already opened is passed as
  `base_session=` and is *held but never owned*, so
  `resolve_production_parent_cache`'s existing cleanup keeps closing the base
  session exactly once. `close()` is idempotent, closes in reverse order, and
  aggregates close failures the same way the cache module's own session does.
* **Provenance.** `iter_batches()` yields `Lfm2SealedEvidenceBatch(merged_index,
  member_ordinal, member_role, member_cache_key, member_batch_index,
  input_sha256, split)`. `input_sha256` is the sealed batch content hash taken
  from the member's own manifest, so any consumer can name which sealed cache
  and which sealed batch a row came from.
* **The hard rules in one place.** An extension request must declare
  `all_rows_are_fit=True` and an all-`fit` split assignment, must carry its
  round-manifest digest, must share the base's parent / implementation /
  layer-topology fingerprints, must sit on a distinct path with a distinct cache
  key, and no two members may repeat a batch input (checked *before* any session
  is acquired). `fit_caches()` is the single accessor a writer iterates, and
  every value it returns is an already sealed immutable cache — that is where
  the "no parent forward" guard lives.
* **Combined identity, one definition.** `lfm2_on_policy_evidence_identity_sha256`
  moved into this module together with `ON_POLICY_EVIDENCE_IDENTITY_SCHEMA`. The
  runner now imports both; `_ResolvedOnPolicyEvidence.identity_sha256` and
  `ResolvedProductionParentCache.on_policy_evidence_identity_sha256` delegate to
  it, so the digest `_derive_sealed_cache_splits` folds into every derived
  phase-cache key and the digest the evidence set publishes are the same
  function, not two copies. The canonical body is unchanged, so warm phase-cache
  roots are unaffected.
* **Merged corpus fingerprint.** `corpus_fingerprint` for a multi-member set is
  derived from the members' own sealed `manifest["corpus"]` bodies (concatenate
  the three ordered list fields, sum `batch_count`, keep the sealed schema
  string) rather than restating the cache module's recipe. Each member's body is
  first re-hashed and compared against its own sealed `corpus_sha256`, which
  proves the derivation on real sealed data and fails closed if the cache module
  ever changes shape. That prediction has to be exact: the merged parent rows
  are aligned against a child cache built over the merged batch list, and
  `TraceRowIndex.assert_aligned` compares corpus fingerprints.

**Pass-through guarantee.** With no extension member the set's `cache_key`,
`corpus_fingerprint`, `splits`, and `batch_count` are the base cache's own, it
opens nothing, and `_merge_sealed_evidence_trace_rows` returns the base
`TraceRowIndex` object unchanged. A configuration without extension evidence is
byte-identical to the pre-wave revision.

## 2. What actually consumes the evidence

`resolve_production_parent_cache` now opens every extension cache and its
selective session, derives each extension's splits with
`_derive_sealed_cache_splits(..., all_rows_are_fit=True)` — the first production
call site of that parameter — and stores the aggregate on
`ResolvedProductionParentCache.evidence_set`. The `except BaseException` block
closes the evidence set *and* the base session.

`run_production_catalog` then materializes the extension corpora through the
sessions the set holds (`residual.calibration_batches_from_cache` +
`residual.extract_trace_rows(..., batch_splits=member.splits)`), validates each
extension corpus against the parent's tokenizer-facing row capacity, and merges:

* `parent_rows` — extension paths are renumbered into merged batch order
  (`batch.NNNNNN` prefix shifted by the member's batch offset) and concatenated
  after the base rows; per-layer fields are `torch.cat`-ed on the row axis; the
  merged index carries the evidence-set cache key and corpus fingerprint and a
  new `SEALED_EVIDENCE_TRACE_ROWS_SCHEMA` digest over the member digests, paths,
  and split labels.
* `calibration_batches` / `calibration_splits` — base batches then extension
  batches, extension labels always `fit`.

Everything downstream reads those merged values, so the extension rows reach:

* `residual.deterministic_path_partition` — extension rows carry `fit`, so they
  can only enter the discovery pool; `selection` and `validation` stay exactly
  the base corpus's rows;
* `CachedProductionEvidenceBuilder` → `residual.build_cached_layer_evidence` —
  the per-layer closed-form endpoint solve (measured in the e2e: fit rows per
  layer grow by exactly the extension row count, validation rows unchanged);
* `CachedTrajectoryEvaluator` — the child now replays the merged corpus, and the
  parent side is presented through `_SealedEvidenceParentCacheView`, a read-only
  merged batch view (same duck-typed shape as the existing
  `_BridgeProjectedParentCache`) that reads batches from sealed member caches
  only.

`ProductionParentCachePayload.cache_key` becomes the evidence-set cache key once
extension evidence is live (identical to the base key otherwise), so
`system.run(evidence_fingerprint=...)`, the target adapter, the ledger attempt,
and the published artifact certificate all bind the exact evidence that was
fitted. Two children fitted on different evidence can no longer claim one
evidence fingerprint. `payload.verify()` checks the merged identities and
re-verifies every held session.

## 3. Alternation semantics

New flag `--on-policy-refold-after-rounds N` (default 0, off).

```
pass 0        fold + publish child A                 -> run round 0 against A
pass 1        fold + publish child B on base+ext0    -> run round 1 against B
...
pass N        fold + publish on base+ext0..ext(N-1)  -> runs no further round
```

* `run_production_catalog_on_policy_alternation(config)` drives it. Each pass is
  one complete `run_production_catalog` invocation, so a pass releases its
  parent rows and sessions before the next pass opens its own.
* A refold pass differs from the pass before it **only** by the round manifests
  it consumes and by where it publishes: same parent, same sealed base cache,
  same blank seed, same topology and counted budget, same ledger, same round
  root, same cache roots. Destinations get the r-style sibling suffix
  `-round-<highest consumed round index>-refold`
  (`published-round-0-refold`, `receipt-round-0-refold.json`,
  `diagnostics-round-0-refold.json`).
* Ledger: every pass appends its fold attempt to the same `ExperimentLedger`
  file, and each pass's receipt reports its own `ledger.new_attempts` for that
  invocation only. NOTES-2A §1.2 still holds for *rounds* — a round is not a
  fold attempt and gets no ledger record — but a refold **is** a fold attempt and
  is recorded by the fold system itself, with no synthesized record.
* Receipt: each pass writes its own sealed receipt. The orchestrator also seals
  `<receipt-stem>-on-policy-alternation.json`
  (`ON_POLICY_REFOLD_ALTERNATION_SCHEMA`) listing, per pass, its role, status,
  configuration digest, receipt path and digest, publish destination, artifact
  tree digest, consumed round manifests, evidence identity, evidence split
  digest, evidence-set cache key, the rounds it sealed, and its parent-capture
  accounting.
* Budget: the alternation executes exactly `rounds × refolds` new rounds, and
  `len(consumed) + rounds × refolds <= on_policy_max_rounds` is a
  configuration-time error. The last pass consumes everything and runs no round.
* `run_production_catalog` **refuses** a config with
  `on_policy_refold_after_rounds > 0` rather than silently executing one pass;
  the orchestrator hands each pass a config with the knob zeroed. `main()`
  dispatches on the flag.
* The knob is deliberately **not** part of `config.identity()`. It does not
  change what a single pass solves, so including it would move every
  `configuration_sha256` and, through it, the derived restart-capsule root
  (`<cache-root>/closed-loop-restart-capsules/<child>-<run-scope>`), breaking
  resume for runs that never asked for an alternation. It is reported in the
  receipt's `on_policy_evidence_policy.refold_after_rounds` instead. Pass 0's
  identity is therefore byte-identical to a plain single-pass run; refold passes
  differ because their publish/receipt paths and consumed manifests differ,
  which also gives every pass its own capsule chain (two passes sharing one
  chain would kill the second at its first accepted layer).

## 4. Lanes that still cannot consume extension evidence — now fail closed

NOTES-2A §1.1 step 4 lists four call sites; three are outside this wave's
ownership. Rather than authenticate extension evidence and then solve base-only,
consuming it is refused at configuration time (and re-asserted at the
consumption site) whenever the lane's writers bind exactly one sealed cache
identity:

| lane | why it cannot consume |
|---|---|
| dense budget (`--compression-ratio`) | `dynamic_lfm2_system` hands `payload.cache` + one `selective_session` to `Lfm2DenseBudgetEvidenceProvider` / `Lfm2DenseBudgetModelOperator`; both bind one cache key |
| `--dense-heterogeneous-layout` | `build_lfm2_cached_heterogeneous_capacity_option_bank` fingerprints one cache |
| `--hybrid-model-fold` | binds one sealed parent cache key |
| `--producer-first-model-fold` | `hf_lfm2_producer_first_model_fold` asserts `parent_rows.cache_key` equals the sealed parent cache key |
| tokenizer bridge / carrier | the bridge trace adapter replays only the sealed base corpus |

*Producing* rounds (`--on-policy-evidence-rounds`) stays available on every lane,
exactly as before; only `--on-policy-round-manifest` consumption and
`--on-policy-refold-after-rounds` are gated. This is a behaviour change for a
configuration that previously ran base-only with shifted phase keys; it is the
fail-visible direction, and no live run passes a round manifest today (none was
sealed).

**Consequence for the production carrier lane (r11 and successors): the
alternation is not available there yet.** Making it available is an adapter /
dense-fold change: those writers must take an evidence set (or an ordered
sequence of `(cache, session, splits)`) instead of one handle, and the option
bank must fold the same identity digest into its evidence fingerprint.

## 5. Verification

Command (repo root, `-p no:randomly`):

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" -m pytest \
    tests/test_run_lfm2_dynamic_catalog.py \
    tests/test_hf_lfm2_sealed_evidence_set.py \
    tests/test_hf_lfm2_on_policy_evidence_round.py -q -p no:randomly
```

`138 passed, 1 skipped` (~62 s). The same three files plus
`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`: `142 passed, 1 skipped`
(~164 s).

Also green and unchanged:

* `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`,
  `tests/test_run_lfm2_dynamic_catalog_portable_runtime.py`,
  `tests/test_run_lfm2_dynamic_catalog_tokenizer_bridge.py`,
  `tests/test_dynamic_lfm2_producer_first_catalog.py`,
  `tests/test_prepare_lfm2_tokenizer_bridge_evidence.py`,
  `tests/test_run_lfm2_fused_attention_diagnostic.py` — 54 passed, ~124 s.
* `tests/test_dynamic_lfm2_dense_budget_fold_stack.py`,
  `tests/test_dynamic_lfm2_cpcf_production.py`,
  `tests/test_hf_lfm2_dynamic_closed_loop_adapter.py`,
  `tests/test_hf_lfm2_circuit_trace_cache.py` — 132 passed, 1 skipped, ~223 s.

New coverage:

* `tests/test_hf_lfm2_sealed_evidence_set.py` (11 tests, real sealed toy caches):
  pass-through with no extension, batch provenance and merged splits, merged
  corpus/cache-key derivation, digest equality with the runner's
  `_ResolvedOnPolicyEvidence.identity_sha256` and with the digest
  `_derive_sealed_cache_splits` folds in, all-fit enforcement, parent/path/key
  distinctness, repeated-batch refusal, **cleanup on failure mid-open** (the
  already-opened sessions are all closed), idempotent close that never closes
  the caller's session, receipt fields, deferred base session.
* `tests/test_run_lfm2_dynamic_catalog.py`:
  `test_extension_evidence_consumption_fails_closed_on_unready_lanes`,
  `test_refold_alternation_configuration_is_bounded_and_on_policy`,
  `test_alternation_pass_configs_change_only_evidence_and_destinations`,
  `test_tiny_real_catalog_alternates_fold_round_and_refold` (~28 s).

The e2e is monkeypatch-light: the only patches are two pure observers (a
wrapper around `_load_cache_bound_parent` that registers a counting forward hook,
and a wrapper around `run_lfm2_on_policy_evidence_round` that samples that
counter before and after). The round itself is real. It asserts:

* publication → real sealed round → refold consuming base+extension → second
  publication, both passes `complete`;
* the refold's `evidence_split_sha256`, `parent_trace_rows_sha256`, and artifact
  `evidence_fingerprint` all differ from the base pass's, and the base pass has
  no evidence identity at all;
* extension rows appear only in fit splits
  (`extension_selection_or_validation_batch_count == 0`, and per layer
  `fit_rows` grows by exactly the extension row count while `validation_rows` is
  unchanged);
* zero parent forwards outside the sanctioned round capture — measured, not
  asserted from a receipt field: forwards before the round are `0`, all forwards
  in pass 0 happen inside the round call, and the refold pass records `0`
  forwards in total;
* both children reload standalone from disk at the exact counted budget and both
  receipts report `fresh_reload_accepted`.

## 6. Deferred / known limitations

1. **Three of the four NOTES-1C §4 call sites are still base-only** and are now
   refused rather than silently solved (section 4). The dense budget lane is the
   one that matters for production; that is the next wave.
2. **Merge peak memory.** `_merge_sealed_evidence_trace_rows` `torch.cat`s each
   per-layer field while the base `TraceRowIndex` is still referenced, so the
   transient peak is roughly twice the retained circuit-row payload. That is
   harmless at toy and small-corpus scale but would be fatal on the ~25 GiB
   exact-4x lane. Freeing incrementally needs `extract_trace_rows` to grow a
   "merge into" entry point (its `layer_fields` are `MappingProxyType`, so the
   source cannot be released field by field from outside), which is in
   `run_lfm2_residual_attention_fold.py` — outside this wave.
3. **Alternation depth > 1 needs the child to actually move.** Round *n+1*
   fails closed if every child-visited trajectory it produces is already sealed
   (`"Every child-visited trajectory is already sealed ..."`). On the toy lane
   the published child is byte-identical between passes because the endpoint
   correction is null there (`mlp_endpoint_cosine == 0.0`,
   `max_endpoint_prior_delta_ratio` defaults to `0.0`, so the fitted down
   weight collapses to the clustered prior), which makes the second round's
   rollout reproduce the first one's corpus. The e2e therefore uses one round
   and one refold. A real lane with a non-null endpoint correction does not have
   this problem, but nothing here *proves* that, so treat `N > 1` as untested.
4. **The toy refold produces a byte-identical child** for the reason in (3).
   What the e2e proves is that the extension evidence entered the solve (fit
   rows per layer grow by exactly the extension row count) and that every
   derived identity moved; it does not prove the extension evidence changed the
   *weights*, because on this lane no evidence changes the weights.
5. **`--on-policy-child-artifact-directory` is refused with an alternation.**
   Each pass's round always reloads the child that pass published; a pinned
   foreign child is not on-policy. Pointing a single non-alternating run at a
   foreign child still works exactly as before.
6. **Round manifests are consumed by absolute path**, so a round root that moves
   changes the sealed round-manifest digest and therefore the evidence identity
   and every derived phase key. Rounds are relocatable for *loading* (the loader
   falls back to the sealed relative location) but the identity is not
   relocation-invariant. Nothing in this wave changes that.
7. **The alternation summary is not a receipt.** It is a separate sealed
   document (`alternation_sha256`) next to the first receipt. It is not written
   into `ExperimentLedger` and it is not read by anything yet.
