# NOTES-1C — wiring sealed on-policy evidence rounds into the catalog runner

Author: wave 1C (on-policy evidence extension)
Date: 2026-07-31
Status: **plan only — no runner file was modified by this wave**

The round driver, its fail-closed loader, and its reference resolver landed in
this wave:

- `src/neural_forge/hf_lfm2_on_policy_evidence_round.py`
- `tests/test_hf_lfm2_on_policy_evidence_round.py`
- `docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md`

Nothing in `experiments/run_lfm2_dynamic_catalog.py`,
`src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`,
`src/neural_forge/dynamic_closed_loop_fold.py`,
`src/neural_forge/dynamic_fold_registry.py`, or
`src/neural_forge/dynamic_fold_executor.py` was touched.  The changes those
files need are recorded here for the wave that owns them.

## 1. Config surface (`ProductionLfm2CatalogConfig`)

Add, all defaulting to the current behavior (rounds off):

| field | type | default | meaning |
|---|---|---|---|
| `on_policy_evidence_rounds` | `int` | `0` | number of extension rounds to run; `0` keeps today's teacher-forced-only lane |
| `on_policy_max_rounds` | `int` | `2` | bounded budget written into every round manifest |
| `on_policy_child_artifact_directory` | `Path \| None` | `None` | published standalone child the round reloads; required when rounds > 0 |
| `on_policy_fit_prompts` | `Path \| None` | `None` | JSON list of fit prompts for the rollout |
| `on_policy_manual_prompt_manifest` | `Path \| None` | `None` | reserved manual-review prompt manifest; **required** when rounds > 0 |
| `on_policy_round_root` | `Path \| None` | `None` | output root; defaults to `cache_root / "on-policy-rounds"` |
| `on_policy_rollout` | rollout knobs | — | `max_prompts`, `max_new_tokens`, `max_context_tokens`, `seed`, `max_capture_batches` |
| `on_policy_round_manifests` | `tuple[Path, ...]` | `()` | already-sealed rounds to consume without running a new one |

`__post_init__` validation (mirror the existing fail-closed style):

- `on_policy_evidence_rounds > 0` requires `sealed_parent_cache_directory`.
  The base cache must already be sealed; the legacy calibration build mode is
  not a valid base for an extension round.
- `on_policy_evidence_rounds > 0` requires
  `on_policy_child_artifact_directory`, `on_policy_fit_prompts`, and
  `on_policy_manual_prompt_manifest`.  A round without a disjointness manifest
  must be rejected at config time, not at run time.
- `on_policy_evidence_rounds <= on_policy_max_rounds`.
- `on_policy_round_root` must be disjoint from `publish_destination`,
  `ledger_path`, `receipt_path`, and the certified child artifact (reuse the
  existing `_managed_child_path` / disjointness helpers in
  `dynamic_lfm2_cpcf_production_cli.py`).

New CLI arguments in `build_parser`, next to `--sealed-parent-cache-directory`:

```
--on-policy-evidence-rounds N
--on-policy-max-rounds N                (default 2)
--on-policy-child-artifact-directory P
--on-policy-fit-prompts P
--on-policy-manual-prompt-manifest P
--on-policy-round-root P
--on-policy-round-manifest P            (repeatable; consume sealed rounds)
--on-policy-max-new-tokens N
--on-policy-max-prompts N
--on-policy-max-capture-batches N
--on-policy-rollout-seed N
```

Add each new field to the config-serialization list around line 1153 and to the
runtime receipt block around line 8332 so the receipt records the round budget
even when it is zero.

## 2. Parent-cache resolution (`resolve_production_parent_cache`, line 3037)

`ResolvedProductionParentCache` currently carries exactly one
`cache: Lfm2CircuitTraceCache`.  Extend it rather than replacing it:

```python
extension_caches: tuple[Lfm2CircuitTraceCache, ...] = ()
extension_selective_sessions: tuple[Lfm2CircuitTraceSelectiveAuthenticatedSession, ...] = ()
on_policy_round_manifest_sha256: tuple[str, ...] = ()
on_policy_round_count: int = 0
on_policy_parent_capture_invocations: int = 0
```

`parent_trace_regeneration_forward_count` **stays 0**.  On-policy capture is not
trace regeneration; give it its own counter
(`on_policy_parent_capture_invocations`, one per round) so the existing
"no parent forwards after the sealed cache" audit keeps its exact meaning.  The
runner's `runtime_receipt_fields()` should emit both.

Resolution order inside the sealed branch:

1. Open the sealed base cache exactly as today (`verify=False`, selective
   session, calibration/split derivation).  Do not change any of it.
2. If `on_policy_round_manifests` is nonempty, call
   `resolve_lfm2_on_policy_evidence_references(base_cache.path, manifests,
   expected_parent_fingerprint=fingerprint_lfm2_parent(parent).identity_sha256)`
   and open the returned extension paths.  This path performs **zero** parent
   forwards and is the normal production route: rounds are produced by a
   separate driver invocation and consumed here.
3. If `on_policy_evidence_rounds > 0` and the requested round index is not yet
   sealed, call `run_lfm2_on_policy_evidence_round(...)` with
   `capture_config=parent_capture_config` (the runner already builds the exact
   capture config the base cache used, so
   `require_matching_capture_selection=True` will hold), then consume the
   returned reference tuple.

Wire the selective sessions the same way the base cache does, and extend the
existing `except BaseException` cleanup block so every extension session is
closed if any one of them fails to open.

## 3. Evidence splits

`_derive_sealed_cache_splits(parent_cache)` derives selection/fit/validation
splits from the sealed base corpus.  Extension caches need the same treatment:

- Derive splits per extension cache with the same deterministic function.
- **All on-policy rows are `fit` rows.**  Do not let a child-visited state
  become a `selection` or `validation` row: selection would let the child pick
  its own frontier, and validation would stop being held out.  Encode that as a
  hard rule in `_derive_sealed_cache_splits` (add an
  `all_rows_are_fit: bool = False` parameter) rather than as a convention.
- `SEALED_CACHE_SPLIT_SCHEMA` payloads must include the extension cache key and
  the round manifest SHA-256 so the split fingerprint changes when evidence is
  added.

## 4. Solver consumption

Every closed-form writer that currently iterates the single parent cache must
iterate the ordered evidence tuple instead.  Concretely, the call sites that
take `resolved.cache` / `resolved.selective_session` today:

- `hf_lfm2_dynamic_closed_loop_adapter.Lfm2DynamicClosedLoopAdapterConfig`
  construction and the boundary/attention/ShortConv/MLP solves it drives;
- `build_lfm2_cached_heterogeneous_capacity_option_bank`;
- `build_lfm2_cached_global_topology_frontier`;
- `hf_lfm2_balanced_basis_phase_cache` context construction.

Preferred shape: introduce one small read-only aggregate
(`Lfm2SealedEvidenceSet`) holding `(base, *extensions)` plus the per-cache split
assignment, and hand *that* to the adapter instead of a bare cache.  Adapter
internals then only need `for cache in evidence.fit_caches()`.  This keeps the
change mechanical and reviewable, and keeps the `no parent forward` guard in one
place.

Phase-cache keys (`Lfm2BalancedBasisPhaseCacheContext`,
`Lfm2GlobalTopologyPhaseCacheContext`) must include every extension cache key
and every round manifest SHA-256.  If they do not, a solve fitted on base-only
evidence will be silently reused for a base+extension request.  This is the one
change that must not be skipped.

## 5. Receipt and ledger

Add to the run receipt:

```json
"on_policy_evidence": {
  "round_count": 0,
  "max_rounds": 2,
  "round_manifest_sha256": [],
  "extension_cache_keys": [],
  "extension_batch_counts": [],
  "parent_capture_invocations": 0,
  "parent_trace_regeneration_forward_count": 0,
  "fit_prompts_disjoint_from_manual_review": true
}
```

The `fit_prompts_disjoint_from_manual_review` flag must be copied from each
round manifest's sealed `disjointness.verified_disjoint`, never recomputed by
the runner.

## 6. Ordering inside the fold sequence

A round is only meaningful against a child that exists.  The natural loop is:

```
seal base cache (existing)
fold + publish child            -> round 0 reloads that published child
run round 0                     -> extension cache 0
re-solve closed-form writers on base + extension 0
fold + publish child            -> round 1 reloads the new published child
run round 1                     -> extension cache 1
re-solve on base + extension 0 + extension 1
```

So the runner needs the round call **after** publication and **before** the next
solve pass, not inside the first solve.  With `on_policy_max_rounds=2` the loop
terminates after two extensions.  `hf_on_policy_refold.run_on_policy_refold` is
the right orchestrator shape for that alternation (it already enforces zero
optimizer/backward/training counters and exact rollback of rejected candidates),
but its `trace_refold_fn` callback contract expects the callback to mutate the
child; a catalog integration should supply a callback that (a) runs the round,
(b) re-solves, and (c) reports the sealed cache identities through
`TraceRefoldResult`.  That is a wave of its own and is deliberately out of scope
here.

## 7. Tests the runner wave should add

- Config rejects `on_policy_evidence_rounds > 0` without a sealed base cache,
  without a child artifact directory, or without a manual-review manifest.
- A tiny end-to-end catalog run with `--on-policy-evidence-rounds 1` produces a
  receipt whose `parent_trace_regeneration_forward_count` is still `0` and whose
  `on_policy_evidence.parent_capture_invocations` is `1`.
- Phase-cache reuse is *not* hit when the same request gains an extension cache
  (the negative test that catches the silent-reuse bug in section 4).
- On-policy rows never land in the `selection` or `validation` split.
