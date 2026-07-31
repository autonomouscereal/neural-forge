# NOTES-2A — catalog integration of restart capsules, replay mode, and on-policy rounds

Author: wave 2A (production catalog runner integration)
Date: 2026-07-31
Scope: `experiments/run_lfm2_dynamic_catalog.py`,
`src/neural_forge/hf_lfm2_dense_budget_model_fold.py`,
`tests/test_run_lfm2_dynamic_catalog.py`

This wave consumed NOTES-1A (restart capsules), NOTES-1B (replay-mode default and
capture merge), and NOTES-1C (on-policy evidence rounds).  Everything below is
either a decision that differs from those notes or work that is deliberately
**not** done yet.  Nothing here is a claim that something works; the verified
items are in the wave report and the tests.

---

## 1. Deferred: base+extension evidence consumption by the closed-form solvers

**Status: not implemented. Do not assume a run with `--on-policy-evidence-rounds N`
fits on extension evidence.**

What *is* wired today:

* the round runs after a successful publication, against the child this run just
  published (`_run_postpublication_on_policy_rounds`);
* its sealed outputs land under `--on-policy-round-root` (default
  `<cache-root>/on-policy-rounds`) as ordinary
  `hf_lfm2_on_policy_evidence_round` round directories;
* the sealed identities are copied into the run receipt
  (`receipt["on_policy_evidence"]`) and into the progress ledger
  (`on_policy_evidence_round_started` / `on_policy_evidence_round_sealed`);
* `--on-policy-round-manifest` authenticates already sealed rounds at parent-cache
  resolution time and folds their identities into every derived phase-cache key
  (section 3 below).

What is **not** wired: no closed-form writer reads an extension cache.  A run
given `--on-policy-round-manifest` today authenticates the extension evidence,
changes its phase-cache keys so a base-only solve cannot be silently reused, and
then still solves on the base cache only.  That is a deliberate,
fail-*visible* half: the key change is the part that cannot be added later
without invalidating everything solved in between, so it landed first.

### 1.1 The precise remaining change

NOTES-1C §4 is still the specification.  Concretely:

1. **Introduce `Lfm2SealedEvidenceSet`** (new module, or beside
   `ResolvedProductionParentCache`) holding `(base, *extensions)` plus the
   per-cache split assignment and per-cache selective session.  It must expose
   `fit_caches()` and keep the "no parent forward" guard in one place.
2. **Open extension caches and their selective sessions in
   `resolve_production_parent_cache`.**  Today `_resolve_on_policy_consumed_rounds`
   authenticates each round manifest (which opens and re-checks the extension
   cache inside `load_lfm2_on_policy_evidence_round`) and then lets the handle
   go.  Extension selective sessions are **not** opened, so the
   `except BaseException` cleanup block in that function only has to close the
   base session.  When extensions become live evidence, that cleanup block must
   close every extension session too — NOTES-1C §2, last paragraph.
3. **Derive extension splits with `all_rows_are_fit=True`.**  The parameter
   already exists on `_derive_sealed_cache_splits` and is unit-tested, but no
   production call site passes it yet because nothing consumes an extension
   corpus.  The rule is hard: a child-visited row must never become a
   `selection` row (the child would pick its own frontier) or a `validation`
   row (the split would stop being held out).
4. **Thread the evidence set through the four call sites in NOTES-1C §4**:
   `Lfm2DynamicClosedLoopAdapterConfig` construction and the
   boundary/attention/ShortConv/MLP solves it drives,
   `build_lfm2_cached_heterogeneous_capacity_option_bank`,
   `build_lfm2_cached_global_topology_frontier`, and
   `hf_lfm2_balanced_basis_phase_cache` context construction.  Three of those
   four modules are outside this wave's ownership boundary.
5. **Alternate fold and round.**  NOTES-1C §6 gives the loop
   (`seal base -> fold+publish -> round 0 -> re-solve -> fold+publish -> round 1`).
   The runner currently executes exactly one fold pass, so it performs at most
   the first `fold+publish -> round` step and stops.  Running the second fold
   pass needs either a re-entrant `run_production_catalog` inner loop or the
   `hf_on_policy_refold.run_on_policy_refold` orchestrator with a
   `trace_refold_fn` callback that (a) runs the round, (b) re-solves, and
   (c) reports the sealed cache identities through `TraceRefoldResult`.
   `--on-policy-max-rounds` (default 2) already bounds that loop and is written
   into every round manifest, so the budget contract does not change.

Until step 4 lands, `--on-policy-evidence-rounds N` with `N > 1` will run `N`
rounds against the *same* published child, which is not the intended
alternation.  It is bounded and sealed, but it is not on-policy in the useful
sense.  Treat `N > 1` as unsupported until the loop exists.

### 1.2 Ledger records

NOTES-1C asks for "receipt and ledger" records.  Receipt records are written.
The **`ExperimentLedger` gets nothing**: its records are fold attempts written by
`dynamic_fold_registry` / the fold system, and synthesising an attempt record for
an evidence round would misreport what the ledger means.  The round is recorded
in the progress journal (the runner's per-run append-only ledger) and in the
sealed receipt instead.  If a durable per-round ledger row is wanted, it needs a
new record type on `ExperimentLedger`, which is that module's decision.

---

## 2. Restart capsules — decisions that differ from NOTES-1A

### 2.1 Config surface lives on `Lfm2DenseBudgetFoldConfig`, not on the operator

NOTES-1A §1 shows the writer being built at the
`hf_lfm2_dense_budget_model_fold.py` call site with a `restart_capsule_directory`
appearing from somewhere.  The catalog reaches the dense operator through
`Lfm2DynamicStackResources.dense_fold_config` in
`src/neural_forge/dynamic_lfm2_system.py`, which is outside this wave's
ownership.  Rather than add a resource field there, the three new knobs
(`restart_capsule_directory`, `restart_capsule_rotation_keep`,
`resume_from_restart_capsule`) are fields on `Lfm2DenseBudgetFoldConfig`, which
the catalog already constructs directly.  `dynamic_lfm2_system.py` is untouched.

### 2.2 `--dense-resume-from-capsule` takes a capsule **root**, not one capsule

The public discovery API (`latest_lfm2_closed_loop_restart_capsule`,
`discover_lfm2_closed_loop_restart_capsules`) is root-oriented, and
`_capsule_root_path` / `_DIRECTORY_PREFIX` / the manifest filename are private.
Resolving an individual capsule path would have meant duck-typing private names.
The flag therefore names a root and always resumes from the newest authenticated
capsule under it.  Rotation keeps the newest `--dense-restart-capsule-keep`
capsules, so "newest" is the only capsule an operator normally wants.  Resuming
from an older capsule means pointing at a root that contains only that capsule.

### 2.3 The derived capsule root is run-scoped

`--dense-restart-capsule-root` defaults to

```
<cache-root>/closed-loop-restart-capsules/<child-name>-<run-scope-16-hex>
```

when `--dense-run-closed-loop` is set.  The scope digest is
`canonical_sha256(identity())` with the `target.dense_restart_capsule` block
removed, so a killed run and its resume — which differ only by the resume flag —
share one chain, while two genuinely different configurations sharing a cache
root never collide.  That collision matters: `rotate_lfm2_closed_loop_restart_capsules`
calls `discover_...`, which fails closed on duplicate or discontiguous capsule
sequence numbers, so two runs writing into one root would kill the *second* run
at its first accepted layer.

### 2.4 `moving_gauge_v3` + resume fails closed twice

`Lfm2DenseBudgetFoldConfig.__post_init__` rejects
`resume_from_restart_capsule` together with
`body_reclosure_mode == moving_gauge_v3` at configuration time, and the runtime
guard (`Lfm2ClosedLoopRestartCapsuleWriter.assert_resume_supported` ->
`Lfm2DynamicClosedLoopAdapter.assert_mid_body_resume_supported`) still runs.
Both are intentional: the config-time error is precise and early, the runtime
guard is the authority.

**Operational consequence for the production carrier lane:** that lane pins
`moving_gauge_v3`, so it *writes* capsules but can never *resume* from them.
Restartability for r10+ therefore needs either a reclosure-disabled body pass or
the "defer reclosure to a separate stage" change NOTES-1A §1 describes.  Nothing
in this wave changes that.

### 2.5 Progress stages added

Dense fold (hash-chained through `_Lfm2DenseFoldProgress`, scalar fields only):

* `dense_restart_capsule_resume_loaded`
* `dense_restart_capsule_writer_bound`
* `dense_restart_capsule_run_completed`
* `dense_closed_loop_started` gained `restart_capsule_enabled` and
  `restart_capsule_resume_requested`

The orchestrator's own capsule stages
(`child_closed_loop_restart_capsule_resume_admitted`,
`child_closed_loop_layer_skipped_restart_resume`,
`child_closed_loop_restart_capsule_published`) already existed and now reach the
catalog progress journal because the writer is bound.

Note the exact kill window: `child_closed_loop_restart_capsule_published` is
emitted by the orchestrator *after* `record_accepted_layer` returns.  A process
that dies inside the publication call leaves a fully sealed capsule on disk with
no published stage in the journal.  The tiny regression test reproduces exactly
that window.

---

## 3. Replay mode and phase-cache keys

### 3.1 `auto` is now always canonical prefix replay

`ProductionLfm2CatalogConfig.resolved_dense_closed_loop_replay_capture_mode`
previously returned `single_pass_causal` for heterogeneous FP32 children.  That
lane certifies every capture against a canonical reference replay *inside* the
loop, so it pays a second replay per capture point.  `auto` now always resolves
to `prefix_by_prefix_canonical`; `single_pass_causal` remains reachable only
through an explicit `--dense-closed-loop-replay-mode single_pass_causal`, which
still requires the authenticated heterogeneous dense FP32 path.

The resolved mode is recorded in the run receipt under
`closed_loop_replay_policy` (all three receipt shapes: interrupted, failed, and
complete), including
`in_loop_single_pass_certification_replay` so an operator can tell from the
receipt alone whether the double replay ran.

The carrier-lane "best proven folding lane" pin now compares the **resolved**
mode instead of the raw policy string, so `auto` is admissible there.

Two existing assertions in `tests/test_run_lfm2_dynamic_catalog.py` encoded the
old default and were updated, not deleted:
`test_dense_closed_loop_replay_policy_resolves_and_fails_closed` and
`test_wiring_uses_cached_heterogeneous_options_and_exact_budget`.

### 3.2 `merge_adjacent_layer_captures`

Surfaced as `--dense-closed-loop-merge-adjacent-layer-captures`, default off, and
threaded into `ClosedLoopCompilerConfig`.  Every other compiler-config field
keeps its library default, so the flag-off path is byte-identical to the previous
revision.  Per NOTES-1A §4 this changes sealed run-report content and the
`child_replay_capture_calls` audit counter, so the launcher — not the runner —
opts in.

**The flag is currently a no-op in this runner, and the launcher should know
that before relying on it.**  `ChildClosedLoopCompiler._merged_capture_hook`
consults `Lfm2DynamicClosedLoopAdapter.supports_merged_child_captures`, which
declines whenever a selective authenticated trace session is bound:

```python
return bool(
    self.selective_session is None
    and self._child_heterogeneous_layout is None
    and self.config.replay_capture_mode == CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
)
```

`run_production_catalog` always binds one — `experiments/run_lfm2_dynamic_catalog.py`
opens `parent_cache.selective_authenticated_session()` before materializing parent
rows if the resolution did not already open one — so the adapter always declines.
Measured directly on the toy dense lane: verdict `False`,
`selective_session_is_none=False`, `heterogeneous_layout_is_none=True`,
`replay_capture_mode='prefix_by_prefix_canonical'`.

The run report seals both `merge_adjacent_layer_captures_requested` (the config)
and `merge_adjacent_layer_captures_enabled` (whether the hook resolved), so an
opted-in run is auditable and honest rather than silently ineffective.  Making
the flag effective is an adapter change — a selected parent-evidence profile is
addressed by one exact layer role, so a union selection has no authenticated
parent inventory — and is out of this wave's ownership.

### 3.3 The phase-key change that must not be skipped (NOTES-1C §4)

Extension evidence enters the derived phase-cache keys through two paths:

* **Global topology frontier**: `_derive_sealed_cache_splits` and the two
  supplied-calibration split digests now include
  `on_policy_evidence_identity_sha256`, which is
  `canonical_sha256({base cache key, ordered extension cache keys, ordered
  extension batch counts, ordered round manifest sha256, round count})`.  That
  digest is `ResolvedProductionParentCache.evidence_split_sha256`, which is
  exactly what `Lfm2GlobalTopologyPhaseCacheContext` keys on.
* **Balanced basis**: `_compose_on_policy_phase_dependencies` adds the same
  digest as the external phase dependency
  `lfm2_on_policy_evidence_extension`, which
  `hf_lfm2_balanced_basis_phase_cache._live_bindings` folds into
  `external_dependency_sha256s`.

With no extension evidence both paths are byte-identical to the pre-extension
revision (asserted in
`test_on_policy_extension_evidence_changes_every_derived_phase_key` and
`test_on_policy_phase_dependency_is_added_only_when_evidence_exists`), so warm
phase-cache roots do not cold-recompute because of this wave.

**Not covered by a phase key:** the *cached heterogeneous capacity option bank*
(`build_lfm2_cached_heterogeneous_capacity_option_bank`) has its own evidence
fingerprint that this wave did not touch.  It is only reachable with
`--dense-heterogeneous-layout`, and no extension evidence reaches it yet, so
there is nothing to mis-reuse today.  It must be included in the same digest
as part of step 4 in section 1.1.

---

## 4. On-policy config deviations from NOTES-1C §1

| NOTES-1C | Here | Why |
|---|---|---|
| `on_policy_child_artifact_directory` **required** when rounds > 0 | optional; defaults to the child this run just published | The round runs after publication in-process, so the artifact always exists. The flag remains for pointing at a foreign standalone child. |
| `--on-policy-round-manifest` repeatable | same | Consumed rounds are authenticated at parent-cache resolution and must be contiguous from index 0. |
| `on_policy_round_root` disjoint from publish/ledger/receipt/certified child | disjoint from publish destination, ledger path, receipt path | The certified child lives under the publish destination, so it is covered. `dynamic_lfm2_cpcf_production_cli._managed_child_path` was not reused; that helper is in another CLI and would have coupled the two entrypoints. |

`on_policy_evidence_rounds > 0` requires `--sealed-parent-cache-directory`,
`--on-policy-fit-prompts`, and `--on-policy-manual-prompt-manifest`, and
`consumed manifests + requested rounds <= on_policy_max_rounds`.  All four are
configuration-time errors.

`fit_prompts_disjoint_from_manual_review` in the receipt is copied from each
sealed round manifest's `disjointness.verified_disjoint` and is never recomputed
by the runner.

---

## 5. Counter semantics

`parent_forward_count_after_cache` and `parent_trace_regeneration_after_cache`
keep their exact pre-extension meaning: both stay 0 and neither is touched by an
on-policy round.  The round's single sanctioned parent capture pass is reported
separately as:

* `receipt["parent"]["on_policy_parent_capture_invocations"]`
* `receipt["on_policy_evidence"]["parent_capture_invocations"]`

`ResolvedProductionParentCache.parent_trace_regeneration_forward_count` remains 0
for the sealed branch, as NOTES-1C §2 requires.

---

## 6. Known gaps in test coverage

* The **real** on-policy round is exercised only by
  `tests/test_hf_lfm2_on_policy_evidence_round.py`.  The catalog test
  monkeypatches `run_lfm2_on_policy_evidence_round` and asserts invocation
  arguments, ordering, and counter accounting.  There is no catalog-level test
  that runs a real round end to end; the toy child's tokenizer is a nine-token
  WordLevel vocabulary, and `_prepare_batch` rejects prompts shorter than three
  tokens, so a real toy round is possible but was not built here.
* `--on-policy-round-manifest` consumption is covered at the split-digest and
  phase-dependency level, not by a catalog run that consumes a real sealed round.
* The capsule kill/resume test uses a 3-layer, hidden-32 toy parent with
  `--dense-body-reclosure-mode disabled`.  Resume under the production carrier
  lane is impossible by construction (section 2.4), so there is nothing to test
  there beyond the fail-closed assertion.
