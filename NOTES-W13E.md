# NOTES-W13E — merged on-policy evidence through the balanced-basis phase cache

Wave 13E, branch `codex/lrc-blank-compression`, working tree at `ecb16477`
(the code this touches is unchanged since `53b353f0`). This closes the
infrastructure defect that failed the r13 refold closed in `NOTES-W13D` §6.4:

```
Lfm2BalancedBasisPhaseCacheError: trace_cache must be Lfm2CircuitTraceCache.
```

Local CPU work only. **Nothing is committed** — no `git add`, no commit.

---

## 1. What the defect actually was — two refusals, not one

`NOTES-W13D` §6.4 identified the first refusal correctly. Tracing it on the
runtime turned up a second one on the same path that the notes' option 1
("drop the four `--balanced-basis-phase-*` flags") would have hit next:

| # | where | refuses |
|---|---|---|
| 1 | `hf_lfm2_balanced_basis_phase_cache.py` `_trace_cache_binding` | `Lfm2MergedSealedEvidenceCache` (the reported failure) |
| 2 | `hf_lfm2_balanced_circuit_basis.py:3195` (pre-change) | `Lfm2MergedSealedEvidenceSession`, in **both** the phase-cached and the DIRECT builder |

`_resolve_dense_evidence_handles` (`hf_lfm2_dense_budget_model_fold.py:508-523`)
swaps **both** handles when extension evidence exists — the cache *and* the
session. The direct builder types `trace_cache: Any`, but it type-checked
`selective_session` against the concrete class:

```
TypeError: selective_session must be an Lfm2CircuitTraceSelectiveAuthenticatedSession or None.
```

reproduced on this runtime before any edit. So dropping the phase-cache flags
would have moved the r13 failure from the balanced-basis binding into the
balanced-basis builder, ~10 minutes later, with a different exception type.
Third refusal, further down, is a genuine incompatibility — §6.

---

## 2. Changes

Three source files. `hf_lfm2_circuit_trace_cache.py` is **byte-identical**
(`16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945`, the same
digest `NOTES-W13D` §1 recorded), and `hf_lfm2_dense_budget_model_fold.py` /
`dynamic_closed_loop_fold.py` were not touched — the routing in them is right.

### 2.1 `src/neural_forge/hf_lfm2_balanced_basis_phase_cache.py`

| symbol | line | what |
|---|---|---|
| `LFM2_BALANCED_BASIS_MERGED_EVIDENCE_SCHEMA` | `:108` | new binding document schema |
| `_MERGED_EVIDENCE_BINDING_FIELD` | `:117` | the one binding key merged evidence adds |
| `_sealed_trace_cache_binding` | `:796` | the pre-change body, renamed, **unchanged line for line** |
| `_merged_evidence_member_bindings` | `:871` | binds every member through `_sealed_trace_cache_binding` |
| `_merged_evidence_cache_binding` | `:928` | the merged binding |
| `_trace_cache_binding` | `:1097` | explicit typed dispatch; anything else still fails closed |
| `_live_bindings` | `:1115` | widened annotations + `balanced_basis_merged_evidence_bound` progress event |
| `_phase_key` | `:1450` | two extra dependencies, merged evidence only |

### 2.2 `src/neural_forge/hf_lfm2_sealed_evidence_set.py`

`Lfm2MergedSealedEvidenceSession.load_selected_tensor_rows` (`:551`) and
`release_selected_tensor_rows` (`:580`) — bounded row reads routed to the member
session that authenticated the bytes, restated in merged coordinates. This
mirrors the existing `load_selected_tensors` / `release_selected_tensors` pair in
the same class exactly: the reader is
`hf_lfm2_circuit_trace_row_selection.load_lfm2_circuit_trace_selected_rows`
called on the member's own concrete session, and only `batch_index` and
`session_sha256` are restated; release restores the member digest before handing
the payload back. `_authenticated_row_api`
(`hf_lfm2_balanced_circuit_basis.py:1216`) already had the duck-typed hook for
exactly this, which is why nothing in the row path needed widening.

### 2.3 `src/neural_forge/hf_lfm2_balanced_circuit_basis.py`

- `:56` imports `Lfm2MergedSealedEvidenceSession`.
- `:3217` accepts the merged session alongside the concrete one. The two names
  are read from the module namespace rather than a prebuilt tuple, because six
  tests in `test_run_lfm2_dynamic_structural_compression.py` substitute
  `balanced.Lfm2CircuitTraceSelectiveAuthenticatedSession` to inject a fake
  session; a module-level tuple silently broke that seam (caught by those six
  failing, then fixed).
- `:3179` new named refusal for streaming vocabulary-margin v2 on merged
  evidence — see §6.

---

## 3. What the merged phase key binds

The binding a merged evidence set produces keeps the **same 14 top-level keys**
a single sealed cache produces (with merged values) and adds exactly one nested,
self-sealed document, `merged_sealed_evidence`:

```
schema                              neural-forge-lfm2-balanced-basis-merged-sealed-evidence-binding-v1
evidence_set_cache_key              the Lfm2SealedEvidenceSet cache key (f62110d7… for the r13 set)
base_cache_key
on_policy_evidence_identity_sha256  the digest the runner already folds into derived keys
member_count / round_count / batch_count / base_batch_count
extension_batch_indices
merged_corpus_sha256
derived_merged_manifest_sha256      canonical digest of the merged manifest body
derived_merged_completion_sha256    canonical digest of the ordered member completions
merged_session_sha256
sealed_evidence_set_source_sha256   source hash of hf_lfm2_sealed_evidence_set.py
ordered_members[]                   per member:
    role / ordinal / batch_offset
    split_sha256 / round_manifest_sha256 / target_preimage_sha256
    all_rows_are_fit_rows
    trace_cache -> the FULL single-cache binding, produced by
                   _sealed_trace_cache_binding: cache_key, manifest_sha256,
                   completion_sha256, parent_identity_sha256, corpus_sha256,
                   layer_topology_sha256, implementation_sha256,
                   collector_source_sha256, canonical_contract_source_sha256,
                   transformers_lfm2_source_sha256, execution_sha256,
                   capture_selection_sha256, selective_session_sha256,
                   batch_count
all_extension_rows_are_fit_rows / merged_view_opens_no_new_session / parent_forward_calls
merged_sealed_evidence_sha256       canonical_sha256 of everything above
```

Two design points worth stating plainly:

- **Every member is bound by the same function that binds a plain cache.** Each
  member is a real `Lfm2CircuitTraceCache` with its own authenticated session, so
  `_merged_evidence_member_bindings` calls `_sealed_trace_cache_binding` per
  member. A member whose manifest, completion, capture selection, collector
  source, implementation, or session identity moved fails closed inside the
  proven single-cache authenticator, not in new code.
- **`derived_*` is named `derived_`.** A merged corpus has no sealed directory,
  so `merged_manifest()` deliberately drops `manifest_sha256`
  (`hf_lfm2_sealed_evidence_set.py:970-1012`: "no file on disk carries this body,
  and pretending otherwise would let a consumer believe it authenticated one").
  The two derived digests are computed here, and the receipt says so.

`_phase_key` gains exactly two dependency entries, **only** when the binding is
merged:

```
merged_sealed_evidence        = merged_sealed_evidence_sha256
on_policy_evidence_identity   = on_policy_evidence_identity_sha256
```

plus `trace_cache` = the evidence-set cache key and `trace_corpus` = the merged
corpus digest, which the existing dependency names already carry. Neither new
name is under `external.`, so no collision with the runner's own
`_compose_on_policy_phase_dependencies` external dependency
(`run_lfm2_dynamic_catalog.py:9535`) — that binding still applies and is now
belt-and-braces rather than the only thing stopping a base-only reuse.

The fail-closed checks inside `_merged_evidence_cache_binding`, before any of
that is assembled: the session must be the merged session **of this same
evidence set**; the four cache↔session identity comparisons the single-cache
path makes (`path`, `cache_key`, `batch_count`, `manifest`); member 0 must be the
base role and every other member an on-policy extension with a round manifest and
an all-fit split; every extension must agree with the base on parent identity,
implementation, and layer topology; the member batch counts must sum to the merged
count; and the merged session's `ordered_member_session_sha256` must equal the
per-member session digests.

---

## 4. Backward-compatibility proof

Two independent proofs, both run on this machine.

**(a) Cross-revision, executed.** The pre-change
`hf_lfm2_balanced_basis_phase_cache.py` was snapshotted before the first edit
and loaded as a sibling module inside `neural_forge`, then both revisions were
called on **one identical live parent + sealed cache + open session**
(`scratchpad/verify_backcompat.py`):

```
binding byte-identical   True
live binding diff keys   ['basis_implementation_sha256']
dependencies differing   ['basis_implementation']
abi identical            True
numeric contract same    True
```

**(b) In-suite pin.**
`test_plain_trace_cache_binding_is_byte_identical_to_the_pre_merge_form`
(`tests/test_hf_lfm2_balanced_basis_phase_cache.py:1380`) rebuilds the
pre-change 14-key construction verbatim in the test body and asserts
`canonical_json_bytes` equality, asserts `merged_sealed_evidence` is absent from
a plain binding, and asserts neither new dependency name appears in a
single-cache phase key.

**The one thing that does move, and why it is not avoidable.**
`_implementation_fingerprint()` (`:1420`) hashes the source of
`hf_lfm2_balanced_basis_phase_cache.py` itself (`phase_adapter: __file__`) and of
`hf_lfm2_balanced_circuit_basis.py`. Any edit to either file — including this one
— rotates the `basis_implementation` dependency and therefore every phase-key
digest. That is the design's own pre-existing property, and the design's own
answer to it is `_load_existing_lfm2_circuit_balanced_basis_for_live_fold`
(`:2351`), which pops exactly `basis_implementation` and `numeric_runtime` before
comparing, so historical sealed entries stay consumable through
`--balanced-basis-existing-phase-key-sha256`. Proof (a) shows that
`basis_implementation` is the **only** thing that moved: the binding, every other
dependency, the ABI and the numeric contract are byte-identical.

`hf_lfm2_sealed_evidence_set.py` is **not** in `_implementation_fingerprint`'s
module list, so §2.2 alone rotates nothing. Its source hash is bound inside
`merged_sealed_evidence` instead, so a change to the evidence-set implementation
invalidates merged phase entries without touching single-cache ones.

---

## 5. Attributes the phase-cached path consumes, and where each comes from

Enumerated by reading every consumption site and then executing all three paths.

**On the cache handle** — `_trace_cache_binding` and, on a cache miss, the cold
builder `build_lfm2_circuit_balanced_basis`:

| attribute | merged cache | note |
|---|---|---|
| `cache_key` | evidence-set cache key | |
| `path` | base cache's directory | reported/compared only; no merged payload is read from it |
| `manifest` | merged manifest | **no `manifest_sha256`** — see §3 and §6 |
| `manifest["implementation"/"execution"/"capture"]` | base's, restated | every member shares them; rechecked per member |
| `parent_fingerprint` | base's | every member must match |
| `corpus_fingerprint` | merged corpus digest | |
| `implementation_fingerprint`, `layer_topology_fingerprint` | base's | rechecked per member |
| `batch_count`, `__len__` | merged | sum of members, asserted |
| `verify()` | reverifies the set and every member | not called by the binding, same as the single-cache path |
| `load_batch` / `iter_batches` | routed to the sealing member | used only when no session is supplied |
| `extension_batch_indices` | merged indices, all fit | reported |

**On the session handle** — `_trace_cache_binding`, `_selective_record_metadata`,
`_load_cache_records`, `_stream_selective_cache_snapshot_grams`:

| attribute | merged session | note |
|---|---|---|
| `verify()` | reverifies every member session | |
| `path` / `cache_key` / `batch_count` / `manifest` | mirror the merged cache | four-way comparison passes |
| `evidence_sha256` | merged session digest | binds `ordered_member_session_sha256` |
| `as_dict()` | merged shape | has `cache_key`, `batch_count`, `session_sha256`, `ordered_member_session_sha256`; **no** `manifest_sha256` / `completion_sha256`, so the merged path compares the three it does have |
| `batch_references` | member references renumbered into merged order, sealed hashes intact | |
| `load_selected_tensors` / `release_selected_tensors` | already existed | |
| `load_selected_tensor_rows` / `release_selected_tensor_rows` | **added, §2.2** | this was the gap |

Nothing needed by the residual-basis phase is missing from the merged cache.
The single missing thing was the session's bounded-row pair, which is a
delegation the class already had the pattern for.

**Numerical equivalence, measured.** On one base(4)+extension(2) evidence set the
three routes produce the *same* basis map:

```
direct builder, merged cache + merged session (row-sliced streaming)  40ddeebf9c0003d5…
direct builder, merged cache + no session     (whole-tensor records)  40ddeebf9c0003d5…
phase-cached builder, merged cache + merged session                   40ddeebf9c0003d5…
```

---

## 6. STOP-and-report: streaming vocabulary-margin v2 stays incompatible

`build_lfm2_streaming_vocabulary_margin_adapter_v2` authenticates the sealed
cache through `_authenticate_selective_session`
(`hf_lfm2_streaming_vocabulary_margin_adapter.py:416-460`), which requires
`trace_cache.manifest["manifest_sha256"]` — a digest of a real file on disk. A
merged corpus has none, on purpose. Executed, pre-guard:

```
Lfm2StreamingVocabularyMarginAdapterError: trace cache manifest_sha256 must be a
lowercase SHA-256 digest.
```

Handing that adapter a derived digest would be inventing an authentication claim,
which is a design call, not a fix. **I did not make it.** What I did instead:
named the refusal where the configuration is composed
(`hf_lfm2_balanced_circuit_basis.py:3179`), in the same voice as the two existing
extension-evidence refusals in `hf_lfm2_dense_budget_model_fold.py:3401-3411`:

> The streaming vocabulary-margin adapter binds one sealed cache manifest digest
> and cannot consume merged on-policy extension evidence.

Measured behaviour of the three margin arms on merged evidence:

| arm | merged evidence |
|---|---|
| margin observability **off** (r13's balanced-basis arm as composed in `NOTES-W13D` §4) | works, row-sliced streaming |
| margin observability **on**, streaming v2 **off** | works; whole-tensor per-batch reads, no row slicing |
| streaming v2 **on** | refused by name |
| `vocabulary_margin_candidate_gate_enabled` | already refused upstream (`dense_budget:3406`) |

**Decision needed before relaunch:** confirm whether the r13 launcher passes
`--dense-streaming-vocabulary-margin-v2` / `--dense-vocabulary-margin-observability`.
`NOTES-W13D` §4 does not list either flag, and the balanced-basis arm it composed
does not require them — but the toy carrier lane in
`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py:813` pins both whenever the
residual basis is `circuit_balanced`, so the production launcher may too. If it
does, the relaunch must drop margin v2 for this arm, or a separate design change
to the margin adapter is required. Note the middle arm's memory profile before
choosing it on the 27.7 GB corpus: observability without streaming v2 abandons row
slicing.

---

## 7. Why the suite never caught this

Both on-policy refold end-to-end lanes
(`test_carrier_dense_core_alternates_fold_round_and_refold`,
`test_bridged_carrier_lane_alternates_fold_round_and_refold`) call
`_lane_arguments(..., carrier_routes=False)`, and that argument builder selects
`residual_basis_mode = "weight_complete"` unless carrier routes are on
(`:675`). `_build_residual_basis` returns from the weight-complete branch before
the balanced basis exists, so no test anywhere combined base+extension evidence
with the balanced-basis phase cache. That is now covered — §8's second and fifth
tests, and in particular
`test_dense_residual_basis_routes_merged_evidence_through_the_phase_cache`, which
drives the production chain
`_resolve_dense_evidence_handles` → `_build_residual_basis` → phase-cached builder
and is the exact call sequence that failed on the server.

---

## 8. Tests

Five new in `tests/test_hf_lfm2_balanced_basis_phase_cache.py`:

| test | line | proves |
|---|---|---|
| `test_plain_trace_cache_binding_is_byte_identical_to_the_pre_merge_form` | `:1380` | §4(b) |
| `test_merged_sealed_evidence_publishes_a_phase_binding_every_member` | `:1434` | the r13 shape publishes a valid entry; the receipt names both members with their cache keys, implementations, round manifest and split; `merged_sealed_evidence_sha256` reproduces from the document; the `balanced_basis_merged_evidence_bound` event fires; a second call **hits** with a forbidden builder and byte-identical tensors |
| `test_merged_evidence_phase_key_binds_every_member_identity` | `:1526` | a different extension cache, a different round manifest, a different base **split**, and the base-only solve all produce different phase keys, and none collide. The split case is the sharp one: the evidence-set cache key is *identical* there, so it proves the binding is strictly stronger than that key alone |
| `test_merged_evidence_binding_fails_closed_on_a_foreign_handle_pair` | `:1584` | merged cache + a member session, plain cache + merged session, and a non-cache object each fail closed |
| `test_dense_residual_basis_routes_merged_evidence_through_the_phase_cache` | `:1609` | §7 |
| `test_merged_evidence_margin_modes_are_exactly_what_the_adapters_support` | `:1666` | §6 |

One new in `tests/test_hf_lfm2_sealed_evidence_set.py`:
`test_merged_session_row_reads_are_routed_to_their_sealed_member` (`:626`) —
merged coordinates out, member bytes in (equal to the member's own row read), and
a payload the merged view did not hand out is refused.

Suites run (`.venv/Scripts/python.exe -m pytest -q -p no:randomly`):

| suite | result |
|---|---|
| `test_hf_lfm2_balanced_basis_phase_cache.py` | **25 passed** (19 before + 6) |
| `test_hf_lfm2_sealed_evidence_set.py` | **15 passed** (14 before + 1) |
| consolidated sweep of the 10 suites importing the three modified modules | **217 passed, 1 skipped** |
| `test_run_lfm2_dynamic_catalog.py` + dense/closed-loop/on-policy consumers | **158 passed, 1 skipped, 1 failed** |
| `test_hf_lfm2_dynamic_closed_loop_adapter.py` + selective mapping/cpcf | **83 passed** |
| `test_run_lfm2_dynamic_catalog_carrier_e2e.py` (the fold -> round -> refold lanes) | **8 passed** in 349 s |

The one failure is
`test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure`, which
belongs to Codex's uncommitted `hf_lfm2_dense_budget_model_fold.py` diff. Proved
pre-existing rather than assumed: with my three source files temporarily reverted
to their pre-change snapshots it fails identically (`DID NOT RAISE
Lfm2DenseBudgetFoldError`), and the files were restored and re-hashed afterwards.

`ruff check` clean on all five touched files. (`ruff format --check` reports the
same three source files as unformatted both before and after the change; this
repo does not use `ruff format`.)

---

## 9. Caveats

- **No production evidence.** Everything here is measured on toy LFM2 parents
  (vocab 31, hidden 8, 2 layers) with 4+2 sealed batches. The r13 evidence set is
  256+115 batches over 73,989 paths. The identity algebra is size-independent;
  the memory behaviour of the merged row path at that scale is not measured here.
- **The relaunch's phase keys are new.** `basis_implementation` rotated (§4), and
  the r13 refold consumes merged evidence for the first time, so the balanced
  basis is a cold build either way. The three global-topology phase entries
  published at 08:55–09:21 in the failed run are keyed on the evidence identity,
  not on this module, and `NOTES-W13D` §6.4's expectation that a relaunch hits
  them is unaffected by anything here.
- **§6 is unresolved by design** and needs the launcher check before relaunch.
- **Not committed.** `git status` shows my three source files, two test files and
  this note as modified/untracked, alongside Codex's own uncommitted
  `hf_lfm2_dense_budget_model_fold.py` and `dynamic_closed_loop_fold.py`, which I
  did not touch.
