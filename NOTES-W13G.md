# NOTES-W13G — streaming vocabulary-margin v2 learns merged sealed evidence

Wave 13G, branch `codex/lrc-blank-compression`. All work is based on
`84acded2`; `a8b65e15` (*Assert gate subjects, not evaluation counts, in the
native lanes*) landed on the branch mid-wave and touches only five native-lane
test files and `NOTES-W13H.md`, none of which this wave reads or edits.
This closes the mutual exclusion `NOTES-W13F` §4 proved: the carrier production
pin **requires** `--dense-streaming-vocabulary-margin-v2` (clause A,
`run_lfm2_dynamic_catalog.py:2123-2141`) and the balanced-basis builder
**refused** v2 on the merged session `_resolve_dense_evidence_handles` hands it
(clause B, `hf_lfm2_balanced_circuit_basis.py:3179-3190`), so a carrier-routed
refold consuming on-policy extension evidence had no executable configuration
in either margin setting.

**Clause A is untouched.** The validator, its conjunction, and its message are
byte-identical. What changed is clause B: the v2 adapter now authenticates
merged sealed evidence for real, member by member, following the pattern
`84acded2` used for the balanced-basis phase binding
(`_merged_evidence_cache_binding`, `hf_lfm2_balanced_basis_phase_cache.py:928`).

Opening that arm exposed a second thing that had to move with it: the balanced
basis's own fit/held-out split was role-blind, so with merged evidence it would
have put child-generated batches into the held-out projection gate. That is
§3, and it is a numeric change for every merged arm.

Local CPU work only. **Nothing is committed** — no `git add`, no commit.

---

## 1. What the v2 adapter authenticates and reads — the full enumeration

Read off every call site in
`src/neural_forge/hf_lfm2_streaming_vocabulary_margin_adapter.py`, then executed
on both handles. "merged" is `Lfm2MergedSealedEvidenceCache` +
`Lfm2MergedSealedEvidenceSession`.

### 1.1 `_authenticate_selective_session` (pre-change `:416`, now `:447`)

Called three times per build (once up front, once after the fit statistics pass,
once per `Lfm2StreamingVocabularyMarginGateV2._authenticate`, which itself runs
on every `as_dict`, `evaluate_projection` and `evaluate_candidate_replay`).

| # | what it authenticates / reads | merged handle |
|---|---|---|
| 1 | `selective_session.verify()` returns the same object | ✅ merged verify reverifies every member session |
| 2 | `trace_cache.manifest` is a `Mapping` | ✅ merged manifest |
| 3 | **`trace_cache.manifest["manifest_sha256"]` is a sha256** | ❌ **the blocker** — `merged_manifest()` drops it on purpose (`hf_lfm2_sealed_evidence_set.py:1019-1047`) |
| 4 | `trace_cache.cache_key` is a sha256 | ✅ evidence-set cache key |
| 5 | `trace_cache.batch_count` is a positive int | ✅ merged count |
| 6 | session `cache_key` / `batch_count` / `manifest` / `path` equal the cache's | ✅ all four mirror |
| 7 | `selective_session.as_dict()` reproduces its own `session_sha256` seal | ❌ merged `as_dict` is **not** self-sealed: `_evidence_sha256` digests a *different* body (`_merged_session_sha256`, `:280`) |
| 8 | receipt `schema == SELECTIVE_AUTHENTICATED_SESSION_SCHEMA` | ❌ merged schema |
| 9 | receipt `cache_key` / `manifest_sha256` / `batch_count` agree | ❌ no `manifest_sha256` |
| 10 | receipt `completion_sha256` is a sha256 | ❌ merged `as_dict` has none |
| 11 | receipt `ordered_authenticated_batches` has `batch_count` entries indexed `0..n-1` | ❌ merged `as_dict` has none |
| 12 | receipt `authentication_contract` certifies 10 integrity claims | ❌ merged `as_dict` has none |
| 13 | optional `expected_session_sha256` pin | ✅ merged session digest |

### 1.2 `_build_phase_inventory` (`:948`)

| # | reads | merged handle |
|---|---|---|
| 14 | `selective_session.batch_references`, length == `cache_receipt["batch_count"]` | ✅ member references renumbered into merged order |
| 15 | per reference: `index`, `input_sha256`, `ordered_tensor_names`, `record`, `record_sha256`, `tensor_count`, `logical_bytes` | ✅ member's own, index restated |
| 16 | cross-check each against `cache_receipt["ordered_authenticated_batches"][index]` | ❌ needs a merged batch list |
| 17 | `record["tensors"]` inventory equals `ordered_tensor_names` | ✅ |
| 18 | `record["capture"]["phase_diagnostics"]` declares the phase horizon | ✅ |
| 19 | final-layer `residual.post_layer` + `.logits` descriptors per phase (shape/dtype/numel/size/sha256) | ✅ |
| 20 | **emits** an inventory report carrying `cache_key`, `manifest_sha256`, `completion_sha256`, `selective_session_sha256` | ❌ two of those do not exist |

### 1.3 Payload reads — `_SelectedPhaseStream` (`:1187`) and `evaluate_candidate_replay`

| # | reads | merged handle |
|---|---|---|
| 21 | `session.load_selected_tensors(record_index, names)` — one phase's residual+logits | ✅ routed to the sealing member, coordinates restated |
| 22 | payload `batch_index`, `input_sha256`, `ordered_tensor_names`, `session_sha256 == session.evidence_sha256`, `tensors`, `selection_sha256`, `logical_bytes` | ✅ |
| 23 | `session.release_selected_tensors(payload)` and `payload.released` | ✅ member digest restored before release |
| 24 | candidate replay: per-phase `load_selected_tensors((logits_name,))` with `session_sha256 == expected_session_sha256` | ✅ |

### 1.4 Target-bridge branch (`build_..._v2`, bridged carrier runs only)

| # | reads | merged handle |
|---|---|---|
| 25 | `selective_session.batch_references` count == `trace_adapter.record_count`, and `references[i].input_sha256 == records[i].donor_cache_input_sha256` | ✅ the round's merged replay is one target-token record per **merged** batch (`_assert_bridge_replay_covers_evidence_set`, `hf_lfm2_dense_budget_model_fold.py:524`) |

### 1.5 Emitted records that restate the evidence identity

Four, all sealed: the inventory report (`inventory_sha256`), the precomputed
observability `provenance` (`provenance_sha256`), the gate body
(`gate_evidence_sha256`), and `report["cache_authentication"]`
(`adapter_sha256`). Each carried exactly `cache_key`, `manifest_sha256`,
`completion_sha256`, `selective_session_sha256`.

**So merged evidence failed on 3, 7–12, 16 and 20 — nine distinct checks, not
one digest.**

---

## 2. The composite binding

New in the adapter, mirroring the phase binding's shape and its naming rules:

| symbol | line | what |
|---|---|---|
| `LFM2_STREAMING_VOCABULARY_MARGIN_MERGED_EVIDENCE_SCHEMA` | `:88` | the document schema |
| `_MERGED_EVIDENCE_FIELD` (`"merged_sealed_evidence"`) | `:99` | the one field merged evidence adds |
| `_module_file_sha256` | `:547` | source hash helper, same contract as the phase cache's |
| `_evidence_identity_fields` | `:562` | the identity fields every emitted record restates |
| `_merged_member_authentication` | `:588` | **per-member FULL authentication** through `_authenticate_selective_session` |
| `_authenticate_merged_evidence_session` | `:671` | the merged authenticator |
| `_authenticate_evidence_session` | `:901` | fail-closed dispatch |
| `Lfm2StreamingVocabularyMarginGateV2._authenticate` | `:1520` | dispatches, and re-compares the composite |

### 2.1 What it authenticates, in order

1. **Type dispatch, fail-closed.** Merged cache without merged session, merged
   session without merged cache, and any non-handle each raise by name. A pair
   where either side is merged routes to the merged authenticator; everything
   else routes to the untouched single-cache one.
2. **Same evidence set.** `selective_session.evidence_set is trace_cache.evidence_set`.
3. `selective_session.verify()` (reverifies every member session and rechecks it
   still matches its member), returning the same object.
4. The merged manifest must **not** claim a `manifest_sha256`. Fail-closed the
   other way round from the single-cache path: the absence is the invariant.
5. The four cache↔session identity comparisons (checks 6 above).
6. **Per member**, in evidence order: `_authenticate_selective_session(member.cache,
   member.session)` — the whole of §1.1 on a real sealed cache, so the member's
   real `manifest_sha256`, `completion_sha256`, `ordered_authenticated_batches`
   and the 10-claim authentication contract are checked by the proven
   authenticator, plus the member's own lineage
   (`parent_identity`, `corpus`, `layer_topology`, `implementation`,
   `collector_source`, `canonical_contract_source`, `transformers_lfm2_source`,
   `execution`, `capture_selection`, `batch_count`) — the same 14 keys
   `_sealed_trace_cache_binding` binds.
7. Composition: member 0 is `base`, every other is `on_policy_extension` with a
   round manifest and an all-fit split; every extension agrees with the base on
   parent identity, implementation and layer topology.
8. The merged session receipt's schema, `cache_key`, `batch_count`,
   `merged_view_opens_no_new_session`, `parent_forward_calls`, `training_steps`
   and `session_sha256 == evidence_sha256`, **and** its
   `ordered_member_session_sha256` must equal the per-member session digests
   authenticated in 6. The merged digest is opaque on its own; what makes it an
   authentication claim is that every session it names was authenticated.
9. Member batch counts sum to the merged count; the merged
   `ordered_authenticated_batches` (member entries with only `batch_index`
   restated, exactly as the merged session restates a payload's coordinates) is
   `0..n-1`; and it agrees with `evidence_set.iter_batches()` on every
   `input_sha256`.

### 2.2 The document

```
schema                               neural-forge-lfm2-streaming-vocabulary-margin-merged-sealed-evidence-v1
evidence_set_cache_key               the Lfm2SealedEvidenceSet cache key
base_cache_key
on_policy_evidence_identity_sha256
member_count / round_count / batch_count / base_batch_count
extension_batch_indices
merged_corpus_sha256
derived_merged_manifest_sha256       canonical digest of the merged manifest body
derived_merged_completion_sha256     canonical digest of the ordered member completions
sealed_merged_manifest_digest_exists false
every_member_sealed_digest_authenticated true
merged_session_sha256
sealed_evidence_set_source_sha256    source hash of hf_lfm2_sealed_evidence_set.py
ordered_members[]                    role / ordinal / batch_offset / split_sha256 /
                                     round_manifest_sha256 / target_preimage_sha256 /
                                     all_rows_are_fit_rows /
                                     trace_cache -> the member's 14-key sealed lineage,
                                     including its REAL manifest_sha256 and completion_sha256
all_extension_rows_are_fit_rows      true
heldout_extension_batch_count        0   <- ENFORCED, see section 3
extension_batches_are_never_heldout_evidence true
merged_view_opens_no_new_session / parent_forward_calls / training_steps
merged_sealed_evidence_sha256        canonical_sha256 of everything above
```

Three naming rules, two taken from the phase binding and kept:

- **Nothing is fabricated.** For merged evidence the emitted
  `manifest_sha256` and `completion_sha256` are **`None`**, not a derived digest
  dressed as a sealed one. The two derived digests exist, are named `derived_`,
  and sit inside the document next to `sealed_merged_manifest_digest_exists:
  false`. (This is stricter than the phase binding, which reuses the outer field
  names for its derived values; that binding feeds a phase key, this one feeds a
  receipt a human reads.)
- **The composition is reconstructable.** `ordered_members` names which caches,
  in which order, at which batch offsets, with which splits, round manifests and
  target preimages produced the margin evidence, each with the digest its own
  sealed file carries.
- **Nothing is claimed that is not checked here.**
  `all_extension_rows_are_fit_rows` and `heldout_extension_batch_count: 0` are
  not restatements of the sealed split labels: the authenticator is handed the
  held-out record set this evidence is about to be measured against and refuses
  before it builds the document if any child-visited batch is in it (§3).

`_evidence_identity_fields` puts that document into all four emitted records, so
the inventory, the observability provenance, the gate and the adapter report all
carry the same `merged_sealed_evidence_sha256`.

### 2.3 The gate

`_authenticate` dispatches on the session type. For merged evidence it
re-authenticates the evidence set's own memoized merged cache (not a restated
`_GateCacheView`), pins `receipt["cache_key"] == self.cache_key`, and compares
the freshly derived `merged_sealed_evidence_sha256` against the one sealed into
the gate report at construction. For a single cache both sides of that
comparison are absent, so it compares nothing and the plain path is unchanged.

### 2.4 Row slicing — measured, not asserted

v2's memory profile is the row-sliced snapshot-Gram stream
(`_stream_selective_cache_snapshot_grams` → `_authenticated_row_api`), which is
selected exactly when `not vocabulary_margin_observability or
streaming_vocabulary_margin_v2` (`hf_lfm2_balanced_circuit_basis.py:3290-3292`).
`84acded2` gave `Lfm2MergedSealedEvidenceSession` the row pair, so merged
evidence reaches it with v2 on. Counted on one base(4)+extension(2) evidence set
(`C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/a763fdb3-d614-4455-9c15-15950803b63f/scratchpad/measure_read_profile.py`):

```
v2 OFF   whole-tensor loads    6 (7,584 B)   row-sliced loads    0 (    0 B)
v2 ON    whole-tensor loads   24 (4,992 B)   row-sliced loads   18 (4,608 B)
```

v2 OFF reads each merged batch's **entire** tensor inventory in one payload —
that is the ~27.7 GB arm `NOTES-W13F` §5 flagged as unmeasured. v2 ON reads at
most one phase's residual+logits pair (the adapter's own reads, identical in
shape to the plain-cache path and routed to the member that sealed them) or one
bounded row window. Every row payload is released: the merged-evidence test
asserts `len(row_loads) == len(row_releases) > 0` and that extension batches are
among them.

---

## 3. Held-out purity: the split had to become role-aware

Adversarial review found this before commit, and it is the sharpest thing in
this wave. Stated plainly: **this diff created the combination, so this diff has
to fix it.**

### 3.1 The defect

`_split_records` ranked every record by
`sha256(split_seed \0 cache_key \0 record_id)` with no member-role awareness and
never consulted `evidence_set.extension_batch_indices`. On merged evidence that
means a **child-generated** extension batch could be ranked into the held-out
split, and the held-out projection gate would then be measured partly on the
child's own outputs.

That contradicts a rule the rest of the stack enforces by name:
`Lfm2SealedEvidenceRequest.__post_init__`
(`hf_lfm2_sealed_evidence_set.py:168-173`) refuses any non-fit split label for an
extension member because *"selection would let the child pick its own frontier
and validation would stop being held out"* — and my composite was about to seal
`all_extension_rows_are_fit_rows: True` next to a held-out set that violated it.

Measured on the exact ranking, not estimated
(`…/scratchpad/measure_split_leak.py`):

```
base    4 + extension    2 (held-out quota   2): 623/1000 splits leaked >=1 extension batch, mean 0.69 (expected 0.67)
base    8 + extension    2 (held-out quota   3): 526/1000 splits leaked >=1 extension batch, mean 0.58 (expected 0.60)
base  256 + extension  115 (held-out quota  93): 200/200 splits leaked >=1 extension batch, mean 28.84 (expected 28.83)
```

At r13 scale **every** split leaks, ~29 child-generated batches into a 93-batch
held-out set. At toy scale it is a 62% coin flip, which is why the carrier e2e
could have passed while silently doing it.

Before this wave the combination had no executable path — v2 refused merged
evidence outright — so the defect is reachable only through the arm this diff
opens, and it is the production-pinned arm.

### 3.2 The fix, at the split

`hf_lfm2_balanced_circuit_basis.py`:

| symbol | line | what |
|---|---|---|
| `_extension_batch_indices` | `:756` | duck-typed merged batch indices; empty for any single cache or session |
| `_resolved_extension_batch_indices` | `:780` | the one set both handles agree on, fail-closed on disagreement |
| `_split_records(..., extension_batch_indices=frozenset())` | `:798` | narrows the held-out **pool** to the base members |
| `build_lfm2_circuit_balanced_basis` | `:3381` | resolves the set once and passes it to both split call sites |
| `_stream_selective_cache_snapshot_grams(..., extension_batch_indices=None)` | `:1741` | takes it, or derives it from the session it was handed |

The ranking function itself is untouched. What changed is the pool it draws from:
extension records are removed from the eligible set, the quota is
`max(1, min(len(eligible) - 1, ceil(count * validation_fraction)))` — so the
requested held-out size is preserved where the base members can supply it (r13:
93 of 256 base) and always leaves at least one base fit record — and every
extension record is placed in fit. Fail-closed: extension indices that are not
among the records, or fewer than two eligible base records, both raise.

The split report gains five keys **only when extension indices are present**:
`on_policy_extension_record_count`, `heldout_eligible_record_count`,
`heldout_extension_record_count: 0`, `extension_records_are_never_heldout: true`,
`heldout_pool: "base-members-only"`. A single-cache report is unchanged, key for
key.

This applies to **every** merged arm, not just v2 — the margin-off and
observability-only arms W13E enabled had the same defect and get the same fix.

### 3.3 The belt to that brace, in the adapter

`_authenticate_merged_evidence_session` now takes `heldout_record_ids` — the
split the margin evidence is about to be measured against — maps
`evidence_set.extension_batch_indices` through the authenticated merged batch
inventory to their `input_sha256` record IDs, and **raises** if the intersection
is nonempty:

> Merged sealed evidence held out N on-policy extension batch(es); every
> extension row is a fit row, and a child-visited state cannot be held-out
> evidence for the child.

Only then does it build the document with `heldout_extension_batch_count: 0`.
The build passes `sorted(heldout_ids)`; the gate passes its own
`heldout_record_ids` on every re-authentication. Because the enforced count is
zero in every legitimate case, the composite digest is identical whether the
caller supplies the held-out set or not — the field records an enforced fact,
not a caller-dependent one, and a violating caller gets an exception instead of
a different digest.

### 3.4 Proof

- Plain-cache split byte identity: §4(c), executed cross-revision, plus the
  in-suite pin `test_plain_cache_split_is_byte_identical_to_the_role_blind_form`.
- Role awareness:
  `test_merged_evidence_split_never_holds_out_a_child_visited_batch` picks a
  split seed the **pre-fix** ranking provably leaks on (asserted in the test, so
  it fails if the fix is reverted rather than passing by luck) and asserts the
  held-out set and the extension records are disjoint, that every extension
  record is in fit, and that the held-out count still honours the requested
  fraction.
- Adapter enforcement:
  `test_merged_evidence_streaming_v2_refuses_a_heldout_extension_batch`.

---

## 4. Plain-cache byte identity

**(a) Cross-revision, executed.** The pre-change adapter was snapshotted before
the first edit (`07949d6c05cada5f20bf0b46f20c1848460db5a01fc376322068c0aa72f680c8`,
the committed `84acded2` blob) and loaded as a sibling module inside
`neural_forge`; both revisions were then called on **one identical live parent +
sealed cache + open selective session**
(`C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/a763fdb3-d614-4455-9c15-15950803b63f/scratchpad/verify_plain_backcompat.py`):

```
adapter      byte-identical: True     (the whole sealed adapter report)
provenance   byte-identical: True     (PrecomputedDownstreamObservability.as_dict)
gram         byte-identical: True     (018fca1f49703ce863861e06d8830592905b5d533b5eadef69147d319cb1d69f)
gate         byte-identical: True     (gate report)
projection   byte-identical: True     (evaluate_projection, incl. gate_sha256)
candidate    byte-identical: True     (evaluate_candidate_replay, incl. candidate_selected_logits_sha256)
ALL IDENTICAL True
```

Unlike `NOTES-W13E` §4 there is **no** moved dependency here: the adapter's own
records do not hash its source.

**(b) In-suite pin.**
`test_plain_cache_streaming_v2_evidence_identity_is_byte_identical`
(`tests/test_hf_lfm2_balanced_basis_phase_cache.py:1758`) rebuilds the
pre-change `cache_authentication` body verbatim and asserts
`canonical_json_bytes` equality, then asserts `merged_sealed_evidence` is absent
from all four emitted records of a single-cache build.

**(c) The split, cross-revision, executed.** The pre-change
`hf_lfm2_balanced_circuit_basis.py`
(`36d3c965a32cf8f9cbeb5321b81ca3822facfbf0b26d1a8be35012e9b041240b`, the
committed `84acded2` blob) was loaded as a sibling module and both revisions ran
`build_lfm2_circuit_balanced_basis` with streaming v2 on, on **one identical
parent + 6-batch sealed cache + open session**
(`…/scratchpad/verify_plain_split_backcompat.py`):

```
basis_sha256   identical: True     (7e57d61e675b5e0ecaf591176aebd984968af4bc0527bc73ed2124bd7cf4d712)
margin         identical: True
split_report   identical: True     (13 keys, none of the five merged-only ones)
whole_report   identical: True     (the entire sealed basis report)
ALL IDENTICAL  True
```

**What does move, and it is pre-existing.**
`hf_lfm2_balanced_basis_phase_cache._implementation_fingerprint` (`:1420`) hashes
both files this wave edits, so the `basis_implementation` dependency and every
balanced-basis **phase key** rotate — the same property `NOTES-W13E` §4 recorded,
with the same escape hatch
(`_load_existing_lfm2_circuit_balanced_basis_for_live_fold` pops
`basis_implementation` and `numeric_runtime`, so historical entries stay
consumable through `--balanced-basis-existing-phase-key-sha256`). The r13 refold
is a cold balanced basis either way. The three published **global-topology**
phase entries are keyed on the evidence identity, not on these modules, so
`NOTES-W13D` §6.4's ~38 min saving is unaffected.

---

## 5. The full-config composition proof

`test_carrier_production_margin_pin_composes_through_a_round`
(`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py:2123`) — **passed, 151 s**.

The counted carrier ABI (4096 rows / H816 / 229,693,184-element parent) is out
of reach at toy scale, so the proof is in two parts and states which is which:

1. **The routed configuration still constructs with both pins intact.** The
   carrier-routed argument set plus the one-round/one-refold alternation knobs
   builds a `ProductionLfm2CatalogConfig` with `uses_carrier_routes`,
   `dense_vocabulary_margin_observability` and
   `dense_streaming_vocabulary_margin_v2` all true. Clause A was never relaxed
   and is re-asserted here.
2. **The lane that pin selects runs, end to end, on merged evidence.** The
   bridged lane composed on the carrier lane's own residual basis and margins —
   `circuit_balanced`, `--dense-vocabulary-margin-observability`,
   `--dense-streaming-vocabulary-margin-v2`, the balanced-basis phase cache, the
   sealed target bridge — plus a real on-policy round, driven through
   `run_production_catalog_on_policy_alternation`: fold → publish → real round →
   **refold on base+extension** → publish. Both passes `complete`.

What the test reads, in the run's own artifacts:

- the refold consumed the round the base pass sealed; evidence set of 2 members,
  `extension_selection_or_validation_batch_count == 0`;
- the base pass handed the margin adapter an `Lfm2CircuitTraceCache` with a real
  64-hex `manifest_sha256` and no merged document;
- the **refold pass handed it an `Lfm2MergedSealedEvidenceCache` +
  `Lfm2MergedSealedEvidenceSession`** — the exact handle pair that used to be
  refused by name — with `cache_key` equal to the evidence-set cache key and
  `manifest_sha256` **`None`**;
- the emitted document names both members with 64-hex sealed manifest digests,
  roles `["base", "on_policy_extension"]`, the round manifest digest, the
  on-policy evidence identity, and reproduces its own seal;
- `balanced_basis_merged_evidence_bound` fires only in the refold pass and agrees
  with the margin document on evidence-set key, evidence identity and member
  count — two independent bindings over one composition;
- both passes: zero parent forwards after the cache, zero trace regeneration,
  zero training steps, and a child that reloads standalone.

### 5.1 One additive catalog change was needed to get there

`_on_policy_alternation_pass_config` (`experiments/run_lfm2_dynamic_catalog.py:13036`)
suffixed the receipt, diagnostics and publish destination for a refold pass but
not `balanced_basis_phase_event_log`, so a phase-cached alternation wrote both
passes into one log and died on
`FoldPhaseEventLogError: Phase event log mixes compiler run IDs.` The event log
is a per-pass artifact for the same reason the receipt is — each pass is its own
compiler run — so it is now suffixed the same way. The phase cache **root** stays
shared, which is the point of caching it.

This is additive plumbing in the same function that already does this for three
other paths. **Clause A, its conjunction and its message are untouched**; so is
every other validator in that file. Nobody had combined the balanced-basis phase
cache with the in-process alternation before, which is why it had not surfaced;
production is unaffected either way, because there each refold is its own
launcher invocation with its own run directory.

---

## 6. Changes

| file | sha256 (before → after) |
|---|---|
| `src/neural_forge/hf_lfm2_streaming_vocabulary_margin_adapter.py` | `07949d6c…` → `2a949a30…` |
| `src/neural_forge/hf_lfm2_balanced_circuit_basis.py` | `36d3c965…` → `e1091d33…` |
| `experiments/run_lfm2_dynamic_catalog.py` | `1f206948…` → `0232c8a1…` |
| `tests/test_hf_lfm2_balanced_basis_phase_cache.py` | → `b5db00df…` |
| `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` | → `5da38ecd…` |

`hf_lfm2_balanced_circuit_basis.py` carries two changes: the role-aware split
(§3.2) and, at `:3181-3195`, the named refusal replaced by fail-closed dispatch
in the same voice:

```python
    if streaming_vocabulary_margin_v2 and not isinstance(
        selective_session,
        (
            Lfm2CircuitTraceSelectiveAuthenticatedSession,
            Lfm2MergedSealedEvidenceSession,
        ),
    ):
        raise Lfm2CircuitBalancedBasisError(
            "The streaming vocabulary-margin adapter authenticates one sealed "
            "cache session or one merged sealed evidence session."
        )
```

Both names are read from the module namespace, not a prebuilt tuple, for the
substitution reason `NOTES-W13E` §2.3 records.

**Untouched, verified by digest:**

| file | sha256 | |
|---|---|---|
| `src/neural_forge/hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **frozen collector holds** |
| `src/neural_forge/hf_lfm2_sealed_evidence_set.py` | `8e6e608908e6eaee384cb54f52ab064bc0f3806621d89d8239b2d84683b2358c` | as `84acded2` |
| `src/neural_forge/hf_lfm2_balanced_basis_phase_cache.py` | `6bb971d558a573a5d1dba9dcaf11220b8cc8294d457258fc26cbcf08435b872f` | as `84acded2` |

`hf_lfm2_dense_budget_model_fold.py`, `dynamic_closed_loop_fold.py` (Codex's live
uncommitted work), `run_lfm2_exact4x_moving_gauge_v5*` and
`hf_lfm2_on_policy_evidence_round.py` were not opened for edit.

---

## 7. Tests

Ten new / rewritten in `tests/test_hf_lfm2_balanced_basis_phase_cache.py` and
`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`:

| test | line | proves |
|---|---|---|
| `test_merged_evidence_margin_modes_are_exactly_what_the_adapters_support` | `phase:1672` | rewritten — both margin arms now publish a phase entry on merged evidence, the v2 entry names the evidence set, and a handle that is neither session type is still refused by name |
| `test_plain_cache_streaming_v2_evidence_identity_is_byte_identical` | `phase:1760` | §4(b) |
| `test_merged_evidence_streaming_v2_publishes_sealed_margin_evidence` | `phase:1811` | valid sealed margin evidence: identity is the evidence set's, no invented sealed digests, every member's real digests and offsets, the composite reproduces from its own body and appears in all four emitted records, projection gate passes, 0 forwards, bounded row reads all released with extension batches among them |
| `test_merged_evidence_streaming_v2_composite_binds_every_member_identity` | `phase:1941` | a different extension cache, a different round manifest and a different **base split** each change the composite and none collide. The split case is the sharp one: the evidence-set cache key is *identical* there, so it proves the composite is strictly stronger than that key |
| `test_merged_evidence_streaming_v2_fails_closed_on_a_foreign_handle_pair` | `phase:2303` | merged cache + member session, plain cache + merged session, garbage + merged session, and garbage + garbage each fail closed by name; a plain pair still binds one real sealed manifest digest and gains no field |
| `test_plain_cache_split_is_byte_identical_to_the_role_blind_form` | `phase:2026` | §3.4 — the pre-fix `_split_records` body verbatim; ranking, both splits and all 13 report keys unchanged for one sealed cache, and neither handle exposes extension indices |
| `test_merged_evidence_split_never_holds_out_a_child_visited_batch` | `phase:2106` | §3.4 — on a seed the role-blind ranking **provably leaks on**, the held-out set and the extension records are disjoint, every extension record is in fit, and the held-out count still honours the requested fraction |
| `test_merged_evidence_streaming_v2_refuses_a_heldout_extension_batch` | `phase:2187` | §3.3 — the authenticator refuses a held-out set containing a child-visited batch, and a legitimate base-only set produces the same composite an empty one does |
| `test_merged_evidence_gate_refuses_a_changed_composition` | `phase:2238` | the composite's only enforcement point **fires**: a member's `split_sha256` moved after gate construction passes the session-digest and cache-key checks and is caught by the composite comparison, on both `as_dict()` and `evaluate_projection` |
| `test_carrier_production_margin_pin_composes_through_a_round` | `carrier:2123` | §5 — **the deadlock is gone** |

Suites (`.venv/Scripts/python.exe -m pytest -q -p no:randomly`):

| suite | result |
|---|---|
| `test_hf_lfm2_balanced_basis_phase_cache.py` + `test_hf_lfm2_sealed_evidence_set.py` | **48 passed** (40 measured before this wave: 25+15 → 33+15) |
| `test_run_lfm2_dynamic_catalog_carrier_e2e.py` | **9 passed** in 495 s (8 before) |
| `test_run_lfm2_dynamic_structural_compression.py` + `test_hf_lfm2_circuit_trace_cache.py` + `test_dynamic_lfm2_dense_budget_fold_stack.py` | **105 passed, 1 skipped** |
| `test_run_lfm2_dynamic_catalog.py` + `test_run_lfm2_dynamic_catalog_tokenizer_bridge.py` + the five exact4x/solver/streaming suites | **186 passed, 1 skipped, 1 failed** (below) |
| `test_hf_lfm2_dynamic_closed_loop_adapter.py` + `test_hf_lfm2_dense_budget_model_fold_heterogeneous.py` + `test_hf_lfm2_cpcf_selective_evidence.py` | **84 passed** |

`84acded2` committed W13E's three source files and its two notes but **not**
its tests, so both test files I edited already carried W13E's uncommitted
merged-evidence tests when this wave started; the "before" totals above are
working-tree totals measured here, and they match `NOTES-W13E` §8.

The one failure is
`test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure`, which
belongs to Codex's uncommitted `hf_lfm2_dense_budget_model_fold.py` diff. Proved
pre-existing rather than assumed: with my three source files stashed it fails
identically (`DID NOT RAISE Lfm2DenseBudgetFoldError`), and their digests were
re-checked after `git stash pop`.

`ruff check` clean on the two source files and both test files.
`experiments/run_lfm2_dynamic_catalog.py` reports the **same three** pre-existing
`F401` unused imports before and after this change (checked with the file
stashed); my edit adds none.

---

## 8. Caveats

- **No production evidence.** Everything here is toy scale (vocab 31, hidden 8,
  2 layers; 4+2 sealed batches, and 8+2 in the carrier lane). The r13 evidence set
  is 256+115 batches over 73,989 paths. The identity algebra is size-independent;
  the *memory* behaviour of the merged v2 arm at that scale is still unmeasured,
  and §2.4's byte counts are toy counts.
- **Per-member authentication costs per-member work.** The merged authenticator
  runs the single-cache authenticator once per member, and the gate re-runs it on
  every `as_dict` / `evaluate_projection`. `Lfm2CircuitTraceSelectiveAuthenticatedSession.verify()`
  rechecks metadata without rereading trace contents, so this is bounded by the
  member count, not by corpus bytes — but nobody has timed it on 371 batches.
- **Fit/held-out splitting changed for merged evidence, deliberately** (§3).
  Every merged arm -- margins off, observability-only, and v2 -- now draws its
  held-out quota from the base members only, so a merged basis solved by this
  revision differs numerically from one solved by `84acded2`. No merged basis has
  ever been published, so nothing existing is invalidated; single-cache solves are
  byte-identical (§4). The held-out *fraction* is unchanged wherever the base
  members can supply it, which at r13 scale they can (93 of 256).
- **The candidate-replay gate is still refused on extension evidence**
  (`hf_lfm2_dense_budget_model_fold.py:3406`) and cross-tokenizer candidate
  replay is still disabled in the gate itself. Clause A does not require either,
  so neither blocks the r13 arm.
- **Clause A is unchanged and the r13 relaunch still needs a deployment bump.**
  This wave makes the composed configuration executable in source; the server at
  `neural-forge-postpub-e893767a` is pinned to `84acded2` and would need the new
  commit before `nf-launch-r12-round-refold.sh` can be restored to its v2-ON
  form (`…bak-20260805T104136Z`, `NOTES-W13F` §2).
- **Phase keys rotate again** (§3). Expected, pre-existing, and already handled
  by the reuse path.
- **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
- **Not committed.** No `git add`, no commit, no push.
