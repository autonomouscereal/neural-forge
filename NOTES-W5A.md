# NOTES-W5A — the bridged lane carries both preimages, so it can consume its own rounds

Author: wave W5A (bridged on-policy rounds + merged target-token replay)
Date: 2026-07-31
Branch: `codex/lrc-blank-compression` (HEAD at start: `ab3ec99a`)

Closes NOTES-W4A §6 row 3 and §8 item 1: the strict tokenizer bridge — and
therefore the routed carrier lane — is no longer refused as a *lane*. What is
still refused, by name, is one specific *round*: a round captured
parent-canonically has no target-token preimage, so no child replay exists for
its batches.

Files changed:

- `src/neural_forge/hf_lfm2_on_policy_evidence_round.py`
- `src/neural_forge/hf_lfm2_sealed_evidence_set.py` (additive)
- `experiments/run_lfm2_dynamic_catalog.py`
- `src/neural_forge/hf_lfm2_dense_budget_model_fold.py` (the re-assert site)
- `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py` (one error split
  out of a compound message; the check itself was already correct)
- `tests/test_hf_lfm2_on_policy_evidence_round.py`,
  `tests/test_run_lfm2_dynamic_catalog.py`,
  `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`,
  `tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py` (one `match=`
  string that followed the split message)

Deliberately untouched: `hf_lfm2_circuit_trace_cache.py`,
`hf_lfm2_tokenizer_bridge_trace_adapter.py`, every `hf_causal_*`,
`dynamic_causal_operator_promotion.py`, `dynamic_fold_planner.py`,
`dynamic_lfm2_system.py`, `docs/`.

`dynamic_behavior_gate.py`, `dynamic_fold_planner.py`,
`dynamic_fold_registry.py`, and `dynamic_model_inspector.py` already carried
uncommitted working-tree changes when this wave started. None was touched here.

---

## 1. The schema decision: additive block, same round schema

`ROUND_MANIFEST_SCHEMA` stays `neural-forge-lfm2-on-policy-evidence-round-v1`.
A bridged round carries one extra top-level block, `target_preimage`, sealed
under its own `target_preimage_sha256` and covered again by the manifest's own
`manifest_sha256`.

Bumping the round schema was the alternative and it is worse here:

* the loader rejects any schema string it does not recognize, so a v2 bump
  means every reader has to accept two schemas forever, and the difference
  between them would still be "does this document have a preimage block";
* **an unbridged round is byte-identical to the pre-wave revision.** Its body
  gains nothing, its `manifest_sha256` is unchanged, and — because the bridge
  fields are added to `request` *only* when a bridge is supplied — its
  `request_sha256` is unchanged too, so an already sealed round directory is
  still idempotent under the same request. Every sealed round on disk keeps
  loading, and the two new unit tests assert exactly that;
* the consumption gate keys on the block's presence, not on a version number,
  which is the honest predicate: what a consumer needs to know is whether a
  target-token replay exists for these batches.

`Lfm2OnPolicyEvidenceRound` gained `target_preimage`, `carries_target_preimage`,
`target_preimage_bridge_artifact_sha256`, and `parent_cache_input_mode`
(`parent-tokenizer-canonical` or
`target-tokenizer-canonical-then-one-row-mapped-to-donor`).

`Lfm2SealedEvidenceRequest`/`Lfm2SealedEvidenceMember` gained an additive
`target_preimage_sha256`, reported in `as_dict()`/`receipt_fields()` and
**never folded into any identity** — the round manifest digest already covers
it, and moving `evidence_set.cache_key` would move every derived phase key for
no new information.

## 2. What a bridged round actually captures, and why it is not the parent tokenizer

NOTES-W4A recorded the refusal as "an on-policy round re-tokenizes the child's
decoded text with the **parent** tokenizer, so its sealed batches have no
target-token preimage". Closing it needs one more step than adding a field,
and the reason is a hard geometric constraint rather than a preference:

`_bridge_aligned_parent_rows` requires `parent_rows.paths == child_rows.paths`,
and a trace path is `batch.NNNNNN/<phase>/row.RRRR` where the prefill phase
contributes `min(prompt_len, sample_last_tokens)` rows and each forced-decode
step contributes one. So the parent capture batch and the child replay batch
for the same sealed batch must have the *same* forced-decode step count and the
same prompt geometry — and, more importantly, row *i* on both sides has to be
the same position in the same text, because the closed-form solve fits child
features to parent features row by row.

Independently tokenizing the same text with a 65,536-row parent tokenizer and a
4,096-row child tokenizer does not satisfy that: the step counts differ in
general, and even when they coincide the rows are different text positions. A
fold fitted on such a pair is not "a bit lossy", it is misaligned evidence.

So a bridged round captures the parent on the **one-row donor image of the
child's own target tokenization** — which is exactly the parent input mode the
bridged lane's sealed *base* cache was already captured under
(`ResolvedProductionTokenizerBridge.receipt_fields()` has always reported
`parent_cache_input_mode = target-tokenizer-canonical-then-one-row-mapped-to-donor`).
Base and extension evidence therefore stay contract-identical, which is the
property a single closed-form solve over both actually needs.

**This is a deliberate deviation from the literal instruction "the
parent-tokenized batches stay exactly as today".** What stays exactly as today
is (a) every unbridged round, byte for byte, and (b) the parent evaluation
*contract* on the bridged lane: one sanctioned
`build_lfm2_circuit_trace_cache` invocation, one parent pass per batch, the
same capture-selection policy as the base cache, the same manifest shape for
the parent-facing batch record. What changed is which token IDs a *bridged*
round hands the parent — and capturing parent-canonically there would have been
the deviation, because it would not match the base corpus of the same lane.

Both preimages are sealed either way. Per extension batch the manifest carries:

| field | meaning |
|---|---|
| `capture.batches[i].input_sha256` | the donor batch the parent was evaluated on (this is the parent preimage; it is also the extension cache's own `ordered_batch_input_sha256[i]`) |
| `target_preimage.records[i].target_input_ids` / `target_forced_decode_ids` | the child's own tokenization of the same decoded text |
| `…records[i].parent_tokenizer_preimage` | what the **parent tokenizer** would have produced from that same text — counts, an ID digest, and its boundary verdict — recorded with `used_for_parent_capture: false` |

## 3. The alignment design

The child's target rows and the parent's donor rows are the same rows by
construction, so the alignment evidence is about *provenance*, not about
guessing a correspondence:

1. **Decoded-text identity.** The rollout corpus
   (`hf_on_policy_probe`) deduplicates child prefixes by exact decoded UTF-8
   and tokenizes every row independently per tokenizer, refusing any tokenizer
   that loses bytes on decode/re-encode. A bridged round reads the `child`
   tokenization of those rows; each sealed record carries
   `prompt_utf8_sha256` and `trajectory_utf8_sha256` of the exact rows it came
   from, plus `visited_prefix_row_ids`.
2. **Boundary check, same pattern as before.** The seed-prefix tokenization
   must be an exact prefix of the full-trajectory tokenization, or the
   trajectory is rejected with
   `target_retokenization_shifted_the_prompt_boundary` (the unbridged reason
   strings are unchanged; only the role name differs).
3. **One-row transport.** The round builds real
   `Lfm2TargetTokenizedRecord`s and runs them through
   `build_lfm2_strict_tokenizer_bridge_trace_adapter`, then asserts tensor for
   tensor that the batch it is about to hand the parent equals
   `record.as_parent_cache_batch()`. The sealed record keeps
   `alignment_sha256`, `target_cache_input_sha256`,
   `donor_cache_input_sha256`, and `numerically_different_token_count`.
4. **Child identity.** `bridge.verify_live_tokenizers(parent_tokenizer,
   child_tokenizer)` proves in one call that the supplied parent tokenizer is
   the sealed donor and the *reloaded published child's own* tokenizer is the
   sealed target, and that the live pair still reproduces the sealed transport
   profile. The published child's
   `neural_forge_tokenizer_bridge_reference.json` is read and required to name
   the same artifact, profile, and target semantics.

**Consumption re-proves all of it rather than trusting the manifest.**
`build_lfm2_on_policy_round_target_records` rebuilds the records from the
sealed IDs, runs them through the **live** transport profile, and requires the
resulting `donor_cache_input_sha256` to equal the digest the sealed extension
cache actually holds at that index. A tampered preimage whose own seals were
recomputed still fails there, and the e2e asserts exactly that case.

## 4. What consumes the preimage

One extra adapter, and almost nothing else moves.

`ResolvedProductionTokenizerBridge` gained `merged_trace_adapter` — one strict
trace adapter over `(base records, *extension records)` in **merged batch
order**, built by `extend_tokenizer_bridge_with_on_policy_preimages` right
after the merged rows are materialized (so the merged parent calibration
batches already exist and no cache payload is loaded twice). Its
`solve_trace_adapter` property returns the merged adapter when there is one and
the base adapter otherwise, and every solve-time consumer binds
`solve_trace_adapter`:

* `ProductionParentCachePayload.child_replay_batches` — so extension batches
  get a target-token child replay;
* `_project_bridge_parent_cache` and `_bridge_aligned_parent_rows` in the
  trajectory gate, which now also project the **merged** cache through the
  merged session;
* the dense stack (`tokenizer_bridge_trace_adapter=`) and the target
  vocabulary-margin binding.

`Lfm2DynamicClosedLoopAdapter` needed no new check: it already required one
target-token record per sealed parent batch and zipped record donor digests
against the parent cache's batch references with `strict=True`. Once
`parent_cache` is the merged view, that check *is* the merged-coverage check.
Only its error message changed — the record-count case is now its own precise
sentence instead of one clause in a four-way compound message.

`Lfm2DenseBudgetModelOperator`'s re-assert site is now
`_assert_bridge_replay_covers_evidence_set`: one target-token record per merged
sealed batch, each record's donor digest equal to that batch's sealed input
digest, or a `Lfm2DenseBudgetFoldError` that names the merged batch and the
member cache it came from.

### The vocabulary-margin binding

`Lfm2TargetTokenizerVocabularyMarginBinding.__post_init__` cross-checks the
runtime receipt against its own adapter (`trace_adapter_evidence_sha256`,
`paired_record_count`, mapping digest). Handing it the merged adapter with a
base-only receipt would have failed there — not at toy scale, where the routed
lane is unreachable, but on the real routed r12 lane, which is the one that
matters. So `receipt_fields()` now reports the corpus a solve pass actually
binds: `trace_adapter_evidence_sha256`, `paired_input_evidence_sha256`,
`paired_record_count`, and the mapping fields all come from
`solve_trace_adapter` / `solve_paired_input_evidence`. With no extension
evidence those are the base values and the receipt body is byte-identical;
with extension evidence the body additionally carries
`base_trace_adapter_evidence_sha256`, `base_paired_record_count`, and the
`on_policy_*` block, so nothing is lost.

## 5. Where the refusal lives now

`_unconsumed_extension_evidence_reason` no longer has a bridge row or a carrier
row. The bridged branch delegates to `_bridged_lane_target_preimage_gap`, which
loads every `--on-policy-round-manifest` in scope and returns, **naming the
manifest**, either

* `on-policy round manifest <path> seals no target-token preimage: it was
  captured parent-canonically, so a bridged child (which speaks the target
  vocabulary) has no replay for its extension batches`, or
* `on-policy round manifest <path> carries no target-token preimage the bridged
  lane can read: <the authentication error>`

and `None` when every round in scope carries one. With **no** manifests in
scope the answer is `None`: an alternation's future rounds are produced by this
same invocation with the bridge in hand, so they always carry the preimage.
The same check is re-asserted at `_resolve_on_policy_consumed_rounds`, where
the rounds are already authenticated, so a caller that bypasses config
validation still fails closed.

The carrier-routes row was removed rather than restated because its stated
reason was entirely "the carrier lane runs on the strict tokenizer bridge,
whose trace adapter replays only the sealed base corpus" — which is no longer
true. See §7 item 1 for what that does and does not buy the routed lane.

## 6. Verification

Commands (repo root, `-p no:randomly`, `.venv` interpreter):

```
-m pytest tests/test_hf_lfm2_on_policy_evidence_round.py \
    tests/test_hf_lfm2_sealed_evidence_set.py \
    tests/test_run_lfm2_dynamic_catalog.py \
    tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly
```
→ `151 passed, 1 skipped` (~366 s) — the four suites this wave had to keep
green, run together, including the three new bridged e2e tests

```
-m pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly
```
→ `8 passed` (~339 s); the same suite was `6 passed` (~170 s) at `ab3ec99a`

```
-m pytest tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py \
    tests/test_hf_lfm2_dense_budget_model_fold_heterogeneous.py \
    tests/test_hf_lfm2_dynamic_closed_loop_adapter.py -q -p no:randomly
```
→ `80 passed` (~186 s)

```
-m pytest tests/test_run_lfm2_residual_attention_fold.py \
    tests/test_run_lfm2_dynamic_catalog_tokenizer_bridge.py \
    tests/test_prepare_lfm2_tokenizer_bridge_evidence.py \
    tests/test_hf_lfm2_child_closed_loop_refinement.py \
    tests/test_hf_lfm2_child_closed_loop_token_bridge.py -q -p no:randomly
```
→ `45 passed` (~18 s)

```
-m pytest tests/test_dynamic_lfm2_dense_budget_fold_stack.py \
    tests/test_dynamic_lfm2_cpcf_production.py \
    tests/test_run_lfm2_dynamic_catalog_portable_runtime.py \
    tests/test_dynamic_lfm2_producer_first_catalog.py -q -p no:randomly
```
→ `59 passed` (~100 s)

```
-m pytest tests/test_hf_lfm2_circuit_trace_cache.py \
    tests/test_hf_on_policy_probe.py -q -p no:randomly
```
→ `58 passed, 1 skipped` (~18 s)

### New coverage

* `tests/test_hf_lfm2_on_policy_evidence_round.py`: an unbridged round seals no
  preimage, keeps its pre-wave request shape, and names itself in the refusal;
  a non-artifact "bridge" is refused.
* `tests/test_run_lfm2_dynamic_catalog.py`: the lane-gate test now asserts the
  bridged lane is capable with nothing in scope and refuses a named manifest
  otherwise.
* `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py`:
  * `test_bridged_carrier_lane_alternates_fold_round_and_refold` — the real
    thing on the bridged toy configuration: publish → one real sealed round
    carrying both preimages → refold on base+extension → second publication,
    both passes `complete`. It asserts every preimage record is a fit row whose
    donor digest is the sealed extension batch, that the row map actually moved
    IDs (`numerically_different_token_count > 0`, mapped prompt ≠ target
    prompt), that the parent-tokenizer preimage is present and marked
    `used_for_parent_capture: false`, that the merged adapter carries one
    record per merged batch with zero parent forwards, that
    `evidence_split_sha256` / `parent_trace_rows_sha256` / the published
    `evidence_fingerprint` all move while `parent_cache_key` does not, and that
    the only parent forwards in the whole alternation are inside the round's
    single sanctioned capture (measured with a forward hook). It then tampers
    one target token ID **with every manifest seal recomputed** and asserts the
    live transport profile refuses it, tampers without re-sealing and asserts
    the seal refuses it, and strips the preimage block entirely and asserts the
    lane refuses that round by name at configuration time.
  * `test_bridged_carrier_lane_refuses_a_round_without_a_target_preimage`.
  * `test_bridged_carrier_lane_without_extension_evidence_is_unchanged`.

### Measured no-extension byte-identity

`config.identity()` is path-dependent, so a pinned digest is not expressible in
a test. The in-test assertion is that the fold identity mentions neither
`preimage` nor `merged_trace_adapter`, that rebuilding the same arguments
reproduces the same `configuration_sha256`, that the run's
`configuration_sha256` equals `canonical_sha256(config.identity())`, and that
the base-only bridged run's `tokenizer_bridge` receipt carries no `on_policy_*`
key and re-seals from its own body.

The cross-revision measurement was done outside the test, with a detached
`git worktree` at `ab3ec99a` and the *same* toy pack, the *same* run root, and
the *same* frozen argument list handed to both trees (only `sys.path` differs,
and each probe asserts it imported the runner from the tree under test):

| lane | `configuration_sha256` at `ab3ec99a` | on this tree |
|---|---|---|
| plain bridged (`_lane_arguments`, `carrier_routes=False`) | `b2126988a4a1091187b4664ec52ab261488b510fa00d4774aebfd240e1739311` | **identical** |
| unbridged dense core + alternation (`_refold_lane_arguments`) | `44ee27977738b40920f4b7feb059e484197fd0045ab52eceb36695197769b43c` | **identical** |

Both identity documents are also byte-length identical (10,341 and 9,857
bytes). So the fold identity, every derived phase-cache key, and the derived
restart-capsule root are unchanged for both the bridged base-only path and the
unbridged path. The `_refold_lane_arguments` refactor (`_alternation_arguments`
factored out) was checked the same way: `ab3ec99a`'s function and this tree's
produce the *identical* argument list for the same pack.

## 7. Deferred / known limitations

1. **The routed carrier configuration is still unreachable at toy scale**
   (NOTES-1E blocker 1: the sealed 4096-row/H816 ABI and the registered
   229,693,184-element parent), so `--carrier-route-publication` and the
   manual-review dossier are exercised by no test here. Removing the
   carrier-routes row from `_unconsumed_extension_evidence_reason` means the
   routed lane is now gated only on preimage presence; the routed lane's *own*
   machinery consuming merged evidence is **unverified at any scale** and the
   first real run should be treated as such.
2. **`--dense-vocabulary-margin-observability` with extension evidence is
   unexercised.** §4 explains why `receipt_fields()` had to follow the solve
   corpus for the binding to authenticate at all, but the toy carrier lane only
   turns that flag on in the routed configuration, which item 1 pins out of
   reach. The change is argued from the binding's own assertions, not from a
   passing test.
3. **`--dense-macro-cell-attractor-gate` and
   `--dense-vocabulary-margin-candidate-gate` still refuse extension evidence**,
   unchanged from NOTES-W4A §6, as do `--hybrid-model-fold` and
   `--producer-first-model-fold`.
4. **A donor-collapsed target trajectory keeps only its first preimage.** The
   strict transport profile is explicitly not required to be injective, so two
   distinct target chains can share one donor image. The donor batch is
   deduplicated (as it always was), the first target chain is the sealed
   preimage, and the others are recorded in
   `collapsed_target_trajectory_sha256` rather than dropped silently. The
   parent cannot distinguish them; the child replays the first. No toy corpus
   here produces a collapse, so that field is empty in every test artifact.
5. **The target tokenizer must round-trip the rollout text byte-exactly.**
   `strict_byte_identity` is enforced per tokenizer by the probe, and the
   child's tokenization is now load-bearing, so a target tokenizer that loses
   bytes on any child-visited prefix fails the whole round rather than that
   prefix. That is the correct answer but it is a new way for a bridged round
   to fail that an unbridged round never had.
6. **`max_context_tokens` now binds on the longer of the two tokenizations.**
   A 4,096-row target tokenizer needs more tokens for the same text than the
   65,536-row donor, so a bound that was comfortable for parent-canonical
   rounds can reject bridged ones. The failure is explicit and fail-closed, not
   silent truncation, unless
   `truncate_forced_decode_to_capture_budget` is set.
7. **NOTES-W4A §8 items 2 through 7 are unchanged.** `circuit_balanced` and the
   balanced-basis phase cache are still not exercised by any e2e, the toy
   refold still produces a byte-identical child at that scale,
   `--on-policy-child-artifact-directory` is still refused with an alternation,
   round manifests are still consumed by absolute path, and the alternation
   summary is still not a receipt.
