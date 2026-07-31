# NOTES-3A — merged adjacent-layer captures under a selective session

Author: wave 3A (multi-GPU replay sharding + capture-merge wiring)
Date: 2026-07-31
Scope: `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`,
`src/neural_forge/hf_lfm2_dense_budget_model_fold.py`,
`experiments/run_lfm2_dynamic_catalog.py`, and their tests.

Consumes NOTES-2B (multi-device child-replay sharding) and NOTES-2A finding
§3.2 (`--dense-closed-loop-merge-adjacent-layer-captures` is a no-op in the
production runner).

---

## 1. Verdict on the capture merge: it is expressible, and it now runs

NOTES-2A §3.2 recorded the decline correctly and its rationale was correct for
the code as it stood:

> a selected parent-evidence profile is addressed by one exact layer role, so a
> union selection has no authenticated parent inventory

That is a statement about the *role taxonomy*, not about the evidence. The
missing piece was a role, not a proof. There is now one.

### 1.1 Why the merge needed parent evidence at all

Under a selective authenticated session the child replay is paired against the
active profile's batches:

```python
paired = pair_lfm2_parent_child_replay(
    self.parent_cache,
    replay,
    parent_batches=(None if self.selective_session is None
                    else parent_profile_borrow.batches),
)
```

A merged replay serves layer L's post-downstream consumer and layer M's
pre-producer consumer. Those two consumers address parent rows at two
different layers, so one profile carrying only layer L's inventory cannot
serve both. That, and not the child replay, is what actually blocked the
merge.

### 1.2 The two-role profile

`_merged_selected_parent_profile_role` seals

```
layer_merge.{L:03d}.post_downstream+{M:03d}.pre_producer
```

with `selected_layer_indices == (L, M)` and
`allowed_capture_points == (post_downstream, pre_producer)`.

The honesty property is that **role plus capture point still names exactly one
addressed layer**. The role string carries both `(layer, point)` pairs in
request order, the two merged points are required to be distinct, and each
consumer holds its own counted `_SelectedParentProfileBorrow` sealing its own
capture point. A borrow therefore never gains the ability to address a layer
its unmerged equivalent could not.

The inventory is derived by the same `_selective_parent_tensor_names` code path
as before, once per selected layer, from that layer's own operator kind
(`full_attention` vs `short_conv`). It is asserted in
`test_merged_captures_take_a_two_role_selected_parent_profile` to equal exactly
the set union of the two single-layer inventories the unmerged captures would
have taken — not a superset, not a widened wildcard.

The profile digest changes only by its role, selection, allowed points, and the
per-batch ordered tensor names, all of which it already sealed. No digest
schema changed.

### 1.3 What else had to move, and why each is still fail-closed

* **`supports_merged_child_captures`** now declines only
  `single_pass_causal`. That mode certifies one exact selection against a
  canonical reference replay, so a union selection would buy a second reference
  replay instead of saving one — the original rationale, kept. The separate
  heterogeneous decline is gone because it was subsumed: the certification
  registry is consulted only when
  `_child_heterogeneous_layout is not None and replay_capture_mode ==
  single_pass_causal`, so under canonical prefix replay a heterogeneous child
  takes byte-identical code paths to a homogeneous one. That is asserted from
  both sides in
  `test_heterogeneous_prefix_replay_admits_merged_layer_captures` and
  `test_heterogeneous_single_pass_declines_merged_layer_captures`.
  Dropping this clause is what makes the flag reachable on the production
  carrier lane at all, which pins `--dense-heterogeneous-layout` and
  `--dense-closed-loop-replay-mode prefix_by_prefix_canonical`.

* **`_release_idle_selected_parent_profile` no longer asserts zero borrows.**
  It asserts that every live borrow is one this adapter minted for a merged
  capture whose second consumer runs in the next layer, and that those borrows
  belong to the active profile; anything else still raises
  `"Terminal selected parent profile still has live borrows."` This is the one
  invariant that was genuinely relaxed, and it was relaxed because the merged
  design *requires* a capture to outlive the layer that produced it — the
  compiler already defers the child payload lease across exactly that window.
  A leaked ordinary capture borrow still fails closed, asserted in
  `test_a_leaked_capture_borrow_still_fails_the_terminal_idle_release`.

* **The secondary borrow is acquired before the primary capture**, so the union
  profile is materialized once and the primary capture's own acquisition finds
  the identical contract already active and only counts a borrow. Failure
  between the two releases the secondary borrow and the now-idle profile.

### 1.4 What was proved

* `test_merged_selective_captures_reproduce_the_unmerged_selective_fold` runs
  two complete `ChildClosedLoopCompiler` folds over one sealed parent under two
  selective sessions — one merged, one not — and asserts the committed children
  have identical portable state and topology digests, that every layer's
  validation decision digest is identical, and that the merged run performs 12
  child replays where the unmerged run performs 13.
* `test_toy_carrier_lane_merges_adjacent_layer_captures` runs the r10 launch
  flag set end to end, twice, at toy scale: heterogeneous child, canonical
  prefix replay, `moving_gauge_v3`, strict tokenizer bridge, selective
  authenticated session. Both runs reach the same status with the same failure
  messages, both traverse every required fold stage through
  `dense_closed_loop_completed`, and only the merged run emits
  `child_closed_loop_layer_merged_capture_completed` and
  `child_closed_loop_layer_capture_served_by_merged_replay`.

**Not proved:** any wall-clock number. Both proofs are correctness proofs on a
CPU box. The merge removes one child replay per layer boundary and adds two
selective parent-evidence materializations per boundary (§2), so the net gain
must be measured on the server before it is claimed.

---

## 2. The cost an operator is opting into

The merge is not free under a selective session, and the receipt does not
pretend otherwise.

**Extra parent-evidence materializations.** `_acquire_selected_parent_profile`
releases the active profile whenever the requested contract differs. Without
merging, layer L's four capture points share one `layer.{L}` profile and that
profile is materialized once per layer. With merging the sequence per layer
becomes `layer.L` -> `layer_merge.L+M` -> `layer.M`, so each layer's selected
parent evidence is read from safetensors, descriptor-verified, and cloned to
CPU **three times instead of once**. That is disk and CPU, not GPU, and it is
paid against one saved full child replay over every sealed batch.

**Peak parent residency.** A merged window holds two layers' selected parent
evidence resident simultaneously instead of one. For scale: the `body_pre`
profile that `moving_gauge_v3` already materializes spans *every* layer at two
suffixes each, while a merged layer pair spans two layers at ~14 suffixes each.
On a 16-layer parent those are the same order of magnitude, so this lane
already sustains a comparable peak — but it is a real increase over the
unmerged layer loop and should be watched on the first production run.

**Interaction with replay sharding.** A merged capture retains two layers of
child rows, and every shard worker holds its own capture buffer. Enabling
`--dense-replay-devices` and
`--dense-closed-loop-merge-adjacent-layer-captures` together multiplies those
two effects. `capture_timing["replay_sharding"]["per_worker"][k]
["peak_capture_bytes"]` is the number to size `--max-capture-bytes` against,
and it is summarized into the run receipt.

---

## 3. Requested vs admitted is now sealed from the fold, not inferred

NOTES-2A §3.2 asked for the runner to be honest about a requested-but-
inapplicable merge. The dense fold emits one scalar-only progress event,
`dense_closed_loop_merge_capture_policy`, immediately after the adapter is
constructed and before any capture:

| field | meaning |
|---|---|
| `merge_adjacent_layer_captures_requested` | the compiler config |
| `merge_adjacent_layer_captures_admitted` | `adapter.supports_merged_child_captures()` |
| `replay_capture_mode` | the resolved capture mode |
| `selective_session_bound` | whether a selective session is bound |
| `heterogeneous_child` | whether the selected target is heterogeneous |

`_ProgressJournal` observes that stage and the run receipt seals it under
`closed_loop_replay_policy.merge_adjacent_layer_captures_observed`. A run that
opts in and is declined now says so in the receipt with the three facts that
determine the decision, instead of looking effective.

This replaces inference from the compiler's own
`merge_adjacent_layer_captures_enabled`, which the runner never sees.

---

## 4. Multi-device child-replay sharding (NOTES-2B consumption)

`replay_devices` is threaded end to end:

`--dense-replay-devices` / `$NEURAL_FORGE_REPLAY_DEVICES` ->
`ProductionLfm2CatalogConfig.dense_replay_devices` ->
`Lfm2DenseBudgetFoldConfig.closed_loop_replay_devices` ->
`Lfm2DynamicClosedLoopAdapterConfig.replay_devices` -> both
`capture_lfm2_child_replay` call sites in
`_replay_and_capture_child_with_parent_profile`.

### 4.1 Resolution is per capture point and never fails the run

`_resolve_child_replay_devices` enforces every NOTES-2B §2 rule *before* the
collector would, and returns `(None, reason)` instead of raising:
`primary_device_unshardable`, `batch_count_below_two_workers`,
`child_carries_hooks`, `all_devices_requires_cuda_primary`,
`no_visible_cuda_devices`, `requested_device_unresolved`, `mixed_device_types`,
`primary_device_not_visible`, `cuda_device_class_unavailable`,
`single_worker_after_resolution`. Worker 0 is always the live child's own
device, CUDA indices are deduplicated, cards whose
`get_device_name`/`get_device_capability` differ from the primary's are
dropped and counted, and the worker list is clamped to the batch count.

The reason reaches the run receipt through
`closed_loop_replay_policy.replay_sharding.runtime.first_fallback_reason`.

### 4.2 The device list is deliberately not in the fold identity

`ProductionLfm2CatalogConfig.identity()` excludes `dense_replay_devices` for
the same reason it excludes `portable_runtime_device`: an honoured shard plan
seals byte-identical replay bytes, so including it would cold-invalidate every
derived phase-cache key for a pure execution choice, and would make a sharded
run non-comparable to the serial run it is supposed to reproduce. Asserted in
`test_dense_replay_device_cli_wins_over_the_environment`.

### 4.3 Precedence

CLI `--dense-replay-devices` > `$NEURAL_FORGE_REPLAY_DEVICES` >
`$NEURAL_FORGE_LFM2_REPLAY_DEVICES` (the name NOTES-2B published; kept so an
operator following that note does not silently get a serial run) > unset.

The environment is consulted **only** when `--dense-run-closed-loop` is set. An
exported device list must not change, or fail, a run with no closed-loop
captures to shard; that case is recorded as `request_source="env_ignored"`
rather than erased. An explicit CLI flag without `--dense-run-closed-loop` is a
configuration error.

The single token `all` means every visible device of the primary's class and is
expanded inside the adapter, where torch and the live child's device are both
known.

---

## 5. Deliberately not done

* **The vocabulary-margin candidate-gate replay is still serial.**
  `_evaluate_vocabulary_margin_candidate_gate` calls
  `capture_lfm2_child_replay` on the held-out subset only, and its sealed gate
  report has no field for a fallback reason. Threading `replay_devices` there
  would either add an unrecorded silent fallback or change a sealed report
  body. It replays a small subset; the body fold is where the time is.
* **Parent-cache build sharding** — still out of scope, per NOTES-2B §7.
* **Replica reuse across capture points.** Every capture point materializes and
  releases its own replica set. With merging enabled there are ~25% fewer
  capture points, so the amortization improves, but the ~7 GB ownership
  question NOTES-2B §7 raises is unchanged.
* **A real speedup number.** This box has no CUDA. The first server run should
  diff a sharded capture point against a serial one
  (`a.replay_sha256 == b.replay_sha256`) before any multiplier is reported, and
  read the ratio from
  `sum(per_worker.replay_seconds) / replay_wall_seconds`, which the receipt now
  carries as `replay_sharding.runtime.observed_speedup`.
