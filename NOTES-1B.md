# NOTES-1B — capture/replay hot-path optimization (files outside my ownership)

Written by the agent that optimized `src/neural_forge/hf_lfm2_circuit_trace_cache.py`.
Everything below is a change I deliberately did **not** make because the file is owned
by another stream. Nothing here is required for the optimization to be correct; each
item is additional headroom or a follow-up.

## 1. Redundant child fingerprint before `capture_lfm2_child_replay`

`src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py:6272` computes

```python
live_child_identity_sha256 = fingerprint_lfm2_parent(child_model).identity_sha256
```

immediately before calling `capture_lfm2_child_replay(...)`, which then computes the
same fingerprint again as its own `child_before`. That is a third full-parameter hash
pass per capture point on top of the before/after pair inside the collector.

`fingerprint_lfm2_parent` is now ~7x faster (parallel per-parameter hashing), so this
is no longer expensive in absolute terms, but the call is still strictly redundant.
If that stream wants it removed, the clean shape is for `capture_lfm2_child_replay` to
accept an optional pre-computed `Lfm2ParentFingerprint` and to reuse it for
`child_before` after verifying the caller-supplied identity is internally consistent.
I did not add that parameter because it changes a public signature that several
non-owned modules call.

## 2. Before/after replay fingerprint cannot be deduplicated

The task asked whether the `child_before` / `child_after` pair can be collapsed to one
computation. It cannot be done soundly, and I kept both:

- Replay runs inside `torch.inference_mode()`. InferenceMode suppresses version-counter
  bumps, so a `Tensor._version` memo of parameter state is **not** a valid proof of
  non-mutation for exactly the window the check guards.
- `_ensure_inference` only proves gradients are off; it does not prove no module
  mutated a parameter in place during its forward.
- The `child_after != child_before` comparison is therefore a real integrity check, not
  redundant work.

What I did instead: parallelized the per-parameter hashing inside
`fingerprint_lfm2_parent` (identical digest feed order, identical `identity_sha256`).
Measured 7.26x on a 60-tensor parent on a 12-core box.

## 3. `Lfm2CircuitTraceCache` reader-ABI fingerprint will change

`src/neural_forge/hf_lfm2_cached_global_topology_frontier.py` (around line 950-1100)
builds `GLOBAL_TOPOLOGY_TRACE_CACHE_INTERFACE_SCHEMA` from:

- `list(cache_type.__dataclass_fields__)` — now includes the new
  `_verified_batch_guards` reuse field, and
- `callable_source_sha256s` for `_cpu_tensor`, `_tensor_sha256`, `_tensor_descriptor`,
  `_tensor_bytes`, ... — all of which changed.

So the global-topology phase-cache fingerprint changes with this revision and seeded
phase evidence will cold-recompute once. That is the documented intent of that
fingerprint ("changes with reader ABI") and all 43 tests in
`tests/test_hf_lfm2_cached_global_topology_frontier.py` +
`tests/test_hf_lfm2_retained_topology_evidence.py` pass, but operators re-using a
warm phase-cache root on the server should expect one cold pass.

Note for that stream: `module._tensor_bytes` is part of that fingerprinted surface. I
briefly removed it (the digest paths now hash a zero-copy buffer instead) and had to
restore it; it is retained as the single explicit byte-copy accessor precisely so the
frontier ABI list keeps resolving. Do not assume it is dead code.

## 4. Parent-cache build path is not pipelined

`build_lfm2_circuit_trace_cache` gets the parallel inventory hashing (via
`_record_for` -> `_ordered_tensor_descriptors`) and the async/staged host copies, but
its per-batch `save_file` + `_file_sha256` publication is still serial and on the
critical path. Overlapping batch N's `_publish_batch` with batch N+1's capture is the
same producer/consumer shape I used for the child replay seal pipeline and would be a
straightforward follow-up. I left it out because the brief targeted child replay
capture and because `_publish_batch` writes to disk, which needs its own failure
ordering audit.

## 5. Pre-existing failures observed in the shared tree (not mine)

While validating I ran every test file that touches the trace cache. 11 tests fail.
I verified by reverting only my four owned files to `HEAD` and re-running: **the same
11 fail identically on the pre-change baseline.** They are attributable to other
in-flight streams, not to this work:

- `tests/test_dynamic_lfm2_cpcf_production.py::test_exact_cpcf_bundle_binds_size_bridge_cache_method_and_one_attempt`
  -> `dynamic_target_synthesis.py:232` "Authenticated component-allocation receipt differs..."
- `tests/test_dynamic_lfm2_dense_budget_fold_stack.py` (6 tests)
  -> `dynamic_fold_artifact.py:801` "Stage-path fresh reload inference parameters are not all frozen."
- `tests/test_hf_lfm2_global_topology_phase_cache_exact4x.py::test_no_carrier_exact4x_control_h816_b3_cold_hit_frontier_equality`
- `tests/test_hf_lfm2_long_horizon_terminal_refold.py::test_replays_each_complete_trajectory_once_and_installs_only_accepted_w2`
- `tests/test_prepare_lfm2_tokenizer_bridge_evidence.py::test_prepare_builds_once_then_exact_rerun_reuses_without_parent`
  -> `experiments/run_lfm2_dynamic_catalog.py:4308` `SimpleNamespace` has no
  `final_performance_metrics_diagnostic_only`

## 6. Untestable here: the CUDA staging path

The pinned-staging / `non_blocking=True` / single-synchronization path only engages
when a hook endpoint's source tensor is on a CUDA device. This dev box is CPU-only, so
end-to-end it is never exercised. I covered the branch with two unit tests in
`tests/test_hf_lfm2_child_replay.py` that drive `_HostTransferStage` through a stub
that reports an accelerator device
(`test_staged_accelerator_copies_land_on_one_synchronization`,
`test_staged_accelerator_copies_defer_the_same_finiteness_failure`). Those verify
pinned-buffer pooling, exactly one synchronization per phase, staged ordering, and
deferred finiteness — but **the real speedup must be measured on the 96-core CUDA
server before this is claimed as a win in production.**
