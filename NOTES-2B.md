# NOTES-2B — multi-GPU child-replay sharding (integration knobs for the runner)

Written by the agent that added multi-device sharding to
`src/neural_forge/hf_lfm2_circuit_trace_cache.py` and
`src/neural_forge/hf_lfm2_replay_sharding.py`.

Everything below is what a caller outside my file ownership needs in order to
turn this on. Nothing here is required for correctness — the default is exactly
the current single-device behavior, byte for byte.

## 1. The knob

`capture_lfm2_child_replay` gained one new keyword-only parameter:

```python
capture_lfm2_child_replay(
    child_model,
    replay_batches,
    selected_layer_indices=...,
    config=...,
    input_device=...,
    replay_capture_mode=...,
    child_evidence_profile=...,
    replay_devices=None,          # <- new; None keeps the current behavior
)
```

`replay_devices` accepts either spelling:

```python
# 1. a plain device sequence, round-robin assignment
replay_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3",
                "cuda:4", "cuda:5", "cuda:6", "cuda:7")

# 2. a config object, if you want the cost-balanced assignment
from neural_forge.hf_lfm2_replay_sharding import Lfm2ReplayShardingConfig
replay_devices=Lfm2ReplayShardingConfig(
    devices=tuple(f"cuda:{i}" for i in range(8)),
    assignment="token_cost",     # or "round_robin" (default)
)
```

`Lfm2ReplayShardingConfig`, `Lfm2ReplayShardingError`, and `Lfm2ReplayShardPlan`
are re-exported from `neural_forge.hf_lfm2_circuit_trace_cache`, so a caller
that already imports the collector does not need a second import.

## 2. Hard rules the call must satisfy (all fail closed)

1. **`replay_devices[0]` must be the live child's own device.** Worker 0 always
   replays the unmodified `child_model`; the sealed envelope's device-bound
   fields (`execution.input_device_type`, the numeric policy's
   `execution_device`, the single-pass causal contract) are computed once from
   that device. If the child is on `cuda:0`, the list starts with `cuda:0`.
   `"cuda"` without an index resolves through `torch.cuda.current_device()`.
2. **One device type per call.** `("cpu", "cuda:0")` is rejected.
3. **One CUDA device class per call.** Every index must report the same
   `torch.cuda.get_device_name` *and* the same `torch.cuda.get_device_capability`.
   On the target server that means the 8 identical cards only — do not mix an
   RTX 3090 shard with an RTX 8000 or a P40.
4. **No duplicate CUDA indices.** Two workers on one card would double the
   resident child without a second stream of compute. Duplicate `"cpu"` entries
   *are* allowed; that is the CPU simulation path.
5. **At least one batch per worker.** A capture point with 51 records can use
   8 devices; a capture point with 3 records cannot. The runner should clamp:
   `devices[: min(len(devices), record_count)]`.
6. Only `cpu` and `cuda` are shardable. Anything else is rejected before any
   replica is built.
7. **The child must carry no forward hooks at call time.** A replica is a deep
   copy, so a caller-installed hook would be duplicated onto it while its
   closure still points at the primary; the fingerprint proves parameter bytes,
   not callbacks. Single-pass causal replay already demanded the same unhooked
   runtime. If the runner instruments the child for diagnostics, remove the
   handles before the capture call or leave `replay_devices=None`.

Every one of these raises `Lfm2CircuitTraceCaptureError` from
`capture_lfm2_child_replay` (the underlying `Lfm2ReplayShardingError` is
re-wrapped so existing `except Lfm2CircuitTraceCaptureError` handlers keep
working).

## 3. Suggested runner integration

The sharding is per capture point, so the natural place is wherever the fold
currently calls `capture_lfm2_child_replay`. A minimal, safe shape:

```python
def _replay_devices_for(child_model, record_count):
    """Return None (current behavior) or a validated device tuple."""
    import os
    import torch

    requested = os.environ.get("NEURAL_FORGE_LFM2_REPLAY_DEVICES", "").strip()
    if not requested or not torch.cuda.is_available():
        return None
    primary = next(child_model.parameters()).device
    if primary.type != "cuda":
        return None
    if requested == "all":
        indices = list(range(torch.cuda.device_count()))
    else:
        indices = [int(part) for part in requested.split(",") if part.strip()]
    primary_index = primary.index if primary.index is not None else torch.cuda.current_device()
    # Worker 0 must be the child's own card.
    indices = [primary_index] + [i for i in indices if i != primary_index]
    # Every worker needs at least one record, and only identical cards qualify.
    name = torch.cuda.get_device_name(primary_index)
    indices = [i for i in indices if torch.cuda.get_device_name(i) == name]
    indices = indices[: max(1, min(len(indices), record_count))]
    if len(indices) < 2:
        return None
    return tuple(f"cuda:{i}" for i in indices)
```

Then pass `replay_devices=_replay_devices_for(child_model, len(records))`.
Returning `None` is always safe and always reproduces today's bytes.

I did **not** wire this into `experiments/run_lfm2_dynamic_catalog.py`,
`hf_lfm2_dense_budget_model_fold.py`, `hf_lfm2_dynamic_closed_loop_adapter.py`,
or `dynamic_closed_loop_fold.py` — those are owned by other streams.

## 4. Memory budget the runner should reason about

Each extra device holds one resident copy of the frozen child. For the ~230M
parameter FP32 child that is roughly 1 GB per card against 24 GB, so an 8-way
shard costs ~7 GB total spread across 7 extra cards, not 7 GB on one.

The replica is materialized with a pre-seeded `copy.deepcopy` memo, so every
parameter and buffer is allocated **directly on the target device**. There is
no transient double-residency on the source card, which is the failure mode a
naive `copy.deepcopy(model).to("cuda:k")` would have.

Activation/capture peak is unchanged per worker — each worker replays one batch
at a time exactly as the serial loop did — but there are now N of them
concurrently. `capture_timing["replay_sharding"]["per_worker"][k]
["peak_capture_bytes"]` reports the largest single-batch capture that worker
saw, so an operator can size `max_capture_bytes` against the real per-device
peak rather than guessing.

## 5. Telemetry the runner can log

`Lfm2ChildReplay.capture_timing` is unsealed wall-clock accounting (it is not
in `metadata` and never enters a hash). Two additions:

- `peak_capture_bytes` — now present on every capture, sharded or not.
- `replay_sharding` — present only when `replay_devices` was supplied:

```
replay_sharding: {
  schema, assignment, worker_count, devices, device_type, device_class,
  batch_assignment,           # worker index per batch index
  batch_costs,                # token cost per batch index
  materialized_replica_count,
  child_identity_sha256,      # the fingerprint every replica had to match
  resident_child_bytes_per_worker, resident_child_bytes_total,
  replica_materialization_seconds, replica_verification_seconds,
  replay_wall_seconds,        # elapsed, not summed
  per_worker: [{worker_index, device, batch_count, batch_indices,
                assigned_cost, wall_seconds, replay_seconds,
                device_synchronizations, staged_host_transfers,
                direct_host_transfers, peak_capture_bytes,
                resident_child_bytes}]
}
```

Speedup is `sum(per_worker[k].replay_seconds) / replay_wall_seconds`. The
top-level `replay_seconds` remains the *sum* across workers so it stays
comparable to a single-device run; `replay_wall_seconds` is the elapsed time.

## 6. What is proved, and what is not

Proved on this CPU-only box (`tests/test_hf_lfm2_child_replay.py`,
`tests/test_hf_lfm2_replay_sharding.py`):

- A 2-worker `("cpu", "cpu")` shard produces a byte-identical
  `replay_sha256`, canonical metadata, every batch record, and every batch
  tensor versus the single-device run — for prefix-by-prefix replay, for
  single-pass causal replay, and under a point-specific evidence profile.
- 1-, 2-, and 3-worker plans and both assignment strategies all seal the same
  bytes while genuinely producing different batch-to-worker assignments.
- Replicas are fingerprint-verified before *and* after replay; a replica that
  drifts mid-replay fails the capture closed.
- An already hooked child, a device that is not the primary's, and more workers
  than batches are all refused before a single replica is materialized.
- Every CUDA-only branch (device class equality, capability equality,
  duplicate index rejection, visible-device bounds, indexless `cuda`
  resolution, unavailable CUDA) is covered with a stub torch, following the
  same pattern the async-D2H work used.

**Not proved here, and not claimed:** any real speedup number. This box has no
CUDA. The 8-way shard has never executed a CUDA kernel. Measure
`sum(per_worker.replay_seconds) / replay_wall_seconds` on the server before
putting a multiplier in any report.

Also unproved here: that two *different physical* RTX 3090s produce
bit-identical kernel output for the same input. The design relies on that
(same device class, same dtype policy, same kernel selection), and the class
check fails closed on anything else — but the first real 8-way run on the
server should be diffed against a single-device run of the same capture point
before the result is trusted. That is a one-command check:

```python
a = capture_lfm2_child_replay(child, records, ..., replay_devices=None)
b = capture_lfm2_child_replay(child, records, ..., replay_devices=tuple(...))
assert a.replay_sha256 == b.replay_sha256
```

## 7. Follow-ups I deliberately left out

- **CUDA streams / `torch.cuda.Stream` per worker.** Threads are enough: each
  worker drives its own device, and the CUDA runtime already serializes per
  device. A per-worker non-default stream would only matter if two workers
  shared a card, which is rejected.
- **Sharding the parent cache build** (`build_lfm2_circuit_trace_cache`). It
  writes to disk per batch, so it needs its own failure-ordering audit — the
  same reason NOTES-1B §4 left its publication pipeline serial.
- **Replicating onto devices the child does not already fit on.** There is no
  device-memory precheck before materialization; an OOM surfaces as a torch
  error wrapped in `Lfm2ReplayShardingError`. A `torch.cuda.mem_get_info`
  precheck would give a better message.
- **Reusing replicas across capture points.** Each call materializes and
  releases its own set. For 51-205 records per point that is amortized fine,
  but a fold that calls the collector many times in a row would benefit from a
  cached replica set keyed on `child_identity_sha256`. That needs an ownership
  decision about who holds the extra ~7 GB between calls.
