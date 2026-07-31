"""Multi-device sharding for the LFM2 child-replay capture hot path.

The per-batch child replay loop is embarrassingly parallel: the child weights
are frozen for the whole capture (proved by the before/after fingerprint pair
in the collector), every batch is an independent forward, and the sealed
inventory is a pure function of the detached CPU tensors a batch produced.
Nothing in that loop needs a single device.

This module owns only the *scheduling and replication* half of that idea and
deliberately knows nothing about capture internals.  It never imports the
collector, so the dependency runs one way and the whole surface can be driven
from a stub in a CPU-only test:

``plan_replay_shards``
    Validate a device list, prove it is one homogeneous device class, and turn
    a per-batch cost vector into a fixed batch-to-worker assignment.
``materialize_replay_replicas``
    Copy the frozen child once per extra device and prove every replica's
    content fingerprint equals the primary's before any replay runs.
``run_replay_shards``
    Run one thread per worker, each pinned to its own replica, and fail closed
    on the lowest failing batch index so an error is reported identically no
    matter which worker lost the race.

Ordering is never the workers' business.  The caller keys every result by its
original batch index, so the merged output is byte-identical to a single
device run: same tensors, same record bodies, same sealed hashes.
"""

from __future__ import annotations

import concurrent.futures
import copy
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


REPLAY_SHARDING_SCHEMA = "neural-forge-lfm2-child-replay-sharding-v1"
ROUND_ROBIN_SHARD_ASSIGNMENT = "round_robin"
TOKEN_COST_SHARD_ASSIGNMENT = "token_cost"
SHARD_ASSIGNMENTS = (
    ROUND_ROBIN_SHARD_ASSIGNMENT,
    TOKEN_COST_SHARD_ASSIGNMENT,
)

_SHA256_LENGTH = 64
_SHARDABLE_DEVICE_TYPES = ("cpu", "cuda")


class Lfm2ReplayShardingError(RuntimeError):
    """A replay shard plan, replica, or worker could not be authenticated."""


@dataclass(frozen=True, slots=True)
class Lfm2ReplayShardingConfig:
    """Opt-in request to replay one capture point across several devices.

    ``devices`` is the exact worker list, in worker order.  Its first entry
    must be the live child's own device: worker 0 always replays on the
    unmodified child, so a one-device config is by construction the current
    single-device behavior.
    """

    devices: tuple[str, ...]
    assignment: str = ROUND_ROBIN_SHARD_ASSIGNMENT

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple) or not self.devices:
            raise Lfm2ReplayShardingError(
                "Replay sharding requires a nonempty device tuple."
            )
        for value in self.devices:
            if not isinstance(value, str) or not value.strip():
                raise Lfm2ReplayShardingError(
                    "Replay sharding devices must be nonempty device strings."
                )
        if self.assignment not in SHARD_ASSIGNMENTS:
            raise Lfm2ReplayShardingError(
                f"Unknown replay shard assignment {self.assignment!r}; "
                f"expected one of {list(SHARD_ASSIGNMENTS)}."
            )

    @classmethod
    def coerce(cls, value: Any) -> Lfm2ReplayShardingConfig:
        """Accept either this config or a plain sequence of devices."""

        if isinstance(value, cls):
            return value
        if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
            raise Lfm2ReplayShardingError(
                "replay_devices must be a sequence of devices or an "
                "Lfm2ReplayShardingConfig."
            )
        try:
            entries = tuple(value)
        except TypeError as exc:
            raise Lfm2ReplayShardingError(
                "replay_devices must be a sequence of devices or an "
                "Lfm2ReplayShardingConfig."
            ) from exc
        return cls(devices=tuple(str(entry) for entry in entries))

    def as_dict(self) -> dict[str, Any]:
        return {
            "devices": list(self.devices),
            "assignment": self.assignment,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ReplayShardPlan:
    """A fixed, reproducible batch-to-worker assignment over one device class."""

    devices: tuple[str, ...]
    assignment: str
    batch_assignment: tuple[int, ...]
    batch_costs: tuple[int, ...]
    device_type: str
    device_class: str | None

    @property
    def worker_count(self) -> int:
        return len(self.devices)

    @property
    def batch_count(self) -> int:
        return len(self.batch_assignment)

    def batches_for_worker(self, worker_index: int) -> tuple[int, ...]:
        return tuple(
            index
            for index, worker in enumerate(self.batch_assignment)
            if worker == worker_index
        )

    def cost_for_worker(self, worker_index: int) -> int:
        return sum(
            self.batch_costs[index] for index in self.batches_for_worker(worker_index)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_SHARDING_SCHEMA,
            "assignment": self.assignment,
            "worker_count": self.worker_count,
            "devices": list(self.devices),
            "device_type": self.device_type,
            "device_class": self.device_class,
            "batch_assignment": list(self.batch_assignment),
            "batch_costs": list(self.batch_costs),
        }


def normalized_device_key(torch: Any, value: Any) -> str:
    """Return the stable ``cpu`` / ``cuda:N`` key for one device.

    ``cuda`` without an index resolves through the active CUDA device so two
    spellings of the same physical card never look like two workers.
    """

    try:
        resolved = torch.device(value)
    except Exception as exc:
        raise Lfm2ReplayShardingError(
            f"Replay sharding could not resolve device {value!r}."
        ) from exc
    device_type = str(resolved.type)
    if device_type not in _SHARDABLE_DEVICE_TYPES:
        raise Lfm2ReplayShardingError(
            f"Replay sharding supports only {list(_SHARDABLE_DEVICE_TYPES)} "
            f"devices, got {device_type!r}."
        )
    if device_type == "cpu":
        return "cpu"
    index = resolved.index
    if index is None:
        current_device = getattr(getattr(torch, "cuda", None), "current_device", None)
        if not callable(current_device):
            raise Lfm2ReplayShardingError(
                "Replay sharding cannot resolve the active CUDA device index."
            )
        index = int(current_device())
    return f"cuda:{int(index)}"


def _cuda_device_class(torch: Any, keys: Sequence[str]) -> str:
    """Prove every CUDA worker is the same device model, or fail closed."""

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        raise Lfm2ReplayShardingError(
            "Replay sharding requested CUDA devices but CUDA is unavailable."
        )
    device_count = getattr(cuda, "device_count", None)
    if not callable(device_count):
        raise Lfm2ReplayShardingError(
            "Replay sharding cannot enumerate the visible CUDA devices."
        )
    visible = int(device_count())
    get_device_name = getattr(cuda, "get_device_name", None)
    get_device_capability = getattr(cuda, "get_device_capability", None)
    if not callable(get_device_name) or not callable(get_device_capability):
        raise Lfm2ReplayShardingError(
            "Replay sharding cannot authenticate the CUDA device class."
        )
    names: set[str] = set()
    capabilities: set[tuple[int, ...]] = set()
    for key in keys:
        index = int(key.split(":", 1)[1])
        if index < 0 or index >= visible:
            raise Lfm2ReplayShardingError(
                f"Replay shard device {key!r} is outside the {visible} visible "
                "CUDA devices."
            )
        names.add(str(get_device_name(index)))
        capabilities.add(tuple(int(value) for value in get_device_capability(index)))
    if len(names) != 1 or len(capabilities) != 1:
        raise Lfm2ReplayShardingError(
            "Replay sharding requires one identical CUDA device class; got "
            f"names {sorted(names)} and capabilities {sorted(capabilities)}."
        )
    return names.pop()


def _round_robin_assignment(worker_count: int, batch_count: int) -> tuple[int, ...]:
    return tuple(index % worker_count for index in range(batch_count))


def _token_cost_assignment(
    worker_count: int,
    batch_costs: Sequence[int],
) -> tuple[int, ...]:
    """Longest-processing-time-first over the per-batch token cost.

    Ties are broken by the lowest accumulated cost, then the lowest worker
    index, and the batch order is the descending cost with the batch index as
    the tiebreak, so the assignment is a pure function of the cost vector.
    """

    loads = [0] * worker_count
    assignment = [0] * len(batch_costs)
    ordered = sorted(
        range(len(batch_costs)),
        key=lambda index: (-int(batch_costs[index]), index),
    )
    for batch_index in ordered:
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignment[batch_index] = worker
        loads[worker] += int(batch_costs[batch_index])
    return tuple(assignment)


def plan_replay_shards(
    torch: Any,
    config: Lfm2ReplayShardingConfig,
    *,
    primary_device: Any,
    batch_costs: Sequence[int],
) -> Lfm2ReplayShardPlan:
    """Authenticate the device list and fix the batch-to-worker assignment."""

    if not isinstance(config, Lfm2ReplayShardingConfig):
        raise Lfm2ReplayShardingError(
            "plan_replay_shards requires an Lfm2ReplayShardingConfig."
        )
    costs = tuple(int(value) for value in batch_costs)
    if not costs:
        raise Lfm2ReplayShardingError(
            "Replay sharding requires at least one batch to schedule."
        )
    if any(value < 0 for value in costs):
        raise Lfm2ReplayShardingError("Replay shard batch costs must be nonnegative.")
    keys = tuple(normalized_device_key(torch, value) for value in config.devices)
    device_types = {key.split(":", 1)[0] for key in keys}
    if len(device_types) != 1:
        raise Lfm2ReplayShardingError(
            f"Replay sharding requires one device type, got {sorted(device_types)}."
        )
    device_type = device_types.pop()
    primary_key = normalized_device_key(torch, primary_device)
    if keys[0] != primary_key:
        raise Lfm2ReplayShardingError(
            f"The first replay shard device must be the live child's own "
            f"device {primary_key!r}, got {keys[0]!r}."
        )
    device_class: str | None = None
    if device_type == "cuda":
        if len(set(keys)) != len(keys):
            raise Lfm2ReplayShardingError(
                "Replay sharding rejects duplicate CUDA devices: a second "
                "worker on one card adds a resident child copy without a "
                "second stream of compute."
            )
        device_class = _cuda_device_class(torch, keys)
    if len(keys) > len(costs):
        raise Lfm2ReplayShardingError(
            f"Replay sharding was given {len(keys)} workers for {len(costs)} "
            "batches; every worker must receive at least one batch."
        )
    if config.assignment == ROUND_ROBIN_SHARD_ASSIGNMENT:
        assignment = _round_robin_assignment(len(keys), len(costs))
    else:
        assignment = _token_cost_assignment(len(keys), costs)
    return Lfm2ReplayShardPlan(
        devices=keys,
        assignment=config.assignment,
        batch_assignment=assignment,
        batch_costs=costs,
        device_type=device_type,
        device_class=device_class,
    )


def model_state_bytes(torch: Any, model: Any) -> int:
    """Sum the resident parameter and buffer bytes of one materialized model."""

    total = 0
    seen: set[int] = set()
    for value in list(model.parameters()) + list(model.buffers()):
        if not torch.is_tensor(value):
            continue
        storage_id = id(value)
        if storage_id in seen:
            continue
        seen.add(storage_id)
        total += int(value.numel()) * int(value.element_size())
    return total


def _replicate_model(torch: Any, model: Any, device: Any) -> Any:
    """Deep-copy one frozen model with its tensors landing on ``device``.

    ``copy.deepcopy`` honors a pre-seeded memo, so every parameter and buffer
    is created directly on the target device instead of being cloned on the
    source device and moved afterwards.  That keeps the transient peak on the
    source card at zero extra copies.  Tied weights appear once in
    ``parameters()``/``buffers()``, so one memo entry preserves the tie.
    """

    memo: dict[int, Any] = {}
    parameter_type = torch.nn.Parameter
    for parameter in model.parameters():
        replicated = parameter.detach().to(device=device, copy=True)
        memo[id(parameter)] = parameter_type(
            replicated,
            requires_grad=bool(parameter.requires_grad),
        )
    for buffer in model.buffers():
        if id(buffer) in memo:
            # Already replicated as a parameter; keep the Parameter wrapper.
            continue
        memo[id(buffer)] = buffer.detach().to(device=device, copy=True)
    try:
        return copy.deepcopy(model, memo)
    except Exception as exc:
        raise Lfm2ReplayShardingError(
            f"Replay sharding could not replicate the child onto {device}."
        ) from exc


@dataclass(slots=True)
class Lfm2ReplicaSet:
    """The live child plus one verified frozen copy per extra worker device."""

    models: tuple[Any, ...]
    devices: tuple[Any, ...]
    device_keys: tuple[str, ...]
    identity_sha256: str
    resident_bytes: tuple[int, ...]
    materialization_seconds: float
    verification_seconds: float

    @property
    def worker_count(self) -> int:
        return len(self.models)

    @property
    def replica_count(self) -> int:
        """Copies materialized by this module; the primary is not one."""

        return max(0, len(self.models) - 1)

    def verify(self, fingerprint: Callable[[Any], Any]) -> float:
        """Re-prove every replica still hashes to the primary's identity."""

        started = time.perf_counter()
        for index in range(1, len(self.models)):
            observed = str(fingerprint(self.models[index]).identity_sha256)
            if observed != self.identity_sha256:
                raise Lfm2ReplayShardingError(
                    f"Replay shard replica on {self.device_keys[index]} does "
                    "not carry the primary child's exact parameter bytes."
                )
        return time.perf_counter() - started

    def release(self, torch: Any | None = None) -> None:
        """Drop every materialized replica; the live primary is never touched."""

        self.models = self.models[:1]
        self.devices = self.devices[:1]
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache) and any(
            key.startswith("cuda") for key in self.device_keys[1:]
        ):
            empty_cache()

    def as_dict(self) -> dict[str, Any]:
        # ``worker_count`` is deliberately absent: the shard plan already
        # reports it and the two must never disagree in the telemetry.
        return {
            "materialized_replica_count": self.replica_count,
            "child_identity_sha256": self.identity_sha256,
            "resident_child_bytes_per_worker": list(self.resident_bytes),
            "resident_child_bytes_total": int(sum(self.resident_bytes)),
            "replica_materialization_seconds": round(self.materialization_seconds, 6),
            "replica_verification_seconds": round(self.verification_seconds, 6),
        }


def materialize_replay_replicas(
    torch: Any,
    model: Any,
    plan: Lfm2ReplayShardPlan,
    *,
    fingerprint: Callable[[Any], Any],
) -> Lfm2ReplicaSet:
    """Copy the frozen child onto every extra device and prove it landed exact.

    ``fingerprint`` is the collector's own parent fingerprint, which hashes
    detached CPU bytes and is therefore device independent: a replica that
    matches the primary's ``identity_sha256`` carries the same parameters,
    buffers, and configuration.  A mismatch fails closed before any replay
    runs, so no shard can silently execute against different weights.
    """

    primary_identity = str(fingerprint(model).identity_sha256)
    if len(primary_identity) != _SHA256_LENGTH:
        raise Lfm2ReplayShardingError(
            "Replay sharding received a malformed child identity fingerprint."
        )
    devices = [torch.device(plan.devices[0])]
    models = [model]
    resident = [model_state_bytes(torch, model)]
    started = time.perf_counter()
    try:
        for key in plan.devices[1:]:
            device = torch.device(key)
            replica = _replicate_model(torch, model, device)
            devices.append(device)
            models.append(replica)
            resident.append(model_state_bytes(torch, replica))
        materialization_seconds = time.perf_counter() - started

        verification_started = time.perf_counter()
        for index in range(1, len(models)):
            observed = str(fingerprint(models[index]).identity_sha256)
            if observed != primary_identity:
                raise Lfm2ReplayShardingError(
                    f"Replay shard replica on {plan.devices[index]} does not "
                    "carry the primary child's exact parameter bytes."
                )
        verification_seconds = time.perf_counter() - verification_started
    except BaseException:
        # An unusable set is dropped here rather than being kept resident by
        # the propagating traceback's reference to this frame.  Only the extra
        # copies are released; ``model`` belongs to the caller.
        replica = None
        models.clear()
        devices.clear()
        raise
    return Lfm2ReplicaSet(
        models=tuple(models),
        devices=tuple(devices),
        device_keys=tuple(plan.devices),
        identity_sha256=primary_identity,
        resident_bytes=tuple(resident),
        materialization_seconds=materialization_seconds,
        verification_seconds=verification_seconds,
    )


@dataclass(frozen=True, slots=True)
class Lfm2ReplayWorkerReport:
    """Wall-clock accounting for one shard worker; never sealed."""

    worker_index: int
    device: str
    batch_indices: tuple[int, ...]
    assigned_cost: int
    wall_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_index": self.worker_index,
            "device": self.device,
            "batch_count": len(self.batch_indices),
            "batch_indices": list(self.batch_indices),
            "assigned_cost": self.assigned_cost,
            "wall_seconds": round(self.wall_seconds, 6),
        }


def run_replay_shards(
    plan: Lfm2ReplayShardPlan,
    worker_body: Callable[[int, int], None],
) -> tuple[Lfm2ReplayWorkerReport, ...]:
    """Run ``worker_body(worker_index, batch_index)`` one thread per worker.

    Each worker owns exactly one replica, so its batches run strictly in
    ascending batch order on that replica and no two threads ever touch the
    same module's forward hooks.  The first failure stops the remaining
    batches, and the exception that propagates is always the one from the
    lowest batch index, so a shard failure reports identically no matter which
    worker observed it first.
    """

    failures: dict[int, BaseException] = {}
    failure_lock = threading.Lock()
    abort = threading.Event()
    reports: list[Lfm2ReplayWorkerReport | None] = [None] * plan.worker_count

    def run_worker(worker_index: int) -> None:
        batches = plan.batches_for_worker(worker_index)
        started = time.perf_counter()
        try:
            for batch_index in batches:
                if abort.is_set():
                    break
                try:
                    worker_body(worker_index, batch_index)
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    with failure_lock:
                        failures[batch_index] = exc
                    abort.set()
                    break
        finally:
            reports[worker_index] = Lfm2ReplayWorkerReport(
                worker_index=worker_index,
                device=plan.devices[worker_index],
                batch_indices=batches,
                assigned_cost=plan.cost_for_worker(worker_index),
                wall_seconds=time.perf_counter() - started,
            )

    if plan.worker_count == 1:
        run_worker(0)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=plan.worker_count,
            thread_name_prefix="lfm2-replay-shard",
        )
        try:
            futures = [
                executor.submit(run_worker, worker_index)
                for worker_index in range(plan.worker_count)
            ]
            for future in futures:
                future.result()
        finally:
            executor.shutdown(wait=True)
    if failures:
        raise failures[min(failures)]
    resolved = tuple(report for report in reports if report is not None)
    if len(resolved) != plan.worker_count:  # pragma: no cover - defensive
        raise Lfm2ReplayShardingError("A replay shard worker produced no report.")
    return resolved


__all__ = [
    "REPLAY_SHARDING_SCHEMA",
    "ROUND_ROBIN_SHARD_ASSIGNMENT",
    "SHARD_ASSIGNMENTS",
    "TOKEN_COST_SHARD_ASSIGNMENT",
    "Lfm2ReplayShardPlan",
    "Lfm2ReplayShardingConfig",
    "Lfm2ReplayShardingError",
    "Lfm2ReplayWorkerReport",
    "Lfm2ReplicaSet",
    "materialize_replay_replicas",
    "model_state_bytes",
    "normalized_device_key",
    "plan_replay_shards",
    "run_replay_shards",
]
