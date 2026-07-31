from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_replay_sharding import (  # noqa: E402
    Lfm2ReplayShardingConfig,
    Lfm2ReplayShardingError,
    materialize_replay_replicas,
    model_state_bytes,
    normalized_device_key,
    plan_replay_shards,
    run_replay_shards,
)


class _PretendCudaTorch:
    """Delegate to real torch while presenting a synthetic CUDA topology.

    ``plan_replay_shards`` only asks the CUDA namespace for availability, the
    visible device count, and each device's name and capability, so this is
    enough to exercise every multi-GPU branch on a CPU-only box.
    """

    def __init__(
        self,
        *,
        names: tuple[str, ...],
        capabilities: tuple[tuple[int, int], ...] | None = None,
        available: bool = True,
        current: int = 0,
    ) -> None:
        self._real = torch
        self._names = names
        self._capabilities = capabilities or tuple((8, 6) for _ in names)
        self.cuda = SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(names),
            get_device_name=lambda index: names[index],
            get_device_capability=lambda index: self._capabilities[index],
            current_device=lambda: current,
        )

    def __getattr__(self, name):
        return getattr(self._real, name)


def _eight_identical_3090s() -> _PretendCudaTorch:
    return _PretendCudaTorch(names=tuple("NVIDIA GeForce RTX 3090" for _ in range(8)))


def _plan(fake, devices, *, primary="cuda:0", costs=(10, 10, 10, 10), assignment=None):
    config = (
        Lfm2ReplayShardingConfig(devices=devices)
        if assignment is None
        else Lfm2ReplayShardingConfig(devices=devices, assignment=assignment)
    )
    return plan_replay_shards(
        fake,
        config,
        primary_device=primary,
        batch_costs=costs,
    )


def test_normalized_device_key_resolves_an_indexless_cuda_device() -> None:
    fake = _PretendCudaTorch(
        names=("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"),
        current=1,
    )

    assert normalized_device_key(fake, "cuda") == "cuda:1"
    assert normalized_device_key(fake, "cuda:0") == "cuda:0"
    assert normalized_device_key(fake, "cpu") == "cpu"
    assert normalized_device_key(fake, torch.device("cpu")) == "cpu"


def test_normalized_device_key_rejects_an_unshardable_device_type() -> None:
    with pytest.raises(Lfm2ReplayShardingError, match="supports only"):
        normalized_device_key(torch, "meta")


def test_eight_identical_cuda_devices_plan_round_robin() -> None:
    fake = _eight_identical_3090s()
    plan = _plan(
        fake,
        tuple(f"cuda:{index}" for index in range(8)),
        costs=tuple(100 + index for index in range(16)),
    )

    assert plan.worker_count == 8
    assert plan.device_type == "cuda"
    assert plan.device_class == "NVIDIA GeForce RTX 3090"
    assert plan.batch_assignment == tuple(index % 8 for index in range(16))
    assert plan.batches_for_worker(0) == (0, 8)
    assert plan.batches_for_worker(7) == (7, 15)
    assert sorted(
        index
        for worker in range(8)
        for index in plan.batches_for_worker(worker)
    ) == list(range(16))
    assert plan.as_dict()["device_class"] == "NVIDIA GeForce RTX 3090"


def test_token_cost_assignment_balances_uneven_records() -> None:
    fake = _eight_identical_3090s()
    plan = _plan(
        fake,
        ("cuda:0", "cuda:1"),
        costs=(100, 10, 10, 10, 10, 10, 10, 10, 10, 10),
        assignment="token_cost",
    )

    # Longest-processing-time-first puts the 100-token record alone and the
    # nine 10-token records on the other worker.
    assert plan.cost_for_worker(0) == 100
    assert plan.cost_for_worker(1) == 90
    assert sorted(plan.batches_for_worker(0) + plan.batches_for_worker(1)) == list(
        range(10)
    )
    # The plan is a pure function of the cost vector.
    assert plan.batch_assignment == _plan(
        fake,
        ("cuda:0", "cuda:1"),
        costs=(100, 10, 10, 10, 10, 10, 10, 10, 10, 10),
        assignment="token_cost",
    ).batch_assignment


def test_mixed_cuda_device_models_fail_closed() -> None:
    fake = _PretendCudaTorch(
        names=("NVIDIA GeForce RTX 3090", "Quadro RTX 8000", "Tesla P40"),
    )

    with pytest.raises(Lfm2ReplayShardingError, match="one identical CUDA device class"):
        _plan(fake, ("cuda:0", "cuda:1"))


def test_mixed_cuda_capabilities_fail_closed() -> None:
    fake = _PretendCudaTorch(
        names=("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"),
        capabilities=((8, 6), (7, 5)),
    )

    with pytest.raises(Lfm2ReplayShardingError, match="one identical CUDA device class"):
        _plan(fake, ("cuda:0", "cuda:1"))


def test_duplicate_cuda_devices_fail_closed() -> None:
    fake = _eight_identical_3090s()

    with pytest.raises(Lfm2ReplayShardingError, match="duplicate CUDA devices"):
        _plan(fake, ("cuda:0", "cuda:1", "cuda:1"))


def test_repeated_cpu_workers_are_allowed() -> None:
    plan = _plan(torch, ("cpu", "cpu"), primary="cpu")

    assert plan.worker_count == 2
    assert plan.device_type == "cpu"
    assert plan.device_class is None


def test_cuda_device_outside_the_visible_set_fails_closed() -> None:
    fake = _PretendCudaTorch(names=("NVIDIA GeForce RTX 3090",) * 2)

    with pytest.raises(Lfm2ReplayShardingError, match="outside the 2 visible"):
        _plan(fake, ("cuda:0", "cuda:5"))


def test_unavailable_cuda_fails_closed() -> None:
    fake = _PretendCudaTorch(names=("NVIDIA GeForce RTX 3090",) * 2, available=False)

    with pytest.raises(Lfm2ReplayShardingError, match="CUDA is unavailable"):
        _plan(fake, ("cuda:0", "cuda:1"))


def test_mixed_device_types_fail_closed() -> None:
    fake = _eight_identical_3090s()

    with pytest.raises(Lfm2ReplayShardingError, match="one device type"):
        _plan(fake, ("cpu", "cuda:0"), primary="cpu")


def test_first_shard_device_must_be_the_live_child_device() -> None:
    fake = _eight_identical_3090s()

    with pytest.raises(Lfm2ReplayShardingError, match="live child's own device"):
        _plan(fake, ("cuda:1", "cuda:2"), primary="cuda:0")


def test_more_workers_than_batches_fails_closed() -> None:
    with pytest.raises(Lfm2ReplayShardingError, match="every worker must receive"):
        _plan(torch, ("cpu", "cpu", "cpu"), primary="cpu", costs=(5, 5))


def test_unknown_assignment_fails_closed() -> None:
    with pytest.raises(Lfm2ReplayShardingError, match="Unknown replay shard"):
        Lfm2ReplayShardingConfig(devices=("cpu",), assignment="whatever")


def test_config_coerces_a_plain_device_sequence() -> None:
    coerced = Lfm2ReplayShardingConfig.coerce(["cpu", torch.device("cpu")])

    assert coerced.devices == ("cpu", "cpu")
    assert coerced.assignment == "round_robin"
    assert Lfm2ReplayShardingConfig.coerce(coerced) is coerced
    with pytest.raises(Lfm2ReplayShardingError, match="sequence of devices"):
        Lfm2ReplayShardingConfig.coerce("cuda:0")


class _TinyFrozen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(4, 4, bias=True)
        self.second = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("scale", torch.arange(4, dtype=torch.float32))
        # A tied weight must survive replication as one shared tensor.
        self.second.weight = self.first.weight


def _content_fingerprint(model):
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().to(device="cpu").contiguous().numpy().tobytes())
    return SimpleNamespace(identity_sha256=digest.hexdigest())


def test_replicas_are_materialized_verified_and_share_no_storage() -> None:
    model = _TinyFrozen()
    plan = _plan(torch, ("cpu", "cpu", "cpu"), primary="cpu", costs=(1, 1, 1))
    replicas = materialize_replay_replicas(
        torch,
        model,
        plan,
        fingerprint=_content_fingerprint,
    )

    assert replicas.worker_count == 3
    assert replicas.replica_count == 2
    assert replicas.models[0] is model
    assert replicas.identity_sha256 == _content_fingerprint(model).identity_sha256
    assert replicas.resident_bytes[0] == model_state_bytes(torch, model)
    assert set(replicas.resident_bytes) == {model_state_bytes(torch, model)}
    for replica in replicas.models[1:]:
        assert replica is not model
        assert torch.equal(replica.first.weight, model.first.weight)
        assert (
            replica.first.weight.data_ptr() != model.first.weight.data_ptr()
        ), "a replica must not alias the primary's storage"
        # The tie is preserved rather than split into two copies.
        assert replica.second.weight is replica.first.weight
    # Mutating a replica cannot reach the primary.
    with torch.no_grad():
        replicas.models[1].first.bias.add_(1.0)
    assert not torch.equal(replicas.models[1].first.bias, model.first.bias)

    replicas.release(torch)
    assert replicas.worker_count == 1


def test_a_replica_that_does_not_match_the_primary_fails_closed() -> None:
    model = _TinyFrozen()
    plan = _plan(torch, ("cpu", "cpu"), primary="cpu", costs=(1, 1))
    seen: list[int] = []

    def drifting_fingerprint(value):
        seen.append(id(value))
        return SimpleNamespace(identity_sha256=f"{len(seen):064d}")

    with pytest.raises(Lfm2ReplayShardingError, match="exact parameter bytes"):
        materialize_replay_replicas(
            torch,
            model,
            plan,
            fingerprint=drifting_fingerprint,
        )
    assert len(seen) == 2


def test_a_malformed_child_fingerprint_fails_closed() -> None:
    model = _TinyFrozen()
    plan = _plan(torch, ("cpu", "cpu"), primary="cpu", costs=(1, 1))

    with pytest.raises(Lfm2ReplayShardingError, match="malformed child identity"):
        materialize_replay_replicas(
            torch,
            model,
            plan,
            fingerprint=lambda _model: SimpleNamespace(identity_sha256="short"),
        )


def test_post_replay_verification_rejects_a_mutated_replica() -> None:
    model = _TinyFrozen()
    plan = _plan(torch, ("cpu", "cpu"), primary="cpu", costs=(1, 1))
    replicas = materialize_replay_replicas(
        torch,
        model,
        plan,
        fingerprint=_content_fingerprint,
    )
    assert replicas.verify(_content_fingerprint) >= 0.0

    with torch.no_grad():
        replicas.models[1].first.weight.add_(1.0)
    with pytest.raises(Lfm2ReplayShardingError, match="exact parameter bytes"):
        replicas.verify(_content_fingerprint)


def test_shard_workers_run_their_own_batches_concurrently() -> None:
    plan = _plan(
        torch,
        ("cpu", "cpu", "cpu"),
        primary="cpu",
        costs=tuple(1 for _ in range(9)),
    )
    observed: dict[int, list[int]] = {0: [], 1: [], 2: []}
    lock = threading.Lock()
    # Every worker's first batch must be in flight at the same time, so the
    # barrier only clears when all three shards really run concurrently.
    overlapped = threading.Barrier(3)

    def worker_body(worker_index: int, batch_index: int) -> None:
        if batch_index < 3:
            overlapped.wait(timeout=30)
        with lock:
            observed[worker_index].append(batch_index)

    reports = run_replay_shards(plan, worker_body)

    assert overlapped.broken is False
    assert observed == {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5, 8]}
    assert [report.worker_index for report in reports] == [0, 1, 2]
    assert [report.batch_indices for report in reports] == [
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
    ]
    assert all(report.wall_seconds >= 0.0 for report in reports)
    assert reports[0].as_dict()["batch_count"] == 3


def test_a_single_worker_plan_runs_on_the_calling_thread() -> None:
    plan = _plan(torch, ("cpu",), primary="cpu", costs=(1, 1))
    calling_thread = threading.get_ident()
    seen: list[int] = []

    run_replay_shards(
        plan,
        lambda _worker, batch_index: seen.append(threading.get_ident()),
    )

    assert seen == [calling_thread, calling_thread]


def test_shard_failure_reports_the_lowest_failing_batch_index() -> None:
    plan = _plan(
        torch,
        ("cpu", "cpu"),
        primary="cpu",
        costs=tuple(1 for _ in range(6)),
    )
    started = threading.Barrier(2)

    def worker_body(worker_index: int, batch_index: int) -> None:
        if batch_index in {0, 1}:
            # Both workers reach their failing batch before either can win.
            started.wait(timeout=10)
            raise ValueError(f"batch {batch_index} failed")

    with pytest.raises(ValueError, match="batch 0 failed"):
        run_replay_shards(plan, worker_body)


def test_a_shard_failure_stops_the_remaining_batches() -> None:
    plan = _plan(
        torch,
        ("cpu", "cpu"),
        primary="cpu",
        costs=tuple(1 for _ in range(8)),
    )
    completed: list[int] = []
    lock = threading.Lock()

    def worker_body(worker_index: int, batch_index: int) -> None:
        if batch_index == 0:
            raise RuntimeError("first batch failed")
        with lock:
            completed.append(batch_index)

    with pytest.raises(RuntimeError, match="first batch failed"):
        run_replay_shards(plan, worker_body)
    # Worker 0 owns 0, 2, 4, 6 and abandons them after batch 0 fails.
    assert not {2, 4, 6}.intersection(completed)
