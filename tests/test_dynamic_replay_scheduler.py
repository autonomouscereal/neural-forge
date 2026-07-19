from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_replay_scheduler import (  # noqa: E402
    CudaReplayDeviceReceipt,
    DeterministicReplayScheduler,
    DynamicReplaySchedulingError,
    InstalledReplayStateFingerprint,
    ReplayReplicaReceipt,
    ReplaySchedulerPolicy,
    ReplayWorkItem,
    StagedReplayPayload,
    build_replay_schedule_plan,
    compare_cpu_cuda_replay,
    fingerprint_installed_replay_state,
    fingerprint_replay_value,
)
from neural_forge.dynamic_fold_executor import _FoldOnlyTorchGuard  # noqa: E402


def _device(index: int, *, name: str = "NVIDIA GeForce RTX 3090"):
    return CudaReplayDeviceReceipt.create(
        device_index=index,
        device_uuid=f"GPU-test-{index}",
        device_name=name,
        pci_domain_id=0,
        pci_bus_id=index + 1,
        pci_device_id=index + 10,
        capability=(8, 6),
        total_memory_bytes=25_769_803_776,
        multiprocessor_count=82,
        torch_version="2.11.0+cu128",
        cuda_runtime_version="12.8",
        cudnn_version=91_900,
        deterministic_algorithms_enabled=True,
        deterministic_debug_mode=2,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cudnn_allow_tf32=False,
        cuda_matmul_allow_tf32=False,
        cuda_matmul_allow_bf16_reduced_precision_reduction=False,
        cuda_matmul_allow_fp16_reduced_precision_reduction=False,
        float32_matmul_precision="highest",
        cublas_workspace_config=":4096:8",
    )


def _state(character: str = "a") -> InstalledReplayStateFingerprint:
    return InstalledReplayStateFingerprint.create(
        state_sha256=character * 64,
        tensor_count=4,
        logical_bytes=512,
    )


def _replicas(
    *,
    second_name: str = "NVIDIA GeForce RTX 3090",
    second_state: InstalledReplayStateFingerprint | None = None,
) -> tuple[ReplayReplicaReceipt, ReplayReplicaReceipt]:
    first_state = _state()
    return (
        ReplayReplicaReceipt.create(
            device=_device(0),
            installed_state=first_state,
        ),
        ReplayReplicaReceipt.create(
            device=_device(4, name=second_name),
            installed_state=(
                first_state if second_state is None else second_state
            ),
        ),
    )


def _policy(**overrides: Any) -> ReplaySchedulerPolicy:
    values = {
        "maximum_pinned_bytes_per_slot": 4096,
        "parity_absolute_tolerance": 1e-5,
        "parity_relative_tolerance": 1e-5,
        "parity_maximum_relative_rmse": 1e-5,
        "parity_minimum_cosine_similarity": 0.99999,
    }
    values.update(overrides)
    return ReplaySchedulerPolicy(**values)


def _item(
    ordinal: int,
    *,
    shard_key: str | None = None,
    value: float | None = None,
    work_id: str | None = None,
) -> ReplayWorkItem:
    return ReplayWorkItem.create(
        ordinal=ordinal,
        work_id=f"work-{ordinal}" if work_id is None else work_id,
        shard_key=(
            f"shard-{ordinal}" if shard_key is None else shard_key
        ),
        payload={
            "input": torch.tensor(
                [float(ordinal if value is None else value)],
                dtype=torch.float32,
            ),
            "metadata": "sealed",
        },
    )


class _FakePinnedDoubleBuffer:
    slot_count = 2
    pinned_host_memory = True
    asynchronous_h2d = True

    def __init__(
        self,
        replica: ReplayReplicaReceipt,
        *,
        maximum_bytes_per_slot: int,
    ) -> None:
        self.replica = replica
        self.maximum_bytes_per_slot = maximum_bytes_per_slot
        self.stage_slots: list[int] = []
        self.wait_slots: list[int] = []
        self.consumed_slots: list[int] = []
        self.closed = False

    def stage(
        self,
        item: ReplayWorkItem,
        slot_index: int,
    ) -> StagedReplayPayload:
        self.stage_slots.append(slot_index)
        tensor_bytes = int(item.payload["input"].numel()) * int(
            item.payload["input"].element_size()
        )
        return StagedReplayPayload(
            value=item.payload,
            slot_index=slot_index,
            source_payload_sha256=item.payload_sha256,
            pinned_payload_sha256=item.payload_sha256,
            tensor_bytes=tensor_bytes,
        )

    def wait_ready(self, slot_index: int) -> None:
        self.wait_slots.append(slot_index)

    def mark_consumed(self, slot_index: int) -> None:
        self.consumed_slots.append(slot_index)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "device_index": self.replica.device.device_index,
            "slot_count": self.slot_count,
            "maximum_bytes_per_slot": self.maximum_bytes_per_slot,
            "stage_count": len(self.stage_slots),
            "stage_slots": list(self.stage_slots),
            "pinned_host_memory": True,
            "asynchronous_h2d": True,
        }

    def close(self) -> None:
        self.closed = True


def test_replay_value_and_installed_state_hashes_are_canonical() -> None:
    left = {
        "z": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "a": (3, "four"),
    }
    right = {
        "a": (3, "four"),
        "z": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
    }
    assert fingerprint_replay_value(left, require_cpu_tensors=True) == (
        fingerprint_replay_value(right, require_cpu_tensors=True)
    )

    first = fingerprint_installed_replay_state(
        {
            "layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "layer.bias": torch.tensor([1.0, -1.0]),
        }
    )
    second = fingerprint_installed_replay_state(
        {
            "layer.bias": torch.tensor([1.0, -1.0]),
            "layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        }
    )
    assert first == second
    assert first.tensor_count == 2
    assert first.logical_bytes == 32


def test_plan_is_canonical_and_keeps_one_shard_on_one_device() -> None:
    replicas = _replicas()
    items = (
        _item(3, shard_key="trajectory-c"),
        _item(1, shard_key="trajectory-a"),
        _item(0, shard_key="trajectory-a"),
        _item(2, shard_key="candidate-b"),
    )
    first = build_replay_schedule_plan(
        items,
        replicas=replicas,
        policy=_policy(),
    )
    second = build_replay_schedule_plan(
        tuple(reversed(items)),
        replicas=tuple(reversed(replicas)),
        policy=_policy(),
    )

    assert first.plan_sha256 == second.plan_sha256
    assert [value.work_id for value in first.assignments] == [
        "work-0",
        "work-1",
        "work-2",
        "work-3",
    ]
    assert [value.device_index for value in first.assignments] == [0, 0, 4, 0]
    payload_by_ordinal = {item.ordinal: item.payload_sha256 for item in items}
    assert all(
        assignment.input_payload_sha256
        == payload_by_ordinal[assignment.ordinal]
        for assignment in first.assignments
    )
    assert first.as_dict()["dynamic_work_stealing"] is False
    assert first.as_dict()["folding_math_changed"] is False


@pytest.mark.parametrize(
    ("replicas", "match"),
    (
        (
            _replicas(second_name="NVIDIA GeForce RTX 4090"),
            "hardware/runtime homogeneous|not 'NVIDIA GeForce RTX 3090'",
        ),
        (
            _replicas(second_state=_state("b")),
            "byte-identical installed state",
        ),
    ),
)
def test_plan_rejects_nonidentical_replicas(
    replicas: tuple[ReplayReplicaReceipt, ReplayReplicaReceipt],
    match: str,
) -> None:
    with pytest.raises(DynamicReplaySchedulingError, match=match):
        build_replay_schedule_plan(
            (_item(0),),
            replicas=replicas,
            policy=_policy(),
        )


def test_policy_rejects_anything_other_than_two_bounded_slots_and_devices() -> None:
    with pytest.raises(DynamicReplaySchedulingError, match="exactly two"):
        ReplaySchedulerPolicy(expected_device_count=3)
    with pytest.raises(DynamicReplaySchedulingError, match="exactly two slots"):
        ReplaySchedulerPolicy(pinned_buffer_slots_per_device=3)


def test_cpu_cuda_parity_uses_numeric_gate_and_exact_metadata() -> None:
    cpu = {
        "endpoint": torch.tensor([1.0, 2.0, -3.0]),
        "kind": "phase",
    }
    cuda = {
        "endpoint": torch.tensor([1.000001, 1.999999, -3.000001]),
        "kind": "phase",
    }
    report = compare_cpu_cuda_replay(
        cpu,
        cuda,
        policy=_policy(),
        device_index=0,
    )
    assert report["passed"] is True
    assert report["tensor_leaf_count"] == 1

    with pytest.raises(
        DynamicReplaySchedulingError,
        match="parity (structure differs|failed)",
    ):
        compare_cpu_cuda_replay(
            cpu,
            {
                "endpoint": torch.tensor([1.0, 2.0, -3.0]),
                "kind": "different",
            },
            policy=_policy(),
            device_index=0,
        )
    with pytest.raises(DynamicReplaySchedulingError, match="parity failed"):
        compare_cpu_cuda_replay(
            cpu,
            {
                "endpoint": torch.tensor([2.0, 3.0, -4.0]),
                "kind": "phase",
            },
            policy=_policy(),
            device_index=0,
        )


def test_scheduler_runs_parity_then_restores_canonical_output_order() -> None:
    replicas = _replicas()
    policy = _policy()
    buffers: dict[int, _FakePinnedDoubleBuffer] = {}
    gpu_calls: list[tuple[str, int]] = []
    device_by_index = {
        replica.device.device_index: replica.device for replica in replicas
    }
    state_by_index = {
        replica.device.device_index: replica.installed_state
        for replica in replicas
    }
    cpu_probe_calls = 0

    def buffer_factory(replica: ReplayReplicaReceipt):
        buffer = _FakePinnedDoubleBuffer(
            replica,
            maximum_bytes_per_slot=policy.maximum_pinned_bytes_per_slot,
        )
        buffers[replica.device.device_index] = buffer
        return buffer

    def gpu_worker(
        item: ReplayWorkItem,
        replica: ReplayReplicaReceipt,
        payload: Any,
    ) -> Any:
        gpu_calls.append((item.work_id, replica.device.device_index))
        if item.work_id == "work-0":
            time.sleep(0.02)
        return {
            "endpoint": payload["input"] * 2.0,
            "kind": "replay",
        }

    def cpu_worker(_item: ReplayWorkItem, payload: Any) -> Any:
        return {
            "endpoint": payload["input"] * 2.0,
            "kind": "replay",
        }

    def cpu_state_probe() -> InstalledReplayStateFingerprint:
        nonlocal cpu_probe_calls
        cpu_probe_calls += 1
        return replicas[0].installed_state

    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=policy,
        gpu_worker=gpu_worker,
        cpu_parity_worker=cpu_worker,
        device_receipt_probe=lambda index: device_by_index[index],
        installed_state_probe=lambda replica: state_by_index[
            replica.device.device_index
        ],
        cpu_installed_state_probe=cpu_state_probe,
        host_buffer_factory=buffer_factory,
    )
    items = tuple(_item(index) for index in reversed(range(6)))
    parity = _item(
        0,
        value=7.0,
        work_id="parity",
        shard_key="parity",
    )
    result = scheduler.execute(items, parity_item=parity)

    assert [
        float(output["endpoint"].item()) for output in result.outputs
    ] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert result.report["canonical_output_work_ids"] == [
        f"work-{index}" for index in range(6)
    ]
    assert [
        row["device_index"] for row in result.report["scheduled_records"]
    ] == [0, 4, 0, 4, 0, 4]
    assert result.report["parity_passed_before_scheduled_work"] is True
    assert result.report["canonical_result_order_restored"] is True
    assert result.report["folding_math_changed"] is False
    assert cpu_probe_calls == 3
    assert all(buffer.closed for buffer in buffers.values())
    assert all(
        buffer.stage_slots[0] == 0
        and set(buffer.stage_slots).issubset({0, 1})
        and buffer.stage_slots.count(0) >= 2
        and 1 in buffer.stage_slots
        for buffer in buffers.values()
    )
    assert set(gpu_calls[:2]) == {("parity", 0), ("parity", 4)}


def test_scheduler_disables_autograd_in_fold_guard_shard_threads() -> None:
    replicas = _replicas()
    policy = _policy()
    observed_grad_modes: list[tuple[str, int, bool]] = []
    device_by_index = {
        replica.device.device_index: replica.device for replica in replicas
    }

    def gpu_worker(
        item: ReplayWorkItem,
        replica: ReplayReplicaReceipt,
        payload: Any,
    ) -> Any:
        observed_grad_modes.append(
            (
                item.work_id,
                replica.device.device_index,
                torch.is_grad_enabled(),
            )
        )
        return {
            "endpoint": payload["input"] * 2.0,
            "kind": "replay",
        }

    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=policy,
        gpu_worker=gpu_worker,
        cpu_parity_worker=lambda _item, payload: {
            "endpoint": payload["input"] * 2.0,
            "kind": "replay",
        },
        device_receipt_probe=lambda index: device_by_index[index],
        installed_state_probe=lambda replica: replica.installed_state,
        cpu_installed_state_probe=lambda: replicas[0].installed_state,
        host_buffer_factory=lambda replica: _FakePinnedDoubleBuffer(
            replica,
            maximum_bytes_per_slot=policy.maximum_pinned_bytes_per_slot,
        ),
    )
    guard = _FoldOnlyTorchGuard()
    with guard:
        assert torch.is_grad_enabled() is False
        result = scheduler.execute(
            (_item(0), _item(1)),
            parity_item=_item(
                0,
                value=7.0,
                work_id="parity",
                shard_key="parity",
            ),
        )
        assert torch.is_grad_enabled() is False

    scheduled_modes = [
        mode
        for work_id, _device_index, mode in observed_grad_modes
        if work_id != "parity"
    ]
    assert scheduled_modes == [False, False]
    assert len(result.outputs) == 2
    assert guard.autograd_grad_calls == 0


def test_scheduler_shard_nested_autograd_enable_still_fails_closed() -> None:
    replicas = _replicas()
    policy = _policy()
    scheduled_calls: list[str] = []
    device_by_index = {
        replica.device.device_index: replica.device for replica in replicas
    }

    def gpu_worker(
        item: ReplayWorkItem,
        _replica: ReplayReplicaReceipt,
        payload: Any,
    ) -> Any:
        if item.work_id != "parity":
            scheduled_calls.append(item.work_id)
            with torch.enable_grad():
                return payload["input"]
        return payload["input"]

    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=policy,
        gpu_worker=gpu_worker,
        cpu_parity_worker=lambda _item, payload: payload["input"],
        device_receipt_probe=lambda index: device_by_index[index],
        installed_state_probe=lambda replica: replica.installed_state,
        cpu_installed_state_probe=lambda: replicas[0].installed_state,
        host_buffer_factory=lambda replica: _FakePinnedDoubleBuffer(
            replica,
            maximum_bytes_per_slot=policy.maximum_pinned_bytes_per_slot,
        ),
    )
    guard = _FoldOnlyTorchGuard()
    with pytest.raises(
        DynamicReplaySchedulingError,
        match="A replay shard failed; no scheduled outputs were published",
    ):
        with guard:
            scheduler.execute(
                (_item(0),),
                parity_item=_item(
                    0,
                    value=7.0,
                    work_id="parity",
                    shard_key="parity",
                ),
            )

    assert scheduled_calls == ["work-0"]
    assert guard.autograd_grad_calls == 1


def test_scheduler_fails_closed_on_parity_before_scheduled_work() -> None:
    replicas = _replicas()
    policy = _policy()
    work_calls = []
    buffers = []

    def buffer_factory(replica: ReplayReplicaReceipt):
        buffer = _FakePinnedDoubleBuffer(
            replica,
            maximum_bytes_per_slot=policy.maximum_pinned_bytes_per_slot,
        )
        buffers.append(buffer)
        return buffer

    def gpu_worker(
        item: ReplayWorkItem,
        _replica: ReplayReplicaReceipt,
        payload: Any,
    ) -> Any:
        if item.work_id != "parity":
            work_calls.append(item.work_id)
        return payload["input"] + 1.0

    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=policy,
        gpu_worker=gpu_worker,
        cpu_parity_worker=lambda _item, payload: payload["input"],
        device_receipt_probe=lambda index: next(
            replica.device
            for replica in replicas
            if replica.device.device_index == index
        ),
        installed_state_probe=lambda replica: replica.installed_state,
        cpu_installed_state_probe=lambda: replicas[0].installed_state,
        host_buffer_factory=buffer_factory,
    )
    with pytest.raises(DynamicReplaySchedulingError, match="parity failed"):
        scheduler.execute(
            (_item(0), _item(1)),
            parity_item=_item(
                0,
                value=5.0,
                work_id="parity",
                shard_key="parity",
            ),
        )
    assert work_calls == []
    assert all(buffer.closed for buffer in buffers)


def test_scheduler_rejects_post_work_installed_state_drift() -> None:
    replicas = _replicas()
    policy = _policy()
    state_by_index = {
        replica.device.device_index: replica.installed_state
        for replica in replicas
    }
    mutated = False

    def gpu_worker(
        item: ReplayWorkItem,
        replica: ReplayReplicaReceipt,
        payload: Any,
    ) -> Any:
        nonlocal mutated
        if item.work_id != "parity" and not mutated:
            state_by_index[replica.device.device_index] = _state("c")
            mutated = True
        return payload["input"]

    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=policy,
        gpu_worker=gpu_worker,
        cpu_parity_worker=lambda _item, payload: payload["input"],
        device_receipt_probe=lambda index: next(
            replica.device
            for replica in replicas
            if replica.device.device_index == index
        ),
        installed_state_probe=lambda replica: state_by_index[
            replica.device.device_index
        ],
        cpu_installed_state_probe=lambda: replicas[0].installed_state,
        host_buffer_factory=lambda replica: _FakePinnedDoubleBuffer(
            replica,
            maximum_bytes_per_slot=policy.maximum_pinned_bytes_per_slot,
        ),
    )
    with pytest.raises(
        DynamicReplaySchedulingError,
        match="Installed replay state changed at post_work",
    ):
        scheduler.execute(
            (_item(0), _item(1)),
            parity_item=_item(
                0,
                value=5.0,
                work_id="parity",
                shard_key="parity",
            ),
        )


def test_work_item_detects_payload_mutation_before_execution() -> None:
    payload = {"input": torch.tensor([1.0])}
    item = ReplayWorkItem.create(
        ordinal=0,
        work_id="mutable",
        shard_key="mutable",
        payload=payload,
    )
    payload["input"].add_(1.0)
    with pytest.raises(DynamicReplaySchedulingError, match="payload seal"):
        replace(item)


def test_receipt_seals_reject_tampering() -> None:
    device = _device(0)
    with pytest.raises(DynamicReplaySchedulingError, match="seal"):
        replace(device, device_name="tampered")
    state = _state()
    with pytest.raises(DynamicReplaySchedulingError, match="seal"):
        replace(state, logical_bytes=1024)
    assert len(hashlib.sha256(device.receipt_sha256.encode()).hexdigest()) == 64
