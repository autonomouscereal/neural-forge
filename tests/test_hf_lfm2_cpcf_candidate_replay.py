from __future__ import annotations

from collections.abc import Mapping
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
)
from neural_forge.hf_lfm2_cpcf_candidate_replay import (  # noqa: E402
    AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
    CpuReferenceLfm2CpcfCandidateReplayEvaluator,
    Lfm2CpcfCandidateReplayBatchResult,
    Lfm2CpcfCandidateReplayError,
    build_lfm2_cpcf_candidate_replay_work_items,
    evaluate_lfm2_cpcf_candidate_states_with_backend,
    replay_lfm2_cpcf_candidate_work_payload,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    Lfm2CpcfShortConvV4SolverPolicy,
    Lfm2CpcfV4ProjectionCase,
    evaluate_candidate_state_target_dtype,
)


def _device(index: int) -> CudaReplayDeviceReceipt:
    return CudaReplayDeviceReceipt.create(
        device_index=index,
        device_uuid=f"GPU-candidate-replay-{index}",
        device_name="NVIDIA GeForce RTX 3090",
        pci_domain_id=0,
        pci_bus_id=index + 1,
        pci_device_id=index + 10,
        capability=(8, 6),
        total_memory_bytes=25_769_803_776,
        multiprocessor_count=82,
        torch_version="2.13.0+cpu-fake-cuda",
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


class _FakePinnedBuffer:
    slot_count = 2
    pinned_host_memory = True
    asynchronous_h2d = True

    def __init__(
        self,
        replica: ReplayReplicaReceipt,
        maximum_bytes_per_slot: int,
    ) -> None:
        self.replica = replica
        self.maximum_bytes_per_slot = maximum_bytes_per_slot
        self.staged = 0

    def stage(
        self,
        item: ReplayWorkItem,
        slot_index: int,
    ) -> StagedReplayPayload:
        self.staged += 1
        return StagedReplayPayload(
            value=item.payload,
            slot_index=slot_index,
            source_payload_sha256=item.payload_sha256,
            pinned_payload_sha256=item.payload_sha256,
            tensor_bytes=1,
        )

    def wait_ready(self, _slot_index: int) -> None:
        return None

    def mark_consumed(self, _slot_index: int) -> None:
        return None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "device_index": self.replica.device.device_index,
            "staged": self.staged,
            "slot_count": 2,
        }

    def close(self) -> None:
        return None


def _fixture() -> tuple[
    dict[str, Mapping[str, Any]],
    tuple[Lfm2CpcfV4ProjectionCase, ...],
    Lfm2CpcfShortConvV4SolverPolicy,
]:
    generator = torch.Generator().manual_seed(8127)
    hidden = 3
    state = {
        "in_proj_weight": torch.randn(
            (3 * hidden, hidden),
            generator=generator,
            dtype=torch.float32,
        ),
        "in_proj_bias": torch.randn(
            (3 * hidden,),
            generator=generator,
            dtype=torch.float32,
        ),
        "fir_kernel": torch.randn(
            (hidden, 3),
            generator=generator,
            dtype=torch.float32,
        ),
        "fir_bias": torch.randn(
            (hidden,),
            generator=generator,
            dtype=torch.float32,
        ),
        "writer_weight": torch.randn(
            (hidden, hidden),
            generator=generator,
            dtype=torch.float32,
        ),
        "writer_bias": torch.randn(
            (hidden,),
            generator=generator,
            dtype=torch.float32,
        ),
    }
    second = {
        name: (
            None
            if value is None
            else (value + 0.01).detach().contiguous()
        )
        for name, value in state.items()
    }
    cases = []
    for index in range(2):
        sequence = 5 + index
        suffix_start = index
        child = torch.randn(
            (2, sequence, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        desired_context = torch.randn(
            child.shape,
            generator=generator,
            dtype=torch.float64,
        )
        desired_write = torch.randn(
            (2, sequence - suffix_start, hidden),
            generator=generator,
            dtype=torch.float64,
        )
        prefix_mask = torch.ones((2, sequence), dtype=torch.bool)
        prefix_mask[1, -1] = False
        cases.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=f"selection-{index}",
                split="selection",
                child_input_history=child,
                desired_context_history=desired_context,
                desired_write_suffix=desired_write,
                suffix_start=suffix_start,
                valid_prefix_mask=prefix_mask,
                valid_suffix_mask=prefix_mask[:, suffix_start:],
            )
        )
    policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype="float32",
    )
    return (
        {"candidate-z": state, "candidate-a": second},
        tuple(cases),
        policy,
    )


def _direct(
    state: Mapping[str, Any],
    cases: tuple[Lfm2CpcfV4ProjectionCase, ...],
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> Mapping[str, Any]:
    return evaluate_candidate_state_target_dtype(
        cases=cases,
        in_proj_weight=state["in_proj_weight"],
        in_proj_bias=state["in_proj_bias"],
        fir_kernel=state["fir_kernel"],
        fir_bias=state["fir_bias"],
        writer_weight=state["writer_weight"],
        writer_bias=state["writer_bias"],
        required_split="selection",
        policy=policy,
    )


def test_injected_cpu_backend_is_byte_equivalent_to_direct_v4_replay() -> None:
    states, cases, policy = _fixture()

    reports, backend, items = (
        evaluate_lfm2_cpcf_candidate_states_with_backend(
            states,
            cases=cases,
            required_split="selection",
            policy=policy,
            evaluator=CpuReferenceLfm2CpcfCandidateReplayEvaluator(),
        )
    )

    assert [item.payload["candidate_id"] for item in items] == [
        "candidate-a",
        "candidate-z",
    ]
    assert reports == {
        candidate_id: _direct(states[candidate_id], cases, policy)
        for candidate_id in sorted(states)
    }
    assert backend["kind"] == "injected_cpu_reference"
    assert backend["closed_form_solve_math_changed"] is False
    assert backend["canonical_candidate_order_restored"] is True
    assert set(backend["compute_device_by_work_id"].values()) == {"cpu"}


def test_work_item_hashes_are_canonical_and_detect_source_mutation() -> None:
    states, cases, policy = _fixture()
    first = build_lfm2_cpcf_candidate_replay_work_items(
        states,
        cases=tuple(reversed(cases)),
        required_split="selection",
        policy=policy,
    )
    second = build_lfm2_cpcf_candidate_replay_work_items(
        dict(reversed(tuple(states.items()))),
        cases=cases,
        required_split="selection",
        policy=policy,
    )

    assert [item.work_id for item in first] == [item.work_id for item in second]
    assert [item.payload_sha256 for item in first] == [
        item.payload_sha256 for item in second
    ]

    first[0].payload["state"]["writer_weight"].add_(1.0)
    with pytest.raises(
        DynamicReplaySchedulingError,
        match="payload seal",
    ):
        ReplayWorkItem(
            ordinal=first[0].ordinal,
            work_id=first[0].work_id,
            shard_key=first[0].shard_key,
            payload=first[0].payload,
            payload_sha256=first[0].payload_sha256,
        )


def test_backend_result_rejects_noncanonical_or_mismatched_outputs() -> None:
    states, cases, policy = _fixture()
    items = build_lfm2_cpcf_candidate_replay_work_items(
        states,
        cases=cases,
        required_split="selection",
        policy=policy,
    )
    outputs = tuple(
        replay_lfm2_cpcf_candidate_work_payload(item.payload)
        for item in items
    )
    devices = {item.work_id: "cpu" for item in items}

    with pytest.raises(
        Lfm2CpcfCandidateReplayError,
        match="does not match work item",
    ):
        Lfm2CpcfCandidateReplayBatchResult.create(
            items=items,
            outputs=tuple(reversed(outputs)),
            compute_device_by_work_id=devices,
            backend_kind="broken_fake",
            execution_report={"deliberately": "reversed"},
            representative_cpu_cuda_parity_passed=False,
        )


def test_authenticated_scheduler_adapter_restores_order_and_binds_receipt() -> None:
    states, cases, policy = _fixture()
    installed = InstalledReplayStateFingerprint.create(
        state_sha256="a" * 64,
        tensor_count=1,
        logical_bytes=4,
    )
    replicas = tuple(
        ReplayReplicaReceipt.create(
            device=_device(index),
            installed_state=installed,
        )
        for index in (0, 4)
    )
    scheduler_policy = ReplaySchedulerPolicy(
        maximum_pinned_bytes_per_slot=1 << 20,
        parity_absolute_tolerance=0.0,
        parity_relative_tolerance=0.0,
        parity_maximum_relative_rmse=0.0,
        parity_minimum_cosine_similarity=0.999999999999,
    )
    device_by_index = {
        replica.device.device_index: replica.device for replica in replicas
    }
    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=scheduler_policy,
        gpu_worker=lambda _item, _replica, payload: (
            replay_lfm2_cpcf_candidate_work_payload(payload)
        ),
        cpu_parity_worker=lambda _item, payload: (
            replay_lfm2_cpcf_candidate_work_payload(payload)
        ),
        device_receipt_probe=lambda index: device_by_index[index],
        installed_state_probe=lambda replica: replica.installed_state,
        cpu_installed_state_probe=lambda: installed,
        host_buffer_factory=lambda replica: _FakePinnedBuffer(
            replica,
            scheduler_policy.maximum_pinned_bytes_per_slot,
        ),
    )

    reports, backend, items = (
        evaluate_lfm2_cpcf_candidate_states_with_backend(
            states,
            cases=cases,
            required_split="selection",
            policy=policy,
            evaluator=(
                AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator(
                    scheduler
                )
            ),
        )
    )

    assert [item.payload["candidate_id"] for item in items] == [
        "candidate-a",
        "candidate-z",
    ]
    assert backend["kind"] == "authenticated_two_cuda_scheduler"
    assert backend["representative_cpu_cuda_parity_passed"] is True
    assert backend["execution_report"][
        "canonical_result_order_restored"
    ] is True
    assert set(backend["compute_device_by_work_id"].values()) == {
        "cuda:0",
        "cuda:4",
    }
    for candidate_id, report in reports.items():
        direct = _direct(states[candidate_id], cases, policy)
        assert report["candidate_state_sha256"] == direct[
            "candidate_state_sha256"
        ]
        assert report["mean_case_relative_rmse"] == direct[
            "mean_case_relative_rmse"
        ]
        assert report["minimum_case_context_cosine_similarity"] == direct[
            "minimum_case_context_cosine_similarity"
        ]
        assert report["report_sha256"] != direct["report_sha256"]


def test_injected_backend_exception_fails_before_any_reports_publish() -> None:
    states, cases, policy = _fixture()

    class BrokenEvaluator:
        def evaluate(self, _items):
            raise RuntimeError("backend exploded")

    with pytest.raises(
        Lfm2CpcfCandidateReplayError,
        match="failed before publication",
    ):
        evaluate_lfm2_cpcf_candidate_states_with_backend(
            states,
            cases=cases,
            required_split="selection",
            policy=policy,
            evaluator=BrokenEvaluator(),
        )
