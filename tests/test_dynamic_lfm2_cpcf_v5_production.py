from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace
from typing import Any

import pytest

import neural_forge.dynamic_lfm2_cpcf_adapter as cpcf_adapter_module
import neural_forge.hf_lfm2_cpcf_short_conv_v4 as short_conv_v4_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_cpcf_adapter import (
    LFM2_CPCF_V5_GPU_REPLAY_SCHEMA,
    Lfm2CpcfCatalogError,
    Lfm2CpcfExternalBehaviorGateContract,
    Lfm2CpcfAdmissionEvaluator,
    Lfm2CpcfV5SelectiveBackend,
    Lfm2CpcfFoldStackAdapter,
    Lfm2CpcfV5SelectiveSealedEvidence,
    Lfm2CpcfV5SelectiveStackResources,
    _authenticate_v5_compiler_callable,
    _authenticate_v5_compiler_result,
    _authenticate_v5_gpu_parity_receipt_file,
    _authenticate_v5_runtime_objects,
    _test_only_authenticate_v5_gpu_parity_receipt_file,
    _blank_report,
    _v5_gpu_parity_source_sha256,
)
from neural_forge.dynamic_lfm2_cpcf_production import (
    Lfm2CpcfProductionBundleError,
    V5_100X_SCHEMA,
    build_lfm2_cpcf_v5_100x_production_bundle,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_replay_scheduler import (
    CudaReplayDeviceReceipt,
    DeterministicReplayScheduler,
    InstalledReplayStateFingerprint,
    ReplayReplicaReceipt,
    ReplaySchedulerPolicy,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
from neural_forge.dynamic_fold_executor import FoldEvaluationContext
from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
    CpuReferenceLfm2CpcfCandidateReplayEvaluator,
)
from neural_forge.hf_lfm2_cpcf_model_fold import Lfm2CpcfModelFoldPolicy
from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)
from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (
    LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
    Lfm2CpcfV5EndpointFirstCandidateSelector,
    Lfm2CpcfV5RuntimePolicy,
    lfm2_cpcf_v5_compiler_implementation,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
)
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
)
from tests.test_dynamic_lfm2_cpcf_production import (
    _fixture,
    _reviewed_gate_bound_to_bridge,
    _sha,
    _trace_and_phases,
)
from tests.test_dynamic_lfm2_cpcf_catalog import (
    _install_fake_compile,
    _operator_case,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")


class _DonorTokenizer:
    """One-row superset used to exercise a real strict bridge in synthesis."""

    def __init__(self, target: Any) -> None:
        self.target = target
        for name in (
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
            "unk_token_id",
            "all_special_ids",
        ):
            setattr(self, name, copy.deepcopy(getattr(target, name)))

    def __len__(self) -> int:
        return len(self.target) + 1

    def get_vocab(self) -> dict[str, int]:
        return {
            **self.target.get_vocab(),
            "<v5-donor-only>": len(self.target),
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == len(self.target):
            return "<v5-donor-only>"
        return self.target.convert_ids_to_tokens(token_id)

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        if token_ids == [len(self.target)]:
            return "<v5-donor-only>"
        return self.target.decode(token_ids, **kwargs)


def _device(index: int) -> CudaReplayDeviceReceipt:
    return CudaReplayDeviceReceipt.create(
        device_index=index,
        device_uuid=f"GPU-v5-production-{index}",
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


def _gpu_evaluator_and_report() -> tuple[Any, dict[str, Any]]:
    installed = InstalledReplayStateFingerprint.create(
        state_sha256=_sha("v5-installed-state"),
        tensor_count=1,
        logical_bytes=4,
    )
    replicas = tuple(
        ReplayReplicaReceipt.create(
            device=_device(index),
            installed_state=installed,
        )
        for index in (0, 1)
    )
    policy = ReplaySchedulerPolicy(
        maximum_pinned_bytes_per_slot=1 << 20,
        parity_absolute_tolerance=0.125,
        parity_relative_tolerance=0.02,
        parity_maximum_relative_rmse=0.02,
        parity_minimum_cosine_similarity=0.999,
    )
    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=policy,
        gpu_worker=lambda _item, _replica, payload: payload,
        cpu_parity_worker=lambda _item, payload: payload,
        device_receipt_probe=lambda index: replicas[index].device,
        installed_state_probe=lambda replica: replica.installed_state,
        cpu_installed_state_probe=lambda: installed,
    )
    parity = {
        "file_sha256": _sha("v5-parity-file"),
        "receipt_sha256": _sha("v5-parity-receipt"),
        "source_sha256": {
            "dynamic_replay_scheduler.py": _sha("scheduler-source"),
            "hf_lfm2_cpcf_candidate_replay.py": _sha("candidate-source"),
        },
        "real_two_gpu_parity_passed": True,
    }
    body = {
        "schema": LFM2_CPCF_V5_GPU_REPLAY_SCHEMA,
        "logical_device_indices": [0, 1],
        "devices": [replica.device.as_dict() for replica in replicas],
        "replicas": [replica.as_dict() for replica in replicas],
        "cpu_installed_state": installed.as_dict(),
        "scheduler_policy": policy.as_dict(),
        "proven_parity_receipt": parity,
        "contract_tensor_names": ["candidate_replay_contract"],
        "contract_cpu_tensors_retained_for_runtime_reauthentication": True,
        "contract_gpu_tensors_retained_for_runtime_reauthentication": True,
        "homogeneous_authenticated_receipts": True,
        "candidate_payload_staging": (
            "bounded_double_buffered_pinned_system_memory_to_cuda"
        ),
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return (
        AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator(scheduler),
        {**body, "report_sha256": canonical_sha256(body)},
    )


def _runtime_policy(
    model_policy: Lfm2CpcfModelFoldPolicy | None = None,
    **overrides: Any,
) -> Lfm2CpcfV5RuntimePolicy:
    values = {
        "model_fold_policy": (
            model_policy
            or Lfm2CpcfModelFoldPolicy(
                minimum_predictive_energy_fraction=0.0,
                promotion_compression_ratio=100.0,
                require_promotion_ratio=True,
            )
        ),
        "maximum_resident_evidence_bytes": 64,
        "full_phase_tree_logical_bytes": 128,
        "minimum_physical_rss_reduction_fraction": 0.10,
        "rss_sample_interval_seconds": 1.0,
        "require_short_conv_v5_endpoint_first_selector": True,
        "require_short_conv_v5_real_numerical_frontier_adapter": True,
    }
    values.update(overrides)
    return Lfm2CpcfV5RuntimePolicy(**values)


def _cache(tmp_path: Path, *, cache_key: str, parent: str) -> Any:
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        Lfm2CircuitTraceCache,
    )

    body = {
        "cache_key": cache_key,
        "parent": {"identity_sha256": parent},
        "corpus": {
            "corpus_sha256": _sha("v5-corpus"),
            "batch_count": 1,
        },
    }
    return Lfm2CircuitTraceCache(
        path=tmp_path,
        _manifest=MappingProxyType({**body, "manifest_sha256": canonical_sha256(body)}),
        _completion=MappingProxyType({}),
    )


def _backend(
    tmp_path: Path,
    *,
    cache_key: str = "a" * 64,
    parent: str = "b" * 64,
    model_policy: Lfm2CpcfModelFoldPolicy | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> Lfm2CpcfV5SelectiveBackend:
    evaluator, gpu_report = _gpu_evaluator_and_report()
    return Lfm2CpcfV5SelectiveBackend(
        cache=_cache(tmp_path, cache_key=cache_key, parent=parent),
        runtime_policy=_runtime_policy(
            model_policy,
            **(runtime_overrides or {}),
        ),
        candidate_replay_evaluator=evaluator,
        candidate_selector=Lfm2CpcfV5EndpointFirstCandidateSelector(),
        numerical_frontier_adapter=(Lfm2CpcfShortConvV5NumericalFrontierAdapter()),
        gpu_backend_report=gpu_report,
        expected_selective_catalog_sha256=_sha("v5-catalog"),
        expected_selective_source_sha256=_sha("v5-source"),
        expected_selective_provider_sha256=_sha("v5-provider"),
        expected_layer_count=2,
        expected_phase_sha256=(
            _sha("v5-phase-prefill"),
            _sha("v5-phase-decode"),
        ),
    )


def test_v5_backend_rejects_fabricated_cpu_cuda_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="exact source-bound runtime",
    ):
        _backend(tmp_path)


def test_v5_runtime_pins_reject_joint_global_and_callback_reseal(
    monkeypatch: Any,
) -> None:
    def forged_cuda(*_args: Any, **_kwargs: Any) -> None:
        return None

    def forged_cpu(*_args: Any, **_kwargs: Any) -> None:
        return None

    class ForgedEvaluator:
        pass

    class ForgedScheduler:
        pass

    class ForgedSchedulerPolicy:
        pass

    class ForgedStateProbes:
        pass

    class ForgedFactory:
        pass

    class ForgedBuffer:
        pass

    replacements = {
        "AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator": (ForgedEvaluator),
        "DeterministicReplayScheduler": ForgedScheduler,
        "ReplaySchedulerPolicy": ForgedSchedulerPolicy,
        "AuthenticatedReplayStateProbes": ForgedStateProbes,
        "AuthenticatedPinnedHostBufferFactory": ForgedFactory,
        "PinnedHostDoubleBuffer": ForgedBuffer,
        "lfm2_cpcf_cuda_replay_worker": forged_cuda,
        "lfm2_cpcf_cpu_parity_worker": forged_cpu,
        "replay_lfm2_cpcf_candidate_work_payload": forged_cpu,
        "capture_cuda_replay_device_receipt": forged_cpu,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(cpcf_adapter_module, name, replacement)
    monkeypatch.setattr(
        cpcf_adapter_module._candidate_replay_module,
        "AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator",
        ForgedEvaluator,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._candidate_replay_module,
        "lfm2_cpcf_cuda_replay_worker",
        forged_cuda,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._candidate_replay_module,
        "lfm2_cpcf_cpu_parity_worker",
        forged_cpu,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._candidate_replay_module,
        "replay_lfm2_cpcf_candidate_work_payload",
        forged_cpu,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._dynamic_replay_scheduler_module,
        "DeterministicReplayScheduler",
        ForgedScheduler,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._dynamic_replay_scheduler_module,
        "ReplaySchedulerPolicy",
        ForgedSchedulerPolicy,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._dynamic_replay_scheduler_module,
        "AuthenticatedReplayStateProbes",
        ForgedStateProbes,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._dynamic_replay_scheduler_module,
        "AuthenticatedPinnedHostBufferFactory",
        ForgedFactory,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._dynamic_replay_scheduler_module,
        "PinnedHostDoubleBuffer",
        ForgedBuffer,
    )
    monkeypatch.setattr(
        cpcf_adapter_module._dynamic_replay_scheduler_module,
        "capture_cuda_replay_device_receipt",
        forged_cpu,
    )

    with pytest.raises(Lfm2CpcfCatalogError, match="changed after import"):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_pins_reject_class_method_drift(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        DeterministicReplayScheduler,
        "authenticate_runtime",
        lambda _self: {},
    )
    with pytest.raises(Lfm2CpcfCatalogError, match="changed after import"):
        _authenticate_v5_runtime_objects()


@pytest.mark.parametrize(
    ("owner", "method_name"),
    (
        (
            cpcf_adapter_module._AUTHENTIC_REPLAY_SCHEDULER_CLASS,
            "_validate_buffer",
        ),
        (
            cpcf_adapter_module._AUTHENTIC_REPLAY_SCHEDULER_CLASS,
            "_run_gpu_one",
        ),
        (
            cpcf_adapter_module._AUTHENTIC_REPLAY_SCHEDULER_CLASS,
            "_execute_shard",
        ),
        (
            cpcf_adapter_module._AUTHENTIC_PINNED_HOST_BUFFER_CLASS,
            "wait_ready",
        ),
        (
            cpcf_adapter_module._AUTHENTIC_PINNED_HOST_BUFFER_CLASS,
            "mark_consumed",
        ),
        (
            cpcf_adapter_module._AUTHENTIC_PINNED_HOST_BUFFER_CLASS,
            "close",
        ),
    ),
)
def test_v5_runtime_class_manifest_rejects_dispatched_method_drift(
    monkeypatch: Any,
    owner: type,
    method_name: str,
) -> None:
    monkeypatch.setattr(owner, method_name, lambda *_args, **_kwargs: None)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="class callable manifest changed after import",
    ):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_class_manifest_rejects_joint_scheduler_buffer_reseal(
    monkeypatch: Any,
) -> None:
    def forged(*_args: Any, **_kwargs: Any) -> None:
        return None

    scheduler = cpcf_adapter_module._AUTHENTIC_REPLAY_SCHEDULER_CLASS
    buffer = cpcf_adapter_module._AUTHENTIC_PINNED_HOST_BUFFER_CLASS
    for owner, names in (
        (
            scheduler,
            (
                "_validate_buffer",
                "_run_gpu_one",
                "_execute_shard",
            ),
        ),
        (
            buffer,
            (
                "wait_ready",
                "mark_consumed",
                "close",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forged)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="class callable manifest changed after import",
    ):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_class_manifest_rejects_added_callable_name(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        cpcf_adapter_module._AUTHENTIC_PINNED_HOST_BUFFER_CLASS,
        "forged_unreviewed_dispatch",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="callable name set changed after import",
    ):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_class_manifest_rejects_property_accessor_drift(
    monkeypatch: Any,
) -> None:
    probes = cpcf_adapter_module._AUTHENTIC_REPLAY_STATE_PROBES_CLASS
    monkeypatch.setattr(
        probes,
        "device_indices",
        property(lambda _self: (0, 1)),
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="class callable manifest changed after import",
    ):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_class_manifest_rejects_in_place_callable_default_drift(
    monkeypatch: Any,
) -> None:
    method = cpcf_adapter_module._AUTHENTIC_REPLAY_SCHEDULER_CLASS._run_gpu_one
    monkeypatch.setattr(method, "__kwdefaults__", {"staged": "forged"})
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="class callable manifest changed after import",
    ):
        _authenticate_v5_runtime_objects()


@pytest.mark.parametrize(
    ("module", "binding_name"),
    (
        (
            cpcf_adapter_module._dynamic_replay_scheduler_module,
            "compare_cpu_cuda_replay",
        ),
        (
            cpcf_adapter_module._dynamic_replay_scheduler_module,
            "build_replay_schedule_plan",
        ),
        (
            cpcf_adapter_module._dynamic_replay_scheduler_module,
            "fingerprint_replay_value",
        ),
        (
            cpcf_adapter_module._candidate_replay_module,
            "_same_device_tensor",
        ),
        (
            cpcf_adapter_module._candidate_replay_module,
            "_state_sha256",
        ),
        (
            short_conv_v4_module,
            "_tensor_sha256",
        ),
    ),
)
def test_v5_runtime_module_manifest_rejects_named_transitive_helper_replacement(
    monkeypatch: Any,
    module: Any,
    binding_name: str,
) -> None:
    monkeypatch.setattr(
        module,
        binding_name,
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="runtime .* changed after import",
    ):
        _authenticate_v5_runtime_objects()


@pytest.mark.parametrize(
    ("module", "binding_name"),
    cpcf_adapter_module._V5_RUNTIME_DISCOVERED_REACHABLE_HELPER_BINDINGS,
    ids=(
        f"{module.__name__.rsplit('.', maxsplit=1)[-1]}::{binding_name}"
        for module, binding_name in (
            cpcf_adapter_module._V5_RUNTIME_DISCOVERED_REACHABLE_HELPER_BINDINGS
        )
    ),
)
def test_v5_runtime_module_manifest_rejects_every_discovered_helper_binding(
    monkeypatch: Any,
    module: Any,
    binding_name: str,
) -> None:
    original = vars(module)[binding_name]
    replacement: Any
    if isinstance(original, type):

        class ForgedRuntimeClass:
            pass

        replacement = ForgedRuntimeClass
    else:

        def forged_runtime_callable(*_args: Any, **_kwargs: Any) -> None:
            return None

        replacement = forged_runtime_callable
    monkeypatch.setattr(module, binding_name, replacement)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="runtime .* changed after import",
    ):
        _authenticate_v5_runtime_objects()


@pytest.mark.parametrize(
    ("module", "binding_name", "replacement"),
    (
        (
            cpcf_adapter_module._dynamic_replay_scheduler_module,
            "DYNAMIC_REPLAY_PARITY_SCHEMA",
            "forged-parity-schema",
        ),
        (
            cpcf_adapter_module._candidate_replay_module,
            "_STATE_ROLES",
            ("forged_weight",),
        ),
    ),
)
def test_v5_runtime_module_manifest_rejects_contract_constant_replacement(
    monkeypatch: Any,
    module: Any,
    binding_name: str,
    replacement: Any,
) -> None:
    monkeypatch.setattr(module, binding_name, replacement)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="runtime .* changed after import",
    ):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_module_manifest_rejects_added_namespace_callable(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        cpcf_adapter_module._candidate_replay_module,
        "forged_reachable_helper",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="namespace name set changed after import",
    ):
        _authenticate_v5_runtime_objects()


def test_v5_runtime_pins_reject_source_byte_drift(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        cpcf_adapter_module,
        "_file_sha256",
        lambda _path: "0" * 64,
    )
    with pytest.raises(Lfm2CpcfCatalogError, match="changed after import"):
        _authenticate_v5_runtime_objects()


def test_v5_backend_rejects_cpu_backend_policy_relaxation_and_gpu_drift(
    tmp_path: Path,
) -> None:
    evaluator, gpu_report = _gpu_evaluator_and_report()
    common = {
        "cache": _cache(
            tmp_path,
            cache_key="a" * 64,
            parent="b" * 64,
        ),
        "runtime_policy": _runtime_policy(),
        "candidate_selector": Lfm2CpcfV5EndpointFirstCandidateSelector(),
        "numerical_frontier_adapter": (Lfm2CpcfShortConvV5NumericalFrontierAdapter()),
        "gpu_backend_report": gpu_report,
        "expected_selective_catalog_sha256": _sha("v5-catalog"),
        "expected_selective_source_sha256": _sha("v5-source"),
        "expected_selective_provider_sha256": _sha("v5-provider"),
        "expected_layer_count": 2,
        "expected_phase_sha256": (
            _sha("v5-phase-prefill"),
            _sha("v5-phase-decode"),
        ),
    }
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="exact authenticated two-GPU",
    ):
        Lfm2CpcfV5SelectiveBackend(
            **common,
            candidate_replay_evaluator=(CpuReferenceLfm2CpcfCandidateReplayEvaluator()),
        )

    with pytest.raises(Lfm2CpcfCatalogError, match="endpoint-first"):
        Lfm2CpcfV5SelectiveBackend(
            **{
                **common,
                "runtime_policy": _runtime_policy(
                    require_short_conv_v5_endpoint_first_selector=False
                ),
            },
            candidate_replay_evaluator=evaluator,
        )

    tampered = copy.deepcopy(gpu_report)
    tampered["candidate_payload_staging"] = "unbounded"
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="exact source-bound runtime",
    ):
        Lfm2CpcfV5SelectiveBackend(
            **{**common, "gpu_backend_report": tampered},
            candidate_replay_evaluator=evaluator,
        )


def _sealed(body: dict[str, Any], seal_name: str) -> dict[str, Any]:
    return {**body, seal_name: canonical_sha256(body)}


def _canonical_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _write_current_source_parity_fixture(
    tmp_path: Path,
    *,
    mutate: Any | None = None,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    source = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "3933eef-two-rtx3090-bf16.json"
    )
    document = json.loads(source.read_text(encoding="ascii"))
    document["source_sha256"] = dict(_v5_gpu_parity_source_sha256())
    if mutate is not None:
        mutate(document)
    body = dict(document)
    body.pop("receipt_sha256", None)
    document["receipt_sha256"] = canonical_sha256(body)
    path = (tmp_path / "parity.json").resolve()
    payload = _canonical_document_bytes(document)
    path.write_bytes(payload)
    policy = document["backend_report"]["execution_report"]["plan"]["policy"]
    reference = {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": document["receipt_sha256"],
        "source_sha256": dict(document["source_sha256"]),
        "scheduler_policy": copy.deepcopy(policy),
        "target_dtype": "bfloat16",
        "real_two_gpu_parity_passed": True,
    }
    authority = {
        "approved_repository_root": path.parent,
        "approved_repository_relative_path": path.name,
        "approved_file_sha256": reference["file_sha256"],
        "approved_receipt_sha256": reference["receipt_sha256"],
    }
    return path, reference, copy.deepcopy(policy), authority


def test_v5_parity_file_authenticator_loads_actual_canonical_file(
    tmp_path: Path,
) -> None:
    _path, reference, policy, authority = _write_current_source_parity_fixture(tmp_path)
    authenticated = _test_only_authenticate_v5_gpu_parity_receipt_file(
        reference,
        expected_scheduler_policy=policy,
        **authority,
    )
    assert dict(authenticated) == reference


def test_public_v5_parity_loader_rejects_caller_resealed_receipt(
    tmp_path: Path,
) -> None:
    _path, reference, policy, _authority = _write_current_source_parity_fixture(
        tmp_path
    )
    with pytest.raises(Lfm2CpcfCatalogError, match="currently approved"):
        _authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
        )
    assert set(
        inspect.signature(_authenticate_v5_gpu_parity_receipt_file).parameters
    ) == {"value", "expected_scheduler_policy"}


def test_v5_parity_file_authenticator_rejects_tampered_file_hash(
    tmp_path: Path,
) -> None:
    _path, reference, policy, authority = _write_current_source_parity_fixture(tmp_path)
    reference["file_sha256"] = "0" * 64
    with pytest.raises(Lfm2CpcfCatalogError, match="exact approved"):
        _test_only_authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
            **authority,
        )


def test_v5_parity_file_authenticator_rejects_unsealed_body(
    tmp_path: Path,
) -> None:
    path, reference, policy, authority = _write_current_source_parity_fixture(tmp_path)
    document = json.loads(path.read_text(encoding="ascii"))
    document["status"] = "forged"
    payload = _canonical_document_bytes(document)
    path.write_bytes(payload)
    reference["file_sha256"] = hashlib.sha256(payload).hexdigest()
    authority["approved_file_sha256"] = reference["file_sha256"]
    with pytest.raises(Lfm2CpcfCatalogError, match="schema, seal"):
        _test_only_authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
            **authority,
        )


def test_v5_parity_file_authenticator_rejects_scheduler_policy_drift(
    tmp_path: Path,
) -> None:
    _path, reference, policy, authority = _write_current_source_parity_fixture(tmp_path)
    policy["maximum_pinned_bytes_per_slot"] += 1
    with pytest.raises(Lfm2CpcfCatalogError, match="scheduler, source"):
        _test_only_authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
            **authority,
        )


def test_v5_parity_file_authenticator_rejects_noncanonical_path(
    tmp_path: Path,
) -> None:
    path, reference, policy, authority = _write_current_source_parity_fixture(tmp_path)
    reference["path"] = os.path.relpath(path, Path.cwd())
    with pytest.raises(Lfm2CpcfCatalogError, match="not canonical"):
        _test_only_authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
            **authority,
        )


@pytest.mark.parametrize(
    "source_attack",
    ("altered", "missing-verifier"),
)
def test_v5_parity_file_authenticator_rejects_source_attack(
    tmp_path: Path,
    source_attack: str,
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        if source_attack == "altered":
            document["source_sha256"]["dynamic_replay_scheduler.py"] = "0" * 64
        else:
            document["source_sha256"].pop(
                "verify_lfm2_cpcf_candidate_replay_gpu_parity.py"
            )

    _path, reference, policy, authority = _write_current_source_parity_fixture(
        tmp_path,
        mutate=mutate,
    )
    with pytest.raises(Lfm2CpcfCatalogError, match="scheduler, source"):
        _test_only_authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
            **authority,
        )


def test_v5_parity_file_authenticator_rejects_relaxed_metric_tolerance(
    tmp_path: Path,
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["maximum_allowed_metric_delta"] = 0.06

    _path, reference, policy, authority = _write_current_source_parity_fixture(
        tmp_path,
        mutate=mutate,
    )
    with pytest.raises(Lfm2CpcfCatalogError, match="zero-operation contract"):
        _test_only_authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
            **authority,
        )


def test_checked_in_parity_receipt_remains_unapproved_and_fail_closed() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "3933eef-two-rtx3090-bf16.json"
    ).resolve()
    document = json.loads(path.read_text(encoding="ascii"))
    policy = document["backend_report"]["execution_report"]["plan"]["policy"]
    reference = {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "receipt_sha256": document["receipt_sha256"],
        "source_sha256": dict(document["source_sha256"]),
        "scheduler_policy": copy.deepcopy(policy),
        "target_dtype": "bfloat16",
        "real_two_gpu_parity_passed": True,
    }
    with pytest.raises(Lfm2CpcfCatalogError, match="currently approved"):
        _authenticate_v5_gpu_parity_receipt_file(
            reference,
            expected_scheduler_policy=policy,
        )


def _fake_v5_compiler_report(
    backend: Lfm2CpcfV5SelectiveBackend,
) -> dict[str, Any]:
    residency = _sealed(
        {
            "schema": "neural-forge-dynamic-evidence-residency-window-v1",
            "catalog_sha256": backend.expected_selective_catalog_sha256,
            "source_sha256": backend.expected_selective_source_sha256,
            "provider_sha256": backend.expected_selective_provider_sha256,
            "state": "active",
            "start_layer_index": 0,
            "current_layer_index": None,
            "occupied_layer_indices": [],
            "max_resident_layers": 2,
            "maximum_resident_bytes": (
                backend.runtime_policy.maximum_resident_evidence_bytes
            ),
            "resident_bytes": 0,
            "prefetch_distance_layers": 1,
            "peak_occupied_layers": 2,
            "peak_resident_bytes": 64,
            "requested_layer_indices": [0, 1],
            "promoted_layer_indices": [0, 1],
            "released_layer_indices": [0, 1],
            "canonical_layer_order_required": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "compiler_fingerprint_binding": (
                "integration_required_not_asserted_by_residency_provider"
            ),
            "resume_schema_binding": (
                "integration_required_not_asserted_by_residency_provider"
            ),
            "standalone_orchestration_only": True,
        },
        "snapshot_sha256",
    )
    physical = _sealed(
        {
            "schema": LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
            "start_rss_bytes": 1_000,
            "current_rss_bytes": 1_010,
            "peak_rss_bytes": 1_010,
            "start_lifetime_peak_bytes": 1_000,
            "current_increase_bytes": 10,
            "peak_increase_bytes": 10,
            "full_phase_tree_logical_bytes": (
                backend.runtime_policy.full_phase_tree_logical_bytes
            ),
            "minimum_reduction_fraction": (
                backend.runtime_policy.minimum_physical_rss_reduction_fraction
            ),
            "maximum_increase_bytes": (
                backend.runtime_policy.maximum_physical_rss_increase_bytes
            ),
            "sample_interval_seconds": (
                backend.runtime_policy.rss_sample_interval_seconds
            ),
            "checkpoint_count": 1,
            "checkpoints": [],
            "sampler_stopped": True,
            "physical_rss_gate_passed": True,
            "comparison_basis": (
                "peak_process_rss_increase_from_precompiler_baseline_"
                "versus_full_phase_tree_logical_bytes"
            ),
        },
        "report_sha256",
    )
    body = {
        "schema": LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
        "artifact_schema": "v5-artifact",
        "resume_schema": "v5-resume",
        "resume_supported": True,
        "resume_fail_closed": True,
        "resume_used": False,
        "resume_start_layer_index": 0,
        "resume_completed_prefix_layer_indices": [],
        "resume_executed_layer_indices": [0, 1],
        "resume_completed_layers_replayed": False,
        "resume_actual_upstream_child_state_restored": False,
        "resume_prefix_install_report_sha256": None,
        "resume_frontier_sha256": None,
        "resume_admission_sha256": None,
        "resume_publication_enabled": False,
        "resume_admitted_prefixes": [],
        "resume_parent_or_cache_tensor_sidecars": 0,
        "resume_checkpoint_receipt_chain_authenticated": True,
        "compiler_implementation": dict(lfm2_cpcf_v5_compiler_implementation()),
        "policy": backend.runtime_policy.as_dict(),
        "phase_count": len(backend.expected_phase_sha256),
        "phase_sha256": list(backend.expected_phase_sha256),
        "selective_catalog_sha256": (backend.expected_selective_catalog_sha256),
        "selective_source_sha256": (backend.expected_selective_source_sha256),
        "selective_provider_sha256": (backend.expected_selective_provider_sha256),
        "authenticated_full_phase_tree_logical_bytes": (
            backend.runtime_policy.full_phase_tree_logical_bytes
        ),
        "residency_receipt": residency,
        "physical_rss_receipt": physical,
        "ordered_layer_result_sha256": [
            _sha("v5-layer-0"),
            _sha("v5-layer-1"),
        ],
        "source_state_preserved": True,
        "target_parameter_count_preserved": True,
        "all_layers_compiled_in_order": True,
        "actual_folded_upstream_replay_used": True,
        "full_all_layer_phase_tree_materialized": False,
        "tensor_free_phase_shells_used": True,
        "active_layer_evidence_released_before_successor_compile": True,
        "final_norm_evidence_separately_bounded_and_reauthenticated": True,
        "candidate_replay_backend_injected": True,
        "preinstall_candidate_selector_provided": True,
        "preinstall_candidate_selector_active": True,
        "preinstall_candidate_selector_layer_indices": [0],
        "short_conv_v5_endpoint_first_selector_integrated": True,
        "short_conv_numerical_frontier_adapter_provided": True,
        "short_conv_numerical_frontier_adapter_active": True,
        "short_conv_numerical_frontier_adapter_layer_indices": [0],
        "short_conv_layer_indices": [0],
        "short_conv_v5_real_numerical_adapter_required": True,
        "short_conv_v5_real_numerical_adapter_concrete_instance_provided": (True),
        "short_conv_v5_real_numerical_provenance_layer_indices": [0],
        "short_conv_v5_real_numerical_layer_proofs": [{}],
        "short_conv_v5_numerical_candidate_generation_integrated": True,
        "short_conv_v5_numerical_policy_integrated": True,
        "short_conv_v5_post_install_rescoring_performed": False,
        "legacy_v4_default_or_fingerprint_changed": False,
        "operation_audit": {
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "generation_calls": 0,
            "parent_cache_generation_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "fallback_paths": 0,
        },
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def test_v5_compiler_dispatch_rejects_module_global_monkeypatch(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_v5_selective_model_fold",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="compiler callable changed",
    ):
        _authenticate_v5_compiler_callable()


def test_v5_compiler_dispatch_rejects_source_drift(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter._file_sha256",
        lambda _path: "0" * 64,
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="compiler callable changed",
    ):
        _authenticate_v5_compiler_callable()


def test_v5_compiler_and_runtime_reject_resume_coordinator_admit_replacement(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        cpcf_adapter_module._v5_resume_module.Lfm2CpcfV5ResumeCoordinator,
        "admit",
        lambda *_args, **_kwargs: SimpleNamespace(
            admission_sha256="f" * 64,
        ),
    )
    for authenticator in (
        _authenticate_v5_compiler_callable,
        _authenticate_v5_runtime_objects,
    ):
        with pytest.raises(
            Lfm2CpcfCatalogError,
            match="changed after import",
        ):
            authenticator()


def test_v5_sidecar_audit_traverses_bound_methods_closures_and_partials() -> None:
    class ResumeResource:
        def reopen(self) -> None:
            return None

    resource = ResumeResource()

    def closure() -> Any:
        captured_resource = resource

        def reopen() -> Any:
            return captured_resource

        return reopen

    def invoke(_resource: Any) -> None:
        return None

    cases = (
        ("bound", resource.reopen, ".__self__"),
        ("closure", closure(), ".__closure__"),
        ("partial", partial(invoke, resource), ".__partial_args__"),
    )
    for name, hook, expected_path in cases:
        model = torch.nn.Module()
        setattr(model, f"{name}_resume_hook", hook)
        paths = cpcf_adapter_module._model_resume_sidecar_paths(
            model,
            forbidden_objects={"resume_coordinator": resource},
        )
        assert len(paths) == 1
        assert expected_path in paths[0]
        assert paths[0].endswith("->resume_coordinator")


def test_v5_fresh_loader_rejects_sidecar_attached_after_deserialization() -> None:
    class ResumeResource:
        def reopen(self) -> None:
            return None

    resource = ResumeResource()

    def loader(_source: Path) -> Any:
        model = torch.nn.Module()
        model.resume_hook = resource.reopen
        return model

    audited_loader = cpcf_adapter_module._V5StandaloneModelLoader(
        loader=loader,
        forbidden_objects={"resume_coordinator": resource},
        v5_resource_fingerprint=_sha("v5-resource"),
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="Fresh V5 loader attached",
    ):
        audited_loader(Path("standalone-child"))


def test_v5_dynamic_authenticator_rejects_self_resealed_blank_prefix_references(
) -> None:
    backend = SimpleNamespace(
        runtime_policy=_runtime_policy(),
        expected_selective_catalog_sha256=_sha("fake-v5-catalog"),
        expected_selective_source_sha256=_sha("fake-v5-source"),
        expected_selective_provider_sha256=_sha("fake-v5-provider"),
        expected_phase_sha256=(
            _sha("fake-v5-phase-0"),
            _sha("fake-v5-phase-1"),
        ),
    )
    report = _fake_v5_compiler_report(backend)
    report["resume_publication_enabled"] = True
    report["resume_admitted_prefixes"] = [
        {
            "next_layer_index": index,
            "frontier_reference": {},
            "checkpoint_reference": {},
                "frontier_sha256": _sha(f"frontier-{index}"),
                "admission_sha256": _sha(f"admission-{index}"),
                "admission_chain_proof": {},
                "admission_chain_proof_sha256": _sha(f"proof-{index}"),
            }
        for index in (0, 1, 2)
    ]
    coordinator = object.__new__(
        cpcf_adapter_module.Lfm2CpcfV5ResumeCoordinator
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="invalid typed artifact references",
    ):
        cpcf_adapter_module._authenticate_v5_model_fold_report(
            report,
            backend=None,
            resume_coordinator=coordinator,
        )


def test_v5_deep_result_authentication_rejects_sidecar_mapping() -> None:
    backend = SimpleNamespace(
        expected_layer_count=2,
        numerical_frontier_adapter=(Lfm2CpcfShortConvV5NumericalFrontierAdapter()),
    )
    forged = SimpleNamespace(
        layer_results=(
            SimpleNamespace(layer_index=0),
            SimpleNamespace(layer_index=1),
        ),
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="exact typed model-fold result",
    ):
        _authenticate_v5_compiler_result(
            forged,
            {"ordered_layer_result_sha256": []},
            backend,
        )


def test_admission_reads_compiler_and_catalog_reports_separately(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    case = _operator_case(tmp_path, "admission-report-routing")
    _install_fake_compile(monkeypatch, case.parent, case.target)
    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )
    decision = Lfm2CpcfAdmissionEvaluator().evaluate(
        FoldEvaluationContext(
            action=case.action,
            execution_signature=case.action.signature,
            execution_context=case.context,
            evidence=case.evidence,
            transaction=transaction,
            incumbent_state_fingerprint=(transaction.incumbent_state_fingerprint),
            candidate_state_fingerprint=(transaction.candidate_state_fingerprint),
        )
    )
    assert decision.accepted is True
    metrics = dict(decision.metrics)
    assert metrics["operation_audit_zero"] is True
    assert metrics["parameter_receipt_valid"] is True
    assert metrics["parameter_path_write_receipt_valid"] is True
    transaction.rollback()


def test_admission_rejects_resealed_unknown_backend_label(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    case = _operator_case(tmp_path, "admission-unknown-backend")
    _install_fake_compile(monkeypatch, case.parent, case.target)
    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )
    forged = dict(transaction.report)
    forged.pop("report_sha256")
    forged["compiler_backend"] = "forged_v99"
    forged["report_sha256"] = canonical_sha256(forged)
    transaction._catalog_compile_report_canonical = cpcf_adapter_module._canonical_json(
        forged
    )
    decision = Lfm2CpcfAdmissionEvaluator().evaluate(
        FoldEvaluationContext(
            action=case.action,
            execution_signature=case.action.signature,
            execution_context=case.context,
            evidence=case.evidence,
            transaction=transaction,
            incumbent_state_fingerprint=(transaction.incumbent_state_fingerprint),
            candidate_state_fingerprint=(transaction.candidate_state_fingerprint),
        )
    )
    assert decision.accepted is False
    assert dict(decision.metrics)["operation_audit_zero"] is False
    transaction.rollback()


def _parent(vocab_size: int = 20_000) -> Any:
    config = transformers.Lfm2Config(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4201)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _v5_production_case(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "tokenizer-source").mkdir()
    fixture = _fixture(tmp_path / "tokenizer-source")
    bridge_directory = tmp_path / "bridge-tokenizer"
    shutil.copytree(fixture["tokenizer_directory"], bridge_directory)
    donor_tokenizer = _DonorTokenizer(fixture["tokenizer"])
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor_tokenizer,
        fixture["tokenizer"],
        bridge_directory,
        donor_row_capacity=20_000,
        target_row_capacity=65,
    )
    parent = _parent()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=4,
                hidden_size_multiple=4,
                minimum_head_dim=4,
                tokenizer_bridge_artifact=bridge,
            ),
        )
    )
    target_plan = plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=100,
            target_vocabulary_size=65,
            maximum_candidates=64,
        ),
        child_name="cpcf-v5-production-child",
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        fold_method_id=DENSE_BUDGET_MODEL_METHOD,
        catalogs=catalogs,
    )
    cache_key = _sha("v5-production-cache")
    trace, phases = _trace_and_phases(bridge, cache_key)
    model_policy = Lfm2CpcfModelFoldPolicy(
        minimum_predictive_energy_fraction=0.0,
        promotion_compression_ratio=100.0,
        require_promotion_ratio=True,
    )
    gate, evidence = _reviewed_gate_bound_to_bridge(
        tmp_path / "gate",
        bridge,
        target_plan=target_plan,
        parent_cache_key=cache_key,
        trace_adapter_sha256=trace.evidence_sha256,
        fold_policy=model_policy,
        phase_sha256=tuple(phase.phase_sha256 for phase in phases),
    )
    cache = _cache(
        tmp_path,
        cache_key=cache_key,
        parent=evidence.cache_live_parent_fingerprint,
    )
    evaluator, gpu_report = _gpu_evaluator_and_report()
    backend = Lfm2CpcfV5SelectiveBackend(
        cache=cache,
        runtime_policy=_runtime_policy(model_policy),
        candidate_replay_evaluator=evaluator,
        candidate_selector=Lfm2CpcfV5EndpointFirstCandidateSelector(),
        numerical_frontier_adapter=(Lfm2CpcfShortConvV5NumericalFrontierAdapter()),
        gpu_backend_report=gpu_report,
        expected_selective_catalog_sha256=_sha("v5-catalog"),
        expected_selective_source_sha256=_sha("v5-source"),
        expected_selective_provider_sha256=_sha("v5-provider"),
        expected_layer_count=2,
        expected_phase_sha256=tuple(phase.phase_sha256 for phase in phases),
    )
    reviewed_map = evidence.reviewed_evidence
    contract = Lfm2CpcfExternalBehaviorGateContract(
        independent_holdout_manifest_sha256=reviewed_map["manifest_sha256"],
        independent_parent_envelope_sha256=reviewed_map["parent_envelope_sha256"],
        parent_relative_manual_packet_sha256=reviewed_map[
            "parent_relative_manual_packet_sha256"
        ],
    )
    tied = (
        parent.get_input_embeddings()
        .weight.detach()
        .index_select(
            0,
            torch.tensor(trace.mapping, dtype=torch.long),
        )
        .clone()
    )
    resources = Lfm2CpcfV5SelectiveStackResources(
        backend=backend,
        trace_adapter=trace,
        parent_graph_fingerprint=parent_graph.fingerprint,
        source_tied_table=tied,
        external_behavior_contract=contract,
        expected_full_reviewed_evidence_fingerprint=(
            evidence.full_reviewed_evidence_fingerprint
        ),
        final_evaluator=gate,
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: None,
        fresh_reload_evaluator=gate,
        saver_id="v5-production-test-saver-v1",
        loader_id="v5-production-test-loader-v1",
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 4),
        evidence_fingerprint=cache_key,
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=bridge,
    )
    return {
        "parent": parent,
        "parent_tokenizer_fingerprint": (bridge.donor_tokenizer_semantics_sha256),
        "target_plan": target_plan,
        "target_adapter": target_adapter,
        "resources": resources,
        "source_parameters": parent_graph.unique_parameter_count,
    }


@pytest.mark.skip(
    reason=(
        "Requires a freshly authenticated two-CUDA runtime and matching "
        "real-v5 reviewed fold artifact; callback wiring alone is not proof."
    )
)
def test_v5_stack_builds_the_existing_generic_execution_and_gate_stack(
    tmp_path: Path,
) -> None:
    case = _v5_production_case(tmp_path)
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((case["target_adapter"],)),
    ).synthesize(
        request=case["target_plan"].target_request,
        parent_models={"parent": case["parent"]},
        parent_tokenizer_fingerprints={"parent": case["parent_tokenizer_fingerprint"]},
    )

    class _BuildHarness(Lfm2CpcfFoldStackAdapter):
        # The fixture reviewed gate intentionally belongs to its synthetic
        # replay model, not the newly materialized blank. Resource/gate drift
        # rejection is tested separately; this narrow harness exercises the
        # v5 build branch after the exact validator boundary.
        def _validate_target(self, value):
            selected = value.adapter_payload
            return (
                selected,
                _blank_report(value, selected),
                resolve_behavior_gate_binding(self.resources.final_evaluator),
            )

    adapter = _BuildHarness(case["resources"])

    stack = adapter.build(target)
    assert isinstance(
        stack.evidence_provider.evidence,
        Lfm2CpcfV5SelectiveSealedEvidence,
    )
    assert stack.final_evaluator is case["resources"].final_evaluator
    assert stack.fresh_reload_evaluator is case["resources"].fresh_reload_evaluator
    assert stack.model_saver is case["resources"].model_saver
    assert stack.model_loader is case["resources"].model_loader
    assert stack.evidence_fingerprint == case["resources"].backend.cache.cache_key


@pytest.mark.skip(
    reason=(
        "Requires a freshly authenticated two-CUDA runtime and matching "
        "real-v5 reviewed fold artifact."
    )
)
def test_v5_100x_bundle_is_exact_dynamic_and_scope_honest(
    tmp_path: Path,
) -> None:
    case = _v5_production_case(tmp_path)
    bundle = build_lfm2_cpcf_v5_100x_production_bundle(
        target_plan=case["target_plan"],
        target_adapter=case["target_adapter"],
        resources=case["resources"],
    )
    candidate = bundle.candidate_specs[0]
    metadata = dict(candidate.metadata)
    target_size = case["target_plan"].target_request.exact_unique_parameters

    assert case["source_parameters"] >= 100 * target_size
    assert bundle.capability.minimum_target_unique_parameters == target_size
    assert bundle.capability.maximum_target_unique_parameters == target_size
    assert bundle.maximum_attempts == 1
    assert metadata["registration_schema"] == V5_100X_SCHEMA
    assert metadata["automatic_target_synthesis"] is True
    assert metadata["automatic_selective_layer_iteration"] is True
    assert metadata["whole_output_gate_required"] is True
    assert metadata["fresh_reload_certification_required"] is True
    assert metadata["cross_family_proven"] is False
    assert metadata["multiple_parent_folding_proven"] is False
    assert metadata["modality_addition_proven"] is False
    assert metadata["quantization_used"] is False
    assert metadata["pruning_used"] is False
    assert metadata["training_used"] is False


@pytest.mark.skip(
    reason=(
        "Requires a freshly authenticated two-CUDA runtime and matching "
        "real-v5 reviewed fold artifact."
    )
)
def test_v5_100x_bundle_rejects_a_sub_100x_plan(tmp_path: Path) -> None:
    case = _v5_production_case(tmp_path)
    plan = case["target_plan"]
    # Simulate an authenticated-plan object whose nested source accounting
    # was altered after construction. The production preflight must reject
    # before it can claim the 100x lane.
    object.__setattr__(
        plan.frontier,
        "source_unique_parameters",
        99 * plan.target_request.exact_unique_parameters,
    )

    with pytest.raises(
        Lfm2CpcfProductionBundleError,
        match="at least 100 source parameters",
    ):
        build_lfm2_cpcf_v5_100x_production_bundle(
            target_plan=plan,
            target_adapter=case["target_adapter"],
            resources=case["resources"],
        )
