#!/usr/bin/env python3
"""Run the opt-in LFM2 CPCF v5 selective fold on one exact dense blank.

This entry point is deliberately narrower than the legacy production runner.
It reuses that runner's strict local input, tokenizer-bridge, trajectory, and
blank-target helpers, but routes the fold through the bounded-residency v5
compiler and the authenticated deterministic two-RTX-3090 replay backend.

The process opens an already sealed local trace cache.  It never regenerates
traces, executes the parent, trains, prunes, quantizes, downloads, resumes, or
publishes a child checkpoint.  Its only output is a new diagnostic JSON file.
Successful compilation therefore proves an in-memory structural fold, not
behavioral intelligence retention or a publishable model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# These must be set before any Transformers or CUDA-backed module is imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


RUN_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-blank-fold-failure-v1"
GPU_REPLAY_SCHEMA = "neural-forge-lfm2-cpcf-v5-authenticated-two-gpu-replay-v1"
PROVEN_GPU_PARITY_RECEIPT = (
    _REPOSITORY_ROOT / "reports" / "3933eef-two-rtx3090-bf16.json"
)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
GPU_LOGICAL_DEVICES = (0, 1)
GPU_PINNED_BYTES_PER_SLOT = 256 << 20
GPU_PARITY_ABSOLUTE_TOLERANCE = 0.125
GPU_PARITY_RELATIVE_TOLERANCE = 0.02
GPU_PARITY_MAXIMUM_RELATIVE_RMSE = 0.02
GPU_PARITY_MINIMUM_COSINE_SIMILARITY = 0.999
MINIMUM_PROMOTION_COMPRESSION_RATIO = 100.0
PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE = 0.0
PREREGISTERED_PER_LAG_COMMUTATOR_GATE = 1.0
_GIB = 1024**3
_FORBIDDEN_REQUEST_FIELDS = (
    "resume",
    "resume_state",
    "resume_store_directory",
    "resume_checkpoint_reference",
    "publish",
    "output_directory",
    "publication_directory",
    "save_child",
    "regenerate_traces",
    "train",
    "prune",
    "quantize",
    "download",
)


class Lfm2CpcfV5SelectiveBlankFoldRunnerError(RuntimeError):
    """The strict local v5 selective experiment failed closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Diagnostic metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _base_runner() -> Any:
    """Import the established strict-local helper module without importing torch."""

    from experiments import run_lfm2_cpcf_blank_fold

    return run_lfm2_cpcf_blank_fold


def _build_preregistered_real_v5_numerical_adapter() -> Any:
    """Construct the explicit source-independent v5 measurement policy."""

    from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
        Lfm2CpcfV5FirBasisPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
        Lfm2CpcfShortConvV5NumericalFrontierAdapter,
    )

    return Lfm2CpcfShortConvV5NumericalFrontierAdapter(
        basis_policy=Lfm2CpcfV5FirBasisPolicy(
            maximum_projected_fir_offdiagonal_relative_frobenius=(
                PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE
            ),
            maximum_per_lag_commutator_relative_frobenius=(
                PREREGISTERED_PER_LAG_COMMUTATOR_GATE
            ),
        )
    )


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{name} must be finite."
        ) from exc
    if not math.isfinite(result):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{name} must be finite.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{name} must be a positive integer."
        )
    return value


@dataclass(frozen=True, slots=True)
class V5SelectiveBlankFoldConfig:
    """One local-only, non-resumable, non-publishing v5 fold request."""

    parent_model_directory: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    trace_cache_directory: Path
    trajectory_corpus: Path
    diagnostic_json: Path
    compression: Any
    maximum_process_rss_gib: float
    maximum_logical_resident_gib: float
    minimum_physical_rss_reduction_fraction: float
    model_device: str
    dtype: str
    parent_name: str
    child_name: str
    blank_seed: int
    minimum_hidden_size: int
    maximum_candidates: int
    selected_candidate_rank: int | None
    ridge: float
    boundary_state_reconstruction_weight: float
    boundary_predictive_energy_floor: float
    boundary_validation_retention_floor: float
    short_conv_v4_ridge_alphas: tuple[float, ...]
    short_conv_v4_variable_projection_passes: int
    short_conv_v4_writer_trust_radius: float
    short_conv_v4_fir_trust_radius: float
    short_conv_v4_bcx_trust_radius: float
    short_conv_v4_correction_rms_floor: float
    short_conv_v4_context_objective_weight: float
    forbidden_requests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "parent_model_directory",
            "donor_tokenizer_directory",
            "target_tokenizer_directory",
            "bridge_directory",
            "trace_cache_directory",
            "trajectory_corpus",
            "diagnostic_json",
        ):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be pathlib.Path.")
        if not hasattr(self.compression, "as_kwargs") or not callable(
            self.compression.as_kwargs
        ):
            raise TypeError("compression must be the shared CompressionRequest.")
        process_limit = _finite_float(
            self.maximum_process_rss_gib,
            "maximum_process_rss_gib",
        )
        if process_limit <= 0.0 or int(process_limit * _GIB) <= 0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "maximum_process_rss_gib must set a positive absolute bound."
            )
        logical_limit = _finite_float(
            self.maximum_logical_resident_gib,
            "maximum_logical_resident_gib",
        )
        if logical_limit < 0.0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "maximum_logical_resident_gib must be nonnegative."
            )
        reduction = _finite_float(
            self.minimum_physical_rss_reduction_fraction,
            "minimum_physical_rss_reduction_fraction",
        )
        if not 0.0 <= reduction <= 1.0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "minimum_physical_rss_reduction_fraction must be in [0, 1]."
            )
        object.__setattr__(self, "maximum_process_rss_gib", process_limit)
        object.__setattr__(
            self,
            "maximum_logical_resident_gib",
            logical_limit,
        )
        object.__setattr__(
            self,
            "minimum_physical_rss_reduction_fraction",
            reduction,
        )
        if self.model_device not in {"cpu", "cuda:0"}:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "model_device must be cpu or cuda:0; replay devices are fixed "
                "to logical CUDA 0 and 1."
            )
        if self.dtype != "bfloat16":
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "The sealed two-RTX-3090 replay policy requires bfloat16."
            )
        if not self.parent_name.strip() or not self.child_name.strip():
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "parent_name and child_name must be nonempty."
            )
        if isinstance(self.blank_seed, bool) or not isinstance(
            self.blank_seed,
            int,
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "blank_seed must be an integer."
            )
        for value, name in (
            (self.minimum_hidden_size, "minimum_hidden_size"),
            (self.maximum_candidates, "maximum_candidates"),
        ):
            _positive_int(value, name)
        if self.selected_candidate_rank is not None and (
            isinstance(self.selected_candidate_rank, bool)
            or not isinstance(self.selected_candidate_rank, int)
            or self.selected_candidate_rank < 0
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "selected_candidate_rank must be nonnegative."
            )
        for value, name in (
            (self.ridge, "ridge"),
            (
                self.boundary_state_reconstruction_weight,
                "boundary_state_reconstruction_weight",
            ),
            (
                self.boundary_predictive_energy_floor,
                "boundary_predictive_energy_floor",
            ),
            (
                self.boundary_validation_retention_floor,
                "boundary_validation_retention_floor",
            ),
        ):
            normalized = _finite_float(value, name)
            if normalized < 0.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"{name} must be nonnegative."
                )
        for value, name in (
            (
                self.boundary_predictive_energy_floor,
                "boundary_predictive_energy_floor",
            ),
            (
                self.boundary_validation_retention_floor,
                "boundary_validation_retention_floor",
            ),
        ):
            if float(value) > 1.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"{name} must be within [0, 1]."
                )
        alphas = tuple(
            _finite_float(value, "short_conv_v4_ridge_alphas")
            for value in self.short_conv_v4_ridge_alphas
        )
        if (
            not alphas
            or any(value < 0.0 for value in alphas)
            or tuple(sorted(set(alphas))) != alphas
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "short_conv_v4_ridge_alphas must be unique, nonnegative, "
                "and increasing."
            )
        object.__setattr__(self, "short_conv_v4_ridge_alphas", alphas)
        if self.short_conv_v4_variable_projection_passes not in {2, 3}:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "short_conv_v4_variable_projection_passes must be 2 or 3."
            )
        for value, name in (
            (
                self.short_conv_v4_writer_trust_radius,
                "short_conv_v4_writer_trust_radius",
            ),
            (
                self.short_conv_v4_fir_trust_radius,
                "short_conv_v4_fir_trust_radius",
            ),
            (
                self.short_conv_v4_bcx_trust_radius,
                "short_conv_v4_bcx_trust_radius",
            ),
            (
                self.short_conv_v4_correction_rms_floor,
                "short_conv_v4_correction_rms_floor",
            ),
        ):
            if _finite_float(value, name) <= 0.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"{name} must be positive."
                )
        context_weight = _finite_float(
            self.short_conv_v4_context_objective_weight,
            "short_conv_v4_context_objective_weight",
        )
        if not 0.0 <= context_weight <= 0.1:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "short_conv_v4_context_objective_weight must be in [0, 0.1]."
            )
        forbidden = tuple(sorted(set(self.forbidden_requests)))
        if forbidden:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "V5 selective folding forbids requested resume, publication, "
                "trace regeneration, training, pruning, quantization, or "
                f"download actions: {', '.join(forbidden)}."
            )
        object.__setattr__(self, "forbidden_requests", forbidden)

    @property
    def maximum_process_rss_bytes(self) -> int:
        return int(self.maximum_process_rss_gib * _GIB)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model_directory": str(self.parent_model_directory),
            "donor_tokenizer_directory": str(self.donor_tokenizer_directory),
            "target_tokenizer_directory": str(self.target_tokenizer_directory),
            "bridge_directory": str(self.bridge_directory),
            "trace_cache_directory": str(self.trace_cache_directory),
            "trajectory_corpus": str(self.trajectory_corpus),
            "diagnostic_json": str(self.diagnostic_json),
            "compression": self.compression.as_kwargs(),
            "maximum_process_rss_gib": self.maximum_process_rss_gib,
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "maximum_logical_resident_gib": (self.maximum_logical_resident_gib),
            "minimum_physical_rss_reduction_fraction": (
                self.minimum_physical_rss_reduction_fraction
            ),
            "model_device": self.model_device,
            "dtype": self.dtype,
            "parent_name": self.parent_name,
            "child_name": self.child_name,
            "blank_seed": self.blank_seed,
            "minimum_hidden_size": self.minimum_hidden_size,
            "maximum_candidates": self.maximum_candidates,
            "selected_candidate_rank": self.selected_candidate_rank,
            "ridge": self.ridge,
            "boundary_state_reconstruction_weight": (
                self.boundary_state_reconstruction_weight
            ),
            "boundary_predictive_energy_floor": (self.boundary_predictive_energy_floor),
            "boundary_validation_retention_floor": (
                self.boundary_validation_retention_floor
            ),
            "short_conv_v4_ridge_alphas": list(self.short_conv_v4_ridge_alphas),
            "short_conv_v4_variable_projection_passes": (
                self.short_conv_v4_variable_projection_passes
            ),
            "short_conv_v4_writer_trust_radius": (
                self.short_conv_v4_writer_trust_radius
            ),
            "short_conv_v4_fir_trust_radius": (self.short_conv_v4_fir_trust_radius),
            "short_conv_v4_bcx_trust_radius": (self.short_conv_v4_bcx_trust_radius),
            "short_conv_v4_correction_rms_floor": (
                self.short_conv_v4_correction_rms_floor
            ),
            "short_conv_v4_context_objective_weight": (
                self.short_conv_v4_context_objective_weight
            ),
            "resume_requested": False,
            "publication_requested": False,
        }


@dataclass(frozen=True, slots=True)
class _GpuReplayBundle:
    evaluator: Any
    report: Mapping[str, Any]


def _configure_torch() -> Any:
    """Pin the deterministic backend before CUDA is initialized."""

    import torch

    if torch.cuda.is_initialized():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "CUDA initialized before the deterministic replay contract."
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    return torch


def _expected_gpu_scheduler_policy() -> dict[str, Any]:
    return {
        "expected_device_count": 2,
        "expected_device_name": EXPECTED_GPU_NAME,
        "assignment": "canonical_round_robin_by_first_shard_key",
        "pinned_buffer_slots_per_device": 2,
        "maximum_pinned_bytes_per_slot": GPU_PINNED_BYTES_PER_SLOT,
        "parity_absolute_tolerance": GPU_PARITY_ABSOLUTE_TOLERANCE,
        "parity_relative_tolerance": GPU_PARITY_RELATIVE_TOLERANCE,
        "parity_maximum_relative_rmse": (GPU_PARITY_MAXIMUM_RELATIVE_RMSE),
        "parity_minimum_cosine_similarity": (GPU_PARITY_MINIMUM_COSINE_SIMILARITY),
        "require_deterministic_debug_mode": True,
        "cpu_cuda_parity_required_before_scheduled_work": True,
        "runtime_reauthenticated_after_parity_and_work": True,
        "installed_state_reauthenticated_after_parity_and_work": True,
        "folding_math_changed": False,
    }


def _load_proven_gpu_parity_receipt(
    path: Path = PROVEN_GPU_PARITY_RECEIPT,
) -> Mapping[str, Any]:
    """Authenticate the prior real two-3090 candidate-replay proof."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The proven local two-RTX-3090 parity receipt is unavailable."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The proven GPU parity receipt is not an object."
        )
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    execution = value.get("backend_report", {}).get(
        "execution_report",
        {},
    )
    policy = execution.get("plan", {}).get("policy")
    devices = value.get("devices")
    if (
        value.get("schema") != "neural-forge-lfm2-cpcf-candidate-gpu-parity-v1"
        or value.get("status") != "passed"
        or claimed != _sha(body)
        or value.get("target_dtype") != "bfloat16"
        or value.get("candidate_state_sha256_equal") is not True
        or value.get("backend_report", {}).get("representative_cpu_cuda_parity_passed")
        is not True
        or execution.get("backend_receipts_reauthenticated") is not True
        or execution.get("installed_state_reauthenticated") is not True
        or policy != _expected_gpu_scheduler_policy()
        or not isinstance(devices, list)
        or [row.get("device_index") for row in devices] != [0, 1]
        or any(row.get("device_name") != EXPECTED_GPU_NAME for row in devices)
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The local two-RTX-3090 parity proof is incomplete or changed."
        )
    source_paths = {
        "dynamic_replay_scheduler.py": (
            _REPOSITORY_ROOT / "src" / "neural_forge" / "dynamic_replay_scheduler.py"
        ),
        "hf_lfm2_cpcf_candidate_replay.py": (
            _REPOSITORY_ROOT
            / "src"
            / "neural_forge"
            / "hf_lfm2_cpcf_candidate_replay.py"
        ),
        "verify_lfm2_cpcf_candidate_replay_gpu_parity.py": (
            _REPOSITORY_ROOT
            / "experiments"
            / "verify_lfm2_cpcf_candidate_replay_gpu_parity.py"
        ),
    }
    expected_sources = value.get("source_sha256")
    observed_sources = {
        name: _file_sha256(source) for name, source in source_paths.items()
    }
    if expected_sources != observed_sources:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "GPU replay source changed after the proven parity receipt."
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "receipt_sha256": claimed,
        "source_sha256": observed_sources,
        "scheduler_policy": dict(policy),
        "target_dtype": "bfloat16",
        "real_two_gpu_parity_passed": True,
    }


def _build_authenticated_gpu_replay_evaluator(torch: Any) -> _GpuReplayBundle:
    """Build the fixed logical-CUDA-0/1 scheduler used by the proven lane."""

    from neural_forge.dynamic_replay_scheduler import (
        DeterministicReplayScheduler,
        ReplayReplicaReceipt,
        ReplaySchedulerPolicy,
        capture_cuda_replay_device_receipt,
        fingerprint_installed_replay_state,
    )
    from neural_forge.hf_lfm2_cpcf_candidate_replay import (
        AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
        replay_lfm2_cpcf_candidate_work_payload,
    )

    parity_proof = dict(_load_proven_gpu_parity_receipt())
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 2:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Logical CUDA devices 0 and 1 must both be available."
        )
    contract_cpu = {
        "candidate_replay_contract": torch.arange(
            64,
            dtype=torch.float32,
        )
    }
    contract_gpu = {
        index: {
            name: value.to(device=f"cuda:{index}")
            for name, value in contract_cpu.items()
        }
        for index in GPU_LOGICAL_DEVICES
    }
    cpu_fingerprint = fingerprint_installed_replay_state(contract_cpu)
    device_receipts = {
        index: capture_cuda_replay_device_receipt(index)
        for index in GPU_LOGICAL_DEVICES
    }
    if (
        tuple(sorted(device_receipts)) != GPU_LOGICAL_DEVICES
        or any(
            receipt.device_index != index or receipt.device_name != EXPECTED_GPU_NAME
            for index, receipt in device_receipts.items()
        )
        or len(
            {receipt.hardware_profile_sha256 for receipt in device_receipts.values()}
        )
        != 1
        or len({receipt.runtime_profile_sha256 for receipt in device_receipts.values()})
        != 1
        or len({receipt.device_uuid for receipt in device_receipts.values()}) != 2
        or len(
            {
                (
                    receipt.pci_domain_id,
                    receipt.pci_bus_id,
                    receipt.pci_device_id,
                )
                for receipt in device_receipts.values()
            }
        )
        != 2
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Logical CUDA 0/1 are not two distinct homogeneous RTX 3090s."
        )
    gpu_fingerprints = {
        index: fingerprint_installed_replay_state(contract_gpu[index])
        for index in GPU_LOGICAL_DEVICES
    }
    if any(fingerprint != cpu_fingerprint for fingerprint in gpu_fingerprints.values()):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "CPU and CUDA replay-contract fingerprints differ."
        )
    replicas = tuple(
        ReplayReplicaReceipt.create(
            device=device_receipts[index],
            installed_state=gpu_fingerprints[index],
        )
        for index in GPU_LOGICAL_DEVICES
    )
    scheduler_policy = ReplaySchedulerPolicy(
        expected_device_count=2,
        expected_device_name=EXPECTED_GPU_NAME,
        pinned_buffer_slots_per_device=2,
        maximum_pinned_bytes_per_slot=GPU_PINNED_BYTES_PER_SLOT,
        parity_absolute_tolerance=GPU_PARITY_ABSOLUTE_TOLERANCE,
        parity_relative_tolerance=GPU_PARITY_RELATIVE_TOLERANCE,
        parity_maximum_relative_rmse=(GPU_PARITY_MAXIMUM_RELATIVE_RMSE),
        parity_minimum_cosine_similarity=(GPU_PARITY_MINIMUM_COSINE_SIMILARITY),
        require_deterministic_debug_mode=True,
    )
    if scheduler_policy.as_dict() != _expected_gpu_scheduler_policy():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Live GPU scheduler policy differs from the proven receipt."
        )
    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=scheduler_policy,
        gpu_worker=lambda _item, _replica, payload: (
            replay_lfm2_cpcf_candidate_work_payload(payload)
        ),
        cpu_parity_worker=lambda _item, payload: (
            replay_lfm2_cpcf_candidate_work_payload(payload)
        ),
        device_receipt_probe=capture_cuda_replay_device_receipt,
        installed_state_probe=lambda replica: fingerprint_installed_replay_state(
            contract_gpu[replica.device.device_index]
        ),
        cpu_installed_state_probe=lambda: fingerprint_installed_replay_state(
            contract_cpu
        ),
    )
    evaluator = AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator(scheduler)
    body = {
        "schema": GPU_REPLAY_SCHEMA,
        "logical_device_indices": list(GPU_LOGICAL_DEVICES),
        "devices": [device_receipts[index].as_dict() for index in GPU_LOGICAL_DEVICES],
        "replicas": [replica.as_dict() for replica in replicas],
        "cpu_installed_state": cpu_fingerprint.as_dict(),
        "scheduler_policy": scheduler_policy.as_dict(),
        "proven_parity_receipt": parity_proof,
        "contract_tensor_names": sorted(contract_cpu),
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
    return _GpuReplayBundle(
        evaluator=evaluator,
        report={**body, "report_sha256": _sha(body)},
    )


def _maximum_current_plus_next_bytes(references: Sequence[Any]) -> int:
    ordered = tuple(references)
    if not ordered:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog contains no layer references."
        )
    return max(
        int(reference.logical_bytes)
        + (0 if index + 1 == len(ordered) else int(ordered[index + 1].logical_bytes))
        for index, reference in enumerate(ordered)
    )


def _resolved_logical_resident_bytes(
    *,
    references: Sequence[Any],
    final_norm_logical_bytes: int,
    requested_gib: float,
) -> tuple[int, int]:
    required = max(
        _maximum_current_plus_next_bytes(references),
        _positive_int(
            int(final_norm_logical_bytes),
            "final_norm_logical_bytes",
        ),
    )
    resolved = required if requested_gib == 0.0 else int(requested_gib * _GIB)
    if resolved < required:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The logical evidence budget is below the largest authenticated "
            "current-plus-next layer pair or final-norm payload."
        )
    return resolved, required


def _derive_effective_rss_reduction_fraction(
    *,
    full_phase_tree_logical_bytes: int,
    compiler_baseline_rss_bytes: int,
    maximum_process_rss_bytes: int,
    requested_minimum_reduction_fraction: float,
) -> tuple[float, int]:
    """Translate the absolute process cap into the compiler's delta policy."""

    full = _positive_int(
        full_phase_tree_logical_bytes,
        "full_phase_tree_logical_bytes",
    )
    baseline = _positive_int(
        compiler_baseline_rss_bytes,
        "compiler_baseline_rss_bytes",
    )
    limit = _positive_int(
        maximum_process_rss_bytes,
        "maximum_process_rss_bytes",
    )
    requested = _finite_float(
        requested_minimum_reduction_fraction,
        "requested_minimum_reduction_fraction",
    )
    if not 0.0 <= requested <= 1.0:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "requested_minimum_reduction_fraction must be in [0, 1]."
        )
    if baseline > limit:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Compiler baseline RSS already exceeds --maximum-process-rss-gib."
        )
    absolute_headroom = limit - baseline
    reduction_headroom = int(math.floor(full * (1.0 - requested)))
    allowed_increase = min(full, absolute_headroom, reduction_headroom)
    effective = max(requested, 1.0 - (allowed_increase / full))
    # Float conversion must never relax the integer byte cap.
    while int(math.floor(full * (1.0 - effective))) > allowed_increase:
        adjusted = math.nextafter(effective, 1.0)
        if adjusted == effective:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Cannot encode the absolute RSS cap in the v5 runtime policy."
            )
        effective = adjusted
    return effective, allowed_increase


def _memory_sample(
    v5_compiler: Any,
    *,
    label: str,
    maximum_process_rss_bytes: int,
) -> dict[str, Any]:
    current = int(v5_compiler._process_rss_bytes())
    peak = int(v5_compiler._process_peak_rss_bytes())
    sample = {
        "label": label,
        "current_rss_bytes": current,
        "process_high_water_bytes": peak,
        "maximum_process_rss_bytes": maximum_process_rss_bytes,
        "passed": max(current, peak) <= maximum_process_rss_bytes,
    }
    if sample["passed"] is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"Process RSS exceeded the absolute cap at {label!r}: "
            f"{max(current, peak)} > {maximum_process_rss_bytes}."
        )
    return sample


def _walk_mappings(value: Any) -> Sequence[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, Mapping):
            result.append(current)
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)
    return result


def _gpu_execution_audit(
    layer_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    executions = [
        mapping
        for report in layer_reports
        for mapping in _walk_mappings(report)
        if mapping.get("schema") == "neural-forge-dynamic-replay-execution-v1"
    ]
    if any(
        execution.get("backend_receipts_reauthenticated") is not True
        or execution.get("installed_state_reauthenticated") is not True
        or execution.get("parity_passed_before_scheduled_work") is not True
        or {
            int(report["device_index"])
            for report in execution.get("parity_reports", ())
        }
        != set(GPU_LOGICAL_DEVICES)
        for execution in executions
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "A candidate replay execution lacks two-GPU parity or "
            "reauthentication evidence."
        )
    return {
        "candidate_gpu_replay_invoked": bool(executions),
        "execution_count": len(executions),
        "execution_report_sha256": [
            execution.get("report_sha256") for execution in executions
        ],
        "both_logical_devices_reauthenticated_per_execution": bool(executions),
        "no_gpu_work_claimed_when_no_candidate_batch_was_scheduled": True,
    }


def _run(config: V5SelectiveBlankFoldConfig) -> dict[str, Any]:
    base = _base_runner()
    parent_directory = base._local_directory(
        config.parent_model_directory,
        "parent_model_directory",
    )
    donor_tokenizer_directory = base._local_directory(
        config.donor_tokenizer_directory,
        "donor_tokenizer_directory",
    )
    target_tokenizer_directory = base._local_directory(
        config.target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    bridge_directory = base._local_directory(
        config.bridge_directory,
        "bridge_directory",
    )
    cache_directory = base._local_directory(
        config.trace_cache_directory,
        "trace_cache_directory",
    )
    trajectory_path = base._local_file(
        config.trajectory_corpus,
        "trajectory_corpus",
    )
    torch = _configure_torch()

    from experiments.run_lfm2_residual_attention_fold import (
        direct_local_io_audit,
        load_local_lfm2_model,
        load_local_tokenizer,
    )
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        fingerprint_lfm2_parent,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_blank_target import (
        plan_and_synthesize_lfm2_cpcf_blank,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldPolicy,
        audit_lfm2_cpcf_parameter_compaction,
        independent_parameter_inventory,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        build_lfm2_cpcf_selective_evidence_provider,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge.hf_tokenizer_bridge import (
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )

    memory_samples = [
        _memory_sample(
            v5_compiler,
            label="deterministic_torch_configured",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    ]
    gpu = _build_authenticated_gpu_replay_evaluator(torch)
    memory_samples.append(
        _memory_sample(
            v5_compiler,
            label="authenticated_two_gpu_replay_ready",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    )
    dtype = base._torch_dtype(torch, config.dtype)
    device = torch.device(config.model_device)
    if not torch.cuda.is_available():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "CUDA is required for authenticated candidate replay."
        )

    donor_tokenizer = load_local_tokenizer(donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    bridge.verify_files()
    donor_fingerprint = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
    target_fingerprint = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    if (
        donor_fingerprint != bridge.donor_tokenizer_semantics_sha256
        or target_fingerprint != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Live local tokenizers differ from the sealed token bridge."
        )
    rows = base._read_trajectory_rows(trajectory_path)
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    trace_adapter = base._trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=cache.manifest,
    )
    parent = load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent.requires_grad_(False)
    parent.eval()
    parent_fingerprint_before = fingerprint_lfm2_parent(parent)
    if parent_fingerprint_before.identity_sha256 != cache.parent_fingerprint:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The explicit local parent differs from the sealed trace cache."
        )
    parent_inventory = independent_parameter_inventory(parent)
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name=config.parent_name,
        parent_tokenizer_fingerprint=donor_fingerprint,
        evidence_fingerprint=cache.cache_key,
        child_name=config.child_name,
        target_vocabulary_size=(bridge.transport_profile.target_row_capacity),
        child_tokenizer_fingerprint=target_fingerprint,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=config.minimum_hidden_size,
        maximum_candidates=config.maximum_candidates,
        selected_candidate_rank=config.selected_candidate_rank,
        blank_seed=config.blank_seed,
        device=device,
        dtype=dtype,
        **config.compression.as_kwargs(),
    )
    target = blank.model
    target.requires_grad_(False)
    target.eval()
    blank_inventory = independent_parameter_inventory(target)
    if config.compression.target_unique_parameters is not None and (
        blank.unique_parameters != config.compression.target_unique_parameters
        or int(blank_inventory["unique_parameters"])
        != config.compression.target_unique_parameters
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The synthesized dense blank misses the exact parameter target."
        )
    structural_compression = audit_lfm2_cpcf_parameter_compaction(
        parent,
        target,
        planned_target_unique_parameters=blank.unique_parameters,
        promotion_compression_ratio=(MINIMUM_PROMOTION_COMPRESSION_RATIO),
    )
    if structural_compression.get("promotion_ratio_met") is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The synthesized blank is below the required 100x experiment "
            "compression floor."
        )
    memory_samples.append(
        _memory_sample(
            v5_compiler,
            label="local_parent_and_exact_dense_blank_ready",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    )

    provider = None
    try:
        provider = build_lfm2_cpcf_selective_evidence_provider(cache)
        catalog = provider.catalog
        references = catalog.dynamic_catalog.references
        logical_budget, required_logical_budget = _resolved_logical_resident_bytes(
            references=references,
            final_norm_logical_bytes=(catalog.final_norm_logical_bytes),
            requested_gib=config.maximum_logical_resident_gib,
        )
        full_phase_tree_bytes = int(catalog.final_norm_logical_bytes) + sum(
            int(reference.logical_bytes) for reference in references
        )
        memory_samples.append(
            _memory_sample(
                v5_compiler,
                label="selective_catalog_authenticated",
                maximum_process_rss_bytes=(config.maximum_process_rss_bytes),
            )
        )
        compiler_baseline = int(v5_compiler._process_rss_bytes())
        effective_reduction, maximum_compiler_increase = (
            _derive_effective_rss_reduction_fraction(
                full_phase_tree_logical_bytes=full_phase_tree_bytes,
                compiler_baseline_rss_bytes=compiler_baseline,
                maximum_process_rss_bytes=(config.maximum_process_rss_bytes),
                requested_minimum_reduction_fraction=(
                    config.minimum_physical_rss_reduction_fraction
                ),
            )
        )
        short_conv_policy = Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=config.short_conv_v4_ridge_alphas,
            variable_projection_passes=(
                config.short_conv_v4_variable_projection_passes
            ),
            writer_trust_radius=(config.short_conv_v4_writer_trust_radius),
            fir_trust_radius=config.short_conv_v4_fir_trust_radius,
            bcx_trust_radius=config.short_conv_v4_bcx_trust_radius,
            correction_rms_floor=(config.short_conv_v4_correction_rms_floor),
            context_objective_weight=(config.short_conv_v4_context_objective_weight),
            target_dtype=config.dtype,
        )
        model_policy = Lfm2CpcfModelFoldPolicy(
            ridge=config.ridge,
            state_reconstruction_weight=(config.boundary_state_reconstruction_weight),
            minimum_predictive_energy_fraction=(
                config.boundary_predictive_energy_floor
            ),
            minimum_validation_block_retention=(
                config.boundary_validation_retention_floor
            ),
            enforce_validation_block_floor=True,
            short_conv_v4_solver_policy=short_conv_policy,
            promotion_compression_ratio=(MINIMUM_PROMOTION_COMPRESSION_RATIO),
            require_promotion_ratio=True,
        )
        runtime_policy = v5_compiler.Lfm2CpcfV5RuntimePolicy(
            model_fold_policy=model_policy,
            maximum_resident_evidence_bytes=logical_budget,
            full_phase_tree_logical_bytes=full_phase_tree_bytes,
            minimum_physical_rss_reduction_fraction=(effective_reduction),
            rss_sample_interval_seconds=0.05,
            require_short_conv_v5_endpoint_first_selector=True,
            require_short_conv_v5_real_numerical_frontier_adapter=True,
        )
        if (
            runtime_policy.maximum_physical_rss_increase_bytes
            > maximum_compiler_increase
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "The effective v5 RSS policy relaxed the absolute process memory bound."
            )
        source_tied_table = base._mapped_source_tied_table(parent, bridge)
        selector = v5_compiler.Lfm2CpcfV5EndpointFirstCandidateSelector()
        numerical_adapter = _build_preregistered_real_v5_numerical_adapter()
        compiled = v5_compiler.compile_lfm2_cpcf_v5_selective_model_fold(
            parent,
            target,
            trace_adapter=trace_adapter,
            evidence_provider=provider,
            source_tied_table=source_tied_table,
            blank_target_report=blank.as_dict(),
            planned_target_unique_parameters=(blank.unique_parameters),
            runtime_policy=runtime_policy,
            candidate_replay_evaluator=gpu.evaluator,
            candidate_selector=selector,
            numerical_frontier_adapter=numerical_adapter,
            resume_state=None,
        )
    finally:
        if provider is not None:
            provider.close()
    memory_samples.append(
        _memory_sample(
            v5_compiler,
            label="v5_compiler_complete_provider_closed",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    )
    parent_fingerprint_after = fingerprint_lfm2_parent(parent)
    if (
        parent_fingerprint_after.identity_sha256
        != parent_fingerprint_before.identity_sha256
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Parent state changed during the selective fold."
        )
    compiled_inventory = independent_parameter_inventory(target)
    if int(compiled_inventory["unique_parameters"]) != int(
        blank_inventory["unique_parameters"]
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Folding changed the dense child parameter count."
        )
    compiled_report = compiled.as_dict()
    layer_reports = [result.as_dict() for result in compiled.layer_results]
    gpu_execution = _gpu_execution_audit(layer_reports)
    if (
        compiled_report.get("short_conv_v5_endpoint_first_selector_integrated")
        is not True
        or compiled_report.get(
            "short_conv_v5_numerical_candidate_generation_integrated"
        )
        is not True
        or compiled_report.get("short_conv_v5_numerical_policy_integrated")
        is not True
        or compiled_report.get(
            "short_conv_v5_real_numerical_adapter_required"
        )
        is not True
        or compiled_report.get("candidate_replay_backend_injected") is not True
        or compiled_report.get("resume_supported") is not False
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The completed fold lacks the required real v5 numerical "
            "provenance, selector, replay injection, or resume rejection "
            "receipt."
        )
    body = {
        "schema": RUN_SCHEMA,
        "status": "compiled_in_memory_diagnostic_not_published",
        "inputs": {
            "parent_model_directory": str(parent_directory),
            "donor_tokenizer_directory": str(donor_tokenizer_directory),
            "target_tokenizer_directory": str(target_tokenizer_directory),
            "bridge_directory": str(bridge_directory),
            "trace_cache_directory": str(cache_directory),
            "trajectory_corpus": str(trajectory_path),
            "trajectory_corpus_sha256": _file_sha256(trajectory_path),
            "cache_key": cache.cache_key,
            "parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "trace_adapter_sha256": trace_adapter.evidence_sha256,
        },
        "request": config.as_dict(),
        "local_io_audit": direct_local_io_audit(parent_directory),
        "blank_target": blank.as_dict(),
        "parent_parameter_inventory": dict(parent_inventory),
        "blank_parameter_inventory": dict(blank_inventory),
        "compiled_parameter_inventory": dict(compiled_inventory),
        "structural_compression": dict(structural_compression),
        "compiled_model_fold": compiled_report,
        "layer_results": layer_reports,
        "boundary_bundle": compiled.boundary_bundle.as_dict(),
        "interface_install": dict(compiled.interface_install_report),
        "neutral_norm_install": dict(compiled.neutral_norm_install_report),
        "final_norm": dict(compiled.final_norm_report),
        "selective_catalog": catalog.as_dict(),
        "selective_residency_policy": {
            "full_phase_tree_logical_bytes": full_phase_tree_bytes,
            "largest_required_current_plus_next_or_final_norm_bytes": (
                required_logical_budget
            ),
            "maximum_resident_evidence_bytes": logical_budget,
            "bounded_current_plus_next_system_memory": True,
            "full_phase_tree_materialized": False,
        },
        "absolute_process_rss_policy": {
            "maximum_process_rss_bytes": (config.maximum_process_rss_bytes),
            "compiler_baseline_rss_bytes": compiler_baseline,
            "maximum_compiler_rss_increase_bytes": (
                runtime_policy.maximum_physical_rss_increase_bytes
            ),
            "effective_minimum_reduction_fraction": (
                runtime_policy.minimum_physical_rss_reduction_fraction
            ),
            "samples": memory_samples,
            "absolute_cap_passed": True,
        },
        "gpu_candidate_replay": dict(gpu.report),
        "gpu_execution_audit": gpu_execution,
        "preregistered_short_conv_v5_measurement_policy": {
            "projected_fir_offdiagonal_gate": (
                PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE
            ),
            "per_lag_commutator_gate": (
                PREREGISTERED_PER_LAG_COMMUTATOR_GATE
            ),
            "basis_policy": (
                numerical_adapter.basis_policy.authentication_body()
            ),
            "basis_policy_sha256": (
                numerical_adapter.basis_policy.authentication_sha256
            ),
            "ridge_policy": (
                numerical_adapter.ridge_policy.authentication_body()
            ),
            "ridge_policy_sha256": (
                numerical_adapter.ridge_policy.authentication_sha256
            ),
            "source_independent_preregistered_before_fit": True,
            "real_v5_adapter_required": True,
        },
        "dynamic_system_contract": {
            "automatic_blank_target_registry_used": True,
            "automatic_parameter_frontier_used": True,
            "automatic_target_topology_selection_used": True,
            "automatic_selective_layer_iteration_used": True,
            "manual_layer_surgery_in_runner": False,
            "current_executable_lane": "lfm2_text_parent_to_lfm2_text_blank",
            "any_family_any_modality_proven_by_this_run": False,
            "architecture_and_modality_adapters_remain_registry_driven": True,
        },
        "operation_audit": {
            "local_parent_model_loads": 1,
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "parent_cache_generation_calls": 0,
            "parent_cache_regeneration_calls": 0,
            "sealed_cache_open_count": 1,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "network_access": False,
            "hub_loader_used": False,
        },
        "claims": {
            "folding_only": True,
            "fresh_exact_dense_zero_blank_used": True,
            "tokenizer_bridge_used": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "trace_regeneration_used": False,
            "download_used": False,
            "resume_used": False,
            "child_checkpoint_saved": False,
            "publication_attempted": False,
            "publication_claimed": False,
            "behavioral_intelligence_retention_proven": False,
            "manual_intelligence_review_performed": False,
        },
    }
    return {**body, "receipt_sha256": _sha(body)}


def run(config: V5SelectiveBlankFoldConfig) -> dict[str, Any]:
    """Execute one strict run while retaining pre-compiler failure context."""

    try:
        return _run(config)
    except Exception as exc:
        wrapped = Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The v5 selective blank fold failed closed."
        )
        wrapped.diagnostic_report = {
            "schema": FAILURE_SCHEMA,
            "request": config.as_dict(),
            "stage": "run",
            "publication_claimed": False,
            "resume_claimed": False,
            "failure_evidence_only": True,
        }
        raise wrapped from exc


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        return "<maximum diagnostic depth reached>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_safe(item, depth=depth + 1) for item in value]
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _exception_chain(error: BaseException) -> list[dict[str, Any]]:
    chain = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reports = {}
        for name in (
            "diagnostic_report",
            "failure_report",
            "report",
            "layer_failure_report",
            "diagnostics",
        ):
            try:
                value = getattr(current, name)
            except Exception:
                continue
            if value is not None:
                reports[name] = _json_safe(value)
        chain.append(
            {
                "type": (f"{type(current).__module__}.{type(current).__qualname__}"),
                "message": str(current),
                "notes": [str(note) for note in getattr(current, "__notes__", ())],
                "structured_reports": reports,
            }
        )
        current = (
            current.__cause__ if current.__cause__ is not None else current.__context__
        )
    return chain


def build_failure_diagnostic(
    error: BaseException,
    *,
    config: V5SelectiveBlankFoldConfig | None = None,
    parsed_args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    chain = _exception_chain(error)
    preserved = [
        {
            "exception_index": index,
            "name": name,
            "report": report,
        }
        for index, record in enumerate(chain)
        for name, report in record["structured_reports"].items()
    ]
    body = {
        "schema": FAILURE_SCHEMA,
        "status": "failed",
        "request": (
            config.as_dict()
            if config is not None
            else _json_safe(vars(parsed_args) if parsed_args is not None else {})
        ),
        "exception_chain": chain,
        "preserved_structured_failure_reports": preserved,
        "structured_failure_report_count": len(preserved),
        "failure_evidence_only": True,
        "compiled_child_success_receipt_issued": False,
        "child_checkpoint_saved": False,
        "publication_attempted": False,
        "publication_claimed": False,
        "resume_attempted": False,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "network_access": False,
    }
    return {**body, "report_sha256": _sha(body)}


def _write_diagnostic(path: Path, value: Mapping[str, Any]) -> None:
    # Reuse the established local runner's create-new atomic JSON contract.
    _base_runner()._atomic_json(path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument(
        "--donor-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument(
        "--trace-cache-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--trajectory-corpus", required=True, type=Path)
    parser.add_argument("--diagnostic-json", required=True, type=Path)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--compression-ratio")
    budget.add_argument("--target-unique-parameters", type=int)
    parser.add_argument(
        "--maximum-process-rss-gib",
        type=float,
        required=True,
        help="Hard absolute process high-water/RSS limit.",
    )
    parser.add_argument(
        "--maximum-logical-resident-gib",
        type=float,
        default=0.0,
        help=(
            "Bound for authenticated CPU evidence. Zero selects the exact "
            "largest current-plus-next layer pair or final-norm payload."
        ),
    )
    parser.add_argument(
        "--minimum-physical-rss-reduction-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--model-device",
        choices=("cpu", "cuda:0"),
        default="cuda:0",
    )
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--parent-name", default="lfm2-parent")
    parser.add_argument(
        "--child-name",
        default="lfm2-cpcf-v5-selective-folded-child",
    )
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--minimum-hidden-size", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument(
        "--boundary-state-reconstruction-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--boundary-predictive-energy-floor",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--boundary-validation-retention-floor",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--short-conv-v4-ridge-alpha",
        action="append",
        type=float,
        dest="short_conv_v4_ridge_alphas",
    )
    parser.add_argument(
        "--short-conv-v4-variable-projection-passes",
        choices=(2, 3),
        type=int,
        default=2,
    )
    parser.add_argument(
        "--short-conv-v4-writer-trust-radius",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--short-conv-v4-fir-trust-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--short-conv-v4-bcx-trust-radius",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--short-conv-v4-correction-rms-floor",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--short-conv-v4-context-objective-weight",
        type=float,
        default=0.05,
    )
    # Explicitly parse disallowed requests so they fail with a safety reason,
    # rather than being silently ignored or mistaken for supported features.
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume-state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume-store-directory",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume-checkpoint-reference",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--publish", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--publication-directory",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    for name in (
        "regenerate-traces",
        "train",
        "prune",
        "quantize",
        "download",
    ):
        parser.add_argument(
            f"--{name}",
            action="store_true",
            help=argparse.SUPPRESS,
        )
    return parser


def config_from_args(args: argparse.Namespace) -> V5SelectiveBlankFoldConfig:
    base = _base_runner()
    forbidden = tuple(
        name
        for name in _FORBIDDEN_REQUEST_FIELDS
        if getattr(args, name, None) not in (None, False)
    )
    return V5SelectiveBlankFoldConfig(
        parent_model_directory=args.parent_model_directory,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        bridge_directory=args.bridge_directory,
        trace_cache_directory=args.trace_cache_directory,
        trajectory_corpus=args.trajectory_corpus,
        diagnostic_json=args.diagnostic_json,
        compression=base.CompressionRequest(
            compression_ratio=args.compression_ratio,
            target_unique_parameters=args.target_unique_parameters,
        ),
        maximum_process_rss_gib=args.maximum_process_rss_gib,
        maximum_logical_resident_gib=(args.maximum_logical_resident_gib),
        minimum_physical_rss_reduction_fraction=(
            args.minimum_physical_rss_reduction_fraction
        ),
        model_device=args.model_device,
        dtype=args.dtype,
        parent_name=args.parent_name,
        child_name=args.child_name,
        blank_seed=args.blank_seed,
        minimum_hidden_size=args.minimum_hidden_size,
        maximum_candidates=args.maximum_candidates,
        selected_candidate_rank=args.selected_candidate_rank,
        ridge=args.ridge,
        boundary_state_reconstruction_weight=(
            args.boundary_state_reconstruction_weight
        ),
        boundary_predictive_energy_floor=(args.boundary_predictive_energy_floor),
        boundary_validation_retention_floor=(args.boundary_validation_retention_floor),
        short_conv_v4_ridge_alphas=(
            (1e-8, 1e-4)
            if args.short_conv_v4_ridge_alphas is None
            else tuple(args.short_conv_v4_ridge_alphas)
        ),
        short_conv_v4_variable_projection_passes=(
            args.short_conv_v4_variable_projection_passes
        ),
        short_conv_v4_writer_trust_radius=(args.short_conv_v4_writer_trust_radius),
        short_conv_v4_fir_trust_radius=(args.short_conv_v4_fir_trust_radius),
        short_conv_v4_bcx_trust_radius=(args.short_conv_v4_bcx_trust_radius),
        short_conv_v4_correction_rms_floor=(args.short_conv_v4_correction_rms_floor),
        short_conv_v4_context_objective_weight=(
            args.short_conv_v4_context_objective_weight
        ),
        forbidden_requests=forbidden,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: V5SelectiveBlankFoldConfig | None = None
    try:
        config = config_from_args(args)
        receipt = run(config)
    except Exception as exc:
        failure = build_failure_diagnostic(
            exc,
            config=config,
            parsed_args=args,
        )
        try:
            _write_diagnostic(args.diagnostic_json, failure)
        except Exception as write_error:
            print(
                "error: v5 fold failed and diagnostic persistence also "
                f"failed: {write_error}",
                file=sys.stderr,
            )
        print(f"error: {exc}", file=sys.stderr)
        for record in _exception_chain(exc):
            print(
                f"cause: {record['type']}: {record['message']}",
                file=sys.stderr,
            )
        return 2
    try:
        _write_diagnostic(config.diagnostic_json, receipt)
    except Exception as exc:
        print(
            "error: the in-memory fold completed but its diagnostic could "
            f"not be persisted: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "diagnostic_json": str(config.diagnostic_json.expanduser().absolute()),
                "child_checkpoint_saved": False,
                "publication_claimed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
