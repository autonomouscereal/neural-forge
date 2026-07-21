"""Prompt-independent coupled weight signatures for gated MLP folding.

The structural unit in a bias-free SwiGLU MLP is one source-neuron triplet:
one row from the gate producer, the matching row from the up producer, and the
matching column from the down consumer.  Given one authenticated orthonormal
source-to-target residual basis ``B``, this module represents every triplet in
the same target residual coordinates::

    gate = W1 @ B
    up   = W3 @ B
    down = (B.T @ W2).T

Each role block is divided by its global RMS before concatenation.  The result
is source-neuron-major and can therefore be supplied directly to a real-medoid
or sparse multi-anchor planner without selecting prompts, traces, activations,
or a subset of neurons.

Construction defaults to deterministic CPU/float64 under one numerical thread.
Callers may instead select one explicitly indexed CUDA device for only the
three projection GEMMs; inputs and results remain canonically authenticated on
CPU and arithmetic remains float64.  Inputs are never mutated and are detached
before arithmetic.  There is no model execution, autograd, optimizer,
training, pruning, or quantization path in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256


GATED_MLP_WEIGHT_SIGNATURE_SCHEMA = (
    "neural-forge-gated-mlp-weight-complete-signature-v1"
)
GATE_INPUT_ROLE = "gate_input_sensitivity"
UP_INPUT_ROLE = "up_input_sensitivity"
DOWN_OUTPUT_ROLE = "down_output_span"
GATED_MLP_WEIGHT_SIGNATURE_ROLES = (
    GATE_INPUT_ROLE,
    UP_INPUT_ROLE,
    DOWN_OUTPUT_ROLE,
)

_INPUT_KEYS = (
    "w1.weight",
    "w3.weight",
    "w2.weight",
    "residual_basis",
)
_NUMERICAL_LOCK = threading.RLock()
_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_rows_read": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)


class GatedMlpWeightSignatureError(ValueError):
    """Base class for structural signature construction failures."""


class GatedMlpWeightSignatureShapeError(GatedMlpWeightSignatureError):
    """A source tensor or residual basis has an incompatible shape."""


class GatedMlpWeightSignatureFiniteError(GatedMlpWeightSignatureError):
    """A source tensor, basis, or computed signature is non-finite."""


class GatedMlpWeightSignatureAuthenticationError(GatedMlpWeightSignatureError):
    """A tensor or report differs from its cryptographic pin."""


class GatedMlpWeightSignatureInvariantError(GatedMlpWeightSignatureError):
    """The completed signature does not prove all structural invariants."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for gated-MLP weight signatures."
        ) from exc
    return torch


@dataclass(frozen=True, slots=True)
class _SignatureComputePolicy:
    requested_device: str
    backend: str
    device_index: int | None
    dtype_name: str


def _compute_policy(
    torch: Any,
    *,
    compute_device: Any,
    compute_dtype: Any,
) -> _SignatureComputePolicy:
    if compute_dtype != "float64":
        raise GatedMlpWeightSignatureShapeError(
            "compute_dtype must be exactly 'float64'; reduced-precision "
            "signature construction is forbidden."
        )
    if compute_device == "cpu":
        return _SignatureComputePolicy(
            requested_device="cpu",
            backend="cpu",
            device_index=None,
            dtype_name="float64",
        )
    if not isinstance(compute_device, str):
        raise GatedMlpWeightSignatureShapeError(
            "compute_device must be 'cpu' or an explicitly indexed CUDA device."
        )
    prefix, separator, index_text = compute_device.partition(":")
    if (
        prefix != "cuda"
        or separator != ":"
        or not index_text.isdecimal()
        or int(index_text) < 0
    ):
        raise GatedMlpWeightSignatureShapeError(
            "CUDA compute_device must be explicitly indexed as 'cuda:N'."
        )
    device_index = int(index_text)
    if not torch.cuda.is_available():
        raise GatedMlpWeightSignatureInvariantError(
            "An indexed CUDA signature backend was requested, but CUDA is unavailable."
        )
    if device_index >= int(torch.cuda.device_count()):
        raise GatedMlpWeightSignatureInvariantError(
            f"CUDA compute_device index {device_index} is unavailable."
        )
    return _SignatureComputePolicy(
        requested_device=compute_device,
        backend="cuda",
        device_index=device_index,
        dtype_name="float64",
    )


def _cuda_backend_state(torch: Any) -> dict[str, bool]:
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }


def _enable_deterministic_cuda_float64(torch: Any) -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _restore_cuda_backend_state(torch: Any, state: Mapping[str, bool]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = state["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = state[
        "cuda_matmul_allow_fp16_reduced_precision_reduction"
    ]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = state[
        "cuda_matmul_allow_bf16_reduced_precision_reduction"
    ]
    torch.use_deterministic_algorithms(
        state["deterministic_algorithms"],
        warn_only=state["deterministic_warn_only"],
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GatedMlpWeightSignatureShapeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GatedMlpWeightSignatureShapeError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GatedMlpWeightSignatureShapeError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise GatedMlpWeightSignatureShapeError(
            f"{name} must be finite and non-negative."
        )
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GatedMlpWeightSignatureAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _source_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or any(int(size) <= 0 for size in value.shape)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise GatedMlpWeightSignatureShapeError(
            f"{name} must be a nonempty dense real floating rank-two tensor."
        )
    detached = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(detached).all().item()):
        raise GatedMlpWeightSignatureFiniteError(f"{name} contains NaN or infinity.")
    return detached


def gated_mlp_weight_signature_tensor_sha256(value: Any) -> str:
    """Return the semantic LFM2-compatible tensor digest for ``value``."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim <= 0
        or any(int(size) <= 0 for size in value.shape)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise GatedMlpWeightSignatureShapeError(
            "Hashed values must be nonempty dense real floating tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        raise GatedMlpWeightSignatureFiniteError(
            "Hashed tensor contains NaN or infinity."
        )
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _expected_hashes(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_INPUT_KEYS):
        raise GatedMlpWeightSignatureAuthenticationError(
            "expected_input_sha256 must pin exactly w1.weight, w3.weight, "
            "w2.weight, and residual_basis."
        )
    return MappingProxyType(
        {
            key: _digest(value[key], f"expected_input_sha256[{key!r}]")
            for key in _INPUT_KEYS
        }
    )


def _rms(torch: Any, value: Any) -> float:
    return float(torch.sqrt(value.square().mean()).item())


def _role_slice(role: str, target_hidden_size: int) -> slice:
    try:
        role_index = GATED_MLP_WEIGHT_SIGNATURE_ROLES.index(role)
    except ValueError as exc:
        raise KeyError(f"Unknown gated-MLP signature role {role!r}.") from exc
    start = role_index * target_hidden_size
    return slice(start, start + target_hidden_size)


@dataclass(frozen=True, slots=True)
class GatedMlpWeightCompleteSignature:
    """Authenticated source-neuron-major data for structural selection."""

    signature: Any
    source_neurons: int
    source_hidden_size: int
    target_hidden_size: int
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        source_neurons = _positive_int(self.source_neurons, "source_neurons")
        source_hidden = _positive_int(self.source_hidden_size, "source_hidden_size")
        target_hidden = _positive_int(self.target_hidden_size, "target_hidden_size")
        if target_hidden > source_hidden:
            raise GatedMlpWeightSignatureInvariantError(
                "A weight-complete folding signature cannot expand hidden width."
            )
        signature = _source_tensor(self.signature, "signature").to(dtype=torch.float64)
        expected_shape = (
            source_neurons,
            len(GATED_MLP_WEIGHT_SIGNATURE_ROLES) * target_hidden,
        )
        if tuple(signature.shape) != expected_shape:
            raise GatedMlpWeightSignatureInvariantError(
                f"signature must have shape {list(expected_shape)}."
            )
        if signature.requires_grad or signature.grad_fn is not None:
            raise GatedMlpWeightSignatureInvariantError(
                "Structural selection data must be detached from autograd."
            )
        if not isinstance(self.report, Mapping):
            raise GatedMlpWeightSignatureInvariantError("report must be a mapping.")
        report = dict(self.report)
        body = dict(report)
        claimed_report_sha256 = body.pop("report_sha256", None)
        if (
            report.get("schema") != GATED_MLP_WEIGHT_SIGNATURE_SCHEMA
            or report.get("source_neurons") != source_neurons
            or report.get("source_hidden_size") != source_hidden
            or report.get("target_hidden_size") != target_hidden
            or report.get("role_order") != list(GATED_MLP_WEIGHT_SIGNATURE_ROLES)
            or report.get("signature_sha256")
            != gated_mlp_weight_signature_tensor_sha256(signature)
            or claimed_report_sha256 != canonical_sha256(body)
            or report.get("invariants", {}).get("all_hold") is not True
        ):
            raise GatedMlpWeightSignatureAuthenticationError(
                "Signature data and its authenticated report disagree."
            )
        balanced_hashes = report.get("balanced_role_sha256")
        if not isinstance(balanced_hashes, Mapping) or any(
            balanced_hashes.get(role)
            != gated_mlp_weight_signature_tensor_sha256(
                signature[:, _role_slice(role, target_hidden)]
            )
            for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
        ):
            raise GatedMlpWeightSignatureAuthenticationError(
                "A balanced role block differs from its authenticated digest."
            )
        object.__setattr__(self, "signature", signature.contiguous().clone())
        object.__setattr__(self, "source_neurons", source_neurons)
        object.__setattr__(self, "source_hidden_size", source_hidden)
        object.__setattr__(self, "target_hidden_size", target_hidden)
        object.__setattr__(self, "report", MappingProxyType(report))

    @property
    def feature_width(self) -> int:
        return int(self.signature.shape[1])

    @property
    def signature_sha256(self) -> str:
        return str(self.report["signature_sha256"])

    @property
    def selection_signature(self) -> Any:
        """Return a detached clone suitable for medoid/anchor selection."""

        self.verify()
        return self.signature.detach().clone()

    def role_block(self, role: str) -> Any:
        """Return one authenticated RMS-balanced role block."""

        self.verify()
        return (
            self.signature[:, _role_slice(role, self.target_hidden_size)]
            .detach()
            .clone()
        )

    def verify(self) -> Mapping[str, Any]:
        """Re-authenticate the materialized signature against its report."""

        observed = gated_mlp_weight_signature_tensor_sha256(self.signature)
        if observed != self.report["signature_sha256"]:
            raise GatedMlpWeightSignatureAuthenticationError(
                "Materialized signature changed after construction."
            )
        body = dict(self.report)
        claimed = body.pop("report_sha256")
        if canonical_sha256(body) != claimed:
            raise GatedMlpWeightSignatureAuthenticationError(
                "Signature report changed after construction."
            )
        return MappingProxyType(
            {
                "signature_sha256": observed,
                "report_sha256": claimed,
                "accepted": True,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_gated_mlp_weight_complete_signature(
    *,
    gate_weight: Any,
    up_weight: Any,
    down_weight: Any,
    residual_basis: Any,
    expected_input_sha256: Mapping[str, str],
    basis_orthogonality_atol: float = 1e-8,
    basis_orthogonality_rtol: float = 1e-8,
    compute_device: str = "cpu",
    compute_dtype: str = "float64",
) -> GatedMlpWeightCompleteSignature:
    """Build and authenticate ``[W1 B, W3 B, (B.T W2).T]``.

    Input pins use :func:`gated_mlp_weight_signature_tensor_sha256`, which is
    byte-compatible with the public semantic-LFM2 tensor digest.  Every source
    neuron participates in every role; no row, prompt, activation, or trace is
    accepted by this API.  ``compute_device`` may be ``"cpu"`` or one
    explicitly indexed CUDA device such as ``"cuda:0"``.  The only supported
    compute dtype is float64; CUDA changes execution placement, not precision.
    """

    torch = _torch()
    compute = _compute_policy(
        torch,
        compute_device=compute_device,
        compute_dtype=compute_dtype,
    )
    atol = _nonnegative_float(basis_orthogonality_atol, "basis_orthogonality_atol")
    rtol = _nonnegative_float(basis_orthogonality_rtol, "basis_orthogonality_rtol")
    expected = _expected_hashes(expected_input_sha256)
    original_values = {
        "w1.weight": gate_weight,
        "w3.weight": up_weight,
        "w2.weight": down_weight,
        "residual_basis": residual_basis,
    }
    inputs = {
        name: _source_tensor(value, name) for name, value in original_values.items()
    }
    observed_before = {
        name: gated_mlp_weight_signature_tensor_sha256(value)
        for name, value in inputs.items()
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed_before[name]}
        for name in _INPUT_KEYS
        if expected[name] != observed_before[name]
    }
    if mismatches:
        raise GatedMlpWeightSignatureAuthenticationError(
            "Structural signature input authentication failed: "
            + json.dumps(mismatches, separators=(",", ":"), sort_keys=True)
        )

    gate = inputs["w1.weight"]
    up = inputs["w3.weight"]
    down = inputs["w2.weight"]
    basis = inputs["residual_basis"]
    source_neurons, source_hidden = map(int, gate.shape)
    if tuple(up.shape) != tuple(gate.shape):
        raise GatedMlpWeightSignatureShapeError(
            "W1 and W3 must have identical [source_neurons, source_hidden] shape."
        )
    if tuple(down.shape) != (source_hidden, source_neurons):
        raise GatedMlpWeightSignatureShapeError(
            "W2 must have shape [source_hidden, source_neurons]."
        )
    if int(basis.shape[0]) != source_hidden:
        raise GatedMlpWeightSignatureShapeError(
            "B must have shape [source_hidden, target_hidden]."
        )
    target_hidden = int(basis.shape[1])
    if target_hidden > source_hidden:
        raise GatedMlpWeightSignatureShapeError(
            "The residual basis cannot expand source hidden width."
        )

    cuda_execution: dict[str, Any] | None = None
    with _NUMERICAL_LOCK:
        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                gate_fp64 = gate.to(dtype=torch.float64)
                up_fp64 = up.to(dtype=torch.float64)
                down_fp64 = down.to(dtype=torch.float64)
                basis_fp64 = basis.to(dtype=torch.float64)
                identity = torch.eye(target_hidden, dtype=torch.float64)
                basis_gram = (basis_fp64.T @ basis_fp64).contiguous()
                basis_error = basis_gram - identity
                basis_maximum_error = float(basis_error.abs().max().item())
                basis_frobenius_error = float(
                    torch.linalg.vector_norm(basis_error).item()
                )
                if not bool(
                    torch.allclose(
                        basis_gram,
                        identity,
                        rtol=rtol,
                        atol=atol,
                    )
                ):
                    raise GatedMlpWeightSignatureInvariantError(
                        "B must have authenticated orthonormal columns."
                    )

                if compute.backend == "cpu":
                    raw_blocks = {
                        GATE_INPUT_ROLE: (gate_fp64 @ basis_fp64).contiguous(),
                        UP_INPUT_ROLE: (up_fp64 @ basis_fp64).contiguous(),
                        DOWN_OUTPUT_ROLE: ((basis_fp64.T @ down_fp64).T.contiguous()),
                    }
                else:
                    workspace_config = os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG",
                        "",
                    )
                    if workspace_config not in {":4096:8", ":16:8"}:
                        raise GatedMlpWeightSignatureInvariantError(
                            "Deterministic CUDA signature GEMMs require "
                            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before "
                            "CUDA initialization."
                        )
                    if compute.device_index is None:  # pragma: no cover - guarded
                        raise AssertionError("Indexed CUDA policy lost its index.")
                    device = torch.device("cuda", compute.device_index)
                    backend_before = _cuda_backend_state(torch)
                    backend_during: dict[str, bool] | None = None
                    properties = torch.cuda.get_device_properties(device)
                    try:
                        _enable_deterministic_cuda_float64(torch)
                        backend_during = _cuda_backend_state(torch)
                        with torch.cuda.device(device):
                            basis_cuda = basis_fp64.to(
                                device=device,
                                dtype=torch.float64,
                            )
                            gate_cuda = gate_fp64.to(
                                device=device,
                                dtype=torch.float64,
                            )
                            gate_block = (gate_cuda @ basis_cuda).to(
                                device="cpu",
                                dtype=torch.float64,
                            )
                            del gate_cuda
                            up_cuda = up_fp64.to(
                                device=device,
                                dtype=torch.float64,
                            )
                            up_block = (up_cuda @ basis_cuda).to(
                                device="cpu",
                                dtype=torch.float64,
                            )
                            del up_cuda
                            down_cuda = down_fp64.to(
                                device=device,
                                dtype=torch.float64,
                            )
                            down_block = (basis_cuda.T @ down_cuda).T.to(
                                device="cpu",
                                dtype=torch.float64,
                            )
                            del down_cuda, basis_cuda
                            torch.cuda.synchronize(device)
                        raw_blocks = {
                            GATE_INPUT_ROLE: gate_block.contiguous(),
                            UP_INPUT_ROLE: up_block.contiguous(),
                            DOWN_OUTPUT_ROLE: down_block.contiguous(),
                        }
                    finally:
                        _restore_cuda_backend_state(torch, backend_before)
                    backend_after = _cuda_backend_state(torch)
                    if backend_after != backend_before:
                        raise GatedMlpWeightSignatureInvariantError(
                            "CUDA signature construction did not restore global "
                            "backend state exactly."
                        )
                    if backend_during != {
                        "deterministic_algorithms": True,
                        "deterministic_warn_only": False,
                        "cuda_matmul_allow_tf32": False,
                        "cudnn_allow_tf32": False,
                        "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
                        "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
                    }:
                        raise GatedMlpWeightSignatureInvariantError(
                            "CUDA signature construction did not hold the required "
                            "deterministic float64 backend policy."
                        )
                    cuda_execution = {
                        "schema": (
                            "neural-forge-gated-mlp-signature-cuda-execution-v1"
                        ),
                        "backend": "cuda",
                        "requested_device": compute.requested_device,
                        "resolved_device": str(device),
                        "device_index": compute.device_index,
                        "device_name": str(properties.name),
                        "compute_capability": [
                            int(properties.major),
                            int(properties.minor),
                        ],
                        "compute_dtype": compute.dtype_name,
                        "cublas_workspace_config": workspace_config,
                        "deterministic_backend_state_during_compute": (backend_during),
                        "global_backend_state_restored": True,
                        "canonical_input_authentication_device": "cpu",
                        "canonical_result_materialization_device": "cpu",
                        "projection_gemm_calls": 3,
                        "cuda_synchronize_calls": 1,
                        "cuda_tensor_stage_count": 4,
                        "cpu_result_materialization_count": 3,
                        "parent_trace_rows_staged": 0,
                        "parent_model_objects_constructed": 0,
                        "parent_model_forward_calls": 0,
                        "child_model_forward_calls": 0,
                        "autograd_grad_calls": 0,
                        "backward_calls": 0,
                        "optimizer_steps": 0,
                        "training_steps": 0,
                    }
                raw_rms = {
                    role: _rms(torch, block) for role, block in raw_blocks.items()
                }
                if any(
                    not math.isfinite(value) or value <= torch.finfo(torch.float64).tiny
                    for value in raw_rms.values()
                ):
                    raise GatedMlpWeightSignatureFiniteError(
                        "Every projected gated-MLP role must have finite nonzero RMS."
                    )
                balanced_blocks = {
                    role: (raw_blocks[role] / raw_rms[role]).contiguous()
                    for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
                }
                balanced_rms = {
                    role: _rms(torch, block) for role, block in balanced_blocks.items()
                }
                signature = torch.cat(
                    tuple(
                        balanced_blocks[role]
                        for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
                    ),
                    dim=1,
                ).contiguous()
        finally:
            torch.set_num_threads(original_threads)

    observed_after = {
        name: gated_mlp_weight_signature_tensor_sha256(value)
        for name, value in original_values.items()
    }
    if observed_after != observed_before:
        raise GatedMlpWeightSignatureInvariantError(
            "A source weight or residual basis changed during construction."
        )
    if (
        tuple(signature.shape) != (source_neurons, 3 * target_hidden)
        or signature.requires_grad
        or signature.grad_fn is not None
        or not bool(torch.isfinite(signature).all().item())
    ):
        raise GatedMlpWeightSignatureInvariantError(
            "The completed structural signature has invalid shape or state."
        )

    raw_role_sha256 = {
        role: gated_mlp_weight_signature_tensor_sha256(raw_blocks[role])
        for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
    }
    balanced_role_sha256 = {
        role: gated_mlp_weight_signature_tensor_sha256(balanced_blocks[role])
        for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
    }
    zero_rows = {
        role: int((raw_blocks[role].square().sum(dim=1) == 0.0).sum().item())
        for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
    }
    rms_balance_error = max(abs(value - 1.0) for value in balanced_rms.values())
    role_coverage = {
        role: {
            "source_neuron_rows_addressed": source_neurons,
            "source_neuron_coverage_fraction": 1.0,
            "source_rows_selected_or_dropped": 0,
            "raw_zero_energy_source_neuron_count": zero_rows[role],
            "raw_rms": raw_rms[role],
            "balanced_rms": balanced_rms[role],
            "raw_sha256": raw_role_sha256[role],
            "balanced_sha256": balanced_role_sha256[role],
        }
        for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
    }
    invariants = {
        "input_sha256_pins_verified": observed_before == dict(expected),
        "source_inputs_unchanged": observed_after == observed_before,
        "basis_columns_orthonormal": True,
        "basis_contracts_or_preserves_hidden_width": target_hidden <= source_hidden,
        "three_structural_roles_present_exactly_once": (
            tuple(raw_blocks) == GATED_MLP_WEIGHT_SIGNATURE_ROLES
        ),
        "all_source_neurons_covered_in_every_role": all(
            value["source_neuron_rows_addressed"] == source_neurons
            and value["source_rows_selected_or_dropped"] == 0
            for value in role_coverage.values()
        ),
        "source_neuron_order_preserved": True,
        "all_full_weight_elements_consumed": True,
        "role_blocks_rms_balanced": rms_balance_error <= 1e-12,
        "signature_source_neuron_major": True,
        "signature_finite": bool(torch.isfinite(signature).all().item()),
        "signature_detached": (
            not signature.requires_grad and signature.grad_fn is None
        ),
        "prompt_independent": True,
        "no_trace_rows_used": True,
        "model_forward_calls_zero": True,
        "training_operations_zero": True,
    }
    if compute.backend == "cpu":
        # Keep the default report byte-compatible with the original
        # deterministic CPU-only implementation.
        invariants["deterministic_cpu_float64_single_thread"] = True
    else:
        invariants.update(
            {
                "deterministic_cuda_float64_explicit_device": (
                    cuda_execution is not None
                    and cuda_execution["requested_device"].startswith("cuda:")
                    and cuda_execution["compute_dtype"] == "float64"
                ),
                "canonical_cpu_input_authentication": True,
                "canonical_cpu_result_materialization": all(
                    block.device.type == "cpu" and block.dtype == torch.float64
                    for block in raw_blocks.values()
                ),
                "cuda_backend_state_restored": (
                    cuda_execution is not None
                    and cuda_execution["global_backend_state_restored"] is True
                ),
                "cuda_projection_scope_exactly_three_gemms": (
                    cuda_execution is not None
                    and cuda_execution["projection_gemm_calls"] == 3
                ),
                "cuda_parent_trace_rows_staged_zero": (
                    cuda_execution is not None
                    and cuda_execution["parent_trace_rows_staged"] == 0
                ),
            }
        )
    invariants["all_hold"] = all(invariants.values())
    if invariants["all_hold"] is not True:
        raise GatedMlpWeightSignatureInvariantError(
            "The structural signature invariants do not all hold."
        )

    signature_sha256 = gated_mlp_weight_signature_tensor_sha256(signature)
    row_index_sha256 = canonical_sha256(
        {
            "start_inclusive": 0,
            "stop_exclusive": source_neurons,
            "order": "ascending-source-neuron-index",
        }
    )
    operation_audit = dict(_ZERO_OPERATIONS)
    if cuda_execution is not None:
        cuda_execution = {
            **cuda_execution,
            "input_sha256": observed_before,
            "raw_result_sha256": raw_role_sha256,
        }
        cuda_execution = {
            **cuda_execution,
            "execution_sha256": canonical_sha256(cuda_execution),
        }
        operation_audit.update(
            {
                "signature_projection_gemm_calls": 3,
                "cpu_signature_projection_gemm_calls": 0,
                "cuda_signature_projection_gemm_calls": 3,
                "cuda_synchronize_calls": 1,
                "cuda_tensor_stage_count": 4,
                "cpu_result_materialization_count": 3,
            }
        )
    body = {
        "schema": GATED_MLP_WEIGHT_SIGNATURE_SCHEMA,
        "method": "full-weight-coupled-gated-mlp-three-role-rms-signature",
        "formula": [
            "gate=W1@B",
            "up=W3@B",
            "down=(B.T@W2).T",
            "signature=concat(gate/rms(gate),up/rms(up),down/rms(down))",
        ],
        "source_neurons": source_neurons,
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "feature_width": 3 * target_hidden,
        "signature_shape": [source_neurons, 3 * target_hidden],
        "signature_dtype": "float64",
        "signature_orientation": "source-neuron-major",
        "role_order": list(GATED_MLP_WEIGHT_SIGNATURE_ROLES),
        "role_slices": {
            role: [
                _role_slice(role, target_hidden).start,
                _role_slice(role, target_hidden).stop,
            ]
            for role in GATED_MLP_WEIGHT_SIGNATURE_ROLES
        },
        "input_sha256": observed_before,
        "input_authentication_required": True,
        "raw_role_sha256": raw_role_sha256,
        "balanced_role_sha256": balanced_role_sha256,
        "raw_role_rms": raw_rms,
        "balanced_role_rms": balanced_rms,
        "maximum_balanced_role_rms_error": rms_balance_error,
        "signature_sha256": signature_sha256,
        "basis_gram_sha256": gated_mlp_weight_signature_tensor_sha256(basis_gram),
        "basis_orthogonality_maximum_absolute_error": basis_maximum_error,
        "basis_orthogonality_frobenius_error": basis_frobenius_error,
        "basis_orthogonality_atol": atol,
        "basis_orthogonality_rtol": rtol,
        "coverage": {
            "source_neuron_count": source_neurons,
            "covered_source_neuron_count": source_neurons,
            "source_neuron_index_sha256": row_index_sha256,
            "role_count": len(GATED_MLP_WEIGHT_SIGNATURE_ROLES),
            "roles": role_coverage,
            "all_source_neurons_have_all_three_roles": True,
            "source_rows_sampled": 0,
            "source_rows_dropped": 0,
            "weight_element_coverage_fraction": 1.0,
        },
        "work": {
            "projection_multiply_accumulate_count": (
                3 * source_neurons * source_hidden * target_hidden
            ),
            "signature_element_count": int(signature.numel()),
            "materialized_full_source_target_assignment_matrix": False,
        },
        "invariants": invariants,
        "trace_rows_used": 0,
        "prompt_rows_used": 0,
        "operation_audit": operation_audit,
    }
    if cuda_execution is not None:
        body["compute_execution"] = cuda_execution
    report = {**body, "report_sha256": canonical_sha256(body)}
    return GatedMlpWeightCompleteSignature(
        signature=signature,
        source_neurons=source_neurons,
        source_hidden_size=source_hidden,
        target_hidden_size=target_hidden,
        report=report,
    )


__all__ = [
    "DOWN_OUTPUT_ROLE",
    "GATED_MLP_WEIGHT_SIGNATURE_ROLES",
    "GATED_MLP_WEIGHT_SIGNATURE_SCHEMA",
    "GATE_INPUT_ROLE",
    "GatedMlpWeightCompleteSignature",
    "GatedMlpWeightSignatureAuthenticationError",
    "GatedMlpWeightSignatureError",
    "GatedMlpWeightSignatureFiniteError",
    "GatedMlpWeightSignatureInvariantError",
    "GatedMlpWeightSignatureShapeError",
    "UP_INPUT_ROLE",
    "build_gated_mlp_weight_complete_signature",
    "gated_mlp_weight_signature_tensor_sha256",
]
