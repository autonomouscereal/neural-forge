"""Closed-form long-horizon semantic refolding for one LFM2 short convolution.

The existing CPCF compiler remains the structural candidate generator: it
folds every parent B/C/x channel into the smaller dense child producer and FIR.
This adapter adds the causal protocol that the compiler deliberately does not
own.  Evidence is captured from the actual child after all earlier layer
installs, trajectories are split four ways, and a safer incumbent-producer
output-writer solve is ordered against the complete structural candidate while
an authenticated no-op copy of the live dense incumbent remains a control.

Only semantic span boundaries are parent-exact endpoints.  Expanded target
subtoken rows may warm the causal FIR state, but they are never relabelled as
parent-exact observations.  Every finite, shape-correct, private dense state
that differs from the incumbent remains eligible.  FIT metrics provide only a
deterministic ordering among those states.  FIT-CHECK, SELECTION, VALIDATION,
and relative-magnitude measurements are diagnostics after that policy; none
can veto the folded state or restore the incumbent.  The incumbent is retained
only when every analytic proposal is structurally invalid or byte-identical.

No model is executed here.  The implementation uses deterministic tensor
algebra under inference/no-grad contexts and performs no backward pass,
optimizer step, training step, pruning operation, or quantization operation.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CompiledCpcfShortConvFold,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvGatePolicy,
    _SHORT_CONV_TORCH_STATE_LOCK,
    compile_lfm2_cpcf_short_conv_fold,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (
    Lfm2SemanticBodyRecipe,
    resolve_lfm2_semantic_body_recipe,
)


LFM2_SEMANTIC_SHORT_CONV_REFOLD_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-refold-v1"
)
LFM2_SEMANTIC_SHORT_CONV_REPLAY_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-replay-evidence-v1"
)
LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-construction-v1"
)
LFM2_SEMANTIC_SHORT_CONV_FHVR_SOURCE_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-fhvr-source-v1"
)
LFM2_SEMANTIC_SHORT_CONV_STRUCTURAL_CONSTRUCTION = "full-source-structural-v1"
LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION = (
    "fhvr-mf-cap10k-sweeps1-full-history-full-coverage-v1"
)
Lfm2SemanticShortConvConstructionId = Literal[
    "full-source-structural-v1",
    "fhvr-mf-cap10k-sweeps1-full-history-full-coverage-v1",
]
LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS: tuple[
    Lfm2SemanticShortConvConstructionId, ...
] = (
    LFM2_SEMANTIC_SHORT_CONV_STRUCTURAL_CONSTRUCTION,
    LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION,
)
_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-short-conv-installation-v1"
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
_CANDIDATE_ELIGIBILITY_BASIS = (
    "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
)
_STRUCTURAL_NOOP_STATUS = "structurally-identical-or-invalid-incumbent-noop"
_FHVR_MATERIALIZATION_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-fhvr-materialization-v1"
)
_FHVR_VALIDATION_ROOT_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-fhvr-validation-root-v1"
)
_NUMERICAL_REQUEST_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-frontier-request-v1"
)
_NUMERICAL_RAW_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-v5-numerical-raw-candidate-v1"
)
_FHVR_RAW_PROVENANCE_SCHEMA = "neural-forge-lfm2-cpcf-short-conv-v5-fhvr-mf-v1"
_NUMERICAL_ZERO_EXECUTION = MappingProxyType(
    {
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)
_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "model_forward_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)


class Lfm2SemanticShortConvRefoldError(RuntimeError):
    """Short-conv replay, candidate gating, or installation failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic short-conv refolding requires PyTorch.") from exc
    return torch


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


def _indexed_cuda_device(torch: Any, value: Any) -> tuple[Any, int]:
    if not isinstance(value, str):
        raise Lfm2SemanticShortConvRefoldError(
            "fold_compute_device must be an explicitly indexed CUDA device."
        )
    prefix, separator, index_text = value.partition(":")
    if (
        prefix != "cuda"
        or separator != ":"
        or not index_text.isdecimal()
        or int(index_text) < 0
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "CUDA fold_compute_device must be explicitly indexed as 'cuda:N'."
        )
    index = int(index_text)
    if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
        raise Lfm2SemanticShortConvRefoldError(
            f"CUDA fold_compute_device index {index} is unavailable."
        )
    return torch.device("cuda", index), index


@contextmanager
def _writer_compute_context(
    torch: Any,
    requested_device: str | None,
) -> Any:
    """Resolve and authenticate the requested writer-solve compute lane."""

    execution: dict[str, Any] = {}
    if requested_device is None:
        # This is deliberately the pre-existing placement policy.  The default
        # semantic compiler omits the new keyword so its candidate/report bytes
        # remain unchanged on both CPU-only and CUDA-capable hosts.
        yield torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), execution
        return
    if requested_device == "cpu":
        device = torch.device("cpu")
        yield device, execution
        body = {
            "schema": "neural-forge-lfm2-short-conv-writer-cpu-execution-v1",
            "backend": "cpu",
            "requested_device": requested_device,
            "requested_device_was_explicit": True,
            "resolved_device": str(device),
            "device_index": None,
            "compute_dtype": "float64",
            "closed_form_linear_solve_count": 1,
            "canonical_input_authentication_device": "cpu",
            "canonical_result_materialization_device": "cpu",
            "cuda_tensor_stage_count": 0,
            "parent_model_forward_calls": 0,
            "child_model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }
        execution.update({**body, "execution_sha256": canonical_sha256(body)})
        return
    device, device_index = _indexed_cuda_device(torch, requested_device)
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise Lfm2SemanticShortConvRefoldError(
            "Deterministic CUDA short-conv writer folding requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before CUDA initialization."
        )
    properties = torch.cuda.get_device_properties(device)
    with _SHORT_CONV_TORCH_STATE_LOCK:
        backend_before = _cuda_backend_state(torch)
        backend_during: dict[str, bool] | None = None
        try:
            _enable_deterministic_cuda_float64(torch)
            backend_during = _cuda_backend_state(torch)
            with torch.cuda.device(device):
                yield device, execution
                torch.cuda.synchronize(device)
        finally:
            _restore_cuda_backend_state(torch, backend_before)
    backend_after = _cuda_backend_state(torch)
    if backend_after != backend_before:
        raise Lfm2SemanticShortConvRefoldError(
            "CUDA short-conv writer solve did not restore global backend state."
        )
    expected_backend = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
    }
    if backend_during != expected_backend:
        raise Lfm2SemanticShortConvRefoldError(
            "CUDA short-conv writer solve did not hold its deterministic FP64 policy."
        )
    body = {
        "schema": "neural-forge-lfm2-short-conv-writer-cuda-execution-v1",
        "backend": "cuda",
        "requested_device": requested_device,
        "resolved_device": str(device),
        "device_index": device_index,
        "device_name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "compute_dtype": "float64",
        "cublas_workspace_config": workspace_config,
        "deterministic_backend_state_during_compute": backend_during,
        "global_backend_state_restored": True,
        "closed_form_linear_solve_count": 1,
        "canonical_input_authentication_device": "cpu",
        "canonical_result_materialization_device": "cpu",
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    execution.update({**body, "execution_sha256": canonical_sha256(body)})


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticShortConvRefoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _finite_cpu_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2SemanticShortConvRefoldError(
            f"{name} must be a detached finite dense rank-{ndim} real tensor."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticShortConvConstruction:
    """One sealed construction choice; metrics cannot rewrite this choice."""

    construction_id: Lfm2SemanticShortConvConstructionId
    construction_sha256: str = ""

    def __post_init__(self) -> None:
        if self.construction_id not in LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS:
            raise Lfm2SemanticShortConvRefoldError(
                "Unknown semantic short-conv construction."
            )
        expected = canonical_sha256(self.authentication_body())
        supplied = str(self.construction_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2SemanticShortConvRefoldError(
                "Semantic short-conv construction authentication is invalid."
            )
        object.__setattr__(self, "construction_sha256", expected)

    @property
    def uses_fhvr_full_coverage(self) -> bool:
        return (
            self.construction_id
            == LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION
        )

    def authentication_body(self) -> dict[str, Any]:
        fhvr = self.uses_fhvr_full_coverage
        return {
            "schema": LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_SCHEMA,
            "construction_id": self.construction_id,
            "construction_kind": (
                "future_hankel_volterra_matrix_free_full_coverage_hybrid"
                if fhvr
                else "existing_full_source_structural"
            ),
            "full_parent_causal_history_required": fhvr,
            "future_observable_realization": fhvr,
            "ordinary_dense_child_b_c_x_fir_output_tensors": True,
            "fit_only_analytic_construction": True,
            "cached_evidence_and_direct_local_weights_only": True,
            "numeric_metrics_are_diagnostic_only": True,
            "numeric_or_binary_behavioral_hard_gate": False,
            "whole_decoded_behavior_is_primary_intelligence_evidence": True,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_shared": 0,
            "parent_sidecars": 0,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "construction_sha256": self.construction_sha256,
        }

    def verify(self) -> bool:
        return self.construction_sha256 == canonical_sha256(self.authentication_body())


def resolve_lfm2_semantic_short_conv_construction(
    value: str | Lfm2SemanticShortConvConstruction | None = None,
) -> Lfm2SemanticShortConvConstruction:
    """Normalize an externally selectable construction without metric choice."""

    if value is None:
        return Lfm2SemanticShortConvConstruction(
            LFM2_SEMANTIC_SHORT_CONV_STRUCTURAL_CONSTRUCTION
        )
    if isinstance(value, Lfm2SemanticShortConvConstruction):
        if not value.verify():
            raise Lfm2SemanticShortConvRefoldError(
                "Semantic short-conv construction changed after authentication."
            )
        return value
    if not isinstance(value, str):
        raise TypeError(
            "short_conv_construction must be an ID or authenticated construction."
        )
    return Lfm2SemanticShortConvConstruction(value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Lfm2SemanticShortConvFhvrSource:
    """Exact local source weights plus the fitted source-to-child chart.

    The direct checkpoint identity admits the weights; cached trajectories
    provide behavior.  The derived context encoder is recomputed here so a
    caller cannot substitute an unrelated source/output chart product.
    """

    layer_index: int
    cache_key: str
    bridge_artifact_sha256: str
    source_model_identity_sha256: str
    parent_session_open_receipt_sha256: str
    parent_identity_reconstruction_sha256: str
    residual_chart_sha256: str
    norm_epsilon: float
    source_operator_norm_weight: Any = field(repr=False, compare=False)
    source_in_proj_weight: Any = field(repr=False, compare=False)
    source_in_proj_bias: Any | None = field(repr=False, compare=False)
    source_fir_weight: Any = field(repr=False, compare=False)
    source_fir_bias: Any | None = field(repr=False, compare=False)
    source_out_proj_weight: Any = field(repr=False, compare=False)
    source_out_proj_bias: Any | None = field(repr=False, compare=False)
    residual_output_chart: Any = field(repr=False, compare=False)
    output_encoder: Any = field(init=False, repr=False, compare=False)
    source_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR source layer_index must be a nonnegative integer."
            )
        for value, name in (
            (self.cache_key, "cache key"),
            (self.bridge_artifact_sha256, "bridge artifact SHA-256"),
            (self.source_model_identity_sha256, "source model identity SHA-256"),
            (
                self.parent_session_open_receipt_sha256,
                "parent session open receipt SHA-256",
            ),
            (
                self.parent_identity_reconstruction_sha256,
                "parent identity reconstruction SHA-256",
            ),
            (self.residual_chart_sha256, "residual chart SHA-256"),
        ):
            _digest(value, name)
        epsilon = self.norm_epsilon
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR source requires one positive finite parent norm epsilon."
            )
        norm = _finite_cpu_tensor(
            self.source_operator_norm_weight,
            "source_operator_norm_weight",
            ndim=1,
        )
        in_weight = _finite_cpu_tensor(
            self.source_in_proj_weight,
            "source_in_proj_weight",
            ndim=2,
        )
        raw_fir = _finite_cpu_tensor(
            self.source_fir_weight,
            "source_fir_weight",
            ndim=int(getattr(self.source_fir_weight, "ndim", -1)),
        )
        if raw_fir.ndim == 3:
            if int(raw_fir.shape[1]) != 1:
                raise Lfm2SemanticShortConvRefoldError(
                    "FHVR source FIR must be depthwise [source,1,length]."
                )
            fir = raw_fir[:, 0, :].contiguous()
        elif raw_fir.ndim == 2:
            fir = raw_fir
        else:
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR source FIR must have rank two or three."
            )
        out_weight = _finite_cpu_tensor(
            self.source_out_proj_weight,
            "source_out_proj_weight",
            ndim=2,
        )
        chart = _finite_cpu_tensor(
            self.residual_output_chart,
            "residual_output_chart",
            ndim=2,
        )
        source_hidden = int(fir.shape[0])
        target_hidden = int(chart.shape[1])
        if (
            target_hidden <= 0
            or target_hidden >= source_hidden
            or tuple(norm.shape) != (source_hidden,)
            or tuple(in_weight.shape) != (3 * source_hidden, source_hidden)
            or tuple(out_weight.shape) != (source_hidden, source_hidden)
            or tuple(chart.shape) != (source_hidden, target_hidden)
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR source weights/chart do not describe one smaller dense child."
            )
        optional = {
            "source_in_proj_bias": (self.source_in_proj_bias, 3 * source_hidden),
            "source_fir_bias": (self.source_fir_bias, source_hidden),
            "source_out_proj_bias": (self.source_out_proj_bias, source_hidden),
        }
        normalized_optional: dict[str, Any | None] = {}
        for name, (value, width) in optional.items():
            normalized = (
                None if value is None else _finite_cpu_tensor(value, name, ndim=1)
            )
            if normalized is not None and tuple(normalized.shape) != (width,):
                raise Lfm2SemanticShortConvRefoldError(
                    f"{name} differs from its source topology."
                )
            normalized_optional[name] = normalized
        output_encoder = (
            out_weight.to(torch.float64).T @ chart.to(torch.float64)
        ).contiguous()
        if not bool(torch.isfinite(output_encoder).all().item()):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR source output encoder is non-finite."
            )
        object.__setattr__(self, "norm_epsilon", float(epsilon))
        object.__setattr__(self, "source_operator_norm_weight", norm)
        object.__setattr__(self, "source_in_proj_weight", in_weight)
        object.__setattr__(
            self, "source_in_proj_bias", normalized_optional["source_in_proj_bias"]
        )
        object.__setattr__(self, "source_fir_weight", fir)
        object.__setattr__(
            self, "source_fir_bias", normalized_optional["source_fir_bias"]
        )
        object.__setattr__(self, "source_out_proj_weight", out_weight)
        object.__setattr__(
            self, "source_out_proj_bias", normalized_optional["source_out_proj_bias"]
        )
        object.__setattr__(self, "residual_output_chart", chart)
        object.__setattr__(self, "output_encoder", output_encoder)
        object.__setattr__(self, "source_sha256", canonical_sha256(self._body()))

    @property
    def source_hidden_size(self) -> int:
        return int(self.source_fir_weight.shape[0])

    @property
    def target_hidden_size(self) -> int:
        return int(self.residual_output_chart.shape[1])

    def _body(self) -> dict[str, Any]:
        tensors = {
            "operator_norm_weight": self.source_operator_norm_weight,
            "in_proj_weight": self.source_in_proj_weight,
            "in_proj_bias": self.source_in_proj_bias,
            "fir_weight": self.source_fir_weight,
            "fir_bias": self.source_fir_bias,
            "out_proj_weight": self.source_out_proj_weight,
            "out_proj_bias": self.source_out_proj_bias,
            "residual_output_chart": self.residual_output_chart,
            "output_encoder": self.output_encoder,
        }
        return {
            "schema": LFM2_SEMANTIC_SHORT_CONV_FHVR_SOURCE_SCHEMA,
            "layer_index": self.layer_index,
            "cache_key": self.cache_key,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "source_model_identity_sha256": self.source_model_identity_sha256,
            "parent_session_open_receipt_sha256": (
                self.parent_session_open_receipt_sha256
            ),
            "parent_identity_reconstruction_sha256": (
                self.parent_identity_reconstruction_sha256
            ),
            "residual_chart_sha256": self.residual_chart_sha256,
            "norm_epsilon": self.norm_epsilon,
            "tensor_sha256": {
                name: None if value is None else _tensor_sha256(value)
                for name, value in tensors.items()
            },
            "source_hidden_size": self.source_hidden_size,
            "target_hidden_size": self.target_hidden_size,
            "output_encoder_formula": "source_out_proj_weight.T@residual_output_chart",
            "exact_local_parent_identity_is_source_admission_authority": True,
            "cached_behavior_is_functional_evidence_source": True,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "network_access_performed": False,
        }

    def verify(self) -> bool:
        return self.source_sha256 == canonical_sha256(self._body())

    def as_dict(self) -> dict[str, Any]:
        if not self.verify():
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR source changed after authentication."
            )
        return {**self._body(), "source_sha256": self.source_sha256}


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _topology_sha256(model: Any) -> str:
    parameters = tuple(model.named_parameters())
    if not parameters:
        raise Lfm2SemanticShortConvRefoldError(
            "Target model has no Parameters to authenticate."
        )
    return canonical_sha256(
        {
            "parameters": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype).removeprefix("torch."),
                    "layout": str(parameter.layout).removeprefix("torch."),
                    "requires_grad": bool(parameter.requires_grad),
                }
                for name, parameter in parameters
            ]
        }
    )


def _span_boundaries(owners: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(int(value) for value in owners)
    return tuple(
        index
        for index, owner in enumerate(selected)
        if index == len(selected) - 1 or selected[index + 1] != owner
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticShortConvReplayEvidence:
    """One authenticated parent history plus replay-fresh child trajectory phase."""

    batch_index: int
    phase: int
    layer_index: int
    alignment_sha256: str
    cache_key: str
    bridge_artifact_sha256: str
    source_capability_sha256: str
    source_history_sha256: str
    compiler_lineage_sha256: str
    captured_child_model_state_sha256: str
    target_to_parent_span_owner_positions: tuple[int, ...]
    endpoint_target_positions: tuple[int, ...]
    actual_child_operator_input_history: Any = field(repr=False, compare=False)
    actual_child_pre_operator_residual_history: Any = field(
        repr=False,
        compare=False,
    )
    parent_b_history: Any = field(repr=False, compare=False)
    parent_c_history: Any = field(repr=False, compare=False)
    parent_x_history: Any = field(repr=False, compare=False)
    parent_post_operator_residual_target_space: Any = field(
        repr=False,
        compare=False,
    )
    endpoint_row_sqrt_weights: Any = field(repr=False, compare=False)
    parent_operator_input_history: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    parent_operator_input_provenance_sha256: str | None = None
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.batch_index, self.phase, self.layer_index)
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Replay batch, phase, and layer coordinates must be nonnegative."
            )
        for value, name in (
            (self.alignment_sha256, "alignment SHA-256"),
            (self.cache_key, "cache key"),
            (self.bridge_artifact_sha256, "bridge artifact SHA-256"),
            (self.source_capability_sha256, "source capability SHA-256"),
            (self.source_history_sha256, "source history SHA-256"),
            (self.compiler_lineage_sha256, "compiler lineage SHA-256"),
            (self.captured_child_model_state_sha256, "captured child state SHA-256"),
        ):
            _digest(value, name)
        child_input = _finite_cpu_tensor(
            self.actual_child_operator_input_history,
            "actual_child_operator_input_history",
            ndim=3,
        )
        child_pre = _finite_cpu_tensor(
            self.actual_child_pre_operator_residual_history,
            "actual_child_pre_operator_residual_history",
            ndim=3,
        )
        parent_b = _finite_cpu_tensor(
            self.parent_b_history,
            "parent_b_history",
            ndim=3,
        )
        parent_c = _finite_cpu_tensor(
            self.parent_c_history,
            "parent_c_history",
            ndim=3,
        )
        parent_x = _finite_cpu_tensor(
            self.parent_x_history,
            "parent_x_history",
            ndim=3,
        )
        parent_post = _finite_cpu_tensor(
            self.parent_post_operator_residual_target_space,
            "parent_post_operator_residual_target_space",
            ndim=3,
        )
        weights = _finite_cpu_tensor(
            self.endpoint_row_sqrt_weights,
            "endpoint_row_sqrt_weights",
            ndim=2,
        ).to(torch.float64)
        parent_input = (
            None
            if self.parent_operator_input_history is None
            else _finite_cpu_tensor(
                self.parent_operator_input_history,
                "parent_operator_input_history",
                ndim=3,
            )
        )
        parent_input_provenance = self.parent_operator_input_provenance_sha256
        if (parent_input is None) != (parent_input_provenance is None):
            raise Lfm2SemanticShortConvRefoldError(
                "Parent operator-input history and provenance must appear together."
            )
        if parent_input_provenance is not None:
            _digest(
                parent_input_provenance,
                "parent operator-input provenance SHA-256",
            )
        target_dtype = child_input.dtype
        if (
            target_dtype not in {torch.bfloat16, torch.float32}
            or child_pre.dtype != target_dtype
            or parent_post.dtype != target_dtype
            or any(value.dtype != torch.bfloat16 for value in (parent_b, parent_c, parent_x))
            or (parent_input is not None and parent_input.dtype != torch.bfloat16)
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Short-conv evidence must keep BF16 parent histories while all "
                "target-coordinate tensors share one BF16 or FP32 dtype."
            )
        batch, target_length, target_hidden = map(int, child_input.shape)
        parent_length = int(parent_b.shape[1])
        source_hidden = int(parent_b.shape[2])
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.target_to_parent_span_owner_positions
        ) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.endpoint_target_positions
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Owner and endpoint positions must be explicit integer indices."
            )
        owners = tuple(self.target_to_parent_span_owner_positions)
        endpoints = tuple(self.endpoint_target_positions)
        boundaries = _span_boundaries(owners) if owners else ()
        endpoint_count = len(endpoints)
        expected_suffix = tuple(range(target_length - endpoint_count, target_length))
        if (
            tuple(child_pre.shape) != tuple(child_input.shape)
            or tuple(parent_c.shape) != tuple(parent_b.shape)
            or tuple(parent_x.shape) != tuple(parent_b.shape)
            or (
                parent_input is not None
                and tuple(parent_input.shape) != tuple(parent_b.shape)
            )
            or int(parent_b.shape[0]) != batch
            or parent_length <= 0
            or source_hidden <= 0
            or len(owners) != target_length
            or not owners
            or owners[0] != 0
            or owners[-1] != parent_length - 1
            or any(
                right - left not in (0, 1) for left, right in zip(owners, owners[1:])
            )
            or sorted(set(owners)) != list(range(parent_length))
            or not endpoints
            or endpoints != expected_suffix
            or any(position not in boundaries for position in endpoints)
            or tuple(parent_post.shape) != (batch, endpoint_count, target_hidden)
            or tuple(weights.shape) != (batch, endpoint_count)
            or bool((weights <= 0.0).any().item())
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Short-conv replay must be one aligned pure-expansion history with "
                "a nonempty parent-exact terminal boundary suffix."
            )
        object.__setattr__(self, "target_to_parent_span_owner_positions", owners)
        object.__setattr__(self, "endpoint_target_positions", endpoints)
        object.__setattr__(self, "actual_child_operator_input_history", child_input)
        object.__setattr__(
            self,
            "actual_child_pre_operator_residual_history",
            child_pre,
        )
        object.__setattr__(self, "parent_b_history", parent_b)
        object.__setattr__(self, "parent_c_history", parent_c)
        object.__setattr__(self, "parent_x_history", parent_x)
        object.__setattr__(
            self,
            "parent_post_operator_residual_target_space",
            parent_post,
        )
        object.__setattr__(self, "endpoint_row_sqrt_weights", weights)
        object.__setattr__(self, "parent_operator_input_history", parent_input)
        object.__setattr__(
            self,
            "parent_operator_input_provenance_sha256",
            parent_input_provenance,
        )
        object.__setattr__(self, "evidence_sha256", canonical_sha256(self._body()))

    @property
    def semantic_span_boundary_positions(self) -> tuple[int, ...]:
        return _span_boundaries(self.target_to_parent_span_owner_positions)

    @property
    def inserted_target_positions(self) -> tuple[int, ...]:
        boundaries = set(self.semantic_span_boundary_positions)
        return tuple(
            index
            for index in range(len(self.target_to_parent_span_owner_positions))
            if index not in boundaries
        )

    def _body(self) -> dict[str, Any]:
        body = {
            "schema": LFM2_SEMANTIC_SHORT_CONV_REPLAY_EVIDENCE_SCHEMA,
            "batch_index": self.batch_index,
            "phase": self.phase,
            "layer_index": self.layer_index,
            "alignment_sha256": self.alignment_sha256,
            "cache_key": self.cache_key,
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "source_capability_sha256": self.source_capability_sha256,
            "source_history_sha256": self.source_history_sha256,
            "compiler_lineage_sha256": self.compiler_lineage_sha256,
            "captured_child_model_state_sha256": (
                self.captured_child_model_state_sha256
            ),
            "target_to_parent_span_owner_positions": list(
                self.target_to_parent_span_owner_positions
            ),
            "semantic_span_boundary_positions": list(
                self.semantic_span_boundary_positions
            ),
            "endpoint_target_positions": list(self.endpoint_target_positions),
            "inserted_target_positions": list(self.inserted_target_positions),
            "tensors": {
                name: _tensor_sha256(getattr(self, name))
                for name in (
                    "actual_child_operator_input_history",
                    "actual_child_pre_operator_residual_history",
                    "parent_b_history",
                    "parent_c_history",
                    "parent_x_history",
                    "parent_post_operator_residual_target_space",
                    "endpoint_row_sqrt_weights",
                )
            },
            "complete_parent_bcx_history_consumed": True,
            "owner_mapping_is_pure_expansion": True,
            "parent_bcx_exact_only_at_semantic_span_boundaries": True,
            "inserted_target_bcx_rows_are_owner_expanded_proxies": True,
            "inserted_target_rows_parent_exact": False,
            "inserted_target_rows_used_for_endpoint_fit": False,
            "inserted_target_bcx_proxy_used_for_source_channel_geometry": True,
            "inserted_target_bcx_proxy_used_for_causal_fir_warmup": True,
            "numeric_parent_child_weight_match_objective": False,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        if self.parent_operator_input_history is not None:
            body["future_observable_parent_input"] = {
                "history_sha256": _tensor_sha256(self.parent_operator_input_history),
                "provenance_sha256": (self.parent_operator_input_provenance_sha256),
                "complete_parent_causal_history": True,
                "reconstructed_from_cached_residual_and_direct_local_rmsnorm": True,
                "parent_forward_calls": 0,
            }
        return body

    def verify(self) -> str:
        return canonical_sha256(self._body())

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2SemanticShortConvEndpointMetrics:
    """Trajectory-balanced whole-endpoint measurements for one partition."""

    case_count: int
    trajectory_count: int
    row_count: int
    endpoint_rmse: float
    endpoint_relative_rmse: float
    endpoint_maximum_absolute_error: float
    endpoint_cosine_similarity_mean: float
    endpoint_cosine_similarity_worst: float

    def __post_init__(self) -> None:
        numeric = (
            self.endpoint_rmse,
            self.endpoint_relative_rmse,
            self.endpoint_maximum_absolute_error,
            self.endpoint_cosine_similarity_mean,
            self.endpoint_cosine_similarity_worst,
        )
        if (
            self.case_count <= 0
            or self.trajectory_count <= 0
            or self.row_count <= 0
            or any(not math.isfinite(value) for value in numeric)
            or any(value < 0.0 for value in numeric[:3])
            or any(value < -1.0 or value > 1.0 for value in numeric[3:])
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Endpoint metrics must be finite measurements of nonempty evidence."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "trajectory_count": self.trajectory_count,
            "row_count": self.row_count,
            "endpoint_rmse": self.endpoint_rmse,
            "endpoint_relative_rmse": self.endpoint_relative_rmse,
            "endpoint_maximum_absolute_error": (self.endpoint_maximum_absolute_error),
            "endpoint_cosine_similarity_mean": (self.endpoint_cosine_similarity_mean),
            "endpoint_cosine_similarity_worst": (self.endpoint_cosine_similarity_worst),
        }


@dataclass(frozen=True, slots=True)
class _BalancedCase:
    evidence: Lfm2SemanticShortConvReplayEvidence
    endpoint_row_weights: Any = field(repr=False, compare=False)
    case_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        torch = _torch()
        weights = _finite_cpu_tensor(
            self.endpoint_row_weights,
            "balanced endpoint row weights",
            ndim=2,
        ).to(torch.float64)
        if tuple(weights.shape) != tuple(
            self.evidence.endpoint_row_sqrt_weights.shape
        ) or bool((weights <= 0.0).any().item()):
            raise Lfm2SemanticShortConvRefoldError(
                "Balanced endpoint weights must remain positive and row aligned."
            )
        object.__setattr__(self, "endpoint_row_weights", weights)
        object.__setattr__(
            self,
            "case_sha256",
            canonical_sha256(
                {
                    "evidence_sha256": self.evidence.evidence_sha256,
                    "endpoint_row_weights_sha256": _tensor_sha256(weights),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    state: Mapping[str, Any] = field(repr=False, compare=False)
    provenance: Mapping[str, Any]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.name not in {
            "incumbent-noop",
            "full-source-structural",
            "incumbent-producer-output-writer",
            "full-source-producer-output-writer",
            "full-history-fhvr-full-coverage-hybrid",
        }:
            raise Lfm2SemanticShortConvRefoldError("Unknown short-conv candidate.")
        normalized = {
            str(name): value.detach().to(device="cpu").contiguous().clone()
            for name, value in self.state.items()
        }
        if not normalized:
            raise Lfm2SemanticShortConvRefoldError("Candidate state is empty.")
        object.__setattr__(self, "state", MappingProxyType(normalized))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(
            self,
            "state_sha256",
            canonical_sha256(
                {
                    "name": self.name,
                    "state": {
                        name: _tensor_sha256(value)
                        for name, value in sorted(normalized.items())
                    },
                    "provenance": dict(self.provenance),
                }
            ),
        )


def _candidate_state_structural_report(
    state: Mapping[str, Any],
    incumbent_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Describe structural safety without applying a numeric-quality veto."""

    torch = _torch()
    exact_roles = set(state) == set(incumbent_state)
    role_reports: dict[str, dict[str, Any]] = {}
    candidate_pointers: list[int] = []
    state_hashes: dict[str, str] = {}
    for name in sorted(set(state) | set(incumbent_state)):
        value = state.get(name)
        incumbent = incumbent_state.get(name)
        tensor = isinstance(value, torch.Tensor)
        incumbent_tensor = isinstance(incumbent, torch.Tensor)
        dense_strided = bool(tensor and value.layout == torch.strided)
        cpu_materialized = bool(tensor and value.device.type == "cpu")
        shape_matches = bool(
            tensor and incumbent_tensor and tuple(value.shape) == tuple(incumbent.shape)
        )
        dtype_matches = bool(
            tensor and incumbent_tensor and value.dtype == incumbent.dtype
        )
        detached = bool(tensor and not value.requires_grad and value.grad_fn is None)
        finite = bool(
            tensor
            and value.device.type != "meta"
            and bool(torch.isfinite(value).all().item())
        )
        owns_exact_storage = False
        private_from_incumbent = False
        if tensor and dense_strided and cpu_materialized and int(value.numel()) > 0:
            storage_elements = int(value.untyped_storage().nbytes()) // int(
                value.element_size()
            )
            owns_exact_storage = bool(
                value._base is None
                and int(value.storage_offset()) == 0
                and storage_elements == int(value.numel())
            )
            candidate_pointers.append(int(value.data_ptr()))
            private_from_incumbent = bool(
                not incumbent_tensor
                or incumbent.device.type == "meta"
                or int(value.data_ptr()) != int(incumbent.data_ptr())
            )
            state_hashes[name] = _tensor_sha256(value)
        role_reports[name] = {
            "present_in_candidate": name in state,
            "present_in_incumbent": name in incumbent_state,
            "tensor": tensor,
            "dense_strided": dense_strided,
            "cpu_materialized": cpu_materialized,
            "shape_matches_incumbent": shape_matches,
            "dtype_matches_incumbent": dtype_matches,
            "detached": detached,
            "all_elements_finite": finite,
            "owns_exact_storage_allocation": owns_exact_storage,
            "private_from_live_incumbent": private_from_incumbent,
            "shape": list(value.shape) if tensor else None,
            "dtype": str(value.dtype).removeprefix("torch.") if tensor else None,
            "element_count": int(value.numel()) if tensor else None,
        }
    storage_is_private_per_role = len(candidate_pointers) == len(
        set(candidate_pointers)
    )
    structural_safety = bool(
        exact_roles
        and role_reports
        and storage_is_private_per_role
        and all(
            report[flag] is True
            for report in role_reports.values()
            for flag in (
                "tensor",
                "dense_strided",
                "cpu_materialized",
                "shape_matches_incumbent",
                "dtype_matches_incumbent",
                "detached",
                "all_elements_finite",
                "owns_exact_storage_allocation",
                "private_from_live_incumbent",
            )
        )
    )
    effectful = bool(
        structural_safety
        and any(
            not torch.equal(state[name], incumbent_state[name])
            for name in sorted(incumbent_state)
        )
    )
    body = {
        "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
        "exact_dense_topology_roles_covered": exact_roles,
        "storage_is_private_per_role": storage_is_private_per_role,
        "role_reports": role_reports,
        "state_tensor_bundle_sha256": (
            canonical_sha256(state_hashes)
            if len(state_hashes) == len(state) and exact_roles
            else None
        ),
        "structural_safety_passed": structural_safety,
        "effectful_nonincumbent": effectful,
        "structurally_eligible": structural_safety and effectful,
        "numeric_metrics_used_for_structural_eligibility": False,
        "numeric_thresholds_used_for_structural_eligibility": False,
        "relative_magnitude_used_for_structural_eligibility": False,
    }
    return MappingProxyType(
        {
            **body,
            "structural_report_sha256": canonical_sha256(body),
        }
    )


def _candidate_relative_magnitude_diagnostics(
    state: Mapping[str, Any],
    incumbent_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Measure mutation size for review; never use it for eligibility or rank."""

    torch = _torch()
    reference_floor = 1e-12
    delta_squared = torch.zeros((), dtype=torch.float64)
    incumbent_squared = torch.zeros((), dtype=torch.float64)
    element_count = 0
    maximum_absolute_delta = 0.0
    for name in sorted(incumbent_state):
        candidate = state[name].to(torch.float64)
        incumbent = incumbent_state[name].to(torch.float64)
        delta = candidate - incumbent
        delta_squared += delta.square().sum()
        incumbent_squared += incumbent.square().sum()
        element_count += int(delta.numel())
        maximum_absolute_delta = max(
            maximum_absolute_delta,
            float(delta.abs().max().item()),
        )
    denominator = max(element_count, 1)
    delta_rms = float(torch.sqrt(delta_squared / denominator).item())
    incumbent_rms = float(torch.sqrt(incumbent_squared / denominator).item())
    body = {
        "delta_rms": delta_rms,
        "incumbent_rms": incumbent_rms,
        "relative_delta_rms_ratio": delta_rms / max(incumbent_rms, reference_floor),
        "maximum_absolute_delta": maximum_absolute_delta,
        "reference_floor": reference_floor,
        "element_count": element_count,
        "diagnostic_only": True,
        "relative_magnitude_threshold_used_for_eligibility": False,
        "relative_magnitude_used_for_candidate_ordering": False,
        "relative_magnitude_has_final_intelligence_authority": False,
    }
    return MappingProxyType(
        {
            **body,
            "magnitude_diagnostics_sha256": canonical_sha256(body),
        }
    )


def _short_conv_module(target_model: Any, layer_index: int) -> Any:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= len(layers)
    ):
        raise Lfm2SemanticShortConvRefoldError(
            f"Target model has no layer {layer_index}."
        )
    operator = getattr(layers[layer_index], "conv", None)
    if operator is None or any(
        getattr(operator, name, None) is None
        for name in ("in_proj", "conv", "out_proj")
    ):
        raise Lfm2SemanticShortConvRefoldError(
            f"Layer {layer_index} is not one ordinary dense LFM2 short convolution."
        )
    return operator


def _parameter_bundle(target_model: Any, layer_index: int) -> Mapping[str, Any]:
    torch = _torch()
    operator = _short_conv_module(target_model, layer_index)
    prefix = f"model.layers.{layer_index}.conv."
    values: dict[str, Any] = {
        prefix + "in_proj.weight": operator.in_proj.weight,
        prefix + "conv.weight": operator.conv.weight,
        prefix + "out_proj.weight": operator.out_proj.weight,
    }
    for module_name in ("in_proj", "conv", "out_proj"):
        bias = getattr(getattr(operator, module_name), "bias", None)
        if bias is not None:
            values[prefix + module_name + ".bias"] = bias
    named = dict(target_model.named_parameters())
    if (
        any(not isinstance(value, torch.nn.Parameter) for value in values.values())
        or any(named.get(name) is not value for name, value in values.items())
        or len({id(value) for value in values.values()}) != len(values)
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Target short-conv Parameters are missing, aliased, or renamed."
        )
    dtypes = {value.dtype for value in values.values()}
    if len(dtypes) != 1 or not dtypes.issubset({torch.bfloat16, torch.float32}):
        raise Lfm2SemanticShortConvRefoldError(
            "Target short-conv Parameters must share one dense BF16 or FP32 dtype."
        )
    hidden = int(operator.out_proj.weight.shape[0])
    fir_length = int(operator.conv.weight.shape[2])
    expected = {
        prefix + "in_proj.weight": (3 * hidden, hidden),
        prefix + "conv.weight": (hidden, 1, fir_length),
        prefix + "out_proj.weight": (hidden, hidden),
    }
    optional = {
        prefix + "in_proj.bias": (3 * hidden,),
        prefix + "conv.bias": (hidden,),
        prefix + "out_proj.bias": (hidden,),
    }
    if (
        hidden <= 0
        or fir_length <= 0
        or any(tuple(values[name].shape) != shape for name, shape in expected.items())
        or any(
            name in values and tuple(values[name].shape) != shape
            for name, shape in optional.items()
        )
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Target short-conv Parameters do not form canonical dense topology."
        )
    return MappingProxyType(values)


def _suffix_state(parameters: Mapping[str, Any], layer_index: int) -> Mapping[str, Any]:
    prefix = f"model.layers.{layer_index}.conv."
    return MappingProxyType(
        {
            name.removeprefix(prefix): value.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
            for name, value in parameters.items()
        }
    )


def _exact_zero_state(state: Mapping[str, Any]) -> bool:
    torch = _torch()
    return bool(state) and all(
        int(torch.count_nonzero(value).item()) == 0 for value in state.values()
    )


def _cold_start_weight_axis_coverage(
    value: Any,
    *,
    parameter_role: str,
) -> Mapping[str, Any]:
    """Prove the same exact axes required by the final stack coverage gate."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.ndim < 2
        or value.numel() <= 0
    ):
        raise Lfm2SemanticShortConvRefoldError(
            f"Cold-start {parameter_role} is not a materialized dense floating tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    finite = bool(torch.isfinite(tensor.to(dtype=torch.float32)).all().item())
    nonzero = tensor.ne(0)
    output_units = int(tensor.shape[0])
    input_units = int(tensor.shape[-1])
    covered_output_units = int(
        nonzero.reshape(output_units, -1).any(dim=1).sum().item()
    )
    covered_input_units = int(nonzero.reshape(-1, input_units).any(dim=0).sum().item())
    nonzero_count = int(nonzero.sum().item())
    body = {
        "schema": "neural-forge-short-conv-cold-start-axis-coverage-v1",
        "parameter_role": parameter_role,
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "tensor_sha256": _tensor_sha256(tensor),
        "element_count": int(tensor.numel()),
        "nonzero_element_count": nonzero_count,
        "finite": finite,
        "output_structural_unit_count": output_units,
        "covered_output_structural_unit_count": covered_output_units,
        "input_structural_unit_count": input_units,
        "covered_input_structural_unit_count": covered_input_units,
        "all_output_structural_units_covered": covered_output_units == output_units,
        "all_input_structural_units_covered": covered_input_units == input_units,
        "structurally_nonzero": bool(
            finite
            and nonzero_count > 0
            and covered_output_units == output_units
            and covered_input_units == input_units
        ),
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _cold_start_structural_axis_coverage(candidate: _Candidate) -> Mapping[str, Any]:
    required_weights = ("in_proj.weight", "conv.weight", "out_proj.weight")
    receipts = {
        name: _cold_start_weight_axis_coverage(
            candidate.state[name],
            parameter_role=name,
        )
        for name in required_weights
    }
    body = {
        "schema": "neural-forge-short-conv-cold-start-axis-coverage-set-v1",
        "required_weight_roles": list(required_weights),
        "role_receipts": {name: dict(receipts[name]) for name in required_weights},
        "all_required_weights_structurally_nonzero": all(
            receipts[name]["structurally_nonzero"] for name in required_weights
        ),
        "blank_or_partially_blank_required_weight_count": sum(
            int(not receipts[name]["structurally_nonzero"]) for name in required_weights
        ),
        "every_output_row_and_input_or_lag_axis_covered": all(
            receipts[name]["all_output_structural_units_covered"]
            and receipts[name]["all_input_structural_units_covered"]
            for name in required_weights
        ),
        "repair_policy": (
            "use-only-authenticated-full-source-structural-coefficients-"
            "otherwise-fail-before-mutation"
        ),
        "synthetic_epsilon_fill_permitted": False,
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _sealed_mapping_matches(value: Any, seal_name: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    body = dict(value)
    supplied = body.pop(seal_name, None)
    return bool(
        isinstance(supplied, str)
        and len(supplied) == 64
        and canonical_sha256(body) == supplied
    )


def _recursive_fhvr_zero_execution_claims(value: Any) -> bool:
    forbidden = {
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "optimizer_constructions",
        "autograd_grad_calls",
        "autograd_graphs",
        "parameters_pruned",
        "parameters_quantized",
    }
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name in forbidden and (type(child) is not int or child != 0):
                return False
            if not _recursive_fhvr_zero_execution_claims(child):
                return False
    elif isinstance(value, (list, tuple)):
        return all(_recursive_fhvr_zero_execution_claims(child) for child in value)
    return True


def _fhvr_compute_execution_authenticated(value: Any) -> bool:
    if not _sealed_mapping_matches(value, "execution_sha256"):
        return False
    execution = dict(value)
    backend = execution.get("backend")
    expected = {
        "schema",
        "backend",
        "requested_device",
        "resolved_device",
        "device_index",
        "compute_dtype",
        "fhvr_policy_sha256",
        "request_sha256",
        "pure_candidate_state_sha256",
        "hybrid_candidate_state_sha256",
        "canonical_input_authentication_device",
        "canonical_result_materialization_device",
        "operation_audit",
        "execution_sha256",
    }
    if backend == "cuda":
        expected.update(
            {"device_name", "compute_capability", "cublas_workspace_config"}
        )
    return bool(
        set(execution) == expected
        and execution.get("schema")
        == "neural-forge-lfm2-semantic-short-conv-fhvr-execution-v1"
        and backend in {"cpu", "cuda"}
        and dict(execution.get("operation_audit", {})) == dict(_ZERO_OPERATIONS)
        and _recursive_fhvr_zero_execution_claims(execution)
    )


def _fhvr_cold_start_provenance_authenticated(
    candidate: _Candidate,
    *,
    expected_root: Mapping[str, Any] | None,
) -> bool:
    """Authenticate the FHVR equivalents of a structural-compiler proof."""

    if not _sealed_mapping_matches(expected_root, "root_sha256"):
        return False
    assert isinstance(expected_root, Mapping)
    provenance = candidate.provenance
    construction = provenance.get("construction")
    source = provenance.get("source")
    request = provenance.get("numerical_frontier_request")
    alignment = provenance.get("parent_context_alignment")
    writer = provenance.get("writer_anchor_report")
    pure = provenance.get("pure_fhvr_report")
    hybrid = provenance.get("hybrid_report")
    execution = provenance.get("fhvr_compute_execution")
    audit = provenance.get("operation_audit")
    materialization = provenance.get("fhvr_materialization_receipt")
    if any(
        not isinstance(value, Mapping)
        for value in (
            construction,
            source,
            request,
            alignment,
            writer,
            pure,
            hybrid,
            execution,
            audit,
            materialization,
        )
    ):
        return False
    assert isinstance(construction, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(request, Mapping)
    assert isinstance(alignment, Mapping)
    assert isinstance(writer, Mapping)
    assert isinstance(pure, Mapping)
    assert isinstance(hybrid, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(audit, Mapping)
    assert isinstance(materialization, Mapping)
    method = hybrid.get("method_provenance")
    pure_method = pure.get("method_provenance")
    writer_method = writer.get("method_provenance")
    coverage = method.get("full_coverage") if isinstance(method, Mapping) else None
    if any(
        not isinstance(value, Mapping)
        for value in (method, pure_method, writer_method, coverage)
    ):
        return False
    assert isinstance(method, Mapping)
    assert isinstance(pure_method, Mapping)
    assert isinstance(writer_method, Mapping)
    assert isinstance(coverage, Mapping)
    writer_count = coverage.get("writer_channel_count")
    fhvr_count = coverage.get("fhvr_channel_count")
    target_count = coverage.get("target_group_count")
    state_inventory = {
        name: _tensor_sha256(value) for name, value in sorted(candidate.state.items())
    }
    state_sha256 = canonical_sha256(state_inventory)
    source_inventory = source.get("tensor_sha256")
    request_inventory = request.get("source_tensor_sha256")
    source_role_map = {
        "source_in_proj_weight": "in_proj_weight",
        "source_in_proj_bias": "in_proj_bias",
        "source_fir_kernel": "fir_weight",
        "source_fir_bias": "fir_bias",
        "source_out_proj_weight": "out_proj_weight",
        "source_out_proj_bias": "out_proj_bias",
        "output_encoder": "output_encoder",
    }
    expected_raw_role_mapping = {
        "in_proj_weight": {
            "semantic_role": "in_proj.weight",
            "transform": "identity",
        },
        "in_proj_bias": {
            "semantic_role": "in_proj.bias",
            "transform": "identity_or_absent",
        },
        "fir_kernel": {
            "semantic_role": "conv.weight",
            "transform": "unsqueeze_depthwise_axis_1",
        },
        "fir_bias": {
            "semantic_role": "conv.bias",
            "transform": "identity_or_absent",
        },
        "writer_weight": {
            "semantic_role": "out_proj.weight",
            "transform": "identity",
        },
        "writer_bias": {
            "semantic_role": "out_proj.bias",
            "transform": "identity_or_absent",
        },
    }
    expected_source_request_inventory = (
        {
            request_role: {
                "source_role": source_role,
                "source_tensor_sha256": source_inventory.get(source_role),
                "numerical_request_tensor_sha256": request_inventory.get(request_role),
            }
            for request_role, source_role in source_role_map.items()
        }
        if isinstance(source_inventory, Mapping)
        and isinstance(request_inventory, Mapping)
        else None
    )
    conversion = materialization.get("target_dtype_conversion")
    conversion_valid = bool(
        isinstance(conversion, Mapping)
        and set(conversion) == set(expected_raw_role_mapping)
        and all(
            isinstance(conversion.get(raw_role), Mapping)
            and conversion[raw_role].get("semantic_role")
            == mapping["semantic_role"]
            and conversion[raw_role].get("transform") == mapping["transform"]
            and conversion[raw_role].get("materialized_tensor_sha256")
            == state_inventory.get(mapping["semantic_role"])
            and conversion[raw_role].get("target_dtype")
            == (
                None
                if mapping["semantic_role"] not in candidate.state
                else str(candidate.state[mapping["semantic_role"]].dtype).removeprefix(
                    "torch."
                )
            )
            and (
                conversion[raw_role].get("raw_tensor_sha256") is None
            )
            == (mapping["semantic_role"] not in candidate.state)
            for raw_role, mapping in expected_raw_role_mapping.items()
        )
    )
    raw_reports = (writer, pure, hybrid)
    raw_methods = (writer_method, pure_method, method)
    return bool(
        candidate.name == "full-history-fhvr-full-coverage-hybrid"
        and expected_root.get("schema") == _FHVR_VALIDATION_ROOT_SCHEMA
        and expected_root.get("kind") == "fhvr_cold_start_validation_root"
        and expected_root.get("raw_hybrid_authentication_verified_at_rooting") is True
        and expected_root.get("construction") == construction
        and expected_root.get("construction_sha256")
        == construction.get("construction_sha256")
        and expected_root.get("source") == source
        and expected_root.get("source_sha256") == source.get("source_sha256")
        and expected_root.get("materialization_receipt") == materialization
        and expected_root.get("materialization_receipt_sha256")
        == materialization.get("report_sha256")
        and expected_root.get("candidate_state_sha256") == candidate.state_sha256
        and provenance.get("method")
        == "semantic_full_history_fhvr_full_coverage_hybrid"
        and _sealed_mapping_matches(construction, "construction_sha256")
        and construction.get("schema")
        == LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_SCHEMA
        and construction.get("construction_id")
        == LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION
        and provenance.get("construction_sha256")
        == construction.get("construction_sha256")
        and all(
            construction.get(name) == 0
            for name in (
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
        and _sealed_mapping_matches(source, "source_sha256")
        and source.get("schema") == LFM2_SEMANTIC_SHORT_CONV_FHVR_SOURCE_SCHEMA
        and provenance.get("source_sha256") == source.get("source_sha256")
        and _sealed_mapping_matches(request, "report_sha256")
        and request.get("schema") == _NUMERICAL_REQUEST_SCHEMA
        and request.get("kind") == "detached_fit_only_numerical_frontier_request"
        and request.get("fit_only") is True
        and request.get("selection_cases_exposed") is False
        and request.get("validation_cases_exposed") is False
        and dict(request.get("execution", {})) == dict(_NUMERICAL_ZERO_EXECUTION)
        and provenance.get("numerical_frontier_request_sha256")
        == request.get("report_sha256")
        and _sealed_mapping_matches(alignment, "result_sha256")
        and provenance.get("parent_context_alignment_sha256")
        == alignment.get("result_sha256")
        and all(_sealed_mapping_matches(value, "report_sha256") for value in raw_reports)
        and all(value.get("schema") == _NUMERICAL_RAW_CANDIDATE_SCHEMA for value in raw_reports)
        and all(
            value.get("kind") == "authenticated_fit_only_numerical_raw_candidate"
            for value in raw_reports
        )
        and all(value.get("fit_only") is True for value in raw_reports)
        and all(value.get("selection_cases_consumed") == 0 for value in raw_reports)
        and all(value.get("validation_cases_consumed") == 0 for value in raw_reports)
        and all(
            dict(value.get("execution", {})) == dict(_NUMERICAL_ZERO_EXECUTION)
            for value in raw_reports
        )
        and all(value.get("fit_only") is True for value in raw_methods)
        and all(value.get("selection_cases_consumed") == 0 for value in raw_methods)
        and all(value.get("validation_cases_consumed") == 0 for value in raw_methods)
        and _recursive_fhvr_zero_execution_claims(raw_reports)
        and provenance.get("writer_anchor_report_sha256")
        == writer.get("report_sha256")
        and provenance.get("pure_fhvr_report_sha256") == pure.get("report_sha256")
        and provenance.get("hybrid_report_sha256") == hybrid.get("report_sha256")
        and writer.get("candidate_kind") == "accepted_fit_projection_path_anchor"
        and pure.get("candidate_kind")
        == "fit_only_future_hankel_volterra_matrix_free_realization"
        and hybrid.get("candidate_kind")
        == "fit_only_full_coverage_fhvr_writer_response_refactor"
        and pure_method.get("schema") == _FHVR_RAW_PROVENANCE_SCHEMA
        and method.get("schema") == _FHVR_RAW_PROVENANCE_SCHEMA
        and hybrid.get("request_sha256") == request.get("report_sha256")
        and hybrid.get("state_sha256")
        == execution.get("hybrid_candidate_state_sha256")
        and method.get("request_sha256") == request.get("report_sha256")
        and method.get("source_writer_candidate_report_sha256")
        == writer.get("report_sha256")
        and method.get("source_writer_candidate_state_sha256")
        == writer.get("state_sha256")
        and method.get("source_fhvr_candidate_report_sha256")
        == pure.get("report_sha256")
        and method.get("source_fhvr_candidate_state_sha256")
        == pure.get("state_sha256")
        and execution.get("pure_candidate_state_sha256") == pure.get("state_sha256")
        and provenance.get("materialized_target_state_sha256") == state_sha256
        and _fhvr_compute_execution_authenticated(execution)
        and provenance.get("fhvr_compute_execution_sha256")
        == execution.get("execution_sha256")
        and dict(audit) == dict(_ZERO_OPERATIONS)
        and _sealed_mapping_matches(materialization, "report_sha256")
        and materialization.get("schema") == _FHVR_MATERIALIZATION_SCHEMA
        and materialization.get("kind")
        == "authenticated_raw_hybrid_to_semantic_dense_materialization"
        and materialization.get("raw_hybrid_authentication_verified_at_materialization")
        is True
        and materialization.get("construction_sha256")
        == construction.get("construction_sha256")
        and materialization.get("source_sha256") == source.get("source_sha256")
        and materialization.get("request_sha256") == request.get("report_sha256")
        and materialization.get("hybrid_report_sha256")
        == hybrid.get("report_sha256")
        and materialization.get("hybrid_state_sha256") == hybrid.get("state_sha256")
        and materialization.get("raw_to_semantic_role_mapping")
        == expected_raw_role_mapping
        and materialization.get("source_to_request_tensor_inventory")
        == expected_source_request_inventory
        and conversion_valid
        and materialization.get("materialized_state_tensor_sha256") == state_inventory
        and materialization.get("materialized_target_state_sha256") == state_sha256
        and dict(materialization.get("operation_audit", {}))
        == dict(_ZERO_OPERATIONS)
        and provenance.get("fhvr_materialization_receipt_sha256")
        == materialization.get("report_sha256")
        and provenance.get("full_source_channel_coverage") is True
        and provenance.get("target_native_dense_weights") is True
        and coverage.get("source_channels_dropped") == 0
        and coverage.get("source_channels_pruned") == 0
        and coverage.get("each_source_channel_assigned_exactly_once") is True
        and isinstance(writer_count, int)
        and not isinstance(writer_count, bool)
        and writer_count > 0
        and fhvr_count == writer_count
        and target_count == writer_count
        and coverage.get("writer_channels_covered") == writer_count
        and coverage.get("fhvr_channels_covered") == writer_count
        and coverage.get("writer_source_channel_coverage") == list(range(writer_count))
        and coverage.get("fhvr_source_channel_coverage") == list(range(writer_count))
        and coverage.get("source_channel_tuples_presented") == 2 * writer_count
        and coverage.get("source_channel_tuples_assigned") == 2 * writer_count
    )


def _validate_cold_start_structural_plant(
    candidate: _Candidate,
    *,
    expected_fhvr_root: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    torch = _torch()
    provenance = candidate.provenance
    state = candidate.state
    required_weights = {"in_proj.weight", "conv.weight", "out_proj.weight"}
    structural_candidate = candidate.name in {
        "full-source-structural",
        "full-source-producer-output-writer",
    }
    structural_provenance = bool(
        structural_candidate
        and provenance.get("full_source_coverage") is True
        and provenance.get("source_channels_dropped") == 0
    )
    fhvr_provenance = _fhvr_cold_start_provenance_authenticated(
        candidate,
        expected_root=expected_fhvr_root,
    )
    if (
        not (structural_provenance or fhvr_provenance)
        or not required_weights.issubset(state)
        or any(not bool(torch.isfinite(value).all().item()) for value in state.values())
        or any(
            int(torch.count_nonzero(state[name]).item()) == 0
            for name in required_weights
        )
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "An exact-zero short-conv incumbent requires one finite, role-complete, "
            "full-source structural plant."
        )
    if structural_candidate:
        compiler_report = provenance.get("structural_compiler_report")
        execution = (
            compiler_report.get("execution")
            if isinstance(compiler_report, Mapping)
            else None
        )
        if (
            not isinstance(execution, Mapping)
            or execution.get("training_steps") != 0
            or execution.get("parameters_pruned") != 0
            or execution.get("parameters_quantized") != 0
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "The cold-start structural plant lacks a zero-training "
                "compression proof."
            )
    coverage = _cold_start_structural_axis_coverage(candidate)
    if (
        coverage["all_required_weights_structurally_nonzero"] is not True
        or coverage["blank_or_partially_blank_required_weight_count"] != 0
        or coverage["every_output_row_and_input_or_lag_axis_covered"] is not True
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "The full-source cold-start structural map leaves a required output "
            "row, input axis, or FIR lag axis disconnected; no authenticated "
            "structural coefficient supports a deterministic repair before install."
        )
    return coverage


def _split_evidence(
    evidence: Sequence[Lfm2SemanticShortConvReplayEvidence],
) -> Mapping[str, tuple[Lfm2SemanticShortConvReplayEvidence, ...]]:
    grouped: dict[int, list[Lfm2SemanticShortConvReplayEvidence]] = {}
    coordinates: set[tuple[int, int]] = set()
    for item in evidence:
        coordinate = (item.batch_index, item.phase)
        if coordinate in coordinates:
            raise Lfm2SemanticShortConvRefoldError(
                "Short-conv evidence repeats one trajectory phase."
            )
        coordinates.add(coordinate)
        grouped.setdefault(item.batch_index, []).append(item)
    batch_indices = tuple(sorted(grouped))
    count = len(batch_indices)
    if count < 4:
        raise Lfm2SemanticShortConvRefoldError(
            "At least four unique batches are required for isolated four-way gates."
        )
    counts = [max(1, count // 2), max(1, (3 * count) // 16), max(1, count // 8), 1]
    while sum(counts) > count:
        for index in (0, 1, 2):
            if counts[index] > 1 and sum(counts) > count:
                counts[index] -= 1
    counts[3] += count - sum(counts)
    cursor = 0
    result: dict[str, tuple[Lfm2SemanticShortConvReplayEvidence, ...]] = {}
    for partition, size in zip(_PARTITIONS, counts, strict=True):
        selected = batch_indices[cursor : cursor + size]
        result[partition] = tuple(
            item
            for batch_index in selected
            for item in sorted(
                grouped[batch_index],
                key=lambda value: (value.phase, value.evidence_sha256),
            )
        )
        cursor += size
    if cursor != count or any(not result[name] for name in _PARTITIONS):
        raise AssertionError("Internal short-conv partitioning failed.")
    return MappingProxyType(result)


def _balance_partition(
    evidence: Sequence[Lfm2SemanticShortConvReplayEvidence],
) -> tuple[tuple[_BalancedCase, ...], Mapping[str, Any]]:
    raw_mass: dict[int, float] = {}
    phase_count: dict[int, int] = {}
    for item in evidence:
        mass = float(item.endpoint_row_sqrt_weights.square().sum().item())
        if not math.isfinite(mass) or mass <= 0.0:
            raise Lfm2SemanticShortConvRefoldError(
                "Replay endpoint row weights have no positive mass."
            )
        raw_mass[item.batch_index] = raw_mass.get(item.batch_index, 0.0) + mass
        phase_count[item.batch_index] = phase_count.get(item.batch_index, 0) + 1
    cases = tuple(
        _BalancedCase(
            evidence=item,
            endpoint_row_weights=(
                item.endpoint_row_sqrt_weights.square() / raw_mass[item.batch_index]
            ),
        )
        for item in evidence
    )
    normalized_mass = {
        batch_index: math.fsum(
            float(case.endpoint_row_weights.sum().item())
            for case in cases
            if case.evidence.batch_index == batch_index
        )
        for batch_index in raw_mass
    }
    if any(
        not math.isclose(value, 1.0, rel_tol=1e-10, abs_tol=1e-12)
        for value in normalized_mass.values()
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Trajectory balancing failed to produce unit mass per trajectory."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-short-conv-trajectory-balance-v1",
        "rule": "normalize_each_trajectory_endpoint_weight_mass_to_one",
        "groups": [
            {
                "batch_index": batch_index,
                "phase_count": phase_count[batch_index],
                "raw_squared_row_weight_mass": raw_mass[batch_index],
                "normalized_row_weight_mass": normalized_mass[batch_index],
            }
            for batch_index in sorted(raw_mass)
        ],
        "longer_trajectories_receive_more_total_weight": False,
    }
    return cases, MappingProxyType({**body, "balance_sha256": canonical_sha256(body)})


def _apply_affine(inputs: Any, weight: Any, bias: Any | None) -> Any:
    result = inputs @ weight.T
    return result if bias is None else result + bias


def _operator_context(case: _BalancedCase, state: Mapping[str, Any]) -> Any:
    torch = _torch()
    evidence = case.evidence
    inputs = evidence.actual_child_operator_input_history.to(torch.float64)
    in_weight = state["in_proj.weight"].to(torch.float64)
    in_bias = state.get("in_proj.bias")
    if in_bias is not None:
        in_bias = in_bias.to(torch.float64)
    projected = _apply_affine(inputs, in_weight, in_bias)
    b_value, c_value, x_value = projected.chunk(3, dim=-1)
    kernel = state["conv.weight"].to(torch.float64)[:, 0, :]
    conv_bias = state.get("conv.bias")
    if conv_bias is not None:
        conv_bias = conv_bias.to(torch.float64)
    batch, sequence, hidden = map(int, b_value.shape)
    fir_length = int(kernel.shape[1])
    gated = (b_value * x_value).transpose(1, 2).contiguous()
    history = torch.nn.functional.pad(gated, (fir_length - 1, 0))
    filtered = torch.nn.functional.conv1d(
        history,
        kernel.reshape(hidden, 1, fir_length),
        bias=conv_bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=hidden,
    )
    if tuple(filtered.shape) != (batch, hidden, sequence):
        raise Lfm2SemanticShortConvRefoldError(
            "Candidate FIR replay did not preserve the child history length."
        )
    return (c_value * filtered.transpose(1, 2)).contiguous()


def _endpoint_prediction(case: _BalancedCase, state: Mapping[str, Any]) -> Any:
    torch = _torch()
    evidence = case.evidence
    positions = torch.tensor(evidence.endpoint_target_positions, dtype=torch.long)
    context = _operator_context(case, state).index_select(1, positions)
    out_weight = state["out_proj.weight"].to(torch.float64)
    out_bias = state.get("out_proj.bias")
    if out_bias is not None:
        out_bias = out_bias.to(torch.float64)
    branch = _apply_affine(context, out_weight, out_bias)
    actual_pre = evidence.actual_child_pre_operator_residual_history.to(
        torch.float64
    ).index_select(1, positions)
    return (actual_pre + branch).contiguous()


def _endpoint_metrics(
    cases: Sequence[_BalancedCase],
    state: Mapping[str, Any],
) -> Lfm2SemanticShortConvEndpointMetrics:
    torch = _torch()
    selected = tuple(cases)
    if not selected:
        raise Lfm2SemanticShortConvRefoldError(
            "Endpoint metrics require a nonempty partition."
        )
    squared_error = torch.zeros((), dtype=torch.float64)
    squared_target = torch.zeros((), dtype=torch.float64)
    weighted_elements = torch.zeros((), dtype=torch.float64)
    weighted_cosine = torch.zeros((), dtype=torch.float64)
    cosine_mass = torch.zeros((), dtype=torch.float64)
    maximum = 0.0
    worst_cosine = 1.0
    row_count = 0
    for case in selected:
        target = case.evidence.parent_post_operator_residual_target_space.to(
            torch.float64
        )
        prediction = _endpoint_prediction(case, state)
        difference = prediction - target
        weights = case.endpoint_row_weights
        expanded = weights.unsqueeze(-1)
        squared_error += (difference.square() * expanded).sum()
        squared_target += (target.square() * expanded).sum()
        weighted_elements += expanded.sum() * int(target.shape[-1])
        maximum = max(maximum, float(difference.abs().max().item()))
        prediction_norm = torch.linalg.vector_norm(prediction, dim=-1)
        target_norm = torch.linalg.vector_norm(target, dim=-1)
        denominator = prediction_norm * target_norm
        both_zero = (prediction_norm == 0.0) & (target_norm == 0.0)
        cosine = torch.where(
            both_zero,
            torch.ones_like(denominator),
            torch.where(
                denominator > torch.finfo(torch.float64).tiny,
                (prediction * target).sum(dim=-1)
                / denominator.clamp_min(torch.finfo(torch.float64).tiny),
                torch.zeros_like(denominator),
            ),
        ).clamp(-1.0, 1.0)
        weighted_cosine += (cosine * weights).sum()
        cosine_mass += weights.sum()
        worst_cosine = min(worst_cosine, float(cosine.min().item()))
        row_count += int(weights.numel())
    denominator = weighted_elements.clamp_min(torch.finfo(torch.float64).tiny)
    rmse = torch.sqrt(squared_error / denominator)
    target_scale = torch.sqrt(squared_target / denominator).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return Lfm2SemanticShortConvEndpointMetrics(
        case_count=len(selected),
        trajectory_count=len({case.evidence.batch_index for case in selected}),
        row_count=row_count,
        endpoint_rmse=float(rmse.item()),
        endpoint_relative_rmse=float((rmse / target_scale).item()),
        endpoint_maximum_absolute_error=maximum,
        endpoint_cosine_similarity_mean=float((weighted_cosine / cosine_mass).item()),
        endpoint_cosine_similarity_worst=worst_cosine,
    )


def _nonregressing(
    candidate: Lfm2SemanticShortConvEndpointMetrics,
    baseline: Lfm2SemanticShortConvEndpointMetrics,
    *,
    tolerance: float,
) -> bool:
    return (
        candidate.endpoint_relative_rmse <= baseline.endpoint_relative_rmse + tolerance
        and candidate.endpoint_maximum_absolute_error
        <= baseline.endpoint_maximum_absolute_error + tolerance
        and candidate.endpoint_cosine_similarity_mean + tolerance
        >= baseline.endpoint_cosine_similarity_mean
        and candidate.endpoint_cosine_similarity_worst + tolerance
        >= baseline.endpoint_cosine_similarity_worst
    )


def _strictly_improving(
    candidate: Lfm2SemanticShortConvEndpointMetrics,
    baseline: Lfm2SemanticShortConvEndpointMetrics,
    *,
    tolerance: float,
) -> bool:
    return _nonregressing(candidate, baseline, tolerance=tolerance) and (
        candidate.endpoint_relative_rmse < baseline.endpoint_relative_rmse - tolerance
        or candidate.endpoint_maximum_absolute_error
        < baseline.endpoint_maximum_absolute_error - tolerance
        or candidate.endpoint_cosine_similarity_mean
        > baseline.endpoint_cosine_similarity_mean + tolerance
        or candidate.endpoint_cosine_similarity_worst
        > baseline.endpoint_cosine_similarity_worst + tolerance
    )


def _structural_cases(
    fit: Sequence[Lfm2SemanticShortConvReplayEvidence],
    fit_check: Sequence[Lfm2SemanticShortConvReplayEvidence],
) -> tuple[Lfm2CpcfShortConvFoldCase, ...]:
    torch = _torch()
    result = []
    for split, values in (("fit", fit), ("validation", fit_check)):
        for evidence in values:
            owners = torch.tensor(
                evidence.target_to_parent_span_owner_positions,
                dtype=torch.long,
            )
            endpoints = torch.tensor(
                evidence.endpoint_target_positions,
                dtype=torch.long,
            )
            batch = int(evidence.actual_child_operator_input_history.shape[0])
            sequence = int(evidence.actual_child_operator_input_history.shape[1])
            valid = torch.zeros((batch, sequence), dtype=torch.bool)
            valid[:, list(evidence.semantic_span_boundary_positions)] = True
            result.append(
                Lfm2CpcfShortConvFoldCase(
                    case_id=(
                        f"batch-{evidence.batch_index}-phase-{evidence.phase}-"
                        f"{evidence.evidence_sha256[:16]}"
                    ),
                    split=split,
                    suffix_start=evidence.endpoint_target_positions[0],
                    actual_child_operator_input_history=(
                        evidence.actual_child_operator_input_history
                    ),
                    parent_b_history=evidence.parent_b_history.index_select(1, owners),
                    parent_c_history=evidence.parent_c_history.index_select(1, owners),
                    parent_x_history=evidence.parent_x_history.index_select(1, owners),
                    parent_post_operator_residual_target_space_suffix=(
                        evidence.parent_post_operator_residual_target_space
                    ),
                    actual_child_pre_operator_residual_suffix=(
                        evidence.actual_child_pre_operator_residual_history.index_select(
                            1,
                            endpoints,
                        )
                    ),
                    valid_prefix_mask=valid,
                )
            )
    return tuple(result)


def _compiled_state(compiled: Lfm2CompiledCpcfShortConvFold) -> Mapping[str, Any]:
    return compiled.state_dict()


def _compile_structural_candidate(
    *,
    source_fir_weight: Any,
    source_fir_bias: Any | None,
    incumbent_state: Mapping[str, Any],
    fit: Sequence[Lfm2SemanticShortConvReplayEvidence],
    fit_check: Sequence[Lfm2SemanticShortConvReplayEvidence],
    fold_compute_device: str = "cpu",
) -> _Candidate:
    compiled = compile_lfm2_cpcf_short_conv_fold(
        source_fir_weight=source_fir_weight,
        source_fir_bias=source_fir_bias,
        target_in_proj_weight_template=incumbent_state["in_proj.weight"],
        target_in_proj_bias_template=incumbent_state.get("in_proj.bias"),
        target_conv_weight_template=incumbent_state["conv.weight"],
        target_conv_bias_template=incumbent_state.get("conv.bias"),
        target_out_proj_weight_template=incumbent_state["out_proj.weight"],
        target_out_proj_bias_template=incumbent_state.get("out_proj.bias"),
        cases=_structural_cases(fit, fit_check),
        ridge=1e-8,
        gate_policy=Lfm2CpcfShortConvGatePolicy(enforce=False),
        **(
            {"compute_device": fold_compute_device}
            if fold_compute_device != "cpu"
            else {}
        ),
    )
    if (
        not isinstance(compiled, Lfm2CompiledCpcfShortConvFold)
        or not compiled.full_source_coverage
        or compiled.report.get("execution", {}).get("training_steps") != 0
        or compiled.report.get("provenance", {}).get("source_channels_dropped") != 0
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Structural compiler returned no complete full-source candidate."
        )
    return _Candidate(
        name="full-source-structural",
        state=_compiled_state(compiled),
        provenance={
            "structural_compiler_schema": compiled.report["schema"],
            "structural_compiler_report_sha256": compiled.report["report_sha256"],
            "structural_compiler_report": compiled.as_dict(),
            "source_hidden_size": compiled.source_hidden_size,
            "target_hidden_size": compiled.target_hidden_size,
            "source_fir_length": compiled.source_fir_length,
            "target_fir_length": compiled.target_fir_length,
            "full_source_coverage": compiled.full_source_coverage,
            "source_channels_selected": 0,
            "source_channels_dropped": 0,
            "source_channels_pruned": 0,
            "fit_check_used_for_fitting": False,
            "selection_used_for_fitting": False,
            "validation_used_for_fitting": False,
        },
    )


def _writer_design(
    cases: Sequence[_BalancedCase],
    producer_state: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    torch = _torch()
    contexts = []
    targets = []
    weights = []
    for case in cases:
        evidence = case.evidence
        positions = torch.tensor(evidence.endpoint_target_positions, dtype=torch.long)
        contexts.append(
            _operator_context(case, producer_state)
            .index_select(1, positions)
            .reshape(-1, int(producer_state["out_proj.weight"].shape[1]))
        )
        pre = evidence.actual_child_pre_operator_residual_history.to(
            torch.float64
        ).index_select(1, positions)
        targets.append(
            (
                evidence.parent_post_operator_residual_target_space.to(torch.float64)
                - pre
            ).reshape(-1, int(producer_state["out_proj.weight"].shape[0]))
        )
        weights.append(case.endpoint_row_weights.reshape(-1))
    return (
        torch.cat(contexts, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(weights, dim=0),
    )


def _fit_metric_ordering_key(
    metrics: Lfm2SemanticShortConvEndpointMetrics | None,
    *deterministic_tiebreakers: Any,
) -> tuple[Any, ...]:
    if metrics is None:
        return (1, 0.0, 0.0, 0.0, 0.0, *deterministic_tiebreakers)
    return (
        0,
        metrics.endpoint_relative_rmse,
        metrics.endpoint_maximum_absolute_error,
        -metrics.endpoint_cosine_similarity_mean,
        -metrics.endpoint_cosine_similarity_worst,
        *deterministic_tiebreakers,
    )


def _diagnostic_endpoint_metrics(
    cases: Sequence[_BalancedCase],
    state: Mapping[str, Any],
) -> tuple[Lfm2SemanticShortConvEndpointMetrics | None, str | None]:
    try:
        return _endpoint_metrics(cases, state), None
    except Lfm2SemanticShortConvRefoldError as exc:
        return None, str(exc)


def _compile_writer_candidate(
    fit_cases: Sequence[_BalancedCase],
    incumbent_state: Mapping[str, Any],
    *,
    fixed_producer_state: Mapping[str, Any] | None = None,
    candidate_name: str = "incumbent-producer-output-writer",
    fixed_producer_provenance: Mapping[str, Any] | None = None,
    fold_compute_device: str | None = None,
    body_recipe: Lfm2SemanticBodyRecipe | None = None,
) -> tuple[_Candidate | None, Mapping[str, Any]]:
    torch = _torch()
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    producer_state = (
        incumbent_state if fixed_producer_state is None else fixed_producer_state
    )
    producer_source = (
        "live-incumbent" if fixed_producer_state is None else "full-source-structural"
    )
    contexts, targets, weights = _writer_design(fit_cases, producer_state)
    fit_context_sha = canonical_sha256(
        {
            "case_sha256": [case.case_sha256 for case in fit_cases],
            "contexts_sha256": _tensor_sha256(contexts),
            "targets_sha256": _tensor_sha256(targets),
            "weights_sha256": _tensor_sha256(weights),
        }
    )
    has_bias = "out_proj.bias" in producer_state
    design = contexts
    if has_bias:
        design = torch.cat(
            (design, torch.ones((design.shape[0], 1), dtype=torch.float64)),
            dim=1,
        )
    incumbent_coefficients = producer_state["out_proj.weight"].to(torch.float64).T
    if has_bias:
        incumbent_coefficients = torch.cat(
            (
                incumbent_coefficients,
                producer_state["out_proj.bias"].to(torch.float64).reshape(1, -1),
            ),
            dim=0,
        )
    writer_compute_execution: dict[str, Any]
    with _writer_compute_context(torch, fold_compute_device) as (
        device,
        writer_compute_execution,
    ):
        x = design.to(device=device, dtype=torch.float64)
        y = targets.to(device=device, dtype=torch.float64)
        root = torch.sqrt(weights.to(device=device, dtype=torch.float64)).unsqueeze(1)
        incumbent = incumbent_coefficients.to(device=device)
        weighted_x = x * root
        residual = (y - x @ incumbent) * root
        gram = weighted_x.T @ weighted_x
        ridge = 1e-4
        scale = float(gram.diagonal().mean().clamp_min(1.0).item())
        effective_ridge = ridge * scale
        identity = torch.eye(int(gram.shape[0]), dtype=torch.float64, device=device)
        with torch.inference_mode():
            delta = torch.linalg.solve(
                gram + effective_ridge * identity,
                weighted_x.T @ residual,
            )
    if not bool(torch.isfinite(delta).all().item()):
        raise Lfm2SemanticShortConvRefoldError(
            "Closed-form short-conv output-writer solve produced nonfinite values."
        )
    baseline = _endpoint_metrics(fit_cases, producer_state)
    frontier: list[
        tuple[
            tuple[Any, ...],
            float,
            Mapping[str, Any],
            Lfm2SemanticShortConvEndpointMetrics | None,
            Mapping[str, Any],
        ]
    ] = []
    frontier_reports: list[dict[str, Any]] = []
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        coefficients = (incumbent + alpha * delta).to(device="cpu")
        state = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in producer_state.items()
        }
        state["out_proj.weight"] = (
            coefficients[: contexts.shape[1], :]
            .T.to(dtype=producer_state["out_proj.weight"].dtype)
            .contiguous()
        )
        if has_bias:
            state["out_proj.bias"] = (
                coefficients[-1, :]
                .to(dtype=producer_state["out_proj.bias"].dtype)
                .contiguous()
            )
        structural = _candidate_state_structural_report(state, incumbent_state)
        metrics: Lfm2SemanticShortConvEndpointMetrics | None = None
        metrics_error: str | None = None
        magnitude: Mapping[str, Any] | None = None
        ordering_key: tuple[Any, ...] | None = None
        if structural["structural_safety_passed"] is True:
            magnitude = _candidate_relative_magnitude_diagnostics(
                state,
                incumbent_state,
            )
        if structural["structurally_eligible"] is True:
            metrics, metrics_error = _diagnostic_endpoint_metrics(fit_cases, state)
            ordering_key = _fit_metric_ordering_key(
                metrics,
                alpha,
                structural["state_tensor_bundle_sha256"],
            )
        fit_improved = bool(
            metrics is not None
            and metrics.endpoint_relative_rmse < baseline.endpoint_relative_rmse - 1e-8
        )
        fit_nonregression = bool(
            metrics is not None and _nonregressing(metrics, baseline, tolerance=1e-8)
        )
        record = {
            "alpha": alpha,
            "exact_target_dtype_materialization": True,
            "materialization_dtype": str(
                producer_state["out_proj.weight"].dtype
            ).removeprefix("torch."),
            "exact_bfloat16_materialization": (
                producer_state["out_proj.weight"].dtype == torch.bfloat16
            ),
            "candidate_state_tensor_bundle_sha256": structural[
                "state_tensor_bundle_sha256"
            ],
            "structural_report": dict(structural),
            "structural_report_sha256": structural["structural_report_sha256"],
            "structural_safety_passed": structural["structural_safety_passed"],
            "effectful_nonincumbent": structural["effectful_nonincumbent"],
            "structurally_eligible": structural["structurally_eligible"],
            "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
            "fit_metrics": None if metrics is None else metrics.as_dict(),
            "fit_metrics_error": metrics_error,
            "fit_improved_diagnostic": fit_improved,
            "fit_nonregression_diagnostic_passed": fit_nonregression,
            "fit_ordering_key": (None if ordering_key is None else list(ordering_key)),
            "relative_magnitude_diagnostics": (
                None if magnitude is None else dict(magnitude)
            ),
            "fit_thresholds_used_for_eligibility": False,
            "fit_nonregression_used_for_eligibility": False,
            "relative_magnitude_threshold_used_for_eligibility": False,
            "fit_metrics_used_for_candidate_ordering": (
                recipe.fit_metrics_order_candidates
            ),
            "fit_metrics_have_final_intelligence_authority": False,
        }
        frontier_reports.append(record)
        if ordering_key is not None:
            frontier.append((ordering_key, alpha, state, metrics, record))
    base_report = {
        "solver": (
            "closed_form_trajectory_weighted_residual_ridge_bfloat16_frontier"
            if producer_state["out_proj.weight"].dtype == torch.bfloat16
            else "closed_form_trajectory_weighted_residual_ridge_float32_frontier"
        ),
        "ridge": ridge,
        "effective_ridge": effective_ridge,
        "fit_context_sha256": fit_context_sha,
        "baseline_fit_metrics": baseline.as_dict(),
        "fit_only": True,
        "fixed_producer_state_source": producer_source,
        "fixed_producer_state_sha256": canonical_sha256(
            {
                name: _tensor_sha256(value)
                for name, value in sorted(producer_state.items())
            }
        ),
        "live_incumbent_producers_fixed": fixed_producer_state is None,
        "full_source_structural_producers_fixed": fixed_producer_state is not None,
        "incumbent_in_proj_fixed": fixed_producer_state is None,
        "incumbent_conv_fixed": fixed_producer_state is None,
        "full_source_coverage": (
            fixed_producer_provenance.get("full_source_coverage") is True
            if isinstance(fixed_producer_provenance, Mapping)
            else False
        ),
        "source_channels_dropped": (
            fixed_producer_provenance.get("source_channels_dropped")
            if isinstance(fixed_producer_provenance, Mapping)
            else None
        ),
        "structural_compiler_report": (
            dict(fixed_producer_provenance["structural_compiler_report"])
            if isinstance(fixed_producer_provenance, Mapping)
            and isinstance(
                fixed_producer_provenance.get("structural_compiler_report"),
                Mapping,
            )
            else None
        ),
        "selection_used_for_solve": False,
        "validation_used_for_solve": False,
        "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
        "structural_safety_is_only_candidate_veto": True,
        "fit_metrics_role": (
            "ordering-among-structurally-eligible-not-threshold-gating"
            if recipe.fit_metrics_order_candidates
            else "diagnostic-only"
        ),
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates
        ),
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_thresholds_used_for_candidate_eligibility": False,
        "fit_nonregression_used_for_candidate_eligibility": False,
        "relative_magnitude_metrics_diagnostic_only": True,
        "numeric_metrics_have_final_intelligence_authority": False,
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["short_conv"],
        "candidate_frontier": frontier_reports,
        "candidate_frontier_count": len(frontier_reports),
        "structurally_eligible_candidate_count": len(frontier),
        "operation_audit": dict(_ZERO_OPERATIONS),
        **(
            {"compute_execution": dict(writer_compute_execution)}
            if writer_compute_execution
            else {}
        ),
    }
    if not frontier:
        return None, MappingProxyType(
            {
                **base_report,
                "status": "no-structurally-eligible-effectful-writer-proposal",
                "selected_alpha": None,
                "selected_fit_ordering_key": None,
            }
        )
    if not recipe.fit_metrics_order_candidates:
        ordering_key, alpha, state, metrics, selected_record = max(
            frontier,
            key=lambda value: (
                value[1],
                value[4]["candidate_state_tensor_bundle_sha256"],
            ),
        )
        selection_status = "recipe-alpha-one-or-strongest-safe-writer-proposal"
    else:
        ordering_key, alpha, state, metrics, selected_record = min(
            frontier,
            key=lambda value: value[0],
        )
        selection_status = "fit-ordered-structurally-eligible-proposal"
    report = {
        **base_report,
        "status": selection_status,
        "selected_alpha": alpha,
        "selected_fit_ordering_key": list(ordering_key),
        "candidate_fit_metrics": None if metrics is None else metrics.as_dict(),
        "candidate_fit_metrics_error": selected_record["fit_metrics_error"],
        "selected_structural_report_sha256": selected_record[
            "structural_report_sha256"
        ],
        "selected_relative_magnitude_diagnostics": selected_record[
            "relative_magnitude_diagnostics"
        ],
    }
    return (
        _Candidate(
            name=candidate_name,
            state=state,
            provenance=report,
        ),
        MappingProxyType(report),
    )


@contextmanager
def _fhvr_compute_device_context(
    fold_compute_device: str,
) -> Any:
    """Hold an explicit CUDA index while the existing FHVR kernel runs."""

    torch = _torch()
    if fold_compute_device == "cpu":
        # The FHVR kernel's deterministic scope calls
        # torch.use_deterministic_algorithms even for CPU work.  That flag is
        # process-global, so serialize the complete kernel invocation just as
        # the CUDA route does instead of exposing a transient global policy to
        # concurrent folds.
        with _SHORT_CONV_TORCH_STATE_LOCK:
            yield {
                "backend": "cpu",
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "device_index": None,
            }
        return
    device, device_index = _indexed_cuda_device(torch, fold_compute_device)
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise Lfm2SemanticShortConvRefoldError(
            "Deterministic CUDA FHVR folding requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8."
        )
    properties = torch.cuda.get_device_properties(device)
    with _SHORT_CONV_TORCH_STATE_LOCK, torch.cuda.device(device):
        yield {
            "backend": "cuda",
            "requested_device": fold_compute_device,
            "resolved_device": str(device),
            "device_index": device_index,
            "device_name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "cublas_workspace_config": workspace_config,
        }


def _fhvr_fit_cases(
    evidence: Sequence[Lfm2SemanticShortConvReplayEvidence],
) -> tuple[Any, ...]:
    torch = _torch()
    from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
        Lfm2CpcfShortConvV4FoldCase,
    )

    result = []
    for item in evidence:
        if item.parent_operator_input_history is None:
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR construction requires authenticated parent operator-input "
                "history for every FIT case."
            )
        owners = torch.tensor(
            item.target_to_parent_span_owner_positions,
            dtype=torch.long,
        )
        endpoints = torch.tensor(item.endpoint_target_positions, dtype=torch.long)
        batch = int(item.actual_child_operator_input_history.shape[0])
        sequence = int(item.actual_child_operator_input_history.shape[1])
        valid = torch.zeros((batch, sequence), dtype=torch.bool)
        valid[:, list(item.semantic_span_boundary_positions)] = True
        result.append(
            Lfm2CpcfShortConvV4FoldCase(
                case_id=(
                    f"fhvr-batch-{item.batch_index}-phase-{item.phase}-"
                    f"{item.evidence_sha256[:16]}"
                ),
                split="fit",
                suffix_start=item.endpoint_target_positions[0],
                actual_child_operator_input_history=(
                    item.actual_child_operator_input_history
                ),
                parent_operator_input_history=(
                    item.parent_operator_input_history.index_select(1, owners)
                ),
                parent_b_history=item.parent_b_history.index_select(1, owners),
                parent_c_history=item.parent_c_history.index_select(1, owners),
                parent_x_history=item.parent_x_history.index_select(1, owners),
                parent_post_operator_residual_target_space_suffix=(
                    item.parent_post_operator_residual_target_space
                ),
                actual_child_pre_operator_residual_suffix=(
                    item.actual_child_pre_operator_residual_history.index_select(
                        1,
                        endpoints,
                    )
                ),
                valid_prefix_mask=valid,
            )
        )
    return tuple(result)


def _fhvr_parent_contexts(
    evidence: Sequence[Lfm2SemanticShortConvReplayEvidence],
    fit_cases: Sequence[Any],
    *,
    source: Lfm2SemanticShortConvFhvrSource,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Replay the true donor FIR before aligning its outputs to child rows.

    Token expansion duplicates donor rows in the target timeline.  A causal
    convolution does not commute with that duplication, so B/C/x must remain
    on the complete donor timeline until after the parent context is replayed.
    The resulting contexts may then be owner-selected into target positions;
    the case mask admits exactly one boundary row for every donor token.
    """

    torch = _torch()
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import causal_depthwise_context

    items = tuple(evidence)
    cases = tuple(fit_cases)
    if (
        not source.verify()
        or len(items) != len(cases)
        or any(
            case.case_id
            != (
                f"fhvr-batch-{item.batch_index}-phase-{item.phase}-"
                f"{item.evidence_sha256[:16]}"
            )
            for item, case in zip(items, cases, strict=True)
        )
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR donor histories and target-aligned fit cases are reordered."
        )
    contexts: dict[str, Any] = {}
    case_receipts: dict[str, Any] = {}
    for item, case in zip(items, cases, strict=True):
        owners = torch.tensor(
            item.target_to_parent_span_owner_positions,
            dtype=torch.long,
        )
        donor_context = causal_depthwise_context(
            item.parent_b_history,
            item.parent_c_history,
            item.parent_x_history,
            source.source_fir_weight,
            source.source_fir_bias,
        )
        aligned_context = donor_context.index_select(1, owners).contiguous()
        boundary_count = len(item.semantic_span_boundary_positions)
        if (
            tuple(donor_context.shape) != tuple(item.parent_b_history.shape)
            or tuple(aligned_context.shape[:2])
            != tuple(case.actual_child_operator_input_history.shape[:2])
            or int(aligned_context.shape[-1]) != source.source_hidden_size
            or boundary_count != int(item.parent_b_history.shape[1])
            or int(case.valid_prefix_mask.sum().item())
            != int(item.parent_b_history.shape[0]) * boundary_count
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR donor-first causal context lost full-history alignment."
            )
        contexts[case.case_id] = aligned_context
        case_receipts[case.case_id] = {
            "evidence_sha256": item.evidence_sha256,
            "case_authentication_sha256": case.authentication_sha256,
            "donor_history_length": int(item.parent_b_history.shape[1]),
            "target_history_length": int(aligned_context.shape[1]),
            "semantic_boundary_count": boundary_count,
            "inserted_target_proxy_count": len(item.inserted_target_positions),
            "donor_context_sha256": _tensor_sha256(donor_context),
            "target_aligned_context_sha256": _tensor_sha256(aligned_context),
        }
    receipt_body = {
        "schema": "neural-forge-lfm2-short-conv-fhvr-parent-context-alignment-v1",
        "source_sha256": source.source_sha256,
        "formula": (
            "owner_select(causal_depthwise_context(parent_b_donor_history,"
            "parent_c_donor_history,parent_x_donor_history,source_fir))"
        ),
        "causal_fir_replayed_before_target_owner_expansion": True,
        "complete_donor_causal_history_consumed": True,
        "inserted_target_rows_are_alignment_proxies_not_parent_timesteps": True,
        "valid_mask_admits_each_donor_semantic_boundary_once": True,
        "case_receipts": case_receipts,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    receipt = {
        **receipt_body,
        "result_sha256": canonical_sha256(receipt_body),
    }
    return MappingProxyType(contexts), MappingProxyType(receipt)


def _compile_fhvr_full_coverage_candidate(
    *,
    construction: Lfm2SemanticShortConvConstruction,
    source: Lfm2SemanticShortConvFhvrSource,
    fit: Sequence[Lfm2SemanticShortConvReplayEvidence],
    incumbent_state: Mapping[str, Any],
    writer_anchor_candidate: _Candidate,
    fold_compute_device: str,
) -> tuple[_Candidate, Mapping[str, Any]]:
    """Construct the existing FHVR hybrid and adapt it to semantic state names."""

    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
        Lfm2CpcfShortConvNumericalFrontierRequest,
        Lfm2CpcfShortConvNumericalRawCandidate,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v5_fhvr_mf import (
        FHVR_MF_CAP10K_SWEEPS1_CANDIDATE_ID,
        FHVR_MF_CAP10K_SWEEPS1_FULL_COVERAGE_HYBRID_ID,
        FHVR_MF_FULL_COVERAGE_HYBRID_KIND,
        Lfm2CpcfFhvrMfPolicy,
        build_lfm2_cpcf_fhvr_mf_candidate,
        build_lfm2_cpcf_fhvr_writer_full_coverage_hybrid_candidate,
    )

    if not construction.uses_fhvr_full_coverage or not source.verify():
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR construction or source authentication is invalid."
        )
    fit_cases = _fhvr_fit_cases(fit)
    parent_contexts, parent_context_alignment = _fhvr_parent_contexts(
        fit,
        fit_cases,
        source=source,
    )
    target_dtype = str(incumbent_state["in_proj.weight"].dtype).removeprefix("torch.")
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(target_dtype=target_dtype)
    request = Lfm2CpcfShortConvNumericalFrontierRequest(
        source_tensors={
            "source_in_proj_weight": source.source_in_proj_weight,
            "source_in_proj_bias": source.source_in_proj_bias,
            "source_fir_kernel": source.source_fir_weight,
            "source_fir_bias": source.source_fir_bias,
            "source_out_proj_weight": source.source_out_proj_weight,
            "source_out_proj_bias": source.source_out_proj_bias,
            "output_encoder": source.output_encoder,
        },
        cases=fit_cases,
        parent_contexts=parent_contexts,
        target_templates={
            "in_proj_weight": incumbent_state["in_proj.weight"],
            "in_proj_bias": incumbent_state.get("in_proj.bias"),
            "fir_kernel": incumbent_state["conv.weight"][:, 0, :],
            "fir_bias": incumbent_state.get("conv.bias"),
            "writer_weight": incumbent_state["out_proj.weight"],
            "writer_bias": incumbent_state.get("out_proj.bias"),
        },
        policy=solver_policy,
    )
    if writer_anchor_candidate.name not in {
        "incumbent-producer-output-writer",
        "full-source-producer-output-writer",
    }:
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR full coverage requires the compiled recipe writer candidate."
        )
    writer_state = writer_anchor_candidate.state
    raw_writer = Lfm2CpcfShortConvNumericalRawCandidate(
        request_sha256=request.request_sha256,
        candidate_id="semantic-full-source-residual-writer",
        candidate_kind="accepted_fit_projection_path_anchor",
        in_proj_weight=writer_state["in_proj.weight"],
        in_proj_bias=writer_state.get("in_proj.bias"),
        fir_kernel=writer_state["conv.weight"][:, 0, :],
        fir_bias=writer_state.get("conv.bias"),
        writer_weight=writer_state["out_proj.weight"],
        writer_bias=writer_state.get("out_proj.bias"),
        method_provenance={
            "method": "semantic_full_source_residual_writer_anchor",
            "request_sha256": request.request_sha256,
            "semantic_writer_candidate_name": writer_anchor_candidate.name,
            "semantic_candidate_state_sha256": (
                writer_anchor_candidate.state_sha256
            ),
            "semantic_candidate_provenance_sha256": canonical_sha256(
                dict(writer_anchor_candidate.provenance)
            ),
            "writer_selected_alpha": writer_anchor_candidate.provenance.get(
                "selected_alpha"
            ),
            "recipe_component_lane": writer_anchor_candidate.provenance.get(
                "recipe_component_lane"
            ),
            "numeric_metrics_used_to_order_writer_frontier": bool(
                writer_anchor_candidate.provenance.get(
                    "fit_metrics_used_for_candidate_ordering"
                )
            ),
            "numeric_thresholds_used_to_admit_writer_anchor": False,
            "external_construction_choice_is_final_selection_authority": True,
            "fit_only": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "execution": dict(_ZERO_OPERATIONS),
        },
    )
    backend = "cpu" if fold_compute_device == "cpu" else "cuda"
    fhvr_policy = Lfm2CpcfFhvrMfPolicy(
        condition_cap=10_000.0,
        factor_sweeps=1,
        backend=backend,
    )
    with _fhvr_compute_device_context(fold_compute_device) as device_receipt:
        pure = build_lfm2_cpcf_fhvr_mf_candidate(
            request,
            context_encoder=source.output_encoder,
            policy=fhvr_policy,
            candidate_id=FHVR_MF_CAP10K_SWEEPS1_CANDIDATE_ID,
        )
        hybrid = build_lfm2_cpcf_fhvr_writer_full_coverage_hybrid_candidate(
            request,
            context_encoder=source.output_encoder,
            writer_candidate=raw_writer,
            fhvr_candidate=pure,
            policy=fhvr_policy,
            candidate_id=(FHVR_MF_CAP10K_SWEEPS1_FULL_COVERAGE_HYBRID_ID),
        )
    if (
        not request.verify_authentication()
        or not raw_writer.verify_authentication()
        or not pure.verify_authentication()
        or not hybrid.verify_authentication()
        or hybrid.candidate_kind != FHVR_MF_FULL_COVERAGE_HYBRID_KIND
        or hybrid.request_sha256 != request.request_sha256
        or hybrid.method_provenance.get("source_writer_candidate_state_sha256")
        != raw_writer.state_sha256
        or hybrid.method_provenance.get("source_fhvr_candidate_state_sha256")
        != pure.state_sha256
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR full-coverage hybrid lost request or source authentication."
        )
    raw_state = hybrid.state_dict()
    target_state = {
        "in_proj.weight": raw_state["in_proj_weight"],
        "conv.weight": raw_state["fir_kernel"].unsqueeze(1),
        "out_proj.weight": raw_state["writer_weight"],
    }
    for local_name, raw_name in (
        ("in_proj.bias", "in_proj_bias"),
        ("conv.bias", "fir_bias"),
        ("out_proj.bias", "writer_bias"),
    ):
        if raw_state[raw_name] is not None:
            target_state[local_name] = raw_state[raw_name]
    if set(target_state) != set(incumbent_state):
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR hybrid differs from the existing six-role short-conv topology."
        )
    materialized = {
        name: value.detach()
        .to(device="cpu", dtype=incumbent_state[name].dtype)
        .contiguous()
        .clone()
        for name, value in target_state.items()
    }
    materialized_target_state_sha256 = canonical_sha256(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(materialized.items())
        }
    )
    construction_document = construction.as_dict()
    source_document = source.as_dict()
    request_document = request.as_dict()
    raw_to_semantic_role_mapping = {
        "in_proj_weight": {
            "semantic_role": "in_proj.weight",
            "transform": "identity",
        },
        "in_proj_bias": {
            "semantic_role": "in_proj.bias",
            "transform": "identity_or_absent",
        },
        "fir_kernel": {
            "semantic_role": "conv.weight",
            "transform": "unsqueeze_depthwise_axis_1",
        },
        "fir_bias": {
            "semantic_role": "conv.bias",
            "transform": "identity_or_absent",
        },
        "writer_weight": {
            "semantic_role": "out_proj.weight",
            "transform": "identity",
        },
        "writer_bias": {
            "semantic_role": "out_proj.bias",
            "transform": "identity_or_absent",
        },
    }
    target_dtype_conversion = {}
    for raw_role, mapping in raw_to_semantic_role_mapping.items():
        semantic_role = mapping["semantic_role"]
        raw_value = raw_state[raw_role]
        semantic_value = materialized.get(semantic_role)
        target_dtype_conversion[raw_role] = {
            **mapping,
            "raw_tensor_sha256": (
                None if raw_value is None else _tensor_sha256(raw_value)
            ),
            "raw_dtype": (
                None
                if raw_value is None
                else str(raw_value.dtype).removeprefix("torch.")
            ),
            "target_dtype": (
                None
                if semantic_value is None
                else str(semantic_value.dtype).removeprefix("torch.")
            ),
            "materialized_tensor_sha256": (
                None if semantic_value is None else _tensor_sha256(semantic_value)
            ),
        }
    source_role_map = {
        "source_in_proj_weight": "in_proj_weight",
        "source_in_proj_bias": "in_proj_bias",
        "source_fir_kernel": "fir_weight",
        "source_fir_bias": "fir_bias",
        "source_out_proj_weight": "out_proj_weight",
        "source_out_proj_bias": "out_proj_bias",
        "output_encoder": "output_encoder",
    }
    source_to_request_tensor_inventory = {
        request_role: {
            "source_role": source_role,
            "source_tensor_sha256": source_document["tensor_sha256"][source_role],
            "numerical_request_tensor_sha256": request_document[
                "source_tensor_sha256"
            ][request_role],
        }
        for request_role, source_role in source_role_map.items()
    }
    if not hybrid.verify_authentication():
        raise Lfm2SemanticShortConvRefoldError(
            "Raw FHVR hybrid changed before semantic materialization was rooted."
        )
    materialization_body = {
        "schema": _FHVR_MATERIALIZATION_SCHEMA,
        "kind": "authenticated_raw_hybrid_to_semantic_dense_materialization",
        "construction_sha256": construction.construction_sha256,
        "source_sha256": source.source_sha256,
        "request_sha256": request.request_sha256,
        "hybrid_report_sha256": hybrid.report["report_sha256"],
        "hybrid_state_sha256": hybrid.state_sha256,
        "raw_hybrid_authentication_verified_at_materialization": True,
        "raw_to_semantic_role_mapping": raw_to_semantic_role_mapping,
        "source_to_request_tensor_inventory": source_to_request_tensor_inventory,
        "target_dtype_conversion": target_dtype_conversion,
        "materialized_state_tensor_sha256": {
            name: _tensor_sha256(value)
            for name, value in sorted(materialized.items())
        },
        "materialized_target_state_sha256": materialized_target_state_sha256,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    materialization_receipt = {
        **materialization_body,
        "report_sha256": canonical_sha256(materialization_body),
    }
    execution_body = {
        "schema": "neural-forge-lfm2-semantic-short-conv-fhvr-execution-v1",
        **device_receipt,
        "compute_dtype": hybrid.method_provenance["compute_dtype"],
        "fhvr_policy_sha256": fhvr_policy.authentication_sha256,
        "request_sha256": request.request_sha256,
        "pure_candidate_state_sha256": pure.state_sha256,
        "hybrid_candidate_state_sha256": hybrid.state_sha256,
        "canonical_input_authentication_device": "cpu",
        "canonical_result_materialization_device": "cpu",
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    execution = {
        **execution_body,
        "execution_sha256": canonical_sha256(execution_body),
    }
    candidate = _Candidate(
        name="full-history-fhvr-full-coverage-hybrid",
        state=materialized,
        provenance={
            "method": "semantic_full_history_fhvr_full_coverage_hybrid",
            "construction": construction_document,
            "construction_sha256": construction.construction_sha256,
            "source": source_document,
            "source_sha256": source.source_sha256,
            "numerical_frontier_request": request_document,
            "numerical_frontier_request_sha256": request.request_sha256,
            "parent_context_alignment": dict(parent_context_alignment),
            "parent_context_alignment_sha256": parent_context_alignment[
                "result_sha256"
            ],
            "writer_anchor_report": raw_writer.as_dict(),
            "writer_anchor_report_sha256": raw_writer.report["report_sha256"],
            "writer_anchor_uses_compiled_recipe_writer_candidate": True,
            "pure_fhvr_report": pure.as_dict(),
            "pure_fhvr_report_sha256": pure.report["report_sha256"],
            "hybrid_report": hybrid.as_dict(),
            "hybrid_report_sha256": hybrid.report["report_sha256"],
            "materialized_target_state_sha256": (
                materialized_target_state_sha256
            ),
            "fhvr_materialization_receipt": materialization_receipt,
            "fhvr_materialization_receipt_sha256": materialization_receipt[
                "report_sha256"
            ],
            "fhvr_compute_execution": execution,
            "fhvr_compute_execution_sha256": execution["execution_sha256"],
            "full_parent_causal_history_consumed": True,
            "future_observable_realization": True,
            "full_source_channel_coverage": True,
            "target_native_dense_weights": True,
            "selection_cases_consumed": 0,
            "validation_cases_consumed": 0,
            "fit_only": True,
            "operation_audit": dict(_ZERO_OPERATIONS),
        },
    )
    if not hybrid.verify_authentication():
        raise Lfm2SemanticShortConvRefoldError(
            "Raw FHVR hybrid changed before its validation root was sealed."
        )
    root_body = {
        "schema": _FHVR_VALIDATION_ROOT_SCHEMA,
        "kind": "fhvr_cold_start_validation_root",
        "construction": copy.deepcopy(construction_document),
        "construction_sha256": construction.construction_sha256,
        "source": copy.deepcopy(source_document),
        "source_sha256": source.source_sha256,
        "materialization_receipt": copy.deepcopy(materialization_receipt),
        "materialization_receipt_sha256": materialization_receipt["report_sha256"],
        "candidate_state_sha256": candidate.state_sha256,
        "raw_hybrid_authentication_verified_at_rooting": True,
    }
    validation_root = MappingProxyType(
        {**root_body, "root_sha256": canonical_sha256(root_body)}
    )
    return candidate, validation_root


def _candidate_report(
    candidate: _Candidate,
    *,
    incumbent_state: Mapping[str, Any],
    banks: Mapping[str, Sequence[_BalancedCase]],
    baseline: Mapping[str, Lfm2SemanticShortConvEndpointMetrics],
    tolerance: float,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> Mapping[str, Any]:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    structural = _candidate_state_structural_report(
        candidate.state,
        incumbent_state,
    )
    metrics: dict[str, Lfm2SemanticShortConvEndpointMetrics | None] = {}
    metric_errors: dict[str, str | None] = {}
    for name in ("fit", "fit_check", "selection"):
        if structural["structural_safety_passed"] is True:
            metrics[name], metric_errors[name] = _diagnostic_endpoint_metrics(
                banks[name],
                candidate.state,
            )
        else:
            metrics[name] = None
            metric_errors[name] = "candidate failed structural safety"
    checks = {
        name: (
            None
            if metrics[name] is None
            else _nonregressing(metrics[name], baseline[name], tolerance=tolerance)
        )
        for name in metrics
    }
    fit_improved = (
        metrics["fit"] is not None
        and metrics["fit"].endpoint_relative_rmse
        < baseline["fit"].endpoint_relative_rmse - tolerance
    )
    incumbent_noop = candidate.name == "incumbent-noop"
    selection_improved = bool(
        metrics["selection"] is not None
        and _strictly_improving(
            metrics["selection"],
            baseline["selection"],
            tolerance=tolerance,
        )
    )
    structurally_eligible = bool(
        not incumbent_noop and structural["structurally_eligible"] is True
    )
    simplicity = {
        "incumbent-producer-output-writer": 0,
        "full-source-producer-output-writer": 0,
        "full-source-structural": 1,
        "full-history-fhvr-full-coverage-hybrid": 2,
    }
    fit_ordering_key = (
        None
        if not structurally_eligible
        else _fit_metric_ordering_key(
            metrics["fit"],
            simplicity[candidate.name],
            candidate.state_sha256,
        )
    )
    magnitude = (
        _candidate_relative_magnitude_diagnostics(
            candidate.state,
            incumbent_state,
        )
        if structural["structural_safety_passed"] is True
        else None
    )
    numeric_diagnostic_miss = any(value is not True for value in checks.values())
    body = {
        "name": candidate.name,
        "candidate_state_sha256": candidate.state_sha256,
        "provenance": dict(candidate.provenance),
        "analytic_candidate": not incumbent_noop,
        "incumbent_noop_candidate": incumbent_noop,
        "fit_improved": fit_improved,
        "heldout_selection_improved": selection_improved,
        "partition_metrics": {
            name: None if metrics[name] is None else metrics[name].as_dict()
            for name in metrics
        },
        "partition_metric_errors": metric_errors,
        "partition_nonregression": checks,
        "structural_report": dict(structural),
        "structural_report_sha256": structural["structural_report_sha256"],
        "structural_safety_passed": structural["structural_safety_passed"],
        "effectful_nonincumbent": structural["effectful_nonincumbent"],
        "structurally_eligible": structurally_eligible,
        "eligible": structurally_eligible,
        "eligible_before_validation": structurally_eligible,
        "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
        "candidate_ranking_split": "fit",
        "fit_ordering_key": (
            None if fit_ordering_key is None else list(fit_ordering_key)
        ),
        "relative_magnitude_diagnostics": (
            None if magnitude is None else dict(magnitude)
        ),
        "fit_metric_nonregression_diagnostic_passed": checks["fit"],
        "fit_check_metric_nonregression_diagnostic_passed": checks["fit_check"],
        "selection_metric_nonregression_diagnostic_passed": checks["selection"],
        "numeric_diagnostic_miss": numeric_diagnostic_miss,
        "structural_safety_is_only_candidate_veto": True,
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates
        ),
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_thresholds_used_for_eligibility": False,
        "fit_check_thresholds_used_for_eligibility": False,
        "selection_thresholds_used_for_eligibility": False,
        "relative_magnitude_threshold_used_for_eligibility": False,
        "fit_check_and_selection_metrics_diagnostic_only": True,
        "numeric_diagnostic_miss_can_exclude_candidate": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "validation_inspected": False,
    }
    return MappingProxyType(
        {
            **body,
            "candidate_report_sha256": canonical_sha256(body),
        }
    )


def _select_candidate(
    candidates: Sequence[_Candidate],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
) -> _Candidate:
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    incumbent = tuple(
        candidate for candidate in candidates if candidate.name == "incumbent-noop"
    )
    if (
        len(incumbent) != 1
        or reports[incumbent[0].name]["structural_safety_passed"] is not True
        or reports[incumbent[0].name]["effectful_nonincumbent"] is not False
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "The authenticated live incumbent candidate is missing or invalid."
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.name != "incumbent-noop"
        and reports[candidate.name]["structurally_eligible"] is True
    ]
    if not eligible:
        return incumbent[0]

    by_name = {candidate.name: candidate for candidate in eligible}
    short_conv_lane = recipe.component_lanes["short_conv"]
    if short_conv_lane == "full_source_structural_priority":
        return by_name.get(
            "full-source-structural",
            by_name.get("incumbent-producer-output-writer", eligible[0]),
        )
    if short_conv_lane == "alpha_one_residual_writer_priority":
        return by_name.get(
            "full-source-producer-output-writer",
            by_name.get(
                "incumbent-producer-output-writer",
                by_name.get("full-source-structural", eligible[0]),
            ),
        )

    def score(candidate: _Candidate) -> tuple[Any, ...]:
        key = reports[candidate.name]["fit_ordering_key"]
        if not isinstance(key, list):
            raise Lfm2SemanticShortConvRefoldError(
                "A structurally eligible candidate has no deterministic FIT key."
            )
        return tuple(key)

    return min(eligible, key=score)


def _install_state_transaction(
    target_model: Any,
    *,
    layer_index: int,
    state: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], int]:
    torch = _torch()
    parameters = _parameter_bundle(target_model, layer_index)
    expected = _suffix_state(parameters, layer_index)
    if set(state) != set(expected):
        raise Lfm2SemanticShortConvRefoldError(
            "Candidate does not cover the complete existing short-conv topology."
        )
    before_model = fingerprint_lfm2_parent(target_model).identity_sha256
    count = _parameter_count(target_model)
    topology = _topology_sha256(target_model)
    snapshots = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in parameters.items()
    }
    parameter_ids = {name: id(value) for name, value in parameters.items()}
    storage_pointers = {
        name: int(value.data_ptr()) for name, value in parameters.items()
    }
    shapes = {name: tuple(value.shape) for name, value in parameters.items()}
    dtypes = {name: value.dtype for name, value in parameters.items()}
    requires_grad = {
        name: bool(value.requires_grad) for name, value in parameters.items()
    }
    gradient_ids = {
        name: None if value.grad is None else id(value.grad)
        for name, value in parameters.items()
    }
    gradient_hashes = {
        name: None if value.grad is None else _tensor_sha256(value.grad)
        for name, value in parameters.items()
    }
    prefix = f"model.layers.{layer_index}.conv."
    candidates = {name: state[name.removeprefix(prefix)] for name in parameters}
    for name, parameter in parameters.items():
        candidate = candidates[name]
        if (
            tuple(candidate.shape) != tuple(parameter.shape)
            or candidate.dtype != parameter.dtype
        ):
            raise Lfm2SemanticShortConvRefoldError(
                f"Candidate changed the topology or dtype of {name}."
            )
    expected_changed = tuple(
        sorted(
            name
            for name, parameter in parameters.items()
            if not torch.equal(candidates[name], parameter.detach().cpu())
        )
    )
    try:
        with torch.no_grad():
            for name in expected_changed:
                parameter = parameters[name]
                parameter.copy_(candidates[name].to(device=parameter.device))
        after_model = fingerprint_lfm2_parent(target_model).identity_sha256
        changed = tuple(
            sorted(
                name
                for name, parameter in parameters.items()
                if not torch.equal(parameter.detach().cpu(), snapshots[name])
            )
        )
        if (
            _parameter_count(target_model) != count
            or _topology_sha256(target_model) != topology
            or changed != expected_changed
            or any(id(parameters[name]) != parameter_ids[name] for name in parameters)
            or any(
                int(parameters[name].data_ptr()) != storage_pointers[name]
                for name in parameters
            )
            or any(tuple(parameters[name].shape) != shapes[name] for name in parameters)
            or any(parameters[name].dtype != dtypes[name] for name in parameters)
            or any(
                bool(parameters[name].requires_grad) != requires_grad[name]
                for name in parameters
            )
            or any(
                (None if parameters[name].grad is None else id(parameters[name].grad))
                != gradient_ids[name]
                for name in parameters
            )
            or any(
                parameters[name].grad is not None
                and _tensor_sha256(parameters[name].grad) != gradient_hashes[name]
                for name in parameters
            )
            or (bool(changed) != (after_model != before_model))
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Short-conv installation changed topology, identities, or gradients."
            )
        return before_model, after_model, changed, count
    except BaseException:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(
                    snapshots[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_model
            or _parameter_count(target_model) != count
            or _topology_sha256(target_model) != topology
            or any(id(parameters[name]) != parameter_ids[name] for name in parameters)
            or any(
                int(parameters[name].data_ptr()) != storage_pointers[name]
                for name in parameters
            )
            or any(
                (None if parameters[name].grad is None else id(parameters[name].grad))
                != gradient_ids[name]
                for name in parameters
            )
            or any(
                parameters[name].grad is not None
                and _tensor_sha256(parameters[name].grad) != gradient_hashes[name]
                for name in parameters
            )
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "Short-conv installation failed and exact rollback was not proven."
            )
        raise


def compile_install_lfm2_semantic_short_conv_postfold(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticShortConvReplayEvidence],
    source_fir_weight: Any,
    cache_key: str,
    bridge_artifact_sha256: str,
    source_model_identity_sha256: str,
    source_fir_bias: Any | None = None,
    fold_compute_device: str = "cpu",
    body_recipe: str | Lfm2SemanticBodyRecipe = "fit_guided",
    short_conv_construction: (
        str | Lfm2SemanticShortConvConstruction
    ) = LFM2_SEMANTIC_SHORT_CONV_STRUCTURAL_CONSTRUCTION,
    fhvr_source: Lfm2SemanticShortConvFhvrSource | None = None,
) -> Mapping[str, Any]:
    """Compile, gate, freeze, and atomically install one short-conv operator."""

    torch = _torch()
    cache = _digest(cache_key, "cache key")
    bridge = _digest(bridge_artifact_sha256, "bridge artifact SHA-256")
    source_identity = _digest(
        source_model_identity_sha256,
        "source model identity SHA-256",
    )
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    construction = resolve_lfm2_semantic_short_conv_construction(
        short_conv_construction
    )
    raw = tuple(evidence)
    if not raw or any(
        not isinstance(item, Lfm2SemanticShortConvReplayEvidence) for item in raw
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Short-conv postfold requires typed replay evidence."
        )
    if any(item.verify() != item.evidence_sha256 for item in raw):
        raise Lfm2SemanticShortConvRefoldError(
            "Short-conv evidence changed after authentication."
        )
    if construction.uses_fhvr_full_coverage:
        if (
            not isinstance(fhvr_source, Lfm2SemanticShortConvFhvrSource)
            or not fhvr_source.verify()
            or any(
                item.parent_operator_input_history is None
                or item.parent_operator_input_provenance_sha256 is None
                for item in raw
            )
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR construction requires an authenticated direct source and "
                "full parent operator-input histories."
            )
    elif fhvr_source is not None:
        raise Lfm2SemanticShortConvRefoldError(
            "The structural construction cannot consume an FHVR source side input."
        )
    current_child_state = fingerprint_lfm2_parent(target_model).identity_sha256
    target_topology_sha256 = _topology_sha256(target_model)
    lineage = raw[0].compiler_lineage_sha256
    if any(
        item.layer_index != layer_index
        or item.cache_key != cache
        or item.bridge_artifact_sha256 != bridge
        or item.captured_child_model_state_sha256 != current_child_state
        or item.compiler_lineage_sha256 != lineage
        for item in raw
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Replay layer/cache/bridge/lineage or captured child state diverged."
        )
    if construction.uses_fhvr_full_coverage and (
        fhvr_source.layer_index != layer_index
        or fhvr_source.cache_key != cache
        or fhvr_source.bridge_artifact_sha256 != bridge
        or fhvr_source.source_model_identity_sha256 != source_identity
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR direct source differs from layer/cache/bridge/parent lineage."
        )
    parameters = _parameter_bundle(target_model, layer_index)
    incumbent_state = _suffix_state(parameters, layer_index)
    exact_zero_incumbent = _exact_zero_state(incumbent_state)
    target_hidden = int(incumbent_state["out_proj.weight"].shape[0])
    source_kernel = _finite_cpu_tensor(
        source_fir_weight,
        "source_fir_weight",
        ndim=int(getattr(source_fir_weight, "ndim", -1)),
    )
    if source_kernel.ndim == 3:
        if int(source_kernel.shape[1]) != 1:
            raise Lfm2SemanticShortConvRefoldError(
                "Source FIR must be depthwise [source, 1, length]."
            )
        source_hidden = int(source_kernel.shape[0])
        normalized_source_kernel = source_kernel[:, 0, :].contiguous()
    elif source_kernel.ndim == 2:
        source_hidden = int(source_kernel.shape[0])
        normalized_source_kernel = source_kernel
    else:
        raise Lfm2SemanticShortConvRefoldError(
            "Source FIR must have rank two or three."
        )
    source_bias = None
    if source_fir_bias is not None:
        source_bias = _finite_cpu_tensor(
            source_fir_bias,
            "source_fir_bias",
            ndim=1,
        )
        if tuple(source_bias.shape) != (source_hidden,):
            raise Lfm2SemanticShortConvRefoldError(
                "Source FIR bias does not match its source channel count."
            )
    if target_hidden >= source_hidden or any(
        int(item.parent_b_history.shape[-1]) != source_hidden
        or int(item.actual_child_operator_input_history.shape[-1]) != target_hidden
        for item in raw
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Evidence widths do not match one smaller dense child and parent FIR."
        )
    if construction.uses_fhvr_full_coverage and (
        fhvr_source.source_hidden_size != source_hidden
        or fhvr_source.target_hidden_size != target_hidden
        or not torch.equal(
            fhvr_source.source_fir_weight,
            normalized_source_kernel,
        )
        or (fhvr_source.source_fir_bias is None) != (source_bias is None)
        or (
            source_bias is not None
            and not torch.equal(fhvr_source.source_fir_bias, source_bias)
        )
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "FHVR direct source and the cache-authenticated FIR disagree."
        )
    partitions = _split_evidence(raw)
    balanced_pairs = {
        name: _balance_partition(partitions[name]) for name in _PARTITIONS
    }
    banks = {name: balanced_pairs[name][0] for name in _PARTITIONS}
    balance_receipts = {name: balanced_pairs[name][1] for name in _PARTITIONS}
    baseline = {
        name: _endpoint_metrics(banks[name], incumbent_state)
        for name in ("fit", "fit_check", "selection")
    }
    incumbent = _Candidate(
        name="incumbent-noop",
        state=incumbent_state,
        provenance={
            "solver": "none-authenticated-live-dense-incumbent",
            "captured_child_model_state_sha256": current_child_state,
            "cache_key": cache,
            "bridge_artifact_sha256": bridge,
            "compiler_lineage_sha256": lineage,
            "fit_used_for_solve": False,
            "fit_check_used_for_solve": False,
            "selection_used_for_solve": False,
            "validation_used_for_solve": False,
            "operation_audit": dict(_ZERO_OPERATIONS),
        },
    )
    structural = _compile_structural_candidate(
        source_fir_weight=source_kernel,
        source_fir_bias=source_bias,
        incumbent_state=incumbent_state,
        fit=partitions["fit"],
        fit_check=partitions["fit_check"],
        **(
            {"fold_compute_device": fold_compute_device}
            if fold_compute_device != "cpu"
            else {}
        ),
    )
    cold_start_axis_coverage: Mapping[str, Any] | None = None
    if exact_zero_incumbent:
        cold_start_axis_coverage = _validate_cold_start_structural_plant(structural)
    blank_residual_writer = bool(
        exact_zero_incumbent
        and recipe.component_lanes["short_conv"] == "alpha_one_residual_writer_priority"
    )
    fhvr_construction = construction.uses_fhvr_full_coverage
    writer, writer_fit_report = _compile_writer_candidate(
        banks["fit"],
        incumbent_state,
        fixed_producer_state=(
            structural.state if blank_residual_writer or fhvr_construction else None
        ),
        candidate_name=(
            "full-source-producer-output-writer"
            if blank_residual_writer or fhvr_construction
            else "incumbent-producer-output-writer"
        ),
        fixed_producer_provenance=(
            structural.provenance
            if blank_residual_writer or fhvr_construction
            else None
        ),
        body_recipe=recipe,
        fold_compute_device=fold_compute_device,
    )
    fhvr_candidate = None
    fhvr_validation_root: Mapping[str, Any] | None = None
    if fhvr_construction:
        if writer is None:
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR full coverage has no structurally eligible recipe writer "
                "candidate to combine with its future-history realization."
            )
        fhvr_candidate, fhvr_validation_root = _compile_fhvr_full_coverage_candidate(
            construction=construction,
            source=fhvr_source,
            fit=partitions["fit"],
            incumbent_state=incumbent_state,
            writer_anchor_candidate=writer,
            fold_compute_device=fold_compute_device,
        )
    compute_proofs: dict[str, Any] = {}
    if fold_compute_device == "cpu":
        writer_execution = writer_fit_report.get("compute_execution")
        if not isinstance(writer_execution, Mapping):
            raise Lfm2SemanticShortConvRefoldError(
                "CPU short-conv writer solve returned no execution proof."
            )
        execution_body = dict(writer_execution)
        execution_sha256 = execution_body.pop("execution_sha256", None)
        if (
            writer_execution.get("backend") != "cpu"
            or writer_execution.get("requested_device") != fold_compute_device
            or writer_execution.get("requested_device_was_explicit") is not True
            or writer_execution.get("resolved_device") != "cpu"
            or writer_execution.get("compute_dtype") != "float64"
            or writer_execution.get("cuda_tensor_stage_count") != 0
            or not isinstance(execution_sha256, str)
            or canonical_sha256(execution_body) != execution_sha256
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "CPU short-conv writer execution proof is invalid."
            )
        compute_proofs = {
            "fold_compute_device": fold_compute_device,
            "writer_compute_execution": dict(writer_execution),
            "writer_compute_execution_sha256": execution_sha256,
            "requested_cpu_short_conv_writer_route_proved": True,
        }
    else:
        structural_report = structural.provenance.get("structural_compiler_report")
        structural_execution = (
            structural_report.get("execution", {}).get("compute_execution")
            if isinstance(structural_report, Mapping)
            and isinstance(structural_report.get("execution"), Mapping)
            else None
        )
        writer_execution = writer_fit_report.get("compute_execution")
        for execution, name in (
            (structural_execution, "structural clustering"),
            (writer_execution, "writer solve"),
        ):
            if not isinstance(execution, Mapping):
                raise Lfm2SemanticShortConvRefoldError(
                    f"CUDA short-conv {name} returned no execution proof."
                )
            execution_body = dict(execution)
            execution_sha256 = execution_body.pop("execution_sha256", None)
            if (
                execution.get("backend") != "cuda"
                or execution.get("requested_device") != fold_compute_device
                or execution.get("compute_dtype") != "float64"
                or not isinstance(execution_sha256, str)
                or canonical_sha256(execution_body) != execution_sha256
            ):
                raise Lfm2SemanticShortConvRefoldError(
                    f"CUDA short-conv {name} execution proof is invalid."
                )
        compute_proofs = {
            "fold_compute_device": fold_compute_device,
            "structural_cluster_compute_execution": dict(structural_execution),
            "structural_cluster_compute_execution_sha256": (
                structural_execution["execution_sha256"]
            ),
            "writer_compute_execution": dict(writer_execution),
            "writer_compute_execution_sha256": writer_execution["execution_sha256"],
            "requested_cuda_short_conv_routes_proved": True,
        }
    if fhvr_candidate is not None:
        fhvr_execution = fhvr_candidate.provenance.get("fhvr_compute_execution")
        if not isinstance(fhvr_execution, Mapping):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR hybrid returned no authenticated compute execution."
            )
        execution_body = dict(fhvr_execution)
        execution_sha256 = execution_body.pop("execution_sha256", None)
        if (
            execution_sha256 != canonical_sha256(execution_body)
            or fhvr_execution.get("requested_device") != fold_compute_device
            or fhvr_execution.get("backend")
            != ("cpu" if fold_compute_device == "cpu" else "cuda")
        ):
            raise Lfm2SemanticShortConvRefoldError(
                "FHVR hybrid compute execution is stale or device-mismatched."
            )
        compute_proofs.update(
            {
                "fhvr_compute_execution": dict(fhvr_execution),
                "fhvr_compute_execution_sha256": execution_sha256,
                "requested_fhvr_compute_route_proved": True,
            }
        )
    candidates = [
        incumbent,
        structural,
        *([] if writer is None else [writer]),
        *([] if fhvr_candidate is None else [fhvr_candidate]),
    ]
    tolerance = 1e-8
    reports = {
        candidate.name: _candidate_report(
            candidate,
            incumbent_state=incumbent_state,
            banks=banks,
            baseline=baseline,
            tolerance=tolerance,
            body_recipe=recipe,
        )
        for candidate in candidates
    }
    if (
        exact_zero_incumbent
        and reports[structural.name]["structurally_eligible"] is not True
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "The compulsory cold-start structural plant is not effectful and safe."
        )
    if (
        fhvr_construction
        and reports[fhvr_candidate.name]["structurally_eligible"] is not True
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "The authenticated FHVR hybrid is not one finite effectful dense state."
        )
    prevalidation_choice = (
        fhvr_candidate
        if fhvr_construction
        else (
            _select_candidate(candidates, reports, body_recipe=recipe)
            if blank_residual_writer
            else (
                structural
                if exact_zero_incumbent
                else _select_candidate(candidates, reports, body_recipe=recipe)
            )
        )
    )
    if exact_zero_incumbent:
        cold_start_axis_coverage = _validate_cold_start_structural_plant(
            prevalidation_choice,
            expected_fhvr_root=fhvr_validation_root,
        )
    chosen = prevalidation_choice
    frozen_state = MappingProxyType(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in chosen.state.items()
        }
    )
    frozen_sha = chosen.state_sha256
    baseline_validation = _endpoint_metrics(
        banks["validation"],
        incumbent_state,
    )
    validation, validation_metrics_error = _diagnostic_endpoint_metrics(
        banks["validation"],
        frozen_state,
    )
    frozen_authentic = (
        canonical_sha256(
            {
                "name": chosen.name,
                "state": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(frozen_state.items())
                },
                "provenance": dict(chosen.provenance),
            }
        )
        == frozen_sha
    )
    if not frozen_authentic:
        raise Lfm2SemanticShortConvRefoldError(
            "Frozen short-conv candidate changed before isolated VALIDATION."
        )
    validation_nonregression = (
        None
        if validation is None
        else _nonregressing(
            validation,
            baseline_validation,
            tolerance=tolerance,
        )
    )
    validation_rejected_candidate: str | None = None
    if (
        canonical_sha256(
            {
                "name": chosen.name,
                "state": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(frozen_state.items())
                },
                "provenance": dict(chosen.provenance),
            }
        )
        != frozen_sha
    ):
        raise Lfm2SemanticShortConvRefoldError(
            "Final frozen short-conv choice failed authentication."
        )
    if fingerprint_lfm2_parent(target_model).identity_sha256 != current_child_state:
        raise Lfm2SemanticShortConvRefoldError(
            "Child changed while the analytic candidate was being compiled."
        )
    before, after, changed, parameter_count = _install_state_transaction(
        target_model,
        layer_index=layer_index,
        state=frozen_state,
    )
    inserted_count = sum(
        int(item.actual_child_operator_input_history.shape[0])
        * len(item.inserted_target_positions)
        for item in raw
    )
    endpoint_count = sum(
        int(item.actual_child_operator_input_history.shape[0])
        * len(item.endpoint_target_positions)
        for item in raw
    )
    boundary_count = sum(
        int(item.actual_child_operator_input_history.shape[0])
        * len(item.semantic_span_boundary_positions)
        for item in raw
    )
    source_fir_tensors = {"weight": source_kernel}
    if source_bias is not None:
        source_fir_tensors["bias"] = source_bias
    body = {
        "schema": _INSTALLATION_SCHEMA,
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["short_conv"],
        **(
            {
                "short_conv_construction": construction.as_dict(),
                "short_conv_construction_id": construction.construction_id,
                "short_conv_construction_sha256": (construction.construction_sha256),
                "fhvr_source": fhvr_source.as_dict(),
                "fhvr_source_sha256": fhvr_source.source_sha256,
                "fhvr_candidate_report": dict(fhvr_candidate.provenance),
                "fhvr_candidate_state_sha256": fhvr_candidate.state_sha256,
                "fhvr_validation_root": dict(fhvr_validation_root),
                "fhvr_validation_root_sha256": fhvr_validation_root["root_sha256"],
                "full_history_future_observable_full_coverage_hybrid": True,
            }
            if fhvr_candidate is not None
            and fhvr_source is not None
            and fhvr_validation_root is not None
            else {}
        ),
        "status": (
            "installed-frozen-semantic-short-conv-candidate"
            if changed
            else _STRUCTURAL_NOOP_STATUS
        ),
        "installed": bool(changed),
        "install_count": int(bool(changed)),
        "layer_index": layer_index,
        "cache_key": cache,
        "bridge_artifact_sha256": bridge,
        "source_model_identity_sha256": source_identity,
        "compiler_lineage_sha256": lineage,
        "transport_evidence_sha256": [item.evidence_sha256 for item in raw],
        "source_capability_sha256": sorted(
            {item.source_capability_sha256 for item in raw}
        ),
        "source_history_sha256": sorted({item.source_history_sha256 for item in raw}),
        "source_fir_sha256": {
            name: _tensor_sha256(value) for name, value in source_fir_tensors.items()
        },
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "target_topology_sha256_before": target_topology_sha256,
        "target_topology_sha256_after": _topology_sha256(target_model),
        "changed_state_keys": list(changed),
        "changed_parameter_roles": [
            name.removeprefix(f"model.layers.{layer_index}.conv.") for name in changed
        ],
        "changed_state_keys_local_to_operator": all(
            name.startswith(f"model.layers.{layer_index}.conv.") for name in changed
        ),
        "parameter_element_count": parameter_count,
        "parameter_count_before": parameter_count,
        "parameter_count_after": parameter_count,
        "parameter_count_changed": False,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "model_identity_unchanged": before == after,
        "exact_noop_state_proven": not changed and before == after,
        "target_hidden_size": target_hidden,
        "source_hidden_size": source_hidden,
        "hidden_width_compression_ratio": source_hidden / target_hidden,
        "partition_batch_indices": {
            name: sorted({item.batch_index for item in partitions[name]})
            for name in _PARTITIONS
        },
        "trajectory_row_weight_balance": {
            name: dict(balance_receipts[name]) for name in _PARTITIONS
        },
        "baseline_partition_metrics": {
            **{
                name: baseline[name].as_dict()
                for name in ("fit", "fit_check", "selection")
            },
            "validation": baseline_validation.as_dict(),
        },
        "candidate_reports_before_validation": {
            name: dict(report) for name, report in reports.items()
        },
        "writer_fit_proposal": dict(writer_fit_report),
        "prevalidation_selected_candidate": prevalidation_choice.name,
        "validation_rejected_candidate": validation_rejected_candidate,
        "selected_candidate": chosen.name,
        "selected_candidate_state_sha256": frozen_sha,
        "selected_candidate_state_identical_to_incumbent": (
            all(
                torch.equal(frozen_state[name], incumbent_state[name])
                for name in incumbent_state
            )
        ),
        "selected_validation_metrics": (
            None if validation is None else validation.as_dict()
        ),
        "selected_validation_metrics_error": validation_metrics_error,
        "candidate_frozen_before_fit_check": True,
        "candidate_frozen_before_selection": True,
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "selection_used_for_candidate_choice": False,
        "validation_used_for_candidate_choice": False,
        "validation_used_only_for_acceptance_or_incumbent_fallback": False,
        "incumbent_noop_candidate_competed": False,
        "incumbent_noop_fallback_used": chosen.name == "incumbent-noop",
        "incumbent_noop_fallback_reason": (
            "all-analytic-proposals-structurally-invalid-or-identical"
            if chosen.name == "incumbent-noop"
            else None
        ),
        "incumbent_selected_by_heldout_gates": False,
        "fit_check_nonregression_passed": reports[chosen.name][
            "partition_nonregression"
        ]["fit_check"],
        "selection_nonregression_passed": reports[chosen.name][
            "partition_nonregression"
        ]["selection"],
        "validation_nonregression_passed": validation_nonregression,
        "validation_diagnostic_would_have_failed_legacy_gate": (
            validation_nonregression is not True
        ),
        "candidate_selection_evidence_scope": (
            "finite-dense-structural-eligibility-plus-external-construction-choice"
            if fhvr_construction
            else (
                "finite-dense-structural-eligibility-plus-fit-ordering"
                if recipe.fit_metrics_order_candidates
                else "finite-dense-structural-eligibility-plus-recipe-priority"
            )
        ),
        "candidate_eligibility_basis": _CANDIDATE_ELIGIBILITY_BASIS,
        "structural_safety_is_only_candidate_veto": True,
        "structurally_eligible_candidate_count": sum(
            int(report["structurally_eligible"] is True)
            for name, report in reports.items()
            if name != "incumbent-noop"
        ),
        "fit_metrics_role": (
            "diagnostic-only"
            if fhvr_construction
            else (
                "ordering-among-structurally-eligible-not-threshold-gating"
                if recipe.fit_metrics_order_candidates
                else "diagnostic-only"
            )
        ),
        "fit_metrics_used_for_candidate_ordering": (
            recipe.fit_metrics_order_candidates and not fhvr_construction
        ),
        "fit_metrics_used_for_pre_review_candidate_selection": (
            recipe.fit_metrics_order_candidates and not fhvr_construction
        ),
        "fit_metrics_used_for_structural_eligibility": False,
        "fit_thresholds_used_for_candidate_eligibility": False,
        "fit_check_thresholds_used_for_candidate_eligibility": False,
        "selection_thresholds_used_for_candidate_eligibility": False,
        "validation_thresholds_used_for_publication": False,
        "relative_magnitude_threshold_used_for_candidate_eligibility": False,
        "relative_magnitude_metrics_diagnostic_only": True,
        "fit_check_metrics_diagnostic_only": True,
        "selection_metrics_diagnostic_only": True,
        "validation_metrics_diagnostic_only": True,
        "heldout_metrics_are_diagnostic_only": True,
        "numeric_metrics_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_candidate_eligibility": False,
        "numeric_thresholds_used_for_publication": False,
        "numeric_diagnostic_miss_can_exclude_structural_candidate": False,
        "validation_publication_veto_applied": False,
        "rollback_authorized": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "fit_metrics_have_final_intelligence_authority": False,
        "behavioral_intelligence_review_outside_short_conv_compiler": True,
        "complete_parent_bcx_history_consumed": True,
        "parent_bcx_exact_only_at_semantic_span_boundaries": True,
        "inserted_target_bcx_rows_are_owner_expanded_proxies": True,
        "complete_source_channel_coverage_candidate_compiled": True,
        "full_source_structural_candidate_competed": True,
        "incumbent_producer_output_writer_candidate_competed": bool(
            writer is not None and writer.name == "incumbent-producer-output-writer"
        ),
        "full_source_producer_output_writer_candidate_competed": bool(
            writer is not None and writer.name == "full-source-producer-output-writer"
        ),
        "full_history_fhvr_full_coverage_candidate_competed": (
            fhvr_candidate is not None
        ),
        "inserted_target_row_count": inserted_count,
        "semantic_boundary_bcx_row_count": boundary_count,
        "parent_exact_endpoint_row_count": endpoint_count,
        "inserted_target_rows_parent_exact": False,
        "inserted_target_rows_used_for_endpoint_fit": False,
        "inserted_target_bcx_proxy_used_for_source_channel_geometry": True,
        "inserted_target_bcx_proxy_used_for_causal_fir_warmup": True,
        "contractions_supported": False,
        "dense_existing_in_proj_conv_out_proj_topology_only": True,
        "numeric_parent_child_weight_match_objective": False,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
        **(
            {
                "exact_zero_incumbent_at_entry": True,
                "cold_start_compulsory_full_source_structural_plant": True,
                "cold_start_heldout_metrics_diagnostic_only": True,
                "cold_start_fit_structural_candidate_forced_before_heldout": (
                    chosen.name == "full-source-structural"
                ),
                "cold_start_recipe_candidate_forced_before_heldout": True,
                "cold_start_selected_state_uses_full_source_structural_producers": (
                    chosen.name
                    in {
                        "full-source-structural",
                        "full-source-producer-output-writer",
                        "full-history-fhvr-full-coverage-hybrid",
                    }
                ),
                "cold_start_residual_writer_component_equivalent_to_structural": (
                    blank_residual_writer and chosen.name == "full-source-structural"
                ),
                "cold_start_fit_check_used_for_publication_gate": False,
                "cold_start_selection_used_for_publication_gate": False,
                "cold_start_validation_used_for_publication_gate": False,
                "cold_start_blank_noop_forbidden": True,
                "cold_start_all_required_operator_weights_finite_nonzero": True,
                "cold_start_optional_biases_finite": True,
                "cold_start_all_required_operator_axes_covered": True,
                "cold_start_structural_axis_coverage": dict(
                    cold_start_axis_coverage or {}
                ),
                "cold_start_structural_axis_coverage_sha256": (
                    cold_start_axis_coverage["result_sha256"]
                    if cold_start_axis_coverage is not None
                    else None
                ),
                "cold_start_structural_axis_repair_applied": False,
            }
            if exact_zero_incumbent
            else {}
        ),
        **compute_proofs,
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


__all__ = [
    "LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS",
    "LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_SCHEMA",
    "LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION",
    "LFM2_SEMANTIC_SHORT_CONV_FHVR_SOURCE_SCHEMA",
    "LFM2_SEMANTIC_SHORT_CONV_REFOLD_SCHEMA",
    "LFM2_SEMANTIC_SHORT_CONV_REPLAY_EVIDENCE_SCHEMA",
    "LFM2_SEMANTIC_SHORT_CONV_STRUCTURAL_CONSTRUCTION",
    "Lfm2SemanticShortConvConstruction",
    "Lfm2SemanticShortConvConstructionId",
    "Lfm2SemanticShortConvEndpointMetrics",
    "Lfm2SemanticShortConvFhvrSource",
    "Lfm2SemanticShortConvRefoldError",
    "Lfm2SemanticShortConvReplayEvidence",
    "compile_install_lfm2_semantic_short_conv_postfold",
    "resolve_lfm2_semantic_short_conv_construction",
]
