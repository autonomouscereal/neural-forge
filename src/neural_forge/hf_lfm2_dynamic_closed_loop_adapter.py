"""Concrete LFM2 adapter for the family-neutral child-closed-loop compiler.

The adapter is deliberately narrow about evidence provenance:

* activations and desired endpoints come only from a verified sealed parent
  :class:`Lfm2CircuitTraceCache`;
* every capture replays every exact token batch through the *current child*;
  canonical uncached prefix replay is the default, while the explicitly
  selected single-pass mode is accepted only under its audited causal runtime
  contract;
* the parent model is retained only as an immutable source-weight container
  and is fingerprint-bound to the sealed cache; and
* no tokenizer, parent forward, trace generation, gradient, backward pass, or
  optimizer is available through this API.

The child must already be a complete blank-target fold made with the supplied
dense residual coordinate map.  The explicit ``initial_child_basis_sha256``
binding prevents this closed-loop refinement pass from being accidentally
applied to a child initialized with a different residual basis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_circuit_ir import JsonScalar, canonical_sha256
from .dynamic_closed_loop_fold import (
    CapturePoint,
    ChildTrajectoryCapture,
    ClosedLoopAudit,
    ClosedLoopFoldError,
    CompileStage,
    CompiledFoldStage,
    FoldScope,
    InstallReceipt,
    ParentEndpointBatch,
    RecordKey,
    TransactionReceipt,
    TrajectoryRecord,
    ValidationDecision,
)
from .hf_closed_form_fit import FitMetrics, fit_diagonal_norm, measure_fit
from .hf_lfm2_canonical import (
    build_lfm2_short_conv_channel_plan,
    inspect_lfm2_heterogeneous_layer,
    inspect_lfm2_layer,
)
from .hf_lfm2_child_closed_loop_refinement import (
    Lfm2FinalNormPairs,
    Lfm2NormFitRequest,
    Lfm2NormMaterializer,
    Lfm2OfflineEndpointComparison,
    Lfm2OperatorProducerCases,
    Lfm2PairedReplay,
    Lfm2WholeBatchSplit,
    build_lfm2_final_norm_pairs,
    build_lfm2_mlp_cases_from_candidate_post_operator,
    build_lfm2_operator_producer_cases,
    compute_lfm2_candidate_post_layer,
    compute_lfm2_candidate_post_operator,
    extract_lfm2_actual_child_contexts,
    pair_lfm2_parent_child_replay,
    project_lfm2_paired_parent_logits,
    split_lfm2_replay_batches,
)
from .hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    capture_lfm2_child_replay,
    fingerprint_lfm2_parent,
)
from .hf_lfm2_dynamic_blank_materializer import (
    extract_lfm2_dynamic_physical_layout_from_config,
)
from .hf_lfm2_dynamic_physical_layout import (
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
)
from .hf_lfm2_macro_cell_attractor_gate import (
    Lfm2MacroCellAttractorGate,
    Lfm2MacroCellAttractorSnapshot,
)
from .hf_lfm2_structural_residual_attention_fold import (
    Lfm2AttentionScorePreservationReport,
    Lfm2AttentionScorePreservationThresholds,
    Lfm2CompiledStructuralAttentionConsumerFold,
    Lfm2CompiledStructuralAttentionProducerFold,
    Lfm2CompiledStructuralAttentionValueOutputFold,
    Lfm2CompiledStructuralMlpFold,
    Lfm2CompiledStructuralShortConvConsumerFold,
    Lfm2CompiledStructuralShortConvProducerFold,
    Lfm2SparseResidualAssignment,
    Lfm2StructuralSplitInstallTransaction,
    compile_lfm2_structural_attention_consumer_fold,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_attention_value_output_fold,
    compile_lfm2_structural_residual_mlp_fold,
    compile_lfm2_structural_short_conv_consumer_fold,
    compile_lfm2_structural_short_conv_producer_fold,
    evaluate_lfm2_compiled_attention_producer_score_preservation,
    install_lfm2_structural_attention_consumer_fold,
    install_lfm2_structural_attention_producer_fold,
    install_lfm2_structural_attention_score_producer_fold,
    install_lfm2_structural_attention_value_output_fold,
    install_lfm2_structural_mlp_fold,
    install_lfm2_structural_short_conv_consumer_fold,
    install_lfm2_structural_short_conv_producer_fold,
    rollback_lfm2_structural_split_install,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2StrictTokenizerBridgeTraceError,
)
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    StrictTokenizerBridgeError,
    compile_strict_vocabulary_operators,
)


LFM2_ADAPTER_SCHEMA = "neural-forge-lfm2-dynamic-closed-loop-adapter-v2"
LFM2_TRAJECTORY_PATH = "lfm2.full_prefix"
_FORBIDDEN_COUNTERS = (
    "parent_forward_calls",
    "parent_trace_regeneration_calls",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
)


class Lfm2DynamicClosedLoopAdapterError(ClosedLoopFoldError):
    """The concrete LFM2 adapter contract or a local transaction failed."""


class DenseResidualCoordinateMap(Protocol):
    """Minimum dense-map contract accepted by the LFM2 adapter."""

    source_width: int
    target_width: int
    map_sha256: str

    def encode(self, source_rows: Any) -> Any: ...

    def fold_consumer_weight(self, source_weight: Any) -> Any: ...

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: Any | None = None,
        device: Any = "cpu",
    ) -> Any: ...


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF boundary
        raise RuntimeError(
            "PyTorch is required for the LFM2 dynamic closed-loop adapter."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be finite and non-negative."
        )
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be finite."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be finite."
        ) from exc
    if not math.isfinite(result):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be finite."
        )
    return result


def _unit_fraction(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0 or result > 1.0:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be within [0, 1]."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{name} must be a positive integer."
        )
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2DynamicClosedLoopAdapterError(
            "Only materialized dense tensors can be fingerprinted."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_state_sha256(model: Any) -> str:
    torch = _torch()
    try:
        state = model.state_dict()
    except Exception as exc:
        raise Lfm2DynamicClosedLoopAdapterError(
            "Could not read the child state dictionary."
        ) from exc
    if not isinstance(state, Mapping) or not state:
        raise Lfm2DynamicClosedLoopAdapterError(
            "The child state dictionary is empty."
        )
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-live-child-state-v1\0")
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Child state entries must be string-named tensors."
            )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_tensor_sha256(value)))
        digest.update(b"\0")
    for name, parameter in sorted(
        model.named_parameters(remove_duplicate=False)
    ):
        digest.update(b"parameter\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(id(parameter)).encode("ascii"))
        digest.update(b"\0")
        gradient = parameter.grad
        if gradient is None:
            digest.update(b"gradient:none\0")
        else:
            digest.update(b"gradient:")
            digest.update(str(id(gradient)).encode("ascii"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_tensor_sha256(gradient)))
            digest.update(b"\0")
    return digest.hexdigest()


def _model_topology_sha256(model: Any) -> str:
    torch = _torch()
    state = model.state_dict()
    parameters = tuple(model.named_parameters(remove_duplicate=False))
    buffers = tuple(model.named_buffers(remove_duplicate=False))
    body = {
        "schema": "neural-forge-lfm2-live-child-topology-v1",
        "model_class": (
            f"{type(model).__module__}.{type(model).__qualname__}"
        ),
        "state": [
            {
                "name": name,
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
                "layout": str(value.layout),
            }
            for name, value in sorted(state.items())
            if isinstance(value, torch.Tensor)
        ],
        "parameters": [
            {
                "name": name,
                "id": id(parameter),
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in parameters
        ],
        "buffers": [
            {
                "name": name,
                "id": id(value),
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
            }
            for name, value in buffers
        ],
    }
    return canonical_sha256(body)


def _layers(model: Any) -> tuple[Any, ...]:
    for value in (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    ):
        if value is None or isinstance(value, (str, bytes)):
            continue
        try:
            result = tuple(value)
        except TypeError:
            continue
        if result:
            return result
    raise Lfm2DynamicClosedLoopAdapterError(
        "Could not locate LFM2 decoder layers."
    )


def _final_norm(model: Any) -> Any:
    body = getattr(model, "model", None)
    for owner in (body, model):
        if owner is None:
            continue
        for name in ("embedding_norm", "norm"):
            module = getattr(owner, name, None)
            if module is not None and getattr(module, "weight", None) is not None:
                return module
    raise Lfm2DynamicClosedLoopAdapterError(
        "Could not locate the LFM2 final embedding norm."
    )


def _heterogeneous_layout(
    model: Any,
    role: str,
) -> Lfm2HeterogeneousPhysicalLayoutDescriptor | None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(
        config,
        "neural_forge_dynamic_blank_config_schema",
    ):
        return None
    try:
        layout = extract_lfm2_dynamic_physical_layout_from_config(config)
    except Exception as exc:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} dynamic LFM2 physical layout failed authentication."
        ) from exc
    if type(layout) is not Lfm2HeterogeneousPhysicalLayoutDescriptor:
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} dynamic child must use the sealed heterogeneous layout."
        )
    return layout


def _inspect_runtime_layer(
    layer: Any,
    layout: Lfm2HeterogeneousPhysicalLayoutDescriptor | None,
) -> Any:
    return (
        inspect_lfm2_layer(layer)
        if layout is None
        else inspect_lfm2_heterogeneous_layer(layer, layout)
    )


def _vocabulary_parameter_pair(
    model: Any,
    role: str,
    *,
    require_tied: bool,
) -> tuple[Any, Any]:
    """Return authenticated dense input/output vocabulary Parameters."""

    torch = _torch()
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} exposes no input/output embedding accessors."
        )
    input_module = get_input()
    output_module = get_output()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    output_bias = getattr(output_module, "bias", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or not isinstance(output_weight, torch.nn.Parameter)
        or input_weight.ndim != 2
        or output_weight.ndim != 2
        or tuple(input_weight.shape) != tuple(output_weight.shape)
        or output_bias is not None
        or (require_tied and input_weight is not output_weight)
    ):
        topology = "tied " if require_tied else ""
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} must use bias-free dense {topology}input/output vocabulary "
            "Parameters with identical shapes."
        )
    for name, parameter in (
        ("input", input_weight),
        ("output", output_weight),
    ):
        if (
            parameter.layout != torch.strided
            or parameter.device.type == "meta"
            or not torch.is_floating_point(parameter)
            or torch.is_complex(parameter)
            or not bool(torch.isfinite(parameter.detach()).all().item())
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                f"{role} {name} vocabulary Parameter is not finite dense real data."
            )
    return input_weight, output_weight


def _tied_vocabulary_parameter(model: Any, role: str) -> Any:
    """Return one genuinely tied input/output vocabulary Parameter."""

    input_weight, output_weight = _vocabulary_parameter_pair(
        model,
        role,
        require_tied=True,
    )
    if input_weight is not output_weight or id(input_weight) != id(output_weight):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} vocabulary Parameter is not tied by identity."
        )
    return input_weight


def _rms_epsilon(module: Any, role: str) -> float:
    for name in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(module, name, None)
        if value is None:
            continue
        result = _finite_float(value, f"{role} epsilon")
        if result <= 0.0:
            break
        return result
    raise Lfm2DynamicClosedLoopAdapterError(
        f"{role} exposes no finite positive RMSNorm epsilon."
    )


def _analytical_rms_norm(value: Any, module: Any, role: str) -> Any:
    torch = _torch()
    weight = getattr(module, "weight", None)
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 2
        or not isinstance(weight, torch.Tensor)
        or weight.ndim != 1
        or int(value.shape[-1]) != int(weight.shape[0])
    ):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} RMSNorm input/weight topology is invalid."
        )
    source = value.detach().to(device="cpu").contiguous()
    runtime_weight = weight.detach().to(
        device="cpu", dtype=source.dtype
    )
    normalized = source.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True)
        + _rms_epsilon(module, role)
    )
    result = runtime_weight * normalized.to(dtype=source.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2DynamicClosedLoopAdapterError(
            f"{role} analytical RMSNorm produced non-finite values."
        )
    return result.contiguous()


def _record_keys(
    records: Sequence[TrajectoryRecord],
) -> tuple[RecordKey, ...]:
    return tuple(sorted(record.key for record in records))


def _metric_dict(metrics: FitMetrics) -> dict[str, JsonScalar]:
    return {
        "rows": int(metrics.rows),
        "output_width": int(metrics.output_width),
        "rmse": float(metrics.rmse),
        "target_rms": float(metrics.target_rms),
        "relative_rmse": float(metrics.relative_rmse),
        "mean_cosine_similarity": float(
            metrics.mean_cosine_similarity
        ),
        "finite": bool(metrics.finite),
    }


def _scalar_metadata(
    values: Mapping[str, JsonScalar],
) -> tuple[tuple[str, JsonScalar], ...]:
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class Lfm2DynamicClosedLoopAdapterConfig:
    """Closed-form fit and held-out acceptance policy."""

    ridge: float = 1e-6
    norm_ridge: float = 1e-6
    max_prior_delta_ratio: float | None = 1.0
    cluster_ridge: float = 1e-8
    cluster_max_iterations: int = 16
    cluster_distance_chunk_size: int = 512
    minimum_layer_relative_rmse_improvement: float = 0.0
    minimum_final_norm_relative_rmse_improvement: float = 0.0
    output_ridge: float = 1e-3
    output_max_prior_delta_ratio: float = 0.25
    output_max_transform_condition_number: float = 4.0
    output_min_transform_singular_value: float = 0.25
    minimum_output_kl_improvement: float = 0.0
    minimum_output_known_sequence_nll_improvement: float = 0.0
    maximum_output_eos_margin_rmse_regression: float = 0.0
    output_validation_tolerance: float = 1e-8
    output_metric_chunk_rows: int = 16
    validation_tolerance: float = 1e-8
    replay_capture_mode: str = CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    attention_consumer_mode: str = "output-only"
    attention_value_output_iterations: int = 2
    attention_value_output_context_ridge: float = 1e-6
    attention_value_output_fit_tolerance: float = 1e-10
    attention_value_output_context_equivalence_tolerance: float = 5e-2
    attention_score_candidate_gate_enabled: bool = False
    attention_score_top_k: int = 8
    attention_score_margin_floor_fraction: float = 0.5
    attention_score_minimum_top1_key_agreement_fraction: float = 0.9
    attention_score_minimum_top_k_pair_order_agreement_fraction: float = 0.9
    attention_score_minimum_winner_margin_sign_agreement_fraction: float = 0.9
    attention_score_minimum_relative_margin_floor_retention_fraction: (
        float
    ) = 0.9
    attention_score_maximum_softmax_fisher_error: float | None = None
    attention_score_maximum_js_divergence: float | None = None
    output_interface_mode: str = "fit-transform"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ridge", _nonnegative_float(self.ridge, "ridge")
        )
        object.__setattr__(
            self,
            "norm_ridge",
            _nonnegative_float(self.norm_ridge, "norm_ridge"),
        )
        if self.max_prior_delta_ratio is not None:
            object.__setattr__(
                self,
                "max_prior_delta_ratio",
                _nonnegative_float(
                    self.max_prior_delta_ratio,
                    "max_prior_delta_ratio",
                ),
            )
        object.__setattr__(
            self,
            "cluster_ridge",
            _nonnegative_float(self.cluster_ridge, "cluster_ridge"),
        )
        object.__setattr__(
            self,
            "cluster_max_iterations",
            _positive_int(
                self.cluster_max_iterations,
                "cluster_max_iterations",
            ),
        )
        object.__setattr__(
            self,
            "cluster_distance_chunk_size",
            _positive_int(
                self.cluster_distance_chunk_size,
                "cluster_distance_chunk_size",
            ),
        )
        object.__setattr__(
            self,
            "minimum_layer_relative_rmse_improvement",
            _finite_float(
                self.minimum_layer_relative_rmse_improvement,
                "minimum_layer_relative_rmse_improvement",
            ),
        )
        object.__setattr__(
            self,
            "minimum_final_norm_relative_rmse_improvement",
            _finite_float(
                self.minimum_final_norm_relative_rmse_improvement,
                "minimum_final_norm_relative_rmse_improvement",
            ),
        )
        object.__setattr__(
            self,
            "output_ridge",
            _nonnegative_float(self.output_ridge, "output_ridge"),
        )
        object.__setattr__(
            self,
            "output_max_prior_delta_ratio",
            _nonnegative_float(
                self.output_max_prior_delta_ratio,
                "output_max_prior_delta_ratio",
            ),
        )
        condition = _finite_float(
            self.output_max_transform_condition_number,
            "output_max_transform_condition_number",
        )
        if condition < 1.0:
            raise Lfm2DynamicClosedLoopAdapterError(
                "output_max_transform_condition_number must be at least 1."
            )
        object.__setattr__(
            self,
            "output_max_transform_condition_number",
            condition,
        )
        minimum_singular = _finite_float(
            self.output_min_transform_singular_value,
            "output_min_transform_singular_value",
        )
        if minimum_singular <= 0.0 or minimum_singular > 1.0:
            raise Lfm2DynamicClosedLoopAdapterError(
                "output_min_transform_singular_value must be within (0, 1]."
            )
        object.__setattr__(
            self,
            "output_min_transform_singular_value",
            minimum_singular,
        )
        object.__setattr__(
            self,
            "minimum_output_kl_improvement",
            _finite_float(
                self.minimum_output_kl_improvement,
                "minimum_output_kl_improvement",
            ),
        )
        object.__setattr__(
            self,
            "minimum_output_known_sequence_nll_improvement",
            _finite_float(
                self.minimum_output_known_sequence_nll_improvement,
                "minimum_output_known_sequence_nll_improvement",
            ),
        )
        object.__setattr__(
            self,
            "maximum_output_eos_margin_rmse_regression",
            _nonnegative_float(
                self.maximum_output_eos_margin_rmse_regression,
                "maximum_output_eos_margin_rmse_regression",
            ),
        )
        object.__setattr__(
            self,
            "output_validation_tolerance",
            _nonnegative_float(
                self.output_validation_tolerance,
                "output_validation_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "output_metric_chunk_rows",
            _positive_int(
                self.output_metric_chunk_rows,
                "output_metric_chunk_rows",
            ),
        )
        object.__setattr__(
            self,
            "validation_tolerance",
            _nonnegative_float(
                self.validation_tolerance, "validation_tolerance"
            ),
        )
        if (
            not isinstance(self.replay_capture_mode, str)
            or self.replay_capture_mode
            not in {
                CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
                CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            }
        ):
            raise ValueError(
                "replay_capture_mode must be 'prefix_by_prefix_canonical' "
                "or 'single_pass_causal'."
            )
        if (
            not isinstance(self.attention_consumer_mode, str)
            or self.attention_consumer_mode
            not in {
                "output-only",
                "coupled-value-output",
            }
        ):
            raise ValueError(
                "attention_consumer_mode must be 'output-only' or "
                "'coupled-value-output'."
            )
        object.__setattr__(
            self,
            "attention_value_output_iterations",
            _positive_int(
                self.attention_value_output_iterations,
                "attention_value_output_iterations",
            ),
        )
        object.__setattr__(
            self,
            "attention_value_output_context_ridge",
            _nonnegative_float(
                self.attention_value_output_context_ridge,
                "attention_value_output_context_ridge",
            ),
        )
        object.__setattr__(
            self,
            "attention_value_output_fit_tolerance",
            _nonnegative_float(
                self.attention_value_output_fit_tolerance,
                "attention_value_output_fit_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "attention_value_output_context_equivalence_tolerance",
            _nonnegative_float(
                self.attention_value_output_context_equivalence_tolerance,
                "attention_value_output_context_equivalence_tolerance",
            ),
        )
        if not isinstance(
            self.attention_score_candidate_gate_enabled,
            bool,
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "attention_score_candidate_gate_enabled must be boolean."
            )
        score_top_k = _positive_int(
            self.attention_score_top_k,
            "attention_score_top_k",
        )
        if score_top_k < 2:
            raise Lfm2DynamicClosedLoopAdapterError(
                "attention_score_top_k must be at least two."
            )
        object.__setattr__(
            self,
            "attention_score_top_k",
            score_top_k,
        )
        for name in (
            "attention_score_margin_floor_fraction",
            "attention_score_minimum_top1_key_agreement_fraction",
            (
                "attention_score_minimum_top_k_pair_order_"
                "agreement_fraction"
            ),
            (
                "attention_score_minimum_winner_margin_sign_"
                "agreement_fraction"
            ),
            (
                "attention_score_minimum_relative_margin_floor_"
                "retention_fraction"
            ),
        ):
            object.__setattr__(
                self,
                name,
                _unit_fraction(getattr(self, name), name),
            )
        if self.attention_score_maximum_softmax_fisher_error is not None:
            object.__setattr__(
                self,
                "attention_score_maximum_softmax_fisher_error",
                _nonnegative_float(
                    self.attention_score_maximum_softmax_fisher_error,
                    "attention_score_maximum_softmax_fisher_error",
                ),
            )
        if self.attention_score_maximum_js_divergence is not None:
            object.__setattr__(
                self,
                "attention_score_maximum_js_divergence",
                _unit_fraction(
                    self.attention_score_maximum_js_divergence,
                    "attention_score_maximum_js_divergence",
                ),
            )
        if (
            not isinstance(self.output_interface_mode, str)
            or self.output_interface_mode
            not in {
                "fit-transform",
                "identity-skip",
            }
        ):
            raise ValueError(
                "output_interface_mode must be 'fit-transform' or "
                "'identity-skip'."
            )


@dataclass(frozen=True, slots=True)
class _CapturePayload:
    replay_sha256: str
    paired: Lfm2PairedReplay
    replay_capture_mode: str
    actual_child_forwards: int
    canonical_prefix_child_forwards: int
    saved_child_forwards: int
    causal_contract_sha256: str | None
    all_batches_replayed: bool = True
    canonical_causal_phase_schema: bool = True


@dataclass(frozen=True, slots=True)
class _EndpointPayload:
    cache_key: str
    batch_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ProducerPayload:
    operator_kind: str
    operator_cases: Lfm2OperatorProducerCases
    compiled: (
        Lfm2CompiledStructuralAttentionProducerFold
        | Lfm2CompiledStructuralShortConvProducerFold
    )
    operator_norm_weight: Any
    operator_norm_report_sha256: str
    baseline_validation: FitMetrics
    attention_score_fit: Lfm2AttentionScorePreservationReport | None
    attention_score_validation: (
        Lfm2AttentionScorePreservationReport | None
    )
    attractor_baseline: Lfm2MacroCellAttractorSnapshot | None
    split_sha256: str


@dataclass(frozen=True, slots=True)
class _ConsumerPayload:
    operator_kind: str
    compiled: (
        Lfm2CompiledStructuralAttentionConsumerFold
        | Lfm2CompiledStructuralAttentionValueOutputFold
        | Lfm2CompiledStructuralShortConvConsumerFold
    )
    operator_cases: Lfm2OperatorProducerCases
    split_sha256: str


@dataclass(frozen=True, slots=True)
class _DownstreamPayload:
    compiled: Lfm2CompiledStructuralMlpFold
    ffn_norm_weight: Any
    ffn_norm_report_sha256: str
    offline_comparison: Lfm2OfflineEndpointComparison
    split_sha256: str


@dataclass(frozen=True, slots=True)
class _FinalNormPayload:
    fitted_weight: Any
    fit_report_sha256: str
    baseline_validation: FitMetrics
    pairs: Lfm2FinalNormPairs
    split_sha256: str


@dataclass(frozen=True, slots=True)
class _OutputInterfacePayload:
    base_weight: Any
    candidate_weight: Any
    transform: Any
    transform_sha256: str
    base_weight_sha256: str
    candidate_weight_sha256: str
    source_binding_sha256: str
    split_sha256: str
    ridge_effective: float
    interpolation_alpha: float
    materialized_delta_ratio: float
    transform_condition_number: float
    transform_min_singular_value: float
    output_subspace_relative_rmse: float
    body_latent_relative_rmse: float
    identity_skip: bool = False


@dataclass(frozen=True, slots=True)
class _GradientSnapshot:
    parameter_name: str
    parameter: Any
    parameter_id: int
    gradient: Any | None
    gradient_id: int | None
    gradient_value: Any | None


class _ChildModelTransaction:
    def __init__(self, adapter: "Lfm2DynamicClosedLoopAdapter") -> None:
        torch = _torch()
        self._adapter = adapter
        self.state_sha256_before = adapter.child_state_fingerprint()
        self.topology_sha256_before = adapter.child_topology_fingerprint()
        self._state = tuple(
            (
                name,
                value.detach().to(device="cpu").contiguous().clone(),
            )
            for name, value in sorted(adapter.child_model.state_dict().items())
        )
        self._gradients = tuple(
            _GradientSnapshot(
                parameter_name=name,
                parameter=parameter,
                parameter_id=id(parameter),
                gradient=parameter.grad,
                gradient_id=(
                    None if parameter.grad is None else id(parameter.grad)
                ),
                gradient_value=(
                    None
                    if parameter.grad is None
                    else parameter.grad.detach()
                    .to(device="cpu")
                    .contiguous()
                    .clone()
                ),
            )
            for name, parameter in adapter.child_model.named_parameters(
                remove_duplicate=False
            )
        )
        self.active = True
        self._torch = torch

    def commit(self) -> TransactionReceipt:
        if not self.active:
            raise Lfm2DynamicClosedLoopAdapterError(
                "The child transaction is inactive."
            )
        after = self._adapter.child_state_fingerprint()
        self.active = False
        self._adapter._active_outer_transaction = None
        return TransactionReceipt(
            action="commit",
            accepted=True,
            exact=True,
            state_sha256_before=self.state_sha256_before,
            state_sha256_after=after,
            metadata=_scalar_metadata(
                {
                    "topology_sha256": self.topology_sha256_before,
                    "whole_child_transaction": True,
                }
            ),
        )

    def rollback(self) -> TransactionReceipt:
        if not self.active:
            raise Lfm2DynamicClosedLoopAdapterError(
                "The child transaction is inactive."
            )
        self._adapter._rollback_open_split_transactions()
        current_parameters = dict(
            self._adapter.child_model.named_parameters(
                remove_duplicate=False
            )
        )
        for snapshot in self._gradients:
            current = current_parameters.get(snapshot.parameter_name)
            if current is not snapshot.parameter or id(current) != snapshot.parameter_id:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "A child Parameter identity changed before rollback."
                )
        current_state = self._adapter.child_model.state_dict()
        saved_state = dict(self._state)
        if set(current_state) != set(saved_state):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Child state keys changed before rollback."
            )
        with self._torch.no_grad():
            for name, value in current_state.items():
                saved = saved_state[name]
                if (
                    tuple(value.shape) != tuple(saved.shape)
                    or value.dtype != saved.dtype
                ):
                    raise Lfm2DynamicClosedLoopAdapterError(
                        "Child state shape/dtype changed before rollback."
                    )
                value.copy_(
                    saved.to(device=value.device, dtype=value.dtype)
                )
            for snapshot in self._gradients:
                parameter = snapshot.parameter
                parameter.grad = snapshot.gradient
                if snapshot.gradient is not None:
                    if id(parameter.grad) != snapshot.gradient_id:
                        raise Lfm2DynamicClosedLoopAdapterError(
                            "Gradient identity could not be restored."
                        )
                    parameter.grad.copy_(
                        snapshot.gradient_value.to(
                            device=parameter.grad.device,
                            dtype=parameter.grad.dtype,
                        )
                    )
        after = self._adapter.child_state_fingerprint()
        topology = self._adapter.child_topology_fingerprint()
        if (
            after != self.state_sha256_before
            or topology != self.topology_sha256_before
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Whole-child rollback was not exact."
            )
        self.active = False
        self._adapter._active_outer_transaction = None
        return TransactionReceipt(
            action="rollback",
            accepted=False,
            exact=True,
            state_sha256_before=self.state_sha256_before,
            state_sha256_after=after,
            metadata=_scalar_metadata(
                {
                    "topology_sha256": topology,
                    "whole_child_transaction": True,
                }
            ),
        )


class Lfm2DynamicClosedLoopAdapter:
    """Adapt a sealed LFM2 parent cache and live child to the generic compiler."""

    def __init__(
        self,
        *,
        source_model: Any,
        child_model: Any,
        parent_cache: Lfm2CircuitTraceCache,
        coordinate_map: DenseResidualCoordinateMap,
        initial_child_basis_sha256: str,
        selected_layer_indices: Sequence[int] | None = None,
        replay_config: Lfm2CircuitTraceCaptureConfig | None = None,
        input_device: Any | None = None,
        config: Lfm2DynamicClosedLoopAdapterConfig | None = None,
        macro_cell_attractor_gate: Lfm2MacroCellAttractorGate | None = None,
        tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None,
        tokenizer_bridge_trace_adapter: (
            Lfm2StrictTokenizerBridgeTraceAdapter | None
        ) = None,
    ) -> None:
        if not isinstance(parent_cache, Lfm2CircuitTraceCache):
            raise TypeError("parent_cache must be Lfm2CircuitTraceCache.")
        self.source_model = source_model
        self.child_model = child_model
        self.parent_cache = parent_cache.verify()
        self.coordinate_map = coordinate_map
        self.config = (
            Lfm2DynamicClosedLoopAdapterConfig()
            if config is None
            else config
        )
        if not isinstance(
            self.config, Lfm2DynamicClosedLoopAdapterConfig
        ):
            raise TypeError(
                "config must be Lfm2DynamicClosedLoopAdapterConfig."
            )
        if (
            macro_cell_attractor_gate is not None
            and not isinstance(
                macro_cell_attractor_gate,
                Lfm2MacroCellAttractorGate,
            )
        ):
            raise TypeError(
                "macro_cell_attractor_gate must be "
                "Lfm2MacroCellAttractorGate or None."
            )
        self.macro_cell_attractor_gate = macro_cell_attractor_gate
        self.input_device = input_device
        self.source_layers = _layers(source_model)
        self.child_layers = _layers(child_model)
        self._source_heterogeneous_layout = _heterogeneous_layout(
            source_model,
            "source model",
        )
        self._child_heterogeneous_layout = _heterogeneous_layout(
            child_model,
            "child model",
        )
        if (
            self._child_heterogeneous_layout is not None
            and self.config.replay_capture_mode
            == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Heterogeneous dynamic LFM2 children require canonical "
                "prefix-by-prefix replay."
            )
        if len(self.source_layers) != len(self.child_layers):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Source and child must retain the same layer count."
            )
        if selected_layer_indices is None:
            selected = tuple(range(len(self.child_layers)))
        else:
            selected = tuple(sorted(selected_layer_indices))
            if (
                not selected
                or len(selected) != len(set(selected))
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(self.child_layers)
                    for index in selected
                )
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "selected_layer_indices must be unique valid indices."
                )
        self._selected_layers = selected
        self._capture_layers = tuple(
            sorted({*selected, len(self.child_layers) - 1})
        )
        source_contracts = tuple(
            _inspect_runtime_layer(layer, self._source_heterogeneous_layout)
            for layer in self.source_layers
        )
        child_contracts = tuple(
            _inspect_runtime_layer(layer, self._child_heterogeneous_layout)
            for layer in self.child_layers
        )
        self._source_contracts = source_contracts
        self._child_contracts = child_contracts
        for index, (source, child) in enumerate(
            zip(source_contracts, child_contracts)
        ):
            if (
                source.layer_index != index
                or child.layer_index != index
                or source.kind != child.kind
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Source/child LFM2 layer ordering or operator kinds differ."
                )
        source_width = getattr(coordinate_map, "source_width", None)
        target_width = getattr(coordinate_map, "target_width", None)
        map_sha256 = _digest(
            getattr(coordinate_map, "map_sha256", None),
            "coordinate map_sha256",
        )
        if (
            isinstance(source_width, bool)
            or not isinstance(source_width, int)
            or isinstance(target_width, bool)
            or not isinstance(target_width, int)
            or source_width <= target_width
            or any(
                contract.hidden_size != source_width
                for contract in source_contracts
            )
            or any(
                contract.hidden_size != target_width
                for contract in child_contracts
            )
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Dense coordinate-map widths do not match source/child LFM2."
            )
        source_input_vocabulary, source_output_vocabulary = (
            _vocabulary_parameter_pair(
                source_model,
                "source model",
                require_tied=self._source_heterogeneous_layout is None,
            )
        )
        child_input_vocabulary, child_output_vocabulary = (
            _vocabulary_parameter_pair(
                child_model,
                "child model",
                require_tied=self._child_heterogeneous_layout is None,
            )
        )
        if (
            self._child_heterogeneous_layout is not None
            and child_input_vocabulary is child_output_vocabulary
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "A heterogeneous child must keep independent input and output "
                "vocabulary Parameters."
            )
        if (
            int(source_input_vocabulary.shape[1]) != source_width
            or int(source_output_vocabulary.shape[1]) != source_width
            or int(child_input_vocabulary.shape[1]) != target_width
            or int(child_output_vocabulary.shape[1]) != target_width
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Source/child vocabulary widths do not match the residual "
                "coordinate contract."
            )
        bridge_values = (
            tokenizer_bridge_artifact,
            tokenizer_bridge_trace_adapter,
        )
        if any(value is None for value in bridge_values) != all(
            value is None for value in bridge_values
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Tokenizer bridge artifact and trace adapter must be supplied "
                "together."
            )
        if tokenizer_bridge_artifact is None:
            if int(source_input_vocabulary.shape[0]) != int(
                child_input_vocabulary.shape[0]
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Different source/child vocabulary sizes require the "
                    "authenticated tokenizer bridge."
                )
        else:
            if not isinstance(
                tokenizer_bridge_artifact,
                StrictTokenizerBridgeArtifact,
            ) or not isinstance(
                tokenizer_bridge_trace_adapter,
                Lfm2StrictTokenizerBridgeTraceAdapter,
            ):
                raise TypeError(
                    "Tokenizer bridge inputs have the wrong authenticated types."
                )
            try:
                tokenizer_bridge_artifact.verify_files()
                tokenizer_bridge_trace_adapter.verify()
            except (ValueError, Lfm2StrictTokenizerBridgeTraceError) as exc:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Tokenizer bridge artifact/trace adapter failed verification."
                ) from exc
            if (
                tokenizer_bridge_trace_adapter.bridge_artifact_sha256
                != tokenizer_bridge_artifact.artifact_sha256
                or tokenizer_bridge_trace_adapter.transport_profile_fingerprint
                != tokenizer_bridge_artifact.profile_fingerprint
                or tokenizer_bridge_trace_adapter.donor_row_capacity
                != int(source_input_vocabulary.shape[0])
                or tokenizer_bridge_trace_adapter.target_row_capacity
                != int(child_input_vocabulary.shape[0])
                or tokenizer_bridge_trace_adapter.record_count
                != len(self.parent_cache)
                or (
                    self._child_heterogeneous_layout is not None
                    and self._child_heterogeneous_layout
                    .token_bridge_evidence_sha256
                    != tokenizer_bridge_artifact.artifact_sha256
                )
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Tokenizer bridge capacities, seals, child layout, or "
                    "record count differ."
                )
            for index, (record, batch) in enumerate(
                zip(
                    tokenizer_bridge_trace_adapter.records,
                    self.parent_cache.iter_batches(),
                    strict=True,
                )
            ):
                if (
                    record.ordinal != index
                    or record.donor_cache_input_sha256 != batch.input_sha256
                ):
                    raise Lfm2DynamicClosedLoopAdapterError(
                        "Tokenizer bridge trace records differ from the sealed "
                        "parent cache ordering or inputs."
                    )
        self.tokenizer_bridge_artifact = tokenizer_bridge_artifact
        self.tokenizer_bridge_trace_adapter = tokenizer_bridge_trace_adapter
        self._child_input_vocabulary_sha256 = _tensor_sha256(
            child_input_vocabulary
        )
        self.map_sha256 = map_sha256
        if (
            _digest(
                initial_child_basis_sha256,
                "initial_child_basis_sha256",
            )
            != map_sha256
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "The live child was not bound to this dense coordinate map."
            )
        source_identity = fingerprint_lfm2_parent(source_model)
        if source_identity.identity_sha256 != parent_cache.parent_fingerprint:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Source weights do not match the sealed parent cache."
            )
        manifest_capture = parent_cache.manifest.get("capture", {})
        if replay_config is None:
            replay_config = Lfm2CircuitTraceCaptureConfig(
                sample_last_tokens=int(
                    manifest_capture.get("sample_last_tokens", 8)
                ),
                max_decode_steps=int(
                    manifest_capture.get("max_decode_steps", 32)
                ),
                max_capture_bytes=int(
                    manifest_capture.get(
                        "max_capture_bytes", 512 * 1024 * 1024
                    )
                ),
                logit_rtol=float(
                    manifest_capture.get("logit_rtol", 1e-5)
                ),
                logit_atol=float(
                    manifest_capture.get("logit_atol", 1e-6)
                ),
                require_top1_agreement=bool(
                    manifest_capture.get(
                        "require_top1_agreement", True
                    )
                ),
                decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
            )
        if (
            not isinstance(replay_config, Lfm2CircuitTraceCaptureConfig)
            or replay_config.decode_trace_mode
            != FULL_PREFIX_CANONICAL_TRACE_MODE
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "replay_config must require canonical full-prefix replay."
            )
        self.replay_config = replay_config
        self._records = self._build_records()
        self._records_by_key = {
            record.key: record for record in self._records
        }
        self._short_conv_residual_assignment = self._build_short_conv_assignment()
        self._short_conv_assignment = self._short_conv_residual_assignment
        self._short_conv_assignments = MappingProxyType(
            {
                index: self._build_layer_short_conv_assignment(index)
                for index, contract in enumerate(self._source_contracts)
                if contract.kind == "short_conv"
            }
        )
        self._active_outer_transaction: _ChildModelTransaction | None = None
        self._split_transactions: dict[
            int, Lfm2StructuralSplitInstallTransaction
        ] = {}
        self._producer_payloads: dict[int, _ProducerPayload] = {}
        self._consumer_payloads: dict[int, _ConsumerPayload] = {}
        self._downstream_payloads: dict[int, _DownstreamPayload] = {}
        self._final_payload: _FinalNormPayload | None = None
        self._output_payload: _OutputInterfacePayload | None = None
        self._counters = {name: 0 for name in _FORBIDDEN_COUNTERS}

    def _build_records(self) -> tuple[TrajectoryRecord, ...]:
        records = []
        for index in range(len(self.parent_cache)):
            batch = self.parent_cache.load_batch(index)
            record_sha = canonical_sha256(
                {
                    "schema": "neural-forge-lfm2-sealed-trajectory-v1",
                    "cache_key": self.parent_cache.cache_key,
                    "batch_index": index,
                    "input_sha256": batch.input_sha256,
                    "path_id": LFM2_TRAJECTORY_PATH,
                }
            )
            records.append(
                TrajectoryRecord(
                    trajectory_id=f"batch-{index:06d}",
                    path_id=LFM2_TRAJECTORY_PATH,
                    record_sha256=record_sha,
                    payload=MappingProxyType({"batch_index": index}),
                )
            )
        if len(records) < 2:
            raise Lfm2DynamicClosedLoopAdapterError(
                "At least two sealed parent batches are required."
            )
        return tuple(records)

    def _coordinate_matrix(self) -> Any:
        torch = _torch()
        maximum = int(self.coordinate_map.source_width) * int(
            self.coordinate_map.target_width
        )
        callback = getattr(self.coordinate_map, "materialize_dense", None)
        if not callable(callback):
            raise Lfm2DynamicClosedLoopAdapterError(
                "The dense coordinate map cannot materialize its basis."
            )
        matrix = callback(
            max_elements=maximum,
            dtype=torch.float64,
            device="cpu",
        )
        if (
            not isinstance(matrix, torch.Tensor)
            or tuple(matrix.shape)
            != (
                int(self.coordinate_map.source_width),
                int(self.coordinate_map.target_width),
            )
            or not bool(torch.isfinite(matrix).all().item())
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Materialized dense coordinate basis is invalid."
            )
        return matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()

    def _build_short_conv_assignment(
        self,
    ) -> Lfm2SparseResidualAssignment:
        matrix = self._coordinate_matrix()
        source_width, target_width = map(int, matrix.shape)
        remaining = set(range(source_width))
        assignments = [-1] * source_width
        coefficients = [0.0] * source_width
        for target in range(target_width):
            ranked = sorted(
                remaining,
                key=lambda source: (
                    -abs(float(matrix[source, target].item())),
                    source,
                ),
            )
            if not ranked:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Dense basis cannot seed every short-conv target channel."
                )
            source = ranked[0]
            coefficient = float(matrix[source, target].item())
            if coefficient == 0.0:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Dense basis has a zero-loading target coordinate."
                )
            assignments[source] = target
            coefficients[source] = coefficient
            remaining.remove(source)
        for source in sorted(remaining):
            target = min(
                range(target_width),
                key=lambda index: (
                    -abs(float(matrix[source, index].item())),
                    index,
                ),
            )
            coefficient = float(matrix[source, target].item())
            if coefficient == 0.0:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Dense basis contains an unloaded source coordinate."
                )
            assignments[source] = target
            coefficients[source] = coefficient
        return Lfm2SparseResidualAssignment(
            target_width=target_width,
            source_to_target=tuple(assignments),
            coefficients=tuple(coefficients),
        )

    def _build_layer_short_conv_assignment(
        self,
        layer_index: int,
    ) -> Lfm2SparseResidualAssignment:
        source_layer = self.source_layers[layer_index]
        target_layer = self.child_layers[layer_index]
        source_operator = getattr(source_layer, "conv", None)
        target_operator = getattr(target_layer, "conv", None)
        source_weight = getattr(
            getattr(source_operator, "conv", None),
            "weight",
            None,
        )
        target_weight = getattr(
            getattr(target_operator, "conv", None),
            "weight",
            None,
        )
        if source_weight is None or target_weight is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                f"ShortConv layer {layer_index} has no live FIR weights."
            )
        source_channels = int(source_weight.shape[0])
        target_channels = int(target_weight.shape[0])
        if target_channels > source_channels:
            raise Lfm2DynamicClosedLoopAdapterError(
                f"ShortConv layer {layer_index} enlarges internal channels."
            )
        source_contract = self._source_contracts[layer_index]
        target_contract = self._child_contracts[layer_index]
        if (
            source_channels == source_contract.hidden_size
            and target_channels == target_contract.hidden_size
        ):
            # Preserve the exact historical native-LFM2 path, including the
            # dense-coordinate grouping, coefficients, and assignment digest.
            # A separate operator-local map is needed only when heterogeneous
            # physical channels are genuinely decoupled from residual width.
            return self._short_conv_residual_assignment
        if target_channels == source_channels:
            source_to_target = tuple(range(source_channels))
        elif self._source_heterogeneous_layout is None:
            try:
                plan = build_lfm2_short_conv_channel_plan(
                    source_layer,
                    target_hidden_size=target_channels,
                    mode="merge",
                )
            except Exception as exc:
                raise Lfm2DynamicClosedLoopAdapterError(
                    f"ShortConv layer {layer_index} could not build its "
                    "source-weight channel merge."
                ) from exc
            source_to_target = tuple(plan.source_to_target)
        else:
            # A dynamic heterogeneous parent already carries a sealed physical
            # channel order.  Preserve that order with a deterministic balanced
            # full-source merge rather than inferring policy from layer IDs.
            source_to_target = tuple(
                (source * target_channels) // source_channels
                for source in range(source_channels)
            )
        return Lfm2SparseResidualAssignment(
            target_width=target_channels,
            source_to_target=source_to_target,
            coefficients=(1.0,) * source_channels,
        )

    def _batch_indices(
        self, records: Sequence[TrajectoryRecord]
    ) -> tuple[int, ...]:
        indices = []
        for record in records:
            canonical = self._records_by_key.get(record.key)
            if (
                canonical is None
                or canonical.record_sha256 != record.record_sha256
                or not isinstance(record.payload, Mapping)
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "A trajectory is not from this sealed cache."
                )
            index = record.payload.get("batch_index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(self.parent_cache)
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "A trajectory carries an invalid sealed batch index."
                )
            indices.append(index)
        result = tuple(sorted(indices))
        if not result or len(result) != len(set(result)):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Trajectory batch indices must be nonempty and unique."
            )
        return result

    def _split(
        self,
        paired: Lfm2PairedReplay,
        fit_records: Sequence[TrajectoryRecord],
    ) -> Lfm2WholeBatchSplit:
        fit = frozenset(self._batch_indices(fit_records))
        validation = tuple(
            index for index in range(len(self.parent_cache)) if index not in fit
        )
        if not validation:
            raise Lfm2DynamicClosedLoopAdapterError(
                "The generic split left no held-out LFM2 batch."
            )
        return split_lfm2_replay_batches(
            paired,
            validation_batch_indices=validation,
        )

    def _project_parent(self, value: Any) -> Any:
        torch = _torch()
        callback = getattr(self.coordinate_map, "encode", None)
        if not callable(callback):
            callback = self.coordinate_map
        source = value.detach().to(device="cpu")
        result = callback(source)
        if (
            not isinstance(result, torch.Tensor)
            or tuple(result.shape[:-1]) != tuple(source.shape[:-1])
            or int(result.shape[-1]) != int(self.coordinate_map.target_width)
            or not bool(torch.isfinite(result).all().item())
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Dense coordinate projection returned invalid target rows."
            )
        return result.detach().to(device="cpu").contiguous()

    def _fit_norm(
        self, target_norm: Any
    ) -> Callable[[Lfm2NormFitRequest], Lfm2NormMaterializer]:
        torch = _torch()
        epsilon = _rms_epsilon(target_norm, "target RMSNorm")
        parameter = getattr(target_norm, "weight", None)
        if (
            not isinstance(parameter, torch.nn.Parameter)
            or parameter.ndim != 1
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Target RMSNorm has no vector Parameter."
            )

        def callback(
            request: Lfm2NormFitRequest,
        ) -> Lfm2NormMaterializer:
            if not request.fit_inputs or (
                len(request.fit_inputs) != len(request.fit_targets)
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    f"{request.role} has no aligned fit rows."
                )
            inputs = torch.cat(
                [
                    value.detach()
                    .to(device="cpu", dtype=torch.float64)
                    .reshape(-1, int(value.shape[-1]))
                    for value in request.fit_inputs
                ],
                dim=0,
            )
            targets = torch.cat(
                [
                    value.detach()
                    .to(device="cpu", dtype=torch.float64)
                    .reshape(-1, int(value.shape[-1]))
                    for value in request.fit_targets
                ],
                dim=0,
            )
            validation_inputs = (
                None
                if not request.validation_inputs
                else torch.cat(
                    [
                        value.detach()
                        .to(device="cpu", dtype=torch.float64)
                        .reshape(-1, int(value.shape[-1]))
                        for value in request.validation_inputs
                    ],
                    dim=0,
                )
            )
            validation_targets = (
                None
                if not request.validation_targets
                else torch.cat(
                    [
                        value.detach()
                        .to(device="cpu", dtype=torch.float64)
                        .reshape(-1, int(value.shape[-1]))
                        for value in request.validation_targets
                    ],
                    dim=0,
                )
            )
            fit = fit_diagonal_norm(
                inputs,
                targets,
                kind="rms_norm",
                epsilon=epsilon,
                has_bias=False,
                ridge=self.config.norm_ridge,
                validation_input_rows=validation_inputs,
                validation_target_rows=validation_targets,
            )
            fitted_weight = (
                fit.weight.detach()
                .to(device="cpu", dtype=parameter.dtype)
                .contiguous()
                .clone()
            )

            def materialize(value: Any) -> Any:
                source = value.detach().to(device="cpu").contiguous()
                normalized = source.float()
                normalized = normalized * torch.rsqrt(
                    normalized.square().mean(dim=-1, keepdim=True)
                    + epsilon
                )
                return (
                    fitted_weight.to(dtype=source.dtype)
                    * normalized.to(dtype=source.dtype)
                ).contiguous()

            report = {
                "schema": "neural-forge-lfm2-installable-rmsnorm-fit-v1",
                "role": request.role,
                "epsilon": epsilon,
                "ridge": self.config.norm_ridge,
                "fitted_weight": fitted_weight,
                "fitted_weight_sha256": _tensor_sha256(fitted_weight),
                "fit_relative_rmse": float(
                    fit.train_metrics.relative_rmse
                ),
                "validation_relative_rmse": (
                    None
                    if fit.validation_metrics is None
                    else float(fit.validation_metrics.relative_rmse)
                ),
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            return Lfm2NormMaterializer(
                role=request.role,
                materialize_callback=materialize,
                report=report,
            )

        return callback

    def _materializer_weight(
        self, materializer: Lfm2NormMaterializer
    ) -> Any:
        torch = _torch()
        weight = materializer.report.get("fitted_weight")
        if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Installable norm materializer omitted its fitted weight."
            )
        return weight.detach().to(device="cpu").contiguous().clone()

    def _norm_report_sha256(
        self, materializer: Lfm2NormMaterializer
    ) -> str:
        return canonical_sha256(
            {
                "schema": materializer.report.get("schema"),
                "role": materializer.role,
                "epsilon": materializer.report.get("epsilon"),
                "ridge": materializer.report.get("ridge"),
                "fitted_weight_sha256": materializer.report.get(
                    "fitted_weight_sha256"
                ),
                "fit_relative_rmse": materializer.report.get(
                    "fit_relative_rmse"
                ),
                "validation_relative_rmse": materializer.report.get(
                    "validation_relative_rmse"
                ),
            }
        )

    def _copy_norm_weight(self, module: Any, weight: Any) -> None:
        torch = _torch()
        parameter = getattr(module, "weight", None)
        if (
            not isinstance(parameter, torch.nn.Parameter)
            or tuple(parameter.shape) != tuple(weight.shape)
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Compiled RMSNorm weight does not match the target."
            )
        parameter_id = id(parameter)
        gradient = parameter.grad
        gradient_id = None if gradient is None else id(gradient)
        with torch.no_grad():
            parameter.copy_(
                weight.to(device=parameter.device, dtype=parameter.dtype)
            )
        if (
            id(parameter) != parameter_id
            or (None if parameter.grad is None else id(parameter.grad))
            != gradient_id
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "RMSNorm installation changed Parameter/gradient identity."
            )

    def _attention_norm_epsilon(self, attention: Any) -> float:
        for name in ("q_layernorm", "k_layernorm"):
            module = getattr(attention, name, None)
            if module is not None:
                return _rms_epsilon(module, f"attention.{name}")
        return 1e-6

    def _operator_kind(self, layer_index: int) -> str:
        return self._child_contracts[layer_index].kind

    def _heldout_layer_metrics(
        self,
        paired: Lfm2PairedReplay,
        *,
        layer_index: int,
        batch_indices: Sequence[int],
    ) -> FitMetrics:
        torch = _torch()
        predictions = []
        targets = []
        for index in batch_indices:
            batch = paired.load_batch(index)
            for phase in batch.phase_order:
                base = f"{phase}.layer.{layer_index:03d}."
                predictions.append(
                    batch.child.tensors[
                        base + "residual.post_layer"
                    ].detach()
                    .to(device="cpu", dtype=torch.float64)
                    .reshape(
                        -1,
                        int(self.coordinate_map.target_width),
                    )
                )
                targets.append(
                    self._project_parent(
                        batch.parent.tensors[
                            base + "residual.post_layer"
                        ]
                    )
                    .to(dtype=torch.float64)
                    .reshape(
                        -1,
                        int(self.coordinate_map.target_width),
                    )
                )
        if not predictions:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Held-out layer validation has no endpoint rows."
            )
        return measure_fit(
            torch.cat(predictions, dim=0),
            torch.cat(targets, dim=0),
        )

    def _validation_decision(
        self,
        *,
        scope: FoldScope,
        records: Sequence[TrajectoryRecord],
        baseline: FitMetrics,
        candidate: FitMetrics,
        minimum_improvement: float,
        endpoint: str,
    ) -> ValidationDecision:
        improvement = (
            float(baseline.relative_rmse)
            - float(candidate.relative_rmse)
        )
        accepted = bool(
            candidate.finite
            and improvement + self.config.validation_tolerance
            >= minimum_improvement
        )
        reason = (
            f"held-out {endpoint} improved/non-regressed"
            if accepted
            else (
                f"held-out {endpoint} relative-RMSE improvement "
                f"{improvement:.9g} is below {minimum_improvement:.9g}"
            )
        )
        metrics: dict[str, JsonScalar] = {
            "baseline_relative_rmse": float(
                baseline.relative_rmse
            ),
            "candidate_relative_rmse": float(
                candidate.relative_rmse
            ),
            "relative_rmse_improvement": improvement,
            "minimum_required_improvement": minimum_improvement,
            "validation_tolerance": self.config.validation_tolerance,
            "candidate_mean_cosine_similarity": float(
                candidate.mean_cosine_similarity
            ),
            "candidate_finite": bool(candidate.finite),
            "heldout_endpoint_compared_to_projected_parent": True,
        }
        return ValidationDecision(
            scope=scope,
            accepted=accepted,
            validation_record_keys=_record_keys(records),
            reason=reason,
            metrics=_scalar_metadata(metrics),
        )

    def _receipt(
        self,
        *,
        scope: FoldScope,
        stage: CompileStage,
        artifact_sha256: str,
        state_before: str,
        topology_before: str,
        metadata: Mapping[str, JsonScalar],
    ) -> InstallReceipt:
        state_after = self.child_state_fingerprint()
        topology_after = self.child_topology_fingerprint()
        return InstallReceipt(
            scope=scope,
            stage=stage,
            artifact_sha256=artifact_sha256,
            accepted=True,
            state_sha256_before=state_before,
            state_sha256_after=state_after,
            topology_preserved=(
                topology_before == topology_after
            ),
            metadata=_scalar_metadata(metadata),
        )

    def _rollback_open_split_transactions(self) -> None:
        for index, transaction in tuple(
            sorted(self._split_transactions.items())
        ):
            if transaction.active:
                report = rollback_lfm2_structural_split_install(
                    transaction
                )
                if not report.accepted:
                    raise Lfm2DynamicClosedLoopAdapterError(
                        f"Layer {index} split rollback was not accepted."
                    )
            self._split_transactions.pop(index, None)

    # -- Generic adapter discovery/evidence contract ---------------------

    def layer_indices(self, audit: ClosedLoopAudit) -> Sequence[int]:
        del audit
        return self._selected_layers

    def parent_trajectory_records(
        self, audit: ClosedLoopAudit
    ) -> Sequence[TrajectoryRecord]:
        del audit
        self.parent_cache.verify()
        return self._records

    def child_state_fingerprint(self) -> str:
        return _model_state_sha256(self.child_model)

    def child_topology_fingerprint(self) -> str:
        return _model_topology_sha256(self.child_model)

    def audit_counters(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._counters))

    def begin_child_transaction(
        self, audit: ClosedLoopAudit
    ) -> _ChildModelTransaction:
        del audit
        if (
            self._active_outer_transaction is not None
            and self._active_outer_transaction.active
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "A child transaction is already active."
            )
        transaction = _ChildModelTransaction(self)
        self._active_outer_transaction = transaction
        return transaction

    def bridge_parent_endpoints(
        self,
        scope: FoldScope,
        records: Sequence[TrajectoryRecord],
        audit: ClosedLoopAudit,
    ) -> ParentEndpointBatch:
        del audit
        indices = self._batch_indices(records)
        return ParentEndpointBatch(
            scope=scope,
            record_keys=_record_keys(records),
            batch_sha256=canonical_sha256(
                {
                    "schema": (
                        "neural-forge-lfm2-sealed-endpoint-selection-v1"
                    ),
                    "cache_key": self.parent_cache.cache_key,
                    "scope": scope.as_dict(),
                    "batch_indices": list(indices),
                    "record_keys": [
                        list(value) for value in _record_keys(records)
                    ],
                }
            ),
            payload=_EndpointPayload(
                cache_key=self.parent_cache.cache_key,
                batch_indices=indices,
            ),
        )

    def replay_and_capture_child(
        self,
        scope: FoldScope,
        point: CapturePoint,
        records: Sequence[TrajectoryRecord],
        audit: ClosedLoopAudit,
    ) -> ChildTrajectoryCapture:
        del audit
        self._batch_indices(records)
        sealed_batches = (
            tuple(self.parent_cache.iter_batches())
            if self.tokenizer_bridge_trace_adapter is None
            else self.tokenizer_bridge_trace_adapter.child_replay_batches()
        )
        replay = capture_lfm2_child_replay(
            self.child_model,
            sealed_batches,
            selected_layer_indices=self._capture_layers,
            config=self.replay_config,
            input_device=self.input_device,
            replay_capture_mode=self.config.replay_capture_mode,
        )
        if len(replay) != len(self.parent_cache):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Child capture did not replay every sealed parent batch."
            )
        runtime = replay.metadata.get("runtime", {})
        forwards = replay.metadata.get("forward_counts", {})
        selection = replay.metadata.get("selection", {})
        benchmark = replay.metadata.get("replay_benchmark", {})
        if (
            runtime.get("parent_forward_count") != 0
            or runtime.get("tokenizer_accessed") is not False
            or forwards.get("parent") != 0
            or selection.get("replay_capture_mode")
            != self.config.replay_capture_mode
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Child replay crossed a parent/tokenizer/capture-mode tripwire."
            )
        actual_child_forwards = benchmark.get("actual_child_forwards")
        canonical_prefix_child_forwards = benchmark.get(
            "canonical_prefix_child_forwards"
        )
        saved_child_forwards = benchmark.get("saved_child_forwards")
        if (
            isinstance(actual_child_forwards, bool)
            or not isinstance(actual_child_forwards, int)
            or isinstance(canonical_prefix_child_forwards, bool)
            or not isinstance(canonical_prefix_child_forwards, int)
            or isinstance(saved_child_forwards, bool)
            or not isinstance(saved_child_forwards, int)
            or actual_child_forwards <= 0
            or canonical_prefix_child_forwards < actual_child_forwards
            or saved_child_forwards
            != canonical_prefix_child_forwards - actual_child_forwards
            or forwards.get("child") != actual_child_forwards
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Child replay forward benchmark metadata is invalid."
            )
        if (
            self.config.replay_capture_mode
            == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
            and actual_child_forwards != len(self.parent_cache)
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Single-pass child replay did not use one forward per batch."
            )
        causal_contract_sha256 = None
        if (
            self.config.replay_capture_mode
            == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        ):
            first_capture = replay.load_batch(0).capture
            causal_contract = first_capture.get(
                "causal_equivalence_contract"
            )
            if (
                not isinstance(causal_contract, Mapping)
                or not isinstance(
                    causal_contract.get("contract_sha256"), str
                )
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Single-pass child replay omitted its causal contract."
                )
            causal_contract_sha256 = str(
                causal_contract["contract_sha256"]
            )
            for batch in replay.iter_batches():
                batch_contract = batch.capture.get(
                    "causal_equivalence_contract"
                )
                if (
                    not isinstance(batch_contract, Mapping)
                    or batch_contract.get("contract_sha256")
                    != causal_contract_sha256
                ):
                    raise Lfm2DynamicClosedLoopAdapterError(
                        "Single-pass child replay causal contracts differ "
                        "between batches."
                    )
        paired = pair_lfm2_parent_child_replay(
            self.parent_cache,
            replay,
            tokenizer_bridge_trace_adapter=(
                self.tokenizer_bridge_trace_adapter
            ),
        )
        payload = _CapturePayload(
            replay_sha256=replay.replay_sha256,
            paired=paired,
            replay_capture_mode=self.config.replay_capture_mode,
            actual_child_forwards=actual_child_forwards,
            canonical_prefix_child_forwards=(
                canonical_prefix_child_forwards
            ),
            saved_child_forwards=saved_child_forwards,
            causal_contract_sha256=causal_contract_sha256,
        )
        return ChildTrajectoryCapture(
            scope=scope,
            point=point,
            record_keys=_record_keys(records),
            capture_sha256=canonical_sha256(
                {
                    "schema": "neural-forge-lfm2-child-capture-envelope-v1",
                    "scope": scope.as_dict(),
                    "point": point.value,
                    "record_keys": [
                        list(value) for value in _record_keys(records)
                    ],
                    "replay_sha256": replay.replay_sha256,
                    "all_batches_replayed": True,
                    "canonical_causal_phase_schema": True,
                    "replay_capture_mode": (
                        self.config.replay_capture_mode
                    ),
                    "actual_child_forwards": actual_child_forwards,
                    "canonical_prefix_child_forwards": (
                        canonical_prefix_child_forwards
                    ),
                    "saved_child_forwards": saved_child_forwards,
                    "causal_contract_sha256": (
                        causal_contract_sha256
                    ),
                }
            ),
            payload=payload,
        )

    def _capture_payload(
        self, capture: ChildTrajectoryCapture
    ) -> _CapturePayload:
        payload = capture.payload
        if (
            not isinstance(payload, _CapturePayload)
            or not payload.all_batches_replayed
            or not payload.canonical_causal_phase_schema
            or payload.replay_capture_mode
            != self.config.replay_capture_mode
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Compile stage did not receive a canonical all-batch child replay."
            )
        return payload

    def _child_vocabulary_parameters(self) -> tuple[Any, Any]:
        return _vocabulary_parameter_pair(
            self.child_model,
            "child model",
            require_tied=self._child_heterogeneous_layout is None,
        )

    def _require_heterogeneous_input_unchanged(self) -> None:
        if self._child_heterogeneous_layout is None:
            return
        child_input, _child_output = self._child_vocabulary_parameters()
        if _tensor_sha256(child_input) != self._child_input_vocabulary_sha256:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Untied output-interface work changed the child input embedding."
            )

    def _source_bound_output_weight(self) -> tuple[Any, str]:
        """Authenticate the live output head against the original dense fold."""

        torch = _torch()
        source_input, source_output = _vocabulary_parameter_pair(
            self.source_model,
            "source model",
            require_tied=self._source_heterogeneous_layout is None,
        )
        child_input, child_output = _vocabulary_parameter_pair(
            self.child_model,
            "child model",
            require_tied=self._child_heterogeneous_layout is None,
        )
        source_input_rows = (
            source_input.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        source_output_rows = (
            source_output.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        transport_report_sha256 = None
        if self.tokenizer_bridge_artifact is None:
            callback = getattr(
                self.coordinate_map, "fold_consumer_weight", None
            )
            if not callable(callback):
                callback = getattr(self.coordinate_map, "encode", None)
            if not callable(callback):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "The coordinate map cannot reproduce the folded vocabulary "
                    "consumer."
                )
            projected_input = callback(source_input_rows)
            projected = callback(source_output_rows)
        else:
            try:
                transported = compile_strict_vocabulary_operators(
                    source_input_rows,
                    source_output_rows,
                    self.tokenizer_bridge_artifact,
                    self.coordinate_map,
                    self.coordinate_map,
                    require_tied_target=False,
                )
            except StrictTokenizerBridgeError as exc:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Strict vocabulary operators could not be reproduced from "
                    "the sealed source, bridge, and coordinate map."
                ) from exc
            projected_input = transported.embedding_weight
            projected = transported.lm_head_weight
            transport_report_sha256 = transported.report["report_sha256"]
        if (
            not isinstance(projected_input, torch.Tensor)
            or not isinstance(projected, torch.Tensor)
            or tuple(projected_input.shape) != tuple(child_input.shape)
            or tuple(projected.shape) != tuple(child_output.shape)
            or not bool(torch.isfinite(projected_input.detach()).all().item())
            or not bool(torch.isfinite(projected.detach()).all().item())
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "The source vocabulary projections are malformed."
            )
        projected_input = projected_input.detach().to(
            device="cpu",
            dtype=child_input.dtype,
        ).contiguous()
        projected = (
            projected.detach()
            .to(device="cpu", dtype=child_output.dtype)
            .contiguous()
        )
        live_input = child_input.detach().to(device="cpu").contiguous()
        live = child_output.detach().to(device="cpu").contiguous()
        if not torch.equal(projected_input, live_input) or not torch.equal(
            projected,
            live,
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "The live vocabulary interface is not the exact source/bridge/"
                "dense-basis fold."
            )
        binding = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-source-bound-output-interface-v2"
                ),
                "source_input_vocabulary_sha256": _tensor_sha256(
                    source_input_rows
                ),
                "source_output_vocabulary_sha256": _tensor_sha256(
                    source_output_rows
                ),
                "map_sha256": self.map_sha256,
                "tokenizer_bridge_artifact_sha256": (
                    None
                    if self.tokenizer_bridge_artifact is None
                    else self.tokenizer_bridge_artifact.artifact_sha256
                ),
                "vocabulary_transport_report_sha256": transport_report_sha256,
                "projected_input_vocabulary_sha256": _tensor_sha256(
                    projected_input
                ),
                "projected_output_vocabulary_sha256": _tensor_sha256(projected),
                "vocabulary_size": int(projected.shape[0]),
                "target_width": int(projected.shape[1]),
                "target_interface_tied": child_input is child_output,
                "source_weights_only": True,
                "parent_forward_calls": 0,
            }
        )
        return live.clone(), binding

    def _output_hidden_rows(
        self,
        batch: Any,
        phase: str,
    ) -> Any:
        value = batch.child.tensors[
            (
                f"{phase}.layer."
                f"{len(self.child_layers) - 1:03d}.residual.post_layer"
            )
        ]
        normalized = _analytical_rms_norm(
            value,
            _final_norm(self.child_model),
            "child output-interface final norm",
        )
        return normalized.reshape(
            -1, int(self.coordinate_map.target_width)
        ).contiguous()

    def _known_sequence_targets(
        self,
        batch: Any,
        phase: str,
        *,
        expected_rows: int,
    ) -> Any:
        """Return next-token IDs for rows whose successor is in the cache."""

        torch = _torch()
        metadata = batch.child.record.get("phase_rows")
        if not isinstance(metadata, list):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Child output replay has no phase-row metadata."
            )
        matches = [
            value for value in metadata if value.get("phase") == phase
        ]
        if len(matches) != 1:
            raise Lfm2DynamicClosedLoopAdapterError(
                f"Child output replay has ambiguous metadata for {phase!r}."
            )
        row = matches[0]
        sample_start = int(row["sample_start"])
        sample_count = int(row["sample_count"])
        input_ids = batch.child.input_ids.detach().to(
            device="cpu", dtype=torch.long
        )
        forced_ids = batch.child.forced_decode_ids.detach().to(
            device="cpu", dtype=torch.long
        )
        trajectory = torch.cat((input_ids, forced_ids), dim=1)
        targets = torch.full(
            (int(input_ids.shape[0]), sample_count),
            -1,
            dtype=torch.long,
        )
        for offset in range(sample_count):
            next_position = sample_start + offset + 1
            if next_position < int(trajectory.shape[1]):
                targets[:, offset] = trajectory[:, next_position]
        result = targets.reshape(-1).contiguous()
        if int(result.numel()) != expected_rows:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Known sequence targets do not align with output rows."
            )
        return result

    def _iter_output_blocks(
        self,
        paired: Lfm2PairedReplay,
        batch_indices: Sequence[int],
        *,
        materialized_weight: Any | None = None,
    ):
        torch = _torch()
        _child_input, child_output = _vocabulary_parameter_pair(
            self.child_model,
            "child model",
            require_tied=self._child_heterogeneous_layout is None,
        )
        vocabulary_size = int(child_output.shape[0])
        if materialized_weight is not None and (
            not isinstance(materialized_weight, torch.Tensor)
            or materialized_weight.ndim != 2
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Offline output weight must be a rank-2 tensor."
            )
        runtime_weight = (
            None
            if materialized_weight is None
            else materialized_weight.detach().to(
                device="cpu", dtype=torch.float64
            )
        )
        for index in batch_indices:
            batch = paired.load_batch(index)
            for phase in batch.phase_order:
                hidden = self._output_hidden_rows(batch, phase)
                if self.tokenizer_bridge_trace_adapter is not None:
                    projected_parent = project_lfm2_paired_parent_logits(
                        paired,
                        batch_index=index,
                        phase=phase,
                        top_k=5,
                    )
                    parent_logits = projected_parent.target_logits
                else:
                    parent_logits = (
                        batch.parent.tensors[f"{phase}.logits"]
                        .detach()
                        .to(device="cpu")
                    )
                parent = parent_logits.reshape(-1, vocabulary_size).contiguous()
                if materialized_weight is None:
                    candidate = (
                        batch.child.tensors[f"{phase}.logits"]
                        .detach()
                        .to(device="cpu")
                        .reshape(-1, vocabulary_size)
                        .contiguous()
                    )
                else:
                    candidate = (
                        hidden.to(dtype=torch.float64)
                        @ runtime_weight.transpose(0, 1)
                    ).contiguous()
                if (
                    tuple(parent.shape) != tuple(candidate.shape)
                    or int(hidden.shape[0]) != int(parent.shape[0])
                    or int(parent.shape[1]) != vocabulary_size
                    or not bool(torch.isfinite(parent).all().item())
                    or not bool(torch.isfinite(candidate).all().item())
                ):
                    raise Lfm2DynamicClosedLoopAdapterError(
                        "Parent/child output-interface rows are malformed or "
                        "misaligned."
                    )
                targets = self._known_sequence_targets(
                    batch,
                    phase,
                    expected_rows=int(parent.shape[0]),
                )
                yield hidden, parent, candidate, targets

    def _eos_token_id(self, vocabulary_size: int) -> int | None:
        value = getattr(
            getattr(self.child_model, "config", None),
            "eos_token_id",
            None,
        )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ):
            value = next(
                (
                    item
                    for item in value
                    if isinstance(item, int)
                    and not isinstance(item, bool)
                    and 0 <= item < vocabulary_size
                ),
                None,
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= vocabulary_size
        ):
            return None
        return value

    def _output_metrics(
        self,
        paired: Lfm2PairedReplay,
        batch_indices: Sequence[int],
        *,
        materialized_weight: Any | None = None,
    ) -> dict[str, JsonScalar]:
        """Measure full-vocabulary teacher-forced output behavior."""

        torch = _torch()
        rows = 0
        vocabulary_size = 0
        sums = {
            "kl": 0.0,
            "cross_entropy": 0.0,
            "parent_entropy": 0.0,
            "winner_nll": 0.0,
            "winner_margin": 0.0,
            "winner_margin_positive": 0.0,
            "top1_agreement": 0.0,
            "logit_squared_error": 0.0,
            "parent_logit_square": 0.0,
            "eos_margin_squared_error": 0.0,
        }
        sequence_nll = 0.0
        sequence_rows = 0
        eos_rows = 0
        for _hidden, parent, candidate, targets in (
            self._iter_output_blocks(
                paired,
                batch_indices,
                materialized_weight=materialized_weight,
            )
        ):
            vocabulary_size = int(parent.shape[1])
            eos_token_id = self._eos_token_id(vocabulary_size)
            for start in range(
                0,
                int(parent.shape[0]),
                self.config.output_metric_chunk_rows,
            ):
                stop = min(
                    int(parent.shape[0]),
                    start + self.config.output_metric_chunk_rows,
                )
                parent_chunk = parent[start:stop].to(
                    dtype=torch.float64
                )
                candidate_chunk = candidate[start:stop].to(
                    dtype=torch.float64
                )
                parent_log_probability = torch.log_softmax(
                    parent_chunk, dim=-1
                )
                candidate_log_probability = torch.log_softmax(
                    candidate_chunk, dim=-1
                )
                parent_probability = parent_log_probability.exp()
                chunk_rows = int(parent_chunk.shape[0])
                rows += chunk_rows
                sums["kl"] += float(
                    (
                        parent_probability
                        * (
                            parent_log_probability
                            - candidate_log_probability
                        )
                    )
                    .sum()
                    .item()
                )
                sums["cross_entropy"] += float(
                    (
                        -parent_probability
                        * candidate_log_probability
                    )
                    .sum()
                    .item()
                )
                sums["parent_entropy"] += float(
                    (
                        -parent_probability
                        * parent_log_probability
                    )
                    .sum()
                    .item()
                )
                parent_winner = parent_chunk.argmax(
                    dim=-1, keepdim=True
                )
                candidate_winner = candidate_chunk.argmax(
                    dim=-1, keepdim=True
                )
                sums["top1_agreement"] += float(
                    (parent_winner == candidate_winner).sum().item()
                )
                winner_log_probability = (
                    candidate_log_probability.gather(
                        1, parent_winner
                    ).squeeze(1)
                )
                sums["winner_nll"] += float(
                    (-winner_log_probability).sum().item()
                )
                top_two = torch.topk(
                    candidate_chunk, k=2, dim=-1
                )
                best_is_parent = (
                    top_two.indices[:, :1] == parent_winner
                ).squeeze(1)
                competitor = torch.where(
                    best_is_parent,
                    top_two.values[:, 1],
                    top_two.values[:, 0],
                )
                winner_logit = candidate_chunk.gather(
                    1, parent_winner
                ).squeeze(1)
                margin = winner_logit - competitor
                sums["winner_margin"] += float(margin.sum().item())
                sums["winner_margin_positive"] += float(
                    (margin > 0.0).sum().item()
                )
                difference = candidate_chunk - parent_chunk
                sums["logit_squared_error"] += float(
                    difference.square().sum().item()
                )
                sums["parent_logit_square"] += float(
                    parent_chunk.square().sum().item()
                )
                target_chunk = targets[start:stop]
                known = target_chunk >= 0
                if bool(known.any().item()):
                    known_targets = target_chunk[known].reshape(-1, 1)
                    sequence_nll += float(
                        (
                            -candidate_log_probability[known].gather(
                                1, known_targets
                            )
                        )
                        .sum()
                        .item()
                    )
                    sequence_rows += int(known.sum().item())
                if eos_token_id is not None:
                    eos = torch.full_like(
                        parent_winner, eos_token_id
                    )
                    parent_top_two = torch.topk(
                        parent_chunk, k=2, dim=-1
                    )
                    parent_best_is_eos = (
                        parent_top_two.indices[:, :1] == eos
                    ).squeeze(1)
                    parent_competitor = torch.where(
                        parent_best_is_eos,
                        parent_top_two.values[:, 1],
                        parent_top_two.values[:, 0],
                    )
                    candidate_best_is_eos = (
                        top_two.indices[:, :1] == eos
                    ).squeeze(1)
                    candidate_competitor = torch.where(
                        candidate_best_is_eos,
                        top_two.values[:, 1],
                        top_two.values[:, 0],
                    )
                    parent_eos_margin = (
                        parent_chunk[:, eos_token_id]
                        - parent_competitor
                    )
                    candidate_eos_margin = (
                        candidate_chunk[:, eos_token_id]
                        - candidate_competitor
                    )
                    sums["eos_margin_squared_error"] += float(
                        (
                            candidate_eos_margin
                            - parent_eos_margin
                        )
                        .square()
                        .sum()
                        .item()
                    )
                    eos_rows += chunk_rows
        if rows <= 0 or vocabulary_size <= 1:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface validation has no full-vocabulary rows."
            )
        target_square = max(
            sums["parent_logit_square"],
            torch.finfo(torch.float64).tiny,
        )
        metrics: dict[str, JsonScalar] = {
            "rows": rows,
            "vocabulary_size": vocabulary_size,
            "full_vocabulary_parent_kl": sums["kl"] / rows,
            "full_vocabulary_parent_distribution_nll": (
                sums["cross_entropy"] / rows
            ),
            "full_vocabulary_parent_entropy": (
                sums["parent_entropy"] / rows
            ),
            "parent_winner_nll": sums["winner_nll"] / rows,
            "parent_winner_top1_agreement": (
                sums["top1_agreement"] / rows
            ),
            "parent_winner_mean_margin": (
                sums["winner_margin"] / rows
            ),
            "parent_winner_positive_margin_fraction": (
                sums["winner_margin_positive"] / rows
            ),
            "full_vocabulary_logit_relative_rmse": math.sqrt(
                sums["logit_squared_error"] / target_square
            ),
            "known_sequence_token_nll": (
                0.0
                if sequence_rows == 0
                else sequence_nll / sequence_rows
            ),
            "known_sequence_token_rows": sequence_rows,
            "eos_token_id_available": eos_rows > 0,
            "eos_margin_rmse": (
                0.0
                if eos_rows == 0
                else math.sqrt(
                    sums["eos_margin_squared_error"] / eos_rows
                )
            ),
            "finite": True,
            "teacher_forced": True,
            "full_vocabulary_not_sparse_anchor_only": True,
        }
        if not all(
            not isinstance(value, float) or math.isfinite(value)
            for value in metrics.values()
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface metrics are non-finite."
            )
        return metrics

    def _offline_live_output_comparison(
        self,
        baseline: Lfm2PairedReplay,
        candidate: Lfm2PairedReplay,
        batch_indices: Sequence[int],
        candidate_weight: Any,
    ) -> dict[str, JsonScalar]:
        torch = _torch()
        squared_error = 0.0
        target_square = 0.0
        rows = 0
        top1_agreement = 0
        baseline_blocks = self._iter_output_blocks(
            baseline,
            batch_indices,
            materialized_weight=candidate_weight,
        )
        candidate_blocks = self._iter_output_blocks(
            candidate,
            batch_indices,
        )
        for baseline_block, candidate_block in zip(
            baseline_blocks, candidate_blocks, strict=True
        ):
            offline = baseline_block[2].to(dtype=torch.float64)
            live = candidate_block[2].to(dtype=torch.float64)
            if tuple(offline.shape) != tuple(live.shape):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Offline/live output logits changed topology."
                )
            difference = offline - live
            squared_error += float(difference.square().sum().item())
            target_square += float(live.square().sum().item())
            rows += int(live.shape[0])
            top1_agreement += int(
                (
                    offline.argmax(dim=-1)
                    == live.argmax(dim=-1)
                )
                .sum()
                .item()
            )
        if rows <= 0:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Offline/live tied-feedback comparison has no rows."
            )
        return {
            "rows": rows,
            "relative_rmse": math.sqrt(
                squared_error
                / max(target_square, torch.finfo(torch.float64).tiny)
            ),
            "top1_agreement": top1_agreement / rows,
            "post_install_replayed_through_input_embedding": True,
        }

    # -- Per-layer producer / consumer / downstream ---------------------

    def compile_producer(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage:
        del audit
        if scope.kind != "layer" or scope.layer_index is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Producer scope must identify one LFM2 layer."
            )
        if not isinstance(
            parent_endpoints.payload, _EndpointPayload
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Producer parent endpoint envelope is invalid."
            )
        layer_index = scope.layer_index
        attractor_baseline = (
            None
            if self.macro_cell_attractor_gate is None
            else self.macro_cell_attractor_gate.capture(
                self.child_model,
                phase=f"pre-macro-cell-layer-{layer_index:06d}",
            )
        )
        capture = self._capture_payload(child_capture)
        split = self._split(capture.paired, fit_records)
        target_layer = self.child_layers[layer_index]
        source_layer = self.source_layers[layer_index]
        operator_kind = self._operator_kind(layer_index)
        operator_cases = build_lfm2_operator_producer_cases(
            split,
            layer_index=layer_index,
            operator_kind=operator_kind,
            coordinate_map_callback=self._project_parent,
            norm_fit_callback=self._fit_norm(
                target_layer.operator_norm
            ),
        )
        baseline = self._heldout_layer_metrics(
            capture.paired,
            layer_index=layer_index,
            batch_indices=split.validation_batch_indices,
        )
        if operator_kind == "full_attention":
            source = source_layer.self_attn
            target = target_layer.self_attn
            target_head_dim = getattr(
                self._child_contracts[layer_index].operator,
                "head_dim",
                None,
            )
            if isinstance(target_head_dim, bool) or not isinstance(
                target_head_dim,
                int,
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    f"Attention layer {layer_index} has no authenticated head_dim."
                )
            compiled = compile_lfm2_structural_attention_producer_fold(
                source_query_weight=source.q_proj.weight,
                source_key_weight=source.k_proj.weight,
                source_value_weight=source.v_proj.weight,
                source_output_weight=source.out_proj.weight,
                target_query_weight_template=target.q_proj.weight,
                target_key_weight_template=target.k_proj.weight,
                target_value_weight_template=target.v_proj.weight,
                target_output_weight_template=target.out_proj.weight,
                cases=operator_cases.fit_cases,
                head_dim=target_head_dim,
                query_norm_weight=getattr(
                    getattr(target, "q_layernorm", None),
                    "weight",
                    None,
                ),
                key_norm_weight=getattr(
                    getattr(target, "k_layernorm", None),
                    "weight",
                    None,
                ),
                query_weight_prior=target.q_proj.weight.detach(),
                key_weight_prior=target.k_proj.weight.detach(),
                value_weight_prior=target.v_proj.weight.detach(),
                output_weight_prior=target.out_proj.weight.detach(),
                residual_assignment_sha256=self.map_sha256,
                norm_epsilon=self._attention_norm_epsilon(target),
                ridge=self.config.ridge,
                max_prior_delta_ratio=(
                    self.config.max_prior_delta_ratio
                ),
                cluster_ridge=self.config.cluster_ridge,
                cluster_max_iterations=(
                    self.config.cluster_max_iterations
                ),
                cluster_distance_chunk_size=(
                    self.config.cluster_distance_chunk_size
                ),
            )
        else:
            source = source_layer.conv
            target = target_layer.conv
            compiled = compile_lfm2_structural_short_conv_producer_fold(
                source_in_proj_weight=source.in_proj.weight,
                source_conv_weight=source.conv.weight,
                source_out_proj_weight=source.out_proj.weight,
                source_in_proj_bias=source.in_proj.bias,
                source_conv_bias=source.conv.bias,
                source_out_proj_bias=source.out_proj.bias,
                target_in_proj_weight_template=target.in_proj.weight,
                target_conv_weight_template=target.conv.weight,
                target_out_proj_weight_template=target.out_proj.weight,
                target_in_proj_bias_template=target.in_proj.bias,
                target_conv_bias_template=target.conv.bias,
                target_out_proj_bias_template=target.out_proj.bias,
                cases=operator_cases.fit_cases,
                residual_assignment=self._short_conv_assignments[layer_index],
                external_residual_assignment=(
                    self._short_conv_residual_assignment
                ),
                in_proj_weight_prior=target.in_proj.weight.detach(),
                in_proj_bias_prior=(
                    None
                    if target.in_proj.bias is None
                    else target.in_proj.bias.detach()
                ),
                out_proj_weight_prior=target.out_proj.weight.detach(),
                out_proj_bias_prior=(
                    None
                    if target.out_proj.bias is None
                    else target.out_proj.bias.detach()
                ),
                ridge=self.config.ridge,
                max_prior_delta_ratio=(
                    self.config.max_prior_delta_ratio
                ),
            )
        attention_score_fit = None
        attention_score_validation = None
        if (
            operator_kind == "full_attention"
            and self.config.attention_score_candidate_gate_enabled
        ):
            score_thresholds = (
                Lfm2AttentionScorePreservationThresholds(
                    maximum_softmax_fisher_error=(
                        self.config
                        .attention_score_maximum_softmax_fisher_error
                    ),
                    maximum_js_divergence=(
                        self.config
                        .attention_score_maximum_js_divergence
                    ),
                    minimum_top1_key_agreement_fraction=(
                        self.config
                        .attention_score_minimum_top1_key_agreement_fraction
                    ),
                    minimum_top_k_pair_order_agreement_fraction=(
                        self.config
                        .attention_score_minimum_top_k_pair_order_agreement_fraction
                    ),
                    minimum_winner_margin_sign_agreement_fraction=(
                        self.config
                        .attention_score_minimum_winner_margin_sign_agreement_fraction
                    ),
                    minimum_relative_margin_floor_retention_fraction=(
                        self.config
                        .attention_score_minimum_relative_margin_floor_retention_fraction
                    ),
                )
            )
            query_norm = getattr(
                getattr(target, "q_layernorm", None),
                "weight",
                None,
            )
            key_norm = getattr(
                getattr(target, "k_layernorm", None),
                "weight",
                None,
            )
            score_options = {
                "query_norm_weight": query_norm,
                "key_norm_weight": key_norm,
                "norm_epsilon": self._attention_norm_epsilon(target),
                "top_k": self.config.attention_score_top_k,
                "relative_margin_floor": (
                    self.config.attention_score_margin_floor_fraction
                ),
            }
            attention_score_fit = (
                evaluate_lfm2_compiled_attention_producer_score_preservation(
                    compiled,
                    operator_cases.fit_cases,
                    thresholds=(
                        Lfm2AttentionScorePreservationThresholds()
                    ),
                    **score_options,
                )
            )
            attention_score_validation = (
                evaluate_lfm2_compiled_attention_producer_score_preservation(
                    compiled,
                    operator_cases.validation_cases,
                    thresholds=score_thresholds,
                    **score_options,
                )
            )
            if not attention_score_validation.accepted:
                failed_checks = [
                    name
                    for name, accepted in (
                        attention_score_validation.checks
                    )
                    if accepted is False
                ]
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Held-out attention score/order/margin gate rejected "
                    f"layer {layer_index}: {','.join(failed_checks)} "
                    f"(report "
                    f"{attention_score_validation.report_sha256})."
                )
        materializer = operator_cases.norm_materializer
        payload = _ProducerPayload(
            operator_kind=operator_kind,
            operator_cases=operator_cases,
            compiled=compiled,
            operator_norm_weight=self._materializer_weight(
                materializer
            ),
            operator_norm_report_sha256=self._norm_report_sha256(
                materializer
            ),
            baseline_validation=baseline,
            attention_score_fit=attention_score_fit,
            attention_score_validation=attention_score_validation,
            attractor_baseline=attractor_baseline,
            split_sha256=split.split_sha256,
        )
        artifact_sha = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-producer-artifact-v1",
                "scope": scope.as_dict(),
                "fit_record_keys": [
                    list(value) for value in _record_keys(fit_records)
                ],
                "operator_kind": operator_kind,
                "compiled_state_sha256": compiled.compiled_state_sha256,
                "operator_norm_report_sha256": (
                    payload.operator_norm_report_sha256
                ),
                "attractor_baseline_snapshot_sha256": (
                    None
                    if attractor_baseline is None
                    else attractor_baseline.snapshot_sha256
                ),
                "attention_score_fit_report_sha256": (
                    None
                    if attention_score_fit is None
                    else attention_score_fit.report_sha256
                ),
                "attention_score_validation_report_sha256": (
                    None
                    if attention_score_validation is None
                    else attention_score_validation.report_sha256
                ),
                "split_sha256": split.split_sha256,
                "map_sha256": self.map_sha256,
            }
        )
        self._producer_payloads[layer_index] = payload
        return CompiledFoldStage(
            scope=scope,
            stage=CompileStage.PRODUCER,
            fit_record_keys=_record_keys(fit_records),
            artifact_sha256=artifact_sha,
            payload=payload,
            metadata=_scalar_metadata(
                {
                    "operator_kind": operator_kind,
                    "compiled_state_sha256": (
                        compiled.compiled_state_sha256
                    ),
                    "all_parent_batches_available": True,
                    "actual_child_capture_used": True,
                    "map_sha256": self.map_sha256,
                    "macro_cell_attractor_gate_enabled": (
                        attractor_baseline is not None
                    ),
                    "attractor_baseline_snapshot_sha256": (
                        None
                        if attractor_baseline is None
                        else attractor_baseline.snapshot_sha256
                    ),
                    "attention_score_candidate_gate_enabled": (
                        attention_score_validation is not None
                    ),
                    "attention_score_candidate_gate_accepted": (
                        None
                        if attention_score_validation is None
                        else attention_score_validation.accepted
                    ),
                    "attention_score_fit_report_sha256": (
                        None
                        if attention_score_fit is None
                        else attention_score_fit.report_sha256
                    ),
                    "attention_score_validation_report_sha256": (
                        None
                        if attention_score_validation is None
                        else attention_score_validation.report_sha256
                    ),
                    "attention_score_validation_top1_key_agreement": (
                        None
                        if attention_score_validation is None
                        else attention_score_validation
                        .top1_key_agreement_fraction
                    ),
                    "attention_score_validation_top_k_pair_order_agreement": (
                        None
                        if attention_score_validation is None
                        else attention_score_validation
                        .top_k_pair_order_agreement_fraction
                    ),
                    "attention_score_validation_margin_floor_retention": (
                        None
                        if attention_score_validation is None
                        else attention_score_validation
                        .relative_margin_floor_retention_fraction
                    ),
                    "attention_score_numeric_weight_matching_used": False,
                    "attention_score_parent_forward_calls": 0,
                }
            ),
        )

    def install_producer(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt:
        del audit
        layer_index = scope.layer_index
        payload = compiled.payload
        if (
            layer_index is None
            or not isinstance(payload, _ProducerPayload)
            or self._producer_payloads.get(layer_index) is not payload
            or layer_index in self._split_transactions
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Producer artifact/state binding is invalid."
            )
        state_before = self.child_state_fingerprint()
        topology_before = self.child_topology_fingerprint()
        target_layer = self.child_layers[layer_index]
        norm = target_layer.operator_norm
        norm_before = norm.weight.detach().to(device="cpu").clone()
        transaction = None
        try:
            self._copy_norm_weight(
                norm, payload.operator_norm_weight
            )
            if payload.operator_kind == "full_attention":
                if (
                    self.config.attention_consumer_mode
                    == "coupled-value-output"
                ):
                    transaction = (
                        install_lfm2_structural_attention_score_producer_fold(
                            target_layer.self_attn,
                            payload.compiled,
                        )
                    )
                else:
                    transaction = (
                        install_lfm2_structural_attention_producer_fold(
                            target_layer.self_attn,
                            payload.compiled,
                        )
                    )
            else:
                transaction = (
                    install_lfm2_structural_short_conv_producer_fold(
                        target_layer.conv, payload.compiled
                    )
                )
        except Exception:
            if transaction is not None and transaction.active:
                rollback_lfm2_structural_split_install(transaction)
            self._copy_norm_weight(norm, norm_before)
            raise
        if not transaction.active:
            self._copy_norm_weight(norm, norm_before)
            raise Lfm2DynamicClosedLoopAdapterError(
                "Producer split transaction is not active."
            )
        self._split_transactions[layer_index] = transaction
        return self._receipt(
            scope=scope,
            stage=CompileStage.PRODUCER,
            artifact_sha256=compiled.artifact_sha256,
            state_before=state_before,
            topology_before=topology_before,
            metadata={
                "producer_installed_before_recapture": True,
                "operator_norm_installed": True,
                "split_transaction_active": True,
                "attention_consumer_mode": (
                    self.config.attention_consumer_mode
                    if payload.operator_kind == "full_attention"
                    else "not-applicable"
                ),
            },
        )

    def fit_consumer(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage:
        del parent_endpoints, audit
        layer_index = scope.layer_index
        if layer_index is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Consumer scope has no layer index."
            )
        producer = self._producer_payloads.get(layer_index)
        transaction = self._split_transactions.get(layer_index)
        if (
            producer is None
            or transaction is None
            or not transaction.active
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Consumer fitting requires an installed producer."
            )
        capture = self._capture_payload(child_capture)
        split = self._split(capture.paired, fit_records)
        contexts = extract_lfm2_actual_child_contexts(
            producer.operator_cases, capture.paired
        )
        target_layer = self.child_layers[layer_index]
        if producer.operator_kind == "full_attention":
            target = target_layer.self_attn
            if (
                self.config.attention_consumer_mode
                == "coupled-value-output"
            ):
                compiled_consumer = (
                    compile_lfm2_structural_attention_value_output_fold(
                        producer=producer.compiled,
                        target_value_weight_template=(
                            target.v_proj.weight
                        ),
                        target_output_weight_template=(
                            target.out_proj.weight
                        ),
                        actual_child_contexts=contexts.fit_contexts,
                        cases=producer.operator_cases.fit_cases,
                        live_value_weight=target.v_proj.weight.detach(),
                        value_weight_prior=(
                            producer.compiled.value_weight
                        ),
                        output_weight_prior=(
                            target.out_proj.weight.detach()
                        ),
                        query_norm_weight=getattr(
                            getattr(target, "q_layernorm", None),
                            "weight",
                            None,
                        ),
                        key_norm_weight=getattr(
                            getattr(target, "k_layernorm", None),
                            "weight",
                            None,
                        ),
                        norm_epsilon=(
                            self._attention_norm_epsilon(target)
                        ),
                        ridge=self.config.ridge,
                        context_ridge=(
                            self.config
                            .attention_value_output_context_ridge
                        ),
                        max_prior_delta_ratio=(
                            self.config.max_prior_delta_ratio
                        ),
                        iterations=(
                            self.config
                            .attention_value_output_iterations
                        ),
                        fit_tolerance=(
                            self.config
                            .attention_value_output_fit_tolerance
                        ),
                        context_equivalence_tolerance=(
                            self.config
                            .attention_value_output_context_equivalence_tolerance
                        ),
                    )
                )
            else:
                compiled_consumer = (
                    compile_lfm2_structural_attention_consumer_fold(
                        target_output_weight_template=(
                            target.out_proj.weight
                        ),
                        actual_child_contexts=contexts.fit_contexts,
                        cases=producer.operator_cases.fit_cases,
                        output_weight_prior=(
                            target.out_proj.weight.detach()
                        ),
                        ridge=self.config.ridge,
                        max_prior_delta_ratio=(
                            self.config.max_prior_delta_ratio
                        ),
                    )
                )
        else:
            target = target_layer.conv
            compiled_consumer = (
                compile_lfm2_structural_short_conv_consumer_fold(
                    target_out_proj_weight_template=(
                        target.out_proj.weight
                    ),
                    target_out_proj_bias_template=(
                        target.out_proj.bias
                    ),
                    actual_child_contexts=contexts.fit_contexts,
                    cases=producer.operator_cases.fit_cases,
                    out_proj_weight_prior=target.out_proj.weight.detach(),
                    out_proj_bias_prior=(
                        None
                        if target.out_proj.bias is None
                        else target.out_proj.bias.detach()
                    ),
                    ridge=self.config.ridge,
                    max_prior_delta_ratio=(
                        self.config.max_prior_delta_ratio
                    ),
                )
            )
        payload = _ConsumerPayload(
            operator_kind=producer.operator_kind,
            compiled=compiled_consumer,
            operator_cases=producer.operator_cases,
            split_sha256=split.split_sha256,
        )
        artifact_sha = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-consumer-artifact-v1",
                "scope": scope.as_dict(),
                "fit_record_keys": [
                    list(value) for value in _record_keys(fit_records)
                ],
                "operator_kind": producer.operator_kind,
                "compiled_state_sha256": (
                    compiled_consumer.compiled_state_sha256
                ),
                "actual_child_context_sha256": (
                    compiled_consumer.actual_child_context_sha256
                ),
                "producer_replay_sha256": (
                    capture.replay_sha256
                ),
                "split_sha256": split.split_sha256,
            }
        )
        self._consumer_payloads[layer_index] = payload
        return CompiledFoldStage(
            scope=scope,
            stage=CompileStage.CONSUMER,
            fit_record_keys=_record_keys(fit_records),
            artifact_sha256=artifact_sha,
            payload=payload,
            metadata=_scalar_metadata(
                {
                    "operator_kind": producer.operator_kind,
                    "compiled_state_sha256": (
                        compiled_consumer.compiled_state_sha256
                    ),
                    "actual_child_context_sha256": (
                        compiled_consumer.actual_child_context_sha256
                    ),
                    "fitted_after_installed_producer": True,
                    "canonical_full_prefix_context": True,
                    "attention_consumer_mode": (
                        self.config.attention_consumer_mode
                        if producer.operator_kind == "full_attention"
                        else "not-applicable"
                    ),
                }
            ),
        )

    def install_consumer(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt:
        del audit
        layer_index = scope.layer_index
        payload = compiled.payload
        if (
            layer_index is None
            or not isinstance(payload, _ConsumerPayload)
            or self._consumer_payloads.get(layer_index) is not payload
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Consumer artifact/state binding is invalid."
            )
        transaction = self._split_transactions.get(layer_index)
        if transaction is None or not transaction.active:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Consumer has no active producer transaction."
            )
        state_before = self.child_state_fingerprint()
        topology_before = self.child_topology_fingerprint()
        target_layer = self.child_layers[layer_index]
        if payload.operator_kind == "full_attention":
            if isinstance(
                payload.compiled,
                Lfm2CompiledStructuralAttentionValueOutputFold,
            ):
                report = (
                    install_lfm2_structural_attention_value_output_fold(
                        target_layer.self_attn,
                        payload.compiled,
                        transaction,
                    )
                )
            else:
                report = install_lfm2_structural_attention_consumer_fold(
                    target_layer.self_attn,
                    payload.compiled,
                    transaction,
                )
        else:
            report = install_lfm2_structural_short_conv_consumer_fold(
                target_layer.conv,
                payload.compiled,
                transaction,
            )
        if not report.accepted or transaction.active:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Consumer split installation was not committed."
            )
        self._split_transactions.pop(layer_index, None)
        return self._receipt(
            scope=scope,
            stage=CompileStage.CONSUMER,
            artifact_sha256=compiled.artifact_sha256,
            state_before=state_before,
            topology_before=topology_before,
            metadata={
                "consumer_fitted_from_actual_child_context": True,
                "split_transaction_committed": True,
                "parameter_values_verified": (
                    report.parameter_values_verified
                ),
                "attention_consumer_mode": (
                    self.config.attention_consumer_mode
                    if payload.operator_kind == "full_attention"
                    else "not-applicable"
                ),
            },
        )

    def compile_downstream(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage:
        del parent_endpoints, audit
        layer_index = scope.layer_index
        if layer_index is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Downstream scope has no layer index."
            )
        producer = self._producer_payloads.get(layer_index)
        consumer = self._consumer_payloads.get(layer_index)
        if producer is None or consumer is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Downstream fitting requires producer and consumer artifacts."
            )
        capture = self._capture_payload(child_capture)
        split = self._split(capture.paired, fit_records)
        contexts = extract_lfm2_actual_child_contexts(
            producer.operator_cases, capture.paired
        )
        target_layer = self.child_layers[layer_index]
        if producer.operator_kind == "full_attention":
            target_operator = target_layer.self_attn
            candidate = compute_lfm2_candidate_post_operator(
                producer.operator_cases,
                contexts,
                output_weight=target_operator.out_proj.weight,
            )
        else:
            target_operator = target_layer.conv
            candidate = compute_lfm2_candidate_post_operator(
                producer.operator_cases,
                contexts,
                output_weight=target_operator.out_proj.weight,
                output_bias=target_operator.out_proj.bias,
            )
        mlp_cases = build_lfm2_mlp_cases_from_candidate_post_operator(
            split,
            producer.operator_cases,
            candidate,
            coordinate_map_callback=self._project_parent,
            norm_fit_callback=self._fit_norm(target_layer.ffn_norm),
        )
        source_mlp = self.source_layers[layer_index].feed_forward
        target_mlp = target_layer.feed_forward
        compiled_mlp = compile_lfm2_structural_residual_mlp_fold(
            source_gate_weight=source_mlp.w1.weight,
            source_up_weight=source_mlp.w3.weight,
            source_down_weight=source_mlp.w2.weight,
            target_gate_weight_template=target_mlp.w1.weight,
            target_up_weight_template=target_mlp.w3.weight,
            target_down_weight_template=target_mlp.w2.weight,
            cases=mlp_cases.fit_cases,
            gate_weight_prior=target_mlp.w1.weight.detach(),
            up_weight_prior=target_mlp.w3.weight.detach(),
            down_weight_prior=target_mlp.w2.weight.detach(),
            residual_assignment_sha256=self.map_sha256,
            ridge=self.config.ridge,
            max_prior_delta_ratio=self.config.max_prior_delta_ratio,
            cluster_ridge=self.config.cluster_ridge,
            cluster_max_iterations=self.config.cluster_max_iterations,
            cluster_distance_chunk_size=(
                self.config.cluster_distance_chunk_size
            ),
        )
        offline_candidate = compute_lfm2_candidate_post_layer(
            mlp_cases,
            gate_weight=compiled_mlp.gate_weight,
            up_weight=compiled_mlp.up_weight,
            down_weight=compiled_mlp.down_weight,
        )
        payload = _DownstreamPayload(
            compiled=compiled_mlp,
            ffn_norm_weight=self._materializer_weight(
                mlp_cases.norm_materializer
            ),
            ffn_norm_report_sha256=self._norm_report_sha256(
                mlp_cases.norm_materializer
            ),
            offline_comparison=offline_candidate.comparison,
            split_sha256=split.split_sha256,
        )
        artifact_sha = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-downstream-artifact-v1",
                "scope": scope.as_dict(),
                "fit_record_keys": [
                    list(value) for value in _record_keys(fit_records)
                ],
                "compiled_state_sha256": (
                    compiled_mlp.compiled_state_sha256
                ),
                "ffn_norm_report_sha256": (
                    payload.ffn_norm_report_sha256
                ),
                "candidate_post_operator_replay_sha256": (
                    candidate.child_replay_sha256
                ),
                "split_sha256": split.split_sha256,
                "map_sha256": self.map_sha256,
            }
        )
        self._downstream_payloads[layer_index] = payload
        return CompiledFoldStage(
            scope=scope,
            stage=CompileStage.DOWNSTREAM,
            fit_record_keys=_record_keys(fit_records),
            artifact_sha256=artifact_sha,
            payload=payload,
            metadata=_scalar_metadata(
                {
                    "compiled_state_sha256": (
                        compiled_mlp.compiled_state_sha256
                    ),
                    "mlp_input_source": (
                        "candidate_post_operator_actual_child_context"
                    ),
                    "candidate_fit_relative_rmse": float(
                        offline_candidate.comparison.candidate_fit.relative_rmse
                    ),
                    "current_fit_relative_rmse": float(
                        offline_candidate.comparison.current_fit.relative_rmse
                    ),
                    "map_sha256": self.map_sha256,
                }
            ),
        )

    def install_downstream(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt:
        del audit
        layer_index = scope.layer_index
        payload = compiled.payload
        if (
            layer_index is None
            or not isinstance(payload, _DownstreamPayload)
            or self._downstream_payloads.get(layer_index) is not payload
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Downstream artifact/state binding is invalid."
            )
        state_before = self.child_state_fingerprint()
        topology_before = self.child_topology_fingerprint()
        target_layer = self.child_layers[layer_index]
        norm_before = (
            target_layer.ffn_norm.weight.detach()
            .to(device="cpu")
            .clone()
        )
        try:
            self._copy_norm_weight(
                target_layer.ffn_norm, payload.ffn_norm_weight
            )
            report = install_lfm2_structural_mlp_fold(
                target_layer.feed_forward, payload.compiled
            )
            if not report.accepted:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "MLP installation report was rejected."
                )
        except Exception:
            self._copy_norm_weight(
                target_layer.ffn_norm, norm_before
            )
            raise
        return self._receipt(
            scope=scope,
            stage=CompileStage.DOWNSTREAM,
            artifact_sha256=compiled.artifact_sha256,
            state_before=state_before,
            topology_before=topology_before,
            metadata={
                "ffn_norm_installed": True,
                "mlp_installed": True,
                "mlp_built_from_candidate_post_operator": True,
            },
        )

    def validate_layer(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision:
        del parent_endpoints, audit
        layer_index = scope.layer_index
        if layer_index is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Layer validation scope has no layer index."
            )
        producer = self._producer_payloads.get(layer_index)
        if producer is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Layer validation has no pre-fold baseline."
            )
        capture = self._capture_payload(child_capture)
        indices = self._batch_indices(validation_records)
        candidate = self._heldout_layer_metrics(
            capture.paired,
            layer_index=layer_index,
            batch_indices=indices,
        )
        endpoint_decision = self._validation_decision(
            scope=scope,
            records=validation_records,
            baseline=producer.baseline_validation,
            candidate=candidate,
            minimum_improvement=(
                self.config.minimum_layer_relative_rmse_improvement
            ),
            endpoint="post-layer residual",
        )
        gate = self.macro_cell_attractor_gate
        attractor_baseline = producer.attractor_baseline
        if gate is None:
            if attractor_baseline is not None:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Layer validation retained attractor evidence without a gate."
                )
            return endpoint_decision
        if attractor_baseline is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Layer validation has no pre-cell attractor baseline."
            )
        attractor_candidate = gate.capture(
            self.child_model,
            phase=f"post-macro-cell-layer-{layer_index:06d}",
        )
        attractor_decision = gate.compare(
            attractor_baseline,
            attractor_candidate,
        )
        attractor_payload = attractor_decision.as_dict()
        attractor_metrics = attractor_candidate.metrics
        accepted = bool(
            endpoint_decision.accepted and attractor_decision.accepted
        )
        reasons = [endpoint_decision.reason]
        if attractor_decision.accepted:
            reasons.append("macro-cell free-rollout attractor gate accepted")
        else:
            decision_checks = attractor_payload.get("checks", {})
            failed_details = []
            for check_name in attractor_decision.failed_checks:
                check = (
                    decision_checks.get(check_name)
                    if isinstance(decision_checks, Mapping)
                    else None
                )
                if (
                    isinstance(check, Mapping)
                    and isinstance(check.get("observed"), (int, float))
                    and not isinstance(check.get("observed"), bool)
                    and isinstance(check.get("maximum"), (int, float))
                    and not isinstance(check.get("maximum"), bool)
                ):
                    failed_details.append(
                        f"{check_name}="
                        f"{float(check['observed']):.17g}>"
                        f"{float(check['maximum']):.17g}"
                    )
                else:
                    failed_details.append(check_name)
            reasons.append(
                "macro-cell free-rollout attractor gate rejected: "
                + ",".join(failed_details)
            )
        metrics = dict(endpoint_decision.metrics)
        metrics.update(
            {
                "macro_cell_attractor_gate_enabled": True,
                "macro_cell_attractor_gate_accepted": (
                    attractor_decision.accepted
                ),
                "macro_cell_attractor_decision_sha256": (
                    attractor_decision.decision_sha256
                ),
                "macro_cell_attractor_incumbent_snapshot_sha256": (
                    attractor_baseline.snapshot_sha256
                ),
                "macro_cell_attractor_candidate_snapshot_sha256": (
                    attractor_candidate.snapshot_sha256
                ),
                "macro_cell_attractor_failed_check_count": len(
                    attractor_decision.failed_checks
                ),
                "macro_cell_attractor_failed_checks": ",".join(
                    attractor_decision.failed_checks
                ),
                "macro_cell_attractor_new_collapsed_records": len(
                    attractor_payload["new_collapsed_record_ids"]
                ),
                "macro_cell_attractor_mean_entropy_parent_distance": float(
                    attractor_metrics["mean_entropy_parent_distance"]
                ),
                "macro_cell_attractor_mean_repetition_excess": float(
                    attractor_metrics["mean_repetition_excess"]
                ),
                "macro_cell_attractor_mean_dominant_token_excess": float(
                    attractor_metrics["mean_dominant_token_excess"]
                ),
                "macro_cell_attractor_mean_eos_position_error": float(
                    attractor_metrics["mean_eos_position_error"]
                ),
                "macro_cell_attractor_collapsed_record_count": int(
                    attractor_metrics["collapsed_record_count"]
                ),
                "macro_cell_exact_token_metrics_used_for_acceptance": False,
                "macro_cell_parent_forward_calls": 0,
                "macro_cell_training_steps": 0,
            }
        )
        decision_checks = attractor_payload.get("checks", {})
        if isinstance(decision_checks, Mapping):
            for check_name, check in sorted(decision_checks.items()):
                if not isinstance(check_name, str) or not isinstance(
                    check,
                    Mapping,
                ):
                    continue
                observed = check.get("observed")
                maximum = check.get("maximum")
                passed = check.get("passed")
                if (
                    isinstance(observed, (int, float))
                    and not isinstance(observed, bool)
                    and isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                    and isinstance(passed, bool)
                ):
                    prefix = f"macro_cell_attractor_check_{check_name}"
                    metrics[f"{prefix}_observed"] = float(observed)
                    metrics[f"{prefix}_maximum"] = float(maximum)
                    metrics[f"{prefix}_passed"] = passed
        return ValidationDecision(
            scope=scope,
            accepted=accepted,
            validation_record_keys=endpoint_decision.validation_record_keys,
            reason="; ".join(reasons),
            metrics=_scalar_metadata(metrics),
        )

    # -- Final normalization --------------------------------------------

    def _parent_final_norm(self, value: Any) -> Any:
        return _analytical_rms_norm(
            value,
            _final_norm(self.source_model),
            "parent final norm",
        )

    def _final_norm_metrics(
        self,
        pairs: Lfm2FinalNormPairs,
        *,
        module: Any,
        validation: bool,
    ) -> FitMetrics:
        torch = _torch()
        inputs = (
            pairs.validation_inputs
            if validation
            else pairs.fit_inputs
        )
        targets = (
            pairs.validation_targets
            if validation
            else pairs.fit_targets
        )
        if not inputs or len(inputs) != len(targets):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Final-norm metric pairs are empty or misaligned."
            )
        predictions = [
            _analytical_rms_norm(
                value, module, "child final norm"
            )
            .to(dtype=torch.float64)
            .reshape(-1, int(self.coordinate_map.target_width))
            for value in inputs
        ]
        target_rows = [
            value.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1, int(self.coordinate_map.target_width))
            for value in targets
        ]
        return measure_fit(
            torch.cat(predictions, dim=0),
            torch.cat(target_rows, dim=0),
        )

    def compile_final_normalization(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage:
        del parent_endpoints, audit
        if scope.kind != "final_normalization":
            raise Lfm2DynamicClosedLoopAdapterError(
                "Final-normalization compiler received a layer scope."
            )
        capture = self._capture_payload(child_capture)
        split = self._split(capture.paired, fit_records)
        final_index = len(self.child_layers) - 1
        pairs = build_lfm2_final_norm_pairs(
            split,
            final_layer_index=final_index,
            coordinate_map_callback=self._project_parent,
            parent_final_norm_callback=self._parent_final_norm,
        )
        child_norm = _final_norm(self.child_model)
        materializer = self._fit_norm(child_norm)(
            Lfm2NormFitRequest(
                role="model.embedding_norm",
                fit_inputs=pairs.fit_inputs,
                fit_targets=pairs.fit_targets,
                validation_inputs=pairs.validation_inputs,
                validation_targets=pairs.validation_targets,
            )
        )
        baseline = self._final_norm_metrics(
            pairs, module=child_norm, validation=True
        )
        payload = _FinalNormPayload(
            fitted_weight=self._materializer_weight(materializer),
            fit_report_sha256=self._norm_report_sha256(materializer),
            baseline_validation=baseline,
            pairs=pairs,
            split_sha256=split.split_sha256,
        )
        artifact_sha = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-final-norm-artifact-v1",
                "fit_record_keys": [
                    list(value) for value in _record_keys(fit_records)
                ],
                "fit_report_sha256": payload.fit_report_sha256,
                "split_sha256": split.split_sha256,
                "map_sha256": self.map_sha256,
                "runs_after_selected_layers": list(
                    self._selected_layers
                ),
            }
        )
        self._final_payload = payload
        return CompiledFoldStage(
            scope=scope,
            stage=CompileStage.FINAL_NORMALIZATION,
            fit_record_keys=_record_keys(fit_records),
            artifact_sha256=artifact_sha,
            payload=payload,
            metadata=_scalar_metadata(
                {
                    "fit_report_sha256": payload.fit_report_sha256,
                    "parent_final_norm_executed_analytically": True,
                    "parent_model_forward_calls": 0,
                    "runs_after_all_selected_layers": True,
                    "map_sha256": self.map_sha256,
                }
            ),
        )

    def install_final_normalization(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt:
        del audit
        payload = compiled.payload
        if (
            not isinstance(payload, _FinalNormPayload)
            or self._final_payload is not payload
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Final norm artifact/state binding is invalid."
            )
        state_before = self.child_state_fingerprint()
        topology_before = self.child_topology_fingerprint()
        self._copy_norm_weight(
            _final_norm(self.child_model), payload.fitted_weight
        )
        return self._receipt(
            scope=scope,
            stage=CompileStage.FINAL_NORMALIZATION,
            artifact_sha256=compiled.artifact_sha256,
            state_before=state_before,
            topology_before=topology_before,
            metadata={
                "final_norm_installed_after_layers": True,
                "fit_report_sha256": payload.fit_report_sha256,
            },
        )

    def validate_final_normalization(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision:
        del parent_endpoints, audit
        payload = self._final_payload
        if payload is None:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Final norm validation has no baseline/artifact."
            )
        capture = self._capture_payload(child_capture)
        fit_indices = tuple(
            index
            for index in range(len(self.parent_cache))
            if index not in frozenset(
                self._batch_indices(validation_records)
            )
        )
        fit_records = tuple(self._records[index] for index in fit_indices)
        split = self._split(capture.paired, fit_records)
        pairs = build_lfm2_final_norm_pairs(
            split,
            final_layer_index=len(self.child_layers) - 1,
            coordinate_map_callback=self._project_parent,
            parent_final_norm_callback=self._parent_final_norm,
        )
        candidate = self._final_norm_metrics(
            pairs,
            module=_final_norm(self.child_model),
            validation=True,
        )
        return self._validation_decision(
            scope=scope,
            records=validation_records,
            baseline=payload.baseline_validation,
            candidate=candidate,
            minimum_improvement=(
                self.config.minimum_final_norm_relative_rmse_improvement
            ),
            endpoint="final normalized residual",
        )

    # -- Global output interface ---------------------------------------

    def compile_output_interface(
        self,
        scope: FoldScope,
        fit_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        child_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> CompiledFoldStage:
        """Fit one global latent map ``E(F) = E0 F.T`` in FP64."""

        del audit
        torch = _torch()
        if scope.kind != "output_interface":
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface compiler received a non-output scope."
            )
        if not isinstance(
            parent_endpoints.payload, _EndpointPayload
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface parent endpoint envelope is invalid."
            )
        capture = self._capture_payload(child_capture)
        split = self._split(capture.paired, fit_records)
        base_weight, source_binding_sha256 = (
            self._source_bound_output_weight()
        )
        base64 = base_weight.to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        width = int(base64.shape[1])
        identity = torch.eye(width, dtype=torch.float64)
        output_is_tied = self._child_heterogeneous_layout is None
        if self.config.output_interface_mode == "identity-skip":
            base_weight_sha256 = _tensor_sha256(base_weight)
            transform_sha256 = _tensor_sha256(identity)
            payload = _OutputInterfacePayload(
                base_weight=base_weight,
                candidate_weight=base_weight.clone(),
                transform=identity,
                transform_sha256=transform_sha256,
                base_weight_sha256=base_weight_sha256,
                candidate_weight_sha256=base_weight_sha256,
                source_binding_sha256=source_binding_sha256,
                split_sha256=split.split_sha256,
                ridge_effective=0.0,
                interpolation_alpha=0.0,
                materialized_delta_ratio=0.0,
                transform_condition_number=1.0,
                transform_min_singular_value=1.0,
                output_subspace_relative_rmse=0.0,
                body_latent_relative_rmse=0.0,
                identity_skip=True,
            )
            artifact_sha = canonical_sha256(
                {
                    "schema": (
                        "neural-forge-lfm2-tied-output-interface-"
                        "identity-skip-v1"
                        if output_is_tied
                        else (
                            "neural-forge-lfm2-untied-output-interface-"
                            "identity-skip-v2"
                        )
                    ),
                    "fit_record_keys": [
                        list(value)
                        for value in _record_keys(fit_records)
                    ],
                    "source_binding_sha256": source_binding_sha256,
                    "base_weight_sha256": base_weight_sha256,
                    "candidate_weight_sha256": base_weight_sha256,
                    "transform_sha256": transform_sha256,
                    "split_sha256": split.split_sha256,
                    "map_sha256": self.map_sha256,
                    "identity_skip": True,
                    "materialized_delta_ratio": 0.0,
                    (
                        "tied_parameter_bytes_unchanged"
                        if output_is_tied
                        else "output_parameter_bytes_unchanged"
                    ): True,
                    (
                        "tied_parameter_bytes_unchanged"
                        if output_is_tied
                        else "input_embedding_bytes_unchanged"
                    ): True,
                }
            )
            self._output_payload = payload
            return CompiledFoldStage(
                scope=scope,
                stage=CompileStage.OUTPUT_INTERFACE,
                fit_record_keys=_record_keys(fit_records),
                artifact_sha256=artifact_sha,
                payload=payload,
                metadata=_scalar_metadata(
                    {
                        "fit_rows": 0,
                        "vocabulary_size": int(base_weight.shape[0]),
                        "target_width": width,
                        "source_binding_sha256": source_binding_sha256,
                        "base_weight_sha256": base_weight_sha256,
                        "candidate_weight_sha256": base_weight_sha256,
                        "transform_sha256": transform_sha256,
                        "output_interface_mode": "identity-skip",
                        "identity_skip": True,
                        (
                            "tied_parameter_bytes_unchanged"
                            if output_is_tied
                            else "output_parameter_bytes_unchanged"
                        ): True,
                        (
                            "tied_parameter_bytes_unchanged"
                            if output_is_tied
                            else "input_embedding_bytes_unchanged"
                        ): True,
                        "materialized_delta_ratio": 0.0,
                        "maximum_delta_ratio": (
                            self.config.output_max_prior_delta_ratio
                        ),
                        "global_latent_transform": False,
                        "per_token_row_fitting": False,
                        "source_weights_and_sealed_cache_only": True,
                        "parent_model_forward_calls": 0,
                        "training_steps": 0,
                        "backward_calls": 0,
                        "optimizer_steps": 0,
                        (
                            "teacher_forced_terminal_acceptance_is_"
                            "necessary_not_sufficient"
                        ): True,
                        (
                            "external_whole_output_behavior_gate_required_"
                            "for_promotion"
                        ): True,
                    }
                ),
            )
        gram = base64.transpose(0, 1) @ base64
        gram_singular = torch.linalg.eigvalsh(gram)
        if (
            not bool(torch.isfinite(gram_singular).all().item())
            or float(gram_singular[0].item())
            <= torch.finfo(torch.float64).eps
            * max(float(gram_singular[-1].item()), 1.0)
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "The tied output basis is rank-deficient in FP64."
            )

        hidden_parts = []
        parent_times_base_parts = []
        parent_logit_square = 0.0
        fit_rows = 0
        for hidden, parent, _candidate, _targets in (
            self._iter_output_blocks(
                capture.paired,
                split.fit_batch_indices,
            )
        ):
            hidden64 = hidden.to(dtype=torch.float64)
            hidden_parts.append(hidden64)
            fit_rows += int(hidden64.shape[0])
            for start in range(
                0,
                int(parent.shape[0]),
                self.config.output_metric_chunk_rows,
            ):
                stop = min(
                    int(parent.shape[0]),
                    start + self.config.output_metric_chunk_rows,
                )
                parent64 = parent[start:stop].to(
                    dtype=torch.float64
                )
                parent_times_base_parts.append(parent64 @ base64)
                parent_logit_square += float(
                    parent64.square().sum().item()
                )
        if fit_rows <= 0 or not hidden_parts:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface fit has no sealed trajectory rows."
            )
        hidden_rows = torch.cat(hidden_parts, dim=0)
        parent_times_base = torch.cat(
            parent_times_base_parts, dim=0
        )
        if (
            int(hidden_rows.shape[0]) != fit_rows
            or int(parent_times_base.shape[0]) != fit_rows
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface latent fit rows are misaligned."
            )
        projected_parent_latent = torch.linalg.solve(
            gram, parent_times_base.transpose(0, 1)
        ).transpose(0, 1)
        if not bool(
            torch.isfinite(projected_parent_latent).all().item()
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-subspace projection produced non-finite rows."
            )
        hidden_gram = hidden_rows.transpose(0, 1) @ hidden_rows
        ridge_scale = max(
            float(torch.trace(hidden_gram).item()) / width,
            torch.finfo(torch.float64).tiny,
        )
        ridge_effective = self.config.output_ridge * ridge_scale
        system = hidden_gram + ridge_effective * identity
        right_hand_side = (
            hidden_rows.transpose(0, 1)
            @ projected_parent_latent
            + ridge_effective * identity
        )
        raw_transform = torch.linalg.solve(
            system, right_hand_side
        ).contiguous()
        if not bool(torch.isfinite(raw_transform).all().item()):
            raise Lfm2DynamicClosedLoopAdapterError(
                "FP64 output-interface solve produced non-finite values."
            )

        base_norm = float(torch.linalg.vector_norm(base64).item())
        if base_norm <= 0.0:
            raise Lfm2DynamicClosedLoopAdapterError(
                "The tied base vocabulary weight has zero norm."
            )
        raw_delta = base64 @ (
            raw_transform.transpose(0, 1) - identity
        )
        raw_delta_ratio = float(
            torch.linalg.vector_norm(raw_delta).item()
        ) / base_norm
        alpha = 1.0
        if (
            raw_delta_ratio
            > self.config.output_max_prior_delta_ratio
        ):
            alpha = (
                self.config.output_max_prior_delta_ratio
                / raw_delta_ratio
            )

        def bounded_transform(value: float):
            result = identity + value * (raw_transform - identity)
            singular = torch.linalg.svdvals(result)
            condition = float(
                (singular[0] / singular[-1]).item()
            )
            return result, singular, condition

        transform, singular_values, condition_number = (
            bounded_transform(alpha)
        )
        for _attempt in range(64):
            if (
                bool(torch.isfinite(singular_values).all().item())
                and condition_number
                <= self.config.output_max_transform_condition_number
                and float(singular_values[-1].item())
                >= self.config.output_min_transform_singular_value
            ):
                break
            alpha *= 0.5
            transform, singular_values, condition_number = (
                bounded_transform(alpha)
            )
        else:
            raise Lfm2DynamicClosedLoopAdapterError(
                "No globally interpolated output transform satisfies the "
                "condition/singular-value gates."
            )

        candidate_weight = None
        materialized_delta_ratio = math.inf
        for _attempt in range(64):
            transform, singular_values, condition_number = (
                bounded_transform(alpha)
            )
            candidate64 = base64 @ transform.transpose(0, 1)
            candidate_weight = (
                candidate64.to(dtype=base_weight.dtype)
                .contiguous()
                .clone()
            )
            materialized_delta_ratio = float(
                torch.linalg.vector_norm(
                    candidate_weight.to(dtype=torch.float64) - base64
                ).item()
            ) / base_norm
            if (
                materialized_delta_ratio
                <= self.config.output_max_prior_delta_ratio
                + 32.0 * torch.finfo(torch.float64).eps
            ):
                break
            alpha *= 0.99 * (
                self.config.output_max_prior_delta_ratio
                / materialized_delta_ratio
            )
        else:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Materialized tied output weight violates its delta cap."
            )
        if candidate_weight is None or not bool(
            torch.isfinite(candidate_weight).all().item()
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Materialized tied output weight is invalid."
            )
        if (
            condition_number
            > self.config.output_max_transform_condition_number
            or float(singular_values[-1].item())
            < self.config.output_min_transform_singular_value
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "The final materialized output transform violates its "
                "condition/singular-value gates."
            )
        base_weight_sha256 = _tensor_sha256(base_weight)
        candidate_weight_sha256 = _tensor_sha256(candidate_weight)
        if candidate_weight_sha256 == base_weight_sha256:
            raise Lfm2DynamicClosedLoopAdapterError(
                "The bounded output fold made no materialized change."
            )

        fitted_latent = hidden_rows @ transform
        latent_denominator = max(
            float(projected_parent_latent.square().sum().item()),
            torch.finfo(torch.float64).tiny,
        )
        body_latent_relative_rmse = math.sqrt(
            float(
                (
                    fitted_latent - projected_parent_latent
                )
                .square()
                .sum()
                .item()
            )
            / latent_denominator
        )
        projected_energy = float(
            (
                parent_times_base * projected_parent_latent
            )
            .sum()
            .item()
        )
        output_subspace_squared_error = max(
            0.0, parent_logit_square - projected_energy
        )
        output_subspace_relative_rmse = math.sqrt(
            output_subspace_squared_error
            / max(
                parent_logit_square,
                torch.finfo(torch.float64).tiny,
            )
        )
        transform_sha256 = _tensor_sha256(transform)
        payload = _OutputInterfacePayload(
            base_weight=base_weight,
            candidate_weight=candidate_weight,
            transform=transform,
            transform_sha256=transform_sha256,
            base_weight_sha256=base_weight_sha256,
            candidate_weight_sha256=candidate_weight_sha256,
            source_binding_sha256=source_binding_sha256,
            split_sha256=split.split_sha256,
            ridge_effective=ridge_effective,
            interpolation_alpha=alpha,
            materialized_delta_ratio=materialized_delta_ratio,
            transform_condition_number=condition_number,
            transform_min_singular_value=float(
                singular_values[-1].item()
            ),
            output_subspace_relative_rmse=(
                output_subspace_relative_rmse
            ),
            body_latent_relative_rmse=body_latent_relative_rmse,
            identity_skip=False,
        )
        artifact_sha = canonical_sha256(
            {
                "schema": (
                    "neural-forge-lfm2-tied-global-output-interface-v1"
                    if output_is_tied
                    else (
                        "neural-forge-lfm2-untied-global-output-interface-v2"
                    )
                ),
                "fit_record_keys": [
                    list(value)
                    for value in _record_keys(fit_records)
                ],
                "source_binding_sha256": source_binding_sha256,
                "base_weight_sha256": base_weight_sha256,
                "candidate_weight_sha256": candidate_weight_sha256,
                "transform_sha256": transform_sha256,
                "split_sha256": split.split_sha256,
                "map_sha256": self.map_sha256,
                "ridge_effective": ridge_effective,
                "interpolation_alpha": alpha,
                "materialized_delta_ratio": (
                    materialized_delta_ratio
                ),
                "transform_condition_number": condition_number,
                "transform_min_singular_value": float(
                    singular_values[-1].item()
                ),
                "global_latent_transform": True,
                "per_token_row_fitting": False,
                "source_weights_and_sealed_cache_only": True,
            }
        )
        self._output_payload = payload
        return CompiledFoldStage(
            scope=scope,
            stage=CompileStage.OUTPUT_INTERFACE,
            fit_record_keys=_record_keys(fit_records),
            artifact_sha256=artifact_sha,
            payload=payload,
            metadata=_scalar_metadata(
                {
                    "fit_rows": fit_rows,
                    "vocabulary_size": int(base_weight.shape[0]),
                    "target_width": width,
                    "source_binding_sha256": source_binding_sha256,
                    "base_weight_sha256": base_weight_sha256,
                    "candidate_weight_sha256": (
                        candidate_weight_sha256
                    ),
                    "transform_sha256": transform_sha256,
                    "fp64_solve": True,
                    "trace_scaled_ridge": True,
                    "ridge_effective": ridge_effective,
                    "global_latent_transform": True,
                    "per_token_row_fitting": False,
                    "all_vocabulary_rows_transformed_together": True,
                    "interpolation_alpha": alpha,
                    "raw_delta_ratio": raw_delta_ratio,
                    "materialized_delta_ratio": (
                        materialized_delta_ratio
                    ),
                    "maximum_delta_ratio": (
                        self.config.output_max_prior_delta_ratio
                    ),
                    "transform_condition_number": condition_number,
                    "maximum_transform_condition_number": (
                        self.config.output_max_transform_condition_number
                    ),
                    "transform_min_singular_value": float(
                        singular_values[-1].item()
                    ),
                    "minimum_transform_singular_value": (
                        self.config.output_min_transform_singular_value
                    ),
                    "output_subspace_relative_rmse": (
                        output_subspace_relative_rmse
                    ),
                    "body_latent_relative_rmse": (
                        body_latent_relative_rmse
                    ),
                    (
                        "tied_input_output_parameter_required"
                        if output_is_tied
                        else "untied_output_parameter_only"
                    ): True,
                    (
                        "source_weights_and_sealed_cache_only"
                        if output_is_tied
                        else "input_embedding_bytes_unchanged"
                    ): True,
                    "source_weights_and_sealed_cache_only": True,
                    "parent_model_forward_calls": 0,
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    (
                        "teacher_forced_terminal_acceptance_is_"
                        "necessary_not_sufficient"
                    ): True,
                    (
                        "external_whole_output_behavior_gate_required_"
                        "for_promotion"
                    ): True,
                }
            ),
        )

    def install_output_interface(
        self,
        scope: FoldScope,
        compiled: CompiledFoldStage,
        audit: ClosedLoopAudit,
    ) -> InstallReceipt:
        del audit
        torch = _torch()
        payload = compiled.payload
        if (
            scope.kind != "output_interface"
            or not isinstance(payload, _OutputInterfacePayload)
            or self._output_payload is not payload
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface artifact/state binding is invalid."
            )
        child_input, parameter = self._child_vocabulary_parameters()
        output_is_tied = child_input is parameter
        self._require_heterogeneous_input_unchanged()
        if _tensor_sha256(parameter) != payload.base_weight_sha256:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output vocabulary bytes changed after output compilation."
            )
        state_before = self.child_state_fingerprint()
        topology_before = self.child_topology_fingerprint()
        parameter_id = id(parameter)
        gradient = parameter.grad
        gradient_id = None if gradient is None else id(gradient)
        if payload.identity_skip:
            if (
                payload.candidate_weight_sha256
                != payload.base_weight_sha256
                or payload.materialized_delta_ratio != 0.0
                or not torch.equal(
                    payload.candidate_weight,
                    payload.base_weight,
                )
                or _tensor_sha256(parameter)
                != payload.base_weight_sha256
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Identity output-interface artifact is not byte-exact."
                )
            _installed_input, installed = self._child_vocabulary_parameters()
            self._require_heterogeneous_input_unchanged()
            if (
                installed is not parameter
                or id(installed) != parameter_id
                or (
                    None
                    if installed.grad is None
                    else id(installed.grad)
                )
                != gradient_id
                or _tensor_sha256(installed)
                != payload.base_weight_sha256
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Identity output-interface skip changed parameter state."
                )
            metadata = {
                "parameter_id_preserved": True,
                "gradient_identity_preserved": True,
                "candidate_weight_sha256": (
                    payload.candidate_weight_sha256
                ),
                "output_interface_mode": "identity-skip",
                "identity_skip": True,
                "parameter_write_calls": 0,
            }
            if output_is_tied:
                metadata.update(
                    {
                        "same_tied_parameter_object_preserved": True,
                        "input_embedding_and_output_head_share_parameter": True,
                        "tied_parameter_bytes_unchanged": True,
                    }
                )
            else:
                metadata.update(
                    {
                        "same_tied_parameter_object_preserved": False,
                        "input_embedding_and_output_head_share_parameter": False,
                        "untied_output_parameter_only": True,
                        "input_embedding_bytes_unchanged": True,
                        "output_parameter_bytes_unchanged": True,
                    }
                )
            return self._receipt(
                scope=scope,
                stage=CompileStage.OUTPUT_INTERFACE,
                artifact_sha256=compiled.artifact_sha256,
                state_before=state_before,
                topology_before=topology_before,
                metadata=metadata,
            )
        try:
            with torch.no_grad():
                parameter.copy_(
                    payload.candidate_weight.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
            _installed_input, installed = self._child_vocabulary_parameters()
            self._require_heterogeneous_input_unchanged()
            if (
                installed is not parameter
                or id(installed) != parameter_id
                or (
                    None
                    if installed.grad is None
                    else id(installed.grad)
                )
                != gradient_id
                or _tensor_sha256(installed)
                != payload.candidate_weight_sha256
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Output installation changed identity or bytes."
                )
        except BaseException:
            with torch.no_grad():
                parameter.copy_(
                    payload.base_weight.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
            self._require_heterogeneous_input_unchanged()
            raise
        metadata = {
            "parameter_id_preserved": True,
            "gradient_identity_preserved": True,
            "candidate_weight_sha256": payload.candidate_weight_sha256,
            "global_transform_materialized_into_existing_parameter": True,
            "output_interface_mode": "fit-transform",
            "identity_skip": False,
        }
        if output_is_tied:
            metadata.update(
                {
                    "same_tied_parameter_object_preserved": True,
                    "input_embedding_and_output_head_share_parameter": True,
                }
            )
        else:
            metadata.update(
                {
                    "same_tied_parameter_object_preserved": False,
                    "input_embedding_and_output_head_share_parameter": False,
                    "untied_output_parameter_only": True,
                    "input_embedding_bytes_unchanged": True,
                }
            )
        return self._receipt(
            scope=scope,
            stage=CompileStage.OUTPUT_INTERFACE,
            artifact_sha256=compiled.artifact_sha256,
            state_before=state_before,
            topology_before=topology_before,
            metadata=metadata,
        )

    def validate_output_interface(
        self,
        scope: FoldScope,
        validation_records: Sequence[TrajectoryRecord],
        parent_endpoints: ParentEndpointBatch,
        baseline_capture: ChildTrajectoryCapture,
        candidate_capture: ChildTrajectoryCapture,
        audit: ClosedLoopAudit,
    ) -> ValidationDecision:
        del audit
        payload = self._output_payload
        if (
            scope.kind != "output_interface"
            or payload is None
            or not isinstance(
                parent_endpoints.payload, _EndpointPayload
            )
            or baseline_capture.point
            is not CapturePoint.PRE_OUTPUT_INTERFACE
            or candidate_capture.point
            is not CapturePoint.POST_OUTPUT_INTERFACE
        ):
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output-interface validation inputs are invalid."
            )
        _child_input, installed = self._child_vocabulary_parameters()
        self._require_heterogeneous_input_unchanged()
        if _tensor_sha256(installed) != payload.candidate_weight_sha256:
            raise Lfm2DynamicClosedLoopAdapterError(
                "Output validation is not running on the installed candidate."
            )
        baseline = self._capture_payload(baseline_capture)
        candidate = self._capture_payload(candidate_capture)
        indices = self._batch_indices(validation_records)
        baseline_metrics = self._output_metrics(
            baseline.paired, indices
        )
        offline_metrics = self._output_metrics(
            baseline.paired,
            indices,
            materialized_weight=payload.candidate_weight,
        )
        candidate_metrics = self._output_metrics(
            candidate.paired, indices
        )
        feedback = self._offline_live_output_comparison(
            baseline.paired,
            candidate.paired,
            indices,
            payload.candidate_weight,
        )
        for key in (
            "rows",
            "vocabulary_size",
            "known_sequence_token_rows",
            "eos_token_id_available",
        ):
            if not (
                baseline_metrics[key]
                == offline_metrics[key]
                == candidate_metrics[key]
            ):
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Output validation metric coverage changed."
                )
        tolerance = self.config.output_validation_tolerance
        kl_improvement = float(
            baseline_metrics["full_vocabulary_parent_kl"]
        ) - float(candidate_metrics["full_vocabulary_parent_kl"])
        known_sequence_rows = int(
            candidate_metrics["known_sequence_token_rows"]
        )
        known_sequence_nll_improvement = float(
            baseline_metrics["known_sequence_token_nll"]
        ) - float(candidate_metrics["known_sequence_token_nll"])
        eos_available = bool(candidate_metrics["eos_token_id_available"])
        eos_margin_rmse_regression = float(
            candidate_metrics["eos_margin_rmse"]
        ) - float(baseline_metrics["eos_margin_rmse"])
        failed_gates = []
        if (
            kl_improvement + tolerance
            < self.config.minimum_output_kl_improvement
        ):
            failed_gates.append("full_vocabulary_parent_kl")
        if known_sequence_rows > 0 and (
            known_sequence_nll_improvement + tolerance
            < self.config.minimum_output_known_sequence_nll_improvement
        ):
            failed_gates.append("known_sequence_token_nll")
        if eos_available and (
            eos_margin_rmse_regression
            > self.config.maximum_output_eos_margin_rmse_regression
            + tolerance
        ):
            failed_gates.append("eos_margin_rmse")
        identity_replay_exact = (
            baseline_metrics == candidate_metrics
        )
        if payload.identity_skip:
            if not identity_replay_exact:
                raise Lfm2DynamicClosedLoopAdapterError(
                    "Identity output-interface skip changed live output "
                    "behavior."
                )
            failed_gates = []
            accepted = True
        else:
            accepted = not failed_gates
        metrics: dict[str, JsonScalar] = {
            "full_vocabulary_kl_improvement": kl_improvement,
            "known_sequence_nll_improvement": (
                known_sequence_nll_improvement
            ),
            "eos_margin_rmse_regression": eos_margin_rmse_regression,
            "minimum_output_kl_improvement": (
                self.config.minimum_output_kl_improvement
            ),
            "minimum_output_known_sequence_nll_improvement": (
                self.config.minimum_output_known_sequence_nll_improvement
            ),
            "maximum_output_eos_margin_rmse_regression": (
                self.config.maximum_output_eos_margin_rmse_regression
            ),
            "output_validation_tolerance": tolerance,
            "failed_gate_count": len(failed_gates),
            "failed_gates": (
                "none" if not failed_gates else ",".join(failed_gates)
            ),
            "output_interface_mode": (
                "identity-skip"
                if payload.identity_skip
                else "fit-transform"
            ),
            "identity_skip": payload.identity_skip,
            "identity_replay_exact": identity_replay_exact,
            "tied_parameter_bytes_unchanged": payload.identity_skip,
            "pre_install_heldout_baseline": True,
            "post_install_full_live_child_replay": True,
            "tied_embedding_feedback_included": True,
            "same_tied_parameter_object_preserved": True,
            "full_vocabulary_metrics": True,
            "kl_known_sequence_nll_and_eos_are_terminal_acceptance_gates": True,
            "parent_distribution_cross_entropy_is_redundant_with_kl": True,
            "parent_winner_top1_and_margin_are_diagnostics": True,
            "exact_token_match_is_not_promotion_authority": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "teacher_forced_terminal_acceptance_is_necessary_not_sufficient": (
                True
            ),
            "external_whole_output_behavior_gate_required_for_promotion": True,
            "promotion_authority": (
                "external_whole_output_semantic_behavior_gate"
            ),
        }
        if self._child_heterogeneous_layout is not None:
            metrics.pop("tied_parameter_bytes_unchanged")
            metrics.pop("tied_embedding_feedback_included")
            metrics.pop("same_tied_parameter_object_preserved")
            metrics.update(
                {
                    "output_parameter_bytes_unchanged": payload.identity_skip,
                    "input_embedding_bytes_unchanged": True,
                    "input_embedding_replay_included": True,
                    "untied_output_parameter_only": True,
                    "input_embedding_and_output_head_share_parameter": False,
                }
            )
        for prefix, values in (
            ("baseline", baseline_metrics),
            ("offline_candidate", offline_metrics),
            ("live_candidate", candidate_metrics),
            ("offline_vs_live", feedback),
        ):
            for key, value in values.items():
                metrics[f"{prefix}_{key}"] = value
        if payload.identity_skip:
            reason = (
                "explicit identity output-interface skip preserved output bytes "
                if self._child_heterogeneous_layout is not None
                else "explicit identity output-interface skip preserved tied bytes "
            ) + (
                "and live output behavior exactly; external whole-output "
                "behavior evaluation is still required"
            )
        else:
            reason = (
                "held-out full-vocabulary live-child output interface accepted; "
                "external whole-output behavior evaluation is still required"
                if accepted
                else (
                    "held-out full-vocabulary live-child output interface "
                    "rejected gates: "
                    + ",".join(failed_gates)
                )
            )
        return ValidationDecision(
            scope=scope,
            accepted=accepted,
            validation_record_keys=_record_keys(validation_records),
            reason=reason,
            metrics=_scalar_metadata(metrics),
        )


__all__ = [
    "DenseResidualCoordinateMap",
    "LFM2_ADAPTER_SCHEMA",
    "LFM2_TRAJECTORY_PATH",
    "Lfm2DynamicClosedLoopAdapter",
    "Lfm2DynamicClosedLoopAdapterConfig",
    "Lfm2DynamicClosedLoopAdapterError",
]
