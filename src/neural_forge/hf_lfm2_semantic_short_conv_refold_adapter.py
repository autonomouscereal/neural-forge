"""Closed-form long-horizon semantic refolding for one LFM2 short convolution.

The existing CPCF compiler remains the structural candidate generator: it
folds every parent B/C/x channel into the smaller dense child producer and FIR.
This adapter adds the causal protocol that the compiler deliberately does not
own.  Evidence is captured from the actual child after all earlier layer
installs, trajectories are split four ways, and a safer incumbent-producer
output-writer solve competes with the complete structural candidate.

Only semantic span boundaries are parent-exact endpoints.  Expanded target
subtoken rows may warm the causal FIR state, but they are never relabelled as
parent-exact observations.  FIT alone produces candidates; FIT-CHECK and
SELECTION select a frozen candidate; VALIDATION can only accept or reject it.

No model is executed here.  The implementation uses deterministic tensor
algebra under inference/no-grad contexts and performs no backward pass,
optimizer step, training step, pruning operation, or quantization operation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CompiledCpcfShortConvFold,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvGatePolicy,
    compile_lfm2_cpcf_short_conv_fold,
)


LFM2_SEMANTIC_SHORT_CONV_REFOLD_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-refold-v1"
)
LFM2_SEMANTIC_SHORT_CONV_REPLAY_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-replay-evidence-v1"
)
_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-short-conv-installation-v1"
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
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


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


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
        return {
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
            "full-source-structural",
            "incumbent-producer-output-writer",
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
    if dtypes != {torch.bfloat16}:
        raise Lfm2SemanticShortConvRefoldError(
            "Target short-conv Parameters must be dense BF16 tensors."
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


def _compile_writer_candidate(
    fit_cases: Sequence[_BalancedCase],
    incumbent_state: Mapping[str, Any],
) -> tuple[_Candidate | None, Mapping[str, Any]]:
    torch = _torch()
    contexts, targets, weights = _writer_design(fit_cases, incumbent_state)
    fit_context_sha = canonical_sha256(
        {
            "case_sha256": [case.case_sha256 for case in fit_cases],
            "contexts_sha256": _tensor_sha256(contexts),
            "targets_sha256": _tensor_sha256(targets),
            "weights_sha256": _tensor_sha256(weights),
        }
    )
    has_bias = "out_proj.bias" in incumbent_state
    design = contexts
    if has_bias:
        design = torch.cat(
            (design, torch.ones((design.shape[0], 1), dtype=torch.float64)),
            dim=1,
        )
    incumbent_coefficients = incumbent_state["out_proj.weight"].to(torch.float64).T
    if has_bias:
        incumbent_coefficients = torch.cat(
            (
                incumbent_coefficients,
                incumbent_state["out_proj.bias"].to(torch.float64).reshape(1, -1),
            ),
            dim=0,
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    baseline = _endpoint_metrics(fit_cases, incumbent_state)
    frontier = []
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        coefficients = (incumbent + alpha * delta).to(device="cpu")
        state = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in incumbent_state.items()
        }
        state["out_proj.weight"] = (
            coefficients[: contexts.shape[1], :]
            .T.to(dtype=incumbent_state["out_proj.weight"].dtype)
            .contiguous()
        )
        if has_bias:
            state["out_proj.bias"] = (
                coefficients[-1, :]
                .to(dtype=incumbent_state["out_proj.bias"].dtype)
                .contiguous()
            )
        if all(
            torch.equal(state[name], incumbent_state[name]) for name in incumbent_state
        ):
            continue
        metrics = _endpoint_metrics(fit_cases, state)
        if (
            metrics.endpoint_relative_rmse < baseline.endpoint_relative_rmse - 1e-8
            and _nonregressing(metrics, baseline, tolerance=1e-8)
        ):
            frontier.append((metrics.endpoint_relative_rmse, alpha, state, metrics))
    base_report = {
        "solver": "closed_form_trajectory_weighted_residual_ridge_bfloat16_frontier",
        "ridge": ridge,
        "effective_ridge": effective_ridge,
        "fit_context_sha256": fit_context_sha,
        "baseline_fit_metrics": baseline.as_dict(),
        "fit_only": True,
        "incumbent_in_proj_fixed": True,
        "incumbent_conv_fixed": True,
        "selection_used_for_solve": False,
        "validation_used_for_solve": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    if not frontier:
        return None, MappingProxyType(
            {
                **base_report,
                "status": "rejected-no-bfloat16-fit-improvement",
                "selected_alpha": None,
            }
        )
    _score, alpha, state, metrics = min(
        frontier, key=lambda value: (value[0], value[1])
    )
    report = {
        **base_report,
        "status": "fit-proposal",
        "selected_alpha": alpha,
        "candidate_fit_metrics": metrics.as_dict(),
    }
    return (
        _Candidate(
            name="incumbent-producer-output-writer",
            state=state,
            provenance=report,
        ),
        MappingProxyType(report),
    )


def _candidate_report(
    candidate: _Candidate,
    *,
    banks: Mapping[str, Sequence[_BalancedCase]],
    baseline: Mapping[str, Lfm2SemanticShortConvEndpointMetrics],
    tolerance: float,
) -> Mapping[str, Any]:
    metrics = {
        name: _endpoint_metrics(banks[name], candidate.state)
        for name in ("fit", "fit_check", "selection")
    }
    checks = {
        name: _nonregressing(metrics[name], baseline[name], tolerance=tolerance)
        for name in metrics
    }
    fit_improved = (
        metrics["fit"].endpoint_relative_rmse
        < baseline["fit"].endpoint_relative_rmse - tolerance
    )
    body = {
        "name": candidate.name,
        "candidate_state_sha256": candidate.state_sha256,
        "provenance": dict(candidate.provenance),
        "fit_improved": fit_improved,
        "partition_metrics": {name: metrics[name].as_dict() for name in metrics},
        "partition_nonregression": checks,
        "eligible_before_validation": fit_improved and all(checks.values()),
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
) -> _Candidate:
    eligible = [
        candidate
        for candidate in candidates
        if reports[candidate.name]["eligible_before_validation"] is True
    ]
    if not eligible:
        raise Lfm2SemanticShortConvRefoldError(
            "Neither structural nor incumbent-producer candidate passed "
            "FIT/FIT-CHECK/SELECTION nonregression."
        )
    by_name = {candidate.name: candidate for candidate in eligible}
    structural = by_name.get("full-source-structural")
    writer = by_name.get("incumbent-producer-output-writer")
    if structural is not None and writer is not None:
        structural_selection = Lfm2SemanticShortConvEndpointMetrics(
            **reports[structural.name]["partition_metrics"]["selection"]
        )
        writer_selection = Lfm2SemanticShortConvEndpointMetrics(
            **reports[writer.name]["partition_metrics"]["selection"]
        )
        if not _nonregressing(
            writer_selection,
            structural_selection,
            tolerance=1e-8,
        ):
            return structural
        return writer
    simplicity = {
        "incumbent-producer-output-writer": 0,
        "full-source-structural": 1,
    }

    def score(candidate: _Candidate) -> tuple[float, float, float, float, int, str]:
        metrics = reports[candidate.name]["partition_metrics"]["selection"]
        return (
            float(metrics["endpoint_relative_rmse"]),
            float(metrics["endpoint_maximum_absolute_error"]),
            -float(metrics["endpoint_cosine_similarity_mean"]),
            -float(metrics["endpoint_cosine_similarity_worst"]),
            simplicity[candidate.name],
            candidate.state_sha256,
        )

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
    snapshots = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in parameters.items()
    }
    parameter_ids = {name: id(value) for name, value in parameters.items()}
    gradient_ids = {
        name: None if value.grad is None else id(value.grad)
        for name, value in parameters.items()
    }
    gradient_hashes = {
        name: None if value.grad is None else _tensor_sha256(value.grad)
        for name, value in parameters.items()
    }
    prefix = f"model.layers.{layer_index}.conv."
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                candidate = state[name.removeprefix(prefix)]
                if (
                    tuple(candidate.shape) != tuple(parameter.shape)
                    or candidate.dtype != parameter.dtype
                ):
                    raise Lfm2SemanticShortConvRefoldError(
                        f"Candidate changed the topology or dtype of {name}."
                    )
                parameter.copy_(candidate.to(device=parameter.device))
        if (
            _parameter_count(target_model) != count
            or any(id(parameters[name]) != parameter_ids[name] for name in parameters)
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
                "Short-conv installation changed topology, identities, or gradients."
            )
        changed = tuple(
            name
            for name, parameter in parameters.items()
            if not torch.equal(parameter.detach().cpu(), snapshots[name])
        )
        after_model = fingerprint_lfm2_parent(target_model).identity_sha256
        if not changed or after_model == before_model:
            raise Lfm2SemanticShortConvRefoldError(
                "Frozen short-conv candidate installed no correction."
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
) -> Mapping[str, Any]:
    """Compile, gate, freeze, and atomically install one short-conv operator."""

    cache = _digest(cache_key, "cache key")
    bridge = _digest(bridge_artifact_sha256, "bridge artifact SHA-256")
    source_identity = _digest(
        source_model_identity_sha256,
        "source model identity SHA-256",
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
    current_child_state = fingerprint_lfm2_parent(target_model).identity_sha256
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
    parameters = _parameter_bundle(target_model, layer_index)
    incumbent_state = _suffix_state(parameters, layer_index)
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
    elif source_kernel.ndim == 2:
        source_hidden = int(source_kernel.shape[0])
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
    structural = _compile_structural_candidate(
        source_fir_weight=source_kernel,
        source_fir_bias=source_bias,
        incumbent_state=incumbent_state,
        fit=partitions["fit"],
        fit_check=partitions["fit_check"],
    )
    writer, writer_fit_report = _compile_writer_candidate(
        banks["fit"],
        incumbent_state,
    )
    candidates = [structural, *([] if writer is None else [writer])]
    tolerance = 1e-8
    reports = {
        candidate.name: _candidate_report(
            candidate,
            banks=banks,
            baseline=baseline,
            tolerance=tolerance,
        )
        for candidate in candidates
    }
    chosen = _select_candidate(candidates, reports)
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
    validation = _endpoint_metrics(banks["validation"], frozen_state)
    if (
        not _nonregressing(
            validation,
            baseline_validation,
            tolerance=tolerance,
        )
        or canonical_sha256(
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
            "Frozen short-conv candidate failed isolated VALIDATION."
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
        "installed": True,
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
        "changed_state_keys": list(changed),
        "changed_state_keys_local_to_operator": all(
            name.startswith(f"model.layers.{layer_index}.conv.") for name in changed
        ),
        "parameter_element_count": parameter_count,
        "parameter_count_changed": False,
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
        "selected_candidate": chosen.name,
        "selected_candidate_state_sha256": frozen_sha,
        "selected_validation_metrics": validation.as_dict(),
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "selection_used_for_candidate_choice": True,
        "validation_used_for_candidate_choice": False,
        "complete_parent_bcx_history_consumed": True,
        "parent_bcx_exact_only_at_semantic_span_boundaries": True,
        "inserted_target_bcx_rows_are_owner_expanded_proxies": True,
        "complete_source_channel_coverage_candidate_compiled": True,
        "full_source_structural_candidate_competed": True,
        "incumbent_producer_output_writer_candidate_competed": writer is not None,
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
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


__all__ = [
    "LFM2_SEMANTIC_SHORT_CONV_REFOLD_SCHEMA",
    "LFM2_SEMANTIC_SHORT_CONV_REPLAY_EVIDENCE_SCHEMA",
    "Lfm2SemanticShortConvEndpointMetrics",
    "Lfm2SemanticShortConvRefoldError",
    "Lfm2SemanticShortConvReplayEvidence",
    "compile_install_lfm2_semantic_short_conv_postfold",
]
