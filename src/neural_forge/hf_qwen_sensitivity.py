"""Bounded held-out sensitivity evidence for Qwen3 model surgery.

The collector in this module performs inference-only interventions.  It does
not fit weights, update assignments, run backward, or perform recovery.  A
neuron intervention zeros one real post-SwiGLU channel.  An attention
intervention zeros the matching Q, K, V, and O slices for one GQA group in the
same forward transaction.

The reported controllability and observability vectors are deliberately named
``zero_ablation_directional_finite_difference`` sketches.  They are finite
differences between an untouched held-out forward and a zero-ablation forward;
they are not presented as infinitesimal JVPs or VJPs.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neural_forge.adaptive_surgery_graph import AnchorSpec
from neural_forge.hf_neuron_codebook import NeuronCodebookLayerView
from neural_forge.hf_qwen_adaptive_surgery import Qwen3AdaptiveSurgeryFeature


SENSITIVITY_SCHEMA = "neural-forge-qwen3-sensitivity-v1"
FINITE_DIFFERENCE_METHOD = "zero_ablation_directional_finite_difference"
_ATTENTION_ROLES = ("q", "k", "v", "o")
_MEBIBYTE = 1024 * 1024


class Qwen3SensitivityError(ValueError):
    """Base error for invalid or unsafe sensitivity collection."""


class Qwen3SensitivityShapeError(Qwen3SensitivityError):
    """A model, batch, candidate, or hook result has an invalid shape."""


class Qwen3SensitivityLimitError(Qwen3SensitivityError):
    """A hard row, device, candidate, or collector-memory cap was exceeded."""


class Qwen3SensitivitySplitError(Qwen3SensitivityError):
    """Calibration and held-out gate rows are not strictly disjoint."""


class Qwen3SensitivityStateError(Qwen3SensitivityError):
    """A forward intervention did not leave the model hook state intact."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3SensitivityShapeError(f"{name} must be a positive integer.")
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3SensitivityShapeError(f"{name} must be a positive integer.") from exc
    if normalized <= 0:
        raise Qwen3SensitivityShapeError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3SensitivityShapeError(f"{name} must be a non-negative integer.")
    try:
        normalized = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3SensitivityShapeError(
            f"{name} must be a non-negative integer."
        ) from exc
    if normalized < 0:
        raise Qwen3SensitivityShapeError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityLimits:
    """Hard bounds for collector-owned work and materialized forward rows."""

    max_calibration_rows: int = 32
    max_gate_rows: int = 256
    max_sequence_length: int = 4096
    max_score_rows: int = 128
    max_neuron_candidates: int = 512
    max_attention_candidates: int = 256
    candidate_chunk_size: int = 8
    max_candidate_forward_rows: int = 256
    max_model_devices: int = 1
    max_collector_bytes: int = 512 * _MEBIBYTE
    max_fingerprint_chunk_bytes: int = 8 * _MEBIBYTE
    max_sketch_work_bytes: int = 8 * _MEBIBYTE
    max_sketch_dim: int = 64
    allowed_device_types: tuple[str, ...] = ("cpu", "cuda")

    def __post_init__(self) -> None:
        integer_fields = (
            "max_calibration_rows",
            "max_gate_rows",
            "max_sequence_length",
            "max_score_rows",
            "max_neuron_candidates",
            "max_attention_candidates",
            "candidate_chunk_size",
            "max_candidate_forward_rows",
            "max_model_devices",
            "max_collector_bytes",
            "max_fingerprint_chunk_bytes",
            "max_sketch_work_bytes",
            "max_sketch_dim",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        device_types = tuple(
            str(item).strip().lower() for item in self.allowed_device_types
        )
        if not device_types or any(not item for item in device_types):
            raise Qwen3SensitivityShapeError(
                "allowed_device_types must contain at least one device type."
            )
        object.__setattr__(self, "allowed_device_types", device_types)
        if self.max_fingerprint_chunk_bytes > self.max_collector_bytes:
            raise Qwen3SensitivityLimitError(
                "max_fingerprint_chunk_bytes cannot exceed max_collector_bytes."
            )
        if self.max_sketch_work_bytes > self.max_collector_bytes:
            raise Qwen3SensitivityLimitError(
                "max_sketch_work_bytes cannot exceed max_collector_bytes."
            )


@dataclass(frozen=True, order=True, slots=True)
class Qwen3NeuronRef:
    """One logical post-SwiGLU neuron execution."""

    layer_index: int
    neuron_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "neuron_index",
            _nonnegative_int(self.neuron_index, "neuron_index"),
        )


@dataclass(frozen=True, order=True, slots=True)
class Qwen3AttentionGroupRef:
    """One logical coupled Q/K/V/O GQA-group execution."""

    layer_index: int
    group_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "group_index",
            _nonnegative_int(self.group_index, "group_index"),
        )


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityFingerprints:
    """Exact model and positioned-token fingerprints for cache isolation."""

    model: str
    calibration: str
    gate: str
    measurement: str
    cache: str
    calibration_row_fingerprints: tuple[str, ...]
    gate_row_fingerprints: tuple[str, ...]
    model_hash_method: str = "sha256_exact_streamed_state_dict"
    row_hash_method: str = "sha256_active_token_ids_and_position_ids"


@dataclass(frozen=True, slots=True)
class Qwen3NeuronSensitivity:
    """Measured held-out damage from zeroing one post-SwiGLU channel."""

    node: Qwen3NeuronRef
    activation_energy: float
    residual_mse: float
    residual_relative_l2: float
    logit_mse: float
    logit_relative_l2: float
    logit_kl: float
    top1_disagreement: float
    damage_score: float
    finite_difference_controllability: tuple[float, ...]
    finite_difference_observability: tuple[float, ...]
    derivative_method: str = FINITE_DIFFERENCE_METHOD


@dataclass(frozen=True, slots=True)
class Qwen3AttentionGroupSensitivity:
    """Measured damage from one atomic coupled Q/K/V/O group ablation."""

    node: Qwen3AttentionGroupRef
    role_activation_energy: tuple[float, float, float, float]
    residual_mse: float
    residual_relative_l2: float
    logit_mse: float
    logit_relative_l2: float
    logit_kl: float
    top1_disagreement: float
    damage_score: float
    finite_difference_controllability: tuple[float, ...]
    finite_difference_observability: tuple[float, ...]
    atomic_projection_roles: tuple[str, str, str, str] = _ATTENTION_ROLES
    role_ablation_call_counts: tuple[int, int, int, int] = (1, 1, 1, 1)
    derivative_method: str = FINITE_DIFFERENCE_METHOD


@dataclass(frozen=True, slots=True)
class Qwen3ExactAnchorMasks:
    """Immutable exact-anchor masks over the complete logical Qwen layout."""

    neuron_mask: tuple[tuple[bool, ...], ...]
    attention_group_mask: tuple[tuple[bool, ...], ...]
    explicit_neurons: frozenset[Qwen3NeuronRef]
    measured_high_damage_neurons: frozenset[Qwen3NeuronRef]
    explicit_attention_groups: frozenset[Qwen3AttentionGroupRef]
    measured_high_damage_attention_groups: frozenset[Qwen3AttentionGroupRef]

    @property
    def exact_neurons(self) -> frozenset[Qwen3NeuronRef]:
        return self.explicit_neurons | self.measured_high_damage_neurons

    @property
    def exact_attention_groups(self) -> frozenset[Qwen3AttentionGroupRef]:
        return (
            self.explicit_attention_groups | self.measured_high_damage_attention_groups
        )

    def is_neuron_exact(self, layer_index: int, neuron_index: int) -> bool:
        return self.neuron_mask[layer_index][neuron_index]

    def is_attention_group_exact(self, layer_index: int, group_index: int) -> bool:
        return self.attention_group_mask[layer_index][group_index]


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityReport:
    """Audit record proving bounded inference-only collection."""

    schema: str
    logical_layers: int
    intermediate_width: int
    hidden_size: int
    key_value_groups: int
    calibration_rows: int
    gate_rows: int
    sequence_length: int
    scored_positions_per_row: int
    neuron_candidates: int
    attention_group_candidates: int
    baseline_forward_passes: int
    candidate_forward_passes: int
    maximum_candidate_chunk: int
    tracked_peak_collector_bytes: int
    collector_byte_limit: int
    model_devices: tuple[str, ...]
    mlp_capture_methods: tuple[str, ...]
    derivative_method: str
    exact_autodiff_used: bool
    finite_difference_reason: str
    backward_calls: int
    optimizer_steps: int
    training_steps: int
    recovery_steps: int
    model_weights_mutated: bool
    hooks_restored: bool


@dataclass(frozen=True, slots=True)
class Qwen3SensitivityEvidence:
    """Immutable evidence bundle consumed by selective surgery planning."""

    fingerprints: Qwen3SensitivityFingerprints
    neuron_activation_energy: tuple[tuple[float, ...], ...]
    attention_role_activation_energy: tuple[
        tuple[tuple[float, float, float, float], ...], ...
    ]
    neurons: tuple[Qwen3NeuronSensitivity, ...]
    attention_groups: tuple[Qwen3AttentionGroupSensitivity, ...]
    anchors: Qwen3ExactAnchorMasks
    report: Qwen3SensitivityReport

    def to_adaptive_surgery_features(
        self,
    ) -> tuple[Qwen3AdaptiveSurgeryFeature, ...]:
        """Translate measured neuron evidence into the existing Qwen bridge API."""

        result: list[Qwen3AdaptiveSurgeryFeature] = []
        for item in self.neurons:
            layer_context = tuple(
                value
                for group in self.attention_role_activation_energy[
                    item.node.layer_index
                ]
                for value in group
            )
            result.append(
                Qwen3AdaptiveSurgeryFeature(
                    item.node.layer_index,
                    item.node.neuron_index,
                    activation=[
                        item.activation_energy,
                        item.residual_relative_l2,
                        item.logit_relative_l2,
                        item.logit_kl,
                        item.top1_disagreement,
                    ],
                    jacobian=item.finite_difference_controllability,
                    dual=item.finite_difference_observability,
                    domain_features=layer_context or None,
                    importance=max(item.damage_score, 1e-12),
                    anchor=(
                        AnchorSpec.pinned()
                        if self.anchors.is_neuron_exact(
                            item.node.layer_index,
                            item.node.neuron_index,
                        )
                        else None
                    ),
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class _Batch:
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    row_fingerprints: tuple[str, ...]
    content_fingerprints: tuple[str, ...]
    fingerprint: str

    @property
    def rows(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass(frozen=True, slots=True)
class _Layout:
    layers: tuple[nn.Module, ...]
    mlp_modes: tuple[Literal["dense", "codebook"], ...]
    intermediate_width: int
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    vocabulary_size: int
    input_device: torch.device
    activation_element_bytes: int
    model_devices: tuple[str, ...]

    @property
    def query_rows_per_group(self) -> int:
        return (self.query_heads // self.key_value_heads) * self.head_dim


@dataclass(slots=True)
class _Baseline:
    logits: Tensor
    residuals: dict[int, Tensor]
    neuron_energy: tuple[tuple[float, ...], ...]
    attention_energy: tuple[tuple[tuple[float, float, float, float], ...], ...]
    capture_methods: tuple[str, ...]


class _HookScope:
    def __init__(self) -> None:
        self._handles: list[Any] = []

    def add(self, handle: Any) -> None:
        self._handles.append(handle)

    def __enter__(self) -> _HookScope:
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash_tensor(digest: Any, value: Tensor, chunk_bytes: int) -> None:
    if value.device.type == "meta":
        raise Qwen3SensitivityShapeError("Model state cannot contain meta tensors.")
    if value.layout != torch.strided:
        raise Qwen3SensitivityShapeError(
            "Exact model fingerprints require strided state tensors."
        )
    flat = value.detach().reshape(-1)
    element_size = max(1, int(value.element_size()))
    chunk_elements = max(1, chunk_bytes // element_size)
    for start in range(0, int(flat.numel()), chunk_elements):
        cpu = flat[start : start + chunk_elements].to("cpu").contiguous()
        byte_view = cpu.view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))


def fingerprint_qwen3_model(
    model: nn.Module,
    *,
    max_chunk_bytes: int = 8 * _MEBIBYTE,
) -> str:
    """Stream an exact SHA-256 over Qwen config and ordinary tensor state."""

    chunk_bytes = _positive_int(max_chunk_bytes, "max_chunk_bytes")
    digest = hashlib.sha256()
    digest.update(SENSITIVITY_SCHEMA.encode("ascii"))
    config = getattr(model, "config", None)
    if config is None:
        raise Qwen3SensitivityShapeError(
            "Exact Qwen fingerprints require model.config."
        )
    config_payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    digest.update(_canonical_json(config_payload))
    state = model.state_dict()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise Qwen3SensitivityShapeError(
                f"Model state {name!r} is not an ordinary tensor."
            )
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(tuple(int(item) for item in value.shape)))
        _hash_tensor(digest, value, chunk_bytes)
    return digest.hexdigest()


def _tensor_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to("cpu").contiguous()
    return memoryview(cpu.view(torch.uint8).numpy()).tobytes()


def _batch(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    position_ids: Tensor | None,
    *,
    split_name: str,
    max_rows: int,
    limits: Qwen3SensitivityLimits,
) -> _Batch:
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise Qwen3SensitivityShapeError(
            f"{split_name}_input_ids must be a rank-2 tensor."
        )
    if input_ids.dtype == torch.bool or input_ids.is_floating_point():
        raise Qwen3SensitivityShapeError(
            f"{split_name}_input_ids must use an integer dtype."
        )
    ids = input_ids.detach().to(device="cpu", dtype=torch.long).clone()
    rows, sequence_length = map(int, ids.shape)
    if rows <= 0 or sequence_length <= 0:
        raise Qwen3SensitivityShapeError(f"{split_name} cannot be empty.")
    if rows > max_rows:
        raise Qwen3SensitivityLimitError(
            f"{split_name} has {rows} rows, exceeding its hard cap {max_rows}."
        )
    if sequence_length > limits.max_sequence_length:
        raise Qwen3SensitivityLimitError(
            f"{split_name} sequence length {sequence_length} exceeds "
            f"{limits.max_sequence_length}."
        )
    if bool((ids < 0).any()):
        raise Qwen3SensitivityShapeError(
            f"{split_name}_input_ids cannot contain negative token IDs."
        )
    if attention_mask is None:
        mask = torch.ones_like(ids, dtype=torch.bool)
    else:
        if not isinstance(attention_mask, Tensor) or tuple(attention_mask.shape) != (
            rows,
            sequence_length,
        ):
            raise Qwen3SensitivityShapeError(
                f"{split_name}_attention_mask must match input_ids."
            )
        mask = attention_mask.detach().to(device="cpu", dtype=torch.bool).clone()
    if bool((mask.sum(dim=1) == 0).any()):
        raise Qwen3SensitivityShapeError(
            f"Every {split_name} row must contain at least one active token."
        )
    if position_ids is None:
        positions = mask.to(torch.long).cumsum(dim=1).sub(1).clamp_min(0)
    else:
        if not isinstance(position_ids, Tensor) or tuple(position_ids.shape) != (
            rows,
            sequence_length,
        ):
            raise Qwen3SensitivityShapeError(
                f"{split_name}_position_ids must match input_ids."
            )
        if position_ids.dtype == torch.bool or position_ids.is_floating_point():
            raise Qwen3SensitivityShapeError(
                f"{split_name}_position_ids must use an integer dtype."
            )
        positions = position_ids.detach().to(device="cpu", dtype=torch.long).clone()
    if bool((positions[mask] < 0).any()):
        raise Qwen3SensitivityShapeError(
            f"Active {split_name} position IDs cannot be negative."
        )

    row_hashes: list[str] = []
    content_hashes: list[str] = []
    for row in range(rows):
        active_ids = ids[row][mask[row]]
        active_positions = positions[row][mask[row]]
        content = hashlib.sha256()
        content.update(_tensor_bytes(active_ids))
        content_hashes.append(content.hexdigest())
        positioned = hashlib.sha256()
        positioned.update(_tensor_bytes(active_ids))
        positioned.update(_tensor_bytes(active_positions))
        row_hashes.append(positioned.hexdigest())
    fingerprint = _sha256_json(
        {
            "split": split_name,
            "shape": [rows, sequence_length],
            "rows": row_hashes,
            "mask": hashlib.sha256(_tensor_bytes(mask)).hexdigest(),
        }
    )
    return _Batch(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        row_fingerprints=tuple(row_hashes),
        content_fingerprints=tuple(content_hashes),
        fingerprint=fingerprint,
    )


def _strictly_disjoint(calibration: _Batch, gate: _Batch) -> None:
    positioned_overlap = set(calibration.row_fingerprints) & set(gate.row_fingerprints)
    content_overlap = set(calibration.content_fingerprints) & set(
        gate.content_fingerprints
    )
    if positioned_overlap or content_overlap:
        raise Qwen3SensitivitySplitError(
            "Calibration and gate rows must be disjoint by active token content "
            "and by token/position pairs."
        )


def _layout(model: nn.Module, limits: Qwen3SensitivityLimits) -> _Layout:
    if not isinstance(model, nn.Module):
        raise Qwen3SensitivityShapeError("model must be a torch module.")
    if model.training:
        raise Qwen3SensitivityStateError(
            "Sensitivity collection requires model.eval(); training mode is refused."
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise Qwen3SensitivityShapeError("model.config.model_type must be 'qwen3'.")
    backbone = getattr(model, "model", None)
    layers_value = getattr(backbone, "layers", None)
    if not isinstance(layers_value, (nn.ModuleList, list, tuple)) or not layers_value:
        raise Qwen3SensitivityShapeError(
            "Qwen3 model.model.layers must be a non-empty layer sequence."
        )
    layers = tuple(layers_value)
    hidden_size = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    query_heads = _positive_int(
        getattr(config, "num_attention_heads", None),
        "num_attention_heads",
    )
    key_value_heads = _positive_int(
        getattr(config, "num_key_value_heads", None),
        "num_key_value_heads",
    )
    head_dim = _positive_int(getattr(config, "head_dim", None), "head_dim")
    if query_heads % key_value_heads:
        raise Qwen3SensitivityShapeError(
            "num_attention_heads must be divisible by num_key_value_heads."
        )

    modes: list[Literal["dense", "codebook"]] = []
    widths: set[int] = set()
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, NeuronCodebookLayerView):
            modes.append("codebook")
            widths.add(int(mlp.intermediate_features))
        else:
            down = getattr(mlp, "down_proj", None)
            if not isinstance(down, nn.Module) or not hasattr(down, "in_features"):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} lacks a dense or codebook Qwen MLP."
                )
            if not callable(getattr(mlp, "act_fn", None)):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} dense MLP lacks act_fn."
                )
            modes.append("dense")
            widths.add(int(down.in_features))
        attention = getattr(layer, "self_attn", None)
        for role in _ATTENTION_ROLES:
            projection = getattr(attention, f"{role}_proj", None)
            if not isinstance(projection, nn.Module):
                raise Qwen3SensitivityShapeError(
                    f"Layer {layer_index} lacks self_attn.{role}_proj."
                )
    if len(widths) != 1:
        raise Qwen3SensitivityShapeError(
            "All Qwen layers must share one intermediate width."
        )
    intermediate = widths.pop()
    output = model.get_output_embeddings()
    vocabulary_size = int(getattr(output, "out_features", 0))
    if vocabulary_size <= 0:
        vocabulary_size = _positive_int(
            getattr(config, "vocab_size", None),
            "vocab_size",
        )
    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, Tensor) or weight.device.type == "meta":
        raise Qwen3SensitivityShapeError(
            "Qwen input embedding must expose a materialized weight tensor."
        )
    devices = {
        str(value.device)
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
        if value.device.type != "meta"
    }
    if not devices:
        devices = {str(weight.device)}
    device_types = {torch.device(item).type for item in devices}
    unsupported = device_types - set(limits.allowed_device_types)
    if unsupported:
        raise Qwen3SensitivityLimitError(
            f"Model uses forbidden device types {sorted(unsupported)!r}."
        )
    if len(devices) > limits.max_model_devices:
        raise Qwen3SensitivityLimitError(
            f"Model spans {len(devices)} devices, exceeding hard cap "
            f"{limits.max_model_devices}."
        )
    return _Layout(
        layers=layers,
        mlp_modes=tuple(modes),
        intermediate_width=intermediate,
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        vocabulary_size=vocabulary_size,
        input_device=weight.device,
        activation_element_bytes=max(1, int(weight.element_size())),
        model_devices=tuple(sorted(devices)),
    )


def _ref(value: Any, kind: Literal["neuron", "attention"]) -> Any:
    expected = Qwen3NeuronRef if kind == "neuron" else Qwen3AttentionGroupRef
    if isinstance(value, expected):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return expected(value[0], value[1])
    label = "(layer, neuron)" if kind == "neuron" else "(layer, GQA group)"
    raise Qwen3SensitivityShapeError(f"Candidate must be {label}.")


def _refs(
    values: Sequence[Any] | None,
    *,
    kind: Literal["neuron", "attention"],
    layout: _Layout,
    maximum: int,
) -> tuple[Any, ...]:
    if values is None:
        total = len(layout.layers) * (
            layout.intermediate_width if kind == "neuron" else layout.key_value_heads
        )
        if total > maximum:
            raise Qwen3SensitivityLimitError(
                f"Implicit all-{kind} collection would create {total} candidates; "
                f"provide an explicit bounded candidate list (cap {maximum})."
            )
        width = (
            layout.intermediate_width if kind == "neuron" else layout.key_value_heads
        )
        values = [
            (layer, index)
            for layer in range(len(layout.layers))
            for index in range(width)
        ]
    result = tuple(sorted({_ref(value, kind) for value in values}))
    if len(result) > maximum:
        raise Qwen3SensitivityLimitError(
            f"{kind} candidate count {len(result)} exceeds hard cap {maximum}."
        )
    width = layout.intermediate_width if kind == "neuron" else layout.key_value_heads
    for item in result:
        index = item.neuron_index if kind == "neuron" else item.group_index
        if item.layer_index >= len(layout.layers) or index >= width:
            raise Qwen3SensitivityShapeError(f"Out-of-range {kind} candidate {item!r}.")
    return result


def _score_indices(
    values: Sequence[int] | None,
    batch: _Batch,
    limits: Qwen3SensitivityLimits,
) -> tuple[int, ...]:
    requested = (-1,) if values is None else tuple(values)
    if not requested:
        raise Qwen3SensitivityShapeError("score_positions cannot be empty.")
    normalized: list[int] = []
    for value in requested:
        if isinstance(value, bool):
            raise Qwen3SensitivityShapeError("score positions must be integers.")
        try:
            index = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Qwen3SensitivityShapeError(
                "score positions must be integers."
            ) from exc
        if index < 0:
            index += batch.sequence_length
        if not 0 <= index < batch.sequence_length:
            raise Qwen3SensitivityShapeError(
                f"Score position {value!r} is outside the calibration sequence."
            )
        normalized.append(index)
    result = tuple(sorted(set(normalized)))
    if batch.rows * len(result) > limits.max_score_rows:
        raise Qwen3SensitivityLimitError(
            f"Scored logit rows {batch.rows * len(result)} exceed hard cap "
            f"{limits.max_score_rows}."
        )
    if not bool(batch.attention_mask[:, list(result)].all()):
        raise Qwen3SensitivityShapeError(
            "Every score position must be active in every calibration row."
        )
    return result


def _hook_inventory(
    layout: _Layout,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    modules: dict[int, nn.Module] = {}
    for layer in layout.layers:
        modules[id(layer)] = layer
        mlp = layer.mlp
        modules[id(mlp)] = mlp
        down = getattr(mlp, "down_proj", None)
        if isinstance(down, nn.Module):
            modules[id(down)] = down
        attention = layer.self_attn
        for role in _ATTENTION_ROLES:
            projection = getattr(attention, f"{role}_proj")
            modules[id(projection)] = projection
    return tuple(
        sorted(
            (
                module_id,
                tuple(module._forward_pre_hooks.keys()),
                tuple(module._forward_hooks.keys()),
            )
            for module_id, module in modules.items()
        )
    )


def _activation(inputs: Tensor, name: str) -> Tensor:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "linear": "identity",
        "swish": "silu",
        "gelu_new": "gelu_tanh",
        "gelu_pytorch_tanh": "gelu_tanh",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "identity":
        return inputs
    if normalized == "silu":
        return F.silu(inputs)
    if normalized == "gelu":
        return F.gelu(inputs, approximate="none")
    if normalized == "gelu_tanh":
        return F.gelu(inputs, approximate="tanh")
    if normalized == "relu":
        return F.relu(inputs)
    raise Qwen3SensitivityShapeError(f"Unsupported codebook activation {name!r}.")


def _masked_channel_energy(values: Tensor, mask: Tensor) -> tuple[float, ...]:
    if values.ndim != 3 or tuple(values.shape[:2]) != tuple(mask.shape):
        raise Qwen3SensitivityShapeError(
            "Captured post-SwiGLU activations do not match calibration rows."
        )
    active = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denominator = max(1, int(mask.sum().item()))
    energy = values.detach().float().square().mul(active.float()).sum((0, 1))
    energy = energy.div(float(denominator)).to("cpu")
    return tuple(float(item) for item in energy.tolist())


def _masked_group_energy(
    values: Tensor,
    mask: Tensor,
    groups: int,
) -> tuple[float, ...]:
    if values.ndim != 3 or tuple(values.shape[:2]) != tuple(mask.shape):
        raise Qwen3SensitivityShapeError(
            "Captured attention activations do not match calibration rows."
        )
    width = int(values.shape[-1])
    if width % groups:
        raise Qwen3SensitivityShapeError(
            "Attention projection width is not divisible by GQA groups."
        )
    per_group = width // groups
    shaped = values.detach().reshape(*values.shape[:2], groups, per_group)
    active = (
        mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1).unsqueeze(-1)
    )
    denominator = max(1, int(mask.sum().item()) * per_group)
    energy = shaped.float().square().mul(active.float()).sum((0, 1, 3))
    energy = energy.div(float(denominator)).to("cpu")
    return tuple(float(item) for item in energy.tolist())


def _forward(
    model: nn.Module,
    batch: _Batch,
    score_positions: tuple[int, ...],
    *,
    repeats: int = 1,
) -> Tensor:
    device = next(model.get_input_embeddings().parameters()).device
    ids = batch.input_ids.to(device).repeat((repeats, 1))
    mask = batch.attention_mask.to(device).repeat((repeats, 1))
    positions = batch.position_ids.to(device).repeat((repeats, 1))
    score = torch.tensor(score_positions, device=device, dtype=torch.long)
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        logits_to_keep=score,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise Qwen3SensitivityShapeError("Qwen forward did not return rank-3 logits.")
    expected_rows = batch.rows * repeats
    if tuple(logits.shape[:2]) != (expected_rows, len(score_positions)):
        raise Qwen3SensitivityShapeError(
            "Qwen logits do not match requested held-out score rows."
        )
    return logits


def _baseline(
    model: nn.Module,
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    residual_layers: frozenset[int],
) -> _Baseline:
    mask = batch.attention_mask
    residuals: dict[int, Tensor] = {}
    neuron_energy: list[tuple[float, ...] | None] = [None] * len(layout.layers)
    capture_methods: list[str] = [""] * len(layout.layers)
    role_energy: list[dict[str, tuple[float, ...]]] = [{} for _ in layout.layers]
    with _HookScope() as hooks:
        for layer_index, layer in enumerate(layout.layers):

            def residual_hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, Tensor) or output.ndim != 3:
                    raise Qwen3SensitivityShapeError(
                        "Qwen decoder layer did not return rank-3 residual state."
                    )
                residuals[index] = output.detach().to("cpu", dtype=torch.float32)

            if layer_index in residual_layers:
                hooks.add(layer.register_forward_hook(residual_hook))
            mlp = layer.mlp
            if layout.mlp_modes[layer_index] == "dense":

                def dense_energy_hook(
                    _module: nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    index: int = layer_index,
                ) -> None:
                    if not inputs or not isinstance(inputs[0], Tensor):
                        raise Qwen3SensitivityShapeError(
                            "Qwen down_proj did not receive post-SwiGLU activations."
                        )
                    neuron_energy[index] = _masked_channel_energy(inputs[0], mask)

                hooks.add(mlp.down_proj.register_forward_pre_hook(dense_energy_hook))
                capture_methods[layer_index] = "real_down_proj_pre_hook"
            else:

                def codebook_energy_hook(
                    module: nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    index: int = layer_index,
                ) -> None:
                    if not isinstance(module, NeuronCodebookLayerView):
                        raise Qwen3SensitivityShapeError(
                            "Expected a NeuronCodebookLayerView."
                        )
                    if not inputs or not isinstance(inputs[0], Tensor):
                        raise Qwen3SensitivityShapeError(
                            "Codebook MLP did not receive hidden states."
                        )
                    bundle = module.registry.materialize_layer(index)
                    hidden = _activation(
                        F.linear(inputs[0], bundle.gate_weight, bundle.gate_bias),
                        module.registry.activation,
                    ) * F.linear(inputs[0], bundle.up_weight, bundle.up_bias)
                    neuron_energy[index] = _masked_channel_energy(hidden, mask)

                hooks.add(mlp.register_forward_pre_hook(codebook_energy_hook))
                capture_methods[layer_index] = (
                    "exact_codebook_recompute_from_real_mlp_input"
                )

            attention = layer.self_attn
            for role in ("q", "k", "v"):
                projection = getattr(attention, f"{role}_proj")

                def attention_output_hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    index: int = layer_index,
                    role_name: str = role,
                ) -> None:
                    if not isinstance(output, Tensor):
                        raise Qwen3SensitivityShapeError(
                            f"Qwen {role_name}_proj did not return a tensor."
                        )
                    role_energy[index][role_name] = _masked_group_energy(
                        output,
                        mask,
                        layout.key_value_heads,
                    )

                hooks.add(projection.register_forward_hook(attention_output_hook))

            def attention_o_hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                index: int = layer_index,
            ) -> None:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Qwen o_proj did not receive attention contexts."
                    )
                role_energy[index]["o"] = _masked_group_energy(
                    inputs[0],
                    mask,
                    layout.key_value_heads,
                )

            hooks.add(attention.o_proj.register_forward_pre_hook(attention_o_hook))
        logits = _forward(model, batch, score_positions)

    if set(residuals) != set(residual_layers):
        raise Qwen3SensitivityStateError(
            "At least one requested decoder residual hook did not execute."
        )
    if any(value is None for value in neuron_energy):
        raise Qwen3SensitivityStateError(
            "At least one post-SwiGLU activation hook did not execute."
        )
    attention_result: list[tuple[tuple[float, float, float, float], ...]] = []
    for layer_index, values in enumerate(role_energy):
        if set(values) != set(_ATTENTION_ROLES):
            raise Qwen3SensitivityStateError(
                f"Layer {layer_index} did not expose all Q/K/V/O activations."
            )
        attention_result.append(
            tuple(
                tuple(values[role][group] for role in _ATTENTION_ROLES)
                for group in range(layout.key_value_heads)
            )
        )
    return _Baseline(
        logits=logits.detach().to("cpu", dtype=torch.float32),
        residuals=residuals,
        neuron_energy=tuple(value for value in neuron_energy if value is not None),
        attention_energy=tuple(attention_result),
        capture_methods=tuple(capture_methods),
    )


def _candidate_residual_hooks(
    hooks: _HookScope,
    layout: _Layout,
    chunk: Sequence[Any],
    rows: int,
    result: dict[int, Tensor],
) -> None:
    by_layer: dict[int, list[int]] = {}
    for local_index, candidate in enumerate(chunk):
        by_layer.setdefault(candidate.layer_index, []).append(local_index)
    for layer_index, local_indices in by_layer.items():

        def capture(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            indices: tuple[int, ...] = tuple(local_indices),
        ) -> None:
            if not isinstance(output, Tensor) or output.ndim != 3:
                raise Qwen3SensitivityShapeError(
                    "Qwen decoder layer did not return rank-3 residual state."
                )
            for local in indices:
                start = local * rows
                result[local] = (
                    output[start : start + rows].detach().to("cpu", dtype=torch.float32)
                )

        hooks.add(layout.layers[layer_index].register_forward_hook(capture))


def _install_neuron_ablation_hooks(
    hooks: _HookScope,
    layout: _Layout,
    chunk: Sequence[Qwen3NeuronRef],
    rows: int,
    call_counts: dict[int, int],
) -> None:
    by_layer: dict[int, list[tuple[int, Qwen3NeuronRef]]] = {}
    for local, candidate in enumerate(chunk):
        by_layer.setdefault(candidate.layer_index, []).append((local, candidate))
    for layer_index, entries in by_layer.items():
        mlp = layout.layers[layer_index].mlp
        if layout.mlp_modes[layer_index] == "dense":

            def dense_ablation(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                selected: tuple[tuple[int, Qwen3NeuronRef], ...] = tuple(entries),
            ) -> tuple[Any, ...]:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Qwen down_proj did not receive a tensor."
                    )
                changed = inputs[0].clone()
                for local, candidate in selected:
                    start = local * rows
                    changed[
                        start : start + rows,
                        :,
                        candidate.neuron_index,
                    ] = 0
                    call_counts[local] = call_counts.get(local, 0) + 1
                return (changed, *inputs[1:])

            hooks.add(mlp.down_proj.register_forward_pre_hook(dense_ablation))
        else:

            def codebook_ablation(
                module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                selected: tuple[tuple[int, Qwen3NeuronRef], ...] = tuple(entries),
            ) -> Tensor:
                if not isinstance(module, NeuronCodebookLayerView):
                    raise Qwen3SensitivityShapeError(
                        "Expected a NeuronCodebookLayerView."
                    )
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Codebook MLP did not receive a tensor."
                    )
                if not isinstance(output, Tensor):
                    raise Qwen3SensitivityShapeError(
                        "Codebook MLP did not return a tensor."
                    )
                bundle = module.registry.materialize_layer(index)
                changed = output.clone()
                for local, candidate in selected:
                    start = local * rows
                    hidden = inputs[0][start : start + rows]
                    neuron = candidate.neuron_index
                    gate_bias = (
                        None
                        if bundle.gate_bias is None
                        else bundle.gate_bias[neuron : neuron + 1]
                    )
                    up_bias = (
                        None
                        if bundle.up_bias is None
                        else bundle.up_bias[neuron : neuron + 1]
                    )
                    post = _activation(
                        F.linear(
                            hidden,
                            bundle.gate_weight[neuron : neuron + 1],
                            gate_bias,
                        ),
                        module.registry.activation,
                    ) * F.linear(
                        hidden,
                        bundle.up_weight[neuron : neuron + 1],
                        up_bias,
                    )
                    contribution = F.linear(
                        post,
                        bundle.down_weight[:, neuron : neuron + 1],
                    )
                    changed[start : start + rows] -= contribution
                    call_counts[local] = call_counts.get(local, 0) + 1
                return changed

            hooks.add(mlp.register_forward_hook(codebook_ablation))


def _install_attention_ablation_hooks(
    hooks: _HookScope,
    layout: _Layout,
    chunk: Sequence[Qwen3AttentionGroupRef],
    rows: int,
    call_counts: dict[int, dict[str, int]],
) -> None:
    by_layer: dict[int, list[tuple[int, Qwen3AttentionGroupRef]]] = {}
    for local, candidate in enumerate(chunk):
        by_layer.setdefault(candidate.layer_index, []).append((local, candidate))
    for layer_index, entries in by_layer.items():
        attention = layout.layers[layer_index].self_attn
        for role in ("q", "k", "v"):
            projection = getattr(attention, f"{role}_proj")
            group_width = (
                layout.query_rows_per_group if role == "q" else layout.head_dim
            )

            def output_ablation(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                role_name: str = role,
                width: int = group_width,
                selected: tuple[tuple[int, Qwen3AttentionGroupRef], ...] = tuple(
                    entries
                ),
            ) -> Tensor:
                if not isinstance(output, Tensor):
                    raise Qwen3SensitivityShapeError(
                        f"Qwen {role_name}_proj did not return a tensor."
                    )
                changed = output.clone()
                for local, candidate in selected:
                    row_start = local * rows
                    column_start = candidate.group_index * width
                    changed[
                        row_start : row_start + rows,
                        :,
                        column_start : column_start + width,
                    ] = 0
                    counts = call_counts.setdefault(local, {})
                    counts[role_name] = counts.get(role_name, 0) + 1
                return changed

            hooks.add(projection.register_forward_hook(output_ablation))

        def o_ablation(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            selected: tuple[tuple[int, Qwen3AttentionGroupRef], ...] = tuple(entries),
        ) -> tuple[Any, ...]:
            if not inputs or not isinstance(inputs[0], Tensor):
                raise Qwen3SensitivityShapeError(
                    "Qwen o_proj did not receive a tensor."
                )
            changed = inputs[0].clone()
            width = layout.query_rows_per_group
            for local, candidate in selected:
                row_start = local * rows
                column_start = candidate.group_index * width
                changed[
                    row_start : row_start + rows,
                    :,
                    column_start : column_start + width,
                ] = 0
                counts = call_counts.setdefault(local, {})
                counts["o"] = counts.get("o", 0) + 1
            return (changed, *inputs[1:])

        hooks.add(attention.o_proj.register_forward_pre_hook(o_ablation))


def _count_sketch(
    values: Tensor,
    dimension: int,
    seed: int,
    max_work_bytes: int,
) -> tuple[float, ...]:
    flat = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not int(flat.numel()):
        return tuple(0.0 for _ in range(dimension))
    result = torch.zeros(dimension, dtype=torch.float32)
    result_bytes = dimension * result.element_size()
    available = max_work_bytes - result_bytes
    if available < 32:
        raise Qwen3SensitivityLimitError(
            "max_sketch_work_bytes is too small for the requested sketch."
        )
    chunk_elements = max(1, available // 32)
    seed_value = int(seed % 2_147_483_647)
    for start in range(0, int(flat.numel()), chunk_elements):
        end = min(int(flat.numel()), start + chunk_elements)
        indices = torch.arange(start, end, dtype=torch.int64)
        buckets = torch.remainder(
            indices * 1_103_515_245 + seed_value,
            dimension,
        )
        signs = torch.where(
            torch.remainder(
                indices * 2_654_435_761 + seed_value * 17 + 13,
                2,
            )
            == 0,
            1.0,
            -1.0,
        )
        result.scatter_add_(0, buckets, flat[start:end] * signs)
    result.div_(math.sqrt(float(flat.numel())))
    return tuple(float(item) for item in result.tolist())


def _damage(
    baseline_residual: Tensor,
    candidate_residual: Tensor,
    baseline_logits: Tensor,
    candidate_logits: Tensor,
    mask: Tensor,
    *,
    sketch_dim: int,
    residual_seed: int,
    logit_seed: int,
    sketch_work_bytes: int,
) -> dict[str, Any]:
    residual_delta = candidate_residual.float() - baseline_residual.float()
    active = mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(residual_delta)
    active_delta = residual_delta[active]
    active_baseline = baseline_residual.float()[active]
    residual_square = float(active_delta.square().sum().item())
    residual_base_square = float(active_baseline.square().sum().item())
    residual_mse = residual_square / max(1, int(active_delta.numel()))
    residual_relative = math.sqrt(
        residual_square / max(residual_base_square, torch.finfo(torch.float32).tiny)
    )

    logit_delta = candidate_logits.float() - baseline_logits.float()
    logit_square = float(logit_delta.square().sum().item())
    logit_base_square = float(baseline_logits.float().square().sum().item())
    logit_mse = logit_square / max(1, int(logit_delta.numel()))
    logit_relative = math.sqrt(
        logit_square / max(logit_base_square, torch.finfo(torch.float32).tiny)
    )
    baseline_log_probability = F.log_softmax(baseline_logits.float(), dim=-1)
    candidate_log_probability = F.log_softmax(candidate_logits.float(), dim=-1)
    baseline_probability = baseline_log_probability.exp()
    kl = float(
        (baseline_probability * (baseline_log_probability - candidate_log_probability))
        .sum(dim=-1)
        .mean()
        .item()
    )
    kl = max(0.0, kl)
    top1 = float(
        (baseline_logits.argmax(dim=-1) != candidate_logits.argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    score = residual_relative + logit_relative + math.sqrt(kl) + top1
    return {
        "residual_mse": residual_mse,
        "residual_relative_l2": residual_relative,
        "logit_mse": logit_mse,
        "logit_relative_l2": logit_relative,
        "logit_kl": kl,
        "top1_disagreement": top1,
        "damage_score": score,
        "finite_difference_controllability": _count_sketch(
            residual_delta * mask.unsqueeze(-1),
            sketch_dim,
            residual_seed,
            sketch_work_bytes,
        ),
        "finite_difference_observability": _count_sketch(
            logit_delta,
            sketch_dim,
            logit_seed,
            sketch_work_bytes,
        ),
    }


def _chunk_capacity(
    *,
    kind: Literal["neuron", "attention"],
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    limits: Qwen3SensitivityLimits,
    persistent_bytes: int,
) -> tuple[int, int]:
    rows = batch.rows
    sequence = batch.sequence_length
    score_rows = len(score_positions)
    dtype_bytes = layout.activation_element_bytes
    residual = rows * sequence * layout.hidden_size * 4
    logits = rows * score_rows * layout.vocabulary_size * (4 + dtype_bytes)
    expanded_inputs = rows * sequence * (8 + 1 + 8)
    if kind == "neuron":
        intervention_width = (
            layout.intermediate_width
            if "dense" in layout.mlp_modes
            else layout.hidden_size + 2
        )
    else:
        query_width = layout.query_heads * layout.head_dim
        key_value_width = layout.key_value_heads * layout.head_dim
        intervention_width = 2 * query_width + 2 * key_value_width
    intervention = rows * sequence * intervention_width * dtype_bytes
    per_candidate = residual + logits + expanded_inputs + intervention
    analysis_scratch = (
        rows * sequence * layout.hidden_size * 4 * 3
        + rows * score_rows * layout.vocabulary_size * 4 * 5
        + limits.max_sketch_work_bytes
    )
    capacity = min(
        limits.candidate_chunk_size,
        limits.max_candidate_forward_rows // rows,
    )
    while capacity > 0:
        peak = persistent_bytes + capacity * per_candidate + analysis_scratch
        if peak <= limits.max_collector_bytes:
            return capacity, peak
        capacity -= 1
    raise Qwen3SensitivityLimitError(
        f"A single {kind} candidate would exceed max_collector_bytes="
        f"{limits.max_collector_bytes}."
    )


def _measure_neurons(
    model: nn.Module,
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    candidates: tuple[Qwen3NeuronRef, ...],
    baseline: _Baseline,
    *,
    chunk_size: int,
    sketch_dim: int,
    residual_seed: int,
    logit_seed: int,
    sketch_work_bytes: int,
) -> tuple[tuple[Qwen3NeuronSensitivity, ...], int]:
    result: list[Qwen3NeuronSensitivity] = []
    forward_passes = 0
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        residuals: dict[int, Tensor] = {}
        calls: dict[int, int] = {}
        with _HookScope() as hooks:
            _install_neuron_ablation_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                calls,
            )
            _candidate_residual_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                residuals,
            )
            logits = (
                _forward(
                    model,
                    batch,
                    score_positions,
                    repeats=len(chunk),
                )
                .detach()
                .to("cpu", dtype=torch.float32)
            )
        forward_passes += 1
        shaped_logits = logits.reshape(
            len(chunk),
            batch.rows,
            len(score_positions),
            layout.vocabulary_size,
        )
        for local, candidate in enumerate(chunk):
            if calls.get(local) != 1 or local not in residuals:
                raise Qwen3SensitivityStateError(
                    f"Neuron ablation {candidate!r} did not execute exactly once."
                )
            metrics = _damage(
                baseline.residuals[candidate.layer_index],
                residuals[local],
                baseline.logits,
                shaped_logits[local],
                batch.attention_mask,
                sketch_dim=sketch_dim,
                residual_seed=residual_seed,
                logit_seed=logit_seed,
                sketch_work_bytes=sketch_work_bytes,
            )
            result.append(
                Qwen3NeuronSensitivity(
                    node=candidate,
                    activation_energy=baseline.neuron_energy[candidate.layer_index][
                        candidate.neuron_index
                    ],
                    **metrics,
                )
            )
    return tuple(result), forward_passes


def _measure_attention(
    model: nn.Module,
    layout: _Layout,
    batch: _Batch,
    score_positions: tuple[int, ...],
    candidates: tuple[Qwen3AttentionGroupRef, ...],
    baseline: _Baseline,
    *,
    chunk_size: int,
    sketch_dim: int,
    residual_seed: int,
    logit_seed: int,
    sketch_work_bytes: int,
) -> tuple[tuple[Qwen3AttentionGroupSensitivity, ...], int]:
    result: list[Qwen3AttentionGroupSensitivity] = []
    forward_passes = 0
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        residuals: dict[int, Tensor] = {}
        calls: dict[int, dict[str, int]] = {}
        with _HookScope() as hooks:
            _install_attention_ablation_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                calls,
            )
            _candidate_residual_hooks(
                hooks,
                layout,
                chunk,
                batch.rows,
                residuals,
            )
            logits = (
                _forward(
                    model,
                    batch,
                    score_positions,
                    repeats=len(chunk),
                )
                .detach()
                .to("cpu", dtype=torch.float32)
            )
        forward_passes += 1
        shaped_logits = logits.reshape(
            len(chunk),
            batch.rows,
            len(score_positions),
            layout.vocabulary_size,
        )
        for local, candidate in enumerate(chunk):
            counts = calls.get(local, {})
            role_counts = tuple(counts.get(role, 0) for role in _ATTENTION_ROLES)
            if role_counts != (1, 1, 1, 1) or local not in residuals:
                raise Qwen3SensitivityStateError(
                    f"Atomic GQA ablation {candidate!r} did not cover Q/K/V/O once."
                )
            metrics = _damage(
                baseline.residuals[candidate.layer_index],
                residuals[local],
                baseline.logits,
                shaped_logits[local],
                batch.attention_mask,
                sketch_dim=sketch_dim,
                residual_seed=residual_seed,
                logit_seed=logit_seed,
                sketch_work_bytes=sketch_work_bytes,
            )
            result.append(
                Qwen3AttentionGroupSensitivity(
                    node=candidate,
                    role_activation_energy=baseline.attention_energy[
                        candidate.layer_index
                    ][candidate.group_index],
                    role_ablation_call_counts=role_counts,
                    **metrics,
                )
            )
    return tuple(result), forward_passes


def _highest_damage(items: Sequence[Any], count: int) -> frozenset[Any]:
    if count <= 0:
        return frozenset()
    ranked = sorted(
        items,
        key=lambda item: (
            -item.damage_score,
            item.node.layer_index,
            getattr(item.node, "neuron_index", getattr(item.node, "group_index", 0)),
        ),
    )
    return frozenset(item.node for item in ranked[:count])


def _anchors(
    layout: _Layout,
    neurons: tuple[Qwen3NeuronSensitivity, ...],
    attention: tuple[Qwen3AttentionGroupSensitivity, ...],
    *,
    explicit_neurons: tuple[Qwen3NeuronRef, ...],
    explicit_attention: tuple[Qwen3AttentionGroupRef, ...],
    neuron_top_k: int,
    attention_top_k: int,
) -> Qwen3ExactAnchorMasks:
    measured_neurons = _highest_damage(neurons, neuron_top_k)
    measured_attention = _highest_damage(attention, attention_top_k)
    exact_neurons = frozenset(explicit_neurons) | measured_neurons
    exact_attention = frozenset(explicit_attention) | measured_attention
    neuron_mask = tuple(
        tuple(
            Qwen3NeuronRef(layer, neuron) in exact_neurons
            for neuron in range(layout.intermediate_width)
        )
        for layer in range(len(layout.layers))
    )
    attention_mask = tuple(
        tuple(
            Qwen3AttentionGroupRef(layer, group) in exact_attention
            for group in range(layout.key_value_heads)
        )
        for layer in range(len(layout.layers))
    )
    return Qwen3ExactAnchorMasks(
        neuron_mask=neuron_mask,
        attention_group_mask=attention_mask,
        explicit_neurons=frozenset(explicit_neurons),
        measured_high_damage_neurons=measured_neurons,
        explicit_attention_groups=frozenset(explicit_attention),
        measured_high_damage_attention_groups=measured_attention,
    )


def collect_qwen3_sensitivity_evidence(
    model: nn.Module,
    calibration_input_ids: Tensor,
    gate_input_ids: Tensor,
    *,
    calibration_attention_mask: Tensor | None = None,
    gate_attention_mask: Tensor | None = None,
    calibration_position_ids: Tensor | None = None,
    gate_position_ids: Tensor | None = None,
    score_positions: Sequence[int] | None = None,
    neuron_candidates: Sequence[Qwen3NeuronRef | tuple[int, int]] | None = None,
    attention_group_candidates: Sequence[Qwen3AttentionGroupRef | tuple[int, int]]
    | None = None,
    explicit_neuron_anchors: Sequence[Qwen3NeuronRef | tuple[int, int]] = (),
    explicit_attention_group_anchors: Sequence[
        Qwen3AttentionGroupRef | tuple[int, int]
    ] = (),
    neuron_high_damage_anchor_count: int = 1,
    attention_high_damage_anchor_count: int = 1,
    sketch_dim: int = 16,
    limits: Qwen3SensitivityLimits | None = None,
) -> Qwen3SensitivityEvidence:
    """Collect deterministic real-forward evidence without training or recovery.

    ``gate_input_ids`` are fingerprinted and checked for strict disjointness but
    are intentionally not evaluated here.  They remain untouched for the
    external accept/reject gate used after a proposed surgery transaction.
    """

    active_limits = limits or Qwen3SensitivityLimits()
    if not isinstance(active_limits, Qwen3SensitivityLimits):
        raise Qwen3SensitivityShapeError(
            "limits must be a Qwen3SensitivityLimits instance."
        )
    dimension = _positive_int(sketch_dim, "sketch_dim")
    if dimension > active_limits.max_sketch_dim:
        raise Qwen3SensitivityLimitError(
            f"sketch_dim {dimension} exceeds hard cap {active_limits.max_sketch_dim}."
        )
    neuron_anchor_count = _nonnegative_int(
        neuron_high_damage_anchor_count,
        "neuron_high_damage_anchor_count",
    )
    attention_anchor_count = _nonnegative_int(
        attention_high_damage_anchor_count,
        "attention_high_damage_anchor_count",
    )
    layout = _layout(model, active_limits)
    calibration = _batch(
        calibration_input_ids,
        calibration_attention_mask,
        calibration_position_ids,
        split_name="calibration",
        max_rows=active_limits.max_calibration_rows,
        limits=active_limits,
    )
    gate = _batch(
        gate_input_ids,
        gate_attention_mask,
        gate_position_ids,
        split_name="gate",
        max_rows=active_limits.max_gate_rows,
        limits=active_limits,
    )
    _strictly_disjoint(calibration, gate)
    positions = _score_indices(score_positions, calibration, active_limits)
    neurons = _refs(
        neuron_candidates,
        kind="neuron",
        layout=layout,
        maximum=active_limits.max_neuron_candidates,
    )
    attention = _refs(
        attention_group_candidates,
        kind="attention",
        layout=layout,
        maximum=active_limits.max_attention_candidates,
    )
    explicit_neurons = _refs(
        explicit_neuron_anchors,
        kind="neuron",
        layout=layout,
        maximum=active_limits.max_neuron_candidates,
    )
    explicit_attention = _refs(
        explicit_attention_group_anchors,
        kind="attention",
        layout=layout,
        maximum=active_limits.max_attention_candidates,
    )
    residual_layers = frozenset(item.layer_index for item in (*neurons, *attention))

    input_bytes = calibration.rows * calibration.sequence_length * (
        8 + 1 + 8
    ) + gate.rows * gate.sequence_length * (8 + 1 + 8)
    baseline_bytes = (
        len(residual_layers)
        * calibration.rows
        * calibration.sequence_length
        * layout.hidden_size
        * 4
        + calibration.rows * len(positions) * layout.vocabulary_size * 4
        + len(layout.layers) * layout.intermediate_width * 8
        + len(layout.layers) * layout.key_value_heads * 4 * 8
    )
    persistent_bytes = input_bytes + baseline_bytes
    activation_rows = calibration.rows * calibration.sequence_length
    baseline_activation_scratch = (
        activation_rows
        * max(
            layout.intermediate_width * (5 if "codebook" in layout.mlp_modes else 3),
            2 * layout.query_heads * layout.head_dim,
        )
        * max(4, layout.activation_element_bytes)
    )
    baseline_logit_scratch = (
        calibration.rows
        * len(positions)
        * layout.vocabulary_size
        * (4 + layout.activation_element_bytes)
    )
    baseline_peak = persistent_bytes + max(
        baseline_activation_scratch,
        baseline_logit_scratch,
    )
    if baseline_peak > active_limits.max_collector_bytes:
        raise Qwen3SensitivityLimitError(
            f"Baseline collector peak {baseline_peak} exceeds "
            f"max_collector_bytes={active_limits.max_collector_bytes}."
        )
    if neurons:
        neuron_chunk, neuron_peak = _chunk_capacity(
            kind="neuron",
            layout=layout,
            batch=calibration,
            score_positions=positions,
            limits=active_limits,
            persistent_bytes=persistent_bytes,
        )
    else:
        neuron_chunk, neuron_peak = 1, baseline_peak
    if attention:
        attention_chunk, attention_peak = _chunk_capacity(
            kind="attention",
            layout=layout,
            batch=calibration,
            score_positions=positions,
            limits=active_limits,
            persistent_bytes=persistent_bytes,
        )
    else:
        attention_chunk, attention_peak = 1, baseline_peak

    model_fingerprint = fingerprint_qwen3_model(
        model,
        max_chunk_bytes=active_limits.max_fingerprint_chunk_bytes,
    )
    measurement_payload = {
        "schema": SENSITIVITY_SCHEMA,
        "model": model_fingerprint,
        "calibration": calibration.fingerprint,
        "gate": gate.fingerprint,
        "score_positions": positions,
        "neurons": [[item.layer_index, item.neuron_index] for item in neurons],
        "attention": [[item.layer_index, item.group_index] for item in attention],
        "sketch_dim": dimension,
        "neuron_chunk": neuron_chunk,
        "attention_chunk": attention_chunk,
    }
    measurement_fingerprint = _sha256_json(measurement_payload)
    cache_fingerprint = _sha256_json(
        {
            "measurement": measurement_fingerprint,
            "explicit_neuron_anchors": [
                [item.layer_index, item.neuron_index] for item in explicit_neurons
            ],
            "explicit_attention_anchors": [
                [item.layer_index, item.group_index] for item in explicit_attention
            ],
            "neuron_high_damage_anchor_count": neuron_anchor_count,
            "attention_high_damage_anchor_count": attention_anchor_count,
        }
    )
    fingerprints = Qwen3SensitivityFingerprints(
        model=model_fingerprint,
        calibration=calibration.fingerprint,
        gate=gate.fingerprint,
        measurement=measurement_fingerprint,
        cache=cache_fingerprint,
        calibration_row_fingerprints=calibration.row_fingerprints,
        gate_row_fingerprints=gate.row_fingerprints,
    )
    residual_seed = int(
        hashlib.sha256(f"{measurement_fingerprint}:residual".encode()).hexdigest()[:15],
        16,
    )
    logit_seed = int(
        hashlib.sha256(f"{measurement_fingerprint}:logit".encode()).hexdigest()[:15],
        16,
    )

    inventory = _hook_inventory(layout)
    with torch.inference_mode():
        baseline = _baseline(
            model,
            layout,
            calibration,
            positions,
            residual_layers,
        )
        neuron_evidence, neuron_forwards = _measure_neurons(
            model,
            layout,
            calibration,
            positions,
            neurons,
            baseline,
            chunk_size=neuron_chunk,
            sketch_dim=dimension,
            residual_seed=residual_seed,
            logit_seed=logit_seed,
            sketch_work_bytes=active_limits.max_sketch_work_bytes,
        )
        attention_evidence, attention_forwards = _measure_attention(
            model,
            layout,
            calibration,
            positions,
            attention,
            baseline,
            chunk_size=attention_chunk,
            sketch_dim=dimension,
            residual_seed=residual_seed,
            logit_seed=logit_seed,
            sketch_work_bytes=active_limits.max_sketch_work_bytes,
        )
    hooks_restored = _hook_inventory(layout) == inventory
    if not hooks_restored:
        raise Qwen3SensitivityStateError(
            "Sensitivity collection changed the model hook inventory."
        )
    anchor_masks = _anchors(
        layout,
        neuron_evidence,
        attention_evidence,
        explicit_neurons=explicit_neurons,
        explicit_attention=explicit_attention,
        neuron_top_k=neuron_anchor_count,
        attention_top_k=attention_anchor_count,
    )
    report = Qwen3SensitivityReport(
        schema=SENSITIVITY_SCHEMA,
        logical_layers=len(layout.layers),
        intermediate_width=layout.intermediate_width,
        hidden_size=layout.hidden_size,
        key_value_groups=layout.key_value_heads,
        calibration_rows=calibration.rows,
        gate_rows=gate.rows,
        sequence_length=calibration.sequence_length,
        scored_positions_per_row=len(positions),
        neuron_candidates=len(neurons),
        attention_group_candidates=len(attention),
        baseline_forward_passes=1,
        candidate_forward_passes=neuron_forwards + attention_forwards,
        maximum_candidate_chunk=max(
            min(neuron_chunk, len(neurons)) if neurons else 0,
            min(attention_chunk, len(attention)) if attention else 0,
        ),
        tracked_peak_collector_bytes=max(
            baseline_peak,
            neuron_peak if neurons else persistent_bytes,
            attention_peak if attention else persistent_bytes,
        ),
        collector_byte_limit=active_limits.max_collector_bytes,
        model_devices=layout.model_devices,
        mlp_capture_methods=baseline.capture_methods,
        derivative_method=FINITE_DIFFERENCE_METHOD,
        exact_autodiff_used=False,
        finite_difference_reason=(
            "bounded inference-only zero-ablation avoids retained autograd graphs "
            "and applies identically to dense and prepared codebook runtimes"
        ),
        backward_calls=0,
        optimizer_steps=0,
        training_steps=0,
        recovery_steps=0,
        model_weights_mutated=False,
        hooks_restored=hooks_restored,
    )
    return Qwen3SensitivityEvidence(
        fingerprints=fingerprints,
        neuron_activation_energy=baseline.neuron_energy,
        attention_role_activation_energy=baseline.attention_energy,
        neurons=neuron_evidence,
        attention_groups=attention_evidence,
        anchors=anchor_masks,
        report=report,
    )


__all__ = [
    "FINITE_DIFFERENCE_METHOD",
    "Qwen3AttentionGroupRef",
    "Qwen3AttentionGroupSensitivity",
    "Qwen3ExactAnchorMasks",
    "Qwen3NeuronRef",
    "Qwen3NeuronSensitivity",
    "Qwen3SensitivityError",
    "Qwen3SensitivityEvidence",
    "Qwen3SensitivityFingerprints",
    "Qwen3SensitivityLimitError",
    "Qwen3SensitivityLimits",
    "Qwen3SensitivityReport",
    "Qwen3SensitivityShapeError",
    "Qwen3SensitivitySplitError",
    "Qwen3SensitivityStateError",
    "collect_qwen3_sensitivity_evidence",
    "fingerprint_qwen3_model",
]
